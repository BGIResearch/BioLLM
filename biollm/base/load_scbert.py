#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: load_scbert.py
@time: 2024/3/11 15:02
"""
from biollm.base.load_llm import LoadLlm
from biollm.repo.scbert.performer_pytorch.performer_pytorch import PerformerLM
from biollm.repo.scgpt.tokenizer.gene_tokenizer import GeneVocab
from biollm.dataset.scbert_dataset import SCDataset, make_scbert_adata
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from scipy.sparse import issparse
import pickle as pkl
import torch
from tqdm import tqdm


class LoadScbert(LoadLlm):
    """
    LoadScbert is a class for loading and managing the SCBERT model.
    This class inherits from LoadLlm and provides functionalities such as loading pretrained models,
    generating gene and cell embeddings, and preparing data for training.

    Attributes:
        num_tokens (int): Number of gene express bins.
        max_seq_len (int): Maximum sequence length for input data.
        use_g2v (bool): Indicator to use gene-to-vector (g2v) embeddings.
        g2v_file (str): Path to the file containing g2v embeddings.
        vocab (GeneVocab): Vocabulary object for gene-to-index mapping.
        model (PerformerLM): Loaded SCBERT model.
        gene2idx (dict): Mapping from gene names to indices.
    """
    def __init__(self, args):
        """
        Initializes LoadScbert with specific model settings and loads necessary components.

        Args:
            args (Namespace): Arguments object with settings for model, device, and file paths.
        """
        super(LoadScbert, self).__init__(args)
        self.num_tokens = args.bin_num
        self.max_seq_len = args.max_seq_len
        self.use_g2v = args.use_g2v
        self.g2v_file = args.g2v_file
        self.vocab = self.load_vocab()
        self.model = self.load_model()
        self.gene2idx = self.get_gene2idx()
        self.init_model()
        self.model = self.model.to(self.args.device)

    def load_model(self):
        """
        Loads the SCBERT model architecture with Performer-based attention.

        Returns:
            PerformerLM: The initialized SCBERT model.
        """
        model = PerformerLM(
            num_tokens=self.num_tokens,
            dim=200,
            depth=6,
            max_seq_len=self.max_seq_len,
            heads=10,
            local_attn_heads=0,
            g2v_position_emb=self.use_g2v,
            g2v_file=self.g2v_file
        )
        return model

    def get_gene_embedding(self, gene_ids):
        """
        Generates embeddings for a set of gene IDs.

        Args:
            gene_ids (torch.Tensor): Tensor of gene IDs to embed.

        Returns:
            np.ndarray: Gene embeddings as a NumPy array.
        """
        self.logger.info('start to get gene embedding!')
        emb = self.model.pos_emb.emb
        self.logger.info('start to get gene embedding!')
        gene_emb = emb(gene_ids)
        print('gpu used: ', torch.cuda.memory_allocated())
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        return gene_emb.detach().cpu().numpy()

    def get_gene2idx(self):
        """
        Retrieves gene-to-index mapping from vocabulary.

        Returns:
            dict: Dictionary mapping gene names to indices.
        """
        return self.vocab.get_stoi()

    def load_vocab(self):
        """
        Loads vocabulary from a file, containing mappings of gene names to indices.

        Returns:
            GeneVocab: Vocabulary object for gene-to-index mapping.
        """
        vocab = GeneVocab.from_file(self.args.vocab_file)
        return vocab

    def freezon_model(self, keep_layers=[-2]):
        """
        Freezes the model layers except for specific normalization and selected layers,
        which can be fine-tuned.

        Args:
            keep_layers (list): List of layers to keep unfrozen for fine-tuning.
        """
        model_param_count = 0
        ft_param_count = 0
        for param in self.model.parameters():
            model_param_count += param.numel()
            param.requires_grad = False
        for param in self.model.norm.parameters():
            param.requires_grad = True
            ft_param_count += param.numel()
        for i in keep_layers:
            for name, param in self.model.performer.net.layers[i].named_parameters():
                param.requires_grad = True
                ft_param_count += param.numel()
        self.logger.info(f"Total pretrain-model Encoder Params {model_param_count}")
        self.logger.info(f"The pretrain_model Encoder Params for training in finetune after freezon: {ft_param_count}")

    def preprocess_adata(self, adata):
        """
        Preprocesses AnnData object for SCBERT, ensuring compatibility with model input requirements.

        Args:
            adata (AnnData): Single-cell data object.

        Returns:
            AnnData: Preprocessed AnnData object.
        """
        idx2gene = dict([(self.gene2idx[k], k) for k in self.gene2idx])
        ref_genes = [idx2gene[i] for i in range(len(idx2gene))]
        if 'do_preprocess' in adata.uns:
            self.logger.info('the adata was already preprocessed, pass the step!')
            return adata
        adata = make_scbert_adata(adata, ref_genes)
        adata.uns['do_preprocess'] = True
        return adata

    def load_dataset(self, adata, split_data=False, label_key=None, do_preprocess=True):
        """
        Prepares dataset for training or evaluation, with an option to split into training and validation.

        Args:
            adata (AnnData): Single-cell data to process.
            split_data (bool): Flag to split the dataset into train/validation sets.
            label_key (str): Column in `adata.obs` containing labels.
            do_preprocess (bool): Flag to preprocess `adata` if needed.

        Returns:
            Tuple[SCDataset, Optional[SCDataset]]: Training and validation datasets.
        """
        if do_preprocess:
            adata = self.preprocess_adata(adata)
        if label_key:
            self.label_dict, self.label = np.unique(np.array(adata.obs[label_key]), return_inverse=True)
        else:
            self.label = np.ones((adata.obs.shape[0]))
        data = adata.X.toarray() if issparse(adata.X) else adata.X
        if split_data:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2021)
            for index_train, index_val in sss.split(data, self.label):
                data_train, label_train = data[index_train], self.label[index_train]
                data_val, label_val = data[index_val], self.label[index_val]
                train_dataset = SCDataset(data_train, self.args.bin_num, label_train)
                test_dataset = SCDataset(data_val, self.args.bin_num, label_val)
                return train_dataset, test_dataset
        else:
            train_dataset = SCDataset(data, self.args.bin_num, self.label)
            return train_dataset, None

    def get_dataloader(self, dataset, drop_last=False, num_workers=0, random_sample=True):
        """
        Generates a DataLoader for batch processing during model training or evaluation.

        Args:
            dataset (SCDataset): The dataset to load.
            drop_last (bool): Flag to drop the last incomplete batch.
            num_workers (int): Number of worker threads for loading data.
            random_sample (bool): Flag to enable random sampling.

        Returns:
            DataLoader: DataLoader for the provided dataset.
        """
        if self.args.distributed:
            sampler = DistributedSampler(dataset, shuffle=random_sample)
        else:
            sampler = SequentialSampler(dataset) if not random_sample else RandomSampler(dataset)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=num_workers,
            sampler=sampler,
        )
        return data_loader

    def get_gene_expression_embedding(self, adata, pool='mean'):
        """
        Obtains gene expression embeddings by pooling the model’s cell-level embeddings.

        Args:
            adata (AnnData): Single-cell data.
            pool (str): Pooling method, either 'mean' or 'sum'.

        Returns:
            np.ndarray: Gene expression embeddings as a NumPy array.
        """
        self.logger.info("start to get gene expression!")
        dataset, _ = self.load_dataset(adata, split_data=False)
        data_loader = self.get_dataloader(dataset, random_sample=False)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            pool_emb = torch.zeros((len(self.gene2idx), self.args.embsize)).to(self.args.device)
            for index, (data, _) in enumerate(data_loader):
                data = data.to(self.args.device)
                cell_encode_x = self.model(data, return_encodings=True)  # [batch size, max_seq_len, dim]
                cell_encode_x = cell_encode_x[:, :-1, :]
                pool_emb += cell_encode_x.sum(dim=0)
                if pool == 'mean':
                    pool_emb = pool_emb / adata.shape[0]
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        self.logger.info("get gene expression Done!")
        return pool_emb.detach().cpu().numpy()

    def get_cell_embedding(self, adata, cell_emb_type='cls'):
        """
        Extracts cell embeddings using specified aggregation methods.

        Args:
            adata (AnnData): Single-cell data.
            cell_emb_type (str): Embedding aggregation type ('cls', 'sum', or 'mean').

        Returns:
            np.ndarray: Cell embeddings as a NumPy array.
        """
        self.logger.info("start to get cell embedding!")
        dataset, _ = self.load_dataset(adata, split_data=False)
        data_loader = self.get_dataloader(dataset, random_sample=False)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            cell_embeddings = []
            for index, (data, _) in tqdm(enumerate(data_loader), desc='get scbert cell embedding: '):
                data = data.to(self.args.device)
                cell_encode_x = self.model(data, return_encodings=True)  # [batch size, max_seq_len, dim]
                if cell_emb_type == 'cls':
                    cell_emb = cell_encode_x[:, -1, :]
                elif cell_emb_type == 'sum':
                    cell_emb = cell_encode_x[:, 0:-1, :].sum(axis=1)
                else:
                    cell_emb = cell_encode_x[:, 0:-1, :].mean(axis=1)
                cell_embeddings.append(cell_emb.detach().cpu().numpy())
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        cell_embeddings = np.concatenate(cell_embeddings, axis=0)
        self.logger.info("end to get cell embedding!")
        return cell_embeddings

    def get_embedding(self, emb_type, adata=None, gene_ids=None):
        """
        Obtains embeddings for either genes, cells, or gene-expression profiles.

        Args:
            emb_type (str): Type of embedding ('gene', 'cell', or 'gene-expression').
            adata (AnnData, optional): Single-cell data for cell or gene-expression embeddings.
            gene_ids (torch.Tensor, optional): Gene IDs for gene embeddings.

        Returns:
            np.ndarray: Embedding data based on the requested type.
        """
        self.model = self.model.eval()
        assert emb_type in ['gene', 'cell',
                            'gene-expression'], 'Invalid embedding type, must be gene, cell or gene-expression'
        if emb_type == 'gene' and gene_ids is None:
            raise ValueError('gene_ids must not be None if emb_type is gene!')
        if emb_type != 'gene' and adata is None:
            raise ValueError('adata must not be None if emb_type is cell or gene-expression!')
        if emb_type == 'gene':
            return self.get_gene_embedding(gene_ids)
        elif emb_type == 'cell':
            return self.get_cell_embedding(adata, cell_emb_type=self.args.cell_emb_type)
        else:
            return self.get_gene_expression_embedding(adata)

    def encoder(self, batch_data):
        """
        Encodes a batch of data using the SCBERT model.

        Args:
            batch_data (torch.Tensor): Batch of input data to encode.

        Returns:
            torch.Tensor: Encoded representations of the input data.
        """
        cell_encode_x = self.model(batch_data)
        return cell_encode_x


if __name__ == '__main__':
    from biollm.utils.utils import load_config
    import scanpy as sc

    config_file = '../config/embeddings/scbert_emb.toml'
    configs = load_config(config_file)
    adata = sc.read_h5ad(configs.input_file)
    adata = adata[0:100, :]
    obj = LoadScbert(configs)
    print(obj.args)
    if 'gene_name' not in adata.var:
        adata.var['gene_name'] = adata.var.index.values
    gene_ids = list(obj.get_gene2idx().values())
    gene_ids = np.array(gene_ids)
    gene_ids = torch.tensor(gene_ids, dtype=torch.long).to(configs.device)
    emb = obj.get_embedding(obj.args.emb_type, adata, gene_ids)
    print('embedding shape:', emb.shape)
    with open(obj.args.output_dir + f'/scbert_{obj.args.emb_type}_emb.pk', 'wb') as w:
        pkl.dump(emb, w)
