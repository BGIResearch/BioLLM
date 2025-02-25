#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: load_scgpt.py
@time: 2024/3/3 11:13
"""
from biollm.base.load_llm import LoadLlm
from biollm.repo.scgpt.tokenizer.gene_tokenizer import GeneVocab
from biollm.repo.scgpt.model import TransformerModel
from biollm.model.perturbation import ScgptPerturbation
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from biollm.repo.scgpt.preprocess import Preprocessor
import numpy as np
from biollm.dataset.scgpt_dataset import SeqDataset, prepare_data, prepare_dataloader, make_dataset


class LoadScgpt(LoadLlm):
    """
    Load and manage the scGPT model for gene expression analysis.

    This class provides methods to load the vocabulary, model parameters,
    and perform preprocessing steps on the single-cell data.

    Args:
        args (Namespace): Command line arguments containing model configuration.

    Attributes:
        vocab (GeneVocab): Vocabulary object for gene representation.
        model (TransformerModel): Loaded Transformer model.
    """
    def __init__(self, args):
        """
        Initializes the LoadScgpt class.

        Args:
            args (Namespace): Command line arguments containing model configuration.
        """
        super(LoadScgpt, self).__init__(args)
        self.vocab = self.load_vocab()
        self.model = self.load_model()
        self.init_model()
        self.model = self.model.to(self.args.device)

    def load_model(self):
        """
        Loads the Transformer model with specified parameters.

        Returns:
            TransformerModel: The initialized Transformer model.
        """
        ntokens = len(self.vocab)
        model_param = {
            'ntoken': ntokens,
            'd_model': self.args.embsize,
            'nhead': self.args.nheads,
            'd_hid': self.args.d_hid,
            'nlayers': self.args.nlayers,
            'nlayers_cls': self.args.nlayers_cls if 'nlayers_cls' in self.args else 3,
            'n_cls': self.args.n_cls if 'n_cls' in self.args else 1,
            'dropout': 0.5,
            'pad_token': "<pad>",
            'do_mvc': False,
            'do_dab': False,
            'use_batch_labels': False,
            'num_batch_labels': None,
            'domain_spec_batchnorm': False,
            'input_emb_style': "continuous",
            'cell_emb_style': "cls",
            'mvc_decoder_style': "inner product",
            'ecs_threshold': 0.3,
            'explicit_zero_prob': False,
            'fast_transformer_backend': "flash",
            'pre_norm': False,
            'vocab': self.vocab,
            'pad_value': self.args.pad_value,
            'n_input_bins': self.args.n_bins,
            'use_fast_transformer': True,
        }
        for i in model_param:
            if i in self.args:
                model_param[i] = self.args[i]
        print(model_param)
        model = TransformerModel(**model_param)
        return model

    def load_pert_model(self):
        """
        Loads the perturbation model with specified parameters.

        Returns:
            ScgptPerturbation: The initialized perturbation model.
        """
        ntokens = len(self.vocab)
        pert_model = ScgptPerturbation(
            ntokens,
            self.args.embsize,
            self.args.nheads,
            self.args.d_hid,
            self.args.nlayers,
            nlayers_cls=self.args.nlayers_cls,
            vocab=self.vocab,
            n_cls=1,
            dropout=self.args.dropout,
            pad_token=self.args.pad_token,
            pad_value=self.args.pad_value,
            pert_pad_id=self.args.pert_pad_id,
            do_mvc=self.args.MVC,
            cell_emb_style=self.args.cell_emb_style,
            mvc_decoder_style=self.args.mvc_decoder_style,
            use_fast_transformer=self.args.use_fast_transformer,
        )
        return pert_model

    def load_vocab(self):
        """
        Loads the gene vocabulary from a file and adds special tokens.

        Returns:
            GeneVocab: The loaded gene vocabulary.
        """
        vocab = GeneVocab.from_file(self.args.vocab_file)
        special_tokens = ['<pad>', '<cls>', '<eoc>']
        for token in special_tokens:
            if token not in vocab:
                vocab.append_token(token)
        vocab.set_default_index(vocab['<pad>'])
        return vocab

    def get_gene2idx(self):
        """
        Retrieves a mapping from gene names to their indices.

        Returns:
            dict: A dictionary mapping gene names to indices.
        """
        return self.vocab.get_stoi()

    def freezon_model(self, keep_layers=[-2]):
        """
        Freezes model parameters except for specified layers.

        Args:
            keep_layers (list): List of layers to keep trainable.
        """
        model_param_count = sum(
            dict((p.data_ptr(), p.numel()) for p in self.model.parameters() if p.requires_grad).values())
        for name, param in self.model.named_parameters():
            if 'encoder' in name and "transformer_encoder" not in name:
                param.requires_grad = False
        ft_param_count = sum(
            dict((p.data_ptr(), p.numel()) for p in self.model.parameters() if p.requires_grad).values())
        self.logger.info(f"Total pretrain-model Encoder Params {model_param_count}")
        self.logger.info(f"The pretrain_model Encoder Params for training in finetune after freezon: {ft_param_count}")

    def get_gene_embedding(self, gene_ids):
        """
        Gets gene embeddings for specified gene IDs.

        Args:
            gene_ids (list): List of gene IDs.

        Returns:
            np.ndarray: Gene embeddings as a NumPy array.
        """
        self.logger.info('start to get gene embedding!')
        gene_embeddings = self.model.encoder(gene_ids)
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        self.logger.info(f'finished get gene embedding!')
        return gene_embeddings.detach().cpu().numpy()

    def preprocess_adata(self, adata):
        """
        Preprocesses the AnnData object for gene expression analysis.

        Args:
            adata (AnnData): The AnnData object to preprocess.

        Returns:
            AnnData: The preprocessed AnnData object.
        """
        if 'gene_name' not in adata.var_keys():
            adata.var['gene_name'] = adata.var_names
        if 'do_preprocess' in adata.uns:
            self.logger.info('the adata was already preprocessed, pass the step!')
            return adata
        adata, _ = self.filter_gene(adata)
        if adata.X.min() >= 0:
            normalize_total = False
            log1p = False
            if adata.X.max() > 20:
                log1p = True
                if adata.X.max() - np.int32(adata.X.max()) == np.int32(0):
                    normalize_total = 10000.0
        else:
            raise Exception('the express matrix have been scale, exit!')
        preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=self.args.filter_gene_by_counts if 'filter_gene_by_counts' in self.args else False,  # step 1
            filter_cell_by_counts=self.args.filter_cell_by_counts if 'filter_cell_by_counts' in self.args else False,  # step 2
            normalize_total=normalize_total,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=log1p,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=self.args.n_hvg if 'n_hvg' in self.args else False,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if log1p else "cell_ranger",
            binning=self.args.n_bins,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )
        preprocessor(adata, batch_key=self.args.batch_key if 'batch_key' in self.args else None)
        adata.uns['do_preprocess'] = True
        return adata

    def filter_gene(self, adata):
        """
        Filters genes in the AnnData object based on the vocabulary.

        Args:
            adata (AnnData): The AnnData object containing gene information.

        Returns:
            tuple: A tuple containing the filtered AnnData object and gene IDs in the vocabulary.
        """
        adata.var["id_in_vocab"] = [
            1 if gene in self.vocab else -1 for gene in adata.var_names.tolist()
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        self.logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(self.vocab)}."
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()
        return adata, gene_ids_in_vocab

    def encoder(self, batch_data):
        """
        Encodes the batch data to obtain gene embeddings.

        Args:
            batch_data (dict): A dictionary containing gene IDs and values.

        Returns:
            Tensor: Encoded embeddings of the input batch.
        """
        input_gene_ids = batch_data["gene_ids"].to(self.args.device)
        input_values = batch_data["values"].to(self.args.device)

        src_key_padding_mask = input_gene_ids.eq(self.vocab[self.args.pad_token])
        embeddings = self.model._encode(
            input_gene_ids,
            input_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )  # batch_size * max_seq_len * dim
        return embeddings

    def get_cell_embedding(self, adata, do_preprocess=False):
        """
        Generates cell embeddings from the given AnnData object.

        This method retrieves cell embeddings using a data loader and the model,
        processing the data in batches. It returns a NumPy array of embeddings
        for each cell in the input AnnData.

        Args:
            adata: AnnData object containing the data to process.
            do_preprocess: (bool) Whether to preprocess the data before loading.
                Defaults to False.

        Returns:
            np.ndarray: A 2D NumPy array of shape (n_cells, embsize),
            where n_cells is the number of cells and embsize is the embedding size.

        Raises:
            RuntimeError: If any issues arise during model inference or data loading.
        """
        self.logger.info('start to get cell embedding!')

        data_loader = self.get_dataloader(adata, do_preprocess=do_preprocess, do_split=False)
        self.logger.info('get dataloader Done!')
        cell_embeddings = np.zeros((adata.shape[0], self.args.embsize), dtype=np.float32)
        # celltypes = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            count = 0
            for batch_data in tqdm(data_loader, desc='Cell embedding'):
                input_gene_ids = batch_data["gene_ids"].to(self.args.device)
                input_values = batch_data["values"].to(self.args.device)
                # celltypes.extend(list(batch_data["celltype_labels"].detach().cpu().numpy()))
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.args.pad_token])
                output = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                )
                embeddings = output['cell_emb']  # get the <cls> position embedding
                embeddings = embeddings.cpu().numpy()
                cell_embeddings[count: count + len(embeddings)] = embeddings
                count += len(embeddings)
            # cell_embeddings = cell_embeddings / np.linalg.norm(
            #     cell_embeddings, axis=1, keepdims=True
            # )
            if self.wandb:
                total_gpu = torch.cuda.memory_allocated() / (1024 ** 3)
                self.wandb.log({'GPU': total_gpu})
            self.logger.info("get cell embedding Done!")
            return cell_embeddings

    def get_gene_expression_embedding(self, adata, do_preprocess=False):
        """
        Computes gene expression embeddings for the provided AnnData object.

        This method processes the data in batches and retrieves embeddings
        for each gene. It averages embeddings across batches and returns
        both the embeddings and corresponding gene names.

        Args:
            adata: AnnData object containing the gene expression data.
            do_preprocess: (bool) If True, preprocesses the data before
                loading. Defaults to False.

        Returns:
            Tuple[np.ndarray, List[str]]: A tuple containing:
                - A 2D NumPy array of shape (n_genes, embsize) with gene
                  embeddings.
                - A list of gene names corresponding to the embeddings.

        Raises:
            RuntimeError: If any issues arise during model inference or data processing.
        """
        self.logger.info("start to get gene expression!")
        data_loader = self.get_dataloader(adata, do_preprocess=do_preprocess, do_split=False)
        gene_embs = {}
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):

            for batch_data in tqdm(data_loader, desc='Gene expression embedding'):

                input_gene_ids = batch_data["gene_ids"].to(self.args.device)
                input_values = batch_data["values"].to(self.args.device)
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.args.pad_token])
                embeddings = self.model._encode(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                )  # batch_size * max_seq_len * dim
                embeddings = embeddings.detach().cpu().numpy()
                for m in range(len(embeddings)):
                    for n in range(len(embeddings[m])):
                        if input_gene_ids[m][n].item() not in gene_embs.keys():
                            gene_embs.update({input_gene_ids[m][n].item(): [embeddings[m][n]]})
                        else:
                            gene_embs[input_gene_ids[m][n].item()].append(embeddings[m][n])


        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        if 60694 in gene_embs:
            del gene_embs[60694]

        for k, v in gene_embs.items():
            gene_embs[k] = np.mean(np.stack(v), axis=0)

        gene_ids = list(gene_embs.keys())
        gene_embs = np.stack(list(gene_embs.values()))

        idx2gene = self.get_idx2gene()
        gene_names = [idx2gene[i] for i in gene_ids]
        self.logger.info("get gene expression Done!")

        return gene_embs, gene_names

    def get_embedding(self, emb_type, adata=None, gene_ids=None):
        """
        Retrieves embeddings based on the specified type.

        This method calls the appropriate embedding method based on the
        provided embedding type, which can be 'gene', 'cell', or
        'gene-expression'.

        Args:
            emb_type: (str) Type of embedding to retrieve. Must be one of
                'gene', 'cell', or 'gene-expression'.
            adata: Optional; AnnData object for cell or gene-expression
                embeddings.
            gene_ids: Optional; List of gene IDs for gene embeddings.
                Required if emb_type is 'gene'.

        Returns:
            Either:
                - np.ndarray: Gene embeddings if emb_type is 'gene'.
                - np.ndarray: Cell embeddings if emb_type is 'cell'.
                - Tuple[np.ndarray, List[str]]: Gene expression embeddings
                  and names if emb_type is 'gene-expression'.

        Raises:
            ValueError: If emb_type is invalid or required arguments are
                missing.
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
            return self.get_cell_embedding(adata, do_preprocess=self.args.do_preprocess)
        else:
            return self.get_gene_expression_embedding(adata, do_preprocess=self.args.do_preprocess)

    def get_dataloader(self, adata, do_preprocess=True, sort_seq_batch=False, do_split=True, shuffle=False, ddp_train=False, drop_last=False):
        """
        Creates a data loader for the provided AnnData object.

        This method preprocesses the data if specified, and splits it into
        training and validation datasets if do_split is True. It prepares
        the data loaders for both training and validation.

        Args:
            adata: AnnData object containing the data to be loaded.
            do_preprocess: (bool) Whether to preprocess the data before
                creating the data loader. Defaults to True.
            sort_seq_batch: (bool) If True, sorts the sequences within
                batches. Defaults to False.
            do_split: (bool) If True, splits the data into training and
                validation sets. Defaults to True.
            shuffle: (bool) If True, shuffles the training data. Defaults to False.
            ddp_train: (bool) Whether to enable distributed data parallel training.
                Defaults to False.
            drop_last: (bool) If True, drops the last incomplete batch.
                Defaults to False.

        Returns:
            If do_split is True:
                Tuple[DataLoader, DataLoader]: A tuple of training and validation
                data loaders.
            Otherwise:
                DataLoader: The training data loader.
        """
        if do_preprocess:
            adata = self.preprocess_adata(adata)
        if do_split:
            tokenized_train, tokenized_valid, train_celltype_labels, valid_celltype_labels, train_batch_labels, valid_batch_labels = make_dataset(
                adata, self.vocab, self.args, do_split)
            train_data_pt = prepare_data(tokenized_train, train_celltype_labels, train_batch_labels, self.args, sort_seq_batch)
            valid_data_pt = prepare_data(tokenized_valid, valid_celltype_labels, valid_batch_labels, self.args, sort_seq_batch)
            train_loader = prepare_dataloader(
                train_data_pt,
                batch_size=self.args.batch_size,
                shuffle=False,
                intra_domain_shuffle=True,
                drop_last=drop_last,
                ddp_train=ddp_train
            )
            valid_loader = prepare_dataloader(
                valid_data_pt,
                batch_size=self.args.batch_size,
                shuffle=False,
                intra_domain_shuffle=False,
                drop_last=drop_last,
                ddp_train=ddp_train
            )
            return train_loader, valid_loader
        else:
            tokenized_train, train_celltype_labels, train_batch_labels = make_dataset(adata, self.vocab, self.args, do_split)
            train_data_pt = prepare_data(tokenized_train, train_celltype_labels, train_batch_labels, self.args,
                                         sort_seq_batch)
            train_loader = prepare_dataloader(
                train_data_pt,
                batch_size=self.args.batch_size,
                shuffle=shuffle,
                intra_domain_shuffle=True,
                drop_last=drop_last,
                ddp_train=ddp_train
            )
            return train_loader

    def get_idx2gene(self):
        """
        Retrieves a mapping from index to gene IDs.

        This method returns a list of gene IDs based on the vocabulary.

        Returns:
            List[str]: A list of gene IDs corresponding to the model's vocabulary.
        """
        return self.vocab.get_itos()
