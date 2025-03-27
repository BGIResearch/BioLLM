#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :load_scFoundation
# @Time      :2024/3/20 15:28
# @Author    :Luni Hu

import torch
import pandas as pd
from biollm.loader.loader_base import LoaderBase
from scipy.sparse import issparse
from biollm.repo.scgpt.tokenizer.gene_tokenizer import GeneVocab
from biollm.repo.scfoundation.load import load_model_frommmf
from biollm.repo.scfoundation.get_embedding import main_gene_selection
import numpy as np
from biollm.repo.scfoundation.load import gatherData, getEncoerDecoderData
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from biollm.utils.preprocess import preprocess_adata


class Scfoundation(LoaderBase):
    """
    LoadScfoundation class is a specialized loader for scFoundation model in the BioLLM framework.
    It initializes and manages a single-cell model, processes gene expression data, and generates gene and cell embeddings.

    Attributes:
    vocab : GeneVocab
        Gene vocabulary loaded from a specified file.
    model : torch.nn.Module
        The pretrained foundational model loaded from the specified file.
    config : dict
        Configuration dictionary for the model, including settings for the encoder and decoder.
    device : torch.device
        The computational device (CPU or GPU) where the model runs.
    gene2idx : dict
        Dictionary mapping gene symbols to indices.
    """
    def __init__(self, args=None, cfs_file=None):
        super(Scfoundation, self).__init__(args, cfs_file)
        self.vocab = self.load_vocab()
        self.model, self.config = self.load_model()
        self.device = torch.device(self.args.device)
        self.model.to(self.device)
        self.gene2idx = self.get_gene2idx()

    def load_model(self):
        """
        Loads the foundational model and configuration from a specified file.

        Returns:
            tuple: A tuple containing the pretrained foundational model (torch.nn.Module)
        """
        model, config = load_model_frommmf(self.args.model_file, key=self.args.key)
        return model, config

    def get_embedding(self, emb_type, adata=None, gene_ids=None):
        """
        Retrieves the embedding for genes, cells, or gene expressions, depending on the specified type.

        Args:
            emb_type (str): Type of embedding to generate ('gene', 'cell', 'gene-expression').
            adata (AnnData, optional): Annotated data object required for 'cell' and 'gene-expression' embeddings.
            gene_ids (list of int, optional): Gene IDs required for 'gene' embedding.

        Returns:
            np.ndarray: The computed embeddings as a NumPy array.
        """

        if adata is not None:
            adata = preprocess_adata(adata, self.args.n_hvg if 'n_hvg' in self.args else False)
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
            return self.get_cell_embedding(adata)
        else:
            return self.get_gene_expression_embedding(adata)
        return emb

    def get_gene_embedding(self, gene_ids):
        """
        Computes embeddings for specified genes using the model's positional embedding layer.

        Args:
            gene_ids (list of int): IDs of genes to generate embeddings for.

        Returns:
            np.ndarray: Gene embeddings as a NumPy array.
        """
        self.logger.info('start to get gene embedding!')
        emb = self.model.pos_emb(gene_ids)
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        self.logger.info('start to get gene embedding!')
        idx2gene = self.get_idx2gene()
        if isinstance(gene_ids, torch.Tensor):
            gene_ids = gene_ids.detach().cpu().numpy()
        gene_names = [idx2gene[i] for i in gene_ids]
        return {'gene_emb': emb.detach().cpu().numpy(), 'gene_names': gene_names}


    def load_data(self, adata=None, max_none_zore=None):
        """
        Loads gene expression data and applies sparse selection based on non-zero thresholding.

        Args:
            adata (AnnData, optional): Annotated data object containing gene expression data.
            max_none_zore (int, optional): Threshold for non-zero values in gene expression data.

        Returns:
            pd.DataFrame: Processed gene expression data.
        """
        if adata is None:
            adata = sc.read_h5ad(self.args.input_file)
        print(adata)
        idx = adata.obs_names.tolist()
        col = adata.var_names.tolist()
        if issparse(adata.X):
            gexpr_feature = adata.X.toarray()
        else:
            gexpr_feature = np.array(adata.X)
        if max_none_zore:
            none_zero = gexpr_feature > 0
            none_zero_num = none_zero.sum(1)
            index = np.argwhere(none_zero_num > max_none_zore).reshape(-1)
            for i in index:
                none_zero_index = np.argwhere(none_zero[i]).reshape(-1)
                np.random.shuffle(none_zero_index)
                mask_num = none_zero_num[i] - max_none_zore
                mask_index = none_zero_index[0: mask_num]
                gexpr_feature[i][mask_index] = 0
        gexpr_feature = pd.DataFrame(gexpr_feature, index=idx, columns=col)
        self.logger.info('covert gene feature into 19264')
        gene_list = list(self.vocab.get_stoi().keys())
        gexpr_feature = gexpr_feature.loc[:, gexpr_feature.columns.isin(gene_list)]
        gexpr_feature, to_fill_columns, var = main_gene_selection(gexpr_feature, gene_list)
        assert gexpr_feature.shape[1] == 19264
        return gexpr_feature

    def get_dataloader(self, dataset, batch_size, shuffle=False, ddp_train=False, drop_last=False, num_workers=0):
        """
        Constructs a DataLoader for efficient batched data processing.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to be loaded.
            batch_size (int): Number of samples per batch.
            shuffle (bool, optional): Whether to shuffle the data.
            ddp_train (bool, optional): Whether to use distributed data parallel (DDP) training.
            drop_last (bool, optional): Whether to drop the last incomplete batch.
            num_workers (int, optional): Number of subprocesses to use for data loading.

        Returns:
            DataLoader: DataLoader instance for the dataset.
        """
        if ddp_train:
            sampler = DistributedSampler(dataset)
        else:
            sampler = SequentialSampler(dataset) if not shuffle else RandomSampler(dataset)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            sampler=sampler,
        )
        return data_loader

    def make_encoder_input(self, gexpr_feature, tgthighres):
        """
        Constructs encoder input features by combining gene expression data and target resolution.

        Args:
            gexpr_feature (pd.DataFrame): Gene expression feature data.
            tgthighres (str): Target resolution information.

        Returns:
            np.ndarray: Prepared input features for the encoder.
        """
        x = gexpr_feature.values
        totalcount = x.sum(axis=1).reshape(-1, 1)
        if tgthighres[0] == 'f':
            pretrain_gene_x = np.concatenate([x, np.log10(totalcount * float(tgthighres[1:])), np.log10(totalcount)], axis=1)
        elif tgthighres[0] == 'a':
            pretrain_gene_x = np.concatenate([x, np.log10(totalcount + float(tgthighres[1:])), np.log10(totalcount)], axis=1)
        elif tgthighres[0] == 't':
            pretrain_gene_x = np.concatenate([x, np.full_like(totalcount, np.float32(tgthighres[1:])), np.log10(totalcount)], axis=1)
        else:
            raise ValueError('tgthighres must be start with f, a or t')

        return pretrain_gene_x

    def get_gene_expression_embedding(self, adata, pool='mean'):
        """
        Obtains gene expression embeddings by pooling the model’s cell-level embeddings.

        Args:
            adata (AnnData): Single-cell data.
            pool (str): Pooling method, either 'mean' or 'max'.

        Returns:
            np.ndarray: Gene expression embeddings as a NumPy array.
        """
        df = self.load_data(adata, max_none_zore=self.args.max_none_zero_num)
        pretrain_gene_x = self.make_encoder_input(df, self.args.tgthighres)
        self.model.to_final = None
        with torch.no_grad(), torch.cuda.amp.autocast():
            pool_emb = torch.zeros((len(self.gene2idx), 512)).to(self.args.device)
            for i in tqdm(range(0, pretrain_gene_x.shape[0], self.args.batch_size)):
                x = pretrain_gene_x[i: i+self.args.batch_size, :]
                x = torch.from_numpy(x).to(self.args.device)
                encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(
                x.float(), x.float(), self.config)
                out = self.model.forward(x=encoder_data, padding_label=encoder_data_padding,
                                    encoder_position_gene_ids=encoder_position_gene_ids,
                                    encoder_labels=encoder_labels,
                                    decoder_data=decoder_data,
                                    mask_gene_name=False,
                                    mask_labels=None,
                                    decoder_position_gene_ids=decoder_position_gene_ids,
                                    decoder_data_padding_labels=decoder_data_padding,
                                    )
                out = out[:, :19264, :].contiguous()
                pool_emb += out.sum(dim=0)
                if pool == 'mean':
                    pool_emb = pool_emb / adata.shape[0]
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        self.logger.info("get gene expression Done!")
        idx2gene = self.get_idx2gene()
        gene_names = [idx2gene[i] for i in range(19264)]
        return {'gene_names': gene_names, 'gene_emb': pool_emb.detach().cpu().numpy()}

    def get_cell_embedding(self, adata, pool='max'):
        """
        Obtains cell embeddings by processing the gene expression data through the model.

        Args:
            adata (AnnData): Single-cell data.
            pool (str): Pooling method, either 'all' or 'max'.

        Returns:
            np.ndarray: Cell embeddings as a NumPy array.
        """
        df = self.load_data(adata, max_none_zore=self.args.max_none_zero_num)
        pretrain_gene_x = self.make_encoder_input(df, self.args.tgthighres)
        cell_embeddings = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for i in tqdm(range(0, pretrain_gene_x.shape[0], self.args.batch_size)):
                x = pretrain_gene_x[i: i+self.args.batch_size, :]
                x = torch.from_numpy(x).to(self.args.device)
                value_labels = x > 0
                data_gene_ids = torch.arange(19266, device=x.device).repeat(x.shape[0], 1)
                x, x_padding = gatherData(x, value_labels, self.config['pad_token_id'])
                position_gene_ids, _ = gatherData(data_gene_ids, value_labels, self.config['pad_token_id'])
                x = self.model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
                position_emb = self.model.pos_emb(position_gene_ids)
                x += position_emb
                geneemb = self.model.encoder(x, x_padding)
                geneemb1 = geneemb[:, -1, :]
                geneemb2 = geneemb[:, -2, :]
                geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
                geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
                if pool == 'all':
                    geneembmerge = torch.concat([geneemb1, geneemb2, geneemb3, geneemb4], axis=1)
                elif pool == 'max':
                    geneembmerge, _ = torch.max(geneemb, dim=1)
                else:
                    raise ValueError('pool_type must be all or max')
                cell_embeddings.append(geneembmerge.detach().cpu().numpy())
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        cell_embeddings = np.concatenate(cell_embeddings, axis=0)
        self.logger.info("end to get cell embedding!")
        return cell_embeddings


if __name__ == '__main__':
    from biollm.utils.utils import load_config
    import pickle as pkl
    import os
    import scanpy as sc

    config_file = '../../tutorials_bak/zero-shot/configs/scfoundation_cell_emb.toml'
    configs = load_config(config_file)

    obj = Scfoundation(configs)
    print(obj.args)

    adata = sc.read_h5ad(configs.input_file)
    adata = adata[:1000, :]
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(configs.emb_type, gene_ids=None, adata=adata)
    print('embedding shape:', emb.shape)
    if not os.path.exists(configs.output_dir):
        os.makedirs(configs.output_dir, exist_ok=True)
    with open(obj.args.output_dir + f'/scfoundation_{obj.args.emb_type}_emb.pk', 'wb') as w:
        res = {'gene_names': list(obj.get_gene2idx().keys()), 'gene_emb': emb}
        pkl.dump(emb, w)
