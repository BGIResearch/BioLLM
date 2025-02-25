#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :load_geneformer.py
# @Time      :2024/3/22 10:17
# @Author    :Luni Hu

import scanpy as sc
import numpy as np


from biollm.base.load_llm import LoadLlm
from transformers import BertForMaskedLM, BertForTokenClassification, BertForSequenceClassification
from biollm.repo.scgpt.tokenizer.gene_tokenizer import GeneVocab
from biollm.repo.geneformer.emb_extractor import get_embs
from biollm.repo.geneformer.tokenizer import TranscriptomeTokenizer
import torch
from scipy.sparse import issparse, csr_matrix
import pickle as pkl
import os


class LoadGeneformer(LoadLlm):
    """
    The LoadGeneformer class provides a specific implementation for loading and utilizing
    the Geneformer model, which can be used in various single-cell and gene expression analysis tasks.
    This class supports loading pre-trained models, generating embeddings, and creating tokenized datasets
    from input data.

    Attributes:
        vocab (GeneVocab): Vocabulary object containing gene-to-index mappings.
        model (torch.nn.Module): Initialized model based on the specified model type and configuration.
    """
    def __init__(self, args=None, cfs_file=None, data_path=None):
        """
        Initializes the LoadGeneformer class, setting up configurations, loading the model,
        and placing it on the specified device.

        Args:
            args (Namespace, optional): Configuration arguments for the model and task.
            cfs_file (str, optional): Path to a configuration file for loading settings.
            data_path (str, optional): Path to the input data file.
        """
        super(LoadGeneformer, self).__init__(args, cfs_file)
        self.vocab = self.load_vocab()
        if data_path is not None:
            self.args.input_file = data_path
        self.model = self.load_model()
        self.model = self.model.to(self.args.device)

    def load_model(self):
        """
        Loads the specified model type based on the arguments provided.
        Supports loading different model types, such as pretrained, gene classifiers, and cell classifiers.

        Returns:
            torch.nn.Module: The initialized model based on specified type and parameters.
        """

        if self.args.model_type == "Pretrained":
            model = BertForMaskedLM.from_pretrained(self.args.model_file,
                                                    output_hidden_states=True,
                                                    output_attentions=False)

        else:
            if os.path.exists(f'{self.args.output_dir}/label_dict.pk'):
                with open(f'{self.args.output_dir}/label_dict.pk', 'rb') as fp:
                    label_list = pkl.load(fp)
                num_classes = len(label_list)
            else:
                adata = sc.read_h5ad(self.args.input_file)
                num_classes = len(adata.obs[self.args.label_key].unique())

            if self.args.model_type == "GeneClassifier":
                model = BertForTokenClassification.from_pretrained(self.args.model_file,
                                                                   num_labels=num_classes,
                                                                   output_hidden_states=True,
                                                                   output_attentions=False)

            elif self.args.model_type == "CellClassifier":
                model = BertForSequenceClassification.from_pretrained(self.args.model_file,
                                                                      num_labels=num_classes,
                                                                      output_hidden_states=True,
                                                                      output_attentions=False)

        return model

    def get_embedding(self, emb_type, adata=None, gene_ids=None):
        """
        Retrieves embeddings for genes or cells based on the specified embedding type.

        Args:
            emb_type (str): Type of embedding to retrieve ("gene" or "cell").
            adata (AnnData, optional): Annotated data object with single-cell data.
            gene_ids (torch.Tensor, optional): Tensor of gene IDs for gene embedding extraction.

        Returns:
            np.ndarray or torch.Tensor: Embeddings of specified type.
            torch.Tensor: Gene IDs if requested in embedding output.
        """
        self.model = self.model.eval()
        if emb_type == "gene":

            emb = self.model.bert.embeddings.word_embeddings(gene_ids)
            if self.wandb:
                total_gpu = torch.cuda.memory_allocated() / (1024 ** 3)
                self.wandb.log({'GPU': total_gpu})
            return emb

        elif emb_type == "cell":

            dataset = self.load_data(adata=adata)

            emb = get_embs(
                self.model,
                dataset=dataset,
                emb_mode=emb_type,
                pad_token_id=self.vocab["<pad>"],
                forward_batch_size=self.args.batch_size, device=self.args.device)
            if self.wandb:
                total_gpu = torch.cuda.memory_allocated() / (1024 ** 3)
                self.wandb.log({'GPU': total_gpu})
            emb = emb.detach().cpu().numpy()

            return emb

        else:
            dataset = self.load_data(adata=adata)

            emb, gene_ids = get_embs(
                self.model,
                dataset=dataset,
                emb_mode=emb_type,
                pad_token_id=self.vocab["<pad>"],
                forward_batch_size=self.args.batch_size, device=self.args.device)
            if self.wandb:
                total_gpu = torch.cuda.memory_allocated() / (1024 ** 3)
                self.wandb.log({'GPU': total_gpu})
            return emb, gene_ids

    def get_gene2idx(self):
        """
        Retrieves the gene-to-index mapping from the vocabulary.

        Returns:
            dict: Mapping of gene names to indices.
        """
        return self.vocab.get_stoi()

    def load_vocab(self):
        """
        Loads the vocabulary used for gene tokenization.

        Returns:
            GeneVocab: Vocabulary object with gene-to-index mappings.
        """
        vocab = GeneVocab.from_file(self.args.vocab_file)
        return vocab

    def load_data(self, adata=None, data_path=None, cell_type_key=None, nproc=16, add_length=True):
        """
        Loads and tokenizes single-cell data, preparing it for embedding extraction.

        Args:
            adata (AnnData, optional): Annotated data object for single-cell data.
            data_path (str, optional): Path to data file if adata is not provided.
            cell_type_key (str, optional): Key for cell type annotation.
            nproc (int, optional): Number of processes for tokenization. Default is 16.
            add_length (bool, optional): Whether to add sequence length information. Default is True.

        Returns:
            Dataset: Tokenized dataset for model input.
        """
        if data_path is not None and adata is None:

            adata = sc.read_h5ad(data_path)
        # if adata.raw is not None:
        #     adata.X = adata.raw.X
        # if adata.X.max() - np.int32(adata.X.max()) != 0:
        #     raise ValueError('Anndata.X must be raw count!')
        if 'n_counts' not in adata.obs.columns:
            if not issparse(adata.X):
                express_x = csr_matrix(adata.X)
            else:
                express_x = adata.X
            adata.obs["n_counts"] = np.ravel(express_x.sum(axis=1))
        if cell_type_key is not None:
            attr_dict = {cell_type_key: "cell_type"}
        else:
            attr_dict = None

        tk = TranscriptomeTokenizer(custom_attr_name_dict=attr_dict,
                                    gene_median_file=self.args.gene_median_file,
                                    token_dictionary_file=self.args.vocab_file,
                                    nproc=nproc)

        tokenized_cells, cell_metadata = tk.tokenize_anndata(adata)

        tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata, add_length=add_length)

        return tokenized_dataset


if __name__ == "__main__":
    from biollm.utils.utils import load_config
    import pickle as pkl
    import os
    import scanpy as sc

    config_file = '../../tutorials/zero-shot/configs/geneformer_gene-expression_emb.toml'
    configs = load_config(config_file)

    obj = LoadGeneformer(configs)
    print(obj.args)
    adata = sc.read_h5ad(configs.input_file)

    obj.model = obj.model.to(configs.device)
    print(obj.model.device)
    emb = obj.get_embedding(obj.args.emb_type, adata=adata)
    print('embedding shape:', emb.shape)
    if not os.path.exists(configs.output_dir):
        os.makedirs(configs.output_dir, exist_ok=True)
    with open(obj.args.output_dir + f'/geneformer_{obj.args.emb_type}_emb.pk', 'wb') as w:
        res = {'gene_names': list(obj.get_gene2idx().keys()), 'gene_emb': emb}
        pkl.dump(emb, w)
