#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: bio_task.py
@time: 2024/3/3 15:02
"""
from biollm.utils.log_manager import LogManager
import torch
from biollm.loader.scgpt import Scgpt
from biollm.loader.mamba import Scmamba
from biollm.loader.scbert import Scbert
from biollm.loader.scfoundation import Scfoundation
from biollm.loader.geneformer import Geneformer
import scanpy as sc
from biollm.utils.utils import load_config
import numpy as np
import wandb
import os
from biollm.utils.preprocess import preprocess_adata
from biollm.repo.scgpt.tokenizer.gene_tokenizer import GeneVocab
import sys
import importlib


class BioTask(object):
    """
    The BioTask class provides a standardized framework for executing analysis tasks on single-cell data.
    It handles model loading, data processing, and device configuration, enabling seamless integration
    of different pre-trained loader for various analytical tasks.

    Attributes:
        cfs_file (str): Path to the configuration file specifying task parameters and model choices.
        args (Namespace): Parsed arguments from the configuration file.
        device (torch.device): Device configuration, set based on args.
        gene2ids (dict): Mapping of genes to identifiers, initialized as None.
        load_obj (object): Model loader object, initialized based on model choice in args.
        model (torch.nn.Module): Loaded model based on the model type in args.
        vocab (dict): Vocabulary for gene identifiers, loaded from model loader if available.
        is_master (bool): Flag to check if the process is the main process for distributed training.
        wandb (wandb.Run or None): Weights & Biases tracking object, initialized if tracking is enabled.

    Methods:
        __init__(self, cfs_file, data_path=None, load_model=True):
            Initializes BioTask, loads configuration, device, and optionally the model.

        load_model(self):
            Loads and returns the pre-trained model based on the specified model type in args.

        read_h5ad(self, h5ad_file=None, preprocess=True, filter_gene=False):
            Reads and preprocesses single-cell data from an h5ad file, with optional gene filtering.

        filter_genes(self, adata):
            Filters genes in the AnnData object based on the vocabulary, logging the match rate.

        run(self):
            Placeholder for the main task execution method, to be implemented in subclasses.
    """
    def __init__(self, cfs_file, data_path=None, load_model=True):
        """
        Initializes the BioTask instance with configuration, device settings, and model loading.

        Args:
            cfs_file (str): Path to the configuration file.
            data_path (str, optional): Path to the input data file, overrides default if provided.
            load_model (bool): Flag to indicate whether the model should be loaded on initialization.

        Raises:
            Exception: If configuration is missing required attributes.
        """
        self.cfs_file = cfs_file
        self.args = load_config(cfs_file)
        self.logger = LogManager().logger
        if self.args.device == 'cpu' or self.args.device.startswith('cuda'):
            self.device = torch.device(self.args.device)
        else:
            self.device = torch.device('cuda:' + self.args.device)
        self.gene2ids = None
        self.load_obj = None

        if data_path is not None:
            self.args.input_file = data_path
        if load_model:
            self.model = self.load_model()
        self.vocab = self.load_vocab()
        self.is_master = int(os.environ['RANK']) == 0 if 'RANK' in os.environ else True
        if 'weight_bias_track' in self.args and self.args.weight_bias_track and self.is_master:

            wandb.init(project=self.args.project_name, name=self.args.exp_name, config=self.args)
            self.wandb = wandb
        else:
            self.wandb = None
        if 'output_dir' in self.args:
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir, exist_ok=True)

        # if self.model is not None:
        #     self.model = self.model.to(self.device)

    def load_vocab(self):
        """
        Loads the vocabulary used for gene tokenization.

        Returns:
            GeneVocab: Vocabulary object with gene-to-index mappings.
        """
        vocab = GeneVocab.from_file(self.args.vocab_file)
        return vocab

    def get_gene2idx(self):
        return self.vocab.get_stoi()

    def get_id2gene(self):
        return self.vocab.get_itos()

    def load_model(self):
        """
        Loads the specified foundational model based on configuration.

        Returns:
            torch.nn.Module: The loaded model instance, or None if model type is unsupported.

        Raises:
            ValueError: If model type in configuration is unsupported.
        """
        if self.args.model_used == 'scgpt':
            self.load_obj = Scgpt(self.args)
            return self.load_obj.model
        elif self.args.model_used == 'scmamba':
            self.load_obj = Scmamba(self.args)
            return self.load_obj.model
        elif self.args.model_used == 'scbert':
            self.load_obj = Scbert(self.args)
            return self.load_obj.model
        elif self.args.model_used == 'scfoundation':
            self.load_obj = Scfoundation(args=None, cfs_file=self.cfs_file)
            return self.load_obj.model
        elif self.args.model_used == 'geneformer':
            self.load_obj = Geneformer(self.args)
            return self.load_obj.model
        else:
            return None

    def read_h5ad(self, h5ad_file=None, preprocess=True, filter_gene=False):
        """
        Reads single-cell data from an h5ad file, with options for preprocessing and gene filtering.

        Args:
            h5ad_file (str, optional): Path to the h5ad file. If None, uses the input file from args.
            preprocess (bool): Whether to apply preprocessing to the data.
            filter_gene (bool): Whether to filter genes based on vocabulary.

        Returns:
            AnnData: The preprocessed single-cell data.

        Raises:
            ValueError: If preprocessing requires specific parameters not found in args.
        """
        if h5ad_file is not None:
            adata = sc.read_h5ad(h5ad_file)
        else:
            adata = sc.read_h5ad(self.args.input_file)
        if filter_gene:
            adata = self.filter_genes(adata)
        if preprocess:
            hvg = self.args.n_hvg if 'n_hvg' in self.args else False
            adata = preprocess_adata(adata, hvg)
        return adata

    def filter_genes(self, adata):
        """
        Filters genes in the AnnData object based on the vocabulary attribute.

        Args:
            adata (AnnData): Annotated single-cell data matrix.

        Returns:
            AnnData: Filtered AnnData object with genes matching the vocabulary.

        Raises:
            Exception: If vocabulary is not set.
        """
        if self.vocab is None:
            raise Exception("No vocabulary, please set vocabulary first")
        adata.var['is_in_vocab'] = [1 if gene in self.vocab else 0 for gene in adata.var_names]
        in_vocab_rate = np.sum(adata.var["is_in_vocab"]) / adata.var.shape[0]
        self.logger.info(f'match {in_vocab_rate} genes in vocab of size {adata.var.shape[0]}')
        adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()
        return adata

    def run(self):
        """
        Placeholder method to execute the specific analysis task. Should be implemented by subclasses.

        Raises:
            NotImplementedError: Indicates that this method should be overridden by subclasses.
        """
        raise NotImplementedError("Not implemented")

    def llm_embedding(self, emb_type, adata=None, gene_ids=None):
        if self.args.model_used not in ['scbert', 'scgpt', 'geneformer', 'scfoundation']:
            return self.get_bgi_emb(emb_type, adata, gene_ids)
        else:
            return self.load_obj.get_embedding(emb_type, adata=adata, gene_ids=gene_ids)

    def get_cfm_emb(self, emb_type, adata=None, gene_ids=None):
        sys.path.insert(0, '/home/share/huadjyin/home/lishaoshuai/02code/CFM-main')
        if self.args.model_used == 'cfm_v1':
            from cfm.task.Get_Embedding import GeneEmbeddingExtractor
            extractor = GeneEmbeddingExtractor(adata, self.args)
        elif self.args.model_used == 'cfm_v2':
            from cfm.task.Get_EmbeddingV2 import GeneEmbeddingExtractorV2
            extractor = GeneEmbeddingExtractorV2(adata, self.args)
        else:
            raise ValueError(f"{self.args.model_used} is out off range...")

        return extractor.get_embedding(emb_type, gene_ids)

    def get_bgi_emb(self, emb_type, adata=None, gene_ids=None):
        sys.path.insert(0, self.args.bgi_linux_path)
        module = importlib.import_module(self.args.bgi_module_name)
        extractor_class = getattr(module, self.args.bgi_class_name)
        extractor = extractor_class(adata, self.args)
        return extractor.get_embedding(emb_type, gene_ids)

