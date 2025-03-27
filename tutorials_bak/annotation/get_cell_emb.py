#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/22 14:45
# @Author  : qiuping
# @File    : get_cell_emb.py.py
# @Email: qiupinghust@163.com

"""
change log:
    2024/7/22 14:45  create file.
"""
from biollm.utils.utils import load_config
import scanpy as sc
import pandas as pd
import numpy as np
from biollm.base.load_scfoundation import LoadScfoundation
import os
import pickle
from biollm.base.load_scgpt import LoadScgpt
from biollm.model.annotation import LinearProbingClassifier


def scfoundation(adata, output_dir):
    import torch
    import re

    config_file = './configs/finetune/scfoundation_cell_emb.toml'
    configs = load_config(config_file)
    obj = LoadScfoundation(configs)
    pretrained_dict = torch.load(configs.finetune_model, map_location='cpu')
    model_dict = obj.model.state_dict()
    pretrained_dict = {re.sub(r'model.', '', k): v for k, v in pretrained_dict.items()}
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    for k, v in pretrained_dict.items():
        print(f"Loading params {k} with shape {v.shape}")
    model_dict.update(pretrained_dict)
    obj.model.load_state_dict(model_dict)
    emb = obj.get_embedding(configs.emb_type, gene_ids=None, adata=adata)
    print('embedding shape:', emb.shape)
    scf_cell_emb = pd.DataFrame(emb, index=adata.obs_names)

    cell_emb_file = os.path.join(output_dir, "scf_cell_emb.pkl")
    with open(cell_emb_file, 'wb') as file:
        pickle.dump(np.array(scf_cell_emb), file)


def scbert(adata, output_dir):
    from biollm.base.load_scbert import LoadScbert
    config_file = './configs/finetune/scbert_cell_emb.toml'
    configs = load_config(config_file)
    obj = LoadScbert(configs)
    print(obj.args)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(configs.emb_type, adata=adata)
    print('embedding shape:', emb.shape)
    scb_cell_emb = pd.DataFrame(emb, index=adata.obs_names)
    cell_emb_file = os.path.join(output_dir, "scbert_cell_emb.pkl")
    with open(cell_emb_file, 'wb') as file:
        pickle.dump(np.array(scb_cell_emb), file)


def scgpt(adata, output_dir):
    config_file = './configs/finetune/scgpt_cell_emb.toml'
    configs = load_config(config_file)
    # adata = sc.read_h5ad(configs.input_file)
    # adata.var_names = adata.var["feature_name"].tolist()
    # adata.var["gene_name"] = adata.var["feature_name"]
    # adata.obs["celltype_id"] = adata.obs["cell_type"].cat.codes
    # adata.obs["batch_id"] = 0
    # sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True)
    obj = LoadScgpt(configs)
    adata = adata[:, adata.var_names.isin(obj.get_gene2idx().keys())].copy()
    # configs.max_seq_len = adata.var.shape[0] + 1
    obj = LoadScgpt(configs)
    print(obj.args)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(configs.emb_type, adata=adata)
    print('embedding shape:', emb.shape)
    scg_cell_emb = pd.DataFrame(emb, index=adata.obs_names)
    cell_emb_file = os.path.join(output_dir, "scg_cell_emb.pkl")
    with open(cell_emb_file, 'wb') as file:
        pickle.dump(np.array(scg_cell_emb), file)


def geneformer(adata, output_dir):
    from biollm.base.load_geneformer import LoadGeneformer
    config_file = './configs/finetune/geneformer_cell_emb.toml'
    configs = load_config(config_file)
    obj = LoadGeneformer(configs)
    print(obj.args)
    # adata = sc.read_h5ad(configs.input_file)
    # adata.var_names = adata.var["feature_name"]
    # sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(obj.args.emb_type, adata=adata)
    print('embedding shape:', emb.shape)
    gf_cell_emb = pd.DataFrame(emb, index=adata.obs_names)
    cell_emb_file = os.path.join(output_dir, "gf_cell_emb.pkl")
    with open(cell_emb_file, 'wb') as file:
        pickle.dump(np.array(gf_cell_emb), file)


if __name__ == '__main__':
    path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/data/annotation/human/organs/hpancreas_intra/train.h5ad'
    output = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/result/anno/cell_emb/hpancreas_intra_hvg_train/'
    adata = sc.read_h5ad(path)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True)
    print(adata)
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    adata.obs['batch_id'] = 0
    adata.obs['celltype_id'] = 0
    scfoundation(adata, output)
    scbert(adata, output)
    scgpt(adata, output)
    geneformer(adata, output)
