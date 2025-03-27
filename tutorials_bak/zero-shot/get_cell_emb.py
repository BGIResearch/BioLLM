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


def scfoundation(adata, output_dir):
    config_file = './configs/scfoundation_cell_emb.toml'
    configs = load_config(config_file)
    obj = LoadScfoundation(configs)
    print(obj.args)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(configs.emb_type, gene_ids=None, adata=adata)
    print('embedding shape:', emb.shape)
    scf_cell_emb = pd.DataFrame(emb, index=adata.obs_names)

    cell_emb_file = os.path.join(output_dir, "scf_cell_emb.pkl")
    with open(cell_emb_file, 'wb') as file:
        pickle.dump(np.array(scf_cell_emb), file)


def scbert(adata, output_dir):
    from biollm.base.load_scbert import LoadScbert
    config_file = './configs/scbert_cell_emb.toml'
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
    config_file = './configs/scgpt_cell_emb.toml'
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
    config_file = './configs/geneformer_cell_emb.toml'
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


def mamba(adata, output_dir):
    from biollm.base.load_mamba import LoadScmamba
    config_file = './configs/scmamba_cell_emb.toml'
    configs = load_config(config_file)
    obj = LoadScmamba(configs)
    if 'gene_name' not in adata.var:
        adata.var['gene_name'] = adata.var.index.values
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(configs.emb_type, adata=adata)
    mamba_cell_emb = pd.DataFrame(emb, index=adata.obs_names)
    cell_emb_file = os.path.join(output_dir, "mamba_cell_emb.pkl")
    with open(cell_emb_file, 'wb') as file:
        pickle.dump(np.array(mamba_cell_emb), file)


if __name__ == '__main__':
    from collections import defaultdict
    import time
    import json


    path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/data/zero-shot/blood/'
    output_dir = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/result/zero-shot/cell_emb/int/hvg'
    datasets = ['14f56226-cbe4-4933-b1a7-d10cb44de300.h5ad']
    # datasets = ['dataset5/hPBMC.h5ad']
    hvg_list = [0, 500, 1000, 1500, 2000, 3000]
    running_time = defaultdict(dict)
    for i in datasets:
        times = {}
        for hvg in hvg_list:
            adata = sc.read_h5ad(path + i)
            output = os.path.join(output_dir + str(hvg), 'blood')
            # output = os.path.join(output_dir + str(hvg), i.split('/')[0])
            if not os.path.exists(output):
                os.makedirs(output, exist_ok=True)
            adata.obs['batch_id'] = 0
            adata.obs['celltype_id'] = 0
            adata.var_names = adata.var['gene_name'].values
            adata.X = adata.raw.X
            sc.pp.normalize_total(adata, target_sum=10000)
            sc.pp.log1p(adata)
            if hvg > 0:
                sc.pp.highly_variable_genes(adata, n_top_genes=hvg, subset=True)
            t1 = time.time()
            scfoundation(adata, output)
            t2 = time.time()
            scbert(adata, output)
            t3 = time.time()
            scgpt(adata, output)
            t4 = time.time()
            geneformer(adata, output)
            t5 = time.time()
            model_time = {'scf': t2-t1, 'scbert': t3-t2, 'scgpt': t4-t3, 'geneformer': t5-t4}
            times[hvg] = model_time
        running_time[i] = times
    with open(output_dir + '/running_time.json', 'w') as f:
        print(running_time)
        json.dump(running_time, f)


    """path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/data/annotation/human/organs/zheng68k/'
    output_dir = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/result/zero-shot/cell_emb/int/hvg'
    datasets = ['blood.h5ad', 'kidney.h5ad', 'liver.h5ad', 'Zheng68K.h5ad']
    datasets = ['all.h5ad']
    # datasets = ['dataset5/hPBMC.h5ad']
    # hvg_list = [500, 1000, 1500, 2000, 3000]
    hvg = False
    running_time = defaultdict(dict)
    for i in datasets:
        times = {}
        adata = sc.read_h5ad(path + i)
        output = os.path.join(output_dir + str(hvg), i.split('.')[0])
        if not os.path.exists(output):
            os.makedirs(output, exist_ok=True)
        adata.obs['batch_id'] = 0
        adata.obs['celltype_id'] = 0
        # sc.pp.highly_variable_genes(adata, n_top_genes=hvg, subset=True)
        t1 = time.time()
        scfoundation(adata, output)
        t2 = time.time()
        scbert(adata, output)
        t3 = time.time()
        scgpt(adata, output)
        t4 = time.time()
        geneformer(adata, output)
        t5 = time.time()
        model_time = {'scf': t2 - t1, 'scbert': t3 - t2, 'scgpt': t4 - t3, 'geneformer': t5 - t4}
        times[hvg] = model_time
        running_time[i] = times
    print(running_time)
    # with open(output_dir + '/running_time_nohvg.json', 'w') as f:
    #     print(running_time)
    #     json.dump(running_time, f)
"""