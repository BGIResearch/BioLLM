#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/18 16:49
# @Author  : qiuping
# @File    : 06.cal_compute_resource.py
# @Email: qiupinghust@163.com

"""
change log: 
    2024/6/18 16:49  create file. 
"""

# scbert gene_embedding

from biollm.utils.utils import load_config
import numpy as np
from biollm.base.load_scbert import LoadScbert
import torch
from biollm.base.load_scgpt import LoadScgpt
from biollm.base.load_geneformer import LoadGeneformer
from biollm.base.load_scfoundation import LoadScfoundation
from biollm.utils.utils import cal_gpu_memory
import wandb
from time import time
import scanpy as sc

batch_size = 8
def scbert():
    t1 = time()
    config_file = config_dir + '/configs/scbert_gene_emb.toml'
    configs = load_config(config_file)
    configs.device = 'cuda:0'
    obj = LoadScbert(configs)
    gene_ids = list(obj.get_gene2idx().values())
    gene_ids = np.array(gene_ids)
    gene_ids = torch.tensor(gene_ids, dtype=torch.long).to(configs.device)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(obj.args.emb_type, gene_ids=gene_ids)
    t2 = time()
    print('embedding shape:', emb.shape)
    gpu_mem = cal_gpu_memory(0)
    wandb.log({'scbert_gpu': gpu_mem, 'scbert_time': t2-t1, 'scbert_gene_num': emb.shape[0], 'scbert_emb_size': emb.shape[1]})


def scgpt():
    t1 = time()
    config_file = config_dir + '/configs/scgpt_gene_emb.toml'
    configs = load_config(config_file)
    configs.device = 'cuda:1'
    obj = LoadScgpt(configs)
    gene_ids = list(obj.get_gene2idx().values())
    gene_ids = np.array(gene_ids)
    gene_ids = torch.tensor(gene_ids, dtype=torch.long).to(configs.device)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(obj.args.emb_type, gene_ids=gene_ids)
    t2 = time()
    print('embedding shape:', emb.shape)
    gpu_mem = cal_gpu_memory(1)
    wandb.log({'scgpt_gpu': gpu_mem, 'scgpt_time': t2-t1, 'scgpt_gene_num': emb.shape[0], 'scgpt_emb_size': emb.shape[1]})


def geneformer():
    t1 = time()
    config_file = config_dir + '/configs/geneformer_gene_emb.toml'
    configs = load_config(config_file)
    configs.device = 'cuda:2'
    obj = LoadGeneformer(configs)
    gene_ids = list(obj.get_gene2idx().values())
    gene_ids = np.array(gene_ids)
    gene_ids = torch.tensor(gene_ids, dtype=torch.long).to(configs.device)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(obj.args.emb_type, gene_ids=gene_ids)
    t2 = time()
    print('embedding shape:', emb.shape)
    gpu_mem = cal_gpu_memory(2)
    wandb.log({'geneformer_gpu': gpu_mem, 'geneformer_time': t2-t1, 'geneformer_gene_num': emb.shape[0], 'geneformer_emb_size': emb.shape[1]})


def scfoundation():
    t1 = time()
    config_file = config_dir + '/configs/scfoundation_gene_emb.toml'
    configs = load_config(config_file)
    configs.device = 'cuda:3'
    obj = LoadScfoundation(configs)
    gene_ids = list(obj.get_gene2idx().values())
    gene_ids = np.array(gene_ids)
    gene_ids = torch.tensor(gene_ids, dtype=torch.long).to(configs.device)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(configs.emb_type, gene_ids=gene_ids)
    t2 = time()
    print('embedding shape:', emb.shape)
    gpu_mem = cal_gpu_memory(3)
    wandb.log({'scfoundation_gpu': gpu_mem, 'scfoundation_time': t2-t1, 'scfoundation_gene_num': emb.shape[0], 'scfoundation_emb_size': emb.shape[1]})


def scbert_exp(adata):
    wandb.init(project='biollm_zero-shot', name=f'scbert_gene_exp_emb_memory_batch_{batch_size}_shape_[{adata.shape[0]},{adata.shape[1]}]')
    t1 = time()
    config_file = config_dir + '/configs/scbert_gene-expression_emb.toml'
    configs = load_config(config_file)
    configs.device = 'cuda'
    obj = LoadScbert(configs)
    # adata = sc.read_h5ad(configs.input_file)
    # adata.var_names = adata.var["gene_symbols"].tolist()
    emb = obj.get_embedding(configs.emb_type, adata=adata)
    t2 = time()
    print('embedding shape:', emb.shape)
    gpu_mem = cal_gpu_memory(0)
    wandb.log({'memory_gpu': gpu_mem, 'time': (t2-t1)/60, 'gene_num': emb.shape[0], 'emb_size': emb.shape[1]})


def scgpt_exp(adata, device=1):
    wandb.init(project='biollm_zero-shot', name=f'scgpt_gene_exp_emb_memory_batch_{batch_size}_shape_[{adata.shape[0]},{adata.shape[1]}]')
    t1 = time()
    config_file = config_dir + '/configs/scgpt_gene-expression_emb.toml'
    configs = load_config(config_file)
    configs.device = 'cuda'
    # adata = sc.read_h5ad(configs.input_file)
    # print(adata)
    # adata.var_names = adata.var["gene_symbols"].tolist()
    # adata.var["gene_name"] = adata.var["feature_name"]
    # adata.obs["celltype_id"] = adata.obs["cell_type"].cat.codes
    # adata.obs["batch_id"] = 0
    obj = LoadScgpt(configs)
    adata, _ = obj.filter_gene(adata)
    configs.max_seq_len = adata.var.shape[0] + 1
    obj = LoadScgpt(configs)
    obj.model = obj.model.to(configs.device)
    emb, _ = obj.get_embedding(configs.emb_type, adata=adata)
    t2 = time()
    print('embedding shape:', emb.shape)
    gpu_mem = cal_gpu_memory(device)
    wandb.log({'memory_gpu': gpu_mem, 'time': (t2-t1)/60, 'gene_num': emb.shape[0], 'emb_size': emb.shape[1]})


def geneformer_exp(adata):
    wandb.init(project='biollm_zero-shot', name=f'geneformer_gene_exp_emb_memory_batch_{batch_size}_shape_[{adata.shape[0]},{adata.shape[1]}]')
    t1 = time()
    config_file = config_dir + '/configs/geneformer_gene-expression_emb.toml'
    configs = load_config(config_file)
    configs.device = 'cuda'
    obj = LoadGeneformer(configs)
    # adata = sc.read_h5ad(configs.input_file)
    # adata.var_names = adata.var["feature_name"]
    obj.model = obj.model.to(configs.device)
    emb, _ = obj.get_embedding(obj.args.emb_type, adata=adata)
    t2 = time()
    print('embedding shape:', emb.shape)
    gpu_mem = cal_gpu_memory(2)
    wandb
    wandb.log({'memory_gpu': gpu_mem, 'time': (t2-t1)/60, 'gene_num': emb.shape[0], 'emb_size': emb.shape[1]})


def scfoundation_exp(adata):
    wandb.init(project='biollm_zero-shot', name=f'scfoundation_gene_exp_emb_memory_batch_{batch_size}_shape_[{adata.shape[0]},{adata.shape[1]}]')
    t1 = time()
    config_file = config_dir + '/configs/scfoundation_gene-expression_emb.toml'
    configs = load_config(config_file)
    configs.device = 'cuda'
    obj = LoadScfoundation(configs)
    # adata = sc.read_h5ad(configs.input_file)
    # adata.var_names = adata.var["gene_symbols"].tolist()
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(configs.emb_type, gene_ids=None, adata=adata)
    t2 = time()
    gpu_mem = cal_gpu_memory(3)
    wandb.log({'memory_gpu': gpu_mem, 'time': (t2-t1)/60, 'gene_num': emb.shape[0], 'emb_size': emb.shape[1]})


def scbert_cell_emb(adata):
    wandb.init(project='biollm_zero-shot',
               name=f'scbert_cell_emb_memory_batch_{batch_size}_shape_[{adata.shape[0]},{adata.shape[1]}]')
    t1 = time()
    config_file = config_dir + '/configs/scbert_cell_emb.toml'
    configs = load_config(config_file)
    configs.device = 'cuda'
    obj = LoadScbert(configs)
    emb = obj.get_embedding(configs.emb_type, adata=adata)
    t2 = time()
    print('embedding shape:', emb.shape)
    gpu_mem = cal_gpu_memory(0)
    wandb.log({'memory_gpu': gpu_mem, 'time': (t2-t1)/60, 'gene_num': emb.shape[0], 'emb_size': emb.shape[1]})


def scgpt_cell_emb(adata):
    wandb.init(project='biollm_zero-shot',
               name=f'scgpt_cell_emb_memory_batch_{batch_size}_shape_[{adata.shape[0]},{adata.shape[1]}]')
    t1 = time()
    config_file = config_dir + '/configs/scgpt_cell_emb.toml'
    configs = load_config(config_file)
    configs.device = 'cuda'
    configs.max_seq_len = adata.var.shape[0] + 1
    obj = LoadScgpt(configs)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(configs.emb_type, adata=adata)
    t2 = time()
    print('embedding shape:', emb.shape)
    gpu_mem = cal_gpu_memory(1)
    wandb.log({'memory_gpu': gpu_mem, 'time': (t2-t1)/60, 'gene_num': emb.shape[0], 'emb_size': emb.shape[1]})


def geneformer_cell_emb(adata):
    wandb.init(project='biollm_zero-shot',
               name=f'geneformer_cell_emb_memory_batch_{batch_size}_shape_[{adata.shape[0]},{adata.shape[1]}]')
    t1 = time()
    config_file = config_dir + '/configs/geneformer_cell_emb.toml'
    configs = load_config(config_file)
    configs.device = 'cuda'
    obj = LoadGeneformer(configs)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(obj.args.emb_type, adata=adata)
    t2 = time()
    print('embedding shape:', emb.shape)
    gpu_mem = cal_gpu_memory(2)
    wandb.log({'memory_gpu': gpu_mem, 'time': (t2-t1)/60, 'gene_num': emb.shape[0], 'emb_size': emb.shape[1]})


def scfoundation_cell_emb(adata):
    wandb.init(project='biollm_zero-shot',
               name=f'scfoundation_cell_emb_memory_batch_{batch_size}_shape_[{adata.shape[0]},{adata.shape[1]}]')
    t1 = time()
    config_file = config_dir + '/configs/scfoundation_cell_emb.toml'
    configs = load_config(config_file)
    configs.device = 'cuda'
    obj = LoadScfoundation(configs)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(configs.emb_type, gene_ids=None, adata=adata)
    t2 = time()
    print('embedding shape:', emb.shape)
    gpu_mem = cal_gpu_memory(3)
    wandb.log({'memory_gpu': gpu_mem, 'time': (t2-t1)/60, 'gene_num': emb.shape[0], 'emb_size': emb.shape[1]})


if __name__ == '__main__':
    import sys

    config_dir = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/tutorials_bak/zero-shot'

    adata = sc.read_h5ad('/home/share/huadjyin/home/s_huluni/cellxgene/h5ads/d7d7e89c-c93a-422d-8958-9b4a90b69558.h5ad')

    print(adata)
    adata.var_names = adata.var["gene_symbols"].tolist()
    adata.var["gene_name"] = adata.var["feature_name"]
    adata.obs["celltype_id"] = adata.obs["cell_type"].cat.codes
    adata.obs["batch_id"] = 0
    # sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    adata = adata[0:15000, :]
    device = int(sys.argv[1])
    # if device == 0:
    #     adata = adata[0:3000, :]
    #     scgpt_exp(adata, 0)
    # elif device == 1:
    #     adata = adata[0:5000, :]
    #     scgpt_exp(adata, 1)
    # elif device == 2:
    #     adata = adata[0:8000, :]
    #     scgpt_exp(adata, 2)
    # elif device == 3:
    #     adata = adata[0:10000, :]
    #     scgpt_exp(adata, 3)
    if device == 0:
        scbert_exp(adata)
    elif device == 1:
        scgpt_exp(adata)
    elif device == 2:
        geneformer_exp(adata)
    elif device == 3:
        scfoundation_exp(adata)
    elif device == 4:
        scbert_cell_emb(adata)
    elif device == 5:
        scgpt_cell_emb(adata)
    elif device == 6:
        geneformer_cell_emb(adata)
    else:
        scfoundation_cell_emb(adata)
