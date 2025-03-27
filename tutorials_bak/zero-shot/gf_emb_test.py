#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :gf_emb_test.py
# @Time      :2024/6/17 18:28
# @Author    :Luni Hu

import sys
sys.path.append("/home/share/huadjyin/home/s_huluni/project/bio_model/biollm")

import os
os.chdir("/home/share/huadjyin/home/s_huluni/project/bio_model/biollm/tutorials_bak/zero-shot")

from biollm.utils.utils import load_config
from biollm.base.load_geneformer import LoadGeneformer
import scanpy as sc

config_file = './configs/geneformer_gene-expression_emb.toml'
configs = load_config(config_file)

obj = LoadGeneformer(configs)
print(obj.args)

adata = sc.read_h5ad(configs.input_file)[0:1000, :]
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
file_name = os.path.basename(configs.input_file)
obj.model = obj.model.to(configs.device)
emb, gene_ids = obj.get_embedding(obj.args.emb_type, adata=adata)
print('embedding shape:', emb.shape)
print(gene_ids)