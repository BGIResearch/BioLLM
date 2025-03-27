#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/3 10:45
# @Author  : qiuping
# @File    : parse_data.py
# @Email: qiupinghust@163.com

"""
change log: 
    2024/7/3 10:45  create file. 
"""
import scanpy as sc
import os
import anndata
from sklearn.model_selection import train_test_split

def train_test_split_adata(adata, test_size=0.2):
    cell_indices = adata.obs.index
    cell_indices = cell_indices[~cell_indices.duplicated(keep='first')]
    train_indices, test_indices = train_test_split(cell_indices, test_size=test_size)
    print(len(cell_indices), len(train_indices), len(test_indices))
    train_data = adata[train_indices]
    test_data = adata[test_indices]
    return train_data, test_data


def cellxgene():
    indir = '/home/share/huadjyin/home/s_huluni/mashubao/Data/select_organ/Data_hsa'
    outputdir = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/data/annotation/human/organs'
    organs = os.listdir(indir)
    for i in organs:
        print(i)
        if not os.path.exists(os.path.join(outputdir, i)):
            os.makedirs(os.path.join(outputdir, i), exist_ok=True)
        adata1 = sc.read_h5ad(os.path.join(indir, i, 'train.h5ad'))
        adata2 = sc.read_h5ad(os.path.join(indir, i, 'eval.h5ad'))
        adata = sc.read_h5ad(os.path.join(indir, i, 'test.h5ad'))
        adata = sc.read_h5ad(os.path.join(indir, i, 'test.h5ad'))
        train_data = anndata.concat([adata1, adata2], merge="same")
        gene_key = 'gene_symbols' if 'gene_symbols' in train_data.var.columns else 'feature_name'
        train_data.var_names = train_data.var[gene_key].values
        train_data.var['gene_name'] = train_data.var[gene_key].values
        adata.var_names = adata.var[gene_key].values
        adata.var['gene_name'] = adata.var[gene_key].values
        train_data.write_h5ad(os.path.join(outputdir, i, 'train.h5ad'))
        adata.write_h5ad(os.path.join(outputdir, i, 'test.h5ad'))


def zheng68k():
    path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/data/annotation/human/zheng68k/all.h5ad'
    output = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/data/annotation/human/zheng68k/'
    adata = sc.read_h5ad(path)
    train_adata, test_adata = train_test_split_adata(adata, test_size=0.1)
    train_adata.write_h5ad(output + '/train.h5ad')
    test_adata.write_h5ad(output + '/test.h5ad')


zheng68k()