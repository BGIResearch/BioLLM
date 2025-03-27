#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/19 14:53
# @Author  : qiuping
# @File    : debug_scgpt_preprocess.py
# @Email: qiupinghust@163.com

"""
change log: 
    2024/7/19 14:53  create file. 
"""

from biollm.repo.scgpt.preprocess import Preprocessor
import scanpy as sc


adata = sc.read_h5ad('/home/share/huadjyin/home/s_qiuping1/hanyuxuan/exp_qp.h5ad')


preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts= False,  # step 1
            filter_cell_by_counts= False,  # step 2
            normalize_total=False,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=False,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg= False,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if False else "cell_ranger",
            binning=51,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )
preprocessor(adata, batch_key=None)