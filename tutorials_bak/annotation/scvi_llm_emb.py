#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/16 15:17
# @Author  : qiuping
# @File    : scvi_llm_emb.py
# @Email: qiupinghust@163.com

"""
change log: 
    2024/8/16 15:17  create file. 
"""
import scvi
import scanpy as sc
import numpy as np



adata = sc.read_h5ad('/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/result/zero-shot/cell_emb/int/dataset4/hPancreas.h5ad')


if adata.X.min() >= 0:
    sc.pp.filter_cells(adata, min_genes=200)
    # Normalization
    normalize_total = False
    log1p = False
    # Normalization
    if adata.X.max() > 25:
        log1p = True
        if adata.X.max() - np.int32(adata.X.max()) == np.int32(0):
            normalize_total = 1e4
    if normalize_total:
        sc.pp.normalize_total(adata, target_sum=normalize_total)
        print("Normalizing Data!")
    if log1p:
        sc.pp.log1p(adata)
        print("Transforming Data to Log1P!")
    # subset highly variable genes

sc.pp.highly_variable_genes(adata, subset=True)
adata_test = sc.read_h5ad('/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/data/annotation/human/organs/hpancreas_intra/test.h5ad')
adata_test = adata_test[:, adata.var_names].copy()
adata.obs['cell_type'] = adata.obs['celltype']
adata_test.obs['cell_type'] = adata_test.obs['celltype']



scvi.model.SCVI.setup_anndata(adata, batch_key=None, labels_key="cell_type")

scvi_model = scvi.model.SCVI(adata, n_latent=30, n_layers=2)

scvi_model.train(20, batch_size=90)

scanvi_model = scvi.model.SCANVI.from_scvi_model(scvi_model, "Unknown")

scanvi_model.train(10, batch_size=90)

SCANVI_LATENT_KEY = "scanvi_emb"
SCANVI_PREDICTIONS_KEY = "scanvi_predicted_cell_type"

adata_test.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata_test)
adata_test.obs[SCANVI_PREDICTIONS_KEY] = scanvi_model.predict(adata_test)

true_labels = adata_test.obs['cell_type']
predicted_labels = adata_test.obs[SCANVI_PREDICTIONS_KEY]
eval_metrics = compute_metrics(true_labels, predicted_labels)
print("SCANVI eval, ", eval_metrics)

eval_save_path = os.path.join("results", f"{prefix}_scanvi_eval.pkl")
with open(eval_save_path, 'wb') as file:
    pickle.dump(eval_metrics, file)
