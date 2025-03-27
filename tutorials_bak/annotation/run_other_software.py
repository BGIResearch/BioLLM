#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :hsa_anno.py
# @Time      :2024/7/30 14:55
# @Author    :Luni Hu

import sys
sys.path.insert(0, "/home/share/huadjyin/home/s_huluni/project/bio_tools/CATree")


from catree.dataset.scDataset import scDataset
from catree.trainer.catree_trainer_v1 import CatreeTrainer
from catree.task.anno_predict import catree_predict
import pickle
import celltypist
import scanpy as sc
import scvi
from catree.utils.utils import compute_metrics
import os


dataset_dict = {
    'hpancreas_intra': '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/data/annotation/human/organs/hpancreas_intra/',
    'hPBMC_intra': '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/data/annotation/human/organs/hPBMC_intra/',
    'zheng68k': '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/data/annotation/human/organs/zheng68k/',
}
label_keys = {
    'hpancreas_intra': 'Celltype',
    'hPBMC_intra': 'cell_type_ontology_term_id',
    'zheng68k': 'celltype',
}

for prefix in dataset_dict:
    print(prefix + " start running")
    adata = sc.read_h5ad(dataset_dict[prefix] + 'train.h5ad')
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
    adata_test = sc.read_h5ad(dataset_dict[prefix] + 'test.h5ad')
    adata_test = adata_test[:, adata.var_names].copy()
    adata.obs['cell_type'] = adata.obs[label_keys[prefix]]
    adata_test.obs['cell_type'] = adata_test.obs[label_keys[prefix]]


    try:
        new_model = celltypist.train(adata, labels='cell_type', max_iter=30, check_expression=False)
        model_path = os.path.join("celltypist", prefix)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        new_model.write(os.path.join(model_path, f'{prefix}_celltypist_train_model.pkl'))
        print("Model saved !")


        predictions = celltypist.annotate(adata_test, model=new_model)

        adata_test.obsm["celltypist_emb"] = predictions.probability_matrix
        adata_test.obs["celltypist_predicted_cell_type"] = predictions.predicted_labels

        true_labels = adata_test.obs['cell_type']
        predicted_labels = adata_test.obs['celltypist_predicted_cell_type']

        eval_metrics = compute_metrics(true_labels, predicted_labels)
        print("CellTypist eval, ", eval_metrics)

        eval_save_path = os.path.join("results", f"{prefix}_celltypist_eval.pkl")
        with open(eval_save_path, 'wb') as file:
            pickle.dump(eval_metrics, file)

    except:
        print(f"Error: {prefix} celltypist")


    try:


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

    except:
        print(f"Error: {prefix} scanvi")

    try:
        import sys
        sys.path.append("/home/share/huadjyin/home/s_huluni/project/bio_tools/CATree/case/Fig3_Ref_Anno/script")

        from catree.utils.onclass_utils import read_ontology_file, read_data
        from catree.utils.config import ontology_data_dir, NHIDDEN, MAX_ITER
        from OnClass.OnClassModel import OnClassModel
        import numpy as np

        print('read ontology data and initialize training model...')
        cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file('cell ontology', ontology_data_dir)
        OnClass_train_obj = OnClassModel(cell_type_nlp_emb_file=cell_type_nlp_emb_file,
                                         cell_type_network_file=cell_type_network_file)

        print('read training single cell data...')
        train_label = 'cell_type_ontology_term_id'

        train_feature, train_genes, train_label, _, _ = read_data(adata=adata, feature_file=None,
                                                                  cell_ontology_ids=OnClass_train_obj.cell_ontology_ids,
                                                                  exclude_non_leaf_ontology=False, tissue_key=None,
                                                                  AnnData_label_key=train_label, filter_key={},
                                                                  nlp_mapping=False, cl_obo_file=cl_obo_file,
                                                                  cell_ontology_file=cell_type_network_file,
                                                                  co2emb=OnClass_train_obj.co2vec_nlp)

        train_feature = train_feature.toarray()

        print('embed cell types using the cell ontology...')
        OnClass_train_obj.EmbedCellTypes(train_label)

        print('read test single cell data...')

        test_label = 'cell_type_ontology_term_id'
        x = adata_test.copy()
        x.X = x.X
        test_label = np.array(x.obs[test_label].tolist())
        test_feature = x.X.toarray()
        test_genes = np.array([x.upper() for x in x.var_names])

        print('generate pretrain model. Save the model to $model_path...')
        cor_train_feature, cor_test_feature, cor_train_genes, cor_test_genes = OnClass_train_obj.ProcessTrainFeature(
            train_feature, train_label, train_genes, test_feature=test_feature, test_genes=test_genes)

        model_path = os.path.join("onclass")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = model_path+f"/{prefix}"
        OnClass_train_obj.BuildModel(ngene=len(cor_train_genes), nhidden=NHIDDEN)
        OnClass_train_obj.Train(cor_train_feature, train_label, save_model=model_path, max_iter=MAX_ITER)

        print('initialize test model. Load the model from $model_path...')
        OnClass_test_obj = OnClassModel(cell_type_nlp_emb_file=cell_type_nlp_emb_file,
                                        cell_type_network_file=cell_type_network_file)
        cor_test_feature = OnClass_train_obj.ProcessTestFeature(cor_test_feature, cor_test_genes,
                                                                use_pretrain=model_path,
                                                                log_transform=False)
        OnClass_test_obj.BuildModel(ngene=None, use_pretrain=model_path)

        # prediction
        pred_Y_seen, pred_Y_all, pred_label = OnClass_test_obj.Predict(cor_test_feature,
                                                                       test_genes=cor_test_genes,
                                                                       use_normalize=False)
        pred_label = np.argmax(pred_Y_seen,axis=1)
        adata_test.obs["onclass_predicted_cell_type"] = [OnClass_test_obj.i2co[l] for l in pred_label]
        adata_test.obsm["onclass_emb"] = pred_Y_seen

        true_labels = adata_test.obs["cell_type_ontology_term_id"]
        pred_labels = adata_test.obs["onclass_predicted_cell_type"]
        eval_metrics = compute_metrics(true_labels, pred_labels)
        print("OnClass eval, ", eval_metrics)

        eval_save_path = os.path.join("results", f"{prefix}_onclass_eval.pkl")
        with open(eval_save_path, 'wb') as file:
            pickle.dump(eval_metrics, file)

    except:
        print(f"Error: {prefix} OnClass running")


    try:
        import singlecellexperiment as sce
        import pandas as pd
        import numpy as np

        ref_data = sce.SingleCellExperiment.from_anndata(adata)
        features = [str(x) for x in ref_data.row_data["feature_name"]]

        import singler
        built = singler.build_single_reference(
            ref_data=ref_data.assay("X"),
            ref_labels=ref_data.col_data.column("cell_type"),
            ref_features=features,
            restrict_to=features,
            num_threads=12
        )

        test_data = sce.SingleCellExperiment.from_anndata(adata_test)

        output = singler.classify_single_reference(
            test_data.assay("X"),
            test_features=features,
            ref_prebuilt=built,
            num_threads=12
        )

        adata_test.obs["singler_predicted_cell_type"] = output["best"]
        adata_test.obsm["singler_emb"] = np.array(pd.DataFrame(output["scores"].data))

        pred_labels = adata_test.obs["singler_predicted_cell_type"]
        true_labels = adata_test.obs["cell_type"]

        eval_metrics = compute_metrics(true_labels, pred_labels)
        print("SingleR eval, ", eval_metrics)

        eval_save_path = os.path.join("results", f"{prefix}_singler_eval.pkl")
        with open(eval_save_path, 'wb') as file:
            pickle.dump(eval_metrics, file)

    except:
        print(f"Error: {prefix} singler")

    sc.write(f"results/{prefix}_preds.h5ad", adata_test)
