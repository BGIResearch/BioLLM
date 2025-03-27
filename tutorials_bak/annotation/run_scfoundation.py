#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/9 14:41
# @Author  : qiuping
# @File    : run_geneformer.py
# @Email: qiupinghust@163.com

"""
change log: 
    2024/7/9 14:41  create file. 
"""
from biollm.task.annotation.anno_task_scf import AnnoTaskScf
import sys
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
# config_file = './configs/scf.toml'
# config_file = sys.argv[1]
# task = AnnoTaskScf(config_file)
# task.run()

finetune = True
if finetune:
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
    config_file = sys.argv[1]
    # config_file = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/tutorials_bak/annotation/configs/organs'
    task = AnnoTaskScf(config_file)
    task.run()
else:
    import scanpy as sc
    import pickle
    from sklearn.metrics import accuracy_score, f1_score

    organs = "colon,eye,kidney,prostate_gland,spleen,esophagus,immune_system,mucosa,small_intestine,lung"
    result = {}
    for i in organs.split(','):
        try:
            config_file = f'./configs/organs/predict/scf_{i}.toml'
            task = AnnoTaskScf(config_file)
            task.run()
            path = f'/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/result/anno/scf/{i}/'
            predict_label = pickle.load(open(path + 'predict_list.pk', 'rb'))
            adata = sc.read_h5ad(
                f'/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/data/annotation/human/organs/{i}/test.h5ad')
            labels = adata.obs['cell_type_ontology_term_id'].values
            acc = accuracy_score(labels, predict_label)
            macro_f1 = f1_score(labels, predict_label, average='macro')
            result[i] = {'acc': '%.5f'% acc, 'macro_f1': '%.5f'% macro_f1}
            print(i, acc, macro_f1)
        except Exception as e:
            print('error:', i, e)
    print(result)
