#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/2 14:32
# @Author  : qiuping
# @File    : run_scbert.py
# @Email: qiupinghust@163.com

"""
change log: 
    2024/7/2 14:32  create file. 
"""
from biollm.task.annotation.anno_task_scgpt import AnnoTaskScgpt
import sys

finetune = True
if finetune:
    config_file = sys.argv[1]
    task = AnnoTaskScgpt(config_file)
    task.run()
else:
    from biollm.task.annotation.anno_task_scgpt import AnnoTaskScgpt
    import scanpy as sc
    import pickle
    from sklearn.metrics import accuracy_score, f1_score

    organs = "colon,eye,kidney,prostate_gland,spleen,esophagus,immune_system,mucosa,small_intestine,lung"
    result = {}
    for i in organs.split(','):
        try:
            config_file = f'./configs/organs/predict/scgpt_{i}.toml'
            task = AnnoTaskScgpt(config_file)
            task.run()
            path = f'/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/result/anno/scgpt/{i}/'
            predict_label = pickle.load(open(path + 'predict_list.pk', 'rb'))
            adata = sc.read_h5ad(
                f'/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/case/data/annotation/human/organs/{i}/test.h5ad')
            labels = adata.obs['cell_type_ontology_term_id'].values
            acc = accuracy_score(labels, predict_label)
            macro_f1 = f1_score(labels, predict_label, average='macro')
            result[i] = {'acc': acc, 'macro_f1': macro_f1}
            print(i, acc, macro_f1)
        except Exception as e:
            print('error:', i, e)
    print(result)

