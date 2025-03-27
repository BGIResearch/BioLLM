#!/usr/bin/env python3
# coding: utf-8
"""
@file: scgpt_eval.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2024/04/18  create file.
"""
from biollm.task.perturbation.pert_task import PertTask


config_file = '../../../biollm/config/pert/scgpt_pert.toml'
obj = PertTask(config_file)
pert_data, gene_ids = obj.make_dataset()

from biollm.evaluate.bm_metrices_pert import eval_scgpt_perturb


best_model_path = obj.args.save_dir + '/best_model.pt'
best_model = obj.load_obj.load_pretrain_model(best_model_path, obj.model)

test_loader = pert_data.dataloader["test_loader"]
test_res = eval_scgpt_perturb(test_loader, best_model, 'cuda:2', 'all', gene_ids)