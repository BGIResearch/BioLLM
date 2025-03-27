#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :pretrain_grn_eval.py
# @Time      :2024/3/21 14:24
# @Author    :Luni Hu
import re
import sys
sys.path.insert(0, "/home/share/huadjyin/home/s_huluni/project/bio_model/biollm")

from biollm.task.grn.network_task import GrnTask
from biollm.evaluate.bm_metrices_grn import evaluate

import os
toml_dir = "/home/share/huadjyin/home/s_huluni/project/bio_model/biollm/biollm/configs"
toml_file = [os.path.join(toml_dir, file) for file in os.listdir(toml_dir) if "grn" in file]

eval_res = dict()

for cfs_file in toml_file:
    task = GrnTask(cfs_file=cfs_file)
    g = task.grn_analysis(quantile_cutoff=99, finetune=False)

    prefix = re.sub(toml_dir+"/", "", cfs_file).split("_")[0]
    result = evaluate(g, modularity=True)
    eval_res.update({prefix:result})

import json
file_path = "pretrain_grn_eval.json"

# Open the file in write mode
with open(file_path, "w") as json_file:
    # Write the dictionary to the file
    json.dump(eval_res, json_file)