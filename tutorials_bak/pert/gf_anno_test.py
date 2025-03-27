#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :gf_anno_test.py
# @Time      :2024/4/22 10:20
# @Author    :Luni Hu


import re

import wandb
wandb.init(mode="offline")

import sys
sys.path.insert(0, "/home/share/huadjyin/home/s_huluni/project/bio_model/biollm")

import os
working_dir = "/home/share/huadjyin/home/s_huluni/project/bio_model/biollm/biollm"
os.chdir(working_dir)

from biollm.task.annotation.anno_task_gf import AnnoTask

h5ad_dir = "/home/share/huadjyin/home/s_huluni/project/bio_model/dataset/anno"
# "SI.h5ad"
h5ad_files = ["kidney.h5ad", "liver.h5ad", "blood.h5ad"]
print(h5ad_files)

for file in h5ad_files:
    data_path = h5ad_dir + "/" + file

    directory = os.path.dirname(os.path.abspath(__file__))
    filename = re.sub(".h5ad", "", file)  # Remove the ".h5ad" extension from the filename

    out_dir = os.path.join(directory, "gf_" + filename)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    anno_task = AnnoTask(cfs_file="configs/geneformer_config.toml",
                         data_path=data_path, out_dir=out_dir)
    print(f"{data_path} start training!")
    anno_task.train()
    predictions = anno_task.predict()
    print(predictions.metrics)