#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :anno_test.opy
# @Time      :2024/4/2 12:01
# @Author    :Luni Hu

import re

import sys
sys.path.insert(0, "/home/share/huadjyin/home/s_huluni/project/bio_model/biollm")

import os
working_dir = "/home/share/huadjyin/home/s_huluni/project/bio_model/biollm/biollm"
os.chdir(working_dir)

from biollm.task.annotation.anno_task_scf import AnnoTask

anno_task = AnnoTask(cfs_file="configs/scfoundation_config.toml")
h5ad_dir = "/home/share/huadjyin/home/s_huluni/project/bio_model/dataset/anno"
h5ad_files = ["SI.h5ad", "blood.h5ad", "kidney.h5ad", "liver.h5ad", "Zheng68K.h5ad"]
print(h5ad_files)

for file in h5ad_files:
    anno_task.args.input_file = h5ad_dir + "/" + file
    print(f"{anno_task.args.input_file} start training!")
    anno_task.train(save_prefix=re.sub(".h5ad", "", file), batch_size=2, num_epochs=30)
    eval_res = anno_task.evaluate()
    print(eval_res)