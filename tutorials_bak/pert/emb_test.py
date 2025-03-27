#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :emb_test.py
# @Time      :2024/4/3 15:33
# @Author    :Luni Hu

import sys
sys.path.insert(0, "/home/share/huadjyin/home/s_huluni/project/bio_model/biollm")

import os
working_dir = "/home/share/huadjyin/home/s_huluni/project/bio_model/biollm/biollm"
os.chdir(working_dir)

from biollm.base.load_scfoundation import LoadScfoundation
cfs_file = "configs/scfoundation_config.toml"
model = LoadScfoundation(cfs_file=cfs_file)
emb = model.get_embedding(emb_mode="gene")

import numpy as np
np.save("pert/scfoundation_emb.npy", emb)

