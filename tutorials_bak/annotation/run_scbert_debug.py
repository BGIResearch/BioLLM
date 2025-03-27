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
from biollm.task.annotation.anno_task_scbert import AnnoTaskScbert
import sys

config_file = 'configs/scbert_predict.toml'
task = AnnoTaskScbert(config_file)
task.run()
