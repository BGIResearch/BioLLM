#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/24 17:10
# @Author  : qiuping
# @File    : predict.py
# @Email: qiupinghust@163.com

"""
change log: 
    2024/7/24 17:10  create file. 
"""


def scbert():
    from biollm.task.annotation.anno_task_scbert import AnnoTaskScbert

    # organs = "colon,eye,kidney,prostate_gland,spleen,esophagus,immune_system,mucosa,small_intestine,lung"
    organs = ["hpancreas_intra", "hPBMC_intra"]
    lr = [5, 10, 15]
    for i in organs:
        config_file = f'./configs/organs/predict/scbert_{i}.toml'
        task = AnnoTaskScbert(config_file)
        task.run()


def scgpt():
    from biollm.task.annotation.anno_task_scgpt import AnnoTaskScgpt

    organs = "colon,eye,kidney,prostate_gland,spleen,esophagus,immune_system,mucosa,small_intestine,lung"
    organs = ["hpancreas_intra", "hPBMC_intra"]
    # organs = "zheng68k"
    lr = [5, 10, 15]
    for i in organs:
        try:
            config_file = f'./configs/organs/predict/scgpt_{i}.toml'
            task = AnnoTaskScgpt(config_file)
            task.run()
        except Exception as e:
            print('error:', i, e)


def gf():
    from biollm.task.annotation.anno_task_gf import AnnoTask
    organs = "colon,eye,kidney,prostate_gland,spleen,esophagus,immune_system,mucosa,small_intestine,lung"
    organs = ["hPBMC_intra"]
    # organs = "zheng68k"
    lr = [5, 10, 15]
    for i in organs:
        try:
            config_file = f'./configs/organs/predict//gf_{i}.toml'
            task = AnnoTask(config_file)
            task.run()
        except Exception as e:
            print('error:', i, e)


def scf():
    from biollm.task.annotation.anno_task_scf import AnnoTaskScf
    organs = "colon,eye,kidney,prostate_gland,spleen,esophagus,immune_system,mucosa,small_intestine,lung"
    organs = ["hpancreas_intra", "hPBMC_intra"]
    # organs = "zheng68k"
    lr = [5, 10, 15]
    for i in organs:
        try:
            config_file = f'./configs/organs/predict/scf_{i}.toml'
            task = AnnoTaskScf(config_file)
            task.run()

        except Exception as e:
            print('error:', i, e)

# scgpt()
# scbert()
# scf()
gf()
