#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/6 14:55
# @Author  : qiuping
# @File    : 04.cal_model_params.py
# @Email: qiupinghust@163.com

"""
change log: 
    2024/6/6 14:55  create file. 
"""
from biollm.utils.utils import cal_model_params, load_config
from biollm.base.load_scgpt import LoadScgpt
from biollm.base.load_scbert import LoadScbert
from biollm.base.load_scfoundation import LoadScfoundation
from biollm.base.load_geneformer import LoadGeneformer

res = {}
config_file = './configs/scbert_cell_emb.toml'
configs = load_config(config_file)
obj = LoadScbert(configs)
params = cal_model_params(obj.model)
res["scbert"] = params

config_file = './configs/scfoundation_cell_emb.toml'
configs = load_config(config_file)
obj = LoadScfoundation(configs)
params = cal_model_params(obj.model)
res["scfoundation"] = params

config_file = './configs/scgpt_cell_emb.toml'
configs = load_config(config_file)
obj = LoadScgpt(configs)
params = cal_model_params(obj.model)
res["scgpt"] = params

config_file = './configs/geneformer_cell_emb.toml'
configs = load_config(config_file)
obj = LoadGeneformer(configs)
params = cal_model_params(obj.model)
res["geneformer"] = params
print(res)
