#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: replicate_anno.py
@time: 2025/3/27 15:53
"""
from biollm.tasks.bio_task import BioTask
import scanpy as sc
import numpy as np
from biollm.evaluate.bm_metrices_anno import compute_metrics
import json
import pickle
import time
from biollm.algorithm.annotation import ClsDecoder
import gc
from biollm.trainer.trainer import Trainer
from biollm.utils.utils import split_data


class ReplicateAnno(BioTask):
    def __init__(self, config_file):
        super(ReplicateAnno, self).__init__(config_file)
        self.logger.info(self.args)