#!/usr/bin/env python3
# coding: utf-8
"""
@file: pert_task.py
@description:
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2024/04/02  create file.
"""
import pickle

from biollm.tasks.bio_task import BioTask

import os
from torch import nn
import torch
from biollm.trainer.anno_scbert_train import train, predict
import scanpy as sc
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from biollm.algorithm.annotation import ScbertClassification
from biollm.repo.st_performer.model.learn_rate import CosineAnnealingWarmupRestarts
import numpy as np
import pickle as pkl
from biollm.evaluate.bm_metrices_anno import compute_metrics
import time


class AnnoTaskScbert(BioTask):
    def __init__(self, config_file):
        super(AnnoTaskScbert, self).__init__(config_file)
        # init the func for the trainer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, eps=1e-4)
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=15,
            cycle_mult=2,
            max_lr=self.args.lr,
            min_lr=1e-6,
            warmup_steps=5,
            gamma=0.9
        )
        self.criterion = nn.CrossEntropyLoss(weight=None)
        # set the distributed trainning
        self.is_master = int(os.environ['RANK']) == 0 if self.args.distributed else True
        if self.is_master:
            self.logger.info(self.args)
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir, exist_ok=True)
        local_rank = int(os.environ['LOCAL_RANK']) if self.args.distributed else 0
        self.logger.info(f'local rank: {local_rank}')
        torch.cuda.set_device(local_rank)
        if self.args.distributed:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
            self.args.device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
        self.args.local_rank = local_rank
        self.criterion = self.criterion.to(local_rank)
        self.wold_size = torch.distributed.get_world_size() if self.args.distributed else 1

    def make_dataloader(self, input_file=None, split_data=True, random_sample=None, pickle_lable=True):
        input_file = input_file if input_file is not None else self.args.input_file
        adata = sc.read_h5ad(input_file)
        if pickle_lable:
            label_dict, label = np.unique(np.array(adata.obs[self.args.label_key]), return_inverse=True)
            with open(f'{self.args.output_dir}/label_dict.pk', 'wb') as fp:
                pkl.dump(label_dict, fp)
            with open(f'{self.args.output_dir}/label.pk', 'wb') as fp:
                pkl.dump(label, fp)
        train_dataset, val_dataset = self.load_obj.load_dataset(adata,
                                                                split_data=split_data,
                                                                label_key=self.args.label_key,
                                                                do_preprocess=self.args.do_preprocess)
        if random_sample is None:
            random_sample = True if self.args.finetune else False
        train_loader = self.load_obj.get_dataloader(train_dataset, random_sample=random_sample, ddp=self.args.distributed)
        val_loader = self.load_obj.get_dataloader(val_dataset, ddp=False,
                                                  random_sample=random_sample) if val_dataset is not None else None
        return train_loader, val_loader

    def run(self):
        t1 = time.time()
        # make the data loader for the trainer
        split_data = True if self.args.finetune else False
        pickle_label = True if self.args.finetune else False
        train_loader, val_loader = self.make_dataloader(split_data=split_data, pickle_lable=pickle_label)
        if not self.args.finetune:
            with open(f'{self.args.output_dir}/label_dict.pk', 'rb') as fp:
                label_list = pkl.load(fp)
                label_dict = dict([(i, label_list[i]) for i in range(len(label_list))])
            class_num = len(label_list)
        else:
            class_num = len(self.load_obj.label_dict)
        # for the finetune setting
        if self.args.finetune:
            self.load_obj.freezon_model(keep_layers=[-2])
        self.model.to_out = ScbertClassification(h_dim=128,
                                                 class_num=class_num,
                                                 max_seq_len=self.args.max_seq_len, dropout=0.).to(self.args.device)
        if not self.args.finetune:
            self.model = self.load_obj.load_pretrain_model(self.args.model_file, self.model)
        self.model = self.model.to(self.args.device)
        if self.args.distributed:
            self.model = DistributedDataParallel(self.model, device_ids=[self.args.local_rank],
                                             output_device=self.args.local_rank, find_unused_parameters=False)
        if self.args.finetune:
            self.model = train(self.model, train_loader, val_loader, self.args, self.args.local_rank, self.wold_size, self.wandb)
        else:
            result = predict(self.model, train_loader, self.wold_size, self.args.device, self.args)
            # loss_fn = nn.CrossEntropyLoss(weight=None)
            # evaluate(0, self.model, train_loader, loss_fn, self.args.device, self.args,  self.args.local_rank, 1, wandb=None)
            predicted_label = [label_dict[i] for i in result]
            adata = sc.read_h5ad(self.args.input_file)
            metrics = compute_metrics(adata.obs[self.args.label_key], predicted_label)
            print(metrics)
            model_file_name = os.path.basename(self.args.model_file)
            with open(self.args.output_dir + f'/predict_list.pk.{model_file_name}.250117', 'wb') as w:
                pickle.dump(predicted_label, w)
        t2 = time.time()
        if self.is_master:
            self.logger.info(f'Total time: {t2 - t1} s.')


if __name__ == "__main__":
    import sys

    config_file = sys.argv[1]
    obj = AnnoTaskScbert(config_file)
    obj.run()
