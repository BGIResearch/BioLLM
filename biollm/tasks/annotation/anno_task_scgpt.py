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

from biollm.tasks.bio_task import BioTask
import os
import scanpy as sc
import numpy as np
import torch
from torch import nn
from biollm.trainer.anno_scgpt_train import train, predict
from biollm.dataset.scgpt_dataset import make_train_data
from biollm.evaluate.bm_metrices_anno import compute_metrics
import pickle as pkl
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import pickle
import time


class AnnoTaskScgpt(BioTask):
    def __init__(self, config_file):
        super(AnnoTaskScgpt, self).__init__(config_file, load_model=False)
        self.check_parameters()
        # init the func for the trainer
        self.criterion = nn.CrossEntropyLoss()
        self.is_master = int(os.environ['RANK']) == 0 if self.args.distributed else True
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir, exist_ok=True)
        local_rank = int(os.environ['LOCAL_RANK']) if self.args.distributed else 0
        self.logger.info(f'local rank: {local_rank}')
        torch.cuda.set_device(local_rank)
        if self.args.distributed:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
            self.args.device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
        self.criterion = self.criterion.to(self.args.device)
        self.args.local_rank = local_rank
        self.criterion = self.criterion.to(local_rank)
        self.world_size = torch.distributed.get_world_size() if self.args.distributed else 1
        if self.is_master:
            self.logger.info(self.args)
        self.args['world_size'] = self.world_size

    def freezon_model(self):
        """
        Freezes model parameters except for specified layers.

        Args:
            keep_layers (list): List of layers to keep trainable.
        """
        model_param_count = sum(
            dict((p.data_ptr(), p.numel()) for p in self.model.parameters() if p.requires_grad).values())
        for name, param in self.model.named_parameters():
            if 'cls_decoder' not in name:
                param.requires_grad = False
                print(name)
        ft_param_count = sum(
            dict((p.data_ptr(), p.numel()) for p in self.model.parameters() if p.requires_grad).values())
        self.logger.info(f"Total pretrain-model Encoder Params {model_param_count}")
        self.logger.info(f"The pretrain_model Encoder Params for training in finetune after freezon: {ft_param_count}")

    def check_parameters(self):
        assert self.args.input_style in ["normed_raw", "log1p", "binned"]
        assert self.args.output_style in ["normed_raw", "log1p", "binned"]
        assert self.args.input_emb_style in ["category", "continuous", "scaling"]
        if self.args.input_style == "binned":
            if self.args.input_emb_style == "scaling":
                raise ValueError("input_emb_style `scaling` is not supported for binned input.")
        elif self.args.input_style == "log1p" or self.args.input_style == "normed_raw":
            if self.args.input_emb_style == "category":
                raise ValueError(
                    "input_emb_style `category` is not supported for log1p or normed_raw input."
                )
        if self.args.input_emb_style == "category":
            self.args.mask_value = self.args.n_bins + 1
            self.args.pad_value = self.args.n_bins  # for padding gene expr values
            self.args.n_input_bins = self.args.n_bins + 2
        else:
            self.args.mask_value = -1
            self.args.pad_value = -2
            self.args.n_input_bins = self.args.n_bins

    def make_dataloader(self, adata):

        train_loader, val_loader = self.load_obj.get_dataloader(adata,
                                                                self.args.do_preprocess,
                                                                ddp_train=self.args.distributed,
                                                                drop_last=True)
        return train_loader, val_loader

    def run(self):
        t1 = time.time()
        # for the finetune setting
        if self.args.finetune:
            # make the data loader for the trainer
            adata = sc.read_h5ad(self.args.input_file)
            label_dict, label = np.unique(np.array(adata.obs[self.args.label_key]), return_inverse=True)
            with open(f'{self.args.output_dir}/label_dict.pk', 'wb') as fp:
                pkl.dump(label_dict, fp)
            with open(f'{self.args.output_dir}/label.pk', 'wb') as fp:
                pkl.dump(label, fp)
            celltype_num = len(label_dict)
            label_dict = {label_dict[i]: i for i in range(len(label_dict))}
            if 'celltype_id' not in adata.obs.columns:
                adata.obs['celltype_id'] = [label_dict[i] for i in adata.obs[self.args.label_key]]
            self.args.n_cls = celltype_num
            self.model = self.load_model()
            if self.args.distributed:
                self.model = DistributedDataParallel(self.model, device_ids=[self.args.local_rank],
                                                     output_device=self.args.local_rank, find_unused_parameters=True)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, eps=1e-4)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, self.args.schedule_interval, gamma=self.args.schedule_ratio
            )
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            train_loader, val_loader = self.make_dataloader(adata)
            # self.freezon_model()
            self.load_obj.freezon_model(keep_layers=[-2])
            self.model = self.model.to(self.args.device)
            best_model = train(
                model=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                scaler=self.scaler,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.args.device,
                args=self.args,
                criterion_cls=self.criterion,
                wandb=self.wandb,
                is_master=self.is_master)
            if self.is_master:
                torch.save(best_model.state_dict(), os.path.join(self.args.output_dir, 'anno_scgpt_best_model.pt'))
        else:
            with open(f'{self.args.output_dir}/label_dict.pk', 'rb') as fp:
                label_list = pkl.load(fp)
                label_dict = dict([(i, label_list[i]) for i in range(len(label_list))])
            celltype_num = len(label_list)
            self.args.n_cls = celltype_num
            self.model = self.load_model()
            adata = sc.read_h5ad(self.args.input_file)
            if self.args.do_preprocess:
                adata = self.load_obj.preprocess_adata(adata)

            if 'celltype_id' not in adata.obs.columns:
                adata.obs['celltype_id'] = 0
            if 'batch_id' not in adata.obs.columns:
                adata.obs['batch_id'] = 0
            self.model = self.model.to(self.args.device)
            train_data = make_train_data(adata, self.load_obj.vocab, self.args)
            # data_loader = prepare_dataloader(train_data, self.args.batch_size)
            predictions = predict(self.model, train_data, self.args)
            predicted_label = [label_dict[i] for i in predictions]
            metrics = compute_metrics(adata.obs[self.args.label_key], predicted_label)
            print(metrics)
            model_file_name = os.path.basename(self.args.model_file)
            with open(self.args.output_dir + f'/predict_list.pk.{model_file_name}', 'wb') as w:
                pickle.dump(predicted_label, w)
        t2 = time.time()
        if self.is_master:
            self.logger.info(f'Total time: {t2 - t1} s.')


if __name__ == "__main__":
    import sys

    config_file = sys.argv[1]
    # config_file = '../../config/anno/scgpt_annotation.toml'
    obj = AnnoTaskScgpt(config_file)
    obj.run()
