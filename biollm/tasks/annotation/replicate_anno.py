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
import json
import torch
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from biollm.trainer import anno_scgpt_train, anno_scbert_train
from biollm.evaluate.bm_metrices_anno import compute_metrics
import pickle
from biollm.algorithm.annotation import ScbertClassification


class ReplicateAnno(BioTask):
    def __init__(self, config_file):
        super(ReplicateAnno, self).__init__(config_file)
        self.logger.info(self.args)
        # init the ddp
        self.is_master = int(os.environ['RANK']) == 0 if self.args.distributed else True
        self.args.is_master = self.is_master
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir, exist_ok=True)
        local_rank = int(os.environ['LOCAL_RANK']) if self.args.distributed else int(self.args.device.lstrip('cuda:'))
        self.logger.info(f'local rank: {local_rank}')
        torch.cuda.set_device(local_rank)
        if self.args.distributed:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
            self.args.device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
        self.args.local_rank = local_rank
        self.world_size = torch.distributed.get_world_size() if self.args.distributed else 1
        if self.is_master:
            self.logger.info(self.args)
        self.args['world_size'] = self.world_size

    def split_adata(self, adata):
        train_adata, val_adata = self.data_handler.split_adata(adata, train_ratio=0.9)
        return train_adata, val_adata

    def get_dataloader(self, adata, obs_key, shuffle, ddp_train, drop_last, obs_id_output=None):
        data_loader = None
        if self.args.model_used == 'scgpt':
            data_loader = self.load_obj.get_dataloader(adata=adata, var_key=self.args.var_key, obs_key=obs_key,
                                                       n_hvg=self.args.n_hvg, bin_num=self.args.n_bins,
                                                       batch_size=self.args.batch_size,
                                                       obs_id_output=obs_id_output,
                                                       ddp_train=ddp_train, shuffle=shuffle, drop_last=drop_last)
        elif self.args.model_used == 'scbert':
            data_loader = self.load_obj.get_dataloader(adata=adata, var_key=self.args.var_key, obs_key=obs_key, n_hvg=0,
                                                       bin_num=self.args.n_bins, batch_size=self.args.batch_size,
                                                       ddp_train=ddp_train, obs_id_output=obs_id_output,
                                                       shuffle=shuffle, drop_last=drop_last)
        return data_loader

    def init_model_for_finetune(self, labels_num):
        if self.args.model_used == 'scgpt':
            self.args.n_cls = labels_num
            self.model = self.load_model()
        if self.args.model_used == 'scbert':
            self.model.to_out = ScbertClassification(h_dim=128,
                                                     class_num=labels_num,
                                                     max_seq_len=self.args.max_seq_len, dropout=0.)
        if self.args.model_used != 'geneformer':
            if self.args.distributed:
                self.model = DistributedDataParallel(self.model, device_ids=[self.args.local_rank],
                                                     output_device=self.args.local_rank, find_unused_parameters=True)
        self.load_obj.freezon_model(keep_layers=[-2])
        self.model = self.model.to(self.args.device)
        return self.model

    def init_model_for_infer(self, labels_num):
        if self.args.model_used == 'scgpt':
            self.args.n_cls = labels_num
            self.model = self.load_model()
        return self.model

    def infer(self, model, data_loader, trainer_module, true_lbaels, label2id):

        predictions = trainer_module[self.args.model_used].predict(model, data_loader, self.args)
        id2label = {v: k for k, v in label2id.items()}
        predicted_label = [id2label[i] for i in predictions]
        metrics = compute_metrics(true_lbaels, predicted_label)
        with open(self.args.output_dir + f'/predict_list.pk.', 'wb') as w:
            pickle.dump(predicted_label, w)
        with open(self.args.output_dir + f'/metrics.json', 'w') as w:
            json.dump(metrics, w)
        print("Metrics in test data:", metrics)

    def train(self, adata, trainer_module):
        self.logger.info("start to split data for training...")
        train_adata, val_adata = self.split_adata(adata)
        self.logger.info("start to get train_dataloader")
        train_loader = self.get_dataloader(train_adata, self.args.label_key, shuffle=True,
                                           obs_id_output=f"{self.args.output_dir}/label2id.json",
                                           ddp_train=self.args.distributed, drop_last=True)
        with open(f"{self.args.output_dir}/label2id.json", 'r') as f:
            label2id = json.load(f)
            label_num = len(label2id)
        model = self.init_model_for_finetune(label_num)
        self.logger.info("start to get val and test dataloader")
        val_loader = self.get_dataloader(val_adata, self.args.label_key, shuffle=False,
                                         ddp_train=self.args.distributed, drop_last=False)

        self.logger.info("start to training...")
        best_model = trainer_module[self.args.model_used].train(model=model, train_loader=train_loader,
                                                                val_loader=val_loader,
                                                                args=self.args, wandb=self.wandb)
        if self.is_master:
            torch.save(best_model.state_dict(), os.path.join(self.args.output_dir, 'anno_scgpt_best_model.pt'))
        if 'test_file' in self.args:
            test_adata = sc.read_h5ad(self.args.test_file)
            test_loader = self.get_dataloader(test_adata, self.args.label_key, shuffle=False,
                                              ddp_train=self.args.distributed, drop_last=False)
            true_labels = test_adata.obs[self.args.label_key]
            self.infer(best_model, test_loader, trainer_module, true_labels, label2id)

    def run(self):
        adata = sc.read_h5ad(self.args.input_file)
        trainer_module = {
            'scgpt': anno_scgpt_train,
            'scbert': anno_scbert_train,
            # 'gf':
        }
        if self.args.finetune:
            self.train(adata, trainer_module)
        else:
            loader = self.get_dataloader(adata, self.args.label_key, shuffle=False,
                                         ddp_train=self.args.distributed, drop_last=False)
            with open(f"{self.args.output_dir}/label2id.json", 'r') as f:
                label2id = json.load(f)
                label_num = len(label2id)
            model = self.init_model_for_finetune(label_num)
            true_labels = adata.obs[self.args.label_key]
            self.infer(model, loader, trainer_module, true_labels, label2id)


if __name__ == '__main__':
    config_file = '/home/share/huadjyin/home/s_qiuping1/workspace/BioLLM1/biollm/config/anno/scbert.toml'
    obj = ReplicateAnno(config_file)
    obj.run()
