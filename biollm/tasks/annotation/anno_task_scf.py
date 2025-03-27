#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :anno_task_scf.py
# @Time      :2024/3/4 14:46
# @Author    :Luni Hu
import time

import numpy as np
import pickle

import torch.cuda.amp

from biollm.tasks.bio_task import BioTask
from torch.utils.data import TensorDataset
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from biollm.algorithm.annotation import *
import torch.distributed as dist
import os
from biollm.utils.utils import get_reduced, distributed_concat, cal_model_params



class AnnoTaskScf(BioTask):
    def __init__(self, cfs_file, random_state=42, frozen=True):
        super(AnnoTaskScf, self).__init__(cfs_file)
        self.random_state = random_state
        self.frozen = frozen
        self.model = self.classifier()
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
        self.args.local_rank = local_rank
        self.world_size = torch.distributed.get_world_size() if self.args.distributed else 1
        self.args['world_size'] = self.world_size
        if self.is_master:
            self.logger.info(self.args)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(local_rank)
        self.scaler = torch.cuda.amp.GradScaler()

    def split_adata(self, adata):
        train_idx, test_idx = train_test_split(range(adata.shape[0]),
                                               test_size=0.2,
                                               random_state=self.random_state)
        adata_train = adata[train_idx].copy()
        adata_test = adata[test_idx].copy()
        return adata_train, adata_test

    def prepare_data(self, adata_train, label_dict, label_key=None):

        array_train = self.load_obj.load_data(adata_train, max_none_zore=self.args.max_none_zero_num)
        if label_key:
            label_train = [label_dict.get(key, len(label_dict)) for key in adata_train.obs[label_key]]
            dataset_train = {"x": array_train, "targets": label_train}
        else:
            dataset_train = {"x": array_train}
        return dataset_train

    def classifier(self):
        model = LinearProbingClassifier(self.load_obj.model, self.load_obj.config, frozenmore=self.frozen)
        return model


    def train(self):
        adata = self.read_h5ad(self.args.input_file, preprocess=True)
        adata_train, adata_eval = self.split_adata(adata)
        celltype = adata_train.obs[self.args.label_key].unique().tolist()
        with open(f'{self.args.output_dir}/label_dict.pk', 'wb') as fp:
            pickle.dump(celltype, fp)
        label_dict = {celltype[i]: i for i in range(len(celltype))}
        dataset_train = self.prepare_data(adata_train, label_dict, self.args.label_key)
        dataset_eval = self.prepare_data(adata_eval, label_dict, self.args.label_key)
        num_classes = len(celltype)
        dataset_train_size = dataset_train["x"].shape[0]
        print(f"training dataset size:{dataset_train_size}; num_class:{num_classes}")

        self.model.build(num_classes=num_classes)
        self.model = self.model.to(self.args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        dataset_train = TensorDataset(torch.tensor(np.array(dataset_train["x"]), dtype=torch.float32),
                                      torch.tensor(dataset_train["targets"]).long())
        dataset_eval = TensorDataset(torch.tensor(np.array(dataset_eval["x"])),
                                      torch.tensor(dataset_eval["targets"]).long())
        train_loader = self.load_obj.get_dataloader(dataset_train,
                                                    self.args.batch_size,
                                                    shuffle=True,
                                                    ddp_train=self.args.distributed,
                                                    drop_last=True)
        eval_loader = self.load_obj.get_dataloader(dataset_eval,
                                                   self.args.batch_size,
                                                   shuffle=False,
                                                   ddp_train=self.args.distributed,
                                                   drop_last=True)
        best_val_loss = float("inf")
        best_model = None
        params = cal_model_params(self.model)
        print('all params: ', params)
        for epoch in range(self.args.epochs):
            epoch_loss = 0.0
            epoch_acc = 0
            num_batches = len(train_loader)
            if self.args.distributed:
                train_loader.sampler.set_epoch(epoch)
            self.model.train()
            for batch, (inputs, targets) in enumerate(train_loader):
                x = inputs  # (B, L)
                value_labels = x > 0
                #
                x, x_padding = gatherData(x, value_labels, self.load_obj.config['pad_token_id'])
                data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
                position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                                  self.load_obj.config['pad_token_id'])

                x = x.to(self.args.device)
                x_padding = x_padding.to(self.args.device)
                position_gene_ids = position_gene_ids.to(self.args.device)
                targets = targets.to(self.args.device)
                with torch.cuda.amp.autocast():
                    logits = self.model(x, position_gene_ids, x_padding)
                    loss = self.criterion(logits, targets)

                    # Accumulate the loss for monitoring the training progress
                    epoch_loss += loss.item()
                    epoch_acc += (logits.argmax(1) == targets).sum().item() / targets.size(0)
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            if self.args.distributed:
                epoch_loss = get_reduced(epoch_loss, self.args.device, 0)
                epoch_acc = get_reduced(epoch_acc, self.args.device, 0)

            if self.is_master:
                self.logger.info(f"Epoch {epoch}: train/loss: {epoch_loss / num_batches}, train/acc: {epoch_acc / num_batches}")
                if self.wandb:
                    self.wandb.log({
                        "train/loss": epoch_loss / num_batches,
                        "train/acc": epoch_acc / num_batches,
                    })

            if self.args.distributed:
                dist.barrier()
            val_loss = self.evaluate(eval_loader, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                if self.is_master:
                    self.logger.info(f"Best model with score {best_val_loss:5.4f}")
            if self.is_master:
                torch.save(
                    self.model.state_dict(),
                    f"{self.args.output_dir}/model_{epoch}.pt",
                )
        return best_model

    def predict(self):
        adata = self.read_h5ad(self.args.input_file, preprocess=True)
        with open(f'{self.args.output_dir}/label_dict.pk', 'rb') as fp:
            celltype = pickle.load(fp)
        label_dict = {celltype[i]: i for i in range(len(celltype))}
        dataset_test = self.prepare_data(adata, label_dict)
        dataset_test = TensorDataset(torch.tensor(np.array(dataset_test["x"])))
        # Create a DataLoader for the test dataset
        test_dataloader = self.load_obj.get_dataloader(dataset_test,
                                                       self.args.batch_size,
                                                       shuffle=False,
                                                       ddp_train=False,
                                                       drop_last=False)
        # Set the classifier to evaluation mode
        with open(f'{self.args.output_dir}/label_dict.pk', 'rb') as fp:
            label_list = pickle.load(fp)
            label_dict = dict([(i, label_list[i]) for i in range(len(label_list))])
        self.model.build(len(label_list))
        self.model = self.load_obj.load_pretrain_model(self.args.finetune_model, self.model)
        self.model = self.model.to(self.args.device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for _, (inputs) in enumerate(test_dataloader):
                # Move the batch data to the appropriate device
                x = inputs[0]
                value_labels = x > 0
                #
                x, x_padding = gatherData(x, value_labels, self.load_obj.config['pad_token_id'])
                data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
                position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                                  self.load_obj.config['pad_token_id'])

                x = x.to(self.args.device)
                x_padding = x_padding.to(self.args.device)
                position_gene_ids = position_gene_ids.to(self.args.device)
                with torch.cuda.amp.autocast():
                    logits = self.model(x, position_gene_ids, x_padding)
                    # Get the predicted labels
                    batch_predictions = logits.argmax(1)
                    # Append the predictions to the list
                    predictions.extend(batch_predictions.cpu().numpy())

        predicted_label = [label_dict[i] for i in predictions]
        model_file_name = os.path.basename(self.args.finetune_model)
        with open(self.args.output_dir + f'/predict_list.pk.{model_file_name}', 'wb') as w:
            pickle.dump(predicted_label, w)

    def evaluate(self, loader, epoch):
        self.model.eval()
        total_loss = 0.0
        predictions = []
        trues = []
        num_batches = len(loader)
        with torch.no_grad():
            for batch, (inputs, targets) in enumerate(loader):
                x = inputs  # (B, L)
                value_labels = x > 0
                x, x_padding = gatherData(x, value_labels, self.load_obj.config['pad_token_id'])
                data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
                position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                                  self.load_obj.config['pad_token_id'])
                x = x.to(self.args.device)
                x_padding = x_padding.to(self.args.device)
                position_gene_ids = position_gene_ids.to(self.args.device)
                # inputs = inputs.to(self.device)
                targets = targets.to(self.args.device)
                with torch.cuda.amp.autocast():
                    logits = self.model(x, position_gene_ids, x_padding)
                    loss = self.criterion(logits, targets)
                    total_loss += loss.item()
                    preds = logits.argmax(dim=-1)
                predictions.append(preds)
                trues.append(targets)
            if self.args.distributed:
                predictions = distributed_concat(torch.cat(predictions, dim=0), len(loader.dataset), self.args.world_size)
                trues = distributed_concat(torch.cat(trues, dim=0), len(loader.dataset), self.args.world_size)
            else:
                predictions = torch.cat(predictions, dim=0)
                trues = torch.cat(trues, dim=0)
        predictions = predictions.cpu().numpy()
        trues = trues.cpu().numpy()
        accuracy = accuracy_score(trues, predictions)
        f1 = f1_score(trues, predictions, average='macro')
        if self.args.distributed:
            accuracy = get_reduced(accuracy, self.args.device, 0)
            f1 = get_reduced(f1, self.args.device, 0)
        if self.is_master:
            self.logger.info(
                f"Epoch {epoch}: eval/loss: {total_loss / num_batches:5.4f}, "
                f"eval/acc: {accuracy:5.3f}, f1-score: {f1:5.3f}")
            if self.wandb:
                self.wandb.log({
                    'eval/loss': total_loss / num_batches,
                    'eval/acc': accuracy,
                    'eval/f1': f1
                })
        return total_loss / num_batches

    def run(self):
        t1 = time.time()
        if self.args.finetune:
            best_model = self.train()
            if self.is_master:
                torch.save(
                    best_model.state_dict(),
                    f"{self.args.output_dir}/model_best.pt",
                )
        else:
            self.predict()
        t2 = time.time()
        if self.is_master:
            self.logger.info(f'Total time: {t2 - t1} s.')
