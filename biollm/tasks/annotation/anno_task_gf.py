#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :anno_task_gf.py
# @Time      :2024/4/10 13:00
# @Author    :Luni Hu
import torch
import pickle
import os
import scanpy as sc
from biollm.tasks.bio_task import BioTask
from transformers import Trainer
from transformers.training_args import TrainingArguments
from biollm.repo.geneformer.collator_for_classification import DataCollatorForCellClassification
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
import pickle as pkl
from torch.utils.data import DataLoader
import time


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)

    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
      'accuracy': acc,
      'macro_f1': macro_f1
    }


class AnnoTask(BioTask):
    def __init__(self, cfs_file, out_dir=None, data_path=None):

        super(AnnoTask, self).__init__(cfs_file, data_path)

        self.out_dir = out_dir if out_dir is not None else self.args.output_dir
        self.data_path = data_path if data_path is not None else self.args.input_file
        self.label_dict = None
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

    def prepare_data(self, cell_type_key="celltype", nproc=16, train_test_split=0.8):

        adata = sc.read_h5ad(self.data_path)
        # adata = self.read_h5ad(self.data_path, preprocess=True)
        add_length = True if self.args.finetune else False
        # if 'ensembl_id' not in adata.var.columns:
        #     adata.var['ensembl_id'] = adata.var[self.args.gene_ensembl_key]
        tokenized_dataset = self.load_obj.load_data(
            adata=adata, cell_type_key=cell_type_key, nproc=nproc, add_length=add_length)
        def classes_to_ids(example):
            example["label"] = target_dict[example["label"]]
            return example
        def if_trained_label(example):
            return example["label"] in trained_labels
        if self.args.finetune:
            ids = torch.tensor([i for i in range(len(tokenized_dataset))])
            # Convert the ids tensor to a list
            ids_list = ids.tolist()
            # Add a new column to store the ids
            tokenized_dataset = tokenized_dataset.add_column("id", ids_list)
            tokenized_dataset = tokenized_dataset.shuffle(seed=42)
            tokenized_dataset = tokenized_dataset.rename_column(self.args.label_key, "label")
            target_names = list(Counter(tokenized_dataset["label"]).keys())
            self.label_dict = target_names
            target_dict = dict(zip(target_names, [i for i in range(len(target_names))]))
            labeled_trainset = tokenized_dataset.map(classes_to_ids, num_proc=nproc)

            # create train/eval splits
            labeled_train_split = labeled_trainset.select([i for i in range(0, round(len(labeled_trainset)*train_test_split))])
            labeled_eval_split = labeled_trainset.select(
                [i for i in range(round(len(labeled_trainset)*train_test_split), len(labeled_trainset))])

            # filter dataset for cell types in corresponding training set
            trained_labels = list(Counter(labeled_train_split["label"]).keys())
            labeled_eval_split = labeled_eval_split.filter(if_trained_label, num_proc=nproc)
            return labeled_train_split, labeled_eval_split
        else:
            return tokenized_dataset

    def classifier(self,  max_lr=5e-5, lr_schedule_fn="linear", warmup_steps=500,
                   epochs=30, train_set=None, eval_set=None, geneformer_batch_size=8):

        if train_set is None and eval_set is None:
            train_set, eval_set = self.prepare_data(cell_type_key=self.args.label_key)

        # set logging steps
        logging_steps = round(len(train_set) / geneformer_batch_size / 10)

        training_args = {
            "learning_rate": max_lr,
            "do_train": True,
            "do_eval": True,
            "logging_steps": logging_steps,
            "group_by_length": True,
            "prediction_loss_only": True,
            "evaluation_strategy": "epoch",
            "length_column_name": "length",
            "disable_tqdm": False,
            "gradient_checkpointing": True,
            "per_device_train_batch_size": geneformer_batch_size,
            "per_device_eval_batch_size": geneformer_batch_size,
            "fp16": True,
            "save_total_limit":1,
            "lr_scheduler_type": lr_schedule_fn,
            "save_strategy": "epoch",
            "warmup_steps": warmup_steps,
            "weight_decay": 0.001,
            "num_train_epochs": epochs,
            "load_best_model_at_end": True,
            "output_dir": self.out_dir
        }

        training_args_init = TrainingArguments(**training_args)

        # create the trainer
        trainer = Trainer(
            model=self.model,
            args=training_args_init,
            data_collator=DataCollatorForCellClassification(),
            train_dataset=train_set,
            eval_dataset=eval_set,
            compute_metrics=compute_metrics
        )

        return trainer

    def train(self):

        trainer = self.classifier(
            epochs=self.args.epoches,
            geneformer_batch_size=self.args.batch_size,
            max_lr=self.args.lr if 'lr' in self.args else 5e-5
        )
        if self.label_dict is not None:
            with open(f'{self.args.output_dir}/label_dict.pk', 'wb') as fp:
                pkl.dump(self.label_dict, fp)
        trainer.train()
        return trainer

    def predict(self, eval_set=None):
        if eval_set is None:
            eval_set = self.prepare_data(cell_type_key=None)

        test_loader = DataLoader(eval_set,
                                 batch_size=self.args.batch_size,
                                 collate_fn=DataCollatorForCellClassification(),
                                 shuffle=False)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for index, inputs in enumerate(test_loader):
                for i in inputs:
                    inputs[i] = inputs[i].to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds = logits.argmax(1)
                predictions.append(preds)
            predictions = torch.cat(predictions, dim=0)
            predictions = predictions.detach().cpu().numpy()
        with open(f'{self.args.output_dir}/label_dict.pk', 'rb') as fp:
            label_list = pkl.load(fp)
            label_dict = dict([(i, label_list[i]) for i in range(len(label_list))])
        predicted_label = [label_dict[i] for i in predictions]
        with open(self.args.output_dir + '/predict_list.pk', 'wb') as w:
            pickle.dump(predicted_label, w)

    def run(self):
        t1 = time.time()
        if self.args.finetune:
            self.train()
        else:
            self.predict()
        t2 = time.time()
        if self.is_master:
            self.logger.info(f'Total time: {t2-t1} s.')
