#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: scgpt.py
@time: 2025/1/15 11:06
"""
from biollm.tasks.bio_task import BioTask
import os
import scanpy as sc
import torch
from torch import nn
import pickle
from tqdm import tqdm


class ScgptImputation(BioTask):
    def __init__(self, config_file):
        super(ScgptImputation, self).__init__(config_file, load_model=False)
        self.check_parameters()
        # init the func for the trainer
        self.criterion = nn.MSELoss()
        self.is_master = int(os.environ['RANK']) == 0 if self.args.distributed else True
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.criterion = self.criterion.to(self.args.device)
        if self.is_master:
            self.logger.info(self.args)

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
        if 'gene_var_key' in self.args:
            adata.var_names = adata.var[self.args.gene_var_key]
        train_loader = self.load_obj.get_dataloader(adata,
                                                    self.args.do_preprocess,
                                                    sort_seq_batch=False,
                                                    do_split=False,
                                                    shuffle=False,
                                                    ddp_train=False,
                                                    drop_last=False)
        return train_loader

    def run(self):
        adata = sc.read_h5ad(self.args.input_file)
        if adata.shape[0] > 10000:
            adata = adata[0: 10000, :].copy()
        self.model = self.load_model()
        data_loader = self.make_dataloader(adata)
        self.model = self.model.to(self.args.device)
        self.model = self.model.eval()
        target_list = []
        pred_list = []
        mse_list = []
        epoch_losses = 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            for batch_data in tqdm(data_loader, desc='MVC Imputation'):
                input_gene_ids = batch_data["gene_ids"].to(self.args.device)
                input_values = batch_data["values"].to(self.args.device)
                src_key_padding_mask = input_gene_ids.eq(self.load_obj.vocab[self.args.pad_token])
                target_values = batch_data["target_values"].to(self.args.device)
                output = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    MVC=True
                )
                value_index = input_values.view(-1) == self.args.mask_value
                target_values = target_values.view(-1)[value_index]
                pred_values = output['mvc_output'].view(-1)[value_index]
                pred_list.extend(list(pred_values.cpu().numpy()))
                target_list.extend(list(target_values.cpu().numpy()))
                mlm_loss = self.criterion(pred_values, target_values)
                epoch_losses += mlm_loss.item()
                mse_list.append(mlm_loss.item())
            mlm_mean = epoch_losses / len(data_loader)
            self.logger.info(f"Mean of MSE: {mlm_mean}")
            with open(self.args.output_dir + '/imputation_res.pk', 'wb') as fd:
                pickle.dump({'pred': pred_list, 'target': target_list, 'mse_mean': mlm_mean, 'mse_list': mse_list}, fd)

        return mlm_mean


if __name__ == "__main__":
    # config_file = sys.argv[1]
    config_file = '../../config/imputation/SCT000000213.toml'
    obj = ScgptImputation(config_file)
    obj.run()
