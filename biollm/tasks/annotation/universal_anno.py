#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: universal_anno.py
@time: 2025/3/5 15:34
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


class UniversalAnno(BioTask):
    def __init__(self, config_file):
        super(UniversalAnno, self).__init__(config_file)
        self.logger.info(self.args)

        # get cell embedding
        adata = sc.read_h5ad(self.args.input_file)
        self.cell_emb = self.llm_embedding(emb_type=self.args.emb_type, adata=adata)

        # init expert model
        if self.args.finetune:
            label = np.unique(np.array(adata.obs[self.args.label_key]))
            self.label_dict = {label[i]: i for i in range(len(label))}
            self.y_true = np.array([self.label_dict[i] for i in adata.obs[self.args.label_key]])
            with open(f'{self.args.output_dir}/label_dict.json', 'w') as fp:
                json.dump(self.label_dict, fp)
        else:
            with open(f'{self.args.output_dir}/label_dict.json', 'r') as fp:
                self.label_dict = json.load(fp)
        self.expert_model = ClsDecoder(d_model=self.cell_emb.shape[1], nlayers=self.args.cls_nlayers,
                                       n_cls=len(self.label_dict))

        del adata
        gc.collect()
        # init trainer
        self.trainer = Trainer(self.expert_model, lr=self.args.lr, batch_size=self.args.batch_size,
                               device=self.args.device, save_path=f"{self.args.output_dir}/best_model.pth")

    def run(self):
        t1 = time.time()
        # for the finetune setting
        if self.args.finetune:
            x_train, y_train, x_val, y_val, x_test, y_test = split_data(self.cell_emb, self.y_true, train_ratio=0.8,
                                                                        val_ratio=0.1, test_ratio=0.1, random_state=42)
            self.trainer.train(x_train, y_train, X_val=x_val, y_val=y_val, epochs=self.args.epochs)
            pred_test, _ = self.trainer.infer(x_test)
            res = compute_metrics(y_test, pred_test)
            accuracy, f1, recall, precision = res['accuracy'], res['macro_f1'], res['recall'], res['precision']
            print(f"Test acc: {accuracy:.4f}, f1: {f1:.4f}, "
                  f"recall: {recall:.4f}, precision: {precision:.4f}")
            return res
        else:
            predictions, _ = self.trainer.infer(self.cell_emb)
            predicted_label = [self.label_dict[i] for i in predictions]
            with open(self.args.output_dir + f'/predict_list.pk', 'wb') as w:
                pickle.dump(predicted_label, w)
        t2 = time.time()
        if self.is_master:
            self.logger.info(f'Total time: {t2 - t1} s.')


if __name__ == "__main__":
    import sys

    # config_file = sys.argv[1]
    config_file = '/home/share/huadjyin/home/s_qiuping1/workspace/BioLLM2/biollm/config/anno/scfoundation.toml'
    obj = UniversalAnno(config_file)
    obj.run()
