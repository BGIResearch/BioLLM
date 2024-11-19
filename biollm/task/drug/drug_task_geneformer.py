import json
import random, sys, os
import numpy as np
import pandas as pd
import time
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from scipy.stats import pearsonr, spearmanr
import anndata
import tempfile
sys.path.append('/home/share/huadjyin/home/s_qiuping1/hanyuxuan')   ####################
from biollm.base.bio_task import BioTask
from biollm.model.drug import PyTorchMultiSourceGCNModel
from biollm.task.drug.drug_data_process import DrugDataProcess


class DrugTask(BioTask):
    def __init__(self, config_file):
        super(DrugTask, self).__init__(config_file)
        self.config_file = config_file
        self.device = self.args.device
        self.vocab = self.load_obj.vocab
        self.TCGA_label_set = ["ALL", "BLCA", "BRCA", "CESC", "DLBC", "LIHC", "LUAD",
                          "ESCA", "GBM", "HNSC", "KIRC", "LAML", "LCML", "LGG",
                          "LUSC", "MESO", "MM", "NB", "OV", "PAAD", "SCLC", "SKCM",
                          "STAD", "THCA", 'COAD/READ']


    def geneformer_inference(self, gexpr_feature):
        adata = anndata.AnnData(X=gexpr_feature)
        geneformer_model = self.load_model()
        gexpr_emb = self.load_obj.get_embedding(self.args.emb_type, adata)
        print('Embedding size is: ', gexpr_emb.shape)
        gexpr_emb = pd.DataFrame(gexpr_emb, index=gexpr_feature.index)
        return gexpr_emb


    # train
    def train(self, model, dataloader, validation_data, optimizer, nb_epoch):
        patience = 10
        best_pcc = -np.Inf  # record best pcc
        best_epoch = 0  # record epoch of best pcc
        counter = 0
        # for every training epoch
        for epoch in range(0, nb_epoch):
            loss_list = []
            t = time.time()
            for ii, data_ in enumerate(dataloader):
                model.train()  # switch on batch normalization and dropout
                data_ = [dat.to(self.device) for dat in data_]
                X_drug_feat_data, X_drug_adj_data, X_mutation_data, X_gexpr_data, X_methylation_data, Y_train = data_
                output = model(X_drug_feat_data, X_drug_adj_data, X_mutation_data, X_gexpr_data,
                               X_methylation_data)  # calculate ouput
                output = output.squeeze(-1)
                loss = F.mse_loss(output, Y_train)
                pcc = torch.corrcoef(torch.stack((output, Y_train)))[0, 1]  # pcc
                # spear, _ = spearmanr(output.detach().cpu().numpy(), Y_train.detach().cpu().numpy())  # spearman
                loss_list.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ii % 500 == 0:
                    print(f"[INFO] epoch {epoch} batch {ii}:  loss {loss.item():.4f}  pcc {pcc:.4f}")

            loss_test, pcc_test, spearman_test = self.test(model, validation_data)  # loss pcc scc
            epoch_loss = sum(loss_list) / len(loss_list)
            torch.save(model.state_dict(), f'{self.args.save_path}/{self.args.model_used}/{epoch}.pth')
            print(f'[INFO] epoch {epoch}: epoch average loss {epoch_loss:.4f}')
            print(f'[INFO] validation data: loss {loss_test:.4f}    pcc {pcc_test:.4f} spearman {spearman_test:.4f}')
            print('=========================================================')
            # early stop
            if pcc_test > best_pcc:
                best_pcc = pcc_test
                best_epoch = epoch
                counter = 0
            else:
                counter += 1
                if counter >= patience:  # early stop
                    print(f'[Early stop] best epoch {best_epoch}   best pcc {best_pcc: .4f}')
                    model.load_state_dict(
                        torch.load(f'{self.args.save_path}/{self.args.model_used}/{best_epoch}.pth', map_location=self.device))
                    print(f'Load best model from {self.args.save_path}/{self.args.model_used}/{best_epoch}.pth')
                    loss, pcc, scc = self.test(model, validation_data)
                    print(f'loss {loss:.4f}   pcc {pcc:.4f}   scc {scc:.4f}')
                    break

    # 测试
    def test(self, model, validation_data):
        print('Testing...')
        model.eval()  # switch off batch normalization and dropout
        with torch.no_grad():
            validation_data[0] = [dat.to(self.device) for dat in validation_data[0]]
            validation_data[1] = validation_data[1].to(self.device)
            X_drug_feat_data_test, X_drug_adj_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test = validation_data[0]
            Y_test = validation_data[1]  # 验证集的标签
            test_data = TensorDataset(X_drug_feat_data_test, X_drug_adj_data_test, X_mutation_data_test,
                                      X_gexpr_data_test, X_methylation_data_test)
            test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
            # predict output with batch size 64
            output_test = torch.empty((0,), device=self.device)  # create an empty tensor
            for ii, data_ in enumerate(test_dataloader):
                X_drug_feat_data, X_drug_adj_data, X_mutation_data, X_gexpr_data, X_methylation_data = data_
                output = model(X_drug_feat_data, X_drug_adj_data, X_mutation_data, X_gexpr_data, X_methylation_data)
                output = output.squeeze(-1)
                output_test = torch.cat((output_test, output), dim=0)
            loss_test = F.mse_loss(output_test, Y_test)
            pcc = torch.corrcoef(torch.stack((output_test, Y_test)))[0, 1]  # pcc
            spearman_test, _ = spearmanr(output_test.detach().cpu().numpy(), Y_test.detach().cpu().numpy())  # spearman
        return loss_test.item(), pcc.item(), spearman_test

    def run(self):
        random.seed(0)
        # data_idx: (cell_line, drug, ln_IC50, cell_type)  example: ('ACH-000534', '9907093', 4.358842, 'DLBC')
        data_obj = DrugDataProcess(self.config_file)
        mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx = data_obj.MetadataGenerate(self.args.drug_info_file,self.args.cell_line_info_file,self.args.genomic_mutation_file,
                                                                                            self.args.drug_feature_file,self.args.gene_expression_file,self.args.methylation_file,False)
        # pretraining model
        gexpr_feature = self.geneformer_inference(gexpr_feature)

        # train / test
        data_train_idx, data_test_idx = data_obj.DataSplit(data_idx)

        # extract feature
        X_drug_data_train, X_mutation_data_train, X_gexpr_data_train, X_methylation_data_train, Y_train, cancer_type_train_list = data_obj.FeatureExtract(
            data_train_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature)
        X_drug_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test, Y_test, cancer_type_test_list = data_obj.FeatureExtract(
            data_test_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature)
        # extract feature matrix and adjacency matrix
        X_drug_feat_data_train = torch.stack([item[0] for item in X_drug_data_train])  # # nb_instance * Max_stom * feat_dim
        X_drug_adj_data_train = torch.stack([item[1] for item in X_drug_data_train])  # nb_instance * Max_stom * Max_stom
        # extract feature matrix and adjacency matrix
        X_drug_feat_data_test = torch.stack([item[0] for item in X_drug_data_test])  # 维度：nb_instance * Max_stom * feat_dim
        X_drug_adj_data_test = torch.stack([item[1] for item in X_drug_data_test])  # 维度：nb_instance * Max_stom * Max_stom

        # validation data
        validation_data = [[X_drug_feat_data_test, X_drug_adj_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test],Y_test]

        # dataLoader
        train_data = TensorDataset(X_drug_feat_data_train, X_drug_adj_data_train, X_mutation_data_train, X_gexpr_data_train,
                                   X_methylation_data_train, Y_train)
        dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

        # model initialization
        model = PyTorchMultiSourceGCNModel(drug_input_dim=X_drug_data_train[0][0].shape[-1], drug_hidden_dim=256, drug_concate_before_dim=100,
                                           mutation_input_dim=X_mutation_data_train.shape[-2], mutation_hidden_dim=256, mutation_concate_before_dim=100,
                                           gexpr_input_dim=X_gexpr_data_train.shape[-1], gexpr_hidden_dim=256, gexpr_concate_before_dim=100,
                                           methy_input_dim=X_methylation_data_train.shape[-1], methy_hidden_dim=256, methy_concate_before_dim=100,
                                           output_dim=300, units_list=self.args.unit_list, use_mut=self.args.use_mut,
                                           use_gexp=self.args.use_gexp, use_methy=self.args.use_methy,
                                           regr=True, use_relu=self.args.use_relu,
                                           use_bn=self.args.use_bn, use_GMP=self.args.use_GMP
                                           ).to(self.device)

        # GPU or CPU
        print('Device Is %s' % self.device)

        # model train / test
        optimizer = Adam(model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.0, amsgrad=False)
        if self.args.mode == 'train':
            print('Begin training...')
            self.train(model, dataloader, validation_data, optimizer, nb_epoch=100)
        if self.args.mode == 'test':
            # model.load_state_dict(torch.load(f'{self.args.save_path}/{self.args.model_used}/6.pth', map_location=self.device))
            # torch.save(model, f'{self.args.save_path}/{self.args.model_used}/best_geneformer_model.pt')
            model = torch.load(f'{self.args.save_path}/{self.args.model_used}/best_geneformer_model.pt')
            loss_test, pcc_test, spearman_test = self.test(model, validation_data)
            print(f'loss {loss_test: .4f}    pcc {pcc_test: .4f}    spearman {spearman_test: .4f}')



if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_file = '../../config/drug/geneformer_drug.toml'
        obj = DrugTask(config_file)
        obj.run()

