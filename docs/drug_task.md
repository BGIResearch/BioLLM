## BioLLM: Drug Response Task

This document explains the implementation of the `DrugTask` class for drug annotation tasks. It covers the major steps involved in training and evaluating models using various available options like `scGPT`, `scBERT`, `scFoundation`, `Geneformer`, and others. The following sections describe the main features, functions, and usage of the `DrugTask` class in detail.

### Table of Contents
1. [Overview](#1-overview)
2. [Model Selection](#2-model-selection)
3. [How to Use](#3-how-to-use)
4. [Functionality Details](#4-functionality-details)
5. [Running the DrugTask](#5-running-the-drugtask)

### 1. Overview
The `DrugTask` is designed to facilitate drug annotation through data processing, model training, and evaluation. It supports multiple models including `scGPT`, `scBERT`, `scFoundation`, and `Geneformer`. The workflow includes:

- Preprocessing input data
- Performing model pretraining (if applicable)
- Extracting features
- Training the model
- Testing and evaluation

### 2. Model Selection
The implementation allows you to use several models to perform drug annotation. Here is a list of models available:

- **scGPT**: A generative pre-trained transformer specialized for single-cell data.
- **scBERT**: A BERT-like model for single-cell annotation.
- **scFoundation**: A foundational model for single-cell data analysis.
- **Geneformer**: A transformer model tailored for gene-level tasks.


To select a specific model, change the value of `model_select` in the `__main__` block of the code:

```python
model_select = 'scgpt'  # Options: 'scgpt', 'scbert', 'scfoundation', 'geneformer'
```

### 3. How to Use
To use the `DrugTask` class, follow these steps:

1. **Configure Model Selection**: Choose the appropriate model by setting the `model_select` variable.
2. **Run the Task**: Execute the script to start the training and evaluation process.

### 4. Functionality Details

#### Key Functions
- **`__init__`**: Initializes the `DrugTask` with the given configuration file, sets up necessary device information, and prepares initial settings.
- **`pretrain_inference()`**: Performs pretraining and generates embeddings for the gene expression features. Depending on the model, different preprocessing and embedding methods are applied.
- **`train()`**: Handles the training process for the selected model. Includes early stopping to prevent overfitting.
- **`test()`**: Evaluates the model on the validation dataset and calculates various metrics such as Pearson Correlation Coefficient (PCC) and Spearman rank correlation.
- **`run()`**: Main function that orchestrates data processing, model pretraining, training, and evaluation.

#### Pretraining and Training
- The pretraining stage involves processing the gene expression features (`gexpr_feature`) using the selected model. For some models like `Geneformer`, additional preprocessing steps are applied.
- During training, the model is trained for several epochs, and metrics are recorded to select the best-performing model.

#### Early Stopping
The training function implements early stopping if there is no improvement in PCC for `patience` consecutive epochs. This helps prevent overfitting and reduces computational costs.

### 5. Running the DrugTask
The `DrugTask` class can be run directly from the command line or integrated into a larger system. Use the following command:

```python
if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_select = None   # Set model_select to one of 'scgpt', 'scbert', 'scfoundation', 'geneformer'
        if model_select is not None:
            config_file = f'/path/to/config/{model_select}_drug.toml'
        else:
            config_file = '/path/to/default/config/drug.toml'
        obj = DrugTask(config_file)
        obj.run()
```

Replace `/path/to/config/` with the appropriate path to your configuration files.

### Example Configurations
Configurations for each model are available in the `biollm/config/drug/test/` directory. Adjust the configuration file to specify paths to the input and output data, model parameters, and other settings as required.

#### Notes
- Ensure the appropriate configuration file is provided.
- Customize paths to the configuration and data directories based on your environment.

### Full Code
#### Task Execution
```python
import json
import random
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
import gc
import csv
from biollm.base.bio_task import BioTask
from biollm.model.drug import PyTorchMultiSourceGCNModel
from biollm.task.drug.drug_data_process import DrugDataProcess

class DrugTask(BioTask):
    def __init__(self, config_file):
        super(DrugTask, self).__init__(config_file)
        self.config_file = config_file
        self.device = self.args.device
        self.leave_drug = 15

    def pretrain_inference(self, gexpr_feature):
        adata = anndata.AnnData(X=gexpr_feature)
        adata.write_h5ad('./exp_qp.h5ad')
        pretrain_model = self.load_model()
        if self.args.model_used != 'scfoundation' and self.args.model_used != 'geneformer':
            adata = self.load_obj.preprocess_adata(adata)
            obj.args['max_seq_len'] = adata.shape[1]
            if 'gene_name' not in adata.var:
                adata.var['gene_name'] = adata.var.index.values
            if 'celltype_id' not in adata.obs:
                adata.obs['celltype_id'] = 0
        if self.args.model_used == 'geneformer':
            adata.var["ensembl_id"] = adata.var_names.values
        gexpr_emb = self.load_obj.get_embedding(self.args.emb_type, adata)
        print('Embedding size is: ', gexpr_emb.shape)
        gexpr_emb = pd.DataFrame(gexpr_emb, index=gexpr_feature.index)
        return gexpr_emb

    # train
    def train(self, model, dataloader, validation_data, optimizer, nb_epoch):
        patience = 5
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
                loss_list.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ii % 500 == 0:
                    print(f"[INFO] epoch {epoch} batch {ii}:  loss {loss.item():.4f}  pcc {pcc:.4f}")

            loss_test, pcc_test, spearman_test = self.test(model, validation_data)  # loss pcc scc
            epoch_loss = sum(loss_list) / len(loss_list)
            torch.save(model.state_dict(), f'{self.args.save_path}/modelSave/{self.args.model_used}/{epoch}.pth')
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
                    model.load_state_dict(torch.load(f'{self.args.save_path}/modelSave/{self.args.model_used}/{best_epoch}.pth', map_location=self.device))
                    loss, pcc, scc = self.test(model, validation_data)
                    if self.args.leave_drug_test:
                        with open(f'{self.args.save_path}/data/leave_drug/{self.args.model_used}_leave.csv', 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([round(loss, 4), round(pcc, 4), round(scc, 4)])
                        print('Write to csv. Leave drug:', self.leave_drug)
                        print(f'loss {loss:.4f}   pcc {pcc:.4f}   scc {scc:.4f}')
                    else:
                        torch.save(model, f'{self.args.save_path}/modelSave/{self.args.model_used}/best_{self.args.model_used}_model.pt')
                        print('Model saved to: ', f'{self.args.save_path}/modelSave/{self.args.model_used}/best_{self.args.model_used}_model.pt')
                        with open(f'{self.args.save_path}/data/random_test/{self.args.model_used}_random_test.csv', 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([round(loss, 4), round(pcc, 4), round(scc, 4)])
                        print('Write to csv.')
                        print(f'loss {loss:.4f}   pcc {pcc:.4f}   scc {scc:.4f}')
                    break

    # Test
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
        drug_feature_file = f'{self.args.save_path}/drug_graph_feat'
        mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx = data_obj.MetadataGenerate(self.args.drug_info_file,self.args.cell_line_info_file,self.args.genomic_mutation_file,
                                                                                            drug_feature_file,self.args.gene_expression_file,self.args.methylation_file,False)
        # pretraining model
        if self.args.model_used != 'no':
            gexpr_feature = self.pretrain_inference(gexpr_feature)

        # train / test
        if self.args.leave_drug_test:
            print('Leave:', self.leave_drug)
            all_drug_ids = sorted(list(set(drug_feature.keys())))
            data_train_idx, data_test_idx = data_obj.DrugSplit(data_idx, all_drug_ids, self.leave_drug)
        else:
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

        if not self.args.leave_drug_test:
            test_data = [[X_drug_feat_data_test, X_drug_adj_data_test, X_mutation_data_test, X_gexpr_data_test,
                          X_methylation_data_test], Y_test, data_test_idx]  # test data
            torch.save(test_data, f'{self.args.save_path}/data/test_data/{self.args.model_used}_test_data.pth')
            print(f'Test data saved to {self.args.save_path}/data/test_data/{self.args.model_used}_test_data.pth')

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
            self.train(model, dataloader, validation_data, optimizer, nb_epoch=300)
        elif self.args.mode == 'test':
            model = torch.load(f'{self.args.save_path}/modelSave/{self.args.model_used}/best_{self.args.model_used}_model.pt')
            loss_test, pcc_test, spearman_test = self.test(model, validation_data)
            print(f'loss {loss_test: .4f}    pcc {pcc_test: .4f}    spearman {spearman_test: .4f}')
        if self.args.leave_drug_test:
            self.leave_drug += 1

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_select = None   # None, scgpt, scbert, scfoundation, geneformer
        # To use a specific model, set model_select to one of the following options:
        # 'scgpt', 'scbert', 'scfoundation', 'geneformer'.
        if model_select is not None:
            config_file = f'./config/drug/test/{model_select}_drug.toml'
        else:
            config_file = './config/drug/test/drug.toml'
        obj = DrugTask(config_file)
        obj.run()
```
#### Visualization
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel data
file_path = './drug.xlsx' # the PCC and SRCC score for each method
df = pd.read_excel(file_path)

# Set the model names as the index (first column)
df.set_index(df.columns[0], inplace=True)

# Calculate the average of PCC and SRCC for each model
df['Average'] = df[['PCC', 'SRCC']].mean(axis=1)

# Sort the dataframe by the average values in descending order
df = df.sort_values(by='Average', ascending=True)

# Plot the grouped bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Create indices for the models (y-axis)
indices = np.arange(len(df.index))

# Width of the bars
bar_width = 0.35

# Plot each category
bars1 = ax.barh(indices, df['PCC'], bar_width, label='Pearson', color="#D86830")
bars2 = ax.barh(indices + bar_width, df['SRCC'], bar_width, label='Spearman', color="#D86830", alpha=0.3)

# Add labels and titles
ax.set_xlabel('Correlation')
ax.set_yticks(indices + bar_width / 2)
ax.set_yticklabels(df.index)

# Add text labels on the bars
for bar in bars1:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.3f}', ha='left', va='center')

for bar in bars2:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.3f}', ha='left', va='center')

# Place the legend outside the plot
ax.legend(title='Variable', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
```
![img_4.png](img_4.png)