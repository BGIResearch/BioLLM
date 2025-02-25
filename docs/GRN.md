#  BioLLM: Gene Regulatory Network (GRN) Task

This document outlines the process of conducting Gene Regulatory Network (GRN) analysis using foundational models integrated into the **BioLLM** framework. The models used in this analysis include **Geneformer**, **scBERT**, **scGPT**, and **scFoundation**.

## Preprocessing and Input Dataset

The **Immune_ALL_human** dataset was used as the input for the GRN analysis. The following preprocessing steps were applied to ensure data quality and relevance:

1. **Gene Filtering**: Genes with fewer than 3 counts were excluded.
2. **Feature Selection**: The dataset was subset to the top 1200 highly variable genes.

Once preprocessed, the dataset was passed through the foundational models to generate **gene embeddings**.

## Embedding Generation and Network Construction

For each foundational model, embeddings are generated based on the processed dataset. After the embeddings are computed, Euclidean distance is calculated between gene pairs to construct an **adjacency matrix**. This matrix represents the relationships between genes, with each element indicating the degree of similarity between gene pairs. The adjacency matrix is then used to infer potential regulatory relationships, forming the basis of the **Gene Regulatory Network** (GRN), where nodes represent genes and edges represent regulatory connections.

## GRN Analysis with Different Foundational Models

Below are the steps and code examples for generating gene embeddings using **Geneformer**, **scBERT**, **scGPT**, and **scFoundation**.

### 1. **Geneformer**:

```python
from biollm.utils.utils import load_config
from biollm.base.load_geneformer import LoadGeneformer
import pickle as pkl
import os
import scanpy as sc

config_file = './configs/zero_shots/geneformer_gene-expression_emb.toml'
configs = load_config(config_file)

obj = LoadGeneformer(configs)
print(obj.args)
adata = sc.read_h5ad(configs.input_file)

obj.model = obj.model.to(configs.device)
print(obj.model.device)
emb = obj.get_embedding(obj.args.emb_type, adata=adata)
print('embedding shape:', emb.shape)
if not os.path.exists(configs.output_dir):
    os.makedirs(configs.output_dir, exist_ok=True)
with open(obj.args.output_dir + f'/geneformer_{obj.args.emb_type}_emb.pk', 'wb') as w:
    res = {'gene_names': list(obj.get_gene2idx().keys()), 'gene_emb': emb}
    pkl.dump(emb, w)
```

### 2. **scBERT**:

```python
from biollm.utils.utils import load_config
import numpy as np
from biollm.base.load_scbert import LoadScbert
import torch
import pickle as pkl
import os
import scanpy as sc

config_file = './configs/zero_shots/scbert_gene-expression_emb.toml'
configs = load_config(config_file)
obj = LoadScbert(configs)
print(obj.args)

gene_ids = list(obj.get_gene2idx().values())
gene_ids = np.array(gene_ids)
gene_ids = torch.tensor(gene_ids, dtype=torch.long).to(configs.device)
obj.model = obj.model.to(configs.device)
adata = sc.read_h5ad(configs.input_file)
emb = obj.get_embedding(configs.emb_type, adata=adata)
print('embedding shape:', emb.shape)
if not os.path.exists(configs.output_dir):
    os.makedirs(configs.output_dir, exist_ok=True)
with open(obj.args.output_dir + f'/scbert_gene-expression_emb.pk', 'wb') as w:
    res = {'gene_names': list(obj.get_gene2idx().keys()), 'gene_emb': emb}
    pkl.dump(emb, w)
```

### 3. **scGPT**:

```python
from biollm.utils.utils import load_config
from biollm.base.load_scgpt import LoadScgpt
import pickle as pkl
import os
import scanpy as sc

config_file = './configs/zero_shots/scgpt_gene-expression_emb.toml'
configs = load_config(config_file)
adata = sc.read_h5ad(configs.input_file)
obj = LoadScgpt(configs)
adata, _ = obj.filter_gene(adata)
configs.max_seq_len = adata.var.shape[0] + 1
obj = LoadScgpt(configs)
print(obj.args)

obj.model = obj.model.to(configs.device)

emb = obj.get_embedding(configs.emb_type, adata=adata)
print('embedding shape:', emb.shape)
if not os.path.exists(configs.output_dir):
    os.makedirs(configs.output_dir, exist_ok=True)
with open(obj.args.output_dir + f'/scgpt_{obj.args.emb_type}_emb.pk', 'wb') as w:
    res = {'gene_names': list(adata.var['gene_name']), 'gene_emb': emb}
    pkl.dump(emb, w)
```

### 4. **scFoundation**:

```python
from biollm.utils.utils import load_config
from biollm.base.load_scfoundation import LoadScfoundation
import pickle as pkl
import os
import scanpy as sc

config_file = './configs/zero_shots/scfoundation_gene-expression_emb.toml'
configs = load_config(config_file)

obj = LoadScfoundation(configs)
print(obj.args)

adata = sc.read_h5ad(configs.input_file)
adata = adata[:1000, :]
obj.model = obj.model.to(configs.device)
emb = obj.get_embedding(configs.emb_type, gene_ids=None, adata=adata)
print('embedding shape:', emb.shape)
if not os.path.exists(configs.output_dir):
    os.makedirs(configs.output_dir, exist_ok=True)
with open(obj.args.output_dir + f'/scfoundation_{obj.args.emb_type}_emb.pk', 'wb') as w:
    res = {'gene_names': list(obj.get_gene2idx().keys()), 'gene_emb': emb}
    pkl.dump(emb, w)
```

## Configurations

The configuration files can be found in the `biollm/docs/` directory. Ensure that paths, dataset details, and parameters are set correctly before running the code.

## Conclusion

This guide demonstrated how to conduct **Gene Regulatory Network (GRN)** analysis using various foundational models within the BioLLM framework. The process involves generating gene embeddings from a preprocessed dataset, calculating Euclidean distances between genes, and constructing an adjacency matrix that serves as the foundation for building the GRN. 

By using the different models, such as **Geneformer**, **scBERT**, **scGPT**, and **scFoundation**, you can analyze gene interactions and construct an inferred regulatory network to explore gene relationships within a given biological context.
