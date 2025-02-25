# BioLLM: Cell Embedding Generation from scFM Models (Zero-Shot)  Task

This documentation explains how to generate cell embeddings from various single-cell foundational models (scFMs) using BioLLM. The models supported include **scGPT**, **Geneformer**, **scFoundation**, and **scBERT**. The following sections describe the setup, configuration, and procedures for generating cell embeddings from each model.

## Prerequisites

Before proceeding, ensure that you have the following installed:

- Python 3.6+
- Required dependencies (use `pip install -r requirements.txt`)

## Overview of Models

BioLLM supports four major foundational models for generating cell embeddings from single-cell RNA-seq data:

- **scGPT**
- **Geneformer**
- **scFoundation**
- **scBERT**

For each model, preprocessing steps and configuration files are provided to ensure the best performance. Below are the details on how to use each model for zero-shot generation of cell embeddings.

## Preprocessing and Embedding Generation

For Geneformer and scGPT, which support input sequence lengths of 2048 and 1200 respectively, we selected 3000 highly variable genes as input features. The other two foundational models, **scBERT** and **scFoundation**, utilized full-length gene sequences without feature selection. 

### Preprocessing Details:
- **scGPT** and **Geneformer**: Input features are selected from the 3000 highly variable genes.
- **scBERT** and **scFoundation**: All gene sequences are used, without feature selection.

The preprocessing steps align with each model's pretraining conditions to ensure optimal performance.

### Normalization:
- **scGPT**, **scBERT**, and **scFoundation**: Log1p transformation of the gene expression data is required.
- **Geneformer**: Raw counts are used without normalization.

### Embedding Generation Methods:
- **scGPT**: The cell embedding is derived from the CLS token embedding.
- **Geneformer**: Embeddings are extracted using the `mean_nonpadding_embs` function, which computes the mean of non-padded token embeddings.
- **scFoundation**: Final cell embeddings are generated through max pooling applied to the token embeddings.
- **scBERT**: Three methods are available for generating cell embeddings: CLS, mean, and sum. The CLS method uses the CLS token embedding, while mean and sum pooling methods aggregate the token embeddings produced by the model's encoder.

This ensures that each model's strengths are effectively leveraged for generating accurate cellular representations for downstream analysis.

## Usage Instructions

### 1. **scGPT:**

```python
from biollm.utils.utils import load_config
import scanpy as sc
import pandas as pd
import numpy as np
from biollm.base.load_scfoundation import LoadScfoundation
import os
import pickle
from biollm.base.load_scgpt import LoadScgpt


def scgpt(adata, output_dir):
    config_file = './configs/zero_shots/scgpt_cell_emb.toml'
    configs = load_config(config_file)
    obj = LoadScgpt(configs)
    adata = adata[:, adata.var_names.isin(obj.get_gene2idx().keys())].copy()
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(configs.emb_type, adata=adata)
    print('embedding shape:', emb.shape)
    scg_cell_emb = pd.DataFrame(emb, index=adata.obs_names)
    cell_emb_file = os.path.join(output_dir, "scg_cell_emb.pkl")
    with open(cell_emb_file, 'wb') as file:
        pickle.dump(np.array(scg_cell_emb), file)

data_path = './liver.h5ad'
output_dir = './output'
adata = sc.read_h5ad(data_path)
scgpt(adata, output_dir)
```

### 2. **Geneformer:**

```python
from biollm.utils.utils import load_config
import scanpy as sc
import pandas as pd
import numpy as np
from biollm.base.load_geneformer import LoadGeneformer
import os
import pickle


def geneformer(adata, output_dir):
    config_file = './configs/zero_shots/geneformer_cell_emb.toml'
    configs = load_config(config_file)
    obj = LoadGeneformer(configs)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(obj.args.emb_type, adata=adata)
    print('embedding shape:', emb.shape)
    gf_cell_emb = pd.DataFrame(emb, index=adata.obs_names)
    cell_emb_file = os.path.join(output_dir, "gf_cell_emb.pkl")
    with open(cell_emb_file, 'wb') as file:
        pickle.dump(np.array(gf_cell_emb), file)

data_path = './liver.h5ad'
output_dir = './output'
adata = sc.read_h5ad(data_path)
geneformer(adata, output_dir)
```

### 3. **scFoundation:**

```python
from biollm.utils.utils import load_config
import scanpy as sc
import pandas as pd
import numpy as np
from biollm.base.load_scfoundation import LoadScfoundation
import os
import pickle


def scfoundation(adata, output_dir):
    config_file = './configs/zero_shots/scfoundation_cell_emb.toml'
    configs = load_config(config_file)
    obj = LoadScfoundation(configs)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(configs.emb_type, gene_ids=None, adata=adata)
    print('embedding shape:', emb.shape)
    scf_cell_emb = pd.DataFrame(emb, index=adata.obs_names)

    cell_emb_file = os.path.join(output_dir, "scf_cell_emb.pkl")
    with open(cell_emb_file, 'wb') as file:
        pickle.dump(np.array(scf_cell_emb), file)

data_path = './liver.h5ad'
output_dir = './output'
adata = sc.read_h5ad(data_path)
scfoundation(adata, output_dir)
```

### 4. **scBERT:**

```python
from biollm.utils.utils import load_config
import scanpy as sc
import pandas as pd
import numpy as np
from biollm.base.load_scbert import LoadScbert
import os
import pickle


def scbert(adata, output_dir):
    config_file = './configs/zero_shots/scbert_cell_emb.toml'
    configs = load_config(config_file)
    obj = LoadScbert(configs)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(configs.emb_type, adata=adata)
    print('embedding shape:', emb.shape)
    scb_cell_emb = pd.DataFrame(emb, index=adata.obs_names)
    cell_emb_file = os.path.join(output_dir, "scbert_cell_emb.pkl")
    with open(cell_emb_file, 'wb') as file:
        pickle.dump(np.array(scb_cell_emb), file)

data_path = './liver.h5ad'
output_dir = './output'
adata = sc.read_h5ad(data_path)
scbert(adata, output_dir)
```

## Config Directory

The configuration files for each model can be found in the `biollm/docs/` directory. Ensure that the path and parameters are adjusted based on your dataset.

## Conclusion

This guide demonstrates how to use BioLLM to generate cell embeddings from various foundational models. The preprocessing steps are tailored to each model’s requirements to ensure optimal performance. By following the instructions and adjusting the configurations, you can effectively generate high-quality cell embeddings for downstream analysis.
