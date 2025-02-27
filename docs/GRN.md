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

Below are the steps and code examples for generating gene embeddings using **Geneformer**, **scBERT**, **scGPT**, and **scFoundation** and conducting GRN analysis.

### 1. Get the gene expression embedding from scFM models (zero-shot)

#### Geneformer:

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

#### scBERT:

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

#### scGPT:

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

#### scFoundation:

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

Note: You can find the configs in biollm/docs.

### 2. Evaluation

```python
import pandas as pd
import gseapy as gp
import scanpy as sc
import numpy as np
import pickle
from biollm.model.grn import GeneEmbedding


adata = sc.read_h5ad("../data/finetune/grn/Immune_ALL_human.h5ad") # dataset for evaluation
gene_emb_paths = {
        'scbert': '../case/result/zero-shot/gene_exp_emb/Immune_ALL_human/scb_gene_emb.pkl',
        'scgpt': '../case/result/zero-shot/gene_exp_emb/Immune_ALL_human/scg_gene_emb.pkl',
        'scfoundation': '../case/result/zero-shot/gene_exp_emb/Immune_ALL_human/scf_gene_emb.pkl',
        'geneformer': '../case/result/zero-shot/gene_exp_emb/Immune_ALL_human/gf_gene_emb.pkl'
    }
gene_name_paths = {
        'scbert': '../case/result/zero-shot/gene_exp_emb/Immune_ALL_human/scb_gene_names.pkl',
        'scgpt': '../case/result/zero-shot/gene_exp_emb/Immune_ALL_human/scg_gene_names.pkl',
        'scfoundation': '../case/result/zero-shot/gene_exp_emb/Immune_ALL_human/scf_gene_names.pkl',
        'geneformer': '../case/result/zero-shot/gene_exp_emb/Immune_ALL_human/gf_gene_names.pkl'
    }

resolutions = np.arange(0.1, 1.1, 0.1)
for model_name, emb_path in gene_emb_paths.items():
    with open(emb_path, 'rb') as f:
        emb = pickle.load(f)
    with open(gene_name_paths[model_name], 'rb') as f:
        gene_names = pickle.load(f)

    print(f"Processing model: {model_name}")
    # get gene list
    genes = list(adata.var.index)
    # get gene expression embeddings from models
    model_emb = {gene: emb[gene_names.index(gene)] for gene in gene_names if gene in genes}
    print(f'obtains the embeddings of {len(model_emb)}/{len(genes)} genes')
    embed = GeneEmbedding(model_emb)
    
    # enrichemnt of GOBP
    enrichment_results_BP = []
    for res in resolutions:
        print(f"Processing resolution: {res}")
        gdata = embed.get_adata_2(resolution=res)
        metagenes = embed.get_metagenes(gdata)
        for mg, gene_module in metagenes.items():
            if len(gene_module) > 0:
                gene_set_path = '../GO_data/c5.go.bp.v2024.1.Hs.symbols.gmt' # The gene sets for enrichemnt
                gmt = gp.read_gmt(gene_set_path)
                enr_up = gp.enrichr(gene_module, gene_sets=[gmt], outdir=None,cutoff=1)
                # Obtain enrichment results and add resolution information
                res2d = enr_up.res2d
                res2d['resolution'] = res
                enrichment_results_BP.append(res2d)
                print("Module finished:", gene_module)
    if enrichment_results_BP:
        df_enrichment_combined = pd.concat(enrichment_results_BP, ignore_index=True)
        output_path = f"../GO_data/results/{model_name}_enrichment_GOBP_nofiltering.csv"
        df_enrichment_combined.to_csv(output_path, index=False)
        print(f"Save the enrichment result of {model_name} to {output_path}")
    
    # enrichemnt of GOMF
    enrichment_results_MF = []
    for res in resolutions:
        print(f"Processing resolution: {res}")
        gdata = embed.get_adata_2(resolution=res)
        metagenes = embed.get_metagenes(gdata)
        for mg, gene_module in metagenes.items():
            if len(gene_module) > 0:
                gene_set_path = '../GO_data/c5.go.mf.v2024.1.Hs.symbols.gmt' # The gene sets for enrichemnt
                gmt = gp.read_gmt(gene_set_path)
                enr_up = gp.enrichr(gene_module, gene_sets=[gmt], outdir=None,cutoff=1)
                # Obtain enrichment results and add resolution information
                res2d = enr_up.res2d
                res2d['resolution'] = res
                enrichment_results_MF.append(res2d)
                print("Module finished:", gene_module)
    if enrichment_results_MF:
        df_enrichment_combined = pd.concat(enrichment_results_MF, ignore_index=True)
        output_path = f"../GO_data/results/{model_name}_enrichment_GOMF_nofiltering.csv"
        df_enrichment_combined.to_csv(output_path, index=False)
        print(f"Save the enrichment result of {model_name} to {output_path}")

    # enrichemnt of GOCC
    enrichment_results_CC = []
    for res in resolutions:
        print(f"Processing resolution: {res}")
        gdata = embed.get_adata_2(resolution=res)
        metagenes = embed.get_metagenes(gdata)
        for mg, gene_module in metagenes.items():
            if len(gene_module) > 0:
                gene_set_path = '../GO_data/c5.go.cc.v2024.1.Hs.symbols.gmt' # The gene sets for enrichemnt
                gmt = gp.read_gmt(gene_set_path)
                enr_up = gp.enrichr(gene_module, gene_sets=[gmt], outdir=None,cutoff=1)
                # Obtain enrichment results and add resolution information
                res2d = enr_up.res2d
                res2d['resolution'] = res
                enrichment_results_CC.append(res2d)
                print("Module finished:", gene_module)
    if enrichment_results_CC:
        df_enrichment_combined = pd.concat(enrichment_results_CC, ignore_index=True)
        output_path = f"../GO_data/results/{model_name}_enrichment_GOCC_nofiltering.csv"
        df_enrichment_combined.to_csv(output_path, index=False)
        print(f"Save the enrichment result of {model_name} to {output_path}")
```

### 03 Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import glob
import os

def count_genes(gene_string):
    if pd.isna(gene_string):
        return 0
    return len(gene_string.split(';'))

# Colors for models
colors = {
    'scbert': '#F4D72D',
    'scfoundation': '#1F9A4C',
    'geneformer': '#1D3D8F',
    'scgpt': '#DF2723'
}

# Store the enrichment results of the GOBP in a folder. The same applies to the enrichment results of GO_MF and GO_CC
file_paths = glob.glob(r'../GO_data/results/GOBP/*.csv')

# Initialize a dictionary to hold data for each model
model_data = {}

# Load data for each model
for file_path in file_paths:
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract model name from file name
    model_name = file_name.split('_')[0]
    df = pd.read_csv(file_path)
    # Filter data based on the adjusted p-value and gene count
    df_filtered = df[df['Adjusted P-value'] < 0.01]
    df_filtered['gene_count'] = df_filtered['Genes'].apply(count_genes)
    df_filtered = df_filtered[df_filtered['gene_count'] > 25]
    # Store the filtered data
    model_data[model_name] = df_filtered

# plotting
fig, ax = plt.subplots(figsize=(6, 3))

for model, df in model_data.items():
    df['term_count'] = df.groupby('resolution')['Term'].transform('count')
    resolution_data = df.groupby('resolution').agg(
        total_term_count=('Term', 'nunique'),
        total_gene_count=('Genes', 'size')
    ).reset_index()
    ax.plot(resolution_data['resolution'], resolution_data['total_term_count'], 
            color=colors[model], alpha=1, label=model, linewidth=2.5)
    ax.scatter(resolution_data['resolution'], resolution_data['total_term_count'],  
               s=30,  # bubble size
               color=colors[model], alpha=1, edgecolors='none', label=model)

ax.set_title(f'Biological Process') # Based on the data used for plotting, modify the title to the corresponding Molecular Function or Cellualr Component
ax.set_xlabel('Resolution')
ax.set_ylabel('Number of enriched GO pathway')

legend_elements_color = [Line2D([0], [0], marker='o', color='w', label=model,
                                markerfacecolor=color, markersize=10) for model, color in colors.items()]
legend_color = ax.legend(handles=legend_elements_color, loc='upper right', bbox_to_anchor=(1.5, 0.7),
                         title="Model Colors", frameon=False)

plt.tight_layout()
plt.show()
```
#### Sample Data
![img_9.png](img_9.png)
#### Figure
![img.png](img.png)
## Configurations

The configuration files can be found in the `biollm/docs/` directory. Ensure that paths, dataset details, and parameters are set correctly before running the code.

## Conclusion

This guide demonstrated how to conduct **Gene Regulatory Network (GRN)** analysis using various foundational models within the BioLLM framework. The process involves generating gene embeddings from a preprocessed dataset, calculating Euclidean distances between genes, and constructing an adjacency matrix that serves as the foundation for building the GRN. 

By using the different models, such as **Geneformer**, **scBERT**, **scGPT**, and **scFoundation**, you can analyze gene interactions and construct an inferred regulatory network to explore gene relationships within a given biological context.
