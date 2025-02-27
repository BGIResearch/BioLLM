## BioLLM: CellType Annotation

### 1. Prediction
#### scGPT:
```python
from biollm.task.annotation.anno_task_scgpt import AnnoTaskScgpt


finetune = True
if finetune:
    config_file = './configs/annotation/scgpt_ft.toml'
    task = AnnoTaskScgpt(config_file)
    task.run()
else:
    try:
        config_file = f'./configs/annotation/scgpt_train.toml'
        task = AnnoTaskScgpt(config_file)
        task.run()
    except Exception as e:
        print('error:', e)

```

#### Geneformer:
```python
from biollm.task.annotation.anno_task_gf import AnnoTask


finetune = True
if finetune:
    config_file = './configs/annotation/gf_ft.toml'
    task = AnnoTask(config_file)
    task.run()
else:
    try:
        config_file = f'./configs/annotation/gf_train.toml'
        task = AnnoTask(config_file)
        task.run()
    except Exception as e:
        print('error:', e)

```

#### scFoundation:
```python
from biollm.task.annotation.anno_task_scf import AnnoTaskScf


finetune = True
if finetune:
    config_file = './configs/annotation/scf_ft.toml'
    task = AnnoTaskScf(config_file)
    task.run()
else:
    try:
        config_file = f'./configs/annotation/scf_train.toml'
        task = AnnoTaskScf(config_file)
        task.run()
    except Exception as e:
        print('error:', e)

```

#### scBERT:
```python
from biollm.task.annotation.anno_task_scbert import AnnoTaskScbert


finetune = True
if finetune:
    config_file = './configs/annotation/scbert_ft.toml'
    task = AnnoTaskScbert(config_file)
    task.run()
else:
    try:
        config_file = f'./configs/annotation/scbert_train.toml'
        task = AnnoTaskScbert(config_file)
        task.run()
    except Exception as e:
        print('error:', e)

```

Note: The config directory can be found in the biollm/docs/. Users can modify the corresponding parameters based on the path of their own input and output.

### 2. Evaluation
#### scGPT:
```python
import scanpy as sc
import pickle
from sklearn.metrics import accuracy_score, f1_score


path = f'./output/scgpt/'  # the outputdir in the config file.
predict_label = pickle.load(open(path + 'predict_list.pk', 'rb'))
adata = sc.read_h5ad(
    f'./zheng68k.h5ad')
labels = adata.obs['celltype'].values
acc = accuracy_score(labels, predict_label)
macro_f1 = f1_score(labels, predict_label, average='macro')
res = {'acc': acc, 'macro_f1': macro_f1}
print(acc, macro_f1)

```

#### Geneformer:
```python
import scanpy as sc
import pickle
from sklearn.metrics import accuracy_score, f1_score


path = f'./output/geneformer/'  # the outputdir in the config file.
predict_label = pickle.load(open(path + 'predict_list.pk', 'rb'))
adata = sc.read_h5ad(
    f'./zheng68k.h5ad')
labels = adata.obs['celltype'].values
acc = accuracy_score(labels, predict_label)
macro_f1 = f1_score(labels, predict_label, average='macro')
res = {'acc': acc, 'macro_f1': macro_f1}
print(acc, macro_f1)

```

#### scFoundation:
```python
import scanpy as sc
import pickle
from sklearn.metrics import accuracy_score, f1_score


path = f'./output/scfoundation/'  # the outputdir in the config file.
predict_label = pickle.load(open(path + 'predict_list.pk', 'rb'))
adata = sc.read_h5ad(
    f'./zheng68k.h5ad')
labels = adata.obs['celltype'].values
acc = accuracy_score(labels, predict_label)
macro_f1 = f1_score(labels, predict_label, average='macro')
res = {'acc': acc, 'macro_f1': macro_f1}
print(acc, macro_f1)

```

#### scBERT:
```python
import scanpy as sc
import pickle
from sklearn.metrics import accuracy_score, f1_score


path = f'./output/scbert/'  # the outputdir in the config file.
predict_label = pickle.load(open(path + 'predict_list.pk', 'rb'))
adata = sc.read_h5ad(
    f'./zheng68k.h5ad')
labels = adata.obs['celltype'].values
acc = accuracy_score(labels, predict_label)
macro_f1 = f1_score(labels, predict_label, average='macro')
res = {'acc': acc, 'macro_f1': macro_f1}
print(acc, macro_f1)

```

### 3. Visualization
```python
import pandas as pd
from typing import Optional
from plottable import ColumnDefinition, Table
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from plottable.plots import bar


_METRIC_TYPE = "Metric Type"

def plot_results_table(df, show: bool = True, save_path: Optional[str] = None) -> Table:
    """Plot the benchmarking results as bar charts for Accuracy and Macro F1 using different colormaps.

    Parameters
    ----------
    show
        Whether to show the plot.
    save_path
        The path to save the plot to. If `None`, the plot is not saved.
    """
    # Delete the 'Metric Type' row as it does not need to be displayed in the final table
    plot_df = df.drop(_METRIC_TYPE, axis=0)
    num_embeds = plot_df.shape[0]

    # Add “Dataset” as a Column
    plot_df["Dataset"] = plot_df.index

    # Define all columns as bar charts
    column_definitions = [
        ColumnDefinition("Dataset", width=1.5, textprops={"ha": "left", "weight": "bold"}),
    ]

    # Extract columns for “Accuracy” and “Macro F1”
    accuracy_cols = df.columns[df.loc[_METRIC_TYPE] == "Accuracy"]
    macro_f1_cols = df.columns[df.loc[_METRIC_TYPE] == "Macro F1"]

    colors = plt.get_cmap('PRGn')(np.linspace(0.25, 1, 256))
    new_colors = colors[::-1]
    new_cmap1 = LinearSegmentedColormap.from_list('modified_magma', new_colors, N=256)

    colors = plt.get_cmap('YlGnBu')(np.linspace(0, 1, 256))
    new_colors = colors[::-1]
    new_cmap2 = LinearSegmentedColormap.from_list('modified_magma', new_colors, N=256)

    # Define a bar chart for the “Accuracy” column
    column_definitions += [
        ColumnDefinition(
            col,
            width=1,
            title=col.split('.')[0],
            plot_fn=bar,
            plot_kw={
                "cmap": new_cmap1,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
            },
            group=df.loc[_METRIC_TYPE, col],
        )
        for col in accuracy_cols
    ]

    # Define a bar chart for the “Macro F1” column
    column_definitions += [
        ColumnDefinition(
            col,
            width=1,
            title=col.split('.')[0],
            plot_fn=bar,
            plot_kw={
                "cmap": new_cmap2,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
            },
            group=df.loc[_METRIC_TYPE, col],
        )
        for col in macro_f1_cols
    ]

    plt.rcParams['pdf.fonttype'] = 42  # Set PDF font type
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(len(df.columns) * 1.3, 3 + 0.35 * num_embeds))
        ax.patch.set_facecolor("white")
        tab = Table(
            plot_df,
            cell_kw={
                "linewidth": 0,
                "edgecolor": "k",
            },
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Dataset",
        ).autoset_fontcolors(colnames=plot_df.columns)
    
    if show:
        plt.show()
    if save_path is not None:
        fig.savefig(save_path, facecolor=ax.get_facecolor(), dpi=300)

    return tab

df = pd.read_csv('./annotation_performance.csv') # Regarding the model performance (Accuracy and Macro F1) of four models and three other annotation tools (scANVI, celltypist and singleR) on different datasets
df = df.set_index("dataset")
plot_results_table(df)

```
#### Sample Data
![img_5.png](img_5.png)

#### Figure
![img_3.png](img_3.png)