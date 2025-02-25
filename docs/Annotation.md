## Annotation

### scGPT
```python
from biollm.task.annotation.anno_task_scgpt import AnnoTaskScgpt

finetune = True
if finetune:
    config_file = './configs/annotation/scgpt_ft.toml'
    task = AnnoTaskScgpt(config_file)
    task.run()
else:
    import scanpy as sc
    import pickle
    from sklearn.metrics import accuracy_score, f1_score


    try:
        config_file = f'./configs/annotation/scgpt_train.toml'
        task = AnnoTaskScgpt(config_file)
        task.run()
        path = f'./output/scgpt/'  # the outputdir in the config file.
        predict_label = pickle.load(open(path + 'predict_list.pk', 'rb'))
        adata = sc.read_h5ad(
            f'./zheng68k.h5ad')
        labels = adata.obs['celltype'].values
        acc = accuracy_score(labels, predict_label)
        macro_f1 = f1_score(labels, predict_label, average='macro')
        res = {'acc': acc, 'macro_f1': macro_f1}
        print(acc, macro_f1)
    except Exception as e:
        print('error:', e)
```

### Geneformer
```python
from biollm.task.annotation.anno_task_gf import AnnoTask

finetune = True
if finetune:
    config_file = './configs/annotation/gf_ft.toml'
    task = AnnoTask(config_file)
    task.run()
else:
    import scanpy as sc
    import pickle
    from sklearn.metrics import accuracy_score, f1_score


    try:
        config_file = f'./configs/annotation/gf_train.toml'
        task = AnnoTask(config_file)
        task.run()
        path = f'./output/scgpt/'  # the outputdir in the config file.
        predict_label = pickle.load(open(path + 'predict_list.pk', 'rb'))
        adata = sc.read_h5ad(
            f'./zheng68k.h5ad')
        labels = adata.obs['celltype'].values
        acc = accuracy_score(labels, predict_label)
        macro_f1 = f1_score(labels, predict_label, average='macro')
        res = {'acc': acc, 'macro_f1': macro_f1}
        print(acc, macro_f1)
    except Exception as e:
        print('error:', e)

```

### scFoundation
```python
from biollm.task.annotation.anno_task_scf import AnnoTaskScf

finetune = True
if finetune:
    config_file = './configs/annotation/scf_ft.toml'
    task = AnnoTaskScf(config_file)
    task.run()
else:
    import scanpy as sc
    import pickle
    from sklearn.metrics import accuracy_score, f1_score


    try:
        config_file = f'./configs/annotation/scf_train.toml'
        task = AnnoTaskScf(config_file)
        task.run()
        path = f'./output/scgpt/'  # the outputdir in the config file.
        predict_label = pickle.load(open(path + 'predict_list.pk', 'rb'))
        adata = sc.read_h5ad(
            f'./zheng68k.h5ad')
        labels = adata.obs['celltype'].values
        acc = accuracy_score(labels, predict_label)
        macro_f1 = f1_score(labels, predict_label, average='macro')
        res = {'acc': acc, 'macro_f1': macro_f1}
        print(acc, macro_f1)
    except Exception as e:
        print('error:', e)

```

### scBERT
```python
from biollm.task.annotation.anno_task_scbert import AnnoTaskScbert

finetune = True
if finetune:
    config_file = './configs/annotation/scbert_ft.toml'
    task = AnnoTaskScbert(config_file)
    task.run()
else:
    import scanpy as sc
    import pickle
    from sklearn.metrics import accuracy_score, f1_score


    try:
        config_file = f'./configs/annotation/scbert_train.toml'
        task = AnnoTaskScbert(config_file)
        task.run()
        path = f'./output/scgpt/'  # the outputdir in the config file.
        predict_label = pickle.load(open(path + 'predict_list.pk', 'rb'))
        adata = sc.read_h5ad(
            f'./zheng68k.h5ad')
        labels = adata.obs['celltype'].values
        acc = accuracy_score(labels, predict_label)
        macro_f1 = f1_score(labels, predict_label, average='macro')
        res = {'acc': acc, 'macro_f1': macro_f1}
        print(acc, macro_f1)
    except Exception as e:
        print('error:', e)

```

Note: The config directory can be found in the biollm/docs/. Users can modify the corresponding parameters based on the path of their own input and output.
