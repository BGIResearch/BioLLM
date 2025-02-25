**Integration of Novel Models into the BioLLM Framework: A Step-by-Step Guide**


---

### Abstract

The BioLLM framework provides a modular system for implementing large language models (LLMs) in single-cell and multi-omics analyses. Here, we describe a systematic procedure for integrating novel models into this framework, followed by the development of custom downstream tasks. Our approach is grounded in a standardized base class, `LoadLlm`, that handles model loading and initialization, thereby enabling consistent interaction and seamless extension. We further illustrate how to implement downstream analyses by extending the `BioTask` class. This step-by-step guide aims to facilitate researchers in adopting the BioLLM framework for diverse models and tasks.

---

## 1. Introduction

Large language models (LLMs) have shown promise in various applications, including natural language processing, computational biology, and single-cell data interpretation. The BioLLM framework is designed to simplify the integration of state-of-the-art models and to streamline downstream analyses on genomic or transcriptomic data. By adhering to a common interface (`LoadLlm`) and a standardized task infrastructure (`BioTask`), developers can rapidly prototype new computational pipelines.

In this guide, we provide detailed instructions on incorporating novel models and implementing user-defined tasks. Our description encompasses the construction of a new `load_newmodel.py` module, modifications required to load the new model within the BioLLM environment, and best practices for designing downstream analyses.

---

## 2. Implementation of a New Model

### 2.1 Creating `load_newmodel.py`

To integrate a new LLM into BioLLM, create a dedicated Python file (e.g., `load_newmodel.py`) within the `base/` directory. Define a class, here named `LoadNewModel`, that inherits from `LoadLlm`, the base class for all model integrations within BioLLM. This design ensures that your new model follows the same lifecycle as existing models, including device placement and parameter management.

```python
# base/load_newmodel.py
from BioLLM.models.base import LoadLlm

class LoadNewModel(LoadLlm):
    def __init__(self, args):
        """
        Initialize the new model, including loading vocabulary, the model weights,
        and any required preprocessing.
        """
        super(LoadNewModel, self).__init__(args)
        self.vocab = self.load_vocab()
        self.model = self.load_model()
        self.init_model()
        self.model = self.model.to(self.args.device)
    
    def load_model(self):
        """
        Load the novel model, for instance from pretrained weights.
        """
        model = SomeModelClass.from_pretrained(self.args.model_path)
        model.to(self.device)
        return model

    def get_dataloader(self, input_data):
        """
        Convert input data into the format expected by the model (e.g., a PyTorch Dataloader).
        """
        return processed_data

    def load_vocab(self):
        """
        Load any required vocabulary specific to the new model.
        """
        return vocab

    def get_gene_embedding(self, gene_ids):
        """
        Obtain gene-level embeddings (to be implemented based on model specifics).
        """
        pass

    def get_cell_embedding(self, adata, do_preprocess=False):
        """
        Obtain cell-level embeddings, optionally preprocessing the data beforehand.
        """
        pass

    def get_gene_expression_embedding(self, adata, do_preprocess=False):
        """
        Obtain embeddings for gene expression data.
        """
        pass

    def get_embedding(self, emb_type, adata=None, gene_ids=None):
        """
        A unified interface for retrieving different embedding types.
        """
        pass

    def freeze_model(self):
        """
        Freeze model parameters to prevent gradient updates.
        """
        for param in self.model.parameters():
            param.requires_grad = False
```

In this example, `LoadNewModel` manages both the loading of model weights from a pretrained checkpoint and the loading of a custom vocabulary. The methods `get_gene_embedding`, `get_cell_embedding`, and `get_gene_expression_embedding` provide task-specific embeddings. Each function can be further refined according to the demands of the downstream tasks.

---

## 3. Development of Downstream Tasks

### 3.1 Extending the `BioTask` Class

BioLLM features a task infrastructure encapsulated by the `BioTask` base class, which governs data input/output, logging, and model interactions. To develop a new downstream task, subclass `BioTask` and implement the core logic in a `run()` method. The following example demonstrates how to retrieve embeddings from your newly integrated model and carry out a sample analysis.

```python
# tasks/my_new_task.py
from bio_task import BioTask

class MyNewTask(BioTask):
    def __init__(self, cfs_file, data_path=None, load_model=True):
        super(MyNewTask, self).__init__(cfs_file, data_path, load_model)

    def run(self):
        # Step 1: Load single-cell data
        adata = self.read_h5ad()
        
        # Step 2: Obtain cell-level embeddings using the loaded model
        embedding = self.load_obj.get_embedding("cell", adata)
        
        # Step 3: Perform task-specific operations on the embeddings
        results = self.process_embedding(embedding)
        
        # Step 4: Log the result
        self.logger.info("Task completed.")

    def process_embedding(self, embedding):
        """
        Example: Compute the mean vector of the obtained embeddings.
        In practice, more sophisticated analyses (e.g. clustering) might be used.
        """
        return embedding.mean(axis=0)
```

This approach allows for the straightforward adaptation of typical single-cell analyses (e.g., clustering, differential expression) to LLM-based embeddings, thereby leveraging advanced contextual representations to derive biological insights.

---

### 3.2 Integrating the New Model in BioLLM

Within the `BioTask` class (or its parent), update the `load_model()` method to handle your newly introduced model type, typically by checking a configuration flag (`args.model_used`) and instantiating the corresponding class:

```python
if self.args.model_used == 'mynewmodel':
    self.load_obj = MyNewModel(self.args)
    return self.load_obj.model
```

This pattern follows the existing structure for loading other built-in models and preserves extensibility for future model additions.

---

### 3.3 Data Loading and Preprocessing

To accommodate specialized preprocessing for your new task, you may override the default `read_h5ad()` method provided by `BioTask`. By selectively calling the parent method through `super()`, you can preserve core functionality while incorporating custom routines:

```python
def read_h5ad(self, h5ad_file=None, preprocess=True, filter_gene=True):
    # Invoke the superclass method to read .h5ad files
    adata = super().read_h5ad(h5ad_file, preprocess, filter_gene)
    
    # Implement task-specific preprocessing, e.g., normalization
    adata = self.custom_preprocess(adata)
    return adata

def custom_preprocess(self, adata):
    # An example: total count normalization using scanpy
    sc.pp.normalize_total(adata, target_sum=1e4)
    return adata
```

Such flexibility ensures compatibility with specialized analyses that might rely on unique preprocessing strategies.

---

### 3.4 Task Execution

The `run()` method is the canonical entry point for executing your newly defined task. The typical workflow includes:

1. **Reading and preprocessing input data**  
2. **Obtaining model embeddings**  
3. **Executing a downstream analysis**  
4. **Logging or outputting results**

```python
def run(self):
    adata = self.read_h5ad()
    embedding = self.load_obj.get_embedding("cell", adata)
    results = self.analyze_embedding(embedding)
    self.logger.info("Analysis completed.")
```

This structure imposes clear boundaries between data processing, model inference, and analysis, aligning well with best practices for reproducible computational biology.

---

### 3.5 Result Tracking and Logging

BioLLM supports a variety of logging utilities, including compatibility with `wandb`. To track intermediate metrics or final outputs, simply integrate logging calls in appropriate sections of the task code:

```python
if self.wandb:
    self.wandb.log({"task_result": results})
```

Such functionality enables experiment management and reproducibility across diverse computational setups.

---

## 4. Conclusion and Future Directions

Herein, we have presented a systematic procedure for integrating new LLMs into the BioLLM framework and creating domain-specific downstream tasks. By adhering to the architectural principles of `LoadLlm` and `BioTask`, developers can ensure consistent interface definitions, maintain code modularity, and expedite subsequent model or task extensions.

In future work, additional functionalities such as interactive model fine-tuning, advanced hyperparameter optimization, and expanded compatibility with cutting-edge single-cell analysis pipelines may further enhance BioLLM. We encourage contributions from the broader research community and welcome feedback, issues, and pull requests on our official repository.

---

**Availability and Reproducibility**  
For detailed documentation, installation instructions, and examples, please refer to [BioLLM’s official GitHub repository](https://github.com/BGIResearch/BioLLM). All code modifications mentioned herein follow the open-source license provided with BioLLM.

---

*Correspondence and requests for materials should be addressed to the BioLLM contributors via the repository’s issue tracker.*