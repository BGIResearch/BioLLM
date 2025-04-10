Release Notes
=============

Version: v1.2.0
Release Date: April 10, 2025

Changelog
---------

This version introduces a comprehensive refactor of the project structure and the cell analysis task pipelines, significantly improving modularity, maintainability, and usability. Major updates include:

1. Data Processing Module Refactor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Added dedicated data processing scripts for each model.
- Introduced a unified data handling base class `DataHandler`, which standardizes the data workflow:

  - `read_h5ad()`: Reads `.h5ad` files and performs preprocessing.
  - `process()`: Placeholder for model-specific data processing logic.
  - `make_dataset()`: Converts an AnnData object into a PyTorch-compatible Dataset.
  - `make_dataloader()`: Builds a PyTorch DataLoader with distributed training support.

- Model-specific data handlers are organized under the `dataset/` directory and inherit from `DataHandler`.

2. Loader Class Standardization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Unified structure, naming conventions, and output format across all model `loader` classes.
- Refined the responsibility of loaders to **only** handle model loading and embedding extraction.
- Moved all data processing logic from loaders to the corresponding `DataHandler` classes.
- Defined a common interface in the base loader class to enforce consistent implementation across all subclasses.

3. Task Module Refactor
~~~~~~~~~~~~~~~~~~~~~~~~
- **Cell Annotation Task Refactor**: Improved task execution logic and introduced a unified script interface to run different models consistently.
- **Cell Embedding Task**: Added new analysis scripts for evaluating cell embeddings, with support for multiple evaluation metrics.
- **Gene Regulatory Task**: Added dedicated analysis scripts and refactored the gene regulatory network evaluation logic for better performance and clarity.
