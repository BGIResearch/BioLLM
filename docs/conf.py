#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/28 15:27
# @Author  : qiuping
# @File    : conf.py
# @Email: qiupinghust@163.com

"""
change log: 
    2024/10/28 15:27  create file. 
"""
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "BioLLM"
copyright = "2024, BGI, Ping Qiu"
author = "Ping Qiu"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "nbsphinx_link",
    "recommonmark",
]
autodoc_mock_imports = [
    "munch",
    "toml",
    "scanpy",
    "gseapy",
    "transformers",
    "torch_geometric",
    "torch",
    "loompy",
    "omegaconf",
    "wandb",
    "pynvml",
    "accelerate",
    "flash-attn",
    "numpy",
    "pandas",
    "scipy",
    "anndata",
    "tqdm",
    "scib", 
    "einops"
]


templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


# -- Options for nbshpinx ----------------------------------------------------
# https://nbsphinx.readthedocs.io/en/0.8.0/configure.html

nbsphinx_execute = "never"