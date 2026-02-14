import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # so Sphinx finds polymon/

# Project info
project = 'polymon'
author = 'PolyMon\'s Team'
release = '0.3.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Mock heavy/optional imports so docs build without torch, rdkit, etc. (e.g. on GitHub Pages CI)
autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
    "rdkit",
    "lightning",
    "mordred",
    "xenonpy",
    "pykan",
    "kan",
    "optuna",
    "tabpfn",
    "torchensemble",
]

# Theme
html_theme = 'sphinx_rtd_theme'
