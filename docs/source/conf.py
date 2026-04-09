import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # so Sphinx finds polymon/

# Project info
project = 'polymon'
author = 'PolyMon\'s Team'
release = '1.0.3'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Mock optional imports that are rarely needed for core documentation
# Note: torch and related packages are installed in docs/requirements.txt
# so autodoc can properly import and document model classes
autodoc_mock_imports = [
    # Uncomment below if building docs without torch installation
    # "torch",
    # "torch_geometric",
    # "torch_scatter",
    # "torch_sparse",
    # "rdkit",
    # "lightning",
    # "mordred",
    # "xenonpy",
    # "pykan",
    # "kan",
    # "optuna",
    # "tabpfn",
    # "torchensemble",
]

# Theme
html_theme = 'sphinx_rtd_theme'
