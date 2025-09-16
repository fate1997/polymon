import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # so Sphinx finds polymon/

# Project info
project = 'polymon'
author = 'PolyMon\'s Team'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Theme
html_theme = 'sphinx_rtd_theme'
