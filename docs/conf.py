# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'Coffee Shop Causality Analysis'
copyright = '2025, Sharath S, PhD'
author = 'Sharath S, PhD'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Extension configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True