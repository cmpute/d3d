# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from setuptools_scm import get_version
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'd3d'
copyright = '2020-2021, Jacob Zhong'
author = 'Jacob Zhong'
here = os.path.dirname(os.path.abspath(__file__))
release = get_version(os.path.dirname(here))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autodoc.typehints',
    'recommonmark',
    # 'autoapi.extension'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/devdocs/', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'PIL': ('https://pillow.readthedocs.io/en/stable/', None)
}

autodoc_mock_imports = [
    'cv2', 'matplotlib', 'PIL', 'scipy', 'sklearn', 'xviz_avs', 'pcl'
]
autodoc_typehints = 'description'
autodoc_default_options = {
    "show-inheritance": True
}
autosectionlabel_prefix_document = True

# AutoApi related configurations
# autoapi_type = 'python'
# autoapi_dirs = ['../d3d']
# autoapi_add_toctree_entry = False
# autoapi_generate_api_docs = False
# autoapi_ignore = ['*.py'] # only use autoapi to doc stub files

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


def setup(app):
    app.add_config_value('recommonmark_config', {
        'auto_toc_tree_section': 'Contents',
        'enable_eval_rst': True,
    }, True)
