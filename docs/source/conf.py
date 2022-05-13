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
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------

project = 'IBM Analog Hardware Acceleration Kit'
copyright = '2020, 2021, 2022 IBM Research'
author = 'IBM Research'


# The full version, including alpha/beta/rc tags
def get_version() -> str:
    """Get the package version."""
    version_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'src', 'aihwkit', 'VERSION.txt')
    with open(version_path) as version_file:
        return version_file.read().strip()


release = get_version()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'recommonmark'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options specific to this project ----------------------------------------

autodoc_typehints = 'description'
autodoc_mock_imports = ['aihwkit.simulator.rpu_base']

# -- Options specific to readthedocs -----------------------------------------

on_readthedocs = os.environ.get('READTHEDOCS') == 'True'
if on_readthedocs:
    tags.add('env_readthedocs')
else:
    tags.add('env_local')
