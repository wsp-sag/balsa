# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project info setup ------------------------------------------------------

from datetime import datetime
from pathlib import Path

from pkg_resources import parse_version

import balsa

version_ = parse_version(balsa.__version__)

# -- Project information -----------------------------------------------------

project = 'wsp-balsa'
author = 'WSP Canada Inc.'
copyright = f'{datetime.today().year}, {author}'

# The short X.Y version
version = version_.base_version

# The full version, including alpha/beta/rc tags
release = version_.public

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx_favicon'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = "_static/logo.png"

html_theme_options = {
    'collapse_navigation': True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/wsp-sag/balsa",
            "icon": "fa-brands fa-square-github",
        }
    ]
}

favicons = [
    "favicon-16x16.png",
    "favicon-32x32.png",
    {"rel": "apple-touch-icon", "href": "apple-touch-icon.png"}
]
