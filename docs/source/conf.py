# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import sys
from pathlib import Path

sys.path.insert(0, Path('../../src').as_posix())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import datetime

import wsp_balsa

project = 'wsp-balsa'
author = 'WSP Canada Inc.'
copyright = f'{datetime.today().year}, {author}'

# The short X.Y version
version = ".".join([str(v) for v in wsp_balsa.__version_tuple__[:3]])

# The full version, including alpha/beta/rc tags
release = wsp_balsa.__version__

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
