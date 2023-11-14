# Configuration file for the Sphinx documentation builder.
import os
import os.path as osp
import sys

import benchmarl

# -- Project information

project = "BenchMARL"
copyright = "Meta"
author = "Matteo Bettini"
version = benchmarl.__version__

# -- General configuration
sys.path.append(osp.join(osp.dirname(os.path.realpath(__file__)), "extension"))

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "patch",
]

add_module_names = False
autodoc_member_order = "bysource"
toc_object_entries = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
html_static_path = ["_static"]

html_logo = "_static/benchmarl_logo.png"
html_theme_options = {"logo_only": True, "navigation_depth": 2}
html_css_files = [
    "css/mytheme.css",
]

# -- Options for HTML output
html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"


def setup(app):
    def rst_jinja_render(app, _, source):
        rst_context = {"benchmarl": benchmarl}
        source[0] = app.builder.templates.render_string(source[0], rst_context)

    app.connect("source-read", rst_jinja_render)
