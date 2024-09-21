# Configuration file for the Sphinx documentation builder.
import os.path as osp
import sys

import benchmarl
import benchmarl_sphinx_theme

# -- Project information

project = "BenchMARL"
copyright = "Meta"
author = "Matteo Bettini"
version = benchmarl.__version__

# -- General configuration
sys.path.append(osp.join(osp.dirname(benchmarl_sphinx_theme.__file__), "extension"))

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "patch",
]

add_module_names = False
autodoc_member_order = "bysource"
toc_object_entries = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torchrl": ("https://pytorch.org/rl/stable/", None),
    "tensordict": ("https://pytorch.org/tensordict/stable", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
html_static_path = [
    osp.join(osp.dirname(benchmarl_sphinx_theme.__file__), "static"),
    "_static",
]


html_theme = "sphinx_rtd_theme"
html_logo = (
    "https://raw.githubusercontent.com/matteobettini/benchmarl_sphinx_theme/master/benchmarl"
    "_sphinx_theme/static/img/benchmarl_logo.png"
)
html_theme_options = {"logo_only": True, "navigation_depth": 2}
# html_favicon = ('')
html_css_files = [
    "css/mytheme.css",
]

# -- Options for EPUB output
epub_show_urls = "footnote"


def setup(app):
    def rst_jinja_render(app, _, source):
        rst_context = {"benchmarl": benchmarl}
        source[0] = app.builder.templates.render_string(source[0], rst_context)

    app.connect("source-read", rst_jinja_render)
    app.add_js_file("js/version_alert.js")
