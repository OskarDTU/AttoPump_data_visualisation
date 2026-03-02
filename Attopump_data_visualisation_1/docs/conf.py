"""Sphinx configuration."""

project = "atto1"
author = "OJBS"
copyright = "2026, OJBS"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "shibuya"
