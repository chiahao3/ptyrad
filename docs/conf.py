# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'PtyRAD'
copyright = '2026'
author = 'Chia-Hao Lee'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",      
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon", 
    "sphinx.ext.viewcode",
    "sphinx_design", # Allows tab, grid card, drop down and more
    "sphinx_togglebutton",
    "sphinx_copybutton", 
    "sphinxcontrib.autodoc_pydantic", 
    "myst_nb",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    "members": True, # This must be true to get autodoc_pydantic showing model fields
    "undoc-members": False,
    'inherited-members': False,
    'show-inheritance': False,
}

exclude_patterns = ['src/ptyrad/starter/',
                    '**.ipynb_checkpoints',
                    '_build'] # Exclude api could also make the build much faster
autosummary_generate = True # This controls the api autosummary, which is quite slow. Toggle off for faster build while testing other pages.

# Autodoc Pydantic configuration
autodoc_pydantic_model_show_json = False # Looks ugly since my fields are often pydantic models as well
autodoc_pydantic_model_show_config_summary = False # Useless since the only config is forbid-extra
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_field_summary = True
autodoc_pydantic_model_member_order = 'bysource'
autodoc_pydantic_model_summary_list_order = 'bysource'
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_show_constraints = False
autodoc_pydantic_field_signature_prefix = "param"
autodoc_pydantic_field_doc_policy = "docstring"

# More comprehensive MyST configuration
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Make sure autodoc works with MyST
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'

nb_execution_mode = "off" # DO NOT execute notebooks. Use the outputs already saved in the file.
source_suffix = ["restructuredtext","myst-nb"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/chiahao3/ptyrad",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "show_navbar_depth": 1,
    "show_toc_level": 1, # The 2nd (in content TOC on the right)
    "home_page_in_toc": True,
    "collapse_navigation": True # This collapses all sections by default
}

# Add your _static directory to the static path
html_static_path = ['_static']

html_css_files = ['custom.css'] # To allow table hover effects in `installation.md`

html_js_files = ['custom.js'] # To allow ref switch the tab in `installation.md`