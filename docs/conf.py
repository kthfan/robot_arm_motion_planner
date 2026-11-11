import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "motion_planner"
copyright = "2026, motion_planner Contributors"
author = "motion_planner Contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

napoleon_google_docstrings = True
napoleon_numpy_docstrings = False

html_theme = "alabaster"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
