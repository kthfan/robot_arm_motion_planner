import importlib.util
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import motion_planner  # noqa: E402

project = "motion_planner"
copyright = "2026, motion_planner Contributors"
author = "motion_planner Contributors"
release = motion_planner.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Build the API pages without the compiled backends (e.g. on a docs host):
# mock pinocchio / EAIK / urchin only if they are not installed.
autodoc_mock_imports = [m for m in ("pinocchio", "eaik", "urchin") if importlib.util.find_spec(m) is None]

napoleon_google_docstrings = True
napoleon_numpy_docstrings = False

html_theme = "alabaster"
html_theme_options = {
    "description": "FK/IK and PTP_T / LINE_T planning for the TM5S",
    "fixed_sidebar": True,
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
exclude_patterns = ["_build"]
