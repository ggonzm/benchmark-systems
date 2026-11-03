"""Sphinx configuration for the public benchmark-systems documentation."""

from pathlib import Path
import os
import tomllib

ROOT = Path(__file__).resolve().parents[1]
metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

project = "benchmark-systems"
author = "SUPPRESS Research Group"
copyright = "2025–2026, SUPPRESS Research Group"
release = metadata["version"]
version = release
language = "en"

os.environ.setdefault("MPLBACKEND", "Agg")
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
]
myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_typehints = "none"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_custom_sections = [
    (name, "Parameters") for name in (
        "Control inputs", "Generic parameters", "Haldane equation parameters",
        "Bioreactor parameters", "Arrhenius parameters:", "Cooling parameters:",
        "Reactor parameters:", "Annulus parameters", "Gas properties",
        "Tubing parameters", "Bottom hole parameters", "Reservoir parameters",
        "Liquid properties", "Reactor parameters", "Reaction invariant concentrations",
        "Dissociation constants", "Inlet flows",
    )
]
autosummary_generate = True

html_theme = "pydata_sphinx_theme"
html_title = project
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 3,
    "header_links_before_dropdown": 4,
    "show_prev_next": False,
    "github_url": "https://github.com/ggonzm/benchmark-systems",
}
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
html_show_sourcelink = False

doctest_global_setup = """
import warnings
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings(
    'ignore',
    message='FigureCanvasAgg is non-interactive, and thus cannot be shown',
    category=UserWarning,
)
"""
doctest_global_cleanup = "import matplotlib.pyplot as plt; plt.close('all')"


def omit_version_type_docstring(app, what, name, obj, options, lines):
    """Show the package version without the inherited str documentation."""
    if name == "benchmark_systems.__version__":
        lines.clear()


def setup(app):
    app.connect("autodoc-process-docstring", omit_version_type_docstring)
