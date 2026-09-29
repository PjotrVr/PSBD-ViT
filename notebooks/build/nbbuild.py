"""Shared helpers for the notebook builder scripts in notebooks/build/."""

import textwrap

import nbformat

REPO = "/lustre/home/pstika/projects/PSBD-ViT"

SETUP = '''import os
import sys
from pathlib import Path

# Anchor at the repository root so every default path in the library resolves the
# same way it does from a script, whichever directory the notebook was opened from.
REPO_ROOT = next(
    parent
    for parent in [Path.cwd(), *Path.cwd().parents]
    if (parent / "pyproject.toml").exists()
)
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import json
import logging
import warnings

logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.WARNING)
# torchvision's ToTensor deprecation fires once per dataset load and says nothing
# about the numbers below.
warnings.filterwarnings("ignore", message="The transform `ToTensor", category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

import scripts.paper._style  # noqa: F401  the figure style every paper figure uses
from scripts.paper._common import word_list

# The shared style selects the Agg backend for the paper build, which keeps every
# figure out of a notebook, so the inline backend comes back after it.
%matplotlib inline


def macro(table, name):
    """1 value of paper/tables/<table>.macros.json, the string headline.tex prints."""
    with open(f"paper/tables/{table}.macros.json") as handle:
        sidecar = json.load(handle)
    value = sidecar["macros"][name]["value"]
    return value


def show_diagram(name, width=None):
    """Embed a TikZ diagram rendered by notebooks/figures/render.py."""
    display(Image(filename=f"notebooks/figures/{name}.png", width=width))
'''


def md(text):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip("\n"))


def code(text):
    return nbformat.v4.new_code_cell(textwrap.dedent(text).strip("\n"))


def write(name, cells):
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.15"},
    }
    notebook.cells = cells
    path = f"{REPO}/notebooks/{name}.ipynb"
    nbformat.write(notebook, path)
    print("wrote", path)


def said(text):
    """A code cell that renders prose whose numbers come from the kernel's variables."""
    body = textwrap.dedent(text).strip("\n")
    assert '"""' not in body
    source = f'display(Markdown(rf"""\n{body}\n"""))'
    return nbformat.v4.new_code_cell(source)
