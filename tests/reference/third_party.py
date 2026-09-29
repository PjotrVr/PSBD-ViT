"""Loading the pinned reference implementations under third_party/ into a test.

The references are research scripts. Most import their own training harness,
datasets, cv2 or a GPU at module level, so importing them whole is impossible
here. What a cross-check needs is the few functions or classes that compute the
statistic, so this module cuts those definitions out of the pinned source with
ast and executes them in a namespace the test supplies. Every textual change a
test makes to a reference (a GPU call pointed at the CPU, a BatchNorm filter
pointed at LayerNorm) goes through load_definitions' replacements argument,
which fails when the text it replaces is absent, so a drifted checkout cannot
pass silently.

A test skips only when the checkout directory itself is missing, with the command
that restores it in the reason. A checkout that exists but lacks the file a test
names is a pin mismatch and fails.
"""

import ast
import json
import os

import pytest
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THIRD_PARTY_ROOT = os.path.join(REPO_ROOT, "third_party")


def reference_file(checkout: str, *relative_parts: str) -> str:
    """The absolute path of a file inside third_party/<checkout>, skipping when absent."""
    checkout_root = os.path.join(THIRD_PARTY_ROOT, checkout)
    if not os.path.isdir(checkout_root):
        pytest.skip(
            f"third_party/{checkout} is not checked out, "
            "run scripts/fetch_third_party.sh to clone it at its pin"
        )

    path = os.path.join(checkout_root, *relative_parts)
    assert os.path.isfile(path), (
        f"{path} is missing although third_party/{checkout} exists, so the "
        "checkout is not at the commit third_party.lock pins"
    )
    return path


def apply_replacements(source: str, replacements: tuple[tuple[str, str], ...]) -> str:
    """source with every (old, new) pair substituted, each old required to occur."""
    for old, new in replacements:
        assert old in source, f"{old!r} no longer occurs in the reference source"
        source = source.replace(old, new)
    return source


def load_definitions(
    path: str,
    names: tuple[str, ...],
    namespace: dict,
    replacements: tuple[tuple[str, str], ...] = (),
) -> dict:
    """Execute the named top-level functions and classes of a Python file in namespace.

    Only the named definitions run, in file order, so the module's own imports
    and script body never execute. namespace must already hold every global the
    definitions read. Returns namespace, now holding the definitions.
    """
    source = open(path).read()
    execute_named_segments(source, names, namespace, replacements, path)
    return namespace


def load_notebook_definitions(
    path: str,
    names: tuple[str, ...],
    namespace: dict,
    replacements: tuple[tuple[str, str], ...] = (),
) -> dict:
    """load_definitions for a Jupyter notebook: the named definitions across its code cells."""
    notebook = json.load(open(path))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    # Cells that hold a named definition are joined in order. Magics and shell
    # lines never appear in the cells this reads, and ast.parse would reject them.
    source = "\n\n".join(
        cell for cell in code_cells if any(f"def {name}" in cell for name in names)
    )
    execute_named_segments(source, names, namespace, replacements, path)
    return namespace


def execute_named_segments(
    source: str,
    names: tuple[str, ...],
    namespace: dict,
    replacements: tuple[tuple[str, str], ...],
    origin: str,
) -> None:
    """Run the named top-level definitions of source in namespace, after replacements."""
    tree = ast.parse(source)
    wanted = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    found = {node.name for node in wanted}
    missing = set(names) - found
    assert not missing, f"{sorted(missing)} not defined at top level in {origin}"

    segments = "\n\n".join(ast.get_source_segment(source, node) for node in wanted)
    patched = apply_replacements(segments, replacements)
    exec(compile(patched, origin, "exec"), namespace)


def keep_cuda_calls_on_the_cpu(monkeypatch) -> None:
    """Make .cuda() on tensors and modules a no-op for the duration of a test.

    BackdoorBox moves every batch and model with .cuda(). This suite runs on the
    CPU and must never touch a GPU, so the call returns its receiver unchanged.
    """
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(nn.Module, "cuda", lambda self, *args, **kwargs: self)
