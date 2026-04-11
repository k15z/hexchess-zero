"""Drift guard: every public symbol on the live `hexchess` module must be
declared in `bindings/python/hexchess.pyi`.

The stub is hand-maintained and ty won't catch a missing entry if no training
code happens to reference it. This test protects against the failure mode
where someone adds a Rust binding, uses it from Python, forgets to update the
stub, and ty silently passes because the stub still typechecks everything
else.

Scope: names only. Signatures are not cross-checked — that would require
parsing Rust + PyO3 macros, and pyo3-stub-gen is a more appropriate tool for
that if we ever want it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import hexchess

STUB_PATH = Path(__file__).resolve().parent.parent / "bindings" / "python" / "hexchess.pyi"


def _stub_top_level_names() -> set[str]:
    tree = ast.parse(STUB_PATH.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _runtime_public_names() -> set[str]:
    return {
        name
        for name in dir(hexchess)
        if not name.startswith("_") and name != "hexchess"  # drop the re-exported inner module
    }


def test_stub_covers_all_runtime_public_names():
    runtime = _runtime_public_names()
    stub = _stub_top_level_names()
    missing = runtime - stub
    assert not missing, (
        f"hexchess.pyi is missing {len(missing)} symbol(s) present on the live "
        f"module: {sorted(missing)}. Add them to bindings/python/hexchess.pyi."
    )


def test_stub_has_no_stale_names():
    """The reverse check: stub should not declare symbols the binding removed."""
    runtime = _runtime_public_names()
    stub = _stub_top_level_names()
    stale = stub - runtime
    assert not stale, (
        f"hexchess.pyi declares {len(stale)} symbol(s) not present on the live "
        f"module: {sorted(stale)}. Remove them from bindings/python/hexchess.pyi."
    )
