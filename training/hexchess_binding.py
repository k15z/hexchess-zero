"""Helpers for importing the native ``hexchess`` Python bindings.

The project treats the PyO3 module as a first-class dependency for runtime
training/eval flows, but several unit tests intentionally run without loading
the native extension (pure function tests, monkeypatched adapters, etc.).

Use ``load_hexchess(required=False)`` in modules that can be imported in those
tests, and ``required=True`` for hard runtime entrypoints.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


_INSTALL_HINT = (
    "hexchess bindings are not available. Build/install them with "
    "`make setup` or `uv run maturin develop --release -m bindings/python/Cargo.toml`."
)


def load_hexchess(*, required: bool) -> ModuleType | None:
    """Load the native ``hexchess`` module.

    Args:
        required: If ``True``, re-raise as an actionable ImportError. If
            ``False``, return ``None`` when bindings are unavailable.
    """
    try:
        return import_module("hexchess")
    except ImportError as exc:
        if required:
            raise ImportError(_INSTALL_HINT) from exc
        return None
