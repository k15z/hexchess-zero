#!/usr/bin/env python3
"""Rewrite the [workspace.package] version in Cargo.toml.

Used by release.yml to apply the computed version to each build runner
without committing the bump to git history. Reads the version from the
VERSION env var. No external deps — runs on plain python3 on every
GitHub-hosted runner OS.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys


def main() -> int:
    version = os.environ.get("VERSION")
    if not version:
        print("VERSION env var is required", file=sys.stderr)
        return 1
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        print(f"VERSION must be SemVer X.Y.Z, got {version!r}", file=sys.stderr)
        return 1

    cargo_toml = pathlib.Path("Cargo.toml")
    text = cargo_toml.read_text()
    new_text, n = re.subn(
        r'(\[workspace\.package\]\nversion = )"[^"]*"',
        rf'\g<1>"{version}"',
        text,
        count=1,
    )
    if n != 1:
        print("Failed to find [workspace.package] version line", file=sys.stderr)
        return 1
    cargo_toml.write_text(new_text)
    print(f"Set workspace version to {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
