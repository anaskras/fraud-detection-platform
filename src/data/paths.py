"""Path helpers for DuckDB SQL."""

from __future__ import annotations

from pathlib import Path


def resolve_path(path: str | Path) -> str:
    """Return an absolute path string safe for DuckDB SQL (forward slashes)."""
    return Path(path).resolve().as_posix()
