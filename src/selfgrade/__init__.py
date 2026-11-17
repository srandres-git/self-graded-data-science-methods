"""selfgrade: runtime for the self-graded data science methods notebooks.

Notebooks import a grading module (e.g. ``selfgrade.m01.tables``) that exposes
``check``, ``hint``, ``predict`` and ``progress``; data comes from ``selfgrade.data``.
Learners run ``python -m selfgrade update`` to get (and refresh) their own copies
of the notebooks in ``workspace/``.
"""
from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"


def repo_root() -> Path | None:
    """The cloned repository root, or None if the package was installed elsewhere."""
    p = Path(__file__).resolve().parents[2]
    return p if (p / "notebooks").is_dir() and (p / "pyproject.toml").is_file() else None


def workspace_dir() -> Path:
    """Where the learner's notebooks and progress logs live."""
    env = os.environ.get("SELFGRADE_WORKSPACE")
    if env:
        return Path(env).expanduser().resolve()
    root = repo_root()
    return (root / "workspace") if root else Path.home() / ".selfgrade" / "workspace"
