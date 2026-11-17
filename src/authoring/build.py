"""Regenerate the blank notebooks in notebooks/ from their builder scripts.

    python src/authoring/build.py              # every notebook
    python src/authoring/build.py m01/03       # only some

Builders live in src/authoring/<module>/<NN_name>/build.py and define a module-level `nb`.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def builder_dirs(selected: list[str] | None = None) -> list[Path]:
    dirs = sorted(p.parent for p in HERE.glob("m*/[0-9][0-9]_*/build.py"))
    if selected:
        want = set(selected)
        dirs = [d for d in dirs if f"{d.parent.name}/{d.name[:2]}" in want]
    return dirs


def build_one(d: Path) -> Path:
    ns = runpy.run_path(str(d / "build.py"), run_name="__builder__")
    nb = ns["nb"]
    out = nb.save()
    print(f"wrote {out.relative_to(HERE.parents[1])} ({len(nb.cells)} cells)")
    return out


if __name__ == "__main__":
    for d in builder_dirs(sys.argv[1:] or None):
        build_one(d)
