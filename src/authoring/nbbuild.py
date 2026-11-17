"""Helpers that builder scripts use to declare a notebook's cells.

    from nbbuild import Notebook
    nb = Notebook(nb_id="m01/01", filename="01_tables-you-can-trust.ipynb",
                  title="Tables you can trust", subtitle="...", grader="selfgrade.m01.tables",
                  can_do=[...], builds_on="...", effort="...", sources=[...], concept_map=[...])
    nb.md(r'''## 1 · ...''')               # explanation / prompt (replaced on every update)
    nb.code(r'''display(toy)''')           # demo code (replaced on every update)
    nb.learner("is-key", r'''def is_key(df, cols): ...''')   # code the learner edits (kept on update)
    nb.answer("reviews-unit")             # a 💬 markdown cell the learner fills in (kept on update)

Inside r'''...''' strings, write docstrings as \"\"\" — they become \"\"\" → triple quotes in the notebook.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from selfgrade.workspace import stub_hash  # noqa: E402

MODULE_DIRS = {
    "m01": "m01-preprocessing-visualization",
}

MARKS = r"""
| Mark | What you do |
|---|---|
| 🔮 **Predict** | Commit to an answer *before* running code. Being wrong here is where the learning happens. |
| ✍️ **Exercise** | Replace the `...` / `raise NotImplementedError`, then run the cell: `check(...)` grades it. |
| ★ **Challenge** | Goes beyond the core material. Optional, but this is where real skill is built. |
| 🐞 **Debug** | Code that runs without errors but gives a wrong answer. Find out why. |
| 💬 **Explain** | Write a short answer in the cell below the question. |
""".strip()


def _clean(s: str) -> str:
    return s.strip("\n")


class Notebook:
    def __init__(self, nb_id: str, filename: str, title: str, subtitle: str, grader: str,
                 can_do: list[str], builds_on: str, effort: str, sources: list[str], concept_map: list[str]):
        self.nb_id, self.filename, self.grader = nb_id, filename, grader
        self.title = title
        self.cells: list[dict] = []
        self._keys: set[str] = set()
        module, number = nb_id.split("/")
        mod_label = {"m01": "Module I"}.get(module, module)
        rows = [
            ("You'll be able to", " · ".join(can_do)),
            ("Builds on", builds_on),
            ("Effort", f"{effort} · {{COUNTS}}"),
            ("Based on", "<br>".join(sources)),
        ]
        table = "| | |\n|---|---|\n" + "\n".join(f"| **{k}** | {v} |" for k, v in rows)
        cmap = " → ".join(concept_map)
        self.md(f"""
# {mod_label} · {number} — {title}
### {subtitle}

{table}

{MARKS}

**Help:** `hint("exercise-id")` reveals one hint per call, and `progress()` shows what you've done. Every check
runs on toy data first (small enough to verify by hand), then on the real data.

**Map:** {cmap}
""")

    # -- cells ---------------------------------------------------------------
    def _add(self, cell_type: str, source: str, role: str, key: str | None = None) -> None:
        source = _clean(source)
        if cell_type == "code":
            source = source.replace('\\"\\"\\"', '"""')
        meta = {"role": role}
        if key is not None:
            if key in self._keys:
                raise ValueError(f"{self.nb_id}: duplicate learner key {key!r}")
            self._keys.add(key)
            meta["key"] = key
            meta["stub"] = stub_hash(source)
        cell = {"cell_type": cell_type, "id": f"c{len(self.cells):03d}",
                "metadata": {"selfgrade": meta}, "source": source.splitlines(True)}
        if cell_type == "code":
            cell.update(execution_count=None, outputs=[])
        self.cells.append(cell)

    def md(self, source: str) -> None:
        self._add("markdown", source, "content")

    def code(self, source: str) -> None:
        self._add("code", source, "content")

    def learner(self, key: str, source: str) -> None:
        self._add("code", source, "learner", key)

    def answer(self, key: str, placeholder: str = "> Your answer:") -> None:
        self._add("markdown", placeholder, "learner", key)

    # -- output ----------------------------------------------------------------
    def _counts(self) -> str:
        import importlib
        S = importlib.import_module(self.grader).S
        n = len(S.exercises)
        star = sum(e.challenge for e in S.exercises.values())
        return f"{n} graded exercises ({star} ★) · {len(S.predictions)} predictions"

    def to_dict(self) -> dict:
        if "{COUNTS}" in "".join(self.cells[0]["source"]):
            self.cells[0]["source"] = "".join(self.cells[0]["source"]).replace(
                "{COUNTS}", self._counts()).splitlines(True)
        build = hashlib.sha1(json.dumps(self.cells, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
        return {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
                "selfgrade": {"nb": self.nb_id, "grader": self.grader, "build": build},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    def output_path(self) -> Path:
        module = self.nb_id.split("/")[0]
        return ROOT / "notebooks" / MODULE_DIRS[module] / self.filename

    def save(self, path: Path | None = None) -> Path:
        path = path or self.output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return path
