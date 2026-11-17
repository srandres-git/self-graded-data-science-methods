"""Keep the learner's copies in workspace/ in sync with the blank notebooks in notebooks/.

    python -m selfgrade update            # copy new notebooks, refresh the ones you started
    python -m selfgrade reset <notebook>  # start one notebook over (your old copy is backed up)

How a refresh keeps your work: every cell the blank notebook generates carries metadata
``{"selfgrade": {"key": ..., "role": "content" | "learner", "stub": <hash>}}``.

* ``content`` cells (explanations, demos) are replaced by the new version.
* ``learner`` cells (exercises, predictions, 💬 answers) keep *your* version if you edited it,
  matched by ``key``. If the exercise itself changed since you answered it, a ⚠️ note is added.
* Cells you added yourself (no selfgrade metadata) stay right after the learner cell above them.
* Your answers to exercises that no longer exist are moved to a section at the end.
* The previous copy is saved under ``workspace/.backup/<timestamp>/`` first.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from . import repo_root, workspace_dir

TOP = "__top__"


def stub_hash(source: str) -> str:
    return hashlib.sha1(source.strip().encode("utf-8")).hexdigest()[:12]


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def _meta(cell: dict) -> dict:
    return cell.get("metadata", {}).get("selfgrade", {})


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(nb: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _is_edited(cell: dict) -> bool:
    m = _meta(cell)
    return m.get("role") == "learner" and stub_hash(_src(cell)) != m.get("stub")


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def merge(blank: dict, mine: dict) -> tuple[dict, dict]:
    """Return (new notebook, stats). ``blank`` is the fresh build, ``mine`` the learner's copy."""
    learner: dict[str, dict] = {}
    extras: dict[str, list[dict]] = {}
    anchor = TOP
    for c in mine["cells"]:
        m = _meta(c)
        if m.get("role") == "learner":
            learner[m["key"]] = c
            anchor = m["key"]
        elif not m:  # a cell the learner added (or an earlier ⚠️ note)
            extras.setdefault(anchor, []).append(c)

    out, kept, changed = [], 0, 0
    blank_keys = set()
    for i, c in enumerate(blank["cells"]):
        m = _meta(c)
        if m.get("role") != "learner":
            out.append(copy.deepcopy(c))
        else:
            key = m["key"]
            blank_keys.add(key)
            old = learner.get(key)
            if old is not None and _is_edited(old):
                new = copy.deepcopy(old)
                new["metadata"] = copy.deepcopy(c["metadata"])
                new["id"] = c.get("id", new.get("id"))
                out.append(new)
                kept += 1
                if _meta(old).get("stub") != m.get("stub"):
                    changed += 1
                    fence = "```python" if c["cell_type"] == "code" else "```"
                    out.append(_md(
                        "⚠️ **This exercise was updated after you answered it.** Re-read the prompt above and "
                        "re-run your answer. The new starter cell was:\n\n" + fence + "\n" + _src(c) + "\n```\n\n"
                        "*(Delete this note when you're done.)*"))
            else:
                out.append(copy.deepcopy(c))
            out.extend(copy.deepcopy(extras.pop(key, [])))
        if i == 0:  # learner cells added before any exercise stay right below the title
            out.extend(copy.deepcopy(extras.pop(TOP, [])))

    orphans = [learner[k] for k in learner if k not in blank_keys and _is_edited(learner[k])]
    orphan_extras = [c for k, cs in extras.items() for c in cs]
    if orphans or orphan_extras:
        out.append(_md("## 🗃️ Your answers from an earlier version\n\nThese exercises no longer exist in this "
                       "notebook (they were moved or rewritten). Your work is kept here so nothing is lost."))
        for c in orphans:
            c = copy.deepcopy(c)
            c["metadata"].pop("selfgrade", None)
            out.append(c)
        out.extend(copy.deepcopy(orphan_extras))

    nb = copy.deepcopy(blank)
    nb["cells"] = out
    return nb, {"kept": kept, "changed": changed, "orphaned": len(orphans)}


def _pairs():
    root = repo_root()
    if root is None:
        raise SystemExit("Run this from a clone of the repository (pip install -e . inside it).")
    ws = workspace_dir()
    for blank in sorted((root / "notebooks").glob("m*/*.ipynb")):
        yield blank, ws / blank.parent.name / blank.name


def _backup(path: Path, stamp: str) -> Path:
    ws = workspace_dir()
    dest = ws / ".backup" / stamp / path.relative_to(ws)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    return dest


def update(dry_run: bool = False) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ws = workspace_dir()
    print(f"workspace: {ws}")
    for blank_path, mine_path in _pairs():
        rel = mine_path.relative_to(ws)
        blank = _read(blank_path)
        if not mine_path.exists():
            print(f"  + {rel}  (new)")
            if not dry_run:
                _write(blank, mine_path)
            continue
        mine = _read(mine_path)
        b_build = blank.get("metadata", {}).get("selfgrade", {}).get("build")
        m_build = mine.get("metadata", {}).get("selfgrade", {}).get("build")
        if b_build and b_build == m_build:
            print(f"  = {rel}  (up to date)")
            continue
        new, st = merge(blank, mine)
        note = f"kept {st['kept']} of your answers"
        if st["changed"]:
            note += f", ⚠️ {st['changed']} exercise(s) changed: look for the notes"
        if st["orphaned"]:
            note += f", {st['orphaned']} moved to the end"
        print(f"  ↻ {rel}  (refreshed: {note})")
        if not dry_run:
            _backup(mine_path, stamp)
            _write(new, mine_path)
    print("Open your notebooks under workspace/. (Your previous copies are in workspace/.backup/.)")


def reset(name: str) -> None:
    def nb_id(b: Path) -> str:  # "m01-preprocessing-visualization/03_x.ipynb" -> "m01/03"
        return b.parent.name.split("-")[0] + "/" + b.name.split("_")[0]

    matches = [(b, m) for b, m in _pairs() if name == nb_id(b) or name in b.name]
    if len(matches) != 1:
        raise SystemExit(f"{name!r} matches {len(matches)} notebooks; be more specific (e.g. m01/03).")
    blank, mine = matches[0]
    if mine.exists():
        print("backed up to", _backup(mine, datetime.now().strftime("%Y%m%d-%H%M%S")))
    _write(_read(blank), mine)
    print("reset", mine)
