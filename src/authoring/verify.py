"""Build and verify notebooks. Run this after every change; it must end with "ALL OK".

    python src/authoring/verify.py             # everything
    python src/authoring/verify.py m01/03      # only some

For each notebook:
  1. build it from its builder;
  2. cross-check ids: every check/predict/hint id in the notebook is registered in the grader, and
     every registered id is used in the notebook;
  3. run reference.py: every exercise must be checked and accepted;
  4. smoke-run the blank notebook top to bottom (with reference functions injected): 0 unexpected errors.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src")]
os.environ.update(SELFGRADE_NO_LOG="1", SELFGRADE_NO_GUARD="1", MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
# PYTHONIOENCODING only reaches the subprocesses; our own console may be cp1252 on Windows (no emoji).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from build import build_one, builder_dirs  # noqa: E402

ID_RE = re.compile(r"""\b(check|predict|hint)\(\s*["']([^"']+)["']""")


def ids_used(nb: dict) -> set[str]:
    return {m.group(2) for c in nb["cells"] if c["cell_type"] == "code" for m in ID_RE.finditer("".join(c["source"]))}


def verify(d: Path) -> list[str]:
    problems = []
    out = build_one(d)
    nb = json.loads(out.read_text(encoding="utf-8"))
    grader = importlib.import_module(nb["metadata"]["selfgrade"]["grader"])
    registered = set(grader.S.exercises) | set(grader.S.predictions)
    used = ids_used(nb)
    if used - registered:
        problems.append(f"ids used in the notebook but not registered: {sorted(used - registered)}")
    if registered - used:
        problems.append(f"ids registered but never used in the notebook: {sorted(registered - used)}")

    ref = d / "reference.py"
    r = subprocess.run([sys.executable, str(ref)], capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    tail = (r.stdout.strip().splitlines() or ["(no output)"])
    print("   reference:", " | ".join(tail[-3:]))
    if r.returncode:
        problems.append("reference failed:\n" + r.stdout[-3000:] + r.stderr[-3000:])

    s = subprocess.run([sys.executable, str(HERE / "smoke.py"), str(out), str(ref)],
                       capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    print("   smoke:", (s.stdout.strip().splitlines() or ["?"])[-1])
    if s.returncode:
        errs = [ln for ln in s.stdout.splitlines() if "[ERROR" in ln]
        problems.append("smoke test errors:\n" + "\n".join(errs) + "\n" + s.stderr[-3000:])
    return problems


if __name__ == "__main__":
    failures = {}
    for d in builder_dirs(sys.argv[1:] or None):
        print(f"== {d.parent.name}/{d.name}")
        p = verify(d)
        if p:
            failures[d.name] = p
    for name, ps in failures.items():
        print(f"\n### {name}")
        for p in ps:
            print(" -", p)
    print("\nALL OK" if not failures else f"\n{len(failures)} notebook(s) with problems")
    sys.exit(1 if failures else 0)
