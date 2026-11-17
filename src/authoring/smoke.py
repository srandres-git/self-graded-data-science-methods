"""Execute a notebook's code cells in order, without Jupyter.

    python src/authoring/smoke.py <notebook.ipynb> [<reference.py>]

* Blank notebook + reference: learner stubs raise NotImplementedError (expected); after every cell the
  reference's functions are injected, so later demo cells that use learner functions can still run.
* A learner's notebook without a reference: runs their code as-is (used by `/tutor review`).
Reports every *unexpected* exception and exits 1 if there was any.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("SELFGRADE_NO_LOG", "1")
os.environ.setdefault("SELFGRADE_NO_GUARD", "1")
os.environ.setdefault("MPLBACKEND", "Agg")


def load_reference(ref_path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("ref", ref_path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
    # the reference's checks marked exercises as passed in the shared grader: reset them
    for m in list(sys.modules.values()):
        sess = getattr(m, "S", None)
        if getattr(m, "__name__", "").startswith("selfgrade.") and hasattr(sess, "exercises"):
            for ex in sess.exercises.values():
                ex.passed, ex.hints_shown = False, 0
    return {n: f for n, f in vars(mod).items()
            if callable(f) and getattr(f, "__module__", None) == "ref" and not n.startswith("_")}


def run(nb_path: Path, ref_path: Path | None = None, verbose: bool = True) -> int:
    nb_path = nb_path.resolve()
    ref = load_reference(ref_path.resolve()) if ref_path else {}
    os.chdir(nb_path.parent)
    import IPython.display as ipd
    ipd.display = lambda *a, **k: None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: plt.close("all")

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__main__"}
    bad = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        first = src.splitlines()[0][:70] if src.strip() else ""
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(src, f"cell{i}", "exec"), ns)
            status = "ok"
        except NotImplementedError:
            status = "stub (expected)"
        except Exception as e:  # noqa: BLE001
            status = f"ERROR {type(e).__name__}: {e}"
            bad += 1
            traceback.print_exc()
        if verbose or status.startswith("ERROR"):
            print(f"--- cell {i:03d} [{status}] {first}")
            out = buf.getvalue().strip()
            if verbose and out:
                print("    " + out.replace("\n", "\n    ")[:600])
        ns.update(ref)
        plt.close("all")
    print(f"unexpected errors: {bad}")
    return bad


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sys.exit(1 if run(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None) else 0)
