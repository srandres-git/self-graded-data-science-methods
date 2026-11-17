"""Grading machinery shared by every notebook.

A grading module creates one ``Session`` and registers predictions and exercises:

    S = Session("m01/01", "Tables you can trust")
    check, hint, predict, progress = S.check, S.hint, S.predict, S.progress

    S.prediction("distinct-orders", 6, "why ...")

    @S.exercise("is-key", "is_key(df, cols)", hints=[...])
    def _(is_key):
        expect(..., "diagnostic message")
        return "one-line takeaway"

Learners call ``check("is-key", is_key)``, ``hint("is-key")``, ``predict("distinct-orders", 6)``
and ``progress()``. Every attempt is appended to ``workspace/.progress/<notebook>.json``.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import repo_root, workspace_dir

# ---------------------------------------------------------------------------
# Helpers used inside checks
# ---------------------------------------------------------------------------


class CheckFailed(Exception):
    """Raised inside a check function to report a wrong answer."""


def expect(condition: Any, message: str) -> None:
    """Fail the current check with ``message`` unless ``condition`` holds."""
    if not condition:
        raise CheckFailed(message)


def expect_close(got: Any, want: float, name: str, tol: float = 1e-6) -> None:
    """Relative-tolerance comparison for float answers."""
    if isinstance(got, (pd.Series, pd.DataFrame, np.ndarray, list, tuple)):
        raise CheckFailed(f"`{name}` should be a single number, got a {type(got).__name__}.")
    try:
        got_f = float(got)
    except (TypeError, ValueError):
        raise CheckFailed(f"`{name}` should be a number, got {type(got).__name__}.") from None
    if not np.isfinite(got_f) or abs(got_f - want) > tol * max(1.0, abs(want)):
        raise CheckFailed(f"`{name}` = {got_f:.6g} is not what I expected. Re-check your population and definitions.")


def as_int(x: Any, name: str) -> int:
    """Accept Python/NumPy integers (not bools, not floats) and return an int."""
    expect(isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)),
           f"`{name}` should be an integer, got {type(x).__name__}.")
    return int(x)


def is_df(x: Any, name: str) -> None:
    expect(isinstance(x, pd.DataFrame), f"`{name}` should be a DataFrame, got {type(x).__name__}.")


def is_series(x: Any, name: str) -> None:
    expect(isinstance(x, pd.Series), f"`{name}` should be a Series, got {type(x).__name__}.")


def raises(fn: Callable, exc: type[BaseException], *a, **k) -> bool | str:
    """True if fn raises exc; the exception's name if it raises something else; False if nothing."""
    try:
        fn(*a, **k)
    except exc:
        return True
    except Exception as e:  # noqa: BLE001
        return type(e).__name__
    return False


def unchanged(fn: Callable, *frames: pd.DataFrame | pd.Series):
    """Call fn(*frames) and fail if it mutated any of its inputs."""
    before = [f.copy() for f in frames]
    out = fn(*frames)
    for b, f in zip(before, frames):
        expect(b.equals(f), "Your function modified the data it received. Work on a copy (or use .assign).")
    return out


def not_attempted(*values: Any) -> bool:
    return any(v is Ellipsis for v in values)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class _Exercise:
    ex_id: str
    title: str
    fn: Callable[..., str | None]
    hints: list[str]
    challenge: bool
    hints_shown: int = 0
    passed: bool = False


@dataclass
class _Prediction:
    ex_id: str
    answer: Any
    explanation: str
    compare: Callable[[Any, Any], bool] | None = None
    committed: Any = field(default=None)


def _logging_disabled() -> bool:
    return bool(os.environ.get("SELFGRADE_NO_LOG"))


def _blank_copy_warning() -> None:
    """Warn when the learner is working in the repo's blank notebooks instead of workspace/."""
    root = repo_root()
    if root is None or os.environ.get("SELFGRADE_NO_GUARD"):
        return
    try:
        Path.cwd().resolve().relative_to(root / "notebooks")
    except ValueError:
        return
    print("⚠️  You're running the BLANK copy in notebooks/. Your work here may be overwritten by `git pull`.\n"
          "    Run `python -m selfgrade update` in a terminal, then open the same notebook under workspace/.\n")


class Session:
    def __init__(self, nb_id: str, title: str = ""):
        self.nb_id = nb_id
        self.title = title
        self.exercises: dict[str, _Exercise] = {}
        self.predictions: dict[str, _Prediction] = {}
        _blank_copy_warning()

    # -- registration -------------------------------------------------------
    def exercise(self, ex_id: str, title: str, hints: list[str] | tuple = (), challenge: bool = False):
        if ex_id in self.exercises or ex_id in self.predictions:
            raise ValueError(f"duplicate id {ex_id!r}")

        def deco(fn):
            self.exercises[ex_id] = _Exercise(ex_id, title, fn, list(hints), challenge)
            return fn
        return deco

    def prediction(self, ex_id: str, answer: Any, explanation: str, compare=None) -> None:
        if ex_id in self.exercises or ex_id in self.predictions:
            raise ValueError(f"duplicate id {ex_id!r}")
        self.predictions[ex_id] = _Prediction(ex_id, answer, explanation, compare)

    # -- learner API ----------------------------------------------------------
    def check(self, ex_id: str, *args: Any, **kwargs: Any) -> bool:
        ex = self.exercises.get(ex_id)
        if ex is None:
            print(f"Unknown exercise id {ex_id!r}. Known: {', '.join(self.exercises)}")
            return False
        star = "★ " if ex.challenge else ""
        if not_attempted(*args, *kwargs.values()):
            print(f"⏳ {star}{ex_id} · {ex.title}: not attempted yet (replace the `...`).")
            return False
        try:
            takeaway = ex.fn(*args, **kwargs)
        except CheckFailed as e:
            print(f"❌ {star}{ex_id} · {ex.title}\n   {e}\n   (hint(\"{ex_id}\") for a nudge)")
            self._log(ex_id, "fail", str(e))
            return False
        except NotImplementedError:
            print(f"⏳ {star}{ex_id} · {ex.title}: not attempted yet.")
            return False
        except Exception as e:  # learner code crashed inside the check
            last = traceback.format_exception_only(type(e), e)[-1].strip()
            print(f"💥 {star}{ex_id} · {ex.title}: your code raised an error while being tested:\n   {last}")
            self._log(ex_id, "error", last)
            return False
        ex.passed = True
        msg = f"✅ {star}{ex_id} · {ex.title}"
        if takeaway:
            msg += f"\n   💡 {takeaway}"
        print(msg)
        self._log(ex_id, "pass", "")
        return True

    def hint(self, ex_id: str) -> None:
        ex = self.exercises.get(ex_id)
        if ex is None or not ex.hints:
            print(f"No hints registered for {ex_id!r}.")
            return
        if ex.hints_shown < len(ex.hints):
            ex.hints_shown += 1
        n = ex.hints_shown
        for i, h in enumerate(ex.hints[:n], 1):
            print(f"Hint {i}/{len(ex.hints)}: {h}")
        if n == len(ex.hints):
            print("(That was the last hint.)")
        self._log(ex_id, "hint", str(n))

    def predict(self, ex_id: str, value: Any) -> None:
        p = self.predictions.get(ex_id)
        if p is None:
            print(f"Unknown prediction id {ex_id!r}. Known: {', '.join(self.predictions)}")
            return
        if value is Ellipsis:
            print(f"⏳ Prediction {ex_id}: write your guess first, then run this cell.")
            return
        try:
            same = bool(p.compare(value, p.answer)) if p.compare else bool(value == p.answer)
        except Exception:  # noqa: BLE001
            same = False
        p.committed = value
        head = "🎯 Correct!" if same else f"🤔 Not quite. Answer: {p.answer!r} (you said {value!r})."
        print(f"{head}\n   {p.explanation}")
        self._log(ex_id, "predict-right" if same else "predict-wrong", repr(value))

    def progress(self) -> None:
        """Show what's done, including passes from earlier kernel sessions (read from the log)."""
        logged = {e["ex"] for e in self._read_log() if e.get("event") == "pass"}
        core = [e for e in self.exercises.values() if not e.challenge]
        star = [e for e in self.exercises.values() if e.challenge]
        print(f"{self.nb_id} · {self.title}")
        for label, group in (("Core", core), ("★ Challenge", star)):
            done = sum(e.passed or e.ex_id in logged for e in group)
            print(f"{label}: {done}/{len(group)} passed")
            for e in group:
                mark = "✅" if (e.passed or e.ex_id in logged) else "·"
                hints = f"  (hints used: {e.hints_shown})" if e.hints_shown else ""
                print(f"  {mark} {e.ex_id} · {e.title}{hints}")
        made = {e["ex"] for e in self._read_log() if str(e.get("event", "")).startswith("predict")}
        made |= {k for k, p in self.predictions.items() if p.committed is not None}
        print(f"Predictions made: {len(made & set(self.predictions))}/{len(self.predictions)}")

    # -- reference self-test (authors only) ------------------------------------
    def verify_reference(self, results: dict[str, bool], skip: tuple[str, ...] = ()) -> None:
        """Called at the end of a reference solution file: all exercises must be checked and pass."""
        missing = set(self.exercises) - set(results) - set(skip)
        failed = [k for k, v in results.items() if not v]
        print(f"\n{sum(results.values())}/{len(results)} reference solutions accepted.")
        if missing:
            print("Exercises never checked by the reference:", sorted(missing))
        if failed:
            print("Failed:", failed)
        sys.exit(0 if not failed and not missing else 1)

    # -- logging --------------------------------------------------------------
    @property
    def log_path(self) -> Path:
        return workspace_dir() / ".progress" / f"{self.nb_id.replace('/', '-')}.json"

    def _read_log(self) -> list[dict]:
        try:
            return json.loads(self.log_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def _log(self, ex_id: str, event: str, detail: str) -> None:
        if _logging_disabled():
            return
        try:
            data = self._read_log()
            data.append({
                "t": datetime.now().isoformat(timespec="seconds"),
                "nb": self.nb_id,
                "ex": ex_id,
                "event": event,
                "detail": detail[:300],
            })
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass  # logging must never break the notebook
