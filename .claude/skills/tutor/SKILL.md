---
name: tutor
description: Write, extend, review or quiz the self-graded data science methods course (notebooks/ + src/). Use when new diploma course material should be integrated into the notebooks, when a notebook needs to be written or revised, when the learner wants their solved notebook reviewed, or when they want to be quizzed.
argument-hint: integrate [course files] | build <mXX/NN> | review <mXX/NN> | quiz [topic]
---

# Tutor: the self-graded course

This repository is a self-graded course that follows the syllabus of the diploma *Métodos Estadísticos y Matemáticos para la Ciencia de Datos* (UNAM, 2026). Read [style.md](style.md) for how the notebooks are written and [format.md](format.md) for how they're built. Both are mandatory before writing anything. `curriculum.yaml` maps modules → notebooks → sections → concepts → source material, and is the single source of truth for what is covered where.

The diploma's syllabus (modules are added as the course advances):
I. Preprocessing and basic data visualization · II. Unsupervised learning to explore, describe and visualize data · III–IV. Introduction to generalized linear models (GLMs) for estimation and inference, parts 1–2 · V. Classical methods for prediction · VI. Modern methods for prediction · VII. Optimization in machine learning.

**Source material** (the official course materials, in Spanish) can differ from module to module: each module in `curriculum.yaml` lists its own `sources` (repository URL and the last commit integrated). Module I's is `chaffrey8/DMEMCD_2026`, usually cloned as the sibling folder `../DMEMCD_2026`. For a new module, ask the user where its material is (a new repository, a folder, files), then record it in `curriculum.yaml`. Link to a repository's **root**, never to a branch or path inside it, because branches get deleted and folders move. Never write a local path into committed files. Never modify source repositories.

**Python:** use the interpreter of the environment where `selfgrade` is installed (`pip install -e .`). If unsure, ask, or check the auto-memory.

Arguments: `$ARGUMENTS`. The first word picks the mode. With no arguments, ask which mode is wanted.

## Change workflow (every mode that edits the repository)

1. **Propose first.** Before editing any file outside `workspace/`, show the user what will change: the plan table, new or changed exercises, and changed explanations. **Wait for explicit approval.** Approval of one plan doesn't cover later, different changes.
2. **Work on a new branch, never directly on `main`.** Start from an up-to-date `main` (`git switch main && git pull`), then `git switch -c content/<short-topic>` (e.g. `content/m02-pca`) or `fix/<short-topic>`.
3. Build, then verify with `python src/authoring/verify.py` (it must end with **ALL OK**). Show the user the result: the notebooks to open, and a summary of the diff.
4. Commit on the branch only when the user approves. Push the branch and merge into `main` only when the user says so: through a pull request on GitHub, or a local merge if they prefer. Never force-push and never rewrite `main`'s history.

---

## Mode `integrate [files]`: new course material arrived

1. **Find what's new.** In the course repo, run `git log` / `git diff` since the last integration (recorded as the module's `integrated_through` in `curriculum.yaml`), or use the files given. Read **all** of it: demo notebooks, labs, helper modules, data files.
2. **Extract concepts.** List the *generalizable* concepts, meaning what a practitioner should be able to do on any dataset, not "the steps in the notebook". For each one, note a common mistake; it becomes a 🔮 or a 🐞.
3. **Map them against `curriculum.yaml`.** For each concept decide one of:
   - **already covered**: at most an applied twist inside an existing exercise; no new drill
   - **extends** a notebook's section, or adds a section to a notebook on the same topic
   - **new notebook**, when it's a new topic or an existing notebook would exceed about 16 exercises
   - **new module folder**, when the material starts the next syllabus module
4. **Show the plan to the user and wait for approval.** Use a table: concept → decision → notebook/section → new exercise ids. Say what you'd leave out and why.
5. **Create the branch** (see Change workflow), then **write** following format.md. Never rename an existing learner key. Check whether the environment needs new libraries; if it does, add them to `pyproject.toml` and note it in the README changelog.
6. **Verify:** `python src/authoring/verify.py <ids>` must end with **ALL OK**. Compute every real-data claim before writing it.
7. **Update** `curriculum.yaml` (sections, concepts, the module's sources and `integrated_through`), the module README's notebook table, and the README changelog. Then tell the user to run `python -m selfgrade update`.

## Mode `build <mXX/NN>`: write or revise one notebook

Same as steps 4–7 of `integrate`, for one notebook: propose, get approval, then work on a branch. When revising, keep learner keys stable. Changing a stub is fine: `update` shows the learner a ⚠️ note.

## Mode `review <mXX/NN>`: feedback on the learner's work

1. Read the learner's copy `workspace/<module-dir>/<NN>_*.ipynb` (with outputs) and `workspace/.progress/mXX-NN.json` (attempts, failures, hints, wrong predictions). If outputs are missing, run `python src/authoring/smoke.py <their notebook>` (no reference argument) to see the results.
2. Give feedback the way a good teacher would:
   - **💬 answers first.** For each one, say what's right, then what's missing or imprecise, and why it matters.
   - **Code** that's correct but non-idiomatic, fragile, or right for the wrong reason. Quote the specific lines.
   - **Wrong or unattempted items.** Don't paste the solution; ask a guiding question or give a smaller sub-step. Give the full solution only if asked.
   - **Patterns from the log.** Name the misconception behind repeated failures, hint use or wrong predictions.
   - Keep praise short and specific.
3. Update `workspace/progress.md`. Under the notebook, mark concepts **solid** / **shaky** / **not yet**, each with one line of evidence and the date, and maintain the "Recurring weak spots" section. This file is private: it's in `workspace/`.
4. Offer 2 or 3 targeted drills. If the learner accepts, write them to `workspace/drills/<topic>.ipynb` with the same machinery, and verify them.

## Mode `quiz [topic]`: spaced repetition in chat

Draw from `workspace/progress.md` (shaky concepts first), or from `topic`. Ask **one question at a time** and wait for the answer. Mix the types: predict the output, spot the bug, pick the method, explain why. Give feedback after each answer. Ask 5 questions by default, then summarize and update `workspace/progress.md`.

---

## Principles (all modes)

- **Organize by topic, not by course upload.** Earlier lessons become tools for later ones, not repeated drills.
- **Self-contained:** a reader who never saw the course can learn from the notebook alone (style.md).
- **Every check teaches:** failure messages diagnose the mistake, and success messages carry a 💡 takeaway.
- **Original work:** the same ideas as the course, with different code, variable names and questions. Credit the source in each notebook's header.
- **Approved changes only, on a branch:** nothing is applied to the repository without the user's approval, and nothing is committed directly to `main` (see Change workflow).
- **Learner work is sacred:** never edit `workspace/` except through `python -m selfgrade update` / `reset`, or when the learner asks.
