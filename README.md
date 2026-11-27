# Self-graded Data Science Methods

A free, self-paced course in statistical and mathematical methods for data science, made of **Jupyter notebooks that grade your answers as you go**. Each notebook explains its concepts from scratch: definitions, formulas, the intuition behind them, and how the code works. You then practice on small toy tables you can verify by hand and on a real e-commerce dataset of about 100,000 orders.

```text
✅ ★ order-totals · order_totals(items, payments)
   💡 Aggregate-then-join is the standard fix for many-to-many. The remaining mismatches are real: …
```

| Mark | What you do |
|---|---|
| 🔮 **Predict** | Commit to an answer before running code, then see the explanation. |
| ✍️ **Exercise** | Write the code; `check(...)` grades it, and its feedback diagnoses the likely mistake. |
| ★ **Challenge** | Harder, beyond the core material. |
| 🐞 **Debug** | Code that runs but gives a wrong answer. |
| 💬 **Explain** | Short written answers, where understanding matters more than code. |

`hint("exercise-id")` reveals hints one at a time, and `progress()` shows what you've done.

**You'll need:** basic-to-intermediate Python, and some calculus and linear algebra. Probability and statistics are explained as they come up.

## Modules

The structure follows the diploma's syllabus. Modules are added as the course advances.

| Module | Topic | Status |
|---|---|---|
| I | [Preprocessing and basic data visualization](notebooks/m01-preprocessing-visualization/) | ✅ 7 notebooks |
| II | Unsupervised learning methods to explore, describe and visualize data | planned |
| III | Introduction to generalized linear models (GLMs) for estimation and inference, part 1 | planned |
| IV | Introduction to generalized linear models (GLMs) for estimation and inference, part 2 | planned |
| V | Classical methods for prediction | planned |
| VI | Modern methods for prediction | planned |
| VII | Optimization in machine learning | planned |

## Getting started

You need Python ≥ 3.10 and git.

```bash
git clone https://github.com/srandres-git/self-graded-data-science-methods.git
cd self-graded-data-science-methods
python -m venv .venv                      # or a conda environment
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e .                          # installs the `selfgrade` package and the libraries it needs
python -m selfgrade update                # creates your own copies of the notebooks in workspace/
```

Then open `workspace/m01-preprocessing-visualization/01_tables-you-can-trust.ipynb` in VS Code or Jupyter, select the environment you just created as the kernel, and run it from the top.

> **Always work in `workspace/`.** The files in `notebooks/` are the blank originals: they get replaced when the course is updated. `workspace/` is yours. Git ignores it, so your answers are never committed and never overwritten by `git pull`.

### Getting new material

```bash
git pull
python -m selfgrade update
```

`update` adds any new notebooks and refreshes the ones you've already started, **keeping your work**:

- The explanation cells are replaced by the new version.
- Your exercise, prediction and 💬 answer cells are kept.
- Cells you added yourself stay where you put them.
- If an exercise changed since you answered it, a ⚠️ note appears next to your answer.
- If an exercise was removed, your answer is moved to a section at the end of the notebook.
- Your previous copy is saved in `workspace/.backup/` first.

To start one notebook over: `python -m selfgrade reset m01/03`. Your copy is backed up first.

Keep personal notes in **cells you add yourself**, not inside the explanation cells, because those are replaced on update. If the [changelog](#changelog) says a release added dependencies, run `pip install -e .` again.

### Data

The notebooks use the [Olist Brazilian E-Commerce dataset](https://github.com/olist/work-at-olist-data). It's downloaded automatically, one table at a time, the first time a notebook needs it. Every file is checksum-verified and cached in `data/`. The total is about 120 MB, half of which is a geolocation table that only some exercises use.

- To download everything up front (for example, before going offline): `python -m selfgrade fetch-data --geolocation`
- If you already have the CSVs, set the environment variable `OLIST_DATA_DIR` to their folder to skip the download.

## Repository layout

```text
notebooks/      📘 the course: blank notebooks, organized by module (generated; don't edit)
src/selfgrade/  the package the notebooks import: grading, hints, data download, `update`
src/authoring/  🔧 the code that generates the notebooks, plus reference solutions and tests
curriculum.yaml map of modules → notebooks → sections → concepts → source material
.claude/skills/ the Claude Code skill used to write, review and extend the course
workspace/      ✍️ your copies and progress logs (created by `update`, ignored by git)
```

> ⚠️ **Spoilers:** the grading modules in `src/selfgrade/` and the reference solutions in `src/authoring/` contain the answers. Try the exercises first.

## For authors

Notebooks are generated from Python builder scripts. Never edit the `.ipynb` files directly.

```bash
python src/authoring/build.py            # regenerate the blank notebooks
python src/authoring/verify.py           # build, then test every notebook: must end with "ALL OK"
```

`verify.py` runs the reference solutions against every check, smoke-runs each blank notebook from top to bottom, and cross-checks exercise ids. The style and format rules are in [`.claude/skills/tutor/style.md`](.claude/skills/tutor/style.md) and [`format.md`](.claude/skills/tutor/format.md).

With [Claude Code](https://claude.com/claude-code), the `/tutor` skill in this repository can:
- `integrate` new course material into the right notebooks
- `review` your solved notebook and give feedback
- `quiz` you on your weak spots

## Credits

- **Source material:** the diploma *Métodos Estadísticos y Matemáticos para la Ciencia de Datos* (Universidad Nacional Autónoma de México, UNAM, 2026). Module I is based on the course materials at [chaffrey8/DMEMCD_2026](https://github.com/chaffrey8/DMEMCD_2026); later modules may draw on other sources, listed in [`curriculum.yaml`](curriculum.yaml). This course follows the diploma's syllabus and topics, but its explanations, exercises and code are original. Each notebook's header names the course materials it's based on.
- **Data:** [Olist, work-at-olist-data](https://github.com/olist/work-at-olist-data), MIT License, © 2019 Olist. It isn't redistributed here; it's downloaded from that repository.

## License

[CC BY-NC-SA 4.0](LICENSE): you may share and adapt this material for **non-commercial** purposes, with attribution, as long as you share your adaptations under the same license.

## Changelog

- **2026-11 · v0.1:** Module I (7 notebooks), the `selfgrade` package, `update`/`reset`/`fetch-data`.
- **2026-11 · m01/03:** §4 now explains the matplotlib basics the plotting exercises need (Figure vs Axes, `ax.hist` bins, log axes, `ax.boxplot`), plus a new 🔮 on `np.logspace`. Exercises 4.1–4.6 are renumbered 4.2–4.7; no answers are lost. Run `python -m selfgrade update`.
