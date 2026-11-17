# Technical format

Canonical example: notebook m01/01. Read its four pieces before writing a new notebook:
`src/authoring/m01/01_tables/build.py`, `src/authoring/m01/01_tables/reference.py`,
`src/selfgrade/m01/tables.py`, and the generated `notebooks/m01-preprocessing-visualization/01_tables-you-can-trust.ipynb`.

## Files per notebook

| File | Purpose |
|---|---|
| `src/authoring/<mXX>/<NN_name>/build.py` | Builder: declares the cells with `nbbuild.Notebook`. The only place notebook text is edited. |
| `src/authoring/<mXX>/<NN_name>/reference.py` | Reference solutions (⚠️ spoilers). Ends with `G.S.verify_reference({...})`, which needs every exercise checked and accepted. |
| `src/selfgrade/<mXX>/<name>.py` | Grading module (⚠️ spoilers): `S = Session("mXX/NN", title)`, toy data, `_ref_*` reference computations, a cached `_real()`, predictions and exercises. |
| `notebooks/<module-dir>/<NN>_<slug>.ipynb` | Generated blank notebook. Never edit it by hand. |

Module folders are listed in `MODULE_DIRS` in `src/authoring/nbbuild.py`. Add new modules there and to `curriculum.yaml`.

## Builder API (`src/authoring/nbbuild.py`)

- `Notebook(nb_id, filename, title, subtitle, grader, can_do, builds_on, effort, sources, concept_map)` generates the header cell.
- `nb.md(text)` and `nb.code(text)` add **content** cells, which are replaced on every learner update.
- `nb.learner(key, code)` adds a code cell the learner edits; `nb.answer(key)` adds a 💬 markdown answer cell. These are **learner** cells, kept on update and matched by `key`. Use the exercise or prediction id as the key. A cell with two predictions gets its own key.
- Inside `r"""..."""` strings, write docstrings as `\"\"\"`.
- **Keys are permanent.** Renaming a key orphans the learner's answer: it moves to an "earlier version" section at the end. If you must rename one, keep the old key or write a migration.

## IDs, stubs, checks

- Ids are kebab-case slugs, unique within a notebook: `is-key`, `two-medians`. The visible numbering (**2.3**) is cosmetic and can change.
- A function exercise is a stub with a docstring, `# your code here` and `raise NotImplementedError`, followed by `check("id", fn)` in the same cell. A value exercise is `name = ...` followed by `check("id", name)`. A prediction is `predict("id", ...)` in its own learner cell, *before* the cell that reveals the answer.
- The blank notebook must run top to bottom with every stub untouched: blank items print ⏳, and only cells that call learner functions raise `NotImplementedError`. Demo cells may use learner **functions** (the smoke test injects the reference's), but never learner **values**.
- Register checks with `@S.exercise(id, title, hints=[...], challenge=bool)`, with 2 to 4 hints going from conceptual to nearly the answer. Use the helpers in `selfgrade.core`: `expect`, `expect_close`, `as_int`, `is_df`, `is_series`, `raises`, `unchanged`.
- Test functions on **toy data with edge cases** first (missing values, duplicates, boundaries, a non-default index), then on the real data. Real-data references come from `_real()` (`lru_cache`) via `selfgrade.data.load_olist`.
- `S.prediction(id, answer, explanation, compare=None)`: the explanation says *why*, in 2 or 3 sentences. Use `compare` for tuples, sets, strings or tolerances.
- **Plots:** have the learner write a function that returns a matplotlib `Axes` (or `Figure`). The check inspects it: `ax.patches` / `ax.containers` (bars), `ax.get_ylim()`, `ax.get_xlabel()`, `ax.get_title()`, the order of `ax.get_xticklabels()`, `ax.collections` (scatter or hexbin), `ax.get_xscale()`. Close figures after checking.

## Verify (not optional)

```
python src/authoring/verify.py mXX/NN     # must end with ALL OK
```

It builds the notebook, cross-checks that every id is used and registered, runs `reference.py`, and smoke-runs the blank notebook. Before writing any claim about the real data into a markdown cell, compute it.
