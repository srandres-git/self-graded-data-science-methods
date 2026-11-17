# Module I · Preprocessing and basic data visualization

From raw CSV files to tables, summaries and charts you can defend. Work through the notebooks in order: each one uses the earlier ones as tools.

> These are the **blank originals**. Work on your own copies in `workspace/` (run `python -m selfgrade update`). See the [main README](../../README.md).

| # | Notebook | You'll learn to | Time |
|---|---|---|---|
| 01 | [Tables you can trust](01_tables-you-can-trust.ipynb) | Read CSVs and check the types you got, find keys, predict join sizes, use anti-joins, write invariants, and build an order-level table | ≈ 2 h |
| 02 | [Filter, group, derive](02_filter-group-derive.ipynb) | Write boolean masks (and avoid index-alignment traps), handle NaT, use groupby `agg` vs `transform` and `crosstab`, derive date variables, vectorize logic, and write defensive functions | ≈ 2 h |
| 03 | [Describing a distribution](03_describing-distributions.ipynb) | Identify scales of measurement; summarize center, spread, quantiles and skewness; use the log scale; draw histograms, KDEs, ECDFs, box plots and Q–Q plots; apply Tukey's fences | ≈ 2 h |
| 04 | [Comparing groups, relating variables](04_comparing-and-relating.ipynb) | Compare conditional distributions, recognize small-group noise and funnel plots, handle partial periods, use Pearson and Spearman, fix overplotting, and spot confounding (Simpson's paradox) | ≈ 2.5 h |
| 05 | [Honest charts](05_honest-charts.ipynb) | Use matplotlib's Figure and Axes, choose encodings people read accurately, chart composition (stacked bars, mosaic), choose color palettes, label charts and export them | ≈ 2 h |
| 06 | [Missing data](06_missing-data.ipynb) | Read pandas' missing-value semantics, profile gaps, reason about MCAR/MAR/MNAR, know the cost of complete-case analysis and of imputation, and use weighting | ≈ 2 h |
| 07 | [Cleaning, transforming, features](07_cleaning-transforming-features.ipynb) | Run validity audits, winsorize, scale and log-transform, bin and encode variables, avoid leakage, and build a logged, reproducible pipeline | ≈ 2 h |

**Data:** the Olist Brazilian E-Commerce dataset, downloaded automatically.
**Based on:** Module I of the diploma *Métodos Estadísticos y Matemáticos para la Ciencia de Datos* (UNAM, 2026). Each notebook's header lists the course materials it draws on.
