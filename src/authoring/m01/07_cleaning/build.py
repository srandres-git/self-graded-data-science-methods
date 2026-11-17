"""Builder: Module I · 07 — Cleaning, transforming, features."""
from nbbuild import Notebook

COURSE = "https://github.com/chaffrey8/DMEMCD_2026"

nb = Notebook(
    nb_id="m01/07",
    filename="07_cleaning-transforming-features.ipynb",
    title="Cleaning, transforming, features",
    subtitle="Audits, outliers, scaling, logs, bins, encodings, leakage, and a pipeline you can defend",
    grader="selfgrade.m01.cleaning",
    can_do=["turn data-quality rules into an audit table",
            "tell an error from a plausible extreme, and flag instead of delete",
            "choose between min-max, z-score, robust scaling and logs",
            "bin and encode variables without inventing information",
            "spot data leakage in preprocessing",
            "write a cleaning pipeline with a decision log and a data fingerprint"],
    builds_on="01 (the order-level table), 03 (quantiles, Tukey fences, skewness), "
              "06 (missing values and indicators). These are used as tools here, not re-taught.",
    effort="≈ 2 h",
    sources=[f"Diplomado *Métodos Estadísticos y Matemáticos para la Ciencia de Datos* (UNAM, 2026), Module I: "
             f"[`04_Limpieza_Transformacion_Work_Olist`, `S03_Notebook_Demostracion_Python` "
             f"(reproducibility)]({COURSE})"],
    concept_map=["validity audits", "outliers: flag, clip, winsorize", "scaling & logs", "discretization",
                 "encoding", "features & leakage", "pipelines & provenance", "mini-project: a cleaning pipeline"],
)

# ---------------------------------------------------------------------------
nb.md(r"""
## 0 · Setup

This notebook works on the Olist tables again, mostly on the **order-level table** from Notebook 01: one row per order, built by aggregating `items` and `payments` first and validating every join. The setup cell rebuilds it.

**The rule for this whole notebook: never modify the raw data.** Every cleaning decision produces *new* columns or a *new* table. The original stays available, so you can compare before and after, and anyone can reverse your choices.
""")
nb.code(r"""
import hashlib
import sys

import numpy as np
import pandas as pd
from IPython.display import display

from selfgrade.data import OLIST_FILES, load_olist, olist_path
from selfgrade.m01.cleaning import check, hint, predict, progress
from selfgrade.m01.cleaning import (toy_items_dirty, toy_dates, toy_prices, toy_payments_type,
                                    toy_order_table)

orders, items = load_olist("orders"), load_olist("items")
payments, customers = load_olist("payments"), load_olist("customers")

# The order-level table (Notebook 01): aggregate the detail tables first, validate every join.
item_agg = items.groupby("order_id").agg(n_items=("price", "size"), items_value=("price", "sum"),
                                         freight_value=("freight_value", "sum"))
paid_agg = payments.groupby("order_id").agg(paid_value=("payment_value", "sum"))
order_table = (orders[["order_id", "order_status", "order_purchase_timestamp", "customer_id"]]
               .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left",
                      validate="many_to_one")
               .merge(item_agg, on="order_id", how="left", validate="one_to_one")
               .merge(paid_agg, on="order_id", how="left", validate="one_to_one")
               .drop(columns="customer_id"))
order_table["n_items"] = order_table["n_items"].fillna(0).astype(int)
assert len(order_table) == len(orders) and order_table["order_id"].is_unique
order_table.head()
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 1 · Validity audits: diagnose before you correct

*Data quality* has several dimensions, and each one turns into rules you can check:

| Dimension | Question | Example rule |
|---|---|---|
| **Validity** | Is each value in its allowed domain? | price ≥ 0, review score ∈ {1, …, 5} |
| **Consistency** | Do related fields agree with each other? | delivered after purchased |
| **Uniqueness** | Does each entity appear once? | `order_id` is a key (Notebook 01) |
| **Completeness** | Is the value there when it should be? | a delivered order has a delivery date |

A rule violation is a **flag, not a verdict**. A negative price is very likely an **error**, unless the system records refunds with a sign. A price of 6,000 BRL is a **plausible extreme**: rare, but perfectly possible for a refrigerator. Deleting either one silently changes the population you're describing. So the workflow is:
1. write each rule as code
2. count the violations
3. look at the flagged rows and their context
4. decide (keep / correct / exclude), and **record the decision**

> ⚠️ **Missing values inside a rule.** Any comparison with `NaN`/`NaT` is `False`. A rule written as "NOT (the good condition)" therefore counts every missing value as a violation, while a rule written as "the bad condition" counts none of them. Neither is automatically right. Decide what "unknown" means for each rule.
""")
nb.code(r"""
dates_toy = toy_dates()
dates_toy
""")
nb.md(r"""
🔮 **1.1** How many rows does `(~(dates_toy["delivered"] >= dates_toy["purchased"])).sum()` count as violations?
""")
nb.learner("nat-violation", r"""
predict("nat-violation", ...)
""")
nb.code(r"""
good = dates_toy["delivered"] >= dates_toy["purchased"]
print("written as NOT(good):", int((~good).sum()))
print("written as the bad event:", int((dates_toy["delivered"] < dates_toy["purchased"]).sum()))
print("unknown (no delivery yet):", int(dates_toy["delivered"].isna().sum()))
""")
nb.md(r"""
✍️ **1.2** Write `audit(df, rules)`. `rules` is a dict `name → function(df) → boolean Series`, where `True` means "this row violates the rule". Return a DataFrame indexed by rule name (in the dict's order) with:
- `n_violations` (int)
- `share` = `n_violations / len(df)`

If a rule returns missing values, count them as **not** violated. They're "unknown", not "bad".
""")
nb.learner("audit-table", r"""
def audit(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    # your code here
    raise NotImplementedError


check("audit-table", audit)
""")
nb.code(r"""
order_rules = {
    "approved before purchase": lambda d: d["order_approved_at"] < d["order_purchase_timestamp"],
    "handed to carrier before purchase": lambda d: d["order_delivered_carrier_date"] < d["order_purchase_timestamp"],
    "delivered to customer before carrier": lambda d: d["order_delivered_customer_date"] < d["order_delivered_carrier_date"],
    "estimate before purchase": lambda d: d["order_estimated_delivery_date"] < d["order_purchase_timestamp"],
    "delivered but no delivery date": lambda d: d["order_status"].eq("delivered") & d["order_delivered_customer_date"].isna(),
}
audit(orders, order_rules)
""")
nb.md(r"""
★ **1.3** One rule is missing from that table: the carrier received the parcel **before** the payment was approved. Count those orders (`n_carrier_before_approval`), and compute the **median** gap between the two timestamps, in hours (`median_gap_hours`). Does it look like an impossible event or like a recording delay?
""")
nb.learner("event-order", r"""
n_carrier_before_approval = ...
median_gap_hours = ...

check("event-order", n_carrier_before_approval, median_gap_hours)
""")
nb.md(r"""
💬 **1.4** For each of these three, would you call it an error or a plausible extreme, and what would you do with it? A `freight_value` of 0, an order whose items cost 13,440 BRL, and a delivery date earlier than the carrier hand-off.
""")
nb.answer("error-or-outlier")

# ---------------------------------------------------------------------------
nb.md(r"""
## 2 · Outliers: flag, clip, winsorize

Once a value is flagged as extreme (e.g. by the Tukey fences of Notebook 03), there are three common treatments:

| Treatment | What it does | Keeps $n$? |
|---|---|---|
| **Trimming** | drops the extreme rows | no |
| **Clipping** (capping) | replaces values beyond fixed bounds $[a, b]$ by $a$ or $b$ | yes |
| **Winsorizing** | clipping where the bounds are sample quantiles, e.g. the 1st and 99th percentiles | yes |

In formulas, clipping is $\operatorname{clip}(x; a, b) = \min(\max(x, a), b)$, and winsorizing at level $p$ uses $a = Q(p)$, $b = Q(1-p)$, where $Q$ is the sample quantile function.

**Why the mean moves and the median doesn't.** The mean is $\frac1n\sum_i x_i$: lowering one value by $\Delta$ lowers the mean by $\Delta / n$, so caps on a long tail pull it down. The median depends only on *which* values sit in the middle of the sorted data. Capping keeps the order of the values (an extreme stays at the end, just closer), so as long as the caps don't reach the middle, the median is unchanged.

> ✅ **Good practice:** treat clipping and winsorizing as a **sensitivity scenario**. Compute your result with and without it. If the conclusion changes, the extremes are driving it, and that's worth reporting, not hiding.
""")
nb.code(r"""
prices_toy = toy_prices()
print(list(prices_toy), "| mean", round(prices_toy.mean(), 2), "| median", prices_toy.median())
""")
nb.md(r"""
🔮 **2.1** Clip `prices_toy` at an upper bound of 50 (`prices_toy.clip(upper=50)`). What is the new **mean**?
""")
nb.learner("clip-mean", r"""
predict("clip-mean", ...)
""")
nb.code(r"""
clipped = prices_toy.clip(upper=50)
print(list(clipped), "| mean", round(clipped.mean(), 2), "| median", clipped.median())
""")
nb.md(r"""
✍️ **2.2** Write `winsorize(s, lower=0.01, upper=0.99)`. It caps the values at the `lower` and `upper` sample quantiles, keeps missing values missing and keeps the index.
""")
nb.learner("winsorize", r"""
def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    # your code here
    raise NotImplementedError


check("winsorize", winsorize)
""")
nb.md(r"""
✍️ **2.3** Winsorize `items["price"]` with the default levels. Compute the **relative** change of the mean (`mean_change`) and of the median (`median_change`): $(\text{after} - \text{before}) / \text{before}$.
""")
nb.learner("winsor-sensitivity", r"""
mean_change = ...
median_change = ...

check("winsor-sensitivity", mean_change, median_change)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 3 · Scaling and logs

**Scaling** puts variables measured in different units on comparable footing. That matters for any method built on distances or gradients, which Modules V–VII cover in depth. The three standard choices, for a column $x$:

| Name | Formula | Result |
|---|---|---|
| **min-max** | $x^* = \dfrac{x - \min x}{\max x - \min x}$ | values in $[0, 1]$ |
| **z-score** (standardization) | $z = \dfrac{x - \bar x}{s}$ | mean 0, standard deviation 1 |
| **robust scaling** | $r = \dfrac{x - \operatorname{median}(x)}{\mathrm{IQR}(x)}$ | median 0, interquartile range 1 |

Here $\bar x$ is the mean, $s$ the sample standard deviation (with $n-1$), and $\mathrm{IQR} = Q_3 - Q_1$ the *interquartile range*, the width of the middle 50 % of the data.
- **min-max** depends on the two most extreme values. A single outlier squeezes everything else into a tiny interval near 0.
- **z-score** reads as "how many standard deviations from the mean", but the mean and $s$ are themselves pulled by outliers.
- **robust scaling** uses statistics that ignore the tails.

All three are **linear** maps $x \mapsto (x - c)/d$, so they change units, not the shape of the distribution: a skewed variable stays exactly as skewed. Each one also divides by something that can be **zero**. A constant column has $\max = \min$ and $s = 0$, and a column where more than half the values are identical can have $\mathrm{IQR} = 0$. Decide and document what happens then.

**Logs change the shape.** $\log(ab) = \log a + \log b$, so a log turns multiplicative effects into additive ones, and $\log x_2 - \log x_1 = \log(x_2/x_1) \approx (x_2 - x_1)/x_1$ for small changes. Equal distances on the log scale are equal *ratios*. For positive, right-skewed quantities spanning orders of magnitude (prices, incomes, populations), the log scale is often closer to symmetric.
- $\log x$ needs $x > 0$.
- $\operatorname{log1p}(x) = \log(1 + x)$ also accepts $x = 0$ (it's defined for $x > -1$), and it behaves like $x$ when $x$ is near 0. Its inverse is $\operatorname{expm1}(y) = e^y - 1$.

> 🧰 **How the code works:** `np.log1p(s)` and `np.expm1(s)` work element-wise on a Series and keep `NaN`. `s.skew()` is the sample *skewness*: the standardized third moment, positive for a long right tail and about 0 for a symmetric distribution.
""")
nb.md(r"""
✍️ **3.1** Write `min_max(s)` and `z_score(s)` (sample standard deviation, `ddof=1`). Both keep missing values missing and keep the index. When the denominator is 0, or undefined because there's a single value, return `0.0` for every observed value.
""")
nb.learner("scalers", r"""
def min_max(s: pd.Series) -> pd.Series:
    # your code here
    raise NotImplementedError


def z_score(s: pd.Series) -> pd.Series:
    # your code here
    raise NotImplementedError


check("scalers", min_max, z_score)
""")
nb.md(r"""
★ **3.2** Write `robust_scale(s)`. When the IQR is 0, the scale is undefined, so raise a `ValueError` that says so. Returning `inf` or a column of `NaN` would hide the problem.
""")
nb.learner("robust-scale", r"""
def robust_scale(s: pd.Series) -> pd.Series:
    # your code here
    raise NotImplementedError


check("robust-scale", robust_scale)
""")
nb.md(r"""
🔮 **3.3** The skewness of `items["price"]` is about 7.9. What is the skewness of `np.log1p(items["price"])`?
**a)** still above 5  **b)** between −1 and 1  **c)** below −5
""")
nb.learner("log-skew", r"""
predict("log-skew", ...)   # "a", "b" or "c"
""")
nb.code(r"""
print("skewness: raw", round(items["price"].skew(), 2), "| log1p", round(np.log1p(items["price"]).skew(), 2))
compare = pd.DataFrame({"raw": items["price"].describe(), "log1p": np.log1p(items["price"]).describe()})
compare.round(2)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 4 · Discretization: `cut` vs `qcut`

*Discretizing* (binning) turns a numeric variable into intervals. Two ways to choose the edges:
- **`pd.cut(x, bins=[...])`**: **fixed edges** you choose because they mean something (a free-shipping threshold, an age bracket). The bins can have very different sizes.
- **`pd.qcut(x, q=4)`**: edges at the **sample quantiles**, so every bin gets roughly the same number of rows (*equal-frequency* bins). The edges are estimated from the data. They change with the sample, and a model must estimate them on training data only (section 6).

Binning always throws information away: 101 BRL and 149 BRL land in the same bin, while 149 and 151 don't. Use it when the categories mean something, or for display, not by default.

> ⚠️ **Ties.** If many rows share a value, several quantiles can coincide, and `qcut` refuses to build empty bins: `ValueError: Bin edges must be unique`. `duplicates="drop"` merges the repeated edges, so you silently get **fewer** bins than you asked for.
""")
nb.md(r"""
🔮 **4.1** `pd.qcut(order_table["n_items"], 4, duplicates="drop")`: how many bins come out?
""")
nb.learner("qcut-ties", r"""
predict("qcut-ties", ...)
""")
nb.code(r"""
print("quantiles of n_items:", order_table["n_items"].quantile([0, .25, .5, .75, 1]).tolist())
pd.qcut(order_table["n_items"], 4, duplicates="drop").value_counts(sort=False)
""")
nb.md(r"""
✍️ **4.2** Write `quartile_bins(s)`, which returns `pd.qcut` quartile labels `"Q1"`…`"Q4"` for a numeric Series, with missing values staying missing. The check also applies it to `paid_value`.
""")
nb.learner("quartile-bins", r"""
def quartile_bins(s: pd.Series) -> pd.Series:
    # your code here
    raise NotImplementedError


check("quartile-bins", quartile_bins)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 5 · Encoding categorical variables

Most models need numbers. How you turn categories into numbers depends on the variable's *scale of measurement*:
- **Nominal** (no order: payment type, state). Use **one-hot encoding**: one 0/1 *indicator* (or *dummy*) column per level. Numbering the levels 0, 1, 2 instead would invent an order (boleto < card < voucher?) *and* equal distances between levels. A model would happily use both.
- **Ordinal** (ordered: review score, size S < M < L). Integer codes that follow the order are reasonable. Keep in mind that they still assume equal steps between levels.

**The dummy-variable trap.** Suppose a model has an intercept, i.e. a column of ones, plus one dummy per level. Its *design matrix*, the matrix whose columns are the model's inputs, is

$$X = \big[\,\mathbf 1 \;\big|\; D_1 \;\big|\; \cdots \;\big|\; D_k\,\big] \in \mathbb R^{n \times (k+1)}.$$

Every row has exactly one $D_j = 1$, so $D_1 + \cdots + D_k = \mathbf 1$. One column is a linear combination of the others, so $\operatorname{rank}(X) \le k$ and $X^\top X$ is singular. Least squares then has **infinitely many** solutions: add $c$ to the intercept and subtract $c$ from every dummy's coefficient, and the fitted values don't change. The fix is to **drop one level** (`drop_first=True`). That level becomes the *reference category*, and each remaining coefficient measures a difference from it. (Some methods, such as tree-based models or regularized regression, are fine with all $k$ columns. Modules III–VI come back to this.)

> 🧰 **How the code works:** `pd.get_dummies(s, prefix="pay", dtype="int8")` builds the indicator columns, ordering the levels alphabetically, and `drop_first=True` drops the first one. Missing values get no column at all unless you add `dummy_na=True`, or label them first (Notebook 06).
""")
nb.code(r"""
pay_toy = toy_payments_type()
dummies = pd.get_dummies(pay_toy["payment_type"], prefix="pay", dtype="int8")
X = np.column_stack([np.ones(len(pay_toy)), dummies.to_numpy()])
display(pd.DataFrame(X, columns=["intercept", *dummies.columns]).astype(int))
""")
nb.md(r"""
🔮 **5.1** `X` has 4 columns. What is its rank, `np.linalg.matrix_rank(X)`?
""")
nb.learner("dummy-rank", r"""
predict("dummy-rank", ...)
""")
nb.code(r"""
print("rank with all dummies:", np.linalg.matrix_rank(X))
X_ref = np.column_stack([np.ones(len(pay_toy)),
                         pd.get_dummies(pay_toy["payment_type"], drop_first=True, dtype="int8").to_numpy()])
print("rank after dropping one level:", np.linalg.matrix_rank(X_ref), "of", X_ref.shape[1], "columns")
""")
nb.md(r"""
✍️ **5.2** Write `one_hot(df, col, drop_first=False)`. It returns a copy of `df` without `col`, with one `int8` column `f"{col}_{level}"` per level, levels in sorted order and placed after the other columns. Add a final `f"{col}_MISSING"` column only if `col` has missing values. `drop_first=True` drops the first level's column (never the MISSING one).
""")
nb.learner("one-hot", r"""
def one_hot(df: pd.DataFrame, col: str, drop_first: bool = False) -> pd.DataFrame:
    # your code here
    raise NotImplementedError


check("one-hot", one_hot)
""")
nb.md(r"""
★ **5.3** Write `ordinal_codes(s, order)`, which maps each value to its position in `order` (0, 1, 2, …). Values that are missing **or not in `order`** become `<NA>`, with dtype `"Int64"`. Then work out why `pd.Categorical(s, categories=order, ordered=True).codes` would be a dangerous shortcut here.
""")
nb.learner("ordinal-codes", r"""
def ordinal_codes(s: pd.Series, order: list) -> pd.Series:
    # your code here
    raise NotImplementedError


check("ordinal-codes", ordinal_codes)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 6 · Feature engineering and data leakage

**Feature engineering** builds new variables that express what you actually care about. A good feature has a clear meaning, a unit, and a population where it's defined:
- **ratios**: freight relative to the products' value. A denominator of 0 produces `inf`, and a missing one `NaN`. Make the rule explicit with `num / den.where(den > 0)`.
- **date parts**: `.dt.dayofweek` (0 = Monday), `.dt.hour`, `.dt.month`, `.dt.to_period("M")`.
- **durations**: differences of timestamps, defined only when both events happened (Notebook 02).
""")
nb.code(r"""
when = order_table["order_purchase_timestamp"]
parts = pd.DataFrame({"weekday": when.dt.day_name(), "hour": when.dt.hour, "month": when.dt.to_period("M")})
display(parts.head(3))
print("busiest weekday:", parts["weekday"].value_counts().idxmax(), "| busiest hour:", parts["hour"].value_counts().idxmax())
""")
nb.md(r"""
✍️ **6.1** Write `freight_ratio(df)` = `freight_value / items_value`, `NaN` where `items_value` is missing or not positive, and never `inf`. Then compute `share_freight_exceeds`: among orders of `order_table` **with a defined ratio**, the share where freight cost more than the products (ratio > 1).
""")
nb.learner("freight-ratio", r"""
def freight_ratio(df: pd.DataFrame) -> pd.Series:
    # your code here
    raise NotImplementedError


share_freight_exceeds = ...

check("freight-ratio", freight_ratio, share_freight_exceeds)
""")
nb.md(r"""
**Data leakage** happens when information from the data you'll *evaluate* on, or from the *future*, sneaks into the data you *build* with. In prediction (Modules V–VI) you split the data into a **training set**, used to fit everything, and a **test set**, held back to measure how well the result generalizes. Preprocessing is part of "everything": a z-score estimates $\bar x$ and $s$, an imputer estimates a median, `qcut` estimates quantiles. Those estimates are **fitted parameters**. If you compute them on all the data, the test set has influenced the model, and the evaluation becomes optimistic.

The rule: **fit on training data, then apply the same fitted parameters to the test data.**

★ **6.2** Take delivered orders with a delivery date, and compute the delivery time in days. **Train** = purchases before 2018-01-01, **test** = from 2018-01-01 on. Standardize the test durations twice:
- *leaky*: with the mean and standard deviation of **all** the rows
- *honest*: with those of the **training** rows only

Report the mean of each set of test z-scores (`z_test_mean_leaky`, `z_test_mean_honest`).
""")
nb.learner("leakage-gap", r"""
z_test_mean_leaky = ...
z_test_mean_honest = ...

check("leakage-gap", z_test_mean_leaky, z_test_mean_honest)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 7 · Pipelines and provenance

A **cleaning pipeline** is a function: raw table in, new table out. Writing it that way gives you three things:
- **It doesn't mutate its input.** Start with `df.copy()` or use `.assign(...)`. Callers keep their raw data, and you can run it twice and get the same result.
- **It records its decisions** in a *decision log*: one row per step, saying which rule it applied, how many rows it affected, what it did, and what the limitation is.
- **It checks invariants** (Notebook 01): same number of rows, key still unique.

**Provenance** answers "where did this number come from?": the exact **data**, the **code**, the **environment** (library versions) and the **decisions**. For the data, a **fingerprint** works well. SHA-256 (*Secure Hash Algorithm*, 256 bits) maps any file to a 64-hex-digit digest. Changing a single byte changes the digest completely, so equal digests mean, for all practical purposes, identical files. The downloader in `selfgrade.data` uses exactly this to verify each CSV against the digests stored in `OLIST_FILES`.
""")
nb.md(r"""
🐞 **7.1** A colleague capped prices "just for a quick chart", then looked at the raw data. Run the cell. Why is the raw maximum 100?
""")
nb.code(r"""
def cap_prices(df, upper):
    df["price"] = df["price"].clip(upper=upper)
    return df


raw_items = items[["order_id", "price"]].head(1000).copy()
capped = cap_prices(raw_items, 100)
print("max price in the raw table:", raw_items["price"].max())
""")
nb.md(r"""
💬 **7.2** What went wrong, and what is the one-line fix?
""")
nb.answer("debug-mutation")
nb.md(r"""
✍️ **7.3** Write `data_fingerprint(path, n=12)`, which returns the first `n` hexadecimal characters of the file's SHA-256 digest. Read the file in binary mode, in blocks, so it also works on files larger than memory.
""")
nb.learner("data-fingerprint", r"""
def data_fingerprint(path, n: int = 12) -> str:
    # your code here
    raise NotImplementedError


check("data-fingerprint", data_fingerprint)
""")
nb.code(r"""
# A provenance stamp: paste it under any result you share
used = ["orders", "items", "payments", "customers"]
for name in used:
    fp = data_fingerprint(olist_path(name))
    status = "matches the published file" if OLIST_FILES[name][2].startswith(fp) else "DIFFERS from the published file"
    print(f"{OLIST_FILES[name][0]:<36} {fp}  ({status})")
print(f"python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}")
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 8 · Mini-project: a cleaning pipeline for the order table

✍️ **8.1** Write `clean_orders(df)` for a table shaped like `order_table`. It returns a tuple `(clean, log)`.

`clean` is a **copy** of `df` (same rows, same order, all original columns) with five new columns:

| column | definition |
|---|---|
| `paid_missing` | `paid_value` is missing (a *missing indicator*, Notebook 06) |
| `no_items` | `n_items == 0` |
| `freight_ratio` | your safe ratio from 6.1 |
| `log_paid` | `log1p(paid_value)` |
| `paid_outlier` | outside the Tukey fences $[Q_1 - 1.5\,\mathrm{IQR},\ Q_3 + 1.5\,\mathrm{IQR}]$ of `log_paid` (missing → `False`) |

`log` is a DataFrame with columns `step, rule, rows_affected, action, limitation`, one row per new column, where `step` is the column's name. `rows_affected` counts the `True` values for the three flags, and the `NaN` values for `freight_ratio` and `log_paid`. Write a short, real `action` and `limitation` for each step.
""")
nb.learner("clean-orders", r"""
def clean_orders(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # your code here
    raise NotImplementedError


check("clean-orders", clean_orders)
""")
nb.code(r"""
clean, decision_log = clean_orders(order_table)
assert len(clean) == len(order_table) and clean["order_id"].equals(order_table["order_id"])
display(decision_log)
clean.groupby("paid_outlier")["paid_value"].describe().round(1)
""")
nb.md(r"""
💬 **8.2** Why are the fences computed on `log_paid` rather than on `paid_value`? Compare how many orders each version flags (the check's 💡 tells you), and say which one you'd defend in a report.
""")
nb.answer("pipeline-decisions")
nb.md(r"""
★ **8.3** *(open, no auto-check)* Write an **outlier policy** for this table in 4–5 lines: what gets flagged, what gets excluded from which analyses (and why), what is never deleted, and how you'd show readers that the conclusions don't depend on it.
""")
nb.answer("outlier-policy")

# ---------------------------------------------------------------------------
nb.md(r"""
## Wrap-up

💬 Answer briefly, in your own words.

1. What's the difference between an error and an outlier, and why does it change what you do with the row?
2. Clipping at the 99th percentile lowered the mean price but not the median. Explain why in two sentences.
3. Why doesn't standardizing a skewed variable make it symmetric, and what would?
4. Why is numbering nominal categories 0, 1, 2 a problem, and what's the dummy-variable trap?
5. Name two preprocessing steps from this notebook that leak information if you fit them on all the data.
""")
nb.answer("wrap-up", "> Your answers:")
nb.code(r"""
progress()
""")
nb.md(r"""
**This closes Module I.** You can now read, join, summarize, visualize, and clean tabular data, and defend every step. If you use Claude Code with this repository, `/tutor review m01/07` reads your notebook and gives you feedback.
""")
