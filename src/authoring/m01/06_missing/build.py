"""Builder: Module I · 06 — Missing data."""
from nbbuild import Notebook

COURSE = "https://github.com/chaffrey8/DMEMCD_2026"

nb = Notebook(
    nb_id="m01/06",
    filename="06_missing-data.ipynb",
    title="Missing data",
    subtitle="What a gap means, why it went missing, and what filling it costs",
    grader="selfgrade.m01.missing",
    can_do=["predict how pandas treats NaN, None, pd.NA and NaT in sums, comparisons, groupbys and joins",
            "profile missingness, overall and conditional on another column",
            "state MCAR, MAR and MNAR precisely, and say what data can and can't reveal about them",
            "measure the bias and the cost of complete-case analysis",
            "impute by group with a fallback, and explain why single imputation shrinks variance"],
    builds_on="01 (keys, joins, `category_label`) and 02 (`groupby`, `agg` vs `transform`). Both are used here "
              "as tools.",
    effort="≈ 2 h",
    sources=[f"Diplomado *Métodos Estadísticos y Matemáticos para la Ciencia de Datos* (UNAM, 2026), Module I: "
             f"[`04_Limpieza_Transformacion_Work_Olist` (§2–4), `03_Visualizaciones_Work_Olist` (missingness map), "
             f"`S03_Notebook_Demostracion_Python`]({COURSE})"],
    concept_map=["how pandas represents missing", "profiling: where and with what", "mechanisms: MCAR, MAR, MNAR",
                 "complete-case analysis", "single imputation and its cost",
                 "mini-project: who writes a comment?"],
)

# ---------------------------------------------------------------------------
nb.md(r"""
## 0 · Setup

The Olist tables are introduced in Notebook 01. Here we use `orders`, `items`, `products`, `categories` and `reviews`, plus `analysis`: one row per item of a **delivered** order, with its purchase date and a readable `category` (English name, else Portuguese name, else `"UNKNOWN"`). That's the `category_label` you built in Notebook 01.
""")
nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from selfgrade.data import load_olist
from selfgrade.m01.missing import check, hint, predict, progress
from selfgrade.m01.missing import toy_gaps, toy_weights

orders, items, reviews = load_olist("orders"), load_olist("items"), load_olist("reviews")
products, categories = load_olist("products"), load_olist("categories")

# analysis: delivered item rows + a clean category (see Notebook 01)
labels = products[["product_category_name"]].merge(categories, on="product_category_name", how="left",
                                                   validate="many_to_one")
category = labels["product_category_name_english"].fillna(labels["product_category_name"]).fillna("UNKNOWN")
category.index = products.index

delivered_orders = orders.loc[orders["order_status"].eq("delivered"), ["order_id", "order_purchase_timestamp"]]
product_labels = products[["product_id"]].assign(category=category)
analysis = (
    items.merge(delivered_orders, on="order_id", how="inner", validate="many_to_one")
         .merge(product_labels, on="product_id", how="left", validate="many_to_one")
)
print("analysis:", analysis.shape)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 1 · How pandas represents "missing"

A missing value is a cell where the data has no value. That's different from a value of zero, an empty string, or the text `"unknown"`. pandas uses four markers, depending on the column's type:

| Marker | Where it appears |
|---|---|
| `np.nan` (*not a number*) | float columns. It's a special floating-point value, so an **integer column with a single gap becomes `float64`**. |
| `None` | Python's "nothing", in `object` (text) columns |
| `pd.NA` | pandas' *nullable* dtypes: `Int64`, `boolean`, `string` (capitalized names). They hold gaps without changing type. |
| `pd.NaT` (*not a time*) | datetime columns |

`s.isna()` (and its opposite `s.notna()`) recognizes all four. How the common operations treat them:

- **Reductions skip them.** `sum`, `mean`, `median`, `std` and friends use `skipna=True`, so `mean` divides by the number of *observed* values. One surprise: the sum of an all-missing column is `0`. Pass `min_count=1` to get `NaN` instead.
- **Comparisons are `False`.** `np.nan == np.nan` is `False`, so `s == np.nan` never finds anything. With `pd.NA` the comparison returns `<NA>` instead, which is three-valued logic.
- **`groupby` and `value_counts` drop missing keys** unless you pass `dropna=False`.
- **`merge` matches missing keys to each other**, unlike SQL, where `NULL` never equals `NULL`.

> 🧰 **How the code works:** `s.isna().sum()` counts gaps and `s.isna().mean()` gives their share (the mean of a 0/1 indicator is a proportion). `s.astype("Int64")` converts a float column of whole numbers with gaps into a nullable integer column.

> ✅ **Good practice:** test for missing with `.isna()`, never with `== np.nan`. Before a join, decide what missing keys should do. Before a sum, decide whether "all missing" should mean 0 or unknown.
""")
nb.code(r"""
t = pd.Series([1.0, np.nan, 3.0])
t
""")
nb.md(r"""
🔮 **1.1** What is `(t.sum(), t.mean(), (t == np.nan).sum())`? Answer as a tuple of three numbers.
""")
nb.learner("missing-semantics", r"""
predict("missing-semantics", ...)
""")
nb.code(r"""
print("sum:", t.sum(), "| mean:", t.mean(), "| == np.nan finds:", (t == np.nan).sum(), "| isna finds:", t.isna().sum())
print("sum of an all-missing Series:", pd.Series([np.nan, np.nan]).sum(),
      "| with min_count=1:", pd.Series([np.nan, np.nan]).sum(min_count=1))
""")
nb.md(r"""
🔮 **1.2** Two tiny tables share a key column `k`. How many rows does the inner join `left.merge(right, on="k")` return?
""")
nb.code(r"""
left = pd.DataFrame({"k": ["a", np.nan], "x": [1, 2]})
right = pd.DataFrame({"k": ["a", np.nan, np.nan], "y": [10, 20, 30]})
display(left, right)
""")
nb.learner("merge-nan-keys", r"""
predict("merge-nan-keys", ...)
""")
nb.code(r"""
left.merge(right, on="k")
""")
nb.md(r"""
✍️ **1.3** In `products`, `product_photos_qty` (number of photos in the listing) is a count, but pandas stored it as `float64`. Create `photos`: the same column as a **nullable integer** Series, with gaps kept as missing.
""")
nb.learner("nullable-int", r"""
photos = ...

check("nullable-int", photos)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 2 · Profiling: where are the gaps, and with what?

Before deciding anything, measure. A **missingness profile** gives the count and share of missing values per column. Two refinements make it far more informative:

- **Conditional profiles.** Compute the share missing *within each value of another column*, for example within each order status. A missing delivery date means something completely different for a canceled order than for a delivered one.
  - A gap is **structural** (*not applicable*) when the value cannot exist: a canceled order has no delivery date, because nothing was delivered.
  - A gap is **unknown** when the value exists in the real world but wasn't recorded: a delivered order without a delivery date.

  Only unknown values are candidates for imputation. Imputing a structural gap invents an event.
- **Co-missingness.** Columns that go missing *together* usually share a single cause, such as a form that was never filled in or a system that didn't send a block of fields.

Let $R_{ij}=1$ if the value of variable $j$ for row $i$ is missing, and $0$ otherwise. This **missingness indicator** matrix $R$ is just `df.isna()`. Its column means are the shares missing, and a picture of it shows the patterns:
""")
nb.code(r"""
toy = toy_gaps()
display(toy, toy.dtypes.to_frame("dtype"))
""")
nb.md(r"""
✍️ **2.1** Write `missing_profile(df)`. It returns a DataFrame indexed by column name, with columns `n_missing` and `share_missing`, sorted by `n_missing` from most to fewest. Ties keep the original column order.
""")
nb.learner("missing-profile", r"""
def missing_profile(df: pd.DataFrame) -> pd.DataFrame:
    # your code here
    raise NotImplementedError


check("missing-profile", missing_profile)
""")
nb.md(r"""
✍️ **2.2** For each `order_status`, what share of orders has a missing `order_delivered_customer_date`? Return a Series `missing_by_status` indexed by status.
""")
nb.learner("missing-by-status", r"""
missing_by_status = ...

check("missing-by-status", missing_by_status)
""")
nb.md(r"""
★ **2.3** The status and the date should agree. Count:
- `n_delivered_no_date`: delivered orders **without** a delivery date
- `n_not_delivered_with_date`: orders with **any other status** that **have** one
""")
nb.learner("status-date-audit", r"""
n_delivered_no_date = ...
n_not_delivered_with_date = ...

check("status-date-audit", n_delivered_no_date, n_not_delivered_with_date)
""")
nb.md(r"""
A picture of $R$: a sample of up to 40 orders per status, one column per order, dark where a value is missing. The rows are the variables. (The status sets the pattern: compare `delivered` with everything else.)
""")
nb.code(r"""
sample = orders.groupby("order_status").head(40).sort_values("order_status")
fig, ax = plt.subplots(figsize=(10, 3))
ax.imshow(sample.isna().T.to_numpy(), aspect="auto", cmap="Greys", interpolation="nearest")
ax.set_yticks(range(sample.shape[1]), sample.columns, fontsize=8)
starts = sample["order_status"].ne(sample["order_status"].shift()).to_numpy().nonzero()[0]
ax.set_xticks(starts, sample["order_status"].iloc[starts], rotation=30, ha="left", fontsize=8)
ax.set_title("Missingness map of orders (dark = missing)")
plt.show()
""")
nb.md(r"""
✍️ **2.4** In `products`, four columns describe the listing: `product_category_name`, `product_name_lenght`, `product_description_lenght` and `product_photos_qty`. (The misspelling "lenght" is in the original data.) Count:
- `n_all_missing`: products missing **all four**
- `n_some_missing`: products missing **at least one but not all**
""")
nb.learner("missing-together", r"""
n_all_missing = ...
n_some_missing = ...

check("missing-together", n_all_missing, n_some_missing)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 3 · Why is it missing? MCAR, MAR and MNAR

Whether a gap harms an analysis depends on **why** it's missing. Write the data as $Y=(Y_{\text{obs}},Y_{\text{mis}})$, its observed and missing parts, and let $R$ be the missingness indicator from section 2. The **missingness mechanism** is the probability distribution of $R$ given the data, $P(R\mid Y_{\text{obs}},Y_{\text{mis}})$. There are three classical cases (Rubin, 1976):

| Mechanism | Definition | In words | Example |
|---|---|---|---|
| **Missing completely at random (MCAR)** | $P(R\mid Y_{\text{obs}},Y_{\text{mis}})=P(R)$ | the gaps are unrelated to any data, seen or unseen | a sensor that randomly drops 1 % of its readings |
| **Missing at random (MAR)** | $P(R\mid Y_{\text{obs}},Y_{\text{mis}})=P(R\mid Y_{\text{obs}})$ | the gaps depend only on things you *do* observe | unhappy customers (observed score) write comments more often |
| **Missing not at random (MNAR)** | depends on $Y_{\text{mis}}$ even given $Y_{\text{obs}}$ | the gaps depend on the missing value itself | people with high incomes decline to report their income |

**Intuition.** Under MCAR, the complete rows are a random subsample, so dropping the incomplete rows costs precision but not correctness. Under MAR, the complete rows are tilted, but the tilt runs through observed variables, so you can correct it by conditioning or reweighting on them (section 6 does exactly that). Under MNAR, the tilt runs through what you can't see, and no amount of observed data fixes it by itself.

**What the data can tell you.** Compare *observed* variables between rows with and without a gap. If they differ, the gaps aren't MCAR. (Little's test formalizes this comparison.) But no check on the observed data can tell MAR from MNAR, because the deciding information is exactly what's missing. That takes knowledge of the process that produced the data, or extra data.

Structural (*not applicable*) gaps sit outside this framework: there's no true value to recover.
""")
nb.md(r"""
★ **3.1** *Is missingness random?* Products without a category end up as `"UNKNOWN"` in `analysis`. Among delivered item rows, compute the share of **rows** with `category == "UNKNOWN"` and the share of **revenue** (`price`) they represent. If missingness were MCAR, the two should be about equal.
""")
nb.learner("unknown-share", r"""
share_rows_unknown = ...
share_revenue_unknown = ...

check("unknown-share", share_rows_unknown, share_revenue_unknown)
""")
nb.md(r"""
💬 **3.2** The gap between your two shares is small. Does that prove the missing categories are MCAR? What else would you compare between products with and without a category, and what could never be settled from the data?
""")
nb.answer("mcar-evidence")

# ---------------------------------------------------------------------------
nb.md(r"""
## 4 · Complete-case analysis

**Complete-case analysis**, also called **listwise deletion**, keeps only the rows with no gaps in the variables you use. It's the default behind `dropna()` and behind many model-fitting functions. **Pairwise deletion** instead uses, for each statistic, every row where *that statistic's* variables are observed. `df.corr()` works that way, so different entries of a correlation matrix can come from different rows.

- **Cost.** If each of $k$ columns is missing independently with probability $p$, a row is complete with probability $(1-p)^k$. That falls fast as $k$ grows.
- **Bias.** Under MCAR, the complete rows are a random subsample, so estimates stay unbiased and only get noisier. Under MAR or MNAR, they generally aren't.

> ✅ **Good practice:** drop by the variables your question uses (`dropna(subset=[...])`), never with a blanket `dropna()`. Report how many rows were dropped and why.
""")
nb.md(r"""
🔮 **4.1** A table has 10 columns, each missing independently 5 % of the time. What share of rows survive `dropna()`? Answer with a number between 0 and 1, to two decimals.
""")
nb.learner("complete-rows", r"""
predict("complete-rows", ...)
""")
nb.md(r"""
✍️ **4.2** On `products`: how many rows does a plain `dropna()` remove (`n_dropped_any`), and how many if you only need `product_weight_g` (`n_dropped_weight`)?
""")
nb.learner("listwise-loss", r"""
n_dropped_any = ...
n_dropped_weight = ...

check("listwise-loss", n_dropped_any, n_dropped_weight)
""")
nb.md(r"""
★ **4.3** *A simulated MNAR.* `product_weight_g` is nearly complete, so we can create a known mechanism. Suppose the system had failed to record **every weight above 5,000 g**: the gap depends on the value itself. Compute the complete-case mean and median of what would remain (`cc_mean`, `cc_median`), and compare them with the full column's `mean()` and `median()`.
""")
nb.learner("mnar-bias", r"""
w = products["product_weight_g"]
cc_mean = ...
cc_median = ...

check("mnar-bias", cc_mean, cc_median)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 5 · Single imputation and what it costs

**Imputation** replaces each gap with a plausible value. *Single* imputation uses one value per gap. (*Multiple imputation* creates several completed datasets to carry the uncertainty forward; it's beyond this module.) Common choices:

| Method | Uses | Fits |
|---|---|---|
| global mean / median | the whole column | MCAR, numeric |
| **group median** | the median within a group defined by an *observed* variable | MAR given that variable |
| mode (most frequent value) | the whole column | rarely a good idea for categories (see ★ 5.6) |
| explicit level, e.g. `"UNKNOWN"` | nothing: the gap becomes a visible category | categorical variables |

Always add a **missing-value indicator** column (`was_missing`, True where you filled). It keeps the information that the value was a guess, so a later analysis or model can use it or exclude it.

**The variance cost, derived.** Fill the $m$ gaps of a numeric column with the observed mean $\bar x_{\text{obs}}$. Then $n=n_{\text{obs}}+m$. The filled values sit exactly at the mean, so the mean doesn't change, and they add *zero* to the sum of squared deviations $\sum_i (x_i-\bar x)^2=(n_{\text{obs}}-1)\,s^2_{\text{obs}}$. Only the divisor grows:

$$s^2_{\text{imputed}}=\frac{(n_{\text{obs}}-1)\,s^2_{\text{obs}}}{n-1}=s^2_{\text{obs}}\cdot\frac{n_{\text{obs}}-1}{n-1}.$$

The variability is understated by the share of gaps, roughly. Correlations with other columns shrink toward 0 for the same reason, and any test or interval computed on the filled column is overconfident.

> ⚠️ When a model will be trained on the data, the imputer's statistics (medians, modes) must be computed on the **training** data only and then applied to new data. Otherwise information leaks from the evaluation data into the model. More on this in later modules.
""")
nb.code(r"""
x = pd.Series([2.0, 4.0, 6.0, np.nan, np.nan])
print("observed variance:", x.var())
""")
nb.md(r"""
🔮 **5.1** What is `x.fillna(x.mean()).var()`?
""")
nb.learner("variance-shrink", r"""
predict("variance-shrink", ...)
""")
nb.code(r"""
print("after mean imputation:", x.fillna(x.mean()).var())
""")
nb.md(r"""
🐞 **5.2** A colleague wanted to fill missing weights with their category's median. The cell runs without errors, but two products still have no weight. Run it, find the cause, and explain it below.
""")
nb.code(r"""
fixed = products.copy()
category_medians = fixed.groupby("product_category_name")["product_weight_g"].median()
fixed["product_weight_g"] = fixed["product_weight_g"].fillna(category_medians)
print("missing weights after imputation:", fixed["product_weight_g"].isna().sum())
""")
nb.md(r"""
💬 **5.3** Why did nothing get filled? (Hint: what is the index of `category_medians`, and what does `fillna` do with a Series?) Which `groupby` method returns a result aligned to the original rows?
""")
nb.answer("debug-fillna-alignment")
nb.md(r"""
✍️ **5.4** Write `impute_by_group(df, value_col, group_col)`. It returns a **copy** of `df` where:
- each gap in `value_col` is filled with the median of its group in `group_col`
- rows whose group has no observed value, or no group at all, get the **overall** median
- a new boolean column `f"{value_col}_was_missing"` marks the filled rows

Keep `df`'s index, and don't modify `df`.
""")
nb.code(r"""
toy_weights()
""")
nb.learner("impute-group-median", r"""
def impute_by_group(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    # your code here
    raise NotImplementedError


check("impute-group-median", impute_by_group)
""")
nb.md(r"""
✍️ **5.5** Check the formula on real data. For `products["product_weight_g"]`, compute the sample variance of the observed values (`var_observed`) and after mean imputation (`var_mean_imputed`).
""")
nb.learner("shrinkage-check", r"""
var_observed = ...
var_mean_imputed = ...

check("shrinkage-check", var_observed, var_mean_imputed)
""")
nb.md(r"""
★ **5.6** *Mode imputation for categories.* Fill the 610 missing `product_category_name` values with the mode. What share of **all** products belongs to the mode category before (`share_top_before`) and after (`share_top_after`)?
""")
nb.learner("mode-inflation", r"""
share_top_before = ...
share_top_after = ...

check("mode-inflation", share_top_before, share_top_after)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 6 · Mini-project: who writes a comment, and does it matter?

Reviews have a score from 1 to 5, which is always present, and an optional text comment, which is missing for most reviews. Suppose a team plans to study customer experience by reading the comments. Is that sample representative?

The table `reviewed` below has one row per review of a delivered order with a known delivery date. Its columns are `days`, the delivery time from purchase to delivery, and `has_comment`.
""")
nb.code(r"""
dv = orders[orders["order_status"].eq("delivered") & orders["order_delivered_customer_date"].notna()]
days = (dv["order_delivered_customer_date"] - dv["order_purchase_timestamp"]).dt.total_seconds() / 86400
reviewed = (reviews[["review_id", "order_id", "review_score", "review_comment_message"]]
            .merge(dv[["order_id"]].assign(days=days.to_numpy()), on="order_id", how="inner",
                   validate="many_to_one")
            .assign(has_comment=lambda d: d["review_comment_message"].notna()))
print(reviewed.shape, "| share with a comment:", round(reviewed["has_comment"].mean(), 3))
""")
nb.md(r"""
🔮 **6.1** Which review score (1–5) has the **highest** share of reviews with a written comment?
""")
nb.learner("comment-who", r"""
predict("comment-who", ...)
""")
nb.md(r"""
✍️ **6.2** On the full `reviews` table: the share of reviews with a comment for each `review_score`, as a Series `comment_by_score` indexed by score.
""")
nb.learner("comment-by-score", r"""
comment_by_score = ...

check("comment-by-score", comment_by_score)
""")
nb.md(r"""
✍️ **6.3** Using `reviewed`: the mean delivery time over all reviews (`mean_days_all`), and over reviews **with** a comment only (`mean_days_commented`). The second one is what a comments-only study would report.
""")
nb.learner("cc-bias", r"""
mean_days_all = ...
mean_days_commented = ...

check("cc-bias", mean_days_all, mean_days_commented)
""")
nb.md(r"""
**Correcting a MAR gap by reweighting.** Suppose whether a review has a comment depends only on its score (MAR given the score). A commented 5-star review then stands in for all the 5-star reviews that look like it, and it should count $1/\hat\pi(5)$ times, where $\hat\pi(s)$ is the share of reviews with score $s$ that have a comment. The **inverse probability weighting** (IPW) estimate of the mean of $y$ uses only the observed (commented) rows:

$$\hat\mu_{\text{IPW}}=\frac{\sum_{i\,\text{observed}} w_i\,y_i}{\sum_{i\,\text{observed}} w_i},\qquad w_i=\frac{1}{\hat\pi(\text{score}_i)}.$$

Rarely-observed kinds of rows get large weights, which makes up for how seldom they appear.

★ **6.4** Compute `ipw_mean_days`: the IPW estimate of the mean delivery time from the commented rows of `reviewed`, with $\hat\pi$ estimated per score within `reviewed`. How close does it get to `mean_days_all`?
""")
nb.learner("ipw-days", r"""
ipw_mean_days = ...

check("ipw-days", ipw_mean_days)
""")
nb.md(r"""
💬 **6.5** Is "comment missing" MCAR, MAR or MNAR with respect to the customer's satisfaction? Argue both ways: what does the score capture, and what might it miss? What would you tell the team planning the comments study?
""")
nb.answer("comments-mechanism")

# ---------------------------------------------------------------------------
nb.md(r"""
## Wrap-up

💬 Answer briefly, in your own words.

1. Give one *structural* and one *unknown* gap from the Olist tables, and say how you'd treat each.
2. Define MCAR, MAR and MNAR in one sentence each. Which can you test from the data, and which not?
3. Why is complete-case analysis unbiased under MCAR but not under MAR?
4. What does mean imputation do to a column's variance, and why does that matter for later conclusions?
5. Why keep a `was_missing` indicator after imputing?
""")
nb.answer("wrap-up", "> Your answers:")
nb.code(r"""
progress()
""")
nb.md(r"""
**Next:** `07 · Cleaning, transforming, features`. If you use Claude Code with this repository, `/tutor review m01/06` reads your notebook and gives you feedback.
""")
