"""Grading for Module I · 06 — Missing data.

⚠️ SPOILERS: the reference computations used for grading live in this file.
Try the exercises first; read this afterwards if you want to compare approaches.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ..core import Session, as_int, expect, expect_close, is_df, is_series, unchanged
from ..data import load_olist

S = Session("m01/06", "Missing data")
check, hint, predict, progress = S.check, S.hint, S.predict, S.progress

LISTING_COLS = ["product_category_name", "product_name_lenght", "product_description_lenght", "product_photos_qty"]

# ===========================================================================
# Toy data
# ===========================================================================


def toy_gaps() -> pd.DataFrame:
    """Every flavour of 'missing' pandas knows, one column each."""
    return pd.DataFrame({
        "price": [10.0, np.nan, 30.0, 40.0],
        "seller": ["s1", None, "s3", None],
        "delivered_at": pd.to_datetime(["2018-01-02", None, "2018-01-05", "2018-01-09"]),
        "n_photos": pd.array([1, 2, pd.NA, pd.NA], dtype="Int64"),
    })


def toy_weights() -> pd.DataFrame:
    """Product weights with gaps: group 'a' has some values, 'b' has none, one row has no group."""
    return pd.DataFrame({
        "category": ["a", "a", "a", "b", "b", None, "c"],
        "weight": [100.0, np.nan, 300.0, np.nan, np.nan, np.nan, 1000.0],
    }, index=[10, 11, 12, 13, 14, 15, 16])


# ===========================================================================
# Reference implementations (used for grading)
# ===========================================================================


def _ref_profile(df: pd.DataFrame) -> pd.DataFrame:
    na = df.isna()
    out = pd.DataFrame({"n_missing": na.sum().astype(int), "share_missing": na.mean()})
    return out.sort_values("n_missing", ascending=False, kind="stable")


def _ref_impute(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    out = df.copy()
    group_median = out.groupby(group_col)[value_col].transform("median")
    out[f"{value_col}_was_missing"] = out[value_col].isna()
    out[value_col] = out[value_col].fillna(group_median).fillna(out[value_col].median())
    return out


def _ref_analysis(orders, items, products, categories) -> pd.DataFrame:
    m = products[["product_category_name"]].merge(categories, on="product_category_name", how="left",
                                                  validate="many_to_one")
    lab = m["product_category_name_english"].fillna(m["product_category_name"]).fillna("UNKNOWN")
    lab.index = products.index
    delivered = orders.loc[orders["order_status"] == "delivered", ["order_id", "order_purchase_timestamp"]]
    df = items.merge(delivered, on="order_id", how="inner", validate="many_to_one")
    prod = products[["product_id"]].assign(category=lab.to_numpy())
    return df.merge(prod, on="product_id", how="left", validate="many_to_one")


def _ref_reviewed(orders, reviews) -> pd.DataFrame:
    dv = orders[orders["order_status"].eq("delivered") & orders["order_delivered_customer_date"].notna()]
    days = (dv["order_delivered_customer_date"] - dv["order_purchase_timestamp"]).dt.total_seconds() / 86400
    return (reviews[["review_id", "order_id", "review_score", "review_comment_message"]]
            .merge(dv[["order_id"]].assign(days=days.to_numpy()), on="order_id", how="inner",
                   validate="many_to_one")
            .assign(has_comment=lambda d: d["review_comment_message"].notna()))


@lru_cache(maxsize=1)
def _real() -> dict:
    orders, items = load_olist("orders"), load_olist("items")
    products, categories, reviews = load_olist("products"), load_olist("categories"), load_olist("reviews")
    r: dict = {}
    r["photos_na"] = int(products["product_photos_qty"].isna().sum())
    r["profile_products"] = _ref_profile(products)
    st = orders["order_status"]
    nodate = orders["order_delivered_customer_date"].isna()
    r["missing_by_status"] = nodate.groupby(st).mean().sort_index()
    r["n_delivered_no_date"] = int((st.eq("delivered") & nodate).sum())
    r["n_not_delivered_with_date"] = int((st.ne("delivered") & ~nodate).sum())
    na = products[LISTING_COLS].isna()
    r["n_all_missing"] = int(na.all(axis=1).sum())
    r["n_some_missing"] = int((na.any(axis=1) & ~na.all(axis=1)).sum())

    an = _ref_analysis(orders, items, products, categories)
    unk = an["category"].eq("UNKNOWN")
    r["share_rows_unknown"] = float(unk.mean())
    r["share_revenue_unknown"] = float(an.loc[unk, "price"].sum() / an["price"].sum())

    r["n_dropped_any"] = len(products) - len(products.dropna())
    r["n_dropped_weight"] = len(products) - len(products.dropna(subset=["product_weight_g"]))
    r["imputed_weights"] = _ref_impute(products, "product_weight_g", "product_category_name")
    w = products["product_weight_g"]
    r["var_observed"] = float(w.var())
    r["var_mean_imputed"] = float(w.fillna(w.mean()).var())
    mode = products["product_category_name"].mode()[0]
    r["share_top_before"] = float(products["product_category_name"].eq(mode).mean())
    r["share_top_after"] = float(products["product_category_name"].fillna(mode).eq(mode).mean())
    kept = w.where(w <= 5000)
    r["mnar_mean"], r["mnar_median"] = float(kept.mean()), float(kept.median())

    rv = _ref_reviewed(orders, reviews)
    r["comment_by_score"] = reviews["review_comment_message"].notna().groupby(reviews["review_score"]).mean().sort_index()
    r["mean_days_all"] = float(rv["days"].mean())
    r["mean_days_commented"] = float(rv.loc[rv["has_comment"], "days"].mean())
    p = rv.groupby("review_score")["has_comment"].transform("mean")
    c = rv["has_comment"]
    r["ipw_days"] = float(np.average(rv.loc[c, "days"], weights=1 / p[c]))
    return r


def _close_series(got, want, name, tol=1e-6):
    is_series(got, name)
    g = got.sort_index()
    expect(list(map(str, g.index)) == list(map(str, want.index)),
           f"`{name}` should have one entry per {want.index.name or 'group'}: {list(want.index)}. Got {list(g.index)}.")
    expect(np.allclose(g.to_numpy(dtype=float), want.to_numpy(dtype=float), atol=tol),
           f"`{name}` has the right groups but wrong values.")


# ===========================================================================
# Predictions
# ===========================================================================


def _tuple_close(v, a, tol=1e-9):
    try:
        return len(v) == len(a) and all(abs(float(x) - float(y)) <= tol for x, y in zip(v, a))
    except (TypeError, ValueError):
        return False


S.prediction("missing-semantics", (4.0, 2.0, 0),
    "sum and mean skip NaN by default (skipna=True), so they act on [1, 3]: 4.0 and 2.0. Note that the mean "
    "divides by 2, not 3. And NaN is not equal to anything, not even NaN, so `t == np.nan` is all False: "
    "always test with .isna().",
    compare=_tuple_close)
S.prediction("merge-nan-keys", 3,
    "'a' matches once, and the left NaN matches BOTH right NaNs: 1 + 2 = 3 rows. Unlike SQL, where NULL never "
    "equals NULL, pandas treats missing keys as equal to each other. Drop or label missing keys before joining.")
S.prediction("complete-rows", 0.6,
    "If each column is missing independently with probability 0.05, a row is complete with probability "
    "0.95^10 ≈ 0.60. Listwise deletion would throw away 40% of the rows, although each column is 95% complete.",
    compare=lambda v, a: abs(float(v) - 0.5987) <= 0.02)
S.prediction("variance-shrink", 2.0,
    "The filled values sit exactly at the mean (4), so they add nothing to the sum of squared deviations (still 8) "
    "while the divisor grows from n_obs−1 = 2 to n−1 = 4: 8/4 = 2.0. The variance halves although no real "
    "value changed.",
    compare=lambda v, a: abs(float(v) - a) <= 0.01)
S.prediction("comment-who", 1,
    "Unhappy customers write. About three in four 1-star reviews come with a comment, against about one in "
    "three 5-star reviews. Whether the comment is missing depends strongly on the (observed) score.",
    compare=lambda v, a: str(v).strip() == "1")

# ===========================================================================
# Exercises
# ===========================================================================

# ---- 1. How pandas represents missing ---------------------------------------


@S.exercise("nullable-int", "a nullable integer column", hints=[
    "`product_photos_qty` is float64 only because it has gaps: NaN is a float.",
    "pandas' nullable integer dtype is spelled with a capital I: `s.astype('Int64')`.",
])
def _(photos):
    is_series(photos, "photos")
    expect(isinstance(photos.dtype, pd.api.extensions.ExtensionDtype) and pd.api.types.is_integer_dtype(photos.dtype),
           f"`photos` should have a nullable integer dtype such as 'Int64', got {photos.dtype}.")
    orig = load_olist("products")["product_photos_qty"]
    expect(len(photos) == len(orig), "Keep one value per product.")
    expect(int(photos.isna().sum()) == _real()["photos_na"],
           "The number of missing values changed. Missing values must stay missing (they show as <NA>).")
    expect((photos.dropna().to_numpy(dtype=float) == orig.dropna().to_numpy()).all(), "The observed values changed.")
    return ("Nullable dtypes (Int64, boolean, string) store gaps as pd.NA without forcing the column to float. "
            "Counts stay integers, and 3 photos never turns into 3.0.")


# ---- 2. Profiling ---------------------------------------------------------------


@S.exercise("missing-profile", "missing_profile(df)", hints=[
    "`df.isna()` is a boolean DataFrame. `.sum()` counts per column, and `.mean()` gives the share per column.",
    "Build a DataFrame with columns n_missing and share_missing, indexed by column name.",
    "Sort with `.sort_values('n_missing', ascending=False, kind='stable')`. 'stable' keeps the original "
    "column order among ties.",
])
def _(missing_profile):
    toy = toy_gaps()
    got = unchanged(missing_profile, toy)
    is_df(got, "missing_profile(...)")
    expect(list(got.columns) == ["n_missing", "share_missing"],
           f"Columns should be ['n_missing', 'share_missing'], got {list(got.columns)}.")
    want = _ref_profile(toy)
    expect(list(got.index) == list(want.index),
           f"Toy check: expected the columns ordered {list(want.index)} (most missing first, ties in original "
           f"order). Got {list(got.index)}.")
    expect(list(got["n_missing"].astype(int)) == list(want["n_missing"]),
           f"Toy check: n_missing should be {list(want['n_missing'])}. Do None, NaT and pd.NA all count?")
    expect(np.allclose(got["share_missing"].to_numpy(dtype=float), want["share_missing"].to_numpy()),
           "Toy check: share_missing should be n_missing / number of rows.")
    real = missing_profile(load_olist("products"))
    want = _real()["profile_products"]
    expect(list(real.index) == list(want.index) and list(real["n_missing"].astype(int)) == list(want["n_missing"]),
           "Toy data passes but `products` doesn't.")
    return ("isna() recognizes NaN, None, NaT and pd.NA alike. A profile is the first table to look at in any "
            "new dataset: it tells you which questions the data can answer at all.")


@S.exercise("missing-by-status", "missing delivery date by order status", hints=[
    "The missing indicator is `orders['order_delivered_customer_date'].isna()`, a boolean Series.",
    "Group that indicator by `orders['order_status']` and take the mean: the share missing in each status.",
])
def _(missing_by_status):
    _close_series(missing_by_status, _real()["missing_by_status"], "missing_by_status")
    return ("For every status except 'delivered', the date is missing because no delivery happened: it's "
            "*not applicable*, not unknown. You don't impute an event that never occurred.")


@S.exercise("status-date-audit", "statuses and dates that disagree", challenge=True, hints=[
    "Two boolean masks: status == 'delivered', and delivery date missing. Combine them with & and ~.",
    "Count the delivered orders without a date, and the orders with any other status that DO have a date.",
])
def _(n_delivered_no_date, n_not_delivered_with_date):
    r = _real()
    expect(as_int(n_delivered_no_date, "n_delivered_no_date") == r["n_delivered_no_date"],
           "n_delivered_no_date is off: status 'delivered' AND delivery date missing.")
    expect(as_int(n_not_delivered_with_date, "n_not_delivered_with_date") == r["n_not_delivered_with_date"],
           "n_not_delivered_with_date is off: status other than 'delivered' AND delivery date present.")
    return ("A handful of rows contradict the status. These gaps are 'unknown', not 'not applicable'. Flag them "
            "for review instead of letting the status decide silently.")


@S.exercise("missing-together", "gaps that travel together", hints=[
    "Select the four listing columns, then `.isna()` gives a boolean DataFrame with one column each.",
    "`.all(axis=1)` is True when a row is missing in every column; `.any(axis=1)` when missing in at least one.",
    "'Some but not all' is any & ~all.",
])
def _(n_all_missing, n_some_missing):
    r = _real()
    expect(as_int(n_all_missing, "n_all_missing") == r["n_all_missing"],
           "n_all_missing is off: count products whose four listing columns are ALL missing.")
    expect(as_int(n_some_missing, "n_some_missing") == r["n_some_missing"],
           "n_some_missing is off: missing in at least one of the four, but not in all of them.")
    return ("All four gaps hit exactly the same products: one cause (the listing text was never captured), not "
            "four independent problems. Missingness patterns point to the process that created them.")


# ---- 3. Mechanisms -----------------------------------------------------------------


@S.exercise("unknown-share", "is missingness random?", challenge=True, hints=[
    "`analysis` already has a `category` column, where 'UNKNOWN' marks the products without a category.",
    "share_rows_unknown = mean of (category == 'UNKNOWN'); share_revenue_unknown = price of UNKNOWN rows / total price.",
])
def _(share_rows_unknown, share_revenue_unknown):
    r = _real()
    expect_close(share_rows_unknown, r["share_rows_unknown"], "share_rows_unknown", tol=1e-4)
    expect_close(share_revenue_unknown, r["share_revenue_unknown"], "share_revenue_unknown", tol=1e-4)
    return ("~1.4% of rows but ~1.3% of revenue: unknown-category items skew slightly cheaper. Small here, but if "
            "missingness correlates with a variable, dropping those rows biases results. That's the MCAR question.")


# ---- 4. Complete-case analysis -------------------------------------------------------


@S.exercise("listwise-loss", "what dropna() costs", hints=[
    "`products.dropna()` drops a row if ANY column is missing; `dropna(subset=[...])` looks only at those columns.",
    "Rows dropped = len(before) − len(after).",
])
def _(n_dropped_any, n_dropped_weight):
    r = _real()
    expect(as_int(n_dropped_any, "n_dropped_any") == r["n_dropped_any"],
           "n_dropped_any is off: how many rows does a plain products.dropna() remove?")
    expect(as_int(n_dropped_weight, "n_dropped_weight") == r["n_dropped_weight"],
           "n_dropped_weight is off: drop only the rows missing product_weight_g.")
    return ("A blanket dropna() throws away 611 products to analyse a column that has 2 gaps. Always drop by "
            "the variables your question actually uses (subset=).")


# ---- 5. Imputation -----------------------------------------------------------------------


@S.exercise("impute-group-median", "impute_by_group(df, value_col, group_col)", hints=[
    "`df.groupby(group_col)[value_col].transform('median')` returns each row's group median, aligned to df's rows.",
    "fillna with that Series, then fillna again with the overall median for rows whose group has no median "
    "(group missing, or no observed values in the group).",
    "Record the indicator BEFORE filling: `out[f'{value_col}_was_missing'] = out[value_col].isna()`. Work on a copy.",
])
def _(impute_by_group):
    toy = toy_weights()
    got = unchanged(lambda d: impute_by_group(d, "weight", "category"), toy)
    is_df(got, "impute_by_group(...)")
    want = _ref_impute(toy, "weight", "category")
    expect(list(got.index) == list(toy.index), "Keep the input's index (the toy table is indexed 10..16).")
    expect("weight_was_missing" in got.columns, "Add a boolean column `weight_was_missing`.")
    expect(list(got["weight_was_missing"].astype(bool)) == list(want["weight_was_missing"]),
           "weight_was_missing should be True exactly where weight was missing in the input.")
    g = got["weight"].to_numpy(dtype=float)
    expect(not np.isnan(g).any(),
           "Some weights are still missing. Groups without any observed value (and rows without a group) need "
           "the overall median as a fallback.")
    expect(np.allclose(g, want["weight"].to_numpy()),
           f"Toy check: expected weights {list(want['weight'])}, got {list(g)}. Row 11 should get group a's "
           "median; rows 13–15 the overall median of the observed weights.")
    real = impute_by_group(load_olist("products"), "product_weight_g", "product_category_name")
    expect(np.allclose(real["product_weight_g"].to_numpy(dtype=float),
                       _real()["imputed_weights"]["product_weight_g"].to_numpy()),
           "Toy data passes but the real products table doesn't.")
    return ("Group-median imputation conditions on an observed variable: it's the right idea under MAR given "
            "that variable. The indicator column keeps the fact 'this was a guess' available for later.")


@S.exercise("shrinkage-check", "variance before and after mean imputation", hints=[
    "Sample variance: `s.var()` (pandas uses ddof=1 and skips missing values).",
    "Mean imputation: `s.fillna(s.mean())`, then the variance of that.",
])
def _(var_observed, var_mean_imputed):
    r = _real()
    expect_close(var_observed, r["var_observed"], "var_observed", tol=1e-9)
    expect_close(var_mean_imputed, r["var_mean_imputed"], "var_mean_imputed", tol=1e-9)
    nobs, n = int(load_olist("products")["product_weight_g"].notna().sum()), len(load_olist("products"))
    expect(abs(var_mean_imputed - var_observed * (nobs - 1) / (n - 1)) <= 1e-6 * var_observed,
           "Your two numbers should satisfy s² = s²_obs·(n_obs−1)/(n−1). Did you use the same column?")
    return ("With 2 gaps in 32,951 rows the shrinkage is invisible; with 30% gaps it would cut the variance by "
            "about 30%. Every test and interval built on the imputed column would be overconfident.")


@S.exercise("mode-inflation", "what filling with the mode does", challenge=True, hints=[
    "`s.mode()[0]` is the most frequent non-missing value.",
    "share_top_before = share of ALL products whose category equals the mode; share_top_after = the same after "
    "`fillna(mode)`.",
])
def _(share_top_before, share_top_after):
    r = _real()
    expect_close(share_top_before, r["share_top_before"], "share_top_before", tol=1e-6)
    expect_close(share_top_after, r["share_top_after"], "share_top_after", tol=1e-6)
    return ("Mode imputation hands 610 unknown products to bed_bath_table, inflating its share by ~2 points and "
            "its lead over the runner-up. An explicit 'UNKNOWN' level keeps the gap visible instead.")


@S.exercise("mnar-bias", "a missingness no column can reveal", challenge=True, hints=[
    "Simulate MNAR: `hidden = w.where(w <= 5000)` keeps light products and hides every heavier one.",
    "The complete-case estimates are just `hidden.mean()` and `hidden.median()` (they skip the gaps).",
])
def _(cc_mean, cc_median):
    r = _real()
    expect_close(cc_mean, r["mnar_mean"], "cc_mean", tol=1e-6)
    expect_close(cc_median, r["mnar_median"], "cc_median", tol=1e-6)
    return ("The mean falls from ~2,276 g to ~941 g and the median from 700 to 570. No observed column could "
            "flag this, because the cause is the hidden value itself. Only knowledge of the process can.")


# ---- 6. Mini-project ---------------------------------------------------------------------


@S.exercise("comment-by-score", "who writes a comment?", hints=[
    "The indicator 'has a comment' is `reviews['review_comment_message'].notna()`.",
    "Group it by `reviews['review_score']` and take the mean.",
])
def _(comment_by_score):
    _close_series(comment_by_score, _real()["comment_by_score"], "comment_by_score")
    return ("Comments are far from MCAR: whether one exists depends on the score. Any analysis of comment texts "
            "describes a sample tilted towards unhappy customers.")


@S.exercise("cc-bias", "complete-case bias in delivery times", hints=[
    "`reviewed` already has `days` and `has_comment`.",
    "mean_days_all over every row; mean_days_commented over the rows where has_comment is True.",
])
def _(mean_days_all, mean_days_commented):
    r = _real()
    expect_close(mean_days_all, r["mean_days_all"], "mean_days_all", tol=1e-6)
    expect_close(mean_days_commented, r["mean_days_commented"], "mean_days_commented", tol=1e-6)
    return ("Commented reviews come from orders that took about a day longer: a complete-case analysis of "
            "'reviews with text' would overstate delivery times.")


@S.exercise("ipw-days", "inverse probability weighting", challenge=True, hints=[
    "π̂(s) = share of reviews with a comment among those with score s: "
    "`reviewed.groupby('review_score')['has_comment'].transform('mean')` gives it on every row.",
    "Weights are 1/π̂ on the commented rows only.",
    "`np.average(values, weights=w)` computes Σ w·y / Σ w.",
])
def _(ipw_mean_days):
    expect_close(ipw_mean_days, _real()["ipw_days"], "ipw_mean_days", tol=1e-6)
    return ("Reweighting by the observed score recovers almost all of the bias (13.49 → 12.59 vs 12.52 for "
            "everyone). The small remainder is what the score doesn't explain about who comments.")
