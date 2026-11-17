"""Grading for Module I · 07 — Cleaning, transforming, features.

⚠️ SPOILERS: the reference computations used for grading live in this file.
Try the exercises first; read this afterwards if you want to compare approaches.
"""
from __future__ import annotations

import hashlib
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import (Session, as_int, expect, expect_close, is_df, is_series, raises, unchanged)
from ..data import OLIST_FILES, load_olist, olist_path

S = Session("m01/07", "Cleaning, transforming, features")
check, hint, predict, progress = S.check, S.hint, S.predict, S.progress

# ===========================================================================
# Toy data
# ===========================================================================


def toy_items_dirty() -> pd.DataFrame:
    """Item lines with a few problems planted on purpose."""
    return pd.DataFrame({
        "order_id": ["o1", "o1", "o2", "o3", "o4"],
        "price": [-5.0, 0.0, 10.0, 20.0, np.nan],
        "freight_value": [3.0, 4.0, 0.0, 7.5, 2.0],
    })


def toy_dates() -> pd.DataFrame:
    """Three orders: one fine, one delivered before purchase (impossible), one not delivered yet."""
    return pd.DataFrame({
        "order_id": ["d1", "d2", "d3"],
        "purchased": pd.to_datetime(["2018-01-02", "2018-01-05", "2018-01-07"]),
        "delivered": pd.to_datetime(["2018-01-09", "2018-01-04", None]),
    })


def toy_prices() -> pd.Series:
    return pd.Series([10.0, 12.0, 15.0, 20.0, 25.0, 400.0], name="price")


def toy_payments_type() -> pd.DataFrame:
    return pd.DataFrame({"payment_type": ["card", "boleto", "voucher", "card", "boleto", "card"]})


def toy_order_table() -> pd.DataFrame:
    """A miniature of the order-level table from Notebook 01."""
    return pd.DataFrame({
        "order_id": ["o1", "o2", "o3", "o4", "o5", "o6", "o7"],
        "order_status": ["delivered", "delivered", "unavailable", "delivered", "canceled", "delivered", "delivered"],
        "order_purchase_timestamp": pd.to_datetime(["2017-03-01", "2017-05-10", "2017-06-02", "2017-08-15",
                                                    "2017-09-09", "2017-10-01", "2017-12-24"]),
        "customer_state": ["SP", "RJ", "SP", "MG", "SP", "BA", "SP"],
        "n_items": [1, 2, 0, 1, 3, 1, 1],
        "items_value": [100.0, 250.0, np.nan, 40.0, 90.0, 5000.0, 60.0],
        "freight_value": [10.0, 30.0, np.nan, 50.0, 15.0, 100.0, 12.0],
        "paid_value": [110.0, 280.0, 30.0, np.nan, 105.0, 5100.0, 72.0],
    })


# ===========================================================================
# Reference implementations (used for grading)
# ===========================================================================


def _ref_audit(df, rules) -> pd.DataFrame:
    n = {name: int(pd.Series(rule(df)).fillna(False).astype(bool).sum()) for name, rule in rules.items()}
    out = pd.DataFrame({"n_violations": pd.Series(n, dtype="int64")})
    out["share"] = out["n_violations"] / len(df)
    return out


def _ref_winsorize(s, lower=0.01, upper=0.99) -> pd.Series:
    lo, hi = s.quantile([lower, upper])
    return s.clip(lower=lo, upper=hi)


def _ref_min_max(s) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if not hi > lo:
        return s.where(s.isna(), 0.0)
    return (s - lo) / (hi - lo)


def _ref_z(s) -> pd.Series:
    s = s.astype(float)
    sd = s.std(ddof=1)
    if pd.isna(sd) or sd == 0:
        return s.where(s.isna(), 0.0)
    return (s - s.mean()) / sd


def _ref_robust(s) -> pd.Series:
    s = s.astype(float)
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if not iqr > 0:
        raise ValueError("IQR is 0: robust scaling is undefined for this column.")
    return (s - s.median()) / iqr


QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]


def _ref_quartile_bins(s) -> pd.Series:
    return pd.qcut(s, 4, labels=QUARTILE_LABELS)


def _ref_one_hot(df, col, drop_first=False) -> pd.DataFrame:
    levels = sorted(df[col].dropna().unique())
    out = df.drop(columns=col).copy()
    if drop_first:
        levels = levels[1:]
    for lv in levels:
        out[f"{col}_{lv}"] = df[col].eq(lv).astype("int8")
    if df[col].isna().any():
        out[f"{col}_MISSING"] = df[col].isna().astype("int8")
    return out


def _ref_ordinal(s, order) -> pd.Series:
    pos = {v: i for i, v in enumerate(order)}
    return s.map(pos).astype("Int64")


def _ref_freight_ratio(df) -> pd.Series:
    den = df["items_value"]
    return (df["freight_value"] / den.where(den > 0)).rename("freight_ratio")


def _ref_fingerprint(path, n=12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


CLEAN_STEPS = ["paid_missing", "no_items", "freight_ratio", "log_paid", "paid_outlier"]
LOG_COLS = ["step", "rule", "rows_affected", "action", "limitation"]


def _ref_clean(df):
    out = df.copy()
    out["paid_missing"] = out["paid_value"].isna()
    out["no_items"] = out["n_items"].eq(0)
    out["freight_ratio"] = _ref_freight_ratio(out)
    out["log_paid"] = np.log1p(out["paid_value"])
    q1, q3 = out["log_paid"].quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    out["paid_outlier"] = (out["log_paid"].lt(lo) | out["log_paid"].gt(hi)).fillna(False)
    counts = {"paid_missing": int(out["paid_missing"].sum()), "no_items": int(out["no_items"].sum()),
              "freight_ratio": int(out["freight_ratio"].isna().sum()), "log_paid": int(out["log_paid"].isna().sum()),
              "paid_outlier": int(out["paid_outlier"].sum())}
    log = pd.DataFrame({"step": CLEAN_STEPS, "rule": "", "rows_affected": [counts[s] for s in CLEAN_STEPS],
                        "action": "flag", "limitation": ""})
    return out, log


def _ref_order_table() -> pd.DataFrame:
    orders, items = load_olist("orders"), load_olist("items")
    payments, customers = load_olist("payments"), load_olist("customers")
    item_agg = items.groupby("order_id").agg(n_items=("price", "size"), items_value=("price", "sum"),
                                             freight_value=("freight_value", "sum"))
    paid_agg = payments.groupby("order_id").agg(paid_value=("payment_value", "sum"))
    t = (orders[["order_id", "order_status", "order_purchase_timestamp", "customer_id"]]
         .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left", validate="many_to_one")
         .merge(item_agg, on="order_id", how="left", validate="one_to_one")
         .merge(paid_agg, on="order_id", how="left", validate="one_to_one")
         .drop(columns="customer_id"))
    t["n_items"] = t["n_items"].fillna(0).astype(int)
    return t


@lru_cache(maxsize=1)
def _real() -> dict:
    """Reference answers on the real Olist data (computed once)."""
    orders, items = load_olist("orders"), load_olist("items")
    r: dict = {}
    flag = orders["order_delivered_carrier_date"] < orders["order_approved_at"]
    gap = (orders["order_approved_at"] - orders["order_delivered_carrier_date"])[flag].dt.total_seconds() / 3600
    r["n_carrier_before_approval"] = int(flag.sum())
    r["median_gap_hours"] = float(gap.median())

    price = items["price"]
    w = _ref_winsorize(price)
    r["winsor"] = w
    r["mean_change"] = float((w.mean() - price.mean()) / price.mean())
    r["median_change"] = float((w.median() - price.median()) / price.median())

    t = _ref_order_table()
    r["order_table"] = t
    r["quartiles"] = _ref_quartile_bins(t["paid_value"])
    fr = _ref_freight_ratio(t)
    r["share_freight_exceeds"] = float((fr[fr.notna()] > 1).mean())

    d = orders[orders["order_status"].eq("delivered") & orders["order_delivered_customer_date"].notna()].copy()
    d["days"] = (d["order_delivered_customer_date"] - d["order_purchase_timestamp"]).dt.total_seconds() / 86400
    train = d[d["order_purchase_timestamp"] < "2018-01-01"]["days"]
    test = d[d["order_purchase_timestamp"] >= "2018-01-01"]["days"]
    r["z_leaky"] = float(((test - d["days"].mean()) / d["days"].std()).mean())
    r["z_honest"] = float(((test - train.mean()) / train.std()).mean())

    r["clean"], r["clean_log"] = _ref_clean(t)
    return r


# ===========================================================================
# Predictions
# ===========================================================================

S.prediction("nat-violation", 2,
    "d2 breaks the rule (delivered before it was purchased), and so does d3, by accident: its delivery date is NaT, "
    "so `delivered >= purchased` is False, and `~False` is True. A comparison with a missing value is always False, "
    "so negating it turns 'unknown' into 'violation'. Write the violation itself (`delivered < purchased`), or "
    "handle missing values explicitly.")
S.prediction("clip-mean", 22.0,
    "Only 400 changes: it becomes 50. The sum drops from 482 to 132, so the mean goes from 80.3 to 132 / 6 = 22. "
    "The median, (15 + 20) / 2 = 17.5, doesn't move, because capping the largest value doesn't change which values "
    "sit in the middle.",
    compare=lambda v, a: abs(float(v) - a) < 0.05)
S.prediction("log-skew", "b",
    "Item prices have a skewness of about 7.9: a long right tail. log1p turns ratios into differences, which "
    "compresses that tail, and the skewness drops to about 0.3. The log scale is close to symmetric, and that's "
    "why prices, incomes and city sizes are so often analyzed on it.",
    compare=lambda v, a: str(v).strip().lower().strip(")") == a)
S.prediction("qcut-ties", 2,
    "About 90 % of orders have at most one item row, so the 25th, 50th and 75th percentiles of n_items are all 1. "
    "The edges [0, 1, 1, 1, 21] collapse to [0, 1, 21] with duplicates='drop', which leaves 2 bins instead of 4. "
    "Equal-frequency bins can't split a value that a quarter of the data shares.")
S.prediction("dummy-rank", 3,
    "There are 4 columns (intercept + 3 dummies), but every row has exactly one dummy equal to 1, so "
    "d_boleto + d_card + d_voucher = the intercept column. One column is a linear combination of the others, so the "
    "rank is 3, not 4. Dropping one dummy restores full column rank.")

# ===========================================================================
# Exercises
# ===========================================================================

# ---- 1. Audits --------------------------------------------------------------


@S.exercise("audit-table", "audit(df, rules)", hints=[
    "Loop over `rules.items()`: each rule is a function that takes the DataFrame and returns a boolean Series.",
    "A rule might return missing values: `.fillna(False)` before summing, so 'unknown' isn't counted as a violation.",
    "Build the result with `pd.DataFrame({'n_violations': ...}, index=list(rules))`, then add `share = n_violations / len(df)`.",
])
def _(audit):
    df = toy_items_dirty()
    rules = {
        "price_negative": lambda d: d["price"] < 0,
        "price_missing": lambda d: d["price"].isna(),
        "free_shipping": lambda d: d["freight_value"].eq(0),
        "unknown_rule": lambda d: pd.Series([True, pd.NA, False, pd.NA, False], dtype="boolean", index=d.index),
    }
    got = unchanged(lambda d: audit(d, rules), df)
    is_df(got, "audit(...)")
    want = _ref_audit(df, rules)
    expect(list(got.index) == list(rules), f"One row per rule, indexed by the rule names in the given order: {list(rules)}.")
    expect({"n_violations", "share"} <= set(got.columns), "Columns needed: `n_violations` and `share`.")
    expect(list(got["n_violations"].astype(int)) == list(want["n_violations"]),
           f"n_violations should be {list(want['n_violations'])}, got {list(got['n_violations'])}. "
           "Missing values returned by a rule (the last one) must not count as violations.")
    expect(np.allclose(got["share"].to_numpy(dtype=float), want["share"].to_numpy()),
           "share = n_violations / number of rows of df.")
    return ("An audit table turns vague worries into numbers you can track from one data delivery to the next. "
            "Each rule is a reviewable, testable function.")


@S.exercise("event-order", "carrier before approval: error or plausible?", challenge=True, hints=[
    "Flag: `orders['order_delivered_carrier_date'] < orders['order_approved_at']`.",
    "Gap in hours: (approved − carrier) for the flagged rows, `.dt.total_seconds() / 3600`, then `.median()`.",
])
def _(n_carrier_before_approval, median_gap_hours):
    r = _real()
    n = as_int(n_carrier_before_approval, "n_carrier_before_approval")
    expect(n == r["n_carrier_before_approval"], "The count is off. Compare carrier hand-off with approval timestamps.")
    expect_close(median_gap_hours, r["median_gap_hours"], "median_gap_hours", tol=1e-3)
    return ("A typical gap of under a day suggests the approval timestamp is often recorded late (batch updates, "
            "time zones), not that parcels shipped before payment. Flag these rows and keep them, and don't "
            "compute 'approval-to-shipping' durations from them.")


# ---- 2. Outliers: flag, clip, winsorize ------------------------------------------


@S.exercise("winsorize", "winsorize(s, lower, upper)", hints=[
    "The caps are the sample quantiles: `lo, hi = s.quantile([lower, upper])`.",
    "`s.clip(lower=lo, upper=hi)` caps the values and keeps NaN and the index as they are.",
])
def _(winsorize):
    s = pd.Series([float(x) for x in range(1, 11)] + [100.0, np.nan], index=range(10, 22))
    got = unchanged(lambda x: winsorize(x, 0.1, 0.9), s)
    is_series(got, "winsorize(...)")
    want = _ref_winsorize(s, 0.1, 0.9)
    expect(list(got.index) == list(s.index), "Keep the input's index.")
    expect(got.isna().equals(s.isna()), "Missing values must stay missing (and nothing else should become missing).")
    expect(np.allclose(got.dropna().to_numpy(), want.dropna().to_numpy()),
           f"Toy check: expected {list(want.dropna())}, got {list(got.dropna())}. Cap at the 10th and 90th percentiles.")
    real = winsorize(load_olist("items")["price"])
    expect(np.allclose(real.to_numpy(dtype=float), _real()["winsor"].to_numpy()),
           "Toy check passes but the real prices don't. Are your defaults lower=0.01, upper=0.99?")
    return "Winsorizing caps extremes at sample quantiles. Unlike trimming, it keeps every row (and n)."


@S.exercise("winsor-sensitivity", "how much do the 1 % tails move the mean?", hints=[
    "Winsorize items['price'] with the defaults (1 % and 99 %).",
    "Relative change = (new − old) / old, for the mean and for the median.",
])
def _(mean_change, median_change):
    r = _real()
    expect_close(mean_change, r["mean_change"], "mean_change", tol=1e-4)
    expect(abs(float(median_change) - r["median_change"]) < 1e-9, "median_change is off: compare the medians before and after.")
    return ("Capping 2 % of the rows lowers the mean price by about 4 % and leaves the median exactly where it was. "
            "Report both scenarios: if a conclusion flips, the extremes are driving it.")


# ---- 3. Scaling and logs -------------------------------------------------------------


@S.exercise("scalers", "min_max(s) and z_score(s)", hints=[
    "min-max: (s − min) / (max − min). z-score: (s − mean) / std, with pandas' default ddof=1.",
    "pandas skips NaN in min/max/mean/std, and arithmetic keeps NaN as NaN. That part is free.",
    "Guard the zero denominator: if max == min (or std is 0 or NaN, as with one value), return 0.0 for the observed "
    "values and NaN for the missing ones: `s.where(s.isna(), 0.0)`.",
])
def _(min_max, z_score):
    cases = [pd.Series([2.0, 4.0, 6.0, np.nan]), pd.Series([7.0, 7.0, 7.0]), pd.Series([5.0]),
             pd.Series([1, 2, 3, 10], index=[9, 8, 7, 6])]
    for s in cases:
        for fn, ref, name in ((min_max, _ref_min_max, "min_max"), (z_score, _ref_z, "z_score")):
            got = unchanged(fn, s)
            is_series(got, f"{name}(...)")
            want = ref(s)
            expect(list(got.index) == list(s.index), f"{name}: keep the input's index.")
            g, w = got.to_numpy(dtype=float), want.to_numpy(dtype=float)
            expect(np.array_equal(np.isnan(g), np.isnan(w)) and np.allclose(np.nan_to_num(g), np.nan_to_num(w)),
                   f"{name}({list(s)}) should be {list(np.round(w, 4))}, got {list(np.round(g, 4))}. "
                   + ("Constant input → 0.0 for every observed value." if s.nunique() <= 1 else ""))
    return ("Scaling is a linear map: it changes units, not shape. A z-score of 2 means 'two standard deviations "
            "above the mean' in any column, which is what makes features comparable.")


@S.exercise("robust-scale", "robust_scale(s): median and IQR", challenge=True, hints=[
    "(s − median) / IQR, where IQR = Q3 − Q1 = s.quantile(.75) − s.quantile(.25).",
    "If the IQR is 0, the scale is undefined: raise ValueError with a message that says so (don't return inf or NaN).",
])
def _(robust_scale):
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    got = unchanged(robust_scale, s)
    is_series(got, "robust_scale(...)")
    expect(np.allclose(got.to_numpy(dtype=float), [-1.0, -0.5, 0.0, 0.5, 48.5]),
           f"robust_scale([1, 2, 3, 4, 100]) should be [-1, -0.5, 0, 0.5, 48.5], got {list(got)}.")
    res = raises(robust_scale, ValueError, pd.Series([1.0, 1.0, 1.0, 1.0, 5.0]))
    expect(res is True, "When the IQR is 0 (here Q1 = Q3 = 1), raise ValueError"
           + (f", not {res}." if res else ", but nothing was raised."))
    res = raises(robust_scale, ValueError, _real()["order_table"]["n_items"])
    expect(res is True, "n_items of the real order table has IQR 0 too: your function should raise there.")
    return ("The median and IQR ignore the 100, so the bulk of the data keeps a sensible spread. But a column where "
            "most values are identical (n_items: Q1 = Q3 = 1) has no robust scale at all, so fail loudly.")


# ---- 4. Discretization -------------------------------------------------------------------


@S.exercise("quartile-bins", "quartile_bins(s) with qcut", hints=[
    "`pd.qcut(s, 4, labels=[...])` cuts at the sample quartiles.",
    "Missing values stay missing: qcut handles that for you.",
])
def _(quartile_bins):
    s = pd.Series([3.0, 1.0, 8.0, 2.0, 7.0, 5.0, 4.0, 6.0, np.nan])
    got = unchanged(quartile_bins, s)
    expect(isinstance(getattr(got, "dtype", None), pd.CategoricalDtype), "Return what pd.qcut returns: a Categorical Series.")
    g = [None if pd.isna(x) else str(x) for x in got]
    expect(g == ["Q2", "Q1", "Q4", "Q1", "Q4", "Q3", "Q2", "Q3", None],
           f"Toy check: got {g}. Use labels {QUARTILE_LABELS} and q=4.")
    real = quartile_bins(_real()["order_table"]["paid_value"])
    want = _real()["quartiles"]
    expect((real.astype(str).to_numpy() == want.astype(str).to_numpy()).all(), "Real data: the bins don't match.")
    counts = real.value_counts()
    return (f"Equal-frequency bins: {counts.min():,} to {counts.max():,} orders each. The edges come from the data, "
            "so a model must learn them on training data only (see section 6).")


# ---- 5. Encoding ---------------------------------------------------------------------------


@S.exercise("one-hot", "one_hot(df, col, drop_first)", hints=[
    "Levels: `sorted(df[col].dropna().unique())`. For each level, `df[col].eq(level).astype('int8')`.",
    "Missing values get their own indicator column `{col}_MISSING` (only when there are any).",
    "drop_first=True removes the column of the first level in that sorted list. Drop `col` itself and keep the other columns first.",
])
def _(one_hot):
    df = pd.DataFrame({"x": [1, 2, 3, 4], "pay": ["card", "boleto", None, "card"]})
    for drop in (False, True):
        got = unchanged(lambda d: one_hot(d, "pay", drop_first=drop), df)
        is_df(got, "one_hot(...)")
        want = _ref_one_hot(df, "pay", drop)
        expect(list(got.columns) == list(want.columns),
               f"drop_first={drop}: columns should be {list(want.columns)}, got {list(got.columns)}.")
        expect((got.to_numpy() == want.to_numpy()).all(), f"drop_first={drop}: values don't match.")
    full = one_hot(df, "pay")
    expect((full.filter(like="pay_").sum(axis=1) == 1).all(), "Every row should have exactly one 1 among the pay_ columns.")
    return ("One-hot encoding turns a nominal variable into numbers without inventing an order. An explicit "
            "MISSING column keeps 'not recorded' visible (Notebook 06).")


@S.exercise("ordinal-codes", "ordinal_codes(s, order)", challenge=True, hints=[
    "Map each value to its position in `order`: a dict {value: position} and `s.map(...)`.",
    "Values that are missing *or not in* `order` should become <NA>, not -1 or 0. Return dtype 'Int64' (capital I).",
    "The trap to avoid: `pd.Categorical(s, categories=order).codes` uses -1 for 'missing', and -1 looks like a real number.",
])
def _(ordinal_codes):
    s = pd.Series(["low", "high", None, "medium", "huge"], index=[5, 6, 7, 8, 9])
    order = ["low", "medium", "high"]
    got = unchanged(lambda x: ordinal_codes(x, order), s)
    is_series(got, "ordinal_codes(...)")
    expect(list(got.index) == list(s.index), "Keep the input's index.")
    g = [None if pd.isna(x) else int(x) for x in got]
    expect(g == [0, 2, None, 1, None],
           f"Expected [0, 2, <NA>, 1, <NA>], got {g}. Unknown values ('huge') and missing values must become <NA>, never -1.")
    expect(str(got.dtype) == "Int64", f"Return the nullable integer dtype 'Int64', got {got.dtype}.")
    return ("Ordinal codes are fine when the order is real, but they still claim equal steps (low→medium = "
            "medium→high). And a sentinel like -1 silently becomes the smallest value in any model.")


# ---- 6. Features and leakage -----------------------------------------------------------------


@S.exercise("freight-ratio", "freight_ratio(df) with a safe denominator", hints=[
    "Divide by `df['items_value'].where(df['items_value'] > 0)`: non-positive denominators become NaN before dividing.",
    "Then share_freight_exceeds: among rows where the ratio is defined (notna), the mean of (ratio > 1).",
])
def _(freight_ratio, share_freight_exceeds):
    df = pd.DataFrame({"items_value": [100.0, 0.0, np.nan, 50.0], "freight_value": [10.0, 5.0, 3.0, 60.0]})
    got = unchanged(freight_ratio, df)
    is_series(got, "freight_ratio(...)")
    g = got.to_numpy(dtype=float)
    expect(not np.isinf(g).any(), "Your ratio contains inf: a zero denominator slipped through.")
    expect(np.array_equal(np.isnan(g), [False, True, True, False]) and np.allclose(g[[0, 3]], [0.1, 1.2]),
           f"Toy check: expected [0.1, NaN, NaN, 1.2], got {list(np.round(g, 3))}.")
    expect_close(share_freight_exceeds, _real()["share_freight_exceeds"], "share_freight_exceeds", tol=1e-4)
    return ("A ratio needs a positive denominator and a population: 'among orders with items'. In about 3 % of "
            "them, shipping cost more than the products: worth a separate look, not deletion.")


@S.exercise("leakage-gap", "a scaler fitted on the wrong data", challenge=True, hints=[
    "Population: delivered orders with a delivery date. days = (delivered − purchased).dt.total_seconds() / 86400.",
    "Train: purchases before 2018-01-01. Test: from 2018-01-01 on.",
    "Leaky: z-score the test days with the mean and std of ALL rows. Honest: with the train mean and std. "
    "Report the mean of each set of test z-scores.",
])
def _(z_test_mean_leaky, z_test_mean_honest):
    r = _real()
    expect_close(z_test_mean_leaky, r["z_leaky"], "z_test_mean_leaky", tol=1e-3)
    expect_close(z_test_mean_honest, r["z_honest"], "z_test_mean_honest", tol=1e-3)
    return ("Deliveries got faster in 2018. Fitted on all the data, the scaler has already seen the test period, "
            "and it hides about half of that shift (−0.04 vs −0.10 SD). Fit every transformation on training data only.")


# ---- 7. Pipelines and provenance ---------------------------------------------------------------


@S.exercise("data-fingerprint", "data_fingerprint(path, n=12)", hints=[
    "`hashlib.sha256()`, then feed it the file's bytes: open in binary mode ('rb').",
    "Read in blocks for big files: `for block in iter(lambda: f.read(1 << 20), b''): h.update(block)`.",
    "Return `h.hexdigest()[:n]`.",
])
def _(data_fingerprint):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "hello.txt"
        p.write_bytes(b"hello")
        got = data_fingerprint(p)
        expect(isinstance(got, str), f"Return a string, got {type(got).__name__}.")
        expect(got == "2cf24dba5fb0", f"sha256(b'hello') starts with '2cf24dba5fb0', got {got!r}. "
               "Hash the raw bytes (binary mode), not decoded text.")
        expect(data_fingerprint(p, n=20) == hashlib.sha256(b"hello").hexdigest()[:20], "Respect the `n` argument.")
    path = olist_path("categories")
    expect(data_fingerprint(path) == _ref_fingerprint(path), "Toy file passes but the real CSV doesn't.")
    return ("Same fingerprint ⇒ same bytes. Store it next to every result: 'computed on "
            "product_category_name_translation.csv @ a81f0d1f27b2'.")


@S.exercise("clean-orders", "clean_orders(order_table) → (clean, log)", hints=[
    "Start with `out = df.copy()`, add the five columns one by one, and count the affected rows as you go.",
    "freight_ratio: your function from section 6. log_paid: np.log1p. paid_outlier: Tukey fences (1.5 × IQR) on "
    "log_paid, with missing → False.",
    "Rows affected: paid_missing → number of True; no_items → number of True; freight_ratio → number of NaN ratios; "
    "log_paid → number of NaN; paid_outlier → number of True.",
    "log = pd.DataFrame(list_of_dicts) with keys step, rule, rows_affected, action, limitation (one dict per step).",
])
def _(clean_orders):
    def compare(df, where):
        res = unchanged(clean_orders, df)
        expect(isinstance(res, tuple) and len(res) == 2, "Return a tuple (clean, log).")
        got, log = res
        is_df(got, "clean")
        is_df(log, "log")
        want, want_log = _ref_clean(df)
        expect(len(got) == len(df) and list(got["order_id"]) == list(df["order_id"]),
               f"{where}: same rows, same order. A cleaning step that flags shouldn't drop or reorder anything.")
        expect(all(c in got.columns for c in df.columns), f"{where}: keep every original column.")
        for c in CLEAN_STEPS:
            expect(c in got.columns, f"{where}: missing column `{c}`.")
            g, w = got[c], want[c]
            if c in ("freight_ratio", "log_paid"):
                a, b = g.to_numpy(dtype=float), w.to_numpy(dtype=float)
                ok = np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b))
            else:
                ok = list(g.astype(bool)) == list(w.astype(bool))
            expect(ok, f"{where}: column `{c}` doesn't match.")
        expect(set(LOG_COLS) <= set(log.columns), f"{where}: the log needs columns {LOG_COLS}.")
        expect(sorted(log["step"]) == sorted(CLEAN_STEPS), f"{where}: one log row per step: {CLEAN_STEPS}.")
        lg = log.set_index("step")["rows_affected"]
        for s, n in zip(want_log["step"], want_log["rows_affected"]):
            expect(int(lg[s]) == n, f"{where}: rows_affected for `{s}` should be {n}, got {lg[s]}.")
        expect(log["action"].astype(str).str.strip().ne("").all() and log["limitation"].astype(str).str.strip().ne("").all(),
               f"{where}: write an action and a limitation for every step (that's the point of the log).")
    compare(toy_order_table(), "Toy check")
    compare(_real()["order_table"], "Real data")
    r = _real()["clean"]
    n_raw = int((r["paid_value"] > r["paid_value"].quantile(.75) + 1.5 * (r["paid_value"].quantile(.75)
                - r["paid_value"].quantile(.25))).sum())
    return (f"On the log scale the fences flag {int(r['paid_outlier'].sum()):,} orders; on the raw scale they'd flag "
            f"{n_raw:,}, mostly the legitimate long tail. Every flag is a column and every decision a log row, so "
            "anyone can re-run, question, or reverse it.")
