"""Grading for Module I · 03 — Describing a distribution.

⚠️ SPOILERS: the reference computations used for grading live in this file.
Try the exercises first; read this afterwards if you want to compare approaches.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from ..core import (Session, as_int, expect, expect_close, is_df, is_series, unchanged)
from ..data import load_olist

S = Session("m01/03", "Describing a distribution")
check, hint, predict, progress = S.check, S.hint, S.predict, S.progress

# ===========================================================================
# Toy data
# ===========================================================================


def toy_days() -> pd.Series:
    """Delivery times (days) of seven orders."""
    return pd.Series([2, 3, 3, 4, 5, 6, 30], name="days", dtype="float64")


def toy_payment_types() -> pd.Series:
    return pd.Series(["credit_card", "boleto", "credit_card", None, "voucher",
                      "credit_card", "boleto", None, "credit_card", "debit_card"], name="payment_type")


def toy_deliveries() -> pd.DataFrame:
    """Six orders: two of them don't belong to the delivered-with-a-date population."""
    return pd.DataFrame({
        "order_id": ["o1", "o2", "o3", "o4", "o5", "o6"],
        "order_status": ["delivered", "delivered", "canceled", "delivered", "shipped", "delivered"],
        "order_purchase_timestamp": pd.to_datetime([
            "2018-01-01 10:00", "2018-01-02 00:00", "2018-01-03 08:00",
            "2018-01-04 12:00", "2018-01-05 09:00", "2018-01-06 06:00"]),
        "order_delivered_customer_date": pd.to_datetime([
            "2018-01-05 22:00", "2018-01-12 12:00", "2018-01-04 08:00",
            None, None, "2018-01-07 18:00"]),
    }, index=[10, 11, 12, 13, 14, 15])


# ===========================================================================
# Reference implementations (used for grading)
# ===========================================================================


def _ref_freq(s: pd.Series) -> pd.DataFrame:
    n = s.value_counts(dropna=False)
    return pd.DataFrame({"count": n, "share": n / n.sum()})


def _ref_quantile(x, p):
    a = np.sort(np.asarray(pd.Series(x, dtype="float64").dropna()))
    h = (len(a) - 1) * p
    lo = int(np.floor(h))
    hi = min(lo + 1, len(a) - 1)
    return float(a[lo] + (h - lo) * (a[hi] - a[lo]))


def _ref_spread(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return pd.Series({"std": s.std(), "iqr": s.quantile(.75) - s.quantile(.25),
                      "mad_scaled": 1.4826 * (s - s.median()).abs().median()})


def _ref_skew(x) -> float:
    a = np.asarray(pd.Series(x, dtype="float64").dropna())
    n = len(a)
    d = a - a.mean()
    m2, m3 = (d ** 2).mean(), (d ** 3).mean()
    return float(np.sqrt(n * (n - 1)) / (n - 2) * m3 / m2 ** 1.5)


def _ref_ecdf(x):
    a = np.asarray(pd.Series(x, dtype="float64").dropna())
    xs, counts = np.unique(a, return_counts=True)
    return xs, np.cumsum(counts) / len(a)


def _ref_fd_width(x) -> float:
    s = pd.Series(x, dtype="float64").dropna()
    return float(2 * (s.quantile(.75) - s.quantile(.25)) * len(s) ** (-1 / 3))


def _ref_qq(x):
    a = np.sort(np.asarray(pd.Series(x, dtype="float64").dropna()))
    n = len(a)
    return stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n), a


def _ref_tukey(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile(.25), s.quantile(.75)
    i = q3 - q1
    return (s < q1 - k * i) | (s > q3 + k * i)


def _ref_delivery_days(orders: pd.DataFrame) -> pd.Series:
    pop = orders[orders["order_status"].eq("delivered") & orders["order_delivered_customer_date"].notna()]
    return (pop["order_delivered_customer_date"] - pop["order_purchase_timestamp"]).dt.total_seconds() / 86400


def _ref_median_after_corruption(prices: np.ndarray, k: int) -> float:
    a = np.sort(prices).copy()
    a[:k] = 1e12
    return float(np.median(a))


@lru_cache(maxsize=1)
def _real() -> dict:
    orders, items, customers = load_olist("orders"), load_olist("items"), load_olist("customers")
    r: dict = {}
    st = orders["order_status"]
    r["share_delivered"] = float(st.eq("delivered").mean())
    r["share_canceled_undelivered"] = float(st[st.ne("delivered")].eq("canceled").mean())

    tot = items.groupby("order_id")["price"].sum()
    value = orders[["order_id"]].merge(tot.rename("items_value"), left_on="order_id", right_index=True,
                                       how="left", validate="one_to_one")["items_value"]
    r["order_value_spread"] = _ref_spread(value)

    p = items["price"].to_numpy(dtype=float)
    lo, hi = 0, len(p)
    while lo < hi:  # smallest k with median > 1000 after replacing k prices by a huge value
        mid = (lo + hi) // 2
        if _ref_median_after_corruption(p, mid) > 1000:
            hi = mid
        else:
            lo = mid + 1
    r["n_to_move_median"] = lo

    price = items["price"]
    r["skew_price"] = float(price.skew())
    r["skew_log_price"] = float(np.log10(price).skew())
    r["fd_bins_price"] = int(np.ceil((price.max() - price.min()) / _ref_fd_width(price)))
    r["share_flag_price"] = float(_ref_tukey(price).mean())
    r["share_flag_log"] = float(_ref_tukey(np.log10(price)).mean())

    days = _ref_delivery_days(orders)
    r["days"] = days
    r["median_days"] = float(days.median())
    r["p90_days"] = float(days.quantile(.9))
    r["share_over_30"] = float(days.gt(30).mean())
    dd = (orders.loc[days.index, ["order_id", "customer_id"]].assign(days=days)
          .merge(customers[["customer_id", "customer_state"]], on="customer_id", validate="one_to_one"))
    top = dd["customer_state"].value_counts().head(5).index
    r["box_order"] = list(dd[dd["customer_state"].isin(top)].groupby("customer_state")["days"].median()
                          .sort_values().index)
    return r


# ===========================================================================
# Predictions
# ===========================================================================

_SCALES = ("nominal", "ordinal", "interval", "ratio")
S.prediction("scale-of", _SCALES,
    "A state code only names a group (nominal). Review scores have an order but no guaranteed spacing: is 5−4 the "
    "same 'amount' as 2−1? (ordinal). Timestamps have meaningful differences but an arbitrary zero, so '2018 is "
    "1.0005 times 2017' means nothing (interval). Freight has a true zero, so '30 BRL is twice 15 BRL' makes sense (ratio).",
    compare=lambda v, a: isinstance(v, (tuple, list)) and tuple(str(x).strip().lower() for x in v) == a)
S.prediction("robust-to-typo", {"mean", "std"},
    "The mean moves by Δ/n = 270/7 ≈ 38.6 days and the std explodes, because both use every value's distance. "
    "The median and the IQR only look at the middle of the sorted data, and the typo stays the maximum, so they "
    "don't move at all.",
    compare=lambda v, a: isinstance(v, (set, list, tuple, frozenset)) and {str(x).strip().lower() for x in v} == a)
S.prediction("ddof-ratio", 0.926,
    "np.std divides by n, pandas by n−1, so the ratio is √((n−1)/n) = √(6/7) ≈ 0.926, whatever the data. "
    "The gap is noticeable for small n and vanishes as n grows.",
    compare=lambda v, a: isinstance(v, (int, float, np.floating)) and abs(float(v) - 0.92582) < 0.002)
S.prediction("ecdf-at-median", 0.571,
    "F̂(4) counts the values ≤ 4: 2, 3, 3, 4, so 4 of 7 ≈ 0.571. With a finite sample the ECDF jumps at each value, "
    "so at the median it is ≥ 0.5, not exactly 0.5.",
    compare=lambda v, a: isinstance(v, (int, float, np.floating)) and abs(float(v) - 4 / 7) < 0.01)
S.prediction("kde-below-zero", "yes",
    "Yes. A Gaussian kernel puts a bell curve on every observation, and the bells of cheap items spill over 0. "
    "A KDE doesn't know that prices can't be negative: that's the boundary problem. Read the left edge of a KDE "
    "of a positive variable with suspicion, or estimate it on the log scale.",
    compare=lambda v, a: str(v).strip().lower() in {"yes", "y", "true", "1"})

# ===========================================================================
# Exercises
# ===========================================================================

# ---- 1. Scales and frequency tables -------------------------------------------


@S.exercise("freq-table", "freq_table(s)", hints=[
    "`s.value_counts(dropna=False)` counts every value, including missing ones, sorted by count.",
    "share = count / count.sum(). Build the DataFrame with `pd.DataFrame({'count': ..., 'share': ...})`.",
])
def _(freq_table):
    s = toy_payment_types()
    got = unchanged(freq_table, s)
    is_df(got, "freq_table(...)")
    expect(list(got.columns) == ["count", "share"], f"Columns should be ['count', 'share'], got {list(got.columns)}.")
    expect(got.index.isna().any(), "Missing values are information: they must appear as their own row "
           "(value_counts drops them unless you pass dropna=False).")
    want = _ref_freq(s)
    g = {("<NA>" if pd.isna(k) else k): v for k, v in got["count"].items()}
    w = {("<NA>" if pd.isna(k) else k): v for k, v in want["count"].items()}
    expect(g == w, f"Counts don't match: expected {w}, got {g}.")
    expect(abs(float(got["share"].sum()) - 1) < 1e-9, "Shares should add up to 1 (denominator: all rows, missing included).")
    expect(got["count"].is_monotonic_decreasing, "Sort by count, largest first.")
    return "A frequency table without its denominator is half an answer. Say *share of what*: here, of all rows, missing included."


@S.exercise("status-shares", "two shares, two denominators", hints=[
    "`mask.mean()` is the share of True values among the rows of the mask's table.",
    "For the second share, first restrict to the orders that are not delivered, then take the share that are canceled.",
])
def _(share_delivered, share_canceled_of_undelivered):
    r = _real()
    expect_close(share_delivered, r["share_delivered"], "share_delivered", tol=1e-6)
    if abs(float(share_canceled_of_undelivered) - 625 / 99441) < 1e-6:
        expect(False, "share_canceled_of_undelivered uses all orders as the denominator. Restrict to non-delivered orders first.")
    expect_close(share_canceled_of_undelivered, r["share_canceled_undelivered"], "share_canceled_of_undelivered", tol=1e-6)
    return ("0.6 % of all orders but 21 % of the undelivered ones: the same 625 cancellations, two very different "
            "numbers. The denominator is part of the claim.")


# ---- 2. Center, spread, position -------------------------------------------------


@S.exercise("quantile-linear", "quantile_linear(x, p)", hints=[
    "Drop missing values, sort, and compute the fractional position h = (n − 1)·p (0-indexed).",
    "lo = floor(h); the answer is a[lo] + (h − lo)·(a[lo+1] − a[lo]). When h is an integer, the second term is 0.",
    "Guard the edge: when lo is the last index (p = 1 or n = 1), there is no a[lo+1]; use a[lo].",
])
def _(quantile_linear):
    cases = [
        ([1, 2, 4, 10], .25), ([1, 2, 4, 10], .5), ([10, 1, 4, 2], .25), ([7], .3), ([3, 1, 2], 0.0),
        ([3, 1, 2], 1.0), ([5, 5, 5, 9], .9), ([1.0, np.nan, 3.0, 2.0], .5), (list(toy_days()), .75),
    ]
    for x, p in cases:
        got = quantile_linear(list(x), p)
        want = pd.Series(x, dtype="float64").quantile(p)
        expect(isinstance(got, (int, float, np.floating, np.integer)) and abs(float(got) - want) < 1e-9,
               f"quantile_linear({x}, {p}) should be {want:g}, got {got!r}.")
    rng = np.random.default_rng(0)
    x = rng.lognormal(3, 1, 101)
    for p in (.01, .37, .99):
        expect(abs(float(quantile_linear(pd.Series(x), p)) - np.quantile(x, p)) < 1e-9, "Random check failed.")
    return ("You now know exactly what describe(), quantile() and box plots report. Other tools (Excel, some "
            "textbooks) use other rules, so small samples can disagree slightly.")


@S.exercise("spread-summary", "spread_summary(s)", hints=[
    "`s.std()` already ignores NaN and uses n − 1. IQR = s.quantile(.75) − s.quantile(.25).",
    "MAD: the median of the absolute deviations from the median, `(s - s.median()).abs().median()`, times 1.4826.",
    "Return `pd.Series({'std': ..., 'iqr': ..., 'mad_scaled': ...})` in that order.",
])
def _(spread_summary):
    for s in (toy_days(), pd.Series([1.0, np.nan, 2, 2, 3, 100], index=list("abcdef"))):
        got = unchanged(spread_summary, s)
        is_series(got, "spread_summary(...)")
        expect(list(got.index) == ["std", "iqr", "mad_scaled"], f"Index should be ['std', 'iqr', 'mad_scaled'], got {list(got.index)}.")
        want = _ref_spread(s)
        expect(np.allclose(got.to_numpy(dtype=float), want.to_numpy(dtype=float)),
               f"Expected {want.round(4).to_dict()}, got {got.round(4).to_dict()}. Is the std using n − 1? Is NaN ignored?")
    return "Three answers to 'how spread out?': one driven by every value (std), two by the middle of the data (IQR, MAD)."


@S.exercise("order-value-spread", "spread of order values", hints=[
    "Apply your spread_summary to `order_table['items_value']`.",
])
def _(order_value_spread):
    is_series(order_value_spread, "order_value_spread")
    want = _real()["order_value_spread"]
    expect(list(order_value_spread.index) == list(want.index), "Use the Series returned by spread_summary.")
    expect(np.allclose(order_value_spread.to_numpy(dtype=float), want.to_numpy(dtype=float), rtol=1e-6),
           "The values don't match. Use the order-level items_value (sum of prices per order).")
    return ("The std (≈ 211 BRL) is about three times the scaled MAD (≈ 71 BRL): a long right tail inflates the "
            "std. For a skewed variable, the robust measures describe the typical order much better.")


@S.exercise("breakdown", "breakdown in practice", challenge=True, hints=[
    "To push the median up, replace the *smallest* prices by a huge value: every replacement moves one value "
    "from below the middle to above it.",
    "The median of n values is (here) the average of the two middle sorted values. Count how many prices are "
    "already above 1,000 and how far that is from the middle.",
    "You can also just simulate: sort the prices, set the first k to 1e12, take np.median, and search for the "
    "smallest k that works (a binary search is fast).",
    "For the mean, ask: can a single replacement push the sum above 1000 · n?",
])
def _(n_to_move_median, n_to_move_mean):
    a = as_int(n_to_move_median, "n_to_move_median")
    b = as_int(n_to_move_mean, "n_to_move_mean")
    want = _real()["n_to_move_median"]
    expect(b == 1, "One huge value is enough to drag the mean anywhere: its breakdown point is 1/n.")
    expect(abs(a - want) <= 1, "n_to_move_median should be roughly half the sample minus the prices already above "
           "1,000. Simulate it: replace the k smallest prices by 1e12 and find the smallest k with median > 1000.")
    return (f"{want:,} replacements for the median (≈ 49 % of all prices) vs 1 for the mean. That's what "
            "'breakdown point 50 % vs 0 %' means in practice.")


# ---- 3. Shape --------------------------------------------------------------------


@S.exercise("skewness", "skewness(x) from the formula", challenge=True, hints=[
    "Drop NaN. Deviations d = x − mean; m2 = mean(d²), m3 = mean(d³) (divide by n, not n − 1).",
    "g1 = m3 / m2**1.5; G1 = g1 · sqrt(n(n−1)) / (n − 2).",
])
def _(skewness):
    cases = [toy_days(), pd.Series([1.0, 2, 3, 4, 100]), pd.Series([10.0, 9, 9, 8, 1, np.nan]),
             pd.Series(np.random.default_rng(1).exponential(2, 500))]
    for s in cases:
        got = skewness(s.copy())
        want = s.skew()
        expect(isinstance(got, (float, int, np.floating)) and abs(float(got) - want) < 1e-9,
               f"Expected {want:.6f}, got {got!r}. Check that m2 and m3 divide by n, and the √(n(n−1))/(n−2) factor.")
    return "Cubing keeps the sign of each deviation and magnifies the far ones, so a long right tail wins: positive skew."


@S.exercise("log-skew", "skewness before and after the log", hints=[
    "`items['price'].skew()` and `np.log10(items['price']).skew()`. All prices are positive, so log10 is safe.",
])
def _(skew_price, skew_log_price):
    r = _real()
    expect_close(skew_price, r["skew_price"], "skew_price", tol=1e-6)
    if abs(float(skew_log_price) - float(np.log1p(load_olist("items")["price"]).skew())) < 1e-9:
        return ("Correct for log1p (≈ 0.29). With log10 you'd get ≈ 0.24: the shift by 1 barely matters for prices "
                "of tens of BRL. Either way, skewness drops from ≈ 7.9 to almost 0.")
    expect_close(skew_log_price, r["skew_log_price"], "skew_log_price", tol=1e-6)
    return ("From ≈ 7.9 to ≈ 0.24: on the log scale, item prices are almost symmetric. Prices vary "
            "*multiplicatively* (a 10 % markup, not a 10 BRL one), and the log turns ratios into differences.")


# ---- 4. Seeing a distribution ---------------------------------------------------------


@S.exercise("log-histogram", "log_histogram(prices, n_bins)", challenge=True, hints=[
    "Equal-width bins on the log scale: `edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)`.",
    "`fig, ax = plt.subplots(); ax.hist(x, bins=edges); ax.set_xscale('log')`, then label the axes, and return ax.",
])
def _(log_histogram):
    import matplotlib.pyplot as plt
    prices = load_olist("items")["price"]
    ax = log_histogram(prices.copy(), 30)
    try:
        expect(hasattr(ax, "patches") and hasattr(ax, "get_xscale"), "Return the matplotlib Axes (`ax`), not the figure or the counts.")
        bars = [p for p in ax.patches if p.get_height() > 0 or p.get_width() > 0]
        expect(len(ax.patches) == 30, f"Expected 30 bars (n_bins=30), found {len(ax.patches)}.")
        expect(ax.get_xscale() == "log", "Set the x axis to a log scale (`ax.set_xscale('log')`).")
        lw = [np.log10(p.get_x() + p.get_width()) - np.log10(p.get_x()) for p in bars]
        expect(np.ptp(lw) < 1e-6, "The bins should have equal width *on the log scale*: build the edges with np.logspace.")
        lefts = sorted(p.get_x() for p in bars)
        expect(abs(lefts[0] - prices.min()) < 1e-6, "The first bin should start at the minimum price.")
        expect("brl" in ax.get_xlabel().lower(), "Label the x axis with its unit (BRL).")
        expect(ax.get_ylabel().strip() != "", "Label the y axis (what does the height count?).")
    finally:
        plt.close("all")
    return ("On log-spaced bins the price distribution looks almost bell-shaped. Each bin covers the same *ratio* "
            "of prices, not the same number of BRL.")


@S.exercise("ecdf", "ecdf(x)", hints=[
    "Drop NaN first: the denominator n counts only observed values.",
    "`np.unique(a, return_counts=True)` gives the distinct sorted values and how often each occurs.",
    "The ECDF at each distinct value is cumulative count / n: `np.cumsum(counts) / n`.",
])
def _(ecdf):
    for x in (toy_days(), pd.Series([3.0, np.nan, 1.0, 3.0, 2.0])):
        xs, ys = ecdf(x.copy())
        wx, wy = _ref_ecdf(x)
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        expect(xs.shape == wx.shape, f"For {list(x)}: return one point per *distinct* observed value ({len(wx)}), got {len(xs)}.")
        expect(np.allclose(xs, wx), f"xs should be the sorted distinct values {list(wx)}, got {list(xs)}.")
        expect(np.allclose(ys, wy), f"ys should be {np.round(wy, 3).tolist()}, got {np.round(ys, 3).tolist()}. "
               "Ties: F̂ jumps by (number of tied values)/n. NaN: excluded from n.")
    return "The ECDF is the data itself, sorted: no bins, no bandwidth, no choices. Any quantile can be read off it."


@S.exercise("fd-bins", "Freedman–Diaconis bins", challenge=True, hints=[
    "h = 2 · IQR · n^(−1/3), with n the number of non-missing values.",
    "Number of bins = ceil((max − min) / h). Use np.ceil, then int().",
])
def _(fd_width, fd_bins_price):
    for x in (toy_days(), pd.Series([1.0, 2, np.nan, 4, 8, 16, 32, 64])):
        got = fd_width(x.copy())
        expect(isinstance(got, (float, int, np.floating)) and abs(float(got) - _ref_fd_width(x)) < 1e-9,
               f"fd_width({list(x)}) should be {_ref_fd_width(x):.6f}, got {got!r}. Is n counting only non-missing values?")
    want = _real()["fd_bins_price"]
    b = as_int(fd_bins_price, "fd_bins_price")
    expect(b == want, f"fd_bins_price should be ceil((max − min)/h) on item prices. Got {b}.")
    return (f"{want:,} bins! The rule adapts the width to the bulk of the data (via the IQR), but the range is "
            "stretched by a few very expensive items. On a log scale the same rule gives a sensible number.")


@S.exercise("qq-points", "qq_points(x)", challenge=True, hints=[
    "Sort the non-missing values: these are the sample quantiles.",
    "The i-th of n sorted values (i = 1..n) sits at probability p_i = (i − 0.5)/n.",
    "Theoretical quantiles: `scipy.stats.norm.ppf(p)`, the inverse of the standard normal CDF.",
])
def _(qq_points):
    for x in (toy_days(), pd.Series([0.5, np.nan, -1.0, 2.0, 0.0])):
        th, sa = qq_points(x.copy())
        wt, ws = _ref_qq(x)
        th, sa = np.asarray(th, dtype=float), np.asarray(sa, dtype=float)
        expect(len(th) == len(wt) and len(sa) == len(ws), f"Return one point per non-missing value ({len(wt)}).")
        expect(np.allclose(sa, ws), "The sample quantiles should be the sorted non-missing values.")
        expect(np.allclose(th, wt), "Theoretical quantiles: norm.ppf((i − 0.5)/n) for i = 1..n.")
    return ("A straight line means 'same shape as a normal, up to location and scale'. A curve bending up at the "
            "right means a heavier right tail than the normal.")


# ---- 5. Tukey's fences -------------------------------------------------------------------


@S.exercise("tukey-flags", "tukey_flags(s, k=1.5)", hints=[
    "q1, q3 = s.quantile(.25), s.quantile(.75) (pandas ignores NaN); iqr = q3 − q1.",
    "Flag = (s < q1 − k·iqr) | (s > q3 + k·iqr). Comparisons with NaN are False, so missing values aren't flagged.",
])
def _(tukey_flags):
    s = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 30, np.nan, -20], index=list("abcdefghij"))
    got = unchanged(tukey_flags, s)
    is_series(got, "tukey_flags(...)")
    expect(list(got.index) == list(s.index), "Keep the input's index.")
    expect(got.dtype == bool, "Return a boolean Series.")
    want = _ref_tukey(s)
    expect(got.equals(want), f"Expected flags on {list(s.index[want])}, got {list(s.index[got])}.")
    expect(tukey_flags(s, k=3).sum() == _ref_tukey(s, 3).sum(), "Use the k argument (k=3 gives Tukey's 'far out' fences).")
    return "A flag is a question ('is this value plausible?'), not an answer. Keep the flag as a column; don't delete rows."


@S.exercise("tukey-shares", "how many flags, and on which scale?", challenge=True, hints=[
    "Apply tukey_flags to items['price'] and to np.log10(items['price']); `.mean()` of a boolean Series is a share.",
])
def _(share_flag_price, share_flag_log):
    r = _real()
    expect_close(share_flag_price, r["share_flag_price"], "share_flag_price", tol=1e-6)
    expect_close(share_flag_log, r["share_flag_log"], "share_flag_log", tol=1e-6)
    return ("7.5 % of prices are 'outliers' on the raw scale but only 1.4 % on the log scale. The 1.5·IQR rule "
            "assumes a roughly symmetric distribution; on a skewed one it flags the ordinary tail.")


# ---- 6. Mini-project ---------------------------------------------------------------------


@S.exercise("delivery-summary", "a robust summary of delivery time", hints=[
    "`deliveries['days'].median()`, `.quantile(.9)`, and the share above 30 days as `.gt(30).mean()`.",
])
def _(median_days, p90_days, share_over_30):
    r = _real()
    expect_close(median_days, r["median_days"], "median_days")
    expect_close(p90_days, r["p90_days"], "p90_days")
    expect_close(share_over_30, r["share_over_30"], "share_over_30")
    return "Half of orders arrive within ≈ 10 days, 9 in 10 within ≈ 23, and about 1 in 21 takes over a month."


@S.exercise("delivery-boxplot", "delivery_boxplot(deliveries)", hints=[
    "Keep the 5 states with the most deliveries: `value_counts().head(5).index`.",
    "Order them by median days: compute the medians, sort, and pass the groups to ax.boxplot in that order "
    "(`ax.boxplot(list_of_arrays, tick_labels=states, showfliers=False)`; older matplotlib calls it `labels=`).",
    "Label the y axis with the unit (days), and return ax.",
])
def _(delivery_boxplot):
    import matplotlib.pyplot as plt
    r = _real()
    orders, customers = load_olist("orders"), load_olist("customers")
    days = r["days"]
    dd = (orders.loc[days.index, ["order_id", "customer_id"]].assign(days=days)
          .merge(customers[["customer_id", "customer_state"]], on="customer_id", validate="one_to_one"))
    ax = delivery_boxplot(dd)
    try:
        expect(hasattr(ax, "get_xticklabels"), "Return the matplotlib Axes (`ax`).")
        labels = [t.get_text() for t in ax.get_xticklabels()]
        expect(labels == r["box_order"],
               f"Expected the 5 busiest states ordered by median delivery time, {r['box_order']}; got {labels}.")
        markers = [ln for ln in ax.lines if ln.get_marker() not in ("None", "", None, " ") and len(ln.get_xdata())]
        expect(not markers, "Don't draw the individual outliers here (showfliers=False): with thousands of orders they "
               "swamp the picture.")
        expect("day" in ax.get_ylabel().lower(), "Label the y axis with its unit (days).")
    finally:
        plt.close("all")
    return ("Sorting by the median turns a row of boxes into a ranking you can read at a glance. Remember: the "
            "whiskers show where data lie, not how uncertain the median is.")
