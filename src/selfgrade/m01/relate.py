"""Grading for Module I · 04 — Comparing groups, relating variables.

⚠️ SPOILERS: the reference computations used for grading live in this file.
Try the exercises first; read this afterwards if you want to compare approaches.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ..core import Session, as_int, expect, expect_close, is_df, is_series, unchanged  # noqa: F401
from ..data import load_olist

S = Session("m01/04", "Comparing groups, relating variables")
check, hint, predict, progress = S.check, S.hint, S.predict, S.progress

DIVERGING = {"coolwarm", "bwr", "seismic", "RdBu", "RdBu_r", "PuOr", "PuOr_r", "BrBG", "BrBG_r", "PiYG",
             "PiYG_r", "PRGn", "PRGn_r", "RdYlBu", "RdYlBu_r", "RdGy", "RdGy_r", "Spectral", "Spectral_r",
             "coolwarm_r", "bwr_r", "seismic_r", "RdYlGn", "RdYlGn_r", "vlag", "icefire"}

# ===========================================================================
# Toy data
# ===========================================================================


def toy_deliveries() -> pd.DataFrame:
    """Delivery times (days) for orders in three states; one order has no observed delivery."""
    return pd.DataFrame({
        "order_id": [f"o{i}" for i in range(1, 11)],
        "customer_state": ["SP", "SP", "SP", "SP", "SP", "RJ", "RJ", "RJ", "AC", "AC"],
        "delivery_days": [3.0, 5.0, 4.0, 10.0, 8.0, 12.0, np.nan, 20.0, 30.0, 14.0],
    })


def toy_xy() -> pd.DataFrame:
    return pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6], "y": [2.0, 3, 5, 4, 6, 7]})


def toy_carriers() -> pd.DataFrame:
    """Orders shipped by two carriers to near and far regions; `late` is 1 for a late delivery."""
    rows = []
    for carrier, region, n, n_late in [("A", "near", 20, 1), ("B", "near", 80, 8),
                                       ("A", "far", 80, 32), ("B", "far", 20, 10)]:
        rows += [(carrier, region, int(i < n_late)) for i in range(n)]
    return pd.DataFrame(rows, columns=["carrier", "region", "late"])


def toy_carriers_uneven() -> pd.DataFrame:
    """Second test table: three regions, uneven sizes."""
    rows = []
    for carrier, region, n, n_late in [("A", "r1", 10, 1), ("B", "r1", 30, 6), ("A", "r2", 50, 10),
                                       ("B", "r2", 10, 3), ("A", "r3", 5, 4), ("B", "r3", 15, 9)]:
        rows += [(carrier, region, int(i < n_late)) for i in range(n)]
    return pd.DataFrame(rows, columns=["carrier", "region", "late"])


# ===========================================================================
# Reference implementations
# ===========================================================================


def _ref_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("customer_state")["delivery_days"]
    out = pd.DataFrame({"n_orders": g.count(), "mean_days": g.mean(), "median_days": g.median(),
                        "p90_days": g.quantile(0.9)})
    return out.sort_values("n_orders", ascending=False)


def _ref_monthly(ts: pd.Series) -> pd.Series:
    m = pd.to_datetime(ts).dropna().dt.to_period("M")
    vc = m.value_counts()
    rng = pd.period_range(vc.index.min(), vc.index.max(), freq="M")
    return vc.reindex(rng, fill_value=0).astype(int)


def _ref_delivered_rate(orders: pd.DataFrame) -> pd.Series:
    m = orders["order_purchase_timestamp"].dt.to_period("M")
    return orders["order_status"].eq("delivered").groupby(m).mean().sort_index()


def _ref_pearson(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    xc, yc = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    den = np.linalg.norm(xc) * np.linalg.norm(yc)
    return float(xc @ yc / den) if den > 0 else float("nan")


def _ref_stratified(df, group, stratum, outcome) -> pd.DataFrame:
    return df.groupby([stratum, group])[outcome].mean().unstack(group)


def _ref_standardize(df, group, stratum, outcome) -> pd.Series:
    w = df[stratum].value_counts(normalize=True)
    p = _ref_stratified(df, group, stratum, outcome)
    return p.mul(w, axis=0).sum(axis=0)


def _ref_reviews_one(reviews) -> pd.DataFrame:
    return (reviews.sort_values(["order_id", "review_answer_timestamp"])
            .drop_duplicates("order_id", keep="last"))


@lru_cache(maxsize=1)
def _real() -> dict:
    orders, items = load_olist("orders"), load_olist("items")
    reviews, customers = load_olist("reviews"), load_olist("customers")
    r: dict = {}
    dl = orders[orders["order_status"].eq("delivered") & orders["order_delivered_customer_date"].notna()].merge(
        customers[["customer_id", "customer_state"]], on="customer_id", how="left", validate="many_to_one")
    dl["delivery_days"] = (dl["order_delivered_customer_date"] - dl["order_purchase_timestamp"]).dt.total_seconds() / 86400
    r["deliveries"] = dl[["order_id", "customer_state", "delivery_days"]]
    r["state_summary"] = _ref_state_summary(r["deliveries"])
    r["monthly"] = _ref_monthly(orders["order_purchase_timestamp"])
    r["delivered_rate"] = _ref_delivered_rate(orders)
    r["r_pf"] = float(items["price"].corr(items["freight_value"]))
    r["rho_pf"] = float(items["price"].corr(items["freight_value"], method="spearman"))

    dv = orders[orders["order_status"].eq("delivered") & orders["order_delivered_customer_date"].notna()].copy()
    dv["is_late"] = dv["order_delivered_customer_date"].dt.normalize() > dv["order_estimated_delivery_date"]
    r["share_late"] = float(dv["is_late"].mean())
    r1 = _ref_reviews_one(reviews)
    r["n_reviewed_orders"] = int(reviews["order_id"].nunique())
    j = dv.merge(r1[["order_id", "review_score"]], on="order_id", how="inner", validate="one_to_one")
    r["mean_late"] = float(j.loc[j["is_late"], "review_score"].mean())
    r["mean_on_time"] = float(j.loc[~j["is_late"], "review_score"].mean())
    j["delay_days"] = (j["order_delivered_customer_date"] - j["order_estimated_delivery_date"]).dt.total_seconds() / 86400
    r["rho_delay"] = float(j["delay_days"].corr(j["review_score"], method="spearman"))
    j = j.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left", validate="many_to_one")
    n = j.groupby("customer_state").size()
    big = n[n >= 1000].index
    jb = j[j["customer_state"].isin(big)]
    m = jb.groupby(["customer_state", "is_late"])["review_score"].mean().unstack()
    r["gap_by_state"] = (m[True] - m[False]).sort_index()
    return r


# ===========================================================================
# Predictions
# ===========================================================================

S.prediction("normalize-direction", "rows",
    "normalize='index' divides each cell by its ROW total, so every row sums to 1: each row is the conditional "
    "distribution of the column variable given that row's category. normalize='columns' does the same per column, "
    "and normalize='all' divides by the grand total.",
    compare=lambda v, a: str(v).strip().lower().rstrip("s") in {"row", "index"})
S.prediction("extremes-small", "b",
    "With the same true rate everywhere, every difference between states is pure sampling noise, and the noise in a "
    "group's average shrinks like 1/√n. The smallest states wander furthest from the truth, in both directions, so "
    "they take the top AND the bottom of the ranking.",
    compare=lambda v, a: str(v).strip().lower().strip(")") == a)
S.prediction("influential-point", "negative",
    "r measures how well ONE straight line fits all the points. The new point is far from the others in x, so it "
    "dominates the sums of products: r drops from +0.94 to about −0.50. Spearman's ρ only sees ranks (the new point "
    "is just 'largest x, smallest y') and falls to about +0.21.",
    compare=lambda v, a: (isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0)
    or str(v).strip().lower().startswith("neg") or str(v).strip() == "-")
S.prediction("monotone-not-linear", "spearman",
    "y = eˣ always increases with x, so the ranks of y are exactly the ranks of x: ρ = 1. The relation isn't a "
    "straight line, so Pearson's r (≈ 0.72 here) is below 1, even though y is a perfect function of x.",
    compare=lambda v, a: str(v).strip().lower() in {"spearman", "rho", "ρ"})
S.prediction("simpson-overall", "B",
    "B is late MORE often within each region, yet less often overall: B ships mostly to 'near' (easy) "
    "destinations, A mostly to 'far' ones. The overall rate is a weighted average of the regional rates, and the "
    "two carriers use very different weights. That's Simpson's paradox.",
    compare=lambda v, a: str(v).strip().upper() == a)

# ===========================================================================
# Exercises
# ===========================================================================

# ---- 1. Conditional distributions ------------------------------------------------


@S.exercise("state-delivery-summary", "state_delivery_summary(df)", hints=[
    "Group by customer_state and select delivery_days: `g = df.groupby('customer_state')['delivery_days']`.",
    "`g.count()` counts non-missing values (an order with no observed delivery isn't counted); also `g.mean()`, "
    "`g.median()`, `g.quantile(0.9)`.",
    "Put them in a DataFrame with the four column names in order, then `.sort_values('n_orders', ascending=False)`.",
])
def _(state_delivery_summary):
    cols = ["n_orders", "mean_days", "median_days", "p90_days"]

    def compare(got, want, where):
        is_df(got, "state_delivery_summary(...)")
        expect(list(got.columns) == cols, f"{where}: columns should be {cols}, in that order. Got {list(got.columns)}.")
        expect(list(got.index) == list(want.index),
               f"{where}: expected states {list(want.index)} as the index (largest n_orders first), got {list(got.index)}.")
        expect(np.allclose(got.to_numpy(dtype=float), want.to_numpy(dtype=float)),
               f"{where}: values don't match. n_orders counts orders WITH an observed delivery time.")

    t = toy_deliveries()
    compare(unchanged(state_delivery_summary, t), _ref_state_summary(t), "Toy check")
    compare(state_delivery_summary(_real()["deliveries"].copy()), _real()["state_summary"], "Real data")
    return ("A conditional summary is a set of distributions, one per group. Compare them with the group sizes in "
            "view: RR's 41 orders and SP's 40,494 don't deserve the same trust.")


# ---- 2. Small groups -----------------------------------------------------------------


@S.exercise("funnel-plot", "funnel_plot(df, group, value)", challenge=True, hints=[
    "Per group: n = count of non-missing `value`, and the group mean. Scatter mean (y) against n (x).",
    "The center line is the overall mean of `value`. The bands are mean ± 2·s/√n, with s = overall sample std "
    "(ddof=1). Evaluate them on a grid of n, e.g. `np.geomspace(n.min(), n.max(), 200)`.",
    "`ax.set_xscale('log')` makes small and large groups both visible. Return `ax`.",
])
def _(funnel_plot):
    import matplotlib.pyplot as plt
    t = toy_deliveries()
    ax = funnel_plot(t, "customer_state", "delivery_days")
    try:
        expect(hasattr(ax, "collections") and hasattr(ax, "lines"), "funnel_plot should return a matplotlib Axes.")
        g = t.groupby("customer_state")["delivery_days"].agg(["count", "mean"])
        pts = [c.get_offsets() for c in ax.collections if len(c.get_offsets())]
        expect(len(pts) >= 1, "Draw the groups as a scatter (ax.scatter).")
        off = np.asarray(pts[0], dtype=float)
        expect(len(off) == len(g), f"Expected one point per group ({len(g)}), got {len(off)}.")
        got_pts = sorted(map(tuple, np.round(off, 6)))
        want_pts = sorted(map(tuple, np.round(g[["count", "mean"]].to_numpy(dtype=float), 6)))
        expect(got_pts == want_pts, "Points should be at x = group size (non-missing count), y = group mean.")
        expect(ax.get_xscale() == "log", "Use a logarithmic x axis (group sizes span orders of magnitude).")
        mu, s = t["delivery_days"].mean(), t["delivery_days"].std()
        lines = [(np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)) for ln in ax.lines]
        center = any(np.allclose(y, mu) for _, y in lines if len(y))
        expect(center, "Draw a horizontal line at the overall mean (ax.axhline or ax.plot).")
        for sign, name in [(1, "upper"), (-1, "lower")]:
            ok = any(len(x) > 2 and np.all(x > 0) and np.allclose(y, mu + sign * 2 * s / np.sqrt(x), rtol=1e-3)
                     for x, y in lines)
            expect(ok, f"Missing the {name} band: mean {'+' if sign > 0 else '−'} 2·s/√n, with s the overall "
                       "sample std of `value`, drawn as a line over a range of n.")
    finally:
        plt.close("all")
    return ("Points outside the funnel are surprising *given their size*. Ranking groups by their raw average "
            "ignores this and rewards the smallest ones.")


# ---- 3. Time ---------------------------------------------------------------------------


@S.exercise("monthly-orders", "monthly_orders(timestamps)", hints=[
    "`ts.dt.to_period('M')` turns each timestamp into its month. Drop NaT first.",
    "`value_counts()` only lists months that occur. Build the full range with "
    "`pd.period_range(first, last, freq='M')` and `.reindex(..., fill_value=0)`.",
    "Return integer counts, indexed by month, in calendar order.",
])
def _(monthly_orders):
    ts = pd.Series(pd.to_datetime(["2020-01-05", "2020-01-31", "2020-03-02", None]))
    got = unchanged(monthly_orders, ts)
    is_series(got, "monthly_orders(...)")
    expect(isinstance(got.index, pd.PeriodIndex), "Index the result by month periods (`.dt.to_period('M')`).")
    expect([str(p) for p in got.index] == ["2020-01", "2020-02", "2020-03"],
           f"Toy check: expected months 2020-01, 2020-02, 2020-03 (February has no orders but must appear), "
           f"got {[str(p) for p in got.index]}.")
    expect(list(got.astype(int)) == [2, 0, 1], f"Toy check: expected counts [2, 0, 1], got {list(got)}.")
    real = monthly_orders(load_olist("orders")["order_purchase_timestamp"])
    want = _real()["monthly"]
    expect(len(real) == len(want) and (real.astype(int).to_numpy() == want.to_numpy()).all(),
           "Real data: counts don't match. Did 2016-11 (zero orders) make it into your index?")
    return ("A missing month and a month with zero orders look the same in value_counts, and very different on a "
            "time axis. Always reindex to the full calendar.")


@S.exercise("delivered-rate-by-month", "delivered_rate_by_month(orders)", challenge=True, hints=[
    "Boolean `is_delivered = orders['order_status'].eq('delivered')`; its mean is a share.",
    "Group it by the purchase month: `is_delivered.groupby(month).mean()`.",
])
def _(delivered_rate_by_month):
    o = load_olist("orders")
    got = unchanged(delivered_rate_by_month, o)
    is_series(got, "delivered_rate_by_month(...)")
    want = _real()["delivered_rate"]
    got = got.sort_index()
    expect([str(p) for p in got.index] == [str(p) for p in want.index],
           "Index by purchase month, with only the months that have orders.")
    expect(np.allclose(got.to_numpy(dtype=float), want.to_numpy(dtype=float)), "Shares don't match.")
    return ("The rate collapses at the end because those orders haven't had time to be delivered yet: the outcome is "
            "*censored*, not bad. Recent periods need a follow-up window before you compare them.")


# ---- 4. Correlation ---------------------------------------------------------------------


def _check_corr_fn(fn, ref, name):
    a = fn(toy_xy()["x"], toy_xy()["y"])
    expect_close(a, ref(toy_xy()["x"], toy_xy()["y"]), f"{name}(x, y) on the toy data", tol=1e-9)
    x = pd.Series([1.0, 2, np.nan, 4, 5, 9])
    y = pd.Series([2.0, np.nan, 7, 3, 8, 8])
    expect_close(fn(x, y), ref(x, y), f"{name} with missing values", tol=1e-9)
    rng = np.random.default_rng(7)
    u = rng.normal(size=50)
    v = 0.5 * u + rng.normal(size=50)
    expect_close(fn(u, v), ref(u, v), f"{name} on NumPy arrays", tol=1e-9)
    c = fn(pd.Series([1.0, 2, 3]), pd.Series([4.0, 4, 4]))
    expect(isinstance(c, (float, np.floating)) and np.isnan(c),
           f"{name} with a constant variable should return nan (the correlation is undefined), got {c!r}.")


def _ref_spearman(x, y):
    x, y = pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))
    ok = x.notna() & y.notna()
    return _ref_pearson(x[ok].rank(), y[ok].rank())


@S.exercise("pearson-by-hand", "pearson(x, y)", hints=[
    "Convert to float arrays, then keep only the positions where neither x nor y is missing.",
    "Center both: xc = x − x̄, yc = y − ȳ. Then r = (xc · yc) / (‖xc‖ ‖yc‖), with `@` and `np.linalg.norm`.",
    "If either norm is 0 (a constant variable), return `float('nan')`.",
])
def _(pearson):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _check_corr_fn(pearson, lambda x, y: _ref_pearson(x, y), "pearson")
    return "r is the cosine of the angle between the centered data vectors. That's why it lives in [−1, 1]."


@S.exercise("spearman-by-hand", "spearman(x, y)", hints=[
    "Spearman's ρ is Pearson's r computed on ranks: reuse your `pearson`.",
    "Drop incomplete pairs FIRST, then rank: `pd.Series(x).rank()` gives tied values the average of their ranks.",
])
def _(spearman):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _check_corr_fn(spearman, _ref_spearman, "spearman")
        ties_x, ties_y = pd.Series([1.0, 1, 2, 3, 3, 3]), pd.Series([5.0, 3, 3, 1, 2, 2])
        expect_close(spearman(ties_x, ties_y), ties_x.corr(ties_y, method="spearman"),
                     "spearman with ties (use average ranks)", tol=1e-9)
    return "Ranking makes ρ blind to the shape of a monotone relation and to how far out the extreme values are."


@S.exercise("price-freight-corr", "price vs freight (items)", hints=[
    "One row of `items` is one item line; use the `price` and `freight_value` columns directly.",
    "`s.corr(t)` is Pearson; `s.corr(t, method='spearman')` is Spearman. (Or use your own functions.)",
])
def _(r_price_freight, rho_price_freight):
    expect_close(r_price_freight, _real()["r_pf"], "r_price_freight", tol=1e-4)
    expect_close(rho_price_freight, _real()["rho_pf"], "rho_price_freight", tol=1e-4)
    return ("A moderate positive association: dearer items tend to be bigger or heavier, so they cost more to ship. "
            "That's a hypothesis, not something the correlation shows. Look at the scatter before you believe a number.")


# ---- 5. Seeing relationships ----------------------------------------------------------------


@S.exercise("jitter-plot", "jitter_scatter(x, y, width, seed)", hints=[
    "`rng = np.random.default_rng(seed)`; `noise = rng.uniform(-width, width, size=len(x))`.",
    "Scatter `x + noise` against `y` (y untouched) with `alpha` below 1, on `ax` (create one if it's None).",
    "Return the Axes.",
])
def _(jitter_scatter):
    import matplotlib.pyplot as plt
    try:
        x = pd.Series([1, 1, 1, 2, 2, 3, 5, 5], dtype=float)
        y = pd.Series([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        ax = unchanged(lambda a, b: jitter_scatter(a, b, width=0.2, seed=1), x, y)
        expect(hasattr(ax, "collections"), "jitter_scatter should return a matplotlib Axes.")
        cols = [c for c in ax.collections if len(c.get_offsets())]
        expect(len(cols) == 1, f"Draw one scatter, got {len(cols)} collections.")
        off = np.asarray(cols[0].get_offsets(), dtype=float)
        expect(len(off) == len(x), f"Expected {len(x)} points, got {len(off)}.")
        dx = off[:, 0] - x.to_numpy()
        expect(np.all(np.abs(dx) <= 0.2 + 1e-12), "Each x must move by at most `width` in either direction.")
        expect(np.any(np.abs(dx) > 1e-9), "The x values weren't jittered at all.")
        expect(np.allclose(off[:, 1], y.to_numpy()), "Leave y unchanged: jitter only the discrete variable.")
        alpha = cols[0].get_alpha()
        expect(alpha is not None and alpha < 1, "Set alpha below 1 so overlapping points show up darker.")
        off2 = np.asarray(jitter_scatter(x, y, width=0.2, seed=1).collections[0].get_offsets(), dtype=float)
        expect(np.allclose(off, off2), "The same seed must give the same picture (use np.random.default_rng(seed)).")
        off3 = np.asarray(jitter_scatter(x, y, width=0.2, seed=2).collections[0].get_offsets(), dtype=float)
        expect(not np.allclose(off, off3), "Different seeds should give different jitter.")
    finally:
        plt.close("all")
    return ("Jitter is a drawing aid, not a data change: never jitter before computing statistics, and fix the seed "
            "so your figure is reproducible.")


# ---- 6. Confounding --------------------------------------------------------------------------


@S.exercise("stratified-rates", "stratified_rates(df, group, stratum, outcome)", hints=[
    "Group by BOTH columns: `df.groupby([stratum, group])[outcome].mean()`.",
    "`.unstack(group)` moves the groups into columns, so each row is a stratum.",
])
def _(stratified_rates):
    for t in (toy_carriers(), toy_carriers_uneven()):
        got = unchanged(lambda d: stratified_rates(d, "carrier", "region", "late"), t)
        is_df(got, "stratified_rates(...)")
        want = _ref_stratified(t, "carrier", "region", "late")
        expect(sorted(got.index) == sorted(want.index) and sorted(got.columns) == sorted(want.columns),
               "Rows should be the strata (regions) and columns the groups (carriers).")
        got = got.loc[want.index, want.columns]
        expect(np.allclose(got.to_numpy(dtype=float), want.to_numpy(dtype=float)), "Rates don't match.")
    return "Compare like with like: within each stratum, the third variable is held (roughly) constant."


@S.exercise("standardize", "standardize(df, group, stratum, outcome)", challenge=True, hints=[
    "Weights: the share of ALL rows in each stratum, `df[stratum].value_counts(normalize=True)`.",
    "Rates: your stratified table (strata × groups). Multiply each row by its stratum's weight, then sum down "
    "the columns: `rates.mul(w, axis=0).sum()`.",
])
def _(standardize):
    for t, name in [(toy_carriers(), "toy"), (toy_carriers_uneven(), "uneven toy")]:
        got = unchanged(lambda d: standardize(d, "carrier", "region", "late"), t)
        is_series(got, "standardize(...)")
        want = _ref_standardize(t, "carrier", "region", "late")
        expect(sorted(got.index) == sorted(want.index), "Return one value per group (index = carriers).")
        expect(np.allclose(got.loc[want.index].to_numpy(dtype=float), want.to_numpy(dtype=float)),
               f"{name}: standardized rates don't match. Use the overall stratum mix as weights for every group.")
    return ("Direct standardization asks: what would each group's rate be if all groups had the same mix of "
            "strata? A fair comparison, but only for the confounders you stratified on.")


# ---- 7. Mini-project ---------------------------------------------------------------------------


@S.exercise("late-share", "share of late deliveries", hints=[
    "Population: status 'delivered' AND order_delivered_customer_date not missing.",
    "Late = delivered date normalized to the day (`.dt.normalize()`) > order_estimated_delivery_date.",
])
def _(share_late):
    expect_close(share_late, _real()["share_late"], "share_late", tol=1e-4)
    return "Definitions matter: comparing timestamps without .dt.normalize() would mark same-day deliveries as late."


@S.exercise("reviews-one", "one review per order", hints=[
    "Sort by ['order_id', 'review_answer_timestamp'] and drop_duplicates('order_id', keep='last').",
    "Then order_id must be a key of reviews_one.",
])
def _(reviews_one):
    is_df(reviews_one, "reviews_one")
    expect(reviews_one["order_id"].is_unique, "order_id is still not unique in reviews_one.")
    expect(len(reviews_one) == _real()["n_reviewed_orders"],
           "You lost or gained orders: keep exactly one row per reviewed order.")
    full = load_olist("reviews")
    latest = full.groupby("order_id")["review_answer_timestamp"].max()
    chosen = reviews_one.set_index("order_id")["review_answer_timestamp"]
    expect((chosen.sort_index() == latest.sort_index()).all(),
           "For each order, keep the review with the latest review_answer_timestamp.")
    return "Deduplication is a *decision* (latest? first? average score?). Write down which one you made and why."


@S.exercise("late-vs-on-time", "late vs on-time review scores", hints=[
    "Join dv to reviews_one on order_id with how='inner' and validate='one_to_one'.",
    "`groupby('is_late')['review_score'].mean()`.",
])
def _(mean_score_late, mean_score_on_time):
    r = _real()
    expect_close(mean_score_late, r["mean_late"], "mean_score_late", tol=1e-4)
    expect_close(mean_score_on_time, r["mean_on_time"], "mean_score_on_time", tol=1e-4)
    return ("A big gap, but it's an association. Late orders may differ in region, seller or product; "
            "those confounders could explain part of it.")


@S.exercise("delay-score-spearman", "Spearman: delay vs score", hints=[
    "delay_days = (order_delivered_customer_date − order_estimated_delivery_date).dt.total_seconds() / 86400. "
    "It's negative for early deliveries.",
    "Compute it on the joined table (delivered orders with their single review), then `.corr(..., method='spearman')`.",
])
def _(rho_delay_score):
    expect_close(rho_delay_score, _real()["rho_delay"], "rho_delay_score", tol=1e-4)
    return ("A continuous delay keeps information the late/on-time split throws away. Still, a monotone association "
            "is not a dose-response curve: plot the mean score per delay bucket to see its shape.")


@S.exercise("gap-within-states", "the gap within each state", challenge=True, hints=[
    "Add customer_state to the joined table (customers, validate='many_to_one'). Keep states with ≥ 1,000 rows.",
    "Mean score per (state, is_late): `groupby(['customer_state', 'is_late'])['review_score'].mean().unstack()`.",
    "gap = late column − on-time column, a Series indexed by state.",
])
def _(gap_by_state):
    is_series(gap_by_state, "gap_by_state")
    want = _real()["gap_by_state"]
    got = gap_by_state.copy()
    got.index = got.index.astype(str)
    expect(sorted(got.index) == list(want.index),
           f"Expected the {len(want)} states with at least 1,000 reviewed delivered orders, got {len(got)}.")
    expect(np.allclose(got.loc[want.index].to_numpy(dtype=float), want.to_numpy()),
           "Gaps don't match. Gap = mean score of late orders − mean score of on-time orders, within each state.")
    return ("The gap survives within every large state, so state alone doesn't explain it. That rules out one "
            "confounder, not all of them.")
