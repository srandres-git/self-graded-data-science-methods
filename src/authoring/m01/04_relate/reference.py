"""⚠️ SPOILERS — reference solutions for Module I · 04. Run by verify.py; every check must pass."""
import os

os.environ["SELFGRADE_NO_LOG"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from selfgrade.data import load_olist  # noqa: E402
from selfgrade.m01 import relate as G  # noqa: E402

orders, items = load_olist("orders"), load_olist("items")
reviews, customers = load_olist("reviews"), load_olist("customers")


def state_delivery_summary(df):
    g = df.groupby("customer_state")["delivery_days"]
    out = pd.DataFrame({"n_orders": g.count(), "mean_days": g.mean(), "median_days": g.median(),
                        "p90_days": g.quantile(0.9)})
    return out.sort_values("n_orders", ascending=False)


def funnel_plot(df, group, value, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    g = df.groupby(group)[value].agg(["count", "mean"])
    g = g[g["count"] > 0]
    mu, s = df[value].mean(), df[value].std()
    ax.scatter(g["count"], g["mean"], s=18)
    n = np.geomspace(g["count"].min(), g["count"].max(), 200)
    ax.axhline(mu, color="gray", lw=1)
    ax.plot(n, mu + 2 * s / np.sqrt(n), color="gray", ls="--")
    ax.plot(n, mu - 2 * s / np.sqrt(n), color="gray", ls="--")
    ax.set_xscale("log")
    return ax


def monthly_orders(timestamps):
    m = timestamps.dropna().dt.to_period("M")
    vc = m.value_counts()
    return vc.reindex(pd.period_range(vc.index.min(), vc.index.max(), freq="M"), fill_value=0).astype(int)


def delivered_rate_by_month(orders):
    month = orders["order_purchase_timestamp"].dt.to_period("M")
    return orders["order_status"].eq("delivered").groupby(month).mean()


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    xc, yc = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    den = np.linalg.norm(xc) * np.linalg.norm(yc)
    return float(xc @ yc / den) if den > 0 else float("nan")


def spearman(x, y):
    x, y = pd.Series(np.asarray(x, dtype=float)), pd.Series(np.asarray(y, dtype=float))
    ok = x.notna() & y.notna()
    return pearson(x[ok].rank(), y[ok].rank())


def jitter_scatter(x, y, width=0.15, seed=0, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    ax.scatter(x + rng.uniform(-width, width, size=len(x)), np.asarray(y, dtype=float), s=8, alpha=0.3)
    return ax


def corr_heatmap(df, method="pearson", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    c = df.corr(method=method, numeric_only=True)
    im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c)), c.columns)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.figure.colorbar(im, ax=ax)
    return ax


def stratified_rates(df, group, stratum, outcome):
    return df.groupby([stratum, group])[outcome].mean().unstack(group)


def standardize(df, group, stratum, outcome):
    w = df[stratum].value_counts(normalize=True)
    return stratified_rates(df, group, stratum, outcome).mul(w, axis=0).sum(axis=0)


dv = orders[orders["order_status"].eq("delivered") & orders["order_delivered_customer_date"].notna()].copy()
dv["is_late"] = dv["order_delivered_customer_date"].dt.normalize() > dv["order_estimated_delivery_date"]
reviews_one = reviews.sort_values(["order_id", "review_answer_timestamp"]).drop_duplicates("order_id", keep="last")
j = dv.merge(reviews_one[["order_id", "review_score"]], on="order_id", how="inner", validate="one_to_one")
means = j.groupby("is_late")["review_score"].mean()
j["delay_days"] = (j["order_delivered_customer_date"] - j["order_estimated_delivery_date"]).dt.total_seconds() / 86400
js = j.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left", validate="many_to_one")
n = js.groupby("customer_state").size()
big = js[js["customer_state"].isin(n[n >= 1000].index)]
m = big.groupby(["customer_state", "is_late"])["review_score"].mean().unstack()
gap_by_state = m[True] - m[False]

G.S.verify_reference({
    "state-delivery-summary": G.check("state-delivery-summary", state_delivery_summary),
    "funnel-plot": G.check("funnel-plot", funnel_plot),
    "monthly-orders": G.check("monthly-orders", monthly_orders),
    "delivered-rate-by-month": G.check("delivered-rate-by-month", delivered_rate_by_month),
    "pearson-by-hand": G.check("pearson-by-hand", pearson),
    "spearman-by-hand": G.check("spearman-by-hand", spearman),
    "price-freight-corr": G.check("price-freight-corr", items["price"].corr(items["freight_value"]),
                                  items["price"].corr(items["freight_value"], method="spearman")),
    "jitter-plot": G.check("jitter-plot", jitter_scatter),
    "stratified-rates": G.check("stratified-rates", stratified_rates),
    "standardize": G.check("standardize", standardize),
    "late-share": G.check("late-share", dv["is_late"].mean()),
    "reviews-one": G.check("reviews-one", reviews_one),
    "late-vs-on-time": G.check("late-vs-on-time", means[True], means[False]),
    "delay-score-spearman": G.check("delay-score-spearman", j["delay_days"].corr(j["review_score"], method="spearman")),
    "gap-within-states": G.check("gap-within-states", gap_by_state),
})
