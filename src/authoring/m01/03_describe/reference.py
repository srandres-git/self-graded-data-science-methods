"""⚠️ SPOILERS — reference solutions for Module I · 03. Run by verify.py; every check must pass."""
import os

os.environ["SELFGRADE_NO_LOG"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from selfgrade.data import load_olist  # noqa: E402
from selfgrade.m01 import describe as G  # noqa: E402

orders, items, customers = load_olist("orders"), load_olist("items"), load_olist("customers")
item_totals = items.groupby("order_id").agg(n_items=("price", "size"), items_value=("price", "sum"))
order_table = orders.merge(item_totals, on="order_id", how="left", validate="one_to_one")


def freq_table(s):
    n = s.value_counts(dropna=False)
    return pd.DataFrame({"count": n, "share": n / n.sum()})


def quantile_linear(x, p):
    a = np.sort(pd.Series(x, dtype="float64").dropna().to_numpy())
    h = (len(a) - 1) * p
    lo = int(np.floor(h))
    hi = min(lo + 1, len(a) - 1)
    return float(a[lo] + (h - lo) * (a[hi] - a[lo]))


def spread_summary(s):
    s = s.dropna()
    return pd.Series({"std": s.std(), "iqr": s.quantile(.75) - s.quantile(.25),
                      "mad_scaled": 1.4826 * (s - s.median()).abs().median()})


def skewness(x):
    a = pd.Series(x, dtype="float64").dropna().to_numpy()
    n, d = len(a), a - a.mean()
    return float(np.sqrt(n * (n - 1)) / (n - 2) * (d ** 3).mean() / (d ** 2).mean() ** 1.5)


def log_histogram(prices, n_bins=40):
    x = prices.dropna()
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(x, bins=edges)
    ax.set_xscale("log")
    ax.set(xlabel="price (BRL, log scale)", ylabel="item rows")
    return ax


def fd_width(x):
    s = pd.Series(x, dtype="float64").dropna()
    return float(2 * (s.quantile(.75) - s.quantile(.25)) * len(s) ** (-1 / 3))


def ecdf(x):
    a = pd.Series(x, dtype="float64").dropna().to_numpy()
    xs, counts = np.unique(a, return_counts=True)
    return xs, np.cumsum(counts) / len(a)


def qq_points(x):
    a = np.sort(pd.Series(x, dtype="float64").dropna().to_numpy())
    n = len(a)
    return stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n), a


def tukey_flags(s, k=1.5):
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def delivery_days(orders):
    pop = orders[orders["order_status"].eq("delivered") & orders["order_delivered_customer_date"].notna()]
    return (pop["order_delivered_customer_date"] - pop["order_purchase_timestamp"]).dt.total_seconds() / 86400


def delivery_boxplot(deliveries):
    top = deliveries["customer_state"].value_counts().head(5).index
    sub = deliveries[deliveries["customer_state"].isin(top)]
    order = sub.groupby("customer_state")["days"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.boxplot([sub.loc[sub["customer_state"] == s, "days"] for s in order], tick_labels=list(order), showfliers=False)
    ax.set(ylabel="delivery time (days)", xlabel="customer state")
    return ax


price = items["price"]
srt = np.sort(price.to_numpy())


def _median_after(k):
    a = srt.copy()
    a[:k] = 1e12
    return np.median(a)


lo, hi = 0, len(srt)
while lo < hi:
    mid = (lo + hi) // 2
    if _median_after(mid) > 1000:
        hi = mid
    else:
        lo = mid + 1

st = orders["order_status"]
d = delivery_days(orders)

G.S.verify_reference({
    "freq-table": G.check("freq-table", freq_table),
    "status-shares": G.check("status-shares", st.eq("canceled").mean(),st[st.ne("delivered")].eq("canceled").mean()),
    "quantile-linear": G.check("quantile-linear", quantile_linear),
    "spread-summary": G.check("spread-summary", spread_summary),
    "order-value-spread": G.check("order-value-spread", spread_summary(order_table["items_value"])),
    "breakdown": G.check("breakdown", lo, 1),
    "skewness": G.check("skewness", skewness),
    "log-skew": G.check("log-skew", price.skew(), np.log10(price).skew()),
    "log-histogram": G.check("log-histogram", log_histogram),
    "fd-bins": G.check("fd-bins", fd_width, int(np.ceil((price.max() - price.min()) / fd_width(price)))),
    "ecdf": G.check("ecdf", ecdf),
    "qq-points": G.check("qq-points", qq_points),
    "tukey-flags": G.check("tukey-flags", tukey_flags),
    "tukey-shares": G.check("tukey-shares", tukey_flags(price).mean(), tukey_flags(np.log10(price)).mean()),
    "delivery-summary": G.check("delivery-summary", d.median(), d.quantile(.9), d.gt(30).mean()),
    "delivery-boxplot": G.check("delivery-boxplot", delivery_boxplot),
})
