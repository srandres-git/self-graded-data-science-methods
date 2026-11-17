"""⚠️ SPOILERS — reference solutions for Module I · 05. Run by verify.py; every check must pass."""
import os

os.environ["SELFGRADE_NO_LOG"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from selfgrade.m01 import charts as G  # noqa: E402


def sorted_bars(counts, ylabel, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    s = counts.sort_values(ascending=False)
    ax.bar(s.index.astype(str), s.to_numpy())
    ax.set(ylabel=ylabel, title=title)
    return ax


def top_categories_barh(item_table, n=10, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 0.4 * n + 1))
    s = item_table.groupby("category")["price"].sum().nlargest(n).sort_values()
    ax.barh(s.index, s.to_numpy())
    ax.set(xlabel="Revenue (BRL, item prices)", title=f"Top {n} categories by revenue")
    return ax


def late_share_chart(order_table, min_orders=1000, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    g = order_table.groupby("customer_state")["is_late"].agg(["count", "mean"])
    s = g.loc[g["count"] >= min_orders, "mean"].astype(float).sort_values(ascending=False)
    ax.bar(s.index, s.to_numpy())
    ax.set(ylabel="Share of delivered orders that arrived late",
           title=f"Late deliveries by state (states with ≥ {min_orders:,} delivered orders)")
    return ax


def composition_chart(counts, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    shares = counts.div(counts.sum(axis=1), axis=0)
    shares.plot.bar(stacked=True, ax=ax, rot=0)
    ax.set(ylim=(0, 1), ylabel="Share of orders in the group")
    ax.legend(title=counts.columns.name, bbox_to_anchor=(1, 1))
    return ax


def dot_plot(shares, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    s = shares.sort_values()
    ax.hlines(s.index, 0, s.to_numpy(), color="0.7")
    ax.scatter(s.to_numpy(), s.index, zorder=3)
    ax.set(xlabel="Share of orders", xlim=(0, None))
    return ax


def mosaic(counts, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    total = counts.to_numpy().sum()
    colors = plt.get_cmap("tab10").colors
    x, centers = 0.0, []
    for row, r in counts.iterrows():
        w = r.sum() / total
        y = 0.0
        for j, (col, n) in enumerate(r.items()):
            h = n / r.sum() if r.sum() else 0
            ax.add_patch(Rectangle((x, y), w, h, facecolor=colors[j % 10], edgecolor="white",
                                   label=str(col) if x == 0 else None))
            y += h
        centers.append(x + w / 2)
        x += w
    ax.set(xlim=(0, 1), ylim=(0, 1), ylabel="Share within the column")
    ax.set_xticks(centers, counts.index.astype(str))
    ax.legend(bbox_to_anchor=(1, 1))
    return ax


def corr_heatmap(corr, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    n = len(corr)
    ax.set_xticks(range(n), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), corr.index)
    ax.figure.colorbar(im, ax=ax, label="Pearson correlation")
    return ax


def deviation_bars(order_table, min_orders=1000, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    national = float(order_table["is_late"].mean())
    g = order_table.groupby("customer_state")["is_late"].agg(["count", "mean"])
    dev = (g.loc[g["count"] >= min_orders, "mean"].astype(float) - national).sort_values(ascending=False)
    ax.bar(dev.index, dev.to_numpy(), color=np.where(dev.to_numpy() > 0, "#c0392b", "#2471a3"))
    ax.axhline(0, color="black", lw=0.8)
    ax.set(ylabel="Late share − national share")
    return ax


def is_monotone_luminance(cmap_name):
    rgb = matplotlib.colormaps[cmap_name](np.linspace(0, 1, 256))[:, :3]
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    d = np.diff(lin @ np.array([0.2126, 0.7152, 0.0722]))
    return bool((d >= -1e-12).all() or (d <= 1e-12).all())


def save_png(fig, path, width_px):
    fig.savefig(path, dpi=width_px / fig.get_figwidth())


def purchase_heatmap(timestamps, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3.5))
    ts = timestamps.dropna()
    grid = (pd.crosstab(ts.dt.dayofweek, ts.dt.hour)
            .reindex(index=range(7), columns=range(24), fill_value=0))
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2))
    ax.set(xlabel="Hour of purchase", title="When are orders placed?")
    ax.figure.colorbar(im, ax=ax, label="Orders")
    return ax


G.S.verify_reference({
    "sorted-bars": G.check("sorted-bars", sorted_bars),
    "sorted-barh": G.check("sorted-barh", top_categories_barh),
    "fixed-state-chart": G.check("fixed-state-chart", late_share_chart),
    "composition-chart": G.check("composition-chart", composition_chart),
    "dot-plot": G.check("dot-plot", dot_plot),
    "mosaic": G.check("mosaic", mosaic),
    "corr-heatmap": G.check("corr-heatmap", corr_heatmap),
    "deviation-bars": G.check("deviation-bars", deviation_bars),
    "luminance-monotone": G.check("luminance-monotone", is_monotone_luminance),
    "export-size": G.check("export-size", save_png),
    "purchase-heatmap": G.check("purchase-heatmap", purchase_heatmap),
})
