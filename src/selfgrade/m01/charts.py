"""Grading for Module I · 05 — Honest charts (matplotlib objects, encodings, parts of a whole, color, export).

⚠️ SPOILERS: the reference computations used for grading live in this file.
Try the exercises first; read this afterwards if you want to compare approaches.

Chart exercises return a matplotlib Axes; the checks inspect it (bar heights and order, tick labels,
limits, colormap, normalization) instead of looking at pixels.
"""
from __future__ import annotations

import os
import tempfile
from functools import lru_cache

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.patches import Rectangle

from ..core import Session, expect, unchanged
from ..data import load_olist

S = Session("m01/05", "Honest charts")
check, hint, predict, progress = S.check, S.hint, S.predict, S.progress

VALUE_BINS = [0, 50, 100, 200, 500, np.inf]
VALUE_LABELS = ["<50", "50–100", "100–200", "200–500", "500+"]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

DIVERGING = {"coolwarm", "bwr", "seismic", "RdBu", "RdYlBu", "RdYlGn", "PuOr", "BrBG", "PiYG", "PRGn",
             "RdGy", "Spectral", "berlin", "managua", "vanimo"}
DIVERGING |= {n + "_r" for n in DIVERGING}

# ===========================================================================
# Toy data
# ===========================================================================


def toy_counts() -> pd.Series:
    """Orders per status, deliberately not sorted."""
    return pd.Series({"shipped": 7, "delivered": 30, "canceled": 2, "invoiced": 4}, name="orders")


def toy_item_revenue() -> pd.DataFrame:
    """Item rows with a category label and a price (BRL)."""
    return pd.DataFrame({
        "category": ["bed_bath_table", "health_beauty", "toys", "bed_bath_table", "watches_gifts",
                     "toys", "auto", "toys", "watches_gifts"],
        "price": [50.0, 200.0, 15.0, 60.0, 120.0, 15.0, 80.0, 15.0, 30.0],
    })


def toy_late_orders() -> pd.DataFrame:
    """Orders with a state and an is_late flag (missing when the order wasn't delivered)."""
    rows = ([("SP", "delivered", False)] * 9 + [("SP", "delivered", True)] + [("SP", "canceled", None)]
            + [("RJ", "delivered", True)] * 2 + [("RJ", "delivered", False)] * 2 + [("AC", "delivered", True)])
    df = pd.DataFrame(rows, columns=["customer_state", "order_status", "is_late"])
    df["is_late"] = df["is_late"].astype("boolean")
    return df


def toy_crosstab() -> pd.DataFrame:
    """Counts: order-value band (rows) × first payment type (columns)."""
    return pd.DataFrame([[3, 6, 1], [4, 14, 2], [1, 8, 1]],
                        index=pd.Index(["<50", "50–100", "100+"], name="value_band"),
                        columns=pd.Index(["boleto", "credit_card", "voucher"], name="first_payment_type"))


def toy_corr() -> pd.DataFrame:
    names = ["price", "freight", "days"]
    return pd.DataFrame([[1.0, 0.4, -0.2], [0.4, 1.0, 0.1], [-0.2, 0.1, 1.0]], index=names, columns=names)


def toy_timestamps() -> pd.Series:
    return pd.Series(pd.to_datetime([
        "2018-01-01 10:15", "2018-01-01 10:50", "2018-01-01 22:05",   # Monday
        "2018-01-02 23:59",                                           # Tuesday
        "2018-01-07 00:01", "2018-01-07 00:30",                       # Sunday
    ]), name="order_purchase_timestamp")


# ===========================================================================
# Reference computations
# ===========================================================================


def _ref_order_table() -> pd.DataFrame:
    orders, items, payments = load_olist("orders"), load_olist("items"), load_olist("payments")
    customers = load_olist("customers")
    first = (payments.sort_values(["order_id", "payment_sequential"]).drop_duplicates("order_id")
             [["order_id", "payment_type"]].rename(columns={"payment_type": "first_payment_type"}))
    per_order = items.groupby("order_id").agg(n_items=("price", "size"), items_value=("price", "sum"),
                                              freight_value=("freight_value", "sum"))
    t = (orders.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left", validate="many_to_one")
         .merge(per_order, on="order_id", how="left", validate="one_to_one")
         .merge(first, on="order_id", how="left", validate="one_to_one"))
    dlv = t["order_status"].eq("delivered") & t["order_delivered_customer_date"].notna()
    t["delivery_days"] = ((t["order_delivered_customer_date"] - t["order_purchase_timestamp"])
                          .dt.total_seconds() / 86400).where(dlv)
    t["delay_days"] = (t["order_delivered_customer_date"].dt.normalize()
                       - t["order_estimated_delivery_date"]).dt.days.where(dlv)
    t["is_late"] = (t["delay_days"] > 0).astype("boolean").where(dlv)
    return t


def _ref_item_table() -> pd.DataFrame:
    items, products, categories = load_olist("items"), load_olist("products"), load_olist("categories")
    lab = products[["product_id", "product_category_name"]].merge(
        categories, on="product_category_name", how="left", validate="many_to_one")
    lab["category"] = lab["product_category_name_english"].fillna(lab["product_category_name"]).fillna("UNKNOWN")
    return items.merge(lab[["product_id", "category"]], on="product_id", how="left", validate="many_to_one")


def _ref_late_share(order_table: pd.DataFrame, min_orders: int) -> pd.Series:
    g = order_table.groupby("customer_state")["is_late"].agg(["count", "mean"])
    return g.loc[g["count"] >= min_orders, "mean"].astype(float).sort_values(ascending=False)


def _ref_deviation(order_table: pd.DataFrame, min_orders: int) -> pd.Series:
    national = float(order_table["is_late"].mean())
    return (_ref_late_share(order_table, min_orders) - national).sort_values(ascending=False)


def _ref_heat(ts: pd.Series) -> np.ndarray:
    ts = ts.dropna()
    return (pd.crosstab(ts.dt.dayofweek, ts.dt.hour)
            .reindex(index=range(7), columns=range(24), fill_value=0).to_numpy())


def _ref_monotone(name: str, n: int = 256) -> bool:
    rgb = matplotlib.colormaps[name](np.linspace(0, 1, n))[:, :3]
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lum = lin @ np.array([0.2126, 0.7152, 0.0722])
    d = np.diff(lum)
    return bool((d >= -1e-12).all() or (d <= 1e-12).all())


@lru_cache(maxsize=1)
def _real() -> dict:
    t = _ref_order_table()
    it = _ref_item_table()
    band = pd.cut(t["items_value"], VALUE_BINS, right=False, labels=VALUE_LABELS)
    return {
        "order_table": t,
        "item_table": it,
        "status_counts": t["order_status"].value_counts(),
        "top10": it.groupby("category")["price"].sum().sort_values(ascending=False).head(10),
        "late_share": _ref_late_share(t, 1000),
        "deviation": _ref_deviation(t, 1000),
        "pay_by_band": pd.crosstab(band, t["first_payment_type"]),
        "pay_shares": t["first_payment_type"].value_counts(normalize=True),
        "corr": t[["items_value", "freight_value", "n_items", "delivery_days", "delay_days"]].corr(),
        "heat": _ref_heat(t["order_purchase_timestamp"]),
        "timestamps": t["order_purchase_timestamp"],
    }


# ===========================================================================
# Inspecting what the learner drew
# ===========================================================================


def _get_ax(out, name: str) -> Axes:
    if isinstance(out, plt.Figure):
        expect(len(out.axes) >= 1, f"`{name}` returned a Figure without Axes.")
        out = out.axes[0]
    expect(isinstance(out, Axes), f"`{name}` should return the matplotlib Axes it drew on, got {type(out).__name__}. "
           "End the function with `return ax`.")
    out.figure.canvas.draw()
    return out


def _close(*axes) -> None:
    for ax in axes:
        if isinstance(ax, Axes):
            plt.close(ax.figure)


def _tick_map(ax: Axes, horizontal: bool):
    ticks = ax.get_yticks() if horizontal else ax.get_xticks()
    labels = [t.get_text() for t in (ax.get_yticklabels() if horizontal else ax.get_xticklabels())]
    return np.asarray(ticks, dtype=float), labels


def _label_at(pos: float, ticks, labels) -> str:
    if len(ticks) == 0:
        return ""
    i = int(np.argmin(np.abs(ticks - pos)))
    return labels[i] if abs(ticks[i] - pos) < 0.45 else ""


def _ordered(pos: list[float], horizontal: bool, ax: Axes) -> list[int]:
    """Indices in reading order: left→right for vertical bars, top→bottom for horizontal ones."""
    if horizontal:
        return sorted(range(len(pos)), key=lambda i: pos[i], reverse=not ax.yaxis_inverted())
    return sorted(range(len(pos)), key=lambda i: pos[i], reverse=ax.xaxis_inverted())


def bars_of(ax: Axes, horizontal: bool = False) -> list[tuple[str, float]]:
    """(tick label, bar value) for every bar, in reading order."""
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    if horizontal:
        pos = [p.get_y() + p.get_height() / 2 for p in rects]
        val = [p.get_width() for p in rects]
    else:
        pos = [p.get_x() + p.get_width() / 2 for p in rects]
        val = [p.get_height() for p in rects]
    ticks, labels = _tick_map(ax, horizontal)
    return [(_label_at(pos[i], ticks, labels), float(val[i])) for i in _ordered(pos, horizontal, ax)]


def points_of(ax: Axes) -> list[tuple[str, float]]:
    """(y tick label, x value) for every marker (scatter or plot with a marker), top→bottom."""
    xs, ys = [], []
    for c in ax.collections:
        if isinstance(c, PathCollection):
            off = np.asarray(c.get_offsets(), dtype=float)
            xs += list(off[:, 0]); ys += list(off[:, 1])
    for ln in ax.lines:
        if ln.get_marker() not in (None, "None", "", " ") and ln.get_linestyle() in ("None", "", " ", "none"):
            xs += list(np.asarray(ln.get_xdata(), dtype=float)); ys += list(np.asarray(ln.get_ydata(), dtype=float))
    ticks, labels = _tick_map(ax, horizontal=True)
    order = _ordered(ys, True, ax)
    return [(_label_at(ys[i], ticks, labels), xs[i]) for i in order]


def mappable_of(ax: Axes):
    if ax.images:
        im = ax.images[0]
        return im, np.asarray(im.get_array(), dtype=float)
    for c in ax.collections:
        if isinstance(c, QuadMesh):
            return c, np.asarray(c.get_array(), dtype=float)
    return None, None


def _cmap_name(m) -> str:
    return m.get_cmap().name


def _tidy(fn):
    """Close every figure a check created, whether it passes or fails (so failures don't leak into later plots)."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **k):
        before = set(plt.get_fignums())
        try:
            return fn(*a, **k)
        finally:
            for n in set(plt.get_fignums()) - before:
                plt.close(n)
    return wrapper


def _has_text(s: str) -> bool:
    return bool(s and s.strip())


# ===========================================================================
# Predictions
# ===========================================================================

S.prediction("axes-shape", (2, 3),
    "`plt.subplots(nrows, ncols)` returns the Figure and a NumPy array of Axes with shape (nrows, ncols): "
    "`axes[0, 2]` is the top-right panel. With a single row or column it's 1-D, and with 1×1 it's a single Axes.",
    compare=lambda v, a: isinstance(v, (tuple, list)) and tuple(int(x) for x in v) == a)
S.prediction("truncated-ratio", 2,
    "The bars start at 90, so their drawn lengths are 95−90 = 5 and 100−90 = 10: the second looks twice as tall "
    "although the values differ by about 5%. A bar encodes its value by its length, so the length must start at 0.",
    compare=lambda v, a: abs(float(v) - a) < 0.05)
S.prediction("stacked-baseline", "a",
    "Only the bottom segments share a common baseline (the axis). Every other segment starts wherever the one "
    "below it ended, so comparing them means comparing lengths with no common start: the reader is reduced to "
    "guessing. That's why stacked bars are good for totals and the bottom part, and poor for the middle parts.",
    compare=lambda v, a: str(v).strip().lower().strip(")") == a)
S.prediction("mosaic-area", "a",
    "Width = P(row) and height = P(column | row), so area = P(row)·P(column | row) = P(row, column), the joint share: "
    "the tile's count divided by the grand total. All the tiles together fill the unit square.",
    compare=lambda v, a: str(v).strip().lower().strip(")") == a)
S.prediction("palette-type", ("qualitative", "sequential", "diverging"),
    "Payment type is nominal (no order) → qualitative: distinct hues of similar lightness. A count goes from low "
    "to high → sequential: lightness changes monotonically. A change vs last year has a meaningful middle (0) "
    "and two directions → diverging: two hues that meet at a neutral color for 0.",
    compare=lambda v, a: isinstance(v, (tuple, list)) and tuple(str(x).strip().lower() for x in v) == a)

# ===========================================================================
# Exercises
# ===========================================================================

# ---- 1. Figures and Axes -------------------------------------------------------


@S.exercise("sorted-bars", "sorted_bars(counts, ylabel, title, ax=None)", hints=[
    "Start with `if ax is None: fig, ax = plt.subplots(figsize=(7, 4))`, and finish with `return ax`.",
    "Sort first: `counts.sort_values(ascending=False)`. Then `ax.bar(s.index, s.to_numpy())`.",
    "`ax.set_ylabel(...)` and `ax.set_title(...)` (or `ax.set(ylabel=..., title=...)`).",
])
@_tidy
def _(sorted_bars):
    c = toy_counts()
    ax = _get_ax(unchanged(lambda s: sorted_bars(s, "Orders", "Orders by status"), c), "sorted_bars(...)")
    got = bars_of(ax)
    want = list(c.sort_values(ascending=False).items())
    expect(len(got) == len(want), f"Expected {len(want)} bars, found {len(got)}.")
    expect([lab for lab, _ in got] == [k for k, _ in want],
           f"Bars should be sorted from largest to smallest: {[k for k, _ in want]}. Got {[lab for lab, _ in got]}.")
    expect(np.allclose([v for _, v in got], [v for _, v in want]), "The bar heights don't match the counts.")
    expect(ax.get_ylim()[0] == 0, "A bar's length is its value, so the y axis must start at 0.")
    expect(ax.get_ylabel() == "Orders" and ax.get_title() == "Orders by status",
           "Use the `ylabel` and `title` arguments for the y-axis label and the title.")
    _close(ax)
    fig, given = plt.subplots()
    out = sorted_bars(c, "Orders", "t", ax=given)
    expect(out is given and len(given.patches) == len(c),
           "When an `ax` is passed in, draw on it (don't create a new figure) and return it.")
    _close(given)
    real = _real()["status_counts"]
    ax = _get_ax(sorted_bars(real.sample(frac=1, random_state=0), "Orders", "t"), "sorted_bars(...)")
    expect([lab for lab, _ in bars_of(ax)] == list(real.index), "Real data: the bars aren't sorted by count.")
    _close(ax)
    return ("A function that takes an optional `ax` and returns it composes: you can call it alone, or place its "
            "chart in any panel of a larger figure.")


# ---- 2. How we read charts --------------------------------------------------------


@S.exercise("sorted-barh", "top_categories_barh(item_table, n=10, ax=None)", hints=[
    "Revenue per category: `item_table.groupby('category')['price'].sum()`, then the n largest with `.nlargest(n)`.",
    "`ax.barh(labels, values)` draws the FIRST value at the BOTTOM. Reverse the order "
    "(`.sort_values()` ascending), or call `ax.invert_yaxis()`, so the largest ends up on top.",
    "Label the value axis with its unit, e.g. `ax.set_xlabel('Revenue (BRL)')`.",
])
@_tidy
def _(top_categories_barh):
    t = toy_item_revenue()
    ax = _get_ax(unchanged(lambda d: top_categories_barh(d, 3), t), "top_categories_barh(...)")
    got = bars_of(ax, horizontal=True)
    want = list(t.groupby("category")["price"].sum().nlargest(3).items())
    expect(len(got) == 3, f"With n=3, expected 3 bars, found {len(got)}.")
    expect([lab for lab, _ in got] == [k for k, _ in want],
           f"Top to bottom, the bars should read {[k for k, _ in want]} (largest on top). Got {[lab for lab, _ in got]}.")
    expect(np.allclose([v for _, v in got], [v for _, v in want]), "Bar lengths should be the revenue sums.")
    expect(ax.get_xlim()[0] == 0, "Horizontal bars encode value by length too: the x axis must start at 0.")
    expect("BRL" in ax.get_xlabel(), "Label the x axis with the unit (BRL).")
    _close(ax)
    real = _real()["top10"]
    ax = _get_ax(top_categories_barh(_real()["item_table"].copy()), "top_categories_barh(...)")
    got = bars_of(ax, horizontal=True)
    expect([lab for lab, _ in got] == list(real.index), "Real data: the top 10 categories don't match.")
    _close(ax)
    return ("Horizontal bars give long labels room to be read without tilting your head, and sorting turns "
            "the chart into a ranking you can read top to bottom.")


@S.exercise("fixed-state-chart", "late_share_chart(order_table, min_orders=1000, ax=None)", hints=[
    "The question is about a *rate*: per state, the share of delivered orders that arrived late. "
    "`is_late` is missing for orders that weren't delivered, and `.mean()` skips missing values.",
    "Per state, compute both `count` (orders with a non-missing is_late) and `mean`, keep states with "
    "count ≥ min_orders, and sort the means descending.",
    "Draw vertical bars and don't touch the y limits: matplotlib starts bars at 0 by itself.",
])
@_tidy
def _(late_share_chart):
    toy = toy_late_orders()
    ax = _get_ax(unchanged(lambda d: late_share_chart(d, 3), toy), "late_share_chart(...)")
    got = bars_of(ax)
    want = _ref_late_share(toy, 3)
    labels = [lab for lab, _ in got]
    expect("AC" not in labels, "AC has a single delivered order: below min_orders, so it should be left out.")
    expect(labels == list(want.index), f"Expected bars {list(want.index)} (sorted by late share, highest first), "
           f"got {labels}.")
    vals = np.array([v for _, v in got])
    if vals.max() > 1.0001:
        vals = vals / 100
    expect(np.allclose(vals, want.to_numpy()),
           "The bar heights should be late SHARES (late ÷ delivered orders in the state), not counts of late orders. "
           f"Expected {np.round(want.to_numpy(), 3).tolist()}.")
    expect(ax.get_ylim()[0] == 0, "Bars must start at 0: the buggy chart's truncated axis is one of the bugs.")
    expect(_has_text(ax.get_ylabel()), "Label the y axis (what is the share of?).")
    _close(ax)
    real = _real()["late_share"]
    ax = _get_ax(late_share_chart(_real()["order_table"].copy()), "late_share_chart(...)")
    expect([lab for lab, _ in bars_of(ax)] == list(real.index), "Real data: the states or their order don't match.")
    _close(ax)
    return ("SP has the most late orders simply because it has the most orders; per order, it's one of the most "
            "punctual states. When groups differ in size, compare rates and say what the denominator is.")


# ---- 3. Parts of a whole --------------------------------------------------------------


@S.exercise("composition-chart", "composition_chart(counts, ax=None)", hints=[
    "Turn counts into shares within each row: `counts.div(counts.sum(axis=1), axis=0)`.",
    "`shares.plot.bar(stacked=True, ax=ax)` draws one stacked bar per row, one colored segment per column.",
    "Fix the y axis to (0, 1), label it as a share, and keep the legend: it says which color is which part.",
])
@_tidy
def _(composition_chart):
    ct = toy_crosstab()
    ax = _get_ax(unchanged(composition_chart, ct), "composition_chart(...)")
    shares = ct.div(ct.sum(axis=1), axis=0)
    conts = [c for c in ax.containers if len(c.patches) == len(ct)]
    expect(len(conts) == ct.shape[1],
           f"Expected one bar series per column ({ct.shape[1]}): {list(ct.columns)}. Found {len(conts)}.")
    labels = [c.get_label() for c in conts]
    expect(sorted(labels) == sorted(map(str, ct.columns)),
           f"Each series should be labeled with its column name (for the legend). Got {labels}.")
    ticks, ticklabels = _tick_map(ax, False)
    for c in conts:
        for p in c.patches:
            row = _label_at(p.get_x() + p.get_width() / 2, ticks, ticklabels)
            expect(row in shares.index, f"Couldn't match a bar to a row label (got {row!r}). Use the rows as x ticks.")
            expect(np.isclose(p.get_height(), shares.loc[row, c.get_label()]),
                   f"Segment ({row}, {c.get_label()}) should be {shares.loc[row, c.get_label()]:.3f}: "
                   "the share of that column WITHIN the row.")
    expect(np.allclose(ax.get_ylim(), (0, 1)), "Fix the y axis to (0, 1): every bar is a whole.")
    expect(ax.get_legend() is not None, "Keep a legend: without it the colors mean nothing.")
    _close(ax)
    real = _real()["pay_by_band"]
    ax = _get_ax(composition_chart(real.copy()), "composition_chart(...)")
    expect(len([c for c in ax.containers if len(c.patches) == len(real)]) == real.shape[1],
           "Real data: expected one series per payment type.")
    _close(ax)
    return ("A 100% stacked bar answers 'what is the mix within each group?' and hides group sizes. Pair it "
            "with the n of each bar, or with a separate chart of totals.")


@S.exercise("dot-plot", "dot_plot(shares, ax=None)", hints=[
    "Sort ascending first, so that after plotting the largest is at the top: `s = shares.sort_values()`.",
    "`ax.scatter(s.to_numpy(), s.index)` puts one dot per category, with categories on the y axis. "
    "`ax.hlines(s.index, 0, s.to_numpy())` adds the 'lollipop' sticks.",
    "Label the x axis with what the value is (a share of orders).",
])
@_tidy
def _(dot_plot):
    s = pd.Series({"credit_card": 0.74, "boleto": 0.19, "voucher": 0.04, "debit_card": 0.03})
    ax = _get_ax(unchanged(dot_plot, s.sample(frac=1, random_state=1)), "dot_plot(...)")
    pts = points_of(ax)
    expect(len(pts) == len(s), f"Expected one dot per category ({len(s)}), found {len(pts)}. "
           "Use ax.scatter (or ax.plot with a marker and no line).")
    expect([lab for lab, _ in pts] == list(s.index),
           f"Top to bottom, the dots should read {list(s.index)} (largest on top). Got {[lab for lab, _ in pts]}.")
    expect(np.allclose([x for _, x in pts], s.to_numpy()), "Each dot's x position should be its share.")
    expect(_has_text(ax.get_xlabel()), "Label the x axis.")
    _close(ax)
    real = _real()["pay_shares"]
    ax = _get_ax(dot_plot(real.copy()), "dot_plot(...)")
    expect([lab for lab, _ in points_of(ax)] == list(real.sort_values(ascending=False).index),
           "Real data: the categories aren't in the right order.")
    _close(ax)
    return ("A dot plot shows the same numbers as a pie, but as positions on a common scale: the most accurately "
            "read encoding there is.")


@S.exercise("mosaic", "mosaic(counts, ax=None)", challenge=True, hints=[
    "Row shares P(row) = row totals / grand total give the column widths; within a row, "
    "P(col | row) = count / row total gives the heights.",
    "Loop over rows keeping an x offset; inside, loop over columns keeping a y offset. Draw each tile with "
    "`ax.add_patch(Rectangle((x, y), width, height, facecolor=...))`, then advance the offsets.",
    "`add_patch` doesn't rescale the axes: set `ax.set_xlim(0, 1)` and `ax.set_ylim(0, 1)` yourself. Put the "
    "row labels at the tile centers with `ax.set_xticks(centers, labels)`.",
])
@_tidy
def _(mosaic):
    ct = toy_crosstab()
    ax = _get_ax(unchanged(mosaic, ct), "mosaic(...)")
    rects = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_width() * p.get_height() > 1e-12]
    joint = (ct / ct.to_numpy().sum()).to_numpy().ravel()
    joint = np.sort(joint[joint > 0])
    expect(len(rects) == len(joint), f"Expected {len(joint)} tiles (one per non-zero cell), found {len(rects)}.")
    areas = np.sort([p.get_width() * p.get_height() for p in rects])
    expect(np.allclose(areas, joint, atol=1e-6),
           "Tile areas should equal the joint shares count / grand total. Check widths (row shares) and heights "
           "(shares within the row).")
    xs = sorted({round(p.get_x(), 6) for p in rects})
    widths = sorted(round(next(p.get_width() for p in rects if round(p.get_x(), 6) == x), 6) for x in xs)
    marg = np.sort(ct.sum(axis=1).to_numpy() / ct.to_numpy().sum())
    expect(np.allclose(widths, marg, atol=1e-6), f"Column widths should be the row shares {np.round(marg, 3).tolist()}.")
    for x in xs:
        h = sum(p.get_height() for p in rects if round(p.get_x(), 6) == x)
        expect(np.isclose(h, 1), "Within each column, the tiles' heights should add up to 1.")
    _close(ax)
    real = _real()["pay_by_band"]
    ax = _get_ax(mosaic(real.copy()), "mosaic(...)")
    area = sum(p.get_width() * p.get_height() for p in ax.patches if isinstance(p, Rectangle))
    expect(np.isclose(area, 1, atol=1e-6), "Real data: the tiles should fill the unit square.")
    _close(ax)
    return ("A mosaic shows three things at once: group sizes (widths), the mix within each group (heights) and "
            "joint shares (areas). If the two variables were independent, every column would split at the same heights.")


# ---- 4. Color -------------------------------------------------------------------------------


@S.exercise("corr-heatmap", "corr_heatmap(corr, ax=None)", hints=[
    "`im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)` (or `pcolormesh`). Fixing vmin/vmax puts 0 at the "
    "neutral middle color, and makes every heatmap of correlations comparable.",
    "Ticks: `ax.set_xticks(range(n), corr.columns, rotation=45, ha='right')`, and the same for y.",
    "`ax.figure.colorbar(im, ax=ax, label=...)` adds the legend for colors.",
])
@_tidy
def _(corr_heatmap):
    c = toy_corr()
    ax = _get_ax(unchanged(corr_heatmap, c), "corr_heatmap(...)")
    m, arr = mappable_of(ax)
    expect(m is not None, "Draw the matrix with ax.imshow or ax.pcolormesh.")
    expect(arr.size == c.size and np.allclose(np.sort(arr.ravel()), np.sort(c.to_numpy().ravel())),
           "The colored cells should be the correlation values.")
    expect(_cmap_name(m) in DIVERGING,
           f"`{_cmap_name(m)}` isn't a diverging colormap. Correlations have a meaningful middle (0) and two "
           "directions: use e.g. 'RdBu_r' or 'coolwarm'.")
    expect(np.isclose(m.norm.vmin, -1) and np.isclose(m.norm.vmax, 1),
           f"Fix the color range to [-1, 1] (got [{m.norm.vmin:.2f}, {m.norm.vmax:.2f}]), so 0 sits at the neutral color.")
    expect(len(ax.figure.axes) >= 2, "Add a colorbar: without it, colors can't be read as numbers.")
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    expect(labels == list(c.columns), f"Label the x ticks with the variable names {list(c.columns)}.")
    _close(ax)
    ax = _get_ax(corr_heatmap(_real()["corr"].copy()), "corr_heatmap(...)")
    _close(ax)
    return ("A diverging map with a fixed, symmetric range makes +0.4 and −0.4 equally strong but clearly "
            "different, and keeps an artificial 'max' from turning a weak correlation into a dark cell.")


@S.exercise("deviation-bars", "deviation_bars(order_table, min_orders=1000, ax=None)", challenge=True, hints=[
    "national = the late share over ALL delivered orders (`order_table['is_late'].mean()`). Each bar is a "
    "state's late share minus national.",
    "Keep states with at least min_orders delivered orders, sort descending, and color by sign: "
    "`colors = np.where(dev > 0, '#c0392b', '#2471a3')`, then `ax.bar(dev.index, dev.to_numpy(), color=colors)`.",
    "`ax.axhline(0, color='black', lw=0.8)` draws the reference line the bars grow from.",
])
@_tidy
def _(deviation_bars):
    toy = toy_late_orders()
    ax = _get_ax(unchanged(lambda d: deviation_bars(d, 3), toy), "deviation_bars(...)")
    got = bars_of(ax)
    want = _ref_deviation(toy, 3)
    expect([lab for lab, _ in got] == list(want.index),
           f"Expected bars {list(want.index)} sorted by deviation (highest first), got {[lab for lab, _ in got]}.")
    vals = np.array([v for _, v in got])
    if np.abs(vals).max() > 1.0001:
        vals = vals / 100
    expect(np.allclose(vals, want.to_numpy()),
           f"Bars should be share − national share, with national computed over every delivered order "
           f"(including small states). Expected {np.round(want.to_numpy(), 3).tolist()}.")
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    pos = {tuple(np.round(p.get_facecolor(), 3)) for p in rects if p.get_height() > 0}
    neg = {tuple(np.round(p.get_facecolor(), 3)) for p in rects if p.get_height() < 0}
    expect(len(pos) == 1 and len(neg) == 1 and pos != neg,
           "Color by sign: one color for above the national share, another for below.")
    _close(ax)
    real = _real()["deviation"]
    ax = _get_ax(deviation_bars(_real()["order_table"].copy()), "deviation_bars(...)")
    expect([lab for lab, _ in bars_of(ax)] == list(real.index), "Real data: the states or their order don't match.")
    _close(ax)
    return ("Plotting the difference from a reference makes 'better or worse than usual' the visual message: "
            "the zero line is the national average, and two colors carry the direction.")


@S.exercise("luminance-monotone", "is_monotone_luminance(cmap_name)", challenge=True, hints=[
    "`matplotlib.colormaps[name](np.linspace(0, 1, 256))` returns a 256×4 array of RGBA values in [0, 1]. "
    "Keep the first 3 columns.",
    "Linearize: c / 12.92 where c ≤ 0.04045, else ((c + 0.055) / 1.055) ** 2.4. Then "
    "Y = 0.2126 R + 0.7152 G + 0.0722 B (a matrix product with that vector).",
    "Monotone means all the consecutive differences `np.diff(Y)` are ≥ 0, or all are ≤ 0 (reversed maps count too).",
])
@_tidy
def _(is_monotone_luminance):
    names = ["viridis", "cividis", "Greys", "Greys_r", "Blues", "magma", "jet", "rainbow", "coolwarm", "turbo", "hsv"]
    for n in names:
        got = is_monotone_luminance(n)
        expect(isinstance(got, (bool, np.bool_)), f"Return True/False, got {type(got).__name__}.")
        expect(bool(got) == _ref_monotone(n),
               f"{n}: expected {_ref_monotone(n)}. " + ("Reversed maps ('_r') decrease monotonically: that counts."
                                                        if n.endswith("_r") else "Check the linearization step."))
    return ("Lightness is what the eye reads as 'more'. jet and rainbow go dark → bright → dark, inventing bands "
            "that aren't in the data, and they collapse in grayscale. viridis and cividis were designed not to.")


# ---- 5. Labels and export -------------------------------------------------------------------


@S.exercise("export-size", "save_png(fig, path, width_px)", challenge=True, hints=[
    "Pixels = inches × dots per inch. The figure's width in inches is `fig.get_figwidth()`.",
    "So `dpi = width_px / fig.get_figwidth()`, then `fig.savefig(path, dpi=dpi)`.",
    "Don't pass `bbox_inches='tight'`: it crops the margins, so the saved size no longer equals figsize × dpi.",
])
@_tidy
def _(save_png):
    import matplotlib.image as mpimg
    with tempfile.TemporaryDirectory() as d:
        for size, w in [((8, 4), 1200), ((6, 3), 1000), ((5, 5), 640)]:
            fig, ax = plt.subplots(figsize=size)
            ax.plot([0, 1], [0, 1])
            path = os.path.join(d, f"chart_{w}.png")
            save_png(fig, path, w)
            expect(os.path.exists(path), "save_png should write the file at `path`.")
            expect(tuple(fig.get_size_inches()) == size, "Don't change the figure's size: choose the dpi instead.")
            img = mpimg.imread(path)
            plt.close(fig)
            expect(abs(img.shape[1] - w) <= 1,
                   f"A {size[0]}×{size[1]} in figure saved for {w} px came out {img.shape[1]} px wide. "
                   "dpi = width_px / width_in, and no bbox_inches='tight'.")
    return ("figsize is in inches and dpi converts inches to pixels. Fix figsize for the layout (font sizes are in "
            "points, 1/72 in), then pick dpi for the resolution you need.")


# ---- 6. Mini-project ----------------------------------------------------------------------


@S.exercise("purchase-heatmap", "purchase_heatmap(timestamps, ax=None)", hints=[
    "`pd.crosstab(ts.dt.dayofweek, ts.dt.hour)` counts purchases per (day, hour). `.reindex(index=range(7), "
    "columns=range(24), fill_value=0)` guarantees the full 7×24 grid even when some cells are empty.",
    "`ax.imshow(grid, aspect='auto', cmap='viridis')`: a SEQUENTIAL map, because counts only go up.",
    "Days 0..6 are Monday..Sunday: `ax.set_yticks(range(7), ['Mon', 'Tue', ...])`. Add axis labels and a colorbar.",
])
@_tidy
def _(purchase_heatmap):
    ts = toy_timestamps()
    ax = _get_ax(unchanged(purchase_heatmap, ts), "purchase_heatmap(...)")
    m, arr = mappable_of(ax)
    expect(m is not None, "Draw the grid with ax.imshow or ax.pcolormesh.")
    want = _ref_heat(ts)
    expect(arr.size == want.size, f"Expected a full 7×24 grid ({want.size} cells), got {arr.size} cells. "
           "Reindex so that empty days and hours appear as 0.")
    expect(np.array_equal(arr.reshape(7, 24), want),
           "The counts per (day, hour) don't match. Rows should be days Monday..Sunday (dt.dayofweek 0..6), "
           "columns hours 0..23.")
    expect(_ref_monotone(_cmap_name(m)) and _cmap_name(m) not in DIVERGING,
           f"`{_cmap_name(m)}` isn't a sequential colormap with steadily changing lightness. Try 'viridis', "
           "'cividis' or 'Blues'.")
    days = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    expect(len(days) == 7 and days[0].lower().startswith("mon"),
           "Label the 7 rows with day names, starting with Monday.")
    expect(_has_text(ax.get_xlabel()), "Label the x axis (hour of purchase).")
    expect(len(ax.figure.axes) >= 2, "Add a colorbar so colors can be read as counts.")
    _close(ax)
    ax = _get_ax(purchase_heatmap(_real()["timestamps"].copy()), "purchase_heatmap(...)")
    _, arr = mappable_of(ax)
    expect(np.array_equal(arr.reshape(7, 24), _real()["heat"]), "Real data: the grid doesn't match.")
    _close(ax)
    return "Two time dimensions, one sequential color scale: the weekly rhythm of a marketplace in one picture."
