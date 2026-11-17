"""Grading for Module I · 02 — Filter, group, derive (masks, groupby, dates, vectorization, defensive functions).

⚠️ SPOILERS: the reference computations used for grading live in this file.
Try the exercises first; read this afterwards if you want to compare approaches.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ..core import Session, as_int, expect, expect_close, is_df, raises, unchanged
from ..data import load_olist

S = Session("m01/02", "Filter, group, derive")
check, hint, predict, progress = S.check, S.hint, S.predict, S.progress

# ===========================================================================
# Toy data: small enough to verify every answer by hand
# ===========================================================================


def toy_orders() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": ["o1", "o2", "o3", "o4", "o5", "o6"],
        "customer_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "order_status": ["delivered", "shipped", "delivered", "canceled", "delivered", "delivered"],
        "order_purchase_timestamp": pd.to_datetime([
            "2017-11-03", "2017-11-24", "2017-11-24", "2017-12-01", "2018-01-15", "2018-01-20",
        ]),
    })


def toy_items() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id":      ["o1", "o1", "o2", "o3", "o3", "o3", "o5", "o6", "o7"],
        "order_item_id": [1, 2, 1, 1, 2, 3, 1, 1, 1],
        "product_id":    ["pA", "pB", "pA", "pC", "pC", "pD", "pE", "pB", "pA"],
        "price":         [50.0, 120.0, 50.0, 200.0, 200.0, 15.0, 80.0, 120.0, 50.0],
        "freight_value": [10.0, 15.0, 8.0, 20.0, 20.0, 5.0, 12.0, 15.0, 9.0],
    })


def toy_customers() -> pd.DataFrame:
    """Listed in a different order than toy_orders (as real tables usually are)."""
    return pd.DataFrame({
        "customer_id": ["c4", "c1", "c6", "c3", "c2", "c5"],
        "customer_state": ["SP", "SP", "MG", "RJ", "RJ", "SP"],
    })


def toy_products() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": ["pA", "pB", "pC", "pD", "pE"],
        "product_category_name": ["cama_mesa_banho", "beleza_saude", "informatica_acessorios", np.nan, "pc_gamer"],
    })


def toy_categories() -> pd.DataFrame:
    return pd.DataFrame({
        "product_category_name": ["cama_mesa_banho", "beleza_saude", "informatica_acessorios"],
        "product_category_name_english": ["bed_bath_table", "health_beauty", "computers_accessories"],
    })


def toy_analysis() -> pd.DataFrame:
    """Delivered item rows with a clean `category` column (the `analysis` table, toy size)."""
    return _ref_delivered_items(toy_orders(), toy_items(), toy_products(), toy_categories())


def toy_deliveries() -> pd.DataFrame:
    """Orders with purchase and delivery timestamps, covering every structural case."""
    return pd.DataFrame({
        "order_id": ["d1", "d2", "d3", "d4", "d5"],
        "order_status": ["delivered", "delivered", "canceled", "shipped", "delivered"],
        "order_purchase_timestamp": pd.to_datetime([
            "2018-03-01 10:00", "2018-03-02 08:00", "2018-03-03 09:00", "2018-03-04 12:00", "2018-03-05 18:00"]),
        "order_delivered_customer_date": pd.to_datetime([
            "2018-03-04 22:00", None, "2018-03-10 09:00", None, "2018-03-06 06:00"]),
    }, index=[10, 11, 12, 13, 14])


# ===========================================================================
# Reference implementations (used for grading)
# ===========================================================================

BAND_LABELS = ["<50", "50-100", "100-200", "200+"]


def _ref_label(products: pd.DataFrame, categories: pd.DataFrame) -> pd.Series:
    m = products[["product_category_name"]].merge(
        categories, on="product_category_name", how="left", validate="many_to_one")
    lab = m["product_category_name_english"].fillna(m["product_category_name"]).fillna("UNKNOWN")
    lab.index = products.index
    return lab


def _ref_delivered_items(orders, items, products, categories) -> pd.DataFrame:
    delivered = orders.loc[orders["order_status"] == "delivered", ["order_id", "order_purchase_timestamp"]]
    df = items.merge(delivered, on="order_id", how="inner", validate="many_to_one")
    prod = products[["product_id"]].assign(category=_ref_label(products, categories).to_numpy())
    return df.merge(prod, on="product_id", how="left", validate="many_to_one")


def _ref_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("category")
            .agg(rows=("price", "size"), distinct_orders=("order_id", "nunique"),
                 price_sum=("price", "sum"), price_median=("price", "median"))
            .sort_values("price_sum", ascending=False))


def _ref_delivery_days(orders: pd.DataFrame) -> pd.Series:
    days = (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.total_seconds() / 86400
    return days.where(orders["order_status"].eq("delivered"))


@lru_cache(maxsize=1)
def _real() -> dict:
    """Reference answers on the real Olist data (computed once)."""
    orders, items = load_olist("orders"), load_olist("items")
    products, categories = load_olist("products"), load_olist("categories")
    customers = load_olist("customers")
    r: dict = {}
    d, a = orders["order_delivered_customer_date"], orders["order_approved_at"]

    r["n_delivered_2017"] = int((orders["order_status"].eq("delivered")
                                 & orders["order_purchase_timestamp"].dt.year.eq(2017) & d.notna()).sum())
    r["n_delivered_before_approved"] = int((d < a).sum())
    sp_ids = customers.loc[customers["customer_state"].eq("SP"), "customer_id"]
    r["delivered_sp_ids"] = set(orders.loc[orders["order_status"].eq("delivered")
                                           & orders["customer_id"].isin(sp_ids), "order_id"])

    an = _ref_delivered_items(orders, items, products, categories)
    r["analysis"] = an
    r["summary"] = _ref_category_summary(an)
    r["median_item_price"] = float(an["price"].median())
    r["median_order_value"] = float(an.groupby("order_id")["price"].sum().median())
    share = an["price"] / an.groupby("order_id")["price"].transform("sum")
    per_order = pd.DataFrame({"order_id": an["order_id"], "share": share}).groupby("order_id")["share"].agg(["max", "size"])
    r["share_dominated"] = float((per_order.loc[per_order["size"] >= 2, "max"] > 0.5).mean())
    nov = an[an["order_purchase_timestamp"].dt.to_period("M") == pd.Period("2017-11", "M")]
    r["top_2017_11"] = nov.groupby("category")["price"].sum().idxmax()
    r["bands"] = pd.cut(an["price"], [0, 50, 100, 200, np.inf], right=False, labels=BAND_LABELS)

    oc = orders.merge(customers[["customer_id", "customer_state"]], on="customer_id", validate="one_to_one")
    r["status_share"] = pd.crosstab(oc["customer_state"], oc["order_status"], normalize="index")

    r["delivery_days"] = _ref_delivery_days(orders)
    oc["delivery_days"] = _ref_delivery_days(oc)
    dd = oc.dropna(subset=["delivery_days"]).copy()
    st = dd.groupby("customer_state")["delivery_days"].agg(["size", "median"])
    r["slowest_state_1000"] = st[st["size"] >= 1000]["median"].idxmax()
    r["slowest_state_any"] = st["median"].idxmax()
    dd["state_median"] = dd.groupby("customer_state")["delivery_days"].transform("median")
    r["share_slow"] = float((dd["delivery_days"] > 2 * dd["state_median"]).mean())
    return r


# ===========================================================================
# Predictions
# ===========================================================================

S.prediction("mask-precedence", "c",
    "`&` binds tighter than `>` and `==`, so Python reads the condition as "
    "`year > (2017 & orders.order_status) == \"delivered\"`. The first thing it computes is "
    "`2017 & orders.order_status`, a bitwise AND between an integer and a column of strings, which raises a "
    "TypeError. (With numeric columns it can get past that step and then fail on the chained comparison "
    "`a > b == c`, which means `(a > b) and (b == c)`; `and` on a Series raises 'truth value is ambiguous'.) "
    "Fix: parenthesize each condition: "
    "(orders.order_purchase_timestamp.dt.year > 2017) & (orders.order_status == \"delivered\").",
    compare=lambda v, a: str(v).strip().lower().strip(")") == a)
S.prediction("nat-complement", False,
    "Any comparison involving NaT (or NaN) is False. So for a row with a missing date, `d < a` is False and "
    "`~(d < a)` is True, but `d >= a` is also False. The two masks disagree on exactly the rows where either "
    "date is missing: 'not before' is not the same as 'on or after' when values can be missing.",
    compare=lambda v, a: str(v).strip().lower() in {"false", "no", "0"})
S.prediction("groupby-drops-nan", 4,
    "groupby drops NaN keys by default, so pD (category missing) silently vanishes: 4 groups, and the group sizes "
    "add up to 4, not 5. Use dropna=False, or better, label missing values explicitly before grouping.")
S.prediction("size-count-nunique", (3, 2, 1),
    "size counts rows (3, including the NaN), count counts non-missing values (2), nunique counts distinct "
    "non-missing values (1, since both are 10).",
    compare=lambda v, a: tuple(v) == a if isinstance(v, (tuple, list)) else False)
S.prediction("timedelta-days", 0,
    "A Timedelta of 12 hours has a `.days` *component* of 0: `.days` is the whole-day part, like the integer part "
    "of a division, not the duration expressed in days. For fractional days use `.total_seconds() / 86400`, "
    "which gives 0.5 here.")

# ===========================================================================
# Exercises
# ===========================================================================

# ---- 1. Masks ------------------------------------------------------------------


@S.exercise("delivered-2017", "delivered in 2017", hints=[
    "Build three boolean Series and combine them with &, each one in parentheses.",
    "Year of a datetime column: `orders['order_purchase_timestamp'].dt.year`. Non-missing: `.notna()`.",
    "Summing a boolean mask counts the True values.",
])
def _(n_delivered_2017):
    n = as_int(n_delivered_2017, "n_delivered_2017")
    expect(n == _real()["n_delivered_2017"],
           "Not quite. Conditions: status == 'delivered' AND purchase year == 2017 AND delivered date not missing.")
    return "Masks are just boolean Series aligned by index: build them, name them, combine them."


@S.exercise("delivered-sp", "delivered orders of São Paulo customers", hints=[
    "The state lives in `customers`, the status in `orders`. A mask must be built from the table you filter.",
    "Option A: the customer_ids of SP customers, then `orders['customer_id'].isin(those_ids)`.",
    "Option B: merge orders with customers (validate!), filter the merged table, then keep only orders' columns.",
])
def _(delivered_sp):
    is_df(delivered_sp, "delivered_sp")
    cols = list(load_olist("orders").columns)
    expect(list(delivered_sp.columns) == cols,
           "delivered_sp should have exactly the columns of `orders` (drop anything a merge added).")
    got = set(delivered_sp["order_id"])
    want = _real()["delivered_sp_ids"]
    expect(len(delivered_sp) == len(got), "Some orders appear twice. Did a join multiply rows?")
    expect(got == want,
           f"Expected {len(want):,} orders, got {len(got):,}"
           + (" (and some of them aren't delivered SP orders)." if got - want else ".")
           + " If your mask came from a different table than the one you filtered, it was aligned by index label, "
             "not by order.")
    return ("Build a mask on the table you filter, or carry the values you need into that table first. isin() "
            "with a list of keys is the simplest cross-table filter there is.")


@S.exercise("delivered-before-approved", "delivered before approved?", challenge=True, hints=[
    "Compare two datetime columns directly: `orders['order_delivered_customer_date'] < orders['order_approved_at']`.",
    "What happens when one side is NaT? (Comparisons with NaT are always False: check that's what you want.)",
])
def _(n_delivered_before_approved):
    n = as_int(n_delivered_before_approved, "n_delivered_before_approved")
    expect(n == _real()["n_delivered_before_approved"],
           "Count doesn't match. Compare delivered-to-customer vs approved timestamps.")
    return ("Impossible orderings of events are classic data-quality flags. Decide (and document) whether to "
            "exclude, fix, or keep them, but never silently.")


# ---- 2. Grouping ------------------------------------------------------------------


@S.exercise("category-summary", "category_summary(df)", hints=[
    "Named aggregation: `df.groupby('category').agg(rows=('price','size'), distinct_orders=('order_id','nunique'), ...)`.",
    "Then `.sort_values('price_sum', ascending=False)`.",
])
def _(category_summary):
    def compare(got, want, where):
        is_df(got, "category_summary(...)")
        cols = ["rows", "distinct_orders", "price_sum", "price_median"]
        expect(list(got.columns) == cols, f"{where}: columns should be {cols} in that order, got {list(got.columns)}.")
        expect(list(got.index) == list(want.index),
               f"{where}: expected categories as the index, sorted by price_sum descending.")
        expect(np.allclose(got.to_numpy(dtype=float), want.to_numpy(dtype=float)), f"{where}: values don't match.")
    ta = toy_analysis()
    compare(unchanged(category_summary, ta), _ref_category_summary(ta), "Toy check")
    compare(category_summary(_real()["analysis"].copy()), _real()["summary"], "Real data")
    return "Look at computers_accessories in the toy table: 2 rows but 1 order. rows ≠ orders, again."


@S.exercise("two-medians", "two medians", hints=[
    "Item level: the median of the `price` column.",
    "Order level: first `groupby('order_id')['price'].sum()`, *then* take the median.",
])
def _(median_item_price, median_order_value):
    r = _real()
    expect_close(median_item_price, r["median_item_price"], "median_item_price")
    expect_close(median_order_value, r["median_order_value"], "median_order_value")
    return ("Same data, different unit of observation, different answer. Always say *median of what*: "
            "'median price per item row' vs 'median order value'.")


@S.exercise("share-of-order", "share_of_order with transform", challenge=True, hints=[
    "`groupby(...).transform('sum')` returns a Series with the *same length as df*, aligned row by row.",
    "share_of_order = price / order total. Return `df.assign(share_of_order=...)` so the input isn't modified.",
    "For share_dominated: per order, take max(share) and number of rows; keep orders with ≥2 rows; mean of (max > 0.5).",
])
def _(add_share_of_order, share_dominated):
    ta = toy_analysis()
    got = unchanged(add_share_of_order, ta)
    is_df(got, "add_share_of_order(...)")
    expect("share_of_order" in got.columns, "Add a column called `share_of_order`.")
    expect(len(got) == len(ta), "The result should have one row per input row (transform, not agg).")
    want = ta["price"] / ta.groupby("order_id")["price"].transform("sum")
    expect(np.allclose(got["share_of_order"].to_numpy(dtype=float), want.to_numpy()), "Toy check: shares don't match.")
    expect_close(share_dominated, _real()["share_dominated"], "share_dominated", tol=1e-4)
    return ("agg collapses groups; transform broadcasts the group result back to every row. You need both. "
            "Curious why it's so low? ~2/3 of multi-item orders are the *same product* repeated: check with nunique.")


@S.exercise("top-category-nov-2017", "top category, Nov 2017", challenge=True, hints=[
    "Filter delivered items purchased in November 2017: `.dt.to_period('M') == '2017-11'`.",
    "groupby category, sum price, then `.idxmax()` gives the label of the largest.",
])
def _(top_category_2017_11):
    expect(isinstance(top_category_2017_11, str), "Give the category label as a string.")
    expect(top_category_2017_11 == _real()["top_2017_11"],
           "Not the top one. Use the `category` labels and sum price over delivered items purchased in Nov 2017.")
    return "Black Friday month. A next question would be: which categories *grew* most vs October?"


@S.exercise("status-share", "status mix per state with crosstab", hints=[
    "Bring the state into orders first: merge with customers on customer_id (validate='one_to_one').",
    "`pd.crosstab(rows, columns, normalize='index')` divides each row by its total, so each row sums to 1.",
])
def _(status_share):
    is_df(status_share, "status_share")
    want = _real()["status_share"]
    expect(np.allclose(status_share.sum(axis=1).to_numpy(dtype=float), 1.0),
           "Each row (state) should sum to 1. Did you use normalize='index'? ('columns' or 'all' divide differently.)")
    expect(sorted(status_share.index) == sorted(want.index) and sorted(status_share.columns) == sorted(want.columns),
           "Rows should be the 27 states and columns the order statuses.")
    got = status_share.loc[want.index, want.columns]
    expect(np.allclose(got.to_numpy(dtype=float), want.to_numpy()), "The shares don't match.")
    return ("normalize= chooses the denominator: 'index' answers 'within each state, what share is X?'. Always say "
            "which denominator a share uses.")


# ---- 3. Dates ---------------------------------------------------------------------


@S.exercise("delivery-days", "add_delivery_days(orders)", hints=[
    "Subtracting two datetime columns gives Timedeltas; `.dt.total_seconds() / 86400` turns them into fractional days.",
    "Population: only orders with status 'delivered' get a duration. Use `.where(condition)`, which keeps values "
    "where the condition is True and puts NaN elsewhere.",
    "Return `orders.assign(delivery_days=...)` so the input isn't modified and the index is kept.",
])
def _(add_delivery_days):
    toy = toy_deliveries()
    got = unchanged(add_delivery_days, toy)
    is_df(got, "add_delivery_days(...)")
    expect("delivery_days" in got.columns, "Add a column called `delivery_days`.")
    expect(list(got.index) == list(toy.index), "Keep the input's index (the toy table is indexed 10..14).")
    g = got["delivery_days"].to_numpy(dtype=float)
    expect(not np.isnan(g[0]) and abs(g[0] - 3.5) < 1e-9 and abs(g[4] - 0.5) < 1e-9,
           f"Toy check: d1 took 3.5 days and d5 half a day, got {g[0]:.4g} and {g[4]:.4g}. "
           "`.dt.days` truncates to whole days; use total_seconds().")
    expect(np.isnan(g[1]) and np.isnan(g[3]), "Toy check: orders without a delivery date must stay NaN, not 0.")
    expect(np.isnan(g[2]),
           "Toy check: d3 is canceled. Even though it has a delivery date, a canceled order has no delivery duration.")
    real = add_delivery_days(load_olist("orders"))["delivery_days"]
    ref = _real()["delivery_days"]
    expect(int(real.notna().sum()) == int(ref.notna().sum()), "Real data: wrong number of orders with a duration.")
    expect_close(real.median(), ref.median(), "median delivery_days", tol=1e-6)
    return ("A missing duration means 'didn't happen' or 'not recorded'. Filling it with 0 would invent thousands "
            "of instant deliveries and drag every average down.")


@S.exercise("slow-vs-state", "slower than twice the state's median", hints=[
    "You need each order's own duration next to its state's median: that's transform, not agg.",
    "`dd.groupby('customer_state')['delivery_days'].transform('median')` gives one value per row.",
    "share_slow = mean of (delivery_days > 2 * state_median), over delivered orders with a duration.",
])
def _(share_slow):
    expect_close(share_slow, _real()["share_slow"], "share_slow", tol=1e-4)
    return ("Comparing each row with its own group's benchmark is the typical transform job: it adjusts for the "
            "fact that 'slow' means something different in a remote state.")


# ---- 4. Vectorization ---------------------------------------------------------------

_EDGE = pd.Series([0.85, 49.99, 50.0, 99.99, 100.0, 199.99, 200.0, 6735.0])
_EDGE_WANT = ["<50", "<50", "50-100", "50-100", "100-200", "100-200", "200+", "200+"]


@S.exercise("price-band-select", "price_band_select(prices)", hints=[
    "np.select(condlist, choicelist, default=...) picks the first condition that is True for each element.",
    "Conditions in order: price < 50, price < 100, price < 200; default '200+'.",
])
def _(price_band_select):
    got = list(np.asarray(price_band_select(_EDGE.copy()), dtype=object))
    expect(got == _EDGE_WANT, f"Edge cases: expected {_EDGE_WANT}, got {got}. Check < vs <= at 50, 100, 200.")
    real = list(np.asarray(price_band_select(_real()["analysis"]["price"].copy()), dtype=object))
    expect(real == list(_real()["bands"].astype(str)), "Edge cases pass but real data doesn't.")
    return "np.select generalizes np.where to many branches: one vectorized pass instead of a Python loop."


@S.exercise("price-band-cut", "price_band_cut(prices)", hints=[
    "pd.cut(prices, bins=[0, 50, 100, 200, np.inf], right=False, labels=[...]).",
    "right=False makes intervals [a, b): 50 goes to '50-100', not '<50'.",
])
def _(price_band_cut):
    got = price_band_cut(_EDGE.copy())
    expect(isinstance(getattr(got, "dtype", None), pd.CategoricalDtype),
           "pd.cut returns a Categorical: return that directly (don't convert to str).")
    expect(list(got.cat.categories if hasattr(got, "cat") else got.categories) == BAND_LABELS,
           f"Categories should be {BAND_LABELS} in that order.")
    expect(list(np.asarray(got, dtype=object)) == _EDGE_WANT, f"Edge cases: expected {_EDGE_WANT}, got {list(got)}.")
    return ("An ordered Categorical keeps '<50' < '50-100' < ... when sorting or plotting; plain strings would sort "
            "'100-200' before '50-100'.")


# ---- 5. Defensive functions ---------------------------------------------------------------


def _top_cat_core(top_categories):
    ta = toy_analysis()
    got = unchanged(lambda d: top_categories(d, 2), ta)
    expect(isinstance(got, pd.Series), f"Return a Series (category → price_sum), got {type(got).__name__}.")
    expect(list(got.index) == ["computers_accessories", "health_beauty"] and np.allclose(got.to_numpy(dtype=float), [400, 240]),
           f"top_categories(toy, 2) should give computers_accessories=400, health_beauty=240. Got {got.to_dict()}.")
    expect(len(top_categories(ta)) == 5, "With the default n=5, the toy table should return all 5 categories.")
    expect(len(top_categories(ta, 10)) == 5, "Asking for more categories than exist should not fail; just return all.")
    for bad, exc, why in [(2.5, TypeError, "n=2.5"), ("3", TypeError, "n='3'"),
                          (0, ValueError, "n=0"), (-1, ValueError, "n=-1")]:
        res = raises(top_categories, exc, ta, bad)
        expect(res is True, f"{why} should raise {exc.__name__}" + (f", but raised {res}." if res else ", but nothing was raised."))
    res = raises(top_categories, KeyError, ta.drop(columns="price"), 2)
    expect(res is True, "A table without a `price` column should raise KeyError with a clear message"
           + (f" (raised {res})." if res else " (nothing raised)."))


@S.exercise("top-categories", "top_categories(df, n=5)", hints=[
    "Validate first, compute later: isinstance(n, int) → else TypeError; n < 1 → ValueError; missing columns → KeyError.",
    "Compute with groupby('category')['price'].sum().sort_values(ascending=False).head(n).",
    "`required = {'category', 'price'}; missing = required - set(df.columns)` makes a nice error message.",
])
def _(top_categories):
    _top_cat_core(top_categories)
    return "Fail early, fail loudly: a clear error at the boundary beats a wrong number three steps later."


@S.exercise("bool-trap", "the bool trap", challenge=True, hints=[
    "In Python, `isinstance(True, int)` is True: bool is a subclass of int!",
    "Reject it explicitly: `if isinstance(n, bool) or not isinstance(n, int): raise TypeError(...)`.",
    "Should numpy integers (np.int64(3)) be accepted? `numbers.Integral` covers them.",
])
def _(top_categories):
    _top_cat_core(top_categories)
    res = raises(top_categories, TypeError, toy_analysis(), True)
    expect(res is True, "top_categories(df, True) should raise TypeError: True is not a sensible 'n'.")
    ok = raises(top_categories, Exception, toy_analysis(), np.int64(2))
    expect(ok is False, "np.int64(2) should be accepted (values from pandas are often numpy integers).")
    return "Type checks in Python have sharp edges: bool ⊂ int, and numpy ints are not `int`. numbers.Integral helps."
