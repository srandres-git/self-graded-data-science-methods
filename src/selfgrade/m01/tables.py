"""Grading for Module I · 01 — Tables you can trust (reading, keys, joins, invariants).

⚠️ SPOILERS: the reference computations used for grading live in this file.
Try the exercises first; read this afterwards if you want to compare approaches.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ..core import (Session, as_int, expect, expect_close, is_df, is_series, raises, unchanged)
from ..data import load_olist, olist_path

S = Session("m01/01", "Tables you can trust")
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


def toy_payments() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id":           ["o1", "o1", "o2", "o3", "o4", "o5", "o6"],
        "payment_sequential": [1, 2, 1, 1, 1, 1, 1],
        "payment_type": ["credit_card", "voucher", "boleto", "credit_card", "credit_card", "credit_card", "debit_card"],
        "payment_value":      [125.0, 70.0, 58.0, 460.0, 40.0, 92.0, 100.0],
    })


def toy_customers() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "customer_unique_id": ["u1", "u2", "u1", "u3", "u2", "u1"],
        "customer_state": ["SP", "RJ", "SP", "MG", "RJ", "SP"],
    })


def toy_products() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": ["pA", "pB", "pC", "pD", "pE"],
        "product_category_name": ["cama_mesa_banho", "beleza_saude", "informatica_acessorios", np.nan, "pc_gamer"],
    })


def toy_products_dirty() -> pd.DataFrame:
    """Same catalog, but pB appears twice (a duplicated key)."""
    p = toy_products()
    return pd.concat([p, p.iloc[[1]]], ignore_index=True)


def toy_categories() -> pd.DataFrame:
    return pd.DataFrame({
        "product_category_name": ["cama_mesa_banho", "beleza_saude", "informatica_acessorios"],
        "product_category_name_english": ["bed_bath_table", "health_beauty", "computers_accessories"],
    })


# ===========================================================================
# Reference implementations (used for grading)
# ===========================================================================


def _ref_zip(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.zfill(5)
    return out.where(s.notna(), pd.NA)


def _ref_label(products: pd.DataFrame, categories: pd.DataFrame) -> pd.Series:
    m = products[["product_category_name"]].merge(
        categories, on="product_category_name", how="left", validate="many_to_one")
    lab = m["product_category_name_english"].fillna(m["product_category_name"]).fillna("UNKNOWN")
    lab.index = products.index
    return lab


def _ref_order_totals(items, payments) -> pd.DataFrame:
    it = items.assign(_v=items["price"] + items["freight_value"]).groupby("order_id")["_v"].sum().rename("items_total")
    pa = payments.groupby("order_id")["payment_value"].sum().rename("paid_total")
    return pd.concat([it, pa], axis=1, join="outer").fillna(0.0).sort_index()


ORDER_TABLE_COLS = ["order_id", "order_status", "customer_unique_id", "customer_state",
                    "n_items", "items_value", "freight_value", "paid_value"]


def _ref_order_table(orders, items, payments, customers) -> pd.DataFrame:
    it = items.groupby("order_id").agg(n_items=("price", "size"), items_value=("price", "sum"),
                                       freight_value=("freight_value", "sum"))
    pa = payments.groupby("order_id").agg(paid_value=("payment_value", "sum"))
    out = (orders[["order_id", "customer_id", "order_status"]]
           .merge(customers[["customer_id", "customer_unique_id", "customer_state"]],
                  on="customer_id", how="left", validate="many_to_one")
           .merge(it, on="order_id", how="left", validate="one_to_one")
           .merge(pa, on="order_id", how="left", validate="one_to_one"))
    out["n_items"] = out["n_items"].fillna(0).astype(int)
    return out[ORDER_TABLE_COLS]


def _ref_geo_by_state(path, chunksize=200_000) -> pd.Series:
    total = pd.Series(dtype="int64")
    for chunk in pd.read_csv(path, usecols=["geolocation_state"], chunksize=chunksize):
        total = total.add(chunk["geolocation_state"].value_counts(), fill_value=0)
    return total.astype(int).sort_index()


@lru_cache(maxsize=1)
def _real() -> dict:
    """Reference answers on the real Olist data (computed once)."""
    orders, items = load_olist("orders"), load_olist("items")
    products, categories = load_olist("products"), load_olist("categories")
    reviews, payments = load_olist("reviews"), load_olist("payments")
    customers = load_olist("customers")
    r: dict = {}

    geo_zips = set(pd.read_csv(olist_path("geolocation"), usecols=["geolocation_zip_code_prefix"],
                               dtype="string")["geolocation_zip_code_prefix"])
    raw = customers["customer_zip_code_prefix"]
    r["zip_fixed"] = _ref_zip(raw)
    r["n_matched_as_text"] = int(raw.isin(geo_zips).sum())
    r["n_unmatched_after_fix"] = int((~r["zip_fixed"].isin(geo_zips)).sum())

    vc = reviews["order_id"].value_counts()
    r["n_dup_review_id"] = int(reviews["review_id"].duplicated().sum())
    r["n_orders_multi_review"] = int((vc > 1).sum())
    r["pair_is_key"] = bool(not reviews.duplicated(["review_id", "order_id"]).any())

    k = customers.groupby("customer_unique_id")["customer_id"].nunique()
    r["n_customer_ids"] = int(customers["customer_id"].nunique())
    r["n_people"] = int(customers["customer_unique_id"].nunique())
    r["n_people_multi"] = int((k > 1).sum())

    no_items = orders[~orders["order_id"].isin(items["order_id"])]
    r["n_orders_without_items"] = len(no_items)
    r["status_without_items"] = no_items["order_status"].value_counts().sort_index()

    ot = _ref_order_totals(items, payments)
    r["order_totals"] = ot
    r["share_mismatch"] = float(((ot["items_total"] - ot["paid_total"]).abs() > 1).mean())

    r["labels"] = _ref_label(products, categories)
    r["untranslated"] = sorted(set(products["product_category_name"].dropna())
                               - set(categories["product_category_name"]))

    t = _ref_order_table(orders, items, payments, customers)
    r["order_table"] = t
    st = t.groupby("customer_state").agg(n=("order_id", "size"), med=("paid_value", "median"))
    r["top_state_1000"] = st[st["n"] >= 1000]["med"].idxmax()
    r["top_state_any"] = st["med"].idxmax()
    return r


# ===========================================================================
# Predictions
# ===========================================================================

S.prediction("dates-as-text", "object",
    "CSV is plain text: it has no notion of 'date'. Unless you pass parse_dates=, pandas keeps the column as "
    "Python strings, which it reports as dtype `object`. Comparisons then work alphabetically, and `.dt` "
    "doesn't exist. Always check `df.dtypes` right after reading.",
    compare=lambda v, a: str(v).strip().lower() in {"object", "str", "string", "o"})
S.prediction("distinct-orders", 6,
    "items_toy has 9 rows but only 6 distinct order_id values: the unit of observation of `items` "
    "is an *item row* (order_id, order_item_id), not an order. Counting rows ≠ counting orders.")
S.prediction("left-join-rows", 9,
    "A LEFT join keeps every row of the left table. Each item matches at most one order (many-to-one), "
    "so the row count stays 9: o7 just gets NaN in the order columns.")
S.prediction("inner-join-rows", 8,
    "An INNER join drops rows without a match. o7 has no order, so 9 → 8. Silent row loss is why "
    "you should always compare row counts before and after a join.")
S.prediction("many-to-many-rows", 10,
    "Many-to-many! o1 has 2 items and 2 payments → 2×2 = 4 rows. Total: o1 4 + o2 1 + o3 3×1 + o5 1 + o6 1 = 10 "
    "(o7 has no payment and o4 has no items, so both disappear in the inner join). Any sum of `price` over this "
    "table double-counts o1. Fix: aggregate each table to the order level first, then join one-to-one.")

# ===========================================================================
# Exercises
# ===========================================================================

# ---- 1. Reading a CSV --------------------------------------------------------


@S.exercise("normalize-zip", "normalize_zip(s)", hints=[
    "Turn everything into text first: `s.astype('string')` works for both integer and text input.",
    "`.str.zfill(5)` left-pads with zeros up to 5 characters ('9790' → '09790') and leaves longer strings alone.",
    "Missing values must stay missing: `.str` methods propagate <NA>, but check that you didn't turn NaN into the text 'nan' "
    "(that happens with `.astype(str)`).",
])
def _(normalize_zip):
    cases = [
        (pd.Series(["9790", "01037", "14409"]), ["09790", "01037", "14409"], "text input"),
        (pd.Series([9790, 1037, 14409]), ["09790", "01037", "14409"], "integer input (as pandas reads it by default)"),
        (pd.Series(["9790", None, "14409"]), ["09790", None, "14409"], "a missing value"),
    ]
    for s, want, desc in cases:
        got = unchanged(normalize_zip, s)
        is_series(got, "normalize_zip(...)")
        expect(list(got.index) == list(s.index), f"{desc}: keep the input's index.")
        g = [None if pd.isna(x) else str(x) for x in got]
        expect(g == want, f"{desc}: expected {want}, got {g}.")
    s = pd.Series(["1", "22"], index=[10, 20])
    expect(list(normalize_zip(s).index) == [10, 20], "Keep the input's index (try a Series indexed 10, 20).")
    real = normalize_zip(load_olist("customers")["customer_zip_code_prefix"])
    expect((real.astype("string").to_numpy() == _real()["zip_fixed"].to_numpy()).all(),
           "Toy cases pass but the real customers column doesn't. Are all results 5 characters long?")
    return ("A key is only a key if both tables write it the same way. Somewhere upstream, a spreadsheet or a "
            "numeric column stripped the leading zeros from 24% of the customers' prefixes.")


@S.exercise("zip-coverage", "how many customers can be located?", hints=[
    "`a.isin(b)` returns a boolean Series: is each value of `a` among the values of `b`?",
    "n_matched_as_text uses the prefixes exactly as loaded; n_unmatched_after_fix uses your normalize_zip and counts the False values.",
    "Build the set of geolocation prefixes once: read only that column, as text: "
    "`pd.read_csv(path, usecols=[...], dtype='string')`.",
])
def _(n_matched_as_text, n_unmatched_after_fix):
    r = _real()
    a = as_int(n_matched_as_text, "n_matched_as_text")
    b = as_int(n_unmatched_after_fix, "n_unmatched_after_fix")
    expect(a == r["n_matched_as_text"],
           "n_matched_as_text is off. Compare the customer prefixes *as loaded* (text, unpadded) with the geolocation "
           "prefixes read as text.")
    expect(b == r["n_unmatched_after_fix"],
           "n_unmatched_after_fix is off. Normalize the customer prefixes, then count those NOT found in geolocation.")
    return ("Fixing the representation recovered ~24,000 customers. The few that are still unmatched are genuinely "
            "missing from the geolocation table: report them instead of dropping them silently.")


@S.exercise("geo-rows-by-state", "geo_rows_by_state(path) with chunks", challenge=True, hints=[
    "`pd.read_csv(path, usecols=['geolocation_state'], chunksize=200_000)` returns an *iterator* of DataFrames.",
    "Keep a running total: `total = total.add(chunk['geolocation_state'].value_counts(), fill_value=0)`.",
    "Start from an empty Series, and convert to int at the end (`.add` with fill_value produces floats).",
])
def _(geo_rows_by_state):
    got = geo_rows_by_state(olist_path("geolocation"))
    is_series(got, "geo_rows_by_state(...)")
    want = _ref_geo_by_state(olist_path("geolocation"))
    got = got.sort_index()
    expect(list(got.index) == list(want.index), f"Expected one entry per state ({len(want)}), got {len(got)}.")
    expect((got.astype(int).to_numpy() == want.to_numpy()).all(), "The counts per state don't match.")
    expect(int(got.sum()) == 1_000_163, "The counts should add up to the file's 1,000,163 rows.")
    return ("Chunking keeps memory bounded by the chunk size, not the file size. Combined with usecols= it lets you "
            "summarize files far bigger than your RAM.")


# ---- 2. Unit of observation and keys -------------------------------------------


@S.exercise("is-key", "is_key(df, cols)", hints=[
    "A set of columns is a key when (a) none of its values are missing and (b) no combination repeats.",
    "`df[cols].isna().any().any()` tells you about (a); `df.duplicated(subset=cols).any()` about (b).",
    "Accept both a single column name and a list: `if isinstance(cols, str): cols = [cols]`. Return a plain bool.",
])
def _(is_key):
    o, it = toy_orders(), toy_items()
    pair = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 1]})
    cases = [
        (o, "order_id", True, "toy orders, 'order_id'"),
        (it, "order_id", False, "toy items, 'order_id' (repeats once per item)"),
        (it, ["order_id", "order_item_id"], True, "toy items, ['order_id', 'order_item_id']"),
        (pd.DataFrame({"k": ["a", None, "b"]}), "k", False, "a column with a missing value and no duplicates"),
        (pd.DataFrame({"k": ["a", "b", "b"]}), ["k"], False, "a column with a duplicate"),
        (pair, ["a", "b"], True, "a pair of columns whose combination is unique"),
        (pair, ["a"], False, "column 'a' alone"),
    ]
    for df, cols, want, desc in cases:
        before = df.copy()
        got = is_key(df, cols)
        expect(isinstance(got, (bool, np.bool_)), f"is_key should return True/False, got {type(got).__name__} ({desc}).")
        expect(bool(got) == want, f"For {desc}: expected {want}, got {bool(got)}.")
        expect(before.equals(df), "is_key modified its input.")
    return "Key = no missing values + no repeated combinations. Check it before every join."


@S.exercise("review-duplicates", "duplicates in reviews", hints=[
    "`Series.duplicated()` marks the 2nd, 3rd, ... occurrence of a value. Summing it counts the *extra* rows.",
    "For orders with more than one review: `reviews['order_id'].value_counts()`, then count how many counts are > 1.",
])
def _(n_dup_review_id, n_orders_multi_review):
    r = _real()
    a = as_int(n_dup_review_id, "n_dup_review_id")
    b = as_int(n_orders_multi_review, "n_orders_multi_review")
    expect(a == r["n_dup_review_id"],
           "n_dup_review_id is off. Count the rows whose review_id already appeared earlier (`.duplicated().sum()`).")
    expect(b == r["n_orders_multi_review"],
           "n_orders_multi_review is off. You want the number of *order_ids* that appear more than once, "
           "not the number of extra rows.")
    return ("Neither review_id nor order_id is a key of reviews. One review can be attached to several orders, "
            "and one order can have several reviews: a many-to-many relationship hiding in a 'simple' table.")


@S.exercise("review-pair-key", "is (review_id, order_id) a key?", challenge=True, hints=[
    "Reuse your is_key on the real reviews table.",
])
def _(pair_is_key):
    expect(isinstance(pair_is_key, (bool, np.bool_)), "Give a True/False answer.")
    expect(bool(pair_is_key) == _real()["pair_is_key"], "Not what the data says. Run is_key(reviews, ['review_id','order_id']).")
    return ("So the unit of observation of `reviews` is a (review, order) link. Before joining reviews to orders "
            "you must decide which review represents each order: Notebook 04 does exactly that.")


@S.exercise("customers-vs-people", "customer_id vs customer_unique_id", hints=[
    "`nunique()` counts distinct values of a column.",
    "For people with several customer_ids: group by customer_unique_id, count distinct customer_id, keep counts > 1.",
])
def _(n_customer_ids, n_people, n_people_multi):
    r = _real()
    a, b, c = (as_int(n_customer_ids, "n_customer_ids"), as_int(n_people, "n_people"),
               as_int(n_people_multi, "n_people_multi"))
    expect(a == r["n_customer_ids"], "n_customer_ids is off: count distinct customer_id values.")
    expect(b == r["n_people"], "n_people is off: count distinct customer_unique_id values.")
    expect(c == r["n_people_multi"],
           "n_people_multi is off: count the customer_unique_id values linked to MORE than one customer_id.")
    return ("Each customer_id appears in exactly one order, so it's really an order-level alias. To count people, "
            "or repeat buyers, you need customer_unique_id. Same table, two units of observation.")


# ---- 3. Joins and cardinality ------------------------------------------------------


@S.exercise("anti-joins", "anti-joins with indicator=True", hints=[
    "`left.merge(right, on='order_id', how='left', indicator=True)` adds a `_merge` column: 'both' or 'left_only'.",
    "Keep the 'left_only' rows, then select only the original columns: `[items_toy.columns]` (this drops `_merge` "
    "and the columns that came from the other table).",
    "Merging with `orders_toy[['order_id']]` instead of the full table avoids bringing extra columns in the first place.",
])
def _(orphan_items, orders_without_items):
    it, o = toy_items(), toy_orders()
    is_df(orphan_items, "orphan_items")
    is_df(orders_without_items, "orders_without_items")
    expect(sorted(orphan_items["order_id"]) == ["o7"],
           f"orphan_items should contain the item rows whose order is missing from orders_toy. "
           f"Got order_ids {sorted(orphan_items['order_id'])}.")
    expect(sorted(orders_without_items["order_id"]) == ["o4"],
           f"orders_without_items should contain orders with no item rows. Got {sorted(orders_without_items['order_id'])}.")
    expect(list(orphan_items.columns) == list(it.columns),
           "orphan_items should have exactly the columns of items_toy (drop `_merge` and columns from orders).")
    expect(list(orders_without_items.columns) == list(o.columns),
           "orders_without_items should have exactly the columns of orders_toy.")
    return "indicator=True turns a join into a diagnostic tool: it tells you what matched and what didn't."


@S.exercise("orders-without-items", "orders without items (real data)", hints=[
    "Same anti-join, now with `orders` and `items`. (`~orders['order_id'].isin(items['order_id'])` is a shortcut.)",
    "status_counts is just `.value_counts()` of order_status on those orders.",
])
def _(n_orders_without_items, status_counts):
    r = _real()
    n = as_int(n_orders_without_items, "n_orders_without_items")
    expect(n == r["n_orders_without_items"], "n_orders_without_items is off. Did you anti-join orders → items (not the other way)?")
    is_series(status_counts, "status_counts")
    got = status_counts.sort_index()
    want = r["status_without_items"]
    expect(list(got.index) == list(want.index) and (got.to_numpy() == want.to_numpy()).all(),
           "status_counts doesn't match. Make sure you computed it on the orders without items.")
    return ("Almost all orders without items are unavailable or canceled: they would silently vanish from any "
            "items-based analysis. That's a selection effect, not a bug, but you must know it's there.")


@S.exercise("order-totals", "order_totals(items, payments)", challenge=True, hints=[
    "Never join two 'many' tables directly. First collapse each one to one row per order_id with groupby().sum().",
    "items_total = sum of price + freight_value per order; paid_total = sum of payment_value per order.",
    "Combine the two order-level Series with pd.concat([...], axis=1) (outer by default) or a merge how='outer', then .fillna(0).",
    "share_mismatch = (abs(items_total - paid_total) > 1).mean() over all rows of your result.",
])
def _(order_totals, share_mismatch):
    def norm(df):
        is_df(df, "order_totals(...)")
        if "order_id" in df.columns:
            df = df.set_index("order_id")
        expect({"items_total", "paid_total"} <= set(df.columns), "Result needs columns `items_total` and `paid_total`.")
        return df[["items_total", "paid_total"]].sort_index()

    got = norm(unchanged(order_totals, toy_items(), toy_payments()))
    want = _ref_order_totals(toy_items(), toy_payments())
    expect(list(got.index) == list(want.index),
           f"Toy check: expected one row per order in *either* table {list(want.index)}, got {list(got.index)}. "
           "Did you use an outer join?")
    expect(np.allclose(got.to_numpy(dtype=float), want.to_numpy()),
           "Toy check: totals don't match. Remember freight_value, and fill missing totals with 0.")
    r = _real()
    got_real = norm(order_totals(load_olist("items"), load_olist("payments")))
    expect(len(got_real) == len(r["order_totals"]), "Real data: wrong number of orders in your result.")
    expect_close(share_mismatch, r["share_mismatch"], "share_mismatch", tol=1e-4)
    return ("Aggregate-then-join is the standard fix for many-to-many. The remaining mismatches are real: "
            "installment interest, vouchers, orders paid but without items...")


# ---- 4. Lookup tables and coverage ---------------------------------------------------


@S.exercise("category-label", "category_label(products, categories)", hints=[
    "Left-join products to categories on product_category_name with validate='many_to_one'.",
    "Fallback chain: English name, else Portuguese name, else 'UNKNOWN' → `.fillna(other_series).fillna('UNKNOWN')`.",
    "A merge returns a new RangeIndex. Return a Series with the *same index as products* (`s.index = products.index`).",
])
def _(category_label):
    p, c = toy_products(), toy_categories()
    got = unchanged(category_label, p, c)
    is_series(got, "category_label(...)")
    want = ["bed_bath_table", "health_beauty", "computers_accessories", "UNKNOWN", "pc_gamer"]
    expect(list(got) == want, f"Toy check: expected {want}, got {list(got)}.")
    p2 = p.set_index(pd.Index([10, 20, 30, 40, 50]))
    expect(list(category_label(p2, c).index) == [10, 20, 30, 40, 50],
           "The result's index must match products' index (try a products table whose index isn't 0..n-1).")
    real = category_label(load_olist("products"), load_olist("categories"))
    expect((real.reset_index(drop=True) == _real()["labels"].reset_index(drop=True)).all(),
           "Toy data passes but real data doesn't. Check categories missing from the translation table.")
    return ("Make 'missing' an explicit, visible category instead of letting NaN silently drop out of groupbys "
            "and joins. Note pc_gamer: present in products but missing from the translation table.")


@S.exercise("untranslated", "categories without a translation", hints=[
    "This is an anti-join on the *set* of category names: which names in products are absent from categories?",
    "Python sets work well here: `set(a.dropna()) - set(b)`. Return a sorted list.",
])
def _(untranslated):
    expect(isinstance(untranslated, (list, tuple, set, pd.Series, np.ndarray, pd.Index)),
           "Give a list of category names.")
    got = sorted(map(str, list(untranslated)))
    expect("nan" not in got, "Leave out the missing category (NaN): it isn't a category name.")
    expect(got == _real()["untranslated"],
           f"Expected {len(_real()['untranslated'])} names, got {len(got)}. Compare products' categories with the "
           "translation table's Portuguese names.")
    return ("A lookup table is a promise of coverage that nobody checked. One line of set arithmetic tells you "
            "exactly which labels will fall back.")


# ---- 5. Invariants ------------------------------------------------------------------------


@S.exercise("checked-left-join", "checked_left_join(left, right, on)", hints=[
    "Use merge(how='left', validate='many_to_one', indicator=True). validate raises MergeError on duplicated right keys.",
    "After merging: assert len(result) == len(left); print how many rows were 'left_only'; drop `_merge` before returning.",
])
def _(checked_left_join):
    it, p = toy_items(), toy_products()
    out = unchanged(lambda a, b: checked_left_join(a, b, "product_id"), it, p)
    is_df(out, "checked_left_join(...)")
    expect(len(out) == len(it), "A many-to-one left join must keep the number of left rows.")
    expect("_merge" not in out.columns, "Drop the `_merge` column before returning.")
    expect("product_category_name" in out.columns, "The right table's columns should be added.")
    res = raises(checked_left_join, Exception, it, toy_products_dirty(), "product_id")
    expect(res is True, "Joining to toy_products_dirty() (pB duplicated) must raise: that's the whole point.")
    res = raises(checked_left_join, Exception, it, toy_orders()[["order_id", "order_status"]], "order_id")
    expect(res is False, "Unmatched rows (o7) should be *reported*, not raise: they're information, not corruption.")
    return "Encode your assumptions as code. A join helper that checks cardinality pays for itself the first time it fires."


# ---- 6. Mini-project: an order-level table ---------------------------------------------


@S.exercise("order-table", "build_order_table(orders, items, payments, customers)", hints=[
    "Aggregate items and payments to one row per order_id FIRST (named aggregation), then left-join everything "
    "onto orders.",
    "Use validate= on every join: customers is many_to_one from orders' side, the aggregates are one_to_one.",
    "n_items is a count, so an order without item rows has 0 items. The money columns stay NaN for orders "
    "without rows: 'no information' isn't 'zero BRL'.",
])
def _(build_order_table):
    o, it, pa, cu = toy_orders(), toy_items(), toy_payments(), toy_customers()
    got = unchanged(build_order_table, o, it, pa, cu)
    is_df(got, "build_order_table(...)")
    expect(list(got.columns) == ORDER_TABLE_COLS, f"Columns should be {ORDER_TABLE_COLS} in that order.")
    want = _ref_order_table(o, it, pa, cu)
    expect(len(got) == len(o) and got["order_id"].is_unique, "Toy check: one row per order, no more, no less.")
    g = got.set_index("order_id").loc[want["order_id"]]
    expect(list(g["n_items"]) == list(want["n_items"]),
           f"Toy check: n_items should be {list(want['n_items'])}. Orders without items have 0 items.")
    for col in ["items_value", "freight_value", "paid_value"]:
        a, b = g[col].to_numpy(dtype=float), want[col].to_numpy(dtype=float)
        expect(np.array_equal(np.isnan(a), np.isnan(b)),
               f"Toy check: `{col}` should be NaN exactly for the orders without rows in that table (o4 has no items).")
        expect(np.allclose(np.nan_to_num(a), np.nan_to_num(b)), f"Toy check: `{col}` values don't match.")
    expect(list(g["customer_unique_id"]) == list(want["customer_unique_id"]), "Toy check: wrong customer_unique_id.")
    r = _real()["order_table"]
    real = build_order_table(load_olist("orders"), load_olist("items"), load_olist("payments"), load_olist("customers"))
    expect(len(real) == len(r) and real["order_id"].is_unique, "Real data: you should get exactly one row per order.")
    real = real.set_index("order_id").loc[r["order_id"]]
    expect(np.allclose(real["paid_value"].fillna(-1).to_numpy(dtype=float), r["paid_value"].fillna(-1).to_numpy()),
           "Real data: paid_value doesn't match.")
    return ("This is the course's 'analytic table': one row per order, every detail table pre-aggregated, every "
            "join validated. Later notebooks build on exactly this shape.")


@S.exercise("top-state-median", "highest median order value", challenge=True, hints=[
    "Group the order table by customer_state: count orders and take the median paid_value.",
    "Keep states with at least 1,000 orders, then `.idxmax()` on the median column.",
])
def _(top_state):
    expect(isinstance(top_state, str), "Give the state code as a string, e.g. 'SP'.")
    r = _real()
    if top_state == r["top_state_any"]:
        expect(False, f"{top_state} has the highest median overall, but it has fewer than 1,000 orders. "
               "Apply the size filter first.")
    expect(top_state == r["top_state_1000"], "Not that one. Filter to states with ≥ 1,000 orders, then take the max median.")
    return ("Without the size filter the 'winner' is a state with 81 orders. Extreme summaries come from small "
            "groups: Notebook 04 explains why, and what to do about it.")
