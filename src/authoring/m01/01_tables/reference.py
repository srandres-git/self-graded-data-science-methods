"""⚠️ SPOILERS — reference solutions for Module I · 01. Run by verify.py; every check must pass."""
import os

os.environ["SELFGRADE_NO_LOG"] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from selfgrade.data import load_olist, olist_path  # noqa: E402
from selfgrade.m01 import tables as G  # noqa: E402

orders, items, payments = load_olist("orders"), load_olist("items"), load_olist("payments")
reviews, customers = load_olist("reviews"), load_olist("customers")
products, categories = load_olist("products"), load_olist("categories")


def normalize_zip(s):
    return s.astype("string").str.zfill(5)


geo_zips = set(pd.read_csv(olist_path("geolocation"), usecols=["geolocation_zip_code_prefix"],
                           dtype="string")["geolocation_zip_code_prefix"])
zips = customers["customer_zip_code_prefix"]


def geo_rows_by_state(path, chunksize=200_000):
    total = pd.Series(dtype="float64")
    for chunk in pd.read_csv(path, usecols=["geolocation_state"], chunksize=chunksize):
        total = total.add(chunk["geolocation_state"].value_counts(), fill_value=0)
    return total.astype(int)


def is_key(df, cols):
    cols = [cols] if isinstance(cols, str) else list(cols)
    return bool(not df[cols].isna().any().any() and not df.duplicated(subset=cols).any())


it, o = G.toy_items(), G.toy_orders()
m = it.merge(o[["order_id"]], on="order_id", how="left", indicator=True)
orphan_items = m.loc[m["_merge"] == "left_only", it.columns]
m2 = o.merge(it[["order_id"]].drop_duplicates(), on="order_id", how="left", indicator=True)
orders_without_items = m2.loc[m2["_merge"] == "left_only", o.columns]
no_items = orders[~orders["order_id"].isin(items["order_id"])]


def order_totals(items, payments):
    a = (items["price"] + items["freight_value"]).groupby(items["order_id"]).sum().rename("items_total")
    b = payments.groupby("order_id")["payment_value"].sum().rename("paid_total")
    return pd.concat([a, b], axis=1).fillna(0.0)


ot = order_totals(items, payments)


def category_label(products, categories):
    m = products[["product_category_name"]].merge(categories, on="product_category_name", how="left",
                                                  validate="many_to_one")
    s = m["product_category_name_english"].fillna(m["product_category_name"]).fillna("UNKNOWN")
    s.index = products.index
    return s


def checked_left_join(left, right, on):
    out = left.merge(right, on=on, how="left", validate="many_to_one", indicator=True)
    assert len(out) == len(left), "row count changed"
    n_unmatched = int((out["_merge"] == "left_only").sum())
    if n_unmatched:
        print(f"  checked_left_join: {n_unmatched} left rows without a match")
    return out.drop(columns="_merge")


def build_order_table(orders, items, payments, customers):
    it = items.groupby("order_id").agg(n_items=("price", "size"), items_value=("price", "sum"),
                                       freight_value=("freight_value", "sum"))
    pa = payments.groupby("order_id").agg(paid_value=("payment_value", "sum"))
    out = (orders[["order_id", "customer_id", "order_status"]]
           .merge(customers[["customer_id", "customer_unique_id", "customer_state"]],
                  on="customer_id", how="left", validate="many_to_one")
           .merge(it, on="order_id", how="left", validate="one_to_one")
           .merge(pa, on="order_id", how="left", validate="one_to_one"))
    out["n_items"] = out["n_items"].fillna(0).astype(int)
    return out[["order_id", "order_status", "customer_unique_id", "customer_state",
                "n_items", "items_value", "freight_value", "paid_value"]]


ordt = build_order_table(orders, items, payments, customers)
st = ordt.groupby("customer_state").agg(n=("order_id", "size"), med=("paid_value", "median"))
k = customers.groupby("customer_unique_id")["customer_id"].nunique()

G.S.verify_reference({
    "normalize-zip": G.check("normalize-zip", normalize_zip),
    "zip-coverage": G.check("zip-coverage", int(zips.isin(geo_zips).sum()),
                            int((~normalize_zip(zips).isin(geo_zips)).sum())),
    "geo-rows-by-state": G.check("geo-rows-by-state", geo_rows_by_state),
    "is-key": G.check("is-key", is_key),
    "review-duplicates": G.check("review-duplicates", int(reviews["review_id"].duplicated().sum()),
                                 int((reviews["order_id"].value_counts() > 1).sum())),
    "review-pair-key": G.check("review-pair-key", is_key(reviews, ["review_id", "order_id"])),
    "customers-vs-people": G.check("customers-vs-people", customers["customer_id"].nunique(),
                                   customers["customer_unique_id"].nunique(), int((k > 1).sum())),
    "anti-joins": G.check("anti-joins", orphan_items, orders_without_items),
    "orders-without-items": G.check("orders-without-items", len(no_items), no_items["order_status"].value_counts()),
    "order-totals": G.check("order-totals", order_totals,
                            ((ot["items_total"] - ot["paid_total"]).abs() > 1).mean()),
    "category-label": G.check("category-label", category_label),
    "untranslated": G.check("untranslated", sorted(set(products["product_category_name"].dropna())
                                                   - set(categories["product_category_name"]))),
    "checked-left-join": G.check("checked-left-join", checked_left_join),
    "order-table": G.check("order-table", build_order_table),
    "top-state-median": G.check("top-state-median", st[st["n"] >= 1000]["med"].idxmax()),
})
