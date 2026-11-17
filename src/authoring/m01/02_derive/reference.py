"""⚠️ SPOILERS — reference solutions for Module I · 02. Run by verify.py; every check must pass."""
import numbers
import os

os.environ["SELFGRADE_NO_LOG"] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from selfgrade.data import load_olist  # noqa: E402
from selfgrade.m01 import derive as G  # noqa: E402

orders, items = load_olist("orders"), load_olist("items")
products, categories, customers = load_olist("products"), load_olist("categories"), load_olist("customers")

labels = products[["product_id", "product_category_name"]].merge(
    categories, on="product_category_name", how="left", validate="many_to_one")
labels["category"] = labels["product_category_name_english"].fillna(labels["product_category_name"]).fillna("UNKNOWN")
delivered_orders = orders.loc[orders["order_status"].eq("delivered"), ["order_id", "order_purchase_timestamp"]]
analysis = (items.merge(delivered_orders, on="order_id", how="inner", validate="many_to_one")
                 .merge(labels[["product_id", "category"]], on="product_id", how="left", validate="many_to_one"))

n_2017 = int((orders["order_status"].eq("delivered")
              & orders["order_purchase_timestamp"].dt.year.eq(2017)
              & orders["order_delivered_customer_date"].notna()).sum())
sp_ids = customers.loc[customers["customer_state"].eq("SP"), "customer_id"]
delivered_sp = orders[orders["order_status"].eq("delivered") & orders["customer_id"].isin(sp_ids)]
n_before = int((orders["order_delivered_customer_date"] < orders["order_approved_at"]).sum())


def category_summary(df):
    return (df.groupby("category")
            .agg(rows=("price", "size"), distinct_orders=("order_id", "nunique"),
                 price_sum=("price", "sum"), price_median=("price", "median"))
            .sort_values("price_sum", ascending=False))


def add_share_of_order(df):
    return df.assign(share_of_order=df["price"] / df.groupby("order_id")["price"].transform("sum"))


sh = add_share_of_order(analysis).groupby("order_id")["share_of_order"].agg(["max", "size"])
share_dominated = (sh.loc[sh["size"] >= 2, "max"] > 0.5).mean()
nov = analysis[analysis["order_purchase_timestamp"].dt.to_period("M") == "2017-11"]

oc = orders.merge(customers[["customer_id", "customer_state"]], on="customer_id", validate="one_to_one")
status_share = pd.crosstab(oc["customer_state"], oc["order_status"], normalize="index")


def add_delivery_days(orders):
    days = (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.total_seconds() / 86400
    return orders.assign(delivery_days=days.where(orders["order_status"].eq("delivered")))


delivered = (add_delivery_days(orders)
             .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left", validate="one_to_one")
             .dropna(subset=["delivery_days"]))
st = delivered.groupby("customer_state")["delivery_days"].agg(["size", "median"])
state_median = delivered.groupby("customer_state")["delivery_days"].transform("median")


def price_band_select(p):
    return np.select([p < 50, p < 100, p < 200], ["<50", "50-100", "100-200"], default="200+")


def price_band_cut(p):
    return pd.cut(p, [0, 50, 100, 200, np.inf], right=False, labels=["<50", "50-100", "100-200", "200+"])


def price_band_loop(prices):
    out = []
    for x in prices:
        if x < 50:
            out.append("<50")
        elif x < 100:
            out.append("50-100")
        elif x < 200:
            out.append("100-200")
        else:
            out.append("200+")
    return out


def top_categories(df, n=5):
    if isinstance(n, bool) or not isinstance(n, numbers.Integral):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 1:
        raise ValueError("n must be >= 1")
    missing = {"category", "price"} - set(df.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")
    return df.groupby("category")["price"].sum().sort_values(ascending=False).head(n).rename("price_sum")


G.S.verify_reference({
    "delivered-2017": G.check("delivered-2017", n_2017),
    "delivered-sp": G.check("delivered-sp", delivered_sp),
    "delivered-before-approved": G.check("delivered-before-approved", n_before),
    "category-summary": G.check("category-summary", category_summary),
    "two-medians": G.check("two-medians", analysis["price"].median(),
                           analysis.groupby("order_id")["price"].sum().median()),
    "share-of-order": G.check("share-of-order", add_share_of_order, share_dominated),
    "top-category-nov-2017": G.check("top-category-nov-2017", nov.groupby("category")["price"].sum().idxmax()),
    "status-share": G.check("status-share", status_share),
    "delivery-days": G.check("delivery-days", add_delivery_days),
    "slow-vs-state": G.check("slow-vs-state", (delivered["delivery_days"] > 2 * state_median).mean()),
    "price-band-select": G.check("price-band-select", price_band_select),
    "price-band-cut": G.check("price-band-cut", price_band_cut),
    "top-categories": G.check("top-categories", top_categories),
    "bool-trap": G.check("bool-trap", top_categories),
})
