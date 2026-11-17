"""⚠️ SPOILERS — reference solutions for Module I · 06. Run by verify.py; every check must pass."""
import os

os.environ["SELFGRADE_NO_LOG"] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from selfgrade.data import load_olist  # noqa: E402
from selfgrade.m01 import missing as G  # noqa: E402

orders, items, reviews = load_olist("orders"), load_olist("items"), load_olist("reviews")
products, categories = load_olist("products"), load_olist("categories")

labels = products[["product_category_name"]].merge(categories, on="product_category_name", how="left",
                                                   validate="many_to_one")
category = labels["product_category_name_english"].fillna(labels["product_category_name"]).fillna("UNKNOWN")
category.index = products.index
delivered_orders = orders.loc[orders["order_status"].eq("delivered"), ["order_id", "order_purchase_timestamp"]]
analysis = (items.merge(delivered_orders, on="order_id", how="inner", validate="many_to_one")
            .merge(products[["product_id"]].assign(category=category), on="product_id", how="left",
                   validate="many_to_one"))


def missing_profile(df):
    na = df.isna()
    out = pd.DataFrame({"n_missing": na.sum(), "share_missing": na.mean()})
    return out.sort_values("n_missing", ascending=False, kind="stable")


nodate = orders["order_delivered_customer_date"].isna()
is_deliv = orders["order_status"].eq("delivered")
listing = products[G.LISTING_COLS].isna()
unk = analysis["category"].eq("UNKNOWN")
w = products["product_weight_g"]


def impute_by_group(df, value_col, group_col):
    out = df.copy()
    out[f"{value_col}_was_missing"] = out[value_col].isna()
    med = out.groupby(group_col)[value_col].transform("median")
    out[value_col] = out[value_col].fillna(med).fillna(out[value_col].median())
    return out


mode = products["product_category_name"].mode()[0]
hidden = w.where(w <= 5000)

dv = orders[is_deliv & ~nodate]
days = (dv["order_delivered_customer_date"] - dv["order_purchase_timestamp"]).dt.total_seconds() / 86400
reviewed = (reviews[["review_id", "order_id", "review_score", "review_comment_message"]]
            .merge(dv[["order_id"]].assign(days=days.to_numpy()), on="order_id", how="inner", validate="many_to_one")
            .assign(has_comment=lambda d: d["review_comment_message"].notna()))
pi = reviewed.groupby("review_score")["has_comment"].transform("mean")
c = reviewed["has_comment"]

G.S.verify_reference({
    "nullable-int": G.check("nullable-int", products["product_photos_qty"].astype("Int64")),
    "missing-profile": G.check("missing-profile", missing_profile),
    "missing-by-status": G.check("missing-by-status", nodate.groupby(orders["order_status"]).mean()),
    "status-date-audit": G.check("status-date-audit", int((is_deliv & nodate).sum()), int((~is_deliv & ~nodate).sum())),
    "missing-together": G.check("missing-together", int(listing.all(axis=1).sum()),
                                int((listing.any(axis=1) & ~listing.all(axis=1)).sum())),
    "unknown-share": G.check("unknown-share", unk.mean(), analysis.loc[unk, "price"].sum() / analysis["price"].sum()),
    "listwise-loss": G.check("listwise-loss", len(products) - len(products.dropna()),
                             len(products) - len(products.dropna(subset=["product_weight_g"]))),
    "impute-group-median": G.check("impute-group-median", impute_by_group),
    "shrinkage-check": G.check("shrinkage-check", w.var(), w.fillna(w.mean()).var()),
    "mode-inflation": G.check("mode-inflation", products["product_category_name"].eq(mode).mean(),
                              products["product_category_name"].fillna(mode).eq(mode).mean()),
    "mnar-bias": G.check("mnar-bias", hidden.mean(), hidden.median()),
    "comment-by-score": G.check("comment-by-score",
                                reviews["review_comment_message"].notna().groupby(reviews["review_score"]).mean()),
    "cc-bias": G.check("cc-bias", reviewed["days"].mean(), reviewed.loc[c, "days"].mean()),
    "ipw-days": G.check("ipw-days", np.average(reviewed.loc[c, "days"], weights=1 / pi[c])),
})
