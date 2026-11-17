"""⚠️ SPOILERS — reference solutions for Module I · 07. Run by verify.py; every check must pass."""
import os

os.environ["SELFGRADE_NO_LOG"] = "1"

import hashlib  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from selfgrade.data import load_olist  # noqa: E402
from selfgrade.m01 import cleaning as G  # noqa: E402

orders, items = load_olist("orders"), load_olist("items")
payments, customers = load_olist("payments"), load_olist("customers")
item_agg = items.groupby("order_id").agg(n_items=("price", "size"), items_value=("price", "sum"),
                                         freight_value=("freight_value", "sum"))
paid_agg = payments.groupby("order_id").agg(paid_value=("payment_value", "sum"))
order_table = (orders[["order_id", "order_status", "order_purchase_timestamp", "customer_id"]]
               .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left",
                      validate="many_to_one")
               .merge(item_agg, on="order_id", how="left", validate="one_to_one")
               .merge(paid_agg, on="order_id", how="left", validate="one_to_one")
               .drop(columns="customer_id"))
order_table["n_items"] = order_table["n_items"].fillna(0).astype(int)


def audit(df, rules):
    n = {name: int(pd.Series(rule(df)).fillna(False).astype(bool).sum()) for name, rule in rules.items()}
    out = pd.DataFrame({"n_violations": n}, index=list(rules))
    out["share"] = out["n_violations"] / len(df)
    return out


flag = orders["order_delivered_carrier_date"] < orders["order_approved_at"]
gap_h = (orders["order_approved_at"] - orders["order_delivered_carrier_date"])[flag].dt.total_seconds() / 3600


def winsorize(s, lower=0.01, upper=0.99):
    lo, hi = s.quantile([lower, upper])
    return s.clip(lower=lo, upper=hi)


p = items["price"]
w = winsorize(p)


def min_max(s):
    lo, hi = s.min(), s.max()
    if not hi > lo:
        return s.where(s.isna(), 0.0).astype(float)
    return (s - lo) / (hi - lo)


def z_score(s):
    sd = s.std()
    if pd.isna(sd) or sd == 0:
        return s.where(s.isna(), 0.0).astype(float)
    return (s - s.mean()) / sd


def robust_scale(s):
    q1, q3 = s.quantile([0.25, 0.75])
    if not q3 - q1 > 0:
        raise ValueError("IQR is 0, so robust scaling is undefined for this column")
    return (s - s.median()) / (q3 - q1)


def quartile_bins(s):
    return pd.qcut(s, 4, labels=["Q1", "Q2", "Q3", "Q4"])


def one_hot(df, col, drop_first=False):
    levels = sorted(df[col].dropna().unique())
    if drop_first:
        levels = levels[1:]
    out = df.drop(columns=col)
    for lv in levels:
        out[f"{col}_{lv}"] = (df[col] == lv).astype("int8")
    if df[col].isna().any():
        out[f"{col}_MISSING"] = df[col].isna().astype("int8")
    return out


def ordinal_codes(s, order):
    return s.map({v: i for i, v in enumerate(order)}).astype("Int64")


def freight_ratio(df):
    den = df["items_value"]
    return df["freight_value"] / den.where(den > 0)


fr = freight_ratio(order_table)

d = orders[orders["order_status"].eq("delivered") & orders["order_delivered_customer_date"].notna()].copy()
d["days"] = (d["order_delivered_customer_date"] - d["order_purchase_timestamp"]).dt.total_seconds() / 86400
train = d.loc[d["order_purchase_timestamp"] < "2018-01-01", "days"]
test = d.loc[d["order_purchase_timestamp"] >= "2018-01-01", "days"]


def data_fingerprint(path, n=12):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


def clean_orders(df):
    out = df.copy()
    log = []
    out["paid_missing"] = out["paid_value"].isna()
    log.append(dict(step="paid_missing", rule="paid_value is NaN", rows_affected=int(out["paid_missing"].sum()),
                    action="indicator column", limitation="doesn't explain why the payment is missing"))
    out["no_items"] = out["n_items"].eq(0)
    log.append(dict(step="no_items", rule="n_items == 0", rows_affected=int(out["no_items"].sum()),
                    action="flag", limitation="mostly canceled/unavailable orders: a selection effect"))
    out["freight_ratio"] = freight_ratio(out)
    log.append(dict(step="freight_ratio", rule="freight / items_value if items_value > 0",
                    rows_affected=int(out["freight_ratio"].isna().sum()), action="new feature, NaN when undefined",
                    limitation="undefined for orders without items"))
    out["log_paid"] = np.log1p(out["paid_value"])
    log.append(dict(step="log_paid", rule="log1p(paid_value)", rows_affected=int(out["log_paid"].isna().sum()),
                    action="new feature", limitation="interpret differences as ratios"))
    q1, q3 = out["log_paid"].quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    out["paid_outlier"] = (out["log_paid"].lt(lo) | out["log_paid"].gt(hi)).fillna(False)
    log.append(dict(step="paid_outlier", rule="Tukey fences on log_paid", rows_affected=int(out["paid_outlier"].sum()),
                    action="flag only, never delete", limitation="fences are a convention; extremes can be real"))
    return out, pd.DataFrame(log)


G.S.verify_reference({
    "audit-table": G.check("audit-table", audit),
    "event-order": G.check("event-order", int(flag.sum()), gap_h.median()),
    "winsorize": G.check("winsorize", winsorize),
    "winsor-sensitivity": G.check("winsor-sensitivity", (w.mean() - p.mean()) / p.mean(),
                                  (w.median() - p.median()) / p.median()),
    "scalers": G.check("scalers", min_max, z_score),
    "robust-scale": G.check("robust-scale", robust_scale),
    "quartile-bins": G.check("quartile-bins", quartile_bins),
    "one-hot": G.check("one-hot", one_hot),
    "ordinal-codes": G.check("ordinal-codes", ordinal_codes),
    "freight-ratio": G.check("freight-ratio", freight_ratio, (fr[fr.notna()] > 1).mean()),
    "leakage-gap": G.check("leakage-gap", ((test - d["days"].mean()) / d["days"].std()).mean(),
                           ((test - train.mean()) / train.std()).mean()),
    "data-fingerprint": G.check("data-fingerprint", data_fingerprint),
    "clean-orders": G.check("clean-orders", clean_orders),
})
