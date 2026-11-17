"""Datasets used by the notebooks, downloaded on first use and cached locally.

Olist Brazilian E-Commerce data, from https://github.com/olist/work-at-olist-data
(MIT License, © 2019 Olist), pinned to one commit so everyone gets identical bytes.

    from selfgrade.data import load_olist, olist_path
    orders = load_olist("orders")          # fresh copy, dates parsed, IDs as strings
    path = olist_path("geolocation")       # just the file path (downloads it if needed)

Set the environment variable OLIST_DATA_DIR to a folder that already contains the CSVs
to skip the download.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import repo_root

OLIST_REPO = "olist/work-at-olist-data"
OLIST_COMMIT = "d9e49802f3e92d09ee94ab9ccc5e457f207a8959"
_RAW = f"https://raw.githubusercontent.com/{OLIST_REPO}/{OLIST_COMMIT}/datasets/"

# name -> (file name, size in bytes, sha256 of the file at OLIST_COMMIT)
OLIST_FILES: dict[str, tuple[str, int, str]] = {
    "orders": ("olist_orders_dataset.csv", 17654914,
               "8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf"),
    "items": ("olist_order_items_dataset.csv", 15007623,
              "4f6abdbbc94036d0df4a76fa0520c072e31a40119d70f7f370fba1e2285d2bcb"),
    "payments": ("olist_order_payments_dataset.csv", 5647783,
                 "61674868ed7b872aac3dcd95bf774448a88f6426b183f21be55f75af021ffdec"),
    "reviews": ("olist_order_reviews_dataset.csv", 14451670,
                "012b61c7593e34f51fa614efdf802b9c7056ce6aae5307ddb93236e7cfc797d7"),
    "customers": ("olist_customers_dataset.csv", 8562270,
                  "c26c17f59f3027a6a0dbeb8f1fa373c38f3323ecc84f225a8f4820e5bb288df8"),
    "sellers": ("olist_sellers_dataset.csv", 163589,
                "31eb9bd50d684526a78f2389413ce61841e97692de7b828fe7d956b534586312"),
    "products": ("olist_products_dataset.csv", 2379446,
                 "3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc"),
    "categories": ("product_category_name_translation.csv", 2613,
                   "a81f0d1f27b27e7293f761bc79e3ce8f348ee39c4b3ed3e49bde38f478586278"),
    "geolocation": ("olist_geolocation_dataset.csv", 61273883,
                    "b514f6fc991b9566aeba02aa5d67e2c3630f034b60a0e05aa0d082a3b66d88d6"),
}

DATE_COLUMNS = {
    "orders": ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
               "order_delivered_customer_date", "order_estimated_delivery_date"],
    "items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}


def data_dir() -> Path:
    """OLIST_DATA_DIR if set, else <repo>/data/olist (or ~/.selfgrade/data/olist outside a clone)."""
    env = os.environ.get("OLIST_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    root = repo_root()
    return (root / "data" / "olist") if root else Path.home() / ".selfgrade" / "data" / "olist"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(fname: str, size: int, sha: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {fname} ({size / 2**20:.1f} MB) from {OLIST_REPO}…", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(_RAW + fname, part)
    except OSError as e:
        part.unlink(missing_ok=True)
        raise ConnectionError(
            f"Could not download {fname}: {e}.\nCheck your internet connection, or download the datasets/ "
            f"folder of https://github.com/{OLIST_REPO} yourself and set OLIST_DATA_DIR to it."
        ) from None
    got = _sha256(part)
    if got != sha:
        part.unlink(missing_ok=True)
        raise ValueError(f"{fname}: checksum mismatch (got {got[:12]}…, expected {sha[:12]}…). Try again.")
    part.replace(dest)
    print("done.")


def olist_path(name: str) -> Path:
    """Path to one Olist CSV, downloading it on first use."""
    if name not in OLIST_FILES:
        raise KeyError(f"Unknown table {name!r}. Available: {', '.join(OLIST_FILES)}")
    fname, size, sha = OLIST_FILES[name]
    path = data_dir() / fname
    if path.is_file():
        return path
    if os.environ.get("OLIST_DATA_DIR"):
        raise FileNotFoundError(f"OLIST_DATA_DIR is set to {data_dir()}, but {fname} isn't there.")
    _download(fname, size, sha, path)
    return path


@lru_cache(maxsize=None)
def _load_cached(name: str) -> pd.DataFrame:
    path = olist_path(name)
    header = pd.read_csv(path, nrows=0).columns
    as_text = {c: "string" for c in header if c.endswith("_zip_code_prefix")}
    return pd.read_csv(path, parse_dates=DATE_COLUMNS.get(name), dtype=as_text or None)


def load_olist(name: str) -> pd.DataFrame:
    """A fresh copy of an Olist table (safe to modify): dates parsed, zip-code prefixes kept as text."""
    return _load_cached(name).copy()


def fetch_all(include_geolocation: bool = False) -> None:
    """Download every table up front (e.g. before going offline)."""
    for name in OLIST_FILES:
        if name != "geolocation" or include_geolocation:
            olist_path(name)
