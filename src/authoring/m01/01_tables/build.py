"""Builder: Module I · 01 — Tables you can trust."""
from nbbuild import Notebook

COURSE = "https://github.com/chaffrey8/DMEMCD_2026"

nb = Notebook(
    nb_id="m01/01",
    filename="01_tables-you-can-trust.ipynb",
    title="Tables you can trust",
    subtitle="Reading CSVs, keys, joins, and the checks that keep them honest",
    grader="selfgrade.m01.tables",
    can_do=["read a CSV and verify what types you actually got",
            "name the unit of observation and the key of any table",
            "predict a join's row count from its cardinality",
            "find unmatched rows with anti-joins",
            "build an analysis table that keeps exactly one row per order"],
    builds_on="Python, plus a first contact with pandas (`DataFrame`, selecting columns, `read_csv`). "
              "Everything else is explained here.",
    effort="≈ 2 h",
    sources=[f"Diplomado *Métodos Estadísticos y Matemáticos para la Ciencia de Datos* (UNAM, 2026), Module I: "
             f"[`S03_Notebook_Demostracion_Python`, `S03_Notebook_Laboratorio_Estudiante_Python`, "
             f"`01_Lectura_Integracion_Work_Olist`]({COURSE})"],
    concept_map=["the data", "reading a CSV", "unit of observation & keys", "joins & cardinality",
                 "lookup tables & coverage", "invariants", "mini-project: an order-level table"],
)

# ---------------------------------------------------------------------------
nb.md(r"""
## 0 · The data and the setup

All of Module I uses the **Olist** dataset: about 100,000 orders placed from 2016 to 2018 on Olist, a Brazilian marketplace that lets small shops sell through the big e-commerce sites. It's real (anonymized) data, published as several CSV files that reference each other, much like the tables of a relational database:

| Table | What it holds |
|---|---|
| `orders` | each order's status and its timestamps: purchase, approval, hand-off to the carrier, delivery, estimated delivery |
| `items` | the products in each order, with price and freight (shipping cost) |
| `payments` | how each order was paid: method, installments, amount |
| `reviews` | the customer's satisfaction review (score 1–5, optional comment) |
| `customers` | who placed the order, and where (city, state, postal-code prefix) |
| `products` | the catalog: category (in Portuguese) and physical attributes |
| `categories` | English translations of the category names |
| `sellers`, `geolocation` | where sellers are, and coordinates for each postal-code prefix |

Amounts are in Brazilian reais (BRL). States are two-letter codes: SP is São Paulo, RJ Rio de Janeiro, MG Minas Gerais, and so on.

`load_olist(name)` downloads a table the first time you ask for it (checksum-verified, cached in `data/`) and returns a fresh copy you can modify freely. It already applies two fixes you'll learn about in section 1: dates are parsed, and postal-code prefixes are kept as text.
""")
nb.code(r"""
import numpy as np
import pandas as pd
from IPython.display import display

from selfgrade.data import load_olist, olist_path
from selfgrade.m01.tables import check, hint, predict, progress
from selfgrade.m01.tables import (toy_orders, toy_items, toy_payments, toy_customers,
                                  toy_products, toy_products_dirty, toy_categories)

orders, items, payments = load_olist("orders"), load_olist("items"), load_olist("payments")
reviews, customers = load_olist("reviews"), load_olist("customers")
products, categories = load_olist("products"), load_olist("categories")

for name, df in [("orders", orders), ("items", items), ("payments", payments), ("reviews", reviews),
                 ("customers", customers), ("products", products), ("categories", categories)]:
    print(f"{name:<11} {df.shape[0]:>7,} rows × {df.shape[1]:>2} columns")
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 1 · Reading a CSV: types are guesses

A CSV file is plain text: a header line, then one line per row, with fields separated by commas. It stores **no types**. When `pd.read_csv` loads it, pandas *infers* a type (a **dtype**) for each column by looking at its values:

- all values parse as integers → `int64`
- numbers with decimals, **or integers with at least one missing value** → `float64`. The missing-value marker `NaN` ("not a number") is a float, so a single gap turns a whole integer column into floats.
- anything else → `object`, meaning each cell holds a Python string

Two consequences bite all the time:
- **Dates aren't recognized.** They stay strings unless you pass `parse_dates=[...]` or convert them afterwards with `pd.to_datetime(col, errors="coerce")`. With `errors="coerce"`, unparseable values become `NaT` ("not a time", the datetime version of `NaN`).
- **Codes that look like numbers become numbers.** Postal codes, phone numbers and product codes lose their leading zeros, and now support arithmetic that makes no sense. Keep them as text with `dtype={"col": "string"}`.

| `read_csv` parameter | Use it to |
|---|---|
| `dtype={"col": "string"}` | force a type instead of letting pandas guess |
| `parse_dates=["col", ...]` | parse those columns as datetimes |
| `usecols=[...]` | read only some columns (faster, and less memory) |
| `nrows=5` | peek at the first rows of a large file |
| `chunksize=100_000` | read the file in pieces, one DataFrame at a time |

> ✅ **Good practice:** right after reading a file, look at `df.dtypes` and `df.head()`, and make sure each column has the type its *meaning* calls for, not the one pandas guessed.
""")
nb.md(r"""
🔮 **1.1** Read `orders` *without* `parse_dates`. What dtype will `order_purchase_timestamp` have? Answer with the name pandas prints, as a string.
""")
nb.learner("dates-as-text", r"""
predict("dates-as-text", ...)
""")
nb.code(r"""
raw_orders = pd.read_csv(olist_path("orders"), nrows=5)
print(raw_orders.dtypes, "\n")
print("as loaded by load_olist:", orders["order_purchase_timestamp"].dtype)
""")
nb.md(r"""
**A key written two ways.** A Brazilian postal code (CEP) has 8 digits. Olist only publishes the first 5, the *zip-code prefix*, to protect privacy. Prefixes that start with `0` belong to São Paulo state, in the capital and its surroundings. Look at what happened to that zero in two different files:
""")
nb.code(r"""
print("customers, read naively:", pd.read_csv(olist_path("customers"), nrows=1000)["customer_zip_code_prefix"].dtype)

geolocation_head = pd.read_csv(olist_path("geolocation"), nrows=3, dtype={"geolocation_zip_code_prefix": "string"})
display(geolocation_head)

lengths = customers["customer_zip_code_prefix"].str.len().value_counts()
print("length of customers' prefixes (as text):\n", lengths.to_string())
""")
nb.md(r"""
The customers file has 23,995 four-character prefixes, all in São Paulo state. Their leading zero was lost *before* the file was published, probably through a spreadsheet or a numeric column. Reading the column as text keeps what's in the file, but it can't restore what isn't there. The geolocation file kept its zeros. A text join between the two would silently miss every one of those customers.

✍️ **1.2** Write `normalize_zip(s)`, which returns the prefixes as **5-character text, left-padded with zeros**. It should:
- accept integers or text
- keep missing values missing
- keep the input's index
- not modify its input
""")
nb.learner("normalize-zip", r"""
def normalize_zip(s: pd.Series) -> pd.Series:
    \"\"\"Zip-code prefixes as 5-character text ('9790' or 9790 -> '09790'); missing stays missing.\"\"\"
    # your code here
    raise NotImplementedError


check("normalize-zip", normalize_zip)
""")
nb.md(r"""
✍️ **1.3** How many customers can actually be placed on a map? Load the set of prefixes in `geolocation`, reading only that column, as text. Then compute:
- `n_matched_as_text`: customers whose prefix, **as loaded**, appears in geolocation
- `n_unmatched_after_fix`: customers whose **normalized** prefix still doesn't appear
""")
nb.learner("zip-coverage", r"""
n_matched_as_text = ...
n_unmatched_after_fix = ...

check("zip-coverage", n_matched_as_text, n_unmatched_after_fix)
""")
nb.md(r"""
★ **1.4** `geolocation` has a million rows (60 MB). You only need a count per state, so there's no reason to hold the whole file in memory. With `chunksize=`, `read_csv` returns an *iterator*: a loop then receives one DataFrame of at most that many rows at a time.

Write `geo_rows_by_state(path)`. It returns a Series `state → number of rows`, computed chunk by chunk and reading only the column you need.
""")
nb.learner("geo-rows-by-state", r"""
def geo_rows_by_state(path, chunksize: int = 200_000) -> pd.Series:
    # your code here
    raise NotImplementedError


check("geo-rows-by-state", geo_rows_by_state)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 2 · Unit of observation and keys

The **unit of observation** is *what one row represents*: an order, an item within an order, a payment, a person. It's the first thing to know about a table, because it decides what "count", "sum" and "median" mean. A count of rows of `items` is a count of item lines, not of orders.

A **key** is a set of columns whose values identify exactly one row. A valid key needs two things:
1. **no missing values** in those columns, and
2. **no repeated combinations**.

When one column isn't enough, a **composite key** combines several: `(order_id, order_item_id)` identifies one item line, since `order_item_id` numbers the items 1, 2, 3, … within each order. A **foreign key** is a column that points to another table's key. `items.order_id` refers to `orders.order_id`, and that link is what joins use.

> 🧰 **How the code works:** `s.is_unique` says whether a Series has no repeats. `df.duplicated(subset=cols)` marks every row whose combination of `cols` already appeared *above* it, so its `.sum()` counts the extra rows. `s.nunique()` counts distinct non-missing values.

Two toy tables, small enough to check by eye:
""")
nb.code(r"""
orders_toy, items_toy = toy_orders(), toy_items()
display(orders_toy, items_toy)
""")
nb.md(r"""
🔮 **2.1** `items_toy` has 9 rows. How many **distinct orders** does it contain? Count by eye, don't run code.
""")
nb.learner("distinct-orders", r"""
predict("distinct-orders", ...)
""")
nb.code(r"""
print("rows:", len(items_toy), "| distinct order_id:", items_toy["order_id"].nunique())
print("order_id unique in orders_toy?", orders_toy["order_id"].is_unique)
print("(order_id, order_item_id) repeated in items_toy?", items_toy.duplicated(["order_id", "order_item_id"]).any())
""")
nb.md(r"""
✍️ **2.2** Write `is_key(df, cols)`, which returns `True` when `cols` (a column name or a list of names) is a valid key of `df`. The check tests it on several tricky tables, including ones with missing values and composite keys.
""")
nb.learner("is-key", r"""
def is_key(df: pd.DataFrame, cols) -> bool:
    \"\"\"True if `cols` uniquely identify the rows of df: no missing values, no repeated combinations.\"\"\"
    # your code here
    raise NotImplementedError


check("is-key", is_key)
""")
nb.md(r"""
✍️ **2.3** Now the real `reviews` table. You'd expect `review_id` to be its key... Compute:
- `n_dup_review_id`: how many rows repeat a `review_id` that already appeared
- `n_orders_multi_review`: how many `order_id`s have **more than one** review
""")
nb.learner("review-duplicates", r"""
n_dup_review_id = ...
n_orders_multi_review = ...

check("review-duplicates", n_dup_review_id, n_orders_multi_review)
""")
nb.md(r"""
★ **2.4** Is the pair `(review_id, order_id)` a key of `reviews`?
""")
nb.learner("review-pair-key", r"""
pair_is_key = ...

check("review-pair-key", pair_is_key)
""")
nb.md(r"""
💬 **2.5** In one or two sentences: what does one row of `reviews` represent? What could go wrong if you joined `reviews` to `orders` on `order_id` without thinking about it?
""")
nb.answer("reviews-unit")
nb.md(r"""
✍️ **2.6** `customers` has two identifiers: `customer_id` and `customer_unique_id`. Compute:
- `n_customer_ids`: distinct `customer_id` values
- `n_people`: distinct `customer_unique_id` values
- `n_people_multi`: how many `customer_unique_id`s are linked to **more than one** `customer_id`
""")
nb.learner("customers-vs-people", r"""
n_customer_ids = ...
n_people = ...
n_people_multi = ...

check("customers-vs-people", n_customer_ids, n_people, n_people_multi)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 3 · Joins and cardinality

A **join** (`merge` in pandas) combines two tables by matching rows on a key. You make two separate decisions.

**1. Which unmatched rows to keep (`how=`).** Let $L$ and $R$ be the two tables:

| `how=` | keeps |
|---|---|
| `"inner"` | only rows that match on both sides |
| `"left"` | every row of $L$; columns from $R$ are `NaN` where nothing matched |
| `"right"` | every row of $R$ (the mirror image of `left`) |
| `"outer"` | every row of both |

**2. What cardinality to expect.** Cardinality is how many rows can share a key value on each side. For an inner join, the result size is

$$|L \bowtie R| \;=\; \sum_{k} n_L(k)\, n_R(k),$$

where $n_L(k)$ and $n_R(k)$ count the rows with key value $k$ on each side, and the sum runs over the values present in both. The four cases:
- **one-to-one**: keys unique on both sides
- **many-to-one**: keys repeat on the left, unique on the right (items → orders). Each left row matches at most once, so **the row count can't grow**.
- **one-to-many**: the mirror image
- **many-to-many**: keys repeat on *both* sides. Rows multiply ($n_L \cdot n_R$ per key), and any sum computed afterwards is silently inflated.

Two arguments turn your assumptions into checks:
- `validate="many_to_one"` (or `"one_to_one"`, `"one_to_many"`) raises `MergeError` when the data contradicts the cardinality you declared.
- `indicator=True` adds a `_merge` column (`both` / `left_only` / `right_only`) showing where each row came from.
""")
nb.md(r"""
🔮 **3.1** Look at `items_toy` and `orders_toy` above. How many rows will each join return?
""")
nb.learner("join-row-counts", r"""
predict("left-join-rows", ...)    # items_toy LEFT join orders_toy on order_id
predict("inner-join-rows", ...)   # items_toy INNER join orders_toy on order_id
""")
nb.code(r"""
left = items_toy.merge(orders_toy, on="order_id", how="left", validate="many_to_one")
inner = items_toy.merge(orders_toy, on="order_id", how="inner", validate="many_to_one")
print("items:", len(items_toy), "| left:", len(left), "| inner:", len(inner))
display(left[left["order_status"].isna()])
""")
nb.md(r"""
🔮 **3.2** Now the dangerous one. `payments_toy` can have several payments per order. How many rows does `items_toy.merge(payments_toy, on="order_id")` (an inner join) return?
""")
nb.code(r"""
payments_toy = toy_payments()
display(payments_toy)
""")
nb.learner("many-to-many-rows", r"""
predict("many-to-many-rows", ...)
""")
nb.code(r"""
mm = items_toy.merge(payments_toy, on="order_id")
print("rows:", len(mm))
print("sum of price: true", items_toy["price"].sum(), "| after the join", mm["price"].sum())

try:
    items_toy.merge(payments_toy, on="order_id", validate="many_to_one")
except pd.errors.MergeError as e:
    print("validate caught it →", e)
""")
nb.md(r"""
**Anti-joins.** An *anti-join* keeps the rows of $L$ that have **no** match in $R$. A *semi-join* keeps the rows of $L$ that **do** have one, without adding any of $R$'s columns. pandas has no dedicated function for either, but a left join with `indicator=True` does both: filter on `_merge`. For a single key column, `L[~L[key].isin(R[key])]` is a quick alternative.

✍️ **3.3** Using `indicator=True`, build:
- `orphan_items`: rows of `items_toy` whose order is missing from `orders_toy`
- `orders_without_items`: rows of `orders_toy` that have no items

Each result must have **exactly the columns of its original table**: no `_merge`, and nothing from the other table.
""")
nb.learner("anti-joins", r"""
orphan_items = ...
orders_without_items = ...

check("anti-joins", orphan_items, orders_without_items)
""")
nb.md(r"""
✍️ **3.4** On the real data: how many orders have no item rows, and what is their `order_status` distribution (`status_counts`, a Series)?
""")
nb.learner("orders-without-items", r"""
n_orders_without_items = ...
status_counts = ...

check("orders-without-items", n_orders_without_items, status_counts)
""")
nb.md(r"""
**Aggregate, then join.** To combine two tables that are both "many" with respect to the key (items and payments, both many per order), first collapse each one to the key's level with `groupby(key).agg(...)`. That gives one row per order on each side. The join is then one-to-one, and you can `validate` it.

★ **3.5** The correct way to compare what was bought with what was paid. Write `order_totals(items, payments)`, returning **one row per order that appears in either table**, with:
- `items_total` = sum of `price + freight_value`
- `paid_total` = sum of `payment_value`

Fill missing totals with 0. Then compute `share_mismatch`: the share of orders where the two totals differ by more than 1 BRL.
""")
nb.learner("order-totals", r"""
def order_totals(items: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    # your code here
    raise NotImplementedError


share_mismatch = ...

check("order-totals", order_totals, share_mismatch)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 4 · Lookup tables and coverage

A **lookup table** (or *dimension table*) maps a code to its attributes. Here, `categories` maps each Portuguese category name to its English translation. Joining one in is a many-to-one join. The question to ask is about **coverage**: does every code in the main table have an entry? Two different kinds of missing can show up after the join:
1. the code itself is missing (a product with no category at all), or
2. the code exists but the lookup table doesn't know it (a category with no translation).

**Coalescing** fills gaps from a sequence of fallbacks: the first non-missing value wins. In pandas that's a chain of `fillna`: `a.fillna(b).fillna("UNKNOWN")`.

> ⚠️ `merge` returns a result with a brand-new index `0..n-1`, whatever the left table's index was. Assigning a Series to a DataFrame column aligns by **index label**, not by position. So if you hand back a merged column, give it the original table's index first.
""")
nb.code(r"""
products_toy, categories_toy = toy_products(), toy_categories()
display(products_toy, categories_toy)
""")
nb.md(r"""
✍️ **4.1** Write `category_label(products, categories)`, which returns a Series aligned with `products` (same index). It falls back in this order: **English name → Portuguese name → `"UNKNOWN"`**.
""")
nb.learner("category-label", r"""
def category_label(products: pd.DataFrame, categories: pd.DataFrame) -> pd.Series:
    # your code here
    raise NotImplementedError


check("category-label", category_label)
""")
nb.md(r"""
✍️ **4.2** Which Portuguese category names in the real `products` have **no translation**? Give a sorted list (`untranslated`).
""")
nb.learner("untranslated", r"""
untranslated = ...

check("untranslated", untranslated)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 5 · Verify before you interpret: invariants

An **invariant** is something that *must* stay true if your pipeline is correct. For example:
- a many-to-one left join keeps the number of left rows
- the total `price` doesn't change when you add category labels
- a key is still unique after deduplication

Write each one down as code: `assert`, or a `validate=` argument. A broken assumption then stops the pipeline instead of quietly producing a wrong number. `assert condition, "message"` raises `AssertionError` with that message when the condition is false.
""")
nb.md(r"""
🐞 **5.1** A colleague computed revenue per category on the toy data and got **480** for `beleza_saude` (health & beauty). The correct value is 240. Run the cell, find the cause, and explain it below.
""")
nb.code(r"""
prod_cat = toy_products_dirty().assign(category=lambda d: d["product_category_name"].fillna("UNKNOWN"))
buggy = (toy_items()
         .merge(toy_orders().query("order_status == 'delivered'")[["order_id"]], on="order_id")
         .merge(prod_cat[["product_id", "category"]], on="product_id"))
buggy.groupby("category")["price"].sum()
""")
nb.md(r"""
💬 **5.2** What went wrong, and which single argument would have caught it?
""")
nb.answer("debug-duplicate-key")
nb.md(r"""
✍️ **5.3** Write `checked_left_join(left, right, on)`, a join you can trust:
- left join, `validate="many_to_one"`, `indicator=True`
- check that the row count equals `len(left)`
- **report** (print) how many left rows had no match, but don't fail on them
- return the result **without** `_merge`
""")
nb.learner("checked-left-join", r"""
def checked_left_join(left: pd.DataFrame, right: pd.DataFrame, on) -> pd.DataFrame:
    # your code here
    raise NotImplementedError


check("checked-left-join", checked_left_join)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 6 · Mini-project: an order-level table

Most questions in this module are about **orders**: how much they're worth, how fast they arrive, how they're reviewed. The detail tables hold many rows per order, so the standard move is to build an **analytic table** with exactly one row per order, by aggregating each detail table first.

✍️ **6.1** Write `build_order_table(orders, items, payments, customers)`. It returns one row per order, with the columns `order_id, order_status, customer_unique_id, customer_state, n_items, items_value, freight_value, paid_value`, in that order.
- `n_items`: number of item rows. Orders without items have `0`.
- `items_value`, `freight_value`: sums of `price` and `freight_value`.
- `paid_value`: sum of `payment_value`.
- Money columns stay **missing** (`NaN`) for orders without rows in that table. Unlike in 3.5, you aren't comparing two totals here, and "no information" isn't "zero BRL".
- Validate every join.
""")
nb.learner("order-table", r"""
def build_order_table(orders, items, payments, customers) -> pd.DataFrame:
    # your code here
    raise NotImplementedError


check("order-table", build_order_table)
""")
nb.code(r"""
order_table = build_order_table(orders, items, payments, customers)
assert len(order_table) == len(orders) and order_table["order_id"].is_unique
order_table.head()
""")
nb.md(r"""
★ **6.2** Among states with **at least 1,000 orders**, which one has the highest **median** `paid_value`? (`top_state`, a state code.)
""")
nb.learner("top-state-median", r"""
top_state = ...

check("top-state-median", top_state)
""")
nb.md(r"""
💬 **6.3** Drop the 1,000-order filter and the "winner" changes to a state with very few orders. Why do small groups tend to produce the most extreme summaries? Give an argument, not a computation.
""")
nb.answer("small-groups")

# ---------------------------------------------------------------------------
nb.md(r"""
## Wrap-up

💬 Answer briefly, in your own words.

1. What is the unit of observation of `items`? Of `order_table`? Of `customers`, if you count by `customer_unique_id`?
2. Give a join from this notebook where `how="inner"` would silently bias a result, and say which way.
3. Why is "aggregate first, then join" the fix for many-to-many?
4. Your colleague's key has the same values in both tables, but the join finds almost nothing. Name two causes.
5. Name three invariants you'd assert in any pipeline built on these tables.
""")
nb.answer("wrap-up", "> Your answers:")
nb.code(r"""
progress()
""")
nb.md(r"""
**Next:** `02 · Filter, group, derive`. If you use Claude Code with this repository, `/tutor review m01/01` reads your notebook and gives you feedback on your code and explanations.
""")
