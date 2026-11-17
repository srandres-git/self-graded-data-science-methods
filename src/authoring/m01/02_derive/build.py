"""Builder: Module I · 02 — Filter, group, derive."""
from nbbuild import Notebook

COURSE = "https://github.com/chaffrey8/DMEMCD_2026"

nb = Notebook(
    nb_id="m01/02",
    filename="02_filter-group-derive.ipynb",
    title="Filter, group, derive",
    subtitle="Boolean masks, groupby, dates, vectorized logic and functions that defend themselves",
    grader="selfgrade.m01.derive",
    can_do=["filter rows with masks without falling into index-alignment and missing-value traps",
            "summarize groups with agg and broadcast group values with transform",
            "derive durations and months from timestamps, keeping 'didn't happen' apart from 'zero'",
            "replace row loops with vectorized conditionals",
            "write functions that validate their inputs and never mutate them"],
    builds_on="Notebook 01: keys, joins with `validate=`, aggregate-then-join, lookup tables.",
    effort="≈ 2 h",
    sources=[f"Diplomado *Métodos Estadísticos y Matemáticos para la Ciencia de Datos* (UNAM, 2026), Module I: "
             f"[`S03_Notebook_Demostracion_Python`, `S03_Notebook_Laboratorio_Estudiante_Python`, "
             f"`01_Lectura_Integracion_Work_Olist`]({COURSE})"],
    concept_map=["boolean masks", "index alignment", "missing values in comparisons", "groupby: agg vs transform",
                 "crosstab", "dates & derived variables", "vectorization", "defensive functions",
                 "mini-project: delivery times by state"],
)

# ---------------------------------------------------------------------------
nb.md(r"""
## 0 · Setup and the analysis table

This notebook uses the same **Olist** e-commerce data as Notebook 01: about 100,000 orders placed in Brazil from 2016 to 2018, with amounts in Brazilian reais (BRL) and states as two-letter codes (SP is São Paulo). `load_olist(name)` returns a fresh copy of a table, with dates already parsed.

Besides the raw tables, several sections use an **analysis table**: one row per *item* of a *delivered* order, with the purchase timestamp and a readable product category. It's built with tools from Notebook 01: a lookup join with a fallback chain (English name → Portuguese name → `"UNKNOWN"`), and joins whose cardinality is validated.
""")
nb.code(r"""
import numpy as np
import pandas as pd
from IPython.display import display

from selfgrade.data import load_olist
from selfgrade.m01.derive import check, hint, predict, progress
from selfgrade.m01.derive import (toy_orders, toy_items, toy_customers, toy_products, toy_categories,
                                  toy_analysis, toy_deliveries)

orders, items = load_olist("orders"), load_olist("items")
products, categories, customers = load_olist("products"), load_olist("categories"), load_olist("customers")
""")
nb.code(r"""
# Category label for each product: English -> Portuguese -> "UNKNOWN"   (lookup + coalesce, Notebook 01)
labels = products[["product_id", "product_category_name"]].merge(
    categories, on="product_category_name", how="left", validate="many_to_one")
labels["category"] = (labels["product_category_name_english"]
                      .fillna(labels["product_category_name"]).fillna("UNKNOWN"))

# One row per item of a delivered order
delivered_orders = orders.loc[orders["order_status"].eq("delivered"), ["order_id", "order_purchase_timestamp"]]
analysis = (items.merge(delivered_orders, on="order_id", how="inner", validate="many_to_one")
                 .merge(labels[["product_id", "category"]], on="product_id", how="left", validate="many_to_one"))

assert analysis["category"].notna().all()
print(analysis.shape)
analysis.head(3)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 1 · Boolean masks: choosing rows

A **filter** keeps the rows that satisfy a condition. In set notation, $D_P = \{x \in D : P(x)\}$. In pandas the condition is a **boolean mask**: a Series of `True`/`False` values, one per row, and `df[mask]` (or `df.loc[mask]`) keeps the `True` rows. Masks come from comparisons and from a few methods:

| Expression | True where |
|---|---|
| `s.eq(v)`, `s == v` (also `ne`, `lt`, `le`, `gt`, `ge`) | the comparison holds |
| `s.isin([a, b])` | the value is one of a list |
| `s.between(a, b)` | $a \le s \le b$ (both ends included by default) |
| `s.isna()`, `s.notna()` | the value is missing / present |

They combine with `&` (and), `|` (or) and `~` (not). These are Python's *bitwise* operators, applied element by element. The words `and` / `or` don't work on Series: they ask Python for a *single* truth value for the whole column, and pandas refuses with "the truth value of a Series is ambiguous".

> ⚠️ **Precedence.** In Python, `&` and `|` bind *tighter* than comparisons like `>` and `==`, the opposite of `and`/`or`. Wrap every condition in parentheses.

> ✅ **Good practice:** give masks names (`is_delivered`, `in_2018`) and combine the names. The filter then reads like its definition, and each piece can be checked with `.sum()`.
""")
nb.md(r"""
🔮 **1.1** What does this line do?
```python
orders[orders.order_purchase_timestamp.dt.year > 2017 & orders.order_status == "delivered"]
```
**a)** returns the right rows  **b)** returns wrong rows silently  **c)** raises an error
""")
nb.learner("mask-precedence", r"""
predict("mask-precedence", ...)   # "a", "b" or "c"
""")
nb.code(r"""
# Readable style: name your masks
is_delivered = orders["order_status"].eq("delivered")
in_2018 = orders["order_purchase_timestamp"].dt.year.eq(2018)
print("delivered:", is_delivered.sum(), "| purchased in 2018:", in_2018.sum(), "| both:", (is_delivered & in_2018).sum())
""")
nb.md(r"""
✍️ **1.2** How many orders are **delivered**, were **purchased in 2017**, and have a **non-missing customer delivery date**?
""")
nb.learner("delivered-2017", r"""
n_delivered_2017 = ...

check("delivered-2017", n_delivered_2017)
""")
nb.md(r"""
### Index alignment: a mask belongs to its table

Every Series and DataFrame carries an **index**: a label for each row (by default $0, 1, \dots, n-1$). When you write `df[mask]`, pandas doesn't match the mask to `df` by *position*. It **aligns them by label**: the row labelled 7 in `df` is kept if the mask's entry labelled 7 is `True`. This makes operations between Series safe when rows have been reordered or filtered, but it has a sharp edge when a mask comes from a *different* table:

| Mask labels vs. table labels | What happens |
|---|---|
| the same labels, the same rows | correct |
| the same labels, **different rows** behind them | **silently wrong**: rows are picked by label, not by meaning |
| the mask lacks some of the table's labels | `IndexingError: Unalignable boolean Series` |

Operations that hand out a fresh index $0..n-1$ include `merge`, `reset_index`, `read_csv`, and `concat(..., ignore_index=True)`. Filtering and sorting keep the original labels. So a mask built on a merged table and applied to one of the inputs works only by coincidence.
""")
nb.md(r"""
🐞 **1.3** A colleague wants the toy orders placed by customers from São Paulo (SP). The state lives in `customers_toy`, so they merged and built a mask. Run it, then compare with the tables: which orders *should* come back?
""")
nb.code(r"""
orders_toy, customers_toy = toy_orders(), toy_customers()
display(orders_toy, customers_toy)

with_state = customers_toy.merge(orders_toy, on="customer_id", validate="one_to_one")
is_sp = with_state["customer_state"].eq("SP")
orders_toy[is_sp]      # "orders from SP"?
""")
nb.md(r"""
💬 **1.4** Which orders came back, and which should have? Explain what pandas did, and give two correct ways to write this filter.
""")
nb.answer("debug-alignment")
nb.md(r"""
✍️ **1.5** On the real data: build `delivered_sp`, the **delivered** orders whose customer is in **SP**. It should have exactly the columns of `orders`, one row per order.
""")
nb.learner("delivered-sp", r"""
delivered_sp = ...

check("delivered-sp", delivered_sp)
""")
nb.md(r"""
### Missing values in comparisons

Missing values follow a rule inherited from floating-point arithmetic (the IEEE 754 standard): **every comparison involving `NaN` is `False`**, even `NaN == NaN`. pandas applies the same rule to `NaT`, the missing datetime. A comparison with a missing value therefore doesn't answer "unknown"; it quietly answers "no". That matters for negations, because the negation of "no" is "yes".

🔮 **1.6** `d` is the delivery date and `a` the approval date of each order, and both can be missing. True or False: `~(d < a)` and `d >= a` select the same orders?
""")
nb.learner("nat-complement", r"""
predict("nat-complement", ...)   # True or False
""")
nb.code(r"""
d, a = orders["order_delivered_customer_date"], orders["order_approved_at"]
print("~(d < a):", (~(d < a)).sum(), "| d >= a:", (d >= a).sum(), "| rows with d or a missing:", (d.isna() | a.isna()).sum())
""")
nb.md(r"""
★ **1.7** *Data-quality audit.* How many orders reached the customer **before** the payment was approved (`order_delivered_customer_date < order_approved_at`)? Think about what happens with missing dates, and whether you need to do anything about them.
""")
nb.learner("delivered-before-approved", r"""
n_delivered_before_approved = ...

check("delivered-before-approved", n_delivered_before_approved)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 2 · Grouping: split → apply → combine

`groupby` follows a pattern known as **split-apply-combine**: it *splits* the rows into groups by the values of one or more key columns, *applies* a function to each group, and *combines* the results. Ask yourself two things every time:
1. **What is the unit of observation of the result?** Usually one row per group.
2. **Do I want to collapse each group to one value (`agg`), or broadcast a group value back to every row (`transform`)?**

Three counting functions look alike but answer different questions:

| Function | Counts |
|---|---|
| `size` | rows, missing values included |
| `count` | non-missing values |
| `nunique` | distinct non-missing values |

**Named aggregation** keeps output columns explicit: `df.groupby("key").agg(new_name=("column", "function"), ...)`.

> 🧰 **How the code works:** the group keys become the result's index (use `.reset_index()` or `as_index=False` to turn them back into columns). Rows whose key is missing are **dropped** by default. Pass `dropna=False` to keep them as their own group.
""")
nb.code(r"""
products_toy = toy_products()
display(products_toy)
""")
nb.md(r"""
🔮 **2.1** How many groups does `products_toy.groupby("product_category_name").size()` return?
""")
nb.learner("groupby-drops-nan", r"""
predict("groupby-drops-nan", ...)
""")
nb.code(r"""
print(products_toy.groupby("product_category_name").size(), "\n")
print(products_toy.groupby("product_category_name", dropna=False).size())
""")
nb.code(r"""
g = pd.DataFrame({"grp": ["a", "a", "a"], "price": [10, 10, np.nan]})
g
""")
nb.md(r"""
🔮 **2.2** What are `(size, count, nunique)` of `price` for group `"a"`? Answer as a tuple.
""")
nb.learner("size-count-nunique", r"""
predict("size-count-nunique", ...)
""")
nb.code(r"""
g.groupby("grp")["price"].agg(["size", "count", "nunique"])
""")
nb.md(r"""
✍️ **2.3** Write `category_summary(df)` for a table with columns `order_id`, `category`, `price`. It returns one row per category, with columns **in this order**: `rows`, `distinct_orders`, `price_sum`, `price_median`, sorted by `price_sum` descending. The check tests it on `toy_analysis()` and on the real `analysis`.
""")
nb.learner("category-summary", r"""
def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    # your code here
    raise NotImplementedError


check("category-summary", category_summary)
""")
nb.code(r"""
display(toy_analysis())
""")
nb.md(r"""
✍️ **2.4** *Median of what?* Using `analysis`, compute:
- `median_item_price`: the median price per **item row**
- `median_order_value`: the median of the **order totals** (the sum of item prices per order)
""")
nb.learner("two-medians", r"""
median_item_price = ...
median_order_value = ...

check("two-medians", median_item_price, median_order_value)
""")
nb.md(r"""
**`transform`** runs a group function but returns a result **the same length as the input**, aligned row by row: every row receives its own group's value. Typical uses are a share of the group total, a deviation from the group mean, or a rank within the group. Compare:
```python
df.groupby("order_id")["price"].agg("sum")        # one row per order
df.groupby("order_id")["price"].transform("sum")  # one row per item, carrying its order's total
```

★ **2.5** Write `add_share_of_order(df)`, which returns a copy of `df` with a new column `share_of_order` = item price ÷ total price of its order.

Then, on `analysis`, compute `share_dominated`: among orders with **at least 2 item rows**, the share where a single item is **more than half** (`> 0.5`) of the order's price.
""")
nb.learner("share-of-order", r"""
def add_share_of_order(df: pd.DataFrame) -> pd.DataFrame:
    # your code here
    raise NotImplementedError


share_dominated = ...

check("share-of-order", add_share_of_order, share_dominated)
""")
nb.md(r"""
★ **2.6** Which category had the highest delivered-item revenue (sum of `price`) for purchases made in **November 2017**?
""")
nb.learner("top-category-nov-2017", r"""
top_category_2017_11 = ...

check("top-category-nov-2017", top_category_2017_11)
""")
nb.md(r"""
**Cross-tabulation.** For two categorical variables, a **contingency table** counts how many rows fall in each combination: $n_{ij}$ rows have row value $i$ and column value $j$. `pd.crosstab(rows, cols)` builds it. `normalize=` picks the denominator, and with it the question the table answers:

| `normalize=` | cell value | answers |
|---|---|---|
| `"all"` | $n_{ij} / n$ | what share of *all* rows is this combination? |
| `"index"` | $n_{ij} / n_{i\cdot}$, where $n_{i\cdot}=\sum_j n_{ij}$ | *within row* $i$, what share has column value $j$? (each row sums to 1) |
| `"columns"` | $n_{ij} / n_{\cdot j}$ | *within column* $j$, what share has row value $i$? (each column sums to 1) |

The `"index"` version is the **conditional distribution** of the column variable given the row variable, an idea that comes back throughout the course.

✍️ **2.7** Build `status_share`: for each customer state (rows), the share of its orders in each `order_status` (columns). Each row must sum to 1.
""")
nb.learner("status-share", r"""
status_share = ...

check("status-share", status_share)
""")
nb.code(r"""
counts = orders.merge(customers[["customer_id", "customer_state"]], on="customer_id")["customer_state"].value_counts()
try:
    print(status_share["canceled"].sort_values(ascending=False).head(3).round(4).to_frame()
          .join(counts.rename("orders")))
except (TypeError, KeyError, AttributeError):
    print("(complete 2.7 to see the states with the highest canceled share, and how many orders they have)")
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 3 · Dates and derived variables

pandas has three time types:

| Type | Is | Example |
|---|---|---|
| `Timestamp` (column dtype `datetime64`) | a point in time | 2018-03-01 10:00 |
| `Timedelta` | a duration | 3 days 12:00:00 |
| `Period` | a span of calendar time | the month 2018-03 |

Their arithmetic follows the meaning: Timestamp − Timestamp = Timedelta, and Timestamp + Timedelta = Timestamp. Column-wise tools live behind the `.dt` accessor: `.dt.year`, `.dt.month`, `.dt.dayofweek` (Monday = 0), `.dt.normalize()` (set the time to midnight), and `.dt.to_period("M")` (the month a timestamp falls in).

**Structural missingness.** Some values are missing because the event *didn't happen*, not because someone forgot to record them. A canceled order has no delivery duration. Replacing that `NaN` with 0 would invent an instant delivery and pull every average down. The population a variable is defined on is part of its definition: in this data, 8 delivered orders have no delivery date, and 6 canceled orders do carry one.

> 🧰 **How the code works:** `Series.where(cond)` keeps values where `cond` is True and puts `NaN` elsewhere. It's the vectorized way to say "only defined for these rows".
""")
nb.md(r"""
🔮 **3.1** What does this return?
```python
(pd.Timestamp("2018-01-02 06:00") - pd.Timestamp("2018-01-01 18:00")).days
```
""")
nb.learner("timedelta-days", r"""
predict("timedelta-days", ...)
""")
nb.code(r"""
delta = pd.Timestamp("2018-01-02 06:00") - pd.Timestamp("2018-01-01 18:00")
print(repr(delta), "| .days =", delta.days, "| total_seconds()/86400 =", delta.total_seconds() / 86400)
""")
nb.md(r"""
✍️ **3.2** Write `add_delivery_days(orders)`. It returns a copy of `orders` with a new column `delivery_days`: the time from purchase to delivery to the customer, in **fractional days**, defined **only for orders with status `delivered`**. Every other order, and any delivered order without a delivery date, gets `NaN`. Keep the input's index. The toy table covers each case:
""")
nb.code(r"""
display(toy_deliveries())
""")
nb.learner("delivery-days", r"""
def add_delivery_days(orders: pd.DataFrame) -> pd.DataFrame:
    # your code here
    raise NotImplementedError


check("delivery-days", add_delivery_days)
""")
nb.code(r"""
days = add_delivery_days(orders)["delivery_days"]
print(f"orders with a duration: {days.notna().sum():,} of {len(days):,}")
days.describe().round(2)
""")
nb.md(r"""
💬 **3.3** Suppose you had filled the missing durations with 0. Which summaries would change, in which direction, and why? Is there any question for which "missing → 0" *would* be right?
""")
nb.answer("zero-vs-missing")

# ---------------------------------------------------------------------------
nb.md(r"""
## 4 · Vectorization: think in columns, not loops

A Python `for` loop over rows runs the interpreter once per element: it looks up types, dispatches the operation, and boxes the result into a Python object, every single time. **Vectorized** operations (arithmetic on whole columns, `np.where`, `np.select`, `pd.cut`) hand the whole array to compiled code that loops over raw numbers in memory. That's typically 10–1000× faster, and usually more readable too.

- `np.where(cond, a, b)`: two branches
- `np.select([cond1, cond2, ...], [val1, val2, ...], default=...)`: many branches. For each element, the **first** true condition wins.
- `pd.cut(x, bins, right=False, labels=...)`: interval binning. `right=False` makes the intervals $[a, b)$, closed on the left. It returns an **ordered Categorical**: a column with a fixed, ordered set of labels, so sorting and plotting follow the bin order, not the alphabet.

Price bands for the next exercises: $[0, 50)$ → `"<50"`, $[50, 100)$ → `"50-100"`, $[100, 200)$ → `"100-200"`, $[200, \infty)$ → `"200+"`.

✍️ **4.1** Implement the bands with `np.select`.
""")
nb.learner("price-band-select", r"""
def price_band_select(prices: pd.Series):
    # your code here
    raise NotImplementedError


check("price-band-select", price_band_select)
""")
nb.md(r"""
✍️ **4.2** Implement them with `pd.cut`.
""")
nb.learner("price-band-cut", r"""
def price_band_cut(prices: pd.Series):
    # your code here
    raise NotImplementedError


check("price-band-cut", price_band_cut)
""")
nb.md(r"""
★ **4.3** Write `price_band_loop` with a plain `for` loop and `if/elif`. Confirm it agrees with your vectorized version and measure the speed-up. (There's no automatic check: the timing is the result.)
""")
nb.learner("price-band-loop", r"""
import time


def price_band_loop(prices):
    # your code here
    raise NotImplementedError


def timed(fn, x):
    t0 = time.perf_counter()
    out = fn(x)
    return out, time.perf_counter() - t0


p = analysis["price"]
out_loop, t_loop = timed(price_band_loop, p)
out_sel, t_sel = timed(price_band_select, p)
print("agree:", list(out_loop) == list(out_sel))
print(f"loop {t_loop*1000:.1f} ms | np.select {t_sel*1000:.1f} ms | speed-up ×{t_loop / t_sel:.0f}")
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 5 · Functions that defend themselves

A good analysis function:
- has **explicit parameters** with sensible defaults, and a docstring
- **validates its inputs at the boundary**, raising a *specific* exception with a *clear* message:
  - `TypeError` for the wrong type
  - `ValueError` for the right type but an invalid value
  - `KeyError` for missing columns
- **doesn't modify its inputs**. Use `.assign` or `.copy()`, because callers don't expect side effects.

> 🧰 **How the code works:** `raise ValueError(f"n must be >= 1, got {n}")` stops the function and reports the problem to the caller. `isinstance(x, T)` tests a type. `numbers.Integral` is an *abstract base class* that both Python `int` and NumPy integer types register with, so `isinstance(np.int64(3), numbers.Integral)` is True.

✍️ **5.1** Write `top_categories(df, n=5)`, which returns a Series `category → price_sum` with the top `n`, largest first. Requirements:
- `n` must be an integer, otherwise raise `TypeError`
- `n >= 1`, otherwise raise `ValueError`
- if `category` or `price` is missing, raise `KeyError`
- if `n` exceeds the number of categories, return all of them without failing
""")
nb.learner("top-categories", r"""
def top_categories(df: pd.DataFrame, n: int = 5) -> pd.Series:
    # your code here
    raise NotImplementedError


check("top-categories", top_categories)
""")
nb.md(r"""
★ **5.2** Does your function accept `n=True`? Should it? And what about `np.int64(2)`, which is the type you get when `n` comes out of a pandas computation? Adjust it until both behave sensibly.
""")
nb.learner("bool-trap", r"""
check("bool-trap", top_categories)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 6 · Mini-project: how long does delivery take, and where?

Brazil is large, and most sellers are in the southeast. Here you combine a derived variable (section 3), a validated join (Notebook 01), and both kinds of grouping (section 2). This table has one row per delivered order with a known duration, plus the customer's state:
""")
nb.code(r"""
deliveries = (add_delivery_days(orders)
             .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left", validate="one_to_one")
             .dropna(subset=["delivery_days"]))
print(f"{len(deliveries):,} delivered orders with a duration")
deliveries[["order_id", "customer_state", "delivery_days"]].head()
""")
nb.code(r"""
# The same move as Notebook 01's top state: summarize per group, then keep groups big enough to trust.
by_state = deliveries.groupby("customer_state")["delivery_days"].agg(["size", "median"])
by_state[by_state["size"] >= 1000].sort_values("median", ascending=False).head()
""")
nb.md(r"""
✍️ **6.1** "Slow" depends on where you live. Compute `share_slow`: the share of orders in `deliveries` whose duration is **more than twice the median of their own state**.
""")
nb.learner("slow-vs-state", r"""
share_slow = ...

check("slow-vs-state", share_slow)
""")
nb.md(r"""
💬 **6.2** Does a long median delivery time in a state mean its sellers or carriers are worse? Name at least two other explanations you could check with the tables you have.
""")
nb.answer("delivery-interpretation")

# ---------------------------------------------------------------------------
nb.md(r"""
## Wrap-up

💬 Answer briefly, in your own words.

1. A mask built on table A is used to filter table B. When does that work, when does it fail loudly, and when does it silently return wrong rows?
2. Why isn't `~(d < a)` the same as `d >= a`? Write the condition you'd use for "delivered on or after approval, both dates known".
3. Give one example each of `agg` and `transform` from this notebook, and say what the unit of observation of each result is.
4. Why does a canceled order get no delivery duration, even when a delivery date is recorded?
5. Name three things a function should check or guarantee before you trust it inside a pipeline.
""")
nb.answer("wrap-up", "> Your answers:")
nb.code(r"""
progress()
""")
nb.md(r"""
**Next:** `03 · Describing a distribution`. If you use Claude Code with this repository, `/tutor review m01/02` reads your notebook and gives you feedback on your code and explanations.
""")
