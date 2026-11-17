"""Builder: Module I · 05 — Honest charts."""
from nbbuild import Notebook

COURSE = "https://github.com/chaffrey8/DMEMCD_2026"

nb = Notebook(
    nb_id="m01/05",
    filename="05_honest-charts.ipynb",
    title="Honest charts",
    subtitle="How people read charts, and how to draw ones that tell the truth",
    grader="selfgrade.m01.charts",
    can_do=["build charts with matplotlib's Figure/Axes objects, as reusable functions",
            "choose an encoding people read accurately, and know when a bar must start at 0",
            "show parts of a whole (stacked, 100 %, dot plot, mosaic) with the right denominator",
            "pick sequential, diverging or qualitative colors, and check a colormap for honesty",
            "label, annotate and export a chart at an exact size"],
    builds_on="01 (the order-level table, joins) and 02 (groupby, `crosstab`): both are used here as tools. "
              "Distributions (histograms, box plots) are in 03, relationships between variables in 04.",
    effort="≈ 2 h",
    sources=[f"Diplomado *Métodos Estadísticos y Matemáticos para la Ciencia de Datos* (UNAM, 2026), Module I: "
             f"[`03_Visualizaciones_Work_Olist`]({COURSE}), sections A (categorical variables), "
             f"G (mosaic, day × hour heatmap) and H (design and export)",
             "W. S. Cleveland & R. McGill (1984), *Graphical Perception*, JASA 79(387); "
             "C. O. Wilke, [*Fundamentals of Data Visualization*](https://clauswilke.com/dataviz/)"],
    concept_map=["Figures and Axes", "how we read charts", "parts of a whole", "color",
                 "labels and export", "mini-project: when do customers shop?"],
)

# ---------------------------------------------------------------------------
nb.md(r"""
## 0 · Setup

The data is the Olist marketplace dataset introduced in Notebook 01: about 100,000 orders placed in Brazil from 2016 to 2018, amounts in Brazilian reais (BRL), states as two-letter codes (SP is São Paulo, RJ Rio de Janeiro, and so on). The setup builds two tables with the tools from Notebooks 01 and 02:

- `order_table`: one row per order, with status, customer state, purchase time, item value, freight, the first payment method, and delivery timing. An order is **late** when it was delivered on a later calendar day than the estimated delivery date. `is_late` is missing for orders that were never delivered, since lateness isn't defined for them.
- `item_table`: one row per item sold, with its price and an English category label.
""")
nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display

from selfgrade.data import load_olist
from selfgrade.m01.charts import check, hint, predict, progress
from selfgrade.m01.charts import (toy_counts, toy_item_revenue, toy_late_orders, toy_crosstab,
                                  toy_corr, toy_timestamps)

orders, items, payments = load_olist("orders"), load_olist("items"), load_olist("payments")
customers, products, categories = load_olist("customers"), load_olist("products"), load_olist("categories")

# One row per order (Notebook 01: aggregate the detail tables first, then join, validating every join)
first_payment = (payments.sort_values(["order_id", "payment_sequential"]).drop_duplicates("order_id")
                 [["order_id", "payment_type"]].rename(columns={"payment_type": "first_payment_type"}))
per_order = items.groupby("order_id").agg(n_items=("price", "size"), items_value=("price", "sum"),
                                          freight_value=("freight_value", "sum"))
order_table = (orders
               .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left", validate="many_to_one")
               .merge(per_order, on="order_id", how="left", validate="one_to_one")
               .merge(first_payment, on="order_id", how="left", validate="one_to_one"))

delivered = order_table["order_status"].eq("delivered") & order_table["order_delivered_customer_date"].notna()
order_table["delivery_days"] = ((order_table["order_delivered_customer_date"] - order_table["order_purchase_timestamp"])
                                .dt.total_seconds() / 86400).where(delivered)
order_table["delay_days"] = (order_table["order_delivered_customer_date"].dt.normalize()
                             - order_table["order_estimated_delivery_date"]).dt.days.where(delivered)
order_table["is_late"] = (order_table["delay_days"] > 0).astype("boolean").where(delivered)

# One row per item, with an English category label (Notebook 01: English → Portuguese → "UNKNOWN")
labels = products[["product_id", "product_category_name"]].merge(
    categories, on="product_category_name", how="left", validate="many_to_one")
labels["category"] = (labels["product_category_name_english"]
                      .fillna(labels["product_category_name"]).fillna("UNKNOWN"))
item_table = items.merge(labels[["product_id", "category"]], on="product_id", how="left", validate="many_to_one")

print(f"order_table {order_table.shape} | item_table {item_table.shape}")
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 1 · Figures and Axes

matplotlib draws on two kinds of objects:

- a **Figure** is the whole canvas: the image you'd save to a file;
- an **Axes** is one plotting area inside it, with its own x and y axis, ticks, labels and title. A Figure can hold several. Careful with the name: *Axes* (one panel) isn't the plural of *axis* (one of its two rulers).

`fig, ax = plt.subplots()` creates a Figure with one Axes. `fig, axes = plt.subplots(nrows, ncols)` creates a grid of them. You then call **methods on the Axes**: `ax.bar(...)`, `ax.set_xlabel(...)`, `ax.set_title(...)`. This is the *object-oriented* (OO) interface.

matplotlib also has a shortcut interface: `plt.bar(...)`, `plt.title(...)`. These functions act on whatever Axes is "current", a hidden global state. That's handy for a one-off plot, but it gets confusing as soon as there are several panels, or when the plotting happens inside a function. pandas plots follow the same pattern: `series.plot.bar(ax=ax)` draws on the Axes you pass in and returns it.

> 🧰 **How the code works:** every chart exercise in this notebook is a function with an `ax=None` parameter. If no Axes is given, it creates one (`fig, ax = plt.subplots(...)`). Either way it draws on `ax` and **returns `ax`**. `check()` then inspects that object (bar heights, their order, labels, axis limits, colormap) instead of looking at pixels.

> ✅ **Good practice:** write each chart as a function of the data. You can then re-run it on a filtered table, or place it in any panel of a bigger figure.
""")
nb.md(r"""
🔮 **1.1** `fig, axes = plt.subplots(2, 3)`. What is `axes.shape`? Answer as a tuple.
""")
nb.learner("axes-shape", r"""
predict("axes-shape", ...)
""")
nb.code(r"""
fig, axes = plt.subplots(2, 3, figsize=(9, 4))
print(type(axes).__name__, axes.shape)
for i, ax in enumerate(axes.flat):
    ax.set_title(f"axes.flat[{i}]", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
axes[0, 2].bar(["a", "b"], [3, 5])
fig.suptitle("One Figure, six Axes")
plt.show()
""")
nb.md(r"""
✍️ **1.2** Write `sorted_bars(counts, ylabel, title, ax=None)`. It draws a vertical bar chart of a Series of counts, **sorted from largest to smallest**, with the given y-axis label and title, and returns the Axes. If an `ax` is passed in, it must draw on that Axes.
""")
nb.learner("sorted-bars", r"""
def sorted_bars(counts: pd.Series, ylabel: str, title: str, ax=None):
    # your code here
    raise NotImplementedError


check("sorted-bars", sorted_bars)
""")
nb.code(r"""
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.5))
sorted_bars(toy_counts(), "Orders", "Toy data", ax=left)
sorted_bars(order_table["order_status"].value_counts(), "Orders", "Real data: orders by status", ax=right)
right.tick_params(axis="x", rotation=30)
plt.show()
""")
nb.md(r"""
On the real data, `delivered` (96,478 of 99,441 orders) dwarfs every other status. That's the honest picture: the rare statuses really are rare. If their exact sizes matter, give a table next to the chart; don't stretch the axis to make them look bigger.
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 2 · How we read charts

A chart **encodes** each number as a visual property: a position, a length, an angle, an area, a color. Readers then *decode* it back into a number, more or less accurately. Cleveland and McGill measured that accuracy in a series of experiments (1984). From most to least accurate:

| Rank | Encoding | Typical chart |
|---|---|---|
| 1 | position along a common scale | dot plot, scatter plot, aligned bars |
| 2 | position along scales that aren't aligned | the same quantity across separate panels |
| 3 | length, direction, angle | bars (length), pie charts (angle) |
| 4 | area | bubbles, treemaps, pie slices |
| 5 | volume, curvature | 3-D charts |
| 6 | shading, color saturation | heatmaps, choropleth maps |

The practical rule is to **put the comparison that matters on position**. Use the lower encodings only when position is already taken.

**Bars are lengths, so they start at 0.** A bar's whole length is its value. If the axis starts at 90, a bar of 95 is drawn 5 units long, and a bar of 100 is drawn 10 units long. A line chart is read by its **position and slope**, not its length, so its axis may start elsewhere, as long as the axis labels make that clear.

**Bar charts aren't histograms.** A bar chart shows **categories**, which are separate things and have gaps between bars. A histogram shows **intervals of a continuous variable**: the bins touch, and their order is fixed by the number line (Notebook 03).

**Order the categories.** For a *nominal* variable (categories with no natural order, like payment method or state), sort the bars by value: the chart becomes a ranking. For an *ordinal* variable (categories with an order, like review score 1–5 or a value band), or for time, keep the natural order.

**Long labels → horizontal bars.** Labels then read left to right, instead of being rotated or overlapping.
""")
nb.md(r"""
🔮 **2.1** Two bars with values **95** and **100** are drawn on a y axis that starts at **90**. How many times taller does the second bar *look*?
""")
nb.learner("truncated-ratio", r"""
predict("truncated-ratio", ...)
""")
nb.code(r"""
fig, axes = plt.subplots(1, 3, figsize=(11, 3))
for ax, bottom in zip(axes[:2], (90, 0)):
    ax.bar(["A", "B"], [95, 100], color="#4C72B0")
    ax.set_ylim(bottom, 101)
    ax.set_title(f"bars, axis from {bottom}")
axes[2].plot([2015, 2016, 2017, 2018], [95, 97, 96, 100], marker="o")
axes[2].set_title("a line may start elsewhere")
plt.tight_layout(); plt.show()
""")
nb.md(r"""
✍️ **2.2** Write `top_categories_barh(item_table, n=10, ax=None)`. It draws the `n` categories with the highest **revenue** (sum of `price`) as horizontal bars, with the **largest at the top**, an x axis that starts at 0, and an x label that names the unit (BRL).

> ⚠️ `ax.barh(labels, values)` puts the **first** value at the **bottom**.
""")
nb.learner("sorted-barh", r"""
def top_categories_barh(item_table: pd.DataFrame, n: int = 10, ax=None):
    # your code here
    raise NotImplementedError


check("sorted-barh", top_categories_barh)
""")
nb.code(r"""
top_categories_barh(item_table)
plt.show()
""")
nb.md(r"""
🐞 **2.3** A colleague wants to show **where late delivery is the biggest problem**. Here's their chart:
""")
nb.code(r"""
late_counts = order_table.groupby("customer_state")["is_late"].sum()
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.bar(late_counts.index, late_counts.to_numpy(), color="#C44E52")
ax.set_ylim(late_counts.median(), late_counts.max() * 1.05)
ax.set_title("Late deliveries by state")
plt.show()
""")
nb.md(r"""
It runs without errors, but it's misleading in at least three ways. Before you fix it, list them for yourself:
- what is each bar's **denominator**? Does the chart answer "where is lateness a *bigger* problem", or just "where are there more orders"?
- where does the axis start?
- in what order are the states?

✍️ **2.4** Write `late_share_chart(order_table, min_orders=1000, ax=None)`, which fixes it. Each bar is a state's **share of delivered orders that arrived late** (shares or percentages are both fine). Keep only states with at least `min_orders` delivered orders, since very small groups give noisy shares. Sort the bars from highest to lowest share, start them at 0, and label the y axis.
""")
nb.learner("fixed-state-chart", r"""
def late_share_chart(order_table: pd.DataFrame, min_orders: int = 1000, ax=None):
    # your code here
    raise NotImplementedError


check("fixed-state-chart", late_share_chart)
""")
nb.code(r"""
late_share_chart(order_table)
plt.show()
""")
nb.md(r"""
💬 **2.5** SP had the tallest bar in the buggy chart. Where does it land now, and why did the two charts disagree? In one sentence, state the rule you'd give your colleague.
""")
nb.answer("fixed-state-why")

# ---------------------------------------------------------------------------
nb.md(r"""
## 3 · Parts of a whole

Some charts split each group into parts: orders in a value band, split by payment method. The question decides the chart.

| Chart | Each part's length is | Good for | Weak at |
|---|---|---|---|
| **grouped bars** (side by side) | its count | comparing one part across groups, since every part starts at 0 | seeing totals |
| **stacked bars** | its count, piled up | totals, plus the bottom part | comparing the middle parts |
| **100 % stacked bars** | its share *within* the group | the mix of each group | group sizes, which are hidden |

A **share within a group** has a denominator: the group's total. Say so on the axis ("share of orders in the band"), and give each group's $n$ (its number of observations) somewhere, because a 100 % bar built on 12 orders looks exactly like one built on 12,000.

**Pie charts** encode shares as angles and areas, ranks 3–4 in the table above. Two slices of 24 % and 26 % are nearly impossible to tell apart. A pie can work for two or three parts with very different sizes. Otherwise a **dot plot** (one dot per category on a common scale, sorted), or its variant the **lollipop** (dots on sticks from 0), shows the same numbers as positions, the most accurate encoding.
""")
nb.code(r"""
ct = toy_crosstab()
display(ct)
shares = ct.div(ct.sum(axis=1), axis=0)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
ct.plot.bar(ax=axes[0], rot=0, title="grouped: counts", legend=False)
ct.plot.bar(stacked=True, ax=axes[1], rot=0, title="stacked: counts", legend=False)
shares.plot.bar(stacked=True, ax=axes[2], rot=0, title="100 % stacked: shares within the band")
axes[2].legend(bbox_to_anchor=(1, 1))
plt.tight_layout(); plt.show()
""")
nb.md(r"""
🔮 **3.1** In a stacked bar chart of counts, which segments can you compare accurately *across* bars?
**a)** only the bottom segments **b)** all segments **c)** only the top segments
""")
nb.learner("stacked-baseline", r"""
predict("stacked-baseline", ...)   # "a", "b" or "c"
""")
nb.md(r"""
✍️ **3.2** Write `composition_chart(counts, ax=None)`. `counts` is a table of counts, rows = groups and columns = parts (what `pd.crosstab` returns). Draw **100 % stacked bars**: one bar per row, each segment the column's share *within that row*. Fix the y axis to (0, 1), label it, and keep the legend.
""")
nb.learner("composition-chart", r"""
def composition_chart(counts: pd.DataFrame, ax=None):
    # your code here
    raise NotImplementedError


check("composition-chart", composition_chart)
""")
nb.code(r"""
value_band = pd.cut(order_table["items_value"], [0, 50, 100, 200, 500, np.inf], right=False,
                    labels=["<50", "50–100", "100–200", "200–500", "500+"])
pay_by_band = pd.crosstab(value_band, order_table["first_payment_type"])
display(pay_by_band)
ax = composition_chart(pay_by_band)
ax.set_xlabel("Items value of the order (BRL)")
plt.show()
""")
nb.md(r"""
Credit cards dominate in every band, and their share grows with the order's value: from 72 % of orders under 50 BRL to 83 % above 500 BRL. *Boleto*, a printed bank slip paid in cash, goes the other way, from 23 % to 16 %.

✍️ **3.3** Write `dot_plot(shares, ax=None)`. Given a Series of shares, draw one dot per category, with the categories on the y axis, **sorted so the largest is at the top**. Sticks from 0 (a lollipop) are optional. Label the x axis.
""")
nb.learner("dot-plot", r"""
def dot_plot(shares: pd.Series, ax=None):
    # your code here
    raise NotImplementedError


check("dot-plot", dot_plot)
""")
nb.code(r"""
pay_shares = order_table["first_payment_type"].value_counts(normalize=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.2))
ax1.pie(pay_shares.to_numpy(), labels=pay_shares.index, autopct="%1.0f%%")
ax1.set_title("pie: angles and areas")
dot_plot(pay_shares, ax=ax2)
ax2.set_title("dot plot: positions on a common scale")
plt.tight_layout(); plt.show()
""")
nb.md(r"""
**Mosaic plots.** A mosaic shows how two categorical variables combine, using areas. Write $n_{ij}$ for the count in row $i$ and column $j$, $n_{i\cdot}$ for row $i$'s total, and $n$ for the grand total. Three kinds of share appear:

- the **marginal** share of row $i$: $P(i) = n_{i\cdot}/n$
- the **conditional** share of column $j$ *given* row $i$: $P(j \mid i) = n_{ij}/n_{i\cdot}$
- the **joint** share of the cell: $P(i, j) = n_{ij}/n$

Each row becomes a vertical strip whose **width** is $P(i)$. The strip is then cut into tiles whose **heights** are $P(j \mid i)$. If the two variables were *independent*, meaning the mix of columns were the same in every row, all the strips would be cut at the same heights.

🔮 **3.4** A tile's **area** equals:
**a)** the joint share $n_{ij}/n$ **b)** the conditional share $n_{ij}/n_{i\cdot}$ **c)** the count $n_{ij}$
""")
nb.learner("mosaic-area", r"""
predict("mosaic-area", ...)   # "a", "b" or "c"
""")
nb.md(r"""
★ **3.5** Write `mosaic(counts, ax=None)`, a mosaic plot drawn with `matplotlib.patches.Rectangle`. Columns are the table's rows, with widths $P(i)$, and tiles have heights $P(j \mid i)$. The tiles fill the unit square with no gaps. Label the strips with the row names and add a legend for the columns.
""")
nb.learner("mosaic", r"""
from matplotlib.patches import Rectangle


def mosaic(counts: pd.DataFrame, ax=None):
    # your code here
    raise NotImplementedError


check("mosaic", mosaic)
""")
nb.code(r"""
ax = mosaic(pay_by_band)
ax.set_xlabel("Items value of the order (BRL); width = share of orders")
plt.show()
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 4 · Color

A **colormap** turns numbers into colors. There are three families, matched to three kinds of data:

| Family | Looks like | Use it for | Examples |
|---|---|---|---|
| **sequential** | one hue, going steadily from light to dark | quantities that go from low to high: counts, prices | `viridis`, `cividis`, `Blues` |
| **diverging** | two hues meeting at a neutral middle | quantities with a meaningful midpoint and two directions: a correlation (around 0), a difference from an average | `RdBu_r`, `coolwarm` |
| **qualitative** | distinct hues of similar lightness | nominal categories, with no order | `tab10`, Okabe–Ito |

Three rules follow:
- **Never encode nominal categories with a sequential map.** Its steady darkening says "more", but states or payment methods aren't more or less of anything.
- **Center a diverging map on the real midpoint.** For correlations, fix the range to $[-1, 1]$ with `vmin=-1, vmax=1`, so 0 gets the neutral color. Otherwise matplotlib stretches the colors to your data's min and max, and the "neutral" color lands wherever the middle of that range happens to be.
- **Think about color vision.** Roughly 1 in 12 men and 1 in 200 women have a *color vision deficiency*, most often trouble telling red from green. Don't make red vs green the only contrast. Prefer maps designed for it (`viridis`, `cividis`), vary lightness as well as hue, and label things directly when you can.

**Luminance** is the perceived brightness of a color. The eye reads "more" mostly through lightness, so a sequential map should change luminance **monotonically**: always getting lighter, or always darker. *Rainbow* maps like `jet` go dark → bright → dark. They create bands that aren't in the data, and their order is lost when printed in grayscale.
""")
nb.code(r"""
gradient = np.linspace(0, 1, 256)[None, :]
fig, axes = plt.subplots(5, 1, figsize=(8, 2.6))
for ax, name in zip(axes, ["viridis", "cividis", "RdBu_r", "tab10", "jet"]):
    ax.imshow(gradient, aspect="auto", cmap=name)
    ax.set_axis_off()
    ax.text(-0.01, 0.5, name, transform=ax.transAxes, ha="right", va="center")
plt.show()
""")
nb.md(r"""
🔮 **4.1** Which family would you use for each of these? (1) payment type, (2) number of orders per state, (3) change in a state's late share compared with last year. Answer as a tuple of `"sequential"`, `"diverging"`, `"qualitative"`.
""")
nb.learner("palette-type", r"""
predict("palette-type", ...)   # e.g. ("sequential", "sequential", "sequential")
""")
nb.md(r"""
✍️ **4.2** A **correlation coefficient** is a number between −1 and 1 that measures how closely two numeric variables move together: +1 means they rise together perfectly, −1 means one falls as the other rises, 0 means no linear relationship. Notebook 04 defines it precisely. `df.corr()` returns the matrix of coefficients for every pair of columns.

Write `corr_heatmap(corr, ax=None)`. Draw the matrix with `imshow` (or `pcolormesh`), using a **diverging** colormap fixed to $[-1, 1]$. Label both axes with the variable names, and add a colorbar.
""")
nb.learner("corr-heatmap", r"""
def corr_heatmap(corr: pd.DataFrame, ax=None):
    # your code here
    raise NotImplementedError


check("corr-heatmap", corr_heatmap)
""")
nb.code(r"""
corr = order_table[["items_value", "freight_value", "n_items", "delivery_days", "delay_days"]].corr()
corr_heatmap(corr)
plt.show()
""")
nb.md(r"""
★ **4.3** Back to late deliveries. Write `deviation_bars(order_table, min_orders=1000, ax=None)`: for each state with at least `min_orders` delivered orders, a bar showing its late share **minus the national late share**. The national share is computed over *all* delivered orders, small states included. Sort the bars from highest to lowest, color them **by sign** (one color above zero, another below), and draw a line at 0.
""")
nb.learner("deviation-bars", r"""
def deviation_bars(order_table: pd.DataFrame, min_orders: int = 1000, ax=None):
    # your code here
    raise NotImplementedError


check("deviation-bars", deviation_bars)
""")
nb.code(r"""
deviation_bars(order_table)
plt.show()
""")
nb.md(r"""
★ **4.4** Test a colormap for honesty. Screens store colors as sRGB values $c \in [0, 1]$ per channel, which are *gamma-encoded*: not proportional to light intensity. The **relative luminance** of a color undoes that encoding, then weighs the channels by how sensitive the eye is to each:

$$c_{\text{lin}} = \begin{cases} c / 12.92 & c \le 0.04045 \\ \left(\dfrac{c + 0.055}{1.055}\right)^{2.4} & \text{otherwise} \end{cases} \qquad Y = 0.2126\,R_{\text{lin}} + 0.7152\,G_{\text{lin}} + 0.0722\,B_{\text{lin}}$$

Green dominates the weights because the eye is most sensitive to it. Write `is_monotone_luminance(cmap_name)`. Sample 256 colors from the map, compute $Y$ for each, and return `True` if $Y$ never decreases or never increases along the map.
""")
nb.learner("luminance-monotone", r"""
def is_monotone_luminance(cmap_name: str) -> bool:
    # your code here
    raise NotImplementedError


check("luminance-monotone", is_monotone_luminance)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 5 · Labels, annotations, export

A chart travels without you, so everything a reader needs must be on it:

- **Title:** the question or the finding ("Late deliveries are 3× more common in CE than in SP"), not just the variable names.
- **Axis labels with units:** "Revenue (BRL)", "Delivery time (days)", "Share of delivered orders".
- **Population and denominator:** which orders, which period, "states with ≥ 1,000 delivered orders". A caption or a note below the chart works.
- **The n of each group** when groups differ in size. `ax.bar_label(container, labels=[...])` writes text on bars.
- **Direct labels** beat legends when there are only a few series. Put the label at the end of the line.

**Exporting.** `fig.savefig(path, dpi=..., bbox_inches=...)`:
- `figsize` is in **inches**, and **dpi** (*dots per inch*) converts inches to pixels: a 8 × 4 in figure at 150 dpi is 1200 × 600 px. Font sizes are in points (1/72 in), so they keep their size relative to the figure whatever the dpi.
- `bbox_inches="tight"` crops the empty margin around the drawing, which is good for documents, but the final size then no longer equals `figsize × dpi`.
- **PNG** is a *raster* format: a fixed grid of pixels, which gets blurry when enlarged. **SVG** and **PDF** are *vector* formats: they store shapes and stay sharp at any size, which suits print and slides.
- Close figures you create in a loop (`plt.close(fig)`). Otherwise they pile up in memory.
""")
nb.code(r"""
fig, ax = plt.subplots(figsize=(8, 4))
print("figure size (in):", fig.get_size_inches(), "| default dpi:", fig.dpi)
print("pixels at 150 dpi:", tuple(int(v * 150) for v in fig.get_size_inches()))
plt.close(fig)
""")
nb.md(r"""
★ **5.1** Write `save_png(fig, path, width_px)`. It saves `fig` as a PNG exactly `width_px` pixels wide, without changing the figure's size in inches.
""")
nb.learner("export-size", r"""
def save_png(fig, path, width_px: int) -> None:
    # your code here
    raise NotImplementedError


check("export-size", save_png)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 6 · Mini-project: when do customers shop?

An online marketplace plans its promotions, server capacity and customer service around **when** people buy. Two time dimensions matter here, the day of the week and the hour of the day. A **heatmap** shows both at once: a grid of cells colored by a count, using a *sequential* colormap, since counts only go from low to high.

> 🧰 `ts.dt.dayofweek` gives 0 for Monday through 6 for Sunday, and `ts.dt.hour` gives 0–23. `pd.crosstab(a, b)` counts each pair of values. Cells with no purchases simply don't appear in it, so `reindex` the result to the full 7 × 24 grid.

✍️ **6.1** Write `purchase_heatmap(timestamps, ax=None)`: a 7 × 24 heatmap of purchase counts, with rows Monday → Sunday and columns 0 → 23 h. Use a sequential colormap, day names as y-tick labels, an x label, and a colorbar.
""")
nb.learner("purchase-heatmap", r"""
def purchase_heatmap(timestamps: pd.Series, ax=None):
    # your code here
    raise NotImplementedError


check("purchase-heatmap", purchase_heatmap)
""")
nb.code(r"""
purchase_heatmap(order_table["order_purchase_timestamp"])
plt.show()
""")
nb.md(r"""
💬 **6.2** Write the **caption** you'd publish under this chart, in two or three sentences: the main pattern, one thing that surprised you, and the population (which orders, which period). Check every claim against the numbers first, for example by summing the grid's rows and columns.
""")
nb.answer("heatmap-caption", "> Your caption:")
nb.md(r"""
★ **6.3** *(open, no auto-check)* Busier days have darker cells everywhere, which hides whether Saturday's *daily rhythm* differs from Monday's. Divide each row by its total, so each cell becomes the share of that day's purchases, and redraw. What changes in the story? Which version would you show a manager planning staff shifts, and which one a marketing team planning a weekend campaign?
""")
nb.learner("heatmap-open", r"""
# ★ 6.3 — your exploration here
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## Wrap-up

💬 Answer briefly, in your own words.

1. Why must a bar chart's axis start at 0, when a line chart's doesn't have to?
2. You need to compare the share of *boleto* payments across five regions. Which chart do you choose, and which one do you avoid?
3. What does a 100 % stacked bar hide, and how do you put that information back?
4. Match each to a colormap family: review score 1–5, product category, profit vs target.
5. List four things a chart needs on it to be read correctly without you there.
""")
nb.answer("wrap-up", "> Your answers:")
nb.code(r"""
progress()
""")
nb.md(r"""
**Next:** `06 · Missing data`. If you use Claude Code with this repository, `/tutor review m01/05` reads your notebook and gives you feedback on your charts and captions.
""")
