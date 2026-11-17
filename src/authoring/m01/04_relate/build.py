"""Builder: Module I · 04 — Comparing groups, relating variables."""
from nbbuild import Notebook

COURSE = "https://github.com/chaffrey8/DMEMCD_2026"

nb = Notebook(
    nb_id="m01/04",
    filename="04_comparing-and-relating.ipynb",
    title="Comparing groups, relating variables",
    subtitle="Conditional distributions, small-group noise, time, correlation, overplotting and confounding",
    grader="selfgrade.m01.relate",
    can_do=["compare groups through their conditional distributions, with their sizes in view",
            "explain why the most extreme group averages come from the smallest groups",
            "build honest monthly series (full calendar, partial periods, censoring)",
            "compute and interpret Pearson's r and Spearman's ρ, and spot influential points",
            "draw relationships without overplotting",
            "recognize confounding and Simpson's paradox, and adjust by stratifying"],
    builds_on="Notebooks 01–03: keys and validated joins, groupby, and mean/median/quantiles/standard deviation. "
              "They're used here as tools.",
    effort="≈ 2.5 h",
    sources=[f"Diplomado *Métodos Estadísticos y Matemáticos para la Ciencia de Datos* (UNAM, 2026), Module I: "
             f"[`02_EDA_Work_Olist` (§5–7, 9), `03_Visualizaciones_Work_Olist` (§C–E)]({COURSE})"],
    concept_map=["conditional distributions", "small groups & noise", "time: calendars, partial periods, censoring",
                 "covariance & correlation", "seeing relationships", "confounding & Simpson's paradox",
                 "mini-project: late deliveries and reviews"],
)

# ---------------------------------------------------------------------------
nb.md(r"""
## 0 · Setup

The same Olist tables as before (Notebook 01 describes them). From them we build `deliveries`: one row per **delivered order with an observed delivery date**, with the customer's state and the delivery time in days. Subtracting two timestamps gives a *timedelta*. `.dt.total_seconds() / 86400` turns it into fractional days, so 1.5 means a day and a half.
""")
nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from selfgrade.data import load_olist
from selfgrade.m01.relate import check, hint, predict, progress
from selfgrade.m01.relate import toy_deliveries, toy_xy, toy_carriers

orders, items = load_olist("orders"), load_olist("items")
reviews, customers = load_olist("reviews"), load_olist("customers")

is_delivered = orders["order_status"].eq("delivered") & orders["order_delivered_customer_date"].notna()
deliveries = orders[is_delivered].merge(customers[["customer_id", "customer_state"]],
                                        on="customer_id", how="left", validate="many_to_one")
deliveries["delivery_days"] = (deliveries["order_delivered_customer_date"]
                               - deliveries["order_purchase_timestamp"]).dt.total_seconds() / 86400
deliveries = deliveries[["order_id", "customer_state", "delivery_days"]]
print(f"{len(deliveries):,} delivered orders in {deliveries['customer_state'].nunique()} states")
deliveries.head()
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 1 · Conditional distributions

The **marginal distribution** of a variable is its distribution over all rows: for example, all delivery times together. The **conditional distribution** of $Y$ given $X = x$, written $Y \mid X = x$, is the distribution of $Y$ among the rows where $X$ equals $x$: for example, delivery times *of orders shipped to RJ*. For a categorical $Y$:

$$P(Y = y \mid X = x) \;=\; \frac{n(x, y)}{n(x)},$$

where $n(x, y)$ counts the rows with both values and $n(x)$ the rows with $X = x$. Comparing groups means comparing conditional distributions: their centers, spreads and tails, not just one number each.

A **contingency table** (or *cross-tabulation*) counts the rows for every pair of categories. Dividing each cell by its row total turns every row into a conditional distribution.

> 🧰 **How the code works:** `pd.crosstab(a, b)` counts the pairs. `normalize="index"` divides by row totals, `"columns"` by column totals, `"all"` by the grand total. For numeric outcomes, `df.groupby(x)[y].agg([...])` gives one summary row per group.

> ✅ **Good practice:** show each group's **size** next to its summary, and name the denominator ("share of *delivered* orders").
""")
nb.md(r"""
🔮 **1.1** In `pd.crosstab(year, status, normalize="index")`, what sums to 1: the `"rows"` or the `"columns"`?
""")
nb.learner("normalize-direction", r"""
predict("normalize-direction", ...)
""")
nb.code(r"""
year = orders["order_purchase_timestamp"].dt.year.rename("purchase_year")
status_mix = pd.crosstab(year, orders["order_status"], normalize="index")
display(status_mix.round(3))
print("row sums:", status_mix.sum(axis=1).round(6).tolist())
""")
nb.md(r"""
✍️ **1.2** Write `state_delivery_summary(df)` for a table with columns `customer_state` and `delivery_days`. It returns one row per state, with the columns `n_orders` (orders with an **observed** delivery time), `mean_days`, `median_days` and `p90_days` (the 90th percentile: 90 % of the orders arrived faster), in that order, sorted by `n_orders` descending.
""")
nb.code(r"""
display(toy_deliveries())
""")
nb.learner("state-delivery-summary", r"""
def state_delivery_summary(df: pd.DataFrame) -> pd.DataFrame:
    # your code here
    raise NotImplementedError


check("state-delivery-summary", state_delivery_summary)
""")
nb.code(r"""
summary = state_delivery_summary(deliveries)
display(summary.round(1))
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 2 · Small groups are noisy

Why did a state with 81 orders top the ranking in Notebook 01? Take $n$ independent observations $X_1, \dots, X_n$ from the same distribution, with mean $\mu$ and variance $\sigma^2$. Their average $\bar X = \frac1n \sum_i X_i$ satisfies

$$\mathbb E[\bar X] = \mu, \qquad \operatorname{Var}(\bar X) = \frac{1}{n^2}\sum_{i=1}^n \operatorname{Var}(X_i) = \frac{\sigma^2}{n}.$$

The variances add because the observations are independent. The **standard error** (SE) of the mean is the standard deviation of $\bar X$:

$$\operatorname{SE}(\bar X) = \frac{\sigma}{\sqrt n}, \quad\text{estimated by}\quad \frac{s}{\sqrt n},$$

where $s$ is the sample standard deviation. For a proportion (the mean of a 0/1 variable with success probability $p$), $\sigma^2 = p(1-p)$, so $\operatorname{SE} = \sqrt{p(1-p)/n}$.

**Intuition:** averaging cancels independent noise, but only at the rate $1/\sqrt n$. Four times the data halves the noise. A group of 40 orders has a standard error about 30 times larger than a group of 40,000. So when you rank groups by their raw average, the smallest groups land at **both** extremes, whatever the truth is.

A **funnel plot** shows this. It plots each group's average against its size $n$, with the overall mean $\bar x$ and the curves $\bar x \pm 2\,s/\sqrt n$. If every group had the same true mean, about 95 % of the groups would fall inside the funnel. That figure comes from the *central limit theorem*: averages of many independent observations are approximately normally distributed. Points outside the funnel are the ones worth a closer look.
""")
nb.md(r"""
🔮 **2.1** Suppose all 27 states had *exactly the same* true late-delivery rate, 7 %. You simulate their orders, with each state's real number of orders, and rank the states by their observed late rate. Which states would occupy the **top and the bottom** of the ranking?

**a)** the largest states  **b)** the smallest states  **c)** no pattern: it's random
""")
nb.learner("extremes-small", r"""
predict("extremes-small", ...)   # "a", "b" or "c"
""")
nb.code(r"""
sizes = deliveries["customer_state"].value_counts()
rng = np.random.default_rng(2026)
smallest5 = set(sizes.nsmallest(5).index)
top_from_small = bottom_from_small = 0
for _ in range(1000):
    rate = pd.Series(rng.binomial(sizes.to_numpy(), 0.07) / sizes.to_numpy(), index=sizes.index)
    top_from_small += rate.idxmax() in smallest5
    bottom_from_small += rate.idxmin() in smallest5
print(f"5 smallest states hold {5 / len(sizes):.0%} of the ranking slots, but in 1,000 simulations they were")
print(f"  the highest rate {top_from_small / 10:.0f}% of the time and the lowest {bottom_from_small / 10:.0f}% of the time")
""")
nb.md(r"""
★ **2.2** Write `funnel_plot(df, group, value, ax=None)`. It scatters each group's **mean of `value`** (y) against its **number of non-missing values** (x, on a **log** axis), draws a horizontal line at the overall mean, and draws the two band curves $\bar x \pm 2\,s/\sqrt n$, with $s$ the overall sample standard deviation of `value`, as lines over a range of $n$. Return the `Axes`.
""")
nb.learner("funnel-plot", r"""
def funnel_plot(df: pd.DataFrame, group: str, value: str, ax=None):
    # your code here
    raise NotImplementedError


check("funnel-plot", funnel_plot)
""")
nb.code(r"""
ax = funnel_plot(deliveries, "customer_state", "delivery_days")
ax.set(xlabel="delivered orders (log scale)", ylabel="mean delivery time (days)",
       title="Delivery time by state, against a no-difference funnel")
plt.show()
""")
nb.md(r"""
Most states fall far outside the funnel. Here the differences between states are real, not noise: distance from the sellers, who are mostly in SP, is a strong driver. The funnel doesn't say *why* a state differs. It says which differences are too large to blame on the group's size.
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 3 · Time: calendars, partial periods, censoring

Time series hide four traps:
- **Coverage.** Data starts and stops somewhere, and the first and last periods are often **partial**. Check the first and last timestamps before reading anything into the edges.
- **Gaps.** `value_counts()` only lists periods that occur. A month with zero events silently disappears, and a line plot then connects its neighbours. Reindex to the full calendar.
- **Denominators.** When volume changes over time, compare **rates** (per order, per customer) rather than raw counts.
- **Censoring.** A quantity that needs time to be observed, such as "was it delivered?" or "did they buy again?", isn't known yet for recent periods. Such an outcome is **right-censored**: it's missing because the observation window closed too early, not because of anything about those orders.

A **cumulative** curve (a running total of counts) can never go down, because it only adds non-negative numbers. It always looks like growth. Its *slope* is the information: a flat stretch means few new events.

> 🧰 **How the code works:** `ts.dt.to_period("M")` maps timestamps to calendar months (`Period` objects), which sort and plot in calendar order. `pd.period_range(start, end, freq="M")` builds the full calendar, and `s.reindex(calendar, fill_value=0)` fills in the missing months.
""")
nb.code(r"""
first, last = orders["order_purchase_timestamp"].min(), orders["order_purchase_timestamp"].max()
print("purchases from", first, "to", last)
""")
nb.md(r"""
✍️ **3.1** Write `monthly_orders(timestamps)`, which returns the number of orders per calendar month: a Series indexed by month periods, from the first to the last month, **including months with zero orders**, as integers. It should ignore missing timestamps.
""")
nb.learner("monthly-orders", r"""
def monthly_orders(timestamps: pd.Series) -> pd.Series:
    # your code here
    raise NotImplementedError


check("monthly-orders", monthly_orders)
""")
nb.code(r"""
monthly = monthly_orders(orders["order_purchase_timestamp"])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))
ax1.plot(monthly.index.to_timestamp(), monthly.to_numpy(), marker="o")
ax1.set(title="Orders per month (full calendar)", ylabel="orders")
ax2.step(monthly.index.to_timestamp(), monthly.cumsum().to_numpy(), where="post")
ax2.set(title="Cumulative orders: never decreases", ylabel="orders so far")
fig.autofmt_xdate(); plt.show()
print(monthly.head(4).to_string(), "\n...\n", monthly.tail(3).to_string())
""")
nb.md(r"""
🐞 **3.2** A colleague wants to know whether sales are seasonal. They run the cell below and report: *"August is our best month, and every year sales collapse in September, to less than half."* Run it and find what's wrong.
""")
nb.code(r"""
by_month = orders.groupby(orders["order_purchase_timestamp"].dt.month).size()
by_month.plot(kind="bar", title="Orders by month", xlabel="month", ylabel="orders")
plt.show()
""")
nb.md(r"""
💬 **3.3** What went wrong? Use the monthly series above to say how many Augusts and how many Septembers each bar actually contains.
""")
nb.answer("debug-month")
nb.md(r"""
★ **3.4** Write `delivered_rate_by_month(orders)`: the share of orders with status `delivered`, by **purchase month** (only months that have orders), as a Series indexed by month.
""")
nb.learner("delivered-rate-by-month", r"""
def delivered_rate_by_month(orders: pd.DataFrame) -> pd.Series:
    # your code here
    raise NotImplementedError


check("delivered-rate-by-month", delivered_rate_by_month)
""")
nb.code(r"""
delivered_rate_by_month(orders).tail(6).round(3)
""")
nb.md(r"""
💬 **3.5** You want to compare review scores across purchase months. Which months would you leave out, or flag, and why? Give two different reasons.
""")
nb.answer("recent-months")

# ---------------------------------------------------------------------------
nb.md(r"""
## 4 · Covariance and correlation

The **sample covariance** of paired observations $(x_i, y_i)$, $i = 1, \dots, n$, is

$$s_{xy} = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar x)(y_i - \bar y).$$

Each term is positive when the point is above both means or below both, and negative when it's above one and below the other. So $s_{xy} > 0$ means the variables tend to move together. Its units are the product of the two variables' units (BRL × days, say), so its size can't be compared across pairs of variables.

**Pearson's correlation coefficient** removes the units: $r = s_{xy} / (s_x s_y)$, with $s_x$ and $s_y$ the standard deviations. Here's the linear-algebra view. Write the **centered** data vectors $\tilde{\mathbf x} = (x_1 - \bar x, \dots, x_n - \bar x)$ and $\tilde{\mathbf y}$ likewise. The factors $\frac{1}{n-1}$ cancel, and

$$r \;=\; \frac{\tilde{\mathbf x}\cdot\tilde{\mathbf y}}{\lVert\tilde{\mathbf x}\rVert\,\lVert\tilde{\mathbf y}\rVert} \;=\; \cos\theta,$$

where $\theta$ is the angle between the two vectors in $\mathbb R^n$. So:
- $-1 \le r \le 1$, by the Cauchy–Schwarz inequality.
- $|r| = 1$ exactly when $\tilde{\mathbf y} = c\,\tilde{\mathbf x}$, meaning all points lie on one straight line.
- $r$ doesn't change if you shift either variable or rescale it by a positive factor.
- $r$ measures **linear** association only. $r = 0$ doesn't mean "unrelated": for $y = x^2$ with $x$ symmetric around 0, $r = 0$.

**Spearman's rank correlation** $\rho$ (rho) is Pearson's $r$ computed on **ranks**: replace each value by its position in sorted order (1 = smallest), and give tied values the average of the positions they share. It measures **monotone** association: $\rho = 1$ when $y$ always increases with $x$, in a straight line or not. A rank can't be "far away", so a single extreme value has bounded influence on $\rho$.

> 🧰 **How the code works:** `s.corr(t)` is Pearson. `s.corr(t, method="spearman")` is Spearman. `df.corr(numeric_only=True)` gives the matrix for every pair of numeric columns. Each pair uses only the rows where both values are present. `s.rank()` uses average ranks for ties by default.

> ⚠️ **Correlation is not causation.** An association can come from a third variable that drives both (**confounding**, section 6), from the effect running the other way (**reverse causation**), from how the rows were selected (**selection**), or from chance. Look at the scatter, report $n$, and say which of these you've ruled out.
""")
nb.code(r"""
xy = toy_xy()
display(xy.T)
print("r =", round(xy["x"].corr(xy["y"]), 3), "| rho =", round(xy["x"].corr(xy["y"], method="spearman"), 3))
""")
nb.md(r"""
🔮 **4.1** Add one point, $(x, y) = (20, 0)$, to the six toy points. What will the **sign** of Pearson's $r$ be: `"positive"` or `"negative"`?
""")
nb.learner("influential-point", r"""
predict("influential-point", ...)
""")
nb.code(r"""
xy7 = pd.concat([xy, pd.DataFrame({"x": [20.0], "y": [0.0]})], ignore_index=True)
print("with (20, 0): r =", round(xy7["x"].corr(xy7["y"]), 3),
      "| rho =", round(xy7["x"].corr(xy7["y"], method="spearman"), 3))
fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(xy7["x"], xy7["y"]); ax.set(title="One influential point", xlabel="x", ylabel="y"); plt.show()
""")
nb.md(r"""
✍️ **4.2** Write `pearson(x, y)` using the cosine formula, with NumPy and without `.corr` or `np.corrcoef`. Use only the pairs where both values are present, accept Series or arrays, and return `nan` when a variable is constant.
""")
nb.learner("pearson-by-hand", r"""
def pearson(x, y) -> float:
    # your code here
    raise NotImplementedError


check("pearson-by-hand", pearson)
""")
nb.md(r"""
✍️ **4.3** Write `spearman(x, y)` by reusing `pearson` on ranks. Drop incomplete pairs **before** ranking.
""")
nb.learner("spearman-by-hand", r"""
def spearman(x, y) -> float:
    # your code here
    raise NotImplementedError


check("spearman-by-hand", spearman)
""")
nb.md(r"""
🔮 **4.4** For $x = 0, 1, \dots, 9$ and $y = e^x$, which coefficient equals exactly 1: `"pearson"` or `"spearman"`?
""")
nb.learner("monotone-not-linear", r"""
predict("monotone-not-linear", ...)
""")
nb.code(r"""
x = pd.Series(np.arange(10.0)); y = np.exp(x)
print("pearson:", round(x.corr(y), 4), "| spearman:", round(x.corr(y, method="spearman"), 4))
""")
nb.md(r"""
✍️ **4.5** On the real `items` table (one row per item line), compute Pearson's and Spearman's correlation between `price` and `freight_value`.
""")
nb.learner("price-freight-corr", r"""
r_price_freight = ...
rho_price_freight = ...

check("price-freight-corr", r_price_freight, rho_price_freight)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 5 · Seeing relationships

With 100,000 points, a scatter plot becomes a solid blob. That's **overplotting**: dense and sparse regions look the same. The fixes:
- **Transparency** (`alpha` between 0 and 1): overlapping points add up to darker areas.
- **Sampling**: plot a random subset, with a fixed `random_state`, so the figure is reproducible.
- **Binning in 2-D**: `hexbin` counts the points in each hexagonal cell, and `hist2d` does the same with rectangles. Color then encodes the count.
- **Log scales** for skewed, money-like variables, so the bulk of the data isn't squeezed into a corner.
- **Jitter** for discrete variables, such as review scores 1–5 or item counts: add a little random noise to the position so stacked points spread out. It's for drawing only, never for computing.
- **Small multiples**: the same plot repeated for each group, with **shared axes** (`sharex=True, sharey=True`) so that positions can be compared across panels.
""")
nb.code(r"""
# Log axes need positive values: items with free shipping (freight 0) are left out of this view only.
pf = items.loc[items["freight_value"].gt(0), ["price", "freight_value"]]
print(f"{(items['freight_value'] == 0).sum():,} free-shipping items not shown")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharex=True, sharey=True)
axes[0].scatter(pf["price"], pf["freight_value"], s=4)
axes[0].set_title("all points: a blob")
sample = pf.sample(5000, random_state=2026)
axes[1].scatter(sample["price"], sample["freight_value"], s=4, alpha=0.2)
axes[1].set_title("5,000 sampled, alpha=0.2")
axes[2].hexbin(pf["price"], pf["freight_value"], gridsize=40, bins="log", xscale="log", yscale="log", mincnt=1)
axes[2].set_title("hexbin (log counts)")
for ax in axes:
    ax.set(xscale="log", yscale="log", xlabel="price (BRL)")
axes[0].set_ylabel("freight (BRL)")
plt.show()
""")
nb.md(r"""
✍️ **5.1** Write `jitter_scatter(x, y, width=0.15, seed=0, ax=None)`. It scatters `x + noise` against `y`, where the noise is uniform on $[-\text{width}, \text{width}]$ and generated by `np.random.default_rng(seed)`. Use `alpha < 1`, leave `y` untouched, and return the `Axes`.
""")
nb.learner("jitter-plot", r"""
def jitter_scatter(x, y, width: float = 0.15, seed: int = 0, ax=None):
    # your code here
    raise NotImplementedError


check("jitter-plot", jitter_scatter)
""")
nb.code(r"""
per_order = items.groupby("order_id").agg(n_items=("price", "size"), value=("price", "sum"))
small = per_order[per_order["n_items"] <= 6].sample(4000, random_state=1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
a1.scatter(small["n_items"], small["value"], s=6)
a1.set(title="no jitter: stacked columns", xlabel="items in the order", ylabel="order value (BRL)", yscale="log")
jitter_scatter(small["n_items"], small["value"], width=0.25, seed=1, ax=a2)
a2.set(title="jittered", xlabel="items in the order")
plt.show()
""")
nb.code(r"""
# Small multiples: one panel per state, same axes everywhere
four = ["SP", "RJ", "BA", "AM"]
fig, axes = plt.subplots(1, 4, figsize=(13, 3), sharex=True, sharey=True)
for ax, st in zip(axes, four):
    ax.hist(deliveries.loc[deliveries["customer_state"].eq(st), "delivery_days"].clip(upper=60),
            bins=30, density=True)
    ax.set_title(st)
axes[0].set_ylabel("density"); fig.supxlabel("delivery time, days (view capped at 60)")
plt.show()
""")
nb.md(r"""
**Correlation matrices as heatmaps.** With many variables, show all pairwise correlations at once as a grid of colored cells. Use a **diverging** colormap fixed to $[-1, 1]$: two hues meeting at a neutral middle, so 0 looks neutral and the sign is the hue. Write the values in the cells too. Notebook 05 covers color choice in detail, and you'll build this chart yourself there.
""")
nb.code(r"""
def corr_heatmap(df: pd.DataFrame, method: str = "pearson", ax=None):
    corr = df.select_dtypes("number").corr(method=method)
    ax = ax or plt.subplots(figsize=(5, 4))[1]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.figure.colorbar(im, ax=ax)
    return ax
""")
nb.code(r"""
num = (items.groupby("order_id").agg(items_value=("price", "sum"), freight=("freight_value", "sum"),
                                     n_items=("price", "size"))
       .merge(deliveries.set_index("order_id")[["delivery_days"]], left_index=True, right_index=True,
              how="inner", validate="one_to_one"))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, m in zip(axes, ["pearson", "spearman"]):
    corr_heatmap(num, method=m, ax=ax)
    ax.set_title(m)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 6 · Confounding and Simpson's paradox

A **confounder** is a third variable that influences both the variable you're comparing on and the outcome. It can create an association that isn't causal, or hide one that is. The standard defense is **stratification**: compare within levels (*strata*) of the suspected confounder, where it's held roughly constant.

**Simpson's paradox** is the extreme case: an association that holds within *every* stratum reverses when the strata are pooled. There's no magic in it. Group $g$'s overall rate is a weighted average of its rates $p_{g,k}$ in the strata $k$:

$$p_g = \sum_k w_{g,k}\, p_{g,k}, \qquad w_{g,k} = \frac{\text{rows of } g \text{ in stratum } k}{\text{rows of } g}.$$

If the groups have very different mixes $w_{g,k}$, the overall rates mostly compare the *mixes*, not the groups. **Direct standardization** gives every group the **same** weights, the overall stratum mix $w_k$: $\;p_g^{\text{std}} = \sum_k w_k\, p_{g,k}$.

Toy data: two carriers deliver to *near* and *far* regions, and `late` is 1 for a late delivery.
""")
nb.code(r"""
carriers = toy_carriers()
display(pd.crosstab([carriers["region"], carriers["carrier"]], carriers["late"], margins=True))
""")
nb.md(r"""
🔮 **6.1** Which carrier has the lower **overall** late rate, `"A"` or `"B"`? Work it out from the counts.
""")
nb.learner("simpson-overall", r"""
predict("simpson-overall", ...)
""")
nb.code(r"""
print("overall late rate:", carriers.groupby("carrier")["late"].mean().to_dict())
""")
nb.md(r"""
✍️ **6.2** Write `stratified_rates(df, group, stratum, outcome)`. It returns a DataFrame with one row per stratum and one column per group, holding the mean of `outcome`.
""")
nb.learner("stratified-rates", r"""
def stratified_rates(df: pd.DataFrame, group: str, stratum: str, outcome: str) -> pd.DataFrame:
    # your code here
    raise NotImplementedError


check("stratified-rates", stratified_rates)
""")
nb.md(r"""
★ **6.3** Write `standardize(df, group, stratum, outcome)`, which returns each group's directly standardized rate (a Series indexed by group), using the overall share of rows in each stratum as the common weights.
""")
nb.learner("standardize", r"""
def standardize(df: pd.DataFrame, group: str, stratum: str, outcome: str) -> pd.Series:
    # your code here
    raise NotImplementedError


check("standardize", standardize)
""")
nb.code(r"""
display(stratified_rates(carriers, "carrier", "region", "late"))
print("standardized:", standardize(carriers, "carrier", "region", "late").round(3).to_dict())
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 7 · Mini-project: do late deliveries get worse reviews?

This puts everything together: a precise population, a deduplication decision, validated joins, a group comparison, a correlation, and a stratified check.
""")
nb.md(r"""
✍️ **7.1** Population: orders with status `delivered` and a non-missing `order_delivered_customer_date`. An order is **late** when its delivery date, normalized to the day with `.dt.normalize()`, is after `order_estimated_delivery_date`. Store the boolean in a column `is_late` of a table called `dv`, and compute `share_late`.
""")
nb.learner("late-share", r"""
dv = ...
share_late = ...

check("late-share", share_late)
""")
nb.md(r"""
✍️ **7.2** Some orders have several reviews (Notebook 01). Build `reviews_one` with **one row per order**, keeping the review with the **latest** `review_answer_timestamp`.
""")
nb.learner("reviews-one", r"""
reviews_one = ...

check("reviews-one", reviews_one)
""")
nb.md(r"""
✍️ **7.3** Join `dv` with `reviews_one` (choose `validate` carefully) and compute the mean `review_score` for late and for on-time orders.
""")
nb.learner("late-vs-on-time", r"""
mean_score_late = ...
mean_score_on_time = ...

check("late-vs-on-time", mean_score_late, mean_score_on_time)
""")
nb.md(r"""
💬 **7.4** Is the gap evidence that lateness *causes* bad reviews? Name at least two plausible **confounders**: variables related both to being late and to the score. How would you check one of them?
""")
nb.answer("confounders")
nb.md(r"""
✍️ **7.5** "Late" throws away *how* late. Compute `delay_days` = (delivery − estimated delivery), in days, from `.dt.total_seconds() / 86400`. It's negative when the order arrived early. Then compute Spearman's $\rho$ between `delay_days` and `review_score` on the joined table.
""")
nb.learner("delay-score-spearman", r"""
rho_delay_score = ...

check("delay-score-spearman", rho_delay_score)
""")
nb.md(r"""
★ **7.6** Test the most obvious confounder, the customer's state (remote states get later deliveries *and* might review differently). Compute `gap_by_state`: for each state with **at least 1,000** reviewed delivered orders, the mean score of late orders minus the mean score of on-time orders.
""")
nb.learner("gap-within-states", r"""
gap_by_state = ...

check("gap-within-states", gap_by_state)
""")
nb.md(r"""
💬 **7.7** Does the gap survive within states? What does that rule out, and what doesn't it rule out?
""")
nb.answer("gap-conclusion")
nb.md(r"""
★ **7.8** *(open, no auto-check)* Plot the mean review score per bucket of `delay_days`: for example, `pd.cut` at −30, −15, −7, 0, 3, 7, 15, 30. Then add the number of orders per bucket. Is the relationship gradual, or is there a threshold? Write a two-sentence conclusion.
""")
nb.learner("explore", r"""
# ★ 7.8: your exploration here
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## Wrap-up

💬 Answer briefly, in your own words.

1. What's the difference between a marginal and a conditional distribution? Give one of each from this notebook.
2. Why do the smallest groups take both extremes of a ranking, and what would you show instead of a raw ranking?
3. Name two ways a monthly series can mislead you at its edges.
4. Give a dataset where Pearson's $r$ is near 0 but the variables are strongly related, and one where a single point decides the sign of $r$.
5. Explain Simpson's paradox to a colleague, using weighted averages.
""")
nb.answer("wrap-up", "> Your answers:")
nb.code(r"""
progress()
""")
nb.md(r"""
**Next:** `05 · Honest charts`. If you use Claude Code with this repository, `/tutor review m01/04` reads your notebook and gives you feedback.
""")
