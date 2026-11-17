"""Builder: Module I · 03 — Describing a distribution."""
from nbbuild import Notebook

COURSE = "https://github.com/chaffrey8/DMEMCD_2026"

nb = Notebook(
    nb_id="m01/03",
    filename="03_describing-distributions.ipynb",
    title="Describing a distribution",
    subtitle="Scales, center, spread, position and shape, and the plots that show them",
    grader="selfgrade.m01.describe",
    can_do=["classify a variable's scale and pick the summaries it allows",
            "choose between classical and robust summaries, and justify the choice",
            "compute quantiles, skewness and Tukey's fences by hand",
            "pick and read a one-variable plot: histogram, KDE, ECDF, box/violin, Q–Q",
            "describe a skewed real variable honestly"],
    builds_on="Notebooks 01–02 (keys, joins, the order-level table, groupby, delivery time as a derived variable). These are used here as tools, not re-taught.",
    effort="≈ 2 h",
    sources=[f"Diplomado *Métodos Estadísticos y Matemáticos para la Ciencia de Datos* (UNAM, 2026), Module I: "
             f"[`02_EDA_Work_Olist` (§1, 3, 4, 8), `03_Visualizaciones_Work_Olist` (§B, Q–Q plot)]({COURSE})"],
    concept_map=["scales & frequency tables", "center, position, spread", "shape: skewness & log scale",
                 "seeing a distribution", "Tukey's fences", "mini-project: delivery time"],
)

# ---------------------------------------------------------------------------
nb.md(r"""
## 0 · Setup

The same Olist tables as in Notebook 01. The cell also rebuilds a small **order-level table**, one row per order, with the item totals aggregated *before* the join (Notebook 01). The 775 orders with no item rows get a missing `items_value`, not a zero.
""")
nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats

from selfgrade.data import load_olist
from selfgrade.m01.describe import check, hint, predict, progress
from selfgrade.m01.describe import toy_days, toy_payment_types, toy_deliveries

orders, items = load_olist("orders"), load_olist("items")
customers, payments = load_olist("customers"), load_olist("payments")

item_totals = items.groupby("order_id").agg(n_items=("price", "size"), items_value=("price", "sum"),
                                            freight_value=("freight_value", "sum"))
order_table = (orders.merge(customers[["customer_id", "customer_state"]], on="customer_id",
                            how="left", validate="many_to_one")
                     .merge(item_totals, on="order_id", how="left", validate="one_to_one"))
assert len(order_table) == len(orders)
print(order_table.shape, "| orders without items:", order_table["items_value"].isna().sum())
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 1 · What kind of variable is it?

Before computing anything, ask what kind of values a column holds. The answer decides which summaries make sense. The classic taxonomy is Stevens' **levels of measurement**:

| Scale | The values are… | Meaningful operations | Olist examples |
|---|---|---|---|
| **Nominal** | labels for groups | $=,\ \neq$, counting | `order_status`, `customer_state`, `payment_type`, zip-code prefixes |
| **Ordinal** | labels with an order | the above, plus $<,\ >$, median, quantiles | `review_score` (1–5) |
| **Interval** | numbers whose *differences* mean something, with an arbitrary zero | the above, plus differences, mean, standard deviation | calendar dates and timestamps |
| **Ratio** | numbers with a true zero | the above, plus ratios ("twice as much") | `price`, `freight_value`, delivery days |

Each level allows everything the levels above it allow. Nominal and ordinal variables are **categorical** (qualitative); interval and ratio variables are **quantitative**. A summary has to respect the scale:
- the *mode* (the most frequent value) works on any scale
- the *median* needs an order
- the *mean* needs meaningful distances

Averaging zip-code prefixes is arithmetic on labels. Review scores get averaged all the time, but that silently assumes the step from 4 to 5 is the same "amount" as the step from 1 to 2. Say so when you do it.

🔮 **1.1** Classify `customer_state`, `review_score`, `order_purchase_timestamp` and `freight_value`. Answer with a tuple of four words from `"nominal"`, `"ordinal"`, `"interval"`, `"ratio"`.
""")
nb.learner("scale-of", r"""
predict("scale-of", ...)   # e.g. ("nominal", "ordinal", "ratio", "ratio")
""")
nb.md(r"""
**Frequency tables.** For a categorical variable, the natural summary is a table of **counts** $n_k$ (how many rows take the value $k$) and **relative frequencies** or *shares* $f_k = n_k / n$. The denominator $n$ is part of the claim, so always say what it counts: all rows? Rows with a known value? Only delivered orders?

> 🧰 **How the code works**
> - `s.value_counts()` counts each distinct value, sorted by count. It **drops missing values** unless you pass `dropna=False`.
> - ⚠️ `s.value_counts(normalize=True)` divides by the number of **non-missing** values, so its shares describe the known values only. With `dropna=False`, it divides by all rows.
> - `mask.mean()` on a boolean Series is the share of `True` values, because `True` counts as 1. The denominator is the length of the mask.

✍️ **1.2** Write `freq_table(s)`. It returns a DataFrame indexed by the values of `s`, **missing values included as their own row**, with columns `count` and `share` (the share of *all* rows), sorted by count, largest first.
""")
nb.code(r"""
toy_payment_types().to_frame().T
""")
nb.learner("freq-table", r"""
def freq_table(s: pd.Series) -> pd.DataFrame:
    # your code here
    raise NotImplementedError


check("freq-table", freq_table)
""")
nb.code(r"""
freq_table(orders["order_status"]).round(4)
""")
nb.md(r"""
✍️ **1.3** Same cancellations, two denominators. Compute:
- `share_delivered`: the share of **all** orders with status `delivered`
- `share_canceled_of_undelivered`: among orders that are **not** delivered, the share with status `canceled`
""")
nb.learner("status-shares", r"""
share_delivered = ...
share_canceled_of_undelivered = ...

check("status-shares", share_delivered, share_canceled_of_undelivered)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 2 · Center, position and spread

For a quantitative variable, three questions summarize most distributions:
- **center**: where is a typical value?
- **position**: where does a given value sit relative to the rest?
- **spread**: how far do values usually lie from the center?

Each question has a classical answer and a **robust** one, meaning one that a few extreme values can't distort. The difference matters a lot for money-like, right-skewed variables like prices.

### Center
- **Mean** $\bar x=\frac1n\sum_{i=1}^{n}x_i$, for observations $x_1,\dots,x_n$. It is the value $c$ that minimizes the total squared distance $\sum_i(x_i-c)^2$: setting the derivative $-2\sum_i(x_i-c)$ to zero gives $c=\bar x$. Because squared distances grow fast, far-away points pull hard. Moving one value by $\Delta$ moves the mean by $\Delta/n$, with no limit.
- **Median**: the middle of the sorted values, or the average of the two middle ones when $n$ is even. It minimizes $\sum_i|x_i-c|$. The derivative of that sum is (number of points below $c$) − (number of points above $c$), which is zero when half the data sit on each side. Sending the maximum to infinity doesn't move it.

The **breakdown point** of an estimate makes "robust" precise: it's the smallest fraction of observations you'd have to corrupt to drag the estimate arbitrarily far. For the mean it's $1/n$, so one value is enough. For the median it's about 50 %.

### Position: quantiles
The **$p$-quantile** $Q(p)$, for $0\le p\le1$, is a value with (about) a fraction $p$ of the data at or below it. The **quartiles** are $Q_1=Q(0.25)$, the median $Q(0.5)$, and $Q_3=Q(0.75)$. **Percentiles** are the same thing on a 0–100 scale: P90 $=Q(0.9)$.

With a finite sample, many values qualify, so software has to pick a rule. pandas and NumPy default to **linear interpolation**. Sort the data as $x_{(0)}\le\dots\le x_{(n-1)}$ (0-indexed), compute the fractional position $h=(n-1)\,p$, and interpolate between the two neighbours:

$$Q(p)=x_{(\lfloor h\rfloor)}+\big(h-\lfloor h\rfloor\big)\big(x_{(\lfloor h\rfloor+1)}-x_{(\lfloor h\rfloor)}\big)$$

For example, $x=[1,2,4,10]$ and $p=0.25$ give $h=0.75$, so $Q=1+0.75\cdot(2-1)=1.75$. R's default ("type 7") uses the same rule. Excel's `QUARTILE.EXC` doesn't, so small samples can disagree across tools.

### Spread
- **Sample variance** $s^2=\frac{1}{n-1}\sum_i(x_i-\bar x)^2$. The **standard deviation** (std) $s=\sqrt{s^2}$ is in the data's own units.
  *Why $n-1$?* The deviations are measured from $\bar x$, which was fitted to these same points. As shown above, $\bar x$ is exactly the center that makes the squared deviations smallest, so they are systematically smaller than deviations from the true mean $\mu$ of the process that generated the data. One can show $\mathbb E\big[\sum_i(x_i-\bar x)^2\big]=(n-1)\,\sigma^2$, where $\sigma^2$ is the true variance, so dividing by $n-1$ makes $s^2$ right on average (*unbiased*). Another way to see it: the $n$ deviations must sum to zero, so only $n-1$ of them are free. These are the *degrees of freedom*.
- **Interquartile range** (IQR) $=Q_3-Q_1$: the width of the middle 50 % of the data. It ignores both tails entirely.
- **Median absolute deviation** (MAD) $=\operatorname{median}_i\,\lvert x_i-\operatorname{median}(x)\rvert$: the typical distance to the median, as robust as the median itself. For normally distributed data, $\text{MAD}\approx0.6745\,\sigma$, because $0.6745$ is the 75th percentile of the standard normal distribution $N(0,1)$. So $1.4826\cdot\text{MAD}$ (since $1/0.6745\approx1.4826$) estimates $\sigma$ and can be compared directly with $s$.

> 🧰 **How the code works**
> - `s.mean()`, `.median()`, `.std()` and `.quantile()` skip missing values by default (`skipna=True`). NumPy functions on an array that contains `NaN` return `nan`. Use `np.nanmedian` and friends there.
> - ⚠️ `Series.std()` uses `ddof=1` (divides by $n-1$), but `np.std(array)` uses `ddof=0` (divides by $n$). Same name, different number. `ddof` means "delta degrees of freedom".
> - `s.quantile([.25, .75])` returns a Series indexed by $p$. `describe()` uses the same interpolation.

> ✅ **Good practice**
> - Report center and spread as a matched pair: mean ± std, *or* median with IQR. Not mean with IQR.
> - For skewed, money-like variables, lead with the median and quantiles. Give the mean when the **total** matters, because mean × n = total revenue.
> - Name the population: "median over delivered orders with an observed delivery date", not just "median delivery time".

Toy data: delivery times in days for seven orders.
""")
nb.code(r"""
days = toy_days()
display(days.to_frame().T)
""")
nb.md(r"""
🔮 **2.1** The 30-day delivery was mis-typed as **300**. Which of `mean`, `median`, `std`, `IQR` change? Answer with a set, e.g. `{"mean"}`.
""")
nb.learner("robust-to-typo", r"""
predict("robust-to-typo", ...)
""")
nb.code(r"""
typo = days.replace(30, 300).rename("with 300")
pd.DataFrame({
    s.name: {"mean": s.mean(), "median": s.median(), "std": s.std(),
             "IQR": s.quantile(.75) - s.quantile(.25)}
    for s in (days, typo)
}).round(2)
""")
nb.md(r"""
✍️ **2.2** Implement `quantile_linear(x, p)` with the interpolation rule above. Don't use `np.quantile` or `Series.quantile`. It should accept any 1-D sequence (unsorted, possibly with missing values, which it ignores) and $0\le p\le1$. The check compares it with pandas on edge cases: $n=1$, $p\in\{0,1\}$, ties, and an unsorted input.
""")
nb.learner("quantile-linear", r"""
def quantile_linear(x, p: float) -> float:
    # your code here
    raise NotImplementedError


check("quantile-linear", quantile_linear)
""")
nb.md(r"""
🔮 **2.3** `a = days.to_numpy()`. What is `np.std(a) / days.std()`, to 3 decimals? Think about the formulas, not the data.
""")
nb.learner("ddof-ratio", r"""
predict("ddof-ratio", ...)
""")
nb.code(r"""
a = days.to_numpy()
print(np.std(a) / days.std(), "=", np.sqrt(6 / 7), "| np.std(a, ddof=1) == days.std():", np.isclose(np.std(a, ddof=1), days.std()))
""")
nb.md(r"""
✍️ **2.4** Write `spread_summary(s)`. It returns a Series with index `["std", "iqr", "mad_scaled"]`: the sample std, the IQR, and $1.4826\cdot$MAD. It ignores missing values and doesn't modify `s`.
""")
nb.learner("spread-summary", r"""
def spread_summary(s: pd.Series) -> pd.Series:
    # your code here
    raise NotImplementedError


check("spread-summary", spread_summary)
""")
nb.md(r"""
✍️ **2.5** On the real data: `order_value_spread = spread_summary(...)` of the order-level `items_value`.
""")
nb.learner("order-value-spread", r"""
order_value_spread = ...

check("order-value-spread", order_value_spread)
""")
nb.md(r"""
★ **2.6** *Breakdown in practice.* You're allowed to replace item prices with any values you like. What is the **minimum number** of prices you must replace to push the **median** item price strictly above 1,000 BRL? And to push the **mean** above 1,000 BRL? Return both as ints.
""")
nb.learner("breakdown", r"""
n_to_move_median = ...
n_to_move_mean = ...

check("breakdown", n_to_move_median, n_to_move_mean)
""")
nb.md(r"""
💬 **2.7** A report says *"the average order is worth 138 BRL"* to describe the typical customer. What would you report instead, and in which situation is the mean actually the right number?
""")
nb.answer("typical-order")

# ---------------------------------------------------------------------------
nb.md(r"""
## 3 · Shape: skewness and the log scale

A distribution is **skewed to the right** (positively skewed) when it has a long tail of large values, and skewed to the left when the long tail is on the small side. Prices, incomes, delivery times and file sizes are typically right-skewed: they can't go below zero, but a few can be huge.

**Skewness** measures this asymmetry. Write $m_k=\frac1n\sum_i(x_i-\bar x)^k$ for the $k$-th **central moment**, so $m_2$ is the variance with denominator $n$. Then

$$g_1=\frac{m_3}{m_2^{3/2}},\qquad G_1=\frac{\sqrt{n(n-1)}}{n-2}\;g_1 .$$

- Dividing by $m_2^{3/2}$ (a std cubed) makes $g_1$ unit-free.
- $G_1$, the *adjusted Fisher–Pearson coefficient*, corrects $g_1$'s small-sample bias. It's what pandas' `.skew()` reports.
- *Intuition:* cubing keeps each deviation's sign and magnifies the far ones, so whichever tail reaches farther dominates the sum.
- $G_1=0$ for symmetric data. As a loose rule of thumb, $|G_1|>1$ is strongly skewed. In right-skewed data the mean usually lies above the median, but that isn't guaranteed.

**The log scale.** Right-skewed positive variables often come from *multiplicative* processes: a price is a base cost times a markup times a shipping factor. The logarithm turns products into sums, $\log(ab)=\log a+\log b$, so on a log axis **equal steps are equal ratios**: 10 → 100 → 1,000 BRL are evenly spaced. Skewed data often become roughly symmetric after a log.
- `np.log10` and `np.log` differ only by a constant factor, so the shape is the same. Use `log10` for axes people read: 1, 2 and 3 mean 10, 100 and 1,000.
- `np.log1p(x)` $=\ln(1+x)$ is defined at 0, which matters for counts or amounts that can be 0. For large $x$ it is almost $\ln x$.
- The mean of the logs, transformed back, is the **geometric mean** $\exp\big(\frac1n\sum_i \ln x_i\big)$. It is always ≤ the ordinary mean.

★ **3.1** Implement `skewness(x)` from the formula for $G_1$. Ignore missing values. The check compares it with pandas.
""")
nb.learner("skewness", r"""
def skewness(x) -> float:
    # your code here
    raise NotImplementedError


check("skewness", skewness)
""")
nb.md(r"""
✍️ **3.2** Compute `skew_price`, the skewness of `items["price"]`, and `skew_log_price`, the skewness of `np.log10(items["price"])`. All prices are positive, so the log is safe.
""")
nb.learner("log-skew", r"""
skew_price = ...
skew_log_price = ...

check("log-skew", skew_price, skew_log_price)
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 4 · Seeing a distribution

Numbers summarize; plots show what the numbers hide: several peaks, gaps, heaps at round values, impossible values. Each plot below answers the question "what does this variable look like?" in a different way.

### Histogram
A **histogram** splits the range into intervals (*bins*) and draws a bar per bin.
- The bar height is the **count** of values in the bin, or, with `density=True`, the count divided by $n\cdot(\text{bin width})$. Then the total area is 1 and the plot estimates a **probability density**.
- A density is not a probability: it can exceed 1. Probabilities are *areas* under it.
- The bin width is a choice, and it changes the picture. Narrow bins are noisy; wide bins hide detail.
- The **Freedman–Diaconis rule** picks the width $h=2\cdot\text{IQR}\cdot n^{-1/3}$. It's robust (it uses the IQR rather than the std), and it shrinks slowly as $n$ grows, so more data buys more detail.
""")
nb.code(r"""
price = items["price"]
view = price[price <= price.quantile(.99)]   # truncating the VIEW to the 99th percentile; the data are unchanged
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=False)
for ax, bins in zip(axes, (10, 80)):
    ax.hist(view, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.set(title=f"Item price ≤ P99, {bins} bins", xlabel="price (BRL)", ylabel="item rows")
plt.tight_layout(); plt.show()
""")
nb.md(r"""
★ **4.1** Write `log_histogram(prices, n_bins)`. It draws a histogram of the prices with **`n_bins` bins of equal width on the log scale**, from the minimum to the maximum price. It sets the x axis to a log scale, labels both axes (the x label mentions BRL), and returns the matplotlib `Axes`.
""")
nb.learner("log-histogram", r"""
def log_histogram(prices: pd.Series, n_bins: int = 40):
    # your code here
    raise NotImplementedError


check("log-histogram", log_histogram)
""")
nb.md(r"""
★ **4.2** Write `fd_width(x)` (the Freedman–Diaconis width, ignoring missing values). Then compute `fd_bins_price`, the number of bins it implies for `items["price"]`: $\lceil(\max-\min)/h\rceil$.
""")
nb.learner("fd-bins", r"""
def fd_width(x) -> float:
    # your code here
    raise NotImplementedError


fd_bins_price = ...

check("fd-bins", fd_width, fd_bins_price)
""")
nb.md(r"""
### Kernel density estimate (KDE)
A **kernel density estimate** (KDE) is a smooth histogram. It places a small bump, the *kernel* $K$, on every observation and adds them up:

$$\hat f_h(x)=\frac{1}{n\,h}\sum_{i=1}^{n}K\!\left(\frac{x-x_i}{h}\right),\qquad K(u)=\frac{1}{\sqrt{2\pi}}e^{-u^2/2}\ \text{(Gaussian kernel)}.$$

- The **bandwidth** $h$ plays the role of the bin width: small $h$ is wiggly, large $h$ is over-smoothed.
- SciPy's `gaussian_kde` picks $h$ by *Scott's rule*, about $\hat\sigma\,n^{-1/5}$, which is ideal for roughly normal data and too wide for skewed or multi-peaked data.
- Each kernel integrates to 1, and the $\frac{1}{nh}$ factor makes the whole curve integrate to 1.

🔮 **4.3** Fit a Gaussian KDE to item prices (all positive). Does the estimated density put any probability on **negative** prices? Answer `"yes"` or `"no"`.
""")
nb.learner("kde-below-zero", r"""
predict("kde-below-zero", ...)
""")
nb.code(r"""
sample = price.sample(5000, random_state=0).to_numpy()
kde = stats.gaussian_kde(sample)
print(f"estimated probability of a negative price: {kde.integrate_box_1d(-np.inf, 0):.3f}")

grid = np.linspace(-100, 600, 400)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(sample[sample <= 600], bins=60, density=True, color="#cccccc", label="histogram (density)")
for factor, style in ((None, "-"), (0.3, "--")):
    k = stats.gaussian_kde(sample, bw_method=factor)
    ax.plot(grid, k(grid), style, label=f"KDE, bandwidth factor {k.factor:.3f}")
ax.axvline(0, color="black", lw=0.8)
ax.set(xlabel="price (BRL)", ylabel="density", title="KDE: bandwidth changes the picture"); ax.legend()
plt.show()
""")
nb.md(r"""
### Empirical cumulative distribution function (ECDF)
The **empirical cumulative distribution function** (ECDF) gives, for every $x$, the share of observations at or below it:

$$\hat F(x)=\frac{\#\{i:\ x_i\le x\}}{n}.$$

- It's a step function that jumps by $1/n$ at each observation (by $k/n$ at a value repeated $k$ times).
- It needs **no tuning**: no bins, no bandwidth.
- Quantiles can be read straight off it: $Q(p)$ is (about) where the curve crosses height $p$.
- Overlaying ECDFs is one of the best ways to compare groups.

🔮 **4.4** For the toy `days` $=[2,3,3,4,5,6,30]$, what is $\hat F(4)$, the ECDF at the median? Three decimals.
""")
nb.learner("ecdf-at-median", r"""
predict("ecdf-at-median", ...)
""")
nb.md(r"""
✍️ **4.5** Write `ecdf(x)`. It returns two arrays `(xs, ys)`: the **distinct** non-missing values in increasing order, and $\hat F$ at each of them. Missing values are excluded from $n$.
""")
nb.learner("ecdf", r"""
def ecdf(x):
    # your code here
    raise NotImplementedError


check("ecdf", ecdf)
""")
nb.code(r"""
xs, ys = ecdf(price)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.step(xs, ys, where="post")
ax.set_xscale("log")
for p in (0.5, 0.9):
    ax.axhline(p, color="grey", lw=0.6, ls=":")
ax.set(xlabel="item price (BRL, log scale)", ylabel="share of item rows ≤ x", title="ECDF of item prices")
plt.show()
""")
nb.md(r"""
### Box plot and violin plot
A **box plot** draws:
- a box from $Q_1$ to $Q_3$, with a line at the median
- *whiskers* out to the most extreme observations within $1.5\cdot\text{IQR}$ of the box
- every point beyond the whiskers individually (*fliers*)

It compresses a distribution into five numbers, which makes it good for comparing many groups side by side. ⚠️ The whiskers show where the **data** lie. They are not a confidence interval, and say nothing about how precisely the median is known.

A **violin plot** mirrors a KDE on both sides of an axis. It shows what a box hides, such as two peaks, at the cost of a bandwidth choice.
""")
nb.code(r"""
groups = order_table.dropna(subset=["items_value"]).assign(
    size=lambda d: np.where(d["n_items"] >= 3, "3+", d["n_items"].astype(int).astype(str)))
labels = ["1", "2", "3+"]
data = [np.log10(groups.loc[groups["size"] == g, "items_value"]) for g in labels]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
axes[0].boxplot(data, showfliers=True)
axes[1].violinplot(data, showmedians=True)
for ax, t in zip(axes, ("box plot", "violin plot")):
    ax.set_xticks([1, 2, 3], labels)
    ax.set(title=f"Order value by number of items: {t}", xlabel="item rows in the order")
axes[0].set_ylabel("log10(order value, BRL)")
plt.tight_layout(); plt.show()
""")
nb.md(r"""
### Quantile–quantile (Q–Q) plot
A **quantile–quantile** (Q–Q) plot compares the sample's quantiles with those of a reference distribution, usually the normal.
- Sort the data, $x_{(1)}\le\dots\le x_{(n)}$, and give the $i$-th value the probability $p_i=(i-0.5)/n$.
- Plot $x_{(i)}$ against $\Phi^{-1}(p_i)$, where $\Phi^{-1}$ is the *quantile function* (inverse cumulative distribution function) of $N(0,1)$.
- If the data are normal with mean $\mu$ and std $\sigma$, then $x_{(i)}\approx\mu+\sigma\,\Phi^{-1}(p_i)$: the points fall on a **straight line** with slope $\sigma$ and intercept $\mu$.
- Departures have a shape. If the right end bends upward, the right tail is heavier than a normal's. An S-shape means both tails are heavier (or lighter).

It's a *diagnostic*: with 100,000 points, every real variable "fails" a strict normality test, so ask whether the departure matters for what you'll do next.

★ **4.6** Write `qq_points(x)`. It returns `(theoretical, sample)` arrays: $\Phi^{-1}(p_i)$ with $p_i=(i-0.5)/n$, and the sorted non-missing values. (`scipy.stats.norm.ppf` is $\Phi^{-1}$.)
""")
nb.learner("qq-points", r"""
def qq_points(x):
    # your code here
    raise NotImplementedError


check("qq-points", qq_points)
""")
nb.code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for ax, (label, values) in zip(axes, (("price (BRL)", price), ("log10(price)", np.log10(price)))):
    th, sa = qq_points(values.sample(3000, random_state=1))
    ax.plot(th, sa, ".", ms=2)
    slope, intercept = np.polyfit(th[100:-100], sa[100:-100], 1)   # line through the bulk of the data
    ax.plot(th, intercept + slope * th, color="black", lw=0.8)
    ax.set(title=f"Normal Q–Q plot: {label}", xlabel="theoretical quantile Φ⁻¹(p)", ylabel="sample quantile")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## 5 · Unusual values: Tukey's fences

**Tukey's fences** turn the box plot's whiskers into a rule. A value is flagged when it lies outside

$$\big[\,Q_1-k\cdot\text{IQR},\ \ Q_3+k\cdot\text{IQR}\,\big],$$

with $k=1.5$ (or $k=3$ for "far out" values). For normally distributed data, the $k=1.5$ fences sit about 2.7 standard deviations from the mean and flag about 0.7 % of the points.

A **flag is a question, not a verdict**. An extreme value can be:
- **real**: an expensive item really was sold
- an **error**: a typo, or a unit mix-up
- a sign that the **scale is wrong**: on a skewed variable, the rule flags the ordinary right tail

> ✅ **Good practice:** keep the flag as a new boolean column and investigate it. Deleting flagged rows changes the population you describe. Cleaning decisions belong in Notebook 07.

✍️ **5.1** Write `tukey_flags(s, k=1.5)`. It returns a boolean Series aligned with `s`. Missing values are never flagged.
""")
nb.learner("tukey-flags", r"""
def tukey_flags(s: pd.Series, k: float = 1.5) -> pd.Series:
    # your code here
    raise NotImplementedError


check("tukey-flags", tukey_flags)
""")
nb.md(r"""
★ **5.2** Compute `share_flag_price`, the share of item prices flagged on the raw scale, and `share_flag_log`, the share flagged on the `np.log10` scale.
""")
nb.learner("tukey-shares", r"""
share_flag_price = ...
share_flag_log = ...

check("tukey-shares", share_flag_price, share_flag_log)
""")
nb.md(r"""
🐞 **5.3** A colleague applies the fences to order values and concludes that *no order is unusual*. The code runs without errors. Find the bug.
""")
nb.code(r"""
v = order_table["items_value"]
q1, q3 = np.percentile(v, [25, 75])
iqr = q3 - q1
flagged = (v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)
print("orders flagged as unusual:", flagged.sum())
""")
nb.md(r"""
💬 **5.4** Why does it print 0, and what are two ways to fix it?
""")
nb.answer("debug-percentile-nan")

# ---------------------------------------------------------------------------
nb.md(r"""
## 6 · Mini-project: how long does delivery take?

Customers care about delivery time, and so does Olist. Describe it properly: define the population, compute a robust summary, and pick a plot that compares states.
""")
nb.code(r"""
# Delivery time in fractional days, defined as in Notebook 02:
# only delivered orders with an observed delivery date have one (anything else is missing, not 0).
def delivery_days(orders: pd.DataFrame) -> pd.Series:
    ok = orders["order_status"].eq("delivered") & orders["order_delivered_customer_date"].notna()
    o = orders.loc[ok]
    return (o["order_delivered_customer_date"] - o["order_purchase_timestamp"]).dt.total_seconds() / 86400
""")
nb.code(r"""
d = delivery_days(orders)
deliveries = (orders.loc[d.index, ["order_id", "customer_id"]].assign(days=d)
                    .merge(customers[["customer_id", "customer_state"]], on="customer_id", validate="one_to_one"))
print(len(deliveries), "delivered orders with a delivery date")
deliveries["days"].describe(percentiles=[.1, .25, .5, .75, .9, .99]).round(2)
""")
nb.md(r"""
✍️ **6.1** Compute `median_days`, `p90_days` (the 90th percentile) and `share_over_30` (the share of deliveries taking more than 30 days), from `deliveries["days"]`.
""")
nb.learner("delivery-summary", r"""
median_days = ...
p90_days = ...
share_over_30 = ...

check("delivery-summary", median_days, p90_days, share_over_30)
""")
nb.md(r"""
✍️ **6.2** Write `delivery_boxplot(deliveries)`. It draws box plots of `days` for the **5 states with the most deliveries**, ordered from the **lowest to the highest median**, without fliers. It labels the y axis with its unit and returns the `Axes`.
""")
nb.learner("delivery-boxplot", r"""
def delivery_boxplot(deliveries: pd.DataFrame):
    # your code here
    raise NotImplementedError


check("delivery-boxplot", delivery_boxplot)
""")
nb.md(r"""
💬 **6.3** Write three sentences for an operations manager. Cover the typical delivery time, the tail, and which plot you'd show and why. Name the population you're describing.
""")
nb.answer("delivery-report")
nb.md(r"""
★ **6.4** *(open, no auto-check)* Overlay the ECDFs of delivery days for SP (São Paulo) and RJ (Rio de Janeiro). By what day have 90 % of each state's orders arrived? Why is an ECDF better than two histograms for this comparison?
""")
nb.learner("ecdf-compare", r"""
# ★ 6.4: your exploration here
""")

# ---------------------------------------------------------------------------
nb.md(r"""
## Wrap-up

💬 Answer briefly, in your own words.

1. A column holds the numbers 1–5. Name two scales it could be on, and one summary that's fine on one of them but not the other.
2. When would you report the mean of a skewed variable anyway?
3. Why is the IQR robust but the std isn't? Use the idea of a breakdown point.
4. Histogram, KDE, ECDF: give one advantage and one weakness of each.
5. A colleague deletes every Tukey-flagged price "to clean the data". What do you tell them?
""")
nb.answer("wrap-up", "> Your answers:")
nb.code(r"""
progress()
""")
nb.md(r"""
**Next:** `04 · Comparing groups, relating variables`. If you use Claude Code with this repository, `/tutor review m01/03` gives you feedback on your work.
""")
