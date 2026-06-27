# Pairs Trading — Methodology & Design Rationale

This document explains *how* the pair-screening pipeline was built and, more
importantly, *why* each design decision was made. The goal of the project is
not to present a polished P&L curve — it is to demonstrate a disciplined,
statistically honest research process for finding tradeable cointegrated pairs.

---

## 1. The idea, and why the naïve version fails

A pairs trade bets that the *spread* between two economically related assets is
**mean-reverting**: when it stretches far from its average, you short the rich
leg and buy the cheap leg, and profit as it converges. The entire edge rests on
one assumption — that the spread is stationary (mean-reverting) rather than a
random walk.

The project began with the textbook example: **Coca-Cola (KO) vs Pepsi (PEP)**.
The first result killed that idea immediately:

```
Cointegration p-value for KO/PEP (last 4 years): 0.6694
Hedge ratio (KO/PEP): -0.6182
```

A p-value of **0.67** means we cannot reject the null hypothesis that the spread
is a random walk — KO/PEP have **not** been cointegrated. The negative hedge
ratio is a second red flag for two stocks that should move together. Reporting a
Sharpe ratio on this pair (the original notebook showed 0.35 from 2.5 trades)
would be meaningless: a performance number computed on a pair with no
statistical edge, from a sample far too small to be significant.

**Conclusion:** the interesting research question is not "how do I trade KO/PEP?"
It is **"how do I systematically find pairs that genuinely cointegrate — and how
do I avoid fooling myself when I do?"** That reframing motivated the screening
pipeline.

---

## 2. Design decisions in the screening pipeline

The pipeline lives in [`screening.py`](screening.py). Each choice below was made
to maximize statistical rigor, not convenience.

### 2.1 Cointegration, not correlation
Correlation measures co-movement of *returns*; it says nothing about whether the
*price spread* stays bounded. Two assets can be highly correlated while their
spread drifts apart forever. We therefore test for **cointegration** (a stable
long-run price relationship) using the **Engle–Granger** test, and use return
correlation only as a cheap pre-filter (Section 2.6).

### 2.2 Log prices
All work is done on **log prices** ([`download_log_prices`](screening.py#L112)).
Log prices give returns that are additive over time, make the hedge-ratio
regression scale-invariant to price level, and are the standard input for
cointegration testing.

### 2.3 Economic grouping by GICS Sub-Industry
Pairs are only tested **within the same GICS Sub-Industry**
([`build_sp500_universe`](screening.py#L50)). This is deliberate:

- **Economic rationale.** A cointegrating relationship should have a *reason* to
  exist — shared input costs, demand drivers, regulation. Two firms in the same
  sub-industry (e.g. Kraft Heinz / Hershey) plausibly share one; a random
  cross-sector pair that "cointegrates" is far more likely a statistical fluke.
- **It controls the multiple-testing problem.** Testing all
  C(500, 2) ≈ 125,000 pairs would guarantee thousands of false positives.
  Restricting to within-group pairs cuts the test count by ~50×.

### 2.4 Train/test split — the most important safeguard
Every screening statistic (hedge ratio, cointegration p-value, ADF, half-life)
is computed on a **70% training window**; the final 30% is **held out**
([`screen_universe`](screening.py#L227)). For each pair we then re-run the
cointegration test on the unseen window (`coint_p_test`).

This directly attacks the central failure mode of this kind of research:
**in-sample cointegration is easy to find by chance and frequently does not
persist.** A pair is only interesting if the relationship holds *out-of-sample*.

### 2.5 Both regression directions
Engle–Granger is **not symmetric**: regressing `A ~ B` gives a different p-value
than `B ~ A`. [`test_pair`](screening.py#L164) runs both and keeps the stronger
direction, recording which leg is dependent (`y`) and which is the hedge (`x`).

### 2.6 Correlation pre-filter
Before running the (relatively expensive) cointegration and ADF tests, pairs
with `|return correlation| < 0.5` are skipped. Assets with no co-movement are
extremely unlikely to cointegrate, so this saves computation without materially
changing results.

### 2.7 Half-life of mean reversion
For each surviving pair we estimate the **half-life** via an Ornstein–Uhlenbeck /
AR(1) fit on the spread ([`half_life`](screening.py#L138)):

```
Δspread_t = a + b · spread_{t-1} + ε_t      (mean reversion ⇒ b < 0)
half-life = −ln(2) / b
```

This is both a **filter** and a **design input**: pairs are kept only when the
half-life is in `[1, 60]` trading days. Too short implies microstructure
noise rather than a tradeable signal; too long means capital is tied up waiting
for a convergence that may never come. The half-life also tells you the *correct
lookback window* for the rolling z-score in the eventual backtest.

### 2.8 Multiple-testing correction (Bonferroni)
Screening *N* pairs at α = 0.05 produces ≈ 0.05·N false positives by chance.
The pipeline counts every test it runs and applies a **Bonferroni** threshold
(`p < α / N`), flagging each pair with `coint_p_bonferroni_ok`
([`screen_universe`](screening.py#L247)). Bonferroni is conservative, but it is
simple to defend and makes the data-snooping risk explicit rather than hidden.

### 2.9 Composite ranking
Surviving pairs are ranked by the average of three ranks — cointegration
p-value, spread ADF p-value, and half-life — so no single metric dominates the
ordering.

---

## 3. The research narrative (what actually happened)

The process was iterative and is documented honestly because the *dead ends are
part of the rigor*:

1. **Started with KO/PEP.** Failed the cointegration test (p = 0.67). Rejected.
2. **Built a small hand-curated screen** (8 sectors, 91 pairs). Six pairs passed
   the in-sample filter — but **none survived Bonferroni**, and every one had an
   out-of-sample p-value between 0.17 and 0.97. The in-sample hits were noise.
3. **Expanded the universe** to the full S&P 500 grouped by GICS Sub-Industry
   (~500 tickers, ~120 groups) to get a large enough search for genuine
   survivors.

---

## 4. Results

From the full S&P 500 screen (5 years of data, 70/30 train/test):

| Filter | Count |
| --- | --- |
| Passed in-sample tradeability filter | 68 |
| Survived Bonferroni correction | **0** |
| Held up **out-of-sample** (`coint_p_test < 0.05`) | **3** |

The three pairs that held out-of-sample:

| Pair | Sub-Industry | Train p | OOS p | Note |
| --- | --- | --- | --- | --- |
| FOXA / FOX | Broadcasting | 0.023 | 0.008 | corr 0.99 — **dual-class shares of the same company**. A sanity check that the screen finds real cointegration, but not a tradeable strategy after costs. |
| KHC / HSY | Packaged Foods | 0.037 | 0.026 | Kraft Heinz / Hershey — a genuine candidate. |
| AMP / APO | Asset Management | 0.017 | 0.031 | Ameriprise / Apollo — a genuine candidate. |

**Interpretation.** Two things stand out, and both are *features*, not failures:

- The screen self-validates by surfacing **FOXA/FOX**, the most obviously
  cointegrated pair in the market (two share classes of one firm).
- **Zero pairs survive Bonferroni**, and only 3 of 68 hold out-of-sample. This
  is a concrete demonstration that the large majority of in-sample cointegration
  is multiple-testing noise — exactly the conclusion a rigorous researcher
  should reach, and one that a cherry-picked Sharpe ratio would have hidden.

---

## 5. Honest limitations

- **Bonferroni is overly conservative.** A false-discovery-rate control
  (Benjamini–Hochberg) would be a more powerful next step than the current
  family-wise correction.
- **Static hedge ratio.** The hedge ratio is a single OLS fit on the training
  window. Real relationships drift; a **Kalman filter** for a time-varying
  hedge ratio is the natural upgrade.
- **Survivorship bias.** The universe is the *current* S&P 500. Companies that
  were delisted or removed are excluded, which biases historical results.
- **No transaction costs yet.** Screening identifies candidates; it does not
  prove profitability. Costs, slippage, and borrow fees are decisive for a
  market-neutral strategy and belong in the backtest.

---

## 6. Next steps

1. **Walk-forward backtest** of the surviving candidates (KHC/HSY, AMP/APO)
   using a **rolling z-score** (lookback set from the estimated half-life) and a
   lagged position to avoid look-ahead.
2. **Realistic costs**: per-trade commission + slippage, dollar-neutral position
   sizing, and a benchmark comparison (SPY).
3. **Full risk report**: max drawdown, Sortino, Calmar, turnover, hit rate —
   not Sharpe alone.
4. **Dynamic hedge ratio** via Kalman filter and a Benjamini–Hochberg FDR
   screen.

---

## How to run

```bash
python3 screening.py        # full S&P 500 screen → screened_pairs.csv
```

Swap `build_sp500_universe(...)` for `DEMO_UNIVERSE` in the `__main__` block for
a fast run on the small hand-curated universe.