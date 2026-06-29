# Session Handoff — Pairs Trading Project

> Paste this to a new Claude session (or just point it at this file) to resume
> with full context. Last updated: 2026-06-29.

## Who / goal
The user is building a **pairs trading project to impress recruiters for
quantitative research roles**. The guiding principle agreed in this project:
**statistical honesty and research rigor matter more than a flattering P&L
curve.** Present dead ends and failed tests as evidence of discipline.

## Where the project started
- Original code: [`pairs.py`](pairs.py) and [`Pairs.ipynb`](Pairs.ipynb) — a
  textbook KO/PEP pairs strategy (download prices → cointegration test → OLS
  hedge ratio → spread → z-score → threshold signals → backtest → Sharpe).
- **Key problem found:** KO/PEP are **not cointegrated** (Engle–Granger
  p = 0.67), and the hedge ratio was negative. The reported Sharpe (0.35 from
  2.5 trades) was meaningless — a performance number on a pair with no edge and
  far too small a sample.
- The notebook had already fixed an earlier look-ahead bias (uses
  `position.shift(1)` in the PnL calc); `pairs.py` still has the unlagged
  version.
- This reframed the project: the real question is **"how do I systematically
  find pairs that genuinely cointegrate without fooling myself?"** → led to
  building a screening pipeline.

## What we built this session
### 1. Screening pipeline — [`screening.py`](screening.py) (main deliverable)
Screens within-sector pairs and ranks tradeable cointegrated candidates.
Design decisions (all intentional, for rigor):
- **Cointegration (Engle–Granger), not correlation** — correlation ignores
  whether the price spread stays bounded. Return correlation used only as a
  cheap pre-filter (|corr| ≥ 0.5).
- **Log prices** throughout.
- **GICS Sub-Industry grouping** via `build_sp500_universe()` — gives economic
  rationale AND controls multiple testing (vs ~125k all-pairs tests).
- **70/30 train/test split** — all stats computed in-sample; cointegration
  re-tested out-of-sample (`coint_p_test`). This is the key safeguard.
- **Both regression directions** tested (Engle–Granger is asymmetric); keep the
  stronger.
- **Half-life** of mean reversion via OU/AR(1) fit; keep pairs in [1, 60] days.
- **Bonferroni** multiple-testing correction; flagged as `coint_p_bonferroni_ok`.
- **Composite ranking** = avg rank of (coint p, ADF p, half-life).
- Scrapes live S&P 500 constituents from Wikipedia via `requests` + a
  User-Agent header (pandas' direct `read_html` hit an SSL error).
- Outputs `screened_pairs.csv`.

### 2. Documentation — [`METHODOLOGY.md`](METHODOLOGY.md)
Recruiter-facing writeup of the process, design rationale, results, limitations,
and next steps. Links into `screening.py` by line.

### 3. [`README.md`](README.md)
Contains the user's own notes — **left intentionally untouched.** (Could add a
pointer to METHODOLOGY.md.)

## Results from the full S&P 500 screen (5y data, 70/30 split)
- 68 pairs passed the in-sample tradeability filter.
- **0 survived Bonferroni** correction.
- **3 held out-of-sample** (`coint_p_test < 0.05`):
  - **FOXA/FOX** — corr 0.99, but dual-class shares of the *same company*; a
    sanity check the screen works, not a tradeable strategy.
  - **KHC/HSY** (Kraft Heinz / Hershey) — genuine candidate.
  - **AMP/APO** (Ameriprise / Apollo) — genuine candidate.
- Framing: "0 survive Bonferroni, 3 of 68 hold OOS" is the honest, impressive
  result — most in-sample cointegration is multiple-testing noise.

## Environment notes (important for the new laptop)
- Packages (pandas, numpy, yfinance 0.2.66, statsmodels 0.14.5, requests, lxml)
  are on **system Python 3.11**, NOT in `.venv` (the repo's `.venv` is empty).
  Run with `python3 screening.py`. May want to `pip install` into the venv.
- Run command: `python3 screening.py` → writes `screened_pairs.csv`.

## What we built this session (session 2)

### 4. Walk-forward backtest — [`backtest.py`](backtest.py)

Design:
- **Rolling OLS** β = ρ(y,x) × σ_y / σ_x estimated over `max(60, 2.5 × half-life)` days, purely from past data.
- **Z-score** from rolling spread mean/std (same window).
- **Signal lagged 1 day** — no look-ahead.
- **Dollar-neutral** sizing: $1 in Y, $|β| in X per unit; P&L normalised to $(1+|β|).
- **Costs**: 7 bps one-way per leg (5bp commission + 2bp slippage), charged on both legs at every position change.
- Uses **previous day's β** for P&L calculation (the ratio actually in effect).
- Risk report: CAGR, Sharpe, Sortino, MaxDD, Calmar, round-trips/yr, trades, avg hold, hit rate.
- Outputs `backtest_equity.png` (equity curves vs SPY).

### Backtest results (5y ending 2026-06-29, venv Python 3.9)

| Pair | CAGR | Sharpe | Sortino | MaxDD | Hit% | Trades |
|------|------|--------|---------|-------|------|--------|
| AMP/APO [Asset Mgmt] | +4.0% | 0.42 | 0.39 | −12.6% | 15% | 13 |
| KHC/HSY [Packaged Foods] | +2.4% | 0.24 | 0.24 | −16.3% | 38% | 13 |
| FOXA/FOX [Dual-Class sanity] | +0.1% | 0.05 | 0.05 | −2.9% | 25% | 12 |

Key insights for recruiter narrative:
- AMP/APO low hit rate (15%) consistent with barely-cointegrated pair — few big wins, many small losses.
- FOXA/FOX near-zero despite being essentially the same company: tight spread + 7bps cost = no edge.
- All Sharpes < 0.5, confirming the honest story: OOS cointegration is fragile.

## Environment notes
- All packages now in `.venv` (Python 3.9). Run with `python3 screening.py` or `python3 backtest.py`.
- `requirements.txt` is up to date (from `pip freeze`).

## Next steps (not yet built)
- Dynamic hedge ratio via Kalman filter (would smooth β and likely improve results).
- Benjamini–Hochberg FDR as a less conservative alternative to Bonferroni in screening.
- Stop-loss / position sizing (Kelly or vol-targeting).
