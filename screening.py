# =============================================================================
# Pairs Screening Pipeline
# -----------------------------------------------------------------------------
# Searches a universe of stocks for tradeable cointegrated pairs.
#
# Methodology (the parts that matter for quant research):
#   1. Only test pairs WITHIN the same sector -> economic rationale, and it
#      drastically cuts the number of tests (which controls false positives).
#   2. All screening statistics are computed on a TRAINING window only. The
#      test window is held out so we can check whether cointegration *persists*
#      out-of-sample -- the single most important robustness check.
#   3. Rank by a blend of: Engle-Granger cointegration p-value, spread
#      stationarity (ADF), and half-life of mean reversion.
#   4. Explicitly correct for multiple testing (Bonferroni) and report how many
#      tests were run, because screening N pairs inflates Type I error.
# =============================================================================

from __future__ import annotations

import io
import itertools
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller


# -----------------------------------------------------------------------------
# Small hand-curated universe -- handy for a fast demo run without scraping.
# For the real screen use build_sp500_universe() below.
# -----------------------------------------------------------------------------
DEMO_UNIVERSE: dict[str, list[str]] = {
    "beverages":   ["KO", "PEP", "MNST", "KDP", "STZ"],
    "mega_tech":   ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN"],
    "banks":       ["JPM", "BAC", "WFC", "C", "USB", "PNC"],
    "oil_majors":  ["XOM", "CVX", "COP", "SLB", "EOG"],
    "retail":      ["WMT", "TGT", "COST", "DG", "DLTR"],
    "payments":    ["V", "MA", "AXP", "PYPL", "FIS"],
    "semis":       ["NVDA", "AMD", "INTC", "TXN", "QCOM", "MU"],
    "autos":       ["GM", "F", "TSLA", "TM"],
}

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def build_sp500_universe(
    level: str = "GICS Sub-Industry",
    min_members: int = 2,
    max_members: int | None = 20,
) -> dict[str, list[str]]:
    """Build a sector-grouped universe from current S&P 500 constituents.

    Groups by a GICS classification scraped from Wikipedia. Defaults to
    "GICS Sub-Industry" -- the tightest economic grouping, so pairs share a
    genuine fundamental driver (the strongest defence against spurious
    cointegration). Use "GICS Sector" for a broader, larger universe.

    Args:
        level: which column to group on ("GICS Sub-Industry" or "GICS Sector").
        min_members: drop groups too small to form a pair.
        max_members: cap group size to keep the test count (and Bonferroni
            penalty) manageable; None for no cap. Largest groups are trimmed
            by ticker name only as a deterministic, reproducible rule.
    """
    html = requests.get(_SP500_URL, headers={"User-Agent": "Mozilla/5.0"},
                        timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]

    # yfinance uses '-' where Wikipedia uses '.' (e.g. BRK.B -> BRK-B).
    table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)

    universe: dict[str, list[str]] = {}
    for group, rows in table.groupby(level):
        tickers = sorted(rows["Symbol"].unique())
        if len(tickers) < min_members:
            continue
        if max_members is not None:
            tickers = tickers[:max_members]
        # sanitise group name for use as a clean key
        key = str(group).lower().replace(" ", "_").replace("&", "and")
        universe[key] = tickers

    n_pairs = sum(len(t) * (len(t) - 1) // 2 for t in universe.values())
    print(f"[universe] {len(universe)} '{level}' groups, "
          f"{sum(len(t) for t in universe.values())} tickers, "
          f"{n_pairs} candidate pairs")
    return universe


@dataclass
class PairResult:
    """All screening statistics for a single candidate pair."""
    sector: str
    y: str                 # dependent leg of the cointegrating regression
    x: str                 # independent leg
    corr: float            # correlation of daily log returns (train)
    hedge_ratio: float     # beta from OLS(y_log ~ x_log) on train
    coint_p_train: float   # Engle-Granger p-value, in-sample
    adf_p_train: float     # ADF p-value on the spread, in-sample
    half_life: float       # mean-reversion half-life in trading days
    coint_p_test: float    # Engle-Granger p-value, OUT-of-sample (held out)
    n_obs: int             # observations in the training window


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def download_log_prices(tickers: list[str], start, end) -> pd.DataFrame:
    """Download adjusted closes and return log prices, columns = tickers.

    Tickers that fail to download or are mostly empty are dropped with a note.
    """
    tickers = sorted(set(tickers))
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]

    # yfinance returns a Series for a single ticker; normalise to DataFrame.
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])

    # Drop tickers with too much missing data, then forward-fill small gaps.
    good = raw.columns[raw.notna().mean() > 0.95]
    dropped = set(raw.columns) - set(good)
    if dropped:
        print(f"  [data] dropped (insufficient history): {sorted(dropped)}")
    raw = raw[good].ffill().dropna()

    return np.log(raw)


# -----------------------------------------------------------------------------
# Per-pair statistics
# -----------------------------------------------------------------------------
def half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion via an OU / AR(1) fit.

    Discretised OU:  d(spread)_t = a + b * spread_{t-1} + e_t
    Mean reversion requires b < 0; half-life = -ln(2) / b.
    Returns np.inf when the spread is not mean-reverting (b >= 0).
    """
    lag = spread.shift(1)
    delta = spread - lag
    df = pd.concat([delta, lag], axis=1).dropna()
    df.columns = ["delta", "lag"]

    beta = sm.OLS(df["delta"], sm.add_constant(df["lag"])).fit().params["lag"]
    if beta >= 0:
        return np.inf
    return float(-np.log(2) / beta)


def _spread(y_log: pd.Series, x_log: pd.Series) -> tuple[pd.Series, float]:
    """OLS hedge ratio and resulting spread for y ~ x."""
    model = sm.OLS(y_log, sm.add_constant(x_log)).fit()
    beta = float(model.params.iloc[1])
    spread = y_log - beta * x_log
    return spread, beta


def test_pair(sector: str, a: str, b: str,
              train: pd.DataFrame, test: pd.DataFrame) -> PairResult | None:
    """Run the full screen on one pair, choosing the better regression direction.

    Engle-Granger is not symmetric in (y, x): which leg is the dependent
    variable changes the p-value. We try both and keep the stronger direction.
    """
    best = None
    for y, x in ((a, b), (b, a)):
        ytr, xtr = train[y], train[x]

        coint_p = coint(ytr, xtr)[1]
        spread, beta = _spread(ytr, xtr)
        adf_p = adfuller(spread, autolag="AIC")[1]
        hl = half_life(spread)

        # correlation of daily log returns over the training window
        corr = float(train[y].diff().corr(train[x].diff()))

        # out-of-sample cointegration on the held-out window
        coint_p_oos = coint(test[y], test[x])[1]

        res = PairResult(
            sector=sector, y=y, x=x, corr=corr, hedge_ratio=beta,
            coint_p_train=coint_p, adf_p_train=adf_p, half_life=hl,
            coint_p_test=coint_p_oos, n_obs=len(ytr),
        )
        # keep the direction with the lower in-sample cointegration p-value
        if best is None or res.coint_p_train < best.coint_p_train:
            best = res
    return best


# -----------------------------------------------------------------------------
# Screen the whole universe
# -----------------------------------------------------------------------------
def screen_universe(
    universe: dict[str, list[str]],
    start, end,
    train_frac: float = 0.7,
    min_abs_corr: float = 0.5,
    max_coint_p: float = 0.05,
    min_half_life: float = 1.0,
    max_half_life: float = 60.0,
) -> pd.DataFrame:
    """Screen every within-sector pair and return a ranked, filtered table.

    Filters applied (all on the TRAINING window):
      - |return correlation| >= min_abs_corr  (cheap economic pre-filter)
      - Bonferroni-adjusted cointegration p-value <= max_coint_p
      - half-life within [min_half_life, max_half_life] trading days
        (too short = noise/microstructure, too long = untradeable)
    """
    results: list[PairResult] = []
    n_tests = 0

    for sector, tickers in universe.items():
        print(f"\n[sector] {sector}: {tickers}")
        log_px = download_log_prices(tickers, start, end)
        if log_px.shape[1] < 2:
            print("  [skip] fewer than 2 usable tickers")
            continue

        split = int(len(log_px) * train_frac)
        train, test = log_px.iloc[:split], log_px.iloc[split:]

        for a, b in itertools.combinations(log_px.columns, 2):
            n_tests += 1
            # cheap pre-filter first to avoid running coint on unrelated pairs
            corr = train[a].diff().corr(train[b].diff())
            if abs(corr) < min_abs_corr:
                continue
            res = test_pair(sector, a, b, train, test)
            if res is not None:
                results.append(res)

    if not results:
        print("\nNo pairs survived the correlation pre-filter.")
        return pd.DataFrame()

    df = pd.DataFrame([asdict(r) for r in results])

    # --- multiple-testing correction ---------------------------------------
    # Screening many pairs inflates Type I error. Bonferroni is conservative
    # but easy to defend: require p < alpha / n_tests.
    bonferroni_threshold = max_coint_p / max(n_tests, 1)
    df["coint_p_bonferroni_ok"] = df["coint_p_train"] < bonferroni_threshold

    # --- final tradeability filter -----------------------------------------
    keep = (
        (df["coint_p_train"] <= max_coint_p)
        & df["half_life"].between(min_half_life, max_half_life)
    )
    df = df[keep].copy()

    # --- composite rank: lower coint p, lower adf p, shorter half-life ------
    # Rank each metric and average; small rank = better candidate.
    df["rank_score"] = (
        df["coint_p_train"].rank()
        + df["adf_p_train"].rank()
        + df["half_life"].rank()
    )
    df = df.sort_values("rank_score").reset_index(drop=True)

    print(f"\n{'='*70}")
    print(f"Tested {n_tests} pairs across {len(universe)} sectors.")
    print(f"Bonferroni threshold (alpha=0.05): p < {bonferroni_threshold:.2e}")
    print(f"{keep.sum()} pairs passed the tradeability filter.")
    print(f"{'='*70}")

    return df


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import datetime

    END = datetime.date.today()
    START = END - datetime.timedelta(days=365 * 5)   # 5 years of history

    # Full S&P 500 screen, grouped by GICS Sub-Industry. Swap for
    # DEMO_UNIVERSE for a fast offline-ish run, or level="GICS Sector"
    # for a broader (much larger) search.
    universe = build_sp500_universe(level="GICS Sub-Industry",
                                    min_members=2, max_members=20)

    ranked = screen_universe(universe, START, END)

    if not ranked.empty:
        cols = ["sector", "y", "x", "corr", "hedge_ratio", "coint_p_train",
                "adf_p_train", "half_life", "coint_p_test",
                "coint_p_bonferroni_ok"]
        pd.set_option("display.width", 140)
        pd.set_option("display.max_columns", None)
        print("\nTop candidate pairs (ranked):\n")
        print(ranked[cols].round(4).to_string(index=False))

        ranked.to_csv("screened_pairs.csv", index=False)
        print("\nSaved full results -> screened_pairs.csv")
