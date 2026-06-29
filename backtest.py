#!/usr/bin/env python3
"""
Walk-forward pairs trading backtest.

Pairs that survived OOS cointegration check in screening.py:
  - AMP / APO  (Asset Management & Custody Banks)
  - KHC / HSY  (Packaged Foods & Meats)
  - FOXA / FOX (Dual-class shares — sanity check; should work trivially)

Design principles (same rigour as screening.py):
  - Rolling OLS hedge ratio estimated purely from the past `lookback` days.
  - Z-score derived from the same rolling window — no future data.
  - Position signal lagged 1 day before execution (no look-ahead bias).
  - Dollar-neutral sizing: $1 in Y, $|β| in X per position unit.
  - Transaction costs + slippage on both legs at every position change.
  - Risk report: CAGR, Sharpe, Sortino, max drawdown, Calmar, turnover, hit rate.
"""

from __future__ import annotations
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── Configuration ────────────────────────────────────────────────────────────

PAIRS = [
    # (y_ticker, x_ticker, half_life_days, display_label)
    ("AMP",  "APO",  23.1, "AMP / APO  [Asset Management]"),
    ("KHC",  "HSY",  27.0, "KHC / HSY  [Packaged Foods]"),
    ("FOXA", "FOX",  18.7, "FOXA / FOX [Dual-Class Sanity Check]"),
]

END   = datetime.date.today()
START = END - datetime.timedelta(days=365 * 5)

ENTRY_Z      = 2.0    # open position when |z| ≥ this
EXIT_Z       = 0.5    # close when |z| has reverted to ≤ this
COST_BPS     = 5.0    # one-way transaction cost per leg (bps)
SLIPPAGE_BPS = 2.0    # one-way slippage per leg (bps)
LOOKBACK_MIN = 60     # minimum rolling-estimation window (trading days)
LOOKBACK_MUL = 2.5    # lookback = max(LOOKBACK_MIN, MUL × half_life)

ONE_WAY_COST = (COST_BPS + SLIPPAGE_BPS) / 10_000  # fractional cost per leg


# ─── Data ─────────────────────────────────────────────────────────────────────

def fetch(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(tickers, start=START, end=END,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])
    # Drop tickers with less than 95% coverage, then forward-fill small gaps.
    # Using dropna(axis=1, thresh=...) keeps the rest of the tickers intact
    # even if one fails (avoids losing the whole DataFrame).
    min_rows = int(0.95 * len(raw))
    raw = raw.dropna(axis=1, thresh=min_rows)
    missing = set(tickers) - set(raw.columns)
    if missing:
        print(f"  [warn] tickers dropped (insufficient data): {sorted(missing)}")
    return raw.ffill().dropna()


# ─── Rolling hedge ratio and z-score ─────────────────────────────────────────

def rolling_z(
    log_y: pd.Series, log_x: pd.Series, lookback: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Rolling OLS hedge ratio and spread z-score — fully causal.

    OLS with intercept:  β = ρ(y, x) × σ_y / σ_x = Cov(y, x) / Var(x)

    Every estimate at time t uses only data on [t−lookback, t], so nothing
    from the future can influence a signal or position.
    """
    r    = log_y.rolling(lookback).corr(log_x)
    sy   = log_y.rolling(lookback).std()
    sx   = log_x.rolling(lookback).std().clip(lower=1e-10)
    beta = r * sy / sx

    spread = log_y - beta * log_x
    mu_s   = spread.rolling(lookback).mean()
    sig_s  = spread.rolling(lookback).std().clip(lower=1e-10)
    z      = (spread - mu_s) / sig_s

    return z, beta, spread


# ─── Signal generation ────────────────────────────────────────────────────────

def make_signal(z: pd.Series, entry: float, exit_z: float) -> pd.Series:
    """Stateful mean-reversion signal.

    +1  long spread  (y cheap relative to x): entered when z ≤ −entry
    −1  short spread (y dear relative to x):  entered when z ≥ +entry
     0  flat: closed when |z| has reverted to ≤ exit_z

    The loop is inherently causal — the signal at time t depends only on
    z[0..t] and the current state.
    """
    arr = np.zeros(len(z))
    cur = 0
    for i in range(len(z)):
        zi = z.iloc[i]
        if np.isnan(zi):
            arr[i] = 0
            continue
        if cur == 0:
            if   zi <= -entry: cur =  1
            elif zi >=  entry: cur = -1
        elif cur == 1:
            if zi >= -exit_z:  cur = 0
        elif cur == -1:
            if zi <=  exit_z:  cur = 0
        arr[i] = cur
    return pd.Series(arr, index=z.index, dtype=float)


# ─── P&L ──────────────────────────────────────────────────────────────────────

def compute_pnl(
    prices: pd.DataFrame,
    y: str,
    x: str,
    signal: pd.Series,
    beta: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Daily net-of-cost P&L as a fraction of $1 total capital deployed.

    Capital per position unit = $1 (y-leg) + $|β| (x-leg).
    We use the *previous day's* hedge ratio to compute today's P&L — the
    ratio in effect when the position was established.

    Transaction cost = ONE_WAY_COST × both legs on every position change:
      • |Δposition| = 1 (open or close)  →  1 × ONE_WAY_COST
      • |Δposition| = 2 (flip +1 ↔ −1)  →  2 × ONE_WAY_COST

    Returns:
        pnl     daily net P&L fraction
        lagged  actual holding (signal shifted 1 day; used for trade accounting)
    """
    lagged      = signal.shift(1).fillna(0)
    lagged_beta = beta.shift(1)                         # hedge ratio known at entry

    ry  = np.log(prices[y]).diff()
    rx  = np.log(prices[x]).diff()
    cap = (1.0 + lagged_beta.abs()).clip(lower=1e-10)

    gross    = lagged * (ry - lagged_beta * rx) / cap
    turnover = lagged.diff().abs()                      # 0, 1, or 2
    cost     = turnover * ONE_WAY_COST

    pnl = (gross - cost).rename("pnl")
    return pnl, lagged


# ─── Trade log ────────────────────────────────────────────────────────────────

def closed_trades(pnl: pd.Series, lagged: pd.Series) -> list[dict]:
    """Return a list of completed trades with their cumulative P&L.

    A trade starts when the held position changes from 0 to ±1 and ends
    when it returns to 0.  Open trades at the end of the series are excluded
    (no exit price observed).
    """
    trades, in_trade, entry_i = [], False, None
    for i in range(len(lagged)):
        pos = lagged.iloc[i]
        if not in_trade and pos != 0:
            in_trade = True
            entry_i  = i
        elif in_trade and pos == 0:
            # Include day i: position is 0 but pnl[i] carries the exit cost.
            trade_pnl = pnl.iloc[entry_i:i + 1].sum()
            trades.append({
                "entry":    lagged.index[entry_i],
                "exit":     lagged.index[i - 1],
                "pnl":      trade_pnl,
                "duration": i - entry_i,
            })
            in_trade = False
    return trades


# ─── Risk metrics ─────────────────────────────────────────────────────────────

def report(pnl: pd.Series, lagged: pd.Series, label: str) -> dict | None:
    pnl = pnl.dropna()
    if pnl.empty:
        print(f"\n  [warn] {label}: no P&L data — skipping")
        return None
    equity = (1 + pnl).cumprod()
    n_days = len(pnl)
    n_yrs  = n_days / 252

    cagr      = equity.iloc[-1] ** (1 / n_yrs) - 1
    ann_vol   = pnl.std() * np.sqrt(252)
    sharpe    = cagr / ann_vol if ann_vol > 0 else np.nan
    dn        = pnl[pnl < 0].std() * np.sqrt(252)
    sortino   = cagr / dn if dn > 0 else np.nan
    roll_max  = equity.cummax()
    dd        = (equity - roll_max) / roll_max
    max_dd    = dd.min()
    calmar    = cagr / abs(max_dd) if max_dd < 0 else np.nan

    trades    = closed_trades(pnl, lagged)
    n_trades  = len(trades)
    hit_rate  = (sum(1 for t in trades if t["pnl"] > 0) / n_trades
                 if n_trades > 0 else np.nan)
    rt_per_yr = n_trades / n_yrs
    avg_dur   = (np.mean([t["duration"] for t in trades])
                 if trades else np.nan)

    W = 20
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(f"  {'Period':<{W}} {pnl.index[0].date()} → {pnl.index[-1].date()}")
    print(f"  {'CAGR':<{W}} {cagr*100:+.1f}%")
    print(f"  {'Ann. vol':<{W}} {ann_vol*100:.1f}%")
    print(f"  {'Sharpe':<{W}} {sharpe:.2f}")
    print(f"  {'Sortino':<{W}} {sortino:.2f}")
    print(f"  {'Max drawdown':<{W}} {max_dd*100:.1f}%")
    print(f"  {'Calmar':<{W}} {calmar:.2f}")
    print(f"  {'Round-trips / yr':<{W}} {rt_per_yr:.1f}")
    print(f"  {'Trades (total)':<{W}} {n_trades}")
    print(f"  {'Avg hold (days)':<{W}} {avg_dur:.0f}" if not np.isnan(avg_dur) else f"  {'Avg hold (days)':<{W}} —")
    if not np.isnan(hit_rate):
        print(f"  {'Hit rate':<{W}} {hit_rate*100:.0f}%")
    else:
        print(f"  {'Hit rate':<{W}} —")

    return dict(
        label=label, cagr=cagr, ann_vol=ann_vol, sharpe=sharpe,
        sortino=sortino, max_dd=max_dd, calmar=calmar,
        rt_per_yr=rt_per_yr, n_trades=n_trades, hit_rate=hit_rate,
        avg_hold=avg_dur, equity=equity, pnl=pnl, drawdown=dd,
    )


# ─── Charts ───────────────────────────────────────────────────────────────────

def plot_results(
    results: list[dict],
    spy_equity: pd.Series,
    outfile: str = "backtest_equity.png",
) -> None:
    n   = len(results)
    fig = plt.figure(figsize=(13, 4.5 * n))
    gs  = fig.add_gridspec(n, 1, hspace=0.45)

    for row, r in enumerate(results):
        ax  = fig.add_subplot(gs[row])
        eq  = r["equity"].dropna()
        spy = spy_equity.reindex(eq.index).ffill().bfill()
        spy = spy / spy.iloc[0]

        ax.plot(eq.index,  eq.values,  lw=1.5, label=r["label"])
        ax.plot(spy.index, spy.values, lw=1,   alpha=0.5, color="grey",
                linestyle="--", label="SPY (rebased to 1)")
        ax.axhline(1, color="black", lw=0.4, linestyle=":")
        ax.fill_between(eq.index, eq.values, 1,
                        where=(eq.values < 1), alpha=0.12, color="red",
                        label="_nolegend_")

        hr_str = (f"  Hit {r['hit_rate']*100:.0f}%"
                  if not np.isnan(r["hit_rate"]) else "")
        ax.set_title(
            f"{r['label']}   "
            f"CAGR {r['cagr']*100:+.1f}%   "
            f"Sharpe {r['sharpe']:.2f}   "
            f"MaxDD {r['max_dd']*100:.1f}%"
            f"{hr_str}",
            fontsize=9,
        )
        ax.legend(fontsize=8)
        ax.set_ylabel("Growth of $1")

    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\nSaved equity chart → {outfile}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_tickers = sorted({t for y, x, *_ in PAIRS for t in (y, x)} | {"SPY"})
    print(f"Downloading {len(all_tickers)} tickers: {all_tickers}")
    print(f"Period: {START} → {END}\n")
    prices = fetch(all_tickers)
    log_p  = np.log(prices)

    spy_ret    = log_p["SPY"].diff().dropna()
    spy_equity = (1 + spy_ret).cumprod()

    results = []
    for y, x, hl, label in PAIRS:
        if y not in prices.columns or x not in prices.columns:
            print(f"\n── {label}  [SKIPPED — missing ticker data]")
            continue
        lookback = max(LOOKBACK_MIN, int(LOOKBACK_MUL * hl))
        print(f"\n── {label}")
        print(f"   lookback = {lookback} days  (= {LOOKBACK_MUL}× {hl:.0f}d half-life)")

        z, beta, _spread = rolling_z(log_p[y], log_p[x], lookback)
        sig              = make_signal(z, ENTRY_Z, EXIT_Z)
        pnl, lagged      = compute_pnl(prices, y, x, sig, beta)
        res              = report(pnl, lagged, label)
        if res is not None:
            results.append(res)

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n\n{'═' * 72}")
    print("  Summary")
    print(f"{'═' * 72}")
    hdr = (f"  {'Pair':<33} {'CAGR':>6}  {'Shrp':>5}  "
           f"{'Sort':>5}  {'MaxDD':>6}  {'Hit%':>5}  {'#Tr':>4}")
    print(hdr)
    print(f"  {'─'*33}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*5}  {'─'*4}")
    for r in results:
        hr = f"{r['hit_rate']*100:.0f}%" if not np.isnan(r["hit_rate"]) else "  —"
        print(f"  {r['label']:<33} {r['cagr']*100:>+5.1f}%  "
              f"{r['sharpe']:>5.2f}  {r['sortino']:>5.2f}  "
              f"{r['max_dd']*100:>5.1f}%  {hr:>5}  {r['n_trades']:>4}")

    plot_results(results, spy_equity)
