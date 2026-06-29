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
DELTA        = 0.0001 # Kalman process-noise parameter; controls β tracking speed
              # ≈ 1/sqrt(delta) ≈ 100 days effective lookback at delta=0.0001

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


# ─── Kalman filter hedge ratio and z-score ───────────────────────────────────

def kalman_z(
    log_y: pd.Series,
    log_x: pd.Series,
    delta: float,
    lookback: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Kalman filter hedge ratio and spread z-score — fully causal.

    State: θ = [α, β], modelled as a random walk (both drift slowly over time).
    Observation: y_t = α_t + β_t · x_t + ε_t

    The process-noise covariance Q = delta/(1−delta) · I controls tracking
    speed: large delta → fast/reactive β; small delta → slow/smooth β.
    Observation noise R is estimated online via an EWMA of squared innovations.

    The z-score is the pre-update innovation (y_t − ŷ_{t|t−1}) normalised by
    its rolling std — a point-in-time, causal signal.  β from the filter is
    used for dollar-neutral position sizing just like rolling OLS.
    """
    y_arr = log_y.values
    x_arr = log_x.values
    n     = len(y_arr)

    theta = np.zeros(2)         # [alpha, beta]; initialised at zero
    P     = np.eye(2) * 1.0     # state covariance; grows quickly via Q
    Q     = (delta / (1.0 - delta)) * np.eye(2)

    innovations = np.full(n, np.nan)
    betas       = np.full(n, np.nan)
    R           = None           # estimated online from the first residual

    for t in range(n):
        if np.isnan(y_arr[t]) or np.isnan(x_arr[t]):
            continue

        H = np.array([[1.0, x_arr[t]]])   # shape (1, 2)

        # Predict
        P_pred = P + Q                     # (2, 2); F = I so F·P·F^T = P

        # Pre-update innovation: ŷ uses last period's state estimate
        e = y_arr[t] - (H @ theta).item()

        # Update observation noise R via EWMA of squared residuals
        R = e ** 2 if R is None else 0.97 * R + 0.03 * e ** 2

        # Kalman gain: K = P_pred · H^T / (H · P_pred · H^T + R)
        S = (H @ P_pred @ H.T).item() + R   # scalar innovation variance
        K = (P_pred @ H.T) / S            # (2, 1)

        # Update state
        theta = theta + K.flatten() * e
        P     = (np.eye(2) - K @ H) @ P_pred

        innovations[t] = e
        betas[t]       = theta[1]

    innovations = pd.Series(innovations, index=log_y.index)
    betas       = pd.Series(betas,       index=log_y.index)

    # Z-score: normalise innovations by rolling std (same window as rolling_z)
    sig_i  = innovations.rolling(lookback).std().clip(lower=1e-10)
    z      = innovations / sig_i

    # Spread for P&L: log_y − β·log_x  (α is already absorbed into the signal)
    spread = log_y - betas * log_x

    return z, betas, spread


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
    pairs_results: list[tuple[str, dict | None, dict | None]],
    spy_equity: pd.Series,
    outfile: str = "backtest_equity.png",
) -> None:
    """One subplot per pair; Rolling OLS and Kalman overlaid on the same axes."""
    n   = len(pairs_results)
    fig = plt.figure(figsize=(13, 4.5 * n))
    gs  = fig.add_gridspec(n, 1, hspace=0.45)

    colors = {"Rolling OLS": "steelblue", "Kalman": "darkorange"}

    for row, (pair_label, res_r, res_k) in enumerate(pairs_results):
        ax = fig.add_subplot(gs[row])

        ref_eq = None

        for method, res, color in [("Rolling OLS", res_r, colors["Rolling OLS"]),
                                    ("Kalman",      res_k, colors["Kalman"])]:
            if res is None:
                continue
            eq = res["equity"].dropna()
            if ref_eq is None:
                ref_eq = eq
            hr = f" Hit {res['hit_rate']*100:.0f}%" if not np.isnan(res["hit_rate"]) else ""
            lbl = (f"{method}  CAGR {res['cagr']*100:+.1f}%  "
                   f"Sh {res['sharpe']:.2f}  DD {res['max_dd']*100:.1f}%{hr}")
            ax.plot(eq.index, eq.values, lw=1.5, color=color, label=lbl)

        if ref_eq is not None:
            spy = spy_equity.reindex(ref_eq.index).ffill().bfill()
            spy = spy / spy.iloc[0]
            ax.plot(spy.index, spy.values, lw=1, alpha=0.45, color="grey",
                    linestyle="--", label="SPY (rebased)")

        ax.axhline(1, color="black", lw=0.4, linestyle=":")
        ax.set_title(pair_label, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7.5)
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

    # pairs_results: list of (pair_label, rolling_result, kalman_result)
    pairs_results: list[tuple[str, dict | None, dict | None]] = []

    for y, x, hl, label in PAIRS:
        if y not in prices.columns or x not in prices.columns:
            print(f"\n── {label}  [SKIPPED — missing ticker data]")
            continue
        lookback = max(LOOKBACK_MIN, int(LOOKBACK_MUL * hl))
        print(f"\n{'━' * 60}")
        print(f"  {label}   (lookback = {lookback} d)")
        print(f"{'━' * 60}")

        # Rolling OLS
        z_r, beta_r, _ = rolling_z(log_p[y], log_p[x], lookback)
        sig_r           = make_signal(z_r, ENTRY_Z, EXIT_Z)
        pnl_r, lag_r    = compute_pnl(prices, y, x, sig_r, beta_r)
        res_r           = report(pnl_r, lag_r, f"{label}  [Rolling OLS]")

        # Kalman filter
        z_k, beta_k, _ = kalman_z(log_p[y], log_p[x], DELTA, lookback)
        sig_k           = make_signal(z_k, ENTRY_Z, EXIT_Z)
        pnl_k, lag_k    = compute_pnl(prices, y, x, sig_k, beta_k)
        res_k           = report(pnl_k, lag_k, f"{label}  [Kalman δ={DELTA}]")

        pairs_results.append((label, res_r, res_k))

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n\n{'═' * 76}")
    print("  Summary — Rolling OLS vs Kalman filter")
    print(f"{'═' * 76}")
    hdr = (f"  {'Method':<42} {'CAGR':>6}  {'Shrp':>5}  "
           f"{'MaxDD':>6}  {'Hit%':>5}  {'#Tr':>4}")
    print(hdr)
    print(f"  {'─'*42}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*5}  {'─'*4}")
    for pair_label, res_r, res_k in pairs_results:
        for res in (res_r, res_k):
            if res is None:
                continue
            hr = f"{res['hit_rate']*100:.0f}%" if not np.isnan(res["hit_rate"]) else "—"
            print(f"  {res['label']:<42} {res['cagr']*100:>+5.1f}%  "
                  f"{res['sharpe']:>5.2f}  {res['max_dd']*100:>5.1f}%  "
                  f"{hr:>5}  {res['n_trades']:>4}")
        print()

    plot_results(pairs_results, spy_equity)
