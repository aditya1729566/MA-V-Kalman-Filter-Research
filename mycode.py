# ================================
# MA vs V — Version 7.3 (10x-leverage-aware, per-year + combined tests)
# - Kalman hedge ratio
# - Slippage / impact / borrow costs
# - ADV-cap and leverage enforcement (MAX_LEVERAGE = 10)
# - Run for multiple years separately and combined
# - Colab/Jupyter single cell
# ================================

# Installs (Colab)
!pip install yfinance pandas numpy scipy statsmodels --quiet

import math, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings("ignore")

# -------------------------
# USER PARAMETERS (edit these)
# -------------------------
TICKER_X = "MA"
TICKER_Y = "V"
START_DATE = "2005-01-01"   # full history start for Kalman estimation
END_DATE = None             # latest if None

LOOKBACK_Z = 60
KF_DELTA = 1e-5

# Pyramid thresholds — keep as in 7.2 but flexible
PYRAMID_Z = [1.25, 1.75, 2.5]
ZS_EXIT = 0.3
STOP_LOSS_Z = 8.0

# Execution / liquidity params
TRADING_COST_PCT = 0.0005    # explicit commission
BORROW_ANNUAL = 0.005        # borrow fee annual (0.5%)
ADV_CAP_FRAC = 0.10          # don't trade >10% of ADV
IMPACT_COEFF = 0.0006        # sqrt-impact constant
MIN_SLIPPAGE_PCT = 0.0002    # minimum slippage floor per trade
PROFIT_TARGET_PCT = 0.02     # per-layer profit target (can set None)

# Sizing & leverage
VOL_TARGET = 0.20            # target annualized vol of strategy (used to scale notional)
BASE_CAPITAL_FRAC = 0.10     # per-layer base fraction of capital (aggressive)
MIN_NOTIONAL_FRAC = 0.01
MAX_LEVERAGE = 10.0          # USER-SPECIFIED: 10x leverage allowed

# Backtest control: list of single years to run and also a combined test
YEARS_TO_RUN = [2024, 2023, 2022, 2008, 2015, 2016]  # user asked years
COMBINED_SPAN = (min(YEARS_TO_RUN), max(YEARS_TO_RUN))  # combined from min to max by default

INITIAL_CAPITAL = 1.0
USER_CAPITAL = 100000.0      # scale normalized P&L to USD for display

# -------------------------
# Utility functions
# -------------------------
def kalman_filter_estimates(y, x, delta=KF_DELTA, R=None):
    n = len(x)
    m = np.zeros(2)
    P = np.eye(2) * 1.0
    Vw = (delta / (1.0 - delta)) * np.eye(2)
    if R is None:
        R = np.var(y - x)
        if R == 0 or np.isnan(R):
            R = 1.0
    alphas = np.zeros(n); betas = np.zeros(n)
    for t in range(n):
        P = P + Vw
        X_t = np.array([1.0, x[t]])
        y_pred = X_t @ m
        S = X_t @ P @ X_t.T + R
        K = P @ X_t / S
        m = m + K * (y[t] - y_pred)
        P = (np.eye(2) - np.outer(K, X_t)) @ P
        alphas[t] = m[0]; betas[t] = m[1]
    return alphas, betas

def market_impact_pct(notional, adv):
    # sqrt impact model calibrated by IMPACT_COEFF
    if adv <= 0:
        return MIN_SLIPPAGE_PCT
    ratio = notional / adv
    impact = IMPACT_COEFF * math.sqrt(max(ratio, 0.0))
    return max(impact, MIN_SLIPPAGE_PCT)

def performance_stats(equity_series):
    returns = equity_series.pct_change().dropna()
    if len(returns) == 0:
        return {'ann_ret': 0.0, 'ann_vol':0.0, 'sharpe':0.0, 'max_dd':0.0}
    ann_ret = (1 + returns.mean()) ** 252 - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-12)
    peak = equity_series.cummax()
    dd = (peak - equity_series) / peak
    max_dd = dd.max()
    return {'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe, 'max_dd': max_dd}

# -------------------------
# Download data once (price + volume)
# -------------------------
print(f"Downloading {TICKER_X} and {TICKER_Y} history...")
raw = yf.download([TICKER_X, TICKER_Y], start=START_DATE, end=END_DATE, progress=False)
if raw is None or raw.empty:
    raise RuntimeError("Failed to download price data. Run in Colab with internet.")
# choose Adj Close if present
price = raw['Adj Close'] if 'Adj Close' in raw.columns else raw['Close']
vol = raw['Volume']
# combine and forward/backfill where necessary
price = price.dropna().ffill().bfill()
vol = vol.ffill().bfill()
px = price[TICKER_X]; py = price[TICKER_Y]
v_x = vol[TICKER_X]; v_y = vol[TICKER_Y]
prices = pd.concat([px, py, v_x, v_y], axis=1).dropna()
prices.columns = [TICKER_X, TICKER_Y, TICKER_X+"_vol", TICKER_Y+"_vol"]
print(f"Downloaded {len(prices)} rows, from {prices.index[0].date()} to {prices.index[-1].date()}")

# -------------------------
# Precompute Kalman states on full history
# -------------------------
y_all = prices[TICKER_Y].values
x_all = prices[TICKER_X].values
alphas, betas = kalman_filter_estimates(y_all, x_all, delta=KF_DELTA)
kf_all = pd.DataFrame({'alpha': alphas, 'beta': betas, 'px': x_all, 'py': y_all}, index=prices.index)
kf_all['spread'] = kf_all['py'] - (kf_all['alpha'] + kf_all['beta'] * kf_all['px'])
kf_all['spread_mean'] = kf_all['spread'].rolling(LOOKBACK_Z, min_periods=max(10, LOOKBACK_Z//2)).mean()
kf_all['spread_std']  = kf_all['spread'].rolling(LOOKBACK_Z, min_periods=max(10, LOOKBACK_Z//2)).std()
kf_all['z'] = (kf_all['spread'] - kf_all['spread_mean']) / (kf_all['spread_std'] + 1e-12)
kf_all = kf_all.dropna(subset=['z']).copy()
if kf_all.empty:
    raise RuntimeError("Insufficient data after Kalman and rolling stats. Increase START_DATE or reduce LOOKBACK_Z.")

# diagnostic ADF on Kalman spread overall
try:
    adf_p_overall = adfuller(kf_all['spread'].dropna(), autolag='AIC', maxlag=1)[1]
except Exception:
    adf_p_overall = np.nan
print(f"ADF p-value on overall Kalman spread: {adf_p_overall:.6e}")

# -------------------------
# Backtest function (slippage-aware, lever-aware)
# -------------------------
def run_backtest_slice(df_kf, prices_df, start_ts, end_ts):
    """
    df_kf: Kalman DataFrame (subset with z, alpha, beta, px, py)
    prices_df: original prices DataFrame (with volume columns) aligned to same index
    start_ts, end_ts: inclusive timestamps (pd.Timestamp or date-like)
    Returns: equity_series, trades_df
    """
    df_slice = df_kf.loc[(df_kf.index >= start_ts) & (df_kf.index <= end_ts)].copy()
    if df_slice.empty:
        return None, None

    capital = INITIAL_CAPITAL
    positions = []   # active layers: dict with unit_x, unit_y, notional, layer_z, entry_date
    trade_log = []
    equity = []

    # baseline spread volatility (daily) for sizing fallback
    spread_daily_vol = df_slice['spread'].diff().dropna().std()
    if spread_daily_vol <= 0 or np.isnan(spread_daily_vol):
        spread_daily_vol = 1e-6

    def gross_leverage_now(positions_local, px_now, py_now, capital_now):
        gross = sum([abs(p['unit_x'])*px_now + abs(p['unit_y'])*py_now for p in positions_local])
        return gross / (capital_now if capital_now>0 else 1.0)

    for idx, row in df_slice.iterrows():
        z = float(row['z'])
        px_i = float(row['px']); py_i = float(row['py'])
        adv_x = float(prices_df.loc[idx, TICKER_X+"_vol"]) * px_i
        adv_y = float(prices_df.loc[idx, TICKER_Y+"_vol"]) * py_i

        local_vol = float(row['spread_std']) if (not np.isnan(row['spread_std']) and row['spread_std']>0) else spread_daily_vol
        local_ann_vol = max(local_vol * math.sqrt(252), 1e-12)

        # per-layer notional frac via vol-targeting
        notional_frac = BASE_CAPITAL_FRAC * (VOL_TARGET / local_ann_vol)
        notional_frac = float(np.clip(notional_frac, MIN_NOTIONAL_FRAC, MAX_LEVERAGE * BASE_CAPITAL_FRAC))
        per_layer_notional = notional_frac * capital

        # pyramid entry: add layers sequentially if thresholds reached
        existing_layers = [p['layer_z'] for p in positions]
        for layer_z in PYRAMID_Z:
            if abs(z) >= layer_z and layer_z not in existing_layers:
                # check ADV cap for both legs
                if (per_layer_notional > ADV_CAP_FRAC * adv_x) or (per_layer_notional > ADV_CAP_FRAC * adv_y):
                    trade_log.append({'entry_date': idx, 'type': 'skipped_adv', 'layer_z': layer_z, 'entry_z': z})
                    continue
                # compute hypothetical units and leverage
                unit_y = float(- per_layer_notional / py_i) if z>0 else float(per_layer_notional / py_i)
                unit_x = float(abs(row['beta']) * per_layer_notional / px_i) if z>0 else float(-abs(row['beta']) * per_layer_notional / px_i)
                new_gross = gross_leverage_now(positions, px_i, py_i, capital) + (abs(unit_x)*px_i + abs(unit_y)*py_i) / (capital if capital>0 else 1.0)
                if new_gross <= MAX_LEVERAGE:
                    # entry slippage/impact
                    adv_worse = max(adv_x, adv_y)
                    impact_pct = market_impact_pct(per_layer_notional, adv_worse)
                    entry_market_cost = per_layer_notional * (impact_pct + TRADING_COST_PCT)
                    capital -= entry_market_cost
                    positions.append({'layer_z': layer_z, 'unit_x': unit_x, 'unit_y': unit_y, 'notional': per_layer_notional, 'entry_date': idx})
                    trade_log.append({'entry_date': idx, 'type': 'enter_layer', 'layer_z': layer_z, 'entry_z': z, 'notional': per_layer_notional, 'impact_pct': impact_pct})
                else:
                    trade_log.append({'entry_date': idx, 'type': 'skipped_leverage', 'layer_z': layer_z, 'entry_z': z})

        # manage positions: profit target/stoploss/mean-exit
        closed_ix = []
        for i,p in enumerate(positions):
            unreal = p['unit_y']*py_i + p['unit_x']*px_i
            # profit target
            if PROFIT_TARGET_PCT is not None:
                if (unreal / p['notional']) >= PROFIT_TARGET_PCT:
                    adv_worse = max(adv_x, adv_y)
                    impact_pct = market_impact_pct(p['notional'], adv_worse)
                    exit_cost = p['notional'] * (impact_pct + TRADING_COST_PCT)
                    capital += unreal - exit_cost
                    closed_ix.append(i)
                    trade_log.append({'exit_date': idx, 'type': 'profit_target', 'layer_z': p['layer_z'], 'exit_z': z, 'pnl_norm': unreal, 'impact_pct': impact_pct})
                    continue
            # stop-loss
            if (p['unit_y'] < 0 and z > STOP_LOSS_Z) or (p['unit_y'] > 0 and z < -STOP_LOSS_Z):
                adv_worse = max(adv_x, adv_y)
                impact_pct = market_impact_pct(p['notional'], adv_worse)
                exit_cost = p['notional'] * (impact_pct + TRADING_COST_PCT)
                capital += unreal - exit_cost
                closed_ix.append(i)
                trade_log.append({'exit_date': idx, 'type': 'stop_loss', 'layer_z': p['layer_z'], 'exit_z': z, 'pnl_norm': unreal, 'impact_pct': impact_pct})
                continue
            # mean-exit
            if abs(z) < ZS_EXIT:
                adv_worse = max(adv_x, adv_y)
                impact_pct = market_impact_pct(p['notional'], adv_worse)
                exit_cost = p['notional'] * (impact_pct + TRADING_COST_PCT)
                capital += unreal - exit_cost
                closed_ix.append(i)
                trade_log.append({'exit_date': idx, 'type': 'mean_exit', 'layer_z': p['layer_z'], 'exit_z': z, 'pnl_norm': unreal, 'impact_pct': impact_pct})
                continue
            # daily borrow fee drain for short legs (approx)
            days = 1.0/252.0
            short_value = 0.0
            if p['unit_y'] < 0:
                short_value += abs(p['unit_y'])*py_i
            if p['unit_x'] < 0:
                short_value += abs(p['unit_x'])*px_i
            borrow_drain = BORROW_ANNUAL * days * short_value
            capital -= borrow_drain

        # remove closed layers (reverse order)
        for ii in sorted(closed_ix, reverse=True):
            positions.pop(ii)

        # mark-to-market equity
        mtm = capital + sum([pos['unit_y']*py_i + pos['unit_x']*px_i for pos in positions])
        equity.append(mtm)

    equity_series = pd.Series(equity, index=df_slice.index) if len(equity)>0 else pd.Series([capital], index=[df_slice.index[0]])
    trades_df = pd.DataFrame(trade_log)
    return equity_series, trades_df

# -------------------------
# Run per-year backtests and combined span
# -------------------------
results = {}
for year in YEARS_TO_RUN:
    # define start and end of year
    start_ts = pd.Timestamp(year, 1, 1)
    end_ts = pd.Timestamp(year, 12, 31)
    eq, trades = run_backtest_slice(kf_all, prices, start_ts, end_ts)
    if eq is None:
        print(f"No data for year {year}.")
        continue
    stats = performance_stats(eq)
    pnl_norm = float(eq.iloc[-1] - INITIAL_CAPITAL)
    results[year] = {'equity': eq, 'trades': trades, 'pnl_norm': pnl_norm, **stats}
    # save csvs
    eq.to_csv(f"ma_v_v7_3_equity_{year}.csv")
    trades.to_csv(f"ma_v_v7_3_trades_{year}.csv", index=False)
    print(f"Year {year}: final cap {eq.iloc[-1]:.6f}, pnl_norm {pnl_norm:.6f}, ann_ret {stats['ann_ret']:.2%}, ann_vol {stats['ann_vol']:.2%}, sharpe {stats['sharpe']:.3f}, trades {len(trades)}")

# Combined span backtest
combined_start = pd.Timestamp(COMBINED_SPAN[0], 1, 1)
combined_end   = pd.Timestamp(COMBINED_SPAN[1], 12, 31)
eq_comb, trades_comb = run_backtest_slice(kf_all, prices, combined_start, combined_end)
if eq_comb is not None:
    stats_c = performance_stats(eq_comb)
    pnl_c = float(eq_comb.iloc[-1] - INITIAL_CAPITAL)
    results['combined'] = {'equity': eq_comb, 'trades': trades_comb, 'pnl_norm': pnl_c, **stats_c}
    eq_comb.to_csv(f"ma_v_v7_3_equity_combined_{COMBINED_SPAN[0]}_{COMBINED_SPAN[1]}.csv")
    trades_comb.to_csv(f"ma_v_v7_3_trades_combined_{COMBINED_SPAN[0]}_{COMBINED_SPAN[1]}.csv", index=False)
    print(f"Combined {COMBINED_SPAN[0]}-{COMBINED_SPAN[1]}: final cap {eq_comb.iloc[-1]:.6f}, pnl_norm {pnl_c:.6f}, ann_ret {stats_c['ann_ret']:.2%}, ann_vol {stats_c['ann_vol']:.2%}, sharpe {stats_c['sharpe']:.3f}, trades {len(trades_comb)}")

# -------------------------
# Summary table printed
# -------------------------
summary_rows = []
for k,v in results.items():
    summary_rows.append({
        'period': k,
        'final_cap_norm': float(v['equity'].iloc[-1]),
        'pnl_norm': float(v['pnl_norm']),
        'ann_ret': v['ann_ret'],
        'ann_vol': v['ann_vol'],
        'sharpe': v['sharpe'],
        'max_dd': v['max_dd'],
        'num_trades': len(v['trades'])
    })
summary_df = pd.DataFrame(summary_rows).set_index('period')
print("\n=== SUMMARY ===")
display(summary_df)

print("\nSaved CSVs for each year and combined. Inspect equity/trades CSV files for per-trade P&L, timestamps, and state files.")
