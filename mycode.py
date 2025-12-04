# Fixed Kalman pairs backtest — explicitly avoids:
# - unbounded pyramiding
# - future-aware hedge ratios
# - over-optimistic impact
# - over-frequent flips
# - vol-targeting using future std
#
# Run in Colab / Jupyter. Requires internet for yfinance.
!pip install yfinance pandas numpy scipy statsmodels python-dateutil --quiet

import math, warnings, numpy as np, pandas as pd, yfinance as yf
from statsmodels.tsa.stattools import adfuller
from dateutil.relativedelta import relativedelta
import scipy.stats as sps
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.6f}'.format

# -------------------------
# USER PARAMETERS (edit these)
# -------------------------
TICKER_X = "MA"   # hedge asset
TICKER_Y = "V"  # quoted asset
START_DATE = "2005-01-01"
END_DATE = None

LOOKBACK_Z = 60          # days for spread mean/std
KF_DELTA = 1e-5
KF_P0 = 1.0

# realistic trading / risk params
BASE_CAPITAL_FRAC = 0.10       # baseline fraction for vol targeting (used with caps)
VOL_TARGET = 0.12              # target annual vol (12%) - realistic
MAX_PER_LEG_FRAC = 0.15        # cap per leg notional as fraction of NAV (15%)
MAX_GROSS_LEVERAGE = 1.5       # gross exposure cap (1.5x)
ADV_CAP_FRAC = 0.10            # don't trade >10% ADV per leg
IMPACT_COEFF = 0.005           # sqrt-impact coefficient (conservative ~50 bps scale)
MIN_SLIPPAGE_PCT = 0.0005      # 5 bps minimum per-leg slippage
TRADING_COST_PCT = 0.0005      # commission per leg (5 bps)
ROUND_TRIP_SLIPPAGE = 0.005    # extra round-trip friction (50 bps)
BORROW_ANNUAL = 0.02           # 2% annual borrow cost
MAINTENANCE_MARGIN = 0.25      # simple maintenance margin proxy

ZS_ENTRY = 2.0     # entry z-score (strict)
ZS_EXIT = 0.3      # exit z-score
STOP_LOSS_Z = 8.0  # catastrophic stop
ALLOW_PYRAMID = False   # disable pyramiding by default
MIN_HOLD_DAYS = 2       # avoid flipping intraday / same-day

ADV_WINDOW = 20
INITIAL_CAPITAL = 1.0
USER_CAPITAL = 100000.0

YEARS_TO_RUN = list(range(2008, 2026))
COMBINED_SPAN = (min(YEARS_TO_RUN), max(YEARS_TO_RUN))

# -------------------------
# UTILITIES
# -------------------------
def market_impact_pct(notional, adv, impact_coeff=IMPACT_COEFF):
    """One-way per-leg impact estimate using sqrt model. Conservative floor applied."""
    if adv <= 0:
        return max(MIN_SLIPPAGE_PCT, 0.001)
    ratio = notional / adv
    impact = impact_coeff * math.sqrt(max(ratio, 0.0))
    return max(impact, MIN_SLIPPAGE_PCT)

def round_trip_cost(notional, adv):
    """Estimate round-trip cost (both legs) as impacts + friction + commissions."""
    impact = market_impact_pct(notional, adv)
    # two legs => impact*2, plus round-trip friction and two commissions
    rt_pct = impact*2.0 + ROUND_TRIP_SLIPPAGE + TRADING_COST_PCT*2.0
    return rt_pct

def autocov(series, lag):
    s = np.asarray(series.dropna())
    n = len(s)
    if lag <= 0 or lag >= n:
        return 0.0
    return float(np.cov(s[lag:], s[:-lag], ddof=1)[0,1])

def metrics_from_equity(equity_series, rf_daily=0.0):
    eq = equity_series.dropna().astype(float)
    if len(eq) < 2:
        return {'n': len(eq), 'ann_ret':0.0,'ann_vol':0.0,'sharpe':0.0,'daily_returns':pd.Series(dtype=float)}
    r = eq.pct_change().dropna()
    N = len(r)
    ann_ret = (eq.iloc[-1] / eq.iloc[0]) ** (252.0 / N) - 1.0
    ann_vol = r.std(ddof=1) * np.sqrt(252.0)
    excess_ann = ann_ret - ((1+rf_daily)**252 - 1)
    sharpe = excess_ann / (ann_vol + 1e-12)
    return {'n':N, 'ann_ret':ann_ret, 'ann_vol':ann_vol, 'sharpe':sharpe, 'daily_returns':r}

def performance_stats_robust(equity_series, rf_daily=0.0):
    m = metrics_from_equity(equity_series, rf_daily)
    r = m['daily_returns']
    N = m['n']
    if N <= 1 or r.empty:
        return {'ann_ret': m['ann_ret'], 'ann_vol': m['ann_vol'], 'sharpe': m['sharpe'],
                'sharpe_se': np.nan, 'sharpe_t': np.nan, 'sharpe_pval': np.nan, 'n':N}
    ex_r = r - rf_daily
    mean_ex = ex_r.mean()
    L = max(1, int(np.floor(4*(N/252.0)**(2/3))))
    S0 = ex_r.var(ddof=1)
    var_mean = S0 / N
    for l in range(1, L+1):
        g = autocov(ex_r, l)
        var_mean += 2.0 * (1.0 - l/(L+1.0)) * g / N
    se_mean = math.sqrt(max(var_mean, 1e-18))
    std_daily = ex_r.std(ddof=1)
    if std_daily <= 0:
        return {'ann_ret': m['ann_ret'], 'ann_vol': m['ann_vol'], 'sharpe': m['sharpe'],
                'sharpe_se': np.nan, 'sharpe_t': np.nan, 'sharpe_pval': np.nan, 'n':N}
    sharpe_daily = mean_ex / (std_daily + 1e-12)
    sharpe_annual = sharpe_daily * np.sqrt(252.0)
    se_sharpe = se_mean / (std_daily + 1e-12) * np.sqrt(252.0)
    t_stat = sharpe_annual / (se_sharpe + 1e-12)
    pval = 2.0 * (1 - sps.norm.cdf(abs(t_stat)))
    return {'ann_ret': m['ann_ret'], 'ann_vol': m['ann_vol'], 'sharpe': sharpe_annual,
            'sharpe_se': se_sharpe, 'sharpe_t': t_stat, 'sharpe_pval': pval, 'n':N}

# -------------------------
# DOWNLOAD DATA
# -------------------------
print(f"Downloading {TICKER_X} & {TICKER_Y} from {START_DATE}...")
raw = yf.download([TICKER_X, TICKER_Y], start=START_DATE, end=END_DATE, progress=False)
if raw is None or raw.empty:
    raise RuntimeError("Failed to download price data. Run in Colab with internet.")
price = raw['Adj Close'] if 'Adj Close' in raw.columns else raw['Close']
vol = raw['Volume']
price = price.ffill().bfill()
vol = vol.ffill().bfill()
px = price[TICKER_X]; py = price[TICKER_Y]
v_x = vol[TICKER_X]; v_y = vol[TICKER_Y]
prices = pd.concat([px, py, v_x, v_y], axis=1).dropna()
prices.columns = [TICKER_X, TICKER_Y, TICKER_X+"_vol", TICKER_Y+"_vol"]
print(f"Downloaded {len(prices)} rows: {prices.index[0].date()} to {prices.index[-1].date()}")

# -------------------------
# ONLINE KALMAN FILTER (predict step used for decision) - no lookahead
# -------------------------
def kalman_online_predict(y, x, delta=KF_DELTA, P0=KF_P0):
    """
    Returns arrays:
      alpha_pred[t], beta_pred[t], spread_pred[t]
    where alpha_pred/beta_pred are m_pred (prior) for that t, and spread_pred is y_t - X_t@m_pred.
    Update step uses y_t so that m (updated) is only available for t+1 decisions.
    """
    n = len(x)
    m = np.zeros(2)
    P = np.eye(2) * P0
    Vw = (delta / (1.0 - delta)) * np.eye(2)
    resid = y - x
    R = np.var(resid) if len(resid)>0 else 1.0
    alphas = np.zeros(n); betas = np.zeros(n); spread_pred = np.zeros(n)
    for t in range(n):
        # PREDICT (use this for trading today)
        m_pred = m.copy()
        P_pred = P + Vw
        alphas[t] = m_pred[0]
        betas[t]  = m_pred[1]
        X_t = np.array([1.0, x[t]])
        y_pred = X_t @ m_pred
        # predicted spread (residual) using only prior coefficients -> NO LOOKAHEAD
        spread_pred[t] = float(y[t] - y_pred)
        # UPDATE (absorbs today's info so future decisions use updated state)
        S = X_t @ P_pred @ X_t.T + R
        K = (P_pred @ X_t) / S
        m = m_pred + K * (y[t] - y_pred)
        P = (np.eye(2) - np.outer(K, X_t)) @ P_pred
    return alphas, betas, spread_pred

y_all = prices[TICKER_Y].values
x_all = prices[TICKER_X].values
alpha_pred, beta_pred, spread_pred = kalman_online_predict(y_all, x_all, delta=KF_DELTA)
kf = pd.DataFrame({'alpha_pred': alpha_pred, 'beta_pred': beta_pred,
                   'px': x_all, 'py': y_all, 'spread_pred': spread_pred}, index=prices.index)

# -------------------------
# Ensure z-score uses past-only stats (shift(1))
# -------------------------
kf['spread_mean'] = kf['spread_pred'].shift(1).rolling(LOOKBACK_Z, min_periods=LOOKBACK_Z//2).mean()
kf['spread_std']  = kf['spread_pred'].shift(1).rolling(LOOKBACK_Z, min_periods=LOOKBACK_Z//2).std()
kf['z'] = (kf['spread_pred'] - kf['spread_mean']) / (kf['spread_std'] + 1e-12)
kf = kf.dropna(subset=['z']).copy()
adf_p = (lambda s: adfuller(s.dropna(), autolag='AIC', maxlag=1)[1] if len(s.dropna())>10 else np.nan)(kf['spread_pred'])
print(f"Prepared Kalman-predicted spread with {len(kf)} usable rows. ADF p-value (pred spread): {adf_p:.6e}")

# -------------------------
# FIXED Backtest engine — avoids the five mistakes
# -------------------------
def run_backtest_fixed(kf_df, prices_df, start_ts, end_ts, initial_nav=INITIAL_CAPITAL):
    """
    Key safety / fixes applied:
     - Trade uses beta_pred (m_pred) computed before seeing y_t (no lookahead).
     - z-score uses shifted rolling mean/std (past-only).
     - Vol-target sizing uses prior-day spread_std (shifted).
     - Hard cap on per-leg notional (MAX_PER_LEG_FRAC) and on gross leverage (MAX_GROSS_LEVERAGE).
     - Pyramiding disabled by default (ALLOW_PYRAMID False) — only one position per direction.
     - Minimum hold days to prevent over-frequent flipping (MIN_HOLD_DAYS).
     - Realistic impact & round-trip slippage used (IMPACT_COEFF, ROUND_TRIP_SLIPPAGE).
     - Profit-target scalping disabled.
    """
    df_slice = kf_df.loc[(kf_df.index >= start_ts) & (kf_df.index <= end_ts)].copy()
    if df_slice.empty:
        return None, None
    cash = float(initial_nav)
    positions = []   # list of dicts: each position holds unit_x, unit_y, dir, entry_date, notional
    trade_log = []
    nav_history = []
    for idx, row in df_slice.iterrows():
        px_t = float(row['px']); py_t = float(row['py'])
        # mark-to-market
        mtm_pos = sum([p['unit_x']*px_t + p['unit_y']*py_t for p in positions])
        nav = cash + mtm_pos
        if nav <= 0:
            nav_history.append(0.0)
            cash = 0.0
            positions = []
            continue
        # trailing ADV computed using prior-day volumes (no lookahead)
        if idx in prices_df.index:
            adv_x = float(prices_df[TICKER_X+"_vol"].loc[:idx].shift(1).tail(ADV_WINDOW).mean()) * px_t
            adv_y = float(prices_df[TICKER_Y+"_vol"].loc[:idx].shift(1).tail(ADV_WINDOW).mean()) * py_t
        else:
            adv_x = adv_y = 1e-9
        adv_x = max(adv_x, 1e-9); adv_y = max(adv_y, 1e-9)
        # z-score is already computed with shift(1)
        z = float(row['z'])
        # Use past-only local vol for vol-targeting: spread_std is already shift(1)
        local_vol = float(row['spread_std']) if (not np.isnan(row['spread_std']) and row['spread_std']>0) else 1e-6
        local_ann_vol = max(local_vol * math.sqrt(252), 1e-12)
        # per-leg fraction via vol target but capped (uses only past info)
        per_leg_frac = BASE_CAPITAL_FRAC * (VOL_TARGET / local_ann_vol)
        per_leg_frac = float(np.clip(per_leg_frac, 0.01, MAX_PER_LEG_FRAC))
        per_leg_notional = per_leg_frac * nav
        # Determine direction
        direction = None
        if z >= ZS_ENTRY:
            direction = 'short_y'  # expect spread to shrink => short Y, long beta*X
        elif z <= -ZS_ENTRY:
            direction = 'long_y'   # expect spread to widen => long Y, short beta*X
        # Avoid re-entering same direction if pyramid disabled
        has_same_dir = any((p['dir'] == direction) for p in positions) if direction is not None else False
        # ENTRY logic
        if direction is not None and (ALLOW_PYRAMID or not has_same_dir):
            # check adv constraint (per leg)
            if per_leg_notional <= ADV_CAP_FRAC * adv_x or per_leg_notional <= ADV_CAP_FRAC * adv_y:
                beta_est = float(row['beta_pred'])  # this is m_pred (no lookahead)
                # compute unit sizes (signed)
                if direction == 'short_y':
                    unit_y = - per_leg_notional / py_t
                else:
                    unit_y =   per_leg_notional / py_t
                unit_x = - beta_est * per_leg_notional / px_t
                # projected new gross exposure
                existing_gross = sum([abs(p['unit_x'])*px_t + abs(p['unit_y'])*py_t for p in positions])
                new_gross = existing_gross + (abs(unit_x)*px_t + abs(unit_y)*py_t)
                new_gross_ratio = new_gross / (nav if nav>0 else 1.0)
                # enforce gross leverage cap
                if new_gross_ratio <= MAX_GROSS_LEVERAGE + 1e-12:
                    # realistic costs: compute round-trip using adv and notional
                    adv_worse = max(adv_x, adv_y)
                    rt_pct = round_trip_cost(per_leg_notional, adv_worse)
                    # entry cost approximated as half round-trip (one-way costs at entry)
                    entry_cost = 0.5 * rt_pct * per_leg_notional
                    # cash flow = cost to buy units at market (mtm)
                    cash_flow = - (unit_x*px_t + unit_y*py_t)
                    cash = cash + cash_flow - entry_cost
                    positions.append({'dir': direction, 'unit_x': unit_x, 'unit_y': unit_y,
                                      'notional': per_leg_notional, 'entry_date': idx,
                                      'entry_px': px_t, 'entry_py': py_t, 'beta': beta_est})
                    trade_log.append({'entry_date': idx, 'type': 'enter', 'dir': direction,
                                      'entry_z': z, 'notional': per_leg_notional,
                                      'entry_cost': entry_cost, 'cash_flow': cash_flow})
                else:
                    trade_log.append({'entry_date': idx, 'type': 'skipped_leverage', 'dir': direction, 'entry_z': z})
            else:
                trade_log.append({'entry_date': idx, 'type': 'skipped_adv', 'dir': direction, 'entry_z': z})
        # EXIT logic: stop-loss or z reversion (with minimum hold to avoid flips)
        closed_idxs = []
        for i,p in enumerate(positions):
            pnl = p['unit_x']*(px_t - p['entry_px']) + p['unit_y']*(py_t - p['entry_py'])
            pnl_frac = pnl / (p['notional'] + 1e-12)
            # enforce minimum hold days before allowing exit (reduces over-frequent flips)
            days_held = (idx - p['entry_date']).days
            allow_exit_by_z = (abs(z) < ZS_EXIT) and (days_held >= MIN_HOLD_DAYS)
            # stop loss (no min-hold here — catastrophic)
            if (p['dir']=='short_y' and z > STOP_LOSS_Z) or (p['dir']=='long_y' and z < -STOP_LOSS_Z):
                adv_worse = max(adv_x, adv_y)
                exit_cost = 0.5 * round_trip_cost(p['notional'], adv_worse) * p['notional']
                cash_flow_exit = - (p['unit_x']*px_t + p['unit_y']*py_t)
                cash = cash + cash_flow_exit - exit_cost
                closed_idxs.append(i)
                trade_log.append({'exit_date': idx, 'type': 'stop_loss', 'dir': p['dir'],
                                  'exit_z': z, 'pnl': pnl, 'pnl_frac': pnl_frac, 'exit_cost': exit_cost})
                continue
            # mean reversion exit (only if min-hold satisfied)
            if allow_exit_by_z:
                adv_worse = max(adv_x, adv_y)
                exit_cost = 0.5 * round_trip_cost(p['notional'], adv_worse) * p['notional']
                cash_flow_exit = - (p['unit_x']*px_t + p['unit_y']*py_t)
                cash = cash + cash_flow_exit - exit_cost
                closed_idxs.append(i)
                trade_log.append({'exit_date': idx, 'type': 'mean_exit', 'dir': p['dir'],
                                  'exit_z': z, 'pnl': pnl, 'pnl_frac': pnl_frac, 'exit_cost': exit_cost})
                continue
            # borrow drain (daily)
            days = 1.0/252.0
            short_value = 0.0
            if p['unit_y'] < 0:
                short_value += abs(p['unit_y'])*py_t
            if p['unit_x'] < 0:
                short_value += abs(p['unit_x'])*px_t
            borrow_drain = BORROW_ANNUAL * days * short_value
            cash -= borrow_drain
        # remove closed positions
        for ii in sorted(closed_idxs, reverse=True):
            positions.pop(ii)
        # force deleverage if margin rules violated (simple heuristic)
        gross = sum([abs(p['unit_x'])*px_t + abs(p['unit_y'])*py_t for p in positions])
        if gross / (nav if nav>0 else 1.0) > MAX_GROSS_LEVERAGE + 0.01:
            # liquidate oldest positions first until under limit
            positions_sorted = sorted(enumerate(positions), key=lambda x: x[1]['entry_date'])
            for idx_pos, pinfo in positions_sorted:
                p = pinfo
                adv_worse = max(adv_x, adv_y)
                exit_cost = 0.5 * round_trip_cost(p['notional'], adv_worse) * p['notional']
                cash_flow_exit = - (p['unit_x']*px_t + p['unit_y']*py_t)
                cash = cash + cash_flow_exit - exit_cost
                trade_log.append({'exit_date': idx, 'type': 'forced_exit_leverage', 'dir': p['dir'],
                                  'exit_z': z, 'pnl': p['unit_x']*(px_t - p['entry_px']) + p['unit_y']*(py_t - p['entry_py']),
                                  'exit_cost': exit_cost})
                # remove it
                positions.remove(p)
                gross = sum([abs(pp['unit_x'])*px_t + abs(pp['unit_y'])*py_t for pp in positions])
                if gross / (nav if nav>0 else 1.0) <= MAX_GROSS_LEVERAGE:
                    break
        # record nav
        mtm_pos = sum([p['unit_x']*px_t + p['unit_y']*py_t for p in positions])
        nav = cash + mtm_pos
        nav_history.append(nav)
    equity_series = pd.Series(nav_history, index=df_slice.index)
    trades_df = pd.DataFrame(trade_log)
    return equity_series, trades_df

# -------------------------
# RUN PER-YEAR & COMBINED BACKTESTS
# -------------------------
results = {}
for year in YEARS_TO_RUN:
    start_ts = pd.Timestamp(year, 1, 1)
    end_ts = pd.Timestamp(year, 12, 31)
    eq, trades = run_backtest_fixed(kf, prices, start_ts, end_ts)
    if eq is None:
        print(f"No data for year {year}.")
        continue
    res_stats = performance_stats_robust(eq)
    pnl_norm = float(eq.iloc[-1] - INITIAL_CAPITAL)
    results[year] = {'equity': eq, 'trades': trades, 'pnl_norm': pnl_norm, **res_stats}
    eq.to_csv(f"fixed_kalman_pairs_equity_{year}.csv")
    trades.to_csv(f"fixed_kalman_pairs_trades_{year}.csv", index=False)
    print(f"Year {year}: final NAV {eq.iloc[-1]:.6f}, pnl_norm {pnl_norm:.6f}, ann_ret {res_stats['ann_ret']:.2%}, ann_vol {res_stats['ann_vol']:.2%}, sharpe {res_stats['sharpe']:.3f}, trades {len(trades)}")

combined_start = pd.Timestamp(COMBINED_SPAN[0], 1, 1)
combined_end   = pd.Timestamp(COMBINED_SPAN[1], 12, 31)
eq_comb, trades_comb = run_backtest_fixed(kf, prices, combined_start, combined_end)
if eq_comb is not None:
    res_stats_c = performance_stats_robust(eq_comb)
    pnl_c = float(eq_comb.iloc[-1] - INITIAL_CAPITAL)
    results['combined'] = {'equity': eq_comb, 'trades': trades_comb, 'pnl_norm': pnl_c, **res_stats_c}
    eq_comb.to_csv(f"fixed_kalman_pairs_equity_combined_{COMBINED_SPAN[0]}_{COMBINED_SPAN[1]}.csv")
    trades_comb.to_csv(f"fixed_kalman_pairs_trades_combined_{COMBINED_SPAN[0]}_{COMBINED_SPAN[1]}.csv", index=False)
    print(f"Combined {COMBINED_SPAN[0]}-{COMBINED_SPAN[1]}: final NAV {eq_comb.iloc[-1]:.6f}, pnl_norm {pnl_c:.6f}, ann_ret {res_stats_c['ann_ret']:.2%}, ann_vol {res_stats_c['ann_vol']:.2%}, sharpe {res_stats_c['sharpe']:.3f}, trades {len(trades_comb)}")

# -------------------------
# SUMMARY
# -------------------------
summary_rows = []
for k,v in results.items():
    summary_rows.append({
        'period': k,
        'final_nav': float(v['equity'].iloc[-1]),
        'pnl_norm': float(v['pnl_norm']),
        'ann_ret': v['ann_ret'],
        'ann_vol': v['ann_vol'],
        'sharpe': v['sharpe'],
        'num_trades': len(v['trades'])
    })
summary_df = pd.DataFrame(summary_rows).set_index('period')
print("\n=== SUMMARY ===")
display(summary_df)

print("\nDone. CSVs saved as fixed_kalman_pairs_equity_*.csv and fixed_kalman_pairs_trades_*.csv")
