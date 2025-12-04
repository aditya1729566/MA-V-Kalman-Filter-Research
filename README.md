 Kalman-Filter Stat-Arb Engine (Realistic, No Lookahead)
A fully realistic statistical arbitrage engine built from first principles.
Includes Kalman-filter hedge ratios, no lookahead bias, ADV-based capacity limits, market impact modeling, borrow costs, and volatility-targeted sizing.
Designed and developed by Aditya, a high-school quant from India documenting the full journey.
 Why This Model Exists
Most backtests look magical because they cheat:
Lookahead hedge ratios
Zero slippage
No market impact
Unlimited leverage
Unrealistic pyramiding
Volatility measured with future data
This engine removes every shortcut and forces the strategy to behave like it would in real markets.
 Core Features
Online Kalman Filter
Uses predict step only for trading (no lookahead)
Dynamic α and β
Execution Cost Model
Nonlinear market impact (sqrt model)
Per-trade slippage floor
Realistic commission
Borrow Cost Model
Short financing charged daily
Capacity Constraints
ADV (Average Daily Volume) caps
Position sizing throttled by liquidity
Risk Controls
Gross leverage limits
Vol-targeting using past-only spread volatility
Stop-loss on extreme Z-scores
Realistic Trade Engine
Layered entries (pyramiding with strict caps)
Mean-reversion exits or profit target
Full cash + position accounting
 Performance Summary (2008–2025)
KO–PEP Pair
Sharpe: 1.90
Annual Return: 22.3%
Annual Vol: ~10.9%
Trades: ~158
No lookahead, realistic impact, ADV & leverage caps
MA–V Pair
Sharpe: 1.31
Annual Return: 7.5%
Very stable, no blowups, matches real-world stat-arb behavior.
These numbers are post-cost, post-impact, fully realistic.
 Architecture Overview
/src
  ├── data_loader.py       # yfinance ingestion + cleaning
  ├── kalman_filter.py     # online KF (predict-only trading)
  ├── signal_gen.py        # Z-score, mean, std, spread calc
  ├── execution.py         # impact, slippage, borrow, commissions
  ├── risk.py              # vol targeting, leverage, ADV sizing
  ├── backtest.py          # trade loop (daily event-driven)
  ├── metrics.py           # Sharpe, drawdown, bootstrap tests
  └── utils.py             # helpers
 How to Run
pip install -r requirements.txt
python backtest.py --x KO --y PEP --start 2005-01-01
Outputs include:
equity curve CSV
trade logs
performance tables
Sharpe significance tests
capacity stats
 Research Philosophy
This engine follows one rule:
If a model doesn’t survive reality, it isn’t alpha.
That means:
No unrealistic fills
No future data
No fantasy leverage
No overfitting
No strategies that die in 2008
Everything built here is meant to mimic how an institutional stat-arb book behaves under real trading frictions.
 Roadmap
 Multi-pair scanning (ADF, Kalman, cointegration)
 Portfolio construction across 20–40 spreads
 Portfolio margin simulation
 Regime filtering (volatility, jumps, clustering)
 Real-time IBKR paper trading
 Web dashboard for analytics
 Acknowledgements
Special thanks to Mike Harris, whose work on execution and market microstructure inspired a big part of this redesign.
 Connect
X / Twitter: @AdityaAgra46739
Email: aditya.eco101@gmail.com
Open to collaboration, discussion, and feedback.

warning : this readme is written by AI If you have any questions DM me
