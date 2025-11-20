# MA-V-Kalman-Filter-Research
 Overview
This repository contains a complete implementation of a Kalman Filter–based statistical arbitrage strategy for the equity pair Mastercard (MA) and Visa (V) from 2008–2024.
The research introduces a dynamic hedge ratio model using a state–space Kalman Filter, allowing the hedge ratio to evolve smoothly over time instead of relying on static OLS. This significantly improves stability during major market shocks (2008, 2011, 2015, 2020, 2022).
The project includes:
Data preprocessing
Kalman Filter state-space model
Dynamic αₜ and βₜ estimation
Spread modeling
Rolling μ–σ z-score normalization
Layered entry/exit rules
Volatility-targeted position sizing
10× leverage model
Full 2008–2024 backtest
Stress tests across crisis regimes
Performance metrics (Sharpe, drawdowns, exposure, turnover)
This work was originally prepared as part of a technical research submission to Carnegie Mellon University, where I am applying for undergraduate admission.
 Key Features
✔ Dynamic Hedge Ratio (Kalman Filter)
State–space formulation
Time-varying beta (βₜ)
Smooth alpha (αₜ) drift
Better stability than OLS in regime shifts
✔ Mean-Reverting Spread Modeling
spreadₜ = Vₜ − (αₜ + βₜ × MAₜ)
✔ Z-Score Framework (Rolling μ/σ)
Rolling 60–90 day mean
60–90 day volatility
Adaptive thresholds
✔ Layered Trading System
Entry tiers at |z| > 1.25, 1.75, 2.25
Exit at |z| < 0.25
Mean reversion behavior systematically captured
✔ Volatility-Targeted Position Sizing
size = vol_target / realized_vol
Keeps risk controlled even during high-volatility periods.
✔ Leverage Model (Up to 10×)
Portfolio margin simulation
Exposure caps
Per-leg limits
ADV-based liquidity safety checks
✔ 2008–2024 Backtest
Includes performance analysis for:
GFC (2008–2009)
Euro Debt Crisis (2011)
China Devaluation (2015)
Volmageddon (2018)
COVID Crash (2020)
Inflation Bear Market (2022)
Annualized Sharpe > 6 in full-period simulation.
 Project Structure
/
├── data/
│   └── ma_v_2008_2024.csv
│
├── kalman_filter/
│   ├── state_space.py
│   ├── kalman_filter.py
│   └── process_noise_config.py
│
├── backtest/
│   ├── backtest_engine.py
│   ├── position_sizing.py
│   └── execution_model.py
│
├── analysis/
│   ├── beta_plot.py
│   ├── spread_plot.py
│   └── cumulative_returns.py
│
├── utils/
│   ├── preprocess.py
│   └── rolling_stats.py
│
├── main.py
├── requirements.txt
└── README.md
 How to Run the Model
1. Install Dependencies
pip install -r requirements.txt
2. Run the Main Backtest
python main.py
3. Generate Plots
python analysis/beta_plot.py
python analysis/spread_plot.py
python analysis/cumulative_returns.py
 Results Summary (2008–2024)
Metric	Value
Annualized Return	~23.8%
Annualized Volatility	~3.7%
Sharpe Ratio	~6.34
Max Drawdown	~3.2%
Total Trades	~3000
Final Return (10× leverage)	~34.5× capital
 Why MA–V Works Well
Mastercard and Visa have:
highly correlated fundamentals
similar revenue cycles
nearly identical market exposures
long-term co-integration
extremely high liquidity
This makes them one of the most stable and production-ready stat-arb equity pairs in the real world.
 Research Motivation
This code accompanies my full technical report submitted to Carnegie Mellon University as part of my application.
My goal is to demonstrate:
ability to conduct independent quantitative research
familiarity with time-series modeling
ability to validate models through multi-regime backtesting
readiness for CMU’s Statistics, Mathematical Sciences, and MSCF ecosystem.
 Contact
Author: Aditya Agrawal
GitHub: github.com/
Email: 
Research PDF: (
If you are a CMU faculty member reviewing my work, I would be happy to provide the raw data, Jupyter notebooks, or breakdown of each component of the Kalman Filter.
 END OF README
