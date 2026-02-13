******Portfolio Optimization with OLS & Neural Networks (ESG + SMFIV)******

****Overview****

This repository contains the code and experimental pipeline for my Master’s thesis:
“OLS and Neural Network Approaches to Portfolio Optimization Integrating SMFIV and ESG Scores with STOXX Europe 600 Constituents (August 2016 – June 2024)”
The project studies how to combine:
Traditional econometric models (Panel OLS)
Neural Networks (Feedforward NN)
with
ESG scores (Bloomberg ESG)
Option-implied volatility signals (SMFIV, model-free implied variance)
to construct and evaluate portfolio optimization strategies under different market regimes.

The main goal is to compare linear vs. nonlinear models in capturing the joint effects of:
Slow-moving fundamental ESG signals
Fast-moving market-implied volatility signals
on portfolio risk-adjusted performance.

****Key Research Questions****

1. ESG Factor Significance
Do ESG scores explain cross-sectional stock returns in the STOXX Europe 600 universe?
2. Incremental Value of SMFIV
Does adding option-implied volatility (SMFIV) improve portfolio Sharpe ratio and drawdown control?
3. OLS vs. Neural Networks
Do neural networks outperform traditional OLS in capturing nonlinear relationships, especially across different market regimes?

****Methodology****
**Data Universe**

Equity universe: STOXX Europe 600 constituents (Aug 2016 – Jun 2024)
ESG: Bloomberg ESG scores (used as ESG factor inputs)
Options: Index-level option-implied volatility to construct SMFIV
Target: Monthly stock returns

**Models**

Panel OLS (benchmark linear model)
Feedforward Neural Network (nonlinear model for return prediction)

**Portfolio Construction**

Forecast-driven portfolio weights
Optimization objective: Sharpe ratio
Evaluation metrics:
Annualized return
Volatility
Sharpe ratio
Maximum drawdown

**Repository Structure**

├── data/                  # Placeholder for user-provided datasets (not included)
├── src/
│   ├── data_processing.py # ESG & SMFIV factor construction
│   ├── ols_model.py       # Panel OLS implementation
│   ├── nn_model.py        # Neural network model
│   ├── backtest.py        # Portfolio backtesting & evaluation
│   └── utils.py
├── notebooks/             # Exploratory analysis & experiments
├── results/               # Figures, tables, performance summaries

****Quick Start****

pip install -r requirements.txt
python ML_Portfolio_Optimization.py

****Main Findings (Summary)****

First, ESG scores alone demonstrated limited and inconsistent predictive power for stock
returns.
Second, option-implied volatility signals, represented by SMFIV, offered additional
information content when combined with ESG factors.
Third, the modelling advantage of neural networks was conditional rather than universal.
Fourth, Panel D synthesises the cumulative investment implications of different model
and factor combinations.

**Thesis Reference**

Liu, S. (2025). OLS and Neural Network Approaches to Portfolio Optimization Integrating SMFIV and ESG Scores with STOXX Europe 600 Constituents (August 2016 – June 2024). Master’s Thesis, Frankfurt School of Finance & Management.

**Data Availability**

⚠️ Raw financial and ESG data are not included due to data licensing restrictions
