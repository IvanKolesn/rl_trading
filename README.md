# RL Trading: Reinforcement Learning for FX Trading

This project implements a custom [Gymnasium](https://gymnasium.farama.org/) environment for an foreign exchange (FX) trading (high-, mid-, low-frequency). It includes data preprocessing, feature preparation, and a training pipeline using [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html).

---

## Overview

The goal is to train an agent to trade multiple currency pairs by modeling flows for each currency pair at every time step. The environment simulates realistic trading conditions including:

- Transaction fees
- Stochastic slippage
- Long‑only positions (extendable to shorting)
- Realistic price data (1‑minute FX bars)

The observation space consists of current portfolio weights (for all currencies) and a set of technical indicators. Actions are continuous and are computed for each currency pair.

We decided not to optimize portfolio weights directly. For a portfolio containing multiple currencies, e.g., {USD, EUR, JPY}, an agent can purchase JPY using both USD and EUR simultaneously - a capability that direct weight optimization would miss. Modeling flows per currency pair implicitly allows such multi‑source trades.

---

## Installation

All nessesary info and dependencies are listed in `pyproject.toml`

Install the package locally:

1. Clone the repository or load it as zip
2. Install it using `pip install`

---

## Usage

### Data Preparation

The project uses 1‑minute FX data from [philipperemy/FX-1-Minute-Data](https://github.com/philipperemy/FX-1-Minute-Data).  
The notebook `notebooks/create_fx_dataset.ipynb` downloads, extracts, cleans, and computes technical indicators.

Steps:
1. Download the zip files (EURJPY, EURUSD, etc.) from the provided Google Drive link.
2. Place them in a folder (e.g., `C:/Users/.../Downloads/`).
3. Run the notebook to generate `data/FX_data.parquet.gzip`.

The final dataset contains OHLCV data plus a wide set of technical indicators (see [Features](#features)) for each currency pair.

### Environment

The core environment is `FxTradingEnv` in `rl_trading/environments/fx_environment.py`. It inherits from `BaseTradingEnv`.

#### Action Space
- **Type**: `Box` (continuous)
- **Shape**: `(n_pairs,)` where `n_pairs` is the number of currency pairs (e.g., `"eurusd"`, `"usdjpy"`, …).
- **Range**: `[-max_delta, max_delta]` with `max_delta = 0.25` (default).

#### Interpretation of an Action
For a given currency pair `XXXYYY` (e.g., `"eurusd"`):
- **Positive action** → buy the **base** currency (`XXX`) using the **quote** currency (`YYY`).  
  `from_currency = YYY`, `to_currency = XXX`.
- **Negative action** → sell the **base** currency (`XXX`) for the **quote** currency (`YYY`).  
  `from_currency = XXX`, `to_currency = YYY`.

#### Usage Example:
```python
import pandas as pd
from rl_trading.environments.fx_environment import FxTradingEnv

# Load preprocessed data
historical_prices = pd.read_parquet("data/historical_prices.parquet")
features = pd.read_parquet("data/features.parquet")

initial_portfolio = {"usd": 100_000}
trading_params = {
    "trade_fee": 0.0001,          # 1 bp
    "slippage": (0.0001, 0.0002), # mean, std of absolute slippage
    "base_currency": "usd",
    "max_delta_in_weights": 0.25,
}

env = FxTradingEnv(
    historical_prices=historical_prices,
    features_dataset=features,
    initial_portfolio=initial_portfolio,
    trading_params=trading_params,
    start_datetime=pd.Timestamp("2023-01-03 09:00:00"),
    episode_length_days=1,
)
env.preprocess_data()

obs, info = env.reset()
action = env.action_space.sample()   # random action
obs, reward, terminated, truncated, info = env.step(action)
```

### Training with PPO

The notebook `notebooks/train_ppo.ipynb` demonstrates creating the environment and training a PPO agent using Ray RLlib.

Key steps:
- Scale features using `RobustScaler`.
- Register the environment and a custom model (`FXModel` in `rl_trading/models/fx_model.py`).
- Configure PPO with appropriate hyperparameters.
- Train and monitor rewards.

The model is trained with the total P&L (log return in basis points) as the reward function.

---

### Trade Execution

Let:
- `holdings[from]` = current amount of the source currency.
- `action` = the scalar action for this pair.

The amount to trade is:
```math
\text{trade\_amount} = \min\Big(\text{holdings[from]},\; \text{holdings[from]} \times |\text{action}|\Big)
```
This caps the trade at the available balance and interprets `|action|` as a fraction of the current holdings in the source currency.

The received amount in the destination currency is:
```math
\text{received} = \text{trade\_amount} \times \text{rate} \times (1 - \text{fee} - \text{slippage})
```
where:
- `rate` = market exchange rate for the pair (e.g., `EURUSD`).
- `fee` = `trade_fee` (e.g., 0.0001 for 1 bps).
- `slippage` = absolute value of a random draw from a normal distribution:
  ```math
  \text{slippage} = \big|\,\mathcal{N}(\mu_{\text{slippage}}, \sigma_{\text{slippage}})\,\big|
  ```
  with default `μ = 0.0001`, `σ = 0.0002`. The absolute value ensures slippage always reduces the received amount (negative impact).

### Reward
The reward after each step is the logarithmic return expressed in basis points:
```math
r_t = \ln\left(\frac{V_t}{V_{t-1}}\right) \times 10\,000
```
where `V_t` is the total portfolio value in the base currency.

---

## Features

Technical indicators are computed per currency pair using `pandas_ta` in `tech_indicators.py`. The list includes:

- **EMA\_21**, **EMA\_9** – exponential moving averages (21‑ and 9‑period).
- **EMA\_gap** = (EMA\_9 – EMA\_21) / close.
- **MACD histogram** – from MACD(12,26,9).
- **ADX** – average directional index (14‑period).
- **DI\_diff** = +DI – –DI.
- **RSI** – relative strength index (14‑period).
- **Stochastic %K** – (14,3,3).
- **ROC\_5** – 5‑period rate of change.
- **ATR percent** = ATR(14) / close × 100.
- **Bollinger position** = (close – lower) / (upper – lower), 20‑period, 2 σ.
- **Donchian position** = (close – low\_20) / (high\_20 – low\_20).
- **BB width** = (upper – lower) / middle.
- **MACD hist delta** – first difference of MACD histogram.
- **Prev high / low** – previous period’s high and low.
- **Close vs prev high/low** – relative differences.

All features are scaled using `RobustScaler` (trained on pre‑2023 data) before being fed to the agent.

---

## Project Structure

```
rl_trading/
├── rl_trading/
│   ├── environments/
│   │   ├── base_environment.py      # Abstract trading environment
│   │   ├── fx_environment.py        # FX-specific implementation
│   │   └── data_processing.py       # Helper functions (reverse tickers)
│   ├── features/
│   │   └── tech_indicators.py       # Technical indicator computation
│   ├── models/
│   │   └── fx_model.py              # Custom PyTorch model for RLlib
│   └── __init__.py
├── notebooks/
│   ├── create_fx_dataset.ipynb      # Data download and feature engineering
│   └── train_ppo.ipynb              # Training pipeline
├── tests/
│   ├── conftest.py                   # Pytest fixtures
│   └── test_fx_environment.py        # Unit tests
├── pyproject.toml                    # Package configuration
└── README.md
```

---

## Further Steps
1. Train the model for more currencies and longer episodes.
2. Add differential Sharpe ratio as an alternative reward function.
3. Experiment with other RL algorithms: SAC, DreamerV3.
4. Test the trained model on validation data and on synthetic data generated by [TimeDiff model](https://github.com/IvanKolesn/market_scenarios_generator).
5(*). Try using Contexual Bandits from [Pearl](https://github.com/facebookresearch/Pearl) or other library
---

## Bibliography

1. Mohammadshafie, A., Mirzaeinia, A., Jumakhan, H., & Mirzaeinia, A. (2024). Deep reinforcement learning strategies in finance: Insights into asset holding, trading behavior, and purchase diversity. In *Proceedings of the CSCE-ICAI’24*. https://arxiv.org/abs/2407.09557

2. Ye, Y., Pei, H., Wang, B., Chen, P.-Y., Zhu, Y., Xiao, J., & Li, B. (2020). Reinforcement-learning based portfolio management with augmented asset movement prediction states. arXiv preprint arXiv:2002.05780. https://arxiv.org/abs/2002.05780

3. Nawathe, S., Panguluri, R., Zhang, J., & Venkatesh, S. (2024). Multimodal deep reinforcement learning for portfolio optimization. arXiv preprint arXiv:2412.17293. https://arxiv.org/abs/2412.17293

4. Lu, L. (2025). Technical Indicator Networks (TINs): An interpretable neural architecture modernizing classical technical analysis for adaptive algorithmic trading. arXiv preprint arXiv:2507.20202. https://arxiv.org/abs/2507.20202
---
