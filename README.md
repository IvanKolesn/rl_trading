# RL Trading: Reinforcement Learning for FX Trading

This project implements a custom [Gymnasium](https://gymnasium.farama.org/) environment for foreign exchange (FX) trading (high‑, mid‑, low‑frequency). It includes data preprocessing, feature engineering, and a training pipeline using [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html).

The environment and model were built from scratch, without relying on existing frameworks like [FinRL](https://github.com/AI4Finance-Foundation/FinRL) or [TradeMaster](https://github.com/TradeMaster-NTU/TradeMaster).

---

## Overview

The goal is to train an agent to trade multiple currency pairs by modelling **flows** for each currency pair at every time step. The environment simulates realistic trading conditions including:

- Transaction fees (1 bps by default)
- Stochastic slippage
- Long‑only positions (easily extendable to shorting)
- Realistic 1‑minute FX bars

The observation space consists of current portfolio weights (for all currencies) and a rich set of technical indicators.

**Why flows instead of direct weight optimisation?**  
For a portfolio containing multiple currencies, e.g. `{USD, EUR, JPY}`, an agent should be able to purchase JPY using both USD and EUR simultaneously – a capability that direct weight optimisation would miss. Modelling flows per currency pair implicitly allows such multi‑source trades. Therefore, actions are continuous and computed for each tradable currency pair, without relying on synthetic rates.

---

## Installation

All dependencies are listed in `pyproject.toml`.

Install the package locally:

```bash
git clone <repository-url>
cd rl_trading
pip install .
```

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

**Data Splitting & Scaling**  
The notebook `train_ppo.ipynb` demonstrates a typical data split:
- First three years → train the `StandardScaler`.
- Next two years → training set.
- Last month → validation set.

Features are scaled using `StandardScaler` (or you may substitute `RobustScaler`). The scaler is fitted only on the earliest part of the data to avoid look‑ahead bias.

### Environment

The core environment is `FxTradingEnv` in `rl_trading/environments/fx_environment.py`. It inherits from `BaseTradingEnv`.

#### Action Space
- **Type**: `Box` (continuous)
- **Shape**: `(n_pairs,)` where `n_pairs` is the number of currency pairs (e.g., `"eurusd"`, `"usdjpy"`, …).
- **Range**: `[-1.0, 1.0]` (interpreted as a fraction of the available source currency, after scaling by `max_delta_in_weights` inside the environment).

#### Interpretation of an Action
For a given currency pair `XXXYYY` (e.g., `"eurusd"`):
- **Positive action** → buy the **base** currency (`XXX`) using the **quote** currency (`YYY`).  
  `from_currency = YYY`, `to_currency = XXX`.
- **Negative action** → sell the **base** currency (`XXX`) for the **quote** currency (`YYY`).  
  `from_currency = XXX`, `to_currency = YYY`.

Inside `step()`, the raw action (in `[-1, 1]`) is first multiplied by `max_delta_in_weights` (default `0.25`). This scaled value is then used to determine the trade amount as a fraction of the source currency’s holdings.

#### Reverse Ticker Creation
The utility `create_reverse_fx_tickers` (in `data_processing.py`) automatically adds missing reverse pairs (e.g., creates `EURUSD` if only `USDEUR` is present) by taking the reciprocal. This ensures that every currency pair is tradable in both directions.

#### Trade Execution

Let:
- `holdings[from]` = current amount of the source currency.
- `scaled_action` = raw action (in `[-1, 1]`) multiplied by `max_delta_in_weights`.

The amount to trade is:
```math
\text{trade\_amount} = \min\Big(\text{holdings[from]},\; \text{holdings[from]} \times |\text{scaled\_action}|\Big)
```
This caps the trade at the available balance and interprets `|scaled_action|` as a fraction of the current holdings in the source currency.

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

#### Reward

The reward after each step is the logarithmic return expressed in basis points, minus a quadratic penalty to discourage large trades:
```math
r_t = \ln\left(\frac{V_t}{V_{t-1}}\right) \;-\; \text{action\_penalty} \times \sum_i (\text{scaled\_action}_i)^2
```
where `V_t` is the total portfolio value in the base currency.  
The quadratic term penalises the sum of squares of the **scaled actions**, promoting smoother position changes. The coefficient `action_penalty` can be tuned (default `0.5`).

**Alternative reward: differential Sharpe ratio**  
If `reward = "diff_sharpe"`, the instantaneous reward is the **differential Sharpe ratio**:
```math
D_t = \frac{B_{t-1}(r_t - A_{t-1}) - \tfrac12 A_{t-1}(r_t^2 - B_{t-1})}{(B_{t-1} - A_{t-1}^2)^{3/2} + \epsilon}
```
with exponential moving averages `A` (mean) and `B` (second moment) updated as:
```math
A_t = A_{t-1} + \eta (r_t - A_{t-1}), \quad B_t = B_{t-1} + \eta (r_t^2 - B_{t-1})
```
where `η = 0.1` (default). This formulation directly optimises the Sharpe ratio over time.

#### Usage Example
```python
import pandas as pd
from rl_trading.environments.fx_environment import FxTradingEnv

# Load preprocessed data (after scaling)
historical_prices = pd.read_parquet("data/historical_prices.parquet")
features = pd.read_parquet("data/features.parquet")

initial_portfolio = {"USD": 100_000}
trading_params = {
    "trade_fee": 0.0001,
    "slippage": (0.0001, 0.0002),
    "base_currency": "USD",
    "max_delta_in_weights": 0.25,
    "action_penalty": 0.5,
    "reward": "total_profit",          # or "diff_sharpe"
    "sharpe_eta": 0.1,                 # used only with diff_sharpe
}

env = FxTradingEnv(
    historical_prices=historical_prices,
    features_dataset=features,
    initial_portfolio=initial_portfolio,
    trading_params=trading_params,
    ticker_set=("EURUSD", "USDJPY", ...),
    start_datetime=pd.Timestamp("2023-01-03 09:00:00"),
    episode_length_days=1,
)
env.preprocess_data()

obs, info = env.reset()
action = env.action_space.sample()   # random action in [-1, 1]
obs, reward, terminated, truncated, info = env.step(action)
```

### Training with PPO

The notebook `notebooks/train_ppo.ipynb` demonstrates a complete training pipeline using Ray RLlib and a custom PyTorch model (`FXModel`).

#### Custom Model Architecture (`fx_model.py`)

The model consists of three parts:

- **Main network** – three hidden layers (256 → 128 → 64) with LayerNorm, GELU activation, and dropout (0.2).
- **Mean network** – linear layer projecting the 64‑dim features to the action dimension, followed by `Tanh` to bound actions in `[-1, 1]`.
- **Log‑std network** – linear layer producing the log‑standard deviations for the action distribution.
- **Value network** – two layers (64 → 32 → 1) with GELU activation.

The forward pass returns a concatenated tensor of means and log‑stds, as required by RLlib’s `StochasticSampling` exploration.

Data objects (`historical_prices`, `features`) are passed to Ray workers via `ray.put` to avoid serialisation overhead.

After training, the PyTorch state dict is saved (e.g., `USDEUR_MODEL_WEIGHTS_1_MIN_1_DAY.pth`) for later evaluation.

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

All features are scaled using `StandardScaler` (trained on pre‑2023 data) before being fed to the agent.

---

## Project Structure

```
rl_trading/
├── rl_trading/
│   ├── environments/
│   │   ├── base_environment.py      # Abstract trading environment
│   │   ├── fx_environment.py        # FX‑specific implementation
│   │   └── data_processing.py       # Helper functions (reverse tickers)
│   ├── features/
│   │   └── tech_indicators.py       # Technical indicator computation
│   ├── models/
│   │   └── fx_model.py              # Custom PyTorch model for RLlib
│   └── __init__.py
├── notebooks/
│   ├── create_fx_dataset.ipynb      # Data download and feature engineering
│   └── train_ppo.ipynb              # Training pipeline with RLlib
├── tests/
│   ├── conftest.py                   # Pytest fixtures
│   └── test_fx_environment.py        # Unit tests
├── pyproject.toml                    # Package configuration
└── README.md
```

---

## Further Steps

1. Train the model for more currencies and longer episodes.
2. Apply PCA on features set for dimensionality reduction
3. Experiment with other RL algorithms: SAC, DreamerV3.
4. Test the trained model on validation data and on synthetic data generated by a [TimeDiff model](https://github.com/IvanKolesn/market_scenarios_generator).
5. (*Optional*) Try Contextual Bandits using [Pearl](https://github.com/facebookresearch/Pearl).

---

## Bibliography

1. Mohammadshafie, A., Mirzaeinia, A., Jumakhan, H., & Mirzaeinia, A. (2024). Deep reinforcement learning strategies in finance: Insights into asset holding, trading behavior, and purchase diversity. In *Proceedings of the CSCE-ICAI’24*. https://arxiv.org/abs/2407.09557

2. Ye, Y., Pei, H., Wang, B., Chen, P.-Y., Zhu, Y., Xiao, J., & Li, B. (2020). Reinforcement-learning based portfolio management with augmented asset movement prediction states. arXiv preprint arXiv:2002.05780. https://arxiv.org/abs/2002.05780

3. Nawathe, S., Panguluri, R., Zhang, J., & Venkatesh, S. (2024). Multimodal deep reinforcement learning for portfolio optimization. arXiv preprint arXiv:2412.17293. https://arxiv.org/abs/2412.17293

4. Lu, L. (2025). Technical Indicator Networks (TINs): An interpretable neural architecture modernizing classical technical analysis for adaptive algorithmic trading. arXiv preprint arXiv:2507.20202. https://arxiv.org/abs/2507.20202
