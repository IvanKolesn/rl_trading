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

## Data Preparation

The project uses 1‑minute FX data from [philipperemy/FX-1-Minute-Data](https://github.com/philipperemy/FX-1-Minute-Data).  
Two Jupyter notebooks handle the complete data pipeline:

### 1. `create_fx_dataset.ipynb`
- Downloads the zip archives (EURJPY, EURUSD, etc.) from the provided Google Drive link.
- Extracts all CSV files and concatenates them into a single DataFrame.
- Filters data to keep only trading hours (9:00–19:00) and weekdays.
- Adds a `date` column and sorts by currency and timestamp.
- Computes a wide set of technical indicators using `pandas_ta` (see [Features](#features)).
- Saves the result as `data/FX_data.parquet.gzip`.

### 2. `prepare_fx_dataset.ipynb`
- Loads the parquet file and adds return columns (lagged returns and squared returns).
- Splits the data into three periods:
  - **Scaler training**: data ≤ 2023‑01‑01 (used to fit `StandardScaler` – avoids look‑ahead bias).
  - **Training set**: 2023‑01‑02 to 2023‑12‑01.
  - **Validation set**: from 2023‑12‑01 onwards.
- Extracts close prices for each currency pair and creates reverse tickers (e.g., if only `EURUSD` exists, `USDEUR` is added as `1 / EURUSD`).
- Scales all features using `StandardScaler` (fitted on the scaler‑training set).
- Applies PCA (keeping components that explain >0.5 % variance) to reduce dimensionality.
- Saves the final preprocessed data (prices + PCA‑reduced features) as `data/preprocessed_data.pkl` (using `dill`).

After these steps, the data is ready for training.

---

## Environment

The core environment is `FxTradingEnv` in `rl_trading/environments/fx_environment.py`. It inherits from `BaseTradingEnv`.

### Action Space
- **Type**: `Box` (continuous)
- **Shape**: `(n_pairs,)` where `n_pairs` is the number of currency pairs (e.g., `"EURUSD"`, `"USDJPY"`, …).
- **Range**: `[-1.0, 1.0]`

#### Interpretation of an Action
For a given currency pair `XXXYYY` (e.g., `"EURUSD"`):
- **Positive action** → buy the **base** currency (`XXX`) using the **quote** currency (`YYY`).  
  `from_currency = YYY`, `to_currency = XXX`.
- **Negative action** → sell the **base** currency (`XXX`) for the **quote** currency (`YYY`).  
  `from_currency = XXX`, `to_currency = YYY`.

Inside `step()`, the raw action is first multiplied by `max_delta_in_weights` (default `0.25`). This scaled value is then used to determine the trade amount as a fraction of the source currency’s holdings.

### Trade Execution
Let:
- `holdings[from]` = current amount of the source currency.
- `scaled_action` = raw action × `max_delta_in_weights`.

The amount to trade is:
```
trade_amount = min(holdings[from], holdings[from] * |scaled_action|)
```
This caps the trade at the available balance and interprets `|scaled_action|` as a fraction of the current holdings in the source currency.

The received amount in the destination currency is:
```
received = trade_amount × rate × (1 - fee - slippage)
```
where:
- `rate` = market exchange rate for the pair (e.g., `EURUSD`).
- `fee` = `trade_fee` (default 0.0001 = 1 bps).
- `slippage` = absolute value of a random draw from a normal distribution:  
  `slippage = |N(μ_slippage, σ_slippage)|` with default `μ = 0.0001`, `σ = 0.0002`. The absolute value ensures slippage always reduces the received amount.

### Reward
The reward after each step is the logarithmic return:
```
r_t = ln(V_t / V_{t-1})
```
where `V_t` is the total portfolio value in the base currency.  
A quadratic penalty discourages large trades:
```
reward = r_t - action_penalty × sum_i (scaled_action_i)^2
```
The coefficient `action_penalty` can be tuned (default `0.5`).  
If `reward = "diff_sharpe"`, the instantaneous reward is the **differential Sharpe ratio** (see the code for the exact formula).

### Observation Space
The observation is a concatenation of:
- **Current portfolio weights** for all currencies (in the base currency).
- **Technical indicators** (from the PCA‑reduced features).
- **Time features** (minute, hour, day of week, etc., cyclic encoded).

The exact shape depends on the number of currencies and the number of PCA components kept.

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

## Training with PPO

The notebook `notebooks/train_ppo.ipynb` demonstrates a complete training pipeline using Ray RLlib and a custom PyTorch model (`FXModel`).

### Custom Model Architecture (`fx_model.py`)

The model consists of three parts:
- **Main network** – three hidden layers (256 → 128 → 64) with LayerNorm, GELU activation, and dropout (0.2).
- **Mean network** – linear layer projecting the 64‑dim features to the action dimension, followed by `Tanh` to bound actions in `[-1, 1]`.
- **Log‑std network** – linear layer producing the log‑standard deviations for the action distribution (required by RLlib’s `StochasticSampling`).
- **Value network** – two layers (64 → 32 → 1) with GELU activation.

The forward pass returns a concatenated tensor of means and log‑stds, as required by RLlib.

### RLlib Configuration
- **Environment**: custom `FxTradingEnv` registered as `"fx_trading_env"`.
- **Framework**: PyTorch.
- **PPO hyperparameters**:
  - Learning rate: `3e-4`
  - Clip parameter: `0.2`
  - Gradient clipping: `0.5`
  - Value function clipping: `10.0`
  - Entropy coefficient: `0.01`
  - Discount factor (gamma): `0.99`
  - GAE lambda: `0.95`
  - Number of SGD epochs per batch: `8`
  - Train batch size: `16384 × episode_length_days`
- **Rollout workers**: uses all available CPUs minus one, with 4 environments per worker.
- **Exploration**: `StochasticSampling` with `random_timesteps=5000`.
- **Data passing**: Large data objects (`historical_prices`, `features`) are passed via `ray.put` to avoid serialisation overhead.

### Early Stopping
Training stops if the average reward over the last 5 epochs does not improve by at least `0.01` for 20 consecutive epochs (after the first 100 epochs).

After training, the PyTorch state dict is saved (e.g., `ALL_CCY_MODEL_WEIGHTS_1_MIN_1_DAY.pth`) for later evaluation.

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
│   ├── prepare_fx_dataset.ipynb     # Scaling, splitting, PCA, saving
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
2. Apply PCA on features set for dimensionality reduction.
3. Experiment with other RL algorithms: SAC, DreamerV3.
4. Test the trained model on validation data and on synthetic data generated by a [TimeDiff model](https://github.com/IvanKolesn/market_scenarios_generator).
5. (*Optional*) Try Contextual Bandits using [Pearl](https://github.com/facebookresearch/Pearl).

---

## Bibliography

1. Mohammadshafie, A., Mirzaeinia, A., Jumakhan, H., & Mirzaeinia, A. (2024). Deep reinforcement learning strategies in finance: Insights into asset holding, trading behavior, and purchase diversity. In *Proceedings of the CSCE-ICAI’24*. https://arxiv.org/abs/2407.09557

2. Ye, Y., Pei, H., Wang, B., Chen, P.-Y., Zhu, Y., Xiao, J., & Li, B. (2020). Reinforcement-learning based portfolio management with augmented asset movement prediction states. arXiv preprint arXiv:2002.05780. https://arxiv.org/abs/2002.05780

3. Nawathe, S., Panguluri, R., Zhang, J., & Venkatesh, S. (2024). Multimodal deep reinforcement learning for portfolio optimization. arXiv preprint arXiv:2412.17293. https://arxiv.org/abs/2412.17293

4. Lu, L. (2025). Technical Indicator Networks (TINs): An interpretable neural architecture modernizing classical technical analysis for adaptive algorithmic trading. arXiv preprint arXiv:2507.20202. https://arxiv.org/abs/2507.20202
```
