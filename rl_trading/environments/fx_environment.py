"""
Gym environment for FX trading
"""

import random

from copy import deepcopy
from typing import Union

import pandas as pd
import numpy as np
import gymnasium as gym
import ray

from gymnasium.core import ActType, ObsType

from rl_trading.environments.base_environment import (
    BaseTradingEnv,
    DEFAULT_TRADING_PARAMS,
)


class FxTradingEnv(BaseTradingEnv):
    """
    Gymnasium environment for FX trading
    """

    def __init__(
        self,
        historical_prices: dict[str, dict[str, float]] | ray.ObjectRef,
        features_dataset: dict[str, list] | ray.ObjectRef,
        ticker_set: tuple[str],
        initial_portfolio: dict[str, float],
        trading_params: dict[str, Union[float, str, bool]] = DEFAULT_TRADING_PARAMS,
        start_datetime: pd.Timestamp = None,
        episode_length_days: int = 1,
    ):
        """
        Gymnasium environment for FX trading
        """
        super().__init__(
            historical_prices=historical_prices,
            features_dataset=features_dataset,
            initial_portfolio=initial_portfolio,
            trading_params=trading_params,
            ticker_set=ticker_set,
            start_datetime=start_datetime,
            episode_length_days=int(episode_length_days),
        )

    def preprocess_data(self) -> None:
        """
        1. Set action space
        1. Validate inputs
        3. Create reverse tickers
        4. Set observation space
        """
        self._validate_inputs()

        self.initial_portfolio_value = self.current_portfolio_value

        self.observation_space = gym.spaces.Box(
            low=-1_000, high=1_000, shape=self._get_state_dim(), dtype=np.float32
        )

    def _validate_inputs(self) -> None:
        """
        Check validity of price history and current portfolio
        """
        super()._validate_inputs()

        currency_set = {y for x in self.existing_tickers for y in (x[:3], x[-3:])}
        self.all_currencies = sorted(currency_set)

        for x in self.all_currencies:
            self.current_portfolio[x] = self.current_portfolio.get(x, 0)

        self.initial_portfolio = deepcopy(self.current_portfolio)

        for x in self.current_portfolio:
            if x not in self.all_currencies:
                raise KeyError(f"ccy {x} has no history")

    def _convert_portfolio_to_base_ccy(self) -> dict:
        """
        converts portfolio to base currency
        """
        portfolio_in_base_ccy = {}
        current_market = self.market_on_date

        for ccy_name, amount in self.current_portfolio.items():
            if ccy_name == self.trading_params["base_currency"]:
                portfolio_in_base_ccy[ccy_name] = amount
            else:
                pair = ccy_name + self.trading_params["base_currency"]
                portfolio_in_base_ccy[ccy_name] = amount * float(current_market[pair])

        return portfolio_in_base_ccy

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        """
        Action:

        1. Do trades as flows: from -100% to 100% for one currency pair
        2. Cap them at max_delta_in_weights
        2. Compute new portfolio
        3. Compute rewards (penalize model for trying to sell more than there is in portfolio)
        """

        target_portfolio = self.current_portfolio
        old_portfolio = target_portfolio.copy()
        old_portfolio_value = self.current_portfolio_value

        if old_portfolio_value <= 1e-2:
            return self._get_state(), 0.0, True, False, {}

        current_market = self.market_on_date

        max_w = self.trading_params["max_delta_in_weights"]

        trade_fee = self.trading_params["trade_fee"]
        slippage_mu, slippage_sigma = self.trading_params["slippage"]
        cost = (
            1
            - trade_fee
            - abs(
                np.random.normal(
                    loc=slippage_mu, scale=slippage_sigma, size=len(action)
                )
            )
        )
        cost = np.maximum(cost, 0)

        for i, (single_action, currency_pair) in enumerate(
            zip(action, self.existing_tickers)
        ):

            if single_action < 0:
                fx_from, fx_to = currency_pair[:3], currency_pair[-3:]
            else:
                fx_from, fx_to = currency_pair[-3:], currency_pair[:3]

            trade_amount = min(
                old_portfolio[fx_from],
                old_portfolio[fx_from] * abs(single_action) * max_w,
            )

            target_portfolio[fx_from] -= trade_amount
            target_portfolio[fx_to] += (
                trade_amount * current_market[fx_from + fx_to] * cost[i]
            )

        self.current_idx += 1
        self.current_datetime = self._all_dates[self.current_idx]

        # in basis points if multiplied by 10000
        reward = np.log(self.current_portfolio_value / old_portfolio_value)  # * 10_000

        terminated = self.current_datetime == self._last_date
        truncated = (
            self.current_datetime - self.initial_datetime
        ).days >= self.episode_length_days

        info = {
            "datetime": str(self.current_datetime),
            "portfolio": target_portfolio.copy(),
        }

        return self._get_state(), reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        """
        State representation
        """
        current_weights = self.current_portfolio_weights
        current_weights = np.array([current_weights[x] for x in self.all_currencies])

        all_indicators = np.array(self.features_dataset[str(self.current_datetime)])

        return np.concatenate([current_weights, np.array(all_indicators)])
