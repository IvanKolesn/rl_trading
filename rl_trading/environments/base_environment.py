"""
Basic trading environment
"""

from typing import Union
from copy import deepcopy
from random import choice

import gymnasium as gym
import pandas as pd
import numpy as np

from gymnasium.core import ActType, ObsType

DEFAULT_TRADING_PARAMS = {
    "trade_fee": 0.0001,  # 1 bp
    "long_only": True,  # todo: add shorting later
    "base_currency": "usd",
    "max_delta_in_weights": 0.25,
}


class BaseTradingEnv(gym.Env):
    """
    Gymnasium for trading
    """

    def __init__(
        self,
        historical_prices: pd.DataFrame,
        features_dataset: pd.DataFrame,
        initial_portfolio: dict[str, float],
        trading_params: dict[str, Union[float, str, bool]] = DEFAULT_TRADING_PARAMS,
        start_datetime: pd.Timestamp = None,
        episode_length_days: int = 1,
    ):
        """
        Gymnasium for trading
        """
        self.initial_portfolio = deepcopy(initial_portfolio)
        self.current_portfolio = deepcopy(initial_portfolio)
        self.historical_prices = historical_prices
        self.features_dataset = features_dataset
        self.trading_params = trading_params
        self.episode_length_days = int(episode_length_days)

        if start_datetime is not None:
            self.current_datetime = start_datetime
        else:
            self.current_datetime = self._get_random_start_date()

        self.initial_datetime = deepcopy(self.current_datetime)
        self.initial_portfolio_value = None

    def preprocess_data(self) -> None:
        """
        validate inputs
        """
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """
        Check validity of price history and current portfolio
        """

        if self.current_datetime not in self.historical_prices.index:
            raise KeyError(f"{self.current_datetime} is missing in data")

    def _convert_portfolio_to_base_ccy(self) -> dict[str, float]:
        """
        converts portfolio to base currency

        returns dict [ticker, value in base currency]
        """

    @property
    def current_market(self):
        """
        Get current market snapshot
        """
        return self.historical_prices.loc[self.current_datetime, :]

    @property
    def current_portfolio_value(self) -> float:
        """
        Get current portfolio value in base currency
        """
        return sum(self._convert_portfolio_to_base_ccy().values())

    @property
    def _eligible_start_times(self):
        """
        reset time
        """

    @property
    def current_portfolio_weights(self) -> dict[str, float]:
        """
        Get current portfolio weights
        """
        portfolio = self._convert_portfolio_to_base_ccy()
        total_value = sum(portfolio.values())
        if total_value == 0:
            return {ccy: 0.0 for ccy in portfolio}
        return {ccy: value / total_value for ccy, value in portfolio.items()}

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        """
        Gym step
        """

    def _get_state(self) -> np.ndarray:
        """
        Current balance, current rates, returns, etc
        """

    def _get_state_dim(self) -> tuple[float]:
        return self._get_state().shape

    def _get_next_date(self) -> pd.Timestamp:
        return self.historical_prices.loc[str(self.current_datetime) :, :].index[1]

    def _get_random_start_date(self):
        if self.episode_length_days >= len(self._eligible_start_times):
            return self._eligible_start_times[0]
        return choice(self._eligible_start_times[: -self.episode_length_days])

    def reset(self, seed=None, options=None):
        """
        Resets environment
        """
        super().reset(seed=seed)

        self.current_datetime = self._get_random_start_date()
        self.initial_datetime = deepcopy(self.current_datetime)

        self.current_portfolio = deepcopy(self.initial_portfolio)

        return self._get_state(), {
            "datetime": self.current_datetime,
            "portfolio": self.current_portfolio,
        }

    def render(self, render_mode: str = None):
        """
        Gym render
        """
