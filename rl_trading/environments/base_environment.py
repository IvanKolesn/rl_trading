"""
Basic trading environment
"""

from abc import ABC, abstractmethod
from typing import Union
from copy import deepcopy
from random import choice

from functools import lru_cache
from functools import cached_property

import gymnasium as gym
import pandas as pd
import numpy as np

from gymnasium.core import ActType, ObsType

DEFAULT_TRADING_PARAMS = {
    "trade_fee": 0.0001,  # 1 bp
    "slippage": (0.0001, 0.0002),  # abs( N(0.0001, 0.0002) )
    "long_only": True,  # todo: add shorting later
    "base_currency": "usd",
    "max_delta_in_weights": 0.25,
}


class BaseTradingEnv(gym.Env, ABC):
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
        seed: int = 42,
    ):
        """
        Gymnasium for trading
        """

        super().__init__()

        np.random.seed(seed)

        self.initial_portfolio = deepcopy(initial_portfolio)
        self.current_portfolio = deepcopy(initial_portfolio)
        self.historical_prices = historical_prices.to_dict(orient="index")
        self.features_dataset = features_dataset.to_dict(orient="index")
        self.trading_params = trading_params
        self.episode_length_days = int(episode_length_days)

        self._all_dates = list(self.historical_prices.keys())
        self._last_date = max(self._all_dates)

        if start_datetime is not None:
            self.current_datetime = start_datetime
            self.random_date = False
        else:
            self.current_datetime = self._get_random_start_date()
            self.random_date = True

        self.current_idx = self._all_dates.index(self.current_datetime)
        self.initial_datetime = deepcopy(self.current_datetime)
        self.initial_portfolio_value = None

    def preprocess_data(self) -> None:
        """
        validate inputs
        """

        ticker_set = {ccy for x in self.historical_prices.values() for ccy in x.keys()}
        self.existing_tickers = sorted(ticker_set)

        self.action_space = gym.spaces.Box(
            low=-self.trading_params["max_delta_in_weights"],
            high=self.trading_params["max_delta_in_weights"],
            shape=(len(self.existing_tickers),),
            dtype=np.float32,
        )

    def _validate_inputs(self) -> None:
        """
        Check validity of price history and current portfolio
        """

        if self.current_datetime not in self.historical_prices:
            raise KeyError(f"{self.current_datetime} is missing in data")

    @abstractmethod
    def _convert_portfolio_to_base_ccy(self) -> dict[str, float]:
        """
        converts portfolio to base currency

        returns dict [ticker, value in base currency]
        """

    @property
    def market_on_date(self):
        """
        Get current market snapshot
        """
        return self.historical_prices[self.current_datetime]

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
        return [x for x in self._all_dates if x.hour == 9 and x.minute < 1][1:]

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

    @abstractmethod
    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        """
        Gym step
        """

    @abstractmethod
    def _get_state(self) -> np.ndarray:
        """
        Current balance, current rates, returns, etc
        """

    def _get_state_dim(self) -> tuple[int]:
        return self._get_state().shape

    def _get_random_start_date(self) -> pd.Timestamp:
        if self.episode_length_days >= len(self._eligible_start_times):
            return self._eligible_start_times[0]
        return choice(self._eligible_start_times[: -self.episode_length_days])

    def reset(self, seed=None, options=None) -> tuple:
        """
        Resets environment
        """
        super().reset(seed=seed)

        if self.random_date:
            self.current_datetime = self._get_random_start_date()
            self.initial_datetime = deepcopy(self.current_datetime)
        else:
            self.current_datetime = deepcopy(self.initial_datetime)

        self.current_portfolio = deepcopy(self.initial_portfolio)
        self.current_idx = self._all_dates.index(self.current_datetime)

        return self._get_state(), {
            "datetime": self.current_datetime,
            "portfolio": self.current_portfolio,
            "portfolio_yalue": self.current_portfolio_value,
        }

    def render(self, render_mode: str = None):
        """
        Gym render
        """
