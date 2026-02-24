"""
Basic trading environment
"""

from abc import ABC, abstractmethod
from typing import Union
from copy import deepcopy
from random import choice

import gymnasium as gym
import pandas as pd
import numpy as np
import ray

from gymnasium.core import ActType, ObsType

DEFAULT_TRADING_PARAMS = {
    "trade_fee": 0.0001,  # 1 bp
    "slippage": (0.0001, 0.0002),  # abs( N(0.0001, 0.0002) )
    "long_only": True,  # todo: add shorting later
    "base_currency": "USD",  # all ccy must be in the upper case
    "max_delta_in_weights": 0.25,
    "action_penalty": 0.0,
    "no_trade_penalty": 0.0,
    "reward": "total_profit",  # or diff_sharpe
    "sharpe_eta": 0.1,
}

KNOWN_REWARDS = ("total_profit", "diff_sharpe")


class BaseTradingEnv(gym.Env, ABC):
    """
    Gymnasium for trading
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
        seed: int = 42,
    ):
        """
        Gymnasium for trading
        """

        super().__init__()

        np.random.seed(seed)

        self.initial_portfolio = deepcopy(initial_portfolio)
        self.current_portfolio = deepcopy(initial_portfolio)

        if isinstance(historical_prices, ray.ObjectRef):
            self.historical_prices = ray.get(historical_prices)
        else:
            self.historical_prices = historical_prices

        if isinstance(features_dataset, ray.ObjectRef):
            self.features_dataset = ray.get(features_dataset)
        else:
            self.features_dataset = features_dataset

        self.trading_params = trading_params

        if self.trading_params["reward"] not in KNOWN_REWARDS:
            raise KeyError(f"Unknown reward {self.trading_params["reward"]}")

        self.episode_length_days = int(episode_length_days)
        self.existing_tickers = ticker_set

        self._all_dates = pd.to_datetime(
            list(self.historical_prices.keys()), format="%Y-%m-%d %H:%M:%S"
        ).to_list()
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

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.existing_tickers),),
            dtype=np.float32,
        )

        if trading_params["reward"] == "diff_sharpe":
            self.sharpe_eta = self.trading_params["sharpe_eta"]
            self.A = 0.0  # MA of returns
            self.B = 1e-6  # MA of squared returns

    @abstractmethod
    def preprocess_data(self) -> None:
        """
        validate inputs
        """

    def _validate_inputs(self) -> None:
        """
        Check validity of price history and current portfolio
        """

        if str(self.current_datetime) not in self.historical_prices:
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
        return self.historical_prices[str(self.current_datetime)]

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

        if self.trading_params["reward"] == "diff_sharpe":
            self.A = 0.0  # MA of returns
            self.B = 1e-6  # MA of squared returns

        return self._get_state(), {
            "datetime": self.current_datetime,
        }

    def render(self, render_mode: str = None):
        """
        Gym render
        """
