"""
Basic trading environment
"""

from copy import deepcopy

import gymnasium as gym
import pandas as pd
import numpy as np

from gymnasium.core import ActType, ObsType


class BaseTradingEnv(gym.Env):

    def __init__(
        self,
        historical_prices: pd.DataFrame,
        initial_portfolio: dict[str, float],
        trade_fee: float = 0.1,
        long_only: bool = True,  # todo: add shorting later
        base_currency: str = "usd",
        start_datetime: pd.Timestamp = None,
    ):
        """
        Gymnasium for trading
        """
        self.initial_portfolio = deepcopy(initial_portfolio)
        self.current_portfolio = deepcopy(initial_portfolio)
        self.historical_prices = historical_prices
        self.base_currency = base_currency
        self.fees = {
            "general": trade_fee,
        }
        self.long_only = long_only
        self.current_datetime = start_datetime or historical_prices.index.min()
        self.initial_datetime = deepcopy(self.current_datetime)

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

    def _convert_portfolio_to_base_ccy(self) -> dict:
        """
        converts portfolio to base currency
        """

    @property
    def current_market(self):
        return self.historical_prices.loc[self.current_datetime, :]

    @property
    def current_portfolio_value(self):
        """
        Get current portfolio value in base currency
        """

    @property
    def current_portfolio_weights(self):
        """
        Get current portfolio weights
        """

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

    def reset(self, seed=None, options=None):
        """
        Resets environment
        """
        super().reset(seed=seed)
        self.current_datetime = deepcopy(self.initial_datetime)
        self.current_portfolio = deepcopy(self.initial_portfolio)
        return self._get_state(), {
            "datetime": self.current_datetime,
            "portfolio": self.current_portfolio,
        }

    def render(self, render_mode: str = None):
        """
        Gym render
        """
