"""
Basic trading environment
"""

import gymnasium as gym
import pandas as pd

from gymnasium.core import ActType, ObsType


class BaseTradingEnv(gym.Env):

    def __init__(
        self,
        historical_prices: pd.DataFrame,
        initial_portfolio: dict[str, float],
        long_position_fee: float = 0.1,
        long_only: bool = True,
        short_position_fee: float = 0.1,  # todo: add shorting later
        base_currency: str = "usd",
        start_datetime: pd.Timestamp = None,
    ):
        """
        Gymnasium for trading
        """
        self.current_portfolio = initial_portfolio
        self.historical_prices = historical_prices
        self.base_currency = base_currency
        self.fees = {"long": long_position_fee, "short": short_position_fee}
        self.long_only = long_only
        self.datetime = start_datetime or historical_prices.index.min()

    def preprocess_data(self) -> None:
        """
        validate inputs
        """
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """
        Check validity of price history and current portfolio
        """

        if self.datetime not in self.historical_prices.index:
            raise KeyError(f"{self.datetime} is missing in data")

    def _convert_portfolio_to_base_ccy(self) -> dict:
        """
        converts portfolio to base currency
        """

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

    def reset(self, seed=None, options=None) -> tuple[ObsType, dict]:
        """
        GymEnv reset
        """
        super().reset(seed=seed)

    def render(self, render_mode: str = None):
        """
        Gym render
        """
