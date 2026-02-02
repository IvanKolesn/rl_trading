"""
Gym environment for FX trading
"""

import pandas as pd
import numpy as np
import gymnasium as gym

from src.environments.base_environment import BaseTradingEnv
from src.environments.data_processing import (
    get_unique_currencies,
    create_reverse_fx_tickers,
)


class FxTradingEnv(BaseTradingEnv):

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
        Gymnasium environment for FX trading

        Assumtions:
            1. We cannot make multiple-currency trades at the same time
            For example if we have USD and we want to buy SGDJPY, then we have to buy USDSGD or USDJPY first
        """
        super().__init__(
            historical_prices=historical_prices,
            initial_portfolio=initial_portfolio,
            long_position_fee=long_position_fee,
            long_only=long_only,
            short_position_fee=short_position_fee,  # todo: add shorting later
            base_currency=base_currency,
            start_datetime=start_datetime,
        )

    def preprocess_data(self) -> None:
        """
        1. Validate inputs
        2. Create reverse tickers
        """
        super().preprocess_data()

        self.existing_currency_pairs = self.historical_prices.columns.copy().to_list()
        self.historical_prices = create_reverse_fx_tickers(self.historical_prices)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.existing_currency_pairs),),
            dtype=np.float32,
        )

        # State space = balances, exchange rates, technical indicators, etc
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self._get_state_dim(), dtype=np.float32
        )

    def _validate_inputs(self) -> None:
        """
        Check validity of price history and current portfolio
        """

        super()._validate_inputs()

        self.all_currencies = get_unique_currencies(self.historical_prices)

        for x in self.all_currencies:
            if x not in self.current_portfolio:
                print(f"{x} not in inital portfolio. Setting quantity to 0")
                self.current_portfolio[x] = 0.0

        for x in self.current_portfolio:
            if x not in self.all_currencies:
                raise KeyError(f"ccy {x} has no history")

    def _convert_portfolio_to_base_ccy(self) -> dict:
        """
        converts portfolio to base currency
        """
        portfolio_in_base_ccy = {}

        current_market = self.historical_prices.loc[self.current_datetime, :]

        for ccy_name, amount in self.current_portfolio.items():

            if ccy_name == self.base_currency:
                mult = 1.0
            else:
                mult = float(current_market[self.base_currency + ccy_name])

            portfolio_in_base_ccy[ccy_name] = amount / mult

        return portfolio_in_base_ccy

    @property
    def current_portfolio_value(self) -> float:
        """
        Get current portfolio value in base currency
        """
        return sum(self._convert_portfolio_to_base_ccy().values())

    @property
    def current_portfolio_weights(self) -> dict:
        """
        Get current portfolio weights
        """
        portfolio = self._convert_portfolio_to_base_ccy()
        total_value = sum(portfolio.values())
        return {ccy: value / total_value for ccy, value in portfolio.items()}

    def _get_state(self) -> np.ndarray:
        """
        Current balance, current rates, returns

        Todo: add technical indicators
        """
        balances = np.fromiter(self.current_portfolio.values(), dtype=np.float32)
        current_rates = self.historical_prices.loc[self.current_datetime, :].to_numpy()
        returns = (
            self.historical_prices.loc[: str(self.current_datetime), :]
            .tail(2)
            .apply(np.log)
            .diff()
            .iloc[-1, :]
            .to_numpy()
        )

        return np.concat([balances, current_rates, returns])
