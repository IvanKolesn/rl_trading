"""
Gym environment for FX trading
"""

from copy import deepcopy

import pandas as pd
import numpy as np
import gymnasium as gym

from gymnasium.core import ActType, ObsType

from rl_trading.environments.base_environment import BaseTradingEnv
from rl_trading.environments.data_processing import (
    get_unique_currencies,
    create_reverse_fx_tickers,
)


class FxTradingEnv(BaseTradingEnv):

    def __init__(
        self,
        historical_prices: pd.DataFrame,
        initial_portfolio: dict[str, float],
        trade_fee: float = 0.001,  # 10 bp
        long_only: bool = True,  # todo: add shorting later
        base_currency: str = "usd",
        start_datetime: pd.Timestamp = None,
        max_delta_in_weights: float = 0.25,  # Max 25% at once
        trade_days: int = 1,
    ):
        """
        Gymnasium environment for FX trading
        """
        super().__init__(
            historical_prices=historical_prices,
            initial_portfolio=initial_portfolio,
            trade_fee=trade_fee,
            long_only=long_only,
            base_currency=base_currency,
            start_datetime=start_datetime,
            max_delta_in_weights=max_delta_in_weights,
            trade_days=trade_days,
        )

    def preprocess_data(self) -> None:
        """
        1. Validate inputs
        2. Create reverse tickers
        """
        super().preprocess_data()

        self.existing_currency_pairs = self.historical_prices.columns.copy().to_list()
        self.historical_prices = create_reverse_fx_tickers(self.historical_prices)
        self.initial_portfolio_value = deepcopy(self.current_portfolio_value)

        self.action_space = gym.spaces.Box(
            low=-self.max_delta_in_weights,
            high=self.max_delta_in_weights,
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
                print(f"{x} not in initial portfolio. Setting quantity to 0")
                self.current_portfolio[x] = 0.0

        for x in self.current_portfolio:
            if x not in self.all_currencies:
                raise KeyError(f"ccy {x} has no history")

    def _convert_portfolio_to_base_ccy(self) -> dict:
        """
        converts portfolio to base currency
        """
        portfolio_in_base_ccy = {}

        for ccy_name, amount in self.current_portfolio.items():
            if ccy_name == self.base_currency:
                portfolio_in_base_ccy[ccy_name] = amount
            else:
                # Try direct rate first (e.g., eurusd for EUR to USD conversion)
                direct_pair = ccy_name + self.base_currency
                if direct_pair in self.current_market:
                    rate = float(self.current_market[direct_pair])
                    portfolio_in_base_ccy[ccy_name] = amount * rate
                else:
                    # Try reverse rate (e.g., usdeur for EUR to USD conversion)
                    reverse_pair = self.base_currency + ccy_name
                    if reverse_pair in self.current_market:
                        rate = 1.0 / float(self.current_market[reverse_pair])
                        portfolio_in_base_ccy[ccy_name] = amount * rate
                    else:
                        raise KeyError(
                            f"No exchange rate found for {ccy_name} to {self.base_currency}"
                        )

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
        if total_value == 0:
            return {ccy: 0.0 for ccy in portfolio}
        return {ccy: value / total_value for ccy, value in portfolio.items()}

    # todo: add early stops for truncation
    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        """
        Action:

        1. Do trades as flows: from -100% to 100% for one currency pair
        2. Compute new portfolio
        3. Compute rewards (penalize model for trying to sell more than there is in portfolio)
        """
        new_portfolio = deepcopy(self.current_portfolio)

        # penalty = False
        old_portfolio_value = self.current_portfolio_value

        # Bankrupt
        if old_portfolio_value < 1e-5:
            return self._get_state(), 0, True, False, {}

        for single_action, currency_pair in zip(action, self.existing_currency_pairs):

            # Skip near-zero actions
            if abs(single_action) < 1e-5:
                continue

            if single_action < 0:
                fx_from, fx_to = currency_pair[:3], currency_pair[-3:]
                mult_to = self.current_market[currency_pair]
            else:
                fx_from, fx_to = currency_pair[-3:], currency_pair[:3]
                mult_to = 1 / self.current_market[currency_pair]

            # Leaving 0.5% of trade amount as buffer
            trade_amount = np.floor(
                self.current_portfolio[fx_from] * abs(single_action) * 0.995
            )

            if trade_amount > new_portfolio[fx_from]:
                # penalty = True
                trade_amount = new_portfolio[fx_from]

            new_portfolio[fx_from] -= trade_amount
            assert new_portfolio[fx_from] >= 0

            new_portfolio[fx_to] += trade_amount * mult_to * (1 - self.fees["general"])

        self.current_datetime = self._get_next_date()
        self.current_portfolio = new_portfolio

        new_portfolio_value = self.current_portfolio_value

        # if penalty:
        #     reward = -1
        # else:
        reward = np.log(new_portfolio_value / old_portfolio_value)

        terminated = self.current_datetime == self.historical_prices.index.max()
        truncated = (
            self.current_datetime - self.initial_datetime
        ).days >= self.trade_days

        info = {
            "datetime": self.current_datetime,
            "portfolio": self.current_portfolio,
        }

        return self._get_state(), reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        """
        State representation
        """
        # Current balances
        balances = np.fromiter(self.current_portfolio.values(), dtype=np.float32)

        # Current exchange rates
        current_rates = self.current_market.to_numpy()

        # Recent returns and volatility
        recent_data = self.historical_prices.loc[: str(self.current_datetime), :].tail(
            20
        )
        returns = np.log(recent_data).diff().iloc[-15:].mean().to_numpy()
        volatility = np.log(recent_data).diff().iloc[-15:].std().to_numpy()

        # Portfolio metrics
        portfolio_return = (
            self.current_portfolio_value - self.initial_portfolio_value
        ) / self.initial_portfolio_value

        # Concatenate all features
        return np.concatenate(
            [balances, current_rates, returns, volatility, [portfolio_return]]
        )
