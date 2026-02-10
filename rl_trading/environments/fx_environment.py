"""
Gym environment for FX trading
"""

import random

from copy import deepcopy
from typing import Union

import pandas as pd
import numpy as np
import gymnasium as gym

from gymnasium.core import ActType, ObsType

from rl_trading.environments.base_environment import (
    BaseTradingEnv,
    DEFAULT_TRADING_PARAMS,
)
from rl_trading.environments.data_processing import (
    get_unique_currencies,
    create_reverse_fx_tickers,
)


class FxTradingEnv(BaseTradingEnv):
    """
    Gymnasium environment for FX trading
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
        Gymnasium environment for FX trading
        """
        super().__init__(
            historical_prices=historical_prices,
            features_dataset=features_dataset,
            initial_portfolio=initial_portfolio,
            trading_params=trading_params,
            start_datetime=start_datetime,
            episode_length_days=int(episode_length_days),
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
            low=-self.trading_params["max_delta_in_weights"],
            high=self.trading_params["max_delta_in_weights"],
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
            self.current_portfolio[x] = self.current_portfolio.get(x, 0)

        self.initial_portfolio = deepcopy(self.current_portfolio)

        for x in self.current_portfolio:
            if x not in self.all_currencies:
                raise KeyError(f"ccy {x} has no history")

    @property
    def _eligible_start_times(self):
        return self.historical_prices.index[
            (self.historical_prices.index.hour == 9)
            & (self.historical_prices.index.minute < 1)
        ].to_list()[1:]

    def _convert_portfolio_to_base_ccy(self) -> dict:
        """
        converts portfolio to base currency
        """
        portfolio_in_base_ccy = {}

        for ccy_name, amount in self.current_portfolio.items():
            if ccy_name == self.trading_params["base_currency"]:
                portfolio_in_base_ccy[ccy_name] = amount
            else:
                direct_pair = ccy_name + self.trading_params["base_currency"]
                if direct_pair in self.current_market:
                    rate = float(self.current_market[direct_pair])
                    portfolio_in_base_ccy[ccy_name] = amount * rate
                else:
                    reverse_pair = self.trading_params["base_currency"] + ccy_name
                    if reverse_pair in self.current_market:
                        rate = 1.0 / float(self.current_market[reverse_pair])
                        portfolio_in_base_ccy[ccy_name] = amount * rate
                    else:
                        raise KeyError(
                            f"No exchange rate found for {ccy_name} to "
                            f"{self.trading_params["base_currency"]}"
                        )

        return portfolio_in_base_ccy

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        """
        Action:

        1. Do trades as flows: from -100% to 100% for one currency pair
        2. Compute new portfolio
        3. Compute rewards (penalize model for trying to sell more than there is in portfolio)
        """
        old_portfolio_value = deepcopy(self.current_portfolio_value)

        # Bankrupt
        if self.current_portfolio_value < 1e-5:
            return self._get_state(), 0, True, False, {}

        combined_actions = list(zip(action, self.existing_currency_pairs))
        random.shuffle(combined_actions)

        for single_action, currency_pair in combined_actions:

            if single_action < 0:
                fx_from, fx_to = currency_pair[:3], currency_pair[-3:]
                mult_to = self.current_market[currency_pair]
            else:
                fx_from, fx_to = currency_pair[-3:], currency_pair[:3]
                mult_to = 1 / self.current_market[currency_pair]

            # Leaving 0.5% of trade amount as buffer
            trade_amount = self.current_portfolio[fx_from] * abs(single_action) * 0.995
            trade_amount = min(self.current_portfolio[fx_from], trade_amount)

            self.current_portfolio[fx_from] -= trade_amount
            self.current_portfolio[fx_to] += (
                trade_amount * mult_to * (1 - self.trading_params["trade_fee"])
            )

        self.current_datetime = self._get_next_date()

        # if penalty:
        #     reward = -1
        # else:
        reward = np.log(self.current_portfolio_value / old_portfolio_value) * 100

        terminated = self.current_datetime == self.historical_prices.index.max()
        truncated = (
            self.current_datetime - self.initial_datetime
        ).days >= self.episode_length_days

        info = {
            "datetime": self.current_datetime,
            "portfolio": self.current_portfolio,
        }

        return self._get_state(), reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        """
        State representation
        """

        current_weights = np.fromiter(
            self.current_portfolio_weights.values(), dtype=np.float32
        )

        all_indicators = (
            self.features_dataset.loc[self.current_datetime, :].to_numpy().flatten()
        )

        return np.concatenate([current_weights, np.array(all_indicators)])
