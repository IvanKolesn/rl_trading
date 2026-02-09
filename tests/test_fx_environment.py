"""
Unit tests for FX trading environment
"""

import pytest
import pandas as pd
import numpy as np

from rl_trading.environments.fx_environment import FxTradingEnv


class TestFxTradingEnv:
    """
    Test suite for FxTradingEnv
    """

    def test_portfolio_conversion_usd_base(
        self, historical_exchange_rate_extended, mixed_portfolio
    ):
        """Test conversion to base currency when base is USD"""
        env = FxTradingEnv(
            historical_prices=historical_exchange_rate_extended,
            initial_portfolio=mixed_portfolio,
            base_currency="usd",
            start_datetime=pd.Timestamp("2024-12-01"),
        )
        env.preprocess_data()

        # On 2024-12-01, rates are: eurusd=2.0, usdjpy=110.0, eurjpy=100.0
        portfolio_in_base = env._convert_portfolio_to_base_ccy()

        # USD: 500 USD = 500 USD
        # EUR: 300 EUR = 300 * 2.0 = 600 USD (since 1 EUR = 2 USD)
        # JPY: 200 JPY = 200 / 110.0 = 1.818 USD (since 1 USD = 110 JPY, so 1 JPY = 1/110 USD)
        expected_usd = 500
        expected_eur_in_usd = 300 * 2.0
        expected_jpy_in_usd = 200 / 110.0

        assert portfolio_in_base["usd"] == pytest.approx(expected_usd, rel=1e-3)
        assert portfolio_in_base["eur"] == pytest.approx(expected_eur_in_usd, rel=1e-3)
        assert portfolio_in_base["jpy"] == pytest.approx(expected_jpy_in_usd, rel=1e-3)

    def test_portfolio_conversion_eur_base(
        self, historical_exchange_rate_extended, mixed_portfolio
    ):
        """
        Test conversion to base currency when base is EUR
        """
        env = FxTradingEnv(
            historical_prices=historical_exchange_rate_extended,
            initial_portfolio=mixed_portfolio,
            base_currency="eur",
            start_datetime=pd.Timestamp("2024-12-01"),
        )
        env.preprocess_data()

        # On 2024-12-01, rates are: eurusd=2.0, usdjpy=110.0, eurjpy=100.0
        portfolio_in_base = env._convert_portfolio_to_base_ccy()

        # EUR: 300 EUR = 300 EUR
        # USD: 500 USD = 500 * (1/2.0) = 250 EUR (since 1 USD = 0.5 EUR)
        # JPY: 200 JPY = 200 / 100.0 = 2 EUR (since 1 EUR = 100 JPY)
        expected_eur = 300
        expected_usd_in_eur = 500 * (1 / 2.0)
        expected_jpy_in_eur = 200 / 100.0

        assert portfolio_in_base["eur"] == pytest.approx(expected_eur, rel=1e-3)
        assert portfolio_in_base["usd"] == pytest.approx(expected_usd_in_eur, rel=1e-3)
        assert portfolio_in_base["jpy"] == pytest.approx(expected_jpy_in_eur, rel=1e-3)

    def test_zero_action_no_change(
        self, historical_exchange_rate_extended, mixed_portfolio
    ):
        """
        Test that portfolio stays the same with zero actions
        """
        env = FxTradingEnv(
            historical_prices=historical_exchange_rate_extended,
            initial_portfolio=mixed_portfolio,
            start_datetime=pd.Timestamp("2024-12-01"),
        )
        env.preprocess_data()

        initial_portfolio = env.current_portfolio.copy()
        initial_value = env.current_portfolio_value

        # Take a step with zero action
        zero_action = np.zeros(len(env.existing_currency_pairs), dtype=np.float32)
        state, reward, terminated, truncated, info = env.step(zero_action)

        # Portfolio amounts should remain unchanged
        assert env.current_portfolio == initial_portfolio

        # Portfolio value may change due to exchange rate movements
        # Reward should reflect the change in portfolio value
        new_value = env.current_portfolio_value
        expected_reward = (new_value - initial_value) / initial_value
        assert reward == pytest.approx(expected_reward, rel=1e-10)

    def test_reset_functionality(
        self, historical_exchange_rate_extended, mixed_portfolio
    ):
        """
        Test that reset returns environment to initial state
        """
        env = FxTradingEnv(
            historical_prices=historical_exchange_rate_extended,
            initial_portfolio=mixed_portfolio,
            start_datetime=pd.Timestamp("2024-12-01"),
        )
        env.preprocess_data()

        # Take a few steps
        zero_action = np.zeros(len(env.existing_currency_pairs), dtype=np.float32)
        env.step(zero_action)  # Move to day 2
        env.step(zero_action)  # Move to day 3

        # Reset
        state, info = env.reset()

        # Should be back to initial datetime
        assert env.current_datetime == pd.Timestamp("2024-12-01")
        # Should be back to initial portfolio
        assert env.current_portfolio == mixed_portfolio
        # Info should contain datetime and portfolio
        assert "datetime" in info
        assert "portfolio" in info

    def test_portfolio_weights_sum_to_one(
        self, historical_exchange_rate_extended, mixed_portfolio
    ):
        """
        Test that portfolio weights sum to 1
        """
        env = FxTradingEnv(
            historical_prices=historical_exchange_rate_extended,
            initial_portfolio=mixed_portfolio,
            start_datetime=pd.Timestamp("2024-12-01"),
        )
        env.preprocess_data()

        weights = env.current_portfolio_weights
        total_weight = sum(weights.values())

        assert total_weight == pytest.approx(1.0, rel=1e-10)

    def test_missing_history_raises_error(self, incomplete_historical_data):
        """
        Test that missing history for a currency raises an error
        """
        # Portfolio has JPY but historical data only has EURUSD
        portfolio = {"usd": 1000, "eur": 500, "jpy": 300}

        env = FxTradingEnv(
            historical_prices=incomplete_historical_data,
            initial_portfolio=portfolio,
            start_datetime=pd.Timestamp("2024-12-01"),
        )

        # Should raise KeyError during preprocessing because JPY has no history
        with pytest.raises(KeyError, match="ccy jpy has no history"):
            env.preprocess_data()

    def test_missing_currency_in_initial_portfolio(
        self, historical_exchange_rate_extended
    ):
        """
        Test that missing currency in initial portfolio is set to 0
        """
        # Initial portfolio missing JPY
        initial_portfolio = {"usd": 1000, "eur": 500}

        env = FxTradingEnv(
            historical_prices=historical_exchange_rate_extended,
            initial_portfolio=initial_portfolio,
            start_datetime=pd.Timestamp("2024-12-01"),
        )

        # This should not raise an error
        env.preprocess_data()

        # JPY should be added with 0 value
        assert "jpy" in env.current_portfolio
        assert env.current_portfolio["jpy"] == 0.0

        # Original currencies should remain unchanged
        assert env.current_portfolio["usd"] == 1000
        assert env.current_portfolio["eur"] == 500

    def test_portfolio_value_calculation(
        self, historical_exchange_rate_extended, mixed_portfolio
    ):
        """
        Test that portfolio value is calculated correctly
        """
        env = FxTradingEnv(
            historical_prices=historical_exchange_rate_extended,
            initial_portfolio=mixed_portfolio,
            base_currency="usd",
            start_datetime=pd.Timestamp("2024-12-01"),
        )
        env.preprocess_data()

        # Calculate manually
        # On 2024-12-01: eurusd=2.0, usdjpy=110.0
        # USD: 500
        # EUR: 300 EUR = 300 * 2.0 = 600 USD
        # JPY: 200 JPY = 200 / 110.0 = 1.818 USD
        expected_value = 500 + (300 * 2.0) + (200 / 110.0)

        assert env.current_portfolio_value == pytest.approx(expected_value, rel=1e-3)

    def test_step_with_trade(
        self, historical_exchange_rate_extended, usd_only_portfolio
    ):
        """
        Test that trading works correctly
        """
        env = FxTradingEnv(
            historical_prices=historical_exchange_rate_extended,
            initial_portfolio=usd_only_portfolio,
            start_datetime=pd.Timestamp("2024-12-01"),
        )
        env.preprocess_data()

        # Action: convert 50% of USD to EUR using EURUSD pair
        # existing_currency_pairs will be ['eurusd', 'usdjpy', 'eurjpy']
        action = np.zeros(3, dtype=np.float32)
        eurusd_idx = env.existing_currency_pairs.index("eurusd")
        action[eurusd_idx] = 0.5  # Positive means buy EUR with USD

        initial_value = env.current_portfolio_value
        state, reward, terminated, truncated, info = env.step(action)

        # Check portfolio after trade
        # 50% of 1000 USD = 500 USD traded
        # EUR received = 500 * (1/2.0) * (1 - 0.001 fee) = 250 * 0.999 = 249.75
        expected_eur = 500 / 2.0 * (1 - env.fees["general"])
        expected_usd = 500  # Remaining USD

        assert env.current_portfolio["eur"] == pytest.approx(expected_eur, rel=1e-3)
        assert env.current_portfolio["usd"] == pytest.approx(expected_usd, rel=1e-3)
        assert env.current_portfolio["jpy"] == 0.0

        # Check that datetime advanced
        assert env.current_datetime == pd.Timestamp("2024-12-02")

        # Check that reward was calculated
        assert not terminated
        assert not truncated

    def test_penalty_for_insufficient_funds(
        self, historical_exchange_rate_extended, usd_only_portfolio
    ):
        """
        Test penalty when trying to sell more than available
        """
        env = FxTradingEnv(
            historical_prices=historical_exchange_rate_extended,
            initial_portfolio=usd_only_portfolio,
            start_datetime=pd.Timestamp("2024-12-01"),
        )
        env.preprocess_data()

        # Try to sell 150% of USD (more than available)
        action = np.zeros(3, dtype=np.float32)
        eurusd_idx = env.existing_currency_pairs.index("eurusd")
        action[eurusd_idx] = 1.5  # More than 100%

        state, reward, terminated, truncated, info = env.step(action)

        # Should get penalty reward
        assert reward == -1.0
        assert terminated == True  # Should terminate on penalty
        # All USD should have been traded (since we tried to trade 150% but only 100% was available)
        assert env.current_portfolio["usd"] == 0.0

    def test_empty_portfolio_zero_value(self, historical_exchange_rate_extended):
        """
        Test that empty portfolio has zero value
        """
        empty_portfolio = {"usd": 0, "eur": 0, "jpy": 0}
        env = FxTradingEnv(
            historical_prices=historical_exchange_rate_extended,
            initial_portfolio=empty_portfolio,
            start_datetime=pd.Timestamp("2024-12-01"),
        )
        env.preprocess_data()

        assert env.current_portfolio_value == 0.0
        weights = env.current_portfolio_weights
        # All weights should be 0 (or handle division by zero gracefully)
        assert all(w == 0.0 for w in weights.values())
