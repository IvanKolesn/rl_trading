"""
Fixtures for testing
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def usd_only_portfolio():
    """
    Portfolio with only USD
    """
    return {"usd": 1_000, "eur": 0, "jpy": 0}


@pytest.fixture
def mixed_portfolio():
    """
    Portfolio with multiple currencies
    """
    return {"usd": 500, "eur": 300, "jpy": 200}


@pytest.fixture
def historical_exchange_rate():
    """
    Fake numbers for EURJPY and EURUSD
    """
    data = {
        "eurjpy": {
            pd.Timestamp("2024-12-01 09:00:00"): 100.0,
            pd.Timestamp("2024-12-02 09:00:00"): 101.0,
            pd.Timestamp("2024-12-03 09:00:00"): 102.0,
        },
        "eurusd": {
            pd.Timestamp("2024-12-01 09:00:00"): 2.0,
            pd.Timestamp("2024-12-02 09:00:00"): 1.95,
            pd.Timestamp("2024-12-03 09:00:00"): 1.90,
        },
    }
    return pd.DataFrame.from_dict(data)


@pytest.fixture
def historical_exchange_rate_extended():
    """
    More comprehensive FX data with multiple currency pairs
    """
    index = pd.date_range("2024-12-01", periods=5, freq="D")
    data = {
        "eurusd": [2.0, 1.95, 1.90, 1.85, 1.80],
        "usdjpy": [110.0, 111.0, 112.0, 113.0, 114.0],
        "eurjpy": [100.0, 101.0, 102.0, 103.0, 104.0],
    }
    return pd.DataFrame(data, index=index)


@pytest.fixture
def incomplete_historical_data():
    """
    Historical data missing one currency pair
    """
    index = pd.date_range("2024-12-01", periods=3, freq="D")
    data = {
        "eurusd": [2.0, 1.95, 1.90],
    }
    return pd.DataFrame(data, index=index)
