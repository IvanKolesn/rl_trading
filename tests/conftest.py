"""
Fixtures for testing
"""

import pytest


@pytest.fixture
def usd_only_portfolio():
    """
    Portfolio with only USD
    """
    return {"usd": 0, "eur": 0}


@pytest.fixture
def historical_exchange_rate():
    """
    Fake numbers
    """
    data = {
        "eurjpy": {
            pd.Timestamp("2024-12-01 09:00:00"): 100.0,
            pd.Timestamp("2024-12-02 09:00:00"): 100.0,
        },
        "eurusd": {
            pd.Timestamp("2024-12-01 09:00:00"): 2.0,
            pd.Timestamp("2024-12-02 09:00:00"): 2.0,
        },
    }
    return pd.DataFrame.from_dict(data)
