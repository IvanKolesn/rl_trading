"""
Useful functions
"""

import pandas as pd


def create_reverse_fx_tickers(historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    If only USDEUR is present, create EURUSD as 1 / USDEUR
    """
    initial_columns = list(historical_data.columns)
    for fx in initial_columns:
        reverse_rate = fx[-3:] + fx[:3]
        if reverse_rate not in initial_columns:
            historical_data[reverse_rate] = 1 / historical_data[fx]
    return historical_data
