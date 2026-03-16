"""
add other features
"""

import pandas as pd
import numpy as np


def create_time_features(date: pd.Timestamp) -> np.ndarray:
    """
    Create timefeatures from timestamp
    """
    list_of_features = [
        np.sin(2 * np.pi * date.hour / 24),
        np.cos(2 * np.pi * date.hour / 24),
        np.sin(2 * np.pi * date.minute / 60),
        np.cos(2 * np.pi * date.minute / 60),
        date.dayofweek,
    ]
    return np.array(list_of_features)
