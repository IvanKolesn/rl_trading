"""
Features (technical indicators) for intraday data.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a compact set of technical indicators for FX intraday data.
    """

    df = df.copy()

    df["ema_21"] = ta.ema(df["close"], length=21)
    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_gap"] = (df["ema_9"] - df["ema_21"]) / df["close"]

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd_hist"] = macd["MACDh_12_26_9"]

    adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx"] = adx_result["ADX_14"]

    di_plus = adx_result["DMP_14"]
    di_minus = adx_result["DMN_14"]
    df["di_diff"] = di_plus - di_minus

    df["rsi"] = ta.rsi(df["close"], length=14)

    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["roc_5"] = df["close"].pct_change(5)

    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr_percent"] = df["atr"] / df["close"] * 100
    df.drop("atr", axis=1, inplace=True)

    bb = ta.bbands(df["close"], length=20, std=2)
    bb_upper = bb["BBU_20_2.0_2.0"]
    bb_lower = bb["BBL_20_2.0_2.0"]
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)

    donchian_upper = df["high"].rolling(20).max()
    donchian_lower = df["low"].rolling(20).min()
    df["donchian_position"] = (df["close"] - donchian_lower) / (
        donchian_upper - donchian_lower
    )

    df["bb_width"] = (bb_upper - bb_lower) / bb["BBM_20_2.0_2.0"]

    df["macd_hist_delta"] = df["macd_hist"].diff(1)

    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)

    df["close_vs_prev_high"] = (df["close"] - df["prev_high"]) / df["prev_high"]
    df["close_vs_prev_low"] = (df["close"] - df["prev_low"]) / df["prev_low"]

    # df.drop(["ema_9", "prev_high", "prev_low"], axis=1, inplace=True, errors="ignore")

    df = df.replace([np.inf, -np.inf], np.nan).ffill()

    return df
