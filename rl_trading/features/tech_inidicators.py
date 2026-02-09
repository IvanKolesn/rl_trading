"""
Compute tenchical indicators
"""

import pandas as pd
import pandas_ta as ta
import numpy as np


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive technical indicators for FX intraday data.

    Parameters:
    df : DataFrame with columns ['open', 'high', 'low', 'close',]

    Returns:
    DataFrame with original data + indicator columns
    """

    df = df.copy()

    # TREND INDICATORS
    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]

    adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx"] = adx_result["ADX_14"]
    df["di_plus"] = adx_result["DMP_14"]
    df["di_minus"] = adx_result["DMN_14"]

    ichimoku = ta.ichimoku(df["high"], df["low"], df["close"])
    df["tenkan_sen"] = ichimoku[0]["ITS_9"]
    df["kijun_sen"] = ichimoku[0]["IKS_26"]
    df["senkou_span_a"] = ichimoku[0]["ISA_9"]
    df["senkou_span_b"] = ichimoku[0]["ISB_26"]

    df["psar"] = ta.psar(df["high"], df["low"])["PSARl_0.02_0.2"]

    # VOLATILITY INDICATORS
    bb = ta.bbands(df["close"], length=20, std=2)
    df["bb_upper"] = bb["BBU_20_2.0_2.0"]
    df["bb_middle"] = bb["BBM_20_2.0_2.0"]
    df["bb_lower"] = bb["BBL_20_2.0_2.0"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df[
        "bb_middle"
    ]  # Normalized width
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )  # Position within BB

    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr_percent"] = df["atr"] / df["close"] * 100  # % volatility

    kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=1.5)
    df["kc_upper"] = kc["KCUe_20_1.5"]
    df["kc_middle"] = kc["KCBe_20_1.5"]
    df["kc_lower"] = kc["KCLe_20_1.5"]

    df["donchian_upper"] = df["high"].rolling(20).max()
    df["donchian_lower"] = df["low"].rolling(20).min()
    df["donchian_middle"] = (df["donchian_upper"] + df["donchian_lower"]) / 2

    # MOMENTUM OSCILLATORS
    df["rsi"] = ta.rsi(df["close"], length=14)

    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]

    df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)

    df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)

    # Crossovers
    df["ema_crossover"] = (df["ema_9"] > df["ema_21"]).astype(int)  # 1 when fast > slow
    df["macd_signal_cross"] = (df["macd"] > df["macd_signal"]).astype(int)
    df["di_crossover"] = (df["di_plus"] > df["di_minus"]).astype(int)  # +DI > -DI

    # Relative positions
    df["close_vs_kc"] = (df["close"] - df["kc_middle"]) / (
        df["kc_upper"] - df["kc_middle"]
    )
    df["close_vs_donchian"] = (df["close"] - df["donchian_lower"]) / (
        df["donchian_upper"] - df["donchian_lower"]
    )

    # Squeeze indicator (Bollinger inside Keltner)
    df["bb_kc_squeeze"] = (
        (df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])
    ).astype(int)

    # Price rate of change
    df["roc_1"] = df["close"].pct_change(1)  # 1-period ROC
    df["roc_5"] = df["close"].pct_change(5)  # 5-period ROC

    # Volatility regime
    df["volatility_regime"] = pd.qcut(df["atr_percent"], q=4, labels=[0, 1, 2, 3])

    # Ichimoku cloud position
    df["above_cloud"] = (
        (df["close"] > df["senkou_span_a"]) & (df["close"] > df["senkou_span_b"])
    ).astype(int) - (
        (df["close"] < df["senkou_span_a"]) & (df["close"] < df["senkou_span_b"])
    ).astype(
        int
    )

    df = df.replace([np.inf, -np.inf], np.nan).ffill()

    return df
