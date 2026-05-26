"""
Indicadores tecnicos computados sobre series de precios.

Todos los indicadores devuelven valores normalizados para ser usados
directamente como features del estado del agente.
"""
import numpy as np
import pandas as pd


def rsi_norm(close: pd.Series, period: int = 14) -> np.ndarray:
    """RSI normalizado a [-0.5, +0.5]. 0 = neutral, +0.5 = sobrecompra, -0.5 = sobreventa."""
    delta = close.diff().fillna(0)   # primer elemento NaN → 0 (sin cambio)
    gain  = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs    = gain / (loss + 1e-9)
    rsi   = 100 - (100 / (1 + rs))
    return (rsi / 100.0 - 0.5).values   # centrado en 0


def bb_pct_norm(close: pd.Series, period: int = 20) -> np.ndarray:
    """Posicion dentro de las bandas de Bollinger, normalizada a [-0.5, +0.5].
    0 = en la media, +0.5 = en banda superior, -0.5 = en banda inferior."""
    mid   = close.rolling(period, min_periods=1).mean()
    std   = close.rolling(period, min_periods=1).std().fillna(1e-9)
    upper = mid + 2 * std
    lower = mid - 2 * std
    pct   = (close - lower) / (upper - lower + 1e-9)
    return (pct.clip(0, 1) - 0.5).values   # centrado en 0


def resample_1h(df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega velas de 5m a 1h.
    El DataFrame de entrada debe tener un DatetimeIndex.
    """
    df_1h = df_5m.resample("1h").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return df_1h.astype(float)
