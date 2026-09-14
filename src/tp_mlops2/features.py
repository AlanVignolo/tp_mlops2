import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_weekend",
    "temp_Madrid", "temp_Barcelona", "temp_Valencia", "temp_Seville", "temp_Bilbao",
    "load_lag_24h", "load_lag_168h",
]

TARGET_COLUMN = "total load actual"


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega codificacion ciclica de hour, dow, month."""
    df = df.copy()
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    return df

def add_lag_features(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """Agrega lags de 24h y 168h del target. Puede introducir Nan al inicio."""
    df = df.copy()
    
    df["load_lag_24h"] = df[target_col].shift(24)
    df["load_lag_168h"] = df[target_col].shift(168)

    return df

def dropna_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Dropea las filas sin lag valido"""
    df = df.copy()
    df = df.dropna(subset=["load_lag_24h", "load_lag_168h"])
    return df

