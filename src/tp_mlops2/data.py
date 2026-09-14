from pathlib import Path

import pandas as pd

def load_raw(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los CSV crudos de energy y weather desde data_dir/raw."""
    raw_dir = data_dir / "raw"
    
    energy = pd.read_csv(raw_dir / "energy_dataset.csv")
    weather = pd.read_csv(raw_dir / "weather_features.csv")
    
    energy['time'] = pd.to_datetime(energy['time'], utc=True)
    energy = energy.set_index('time').sort_index()
    
    return energy, weather

def clean_and_merge(energy: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Limpia y combina los DataFrames de energy y weather."""

    energy = energy.drop(
        columns=[
            'generation hydro pumped storage aggregated',
            'forecast wind offshore eday ahead',
        ]
    )
    numeric_cols = energy.select_dtypes(include='number').columns # Numeric columns to interpolate
    energy[numeric_cols] = energy[numeric_cols].interpolate(method='time')  # Interpolate missing numeric values based on time index
    
    weather = weather.copy()
    weather["dt_iso"] = pd.to_datetime(weather["dt_iso"], utc=True) # Convert dt_iso column to datetime
    weather = weather.drop_duplicates(subset=["dt_iso", "city_name"], keep="first") # Drop duplicates based on the dt_iso and city_name columns
    
    for col in ["temp", "temp_min", "temp_max"]:
        weather[col] = weather[col] - 273.15
    
    weather_wide = weather.pivot_table(index="dt_iso", columns="city_name", values="temp")
    weather_wide.columns = [f"temp_{c.strip()}" for c in weather_wide.columns]
    
    df = energy.join(weather_wide, how="left")
    df.index = pd.to_datetime(df.index, utc=True)

    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    return df