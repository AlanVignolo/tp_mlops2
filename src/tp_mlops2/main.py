from pathlib import Path
from tp_mlops2.data import load_raw, clean_and_merge
import pandas as pd
import numpy as np
from tp_mlops2.features import add_cyclical_features, add_lag_features, dropna_lags, FEATURE_COLUMNS
from tp_mlops2.predict import load_model, predict
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

energy, weather = load_raw(Path("data"))
df = clean_and_merge(energy, weather)

ref = pd.read_parquet("data/processed/energy_weather_clean.parquet")
print(df.shape, ref.shape)
print(df.equals(ref))

df_feat = add_cyclical_features(df)
df_feat = add_lag_features(df_feat)
df_feat = dropna_lags(df_feat)

print(df_feat.shape)
print(df_feat[FEATURE_COLUMNS].isna().sum().sum())  # debería dar 0

model, feature_cols = load_model(Path("models"))
y_pred = predict(model, df_feat, feature_cols)

test = df_feat[df_feat.index.year >= 2018]
y_pred_test = predict(model, test, feature_cols)

mae = mean_absolute_error(test["total load actual"], y_pred_test)
mape = mean_absolute_percentage_error(test["total load actual"], y_pred_test)
print(f"MAE: {mae:.1f} MW | MAPE: {mape:.2%}")