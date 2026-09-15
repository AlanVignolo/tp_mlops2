from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def load_model(models_dir: Path) -> tuple[RandomForestRegressor, list[str], dict]:
    """Carga el modelo entrenado y la lista de features con la que fue entrenados."""
    model = joblib.load(models_dir / "random_forest_v1.joblib")
    feature_columns = joblib.load(models_dir / "feature_cols_v1.joblib")
    metrics = joblib.load(models_dir / "metrics_v1.joblib")

    return model, feature_columns, metrics

def predict(
    model: RandomForestRegressor,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """Predice la demanda para cada fila de df, usando el orden de features del modelo."""
    X = df[feature_cols]
    y_pred = model.predict(X)

    return pd.Series(y_pred, index=df.index, name="predicted_load")
