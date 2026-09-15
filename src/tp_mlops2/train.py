from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from tp_mlops2.data import clean_and_merge, load_raw
from tp_mlops2.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_cyclical_features,
    add_lag_features,
    dropna_lags,
)

PARAM_DIST_RF = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [8, 12, 15, 20, None],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", 0.5, 0.8, 1.0],
}

def build_dataset(data_dir: Path):
    """Corre el pipeline completo (data + features) y devuelve train/test ya listos para modelar."""
    energy, weather = load_raw(data_dir)
    df = clean_and_merge(energy, weather)

    df = add_cyclical_features(df)
    df = add_lag_features(df)
    df = dropna_lags(df)

    train = df[df.index.year < 2018]
    test = df[df.index.year >= 2018]

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    return X_train, y_train, X_test, y_test

def train_model(data_dir: Path, models_dir: Path, quick: bool = False) -> None:
    """Tunea, entrena, evalúa y guarda el modelo Random Forest y sus artefactos.

    quick=True reduce drásticamente el espacio de búsqueda (para CI/smoke tests).
    """
    X_train, y_train, X_test, y_test = build_dataset(data_dir)

    n_splits = 2 if quick else 5
    n_iter = 2 if quick else 20
    param_dist = (
        {"n_estimators": [50], "max_depth": [8], "min_samples_leaf": [10], "max_features": ["sqrt"]}
        if quick
        else PARAM_DIST_RF
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        "mae_mw": mean_absolute_error(y_test, y_pred),
        "mape": mean_absolute_percentage_error(y_test, y_pred),
        "rmse_mw": root_mean_squared_error(y_test, y_pred),
    }

    models_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, models_dir / "random_forest_v1.joblib")
    joblib.dump(FEATURE_COLUMNS, models_dir / "feature_cols_v1.joblib")
    joblib.dump(metrics, models_dir / "metrics_v1.joblib")


    print("Mejores parámetros:", search.best_params_)
    print(
        f"MAE: {metrics['mae_mw']:.1f} MW | "
        f"MAPE: {metrics['mape']:.2%} | "
        f"RMSE: {metrics['rmse_mw']:.1f} MW"
    )


if __name__ == "__main__":
    import os

    quick = os.environ.get("TP_MLOPS2_QUICK_TRAIN") == "1"
    train_model(Path("data"), Path("models"), quick=quick)
