import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
import os

from tp_mlops2.features import FEATURE_COLUMNS

load_dotenv()

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME = "random_forest_demanda"
MODEL_ALIAS = "production"

def load_model():
    """Carga el modelo en produccion y sus metricas desde el MLflow Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    run = client.get_run(model_version.run_id)
    metrics = run.data.metrics

    return model, FEATURE_COLUMNS, metrics


def predict(model, df: pd.DataFrame, feature_columns: list[str])-> pd.Series:
    """Predice la demanda para cada fila de df, usando el roden de features del modelo."""
    X = df[feature_columns]
    y_pred = model.predict(X)

    return pd.Series(y_pred, index=df.index, name="predicted_load")
