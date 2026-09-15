from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from tp_mlops2.api.schemas import ModelInfo, PredictionRequest, PredictionResponse
from tp_mlops2.features import add_cyclical_features
from tp_mlops2.predict import load_model, predict

BASE_DIR = Path(__file__).resolve().parents[3]
MODELS_DIR = BASE_DIR / "models"
MODEL_VERSION = "random_forest_v1"

app = FastAPI(title="Demanda Electrica España API", version="1.0")

model, feature_cols, metrics = load_model(MODELS_DIR)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    return ModelInfo(
        model_name="random_forest",
        model_version=MODEL_VERSION,
        feature_cols=feature_cols,
        mae_mw=metrics["mae_mw"],
        mape=metrics["mape"],
    )


@app.post("/v1/predict", response_model=PredictionResponse)
def predict_load(request: PredictionRequest) -> PredictionResponse:
    row = pd.DataFrame(
        [{
            "hour": request.timestamp.hour,
            "dow": request.timestamp.weekday(),
            "month": request.timestamp.month,
            "is_weekend": int(request.timestamp.weekday() >= 5),
            "temp_Madrid": request.temp_madrid,
            "temp_Barcelona": request.temp_barcelona,
            "temp_Valencia": request.temp_valencia,
            "temp_Seville": request.temp_seville,
            "temp_Bilbao": request.temp_bilbao,
            "load_lag_24h": request.load_24h_ago,
            "load_lag_168h": request.load_168h_ago,
        }],
        index=[request.timestamp],
    )
    row = add_cyclical_features(row)

    y_pred = predict(model, row, feature_cols)

    return PredictionResponse(
        predicted_load_mw=float(y_pred.iloc[0]),
        timestamp=request.timestamp,
        model_version=MODEL_VERSION,
    )
