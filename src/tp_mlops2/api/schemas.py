from datetime import datetime

from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    timestamp: datetime = Field(..., description="Momento para el qeu se predice la demanda")
    temp_madrid: float = Field(..., description="Temperatura en Madrid, en Celsius")
    temp_barcelona: float
    temp_valencia: float
    temp_seville: float
    temp_bilbao: float
    load_24h_ago: float = Field(..., description="Demanda real (MW) hace 24 horas")
    load_168h_ago: float = Field(..., description="Demanda real (MW) hace 168 horas (1 semana)")
    
class PredictionResponse(BaseModel):
    predicted_load_mw: float
    timestamp: datetime
    model_version: str
    
class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    feature_cols: list[str]
    mae_mw: float
    mape: float
