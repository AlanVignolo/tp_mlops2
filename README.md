# TP MLOps II — Forecasting de demanda eléctrica (España)

Predicción de la demanda eléctrica horaria de España (`total load actual`), usando el dataset [ENTSO-E / Kaggle](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) (consumo, generación, precios y clima de las 5 ciudades más grandes de España, 2015-2018).

Trabajo práctico integrador de *Operaciones de Aprendizaje de Máquina II* (CEIA/FIUBA). Nivel objetivo: **contenedores** (Docker + Airflow + MLflow + PostgreSQL + MinIO + FastAPI) — en construcción por etapas siguiendo el curso.

## Setup

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.12.

```powershell
uv venv .venv --python 3.12
uv sync
```

Esto instala el paquete `tp_mlops2` en modo editable junto con sus dependencias (`numpy`, `pandas`, `scikit-learn`, `fastapi`, etc.) y el grupo de desarrollo (`jupyter`, `pytest`, `requests`).

Para abrir los notebooks de `notebooks/` con este entorno:

```powershell
uv run python -m ipykernel install --user --name=tp-mlops2 --display-name "Python (TP MLOps II)"
```

### Datos

`data/raw/`, `data/processed/` y `models/*.joblib` no se versionan en git (pesan varias decenas de MB y son regenerables). Para reproducir el proyecto desde cero:

1. Descargar el dataset de Kaggle: [energy-consumption-generation-prices-and-weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather).
2. Colocar `energy_dataset.csv` y `weather_features.csv` en `data/raw/`.
3. Correr `uv run python -m tp_mlops2.train` (ver [Entrenar el modelo](#entrenar-el-modelo)) — genera `data/processed/` y `models/` automáticamente.

## Estructura del proyecto

```
tp_mlops2/
├── data/
│   ├── raw/            # CSVs originales (energy_dataset.csv, weather_features.csv)
│   └── processed/       # dataset limpio (energy_weather_clean.parquet)
├── models/               # artefactos del modelo entrenado (.joblib)
├── notebooks/            # análisis exploratorio (EDA, comparación de modelos) — no producción
├── src/tp_mlops2/
│   ├── data.py           # carga y limpieza de datos crudos
│   ├── features.py       # feature engineering (cíclicas, lags)
│   ├── train.py          # tuning + entrenamiento + guardado de artefactos
│   ├── predict.py        # carga de modelo + inferencia
│   └── api/
│       ├── schemas.py     # contratos Pydantic (request/response)
│       └── main.py        # app FastAPI
└── tests/
    └── test_cliente.py   # cliente de prueba de la API (caso válido + caso inválido)
```

El pipeline (`data.py` → `features.py` → `train.py`/`predict.py`) es código reproducible, independiente de los notebooks — estos últimos se usan solo para exploración y análisis, no para generar artefactos de producción.

## Entrenar el modelo

```powershell
uv run python -m tp_mlops2.train
```

Corre un `RandomizedSearchCV` (Random Forest, `TimeSeriesSplit` de 5 folds) sobre el dataset limpio, evalúa en el split de test (año 2018) y guarda en `models/`:

- `random_forest_v1.joblib` — modelo entrenado
- `feature_cols_v1.joblib` — orden de features esperado por el modelo
- `metrics_v1.joblib` — métricas de evaluación (MAE, MAPE, RMSE)

Métrica de referencia actual: **MAE ≈ 1736.8 MW / MAPE ≈ 6.00%** en test 2018 (compite cabeza a cabeza con el forecast oficial del operador de red, TSO).

## Levantar la API

```powershell
uv run uvicorn tp_mlops2.api.main:app --reload
```

Documentación interactiva (Swagger) en [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Endpoints

- `GET /health` — chequeo de disponibilidad.
- `GET /model/info` — nombre, versión y métricas del modelo servido.
- `POST /v1/predict` — predicción de demanda dado un timestamp, temperaturas por ciudad y la demanda real de hace 24h/168h.

Ejemplo de request válido:

```json
{
  "timestamp": "2018-06-15T14:00:00",
  "temp_madrid": 28.0,
  "temp_barcelona": 25.0,
  "temp_valencia": 27.0,
  "temp_seville": 32.0,
  "temp_bilbao": 20.0,
  "load_24h_ago": 30000,
  "load_168h_ago": 29500
}
```

Un request con un tipo de dato inválido (ej. `"temp_madrid": "abc"`) es rechazado con `422 Unprocessable Entity` por la validación de Pydantic.

## Tests

Con la API corriendo en otra terminal:

```powershell
uv run pytest tests/test_cliente.py -v
```

Verifica que un caso válido responda `200` y que un caso inválido responda `422`.
