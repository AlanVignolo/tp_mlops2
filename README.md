# TP MLOps II — Forecasting de demanda eléctrica (España)

TP integrador de Operaciones de Aprendizaje de Máquina II (CEIA/FIUBA). El problema es predecir la demanda eléctrica horaria de España (`total load actual`) con el dataset de Kaggle [energy-consumption-generation-prices-and-weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather), que trae consumo, generación, precios y clima de las 5 ciudades más grandes del país entre 2015 y 2018.

Apunto al nivel contenedores: todo corre en Docker Compose (Postgres, MinIO, MLflow, Neo4j, y dos APIs, una REST y una GraphQL). Lo fui armando por etapas siguiendo el orden de las clases.

## Setup local (sin Docker, para desarrollo)

Necesita [uv](https://docs.astral.sh/uv/) y Python 3.12.

```powershell
uv venv .venv --python 3.12
uv sync
```

Para los notebooks de `notebooks/`, registrar el kernel:

```powershell
uv run python -m ipykernel install --user --name=tp-mlops2 --display-name "Python (TP MLOps II)"
```

`data/raw/` y `data/processed/` no están en el repo (pesan bastante y se regeneran solos). Hay que bajar el dataset de Kaggle, poner `energy_dataset.csv` y `weather_features.csv` en `data/raw/`, y correr el entrenamiento (ver abajo).

## Levantar todo con Docker

Copiar `.env.example` a `.env` y completar las credenciales (usuario/password de Postgres, MinIO y Neo4j).

```powershell
docker compose up -d --build
```

Esto levanta 6 contenedores: `postgres` (guarda los experimentos/métricas de MLflow), `minio` (guarda los modelos serializados, habla el protocolo S3), `mlflow` (tracking + registry, puerto 5000), `neo4j` (grafo de linaje, puerto 7474 la consola / 7687 el driver), `api` (REST, puerto 8000) y `graphql-api` (puerto 8001).

La primera vez hay 3 pasos manuales:
1. Crear el bucket `mlflow-artifacts` en la consola de MinIO (`localhost:9001`).
2. Entrenar un modelo (siguiente sección) y, en la UI de MLflow (`localhost:5000`), asignarle el alias `production` a la versión que quiero servir.
3. Sembrar el grafo de linaje: `uv run python scripts/seed_neo4j.py`.

## Entrenar el modelo

```powershell
uv run python -m tp_mlops2.train
```

Hace tuning con `RandomizedSearchCV` sobre un Random Forest (`TimeSeriesSplit`, 5 folds), evalúa contra el año 2018 como test, y loguea todo a MLflow: hiperparámetros, métricas, y el modelo mismo (queda registrado como nueva versión en el Model Registry). No se guarda nada en `models/*.joblib` — el modelo vive en MLflow/MinIO.

Con los datos que tengo ahora da MAE ≈ 1736.8 MW / MAPE ≈ 6.00% contra el test de 2018, cerca del forecast oficial del operador de la red española.

Para no esperar el tuning completo (útil para probar que el pipeline no está roto), hay un modo rápido:

```powershell
$env:TP_MLOPS2_QUICK_TRAIN="1"
uv run python -m tp_mlops2.train
```

## API REST

```powershell
uv run uvicorn tp_mlops2.api.main:app --reload
```

Swagger en `localhost:8000/docs`. Tiene `GET /health`, `GET /model/info` (metadata + métricas del modelo activo) y `POST /v1/predict`, que recibe un timestamp, la temperatura de las 5 ciudades, y la demanda real de hace 24h y 168h (el modelo usa esos dos como lags), y devuelve la predicción.

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

Un dato mal tipado (por ejemplo una temperatura como string) tira 422, gracias a la validación de Pydantic.

## API GraphQL

```powershell
uv run uvicorn tp_mlops2.graphql_api.main:app --reload --port 8001
```

GraphiQL en `localhost:8001/graphql`. La query principal:

```graphql
{
  model(name: "random_forest_demanda") {
    name
    version
    metrics { maeMw mape rmseMw }
    lineage { name kind }
  }
}
```

`lineage` va a buscar a Neo4j en el momento y devuelve la cadena completa: dataset crudo → dataset limpio → features que usó el modelo → experimento → modelo.

**REST vs GraphQL, comparando el mismo dato:** `/model/info` en REST siempre devuelve todos los campos aunque solo me interese uno; en GraphQL pido exactamente lo que necesito. Para traer el linaje además de las métricas, en REST necesitaría un segundo endpoint y una segunda llamada; en GraphQL lo pido anidado en la misma query. REST usa el código HTTP para errores (422, 404, etc.); GraphQL casi siempre devuelve 200 y el error va adentro del JSON, en `"errors"` — hay que mirar el body, no alcanza con el status code. A cambio, GraphQL pide que el cliente sepa armar la query; para un caso tan chico como este la ventaja es marginal, se nota más cuando hay muchas entidades relacionadas entre sí.

## Grafo de linaje

`scripts/seed_neo4j.py` lee el modelo activo de MLflow y arma el grafo: nodos de dataset (crudo y limpio), uno por cada feature que usa el modelo, el experimento y el modelo, todos conectados. Se puede correr de nuevo sin duplicar nada (usa `MERGE`, no `CREATE`).

```powershell
uv run python scripts/seed_neo4j.py
```

## Estructura

```
tp_mlops2/
├── data/                # raw/ y processed/, no versionados
├── notebooks/            # EDA y análisis, no es código de producción
├── scripts/
│   └── seed_neo4j.py
├── src/tp_mlops2/
│   ├── data.py           # limpieza
│   ├── features.py       # cíclicas + lags
│   ├── train.py          # tuning + logging a MLflow
│   ├── predict.py        # carga el modelo desde el registry
│   ├── api/               # REST
│   └── graphql_api/       # GraphQL
├── tests/test_cliente.py
├── docker-compose.yml
└── Dockerfile.api / Dockerfile.mlflow / Dockerfile.graphql
```

El pipeline (`data.py` → `features.py` → `train.py`/`predict.py`) es independiente de los notebooks. Los notebooks quedaron solo para el análisis exploratorio, no generan nada que use el sistema en producción.

## Tests

Con la API REST y la GraphQL corriendo:

```powershell
uv run pytest tests/test_cliente.py -v
```

Prueba un caso válido y uno inválido contra REST, y una query contra GraphQL.

## CI

En cada push a `main`, GitHub Actions corre el lint (Ruff), baja el dataset de Kaggle, entrena en modo rápido, levanta la API y corre los tests contra un servidor real.
