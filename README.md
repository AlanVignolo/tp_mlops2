# TP MLOps II — demanda eléctrica España

TP integrador de Operaciones de Aprendizaje de Máquina II (CEIA/FIUBA). La idea es predecir la demanda eléctrica horaria de España (`total load actual`) con el dataset de Kaggle [energy-consumption-generation-prices-and-weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather): consumo, generación, precios y clima de las 5 ciudades más grandes del país, 2015-2018.

Voy por el nivel contenedores del TP: todo corre en Docker Compose (Postgres, MinIO, MLflow, Neo4j, y dos APIs — REST y GraphQL). Lo fui armando de a poco, mini-TP por mini-TP, según el orden del curso.

## Setup local (sin Docker)

Necesita [uv](https://docs.astral.sh/uv/) y Python 3.12.

```powershell
uv venv .venv --python 3.12
uv sync
```

Para los notebooks, registro el kernel:

```powershell
uv run python -m ipykernel install --user --name=tp-mlops2 --display-name "Python (TP MLOps II)"
```

`data/raw/` y `data/processed/` no están en el repo, pesan bastante. Hay que bajar el dataset de Kaggle y poner `energy_dataset.csv` y `weather_features.csv` en `data/raw/` antes de entrenar.

## Levantar con Docker

Copiar `.env.example` a `.env` y completar las credenciales.

```powershell
docker compose up -d --build
```

Levanta 6 contenedores: `postgres` (guarda experimentos/métricas de MLflow), `minio` (guarda los modelos serializados, S3-compatible), `mlflow` (tracking + registry, puerto 5000), `neo4j` (grafo de linaje, 7474 consola / 7687 driver), `api` (REST, 8000) y `graphql-api` (8001).

La primera vez hay que hacer 3 cosas a mano:
1. Crear el bucket `mlflow-artifacts` en MinIO (`localhost:9001`).
2. Entrenar un modelo y, en la UI de MLflow (`localhost:5000`), ponerle el alias `production` a la versión que quiero servir.
3. Sembrar el grafo de linaje: `uv run python scripts/seed_neo4j.py`.

## Entrenar

```powershell
uv run python -m tp_mlops2.train
```

Tuning con `RandomizedSearchCV` sobre un Random Forest (`TimeSeriesSplit`, 5 folds), evaluado contra 2018 como test. Loguea todo a MLflow — hiperparámetros, métricas y el modelo, que queda registrado como nueva versión. Ya no guarda nada en `.joblib` local, todo vive en MLflow/MinIO.

Ahora mismo da MAE ≈ 1736.8 MW / MAPE ≈ 6% contra 2018, bastante cerca del forecast oficial del operador de red español.

Hay un modo rápido para no esperar el tuning completo (útil para chequear que el pipeline no se rompió):

```powershell
$env:TP_MLOPS2_QUICK_TRAIN="1"
uv run python -m tp_mlops2.train
```

## API REST

```powershell
uv run uvicorn tp_mlops2.api.main:app --reload
```

Swagger en `localhost:8000/docs`. `GET /health`, `GET /model/info` (metadata + métricas del modelo activo), `POST /v1/predict` (timestamp + temperatura de las 5 ciudades + demanda de hace 24h y 168h, devuelve la predicción).

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

Si mando algo mal tipado tira 422 (validación de Pydantic).

## API GraphQL

```powershell
uv run uvicorn tp_mlops2.graphql_api.main:app --reload --port 8001
```

GraphiQL en `localhost:8001/graphql`.

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

`lineage` consulta Neo4j al momento y trae la cadena completa: dataset crudo → dataset limpio → features del modelo → experimento → modelo.

**REST vs GraphQL con el mismo dato:** `/model/info` en REST siempre trae todos los campos aunque me interese uno solo; en GraphQL pido justo lo que necesito. Para traer el linaje además de las métricas, en REST tendría que pegarle a un segundo endpoint; en GraphQL lo pido anidado en la misma query. REST usa el código HTTP para errores; GraphQL casi siempre devuelve 200 y el error viaja adentro del JSON (`"errors"`), así que hay que mirar el body. A cambio GraphQL exige que el cliente sepa armar la query — para algo tan chico como esto la ventaja no es enorme, se nota más con muchas entidades relacionadas.

## API gRPC

```powershell
uv run python -m tp_mlops2.grpc_api.server
```

Queda escuchando en el puerto 50051, con el modelo cargado una sola vez al arrancar. El contrato está en `scoring.proto`: `Predict` (unary, una request → una response) y `PredictBatch` (server-streaming, mando varios items en un solo request y recibo las predicciones de a una a medida que están listas).

```powershell
uv run python -m tp_mlops2.grpc_api.client
```

Prueba ambos métodos contra el servidor.

**Latencia gRPC vs REST**, 50 requests a cada uno (`scripts/benchmark_latency.py`), misma predicción en los dos casos:

```
REST: media 212.20 ms | mediana 95.40 ms | p95 127.14 ms
gRPC: media  71.11 ms | mediana 72.22 ms | p95  93.49 ms
```

gRPC dio bastante más rápido y más estable (la media de REST se aleja mucho de su mediana, señal de que hubo algunas requests lentas sueltas; gRPC se mantuvo parejo). Tiene sentido con lo que vimos en la teoría: gRPC usa HTTP/2 (multiplexado, binario) y Protobuf en vez de JSON sobre HTTP/1.1, así que paga menos overhead de serialización y de conexión por request. La contra es que perdés la inspección fácil que tenés con REST (no puedo pegarle con curl o abrir `/docs` en el navegador) y hay que generar y mantener los stubs cuando cambia el contrato.

```powershell
uv run python scripts/benchmark_latency.py
```

## Linaje (Neo4j)

`scripts/seed_neo4j.py` lee el modelo activo de MLflow y arma el grafo: dataset crudo, dataset limpio, una feature por cada columna que usa el modelo, el experimento y el modelo, todo conectado. Se puede correr de nuevo sin duplicar nada.

```powershell
uv run python scripts/seed_neo4j.py
```

## Estructura

```
tp_mlops2/
├── data/                 # raw/ y processed/, no versionados
├── notebooks/             # EDA, no es código de producción
├── scripts/seed_neo4j.py
├── src/tp_mlops2/
│   ├── data.py            # limpieza
│   ├── features.py        # cíclicas + lags
│   ├── train.py           # tuning + logging a MLflow
│   ├── predict.py         # carga el modelo desde el registry
│   ├── api/                # REST
│   └── graphql_api/        # GraphQL
├── tests/test_cliente.py
├── docker-compose.yml
└── Dockerfile.api / Dockerfile.mlflow / Dockerfile.graphql
```

El pipeline (`data.py` → `features.py` → `train.py`/`predict.py`) no depende de los notebooks para nada — esos quedaron solo para el análisis exploratorio.

## Tests

Con la API REST y la GraphQL corriendo:

```powershell
uv run pytest tests/test_cliente.py -v
```

Prueba un caso válido y uno inválido contra REST, y una query contra GraphQL.

## CI

En cada push a `main`, GitHub Actions corre el lint (Ruff) y chequea que los módulos del pipeline importen bien. No entrena ni levanta las APIs porque eso requiere Postgres/MinIO/MLflow corriendo, y todavía no vale la pena levantar todo ese stack solo para el CI.
