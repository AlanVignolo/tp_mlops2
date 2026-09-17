# TP MLOps II — Forecasting de demanda eléctrica (España)

Predicción de la demanda eléctrica horaria de España (`total load actual`), usando el dataset [ENTSO-E / Kaggle](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) (consumo, generación, precios y clima de las 5 ciudades más grandes de España, 2015-2018).

Trabajo práctico integrador de *Operaciones de Aprendizaje de Máquina II* (CEIA/FIUBA). Nivel objetivo: **contenedores** (Docker + MLflow + PostgreSQL + MinIO + Neo4j + FastAPI/GraphQL) — en construcción por etapas siguiendo el curso.

## Arquitectura

```
                    ┌─────────────┐
                    │  PostgreSQL │◄──── backend store (experimentos/métricas)
                    └─────────────┘
                           ▲
┌──────────┐        ┌─────┴──────┐        ┌───────┐
│  MinIO   │◄───────┤   MLflow   │        │ Neo4j │◄──── grafo de linaje
│ (S3 art.)│        │ (tracking  │        └───┬───┘
└──────────┘        │ + registry)│            │
                     └─────┬──────┘            │
                           │                    │
              ┌────────────┼────────────┐       │
              ▼                          ▼       │
        ┌───────────┐            ┌─────────────┐ │
        │ API REST  │            │ API GraphQL │─┘
        │ (FastAPI) │            │ (Strawberry)│
        │  :8000    │            │   :8001     │
        └───────────┘            └─────────────┘
```

Todos los servicios corren en Docker Compose. La API REST y la API GraphQL cargan el modelo desde el **MLflow Model Registry** (alias `production`), no desde archivos locales.

## Setup local (desarrollo, sin Docker)

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.12.

```powershell
uv venv .venv --python 3.12
uv sync
```

Para abrir los notebooks de `notebooks/` con este entorno:

```powershell
uv run python -m ipykernel install --user --name=tp-mlops2 --display-name "Python (TP MLOps II)"
```

### Datos

`data/raw/` y `data/processed/` no se versionan en git (pesan varias decenas de MB y son regenerables):

1. Descargar el dataset de Kaggle: [energy-consumption-generation-prices-and-weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather).
2. Colocar `energy_dataset.csv` y `weather_features.csv` en `data/raw/`.
3. Correr `uv run python -m tp_mlops2.train` (ver [Entrenar el modelo](#entrenar-el-modelo)).

## Infraestructura (Docker Compose)

Copiá `.env.example` a `.env` y completá los valores (credenciales de Postgres, MinIO y Neo4j — no se versionan):

```powershell
cp .env.example .env
```

Levantar todo el stack:

```powershell
docker compose up -d --build
docker compose ps
```

Servicios y puertos:

| Servicio | Puerto(s) | Qué es |
|---|---|---|
| `postgres` | 5432 | Backend store de MLflow (experimentos, runs, registry) |
| `minio` | 9000 (API), 9001 (consola) | Artifact store S3-compatible de MLflow (modelos serializados) |
| `mlflow` | 5000 | Servidor de tracking + Model Registry |
| `neo4j` | 7474 (consola), 7687 (Bolt) | Grafo de linaje (dataset → features → experimento → modelo) |
| `api` | 8000 | API REST (FastAPI) |
| `graphql-api` | 8001 | API GraphQL (Strawberry) |

Primer arranque — pasos manuales una sola vez:

1. Crear el bucket `mlflow-artifacts` en la consola de MinIO (`http://localhost:9001`).
2. Entrenar y registrar un modelo (ver abajo).
3. Promover la versión a producción: en `http://localhost:5000` → **Models** → `random_forest_demanda` → asignar el **alias `production`** a la versión deseada.
4. Sembrar el grafo de linaje: `uv run python scripts/seed_neo4j.py`.

## Entrenar el modelo

```powershell
uv run python -m tp_mlops2.train
```

Corre un `RandomizedSearchCV` (Random Forest, `TimeSeriesSplit` de 5 folds) sobre el dataset limpio, evalúa en el split de test (año 2018), y **loguea todo a MLflow**: parámetros, métricas y el modelo (registrado como nueva versión de `random_forest_demanda` en el Model Registry). Ya no se guardan artefactos `.joblib` locales — todo vive en MLflow/MinIO.

Métrica de referencia actual: **MAE ≈ 1736.8 MW / MAPE ≈ 6.00%** en test 2018 (compite cabeza a cabeza con el forecast oficial del operador de red, TSO).

Para un smoke-test rápido del pipeline (sin buscar buenos hiperparámetros — útil en CI):

```powershell
$env:TP_MLOPS2_QUICK_TRAIN="1"
uv run python -m tp_mlops2.train
```

## API REST

```powershell
uv run uvicorn tp_mlops2.api.main:app --reload
```

Documentación interactiva (Swagger) en [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Endpoints

- `GET /health` — chequeo de disponibilidad.
- `GET /model/info` — nombre, versión y métricas del modelo servido (leídas del registry).
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

## API GraphQL

```powershell
uv run uvicorn tp_mlops2.graphql_api.main:app --reload --port 8001
```

Interfaz interactiva (GraphiQL) en [http://127.0.0.1:8001/graphql](http://127.0.0.1:8001/graphql).

Ejemplo de query:

```graphql
{
  model(name: "random_forest_demanda") {
    name
    version
    metrics {
      maeMw
      mape
      rmseMw
    }
    lineage {
      name
      kind
    }
  }
}
```

`lineage` recorre el grafo de Neo4j en tiempo real (relaciones `DERIVES`/`HAS_FEATURE`/`USED_IN`/`PRODUCES`), devolviendo la cadena completa: dataset crudo → dataset limpio → features usadas → experimento → modelo.

### REST vs GraphQL — comparación

Para leer el mismo dato (metadata + métricas del modelo), `GET /model/info` (REST) y la query `model { name version metrics { ... } }` (GraphQL) devuelven equivalentes. Las diferencias observadas:

- **Forma de la respuesta:** REST siempre trae *todos* los campos definidos en `ModelInfo` (`model_name`, `model_version`, `feature_cols`, `mae_mw`, `mape`) aunque el cliente solo necesite uno. GraphQL deja pedir exactamente los campos que hacen falta (ej. solo `mape`), evitando over-fetching.
- **Datos relacionados:** para traer el *linaje* del modelo además de sus métricas, REST necesitaría un segundo endpoint (`/model/lineage`) y una segunda llamada HTTP. GraphQL lo resuelve en una sola query, pidiendo `lineage { name kind }` anidado dentro de `model` — el resolver decide en el servidor cómo buscarlo (en este caso, consultando Neo4j), pero el cliente hace una sola llamada.
- **Códigos de estado:** REST usa el código HTTP para señalar éxito/error (`200`, `422`, `404`, etc.). GraphQL responde casi siempre `200 OK`, y los errores viajan dentro del body, en una clave `"errors"` — el cliente tiene que inspeccionar el JSON, no solo el status code.
- **Costo de la flexibilidad:** GraphQL requiere que el cliente sepa escribir una query (o tener un cliente que la genere), mientras que REST es más directo de consumir con un simple `GET`. Para un endpoint tan chico como este, la ganancia de GraphQL es modesta; se vuelve más valiosa a medida que el grafo de datos relacionados crece (más entidades, más relaciones entre ellas).

## Grafo de linaje (Neo4j)

`scripts/seed_neo4j.py` construye el grafo reflejando el pipeline real: lee el modelo/run activo desde MLflow y crea los nodos `Dataset` (raw y processed), `Feature` (las 14 del modelo) y `Experiment`/`Model`, conectados por las relaciones que el resolver `lineage` de GraphQL recorre. Es idempotente — se puede re-ejecutar sin duplicar nodos.

```powershell
uv run python scripts/seed_neo4j.py
```

Consola de Neo4j: [http://localhost:7474](http://localhost:7474).

## Estructura del proyecto

```
tp_mlops2/
├── data/
│   ├── raw/                    # CSVs originales (no versionado)
│   └── processed/               # dataset limpio (no versionado)
├── notebooks/                   # análisis exploratorio — no producción
├── scripts/
│   └── seed_neo4j.py            # siembra el grafo de linaje desde MLflow
├── src/tp_mlops2/
│   ├── data.py                  # carga y limpieza de datos crudos
│   ├── features.py               # feature engineering (cíclicas, lags)
│   ├── train.py                  # tuning + entrenamiento + logging a MLflow
│   ├── predict.py                # carga de modelo desde el registry + inferencia
│   ├── api/                      # API REST (FastAPI)
│   │   ├── schemas.py
│   │   └── main.py
│   └── graphql_api/              # API GraphQL (Strawberry)
│       ├── schema.py
│       └── main.py
├── tests/
│   └── test_cliente.py          # cliente de prueba (REST + GraphQL)
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.mlflow
└── .github/workflows/ci.yml     # lint + smoke test en cada push
```

El pipeline (`data.py` → `features.py` → `train.py`/`predict.py`) es código reproducible, independiente de los notebooks — estos últimos se usan solo para exploración y análisis, no para generar artefactos de producción.

## Tests

Con la API REST (puerto 8000) y la API GraphQL (puerto 8001) corriendo:

```powershell
uv run pytest tests/test_cliente.py -v
```

Verifica: caso válido REST (`200`), caso inválido REST (`422`), y una query GraphQL válida (`200`, sin `errors`).

## CI

`.github/workflows/ci.yml` corre en cada push/PR a `main`: lint con Ruff, descarga del dataset (Kaggle API vía GitHub Secrets), entrenamiento en modo rápido, y los tests de la API REST contra un servidor real.
