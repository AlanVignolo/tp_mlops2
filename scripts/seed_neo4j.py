import os

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from neo4j import GraphDatabase

from tp_mlops2.features import FEATURE_COLUMNS
from tp_mlops2.predict import MLFLOW_TRACKING_URI, MODEL_ALIAS, REGISTERED_MODEL_NAME

load_dotenv()

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]


def seed():
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    model_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    run = client.get_run(model_version.run_id)

    model_name = f"{REGISTERED_MODEL_NAME} v{model_version.version}"
    experiment_name = f"run {model_version.run_id[:8]}"
    metrics = run.data.metrics

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        session.run(
            """
            MERGE (raw:Dataset {name: 'energy_dataset_raw'})
            MERGE (clean:Dataset {name: 'energy_weather_clean'})
            MERGE (raw)-[:DERIVES]->(clean)
            MERGE (exp:Experiment {name: $experiment_name, run_id: $run_id})
            MERGE (model:Model {name: $model_name, version: $version, mae_mw: $mae_mw, mape: $mape})
            MERGE (exp)-[:PRODUCES]->(model)
            """,
            experiment_name=experiment_name,
            run_id=model_version.run_id,
            model_name=REGISTERED_MODEL_NAME,
            version=model_version.version,
            mae_mw=metrics.get("mae_mw"),
            mape=metrics.get("mape"),
        )

        for feature_name in FEATURE_COLUMNS:
            session.run(
                """
                MATCH (clean:Dataset {name: 'energy_weather_clean'})
                MATCH (exp:Experiment {name: $experiment_name})
                MERGE (f:Feature {name: $feature_name})
                MERGE (clean)-[:HAS_FEATURE]->(f)
                MERGE (f)-[:USED_IN]->(exp)
                """,
                experiment_name=experiment_name,
                feature_name=feature_name,
            )

    driver.close()
    print(
        f"Grafo sembrado: {len(FEATURE_COLUMNS)} features, "
        f"model {model_name}, run {model_version.run_id}"
    )



if __name__ == "__main__":
    seed()
