import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

from tp_mlops2.graphql_api.schema import schema

app = FastAPI(title="Demanda Electrica España - GraphQL API")
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

