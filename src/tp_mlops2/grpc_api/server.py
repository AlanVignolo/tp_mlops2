import time
from concurrent import futures

import grpc

from tp_mlops2.features import add_cyclical_features
from tp_mlops2.grpc_api import scoring_pb2, scoring_pb2_grpc
from tp_mlops2.predict import MODEL_ALIAS, REGISTERED_MODEL_NAME, load_model, predict
import pandas as pd


class ScoringServicer(scoring_pb2_grpc.ScoringServicer):
    def __init__(self):
        self.model, self.feature_cols, _metrics = load_model()
        self.model_version = f"{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

    def _build_row(self, request):
        timestamp = pd.Timestamp(request.timestamp)
        row = pd.DataFrame(
            [{
                "hour": timestamp.hour,
                "dow": timestamp.weekday(),
                "month": timestamp.month,
                "is_weekend": int(timestamp.weekday() >= 5),
                "temp_Madrid": request.temp_madrid,
                "temp_Barcelona": request.temp_barcelona,
                "temp_Valencia": request.temp_valencia,
                "temp_Seville": request.temp_seville,
                "temp_Bilbao": request.temp_bilbao,
                "load_lag_24h": request.load_24h_ago,
                "load_lag_168h": request.load_168h_ago,
            }],
            index=[timestamp],
        )
        return add_cyclical_features(row)

    def Predict(self, request, context):
        row = self._build_row(request)
        y_pred = predict(self.model, row, self.feature_cols)
        return scoring_pb2.PredictResponse(
            predicted_load_mw=float(y_pred.iloc[0]),
            model_version=self.model_version,
        )

    def PredictBatch(self, request, context):
        for req in request.requests:
            row = self._build_row(req)
            y_pred = predict(self.model, row, self.feature_cols)
            yield scoring_pb2.PredictResponse(
                predicted_load_mw=float(y_pred.iloc[0]),
                model_version=self.model_version,
            )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    scoring_pb2_grpc.add_ScoringServicer_to_server(ScoringServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Servidor gRPC escuchando en el puerto 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
