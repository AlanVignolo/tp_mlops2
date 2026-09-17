import statistics
import time

import grpc
import requests

from tp_mlops2.grpc_api import scoring_pb2, scoring_pb2_grpc

N_REQUESTS = 50

REST_URL = "http://localhost:8000/v1/predict"
GRPC_TARGET = "localhost:50051"

SAMPLE_REQUEST = {
    "timestamp": "2018-06-15T14:00:00",
    "temp_madrid": 28.0,
    "temp_barcelona": 25.0,
    "temp_valencia": 27.0,
    "temp_seville": 32.0,
    "temp_bilbao": 20.0,
    "load_24h_ago": 30000.0,
    "load_168h_ago": 29500.0,
}


def benchmark_rest(n=N_REQUESTS):
    times = []
    for _ in range(n):
        start = time.perf_counter()
        requests.post(REST_URL, json=SAMPLE_REQUEST)
        times.append((time.perf_counter() - start) * 1000)
    return times


def benchmark_grpc(n=N_REQUESTS):
    times = []
    with grpc.insecure_channel(GRPC_TARGET) as channel:
        stub = scoring_pb2_grpc.ScoringStub(channel)
        request = scoring_pb2.PredictRequest(**SAMPLE_REQUEST)
        for _ in range(n):
            start = time.perf_counter()
            stub.Predict(request)
            times.append((time.perf_counter() - start) * 1000)
    return times


def summarize(name, times):
    print(f"{name}: media {statistics.mean(times):.2f} ms | "
          f"mediana {statistics.median(times):.2f} ms | "
          f"p95 {sorted(times)[int(len(times) * 0.95)]:.2f} ms")


if __name__ == "__main__":
    print(f"Benchmark con {N_REQUESTS} requests cada uno...\n")

    rest_times = benchmark_rest()
    summarize("REST", rest_times)

    grpc_times = benchmark_grpc()
    summarize("gRPC", grpc_times)
