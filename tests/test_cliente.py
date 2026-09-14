import requests

BASE_URL = "http://127.0.0.1:8000"

VALID_PAYLOAD = {
    "timestamp": "2018-06-15T14:00:00",
    "temp_madrid": 28.0,
    "temp_barcelona": 25.0,
    "temp_valencia": 27.0,
    "temp_seville": 32.0,
    "temp_bilbao": 20.0,
    "load_24h_ago": 30000,
    "load_168h_ago": 29500,
}

INVALID_PAYLOAD = {
    **VALID_PAYLOAD,
    "temp_madrid": "no-es-un-numero",
}

def test_valid_requests():
    response = requests.post(f"{BASE_URL}/v1/predict", json=VALID_PAYLOAD)
    print("Caso válido ->", response.status_code)
    print(response.json())
    assert response.status_code == 200


def test_invalid_request():
    response = requests.post(f"{BASE_URL}/v1/predict", json=INVALID_PAYLOAD)
    print("Caso inválido ->", response.status_code)
    print(response.json())
    assert response.status_code == 422
