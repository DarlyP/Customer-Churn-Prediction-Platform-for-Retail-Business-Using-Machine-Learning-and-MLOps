from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app


MODEL_PATH = Path("models/best_model.pkl")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")

client = TestClient(app)


@pytest.fixture
def sample_payload():
    return {
        "customer_id": "CUST_API_001",
        "age_group": "35-44",
        "gender": "Female",
        "region": "West",
        "customer_segment": "Returning",
        "preferred_channel": "Mobile App",
        "purchase_frequency": 3,
        "avg_order_value": 45.5,
        "total_spent": 1250.0,
        "recency_days": 75,
        "website_visits": 20,
        "discount_usage_rate": 0.65,
        "email_open_rate": 0.30,
        "cart_abandonment_rate": 0.72,
        "loyalty_score": 35,
        "engagement_score": 40,
    }


def test_health_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert "message" in data


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert "model_path" in data
    assert "preprocessor_path" in data
    assert "model_available" in data
    assert "preprocessor_available" in data


def test_predict_endpoint_schema(sample_payload):
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        pytest.skip("Model or preprocessor not available. Run dvc repro first.")

    response = client.post("/predict", json=sample_payload)

    assert response.status_code == 200

    data = response.json()

    expected_fields = [
        "customer_id",
        "churn_probability",
        "prediction_label",
        "prediction",
        "risk_level",
        "recommended_action",
    ]

    missing_fields = [
        field for field in expected_fields
        if field not in data
    ]

    assert not missing_fields, f"Missing API response fields: {missing_fields}"
    assert 0 <= data["churn_probability"] <= 1
    assert data["prediction_label"] in [0, 1]