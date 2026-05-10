from pathlib import Path

import pandas as pd
import pytest

from src.models.predict_model import predict_customer_churn


MODEL_PATH = Path("models/best_model.pkl")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")


@pytest.fixture
def sample_customer():
    return {
        "customer_id": "CUST_TEST_001",
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


def test_model_and_preprocessor_exist():
    assert MODEL_PATH.exists(), "Best model file is missing: models/best_model.pkl"
    assert PREPROCESSOR_PATH.exists(), "Preprocessor file is missing: models/preprocessor.pkl"


def test_prediction_output_schema(sample_customer):
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        pytest.skip("Model or preprocessor not available. Run dvc repro first.")

    prediction_df = predict_customer_churn(sample_customer)

    expected_columns = [
        "customer_id",
        "churn_probability",
        "prediction_label",
        "prediction_text",
        "risk_level",
        "recommended_action",
    ]

    missing_columns = [
        col for col in expected_columns
        if col not in prediction_df.columns
    ]

    assert not missing_columns, f"Missing prediction output columns: {missing_columns}"


def test_prediction_probability_range(sample_customer):
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        pytest.skip("Model or preprocessor not available. Run dvc repro first.")

    prediction_df = predict_customer_churn(sample_customer)

    probability = prediction_df.loc[0, "churn_probability"]

    assert 0 <= probability <= 1, "Churn probability must be between 0 and 1."


def test_prediction_label_is_binary(sample_customer):
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        pytest.skip("Model or preprocessor not available. Run dvc repro first.")

    prediction_df = predict_customer_churn(sample_customer)

    prediction_label = prediction_df.loc[0, "prediction_label"]

    assert prediction_label in [0, 1], "Prediction label must be 0 or 1."


def test_risk_level_is_valid(sample_customer):
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        pytest.skip("Model or preprocessor not available. Run dvc repro first.")

    prediction_df = predict_customer_churn(sample_customer)

    risk_level = prediction_df.loc[0, "risk_level"]

    valid_risk_levels = [
        "Low Risk",
        "Medium Risk",
        "High Risk",
        "Critical Risk",
    ]

    assert risk_level in valid_risk_levels, f"Invalid risk level: {risk_level}"