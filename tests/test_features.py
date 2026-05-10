from pathlib import Path

import pandas as pd


PROCESSED_DATA_PATH = Path("data/processed/processed_churn.csv")
TRAIN_DATA_PATH = Path("data/processed/train.csv")
TEST_DATA_PATH = Path("data/processed/test.csv")
FEATURE_LIST_PATH = Path("data/processed/feature_list.json")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")

TARGET_COLUMN = "churn_flag"

EXPECTED_ENGINEERED_COLUMNS = [
    "spend_per_visit",
    "spend_per_purchase",
    "engagement_to_recency_ratio",
    "discount_engagement_interaction",
    "cart_abandonment_engagement_gap",
    "customer_value_score",
    "recency_severity_score",
    "behavioral_churn_risk_score",
    "risk_segment",
    "revenue_at_risk",
]


def test_processed_data_exists():
    assert PROCESSED_DATA_PATH.exists(), (
        "Processed data not found. Run: python -m src.features.build_features"
    )


def test_feature_engineering_outputs_exist():
    expected_files = [
        PROCESSED_DATA_PATH,
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
        FEATURE_LIST_PATH,
        PREPROCESSOR_PATH,
    ]

    missing_files = [str(path) for path in expected_files if not path.exists()]

    assert not missing_files, f"Missing feature engineering outputs: {missing_files}"


def test_engineered_columns_exist():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    missing_columns = [
        col for col in EXPECTED_ENGINEERED_COLUMNS
        if col not in df.columns
    ]

    assert not missing_columns, f"Missing engineered columns: {missing_columns}"


def test_train_test_have_target_column():
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    assert TARGET_COLUMN in train_df.columns, "Target column missing in train data."
    assert TARGET_COLUMN in test_df.columns, "Target column missing in test data."


def test_train_test_are_not_empty():
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    assert not train_df.empty, "Train data should not be empty."
    assert not test_df.empty, "Test data should not be empty."