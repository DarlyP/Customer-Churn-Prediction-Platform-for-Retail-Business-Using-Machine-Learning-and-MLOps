from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/retail_churn.csv")
INTERIM_DATA_PATH = Path("data/interim/cleaned_raw.csv")
PROCESSED_DATA_PATH = Path("data/processed/processed_churn.csv")

TARGET_COLUMN = "churn_flag"

REQUIRED_COLUMNS = [
    "customer_id",
    "age_group",
    "gender",
    "region",
    "customer_segment",
    "preferred_channel",
    "purchase_frequency",
    "avg_order_value",
    "total_spent",
    "recency_days",
    "website_visits",
    "discount_usage_rate",
    "email_open_rate",
    "cart_abandonment_rate",
    "loyalty_score",
    "engagement_score",
    "churn_risk",
    "churn_flag",
]


def get_available_dataset_path() -> Path:
    for path in [PROCESSED_DATA_PATH, INTERIM_DATA_PATH, RAW_DATA_PATH]:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No dataset found. Run data pipeline first: "
        "python -m src.data.load_data && python -m src.features.build_features"
    )


def test_dataset_is_not_empty():
    data_path = get_available_dataset_path()
    df = pd.read_csv(data_path)

    assert not df.empty, "Dataset should not be empty."
    assert df.shape[0] > 0, "Dataset should contain at least one row."
    assert df.shape[1] > 0, "Dataset should contain at least one column."


def test_target_column_exists():
    data_path = get_available_dataset_path()
    df = pd.read_csv(data_path)

    assert TARGET_COLUMN in df.columns, f"Target column `{TARGET_COLUMN}` is missing."


def test_required_columns_exist():
    data_path = get_available_dataset_path()
    df = pd.read_csv(data_path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    assert not missing_columns, f"Missing required columns: {missing_columns}"


def test_customer_id_has_no_missing_values():
    data_path = get_available_dataset_path()
    df = pd.read_csv(data_path)

    assert "customer_id" in df.columns, "`customer_id` column is missing."
    assert df["customer_id"].isna().sum() == 0, "`customer_id` should not contain missing values."