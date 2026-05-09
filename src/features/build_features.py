from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Path Configuration
# ============================================================

INPUT_DATA_PATH = "data/interim/cleaned_raw.csv"

PROCESSED_DATA_PATH = "data/processed/processed_churn.csv"
TRAIN_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH = "data/processed/test.csv"
FEATURE_LIST_PATH = "data/processed/feature_list.json"

PREPROCESSOR_PATH = "models/preprocessor.pkl"

TARGET_COLUMN = "churn_flag"
ID_COLUMN = "customer_id"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ============================================================
# Column Configuration
# ============================================================

BASE_CATEGORICAL_COLUMNS = [
    "age_group",
    "gender",
    "region",
    "customer_segment",
    "preferred_channel",
]

RATE_COLUMNS = [
    "discount_usage_rate",
    "email_open_rate",
    "cart_abandonment_rate",
]

SCORE_COLUMNS = [
    "loyalty_score",
    "engagement_score",
]

LEAKAGE_COLUMNS = [
    ID_COLUMN,
    TARGET_COLUMN,
    "churn_risk",
    "revenue_at_risk",
]


# ============================================================
# Load Data
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input data not found: {path}")

    df = pd.read_csv(file_path)

    return df


# ============================================================
# Target Handling
# ============================================================

def normalize_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure target column is numeric binary: 0 or 1.
    """
    df = df.copy()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    if df[TARGET_COLUMN].dtype == "object":
        target_mapping = {
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "churn": 1,
            "not churn": 0,
            "1": 1,
            "0": 0,
        }

        df[TARGET_COLUMN] = (
            df[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(target_mapping)
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Target column contains unmapped or missing values: {TARGET_COLUMN}"
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df


# ============================================================
# Missing Value Handling
# ============================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values before feature engineering.
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in numeric_cols:
        if col == TARGET_COLUMN:
            continue

        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if col == ID_COLUMN:
            continue

        df[col] = df[col].fillna("Unknown")

    return df


def clean_infinite_and_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean infinite values and remaining missing values after feature engineering.
    """
    df = df.copy()

    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in numeric_cols:
        if col == TARGET_COLUMN:
            continue

        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if col == ID_COLUMN:
            continue

        df[col] = df[col].astype(str).fillna("Unknown")

    return df


# ============================================================
# Rate and Score Normalization
# ============================================================

def normalize_rate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure rate columns are in 0-1 scale.
    If a rate column appears to be in 0-100 scale, convert it to 0-1.
    """
    df = df.copy()

    for col in RATE_COLUMNS:
        if col not in df.columns:
            continue

        max_value = df[col].max()

        if max_value > 1:
            df[col] = df[col] / 100

        df[col] = df[col].clip(lower=0, upper=1)

    return df


def normalize_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure score columns are in 0-100 scale.
    """
    df = df.copy()

    for col in SCORE_COLUMNS:
        if col not in df.columns:
            continue

        df[col] = df[col].clip(lower=0, upper=100)

    return df


# ============================================================
# Advanced Business Feature Engineering
# ============================================================

def create_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional business-oriented features based on customer churn EDA findings.

    Important:
    - This function does not use churn_flag or churn_risk.
    - Features are created only from customer behavior, spending, engagement, and loyalty variables.
    """
    df = df.copy()

    required_cols = [
        "total_spent",
        "website_visits",
        "purchase_frequency",
        "engagement_score",
        "recency_days",
        "discount_usage_rate",
        "cart_abandonment_rate",
        "avg_order_value",
        "loyalty_score",
        "email_open_rate",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # ------------------------------------------------------------
    # 1. Existing business ratio features
    # ------------------------------------------------------------

    df["spend_per_visit"] = df["total_spent"] / (df["website_visits"] + 1)

    df["spend_per_purchase"] = df["total_spent"] / (
        df["purchase_frequency"] + 1
    )

    df["engagement_to_recency_ratio"] = (
        df["engagement_score"] / (df["recency_days"] + 1)
    )

    df["discount_engagement_interaction"] = (
        df["discount_usage_rate"] * df["engagement_score"]
    )

    df["cart_abandonment_engagement_gap"] = (
        df["cart_abandonment_rate"] * (100 - df["engagement_score"])
    )

    df["customer_value_score"] = (
        df["total_spent"] * 0.4
        + df["avg_order_value"] * 0.2
        + df["purchase_frequency"] * 0.2
        + df["loyalty_score"] * 0.2
    )

    # ------------------------------------------------------------
    # 2. Recency-based churn risk features
    # ------------------------------------------------------------

    df["recency_risk_level"] = pd.cut(
        df["recency_days"],
        bins=[0, 92, 182, 274, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)

    df["is_recently_active"] = (df["recency_days"] <= 92).astype(int)
    df["is_inactive_90_plus_days"] = (df["recency_days"] > 92).astype(int)
    df["is_inactive_180_plus_days"] = (df["recency_days"] > 182).astype(int)
    df["is_inactive_270_plus_days"] = (df["recency_days"] > 274).astype(int)

    df["recency_severity_score"] = np.minimum(
        df["recency_days"] / 365,
        1,
    )

    # ------------------------------------------------------------
    # 3. Purchase frequency risk features
    # ------------------------------------------------------------

    df["purchase_frequency_level"] = pd.cut(
        df["purchase_frequency"],
        bins=[0, 2, 8, 12, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)

    df["is_low_frequency_customer"] = (
        df["purchase_frequency"] <= 2
    ).astype(int)

    df["is_high_frequency_customer"] = (
        df["purchase_frequency"] > 12
    ).astype(int)

    df["purchase_per_recency_day"] = (
        df["purchase_frequency"] / (df["recency_days"] + 1)
    )

    df["visit_to_purchase_ratio"] = (
        df["website_visits"] / (df["purchase_frequency"] + 1)
    )

    df["purchase_conversion_proxy"] = (
        df["purchase_frequency"] / (df["website_visits"] + 1)
    )

    # ------------------------------------------------------------
    # 4. Total spent and customer value features
    # ------------------------------------------------------------

    df["total_spent_level"] = pd.cut(
        df["total_spent"],
        bins=[0, 381, 1089, 2952, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)

    df["is_low_value_customer"] = (
        df["total_spent"] <= 381
    ).astype(int)

    df["is_high_value_customer"] = (
        df["total_spent"] > 2952
    ).astype(int)

    df["revenue_intensity"] = (
        df["total_spent"] / (df["recency_days"] + 1)
    )

    df["customer_value_percentile"] = df["total_spent"].rank(pct=True)

    df["balanced_customer_value_score"] = (
        df["total_spent"].rank(pct=True) * 0.40
        + df["avg_order_value"].rank(pct=True) * 0.20
        + df["purchase_frequency"].rank(pct=True) * 0.20
        + df["loyalty_score"].rank(pct=True) * 0.20
    )

    # ------------------------------------------------------------
    # 5. Cart abandonment risk features
    # ------------------------------------------------------------

    df["cart_abandonment_risk_level"] = pd.cut(
        df["cart_abandonment_rate"],
        bins=[0, 0.25, 0.45, 0.65, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)

    df["is_high_cart_abandoner"] = (
        df["cart_abandonment_rate"] > 0.65
    ).astype(int)

    df["is_medium_high_cart_abandoner"] = (
        df["cart_abandonment_rate"] > 0.45
    ).astype(int)

    df["cart_abandonment_recency_interaction"] = (
        df["cart_abandonment_rate"] * df["recency_severity_score"]
    )

    df["cart_abandonment_value_risk"] = (
        df["cart_abandonment_rate"] * df["balanced_customer_value_score"]
    )

    # ------------------------------------------------------------
    # 6. Engagement features
    # ------------------------------------------------------------

    df["engagement_level"] = pd.cut(
        df["engagement_score"],
        bins=[0, 39, 51, 63, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)

    df["is_low_engagement_customer"] = (
        df["engagement_score"] <= 39
    ).astype(int)

    df["is_high_engagement_customer"] = (
        df["engagement_score"] > 63
    ).astype(int)

    df["low_engagement_score"] = (
        100 - df["engagement_score"]
    )

    df["engagement_recency_risk"] = (
        df["low_engagement_score"] * df["recency_severity_score"]
    )

    df["engagement_cart_risk"] = (
        df["low_engagement_score"] * df["cart_abandonment_rate"]
    )

    # ------------------------------------------------------------
    # 7. Loyalty features
    # ------------------------------------------------------------

    df["low_loyalty_score"] = (
        100 - df["loyalty_score"]
    )

    df["loyalty_engagement_gap"] = (
        df["loyalty_score"] - df["engagement_score"]
    )

    df["loyalty_recency_risk"] = (
        df["low_loyalty_score"] * df["recency_severity_score"]
    )

    df["loyalty_cart_risk"] = (
        df["low_loyalty_score"] * df["cart_abandonment_rate"]
    )

    df["loyalty_engagement_score"] = (
        df["loyalty_score"] * 0.5
        + df["engagement_score"] * 0.5
    )

    # ------------------------------------------------------------
    # 8. Email and discount interaction features
    # ------------------------------------------------------------

    df["email_engagement_interaction"] = (
        df["email_open_rate"] * df["engagement_score"]
    )

    df["email_loyalty_interaction"] = (
        df["email_open_rate"] * df["loyalty_score"]
    )

    df["discount_loyalty_interaction"] = (
        df["discount_usage_rate"] * df["loyalty_score"]
    )

    df["discount_cart_interaction"] = (
        df["discount_usage_rate"] * df["cart_abandonment_rate"]
    )

    df["discount_dependency_score"] = (
        df["discount_usage_rate"] / (df["purchase_frequency"] + 1)
    )

    # ------------------------------------------------------------
    # 9. Behavioral churn risk score
    # ------------------------------------------------------------

    df["behavioral_churn_risk_score"] = (
        df["recency_severity_score"] * 0.35
        + df["cart_abandonment_rate"] * 0.25
        + (df["low_engagement_score"] / 100) * 0.15
        + (df["low_loyalty_score"] / 100) * 0.15
        + (1 - df["purchase_frequency"].rank(pct=True)) * 0.10
    )

    # ------------------------------------------------------------
    # 10. Revenue-at-risk priority features
    # ------------------------------------------------------------

    df["revenue_at_risk_score"] = (
        df["balanced_customer_value_score"]
        * df["behavioral_churn_risk_score"]
    )

    df["is_high_value_high_risk_customer"] = (
        (df["balanced_customer_value_score"] >= 0.75)
        & (
            df["behavioral_churn_risk_score"]
            >= df["behavioral_churn_risk_score"].quantile(0.75)
        )
    ).astype(int)

    df["retention_priority_score"] = (
        df["revenue_at_risk_score"].rank(pct=True)
    )

    # ------------------------------------------------------------
    # 11. Clean infinite values if any
    # ------------------------------------------------------------

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


# ============================================================
# Business Risk Segment
# ============================================================

def create_risk_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interpretable risk segment from behavioral_churn_risk_score.

    This does not use churn_flag or churn_risk, so it is safe from target leakage.
    """
    df = df.copy()

    if "behavioral_churn_risk_score" not in df.columns:
        raise ValueError("behavioral_churn_risk_score not found.")

    df["risk_segment"] = pd.qcut(
        df["behavioral_churn_risk_score"],
        q=4,
        labels=[
            "Low Risk",
            "Medium Risk",
            "High Risk",
            "Critical Risk",
        ],
        duplicates="drop",
    )

    df["risk_segment"] = df["risk_segment"].astype(str)

    return df


# ============================================================
# Revenue at Risk for Business Reporting
# ============================================================

def create_revenue_at_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create revenue_at_risk for business reporting only.

    Important:
    - This uses churn_flag.
    - This column must be excluded from model features.
    """
    df = df.copy()

    df["revenue_at_risk"] = np.where(
        df[TARGET_COLUMN] == 1,
        df["total_spent"],
        0,
    )

    return df


# ============================================================
# Save Processed Dataset
# ============================================================

def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)

    print(f"Processed data saved to: {output_path}")


# ============================================================
# Preprocessing
# ============================================================

def get_one_hot_encoder() -> OneHotEncoder:
    """
    Handle sklearn version differences.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select features for model training.

    Excludes:
    - customer_id
    - churn_flag
    - churn_risk
    - revenue_at_risk

    churn_risk is excluded because it may already contain churn-related information.
    revenue_at_risk is excluded because it uses churn_flag.
    """
    feature_columns = [
        col for col in df.columns
        if col not in LEAKAGE_COLUMNS
    ]

    return feature_columns


def build_train_test_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """
    Build encoded and scaled train-test datasets.
    """
    df = df.copy()

    feature_columns = get_model_feature_columns(df)

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", get_one_hot_encoder(), categorical_columns),
        ],
        remainder="drop",
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    processed_feature_names = preprocessor.get_feature_names_out().tolist()

    train_df = pd.DataFrame(
        X_train_processed,
        columns=processed_feature_names,
        index=X_train.index,
    )

    test_df = pd.DataFrame(
        X_test_processed,
        columns=processed_feature_names,
        index=X_test.index,
    )

    train_df[TARGET_COLUMN] = y_train.values
    test_df[TARGET_COLUMN] = y_test.values

    Path(PREPROCESSOR_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")

    return train_df, test_df, feature_columns, processed_feature_names


def save_train_test_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    Path(TRAIN_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print(f"Train data saved to: {TRAIN_DATA_PATH}")
    print(f"Test data saved to: {TEST_DATA_PATH}")


def save_feature_list(
    raw_feature_columns: list[str],
    processed_feature_names: list[str],
) -> None:
    output_file = Path(FEATURE_LIST_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    feature_metadata = {
        "target_column": TARGET_COLUMN,
        "raw_feature_count": len(raw_feature_columns),
        "processed_feature_count": len(processed_feature_names),
        "raw_features": raw_feature_columns,
        "processed_features": processed_feature_names,
        "excluded_columns": LEAKAGE_COLUMNS,
        "notes": {
            "churn_risk": "Excluded to avoid potential target leakage.",
            "revenue_at_risk": "Excluded because it is derived using churn_flag.",
            "risk_segment": "Included because it is derived from behavioral features only.",
        },
    }

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(feature_metadata, file, indent=4)

    print(f"Feature list saved to: {FEATURE_LIST_PATH}")


# ============================================================
# Main Pipeline
# ============================================================

def main() -> None:
    print("Starting feature engineering pipeline...")

    df = load_data(INPUT_DATA_PATH)

    print(f"Raw input shape: {df.shape}")

    df = normalize_target(df)
    df = handle_missing_values(df)
    df = normalize_rate_columns(df)
    df = normalize_score_columns(df)

    df = create_business_features(df)
    df = create_risk_segment(df)
    df = create_revenue_at_risk(df)
    df = clean_infinite_and_missing_values(df)

    print(f"Processed full dataset shape: {df.shape}")

    save_processed_data(df, PROCESSED_DATA_PATH)

    train_df, test_df, raw_feature_columns, processed_feature_names = (
        build_train_test_data(df)
    )

    save_train_test_data(train_df, test_df)
    save_feature_list(raw_feature_columns, processed_feature_names)

    print("Feature engineering pipeline completed successfully.")


if __name__ == "__main__":
    main()