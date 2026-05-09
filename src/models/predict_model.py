from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Path Configuration
# ============================================================

MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

PREDICTION_OUTPUT_PATH = "data/predictions/sample_prediction.csv"


# ============================================================
# Input Column Configuration
# ============================================================

REQUIRED_INPUT_COLUMNS = [
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
]

NUMERICAL_COLUMNS = [
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
]

CATEGORICAL_COLUMNS = [
    "age_group",
    "gender",
    "region",
    "customer_segment",
    "preferred_channel",
]


# ============================================================
# Load Model and Preprocessor
# ============================================================

def load_model():
    model_file = Path(MODEL_PATH)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    return joblib.load(model_file)


def load_preprocessor():
    preprocessor_file = Path(PREPROCESSOR_PATH)

    if not preprocessor_file.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {PREPROCESSOR_PATH}")

    return joblib.load(preprocessor_file)


# ============================================================
# Input Validation
# ============================================================

def validate_input_columns(df: pd.DataFrame) -> None:
    missing_columns = [
        col for col in REQUIRED_INPUT_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required input columns: {missing_columns}")


def normalize_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize raw customer input.
    """
    df = df.copy()

    validate_input_columns(df)

    for col in NUMERICAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str).fillna("Unknown")

    return df


def normalize_rate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure rate columns are in 0-1 scale.
    If values appear in 0-100 scale, convert to 0-1.
    """
    df = df.copy()

    rate_columns = [
        "discount_usage_rate",
        "email_open_rate",
        "cart_abandonment_rate",
    ]

    for col in rate_columns:
        if df[col].max() > 1:
            df[col] = df[col] / 100

        df[col] = df[col].clip(lower=0, upper=1)

    return df


def normalize_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure score columns are in 0-100 scale.
    """
    df = df.copy()

    score_columns = [
        "loyalty_score",
        "engagement_score",
    ]

    for col in score_columns:
        df[col] = df[col].clip(lower=0, upper=100)

    return df


# ============================================================
# Feature Engineering for Prediction
# ============================================================

def create_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the same business features used during training.

    Important:
    This function does not use churn_flag or churn_risk.
    """
    df = df.copy()

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

    df["behavioral_churn_risk_score"] = (
        df["recency_severity_score"] * 0.35
        + df["cart_abandonment_rate"] * 0.25
        + (df["low_engagement_score"] / 100) * 0.15
        + (df["low_loyalty_score"] / 100) * 0.15
        + (1 - df["purchase_frequency"].rank(pct=True)) * 0.10
    )

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

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def create_risk_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create risk segment from behavioral_churn_risk_score.

    For prediction, do not use pd.qcut because prediction input can contain
    only one customer. qcut needs enough rows to create quantile-based bins.

    This function uses fixed thresholds so it works for both single-customer
    and batch prediction.
    """
    df = df.copy()

    if "behavioral_churn_risk_score" not in df.columns:
        raise ValueError("behavioral_churn_risk_score not found.")

    df["risk_segment"] = pd.cut(
        df["behavioral_churn_risk_score"],
        bins=[-np.inf, 0.30, 0.60, 0.80, np.inf],
        labels=[
            "Low Risk",
            "Medium Risk",
            "High Risk",
            "Critical Risk",
        ],
        include_lowest=True,
    )

    df["risk_segment"] = df["risk_segment"].astype(str)

    return df


def clean_after_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if col != "customer_id":
            df[col] = df[col].astype(str).fillna("Unknown")

    return df


def prepare_features_for_prediction(
    df: pd.DataFrame,
    preprocessor,
) -> pd.DataFrame:
    """
    Apply feature engineering and preprocessor to raw input.
    """
    df = normalize_input_data(df)
    df = normalize_rate_columns(df)
    df = normalize_score_columns(df)
    df = create_business_features(df)
    df = create_risk_segment(df)
    df = clean_after_feature_engineering(df)

    raw_feature_names = preprocessor.feature_names_in_.tolist()

    missing_features = [
        col for col in raw_feature_names
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features required by preprocessor: {missing_features}"
        )

    X_raw = df[raw_feature_names]

    X_processed = preprocessor.transform(X_raw)

    processed_feature_names = preprocessor.get_feature_names_out().tolist()

    X_processed_df = pd.DataFrame(
        X_processed,
        columns=processed_feature_names,
        index=df.index,
    )

    return X_processed_df


# ============================================================
# Prediction Logic
# ============================================================

def get_risk_level(churn_probability: float) -> str:
    if churn_probability <= 0.30:
        return "Low Risk"
    elif churn_probability <= 0.60:
        return "Medium Risk"
    elif churn_probability <= 0.80:
        return "High Risk"
    else:
        return "Critical Risk"


def get_recommended_action(risk_level: str) -> str:
    action_mapping = {
        "Low Risk": "Maintain engagement",
        "Medium Risk": "Send personalized offer",
        "High Risk": "Loyalty discount campaign",
        "Critical Risk": "Immediate retention call / premium offer",
    }

    return action_mapping[risk_level]


def predict_customer_churn(input_data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    """
    Predict churn for one or multiple customers.

    Accepted input:
    - pandas DataFrame
    - single dictionary
    - list of dictionaries
    """
    if isinstance(input_data, dict):
        raw_df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        raw_df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        raw_df = input_data.copy()
    else:
        raise TypeError(
            "input_data must be a pandas DataFrame, dictionary, or list of dictionaries."
        )

    customer_ids = raw_df["customer_id"].astype(str).values

    model = load_model()
    preprocessor = load_preprocessor()

    X_processed = prepare_features_for_prediction(
        df=raw_df,
        preprocessor=preprocessor,
    )

    churn_probabilities = model.predict_proba(X_processed)[:, 1]
    prediction_labels = (churn_probabilities >= 0.50).astype(int)

    results = []

    for customer_id, probability, label in zip(
        customer_ids,
        churn_probabilities,
        prediction_labels,
    ):
        risk_level = get_risk_level(float(probability))
        recommended_action = get_recommended_action(risk_level)

        results.append(
            {
                "customer_id": customer_id,
                "churn_probability": round(float(probability), 6),
                "prediction_label": int(label),
                "prediction_text": "Churn" if label == 1 else "Not Churn",
                "risk_level": risk_level,
                "recommended_action": recommended_action,
            }
        )

    prediction_df = pd.DataFrame(results)

    return prediction_df


# ============================================================
# Save Prediction
# ============================================================

def save_predictions(prediction_df: pd.DataFrame, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    prediction_df.to_csv(output_file, index=False)

    print(f"Predictions saved to: {output_path}")


# ============================================================
# Example Usage
# ============================================================

def main() -> None:
    sample_customer = {
        "customer_id": "CUST_001",
        "age_group": "35-44",
        "gender": "Female",
        "region": "West",
        "customer_segment": "Returning",
        "preferred_channel": "Mobile App",
        "purchase_frequency": 2,
        "avg_order_value": 45.5,
        "total_spent": 250.0,
        "recency_days": 210,
        "website_visits": 15,
        "discount_usage_rate": 0.65,
        "email_open_rate": 0.30,
        "cart_abandonment_rate": 0.72,
        "loyalty_score": 35,
        "engagement_score": 40,
    }

    prediction_df = predict_customer_churn(sample_customer)

    print("\nPrediction Result:")
    print(prediction_df)

    save_predictions(
        prediction_df=prediction_df,
        output_path=PREDICTION_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()