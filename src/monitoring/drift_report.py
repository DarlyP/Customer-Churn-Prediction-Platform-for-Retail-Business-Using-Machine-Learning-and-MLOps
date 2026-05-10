from pathlib import Path
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Path Configuration
# ============================================================

TRAIN_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH = "data/processed/test.csv"
MODEL_PATH = "models/best_model.pkl"

OUTPUT_DIR = "reports/monitoring"
DRIFT_REPORT_HTML_PATH = f"{OUTPUT_DIR}/data_drift_report.html"
MONITORING_SUMMARY_JSON_PATH = f"{OUTPUT_DIR}/monitoring_summary.json"
SIMULATED_CURRENT_DATA_PATH = f"{OUTPUT_DIR}/simulated_current_month_data.csv"

TARGET_COLUMN = "churn_flag"
PREDICTION_PROBA_COLUMN = "prediction_probability"
PREDICTION_LABEL_COLUMN = "prediction_label"

RANDOM_STATE = 42


# ============================================================
# Load Data and Model
# ============================================================

def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_file = Path(TRAIN_DATA_PATH)
    test_file = Path(TEST_DATA_PATH)

    if not train_file.exists():
        raise FileNotFoundError(f"Train data not found: {TRAIN_DATA_PATH}")

    if not test_file.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_DATA_PATH}")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    return train_df, test_df


def load_model():
    model_file = Path(MODEL_PATH)

    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return joblib.load(model_file)


# ============================================================
# Simulate Current Monthly Data
# ============================================================

def simulate_current_month_data(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate new monthly production data by applying controlled changes
    to selected features.

    This is for portfolio/demo purposes only.
    In real production, current_data should come from actual model scoring logs.
    """
    current_df = test_df.copy()

    drift_adjustments = {
        "num__recency_days": 0.35,
        "num__cart_abandonment_rate": 0.25,
        "num__loyalty_score": -0.25,
        "num__engagement_score": -0.20,
        "num__behavioral_churn_risk_score": 0.25,
    }

    for column, shift_value in drift_adjustments.items():
        if column in current_df.columns:
            current_df[column] = current_df[column] + shift_value

    return current_df


def add_predictions(
    df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """
    Add prediction columns to reference/current dataset.
    """
    df = df.copy()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    X = df.drop(columns=[TARGET_COLUMN])

    prediction_proba = model.predict_proba(X)[:, 1]
    prediction_label = (prediction_proba >= 0.50).astype(int)

    df[PREDICTION_PROBA_COLUMN] = prediction_proba
    df[PREDICTION_LABEL_COLUMN] = prediction_label

    return df


def simulate_missing_value_change(
    current_df: pd.DataFrame,
    missing_fraction: float = 0.02,
) -> pd.DataFrame:
    """
    Add a small amount of missing values to simulate data quality change.

    Missing values are added after prediction columns are created.
    """
    current_df = current_df.copy()
    rng = np.random.default_rng(RANDOM_STATE)

    candidate_columns = [
        "num__recency_days",
        "num__cart_abandonment_rate",
        "num__loyalty_score",
        "num__engagement_score",
    ]

    existing_columns = [
        col for col in candidate_columns
        if col in current_df.columns
    ]

    if not existing_columns:
        return current_df

    n_rows = len(current_df)
    n_missing = max(1, int(n_rows * missing_fraction))

    for col in existing_columns:
        missing_indices = rng.choice(
            current_df.index,
            size=n_missing,
            replace=False,
        )
        current_df.loc[missing_indices, col] = np.nan

    return current_df


# ============================================================
# Manual Monitoring Summary
# ============================================================

def calculate_missing_value_summary(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict[str, Any]:
    reference_missing = reference_df.isna().mean()
    current_missing = current_df.isna().mean()

    summary_df = pd.DataFrame(
        {
            "reference_missing_rate": reference_missing,
            "current_missing_rate": current_missing,
        }
    )

    summary_df["missing_rate_change"] = (
        summary_df["current_missing_rate"]
        - summary_df["reference_missing_rate"]
    )

    summary_df = summary_df.sort_values(
        by="missing_rate_change",
        ascending=False,
    )

    return {
        "top_missing_value_changes": summary_df.head(20).to_dict(orient="index"),
        "total_reference_missing_cells": int(reference_df.isna().sum().sum()),
        "total_current_missing_cells": int(current_df.isna().sum().sum()),
    }


def calculate_prediction_summary(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict[str, Any]:
    reference_prediction_mean = float(reference_df[PREDICTION_PROBA_COLUMN].mean())
    current_prediction_mean = float(current_df[PREDICTION_PROBA_COLUMN].mean())

    reference_predicted_churn_rate = float(reference_df[PREDICTION_LABEL_COLUMN].mean())
    current_predicted_churn_rate = float(current_df[PREDICTION_LABEL_COLUMN].mean())

    return {
        "reference_average_prediction_probability": reference_prediction_mean,
        "current_average_prediction_probability": current_prediction_mean,
        "prediction_probability_change": (
            current_prediction_mean - reference_prediction_mean
        ),
        "reference_predicted_churn_rate": reference_predicted_churn_rate,
        "current_predicted_churn_rate": current_predicted_churn_rate,
        "predicted_churn_rate_change": (
            current_predicted_churn_rate - reference_predicted_churn_rate
        ),
    }


def calculate_target_summary(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict[str, Any]:
    reference_target_rate = float(reference_df[TARGET_COLUMN].mean())
    current_target_rate = float(current_df[TARGET_COLUMN].mean())

    return {
        "reference_actual_churn_rate": reference_target_rate,
        "current_actual_churn_rate": current_target_rate,
        "target_rate_change": current_target_rate - reference_target_rate,
    }


def save_monitoring_summary(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str,
) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "data_shape": {
            "reference_rows": len(reference_df),
            "reference_columns": len(reference_df.columns),
            "current_rows": len(current_df),
            "current_columns": len(current_df.columns),
        },
        "target_drift_summary": calculate_target_summary(reference_df, current_df),
        "prediction_drift_summary": calculate_prediction_summary(reference_df, current_df),
        "missing_value_change_summary": calculate_missing_value_summary(
            reference_df,
            current_df,
        ),
        "notes": {
            "reference_data": "Training data",
            "current_data": "Simulated new monthly customer data based on test data",
            "purpose": "Portfolio simulation for production monitoring workflow",
        },
    }

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Monitoring summary saved to: {output_path}")


# ============================================================
# Evidently Report Generation
# ============================================================

def generate_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Generate Evidently data drift report.

    This function supports both newer and older Evidently APIs.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Newer Evidently API
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(
            [
                DataDriftPreset(),
            ],
            include_tests=True,
        )

        result = report.run(
            current_df,
            reference_df,
        )

        save_object = result if hasattr(result, "save_html") else report
        save_object.save_html(str(output_file))

    except Exception as new_api_error:
        print(f"New Evidently API failed. Trying legacy API. Error: {new_api_error}")

        try:
            # Legacy Evidently API, generally used by <=0.6.x
            from evidently import ColumnMapping
            from evidently.report import Report
            from evidently.metric_preset import (
                DataDriftPreset,
                DataQualityPreset,
                TargetDriftPreset,
            )

            column_mapping = ColumnMapping(
                target=TARGET_COLUMN,
                prediction=PREDICTION_PROBA_COLUMN,
            )

            report = Report(
                metrics=[
                    DataDriftPreset(),
                    TargetDriftPreset(),
                    DataQualityPreset(),
                ]
            )

            report.run(
                reference_data=reference_df,
                current_data=current_df,
                column_mapping=column_mapping,
            )

            report.save_html(str(output_file))

        except Exception as legacy_api_error:
            raise RuntimeError(
                "Failed to generate Evidently report with both new and legacy APIs. "
                f"New API error: {new_api_error}. "
                f"Legacy API error: {legacy_api_error}."
            )

    print(f"Evidently drift report saved to: {output_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Starting monitoring report generation...")

    model = load_model()
    train_df, test_df = load_processed_data()

    print(f"Reference data shape: {train_df.shape}")
    print(f"Base current data shape: {test_df.shape}")

    reference_df = add_predictions(train_df, model)

    current_df = simulate_current_month_data(test_df)
    current_df = add_predictions(current_df, model)
    current_df = simulate_missing_value_change(current_df)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_df.to_csv(SIMULATED_CURRENT_DATA_PATH, index=False)
    print(f"Simulated current data saved to: {SIMULATED_CURRENT_DATA_PATH}")

    save_monitoring_summary(
        reference_df=reference_df,
        current_df=current_df,
        output_path=MONITORING_SUMMARY_JSON_PATH,
    )

    generate_evidently_report(
        reference_df=reference_df,
        current_df=current_df,
        output_path=DRIFT_REPORT_HTML_PATH,
    )

    print("Monitoring report generation completed successfully.")


if __name__ == "__main__":
    main()