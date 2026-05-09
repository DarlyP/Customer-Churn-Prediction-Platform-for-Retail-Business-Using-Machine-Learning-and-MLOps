from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# Path Configuration
# ============================================================

TRAIN_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH = "data/processed/test.csv"
FEATURE_LIST_PATH = "data/processed/feature_list.json"

TARGET_COLUMN = "churn_flag"

MODEL_OUTPUT_PATH = "models/champion_model.pkl"
METRICS_OUTPUT_PATH = "reports/model_metrics/mlflow_model_comparison.csv"
BEST_METRICS_OUTPUT_PATH = "reports/model_metrics/champion_model_metrics.json"

CONFUSION_MATRIX_DIR = "reports/figures/confusion_matrices"
CLASSIFICATION_REPORT_DIR = "reports/classification_reports"

MLFLOW_EXPERIMENT_NAME = "retail_churn_prediction"

RANDOM_STATE = 42


# ============================================================
# Data Loading
# ============================================================

def load_train_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def load_feature_list() -> dict:
    feature_file = Path(FEATURE_LIST_PATH)

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature list not found: {FEATURE_LIST_PATH}")

    with open(feature_file, "r", encoding="utf-8") as file:
        feature_metadata = json.load(file)

    return feature_metadata


# ============================================================
# Model Configuration
# ============================================================

def get_baseline_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    return models


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, X_test, y_test) -> tuple[dict, str, pd.DataFrame]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_churn": precision_score(y_test, y_pred, zero_division=0),
        "recall_churn": recall_score(y_test, y_pred, zero_division=0),
        "f1_churn": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    report_text = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()

    return metrics, report_text, report_df


def save_confusion_matrix(
    model_name: str,
    model,
    X_test,
    y_test,
) -> str:
    output_dir = Path(CONFUSION_MATRIX_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model_name = model_name.lower().replace(" ", "_")
    output_path = output_dir / f"{safe_model_name}_confusion_matrix.png"

    y_pred = model.predict(X_test)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        values_format="d",
    )

    disp.ax_.set_title(f"Confusion Matrix - {model_name}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return str(output_path)


def save_classification_report(
    model_name: str,
    report_text: str,
    report_df: pd.DataFrame,
) -> tuple[str, str]:
    output_dir = Path(CLASSIFICATION_REPORT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model_name = model_name.lower().replace(" ", "_")

    txt_path = output_dir / f"{safe_model_name}_classification_report.txt"
    csv_path = output_dir / f"{safe_model_name}_classification_report.csv"

    txt_path.write_text(report_text, encoding="utf-8")
    report_df.to_csv(csv_path)

    return str(txt_path), str(csv_path)


# ============================================================
# MLflow Logging
# ============================================================

def log_model_to_mlflow(
    model_name: str,
    model,
    metrics: dict,
    confusion_matrix_path: str,
    report_txt_path: str,
    report_csv_path: str,
    feature_metadata: dict,
) -> None:
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("project", "retail_customer_churn_prediction")
        mlflow.set_tag("target", TARGET_COLUMN)
        mlflow.set_tag("business_priority", "maximize churn recall and PR-AUC")

        model_params = model.get_params()

        for param_name, param_value in model_params.items():
            mlflow.log_param(param_name, param_value)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.log_artifact(confusion_matrix_path)
        mlflow.log_artifact(report_txt_path)
        mlflow.log_artifact(report_csv_path)
        mlflow.log_artifact(FEATURE_LIST_PATH)

        feature_count = feature_metadata.get("processed_feature_count")
        raw_feature_count = feature_metadata.get("raw_feature_count")

        if feature_count is not None:
            mlflow.log_param("processed_feature_count", feature_count)

        if raw_feature_count is not None:
            mlflow.log_param("raw_feature_count", raw_feature_count)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )


# ============================================================
# Save Outputs
# ============================================================

def select_champion_model(results_df: pd.DataFrame) -> str:
    """
    Select champion model using business-first metrics.

    Priority:
    1. recall_churn
    2. pr_auc
    3. f1_churn
    4. roc_auc
    """
    sorted_results = results_df.sort_values(
        by=["recall_churn", "pr_auc", "f1_churn", "roc_auc"],
        ascending=False,
    )

    return sorted_results.iloc[0]["model"]


def save_champion_model(model, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_file)

    print(f"Champion model saved to: {output_path}")


def save_metrics(
    results_df: pd.DataFrame,
    champion_model_name: str,
) -> None:
    metrics_output_file = Path(METRICS_OUTPUT_PATH)
    metrics_output_file.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(metrics_output_file, index=False)

    champion_metrics = (
        results_df[results_df["model"] == champion_model_name]
        .iloc[0]
        .to_dict()
    )

    with open(BEST_METRICS_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(champion_metrics, file, indent=4)

    print(f"Model comparison metrics saved to: {METRICS_OUTPUT_PATH}")
    print(f"Champion model metrics saved to: {BEST_METRICS_OUTPUT_PATH}")


# ============================================================
# Main Pipeline
# ============================================================

def main() -> None:
    print("Starting MLflow experiment tracking...")

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_train_test_data()
    feature_metadata = load_feature_list()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    models = get_baseline_models()

    results = []
    trained_models = {}

    for model_name, model in models.items():
        print(f"\nTraining and logging model: {model_name}")

        model.fit(X_train, y_train)

        metrics, report_text, report_df = evaluate_model(
            model,
            X_test,
            y_test,
        )

        confusion_matrix_path = save_confusion_matrix(
            model_name,
            model,
            X_test,
            y_test,
        )

        report_txt_path, report_csv_path = save_classification_report(
            model_name,
            report_text,
            report_df,
        )

        log_model_to_mlflow(
            model_name=model_name,
            model=model,
            metrics=metrics,
            confusion_matrix_path=confusion_matrix_path,
            report_txt_path=report_txt_path,
            report_csv_path=report_csv_path,
            feature_metadata=feature_metadata,
        )

        metrics["model"] = model_name
        results.append(metrics)
        trained_models[model_name] = model

    results_df = pd.DataFrame(results)

    print("\nMLflow experiment comparison:")
    print(
        results_df.sort_values(
            by=["recall_churn", "pr_auc", "f1_churn", "roc_auc"],
            ascending=False,
        )
    )

    champion_model_name = select_champion_model(results_df)
    champion_model = trained_models[champion_model_name]

    print(f"\nChampion model selected: {champion_model_name}")

    save_champion_model(champion_model, MODEL_OUTPUT_PATH)
    save_metrics(results_df, champion_model_name)

    print("MLflow experiment tracking completed successfully.")


if __name__ == "__main__":
    main()