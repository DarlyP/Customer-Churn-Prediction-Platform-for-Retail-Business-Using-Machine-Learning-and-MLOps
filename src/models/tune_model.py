from pathlib import Path
import json
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# ============================================================
# Warning Control
# ============================================================

warnings.filterwarnings("ignore")


# ============================================================
# Path Configuration
# ============================================================

TRAIN_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH = "data/processed/test.csv"
FEATURE_LIST_PATH = "data/processed/feature_list.json"

TARGET_COLUMN = "churn_flag"

MODEL_DIR = "models"
REPORT_DIR = "reports/model_metrics"
FIGURE_DIR = "reports/figures/tuning"
CLASSIFICATION_REPORT_DIR = "reports/classification_reports/tuning"

TUNING_COMPARISON_OUTPUT_PATH = f"{REPORT_DIR}/tuning_comparison.csv"
BEST_PARAMS_OUTPUT_PATH = f"{REPORT_DIR}/best_params.json"
BEST_METRICS_OUTPUT_PATH = f"{REPORT_DIR}/best_tuned_model_metrics.json"

BEST_MODEL_OUTPUT_PATH = f"{MODEL_DIR}/best_model.pkl"
TUNED_MODEL_OUTPUT_PATH = f"{MODEL_DIR}/tuned_model.pkl"

MLFLOW_EXPERIMENT_NAME = "retail_churn_prediction"

RANDOM_STATE = 42
N_ITER = 25
CV_SPLITS = 5
SCORING = "average_precision"


# ============================================================
# Data Loading
# ============================================================

def load_train_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Target column not found in train data: {TARGET_COLUMN}")

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"Target column not found in test data: {TARGET_COLUMN}")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(int)

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(int)

    return X_train, X_test, y_train, y_test


def load_feature_metadata() -> dict:
    feature_file = Path(FEATURE_LIST_PATH)

    if not feature_file.exists():
        return {}

    with open(feature_file, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """
    Calculate class imbalance ratio.

    Formula:
    scale_pos_weight = negative_count / positive_count
    """
    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())

    if positive_count == 0:
        raise ValueError("No positive class found in target column.")

    return negative_count / positive_count


# ============================================================
# Model Configuration
# ============================================================

def get_lightgbm_model(scale_pos_weight: float) -> LGBMClassifier:
    """
    LightGBM model with safer defaults.

    Notes:
    - verbosity=-1 reduces repeated LightGBM split warnings.
    - scale_pos_weight handles binary class imbalance.
    - class_weight is intentionally not used.
    """
    model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        scale_pos_weight=scale_pos_weight,
    )

    return model


def get_lightgbm_param_distributions(scale_pos_weight: float) -> dict:
    """
    Safer LightGBM search space.

    The previous warning can happen more often when the search space
    creates trees that are too constrained or too aggressive.
    """
    param_distributions = {
        "n_estimators": [100, 200, 300, 500, 700],
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],

        "max_depth": [3, 4, 5, 6, 8, -1],
        "num_leaves": [7, 15, 31, 63],

        "min_child_samples": [5, 10, 20, 30, 50],
        "min_child_weight": [1e-3, 1e-2, 0.1],
        "min_split_gain": [0.0, 0.001, 0.01],

        "subsample": [0.7, 0.8, 0.9, 1.0],
        "subsample_freq": [1],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],

        "reg_alpha": [0.0, 0.01, 0.1, 0.5],
        "reg_lambda": [0.0, 0.01, 0.1, 0.5, 1.0],

        "scale_pos_weight": [
            1.0,
            max(scale_pos_weight * 0.5, 1.0),
            scale_pos_weight,
            scale_pos_weight * 1.5,
        ],
    }

    return param_distributions


def get_xgboost_model(scale_pos_weight: float) -> XGBClassifier:
    """
    XGBoost model with binary classification setup.
    """
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        scale_pos_weight=scale_pos_weight,
    )

    return model


def get_xgboost_param_distributions(scale_pos_weight: float) -> dict:
    """
    XGBoost search space.
    """
    param_distributions = {
        "n_estimators": [100, 200, 300, 500, 700],
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],

        "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 3, 5, 7, 10],
        "gamma": [0.0, 0.01, 0.1, 0.3, 0.5],

        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],

        "reg_alpha": [0.0, 0.01, 0.1, 0.5],
        "reg_lambda": [0.5, 1.0, 1.5, 2.0],

        "scale_pos_weight": [
            1.0,
            max(scale_pos_weight * 0.5, 1.0),
            scale_pos_weight,
            scale_pos_weight * 1.5,
        ],
    }

    return param_distributions


def get_tuning_candidates(scale_pos_weight: float) -> dict:
    candidates = {
        "LightGBM": {
            "model": get_lightgbm_model(scale_pos_weight),
            "param_distributions": get_lightgbm_param_distributions(scale_pos_weight),
        },
        "XGBoost": {
            "model": get_xgboost_model(scale_pos_weight),
            "param_distributions": get_xgboost_param_distributions(scale_pos_weight),
        },
    }

    return candidates


# ============================================================
# Tuning
# ============================================================

def run_randomized_search(
    model_name: str,
    model,
    param_distributions: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomizedSearchCV:
    """
    Run RandomizedSearchCV for one model.
    """
    print(f"\nStarting tuning for: {model_name}")

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring=SCORING,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
        refit=True,
        error_score=np.nan,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    print(f"Best CV {SCORING} for {model_name}: {search.best_score_:.4f}")
    print(f"Best params for {model_name}:")
    print(search.best_params_)

    return search


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict, str]:
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

    return metrics, report_text


def save_confusion_matrix(
    model_name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> str:
    output_dir = Path(FIGURE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model_name = model_name.lower().replace(" ", "_")
    output_path = output_dir / f"{safe_model_name}_confusion_matrix.png"

    y_pred = model.predict(X_test)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        values_format="d",
    )

    disp.ax_.set_title(f"Confusion Matrix - Tuned {model_name}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")

    return str(output_path)


def save_classification_report(
    model_name: str,
    report_text: str,
) -> str:
    output_dir = Path(CLASSIFICATION_REPORT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model_name = model_name.lower().replace(" ", "_")
    output_path = output_dir / f"{safe_model_name}_classification_report.txt"

    output_path.write_text(report_text, encoding="utf-8")

    print(f"Classification report saved to: {output_path}")

    return str(output_path)


# ============================================================
# Save Outputs
# ============================================================

def save_json(data: dict, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print(f"JSON saved to: {output_path}")


def save_model(model, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_file)

    print(f"Model saved to: {output_path}")


def save_tuning_results(
    results_df: pd.DataFrame,
    best_model_name: str,
    best_params: dict,
    best_metrics: dict,
) -> None:
    output_file = Path(TUNING_COMPARISON_OUTPUT_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_file, index=False)

    best_params_payload = {
        "best_model": best_model_name,
        "best_params": best_params,
    }

    best_metrics_payload = {
        "best_model": best_model_name,
        "metrics": best_metrics,
    }

    save_json(best_params_payload, BEST_PARAMS_OUTPUT_PATH)
    save_json(best_metrics_payload, BEST_METRICS_OUTPUT_PATH)

    print(f"Tuning comparison saved to: {TUNING_COMPARISON_OUTPUT_PATH}")


# ============================================================
# Model Selection
# ============================================================

def select_best_tuned_model(results_df: pd.DataFrame) -> str:
    """
    Select best tuned model using business-first churn metrics.

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


# ============================================================
# MLflow Logging
# ============================================================

def filter_numeric_metrics(metrics: dict) -> dict:
    """
    Keep only numeric metrics for MLflow logging.

    MLflow metrics must be numeric.
    Non-numeric values such as model names should be logged as tags or params,
    not as metrics.
    """
    numeric_metrics = {}

    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            if np.isfinite(value):
                numeric_metrics[key] = float(value)

    return numeric_metrics

def log_tuned_model_to_mlflow(
    model_name: str,
    model,
    search: RandomizedSearchCV,
    metrics: dict,
    confusion_matrix_path: str,
    classification_report_path: str,
    feature_metadata: dict,
    is_best_model: bool,
) -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"Tuned {model_name}"):
        mlflow.set_tag("stage", "hyperparameter_tuning")
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("selection_metric", SCORING)
        mlflow.set_tag("business_priority", "maximize churn recall, PR-AUC, and F1-score")
        mlflow.set_tag("is_best_tuned_model", str(is_best_model))

        mlflow.log_param("search_method", "RandomizedSearchCV")
        mlflow.log_param("n_iter", N_ITER)
        mlflow.log_param("cv_splits", CV_SPLITS)
        mlflow.log_param("scoring", SCORING)
        mlflow.log_metric("best_cv_score", float(search.best_score_))

        for param_name, param_value in search.best_params_.items():
            mlflow.log_param(f"best_{param_name}", param_value)

        numeric_metrics = filter_numeric_metrics(metrics)

        for metric_name, metric_value in numeric_metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        if Path(confusion_matrix_path).exists():
            mlflow.log_artifact(confusion_matrix_path)

        if Path(classification_report_path).exists():
            mlflow.log_artifact(classification_report_path)

        if Path(FEATURE_LIST_PATH).exists():
            mlflow.log_artifact(FEATURE_LIST_PATH)

        processed_feature_count = feature_metadata.get("processed_feature_count")
        raw_feature_count = feature_metadata.get("raw_feature_count")

        if processed_feature_count is not None:
            mlflow.log_param("processed_feature_count", processed_feature_count)

        if raw_feature_count is not None:
            mlflow.log_param("raw_feature_count", raw_feature_count)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )


# ============================================================
# Main Pipeline
# ============================================================

def main() -> None:
    print("Starting hyperparameter tuning for LightGBM and XGBoost...")

    X_train, X_test, y_train, y_test = load_train_test_data()
    feature_metadata = load_feature_metadata()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")

    tuning_candidates = get_tuning_candidates(scale_pos_weight)

    tuning_results = []
    tuned_models = {}
    search_objects = {}
    saved_artifacts = {}

    for model_name, candidate in tuning_candidates.items():
        search = run_randomized_search(
            model_name=model_name,
            model=candidate["model"],
            param_distributions=candidate["param_distributions"],
            X_train=X_train,
            y_train=y_train,
        )

        best_model = search.best_estimator_

        metrics, report_text = evaluate_model(
            best_model,
            X_test,
            y_test,
        )

        print(f"\nTest metrics for tuned {model_name}:")
        print(metrics)

        confusion_matrix_path = save_confusion_matrix(
            model_name=model_name,
            model=best_model,
            X_test=X_test,
            y_test=y_test,
        )

        classification_report_path = save_classification_report(
            model_name=model_name,
            report_text=report_text,
        )

        model_output_path = f"{MODEL_DIR}/tuned_{model_name.lower()}_model.pkl"
        save_model(best_model, model_output_path)

        result_row = {
            "model": model_name,
            "best_cv_score": float(search.best_score_),
            **metrics,
        }

        tuning_results.append(result_row)
        tuned_models[model_name] = best_model
        search_objects[model_name] = search
        saved_artifacts[model_name] = {
            "confusion_matrix_path": confusion_matrix_path,
            "classification_report_path": classification_report_path,
            "model_output_path": model_output_path,
        }

    results_df = pd.DataFrame(tuning_results)

    print("\nTuning comparison:")
    print(
        results_df.sort_values(
            by=["recall_churn", "pr_auc", "f1_churn", "roc_auc"],
            ascending=False,
        )
    )

    best_model_name = select_best_tuned_model(results_df)
    best_model = tuned_models[best_model_name]
    best_search = search_objects[best_model_name]

    best_metrics = (
        results_df[results_df["model"] == best_model_name]
        .iloc[0]
        .to_dict()
    )

    print(f"\nBest tuned model selected: {best_model_name}")

    save_model(best_model, BEST_MODEL_OUTPUT_PATH)
    save_model(best_model, TUNED_MODEL_OUTPUT_PATH)

    save_tuning_results(
        results_df=results_df,
        best_model_name=best_model_name,
        best_params=best_search.best_params_,
        best_metrics=best_metrics,
    )

    for model_name in tuning_candidates.keys():
        print(f"Logging tuned model to MLflow: {model_name}")

        model_metrics = (
            results_df[results_df["model"] == model_name]
            .iloc[0]
            .to_dict()
        )

        log_tuned_model_to_mlflow(
            model_name=model_name,
            model=tuned_models[model_name],
            search=search_objects[model_name],
            metrics=model_metrics,
            confusion_matrix_path=saved_artifacts[model_name]["confusion_matrix_path"],
            classification_report_path=saved_artifacts[model_name]["classification_report_path"],
            feature_metadata=feature_metadata,
            is_best_model=(model_name == best_model_name),
        )

    print("Hyperparameter tuning completed successfully.")


if __name__ == "__main__":
    main()