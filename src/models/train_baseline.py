from pathlib import Path
import json

import joblib
import pandas as pd

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
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


TRAIN_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH = "data/processed/test.csv"

TARGET_COLUMN = "churn_flag"

MODEL_OUTPUT_PATH = "models/baseline_model.pkl"
METRICS_OUTPUT_PATH = "reports/model_metrics/baseline_metrics.csv"
BEST_METRICS_OUTPUT_PATH = "reports/model_metrics/best_model_metrics.json"

RANDOM_STATE = 42


def load_train_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


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


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return metrics


def train_and_evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
) -> tuple[pd.DataFrame, dict]:
    models = get_baseline_models()

    results = []
    trained_models = {}

    for model_name, model in models.items():
        print(f"Training model: {model_name}")

        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        metrics["model"] = model_name

        results.append(metrics)
        trained_models[model_name] = model

    results_df = pd.DataFrame(results)

    return results_df, trained_models


def select_best_model(results_df: pd.DataFrame) -> str:
    """
    Select best model for churn prediction.

    Priority:
    1. Recall
    2. PR-AUC
    3. F1-score
    """
    sorted_results = results_df.sort_values(
        by=["recall", "pr_auc", "f1_score"],
        ascending=False,
    )

    best_model_name = sorted_results.iloc[0]["model"]

    return best_model_name


def save_model(model, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_file)

    print(f"Best baseline model saved to: {output_path}")


def save_metrics(
    results_df: pd.DataFrame,
    best_model_name: str,
) -> None:
    metrics_output_file = Path(METRICS_OUTPUT_PATH)
    metrics_output_file.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(metrics_output_file, index=False)

    best_metrics = (
        results_df[results_df["model"] == best_model_name]
        .iloc[0]
        .to_dict()
    )

    with open(BEST_METRICS_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(best_metrics, file, indent=4)

    print(f"All baseline metrics saved to: {METRICS_OUTPUT_PATH}")
    print(f"Best model metrics saved to: {BEST_METRICS_OUTPUT_PATH}")


def main() -> None:
    print("Starting baseline model training...")

    X_train, X_test, y_train, y_test = load_train_test_data()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    results_df, trained_models = train_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nModel comparison:")
    print(
        results_df.sort_values(
            by=["recall", "pr_auc", "f1_score"],
            ascending=False,
        )
    )

    best_model_name = select_best_model(results_df)
    best_model = trained_models[best_model_name]

    print(f"\nBest model selected: {best_model_name}")

    save_model(best_model, MODEL_OUTPUT_PATH)
    save_metrics(results_df, best_model_name)

    print("Baseline model training completed successfully.")


if __name__ == "__main__":
    main()