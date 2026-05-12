from pathlib import Path
import json

import joblib
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
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


# ============================================================
# Path Configuration
# ============================================================

TEST_DATA_PATH = "data/processed/test.csv"
FEATURE_LIST_PATH = "data/processed/feature_list.json"
MODEL_PATH = "models/best_model.pkl"

TARGET_COLUMN = "churn_flag"

OUTPUT_DIR = "reports/final_evaluation"

FINAL_METRICS_PATH = f"{OUTPUT_DIR}/final_metrics.json"
CLASSIFICATION_REPORT_PATH = f"{OUTPUT_DIR}/final_classification_report.txt"

CONFUSION_MATRIX_PATH = f"{OUTPUT_DIR}/final_confusion_matrix.png"
ROC_CURVE_PATH = f"{OUTPUT_DIR}/final_roc_curve.png"
PR_CURVE_PATH = f"{OUTPUT_DIR}/final_precision_recall_curve.png"

FEATURE_IMPORTANCE_CSV_PATH = f"{OUTPUT_DIR}/final_feature_importance.csv"
FEATURE_IMPORTANCE_PNG_PATH = f"{OUTPUT_DIR}/final_feature_importance.png"

BUSINESS_INTERPRETATION_PATH = f"{OUTPUT_DIR}/final_business_interpretation.md"


# ============================================================
# Load Data and Model
# ============================================================

def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    test_file = Path(TEST_DATA_PATH)

    if not test_file.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_DATA_PATH}")

    test_df = pd.read_csv(test_file)

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(int)

    return X_test, y_test


def load_model():
    model_file = Path(MODEL_PATH)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(model_file)

    return model


def load_feature_metadata() -> dict:
    feature_file = Path(FEATURE_LIST_PATH)

    if not feature_file.exists():
        return {}

    with open(feature_file, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, str]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    total_churn_customers = int(tp + fn)
    detected_churn_customers = int(tp)
    missed_churn_customers = int(fn)

    total_non_churn_customers = int(tn + fp)
    wrongly_flagged_normal_customers = int(fp)

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

        "total_churn_customers": total_churn_customers,
        "detected_churn_customers": detected_churn_customers,
        "missed_churn_customers": missed_churn_customers,

        "total_non_churn_customers": total_non_churn_customers,
        "wrongly_flagged_normal_customers": wrongly_flagged_normal_customers,

        "detected_churn_rate": detected_churn_customers / total_churn_customers
        if total_churn_customers > 0 else 0,

        "false_alarm_rate": wrongly_flagged_normal_customers / total_non_churn_customers
        if total_non_churn_customers > 0 else 0,
    }

    report_text = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    return metrics, report_text


# ============================================================
# Save Reports
# ============================================================

def save_json(data: dict, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print(f"JSON saved to: {output_path}")


def save_text(text: str, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(text, encoding="utf-8")

    print(f"Text saved to: {output_path}")


def save_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    output_file = Path(CONFUSION_MATRIX_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        values_format="d",
    )

    disp.ax_.set_title("Final Confusion Matrix")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


def save_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    output_file = Path(ROC_CURVE_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("Final ROC Curve")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"ROC curve saved to: {ROC_CURVE_PATH}")


def save_precision_recall_curve(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    output_file = Path(PR_CURVE_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title("Final Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Precision-Recall curve saved to: {PR_CURVE_PATH}")


# ============================================================
# Feature Importance
# ============================================================

def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_values = np.abs(model.coef_).ravel()
    else:
        raise ValueError(
            "Model does not support feature_importances_ or coef_."
        )

    if len(importance_values) != len(feature_names):
        raise ValueError(
            "Feature importance length does not match feature names length."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    )

    return importance_df


def save_feature_importance(model, X_test: pd.DataFrame) -> None:
    try:
        importance_df = get_feature_importance(
            model=model,
            feature_names=X_test.columns.tolist(),
        )

        output_csv = Path(FEATURE_IMPORTANCE_CSV_PATH)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(output_csv, index=False)

        top_features = importance_df.head(20)

        plt.figure(figsize=(10, 8))
        plt.barh(
            top_features["feature"][::-1],
            top_features["importance"][::-1],
        )
        plt.title("Top 20 Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_PNG_PATH, dpi=300)
        plt.close()

        print(f"Feature importance CSV saved to: {FEATURE_IMPORTANCE_CSV_PATH}")
        print(f"Feature importance plot saved to: {FEATURE_IMPORTANCE_PNG_PATH}")

    except ValueError as error:
        print(f"Feature importance skipped: {error}")


# ============================================================
# Business Interpretation
# ============================================================

def determine_campaign_strategy(metrics: dict) -> str:
    recall = metrics["recall_churn"]
    precision = metrics["precision_churn"]
    false_alarm_rate = metrics["false_alarm_rate"]

    if recall >= 0.85 and precision >= 0.80:
        strategy = (
            "The model is suitable for a strong targeted retention campaign. "
            "It detects most churn customers while maintaining strong precision, "
            "so the business can confidently prioritize customers flagged as churn risk."
        )
    elif recall >= 0.85 and precision < 0.80:
        strategy = (
            "The model is more suitable for an aggressive retention campaign. "
            "It captures many churn customers, but it may also flag more normal customers. "
            "This approach is useful when the cost of losing customers is higher than "
            "the cost of offering retention incentives."
        )
    elif precision >= 0.80 and false_alarm_rate <= 0.20:
        strategy = (
            "The model is more suitable for a selective retention campaign. "
            "It is relatively careful when flagging customers, which helps control "
            "campaign costs and reduce unnecessary offers."
        )
    else:
        strategy = (
            "The model should be used carefully for retention prioritization. "
            "Further threshold tuning may be needed depending on whether the business "
            "prefers higher recall or higher precision."
        )

    return strategy


def generate_business_interpretation(metrics: dict) -> str:
    strategy = determine_campaign_strategy(metrics)

    interpretation = f"""
# Final Model Business Interpretation

## 1. Model Performance Summary

The final model was evaluated on the test dataset using churn-focused metrics.

| Metric | Value |
|---|---:|
| Accuracy | {metrics["accuracy"]:.4f} |
| Precision Churn | {metrics["precision_churn"]:.4f} |
| Recall Churn | {metrics["recall_churn"]:.4f} |
| F1 Churn | {metrics["f1_churn"]:.4f} |
| ROC-AUC | {metrics["roc_auc"]:.4f} |
| PR-AUC | {metrics["pr_auc"]:.4f} |

## 2. Confusion Matrix Interpretation

| Component | Count | Business Meaning |
|---|---:|---|
| True Negative | {metrics["true_negative"]} | Non-churn customers correctly predicted as non-churn |
| False Positive | {metrics["false_positive"]} | Normal customers incorrectly flagged as churn risk |
| False Negative | {metrics["false_negative"]} | Churn customers missed by the model |
| True Positive | {metrics["true_positive"]} | Churn customers correctly detected |

## 3. Key Business Questions

### How many churn customers were successfully detected?

The model successfully detected **{metrics["detected_churn_customers"]} out of {metrics["total_churn_customers"]}** actual churn customers.

This means the model captured **{metrics["detected_churn_rate"]:.2%}** of churn customers.

### How many normal customers were incorrectly flagged as churn?

The model incorrectly flagged **{metrics["wrongly_flagged_normal_customers"]} out of {metrics["total_non_churn_customers"]}** normal customers as churn risk.

This means the false alarm rate is **{metrics["false_alarm_rate"]:.2%}**.

### Is this model better for aggressive or selective campaigns?

{strategy}

## 4. Business Recommendation

For a retail churn use case, recall is especially important because a false negative means the company fails to identify a customer who is likely to churn.

However, retention campaigns also have a cost. Therefore, the model should be used together with customer value information such as total spending, loyalty score, and retention priority score.

Recommended usage:

1. Prioritize customers with high churn probability and high customer value.
2. Use aggressive campaigns for high-value high-risk customers.
3. Use lower-cost campaigns such as email or discount reminders for medium-risk customers.
4. Avoid spending expensive retention offers on low-value customers unless strategically necessary.

## 5. Final Notes

The final model should not be judged by accuracy alone. Churn prediction should prioritize recall, PR-AUC, F1-score, and the balance between missed churn customers and unnecessary campaign targeting.
"""

    return interpretation.strip()


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Starting final model evaluation...")

    model = load_model()
    X_test, y_test = load_test_data()

    print(f"Loaded model from: {MODEL_PATH}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    metrics, report_text = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    business_interpretation = generate_business_interpretation(metrics)

    save_json(metrics, FINAL_METRICS_PATH)
    save_text(report_text, CLASSIFICATION_REPORT_PATH)
    save_text(business_interpretation, BUSINESS_INTERPRETATION_PATH)

    save_confusion_matrix(model, X_test, y_test)
    save_roc_curve(model, X_test, y_test)
    save_precision_recall_curve(model, X_test, y_test)
    save_feature_importance(model, X_test)

    print("\nFinal evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nFinal model evaluation completed successfully.")


if __name__ == "__main__":
    main()