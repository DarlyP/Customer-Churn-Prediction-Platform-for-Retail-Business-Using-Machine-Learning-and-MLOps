from pathlib import Path
import json

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    CustomerInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.models.predict_model import predict_customer_churn


MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
METRICS_PATH = "reports/model_metrics/best_tuned_model_metrics.json"


app = FastAPI(
    title="Retail Customer Churn Prediction API",
    description=(
        "FastAPI service for predicting customer churn risk, "
        "risk level, and recommended retention action."
    ),
    version="1.0.0",
)


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(
        status="ok",
        message="Retail Customer Churn Prediction API is running.",
    )


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    model_exists = Path(MODEL_PATH).exists()
    preprocessor_exists = Path(PREPROCESSOR_PATH).exists()

    if not model_exists or not preprocessor_exists:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Model or preprocessor file is missing.",
                "model_available": model_exists,
                "preprocessor_available": preprocessor_exists,
            },
        )

    return HealthResponse(
        status="ok",
        message="Model and preprocessor are available.",
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    model_path = Path(MODEL_PATH)
    preprocessor_path = Path(PREPROCESSOR_PATH)

    metrics = None

    if Path(METRICS_PATH).exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

    model_type = None

    if metrics:
        model_type = metrics.get("best_model") or metrics.get("model")

    return ModelInfoResponse(
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        model_available=model_path.exists(),
        preprocessor_available=preprocessor_path.exists(),
        model_type=model_type,
        metrics=metrics,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput) -> PredictionResponse:
    try:
        input_df = pd.DataFrame([customer.model_dump()])

        prediction_df = predict_customer_churn(input_df)

        result = prediction_df.iloc[0].to_dict()

        return PredictionResponse(
            customer_id=result["customer_id"],
            churn_probability=result["churn_probability"],
            prediction_label=result["prediction_label"],
            prediction=result["prediction_text"],
            risk_level=result["risk_level"],
            recommended_action=result["recommended_action"],
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    try:
        input_data = [
            customer.model_dump()
            for customer in request.customers
        ]

        input_df = pd.DataFrame(input_data)

        prediction_df = predict_customer_churn(input_df)

        predictions = []

        for _, row in prediction_df.iterrows():
            predictions.append(
                PredictionResponse(
                    customer_id=row["customer_id"],
                    churn_probability=row["churn_probability"],
                    prediction_label=row["prediction_label"],
                    prediction=row["prediction_text"],
                    risk_level=row["risk_level"],
                    recommended_action=row["recommended_action"],
                )
            )

        return BatchPredictionResponse(predictions=predictions)

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )