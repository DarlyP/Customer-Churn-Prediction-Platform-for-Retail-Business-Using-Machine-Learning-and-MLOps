from typing import List

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    customer_id: str = Field(..., example="CUST_001")
    age_group: str = Field(..., example="35-44")
    gender: str = Field(..., example="Female")
    region: str = Field(..., example="West")
    customer_segment: str = Field(..., example="Returning")
    preferred_channel: str = Field(..., example="Mobile App")

    purchase_frequency: float = Field(..., example=3)
    avg_order_value: float = Field(..., example=45.5)
    total_spent: float = Field(..., example=1250.0)
    recency_days: float = Field(..., example=75)
    website_visits: float = Field(..., example=20)

    discount_usage_rate: float = Field(..., example=0.65)
    email_open_rate: float = Field(..., example=0.30)
    cart_abandonment_rate: float = Field(..., example=0.72)

    loyalty_score: float = Field(..., example=35)
    engagement_score: float = Field(..., example=40)


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    prediction_label: int
    prediction: str
    risk_level: str
    recommended_action: str


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerInput]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    message: str


class ModelInfoResponse(BaseModel):
    model_path: str
    preprocessor_path: str
    model_available: bool
    preprocessor_available: bool
    model_type: str | None = None
    metrics: dict | None = None