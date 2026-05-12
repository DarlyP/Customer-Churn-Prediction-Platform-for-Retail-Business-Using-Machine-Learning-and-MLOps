# Retail Customer Churn Prediction Platform using Machine Learning and MLOps

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Churn%20Prediction-green)
![MLOps](https://img.shields.io/badge/MLOps-MLflow%20%7C%20DVC%20%7C%20Docker-orange)
![API](https://img.shields.io/badge/API-FastAPI-teal)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Tests](https://img.shields.io/badge/Tests-Pytest-brightgreen)

## Project Overview

This project demonstrates an end-to-end machine learning system for retail customer churn prediction, including data validation, feature engineering, model training, experiment tracking, hyperparameter tuning, model explainability, model serving, dashboarding, monitoring, testing, Dockerization, and reproducible ML pipelines.

The goal is to help retail businesses identify customers who are likely to churn and recommend appropriate retention actions based on churn probability and risk level.

---

## Business Problem

Customer churn is a major challenge for retail businesses because losing existing customers can reduce revenue, increase acquisition costs, and weaken long-term customer value.

This project answers key business questions:

- Which customers are most likely to churn?
- What behavioral signals drive churn risk?
- Which customers should be prioritized for retention campaigns?
- Should the business use aggressive or selective retention strategies?
- How can the model be monitored after deployment?

---

## Solution Overview

The project provides an end-to-end churn prediction platform with the following capabilities:

- Data loading and validation
- Feature engineering for customer behavior and churn risk
- Baseline model training
- Hyperparameter tuning with LightGBM and XGBoost
- MLflow experiment tracking
- Final model evaluation
- SHAP model explainability
- FastAPI model serving
- Streamlit business dashboard
- Evidently AI monitoring report
- DVC reproducible pipeline
- Pytest automated testing
- Docker and Docker Compose support
- GitHub Actions CI pipeline

---

## Architecture

```text
Raw Customer Data
        ↓
Data Loading
        ↓
Data Validation
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
Baseline Modeling
        ↓
MLflow Experiment Tracking
        ↓
Hyperparameter Tuning
        ↓
Final Model Evaluation
        ↓
SHAP Explainability
        ↓
FastAPI Model Serving
        ↓
Streamlit Dashboard
        ↓
Monitoring with Evidently AI
        ↓
DVC Reproducible Pipeline
```

---

## Dataset Description

The dataset contains retail customer behavior information used to predict customer churn.

Key columns include:

| Column | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `age_group` | Customer age group |
| `gender` | Customer gender |
| `region` | Customer region |
| `customer_segment` | Customer segment |
| `preferred_channel` | Preferred shopping channel |
| `purchase_frequency` | Number of purchases |
| `avg_order_value` | Average order value |
| `total_spent` | Total customer spending |
| `recency_days` | Days since last activity or purchase |
| `website_visits` | Number of website visits |
| `discount_usage_rate` | Discount usage behavior |
| `email_open_rate` | Email engagement rate |
| `cart_abandonment_rate` | Cart abandonment behavior |
| `loyalty_score` | Customer loyalty score |
| `engagement_score` | Customer engagement score |
| `churn_flag` | Target variable |

---

## Tech Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, LightGBM, XGBoost |
| Experiment Tracking | MLflow |
| Explainability | SHAP |
| Data Validation | Great Expectations |
| Monitoring | Evidently AI |
| Pipeline Versioning | DVC |
| API | FastAPI |
| Dashboard | Streamlit |
| Testing | Pytest, Ruff |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |

---

## Project Structure

```text
.
├── api/
│   ├── main.py
│   └── schemas.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── predictions/
│
├── docs/
│   └── assets/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_experiment.ipynb
│   └── 03_model_explainability.ipynb
│
├── reports/
│   ├── final_evaluation/
│   ├── figures/
│   ├── monitoring/
│   └── shap/
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── validate_data.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── train_baseline.py
│   │   ├── train_with_mlflow.py
│   │   ├── tune_model.py
│   │   ├── evaluate_model.py
│   │   └── predict_model.py
│   │
│   ├── monitoring/
│   │   └── drift_report.py
│   │
│   └── utils/
│       └── config.py
│
├── tests/
│   ├── test_data_validation.py
│   ├── test_features.py
│   ├── test_prediction.py
│   └── test_api.py
│
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
├── Makefile
├── requirements.txt
├── requirements-api.txt
├── requirements-ci.txt
└── README.md
```

---

## Machine Learning Pipeline

The machine learning pipeline includes:

1. Data loading
2. Data validation
3. Feature engineering
4. Train-test split
5. Baseline model training
6. MLflow experiment tracking
7. Hyperparameter tuning
8. Final model evaluation
9. SHAP explainability
10. Prediction pipeline
11. Monitoring report generation

The pipeline is designed to be reproducible using DVC.

Run the full pipeline:

```bash
dvc repro
```

---

## Model Training

The project compares several machine learning models:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Tuned XGBoost
- Tuned LightGBM

The final model is selected based on business-oriented churn metrics, especially:

- Recall for churn class
- PR-AUC
- F1-score
- False negative count

Accuracy is not used as the primary metric because false negatives are costly in churn prediction. A false negative means the business fails to detect a customer who is likely to churn.

---

## Model Performance

Final model performance:

| Metric | Value |
|---|---:|
| Accuracy | 0.9448 |
| Precision Churn | 0.9395 |
| Recall Churn | 0.9412 |
| F1 Churn | 0.9403 |
| ROC-AUC | 0.9901 |
| PR-AUC | 0.9889 |

The model captures a large proportion of churn customers while maintaining strong precision, making it suitable for targeted retention campaigns.

### Confusion Matrix

![Confusion Matrix](docs/assets/confusion_matrix.png)

### ROC Curve

![ROC Curve](docs/assets/roc_curve.png)

### Precision-Recall Curve

![Precision Recall Curve](docs/assets/precision_recall_curve.png)

### Feature Importance

![Feature Importance](docs/assets/feature_importance.png)

---

## Experiment Tracking with MLflow

MLflow is used to track:

- Model name
- Hyperparameters
- Evaluation metrics
- Confusion matrix
- Classification report
- Feature list
- Model artifact

The experiment tracking workflow helps compare baseline and tuned models and select the champion model based on churn-specific business metrics.

Run MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://localhost:5000
```

---

## Model Explainability with SHAP

SHAP is used to explain both global and local model behavior.

The explainability analysis includes:

- Global feature importance
- SHAP summary plot
- Dependence plots
- Local customer waterfall explanation

SHAP helps identify key churn drivers such as customer inactivity, behavioral churn risk, loyalty behavior, engagement behavior, cart abandonment behavior, and spending behavior.

### SHAP Global Feature Importance

![SHAP Global Feature Importance](docs/assets/shap_global_feature_importance.png)

### SHAP Summary Plot

![SHAP Summary Plot](docs/assets/shap_summary_plot.png)

### SHAP Waterfall Plot for High-Risk Customer

![SHAP Waterfall Plot](docs/assets/shap_waterfall_high_risk_customer.png)

---

## Prediction Logic

The prediction pipeline returns:

- `customer_id`
- `churn_probability`
- `prediction_label`
- `prediction_text`
- `risk_level`
- `recommended_action`

Risk level mapping:

| Churn Probability | Risk Level | Recommended Action |
|---:|---|---|
| 0.00 - 0.30 | Low Risk | Maintain engagement |
| 0.31 - 0.60 | Medium Risk | Send personalized offer |
| 0.61 - 0.80 | High Risk | Loyalty discount campaign |
| 0.81 - 1.00 | Critical Risk | Immediate retention call / premium offer |

---

## FastAPI Model Serving

The trained model is served using FastAPI.

Available endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root health response |
| GET | `/health` | Check API and model availability |
| GET | `/model-info` | Return model metadata |
| POST | `/predict` | Predict churn for one customer |
| POST | `/batch-predict` | Predict churn for multiple customers |

Run API locally:

```bash
uvicorn api.main:app --reload
```

Open API documentation:

```text
http://localhost:8000/docs
```

Example request:

```json
{
  "customer_id": "CUST_001",
  "age_group": "35-44",
  "gender": "Female",
  "region": "West",
  "customer_segment": "Returning",
  "preferred_channel": "Mobile App",
  "purchase_frequency": 3,
  "avg_order_value": 45.5,
  "total_spent": 1250.0,
  "recency_days": 75,
  "website_visits": 20,
  "discount_usage_rate": 0.65,
  "email_open_rate": 0.3,
  "cart_abandonment_rate": 0.72,
  "loyalty_score": 35,
  "engagement_score": 40
}
```

Example response:

```json
{
  "customer_id": "CUST_001",
  "churn_probability": 0.873421,
  "prediction_label": 1,
  "prediction": "Churn",
  "risk_level": "Critical Risk",
  "recommended_action": "Immediate retention call / premium offer"
}
```

---

## Streamlit Dashboard

The project includes a Streamlit dashboard for business users.

Dashboard pages:

1. Executive Summary
2. Customer Segmentation
3. Prediction App
4. Model Insights
5. Monitoring

Run dashboard:

```bash
streamlit run dashboard/app.py
```

Open:

```text
http://localhost:8501
```

The dashboard allows users to explore customer churn patterns, review model insights, and generate churn predictions interactively.

---

## Monitoring with Evidently AI

This project includes a simulated production monitoring module using Evidently AI.

The monitoring workflow compares:

- Reference data: training data
- Current data: simulated new monthly customer data

The monitoring report helps detect:

- Data drift
- Feature drift
- Prediction drift
- Target drift
- Missing value changes

This is important because machine learning models can degrade over time when customer behavior changes.

Generate monitoring report:

```bash
python -m src.monitoring.drift_report
```

Output:

```text
reports/monitoring/data_drift_report.html
reports/monitoring/monitoring_summary.json
reports/monitoring/simulated_current_month_data.csv
```

---

## Reproducible Pipeline with DVC

The project uses DVC to make the machine learning workflow reproducible.

Run the full pipeline:

```bash
dvc repro
```

DVC pipeline stages include:

- Data validation
- Feature engineering
- Model training
- Hyperparameter tuning
- Final evaluation
- Monitoring report generation

This allows the full ML pipeline to be reproduced consistently.

---

## Testing with Pytest

The project includes automated tests for:

- Data validation
- Feature engineering outputs
- Prediction pipeline
- FastAPI response schema

Run tests:

```bash
pytest -v
```

Current test result:

```text
17 passed
```

---

## Linting with Ruff

Ruff is used for code quality checks.

Run linting:

```bash
ruff check api src tests
```

Auto-fix simple linting issues:

```bash
ruff check api src tests --fix
```

---

## Docker

Build the FastAPI Docker image:

```bash
docker build -t retail-churn-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 retail-churn-api
```

Open:

```text
http://localhost:8000/docs
```

---

## Docker Compose

Run FastAPI, Streamlit, and MLflow together:

```bash
docker compose up --build
```

Services:

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |
| MLflow | http://localhost:5000 |

Stop services:

```bash
docker compose down
```

---

## GitHub Actions CI

This project includes a GitHub Actions CI pipeline that automatically runs on every push and pull request to the `main` branch.

The CI pipeline performs:

- Repository checkout
- Python environment setup
- Dependency installation
- Linting with Ruff
- Automated testing with Pytest

This helps ensure that the data pipeline, prediction logic, and API schema remain stable after code changes.

---

## How to Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/DarlyP/Customer-Churn-Prediction-Platform-for-Retail-Business-Using-Machine-Learning-and-MLOps.git
cd Customer-Churn-Prediction-Platform-for-Retail-Business-Using-Machine-Learning-and-MLOps
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate environment on Windows:

```bash
.venv\Scripts\activate
```

Activate environment on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Tests

```bash
pytest -v
```

### 5. Run FastAPI

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

### 6. Run Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

Open:

```text
http://localhost:8501
```

### 7. Run MLflow UI

```bash
mlflow ui
```

Open:

```text
http://localhost:5000
```

### 8. Run Monitoring Report

```bash
python -m src.monitoring.drift_report
```

---

## Makefile Commands

This project includes a Makefile for common commands.

Example:

```bash
make test
make api
make dashboard
make monitor
make dvc-repro
```

For Windows users without `make`, run the equivalent commands directly:

```bash
pytest -v
uvicorn api.main:app --reload
streamlit run dashboard/app.py
python -m src.monitoring.drift_report
dvc repro
```

---

## Business Recommendations

Based on the churn model and explainability analysis, the business should:

1. Prioritize customers with high churn probability and high customer value.
2. Use immediate retention calls or premium offers for Critical Risk customers.
3. Use loyalty discount campaigns for High Risk customers.
4. Use personalized offers for Medium Risk customers.
5. Maintain engagement for Low Risk customers.
6. Monitor customer inactivity, cart abandonment, loyalty score, engagement score, and behavioral churn risk as key churn signals.
7. Use model monitoring to decide when retraining is needed.

---

## Key Business Value

This project helps retail teams move from reactive churn analysis to proactive churn prevention.

Instead of waiting until customers leave, the business can:

- Identify high-risk customers earlier
- Prioritize retention campaigns
- Reduce missed churn cases
- Improve customer lifetime value
- Monitor model stability over time
- Integrate churn scoring into business systems through API

---

## Future Improvements

Potential improvements include:

- Deploy FastAPI to a cloud platform
- Add DVC remote storage using S3 or Google Drive
- Add automated retraining pipeline
- Add model registry promotion workflow
- Add real production monitoring from monthly scoring logs
- Add authentication to API endpoints
- Add batch prediction upload in Streamlit
- Add CI Docker build after Docker image is fully stabilized
- Add cloud deployment with AWS, GCP, or Azure

---

## Project Status

End-to-end machine learning and MLOps portfolio project completed with:

- Data validation
- Feature engineering
- Model training
- MLflow experiment tracking
- Hyperparameter tuning
- Final evaluation
- SHAP explainability
- FastAPI serving
- Streamlit dashboard
- Evidently monitoring
- DVC pipeline
- Pytest testing
- Docker support
- GitHub Actions CI
