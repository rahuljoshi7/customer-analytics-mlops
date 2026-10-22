
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"

app = FastAPI(
    title="Customer Analytics ML API",
    description="Customer churn prediction with single and batch REST endpoints",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)

FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "support_tickets",
]


class Customer(BaseModel):
    age: int = Field(ge=18, le=100)
    tenure_months: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    support_tickets: int = Field(ge=0)


class CustomerBatch(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=500)


def predict_customers(customers: list[Customer]) -> list[dict]:
    data = pd.DataFrame(
        [customer.model_dump() for customer in customers],
        columns=FEATURES,
    )

    predictions = model.predict(data)
    results = []

    for prediction in predictions:
        results.append(
            {
                "churn_prediction": int(prediction),
                "churn_label": (
                    "Churn" if int(prediction) == 1 else "No Churn"
                ),
            }
        )

    return results


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(customer: Customer):
    try:
        return predict_customers([customer])[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from exc


@app.post("/predict/batch")
def predict_batch(batch: CustomerBatch):
    try:
        predictions = predict_customers(batch.customers)
        return {
            "count": len(predictions),
            "predictions": predictions,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed",
        ) from exc
