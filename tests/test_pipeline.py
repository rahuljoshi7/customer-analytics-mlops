
import json
import time

import joblib
import pandas as pd

from src.validate_data import validate_data

BASE_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"


def test_data_validation():
    df = validate_data()
    assert len(df) == 40
    assert "churn" in df.columns


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_model_can_predict():
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([{
        "age": 30,
        "tenure_months": 12,
        "monthly_charges": 799,
        "support_tickets": 3,
    }])

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_prediction_latency():
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([{
        "age": 30,
        "tenure_months": 12,
        "monthly_charges": 799,
        "support_tickets": 3,
    }])

    start = time.perf_counter()
    model.predict(sample)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"Prediction took {elapsed:.3f} seconds"


def test_model_metrics():
    assert METRICS_PATH.exists()

    metrics = json.loads(METRICS_PATH.read_text())

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["training_rows"] > 0
    assert metrics["testing_rows"] > 0
