
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "customers.csv"

REQUIRED_COLUMNS = [
    "customer_id",
    "age",
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "churn",
]


def validate_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # 1. Check required columns
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    # 2. Check missing values
    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Dataset contains missing values")

    # 3. Check unique customer IDs
    if df["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs found")

    # 4. Check numeric data types
    numeric_columns = [
        "age", "tenure_months", "monthly_charges",
        "support_tickets", "churn"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    # 5. Check valid value ranges
    if not df["age"].between(18, 100).all():
        raise ValueError("Age must be between 18 and 100")

    if (df["tenure_months"] < 0).any():
        raise ValueError("Tenure cannot be negative")

    if (df["monthly_charges"] < 0).any():
        raise ValueError("Monthly charges cannot be negative")

    if (df["support_tickets"] < 0).any():
        raise ValueError("Support tickets cannot be negative")

    if not df["churn"].isin([0, 1]).all():
        raise ValueError("Churn must contain only 0 or 1")

    if df.empty:
        raise ValueError("Dataset is empty")

    print("Data validation PASSED!")
    print(f"Total customer records: {len(df)}")
    print("Required columns: OK")
    print("Missing values: NONE")
    print("Customer IDs: UNIQUE")
    print("Numeric ranges: VALID")
    print("Churn values: VALID")

    return df


if __name__ == "__main__":
    validate_data()
