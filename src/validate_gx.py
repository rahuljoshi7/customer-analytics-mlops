
import pandas as pd
import great_expectations as gx
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "customers.csv"


def validate_with_great_expectations():
    df = pd.read_csv(DATA_PATH)

    # Create a Great Expectations Data Context
    context = gx.get_context(mode="ephemeral")

    # Create a Pandas Data Source
    data_source = context.data_sources.add_pandas(
        name="customer_data_source"
    )

    # Create a Data Asset
    data_asset = data_source.add_dataframe_asset(
        name="customer_data"
    )

    # Create a Batch Definition
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "customer_batch"
    )

    # Create a batch from the dataframe
    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": df}
    )

    # Define expectations
    expectations = [
        gx.expectations.ExpectColumnToExist(column="customer_id"),
        gx.expectations.ExpectColumnToExist(column="age"),
        gx.expectations.ExpectColumnToExist(column="tenure_months"),
        gx.expectations.ExpectColumnToExist(column="monthly_charges"),
        gx.expectations.ExpectColumnToExist(column="support_tickets"),
        gx.expectations.ExpectColumnToExist(column="churn"),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="customer_id"
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="age", min_value=18, max_value=100
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="churn", value_set=[0, 1]
        ),
    ]

    results = []

    for expectation in expectations:
        result = batch.validate(expectation)
        results.append(result)
        print(f"{expectation.expectation_type}: {result.success}")

    failed = [result for result in results if not result.success]

    if failed:
        raise AssertionError(
            f"Great Expectations validation failed: {len(failed)} checks"
        )

    print(f"\nGreat Expectations validation PASSED! ({len(results)} checks)")


if __name__ == "__main__":
    validate_with_great_expectations()
