"""
Unit tests for DataLoader and data validation.
"""

import os
import pandas as pd
import pytest
from src.data_loader import DataLoader


@pytest.fixture
def sample_data_path():
    return "data/raw/churn_analysis.csv"


def test_data_loader_init():
    loader = DataLoader()
    assert loader.target_col == "churn_flag"
    assert loader.id_col == "customerID"
    assert "tenure" in loader.numerical_cols


def test_load_data_valid_file(sample_data_path):
    if not os.path.exists(sample_data_path):
        pytest.skip("Sample data file not available.")
    loader = DataLoader()
    df = loader.load_data(sample_data_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "churn_flag" in df.columns


def test_load_data_missing_file():
    loader = DataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_data("data/non_existent_file.csv")


def test_validate_schema_missing_columns():
    loader = DataLoader()
    bad_df = pd.DataFrame({"customerID": ["123"], "tenure": [12]})
    with pytest.raises(ValueError, match="missing required columns"):
        loader.validate_schema(bad_df)


def test_clean_data_null_handling():
    loader = DataLoader()
    df = pd.DataFrame({
        "customerID": ["1", "2", "3"],
        "tenure": [1, 2, 3],
        "MonthlyCharges": [20.0, 30.0, 40.0],
        "TotalCharges": ["20.0", " ", "120.0"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaymentMethod": ["Mailed check", "Electronic check", "Mailed check"],
        "churn_flag": ["0", "1", "0"],
    })
    cleaned = loader.clean_data(df)
    assert len(cleaned) == 2  # Row with ' ' should be dropped
    assert cleaned["TotalCharges"].dtype in ["float64", "float32"]
    assert cleaned["churn_flag"].dtype == int


def test_get_summary_statistics():
    loader = DataLoader()
    df = pd.DataFrame({
        "customerID": ["1", "2", "3", "4"],
        "tenure": [10, 20, 30, 40],
        "MonthlyCharges": [50.0, 50.0, 100.0, 100.0],
        "TotalCharges": [500.0, 1000.0, 3000.0, 4000.0],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)", "Electronic check"],
        "churn_flag": [0, 1, 0, 1],
    })
    stats = loader.get_summary_statistics(df)
    assert stats["total_customers"] == 4
    assert stats["churn_rate_pct"] == 50.0
    assert stats["total_revenue"] == 8500.0
    assert stats["avg_monthly_charges"] == 75.0
