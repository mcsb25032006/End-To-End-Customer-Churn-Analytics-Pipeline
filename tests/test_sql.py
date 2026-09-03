"""
Unit tests for embedded SQLite runner and SQL scripts.
"""

import os
import pandas as pd
import pytest
from src.sql_runner import SQLRunner


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customerID": ["C1", "C2", "C3", "C4"],
        "tenure": [2, 24, 60, 8],
        "MonthlyCharges": [70.0, 40.0, 20.0, 90.0],
        "TotalCharges": [140.0, 960.0, 1200.0, 720.0],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)", "Electronic check"],
        "churn_flag": [1, 0, 0, 1],
    })


def test_sql_runner_load_and_query(sample_df):
    with SQLRunner(":memory:") as runner:
        runner.load_table_from_df(sample_df, "telco_customers")
        df_out = runner.query("SELECT COUNT(*) AS cnt FROM telco_customers;")
        assert df_out["cnt"].iloc[0] == 4


def test_sql_runner_analytics_suite(sample_df):
    with SQLRunner(":memory:") as runner:
        runner.load_table_from_df(sample_df, "telco_customers")
        results = runner.run_analytics_suite()
        assert "overall_kpis" in results
        assert "churn_by_contract" in results
        assert "churn_by_payment_method" in results
        assert "churn_by_charge_tier" in results

        kpis = results["overall_kpis"]
        assert kpis["total_customers"].iloc[0] == 4
        assert kpis["total_churned"].iloc[0] == 2
        assert kpis["churn_rate_pct"].iloc[0] == 50.0
