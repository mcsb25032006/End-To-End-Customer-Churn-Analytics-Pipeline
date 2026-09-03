"""
Unit tests for PreprocessingPipeline and feature engineering.
"""

import pandas as pd
import pytest
from src.preprocessing import PreprocessingPipeline


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customerID": ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08"],
        "tenure": [1, 12, 24, 36, 48, 60, 72, 5],
        "MonthlyCharges": [25.0, 50.0, 75.0, 100.0, 45.0, 85.0, 110.0, 30.0],
        "TotalCharges": [25.0, 600.0, 1800.0, 3600.0, 2160.0, 5100.0, 7920.0, 150.0],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year", "Two year", "Month-to-month"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)", "Bank transfer (automatic)", "Mailed check", "Credit card (automatic)", "Bank transfer (automatic)", "Electronic check"],
        "churn_flag": [1, 0, 0, 1, 0, 0, 0, 1],
    })


def test_prepare_features_and_target_drops_id(sample_df):
    prep = PreprocessingPipeline()
    X, y = prep.prepare_features_and_target(sample_df)
    assert "customerID" not in X.columns
    assert "churn_flag" not in X.columns
    assert len(X.columns) == 5
    assert len(y) == len(sample_df)


def test_build_preprocessor(sample_df):
    prep = PreprocessingPipeline()
    X, y = prep.prepare_features_and_target(sample_df)
    transformer = prep.build_preprocessor()
    transformed = transformer.fit_transform(X)
    assert transformed.shape[0] == len(sample_df)
    assert transformed.shape[1] > len(prep.numerical_features)


def test_split_data_shapes(sample_df):
    prep = PreprocessingPipeline()
    X, y = prep.prepare_features_and_target(sample_df)
    X_train, X_test, y_train, y_test = prep.split_data(X, y, test_size=0.25, random_state=42, stratify=True)
    assert len(X_train) == 6
    assert len(X_test) == 2
    assert len(y_train) == 6
    assert len(y_test) == 2


def test_get_feature_names(sample_df):
    prep = PreprocessingPipeline()
    X, y = prep.prepare_features_and_target(sample_df)
    transformer = prep.build_preprocessor()
    transformer.fit(X)
    feature_names = prep.get_feature_names(transformer)
    assert "tenure" in feature_names
    assert "MonthlyCharges" in feature_names
    assert any("Contract" in f for f in feature_names)
    assert any("PaymentMethod" in f for f in feature_names)
