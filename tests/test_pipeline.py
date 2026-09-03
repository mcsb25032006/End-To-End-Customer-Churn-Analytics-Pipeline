"""
Integration test for the complete customer churn analytics pipeline.
"""

import os
import pytest
from run_pipeline import run_pipeline


def test_full_pipeline_execution():
    config_file = "config/config.yaml"
    if not os.path.exists(config_file):
        pytest.skip("Config file not found.")

    exit_code = run_pipeline(config_path=config_file)
    assert exit_code == 0

    assert os.path.exists("models/churn_pipeline.joblib")
    assert os.path.exists("models/metrics.json")
    assert os.path.exists("reports/figures/confusion_matrix.png")
    assert os.path.exists("reports/figures/roc_curve.png")
    assert os.path.exists("reports/figures/feature_importance.png")
    assert os.path.exists("reports/figures/churn_distribution.png")
