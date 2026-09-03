"""
Unit tests for ChurnModel training, inference, serialization, and evaluation.
"""

import os
import numpy as np
import pandas as pd
import pytest
from src.evaluate import ModelEvaluator
from src.model import ChurnModel
from src.preprocessing import PreprocessingPipeline


@pytest.fixture
def clean_dataset():
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "customerID": [f"ID_{i}" for i in range(n)],
        "tenure": np.random.randint(1, 72, size=n),
        "MonthlyCharges": np.random.uniform(20.0, 110.0, size=n),
        "TotalCharges": np.random.uniform(100.0, 7000.0, size=n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
        "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Credit card (automatic)", "Bank transfer (automatic)"], size=n),
        "churn_flag": np.random.choice([0, 1], size=n, p=[0.75, 0.25]),
    })


def test_model_fit_predict(clean_dataset):
    prep = PreprocessingPipeline()
    X, y = prep.prepare_features_and_target(clean_dataset)
    X_train, X_test, y_train, y_test = prep.split_data(X, y, test_size=0.3, random_state=42)

    preprocessor = prep.build_preprocessor()
    model = ChurnModel(preprocessor=preprocessor)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    assert len(preds) == len(X_test)
    assert probs.shape == (len(X_test), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert set(preds).issubset({0, 1})


def test_model_save_load(clean_dataset, tmp_path):
    prep = PreprocessingPipeline()
    X, y = prep.prepare_features_and_target(clean_dataset)
    preprocessor = prep.build_preprocessor()

    model = ChurnModel(preprocessor=preprocessor)
    model.fit(X, y)

    save_file = str(tmp_path / "test_model.joblib")
    model.save_model(save_file)
    assert os.path.exists(save_file)

    loaded_pipeline = ChurnModel.load_model(save_file)
    loaded_preds = loaded_pipeline.predict(X)
    assert len(loaded_preds) == len(X)


def test_model_evaluator(clean_dataset, tmp_path):
    evaluator = ModelEvaluator(figures_dir=str(tmp_path))
    y_true = clean_dataset["churn_flag"]
    y_pred = y_true.values
    y_prob = np.where(y_true == 1, 0.9, 0.1)

    metrics = evaluator.evaluate(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision_churn"] == 1.0
    assert metrics["recall_churn"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert "confusion_matrix" in metrics
