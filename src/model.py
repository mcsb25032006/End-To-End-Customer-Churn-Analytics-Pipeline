"""
Machine Learning Model Management Module.

Constructs, trains, evaluates, serializes, and interprets the Logistic Regression
customer churn classification model.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class ChurnModel:
    """Manages training, inference, and serialization of the churn model."""

    def __init__(
        self,
        preprocessor: ColumnTransformer,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.hyperparameters = hyperparameters or {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
        }
        self.pipeline: Optional[Pipeline] = None

    def build_pipeline(self) -> Pipeline:
        """Construct full Scikit-Learn pipeline with preprocessor and classifier.

        Returns
        -------
        Pipeline
            Configured Pipeline ready for fitting.
        """
        classifier = LogisticRegression(**self.hyperparameters)
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", classifier),
            ]
        )
        return self.pipeline

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """Train the full churn prediction pipeline.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Training target vector.

        Returns
        -------
        Pipeline
            Trained Pipeline instance.
        """
        if self.pipeline is None:
            self.build_pipeline()
        self.pipeline.fit(X_train, y_train)
        return self.pipeline

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary churn labels (0 = Retained, 1 = Churned).

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict.

        Returns
        -------
        np.ndarray
            Predicted binary class labels.
        """
        if self.pipeline is None:
            raise ValueError("Model pipeline has not been trained or loaded.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict churn probability distribution.

        Parameters
        ----------
        X : pd.DataFrame
            Features to evaluate.

        Returns
        -------
        np.ndarray
            Probability array of shape (n_samples, 2).
        """
        if self.pipeline is None:
            raise ValueError("Model pipeline has not been trained or loaded.")
        return self.pipeline.predict_proba(X)

    def save_model(self, file_path: str) -> None:
        """Serialize trained pipeline to disk.

        Parameters
        ----------
        file_path : str
            Destination path for joblib artifact.
        """
        if self.pipeline is None:
            raise ValueError("Cannot save an untrained pipeline.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.pipeline, file_path)

    @classmethod
    def load_model(cls, file_path: str) -> Pipeline:
        """Load serialized pipeline from disk.

        Parameters
        ----------
        file_path : str
            Path to saved joblib file.

        Returns
        -------
        Pipeline
            Loaded Pipeline object.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found at: {file_path}")
        return joblib.load(file_path)

    def get_feature_importances(self, feature_names: List[str]) -> pd.DataFrame:
        """Extract logistic regression coefficients and odds ratios for interpretation.

        Parameters
        ----------
        feature_names : List[str]
            Transformed feature column names.

        Returns
        -------
        pd.DataFrame
            DataFrame with Feature, Coefficient, and Odds_Ratio sorted by impact.
        """
        if self.pipeline is None:
            raise ValueError("Model has not been fitted.")

        classifier: LogisticRegression = self.pipeline.named_steps["classifier"]
        coefficients = classifier.coef_[0]

        df_importance = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient": coefficients,
                "Odds_Ratio": np.exp(coefficients),
                "Abs_Impact": np.abs(coefficients),
            }
        ).sort_values(by="Abs_Impact", ascending=False).reset_index(drop=True)

        return df_importance
