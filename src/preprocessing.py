"""
Preprocessing and Feature Engineering Module.

Provides preprocessor pipelines, categorical encoding, feature scaling,
stratified data splitting, and prevention of data leakage.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class PreprocessingPipeline:
    """Prepares and transforms raw features for machine learning modeling."""

    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        target_column: str = "churn_flag",
        id_column: str = "customerID",
    ) -> None:
        self.numerical_features = numerical_features or [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ]
        self.categorical_features = categorical_features or [
            "Contract",
            "PaymentMethod",
        ]
        self.target_column = target_column
        self.id_column = id_column
        self.preprocessor: Optional[ColumnTransformer] = None

    def build_preprocessor(self) -> ColumnTransformer:
        """Construct Scikit-Learn ColumnTransformer for scaling and one-hot encoding.

        Returns
        -------
        ColumnTransformer
            Configured ColumnTransformer instance.
        """
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numerical_features),
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                    self.categorical_features,
                ),
            ],
            remainder="drop",
        )
        return self.preprocessor

    def prepare_features_and_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate predictive feature matrix X and target vector y, dropping identifiers.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned dataset.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Feature DataFrame X (without customerID) and target Series y.
        """
        cols_to_drop = [self.target_column]
        if self.id_column in df.columns:
            cols_to_drop.append(self.id_column)

        X = df.drop(columns=cols_to_drop, errors="ignore")
        y = df[self.target_column]
        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.25,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Perform train/test splitting with optional stratification.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        test_size : float, default=0.25
            Fraction of data for testing.
        random_state : int, default=42
            Random seed for reproducibility.
        stratify : bool, default=True
            Whether to stratify on target class distribution.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test
        """
        stratify_col = y if stratify else None
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col,
        )

    def get_feature_names(self, preprocessor: ColumnTransformer) -> List[str]:
        """Extract output feature names after transformation.

        Parameters
        ----------
        preprocessor : ColumnTransformer
            Fitted ColumnTransformer.

        Returns
        -------
        List[str]
            List of transformed feature names.
        """
        feature_names: List[str] = list(self.numerical_features)
        cat_encoder = preprocessor.named_transformers_.get("cat")
        if cat_encoder and hasattr(cat_encoder, "get_feature_names_out"):
            cat_names = list(cat_encoder.get_feature_names_out(self.categorical_features))
            feature_names.extend(cat_names)
        return feature_names
