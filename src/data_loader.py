"""
Data Loader and Validation Module.

Handles dataset ingestion, structural validation, type casting, missing value handling,
and summary diagnostics.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class DataLoader:
    """Loads and validates customer churn datasets."""

    def __init__(
        self,
        target_col: str = "churn_flag",
        id_col: str = "customerID",
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> None:
        self.target_col = target_col
        self.id_col = id_col
        self.numerical_cols = numerical_cols or ["tenure", "MonthlyCharges", "TotalCharges"]
        self.categorical_cols = categorical_cols or ["Contract", "PaymentMethod"]

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load customer dataset from CSV file.

        Parameters
        ----------
        file_path : str
            Path to the CSV dataset.

        Returns
        -------
        pd.DataFrame
            Loaded and validated DataFrame.

        Raises
        ------
        FileNotFoundError
            If file_path does not exist.
        ValueError
            If expected columns are missing.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at: {file_path}")

        df = pd.read_csv(file_path)
        self.validate_schema(df)
        return df

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Validate required columns exist in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to check.

        Raises
        ------
        ValueError
            If required columns are absent.
        """
        required_cols = set(self.numerical_cols + self.categorical_cols + [self.target_col])
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset by handling data types, missing values, and invalid entries.

        Parameters
        ----------
        df : pd.DataFrame
            Raw or staged DataFrame.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame.
        """
        cleaned_df = df.copy()

        # Ensure TotalCharges is numeric, coercing spaces/blanks to NaN
        if "TotalCharges" in cleaned_df.columns:
            cleaned_df["TotalCharges"] = pd.to_numeric(
                cleaned_df["TotalCharges"], errors="coerce"
            )

        # Impute or drop null TotalCharges (new customers with tenure=0 or missing)
        null_count = cleaned_df["TotalCharges"].isnull().sum()
        if null_count > 0:
            cleaned_df = cleaned_df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

        # Standardize target column to integer 0/1
        if self.target_col in cleaned_df.columns:
            if cleaned_df[self.target_col].dtype == object:
                cleaned_df[self.target_col] = cleaned_df[self.target_col].map(
                    {"Yes": 1, "No": 0, "1": 1, "0": 0, 1: 1, 0: 0}
                )
            cleaned_df[self.target_col] = cleaned_df[self.target_col].astype(int)

        # Remove duplicate rows if any
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

        return cleaned_df

    def get_summary_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute key business KPIs and dataset diagnostics.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to evaluate.

        Returns
        -------
        Dict[str, float]
            Dictionary of metrics (total customers, churn rate %, revenue, etc.).
        """
        total_customers = len(df)
        churn_rate = float(df[self.target_col].mean() * 100) if total_customers > 0 else 0.0
        total_revenue = float(df["TotalCharges"].sum()) if "TotalCharges" in df.columns else 0.0
        avg_monthly = float(df["MonthlyCharges"].mean()) if "MonthlyCharges" in df.columns else 0.0

        return {
            "total_customers": total_customers,
            "churn_rate_pct": round(churn_rate, 2),
            "total_revenue": round(total_revenue, 2),
            "avg_monthly_charges": round(avg_monthly, 2),
        }
