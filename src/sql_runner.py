"""
SQL Execution and Database Runner Module.

Executes and validates SQL schemas, transformations, and analytical queries
using an embedded SQLite engine for zero-dependency local reproducibility.
"""

import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class SQLRunner:
    """Manages SQLite database initialization, data ingestion, and query execution."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close SQLite database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self) -> "SQLRunner":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def load_table_from_df(
        self, df: pd.DataFrame, table_name: str = "telco_customers", if_exists: str = "replace"
    ) -> None:
        """Load a pandas DataFrame into SQLite table.

        Parameters
        ----------
        df : pd.DataFrame
            Source data.
        table_name : str, default="telco_customers"
            Target SQL table.
        if_exists : str, default="replace"
            How to behave if table exists ('fail', 'replace', 'append').
        """
        df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)

    def execute_script(self, sql_script_path: str) -> None:
        """Execute an external SQL script file containing multiple statements.

        Parameters
        ----------
        sql_script_path : str
            Path to .sql file.
        """
        if not os.path.exists(sql_script_path):
            raise FileNotFoundError(f"SQL script not found: {sql_script_path}")

        with open(sql_script_path, "r", encoding="utf-8") as file:
            script_content = file.read()

        cursor = self.conn.cursor()
        cursor.executescript(script_content)
        self.conn.commit()

    def query(self, sql: str, params: Optional[Tuple[Any, ...]] = None) -> pd.DataFrame:
        """Execute a SELECT SQL query and return results as a pandas DataFrame.

        Parameters
        ----------
        sql : str
            SQL query string.
        params : Optional[Tuple[Any, ...]]
            Query parameters.

        Returns
        -------
        pd.DataFrame
            Query result set.
        """
        return pd.read_sql_query(sql, self.conn, params=params)

    def run_analytics_suite(self) -> Dict[str, pd.DataFrame]:
        """Run all core business analytical queries and return results dictionary.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Named query results for churn KPIs, contract breakdowns, cohorts, etc.
        """
        queries = {
            "overall_kpis": """
                SELECT 
                    COUNT(*) AS total_customers,
                    SUM(churn_flag) AS total_churned,
                    ROUND(AVG(churn_flag) * 100.0, 2) AS churn_rate_pct,
                    ROUND(SUM(TotalCharges), 2) AS total_revenue,
                    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
                FROM telco_customers;
            """,
            "churn_by_contract": """
                SELECT 
                    Contract,
                    COUNT(*) AS total_customers,
                    SUM(churn_flag) AS churned_customers,
                    ROUND(AVG(churn_flag) * 100.0, 2) AS churn_rate_pct,
                    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_revenue
                FROM telco_customers
                GROUP BY Contract
                ORDER BY churn_rate_pct DESC;
            """,
            "churn_by_payment_method": """
                SELECT 
                    PaymentMethod,
                    COUNT(*) AS total_customers,
                    SUM(churn_flag) AS churned_customers,
                    ROUND(AVG(churn_flag) * 100.0, 2) AS churn_rate_pct
                FROM telco_customers
                GROUP BY PaymentMethod
                ORDER BY churn_rate_pct DESC;
            """,
            "churn_by_charge_tier": """
                SELECT 
                    CASE 
                        WHEN MonthlyCharges < 40 THEN '1. Low (< $40)'
                        WHEN MonthlyCharges BETWEEN 40 AND 80 THEN '2. Medium ($40 - $80)'
                        ELSE '3. High (> $80)'
                    END AS charge_tier,
                    COUNT(*) AS total_customers,
                    SUM(churn_flag) AS churned_customers,
                    ROUND(AVG(churn_flag) * 100.0, 2) AS churn_rate_pct
                FROM telco_customers
                GROUP BY charge_tier
                ORDER BY charge_tier;
            """,
            "churn_by_tenure_cohort": """
                SELECT 
                    CASE 
                        WHEN tenure <= 12 THEN '0 - 12 Months'
                        WHEN tenure BETWEEN 13 AND 36 THEN '13 - 36 Months'
                        WHEN tenure BETWEEN 37 AND 60 THEN '37 - 60 Months'
                        ELSE '60+ Months'
                    END AS tenure_cohort,
                    COUNT(*) AS total_customers,
                    SUM(churn_flag) AS churned_customers,
                    ROUND(AVG(churn_flag) * 100.0, 2) AS churn_rate_pct,
                    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
                FROM telco_customers
                GROUP BY tenure_cohort
                ORDER BY churn_rate_pct DESC;
            """,
            "revenue_at_risk": """
                SELECT 
                    Contract,
                    PaymentMethod,
                    COUNT(*) AS at_risk_customers,
                    ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue_at_risk,
                    ROUND(AVG(churn_flag) * 100.0, 2) AS segment_churn_rate_pct
                FROM telco_customers
                WHERE churn_flag = 1
                GROUP BY Contract, PaymentMethod
                ORDER BY monthly_revenue_at_risk DESC;
            """,
        }

        results: Dict[str, pd.DataFrame] = {}
        for name, sql in queries.items():
            results[name] = self.query(sql)
        return results
