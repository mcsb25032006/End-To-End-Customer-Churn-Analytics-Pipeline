"""
Exploratory Data Analysis (EDA) Visualization Module.

Generates diagnostic charts for churn distributions, contract dynamics, payment behavior,
and charge patterns.
"""

import os
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class EDARunner:
    """Generates and exports exploratory data analysis charts."""

    def __init__(self, figures_dir: str = "reports/figures") -> None:
        self.figures_dir = figures_dir
        os.makedirs(self.figures_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")

    def generate_all_plots(self, df: pd.DataFrame) -> None:
        """Generate all EDA figures and save to figures directory.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned customer churn dataset.
        """
        self.plot_churn_distribution(df)
        self.plot_contract_churn(df)
        self.plot_payment_method_churn(df)
        self.plot_charges_distribution(df)

    def plot_churn_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """Plot and save overall churn class distribution."""
        output_path = save_path or os.path.join(self.figures_dir, "churn_distribution.png")

        fig, ax = plt.subplots(figsize=(6, 4.5))
        counts = df["churn_flag"].value_counts().rename({0: "Retained (0)", 1: "Churned (1)"})
        sns.barplot(
            x=counts.index,
            y=counts.values,
            hue=counts.index,
            ax=ax,
            palette=["#2ca02c", "#d62728"],
            legend=False,
        )
        ax.set_title("Customer Churn Distribution", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel("Customer Count", fontsize=10)
        ax.set_xlabel("Status", fontsize=10)

        # Add data labels
        total = len(df)
        for i, val in enumerate(counts.values):
            pct = (val / total) * 100
            ax.text(i, val + (total * 0.015), f"{val} ({pct:.1f}%)", ha="center", fontsize=10, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path

    def plot_contract_churn(self, df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """Plot churn rate broken down by contract type."""
        output_path = save_path or os.path.join(self.figures_dir, "churn_by_contract.png")

        contract_churn = (
            df.groupby("Contract")["churn_flag"]
            .agg(total="count", churn_rate="mean")
            .reset_index()
            .sort_values(by="churn_rate", ascending=False)
        )
        contract_churn["churn_rate_pct"] = contract_churn["churn_rate"] * 100

        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(
            data=contract_churn,
            x="Contract",
            y="churn_rate_pct",
            hue="Contract",
            palette="Blues_r",
            ax=ax,
            legend=False,
        )
        ax.set_title("Churn Rate by Contract Type (%)", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel("Churn Rate (%)", fontsize=10)
        ax.set_xlabel("Contract Type", fontsize=10)

        for i, row in contract_churn.iterrows():
            idx = list(contract_churn["Contract"]).index(row["Contract"])
            ax.text(idx, row["churn_rate_pct"] + 1, f"{row['churn_rate_pct']:.1f}%\n(n={row['total']})", ha="center", fontsize=9)

        plt.ylim(0, max(contract_churn["churn_rate_pct"]) + 10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path

    def plot_payment_method_churn(self, df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """Plot churn rate across payment methods."""
        output_path = save_path or os.path.join(self.figures_dir, "churn_by_payment_method.png")

        pm_churn = (
            df.groupby("PaymentMethod")["churn_flag"]
            .agg(total="count", churn_rate="mean")
            .reset_index()
            .sort_values(by="churn_rate", ascending=False)
        )
        pm_churn["churn_rate_pct"] = pm_churn["churn_rate"] * 100

        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(
            data=pm_churn,
            y="PaymentMethod",
            x="churn_rate_pct",
            hue="PaymentMethod",
            palette="Reds_r",
            ax=ax,
            legend=False,
        )
        ax.set_title("Churn Rate by Payment Method (%)", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Churn Rate (%)", fontsize=10)
        ax.set_ylabel("Payment Method", fontsize=10)

        for idx, row in pm_churn.iterrows():
            ax.text(row["churn_rate_pct"] + 0.8, idx, f"{row['churn_rate_pct']:.1f}%", va="center", fontsize=9, fontweight="bold")

        plt.xlim(0, max(pm_churn["churn_rate_pct"]) + 8)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path

    def plot_charges_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """Plot Monthly Charges vs Churn boxplot and distribution."""
        output_path = save_path or os.path.join(self.figures_dir, "monthly_charges_vs_churn.png")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        sns.boxplot(
            x="churn_flag",
            y="MonthlyCharges",
            hue="churn_flag",
            data=df,
            palette=["#2ca02c", "#d62728"],
            ax=ax1,
            legend=False,
        )
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(["Retained (0)", "Churned (1)"])
        ax1.set_title("Monthly Charges vs Churn Status", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Customer Status", fontsize=10)
        ax1.set_ylabel("Monthly Charges ($)", fontsize=10)

        sns.kdeplot(
            data=df,
            x="MonthlyCharges",
            hue="churn_flag",
            common_norm=False,
            palette=["#2ca02c", "#d62728"],
            fill=True,
            alpha=0.3,
            ax=ax2,
        )
        ax2.set_title("Monthly Charges Density", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Monthly Charges ($)", fontsize=10)
        ax2.legend(title="Status", labels=["Churned (1)", "Retained (0)"])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path
