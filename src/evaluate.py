"""
Model Evaluation and Metrics Module.

Calculates comprehensive classification metrics, confusion matrices, ROC-AUC curves,
odds ratios, and exports visual report artifacts.
"""

import json
import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ModelEvaluator:
    """Evaluates classification models and creates reporting artifacts."""

    def __init__(self, figures_dir: str = "reports/figures") -> None:
        self.figures_dir = figures_dir
        os.makedirs(self.figures_dir, exist_ok=True)

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive performance metrics for binary classification.

        Parameters
        ----------
        y_true : pd.Series
            Ground truth binary target labels.
        y_pred : np.ndarray
            Predicted binary class labels.
        y_prob : Optional[np.ndarray], default=None
            Predicted probabilities for class 1 (churn).

        Returns
        -------
        Dict[str, Any]
            Dictionary of computed metrics.
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_churn": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall_churn": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_churn": float(f1_score(y_true, y_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            ),
        }

        if y_prob is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

        return metrics

    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
    ) -> str:
        """Plot and save confusion matrix heatmap.

        Parameters
        ----------
        y_true : pd.Series
            Actual labels.
        y_pred : np.ndarray
            Predicted labels.
        save_path : Optional[str]
            Path to save figure.

        Returns
        -------
        str
            Saved file path.
        """
        output_path = save_path or os.path.join(self.figures_dir, "confusion_matrix.png")
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Retained (0)", "Churned (1)"],
            yticklabels=["Retained (0)", "Churned (1)"],
            annot_kws={"size": 14},
        )
        plt.title("Customer Churn Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Predicted Label", fontsize=11)
        plt.ylabel("Actual Label", fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        return output_path

    def plot_roc_curve(
        self,
        y_true: pd.Series,
        y_prob: np.ndarray,
        save_path: Optional[str] = None,
    ) -> str:
        """Plot and save ROC-AUC curve.

        Parameters
        ----------
        y_true : pd.Series
            Actual labels.
        y_prob : np.ndarray
            Predicted churn probabilities.
        save_path : Optional[str]
            Path to save figure.

        Returns
        -------
        str
            Saved file path.
        """
        output_path = save_path or os.path.join(self.figures_dir, "roc_curve.png")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_score = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC curve (AUC = {roc_score:.3f})")
        plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=11)
        plt.ylabel("True Positive Rate (Recall)", fontsize=11)
        plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        return output_path

    def plot_feature_importance(
        self,
        df_importance: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> str:
        """Plot and save logistic regression feature coefficients and odds ratios.

        Parameters
        ----------
        df_importance : pd.DataFrame
            DataFrame with Feature, Coefficient, and Odds_Ratio.
        save_path : Optional[str]
            Path to save figure.

        Returns
        -------
        str
            Saved file path.
        """
        output_path = save_path or os.path.join(self.figures_dir, "feature_importance.png")

        df_sorted = df_importance.sort_values(by="Coefficient", ascending=True)
        colors = ["#d62728" if c > 0 else "#2ca02c" for c in df_sorted["Coefficient"]]

        plt.figure(figsize=(8, 6))
        bars = plt.barh(df_sorted["Feature"], df_sorted["Coefficient"], color=colors, alpha=0.85)
        plt.axvline(0, color="black", linestyle="--", linewidth=1)
        plt.title("Logistic Regression Feature Coefficients (Log-Odds)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Coefficient (Positive = Increases Churn Risk, Negative = Protects Retention)", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        return output_path

    def save_metrics(self, metrics: Dict[str, Any], save_path: str = "models/metrics.json") -> None:
        """Save metrics dictionary to JSON file.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Calculated metrics.
        save_path : str
            Destination file path.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
