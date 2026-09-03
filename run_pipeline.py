#!/usr/bin/env python3
"""
End To End Customer Churn Analytics Pipeline Execution Script.

Coordinates the full analytics and ML workflow:
Data Ingestion -> SQL Transformation -> Preprocessing -> Model Training -> Evaluation -> Visual Reports
"""

import argparse
import os
import sys
import pandas as pd

from src.config_loader import load_config
from src.data_loader import DataLoader
from src.eda import EDARunner
from src.evaluate import ModelEvaluator
from src.model import ChurnModel
from src.preprocessing import PreprocessingPipeline
from src.sql_runner import SQLRunner


def run_pipeline(config_path: str = "config/config.yaml") -> int:
    """Execute the end-to-end customer churn analytics pipeline.

    Parameters
    ----------
    config_path : str, default="config/config.yaml"
        Path to pipeline YAML configuration.

    Returns
    -------
    int
        0 on success, non-zero on error.
    """
    print("=" * 75)
    print(" STARTING: End To End Customer Churn Analytics Pipeline")
    print("=" * 75)

    # 1. Load Configuration
    print("\n[Step 1/7] Loading configuration...")
    config = load_config(config_path)
    paths = config["paths"]
    data_cfg = config["data"]
    split_cfg = config["split"]
    model_cfg = config["model"]
    print(f"  [OK] Configuration loaded from: {config_path}")

    # 2. Ingest and Validate Data
    print("\n[Step 2/7] Ingesting and validating dataset...")
    data_loader = DataLoader(
        target_col=data_cfg["target_column"],
        id_col=data_cfg["id_column"],
        numerical_cols=data_cfg["numerical_features"],
        categorical_cols=data_cfg["categorical_features"],
    )

    raw_path = paths["raw_data_path"]
    df_raw = data_loader.load_data(raw_path)
    df_clean = data_loader.clean_data(df_raw)
    stats = data_loader.get_summary_statistics(df_clean)

    # Save processed data
    os.makedirs(os.path.dirname(paths["processed_data_path"]), exist_ok=True)
    df_clean.to_csv(paths["processed_data_path"], index=False)

    print(f"  [OK] Total Customer Records: {stats['total_customers']}")
    print(f"  [OK] Overall Churn Rate:     {stats['churn_rate_pct']}%")
    print(f"  [OK] Total Revenue:          ${stats['total_revenue']:,.2f}")
    print(f"  [OK] Avg Monthly Charges:    ${stats['avg_monthly_charges']:,.2f}")
    print(f"  [OK] Processed dataset written to: {paths['processed_data_path']}")

    # 3. SQL Transformation & Analytics Verification
    print("\n[Step 3/7] Running embedded SQL analytics queries...")
    with SQLRunner(":memory:") as sql_runner:
        sql_runner.load_table_from_df(df_clean, "telco_customers")
        sql_runner.execute_script(os.path.join(paths["sql_dir"], "02_data_cleaning.sql"))
        analytics_results = sql_runner.run_analytics_suite()
        print(f"  [OK] Executed {len(analytics_results)} SQL analytics queries successfully.")
        
        # Display sample SQL aggregation (Contract breakdown)
        print("\n  SQL Insight - Churn by Contract Type:")
        print("  " + "-" * 55)
        for _, row in analytics_results["churn_by_contract"].iterrows():
            print(f"   * {row['Contract']:<15} | Customers: {row['total_customers']:<4} | Churn: {row['churn_rate_pct']}%")
        print("  " + "-" * 55)

    # 4. Generate Exploratory Data Visualizations
    print("\n[Step 4/7] Generating exploratory data analysis charts...")
    eda_runner = EDARunner(figures_dir=paths["figures_dir"])
    eda_runner.generate_all_plots(df_clean)
    print(f"  [OK] EDA figures saved to: {paths['figures_dir']}/")

    # 5. Preprocessing & Stratified Train/Test Split
    print("\n[Step 5/7] Preparing features and splitting dataset...")
    prep = PreprocessingPipeline(
        numerical_features=data_cfg["numerical_features"],
        categorical_features=data_cfg["categorical_features"],
        target_column=data_cfg["target_column"],
        id_column=data_cfg["id_column"],
    )

    X, y = prep.prepare_features_and_target(df_clean)
    X_train, X_test, y_train, y_test = prep.split_data(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=split_cfg["stratify"],
    )
    preprocessor = prep.build_preprocessor()
    print(f"  [OK] Feature Matrix Shape: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"  [OK] Target Distribution: y_train={dict(y_train.value_counts())}, y_test={dict(y_test.value_counts())}")

    # 6. Model Training & Serialization
    print("\n[Step 6/7] Training Logistic Regression model pipeline...")
    churn_model = ChurnModel(
        preprocessor=preprocessor,
        hyperparameters=model_cfg["hyperparameters"],
    )
    churn_model.fit(X_train, y_train)
    churn_model.save_model(paths["model_save_path"])
    print(f"  [OK] Model serialized to: {paths['model_save_path']}")

    # 7. Model Evaluation & Reporting
    print("\n[Step 7/7] Evaluating model performance and generating reports...")
    y_pred = churn_model.predict(X_test)
    y_prob = churn_model.predict_proba(X_test)[:, 1]

    evaluator = ModelEvaluator(figures_dir=paths["figures_dir"])
    metrics = evaluator.evaluate(y_test, y_pred, y_prob)
    evaluator.save_metrics(metrics, paths["metrics_save_path"])

    evaluator.plot_confusion_matrix(y_test, y_pred)
    evaluator.plot_roc_curve(y_test, y_prob)

    fitted_preprocessor = churn_model.pipeline.named_steps["preprocessor"]
    transformed_feature_names = prep.get_feature_names(fitted_preprocessor)
    df_importance = churn_model.get_feature_importances(transformed_feature_names)
    evaluator.plot_feature_importance(df_importance)

    # Print Executive Model Summary
    print("\n" + "=" * 75)
    print(" MODEL EVALUATION RESULTS SUMMARY")
    print("=" * 75)
    print(f"  * Model Accuracy:      {metrics['accuracy'] * 100:.2f}%")
    print(f"  * ROC-AUC Score:       {metrics['roc_auc']:.4f}")
    print(f"  * Churn Precision:     {metrics['precision_churn'] * 100:.2f}%")
    print(f"  * Churn Recall:        {metrics['recall_churn'] * 100:.2f}%")
    print(f"  * Churn F1-Score:      {metrics['f1_churn']:.4f}")
    print(f"  * Specificity:         {metrics['specificity'] * 100:.2f}%")
    print(f"  * Brier Score:         {metrics['brier_score']:.4f}")
    print("\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"    - True Negatives (Retained correctly):  {cm['true_negatives']}")
    print(f"    - False Positives (False Churn Alarm):  {cm['false_positives']}")
    print(f"    - False Negatives (Missed Churners):    {cm['false_negatives']}")
    print(f"    - True Positives (Caught Churners):     {cm['true_positives']}")

    print("\n  Key Churn Drivers (Odds Ratios):")
    print("  " + "-" * 55)
    for _, row in df_importance.iterrows():
        direction = "INCREASES churn risk" if row["Coefficient"] > 0 else "DECREASES churn risk (protective)"
        print(f"   * {row['Feature']:<35} | Odds Ratio: {row['Odds_Ratio']:.3f} ({direction})")
    print("  " + "-" * 55)

    print("\n [SUCCESS] Pipeline executed successfully from end to end!")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run End To End Customer Churn Analytics Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()

    sys.exit(run_pipeline(config_path=args.config))
