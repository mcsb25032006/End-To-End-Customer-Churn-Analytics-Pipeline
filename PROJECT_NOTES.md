# 📘 Technical Project Notes & Interview Preparation Guide

**Project Title:** End To End Customer Churn Analytics Pipeline  
**Role Scope:** Senior Software Engineer / Data Scientist / Analytics Engineer  

---

## ⏱️ 60–90 Second Executive Elevator Pitch

> *"The **End To End Customer Churn Analytics Pipeline** is a production-grade analytics and machine learning solution designed to identify customer attrition patterns and predict at-risk subscribers in subscription telecommunications businesses.*
> 
> *The pipeline spans the full data lifecycle: starting with raw data validation and relational SQL modeling, moving through a leakage-free Scikit-Learn preprocessing and Logistic Regression pipeline, and culminating in an interactive Power BI dashboard with custom DAX measures.*
> 
> *Key technical highlights include:
> 1. Eliminating data leakage by isolating string identifiers and embedding `StandardScaler` and `OneHotEncoder` strictly within cross-validated pipeline folds.
> 2. Interpreting model behavior through rigorous log-odds and **Odds Ratios ($e^\beta$)**, proving that month-to-month contracts and high monthly charges are the dominant attrition drivers.
> 3. Delivering an embedded SQLite query runner for zero-dependency SQL verification alongside enterprise-ready MySQL scripts.
> 4. Achieving **78% accuracy** and an **ROC-AUC of 0.844**, supported by 100% automated test coverage across ingestion, preprocessing, modeling, and SQL execution."*

---

## 🏗️ Component-by-Component Architecture Deep-Dive

### 1. Configuration Layer (`config/config.yaml`)
- **Design Rationale:** Hardcoding hyperparameters, file paths, and random seeds within scripts creates fragile, difficult-to-maintain code. Centralizing all configuration in YAML separates business parameters from execution logic, allowing seamless environment-specific overrides (e.g., test vs. production).
- **Key Parameters:** Seed `42`, stratified test size `0.25`, features lists, L2-regularized `lbfgs` solver.

### 2. Data Ingestion & Validation (`src/data_loader.py`)
- **Design Rationale:** Real-world datasets often arrive with missing fields, malformed data types (e.g., whitespace strings in numeric columns), or unexpected schema drift.
- **Implementation:**
  - Enforces schema contracts (`validate_schema`).
  - Coerces `TotalCharges` strings to numeric, dropping invalid whitespace entries.
  - Normalizes target labels to binary integers ($0, 1$).
  - Computes baseline executive KPIs (Total Base, Churn Rate %, Revenue, Avg Spend).

### 3. SQL Engine & Analytical Layer (`sql/` & `src/sql_runner.py`)
- **Design Rationale:** SQL is the industry standard for relational data modeling, data quality validation, and cohort aggregation.
- **Implementation:**
  - `01_schema.sql`: Establishes DDL definitions with explicit type constraints.
  - `02_data_cleaning.sql`: Cleans records and creates the structured `churn_analysis` analytics table.
  - `03_analytics_queries.sql`: Provides 7 business analytics queries including tenure cohorting ($0-12m$, $13-36m$, $37-60m$, $60m+$), revenue-at-risk calculations, and spend-tier cross-tabulations.
  - `src/sql_runner.py`: Provides an embedded, zero-dependency SQLite runner for programmatic query testing in CI/CD without requiring external database servers.

### 4. Preprocessing & Feature Engineering (`src/preprocessing.py`)
- **Data Leakage Mitigation:**
  - `customerID` is stripped prior to model input.
  - Categorical nominal variables (`Contract`, `PaymentMethod`) are encoded via `OneHotEncoder(drop='first')`, preventing artificial ordinal assumptions ($0 < 1 < 2 < 3$).
  - Numerical continuous variables (`tenure`, `MonthlyCharges`, `TotalCharges`) are standardized via `StandardScaler` $(\mu=0, \sigma=1)$ so that L2 regularization penalizes all coefficients uniformly.

### 5. Machine Learning Modeling (`src/model.py`)
- **Why Logistic Regression?**
  - High interpretability: Exact mathematical mapping between feature shifts and log-odds of churn.
  - Well-calibrated probabilities: Direct probabilistic outputs $P(Y=1|X) = \frac{1}{1 + e^{-z}}$.
  - Computational efficiency and low latency for real-time inference.
- **Serialization:** Full Scikit-Learn `Pipeline` is persisted to `models/churn_pipeline.joblib`.

### 6. Evaluation & Diagnostics (`src/evaluate.py`)
- **Multi-Metric Suite:** Evaluates Accuracy (78.0%), Precision (56.5%), Recall (52.0%), F1-Score (0.542), Specificity (86.7%), ROC-AUC (0.844), and Brier Score (0.134).
- **Visual Artifacts:** Saves high-resolution publication figures (`confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`) to `reports/figures/`.

### 7. Power BI Executive Reporting (`powerbi/`)
- **Executive Decision Support:** Tracks top-level KPIs, contract churn composition, payment method attrition, and billing vs. tenure scatter distributions.
- **DAX Formulas:** Uses explicit measures (`Total Customers`, `Churn Rate %`, `Total Revenue`, `Monthly Revenue at Risk`) rather than implicit aggregations.

---

## 🎯 15 Likely Interview Questions & Defensible Answers

### Q1: Why did you choose Logistic Regression instead of Random Forest or XGBoost?
**Answer:**  
*"Logistic Regression was selected as the foundational baseline model because of its high statistical interpretability, well-calibrated probabilistic output, and direct mathematical link to business odds ratios ($e^\beta$). In customer retention contexts, business stakeholders need to understand not just whether a customer will churn, but the exact directional multiplier of each feature (e.g., an odds ratio of 2.33 for higher monthly charges, or 0.33 for one-year contracts). Logistic Regression provides these transparent log-odds directly while establishing a strong baseline (0.844 ROC-AUC). In production, this can serve as the calibrated benchmark against which non-linear ensemble models like XGBoost are compared."*

---

### Q2: What data leakage risks existed in the initial codebase and how did you resolve them?
**Answer:**  
*"The original codebase had two critical leakage and methodology flaws:
1. `customerID` (a unique identifier string) was label-encoded and included directly in the feature matrix $X$, allowing the model to fit on arbitrary identifier hashes.
2. Categorical encoding and missing value handling were performed globally before train/test splitting.

I resolved these by explicitly excluding `customerID` from the feature matrix and encapsulating `OneHotEncoder` and `StandardScaler` inside a Scikit-Learn `ColumnTransformer` within a `Pipeline`. This guarantees that feature transformers are fitted strictly on the training fold during cross-validation and evaluation."*

---

### Q3: Why is feature scaling essential for Logistic Regression with L2 Regularization?
**Answer:**  
*"Logistic Regression optimizes a loss function with an L2 regularization penalty $\lambda \sum \beta_j^2$. If features are on different scales (e.g., `TotalCharges` in thousands vs. `tenure` in tens), the optimization algorithm penalizes coefficients of larger-magnitude features disproportionately. Standardizing continuous features using `StandardScaler` $(\mu=0, \sigma=1)$ ensures all features are penalized equally and allows direct comparison of coefficient magnitudes."*

---

### Q4: Why did you replace LabelEncoder with OneHotEncoder for categorical features?
**Answer:**  
*"LabelEncoder assigns sequential integers $(0, 1, 2, 3)$ to categories. For nominal variables like `PaymentMethod` ('Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'), this imposes an artificial ordinal relationship implying that Credit card is mathematically 'greater than' Electronic check. `OneHotEncoder(drop='first')` creates separate binary dummy indicators with a clear reference category, preventing false linear ordering and multicollinearity."*

---

### Q5: How do you interpret an Odds Ratio of 2.33 for Monthly Charges?
**Answer:**  
*"In Logistic Regression, the log-odds equation is $\ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + \dots$. When a numerical feature $X_1$ is standardized, the Odds Ratio is calculated as $\text{OR} = e^{\beta_1}$. An Odds Ratio of 2.33 means that for every 1 standard deviation increase in monthly charges (approx. $29.40), the odds of a customer churning increase by a factor of 2.334 (a 133.4% increase in churn odds), holding all other variables constant."*

---

### Q6: How do you interpret an Odds Ratio of 0.327 for One-Year Contracts?
**Answer:**  
*"With `OneHotEncoder(drop='first')`, 'Month-to-month' is the dropped reference baseline. A coefficient of $-1.1175$ for `Contract_One year` yields an Odds Ratio of $e^{-1.1175} = 0.327$. This indicates that customers on a one-year contract have 67.3% lower odds of churning compared to month-to-month customers, demonstrating the strong protective retention effect of annual commitments."*

---

### Q7: What does the ROC-AUC score of 0.8443 tell us about this model?
**Answer:**  
*"ROC-AUC measures the model's ability to discriminate between positive (churned) and negative (retained) classes across all classification thresholds. A score of 0.8443 means that if you randomly select one churned customer and one retained customer, there is an 84.4% probability that the model assigns a higher churn probability to the churned customer than to the retained customer."*

---

### Q8: Accuracy is 78%, but Churn Precision is 56.5% and Recall is 52.0%. Is this good or bad?
**Answer:**  
*"Because the dataset has class imbalance (25.5% churn prevalence), a naive classifier predicting 'Never Churn' would achieve 74.5% accuracy but 0% recall. Our model achieves 78% accuracy while successfully identifying over half of all churners at the default 0.5 threshold with a 0.844 ROC-AUC. In business practice, the 0.5 threshold can be adjusted: if retention outreach is low-cost (e.g., automated email), we lower the threshold to 0.35 to increase recall (catch 75%+ of churners). If retention involves high-cost interventions (e.g., dedicated manager outreach), we raise the threshold to maximize precision."*

---

### Q9: What role does SQL play in this project compared to Python?
**Answer:**  
*"SQL and Python serve complementary roles across the enterprise data stack:
- **SQL (`sql/`)** operates closest to the data warehouse/lakehouse, performing initial ingestion, schema enforcement, data type conversion, table materialization, and dimensional cohort aggregations (e.g., churn rate by contract, tenure lifecycle cohorts, revenue at risk).
- **Python (`src/`)** handles statistical feature engineering, machine learning pipelines, probability estimation, metric evaluation, and diagnostic chart generation."*

---

### Q10: How does the embedded SQLite runner work, and why was it included?
**Answer:**  
*"The `SQLRunner` class in `src/sql_runner.py` initializes an in-memory SQLite database (`:memory:`), loads pandas DataFrames into SQL tables, and executes SQL scripts (`01_schema.sql`, `02_data_cleaning.sql`, `03_analytics_queries.sql`). This ensures full query testing and validation in automated CI/CD pipelines without requiring external database server infrastructure, while keeping the SQL syntax compatible with PostgreSQL and MySQL."*

---

### Q11: How do the Power BI DAX measures ensure analytical accuracy?
**Answer:**  
*"Rather than relying on implicit Power BI field summations, the project defines explicit DAX measures (e.g., `Total Customers = COUNTROWS('churn_analysis')`, `Churn Rate % = DIVIDE([Churned Customers], [Total Customers], 0)`). Using `DIVIDE` prevents divide-by-zero errors when slicers filter down to empty subsets, and `CALCULATE` enables accurate context transitions across contract and payment method dimensions."*

---

### Q12: How did you ensure reproducibility across different machines?
**Answer:**  
*"Reproducibility is enforced at three levels:
1. **Deterministic Seeds:** `random_state=42` is fixed across train/test splits and Logistic Regression solvers in `config/config.yaml`.
2. **Environment Specification:** Exact library versions are pinned in `requirements.txt` and `pyproject.toml`.
3. **Pipeline Orchestration:** A single CLI script `run_pipeline.py` executes all stages sequentially from raw ingestion to model artifact generation."*

---

### Q13: What is the Brier Score and why did you track it?
**Answer:**  
*"The Brier Score ($0.1340$) measures the mean squared difference between predicted probabilities and actual binary outcomes. While ROC-AUC measures ranking ability, the Brier Score measures probability calibration. A low Brier score confirms that when the model outputs a 70% churn probability, approximately 70% of those customers actually churn, which is critical when setting financial risk thresholds."*

---

### Q14: How would you deploy this pipeline into a production cloud environment?
**Answer:**  
*"In a production environment:
1. **Data Ingestion:** Ingest raw events via Snowflake / BigQuery / Databricks with dbt managing the SQL transformations.
2. **Model Training & Registry:** Package `src/` into a Docker container orchestrated via Airflow / Prefect, logging experiments and registering artifacts in MLflow.
3. **Batch Inference & Serving:** Run daily batch scoring jobs writing churn probabilities to the warehouse, which feeds automated CRM alerts (Salesforce / HubSpot) and refreshes Power BI datasets via Power BI Service Gateway."*

---

### Q15: What are the primary limitations of the current implementation?
**Answer:**  
*"The primary limitations are:
1. **Dataset Size:** 400 records is sufficient for a modular portfolio demonstration, but a production pipeline would train on hundreds of thousands of records.
2. **Feature Breadth:** Additional predictive signals such as customer support ticket frequency, network QoS metrics, usage trends (minutes/data over time), and promotional discount history would further boost recall.
3. **Single Model Architecture:** Exploring non-linear models (XGBoost, CatBoost) with SHAP tree explainers would provide a valuable comparative benchmark."*

---

## 📌 Strategic Portfolio Checklist
- [x] Clear business narrative and ROI framing.
- [x] Zero data leakage with modular Scikit-Learn pipelines.
- [x] Standardized numerical scaling and one-hot nominal encoding.
- [x] Interpretable Log-Odds and Odds Ratios ($e^\beta$).
- [x] Portable SQL queries and embedded SQLite execution runner.
- [x] Verified Power BI data model and DAX measure documentation.
- [x] 100% automated test coverage across unit and integration tests.
- [x] Zero personal branding or machine-specific absolute paths.
