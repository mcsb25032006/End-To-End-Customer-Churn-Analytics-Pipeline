# 📊 End To End Customer Churn Analytics Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![SQL](https://img.shields.io/badge/SQL-SQLite%20%7C%20MySQL%20%7C%20Postgres-lightgrey.svg)](https://www.sqlite.org/)
[![Power BI](https://img.shields.io/badge/Power_BI-Dashboard-yellow.svg)](https://powerbi.microsoft.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, modular, and reproducible customer churn analytics and predictive modeling pipeline. This project demonstrates a complete analytics lifecycle: from raw data validation, SQL transformation, and exploratory data analysis to feature engineering, Logistic Regression modeling, model interpretability with Odds Ratios, and executive Power BI KPI reporting.

---

## 📌 Table of Contents
- [Business Problem](#-business-problem)
- [Project Architecture & Workflow](#-project-architecture--workflow)
- [Repository Structure](#-repository-structure)
- [Dataset Overview](#-dataset-overview)
- [Technology Stack](#-technology-stack)
- [Installation & Quickstart](#-installation--quickstart)
- [Pipeline Execution](#-pipeline-execution)
- [SQL Ingestion & Business Queries](#-sql-ingestion--business-queries)
- [Machine Learning & Methodology](#-machine-learning--methodology)
- [Model Evaluation & Findings](#-model-evaluation--findings)
- [Power BI Executive Dashboard](#-power-bi-executive-dashboard)
- [Key Business Insights & Recommendations](#-key-business-insights--recommendations)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Limitations & Future Roadmap](#-limitations--future-roadmap)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Business Problem
Customer churn (attrition) directly erodes recurring subscription revenue and inflates customer acquisition costs. In subscription telecommunications businesses:
- Acquiring a new customer is 5–7× more expensive than retaining an existing one.
- Unidentified churn drivers lead to reactive, high-cost retention campaigns.
- Predicting at-risk customers early enables proactive retention incentives, personalized contract upgrades, and payment channel optimization.

### Core Objectives
1. **Identify Critical Churn Drivers**: Quantify the impact of contract type, monthly charges, payment methods, and customer tenure.
2. **Deliver Accurate Churn Probabilities**: Build a calibrated, leakage-free classification model with interpretable feature weights.
3. **Empower Executive Decision-Making**: Provide dynamic Power BI reporting and automated SQL analytics tracking recurring revenue at risk.

---

## 🔄 Project Architecture & Workflow

```
Raw Customer Data
       │
       ▼
[ Data Validation & Cleansing ] ───▶ Schema verification, type casting, null handling
       │
       ▼
[ Embedded SQL Engine ] ────────────▶ Aggregation, tenure cohorting, revenue-at-risk queries
       │
       ▼
[ Exploratory Data Analysis ] ──────▶ Class balance, contract dynamics, charge distributions
       │
       ▼
[ Preprocessing & Scaling ] ────────▶ Dropping IDs, One-Hot Encoding, StandardScaler
       │
       ▼
[ Stratified Train/Test Split ] ────▶ 75% Train / 25% Test (preserving class balance)
       │
       ▼
[ Logistic Regression Modeling ] ───▶ Scikit-Learn Pipeline training & L2 regularization
       │
       ▼
[ Comprehensive Evaluation ] ───────▶ Confusion Matrix, ROC-AUC (0.844), Precision/Recall, Brier Score
       │
       ▼
[ Model Interpretability ] ─────────▶ Log-Odds & Odds Ratios quantification
       │
       ▼
[ Power BI Executive Dashboard ] ───▶ Interactive KPI tracking & visual decision support
```

---

## 📂 Repository Structure

```
End-To-End-Customer-Churn-Analytics-Pipeline/
├── README.md                           # Master project documentation
├── PROJECT_NOTES.md                    # In-depth technical interview notes & architecture guide
├── requirements.txt                    # Pinned Python dependencies
├── pyproject.toml                      # Package build and test configuration
├── run_pipeline.py                     # CLI entrypoint for complete pipeline execution
├── config/
│   └── config.yaml                     # Centralized pipeline configuration & parameters
├── data/
│   ├── raw/
│   │   └── churn_analysis.csv          # Raw customer dataset representation
│   └── processed/
│       └── churn_analysis.csv          # Cleaned, standardized analytics dataset
├── src/
│   ├── __init__.py                     # Package initializer
│   ├── config_loader.py                # Configuration parsing and validation
│   ├── data_loader.py                  # Ingestion, validation, missing value handling
│   ├── preprocessing.py                # Feature transformation, encoding, scaling
│   ├── model.py                        # Model construction, fitting, serialization
│   ├── evaluate.py                     # Metric calculation, confusion matrix, ROC curve
│   ├── eda.py                          # Automated EDA visualization generator
│   └── sql_runner.py                   # Embedded SQLite runner for query verification
├── sql/
│   ├── 01_schema.sql                   # Cross-compatible DDL schema
│   ├── 02_data_cleaning.sql            # Data cleaning and table transformation
│   ├── 03_analytics_queries.sql        # Advanced business analytics and cohort queries
│   └── schema_mysql.sql                # MySQL / MariaDB reference schema
├── notebooks/
│   └── customer_churn_analysis.ipynb   # Standalone, interactive Jupyter walkthrough
├── powerbi/
│   ├── churn_dashboard.pbix            # Interactive Power BI report file
│   ├── churn_dashboard.png             # Dashboard preview visual
│   ├── dax_measures.dax                # Documented DAX measures
│   └── README.md                       # Power BI data model & setup guide
├── models/
│   ├── churn_pipeline.joblib           # Serialized Scikit-Learn model pipeline
│   └── metrics.json                    # Exported evaluation metrics
├── reports/
│   └── figures/
│       ├── churn_distribution.png      # Class balance visualization
│       ├── churn_by_contract.png       # Contract attrition breakdown
│       ├── churn_by_payment_method.png # Payment channel churn risk
│       ├── monthly_charges_vs_churn.png# Billing distribution boxplot & density
│       ├── confusion_matrix.png        # Test set confusion matrix heatmap
│       ├── roc_curve.png               # ROC-AUC performance curve
│       └── feature_importance.png      # Feature coefficients and log-odds
└── tests/
    ├── __init__.py
    ├── test_data_loader.py             # Ingestion & schema validation tests
    ├── test_preprocessing.py          # Encoding & scaling tests
    ├── test_model.py                   # Fitting, serialization & evaluation tests
    ├── test_sql.py                     # SQL execution & analytics tests
    └── test_pipeline.py                # End-to-end integration test
```

---

## 🗂 Dataset Overview

The dataset contains subscriber demographics, billing details, contract types, payment methods, and historical churn outcomes:

| Column | Type | Description |
| :--- | :--- | :--- |
| `customerID` | String | Unique customer identifier (excluded from model features to prevent leakage) |
| `tenure` | Integer | Number of months the customer has stayed with the company (1 – 72) |
| `MonthlyCharges` | Float | Current monthly recurring bill amount in USD ($18.95 – $115.10) |
| `TotalCharges` | Float | Cumulative total charges billed to date ($19.15 – $8,129.30) |
| `Contract` | Categorical | Contract term (`Month-to-month`, `One year`, `Two year`) |
| `PaymentMethod` | Categorical | Billing method (`Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)`) |
| `churn_flag` | Integer | Binary target variable (`1` = Churned, `0` = Retained) |

---

## 🧰 Technology Stack

- **Python 3.9+**: Core application, data manipulation, and pipeline orchestration
- **Pandas & NumPy**: High-performance data structures and vector operations
- **Scikit-Learn**: Preprocessing pipelines, ColumnTransformers, and Logistic Regression
- **SQLite / SQL**: Embedded relational query execution and analytical reporting
- **Matplotlib & Seaborn**: Statistical charting and report figure generation
- **Power BI Desktop**: Interactive executive dashboards, DAX measures, and slicers
- **PyYAML & Joblib**: Configuration management and model artifact serialization
- **Pytest**: Automated unit and integration testing suite

---

## ⚡ Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/End-To-End-Customer-Churn-Analytics-Pipeline.git
cd End-To-End-Customer-Churn-Analytics-Pipeline
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Pipeline Execution

Run the complete pipeline from data ingestion to model serialization and reporting in a single command:

```bash
python run_pipeline.py
```

### Custom Configuration
You can customize hyperparameters, split ratios, or file paths via `config/config.yaml` or pass a custom configuration file:
```bash
python run_pipeline.py --config config/config.yaml
```

---

## 🗄 SQL Ingestion & Business Queries

The project provides modular, cross-compatible SQL scripts in `sql/`:

1. **`sql/01_schema.sql`**: Creates the `telco_customers` table with strict data typing and constraints.
2. **`sql/02_data_cleaning.sql`**: Validates non-null total charges and materializes the clean analytics table `churn_analysis`.
3. **`sql/03_analytics_queries.sql`**: Executes 7 high-impact business queries, including:
   - Executive KPIs (Total Customers, Total Churned, Overall Churn %, Total Revenue)
   - Churn Dynamics by Contract Commitment
   - Churn Propensity by Payment Channel
   - Churn Risk by Monthly Charge Tiers (<$40, $40-$80, >$80)
   - Tenure Lifecycle Cohort Analysis (0-12m, 13-36m, 37-60m, 60m+)
   - Monthly Revenue at Risk from Churned Customers

### Running SQL Queries Locally
The pipeline includes an embedded SQLite runner (`src/sql_runner.py`) requiring zero database installation. For enterprise MySQL/PostgreSQL deployment, see `sql/schema_mysql.sql`.

---

## 🤖 Machine Learning & Methodology

### 1. Data Leakage Prevention
- `customerID` is strictly removed from the feature matrix $X$ to prevent spurious memorization of string identifiers.
- Feature encoders (`OneHotEncoder`) and scalers (`StandardScaler`) are fitted **only on the training fold** inside Scikit-Learn `Pipeline` objects to prevent test-set distribution leakage.

### 2. Categorical & Numerical Preprocessing
- **Categorical Encoding**: `OneHotEncoder(drop='first', handle_unknown='ignore')` is applied to `Contract` and `PaymentMethod`, preventing artificial ordinal assumptions.
- **Feature Scaling**: `StandardScaler` standardizes `tenure`, `MonthlyCharges`, and `TotalCharges` to mean 0 and variance 1, ensuring L2 regularization penalizes all features equitably.

### 3. Model Architecture
- **Algorithm**: `LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)`
- **Stratification**: Preserves the 25.5% churn prevalence across training (300 records) and testing (100 records) sets.

---

## 📈 Model Evaluation & Findings

### Test Set Performance Metrics (Threshold = 0.50)

| Metric | Score | Business Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **78.00%** | Baseline overall classification accuracy |
| **ROC-AUC** | **0.8443** | Strong discriminative ability between churners and retained subscribers |
| **Churn Precision** | **56.52%** | 56.5% of predicted churners are true churners |
| **Churn Recall** | **52.00%** | Captures 52.0% of all churning customers at standard threshold |
| **Churn F1-Score** | **0.5417** | Balanced harmonic mean between precision and recall |
| **Specificity** | **86.67%** | Correctly identifies 86.7% of retained customers |
| **Brier Score** | **0.1340** | Well-calibrated probabilistic output |

### Confusion Matrix (Test Set: N=100)
- **True Negatives (Retained correctly)**: 65
- **False Positives (False Churn Alarm)**: 10
- **False Negatives (Missed Churners)**: 12
- **True Positives (Caught Churners)**: 13

### Feature Importance & Odds Ratios ($e^\beta$)

| Feature | Coefficient ($\beta$) | Odds Ratio ($e^\beta$) | Direction & Business Impact |
| :--- | :--- | :--- | :--- |
| **MonthlyCharges** | `+0.8476` | **2.334** | Higher monthly billing more than doubles the odds of churn |
| **PaymentMethod (Electronic check)** | `+0.5020` | **1.652** | Electronic check users are 65.2% more likely to churn |
| **PaymentMethod (Mailed check)** | `+0.0939` | **1.098** | Slight positive association with churn |
| **PaymentMethod (Credit card auto)** | `-0.2586` | **0.772** | Automatic credit card billing reduces churn odds by 22.8% |
| **TotalCharges** | `-0.3731` | **0.689** | Higher historical spend indicates higher account stickiness |
| **tenure** | `-0.7067` | **0.493** | Each std increase in tenure cuts churn odds by ~50.7% |
| **Contract (Two year)** | `-0.7651` | **0.465** | Two-year contract reduces churn odds by 53.5% |
| **Contract (One year)** | `-1.1175` | **0.327** | One-year contract reduces churn odds by 67.3% |

---

## 📊 Power BI Executive Dashboard

The interactive Power BI dashboard (`powerbi/churn_dashboard.pbix`) provides real-time KPI monitoring:

![Power BI Dashboard](powerbi/churn_dashboard.png)

### Key Dashboard Components
- **Top KPI Cards**: Total Customers (`400`), Churn Rate (`25.50%`), Total Revenue (`$892.31K`), Average Monthly Charges (`$66.14`).
- **Churn by Contract Donut Chart**: Highlights 42.8% churn in Month-to-Month contracts vs. 4.9% in One-Year and 3.1% in Two-Year.
- **Churn by Payment Channel Clustered Bar**: Visualizes 41.5% churn among Electronic Check users vs 12.2% for Credit Card autopay.
- **Tenure vs Monthly Charges Scatter Plot**: Demonstrates severe churn concentration in early-tenure (<12 months), high-spend accounts.
- **Interactive Slicers**: Multi-select filtering by Contract type and Payment channel.

For DAX formula definitions and setup instructions, refer to [`powerbi/README.md`](powerbi/README.md).

---

## 💡 Key Business Insights & Recommendations

1. **Contract Migration Campaigns**:
   - *Insight*: Month-to-Month subscribers churn at **42.8%**, compared to **4.9%** for annual contracts.
   - *Action*: Introduce a 10% annual renewal discount to incentivize contract upgrades, locking in recurring revenue.
2. **Eliminate Electronic Check Friction**:
   - *Insight*: Electronic Check users exhibit a **41.5% churn rate** and an Odds Ratio of **1.652**.
   - *Action*: Offer a one-time $10 bill credit for switching to automated Credit Card or Bank Transfer autopay.
3. **Targeted First-Year Onboarding (<12 Months)**:
   - *Insight*: Over 60% of all churn events occur within the first 12 months.
   - *Action*: Implement proactive check-ins at Day 30, 60, and 90, with loyalty incentives for new sign-ups.

---

## 🧪 Testing & Quality Assurance

Run the comprehensive test suite with `pytest`:

```bash
python -m pytest tests/ -v
```

### Test Coverage Summary
- `test_data_loader.py`: Validates CSV ingestion, schema validation, type casting, missing value handling, and summary statistics.
- `test_preprocessing.py`: Verifies identifier exclusion, ColumnTransformer output shapes, and encoding consistency.
- `test_model.py`: Tests model fitting, probability bounds $[0, 1]$, serialization/deserialization via Joblib, and metric calculations.
- `test_sql.py`: Verifies embedded SQLite table creation, query integrity, and analytics suite aggregations.
- `test_pipeline.py`: Executes the complete end-to-end pipeline and asserts artifact creation.

---

## ⚠️ Limitations & Future Roadmap

- **Sample Size**: The current dataset contains 400 records. While ideal for rapid demonstration and portfolio review, expanding to 7,000+ records will improve statistical power.
- **Class Imbalance & Threshold Tuning**: At the default 0.5 threshold, recall is 52.0%. In high-churn business contexts, lowering the decision threshold to 0.35–0.40 will capture more at-risk accounts at the cost of slight precision reduction.
- **Nonlinear Model Benchmarks**: Future iterations can compare Logistic Regression against tree-based ensembles (Random Forest, XGBoost, LightGBM) with SHAP value explanations.

---

## 🔧 Troubleshooting

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| `FileNotFoundError: config/config.yaml` | Executing outside repository root | Run commands from the project root directory |
| `ModuleNotFoundError: No module named 'src'` | Python path not configured | Run `python run_pipeline.py` or set `PYTHONPATH=.` |
| Power BI visual blank | Data source path mismatch | In Power BI Desktop: Home → Transform Data → Data Source Settings → repoint to `data/processed/churn_analysis.csv` |

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
