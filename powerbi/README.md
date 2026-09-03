# 📊 Power BI Customer Churn Analytics Dashboard

## 📌 Dashboard Overview
The **Customer Churn Analytics Dashboard** (`churn_dashboard.pbix`) provides interactive business intelligence and executive KPI tracking for retention teams, revenue operations, and executive leadership.

![Power BI Dashboard Preview](churn_dashboard.png)

---

## 🎯 Key Performance Indicators (KPIs)
The dashboard tracks four top-level summary cards:
1. **Total Customers**: Total subscriber base analyzed (`400`).
2. **Churn Rate (%)**: Proportion of customers lost (`25.50%`).
3. **Total Revenue**: Cumulative revenue generated (`$892.31K`).
4. **Average Monthly Charge**: Average customer billing rate (`$66.14`).

---

## 📈 Visual Breakdowns & Analytical Views

| Visual Container | Visual Type | Dimension / Metric | Business Purpose |
| :--- | :--- | :--- | :--- |
| **KPI Card 1** | `cardVisual` | `[Total Customers]` | Measures customer volume |
| **KPI Card 2** | `cardVisual` | `[Churn Rate %]` | Core retention KPI |
| **KPI Card 3** | `cardVisual` | `[Total Revenue]` | Revenue scale and historical value |
| **KPI Card 4** | `cardVisual` | `[Avg Monthly Charge]` | Pricing benchmark |
| **Contract Breakdown** | `donutChart` | `Contract` by `[Churn Rate %]` | Highlights extreme churn in Month-to-Month contracts (42.8%) vs One-Year (4.9%) and Two-Year (3.1%) |
| **Payment Channel** | `clusteredBarChart` | `PaymentMethod` by `[Churn Rate %]` | Identifies Electronic Check users as highest churn propensity (41.5%) vs Credit Card (12.2%) |
| **Tenure vs Charges** | `scatterChart` | X: `tenure`, Y: `MonthlyCharges`, Legend: `[Churn Label]` | Reveals churn clustering among short-tenure (<12m), high monthly charge (>$70) users |
| **Contract Slicer** | `slicer` | `Contract` | Dynamic filtering across contract tiers |
| **Payment Slicer** | `slicer` | `PaymentMethod` | Dynamic filtering across payment methods |

---

## 🧮 DAX Measure Reference
All DAX measures are documented in [`dax_measures.dax`](dax_measures.dax):

```dax
Total Customers = COUNTROWS('churn_analysis')

Churned Customers = 
CALCULATE(
    COUNTROWS('churn_analysis'),
    'churn_analysis'[churn_flag] = 1
)

Churn Rate % = 
DIVIDE(
    [Churned Customers],
    [Total Customers],
    0
)

Total Revenue = SUM('churn_analysis'[TotalCharges])

Avg Monthly Charge = AVERAGE('churn_analysis'[MonthlyCharges])
```

---

## 🛠 Manual Configuration & Refresh Guide
If opening or reconnecting `churn_dashboard.pbix` in Power BI Desktop:
1. Open **Power BI Desktop**.
2. File → Open Report → select `powerbi/churn_dashboard.pbix`.
3. In the Home ribbon, click **Transform Data** → **Data Source Settings**.
4. Select `churn_analysis.csv` and point the file path to:
   `data/processed/churn_analysis.csv`
5. Click **Close & Apply**.
6. Verify all visual cards and charts match the expected figures (400 customers, 25.5% churn, $892.31K revenue).
