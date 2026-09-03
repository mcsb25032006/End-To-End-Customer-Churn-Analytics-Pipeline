-- ==============================================================================
-- 03_analytics_queries.sql: Business Analytics and Churn Risk Insights
-- ==============================================================================

-- 1. Executive Summary KPIs: Total Base, Churned Base, Overall Churn Rate, and Revenue
SELECT 
    COUNT(*) AS total_customers,
    SUM(churn_flag) AS churned_customers,
    COUNT(*) - SUM(churn_flag) AS retained_customers,
    ROUND(AVG(churn_flag) * 100.0, 2) AS churn_rate_pct,
    ROUND(SUM(TotalCharges), 2) AS total_historical_revenue,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_bill
FROM churn_analysis;


-- 2. Churn Dynamics by Contract Commitment
SELECT 
    Contract,
    COUNT(*) AS customer_count,
    SUM(churn_flag) AS churned_count,
    ROUND(AVG(churn_flag) * 100.0, 2) AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_billings,
    ROUND(AVG(tenure), 1) AS avg_tenure_months
FROM churn_analysis
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- 3. Churn Propensity by Payment Channel
SELECT 
    PaymentMethod,
    COUNT(*) AS customer_count,
    SUM(churn_flag) AS churned_count,
    ROUND(AVG(churn_flag) * 100.0, 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge
FROM churn_analysis
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;


-- 4. Churn Risk by Monthly Charge Tiers
SELECT 
    CASE 
        WHEN MonthlyCharges < 40 THEN 'Low (< $40)'
        WHEN MonthlyCharges BETWEEN 40 AND 80 THEN 'Medium ($40 - $80)'
        ELSE 'High (> $80)'
    END AS charge_tier,
    COUNT(*) AS total_customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(AVG(churn_flag) * 100.0, 2) AS churn_rate_pct,
    ROUND(AVG(tenure), 1) AS avg_tenure
FROM churn_analysis
GROUP BY charge_tier
ORDER BY churn_rate_pct DESC;


-- 5. Tenure Lifecycle Cohort Analysis
SELECT 
    CASE 
        WHEN tenure <= 12 THEN '01. First-Year (0-12m)'
        WHEN tenure BETWEEN 13 AND 36 THEN '02. Growth (13-36m)'
        WHEN tenure BETWEEN 37 AND 60 THEN '03. Mature (37-60m)'
        ELSE '04. Loyal (60m+)'
    END AS tenure_cohort,
    COUNT(*) AS total_customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(AVG(churn_flag) * 100.0, 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_bill
FROM churn_analysis
GROUP BY tenure_cohort
ORDER BY tenure_cohort;


-- 6. High-Value Customer Churn Risk (Top 25% Monthly Spend)
WITH HighSpendThreshold AS (
    SELECT AVG(MonthlyCharges) AS avg_spend FROM churn_analysis
)
SELECT 
    c.customerID,
    c.Contract,
    c.PaymentMethod,
    c.tenure,
    c.MonthlyCharges,
    c.TotalCharges,
    c.churn_flag
FROM churn_analysis c, HighSpendThreshold t
WHERE c.MonthlyCharges > t.avg_spend AND c.churn_flag = 1
ORDER BY c.MonthlyCharges DESC
LIMIT 15;


-- 7. Monthly Revenue Loss / Impact by Contract & Payment Method
SELECT 
    Contract,
    PaymentMethod,
    COUNT(*) AS lost_customers,
    ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue_lost,
    ROUND(AVG(tenure), 1) AS avg_tenure_lost
FROM churn_analysis
WHERE churn_flag = 1
GROUP BY Contract, PaymentMethod
ORDER BY monthly_revenue_lost DESC;
