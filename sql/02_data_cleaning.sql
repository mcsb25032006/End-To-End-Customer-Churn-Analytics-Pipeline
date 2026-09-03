-- ==============================================================================
-- 02_data_cleaning.sql: Data Cleaning and Standardization Transformations
-- ==============================================================================

-- 1. Remove rows with null or invalid TotalCharges
DELETE FROM telco_customers
WHERE TotalCharges IS NULL;

-- 2. Verify all records have valid numeric ranges
SELECT 
    COUNT(*) AS total_valid_rows,
    MIN(tenure) AS min_tenure,
    MAX(tenure) AS max_tenure,
    MIN(MonthlyCharges) AS min_monthly_charges,
    MAX(MonthlyCharges) AS max_monthly_charges,
    MIN(TotalCharges) AS min_total_charges,
    MAX(TotalCharges) AS max_total_charges
FROM telco_customers;

-- 3. Create Clean Analytics Table / View
DROP TABLE IF EXISTS churn_analysis;

CREATE TABLE churn_analysis AS
SELECT
    customerID,
    tenure,
    MonthlyCharges,
    TotalCharges,
    Contract,
    PaymentMethod,
    churn_flag
FROM telco_customers;
