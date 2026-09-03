-- ==============================================================================
-- 01_schema.sql: Raw Customer Ingestion Schema
-- Compatible with SQLite, PostgreSQL, and MySQL
-- ==============================================================================

DROP TABLE IF EXISTS telco_customers;

CREATE TABLE telco_customers (
    customerID VARCHAR(20) PRIMARY KEY,
    tenure INT NOT NULL,
    MonthlyCharges DECIMAL(10, 2) NOT NULL,
    TotalCharges DECIMAL(10, 2),
    Contract VARCHAR(30) NOT NULL,
    PaymentMethod VARCHAR(50) NOT NULL,
    churn_flag INT NOT NULL
);
