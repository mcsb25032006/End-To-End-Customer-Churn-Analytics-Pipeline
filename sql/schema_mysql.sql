-- ==============================================================================
-- schema_mysql.sql: MySQL / MariaDB Reference Script
-- ==============================================================================

CREATE DATABASE IF NOT EXISTS telco_churn;
USE telco_churn;

DROP TABLE IF EXISTS telco_customers;

CREATE TABLE telco_customers (
    customerID VARCHAR(20) PRIMARY KEY,
    tenure INT NOT NULL,
    MonthlyCharges DECIMAL(10, 2) NOT NULL,
    TotalCharges DECIMAL(10, 2) NULL,
    Contract VARCHAR(30) NOT NULL,
    PaymentMethod VARCHAR(50) NOT NULL,
    churn_flag TINYINT(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Clean TotalCharges
DELETE FROM telco_customers WHERE TotalCharges IS NULL;

-- Create analytics table
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
