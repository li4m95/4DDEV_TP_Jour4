import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

spark = SparkSession.builder.appName("AgeGroupAnalysis").getOrCreate()

# ── Part 12 – Purchase Analysis by Age Group (PySpark) ───────────────────────
#
# Age is not in the Chinook dataset — we simulate it with a fixed random seed
# so results are reproducible across runs.

pdf_customers = pd.read_csv("DataSet_csv/Customer.csv")

rng = np.random.default_rng(seed=42)
pdf_customers["Age"] = rng.integers(18, 71, size=len(pdf_customers))

pdf_customers["AgeGroup"] = pd.cut(
    pdf_customers["Age"],
    bins=[17, 25, 35, 45, 55, 70],
    labels=["18-25", "26-35", "36-45", "46-55", "56-70"]
)

customers = spark.createDataFrame(pdf_customers[["CustomerId", "Age", "AgeGroup"]].astype(str))

invoices = (
    spark.read.option("header", True).csv("DataSet_csv/Invoice.csv")
    .withColumn("Total", F.col("Total").cast("double"))
)

age_analysis = (
    invoices
    .join(customers, "CustomerId")
    .groupBy("AgeGroup")
    .agg(
        F.count("InvoiceId").alias("purchase_count"),
        F.round(F.sum("Total"), 2).alias("total_revenue")
    )
    .orderBy("AgeGroup")
)

print("=== Purchase Analysis by Age Group (PySpark) ===")
age_analysis.show(truncate=False)

spark.stop()

# ── Part 12 – Purchase Analysis by Age Group (PyFlink SQL) ───────────────────

from pyflink.table import EnvironmentSettings, TableEnvironment

env_settings = EnvironmentSettings.in_batch_mode()
table_env = TableEnvironment.create(env_settings)

pdf_invoices = pd.read_csv("DataSet_csv/Invoice.csv")
pdf_invoices["Total"] = pd.to_numeric(pdf_invoices["Total"])

# Reuse the same simulated ages (same seed = same ages)
pdf_customers_with_age = pdf_customers[["CustomerId", "AgeGroup"]].copy()
pdf_customers_with_age["AgeGroup"] = pdf_customers_with_age["AgeGroup"].astype(str)
pdf_customers_with_age["CustomerId"] = pdf_customers_with_age["CustomerId"].astype(str)
pdf_invoices["CustomerId"] = pdf_invoices["CustomerId"].astype(str)

table_env.create_temporary_view("customers_age", table_env.from_pandas(pdf_customers_with_age))
table_env.create_temporary_view("invoices",      table_env.from_pandas(pdf_invoices))

print("\n=== Purchase Analysis by Age Group (PyFlink SQL) ===")
table_env.execute_sql("""
    SELECT c.AgeGroup,
           COUNT(i.InvoiceId)               AS purchase_count,
           ROUND(SUM(CAST(i.Total AS DOUBLE)), 2) AS total_revenue
    FROM invoices i
    JOIN customers_age c ON i.CustomerId = c.CustomerId
    GROUP BY c.AgeGroup
    ORDER BY c.AgeGroup
""").print()
