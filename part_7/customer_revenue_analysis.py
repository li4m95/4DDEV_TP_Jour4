import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

spark = SparkSession.builder.appName("CustomerRevenueAnalysis").getOrCreate()

# ── Part 7 – Top 10 most profitable customers (PySpark) ─────────────────────

customers = spark.read.option("header", True).csv("DataSet_csv/Customer.csv")
invoices  = spark.read.option("header", True).csv("DataSet_csv/Invoice.csv")
invoices  = invoices.withColumn("Total", F.col("Total").cast("double"))

top10_customers = (
    invoices
    .groupBy("CustomerId")
    .agg(F.round(F.sum("Total"), 2).alias("total_revenue"))
    .join(customers.select("CustomerId", "FirstName", "LastName"), "CustomerId")
    .select(
        F.concat(F.col("FirstName"), F.lit(" "), F.col("LastName")).alias("CustomerName"),
        "total_revenue"
    )
    .orderBy(F.desc("total_revenue"))
    .limit(10)
)

print("=== Top 10 most profitable customers (PySpark) ===")
top10_customers.show(truncate=False)

spark.stop()

# ── Part 7 – Top 10 most profitable customers (PyFlink SQL) ─────────────────

from pyflink.table import EnvironmentSettings, TableEnvironment

env_settings = EnvironmentSettings.in_batch_mode()
table_env = TableEnvironment.create(env_settings)

pdf_customers = pd.read_csv("DataSet_csv/Customer.csv")
pdf_invoices  = pd.read_csv("DataSet_csv/Invoice.csv")

table_env.create_temporary_view("customers", table_env.from_pandas(pdf_customers))
table_env.create_temporary_view("invoices",  table_env.from_pandas(pdf_invoices))

print("\n=== Top 10 most profitable customers (PyFlink SQL) ===")
table_env.execute_sql("""
    SELECT CONCAT(c.FirstName, ' ', c.LastName) AS CustomerName,
           ROUND(SUM(CAST(i.Total AS DOUBLE)), 2) AS total_revenue
    FROM invoices i
    JOIN customers c ON i.CustomerId = c.CustomerId
    GROUP BY c.CustomerId, c.FirstName, c.LastName
    ORDER BY total_revenue DESC
    LIMIT 10
""").print()
