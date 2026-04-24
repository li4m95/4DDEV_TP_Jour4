import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

#  PySpark

spark = SparkSession.builder.appName("CustomerAnalysis").getOrCreate()

df = spark.read.option("header", True).csv("DataSet_csv/Customer.csv")
df = df.fillna("Unknown", subset=["Country"])

top10_spark = (
    df.groupBy("Country")
    .count()
    .orderBy(F.desc("count"))
    .limit(10)
)

print("=== Top 10 countries (PySpark) ===")
top10_spark.show()

top10_spark.toPandas().to_parquet("top10_countries_parquet/top10_countries.parquet", index=False)
print("Saved to top10_countries.parquet")

spark.stop()

# PyFlink SQL API

from pyflink.table import EnvironmentSettings, TableEnvironment

env_settings = EnvironmentSettings.in_batch_mode()
table_env = TableEnvironment.create(env_settings)

pdf = pd.read_csv("DataSet_csv/Customer.csv")
pdf["Country"] = pdf["Country"].fillna("Unknown")

table_env.create_temporary_view("customers", table_env.from_pandas(pdf))

result = table_env.execute_sql("""
    SELECT Country, COUNT(*) AS customer_count
    FROM customers
    GROUP BY Country
    ORDER BY customer_count DESC
    LIMIT 10
""")

print("\n=== Top 10 countries (PyFlink SQL) ===")
result.print()
