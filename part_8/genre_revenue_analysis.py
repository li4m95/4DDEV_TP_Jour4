import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

spark = SparkSession.builder.appName("GenreRevenueAnalysis").getOrCreate()

# ── Part 8 – Top 5 most profitable genres (PySpark) ─────────────────────────

invoice_lines = spark.read.option("header", True).csv("DataSet_csv/InvoiceLine.csv")
tracks        = spark.read.option("header", True).csv("DataSet_csv/Track.csv")
genres        = spark.read.option("header", True).csv("DataSet_csv/Genre.csv")

invoice_lines = (
    invoice_lines
    .withColumn("UnitPrice", F.col("UnitPrice").cast("double"))
    .withColumn("Quantity",  F.col("Quantity").cast("int"))
)

top5_genres = (
    invoice_lines
    .withColumn("revenue", F.col("UnitPrice") * F.col("Quantity"))
    .groupBy("TrackId")
    .agg(F.sum("revenue").alias("track_revenue"))
    .join(tracks.select("TrackId", "GenreId"), "TrackId")
    .groupBy("GenreId")
    .agg(F.round(F.sum("track_revenue"), 2).alias("total_revenue"))
    .join(genres.select("GenreId", F.col("Name").alias("GenreName")), "GenreId")
    .select("GenreName", "total_revenue")
    .orderBy(F.desc("total_revenue"))
    .limit(5)
)

print("=== Top 5 most profitable genres (PySpark) ===")
top5_genres.show(truncate=False)

spark.stop()

# ── Part 8 – Top 5 most profitable genres (PyFlink SQL) ─────────────────────

from pyflink.table import EnvironmentSettings, TableEnvironment

env_settings = EnvironmentSettings.in_batch_mode()
table_env = TableEnvironment.create(env_settings)

pdf_invoice_lines = pd.read_csv("DataSet_csv/InvoiceLine.csv")
pdf_tracks        = pd.read_csv("DataSet_csv/Track.csv")
pdf_genres        = pd.read_csv("DataSet_csv/Genre.csv")

table_env.create_temporary_view("invoice_lines", table_env.from_pandas(pdf_invoice_lines))
table_env.create_temporary_view("tracks",        table_env.from_pandas(pdf_tracks))
table_env.create_temporary_view("genres",        table_env.from_pandas(pdf_genres))

print("\n=== Top 5 most profitable genres (PyFlink SQL) ===")
table_env.execute_sql("""
    SELECT g.Name AS GenreName,
           ROUND(SUM(CAST(il.UnitPrice AS DOUBLE) * CAST(il.Quantity AS INT)), 2) AS total_revenue
    FROM invoice_lines il
    JOIN tracks t ON il.TrackId = t.TrackId
    JOIN genres g ON t.GenreId  = g.GenreId
    GROUP BY g.GenreId, g.Name
    ORDER BY total_revenue DESC
    LIMIT 5
""").print()
