import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

spark = SparkSession.builder.appName("RevenueTracksAnalysis").getOrCreate()

# ── Part 4 – Revenue per country (PySpark) ───────────────────────────────────

invoices = spark.read.option("header", True).csv("DataSet_csv/Invoice.csv")
invoices = invoices.withColumn("Total", F.col("Total").cast("double"))

revenue_by_country = (
    invoices
    .groupBy("BillingCountry")
    .agg(F.round(F.sum("Total"), 2).alias("total_revenue"))
    .orderBy(F.desc("total_revenue"))
)

print("=== Revenue per country (PySpark) ===")
revenue_by_country.show()

pdf_revenue = revenue_by_country.toPandas()
pdf_revenue.to_csv("revenue_by_country.csv", index=False)
pdf_revenue.to_parquet("revenue_by_country.parquet", index=False)
print("Saved: revenue_by_country.csv / revenue_by_country.parquet")

# ── Part 5 – Top 10 best-selling tracks (PySpark) ────────────────────────────

invoice_lines = spark.read.option("header", True).csv("DataSet_csv/InvoiceLine.csv")
tracks  = spark.read.option("header", True).csv("DataSet_csv/Track.csv")
albums  = spark.read.option("header", True).csv("DataSet_csv/Album.csv")
artists = spark.read.option("header", True).csv("DataSet_csv/Artist.csv")

invoice_lines = (
    invoice_lines
    .withColumn("UnitPrice", F.col("UnitPrice").cast("double"))
    .withColumn("Quantity",  F.col("Quantity").cast("int"))
)

top10_tracks = (
    invoice_lines
    .withColumn("revenue", F.col("UnitPrice") * F.col("Quantity"))
    .groupBy("TrackId")
    .agg(F.round(F.sum("revenue"), 2).alias("total_revenue"))
    .join(tracks.select("TrackId", "Name", "AlbumId"), "TrackId")
    .join(albums.select("AlbumId", "ArtistId"), "AlbumId")
    .join(artists.select("ArtistId", F.col("Name").alias("ArtistName")), "ArtistId")
    .select("Name", "ArtistName", "total_revenue")
    .orderBy(F.desc("total_revenue"))
    .limit(10)
)

print("=== Top 10 best-selling tracks (PySpark) ===")
top10_tracks.show(truncate=False)

spark.stop()

# ── Part 4 – Revenue per country (PyFlink SQL) ───────────────────────────────

from pyflink.table import EnvironmentSettings, TableEnvironment

env_settings = EnvironmentSettings.in_batch_mode()
table_env = TableEnvironment.create(env_settings)

pdf_invoices = pd.read_csv("DataSet_csv/Invoice.csv")
table_env.create_temporary_view("invoices", table_env.from_pandas(pdf_invoices))

result4 = table_env.execute_sql("""
    SELECT BillingCountry,
           ROUND(SUM(Total), 2) AS total_revenue
    FROM invoices
    GROUP BY BillingCountry
    ORDER BY total_revenue DESC
""")

print("\n=== Revenue per country (PyFlink SQL) ===")
result4.print()

# ── Part 5 – Top 10 best-selling tracks (PyFlink SQL) ────────────────────────

pdf_invoice_lines = pd.read_csv("DataSet_csv/InvoiceLine.csv")
pdf_tracks        = pd.read_csv("DataSet_csv/Track.csv")
pdf_albums        = pd.read_csv("DataSet_csv/Album.csv")
pdf_artists       = pd.read_csv("DataSet_csv/Artist.csv")

table_env.create_temporary_view("invoice_lines", table_env.from_pandas(pdf_invoice_lines))
table_env.create_temporary_view("tracks",        table_env.from_pandas(pdf_tracks))
table_env.create_temporary_view("albums",        table_env.from_pandas(pdf_albums))
table_env.create_temporary_view("artists",       table_env.from_pandas(pdf_artists))

result5 = table_env.execute_sql("""
    SELECT t.Name        AS track_name,
           ar.Name       AS artist_name,
           ROUND(SUM(CAST(il.UnitPrice AS DOUBLE) * CAST(il.Quantity AS INT)), 2) AS total_revenue
    FROM invoice_lines il
    JOIN tracks  t  ON il.TrackId  = t.TrackId
    JOIN albums  al ON t.AlbumId   = al.AlbumId
    JOIN artists ar ON al.ArtistId = ar.ArtistId
    GROUP BY t.Name, ar.Name
    ORDER BY total_revenue DESC
    LIMIT 10
""")

print("\n=== Top 10 best-selling tracks (PyFlink SQL) ===")
result5.print()
