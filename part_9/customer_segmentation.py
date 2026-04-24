import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

spark = SparkSession.builder.appName("CustomerSegmentation").getOrCreate()

# ── Part 9 – Customer Segmentation (PySpark) ────────────────────────────────
#
# Segments based on total revenue per customer:
#   Low    : total_revenue < 20
#   Medium : 20 <= total_revenue < 40
#   High   : total_revenue >= 40

invoices      = spark.read.option("header", True).csv("DataSet_csv/Invoice.csv")
invoice_lines = spark.read.option("header", True).csv("DataSet_csv/InvoiceLine.csv")
tracks        = spark.read.option("header", True).csv("DataSet_csv/Track.csv")
genres        = spark.read.option("header", True).csv("DataSet_csv/Genre.csv")

invoices = invoices.withColumn("Total", F.col("Total").cast("double"))

# Step 1 – revenue + purchase count + segment per customer
customer_segments = (
    invoices
    .groupBy("CustomerId")
    .agg(
        F.round(F.sum("Total"), 2).alias("total_revenue"),
        F.count("InvoiceId").alias("purchase_count")
    )
    .withColumn("segment",
        F.when(F.col("total_revenue") < 20, "Low")
         .when(F.col("total_revenue") < 40, "Medium")
         .otherwise("High")
    )
)

# Step 2 – average purchases per segment
avg_purchases = (
    customer_segments
    .groupBy("segment")
    .agg(F.round(F.avg("purchase_count"), 2).alias("avg_purchases"))
)


# Step 3 – most popular genre per segment
genre_counts = (
    invoices.select("InvoiceId", "CustomerId")
    .join(customer_segments.select("CustomerId", "segment"), "CustomerId")
    .join(invoice_lines.select("InvoiceId", "TrackId"), "InvoiceId")
    .join(tracks.select("TrackId", "GenreId"), "TrackId")
    .join(genres.select("GenreId", F.col("Name").alias("GenreName")), "GenreId")
    .groupBy("segment", "GenreName")
    .agg(F.count("*").alias("genre_count"))
)

window = Window.partitionBy("segment").orderBy(F.desc("genre_count"))
top_genre = (
    genre_counts
    .withColumn("rank", F.rank().over(window))
    .filter(F.col("rank") == 1)
    .select("segment", "GenreName")
)

# Step 4 – final report
result = (
    avg_purchases
    .join(top_genre, "segment")
    .select("segment", "avg_purchases", F.col("GenreName").alias("top_genre"))
    .orderBy("segment")
)

print("=== Customer Segmentation (PySpark) ===")
result.show(truncate=False)

spark.stop()

# ── Part 9 – Customer Segmentation (PyFlink SQL) ─────────────────────────────

from pyflink.table import EnvironmentSettings, TableEnvironment

env_settings = EnvironmentSettings.in_batch_mode()
table_env = TableEnvironment.create(env_settings)
table_env.get_config().set("taskmanager.memory.managed.size", "512mb")
table_env.get_config().set("parallelism.default", "1")

pdf_invoices      = pd.read_csv("DataSet_csv/Invoice.csv")
pdf_invoice_lines = pd.read_csv("DataSet_csv/InvoiceLine.csv")
pdf_tracks        = pd.read_csv("DataSet_csv/Track.csv")
pdf_genres        = pd.read_csv("DataSet_csv/Genre.csv")

table_env.create_temporary_view("invoices",      table_env.from_pandas(pdf_invoices))
table_env.create_temporary_view("invoice_lines", table_env.from_pandas(pdf_invoice_lines))
table_env.create_temporary_view("tracks",        table_env.from_pandas(pdf_tracks))
table_env.create_temporary_view("genres",        table_env.from_pandas(pdf_genres))

# Step 1 – revenue + purchase count + segment per customer
table_env.execute_sql("""
    CREATE TEMPORARY VIEW customer_segments AS
    SELECT
        CustomerId,
        ROUND(SUM(CAST(Total AS DOUBLE)), 2) AS total_revenue,
        COUNT(InvoiceId)                      AS purchase_count,
        CASE
            WHEN SUM(CAST(Total AS DOUBLE)) < 20 THEN 'Low'
            WHEN SUM(CAST(Total AS DOUBLE)) < 40 THEN 'Medium'
            ELSE 'High'
        END AS segment
    FROM invoices
    GROUP BY CustomerId
""")

# Step 2 – average purchases per segment
table_env.execute_sql("""
    CREATE TEMPORARY VIEW avg_purchases AS
    SELECT segment,
           ROUND(AVG(CAST(purchase_count AS DOUBLE)), 2) AS avg_purchases
    FROM customer_segments
    GROUP BY segment
""")

# Step 3 – genre count per segment
table_env.execute_sql("""
    CREATE TEMPORARY VIEW genre_counts AS
    SELECT cs.segment,
           g.Name  AS GenreName,
           COUNT(*) AS genre_count
    FROM invoices i
    JOIN customer_segments cs ON i.CustomerId  = cs.CustomerId
    JOIN invoice_lines il     ON i.InvoiceId   = il.InvoiceId
    JOIN tracks t             ON il.TrackId    = t.TrackId
    JOIN genres g             ON t.GenreId     = g.GenreId
    GROUP BY cs.segment, g.Name
""")

# Step 4 – rank genres within each segment, keep only rank 1
table_env.execute_sql("""
    CREATE TEMPORARY VIEW top_genre AS
    SELECT segment, GenreName
    FROM (
        SELECT segment, GenreName,
               ROW_NUMBER() OVER (PARTITION BY segment ORDER BY genre_count DESC) AS rn
        FROM genre_counts
    ) ranked
    WHERE rn = 1
""")

print("\n=== Customer Segmentation (PyFlink SQL) ===")
table_env.execute_sql("""
    SELECT ap.segment,
           ap.avg_purchases,
           tg.GenreName AS top_genre
    FROM avg_purchases ap
    JOIN top_genre tg ON ap.segment = tg.segment
    ORDER BY ap.segment
""").print()
