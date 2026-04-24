import os
import pandas as pd
import sqlite3
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

spark = SparkSession.builder.appName("PlaylistAnalysis").getOrCreate()

# ── Part 6 – Playlist Report (PySpark) ──────────────────────────────────────

playlists      = spark.read.option("header", True).csv("DataSet_csv/Playlist.csv")
playlist_tracks = spark.read.option("header", True).csv("DataSet_csv/PlaylistTrack.csv")
tracks         = spark.read.option("header", True).csv("DataSet_csv/Track.csv")
invoice_lines  = spark.read.option("header", True).csv("DataSet_csv/InvoiceLine.csv")

invoice_lines = (
    invoice_lines
    .withColumn("UnitPrice", F.col("UnitPrice").cast("double"))
    .withColumn("Quantity",  F.col("Quantity").cast("int"))
)

# Revenue per track
track_revenue = (
    invoice_lines
    .withColumn("revenue", F.col("UnitPrice") * F.col("Quantity"))
    .groupBy("TrackId")
    .agg(F.round(F.sum("revenue"), 2).alias("track_revenue"))
)

report = (
    playlists.select("PlaylistId", F.col("Name").alias("PlaylistName"))
    .join(playlist_tracks.dropDuplicates(["PlaylistId", "TrackId"]), "PlaylistId")
    .join(tracks.select("TrackId", F.col("Name").alias("TrackName"), "Composer"), "TrackId")
    .join(track_revenue, "TrackId", "left")
    .fillna(0.0, subset=["track_revenue"])
    .select("PlaylistId","PlaylistName", "TrackName", "Composer", "track_revenue")
    .orderBy("PlaylistName", "TrackName")
)

print("=== Playlist Report (PySpark) ===")
report.show(20, truncate=False)

# Save CSV + Parquet
pdf_report = report.toPandas()
pdf_report.to_csv("playlist_report.csv", index=False)
pdf_report.to_parquet("playlist_report.parquet", index=False)
print("Saved: playlist_report.csv / playlist_report.parquet")

# Save to SQLite table
conn = sqlite3.connect("playlist_report.db")
pdf_report.to_sql("playlist_report", conn, if_exists="replace", index=False)
conn.close()
print("Saved: playlist_report.db (table: playlist_report)")

spark.stop()

# ── Part 6 – Load Parquet in PyFlink + same analysis ────────────────────────

from pyflink.table import EnvironmentSettings, TableEnvironment

env_settings = EnvironmentSettings.in_batch_mode()
table_env = TableEnvironment.create(env_settings)
table_env.get_config().set("taskmanager.memory.managed.size", "512mb")
table_env.get_config().set("parallelism.default", "1")

# Load the Parquet saved by PySpark
pdf_parquet = pd.read_parquet("playlist_report.parquet")
table_env.create_temporary_view("playlist_report", table_env.from_pandas(pdf_parquet))

print("\n=== Playlist Report loaded from Parquet (PyFlink SQL) ===")
table_env.execute_sql("""
    SELECT PlaylistName, TrackName, Composer, track_revenue
    FROM playlist_report
    ORDER BY PlaylistName, TrackName
""").print()

# Same analysis from raw sources in PyFlink SQL
pdf_playlists       = pd.read_csv("DataSet_csv/Playlist.csv")
pdf_playlist_tracks = pd.read_csv("DataSet_csv/PlaylistTrack.csv")
pdf_tracks          = pd.read_csv("DataSet_csv/Track.csv").rename(columns={"Name": "TrackName"})
pdf_invoice_lines   = pd.read_csv("DataSet_csv/InvoiceLine.csv")

table_env.create_temporary_view("playlists",       table_env.from_pandas(pdf_playlists))
table_env.create_temporary_view("playlist_tracks", table_env.from_pandas(pdf_playlist_tracks))
table_env.create_temporary_view("tracks",          table_env.from_pandas(pdf_tracks))
table_env.create_temporary_view("invoice_lines",   table_env.from_pandas(pdf_invoice_lines))

print("\n=== Playlist Report from raw CSV (PyFlink SQL) ===")

table_env.execute_sql("""
    CREATE TEMPORARY VIEW track_revenue_view AS
    SELECT TrackId,
           ROUND(SUM(CAST(UnitPrice AS DOUBLE) * CAST(Quantity AS INT)), 2) AS track_revenue
    FROM invoice_lines
    GROUP BY TrackId
""")

table_env.execute_sql("""
    CREATE TEMPORARY VIEW unique_playlist_tracks AS
    SELECT DISTINCT PlaylistId, TrackId FROM playlist_tracks
""")

table_env.execute_sql("""
    SELECT p.Name AS PlaylistName,
           t.TrackName,
           t.Composer,
           COALESCE(tr.track_revenue, 0.0) AS track_revenue
    FROM playlists p
    JOIN unique_playlist_tracks pt ON p.PlaylistId = pt.PlaylistId
    JOIN tracks t  ON pt.TrackId = t.TrackId
    LEFT JOIN track_revenue_view tr ON t.TrackId = tr.TrackId
    ORDER BY PlaylistName, TrackName
""").print()
