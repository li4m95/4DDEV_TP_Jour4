import os
import pandas as pd
import sqlite3

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ── Part 11 – Database Storage (SQLite) ─────────────────────────────────────

conn = sqlite3.connect("analysis_results.db")

# ── Table 1 : top 10 countries — customer count + total revenue ──────────────

pdf_customers = pd.read_csv("DataSet_csv/Customer.csv")[["CustomerId", "Country"]]
pdf_customers["Country"] = pdf_customers["Country"].fillna("Unknown")

pdf_invoices  = pd.read_csv("DataSet_csv/Invoice.csv")[["CustomerId", "Total"]]
pdf_invoices["Total"] = pd.to_numeric(pdf_invoices["Total"])

# Join on CustomerId so revenue is grouped by the customer's country
top10_countries = (
    pdf_customers
    .merge(pdf_invoices, on="CustomerId")
    .groupby("Country")
    .agg(
        customer_count=("CustomerId", "nunique"),
        total_revenue=("Total", "sum")
    )
    .reset_index()
    .assign(total_revenue=lambda df: df["total_revenue"].round(2))
    .sort_values("customer_count", ascending=False)
    .head(10)
)

top10_countries.to_sql("top10_countries", conn, if_exists="replace", index=False)
print("Saved table: top10_countries")
print(top10_countries.to_string(index=False))

# ── Table 2 : playlist revenue report ────────────────────────────────────────

pdf_playlist = pd.read_parquet("playlist_report.parquet")
pdf_playlist.to_sql("playlist_report", conn, if_exists="replace", index=False)
print(f"\nSaved table: playlist_report ({len(pdf_playlist)} rows)")

conn.close()
print("\nDatabase: analysis_results.db")

# ── Verify both tables ────────────────────────────────────────────────────────

conn = sqlite3.connect("analysis_results.db")
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print("\nTables in analysis_results.db:", tables["name"].tolist())
conn.close()
