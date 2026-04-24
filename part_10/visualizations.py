import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ── Part 10 – Data Visualizations ───────────────────────────────────────────

# ── Chart 1 : Top 10 countries by customer count ────────────────────────────

pdf_customers = pd.read_csv("DataSet_csv/Customer.csv")
pdf_customers["Country"] = pdf_customers["Country"].fillna("Unknown")

top10_countries = (
    pdf_customers.groupby("Country")
    .size()
    .reset_index(name="customer_count")
    .sort_values("customer_count", ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(top10_countries["Country"], top10_countries["customer_count"], color="steelblue")
ax.set_title("Top 10 countries by customer count")
ax.set_xlabel("Country")
ax.set_ylabel("Number of customers")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("top10_countries_chart.png", dpi=150)
print("Saved: top10_countries_chart.png")
plt.show()

# ── Chart 2 : Top 10 most profitable tracks ──────────────────────────────────

pdf_invoice_lines = pd.read_csv("DataSet_csv/InvoiceLine.csv")
pdf_tracks        = pd.read_csv("DataSet_csv/Track.csv")[["TrackId", "Name"]]

top10_tracks = (
    pdf_invoice_lines
    .assign(revenue=pdf_invoice_lines["UnitPrice"] * pdf_invoice_lines["Quantity"])
    .groupby("TrackId")["revenue"]
    .sum()
    .reset_index(name="total_revenue")
    .merge(pdf_tracks, on="TrackId")
    .sort_values("total_revenue", ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(top10_tracks["Name"], top10_tracks["total_revenue"], color="darkorange")
ax.set_title("Top 10 most profitable tracks")
ax.set_xlabel("Total revenue (€)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("top10_tracks_chart.png", dpi=150)
print("Saved: top10_tracks_chart.png")
plt.show()
