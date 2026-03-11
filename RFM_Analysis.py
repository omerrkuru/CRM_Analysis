import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Read dataset
df_ = pd.read_csv(r"C:\Users\omerkuru\Desktop\Omer_Hub\Data.csv")
df = df_.copy()

# General overview
print("First 5 observations:")
print(df.head())

print("\nShape:", df.shape)

print("\nColumns:", df.columns)

print("\nInfo:")
print(df.info())

print("Describe:")
print(df.describe().T)

print(df.isnull().sum())

# Convert date columns
date_cols = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# Derived variables
df["total_orders"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_values"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df["avg_order_value"] = df["total_values"] / df["total_orders"]

# Basic metrics
print("\nTotal number of customers:", df["master_id"].nunique())
print("Total online orders:", df["order_num_total_ever_online"].sum())
print("Total offline orders:", df["order_num_total_ever_offline"].sum())

# Top 10 customers with the most orders
top_10_orders = df.sort_values("total_orders", ascending=False)[["master_id", "total_orders", "total_values"]].head(10)
print("\nTop 10 customers with the highest number of orders:")
print(top_10_orders)

# Top 10 customers with the highest spending
top_10_value = df.sort_values("total_values", ascending=False)[["master_id", "total_values", "total_orders"]].head(10)
print("\nTop 10 customers with the highest spending:")
print(top_10_value)

# Online vs Offline order comparison
total_online_orders = df["order_num_total_ever_online"].sum()
total_offline_orders = df["order_num_total_ever_offline"].sum()

if total_online_orders > total_offline_orders:
    diff = total_online_orders - total_offline_orders
    print(f"\nOnline orders exceed offline orders by {diff} orders.")

elif total_offline_orders > total_online_orders:
    diff = total_offline_orders - total_online_orders
    print(f"\nOffline orders exceed online orders by {diff} orders.")

else:
    print("\nOnline and offline order counts are equal.")

# Online vs Offline revenue comparison
total_online_value = df["customer_value_total_ever_online"].sum()
total_offline_value = df["customer_value_total_ever_offline"].sum()

print("\nTotal Online Revenue:", total_online_value)
print("Total Offline Revenue:", total_offline_value)

if total_online_value > total_offline_value:
    print(f"Online revenue exceeds offline revenue by {total_online_value - total_offline_value:.2f} TL.")
elif total_offline_value > total_online_value:
    print(f"Offline revenue exceeds online revenue by {total_offline_value - total_online_value:.2f} TL.")
else:
    print("Online and offline revenues are equal.")

# Overall average order value
general_avg_order_value = df["total_values"].sum() / df["total_orders"].sum()
print("\nOverall average order value:", round(general_avg_order_value, 2))

# 20 customers with the oldest last purchase date
oldest_customers = df.sort_values("last_order_date", ascending=True)[["master_id", "last_order_date"]].head(20)
print("\n20 customers with the oldest last purchase date:")
print(oldest_customers)

# Top 10% most valuable customers
top_10_percent_threshold = df["total_values"].quantile(0.90)
top_10_percent_customers = df[df["total_values"] >= top_10_percent_threshold][["master_id", "total_values", "total_orders"]]

print("\nTop 10% most valuable customers:")
print(top_10_percent_customers.sort_values("total_values", ascending=False).head())

# RFM ANALYSIS
analysis_date = df["last_order_date"].max() + pd.Timedelta(days=2)

rfm = df[["master_id", "last_order_date", "total_orders", "total_values"]].copy()
rfm.columns = ["customer_id", "last_order_date", "frequency", "monetary"]

# Recency calculation
rfm["recency"] = (analysis_date - rfm["last_order_date"]).dt.days

# Remove invalid values
rfm = rfm[(rfm["frequency"] > 0) & (rfm["monetary"] > 0)]

# Scores
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

print("\nFirst 5 rows of RFM:")
print(rfm.head())

# Segment summary
segment_summary = rfm.groupby("segment").agg({
    "recency": ["mean", "count"],
    "frequency": "mean",
    "monetary": "mean"
}).round(2)

print("\nSegment summary:")
print(segment_summary)

# Visualization 1: Segment distribution
plt.figure(figsize=(10, 6))
rfm["segment"].value_counts().plot(kind="bar")
plt.title("RFM Segment Distribution")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization 2: Monetary distribution
plt.figure(figsize=(10, 6))
sns.histplot(rfm["monetary"], bins=30, kde=True)
plt.title("Monetary Distribution")
plt.xlabel("Monetary")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

