import pandas as pd

pd.set_option("display.max_columns", None)


def load_data():
    df_d = pd.read_csv(r"C:\Users\omerkuru\Desktop\Omer_Hub\Flo_Data.csv")
    return df_d

df = load_data()

# ================================
# Exploratory Data Analysis (EDA)
# ================================

print("First 5 observations")
print(df.head())

print("\nDataset shape")
print(df.shape)

print("\nData types and general info")
print(df.info())

print("\nMissing values")
print(df.isnull().sum())

print("\nDescriptive statistics")
print(df.describe().T)

print("\nUnique value counts")
print(df.nunique())

def outlier_thresholds(dataframe, variable):
    q1 = dataframe[variable].quantile(0.01)
    q3 = dataframe[variable].quantile(0.99)
    iqr = q3 - q1
    low_limit = q1 - 1.5 * iqr
    up_limit = q3 + 1.5 * iqr
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit


df = load_data()

cols = [
    "order_num_total_ever_online",
    "order_num_total_ever_offline",
    "customer_value_total_ever_offline",
    "customer_value_total_ever_online"
]

for col in cols:
    replace_with_thresholds(df, col)

# Total order count and total customer value
df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_customer_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]



# Create CLTV dataframe
cltv = pd.DataFrame()
cltv["customer_id"] = df["master_id"]
cltv["total_order"] = df["total_order_num"]
cltv["total_price"] = df["total_customer_value"]

# Average order value
cltv["average_order_value"] = cltv["total_price"] / cltv["total_order"]

# Average purchase frequency per customer
purchase_frequency = cltv["total_order"].sum() / cltv.shape[0]
cltv["purchase_frequency"] = purchase_frequency

# Repeat purchase rate and churn rate
repeat_rate = cltv[cltv["total_order"] > 1].shape[0] / cltv.shape[0]
churn_rate = 1 - repeat_rate

# If churn rate is extremely low or zero, assign an assumed value
if churn_rate <= 0:
    churn_rate = 0.10

# Estimated profit (assuming 10% margin)
cltv["profit"] = cltv["total_price"] * 0.10

# Customer value
cltv["customer_value"] = cltv["average_order_value"] * cltv["purchase_frequency"]

# CLTV calculation
cltv["cltv"] = (cltv["customer_value"] / churn_rate) * cltv["profit"]

# Customer segmentation
cltv["segment"] = pd.qcut(cltv["cltv"], 4, labels=["D", "C", "B", "A"])

print("\nCustomers with the highest CLTV:")
print(cltv.sort_values(by="cltv", ascending=False).head())

print("\nSegment summary:")
print(
    cltv.groupby("segment").agg({
        "customer_id": "count",
        "total_order": "mean",
        "total_price": "mean",
        "cltv": ["mean", "sum"]
    })
)