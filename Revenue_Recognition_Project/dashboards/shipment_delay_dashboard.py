# dashboards/shipment_delay_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder


st.set_page_config(page_title="📦 Revenue Recognition - Shipment Delay Dashboard",
                   layout="wide")

# -------------------------------
# Load Data and Model
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\vinna\OneDrive\AI ML Projects\V Projects\Revenue_Recognition_Project\data\revenue_data.csv")
    #st.write("Columns in dataset:", df.columns.tolist())
    return df

@st.cache_resource
def load_model():
    with open("../src/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

df = load_data()
model = load_model()

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("🔎 Filter Shipments")

mode = st.sidebar.multiselect(
    "Select Mode of Shipment",
    options=df["Mode_of_Shipment"].unique(),
    default=df["Mode_of_Shipment"].unique()
)

warehouse = st.sidebar.multiselect(
    "Select Warehouse Block",
    options=df["Warehouse_block"].unique(),
    default=df["Warehouse_block"].unique()
)

filtered_df = df.query("Mode_of_Shipment in @mode and Warehouse_block in @warehouse")

# -------------------------------
# KPIs
# -------------------------------
st.title("📦 Revenue Recognition - Shipment Delay Dashboard")
st.markdown("Analyze shipment delays that impact revenue recognition")

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Shipments", len(filtered_df))
with col2:
    st.metric("Delayed Shipments", filtered_df["Reached_on_Time"].value_counts().get(0, 0))

# -------------------------------
# Charts
# -------------------------------
st.subheader("📊 Shipment Delay Distribution")
fig_delay = px.histogram(filtered_df, x="Reached_on_Time", color="Reached_on_Time",
                         labels={"Reached_on_Time": "Delay (1 = On-Time, 0 = Delayed)"},
                         title="On-Time vs Delayed Shipments")
st.plotly_chart(fig_delay, use_container_width=True)

st.subheader("🚚 Mode of Shipment Breakdown")
fig_mode = px.pie(filtered_df, names="Mode_of_Shipment", title="Shipment Mode Share")
st.plotly_chart(fig_mode, use_container_width=True)

# -------------------------------
# SHAP Feature Importance
# -------------------------------
st.subheader("📈 Feature Importance (SHAP)")

# Load training feature names from model
feature_names = model.get_booster().feature_names

# Prepare data with same columns as training
X = filtered_df.drop(columns=["Reached_on_Time"]).copy()

# Encode categorical columns consistently
categorical_cols = ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender"]
for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# Reindex to match training feature order
X = X.reindex(columns=feature_names, fill_value=0)

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig)