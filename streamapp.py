# ================================
# Streamlit App: Integrated Retail Analytics
# ================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the saved Gradient Boosting model
model = joblib.load("/home/vishwesh/Documents/Labmentix_Internship/Project_9/best_gradient_boosting_model.joblib")

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.set_page_config(page_title="Retail Analytics - Sales Prediction", layout="wide")
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "EDA"])

# ================================
# PREDICTION PAGE
# ================================
if page == "Prediction":
    st.title("🔮 Weekly Sales Prediction")
    st.write("Enter store and economic features to predict **Weekly Sales**.")

    col1, col2 = st.columns(2)

    with col1:
        store = st.number_input("Store ID", min_value=1, max_value=50, step=1)
        dept = st.number_input("Department ID", min_value=1, max_value=100, step=1)
        temperature = st.number_input("Temperature (°F)", min_value=-20.0, max_value=120.0, value=70.0)
        fuel_price = st.number_input("Fuel Price ($)", min_value=0.5, max_value=10.0, value=3.5)
        markdown1 = st.number_input("Markdown 1", min_value=0.0, value=0.0)
        markdown2 = st.number_input("Markdown 2", min_value=0.0, value=0.0)

    with col2:
        markdown3 = st.number_input("Markdown 3", min_value=0.0, value=0.0)
        markdown4 = st.number_input("Markdown 4", min_value=0.0, value=0.0)
        markdown5 = st.number_input("Markdown 5", min_value=0.0, value=0.0)
        cpi = st.number_input("CPI", min_value=50.0, max_value=300.0, value=150.0)
        unemployment = st.number_input("Unemployment Rate (%)", min_value=0.0, max_value=20.0, value=5.0)
        is_holiday = st.selectbox("Is Holiday?", ["No", "Yes"])

    # Encode categorical
    is_holiday_flag = 1 if is_holiday == "Yes" else 0

    # Prepare input dataframe (12 features exactly as in training)
    input_data = pd.DataFrame([[
        store, dept, temperature, fuel_price,
        markdown1, markdown2, markdown3, markdown4, markdown5,
        cpi, unemployment, is_holiday_flag
    ]], columns=[
        "Store", "Dept", "Temperature", "Fuel_Price",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
        "CPI", "Unemployment", "IsHoliday"
    ])

    # Predict button
    if st.button("🚀 Predict Weekly Sales"):
        prediction = model.predict(input_data)[0]
        st.success(f"✅ Predicted Weekly Sales: **${prediction:,.2f}**")

# ================================
# EDA PAGE
# ================================
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.write("Upload your CSV files (Sales, Features, Stores) to analyze trends and insights.")

    sales_file = st.file_uploader("Upload Sales Dataset", type=["csv"])
    features_file = st.file_uploader("Upload Features Dataset", type=["csv"])
    stores_file = st.file_uploader("Upload Stores Dataset", type=["csv"])

    if sales_file and features_file and stores_file:
        sales = pd.read_csv(sales_file)
        features = pd.read_csv(features_file)
        stores = pd.read_csv(stores_file)

        # Merge datasets
        df = pd.merge(sales, features, on=["Store", "Date", "IsHoliday"])
        df = pd.merge(df, stores, on="Store")

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

        # Show results only when button is clicked
        if st.button("📊 Show Results"):
            st.subheader("📌 Dataset Overview")
            st.write(df.head())
            st.write("Shape:", df.shape)
            st.write("Missing Values:", df.isnull().sum())

            # Sales over time
            st.subheader("📈 Weekly Sales Over Time")
            fig, ax = plt.subplots(figsize=(12,6))
            df.groupby("Date")["Weekly_Sales"].sum().plot(ax=ax)
            ax.set_title("Total Weekly Sales Over Time")
            ax.set_ylabel("Weekly Sales ($)")
            st.pyplot(fig)

            # Holiday vs Non-Holiday Sales
            st.subheader("🎉 Holiday vs Non-Holiday Sales")
            fig, ax = plt.subplots(figsize=(8,5))
            sns.boxplot(x="IsHoliday", y="Weekly_Sales", data=df, ax=ax)
            ax.set_title("Holiday Impact on Sales")
            st.pyplot(fig)

            # Store Type vs Sales
            st.subheader("🏬 Store Type vs Sales")
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x="Type", y="Weekly_Sales", data=df, estimator=np.mean, ci=None, ax=ax)
            ax.set_title("Average Sales by Store Type")
            st.pyplot(fig)

            # Correlation Heatmap
            st.subheader("🔥 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10,6))
            corr = df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap of Features")
            st.pyplot(fig)
