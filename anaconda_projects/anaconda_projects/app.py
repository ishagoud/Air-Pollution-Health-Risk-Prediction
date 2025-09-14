import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- LOAD MODEL & ARTIFACTS --------------------
@st.cache_resource
def load_artifacts():
    rf_smote = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    X_train_cols = joblib.load("X_train_columns.pkl")
    return rf_smote, scaler, X_train_cols

rf_smote, scaler, X_train_cols = load_artifacts()

# Mapping for the predicted classes
class_mapping = {
    0: "No Impact",
    1: "Low Impact",
    2: "Medium Impact",
    3: "High Impact"
}

# -------------------- STREAMLIT LAYOUT --------------------
st.title(" Air Quality Health Impact Prediction")
st.markdown("Upload environmental and health data to predict the **Health Impact Class**.")

# -------------------- SECTION 1: PREDICT FROM CSV --------------------
st.header("📂 Upload a CSV File for Batch Prediction")
st.write("Upload a CSV file with new data to predict Health Impact Class.")

uploaded_file = st.file_uploader("Upload CSV file here", type="csv")

if uploaded_file is not None:
    try:
        new_data = pd.read_csv(uploaded_file)

        st.subheader(" Uploaded Data Preview")
        st.dataframe(new_data.head())

        # -------------------- FEATURE ENGINEERING --------------------
        pollutants = ['PM10', 'PM2_5', 'NO2', 'SO2', 'O3']
        existing_pollutants = [c for c in pollutants if c in new_data.columns]

        if existing_pollutants:
            new_data['AirPollutionIndex'] = new_data[existing_pollutants].mean(axis=1)
            new_data['Respiratory_Risk'] = new_data[existing_pollutants].mean(axis=1)

        if 'Temperature' in new_data.columns and 'Humidity' in new_data.columns:
            new_data['TempHumidityIndex'] = new_data['Temperature'] * new_data['Humidity'] / 100

        # -------------------- PRE-PROCESSING --------------------
        original_data_for_display = new_data.copy()

        categorical_cols = new_data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            try:
                le = joblib.load(f"le_{col}.pkl")
                new_data[col] = le.transform(new_data[col])
            except FileNotFoundError:
                st.warning(f"⚠️ LabelEncoder for column '{col}' not found. Using default encoding 0.")
                new_data[col] = 0

        # Ensure all training columns are present
        for col in X_train_cols:
            if col not in new_data.columns:
                new_data[col] = 0

        new_data = new_data[X_train_cols]

        # Scale numeric features
        numeric_cols = new_data.select_dtypes(include=np.number).columns
        new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])

        # -------------------- MAKE PREDICTIONS --------------------
        predictions = rf_smote.predict(new_data)
        original_data_for_display['Predicted_HealthImpactClass'] = predictions
        original_data_for_display['Predicted_HealthImpactLabel'] = original_data_for_display['Predicted_HealthImpactClass'].map(class_mapping)

        # -------------------- DISPLAY RESULTS --------------------
        st.subheader(" Prediction Results")
        st.dataframe(original_data_for_display)

        # -------------------- VISUALIZATION OF PREDICTIONS --------------------
        st.subheader("📊 Prediction Summary")
        fig, ax = plt.subplots()
        sns.countplot(
            x='Predicted_HealthImpactLabel',
            data=original_data_for_display,
            ax=ax,
            order=list(class_mapping.values())
        )
        plt.title('Distribution of Predicted Health Impact Classes')
        plt.xlabel("Health Impact Class")
        plt.ylabel("Count")
        st.pyplot(fig)

        # -------------------- DOWNLOAD LINK --------------------
        csv = original_data_for_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )

        st.success("🎉 Prediction completed successfully!")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}. Please check your CSV file format.")
