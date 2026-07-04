import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Rising Water",
    page_icon="🌊",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------

model = joblib.load("models/flood_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -------------------------------
# Title
# -------------------------------

st.title("🌊 Rising Water")
st.subheader("AI Flood Prediction System")

st.markdown("---")

# -------------------------------
# Input Layout
# -------------------------------

col1, col2 = st.columns(2)

with col1:

    st.markdown("## 🌦 Weather Information")

    temp = st.number_input(
        "Temperature (°C)",
        value=30.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        value=70.0
    )

    cloud = st.number_input(
        "Cloud Cover (%)",
        value=40.0
    )

with col2:

    st.markdown("## 🌧 Rainfall Information")

    annual = st.number_input(
        "Annual Rainfall (mm)",
        value=3000.0
    )

    jan = st.number_input(
        "Jan-Feb Rainfall (mm)",
        value=40.0
    )

    mar = st.number_input(
        "Mar-May Rainfall (mm)",
        value=350.0
    )

    jun = st.number_input(
        "Jun-Sep Rainfall (mm)",
        value=2100.0
    )

    octt = st.number_input(
        "Oct-Dec Rainfall (mm)",
        value=500.0
    )

    avg = st.number_input(
        "Average June Rainfall (mm)",
        value=250.0
    )

    sub = st.number_input(
        "Subdivision Rainfall (mm)",
        value=500.0
    )

st.markdown("---")

# -------------------------------
# Prediction
# -------------------------------

if st.button("🚀 Predict Flood"):

    input_data = pd.DataFrame(
        [[
            temp,
            humidity,
            cloud,
            annual,
            jan,
            mar,
            jun,
            octt,
            avg,
            sub
        ]],
        columns=[
            "Temp",
            "Humidity",
            "Cloud Cover",
            "ANNUAL",
            "Jan-Feb",
            "Mar-May",
            "Jun-Sep",
            "Oct-Dec",
            "avgjune",
            "sub"
        ]
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠ Flood Expected")

    else:

        st.success("✅ No Flood Expected")

    st.metric(
        label="Flood Probability",
        value=f"{probability*100:.2f}%"
    )

    st.progress(float(probability))

    st.subheader("Input Summary")

    st.dataframe(input_data)