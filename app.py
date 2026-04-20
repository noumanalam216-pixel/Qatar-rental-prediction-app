import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL ----------------
with open("models/rent_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Qatar Rental Predictor", page_icon="🏠", layout="centered")

# ---------------- TITLE ----------------
st.title("🏠 Qatar Rental Price Predictor")
st.markdown("Estimate monthly rent using Machine Learning with AI insights")

st.subheader("📋 Property Details")

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Location", ["Doha", "Lusail", "Al Wakra", "Al Khor"])
    bedrooms = st.number_input("Bedrooms", 1, 10, 2)
    bathrooms = st.number_input("Bathrooms", 1, 10, 2)

with col2:
    area = st.number_input("Area (sqm)", 50, 500, 120)
    property_type = st.selectbox("Property Type", ["Apartment", "Villa"])

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Rent"):

    area_per_room = area / bedrooms

    input_data = pd.DataFrame({
        "location": [location],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "area": [area],
        "area_per_room": [area_per_room],
        "property_type": [property_type]
    })

    # Prediction
    log_pred = model.predict(input_data)
    price = np.expm1(log_pred)[0]

    lower = price * 0.9
    upper = price * 1.1
    rent_per_sqm = price / area

    st.divider()

    # ---------------- RESULT CARD ----------------
    st.subheader("💰 Estimated Monthly Rent")

    st.markdown("""
    <style>
    .result-card {
        background-color: #f1f5f9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #22c55e;
    }
    .big-price {
        font-size: 28px;
        font-weight: bold;
        color: #16a34a;
    }
    .range-text {
        color: #334155;
        font-size: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <p class="big-price">QAR {price:,.0f}</p>
        <p class="range-text">Range: QAR {lower:,.0f} — QAR {upper:,.0f}</p>
        <p class="range-text">Rent per sqm: {rent_per_sqm:.2f} QAR</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- LOCATION INSIGHT ----------------
    location_avg = {
        "Doha": 6500,
        "Lusail": 7200,
        "Al Wakra": 4000,
        "Al Khor": 3500
    }

    avg_rent = location_avg.get(location, price)

    st.info(f"📍 Average rent in {location}: QAR {avg_rent:,.0f}")

    if price > avg_rent:
        st.success("📈 This property is above average pricing")
    else:
        st.info("📉 This property is below average pricing")

    # ---------------- CHART ----------------
    st.subheader("📊 Price Comparison")

    labels = ["Predicted", "Average"]
    values = [price, avg_rent]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    st.pyplot(fig)

    # ---------------- SHAP EXPLANATION ----------------
    st.subheader("🧠 Why this price?")

    try:
        explainer = shap.Explainer(model.named_steps["regressor"])
        transformed_input = model.named_steps["preprocessor"].transform(input_data)
        shap_values = explainer(transformed_input)

        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        for name, val in zip(feature_names, shap_values.values[0]):
            st.write(f"{name}: {val:.2f}")

    except:
        st.warning("SHAP explanation not available for this model structure.")

    # ---------------- AI INSIGHTS ----------------
    st.subheader("🤖 AI Insights")

    if area > 150:
        st.write("✔ Large property size increases rental value.")
    if bedrooms > 3:
        st.write("✔ More bedrooms increase property demand.")
    if location == "Doha":
        st.write("✔ Premium location significantly impacts price.")
    if rent_per_sqm > 60:
        st.write("✔ High rent per sqm indicates luxury pricing.")

    # ---------------- HOW IT WORKS ----------------
    st.subheader("🧠 How This Works")

    st.markdown("""
    - Model: Gradient Boosting Regressor  
    - Uses log transformation for better accuracy  
    - Feature engineering: area per room  
    - Includes location-based and AI-driven insights  
    - Provides explainable predictions using SHAP  
    """)