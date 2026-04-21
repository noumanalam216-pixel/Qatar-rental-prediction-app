import streamlit as st
import pandas as pd 
import numpy as np 
import pickle 

#Load Model

with open("models/rent_model.pkl", "rb") as f:
    model = pickle.load(f)


#Page config

st.set_page_config(
    page_title="Qatar Rental Price Predictor",
    page_icon="QA",
    layout="wide"
)

#CSS
st.markdown("""
<style>

.result-box {
    background-color:#1e293b;
    padding:10px 20px;
    border-radius:10px;
    text-align:center;
    color:white;
    margin-top:15px;
}

.big-font {
    font-size:34px;
    font-weight:bold;
    color:#00e676;
}

.range-text {
    font-size:16px;
    margin-top:5px;
}

.metric-text {
    font-size:15px;
    color:#334155;
    font-weight:600;
}

</style>
""", unsafe_allow_html=True)

#Title

st.markdown(
"# 🇶🇦 Qatar Rental Price Predictor"
)

st.caption(
"AI-powered rental price estimation for apartments in Qatar"
)


#Input Section

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox(
        "Location",
        ["Doha", "Lusail", "Al Wakra", "Other"]

    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=6,
        value=2
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=5,
        value=2
    )

with col2:
    area = st.number_input(
        "Area (sqm)",
        min_value = 30,
        max_value=500,
        value=120

    )

    Property_type = st.selectbox(
        "Property type",
        ["Apartment"]
    )


#Predict Button

if st.button("🔍 Predict Rent"):

    st.divider()

    area_per_room = area / bedrooms

    input_data = pd.DataFrame({
        "location": [location],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "area": [area],
        "area_per_room": [area_per_room],
        "property_type": [Property_type]
    })

    log_prediction = model.predict(input_data)

    price = np.expm1(log_prediction)[0]

    lower = price * 0.9
    upper = price * 1.1

    # IMPORTANT
    rent_per_sqm = price / area


# Display Result
  
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    st.markdown(
        f'<p style="font-size:20px;">Estimated Monthly Rent</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<p class="big-font">QAR {price:,.0f}</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<p class="range-text">Estimated Range: QAR {lower:,.0f} — QAR {upper:,.0f}</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<p class="metric-text">Rent per sqm: {rent_per_sqm:.2f} QAR</p>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)


# Explanation Section

st.markdown("---")

st.subheader("🧠 How This Prediction Works")

st.markdown("""
- The model uses **Gradient Boosting Regression**.
- Prices were **log-transformed** to improve prediction accuracy.
- Features used:
    - Location
    - Bedrooms
    - Bathrooms
    - Area
    - Property Type
    - Area per room
""")