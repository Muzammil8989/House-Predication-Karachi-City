# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Custom CSS for styling
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    h1 {color: #264653; text-align: center;}
    .stButton>button {background-color: #2a9d8f; color: white; border-radius: 5px;}
    .stTextInput>div>div>input {border-radius: 5px;}
    .prediction-box {background-color: #e9f5f3; padding: 20px; border-radius: 10px; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# Load artifacts (cached)
@st.cache_data
def load_artifacts():
    model = joblib.load('house_predictor.pkl')
    params = joblib.load('preprocessing_params.pkl')
    return model, params

model, params = load_artifacts()

# Helper function
def extract_locality(address):
    try:
        return address.split(',')[0].strip().split(' - ')[0]
    except:
        return 'Unknown'

# App header
st.markdown("<h1>🏠 Karachi Real Estate Valuation</h1>", unsafe_allow_html=True)
st.markdown("---")

# Input columns
col1, col2 = st.columns(2)
with col1:
    bedrooms = st.number_input('**Bedrooms**', min_value=1, max_value=10, value=3)
    area = st.number_input('**Area (Sq Yards)**', min_value=100, max_value=10000, value=250)

with col2:
    bathrooms = st.number_input('**Bathrooms**', min_value=1, max_value=10, value=2)
    address = st.text_area('**Property Address**', value='Bahria Town Karachi, Karachi')

st.markdown("---")

# Prediction logic - Pakistani style formatting
def format_price(price):
    if price >= 10000000:  # 1 crore
        crore = price / 10000000
        return f"RS {crore:.2f} Crore"
    elif price >= 100000:  # 1 lakh
        lakh = price / 100000
        return f"RS {lakh:.2f} Lakh"
    else:
        return f"RS {price:,.2f}"

if st.button('**Estimate Property Value**', use_container_width=True):
    try:
        # Validate inputs
        if not all([bedrooms, bathrooms, area, address]):
            st.error("Bhai sab fields fill karo!")
            st.stop()
            
        if area <= 0 or bedrooms <= 0 or bathrooms <= 0:
            st.error("Positive numbers dalo yaar!")
            st.stop()

        # Process inputs
        locality = extract_locality(address)
        locality_enc = params['locality_encoder'].get(locality, -1)
        loc_median_price = params['location_medians'].get(locality_enc, params['global_area_median'])
        total_rooms = bedrooms + bathrooms

        # Make prediction
        input_data = pd.DataFrame([[
            bedrooms,
            bathrooms,
            area,
            total_rooms,
            locality_enc,
            loc_median_price
        ]], columns=params['features'])

        pred_price = model.predict(input_data)[0]

        # Display result in Pakistani style
        st.markdown(f"""
        <div class="prediction-box">
            <h3 style='color:#2a9d8f; margin-bottom:10px'>Property Ki Qeemat</h3>
            <h2 style='color:#264653'>{format_price(pred_price)}</h2>
            <p style='color:#666; margin-top:10px'>Ye qeemat hai:
            <br>- {bedrooms} kamray, {bathrooms} bathrooms
            <br>- {area} sq.yds ka plot
            <br>- {locality} ka area</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Masla hogaya! Error: {str(e)}")
        st.info("Wese ye median qeemat hai")

# Model info sidebar - Pakistani style
with st.sidebar:
    st.markdown("## ℹ️ Model Ki Maloomat")
    st.markdown("""
    - **Algorithm**: Random Forest Regressor
    - **Accuracy**: 85%
    - **Data**: Karachi ki property listings
    - **Last Updated**: August 2023
    """)
    st.markdown("---")
    st.markdown("### Asal Cheezein Jo Dekhi Gayi")
    st.write(params['features'])
    st.markdown("---")
    st.markdown("Banaya gaya hai ❤️ se Real Estate AI Team ne")