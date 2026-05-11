# ==========================================
# RESTAURANT RATING PREDICTION APP
# ==========================================

import streamlit as st
import pandas as pd
import joblib

# ==========================================
# LOAD MODEL FILES
# ==========================================

model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
features = joblib.load("features.pkl")

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Restaurant Rating Prediction",
    page_icon="🍽️",
    layout="centered"
)

# ==========================================
# TITLE
# ==========================================

st.title("🍽️ Restaurant Rating Prediction using ML")

st.write("Predict restaurant ratings using Machine Learning")

# ==========================================
# USER INPUTS
# ==========================================

votes = st.number_input(
    "Number of Votes",
    min_value=0,
    step=1
)

price_range = st.slider(
    "Price Range",
    1,
    4
)

city = st.text_input("City")

cuisines = st.text_input("Cuisines")

currency = st.text_input("Currency")

online_delivery = st.selectbox(
    "Has Online Delivery",
    ["Yes", "No"]
)

table_booking = st.selectbox(
    "Has Table Booking",
    ["Yes", "No"]
)

# ==========================================
# ENCODE FUNCTION
# ==========================================

def encode_value(column, value):

    if column in label_encoders:

        le = label_encoders[column]

        if value in le.classes_:
            return le.transform([value])[0]

    return 0

# ==========================================
# CREATE INPUT DATA
# ==========================================

input_data = {}

for feature in features:
    input_data[feature] = 0

# Numeric Inputs
if "Votes" in input_data:
    input_data["Votes"] = votes

if "Price range" in input_data:
    input_data["Price range"] = price_range

# Encoded Inputs
if "City" in input_data:
    input_data["City"] = encode_value(
        "City",
        city
    )

if "Cuisines" in input_data:
    input_data["Cuisines"] = encode_value(
        "Cuisines",
        cuisines
    )

if "Currency" in input_data:
    input_data["Currency"] = encode_value(
        "Currency",
        currency
    )

if "Has Online delivery" in input_data:
    input_data["Has Online delivery"] = encode_value(
        "Has Online delivery",
        online_delivery
    )

if "Has Table booking" in input_data:
    input_data["Has Table booking"] = encode_value(
        "Has Table booking",
        table_booking
    )

# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button("Predict Rating"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    st.success(
        f"⭐ Predicted Restaurant Rating: {prediction:.2f}"
    )