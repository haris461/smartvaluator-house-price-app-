#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Custom CSS for styling (dark background and other elements)
st.markdown("""
    <style>
        body {
            background-color: #000000;
            color: white;
        }
        .main {
            background-color: #000000;
            padding: 20px;
            border-radius: 10px;
        }
        h1, h2 {
            color: #00ffff;
        }
        .stButton button {
            background-color: #00cccc;
            color: black;
        }
        .css-1v0mbdj.edgvbvh3 {  /* Adjust input box text color */
            color: white;
        }
        .css-1d391kg {
            background-color: #000000;
        }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("house_data.csv")

df = load_data()
st.markdown("<h1 style='text-align:center;'>🏠 SmartValuator (House Price predicter App)</h1>", unsafe_allow_html=True)

# House input form
st.subheader("🔮 Enter House Details to Predict Price")

area = st.number_input("🏡 Area (in sq. ft)", min_value=300, max_value=10000, value=1000, step=50)
bedrooms = st.number_input("🛏️ Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("🛁 Number of Bathrooms", min_value=1, max_value=5, value=2)
stories = st.number_input("🏗️ Number of Stories", min_value=1, max_value=4, value=2)
parking = st.number_input("🚗 Parking Spaces", min_value=0, max_value=4, value=1)
location = st.text_input("📍 Location (Urban / Suburban / Rural)", value="Urban")
age = st.slider("⏳ Age of House (Years)", min_value=0, max_value=100, value=10)  # Changed to slider

# Preprocess for prediction
X = df.drop("price", axis=1)
y = df["price"]
X_encoded = pd.get_dummies(X, columns=["location"], drop_first=True)

input_data = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "parking": [parking],
    "location": [location.title()],
    "age": [age]
})

input_encoded = pd.get_dummies(input_data, columns=["location"])
for col in X_encoded.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_encoded.columns]

# Train model
model = LinearRegression()
model.fit(X_encoded, y)

# Predict
if st.button("🔍 Predict House Price"):
    predicted_price = model.predict(input_encoded)[0]
    st.success(f"🏷️ Estimated House Price: ${predicted_price:,.0f}")