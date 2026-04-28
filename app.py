import streamlit as st
import pandas as pd
import pickle
import os
# Load the trained model
model, columns=pickle.load(open(os.path.join("demand_model.pkl"), "rb"))
st.title("Demand Prediction App")
#Input
date= st.date_input("Select Date")
price= st.number_input("Price", value=50.0)
inventory= st.number_input("Inventory", value=100)
discount= st.number_input("Discount", value=0.0)
store=st.selectbox("Store", options=["S001", "S002"])
product=st.selectbox("Product", options=["P001", "P002"])
category=st.selectbox("Category", ["Electronics", "Clothing", "Groceries"])
region=st.selectbox("Region", ["North", "South"])
weather=st.selectbox("Weather", ["Sunny", "Rainy", "Snowy", "Cloudy"])
season=st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
if st.button("Predict Demand"):
    input_data = pd.DataFrame({
        "Date": [date],
        "Price": [price],
        "Inventory": [inventory],
        "Discount": [discount],
        "Store": [store],
        "Product": [product],
        "Category": [category],
        "Region": [region],
        "Weather": [weather],
        "Season": [season]
    })
    input_data = pd.get_dummies(input_data, columns=["Store", "Product", "Category", "Region", "Weather", "Season"])
    input_data = input_data.reindex(columns=columns, fill_value=0)
    prediction = model.predict(input_data)
    st.write(f"Predicted Demand: {prediction[0]:.2f}")  