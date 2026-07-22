import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# 1. Load model only
model = pk.load(open('model.pkl', 'rb'))

st.header('Car Price Prediction ML Model')

# 2. Input widgets
cars_name = ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz', 'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel']

name = st.selectbox('Select Car Brand', cars_name)
year = st.slider('Car Manufactured Year', 1994, 2026)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', ['Diesel', 'Petrol', 'LPG', 'CNG'])
seller_type = st.selectbox('Seller type', ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox('Transmission type', ['Manual', 'Automatic'])
owner = st.selectbox('Owner type', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
mileage = st.slider('Car Mileage (kmpl)', 10.0, 40.0)
engine = st.slider('Engine Capacity (CC)', 700, 5000)
max_power = st.slider('Max Power (bhp)', 0.0, 500.0)
seats = st.slider('No of Seats', 2, 10)

# 3. Prediction logic
if st.button("Predict"):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    # Encode categorical columns
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data_model['name'].replace(cars_name, list(range(1, len(cars_name) + 1)), inplace=True)

    # Predict directly without scaler
    car_price = model.predict(input_data_model)

    predicted_price = max(0, float(car_price[0]))

    st.markdown(f'Car Price is going to be **₹{predicted_price:,.2f}**')
    st.success('Car Price Predicted Successfully!')