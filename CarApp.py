import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

model = pk.load(open('model.pkl', 'rb'))

st.header('Car Price Prediction ML Model')
cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2026, value=1994)
km_driven = st.slider('No of kms Driven', 11, 200000, value=11)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Owner type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage', 10, 40, value=10)
engine = st.slider('Engine CC', 700, 5000, value=700)
max_power = st.slider('Max Power', 0, 200, value=0)
seats = st.slider('No of Seats', 5, 10, value=5)

if st.button("Predict"):
    brand_map = {val: i for i, val in enumerate(sorted(cars_data['name'].unique()))}
    fuel_map = {val: i for i, val in enumerate(sorted(cars_data['fuel'].unique()))}
    seller_map = {val: i for i, val in enumerate(sorted(cars_data['seller_type'].unique()))}
    trans_map = {val: i for i, val in enumerate(sorted(cars_data['transmission'].unique()))}
    owner_map = {val: i for i, val in enumerate(sorted(cars_data['owner'].unique()))}

    mapped_name = brand_map[name]
    mapped_fuel = fuel_map[fuel]
    mapped_seller = seller_map[seller_type]
    mapped_trans = trans_map[transmission]
    mapped_owner = owner_map[owner]

    input_data_model = pd.DataFrame([[mapped_name, year, km_driven, mapped_fuel, mapped_seller, mapped_trans, mapped_owner, mileage, engine, max_power, seats]],
                                    columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])

    log_price = model.predict(input_data_model)[0]
    final_price = np.exp(log_price)
    st.markdown(f'### Car Price is going to be ₹ {round(final_price, 2):,}')
    st.success("Car Price Predicted Successfully..")