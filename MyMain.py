import streamlit as st
import pandas as pd
import joblib

# Load models
clf = joblib.load('classifier_model.pkl')
regressor_pipe = joblib.load('xgboost_model.pkl')

# Define function to predict location
def predict_location(bath, balcony, total_sqft_int, bhk):
    input_data = pd.DataFrame({'bath': [bath],
                               'balcony': [balcony],
                               'total_sqft_int': [total_sqft_int],
                               'bhk': [bhk]})
    location_pred = clf.predict(input_data)[0]
    return location_pred

# Define function to predict price
def predict_price(location, bath, balcony, total_sqft_int, bhk):
    input_data = pd.DataFrame({'location': [location],
                               'bath': [bath],
                               'balcony': [balcony],
                               'total_sqft_int': [total_sqft_int],
                               'bhk': [bhk]})
    price_pred = regressor_pipe.predict(input_data)[0]
    return price_pred

# Streamlit app
st.title('House Recommendation System')

bath = st.number_input('Number of Bathrooms', min_value=1, max_value=10, step=1)
balcony = st.number_input('Number of Balconies', min_value=0, max_value=5, step=1)
total_sqft_int = st.number_input('Total Square Feet', min_value=100, max_value=100000, step=1)
bhk = st.number_input('Number of Bedrooms (BHK)', min_value=1, max_value=10, step=1)

if st.button('Predict'):
    location_prediction = predict_location(bath, balcony, total_sqft_int, bhk)
    st.success(f'Predicted Location: {location_prediction}')

    price_prediction = predict_price(location_prediction, bath, balcony, total_sqft_int, bhk)
    rounded_price_prediction = round(price_prediction, 2)
    st.success(f'Predicted Price: {rounded_price_prediction}''lacs')

