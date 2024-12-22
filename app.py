import streamlit as st
import pandas as pd
import joblib
import numpy as np
@st.cache_resource
def load_model():
    return joblib.load('bike_buyers_model.joblib')
model = load_model()
st.title('Bike Buyers Prediction')
income = st.number_input('Income', min_value=0)
children = st.number_input('Number of Children', min_value=0, max_value=10, step=1)
cars = st.number_input('Number of Cars', min_value=0, max_value=10, step=1)
age = st.number_input('Age', min_value=0, max_value=120, step=1)
marital_status = st.selectbox('Marital Status', ['Single', 'Married'])
gender = st.selectbox('Gender', ['Male', 'Female'])
education = st.selectbox('Education', ['High School', 'Partial College', 'Bachelors', 'Graduate Degree'])
occupation = st.selectbox('Occupation', ['Manual', 'Skilled Manual', 'Clerical', 'Professional', 'Management'])
home_owner = st.selectbox('Home Owner', ['Yes', 'No'])
commute_distance = st.selectbox('Commute Distance', ['0-1 Miles', '1-2 Miles', '2-5 Miles', '5-10 Miles', '10+ Miles'])
region = st.selectbox('Region', ['Europe', 'Pacific', 'North America'])
if st.button('Predict'):
    try:
        input_data = pd.DataFrame({
            'Income': [income],
            'Children': [children],
            'Cars': [cars],
            'Age': [age],
            'Marital Status': [marital_status],
            'Gender': [gender],
            'Education': [education],
            'Occupation': [occupation],
            'Home Owner': [home_owner],
            'Commute Distance': [commute_distance],
            'Region': [region]
        })
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        st.write('Prediction:', 'Will buy a bike' if prediction[0] == 1 else 'Will not buy a bike')
        st.write(f'Probability of buying a bike: {prediction_proba[0][1]:.2f}')
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
st.markdown("""
## About this app
This app predicts whether a person is likely to buy a bike based on various features.
Enter the required information and click 'Predict' to see the result.
""")