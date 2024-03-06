import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model = joblib.load('restaurant_review_model.pkl')

# Load your DataFrame with unique values
df_unique_values = pd.read_csv(r'C:\Users\ziadz\Desktop\py project\archive (8)\adult.csv')

# Define the feature input function
def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 30)
    workclass = st.sidebar.selectbox('Workclass', df_unique_values['workclass'].unique())
    education_num = st.sidebar.slider('Education (Number of years)', 1, 20, 10)
    marital_status = st.sidebar.selectbox('Marital Status', df_unique_values['marital-status'].unique())
    occupation = st.sidebar.selectbox('Occupation', df_unique_values['occupation'].unique())
    gender = st.sidebar.selectbox('Gender', df_unique_values['gender'].unique())
    hours_per_week = st.sidebar.slider('Hours per Week', 1, 100, 40)
    native_country = st.sidebar.selectbox('Native Country', df_unique_values['native-country'].unique())

    data = {'Age': age,
            'Workclass': workclass,
            'Education_Num': education_num,
            'Marital_Status': marital_status,
            'Occupation': occupation,
            'Gender': gender,
            'Hours_per_Week': hours_per_week,
            'Native_Country': native_country}

    features = pd.DataFrame(data, index=[0])
    return features

# Get the features input from the user
input_df = user_input_features()

# Display the user input features
st.subheader('User Input features')
st.write(input_df)

# Predict the income category
prediction = 0

# Display the prediction
st.subheader('Prediction')
income_category = '<=50K' if prediction[0] == 0 else '>50K'
st.write(income_category)