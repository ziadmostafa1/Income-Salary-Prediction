import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import warnings
warnings.filterwarnings('ignore')

# Load the pre-trained model.
model = joblib.load('model.pkl')

# Load your DataFrame with unique values
df_unique_values = pd.read_csv('Features.csv')

# Define the feature input function
def user_input_features():
    age = st.sidebar.slider('age', 18, 100, 30)
    workclass = st.sidebar.selectbox('workclass', df_unique_values['workclass'].unique())
    education_num = st.sidebar.slider('educational-num', 1, 20, 10)
    marital_status = st.sidebar.selectbox('marital-status', df_unique_values['marital-status'].unique())
    occupation = st.sidebar.selectbox('occupation', df_unique_values['occupation'].unique())
    # Define a dictionary for 'Gender'
    gender_dict = {'Male': 1, 'Female': 0}

    # Use the dictionary keys as the options in the selectbox
    gender = st.sidebar.selectbox('gender', list(gender_dict.keys()))

    # Map the selected option to its corresponding value in the dictionary
    gender = gender_dict[gender]

    hours_per_week = st.sidebar.slider('hours-per-week', 1, 100, 40)
    native_country = st.sidebar.selectbox('native-country', df_unique_values['native-country'].unique())

    data = {'age': age,
            'workclass': workclass,
            'educational-num': education_num,
            'marital-status': marital_status,
            'occupation': occupation,
            'gender': gender,
            'hours-per-week': hours_per_week,
            'native-country': native_country}

    features = pd.DataFrame(data, index=[0])
    return features

# Get the features input from the user
input_df = user_input_features()

# Display the user input features
st.subheader('User Input features')
st.write(input_df)
input_df.columns =['age', 'workclass', 'educational-num', 'marital-status', 'occupation', 'gender', 'hours-per-week', 'native-country']


# Use the function to preprocess df1
df_transformed = input_df.copy()

def align_dataframe(df, model_columns):
    # Create a new DataFrame with the same number of rows, filled with zeros
    aligned_df = pd.DataFrame(0, index=np.arange(len(df)), columns=model_columns)

    # For the specified columns, check if the values match the column names in model_columns
    for column in ['workclass', 'marital-status', 'occupation', 'native-country']:
        if column in df.columns:
            for index, value in df[column].items():
                column_name = f"{column}_{value}"
                if column_name in model_columns:
                    aligned_df.at[index, column_name] = 1

    # For the other columns, just assign the values from df_transformed
    other_columns = list(set(df.columns) - set(['workclass', 'marital-status', 'occupation', 'native-country']))
    aligned_df[other_columns] = df[other_columns]

    # Drop the original columns
    aligned_df.drop(['workclass', 'marital-status', 'occupation', 'native-country'], axis=1, inplace=True, errors='ignore')

    return aligned_df
# Use the function to align df_transformed


# Use the function to align df_transformed
model_columns = ['educational-num', 'gender', 'age','hours-per-week',
       'workclass_Federal-gov', 'workclass_Local-gov',
       'workclass_Never-worked', 'workclass_Private',
       'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc',
       'workclass_State-gov', 'workclass_Without-pay',
       'marital-status_divorced', 'marital-status_married',
       'marital-status_single', 'occupation_Adm-clerical',
       'occupation_Armed-Forces', 'occupation_Craft-repair',
       'occupation_Exec-managerial', 'occupation_Farming-fishing',
       'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct',
       'occupation_Other-service', 'occupation_Priv-house-serv',
       'occupation_Prof-specialty', 'occupation_Protective-serv',
       'occupation_Sales', 'occupation_Tech-support',
       'occupation_Transport-moving', 'native-country_Cambodia',
       'native-country_Canada', 'native-country_China',
       'native-country_Columbia', 'native-country_Cuba',
       'native-country_Dominican-Republic', 'native-country_Ecuador',
       'native-country_El-Salvador', 'native-country_England',
       'native-country_France', 'native-country_Germany',
       'native-country_Greece', 'native-country_Guatemala',
       'native-country_Haiti', 'native-country_Holand-Netherlands',
       'native-country_Honduras', 'native-country_Hong',
       'native-country_Hungary', 'native-country_India',
       'native-country_Iran', 'native-country_Ireland',
       'native-country_Italy', 'native-country_Jamaica',
       'native-country_Japan', 'native-country_Laos',
       'native-country_Mexico', 'native-country_Nicaragua',
       'native-country_Peru', 'native-country_Philippines',
       'native-country_Poland', 'native-country_Portugal',
       'native-country_Puerto-Rico', 'native-country_Scotland',
       'native-country_South', 'native-country_Taiwan',
       'native-country_Thailand', 'native-country_Trinadad&Tobago',
       'native-country_US Minor Islands', 'native-country_United-States',
       'native-country_Vietnam', 'native-country_Yugoslavia']

df_aligned = align_dataframe(df_transformed, model_columns)
df_aligned
# Predict the income category
prediction = model.predict(df_aligned)

# Display the prediction
st.subheader('Prediction')
income_category = '<=50K' if prediction[0] == 1 else '\>50K'
st.write(income_category)