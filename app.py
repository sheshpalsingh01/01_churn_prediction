import streamlit as st 
import numpy as np 
import pandas as pd 
import pickle 
import tensorflow as tf 
from sklearn.preprocessing import  StandardScaler, LabelEncoder,OneHotEncoder


# Load the Train model
model=tf.keras.models.load_model("model.h5")

#load pkl file
with open("label_encoder_gender.pkl",'rb') as f:
    label_encoder_gender =pickle.load(f)
    
with open("onehot_encoder_geo.pkl",'rb') as f:
    onehot_encoder_geo =pickle.load(f)
    
with open('scaler.pkl','rb') as f:
    scaler =pickle.load(f)
    
# Stramlit app
st.title("Customer Churn Prediction")

# Initialize session state
if 'geography' not in st.session_state:
    st.session_state.geography = 'France'
if 'gender' not in st.session_state:
    st.session_state.gender = label_encoder_gender.classes_[0]
if 'age' not in st.session_state:
    st.session_state.age = 18
if 'balance' not in st.session_state:
    st.session_state.balance = 0.0
if 'credit_score' not in st.session_state:
    st.session_state.credit_score = 0.0
if 'estimated_salary' not in st.session_state:
    st.session_state.estimated_salary = 0.0
if 'tenure' not in st.session_state:
    st.session_state.tenure = 0
if 'num_of_products' not in st.session_state:
    st.session_state.num_of_products = 1
if 'has_cr_card' not in st.session_state:
    st.session_state.has_cr_card = 0
if 'is_active_member' not in st.session_state:
    st.session_state.is_active_member = 0

# User input
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'], key='geography')
gender = st.selectbox('Gender', label_encoder_gender.classes_, key='gender')
age = st.slider('Age', 18, 92, key='age')
balance = st.number_input('Balance', key='balance')
credit_score = st.number_input('Credit Score', key='credit_score')
estimated_salary = st.number_input('Estimated Salary', key='estimated_salary')
tenure = st.slider('Tenure', 0, 10, key='tenure')
num_of_products = st.slider('Number of Products', 1, 4, key='num_of_products')
has_cr_card = st.selectbox('Has Credit Card', [0, 1], key='has_cr_card')
is_active_member = st.selectbox('Is Active Member', [0, 1], key='is_active_member')

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# predition of model

prediction=model.predict(input_data_scaled)
prediction_probability= prediction[0][0]

st.write(f"Churn Probability: {prediction_probability:.2f}")

if prediction_probability>0.5:
    st.write('The Customer is likely to Churn.')
else:
    st.write('The Customer is not likely to churn')
