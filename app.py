import streamlit as st
import numpy as np
import pandas as pd
import pickle
import sklearn

df = pickle.load(open('df.pkl', 'rb'))
pipe = pickle.load(open('pipe.pkl', 'rb'))


st.title('🏡 House Price Prediction')

categorical_cols = [df.columns[3], df.columns[4]]
numerical_cols = [col for col in df.columns if col not in categorical_cols + ['Price']]


city = st.selectbox('City', df['City'].unique())
location = st.selectbox('Location', df['Location'].unique())
floor_area = st.number_input('Area SQ/Feet')
floor_no = st.selectbox('Floor No', df['Floor_No'].unique())
bedrooms = st.selectbox('Bedrooms', df['Bedrooms'].unique())
bathrooms = st.selectbox('Bathrooms', df['Bathrooms'].unique())



if st.button("Predict Price"):
    # Input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, floor_area, city, location, floor_no]],
                              columns=['Bedrooms', 'Bathrooms', 'Floor_area', 'City', 'Location', 'Floor_No'])

    # Apply the pipeline (which includes the preprocessing steps)
    prediction = pipe.predict(input_data)[0]

    # Display the result
    st.success(f"🏠 Estimated Price: **TK{np.exp(prediction):,.2f}**")



