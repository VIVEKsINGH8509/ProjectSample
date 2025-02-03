import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
data = pd.DataFrame(np.hstack([X, y]), columns=['SquareFootage', 'Price'])

# Train the model
model = LinearRegression()
model.fit(data[['SquareFootage']], data['Price'])

# Streamlit app
st.title("Apna MAYURESH CUTIEPIE")

st.write("""
## Simple Linear Regression Model
This app predicts the **House Price** based on the **Square Footage**!
""")

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    square_footage = st.sidebar.slider('Square Footage', min_value=0.0, max_value=2.0, value=1.0)
    return pd.DataFrame({'SquareFootage': [square_footage]})

input_df = user_input_features()

# Display user input
st.subheader('User Input parameters')
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)

# Display prediction
st.subheader('Prediction')
st.write(f"Predicted House Price: ${prediction[0]:.2f}")