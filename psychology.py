import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('model.pkl', 'rb') as f:
    model=pickle.load(f)

# Define the feature columns (excluding the target variable)
feature_columns = [
    'WorkTimeInSeconds', 'annotatorAge', 'distracted', 'draining', 'frequency', 'importance',
    'logTimeSinceEvent', 'openness', 'similarity', 'stressful',
    'annotatorGender_na', 'annotatorGender_nonBinary', 'annotatorGender_other',
    'annotatorGender_transman', 'annotatorGender_transwoman', 'annotatorGender_woman',
    'annotatorRace_black', 'annotatorRace_hisp', 'annotatorRace_indian',
    'annotatorRace_islander', 'annotatorRace_middleEastern', 'annotatorRace_na',
    'annotatorRace_native', 'annotatorRace_other', 'annotatorRace_white'
]

# Create an empty DataFrame to hold user inputs
user_input_df = pd.DataFrame(columns=feature_columns)

# Streamlit app title
st.title("Narrative Type Forecast")

# User input components for numerical features
st.header("Enter Numerical Features")

for col in feature_columns[:10]:  # Adjust the range based on your numerical features
    user_input = st.number_input(f"Enter {col}:")
    user_input_df[col] = [user_input]

# User input components for categorical features
st.header("Select Categorical Features")

for col in feature_columns[10:]:  # Adjust the range based on your categorical features
    user_input = st.checkbox(f"Is {col}?", key=col)
    user_input_df[col] = [user_input]

# When the user clicks the "Predict" button
if st.button("Predict"):
    # Make predictions
    prediction = model.predict(user_input_df)


    if prediction==0:
        st.header("IMAGINED!!!")
    elif prediction==1:
        st.header("RECALLED!!!")
    else:
        st.header("RETOLD!!!")
    