import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os

st.set_page_config(page_title="Concrete Strength AI", layout="wide")

# Function to get the model
@st.cache_resource # This keeps the model in memory so it doesn't re-train on every click
def load_and_train_model():
    # Load the data from your uploaded CSV
    df = pd.read_csv('Concrete_Compressive_Strength_Data.csv')
    
    # Define features and target based on your specific column names
    X = df.drop('Concrete_compressive_strength', axis=1)
    y = df['Concrete_compressive_strength']
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Initialize Model
try:
    model = load_and_train_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error training model: {e}")
    st.stop()

# --- UI Layout ---
st.title("🏗️ Smart Concrete AI Advisor")

with st.sidebar:
    st.header("Input Mix Design")
    cement = st.number_input("Cement_1", value=540.0)
    slag = st.number_input("Blast_Furnace_Slag_2", value=0.0)
    fly_ash = st.number_input("Fly_ash_3", value=0.0)
    water = st.number_input("Water_4", value=162.0)
    super_p = st.number_input("Superplasticizer_5", value=2.5)
    coarse_agg = st.number_input("Coarse_Aggregate_6", value=1040.0)
    fine_agg = st.number_input("Fine_Aggregate_7", value=676.0)
    age = st.slider("Age (Days)", 1, 365, 28)

# Prediction
input_df = pd.DataFrame([[cement, slag, fly_ash, water, super_p, coarse_agg, fine_agg, age]],
                        columns=['Cement_1', 'Blast_Furnace_Slag_2', 'Fly_ash_3', 'Water_4', 
                                 'Superplasticizer_5', 'Coarse_Aggregate_6', 'Fine_Aggregate_7', 'Age'])

prediction = model.predict(input_df)[0]

# Display Results
st.metric("Predicted Compressive Strength", f"{prediction:.2f} MPa")

# Advisor Logic
if prediction < 30.0:
    st.warning("Prediction is below standard target (30 MPa). Consider reducing the Water-Cement ratio.")
else:
    st.success("Target strength achieved!")