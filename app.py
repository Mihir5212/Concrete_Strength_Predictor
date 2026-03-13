import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# --- Page Configuration ---
st.set_page_config(page_title="Concrete Strength AI", page_icon="🏗️", layout="wide")

# --- Model Loading ---
@st.cache_resource 
def load_and_train_model():
    df = pd.read_csv('Concrete_Compressive_Strength_Data.csv')
    X = df.drop('Concrete_compressive_strength', axis=1)
    y = df['Concrete_compressive_strength']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Initialize Model Silently (Removes the annoying pop-up on every click)
try:
    with st.spinner("Initializing AI Model..."):
        model = load_and_train_model()
except Exception as e:
    st.error(f"Error training model. Please ensure 'Concrete_Compressive_Strength_Data.csv' is in the directory. \n\nDetails: {e}")
    st.stop()

# --- Main Header ---
st.title("🏗️ Smart Concrete AI Advisor")
st.markdown("Optimize your concrete mix design parameters to predict compressive strength.")
st.divider()

# --- Dashboard Layout ---
# Split the screen: 60% for inputs, 40% for results
input_col, result_col = st.columns([1.5, 1], gap="large")

with input_col:
    st.subheader("📋 Mix Design Parameters (kg/m³)")
    
    # Categorize inputs into 3 neat sub-columns
    binders, liquids, aggregates = st.columns(3)
    
    with binders:
        st.markdown("**Binders**")
        cement = st.number_input("Cement", value=540.0, step=10.0)
        slag = st.number_input("Blast Furnace Slag", value=0.0, step=10.0)
        fly_ash = st.number_input("Fly Ash", value=0.0, step=10.0)
        
    with liquids:
        st.markdown("**Liquids & Admixtures**")
        water = st.number_input("Water", value=162.0, step=5.0)
        super_p = st.number_input("Superplasticizer", value=2.5, step=0.5)
        # Moved age here to balance the columns visually
        age = st.slider("Curing Age (Days)", 1, 365, 28)
        
    with aggregates:
        st.markdown("**Aggregates**")
        coarse_agg = st.number_input("Coarse Aggregate", value=1040.0, step=10.0)
        fine_agg = st.number_input("Fine Aggregate", value=676.0, step=10.0)

# Prepare DataFrame with original column names required by the model
input_df = pd.DataFrame([[cement, slag, fly_ash, water, super_p, coarse_agg, fine_agg, age]],
                        columns=['Cement_1', 'Blast_Furnace_Slag_2', 'Fly_ash_3', 'Water_4', 
                                 'Superplasticizer_5', 'Coarse_Aggregate_6', 'Fine_Aggregate_7', 'Age'])

# Generate Prediction
prediction = model.predict(input_df)[0]

with result_col:
    st.subheader("📊 Prediction Results")
    
    # Create a visual card/container for the output
    with st.container(border=True):
        st.metric(label="Predicted Compressive Strength", value=f"{prediction:.2f} MPa")
        
        # Calculate W/B ratio
        total_binder = cement + slag + fly_ash
        wc_ratio = water / total_binder if total_binder > 0 else 0
        st.caption(f"Calculated Water/Binder Ratio: **{wc_ratio:.2f}**")
        
    # AI Advisor Logic with better visual callouts
    st.markdown("### 🤖 AI Advisor")
    if prediction < 30.0:
        st.warning("""
        **⚠️ Low Strength Warning** Prediction is below the standard target of 30 MPa. 
        *Consider reducing the Water-Binder ratio or increasing the curing age.*
        """)
    else:
        st.success("""
        **✅ Target Achieved** This mix design meets or exceeds the 30 MPa standard.
        """)
