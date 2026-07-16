import streamlit as st
import pandas as pd
import joblib

# Load your production model pipeline
model = joblib.load('telco_churn_production_model.pkl')
OPTIMAL_THRESHOLD = 0.1429

st.title("Telco Customer Churn Risk Assessment")
st.write("Input customer demographics and account details to predict risk profile.")

# Create input forms matching your original dataset columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with col2:
    internet = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

# Build features dataframe for model evaluation
input_data = pd.DataFrame([{
    'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
    'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': internet,
    'OnlineSecurity': security, 'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
    'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': contract, 'PaperlessBilling': billing,
    'PaymentMethod': payment, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
}])

if st.button("Evaluate Churn Risk"):
    # Grab probability out of the pipeline
    prob = model.predict_proba(input_data)[0, 1]
    
    st.subheader(f"Calculated Churn Probability: {prob:.2%}")
    
    if prob >= OPTIMAL_THRESHOLD:
        st.error(f"⚠️ HIGH RISK PROFILE (Threshold exceeded: {OPTIMAL_THRESHOLD})")
        st.write("Recommendation: Enroll customer into immediate active retention campaign.")
    else:
        st.success("✅ LOW RISK PROFILE")