from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.write("Estimate whether a telecom customer may cancel service.")
path = Path("models/churn_model.joblib")
if not path.exists():
    st.error("Model not found. Run: python train.py")
    st.stop()
model = joblib.load(path)

with st.form("form"):
    tenure = st.slider("Tenure in months", 1, 72, 18)
    monthly = st.number_input("Monthly charge", 20.0, 180.0, 85.0, 5.0)
    support = st.slider("Support calls", 0, 10, 2)
    late = st.slider("Late payments", 0, 6, 1)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "None"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    submitted = st.form_submit_button("Predict churn risk")

if submitted:
    row = pd.DataFrame([{"tenure_months": tenure, "monthly_charge": monthly,
        "support_calls": support, "late_payments": late, "contract_type": contract,
        "internet_service": internet, "paperless_billing": paperless,
        "senior_citizen": 1 if senior == "Yes" else 0}])
    probability = float(model.predict_proba(row)[0, 1])
    st.metric("Estimated churn probability", f"{probability:.1%}")
    st.progress(probability)
    st.caption("Educational demonstration using synthetic data. Do not use for real customer decisions.")
