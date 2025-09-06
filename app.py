import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Credit Risk Scoring App")
st.write("Enter applicant details to predict loan approval.")

income = st.number_input("Income", min_value=0.0, value=50000.0)
age = st.number_input("Age", min_value=18, value=30)
loan_amount = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
years_employed = st.number_input("Years Employed", min_value=0, value=5)

features = pd.DataFrame([[income, age, loan_amount, credit_score, years_employed]],
    columns=["income", "age", "loan_amount", "credit_score", "years_employed"])

if st.button("Predict"):
    prediction = model.predict(features)[0]
    result = "Approved ✅" if prediction == 1 else "Rejected ❌"
    st.subheader(f"Prediction: {result}")

    # Explainability with SHAP
    explainer = shap.Explainer(model, features)
    shap_values = explainer(features)

    st.write("### SHAP Explanation")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
