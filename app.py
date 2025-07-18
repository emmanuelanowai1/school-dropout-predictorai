import streamlit as st
import pandas as pd
import joblib
from gemini import generate_gemini_response

st.set_page_config(page_title="School Dropout Predictor AI", layout="centered")

# Load trained model
model = joblib.load("model.pkl")

st.title("📚 School Dropout Predictor AI")

st.markdown("Enter student data below:")

# Input form
with st.form("input_form"):
    age = st.number_input("Age", min_value=10, max_value=25, value=16)
    cgpa = st.slider("CGPA (out of 5.0)", 0.0, 5.0, 2.5, step=0.1)
    attendance = st.slider("Attendance Rate (%)", 0, 100, 75)
    behaviour = st.slider("Behavioural Rating (%)", 0, 100, 70)
    study_time = st.slider("Study Time (hrs/week)", 0, 40, 10)
    support = st.selectbox("Parental Support", ["No", "Yes"])
    paid_class = st.selectbox("Extra Paid Class", ["No", "Yes"])
    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        "Age": age,
        "CGPA": cgpa,
        "Attendance Rate": attendance,
        "Behavioural Rating": behaviour,
        "Study Time": study_time,
        "Parental Support": 1 if support == "Yes" else 0,
        "Extra Paid Class": 1 if paid_class == "Yes" else 0,
    }])

    prediction = model.predict(input_df)[0]
    risk = "High Risk of Dropout" if prediction == 1 else "Low Risk of Dropout"

    st.subheader("🎯 Prediction Result")
    st.success(risk)

    st.subheader("🤖 Gemini AI Copilot Suggestion")
    advice = generate_gemini_response(input_df.iloc[0])
    st.info(advice)
