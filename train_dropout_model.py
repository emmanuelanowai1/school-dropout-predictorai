# ✅ Final Streamlit App — All Features Included

import streamlit as st
import pandas as pd
import joblib
from gemini import generate_gemini_response  # Replace with real Gemini API function

# Load trained model
model = joblib.load("dropout_model.pkl")

# Set page config
st.set_page_config(
    page_title="🎓 School Dropout Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title
st.title("🎓 School Dropout Predictor with AI Copilot")
st.markdown("---")

# Tabs layout for navigation
tab1, tab2 = st.tabs(["📋 Manual Prediction", "📤 Bulk Upload"])

# ======= MANUAL PREDICTION TAB =======
with tab1:
    st.header("📋 Enter Student Data")
    st.markdown("Fill out the form below to predict dropout risk and get AI feedback.")

    with st.form("manual_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            student_id = st.text_input("Student ID")
            age = st.number_input("Age", min_value=10, max_value=30, value=18)
            gender = st.selectbox("Gender", ["Male", "Female"])

        with col2:
            cgpa = st.slider("CGPA (Out of 5.0)", min_value=0.0, max_value=5.0, step=0.1)
            attendance = st.slider("Attendance (%)", 0, 100, 80)
            behavioral = st.slider("Behavioral Rating (%)", 0, 100, 70)

        with col3:
            study_time = st.number_input("Study Time (Hours/Week)", min_value=0, value=10)
            parental_support = st.selectbox("Parental Support", ["Yes", "No"])
            extra_class = st.selectbox("Extra Paid Class", ["Yes", "No"])

        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [1 if gender == "Male" else 0],
            "CGPA": [cgpa],
            "Attendance": [attendance],
            "Behavioural Rating": [behavioral],
            "Study Time": [study_time],
            "Parental Support": [1 if parental_support == "Yes" else 0],
            "Extra Paid Class": [1 if extra_class == "Yes" else 0]
        })

        # Predict
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] * 100

        st.markdown("---")
        st.subheader(f"🎯 Dropout Risk Score: **{prob:.2f}%**")
        if prediction:
            st.error("❌ This student is at **risk of dropping out**.")
        else:
            st.success("✅ This student is **not at immediate risk**.")

        # AI Copilot
        st.markdown("### 🤖 Gemini AI Copilot Suggestion")
        ai_response = generate_gemini_response(input_df.iloc[0], student_id)
        st.info(ai_response)

# ======= BULK PREDICTION TAB =======
with tab2:
    st.header("📤 Upload Dataset for Bulk Prediction")
    st.markdown("Upload a CSV with the same column structure as the training data.")

    # Sample data preview
    sample_data = pd.read_csv("MODEL TRAINING DATASET.csv").head()
    st.markdown("### 📌 Sample Format:")
    st.dataframe(sample_data)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df_encoded = df.copy()

        df_encoded['Gender'] = df_encoded['Gender'].map({'Male': 1, 'Female': 0})
        df_encoded['Parental Support'] = df_encoded['Parental Support'].map({'Yes': 1, 'No': 0})
        df_encoded['Extra Paid Class'] = df_encoded['Extra Paid Class'].map({'Yes': 1, 'No': 0})

        X = df_encoded.drop(columns=['Dropout'], errors='ignore')

        df['Dropout Prediction'] = model.predict(X)
        df['Dropout Risk (%)'] = model.predict_proba(X)[:, 1] * 100

        df['AI Advice'] = [generate_gemini_response(row, row['Student ID']) for _, row in X.iterrows()]

        st.markdown("---")
        st.subheader("📊 Prediction Results")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "predictions.csv", "text/csv")

# ======= FOOTER LINKS =======
st.markdown("---")
st.markdown("💡 [Calculate CGPA Online](https://www.schooltry.com.ng/2023/10/how-to-calculate-your-cgpa-in.html)")
