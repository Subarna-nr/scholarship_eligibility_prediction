import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------
# Streamlit page settings
# -----------------------
st.set_page_config(page_title="Scholarship Eligibility Predictor", layout="wide")

# -----------------------
# Load trained model safely
# -----------------------
@st.cache_resource
def load_model():
    # Build absolute path to model (works both locally and on Streamlit Cloud)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.joblib')
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please ensure 'models/best_model.joblib' exists.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# -----------------------
# App Title
# -----------------------
st.title("🎓 Scholarship Eligibility Prediction")
st.write("Enter student details to predict scholarship eligibility")

# -----------------------
# User Input Form
# -----------------------
with st.form("student_form"):
    st.subheader("Student Information")

    # Categorical fields
    education = st.selectbox("Education Qualification",
                             ["Undergraduate", "Postgraduate", "PhD", "Diploma", "Other"])
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    community = st.selectbox("Community", ["Reserved", "Unreserved"])
    religion = st.selectbox("Religion", ["Hindu", "Muslim", "Christian", "Other"])
    exservice = st.selectbox("Family of ex-servicemen", ["Yes", "No"])
    disability = st.selectbox("Disability Status", ["Yes", "No"])
    sports = st.selectbox("Sports Quota", ["Yes", "No"])
    india = st.selectbox("Indian nationality", ["Yes", "No"])

    # Numeric fields
    annual_percentage = st.slider("Annual Percentage (%)", 0, 100, 75)
    income_num = st.number_input("Annual Income (in INR)", value=200000, step=10000)

    submitted = st.form_submit_button("Predict")

# -----------------------
# Make prediction
# -----------------------
if submitted:
    # Build dataframe matching training columns
    X_new = pd.DataFrame([{
        'Education': education,
        'Gender': gender,
        'Community': community,
        'Religion': religion,
        'Exservice-men': exservice,
        'Disability': disability,
        'Sports': sports,
        'India': india,
        'AnnualPercentage': annual_percentage,
        'IncomeNum': income_num
    }])

    try:
        pred = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0][1]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("Prediction Result")
    if pred == 1:
        st.success(f"✅ Eligible for Scholarship (Probability: {prob:.2f})")
    else:
        st.error(f"❌ Not Eligible for Scholarship (Probability: {prob:.2f})")

# -----------------------
# Footer
# -----------------------
st.write("---")
st.caption("Built with Streamlit + Scikit-learn | Scholarship Eligibility Project")


