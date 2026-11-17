import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

st.set_page_config(page_title="Student Performance Risk Predictor", layout="centered")

@st.cache_resource
def load_artifacts():
    model = keras.models.load_model("models/student_performance.keras")
    scaler = joblib.load("models/scaler.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return model, scaler, le

st.title("🎓 AI-Based Student Performance Risk Predictor")
st.markdown("Enter a student's academic and behavioral metrics below to predict their performance category (**Low / Medium / High**).")

model, scaler, le = load_artifacts()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=15, max_value=30, value=20, step=1)
    attendance_pct = st.slider("Attendance (%)", 40.0, 100.0, 82.0, 0.5)
    study_hours = st.slider("Study Hours / Week", 0.0, 45.0, 18.5, 0.5)
    previous_score = st.slider("Previous Score (0-100)", 0.0, 100.0, 71.0, 0.5)
    assignment_pct = st.slider("Assignment Completion (%)", 0.0, 100.0, 80.0, 0.5)

with col2:
    internal_score = st.slider("Internal Assessment Score (0-100)", 0.0, 100.0, 68.0, 0.5)
    participation = st.slider("Participation Score (1-10)", 1.0, 10.0, 7.0, 0.1)
    sleep_hours = st.slider("Sleep Hours / Day", 3.0, 12.0, 7.0, 0.1)
    extracurricular = st.slider("Extracurricular Hours / Week", 0.0, 20.0, 5.0, 0.5)
    failed_subjects = st.number_input("Failed Subjects", min_value=0, max_value=10, value=0, step=1)

if st.button("Predict Performance Category", use_container_width=True):
    features = np.array([[
        age, attendance_pct, study_hours, previous_score,
        assignment_pct, internal_score, participation,
        sleep_hours, extracurricular, failed_subjects
    ]])
    features_scaled = scaler.transform(features)
    probs = model.predict(features_scaled, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx]) * 100

    st.subheader(f"Predicted Performance: **{pred_label}**")
    st.metric(label="Prediction Probability / Confidence", value=f"{confidence:.1f}%")

    st.write("**Class Probability Breakdown:**")
    prob_dict = {le.classes_[i]: f"{probs[i]*100:.2f}%" for i in range(len(le.classes_))}
    st.json(prob_dict)