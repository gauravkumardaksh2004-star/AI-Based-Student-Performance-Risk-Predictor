from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import joblib
from tensorflow import keras

app = FastAPI(
    title="Student Performance ANN API",
    description="REST API for multi-class student performance prediction using Keras ANN",
    version="1.0.0"
)

model = keras.models.load_model("models/student_performance.keras")
scaler = joblib.load("models/scaler.pkl")
le = joblib.load("models/label_encoder.pkl")

class StudentInput(BaseModel):
    age: int = Field(20, ge=15, le=35, description="Student age in years")
    attendance_pct: float = Field(82.0, ge=0.0, le=100.0)
    study_hours_per_week: float = Field(18.5, ge=0.0, le=60.0)
    previous_score: float = Field(71.0, ge=0.0, le=100.0)
    assignment_completion_pct: float = Field(80.0, ge=0.0, le=100.0)
    internal_assessment_score: float = Field(68.0, ge=0.0, le=100.0)
    participation_score: float = Field(7.0, ge=1.0, le=10.0)
    sleep_hours_per_day: float = Field(7.0, ge=1.0, le=16.0)
    extracurricular_hours_per_week: float = Field(5.0, ge=0.0, le=40.0)
    failed_subjects: int = Field(0, ge=0, le=10)

@app.get("/")
def health_check():
    return {"status": "online", "model": "student_performance.keras"}

@app.post("/predict")
def predict_performance(student: StudentInput):
    x = np.array([[
        student.age,
        student.attendance_pct,
        student.study_hours_per_week,
        student.previous_score,
        student.assignment_completion_pct,
        student.internal_assessment_score,
        student.participation_score,
        student.sleep_hours_per_day,
        student.extracurricular_hours_per_week,
        student.failed_subjects
    ]])
    x_scaled = scaler.transform(x)
    probs = model.predict(x_scaled, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = str(le.inverse_transform([pred_idx])[0])

    return {
        "predicted_performance": pred_class,
        "prediction_probability_pct": round(float(probs[pred_idx]) * 100, 2),
        "class_probabilities": {
            str(le.classes_[i]): round(float(probs[i]), 4)
            for i in range(len(le.classes_))
        }
    }