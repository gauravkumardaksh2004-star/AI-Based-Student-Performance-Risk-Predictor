# AI-Based Student Performance Risk Predictor (Month 3 Project)

## 1. Problem Overview
This project builds a Feed-Forward Artificial Neural Network (ANN) in Keras/TensorFlow to classify students into three performance risk categories (**Low**, **Medium**, **High**) using 10 academic and behavioral features.

## 2. Dataset Summary
- **Records:** 15,000 students (`data/student_data.csv`), 0 missing values, 0 duplicates.
- **Input Features (10):** `age`, `attendance_pct`, `study_hours_per_week`, `previous_score`, `assignment_completion_pct`, `internal_assessment_score`, `participation_score`, `sleep_hours_per_day`, `extracurricular_hours_per_week`, `failed_subjects`.
- **Target Variable:** `performance_category` (`Low`: 4,950, `Medium`: 5,100, `High`: 4,950).

## 3. Preprocessing & Architecture
- **Preprocessing:** `StandardScaler` fit on 80% stratified training data (12,000 samples) and applied to 20% test data (3,000 samples).
- **ANN Architecture:** Input (10) → Dense (32, ReLU, L2=0.005) → Dense (16, ReLU, L2=0.005) → Output (3, Softmax).
- **Training:** Adam optimizer (`lr=0.001`), `sparse_categorical_crossentropy` loss, batch size 64, EarlyStopping on validation loss.

## 4. Evaluation Summary (Test Set, n = 3,000)
- **Logistic Regression Baseline:** 69.20% Accuracy (Macro F1: 0.6946)
- **Keras ANN:** 69.03% Accuracy (Macro F1: 0.6938)

## 5. Setup & Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt