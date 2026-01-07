import streamlit as st
import numpy as np
import joblib 

model = joblib.load("best_model.pkl")

st.title("Student Exam Score Predictor")

study_hours = st.slider("study hours per day", 0.0, 12.0, 2.0)
attendance = st.slider("attendance percentage", 0.0, 100.0, 80.0)
mental_health = st.slider("mental health rating (1-10)", 1, 10, 7)
sleep_hours = st.slider("Sleep hours per night", 0.0, 12.0, 8.0)
part_time_job = st.selectbox("Part time job", ["No", "yes"])

ptj_encoded = 1 if part_time_job == "Yes" else 0 

if st.button("Predict exam score"):
    input_data = np.array([study_hours, attendance, mental_health, sleep_hours, ptj_encoded])
    input_data_reshaped = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data_reshaped)[0]

    prediction = max(0, min(100, prediction))

    st.success(f"Predicted exam score: {prediction:.2f}")