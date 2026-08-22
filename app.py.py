import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Hospital Stay Cost Prediction",
    page_icon="🏥"
)

# Load saved model
model = joblib.load("Hospital_Cost_Model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Title
st.title("🏥 Hospital Stay Cost Prediction")
st.write("Enter patient details to predict hospital stay cost.")

st.divider()

# Patient inputs
age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=30
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

department = st.selectbox(
    "Department",
    ["Cardiology", "Neurology", "Orthopedics", "General"]
)

length_of_stay = st.number_input(
    "Length of Stay (Days)",
    min_value=1,
    max_value=100,
    value=5
)

surgery = st.selectbox(
    "Surgery",
    ["Yes", "No"]
)

insurance = st.selectbox(
    "Insurance",
    ["Yes", "No"]
)

room_type = st.selectbox(
    "Room Type",
    ["General", "Semi-Private", "Private"]
)

previous_admissions = st.number_input(
    "Previous Admissions",
    min_value=0,
    max_value=20,
    value=0
)

# Prediction button
if st.button("Predict Hospital Stay Cost"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Department": [department],
        "Length_of_Stay": [length_of_stay],
        "Surgery": [surgery],
        "Insurance": [insurance],
        "Room_Type": [room_type],
        "Previous_Admissions": [previous_admissions]
    })

    # Encode categorical columns
    input_data = pd.get_dummies(input_data)

    # Match training columns
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Hospital Stay Cost: ₹ {prediction:,.2f}"
    )
