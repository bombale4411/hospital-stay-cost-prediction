import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hospital Stay Cost Prediction",
    page_icon="🏥",
    layout="wide"
)

# =========================================================
# LOAD FILES
# =========================================================

model = joblib.load("Hospital_Cost_Model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

dataset = pd.read_csv("Hospital_Stay_Cost_Dataset.csv")


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🏥 Hospital Cost Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Data Understanding",
        "📈 EDA & Analysis",
        "🤖 Model Information",
        "🔮 Cost Prediction",
        "ℹ️ About Project"
    ]
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title("🏥 Hospital Stay Cost Prediction")

    st.subheader("Predict Hospital Stay Cost Using Machine Learning")

    st.write(
        """
        This project predicts the estimated hospital stay cost
        based on patient and hospitalization details.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(dataset))

    with col2:
        st.metric("Features", 8)

    with col3:
        st.metric("ML Model", "Gradient Boosting")

    st.divider()

    st.subheader("📌 Project Workflow")

    st.write(
        """
        1. Data Collection  
        2. Data Understanding  
        3. Data Cleaning  
        4. Exploratory Data Analysis  
        5. Feature Preparation  
        6. Model Training  
        7. Model Evaluation  
        8. Hospital Cost Prediction
        """
    )


# =========================================================
# DATA UNDERSTANDING
# =========================================================

elif page == "📊 Data Understanding":

    st.title("📊 Data Understanding")

    st.write(
        """
        The dataset contains patient information and hospital
        stay details used to predict the hospital cost.
        """
    )

    st.subheader("Dataset Preview")

    st.dataframe(dataset.head(10), use_container_width=True)

    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"Rows: {dataset.shape[0]}")

    with col2:
        st.info(f"Columns: {dataset.shape[1]}")

    st.subheader("Column Information")

    column_info = pd.DataFrame({
        "Column Name": dataset.columns,
        "Data Type": dataset.dtypes.astype(str)
    })

    st.dataframe(column_info, use_container_width=True)

    st.subheader("Missing Values")

    missing_values = dataset.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values
    })

    st.dataframe(missing_df, use_container_width=True)

    st.subheader("Statistical Summary")

    st.dataframe(
        dataset.describe(include="all").T,
        use_container_width=True
    )


# =========================================================
# EDA & ANALYSIS
# =========================================================

elif page == "📈 EDA & Analysis":

    st.title("📈 Exploratory Data Analysis")

    st.write(
        """
        Exploratory Data Analysis helps us understand patterns,
        distributions and relationships in the hospital dataset.
        """
    )

    # Age Distribution
    st.subheader("👤 Age Distribution")

    fig, ax = plt.subplots()

    ax.hist(dataset["Age"], bins=10)

    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Age Distribution")

    st.pyplot(fig)

    # Length of Stay
    st.subheader("🛏️ Length of Stay Distribution")

    fig, ax = plt.subplots()

    ax.hist(dataset["Length_of_Stay"], bins=10)

    ax.set_xlabel("Length of Stay")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Length of Stay Distribution")

    st.pyplot(fig)

    # Department
    st.subheader("🏥 Patients by Department")

    department_count = dataset["Department"].value_counts()

    fig, ax = plt.subplots()

    department_count.plot(kind="bar", ax=ax)

    ax.set_xlabel("Department")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Department-wise Patient Count")

    st.pyplot(fig)

    # Average Cost by Department
    st.subheader("💰 Average Hospital Cost by Department")

    department_cost = dataset.groupby(
        "Department"
    )["Target_Column"].mean().sort_values()

    fig, ax = plt.subplots()

    department_cost.plot(kind="bar", ax=ax)

    ax.set_xlabel("Department")
    ax.set_ylabel("Average Cost")
    ax.set_title("Average Hospital Cost by Department")

    st.pyplot(fig)


# =========================================================
# MODEL INFORMATION
# =========================================================

elif page == "🤖 Model Information":

    st.title("🤖 Model Information")

    st.subheader("Selected Machine Learning Model")

    st.success("Gradient Boosting Regressor")

    st.write(
        """
        Gradient Boosting Regressor is a supervised machine learning
        algorithm used for regression problems.

        It builds multiple decision trees sequentially.
        Each new tree attempts to reduce the errors made by
        the previous trees.
        """
    )

    st.subheader("🎯 Target Variable")

    st.write("Target_Column")

    st.subheader("📌 Input Features")

    features = [
        "Age",
        "Gender",
        "Department",
        "Length_of_Stay",
        "Surgery",
        "Insurance",
        "Room_Type",
        "Previous_Admissions"
    ]

    for feature in features:
        st.write("•", feature)

    st.subheader("🔄 How the Model Works")

    st.write(
        """
        Patient Information
                ↓
        Data Preprocessing
                ↓
        Categorical Encoding
                ↓
        Feature Matching
                ↓
        Gradient Boosting Model
                ↓
        Predicted Hospital Cost
        """
    )


# =========================================================
# COST PREDICTION
# =========================================================

elif page == "🔮 Cost Prediction":

    st.title("🔮 Hospital Stay Cost Prediction")

    st.write(
        "Enter patient details to predict the estimated hospital stay cost."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            ["M", "F"]
        )

        department = st.selectbox(
            "Department",
            [
                "Cardiology",
                "Orthopedics",
                "ICU",
                "Neurology",
                "General"
            ]
        )

        length_of_stay = st.number_input(
            "Length of Stay (Days)",
            min_value=1,
            max_value=100,
            value=5
        )

    with col2:

        surgery = st.selectbox(
            "Surgery",
            [True, False]
        )

        insurance = st.selectbox(
            "Insurance",
            [True, False]
        )

        room_type = st.selectbox(
            "Room Type",
            [
                "General",
                "Semi-Private",
                "Private"
            ]
        )

        previous_admissions = st.number_input(
            "Previous Admissions",
            min_value=0,
            max_value=20,
            value=0
        )

    st.divider()

    if st.button(
        "🔮 Predict Hospital Stay Cost",
        use_container_width=True
    ):

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

        # Encode categorical variables
        input_data = pd.get_dummies(input_data)

        # Match training columns
        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Prediction
        prediction = model.predict(input_data)[0]

        st.success(
            f"💰 Predicted Hospital Stay Cost: ₹ {prediction:,.2f}"
        )


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Project")

    st.subheader("Hospital Stay Cost Prediction")

    st.write(
        """
        This Machine Learning project predicts hospital stay cost
        using patient and hospitalization-related information.

        The project uses supervised machine learning and a
        Gradient Boosting Regressor for cost prediction.
        """
    )

    st.subheader("🛠️ Technologies Used")

    st.write(
        """
        • Python  
        • Pandas  
        • NumPy  
        • Scikit-learn  
        • Joblib  
        • Streamlit  
        • Matplotlib
        """
    )

    st.subheader("📌 Project Objective")

    st.write(
        """
        The main objective is to estimate hospital stay cost
        using important patient and hospitalization features.
        """
    )

    st.success(
        "🏥 Hospital Stay Cost Prediction using Machine Learning"
    )

  
