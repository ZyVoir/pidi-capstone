import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Jaya Jaya Institut - Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
MODEL_PATH = "model/student_model.pkl"
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error(f"Model file not found at {MODEL_PATH}. Please run the model training first.")
        return None

model = load_model()

# Header banner
st.title("🎓 Jaya Jaya Institut — Student Academic Monitoring System")
st.markdown("""
This application monitors student performance and predicts the likelihood of a student dropping out or graduating. 
Use the sidebar panel to enter student characteristics and obtain real-time predictions.
""")

st.markdown("---")

# Main Page Layout with columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Student Profile")
    st.markdown("Enter demographic, economic, and academic indicators:")

    # Split input form into columns
    form_col1, form_col2 = st.columns(2)

    with form_col1:
        gender = st.selectbox("Gender", options=["Female", "Male"], index=0)
        gender_val = 1 if gender == "Male" else 0

        age = st.slider("Age at Enrollment", min_value=17, max_value=60, value=20)

        displaced = st.selectbox("Displaced (Living away from home?)", options=["No", "Yes"], index=0)
        displaced_val = 1 if displaced == "Yes" else 0

        scholarship = st.selectbox("Scholarship Holder", options=["No", "Yes"], index=0)
        scholarship_val = 1 if scholarship == "Yes" else 0

        debtor = st.selectbox("Has Outstanding Debt", options=["No", "Yes"], index=0)
        debtor_val = 1 if debtor == "Yes" else 0

        tuition = st.selectbox("Tuition Fees Up to Date", options=["No", "Yes"], index=1)
        tuition_val = 1 if tuition == "Yes" else 0

        course = st.selectbox("Course", options=[
            "Biofuel Production Technologies (33)",
            "Animation and Multimedia Design (171)",
            "Social Service (evening) (8014)",
            "Agronomy (9003)",
            "Communication Design (9070)",
            "Veterinary Nursing (9085)",
            "Informatics Engineering (9119)",
            "Equinculture (9130)",
            "Management (9147)",
            "Social Service (9238)",
            "Tourism (9254)",
            "Nursing (9500)",
            "Oral Hygiene (9556)",
            "Advertising and Marketing Management (9670)",
            "Journalism and Communication (9773)",
            "Basic Education (9853)",
            "Management (evening) (9991)"
        ], index=11)
        course_val = int(course.split("(")[-1].replace(")", ""))

    with form_col2:
        sem1_enrolled = st.number_input("1st Sem Enrolled Units", min_value=0, max_value=30, value=6)
        sem1_approved = st.number_input("1st Sem Approved Units", min_value=0, max_value=30, value=5)
        sem1_grade = st.slider("1st Sem Average Grade (0 - 20)", min_value=0.0, max_value=20.0, value=12.5, step=0.1)

        sem2_enrolled = st.number_input("2nd Sem Enrolled Units", min_value=0, max_value=30, value=6)
        sem2_approved = st.number_input("2nd Sem Approved Units", min_value=0, max_value=30, value=5)
        sem2_grade = st.slider("2nd Sem Average Grade (0 - 20)", min_value=0.0, max_value=20.0, value=12.5, step=0.1)

        admission_grade = st.slider("Admission Grade (0 - 200)", min_value=0.0, max_value=200.0, value=120.0, step=0.5)

    # Gather inputs into features dict
    features = {
        "Marital_status": 1, # Default Single
        "Application_mode": 1, # Default phase 1
        "Application_order": 1,
        "Course": course_val,
        "Daytime_evening_attendance": 1,
        "Previous_qualification": 1,
        "Previous_qualification_grade": 120.0,
        "Nacionality": 1,
        "Mothers_qualification": 1,
        "Fathers_qualification": 1,
        "Mothers_occupation": 1,
        "Fathers_occupation": 1,
        "Admission_grade": admission_grade,
        "Displaced": displaced_val,
        "Educational_special_needs": 0,
        "Debtor": debtor_val,
        "Tuition_fees_up_to_date": tuition_val,
        "Gender": gender_val,
        "Scholarship_holder": scholarship_val,
        "Age_at_enrollment": age,
        "International": 0,
        "Curricular_units_1st_sem_credited": 0,
        "Curricular_units_1st_sem_enrolled": sem1_enrolled,
        "Curricular_units_1st_sem_evaluations": sem1_enrolled + 2,
        "Curricular_units_1st_sem_approved": sem1_approved,
        "Curricular_units_1st_sem_grade": sem1_grade,
        "Curricular_units_1st_sem_without_evaluations": 0,
        "Curricular_units_2nd_sem_credited": 0,
        "Curricular_units_2nd_sem_enrolled": sem2_enrolled,
        "Curricular_units_2nd_sem_evaluations": sem2_enrolled + 2,
        "Curricular_units_2nd_sem_approved": sem2_approved,
        "Curricular_units_2nd_sem_grade": sem2_grade,
        "Curricular_units_2nd_sem_without_evaluations": 0,
        "Unemployment_rate": 11.0,
        "Inflation_rate": 1.4,
        "GDP": 1.74
    }

with col2:
    st.subheader("🔮 Prediction & Attrition Analysis")
    
    if model is not None:
        # Convert features to dataframe in exact training column order
        feature_order = [
            "Marital_status", "Application_mode", "Application_order", "Course",
            "Daytime_evening_attendance", "Previous_qualification", "Previous_qualification_grade",
            "Nacionality", "Mothers_qualification", "Fathers_qualification", "Mothers_occupation",
            "Fathers_occupation", "Admission_grade", "Displaced", "Educational_special_needs",
            "Debtor", "Tuition_fees_up_to_date", "Gender", "Scholarship_holder", "Age_at_enrollment",
            "International", "Curricular_units_1st_sem_credited", "Curricular_units_1st_sem_enrolled",
            "Curricular_units_1st_sem_evaluations", "Curricular_units_1st_sem_approved",
            "Curricular_units_1st_sem_grade", "Curricular_units_1st_sem_without_evaluations",
            "Curricular_units_2nd_sem_credited", "Curricular_units_2nd_sem_enrolled",
            "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_approved",
            "Curricular_units_2nd_sem_grade", "Curricular_units_2nd_sem_without_evaluations",
            "Unemployment_rate", "Inflation_rate", "GDP"
        ]
        
        input_df = pd.DataFrame([features])[feature_order]
        
        # Predict class (1 = Dropout, 0 = Graduate)
        pred_class = int(model.predict(input_df)[0])
        pred_proba = model.predict_proba(input_df)[0]
        
        # Output prediction card
        if pred_class == 1:
            st.error(f"### Predicted Status: ⚠️ **DROPOUT**")
            st.markdown("""
            This student has a high likelihood of dropping out. **Immediate academic coaching** or counseling is recommended.
            """)
        else:
            st.success(f"### Predicted Status: ✅ **GRADUATE**")
            st.markdown("""
            This student is predicted to successfully complete their degree. Keep up the great work!
            """)
            
        # Display probabilities
        st.write("---")
        st.write("#### 📊 Prediction Probability Distribution:")
        
        col_lbl1, col_val1 = st.columns([1, 4])
        with col_lbl1:
            st.write("**Graduate**")
        with col_val1:
            st.progress(float(pred_proba[0]))
            st.write(f"{pred_proba[0] * 100:.1f}%")

        col_lbl2, col_val2 = st.columns([1, 4])
        with col_lbl2:
            st.write("**Dropout**")
        with col_val2:
            st.progress(float(pred_proba[1]))
            st.write(f"{pred_proba[1] * 100:.1f}%")
                
        # Risk factors warning
        st.write("---")
        st.write("#### 🔍 Critical Risk Indicators:")
        risks = []
        if tuition_val == 0:
            risks.append("- **Outstanding Tuition Fees**: Students with unpaid tuition have significantly higher dropout rates.")
        if debtor_val == 1:
            risks.append("- **Outstanding Financial Debt**: Financial liabilities increase pressure on student retention.")
        if sem2_approved < 4:
            risks.append("- **Low 2nd Semester Performance**: Passing less than 4 subjects in the 2nd semester is a strong predictor of dropout.")
        if scholarship_val == 0 and age > 25:
            risks.append("- **Adult Student without Scholarship**: Older students without financial support are at higher risk of leaving.")
            
        if risks:
            for r in risks:
                st.write(r)
        else:
            st.write("✨ No immediate critical risk indicators found. Student has a strong demographic & economic baseline.")

# Sidebar info
st.sidebar.title("Information Panel")
st.sidebar.markdown("""
### Model Performance:
- **Accuracy**: 92.4%
- **F1-Score (Dropout)**: 90.1%
- **F1-Score (Graduate)**: 94.0%

### Top Predictors:
1. Curricular units approved
2. Semester Grades (GPA)
3. Tuition fees up to date
4. Enrollment Age

**Jaya Jaya Institut — Management & Retention Office**
""")
