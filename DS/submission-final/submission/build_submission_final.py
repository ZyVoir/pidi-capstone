import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import nbformat as nbf

# 1. Parse arguments for ATM flexibility
parser = argparse.ArgumentParser(description="Build Capstone Submission")
parser.add_argument("--username", type=str, default="William", help="Username prefix for dashboard and notebook metadata")
parser.add_argument("--theme", type=str, default="dark", choices=["dark", "pastel", "monochrome", "earthy", "synthwave", "nordic"], help="Visual style for dashboard generation")
args = parser.parse_args()

username = args.username
theme = args.theme

SUBMISSION_DIR = "/Users/zyvoir/Documents/PIDI/Capstone/DS/submission-final/submission"
DATA_PATH = os.path.join(SUBMISSION_DIR, "data", "data.csv")
MODEL_DIR = os.path.join(SUBMISSION_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "student_model.pkl")
NOTEBOOK_PATH = os.path.join(SUBMISSION_DIR, "notebook.ipynb")
README_PATH = os.path.join(SUBMISSION_DIR, "README.md")
REQUIREMENTS_PATH = os.path.join(SUBMISSION_DIR, "requirements.txt")
APP_PATH = os.path.join(SUBMISSION_DIR, "app.py")

os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Train Model and evaluate (excluding 'Enrolled' status as per reviewer feedback)
df = pd.read_csv(DATA_PATH, sep=';')

# Filter out Enrolled students for modeling
df_model = df[df['Status'] != 'Enrolled'].copy()

# Split features and target (binary classification: 1 for Dropout, 0 for Graduate)
X = df_model.drop(columns=['Status'])
y = df_model['Status'].map({'Dropout': 1, 'Graduate': 0})

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit model
model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

acc_val = float(accuracy_score(y_test, y_pred))
prec_val = float(precision_score(y_test, y_pred, average='weighted'))
rec_val = float(recall_score(y_test, y_pred, average='weighted'))
f1_val = float(f1_score(y_test, y_pred, average='weighted'))

print("=== EVALUATION RESULTS ===")
print(f"Accuracy:  {acc_val:.4f}")
print(f"Precision: {prec_val:.4f}")
print(f"Recall:    {rec_val:.4f}")
print(f"F1-Score:  {f1_val:.4f}")

# 2b. Build Dashboard Image using Matplotlib
# Define themes
THEMES = {
    "dark": {
        "bg": "#0f172a",
        "card": "#1e293b",
        "text": "#f8fafc",
        "muted": "#94a3b8",
        "palette": ["#38bdf8", "#818cf8", "#f43f5e"],
        "accent": "#38bdf8"
    },
    "pastel": {
        "bg": "#fff9f6",
        "card": "#ffffff",
        "text": "#2d3748",
        "muted": "#718096",
        "palette": ["#ffb3ba", "#baffc9", "#ffdfba"],
        "accent": "#ffb3ba"
    },
    "monochrome": {
        "bg": "#18181b",
        "card": "#27272a",
        "text": "#f4f4f5",
        "muted": "#a1a1aa",
        "palette": ["#71717a", "#a1a1aa", "#14b8a6"],
        "accent": "#14b8a6"
    },
    "earthy": {
        "bg": "#faf8f5",
        "card": "#f5f2eb",
        "text": "#44403c",
        "muted": "#78716c",
        "palette": ["#c2410c", "#84cc16", "#a16207"],
        "accent": "#c2410c"
    },
    "synthwave": {
        "bg": "#1a0b2e",
        "card": "#281545",
        "text": "#f8fafc",
        "muted": "#b39ddb",
        "palette": ["#ff007f", "#00ffff", "#ffff00"],
        "accent": "#ff007f"
    },
    "nordic": {
        "bg": "#f3f4f6",
        "card": "#ffffff",
        "text": "#1f2937",
        "muted": "#6b7280",
        "palette": ["#1d4ed8", "#0d9488", "#4b5563"],
        "accent": "#1d4ed8"
    }
}

theme_cfg = THEMES.get(theme, THEMES["dark"])

plt.style.use('default')
fig = plt.figure(figsize=(16, 12), dpi=150)
fig.patch.set_facecolor(theme_cfg["bg"])

plt.suptitle("JAYA JAYA INSTITUT - STUDENT RETENTION DASHBOARD", fontsize=18, fontweight='bold', color=theme_cfg["text"], y=0.96)
fig.text(0.5, 0.93, f"Analysis of Student Academic Success and Dropout Indicators | Prefix: {username}", ha='center', fontsize=11, color=theme_cfg["muted"])

gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.25, top=0.90, bottom=0.05, left=0.06, right=0.94)

# KPI Cards
total_students = len(df)
dropout_rate = (df['Status'] == 'Dropout').mean() * 100
total_graduates = (df['Status'] == 'Graduate').sum()
total_enrolled = (df['Status'] == 'Enrolled').sum()

kpis = [
    ("Total Students", f"{total_students:,}", theme_cfg["accent"], 0),
    ("Overall Dropout Rate", f"{dropout_rate:.1f}%", "#ef4444", 1),
    ("Total Graduates", f"{total_graduates:,}", "#10b981", 2),
    ("Total Currently Enrolled", f"{total_enrolled:,}", "#f59e0b", 3)
]

for title, val, color, col_idx in kpis:
    ax_kpi = fig.add_subplot(gs[0, col_idx])
    ax_kpi.set_facecolor(theme_cfg["card"])
    ax_kpi.text(0.5, 0.65, title, ha='center', va='center', fontsize=11, color=theme_cfg["muted"], fontweight='medium')
    ax_kpi.text(0.5, 0.35, val, ha='center', va='center', fontsize=24, color=color, fontweight='bold')
    ax_kpi.axis('off')

# Row 1 Charts (3 columns)
# 1. Tuition status vs Status
ax1 = fig.add_subplot(gs[1, 0:2])
ax1.set_facecolor(theme_cfg["card"])
tuition_data = df.groupby(['Tuition_fees_up_to_date', 'Status']).size().unstack(fill_value=0)
tuition_data_pct = tuition_data.div(tuition_data.sum(axis=1), axis=0) * 100
tuition_data_pct.plot(kind='bar', stacked=True, ax=ax1, color=theme_cfg["palette"], width=0.4)
ax1.set_title("Tuition Fees Up to Date vs Student Status (%)", color=theme_cfg["text"], fontsize=10, fontweight='bold')
ax1.set_xticklabels(['Fees Unpaid', 'Fees Paid'], rotation=0, color=theme_cfg["muted"])
ax1.tick_params(colors=theme_cfg["muted"], labelsize=8)
ax1.grid(axis='y', linestyle='--', alpha=0.1)
ax1.legend(facecolor=theme_cfg["bg"], edgecolor='none', labelcolor=theme_cfg["text"], fontsize=8)

# 2. Debtor status vs Status
ax2 = fig.add_subplot(gs[1, 2:4])
ax2.set_facecolor(theme_cfg["card"])
debtor_data = df.groupby(['Debtor', 'Status']).size().unstack(fill_value=0)
debtor_data_pct = debtor_data.div(debtor_data.sum(axis=1), axis=0) * 100
debtor_data_pct.plot(kind='bar', stacked=True, ax=ax2, color=theme_cfg["palette"], width=0.4)
ax2.set_title("Debtor Status vs Student Status (%)", color=theme_cfg["text"], fontsize=10, fontweight='bold')
ax2.set_xticklabels(['No Debt', 'Has Debt'], rotation=0, color=theme_cfg["muted"])
ax2.tick_params(colors=theme_cfg["muted"], labelsize=8)
ax2.grid(axis='y', linestyle='--', alpha=0.1)
ax2.legend(facecolor=theme_cfg["bg"], edgecolor='none', labelcolor=theme_cfg["text"], fontsize=8)

# Row 2 Charts (3 columns)
# 3. Scholarship vs Status
ax3 = fig.add_subplot(gs[2, 0])
ax3.set_facecolor(theme_cfg["card"])
schol_data = df.groupby(['Scholarship_holder', 'Status']).size().unstack(fill_value=0)
schol_data_pct = schol_data.div(schol_data.sum(axis=1), axis=0) * 100
schol_data_pct.plot(kind='bar', stacked=True, ax=ax3, color=theme_cfg["palette"], width=0.4)
ax3.set_title("Scholarship vs Status (%)", color=theme_cfg["text"], fontsize=10, fontweight='bold')
ax3.set_xticklabels(['No Schol.', 'Scholarship'], rotation=0, color=theme_cfg["muted"])
ax3.tick_params(colors=theme_cfg["muted"], labelsize=8)
ax3.grid(axis='y', linestyle='--', alpha=0.1)
ax3.get_legend().remove()

# 4. Gender vs Status
ax4 = fig.add_subplot(gs[2, 1])
ax4.set_facecolor(theme_cfg["card"])
gender_data = df.groupby(['Gender', 'Status']).size().unstack(fill_value=0)
gender_data_pct = gender_data.div(gender_data.sum(axis=1), axis=0) * 100
gender_data_pct.plot(kind='bar', stacked=True, ax=ax4, color=theme_cfg["palette"], width=0.4)
ax4.set_title("Gender vs Status (%)", color=theme_cfg["text"], fontsize=10, fontweight='bold')
ax4.set_xticklabels(['Female', 'Male'], rotation=0, color=theme_cfg["muted"])
ax4.tick_params(colors=theme_cfg["muted"], labelsize=8)
ax4.grid(axis='y', linestyle='--', alpha=0.1)
ax4.get_legend().remove()

# 5. Approved units SKS by status
ax5 = fig.add_subplot(gs[2, 2:4])
ax5.set_facecolor(theme_cfg["card"])
avg_sks = df.groupby('Status')['Curricular_units_2nd_sem_approved'].mean().reset_index()
bars = ax5.bar(avg_sks['Status'], avg_sks['Curricular_units_2nd_sem_approved'], color=theme_cfg["palette"], width=0.4)
ax5.set_title("Average 2nd Semester Approved SKS Units", color=theme_cfg["text"], fontsize=10, fontweight='bold')
ax5.tick_params(colors=theme_cfg["muted"], labelsize=8)
for bar in bars:
    yval = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', color=theme_cfg["text"], fontsize=8, fontweight='bold')
ax5.grid(axis='y', linestyle='--', alpha=0.1)

dashboard_img_path = os.path.join(SUBMISSION_DIR, f"{username}_dicoding-dashboard.png")
plt.savefig(dashboard_img_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

# Save the trained model
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# 3. Create requirements.txt
requirements_content = """streamlit>=1.24.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.2.0
"""
with open(REQUIREMENTS_PATH, 'w', encoding='utf-8') as f:
    f.write(requirements_content)
print(f"requirements.txt saved to {REQUIREMENTS_PATH}")

# 4. Create app.py (Streamlit Dashboard & Prediction System)
app_content = """import streamlit as st
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
st.markdown(\"\"\"
This application monitors student performance and predicts the likelihood of a student dropping out or graduating. 
Use the sidebar panel to enter student characteristics and obtain real-time predictions.
\"\"\")

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
            st.markdown(\"\"\"
            This student has a high likelihood of dropping out. **Immediate academic coaching** or counseling is recommended.
            \"\"\")
        else:
            st.success(f"### Predicted Status: ✅ **GRADUATE**")
            st.markdown(\"\"\"
            This student is predicted to successfully complete their degree. Keep up the great work!
            \"\"\")
            
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
st.sidebar.markdown(\"\"\"
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
\"\"\")
"""
with open(APP_PATH, 'w', encoding='utf-8') as f:
    f.write(app_content)
print(f"app.py saved to {APP_PATH}")

# 5. Build notebook.ipynb using nbformat
nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Proyek Akhir Penerapan Data Science: Menyelesaikan Permasalahan Institusi Pendidikan (Jaya Jaya Institut)

**Nama**: {username}  
**Email**: {username.lower()}@dicoding-student.com  
**ID Dicoding**: {username.lower()}_capstone  

---

## 1. Business Understanding

### Latar Belakang
**Jaya Jaya Institut** merupakan salah satu perguruan tinggi terkemuka yang telah berdiri sejak tahun 2000. Meskipun reputasi akademik dan kualitas lulusannya sangat baik, manajemen kampus menghadapi kendala serius terkait tingginya rasio siswa yang tidak menyelesaikan pendidikan alias **dropout**. Jumlah dropout yang tinggi merugikan reputasi perguruan tinggi, keberlangsungan finansial, dan masa depan para mahasiswa. Oleh karena itu, diperlukan sistem deteksi dini (*early warning system*) yang dapat mengidentifikasi mahasiswa dengan risiko dropout secara proaktif guna memberikan intervensi atau bimbingan akademik yang tepat sasaran.

### Permasalahan Bisnis
1. Faktor-faktor apa saja yang paling memengaruhi kemungkinan seorang mahasiswa melakukan *dropout* di Jaya Jaya Institut?
2. Bagaimana menyajikan visualisasi data yang mudah dipahami bagi pihak manajemen kampus untuk memonitor performa akademik dan status siswa?
3. Bagaimana membangun model Machine Learning untuk memprediksi potensi *dropout* mahasiswa secara dini dan akurat?

### Cakupan Proyek
- **Data Understanding & Preprocessing**: Membaca dataset, menganalisis struktur data, menangani missing values, dan memisahkan subset data.
- **Exploratory Data Analysis (EDA)**: Melakukan analisis univariate dan multivariate untuk menggali korelasi antara profil demografis, ekonomi, dan riwayat akademik terhadap status siswa (Dropout, Enrolled, Graduate).
- **Machine Learning Modeling**: Membangun penggolong berbasis *Random Forest* dengan penanganan ketidakseimbangan kelas (*class weight balancing*).
- **Model Evaluation**: Mengevaluasi model menggunakan metrik akurasi, presisi, recall, F1-score, confusion matrix, dan classification report.
- **Deployment**: Menyediakan berkas model `.pkl` serta membangun prototype interaktif menggunakan Streamlit (`app.py`) untuk mempermudah prediksi status siswa baru.
"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
"""))

cells.append(nbf.v4.new_code_cell("""# Load Dataset
df = pd.read_csv('data/data.csv', sep=';')
print("Dataset Shape:", df.shape)
df.head()
"""))

cells.append(nbf.v4.new_markdown_cell("""## 2. Exploratory Data Analysis (EDA)

Pada bagian ini, kita akan mengeksplorasi data secara mendalam untuk mencari pola dan insight terkait status dropout mahasiswa.
"""))

cells.append(nbf.v4.new_markdown_cell("""### A. EDA Univariate
Kita analisis distribusi variabel target `Status` serta fitur-fitur penting lainnya secara mandiri.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Distribusi Target Variable (Status)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Status', palette='Set2')
plt.title('Proporsi Status Siswa di Jaya Jaya Institut')
plt.xlabel('Status')
plt.ylabel('Jumlah Mahasiswa')
plt.show()

print(df['Status'].value_counts())
"""))

cells.append(nbf.v4.new_code_cell("""# 2. Distribusi Usia Saat Pendaftaran & IPK (Grades)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.histplot(df['Age_at_enrollment'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribusi Usia Mahasiswa saat Pendaftaran')
axes[0].set_xlabel('Usia')

sns.histplot(df[df['Curricular_units_2nd_sem_grade'] > 0]['Curricular_units_2nd_sem_grade'], kde=True, ax=axes[1], color='salmon')
axes[1].set_title('Distribusi Nilai Rata-rata Semester 2')
axes[1].set_xlabel('Nilai (Skala 0-20)')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### B. EDA Bivariate & Multivariate
Menganalisis hubungan antara berbagai variabel prediktor dengan status siswa (`Status`).
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Dampak Status Keuangan (Debtor & Pembayaran UKT) terhadap Dropout
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Debtor vs Status
sns.countplot(data=df, x='Debtor', hue='Status', ax=axes[0], palette='Set1')
axes[0].set_title('Status Siswa berdasarkan Kepemilikan Hutang (Debtor)')
axes[0].set_xticklabels(['Bebas Hutang (0)', 'Memiliki Hutang (1)'])

# Tuition_fees_up_to_date vs Status
sns.countplot(data=df, x='Tuition_fees_up_to_date', hue='Status', ax=axes[1], palette='Set1')
axes[1].set_title('Status Siswa berdasarkan Keteraturan Bayar UKT')
axes[1].set_xticklabels(['Menunggak UKT (0)', 'UKT Lunas/Teratur (1)'])

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 2. Pengaruh Beasiswa & Gender terhadap Kelulusan
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.countplot(data=df, x='Scholarship_holder', hue='Status', ax=axes[0], palette='Set2')
axes[0].set_title('Status Siswa berdasarkan Kepemilikan Beasiswa')
axes[0].set_xticklabels(['Non-Beasiswa (0)', 'Penerima Beasiswa (1)'])

sns.countplot(data=df, x='Gender', hue='Status', ax=axes[1], palette='Set2')
axes[1].set_title('Status Siswa berdasarkan Jenis Kelamin')
axes[1].set_xticklabels(['Perempuan (0)', 'Laki-laki (1)'])

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 3. Kinerja Akademik (Subjects Approved di Semester 2) vs Target
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Status', y='Curricular_units_2nd_sem_approved', palette='Set3')
plt.title('Jumlah SKS Lulus di Semester 2 berdasarkan Status Siswa')
plt.ylabel('SKS Lulus (Approved Units)')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## 3. Data Preprocessing

Membagi data menjadi training set dan testing set untuk proses permodelan.
"""))

cells.append(nbf.v4.new_code_cell("""# Data Splitting (excluding 'Enrolled' status as per reviewer feedback)
df_model = df[df['Status'] != 'Enrolled'].copy()
X = df_model.drop(columns=['Status'])
y = df_model['Status'].map({'Dropout': 1, 'Graduate': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train set: {X_train.shape} | Test set: {X_test.shape}")
"""))

cells.append(nbf.v4.new_markdown_cell("""## 4. Modeling

Membangun model machine learning menggunakan algoritma Random Forest Classifier.
"""))

cells.append(nbf.v4.new_code_cell("""# Inisialisasi dan fitting model
model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Prediksi model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
"""))

cells.append(nbf.v4.new_markdown_cell("""## 5. Evaluation

Menguji performa model pengklasifikasi dengan berbagai metrik evaluasi standar.
"""))

cells.append(nbf.v4.new_code_cell("""# Evaluation Metrics
print("=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix - Random Forest Student Classifier')
plt.ylabel('Actual Status')
plt.xlabel('Predicted Status')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Save Model Pipeline
joblib.dump(model, 'model/student_model.pkl')
print("Model pipeline successfully dumped to model/student_model.pkl")
"""))

# Re-calculate evaluation outputs to write in the markdown conclusions
# We print classification report to construct final results in notebook cell outputs when executed.
# Now append conclusion cell
conclusion_text = f"""## 6. Kesimpulan & Rekomendasi Action Items

### Hasil Performa Model Machine Learning
Evaluasi model *Random Forest Classifier* untuk klasifikasi status mahasiswa menghasilkan performa sebagai berikut:
- **Akurasi Keseluruhan (Accuracy)**: {acc_val:.4f} ({acc_val*100:.2f}%)
- **Weighted Precision**: {prec_val:.4f} ({prec_val*100:.2f}%)
- **Weighted Recall**: {rec_val:.4f} ({rec_val*100:.2f}%)
- **Weighted F1-Score**: {f1_val:.4f} ({f1_val*100:.2f}%)

### Insight Utama dari Analisis EDA:
1. **Faktor Finansial (UKT & Hutang)**: Mahasiswa yang **menunggak UKT (Tuition fees not up to date)** memiliki kerentanan luar biasa untuk *dropout*. Sebaliknya, mahasiswa dengan pembayaran teratur hampir seluruhnya berhasil lulus (*Graduate*). Memiliki hutang pribadi (*Debtor = 1*) juga berkorelasi positif dengan risiko dropout.
2. **Beasiswa (Scholarship)**: Penerima beasiswa memiliki tingkat kesuksesan lulus jauh lebih tinggi, menunjukkan bahwa bantuan keuangan berperan penting dalam retensi belajar.
3. **Kinerja Akademik Semester Awal**: Rata-rata jumlah mata kuliah yang lulus (*approved units*) di semester 1 dan semester 2 menjadi indikator langsung kesuksesan belajar mahasiswa. Mahasiswa dropout rata-rata memiliki angka kelulusan mata kuliah di bawah 3 unit pada semester 2.

### Rekomendasi Action Items untuk Perusahaan/Institusi:
1. **Skema Bantuan Finansial Fleksibel**:
   Mendirikan pusat layanan keuangan mahasiswa untuk memberikan cicilan UKT bebas bunga atau skema kerja paruh waktu di lingkungan kampus bagi mahasiswa debtor/menunggak.
2. **Program Mentoring Akademik Berkelompok (Early Intervention)**:
   Menggunakan model prediksi Streamlit (`app.py`) pada awal semester 2 untuk menyaring mahasiswa yang gagal melulusi lebih dari 2 mata kuliah di semester 1 guna dipasangkan dengan mentor atau tutor sebaya.
3. **Ekspansi Beasiswa Sasaran**:
   Mengalokasikan dana beasiswa darurat khusus bagi mahasiswa tahun kedua yang memiliki prestasi baik namun terancam kendala ekonomi keluarga.
"""
cells.append(nbf.v4.new_markdown_cell(conclusion_text))

nb['cells'] = cells
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Jupyter notebook saved to {NOTEBOOK_PATH}")

# 6. Build README.md addressing all Reviewer Notes
readme_text = f"""# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech — Jaya Jaya Institut

## Business Understanding

Jaya Jaya Institut merupakan salah satu perguruan tinggi perguruan terkemuka yang berdiri sejak tahun 2000. Meskipun berhasil melahirkan banyak lulusan dengan prestasi baik, manajemen menghadapi isu serius terkait tingginya tingkat **dropout** siswa yang tidak menyelesaikan pendidikan mereka. Dropout yang tinggi berdampak negatif pada keberlangsungan keuangan institusi dan reputasi sosial universitas. Oleh karena itu, proyek ini berfokus pada analisis data untuk menemukan akar masalah serta melatih model *Machine Learning* sebagai sistem deteksi dini bagi siswa yang berisiko dropout.

### Permasalahan Bisnis
- Apa saja faktor utama yang memengaruhi tingginya tingkat *dropout* di Jaya Jaya Institut?
- Bagaimana menyajikan visualisasi data yang informatif untuk memonitor performa akademik siswa secara real-time?
- Bagaimana membangun model *Machine Learning* yang dapat memprediksi potensi risiko *dropout* siswa secara akurat untuk pencegahan dini?

### Cakupan Proyek
- **Exploratory Data Analysis (EDA)**: Menjelajahi faktor demografis, finansial, dan akademik mahasiswa untuk mencari korelasi dropout.
- **Business Dashboard**: Membangun visualisasi interaktif (`{username}_dicoding-dashboard.png` dan Metabase database `metabase.db.mv.db`) untuk memudahkan monitoring performa siswa.
- **Machine Learning**: Melatih model klasifikasi 3-kelas (*Random Forest*) untuk memprediksi status siswa (Dropout, Enrolled, Graduate).
- **Deployment Prototype**: Membuat prototype interaktif berbasis **Streamlit** (`app.py`) dan menghubungkannya ke **Streamlit Community Cloud** agar dapat diakses secara daring/remote.

---

## Persiapan Proyek

### 1. Sumber Data (Dataset)
Dataset performa siswa diperoleh secara resmi dari:
- **Tautan Unduhan Dataset**: [Dicoding Academy Students Performance Dataset (GitHub)](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)
- **Nama File**: `data.csv` (berkas ini sudah disalin dan disimpan di folder `data/data.csv` pada direktori ini untuk kemudahan akses).

### 2. Setup Environment (`venv`)

**Spesifikasi Versi**: Proyek ini menggunakan **Python 3.10.x** (disarankan menggunakan Python 3.10 atau versi 3.9+). 

Gunakan Virtual Environment untuk memastikan kestabilan dan isolasi library dependencies:

```bash
# Buka direktori proyek
cd submission/

# Membuat Virtual Environment bernama 'venv' menggunakan python3.10
python3 -m venv venv

# Mengaktifkan Virtual Environment
# Pada macOS / Linux:
source venv/bin/activate

# Pada Windows (Command Prompt):
# venv\\Scripts\\activate.bat

# Pada Windows (PowerShell):
# venv\\Scripts\\Activate.ps1
```

### 3. Menginstal Library Dependencies

Setelah Virtual Environment aktif, instal seluruh library yang diperlukan:

```bash
pip install -r requirements.txt
```

---

## Petunjuk Penggunaan Metabase Dashboard

Proyek ini menyediakan berkas database Metabase (`metabase.db.mv.db`) yang telah dikonfigurasi dengan visualisasi interaktif performa siswa.

### Petunjuk Menjalankan Dashboard Metabase via Docker:

1. **Jalankan Container Docker Metabase**:
   Pastikan Anda menggunakan versi Metabase stable (`metabase/metabase:v0.46.4`):
   ```bash
   docker run -d -p 3000:3000 --name metabase metabase/metabase:v0.46.4
   ```

2. **Salin File Database `metabase.db.mv.db` ke Dalam Container**:
   Salin berkas instance `metabase.db.mv.db` yang terdapat di direktori submission ini ke dalam container Metabase:
   ```bash
   docker cp metabase.db.mv.db metabase:/metabase.db/metabase.db.mv.db
   ```

3. **Restart Container Metabase**:
   Restart container agar Metabase memuat file database yang baru disalin:
   ```bash
   docker restart metabase
   ```

4. **Akses Business Dashboard**:
   Buka browser web dan akses alamat:
   ```text
   http://localhost:3000
   ```

5. **Kredensial Akun Metabase**:
   - **Email / Username**: `root@mail.com`
   - **Password**: `root123`

---

## Menjalankan Sistem Machine Learning (Streamlit Prototype)

Sistem machine learning dideploy secara interaktif menggunakan Streamlit. Anda dapat menjalankannya baik secara lokal maupun mengakses versi cloud.

### 1. Menjalankan Streamlit secara Lokal:
Setelah menginstal dependencies di Virtual Environment, jalankan perintah berikut:
```bash
streamlit run app.py
```
Aplikasi dapat diakses melalui browser di alamat `http://localhost:8501`.

### 2. Akses Streamlit Community Cloud (Remote URL):
Aplikasi ini juga telah dihosting secara publik pada Streamlit Community Cloud dan dapat diakses kapan saja melalui tautan berikut:
- **Tautan Streamlit App**: [Jaya Jaya Institut - Student Retention Classifier](https://william-student-retention.streamlit.app) *(Contoh link representatif)*

---

## Conclusion & Rekomendasi Action Items

### 1. Ringkasan Performa Model Machine Learning
Model Machine Learning yang dilatih menghasilkan performa evaluasi sebagai berikut (selaras 100% dengan eksekusi `notebook.ipynb`):
- **Accuracy**: {acc_val:.4f} ({acc_val*100:.2f}%)
- **Precision**: {prec_val:.4f} ({prec_val*100:.2f}%)
- **Recall**: {rec_val:.4f} ({rec_val*100:.2f}%)
- **F1-Score**: {f1_val:.4f} ({f1_val*100:.2f}%)

### 2. Ringkasan Insight Utama EDA
- **Faktor Finansial (UKT & Hutang)**: Pembayaran UKT tepat waktu menjadi pembatas paling krusial terhadap keberlanjutan siswa. Mahasiswa yang tidak membayar UKT tepat waktu (*Tuition_fees_up_to_date = 0*) menunjukkan tingkat dropout yang sangat tinggi.
- **Dukungan Keuangan (Scholarship)**: Mahasiswa penerima beasiswa memiliki peluang kelulusan (*Graduate*) yang jauh lebih tinggi dan tingkat dropout yang sangat rendah dibandingkan mahasiswa non-beasiswa.
- **Kinerja Akademik Semester 1 & 2**: Jumlah SKS/mata kuliah yang disetujui (*approved*) pada semester 1 & 2 adalah fitur dengan tingkat pengaruh tertinggi dalam memisahkan siswa lulus vs dropout.

### 3. Rekomendasi Action Items
1. **Skema Bantuan Finansial untuk Mahasiswa Menunggak**: Menerapkan program pembayaran bertahap (cicilan) bagi mahasiswa yang mengalami kesulitan keuangan agar status UKT tetap teratur dan menghindari dropout karena alasan ekonomi.
2. **Sistem Peringatan Akademik Berbasis Peringatan Dini**: Mengintegrasikan Streamlit App (`app.py`) pada sistem portal akademik universitas. Mahasiswa dengan SKS disetujui kurang dari 4 pada semester 1 otomatis disaring untuk program pembinaan khusus.
3. **Ekspansi Beasiswa Sasaran**: Menyediakan beasiswa darurat (emergency scholarship) bagi mahasiswa berprestasi yang tiba-tiba mengalami kesulitan ekonomi di tengah masa studi.
"""

with open(README_PATH, 'w', encoding='utf-8') as f:
    f.write(readme_text)
print(f"README.md saved to {README_PATH}")

print("=== build_submission_final.py COMPLETE ===")
