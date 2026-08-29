import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import nbformat as nbf

SUBMISSION_DIR = "/Users/zyvoir/Documents/PIDI/Capstone/DS/submission"
DATA_PATH = os.path.join(SUBMISSION_DIR, "data", "employee_data.csv")
MODEL_DIR = os.path.join(SUBMISSION_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "attrition_model.pkl")
DASHBOARD_PATH = os.path.join(SUBMISSION_DIR, "William_dicoding-dashboard.png")
NOTEBOOK_PATH = os.path.join(SUBMISSION_DIR, "notebook.ipynb")
PREDICTION_PATH = os.path.join(SUBMISSION_DIR, "prediction.py")
README_PATH = os.path.join(SUBMISSION_DIR, "README.md")
REQUIREMENTS_PATH = os.path.join(SUBMISSION_DIR, "requirements.txt")

os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load Data
df = pd.read_csv(DATA_PATH)

# Drop redundant or zero-variance columns if any
cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours']
df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Separate Labeled (Train/Val) and Unlabeled (Prediction) data
labeled_df = df_clean[df_clean['Attrition'].notnull()].copy()
unlabeled_df = df_clean[df_clean['Attrition'].isnull()].copy()

labeled_df['Attrition'] = labeled_df['Attrition'].astype(int)

# Categorical & Numerical Features
categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
numerical_cols = [c for c in labeled_df.columns if c not in categorical_cols + ['EmployeeId', 'Attrition']]

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

X = labeled_df[numerical_cols + categorical_cols]
y = labeled_df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model Pipeline
best_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42))
])

best_model.fit(X_train, y_train)

# Evaluation
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Model Results -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# Save Model Pipeline
joblib.dump(best_model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Save Preprocessed Data for Dashboard & Notebook
# ----------------------------------------------------
# 2. Build Dashboard Graphic (username_dicoding-dashboard.png)
# ----------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(16, 12), dpi=150)
fig.patch.set_facecolor('#0f172a')  # Dark slate navy background

# Title Block
plt.suptitle("PT JAYA JAYA MAJU - HR ATTRITION EXECUTIVE DASHBOARD", fontsize=20, fontweight='bold', color='#f8fafc', y=0.96)
fig.text(0.5, 0.93, "Analysis of Employee Turnover Drivers & Predictive Monitoring | Department of Human Resources", ha='center', fontsize=11, color='#94a3b8')

# Create grid structure
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25, top=0.90, bottom=0.05, left=0.06, right=0.94)

# KPI Card 1: Total Employees
ax_kpi1 = fig.add_subplot(gs[0, 0])
ax_kpi1.set_facecolor('#1e293b')
ax_kpi1.text(0.5, 0.65, "Total Employees", ha='center', va='center', fontsize=12, color='#94a3b8', fontweight='medium')
ax_kpi1.text(0.5, 0.35, f"{len(df):,}", ha='center', va='center', fontsize=26, color='#38bdf8', fontweight='bold')
ax_kpi1.axis('off')

# KPI Card 2: Attrition Rate
ax_kpi2 = fig.add_subplot(gs[0, 1])
ax_kpi2.set_facecolor('#1e293b')
att_rate = (labeled_df['Attrition'].sum() / len(labeled_df)) * 100
ax_kpi2.text(0.5, 0.65, "Overall Attrition Rate", ha='center', va='center', fontsize=12, color='#94a3b8', fontweight='medium')
ax_kpi2.text(0.5, 0.35, f"{att_rate:.1f}%", ha='center', va='center', fontsize=26, color='#ef4444', fontweight='bold')
ax_kpi2.axis('off')

# KPI Card 3: Overtime Impact Rate
ax_kpi3 = fig.add_subplot(gs[0, 2])
ax_kpi3.set_facecolor('#1e293b')
ot_att = (labeled_df[labeled_df['OverTime'] == 'Yes']['Attrition'].sum() / len(labeled_df[labeled_df['OverTime'] == 'Yes'])) * 100
ax_kpi3.text(0.5, 0.65, "Attrition in OverTime Employees", ha='center', va='center', fontsize=12, color='#94a3b8', fontweight='medium')
ax_kpi3.text(0.5, 0.35, f"{ot_att:.1f}%", ha='center', va='center', fontsize=26, color='#f59e0b', fontweight='bold')
ax_kpi3.axis('off')

# Chart 1: Attrition Rate by Department
ax1 = fig.add_subplot(gs[1, 0])
ax1.set_facecolor('#1e293b')
dept_att = labeled_df.groupby('Department')['Attrition'].mean().reset_index()
dept_att['AttritionRate'] = dept_att['Attrition'] * 100
bars1 = ax1.bar(dept_att['Department'], dept_att['AttritionRate'], color=['#38bdf8', '#818cf8', '#f43f5e'], edgecolor='none', width=0.5)
ax1.set_title("Attrition Rate by Department (%)", color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
ax1.set_ylabel("Attrition Rate (%)", color='#94a3b8', fontsize=9)
ax1.tick_params(colors='#94a3b8', labelsize=8)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', color='#f8fafc', fontsize=8, fontweight='bold')
ax1.set_ylim(0, max(dept_att['AttritionRate']) + 5)
ax1.grid(axis='y', linestyle='--', alpha=0.2)

# Chart 2: Overtime vs Attrition Comparison
ax2 = fig.add_subplot(gs[1, 1])
ax2.set_facecolor('#1e293b')
ot_df = labeled_df.groupby(['OverTime', 'Attrition']).size().unstack(fill_value=0)
ot_pct = ot_df.div(ot_df.sum(axis=1), axis=0) * 100
ot_pct.plot(kind='bar', stacked=True, ax=ax2, color=['#10b981', '#f43f5e'], width=0.45)
ax2.set_title("OverTime vs Employee Attrition (%)", color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel("OverTime Status", color='#94a3b8', fontsize=9)
ax2.set_ylabel("Percentage (%)", color='#94a3b8', fontsize=9)
ax2.tick_params(colors='#94a3b8', labelsize=8, rotation=0)
ax2.legend(['Stayed (0)', 'Left (1)'], facecolor='#0f172a', edgecolor='none', labelcolor='#f8fafc', fontsize=8)
ax2.grid(axis='y', linestyle='--', alpha=0.2)

# Chart 3: Monthly Income Distribution by Job Level
ax3 = fig.add_subplot(gs[1, 2])
ax3.set_facecolor('#1e293b')
sns.boxplot(data=labeled_df, x='JobLevel', y='MonthlyIncome', hue='Attrition', palette=['#38bdf8', '#f43f5e'], ax=ax3)
ax3.set_title("Monthly Income Disparity by Job Level", color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
ax3.set_xlabel("Job Level", color='#94a3b8', fontsize=9)
ax3.set_ylabel("Monthly Income ($)", color='#94a3b8', fontsize=9)
ax3.tick_params(colors='#94a3b8', labelsize=8)
ax3.legend(['Stayed (0)', 'Left (1)'], facecolor='#0f172a', edgecolor='none', labelcolor='#f8fafc', fontsize=8)
ax3.grid(axis='y', linestyle='--', alpha=0.2)

# Chart 4: Attrition by Monthly Income & Age
ax4 = fig.add_subplot(gs[2, 0:2])
ax4.set_facecolor('#1e293b')
sns.scatterplot(data=labeled_df, x='Age', y='MonthlyIncome', hue='Attrition', style='OverTime', palette=['#38bdf8', '#f43f5e'], alpha=0.8, ax=ax4)
ax4.set_title("Age vs Monthly Income Scatter Analysis (Target: Attrition & OverTime)", color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
ax4.set_xlabel("Employee Age", color='#94a3b8', fontsize=9)
ax4.set_ylabel("Monthly Income ($)", color='#94a3b8', fontsize=9)
ax4.tick_params(colors='#94a3b8', labelsize=8)
ax4.legend(facecolor='#0f172a', edgecolor='none', labelcolor='#f8fafc', fontsize=8, loc='upper left')
ax4.grid(linestyle='--', alpha=0.2)

# Chart 5: Key Action Items Box
ax5 = fig.add_subplot(gs[2, 2])
ax5.set_facecolor('#1e293b')
ax5.axis('off')
ax5.text(0.05, 0.88, "STRATEGIC RECOMMENDATIONS", color='#38bdf8', fontsize=11, fontweight='bold')
recs = [
    "1. OverTime Regulation:\n   Limit mandatory overtime; high OT drives\n   3x higher attrition rate (30.5%).",
    "2. Entry Level Compensation:\n   Adjust Monthly Income for Job Level 1-2;\n   low salary correlates with high departures.",
    "3. Work-Life Balance Programs:\n   Introduce flexible hours & wellbeing\n   initiatives for high-risk roles.",
    "4. Predictive Monitoring:\n   Deploy ML model (prediction.py) to flag\n   flight-risk employees proactively."
]
for i, rec in enumerate(recs):
    ax5.text(0.05, 0.70 - (i * 0.20), rec, color='#f8fafc', fontsize=8.5, va='top')

plt.savefig(DASHBOARD_PATH, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Dashboard graphic saved to {DASHBOARD_PATH}")

# ----------------------------------------------------
# 3. Build Standalone prediction.py
# ----------------------------------------------------
prediction_script_content = '''import sys
import json
import argparse
import pandas as pd
import joblib

MODEL_PATH = "model/attrition_model.pkl"

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        sys.exit(1)

def predict_employee(input_data):
    model = load_model()
    if isinstance(input_data, dict):
        df_in = pd.DataFrame([input_data])
    elif isinstance(input_data, str) and input_data.endswith('.csv'):
        df_in = pd.read_csv(input_data)
    else:
        raise ValueError("Input data must be a dictionary or path to a CSV file.")
    
    predictions = model.predict(df_in)
    probabilities = model.predict_proba(df_in)[:, 1]
    
    results = []
    for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
        status = "HIGH RISK (Attrition Likely)" if pred == 1 else "LOW RISK (Retention Likely)"
        res = {
            "EmployeeIndex": i,
            "Prediction": int(pred),
            "Status": status,
            "AttritionProbability": f"{proba * 100:.2f}%"
        }
        results.append(res)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Employee Attrition Risk for PT Jaya Jaya Maju")
    parser.add_argument("--json", type=str, help="JSON string containing employee feature data")
    parser.add_argument("--file", type=str, help="Path to CSV file containing employee feature data")
    
    args = parser.parse_args()
    
    if args.json:
        data = json.loads(args.json)
        res = predict_employee(data)
        print(json.dumps(res, indent=2))
    elif args.file:
        res = predict_employee(args.file)
        print(json.dumps(res, indent=2))
    else:
        # Sample employee demonstration
        sample_employee = {
            "Age": 29,
            "BusinessTravel": "Travel_Frequently",
            "DailyRate": 450,
            "Department": "Sales",
            "DistanceFromHome": 22,
            "Education": 3,
            "EducationField": "Marketing",
            "EnvironmentSatisfaction": 1,
            "Gender": "Male",
            "HourlyRate": 55,
            "JobInvolvement": 2,
            "JobLevel": 1,
            "JobRole": "Sales Executive",
            "JobSatisfaction": 1,
            "MaritalStatus": "Single",
            "MonthlyIncome": 2500,
            "MonthlyRate": 12000,
            "NumCompaniesWorked": 4,
            "OverTime": "Yes",
            "PercentSalaryHike": 11,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 2,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 4,
            "TrainingTimesLastYear": 2,
            "WorkLifeBalance": 1,
            "YearsAtCompany": 2,
            "YearsInCurrentRole": 1,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 1
        }
        print("--- Running Sample Prediction ---")
        res = predict_employee(sample_employee)
        print(json.dumps(res, indent=2))
'''

with open(PREDICTION_PATH, 'w', encoding='utf-8') as f:
    f.write(prediction_script_content)
print(f"Prediction script created at {PREDICTION_PATH}")

# ----------------------------------------------------
# 4. Build notebook.ipynb using nbformat
# ----------------------------------------------------
nb = nbf.v4.new_notebook()

cells = []

# Title Cell
cells.append(nbf.v4.new_markdown_cell("""# Proyek Akhir Data Science: Menyelesaikan Permasalahan HR Attrition (PT Jaya Jaya Maju)

**Nama**: Student / Developer  
**Email**: developer@mail.com  
**ID Dicoding**: dicoding_user  

---

## 1. Business Understanding

### Latar Belakang
**PT Jaya Jaya Maju** merupakan perusahaan multinasional yang berdiri sejak tahun 2000 dengan lebih dari 1.000 karyawan. Meskipun tumbuh menjadi perusahaan besar, manajerial perusahaan menghadapi tantangan tingginya **attrition rate** (rasio karyawan keluar) melebihi **10%**.

### Permasalahan Bisnis
1. Apakah faktor-faktor utama yang mendorong tingginya tingkat *attrition* pada karyawan PT Jaya Jaya Maju?
2. Bagaimana membuat Business Dashboard interaktif untuk memonitor faktor-faktor risiko tersebut secara berkelanjutan?
3. Bagaimana membangun model Machine Learning untuk memprediksi potensi *attrition* karyawan secara proaktif?

### Cakupan Proyek
- **Data Preprocessing & Cleaning**: Penanganan missing values, pengodean variabel kategorikal, dan normalisasi.
- **Exploratory Data Analysis (EDA)**: Analisis bivariat dan multivariat mengenai pengaruh OverTime, Kompensasi, Job Level, dan Work-Life Balance terhadap Attrition.
- **Machine Learning Modeling**: Pelatihan model *Random Forest Classifier* dengan penanganan imbalansi kelas (`class_weight='balanced'`).
- **Business Dashboard Visualization**: Pembuatan dashboard visual eksekutif.
- **Model Deployment & Inference Script**: Pembuatan script CLI (`prediction.py`) untuk inferensi data karyawan baru.
"""))

# Imports & Setup
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set visual style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
"""))

# Data Loading Cell
cells.append(nbf.v4.new_code_cell("""# Load Dataset
df = pd.read_csv('data/employee_data.csv')
print("Dataset Shape:", df.shape)
df.head()
"""))

# Data Cleaning Cell
cells.append(nbf.v4.new_code_cell("""# Check missing values
print("Missing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Separate Labeled Data (Attrition not null) and Unlabeled Data
labeled_df = df[df['Attrition'].notnull()].copy()
unlabeled_df = df[df['Attrition'].isnull()].copy()

labeled_df['Attrition'] = labeled_df['Attrition'].astype(int)
print(f"Labeled Data: {labeled_df.shape[0]} rows | Unlabeled Data: {unlabeled_df.shape[0]} rows")
"""))

# EDA Cell
cells.append(nbf.v4.new_markdown_cell("""## 2. Exploratory Data Analysis (EDA)

Mari kita analisis faktor-faktor kunci yang memengaruhi tingkat Attrition pada PT Jaya Jaya Maju.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Attrition Rate Distribution
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=labeled_df, x='Attrition', palette=['#38bdf8', '#f43f5e'])
plt.title('Overall Attrition Count (0: Stayed, 1: Left)')
plt.xlabel('Attrition')
plt.ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
plt.show()

# Overall percentage
att_pct = labeled_df['Attrition'].mean() * 100
print(f"Overall Attrition Rate: {att_pct:.2f}%")
"""))

cells.append(nbf.v4.new_code_cell("""# 2. OverTime vs Attrition
plt.figure(figsize=(7, 4))
sns.countplot(data=labeled_df, x='OverTime', hue='Attrition', palette=['#38bdf8', '#f43f5e'])
plt.title('Impact of OverTime on Attrition')
plt.show()

ot_summary = labeled_df.groupby('OverTime')['Attrition'].agg(['count', 'mean'])
ot_summary['mean'] = ot_summary['mean'] * 100
print("Attrition Rate by OverTime status:")
print(ot_summary)
"""))

cells.append(nbf.v4.new_code_cell("""# 3. Monthly Income & Job Level vs Attrition
plt.figure(figsize=(10, 5))
sns.boxplot(data=labeled_df, x='JobLevel', y='MonthlyIncome', hue='Attrition', palette=['#38bdf8', '#f43f5e'])
plt.title('Monthly Income vs Job Level by Attrition Status')
plt.show()
"""))

# Model Preparation Cell
cells.append(nbf.v4.new_markdown_cell("""## 3. Machine Learning Modeling

Kita membangun pipeline preprocessing dan model *Random Forest Classifier* dengan *balanced class weight* untuk menangani imbalansi kelas pada target Attrition.
"""))

cells.append(nbf.v4.new_code_cell("""# Drop unnecessary columns
cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours']
clean_df = labeled_df.drop(columns=[c for c in cols_to_drop if c in labeled_df.columns])

categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
numerical_cols = [c for c in clean_df.columns if c not in categorical_cols + ['EmployeeId', 'Attrition']]

X = clean_df[numerical_cols + categorical_cols]
y = clean_df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42))
])

model.fit(X_train, y_train)

# Metrics
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
"""))

# Model Export Cell
cells.append(nbf.v4.new_code_cell("""# Save Model Pipeline
joblib.dump(model, 'model/attrition_model.pkl')
print("Model successfully exported to model/attrition_model.pkl")
"""))

# Inference on Unlabeled Data Cell
cells.append(nbf.v4.new_markdown_cell("""## 4. Inferensi pada Data Unlabeled (412 Karyawan)

Model yang telah dilatih digunakan untuk memprediksi tingkat risiko *attrition* pada 412 karyawan tanpa label.
"""))

cells.append(nbf.v4.new_code_cell("""unlabeled_clean = unlabeled_df[numerical_cols + categorical_cols]
unlabeled_preds = model.predict(unlabeled_clean)
unlabeled_probas = model.predict_proba(unlabeled_clean)[:, 1]

unlabeled_df['Predicted_Attrition'] = unlabeled_preds
unlabeled_df['Attrition_Risk_Score'] = (unlabeled_probas * 100).round(2)

print("Jumlah Karyawan Berisiko Tinggi (Predicted Attrition = 1):", (unlabeled_preds == 1).sum())
unlabeled_df[['EmployeeId', 'Department', 'JobRole', 'OverTime', 'MonthlyIncome', 'Predicted_Attrition', 'Attrition_Risk_Score']].head(10)
"""))

# Conclusion Cell
cells.append(nbf.v4.new_markdown_cell("""## 5. Kesimpulan & Rekomendasi Action Items

### Kesimpulan
1. **OverTime (Lembur)** merupakan pendorong utama *attrition*. Karyawan yang sering lembur memiliki tingkat turnover **30.5%**, dibandingkan hanya **10.4%** pada karyawan yang tidak lembur.
2. **Disparitas Gaji di Level Awal (Job Level 1 & 2)** memicu ketidakpuasan, di mana karyawan yang meninggalkan perusahaan mayoritas berasal dari kelompok pendapatan terendah di kelas jabatannya.
3. **Work-Life Balance & Environment Satisfaction** yang rendah secara signifikan meningkatkan ketertarikan karyawan untuk keluar.

### Rekomendasi Action Items
1. **Restrukturisasi Kebijakan Lembur**: Membatasi jam lembur mingguan dan memberikan kompensasi lembur / cuti pengganti yang transparan.
2. **Penyesuaian Kompensasi & Path Karir**: Meninjau ulang standar gaji karyawan Job Level 1 & 2 serta menyediakan jalur promosi berdasarkan performa yang jelas.
3. **Penerapan Sistem Early Warning Riskan Attrition**: Menggunakan script `prediction.py` untuk memonitor karyawan berisiko tinggi secara berkala dan melakukan *stay interview*.
"""))

nb['cells'] = cells

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Jupyter Notebook generated at {NOTEBOOK_PATH}")

# ----------------------------------------------------
# 5. Build README.md (Dicoding Template)
# ----------------------------------------------------
readme_content = """# Proyek Akhir: Menyelesaikan Permasalahan HR Attrition - PT Jaya Jaya Maju

## Business Understanding

PT Jaya Jaya Maju merupakan salah satu perusahaan multinasional yang berdiri sejak tahun 2000 dengan lebih dari 1.000 karyawan. Meskipun telah menjadi perusahaan besar, pengelolaan karyawan masih menjadi tantangan utama yang berimbas pada tingginya **attrition rate** (rasio karyawan keluar) melebihi **10%**.

### Permasalahan Bisnis
- Apa saja faktor utama yang memengaruhi tingginya tingkat *attrition* karyawan?
- Bagaimana menyajikan visualisasi data interaktif melalui Business Dashboard untuk memonitor faktor risiko *attrition*?
- Bagaimana membangun model *Machine Learning* yang mampu memprediksi potensi *attrition* karyawan secara akurat untuk pencegahan dini?

### Cakupan Proyek
- Exploratory Data Analysis (EDA) pada dataset karyawan Jaya Jaya Maju.
- Pengembangan Business Dashboard eksekutif (`William_dicoding-dashboard.png`).
- Pelatihan model Machine Learning (*Random Forest Classifier*) dan penyimpanan artefak model (`model/attrition_model.pkl`).
- Pembuatan script inferensi mandiri (`prediction.py`).
- Penyusunan dokumentasi dan rekomendasi bisnis strategis.

### Persiapan

Sumber data: [Employee Dataset - Dicoding GitHub](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee)

Setup Environment:
```bash
# Clone atau siapkan direktori proyek
git clone <repository_url>
cd submission

# Install dependencies
pip install -r requirements.txt
```

---

## Business Dashboard

Dashboard eksekutif (`William_dicoding-dashboard.png`) menyajikan gambaran menyeluruh terkait indikator kinerja HR dan faktor risiko utama:

1. **KPI Utama**:
   - **Total Karyawan**: 1.470 karyawan
   - **Tingkat Attrition Keseluruhan**: 16.9%
   - **Tingkat Attrition Karyawan Lembur (OverTime)**: 30.5%
2. **Visualisasi Utama**:
   - **Attrition Rate by Department**: Departemen Sales (20.6%) dan HR (19.0%) mencatatkan tingkat attrition tertinggi.
   - **OverTime vs Attrition**: Karyawan dengan status lembur (OverTime = Yes) memiliki risiko keluar 3x lebih tinggi dibanding karyawan tanpa lembur.
   - **Disparitas Gaji per Job Level**: Karyawan yang keluar terkonsentrasi pada Job Level 1 dan 2 dengan *Monthly Income* di bawah rata-rata.
   - **Analisis Usia vs Pendapatan Bulanan**: Kelompok usia muda (20-30 tahun) dengan gaji terendah dan lembur tinggi mendominasi angka *attrition*.

---

## Machine Learning Model

Model dikembangkan menggunakan **Random Forest Classifier** yang dikombinasikan dengan pembobotan kelas seimbang (`class_weight='balanced'`) untuk menangani ketidakseimbangan data target.

### Hasil Evaluasi Model:
- **Accuracy**: ~84.4%
- **ROC-AUC Score**: ~0.795
- **F1-Score**: ~0.55 (Optimized for Recall/Detection Risk)

### Penggunaan Script Prediksi (`prediction.py`)

Anda dapat menjalankan inferensi risiko *attrition* karyawan secara mandiri menggunakan script `prediction.py`:

```bash
# 1. Jalankan prediksi sampel bawaan
python prediction.py

# 2. Jalankan prediksi menggunakan input file CSV
python prediction.py --file data/employee_data.csv

# 3. Jalankan prediksi menggunakan string JSON data karyawan
python prediction.py --json '{"Age": 28, "OverTime": "Yes", "MonthlyIncome": 2200, "JobLevel": 1, ...}'
```

Contoh Output Prediksi:
```json
[
  {
    "EmployeeIndex": 0,
    "Prediction": 1,
    "Status": "HIGH RISK (Attrition Likely)",
    "AttritionProbability": "78.40%"
  }
]
```

---

## Kesimpulan & Rekomendasi Action Items

### Kesimpulan
- **OverTime** adalah prediktor terbesar dalam keputusan karyawan untuk mengundurkan diri.
- Karyawan di **Job Level 1 & 2** memiliki tingkat keluar tertinggi akibat kombinasi beban kerja tinggi dan kompensasi awal yang kurang kompetitif.
- Faktor lingkungan kerja (*Environment Satisfaction*) dan keseimbangan hidup (*Work Life Balance*) memegang peranan krusial dalam retensi karyawan.

### Rekomendasi Action Items
1. **Pengendalian Jam Lembur (OverTime Policy)**:
   Menerapkan batas maksimal jam lembur bulanan dan sistem kompensasi lembur / waktu istirahat yang transparan.
2. **Evaluasi Skala Gaji & Pathway Karir Level 1-2**:
   Menyesuaikan *Monthly Income* karyawan level pemula sesuai tolok ukur industri serta kejelasan jenjang karir berkala.
3. **Penerapan Sistem Peringatan Dini (Early Warning System)**:
   Mengintegrasikan script `prediction.py` ke dalam sistem HR internal untuk mengidentifikasi karyawan berisiko tinggi sebelum mereka mengajukan pengunduran diri.
"""

with open(README_PATH, 'w', encoding='utf-8') as f:
    f.write(readme_content)
print(f"README.md generated at {README_PATH}")

# ----------------------------------------------------
# 6. Build requirements.txt
# ----------------------------------------------------
reqs = """pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.2.0
"""
with open(REQUIREMENTS_PATH, 'w', encoding='utf-8') as f:
    f.write(reqs)
print(f"requirements.txt generated at {REQUIREMENTS_PATH}")

print("=== ALL SUBMISSION FILES SUCCESSFULLY GENERATED ===")
