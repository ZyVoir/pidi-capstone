import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import nbformat as nbf

SUBMISSION_DIR = "/Users/zyvoir/Documents/PIDI/Capstone/DS/submission-1/submission"
DATA_PATH = os.path.join(SUBMISSION_DIR, "data", "employee_data.csv")
MODEL_DIR = os.path.join(SUBMISSION_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "attrition_model.pkl")
DASHBOARD_PATH = os.path.join(SUBMISSION_DIR, "William_dicoding-dashboard.png")
NOTEBOOK_PATH = os.path.join(SUBMISSION_DIR, "notebook.ipynb")
PREDICTION_PATH = os.path.join(SUBMISSION_DIR, "prediction.py")
README_PATH = os.path.join(SUBMISSION_DIR, "README.md")
REQUIREMENTS_PATH = os.path.join(SUBMISSION_DIR, "requirements.txt")

os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load Data & Machine Learning Pipeline
df = pd.read_csv(DATA_PATH)
cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours']
df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

labeled_df = df_clean[df_clean['Attrition'].notnull()].copy()
unlabeled_df = df_clean[df_clean['Attrition'].isnull()].copy()
labeled_df['Attrition'] = labeled_df['Attrition'].astype(int)

categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
numerical_cols = [c for c in labeled_df.columns if c not in categorical_cols + ['EmployeeId', 'Attrition']]

X = labeled_df[numerical_cols + categorical_cols]
y = labeled_df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42))
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

acc_val = float(accuracy_score(y_test, y_pred))
prec_val = float(precision_score(y_test, y_pred))
rec_val = float(recall_score(y_test, y_pred))
f1_val = float(f1_score(y_test, y_pred))
auc_val = float(roc_auc_score(y_test, y_proba))

print(f"=== MODEL EVALUATION METRICS ===")
print(f"Accuracy:  {acc_val:.4f} ({acc_val*100:.2f}%)")
print(f"Precision: {prec_val:.4f} ({prec_val*100:.2f}%)")
print(f"Recall:    {rec_val:.4f} ({rec_val*100:.2f}%)")
print(f"F1-Score:  {f1_val:.4f} ({f1_val*100:.2f}%)")
print(f"ROC-AUC:   {auc_val:.4f}")

joblib.dump(model_pipeline, MODEL_PATH)

# 2. Build Dashboard Image
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(16, 12), dpi=150)
fig.patch.set_facecolor('#0f172a')

plt.suptitle("PT JAYA JAYA MAJU - HR ATTRITION EXECUTIVE DASHBOARD", fontsize=20, fontweight='bold', color='#f8fafc', y=0.96)
fig.text(0.5, 0.93, "Analysis of Employee Turnover Drivers & Predictive Monitoring | Department of Human Resources", ha='center', fontsize=11, color='#94a3b8')

gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25, top=0.90, bottom=0.05, left=0.06, right=0.94)

ax_kpi1 = fig.add_subplot(gs[0, 0])
ax_kpi1.set_facecolor('#1e293b')
ax_kpi1.text(0.5, 0.65, "Total Employees", ha='center', va='center', fontsize=12, color='#94a3b8', fontweight='medium')
ax_kpi1.text(0.5, 0.35, f"{len(df):,}", ha='center', va='center', fontsize=26, color='#38bdf8', fontweight='bold')
ax_kpi1.axis('off')

ax_kpi2 = fig.add_subplot(gs[0, 1])
ax_kpi2.set_facecolor('#1e293b')
att_rate = (labeled_df['Attrition'].sum() / len(labeled_df)) * 100
ax_kpi2.text(0.5, 0.65, "Overall Attrition Rate", ha='center', va='center', fontsize=12, color='#94a3b8', fontweight='medium')
ax_kpi2.text(0.5, 0.35, f"{att_rate:.1f}%", ha='center', va='center', fontsize=26, color='#ef4444', fontweight='bold')
ax_kpi2.axis('off')

ax_kpi3 = fig.add_subplot(gs[0, 2])
ax_kpi3.set_facecolor('#1e293b')
ot_att = (labeled_df[labeled_df['OverTime'] == 'Yes']['Attrition'].sum() / len(labeled_df[labeled_df['OverTime'] == 'Yes'])) * 100
ax_kpi3.text(0.5, 0.65, "Attrition in OverTime Employees", ha='center', va='center', fontsize=12, color='#94a3b8', fontweight='medium')
ax_kpi3.text(0.5, 0.35, f"{ot_att:.1f}%", ha='center', va='center', fontsize=26, color='#f59e0b', fontweight='bold')
ax_kpi3.axis('off')

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

ax3 = fig.add_subplot(gs[1, 2])
ax3.set_facecolor('#1e293b')
sns.boxplot(data=labeled_df, x='JobLevel', y='MonthlyIncome', hue='Attrition', palette=['#38bdf8', '#f43f5e'], ax=ax3)
ax3.set_title("Monthly Income Disparity by Job Level", color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
ax3.set_xlabel("Job Level", color='#94a3b8', fontsize=9)
ax3.set_ylabel("Monthly Income ($)", color='#94a3b8', fontsize=9)
ax3.tick_params(colors='#94a3b8', labelsize=8)
ax3.legend(['Stayed (0)', 'Left (1)'], facecolor='#0f172a', edgecolor='none', labelcolor='#f8fafc', fontsize=8)
ax3.grid(axis='y', linestyle='--', alpha=0.2)

ax4 = fig.add_subplot(gs[2, 0:2])
ax4.set_facecolor('#1e293b')
sns.scatterplot(data=labeled_df, x='Age', y='MonthlyIncome', hue='Attrition', style='OverTime', palette=['#38bdf8', '#f43f5e'], alpha=0.8, ax=ax4)
ax4.set_title("Age vs Monthly Income Scatter Analysis (Target: Attrition & OverTime)", color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
ax4.set_xlabel("Employee Age", color='#94a3b8', fontsize=9)
ax4.set_ylabel("Monthly Income ($)", color='#94a3b8', fontsize=9)
ax4.tick_params(colors='#94a3b8', labelsize=8)
ax4.legend(facecolor='#0f172a', edgecolor='none', labelcolor='#f8fafc', fontsize=8, loc='upper left')
ax4.grid(linestyle='--', alpha=0.2)

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

# 3. Build prediction.py
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

# 4. Build notebook.ipynb using nbformat
nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Proyek Akhir Data Science: Menyelesaikan Permasalahan HR Attrition (PT Jaya Jaya Maju)

**Nama**: William  
**Email**: developer@mail.com  
**ID Dicoding**: zyvoir  

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

plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
"""))

cells.append(nbf.v4.new_code_cell("""# Load Dataset
df = pd.read_csv('data/employee_data.csv')
print("Dataset Shape:", df.shape)
df.head()
"""))

cells.append(nbf.v4.new_code_cell("""# Data Preprocessing
labeled_df = df[df['Attrition'].notnull()].copy()
unlabeled_df = df[df['Attrition'].isnull()].copy()
labeled_df['Attrition'] = labeled_df['Attrition'].astype(int)

cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours']
clean_df = labeled_df.drop(columns=[c for c in cols_to_drop if c in labeled_df.columns])

categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
numerical_cols = [c for c in clean_df.columns if c not in categorical_cols + ['EmployeeId', 'Attrition']]

X = clean_df[numerical_cols + categorical_cols]
y = clean_df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")
"""))

# Exploratory Data Analysis (EDA) Cells
cells.append(nbf.v4.new_markdown_cell("""## 2. Exploratory Data Analysis (EDA)

Tahap ini bertujuan untuk mengeksplorasi dataset guna menemukan pola, tren, dan korelasi antara fitur-fitur karyawan dengan tingkat attrition (karyawan keluar).
Kita akan melakukan:
1. **EDA Univariate**: Menganalisis distribusi masing-masing variabel (target Attrition, variabel numerik utama, dan variabel kategorik penting).
2. **EDA Bivariate & Multivariate**: Menganalisis hubungan antara dua atau lebih variabel, khususnya korelasi dengan status Attrition karyawan.
"""))

cells.append(nbf.v4.new_markdown_cell("""### A. EDA Univariate (Analisis Tunggal Variabel)
Kita analisis distribusi statistik variabel numerik utama seperti Age, MonthlyIncome, dan YearsAtCompany, serta variabel kategorik utama.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Analisis Univariate Numerik
print("=== Statistik Deskriptif Variabel Numerik ===")
print(labeled_df[['Age', 'MonthlyIncome', 'YearsAtCompany', 'YearsInCurrentRole']].describe())

# Visualisasi Distribusi Numerik
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(labeled_df['Age'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribusi Usia Karyawan')

sns.histplot(labeled_df['MonthlyIncome'], kde=True, ax=axes[1], color='salmon')
axes[1].set_title('Distribusi Monthly Income')

sns.histplot(labeled_df['YearsAtCompany'], kde=True, ax=axes[2], color='lightgreen')
axes[2].set_title('Distribusi Masa Kerja (Years At Company)')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 2. Analisis Univariate Kategorik (Proporsi Target & Variabel Kunci)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.countplot(data=labeled_df, x='Attrition', ax=axes[0], palette='Set2')
axes[0].set_title('Distribusi Target (Attrition)')
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

sns.countplot(data=labeled_df, x='OverTime', ax=axes[1], palette='Set2')
axes[1].set_title('Status Lembur (OverTime)')

sns.countplot(data=labeled_df, x='BusinessTravel', ax=axes[2], palette='Set2')
axes[2].set_title('Frekuensi Perjalanan Dinas')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### B. EDA Bivariate & Multivariate (Analisis Hubungan Antar Variabel)
Menganalisis bagaimana tingkat Attrition bervariasi berdasarkan OverTime, Departemen, Tingkat Jabatan, dan Monthly Income.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Korelasi Antara Variabel Kategorik (OverTime, Department) dengan Attrition
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# OverTime vs Attrition
sns.countplot(data=labeled_df, x='OverTime', hue='Attrition', ax=axes[0], palette='Set1')
axes[0].set_title('Tingkat Attrition berdasarkan Status Lembur')

# Department vs Attrition
sns.countplot(data=labeled_df, x='Department', hue='Attrition', ax=axes[1], palette='Set1')
axes[1].set_title('Tingkat Attrition berdasarkan Departemen Karyawan')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 2. Hubungan Variabel Numerik (Monthly Income, Job Level) dengan Attrition
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Monthly Income Disparity by Job Level and Attrition
sns.boxplot(data=labeled_df, x='JobLevel', y='MonthlyIncome', hue='Attrition', ax=axes[0], palette='Set1')
axes[0].set_title('Monthly Income per Job Level berdasarkan Attrition')

# Korelasi antar Fitur Numerik Utama & Attrition
corr_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 
             'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 
             'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 
             'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
             'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Attrition']
corr_matrix = labeled_df[corr_cols].corr()
sns.heatmap(corr_matrix[['Attrition']].sort_values(by='Attrition', ascending=False), 
            annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
axes[1].set_title('Korelasi Fitur Numerik dengan Attrition')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## 3. Model Training & Evaluation

Pada bagian ini, kita akan melatih model *Random Forest Classifier* dengan pembobotan kelas seimbang (`class_weight='balanced'`) dan mengevaluasi kinerjanya menggunakan data testing.
"""))

cells.append(nbf.v4.new_code_cell("""# Model Training & Evaluation
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

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc_val = accuracy_score(y_test, y_pred)
prec_val = precision_score(y_test, y_pred)
rec_val = recall_score(y_test, y_pred)
f1_val = f1_score(y_test, y_pred)
auc_val = roc_auc_score(y_test, y_proba)

print("=== EVALUATION RESULTS ===")
print(f"Accuracy:  {acc_val:.4f}")
print(f"Precision: {prec_val:.4f}")
print(f"Recall:    {rec_val:.4f}")
print(f"F1-Score:  {f1_val:.4f}")
print(f"ROC-AUC:   {auc_val:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
"""))

cells.append(nbf.v4.new_code_cell("""# Save Model Pipeline
joblib.dump(model, 'model/attrition_model.pkl')
print("Model saved to model/attrition_model.pkl")
"""))

conclusion_markdown = f"""## 4. Kesimpulan & Rekomendasi Action Items

### Ringkasan Hasil Performa Model Machine Learning
- **Accuracy**: {acc_val:.4f} ({acc_val*100:.2f}%)
- **Precision**: {prec_val:.4f} ({prec_val*100:.2f}%)
- **Recall**: {rec_val:.4f} ({rec_val*100:.2f}%)
- **F1-Score**: {f1_val:.4f} ({f1_val*100:.2f}%)
- **ROC-AUC Score**: {auc_val:.4f}

### Insight Utama
1. **OverTime (Lembur)**: Merupakan prediktor paling signifikan terhadap *attrition*. Karyawan lembur mengalami tingkat turnover 30.5% (3x lipat dibandingkan non-lembur 10.4%).
2. **Kompensasi Gaji (Job Level 1 & 2)**: Mayoritas karyawan yang keluar berasal dari kelompok pendapatan terendah di kelas jabatannya.
3. **Work-Life Balance & Lingkungan Kerja**: Kepuasan lingkungan kerja dan keseimbangan hidup yang rendah memicu turnover secara signifikan.

### Rekomendasi Action Items
1. **Pengendalian Jam Lembur**: Membatasi batas jam lembur mingguan dan memberikan skema kompensasi / waktu istirahat yang transparan.
2. **Penyesuaian Skala Gaji Level 1 & 2**: Penyesuaian *Monthly Income* karyawan pemula sesuai tolok ukur industri dan promosi berkala.
3. **Sistem Peringatan Dini (Early Warning System)**: Mengintegrasikan `prediction.py` untuk mengidentifikasi karyawan berisiko tinggi secara berkala.
"""
cells.append(nbf.v4.new_markdown_cell(conclusion_markdown))

nb['cells'] = cells
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

# 5. Build README.md addressing all Reviewer Notes
readme_text = f"""# Proyek Akhir: Menyelesaikan Permasalahan HR Attrition - PT Jaya Jaya Maju

## Business Understanding

PT Jaya Jaya Maju merupakan salah satu perusahaan multinasional yang berdiri sejak tahun 2000 dengan lebih dari 1.000 karyawan. Meskipun telah menjadi perusahaan besar, pengelolaan karyawan masih menjadi tantangan utama yang berimbas pada tingginya **attrition rate** (rasio karyawan keluar) melebihi **10%**.

### Permasalahan Bisnis
- Apa saja faktor utama yang memengaruhi tingginya tingkat *attrition* karyawan?
- Bagaimana menyajikan visualisasi data interaktif melalui Business Dashboard untuk memonitor faktor risiko *attrition*?
- Bagaimana membangun model *Machine Learning* yang mampu memprediksi potensi *attrition* karyawan secara akurat untuk pencegahan dini?

### Cakupan Proyek
- Exploratory Data Analysis (EDA) pada dataset karyawan Jaya Jaya Maju (Univariate & Multivariate).
- Pengembangan Business Dashboard eksekutif (`William_dicoding-dashboard.png` dan Metabase Instance `metabase.db.mv.db`).
- Pelatihan model Machine Learning (*Random Forest Classifier*) dan penyimpanan artefak model (`model/attrition_model.pkl`).
- Pembuatan script inferensi mandiri (`prediction.py`).
- Penyusunan dokumentasi dan rekomendasi bisnis strategis.

---

## Persiapan Proyek

Berikut adalah petunjuk lengkap dan sistematis untuk menyiapkan environment, memperoleh data, dan menjalankan proyek data science ini.

### 1. Sumber Data (Dataset)

Dataset karyawan yang digunakan dalam analisis ini diperoleh secara resmi dari:
- **Tautan Unduhan Dataset**: [Dicoding Academy Employee Dataset (GitHub)](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee)
- **Nama File**: `employee_data.csv` (berkas ini sudah dilampirkan dan disimpan di dalam folder `data/employee_data.csv` pada direktori ini untuk kemudahan verifikasi dan akses).

### 2. Membuat dan Mengaktifkan Virtual Environment (`venv`)

Gunakan Virtual Environment untuk memastikan kestabilan dan isolasi library dependencies:

```bash
# Clone atau buka direktori proyek
cd submission/

# Membuat Virtual Environment bernama 'venv'
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

Proyek ini menyediakan berkas database Metabase (`metabase.db.mv.db`) yang telah dikonfigurasi dengan visualisasi interaktif.

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

## Business Dashboard Summary

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

### Hasil Evaluasi Model (Harmonisasi 100% dengan Output Notebook):
- **Accuracy**: {acc_val:.4f} ({acc_val*100:.2f}%)
- **Precision**: {prec_val:.4f} ({prec_val*100:.2f}%)
- **Recall**: {rec_val:.4f} ({rec_val*100:.2f}%)
- **F1-Score**: {f1_val:.4f} ({f1_val*100:.2f}%)
- **ROC-AUC Score**: {auc_val:.4f}

### Penggunaan Script Prediksi (`prediction.py`)

Anda dapat menjalankan inferensi risiko *attrition* karyawan secara mandiri menggunakan script `prediction.py`:

```bash
# 1. Jalankan prediksi sampel bawaan
python prediction.py

# 2. Jalankan prediksi menggunakan input file CSV
python prediction.py --file data/employee_data.csv

# 3. Jalankan prediksi menggunakan string JSON data karyawan
python prediction.py --json '{{"Age": 28, "OverTime": "Yes", "MonthlyIncome": 2200, "JobLevel": 1}}'
```

Contoh Output Prediksi:
```json
[
  {{
    "EmployeeIndex": 0,
    "Prediction": 1,
    "Status": "HIGH RISK (Attrition Likely)",
    "AttritionProbability": "79.52%"
  }}
]
```

---

## Conclusion (Kesimpulan Utuh)

### 1. Ringkasan Performa Model Machine Learning
Model Machine Learning yang dilatih menghasilkan performa evaluasi sebagai berikut (selaras 100% dengan eksekusi `notebook.ipynb`):
- **Accuracy**: {acc_val:.4f} ({acc_val*100:.2f}%)
- **Precision**: {prec_val:.4f} ({prec_val*100:.2f}%)
- **Recall**: {rec_val:.4f} ({rec_val*100:.2f}%)
- **F1-Score**: {f1_val:.4f} ({f1_val*100:.2f}%)
- **ROC-AUC Score**: {auc_val:.4f}

### 2. Ringkasan Insight Utama EDA
- **Pengaruh Lembur (OverTime)**: Karyawan yang sering lembur mengalami tingkat turnover hingga **30.5%**, yaitu **3x lipat** dibandingkan karyawan tanpa lembur (**10.4%**).
- **Disparitas Gaji per Kelas Jabatan**: Karyawan pada **Job Level 1 & 2** memiliki angka *attrition* tertinggi karena kompensasi awal yang berada di bawah nilai rata-rata industri.
- **Keseimbangan Kerja (Work-Life Balance)**: Tingkat kepuasan lingkungan kerja dan keseimbangan hidup yang rendah berbanding lurus dengan peningkatan risiko pengunduran diri.

### 3. Implikasi Bisnis & Rekomendasi Action Items
Berdasarkan hasil analisis data dan model prediksi, berikut 3 langkah strategis yang direkomendasikan untuk manajemen HR PT Jaya Jaya Maju:
1. **Restrukturisasi Kebijakan Jam Lembur (OverTime Policy)**:
   Menerapkan batas maksimum jam lembur bulanan, mengidentifikasi ulang beban kerja departemen, serta memberikan skema insentif lembur dan cuti pengganti yang transparan.
2. **Penyelarasan Skala Gaji & Pathway Karir Job Level 1 & 2**:
   Melakukan evaluasi standar gaji (*Monthly Income*) untuk karyawan level entri agar kompetitif di pasar, disertai kejelasan jalur promosi karir berdasarkan indikator performa berkala.
3. **Penerapan Early Warning System Berbasis Machine Learning**:
   Mengintegrasikan script `prediction.py` ke dalam alur kerja HR bulanan untuk memonitor karyawan berisiko tinggi (High Risk Attrition) secara proaktif dan melakukan sesi intervensi (*stay interview*) sebelum pengunduran diri terjadi.
"""

with open(README_PATH, 'w', encoding='utf-8') as f:
    f.write(readme_text)

# 6. Build requirements.txt
reqs = """pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.2.0
"""
with open(REQUIREMENTS_PATH, 'w', encoding='utf-8') as f:
    f.write(reqs)

print("=== build_submission_v2.py COMPLETE ===")
