# Submission 1: Menyelesaikan Permasalahan Human Resources — PT Jaya Jaya Maju

> **Course**: Belajar Penerapan Data Science — Dicoding Indonesia  
> **Status**: Re-submitted (addressing reviewer feedback)

---

## 📋 Overview

This submission tackles the high employee **attrition rate (>10%)** at PT Jaya Jaya Maju by delivering a complete end-to-end data science solution: exploratory analysis, an executive business dashboard, and a machine learning prediction pipeline.

---

## 📁 Folder Structure

```text
submission-1/
├── Artifact/                         # Reference materials & reviewer feedback
│   ├── context.txt                   # Parsed submission guidance
│   ├── Instruksi Submission...html   # Original Dicoding instructions page
│   ├── Review_1/                     # Reviewer feedback (Submission 1 rejection notes - Review 1)
│   └── Review_3.txt                  # Reviewer feedback (Submission 1 rejection notes - Review 3)
└── submission/                       # 📦 Final submission files
    ├── data/
    │   └── employee_data.csv         # Source dataset (1,470 rows, 35 columns) - [Dataset Source](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee)
    ├── model/
    │   └── attrition_model.pkl       # Trained Random Forest pipeline
    ├── notebook.ipynb                # Executed data science notebook
    ├── prediction.py                 # Standalone CLI inference script
    ├── README.md                     # Full project documentation
    ├── requirements.txt              # Python library dependencies
    ├── William_dicoding-dashboard.png  # Executive HR Attrition Dashboard
    └── metabase.db.mv.db             # Metabase database instance
```

---

## 🔑 Key Findings

| Factor | Impact |
| :--- | :--- |
| **OverTime** | Employees who work overtime have a **30.5% attrition rate** — 3× higher than those who don't (10.4%) |
| **Job Level 1 & 2 Salary** | Lowest-income employees dominate attrition figures — compensation gap drives departures |
| **Work-Life Balance** | Low satisfaction scores correlate strongly with turnover intent |
| **Department** | Sales (20.6%) and HR (19.0%) record the highest attrition rates |

---

## 🤖 ML Model Performance

Model: **Random Forest Classifier** (`class_weight='balanced'`, `n_estimators=150`, `max_depth=8`)

| Metric | Score |
| :--- | :--- |
| Accuracy | 84.91% |
| Precision | 70.00% |
| Recall | 19.44% |
| F1-Score | 30.43% |
| ROC-AUC | 0.7756 |

---

## 🚀 Quick Start

```bash
# 1. Navigate to the submission folder
cd submission-1/submission

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate.bat    # Windows CMD

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run a sample attrition prediction
python3 prediction.py
```

---

## 📊 Dashboard Access (Metabase)

```bash
# Step 1 — Start Metabase container (v0.46.4)
docker run -d -p 3000:3000 --name metabase metabase/metabase:v0.46.4

# Step 2 — Copy database into container
docker cp metabase.db.mv.db metabase:/metabase.db/metabase.db.mv.db

# Step 3 — Restart container
docker restart metabase
```

Then open: **http://localhost:3000**  
Credentials: `root@mail.com` / `root123`

---

## 📝 Full Documentation

→ See [`submission/README.md`](./submission/README.md) for the complete project documentation including all reviewer-addressed items.
