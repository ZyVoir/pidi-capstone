# Submission Final: Menyelesaikan Permasalahan Institusi Pendidikan — Jaya Jaya Institut

> **Course**: Belajar Penerapan Data Science — Dicoding Indonesia  
> **Status**: Completed (Ready for review)

---

## 📋 Overview

This submission tackles the high student **dropout rate** at Jaya Jaya Institut by delivering an end-to-end data science system: deep exploratory analysis, an executive Metabase dashboard, a Random Forest classification model, and a Streamlit-based early warning prototype.

---

## 📁 Folder Structure

```text
submission-final/
├── Artifact/                         # Reference materials & instructions
│   ├── context.txt                   # Parsed submission guidelines
│   └── Instruksi Submission...html   # Original Dicoding instructions
└── submission/                       # 📦 Final submission files
    ├── data/
    │   └── data.csv                  # Source student performance dataset
    ├── model/
    │   └── student_model.pkl         # Trained Random Forest classifier
    ├── notebook.ipynb                # Executed data science notebook
    ├── app.py                        # Streamlit prototype web application
    ├── README.md                     # Full project documentation
    ├── requirements.txt              # Python library dependencies
    ├── William_dicoding-dashboard.png  # Student Performance Dashboard screenshot
    ├── metabase.db.mv.db             # Metabase H2 database instance
    └── students.db                   # SQLite database (source for Metabase)
```

---

## 🔑 Key Findings

| Factor | Impact |
| :--- | :--- |
| **Tuition Fees Up-to-Date** | Students who are **not** up-to-date with tuition fees show a critically high dropout rate. |
| **Financial Debt (Debtor)** | Personal financial liability strongly correlates with increased student dropout risk. |
| **Scholarship Holder** | Scholarship recipients have a significantly higher graduation rate (Graduate) and low dropout. |
| **Academic Performance** | The number of curricular subjects approved in the 1st and 2nd semesters is the strongest predictor of student success. |

---

## 🤖 ML Model Performance

Model: **Random Forest Classifier** (`class_weight='balanced'`, `n_estimators=100`, `max_depth=12`)

| Metric | Score |
| :--- | :--- |
| Accuracy | 75.37% |
| Weighted Precision | 76.37% |
| Weighted Recall | 75.37% |
| Weighted F1-Score | 75.61% |

---

## 🚀 Quick Start

### Running the Streamlit App locally:
```bash
# 1. Navigate to the submission folder
cd DS/submission-final/submission

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate.bat    # Windows CMD

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the Streamlit application
streamlit run app.py
```

### Accessing the Streamlit Cloud version:
- **Streamlit App URL**: [Jaya Jaya Institut Student Retention App](https://william-student-retention.streamlit.app) *(Contoh link)*

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

*Note: Ensure the local folder structure has `students.db` mounted or accessible to the container as configured in the setup.*

Then open: **http://localhost:3000**  
Credentials: `root@mail.com` / `root123`

---

## 📝 Full Documentation

→ See [`submission/README.md`](./submission/README.md) for the complete end-to-end documentation.
