# PIDI Capstone Project Repository

Welcome to the **PIDI Capstone Project** repository (`pidi-capstone`). This workspace serves as the central hub for data science and machine learning capstone projects, structured analysis pipelines, and project submissions.

---

## 📌 Repository Overview

This repository is designed to host multi-stage capstone submissions across Data Science (**DS**) and Machine Learning (**ML**) domain tracks.

```text
pidi-capstone/
├── DS/                          # Data Science Track
│   ├── submission-1/            # Submission 1: HR Attrition (Approved)
│   │   ├── submission/
│   │   ├── submission.zip
│   │   └── README.md
│   └── submission-final/        # Submission 2: Student Dropout (Final Submission)
│       ├── submission/          # Code, model, notebook, database & Streamlit app
│       ├── submission.zip       # Final submission ZIP archive
│       └── README.md            # ← Final submission overview & quick start
├── ML/                          # (Planned) Machine Learning Track
└── .agents/
    └── skills/
        └── atm/                 # ATM (Amati, Tiru, Modifikasi) Workflow Skill
```

---

## 🚀 Track Roadmap & Future Submissions

### 1. Data Science Track (`DS/`)
* **[Submission 1 — HR Attrition (Approved/Lulus)](./DS/submission-1/README.md)** (`DS/submission-1/`): Covers EDA, executive business dashboard (`William_dicoding-dashboard.png`), Metabase instance (`metabase.db.mv.db`), machine learning risk model (`prediction.py`), and full documentation. [Status: Approved / Sudah di-approve] → [View Details](./DS/submission-1/README.md)
* **[Submission 2 — Student Dropout (Final Submission)](./DS/submission-final/README.md)** (`DS/submission-final/`): Covers full EDA, Metabase dashboard (`William_dicoding-dashboard.png`), database instance (`metabase.db.mv.db`, `students.db`), Streamlit app (`app.py`), Random Forest model (`student_model.pkl`), and actionable recommendations. [Status: Completed / Ready for review] → [View Details](./DS/submission-final/README.md)

### 2. Machine Learning Track (`ML/`)
* **Future Expansion**: Open possibility to add a dedicated `ML/` directory for pure Machine Learning capstone modules.
* **Contribution Workflow**: Any new ML track features or submissions should be developed in a feature branch and submitted via **Pull Request (PR)** before merging into `main`.

---

## 🎨 ATM (Amati, Tiru, Modifikasi) Skill Integration

This repository includes a custom workspace skill: **ATM Framework** located at [`.agents/skills/atm/SKILL.md`](.agents/skills/atm/SKILL.md).

### What is the ATM Skill?
When invoked, the **ATM** skill ensures:
1. **AMATI (Observe)**: Analyzes the reference requirements and data schemas thoroughly.
2. **TIRU (Imitate)**: Guarantees 100% functional match, calculation accuracy, and test compliance with target output specs.
3. **MODIFIKASI (Modify)**: Dynamically customizes output presentation across every run:
   - **Unique Wording & Phrasing**: Rotates text summaries, narrative commentary, and docstrings.
   - **Code Architecture**: Varies function organization, variable naming, and refactoring styles.
   - **Dynamic PNG & Graphic Aesthetics**: Generates custom color schemes (Pastel, Dark Mode, Monochrome, Neon, Earthy, or custom palettes) and visual layouts so every deliverable has a distinct, aesthetic signature.

To invoke the ATM skill, prompt the assistant with commands like:
> *"Run the ATM skill to generate a uniquely styled submission..."*

---

## 🛠️ Getting Started

To run or evaluate individual submissions:

```bash
# Clone the repository
git clone https://github.com/ZyVoir/pidi-capstone.git
cd pidi-capstone

# Navigate to DS Submission 1
cd DS/submission-1/submission

# Install dependencies
pip install -r requirements.txt

# Run inference script
python3 prediction.py
```
