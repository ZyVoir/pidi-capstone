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
* **[Submission 2 — Student Dropout (Final Submission)](./DS/submission-final/README.md)** (`DS/submission-final/`): Covers full EDA, Metabase dashboard (`William_dicoding-dashboard.png`), database instance (`metabase.db.mv.db`, `students.db`), Streamlit app (`app.py`), Random Forest model (`student_model.pkl`), and actionable recommendations. [Status: Approved / Sudah di-approve] → [View Details](./DS/submission-final/README.md)

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

## 🛠️ Beginner's Guide: How to Use this Repo

If you are a beginner looking to submit your Dicoding Capstone projects or customize them, follow this simple guide.

### 1. What to Submit to Dicoding?
To pass the review, you must upload the **`submission.zip`** file located inside the respective project folders:
- **For Submission 1 (HR Attrition)**: Upload [`DS/submission-1/submission.zip`](./DS/submission-1/submission.zip)
- **For Submission 2 (Final Student Performance)**: Upload [`DS/submission-final/submission.zip`](./DS/submission-final/submission.zip)

These zip files already contain everything (code, model, executed notebooks, database instance, requirements, and dashboard screenshot) with the default name **`William`** and are fully ready to submit.

---

### 2. How to Customize with the ATM Skill (For AI Assistants)
If you want to customize the project with your own name, custom wording, and a unique visual style (so your submission looks different from everyone else's), you can use the **ATM Skill** with an AI coding assistant.

#### How to Prompt the AI Assistant:
Open this repository in your AI coding assistant (like Gemini Antigravity) and send a prompt like this:
> *"I want to generate a new submission for the final capstone. Please run the ATM skill. Use the name '**[Your Name Here]**' and style preference '**[Pastel / Earthy / Synthwave / Nordic / Dark]**'."*

#### What to Expect:
When you prompt the AI with this:
1. The assistant will ask you for confirmation of your **Username** and **Styling Preference**.
2. It will run the build scripts (`build_submission_v2.py` or `build_submission_final.py`) with your inputs.
3. The script will automatically:
   - Generate a custom dashboard screenshot (`[YourName]_dicoding-dashboard.png`) using the custom colors of your selected style theme.
   - Generate a custom Jupyter notebook (`notebook.ipynb`) and documentation (`README.md`) labeled with your name.
   - Run the machine learning pipeline and save the model.
   - Re-execute the Jupyter notebook programmatically to save all cell outputs.
   - Package all customized files into a fresh **`submission.zip`** ready for you to download and submit!

---

### 3. How to Run Locally (Manual Setup)

If you want to manually run the scripts or start the web applications on your computer:

```bash
# Clone the repository
git clone https://github.com/ZyVoir/pidi-capstone.git
cd pidi-capstone

# Navigate to the submission directory (e.g. final submission)
cd DS/submission-final/submission

# Create a virtual environment and activate it
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit web app
streamlit run app.py

# Or regenerate the submission with custom parameters manually:
python3 build_submission_final.py --username "Alex" --theme "pastel"
```
