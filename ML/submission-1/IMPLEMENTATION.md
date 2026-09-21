# ML Submission 1 — Proyek Pertama: Predictive Analytics

> **Course**: Machine Learning Terapan (Dicoding, academy 319)
> **Status**: ✅ **Approved / Sudah di-approve** (Submission ID 5049831)
> **Sources**: `artifact/Proyek Pertama Kriteria Submission ...html` (module 6.1) · `artifact/Detail Laporan ...html` (module 6.2)
> **Not saved locally**: module 6.3 Proyek Pertama Kirim Submission dan Review
> **Review feedback**: `artifact/Review/Review 1.html` — an *earlier rejected attempt* (Submission ID
> 5049831, sent 19 Sep 2026 01:28) on Model Development + Evaluation. Note: the files currently in
> `submission/` are the 01:10 build, i.e. the rejected revision — the revision that was approved is
> not saved here. Both findings are still visible in this build (metric tables at report lines 270
> and 291 sit inside `## Modeling`; `max_iter`, `n_jobs`, `random_state`, `StratifiedKFold`,
> `test_size` are absent from the report). Kept as a record of the rubric's expectations.
>
> **Resubmission note**: the approved revision resolved both findings — parameters documented per
> algorithm (including which were left at *default*) and all metric tables moved out of Modeling
> into Evaluation. `ML/submission-final` carries those fixes forward.

---

## 1. Hard requirements (Kriteria Submission)

From module 6.1. Fail any of these → submission **rejected**.

- [ ] **Original work.** Your own project. Never used for a previous Dicoding ML class submission, never published on any platform.
- [ ] **Quantitative dataset, minimum 500 samples.**
- [ ] **Notebook text cells** documenting every stage of the project (`.ipynb`).
- [ ] **Approach** — pick exactly one: **Klasifikasi** / **Regresi** / **Time series dan forecasting**.
- [ ] **Report draft** covering problem domain → data understanding → data preparation → modeling → evaluation, per module 6.2.

There is **no prescribed case study or dataset** — you choose the domain yourself (see §5 Tips).

### Auto-reject conditions

| Condition |
| :--- |
| Not submitted as `.zip` |
| No report in Markdown (`.md`) |
| No ML project files (`.py` **and** `.ipynb`) |
| Jupyter Notebook not executed |
| Not all mandatory rubric items applied (incomplete rubric) |
| Submission file cannot be loaded by the reviewer |

---

## 2. Star rating

Only applies **if the submission passes**. Rejected = no rating.

| Stars | Requirement |
| :--- | :--- |
| ★ | All requirements met, but there is an indication of **plagiarism** (someone else's project, content merely altered) |
| ★★ | All requirements met, but **code and report are messy** |
| ★★★ | All requirements met, code and report are reasonably good |
| ★★★★ | All met + **at least 3** of the Additional Assessment Rubric criteria applied |
| ★★★★★ | All met + **all 6** Additional Assessment Rubric criteria applied |

The 6 additional criteria are the "tambahan" column across the six report categories — see §4.

---

## 3. Report structure (module 6.2)

Template: **https://github.com/dicodingacademy/contoh-laporan-mlt** → `format_laporan_submission_1.md`

Title line: `# Laporan Proyek Machine Learning - <Nama Anda>`

```
## Domain Proyek
## Business Understanding
    ### Problem Statements
    ### Goals
    ### Solution statements          <- additional rubric
## Data Understanding
    ### Variabel-variabel pada <dataset> adalah sebagai berikut:
## Data Preparation
## Modeling
## Evaluation
---Ini adalah bagian akhir laporan---
```

Section order is fixed. **Data Preparation techniques must appear in the notebook and the report in the same order** — this is explicitly checked.

---

## 4. Mandatory vs additional rubric

### Wajib — must be satisfied to pass

| Category | Requirement |
| :--- | :--- |
| **Domain Proyek** | Relevant background for the project raised. |
| **Business Understanding** | Explain the problem-clarification process. Must contain **Problem Statements** and **Goals**, all clearly elaborated. |
| **Data Understanding** | Row count, data condition, info about the data. **Include the dataset download link.** Describe **every variable/feature**. |
| **Data Preparation** | Apply **and name** the techniques used. Notebook and report order must match. |
| **Modeling** | Build an ML model. Explain the stages and the **parameters** used. |
| **Evaluation** | State the metric(s), explain results against them. **Metric must fit the data context, problem statement, and desired solution.** |
| **Struktur Laporan** | Organized per template. Code snippets only where they explain something — not the whole project. Images/resources must render in Markdown. |

### Tambahan — the 6 star levers

- [ ] **1. Domain Proyek** — Explain *why* and *how* the problem must be solved. Include related research / references from credible sources with a clear author. Cite IEEE or APA (guide: https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf). Source hunting: https://scholar.google.com/
- [ ] **2. Business Understanding** — Add a **Solution Statement** section:
      - Propose **2 or more** solutions — either **two or more algorithms**, or an **improvement on a baseline model via hyperparameter tuning**.
      - Every solution must be **measurable by an evaluation metric**.
- [ ] **3. Data Understanding** — Do the extra steps: **data visualization** and/or **EDA**.
- [ ] **4. Data Preparation** — Explain *what* each step does **and why it is necessary**.
- [ ] **5. Modeling** — Explain **pros and cons of each algorithm**. One algorithm → **hyperparameter tuning** + explain the improvement. Two or more → **pick the best** and explain **why**.
- [ ] **6. Evaluation** — Explain the metric's **formula** and **how it works**.

### Shortest path to ★★★★★

Do all 6. Note levers 2 and 5 overlap: **two algorithms + hyperparameter tuning + best-model justification** satisfies both branches of lever 5 and lever 2 in one pass, and lever 6 is just writing the metric formula out. Levers 1, 3, 4 are writing effort, not extra modeling.

---

## 5. Submission package

Module 6.1 specifies **exactly 3 files** in the `.zip`:

```text
submission.zip
├── notebook.ipynb    # EXECUTED — outputs must be visible
├── <name>.py         # Python script
└── Laporan_Proyek_Machine_Learning.md
```

No dataset CSV, no `requirements.txt`, no `model/` folder required by the spec. Add them only if
you want to; the reviewer loads the zip, so keep it small and loadable.

### Tips from the module

- Domain is free choice, suggested (not limited to): Kesehatan · Ekonomi dan bisnis · Keuangan · Pertanian dan peternakan · Pendidikan · Lingkungan · Astronomi · Kelautan · Sosial · Telekomunikasi · dsb.
- Exporting from Colab: **File → Download .ipynb** and **Download .py**.
- Training on Colab from Drive data: https://www.youtube.com/watch?v=Gvwuyx_F-28&t=384s

---

## 6. Build order

1. Pick domain + dataset (≥500 samples, quantitative) and the approach: classification / regression / time series.
2. Notebook text cells for every stage — this is a hard requirement, write them as you go.
3. Data Understanding — shape, condition, full feature table, dataset link.
4. EDA / visualization *(lever 3)*.
5. Data Preparation — ordered list, each step named + justified *(lever 4)*.
6. Modeling — two algorithms, hyperparameter-tune, pros/cons of each *(levers 2, 5)*.
7. Evaluation — pick the best model, justify it, state metric formulas, interpret *(lever 6)*.
8. Write the report in §3 order, keeping section order identical to the notebook.
9. Add references, IEEE/APA *(lever 1)*.
10. Execute the notebook end-to-end, export `.ipynb` + `.py`, zip the 3 files.

---

## 7. Reference links

- Report template: https://github.com/dicodingacademy/contoh-laporan-mlt
- Mastering Markdown: https://guides.github.com/features/mastering-markdown/
- Markdown editor: https://dillinger.io/
- IEEE citation guide: https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf
- Google Scholar: https://scholar.google.com/
