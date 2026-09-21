# ML Submission 1: Proyek Pertama — Predictive Analytics

> **Course**: Machine Learning Terapan — Dicoding Indonesia (academy 319)
> **Status**: ✅ **Approved / Sudah di-approve** (Submission ID 5049831)

---

## 📋 Overview

Submission pertama kelas Machine Learning Terapan. Proyek membangun model **klasifikasi biner** untuk
memprediksi risiko gagal bayar nasabah kartu kredit, memakai *Default of Credit Card Clients*
dataset dari UCI Machine Learning Repository (30.000 sampel, 23 fitur).

Tiga algoritma dibandingkan — Logistic Regression sebagai *baseline* linear, Random Forest, dan
Gradient Boosting — dilanjutkan *hyperparameter tuning* dengan `GridSearchCV` dan Stratified 5-Fold
Cross-Validation. **Gradient Boosting** terpilih sebagai model terbaik dengan ROC-AUC 0.7820.

---

## 📁 Folder Structure

```text
submission-1/
├── IMPLEMENTATION.md             # Spec analysis: requirements, rubric, build order
├── credit_default.csv            # Source dataset (mirror)
├── artifact/                     # Reference materials
│   ├── Proyek Pertama Kriteria Submission ...html   # Module 6.1 — criteria
│   ├── Detail Laporan ...html                       # Module 6.2 — report format
│   └── Review/
│       └── Review 1.html         # Reviewer feedback (earlier rejected attempt)
├── submission/                   # 📦 Files submitted to Dicoding
│   ├── notebook.ipynb            # Executed notebook
│   ├── notebook.py               # Colab-style Python export
│   └── Laporan Proyek Machine Learning - William.md
├── submission.zip                # The uploaded archive (3 files)
└── work/
    ├── build_submission.py       # Build script — regenerates everything above
    ├── results.json              # Metrics emitted by the notebook
    └── build.log
```

---

## 🎯 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (baseline) | 0.7445 | 0.4428 | 0.6014 | 0.5101 | 0.7449 |
| Random Forest | 0.8110 | 0.6388 | 0.3346 | 0.4392 | 0.7586 |
| Random Forest (tuned) | 0.7697 | 0.4835 | 0.6059 | 0.5378 | 0.7763 |
| Gradient Boosting | 0.8198 | 0.6676 | 0.3693 | 0.4755 | 0.7817 |
| **Gradient Boosting (tuned)** | **0.8190** | **0.6657** | 0.3647 | 0.4713 | **0.7820** |

Model terbaik dipilih berdasarkan **ROC-AUC** karena metrik ini tidak bergantung pada *threshold*
klasifikasi dan tidak terpengaruh ketidakseimbangan kelas (kelas gagal bayar 22.12%).

Fitur paling berpengaruh adalah perilaku pembayaran historis — `PAY_0` (0.5324), `MAX_DELAY`
(0.1477), `NUM_DELAYS` (0.0969) — bukan karakteristik demografis.

---

## 📝 Reviewer Feedback & Resolution

`artifact/Review/Review 1.html` menyimpan catatan reviewer pada **percobaan yang ditolak**
(Submission ID 5049831). Dua temuan:

1. **Kriteria 8 — Model Development**: seluruh parameter setiap algoritma harus dijelaskan, baik
   pada model *baseline* maupun setelah *hyperparameter tuning*, dan harus sesuai dengan
   implementasi di notebook. Parameter *default* pun harus dinyatakan sebagai *default*.
2. **Evaluation**: tabel metrik dan pemilihan model terbaik harus berada pada tahapan **Evaluasi**,
   bukan **Modeling**. Modeling fokus pada cara kerja algoritma, parameter, dan konfigurasi.

> **Catatan**: berkas di `submission/` adalah build 01:10 — yaitu revisi yang **ditolak**. Revisi
> yang disetujui tidak tersimpan di folder ini. Kedua temuan di atas masih terlihat pada build
> tersebut dan dipertahankan sebagai rujukan ekspektasi rubrik.

Perbaikan kedua temuan tersebut dibawa ke proyek akhir di [`ML/submission-final`](../submission-final).

---

## 🔁 Regenerating

```bash
cd work
python3 build_submission.py
```

Skrip membangun notebook, `notebook.py`, laporan Markdown, menjalankan notebook, lalu mengemas
`submission.zip` berisi tepat 3 berkas. Dataset diunduh otomatis dari UCI bila belum tersedia.

---

## 🔗 Reference

- **Dataset**: [UCI — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Spec**: `IMPLEMENTATION.md`
