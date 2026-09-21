# ML Submission Final: Membuat Model Sistem Rekomendasi — Rekomendasi Film

> **Course**: Machine Learning Terapan — Dicoding Indonesia (academy 319)
> **Status**: ✅ **Approved / Sudah di-approve**
> **Note**: status ini ditandai atas permintaan pemilik repo. Tidak ada artefak *approval* dari
> Dicoding yang tersimpan di `artifact/` — yang ada hanya `artifact/Review/Review 1.html`, yaitu
> catatan reviewer untuk **Proyek Pertama** (submission-1), bukan untuk proyek akhir ini.

---

## 📋 Overview

Proyek akhir kelas Machine Learning Terapan. Membangun **sistem rekomendasi film** di atas
*MovieLens Latest Small* (100.836 rating, 610 pengguna, 9.724 film), dengan dua pendekatan berbeda
sebagaimana diminta rubrik *Solution statements*:

1. **Content-Based Filtering** — film direpresentasikan sebagai vektor TF-IDF genre, profil pengguna
   dibentuk dari selisih rating terhadap rata-rata, kemiripan diukur dengan *cosine similarity*.
2. **Collaborative Filtering (SVD)** — matriks rating yang dipusatkan terhadap rata-rata pengguna
   difaktorkan dengan *truncated SVD* menjadi 50 faktor laten.

Keduanya menyajikan **top-N recommendation** sebagai keluaran.

---

## 📁 Folder Structure

```text
submission-final/
├── artifact/                     # Reference materials
│   ├── Proyek Akhir Kriteria Submission ...html   # Criteria + package requirements
│   ├── Detail Laporan ...html                     # Report format template
│   ├── project.md                                 # Parsed project brief
│   └── Review/
│       └── Review 1.html         # Reviewer feedback (Proyek Pertama — see Note above)
├── submission/                   # 📦 Files submitted to Dicoding
│   ├── notebook.ipynb            # Executed notebook (37 cells, 15 code)
│   ├── notebook.py               # Colab-style Python export
│   └── Laporan Proyek Machine Learning - William.md
├── submission.zip                # The uploaded archive (3 files)
└── work/
    ├── build_submission.py       # Build script — regenerates everything above
    └── results.json              # Metrics emitted by the notebook
```

> `submission/` juga memuat `ml-latest-small/` dan `ml-latest-small.zip` sebagai sisa eksekusi
> notebook. Keduanya *gitignored* — notebook mengunduhnya ulang otomatis dari GroupLens.

---

## 🎯 Results

| Model | RMSE | MAE | Precision@10 |
| :--- | :---: | :---: | :---: |
| Baseline rata-rata global | 1.0275 | 0.8146 | — |
| Baseline popularitas | — | — | **0.1202** |
| **Content-Based Filtering** | **0.8502** | **0.6458** | 0.0289 |
| Collaborative Filtering (SVD) | 0.9173 | 0.7062 | 0.1183 |

Tidak ada satu model yang unggul di semua metrik, dan laporan menyatakannya secara terbuka:
*Content-Based Filtering* lebih baik dalam memprediksi **angka** rating (RMSE 0.8502), sedangkan
*Collaborative Filtering* jauh lebih baik dalam menyusun **daftar** rekomendasi (Precision@10
0.1183 vs 0.0289). Karena tujuan proyek adalah menyajikan top-N, **Collaborative Filtering (SVD)**
adalah pilihan yang lebih tepat untuk diterapkan.

**Keduanya masih kalah dari *baseline* popularitas** pada Precision@10 (0.1202), sehingga belum ada
model yang layak langsung dipakai di produksi tanpa perbaikan lebih lanjut. Keterbatasan ini
dinyatakan eksplisit pada bagian Kesimpulan.

---

## 📦 What the Zip Contains

Sesuai **"Ketentuan Berkas Submission"**, arsip berisi tepat **3 berkas**:

| File | Requirement |
| :--- | :--- |
| `notebook.ipynb` | Jupyter Notebook — **sudah dijalankan** (15/15 sel kode, tanpa error) |
| `notebook.py` | File Python |
| `Laporan Proyek Machine Learning - William.md` | Laporan Markdown |

Kondisi yang menyebabkan **penolakan otomatis** — semuanya sudah dihindari: bukan `.zip`, laporan
bukan `.md`, tidak ada berkas `.py`/`.ipynb`, notebook belum dijalankan, rubrik wajib tidak lengkap,
atau berkas tidak bisa di-*load* reviewer.

---

## ✅ Review Mandiri

| # | Checklist | Status |
| :---: | :--- | :---: |
| 1 | Zip berisi 3 berkas (.md, .py, .ipynb) | ✅ |
| 2 | Dokumentasi setiap cell code dengan text cell | ✅ |
| 3 | Dataset bebas, dapat dipakai untuk sistem rekomendasi | ✅ |
| 4 | Memilih pendekatan Content-based / Collaborative Filtering | ✅ |
| 5 | Rubrik Project Overview | ✅ |
| 6 | Rubrik Business Understanding | ✅ |
| 7 | Rubrik Data Understanding | ✅ |
| 8 | Rubrik Data Preparation | ✅ |
| 9 | Rubrik Modeling and Results | ✅ |
| 10 | Rubrik Evaluation | ✅ |
| 11 | Rubrik Struktur Laporan | ✅ |
| 12 | Notebook dijalankan tanpa error | ✅ |

---

## 🔁 Regenerating

```bash
cd work
python3 build_submission.py
```

Skrip membangun notebook, `notebook.py`, laporan Markdown, menjalankan notebook, lalu mengemas
`submission.zip`. Dataset diunduh otomatis dari GroupLens bila belum tersedia.

---

## 🔗 Reference

- **Dataset**: [MovieLens Latest Small — GroupLens Research](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
- **Harper & Konstan (2015)**, *The MovieLens Datasets: History and Context*, ACM TiiS.
