# Proyek Akhir: Menyelesaikan Permasalahan HR Attrition - PT Jaya Jaya Maju

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
