# Proyek Akhir: Menyelesaikan Permasalahan HR Attrition - PT Jaya Jaya Maju

## Business Understanding

PT Jaya Jaya Maju merupakan salah satu perusahaan multinasional yang berdiri sejak tahun 2000 dengan lebih dari 1.000 karyawan. Meskipun telah menjadi perusahaan besar, pengelolaan karyawan masih menjadi tantangan utama yang berimbas pada tingginya **attrition rate** (rasio karyawan keluar) melebihi **10%**.

### Permasalahan Bisnis
- Apa saja faktor utama yang memengaruhi tingginya tingkat *attrition* karyawan?
- Bagaimana menyajikan visualisasi data interaktif melalui Business Dashboard untuk memonitor faktor risiko *attrition*?
- Bagaimana membangun model *Machine Learning* yang mampu memprediksi potensi *attrition* karyawan secara akurat untuk pencegahan dini?

### Cakupan Proyek
- Exploratory Data Analysis (EDA) pada dataset karyawan Jaya Jaya Maju (Univariate & Multivariate).
- Pengembangan Business Dashboard eksekutif (`testuser_dicoding-dashboard.png` dan Metabase Instance `metabase.db.mv.db`).
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
# venv\Scripts\activate.bat

# Pada Windows (PowerShell):
# venv\Scripts\Activate.ps1
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

Dashboard eksekutif (`testuser_dicoding-dashboard.png`) menyajikan gambaran menyeluruh terkait indikator kinerja HR dan faktor risiko utama:

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
- **Accuracy**: 0.8491 (84.91%)
- **Precision**: 0.7000 (70.00%)
- **Recall**: 0.1944 (19.44%)
- **F1-Score**: 0.3043 (30.43%)
- **ROC-AUC Score**: 0.7756

### Penggunaan Script Prediksi (`prediction.py`)

Anda dapat menjalankan inferensi risiko *attrition* karyawan secara mandiri menggunakan script `prediction.py`:

```bash
# 1. Jalankan prediksi sampel bawaan
python prediction.py

# 2. Jalankan prediksi menggunakan input file CSV
python prediction.py --file data/employee_data.csv

# 3. Jalankan prediksi menggunakan string JSON data karyawan
python prediction.py --json '{"Age": 28, "OverTime": "Yes", "MonthlyIncome": 2200, "JobLevel": 1}'
```

Contoh Output Prediksi:
```json
[
  {
    "EmployeeIndex": 0,
    "Prediction": 1,
    "Status": "HIGH RISK (Attrition Likely)",
    "AttritionProbability": "79.52%"
  }
]
```

---

## Conclusion (Kesimpulan Utuh)

### 1. Ringkasan Performa Model Machine Learning
Model Machine Learning yang dilatih menghasilkan performa evaluasi sebagai berikut (selaras 100% dengan eksekusi `notebook.ipynb`):
- **Accuracy**: 0.8491 (84.91%)
- **Precision**: 0.7000 (70.00%)
- **Recall**: 0.1944 (19.44%)
- **F1-Score**: 0.3043 (30.43%)
- **ROC-AUC Score**: 0.7756

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
