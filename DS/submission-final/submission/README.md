# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech — Jaya Jaya Institut

## Business Understanding

Jaya Jaya Institut merupakan salah satu perguruan tinggi perguruan terkemuka yang berdiri sejak tahun 2000. Meskipun berhasil melahirkan banyak lulusan dengan prestasi baik, manajemen menghadapi isu serius terkait tingginya tingkat **dropout** siswa yang tidak menyelesaikan pendidikan mereka. Dropout yang tinggi berdampak negatif pada keberlangsungan keuangan institusi dan reputasi sosial universitas. Oleh karena itu, proyek ini berfokus pada analisis data untuk menemukan akar masalah serta melatih model *Machine Learning* sebagai sistem deteksi dini bagi siswa yang berisiko dropout.

### Permasalahan Bisnis
- Apa saja faktor utama yang memengaruhi tingginya tingkat *dropout* di Jaya Jaya Institut?
- Bagaimana menyajikan visualisasi data yang informatif untuk memonitor performa akademik siswa secara real-time?
- Bagaimana membangun model *Machine Learning* yang dapat memprediksi potensi risiko *dropout* siswa secara akurat untuk pencegahan dini?

### Cakupan Proyek
- **Exploratory Data Analysis (EDA)**: Menjelajahi faktor demografis, finansial, dan akademik mahasiswa untuk mencari korelasi dropout.
- **Business Dashboard**: Membangun visualisasi interaktif (`William_dicoding-dashboard.png` dan Metabase database `metabase.db.mv.db`) untuk memudahkan monitoring performa siswa.
- **Machine Learning**: Melatih model klasifikasi 3-kelas (*Random Forest*) untuk memprediksi status siswa (Dropout, Enrolled, Graduate).
- **Deployment Prototype**: Membuat prototype interaktif berbasis **Streamlit** (`app.py`) dan menghubungkannya ke **Streamlit Community Cloud** agar dapat diakses secara daring/remote.

---

## Persiapan Proyek

### 1. Sumber Data (Dataset)
Dataset performa siswa diperoleh secara resmi dari:
- **Tautan Unduhan Dataset**: [Dicoding Academy Students Performance Dataset (GitHub)](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)
- **Nama File**: `data.csv` (berkas ini sudah disalin dan disimpan di folder `data/data.csv` pada direktori ini untuk kemudahan akses).

### 2. Setup Environment (`venv`)

**Spesifikasi Versi**: Proyek ini menggunakan **Python 3.10.x** (disarankan menggunakan Python 3.10 atau versi 3.9+). 

Gunakan Virtual Environment untuk memastikan kestabilan dan isolasi library dependencies:

```bash
# Buka direktori proyek
cd submission/

# Membuat Virtual Environment bernama 'venv' menggunakan python3.10
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

Proyek ini menyediakan berkas database Metabase (`metabase.db.mv.db`) yang telah dikonfigurasi dengan visualisasi interaktif performa siswa.

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

## Menjalankan Sistem Machine Learning (Streamlit Prototype)

Sistem machine learning dideploy secara interaktif menggunakan Streamlit. Anda dapat menjalankannya baik secara lokal maupun mengakses versi cloud.

### 1. Menjalankan Streamlit secara Lokal:
Setelah menginstal dependencies di Virtual Environment, jalankan perintah berikut:
```bash
streamlit run app.py
```
Aplikasi dapat diakses melalui browser di alamat `http://localhost:8501`.

### 2. Akses Streamlit Community Cloud (Remote URL):
Aplikasi ini juga telah dihosting secara publik pada Streamlit Community Cloud dan dapat diakses kapan saja melalui tautan berikut:
- **Tautan Streamlit App**: [Jaya Jaya Institut - Student Retention Classifier](https://pidi-capstone-qrcmtocanq3t2edwdncam8.streamlit.app/) *(Live Deployment)*

---

## Conclusion & Rekomendasi Action Items

### 1. Ringkasan Performa Model Machine Learning
Model Machine Learning yang dilatih menghasilkan performa evaluasi sebagai berikut (selaras 100% dengan eksekusi `notebook.ipynb`):
- **Accuracy**: 0.9242 (92.42%)
- **Precision**: 0.9241 (92.41%)
- **Recall**: 0.9242 (92.42%)
- **F1-Score**: 0.9240 (92.40%)

### 2. Ringkasan Insight Utama EDA
- **Faktor Finansial (UKT & Hutang)**: Pembayaran UKT tepat waktu menjadi pembatas paling krusial terhadap keberlanjutan siswa. Mahasiswa yang tidak membayar UKT tepat waktu (*Tuition_fees_up_to_date = 0*) menunjukkan tingkat dropout yang sangat tinggi.
- **Dukungan Keuangan (Scholarship)**: Mahasiswa penerima beasiswa memiliki peluang kelulusan (*Graduate*) yang jauh lebih tinggi dan tingkat dropout yang sangat rendah dibandingkan mahasiswa non-beasiswa.
- **Kinerja Akademik Semester 1 & 2**: Jumlah SKS/mata kuliah yang disetujui (*approved*) pada semester 1 & 2 adalah fitur dengan tingkat pengaruh tertinggi dalam memisahkan siswa lulus vs dropout.

### 3. Rekomendasi Action Items
1. **Skema Bantuan Finansial untuk Mahasiswa Menunggak**: Menerapkan program pembayaran bertahap (cicilan) bagi mahasiswa yang mengalami kesulitan keuangan agar status UKT tetap teratur dan menghindari dropout karena alasan ekonomi.
2. **Sistem Peringatan Akademik Berbasis Peringatan Dini**: Mengintegrasikan Streamlit App (`app.py`) pada sistem portal akademik universitas. Mahasiswa dengan SKS disetujui kurang dari 4 pada semester 1 otomatis disaring untuk program pembinaan khusus.
3. **Ekspansi Beasiswa Sasaran**: Menyediakan beasiswa darurat (emergency scholarship) bagi mahasiswa berprestasi yang tiba-tiba mengalami kesulitan ekonomi di tengah masa studi.
