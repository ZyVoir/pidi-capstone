# Laporan Proyek Machine Learning - William

**Proyek Pertama: Predictive Analytics — Klasifikasi Risiko Gagal Bayar Kartu Kredit**

---

## Domain Proyek

### Latar Belakang

Industri kartu kredit bertumpu pada satu asumsi dasar: sebagian pemegang kartu akan gagal
memenuhi kewajiban pembayarannya. Ketika kegagalan bayar (*default*) terjadi, kerugian tidak
hanya dialami penerbit kartu, tetapi juga merambat ke sistem keuangan yang lebih luas. Bank
Indonesia mencatat bahwa kredit konsumsi—termasuk kartu kredit—merupakan salah satu segmen yang
paling sensitif terhadap perubahan kualitas daya bayar rumah tangga, sehingga pengelolaan
risikonya menjadi prioritas pengawasan.

Persoalannya, penilaian kelayakan kredit secara konvensional masih banyak mengandalkan penilaian
manual dan indikator keuangan yang terbatas pada satu titik waktu. Pendekatan ini memiliki dua
kelemahan mendasar. Pertama, ia tidak mampu menangkap pola perilaku pembayaran yang bersifat
historis dan multidimensi. Kedua, ia tidak dapat diskalakan untuk memproses ribuan aplikasi dalam
waktu singkat. Padahal data historis nasabah—riwayat tagihan, riwayat pembayaran, dan
keterlambatan bulanan—sebenarnya tersimpan dan dapat dimanfaatkan.

### Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan

Masalah ini harus diselesaikan karena dampaknya bersifat langsung terhadap profitabilitas dan
stabilitas institusi keuangan. Setiap nasabah yang gagal bayar merepresentasikan kerugian pokok
ditambah biaya penagihan, sementara nasabah yang sebenarnya layak tetapi ditolak berarti
kehilangan potensi pendapatan. Kedua jenis kesalahan ini memiliki biaya nyata, sehingga keputusan
kredit harus diambil berdasarkan estimasi risiko yang terkalibrasi, bukan intuisi.

Pendekatannya adalah dengan membangun model klasifikasi biner yang memprediksi status gagal bayar
bulan berikutnya berdasarkan data historis enam bulan terakhir. Model kemudian dievaluasi
menggunakan metrik yang sesuai dengan konteks data yang tidak seimbang (*imbalanced*), sehingga
keputusan yang diambil berbasis angka yang dapat dipertanggungjawabkan.

### Referensi

1. Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive
   accuracy of probability of default of credit card clients. *Expert Systems with Applications*,
   36(2), 2473–2480.
2. Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015). Benchmarking state-of-the-art
   classification algorithms for credit scoring: An update of research. *European Journal of
   Operational Research*, 247(1), 124–136.
3. Barboza, F., Kimura, H., & Altman, E. (2017). Machine learning models and bankruptcy prediction.
   *Expert Systems with Applications*, 83, 405–417.
4. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
5. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine.
   *Annals of Statistics*, 29(5), 1189–1232.

---

## Business Understanding

### Problem Statements

- **Pernyataan Masalah 1:** Faktor-faktor apa saja dari data historis nasabah—meliputi limit
  kredit, demografi, riwayat tagihan, dan riwayat pembayaran—yang paling berpengaruh terhadap
  terjadinya gagal bayar pada bulan berikutnya?
- **Pernyataan Masalah 2:** Bagaimana membangun model machine learning yang mampu memprediksi
  probabilitas seorang nasabah akan gagal bayar pada bulan berikutnya, dengan tingkat akurasi yang
  dapat diandalkan?
- **Pernyataan Masalah 3:** Algoritma mana yang memberikan performa terbaik dalam menyelesaikan
  permasalahan klasifikasi gagal bayar ini, dan mengapa algoritma tersebut dipilih?

### Goals

- **Jawaban Pernyataan Masalah 1:** Mengidentifikasi dan mengukur tingkat kepentingan
  (*feature importance*) setiap fitur terhadap target gagal bayar, sehingga faktor risiko dominan
  dapat diketahui secara kuantitatif.
- **Jawaban Pernyataan Masalah 2:** Menghasilkan model klasifikasi yang mampu memprediksi status
  gagal bayar nasabah dengan performa yang terukur melalui metrik evaluasi yang sesuai untuk data
  tidak seimbang.
- **Jawaban Pernyataan Masalah 3:** Membandingkan performa beberapa algoritma klasifikasi secara
  objektif, kemudian memilih satu model terbaik sebagai solusi akhir beserta alasan pemilihannya.

### Solution statements

Untuk mencapai goals di atas, diajukan tiga solusi yang masing-masing dapat diukur dengan metrik evaluasi:

1. **Logistic Regression sebagai baseline.** Model linear yang sederhana dan *interpretable*,
   digunakan sebagai titik acuan performa. Keberhasilannya diukur melalui ROC-AUC dan F1-Score
   pada kelas minoritas.
2. **Random Forest Classifier.** Algoritma *ensemble* berbasis *bagging* yang mampu menangkap
   hubungan non-linear antar fitur dan relatif tahan terhadap *outlier*. Diukur dengan metrik yang
   sama agar dapat dibandingkan langsung dengan baseline.
3. **Gradient Boosting Classifier.** Algoritma *ensemble* berbasis *boosting* yang membangun model
   secara sekuensial untuk memperbaiki kesalahan model sebelumnya. Diukur dengan metrik yang sama.

Ketiga solusi tersebut kemudian ditingkatkan melalui **hyperparameter tuning** menggunakan
`GridSearchCV` dengan *stratified 5-fold cross-validation*, sehingga perbaikan performa dibuktikan
secara kuantitatif, bukan diasumsikan.

---

## Data Understanding

Dataset yang digunakan adalah **Default of Credit Card Clients Dataset** dari UCI Machine Learning
Repository. Dataset ini merekam data nasabah kartu kredit di Taiwan pada periode April–September
2005, mencakup informasi demografi, limit kredit, riwayat tagihan enam bulan, dan riwayat
pembayaran enam bulan.

- **Sumber data (tautan unduh):** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **Jumlah data:** 30,000 baris dan 31 kolom
  (29 fitur prediktor setelah `ID` dibuang dan fitur turunan ditambahkan)
- **Kondisi data:** tidak terdapat nilai hilang (*missing value*) maupun baris duplikat
- **Jenis data:** kuantitatif dan kategorikal terenkode numerik
- **Target:** `default payment next month` — 1 jika nasabah gagal bayar bulan berikutnya, 0 jika tidak
- **Distribusi kelas:** kelas 0 sebanyak 23,364 (77.88%)
  dan kelas 1 sebanyak 6,636 (22.12%), sehingga data
  tergolong **tidak seimbang** (*imbalanced*)

### Variabel-variabel pada Default of Credit Card Clients dataset adalah sebagai berikut:

| Variabel | Tipe | Keterangan |
| :--- | :--- | :--- |
| `ID` | int | Identitas unik nasabah. Tidak digunakan sebagai fitur karena tidak memiliki daya prediksi. |
| `LIMIT_BAL` | float | Jumlah limit kredit yang diberikan, dalam Dolar Taiwan Baru (NT$). |
| `SEX` | kategori | Jenis kelamin: 1 = laki-laki, 2 = perempuan. |
| `EDUCATION` | kategori | Tingkat pendidikan: 1 = pascasarjana, 2 = universitas, 3 = SMA, 4 = lainnya. |
| `MARRIAGE` | kategori | Status pernikahan: 1 = menikah, 2 = lajang, 3 = lainnya. |
| `AGE` | int | Usia nasabah dalam tahun. |
| `PAY_0` | ordinal | Status pembayaran September 2005: -2 = tidak ada konsumsi, -1 = bayar penuh, 0 = kredit berputar, 1–9 = bulan keterlambatan. |
| `PAY_2` | ordinal | Status pembayaran Agustus 2005 (skala sama dengan `PAY_0`). |
| `PAY_3` | ordinal | Status pembayaran Juli 2005 (skala sama dengan `PAY_0`). |
| `PAY_4` | ordinal | Status pembayaran Juni 2005 (skala sama dengan `PAY_0`). |
| `PAY_5` | ordinal | Status pembayaran Mei 2005 (skala sama dengan `PAY_0`). |
| `PAY_6` | ordinal | Status pembayaran April 2005 (skala sama dengan `PAY_0`). |
| `BILL_AMT1`–`BILL_AMT6` | float | Jumlah tagihan bulan September 2005 hingga April 2005 (NT$). |
| `PAY_AMT1`–`PAY_AMT6` | float | Jumlah pembayaran bulan September 2005 hingga April 2005 (NT$). |
| `default payment next month` | target | 1 = gagal bayar bulan berikutnya, 0 = tidak gagal bayar. |

**Catatan penting mengenai `PAY_0`:** pada dataset asli UCI, kolom ini sebenarnya merepresentasikan
status pembayaran bulan September 2005 (bulan terakhir jendela observasi) dan semestinya diberi
nama `PAY_1`. Penamaan `PAY_0` dipertahankan agar konsisten dengan sumber asli, namun perlu
dipahami bahwa kolom ini adalah **status pembayaran paling mutakhir** dan karena itu sangat prediktif.

### Exploratory Data Analysis

EDA dilakukan dalam tiga tahap yang divisualisasikan pada notebook:

1. **Distribusi fitur numerik** (`LIMIT_BAL`, `AGE`, `BILL_AMT1`, `PAY_AMT1`, `PAY_AMT2`) —
   menunjukkan bahwa fitur tagihan dan pembayaran memiliki sebaran sangat *right-skewed*,
   mengindikasikan adanya nasabah dengan nilai transaksi ekstrem.
2. **Matriks korelasi** — memperlihatkan bahwa fitur status pembayaran (`PAY_0`, `PAY_2`, `PAY_3`)
   memiliki korelasi positif paling kuat terhadap target, sedangkan fitur demografis
   (`AGE`, `SEX`, `MARRIAGE`) nyaris tidak berkorelasi.
3. **Analisis bivariat** — tingkat gagal bayar menurut `PAY_0`, `EDUCATION`, kelompok limit kredit,
   dan distribusi usia per status target.

**Temuan kunci EDA:** tingkat gagal bayar meningkat tajam seiring memburuknya status pembayaran
terakhir, dan menurun pada kelompok limit kredit yang lebih tinggi. Sebaliknya, faktor demografis
menunjukkan perbedaan yang tipis. Temuan ini menjadi dasar pemilihan fitur turunan pada tahap
*data preparation*.

---

## Data Preparation

Tahapan *data preparation* dilakukan secara berurutan. Urutan di bawah ini identik dengan urutan
code cell pada notebook.

### 1. Menyeragamkan nama kolom target

Kolom `default payment next month` diubah menjadi `default`. Nama kolom asli mengandung spasi
sehingga menyulitkan penulisan kode; penamaan ringkas mengurangi risiko kesalahan penulisan.

### 2. Membersihkan nilai kategori yang tidak terdefinisi

Pada `EDUCATION`, nilai 0, 5, dan 6 digabungkan ke kategori 4 (*others*). Pada `MARRIAGE`, nilai 0
digabungkan ke kategori 3 (*others*). Dokumentasi resmi UCI hanya mendefinisikan `EDUCATION` =
{1, 2, 3, 4} dan `MARRIAGE` = {1, 2, 3}, sehingga nilai di luarnya merupakan kategori tidak
terdokumentasi. Jika dibiarkan, model akan memperlakukannya sebagai kategori tersendiri dengan
jumlah sampel sangat sedikit—kondisi yang memicu *overfitting* pada kategori langka.

### 3. Rekayasa fitur (*feature engineering*)

Dibentuk enam fitur turunan baru:

| Fitur baru | Formula | Rasional |
| :--- | :--- | :--- |
| `TOTAL_BILL` | Σ `BILL_AMT1..6` | Total tagihan enam bulan; merangkum besaran eksposur kredit. |
| `TOTAL_PAY` | Σ `PAY_AMT1..6` | Total pembayaran enam bulan; merangkum kemampuan bayar aktual. |
| `AVG_UTILIZATION` | mean(`BILL_AMT1..6`) / `LIMIT_BAL` | Porsi limit yang terpakai; indikator tekanan likuiditas. |
| `PAY_TO_BILL` | `TOTAL_PAY` / `TOTAL_BILL` | Rasio pembayaran terhadap tagihan; mengukur disiplin pembayaran. |
| `MAX_DELAY` | max(`PAY_0..PAY_6`) | Keterlambatan terburuk; menangkap risiko ekstrem. |
| `NUM_DELAYS` | Σ (`PAY_x` ≥ 1) | Frekuensi keterlambatan; menangkap pola berulang. |

Fitur mentah per bulan bersifat *sparse* dan terpisah-pisah, sehingga model harus mempelajari
sendiri bahwa tagihan bulan ke-1 hingga ke-6 saling berkaitan. Fitur agregat menyajikan informasi
tersebut secara langsung. `AVG_UTILIZATION` dan `PAY_TO_BILL` khususnya mengubah besaran absolut
menjadi rasio yang dapat dibandingkan antar nasabah dengan limit kredit berbeda.

### 4. Menghapus kolom yang tidak informatif

Kolom `ID` dihapus karena hanya penanda unik baris tanpa makna ekonomi. Membiarkannya berisiko
membuat model menghafal pola berdasarkan identitas baris alih-alih mempelajari perilaku kredit.

### 5. Pembagian data latih dan data uji

Data dibagi 80% latih dan 20% uji dengan `stratify=y`. Model harus dievaluasi pada data yang belum
pernah dilihat agar hasilnya mencerminkan kemampuan generalisasi. Stratifikasi menjaga proporsi
kelas minoritas tetap sama pada kedua bagian—penting karena kelas gagal bayar hanya
22.12% dari total data.

### 6. Standardisasi fitur

Seluruh fitur numerik distandardisasi dengan `StandardScaler` yang di-*fit* **hanya pada data
latih**. Fitur pada dataset ini memiliki skala yang sangat berbeda—`AGE` bernilai puluhan sedangkan
`TOTAL_BILL` dapat bernilai ratusan juta. Logistic Regression sangat sensitif terhadap perbedaan
skala ini karena mengoptimalkan koefisien melalui jarak, sehingga fitur berskala besar akan
mendominasi. Scaler hanya di-*fit* pada data latih untuk mencegah kebocoran informasi dari data uji.

---

## Modeling

Pemodelan dilakukan dalam dua fase. Fase pertama melatih tiga algoritma pada konfigurasi awal;
fase kedua melakukan *hyperparameter tuning* pada dua algoritma *tree-based* (Random Forest dan
Gradient Boosting). Logistic Regression tidak di-*tuning* karena tidak memiliki *hyperparameter*
struktural yang bermakna—perannya murni sebagai baseline linear.

### Algoritma yang Digunakan

#### 1. Logistic Regression (baseline)

Model linear yang memodelkan probabilitas kelas melalui fungsi sigmoid terhadap kombinasi linear fitur.

- **Kelebihan:** sangat cepat dilatih, mudah diinterpretasikan melalui nilai koefisien, menghasilkan
  probabilitas yang terkalibrasi baik, dan kecil risiko *overfitting* pada data berdimensi sedang.
- **Kekurangan:** hanya mampu menangkap hubungan linear antara fitur dan *log-odds* target. Pada
  data kredit yang penuh interaksi non-linear antar fitur, kemampuan ini menjadi batasan nyata.

#### 2. Random Forest Classifier

*Ensemble* dari banyak pohon keputusan yang dilatih pada sampel *bootstrap* berbeda dengan pemilihan
fitur acak pada setiap percabangan, lalu digabungkan melalui *majority voting*.

- **Kelebihan:** mampu menangkap hubungan non-linear dan interaksi antar fitur secara otomatis;
  relatif tahan terhadap *outlier* dan fitur berskala berbeda; menyediakan *feature importance*
  sehingga hasilnya dapat dijelaskan; risiko *overfitting* lebih rendah dibanding pohon tunggal
  berkat mekanisme *bagging*.
- **Kekurangan:** secara komputasi lebih berat dan kurang *interpretable* dibanding model linear;
  cenderung bias pada fitur kategorikal dengan banyak level; ukuran *feature importance* dapat
  terpecah antar fitur yang berkorelasi tinggi.

#### 3. Gradient Boosting Classifier

*Ensemble* yang membangun pohon secara **sekuensial**, di mana setiap pohon baru dilatih untuk
memperbaiki *residual error* pohon-pohon sebelumnya.

- **Kelebihan:** umumnya memberikan akurasi tertinggi pada data tabular; mampu memodelkan pola
  kompleks dengan jumlah pohon relatif sedikit; memiliki regularisasi bawaan (*learning rate*,
  kedalaman pohon) untuk mengendalikan *overfitting*.
- **Kekurangan:** pelatihan lebih lambat karena bersifat sekuensial dan tidak dapat diparalelkan
  sepenuhnya; sangat sensitif terhadap *hyperparameter*; lebih rentan *overfitting* bila jumlah
  estimator terlalu besar tanpa pengaturan *learning rate* yang tepat.

### Penanganan Ketidakseimbangan Kelas

Kelas gagal bayar hanya mencakup 22.12% data. Bila dibiarkan, model cenderung
memprediksi kelas mayoritas dan mengabaikan kelas minoritas—padahal justru kelas minoritas yang
menjadi objek prediksi. Karena itu, `class_weight="balanced"` diterapkan pada Logistic Regression
dan Random Forest, sehingga bobot kelas minoritas ditingkatkan secara proporsional terhadap
frekuensinya.

### Fase 1 — Performa Konfigurasi Awal

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7445 | 0.4428 | 0.6014 | 0.5101 | 0.7449 |
| **Random Forest** | 0.8110 | 0.6388 | 0.3346 | 0.4392 | 0.7586 |
| **Gradient Boosting** | 0.8198 | 0.6676 | 0.3693 | 0.4755 | 0.7817 |

### Fase 2 — Hyperparameter Tuning

Tuning dilakukan dengan `GridSearchCV` menggunakan **Stratified 5-Fold Cross-Validation** dan
metrik *scoring* ROC-AUC. Grid parameter yang diuji:

- **Random Forest:** `n_estimators` ∈ {200}, `max_depth` ∈ {8, 12, None}, `min_samples_leaf` ∈ {1, 3}
- **Gradient Boosting:** `n_estimators` ∈ {100, 200}, `learning_rate` ∈ {0.05, 0.1}, `max_depth` ∈ {3}

Konfigurasi terpilih untuk masing-masing model:

- **Random Forest:** `n_estimators=200`, `max_depth=8`, `min_samples_leaf=3`
- **Gradient Boosting:** `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, `min_samples_leaf=1`

### Pemilihan Model Terbaik

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.7697 | 0.4835 | 0.6059 | 0.5378 | 0.7763 |
| **Gradient Boosting** | 0.8190 | 0.6657 | 0.3647 | 0.4713 | 0.7820 |

**Model terbaik adalah Gradient Boosting** dengan ROC-AUC 0.7820.

Alasan pemilihannya:

1. **ROC-AUC tertinggi** di antara seluruh model yang diuji. ROC-AUC dipilih sebagai acuan karena
   tidak bergantung pada *threshold* klasifikasi dan tidak terpengaruh oleh ketidakseimbangan kelas.
2. **Kemampuan menangkap hubungan non-linear.** Berbeda dengan Logistic Regression yang hanya
   memodelkan hubungan linear, Gradient Boosting mampu menangkap interaksi kompleks antar fitur perilaku
   pembayaran—sesuai dengan temuan EDA bahwa pengaruh status pembayaran terhadap gagal bayar
   bersifat tidak linear (melonjak pada keterlambatan tertentu).
3. **Mekanisme boosting memperbaiki kesalahan secara iteratif.** Setiap pohon dilatih untuk
   mengoreksi kesalahan pohon sebelumnya, sehingga model akhirnya lebih fokus pada sampel yang
   sulit diklasifikasi—tepat pada kelas minoritas yang menjadi perhatian.

### Perbandingan Sebelum dan Sesudah Tuning

Tabel berikut membandingkan performa setiap model yang di-*tuning* pada konfigurasi awal
(`awal`) dengan konfigurasi hasil `GridSearchCV` (`tuned`), sehingga efek tuning dapat
dibaca per model:

| Model | Metrik | Awal | Tuned | Selisih |
| :--- | :--- | :---: | :---: | :---: |
| Random Forest | Accuracy | 0.8110 | 0.7697 | -0.0413 |
| Random Forest | Precision | 0.6388 | 0.4835 | -0.1553 |
| Random Forest | Recall | 0.3346 | 0.6059 | +0.2713 |
| Random Forest | F1-Score | 0.4392 | 0.5378 | +0.0986 |
| Random Forest | ROC-AUC | 0.7586 | 0.7763 | +0.0177 |
| Gradient Boosting | Accuracy | 0.8198 | 0.8190 | -0.0008 |
| Gradient Boosting | Precision | 0.6676 | 0.6657 | -0.0019 |
| Gradient Boosting | Recall | 0.3693 | 0.3647 | -0.0046 |
| Gradient Boosting | F1-Score | 0.4755 | 0.4713 | -0.0042 |
| Gradient Boosting | ROC-AUC | 0.7817 | 0.7820 | +0.0003 |

Tuning meningkatkan ROC-AUC pada kedua model. Pada Random Forest peningkatannya paling
terasa karena konfigurasi awal memakai pohon tanpa batas kedalaman (`max_depth=None`) yang
rentan *overfitting*; pembatasan `max_depth` dan `min_samples_leaf` menaikkan kemampuan
generalisasi secara jelas. Pada Gradient Boosting peningkatannya tipis karena konfigurasi
awalnya sudah mendekati optimal untuk grid yang diuji.

---

## Evaluation

### Metrik Evaluasi yang Digunakan

Konteks permasalahan ini adalah klasifikasi biner pada data yang **tidak seimbang** (kelas gagal
bayar 22.12%), di mana biaya kesalahan tidak simetris: melewatkan nasabah
yang akan gagal bayar (*false negative*) jauh lebih merugikan daripada salah menandai nasabah lancar
sebagai berisiko (*false positive*). Karena itu, **akurasi saja tidak cukup**—model yang selalu
memprediksi "lancar" akan mencapai akurasi 77.88% tanpa mendeteksi satu pun gagal bayar.

#### 1. Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Menghitung proporsi prediksi yang benar dari keseluruhan prediksi. Kelemahannya pada data tidak
seimbang adalah metrik ini memberi bobot sama pada kedua kelas, sehingga performa baik pada kelas
mayoritas dapat menutupi kegagalan total pada kelas minoritas.

#### 2. Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

Dari semua nasabah yang **ditandai** berisiko gagal bayar, berapa proporsi yang benar-benar gagal
bayar. Precision rendah berarti banyak nasabah lancar yang keliru ditolak, yang berarti kehilangan
nasabah potensial.

#### 3. Recall

$$\text{Recall} = \frac{TP}{TP + FN}$$

Dari semua nasabah yang **benar-benar** gagal bayar, berapa proporsi yang berhasil terdeteksi model.
Recall rendah berarti banyak nasabah berisiko yang lolos tanpa peringatan—kesalahan yang paling
mahal dalam konteks manajemen risiko kredit.

#### 4. F1-Score

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Rata-rata harmonik dari precision dan recall. Karena menggunakan rata-rata harmonik, nilai F1 akan
rendah apabila salah satu komponen rendah—sehingga metrik ini tidak dapat "ditipu" dengan
mengorbankan salah satu sisi.

#### 5. ROC-AUC

$$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})$$

ROC-AUC sama dengan probabilitas bahwa model memberi skor risiko lebih tinggi pada satu nasabah yang
benar-benar gagal bayar dibanding satu nasabah yang lancar, dipilih secara acak. Nilai 0,5 berarti
model tidak lebih baik dari tebakan acak, sedangkan 1,0 berarti pemisahan sempurna. Metrik ini
dipilih sebagai **acuan pemilihan model** karena tidak bergantung pada *threshold*.

### Hasil Evaluasi Model Terbaik — Gradient Boosting

**Confusion Matrix:**

| | Diprediksi Lancar | Diprediksi Gagal Bayar |
| :--- | :---: | :---: |
| **Aktual Lancar** | TN = 4,430 | FP = 243 |
| **Aktual Gagal Bayar** | FN = 843 | TP = 484 |

**Performa akhir:**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.7697 | 0.4835 | 0.6059 | 0.5378 | 0.7763 |
| **Gradient Boosting** | 0.8190 | 0.6657 | 0.3647 | 0.4713 | 0.7820 |

**Interpretasi hasil:**

- Model mencapai **ROC-AUC 0.7820**, yang berarti model mampu membedakan nasabah
  berisiko dan tidak berisiko jauh lebih baik daripada tebakan acak (0,5).
- Dari 1,327 nasabah yang benar-benar gagal bayar pada data uji, model berhasil
  mendeteksi **484 nasabah (recall 0.3647)**.
- **243 nasabah lancar** keliru ditandai berisiko. Dalam praktik, kelompok ini dapat
  ditindaklanjuti dengan verifikasi manual tambahan alih-alih penolakan langsung, sehingga biaya
  kesalahannya dapat ditekan.
- Metrik yang digunakan sesuai dengan konteks data (klasifikasi biner tidak seimbang), problem
  statement (memprediksi probabilitas gagal bayar), dan solusi yang diinginkan (model yang mampu
  mendeteksi risiko secara andal).

### Feature Importance — Menjawab Pernyataan Masalah 1

| Peringkat | Fitur | Importance |
| :---: | :--- | :---: |
| 1 | `PAY_0` | 0.5324 |
| 2 | `MAX_DELAY` | 0.1477 |
| 3 | `NUM_DELAYS` | 0.0969 |
| 4 | `PAY_TO_BILL` | 0.0263 |
| 5 | `AVG_UTILIZATION` | 0.0261 |
| 6 | `TOTAL_BILL` | 0.0257 |
| 7 | `TOTAL_PAY` | 0.0220 |
| 8 | `BILL_AMT1` | 0.0185 |
| 9 | `LIMIT_BAL` | 0.0176 |
| 10 | `PAY_AMT2` | 0.0110 |
| 11 | `AGE` | 0.0102 |
| 12 | `PAY_AMT1` | 0.0069 |
| 13 | `PAY_2` | 0.0067 |
| 14 | `EDUCATION` | 0.0066 |
| 15 | `PAY_AMT3` | 0.0063 |

Fitur yang paling berpengaruh adalah fitur yang berkaitan dengan **perilaku pembayaran**
(`PAY_0`, `MAX_DELAY`, `NUM_DELAYS`) dan **pemanfaatan limit kredit** (`AVG_UTILIZATION`).
Sebaliknya, fitur demografis seperti `AGE`, `SEX`, dan `MARRIAGE` berada di peringkat bawah. Ini
menjawab Pernyataan Masalah 1: risiko gagal bayar pada dataset ini lebih ditentukan oleh perilaku
pembayaran historis dibandingkan karakteristik demografis.

---

## Kesimpulan

1. **Faktor paling berpengaruh terhadap gagal bayar** adalah riwayat keterlambatan pembayaran
   (`PAY_0`, `MAX_DELAY`, `NUM_DELAYS`) dan rasio pemanfaatan limit kredit (`AVG_UTILIZATION`).
   Risiko gagal bayar lebih ditentukan oleh **perilaku pembayaran historis** dibandingkan faktor
   demografis. Hal ini menjawab Pernyataan Masalah 1.

2. **Model klasifikasi berhasil dibangun** dan mampu memprediksi probabilitas gagal bayar dengan
   performa terukur, mencapai ROC-AUC 0.7820 pada data uji. Hal ini menjawab
   Pernyataan Masalah 2.

3. **Gradient Boosting adalah model terbaik** berdasarkan ROC-AUC. Hyperparameter tuning melalui `GridSearchCV`
   dengan *stratified 5-fold cross-validation* meningkatkan ROC-AUC pada kedua model yang
   di-*tuning*: Random Forest 0.7586 → 0.7763
   dan Gradient Boosting 0.7817 → 0.7820.
   Peningkatan terbesar diperoleh Random Forest, sedangkan pada Gradient Boosting perbaikannya
   tipis karena konfigurasi awalnya sudah mendekati optimal untuk grid yang diuji. Hal ini
   menjawab Pernyataan Masalah 3.

### Keterbatasan

- Dataset berasal dari satu wilayah (Taiwan) pada periode 2005, sehingga pola yang dipelajari belum
  tentu berlaku pada populasi dan periode yang berbeda.
- Kelas target tidak seimbang (22.12%). Model terbaik yang dipilih berbasis
  ROC-AUC justru memiliki recall rendah pada kelas minoritas, karena ROC-AUC mengukur kualitas
  pemeringkatan risiko dan tidak mengoptimalkan *threshold* klasifikasi. Menurunkan *threshold*
  keputusan dapat menaikkan recall dengan konsekuensi precision turun.
- Model belum mempertimbangkan aspek *fairness*; fitur seperti `SEX`, `EDUCATION`, dan `MARRIAGE`
  berpotensi menimbulkan bias yang perlu ditinjau sebelum penerapan nyata.

**---Ini adalah bagian akhir laporan---**

---

_Catatan:_ Gambar dan visualisasi lengkap beserta output eksekusinya tersedia pada
`notebook.ipynb`. Seluruh angka pada laporan ini dihasilkan langsung dari eksekusi notebook tersebut.
