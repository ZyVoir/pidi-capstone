# Laporan Proyek Machine Learning - William

**Proyek Akhir: Membuat Model Sistem Rekomendasi — Rekomendasi Film**

---

## Domain Proyek

### Latar Belakang

Industri layanan streaming film menghadapi persoalan yang oleh para peneliti disebut sebagai
*long tail*: katalog yang tersedia sangat besar, tetapi perhatian pengguna terkonsentrasi pada
segelintir judul populer. Pada dataset yang digunakan dalam proyek ini, 10% film terpopuler
menyerap **60.0%** dari seluruh rating yang tercatat. Akibatnya, pengguna dihadapkan pada
*information overload* — terlalu banyak pilihan, terlalu sedikit panduan — sementara sebagian
besar katalog tidak pernah tersentuh.

Sistem rekomendasi hadir sebagai jawaban atas persoalan tersebut. Alih-alih meminta pengguna
menelusuri ribuan judul, sistem mempelajari preferensi mereka dari riwayat interaksi, lalu
menyajikan sejumlah kecil kandidat yang paling relevan. Ricci, Rokach, dan Shapira (2015)
menempatkan sistem rekomendasi sebagai komponen yang tidak terpisahkan dari platform modern
karena kemampuannya menurunkan biaya pencarian (*search cost*) sekaligus meningkatkan keterlibatan
pengguna.

### Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan

Masalah ini penting untuk diselesaikan karena dampaknya terukur pada dua sisi sekaligus. Bagi
pengguna, rekomendasi yang relevan mempersingkat waktu yang dibutuhkan untuk menemukan tontonan
yang sesuai. Bagi penyedia layanan, rekomendasi yang baik meningkatkan jumlah film yang ditonton
per sesi, memperpanjang masa berlangganan, dan membuka eksposur bagi judul-judul non-populer yang
sebelumnya tidak terlihat.

Pendekatannya adalah membangun dua model rekomendasi dengan asumsi kerja yang berbeda.
*Content-based filtering* memanfaatkan atribut film (genre) sehingga mampu merekomendasikan judul
baru yang belum memiliki rating sama sekali — mengatasi *item cold-start*. *Collaborative
filtering* memanfaatkan matriks rating pengguna-film sehingga mampu menangkap preferensi yang
tidak terlihat dari atribut film. Kedua model dievaluasi dengan metrik yang sama agar
perbandingannya sahih, kemudian model terbaik dipilih berdasarkan hasil kuantitatif tersebut.

### Referensi

1. Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender systems: Introduction and challenges.
   In *Recommender Systems Handbook* (pp. 1–34). Springer.
2. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender
   systems. *Computer*, 42(8), 30–37.
3. Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based recommender systems: State of
   the art and trends. In *Recommender Systems Handbook* (pp. 73–105). Springer.
4. Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering
   recommendation algorithms. In *Proceedings of the 10th International Conference on World Wide
   Web* (pp. 285–295).
5. Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context.
   *ACM Transactions on Interactive Intelligent Systems*, 5(4), 1–19.

---

## Business Understanding

### Problem Statements

- **Pernyataan Masalah 1:** Bagaimana memanfaatkan atribut konten film (genre) untuk
  merekomendasikan film yang relevan bagi seorang pengguna, termasuk untuk film yang belum pernah
  diberi rating oleh siapa pun?
- **Pernyataan Masalah 2:** Bagaimana memanfaatkan pola rating historis seluruh pengguna untuk
  memprediksi rating yang akan diberikan seorang pengguna pada film yang belum ia tonton?
- **Pernyataan Masalah 3:** Pendekatan mana — *content-based filtering* atau *collaborative
  filtering* — yang memberikan kinerja lebih baik pada dataset ini, dan atas dasar apa pilihan
  tersebut diambil?

### Goals

- **Jawaban Pernyataan Masalah 1:** Membangun model *content-based filtering* yang
  merepresentasikan film sebagai vektor fitur genre dan mengukur kemiripan antar film, sehingga
  mampu menghasilkan daftar top-N rekomendasi personal untuk setiap pengguna.
- **Jawaban Pernyataan Masalah 2:** Membangun model *collaborative filtering* berbasis *matrix
  factorization* yang menguraikan matriks rating pengguna-film menjadi faktor laten, sehingga
  mampu mengestimasi rating pada pasangan pengguna-film yang belum teramati.
- **Jawaban Pernyataan Masalah 3:** Mengevaluasi kedua model dengan metrik yang sama
  (RMSE, MAE, dan Precision@10), membandingkan hasilnya secara kuantitatif, lalu memilih model
  terbaik beserta alasan pemilihannya.

### Solution statements

Untuk mencapai goals di atas, diajukan dua solusi yang masing-masing dapat diukur dengan metrik
evaluasi:

1. **Content-Based Filtering (TF-IDF + Cosine Similarity).** Setiap film direpresentasikan sebagai
   vektor TF-IDF dari genre-nya. Tingkat kemiripan antar film dihitung dengan *cosine similarity*.
   Rating diprediksi sebagai rata-rata berbobot rating pengguna pada film-film yang paling mirip
   secara konten. Diukur dengan **RMSE**, **MAE**, dan **Precision@10**.
2. **Collaborative Filtering (Matrix Factorization / SVD).** Matriks rating yang telah
   dinormalisasi terhadap rata-rata pengguna difaktorkan menjadi matriks laten pengguna dan film
   berukuran `k = 50` menggunakan *truncated SVD*. Rating diprediksi dari hasil kali
   faktor laten kedua pihak. Diukur dengan **RMSE**, **MAE**, dan **Precision@10**.

Kedua solusi menghasilkan *top-N recommendation* sebagai keluaran akhir dan dievaluasi pada
himpunan data uji yang identik, sehingga perbandingannya bersifat *apples-to-apples*.

---

## Data Understanding

Dataset yang digunakan adalah **MovieLens Latest Small**, dikelola oleh GroupLens Research,
University of Minnesota. Dataset ini berisi rating film yang diberikan oleh pengguna nyata dan
merupakan salah satu *benchmark* paling umum dalam penelitian sistem rekomendasi
(Harper & Konstan, 2015).

**Tautan unduh:** https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

### Kondisi Data

| Keterangan | Sebelum Persiapan | Setelah Persiapan |
| :--- | ---: | ---: |
| Jumlah rating | 100,836 | 90,274 |
| Jumlah pengguna | 610 | 610 |
| Jumlah film | 9,724 | 3,650 |
| Kepadatan matriks | — | 3.24% |

Setelah tahapan *data preparation*, matriks rating berdimensi
**610 pengguna × 3,650 film** dengan tingkat kepadatan hanya
**3.24%**. Artinya lebih dari 96.8% sel matriks tidak teramati — inilah
yang membuat personalisasi menjadi masalah yang sulit sekaligus menarik. Data dibagi menjadi
**72,219 rating latih** dan **18,055 rating uji**.

Sebaran nilai rating adalah sebagai berikut:

| Rating | Jumlah | Proporsi |
| :---: | ---: | ---: |
| 0.5 | 1,370 | 1.36% |
| 1.0 | 2,811 | 2.79% |
| 1.5 | 1,791 | 1.78% |
| 2.0 | 7,551 | 7.49% |
| 2.5 | 5,550 | 5.50% |
| 3.0 | 20,047 | 19.88% |
| 3.5 | 13,136 | 13.03% |
| 4.0 | 26,818 | 26.60% |
| 4.5 | 8,551 | 8.48% |
| 5.0 | 13,211 | 13.10% |

Rata-rata rating global adalah **3.5391**, dan **39.7%** rating bernilai 4.0 atau 5.0.

### Variabel-variabel pada MovieLens Latest Small dataset adalah sebagai berikut:

**`ratings.csv`**

- `userId` : identitas unik pengguna yang memberikan rating. Bertipe numerik diskret dan
  digunakan sebagai salah satu sumbu matriks rating.
- `movieId` : identitas unik film yang diberi rating. Menjadi kunci penghubung ke `movies.csv`.
- `rating` : nilai rating yang diberikan pengguna, berskala 0.5–5.0 dengan kelipatan 0.5.
  Inilah variabel target yang diprediksi oleh model *collaborative filtering*.
- `timestamp` : waktu pemberian rating dalam format *Unix epoch*. Tidak digunakan dalam pemodelan
  karena proyek ini tidak menangani aspek temporal.

**`movies.csv`**

- `movieId` : identitas unik film, kunci penghubung ke `ratings.csv`.
- `title` : judul film beserta tahun rilis. Digunakan hanya untuk menampilkan hasil rekomendasi
  agar dapat dibaca manusia.
- `genres` : daftar genre film yang dipisahkan karakter `|`, misalnya
  `Adventure|Animation|Children|Comedy|Fantasy`. Variabel inilah yang menjadi fitur konten pada
  model *content-based filtering*.

### Exploratory Data Analysis

**Sebaran rating.** Histogram pada notebook menunjukkan puncak yang jelas pada nilai 4.0. Ini
adalah *positivity bias* yang lazim pada data eksplisit: pengguna lebih sering menonton — dan
memberi rating — pada film yang memang mereka perkirakan disukai. Konsekuensinya, model yang
selalu memprediksi nilai mendekati rata-rata akan tampak cukup baik bila hanya diukur dengan RMSE,
sehingga RMSE perlu didampingi metrik peringkat seperti Precision@10.

**Distribusi rating per film (long tail).** Kurva rating per film yang diurutkan menurut
popularitas turun sangat tajam pada skala logaritmik. Sebagian kecil film menyerap porsi rating
yang jauh melebihi proporsinya, sementara ekor distribusinya panjang. Model *collaborative
filtering* akan kesulitan pada film di ekor distribusi karena bukti yang tersedia terlalu sedikit
— inilah *item cold-start* yang menjadi keunggulan *content-based filtering*.

**Jumlah rating per pengguna.** Distribusinya melebar: ada pengguna dengan riwayat sangat tipis
dan ada yang memberi rating dalam jumlah besar. Pengguna dengan riwayat tipis menghasilkan profil
yang kurang andal, sehingga penyaringan minimum rating diperlukan pada tahap *data preparation*.

**Genre.** Jumlah film dan rata-rata rating per genre adalah sebagai berikut:

| Genre | Jumlah Film | Rata-rata Rating |
| :--- | ---: | ---: |
| Drama | 4,361 | 3.656 |
| Comedy | 3,756 | 3.385 |
| Thriller | 1,894 | 3.494 |
| Action | 1,828 | 3.448 |
| Romance | 1,596 | 3.507 |
| Adventure | 1,263 | 3.509 |
| Crime | 1,199 | 3.658 |
| Sci-Fi | 980 | 3.456 |
| Horror | 978 | 3.258 |
| Fantasy | 779 | 3.491 |
| Children | 664 | 3.413 |
| Animation | 611 | 3.630 |
| Mystery | 573 | 3.632 |
| Documentary | 440 | 3.798 |
| War | 382 | 3.808 |
| Musical | 334 | 3.564 |
| Western | 167 | 3.584 |
| IMAX | 158 | 3.618 |
| Film-Noir | 87 | 3.920 |
| (no genres listed) | 34 | 3.489 |

Drama dan Komedi mendominasi katalog, sedangkan genre seperti *Film-Noir* dan *Documentary* jauh
lebih sedikit. Rata-rata rating antar genre pun berbeda, yang mengonfirmasi bahwa genre membawa
sinyal preferensi yang layak dijadikan fitur konten.

---

## Data Preparation

Tahapan *data preparation* dilakukan dalam urutan berikut. Urutan ini identik dengan yang
dijelaskan pada notebook.

### 1. Menggabungkan rating dengan metadata film

*Mengapa diperlukan:* model *content-based filtering* membutuhkan kolom `genres`, yang hanya
tersedia di `movies.csv`. Penggabungan dilakukan dengan *left join* agar seluruh baris rating
tetap terjaga meskipun ada film yang tidak ditemukan metadatanya.

### 2. Menghapus duplikat dan menangani nilai kosong

*Mengapa diperlukan:* satu pasangan pengguna-film seharusnya hanya memiliki satu rating. Duplikat
akan memberi bobot ganda pada satu interaksi sehingga matriks rating menjadi bias. Baris tanpa
`rating` atau `genres` juga dibuang karena tidak dapat dipakai oleh model mana pun.

### 3. Menyaring pengguna dan film dengan rating minimum

*Mengapa diperlukan:* pengguna dengan riwayat sangat tipis dan film dengan sedikit rating
menghasilkan vektor laten yang tidak stabil pada *collaborative filtering*, sekaligus membuat
evaluasi top-N tidak bermakna. Ambang dipilih `>= 5` rating dan diterapkan secara
iteratif hingga himpunan stabil, karena penyaringan film dapat menurunkan jumlah rating pengguna
dan sebaliknya. Setelah penyaringan, tersisa **610 pengguna**, **3,650 film**,
dan **90,274 rating**.

### 4. Membagi data latih dan data uji (80:20)

*Mengapa diperlukan:* evaluasi harus dilakukan pada data yang tidak dilihat model. Pembagian acak
pada tingkat rating adalah prosedur standar untuk mengukur akurasi prediksi rating. Parameter yang
dipakai: `test_size=0.2` (80% latih / 20% uji) dan `random_state=RANDOM_STATE` (`42`)
agar pembagian dapat direproduksi. Dihasilkan **72,219 rating latih** dan
**18,055 rating uji**.

### 5. Membentuk matriks rating pengguna-film dari data latih

*Mengapa diperlukan:* baik *content-based* maupun *collaborative filtering* bekerja di atas
representasi matriks. Matriks dibangun **hanya dari data latih** agar tidak terjadi *data leakage*;
sel yang tidak teramati bernilai 0 dan ditandai terpisah melalui matriks `observed`.

### 6. Mengekstraksi fitur konten dengan TF-IDF

*Mengapa diperlukan:* genre adalah data kategorikal multi-nilai. TF-IDF mengubahnya menjadi vektor
numerik yang bobotnya menurunkan pengaruh genre yang sangat umum (misalnya Drama) dan menaikkan
pengaruh genre yang lebih spesifik, sehingga kemiripan antar film menjadi lebih informatif
dibandingkan *one-hot encoding* biasa. Dihasilkan matriks fitur konten berdimensi
**3,650 film × 20 genre**.

---

## Modeling

Proyek ini membangun **dua solusi** sistem rekomendasi dengan pendekatan yang berbeda, sesuai
ketentuan *Solution statements*. Keduanya dilatih pada data latih yang sama dan menghasilkan
keluaran berupa **top-N recommendation**. Bagian ini menjelaskan cara kerja dan konfigurasi
parameter setiap solusi; hasil metrik beserta pemilihan model terbaik dibahas pada tahapan
**Evaluation**.

Seluruh nilai parameter di bawah dituliskan persis seperti pada `notebook.ipynb` sehingga dapat
diverifikasi. Parameter yang tidak disebutkan dibiarkan pada nilai *default* pustaka dan
dinyatakan secara eksplisit pada tabel masing-masing solusi.

### Solusi 1 — Content-Based Filtering

Pendekatan ini bertumpu pada asumsi bahwa **film yang mirip secara konten akan disukai oleh
pengguna yang sama**. Setiap film direpresentasikan sebagai vektor TF-IDF genre, lalu kemiripan
antar film dihitung dengan *cosine similarity*:

$$\text{sim}(i, j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\lVert \mathbf{v}_i \rVert \, \lVert \mathbf{v}_j \rVert}$$

Model ini memakai dua rumus untuk dua tugas yang berbeda, sebagaimana lazim pada sistem
rekomendasi berbasis konten.

**(a) Prediksi rating** — rata-rata berbobot selisih rating terhadap rata-rata film, diambil dari
30 film yang paling mirip secara konten:

$$\hat{r}_{ui} = \mu_i + \frac{\sum_{j \in N_k(i)} \text{sim}(i, j) \cdot (r_{uj} - \mu_j)}{\sum_{j \in N_k(i)} \text{sim}(i, j)}$$

Selisih terhadap rata-rata film ($r_{uj} - \mu_j$) dipakai, bukan rating mentah, agar film yang
memang disukai semua orang tidak otomatis dianggap sangat relevan hanya karena rata-ratanya
tinggi. $\mu_i$ adalah rata-rata rating film $i$ pada data latih.

**(b) Skor peringkat top-N** — dibentuk *profil pengguna* berupa vektor genre berbobot, yaitu
jumlah dari vektor TF-IDF film yang pernah dirating, ditimbang dengan selisih rating pengguna
terhadap rata-ratanya sendiri:

$$\mathbf{p}_u = \sum_{j \in I_u} (r_{uj} - \mu_u) \, \mathbf{v}_j, \qquad
\text{skor}(u, i) = \frac{\mathbf{p}_u \cdot \mathbf{v}_i}{\lVert \mathbf{p}_u \rVert \, \lVert \mathbf{v}_i \rVert}$$

Skor ini **tidak** dinormalisasi terhadap jumlah kemiripan. Pembagian semacam itu membuat seluruh
film dengan genre identik memperoleh skor yang persis sama, sehingga urutan top-N menjadi
sewenang-wenang — masalah yang terkonfirmasi saat pengembangan dan sengaja dihindari di sini.
Film yang sudah pernah ditonton dikeluarkan dari daftar kandidat, dan popularitas dipakai sebagai
pemecah seri agar hasilnya deterministik.

**Parameter Solusi 1.**

| Parameter | Nilai | Keterangan |
| :--- | :--- | :--- |
| `TfidfVectorizer.tokenizer` | `lambda s: s.split("\|")` | Memecah kolom `genres` yang multi-nilai (`Action\|Comedy`) menjadi satu token per genre. |
| `TfidfVectorizer.token_pattern` | `None` | *Default* dimatikan agar tokenizer kustom di atas dipakai apa adanya. |
| `TfidfVectorizer.norm` | *default* (`l2`) | Normalisasi L2 pada vektor TF-IDF; tidak diubah. |
| `TfidfVectorizer.use_idf` | *default* (`True`) | IDF aktif sehingga genre umum (Drama) berbobot lebih rendah daripada genre spesifik. |
| `TfidfVectorizer.sublinear_tf` | *default* (`False`) | Tidak diubah; istilah dalam satu film tidak berulang sehingga *sublinear scaling* tidak berpengaruh. |
| `TfidfVectorizer.min_df` / `.max_df` | *default* (`1` / `1.0`) | Tidak ada pemotongan *vocabulary*. Dengan hanya 20 genre, membuang istilah justru menghilangkan informasi. |
| `TfidfVectorizer.stop_words` | *default* (`None`) | Tidak diubah; tidak ada genre yang berperan sebagai *stop word*. |
| `K_NEIGHBORS` | `30` | Jumlah tetangga terdekat pada rumus (a). Dipilih agar prediksi rating distabilkan oleh cukup banyak tetangga tanpa kehilangan sifat lokal; dengan hanya 20 genre, nilai yang jauh lebih kecil membuat prediksi sensitif terhadap satu film saja. |
| `k` pada `scipy.sparse.triu` | `1` | Diagonal matriks kemiripan dibuang agar sebuah film tidak dianggap mirip dengan dirinya sendiri. |
| `cbf_top_n(n)` | `10` | Panjang daftar top-N yang disajikan sebagai keluaran. |
| Pemotongan skor | `clip(0.5, 5.0)` | Seluruh prediksi rating dipotong ke rentang rating yang valid pada dataset. |

Solusi 1 **tidak melalui hyperparameter tuning** dan tidak memiliki parameter yang dilatih:
vektor TF-IDF dan matriks kemiripan dihitung langsung dari data, bukan dioptimalkan terhadap
fungsi kerugian. Karena itu tidak ada *best parameters* untuk dilaporkan pada solusi ini.

**Keluaran top-N.** Berikut rekomendasi untuk pengguna contoh `1`:

| Peringkat | Judul Film |
| :---: | :--- |
| 1 | Anastasia (1997) |
| 2 | Pocahontas (1995) |
| 3 | Hunchback of Notre Dame, The (1996) |
| 4 | Sleeping Beauty (1959) |
| 5 | Ratatouille (2007) |
| 6 | Fox and the Hound, The (1981) |
| 7 | Snow White and the Seven Dwarfs (1937) |
| 8 | Prince of Egypt, The (1998) |
| 9 | Persepolis (2007) |
| 10 | Lion King, The (1994) |

### Solusi 2 — Collaborative Filtering (Matrix Factorization / SVD)

Pendekatan ini bertumpu pada asumsi bahwa **pola rating antar pengguna mengandung informasi yang
tidak terlihat dari atribut film**. Matriks rating $R$ yang telah dinormalisasi terhadap rata-rata
tiap pengguna difaktorkan menjadi dua matriks berdimensi rendah:

$$R \approx U \Sigma V^{T}, \qquad P = U\Sigma, \quad Q = V$$

dengan $k = 50$ faktor laten. Prediksi rating diperoleh dari hasil kali vektor laten pengguna dan
film, dikembalikan ke skala semula melalui rata-rata pengguna:

$$\hat{r}_{ui} = \mu_u + \mathbf{p}_u^{T} \mathbf{q}_i$$

Nilai prediksi dijepit (*clip*) ke rentang sah $[0.5, 5.0]$. Top-N disusun dengan menghitung
$\hat{r}_{ui}$ untuk seluruh film, mengeluarkan film yang sudah pernah ditonton, lalu mengambil
10 skor tertinggi.

**Parameter Solusi 2.**

| Parameter | Nilai | Keterangan |
| :--- | :--- | :--- |
| `K_FACTORS` (`k` pada `scipy.sparse.linalg.svds`) | `50` | Jumlah faktor laten. Menentukan kapasitas model: cukup besar untuk menampung variasi selera, cukup kecil untuk meredam *overfitting* pada matriks ber-*density* hanya 3.24%. |
| `svds(..., k=...)` argumen lain | *default* | `which="LM"` (nilai singular terbesar), `tol=0`, `maxiter=None`, `return_singular_vectors=True` — seluruhnya dibiarkan *default*. |
| `RANDOM_STATE` | `42` | Dipakai pada pembagian data latih/uji agar hasil dapat direproduksi. `svds` sendiri bersifat deterministik dan tidak menerima *seed*. |
| Normalisasi sebelum faktorisasi | `centered = R - μ_u` | Matriks dipusatkan terhadap rata-rata tiap pengguna, dan sel tak teramati dimasking menjadi 0 agar tidak ikut terfaktorisasi. |
| `cf_top_n(n)` | `10` | Panjang daftar top-N yang disajikan sebagai keluaran. |
| Pemotongan skor | `clip(0.5, 5.0)` | Seluruh prediksi rating dipotong ke rentang rating yang valid pada dataset. |

Solusi 2 juga **tidak melalui hyperparameter tuning**: `K_FACTORS` ditetapkan pada 50
dan tidak dicari melalui pencarian grid, sehingga tidak ada *best parameters* hasil tuning untuk
dilaporkan. Nilai ini dipilih langsung dengan pertimbangan bahwa jumlah pengguna hanya
610 dan *density* matriks 3.24%, sehingga dimensi laten yang jauh lebih
besar berisiko menghafal data latih.

**Keluaran top-N.** Berikut rekomendasi untuk pengguna contoh `1`:

| Peringkat | Judul Film |
| :---: | :--- |
| 1 | Lord of the Rings: The Two Towers, The (2002) |
| 2 | Lord of the Rings: The Fellowship of the Ring, The (2001) |
| 3 | Big (1988) |
| 4 | Christmas Story, A (1983) |
| 5 | Austin Powers: International Man of Mystery (1997) |
| 6 | Lord of the Rings: The Return of the King, The (2003) |
| 7 | Godfather: Part II, The (1974) |
| 8 | Snatch (2000) |
| 9 | Godfather, The (1972) |
| 10 | Blade Runner (1982) |

Kedua daftar beririsan pada **0 dari 10 film**, yang menunjukkan bahwa kedua
pendekatan menangkap sinyal yang berbeda: *content-based* terikat pada kemiripan genre, sedangkan
*collaborative filtering* mengikuti pola rating kolektif yang dapat melintasi batas genre.
### Kelebihan dan Kekurangan Pendekatan

**Content-Based Filtering**

| Kelebihan | Kekurangan |
| :--- | :--- |
| Tidak bergantung pada rating pengguna lain, sehingga tetap bekerja saat data rating langka. | Hanya mampu merekomendasikan film yang mirip dengan yang sudah pernah disukai — *serendipity* rendah. |
| Mampu menangani *item cold-start*: film baru tanpa rating sekalipun dapat direkomendasikan. | Kualitas rekomendasi dibatasi oleh kekayaan fitur konten; di sini hanya genre yang tersedia. |
| Alasan rekomendasi dapat dijelaskan (*explainable*) karena berbasis atribut yang terbaca manusia. | Cenderung terjebak dalam gelembung genre yang sama dan sulit menangkap selera lintas genre. |

**Collaborative Filtering (SVD)**

| Kelebihan | Kekurangan |
| :--- | :--- |
| Mampu menangkap preferensi implisit yang tidak tercermin pada atribut film. | Tidak dapat merekomendasikan film baru yang belum memiliki rating (*item cold-start*). |
| Representasi laten berdimensi rendah sehingga efisien secara komputasi dan memori. | Bergantung pada kuantitas rating; pengguna dengan riwayat tipis menghasilkan faktor laten yang tidak stabil. |
| Umumnya menghasilkan akurasi prediksi rating terbaik pada data eksplisit berskala besar. | Hasil sulit dijelaskan karena faktor laten tidak memiliki makna langsung. |

---

## Evaluation

### Metrik Evaluasi

Tiga metrik digunakan. Dua metrik pertama mengukur akurasi prediksi rating, metrik ketiga
mengukur kualitas daftar rekomendasi — sesuai dengan keluaran top-N yang menjadi tujuan proyek.

#### 1. Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{(u,i) \in D_{test}} (r_{ui} - \hat{r}_{ui})^2}$$

**Cara kerja:** selisih antara rating sebenarnya dan rating prediksi dikuadratkan, dirata-ratakan,
lalu diakarkan. Pengkuadratan membuat kesalahan besar dihukum jauh lebih berat daripada kesalahan
kecil, sehingga RMSE peka terhadap prediksi yang meleset jauh. Nilainya berada pada skala yang
sama dengan rating (0.5–5.0), dan semakin kecil semakin baik.

#### 2. Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{N} \sum_{(u,i) \in D_{test}} \lvert r_{ui} - \hat{r}_{ui} \rvert$$

**Cara kerja:** rata-rata selisih absolut antara rating sebenarnya dan prediksi. Berbeda dari RMSE,
setiap kesalahan diberi bobot proporsional sehingga MAE lebih tahan terhadap pencilan. Selisih MAE
dan RMSE yang lebar menandakan adanya sejumlah kecil prediksi yang meleset jauh.

#### 3. Precision@10

$$\text{Precision@}K = \frac{\lvert \{i \in \text{Top-}K(u) : r_{ui} \geq 4.0\} \rvert}{K}$$

**Cara kerja:** untuk setiap pengguna di data uji, sistem diminta merekomendasikan 10 film yang
belum pernah ia tonton. Sebuah rekomendasi dihitung *relevan* bila pengguna terbukti memberi rating
$\geq 4.0$ pada film tersebut di data uji. Precision@10 adalah proporsi rekomendasi relevan,
dirata-ratakan pada seluruh pengguna yang memiliki minimal satu film relevan. Metrik ini mengukur
hal yang sesungguhnya ingin dicapai proyek — kualitas daftar rekomendasi — bukan sekadar kedekatan
angka prediksi.

### Hasil Evaluasi

Sebagai titik acuan, dua *baseline* naif dihitung. *Baseline* pertama selalu memprediksi
rata-rata global (3.5391) dan menghasilkan **RMSE 1.0275** serta
**MAE 0.8146**. *Baseline* kedua selalu merekomendasikan film terpopuler yang belum
ditonton pengguna dan menghasilkan **Precision@10 0.1202**.

| Model | RMSE | MAE | Precision@10 |
| :--- | :---: | :---: | :---: |
| **Content-Based Filtering** | 0.8502 | 0.6458 | 0.0289 |
| **Collaborative Filtering (SVD)** | 0.9173 | 0.7062 | 0.1183 |
| *Baseline popularitas* | — | — | 0.1202 |

Evaluasi dilakukan pada **595 pengguna** di data uji yang memiliki minimal satu film relevan.

Pemenang tiap metrik:

| Metrik | Model Unggul | Nilai |
| :--- | :--- | ---: |
| RMSE | Content-Based Filtering | 0.8502 |
| MAE | Content-Based Filtering | 0.6458 |
| Precision@10 | Collaborative Filtering (SVD) | 0.1183 |

### Interpretasi Hasil

**Kedua model menang pada metrik yang berbeda, dan perbedaan ini justru merupakan temuan yang
paling penting dari proyek ini.**

**Akurasi prediksi rating — dimenangkan oleh Content-Based Filtering.** Nilai RMSE 0.8502
berbanding 0.9173, dengan MAE 0.6458 berbanding
0.7062. Kedua model mengalahkan *baseline* rata-rata global
(RMSE 1.0275), sehingga keduanya terbukti mempelajari sinyal personalisasi dan bukan
sekadar menebak nilai tengah. Namun selisih antar keduanya tipis — sekitar
**7.3%** — yang menunjukkan bahwa untuk tugas
*menebak angka rating*, informasi genre saja hampir sepadan dengan seluruh pola rating kolektif.

**Kualitas daftar rekomendasi — dimenangkan oleh Collaborative Filtering (SVD), dan dengan selisih yang jauh
lebih besar.** Precision@10 0.1183 berbanding
0.0289 — sekitar **4 kali lipat**. Artinya dari setiap 10 film
yang direkomendasikan Collaborative Filtering (SVD), sekitar **1.2 film** terbukti relevan,
dibandingkan hanya sekitar **0.3 film** pada Content-Based Filtering.

**Mengapa hasilnya terbelah seperti ini?** Kedua metrik mengukur hal yang fundamentally berbeda.
RMSE menilai seberapa dekat *angka* prediksi dengan rating sebenarnya — dan menebak nilai
mendekati rata-rata sudah cukup untuk mendapat RMSE yang wajar. Precision@10 menilai apakah film
yang *diurutkan di posisi teratas* benar-benar disukai — dan untuk itu, menebak nilai tengah tidak
menghasilkan apa pun.

*Content-based filtering* dalam proyek ini hanya memiliki 20 dimensi informasi tentang setiap
film, yaitu genre-nya. Dua film yang sangat berbeda tetap dianggap identik bila genre-nya sama,
sehingga skor kemiripan menumpuk pada satu nilai yang sama untuk banyak film sekaligus. Akibatnya
urutan top-N ditentukan oleh pemecah seri, bukan oleh preferensi — dan daftar yang dihasilkan
nyaris tidak mengandung film yang benar-benar disukai pengguna. *Collaborative filtering*
sebaliknya belajar dari puluhan ribu keputusan rating nyata dan dapat menemukan pola yang tidak
terlihat pada atribut film, misalnya kecenderungan seorang pengguna menyukai film dari sutradara
atau era tertentu.

**Catatan penting: kedua model kalah dari baseline popularitas pada Precision@10.**
*Baseline* yang hanya merekomendasikan film terpopuler memperoleh Precision@10
0.1202, lebih tinggi daripada Collaborative Filtering (0.1183)
maupun Content-Based Filtering (0.0289). Ini adalah temuan yang lazim pada
dataset MovieLens dan bukan tanda bahwa modelnya rusak: karena rating condong positif, film
populer memiliki peluang jauh lebih besar untuk kebetulan memperoleh rating tinggi di data uji.
Model yang benar-benar personal harus mengalahkan angka ini agar layak diterapkan, dan pada
konfigurasi saat ini belum ada yang berhasil. Peningkatan yang paling menjanjikan adalah
menambahkan fitur konten di luar genre (tahun rilis, sutradara, aktor, sinopsis), memakai
*factorization machines* yang menggabungkan sinyal konten dan kolaboratif, serta menyetel ulang
`k` faktor laten dan ambang relevansi.

**Catatan mengenai selisih MAE dan RMSE.** Pada kedua model, RMSE hanya sedikit lebih tinggi
daripada MAE. Selisih yang sempit ini menandakan distribusi kesalahan cukup merata dan tidak
didominasi oleh segelintir prediksi ekstrem — indikasi bahwa kedua model terkalibrasi dengan wajar.

**Keterbatasan.** Evaluasi ini memakai protokol *random split* pada tingkat rating, yang
cenderung lebih optimistis dibandingkan protokol *leave-one-out* atau pemisahan berdasarkan waktu,
karena rating yang diberikan belakangan bisa "membocorkan" preferensi masa depan ke data latih.
Kedua model juga belum menangani *user cold-start*: pengguna yang sama sekali belum memiliki
riwayat rating tidak dapat dilayani oleh pendekatan mana pun dalam proyek ini.

---

## Kesimpulan

1. **Pernyataan Masalah 1 terjawab.** Model *content-based filtering* berhasil dibangun dengan
   merepresentasikan film sebagai vektor TF-IDF genre, membentuk profil pengguna berbobot, dan
   mengukur kemiripan melalui *cosine similarity*. Model ini mampu menghasilkan top-N rekomendasi
   personal dan — karena hanya bergantung pada atribut film — secara prinsip dapat merekomendasikan
   judul yang belum memiliki rating sama sekali. Namun hasil evaluasi menunjukkan bahwa dengan
   hanya 20 genre sebagai fitur, daya pisah model ini terbatas: skor kemiripan menumpuk pada
   nilai yang sama untuk banyak film, sehingga urutan top-N menjadi kurang bermakna
   (Precision@10 hanya 0.0289).
2. **Pernyataan Masalah 2 terjawab.** Model *collaborative filtering* berbasis SVD dengan
   50 faktor laten berhasil mengestimasi rating pada pasangan pengguna-film yang belum
   teramati, dengan RMSE 0.9173 dan MAE 0.7062. Kualitas daftar
   rekomendasinya jauh lebih baik daripada *content-based filtering*
   (Precision@10 0.1183 berbanding 0.0289).
3. **Pernyataan Masalah 3 terjawab.** Tidak ada satu model yang unggul di semua metrik, dan itulah
   jawaban yang jujur: **Content-Based Filtering** lebih baik dalam memprediksi *angka* rating, sementara
   **Collaborative Filtering (SVD)** jauh lebih baik dalam menyusun *daftar* rekomendasi. Karena tujuan proyek ini
   adalah menyajikan top-N recommendation, **Collaborative Filtering (SVD)** adalah pilihan yang lebih tepat untuk
   diterapkan. Akan tetapi, keduanya masih kalah dari *baseline* popularitas pada Precision@10
   (0.1202), sehingga belum ada model dalam proyek ini yang layak langsung
   diterapkan pada sistem produksi tanpa perbaikan lebih lanjut.

**Saran pengembangan.** Menambahkan fitur konten di luar genre (tahun rilis, sutradara, aktor,
sinopsis) akan memberi *content-based filtering* dimensi informasi yang cukup untuk membedakan
film dalam genre yang sama. Menggabungkan kedua pendekatan dalam model *hybrid* — misalnya
*factorization machines* yang memakai fitur konten sebagai masukan tambahan — berpotensi
mengatasi kelemahan masing-masing sekaligus menaikkan Precision@10 di atas *baseline* popularitas.

**---Ini adalah bagian akhir laporan---**

---

_Catatan:_ Gambar dan visualisasi lengkap beserta output eksekusinya tersedia pada
`notebook.ipynb`. Seluruh angka pada laporan ini dihasilkan langsung dari eksekusi notebook tersebut.
