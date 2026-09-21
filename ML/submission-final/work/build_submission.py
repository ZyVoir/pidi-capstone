#!/usr/bin/env python3
"""
Build ML Final Submission (Proyek Akhir: Membuat Model Sistem Rekomendasi).

Single source of truth for the deliverable cells -> emits notebook.ipynb,
notebook.py, and the Markdown report, executes the notebook, then zips the
3 files the spec requires.

Usage:  python3 build_submission.py
"""
import json
import os
import shutil
import sys
import zipfile as zf
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
SUB = HERE.parent / "submission"
NAME = "William"
TITLE = "Laporan Proyek Machine Learning"
REPORT = f"{TITLE} - {NAME}"

SUB.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Cells: (kind, source)
# --------------------------------------------------------------------------
CELLS = [
("md", f"""# {REPORT}

**Proyek Akhir: Membuat Model Sistem Rekomendasi — Rekomendasi Film**

Notebook ini mendokumentasikan alur proyek machine learning secara utuh, mulai dari
*project overview*, *business understanding*, *data understanding*, *data preparation*,
*modeling and result*, hingga *evaluation*. Setiap tahapan dijelaskan melalui text cell dan
diimplementasikan pada code cell di bawahnya, dengan urutan yang sama seperti pada laporan
Markdown yang menyertai submission ini.

Dua pendekatan sistem rekomendasi dibangun dan dibandingkan:

1. **Content-Based Filtering** — merekomendasikan film berdasarkan kemiripan konten (genre).
2. **Collaborative Filtering** — merekomendasikan film berdasarkan pola rating antar pengguna
   melalui *matrix factorization* (SVD).
"""),

("md", """## Domain Proyek

### Latar Belakang

Industri layanan streaming film menghadapi persoalan yang oleh para peneliti disebut sebagai
*long tail*: katalog yang tersedia sangat besar, tetapi perhatian pengguna terkonsentrasi pada
segelintir judul populer. Pada dataset yang digunakan dalam proyek ini, 10% film terpopuler
menyerap mayoritas dari seluruh rating yang tercatat. Akibatnya, pengguna dihadapkan pada
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
yang sesuai. Bagi penyedia layanan, rekomendasi yang baik meningkatkan jumlah film yang
ditonton per sesi, memperpanjang masa berlangganan, dan membuka eksposur bagi judul-judul
non-populer yang sebelumnya tidak terlihat.

Pendekatannya adalah membangun dua model rekomendasi dengan asumsi kerja yang berbeda.
*Content-based filtering* memanfaatkan atribut film (genre) sehingga mampu merekomendasikan
judul baru yang belum memiliki rating sama sekali — mengatasi *item cold-start*. *Collaborative
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
"""),

("md", """## Business Understanding

### Problem Statements

- **Pernyataan Masalah 1:** Bagaimana memanfaatkan atribut konten film (genre) untuk
  merekomendasikan film yang relevan bagi seorang pengguna, termasuk untuk film yang belum
  pernah diberi rating oleh siapa pun?
- **Pernyataan Masalah 2:** Bagaimana memanfaatkan pola rating historis seluruh pengguna untuk
  memprediksi rating yang akan diberikan seorang pengguna pada film yang belum ia tonton?
- **Pernyataan Masalah 3:** Pendekatan mana — *content-based filtering* atau *collaborative
  filtering* — yang memberikan kinerja lebih baik pada dataset ini, dan atas dasar apa pilihan
  tersebut diambil?

### Goals

- **Jawaban Pernyataan Masalah 1:** Membangun model *content-based filtering* yang merepresentasikan
  film sebagai vektor fitur genre dan mengukur kemiripan antar film, sehingga mampu menghasilkan
  daftar top-N rekomendasi personal untuk setiap pengguna.
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
2. **Collaborative Filtering (Matrix Factorization / SVD).** Matriks rating yang telah dinormalisasi
   terhadap rata-rata pengguna difaktorkan menjadi matriks laten pengguna dan film berukuran
   `k = 50` menggunakan *truncated SVD*. Rating diprediksi dari hasil kali faktor laten kedua
   pihak. Diukur dengan **RMSE**, **MAE**, dan **Precision@10**.

Kedua solusi menghasilkan *top-N recommendation* sebagai keluaran akhir dan dievaluasi pada
himpunan data uji yang identik, sehingga perbandingannya bersifat *apples-to-apples*.
"""),

("md", """## Data Understanding

Dataset yang digunakan adalah **MovieLens Latest Small**, dikelola oleh GroupLens Research,
University of Minnesota. Dataset ini berisi rating film yang diberikan oleh pengguna nyata dan
merupakan salah satu *benchmark* paling umum dalam penelitian sistem rekomendasi
(Harper & Konstan, 2015).

**Tautan unduh:** https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

Dataset terdiri atas dua berkas yang digunakan dalam proyek ini:

- `ratings.csv` — pasangan pengguna-film beserta nilai rating.
- `movies.csv` — metadata film: judul dan genre.

Sel berikut memuat dataset dan memeriksa kondisi awalnya.
"""),

("code", """%matplotlib inline

import json
import os
import shutil
import ssl
import urllib.request
import warnings
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

DATA_DIR = "ml-latest-small"
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Unduh otomatis bila berkas belum tersedia, agar notebook tetap dapat dijalankan ulang
# oleh reviewer tanpa langkah manual.
if not os.path.exists(os.path.join(DATA_DIR, "ratings.csv")):
    try:  # beberapa instalasi Python tidak punya CA bawaan; certifi menyediakannya
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(DATA_URL, context=ctx) as resp, open("ml-latest-small.zip", "wb") as fp:
        shutil.copyfileobj(resp, fp)
    with zipfile.ZipFile("ml-latest-small.zip") as z:
        z.extractall(".")

ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))

print("Ukuran ratings :", ratings.shape)
print("Ukuran movies  :", movies.shape)
ratings.head()"""),

("md", """### Variabel-variabel pada MovieLens Latest Small dataset adalah sebagai berikut:

**`ratings.csv`**

- `userId` : identitas unik pengguna yang memberikan rating. Bertipe numerik diskret dan
  digunakan sebagai salah satu sumbu matriks rating.
- `movieId` : identitas unik film yang diberi rating. Menjadi kunci penghubung ke `movies.csv`.
- `rating` : nilai rating yang diberikan pengguna, berskala 0.5–5.0 dengan kelipatan 0.5.
  Inilah variabel target yang diprediksi oleh model *collaborative filtering*.
- `timestamp` : waktu pemberian rating dalam format *Unix epoch*. Tidak digunakan dalam
  pemodelan karena proyek ini tidak menangani aspek temporal.

**`movies.csv`**

- `movieId` : identitas unik film, kunci penghubung ke `ratings.csv`.
- `title` : judul film beserta tahun rilis. Digunakan hanya untuk menampilkan hasil rekomendasi
  agar dapat dibaca manusia.
- `genres` : daftar genre film yang dipisahkan karakter `|`, misalnya
  `Adventure|Animation|Children|Comedy|Fantasy`. Variabel inilah yang menjadi fitur konten pada
  model *content-based filtering*.
"""),

("code", """print("=== Informasi ratings ===")
ratings.info()
print("\\n=== Statistik deskriptif ratings ===")
print(ratings.describe().T.to_string())
print("\\n=== Nilai kosong per kolom ===")
print(ratings.isnull().sum().to_string())
print("\\n=== Nilai kosong pada movies ===")
print(movies.isnull().sum().to_string())
print("\\nBaris duplikat (userId, movieId) :", ratings.duplicated(["userId", "movieId"]).sum())
print("Jumlah pengguna unik             :", ratings.userId.nunique())
print("Jumlah film unik yang dirating   :", ratings.movieId.nunique())
print("Jumlah film di katalog           :", movies.movieId.nunique())
print("Rentang waktu rating             :",
      pd.to_datetime(ratings.timestamp, unit="s").min().date(), "-",
      pd.to_datetime(ratings.timestamp, unit="s").max().date())"""),

("md", """### Exploratory Data Analysis

Tahap EDA dilakukan untuk memahami tiga hal yang menentukan rancangan model: sebaran nilai
rating, seberapa aktif setiap pengguna memberi rating, dan seberapa timpang distribusi rating
antar film.
"""),

("code", """fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

sns.countplot(x="rating", data=ratings, ax=axes[0], color="#4C72B0")
axes[0].set_title("Distribusi Nilai Rating")
axes[0].set_xlabel("Rating")
axes[0].set_ylabel("Jumlah rating")

per_user = ratings.groupby("userId").size()
sns.histplot(per_user, bins=40, ax=axes[1], color="#55A868")
axes[1].set_title("Jumlah Rating per Pengguna")
axes[1].set_xlabel("Jumlah rating yang diberikan")
axes[1].set_ylabel("Jumlah pengguna")

per_movie = ratings.groupby("movieId").size().sort_values(ascending=False).values
axes[2].plot(np.arange(1, len(per_movie) + 1), per_movie, color="#C44E52")
axes[2].set_yscale("log")
axes[2].set_title("Rating per Film (long tail)")
axes[2].set_xlabel("Peringkat film menurut popularitas")
axes[2].set_ylabel("Jumlah rating (skala log)")

plt.tight_layout()
plt.show()

top10_share = per_movie[: int(len(per_movie) * 0.1)].sum() / per_movie.sum() * 100
print(f"Rata-rata rating per pengguna : {per_user.mean():.1f}")
print(f"Median rating per film        : {np.median(per_movie):.0f}")
print(f"10% film terpopuler menyerap  : {top10_share:.1f}% dari seluruh rating")
print(f"Rating >= 4.0                 : {(ratings.rating >= 4).mean() * 100:.1f}% dari seluruh rating")"""),

("md", """#### Distribusi genre

*Insight yang dicari:* genre mana yang paling banyak tersedia di katalog, dan genre mana yang
rata-rata ratingnya tertinggi. Kedua hal ini menentukan seberapa informatif fitur genre bagi
model *content-based filtering*.
"""),

("code", """genres_exploded = movies.assign(genre=movies.genres.str.split("|")).explode("genre")
genre_counts = genres_exploded.genre.value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=axes[0], color="#4C72B0")
axes[0].set_title("Jumlah Film per Genre")
axes[0].set_xlabel("Jumlah film")
axes[0].set_ylabel("")

merged_eda = ratings.merge(movies, on="movieId")
merged_genre = merged_eda.assign(genre=merged_eda.genres.str.split("|")).explode("genre")
genre_stat = merged_genre.groupby("genre").rating.agg(["mean", "count"]).sort_values("mean")

sns.barplot(x=genre_stat["mean"], y=genre_stat.index, ax=axes[1], color="#55A868")
axes[1].set_title("Rata-rata Rating per Genre")
axes[1].set_xlabel("Rata-rata rating")
axes[1].set_ylabel("")
axes[1].set_xlim(3.0, 4.3)

plt.tight_layout()
plt.show()
print(genre_stat.sort_values("mean", ascending=False).round(3).to_string())"""),

("md", """### Insight dari EDA

1. **Rating condong ke positif.** Sebaran rating memuncak pada nilai 4.0 dan mayoritas rating
   bernilai 3.0 ke atas. Ini adalah *positivity bias* yang lazim pada data eksplisit: pengguna
   lebih sering menonton — dan memberi rating — pada film yang memang mereka perkirakan disukai.
   Konsekuensinya, model yang selalu memprediksi nilai mendekati rata-rata akan tampak cukup baik
   bila hanya diukur dengan RMSE, sehingga RMSE perlu didampingi metrik peringkat (Precision@10).
2. **Distribusi rating sangat timpang.** Sebagian kecil film menyerap porsi rating yang jauh
   melebihi proporsinya, sementara ekor distribusinya panjang. Model *collaborative filtering*
   akan kesulitan pada film di ekor distribusi karena bukti yang tersedia terlalu sedikit —
   inilah *item cold-start* yang menjadi keunggulan *content-based filtering*.
3. **Jumlah rating per pengguna bervariasi lebar.** Ada pengguna yang hanya memberi belasan
   rating dan ada yang memberi ribuan. Pengguna dengan riwayat tipis menghasilkan profil yang
   kurang andal, sehingga penyaringan minimum rating diperlukan pada tahap *data preparation*.
4. **Genre tidak seimbang.** Drama dan Komedi mendominasi katalog, sedangkan genre seperti
   *Film-Noir* dan *Documentary* jauh lebih sedikit. Rata-rata rating antar genre pun berbeda,
   yang mengonfirmasi bahwa genre membawa sinyal preferensi yang layak dijadikan fitur konten.
"""),

("md", """## Data Preparation

Tahapan *data preparation* dilakukan dalam enam langkah berikut. Urutan ini identik dengan yang
dijelaskan pada laporan Markdown. Setiap langkah diberi penjelasan singkat mengenai *mengapa*
tahapan tersebut diperlukan.
"""),

("md", """#### Persiapan 1 — Menggabungkan rating dengan metadata film

*Mengapa:* model *content-based filtering* membutuhkan kolom `genres`, yang hanya tersedia di
`movies.csv`. Penggabungan dilakukan dengan *left join* agar seluruh baris rating tetap terjaga.
"""),

("code", """# Persiapan 1 - Menggabungkan rating dengan metadata film
df = ratings.merge(movies, on="movieId", how="left")"""),

("md", """#### Persiapan 2 — Menghapus duplikat dan menangani nilai kosong

*Mengapa:* satu pasangan pengguna-film seharusnya hanya memiliki satu rating. Duplikat akan
memberi bobot ganda pada satu interaksi sehingga matriks rating menjadi bias. Baris tanpa `rating`
atau `genres` juga dibuang karena tidak dapat dipakai oleh model mana pun.
"""),

("code", """# Persiapan 2 - Menghapus duplikat dan menangani nilai kosong
before = len(df)
df = df.drop_duplicates(subset=["userId", "movieId"]).dropna(subset=["rating", "genres"])
print(f"Baris sebelum pembersihan : {before:,}")
print(f"Baris sesudah pembersihan : {len(df):,}")
print(f"Nilai kosong tersisa      : {int(df.isnull().sum().sum())}")"""),

("md", """#### Persiapan 3 — Menyaring pengguna dan film dengan rating minimum

*Mengapa:* pengguna dengan riwayat sangat tipis dan film dengan sedikit rating menghasilkan vektor
laten yang tidak stabil pada *collaborative filtering*, sekaligus membuat evaluasi top-N tidak
bermakna. Ambang dipilih `>= 5` rating dan diterapkan secara iteratif hingga himpunan stabil,
karena penyaringan film dapat menurunkan jumlah rating pengguna dan sebaliknya.
"""),

("code", """# Persiapan 3 - Menyaring pengguna dan film dengan rating minimum
MIN_RATING = 5
while True:
    u_count = df.groupby("userId").size()
    m_count = df.groupby("movieId").size()
    keep = df.userId.isin(u_count[u_count >= MIN_RATING].index) & df.movieId.isin(
        m_count[m_count >= MIN_RATING].index
    )
    if keep.all():
        break
    df = df[keep]

print(f"Setelah penyaringan (min {MIN_RATING} rating):")
print(f"  Pengguna : {df.userId.nunique():,}")
print(f"  Film     : {df.movieId.nunique():,}")
print(f"  Rating   : {len(df):,}")
print(f"  Rating hilang: {before - len(df):,} ({(before - len(df)) / before * 100:.2f}%)")"""),

("md", """#### Persiapan 4 — Membagi data latih dan data uji (80:20)

*Mengapa:* evaluasi harus dilakukan pada data yang tidak dilihat model. Pembagian acak pada tingkat
rating adalah prosedur standar untuk mengukur akurasi prediksi rating. Parameter yang dipakai:
`test_size=0.2` dan `random_state=RANDOM_STATE`.
"""),

("code", """from sklearn.model_selection import train_test_split

# Persiapan 4 - Membagi data latih dan data uji (80:20)
# Parameter: test_size=0.2 (80% latih / 20% uji) dan random_state=RANDOM_STATE (42)
# agar pembagian dapat direproduksi.
train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
print(f"Data latih : {len(train):,} rating")
print(f"Data uji   : {len(test):,} rating")"""),

("md", """#### Persiapan 5 — Membentuk matriks rating pengguna-film dari data latih

*Mengapa:* baik *content-based* maupun *collaborative filtering* bekerja di atas representasi
matriks. Matriks dibangun **hanya dari data latih** agar tidak terjadi *data leakage*; sel yang
tidak teramati bernilai 0 dan ditandai terpisah melalui matriks `observed`.
"""),

("code", """# Persiapan 5 - Membentuk matriks rating pengguna-film dari data latih
user_ids = np.sort(df.userId.unique())
movie_ids = np.sort(df.movieId.unique())
user_pos = {u: i for i, u in enumerate(user_ids)}
movie_pos = {m: i for i, m in enumerate(movie_ids)}
n_users, n_items = len(user_ids), len(movie_ids)

R = np.zeros((n_users, n_items))
for u, m, r in train[["userId", "movieId", "rating"]].itertuples(index=False):
    R[user_pos[u], movie_pos[m]] = r
observed = R > 0
global_mean = float(R[observed].mean())

print(f"Dimensi matriks rating : {R.shape[0]:,} pengguna x {R.shape[1]:,} film")
print(f"Sel terisi             : {int(observed.sum()):,} ({observed.mean() * 100:.2f}% density)")
print(f"Rata-rata rating global: {global_mean:.4f}")"""),

("md", """#### Persiapan 6 — Mengekstraksi fitur konten dengan TF-IDF

*Mengapa:* genre adalah data kategorikal multi-nilai. TF-IDF mengubahnya menjadi vektor numerik
yang bobotnya menurunkan pengaruh genre yang sangat umum (misalnya Drama) dan menaikkan pengaruh
genre yang lebih spesifik, sehingga kemiripan antar film menjadi lebih informatif dibandingkan
*one-hot encoding* biasa.
"""),

("code", """from sklearn.feature_extraction.text import TfidfVectorizer

# Persiapan 6 - Mengekstraksi fitur konten dengan TF-IDF
# Parameter: tokenizer kustom (pecah genres pada "|"), token_pattern=None; parameter lain
# (norm="l2", use_idf=True, sublinear_tf=False, min_df=1, max_df=1.0, stop_words=None) default.
catalog = (
    movies.drop_duplicates("movieId").set_index("movieId").loc[movie_ids]
)
tfidf = TfidfVectorizer(tokenizer=lambda s: s.split("|"), token_pattern=None)
item_content = tfidf.fit_transform(catalog.genres)

print("Matriks fitur konten :", item_content.shape)
print("Jumlah genre unik    :", len(tfidf.get_feature_names_out()))
print("Contoh genre         :", ", ".join(tfidf.get_feature_names_out()[:10]))"""),

("md", """## Modeling

### Solusi 1 — Content-Based Filtering

Pendekatan ini bertumpu pada asumsi bahwa **film yang mirip secara konten akan disukai oleh
pengguna yang sama**. Setiap film direpresentasikan sebagai vektor TF-IDF genre, lalu kemiripan
antar film dihitung dengan *cosine similarity*:

$$\\text{sim}(i, j) = \\frac{\\mathbf{v}_i \\cdot \\mathbf{v}_j}{\\lVert \\mathbf{v}_i \\rVert \\, \\lVert \\mathbf{v}_j \\rVert}$$

Model ini memakai dua rumus untuk dua tugas yang berbeda, sebagaimana lazim pada sistem
rekomendasi berbasis konten.

**(a) Prediksi rating** — rata-rata berbobot selisih rating terhadap rata-rata film, diambil dari
30 film yang paling mirip secara konten:

$$\\hat{r}_{ui} = \\mu_i + \\frac{\\sum_{j \\in N_k(i)} \\text{sim}(i, j) \\cdot (r_{uj} - \\mu_j)}{\\sum_{j \\in N_k(i)} \\text{sim}(i, j)}$$

Selisih terhadap rata-rata film ($r_{uj} - \\mu_j$) dipakai, bukan rating mentah, agar film yang
memang universally disukai tidak otomatis dianggap sangat relevan hanya karena rata-ratanya
tinggi. $\\mu_i$ adalah rata-rata rating film $i$ pada data latih.

**(b) Skor peringkat top-N** — dibentuk *profil pengguna* berupa vektor genre berbobot, yaitu
jumlah dari vektor TF-IDF film yang pernah dirating, ditimbang dengan selisih rating pengguna
terhadap rata-ratanya sendiri:

$$\\mathbf{p}_u = \\sum_{j \\in I_u} (r_{uj} - \\mu_u) \\, \\mathbf{v}_j, \\qquad
\\text{skor}(u, i) = \\frac{\\mathbf{p}_u \\cdot \\mathbf{v}_i}{\\lVert \\mathbf{p}_u \\rVert \\, \\lVert \\mathbf{v}_i \\rVert}$$

Skor ini **tidak** dinormalisasi terhadap jumlah kemiripan. Pembagian semacam itu membuat seluruh
film dengan genre identik memperoleh skor yang persis sama, sehingga urutan top-N menjadi
sewenang-wenang — masalah yang terkonfirmasi saat pengembangan dan sengaja dihindari di sini.

"""),

("code", """from scipy import sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Model 1 - Content-Based Filtering
# Parameter: K_NEIGHBORS = 30 tetangga terdekat; top-N n = 10; prediksi dijepit ke [0.5, 5.0].
# Tidak ada hyperparameter tuning - TF-IDF dan matriks kemiripan dihitung langsung dari data.
item_sim = cosine_similarity(item_content, dense_output=False)
item_sim = sp.triu(item_sim, k=1)           # buang diagonal: film tidak dianggap mirip dirinya sendiri
item_sim = (item_sim + item_sim.T).tocsr()  # kembalikan simetri

K_NEIGHBORS = 30

# Rata-rata rating tiap film dan tiap pengguna, dihitung dari data latih saja.
item_count = observed.sum(axis=0)
item_mean = np.divide(
    np.where(observed, R, 0).sum(axis=0), item_count,
    out=np.full(n_items, global_mean), where=item_count > 0,
)
user_count = observed.sum(axis=1)
user_mean = np.divide(
    R.sum(axis=1), user_count, out=np.full(n_users, global_mean), where=user_count > 0
)


def cbf_predict(u_idx, i_idx):
    \"\"\"Prediksi rating = rata-rata film + rata-rata berbobot selisih rating tetangga terdekat.\"\"\"
    rated = np.where(observed[u_idx])[0]
    if rated.size == 0:
        return float(np.clip(item_mean[i_idx], 0.5, 5.0))
    sims = np.asarray(item_sim[i_idx].todense()).ravel()[rated]
    top = np.argsort(sims)[-K_NEIGHBORS:]
    w = sims[top]
    if w.sum() <= 0:
        return float(np.clip(item_mean[i_idx], 0.5, 5.0))
    dev = R[u_idx, rated[top]] - item_mean[rated[top]]
    return float(np.clip(item_mean[i_idx] + np.dot(w, dev) / w.sum(), 0.5, 5.0))


# Profil pengguna: vektor genre berbobot selisih rating terhadap rata-rata pengguna.
profile = normalize(sp.csr_matrix(np.where(observed, R - user_mean[:, None], 0.0)).dot(item_content))
item_norm = normalize(item_content)
cbf_scores_all = np.asarray(profile.dot(item_norm.T).todense())


def cbf_scores(u_idx):
    \"\"\"Skor kemiripan konten antara profil pengguna dan seluruh film.\"\"\"
    return cbf_scores_all[u_idx]


def cbf_top_n(u_idx, n=10):
    scores = cbf_scores(u_idx).copy()
    scores[observed[u_idx]] = -np.inf  # keluarkan film yang sudah pernah ditonton
    # lexsort menjadikan popularitas sebagai pemecah seri agar hasilnya deterministik
    return np.lexsort((popularity, scores))[::-1][:n]


popularity = np.asarray(observed.sum(axis=0)).ravel()

sample_user = int(user_ids[0])
sample_idx = user_pos[sample_user]
rec_cbf = cbf_top_n(sample_idx)
print(f"Top-10 rekomendasi Content-Based Filtering untuk pengguna {sample_user}:")
for rank, i in enumerate(rec_cbf, 1):
    print(f"  {rank:2d}. {catalog.title.iloc[i]:<45} [{catalog.genres.iloc[i]}]")"""),

("md", """### Solusi 2 — Collaborative Filtering (Matrix Factorization / SVD)

Pendekatan ini bertumpu pada asumsi bahwa **pola rating antar pengguna mengandung informasi yang
tidak terlihat dari atribut film**. Matriks rating $R$ yang telah dinormalisasi terhadap rata-rata
tiap pengguna difaktorkan menjadi dua matriks berdimensi rendah:

$$R \\approx U \\Sigma V^{T}, \\qquad P = U\\Sigma, \\quad Q = V$$

dengan $k = 50$ faktor laten. Prediksi rating diperoleh dari hasil kali vektor laten pengguna dan
film, dikembalikan ke skala semula melalui rata-rata pengguna:

$$\\hat{r}_{ui} = \\mu_u + \\mathbf{p}_u^{T} \\mathbf{q}_i$$

Nilai prediksi dijepit (*clip*) ke rentang sah $[0.5, 5.0]$. Top-N disusun dengan menghitung
$\\hat{r}_{ui}$ untuk seluruh film, mengeluarkan film yang sudah pernah ditonton, lalu mengambil
10 skor tertinggi.
"""),

("code", """from scipy.sparse.linalg import svds

# Model 2 - Collaborative Filtering dengan matrix factorization (SVD)
# Parameter: K_FACTORS = 50 faktor laten; svds memakai argumen default lainnya
# (which="LM", tol=0, maxiter=None). Tidak ada hyperparameter tuning pada model ini.
K_FACTORS = 50
mask = observed.astype(float)
user_mean = np.divide(
    R.sum(axis=1), mask.sum(axis=1), out=np.full(n_users, global_mean), where=mask.sum(axis=1) > 0
)
centered = (R - user_mean[:, None]) * mask

U, sigma, Vt = svds(sp.csr_matrix(centered), k=K_FACTORS)
P = U * sigma
Q = Vt.T


def cf_predict(u_idx, i_idx):
    \"\"\"Prediksi rating = rata-rata pengguna + hasil kali faktor laten pengguna dan film.\"\"\"
    return float(np.clip(user_mean[u_idx] + P[u_idx].dot(Q[i_idx]), 0.5, 5.0))


def cf_top_n(u_idx, n=10):
    scores = user_mean[u_idx] + P[u_idx].dot(Q.T)
    scores[observed[u_idx]] = -np.inf  # keluarkan film yang sudah pernah ditonton
    return np.lexsort((popularity, scores))[::-1][:n]


rec_cf = cf_top_n(sample_idx)
print(f"Top-10 rekomendasi Collaborative Filtering untuk pengguna {sample_user}:")
for rank, i in enumerate(rec_cf, 1):
    print(f"  {rank:2d}. {catalog.title.iloc[i]:<45} [{catalog.genres.iloc[i]}]")

overlap = len(set(rec_cbf) & set(rec_cf))
print(f"\\nIrisan kedua daftar top-10: {overlap} film")"""),

("md", """### Kelebihan dan Kekurangan Pendekatan

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
"""),

("md", """## Evaluation

### Metrik Evaluasi

Tiga metrik digunakan. Dua metrik pertama mengukur akurasi prediksi rating, metrik ketiga
mengukur kualitas daftar rekomendasi — sesuai dengan keluaran top-N yang menjadi tujuan proyek.

**1. Root Mean Squared Error (RMSE)**

$$\\text{RMSE} = \\sqrt{\\frac{1}{N} \\sum_{(u,i) \\in D_{test}} (r_{ui} - \\hat{r}_{ui})^2}$$

Cara kerja: selisih antara rating sebenarnya dan rating prediksi dikuadratkan, dirata-ratakan,
lalu diakarkan. Pengkuadratan membuat kesalahan besar dihukum jauh lebih berat daripada kesalahan
kecil, sehingga RMSE peka terhadap prediksi yang meleset jauh. Nilainya berada pada skala yang
sama dengan rating (0.5–5.0), dan semakin kecil semakin baik.

**2. Mean Absolute Error (MAE)**

$$\\text{MAE} = \\frac{1}{N} \\sum_{(u,i) \\in D_{test}} \\lvert r_{ui} - \\hat{r}_{ui} \\rvert$$

Cara kerja: rata-rata selisih absolut antara rating sebenarnya dan prediksi. Berbeda dari RMSE,
setiap kesalahan diberi bobot proporsional sehingga MAE lebih tahan terhadap pencilan. Selisih
MAE dan RMSE yang lebar menandakan adanya sejumlah kecil prediksi yang meleset jauh.

**3. Precision@10**

$$\\text{Precision@}K = \\frac{\\lvert \\{i \\in \\text{Top-}K(u) : r_{ui} \\geq 4.0\\} \\rvert}{K}$$

Cara kerja: untuk setiap pengguna di data uji, sistem diminta merekomendasikan 10 film yang belum
pernah ia tonton. Sebuah rekomendasi dihitung *relevan* bila pengguna terbukti memberi rating
$\\geq 4.0$ pada film tersebut di data uji. Precision@10 adalah proporsi rekomendasi relevan,
dirata-ratakan pada seluruh pengguna yang memiliki minimal satu film relevan. Metrik ini mengukur
hal yang sesungguhnya ingin dicapai proyek — kualitas daftar rekomendasi — bukan sekadar
kedekatan angka prediksi.
"""),

("code", """from sklearn.metrics import mean_absolute_error

# Evaluasi akurasi prediksi rating pada data uji
test_pairs = [
    (user_pos[u], movie_pos[m], r)
    for u, m, r in test[["userId", "movieId", "rating"]].itertuples(index=False)
    if u in user_pos and m in movie_pos
]
y_true = np.array([r for _, _, r in test_pairs])
y_cbf = np.array([cbf_predict(u, i) for u, i, _ in test_pairs])
y_cf = np.array([cf_predict(u, i) for u, i, _ in test_pairs])


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# Evaluasi kualitas daftar rekomendasi
K_REC = 10
RELEVANT = 4.0


def precision_at_k(rec_fn):
    scores = []
    for u in test.userId.unique():
        if u not in user_pos:
            continue
        relevant = set(test[(test.userId == u) & (test.rating >= RELEVANT)].movieId)
        if not relevant:
            continue
        recs = {movie_ids[i] for i in rec_fn(user_pos[u], K_REC)}
        scores.append(len(recs & relevant) / K_REC)
    return float(np.mean(scores)), len(scores)


p_cbf, n_eval = precision_at_k(cbf_top_n)
p_cf, _ = precision_at_k(cf_top_n)


def popularity_top_n(u_idx, n=10):
    # Baseline acuan: rekomendasikan film terpopuler yang belum ditonton pengguna.
    scores = popularity.astype(float).copy()
    scores[observed[u_idx]] = -np.inf
    return np.lexsort((np.arange(n_items), scores))[::-1][:n]


p_pop, _ = precision_at_k(popularity_top_n)

# Baseline naif sebagai titik acuan: selalu prediksi rata-rata global
rmse_base = rmse(y_true, np.full_like(y_true, global_mean))
mae_base = float(mean_absolute_error(y_true, np.full_like(y_true, global_mean)))

metrics = {
    "Content-Based Filtering": {
        "RMSE": rmse(y_true, y_cbf),
        "MAE": float(mean_absolute_error(y_true, y_cbf)),
        "Precision@10": p_cbf,
    },
    "Collaborative Filtering (SVD)": {
        "RMSE": rmse(y_true, y_cf),
        "MAE": float(mean_absolute_error(y_true, y_cf)),
        "Precision@10": p_cf,
    },
}
best_model = min(metrics, key=lambda k: metrics[k]["RMSE"])

print(f"Baseline rata-rata global {global_mean:.4f} : RMSE = {rmse_base:.4f} | MAE = {mae_base:.4f}")
print(f"Baseline popularitas        : Precision@10 = {p_pop:.4f}\\n")
print(f"{'Model':<32}{'RMSE':>8}{'MAE':>8}{'Precision@10':>14}")
print("-" * 62)
for name, vals in metrics.items():
    print(f"{name:<32}{vals['RMSE']:>8.4f}{vals['MAE']:>8.4f}{vals['Precision@10']:>14.4f}")
print(f"{'Baseline popularitas':<32}{'-':>8}{'-':>8}{p_pop:>14.4f}")
print(f"\\nDievaluasi pada {n_eval:,} pengguna di data uji.")
print(f"Model terbaik berdasarkan RMSE: {best_model}")"""),

("md", """#### Visualisasi perbandingan model

Grafik berikut membandingkan kedua model terhadap *baseline* popularitas pada ketiga metrik,
sehingga posisi masing-masing model dapat dibaca sekaligus.
"""),

("code", """fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
names = list(metrics)
labels = ["Content-Based", "Collaborative\\n(SVD)", "Baseline\\nPopularitas"]
colors = ["#4C72B0", "#55A868", "#C44E52"]

for ax, metric in zip(axes, ["RMSE", "MAE", "Precision@10"]):
    vals = [metrics[n][metric] for n in names] + [p_pop if metric == "Precision@10" else np.nan]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_ylim(0, np.nanmax(vals) * 1.25)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom")

plt.suptitle("Perbandingan Kinerja Model Rekomendasi terhadap Baseline", fontsize=13)
plt.tight_layout()
plt.show()"""),

("md", """#### Menyimpan hasil evaluasi

Seluruh angka yang dilaporkan pada laporan Markdown dihasilkan dari eksekusi sel ini. Berkas
`results.json` ditulis agar laporan dan notebook tidak dapat saling menyimpang.
"""),

("code", """results = {
    "dataset": {
        "n_ratings_raw": int(len(ratings)),
        "n_users_raw": int(ratings.userId.nunique()),
        "n_movies_raw": int(ratings.movieId.nunique()),
        "n_ratings": int(len(df)),
        "n_users": int(n_users),
        "n_movies": int(n_items),
        "density_pct": round(float(observed.mean() * 100), 2),
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "global_mean": round(global_mean, 4),
        "min_rating_filter": MIN_RATING,
        "n_genres": int(item_content.shape[1]),
        "k_factors": K_FACTORS,
        "k_neighbors": K_NEIGHBORS,
        "n_eval_users": int(n_eval),
        "top10_share": round(float(top10_share), 1),
        "random_state": RANDOM_STATE,
    },
    "rating_dist": {f"{k:.1f}": int(v) for k, v in ratings.rating.value_counts().sort_index().items()},
    "genre_counts": {k: int(v) for k, v in genre_counts.items()},
    "genre_mean_rating": {k: round(float(v), 3) for k, v in genre_stat["mean"].items()},
    "baseline": {"RMSE": rmse_base, "MAE": mae_base, "Precision@10": p_pop},
    "metrics": metrics,
    "best_model": best_model,
    "sample_user": sample_user,
    "sample_recs_cbf": [catalog.title.iloc[i] for i in rec_cbf],
    "sample_recs_cf": [catalog.title.iloc[i] for i in rec_cf],
    "sample_overlap": int(overlap),
}

with open("results.json", "w", encoding="utf-8") as fp:
    json.dump(results, fp, indent=2, ensure_ascii=False)
print("results.json tersimpan.")"""),

("md", """### Hasil Evaluasi

Angka pada tabel di bawah dihasilkan langsung dari eksekusi notebook ini dan disalin ke laporan
Markdown secara otomatis, sehingga tidak ada kemungkinan perbedaan antara keduanya.

Ringkasnya: kedua model mengalahkan *baseline* rata-rata global pada RMSE dan MAE, sehingga
keduanya benar-benar mempelajari sinyal personalisasi. Namun keduanya **belum** mengalahkan
*baseline* popularitas pada Precision@10. Interpretasi lengkap beserta keterbatasannya diuraikan
pada laporan Markdown.

### Kesimpulan

1. **Pernyataan Masalah 1 terjawab.** Model *content-based filtering* berhasil dibangun dengan
   merepresentasikan film sebagai vektor TF-IDF genre, membentuk profil pengguna berbobot, dan
   mengukur kemiripan melalui *cosine similarity*. Model ini mampu menghasilkan top-N rekomendasi
   personal dan — karena hanya bergantung pada atribut film — secara prinsip dapat merekomendasikan
   judul yang belum memiliki rating sama sekali. Namun dengan hanya 20 genre sebagai fitur, daya
   pisahnya terbatas.
2. **Pernyataan Masalah 2 terjawab.** Model *collaborative filtering* berbasis SVD dengan 50
   faktor laten berhasil mengestimasi rating pada pasangan pengguna-film yang belum teramati, dan
   menghasilkan daftar rekomendasi yang jauh lebih relevan daripada *content-based filtering*.
3. **Pernyataan Masalah 3 terjawab.** Tidak ada satu model yang unggul di semua metrik:
   *content-based filtering* sedikit lebih baik dalam memprediksi *angka* rating, sementara
   *collaborative filtering* jauh lebih baik dalam menyusun *daftar* rekomendasi. Karena tujuan
   proyek ini adalah menyajikan top-N recommendation, *collaborative filtering* adalah pilihan
   yang lebih tepat — dengan catatan keduanya masih perlu diperbaiki sebelum layak diterapkan
   pada sistem produksi.

**---Ini adalah bagian akhir laporan---**

---

_Catatan:_ Gambar dan visualisasi lengkap beserta output eksekusinya tersedia pada
`notebook.ipynb`. Seluruh angka pada laporan ini dihasilkan langsung dari eksekusi notebook tersebut.
"""),
]

# --------------------------------------------------------------------------
# Emit notebook.ipynb
# --------------------------------------------------------------------------
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
    for kind, src in CELLS
]
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.11"},
}
nb_path = SUB / "notebook.ipynb"
nbf.write(nb, str(nb_path))
print(f"[1/5] notebook.ipynb ditulis ({len(nb.cells)} sel)")

# --------------------------------------------------------------------------
# Emit notebook.py (Colab-style script export, same cells)
# --------------------------------------------------------------------------
py_lines = [
    "#!/usr/bin/env python3",
    f'"""# {REPORT}',
    "",
    "Versi skrip Python dari notebook.ipynb (ekspor gaya Google Colab).",
    "Jalankan: python3 notebook.py",
    '"""',
    "",
]
for kind, src in CELLS:
    if kind == "md":
        py_lines.append("# " + "-" * 74)
        py_lines.extend(f"# {line}" if line.strip() else "#" for line in src.splitlines())
        py_lines.append("# " + "-" * 74)
    else:
        py_lines.append("")
        # Magic notebook tidak valid di skrip Python; ganti dengan backend non-interaktif.
        src_py = src.replace(
            "%matplotlib inline",
            'import matplotlib\nmatplotlib.use("Agg")  # backend non-interaktif untuk skrip',
        )
        py_lines.extend(src_py.splitlines())
    py_lines.append("")
(SUB / "notebook.py").write_text("\n".join(py_lines), encoding="utf-8")
print("[2/5] notebook.py ditulis")

# --------------------------------------------------------------------------
# Execute the notebook
# --------------------------------------------------------------------------
results_path = SUB / "results.json"
print("[3/5] menjalankan notebook (ini butuh beberapa menit)...", flush=True)
# Jalankan in-process dari SUB. Tidak memakai nbconvert: notebook.py di folder ini akan
# membayangi paket `notebook` yang di-import nbconvert.
# MPLBACKEND sengaja TIDAK di-set: %matplotlib inline di dalam notebook yang memilih backend,
# dan memaksa Agg lewat env var membuat gambar tidak tersimpan sebagai output.
os.environ.pop("MPLBACKEND", None)
os.chdir(SUB)
try:
    nb = nbf.read(str(nb_path), as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(SUB)}}).execute()
except Exception as exc:
    sys.exit(f"eksekusi notebook gagal: {type(exc).__name__}: {exc}")
nbf.write(nb, str(nb_path))

executed = nb
code_cells = [c for c in executed.cells if c.cell_type == "code"]
errors = [o for c in code_cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
if errors:
    sys.exit(f"notebook menghasilkan error: {errors[0].get('ename')}: {errors[0].get('evalue')}")
print(f"      {len(code_cells)} sel kode dieksekusi tanpa error")

results = json.loads(results_path.read_text(encoding="utf-8"))
print(f"      hasil terbaca: model terbaik = {results['best_model']}")
print("[4/5] menyusun laporan Markdown...")

# --------------------------------------------------------------------------
# Emit the Markdown report, numbers injected from the executed notebook
# --------------------------------------------------------------------------
d = results["dataset"]
m = results["metrics"]
base = results["baseline"]
names = list(m)
METRICS = ["RMSE", "MAE", "Precision@10"]
lower_is_better = {"RMSE": True, "MAE": True, "Precision@10": False}


def winner_rows():
    out = []
    for k in METRICS:
        vals = {n: m[n][k] for n in names}
        w = min(vals, key=vals.get) if lower_is_better[k] else max(vals, key=vals.get)
        out.append(f"| {k} | {w} | {vals[w]:.4f} |")
    return "\n".join(out)


rating_rows = "\n".join(
    f"| {k} | {v:,} | {v / d['n_ratings_raw'] * 100:.2f}% |"
    for k, v in results["rating_dist"].items()
)

genre_rows = "\n".join(
    f"| {g} | {c:,} | {results['genre_mean_rating'][g]:.3f} |"
    for g, c in sorted(results["genre_counts"].items(), key=lambda kv: -kv[1])
)

rec_rows_cbf = "\n".join(f"| {i} | {t} |" for i, t in enumerate(results["sample_recs_cbf"], 1))
rec_rows_cf = "\n".join(f"| {i} | {t} |" for i, t in enumerate(results["sample_recs_cf"], 1))

# Narasi disusun dari hasil aktual, bukan dari asumsi bahwa salah satu model menang di semua metrik.
acc_winner = names[0] if m[names[0]]["RMSE"] < m[names[1]]["RMSE"] else names[1]
acc_loser = names[1] if acc_winner == names[0] else names[0]
rank_winner = names[1] if m[names[1]]["Precision@10"] > m[names[0]]["Precision@10"] else names[0]
rank_loser = names[1] if rank_winner == names[0] else names[0]
prec_ratio = max(
    m[rank_winner]["Precision@10"] / max(m[rank_loser]["Precision@10"], 1e-12),
    max(m[rank_loser]["Precision@10"], 1e-12) / max(m[rank_winner]["Precision@10"], 1e-12),
)

report = f"""# {REPORT}

**Proyek Akhir: Membuat Model Sistem Rekomendasi — Rekomendasi Film**

---

## Domain Proyek

### Latar Belakang

Industri layanan streaming film menghadapi persoalan yang oleh para peneliti disebut sebagai
*long tail*: katalog yang tersedia sangat besar, tetapi perhatian pengguna terkonsentrasi pada
segelintir judul populer. Pada dataset yang digunakan dalam proyek ini, 10% film terpopuler
menyerap **{d['top10_share']}%** dari seluruh rating yang tercatat. Akibatnya, pengguna dihadapkan pada
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
   berukuran `k = {d['k_factors']}` menggunakan *truncated SVD*. Rating diprediksi dari hasil kali
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
| Jumlah rating | {d['n_ratings_raw']:,} | {d['n_ratings']:,} |
| Jumlah pengguna | {d['n_users_raw']:,} | {d['n_users']:,} |
| Jumlah film | {d['n_movies_raw']:,} | {d['n_movies']:,} |
| Kepadatan matriks | — | {d['density_pct']}% |

Setelah tahapan *data preparation*, matriks rating berdimensi
**{d['n_users']:,} pengguna × {d['n_movies']:,} film** dengan tingkat kepadatan hanya
**{d['density_pct']}%**. Artinya lebih dari {100 - d['density_pct']:.1f}% sel matriks tidak teramati — inilah
yang membuat personalisasi menjadi masalah yang sulit sekaligus menarik. Data dibagi menjadi
**{d['train_size']:,} rating latih** dan **{d['test_size']:,} rating uji**.

Sebaran nilai rating adalah sebagai berikut:

| Rating | Jumlah | Proporsi |
| :---: | ---: | ---: |
{rating_rows}

Rata-rata rating global adalah **{d['global_mean']}**, dan **{(results['rating_dist']['4.0'] + results['rating_dist']['5.0']) / d['n_ratings_raw'] * 100:.1f}%** rating bernilai 4.0 atau 5.0.

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
{genre_rows}

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
evaluasi top-N tidak bermakna. Ambang dipilih `>= {d['min_rating_filter']}` rating dan diterapkan secara
iteratif hingga himpunan stabil, karena penyaringan film dapat menurunkan jumlah rating pengguna
dan sebaliknya. Setelah penyaringan, tersisa **{d['n_users']:,} pengguna**, **{d['n_movies']:,} film**,
dan **{d['n_ratings']:,} rating**.

### 4. Membagi data latih dan data uji (80:20)

*Mengapa diperlukan:* evaluasi harus dilakukan pada data yang tidak dilihat model. Pembagian acak
pada tingkat rating adalah prosedur standar untuk mengukur akurasi prediksi rating. Parameter yang
dipakai: `test_size=0.2` (80% latih / 20% uji) dan `random_state=RANDOM_STATE` (`{d['random_state']}`)
agar pembagian dapat direproduksi. Dihasilkan **{d['train_size']:,} rating latih** dan
**{d['test_size']:,} rating uji**.

### 5. Membentuk matriks rating pengguna-film dari data latih

*Mengapa diperlukan:* baik *content-based* maupun *collaborative filtering* bekerja di atas
representasi matriks. Matriks dibangun **hanya dari data latih** agar tidak terjadi *data leakage*;
sel yang tidak teramati bernilai 0 dan ditandai terpisah melalui matriks `observed`.

### 6. Mengekstraksi fitur konten dengan TF-IDF

*Mengapa diperlukan:* genre adalah data kategorikal multi-nilai. TF-IDF mengubahnya menjadi vektor
numerik yang bobotnya menurunkan pengaruh genre yang sangat umum (misalnya Drama) dan menaikkan
pengaruh genre yang lebih spesifik, sehingga kemiripan antar film menjadi lebih informatif
dibandingkan *one-hot encoding* biasa. Dihasilkan matriks fitur konten berdimensi
**{d['n_movies']:,} film × {d['n_genres']} genre**.

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

$$\\text{{sim}}(i, j) = \\frac{{\\mathbf{{v}}_i \\cdot \\mathbf{{v}}_j}}{{\\lVert \\mathbf{{v}}_i \\rVert \\, \\lVert \\mathbf{{v}}_j \\rVert}}$$

Model ini memakai dua rumus untuk dua tugas yang berbeda, sebagaimana lazim pada sistem
rekomendasi berbasis konten.

**(a) Prediksi rating** — rata-rata berbobot selisih rating terhadap rata-rata film, diambil dari
{d['k_neighbors']} film yang paling mirip secara konten:

$$\\hat{{r}}_{{ui}} = \\mu_i + \\frac{{\\sum_{{j \\in N_k(i)}} \\text{{sim}}(i, j) \\cdot (r_{{uj}} - \\mu_j)}}{{\\sum_{{j \\in N_k(i)}} \\text{{sim}}(i, j)}}$$

Selisih terhadap rata-rata film ($r_{{uj}} - \\mu_j$) dipakai, bukan rating mentah, agar film yang
memang disukai semua orang tidak otomatis dianggap sangat relevan hanya karena rata-ratanya
tinggi. $\\mu_i$ adalah rata-rata rating film $i$ pada data latih.

**(b) Skor peringkat top-N** — dibentuk *profil pengguna* berupa vektor genre berbobot, yaitu
jumlah dari vektor TF-IDF film yang pernah dirating, ditimbang dengan selisih rating pengguna
terhadap rata-ratanya sendiri:

$$\\mathbf{{p}}_u = \\sum_{{j \\in I_u}} (r_{{uj}} - \\mu_u) \\, \\mathbf{{v}}_j, \\qquad
\\text{{skor}}(u, i) = \\frac{{\\mathbf{{p}}_u \\cdot \\mathbf{{v}}_i}}{{\\lVert \\mathbf{{p}}_u \\rVert \\, \\lVert \\mathbf{{v}}_i \\rVert}}$$

Skor ini **tidak** dinormalisasi terhadap jumlah kemiripan. Pembagian semacam itu membuat seluruh
film dengan genre identik memperoleh skor yang persis sama, sehingga urutan top-N menjadi
sewenang-wenang — masalah yang terkonfirmasi saat pengembangan dan sengaja dihindari di sini.
Film yang sudah pernah ditonton dikeluarkan dari daftar kandidat, dan popularitas dipakai sebagai
pemecah seri agar hasilnya deterministik.

**Parameter Solusi 1.**

| Parameter | Nilai | Keterangan |
| :--- | :--- | :--- |
| `TfidfVectorizer.tokenizer` | `lambda s: s.split("\\|")` | Memecah kolom `genres` yang multi-nilai (`Action\\|Comedy`) menjadi satu token per genre. |
| `TfidfVectorizer.token_pattern` | `None` | *Default* dimatikan agar tokenizer kustom di atas dipakai apa adanya. |
| `TfidfVectorizer.norm` | *default* (`l2`) | Normalisasi L2 pada vektor TF-IDF; tidak diubah. |
| `TfidfVectorizer.use_idf` | *default* (`True`) | IDF aktif sehingga genre umum (Drama) berbobot lebih rendah daripada genre spesifik. |
| `TfidfVectorizer.sublinear_tf` | *default* (`False`) | Tidak diubah; istilah dalam satu film tidak berulang sehingga *sublinear scaling* tidak berpengaruh. |
| `TfidfVectorizer.min_df` / `.max_df` | *default* (`1` / `1.0`) | Tidak ada pemotongan *vocabulary*. Dengan hanya {d['n_genres']} genre, membuang istilah justru menghilangkan informasi. |
| `TfidfVectorizer.stop_words` | *default* (`None`) | Tidak diubah; tidak ada genre yang berperan sebagai *stop word*. |
| `K_NEIGHBORS` | `{d['k_neighbors']}` | Jumlah tetangga terdekat pada rumus (a). Dipilih agar prediksi rating distabilkan oleh cukup banyak tetangga tanpa kehilangan sifat lokal; dengan hanya {d['n_genres']} genre, nilai yang jauh lebih kecil membuat prediksi sensitif terhadap satu film saja. |
| `k` pada `scipy.sparse.triu` | `1` | Diagonal matriks kemiripan dibuang agar sebuah film tidak dianggap mirip dengan dirinya sendiri. |
| `cbf_top_n(n)` | `10` | Panjang daftar top-N yang disajikan sebagai keluaran. |
| Pemotongan skor | `clip(0.5, 5.0)` | Seluruh prediksi rating dipotong ke rentang rating yang valid pada dataset. |

Solusi 1 **tidak melalui hyperparameter tuning** dan tidak memiliki parameter yang dilatih:
vektor TF-IDF dan matriks kemiripan dihitung langsung dari data, bukan dioptimalkan terhadap
fungsi kerugian. Karena itu tidak ada *best parameters* untuk dilaporkan pada solusi ini.

**Keluaran top-N.** Berikut rekomendasi untuk pengguna contoh `{results['sample_user']}`:

| Peringkat | Judul Film |
| :---: | :--- |
{rec_rows_cbf}

### Solusi 2 — Collaborative Filtering (Matrix Factorization / SVD)

Pendekatan ini bertumpu pada asumsi bahwa **pola rating antar pengguna mengandung informasi yang
tidak terlihat dari atribut film**. Matriks rating $R$ yang telah dinormalisasi terhadap rata-rata
tiap pengguna difaktorkan menjadi dua matriks berdimensi rendah:

$$R \\approx U \\Sigma V^{{T}}, \\qquad P = U\\Sigma, \\quad Q = V$$

dengan $k = {d['k_factors']}$ faktor laten. Prediksi rating diperoleh dari hasil kali vektor laten pengguna dan
film, dikembalikan ke skala semula melalui rata-rata pengguna:

$$\\hat{{r}}_{{ui}} = \\mu_u + \\mathbf{{p}}_u^{{T}} \\mathbf{{q}}_i$$

Nilai prediksi dijepit (*clip*) ke rentang sah $[0.5, 5.0]$. Top-N disusun dengan menghitung
$\\hat{{r}}_{{ui}}$ untuk seluruh film, mengeluarkan film yang sudah pernah ditonton, lalu mengambil
10 skor tertinggi.

**Parameter Solusi 2.**

| Parameter | Nilai | Keterangan |
| :--- | :--- | :--- |
| `K_FACTORS` (`k` pada `scipy.sparse.linalg.svds`) | `{d['k_factors']}` | Jumlah faktor laten. Menentukan kapasitas model: cukup besar untuk menampung variasi selera, cukup kecil untuk meredam *overfitting* pada matriks ber-*density* hanya {d['density_pct']}%. |
| `svds(..., k=...)` argumen lain | *default* | `which="LM"` (nilai singular terbesar), `tol=0`, `maxiter=None`, `return_singular_vectors=True` — seluruhnya dibiarkan *default*. |
| `RANDOM_STATE` | `{d['random_state']}` | Dipakai pada pembagian data latih/uji agar hasil dapat direproduksi. `svds` sendiri bersifat deterministik dan tidak menerima *seed*. |
| Normalisasi sebelum faktorisasi | `centered = R - μ_u` | Matriks dipusatkan terhadap rata-rata tiap pengguna, dan sel tak teramati dimasking menjadi 0 agar tidak ikut terfaktorisasi. |
| `cf_top_n(n)` | `10` | Panjang daftar top-N yang disajikan sebagai keluaran. |
| Pemotongan skor | `clip(0.5, 5.0)` | Seluruh prediksi rating dipotong ke rentang rating yang valid pada dataset. |

Solusi 2 juga **tidak melalui hyperparameter tuning**: `K_FACTORS` ditetapkan pada {d['k_factors']}
dan tidak dicari melalui pencarian grid, sehingga tidak ada *best parameters* hasil tuning untuk
dilaporkan. Nilai ini dipilih langsung dengan pertimbangan bahwa jumlah pengguna hanya
{d['n_users']:,} dan *density* matriks {d['density_pct']}%, sehingga dimensi laten yang jauh lebih
besar berisiko menghafal data latih.

**Keluaran top-N.** Berikut rekomendasi untuk pengguna contoh `{results['sample_user']}`:

| Peringkat | Judul Film |
| :---: | :--- |
{rec_rows_cf}

Kedua daftar beririsan pada **{results['sample_overlap']} dari 10 film**, yang menunjukkan bahwa kedua
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

$$\\text{{RMSE}} = \\sqrt{{\\frac{{1}}{{N}} \\sum_{{(u,i) \\in D_{{test}}}} (r_{{ui}} - \\hat{{r}}_{{ui}})^2}}$$

**Cara kerja:** selisih antara rating sebenarnya dan rating prediksi dikuadratkan, dirata-ratakan,
lalu diakarkan. Pengkuadratan membuat kesalahan besar dihukum jauh lebih berat daripada kesalahan
kecil, sehingga RMSE peka terhadap prediksi yang meleset jauh. Nilainya berada pada skala yang
sama dengan rating (0.5–5.0), dan semakin kecil semakin baik.

#### 2. Mean Absolute Error (MAE)

$$\\text{{MAE}} = \\frac{{1}}{{N}} \\sum_{{(u,i) \\in D_{{test}}}} \\lvert r_{{ui}} - \\hat{{r}}_{{ui}} \\rvert$$

**Cara kerja:** rata-rata selisih absolut antara rating sebenarnya dan prediksi. Berbeda dari RMSE,
setiap kesalahan diberi bobot proporsional sehingga MAE lebih tahan terhadap pencilan. Selisih MAE
dan RMSE yang lebar menandakan adanya sejumlah kecil prediksi yang meleset jauh.

#### 3. Precision@10

$$\\text{{Precision@}}K = \\frac{{\\lvert \\{{i \\in \\text{{Top-}}K(u) : r_{{ui}} \\geq 4.0\\}} \\rvert}}{{K}}$$

**Cara kerja:** untuk setiap pengguna di data uji, sistem diminta merekomendasikan 10 film yang
belum pernah ia tonton. Sebuah rekomendasi dihitung *relevan* bila pengguna terbukti memberi rating
$\\geq 4.0$ pada film tersebut di data uji. Precision@10 adalah proporsi rekomendasi relevan,
dirata-ratakan pada seluruh pengguna yang memiliki minimal satu film relevan. Metrik ini mengukur
hal yang sesungguhnya ingin dicapai proyek — kualitas daftar rekomendasi — bukan sekadar kedekatan
angka prediksi.

### Hasil Evaluasi

Sebagai titik acuan, dua *baseline* naif dihitung. *Baseline* pertama selalu memprediksi
rata-rata global ({d['global_mean']}) dan menghasilkan **RMSE {base['RMSE']:.4f}** serta
**MAE {base['MAE']:.4f}**. *Baseline* kedua selalu merekomendasikan film terpopuler yang belum
ditonton pengguna dan menghasilkan **Precision@10 {base['Precision@10']:.4f}**.

| Model | RMSE | MAE | Precision@10 |
| :--- | :---: | :---: | :---: |
| **Content-Based Filtering** | {m[names[0]]['RMSE']:.4f} | {m[names[0]]['MAE']:.4f} | {m[names[0]]['Precision@10']:.4f} |
| **Collaborative Filtering (SVD)** | {m[names[1]]['RMSE']:.4f} | {m[names[1]]['MAE']:.4f} | {m[names[1]]['Precision@10']:.4f} |
| *Baseline popularitas* | — | — | {base['Precision@10']:.4f} |

Evaluasi dilakukan pada **{d['n_eval_users']:,} pengguna** di data uji yang memiliki minimal satu film relevan.

Pemenang tiap metrik:

| Metrik | Model Unggul | Nilai |
| :--- | :--- | ---: |
{winner_rows()}

### Interpretasi Hasil

**Kedua model menang pada metrik yang berbeda, dan perbedaan ini justru merupakan temuan yang
paling penting dari proyek ini.**

**Akurasi prediksi rating — dimenangkan oleh {acc_winner}.** Nilai RMSE {m[acc_winner]['RMSE']:.4f}
berbanding {m[acc_loser]['RMSE']:.4f}, dengan MAE {m[acc_winner]['MAE']:.4f} berbanding
{m[acc_loser]['MAE']:.4f}. Kedua model mengalahkan *baseline* rata-rata global
(RMSE {base['RMSE']:.4f}), sehingga keduanya terbukti mempelajari sinyal personalisasi dan bukan
sekadar menebak nilai tengah. Namun selisih antar keduanya tipis — sekitar
**{abs(m[acc_winner]['RMSE'] - m[acc_loser]['RMSE']) / m[acc_loser]['RMSE'] * 100:.1f}%** — yang menunjukkan bahwa untuk tugas
*menebak angka rating*, informasi genre saja hampir sepadan dengan seluruh pola rating kolektif.

**Kualitas daftar rekomendasi — dimenangkan oleh {rank_winner}, dan dengan selisih yang jauh
lebih besar.** Precision@10 {m[rank_winner]['Precision@10']:.4f} berbanding
{m[rank_loser]['Precision@10']:.4f} — sekitar **{prec_ratio:.0f} kali lipat**. Artinya dari setiap 10 film
yang direkomendasikan {rank_winner}, sekitar **{m[rank_winner]['Precision@10'] * 10:.1f} film** terbukti relevan,
dibandingkan hanya sekitar **{m[rank_loser]['Precision@10'] * 10:.1f} film** pada {rank_loser}.

**Mengapa hasilnya terbelah seperti ini?** Kedua metrik mengukur hal yang fundamentally berbeda.
RMSE menilai seberapa dekat *angka* prediksi dengan rating sebenarnya — dan menebak nilai
mendekati rata-rata sudah cukup untuk mendapat RMSE yang wajar. Precision@10 menilai apakah film
yang *diurutkan di posisi teratas* benar-benar disukai — dan untuk itu, menebak nilai tengah tidak
menghasilkan apa pun.

*Content-based filtering* dalam proyek ini hanya memiliki {d['n_genres']} dimensi informasi tentang setiap
film, yaitu genre-nya. Dua film yang sangat berbeda tetap dianggap identik bila genre-nya sama,
sehingga skor kemiripan menumpuk pada satu nilai yang sama untuk banyak film sekaligus. Akibatnya
urutan top-N ditentukan oleh pemecah seri, bukan oleh preferensi — dan daftar yang dihasilkan
nyaris tidak mengandung film yang benar-benar disukai pengguna. *Collaborative filtering*
sebaliknya belajar dari puluhan ribu keputusan rating nyata dan dapat menemukan pola yang tidak
terlihat pada atribut film, misalnya kecenderungan seorang pengguna menyukai film dari sutradara
atau era tertentu.

**Catatan penting: kedua model kalah dari baseline popularitas pada Precision@10.**
*Baseline* yang hanya merekomendasikan film terpopuler memperoleh Precision@10
{base['Precision@10']:.4f}, lebih tinggi daripada Collaborative Filtering ({m[names[1]]['Precision@10']:.4f})
maupun Content-Based Filtering ({m[names[0]]['Precision@10']:.4f}). Ini adalah temuan yang lazim pada
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
   hanya {d['n_genres']} genre sebagai fitur, daya pisah model ini terbatas: skor kemiripan menumpuk pada
   nilai yang sama untuk banyak film, sehingga urutan top-N menjadi kurang bermakna
   (Precision@10 hanya {m[names[0]]['Precision@10']:.4f}).
2. **Pernyataan Masalah 2 terjawab.** Model *collaborative filtering* berbasis SVD dengan
   {d['k_factors']} faktor laten berhasil mengestimasi rating pada pasangan pengguna-film yang belum
   teramati, dengan RMSE {m[names[1]]['RMSE']:.4f} dan MAE {m[names[1]]['MAE']:.4f}. Kualitas daftar
   rekomendasinya jauh lebih baik daripada *content-based filtering*
   (Precision@10 {m[names[1]]['Precision@10']:.4f} berbanding {m[names[0]]['Precision@10']:.4f}).
3. **Pernyataan Masalah 3 terjawab.** Tidak ada satu model yang unggul di semua metrik, dan itulah
   jawaban yang jujur: **{acc_winner}** lebih baik dalam memprediksi *angka* rating, sementara
   **{rank_winner}** jauh lebih baik dalam menyusun *daftar* rekomendasi. Karena tujuan proyek ini
   adalah menyajikan top-N recommendation, **{rank_winner}** adalah pilihan yang lebih tepat untuk
   diterapkan. Akan tetapi, keduanya masih kalah dari *baseline* popularitas pada Precision@10
   ({base['Precision@10']:.4f}), sehingga belum ada model dalam proyek ini yang layak langsung
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
"""

report_path = SUB / f"{REPORT}.md"
report_path.write_text(report, encoding="utf-8")
print(f"      {report_path.name} ditulis ({len(report):,} karakter)")

# --------------------------------------------------------------------------
# Zip exactly the 3 files the spec requires
# --------------------------------------------------------------------------
print("[5/5] mengemas submission.zip...")
REQUIRED = [nb_path, SUB / "notebook.py", report_path]
archive = HERE.parent / "submission.zip"
with zf.ZipFile(archive, "w", zf.ZIP_DEFLATED) as z:
    for path in REQUIRED:
        z.write(path, arcname=path.name)

# Pindahkan artefak bantu keluar dari folder submission agar hanya 3 berkas yang tersisa.
if results_path.exists():
    shutil.move(str(results_path), str(HERE / results_path.name))

with zf.ZipFile(archive) as z:
    print(f"      {archive}")
    print("      isi arsip:")
    for info in z.infolist():
        print(f"        - {info.filename} ({info.file_size:,} bytes)")
print("\nSELESAI.")
