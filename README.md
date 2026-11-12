# Klasifikasi Penerima BLT Dana Desa — Naïve Bayes

Studi kasus: Kantor Desa Talulobutu, Kec. Tapa, Kab. Bone Bolango.

## Struktur folder

```
blt-dd-naive-bayes/
├── app.py                         # aplikasi Streamlit
├── requirements.txt
├── data/
│   └── Dataset Desa Talulobutu Terbaru.xlsx
├── notebooks/
│   └── 01_pemodelan_naive_bayes_blt_dd.ipynb   # pemodelan (alur Gambar 3.1)
├── src/
│   └── blt_dd.py                  # pra-pemrosesan, Tomek Links, pelatihan, evaluasi
├── models/                        # dibuat otomatis: model_blt_dd.joblib
└── outputs/                       # dibuat otomatis: grafik & tabel hasil (PNG/CSV)
```

## Instalasi (Windows)

```bat
cd D:\workspace\blt-dd-naive-bayes
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 atau lebih baru.

## 1. Jalankan notebook pemodelan

```bat
jupyter lab
```

Buka `notebooks/01_pemodelan_naive_bayes_blt_dd.ipynb`, lalu **Run → Run All Cells**.
Hasilnya:
- `models/model_blt_dd.joblib` — model untuk aplikasi
- `outputs/gambar/gNN_*.png` — gambar manuskrip (300 dpi), bernomor sesuai urutan kemunculan di notebook
- `outputs/tabel/tNN_*.csv` — tabel per tahapan Bab IV (preprocessing, prior, mean/varians, Tomek Links, prediksi, confusion matrix, perhitungan metrik, perhitungan manual, representasi pengetahuan) dan pengujian rasio pembagian data
- `outputs/tabel_hasil.xlsx` — seluruh tabel dalam satu file (satu sheet per tabel + sheet "Daftar Tabel")
- `outputs/rapidminer/` — data training/testing (asli dan numerik) untuk validasi di RapidMiner

## 2. Jalankan aplikasi Streamlit

```bat
streamlit run app.py
```

Menu aplikasi:
- **Data & Proses** — distribusi kelas, perbandingan atribut status, alur seleksi, 7 kriteria, sebaran atribut
- **Tahapan Proses** — setiap langkah klasifikasi (pembersihan, seleksi, pembagian, transformasi, Skenario 1, Skenario 2, evaluasi)
- **Prediksi Perorangan** — isi 15 atribut, hasil Layak/Tidak Layak + probabilitas
- **Prediksi Massal** — unggah Excel/CSV (unduh templatenya di aplikasi), hasil dapat diunduh
- **Evaluasi Model** — tabel metrik, confusion matrix, grafik perbandingan
- **Latih ulang model** (menu samping) — melatih model tanpa membuka notebook

## Catatan

- Encoder, Tomek Links, dan Class Weighting hanya di-*fit* pada data training untuk mencegah kebocoran data.
- Tomek Links diimplementasikan sendiri di `src/blt_dd.py` (setara `imblearn.under_sampling.TomekLinks(sampling_strategy="majority")`), sehingga tidak perlu menginstal `imbalanced-learn`.
- File model terikat pada versi scikit-learn. Jika pustaka diperbarui, jalankan ulang notebook atau klik **Latih model** di aplikasi.
- Folder `data/` berisi NIK dan nama warga. Jangan unggah ke repositori publik (sudah dikecualikan di `.gitignore`).
