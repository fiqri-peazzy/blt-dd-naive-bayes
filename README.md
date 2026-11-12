# Klasifikasi Penerima BLT Dana Desa Menggunakan Naïve Bayes

Studi kasus: Kantor Desa Talulobutu, Kecamatan Tapa, Kabupaten Bone Bolango.

Program ini terdiri atas dua bagian:

1. **Notebook pemodelan** (`notebooks/01_pemodelan_naive_bayes_blt_dd.ipynb`) berisi seluruh tahapan penelitian, mulai dari data, preprocessing, transformasi, pemodelan Gaussian Naïve Bayes, Tomek Links, Class Weighting, evaluasi, hingga perhitungan manual. Notebook ini juga menghasilkan semua tabel dan gambar untuk naskah.
2. **Aplikasi web Streamlit** (`app.py`) berisi tampilan interaktif tahapan penelitian, perbandingan hasil, serta prediksi kelayakan untuk data calon penerima baru.

Panduan ini ditulis untuk Windows dan memakai **Visual Studio Code (VS Code)** dengan terminal **Command Prompt (cmd)** atau **Git Bash**. Ikuti langkahnya secara berurutan.

---

## Daftar Isi

1. [Yang perlu disiapkan](#1-yang-perlu-disiapkan)
2. [Instalasi Python 3.13](#2-instalasi-python-313)
3. [Instalasi Visual Studio Code](#3-instalasi-visual-studio-code)
4. [Membuka folder program di VS Code](#4-membuka-folder-program-di-vs-code)
5. [Membuka terminal (cmd atau Git Bash)](#5-membuka-terminal-cmd-atau-git-bash)
6. [Membuat virtual environment dan menginstal pustaka](#6-membuat-virtual-environment-dan-menginstal-pustaka-cukup-sekali)
7. [Menjalankan aplikasi Streamlit](#7-menjalankan-aplikasi-streamlit)
8. [Menjalankan notebook pemodelan](#8-menjalankan-notebook-pemodelan)
9. [Menjalankan lagi di lain waktu](#9-menjalankan-lagi-di-lain-waktu)
10. [Isi aplikasi Streamlit](#10-isi-aplikasi-streamlit)
11. [Struktur folder](#11-struktur-folder)
12. [Mengatasi masalah yang sering terjadi](#12-mengatasi-masalah-yang-sering-terjadi)

---

## 1. Yang perlu disiapkan

| Kebutuhan | Keterangan |
|---|---|
| Sistem operasi | Windows 10 atau Windows 11 |
| Python | **Versi 3.13** (versi lain, terutama 3.14 ke atas, dapat gagal saat instalasi pustaka) |
| Editor | Visual Studio Code |
| Koneksi internet | Diperlukan saat instalasi pustaka (sekitar 300 MB) |
| Ruang disk | Sekitar 1,5 GB |

Ekstrak file zip program ke folder yang mudah ditemukan, misalnya `D:\blt-dd-naive-bayes`.
**Hindari** menaruh folder di dalam folder OneDrive yang sedang sinkron, karena dapat memperlambat atau mengunci file.

---

## 2. Instalasi Python 3.13

### 2.1 Cek apakah Python 3.13 sudah terpasang

Buka **Command Prompt** (tekan `Windows`, ketik `cmd`, Enter), lalu ketik:

```bat
py -3.13 --version
```

- Jika muncul `Python 3.13.x`, Python sudah siap. Lanjut ke [langkah 3](#3-instalasi-visual-studio-code).
- Jika muncul pesan error atau versi lain, lakukan langkah 2.2.

### 2.2 Unduh dan instal Python 3.13

1. Buka <https://www.python.org/downloads/windows/>.
2. Cari **Python 3.13.x** (versi 3.13 terbaru), lalu klik **Download Windows installer (64-bit)**.
3. Jalankan file installer yang diunduh.
4. Di layar pertama installer, **centang**:
   - **Add python.exe to PATH**
   - **Use admin privileges when installing py.exe** (bila ada)
5. Klik **Install Now**, tunggu sampai selesai, lalu klik **Close**.
6. Tutup Command Prompt, buka lagi, lalu cek ulang:

   ```bat
   py -3.13 --version
   ```

   Harus muncul `Python 3.13.x`.

> Bila komputer sudah memiliki versi Python lain (misalnya 3.15), tidak perlu dihapus. Panduan ini selalu memakai perintah `py -3.13` agar yang digunakan pasti versi 3.13.

---

## 3. Instalasi Visual Studio Code

1. Unduh dari <https://code.visualstudio.com/> lalu instal dengan pengaturan bawaan.
2. Buka VS Code, klik ikon **Extensions** di bilah kiri (ikon empat kotak) atau tekan `Ctrl+Shift+X`.
3. Cari dan instal ekstensi berikut:
   - **Python** (penerbit: Microsoft)
   - **Jupyter** (penerbit: Microsoft), diperlukan bila ingin membuka notebook langsung di VS Code

---

## 4. Membuka folder program di VS Code

1. Di VS Code pilih menu **File → Open Folder...**
2. Pilih folder hasil ekstrak, misalnya `D:\blt-dd-naive-bayes` (folder yang berisi `app.py` dan `README.md`).
3. Bila muncul pertanyaan *"Do you trust the authors of the files in this folder?"*, klik **Yes, I trust the authors**.

---

## 5. Membuka terminal (cmd atau Git Bash)

1. Pilih menu **Terminal → New Terminal** (atau tekan ``Ctrl+` ``).
2. Terminal akan muncul di bagian bawah VS Code. Pastikan alamatnya sudah berada di folder program, misalnya `D:\blt-dd-naive-bayes>`.
3. Untuk memilih jenis terminal, klik tanda panah kecil di samping tombol **+** pada panel terminal, lalu pilih:
   - **Command Prompt**, atau
   - **Git Bash** (hanya muncul bila Git for Windows sudah terpasang)

Pada langkah berikutnya, setiap perintah disediakan dalam dua versi. Gunakan yang sesuai dengan terminal yang dipilih.

> Bila yang terbuka adalah **PowerShell**, sebaiknya ganti ke Command Prompt seperti langkah 3 di atas agar perintah sama persis dengan panduan.

---

## 6. Membuat virtual environment dan menginstal pustaka (cukup sekali)

Virtual environment (`venv`) adalah folder tempat Python dan pustaka khusus program ini diinstal, sehingga tidak bercampur dengan program lain.

### 6.1 Buat virtual environment

**Command Prompt**
```bat
py -3.13 -m venv venv
```

**Git Bash**
```bash
py -3.13 -m venv venv
```

Akan muncul folder baru bernama `venv` di dalam folder program.

### 6.2 Aktifkan virtual environment

**Command Prompt**
```bat
venv\Scripts\activate
```

**Git Bash**
```bash
source venv/Scripts/activate
```

Bila berhasil, di awal baris terminal muncul tulisan **`(venv)`**.

> Bila VS Code menampilkan pesan *"We noticed a new environment has been created. Do you want to select it for the workspace folder?"*, klik **Yes**.

### 6.3 Instal pustaka

Pastikan tulisan `(venv)` masih tampil, lalu jalankan:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Tunggu hingga selesai (sekitar 3 sampai 10 menit, tergantung kecepatan internet). Instalasi berhasil bila baris terakhir berbunyi `Successfully installed ...` tanpa tulisan `ERROR`.

### 6.4 Cek instalasi

```bash
python -c "import pandas, sklearn, streamlit; print('Instalasi berhasil')"
```

Bila tertulis `Instalasi berhasil`, program siap dijalankan.

---

## 7. Menjalankan aplikasi Streamlit

Pastikan virtual environment aktif (ada tulisan `(venv)`), lalu jalankan:

```bash
streamlit run app.py
```

- Browser akan terbuka otomatis ke alamat **<http://localhost:8501>**.
- Bila browser tidak terbuka, salin alamat `Local URL` yang tertera di terminal ke browser.
- Saat pertama kali dijalankan, Streamlit mungkin menanyakan email. Cukup tekan **Enter** untuk melewatinya.
- Untuk **menghentikan** aplikasi, kembali ke terminal lalu tekan `Ctrl+C`.

Model sudah tersedia di folder `models/`, sehingga aplikasi dapat langsung digunakan tanpa menjalankan notebook terlebih dahulu.

---

## 8. Menjalankan notebook pemodelan

Notebook digunakan untuk melihat dan menjalankan ulang seluruh tahapan penelitian. Pilih **salah satu** cara berikut.

### Cara A: JupyterLab di browser

Pastikan virtual environment aktif, lalu jalankan:

```bash
jupyter lab
```

1. Browser akan membuka JupyterLab.
2. Di panel kiri, buka folder `notebooks`, lalu klik dua kali `01_pemodelan_naive_bayes_blt_dd.ipynb`.
3. Pilih menu **Kernel → Restart Kernel and Run All Cells...** lalu klik **Restart**.
4. Tunggu hingga semua sel selesai (sekitar 1 sampai 2 menit).
5. Untuk menutup JupyterLab, kembali ke terminal lalu tekan `Ctrl+C`.

### Cara B: Langsung di VS Code

1. Di panel Explorer VS Code, buka `notebooks/01_pemodelan_naive_bayes_blt_dd.ipynb`.
2. Klik **Select Kernel** di pojok kanan atas, pilih **Python Environments**, lalu pilih yang bertanda **venv** (`venv\Scripts\python.exe`, Python 3.13).
3. Klik **Run All** di bagian atas notebook.

### Hasil notebook

Setelah notebook selesai dijalankan, file berikut dibuat atau diperbarui:

| Lokasi | Isi |
|---|---|
| `models/model_blt_dd.joblib` | Model yang dipakai aplikasi Streamlit |
| `outputs/gambar/` | Gambar untuk naskah (format PNG, 300 dpi), bernomor sesuai urutan di notebook |
| `outputs/tabel/` | Tabel per tahapan (format CSV) |
| `outputs/tabel_hasil.xlsx` | Seluruh tabel dalam satu file Excel (satu sheet per tabel dan sheet "Daftar Tabel") |
| `outputs/rapidminer/` | Data training dan testing untuk validasi di RapidMiner |

---

## 9. Menjalankan lagi di lain waktu

Instalasi (langkah 2, 3, dan 6) cukup dilakukan **sekali**. Untuk penggunaan berikutnya:

1. Buka VS Code, lalu **File → Open Folder...** ke folder program.
2. Buka terminal (**Terminal → New Terminal**).
3. Aktifkan virtual environment:
   - Command Prompt: `venv\Scripts\activate`
   - Git Bash: `source venv/Scripts/activate`
4. Jalankan program yang diinginkan:
   - Aplikasi: `streamlit run app.py`
   - Notebook: `jupyter lab`

---

## 10. Isi aplikasi Streamlit

| Menu | Isi |
|---|---|
| **Beranda** | Alur penelitian dan ringkasan hasil kedua eksperimen |
| **Data Penelitian** | Cuplikan dataset, atribut status, alur seleksi, distribusi kelas, 7 kriteria BLT-DD, sebaran atribut |
| **Tahapan Pemodelan** | Pilih Eksperimen 1 (15 atribut) atau Eksperimen 2 (8 atribut), lalu lihat setiap tahap: preprocessing, seleksi dan transformasi, Skenario 1, Skenario 2, evaluasi, dan perhitungan manual (data testing dapat dipilih) |
| **Perbandingan Hasil** | Rekap kedua eksperimen, aturan IF-ELSE, pengujian beberapa rasio pembagian data |
| **Prediksi Data Baru** | Isi 8 atribut calon penerima untuk mendapatkan hasil Layak atau Tidak Layak beserta probabilitasnya. Nilai di luar dataset tetap dapat diprediksi dan diberi catatan. Riwayat prediksi dapat diunduh. |
| **Prediksi Massal** | Unggah file Excel atau CSV (template tersedia di aplikasi), hasil prediksi dan daftar prioritas berdasarkan kuota dapat diunduh |
| **Tentang** | Informasi penelitian |

Menu samping **Latih ulang model** digunakan untuk melatih ulang model dari dataset tanpa membuka notebook.

### Ringkasan penelitian

- **Target:** Status Kelayakan Verifikasi (Layak / Tidak Layak), 392 kepala keluarga.
- **Eksperimen 1:** 15 atribut, termasuk 7 atribut kriteria BLT-DD. Akurasi mencapai 100% karena label disusun dari aturan *minimal satu kriteria terpenuhi*, sehingga model hanya meniru aturan tersebut.
- **Eksperimen 2 (model final):** 8 atribut sosial-ekonomi tanpa 7 atribut kriteria.
- **Skenario 1:** Gaussian Naïve Bayes. **Skenario 2:** Tomek Links + Class Weighting + Gaussian Naïve Bayes.
- Pembagian data 80% training dan 20% testing (*stratified*, `random_state = 42`).

---

## 11. Struktur folder

```
blt-dd-naive-bayes/
├── app.py                     Aplikasi Streamlit
├── requirements.txt           Daftar pustaka beserta versinya
├── README.md                  Panduan ini
├── data/
│   └── Dataset Desa Talulobutu Terbaru.xlsx
├── notebooks/
│   └── 01_pemodelan_naive_bayes_blt_dd.ipynb
├── src/
│   ├── blt_dd.py              Pembersihan data, transformasi, Tomek Links, pelatihan, evaluasi
│   ├── blt_steps.py           Tabel setiap tahapan (dipakai notebook dan aplikasi)
│   └── blt_viz.py             Grafik (dipakai notebook dan aplikasi)
├── models/
│   └── model_blt_dd.joblib    Model hasil pelatihan
├── outputs/                   Gambar dan tabel hasil notebook
└── venv/                      Dibuat sendiri pada langkah 6 (tidak termasuk di dalam zip)
```

> **Penting:** folder `data/` berisi NIK dan nama warga. Jangan dibagikan ke pihak lain atau diunggah ke internet.

---

## 12. Mengatasi masalah yang sering terjadi

| Masalah | Penyebab | Solusi |
|---|---|---|
| `'py' is not recognized ...` | Python belum terpasang atau py launcher tidak ikut terpasang | Ulangi [langkah 2.2](#22-unduh-dan-instal-python-313), pastikan opsi **Add python.exe to PATH** dicentang, lalu buka ulang terminal dan VS Code |
| `Python 3.13 not found` saat menjalankan `py -3.13` | Versi 3.13 belum terpasang | Instal Python 3.13 sesuai [langkah 2.2](#22-unduh-dan-instal-python-313) |
| `'streamlit' is not recognized ...` atau `'jupyter' is not recognized ...` | Virtual environment belum aktif | Aktifkan dahulu (`venv\Scripts\activate` atau `source venv/Scripts/activate`) sampai muncul `(venv)` |
| Muncul error `Microsoft Visual C++ 14.0 or greater is required` saat `pip install` | Versi Python yang dipakai bukan 3.13 | Hapus folder `venv`, lalu ulangi langkah 6 dengan perintah `py -3.13 -m venv venv` |
| `No module named ...` saat menjalankan notebook di VS Code | Kernel yang dipilih bukan `venv` | Klik **Select Kernel** lalu pilih Python dari folder `venv` |
| Notebook error `cannot import name ...` setelah file di `src/` berubah | Kernel masih menyimpan versi lama modul | Pilih **Kernel → Restart Kernel and Run All Cells** |
| `Port 8501 is already in use` | Aplikasi Streamlit masih berjalan di terminal lain | Tutup terminal lama atau jalankan `streamlit run app.py --server.port 8502` |
| Aplikasi menampilkan "Model belum tersedia" atau "Model gagal dibaca" | File model hilang atau versi pustaka berbeda | Klik **Latih ulang model** pada menu samping aplikasi, atau jalankan ulang notebook |
| Error `Permission denied` saat notebook menyimpan file Excel | File di folder `outputs/` sedang dibuka di Excel | Tutup file Excel tersebut, lalu jalankan ulang notebook |
| Git Bash: `venv/Scripts/activate: No such file or directory` | Perintah dijalankan di luar folder program | Pindah ke folder program, misalnya `cd /d/blt-dd-naive-bayes`, lalu ulangi |

Bila masalah belum teratasi, kirimkan tangkapan layar pesan error lengkap dari terminal.
