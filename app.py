"""
Aplikasi Streamlit - Klasifikasi Kelayakan Penerima BLT Dana Desa (Gaussian Naive Bayes)
Studi kasus: Desa Talulobutu, Kec. Tapa, Kab. Bone Bolango

Jalankan:  streamlit run app.py

Alur penelitian yang ditampilkan:
- Eksperimen 1: 15 atribut (termasuk 7 atribut kriteria BLT-DD)
- Eksperimen 2: 8 atribut sosial-ekonomi (model final yang dipakai untuk prediksi)
Setiap eksperimen diuji dengan Skenario 1 (Gaussian NB) dan Skenario 2 (Tomek Links +
Class Weighting + Gaussian NB). Seluruh tabel memakai modul yang sama dengan notebook.
"""
import io
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
import blt_dd  # noqa: E402
import blt_steps as steps  # noqa: E402
import blt_viz as viz  # noqa: E402
from blt_dd import (CRITERIA_COLS, EDU_ORDER, FEATURE_COLS, FEATURE_COLS_15,  # noqa: E402
                    TARGET, build_artifact, clean_data, encode_target, evaluate)

MODEL_PATH = BASE_DIR / "models" / "model_blt_dd.joblib"
DATA_PATH = BASE_DIR / "data" / "Dataset Desa Talulobutu Terbaru.xlsx"
TABEL_DIR = BASE_DIR / "outputs" / "tabel"
METRIK = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity"]
NAMA_EKSP = {"Eksperimen 1 - 15 atribut": "e1", "Eksperimen 2 - 8 atribut sosial-ekonomi": "e2"}

st.set_page_config(page_title="Klasifikasi BLT-DD", layout="wide")


# ---------------------------------------------------------------------------
# Utilitas tampilan
# ---------------------------------------------------------------------------
def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def tabel(df: pd.DataFrame, judul: str = "", desimal: int = 4, tinggi: int | None = None):
    """Tabel dengan format angka Indonesia (koma desimal), sama seperti notebook."""
    if judul:
        st.markdown(f"**{judul}**")
    df = df.copy()
    for c in df.columns:   # kolom campuran (mis. baris Total/Selisih) -> teks berformat Indonesia
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: viz.angka_id(v, desimal) if isinstance(v, float)
                              else ("" if v is None else str(v)))
    kw = {"height": tinggi} if tinggi else {}
    st.dataframe(df.style.format(precision=desimal, decimal=",", thousands="."),
                 hide_index=True, width="stretch", **kw)


def info(teks: str):
    st.markdown(f"<div style='color:#444;font-size:0.95em;line-height:1.55'>{teks}</div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data, eksperimen, dan model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifact(path: str, mtime: float):
    return joblib.load(path)


def get_artifact():
    if MODEL_PATH.exists():
        try:
            return load_artifact(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
        except Exception as e:  # versi pustaka berbeda, file rusak, dll.
            st.sidebar.error(f"Model gagal dibaca: {e}. Silakan latih ulang model.")
    return None


def save_artifact(artifact):
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    load_artifact.clear()


@st.cache_resource
def load_penelitian(path: str, mtime: float) -> dict:
    """Menjalankan ulang kedua eksperimen dari dataset (identik dengan notebook)."""
    df_raw = pd.read_excel(path)
    df = clean_data(df_raw)
    E15 = steps.jalankan_eksperimen(df, FEATURE_COLS_15)
    E8 = steps.jalankan_eksperimen(df, FEATURE_COLS)
    return {"df_raw": df_raw, "df": df, "e1": E15, "e2": E8,
            "rekap": steps.rekap_eksperimen(E15, E8),
            "aturan_if": steps.evaluasi_aturan_if(df, E15),
            "rasio_e1": steps.pengujian_rasio_split(df, feature_cols=FEATURE_COLS_15),
            "rasio_e2": steps.pengujian_rasio_split(df, feature_cols=FEATURE_COLS)}


def predict_df(artifact, raw: pd.DataFrame, model_name: str) -> pd.DataFrame:
    enc = artifact["encoder"]
    model = artifact["models"][model_name]
    X = enc.transform(clean_data(raw)[artifact["feature_cols"]])
    proba = model.predict_proba(X)[:, 1]
    out = raw.copy()
    out["Prediksi"] = np.where(proba >= 0.5, "Layak", "Tidak Layak")
    out["Probabilitas Layak"] = proba.round(4)
    return out


def ranking_prioritas(hasil: pd.DataFrame) -> pd.DataFrame:
    """Urutkan warga berprediksi Layak: jumlah kriteria (bila kolom tersedia), probabilitas, lalu usia."""
    layak = hasil[hasil["Prediksi"] == "Layak"].copy()
    if all(c in layak.columns for c in CRITERIA_COLS):
        layak["Jumlah Kriteria Terpenuhi"] = viz.kriteria_terpenuhi(clean_data(layak[CRITERIA_COLS])).sum(axis=1).values
    else:
        layak["Jumlah Kriteria Terpenuhi"] = 0
    layak["Usia"] = pd.to_numeric(layak["Usia"], errors="coerce")
    layak = layak.sort_values(["Jumlah Kriteria Terpenuhi", "Probabilitas Layak", "Usia"], ascending=False)
    layak.insert(0, "Peringkat", range(1, len(layak) + 1))
    return layak


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Klasifikasi BLT-DD")
st.sidebar.caption("Gaussian Naive Bayes - Desa Talulobutu")

artifact = get_artifact()
ada_data = DATA_PATH.exists()

HALAMAN = ["Beranda", "Data Penelitian", "Tahapan Pemodelan", "Perbandingan Hasil",
           "Prediksi Data Baru", "Prediksi Massal", "Tentang"]
halaman = st.sidebar.radio("Menu", HALAMAN)

with st.sidebar.expander("Latih ulang model", expanded=artifact is None):
    st.write("Melatih ulang model Eksperimen 2 (8 atribut) dengan alur yang sama seperti notebook "
             "(pembagian 80:20 stratified, random_state 42).")
    up_train = st.file_uploader("Dataset (.xlsx), kosongkan untuk memakai data bawaan", type=["xlsx"], key="train")
    if st.button("Latih model", type="primary"):
        try:
            if up_train is not None:
                df_train = pd.read_excel(up_train)
            elif ada_data:
                df_train = pd.read_excel(DATA_PATH)
            else:
                st.error(f"Dataset tidak ditemukan di {DATA_PATH}")
                st.stop()
            with st.spinner("Melatih model..."):
                save_artifact(build_artifact(df_train))
            st.success("Model berhasil dilatih dan disimpan.")
            st.rerun()
        except Exception as e:
            st.error(f"Gagal melatih model: {e}")

if artifact is not None:
    model_names = list(artifact["models"].keys())
    model_name = st.sidebar.radio("Model untuk prediksi", model_names,
                                  index=model_names.index(artifact.get("default_model", model_names[-1])),
                                  format_func=lambda n: n.split(" - ")[0] + (" (usulan)" if "Tomek" in n else " (baseline)"))
    st.sidebar.markdown(
        f"**Atribut model:** {len(artifact['feature_cols'])}  \n"
        f"**Data training / testing:** {artifact['n_train']} / {artifact['n_test']}  \n"
        f"**Dilatih:** {artifact['created_at']}")

P = load_penelitian(str(DATA_PATH), DATA_PATH.stat().st_mtime) if ada_data else None


def butuh_data():
    if P is None:
        st.info(f"Halaman ini memerlukan dataset penelitian di {DATA_PATH}.")
        st.stop()


def butuh_model():
    if artifact is None:
        st.warning("Model belum tersedia. Jalankan notebook pemodelan atau klik Latih model pada menu samping.")
        st.stop()


# ---------------------------------------------------------------------------
# Halaman: Beranda
# ---------------------------------------------------------------------------
if halaman == "Beranda":
    st.title("Klasifikasi Kelayakan Penerima BLT Dana Desa")
    st.caption("Studi kasus Desa Talulobutu, Kecamatan Tapa, Kabupaten Bone Bolango - Metode Gaussian Naive Bayes")
    info("Aplikasi ini menyajikan seluruh tahapan penelitian, mulai dari data, preprocessing, transformasi, "
         "pemodelan Gaussian Naive Bayes, penanganan ketidakseimbangan kelas dengan Tomek Links dan Class "
         "Weighting, evaluasi confusion matrix, hingga prediksi kelayakan calon penerima baru.")

    st.subheader("Alur penelitian")
    st.markdown("""
1. **Pengumpulan data**: 392 kepala keluarga, 21 kolom (data kependudukan, berita acara musyawarah desa, dan DTKS).
2. **Preprocessing**: pemeriksaan data, seleksi atribut, pembagian data 80% training dan 20% testing (stratified), transformasi data.
3. **Eksperimen 1 (15 atribut)**: seluruh atribut sosial-ekonomi termasuk 7 atribut kriteria BLT-DD.
4. **Analisis Eksperimen 1**: label Status Kelayakan Verifikasi ternyata mengikuti aturan *minimal satu kriteria terpenuhi*, sehingga model hanya meniru aturan IF-ELSE dan akurasi mencapai 100%.
5. **Eksperimen 2 (8 atribut)**: 7 atribut kriteria dikeluarkan, model mempelajari kelayakan dari kondisi sosial-ekonomi. Model ini menjadi model final.
6. **Setiap eksperimen** diuji dengan Skenario 1 (Gaussian Naive Bayes) dan Skenario 2 (Tomek Links + Class Weighting + Gaussian Naive Bayes), lalu dievaluasi dengan confusion matrix.
""")
    if P is not None:
        st.subheader("Ringkasan hasil pada data testing")
        r = P["rekap"]
        c = st.columns(4)
        for col, (_, row) in zip(c, r.iterrows()):
            col.metric(f"{row['Eksperimen'].split(' (')[0]} - {row['Skenario']}",
                       viz.angka_id(row["Accuracy"] * 100, 2) + "%",
                       "Recall " + viz.angka_id(row["Recall"] * 100, 2) + "%", delta_color="off")
        tabel(r, "Rekapitulasi kedua eksperimen")
        info("Eksperimen 1 mencapai 100% karena label dapat ditentukan langsung dari 7 atribut kriteria. "
             "Eksperimen 2 menghasilkan akurasi sekitar 80% dan memperlihatkan pengaruh Tomek Links dan "
             "Class Weighting pada kelas minoritas (Layak).")


# ---------------------------------------------------------------------------
# Halaman: Data Penelitian
# ---------------------------------------------------------------------------
elif halaman == "Data Penelitian":
    butuh_data()
    df = P["df"]
    st.title("Data Penelitian")
    n = len(df)
    n_layak = int((df[TARGET] == "Layak").sum())
    n_pen = int((df[viz.STATUS_PENERIMA] == "Ya").sum())
    m = st.columns(4)
    m[0].metric("Jumlah KK", n)
    m[1].metric("Lolos verifikasi (Layak)", n_layak)
    m[2].metric("Penerima BLT-DD", n_pen)
    m[3].metric("Rasio Tidak Layak : Layak", f"{(n - n_layak) / n_layak:.2f} : 1".replace(".", ","))

    t1, t2, t3, t4, t5 = st.tabs(["Dataset", "Atribut status dan alur seleksi", "Distribusi kelas",
                                  "Tujuh kriteria BLT-DD", "Sebaran atribut"])
    with t1:
        info("Cuplikan dataset hasil pengumpulan. NIK dan Nama disamarkan untuk menjaga kerahasiaan data pribadi.")
        jumlah = st.slider("Jumlah baris yang ditampilkan", 5, 50, 10)
        tabel(steps.cuplikan_dataset(P["df_raw"], n=jumlah), desimal=0)
    with t2:
        info("Dataset memuat tiga atribut status. Status Kelayakan dan Status Penerima BLT-DD identik dan "
             "berisi penerima akhir (dipengaruhi kuota anggaran), sedangkan Status Kelayakan Verifikasi berisi "
             "warga yang lolos verifikasi kriteria dan dipakai sebagai target penelitian.")
        tabel(viz.tabel_atribut_status(df), "Perbandingan atribut status")
        fig, tstatus = viz.fig_alur_seleksi(df)
        c1, c2 = st.columns([1.5, 1])
        with c1:
            show(fig)
        with c2:
            tabel(tstatus, "Hubungan ketiga atribut status", desimal=0)
    with t3:
        info("Kelas Layak merupakan kelas minoritas (36,22%) sehingga terdapat ketidakseimbangan kelas tingkat ringan "
             "dengan rasio 1:1,76.")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            show(viz.fig_distribusi_kelas(df))
        with c2:
            tabel(viz.tabel_distribusi_kelas(df), "Distribusi Status Kelayakan Verifikasi", desimal=2)
    with t4:
        info("Seluruh warga yang memenuhi minimal satu dari 7 kriteria berstatus Layak, dan seluruh warga yang tidak "
             "memenuhi satu pun kriteria berstatus Tidak Layak, tanpa pengecualian. Pola inilah yang membuat "
             "Eksperimen 1 mencapai akurasi 100%.")
        tabel(viz.tabel_aturan_kriteria(df), "Pemenuhan 7 kriteria terhadap Status Kelayakan Verifikasi", desimal=0)
        fig, tk = viz.fig_kriteria(df)
        show(fig)
        c1, c2 = st.columns(2)
        with c1:
            tabel(tk, "Jumlah kriteria terpenuhi per kelompok", desimal=0)
        with c2:
            tabel(viz.tabel_profil_kelompok(df), "Profil rata-rata per kelompok", desimal=2)
    with t5:
        info("Sebaran atribut numerik dan kategorikal terhadap Status Kelayakan Verifikasi.")
        show(viz.fig_atribut_numerik(df))
        show(viz.fig_atribut_kategorikal(df))


# ---------------------------------------------------------------------------
# Halaman: Tahapan Pemodelan
# ---------------------------------------------------------------------------
elif halaman == "Tahapan Pemodelan":
    butuh_data()
    st.title("Tahapan Pemodelan")
    pilih = st.radio("Pilih eksperimen", list(NAMA_EKSP), horizontal=True)
    kode = NAMA_EKSP[pilih]
    E = P[kode]
    if kode == "e1":
        info("Eksperimen 1 memakai 15 atribut, termasuk 7 atribut kriteria BLT-DD (DTKS, KK miskin, perempuan KK "
             "miskin, KK tunggal lanjut usia, kehilangan mata pencaharian, anggota keluarga sakit kronis, dan anggota "
             "keluarga disabilitas).")
    else:
        info("Eksperimen 2 memakai 8 atribut sosial-ekonomi. Tujuh atribut kriteria dikeluarkan karena label disusun "
             "dari aturan minimal satu kriteria terpenuhi, sehingga Naive Bayes benar-benar diuji untuk mempelajari "
             "pola kelayakan dari kondisi sosial-ekonomi warga.")

    tabs = st.tabs(["Preprocessing", "Seleksi dan transformasi", "Skenario 1", "Skenario 2",
                    "Evaluasi", "Perhitungan manual"])
    with tabs[0]:
        info("Pembersihan data memeriksa nilai kosong, NIK ganda, dan baris duplikat. Data kemudian dibagi 80% "
             "training dan 20% testing secara stratified sehingga proporsi kelas pada kedua bagian sama. Pembagian yang "
             "sama dipakai pada kedua eksperimen.")
        tabel(steps.tabel_pembersihan(P["df_raw"]), "Hasil pembersihan data", desimal=0)
        c1, c2 = st.columns([1, 1.2])
        with c1:
            tabel(E["t_pembagian"], "Pembagian data", desimal=0)
        with c2:
            show(viz.fig_pembagian_data(E["y_train"], E["y_test"]))
    with tabs[1]:
        info("Atribut identitas dan status administratif dikeluarkan. Atribut kategorikal diubah menjadi angka: "
             "atribut biner menjadi 0/1, pendidikan menjadi ordinal, serta pekerjaan dan status pernikahan menjadi "
             "kolom one-hot. Aturan transformasi ditentukan dari data training lalu diterapkan ke data testing.")
        tabel(E["t_seleksi"], "Hasil seleksi atribut")
        tabel(E["t_cuplikan_seleksi"], "Cuplikan data hasil seleksi", desimal=0)
        tabel(E["t_aturan"], "Aturan transformasi data")
        tabel(E["t_cuplikan_transformasi"], f"Cuplikan data training hasil transformasi ({E['X_train'].shape[1]} kolom)",
              desimal=0)
    with tabs[2]:
        info("Skenario 1 melatih Gaussian Naive Bayes tanpa penanganan ketidakseimbangan kelas. Langkah 1 menghitung "
             "probabilitas prior P(C), Langkah 2 menghitung rata-rata dan varians setiap atribut pada setiap kelas, dan "
             "Langkah 3 menghitung posterior setiap data testing serta menentukan kelas dengan posterior terbesar.")
        tabel(E["t_prior1"], "Langkah 1 - Probabilitas prior")
        tabel(E["t_param1"], "Langkah 2 - Mean dan varians setiap atribut per kelas", tinggi=380)
        tabel(E["t_pred1"], "Langkah 3 - Posterior dan hasil prediksi data testing", tinggi=380)
    with tabs[3]:
        info("Skenario 2 menerapkan Tomek Links pada data training: pasangan data berbeda kelas yang saling menjadi "
             "tetangga terdekat dicari, lalu data kelas mayoritas (Tidak Layak) pada pasangan tersebut dihapus. "
             "Selanjutnya Class Weighting memberi bobot lebih besar pada kelas minoritas sehingga prior kedua kelas "
             "menjadi seimbang.")
        tabel(E["t_tomek"], f"Langkah 1 - Pasangan Tomek Link ({len(E['pasangan'])} pasangan)")
        c1, c2 = st.columns([1.6, 1])
        with c1:
            show(viz.fig_tomek_links(E["Xs"], E["y_train"], E["idx_hapus"], E["pasangan"]))
        with c2:
            tabel(E["t_hasil_tomek"], "Data training sebelum dan sesudah Tomek Links", desimal=0)
        tabel(E["t_prior2"], "Langkah 2 - Bobot kelas dan prior berbobot")
        show(viz.fig_class_weighting(E["y_train"], E["y_tl"], E["bobot"], E["prior1"], E["prior2"])[0])
        tabel(E["t_param2"], "Langkah 3 - Mean dan varians berbobot setiap atribut per kelas", tinggi=380)
        tabel(E["t_pred2"], "Langkah 4 - Posterior dan hasil prediksi data testing", tinggi=380)
    with tabs[4]:
        info("Kedua skenario diuji pada data testing yang sama. Hasilnya disusun dalam confusion matrix, kemudian "
             "dihitung accuracy, precision, recall, F1-score, dan specificity dengan kelas positif Layak.")
        show(viz.fig_confusion_pair(E["hasil_s1"], E["hasil_s2"]))
        c1, c2 = st.columns(2)
        with c1:
            tabel(E["t_cm1"], "Confusion matrix Skenario 1", desimal=0)
            tabel(E["t_metrik1"], "Perhitungan metrik Skenario 1")
        with c2:
            tabel(E["t_cm2"], "Confusion matrix Skenario 2", desimal=0)
            tabel(E["t_metrik2"], "Perhitungan metrik Skenario 2")
        tabel(E["t_perbandingan"], "Perbandingan Skenario 1 dan Skenario 2")
        perb = pd.DataFrame({"Skenario 1 (Baseline)": E["hasil_s1"], "Skenario 2 (Tomek Links + CW)": E["hasil_s2"]}).T
        c1, c2 = st.columns(2)
        with c1:
            show(viz.fig_perbandingan_metrik(perb))
        with c2:
            show(viz.fig_probabilitas(E["y_test"], E["proba1"], E["proba2"]))
        if kode == "e1":
            info("Pada Eksperimen 1, aturan IF-ELSE sederhana (minimal satu kriteria terpenuhi = Layak) menghasilkan "
                 "nilai yang sama persis dengan Gaussian Naive Bayes. Artinya model hanya meniru aturan penyusun label.")
            tabel(P["aturan_if"], "Perbandingan aturan IF-ELSE dan Gaussian Naive Bayes (data testing yang sama)")
    with tabs[5]:
        info("Perhitungan manual Gaussian Naive Bayes (Skenario 1) untuk satu data testing. Likelihood dihitung dalam "
             "bentuk logaritma natural agar tidak terjadi underflow, lalu posterior manual dibandingkan dengan "
             "posterior yang dihitung sistem.")
        no = E["no_test"]
        default = steps.perhitungan_manual(E)["urutan"]
        urutan = st.selectbox("Pilih data testing", list(range(len(no))), index=default,
                              format_func=lambda i: f"Urutan {i + 1} - No. data {no[i]} - aktual "
                                                    f"{'Layak' if E['y_test'][i] == 1 else 'Tidak Layak'}")
        M = steps.perhitungan_manual(E, urutan)
        c1, c2 = st.columns([1, 2])
        with c1:
            tabel(M["nilai_asli"], "Nilai atribut data yang dihitung", desimal=0)
        with c2:
            tabel(M["t_validasi"], "Posterior perhitungan manual dan sistem")
            st.markdown(f"Kelas prediksi: **{M['prediksi']}** (aktual: {M['aktual']})")
        tabel(M["t_likelihood"], "Mean, varians, dan likelihood per atribut", tinggi=420)


# ---------------------------------------------------------------------------
# Halaman: Perbandingan Hasil
# ---------------------------------------------------------------------------
elif halaman == "Perbandingan Hasil":
    butuh_data()
    st.title("Perbandingan Hasil")
    info("Eksperimen 1 (15 atribut) mencapai nilai sempurna pada kedua skenario karena 7 atribut kriteria menentukan "
         "label secara langsung. Setelah atribut kriteria dikeluarkan, Eksperimen 2 (8 atribut) menghasilkan "
         "akurasi sekitar 80%, dan Skenario 2 menurunkan False Negative sehingga recall dan F1-score kelas Layak "
         "meningkat.")
    show(viz.fig_rekap_eksperimen(P["rekap"]))
    tabel(P["rekap"], "Rekapitulasi kedua eksperimen")
    tabel(P["aturan_if"], "Aturan IF-ELSE dibandingkan Gaussian Naive Bayes pada Eksperimen 1")

    st.subheader("Pengujian beberapa rasio pembagian data")
    info("Kedua skenario diuji ulang pada rasio pembagian data 70:30 sampai 90:10 untuk melihat konsistensi hasil.")
    pilih = st.radio("Eksperimen", list(NAMA_EKSP), horizontal=True, key="rasio")
    tabel(P["rasio_" + NAMA_EKSP[pilih]])

    csv_banding = sorted(TABEL_DIR.glob("*cadangan_pembanding_algoritma.csv"))
    if csv_banding:
        st.subheader("Pembanding algoritma pada 8 atribut")
        info("Validasi silang 10-fold yang diulang 5 kali. Algoritma lain pun berada pada kisaran akurasi yang sama, "
             "sehingga batas akurasi berasal dari informasi pada 8 atribut sosial-ekonomi, bukan dari metode.")
        tabel(pd.read_csv(csv_banding[-1]), "Pembanding algoritma (hasil notebook)")


# ---------------------------------------------------------------------------
# Halaman: Prediksi Data Baru
# ---------------------------------------------------------------------------
elif halaman == "Prediksi Data Baru":
    butuh_model()
    enc = artifact["encoder"]
    st.title("Prediksi Data Calon Penerima Baru")
    info(f"Isi {len(artifact['feature_cols'])} atribut sosial-ekonomi calon penerima. Nilai di luar dataset penelitian "
         "(misalnya pekerjaan atau status pernikahan lain, pendidikan S3, usia atau tanggungan di luar rentang data "
         "training) tetap dapat diprediksi, dan sistem menampilkan catatan penanganannya.")
    LAIN = "Lainnya (tulis di kolom isian)"
    pekerjaan_opts = [c for c in enc.categories_["Pekerjaan"] if c != "Lainnya"] + [LAIN]
    nikah_opts = list(dict.fromkeys([c for c in enc.categories_["Status Pernikahan"] if c != "Lainnya"]
                                    + ["Cerai Mati", LAIN]))
    edu_opts = list(EDU_ORDER)

    with st.form("form_prediksi"):
        nama = st.text_input("Nama atau keterangan (opsional, hanya untuk riwayat)")
        c1, c2, c3 = st.columns(3)
        with c1:
            usia = st.number_input("Usia (tahun)", min_value=0, max_value=120, value=45)
            jk = st.selectbox("Jenis Kelamin", ["L", "P"], format_func=lambda v: "Laki-laki" if v == "L" else "Perempuan")
            pendidikan = st.selectbox("Pendidikan Terakhir", edu_opts, index=edu_opts.index("SMA"))
        with c2:
            pekerjaan_sel = st.selectbox("Pekerjaan", pekerjaan_opts)
            pekerjaan_lain = st.text_input("Pekerjaan lain (bila tidak ada di daftar)")
            nikah_sel = st.selectbox("Status Pernikahan", nikah_opts,
                                     index=nikah_opts.index("Kawin") if "Kawin" in nikah_opts else 0)
            nikah_lain = st.text_input("Status pernikahan lain (bila tidak ada di daftar)")
        with c3:
            tanggungan = st.number_input("Tanggungan Keluarga (orang)", min_value=0, max_value=30, value=2)
            tinggal = st.radio("Status Tempat Tinggal", ["Milik Sendiri", "Menumpang"], horizontal=True)
            bantuan = st.radio("Menerima Bantuan Lain", ["Tidak", "Ya"], horizontal=True)
        kirim = st.form_submit_button("Prediksi", type="primary")

    if kirim:
        pekerjaan = pekerjaan_lain.strip() or (pekerjaan_sel if pekerjaan_sel != LAIN else "Lainnya")
        nikah = nikah_lain.strip() or (nikah_sel if nikah_sel != LAIN else "Lainnya")
        row = pd.DataFrame([{"Usia": usia, "Jenis Kelamin": jk, "Pekerjaan": pekerjaan,
                             "Pendidikan terakhir": pendidikan, "Status Pernikahan": nikah,
                             "Tanggungan Keluarga": tanggungan, "Status Tempat Tinggal": tinggal,
                             "Menerima Bantuan Lain": bantuan}])
        try:
            per_model = {n: predict_df(artifact, row, n).iloc[0] for n in artifact["models"]}
        except Exception as e:
            st.error(f"Data tidak dapat diproses: {e}")
            st.stop()
        hasil = per_model[model_name]
        p = float(hasil["Probabilitas Layak"])
        st.subheader(f"Hasil prediksi - {model_name.split(' - ')[0]}")
        persen = viz.angka_id(p * 100, 2) + "%"
        if hasil["Prediksi"] == "Layak":
            st.success(f"LAYAK menerima BLT-DD. Probabilitas Layak: {persen}")
        else:
            st.error(f"TIDAK LAYAK menerima BLT-DD. Probabilitas Layak: {persen}")
        st.progress(min(max(p, 0.0), 1.0))
        tabel(pd.DataFrame([{"Skenario": n.split(" - ")[0], "Prediksi": h["Prediksi"],
                             "Probabilitas Layak": float(h["Probabilitas Layak"])} for n, h in per_model.items()]),
              "Perbandingan kedua skenario")
        catatan = enc.out_of_dataset(clean_data(row).iloc[0]) if hasattr(enc, "out_of_dataset") else []
        if catatan:
            st.warning("Input di luar dataset penelitian:\n\n" + "\n".join(f"- {c}" for c in catatan))
        else:
            st.caption("Seluruh nilai input berada dalam cakupan data training.")
        st.caption("Hasil model merupakan alat bantu keputusan. Penetapan penerima tetap melalui verifikasi kriteria "
                   "dan musyawarah desa.")
        st.session_state.setdefault("riwayat", []).append({
            "Waktu": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "Nama / keterangan": nama,
            **row.iloc[0].to_dict(),
            **{f"Prediksi {n.split(' - ')[0]}": h["Prediksi"] for n, h in per_model.items()},
            **{f"P(Layak) {n.split(' - ')[0]}": float(h["Probabilitas Layak"]) for n, h in per_model.items()},
            "Di luar dataset": "Ya" if catatan else "Tidak"})

    if st.session_state.get("riwayat"):
        st.divider()
        riwayat = pd.DataFrame(st.session_state["riwayat"])
        tabel(riwayat, f"Riwayat prediksi sesi ini ({len(riwayat)} data)")
        r1, r2 = st.columns(2)
        buf = io.BytesIO()
        riwayat.to_excel(buf, index=False)
        r1.download_button("Unduh riwayat (.xlsx)", buf.getvalue(), "riwayat_prediksi_blt_dd.xlsx")
        if r2.button("Hapus riwayat"):
            st.session_state["riwayat"] = []
            st.rerun()


# ---------------------------------------------------------------------------
# Halaman: Prediksi Massal
# ---------------------------------------------------------------------------
elif halaman == "Prediksi Massal":
    butuh_model()
    st.title("Prediksi Massal dari File Excel / CSV")
    info(f"File harus memuat {len(artifact['feature_cols'])} kolom atribut berikut dengan nama kolom sama persis seperti "
         "dataset penelitian. Bila file juga memuat kolom 7 kriteria BLT-DD, daftar prioritas diurutkan berdasarkan "
         "jumlah kriteria terpenuhi.")
    st.code(", ".join(artifact["feature_cols"]), language=None)
    buf_t = io.BytesIO()
    pd.DataFrame(columns=artifact["feature_cols"] + CRITERIA_COLS).to_excel(buf_t, index=False)
    st.download_button("Unduh template Excel", buf_t.getvalue(), "template_prediksi_blt_dd.xlsx")

    up = st.file_uploader("Unggah file data calon penerima", type=["xlsx", "csv"], key="batch")
    if up is not None:
        try:
            raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
            raw.columns = [str(c).strip() for c in raw.columns]
            kurang = [c for c in artifact["feature_cols"] if c not in raw.columns]
            if kurang:
                st.error(f"Kolom berikut tidak ditemukan: {kurang}")
            else:
                hasil = predict_df(artifact, raw, model_name)
                k1, k2, k3 = st.columns(3)
                k1.metric("Jumlah data", len(hasil))
                k2.metric("Prediksi Layak", int((hasil["Prediksi"] == "Layak").sum()))
                k3.metric("Prediksi Tidak Layak", int((hasil["Prediksi"] == "Tidak Layak").sum()))
                tabel(hasil.drop(columns=[c for c in ["NIK"] if c in hasil.columns]), "Hasil prediksi", tinggi=380)
                if TARGET in raw.columns:
                    m = evaluate(encode_target(raw[TARGET]), encode_target(hasil["Prediksi"]))
                    tabel(pd.DataFrame([m])[["TP", "TN", "FP", "FN"] + METRIK], "Evaluasi terhadap label pada file")
                buf = io.BytesIO()
                hasil.to_excel(buf, index=False)
                st.download_button("Unduh hasil prediksi (.xlsx)", buf.getvalue(), "hasil_prediksi_blt_dd.xlsx",
                                   type="primary")

                st.subheader("Daftar prioritas calon penerima")
                rank = ranking_prioritas(hasil)
                if rank.empty:
                    st.info("Tidak ada data yang diprediksi Layak.")
                else:
                    kuota = st.number_input("Kuota penerima BLT-DD", min_value=1, max_value=len(rank),
                                            value=min(20, len(rank)))
                    rank["Keterangan"] = np.where(rank["Peringkat"] <= kuota, "Prioritas (dalam kuota)", "Cadangan")
                    kolom = ["Peringkat", "Keterangan", "Jumlah Kriteria Terpenuhi", "Probabilitas Layak", "Usia"] + \
                            [c for c in ["Nama"] if c in rank.columns]
                    tabel(rank[kolom], tinggi=380)
                    st.caption("Urutan: jumlah kriteria terpenuhi, lalu probabilitas Layak, lalu usia. Urutan ini adalah "
                               "usulan pendukung keputusan; penetapan penerima tetap melalui musyawarah desa.")
                    buf_r = io.BytesIO()
                    rank.to_excel(buf_r, index=False)
                    st.download_button("Unduh daftar prioritas (.xlsx)", buf_r.getvalue(), "prioritas_penerima_blt_dd.xlsx")
        except Exception as e:
            st.error(f"File tidak dapat diproses: {e}")


# ---------------------------------------------------------------------------
# Halaman: Tentang
# ---------------------------------------------------------------------------
elif halaman == "Tentang":
    st.title("Tentang Aplikasi")
    st.markdown(f"""
Aplikasi ini merupakan implementasi penelitian **Klasifikasi Penerima Bantuan Langsung Tunai Dana Desa Menggunakan
Metode Naive Bayes (Studi Kasus: Kantor Desa Talulobutu, Kec. Tapa, Kab. Bone Bolango)**.

**Target:** Status Kelayakan Verifikasi (Layak / Tidak Layak).

**Atribut model final (Eksperimen 2):** {", ".join(FEATURE_COLS)}.

**Atribut yang dikeluarkan pada Eksperimen 2:** 7 atribut kriteria BLT-DD ({", ".join(CRITERIA_COLS)}), karena label
disusun dari aturan minimal satu kriteria terpenuhi.

**Skenario pengujian:** Skenario 1 Gaussian Naive Bayes (baseline) dan Skenario 2 Tomek Links + Class Weighting +
Gaussian Naive Bayes (usulan), dievaluasi dengan confusion matrix (accuracy, precision, recall, F1-score, specificity).

Hasil prediksi bersifat pendukung keputusan dan tidak menggantikan verifikasi resmi maupun musyawarah desa.
""")
