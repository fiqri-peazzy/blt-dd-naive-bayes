"""
blt_steps.py
Tabel per tahapan klasifikasi Gaussian Naive Bayes (format dokumentasi Bab IV & Bab V).
Dipakai bersama oleh notebook dan aplikasi Streamlit (tab "Tahapan Proses") agar isinya identik.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from blt_dd import (BINARY_COLS, CRITERIA_COLS, DROP_COLS, EDU_COL, EDU_ORDER, FEATURE_COLS, NOMINAL_COLS,
                    NUMERIC_COLS, RANDOM_STATE, TARGET, FeatureEncoder, clean_data, encode_target, evaluate,
                    tomek_links, train_scenario)

KELAS = {0: "Tidak Layak", 1: "Layak"}


def _id(v, d=4):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------------------
# 4.1 Pengumpulan data
# ---------------------------------------------------------------------------
def cuplikan_dataset(df_raw: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Cuplikan dataset dengan NIK & Nama disamarkan."""
    d = df_raw.head(n).copy()
    if "NIK" in d:
        d["NIK"] = d["NIK"].astype(str).str[:6] + "**********"
    if "Nama" in d:
        d["Nama"] = d["Nama"].astype(str).str[0] + "***"
    d.insert(0, "No", range(1, len(d) + 1))
    return d


# ---------------------------------------------------------------------------
# 4.2.1 Preprocessing
# ---------------------------------------------------------------------------
def tabel_pembersihan(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = clean_data(df_raw)
    kosong = int(df.isna().any(axis=1).sum())
    dup_nik = int(df["NIK"].duplicated().sum()) if "NIK" in df else 0
    dup = int(df.duplicated().sum())
    akhir = len(df.dropna().drop_duplicates())
    return pd.DataFrame([
        {"Pemeriksaan": "Jumlah data awal", "Hasil": len(df_raw), "Tindakan": "-"},
        {"Pemeriksaan": "Baris dengan nilai kosong (missing value)", "Hasil": kosong,
         "Tindakan": "Dihapus" if kosong else "Tidak ada yang dihapus"},
        {"Pemeriksaan": "NIK ganda", "Hasil": dup_nik, "Tindakan": "Dihapus" if dup_nik else "Tidak ada yang dihapus"},
        {"Pemeriksaan": "Baris duplikat", "Hasil": dup, "Tindakan": "Dihapus" if dup else "Tidak ada yang dihapus"},
        {"Pemeriksaan": "Penyeragaman teks (spasi, penulisan Diploma)", "Hasil": "-", "Tindakan": "Dirapikan"},
        {"Pemeriksaan": "Jumlah data akhir", "Hasil": akhir, "Tindakan": "-"},
    ])


def tabel_seleksi_atribut(feature_cols=None) -> pd.DataFrame:
    """Daftar atribut dikeluarkan / input / target untuk satu eksperimen."""
    feature_cols = list(feature_cols or FEATURE_COLS)
    alasan = {"NIK": "Identitas, tidak bersifat prediktif", "Nama": "Identitas, tidak bersifat prediktif",
              "Alamat": "Identitas, tidak bersifat prediktif",
              "Status Kelayakan": "Keputusan administratif penerima akhir (berpotensi kebocoran data)",
              "Status Penerima BLT-DD": "Keputusan administratif penerima akhir (berpotensi kebocoran data)"}
    rows = [{"Atribut": c, "Keterangan": "Dikeluarkan", "Alasan": alasan[c]} for c in DROP_COLS]
    rows += [{"Atribut": c, "Keterangan": "Dikeluarkan",
              "Alasan": "Kriteria BLT-DD penentu label (minimal 1 kriteria = Layak)"}
             for c in CRITERIA_COLS if c not in feature_cols]
    rows += [{"Atribut": c, "Keterangan": "Atribut input",
              "Alasan": "Kriteria BLT-DD" if c in CRITERIA_COLS else "Kondisi sosial-ekonomi"} for c in feature_cols]
    rows += [{"Atribut": TARGET, "Keterangan": "Target (label)", "Alasan": "Layak / Tidak Layak"}]
    return pd.DataFrame(rows)


def cuplikan_seleksi(df: pd.DataFrame, n: int = 10, feature_cols=None) -> pd.DataFrame:
    d = df[list(feature_cols or FEATURE_COLS) + [TARGET]].head(n).copy()
    d.insert(0, "No", range(1, len(d) + 1))
    return d


def tabel_aturan_transformasi(encoder: FeatureEncoder) -> pd.DataFrame:
    rows = []
    for c in encoder.feature_cols:
        if c in NUMERIC_COLS:
            rows.append({"Atribut": c, "Jenis": "Numerik", "Nilai asli": "angka", "Hasil transformasi": "nilai tetap"})
        elif c in BINARY_COLS:
            nilai = encoder.seen_values_.get(c, []) if hasattr(encoder, "seen_values_") else []
            lain = [v for v in nilai if v != BINARY_COLS[c]]
            rows.append({"Atribut": c, "Jenis": "Biner",
                         "Nilai asli": f"{BINARY_COLS[c]} / {', '.join(lain) or 'lainnya'}",
                         "Hasil transformasi": f"{BINARY_COLS[c]} = 1, lainnya = 0"})
        elif c == EDU_COL:
            rows.append({"Atribut": c, "Jenis": "Ordinal", "Nilai asli": ", ".join(k for k in EDU_ORDER if k != "S3"),
                         "Hasil transformasi": ", ".join(f"{k}={v}" for k, v in EDU_ORDER.items() if k != "S3")})
        elif c in NOMINAL_COLS:
            cats = encoder.categories_[c]
            rows.append({"Atribut": c, "Jenis": "Nominal (one-hot)", "Nilai asli": ", ".join(cats),
                         "Hasil transformasi": f"{len(cats)} kolom biner ({c}_<kategori>); kategori < 5 data = Lainnya"})
    return pd.DataFrame(rows)


def cuplikan_transformasi(X_enc: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    d = X_enc.head(n).copy()
    d.insert(0, "No", range(1, len(d) + 1))
    return d


def tabel_pembagian(y_train, y_test) -> pd.DataFrame:
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    rows = []
    for nama, yy in [("Data training (80%)", y_train), ("Data testing (20%)", y_test)]:
        rows.append({"Data": nama, "Layak": int((yy == 1).sum()), "Tidak Layak": int((yy == 0).sum()), "Jumlah": len(yy)})
    rows.append({"Data": "Total", "Layak": rows[0]["Layak"] + rows[1]["Layak"],
                 "Tidak Layak": rows[0]["Tidak Layak"] + rows[1]["Tidak Layak"], "Jumlah": len(y_train) + len(y_test)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4.2.2 - 4.2.3 Penerapan Gaussian Naive Bayes (Skenario 1 & 2)
# ---------------------------------------------------------------------------
def tabel_prior(y, sample_weight=None) -> pd.DataFrame:
    y = np.asarray(y)
    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    rows = []
    for k in (0, 1):
        n_k, w_k = int((y == k).sum()), float(w[y == k].sum())
        rows.append({"Kelas": KELAS[k], "Jumlah data": n_k,
                     "Bobot per data": float(w[y == k][0]) if n_k else 0.0,
                     "Total bobot": w_k, "Prior P(C)": w_k / w.sum()})
    return pd.DataFrame(rows)


def tabel_parameter_gaussian(X: pd.DataFrame, y, sample_weight=None) -> pd.DataFrame:
    """Rata-rata (mu) dan varians (sigma^2) setiap atribut per kelas (berbobot bila ada sample weight)."""
    y = np.asarray(y)
    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    Xv = X.to_numpy(dtype=float)
    out = pd.DataFrame({"Atribut": X.columns})
    for k in (0, 1):
        m = y == k
        mu = np.average(Xv[m], axis=0, weights=w[m])
        var = np.average((Xv[m] - mu) ** 2, axis=0, weights=w[m])
        out[f"Mean {KELAS[k]}"] = mu
        out[f"Varians {KELAS[k]}"] = var
    return out


def tabel_pasangan_tomek(Xs_train, y_train, pasangan, nomor_data) -> pd.DataFrame:
    """Daftar pasangan Tomek Link. nomor_data: nomor baris asli dataset untuk setiap data training."""
    y_train = np.asarray(y_train)
    nomor_data = np.asarray(nomor_data)
    rows = []
    for no, (i, j) in enumerate(pasangan, 1):
        maj, mino = (i, j) if y_train[i] == 0 else (j, i)
        rows.append({"Pasangan": no, "No. data Tidak Layak": int(nomor_data[maj]),
                     "No. data Layak": int(nomor_data[mino]),
                     "Jarak Euclidean": float(np.linalg.norm(np.asarray(Xs_train)[i] - np.asarray(Xs_train)[j])),
                     "Data dihapus": f"No. {int(nomor_data[maj])} (Tidak Layak)"})
    return pd.DataFrame(rows)


def tabel_prediksi(model, X_test, y_test, nomor_data, n: int | None = 10) -> pd.DataFrame:
    proba = model.predict_proba(X_test)
    pred = proba.argmax(axis=1)
    d = pd.DataFrame({"No. data": np.asarray(nomor_data, dtype=int),
                      "Aktual": [KELAS[int(v)] for v in y_test],
                      "P(Tidak Layak | X)": proba[:, 0], "P(Layak | X)": proba[:, 1],
                      "Prediksi": [KELAS[int(v)] for v in pred]})
    d["Keterangan"] = np.where(d["Aktual"] == d["Prediksi"], "Benar", "Salah")
    return d if n is None else d.head(n)


# ---------------------------------------------------------------------------
# 4.2.4 Evaluasi
# ---------------------------------------------------------------------------
def tabel_confusion(h: dict) -> pd.DataFrame:
    return pd.DataFrame({"Aktual \\ Prediksi": ["Tidak Layak", "Layak"],
                         "Prediksi Tidak Layak": [h["TN"], h["FN"]], "Prediksi Layak": [h["FP"], h["TP"]]})


def tabel_perhitungan_metrik(h: dict) -> pd.DataFrame:
    TP, TN, FP, FN = (int(h[k]) for k in ("TP", "TN", "FP", "FN"))
    P, R = h["Precision"], h["Recall"]
    return pd.DataFrame([
        {"Metrik": "Accuracy", "Rumus": "(TP + TN) / (TP + TN + FP + FN)",
         "Substitusi": f"({TP} + {TN}) / ({TP} + {TN} + {FP} + {FN})", "Hasil": h["Accuracy"]},
        {"Metrik": "Precision", "Rumus": "TP / (TP + FP)", "Substitusi": f"{TP} / ({TP} + {FP})", "Hasil": h["Precision"]},
        {"Metrik": "Recall", "Rumus": "TP / (TP + FN)", "Substitusi": f"{TP} / ({TP} + {FN})", "Hasil": h["Recall"]},
        {"Metrik": "F1-Score", "Rumus": "2 × (Precision × Recall) / (Precision + Recall)",
         "Substitusi": f"2 × ({_id(P)} × {_id(R)}) / ({_id(P)} + {_id(R)})", "Hasil": h["F1-Score"]},
        {"Metrik": "Specificity", "Rumus": "TN / (TN + FP)", "Substitusi": f"{TN} / ({TN} + {FP})",
         "Hasil": h["Specificity"]},
    ])


# ---------------------------------------------------------------------------
# 4.2.5 Representasi pengetahuan
# ---------------------------------------------------------------------------
def tabel_proporsi_atribut(X_enc: pd.DataFrame, y, top: int | None = 15) -> pd.DataFrame:
    """Rata-rata / proporsi setiap atribut hasil transformasi per kelas, diurutkan dari selisih terbesar."""
    y = np.asarray(y)
    d = pd.DataFrame({"Atribut": X_enc.columns,
                      "Tidak Layak": X_enc[y == 0].mean().values, "Layak": X_enc[y == 1].mean().values})
    d["Jenis"] = ["Numerik (rata-rata)" if c in NUMERIC_COLS or c == EDU_COL else "Proporsi bernilai 1"
                  for c in d["Atribut"]]
    d["Selisih (Layak - Tidak Layak)"] = d["Layak"] - d["Tidak Layak"]
    biner = d[d["Jenis"] == "Proporsi bernilai 1"].copy()
    biner = biner.reindex(biner["Selisih (Layak - Tidak Layak)"].abs().sort_values(ascending=False).index)
    out = pd.concat([biner, d[d["Jenis"] != "Proporsi bernilai 1"]])
    out = out[["Atribut", "Jenis", "Tidak Layak", "Layak", "Selisih (Layak - Tidak Layak)"]]
    return out if top is None else out.head(top)


# ---------------------------------------------------------------------------
# Bab V: pengujian & validasi eksternal
# ---------------------------------------------------------------------------
def pengujian_rasio_split(df: pd.DataFrame, test_sizes=(0.30, 0.25, 0.20, 0.15, 0.10),
                          random_state: int = RANDOM_STATE, feature_cols=None) -> pd.DataFrame:
    feature_cols = list(feature_cols or FEATURE_COLS)
    X, y = df[feature_cols], encode_target(df[TARGET])
    rows = []
    for ts in test_sizes:
        a, b, ytr, yte = train_test_split(X, y, test_size=ts, stratify=y, random_state=random_state)
        enc = FeatureEncoder(feature_cols=feature_cols).fit(a)
        Xa, Xb = enc.transform(a), enc.transform(b)
        for nama, (t, w) in {"Skenario 1": (False, False), "Skenario 2": (True, True)}.items():
            m = train_scenario(Xa, ytr, t, w)
            h = evaluate(yte, m.predict(Xb))
            rows.append({"Rasio (training : testing)": f"{round((1 - ts) * 100)} : {round(ts * 100)}",
                         "Data training": len(ytr), "Data testing": len(yte), "Skenario": nama,
                         "Data dihapus Tomek": m.n_removed, **{k: h[k] for k in ("TP", "TN", "FP", "FN")},
                         **{k: h[k] for k in ("Accuracy", "Precision", "Recall", "F1-Score")}})
    return pd.DataFrame(rows)


def ekspor_rapidminer(X_train_raw, X_test_raw, y_train, y_test, encoder: FeatureEncoder, out_dir) -> list:
    """Ekspor data training/testing (asli & numerik hasil transformasi) untuk validasi di RapidMiner."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lab = lambda yy: [KELAS[int(v)] for v in yy]
    files = []
    for nama, Xr, yy in [("training", X_train_raw, y_train), ("testing", X_test_raw, y_test)]:
        asli = Xr.copy()
        asli[TARGET] = lab(yy)
        num = encoder.transform(Xr).copy()
        num[TARGET] = lab(yy)
        for jenis, d in [("asli", asli), ("numerik", num)]:
            p = out_dir / f"data_{nama}_{jenis}.xlsx"
            d.to_excel(p, index=False)
            files.append(p)
    return files


def standardisasi(X_train) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(X_train, dtype=float))


# ---------------------------------------------------------------------------
# Satu eksperimen lengkap (dipakai notebook, Streamlit, dan pembuatan naskah)
# ---------------------------------------------------------------------------
def jalankan_eksperimen(df: pd.DataFrame, feature_cols, test_size: float = 0.2,
                        random_state: int = RANDOM_STATE) -> dict:
    """Menjalankan seluruh tahapan untuk satu himpunan atribut: pembagian data, transformasi,
    Skenario 1 (Gaussian NB) dan Skenario 2 (Tomek Links + Class Weighting + Gaussian NB), evaluasi."""
    from sklearn.utils.class_weight import compute_sample_weight

    from blt_dd import class_weights

    feature_cols = list(feature_cols)
    X, y = df[feature_cols], encode_target(df[TARGET])
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y,
                                                  random_state=random_state)
    no_tr, no_te = Xtr_raw.index.to_numpy() + 1, Xte_raw.index.to_numpy() + 1
    enc = FeatureEncoder(feature_cols=feature_cols).fit(Xtr_raw)
    Xtr, Xte = enc.transform(Xtr_raw), enc.transform(Xte_raw)

    s1 = train_scenario(Xtr, ytr, False, False)
    s2 = train_scenario(Xtr, ytr, True, True)
    Xs = standardisasi(Xtr)
    _, y_tl, idx_hapus, pasangan = tomek_links(Xs, ytr)
    keep = np.ones(len(ytr), dtype=bool)
    keep[idx_hapus] = False
    sw = compute_sample_weight("balanced", y_tl)
    h1, h2 = evaluate(yte, s1.predict(Xte)), evaluate(yte, s2.predict(Xte))

    hasil_tl = pd.DataFrame({
        "Kelas": ["Tidak Layak", "Layak", "Total"],
        "Sebelum Tomek Links": [int((ytr == 0).sum()), int((ytr == 1).sum()), len(ytr)],
        "Sesudah Tomek Links": [int((y_tl == 0).sum()), int((y_tl == 1).sum()), len(y_tl)],
    })
    hasil_tl["Dihapus"] = hasil_tl["Sebelum Tomek Links"] - hasil_tl["Sesudah Tomek Links"]
    perb = pd.DataFrame({"Skenario 1 (Baseline)": h1, "Skenario 2 (Tomek Links + CW)": h2}).T
    tbl_perb = perb[["TP", "TN", "FP", "FN"]].astype(int).join(
        perb[["Accuracy", "Precision", "Recall", "F1-Score", "Specificity"]].astype(float)).reset_index(names="Skenario")

    return {
        "feature_cols": feature_cols, "encoder": enc,
        "X_train_raw": Xtr_raw, "X_test_raw": Xte_raw, "y_train": ytr, "y_test": yte,
        "no_train": no_tr, "no_test": no_te, "X_train": Xtr, "X_test": Xte,
        "s1": s1, "s2": s2, "hasil_s1": h1, "hasil_s2": h2,
        "Xs": Xs, "y_tl": y_tl, "idx_hapus": idx_hapus, "pasangan": pasangan, "keep": keep, "sw": sw,
        "bobot": class_weights(y_tl), "prior1": s1.model.class_prior_, "prior2": s2.model.class_prior_,
        "proba1": s1.predict_proba(Xte)[:, 1], "proba2": s2.predict_proba(Xte)[:, 1],
        "t_seleksi": tabel_seleksi_atribut(feature_cols),
        "t_cuplikan_seleksi": cuplikan_seleksi(df, feature_cols=feature_cols),
        "t_pembagian": tabel_pembagian(ytr, yte),
        "t_aturan": tabel_aturan_transformasi(enc),
        "t_cuplikan_transformasi": cuplikan_transformasi(Xtr),
        "t_prior1": tabel_prior(ytr),
        "t_param1": tabel_parameter_gaussian(Xtr, ytr),
        "t_pred1": tabel_prediksi(s1, Xte, yte, no_te, n=None),
        "t_tomek": tabel_pasangan_tomek(Xs, ytr, pasangan, no_tr),
        "t_hasil_tomek": hasil_tl,
        "t_prior2": tabel_prior(y_tl, sw),
        "t_param2": tabel_parameter_gaussian(Xtr[keep], y_tl, sw),
        "t_pred2": tabel_prediksi(s2, Xte, yte, no_te, n=None),
        "t_cm1": tabel_confusion(h1), "t_cm2": tabel_confusion(h2),
        "t_metrik1": tabel_perhitungan_metrik(h1), "t_metrik2": tabel_perhitungan_metrik(h2),
        "t_perbandingan": tbl_perb,
        "t_proporsi": tabel_proporsi_atribut(Xtr, ytr),
    }


def perhitungan_manual(E: dict, urutan: int | None = None) -> dict:
    """Perhitungan manual Gaussian NB (Skenario 1) untuk satu data testing.
    Bila urutan None, dipilih data testing pertama yang diprediksi Layak dengan benar."""
    from sklearn.naive_bayes import GaussianNB

    Xtr, ytr, Xte, yte = E["X_train"], E["y_train"], E["X_test"], E["y_test"]
    gnb = GaussianNB().fit(Xtr, ytr)
    if urutan is None:
        pred = gnb.predict(Xte)
        cocok = np.where((pred == 1) & (yte == 1))[0]
        urutan = int(cocok[0]) if len(cocok) else 0
    x = Xte.iloc[urutan].to_numpy(dtype=float)
    baris, log_post = [], {}
    for k, nama in KELAS.items():
        mu, var = gnb.theta_[k], gnb.var_[k]
        ll = -0.5 * np.log(2 * np.pi * var) - (x - mu) ** 2 / (2 * var)
        log_post[nama] = np.log(gnb.class_prior_[k]) + ll.sum()
        for f, xv, m_, v_, l_ in zip(Xtr.columns, x, mu, var, ll):
            baris.append({"Atribut": f, "Nilai x": xv, "Kelas": nama, "Mean": m_, "Varians": v_,
                          "P(x|C)": np.exp(l_), "ln P(x|C)": l_})
    d = pd.DataFrame(baris)
    lik = d.pivot_table(index=["Atribut", "Nilai x"], columns="Kelas",
                        values=["Mean", "Varians", "P(x|C)", "ln P(x|C)"], sort=False)
    cols = [(v, k) for k in ("Tidak Layak", "Layak") for v in ("Mean", "Varians", "P(x|C)", "ln P(x|C)")]
    lik = lik[cols]
    lik.columns = [f"{v} ({'TL' if k == 'Tidak Layak' else 'L'})" for v, k in cols]
    lik = lik.reset_index()
    lp = np.array(list(log_post.values()))
    post = np.exp(lp - lp.max())
    post /= post.sum()
    validasi = pd.DataFrame({
        "Kelas": list(log_post.keys()), "ln P(C)": np.log(gnb.class_prior_),
        "Σ ln P(x|C)": lp - np.log(gnb.class_prior_), "ln P(C) + Σ ln P(x|C)": lp,
        "Posterior (manual)": post, "Posterior (sistem)": E["s1"].predict_proba(Xte.iloc[[urutan]])[0]})
    return {"urutan": urutan, "no_data": int(E["no_test"][urutan]), "aktual": KELAS[int(yte[urutan])],
            "prediksi": validasi.loc[validasi["Posterior (manual)"].idxmax(), "Kelas"],
            "nilai_asli": E["X_test_raw"].iloc[[urutan]].T.reset_index().set_axis(["Atribut", "Nilai"], axis=1),
            "t_likelihood": lik, "t_validasi": validasi}


def evaluasi_aturan_if(df: pd.DataFrame, E: dict) -> pd.DataFrame:
    """Akurasi aturan IF-ELSE 'minimal 1 kriteria Ya/DTKS -> Layak' pada data testing yang sama."""
    uji = df.loc[E["X_test_raw"].index, CRITERIA_COLS]
    pred = uji.isin(["Ya", "DTKS"]).any(axis=1).astype(int).to_numpy()
    h = evaluate(E["y_test"], pred)
    return pd.DataFrame([
        {"Metode": "Aturan IF-ELSE (minimal 1 kriteria = Layak)", **{k: h[k] for k in ("TP", "TN", "FP", "FN")},
         **{k: h[k] for k in ("Accuracy", "Precision", "Recall", "F1-Score")}},
        {"Metode": "Gaussian NB Skenario 1 (15 atribut)", **{k: E["hasil_s1"][k] for k in ("TP", "TN", "FP", "FN")},
         **{k: E["hasil_s1"][k] for k in ("Accuracy", "Precision", "Recall", "F1-Score")}},
        {"Metode": "Gaussian NB Skenario 2 (15 atribut)", **{k: E["hasil_s2"][k] for k in ("TP", "TN", "FP", "FN")},
         **{k: E["hasil_s2"][k] for k in ("Accuracy", "Precision", "Recall", "F1-Score")}},
    ])


def rekap_eksperimen(E15: dict, E8: dict) -> pd.DataFrame:
    rows = []
    for nama, E in [("Eksperimen 1 (15 atribut)", E15), ("Eksperimen 2 (8 atribut)", E8)]:
        for sk, h in [("Skenario 1", E["hasil_s1"]), ("Skenario 2", E["hasil_s2"])]:
            rows.append({"Eksperimen": nama, "Skenario": sk, **{k: int(h[k]) for k in ("TP", "TN", "FP", "FN")},
                         **{k: float(h[k]) for k in ("Accuracy", "Precision", "Recall", "F1-Score", "Specificity")}})
    return pd.DataFrame(rows)
