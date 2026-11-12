"""
blt_viz.py
Fungsi visualisasi bersama untuk notebook (gambar manuskrip) dan aplikasi Streamlit.
Setiap fungsi mengembalikan objek matplotlib Figure (dan tabel pendukung bila ada).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blt_dd import CRITERIA_COLS, TARGET

WARNA = {"Tidak Layak": "#4C72B0", "Layak": "#DD8452"}
WARNA_SKENARIO = ["#4C72B0", "#DD8452"]

STATUS_KELAYAKAN = "Status Kelayakan"
STATUS_PENERIMA = "Status Penerima BLT-DD"
KELOMPOK = ["Tidak Layak", "Layak, tidak menerima", "Penerima BLT-DD"]
WARNA_KELOMPOK = ["#4C72B0", "#DD8452", "#55A868"]

KRITERIA_SINGKAT = {
    "DTKS/Non DTKS": "Terdaftar DTKS",
    "Kepala Keluarga Miskin": "KK miskin",
    "perempuan kepala keluarga miskin": "Perempuan KK miskin",
    "Kepala keluarga tunggal Lanjut Usia": "KK tunggal lansia",
    "Kehilangan Mata Pencaharian": "Kehilangan mata pencaharian",
    "Anggota Keluarga yang sakit menahun/Kronis": "Anggota sakit kronis",
    "anggota keluarga disabilitas": "Anggota disabilitas",
}

METRIK = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity"]


# ---------------------------------------------------------------------------
# Data & kelompok
# ---------------------------------------------------------------------------
def kriteria_terpenuhi(df: pd.DataFrame) -> pd.DataFrame:
    """Matriks 0/1: kriteria BLT-DD bernilai Ya / DTKS."""
    return df[CRITERIA_COLS].isin(["Ya", "DTKS"]).astype(int)


def kelompok_seleksi(df: pd.DataFrame) -> pd.Series:
    """Tidak Layak / Layak, tidak menerima / Penerima BLT-DD."""
    g = np.select([df[STATUS_KELAYAKAN] == "Layak", df[TARGET] == "Layak"],
                  [KELOMPOK[2], KELOMPOK[1]], KELOMPOK[0])
    return pd.Series(g, index=df.index, name="Kelompok")


def angka_id(v, desimal: int = 2) -> str:
    """Format angka gaya Indonesia: 36.22 -> '36,22'."""
    return f"{v:,.{desimal}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def tabel_distribusi_kelas(df: pd.DataFrame) -> pd.DataFrame:
    dist = df[TARGET].value_counts().reindex(["Layak", "Tidak Layak"])
    out = pd.DataFrame({"Kelas": dist.index, "Jumlah KK": dist.values,
                        "Persentase (%)": (dist.values / dist.sum() * 100).round(2)})
    out.loc[len(out)] = ["Total", int(dist.sum()), 100.0]
    return out


def tabel_atribut_status(df: pd.DataFrame) -> pd.DataFrame:
    """Perbandingan tiga atribut status (Tabel 4.2 naskah)."""
    n = len(df)
    rows = []
    for kol, pos, ket in [(TARGET, "Layak", "Lolos verifikasi kriteria (target penelitian)"),
                          (STATUS_KELAYAKAN, "Layak", "Ditetapkan sebagai penerima"),
                          (STATUS_PENERIMA, "Ya", "Menerima BLT-DD")]:
        p = int((df[kol] == pos).sum())
        rows.append({"Atribut Status": kol,
                     "Layak / Ya": f"{p} ({angka_id(p / n * 100)}%)",
                     "Tidak Layak / Tidak": f"{n - p} ({angka_id((n - p) / n * 100)}%)",
                     "Rasio": f"1 : {angka_id((n - p) / p)}",
                     "Keterangan": ket})
    return pd.DataFrame(rows)


def tabel_aturan_kriteria(df: pd.DataFrame) -> pd.DataFrame:
    """Jumlah kriteria terpenuhi (0 vs >= 1) terhadap Status Kelayakan Verifikasi."""
    aturan = np.where(kriteria_terpenuhi(df).sum(axis=1) > 0, "Minimal 1 kriteria Ya/DTKS", "Tidak ada kriteria")
    ct = pd.crosstab(pd.Series(aturan, name="Aturan 7 kriteria"), df[TARGET].values).reindex(
        columns=["Layak", "Tidak Layak"], fill_value=0)
    ct.columns.name = None
    ct["Total"] = ct.sum(axis=1)
    ct.loc["Total"] = ct.sum()
    return ct.reset_index()


def tabel_profil_kelompok(df: pd.DataFrame) -> pd.DataFrame:
    krit = kriteria_terpenuhi(df)
    profil = pd.concat([
        krit.rename(columns=KRITERIA_SINGKAT),
        krit.sum(axis=1).rename("Jumlah kriteria terpenuhi"),
        df[["Usia", "Tanggungan Keluarga"]],
        (df["Menerima Bantuan Lain"] == "Ya").astype(int).rename("Menerima bantuan lain"),
    ], axis=1)
    out = profil.groupby(kelompok_seleksi(df)).mean().reindex(KELOMPOK).T.round(2)
    out.loc["Jumlah KK"] = kelompok_seleksi(df).value_counts().reindex(KELOMPOK).values
    out.index.name = "Atribut (rata-rata / proporsi)"
    out.columns.name = None
    return out.reset_index()


# ---------------------------------------------------------------------------
# Gambar eksplorasi data
# ---------------------------------------------------------------------------
def fig_distribusi_kelas(df: pd.DataFrame):
    dist = df[TARGET].value_counts().reindex(list(WARNA))
    pct = dist / dist.sum() * 100
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bars = ax.bar(dist.index, dist.values, color=list(WARNA.values()))
    ax.bar_label(bars, labels=[f"{v} ({p:.2f}%)" for v, p in zip(dist.values, pct.values)])
    ax.set_title(f"Distribusi Status Kelayakan Verifikasi\n(rasio {dist.max() / dist.min():.2f} : 1)")
    ax.set_ylabel("Jumlah KK")
    ax.set_ylim(0, dist.max() * 1.2)
    fig.tight_layout()
    return fig


def fig_alur_seleksi(df: pd.DataFrame):
    n = len(df)
    tahap = pd.Series({
        "Seluruh KK": n,
        "Lolos verifikasi kriteria\n(Status Kelayakan Verifikasi = Layak)": int((df[TARGET] == "Layak").sum()),
        "Penerima BLT-DD\n(Status Penerima = Ya)": int((df[STATUS_PENERIMA] == "Ya").sum()),
    })
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.barh(tahap.index[::-1], tahap.values[::-1], color=["#55A868", "#DD8452", "#4C72B0"])
    ax.bar_label(bars, labels=[f"{v} KK ({v / n:.1%})" for v in tahap.values[::-1]], padding=4)
    ax.set_xlim(0, n * 1.45)
    ax.set_xlabel("Jumlah KK")
    ax.set_title("Alur Seleksi Penerima BLT-DD Desa Talulobutu")
    fig.tight_layout()
    tabel = pd.crosstab([df[STATUS_KELAYAKAN], df[STATUS_PENERIMA]], df[TARGET],
                        margins=True, margins_name="Total")
    tabel.columns.name = TARGET
    return fig, tabel.reset_index()


def fig_kriteria(df: pd.DataFrame):
    krit = kriteria_terpenuhi(df)
    kel = kelompok_seleksi(df)
    per_krit = krit.groupby(kel).sum().T.reindex(columns=KELOMPOK).rename(index=KRITERIA_SINGKAT)
    n_krit = krit.sum(axis=1).rename("Jumlah kriteria terpenuhi")
    ct = pd.crosstab(n_krit, kel).reindex(columns=KELOMPOK, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    per_krit.drop(columns=KELOMPOK[0]).sort_values(KELOMPOK[2]).plot(
        kind="barh", stacked=True, color=WARNA_KELOMPOK[1:], ax=axes[0])
    axes[0].set_xlabel("Jumlah KK bernilai Ya / DTKS")
    axes[0].set_ylabel("")
    axes[0].set_title("Sebaran 7 Kriteria BLT-DD")
    ct.plot(kind="bar", stacked=True, color=WARNA_KELOMPOK, ax=axes[1], rot=0)
    axes[1].set_ylabel("Jumlah KK")
    axes[1].set_title("Jumlah Kriteria Terpenuhi per Kelompok")
    for ax in axes:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    ct = ct.assign(Total=ct.sum(axis=1))
    ct.loc["Total"] = ct.sum()
    ct.columns.name = None
    return fig, ct.reset_index()


def fig_atribut_numerik(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, c in zip(axes, ["Usia", "Tanggungan Keluarga"]):
        data = [df.loc[df[TARGET] == k, c].dropna() for k in WARNA]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        ax.set_xticks([1, 2], list(WARNA))
        for patch, w in zip(bp["boxes"], WARNA.values()):
            patch.set_facecolor(w)
            patch.set_alpha(0.7)
        ax.set_title(c)
        ax.set_ylabel(c)
    fig.tight_layout()
    return fig


def fig_atribut_kategorikal(df: pd.DataFrame):
    kolom = ["Jenis Kelamin", "Pekerjaan", "Pendidikan terakhir",
             "Status Pernikahan", "Status Tempat Tinggal", "Menerima Bantuan Lain"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, c in zip(axes.ravel(), kolom):
        ct = pd.crosstab(df[c], df[TARGET]).reindex(columns=list(WARNA), fill_value=0)
        ct = ct.loc[ct.sum(axis=1).sort_values().index]
        ct.plot(kind="barh", stacked=True, color=list(WARNA.values()), ax=ax, legend=False)
        ax.set_title(c)
        ax.set_ylabel("")
        ax.set_xlabel("Jumlah KK")
    h, lbl = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lbl, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ---------------------------------------------------------------------------
# Gambar proses pemodelan
# ---------------------------------------------------------------------------
def fig_pembagian_data(y_train, y_test):
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    r = pd.DataFrame({"Tidak Layak": [(y_train == 0).sum(), (y_test == 0).sum()],
                      "Layak": [(y_train == 1).sum(), (y_test == 1).sum()]},
                     index=[f"Training ({len(y_train)})", f"Testing ({len(y_test)})"])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    r.plot(kind="bar", stacked=True, color=list(WARNA.values()), ax=ax, rot=0)
    for cont in ax.containers:
        ax.bar_label(cont, label_type="center", color="white", fontsize=9)
    ax.set_ylabel("Jumlah KK")
    ax.set_title("Pembagian Data Training dan Testing (Stratified 80:20)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def fig_tomek_links(Xs_train, y_train, idx_hapus, pasangan, random_state: int = 42):
    """Visualisasi PCA 2 dimensi data training sebelum & sesudah Tomek Links."""
    from sklearn.decomposition import PCA

    y_train = np.asarray(y_train)
    pca = PCA(n_components=2, random_state=random_state).fit(Xs_train)
    Z = pca.transform(Xs_train)
    hapus = np.zeros(len(y_train), dtype=bool)
    hapus[np.asarray(idx_hapus, dtype=int)] = True

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True, sharey=True)
    for ax, judul, tampil in [(axes[0], "Sebelum Tomek Links", np.ones_like(hapus)),
                              (axes[1], "Sesudah Tomek Links", ~hapus)]:
        for k, nama in [(0, "Tidak Layak"), (1, "Layak")]:
            m = tampil & (y_train == k)
            ax.scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.6, color=WARNA[nama], label=f"{nama} ({m.sum()})")
        ax.set_title(judul)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} varians)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} varians)")
    for i, j in pasangan:
        axes[0].plot(Z[[i, j], 0], Z[[i, j], 1], color="red", lw=0.8)
    axes[0].scatter(Z[hapus, 0], Z[hapus, 1], s=110, facecolors="none", edgecolors="red",
                    linewidths=1.5, label=f"Dihapus ({hapus.sum()})")
    for ax in axes:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def fig_class_weighting(y_train, y_tomek, bobot: dict, prior_s1, prior_s2):
    y_train, y_tomek = np.asarray(y_train), np.asarray(y_tomek)
    kelas = ["Tidak Layak", "Layak"]
    jml = pd.DataFrame({"Sebelum\nTomek Links": [(y_train == 0).sum(), (y_train == 1).sum()],
                        "Sesudah\nTomek Links": [(y_tomek == 0).sum(), (y_tomek == 1).sum()]}, index=kelas)
    bb = pd.Series({"Tidak Layak": bobot[0], "Layak": bobot[1]})
    prior = pd.DataFrame({"Skenario 1\n(tanpa bobot)": prior_s1,
                          "Skenario 2\n(Tomek Links + CW)": prior_s2}, index=kelas)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    jml.T.plot(kind="bar", color=list(WARNA.values()), ax=axes[0], rot=0)
    axes[0].set_title("Jumlah Data Training")
    bb.plot(kind="bar", color=list(WARNA.values()), ax=axes[1], rot=0)
    axes[1].axhline(1, ls="--", color="gray", lw=1)
    axes[1].set_title("Bobot Kelas (Class Weighting)")
    prior.T.plot(kind="bar", color=list(WARNA.values()), ax=axes[2], rot=0)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Probabilitas Prior Kelas")
    for ax, fmt in zip(axes, ["%d", "%.3f", "%.3f"]):
        for cont in ax.containers:
            ax.bar_label(cont, fmt=fmt, fontsize=8)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        if ax.get_legend():
            ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, prior


def plot_cm(ax, h: dict, judul: str):
    cm = np.array([[h["TN"], h["FP"]], [h["FN"], h["TP"]]], dtype=int)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            lbl = [["TN", "FP"], ["FN", "TP"]][i][j]
            ax.text(j, i, f"{lbl}\n{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1], ["Tidak Layak", "Layak"])
    ax.set_yticks([0, 1], ["Tidak Layak", "Layak"])
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title(judul, fontsize=10)


def fig_confusion_pair(h1: dict, h2: dict, judul1: str = "Skenario 1 (Baseline)",
                       judul2: str = "Skenario 2 (Tomek Links + Class Weighting)"):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    plot_cm(axes[0], h1, judul1)
    plot_cm(axes[1], h2, judul2)
    fig.tight_layout()
    return fig


def fig_perbandingan_metrik(tabel: pd.DataFrame, judul: str = "Perbandingan Performa Model pada Data Testing"):
    """tabel: index = nama skenario, kolom memuat METRIK."""
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    tabel[METRIK].astype(float).T.plot(kind="bar", ax=ax, rot=0, color=WARNA_SKENARIO[:len(tabel)])
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Nilai")
    ax.set_title(judul)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.3f", fontsize=8)
    ax.legend(loc="upper center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def fig_probabilitas(y_test, proba_s1, proba_s2):
    y_test = np.asarray(y_test)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, p, judul in [(axes[0], proba_s1, "Skenario 1 (Baseline)"),
                         (axes[1], proba_s2, "Skenario 2 (Tomek Links + CW)")]:
        for k, nama in [(0, "Tidak Layak"), (1, "Layak")]:
            ax.hist(np.asarray(p)[y_test == k], bins=np.linspace(0, 1, 21), alpha=0.7,
                    color=WARNA[nama], label=f"Aktual {nama}")
        ax.axvline(0.5, ls="--", color="gray", lw=1)
        ax.set_title(judul)
        ax.set_xlabel("Probabilitas Layak")
    axes[0].set_ylabel("Jumlah KK (data testing)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def fig_rekap_eksperimen(rekap: pd.DataFrame):
    """Perbandingan metrik keempat kombinasi eksperimen x skenario."""
    d = rekap.copy()
    d["Label"] = d["Eksperimen"].str.replace(r"Eksperimen (\d) \((.*)\)", r"Eks. \1 (\2)", regex=True) + "\n" + d["Skenario"]
    met = ["Accuracy", "Precision", "Recall", "F1-Score"]
    warna = ["#9DB4D6", "#F2B98C", "#4C72B0", "#DD8452"]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(met))
    w = 0.2
    for i, (_, r) in enumerate(d.iterrows()):
        bars = ax.bar(x + (i - 1.5) * w, [r[m] for m in met], w, label=r["Label"].replace("\n", " - "), color=warna[i])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=2)
    ax.set_xticks(x, met)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("Nilai")
    ax.set_title("Perbandingan Hasil Eksperimen 1 (15 atribut) dan Eksperimen 2 (8 atribut)")
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="upper center")
    fig.tight_layout()
    return fig


def fig_tanpa_kriteria(h1: dict, h2: dict, tabel: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.7]})
    plot_cm(axes[0], h1, "Skenario 1 (8 atribut)")
    plot_cm(axes[1], h2, "Skenario 2 (8 atribut)")
    tabel[METRIK].astype(float).T.plot(kind="bar", ax=axes[2], rot=0, color=WARNA_SKENARIO)
    axes[2].set_ylim(0, 1.35)
    axes[2].set_title("Perbandingan Metrik (tanpa 7 atribut kriteria)")
    for cont in axes[2].containers:
        axes[2].bar_label(cont, fmt="%.3f", fontsize=7, rotation=90, padding=2)
    axes[2].legend(loc="upper center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig
