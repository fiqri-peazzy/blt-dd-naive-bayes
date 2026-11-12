"""
blt_dd.py
Modul bersama untuk penelitian:
"Klasifikasi Penerima BLT Dana Desa Menggunakan Metode Naive Bayes
 (Studi Kasus: Kantor Desa Talulobutu, Kec. Tapa, Kab. Bone Bolango)"

Isi modul:
- konstanta kolom dataset
- pembersihan data & seleksi atribut
- FeatureEncoder  : transformasi atribut kategorikal -> numerik
- tomek_links()   : undersampling Tomek Links (implementasi mandiri, setara
                    imblearn.under_sampling.TomekLinks(sampling_strategy="majority"))
- train_scenario(): melatih Gaussian Naive Bayes untuk Skenario 1 / Skenario 2
- evaluate()      : menghitung confusion matrix & metrik evaluasi

Modul ini dipakai oleh notebook (pemodelan) dan aplikasi Streamlit (prediksi),
sehingga proses transformasi data di keduanya selalu identik.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# 1. Konstanta kolom
# ---------------------------------------------------------------------------
TARGET = "Status Kelayakan Verifikasi"
POS_LABEL = "Layak"

# Atribut identitas / keputusan administratif yang TIDAK dipakai sebagai fitur
DROP_COLS = ["NIK", "Nama", "Alamat", "Status Kelayakan", "Status Penerima BLT-DD"]

NUMERIC_COLS = ["Usia", "Tanggungan Keluarga"]

# Atribut biner -> nilai yang dikodekan 1
BINARY_COLS = {
    "Jenis Kelamin": "P",                       # 1 = perempuan
    "Kehilangan Mata Pencaharian": "Ya",
    "Status Tempat Tinggal": "Menumpang",       # 1 = menumpang
    "DTKS/Non DTKS": "DTKS",                    # 1 = terdaftar DTKS
    "Menerima Bantuan Lain": "Ya",
    "Kepala Keluarga Miskin": "Ya",
    "Anggota Keluarga yang sakit menahun/Kronis": "Ya",
    "anggota keluarga disabilitas": "Ya",
    "Kepala keluarga tunggal Lanjut Usia": "Ya",
    "perempuan kepala keluarga miskin": "Ya",
}

# Pendidikan bersifat bertingkat -> ordinal
EDU_ORDER = {"Tidak Tamat SD": 0, "SD": 1, "SMP": 2, "SMA": 3,
             "Diploma": 4, "S1": 5, "S2": 6, "S3": 7}
EDU_ALIAS = {"D-1": "Diploma", "D-2": "Diploma", "D-3": "Diploma", "D-4": "Diploma",
             "D1": "Diploma", "D2": "Diploma", "D3": "Diploma", "D4": "Diploma"}

# Atribut nominal -> one-hot encoding
NOMINAL_COLS = ["Pekerjaan", "Status Pernikahan"]
EDU_COL = "Pendidikan terakhir"

# 15 atribut sosial-ekonomi pada dataset (urutan sesuai Tabel 3.1 versi awal)
FEATURE_COLS_15 = [
    "Usia", "Jenis Kelamin", "Pekerjaan", "Pendidikan terakhir", "Status Pernikahan",
    "Tanggungan Keluarga", "Kehilangan Mata Pencaharian", "Status Tempat Tinggal",
    "DTKS/Non DTKS", "Menerima Bantuan Lain", "Kepala Keluarga Miskin",
    "Anggota Keluarga yang sakit menahun/Kronis", "anggota keluarga disabilitas",
    "Kepala keluarga tunggal Lanjut Usia", "perempuan kepala keluarga miskin",
]

# 7 kriteria kelayakan BLT-DD. Label Status Kelayakan Verifikasi disusun dari aturan
# "minimal satu kriteria terpenuhi", sehingga ketujuh atribut ini TIDAK dipakai sebagai input
# (bila dipakai, model hanya meniru aturan dan kedua kelas terpisah sempurna).
CRITERIA_COLS = [
    "DTKS/Non DTKS", "Kepala Keluarga Miskin", "perempuan kepala keluarga miskin",
    "Kepala keluarga tunggal Lanjut Usia", "Kehilangan Mata Pencaharian",
    "Anggota Keluarga yang sakit menahun/Kronis", "anggota keluarga disabilitas",
]

# 8 atribut input model: kondisi sosial-ekonomi tanpa 7 atribut kriteria penentu label
FEATURE_COLS = [c for c in FEATURE_COLS_15 if c not in CRITERIA_COLS]
FEATURE_COLS_SOSEK = FEATURE_COLS   # alias lama

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 2. Pembersihan data
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Merapikan spasi, tipe data, dan kategori pendidikan."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype(str).str.strip()
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if EDU_COL in df.columns:
        df[EDU_COL] = df[EDU_COL].replace(EDU_ALIAS)
    return df


def check_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan pemeriksaan data: tipe, nilai kosong, jumlah kategori unik."""
    return pd.DataFrame({
        "Tipe": df.dtypes.astype(str),
        "Nilai Kosong": df.isna().sum(),
        "Jumlah Nilai Unik": df.nunique(),
    })


# ---------------------------------------------------------------------------
# 3. Transformasi data (encoding)
# ---------------------------------------------------------------------------
@dataclass
class FeatureEncoder:
    """Mengubah 15 atribut mentah menjadi matriks numerik.

    - Numerik      : Usia, Tanggungan Keluarga (apa adanya)
    - Biner        : Ya/Tidak dan sejenisnya -> 1/0
    - Ordinal      : Pendidikan terakhir (Tidak Tamat SD=0 ... S2=6)
    - One-hot      : Pekerjaan, Status Pernikahan
      Kategori pekerjaan yang sangat jarang (< min_freq) digabung menjadi "Lainnya".
    """
    min_freq: int = 5
    feature_cols: list = field(default_factory=lambda: list(FEATURE_COLS))
    categories_: dict = field(default_factory=dict)
    columns_: list = field(default_factory=list)
    ranges_: dict = field(default_factory=dict)       # rentang atribut numerik pada data training
    seen_values_: dict = field(default_factory=dict)  # nilai asli yang muncul pada data training

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        for c in NOMINAL_COLS:
            if c not in self.feature_cols:
                continue
            vc = df[c].value_counts()
            keep = sorted(vc[vc >= self.min_freq].index.tolist())
            self.categories_[c] = keep + ["Lainnya"]
        for c in self.feature_cols:
            if c in NUMERIC_COLS:
                v = pd.to_numeric(df[c], errors="coerce")
                self.ranges_[c] = (float(v.min()), float(v.max()))
            else:
                self.seen_values_[c] = sorted(df[c].astype(str).str.strip().replace(EDU_ALIAS).unique().tolist())
        self.columns_ = list(self._transform(df).columns)
        return self

    def out_of_dataset(self, row: pd.Series) -> list:
        """Daftar catatan untuk nilai input yang tidak pernah muncul / di luar rentang data training."""
        notes = []
        for c in self.feature_cols:
            v = row.get(c)
            if c in NUMERIC_COLS and c in getattr(self, "ranges_", {}):
                lo, hi = self.ranges_[c]
                if pd.notna(v) and not (lo <= float(v) <= hi):
                    notes.append(f"{c} = {v} berada di luar rentang data training ({lo:g}–{hi:g}); "
                                 "model melakukan ekstrapolasi.")
            elif c in NOMINAL_COLS:
                if str(v).strip() not in self.categories_.get(c, []):
                    notes.append(f"{c} = '{v}' tidak ada (atau jarang) pada data training; diperlakukan sebagai "
                                 "kategori 'Lainnya'.")
            elif c in getattr(self, "seen_values_", {}) and str(v).strip() not in self.seen_values_[c]:
                if c == EDU_COL and str(v).strip() in EDU_ORDER:
                    notes.append(f"{c} = '{v}' tidak pernah muncul pada data training; dikodekan secara ordinal "
                                 f"sebagai {EDU_ORDER[str(v).strip()]} (urutan jenjang pendidikan).")
                else:
                    notes.append(f"{c} = '{v}' tidak pernah muncul pada data training.")
        return notes

    def _group(self, s: pd.Series, col: str) -> pd.Series:
        known = set(self.categories_[col])
        return s.where(s.isin(known), "Lainnya")

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in self.feature_cols:
            if c in NUMERIC_COLS:
                out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
            elif c in BINARY_COLS:
                out[c] = (df[c].astype(str).str.strip() == BINARY_COLS[c]).astype(int)
            elif c == EDU_COL:
                out[c] = df[c].replace(EDU_ALIAS).map(EDU_ORDER).astype(float)
            elif c in NOMINAL_COLS:
                g = self._group(df[c].astype(str).str.strip(), c)
                for cat in self.categories_[c]:
                    out[f"{c}_{cat}"] = (g == cat).astype(int)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._transform(df)
        out = out.reindex(columns=self.columns_, fill_value=0)
        if out.isna().any().any():
            bad = out.columns[out.isna().any()].tolist()
            raise ValueError(f"Nilai tidak dikenali / kosong pada kolom: {bad}")
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def encode_target(y: pd.Series) -> np.ndarray:
    """Layak -> 1, Tidak Layak -> 0."""
    return (y.astype(str).str.strip() == POS_LABEL).astype(int).to_numpy()


# ---------------------------------------------------------------------------
# 4. Tomek Links
# ---------------------------------------------------------------------------
def tomek_links(X: np.ndarray, y: np.ndarray, majority_label: int | None = None):
    """Undersampling Tomek Links.

    Dua sampel (xi, xj) membentuk Tomek Link jika keduanya saling menjadi
    tetangga terdekat (jarak Euclidean) dan memiliki kelas berbeda.
    Sampel dari kelas mayoritas pada setiap pasangan dihapus.

    Return: X_res, y_res, index sampel yang dihapus, daftar pasangan Tomek Link
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if majority_label is None:
        vals, cnts = np.unique(y, return_counts=True)
        majority_label = vals[np.argmax(cnts)]

    nn = NearestNeighbors(n_neighbors=2).fit(X)
    _, idx = nn.kneighbors(X)
    nearest = idx[:, 1]                      # tetangga terdekat selain dirinya

    pairs, remove = [], set()
    for i, j in enumerate(nearest):
        if nearest[j] == i and y[i] != y[j] and i < j:   # mutual NN & beda kelas
            pairs.append((i, j))
            remove.add(i if y[i] == majority_label else j)

    remove = np.array(sorted(remove), dtype=int)
    mask = np.ones(len(y), dtype=bool)
    mask[remove] = False
    return X[mask], y[mask], remove, pairs


# ---------------------------------------------------------------------------
# 5. Pelatihan model
# ---------------------------------------------------------------------------
@dataclass
class ScenarioModel:
    """Model per skenario: scaler + Gaussian NB (+ info Tomek Links)."""
    name: str
    scaler: StandardScaler
    model: GaussianNB
    use_tomek: bool
    use_weight: bool
    n_train_before: int = 0
    n_train_after: int = 0
    n_removed: int = 0
    n_tomek_pairs: int = 0

    def predict(self, X) -> np.ndarray:
        return self.model.predict(self.scaler.transform(np.asarray(X, dtype=float)))

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(np.asarray(X, dtype=float)))


def train_scenario(X_train, y_train, use_tomek: bool, use_weight: bool,
                   name: str = "") -> ScenarioModel:
    """Skenario 1: use_tomek=False, use_weight=False (baseline)
       Skenario 2: use_tomek=True,  use_weight=True  (model usulan)

    Data distandardisasi agar jarak Euclidean pada Tomek Links tidak didominasi
    atribut berskala besar (mis. Usia). Gaussian NB sendiri tidak terpengaruh
    oleh standardisasi linear per atribut.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)

    n_before = len(y_train)
    removed, pairs = np.array([], dtype=int), []
    if use_tomek:
        Xs, y_train, removed, pairs = tomek_links(Xs, y_train)

    # Class Weighting: w_i = n_samples / (n_classes * n_samples_i)
    sw = compute_sample_weight("balanced", y_train) if use_weight else None
    gnb = GaussianNB().fit(Xs, y_train, sample_weight=sw)

    return ScenarioModel(name=name, scaler=scaler, model=gnb,
                         use_tomek=use_tomek, use_weight=use_weight,
                         n_train_before=n_before, n_train_after=len(y_train),
                         n_removed=len(removed), n_tomek_pairs=len(pairs))


# ---------------------------------------------------------------------------
# 6. Evaluasi
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred) -> dict:
    """Metrik dengan kelas positif = Layak (1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
    }


def class_weights(y) -> dict:
    """Bobot kelas balanced: n_samples / (n_classes * n_samples_i)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    return {int(c): len(y) / (len(classes) * n) for c, n in zip(classes, counts)}


# ---------------------------------------------------------------------------
# 7. Pipeline lengkap (dipakai tombol "Latih Ulang" di aplikasi Streamlit)
# ---------------------------------------------------------------------------
SCENARIO_NAMES = {
    "s1": "Skenario 1 - Gaussian NB (Baseline)",
    "s2": "Skenario 2 - Tomek Links + Class Weighting + Gaussian NB",
}


def build_artifact(df_raw: pd.DataFrame, test_size: float = 0.2,
                   random_state: int = RANDOM_STATE) -> dict:
    """Menjalankan alur penelitian (Gambar 3.1) dan mengembalikan artefak model.

    Alur: pembersihan -> seleksi atribut -> split 80:20 (stratified)
          -> encoding (fit pada data training) -> Skenario 1 & 2 -> evaluasi.
    """
    from datetime import datetime

    import sklearn
    from sklearn.model_selection import train_test_split

    df = clean_data(df_raw)
    X = df[FEATURE_COLS]
    y = encode_target(df[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    encoder = FeatureEncoder().fit(X_train)
    Xtr = encoder.transform(X_train).to_numpy(dtype=float)
    Xte = encoder.transform(X_test).to_numpy(dtype=float)

    models = {
        SCENARIO_NAMES["s1"]: train_scenario(Xtr, y_train, False, False, SCENARIO_NAMES["s1"]),
        SCENARIO_NAMES["s2"]: train_scenario(Xtr, y_train, True, True, SCENARIO_NAMES["s2"]),
    }
    metrics = {name: evaluate(y_test, m.predict(Xte)) for name, m in models.items()}

    # Pembanding cadangan: skenario yang sama bila 7 atribut kriteria ikut sebagai input (15 atribut)
    X15_train, X15_test = df.loc[X_train.index, FEATURE_COLS_15], df.loc[X_test.index, FEATURE_COLS_15]
    enc15 = FeatureEncoder(feature_cols=list(FEATURE_COLS_15)).fit(X15_train)
    Xtr15 = enc15.transform(X15_train).to_numpy(dtype=float)
    Xte15 = enc15.transform(X15_test).to_numpy(dtype=float)
    models15 = {
        SCENARIO_NAMES["s1"]: train_scenario(Xtr15, y_train, False, False, SCENARIO_NAMES["s1"]),
        SCENARIO_NAMES["s2"]: train_scenario(Xtr15, y_train, True, True, SCENARIO_NAMES["s2"]),
    }
    analisis_15 = {
        "feature_cols": list(FEATURE_COLS_15),
        "encoder": enc15,
        "models": models15,
        "metrics": {name: evaluate(y_test, m.predict(Xte15)) for name, m in models15.items()},
    }

    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sklearn_version": sklearn.__version__,
        "encoder": encoder,
        "models": models,
        "metrics": metrics,
        "n_data": int(len(df)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_counts": {"Layak": int(y.sum()), "Tidak Layak": int(len(y) - y.sum())},
        "feature_cols": list(FEATURE_COLS),
        "default_model": SCENARIO_NAMES["s2"],
        "analisis_15": analisis_15,
    }
