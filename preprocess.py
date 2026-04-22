"""
Preprocessing script: unifies MIT-BIH, PTB-DB, and PhysioNet 2017 ECG datasets
into a single HDF5 file with three-class labels:
  0 = normal
  1 = abnormal
  2 = noisy / unknown

Signal lengths:
  MIT-BIH and PTB-DB: 187 samples (kept as-is)
  PhysioNet 2017:     2000 samples → resampled to 187 via scipy

Output: processed-data/ecg_unified.h5
  Datasets: signals (N, 187), labels (N,), source (N,)
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import resample
import h5py

RAW = "raw-data"
OUT_DIR = "build"
OUT_FILE = os.path.join(OUT_DIR, "ecg_unified.h5")
TARGET_LEN = 187

# MIT-BIH label mapping
# 0 → normal, 1/2/3 → abnormal, 4 → noisy/unknown
MITBIH_MAP = {0.0: 0, 1.0: 1, 2.0: 1, 3.0: 1, 4.0: 2}

# PhysioNet 2017 label mapping
# 0 → normal, 1 (AF) / 2 (other) → abnormal, 3 (noisy) → noisy/unknown
PHYSIONET_MAP = {0: 0, 1: 1, 2: 1, 3: 2}

# PTB-DB: normal file → 0, abnormal file → 1 (already clean, no noisy class)


def load_mitbih():
    train = pd.read_csv(os.path.join(RAW, "mitbih", "mitbih_train.csv"), header=None)
    test = pd.read_csv(os.path.join(RAW, "mitbih", "mitbih_test.csv"), header=None)
    df = pd.concat([train, test], ignore_index=True)

    signals = df.iloc[:, :TARGET_LEN].values.astype(np.float32)
    labels = df.iloc[:, -1].map(MITBIH_MAP).values.astype(np.int8)
    sources = np.full(len(df), "mitbih", dtype=object)

    print(f"MIT-BIH:      {len(df):>7,} records")
    return signals, labels, sources


def load_ptbdb():
    normal = pd.read_csv(os.path.join(RAW, "ptbdb", "ptbdb_normal.csv"), header=None)
    abnormal = pd.read_csv(os.path.join(RAW, "ptbdb", "ptbdb_abnormal.csv"), header=None)
    df = pd.concat([normal, abnormal], ignore_index=True)

    signals = df.iloc[:, :TARGET_LEN].values.astype(np.float32)
    labels = df.iloc[:, -1].values.astype(np.int8)
    sources = np.full(len(df), "ptbdb", dtype=object)

    print(f"PTB-DB:       {len(df):>7,} records")
    return signals, labels, sources


def load_physionet2017():
    df = pd.read_csv(os.path.join(RAW, "physionet2017", "physionet2017.csv"))

    # Drop non-signal columns; signal columns are named '0' through '1999'
    signal_cols = [str(i) for i in range(2000)]
    raw_signals = df[signal_cols].values.astype(np.float32)

    # Resample each record from 2000 → TARGET_LEN samples
    signals = resample(raw_signals, TARGET_LEN, axis=1).astype(np.float32)

    labels = df["label"].map(PHYSIONET_MAP).values.astype(np.int8)
    sources = np.full(len(df), "physionet2017", dtype=object)

    print(f"PhysioNet 17: {len(df):>7,} records (resampled 2000 → {TARGET_LEN})")
    return signals, labels, sources


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_signals, all_labels, all_sources = [], [], []
    for loader in [load_mitbih, load_ptbdb, load_physionet2017]:
        s, l, src = loader()
        all_signals.append(s)
        all_labels.append(l)
        all_sources.append(src)

    signals = np.concatenate(all_signals, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    sources = np.concatenate(all_sources, axis=0)

    print(f"\nUnified:      {len(signals):>7,} records  shape={signals.shape}")
    counts = {0: "normal", 1: "abnormal", 2: "noisy/unknown"}
    for cls, name in counts.items():
        n = (labels == cls).sum()
        print(f"  {name:<15} {n:>7,}  ({100*n/len(labels):.1f}%)")

    # Shuffle before saving
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(signals))
    signals, labels, sources = signals[idx], labels[idx], sources[idx]

    with h5py.File(OUT_FILE, "w") as f:
        f.create_dataset("signals", data=signals, compression="gzip", compression_opts=4)
        f.create_dataset("labels", data=labels, compression="gzip")
        f.create_dataset("source", data=sources.astype("S12"), compression="gzip")
        f.attrs["target_length"] = TARGET_LEN
        f.attrs["label_map"] = "0=normal, 1=abnormal, 2=noisy/unknown"
        f.attrs["classes"] = ["normal", "abnormal", "noisy/unknown"]

    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
