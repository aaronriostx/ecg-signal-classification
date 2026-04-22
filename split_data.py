"""
Reads build/ecg_unified.h5 and produces stratified splits:
  75% train  → build/train.h5
  10% val    → build/val.h5
  15% test   → build/test.h5
"""

import os
import numpy as np
import h5py
from sklearn.model_selection import train_test_split

IN_FILE  = os.path.join("build", "ecg_unified.h5")
OUT_DIR  = "build"

TRAIN_RATIO = 0.75
VAL_RATIO   = 0.10
TEST_RATIO  = 0.15
SEED        = 42


def save_split(name, signals, labels, sources, label_map):
    path = os.path.join(OUT_DIR, f"{name}.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("signals", data=signals, compression="gzip", compression_opts=4)
        f.create_dataset("labels",  data=labels,  compression="gzip")
        f.create_dataset("source",  data=sources, compression="gzip")
        f.attrs["label_map"] = label_map
        f.attrs["classes"]   = ["normal", "abnormal", "noisy/unknown"]
        f.attrs["n_samples"] = len(signals)
    return path


def class_breakdown(labels):
    names = {0: "normal", 1: "abnormal", 2: "noisy/unknown"}
    total = len(labels)
    parts = []
    for cls, name in names.items():
        n = (labels == cls).sum()
        parts.append(f"{name}: {n:,} ({100*n/total:.1f}%)")
    return "  |  ".join(parts)


def main():
    with h5py.File(IN_FILE, "r") as f:
        signals   = f["signals"][:]
        labels    = f["labels"][:]
        sources   = f["source"][:]
        label_map = f.attrs["label_map"]

    # Split off test first, then split remainder into train/val
    val_of_trainval = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

    idx = np.arange(len(signals))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=TEST_RATIO, stratify=labels, random_state=SEED
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_of_trainval,
        stratify=labels[idx_trainval], random_state=SEED
    )

    splits = {
        "train": idx_train,
        "val":   idx_val,
        "test":  idx_test,
    }

    print(f"Total: {len(signals):,} records\n")
    for name, idx in splits.items():
        path = save_split(
            name,
            signals[idx],
            labels[idx],
            sources[idx],
            label_map,
        )
        pct = 100 * len(idx) / len(signals)
        print(f"{name:<5}  {len(idx):>7,}  ({pct:.0f}%)  →  {path}")
        print(f"       {class_breakdown(labels[idx])}")


if __name__ == "__main__":
    main()
