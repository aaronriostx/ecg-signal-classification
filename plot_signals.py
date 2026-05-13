"""
Plots all ECG signals from build/ecg_unified.h5 in a single overlaid plot.
Each signal is colored by classification. Low alpha + density reveals structure.

Output: build/ecg_signals.png
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

H5_FILE = os.path.join("build", "ecg_unified.h5")
OUT_FILE = os.path.join("build", "ecg_signals.png")

CLASSES = {
    0: ("Normal",        "#2ecc71", 0.008),
    1: ("Abnormal",      "#e74c3c", 0.025),
    2: ("Noisy/Unknown", "#f39c12", 0.055),
}


def main():
    with h5py.File(H5_FILE, "r") as f:
        signals = f["signals"][:]
        labels  = f["labels"][:]

    x = np.arange(signals.shape[1])

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.patch.set_alpha(0.0)

    for ax, (cls, (label, color, alpha)) in zip(axes, CLASSES.items()):
        mask = labels == cls
        sigs = signals[mask]

        lo = np.percentile(sigs, 1) - 0.05
        hi = np.percentile(sigs, 99) + 0.05

        segs = np.stack([np.broadcast_to(x, sigs.shape), sigs], axis=-1)
        lc = LineCollection(segs, color=color, alpha=alpha, linewidth=0.4, rasterized=True)
        ax.add_collection(lc)

        ax.set_facecolor("none")
        ax.set_xlim(0, signals.shape[1] - 1)
        ax.set_ylim(lo, hi)
        ax.set_ylabel("Amplitude", color="#aaaacc", fontsize=14)
        ax.tick_params(colors="#aaaacc", labelsize=14)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        ax.text(
            0.01, 0.93, f"{label}  ({mask.sum():,})",
            transform=ax.transAxes,
            color=color, fontsize=15, fontweight="bold", va="top",
        )

        print(f"  {label:<15} {mask.sum():>7,} signals")

    axes[-1].set_xlabel("Sample", color="#aaaacc", fontsize=15)
    fig.tight_layout()

    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
