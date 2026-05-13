"""
Visualizes the CNN1D architecture as a layer diagram.

Writes: build/cnn_1d/architecture.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BUILD   = "build"
OUT_DIR = os.path.join(BUILD, "cnn_1d")

DARK_BG  = "#0f0f1a"
DARK_FIG = "#1a1a2e"
TEXT_COL = "#aaaacc"
WHITE    = "#e8e8ff"
GOLD     = "#f39c12"

COL_INPUT = "#1c2833"
COL_CONV  = "#154360"
COL_FC    = "#145a32"
COL_OUT   = "#4a235a"


def draw_box(ax, cx, cy, w, h, title, ops, shape, color):
    rect = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="#4466bb", linewidth=1.2, zorder=2,
    )
    ax.add_patch(rect)

    top = cy + h / 2
    ax.text(cx, top - 0.22, title,
            ha="center", va="top", color=WHITE,
            fontsize=9, fontweight="bold", zorder=3)

    y = top - 0.58
    for op in ops:
        ax.text(cx, y, op, ha="center", va="top", color=TEXT_COL, fontsize=7, zorder=3)
        y -= 0.40

    ax.text(cx, cy - h / 2 - 0.18, shape,
            ha="center", va="top", color=GOLD, fontsize=7.5, fontstyle="italic", zorder=3)


def draw_arrow(ax, x1, x2, cy):
    ax.annotate(
        "", xy=(x2, cy), xytext=(x1, cy),
        arrowprops=dict(arrowstyle="-|>", color="#5577cc", lw=1.4),
        zorder=4,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 7))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 7)
    ax.axis("off")

    cy = 3.6

    # (cx, w, h, title, ops, shape_label, color)
    stages = [
        (1.2,  1.8, 1.6,
         "Input", [],
         "(1 × 187)", COL_INPUT),

        (3.9,  2.5, 3.2,
         "Block 1",
         ["Conv1d(1→32, k=7, p=3)", "BatchNorm1d + ReLU", "MaxPool1d(2)"],
         "(32 × 93)", COL_CONV),

        (6.8,  2.5, 3.2,
         "Block 2",
         ["Conv1d(32→64, k=5, p=2)", "BatchNorm1d + ReLU", "MaxPool1d(2)"],
         "(64 × 46)", COL_CONV),

        (9.7,  2.5, 3.2,
         "Block 3",
         ["Conv1d(64→128, k=3, p=1)", "BatchNorm1d + ReLU", "MaxPool1d(2)"],
         "(128 × 23)", COL_CONV),

        (12.6, 2.5, 3.2,
         "Block 4",
         ["Conv1d(128→256, k=3, p=1)", "BatchNorm1d + ReLU", "AdaptiveAvgPool1d(1)"],
         "(256 × 1)", COL_CONV),

        (15.7, 2.8, 4.2,
         "Classifier",
         ["Flatten  →  (256,)", "Linear(256 → 128)", "ReLU + Dropout(0.4)", "Linear(128 → 3)"],
         "(3,)", COL_FC),

        (18.6, 1.8, 3.2,
         "Output",
         ["Normal", "Abnormal", "Noisy/Unknown"],
         "3 classes", COL_OUT),
    ]

    for i, (cx, w, h, title, ops, shape, color) in enumerate(stages):
        draw_box(ax, cx, cy, w, h, title, ops, shape, color)
        if i < len(stages) - 1:
            next_cx, next_w = stages[i + 1][0], stages[i + 1][1]
            draw_arrow(ax, cx + w / 2, next_cx - next_w / 2, cy)

    # Parameter count annotation
    ax.text(10.0, 0.4, "Total trainable parameters: 168,067  ·  Estimated size: 0.67 MB",
            ha="center", va="bottom", color=TEXT_COL, fontsize=8)

    ax.set_title("CNN1D Architecture — ECG Signal Classification",
                 color=WHITE, fontsize=13, fontweight="bold", pad=14)

    out_path = os.path.join(OUT_DIR, "architecture.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
