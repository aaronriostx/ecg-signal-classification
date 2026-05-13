"""
Visualizes the ResNet1D architecture as a layer diagram.

Writes: build/resnet_1d/architecture.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

BUILD   = "build"
OUT_DIR = os.path.join(BUILD, "resnet_1d")

TEXT_COL = "#aaaacc"
WHITE    = "#e8e8ff"
GOLD     = "#f39c12"
SKIP_ID  = "#2ecc71"
SKIP_PRJ = "#e74c3c"

COL_INPUT = "#1c2833"
COL_STEM  = "#154360"
COL_RES   = "#1b3a6b"
COL_FC    = "#145a32"
COL_OUT   = "#4a235a"
COL_PANEL = "#0d1a2d"


def draw_box(ax, cx, cy, w, h, title, ops, shape, color):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="#4466bb", linewidth=1.2, zorder=2,
    ))
    top = cy + h / 2
    ax.text(cx, top - 0.22, title,
            ha="center", va="top", color=WHITE, fontsize=9, fontweight="bold", zorder=3)
    y = top - 0.58
    for op in ops:
        ax.text(cx, y, op, ha="center", va="top", color=TEXT_COL, fontsize=7, zorder=3)
        y -= 0.40
    ax.text(cx, cy - h / 2 - 0.18, shape,
            ha="center", va="top", color=GOLD, fontsize=7.5, fontstyle="italic", zorder=3)


def h_arrow(ax, x1, x2, cy):
    ax.annotate("", xy=(x2, cy), xytext=(x1, cy),
                arrowprops=dict(arrowstyle="-|>", color="#5577cc", lw=1.4), zorder=4)


def v_arrow(ax, cx, y_from, y_to, color="#5577cc"):
    ax.annotate("", xy=(cx, y_to), xytext=(cx, y_from),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2), zorder=4)


def small_box(ax, cx, cy, w, h, text, color, text_color=None, fs=7):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="#4466bb", linewidth=0.9, zorder=2,
    ))
    ax.text(cx, cy, text, ha="center", va="center",
            color=text_color or TEXT_COL, fontsize=fs, zorder=3)


def draw_block_detail(ax, cx, top_y, stride):
    bw, bh, step = 3.8, 0.62, 1.05
    ys = [top_y - i * step for i in range(4)]

    # Input label + arrow
    ax.text(cx, ys[0] + bh / 2 + 0.55, "x",
            ha="center", va="bottom", color=WHITE, fontsize=10, fontweight="bold", zorder=3)
    v_arrow(ax, cx, ys[0] + bh / 2 + 0.5, ys[0] + bh / 2 + 0.04)

    # Main path boxes
    labels = [
        f"Conv1d(k=3, stride={stride})  +  BN  +  ReLU",
        "Conv1d(k=3, stride=1)  +  BN",
        "⊕   add",
        "ReLU",
    ]
    box_colors  = [COL_RES, COL_RES, "#2c3e60", "#2c3e60"]
    text_colors = [None,    None,    WHITE,     None]

    for i, (lbl, bc, tc, y) in enumerate(zip(labels, box_colors, text_colors, ys)):
        small_box(ax, cx, y, bw, bh, lbl, bc, text_color=tc)
        if i < len(ys) - 1:
            v_arrow(ax, cx, y - bh / 2 - 0.02, ys[i + 1] + bh / 2 + 0.02)

    # Output arrow + label
    v_arrow(ax, cx, ys[-1] - bh / 2 - 0.02, ys[-1] - bh / 2 - 0.5)
    ax.text(cx, ys[-1] - bh / 2 - 0.62, "output",
            ha="center", va="top", color=WHITE, fontsize=9, fontweight="bold", zorder=3)

    # Skip connection (L-shaped path on the right)
    skip_color = SKIP_ID if stride == 1 else SKIP_PRJ
    skip_label = "Identity" if stride == 1 else "Projection"
    sx    = cx + bw / 2 + 0.55
    y_top = ys[0] + bh / 2 + 0.4
    y_bot = ys[2]

    ax.plot([cx + bw / 2, sx], [y_top, y_top], color=skip_color, lw=1.5, zorder=3)
    ax.plot([sx, sx],          [y_top, y_bot], color=skip_color, lw=1.5, zorder=3)
    ax.annotate("", xy=(cx + bw / 2 + 0.05, y_bot), xytext=(sx, y_bot),
                arrowprops=dict(arrowstyle="-|>", color=skip_color, lw=1.5), zorder=4)
    ax.text(sx + 0.12, (y_top + y_bot) / 2, skip_label,
            ha="center", va="center", color=skip_color,
            fontsize=7, rotation=90, zorder=3)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(24, 12))
    fig.patch.set_facecolor("none")

    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.1], hspace=0.08)
    ax_pipe   = fig.add_subplot(gs[0])
    ax_detail = fig.add_subplot(gs[1])

    for ax in (ax_pipe, ax_detail):
        ax.set_facecolor("none")
        ax.axis("off")

    # ── Pipeline ─────────────────────────────────────────────────────────────
    ax_pipe.set_xlim(0, 24)
    ax_pipe.set_ylim(0, 7)
    cy = 3.8

    stages = [
        (1.1,  1.6, 1.6, "Input", [],
         "(1 × 187)", COL_INPUT),
        (3.5,  2.4, 3.2, "Stem",
         ["Conv1d(1→64, k=7, s=2, p=3)", "BatchNorm1d + ReLU", "MaxPool1d(k=3, s=2, p=1)"],
         "(64 × 47)", COL_STEM),
        (6.3,  2.4, 3.2, "Layer 1",
         ["2 × ResBlock(64→64)", "stride=1,  identity skip"],
         "(64 × 47)", COL_RES),
        (9.1,  2.4, 3.2, "Layer 2",
         ["ResBlock(64→128, s=2)", "ResBlock(128→128, s=1)"],
         "(128 × 24)", COL_RES),
        (11.9, 2.4, 3.2, "Layer 3",
         ["ResBlock(128→256, s=2)", "ResBlock(256→256, s=1)"],
         "(256 × 12)", COL_RES),
        (14.7, 2.4, 3.2, "Layer 4",
         ["ResBlock(256→512, s=2)", "ResBlock(512→512, s=1)"],
         "(512 × 6)", COL_RES),
        (17.1, 2.1, 1.6, "AvgPool",
         ["AdaptiveAvgPool1d(1)"],
         "(512 × 1)", COL_STEM),
        (19.8, 2.8, 4.2, "Classifier",
         ["Flatten  →  (512,)", "Linear(512 → 256)", "ReLU + Dropout(0.4)", "Linear(256 → 3)"],
         "(3,)", COL_FC),
        (22.7, 1.6, 3.2, "Output",
         ["Normal", "Abnormal", "Noisy/Unknown"],
         "3 classes", COL_OUT),
    ]

    for i, (cx, w, h, title, ops, shape, color) in enumerate(stages):
        draw_box(ax_pipe, cx, cy, w, h, title, ops, shape, color)
        if i < len(stages) - 1:
            nxt_cx, nxt_w = stages[i + 1][0], stages[i + 1][1]
            h_arrow(ax_pipe, cx + w / 2, nxt_cx - nxt_w / 2, cy)

    ax_pipe.text(12.0, 0.4,
                 "Total trainable parameters: ~3.97M  ·  Layers 2–4 use stride-2 projection shortcuts when channels double",
                 ha="center", va="bottom", color=TEXT_COL, fontsize=8)
    ax_pipe.set_title("ResNet1D Architecture — ECG Signal Classification",
                      color=WHITE, fontsize=13, fontweight="bold", pad=10)

    # ── Residual block detail ─────────────────────────────────────────────────
    ax_detail.set_xlim(0, 24)
    ax_detail.set_ylim(0, 9)

    ax_detail.text(12.0, 8.8, "ResidualBlock — Two Variants",
                   ha="center", va="top", color=WHITE, fontsize=11, fontweight="bold")
    ax_detail.axhline(y=8.35, xmin=0.05, xmax=0.95, color="#333355", lw=0.8)

    panels = [
        (6.0,  1, SKIP_ID,
         "Variant A  ·  Same dimensions  (stride=1)\nIdentity shortcut — zero added parameters"),
        (18.0, 2, SKIP_PRJ,
         "Variant B  ·  Expanding channels  (stride=2)\nProjection shortcut — Conv1d(k=1, s=2) + BN"),
    ]

    for cx_p, stride, sc, label in panels:
        ax_detail.add_patch(mpatches.FancyBboxPatch(
            (cx_p - 4.0, 0.3), 8.0, 7.8,
            boxstyle="round,pad=0.1",
            facecolor=COL_PANEL, edgecolor="#333355", linewidth=1.0, zorder=1,
        ))
        ax_detail.text(cx_p, 8.0, label,
                       ha="center", va="top", color=sc,
                       fontsize=8, fontweight="bold", zorder=3)
        draw_block_detail(ax_detail, cx_p, top_y=6.2, stride=stride)

    out_path = os.path.join(OUT_DIR, "architecture.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
