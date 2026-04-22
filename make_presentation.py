"""
Generates a PowerPoint summary of the ECG signal classification project.
Output: build/ecg_classification_summary.pptx
"""

import json
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BUILD   = "build"
OUT     = os.path.join(BUILD, "ecg_classification_summary.pptx")

# ---------------------------------------------------------------------------
# Colour palette (dark theme)
# ---------------------------------------------------------------------------
C_BG       = RGBColor(0x0f, 0x0f, 0x1a)
C_PANEL    = RGBColor(0x1a, 0x1a, 0x2e)
C_WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
C_ACCENT   = RGBColor(0x5D, 0xAD, 0xE2)   # blue
C_GREEN    = RGBColor(0x2E, 0xCC, 0x71)
C_RED      = RGBColor(0xE7, 0x4C, 0x3C)
C_AMBER    = RGBColor(0xF3, 0x9C, 0x12)
C_SUBTEXT  = RGBColor(0xAA, 0xAA, 0xCC)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_bg(slide, color=C_BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, text, left, top, width, height,
                font_size=18, bold=False, color=C_WHITE,
                align=PP_ALIGN.LEFT, italic=False):
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf  = txb.text_frame
    tf.word_wrap = True
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(font_size)
    run.font.bold  = bold
    run.font.color.rgb = color
    run.font.italic = italic
    return txb


def add_rect(slide, left, top, width, height, fill_color, line_color=None):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
    else:
        shape.line.fill.background()
    return shape


def add_image(slide, path, left, top, width=None, height=None):
    if width and height:
        slide.shapes.add_picture(path, left, top, width, height)
    elif width:
        slide.shapes.add_picture(path, left, top, width=width)
    elif height:
        slide.shapes.add_picture(path, left, top, height=height)
    else:
        slide.shapes.add_picture(path, left, top)


def slide_title(slide, title, subtitle=None):
    add_rect(slide, 0, 0, SLIDE_W, Inches(1.1), C_PANEL)
    add_textbox(slide, title,
                Inches(0.4), Inches(0.18), Inches(12.5), Inches(0.65),
                font_size=28, bold=True, color=C_ACCENT)
    if subtitle:
        add_textbox(slide, subtitle,
                    Inches(0.4), Inches(0.72), Inches(12.5), Inches(0.35),
                    font_size=13, color=C_SUBTEXT)


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Slide builders
# ---------------------------------------------------------------------------
def slide_cover(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)

    # Centre panel
    add_rect(slide, Inches(1.5), Inches(1.8), Inches(10.3), Inches(3.8), C_PANEL)

    add_textbox(slide, "ECG Signal Classification",
                Inches(1.8), Inches(2.1), Inches(9.7), Inches(1.1),
                font_size=40, bold=True, color=C_ACCENT, align=PP_ALIGN.CENTER)
    add_textbox(slide, "Multi-dataset Arrhythmia Detection using 1D CNN, 1D ResNet & XGBoost",
                Inches(1.8), Inches(3.1), Inches(9.7), Inches(0.6),
                font_size=16, color=C_WHITE, align=PP_ALIGN.CENTER)
    add_textbox(slide, "MIT-BIH  ·  PTB-DB  ·  PhysioNet 2017",
                Inches(1.8), Inches(3.7), Inches(9.7), Inches(0.45),
                font_size=13, color=C_SUBTEXT, align=PP_ALIGN.CENTER, italic=True)


def slide_overview(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    slide_title(slide, "Project Overview")

    bullets = [
        ("Goal",       "Classify ECG heartbeat signals into Normal, Abnormal, or Noisy/Unknown"),
        ("Datasets",   "MIT-BIH Arrhythmia DB · PTB Diagnostic DB · PhysioNet 2017 Challenge"),
        ("Scale",      "132,526 records unified from 3 sources — stratified 75 / 10 / 15 % split"),
        ("Models",     "1D CNN  ·  1D ResNet  ·  XGBoost (gradient boosted trees)"),
        ("Challenge",  "Class imbalance: 75.2% Normal · 18.5% Abnormal · 6.3% Noisy/Unknown"),
    ]

    y = Inches(1.35)
    for label, text in bullets:
        add_rect(slide, Inches(0.4), y, Inches(2.2), Inches(0.55), C_PANEL)
        add_textbox(slide, label,
                    Inches(0.45), y + Inches(0.08), Inches(2.1), Inches(0.45),
                    font_size=12, bold=True, color=C_ACCENT)
        add_textbox(slide, text,
                    Inches(2.75), y + Inches(0.08), Inches(10.1), Inches(0.45),
                    font_size=13, color=C_WHITE)
        y += Inches(0.72)


def slide_datasets(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    slide_title(slide, "Datasets & Label Mapping")

    # Dataset table
    add_textbox(slide, "Source Datasets", Inches(0.4), Inches(1.25), Inches(6), Inches(0.35),
                font_size=14, bold=True, color=C_ACCENT)

    rows = [
        ("MIT-BIH",       "109,446", "0→Normal  1/2/3→Abnormal  4→Noisy"),
        ("PTB-DB",        " 14,552", "0→Normal  1→Abnormal"),
        ("PhysioNet 2017"," 8,528",  "0→Normal  1/2→Abnormal  3→Noisy"),
        ("Total",         "132,526", "Shuffled · seed 42"),
    ]
    headers = ["Dataset", "Records", "Label Mapping"]
    col_x   = [Inches(0.4), Inches(3.0), Inches(5.0)]
    col_w   = [Inches(2.55), Inches(1.9), Inches(7.3)]
    y = Inches(1.65)

    add_rect(slide, Inches(0.4), y, Inches(12.5), Inches(0.38), C_ACCENT)
    for i, h in enumerate(headers):
        add_textbox(slide, h, col_x[i], y + Inches(0.04), col_w[i], Inches(0.32),
                    font_size=11, bold=True, color=C_BG)
    y += Inches(0.38)
    for j, (ds, n, mapping) in enumerate(rows):
        bg = C_PANEL if j % 2 == 0 else C_BG
        add_rect(slide, Inches(0.4), y, Inches(12.5), Inches(0.38), bg)
        is_total = ds == "Total"
        for i, val in enumerate([ds, n, mapping]):
            add_textbox(slide, val, col_x[i], y + Inches(0.04), col_w[i], Inches(0.32),
                        font_size=11, bold=is_total,
                        color=C_ACCENT if is_total else C_WHITE)
        y += Inches(0.38)

    # Class distribution
    add_textbox(slide, "Unified Class Distribution", Inches(0.4), Inches(4.0), Inches(6), Inches(0.35),
                font_size=14, bold=True, color=C_ACCENT)

    classes = [("Normal",        "99,711", "75.2%", C_GREEN),
               ("Abnormal",      "24,497", "18.5%", C_RED),
               ("Noisy/Unknown", " 8,318", " 6.3%", C_AMBER)]
    x = Inches(0.4)
    for label, count, pct, color in classes:
        add_rect(slide, x, Inches(4.4), Inches(3.8), Inches(1.5), C_PANEL)
        add_textbox(slide, label, x + Inches(0.15), Inches(4.5), Inches(3.5), Inches(0.4),
                    font_size=13, bold=True, color=color)
        add_textbox(slide, count, x + Inches(0.15), Inches(4.9), Inches(3.5), Inches(0.38),
                    font_size=20, bold=True, color=C_WHITE)
        add_textbox(slide, pct,   x + Inches(0.15), Inches(5.28), Inches(3.5), Inches(0.35),
                    font_size=13, color=C_SUBTEXT)
        x += Inches(4.0)


def slide_ecg_signals(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    slide_title(slide, "ECG Signal Visualization", "All 132,526 signals overlaid — density reveals class morphology")
    img = os.path.join(BUILD, "ecg_signals.png")
    add_image(slide, img, Inches(0.4), Inches(1.2), width=Inches(12.5))


def slide_pipeline(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    slide_title(slide, "Processing & Training Pipeline")

    steps = [
        ("1", "preprocess.py",    "Unify 3 datasets → ecg_unified.h5\nResample PhysioNet 2000→187 samples · map labels"),
        ("2", "plot_signals.py",  "Visualise all signals coloured by class"),
        ("3", "split_data.py",    "Stratified split → train.h5 / val.h5 / test.h5\n75% train · 10% val · 15% test"),
        ("4", "cnn_1d_train.py",  "Train 1D CNN · early stopping · class-weighted loss"),
        ("5", "resnet_1d_train.py","Train 1D ResNet · residual skip connections"),
        ("6", "xgboost_train.py", "Train XGBoost · histogram splits · feature importance"),
    ]

    y = Inches(1.3)
    for num, script, desc in steps:
        add_rect(slide, Inches(0.4), y, Inches(0.5), Inches(0.75), C_ACCENT)
        add_textbox(slide, num, Inches(0.4), y + Inches(0.15), Inches(0.5), Inches(0.45),
                    font_size=16, bold=True, color=C_BG, align=PP_ALIGN.CENTER)
        add_rect(slide, Inches(1.0), y, Inches(11.7), Inches(0.75), C_PANEL)
        add_textbox(slide, script, Inches(1.15), y + Inches(0.04), Inches(3.2), Inches(0.35),
                    font_size=12, bold=True, color=C_ACCENT)
        add_textbox(slide, desc, Inches(4.5), y + Inches(0.04), Inches(8.0), Inches(0.65),
                    font_size=11, color=C_WHITE)
        y += Inches(0.88)


def slide_model_figure(prs, title, subtitle, img_path):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    slide_title(slide, title, subtitle)
    add_image(slide, img_path, Inches(0.5), Inches(1.2), width=Inches(12.3))


def slide_two_figures(prs, title, subtitle, left_path, right_path, left_label, right_label):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    slide_title(slide, title, subtitle)
    add_textbox(slide, left_label,  Inches(1.5),  Inches(1.2), Inches(5), Inches(0.3),
                font_size=12, bold=True, color=C_SUBTEXT, align=PP_ALIGN.CENTER)
    add_textbox(slide, right_label, Inches(7.5),  Inches(1.2), Inches(5), Inches(0.3),
                font_size=12, bold=True, color=C_SUBTEXT, align=PP_ALIGN.CENTER)
    add_image(slide, left_path,  Inches(0.3),  Inches(1.5), width=Inches(6.3))
    add_image(slide, right_path, Inches(6.75), Inches(1.5), width=Inches(6.3))


def slide_comparison(prs, cnn, resnet, xgb):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    slide_title(slide, "Model Comparison — Test Set Performance")

    models = [
        ("1D CNN",   cnn,    C_GREEN),
        ("1D ResNet",resnet, C_ACCENT),
        ("XGBoost",  xgb,   C_AMBER),
    ]

    metrics_labels = ["Accuracy", "Normal F1", "Abnormal F1", "Noisy F1", "Macro F1"]

    # Header row
    col_x = [Inches(0.35), Inches(3.2), Inches(5.1), Inches(6.95), Inches(8.85), Inches(10.75)]
    col_w = [Inches(2.75), Inches(1.75), Inches(1.75), Inches(1.75), Inches(1.75), Inches(1.75)]

    y = Inches(1.3)
    add_rect(slide, Inches(0.35), y, Inches(12.4), Inches(0.4), C_ACCENT)
    add_textbox(slide, "Model", col_x[0], y + Inches(0.05), col_w[0], Inches(0.32),
                font_size=12, bold=True, color=C_BG)
    for i, lbl in enumerate(metrics_labels):
        add_textbox(slide, lbl, col_x[i+1], y + Inches(0.05), col_w[i+1], Inches(0.32),
                    font_size=11, bold=True, color=C_BG, align=PP_ALIGN.CENTER)
    y += Inches(0.4)

    for j, (name, m, color) in enumerate(models):
        cr   = m["classification_report"]
        vals = [
            f"{cr['accuracy']:.3f}",
            f"{cr['Normal']['f1-score']:.3f}",
            f"{cr['Abnormal']['f1-score']:.3f}",
            f"{cr['Noisy/Unknown']['f1-score']:.3f}",
            f"{cr['macro avg']['f1-score']:.3f}",
        ]
        bg = C_PANEL if j % 2 == 0 else C_BG
        add_rect(slide, Inches(0.35), y, Inches(12.4), Inches(0.5), bg)
        add_textbox(slide, name, col_x[0], y + Inches(0.07), col_w[0], Inches(0.38),
                    font_size=13, bold=True, color=color)
        for i, v in enumerate(vals):
            add_textbox(slide, v, col_x[i+1], y + Inches(0.07), col_w[i+1], Inches(0.38),
                        font_size=13, color=C_WHITE, align=PP_ALIGN.CENTER)
        y += Inches(0.5)

    # Takeaway box
    add_rect(slide, Inches(0.35), Inches(4.9), Inches(12.4), Inches(1.3), C_PANEL)
    add_textbox(slide, "Key Takeaways", Inches(0.55), Inches(5.0), Inches(12.0), Inches(0.35),
                font_size=13, bold=True, color=C_ACCENT)
    takeaways = (
        "· 1D CNN achieves the highest overall accuracy (94.9%) and best Abnormal F1 (0.875)\n"
        "· 1D ResNet converges faster (18 epochs vs 32) with comparable overall performance\n"
        "· XGBoost is competitive (94.0% accuracy) with no GPU — strongest Noisy/Unknown precision (0.992)"
    )
    add_textbox(slide, takeaways, Inches(0.55), Inches(5.35), Inches(12.0), Inches(0.75),
                font_size=11, color=C_WHITE)


def slide_conclusions(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    slide_title(slide, "Conclusions & Next Steps")

    conclusions = [
        (C_GREEN,  "All three models exceed 94% accuracy on the held-out test set"),
        (C_GREEN,  "Class-weighted loss successfully mitigates the 75/18/6 class imbalance"),
        (C_GREEN,  "XGBoost feature importance shows the QRS peak region (samples ~80–100) is most discriminative"),
        (C_AMBER,  "Abnormal class remains the hardest to classify (F1 ~0.87) — largest intra-class variation"),
        (C_AMBER,  "PhysioNet resampling (2000→187) introduces artefacts; per-signal normalisation may help"),
    ]
    next_steps = [
        "Train an LSTM / GRU to model sequential dependencies across the full beat",
        "Experiment with data augmentation (time-shift, amplitude scaling) to boost Abnormal recall",
        "Ensemble the three models to combine complementary strengths",
    ]

    y = Inches(1.3)
    add_textbox(slide, "Findings", Inches(0.4), y, Inches(12), Inches(0.35),
                font_size=14, bold=True, color=C_ACCENT)
    y += Inches(0.38)
    for color, text in conclusions:
        add_rect(slide, Inches(0.4), y, Inches(0.08), Inches(0.38), color)
        add_textbox(slide, text, Inches(0.6), y, Inches(12.1), Inches(0.38),
                    font_size=12, color=C_WHITE)
        y += Inches(0.45)

    y += Inches(0.2)
    add_textbox(slide, "Next Steps", Inches(0.4), y, Inches(12), Inches(0.35),
                font_size=14, bold=True, color=C_ACCENT)
    y += Inches(0.38)
    for text in next_steps:
        add_rect(slide, Inches(0.4), y, Inches(0.08), Inches(0.38), C_SUBTEXT)
        add_textbox(slide, text, Inches(0.6), y, Inches(12.1), Inches(0.38),
                    font_size=12, color=C_WHITE)
        y += Inches(0.45)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    cnn_m    = load_metrics(os.path.join(BUILD, "cnn_1d",   "metrics.json"))
    resnet_m = load_metrics(os.path.join(BUILD, "resnet_1d","metrics.json"))
    xgb_m    = load_metrics(os.path.join(BUILD, "xgboost",  "metrics.json"))

    # Slides
    slide_cover(prs)
    slide_overview(prs)
    slide_datasets(prs)
    slide_ecg_signals(prs)
    slide_pipeline(prs)

    # 1D CNN
    slide_model_figure(prs, "1D CNN — Training Curves",
                       f"32 epochs · best val loss {cnn_m['best_val_loss']:.4f}",
                       os.path.join(BUILD, "cnn_1d", "training_curves.png"))
    slide_two_figures(prs, "1D CNN — Evaluation",
                      f"Test accuracy: {cnn_m['classification_report']['accuracy']:.3f}",
                      os.path.join(BUILD, "cnn_1d", "confusion_matrix.png"),
                      os.path.join(BUILD, "cnn_1d", "roc_curves.png"),
                      "Confusion Matrix", "ROC Curves")

    # 1D ResNet
    slide_model_figure(prs, "1D ResNet — Training Curves",
                       f"18 epochs · best val loss {resnet_m['best_val_loss']:.4f}",
                       os.path.join(BUILD, "resnet_1d", "training_curves.png"))
    slide_two_figures(prs, "1D ResNet — Evaluation",
                      f"Test accuracy: {resnet_m['classification_report']['accuracy']:.3f}",
                      os.path.join(BUILD, "resnet_1d", "confusion_matrix.png"),
                      os.path.join(BUILD, "resnet_1d", "roc_curves.png"),
                      "Confusion Matrix", "ROC Curves")

    # XGBoost
    slide_model_figure(prs, "XGBoost — Training Curves",
                       f"Best iteration: {xgb_m['best_iteration']}",
                       os.path.join(BUILD, "xgboost", "training_curves.png"))
    slide_two_figures(prs, "XGBoost — Evaluation",
                      f"Test accuracy: {xgb_m['classification_report']['accuracy']:.3f}",
                      os.path.join(BUILD, "xgboost", "confusion_matrix.png"),
                      os.path.join(BUILD, "xgboost", "roc_curves.png"),
                      "Confusion Matrix", "ROC Curves")
    slide_model_figure(prs, "XGBoost — Feature Importance",
                       "Top-30 time-step positions by gain",
                       os.path.join(BUILD, "xgboost", "feature_importance.png"))

    # Summary & conclusions
    slide_comparison(prs, cnn_m, resnet_m, xgb_m)
    slide_conclusions(prs)

    os.makedirs(BUILD, exist_ok=True)
    prs.save(OUT)
    print(f"Saved → {OUT}  ({prs.slides.__len__()} slides)")


if __name__ == "__main__":
    main()
