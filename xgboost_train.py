"""
Trains an XGBoost classifier on the pre-split ECG datasets and evaluates performance.

Reads:  build/train.h5, build/val.h5, build/test.h5
Writes: build/xgboost/
  model.json            – saved XGBoost model
  metrics.json          – classification report + hyperparameters
  training_curves.png   – log-loss per boosting round (train vs val)
  confusion_matrix.png  – normalised confusion matrix
  roc_curves.png        – one-vs-rest ROC curves per class
  feature_importance.png – top-30 features by gain
"""

import os
import json
import time
import numpy as np
import h5py
import xgboost as xgb  # pip: xgboost
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BUILD   = "build"
OUT_DIR = os.path.join(BUILD, "xgboost")
CLASSES = ["Normal", "Abnormal", "Noisy/Unknown"]

PARAMS = {
    "objective":        "multi:softprob",
    "num_class":        3,
    "eval_metric":      "mlogloss",
    "learning_rate":    0.1,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "n_estimators":     500,
    "early_stopping_rounds": 20,
    "seed":             42,
    "n_jobs":           -1,
    "tree_method":      "hist",   # fast histogram-based split finding
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_split(h5_path):
    with h5py.File(h5_path, "r") as f:
        signals = f["signals"][:]          # (N, 187) — already flat, no channel dim needed
        labels  = f["labels"][:].astype(int)
    return signals, labels


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
DARK_BG  = "#0f0f1a"
DARK_FIG = "#1a1a2e"
TEXT_COL = "#aaaacc"
ACCENT   = ["#2ecc71", "#e74c3c", "#f39c12"]


def plot_training_curves(evals_result, out_path):
    train_loss = evals_result["validation_0"]["mlogloss"]
    val_loss   = evals_result["validation_1"]["mlogloss"]
    rounds     = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    ax.tick_params(colors=TEXT_COL)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")

    ax.plot(rounds, train_loss, color="#5dade2", label="Train")
    ax.plot(rounds, val_loss,   color="#f39c12", label="Val", linestyle="--")
    ax.set_xlabel("Boosting Round", color=TEXT_COL, fontsize=11)
    ax.set_ylabel("Log Loss",       color=TEXT_COL, fontsize=11)
    ax.legend(facecolor=DARK_BG, labelcolor="white")
    ax.set_title("Training Curves — XGBoost", color="white", fontsize=14, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_confusion_matrix(labels, preds, out_path):
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=CLASSES, yticklabels=CLASSES,
        ax=ax, linewidths=0.5, linecolor="#333355",
        annot_kws={"size": 11},
    )
    ax.set_xlabel("Predicted", color=TEXT_COL, fontsize=11)
    ax.set_ylabel("True",      color=TEXT_COL, fontsize=11)
    ax.tick_params(colors=TEXT_COL)
    ax.set_title("Confusion Matrix (normalised) — XGBoost", color="white", fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_roc_curves(labels, probs, out_path):
    labels_bin = label_binarize(labels, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    for i, (cls, color) in enumerate(zip(CLASSES, ACCENT)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{cls}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#555577", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate", color=TEXT_COL, fontsize=11)
    ax.set_ylabel("True Positive Rate",  color=TEXT_COL, fontsize=11)
    ax.tick_params(colors=TEXT_COL)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")
    ax.legend(facecolor=DARK_BG, labelcolor="white", fontsize=10)
    ax.set_title("ROC Curves (one-vs-rest) — XGBoost", color="white", fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_feature_importance(model, out_path, top_n=30):
    scores = model.get_booster().get_score(importance_type="gain")
    if not scores:
        return

    # Sort and take top N
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, values = zip(*sorted_scores)
    # Convert feature names (f0, f1, ...) to sample indices
    indices = [int(f[1:]) for f in features]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    ax.tick_params(colors=TEXT_COL)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")

    bars = ax.barh(range(len(indices)), values, color="#5dade2", alpha=0.85)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f"Sample {i}" for i in indices], color=TEXT_COL, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Gain", color=TEXT_COL, fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances (Gain) — XGBoost",
                 color="white", fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    X_train, y_train = load_split(os.path.join(BUILD, "train.h5"))
    X_val,   y_val   = load_split(os.path.join(BUILD, "val.h5"))
    X_test,  y_test  = load_split(os.path.join(BUILD, "test.h5"))
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}\n")

    # Class-balanced sample weights
    sample_weights = compute_sample_weight("balanced", y=y_train)

    model = xgb.XGBClassifier(**PARAMS)

    print("Training XGBoost...")
    t_start = time.time()
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=10,
    )
    training_time = time.time() - t_start
    evals_result = model.evals_result()

    model.save_model(os.path.join(OUT_DIR, "model.json"))
    print(f"\nBest round: {model.best_iteration}")

    print("\nEvaluating on test set...")
    probs = model.predict_proba(X_test)
    preds = probs.argmax(axis=1)

    report = classification_report(y_test, preds, target_names=CLASSES, output_dict=True)
    print(classification_report(y_test, preds, target_names=CLASSES))

    metrics = {
        "hyperparameters": PARAMS,
        "best_iteration": int(model.best_iteration),
        "training_time_seconds": round(training_time, 1),
        "classification_report": report,
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_training_curves(evals_result,            os.path.join(OUT_DIR, "training_curves.png"))
    plot_confusion_matrix(y_test, preds,           os.path.join(OUT_DIR, "confusion_matrix.png"))
    plot_roc_curves(y_test, probs,                 os.path.join(OUT_DIR, "roc_curves.png"))
    plot_feature_importance(model,                 os.path.join(OUT_DIR, "feature_importance.png"))

    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
