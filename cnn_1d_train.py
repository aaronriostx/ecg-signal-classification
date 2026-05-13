"""
Trains a 1D CNN on the pre-split ECG datasets and evaluates performance.

Reads:  build/train.h5, build/val.h5, build/test.h5
Writes: build/cnn_1d/
  model.pt              – saved model weights
  metrics.json          – classification report + hyperparameters
  training_curves.png   – loss & accuracy over epochs
  confusion_matrix.png  – normalised confusion matrix
  roc_curves.png        – one-vs-rest ROC curves per class
"""

import os
import json
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
from torchinfo import summary

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BUILD   = "build"
OUT_DIR = os.path.join(BUILD, "cnn_1d")
CLASSES = ["Normal", "Abnormal", "Noisy/Unknown"]

BATCH_SIZE = 256
LR         = 1e-3
EPOCHS     = 50
PATIENCE   = 10
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ECGDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            # Add channel dim: (N, 1, 187)
            self.signals = torch.tensor(f["signals"][:], dtype=torch.float32).unsqueeze(1)
            self.labels  = torch.tensor(f["labels"][:],  dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CNN1D(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,   32,  kernel_size=7, padding=3), nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,  64,  kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,  128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.numpy())
    return np.vstack(all_probs), np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
DARK_BG  = "#0f0f1a"
DARK_FIG = "#1a1a2e"
TEXT_COL = "#aaaacc"
ACCENT   = ["#2ecc71", "#e74c3c", "#f39c12"]


def plot_training_curves(history, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("none")

    for ax in (ax1, ax2):
        ax.set_facecolor("none")
        ax.tick_params(colors=TEXT_COL)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], color="#5dade2", label="Train")
    ax1.plot(epochs, history["val_loss"],   color="#f39c12", label="Val", linestyle="--")
    ax1.set_title("Loss",     color="white", fontsize=12)
    ax1.set_xlabel("Epoch",   color=TEXT_COL)
    ax1.set_ylabel("Loss",    color=TEXT_COL)
    ax1.legend(facecolor=DARK_BG, labelcolor="white")

    ax2.plot(epochs, history["train_acc"], color="#5dade2", label="Train")
    ax2.plot(epochs, history["val_acc"],   color="#f39c12", label="Val", linestyle="--")
    ax2.set_title("Accuracy", color="white", fontsize=12)
    ax2.set_xlabel("Epoch",   color=TEXT_COL)
    ax2.set_ylabel("Accuracy", color=TEXT_COL)
    ax2.legend(facecolor=DARK_BG, labelcolor="white")

    fig.suptitle("Training Curves", color="white", fontsize=14, fontweight="bold")
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
    ax.set_title("Confusion Matrix (normalised)", color="white", fontsize=13, fontweight="bold", pad=12)

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
    ax.set_title("ROC Curves (one-vs-rest)", color="white", fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps")  if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}\n")

    # Datasets & loaders
    train_ds = ECGDataset(os.path.join(BUILD, "train.h5"))
    val_ds   = ECGDataset(os.path.join(BUILD, "val.h5"))
    test_ds  = ECGDataset(os.path.join(BUILD, "test.h5"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Class-weighted loss to handle imbalance
    train_labels = train_ds.labels.numpy()
    weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=train_labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))

    model     = CNN1D(n_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    summary(model, input_size=(1, 1, 187), device=str(device))

    # Training loop with early stopping
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model,  val_loader,   criterion, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"train loss {tr_loss:.4f}  acc {tr_acc:.4f}  |  "
              f"val loss {vl_loss:.4f}  acc {vl_acc:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    training_time = time.time() - t_start

    # Load best weights for evaluation
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "model.pt"), map_location=device))

    # Evaluate on test set
    print("\nEvaluating on test set...")
    probs, preds, true_labels = predict(model, test_loader, device)

    report = classification_report(
        true_labels, preds,
        target_names=CLASSES,
        output_dict=True,
    )
    print(classification_report(true_labels, preds, target_names=CLASSES))

    # Save metrics
    metrics = {
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "max_epochs": EPOCHS,
            "early_stopping_patience": PATIENCE,
        },
        "training_epochs": len(history["train_loss"]),
        "training_time_seconds": round(training_time, 1),
        "best_val_loss": best_val_loss,
        "classification_report": report,
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Figures
    plot_training_curves(history,              os.path.join(OUT_DIR, "training_curves.png"))
    plot_confusion_matrix(true_labels, preds,  os.path.join(OUT_DIR, "confusion_matrix.png"))
    plot_roc_curves(true_labels, probs,        os.path.join(OUT_DIR, "roc_curves.png"))

    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
