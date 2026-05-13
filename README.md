# ECG Signal Classification

Multi-dataset ECG signal classification using MIT-BIH, PTB-DB, and PhysioNet 2017. Signals are unified into three classes: **Normal**, **Abnormal**, and **Noisy/Unknown**.

## Project Structure

```
ecg-signal-classification/
├── raw-data/
│   ├── mitbih/          # mitbih_train.csv, mitbih_test.csv
│   ├── ptbdb/           # ptbdb_normal.csv, ptbdb_abnormal.csv
│   └── physionet2017/   # physionet2017.csv
├── build/               # Generated outputs
│   ├── ecg_unified.h5   # Unified dataset (132,526 records)
│   ├── train.h5         # 75% training split (99,394 records)
│   ├── val.h5           # 10% validation split (13,253 records)
│   ├── test.h5          # 15% test split (19,879 records)
│   ├── ecg_signals.png  # Signal visualization
│   ├── cnn_1d/
│   │   ├── model.pt             # Saved model weights
│   │   ├── metrics.json         # Classification report & hyperparameters
│   │   ├── architecture.png     # Layer diagram (visualize_cnn.py)
│   │   ├── training_curves.png  # Loss & accuracy over epochs
│   │   ├── confusion_matrix.png # Normalised confusion matrix
│   │   └── roc_curves.png       # One-vs-rest ROC curves
│   ├── resnet_1d/
│   │   ├── model.pt             # Saved model weights
│   │   ├── metrics.json         # Classification report & hyperparameters
│   │   ├── architecture.png     # Layer diagram + residual block detail (visualize_resnet.py)
│   │   ├── training_curves.png  # Loss & accuracy over epochs
│   │   ├── confusion_matrix.png # Normalised confusion matrix
│   │   └── roc_curves.png       # One-vs-rest ROC curves
│   └── xgboost/
│       ├── model.json           # Saved XGBoost model
│       ├── metrics.json         # Classification report & hyperparameters
│       ├── training_curves.png  # Log-loss per boosting round (train vs val)
│       ├── confusion_matrix.png # Normalised confusion matrix
│       ├── roc_curves.png       # One-vs-rest ROC curves
│       └── feature_importance.png # Top-30 features by gain
├── environment.yml
├── preprocess.py
├── plot_signals.py
├── split_data.py
├── cnn_1d_train.py
├── resnet_1d_train.py
├── xgboost_train.py
├── visualize_cnn.py
├── visualize_resnet.py
└── summary.py
```

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate ecg-classification
```

## Usage

### 1. Preprocess

Unifies all three datasets into a single HDF5 file at `build/ecg_unified.h5`:

```bash
python preprocess.py
```

| Dataset | Records | Notes |
|---|---|---|
| MIT-BIH (train + test) | 109,446 | Labels 1/2/3 → abnormal, 4 → noisy/unknown |
| PTB-DB (normal + abnormal) | 14,552 | Binary labels |
| PhysioNet 2017 | 8,528 | Resampled from 2000 → 187 samples |
| **Total** | **132,526** | Shuffled with seed 42 |

Label mapping:

| Label | Class | Count |
|---|---|---|
| 0 | Normal | 99,711 (75.2%) |
| 1 | Abnormal | 24,497 (18.5%) |
| 2 | Noisy/Unknown | 8,318 (6.3%) |

### 2. Visualize

Plots all 132,526 ECG signals in a 3-panel figure colored by classification, saved to `build/ecg_signals.png`. The output has a transparent background so it can be dropped directly into slides or documents:

```bash
python plot_signals.py
```

### 3. Split

Splits `build/ecg_unified.h5` into stratified train, validation, and test sets:

```bash
python split_data.py
```

| Split | File | Records | Ratio |
|---|---|---|---|
| Train | `build/train.h5` | 99,394 | 75% |
| Validation | `build/val.h5` | 13,253 | 10% |
| Test | `build/test.h5` | 19,879 | 15% |

Class distribution is preserved across all splits (75.2% normal / 18.5% abnormal / 6.3% noisy/unknown).

### 4. Train 1D CNN

Trains a 1D CNN classifier on the split data and evaluates it on the test set:

```bash
python cnn_1d_train.py
```

The script will automatically use CUDA, Apple MPS, or CPU depending on what is available. Outputs are saved to `build/cnn_1d/`:

| File | Description |
|---|---|
| `model.pt` | Best model weights (saved at lowest validation loss) |
| `metrics.json` | Classification report, hyperparameters, and training summary |
| `training_curves.png` | Loss and accuracy per epoch for train and validation |
| `confusion_matrix.png` | Normalised confusion matrix on the test set |
| `roc_curves.png` | One-vs-rest ROC curves with AUC per class |

Training uses class-weighted cross-entropy loss to account for the class imbalance, and stops early if validation loss does not improve for 10 consecutive epochs.

### 5. Train 1D ResNet

Trains a 1D ResNet classifier on the split data and evaluates it on the test set:

```bash
python resnet_1d_train.py
```

The ResNet uses residual skip connections across 4 layers (64 → 128 → 256 → 512 channels), making it deeper and more expressive than the 1D CNN. Outputs are saved to `build/resnet_1d/`:

| File | Description |
|---|---|
| `model.pt` | Best model weights (saved at lowest validation loss) |
| `metrics.json` | Classification report, hyperparameters, and training summary |
| `training_curves.png` | Loss and accuracy per epoch for train and validation |
| `confusion_matrix.png` | Normalised confusion matrix on the test set |
| `roc_curves.png` | One-vs-rest ROC curves with AUC per class |

Training uses class-weighted cross-entropy loss to account for the class imbalance, and stops early if validation loss does not improve for 10 consecutive epochs.

### 6. Train XGBoost

Trains an XGBoost gradient boosted tree classifier on the split data and evaluates it on the test set:

```bash
python xgboost_train.py
```

Unlike the deep learning models, XGBoost operates directly on the flat 187-sample feature vector with no GPU required. Outputs are saved to `build/xgboost/`:

| File | Description |
|---|---|
| `model.json` | Saved XGBoost model (native format) |
| `metrics.json` | Classification report, hyperparameters, and best boosting round |
| `training_curves.png` | Log-loss per boosting round for train and validation |
| `confusion_matrix.png` | Normalised confusion matrix on the test set |
| `roc_curves.png` | One-vs-rest ROC curves with AUC per class |
| `feature_importance.png` | Top-30 time-step positions by gain — shows which part of the ECG beat is most discriminative |

Training uses class-balanced sample weights to account for class imbalance, and stops early if validation log-loss does not improve for 20 consecutive boosting rounds.

### 7. Visualize Architectures

Generates architecture diagrams for the two deep learning models:

```bash
python visualize_cnn.py     # build/cnn_1d/architecture.png
python visualize_resnet.py  # build/resnet_1d/architecture.png
```

| File | Description |
|---|---|
| `build/cnn_1d/architecture.png` | Horizontal pipeline showing all 4 conv blocks, classifier head, and output shapes |
| `build/resnet_1d/architecture.png` | Pipeline overview + side-by-side detail of both `ResidualBlock` variants (identity vs. projection shortcut) |

All output PNGs use transparent backgrounds.

### 9. Compare Models

Prints a side-by-side comparison of all trained models to the console:

```bash
python summary.py
```

Displays the following metrics for each model, with the best value highlighted green and worst red per row:

| Metric | Description |
|---|---|
| Accuracy | Overall fraction of correct predictions on the test set |
| Macro F1 | Unweighted mean F1 across all three classes |
| Macro Precision | Unweighted mean precision across all three classes |
| Macro Recall | Unweighted mean recall across all three classes |
| Weighted F1 | F1 weighted by class support |
| Training Time | Wall-clock time for the training loop |
| Epochs / Rounds | Epochs trained (CNN, ResNet) or boosting rounds (XGBoost) |

Per-class precision, recall, F1, and support are also shown for Normal, Abnormal, and Noisy/Unknown.
