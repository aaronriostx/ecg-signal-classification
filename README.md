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
│   └── cnn_1d/
│       ├── model.pt             # Saved model weights
│       ├── metrics.json         # Classification report & hyperparameters
│       ├── training_curves.png  # Loss & accuracy over epochs
│       ├── confusion_matrix.png # Normalised confusion matrix
│       └── roc_curves.png       # One-vs-rest ROC curves
├── environment.yml
├── preprocess.py
├── plot_signals.py
├── split_data.py
└── cnn_1d_train.py
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

Plots all 132,526 ECG signals in a 3-panel figure colored by classification, saved to `build/ecg_signals.png`:

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
