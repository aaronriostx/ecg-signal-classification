# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

```bash
conda env create -f environment.yml   # first time only
conda activate ecg-classification
```

Default environment uses CPU-only PyTorch. To enable GPU, replace `cpuonly` in `environment.yml` with `pytorch-cuda=12.1`.

## Pipeline — run in order

Each script depends on the outputs of the previous one. All scripts must be run from the project root.

```bash
python preprocess.py        # raw-data/ → build/ecg_unified.h5
python plot_signals.py      # build/ecg_unified.h5 → build/ecg_signals.png (transparent background, no title)
python split_data.py        # build/ecg_unified.h5 → build/{train,val,test}.h5
python cnn_1d_train.py      # build/{train,val,test}.h5 → build/cnn_1d/
python resnet_1d_train.py   # build/{train,val,test}.h5 → build/resnet_1d/
python xgboost_train.py     # build/{train,val,test}.h5 → build/xgboost/
python summary.py           # prints side-by-side model comparison to console
python make_presentation.py # all build/ outputs → build/ecg_classification_summary.pptx
```

Training scripts (steps 4–6) are independent of each other and can be run in any order or re-run individually without re-running earlier steps.

## Label scheme

All three source datasets are unified into three classes:

| Label | Class | Source mapping |
|---|---|---|
| 0 | Normal | MIT-BIH 0, PTB-DB 0, PhysioNet 0 |
| 1 | Abnormal | MIT-BIH 1/2/3, PTB-DB 1, PhysioNet 1/2 |
| 2 | Noisy/Unknown | MIT-BIH 4, PhysioNet 3 |

## Data format

All `.h5` files share the same schema:
- `signals`: float32 `(N, 187)` — per-signal min-max normalised to [0, 1]
- `labels`: int8 `(N,)` — 0/1/2
- `source`: fixed-length bytes `(N,)` — `b'mitbih'`, `b'ptbdb'`, `b'physionet201'` (truncated to 12 chars)

The `source` field is stored as `S12` bytes. When filtering by source use `np.char.startswith(sources.astype(str), 'physionet')` rather than exact equality.

PhysioNet 2017 signals are resampled from 2000→187 samples via `scipy.signal.resample` (FFT-based). Per-signal normalisation in `preprocess.py` corrects the Gibbs ringing artefacts this introduces.

## Shared training conventions

Both deep learning training scripts (`cnn_1d_train.py`, `resnet_1d_train.py`) follow identical conventions:

- **ECGDataset**: loads HDF5, adds channel dim via `.unsqueeze(1)` → shape `(N, 1, 187)`
- **Class imbalance**: `compute_class_weight("balanced")` passed as `weight=` to `nn.CrossEntropyLoss`
- **Scheduler**: `ReduceLROnPlateau(patience=4, factor=0.5)` on val loss
- **Early stopping**: `PATIENCE = 10` epochs without val loss improvement; best weights saved to `model.pt`
- **Device**: auto-selects CUDA → MPS → CPU
- **Timing**: `training_time_seconds` recorded from loop start to loop end and saved to `metrics.json`

XGBoost uses `compute_sample_weight("balanced")` passed to `fit()`, and `eval_set` with `early_stopping_rounds=20`.

## Model architectures

**CNN1D** (`cnn_1d_train.py`): 4 conv blocks (1→32→64→128→256 channels, kernels 7/5/3/3), each followed by BatchNorm + ReLU + MaxPool. AdaptiveAvgPool → Linear(256→128) → Dropout(0.4) → Linear(128→3).

**ResNet1D** (`resnet_1d_train.py`): Stem conv (64, k=7, stride=2) + MaxPool, then 4 residual layers (64→128→256→512) each with 2 `ResidualBlock`s. Blocks use a 1×1 projection shortcut when stride≠1 or channels change. AdaptiveAvgPool → Linear(512→256) → Dropout(0.4) → Linear(256→3).

**XGBoost** (`xgboost_train.py`): Takes flat `(N, 187)` input (no channel dim). `multi:softprob` objective, `tree_method="hist"`, 500 estimators, max_depth=6, subsample=0.8, colsample_bytree=0.8. `evals_result()` keys are `"validation_0"` (train) and `"validation_1"` (val).

## metrics.json schema

All three training scripts write `metrics.json` with this structure (XGBoost uses `best_iteration` instead of `training_epochs`):

```json
{
  "hyperparameters": { ... },
  "training_epochs": 22,
  "training_time_seconds": 602.8,
  "best_val_loss": 0.1017,
  "classification_report": {
    "accuracy": 0.962,
    "Normal": { "precision": ..., "recall": ..., "f1-score": ..., "support": ... },
    "Abnormal": { ... },
    "Noisy/Unknown": { ... },
    "macro avg": { ... },
    "weighted avg": { ... }
  }
}
```

`summary.py` and `make_presentation.py` both read this schema — adding new keys to `metrics.json` is safe but changing existing key names will break both scripts.

## Presentation

`make_presentation.py` uses `python-pptx` (installed via pip, not in `environment.yml`). Install with `pip install python-pptx` if missing. The slide deck is dark-themed; all colour constants are defined at the top of the file as `C_*` variables.
