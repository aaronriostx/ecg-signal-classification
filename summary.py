"""
Prints a comparison of all trained models to the console.
Reads metrics.json from build/cnn_1d/, build/resnet_1d/, and build/xgboost/.
"""

import json
import os

MODELS = {
    "1D CNN":    "build/cnn_1d/metrics.json",
    "1D ResNet": "build/resnet_1d/metrics.json",
    "XGBoost":   "build/xgboost/metrics.json",
}
CLASSES = ["Normal", "Abnormal", "Noisy/Unknown"]

# ANSI colours
BOLD  = "\033[1m"
RESET = "\033[0m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
AMBER = "\033[93m"
RED   = "\033[91m"
DIM   = "\033[2m"


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt_time(seconds):
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def highlight(value, values, higher_is_better=True):
    """Colour the best value in a row green, worst red."""
    best  = max(values) if higher_is_better else min(values)
    worst = min(values) if higher_is_better else max(values)
    if value == best:
        return f"{GREEN}{value}{RESET}"
    if value == worst:
        return f"{RED}{value}{RESET}"
    return str(value)


def col(text, width):
    # Strip ANSI codes for length calculation
    import re
    plain = re.sub(r'\033\[[0-9;]*m', '', str(text))
    pad   = max(0, width - len(plain))
    return str(text) + " " * pad


def print_table(headers, rows, col_widths):
    sep = "  "
    header_line = sep.join(col(h, w) for h, w in zip(headers, col_widths))
    print(f"{BOLD}{CYAN}{header_line}{RESET}")
    print(DIM + "-" * sum(col_widths + [len(sep)] * (len(col_widths)-1)) + RESET)
    for row in rows:
        print(sep.join(col(v, w) for v, w in zip(row, col_widths)))


def main():
    metrics = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"{AMBER}Warning: {path} not found — skipping {name}{RESET}")
            continue
        metrics[name] = load(path)

    if not metrics:
        print("No metrics files found. Train at least one model first.")
        return

    model_names = list(metrics.keys())
    divider = "=" * 72

    # ------------------------------------------------------------------
    print(f"\n{BOLD}{CYAN}{'ECG CLASSIFICATION — MODEL SUMMARY':^72}{RESET}")
    print(divider)

    # ------------------------------------------------------------------
    print(f"\n{BOLD}OVERALL METRICS{RESET}")
    headers   = ["Metric"] + model_names
    col_widths = [22] + [18] * len(model_names)

    accuracies = [metrics[m]["classification_report"]["accuracy"] for m in model_names]
    mac_f1s    = [metrics[m]["classification_report"]["macro avg"]["f1-score"] for m in model_names]
    mac_precs  = [metrics[m]["classification_report"]["macro avg"]["precision"] for m in model_names]
    mac_recs   = [metrics[m]["classification_report"]["macro avg"]["recall"] for m in model_names]
    wt_f1s     = [metrics[m]["classification_report"]["weighted avg"]["f1-score"] for m in model_names]
    times      = [metrics[m].get("training_time_seconds") for m in model_names]
    epochs     = [metrics[m].get("training_epochs") or metrics[m].get("best_iteration") for m in model_names]

    acc_fmt  = [f"{v:.4f}" for v in accuracies]
    mf1_fmt  = [f"{v:.4f}" for v in mac_f1s]
    mpr_fmt  = [f"{v:.4f}" for v in mac_precs]
    mre_fmt  = [f"{v:.4f}" for v in mac_recs]
    wf1_fmt  = [f"{v:.4f}" for v in wt_f1s]
    time_fmt = [fmt_time(t) for t in times]
    ep_fmt   = [str(e) if e is not None else "n/a" for e in epochs]

    rows = [
        ["Accuracy"]         + [highlight(v, acc_fmt)  for v in acc_fmt],
        ["Macro F1"]         + [highlight(v, mf1_fmt)  for v in mf1_fmt],
        ["Macro Precision"]  + [highlight(v, mpr_fmt)  for v in mpr_fmt],
        ["Macro Recall"]     + [highlight(v, mre_fmt)  for v in mre_fmt],
        ["Weighted F1"]      + [highlight(v, wf1_fmt)  for v in wf1_fmt],
        ["Training Time"]    + [highlight(v, time_fmt, higher_is_better=False) for v in time_fmt],
        ["Epochs / Rounds"]  + ep_fmt,
    ]
    print_table(headers, rows, col_widths)

    # ------------------------------------------------------------------
    print(f"\n{BOLD}PER-CLASS METRICS{RESET}")

    for cls in CLASSES:
        print(f"\n  {BOLD}{cls}{RESET}")
        headers    = ["  Metric"] + model_names
        col_widths = [22] + [18] * len(model_names)

        precs = [metrics[m]["classification_report"][cls]["precision"] for m in model_names]
        recs  = [metrics[m]["classification_report"][cls]["recall"]    for m in model_names]
        f1s   = [metrics[m]["classification_report"][cls]["f1-score"]  for m in model_names]
        sups  = [int(metrics[m]["classification_report"][cls]["support"]) for m in model_names]

        pr_fmt  = [f"{v:.4f}" for v in precs]
        re_fmt  = [f"{v:.4f}" for v in recs]
        f1_fmt  = [f"{v:.4f}" for v in f1s]
        sup_fmt = [str(s) for s in sups]

        rows = [
            ["  Precision"] + [highlight(v, pr_fmt) for v in pr_fmt],
            ["  Recall"]    + [highlight(v, re_fmt) for v in re_fmt],
            ["  F1-Score"]  + [highlight(v, f1_fmt) for v in f1_fmt],
            ["  Support"]   + sup_fmt,
        ]
        print_table(headers, rows, col_widths)

    print(f"\n{divider}\n")


if __name__ == "__main__":
    main()
