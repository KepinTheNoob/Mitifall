from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

COUNT_KEYS = ("tp", "fp", "tn", "fn")
METRIC_KEYS = (
    "accuracy",
    "f1_macro",
    "f1_fall",
    "sensitivity",
    "specificity",
    "precision",
    "balanced_accuracy",
)

def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    t = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    return np.array([(t & p).sum(), (~t & p).sum(), (~t & ~p).sum(), (t & ~p).sum()], dtype=np.int64)

def _ratio(num: float, den: float) -> float:
    return float(num) / den if den else float("nan")

def _f1(precision: float, recall: float) -> float:
    if np.isnan(precision) or np.isnan(recall):
        return float("nan")
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)

def metrics_from_counts(counts: np.ndarray) -> dict[str, float]:
    tp, fp, tn, fn = (float(v) for v in counts)
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    precision = _ratio(tp, tp + fp)
    npv = _ratio(tn, tn + fn)
    f1_fall = _f1(precision, sensitivity)
    f1_adl = _f1(npv, specificity)
    return {
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "f1_macro": 0.5 * (f1_fall + f1_adl),
        "f1_fall": f1_fall,
        "f1_adl": f1_adl,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
    }

def threshold_free(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    if y.min() == y.max():
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
    }

def select_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    y = np.asarray(y_true).astype(bool)
    if y.all() or not y.any():
        return 0.5
    grid = np.linspace(0.02, 0.98, 97)
    pred = np.asarray(proba)[None, :] >= grid[:, None]
    sens = (pred & y).sum(axis=1) / y.sum()
    spec = (~pred & ~y).sum(axis=1) / (~y).sum()
    j = sens + spec - 1.0
    best = np.flatnonzero(np.isclose(j, j.max()))
    return float(grid[best[np.argmin(np.abs(grid[best] - 0.5))]])
