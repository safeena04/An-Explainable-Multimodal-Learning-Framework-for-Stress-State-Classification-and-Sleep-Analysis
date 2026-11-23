"""Performance metrics for classification (stress) and regression (sleep)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)

CLASSIFICATION_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc"]


def classification_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    """Positive class = stress (1). Threshold fixed a priori at 0.5 (not tuned)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    pred = (p >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
    }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in out.items()}


def pairwise_accuracy(pids, y_true, prob) -> float:
    """Within-participant ranking accuracy.

    For each test participant who has both phases, is the stress record given a
    higher stress probability than the relax record? Ties count 0.5. This uses
    only test-set rows of the same (held-out) participant, so it is leakage-free,
    and it removes between-person baseline differences from the evaluation.
    """
    df = pd.DataFrame({"pid": pids, "y": np.asarray(y_true), "p": np.asarray(prob)})
    scores = []
    for _, g in df.groupby("pid"):
        if set(g["y"]) != {0, 1}:
            continue
        ps, pr = g.loc[g.y == 1, "p"].iloc[0], g.loc[g.y == 0, "p"].iloc[0]
        scores.append(1.0 if ps > pr else 0.5 if ps == pr else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def regression_metrics(y_true, y_pred) -> dict:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p))}


def summarize_folds(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Mean, SD and normal-approximation 95 % CI across outer folds."""
    rows = {}
    for m in metrics:
        v = df[m].dropna()
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        rows[m] = {"mean": v.mean(), "sd": v.std(ddof=1), "ci95_low": v.mean() - 1.96 * se,
                   "ci95_high": v.mean() + 1.96 * se, "n_folds": len(v)}
    return pd.DataFrame(rows).T
