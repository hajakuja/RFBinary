"""Numpy-only binary classification metrics for reporting/eval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y, x))


@dataclass
class BinaryMetrics:
    threshold: float
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    far: float
    confusion_matrix: Dict[str, int]


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def precision_recall_f1(cm: Dict[str, int]) -> Tuple[float, float, float]:
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    return float(precision), float(recall), float(f1)


def false_alarm_rate(cm: Dict[str, int]) -> float:
    fp, tn = cm["fp"], cm["tn"]
    return float(fp / (fp + tn + 1e-12))


def _binary_curve(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    score_sorted = y_score[order]

    thresholds = np.r_[np.inf, np.unique(score_sorted)]
    tps = np.zeros_like(thresholds, dtype=np.float64)
    fps = np.zeros_like(thresholds, dtype=np.float64)

    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)

    for i, thr in enumerate(thresholds):
        y_pred = (y_score >= thr).astype(np.int64)
        tps[i] = np.sum((y_pred == 1) & (y_true == 1))
        fps[i] = np.sum((y_pred == 1) & (y_true == 0))

    tpr = tps / (pos + 1e-12)
    fpr = fps / (neg + 1e-12)
    precision = tps / (tps + fps + 1e-12)
    recall = tpr
    return thresholds, fpr, tpr, precision, recall


def roc_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    _, fpr, tpr, _, _ = _binary_curve(y_true, y_score)
    order = np.argsort(fpr)
    return _trapezoid(tpr[order], fpr[order])


def pr_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    _, _, _, precision, recall = _binary_curve(y_true, y_score)
    order = np.argsort(recall)
    return _trapezoid(precision[order], recall[order])


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_far: float = 0.05) -> float:
    candidates = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_f1 = -1.0
    backup_thr = 0.5
    backup_far = 1e9

    for thr in candidates:
        pred = (y_prob >= thr).astype(np.int64)
        cm = confusion_counts(y_true, pred)
        far = false_alarm_rate(cm)
        _, _, f1 = precision_recall_f1(cm)

        if far < backup_far:
            backup_far = far
            backup_thr = float(thr)

        if far <= target_far and f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr if best_f1 >= 0 else backup_thr


def summarize_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> BinaryMetrics:
    y_pred = (y_prob >= threshold).astype(np.int64)
    cm = confusion_counts(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(cm)
    return BinaryMetrics(
        threshold=float(threshold),
        roc_auc=roc_auc_np(y_true, y_prob),
        pr_auc=pr_auc_np(y_true, y_prob),
        f1=f1,
        precision=precision,
        recall=recall,
        far=false_alarm_rate(cm),
        confusion_matrix=cm,
    )
