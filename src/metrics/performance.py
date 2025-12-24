"""Performance-related metric computations (M1-M8, M39, M40)."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .base import MetricResult, bootstrap_ci, ensure_arrays, proportion_ci, safe_divide


def _extract_binary_arrays(context: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    if "labels_binary" not in context:
        raise KeyError("labels_binary missing from metric context")
    y_true = np.asarray(context["labels_binary"], dtype=int)

    if "pred_binary" in context:
        y_pred = np.asarray(context["pred_binary"], dtype=int)
    elif "rul_pred" in context and "lead_time_cycles" in context:
        rul_pred = np.asarray(context["rul_pred"], dtype=float)
        lead_time = float(context["lead_time_cycles"])
        y_pred = (rul_pred <= lead_time).astype(int)
    else:
        raise KeyError("Neither pred_binary nor (rul_pred, lead_time_cycles) provided")

    return y_true, y_pred


def _make_result(metric_id: str, context: Mapping[str, object], value: float, ci: tuple[float, float] | None = None) -> MetricResult:
    metadata = context["metadata"]
    name = metadata.name if hasattr(metadata, "name") else metadata["name"]
    unit = metadata.unit if hasattr(metadata, "unit") else metadata["unit"]
    ci_low, ci_high = (ci if ci else (None, None))
    return MetricResult(metric_id=metric_id, name=name, value=float(value), unit=unit, ci_low=ci_low, ci_high=ci_high)


def metric_M1(context: Mapping[str, object]) -> MetricResult:
    y_true, y_pred = _extract_binary_arrays(context)
    accuracy = np.mean(y_true == y_pred)
    ci = proportion_ci(int((y_true == y_pred).sum()), len(y_true))
    return _make_result("M1", context, accuracy, ci)


def metric_M2(context: Mapping[str, object]) -> MetricResult:
    y_true, y_pred = _extract_binary_arrays(context)
    precision = precision_score(y_true, y_pred, zero_division=0)

    def prec_fn(a: np.ndarray, b: np.ndarray) -> float:
        return precision_score(a, b, zero_division=0)

    ci = bootstrap_ci(prec_fn, y_true, y_pred)
    return _make_result("M2", context, precision, ci)


def metric_M3(context: Mapping[str, object]) -> MetricResult:
    y_true, y_pred = _extract_binary_arrays(context)
    recall = recall_score(y_true, y_pred, zero_division=0)

    def rec_fn(a: np.ndarray, b: np.ndarray) -> float:
        return recall_score(a, b, zero_division=0)

    ci = bootstrap_ci(rec_fn, y_true, y_pred)
    return _make_result("M3", context, recall, ci)


def metric_M4(context: Mapping[str, object]) -> MetricResult:
    y_true, y_pred = _extract_binary_arrays(context)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    def f1_fn(a: np.ndarray, b: np.ndarray) -> float:
        return f1_score(a, b, zero_division=0)

    ci = bootstrap_ci(f1_fn, y_true, y_pred)
    return _make_result("M4", context, f1, ci)


def metric_M39(context: Mapping[str, object]) -> MetricResult:
    y_true, y_pred = _extract_binary_arrays(context)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    negatives = int((y_true == 0).sum())
    fpr = safe_divide(fp, negatives, default=0.0)
    ci = proportion_ci(fp, negatives) if negatives > 0 else (np.nan, np.nan)
    return _make_result("M39", context, fpr, ci)


def metric_M40(context: Mapping[str, object]) -> MetricResult:
    y_true, y_pred = _extract_binary_arrays(context)
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    positives = int((y_true == 1).sum())
    fnr = safe_divide(fn, positives, default=0.0)
    ci = proportion_ci(fn, positives) if positives > 0 else (np.nan, np.nan)
    return _make_result("M40", context, fnr, ci)


def metric_M5(context: Mapping[str, object]) -> MetricResult:
    if "rul_true" not in context or "rul_pred" not in context:
        raise KeyError("rul_true or rul_pred missing from context")
    rul_true, rul_pred = ensure_arrays(context["rul_true"], context["rul_pred"])
    rmse = float(np.sqrt(np.mean((rul_true - rul_pred) ** 2)))

    def rmse_fn(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    ci = bootstrap_ci(rmse_fn, rul_true, rul_pred)
    return _make_result("M5", context, rmse, ci)


def metric_M6(context: Mapping[str, object]) -> MetricResult:
    if "rul_true" not in context or "rul_pred" not in context:
        raise KeyError("rul_true or rul_pred missing from context")
    rul_true, rul_pred = ensure_arrays(context["rul_true"], context["rul_pred"])
    errors = rul_pred - rul_true
    score = 0.0
    for e in errors:
        if e < 0:
            score += np.exp(-e / 13) - 1
        else:
            score += np.exp(e / 10) - 1
    score = float(score)

    def score_fn(a: np.ndarray, b: np.ndarray) -> float:
        errs = b - a
        value = 0.0
        for err in errs:
            if err < 0:
                value += np.exp(-err / 13) - 1
            else:
                value += np.exp(err / 10) - 1
        return float(value)

    ci = bootstrap_ci(score_fn, rul_true, rul_pred)
    return _make_result("M6", context, score, ci)


def metric_M7(context: Mapping[str, object]) -> MetricResult:
    y_true, _ = _extract_binary_arrays(context)
    if "y_score" in context:
        y_score = np.asarray(context["y_score"], dtype=float)
    elif "rul_pred" in context:
        rul_pred = np.asarray(context["rul_pred"], dtype=float)
        y_score = -rul_pred
    else:
        raise KeyError("Neither y_score nor rul_pred provided for AUC")

    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")

    def auc_fn(a: np.ndarray, b: np.ndarray) -> float:
        if len(np.unique(a)) < 2:
            return float("nan")
        return float(roc_auc_score(a, b))

    ci = bootstrap_ci(auc_fn, y_true, y_score)
    return _make_result("M7", context, auc, ci)


def metric_M8(context: Mapping[str, object]) -> MetricResult:
    if "train_metric" not in context or "val_metric" not in context:
        raise KeyError("train_metric or val_metric missing for generalization gap")
    train_metric = float(context["train_metric"])
    val_metric = float(context["val_metric"])

    # Historical pipeline versions stored mean-squared error values. Convert to RMSE so
    # gap comparisons align with threshold definitions (which expect units matching M5).
    train_rmse = float(np.sqrt(train_metric)) if train_metric >= 0 else float("nan")
    val_rmse = float(np.sqrt(val_metric)) if val_metric >= 0 else float("nan")

    if not np.isfinite(train_rmse) or not np.isfinite(val_rmse):
        gap = float("nan")
    else:
        baseline = max(val_rmse, 1e-6)
        gap = abs(train_rmse - val_rmse) / baseline

    return _make_result("M8", context, gap, None)

