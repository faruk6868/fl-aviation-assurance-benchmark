"""Fairness-related metrics (M13-M17, M35)."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Mapping

import numpy as np

from .base import MetricResult


def _make_result(metric_id: str, context: Mapping[str, object], value: float, extras: Dict[str, object] | None = None) -> MetricResult:
    metadata = context["metadata"]
    name = metadata.name if hasattr(metadata, "name") else metadata["name"]
    unit = metadata.unit if hasattr(metadata, "unit") else metadata["unit"]
    return MetricResult(metric_id=metric_id, name=name, value=float(value), unit=unit, extras=extras or {})


def _get_group_arrays(context: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "labels_binary" not in context or "pred_binary" not in context or "group_labels" not in context:
        raise KeyError("labels_binary, pred_binary, and group_labels required for fairness metrics")
    y_true = np.asarray(context["labels_binary"], dtype=int)
    y_pred = np.asarray(context["pred_binary"], dtype=int)
    groups = np.asarray(context["group_labels"], dtype=int)
    return y_true, y_pred, groups


def _split_by_group(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> Dict[int, Dict[str, float]]:
    stats: Dict[int, Dict[str, float]] = {}
    for group in np.unique(groups):
        mask = groups == group
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        positives = (y_true_g == 1).sum()
        negatives = (y_true_g == 0).sum()
        tp = ((y_true_g == 1) & (y_pred_g == 1)).sum()
        fp = ((y_true_g == 0) & (y_pred_g == 1)).sum()
        stats[int(group)] = {
            "size": int(mask.sum()),
            "positive_rate": float((y_pred_g == 1).mean()),
            "tp_rate": float(tp / positives) if positives > 0 else 0.0,
            "fp_rate": float(fp / negatives) if negatives > 0 else 0.0,
            "accuracy": float((y_true_g == y_pred_g).mean()),
        }
    return stats


def metric_M13(context: Mapping[str, object]) -> MetricResult:
    client_metrics = context.get("client_metrics")
    if not client_metrics:
        raise KeyError("client_metrics required for M13")
    accuracies = np.array([metrics.get("accuracy", np.nan) for metrics in client_metrics.values()], dtype=float)
    variance = float(np.nanvar(accuracies, ddof=1))
    extras = {"per_client_accuracy": client_metrics}
    return _make_result("M13", context, variance, extras)


def metric_M14(context: Mapping[str, object]) -> MetricResult:
    client_metrics = context.get("client_metrics")
    if not client_metrics:
        raise KeyError("client_metrics required for M14")
    accuracies = np.array([metrics.get("accuracy", np.nan) for metrics in client_metrics.values()], dtype=float)
    std_dev = float(np.nanstd(accuracies, ddof=1))
    extras = {"per_client_accuracy": client_metrics}
    return _make_result("M14", context, std_dev, extras)


def metric_M35(context: Mapping[str, object]) -> MetricResult:
    client_metrics = context.get("client_metrics")
    if not client_metrics:
        raise KeyError("client_metrics required for M35")
    accuracies = np.array([metrics.get("accuracy", np.nan) for metrics in client_metrics.values()], dtype=float)
    variance = float(np.nanvar(accuracies, ddof=1))
    extras = {"per_client_accuracy": client_metrics}
    return _make_result("M35", context, variance, extras)


def metric_M15(context: Mapping[str, object]) -> MetricResult:
    y_true, y_pred, groups = _get_group_arrays(context)
    stats = _split_by_group(y_true, y_pred, groups)
    parity_values = [group_stats["positive_rate"] for group_stats in stats.values()]
    spd = float(np.max(parity_values) - np.min(parity_values)) if parity_values else float("nan")
    extras = {"group_stats": stats, "worst_group_positive_rate": float(np.max(parity_values)) if parity_values else float("nan")}
    return _make_result("M15", context, spd, extras)


def metric_M16(context: Mapping[str, object]) -> MetricResult:
    y_true, y_pred, groups = _get_group_arrays(context)
    stats = _split_by_group(y_true, y_pred, groups)
    tpr_values = [group_stats["tp_rate"] for group_stats in stats.values()]
    eod = float(np.max(tpr_values) - np.min(tpr_values)) if tpr_values else float("nan")
    extras = {"group_stats": stats, "worst_group_tpr": float(np.min(tpr_values)) if tpr_values else float("nan")}
    return _make_result("M16", context, eod, extras)


def metric_M17(context: Mapping[str, object]) -> MetricResult:
    y_true, y_pred, groups = _get_group_arrays(context)
    stats = _split_by_group(y_true, y_pred, groups)
    recalls = np.array([group_stats["tp_rate"] for group_stats in stats.values()], dtype=float)
    if len(recalls) == 0:
        jfi = float("nan")
    else:
        numerator = np.square(np.sum(recalls))
        denominator = len(recalls) * np.sum(np.square(recalls))
        jfi = float(numerator / denominator) if denominator > 0 else float("nan")
    extras = {"group_stats": stats}
    return _make_result("M17", context, jfi, extras)

