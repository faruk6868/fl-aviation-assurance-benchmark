"""Calibration and uncertainty metrics (M28, M29)."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .base import MetricResult


def _make_result(metric_id: str, context: Mapping[str, object], value: float, extras: dict | None = None) -> MetricResult:
    metadata = context["metadata"]
    name = metadata.name if hasattr(metadata, "name") else metadata["name"]
    unit = metadata.unit if hasattr(metadata, "unit") else metadata["unit"]
    return MetricResult(metric_id=metric_id, name=name, value=float(value), unit=unit, extras=extras or {})


def metric_M28(context: Mapping[str, object]) -> MetricResult:
    if "labels_binary" not in context:
        raise KeyError("labels_binary required for calibration error")
    if "y_prob" not in context:
        raise KeyError("y_prob required for calibration error")
    y_true = np.asarray(context["labels_binary"], dtype=int)
    y_prob = np.asarray(context["y_prob"], dtype=float)

    num_bins = int(context.get("ece_bins", 10))
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    ece = 0.0
    bin_stats = {}
    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        if not np.any(mask):
            continue
        prob_avg = y_prob[mask].mean()
        acc_avg = y_true[mask].mean()
        weight = mask.mean()
        ece += abs(prob_avg - acc_avg) * weight
        bin_stats[bin_idx] = {
            "prob_mean": float(prob_avg),
            "acc_mean": float(acc_avg),
            "weight": float(weight),
        }
    return _make_result("M28", context, float(ece), {"bin_stats": bin_stats})


def metric_M29(context: Mapping[str, object]) -> MetricResult:
    if "confidence_widths" in context:
        widths = np.asarray(context["confidence_widths"], dtype=float)
        width = float(np.nanmean(widths)) if widths.size else float("nan")
    elif "prediction_intervals" in context:
        intervals = np.asarray(context["prediction_intervals"], dtype=float)
        if intervals.shape[1] != 2:
            raise ValueError("prediction_intervals must be of shape (n, 2)")
        width = float(np.nanmean(intervals[:, 1] - intervals[:, 0]))
    else:
        raise KeyError("confidence_widths or prediction_intervals required for M29")

    lead_time = context.get("lead_time_cycles")
    reference: float | None = None
    if isinstance(lead_time, (int, float)) and float(lead_time) > 0:
        reference = float(lead_time)
    elif "rul_true" in context:
        rul_true = np.asarray(context["rul_true"], dtype=float)
        if rul_true.size:
            span = float(np.ptp(rul_true))
            reference = span if span > 0 else float(np.mean(np.abs(rul_true)))

    if reference is None or not np.isfinite(reference) or reference <= 0:
        reference = 1.0

    normalised_width = width / reference if np.isfinite(width) else float("nan")
    return _make_result("M29", context, normalised_width, {"raw_width": width, "reference_scale": reference})

