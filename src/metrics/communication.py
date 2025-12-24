"""Communication and convergence related metrics."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .base import MetricResult, safe_divide


def _make_result(metric_id: str, context: Mapping[str, object], value: float) -> MetricResult:
    metadata = context["metadata"]
    name = metadata.name if hasattr(metadata, "name") else metadata["name"]
    unit = metadata.unit if hasattr(metadata, "unit") else metadata["unit"]
    return MetricResult(metric_id=metric_id, name=name, value=float(value), unit=unit)


def metric_M9(context: Mapping[str, object]) -> MetricResult:
    history = context.get("validation_loss_history")
    if history is None or len(history) < 2:
        raise KeyError("validation_loss_history required for M9")
    history = np.asarray(history, dtype=float)
    deltas = np.abs(np.diff(history))
    rate = float(np.mean(deltas))
    return _make_result("M9", context, rate)


def metric_M10(context: Mapping[str, object]) -> MetricResult:
    if "rounds_to_convergence" not in context:
        raise KeyError("rounds_to_convergence missing for M10")
    return _make_result("M10", context, float(context["rounds_to_convergence"]))


def metric_M11(context: Mapping[str, object]) -> MetricResult:
    if "communication_history" in context:
        history = context["communication_history"]
        bytes_per_round = [entry.get("bytes_sent", 0.0) + entry.get("bytes_received", 0.0) for entry in history]
        mb_per_round = float(np.mean(bytes_per_round) / (1024 ** 2))
    elif "mb_per_round" in context:
        mb_per_round = float(context["mb_per_round"])
    else:
        raise KeyError("communication history or mb_per_round required for M11")
    return _make_result("M11", context, mb_per_round)


def metric_M12(context: Mapping[str, object]) -> MetricResult:
    if "efficiency_ratio" in context:
        ratio = float(context["efficiency_ratio"])
    else:
        accuracy = float(context.get("global_accuracy", np.nan))
        mb_per_round = float(context.get("mb_per_round", np.nan))
        ratio = safe_divide(accuracy, mb_per_round)
    return _make_result("M12", context, ratio)


def metric_M30(context: Mapping[str, object]) -> MetricResult:
    if "time_to_convergence_s" not in context:
        raise KeyError("time_to_convergence_s required for M30")
    return _make_result("M30", context, float(context["time_to_convergence_s"]))


def metric_M37(context: Mapping[str, object]) -> MetricResult:
    if "memory_mb" not in context:
        raise KeyError("memory_mb required for M37")
    return _make_result("M37", context, float(context["memory_mb"]))


def metric_M38(context: Mapping[str, object]) -> MetricResult:
    if "latency_ms" not in context:
        raise KeyError("latency_ms required for M38")
    return _make_result("M38", context, float(context["latency_ms"]))

