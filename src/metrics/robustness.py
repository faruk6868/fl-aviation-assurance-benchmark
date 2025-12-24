"""Security and robustness metrics (M21-M23, M27, M36)."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .base import MetricResult


def _make_result(metric_id: str, context: Mapping[str, object], value: float) -> MetricResult:
    metadata = context["metadata"]
    name = metadata.name if hasattr(metadata, "name") else metadata["name"]
    unit = metadata.unit if hasattr(metadata, "unit") else metadata["unit"]
    return MetricResult(metric_id=metric_id, name=name, value=float(value), unit=unit)


def metric_M21(context: Mapping[str, object]) -> MetricResult:
    attack = context.get("attack_metrics", {})
    if "detection_rate" not in attack:
        raise KeyError("attack_metrics.detection_rate required for M21")
    return _make_result("M21", context, float(attack["detection_rate"]))


def metric_M22(context: Mapping[str, object]) -> MetricResult:
    attack = context.get("attack_metrics", {})
    if "accuracy_clean" in attack and "accuracy_attack" in attack:
        ratio = float(attack["accuracy_attack"]) / max(float(attack["accuracy_clean"]), 1e-8)
    elif "robustness_ratio" in attack:
        ratio = float(attack["robustness_ratio"])
    else:
        raise KeyError("attack_metrics accuracy data required for M22")
    return _make_result("M22", context, ratio)


def metric_M23(context: Mapping[str, object]) -> MetricResult:
    attack = context.get("attack_metrics", {})
    if "update_cosine_similarities" not in attack:
        raise KeyError("attack_metrics.update_cosine_similarities required for M23")
    similarities = np.asarray(attack["update_cosine_similarities"], dtype=float)
    similarity = float(np.nanmean(similarities)) if similarities.size else float("nan")
    return _make_result("M23", context, similarity)


def metric_M27(context: Mapping[str, object]) -> MetricResult:
    attack = context.get("attack_metrics", {})
    if "attack_success_rate" not in attack:
        raise KeyError("attack_metrics.attack_success_rate required for M27")
    return _make_result("M27", context, float(attack["attack_success_rate"]))


def metric_M36(context: Mapping[str, object]) -> MetricResult:
    dropout = context.get("dropout_metrics")
    if dropout is None or "tolerance" not in dropout:
        raise KeyError("dropout_metrics.tolerance required for M36")
    return _make_result("M36", context, float(dropout["tolerance"]))

