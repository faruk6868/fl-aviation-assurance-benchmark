"""Explainability and traceability metrics (M31-M34)."""

from __future__ import annotations

from typing import Mapping

from .base import MetricResult


def _make_result(metric_id: str, context: Mapping[str, object], value: float, extras: dict | None = None) -> MetricResult:
    metadata = context["metadata"]
    name = metadata.name if hasattr(metadata, "name") else metadata["name"]
    unit = metadata.unit if hasattr(metadata, "unit") else metadata["unit"]
    return MetricResult(metric_id=metric_id, name=name, value=float(value), unit=unit, extras=extras or {})


def metric_M31(context: Mapping[str, object]) -> MetricResult:
    if "explainability_score" not in context:
        raise KeyError("explainability_score required for M31")
    extras = {"top_features": context.get("top_features")}
    return _make_result("M31", context, float(context["explainability_score"]), extras)


def metric_M32(context: Mapping[str, object]) -> MetricResult:
    if "attribution_stability" not in context:
        raise KeyError("attribution_stability required for M32")
    return _make_result("M32", context, float(context["attribution_stability"]))


def metric_M33(context: Mapping[str, object]) -> MetricResult:
    if "version_consistency" not in context:
        raise KeyError("version_consistency required for M33")
    extras = {"reference_version": context.get("reference_version"), "current_version": context.get("current_version")}
    return _make_result("M33", context, float(context["version_consistency"]), extras)


def metric_M34(context: Mapping[str, object]) -> MetricResult:
    if "reproducibility_score" not in context:
        raise KeyError("reproducibility_score required for M34")
    extras = {"replication_runs": context.get("replication_runs")}
    return _make_result("M34", context, float(context["reproducibility_score"]), extras)

