"""Privacy-related metrics."""

from __future__ import annotations

from typing import Mapping

from .base import MetricResult


def metric_M20(context: Mapping[str, object]) -> MetricResult:
    if "privacy_epsilon" not in context:
        raise KeyError("privacy_epsilon required for M20")
    metadata = context["metadata"]
    name = metadata.name if hasattr(metadata, "name") else metadata["name"]
    unit = metadata.unit if hasattr(metadata, "unit") else metadata["unit"]
    return MetricResult(metric_id="M20", name=name, value=float(context["privacy_epsilon"]), unit=unit)

