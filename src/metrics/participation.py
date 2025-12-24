"""Participation monitoring metrics (M18)."""

from __future__ import annotations

from typing import Mapping

from .base import MetricResult


def metric_M18(context: Mapping[str, object]) -> MetricResult:
    contributions = context.get("client_contributions")
    if contributions is None:
        raise KeyError("client_contributions required for M18")
    metadata = context["metadata"]
    name = metadata.name if hasattr(metadata, "name") else metadata["name"]
    unit = metadata.unit if hasattr(metadata, "unit") else metadata["unit"]
    return MetricResult(
        metric_id="M18",
        name=name,
        value=0.0,
        unit=unit,
        extras={"client_contributions": contributions},
    )

