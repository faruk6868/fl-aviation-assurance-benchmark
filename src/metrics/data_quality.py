"""Distributional and data quality metrics (M19, M25, M26)."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.special import rel_entr

from .base import MetricResult


def _make_result(metric_id: str, context: Mapping[str, object], value: float) -> MetricResult:
    metadata = context["metadata"]
    name = metadata.name if hasattr(metadata, "name") else metadata["name"]
    unit = metadata.unit if hasattr(metadata, "unit") else metadata["unit"]
    return MetricResult(metric_id=metric_id, name=name, value=float(value), unit=unit)


def metric_M19(context: Mapping[str, object]) -> MetricResult:
    if "reference_distribution" not in context or "observed_distribution" not in context:
        raise KeyError("reference_distribution and observed_distribution required for M19")
    reference = np.asarray(context["reference_distribution"], dtype=float)
    observed = np.asarray(context["observed_distribution"], dtype=float)
    if reference.shape != observed.shape:
        raise ValueError("reference and observed distributions must match shape")
    reference = np.clip(reference, 1e-12, None)
    observed = np.clip(observed, 1e-12, None)
    reference /= reference.sum()
    observed /= observed.sum()
    kl = float(np.sum(rel_entr(observed, reference)))
    return _make_result("M19", context, kl)


def metric_M25(context: Mapping[str, object]) -> MetricResult:
    if "data_quality_score" not in context:
        raise KeyError("data_quality_score required for M25")
    return _make_result("M25", context, float(context["data_quality_score"]))


def metric_M26(context: Mapping[str, object]) -> MetricResult:
    if "odd_coverage" not in context:
        raise KeyError("odd_coverage required for M26")
    return _make_result("M26", context, float(context["odd_coverage"]))

