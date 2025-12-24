"""Concept drift metric (M24)."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.stats import ks_2samp

from .base import MetricResult


def metric_M24(context: Mapping[str, object]) -> MetricResult:
    baseline = context.get("drift_baseline")
    current = context.get("drift_current")
    if baseline is None or current is None:
        raise KeyError("drift_baseline and drift_current required for M24")
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    statistic, p_value = ks_2samp(baseline, current)
    metadata = context["metadata"]
    name = metadata.name if hasattr(metadata, "name") else metadata["name"]
    unit = metadata.unit if hasattr(metadata, "unit") else metadata["unit"]
    return MetricResult(metric_id="M24", name=name, value=float(p_value), unit=unit, extras={"ks_statistic": float(statistic)})

