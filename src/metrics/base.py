"""Base classes and utilities for metric computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import numpy as np


class MetricComputationError(RuntimeError):
    """Raised when a metric cannot be computed with the provided context."""


@dataclass
class MetricResult:
    metric_id: str
    name: str
    value: float | int | str
    unit: str
    ci_low: float | None = None
    ci_high: float | None = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "Metric_ID": self.metric_id,
            "Metric_Name": self.name,
            "Observed_Value": self.value,
            "Unit": self.unit,
        }
        if self.ci_low is not None and self.ci_high is not None:
            payload["CI_Lower"] = self.ci_low
            payload["CI_Upper"] = self.ci_high
        payload.update(self.extras)
        return payload


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_bootstrap: int = 500,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a metric that depends on sample pairs."""

    if len(y_true) == 0:
        raise MetricComputationError("Cannot compute CI with no samples")

    rng = np.random.default_rng(seed)
    estimates = []
    n = len(y_true)
    for _ in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        estimates.append(metric_fn(y_true[idx], y_pred[idx]))
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def proportion_ci(successes: int, total: int, method: str = "wilson", alpha: float = 0.05) -> tuple[float, float]:
    """Return confidence interval for a binomial proportion."""

    if total == 0:
        return (np.nan, np.nan)

    try:
        from statsmodels.stats.proportion import proportion_confint
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise MetricComputationError("statsmodels required for proportion CI") from exc

    low, high = proportion_confint(successes, total, alpha=alpha, method=method)
    return float(low), float(high)


def safe_divide(numerator: float, denominator: float, default: float = np.nan) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def ensure_arrays(*arrays: Iterable[Any]) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(arr) for arr in arrays)

