"""Utilities for loading test bed measurement outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from src.evaluation.types import TestOutput
from src.metrics import MetricResult
from src.utils import MetricMetadata


class MeasurementFormatError(ValueError):
    """Raised when measurement files do not follow the expected schema."""


def _coerce_entries(payload: object) -> List[Mapping[str, object]]:
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, Mapping)]
    if isinstance(payload, Mapping):
        # allow the root object to contain {"metrics": [...]}
        if "metrics" in payload and isinstance(payload["metrics"], list):
            return [entry for entry in payload["metrics"] if isinstance(entry, Mapping)]
        return [payload]
    raise MeasurementFormatError("Measurement payload must be a list or mapping of metric entries.")


def _resolve_metric_metadata(metric_id: str, metadata: Mapping[str, MetricMetadata]) -> MetricMetadata:
    if metric_id not in metadata:
        missing = ", ".join(sorted(metadata.keys()))
        raise MeasurementFormatError(f"Measurement references unknown metric '{metric_id}'. Known metrics: {missing}")
    return metadata[metric_id]


def load_measurements(
    tb_id: str,
    algorithm: str,
    metadata: Mapping[str, MetricMetadata],
    measurement_path: Path,
) -> TestOutput:
    """Load metric observations for a given test bed + algorithm pair.

    The measurement file must be JSON formatted and contain either:

    - A list of objects with ``metric_id`` and ``value`` fields.
    - A JSON object with a top-level ``metrics`` list following the same schema.

    Optional fields per entry:
    - ``unit``: overrides the default unit from configuration.
    - ``ci_low`` / ``ci_high``: confidence interval bounds.
    - ``extras``: arbitrary dictionary merged into the resulting :class:`MetricResult`.
    """

    if not measurement_path.exists():
        raise FileNotFoundError(f"Measurement file not found for {tb_id}/{algorithm}: {measurement_path}")

    with measurement_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    entries = _coerce_entries(payload)
    metric_results: List[MetricResult] = []

    for entry in entries:
        metric_id_raw = entry.get("metric_id") or entry.get("id")
        if not isinstance(metric_id_raw, str):
            raise MeasurementFormatError("Each metric entry must contain a 'metric_id' string.")
        metric_id = metric_id_raw.strip()
        raw_value = entry.get("value")
        if raw_value is None:
            raise MeasurementFormatError(f"Metric '{metric_id}' entry missing 'value'.")

        meta = _resolve_metric_metadata(metric_id, metadata)
        unit = entry.get("unit") if isinstance(entry.get("unit"), str) else meta.unit
        ci_low = entry.get("ci_low")
        ci_high = entry.get("ci_high")
        extras = entry.get("extras")
        if extras is None or not isinstance(extras, Mapping):
            extras = {}

        if isinstance(raw_value, (int, float)):
            value = float(raw_value)
        elif isinstance(raw_value, bool):
            value = float(raw_value)
        else:
            value = raw_value

        metric_results.append(
            MetricResult(
                metric_id=metric_id,
                name=meta.name,
                value=value,
                unit=unit,
                ci_low=float(ci_low) if isinstance(ci_low, (int, float)) else None,
                ci_high=float(ci_high) if isinstance(ci_high, (int, float)) else None,
                extras=dict(extras),
            )
        )

    context = {
        "testbed_id": tb_id,
        "algorithm": algorithm,
        "measurement_path": str(measurement_path),
    }
    return TestOutput(test_id=tb_id, metrics=metric_results, context=context)

