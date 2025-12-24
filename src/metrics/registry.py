"""Metric registry and dispatcher."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping

from .base import MetricComputationError, MetricResult
from . import communication, data_quality, drift, explainability, fairness, participation, performance, privacy, robustness, uncertainty


MetricFunction = Callable[[Mapping[str, Any]], MetricResult]

METRIC_ALIAS_MAP: Dict[str, str] = {
    "M.SAF.ACC": "M1",
    "M.CLS.Pre": "M2",
    "M.SAF.REC": "M3",
    "M.CLS.F1_Sc": "M4",
    "M.SAF.RMSE": "M5",
    "M.PRG.PHM": "M6",
    "M.CLS.AUC": "M7",
    "M.GEN.GAP": "M8",
    "M.FL.CONV_TIME": "M10",
    "M.FL.COMM_BYTES": "M11",
    "M.FL.COMM_COMP": "M12",
    "M.FL.PRIV_DP": "M20",
    "M.FL.ATTACK_DET": "M21",
    "M.FL.BYZ_TOL": "M22",
    "M.FL.ATTACK_RES": "M27",
    "M.DRIFT.KS": "M24",
    "M.DATA.ODD_COV": "M26",
    "M.STAT.CI": "M29",
    "M.RT.LAT": "M38",
}


class MetricRegistry:
    """Registers and computes metrics using metadata definitions."""

    def __init__(self, metadata: Mapping[str, Any]) -> None:
        self.metadata = metadata
        self._registry: Dict[str, MetricFunction] = {}

    def register(self, metric_id: str, func: MetricFunction) -> None:
        if metric_id in self._registry:
            raise ValueError(f"Metric {metric_id} already registered")
        self._registry[metric_id] = func

    def compute(self, metric_ids: Iterable[str], context: Mapping[str, Any]) -> List[MetricResult]:
        results: List[MetricResult] = []
        for metric_id in metric_ids:
            if metric_id not in self._registry:
                raise MetricComputationError(f"Metric {metric_id} not registered")
            metadata = self.metadata[metric_id]
            func = self._registry[metric_id]
            result = func({**context, "metadata": metadata})
            if result.metric_id != metric_id:
                raise MetricComputationError(
                    f"Metric {metric_id} returned result with mismatched id {result.metric_id}"
                )
            results.append(result)
        return results

    def registered_metric_ids(self) -> set[str]:
        """Return the set of metric identifiers registered in this registry."""

        return set(self._registry.keys())

    def register_alias(self, alias_id: str, target_id: str) -> None:
        if alias_id in self._registry:
            raise ValueError(f"Alias {alias_id} already registered")
        if target_id not in self._registry:
            raise ValueError(f"Cannot create alias for unknown metric {target_id}")

        target_func = self._registry[target_id]
        alias_metadata = self.metadata.get(alias_id)
        target_metadata = self.metadata.get(target_id)

        def alias_func(context: Mapping[str, Any]) -> MetricResult:
            wrapped_context = dict(context)
            if alias_metadata is not None:
                wrapped_context["metadata"] = alias_metadata
            result = target_func(wrapped_context)

            def _resolve(attr: str, meta: Any | None) -> Any | None:
                if meta is None:
                    return None
                if hasattr(meta, attr):
                    return getattr(meta, attr)
                return meta.get(attr)

            name = _resolve("name", alias_metadata) or _resolve("name", target_metadata) or result.name
            unit = _resolve("unit", alias_metadata) or _resolve("unit", target_metadata) or result.unit

            return MetricResult(
                metric_id=alias_id,
                name=name,
                value=result.value,
                unit=unit,
                ci_low=result.ci_low,
                ci_high=result.ci_high,
                extras=result.extras,
            )

        self._registry[alias_id] = alias_func


def create_default_registry(metadata: Mapping[str, Any]) -> MetricRegistry:
    registry = MetricRegistry(metadata)

    registry.register("M1", performance.metric_M1)
    registry.register("M2", performance.metric_M2)
    registry.register("M3", performance.metric_M3)
    registry.register("M4", performance.metric_M4)
    registry.register("M5", performance.metric_M5)
    registry.register("M6", performance.metric_M6)
    registry.register("M7", performance.metric_M7)
    registry.register("M8", performance.metric_M8)
    registry.register("M9", communication.metric_M9)
    registry.register("M10", communication.metric_M10)
    registry.register("M11", communication.metric_M11)
    registry.register("M12", communication.metric_M12)
    registry.register("M13", fairness.metric_M13)
    registry.register("M14", fairness.metric_M14)
    registry.register("M15", fairness.metric_M15)
    registry.register("M16", fairness.metric_M16)
    registry.register("M17", fairness.metric_M17)
    registry.register("M18", participation.metric_M18)
    registry.register("M19", data_quality.metric_M19)
    registry.register("M20", privacy.metric_M20)
    registry.register("M21", robustness.metric_M21)
    registry.register("M22", robustness.metric_M22)
    registry.register("M23", robustness.metric_M23)
    registry.register("M24", drift.metric_M24)
    registry.register("M25", data_quality.metric_M25)
    registry.register("M26", data_quality.metric_M26)
    registry.register("M27", robustness.metric_M27)
    registry.register("M28", uncertainty.metric_M28)
    registry.register("M29", uncertainty.metric_M29)
    registry.register("M30", communication.metric_M30)
    registry.register("M31", explainability.metric_M31)
    registry.register("M32", explainability.metric_M32)
    registry.register("M33", explainability.metric_M33)
    registry.register("M34", explainability.metric_M34)
    registry.register("M35", fairness.metric_M35)
    registry.register("M36", robustness.metric_M36)
    registry.register("M37", communication.metric_M37)
    registry.register("M38", communication.metric_M38)
    registry.register("M39", performance.metric_M39)
    registry.register("M40", performance.metric_M40)

    for alias_id, target_id in METRIC_ALIAS_MAP.items():
        if alias_id in metadata and target_id in registry._registry:
            try:
                registry.register_alias(alias_id, target_id)
            except ValueError:
                continue

    return registry

