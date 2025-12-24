"""Metric threshold assessment and pass/fail determination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import math

import numpy as np
import pandas as pd

from src.metrics import MetricResult
from src.utils import MetricMetadata, RequirementMapping

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from src.evaluation.types import TestOutput


LEVEL_SEQUENCE: Sequence[Tuple[str, str]] = (
    ("Catastrophic", "Cat"),
    ("Hazardous", "Haz"),
    ("Major", "Maj"),
)

FHA_METRICS = {"M.SAF.ACC", "M.SAF.REC", "M.SAF.RMSE", "M.SAF.FPFN"}

COMPOSITE_METRICS = {
    "M.FL.PRIV_DP": ["M.FL.PRIV_EPS", "M.FL.PRIV_DELTA"],
    "M.SAF.FPFN": ["M.SAF.FPR_MAX", "M.SAF.FNR_MAX"],
    "M.XAI.FIDE_STAB": ["M.XAI.FIDELITY", "M.XAI.STABILITY"],
}


HAZARD_MITIGATIONS: Dict[str, str] = {
    "fn": "Increase recall through threshold tuning or focused retraining on failure cases.",
    "fp": "Recalibrate decision thresholds or adjust loss weighting to curb false alarms.",
    "fn_fp": "Balance precision/recall via class weighting or adaptive thresholds.",
    "delay": "Improve RUL estimator accuracy or extend the proactive maintenance lead time.",
    "disc": "Strengthen discriminative performance (e.g., richer features or longer training).",
    "overfit": "Apply regularisation or data augmentation to shrink the generalisation gap.",
    "staleness": "Increase aggregation frequency or shorten local epochs to reduce staleness.",
    "availability": "Optimise communication budget and scheduling (compression, partial participation).",
    "fairness": "Apply fairness-aware aggregation or reweight under-performing client groups.",
    "privacy": "Tune DP noise/clipping or adopt privacy amplification to meet epsilon bounds.",
    "security": "Enable robust aggregation and investigate anomalous client behaviours.",
    "drift": "Trigger drift response plan and recalibrate with recent operational data.",
    "data_quality": "Audit and cleanse data pipelines; recover missing or inconsistent records.",
    "coverage": "Collect additional data for uncovered operating regimes to raise ODD coverage.",
    "calibration": "Recalibrate probabilities (e.g., temperature scaling) to tighten calibration error.",
    "uncertainty": "Refine predictive intervals via ensembling or Bayesian calibration.",
    "traceability": "Tighten configuration management and reproducibility controls.",
    "assurance": "Review explainability pipeline and refresh attribution baselines.",
    "participation": "Monitor client participation balance and incentivise under-represented fleets.",
    "distribution": "Mitigate client heterogeneity via reweighting or shared feature normalisation.",
    "responsiveness": "Profile inference path and deploy optimisations to cut latency.",
}


@dataclass
class MetricAssessmentRecord:
    """Structured record summarising threshold comparison for a single metric."""

    test_id: str
    metric_id: str
    metric_name: str
    observed_value: float
    unit: str
    min_threshold: float | None
    max_threshold: float | None
    direction: str
    pass_level: str
    status: str
    comment: str
    ci_low: float | None = None
    ci_high: float | None = None
    hazard_link: str | None = None
    category: str | None = None
    regulatory_requirements: str | None = None
    assumptions: str | None = None
    method: str | None = None
    rationale: str | None = None
    optimal_target: str | None = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "Test": self.test_id,
            "Metric_ID": self.metric_id,
            "Metric_Name": self.metric_name,
            "Observed_Value": self.observed_value,
            "Unit": self.unit,
            "Direction": self.direction,
            "Pass_Level": self.pass_level,
            "Status": self.status,
            "Comment": self.comment,
            "Min_Threshold": self.min_threshold,
            "Max_Threshold": self.max_threshold,
            "Hazard_Link": self.hazard_link,
            "Category": self.category,
            "Regulatory_Requirements": self.regulatory_requirements,
            "Method": self.method,
            "Assumptions": self.assumptions,
            "Rationale": self.rationale,
            "Optimal_Target": self.optimal_target,
        }
        if self.ci_low is not None and self.ci_high is not None:
            payload["CI_Lower"] = self.ci_low
            payload["CI_Upper"] = self.ci_high
        return payload


class AssuranceEvaluator:
    """Compare metric observations against Cat/Haz/Maj thresholds."""

    def __init__(
        self,
        metadata: Mapping[str, MetricMetadata],
        thresholds: Mapping[str, pd.DataFrame],
        requirements: Mapping[str, RequirementMapping],
    ) -> None:
        self.metadata = metadata
        self.thresholds = thresholds
        self.requirements = requirements

    def evaluate_test(self, test: "TestOutput") -> List[MetricAssessmentRecord]:
        records: List[MetricAssessmentRecord] = []
        for result in test.metrics:
            record = self._assess_metric(test.test_id, result)
            records.append(record)
        self._apply_composite_dependencies(records)
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _assess_metric(self, test_id: str, result: MetricResult) -> MetricAssessmentRecord:
        metric_id = result.metric_id
        metadata = self._get_metadata(metric_id)
        extras = result.extras or {}

        self._validate_metric_extras(metric_id, extras)

        direction = metadata.direction.lower()

        range_row = self._collect_threshold_range(metric_id)
        min_value = self._coerce_optional_float(range_row.get("Threshold_Min")) if range_row is not None else None
        max_value = self._coerce_optional_float(range_row.get("Threshold_Max")) if range_row is not None else None
        assumptions = range_row.get("Assumptions") if range_row is not None else None
        method = range_row.get("Method") if range_row is not None else None
        rationale = range_row.get("Rationale") if range_row is not None else None
        enforce = bool(range_row.get("Enforce")) if range_row is not None else False

        value = self._coerce_float(result.value)
        ci_low = self._coerce_optional_float(result.ci_low)
        ci_high = self._coerce_optional_float(result.ci_high)

        status, pass_level, detail = self._evaluate_against_range(value, min_value, max_value, enforce)

        hazard_link = metadata.hazard_link
        comment = self._build_comment(
            metric_id=metric_id,
            direction=direction,
            observed=value,
            status=status,
            pass_level=pass_level,
            detail=detail,
            extras=extras,
            hazard_link=hazard_link,
        )

        requirements_info = self.requirements.get(metric_id, RequirementMapping())

        return MetricAssessmentRecord(
            test_id=test_id,
            metric_id=metric_id,
            metric_name=metadata.name,
            observed_value=value,
            unit=metadata.unit,
            min_threshold=min_value,
            max_threshold=max_value,
            direction=direction,
            pass_level=pass_level,
            status=status,
            comment=comment,
            ci_low=ci_low,
            ci_high=ci_high,
            hazard_link=hazard_link,
            category=requirements_info.category,
            regulatory_requirements=requirements_info.requirements,
            assumptions=assumptions,
            method=method,
            rationale=rationale,
            optimal_target=requirements_info.optimal_target,
        )

    def _collect_threshold_range(self, metric_id: str) -> pd.Series | None:
        table = self.thresholds.get("Ranges")
        if table is None or metric_id not in table.index:
            return None
        row = table.loc[metric_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row
    def _coerce_float(self, value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def _coerce_optional_float(self, value: object) -> float | None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _evaluate_against_range(
        self,
        value: float,
        min_threshold: float | None,
        max_threshold: float | None,
        enforce: bool,
    ) -> Tuple[str, str, Dict[str, object]]:
        if not enforce:
            return "MONITOR", "Monitor", {"comparison_level": "Monitor", "threshold": None, "margin": None}

        min_pass = True
        max_pass = True
        margin_min = None
        margin_max = None

        if min_threshold is not None:
            min_pass = value >= min_threshold
            margin_min = value - min_threshold
        if max_threshold is not None:
            max_pass = value <= max_threshold
            margin_max = max_threshold - value

        if min_pass and max_pass:
            return "PASS", "Range", {
                "comparison_level": "Range",
                "threshold": {"min": min_threshold, "max": max_threshold},
                "target_level": "Range",
            }

        return "FAIL", "None", {
            "comparison_level": "Range",
            "threshold": {"min": min_threshold, "max": max_threshold},
            "target_level": "Range",
        }

    def _build_comment(
        self,
        metric_id: str,
        direction: str,
        observed: float,
        status: str,
        pass_level: str,
        detail: Mapping[str, object],
        extras: Mapping[str, object],
        hazard_link: str | None,
    ) -> str:
        parts: List[str] = []
        threshold = detail.get("threshold")
        comparison_level = detail.get("comparison_level")
        target_level = detail.get("target_level", comparison_level)

        if status == "PASS":
            if isinstance(threshold, dict):
                parts.append(
                    self._format_range_pass(
                        observed,
                        threshold.get("min"),
                        threshold.get("max"),
                        pass_level,
                    )
                )
            else:
                parts.append(f"Meets {pass_level} requirement.")
        elif status == "FAIL":
            if isinstance(threshold, dict):
                parts.append(
                    self._format_range_fail(
                        observed,
                        threshold.get("min"),
                        threshold.get("max"),
                    )
                )
            else:
                parts.append("Threshold violation detected.")
        elif status == "ALARM":
            if threshold is not None:
                parts.append(f"Alarm triggered at {pass_level} (value {observed:.4g} < alpha {threshold:.4g}).")
            else:
                parts.append("Alarm triggered (alpha threshold).")
        else:
            parts.append("Metric monitored without gating threshold.")

        extras_comment = self._extras_comment(metric_id, extras)
        if extras_comment:
            parts.append(extras_comment)

        if status in {"FAIL", "ALARM"}:
            mitigation = self._hazard_mitigation(hazard_link)
            if mitigation:
                parts.append(mitigation)

        return " ".join(parts)

    def _extras_comment(self, metric_id: str, extras: Mapping[str, object]) -> str:
        if not extras:
            return ""

        if metric_id in {"M.FL.ATTACK_RES", "M.FL.BYZ_TOL", "M.FL.ATTACK_DET"}:
            pieces: List[str] = []
            if "resistant_scenarios" in extras:
                pieces.append(
                    f"{extras.get('resistant_scenarios')} of {extras.get('total_scenarios')} scenarios resistant"
                )
            tolerated = extras.get("tolerated_fraction")
            if isinstance(tolerated, (int, float)):
                pieces.append(f"Tolerated fraction {tolerated:.2f}")
            detection = extras.get("average_detection_recall")
            if isinstance(detection, (int, float)):
                pieces.append(f"Detection recall {detection:.2f}")
            threshold = extras.get("max_relative_degradation")
            if isinstance(threshold, (int, float)):
                pieces.append(f"Δmax {threshold:.2f}")
            return "; ".join(pieces) + "." if pieces else ""

        if metric_id == "M.RT.LAT":
            hw_profile = extras.get("hw_profile")
            mean_ms = extras.get("mean_ms")
            p95_ms = extras.get("p95_ms")
            max_ms = extras.get("max_ms")
            pieces = []
            if isinstance(hw_profile, str):
                pieces.append(f"HW {hw_profile}")
            if isinstance(mean_ms, (int, float)):
                pieces.append(f"mean {mean_ms:.2f} ms")
            if isinstance(p95_ms, (int, float)):
                pieces.append(f"P95 {p95_ms:.2f} ms")
            if isinstance(max_ms, (int, float)):
                pieces.append(f"max {max_ms:.2f} ms")
            return "; ".join(pieces) + "." if pieces else ""

        if metric_id == "M.XAI.FIDE_STAB":
            median = extras.get("median")
            min_value = extras.get("min")
            samples = extras.get("samples")
            pieces = []
            if isinstance(median, (int, float)):
                pieces.append(f"median {median:.3f}")
            if isinstance(min_value, (int, float)):
                pieces.append(f"min {min_value:.3f}")
            if isinstance(samples, int):
                pieces.append(f"{samples} samples")
            return "; ".join(pieces) + "." if pieces else ""

        if metric_id == "M.FL.PRIV_DP":
            variant = extras.get("variant")
            epsilon = extras.get("epsilon", extras.get("value"))
            delta = extras.get("delta")
            utility_delta = extras.get("utility_delta")
            guardrail = extras.get("guardrail_pass")
            pieces: List[str] = []
            if isinstance(variant, str) and variant:
                pieces.append(f"Variant {variant}")
            if isinstance(epsilon, (int, float)):
                pieces.append(f"ε={float(epsilon):.3f}")
            if isinstance(delta, (int, float)):
                pieces.append(f"δ={float(delta):.0e}")
            if isinstance(utility_delta, (int, float)):
                pieces.append(f"Δutility {utility_delta * 100:.1f}%")
            if isinstance(guardrail, bool):
                pieces.append("guardrail pass" if guardrail else "guardrail fail")
            return "; ".join(pieces) + "." if pieces else ""

        if metric_id in {"M.FL.COMM_BYTES", "M.FL.COMM_COMP"}:
            variant = extras.get("variant")
            avg_bytes = extras.get("avg_bytes")
            utility_delta = extras.get("utility_delta")
            guardrail = extras.get("guardrail_pass")
            pieces: List[str] = []
            if isinstance(variant, str) and variant:
                pieces.append(f"Variant {variant}")
            if isinstance(avg_bytes, (int, float)):
                mb_value = float(avg_bytes) / 1_000_000.0
                pieces.append(f"{mb_value:.3f} MB/round")
            if isinstance(utility_delta, (int, float)):
                pieces.append(f"Δutility {utility_delta * 100:.1f}%")
            if isinstance(guardrail, bool):
                pieces.append("guardrail pass" if guardrail else "guardrail fail")
            return "; ".join(pieces) + "." if pieces else ""

        if metric_id in {"M13", "M14", "M35"}:
            per_client = extras.get("per_client_accuracy")
            if isinstance(per_client, Mapping) and per_client:
                accuracies = [float(metrics.get("accuracy", float("nan"))) for metrics in per_client.values()]
                accuracies = [a for a in accuracies if not math.isnan(a)]
                if accuracies:
                    return f"Per-client accuracy range {min(accuracies):.4f}-{max(accuracies):.4f}."

        if metric_id == "M15":
            stats = extras.get("group_stats")
            if isinstance(stats, Mapping) and stats:
                positives = [float(group.get("positive_rate", float("nan"))) for group in stats.values()]
                positives = [p for p in positives if not math.isnan(p)]
                if positives:
                    return f"Group positive-rate spread {min(positives):.4f}-{max(positives):.4f}."

        if metric_id == "M16":
            stats = extras.get("group_stats")
            if isinstance(stats, Mapping) and stats:
                tprs = [float(group.get("tp_rate", float("nan"))) for group in stats.values()]
                tprs = [t for t in tprs if not math.isnan(t)]
                if tprs:
                    return f"Group recall spread {min(tprs):.4f}-{max(tprs):.4f}."

        if metric_id == "M17":
            stats = extras.get("group_stats")
            if isinstance(stats, Mapping) and stats:
                supports = [int(group.get("size", 0)) for group in stats.values()]
                if supports:
                    return f"Evaluated across {len(supports)} groups (min support {min(supports)})."

        if metric_id == "M24":
            ks_stat = extras.get("ks_statistic")
            if isinstance(ks_stat, (int, float)) and not math.isnan(float(ks_stat)):
                return f"KS statistic {float(ks_stat):.4f}."

        if metric_id in {"M21", "M22", "M27"}:
            attack = extras.get("attack_metrics") or extras
            if isinstance(attack, Mapping):
                detected = attack.get("detection_rate")
                if metric_id == "M21" and isinstance(detected, (int, float)):
                    return f"Detected {float(detected):.4f} of malicious clients."
                if metric_id == "M22":
                    ratio = attack.get("robustness_ratio") or attack.get("accuracy_attack")
                    if isinstance(ratio, (int, float)):
                        return f"Robustness ratio {float(ratio):.4f}."
                if metric_id == "M27":
                    asr = attack.get("attack_success_rate")
                    if isinstance(asr, (int, float)):
                        return f"Attack success rate {float(asr):.4f}."

        if metric_id == "M36":
            tolerance = extras.get("tolerance") or extras.get("dropout_metrics", {}).get("tolerance")
            if isinstance(tolerance, (int, float)):
                return f"Sustains performance up to {float(tolerance):.2g} dropout."  # type: ignore[arg-type]

        return ""

    def _hazard_mitigation(self, hazard_link: str | None) -> str | None:
        if not hazard_link:
            return None
        key = hazard_link.lower()
        if key in HAZARD_MITIGATIONS:
            return HAZARD_MITIGATIONS[key]
        return None

    @staticmethod
    def _format_range_pass(value: float, min_thr: float | None, max_thr: float | None, label: str) -> str:
        min_text = f"{min_thr:.4g}" if isinstance(min_thr, (int, float)) else "-inf"
        max_text = f"{max_thr:.4g}" if isinstance(max_thr, (int, float)) else "+inf"
        return f"Value {value:.4g} within {label} range [{min_text}, {max_text}]."

    @staticmethod
    def _format_range_fail(value: float, min_thr: float | None, max_thr: float | None) -> str:
        messages: List[str] = []
        if isinstance(min_thr, (int, float)) and value < min_thr:
            messages.append(f"{min_thr - value:.4g} below minimum {min_thr:.4g}")
        if isinstance(max_thr, (int, float)) and value > max_thr:
            messages.append(f"{value - max_thr:.4g} above maximum {max_thr:.4g}")
        if not messages:
            return "Threshold violation detected."
        return " and ".join(messages) + "."

    def _validate_metric_extras(self, metric_id: str, extras: Mapping[str, object]) -> None:
        if metric_id in {"M15", "M16", "M17"} and "group_stats" not in extras:
            raise ValueError(f"Fairness metric {metric_id} requires group_stats in extras")
        if metric_id in {"M13", "M14", "M35"} and "per_client_accuracy" not in extras:
            raise ValueError(f"Client variance metric {metric_id} requires per_client_accuracy in extras")
        if metric_id == "M24" and "ks_statistic" not in extras:
            raise ValueError("Metric M24 requires ks_statistic in extras for traceability")

    def _get_metadata(self, metric_id: str) -> MetricMetadata:
        if metric_id not in self.metadata:
            raise KeyError(f"Unknown metric id {metric_id}")
        return self.metadata[metric_id]

    def _apply_composite_dependencies(self, records: List[MetricAssessmentRecord]) -> None:
        if not records:
            return
        record_map = {record.metric_id: record for record in records}
        for parent, children in COMPOSITE_METRICS.items():
            parent_record = record_map.get(parent)
            if parent_record is None:
                continue
            child_records = [record_map[child] for child in children if child in record_map]
            if not child_records:
                continue
            failing_child = next((child for child in child_records if child.status != "PASS"), None)
            if failing_child:
                parent_record.status = failing_child.status
                parent_record.pass_level = failing_child.pass_level
                parent_record.comment = (
                    f"Composite metric relies on {', '.join(children)}; "
                    f"{failing_child.metric_id} is {failing_child.status}."
                )
            else:
                parent_record.status = "PASS"
                parent_record.pass_level = "Composite"
                parent_record.comment = (
                    f"Composite metric satisfied via sub-metrics {', '.join(children)}."
                )


