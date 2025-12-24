"""Utilities for persisting experiment assessment results."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from .assessment import MetricAssessmentRecord


DEFAULT_COLUMNS: Sequence[str] = (
    "Test",
    "Metric_ID",
    "Metric_Name",
    "Observed_Value",
    "Unit",
    "CI_Lower",
    "CI_Upper",
    "Min_Threshold",
    "Max_Threshold",
    "Pass_Level",
    "Status",
    "Comment",
    "Hazard_Link",
    "Category",
    "Regulatory_Requirements",
    "Method",
    "Assumptions",
    "Rationale",
    "Optimal_Target",
)


def records_to_dataframe(records: Sequence[MetricAssessmentRecord]) -> pd.DataFrame:
    rows = [record.to_dict() for record in records]
    if not rows:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    df = pd.DataFrame(rows)
    for column in DEFAULT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    df = df[list(DEFAULT_COLUMNS)]
    df.sort_values(by=["Test", "Metric_ID"], inplace=True)
    return df.reset_index(drop=True)


def write_test_report(
    test_id: str,
    records: Sequence[MetricAssessmentRecord],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = records_to_dataframe([r for r in records if r.test_id == test_id])
    output_path = output_dir / f"{test_id}_results.csv"
    df.to_csv(output_path, index=False)
    return output_path


def write_consolidated_report(records: Sequence[MetricAssessmentRecord], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = records_to_dataframe(records)
    df.to_csv(output_path, index=False)
    return output_path


def compute_summary(records: Sequence[MetricAssessmentRecord]) -> Dict[str, object]:
    total = len(records)
    pass_count = sum(1 for r in records if r.status == "PASS")
    fail_count = sum(1 for r in records if r.status == "FAIL")
    alarm_count = sum(1 for r in records if r.status == "ALARM")
    monitor_count = sum(1 for r in records if r.status == "MONITOR")

    fails_by_category = Counter(
        (r.category or "Unspecified") for r in records if r.status in {"FAIL", "ALARM"}
    )

    return {
        "total": total,
        "pass_count": pass_count,
        "pass_rate": (pass_count / total) if total else 0.0,
        "fail_count": fail_count,
        "fail_rate": (fail_count / total) if total else 0.0,
        "alarm_count": alarm_count,
        "monitor_count": monitor_count,
        "fails_by_category": dict(sorted(fails_by_category.items(), key=lambda item: item[0])),
    }


def generate_summary_markdown(
    records: Sequence[MetricAssessmentRecord],
    summary: Mapping[str, object],
) -> str:
    lines: List[str] = []
    lines.append("# Assurance Summary")
    lines.append("")

    lines.append("## Overview")
    lines.append(f"- Total metrics evaluated: {summary.get('total', 0)}")
    lines.append(f"- PASS count: {summary.get('pass_count', 0)} (rate {summary.get('pass_rate', 0.0):.2%})")
    lines.append(f"- FAIL count: {summary.get('fail_count', 0)} (rate {summary.get('fail_rate', 0.0):.2%})")
    lines.append(f"- ALARM count: {summary.get('alarm_count', 0)}")
    lines.append(f"- MONITOR count: {summary.get('monitor_count', 0)}")
    lines.append("")

    lines.append("## Failures and Alerts")
    failure_records = [r for r in records if r.status in {"FAIL", "ALARM"}]
    if not failure_records:
        lines.append("- None")
    else:
        for record in failure_records:
            thresholds = []
            if record.min_threshold is not None:
                thresholds.append(f"Min {record.min_threshold:.4g}")
            if record.max_threshold is not None:
                thresholds.append(f"Max {record.max_threshold:.4g}")
            threshold_text = ", ".join(thresholds) if thresholds else "Not enforced"
            lines.append(
                f"- {record.test_id} / {record.metric_id} ({record.metric_name}): "
                f"Observed {record.observed_value:.4g} {record.unit}; thresholds [{threshold_text}]. {record.comment}"
            )
    lines.append("")

    lines.append("## Failed Metrics by Category")
    fails_by_category = summary.get("fails_by_category", {})
    if not fails_by_category:
        lines.append("- None")
    else:
        for category, count in fails_by_category.items():
            lines.append(f"- {category}: {count}")
    lines.append("")

    lines.append("## Next Steps")
    if failure_records:
        lines.append("- Investigate alerts/failures above and execute recommended mitigations.")
    else:
        lines.append("- Maintain monitoring cadence; all gated metrics meet configured thresholds.")

    return "\n".join(lines) + "\n"


def write_summary_report(
    records: Sequence[MetricAssessmentRecord],
    output_path: Path,
    summary: Mapping[str, object],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = generate_summary_markdown(records, summary)
    output_path.write_text(text, encoding="utf-8")
    return output_path


