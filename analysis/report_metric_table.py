"""Generate an overview table of metric results per algorithm."""

from __future__ import annotations

import csv
import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config" / "config_v2"
REPORT_DIR = PROJECT_ROOT / "results" / "assurance_reports"
OUTPUT_FILE = REPORT_DIR / "metrics_overview.md"
ALGOS = ["fedavg", "fedprox", "scaffold"]
STATUS_COLORS = {
    "PASS": "#c6efce",
    "FAIL": "#f4c7c3",
    "ALARM": "#ffe699",
    "MONITOR": "#fff3cd",
}


def load_metric_catalog() -> Dict[str, str]:
    catalog_path = CONFIG_DIR / "C.Metric_catalog.csv"
    df = pd.read_csv(catalog_path)
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        metric_id = str(row.get("Metric ID", "")).strip()
        metric_name = str(row.get("Metric Name", "")).strip()
        if metric_id:
            mapping[metric_id] = metric_name
    return mapping


def _clean_threshold_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def load_thresholds() -> Dict[str, Dict[str, str]]:
    thresholds_path = CONFIG_DIR / "D.Thresholds.csv"
    df = pd.read_csv(thresholds_path)
    data: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        metric_id = str(row.get("Metric ID", "")).strip()
        if not metric_id:
            continue
        min_raw = _clean_threshold_value(row.get("Threshold Min", ""))
        max_raw = _clean_threshold_value(row.get("Threshold max", ""))
        unit = _clean_threshold_value(row.get("Unit", ""))
        data[metric_id] = {
            "min": min_raw,
            "max": max_raw,
            "unit": unit,
        }
    return data


def load_metric_order_and_mapping() -> Tuple[List[str], Dict[str, List[str]]]:
    testbeds_path = CONFIG_DIR / "E.Test_beds.csv"
    metric_order: List[str] = []
    metric_to_tb: Dict[str, List[str]] = defaultdict(list)

    with testbeds_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            test_id = str(row.get("Test ID", "")).strip()
            metrics_field = row.get("Metrics evaluated", "")
            if not metrics_field:
                continue
            tokens = [token.strip() for token in metrics_field.split(";")]
            for token in tokens:
                if not token:
                    continue
                metric_id = token.split()[0].strip('",')
                if not metric_id:
                    continue
                if metric_id not in metric_order:
                    metric_order.append(metric_id)
                if test_id and test_id not in metric_to_tb[metric_id]:
                    metric_to_tb[metric_id].append(test_id)
    return metric_order, metric_to_tb


def load_pass_fail_file(tb_id: str, algo: str) -> pd.DataFrame | None:
    path = REPORT_DIR / f"{tb_id}_{algo}_pass_fail.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def lookup_metric_rows(metric_id: str, tb_ids: List[str], algo: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for tb_id in tb_ids:
        df = load_pass_fail_file(tb_id, algo)
        if df is None:
            continue
        matches = df[df["Metric_ID"].astype(str).str.strip() == metric_id]
        if matches.empty:
            continue
        row = matches.iloc[0]
        observed = row.get("Observed_Value")
        unit = str(row.get("Unit", "")).strip()
        status = str(row.get("Status", "")).strip().upper()
        value_str = format_value(observed, unit)
        rows.append(
            {
                "tb": tb_id,
                "value": value_str,
                "status": status or "UNKNOWN",
            }
        )
    return rows


def format_value(value: object, unit: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if isinstance(value, str):
        value_str = value.strip()
    else:
        try:
            num = float(value)
            value_str = f"{num:.4g}"
        except (TypeError, ValueError):
            value_str = str(value)
    if unit:
        return f"{value_str} {unit}"
    return value_str


def combine_rows(rows: List[Dict[str, object]]) -> Tuple[str, str]:
    if not rows:
        return "N/A", ""
    # Determine worst status
    severity_order = {"FAIL": 3, "ALARM": 2, "MONITOR": 1, "PASS": 0}
    worst_row = max(rows, key=lambda r: severity_order.get(r["status"], 1))

    parts = [f"{entry['tb']}: {entry['value']} ({entry['status']})" for entry in rows]
    content = "<br>".join(parts)
    return content, worst_row["status"]


def acceptance_rule_text(threshold: Dict[str, str] | None) -> str:
    if not threshold:
        return "-"
    min_text = threshold.get("min", "").strip()
    max_text = threshold.get("max", "").strip()
    parts: List[str] = []
    if min_text:
        parts.append(f"≥ {min_text}")
    if max_text:
        parts.append(f"≤ {max_text}")
    if not parts:
        return "-"
    return " and ".join(parts)


def wrap_with_status(text: str, status: str) -> str:
    color = STATUS_COLORS.get(status.upper())
    if color:
        return f'<span style="background-color:{color}; padding:2px 4px; display:inline-block;">{text}</span>'
    return text


def main() -> None:
    metric_catalog = load_metric_catalog()
    thresholds = load_thresholds()
    metric_order, metric_to_tb = load_metric_order_and_mapping()

    rows_markdown: List[str] = []
    header = "| Metric | Acceptance Rule | FedAvg | FedProx | SCAF |\n"
    divider = "| --- | --- | --- | --- | --- |\n"
    rows_markdown.append(header)
    rows_markdown.append(divider)

    for metric_id in metric_order:
        metric_name = metric_catalog.get(metric_id, "")
        label = f"{metric_id} – {metric_name}" if metric_name else metric_id
        acceptance = acceptance_rule_text(thresholds.get(metric_id))
        tb_ids = metric_to_tb.get(metric_id, [])

        algo_cells: List[str] = []
        for algo in ALGOS:
            entries = lookup_metric_rows(metric_id, tb_ids, algo)
            cell_text, status = combine_rows(entries)
            algo_cells.append(wrap_with_status(cell_text, status) if entries else "N/A")

        row_md = f"| {label} | {acceptance} | {' | '.join(algo_cells)} |\n"
        rows_markdown.append(row_md)

    OUTPUT_FILE.write_text(
        "# Metric Overview\n\n"
        "Consolidated metric values per algorithm, using assurance report outputs.\n\n"
        + "".join(rows_markdown),
        encoding="utf-8",
    )
    print(f"Wrote overview to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

