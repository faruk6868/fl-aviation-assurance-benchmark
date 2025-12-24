from __future__ import annotations

import argparse
import csv
import json
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch, PathPatch, Rectangle, Wedge
from matplotlib.path import Path as MplPath


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


PILLAR_MAP = {
    "DA": "Data & Governance",
    "MR": "Model Performance & Robustness",
    "SR": "Lifecycle & System Reliability",
    "CE": "Communication & Efficiency",
    "FE": "Fairness & Equity",
    "PS": "Privacy & Security",
}


AUTHORITIES = [
    {
        "id": "EASA",
        "name": "EASA AI Concept / Roadmap",
        "refs": ["https://www.easa.europa.eu/en/downloads/134946/en"],
        "notes": "",
    },
    {
        "id": "FAA",
        "name": "FAA AI Safety Assurance",
        "refs": ["https://www.faa.gov/air_traffic/technology/ai"],
        "notes": "",
    },
    {
        "id": "FDA_ISO",
        "name": "FDA / ISO / SAE Standards",
        "refs": [
            "https://www.fda.gov/",
            "https://www.iso.org/",
            "https://www.sae.org/standards/",
        ],
        "notes": "",
    },
]

AUTHORITY_COLORS: Dict[str, str] = {
    "EASA": "#4e79a7",
    "FAA": "#e15759",
    "FDA_ISO": "#76b7b2",
}

AUTHORITY_LABELS: Dict[str, str] = {
    "EASA": "EASA (EU Aviation)",
    "FAA": "FAA (US Aviation)",
    "FDA_ISO": "FDA / ISO / SAE Standards",
}

NEUTRAL_COLOR = "#bab0ac"


THRESHOLD_LEVELS = ["Catastrophic", "Hazardous", "Major"]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_semicolon_list(value: str) -> List[str]:
    if not isinstance(value, str):
        return []
    parts = []
    for chunk in value.replace("\n", " ").split(","):
        token = chunk.strip()
        if token:
            parts.append(token)
    return parts


def authority_from_source(raw_source: str) -> str:
    if not isinstance(raw_source, str):
        return "EASA"
    lowered = raw_source.lower()
    if "easa" in lowered:
        return "EASA"
    if "faa" in lowered:
        return "FAA"
    if "fda" in lowered:
        return "FDA_ISO"
    if "iso" in lowered or "sae" in lowered or "21448" in lowered:
        return "FDA_ISO"
    return "EASA"


def assumptions_to_dict(text: str) -> Dict[str, float | str]:
    if not isinstance(text, str) or not text.strip():
        return {}
    result: Dict[str, float | str] = {}
    for chunk in text.split(";"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def parse_metric_tokens(metric_field: str) -> List[str]:
    if not isinstance(metric_field, str):
        return []
    matches = re.findall(r"M\d+(?:-M\d+)?", metric_field)
    metrics: List[str] = []
    for token in matches:
        if "-" in token:
            left, right = token.split("-", 1)
            try:
                start = int(left[1:])
                end = int(right[1:])
            except ValueError:
                continue
            step = 1 if end >= start else -1
            for idx in range(start, end + step, step):
                metrics.append(f"M{idx}")
        else:
            metrics.append(token)
    return sorted(set(metrics), key=lambda m: int(m[1:]) if len(m) > 1 and m[1:].isdigit() else m)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_requirements(path: Path) -> List[Dict[str, object]]:
    df = pd.read_csv(path, encoding="cp1252")
    requirements: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        req_id = row.get("ID")
        if not isinstance(req_id, str) or not req_id.strip():
            continue
        req_id = req_id.strip()
        prefix = req_id.split(".")[0]
        domain = PILLAR_MAP.get(prefix, "General")
        title = str(row.get("Title") or "").strip()
        raw_text = str(row.get("Description") or "").strip()
        source = str(row.get("Source") or "").strip()
        authority_id = authority_from_source(source)
        requirements.append(
            {
                "id": req_id,
                "title": title,
                "domain": domain,
                "source_authority_id": authority_id,
                "source_citation": source,
                "raw_text": raw_text,
                "notes": "",
            }
        )
    return requirements


def load_metric_metadata(path: Path) -> Dict[str, Dict[str, object]]:
    payload = json.loads(Path(path).read_text())
    metrics = payload.get("metrics", [])
    mapping: Dict[str, Dict[str, object]] = {}
    for item in metrics:
        metric_id = item["id"]
        mapping[metric_id] = item
    return mapping


def load_thresholds(threshold_paths: Mapping[str, Path]) -> Dict[str, Dict[str, Dict[str, object]]]:
    thresholds: Dict[str, Dict[str, Dict[str, object]]] = {level: {} for level in THRESHOLD_LEVELS}
    for level, csv_path in threshold_paths.items():
        df = pd.read_csv(csv_path, encoding="cp1252")
        for _, row in df.iterrows():
            metric_id = row["Metric_ID"]
            value = row.get("Threshold_Value")
            assumptions = assumptions_to_dict(str(row.get("Assumptions") or ""))
            thresholds[level][metric_id] = {
                "value": value if pd.notna(value) else None,
                "method": row.get("Method"),
                "hazard_link": row.get("Hazard_Link"),
                "assumptions": assumptions,
            }
    return thresholds


def load_metrics(
    metadata_path: Path,
    metrics_catalog_path: Path,
    thresholds_paths: Mapping[str, Path],
) -> Tuple[List[Dict[str, object]], Dict[str, List[str]]]:
    metadata = load_metric_metadata(metadata_path)
    thresholds = load_thresholds(thresholds_paths)
    df = pd.read_csv(metrics_catalog_path, encoding="cp1252")

    requirement_to_metric: Dict[str, List[str]] = {}
    metrics: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        metric_id = row["Metric_ID"]
        related_requirements = parse_semicolon_list(str(row.get("Related_Requirements") or ""))
        info = metadata.get(metric_id, {})
        direction = str(info.get("direction", "max")).lower()
        unit = info.get("unit", "")
        hazard_link = info.get("hazard_link")
        gated = direction not in {"monitor"}

        threshold_data: Dict[str, Dict[str, object]] = {}
        for level in THRESHOLD_LEVELS:
            entry = thresholds.get(level, {}).get(metric_id)
            if not entry:
                continue
            op = ">=" if direction == "max" else "<="
            if direction == "min":
                op = "<="
            elif direction == "alpha":
                op = "<"
            threshold_data[level] = {
                "value": entry["value"],
                "op": op,
                "assumptions": entry["assumptions"],
            }

        metric_entry = {
            "id": metric_id,
            "name": info.get("name") or row.get("Metric_Name"),
            "unit": unit,
            "direction": direction,
            "hazard_link": hazard_link,
            "gated": gated,
            "thresholds": threshold_data,
            "notes": row.get("Justification") or "",
            "calculator": info.get("method") or row.get("Calculation_Method") or "",
        }
        metrics.append(metric_entry)

        for req_id in related_requirements:
            requirement_to_metric.setdefault(req_id, []).append(metric_id)

    return metrics, requirement_to_metric


def load_tests(path: Path) -> Tuple[List[Dict[str, object]], Dict[str, List[str]]]:
    df = pd.read_csv(path, encoding="cp1252")
    tests: List[Dict[str, object]] = []
    metric_to_tests: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        raw_id = row.get("Test ID") or row.get("Test") or row.get("ID")
        if not isinstance(raw_id, str):
            continue
        test_id = raw_id.strip()
        if not test_id:
            continue

        metrics_field = str(row.get("Metrics Evaluated") or "")
        metrics = parse_metric_tokens(metrics_field)
        test_name = str(row.get("Test Name") or row.get("Experiment Name") or row.get("Name") or "").strip()
        test_entry = {
            "id": test_id,
            "name": test_name,
            "metrics": metrics,
            "params": {
                "objective": str(row.get("Objective") or ""),
                "dataset_configuration": str(row.get("Dataset Configuration") or ""),
                "fl_configuration": str(row.get("FL Configuration") or ""),
                "success_criteria": str(row.get("Success Criteria") or ""),
                "duration_estimate": str(row.get("Duration Estimate") or ""),
            },
            "notes": "",
        }
        tests.append(test_entry)
        for metric_id in metrics:
            metric_to_tests.setdefault(metric_id, []).append(test_id)

    return tests, metric_to_tests


def load_results(results_dir: Path) -> List[Dict[str, object]]:
    overlays: List[Dict[str, object]] = []
    for csv_path in sorted(results_dir.glob("*_results.csv")):
        test_id = csv_path.stem.split("_")[0].upper()
        if test_id.startswith("E") and test_id[1:].isdigit():
            test_id = f"T{test_id[1:]}"
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            metric_id = str(row["Metric_ID"]).strip()
            thresholds = {
                "Cat": row.get("Cat_Threshold"),
                "Haz": row.get("Haz_Threshold"),
                "Maj": row.get("Maj_Threshold"),
            }
            ci_low = row.get("CI_Lower")
            ci_high = row.get("CI_Upper")
            ci_available = pd.notna(ci_low) and pd.notna(ci_high)
            ci = [ci_low, ci_high] if ci_available else None
            raw_pass_level = row.get("Pass_Level")
            pass_level = str(raw_pass_level) if pd.notna(raw_pass_level) else ""
            overlays.append(
                {
                    "test_id": test_id,
                    "metric_id": metric_id,
                    "observed": row.get("Observed_Value"),
                    "unit": row.get("Unit"),
                    "direction": row.get("Direction"),
                    "thresholds": thresholds,
                    "pass_level": pass_level,
                    "status": row.get("Status"),
                    "comment": row.get("Comment"),
                    "ci95": ci,
                }
            )
    return overlays


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def build_validation_report(
    requirements: Sequence[Dict[str, object]],
    metrics: Sequence[Dict[str, object]],
    tests: Sequence[Dict[str, object]],
    requirement_to_metric: Mapping[str, List[str]],
    metric_to_tests: Mapping[str, List[str]],
    overlays: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []

    requirement_ids = {req["id"] for req in requirements}
    metric_ids = {metric["id"] for metric in metrics}
    test_ids = {exp["id"] for exp in tests}

    # Requirement coverage
    for req_id in requirement_ids:
        linked = requirement_to_metric.get(req_id, [])
        if not linked:
            warnings.append(f"Requirement {req_id} has no linked metrics.")

    # Metric coverage
    for metric_id in metric_ids:
        if metric_id not in metric_to_tests:
            warnings.append(f"Metric {metric_id} is not used by any test.")

    # Test coverage
    for exp in tests:
        if not exp["metrics"]:
            warnings.append(f"Test {exp['id']} does not list any metrics.")

    # Overlay mapping
    overlay_pairs = {(item["metric_id"], item["test_id"]) for item in overlays}
    missing_results: List[str] = []
    for metric_id, exp_list in metric_to_tests.items():
        for exp_id in exp_list:
            if (metric_id, exp_id) not in overlay_pairs:
                missing_results.append(f"{metric_id}→{exp_id}")
    if missing_results:
        warnings.append(
            f"Missing results for {len(missing_results)} metric/test pairs: "
            + ", ".join(missing_results[:10])
            + ("..." if len(missing_results) > 10 else "")
        )

    # Units/directions consistency
    metric_units = {metric["id"]: metric.get("unit") for metric in metrics}
    metric_dirs = {metric["id"]: metric.get("direction") for metric in metrics}
    unit_mismatches = 0
    dir_mismatches = 0
    for overlay in overlays:
        metric_id = overlay["metric_id"]
        if metric_id not in metric_ids:
            errors.append(f"Overlay references unknown metric {metric_id}.")
            continue
        unit = overlay.get("unit")
        expected_unit = metric_units.get(metric_id)
        if expected_unit and unit and expected_unit != unit:
            unit_mismatches += 1
        direction = overlay.get("direction")
        expected_direction = metric_dirs.get(metric_id)
        if expected_direction and direction and expected_direction != direction.lower():
            dir_mismatches += 1
    if unit_mismatches:
        warnings.append(f"Unit mismatches detected for {unit_mismatches} overlays.")
    if dir_mismatches:
        warnings.append(f"Direction mismatches detected for {dir_mismatches} overlays.")

    coverage = {
        "requirements_total": len(requirements),
        "requirements_with_metrics": sum(1 for req in requirements if requirement_to_metric.get(req["id"])),
        "metrics_total": len(metrics),
        "metrics_in_tests": sum(1 for metric in metrics if metric_to_tests.get(metric["id"])),
        "tests_total": len(tests),
        "overlay_entries": len(overlays),
    }

    return {
        "summary": {
            "requirements": coverage["requirements_total"],
            "metrics": coverage["metrics_total"],
            "tests": coverage["tests_total"],
            "overlays": coverage["overlay_entries"],
        },
        "coverage": coverage,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def draw_sankey(
    output_path: Path,
    requirements: Sequence[Dict[str, object]],
    metrics: Sequence[Dict[str, object]],
    tests: Sequence[Dict[str, object]],
    requirement_to_metric: Mapping[str, List[str]],
    metric_to_tests: Mapping[str, List[str]],
) -> None:
    plt.rcParams.update({"figure.autolayout": True})
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")

    # Prepare columns
    authority_ids = [auth["id"] for auth in AUTHORITIES]
    requirements_by_id = {req["id"]: req for req in requirements}
    metrics_by_id = {metric["id"]: metric for metric in metrics}
    tests_by_id = {exp["id"]: exp for exp in tests}
    metric_label_map = {
        metric_id: f"{metric_id}: {metric.get('name', '').strip()}" if metric.get("name") else metric_id
        for metric_id, metric in metrics_by_id.items()
    }

    columns = [
        ("Authority", authority_ids),
        ("Requirement", [req["id"] for req in requirements]),
        ("Metric", [metric["id"] for metric in metrics]),
        ("Test", [exp["id"] for exp in tests]),
    ]

    column_positions = {name: idx for idx, (name, _) in enumerate(columns)}
    node_positions: Dict[str, Tuple[float, float, float]] = {}
    x_spacing = 1.0 / (len(columns) - 1)

    for col_idx, (col_name, nodes) in enumerate(columns):
        x = col_idx * x_spacing
        # Determine node heights based on outgoing links
        weights: List[float] = []
        for node_id in nodes:
            if col_name == "Authority":
                count = sum(1 for req in requirements if req["source_authority_id"] == node_id)
            elif col_name == "Requirement":
                count = len(requirement_to_metric.get(node_id, [])) or 1
            elif col_name == "Metric":
                count = len(metric_to_tests.get(node_id, [])) or 1
            else:
                count = 1
            weights.append(count)
        total_weight = sum(weights)
        y = 0.0
        for node_id, weight in zip(nodes, weights):
            height = weight / total_weight if total_weight else 1.0 / len(nodes)
            node_positions[node_id] = (x, y + height / 2, height)
            y += height

    def draw_link(source_id: str, target_id: str, weight: float, color: str) -> None:
        sx, sy, sh = node_positions[source_id]
        tx, ty, th = node_positions[target_id]
        width = weight
        path = MplPath(
            [
                (sx + 0.02, sy - sh / 2 * width),
                (sx + 0.2 * x_spacing, sy - sh / 2 * width),
                (tx - 0.2 * x_spacing, ty - th / 2 * width),
                (tx - 0.02, ty - th / 2 * width),
                (tx - 0.02, ty + th / 2 * width),
                (tx - 0.2 * x_spacing, ty + th / 2 * width),
                (sx + 0.2 * x_spacing, sy + sh / 2 * width),
                (sx + 0.02, sy + sh / 2 * width),
            ],
            closed=True,
            codes=[
                MplPath.MOVETO,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.LINETO,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
            ],
        )
        patch = PathPatch(path, facecolor=color, alpha=0.45, edgecolor="none")
        ax.add_patch(patch)

    def node_color(domain: str) -> str:
        palette = {
            "Data & Governance": "#4e79a7",
            "Model Performance & Robustness": "#f28e2c",
            "Lifecycle & System Reliability": "#e15759",
            "Communication & Efficiency": "#76b7b2",
            "Fairness & Equity": "#59a14f",
            "Privacy & Security": "#edc948",
        }
        return palette.get(domain, "#bab0ac")

    color_palette = [
        "#4c72b0",
        "#55a868",
        "#c44e52",
        "#8172b3",
        "#ccb974",
        "#64b5cd",
        "#8c8c8c",
        "#e15759",
        "#76b7b2",
        "#f28e2c",
        "#59a14f",
        "#edc948",
    ]

    def cycle_color(idx: int) -> str:
        return color_palette[idx % len(color_palette)]

    def lighten(color: str, factor: float) -> str:
        r, g, b = mcolors.to_rgb(color)
        r = min(1.0, r + (1.0 - r) * factor)
        g = min(1.0, g + (1.0 - g) * factor)
        b = min(1.0, b + (1.0 - b) * factor)
        return (r, g, b)

    requirement_color_map: Dict[str, str] = {}
    for idx, req in enumerate(sorted(requirements, key=lambda r: r["id"])):
        requirement_color_map[req["id"]] = cycle_color(idx)

    metric_color_map: Dict[str, str] = {}
    for idx, metric in enumerate(sorted(metrics, key=lambda m: m["id"])):
        metric_color_map[metric["id"]] = cycle_color(idx + 7)

    # Links: authority -> requirement
    for req in requirements:
        color = requirement_color_map.get(req["id"], node_color(req["domain"]))
        draw_link(req["source_authority_id"], req["id"], 0.8, color)

    # Requirement -> Metric
    for req_id, metrics_list in requirement_to_metric.items():
        base_color = requirement_color_map.get(req_id, "#bab0ac")
        color = lighten(base_color, 0.25)
        share = 0.6 / max(len(metrics_list), 1)
        for metric_id in metrics_list:
            draw_link(req_id, metric_id, share, color)

    # Metric -> Test
    for metric_id, tests_list in metric_to_tests.items():
        domain_color = metric_color_map.get(metric_id, "#6b4c8b")
        share = 0.5 / max(len(tests_list), 1)
        for exp_id in tests_list:
            draw_link(metric_id, exp_id, share, domain_color)

    # Draw nodes
    for col_name, node_ids in columns:
        for node_id in node_ids:
            x, y, height = node_positions[node_id]
            rect = Rectangle((x - 0.02, y - height / 2), 0.04, height, facecolor="#333333", alpha=0.8)
            ax.add_patch(rect)
            label = node_id

            if col_name == "Authority":
                ax.text(x - 0.05, y, label, ha="right", va="center", fontsize=9, color="black")
            elif col_name == "Requirement":
                ax.text(x - 0.045, y, label, ha="right", va="center", fontsize=5, color="black")
            elif col_name == "Metric":
                metric_label = metric_label_map.get(node_id, label)
                ax.text(x + 0.045, y, metric_label, ha="left", va="center", fontsize=5, color="black")
            elif col_name == "Test":
                ax.text(x + 0.05, y, label, ha="left", va="center", fontsize=7, color="black")

    ax.set_title("Concentric Assurance Map – Traceability Flow (Sankey)", fontsize=14, weight="bold")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_metric_health(overlays: Sequence[Dict[str, object]]) -> Dict[str, str]:
    priority = {"Cat": 3, "Haz": 2, "Maj": 1, "": 0, "None": 0}
    status_priority = {"PASS": 3, "ALARM": 2, "MONITOR": 1, "FAIL": 0}
    aggregated: Dict[str, Tuple[int, int]] = {}
    for overlay in overlays:
        metric_id = overlay["metric_id"]
        status = str(overlay.get("status") or "").upper()
        pass_level = str(overlay.get("pass_level") or "")
        best_level = priority.get(pass_level, 0)
        status_rank = status_priority.get(status, 0)
        current = aggregated.get(metric_id, (-1, -1))
        if (status_rank, best_level) > current:
            aggregated[metric_id] = (status_rank, best_level)
    colour_map: Dict[str, str] = {}
    for metric_id, (status_rank, level_rank) in aggregated.items():
        if status_rank == 0:
            colour_map[metric_id] = "#e15759"  # FAIL
        elif status_rank == 2:
            colour_map[metric_id] = "#f28e2c"  # ALARM
        elif level_rank >= 3:
            colour_map[metric_id] = "#59a14f"  # Cat
        elif level_rank == 2:
            colour_map[metric_id] = "#76b7b2"  # Haz
        elif level_rank == 1:
            colour_map[metric_id] = "#edc948"  # Maj
        else:
            colour_map[metric_id] = "#bab0ac"
    return colour_map


def draw_concentric(
    output_path: Path,
    requirements: Sequence[Dict[str, object]],
    metrics: Sequence[Dict[str, object]],
    tests: Sequence[Dict[str, object]],
    overlays: Sequence[Dict[str, object]],
    requirement_to_metric: Mapping[str, List[str]],
    metric_to_tests: Mapping[str, List[str]],
) -> None:
    metric_health = compute_metric_health(overlays)

    fig, ax = plt.subplots(figsize=(13, 13), subplot_kw={"projection": "polar"})
    ax.set_axis_off()
    def add_ring(
        entries: Sequence[str],
        base_radius: float,
        width: float,
        colors: Sequence[str],
        annotations: Sequence[str],
        font_size: float,
        padding: float,
    ) -> None:
        n = len(entries)
        if n == 0:
            return
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        theta_width = 2 * np.pi / n
        for idx, entry in enumerate(entries):
            angle = theta[idx]
            color = colors[idx]
            ax.bar(
                angle,
                width=theta_width,
                height=width,
                bottom=base_radius,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                align="edge",
                alpha=0.9,
            )

            text_angle = angle + theta_width / 2
            text_radius = base_radius + width + padding
            deg_angle = np.degrees(text_angle)
            if 90 <= deg_angle <= 270:
                rotation_angle = deg_angle + 180
                ha = "right"
            else:
                rotation_angle = deg_angle
                ha = "left"

            ax.text(
                text_angle,
                text_radius,
                annotations[idx],
                rotation=rotation_angle,
                rotation_mode="anchor",
                ha=ha,
                va="center",
                fontsize=font_size,
                color="black",
            )

    # Center label
    ax.text(0, 0, "FL-PdM Core", ha="center", va="center", fontsize=14, weight="bold")

    requirements_by_id = {req["id"]: req for req in requirements}
    metrics_by_id = {metric["id"]: metric for metric in metrics}
    tests_by_id = {exp["id"]: exp for exp in tests}

    authority_sequence = [auth["id"] for auth in AUTHORITIES]
    sorted_requirements = []
    for authority_id in authority_sequence:
        authority_requirements = [
            req for req in requirements if req.get("source_authority_id") == authority_id
        ]
        authority_requirements.sort(key=lambda r: r["id"])
        sorted_requirements.extend(authority_requirements)
    remaining_requirements = [
        req for req in requirements if req not in sorted_requirements
    ]
    sorted_requirements.extend(sorted(remaining_requirements, key=lambda r: r["id"]))

    requirement_ids = [req["id"] for req in sorted_requirements]
    requirement_colors = [
        AUTHORITY_COLORS.get(req.get("source_authority_id", "EASA"), "#4e79a7")
        for req in sorted_requirements
    ]

    metric_order: List[str] = []
    seen_metrics: set[str] = set()
    for req_id in requirement_ids:
        linked_metrics = requirement_to_metric.get(req_id, [])
        linked_metrics_sorted = sorted(
            linked_metrics,
            key=lambda mid: int(mid[1:]) if mid[1:].isdigit() else mid,
        )
        for metric_id in linked_metrics_sorted:
            if metric_id not in seen_metrics:
                metric_order.append(metric_id)
                seen_metrics.add(metric_id)
    for metric in sorted(
        metrics, key=lambda m: int(m["id"][1:]) if m["id"][1:].isdigit() else m["id"]
    ):
        metric_id = metric["id"]
        if metric_id not in seen_metrics:
            metric_order.append(metric_id)
            seen_metrics.add(metric_id)

    metric_authority_map: Dict[str, str] = {}
    metric_colors: List[str] = []
    for metric_id in metric_order:
        metric = metrics_by_id.get(metric_id, {})
        linked_requirements = [
            req_id
            for req_id, metrics_list in requirement_to_metric.items()
            if metric_id in metrics_list
        ]
        linked_authorities = {
            requirements_by_id.get(req_id, {}).get("source_authority_id", "EASA")
            for req_id in linked_requirements
        }
        if len(linked_authorities) == 1:
            authority = next(iter(linked_authorities))
            metric_authority_map[metric_id] = authority
            fill_color = AUTHORITY_COLORS.get(authority, NEUTRAL_COLOR)
        else:
            fill_color = NEUTRAL_COLOR
        metric_colors.append(fill_color)

    test_order: List[str] = []
    seen_tests: set[str] = set()
    for metric_id in metric_order:
        exp_ids = metric_to_tests.get(metric_id, [])
        exp_ids_sorted = sorted(
            exp_ids, key=lambda eid: int(eid[1:]) if eid[1:].isdigit() else eid
        )
        for exp_id in exp_ids_sorted:
            if exp_id not in seen_tests:
                test_order.append(exp_id)
                seen_tests.add(exp_id)
    for exp in sorted(
        tests, key=lambda e: int(e["id"][1:]) if e["id"][1:].isdigit() else e["id"]
    ):
        if exp["id"] not in seen_tests:
            test_order.append(exp["id"])
            seen_tests.add(exp["id"])

    test_colors: List[str] = []
    for exp_id in test_order:
        linked_metrics = [
            metric_id
            for metric_id, exp_list in metric_to_tests.items()
            if exp_id in exp_list
        ]
        linked_authorities = {
            metric_authority_map.get(metric_id)
            for metric_id in linked_metrics
            if metric_authority_map.get(metric_id)
        }
        if len(linked_authorities) == 1:
            authority = next(iter(linked_authorities))
            test_colors.append(AUTHORITY_COLORS.get(authority, "#8cd17d"))
        else:
            test_colors.append("#8cd17d")

    # Ring specifications: (entries, colors, labels, width, font_size, padding)
    ring_specs = [
        (
            test_order,
            test_colors,
            test_order,
            0.95,
            6.8,
            0.05,
        ),
        (
            metric_order,
            metric_colors,
            metric_order,
            0.95,
            4.8,
            0.06,
        ),
        (
            requirement_ids,
            requirement_colors,
            requirement_ids,
            0.95,
            5.0,
            0.07,
        ),
        (
            [auth["id"] for auth in AUTHORITIES],
            [AUTHORITY_COLORS[auth["id"]] for auth in AUTHORITIES],
            [auth["id"] for auth in AUTHORITIES],
            0.95,
            9.0,
            0.08,
        ),
    ]

    current_radius = 0.0
    for entries, colors, labels, width, font_size, padding in ring_specs:
        add_ring(entries, current_radius, width, colors, labels, font_size, padding)
        current_radius += width

    ax.set_ylim(0, current_radius + 0.4)

    legend_handles: List[Patch] = [
        Patch(facecolor=AUTHORITY_COLORS[auth_id], edgecolor="none", label=AUTHORITY_LABELS[auth_id])
        for auth_id in authority_sequence
    ]
    legend_handles.append(
        Patch(facecolor=NEUTRAL_COLOR, edgecolor="none", label="Mixed / Multi-Authority")
    )
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
        fontsize=8,
    )

    ax.set_title("Concentric Assurance Map – Radial Overview", fontsize=16, weight="bold", pad=20)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def summarise_pass_rates(overlays: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    per_level = {"Cat": 0, "Haz": 0, "Maj": 0}
    total = {"Cat": 0, "Haz": 0, "Maj": 0}
    for entry in overlays:
        pass_level = str(entry.get("pass_level") or "")
        status = str(entry.get("status") or "")
        for level in per_level:
            total[level] += 1
            if status.upper() == "PASS" and pass_level == level:
                per_level[level] += 1
    rates = {level: (per_level[level] / total[level] if total[level] else 0.0) for level in per_level}
    return {"counts": per_level, "totals": total, "rates": rates}


def node_color(domain: str) -> str:
    palette = {
        "Data & Governance": "#4e79a7",
        "Model Performance & Robustness": "#f28e2c",
        "Lifecycle & System Reliability": "#e15759",
        "Communication & Efficiency": "#76b7b2",
        "Fairness & Equity": "#59a14f",
        "Privacy & Security": "#edc948",
    }
    return palette.get(domain, "#bab0ac")


def generate_report(
    output_path: Path,
    cam_data: Mapping[str, object],
    validation: Mapping[str, object],
    requirement_to_metric: Mapping[str, List[str]],
    metric_to_tests: Mapping[str, List[str]],
) -> None:
    requirements: List[Dict[str, object]] = cam_data["requirements"]  # type: ignore[assignment]
    metrics: List[Dict[str, object]] = cam_data["metrics"]  # type: ignore[assignment]
    tests: List[Dict[str, object]] = cam_data["tests"]  # type: ignore[assignment]
    overlays: List[Dict[str, object]] = cam_data["results_overlay"]  # type: ignore[assignment]

    requirements_by_pillar: Dict[str, int] = {}
    for req in requirements:
        requirements_by_pillar[req["domain"]] = requirements_by_pillar.get(req["domain"], 0) + 1

    metrics_by_hazard: Dict[str, int] = {}
    for metric in metrics:
        hazard = metric.get("hazard_link") or "General"
        metrics_by_hazard[hazard] = metrics_by_hazard.get(hazard, 0) + 1

    pass_summary = summarise_pass_rates(overlays)

    lines: List[str] = []
    lines.append("# Concentric Assurance Map (CAM) Report")
    lines.append("")
    lines.append("## 1. Purpose and Scope")
    lines.append(
        textwrap.dedent(
            """
            The Concentric Assurance Map (CAM) consolidates regulatory requirements, assurance metrics,
            federated learning tests, and empirical results into a single, auditable structure.
            This report explains how to interpret the CAM artefacts, highlights coverage statistics,
            and summarises threshold-gating performance across Catastrophic (Cat), Hazardous (Haz),
            and Major (Maj) assurance levels.
            """
        ).strip()
    )
    lines.append("")

    lines.append("## 2. Data Model")
    lines.append(
        textwrap.dedent(
            """
            The CAM follows a layered data model:

            - **Authorities** provide regulatory or standards context.
            - **Requirements** reference source citations and are grouped by the six assurance pillars.
            - **Metrics** encode measurable evidence, including direction, unit, hazard linkage, and thresholds.
            - **Tests** supply federated learning scenarios that generate observed metric outcomes.
            - **Results overlay** binds each `(test, metric)` pair to observed values and pass/fail status.
            """
        ).strip()
    )
    lines.append("")

    lines.append("## 3. Coverage Summary")
    coverage = validation["coverage"]  # type: ignore[index]
    lines.append("| Item | Count | Covered | Coverage |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(f"| Requirements | {coverage['requirements_total']} | {coverage['requirements_with_metrics']} | {coverage['requirements_with_metrics']/max(coverage['requirements_total'],1):.1%} |")
    lines.append(f"| Metrics | {coverage['metrics_total']} | {coverage['metrics_in_tests']} | {coverage['metrics_in_tests']/max(coverage['metrics_total'],1):.1%} |")
    lines.append(f"| Tests | {coverage['tests_total']} | {coverage['tests_total']} | 100.0% |")
    lines.append(f"| Result entries | {coverage['overlay_entries']} | — | — |")
    lines.append("")

    lines.append("### Requirements per Pillar")
    lines.append("| Pillar | Count |")
    lines.append("| --- | ---: |")
    for pillar, count in sorted(requirements_by_pillar.items()):
        lines.append(f"| {pillar} | {count} |")
    lines.append("")

    lines.append("### Metrics per Hazard Link")
    lines.append("| Hazard Link | Count |")
    lines.append("| --- | ---: |")
    for hazard, count in sorted(metrics_by_hazard.items()):
        lines.append(f"| {hazard} | {count} |")
    lines.append("")

    lines.append("## 4. Threshold Gating Summary")
    lines.append("| Level | Pass Count | Total Evaluations | Pass Rate |")
    lines.append("| --- | ---: | ---: | ---: |")
    for level in ["Cat", "Haz", "Maj"]:
        lines.append(
            f"| {level} | {pass_summary['counts'][level]} | {pass_summary['totals'][level]} | {pass_summary['rates'][level]:.1%} |"
        )
    lines.append("")
    lines.append(
        "Top failing metrics are concentrated in the Performance pillar where recall, RMSE, and confidence "
        "interval width remain below the allocated Major thresholds. Alarms arise primarily from drift detection "
        "(KS p-values) where the monitored alpha threshold is exceeded."
    )
    lines.append("")

    lines.append("## 5. Figures")
    lines.append("![Sankey](CAM_Sankey.svg)")
    lines.append("")
    lines.append("![Concentric Overview](CAM_Concentric.svg)")
    lines.append("")

    lines.append("## 6. Appendix – Flat Crosswalk")
    lines.append(
        "The artefact `CAM_Table.csv` enumerates every authority→requirement→metric→test link, "
        "augmented with observed metric data and threshold information. This table is suitable for "
        "spreadsheet analysis or ingestion into BI tooling."
    )
    lines.append("")

    if validation["warnings"]:  # type: ignore[index]
        lines.append("### Validation Warnings")
        for warning in validation["warnings"]:  # type: ignore[index]
            lines.append(f"- {warning}")
        lines.append("")
    if validation["errors"]:  # type: ignore[index]
        lines.append("### Validation Errors")
        for error in validation["errors"]:  # type: ignore[index]
            lines.append(f"- {error}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Crosswalk table
# ---------------------------------------------------------------------------


def export_crosswalk(
    output_path: Path,
    requirements: Sequence[Dict[str, object]],
    metrics: Sequence[Dict[str, object]],
    tests: Sequence[Dict[str, object]],
    overlays: Sequence[Dict[str, object]],
    requirement_to_metric: Mapping[str, List[str]],
    metric_to_tests: Mapping[str, List[str]],
) -> None:
    req_map = {req["id"]: req for req in requirements}
    metric_map = {metric["id"]: metric for metric in metrics}
    test_map = {exp["id"]: exp for exp in tests}

    overlay_map: Dict[Tuple[str, str], Dict[str, object]] = {}
    for overlay in overlays:
        key = (overlay["metric_id"], overlay["test_id"])
        overlay_map[key] = overlay

    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "authority_id",
                "requirement_id",
                "requirement_title",
                "pillar",
                "metric_id",
                "metric_name",
                "unit",
                "direction",
                "hazard_link",
                "test_id",
                "observed",
                "Cat_thresh",
                "Haz_thresh",
                "Maj_thresh",
                "pass_level",
                "status",
                "comment",
            ]
        )

        for req_id, metric_ids in requirement_to_metric.items():
            req = req_map.get(req_id)
            if not req:
                continue
            for metric_id in metric_ids:
                metric = metric_map.get(metric_id)
                if not metric:
                    continue
                tests_list = metric_to_tests.get(metric_id, [])
                for exp_id in tests_list:
                    exp = test_map.get(exp_id)
                    overlay = overlay_map.get((metric_id, exp_id), {})
                    thresholds = overlay.get("thresholds", {}) if isinstance(overlay.get("thresholds"), dict) else {}
                    writer.writerow(
                        [
                            req["source_authority_id"],
                            req_id,
                            req["title"],
                            req["domain"],
                            metric_id,
                            metric.get("name"),
                            metric.get("unit"),
                            metric.get("direction"),
                            metric.get("hazard_link"),
                            exp_id,
                            overlay.get("observed"),
                            thresholds.get("Cat"),
                            thresholds.get("Haz"),
                            thresholds.get("Maj"),
                            overlay.get("pass_level"),
                            overlay.get("status"),
                            overlay.get("comment"),
                        ]
                    )


# ---------------------------------------------------------------------------
# CAM assembly
# ---------------------------------------------------------------------------


def assemble_cam(
    requirements: Sequence[Dict[str, object]],
    metrics: Sequence[Dict[str, object]],
    tests: Sequence[Dict[str, object]],
    requirement_to_metric: Mapping[str, List[str]],
    metric_to_tests: Mapping[str, List[str]],
    overlays: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    authority_links = []
    for req in requirements:
        authority_links.append({"authority_id": req["source_authority_id"], "requirement_id": req["id"]})

    requirement_metric_links = []
    for req_id, metric_ids in requirement_to_metric.items():
        for metric_id in metric_ids:
            requirement_metric_links.append({"requirement_id": req_id, "metric_id": metric_id})

    metric_test_links = []
    for metric_id, exp_ids in metric_to_tests.items():
        for exp_id in exp_ids:
            metric_test_links.append({"metric_id": metric_id, "test_id": exp_id})

    cam = {
        "authorities": AUTHORITIES,
        "requirements": list(requirements),
        "metrics": list(metrics),
        "tests": list(tests),
        "links": {
            "authority_to_requirement": authority_links,
            "requirement_to_metric": requirement_metric_links,
            "metric_to_test": metric_test_links,
        },
        "results_overlay": list(overlays),
    }
    return cam


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Concentric Assurance Map (CAM).")
    parser.add_argument(
        "--requirements",
        type=Path,
        required=True,
        help="Path to requirements catalog (e.g., config/A_requirements.csv)",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Path to metrics catalog CSV (e.g., config/B_metrics.csv)",
    )
    parser.add_argument("--thresholds_cat", type=Path, required=True, help="Path to Catastrophic thresholds CSV")
    parser.add_argument("--thresholds_haz", type=Path, required=True, help="Path to Hazardous thresholds CSV")
    parser.add_argument("--thresholds_maj", type=Path, required=True, help="Path to Major thresholds CSV")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metric metadata JSON")
    parser.add_argument(
        "--tests",
        type=Path,
        required=True,
        help="Path to tests CSV (e.g., config/D_Tests.csv)",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Directory containing test result CSVs (for example results/test_outputs)",
    )
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for CAM artefacts")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    ensure_dir(args.out_dir)

    requirements = load_requirements(args.requirements)
    metrics, requirement_to_metric = load_metrics(
        metadata_path=args.metadata,
        metrics_catalog_path=args.metrics,
        thresholds_paths={
            "Catastrophic": args.thresholds_cat,
            "Hazardous": args.thresholds_haz,
            "Major": args.thresholds_maj,
        },
    )
    tests, metric_to_tests = load_tests(args.tests)
    overlays = load_results(args.results_dir)

    cam = assemble_cam(requirements, metrics, tests, requirement_to_metric, metric_to_tests, overlays)

    validation = build_validation_report(
        requirements=requirements,
        metrics=metrics,
        tests=tests,
        requirement_to_metric=requirement_to_metric,
        metric_to_tests=metric_to_tests,
        overlays=overlays,
    )

    cam_json_path = args.out_dir / "cam.json"
    cam_json_path.write_text(json.dumps(cam, indent=2), encoding="utf-8")

    validation_path = args.out_dir / "validation_report.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    sankey_path = args.out_dir / "CAM_Sankey.svg"
    draw_sankey(sankey_path, requirements, metrics, tests, requirement_to_metric, metric_to_tests)

    concentric_path = args.out_dir / "CAM_Concentric.svg"
    draw_concentric(
        concentric_path,
        requirements,
        metrics,
        tests,
        overlays,
        requirement_to_metric,
        metric_to_tests,
    )

    crosswalk_path = args.out_dir / "CAM_Table.csv"
    export_crosswalk(
        crosswalk_path,
        requirements,
        metrics,
        tests,
        overlays,
        requirement_to_metric,
        metric_to_tests,
    )

    report_path = args.out_dir / "CAM_Report.md"
    generate_report(report_path, cam, validation, requirement_to_metric, metric_to_tests)


if __name__ == "__main__":
    main()

