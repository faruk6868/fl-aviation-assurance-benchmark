"""Plot pass/fail distribution per metric theme and algorithm."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config" / "config_v2"
REPORT_DIR = PROJECT_ROOT / "results" / "assurance_reports"
OUTPUT_PATH = PROJECT_ROOT / "results" / "figures" / "metric_theme_pass_fail.png"
TABLE_OUTPUT = PROJECT_ROOT / "results" / "assurance_reports" / "metric_theme_summary.md"
ALGOS = ["fedavg", "fedprox", "scaffold"]
SEVERITY_ORDER = {"FAIL": 3, "ALARM": 2, "MONITOR": 1, "PASS": 0}

THEME_ORDER = [
    "Federated Learning",
    "Model Performance",
    "Drift & Lifecycle",
    "Safety-Linked Thresholds",
    "Data Quality & Coverage",
    "Robustness & Stability",
    "Verification & Validation",
    "Bias & Fairness",
    "Runtime & Deployment",
    "Statistical Uncertainty",
    "Explainability",
]

THEME_MAP: Dict[str, str] = {
    "M.FL.NONIID": "Federated Learning",
    "M.FL.COMM_BYTES": "Federated Learning",
    "M.FL.COMM_COMP": "Federated Learning",
    "M.FL.CONV_TIME": "Federated Learning",
    "M.FL.PRIV_DP": "Federated Learning",
    "M.FL.PRIV_EPS": "Federated Learning",
    "M.FL.PRIV_DELTA": "Federated Learning",
    "M.FL.ATTACK_RES": "Federated Learning",
    "M.FL.BYZ_TOL": "Federated Learning",
    "M.FL.WORST_CLIENT": "Federated Learning",
    "M.FL.BENEFIT_EQ": "Federated Learning",
    "M.FL.ATTACK_DET": "Federated Learning",
    "M.CLS.Pre": "Model Performance",
    "M.CLS.F1_Sc": "Model Performance",
    "M.CLS.AUC": "Model Performance",
    "M.PRG.PHM": "Model Performance",
    "M.PERF.CROSS_STAB": "Bias & Fairness",
    "M.GEN.GAP": "Model Performance",
    "M.PERF.COV": "Verification & Validation",
    "M.PERF.STAT_REP": "Verification & Validation",
    "M.DATA.ODD_COV": "Data Quality & Coverage",
    "M.DATA.SPLIT_INT": "Data Quality & Coverage",
    "M.DATA.PREPROC_AUDIT": "Data Quality & Coverage",
    "M.SAF.ACC": "Safety-Linked Thresholds",
    "M.SAF.REC": "Safety-Linked Thresholds",
    "M.SAF.FPFN": "Safety-Linked Thresholds",
    "M.SAF.FNR_MAX": "Safety-Linked Thresholds",
    "M.SAF.FPR_MAX": "Safety-Linked Thresholds",
    "M.SAF.RMSE": "Safety-Linked Thresholds",
    "M.STAT.CI": "Statistical Uncertainty",
    "M.BIAS.COMPL": "Bias & Fairness",
    "M.BIAS.TRADEOFF": "Bias & Fairness",
    "M.XFORM.DELTA": "Model Performance",
    "M.ROB.STAB_ODD": "Robustness & Stability",
    "M.ROB.DEGRAD": "Robustness & Stability",
    "M.INFER.STAB": "Robustness & Stability",
    "M.DRIFT.DETECT.MTTD": "Drift & Lifecycle",
    "M.DRIFT.DETECT.RECALL": "Drift & Lifecycle",
    "M.DRIFT.DETECT.FAR": "Drift & Lifecycle",
    "M.DRIFT.KS": "Drift & Lifecycle",
    "M.RT.LAT": "Runtime & Deployment",
    "M.XAI.FIDE_STAB": "Explainability",
    "M.XAI.FIDELITY": "Explainability",
    "M.XAI.STABILITY": "Explainability",
}


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


def lookup_metric_status(metric_id: str, tb_ids: List[str], algo: str) -> Tuple[str, str]:
    rows: List[Dict[str, object]] = []
    for tb_id in tb_ids:
        df = load_pass_fail_file(tb_id, algo)
        if df is None:
            continue
        matches = df[df["Metric_ID"].astype(str).str.strip() == metric_id]
        if matches.empty:
            continue
        row = matches.iloc[0]
        status = str(row.get("Status", "")).strip().upper() or "MONITOR"
        observed = row.get("Observed_Value")
        unit = str(row.get("Unit", "")).strip()
        value_str = format_value(observed, unit)
        rows.append({"status": status, "value": value_str, "tb": tb_id})
    if not rows:
        return "MISSING", ""
    worst = max(rows, key=lambda r: SEVERITY_ORDER.get(r["status"], 1))
    return worst["status"], worst["value"]


def format_value(value: object, unit: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            num = float(value)
            text = f"{num:.4g}"
        except (TypeError, ValueError):
            text = str(value)
    return f"{text} {unit}".strip() if unit else text


def aggregate_theme_statuses() -> Dict[str, Dict[str, Dict[str, int]]]:
    metric_order, metric_to_tb = load_metric_order_and_mapping()
    counts: Dict[str, Dict[str, Dict[str, int]]] = {
        theme: {algo: {"PASS": 0, "NON_PASS": 0, "TOTAL": 0} for algo in ALGOS}
        for theme in THEME_ORDER
    }

    for metric_id in metric_order:
        theme = THEME_MAP.get(metric_id)
        if theme is None:
            continue
        tb_ids = metric_to_tb.get(metric_id, [])
        if not tb_ids:
            continue
        for algo in ALGOS:
            status, _ = lookup_metric_status(metric_id, tb_ids, algo)
            if status == "MISSING":
                continue
            bucket = counts[theme][algo]
            bucket["TOTAL"] += 1
            if status == "PASS":
                bucket["PASS"] += 1
            else:
                bucket["NON_PASS"] += 1

    return counts


def plot_theme_summary(counts: Dict[str, Dict[str, Dict[str, int]]]) -> None:
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(len(ALGOS), 1, figsize=(12, 10), sharex=True)
    if len(ALGOS) == 1:
        axes = [axes]

    x = np.arange(len(THEME_ORDER))
    bar_height = 0.8
    for idx, algo in enumerate(ALGOS):
        ax = axes[idx]
        pass_vals = []
        nonpass_vals = []
        labels = []
        for theme in THEME_ORDER:
            data = counts.get(theme, {}).get(algo, {})
            total = data.get("TOTAL", 0)
            labels.append(theme)
            if total == 0:
                pass_vals.append(0)
                nonpass_vals.append(0)
                continue
            pass_ratio = data.get("PASS", 0) / total
            nonpass_ratio = data.get("NON_PASS", 0) / total
            pass_vals.append(pass_ratio)
            nonpass_vals.append(nonpass_ratio)

        pass_vals = np.array(pass_vals)
        nonpass_vals = np.array(nonpass_vals)

        ax.barh(x, nonpass_vals, color="#d95f02", label="Non-PASS")
        ax.barh(x, pass_vals, left=nonpass_vals, color="#1b9e77", label="PASS")
        ax.set_yticks(x)
        ax.set_yticklabels(labels if idx == 0 else [""] * len(labels))
        ax.set_xlim(0, 1)
        ax.set_title(f"{algo.capitalize()} (share of metrics meeting acceptance)", loc="left")
        for i, (p, npv) in enumerate(zip(pass_vals, nonpass_vals)):
            total_ratio = p + npv
            if total_ratio == 0:
                ax.text(0.02, i, "N/A", va="center", ha="left", color="#666666")
                continue
            ax.text(
                npv + p / 2,
                i,
                f"{p*100:.0f}%",
                va="center",
                ha="center",
                color="white",
                fontweight="bold",
            )
        ax.grid(axis="x", linestyle=":", alpha=0.5)

    axes[-1].set_xticks(np.linspace(0, 1, 5))
    axes[-1].set_xticklabels([f"{int(v*100)}%" for v in np.linspace(0, 1, 5)])
    axes[-1].set_xlabel("Percent of metrics passing within theme")
    axes[0].legend(loc="lower right")

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    counts = aggregate_theme_statuses()
    plot_theme_summary(counts)
    write_markdown_table(counts)
    print(f"Wrote theme status figure to {OUTPUT_PATH}")
    print(f"Wrote theme table to {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()


def write_markdown_table(counts: Dict[str, Dict[str, Dict[str, int]]]) -> None:
    lines = [
        "# Metric Theme Status\n\n",
        "| Theme | Algorithm | Metrics (PASS / Total) | Pass % |\n",
        "| --- | --- | --- | --- |\n",
    ]
    for theme in THEME_ORDER:
        theme_counts = counts.get(theme, {})
        for algo in ALGOS:
            data = theme_counts.get(algo, {"PASS": 0, "TOTAL": 0})
            total = data.get("TOTAL", 0)
            passed = data.get("PASS", 0)
            pct = (passed / total * 100) if total else 0.0
            lines.append(
                f"| {theme} | {algo.capitalize()} | {passed} / {total} | {pct:.1f}% |\n"
            )
    TABLE_OUTPUT.write_text("".join(lines), encoding="utf-8")

