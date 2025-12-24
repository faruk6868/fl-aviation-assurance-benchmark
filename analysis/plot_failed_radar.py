"""Radar chart for only failed metrics showing closeness to thresholds per algorithm.

Score heuristic:
- If value inside bounds -> 1.0
- If below min (when defined) -> value/min  (0..1)
- If above max (when defined) -> max/value  (0..1)
- If only one bound exists, use the corresponding ratio; if both missing -> NaN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config" / "config_v2"
REPORT_DIR = PROJECT_ROOT / "results" / "assurance_reports"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

ALGORITHMS: List[Tuple[str, str]] = [
    ("FedAvg", "fedavg"),
    ("FedProx", "fedprox"),
    ("Scaffold", "scaffold"),
]


def load_metric_list() -> List[str]:
    """Return unique metric IDs from E.Test_beds.csv."""
    df = pd.read_csv(CONFIG_DIR / "E.Test_beds.csv")
    metrics: List[str] = []
    for s in df["Metrics evaluated"].dropna():
        for tok in str(s).split(";"):
            mid = tok.strip().split()[0].strip('",')
            if mid and mid not in metrics:
                metrics.append(mid)
    return metrics


def load_pass_fail(tb_id: str, algo_code: str) -> pd.DataFrame | None:
    path = REPORT_DIR / f"{tb_id}_{algo_code}_pass_fail.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["Metric_ID"] = df["Metric_ID"].astype(str).str.strip()
    df["Status"] = df["Status"].astype(str).str.strip().str.upper()
    return df


def score_against_threshold(value: float, min_thr: float | float("nan"), max_thr: float | float("nan")) -> float:
    """Return closeness score in [0,1]."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    has_min = min_thr is not None and not (isinstance(min_thr, float) and math.isnan(min_thr))
    has_max = max_thr is not None and not (isinstance(max_thr, float) and math.isnan(max_thr))

    if has_min and has_max:
        if value < min_thr:
            return max(0.0, min(1.0, value / min_thr))
        if value > max_thr:
            return max(0.0, min(1.0, max_thr / value))
        return 1.0
    if has_min and not has_max:
        if value < min_thr:
            return max(0.0, min(1.0, value / min_thr))
        return 1.0
    if has_max and not has_min:
        if value > max_thr:
            return max(0.0, min(1.0, max_thr / value))
        return 1.0
    return float("nan")


def collect_failed_metrics(metric_ids: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Return DataFrame with scores for failed metrics and the ordered metric list."""
    df_tb = pd.read_csv(CONFIG_DIR / "E.Test_beds.csv")
    tb_map: Dict[str, List[str]] = {}
    for _, row in df_tb.iterrows():
        tb_id = str(row.get("Test ID", "")).strip()
        mids = []
        for tok in str(row.get("Metrics evaluated", "")).split(";"):
            mid = tok.strip().split()[0].strip('",')
            if mid:
                mids.append(mid)
        tb_map[tb_id] = mids

    records = []
    failed_metrics: List[str] = []

    for metric_id in metric_ids:
        # gather all TBs that include this metric
        tbs = [tb for tb, mids in tb_map.items() if metric_id in mids]
        if not tbs:
            continue
        for algo_label, algo_code in ALGORITHMS:
            chosen_row = None
            for tb in tbs:
                df = load_pass_fail(tb, algo_code)
                if df is None:
                    continue
                matches = df[df["Metric_ID"] == metric_id]
                if matches.empty:
                    continue
                row = matches.iloc[0]
                status = str(row.get("Status", "")).upper()
                if status != "PASS":
                    chosen_row = row
                    break
            if chosen_row is None:
                continue

            status = str(chosen_row.get("Status", "")).upper()
            obs = chosen_row.get("Observed_Value")
            try:
                obs_val = float(obs)
            except (TypeError, ValueError):
                obs_val = float("nan")
            min_thr = chosen_row.get("Min_Threshold")
            max_thr = chosen_row.get("Max_Threshold")
            try:
                min_thr = float(min_thr)
            except (TypeError, ValueError):
                min_thr = float("nan")
            try:
                max_thr = float(max_thr)
            except (TypeError, ValueError):
                max_thr = float("nan")

            score = score_against_threshold(obs_val, min_thr, max_thr)
            failed_metrics.append(metric_id)
            records.append(
                {
                    "Metric": metric_id,
                    "Algorithm": algo_label,
                    "Status": status,
                    "Observed": obs_val,
                    "Min": min_thr,
                    "Max": max_thr,
                    "Score": score,
                }
            )

    failed_order = []
    for mid in metric_ids:
        if mid in failed_metrics and mid not in failed_order:
            failed_order.append(mid)

    df_scores = pd.DataFrame(records)
    return df_scores, failed_order


def plot_radar(df_scores: pd.DataFrame, metrics_order: List[str]) -> None:
    """Plot radar using scores per metric for three algorithms."""
    if df_scores.empty or not metrics_order:
        print("No failed metrics to plot.")
        return

    angles = np.linspace(0, 2 * np.pi, len(metrics_order), endpoint=False).tolist()
    angles += angles[:1]

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    colors = {
        "FedAvg": "#1f77b4",
        "FedProx": "#2ca02c",
        "Scaffold": "#d62728",
    }

    for algo_label, _ in ALGORITHMS:
        series = (
            df_scores[df_scores["Algorithm"] == algo_label]
            .set_index("Metric")
            .reindex(metrics_order)
        )
        values = series["Score"].astype(float).tolist()
        values += values[:1]
        ax.plot(angles, values, label=algo_label, color=colors.get(algo_label), linewidth=2)
        ax.fill(angles, values, color=colors.get(algo_label), alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_order, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title("Failed Metrics – Closeness to Threshold", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.05))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = OUTPUT_DIR / "failed_metrics_radar.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


def main() -> None:
    metric_ids = load_metric_list()
    df_scores, metrics_order = collect_failed_metrics(metric_ids)
    # save table
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_scores.to_csv(OUTPUT_DIR / "failed_metrics_radar.csv", index=False)
    plot_radar(df_scores, metrics_order)


if __name__ == "__main__":
    main()







