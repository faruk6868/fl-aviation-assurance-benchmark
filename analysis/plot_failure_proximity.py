"""Failure Proximity Analysis: normalized compliance scores for failed metrics.

Score heuristic (per metric/algorithm):
- Inside thresholds => 1.0
- Below min (if min exists) => value / min  (clipped 0..1)
- Above max (if max exists) => max / value (clipped 0..1)
- Only one bound present: use that bound’s ratio. If no bounds -> NaN.

We collect metrics that failed in at least one algorithm, then plot all algorithms
for those metrics (PASS will appear with score=1 if within bounds).
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


def score_against_threshold(
    value: float,
    min_thr: float | float("nan"),
    max_thr: float | float("nan"),
    status: str,
) -> float:
    """Normalized closeness; if no thresholds and status!=PASS, force 0 to surface the failure."""
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
    if has_min:
        if value < min_thr:
            return max(0.0, min(1.0, value / min_thr))
        return 1.0
    if has_max:
        if value > max_thr:
            return max(0.0, min(1.0, max_thr / value))
        return 1.0
    return 0.0 if status != "PASS" else float("nan")


def collect_scores(metric_ids: List[str]) -> Tuple[pd.DataFrame, List[str]]:
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

    rows = []
    metrics_with_fail = set()

    for metric_id in metric_ids:
        tbs = [tb for tb, mids in tb_map.items() if metric_id in mids]
        if not tbs:
            continue
        # First detect if metric fails anywhere
        fails_any = False
        cache: Dict[Tuple[str, str], pd.DataFrame | None] = {}
        for tb in tbs:
            for _, code in ALGORITHMS:
                cache[(tb, code)] = load_pass_fail(tb, code)
        for tb in tbs:
            for algo_label, code in ALGORITHMS:
                df = cache.get((tb, code))
                if df is None:
                    continue
                match = df[df["Metric_ID"] == metric_id]
                if match.empty:
                    continue
                status = str(match.iloc[0].get("Status", "")).upper()
                if status != "PASS":
                    fails_any = True
                    break
            if fails_any:
                break
        if not fails_any:
            continue
        metrics_with_fail.add(metric_id)

        # Collect scores for all algos (first TB hit per algo)
        for algo_label, code in ALGORITHMS:
            chosen_row = None
            for tb in tbs:
                df = cache.get((tb, code))
                if df is None:
                    continue
                match = df[df["Metric_ID"] == metric_id]
                if match.empty:
                    continue
                chosen_row = match.iloc[0]
                break
            if chosen_row is None:
                continue
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

            status = str(chosen_row.get("Status", "")).upper()
            score = score_against_threshold(obs_val, min_thr, max_thr, status)
            rows.append(
                {
                    "Metric": metric_id,
                    "Algorithm": algo_label,
                    "Observed": obs_val,
                    "Min": min_thr,
                    "Max": max_thr,
                    "Score": score,
                }
            )

    df_scores = pd.DataFrame(rows)
    ordered_metrics = sorted(metrics_with_fail)
    return df_scores, ordered_metrics


def plot_failure_proximity(df_scores: pd.DataFrame, metric_order: List[str]) -> None:
    if df_scores.empty:
        print("No failures found.")
        return
    # Order metrics as provided
    df_scores["Metric"] = pd.Categorical(df_scores["Metric"], categories=metric_order, ordered=True)
    df_scores = df_scores.sort_values("Metric")

    colors = {
        "FedAvg": "#1f77b4",
        "FedProx": "#ff7f0e",
        "Scaffold": "#2ca02c",
    }
    markers = {
        "FedAvg": "o",
        "FedProx": "s",
        "Scaffold": "^",
    }
    # Small x-jitter to avoid marker overlap when scores coincide
    offsets = {
        "FedAvg": -0.01,
        "FedProx": 0.0,
        "Scaffold": 0.01,
    }

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(9, 9))

    for algo, group in df_scores.groupby("Algorithm"):
        ax.scatter(
            group["Score"] + offsets.get(algo, 0.0),
            group["Metric"],
            label=algo,
            color=colors.get(algo, None),
            marker=markers.get(algo, "o"),
            s=55,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )

    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2, label="Acceptance Threshold")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Normalized Compliance Score (1.0 = Threshold)")
    ax.set_ylabel("Metric")
    ax.set_title("Failure Proximity Analysis\n(How close were the failures to passing?)", pad=12)
    ax.legend(loc="lower right")
    plt.subplots_adjust(left=0.3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = OUTPUT_DIR / "failure_proximity.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"Saved {outfile}")


def main() -> None:
    metrics = load_metric_list()
    df_scores, metric_order = collect_scores(metrics)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_scores.to_csv(OUTPUT_DIR / "failure_proximity.csv", index=False)
    plot_failure_proximity(df_scores, metric_order)


if __name__ == "__main__":
    main()



