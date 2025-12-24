"""Generate Trade-off Profile Radar Chart per algorithm using pass ratios.

Pass ratios are computed across all test beds for each metric category,
based on pass/fail CSVs under results/assurance_reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

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

CATEGORY_ORDER: List[str] = [
    "Federated Learning Specific",
    "Model Performance",
    "Safety Linked Threshold",
    "Drift & Lifecycle",
    "Robustness & Stability",
    "Verification & validation",
    "Data Quality & Coverage",
    "Runtime & Deployment",
    "Bias & Fairness",
    "Statistical Uncertainity",
    "Explainability",
]

METRIC_CATEGORY: Dict[str, str] = {
    # Federated Learning Specific (10)
    "M.FL.NONIID": "Federated Learning Specific",
    "M.FL.COMM_BYTES": "Federated Learning Specific",
    "M.FL.COMM_COMP": "Federated Learning Specific",
    "M.FL.CONV_TIME": "Federated Learning Specific",
    "M.FL.PRIV_DP": "Federated Learning Specific",
    "M.FL.PRIV_EPS": "Federated Learning Specific",
    "M.FL.PRIV_DELTA": "Federated Learning Specific",
    "M.FL.ATTACK_RES": "Federated Learning Specific",
    "M.FL.BYZ_TOL": "Federated Learning Specific",
    "M.FL.ATTACK_DET": "Federated Learning Specific",
    # Model Performance (5)
    "M.CLS.Pre": "Model Performance",
    "M.CLS.F1_Sc": "Model Performance",
    "M.CLS.AUC": "Model Performance",
    "M.PRG.PHM": "Model Performance",
    "M.GEN.GAP": "Model Performance",
    # Safety Linked Threshold (4)
    "M.SAF.ACC": "Safety Linked Threshold",
    "M.SAF.REC": "Safety Linked Threshold",
    "M.SAF.FPR_MAX": "Safety Linked Threshold",
    "M.SAF.FNR_MAX": "Safety Linked Threshold",
    # Drift & Lifecycle (4)
    "M.DRIFT.DETECT.MTTD": "Drift & Lifecycle",
    "M.DRIFT.DETECT.RECALL": "Drift & Lifecycle",
    "M.DRIFT.DETECT.FAR": "Drift & Lifecycle",
    "M.DRIFT.KS": "Drift & Lifecycle",
    # Robustness & Stability (4)
    "M.ROB.STAB_ODD": "Robustness & Stability",
    "M.ROB.DEGRAD": "Robustness & Stability",
    "M.INFER.STAB": "Robustness & Stability",
    "M.XFORM.DELTA": "Robustness & Stability",
    # Verification & validation (2)
    "M.PERF.COV": "Verification & validation",
    "M.PERF.STAT_REP": "Verification & validation",
    # Data Quality & Coverage (3)
    "M.DATA.ODD_COV": "Data Quality & Coverage",
    "M.DATA.SPLIT_INT": "Data Quality & Coverage",
    "M.DATA.PREPROC_AUDIT": "Data Quality & Coverage",
    # Runtime & Deployment (1)
    "M.RT.LAT": "Runtime & Deployment",
    # Bias & Fairness (3)
    "M.BIAS.COMPL": "Bias & Fairness",
    "M.BIAS.TRADEOFF": "Bias & Fairness",
    "M.PERF.CROSS_STAB": "Bias & Fairness",
    # Statistical Uncertainity (1)
    "M.STAT.CI": "Statistical Uncertainity",
    # Explainability (1)
    "M.XAI.STABILITY": "Explainability",
}


def load_tb_metric_ids() -> Dict[str, List[str]]:
    """Return mapping TB_ID -> metric list based on E.Test_beds.csv."""
    testbed_path = CONFIG_DIR / "E.Test_beds.csv"
    df = pd.read_csv(testbed_path)
    mapping: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        tb_id = str(row.get("Test ID", "")).strip()
        metrics_field = str(row.get("Metrics evaluated", "")).strip()
        if not tb_id or not metrics_field:
            continue
        metrics: List[str] = []
        for token in metrics_field.split(";"):
            token = token.strip()
            if not token:
                continue
            metric_id = token.split()[0].strip('",')
            if metric_id:
                metrics.append(metric_id)
        mapping[tb_id] = metrics
    return mapping


def load_pass_fail(tb_id: str, algo_code: str) -> pd.DataFrame | None:
    """Read the pass/fail CSV for a TB/algorithm pair."""
    path = REPORT_DIR / f"{tb_id}_{algo_code}_pass_fail.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["Metric_ID"] = df["Metric_ID"].astype(str).str.strip()
    df["Status"] = df["Status"].astype(str).str.strip().str.upper()
    return df


def compute_global_pass_ratio(
    tb_metrics_map: Dict[str, List[str]]
) -> pd.DataFrame:
    """Return pass ratio per category for each algorithm."""
    ratios = pd.DataFrame(index=CATEGORY_ORDER, columns=[a[0] for a in ALGORITHMS])
    pass_fail_cache: Dict[Tuple[str, str], pd.DataFrame | None] = {}
    for tb_id in tb_metrics_map:
        for _, code in ALGORITHMS:
            pass_fail_cache[(tb_id, code)] = load_pass_fail(tb_id, code)

    for display_name, code in ALGORITHMS:
        for category in CATEGORY_ORDER:
            total = 0
            passed = 0
            for tb_id, metric_ids in tb_metrics_map.items():
                df = pass_fail_cache.get((tb_id, code))
                if df is None:
                    continue
                status_lookup = dict(zip(df["Metric_ID"], df["Status"]))
                for metric_id in metric_ids:
                    if METRIC_CATEGORY.get(metric_id) != category:
                        continue
                    total += 1
                    if status_lookup.get(metric_id) == "PASS":
                        passed += 1
            ratios.at[category, display_name] = passed / total if total else np.nan
    return ratios


def plot_radar(ratios: pd.DataFrame) -> None:
    """Plot radar chart with one polygon per algorithm."""
    categories = CATEGORY_ORDER
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = {
        "FedAvg": "#1f77b4",
        "FedProx": "#2ca02c",
        "Scaffold": "#d62728",
    }

    for algo in ratios.columns:
        values = ratios[algo].astype(float).tolist()
        values += values[:1]
        ax.plot(angles, values, label=algo, color=colors.get(algo, None), linewidth=2)
        ax.fill(angles, values, color=colors.get(algo, None), alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title("Trade-off Profile Radar (Pass Ratio per Category)", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.05))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = OUTPUT_DIR / "tradeoff_radar.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved radar chart to {outfile}")


def main() -> None:
    tb_metrics_map = load_tb_metric_ids()
    ratios = compute_global_pass_ratio(tb_metrics_map)
    plot_radar(ratios)


if __name__ == "__main__":
    main()







