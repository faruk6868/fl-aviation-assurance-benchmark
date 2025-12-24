"""Create per-testbed heatmaps of pass ratios per metric category and algorithm.

The heatmaps follow the requested categories and cardinalities:
- Federated Learning Specific: 10 metrics
- Model Performance: 5
- Safety Linked Threshold: 4
- Drift & Lifecycle: 4
- Robustness & Stability: 4
- Verification & validation: 2
- Data Quality & Coverage: 3
- Runtime & Deployment: 1
- Bias & Fairness: 3
- Statistical Uncertainity: 1
- Explainability: 1

Each cell encodes pass_ratio = (#PASS / total metrics in that category for the TB)
for FedAvg, FedProx, and Scaffold. Missing categories for a TB are shown as N/A.
Color scale: 0 (red) -> 1 (green) with linear interpolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config" / "config_v2"
REPORT_DIR = PROJECT_ROOT / "results" / "assurance_reports"
OUTPUT_DIR = REPORT_DIR / "heatmaps"

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

# Metric-to-category mapping aligned with the requested category counts (N=38 metrics).
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute aggregated pass ratios across all TBs for each category/algorithm."""
    ratios = pd.DataFrame(index=CATEGORY_ORDER, columns=[a[0] for a in ALGORITHMS])
    counts = pd.DataFrame(index=CATEGORY_ORDER, columns=[a[0] for a in ALGORITHMS])

    # Pre-load all pass/fail files once per algorithm to speed aggregation.
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
            if total == 0:
                ratios.at[category, display_name] = np.nan
                counts.at[category, display_name] = "—"
            else:
                ratios.at[category, display_name] = passed / total
                counts.at[category, display_name] = f"{passed}/{total}"

    return ratios, counts


def plot_heatmap(label: str, ratios: pd.DataFrame, counts: pd.DataFrame) -> None:
    """Render and save a heatmap for the provided ratios/counts."""
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    discrete_colors = sns.color_palette("RdYlGn", 5)
    cmap = mcolors.ListedColormap(discrete_colors)
    bounds = np.linspace(0.0, 1.0, 6)  # five bands
    norm = mcolors.BoundaryNorm(bounds, ncolors=cmap.N)
    sns.heatmap(
        ratios.astype(float),
        ax=ax,
        cmap=cmap,
        norm=norm,
        annot=counts,
        fmt="",
        cbar_kws={
            "label": "Percentage of metrics passing threshold",
            "ticks": np.linspace(0.0, 1.0, 5),
            "format": mticker.PercentFormatter(xmax=1.0, decimals=0),
        },
        linewidths=0.5,
        linecolor="#e0e0e0",
        mask=ratios.isna(),
        square=False,
    )

    ax.set_title(f"{label} - Category Pass Heatmap", fontsize=12, pad=12)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Metric Category")

    # Grey-out NaNs explicitly
    sns.heatmap(
        ratios.isna(),
        ax=ax,
        cmap=sns.color_palette(["#f8f8f8"], as_cmap=True),
        cbar=False,
        linewidths=0.5,
        linecolor="#e0e0e0",
        mask=~ratios.isna(),
    )

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace(" ", "_").replace("/", "_")
    outfile = OUTPUT_DIR / f"{safe_label}_heatmap.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"Saved {outfile}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tb_metrics_map = load_tb_metric_ids()

    # Global aggregation across all TBs (single heatmap as requested).
    ratios, counts = compute_global_pass_ratio(tb_metrics_map)
    plot_heatmap("ALL_TBs", ratios, counts)

    summary_rows: List[Dict[str, object]] = []
    for category in CATEGORY_ORDER:
        for display_name in ratios.columns:
            summary_rows.append(
                {
                    "Scope": "ALL_TBs",
                    "Category": category,
                    "Algorithm": display_name,
                    "Pass_Ratio": ratios.at[category, display_name],
                    "Pass_Counts": counts.at[category, display_name],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "all_tb_heatmap_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote summary table to {summary_path}")


if __name__ == "__main__":
    main()
