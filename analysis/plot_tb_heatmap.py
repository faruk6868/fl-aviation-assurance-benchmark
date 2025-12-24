"""Generate a test-bed status heatmap (PASS / FAIL / MONITOR)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Patch


RESULT_DIR = Path("results/assurance_reports")
OUTPUT_PATH = Path("results/figures/tb_status_heatmap.png")
ALGOS = ["fedavg", "fedprox", "scaffold"]
ALGO_SHORT = {"fedavg": "A", "fedprox": "B", "scaffold": "C"}
ALGO_LONG = {"fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAF"}


def _load_status_counts() -> Dict[Tuple[str, str], Dict[str, int]]:
    status_counts: Dict[Tuple[str, str], Dict[str, int]] = {}
    for csv_path in sorted(RESULT_DIR.glob("TB-*_pass_fail.csv")):
        tb_id, algo, _ = csv_path.stem.split("_", 2)
        key = (tb_id, algo)
        stats = status_counts.setdefault(
            key, {"pass": 0, "fail": 0, "monitor": 0, "alarm": 0}
        )
        with csv_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                status = row["Status"].strip().upper()
                if "FAIL" in status:
                    stats["fail"] += 1
                elif "PASS" in status:
                    stats["pass"] += 1
                elif "ALARM" in status:
                    stats["alarm"] += 1
                else:
                    stats["monitor"] += 1
    return status_counts


def _build_matrix(status_counts: Dict[Tuple[str, str], Dict[str, int]]):
    tb_ids = sorted({tb for tb, _ in status_counts.keys()}, key=lambda x: int(x.split("-")[1]))
    matrix = np.full((len(tb_ids), len(ALGOS)), np.nan)
    labels: List[List[str]] = [["" for _ in ALGOS] for _ in tb_ids]

    for i, tb in enumerate(tb_ids):
        for j, algo in enumerate(ALGOS):
            stats = status_counts.get((tb, algo))
            if not stats:
                continue
            if stats["fail"] > 0:
                matrix[i, j] = -1
                labels[i][j] = f"F{stats['fail']}"
            elif stats["pass"] > 0:
                matrix[i, j] = 1
                labels[i][j] = f"P{stats['pass']}"
            else:
                monitor_total = stats["monitor"] + stats["alarm"]
                matrix[i, j] = 0
                labels[i][j] = f"M{monitor_total}" if monitor_total else "—"
    return tb_ids, matrix, labels


def main() -> None:
    status_counts = _load_status_counts()
    tb_ids, matrix, labels = _build_matrix(status_counts)

    cmap = mcolors.ListedColormap(["#d73027", "#f0f0f0", "#1a9850"])
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(len(ALGOS)))
    ax.set_xticklabels([ALGO_SHORT.get(algo, algo[0].upper()) for algo in ALGOS], fontsize=12)
    ax.set_yticks(np.arange(len(tb_ids)))
    ax.set_yticklabels(tb_ids, fontsize=11)
    ax.tick_params(axis="x", pad=8, labelrotation=0)
    ax.tick_params(axis="y", pad=6)
    mapping = ", ".join(f"{ALGO_SHORT[a]}={ALGO_LONG[a]}" for a in ALGOS)
    ax.set_xlabel(f"Federated Algorithm ({mapping})", fontsize=12)
    ax.set_ylabel("Test Bed", fontsize=12)
    ax.set_title("Assurance Status Heatmap")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=9, color="black")

    legend_elements = [
        Patch(facecolor="#1a9850", label="PASS (>=1 gating metric, no FAIL)"),
        Patch(facecolor="#f0f0f0", label="MONITOR / no gated metric"),
        Patch(facecolor="#d73027", label="FAIL present"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fontsize=10,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.24, bottom=0.12, right=0.9, top=0.9)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

