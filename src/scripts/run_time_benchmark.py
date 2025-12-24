"""Batch runtime benchmark for all test beds across multiple algorithms."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

# Ensure project root is on path when executed as a script
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.run_testbed import (
    run_tb01,
    run_tb02,
    run_tb03,
    run_tb04,
    run_tb05,
    run_tb06,
    run_tb07,
    run_tb08,
    run_tb09,
    run_tb10,
    run_tb11,
    run_tb12,
    run_tb13,
    run_tb14,
)

DEFAULT_ALGORITHMS = ("fedavg", "fedprox", "scaffold")

TESTBED_RUNNERS = {
    "TB-01": run_tb01,
    "TB-02": run_tb02,
    "TB-03": run_tb03,
    "TB-04": run_tb04,
    "TB-05": run_tb05,
    "TB-06": run_tb06,
    "TB-07": run_tb07,
    "TB-08": run_tb08,
    "TB-09": run_tb09,
    "TB-10": run_tb10,
    "TB-11": run_tb11,
    "TB-12": run_tb12,
    "TB-13": run_tb13,
    "TB-14": run_tb14,
}

# Test beds that reuse artifacts from another test bed (e.g., TB-13/14 depend on TB-04 models).
# For these, we will show the baseline test bed's timing instead of re-running.
DERIVED_FROM = {
    "TB-13": "TB-04",
    "TB-14": "TB-04",
}

# Only these test beds accept a rounds_override parameter.
ROUND_OVERRIDES = {
    "TB-03",
    "TB-04",
    "TB-05",
    "TB-06",
    "TB-07",
    "TB-08",
    "TB-09",
    "TB-10",
    "TB-11",
    "TB-12",
}


def _tb_sort_key(tb_id: str) -> int:
    try:
        return int(tb_id.split("-")[1])
    except Exception:
        return 0


def _plot_runtime(df: pd.DataFrame, output_dir: Path, rounds: int) -> Path | None:
    plot_df = df[df["status"] == "ok"].copy()
    if plot_df.empty:
        print("[WARN] No successful runs to plot.")
        return None

    plot_df["tb_order"] = plot_df["testbed"].apply(_tb_sort_key)
    plot_df.sort_values(by=["tb_order", "testbed"], inplace=True)
    pivot = plot_df.pivot_table(
        index=["tb_order", "testbed"],
        columns="algorithm",
        values="effective_seconds",
        aggfunc="mean",
    )
    pivot.sort_index(inplace=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = pivot.index.get_level_values("testbed")
    lines = []
    for algo in pivot.columns:
        line = ax.plot(labels, pivot[algo], marker="o", label=algo.upper())
        lines.extend(line)

    # Mark inherited timings with an 'x' overlay.
    color_map = {col: line.get_color() for col, line in zip(pivot.columns, lines)}
    inherited_rows = plot_df[plot_df["inherited_from"].notna()]
    for _, row in inherited_rows.iterrows():
        tb = row["testbed"]
        algo = row["algorithm"]
        color = color_map.get(algo, None)
        ax.scatter(tb, row["effective_seconds"], marker="x", color=color, s=70, zorder=5)

    ax.set_xlabel("Test bed")
    ax.set_ylabel("Wall-clock seconds (effective)")
    ax.set_title(
        f"Runtime comparison across test beds ({rounds} rounds)\n"
        "(inherited timings marked with 'x')"
    )
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    ax.legend()

    fig.tight_layout()
    plot_path = output_dir / "runtime_line_plot.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    return plot_path


def _inherit_timings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute effective_seconds using baseline timings for derived test beds."""
    df = df.copy()
    df["effective_seconds"] = df["seconds"]
    df["inherited_from"] = pd.NA

    for tb, base_tb in DERIVED_FROM.items():
        for algo in df["algorithm"].unique():
            base_row = df[
                (df["testbed"] == base_tb)
                & (df["algorithm"] == algo)
                & (df["status"] == "ok")
            ]
            if base_row.empty:
                continue
            base_seconds = base_row.iloc[0]["seconds"]
            mask = (df["testbed"] == tb) & (df["algorithm"] == algo)
            df.loc[mask, "effective_seconds"] = base_seconds
            df.loc[mask, "inherited_from"] = base_tb

    return df


def run_benchmark(algorithms: Sequence[str], rounds: int) -> Dict[str, Path | None]:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "results" / "time_benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    for tb_id in sorted(TESTBED_RUNNERS.keys(), key=_tb_sort_key):
        runner = TESTBED_RUNNERS[tb_id]
        for algo in algorithms:
            start = time.perf_counter()
            status = "ok"
            error: str | None = None
            try:
                if tb_id in ROUND_OVERRIDES:
                    runner(project_root, algo, rounds_override=rounds)
                else:
                    runner(project_root, algo)
            except Exception as exc:  # pragma: no cover - runtime logging path
                status = "error"
                error = str(exc)
            elapsed = time.perf_counter() - start
            records.append(
                {
                    "testbed": tb_id,
                    "algorithm": algo,
                    "seconds": elapsed,
                    "status": status,
                    "error": error,
                }
            )
            print(f"[{tb_id}] {algo}: {status} in {elapsed:.1f}s")

    df = pd.DataFrame(records)
    df = _inherit_timings(df)
    summary_path = output_dir / "runtime_summary.csv"
    df.to_csv(summary_path, index=False)
    plot_path = _plot_runtime(df, output_dir, rounds)
    return {"summary_csv": summary_path, "plot": plot_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark runtime across all test beds.")
    parser.add_argument(
        "--algorithms",
        type=str,
        default=",".join(DEFAULT_ALGORITHMS),
        help="Comma-separated algorithms to run (default: fedavg,fedprox,scaffold).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=50,
        help="Federated rounds override for applicable test beds (default: 50).",
    )
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    algos = [algo.strip().lower() for algo in args.algorithms.split(",") if algo.strip()]
    if not algos:
        raise ValueError("At least one algorithm must be provided.")
    run_benchmark(algorithms=algos, rounds=args.rounds)

