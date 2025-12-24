"""
Generate paper-ready runtime figures with TB-04 dependency accounting.

For TB-08, TB-13 and TB-14, the total runtime should include the cost of
training TB-04 (their required base model). This script:
 1) Loads the existing runtime_summary.csv.
 2) Adds adjusted totals for TB-08/13/14 by adding the corresponding TB-04 time.
 3) Writes an enriched CSV for transparency.
 4) Produces three figures:
    - runtime_line_adjusted.png : line plot of adjusted runtimes across all test beds.
    - runtime_stacked_derived.png : stacked bars (TB-04 base + incremental) for derived test beds.
    - runtime_raw_vs_adjusted.png : raw vs adjusted comparison for derived test beds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "results" / "time_benchmarks" / "runtime_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "time_benchmarks"

DERIVED_TBS = {"TB-08", "TB-13", "TB-14"}
BASE_TB = "TB-04"


def _load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary CSV at {SUMMARY_PATH}")
    df = pd.read_csv(SUMMARY_PATH)
    required_cols = {"testbed", "algorithm", "seconds"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"runtime_summary.csv missing columns {required_cols - set(df.columns)}")
    return df


def _build_base_lookup(df: pd.DataFrame) -> Dict[str, float]:
    base_rows = df[(df["testbed"] == BASE_TB) & (df["status"] == "ok")]
    return dict(zip(base_rows["algorithm"], base_rows["seconds"]))


def _compute_adjusted(df: pd.DataFrame, base_lookup: Dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["is_derived"] = df["testbed"].isin(DERIVED_TBS)
    df["base_seconds"] = df["algorithm"].map(base_lookup)
    df["adjusted_seconds"] = df.apply(
        lambda row: row["seconds"] + row["base_seconds"]
        if row["is_derived"] and pd.notna(row["base_seconds"])
        else row["seconds"],
        axis=1,
    )
    df["incremental_seconds"] = np.where(df["is_derived"], df["seconds"], 0.0)
    df["inherited_from"] = np.where(df["is_derived"], BASE_TB, pd.NA)
    return df


def _tb_sort_key(tb_id: str) -> Tuple[int, str]:
    try:
        return int(tb_id.split("-")[1]), tb_id
    except Exception:
        return 0, tb_id


def _algo_colors(algos: Iterable[str]) -> Dict[str, str]:
    palette = plt.get_cmap("tab10")
    algos = list(sorted(set(algos)))
    return {algo: palette(idx % 10) for idx, algo in enumerate(algos)}


def plot_runtime_line(df: pd.DataFrame) -> Path:
    plot_df = df[df["status"] == "ok"].copy()
    plot_df["tb_order"] = plot_df["testbed"].apply(_tb_sort_key)
    plot_df.sort_values(by=["tb_order", "testbed"], inplace=True)
    pivot = plot_df.pivot_table(
        index=["tb_order", "testbed"],
        columns="algorithm",
        values="adjusted_seconds",
        aggfunc="mean",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = pivot.index.get_level_values("testbed")
    colors = _algo_colors(pivot.columns)
    for algo in pivot.columns:
        ax.plot(labels, pivot[algo], marker="o", label=algo.upper(), color=colors[algo])

    # Mark derived test beds
    derived_rows = plot_df[plot_df["is_derived"]]
    for _, row in derived_rows.iterrows():
        ax.scatter(
            row["testbed"],
            row["adjusted_seconds"],
            marker="x",
            color=colors.get(row["algorithm"], "black"),
            s=70,
            zorder=4,
        )

    ax.set_xlabel("Test bed")
    ax.set_ylabel("Wall-clock seconds")
    ax.set_title("Runtime across test beds (TB-08/13/14 include TB-04 cost)")
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    ax.legend(ncol=3)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "runtime_line_adjusted.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_stacked_derived(df: pd.DataFrame) -> Path:
    derived = df[df["is_derived"] & (df["status"] == "ok")].copy()
    if derived.empty:
        raise RuntimeError("No derived test beds found to plot.")

    derived.sort_values(by=["testbed", "algorithm"], key=lambda s: s.map(_tb_sort_key), inplace=True)
    colors = _algo_colors(derived["algorithm"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = []
    x_pos = np.arange(len(derived))
    width = 0.8

    ax.bar(
        x_pos,
        derived["base_seconds"],
        width=width,
        label=f"{BASE_TB} baseline",
        color="lightgray",
        edgecolor="black",
    )
    ax.bar(
        x_pos,
        derived["incremental_seconds"],
        width=width,
        bottom=derived["base_seconds"],
        label="Incremental (TB-08/13/14)",
        color=[colors[a] for a in derived["algorithm"]],
        alpha=0.8,
    )

    for tb, algo in zip(derived["testbed"], derived["algorithm"]):
        x_labels.append(f"{tb}\n{algo.upper()}")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Wall-clock seconds")
    ax.set_title("Derived test beds: TB-04 cost + incremental component")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path = OUTPUT_DIR / "runtime_stacked_derived.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_raw_vs_adjusted(df: pd.DataFrame) -> Path:
    derived = df[df["is_derived"] & (df["status"] == "ok")].copy()
    if derived.empty:
        raise RuntimeError("No derived test beds found to plot.")

    derived.sort_values(by=["testbed", "algorithm"], key=lambda s: s.map(_tb_sort_key), inplace=True)
    colors = _algo_colors(derived["algorithm"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(derived))
    width = 0.35

    ax.bar(x - width / 2, derived["seconds"], width, label="Raw (only TB-08/13/14)", color="lightgray")
    ax.bar(
        x + width / 2,
        derived["adjusted_seconds"],
        width,
        label="Adjusted (includes TB-04)",
        color=[colors[a] for a in derived["algorithm"]],
    )

    labels = [f"{tb}\n{algo.upper()}" for tb, algo in zip(derived["testbed"], derived["algorithm"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Wall-clock seconds")
    ax.set_title("Impact of including TB-04 cost on derived test beds")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path = OUTPUT_DIR / "runtime_raw_vs_adjusted.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    df = _load_summary()
    base_lookup = _build_base_lookup(df)
    enriched = _compute_adjusted(df, base_lookup)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    enriched_path = OUTPUT_DIR / "runtime_summary_with_tb04.csv"
    enriched.to_csv(enriched_path, index=False)

    paths = [
        plot_runtime_line(enriched),
        plot_stacked_derived(enriched),
        plot_raw_vs_adjusted(enriched),
    ]
    print("Wrote enriched summary to:", enriched_path)
    for p in paths:
        print("Saved figure:", p)


if __name__ == "__main__":
    main()

