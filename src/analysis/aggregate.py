"""Aggregation helpers for assurance outputs and paper assets."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List

import pandas as pd


PASS_FAIL_SUFFIX = "_pass_fail.csv"


def collect_pass_fail(assurance_dir: Path) -> pd.DataFrame:
    assurance_dir = assurance_dir.expanduser().resolve()
    records: List[pd.DataFrame] = []
    if not assurance_dir.exists():
        return pd.DataFrame()

    for csv_path in assurance_dir.glob(f"*{PASS_FAIL_SUFFIX}"):
        parts = csv_path.stem.split("_")
        if len(parts) < 3:
            continue
        tb_id = parts[0]
        algo = "_".join(parts[1:-2]) if len(parts) > 3 else parts[1]
        df = pd.read_csv(csv_path)
        df.insert(0, "tb_id", tb_id)
        df.insert(1, "algorithm", algo)
        records.append(df)

    if not records:
        return pd.DataFrame()

    merged = pd.concat(records, ignore_index=True)
    merged.rename(columns={
        "Metric_ID": "metric_id",
        "Result": "result",
        "Value": "value",
        "Threshold": "threshold",
    }, inplace=True)
    return merged


def write_all_results(assurance_dir: Path, output_csv: Path) -> Path | None:
    df = collect_pass_fail(assurance_dir)
    if df.empty:
        print("[INFO] No pass/fail CSVs found to aggregate.")
        return None
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Aggregated results written to {output_csv}")
    return output_csv


def copy_figures(sources: Iterable[Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in sources:
        if not source.exists():
            continue
        for path in source.glob("**/*"):
            if path.is_dir():
                continue
            if path.suffix.lower() not in {".png", ".csv", ".html", ".svg"}:
                continue
            rel = path.relative_to(source)
            dest = target_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            copied += 1
    print(f"[INFO] Copied {copied} figure/benchmark artifacts into {target_dir}")
