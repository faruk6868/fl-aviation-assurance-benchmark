"""Entrypoint to run the assurance benchmark from a single config file."""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import yaml

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch optional in CI
    TORCH_AVAILABLE = False

from src.analysis.aggregate import copy_figures, write_all_results
from src.evaluation.pipeline import AssurancePipeline


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def snapshot_config(config: Dict, snapshot_dir: Path) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = snapshot_dir / f"config_snapshot_{ts}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


def _resolve_tb_ids(pipeline: AssurancePipeline, config: Dict) -> List[str]:
    include = config.get("test_beds", {}).get("include") or []
    run_all = config.get("run", {}).get("run_all", True)
    if run_all or not include:
        return sorted(pipeline.assurance_config.test_beds.keys())
    return [tb.strip() for tb in include if tb.strip()]


def _resolve_algos(config: Dict) -> List[str]:
    algos = config.get("run", {}).get("algos") or ["fedavg"]
    return [a.strip() for a in algos if a.strip()]


def run_from_config(config_path: Path, dry_run: bool = False, smoke_run: bool = False) -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_yaml(config_path)
    run_cfg = config.get("run", {})

    set_seed(run_cfg.get("seed"))

    snapshot_dir = project_root / run_cfg.get("snapshot_dir", "results/run_configs")
    snap = snapshot_config(config, snapshot_dir)
    verbose = not run_cfg.get("quiet", False)
    if verbose:
        print(f"[INFO] Loaded config from {config_path} (snapshot: {snap})")

    if dry_run:
        print("[INFO] Dry run enabled; no test beds executed.")
        return

    pipeline = AssurancePipeline(project_root, verbose=verbose)
    tb_ids = _resolve_tb_ids(pipeline, config)
    algos = _resolve_algos(config)

    if smoke_run and tb_ids:
        tb_ids = [tb_ids[0]]
        if verbose:
            print(f"[INFO] Smoke run: limiting to {tb_ids[0]}")

    for tb in tb_ids:
        for algo in algos:
            if verbose:
                print(f"[RUN] {tb} :: {algo}")
            try:
                pipeline.run(tb_id=tb, algorithm=algo)
            except Exception as exc:  # pragma: no cover - surface runtime issues
                print(f"[ERROR] {tb}/{algo} failed: {exc}", file=sys.stderr)
                raise

    # Aggregate results
    assurance_dir = project_root / run_cfg.get("assurance_reports_dir", "results/assurance_reports")
    aggregate_target = project_root / "results" / "all_results.csv"
    write_all_results(assurance_dir, aggregate_target)

    # Copy figures/time benchmarks into paper assets
    analysis_cfg = config.get("analysis", {})
    figure_sources = [project_root / Path(p) for p in analysis_cfg.get("figure_sources", [])]
    figure_targets = [project_root / Path(p) for p in analysis_cfg.get("figure_targets", ["paper_assets/figures"])]
    if figure_targets:
        copy_figures(figure_sources, figure_targets[0])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the FL assurance benchmark")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Load config and exit without running")
    parser.add_argument("--smoke-run", action="store_true", help="Run only the first test bed (for CI)")
    args = parser.parse_args(argv)

    run_from_config(Path(args.config), dry_run=args.dry_run, smoke_run=args.smoke_run)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
