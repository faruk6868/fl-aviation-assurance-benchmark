"""Aggregate results and copy figures into paper_assets.

Usage:
  python scripts/generate_paper_assets.py --results-dir results --paper-assets-dir paper_assets
"""

import argparse
from pathlib import Path

from src.analysis.aggregate import copy_figures, write_all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate results and stage paper assets")
    parser.add_argument("--results-dir", default="results", help="Root results directory")
    parser.add_argument("--paper-assets-dir", default="paper_assets", help="Target directory for figures/tables")
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    assurance_dir = results_root / "assurance_reports"
    write_all_results(assurance_dir, results_root / "all_results.csv")

    sources = [results_root / "figures", results_root / "time_benchmarks"]
    copy_figures(sources, Path(args.paper_assets_dir) / "figures")


if __name__ == "__main__":
    main()
