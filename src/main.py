"""Entry point for running the federated assurance pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.pipeline import AssurancePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one or more test bed measurement files.")
    parser.add_argument(
        "--tb",
        type=str,
        help="Test bed identifier (e.g., TB-03). Use with --algos to evaluate a single bed.",
    )
    parser.add_argument(
        "--algos",
        type=str,
        default="fedavg",
        help="Comma-separated list of algorithms (default: fedavg).",
    )
    parser.add_argument(
        "--measurements",
        type=str,
        help="Path to a specific measurement JSON (only valid with --tb).",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Evaluate all configured test beds (ignores --tb and --measurements).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress status messages.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    verbose = not args.quiet

    pipeline = AssurancePipeline(project_root, verbose=verbose)
    algos = [algo.strip() for algo in args.algos.split(",") if algo.strip()]
    if not algos:
        raise ValueError("No algorithms specified via --algos.")

    if args.run_all:
        run_all(pipeline, algos, verbose)
        return

    if not args.tb:
        raise ValueError("Please specify --tb <ID> or use --run-all.")

    measurement_path = Path(args.measurements) if args.measurements else None
    run_single(pipeline, args.tb, algos, measurement_path, verbose)


def run_single(
    pipeline: AssurancePipeline,
    tb_id: str,
    algos: list[str],
    measurement_path: Path | None,
    verbose: bool,
) -> None:
    for algo in algos:
        if verbose:
            print(f"[{tb_id}] Evaluating algorithm={algo}")
        try:
            pipeline.run(tb_id=tb_id, algorithm=algo, measurement_path=measurement_path)
        except FileNotFoundError as exc:
            print(f"[WARNING] {exc}")


def run_all(
    pipeline: AssurancePipeline,
    algos: list[str],
    verbose: bool,
) -> None:
    tb_ids = sorted(pipeline.assurance_config.test_beds.keys())
    if verbose:
        tb_names = ", ".join(tb_ids)
        print(f"[INFO] Running all test beds: {tb_names}")

    for tb_id in tb_ids:
        tb_name = pipeline.assurance_config.test_beds[tb_id].name or ""
        if verbose:
            print(f"\n=== {tb_id}: {tb_name} ===")
        run_single(pipeline, tb_id, algos, None, verbose)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()


