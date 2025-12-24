"""Utilities for cross-algorithm federated learning benchmarking."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.runner import TestOutput, TestRunner
from src.federated import FederatedAlgorithm


DEFAULT_TESTS = ("T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T12")
DEFAULT_ALGORITHMS = (
    FederatedAlgorithm.FEDAVG.value,
    FederatedAlgorithm.FEDPROX.value,
    FederatedAlgorithm.SCAFFOLD.value,
)
KEY_METRICS = ("M3", "M5", "M7", "M10", "M15", "M21", "M22", "M36")


class AlgorithmComparisonSuite:
    """Runs a matrix of tests × algorithms and generates publication-ready assets."""

    def __init__(
        self,
        project_root: Path,
        output_root: Optional[Path] = None,
        verbose: bool = True,
    ) -> None:
        self.project_root = Path(project_root)
        self.output_root = output_root or (self.project_root / "results" / "algorithm_benchmarks")
        self.json_dir = self.output_root / "json"
        self.tables_dir = self.output_root / "tables"
        self.figures_dir = self.output_root / "figures"
        self.runner = TestRunner(self.project_root, verbose=verbose)
        self.verbose = verbose

    def run(
        self,
        tests: Sequence[str],
        algorithms: Sequence[str | FederatedAlgorithm],
        algorithm_params: Optional[Mapping[str, Mapping[str, object]]] = None,
        verbose: Optional[bool] = None,
    ) -> Dict[str, object]:
        """Execute the benchmark suite and persist JSON/tables/figures."""

        verbosity = self.verbose if verbose is None else verbose
        normalized_params = self._normalize_algorithm_params(algorithm_params)
        algorithm_outputs: Dict[str, List[TestOutput]] = {}
        executed_algorithms: List[str] = []

        for algorithm in algorithms:
            algorithm_name = self._normalize_algorithm_name(algorithm)
            params = normalized_params.get(algorithm_name)
            if verbosity:
                print(
                    f"[Comparison] Running algorithms={algorithm_name} on tests={','.join(tests)}",
                    flush=True,
                )

            outputs = self.runner.run_sequence(
                tests,
                verbose=verbosity,
                algorithm=algorithm_name,
                algorithm_params=params,
            )
            algorithm_outputs[algorithm_name] = outputs
            executed_algorithms.append(algorithm_name)
            self._write_json_payloads(outputs, algorithm_name, params)

        metrics_df = self._metrics_dataframe(algorithm_outputs)
        combined_df = self._write_tables(metrics_df, executed_algorithms)
        figure_paths = self._plot_metric_panels(combined_df)

        return {
            "metrics": combined_df,
            "json_dir": self.json_dir,
            "tables_dir": self.tables_dir,
            "figures": figure_paths,
        }

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def _write_json_payloads(
        self,
        outputs: Sequence[TestOutput],
        algorithm: str,
        algorithm_params: Optional[Mapping[str, object]],
    ) -> None:
        self.json_dir.mkdir(parents=True, exist_ok=True)
        for output in outputs:
            payload = self._serialize_output(output, algorithm, algorithm_params)
            output_path = self.json_dir / f"{output.test_id}_{algorithm}.json"
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _serialize_output(
        self,
        output: TestOutput,
        algorithm: str,
        algorithm_params: Optional[Mapping[str, object]],
    ) -> Dict[str, object]:
        return {
            "test_id": output.test_id,
            "algorithm": algorithm,
            "algorithm_params": algorithm_params or {},
            "metrics": [self._sanitize_metric(metric.to_dict()) for metric in output.metrics],
            "context_summary": self._summarize_context(output.context),
        }

    def _sanitize_metric(self, metric_dict: Mapping[str, object]) -> Dict[str, object]:
        sanitized: Dict[str, object] = {}
        for key, value in metric_dict.items():
            sanitized[key] = self._coerce_scalar(value)
        return sanitized

    def _coerce_scalar(self, value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (np.ndarray, list, tuple)):
            return self._summarize_value(value)
        return value

    def _summarize_context(self, context: Mapping[str, object]) -> Dict[str, object]:
        summary: Dict[str, object] = {}
        for key, value in context.items():
            summary[key] = self._summarize_value(value)
        return summary

    def _summarize_value(self, value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        if isinstance(value, np.ndarray):
            return self._summarize_array(value)
        if isinstance(value, (list, tuple)):
            arr = np.asarray(value)
            if arr.dtype.kind in "biuf" and arr.size > 0:
                return self._summarize_array(arr)
            return {"type": "sequence", "length": len(value)}
        if isinstance(value, Mapping):
            truncated: Dict[str, object] = {}
            for idx, (nested_key, nested_value) in enumerate(value.items()):
                if idx >= 12:
                    truncated["_truncated"] = len(value)
                    break
                truncated[nested_key] = self._summarize_value(nested_value)
            return truncated
        return str(value)

    def _summarize_array(self, array_like: Iterable[object]) -> Dict[str, object]:
        arr = np.asarray(array_like, dtype=float)
        if arr.size == 0:
            return {"shape": list(arr.shape), "mean": None, "std": None, "min": None, "max": None}
        return {
            "shape": list(arr.shape),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    # ------------------------------------------------------------------
    # Tabular + visual summaries
    # ------------------------------------------------------------------
    def _metrics_dataframe(self, outputs: Mapping[str, Sequence[TestOutput]]) -> pd.DataFrame:
        records: List[Dict[str, object]] = []
        for algorithm, test_outputs in outputs.items():
            for output in test_outputs:
                for metric in output.metrics:
                    metric_dict = metric.to_dict()
                    metric_dict["Test"] = output.test_id
                    metric_dict["Algorithm"] = algorithm
                    records.append(metric_dict)
        if not records:
            return pd.DataFrame(columns=["Test", "Algorithm", "Metric_ID", "Observed_Value"])

        df = pd.DataFrame(records)
        if "Observed_Value" in df.columns:
            df["Observed_Value"] = pd.to_numeric(df["Observed_Value"], errors="ignore")
        df.sort_values(by=["Metric_ID", "Test", "Algorithm"], inplace=True)
        return df.reset_index(drop=True)

    def _write_tables(self, df: pd.DataFrame, algorithms_in_run: Sequence[str]) -> pd.DataFrame:
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        all_metrics_path = self.tables_dir / "all_metrics.csv"

        combined_df = df.copy()
        if all_metrics_path.exists():
            existing = pd.read_csv(all_metrics_path)
            if algorithms_in_run:
                existing = existing[~existing["Algorithm"].isin(algorithms_in_run)]
            combined_df = pd.concat([existing, df], ignore_index=True)

        if combined_df.empty:
            if not all_metrics_path.exists():
                combined_df.to_csv(all_metrics_path, index=False)
            return combined_df

        combined_df.sort_values(by=["Metric_ID", "Test", "Algorithm"], inplace=True)
        combined_df.drop_duplicates(
            subset=["Test", "Algorithm", "Metric_ID"], keep="last", inplace=True
        )
        combined_df.reset_index(drop=True, inplace=True)
        combined_df.to_csv(all_metrics_path, index=False)

        key_subset = combined_df[combined_df["Metric_ID"].isin(KEY_METRICS)]
        if not key_subset.empty:
            pivot = (
                key_subset.pivot_table(
                    index=["Test", "Algorithm"],
                    columns="Metric_ID",
                    values="Observed_Value",
                    aggfunc="mean",
                )
                .sort_index()
            )
            pivot.reset_index().to_csv(self.tables_dir / "key_metrics_pivot.csv", index=False)
        elif (self.tables_dir / "key_metrics_pivot.csv").exists():
            (self.tables_dir / "key_metrics_pivot.csv").unlink()

        return combined_df

    def _plot_metric_panels(self, df: pd.DataFrame) -> List[Path]:
        if df.empty:
            return []

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        metric_map = {
            "M3": "Recall (M3)",
            "M5": "RMSE (M5)",
            "M7": "AUC (M7)",
            "M10": "Rounds (M10)",
            "M15": "Fairness (M15)",
            "M21": "Attack Detection (M21)",
        }

        rows = len(metric_map)
        fig, axes = plt.subplots(rows, 1, figsize=(10, 3.2 * rows), sharex=False)
        if rows == 1:
            axes = [axes]

        for ax, (metric_id, title) in zip(axes, metric_map.items()):
            subset = df[df["Metric_ID"] == metric_id]
            if subset.empty:
                ax.axis("off")
                ax.set_title(f"{title} (no data)")
                continue
            pivot = subset.pivot_table(
                index="Test",
                columns="Algorithm",
                values="Observed_Value",
                aggfunc="mean",
            )
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Test")
            ax.set_ylabel("Value")
            ax.grid(alpha=0.3, axis="y")

        fig.tight_layout()
        figure_path = self.figures_dir / "algorithm_metric_panels.png"
        fig.savefig(figure_path, dpi=300)
        plt.close(fig)
        return [figure_path]

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _normalize_algorithm_params(
        self,
        params: Optional[Mapping[str, Mapping[str, object]]],
    ) -> Dict[str, Mapping[str, object]]:
        if not params:
            return {}
        normalized: Dict[str, Mapping[str, object]] = {}
        for key, value in params.items():
            algorithm_name = self._normalize_algorithm_name(key)
            normalized[algorithm_name] = value
        return normalized

    def _normalize_algorithm_name(self, algorithm: str | FederatedAlgorithm) -> str:
        return FederatedAlgorithm.from_value(algorithm).value


def _parse_algorithm_param_overrides(entries: Optional[Sequence[str]]) -> Dict[str, Dict[str, object]]:
    overrides: Dict[str, Dict[str, object]] = {}
    if not entries:
        return overrides
    for entry in entries:
        if "=" not in entry or "." not in entry:
            raise ValueError(f"Invalid algorithm parameter override '{entry}'. Expected format algo.param=value")
        algo_field, raw_value = entry.split("=", 1)
        algo_name, param_key = algo_field.split(".", 1)
        parsed_value = _safe_literal(raw_value)
        overrides.setdefault(algo_name.strip().lower(), {})[param_key.strip()] = parsed_value
    return overrides


def _safe_literal(value: str) -> object:
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple federated algorithms across tests.")
    parser.add_argument(
        "--tests",
        type=str,
        default=",".join(DEFAULT_TESTS),
        help="Comma-separated test IDs to run (default: T2–T10,T12).",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=",".join(DEFAULT_ALGORITHMS),
        help="Comma-separated list of algorithms (fedavg,fedprox,scaffold).",
    )
    parser.add_argument(
        "--algorithm-param",
        dest="algorithm_params",
        action="append",
        help="Override algorithm hyperparameters (e.g., fedprox.proximal_mu=0.1). Can be repeated.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging.",
    )
    args = parser.parse_args()

    test_ids = [exp.strip().upper() for exp in args.tests.split(",") if exp.strip()]
    algorithms = [algo.strip().lower() for algo in args.algorithms.split(",") if algo.strip()]
    overrides = _parse_algorithm_param_overrides(args.algorithm_params)

    project_root = Path(__file__).resolve().parents[2]
    suite = AlgorithmComparisonSuite(project_root, verbose=not args.quiet)
    suite.run(test_ids, algorithms, algorithm_params=overrides, verbose=not args.quiet)


if __name__ == "__main__":  # pragma: no cover
    main()


