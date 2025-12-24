"""Assurance pipeline orchestrating test-bed measurements and reporting."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set

import pandas as pd
from scipy.stats import norm

from src.evaluation.types import TestOutput
from src.testbeds.measurement import load_measurements
from src.utils import AssuranceConfigV2, TestBedDefinition

from .assessment import AssuranceEvaluator, FHA_METRICS, LEVEL_SEQUENCE, MetricAssessmentRecord
from .reporting import (
    compute_summary,
    write_consolidated_report,
    write_test_report,
    write_summary_report,
)


class AssurancePipeline:
    """Evaluates measurement artifacts against configuration thresholds."""

    def __init__(self, project_root: Path, verbose: bool = True) -> None:
        self.project_root = Path(project_root)
        self.verbose = verbose
        self.config_root = self.project_root / "config"
        self.assurance_config = AssuranceConfigV2(self.config_root)

        self.evaluator = AssuranceEvaluator(
            metadata=self.assurance_config.metric_metadata,
            thresholds=self.assurance_config.threshold_tables,
            requirements=self.assurance_config.requirement_mapping,
        )

        self.test_dir = self.project_root / "results" / "testbeds"
        self.assurance_dir = self.project_root / "results" / "assurance_reports"
        self.expected_metrics = self._build_expected_metric_map()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        tb_id: str,
        algorithm: str,
        measurement_path: Optional[Path] = None,
    ) -> Dict[str, object]:
        tb_id_norm = tb_id.strip().upper()
        if tb_id_norm not in self.assurance_config.test_beds:
            known = ", ".join(sorted(self.assurance_config.test_beds.keys()))
            raise KeyError(f"Unknown test bed '{tb_id}'. Known IDs: {known}")
        testbed = self.assurance_config.test_beds[tb_id_norm]

        resolved_measurement_path = self._resolve_measurement_path(tb_id_norm, algorithm, measurement_path)
        test_output = load_measurements(
            tb_id_norm,
            algorithm,
            self.assurance_config.metric_metadata,
            resolved_measurement_path,
        )

        self._validate_metrics(testbed, test_output)
        records = self.evaluator.evaluate_test(test_output)

        output_dir = self.test_dir / tb_id_norm / algorithm
        output_dir.mkdir(parents=True, exist_ok=True)
        write_test_report(tb_id_norm, records, output_dir)

        consolidated_path = self.assurance_dir / f"{tb_id_norm}_{algorithm}_pass_fail.csv"
        write_consolidated_report(records, consolidated_path)

        summary = compute_summary(records)
        write_summary_report(records, self.assurance_dir / f"{tb_id_norm}_{algorithm}_summary.md", summary)

        self._generate_threshold_sensitivity(self.assurance_dir / "threshold_sensitivity_analysis.csv")

        return {"records": records, "summary": summary}

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _build_expected_metric_map(self) -> Dict[str, Set[str]]:
        mapping: Dict[str, Set[str]] = {}
        for tb_id, definition in self.assurance_config.test_beds.items():
            mapping[tb_id] = set(definition.metrics)
        return mapping

    def _validate_metrics(self, testbed: TestBedDefinition, output: TestOutput) -> None:
        expected = self.expected_metrics.get(testbed.test_id)
        if expected is None:
            return
        actual = {result.metric_id for result in output.metrics}
        missing = sorted(expected - actual)
        if missing:
            message = (
                f"[{testbed.test_id}] measurement missing metrics: {missing}"
            )
            if self.verbose:
                print(f"WARNING: {message}")

    # ------------------------------------------------------------------
    # Threshold sensitivity analysis
    # ------------------------------------------------------------------
    def _generate_threshold_sensitivity(self, output_path: Path) -> Optional[Path]:
        records: List[Dict[str, object]] = []
        for metric_id in FHA_METRICS:
            for level_name, label in LEVEL_SEQUENCE:
                table = self.assurance_config.threshold_tables.get(level_name)
                if table is None or metric_id not in table.index:
                    continue
                row = table.loc[metric_id]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                assumptions = row.get("Assumptions")
                if not isinstance(assumptions, str) or not assumptions.strip():
                    continue

                params = self._parse_assumptions(assumptions)
                try:
                    baseline = self._compute_fha_threshold(metric_id, params)
                except (KeyError, ValueError):
                    continue

                actual_threshold = self._coerce_float(row.get("Threshold_Value"))
                adjustments = self._compute_adjustments(metric_id, params)

                record = {
                    "Metric_ID": metric_id,
                    "Metric_Name": row.get("Metric_Name", self.assurance_config.metric_metadata[metric_id].name),
                    "Level": label,
                    "Actual_Threshold": actual_threshold,
                    "Recomputed_Threshold": baseline,
                    "Assumptions": assumptions,
                }
                record.update(adjustments)
                records.append(record)

        if not records:
            return None

        df = pd.DataFrame(records)
        df.sort_values(by=["Metric_ID", "Level"], inplace=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path

    def _compute_adjustments(self, metric_id: str, params: Dict[str, float]) -> Dict[str, float | None]:
        adjustments: Dict[str, float | None] = {
            "proc_fail_x0.1": None,
            "proc_fail_x10": None,
            "cycles_per_fh_x0.1": None,
            "cycles_per_fh_x10": None,
        }

        proc_fail = params.get("proc_fail")
        cycles = params.get("cycles_per_fh")

        if proc_fail and proc_fail > 0:
            adjustments["proc_fail_x0.1"] = self._compute_fha_threshold(
                metric_id, {**params, "proc_fail": proc_fail * 0.1}
            )
            adjustments["proc_fail_x10"] = self._compute_fha_threshold(
                metric_id, {**params, "proc_fail": proc_fail * 10.0}
            )

        if cycles and cycles > 0:
            adjustments["cycles_per_fh_x0.1"] = self._compute_fha_threshold(
                metric_id, {**params, "cycles_per_fh": cycles * 0.1}
            )
            adjustments["cycles_per_fh_x10"] = self._compute_fha_threshold(
                metric_id, {**params, "cycles_per_fh": cycles * 10.0}
            )

        return adjustments

    def _compute_fha_threshold(self, metric_id: str, params: Mapping[str, float]) -> float:
        p_alloc_fh = params.get("hazard_rate_fh")
        proc_fail = params.get("proc_fail")
        cycles_per_fh = params.get("cycles_per_fh")
        lead_time = params.get("lead_time_cycles")

        if p_alloc_fh is None or proc_fail is None or cycles_per_fh is None:
            raise KeyError("Missing FHA parameters")

        if proc_fail <= 0 or cycles_per_fh <= 0:
            raise ValueError("FHA parameters must be positive")

        if metric_id in {"M.SAF.REC", "M.SAF.ACC"}:
            return 1.0 - (p_alloc_fh / proc_fail) * cycles_per_fh
        if metric_id == "M.SAF.FPFN":
            return (p_alloc_fh / proc_fail) * cycles_per_fh
        if metric_id == "M.SAF.RMSE":
            if lead_time is None:
                raise KeyError("lead_time_cycles required for RMSE threshold")
            p_cycle = p_alloc_fh * cycles_per_fh
            alpha = max(min(p_cycle / 2.0, 0.499999999), 1e-15)
            quantile = norm.ppf(1 - alpha)
            if quantile == 0.0:
                raise ValueError("Quantile computation returned zero")
            return lead_time / quantile

        raise ValueError(f"Unsupported FHA metric {metric_id}")

    def _parse_assumptions(self, assumptions: str) -> Dict[str, float]:
        params: Dict[str, float] = {}
        for chunk in assumptions.split(";"):
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            key = key.strip()
            value = value.strip().split(",")[0]
            try:
                params[key] = float(value)
            except ValueError:
                match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
                if match:
                    params[key] = float(match.group(0))
        return params

    @staticmethod
    def _coerce_float(value: object) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _resolve_measurement_path(
        self,
        tb_id: str,
        algorithm: str,
        provided_path: Optional[Path],
    ) -> Path:
        if provided_path is not None:
            return provided_path
        return self.project_root / "artifacts" / "testbeds" / tb_id / f"{algorithm}.json"


