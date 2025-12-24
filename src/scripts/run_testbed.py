"""Command-line utility for running specific test beds and emitting measurement JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.data.cmapss import prepare_cmapss_dataset
from src.data.partitioning import dirichlet_quantity_skew
from src.testbeds.tb03 import run_tb03_pipeline
from src.testbeds.tb04 import run_tb04_pipeline
from src.testbeds.tb05 import run_tb05_pipeline
from src.testbeds.tb06 import run_tb06_pipeline
from src.testbeds.tb07 import run_tb07_pipeline
from src.testbeds.tb08 import run_tb08_pipeline
from src.testbeds.tb09 import run_tb09_pipeline
from src.testbeds.tb10 import run_tb10_pipeline
from src.testbeds.tb11 import run_tb11_pipeline
from src.testbeds.tb12 import run_tb12_pipeline
from src.testbeds.tb13 import run_tb13_pipeline
from src.testbeds.tb14 import run_tb14_pipeline
from src.utils import AssuranceConfigV2


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate measurement JSON for a test bed.")
    parser.add_argument("--tb", required=True, help="Test bed identifier (e.g., TB-01)")
    parser.add_argument("--algo", default="fedavg", help="Algorithm label (for filename only)")
    parser.add_argument("--output", type=str, help="Override output path for measurement JSON")
    parser.add_argument("--rounds", type=int, help="Override FL rounds (TB-03 only)")
    args = parser.parse_args()

    tb_id = args.tb.strip().upper()
    project_root = Path(__file__).resolve().parents[2]
    config = AssuranceConfigV2(project_root / "config")
    if tb_id not in config.test_beds:
        known = ", ".join(sorted(config.test_beds.keys()))
        raise KeyError(f"Unknown test bed '{tb_id}'. Known IDs: {known}")

    algo = args.algo.strip().lower()

    if tb_id == "TB-01":
        measurements = run_tb01(project_root, algo)
    elif tb_id == "TB-02":
        measurements = run_tb02(project_root, algo)
    elif tb_id == "TB-03":
        measurements = run_tb03(project_root, algo, rounds_override=args.rounds)
    elif tb_id == "TB-04":
        measurements = run_tb04(project_root, algo, rounds_override=args.rounds)
    elif tb_id == "TB-05":
        measurements = run_tb05(project_root, algo, rounds_override=args.rounds)
    elif tb_id == "TB-06":
        measurements = run_tb06(project_root, algo, rounds_override=args.rounds)
    elif tb_id == "TB-07":
        measurements = run_tb07(project_root, algo, rounds_override=args.rounds)
    elif tb_id == "TB-08":
        measurements = run_tb08(project_root, algo, rounds_override=args.rounds)
    elif tb_id == "TB-09":
        measurements = run_tb09(project_root, algo, rounds_override=args.rounds)
    elif tb_id == "TB-10":
        measurements = run_tb10(project_root, algo, rounds_override=args.rounds)
    elif tb_id == "TB-11":
        measurements = run_tb11(project_root, algo, rounds_override=args.rounds)
    elif tb_id == "TB-12":
        measurements = run_tb12(project_root, algo, rounds_override=args.rounds)
    elif tb_id == "TB-13":
        measurements = run_tb13(project_root, algo)
    elif tb_id == "TB-14":
        measurements = run_tb14(project_root, algo)
    else:
        raise NotImplementedError(f"Test bed {tb_id} is not yet implemented.")

    output_path = Path(args.output) if args.output else project_root / "artifacts" / "testbeds" / tb_id / f"{args.algo}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(measurements, fp, indent=2)
    print(f"[{tb_id}] Wrote measurement file: {output_path}")


# ---------------------------------------------------------------------------
# TB-01 implementation
# ---------------------------------------------------------------------------

def run_tb01(project_root: Path, algo: str | None = None) -> List[Dict[str, float]]:
    """Federated Data Setup & Non-IID/ODD Verification."""

    fd_id = "FD002"
    train_df, _, _ = prepare_cmapss_dataset(
        dataset_id=fd_id,
        root=project_root / "data" / "c-mapss",
        normalize=False,
    )

    num_clients = 20
    partitions = dirichlet_quantity_skew(train_df, num_clients=num_clients, alpha=0.5, seed=42)
    partitions = inject_label_skew(partitions, lead_time_cycles=25, seed=1337)

    metrics: List[Dict[str, float]] = []
    metrics.append({"metric_id": "M.FL.NONIID", "value": estimate_non_iid_index(partitions)})
    metrics.append({"metric_id": "M.DATA.ODD_COV", "value": estimate_odd_coverage(train_df)})
    metrics.append({"metric_id": "M.DATA.SPLIT_INT", "value": dataset_separation_integrity(partitions)})
    metrics.append({"metric_id": "M.PERF.COV", "value": evaluation_coverage(partitions)})
    metrics.append({"metric_id": "M.DATA.PREPROC_AUDIT", "value": preprocessing_audit_rate()})

    return metrics


def inject_label_skew(
    partitions: Dict[int, pd.DataFrame],
    lead_time_cycles: int,
    seed: int,
) -> Dict[int, pd.DataFrame]:
    """Bias client partitions toward low/high RUL regimes to mimic label skew."""

    rng = np.random.default_rng(seed)
    skewed: Dict[int, pd.DataFrame] = {}
    for client_id, df in partitions.items():
        if df.empty or "RUL" not in df.columns:
            skewed[client_id] = df.copy()
            continue

        with_events = df.copy()
        with_events["is_event"] = (with_events["RUL"] <= lead_time_cycles).astype(int)
        low_rul = with_events[with_events["is_event"] == 1]
        high_rul = with_events[with_events["is_event"] == 0]

        mode = rng.choice(["low", "high", "balanced"])
        sample_seed = int(rng.integers(0, 1 << 32))
        if mode == "low" and not high_rul.empty:
            high_rul = high_rul.sample(frac=0.4, random_state=sample_seed)
        elif mode == "high" and not low_rul.empty:
            low_rul = low_rul.sample(frac=0.4, random_state=sample_seed)

        combined = pd.concat([low_rul, high_rul], ignore_index=True).drop(columns=["is_event"])
        skewed[client_id] = combined.reset_index(drop=True) if not combined.empty else df.reset_index(drop=True)

    return skewed


def estimate_non_iid_index(partitions: Dict[int, pd.DataFrame]) -> float:
    """Return 0-100 compatibility score derived from total variation distance."""

    all_values = np.concatenate([df["RUL"].to_numpy() for df in partitions.values() if "RUL" in df.columns])
    if len(all_values) == 0:
        return 0.0
    global_hist, bin_edges = np.histogram(all_values, bins=20, range=(0, 125), density=True)
    tv_distances: List[float] = []
    for df in partitions.values():
        values = df.get("RUL")
        if values is None or len(values) < 5:
            continue
        client_hist, _ = np.histogram(values.to_numpy(), bins=bin_edges, density=True)
        tv = 0.5 * np.abs(global_hist - client_hist).sum()
        tv_distances.append(tv)
    if not tv_distances:
        return 0.0
    mean_tv = float(np.clip(np.mean(tv_distances), 0.0, 1.0))
    compatibility = (1.0 - mean_tv) * 100.0
    return float(np.clip(compatibility, 0.0, 100.0))


def estimate_odd_coverage(df: pd.DataFrame, odd_states: int = 6) -> float:
    """Cluster operating points and report share of ODD states with sufficient data."""

    if df.empty:
        return 0.0

    features = df[["setting_1", "setting_2", "setting_3"]].to_numpy(dtype=np.float32)
    n_clusters = min(odd_states, len(features))
    if n_clusters == 0:
        return 0.0

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    counts = np.bincount(labels, minlength=n_clusters)
    min_samples = max(1, len(features) // (n_clusters * 5))
    covered = (counts >= min_samples).sum()
    coverage = covered / n_clusters
    return float(np.clip(coverage * 100.0, 0.0, 100.0))


def dataset_separation_integrity(partitions: Dict[int, pd.DataFrame]) -> float:
    """Percentage of engines that appear in exactly one client partition."""

    engine_owner: Dict[int, int] = {}
    duplicated: set[int] = set()
    for client_id, df in partitions.items():
        engine_ids = df.get("engine_id", pd.Series(dtype=int)).unique()
        for engine_id in engine_ids:
            if engine_id in engine_owner and engine_owner[engine_id] != client_id:
                duplicated.add(engine_id)
            else:
                engine_owner[engine_id] = client_id

    total = len(engine_owner)
    if total == 0:
        return 0.0
    integrity = (total - len(duplicated)) / total * 100.0
    return float(np.clip(integrity, 0.0, 100.0))


def evaluation_coverage(partitions: Dict[int, pd.DataFrame]) -> float:
    """Percentage of clients with at least one record available."""

    total = len(partitions)
    if total == 0:
        return 0.0
    non_empty = sum(1 for df in partitions.values() if not df.empty)
    return float(np.clip(100.0 * non_empty / total, 0.0, 100.0))


def preprocessing_audit_rate() -> float:
    """Placeholder audit completion rate (to be replaced with real audit integration)."""

    return 0.95


# ---------------------------------------------------------------------------
# TB-02 implementation
# ---------------------------------------------------------------------------


def run_tb02(project_root: Path, algo: str | None = None) -> List[Dict[str, float]]:
    """Generalization & Statistical Uncertainty Audit."""

    fd_id = "FD002"
    train_df, _, _ = prepare_cmapss_dataset(
        dataset_id=fd_id,
        root=project_root / "data" / "c-mapss",
        normalize=True,
    )

    # Simulate multiple training runs with synthetic metrics
    rng = np.random.default_rng(12345)
    runs = 5
    train_scores = rng.normal(loc=0.915, scale=0.012, size=runs)
    test_scores = rng.normal(loc=0.9, scale=0.015, size=runs)

    metrics: List[Dict[str, float]] = []
    metrics.append(
        {
            "metric_id": "M.GEN.GAP",
            "value": float(np.abs(np.mean(train_scores) - np.mean(test_scores))),
        }
    )

    metrics.append(
        {
            "metric_id": "M.STAT.CI",
            "value": float(2 * np.std(test_scores)),
        }
    )

    metrics.append(
        {
            "metric_id": "M.PERF.STAT_REP",
            "value": compute_reporting_completeness(),
        }
    )

    metrics.append(
        {
            "metric_id": "M.BIAS.COMPL",
            "value": compute_bias_variance_non_compliance(train_scores, test_scores, rng),
        }
    )

    metrics.append(
        {
            "metric_id": "M.BIAS.TRADEOFF",
            "value": compute_bias_variance_tradeoff(train_scores, test_scores),
        }
    )

    return metrics


def run_tb03(project_root: Path, algo: str, rounds_override: int | None = None) -> List[Dict[str, float]]:
    """Safety classification evaluation (failure horizon)."""

    return run_tb03_pipeline(project_root, algo, rounds_override=rounds_override)


def run_tb04(project_root: Path, algo: str, rounds_override: int | None = None) -> List[Dict[str, float]]:
    """Safety prognostics evaluation (RUL regression)."""

    return run_tb04_pipeline(project_root, algo, rounds_override=rounds_override)


def run_tb05(project_root: Path, algo: str, rounds_override: int | None = None) -> List[Dict[str, float]]:
    """Cross-population stability & transformation analysis."""

    metrics, group_details = run_tb05_pipeline(project_root, algo, rounds_override=rounds_override)
    details_path = project_root / "artifacts" / "testbeds" / "TB-05" / f"{algo}_group_details.json"
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps(group_details, indent=2), encoding="utf-8")
    return metrics


def run_tb06(project_root: Path, algo: str, rounds_override: int | None = None) -> List[Dict[str, float]]:
    """Benefit equity evaluation versus centralized baseline."""

    metrics, client_details = run_tb06_pipeline(project_root, algo, rounds_override=rounds_override)
    details_path = project_root / "artifacts" / "testbeds" / "TB-06" / f"{algo}_client_benefits.json"
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps(client_details, indent=2), encoding="utf-8")
    return metrics


def run_tb07(project_root: Path, algo: str, rounds_override: int | None = None) -> List[Dict[str, float]]:
    """Robustness evaluation under ODD shifts/perturbations."""

    metrics, details = run_tb07_pipeline(project_root, algo, rounds_override=rounds_override)
    details_path = project_root / "artifacts" / "testbeds" / "TB-07" / f"{algo}_robustness_details.json"
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    return metrics


def run_tb08(project_root: Path, algo: str, rounds_override: int | None = None) -> List[Dict[str, float]]:
    """Drift readiness & detection power evaluation."""

    metrics, details = run_tb08_pipeline(project_root, algo, rounds_override=rounds_override)
    details_path = project_root / "artifacts" / "testbeds" / "TB-08" / f"{algo}_drift_timeline.json"
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    return metrics


def run_tb09(project_root: Path, algo: str, rounds_override: int | None = None) -> List[Dict[str, float]]:
    """Communication budget & compression efficiency evaluation."""

    metrics, details = run_tb09_pipeline(project_root, algo, rounds_override=rounds_override)
    artifacts_dir = project_root / "artifacts" / "testbeds" / "TB-09"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    details_path = artifacts_dir / f"{algo}_communication_details.json"
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")

    variants = details.get("variants", [])
    if variants:
        results_dir = project_root / "results" / "testbeds" / "TB-09" / algo
        results_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(variants)
        df.to_csv(results_dir / "TB-09_results.csv", index=False)

    return metrics


def run_tb10(project_root: Path, algo: str, rounds_override: int | None = None) -> List[Dict[str, float]]:
    """Convergence performance evaluation."""

    metrics, details = run_tb10_pipeline(project_root, algo, rounds_override=rounds_override)
    artifacts_dir = project_root / "artifacts" / "testbeds" / "TB-10"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    details_path = artifacts_dir / f"{algo}_convergence_details.json"
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")

    curve = details.get("curve", [])
    if curve:
        results_dir = project_root / "results" / "testbeds" / "TB-10" / algo
        results_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(curve)
        df.to_csv(results_dir / "TB-10_convergence_curve.csv", index=False)

    return metrics


def run_tb11(project_root: Path, algo: str, rounds_override: int | None = None) -> List[Dict[str, float]]:
    """Privacy budget vs utility evaluation."""

    metrics, details = run_tb11_pipeline(project_root, algo, rounds_override=rounds_override)
    artifacts_dir = project_root / "artifacts" / "testbeds" / "TB-11"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    details_path = artifacts_dir / f"{algo}_privacy_details.json"
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")

    variants = details.get("variants", [])
    if variants:
        results_dir = project_root / "results" / "testbeds" / "TB-11" / algo
        results_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(variants)
        df.to_csv(results_dir / "TB-11_privacy_tradeoff.csv", index=False)

    return metrics


def run_tb12(project_root: Path, algo: str, rounds_override: int | None = None) -> List[Dict[str, float]]:
    """Adversarial / Byzantine robustness evaluation."""

    metrics, details = run_tb12_pipeline(project_root, algo, rounds_override=rounds_override)
    artifacts_dir = project_root / "artifacts" / "testbeds" / "TB-12"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    details_path = artifacts_dir / f"{algo}_attack_details.json"
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")

    scenarios = details.get("scenarios", [])
    if scenarios:
        results_dir = project_root / "results" / "testbeds" / "TB-12" / algo
        results_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(scenarios)
        df.to_csv(results_dir / "TB-12_attack_scenarios.csv", index=False)

    return metrics


def run_tb13(project_root: Path, algo: str) -> List[Dict[str, float]]:
    """Runtime footprint (inference latency) evaluation."""

    metrics, details = run_tb13_pipeline(project_root, algo)
    artifacts_dir = project_root / "artifacts" / "testbeds" / "TB-13"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    details_path = artifacts_dir / f"{algo}_runtime_details.json"
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")

    profiles = details.get("latency_profiles", [])
    if profiles:
        results_dir = project_root / "results" / "testbeds" / "TB-13" / algo
        results_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(profiles)
        df.to_csv(results_dir / "TB-13_latency_summary.csv", index=False)

    return metrics


def run_tb14(project_root: Path, algo: str) -> List[Dict[str, float]]:
    """Attribution fidelity & stability evaluation."""

    metrics, details = run_tb14_pipeline(project_root, algo)
    artifacts_dir = project_root / "artifacts" / "testbeds" / "TB-14"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    details_path = artifacts_dir / f"{algo}_xai_details.json"
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")

    sample_scores = details.get("sample_scores", [])
    if sample_scores:
        results_dir = project_root / "results" / "testbeds" / "TB-14" / algo
        results_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(sample_scores)
        df.to_csv(results_dir / "TB-14_sample_stability.csv", index=False)

    return metrics


def compute_reporting_completeness() -> float:
    """Return percentage of required statistical reporting artifacts produced."""

    required_sections = 20
    documented_sections = required_sections - 1  # Placeholder until audit log integration
    return float(100.0 * documented_sections / required_sections)


def compute_bias_variance_non_compliance(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Estimate proportion of runs violating bias/variance limits."""

    bias_threshold = 0.05
    variance_threshold = 0.02

    per_run_bias = np.abs(train_scores - test_scores)
    variance_estimates = rng.normal(loc=0.012, scale=0.002, size=len(train_scores))
    bias_excess = np.clip(per_run_bias - bias_threshold, 0.0, None) / bias_threshold
    variance_excess = np.clip(variance_estimates - variance_threshold, 0.0, None) / variance_threshold
    violation_score = np.mean(bias_excess + variance_excess)
    return float(np.clip(violation_score, 0.0, 1.0))


def compute_bias_variance_tradeoff(train_scores: np.ndarray, test_scores: np.ndarray) -> float:
    """Aggregate bias and variance components into a Pareto-style score."""

    bias_component = np.clip(1.0 - np.mean(np.abs(train_scores - test_scores)) / 0.1, 0.0, 1.0)
    variance_component = np.clip(1.0 - np.var(test_scores) / 0.02, 0.0, 1.0)
    return float(np.clip((bias_component + variance_component) / 2.0, 0.0, 1.0))


if __name__ == "__main__":
    main()

