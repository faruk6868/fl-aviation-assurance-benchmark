# Results directory

- `testbeds/` (created at runtime): per-test-bed outputs from `src.benchmark.run`.
- `assurance_reports/` (created at runtime): consolidated pass/fail and summaries per test bed and algorithm.
- `all_results.csv`: aggregated pass/fail table across TB/algorithm (written by `scripts/generate_paper_assets.py`).
- `time_benchmarks/`: runtime plots/CSVs used in the paper (regenerated via scripts in `analysis/` or copied into `paper_assets/figures/`).

Schema highlights:
- Pass/fail CSVs: columns include `Metric_ID`, `Result`, `Value`, `Threshold`, `Requirement_ID`.
- Aggregated `all_results.csv`: `tb_id`, `algorithm`, `metric_id`, `result`, `value`, `threshold`.
