# Reproducibility guide

## Environment
- OS: Linux/macOS/Windows (tested on Python 3.10).
- Python: 3.10 recommended; install via `python -m venv .venv && .venv\Scripts\activate` (or `source .venv/bin/activate`).
- Dependencies: `pip install -r requirements.txt` (or `environment/environment.yml` with Conda).

## Expected runtimes (using provided measurement JSONs, no model training)
- TB-01 to TB-05: ~5-15s each on laptop CPU (uses provided measurements).
- TB-06 to TB-10: ~10-20s each.
- TB-11 to TB-14: ~10-25s each.
- End-to-end (TB-01 to TB-14, fedavg): typically under a few minutes on laptop CPU.
- Heatmap/time-benchmark figure generation: seconds once results exist.

## Determinism
- Seeds set via `configs/default.yaml` and `src/benchmark/run.py` (Python, NumPy, optional PyTorch).
- Config snapshots are written under `results/run_configs/` per run.

## Troubleshooting
- Missing data: run `python scripts/download_cmapss.py --source-url <url> --target-dir data/c-mapss`.
- GPU/CPU: Pipeline is CPU-friendly; GPU not required for assurance checks.
- Memory: Results aggregation operates on CSVs; if memory-constrained, set `analysis.copy_figures: false` and prune large figure files.
- Permissions: Ensure write access to `results/` and `paper_assets/`.
