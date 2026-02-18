#!/usr/bin/env bash
set -euo pipefail

# Run all test beds and aggregate results
python -m src.benchmark.run --config configs/default.yaml
python scripts/generate_paper_assets.py --results-dir results --paper-assets-dir paper_assets
