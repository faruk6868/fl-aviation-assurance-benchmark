# Assurance framework (plain language)

- Clauses: Derived from aviation safety guidance (EASA/FAA), SAE ARP4754A/4761, SOTIF, and NIST AI RMF.
- Requirements: Each clause maps to one or more measurable requirements (see `config/config_v2/B.Requirements.csv`).
- Metrics: Each requirement is evidenced by one or more metrics (see `config/config_v2/C.*`).
- Thresholds: Each metric has target values per assurance level (see `config/config_v2/D.Thresholds.csv`).
- Test beds: TB-01 to TB-14 instantiate scenarios to collect metric evidence.
- Evidence: Measurement JSONs per TB/algorithm live in `artifacts/testbeds/` and are parsed by the pipeline.
- Reporting: Pass/fail and summaries are written to `results/assurance_reports/`; consolidated in `results/all_results.csv`.
