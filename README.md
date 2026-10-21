# PI-BDP × N-CMAPSS — pre-registered confirmatory evaluation (freeze v1.0)

This repository fixes the complete design **before** the N-CMAPSS test split is evaluated.

- `PROTOCOL_v1.0.md` — frozen protocol (data, model configuration, calibration, comparison models, metrics, statistics, known limitations)
- `pibdp_ncm.py` — PI-BDP library used in all phases (its SHA-256 is recorded in `freeze_v1/MANIFEST.json`)
- `NCMAPSS_Freeze_고정패키지.ipynb` — builds `freeze_v1/` (settings, phase summaries, hashes) and verifies that the sealed test files are unchanged
- `NCMAPSS_Phase3_*.ipynb` — test evaluation (run once, after this commit). They refuse to run unless the frozen hashes match.
- `freeze_v1/` — configuration files selected on the development split only, stage summaries, and `MANIFEST.json`

Any change after this commit is reported as a post-hoc change.
