# Recomputed feature tables (not tracked in git: >100 MB total)

Files: features_5050_recomputed.csv, features_4654_recomputed.csv, features_6436_recomputed.csv
(references, 11 snapshots each) and features_polyamorphous_recomputed.csv (target, 9 snapshots,
1e6-step index spacing, dt = 2 fs).

They are produced by `pipeline_orchestrator.py` with `ANALYSIS_LEVEL: temporal_first_spatial`,
using the same config.yaml as the repo, with SNAPSHOT_DIRECTORY pointing to each snapshot folder.

Requirements:
- `spatial_analysis_levels.py` was removed from src/ in commit eb06523. Restore it from history:
  `git show eb06523^:Atomic_Simulation_Post-Processing_Pipeline/src/spatial_analysis_levels.py`
- freud-analysis==3.2, pyvoro==1.3.2 (see environment.yml).
- Target run needs about 35 min with one worker; 36,828 atoms per frame use a lot of RAM
  (8 workers ran out of memory on a 7 GB machine).
- WORKER_TIMEOUT must be raised for the 36,828-atom target (e.g. 20000 s).

Validation: with these settings the reference tables reproduce voronoi_volume, CN, q4/q6,
stress, energy and pentagon/asphericity values exactly. R1_SG and the Gaussian peak centres
differ from the earlier tables (correlation ~0.85–0.96), and pe_volume_residual is not produced.
