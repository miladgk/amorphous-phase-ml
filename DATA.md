# Data Files: Regeneration Guide

## Reference Structure CSVs

The four `*_csro_fixed.csv` files are **not committed to git** due to their size (~173MB total). They are regenerated from the raw LAMMPS snapshots in `files/snapshots_*/` using the post-processing pipeline.

### Files to regenerate

| File | Approx. Size | Source |
|------|-------------|--------|
| `Atomic_Simulation_Post-Processing_Pipeline/outputs/features_5050_csro_fixed.csv` | 28 MB | 11 snapshots in `files/snapshots_5050/` |
| `Atomic_Simulation_Post-Processing_Pipeline/outputs/features_4654_csro_fixed.csv` | 28 MB | 11 snapshots in `files/snapshots_4654/` |
| `Atomic_Simulation_Post-Processing_Pipeline/outputs/features_6436_csro_fixed.csv` | 32 MB | 11 snapshots in `files/snapshots_6436/` |
| `Atomic_Simulation_Post-Processing_Pipeline/outputs/features_polyamorphous_csro_fixed.csv` | 85 MB | 9 frames in `Machine_Learning_Pipeline_for_Materials_Science/data/polyamorphous/` |

### Regeneration commands

Each reference structure requires a separate pipeline run with the correct composition:

```bash
# Cu50Zr50 (type_1=0.50, type_2=0.50)
sed -i 's|SNAPSHOT_DIRECTORY: .*|SNAPSHOT_DIRECTORY: "files/snapshots_5050/"|;
        s|output_csv: .*|output_csv: "outputs/features_5050_csro_fixed.csv"|;
        s|type_1_fraction: .*|type_1_fraction: 0.50|;
        s|type_2_fraction: .*|type_2_fraction: 0.50|' Atomic_Simulation_Post-Processing_Pipeline/config.yaml
conda run -n materials-sim-ml python3 Atomic_Simulation_Post-Processing_Pipeline/src/pipeline_orchestrator.py \
    --config Atomic_Simulation_Post-Processing_Pipeline/config.yaml

# Cu46Zr54 (type_1=0.46, type_2=0.54) — same pattern with files/snapshots_4654/
# Cu64Zr36 (type_1=0.64, type_2=0.36) — same pattern with files/snapshots_6436/
```

Each run takes approximately **30 minutes** per structure (feature extraction across 11 snapshots + temporal averaging + RDF peak fitting + CSRO computation).

### Polyamorphous target

The polyamorphous trajectory consists of 9 frames. Each frame requires an independent pipeline run:

```bash
# For each frame in data/polyamorphous/:
conda run -n materials-sim-ml python3 Machine_Learning_Pipeline_for_Materials_Science/src/process_trajectory.py --lite
```

The `--lite` flag sets MRO_DEPTH=1 (features beyond the 18-feature whitelist are skipped). Each frame takes ~3 minutes in lite mode vs ~30 minutes in full mode.

## Per-snapshot feature files

The 33 per-snapshot CSV files (`outputs/per_snapshot/features_*.csv`) are intermediate artifacts, not stored in git. They are produced alongside the temporally-averaged CSVs during each pipeline run.

## Model and normalization parameters

The following are **committed to git** as they are the current validated pipeline artifacts (small, <4MB):
- `general_phase_id_output/model.pkl` — trained HistGB classifier
- `general_phase_id_output/norm_params.pkl` — frozen normalization parameters
- `general_phase_id_output/feature_ranges.pkl` — OOD detection ranges
- `general_phase_id_output/predicted_phases.lammpstrj` — final LAMMPS output

## Raw LAMMPS snapshots

The raw MD trajectory dumps in `files/snapshots_*/` and `data/polyamorphous/` are **committed to git** as they are the original, irreplaceable simulation output. No LAMMPS input scripts exist in the repository — these cannot be reproduced without re-running the full MD simulation from scratch.