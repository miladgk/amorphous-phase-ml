# Machine Learning Pipeline for Materials Science (Cu–Zr polyamorphous phases)

Classifies per-atom structural phases in Cu–Zr metallic glasses from descriptors computed by
`Atomic_Simulation_Post-Processing_Pipeline`. The model is a scikit-learn `HistGradientBoostingClassifier`
trained on three homogeneous reference glasses (Cu50Zr50, Cu46Zr54, Cu64Zr36) and applied to a
polyamorphous target box.

## Current result (17-feature model)

Spatial holdout (atoms with |x − median| < 3.5 Å removed; 16,227 test atoms):

| Metric | Value |
|---|---|
| CV macro F1 | 0.670 ± 0.005 |
| Holdout accuracy | 67.8 % |
| F1 Cu50Zr50 / Cu46Zr54 / Cu64Zr36 | 0.504 / 0.590 / 0.900 |

These are the only numbers this pipeline currently supports. Older figures (0.854 CV, 74.6 % holdout) came
from earlier feature tables and are no longer reproducible.

## Inputs

- Reference tables: `Atomic_Simulation_Post-Processing_Pipeline/outputs/features_{5050,4654,6436}_recomputed.csv`
  (11 snapshots each, averaged per atom).
- Target table: `Atomic_Simulation_Post-Processing_Pipeline/outputs/features_polyamorphous_recomputed.csv`
  (9 snapshots, averaged per atom; dt = 2 fs, so frames are about 2 ns apart).
- Target box for the periodic consensus filter: `data/polyamorphous/anneal_combined_0.lammpstrj`.
- These tables are not tracked (size). See `Atomic_Simulation_Post-Processing_Pipeline/outputs/README_recomputed.md` for how to regenerate them.

## Run

```bash
python src/general_phase_id/train_and_predict.py --config src/general_phase_id/ref_config.yaml
```

Outputs go to `output_dir` in the config (default `general_phase_id_output/`): `model.pkl`, `predictions.csv`,
`predicted_phases.lammpstrj`, `predicted_phases.xyz`, plots and `config_used.yaml`.

Requirements: numpy, pandas, scipy, scikit-learn (1.2.x tested), joblib, pyyaml, freud-analysis 3.2 (imported),
pyvoro 1.3.2 (see root `environment.yml`).

## Features (17)

Configured in `src/general_phase_id/ref_config.yaml` (`feature_whitelist`):

- Structural: `R1_SG`, `voronoi_volume_temporal`, `CN_temporal`, `q4`, `q6`, `gaussian_peak2_center`, `gaussian_peak3_center`, `neighbor_1`, `neighbor_2`
- Energy and stress: `excess_energy_temporal`, `local_energy_fluctuation_temporal`, `hydrostatic_pressure_temporal`, `von_mises_stress_temporal`, `deviatoric_norm_temporal`, `stress_anisotropy_temporal`, `local_enthalpy_temporal`, `pe_atom_std_temporal`

`pe_volume_residual` appears in older lists but the current pipeline does not produce it.

Excluded on purpose:
- MRO depth-2/3 features (`mean_*_neighbor_*`, `cu_fraction_*`): they leaked simulation-box identity (see `PIPELINE.md` §8.2 and `docs/PROJECT_BRIEF.md`).
- Chemical SRO and neighbour ratios: not validated as non-leaky in the current setup.

## Known limitations

- Each reference phase is one simulation box. The spatial holdout cannot separate phase structure from box-specific signatures (`PIPELINE.md` §9.9).
- Cu50Zr50 and Cu46Zr54 differ by about 4 at.% Cu inside a ~40 Å gradient; F1 for these two classes is low by construction.
- `ref_config.yaml` previously referenced an MRO override and extra features; those have been removed.
- Experiments that did not enter the model: species-blind SOAP (`experiments/soap_species_blind/RESULT.md`).

## Files

- `src/general_phase_id/train_and_predict.py`: training, holdout evaluation, prediction, consensus smoothing.
- `src/general_phase_id/build_descriptors.py`: feature selection and normalisation.
- `src/general_phase_id/ref_config.yaml`: whitelist, model and paths.
- `src/process_trajectory.py`: older trajectory helper (its whitelist still lists CSRO features; not used by the current pipeline).
- `experiments/`: exploratory results (not part of the model).
