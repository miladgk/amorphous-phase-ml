# Machine Learning Pipeline for Materials Science (Cu–Zr polyamorphous phases)

Classifies per-atom structural phases in Cu–Zr metallic glasses from descriptors computed by
`Atomic_Simulation_Post-Processing_Pipeline`. The model is a scikit-learn `HistGradientBoostingClassifier`
trained on three homogeneous reference glasses (Cu50Zr50, Cu46Zr54, Cu64Zr36) and applied to a
polyamorphous target box.

## Current result (17 features)

Spatial holdout (atoms with |x − median| < 3.5 Å removed from the target box; 16,227 test atoms):

| Metric | Value |
|---|---|
| CV macro F1 (5-fold stratified, training split) | 0.670 ± 0.005 |
| Holdout accuracy | 67.8 % (11,003 / 16,227) |
| F1 Cu50Zr50 / Cu46Zr54 / Cu64Zr36 | 0.504 / 0.590 / 0.900 |

Older figures (0.854 CV, 74.6 % holdout, R² = 0.938 / 0.972) came from earlier feature tables and are not reproduced by this code.

## Inputs

- Reference tables: `Atomic_Simulation_Post-Processing_Pipeline/outputs/features_{5050,4654,6436}_recomputed.csv`
  (11 snapshots each, averaged per atom).
- Target table: `Atomic_Simulation_Post-Processing_Pipeline/outputs/features_polyamorphous_recomputed.csv`
  (9 snapshots, averaged per atom; dt = 2 fs and timestep index step 1e6, so snapshots are about 2 ns apart).
- Target dump for the periodic box in the consensus filter: `data/polyamorphous/anneal_combined_0.lammpstrj`.
- The feature tables are not tracked in git (about 170 MB). See `DATA.md` and `Atomic_Simulation_Post-Processing_Pipeline/outputs/README_recomputed.md`.

## Run

Run from the repository root, because the paths in `ref_config.yaml` are relative to it:

```bash
python Machine_Learning_Pipeline_for_Materials_Science/src/general_phase_id/train_and_predict.py \
    --config Machine_Learning_Pipeline_for_Materials_Science/src/general_phase_id/ref_config.yaml
```

Outputs go to `output_dir` in the config (`general_phase_id_output/`): `model.pkl`, `predictions.csv`,
`predicted_phases.lammpstrj`, `predicted_phases.xyz`, `config_used.yaml` and plots.

Tests (also run by CI):

```bash
pytest Machine_Learning_Pipeline_for_Materials_Science/tests/ -v
```

Requirements: numpy, pandas, scipy, scikit-learn (1.2.x tested), joblib, pyyaml, freud-analysis 3.2 and pyvoro 1.3.2
(see the root `environment.yml`).

## Features (17)

Configured in `src/general_phase_id/ref_config.yaml` (`feature_whitelist`):

- Structural: `R1_SG`, `voronoi_volume_temporal`, `CN_temporal`, `q4`, `q6`, `gaussian_peak2_center`, `gaussian_peak3_center`, `neighbor_1`, `neighbor_2`
- Energy and stress: `excess_energy_temporal`, `local_energy_fluctuation_temporal`, `hydrostatic_pressure_temporal`, `von_mises_stress_temporal`, `deviatoric_norm_temporal`, `stress_anisotropy_temporal`, `local_enthalpy_temporal`, `pe_atom_std_temporal`

`pe_volume_residual` appeared in earlier lists, but the current pipeline does not produce it, so the model has 17 features, not 18.

Excluded on purpose:
- MRO depth-2/3 features (`mean_*_neighbor_*`, `cu_fraction_*`): they leak simulation-box identity (`PIPELINE.md` §8.2).
- Chemical SRO and neighbour ratios: not validated as non-leaky in the current setup.

## Known limitations

- Each reference phase is one simulation box. The spatial holdout cannot separate phase structure from box-specific signatures (`PIPELINE.md` §9.9).
- Cu50Zr50 and Cu46Zr54 differ by 4 at.% Cu, and their descriptors overlap strongly, so their F1 values are low.
- Species-blind SOAP was tested as an experiment and is not in the model (`experiments/soap_species_blind/RESULT.md`).

## Files

- `src/general_phase_id/train_and_predict.py`: training, holdout evaluation, prediction, spatial consensus smoothing.
- `src/general_phase_id/build_descriptors.py`: feature selection and normalisation.
- `src/general_phase_id/ref_config.yaml`: feature whitelist, model parameters and paths.
- `src/process_trajectory.py`: older trajectory-processing helper. It is not used by the current pipeline, and its whitelist still lists CSRO features. Treat it as legacy.
- `config.yaml` (this folder): legacy configuration from the earlier 2-class/3-class runs; it refers to CSV files that are not in the repository. The current configuration is `src/general_phase_id/ref_config.yaml`.
- `scripts/generate_rigor_artifacts.py`: helper for the rigor artifacts.
- `tests/`: unit tests (4 passing).
- `experiments/`: exploratory results that did not enter the model.
