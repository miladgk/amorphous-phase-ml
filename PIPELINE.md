# Pipeline Technical Specification

## Current State (updated 2026-11-18)

**Deployed model: 17 features** (the whitelist in `Machine_Learning_Pipeline_for_Materials_Science/src/general_phase_id/ref_config.yaml`). `pe_volume_residual` appeared in earlier lists, but the current pipeline does not produce it, so it is not in the model.

Results on the spatial holdout (16,227 test atoms from the three reference boxes; see §5):
- Test accuracy 67.8 % (11,003 / 16,227), CV macro F1 0.670 ± 0.005
- F1: Cu50Zr50 0.504, Cu46Zr54 0.590, Cu64Zr36 0.900

Feature tables are computed from the snapshots with `ANALYSIS_LEVEL: temporal_first_spatial`:
- References: 11 snapshots each, averaged per atom.
- Target: 9 snapshots (`anneal_combined_0 … 8000000`), averaged per atom.
- Snapshot spacing: timestep index step 1e6 with dt = 2 fs (about 2 ns between saved frames). Dynamic (time-correlation) descriptors are therefore not meaningful here.

This document traces the code paths of the feature-extraction and ML pipeline. Every statement below was checked against the code on 2026-11-18. Where a design decision was made for a specific reason, it is noted.

---

## 1. INPUT: Raw Simulation Data

### 1.1 File format
LAMMPS dump files (`.lammpstrj`), one snapshot per file:
- References: `Atomic_Simulation_Post-Processing_Pipeline/files/snapshots_{5050,4654,6436}/anneal_*.lammpstrj` (11 each; 11,664 / 11,664 / 13,500 atoms).
- Target: `Machine_Learning_Pipeline_for_Materials_Science/data/polyamorphous/anneal_combined_*.lammpstrj` (9 frames; 36,828 atoms; box 181.16 × 58.96 × 58.96 Å).

Columns read by `io_module.py` from the `ITEM: ATOMS` section:
- `id`, `type` (1 = Cu, 2 = Zr), `x`, `y`, `z` (Å)
- `c_pe_atom`: per-atom potential energy (eV). If missing, set to NaN with a warning.
- `c_stress_atom[1]` … `c_stress_atom[6]`: per-atom stress × volume (bar·Å³), order xx, yy, zz, xy, xz, yz.
  - LAMMPS defines per-atom stress as the **negative** of the per-atom pressure tensor, so a positive diagonal value means tension. The feature named `hydrostatic_pressure` (§2.2) is the mean of these diagonal components, so positive = tension, negative = compression. The dashboard legend uses this convention.
- The dumps also contain `vx`, `vy`, `vz`; they are not used for the features.

### 1.2 Config parameters
`Atomic_Simulation_Post-Processing_Pipeline/config.yaml` (main keys):
- `SNAPSHOT_DIRECTORY`, `output_csv`: input folder and output table (resolved relative to the config file).
- `ANALYSIS_LEVEL: temporal_first_spatial`: temporal average, then spatial average of g_i(r) over Voronoi neighbours, then per-atom peak analysis (§2.3).
- RDF: `R_MIN 0.01`, `R_MAX 12.0` Å, `NUM_BINS 100`; Savitzky–Golay `SMOOTHING_WINDOW 11`, `SMOOTHING_POLYORDER 5`.
- Peak search: `G_MIN_PEAK_HEIGHT 0.2`, `G_MIN_PEAK_PROMINENCE 0.05`, `G_MIN_PEAK_DISTANCE 1`, `R1_MAX_SEARCH_R 4.0`; targeted search `T_MIN_PEAK_PROMINENCE 0.05`, `T_MIN_PEAK_DISTANCE 4`, `T_R_MIN_FACTOR 1.5`, `T_R_MAX_FACTOR 2.2`.
- Gaussian fit: `ENABLE_GAUSSIAN_FIT`, `FIT_R_MIN_FACTOR 1.5`, `FIT_R_MAX_FACTOR 2.2`, `FIT_PEAK_WINDOW 0.5`, `FIT_MAXFEV 2000`.
- `COMPUTE_SECOND_LEVEL_MRO: true`, `MRO_DEPTH: 3`: neighbour-averaged (MRO) columns; computed but excluded from the model (§2.4).
- `atomic_radii`: 1 → 1.28 Å, 2 → 1.60 Å (used by the pyvoro fallback, §2.2).
- `composition.compute_from_snapshot: true`: the global composition is counted from each snapshot. It is used only for the CSRO columns, which are not in the model. `type_1_fraction` / `type_2_fraction` are ignored while `compute_from_snapshot` is true.
- `WORKER_TIMEOUT: 600` s per snapshot worker; raise it (e.g. 20000 s) for the 36,828-atom target.
- The number of snapshots is the number of files in `SNAPSHOT_DIRECTORY`; there is no snapshot-count key.

---

## 2. PER-ATOM FEATURE EXTRACTION

Files: `Atomic_Simulation_Post-Processing_Pipeline/src/snapshot_processor.py` (per snapshot), `temporal_averaging.py`, `spatial_analysis_levels.py`, `peak_analysis.py`, orchestrated by `pipeline_orchestrator.py`.

### 2.1 Per-atom values
All 17 model features are per-atom values. Each is averaged over the snapshots of its box (11 or 9), except `pe_atom_std_temporal`, which is a standard deviation over the snapshots.

### 2.2 Feature-by-feature specification

#### `voronoi_volume_temporal`, `CN_temporal`
- **What:** Voronoi cell volume (Å³) and number of Voronoi neighbours (faces).
- **Algorithm:** `voronoi.py`. Primary backend: `freud.locality.Voronoi` on wrapped positions in the periodic box (unweighted). Fallback if freud fails: `pyvoro` with the `atomic_radii` weights. Voronoi face counts (⟨n3,n4,n5,n6⟩) come from freud polytopes + scipy `ConvexHull` (commit `17396ea`; this replaced pyvoro, which ran out of memory).
- **Temporal aggregation:** per-atom mean.

#### `neighbor_1`, `neighbor_2`
- **What:** Number of type-1 (Cu) and type-2 (Zr) Voronoi neighbours (`neighbors_by_type` in `snapshot_processor.py`).
- **Temporal aggregation:** per-atom mean (`neighbor_records.groupby(level=0).mean()` in `temporal_averaging.py`).

#### `q4`, `q6`
- **What:** Steinhardt bond-orientational order parameters, l = 4 and l = 6.
- **Algorithm:** `freud.order.Steinhardt(l, average=False)` with a neighbour list from a ball query with `r_max = 3.5` Å, hard-coded in `snapshot_processor.py` (line 618). The Lechner–Dellago variants (`q4_avg`, `q6_avg`) and `w4` are also written to the table but are not in the model. There are no 5 Å variants in the current tables.
- **Temporal aggregation:** per-atom mean.

#### `R1_SG`, `gaussian_peak2_center`, `gaussian_peak3_center`
- **What:** First-peak position of the atom's smoothed g_i(r) (`R1_SG`), and the centres of two Gaussians fitted to the second shell.
- **Algorithm:** `rdf.py` computes a per-atom g_i(r) (KD-tree, minimum image) for each snapshot. `temporal_averaging.py` averages it over the snapshots. With `temporal_first_spatial`, `spatial_analysis_levels.py` then averages each atom's g_i(r) with those of its Voronoi neighbours, and `peak_analysis.py` analyses each atom's averaged curve:
  - Savitzky–Golay smoothing (window 11, order 5), peak search with `scipy.signal.find_peaks` (parameters in §1.2), R1 = the highest peak below `R1_MAX_SEARCH_R`.
  - Two-Gaussian fit (`scipy.optimize.curve_fit`) in the range `FIT_R_MIN_FACTOR × R1` to `FIT_R_MAX_FACTOR × R1` (1.5–2.2 × R1), initial guesses at √3 × R1 and √4 × R1.
- **Note:** with 100 bins up to 12 Å, R1_SG takes few distinct values (10 in the target table, 9 in each reference table).

#### `excess_energy_temporal`
- **What:** `pe_atom` minus the mean `pe_atom` of atoms of the same type in the same snapshot (eV).
- **Temporal aggregation:** per-atom mean.

#### `local_energy_fluctuation_temporal`
- **What:** Standard deviation of `pe_atom` over the atom's Voronoi neighbours in one snapshot (a spatial fluctuation, eV). Computed in `snapshot_processor.py` together with the other first-shell (MRO depth-1) values.
- **Temporal aggregation:** per-atom mean.

#### Stress invariants: `hydrostatic_pressure_temporal`, `von_mises_stress_temporal`, `deviatoric_norm_temporal`, `stress_anisotropy_temporal`
`compute_stress_invariants()` in `snapshot_processor.py`. Each LAMMPS stress × volume component is divided by the atom's Voronoi volume to give stress in bar (s_ij = stress_ij / V_i). Then:
- `hydrostatic = (sxx + syy + szz) / 3` (bar; positive = tension, see §1.1)
- `von_mises = sqrt(0.5·((sxx−syy)² + (syy−szz)² + (szz−sxx)²) + 3·(sxy² + sxz² + syz²))` (bar)
- `deviatoric_norm = sqrt(dxx² + dyy² + dzz² + 2·(sxy² + sxz² + syz²))` with d_ii = s_ii − hydrostatic (bar)
- `stress_anisotropy = von_mises / (|hydrostatic| + 1e-10)` (dimensionless)
- **Temporal aggregation:** per-atom mean.
- The absolute values depend on dividing by the Voronoi volume. The dashboard and figures divide bar by 10 to show MPa.

#### `local_enthalpy_temporal`
- **What:** `pe_atom + hydrostatic × V_voronoi × 6.2415e-7` (eV; 1 bar·Å³ = 6.2415e-7 eV).
- **Temporal aggregation:** per-atom mean.

#### `pe_atom_std_temporal`
- **What:** Standard deviation of each atom's `pe_atom` over the snapshots (`grouped_atoms['pe_atom'].std()` in `temporal_averaging.py`).

### 2.3 Coordinates in the output table
The `x`, `y`, `z` columns are not time-averaged. `temporal_averaging.py` keeps the coordinates of the last record of each atom in the list of processed snapshots. That list is filled in the order the parallel workers finish (`as_completed`), so the coordinates come from one snapshot, not necessarily the last one in time. The spatial holdout (§5) and the consensus filter (§8) use these coordinates.

### 2.4 Columns computed but not in the model
- **MRO depth-1/2/3** (`mean_neighbor_*`, `mean_2nd_neighbor_*`, `mean_3rd_neighbor_*`, `cu_fraction_*`): excluded because they leak simulation-box identity (§9.2).
- **sqrt-ratio features** (`sqrt3_ratio`, …): assume crystal geometries.
- **`w4`, `q4_avg`, `q6_avg`, `s2_entropy_*`**, **angle features** (`mean_angle_*`): not in the whitelist.
- **CSRO** (`csro_unlike_temporal`, `csro_like_temporal`, `csro_unlike_std_temporal`): not validated as leak-free in the current setup; not in the whitelist.

---

## 3. TEMPORAL AGGREGATION

File: `Atomic_Simulation_Post-Processing_Pipeline/src/temporal_averaging.py`, function `perform_temporal_averaging()`.

All snapshots' atom records are flattened into one DataFrame and grouped by atom `id`:
1. Most columns: `grouped_atoms[col].mean()`.
2. g_i(r): summed over snapshots and divided by the count.
3. `pe_atom_std_temporal` and `csro_unlike_std_temporal`: `.std()` over snapshots.

Every `*_temporal` feature is therefore a per-atom mean (or standard deviation) over 11 (reference) or 9 (target) snapshots, saved every 1e6 MD steps (dt = 2 fs, about 2 ns apart).

---

## 4. NORMALIZATION

File: `Machine_Learning_Pipeline_for_Materials_Science/src/general_phase_id/build_descriptors.py`, function `build_general_descriptors()`.

| Feature | Rule | Model column |
|---------|------|--------------|
| `R1_SG` | Raw passthrough (length-scale reference) | `R1_SG` |
| `voronoi_volume_temporal` | ÷ per-atom R1_SG³ | `voronoi_volume_temporal_norm` |
| `CN_temporal`, `q4`, `q6`, `neighbor_1`, `neighbor_2` | Raw passthrough (dimensionless) | same name |
| `gaussian_peak2_center`, `gaussian_peak3_center` | ÷ per-atom R1_SG | `*_norm` |
| `excess_energy_temporal`, `local_energy_fluctuation_temporal` | Raw passthrough (already local) | same name |
| `hydrostatic_pressure_temporal`, `von_mises_stress_temporal`, `deviatoric_norm_temporal`, `stress_anisotropy_temporal`, `local_enthalpy_temporal`, `pe_atom_std_temporal` | z-score with pooled reference mean/std | `*_norm` |

### norm_params source
`load_reference_phases()` in `train_and_predict.py` reads all reference tables, pools the stress/energy columns, and computes mean and std (`norm_params`) and min/max (`feature_ranges`, used only for an out-of-distribution warning). The target is z-scored with the same reference parameters, not with its own statistics.

These values are computed from the reference tables at each run and written to `output_dir` as `norm_params.pkl` and `feature_ranges.pkl` (joblib dicts `{feature: {mean, std}}` and `{feature: {min, max}}`) by `save_run_summaries()`. They record what the run used; inference inside `train_and_predict.py` still recomputes them rather than loading the files.

---

## 5. TRAIN/TEST SPLIT

File: `train_and_predict.py`, function `spatial_holdout_split()`. Parameters in `ref_config.yaml`: `split_axis: x`, `boundary_cutoff: 3.5`.

The split is made **inside each reference box**; the target box is not used for validation (it has no labels):
1. Read the box's feature table (it contains `x`, `y`, `z`; see §2.3).
2. `median` = median of x.
3. `boundary = median − 3.5 ≤ x ≤ median + 3.5` (a 7 Å band).
4. `train = x ≤ median AND NOT boundary`; `test = x > median AND NOT boundary`.

| Box | Atoms | Train | Test | Boundary (excluded) |
|---|---|---|---|---|
| Cu50Zr50 | 11,664 | 5,144 | 5,132 | 1,388 |
| Cu46Zr54 | 11,664 | 5,132 | 5,137 | 1,395 |
| Cu64Zr36 | 13,500 | 5,969 | 5,958 | 1,573 |
| **Total** | 36,828 | 16,245 | **16,227** | 4,356 |

No test rows are dropped: the 16,227 test atoms are the row sums of the confusion matrix. Each atom has one row, so no atom is in both train and test. The split is not periodic: atoms at the two ends of the x range are neighbours across the periodic boundary, but they are in different sets.

---

## 6. MODEL INPUT

### 6.1 Classifier input
- **Shape:** `(16,245, 17)` for training, `(16,227, 17)` for the holdout test.
- **Column order** (as stored in `model.pkl` → `training_meta['feature_order']`; this is the order of the normalised table, not the whitelist order):

```
voronoi_volume_temporal_norm, CN_temporal, R1_SG,
gaussian_peak2_center_norm, gaussian_peak3_center_norm, q4, q6,
local_energy_fluctuation_temporal, pe_atom_std_temporal_norm,
hydrostatic_pressure_temporal_norm, von_mises_stress_temporal_norm,
deviatoric_norm_temporal_norm, stress_anisotropy_temporal_norm,
local_enthalpy_temporal_norm, excess_energy_temporal,
neighbor_1, neighbor_2
```

- **Labels:** 0 = Cu50Zr50, 1 = Cu46Zr54, 2 = Cu64Zr36.

### 6.2 Feature column order is critical
Tree models index features by position, not by name. `training_meta['feature_order']` stores the exact order; inference must build columns in the same order.

---

## 7. MODEL

### 7.1 Algorithm
`sklearn.ensemble.HistGradientBoostingClassifier` (scikit-learn 1.2.2 in the tested environment). HistGB handles NaN values natively; no rows are dropped.

### 7.2 Hyperparameters (from ref_config.yaml)
```yaml
random_state: 42
early_stopping: true
validation_fraction: 0.1
n_iter_no_change: 10
max_iter: 500
learning_rate: 0.05
max_leaf_nodes: 31
min_samples_leaf: 20
```
Cross-validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the training split, scoring `f1_macro`. Permutation importance: `permutation_importance(n_repeats=10, random_state=42)` on the holdout test set.

### 7.3 Files written by `train_and_predict.py`
Path: `output_dir` in the config (`general_phase_id_output/`).

| File | Contents |
|------|----------|
| `model.pkl` | joblib dict with `model` and `training_meta` (model type, 17 features, class names, whitelist, feature order) |
| `config_used.yaml` | Copy of the config used |
| `predictions.csv` | Target table plus `predicted_phase_index`, `predicted_phase_name`, `confidence` (not tracked; ~87 MB) |
| `predicted_phases.xyz` | Extended XYZ for OVITO: species, x, y, z, phase, confidence (not tracked) |
| `phase_mapping.csv` | phase index → name |
| `validation_summary.json` | Holdout accuracy, CV macro F1 (mean and std), confusion matrix, per-class precision/recall/F1, per-box split counts, target phase fractions at each threshold, feature order |
| `permutation_importance.json` | Top-10 permutation importance on the holdout (mean and std; whitelist name and model column) |
| `norm_params.pkl`, `feature_ranges.pkl` | Pooled reference statistics used in this run (§4) |

The last four are written by `save_run_summaries()` (added 2026-11-18). `scripts/generate_real_dashboard_data.py` copies `validation_summary.json` to `docs/assets/` and `results/`, where `scripts/generate_dark_figures.py` reads it.

### 7.4 training_meta structure
```python
{
    'model_type': 'HistGradientBoostingClassifier',
    'n_features': 17,
    'n_classes': 3,
    'class_names': ['Cu50Zr50', 'Cu46Zr54', 'Cu64Zr36'],
    'mro_depth': 3,              # default value; MRO columns are not in the model
    'feature_whitelist': [...],  # the 17 whitelist names
    'feature_order': [...],      # exact column order (§6.1)
}
```

---

## 8. INFERENCE ON THE TARGET

1. **Post-processing** (`pipeline_orchestrator.py`): all descriptors for the 9 target snapshots, temporal and spatial averaging. About 35 minutes in total with one worker; eight workers ran out of memory on a 7 GB machine.
2. **Descriptor building** (`build_descriptors.py`): R1 normalisation and z-scoring with the reference `norm_params` (§4).
3. **Prediction**: `model.predict` and `model.predict_proba`; confidence = maximum probability.
4. **Spatial consensus** (`apply_spatial_consensus()` in `train_and_predict.py`; `inference.spatial_consensus` in the config: `cutoff 3.6` Å, `weight 0.35`):
   - Neighbours: all pairs within 3.6 Å (`scipy.spatial.KDTree.query_pairs`), periodic. The box is read from `BOX BOUNDS` of `target_structure.dump_path`, and positions are wrapped into the cell.
   - `P_smoothed(i) = 0.65 · P(i) + 0.35 · mean_j P(j)`, renormalised; label = argmax, confidence = max of `P_smoothed`.
   - One pass. Without a dump file the filter runs non-periodic and prints a warning.
5. **Thresholds** (`inference.multiple_thresholds`: 0.5–0.9): atoms with confidence below the threshold are counted as "uncertain" in the report. Every atom still gets a label in the output files.
6. **Export**: `predicted_phases.xyz`, `phase_mapping.csv`, `predictions.csv`, then the run summaries (§7.3).

Target phase fractions (17-feature run, after consensus):

| Threshold | Cu50Zr50 | Cu46Zr54 | Cu64Zr36 | Uncertain |
|---|---|---|---|---|
| 0.50 | 22.6 % (8,330) | 25.9 % (9,524) | 35.9 % (13,211) | 15.6 % (5,763) |
| 0.60 | 4.4 % | 14.0 % | 33.1 % | 48.5 % |
| 0.70 | 0.2 % | 5.9 % | 29.8 % | 64.1 % |
| 0.80 | 0.0 % | 1.1 % | 25.9 % | 73.0 % |
| 0.90 | 0.0 % | 0.0 % | 18.6 % | 81.4 % |

---

## 9. KNOWN LIMITATIONS AND CAVEATS

### 9.1 Holdout is one simulation box per phase
Each reference phase is a single box. The spatial holdout splits atoms inside one box, so it cannot separate phase structure from a box-specific signature. A species-blind SOAP test (`Machine_Learning_Pipeline_for_Materials_Science/experiments/soap_species_blind/RESULT.md`) gave Cu50/Cu46 F1 ≈ 0.64/0.66 but is unresolved for the same reason. The reference boxes also differ in density (Cu50 0.0569, Cu46 0.0556, Cu64 0.0628 atoms/Å³; target 0.0585).

### 9.2 MRO leakage (resolved by exclusion)
Neighbour-averaged depth-2/3 features gave high CV scores but leaked box identity on the spatial holdout (commit `0ed3604`). They and `cu_fraction_*` are excluded by the whitelist.

### 9.3 Cu50Zr50 vs Cu46Zr54
The two compositions differ by 4 at.% Cu. Their descriptors overlap strongly (F1 0.50 and 0.59; about 40 % of each class is predicted as the other). On the target, almost no Cu50Zr50 atom reaches confidence 0.70.

### 9.4 q4/q6 at 3.5 Å only
`r_max = 3.5` Å is hard-coded in `snapshot_processor.py`. Other cutoffs were not evaluated for the 17-feature model.

### 9.5 Coarse R1
With 100 bins up to 12 Å, R1_SG has only 9–10 distinct values per table. Compared with the earlier tables, R1_SG and the Gaussian centres differ for about 4–9 % of atoms (correlation about 0.85–0.96); the cause is not established.

### 9.6 norm_params recomputed at each run
`norm_params` are recomputed from the reference tables at each run (and saved, §4). If the tables move or change, the normalisation changes; compare the saved `norm_params.pkl` between runs to detect this.

### 9.7 Coordinates from one snapshot
The split and the consensus filter use coordinates from one snapshot, chosen by worker completion order (§2.3), while the features are averaged over all snapshots.

### 9.8 Regression not re-run
An earlier version trained a `HistGradientBoostingRegressor` for the local Cu fraction and a Voronoi majority-vote smoothing (`SmoothedPhaseID`, `LocalCuFraction` in the old `predicted_phases.lammpstrj`). Neither is part of the current code path (those export files were removed on 2026-11-18), and their earlier results (R² = 0.938 / 0.972) are not reproduced and must not be quoted.

### 9.9 Legacy code
`Machine_Learning_Pipeline_for_Materials_Science/src/process_trajectory.py` and the ML folder's `config.yaml` belong to the earlier pipeline. The `WHITELIST` in `process_trajectory.py` still includes CSRO features. The current path is `train_and_predict.py` + `ref_config.yaml`.

### 9.10 Spatial analysis module
`spatial_analysis_levels.py` was removed in commit `eb06523` and restored unchanged from `eb06523^`. `temporal_first_spatial` needs it; `temporal_only` runs without it but does not produce the spatially averaged R1/Gaussian features.
