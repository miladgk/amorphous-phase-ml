# Pipeline Technical Specification

## Current State as of 2026-07-09

This document traces the actual code paths for every step of the feature-extraction and ML pipeline. Where a design decision was made for a specific reason, it's noted as institutional memory.

---

## 1. INPUT: Raw Simulation Data

### 1.1 File format
LAMMPS dump files (`.lammpstrj`), potentially in multi-frame format, are split into individual snapshots by `pipeline_orchestrator.py`.

Required columns (must be present in `ITEM: ATOMS` section):
- `id` — atom ID (integer)
- `type` — atom type (integer; 1=Cu, 2=Zr in current usage)
- `x`, `y`, `z` — cartesian coordinates (Angstrom)
- `vx`, `vy`, `vz` — velocities
- `c_pe_atom` — per-atom potential energy (eV)
- `c_stress_atom[1]` through `c_stress_atom[6]` — per-atom stress tensor components (bar·Å³)
  - LAMMPS convention: xx, yy, zz, xy, xz, yz
  - Positive = compression

### 1.2 Config parameters
Controlled by `Atomic_Simulation_Post-Processing_Pipeline/config.yaml`:
- `SNAPSHOT_DIRECTORY`: directory containing `.lammpstrj` files
- `type_1_fraction`, `type_2_fraction`: bulk composition fractions (used for `excess_energy` baseline)
- `COMPUTE_SECOND_LEVEL_MRO`, `MRO_DEPTH`: depth of neighbor-averaged features
- `ENABLE_GAUSSIAN_FIT`: whether to compute Gaussian peak fit on RDF
- `TOTAL_SNAPSHOTS`, `EVERY_N_SNAPSHOTS`: temporal averaging window (currently 11 snapshots for reference structures, 9 for polyamorphous)

---

## 2. PER-SNAPSHOT FEATURE EXTRACTION

File: `Atomic_Simulation_Post-Processing_Pipeline/src/snapshot_processor.py`
Function: `process_snapshot()`

Each snapshot is processed independently. The following describes every feature in the current 18-feature whitelist (plus q4_5a/q6_5a which are present in CSV but not in the deployed model).

### 2.1 Per-atom vs per-structure-global
**Critical distinction:** Some features are per-atom (each atom gets its own value), others are per-structure-global (one value for the whole snapshot assigned to every atom). Tracking this distinction is essential for understanding what the model can and cannot learn.

### 2.2 Feature-by-feature specification

#### `R1_SG` — **per-atom**
- **What:** Position (in Angstrom) of the first peak in each atom's local gᵢ(r).
- **Algorithm:** `snapshot_processor.py` calls `rdf.compute_all_local_rdfs()` → `peak_analysis.py` identifies the first peak in the global g(r), then assigns it per-structure. For per-atom R1, it uses the per-atom gᵢ(r) peak position.
- **Parameters:** RDF computed from 0 to `R_MAX` (default ~10Å) with `NUM_BINS` bins. Peak detection uses `scipy.signal.find_peaks` with `prominence=0.1` and `height=0.05`.
- **Note:** R1_SG has 13 unique values across 36,828 atoms for the polyamorphous target, confirming it's genuinely per-atom variable.

#### `voronoi_volume_temporal` — **per-atom**
- **What:** Volume (Å³) of each atom's Voronoi cell.
- **Algorithm:** `snapshot_processor.py` uses `freud.locality.Voronoi()` on the wrapped atomic positions. Cell volume from the Voronoi polytope.
- **Parameters:** Standard Voronoi tessellation with freud. No explicit cutoff (Voronoi is global by construction).
- **Temporal aggregation:** Per-atom mean across snapshots.
- **Normalization:** Divided by per-atom R1_SG³ → becomes unit-free volume ratio.

#### `CN_temporal` — **per-atom**
- **What:** Coordination number = number of Voronoi neighbors.
- **Algorithm:** Number of faces in the Voronoi cell. From `freud.locality.Voronoi()` polytope neighbor count.
- **Parameters:** Same as Voronoi above.
- **Temporal aggregation:** Per-atom mean across snapshots.
- **Normalization:** Already dimensionless, passed through.

#### `q4`, `q6` — **per-atom**
- **What:** Steinhardt bond orientational order parameters (rotationally invariant 2nd-order, l=4 and l=6).
- **Algorithm:** `freud.order.Steinhardt(l=4, average=False)` and `(l=6, average=False)`.
- **Critical parameter:** `r_max=3.5` Å. This is **hardcoded** in `snapshot_processor.py` line 618:
  ```python
  neighbor_list = neighbor_query.query(
      positions, {'r_max': 3.5, 'exclude_ii': True}
  ).toNeighborList()
  ```
  This cutoff was chosen because it corresponds to the first coordination shell in Cu-Zr metallic glasses. Wang et al. (2026) found 5Å carries more information, but q4/q6 at 3.5Å are what the 18-feature model uses.
- **Temporal aggregation:** Per-atom mean across snapshots (via `grouped_atoms['q4'].mean()` in `temporal_averaging.py`).
- **Normalization:** Already dimensionless, passed through.

#### `gaussian_peak2_center` — **per-atom** (with per-structure global R1 normalization)
- **What:** Center position of the second Gaussian peak fitted to the global RDF.
- **Algorithm:** `peak_analysis.py` → class `GaussianFitter` fits a 2-Gaussian model to the RDF in the region `R1 × FIT_R_MIN_FACTOR` to `R1 × FIT_R_MAX_FACTOR`. Default: `FIT_R_MIN_FACTOR=1.2`, `FIT_R_MAX_FACTOR=2.2` (roughly 1.2×R1 to 2.2×R1). The fit uses `scipy.optimize.curve_fit`.
- **Initial guesses:** Peak positions at R1×√3 (~1.732×R1) and R1×√4 (~2.0×R1). These are from FCC/HCP expected ratios for the 2nd and 3rd shells.
- **Important:** This is computed from the **global** RDF of the whole structure, not per-atom. But it's assigned per-atom and then normalized by per-atom R1_SG. The per-atom R1 normalization makes it effectively per-atom.
- **Temporal aggregation:** Per-atom mean across snapshots.
- **Normalization:** Divided by per-atom R1_SG → unit-free ratio (~1.4-1.8).

#### `gaussian_peak3_center` — **per-atom** (same as peak2)
- Same as above for the third peak. Position at R1×√4 ≈ R1×2.0.

#### `neighbor_1` — **per-atom**
- **What:** Number of type-1 (Cu) Voronoi neighbors.
- **Algorithm:** From `neighbors_by_type` dict computed during Voronoi processing. `neighbor_1 = neighbors_by_type.get(1, 0)`.
- **Temporal aggregation:** Per-atom mean across snapshots (via `neighbor_records.groupby(level=0).mean()`).
- **Normalization:** Already dimensionless, passed through.

#### `neighbor_2` — **per-atom**
- Same as neighbor_1 but for type-2 (Zr).

#### `excess_energy_temporal` — **per-atom**
- **What:** Potential energy minus the per-type-per-snapshot mean baseline. `E_excess = E_i - mean(E_atoms_of_same_type_in_this_snapshot)`.
- **Algorithm:** `snapshot_processor.py`: for each snapshot, compute the mean `pe_atom` for type 1 and type 2 separately, then subtract each atom's type-mean from its pe_atom. This makes it a genuinely local feature (how much higher/lower is this atom's energy than average for its chemical species).
- **Temporal aggregation:** Per-atom mean across snapshots.
- **Normalization:** Already local (per-atom, per-type), passed through without re-normalization.

#### `local_energy_fluctuation_temporal` — **per-atom**
- **What:** Not a separate computation in `snapshot_processor.py`. Source: `local_energy_fluctuation` column in CSV. This is the per-atom standard deviation of energy across the temporal window? The code in `snapshot_processor.py` doesn't compute this directly — it's read from the input data frame at line 173: `local_energy_fluctuation_avg = grouped_atoms['local_energy_fluctuation'].mean().to_dict()`.
- **Status:** This feature's computation needs verification against the original pipeline code. The CSV already contains it, suggesting it was computed earlier in the pipeline evolution.
- **Normalization:** Already local, passed through.

#### `hydrostatic_pressure_temporal` — **per-atom**
- **What:** Isotropic stress component: `(sxx + syy + szz) / 3` where sxx = stress_xx / Voronoi_volume, etc. Units: bar.
- **Algorithm:** `snapshot_processor.py` function `compute_stress_invariants()`. Converts from LAMMPS stress×volume to stress (divide by Voronoi volume), then computes hydrostatic as mean of diagonal.
- **Temporal aggregation:** Per-atom mean across snapshots.
- **Normalization:** z-scored using pooled reference statistics (not self-referential). Currently uses parameters computed from all reference structures combined.

#### `von_mises_stress_temporal` — **per-atom**
- **What:** Scalar shear stress invariant. Formula: `sqrt(0.5 * ((sxx-syy)² + (syy-szz)² + (szz-sxx)² + 6*(sxy² + sxz² + syz²)))`. Units: bar.
- **Algorithm:** Same `compute_stress_invariants()` function.
- **Normalization:** z-scored using pooled reference statistics.

#### `deviatoric_norm_temporal` — **per-atom**
- **What:** Norm of the deviatoric stress tensor: `sqrt(sxx² + syy² + szz² + 2*(sxy² + sxz² + syz²))`. Units: bar.
- **Algorithm:** Same `compute_stress_invariants()` function.
- **Normalization:** z-scored using pooled reference statistics.

#### `stress_anisotropy_temporal` — **per-atom**
- **What:** Ratio of von Mises to hydrostatic stress: `von_mises / hydrostatic` (with guard for near-zero hydrostatic). Dimensionless.
- **Algorithm:** Same `compute_stress_invariants()` function.
- **Normalization:** z-scored using pooled reference statistics. (Despite being a ratio, the scale varies across systems.)

#### `local_enthalpy_temporal` — **per-atom**
- **What:** Enthalpy proxy: `E_i + PV_i` where `P = hydrostatic_pressure` and `V = Voronoi_volume`. The stress_x volume terms are converted from bar·Å³ to eV using the conversion factor `1 bar·Å³ = 6.2415e-7 eV`.
- **Algorithm:** Same `compute_stress_invariants()` function.
- **Normalization:** z-scored using pooled reference statistics.

#### `pe_atom_std_temporal` — **per-atom**
- **What:** Standard deviation of each atom's potential energy across the temporal averaging window. Measures energy fluctuation over time.
- **Algorithm:** `temporal_averaging.py`: `pe_atom_std_over_time = grouped_atoms['pe_atom'].std().to_dict()` — per-atom std across snapshot copies.
- **Normalization:** z-scored using pooled reference statistics.

#### `pe_volume_residual` — **per-atom**
- **What:** Residual of a per-atom linear regression of PE vs Voronoi volume: `E_i - (a * V_i + b)` where a,b are fit per snapshot.
- **Algorithm:** `snapshot_processor.py`: for each snapshot, fit a line to all atoms' `pe_atom` vs `voronoi_volume`, then `residual = pe_atom - (slope * volume + intercept)`.
- **Temporal aggregation:** Per-atom mean across snapshots.
- **Normalization:** z-scored using pooled reference statistics.

### 2.3 Features NOT in the whitelist (but computed)

These are computed by the pipeline but excluded from training due to the 18-feature whitelist:

- **MRO depth-1:** `mean_neighbor_volume_temporal`, `mean_neighbor_CN_temporal`, `mean_neighbor_pe_temporal`, etc. — averaged over Voronoi neighbors. Excluded because spatial holdout testing showed they leak simulation-box identity.
- **MRO depth-2/3:** `mean_2nd_neighbor_*`, `mean_3rd_neighbor_*` — same leak issue.
- **sqrt-ratio features:** `sqrt3_ratio`, `sqrt4_ratio`, etc. — assume specific crystal geometries. Not material-agnostic.
- **w4, w6:** 3rd-order Steinhardt invariants. Dimensionless but not in the whitelist.
- **q4_avg, q6_avg:** Lechner-Dellago averaged Steinhardt. Not in the whitelist.
- **s2_entropy:** Pair entropy. Not in the whitelist.
- **CSRO features:** `csro_unlike_temporal`, `csro_like_temporal` — confirmed non-leaky in A-vs-B binary test but not in the deployed 18-feature model.
- **Angle features:** `mean_angle_all_temporal`, etc. — not in whitelist.

---

## 3. TEMPORAL AGGREGATION

File: `Atomic_Simulation_Post-Processing_Pipeline/src/temporal_averaging.py`
Function: `perform_temporal_averaging()`

### Mechanism
All snapshots' atom data are flattened into a single DataFrame. Then:
1. **Steinhardt parameters:** Each snapshot's per-atom q4/q6 values are first averaged from per-snapshot arrays to scalars (the Steinhardt function returns an array per atom if `average=False`), then grouped by `id` and mean-averaged across snapshots.
2. **g_i(r):** Cumulative sum across snapshots (vectorized), then divided by count. Per-atom RDFs are preserved.
3. **Most other features:** Simple `grouped_atoms[col].mean().to_dict()` per atom across snapshots.
4. **Special: `pe_atom_std_temporal`** uses `grouped_atoms['pe_atom'].std()` — this is the per-atom std across time, not a mean.
5. **`csro_unlike_std_temporal`** similarly uses `.std()` rather than `.mean()`.

### What this means
Every `*_temporal` feature is a per-atom mean (or std for two features) across the temporal window of 11 (reference) or 9 (polyamorphous) snapshots. The snapshots are independent trajectory frames at 50ps intervals after equilibration.

---

## 4. NORMALIZATION

File: `Machine_Learning_Pipeline_for_Materials_Science/src/general_phase_id/build_descriptors.py`
Function: `build_general_descriptors()`

### Current normalization rules

| Feature | Rule | Details |
|---------|------|---------|
| `R1_SG` | Raw passthrough | Whitelisted as a length-scale reference feature |
| `voronoi_volume_temporal` | ÷ per-atom R1³ | Each atom's volume ÷ (its own R1_SG)³ |
| `CN_temporal` | Raw passthrough | Already dimensionless |
| `q4`, `q6` | Raw passthrough | Already dimensionless |
| `gaussian_peak2_center` | ÷ per-atom R1 | Each atom's peak position ÷ its own R1_SG |
| `gaussian_peak3_center` | ÷ per-atom R1 | Same as peak2 |
| `neighbor_1`, `neighbor_2` | Raw passthrough | Already dimensionless |
| `excess_energy_temporal` | Raw passthrough | Already local per-atom, per-type |
| `local_energy_fluctuation_temporal` | Raw passthrough | Already local |
| `hydrostatic_pressure_temporal` | z-score from norm_params | Fixed mean/std from reference pool |
| `von_mises_stress_temporal` | z-score from norm_params | Fixed mean/std from reference pool |
| `deviatoric_norm_temporal` | z-score from norm_params | Fixed mean/std from reference pool |
| `stress_anisotropy_temporal` | z-score from norm_params | Fixed mean/std from reference pool |
| `local_enthalpy_temporal` | z-score from norm_params | Fixed mean/std from reference pool |
| `pe_atom_std_temporal` | z-score from norm_params | Fixed mean/std from reference pool |
| `pe_volume_residual` | z-score from norm_params | Fixed mean/std from reference pool |

### norm_params source
Norm_params are computed in `train_and_predict.py` → `load_reference_phases()`:
1. Load all reference CSVs
2. Pool the raw values of stress/energy features across all atoms in all reference structures
3. Compute `mean` and `std` of the pool
4. Store in a dict

**Critical:** These are NOT saved to disk by the current train_and_predict pipeline. They're computed at inference time by re-reading the reference CSVs. The ad-hoc export scripts also compute them from scratch.

### Per-atom vs per-structure normalization
**All normalization is per-atom.** R1_SG varies per atom (13 unique values in polyamorphous target), so the R1-based normalization is genuinely per-atom. z-score normalization uses fixed global parameters but applies to each atom's value individually.

---

## 5. TRAIN/TEST SPLIT

File: `Machine_Learning_Pipeline_for_Materials_Science/src/general_phase_id/train_and_predict.py`
Function: `spatial_holdout_split()`

### Parameters (from ref_config.yaml)
- `split_axis: "x"` — split on the x-coordinate
- `boundary_cutoff: 3.5` Å — atoms within this distance of the split plane are excluded from both train and test

### Algorithm
1. For each reference structure's CSV: read coordinates from the raw CSV (same file as features CSV)
2. Compute `axis_median` = median of x-coordinates
3. `left = x <= median`, `right = x > median`
4. `boundary = |x - median| < 3.5Å`
5. `train = left AND NOT boundary`, `test = right AND NOT boundary`
6. Boundary atoms (~1388 per structure, ~11.9%) are discarded from training and testing

### Atom-ID grouping
**There is NO atom-ID grouping.** The split is purely spatial along x. An atom that appears left of the median goes to train; one right of the median goes to test. Since the temporal averaging produces ONE row per atom (not per snapshot), there's no issue of the same atom appearing in both train and test — each atom has one row.

However, since the raw CSV contains positions from ONE snapshot (the last one), and the features are temporally averaged across 11 snapshots during which atoms diffuse, a given physical atom's time-averaged position may differ from its position in the one snapshot used for the spatial split. This is an acknowledged limitation.

---

## 6. MODEL INPUT

### 6.1 Classifier input
- **Shape:** `(N_train, 18)` where N_train ≈ 16,245 (after spatial holdout split from 36,828 reference atoms)
- **Column order** (18 features):

```
R1_SG, voronoi_volume_temporal, CN_temporal, q4, q6,
gaussian_peak2_center, gaussian_peak3_center,
neighbor_1, neighbor_2,
excess_energy_temporal, local_energy_fluctuation_temporal,
hydrostatic_pressure_temporal, von_mises_stress_temporal,
deviatoric_norm_temporal, stress_anisotropy_temporal,
local_enthalpy_temporal, pe_atom_std_temporal, pe_volume_residual
```

- **Labels:** integer 0=Cu50Zr50, 1=Cu46Zr54, 2=Cu64Zr36

### 6.2 Regressor input
- **Same 18 features** as the classifier, same column order, same spatial splits.
- **Target:** Continuous Voronoi Cu fraction (0.0-1.0), computed as `(#Cu Voronoi neighbors) / (total Voronoi neighbors)`.
- **Shape:** Same as classifier: `(N_train, 18)`.

### 6.3 Feature column order is CRITICAL
Tree-based models (HistGB) index features by **position**, not by name. The `training_meta` dict stores `feature_order` — the exact list of column names in the order they were used during training. Inference MUST produce columns in the exact same order, or predictions will be silently wrong.

---

## 7. MODEL

### 7.1 Algorithm
`HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` from scikit-learn.

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

### 7.3 Saved artifacts

Path: `general_phase_id_output/`

| File | Contents | Created by |
|------|----------|------------|
| `model.pkl` | Dict with `model` (classifier), `training_meta` | `train_and_predict.py` |
| `config_used.yaml` | Copy of `ref_config.yaml` at training time | `train_and_predict.py` |
| `predicted_phases.lammpstrj` | LAMMPS dump with PhaseID, PhaseConfidence, SmoothedPhaseID, LocalCuFraction | Ad-hoc export scripts |
| `README.txt` | Documentation of A/B gradient caveat | Ad-hoc export scripts |
| `predictions.csv` | Full per-atom prediction results | `train_and_predict.py` |

**NOT saved** (must be recomputed): `norm_params.pkl`, `feature_ranges.pkl`. These are computed from reference CSV files at each run.

### 7.4 training_meta structure
```python
{
    'model_type': 'HistGradientBoostingClassifier',
    'n_features': 18,
    'n_classes': 3,
    'class_names': ['Cu50Zr50', 'Cu46Zr54', 'Cu64Zr36'],
    'mro_depth': 3,
    'feature_whitelist': [...],  # the 18-feature list
    'feature_order': [...],  # exact column order used for X
}
```

---

## 8. INFERENCE ON NEW/TARGET STRUCTURES

### 8.1 Steps (from raw dump to final LAMMPS output)

1. **Post-processing pipeline** (`pipeline_orchestrator.py` → `snapshot_processor.py`): Read raw LAMMPS dump(s), compute all per-atom descriptors for the target structure's snapshots. Takes ~30 min per snapshot.

2. **Temporal averaging** (`temporal_averaging.py`): Average per-atom features across the target's snapshots (9 frames).

3. **Descriptor building** (`build_general_descriptors.py`): Apply per-atom R1 normalization and z-score using reference-pooled norm_params.

4. **Classifier prediction** (`model.predict(X_target)`): Get discrete phase labels and confidence (max probability).

5. **Regressor prediction** (ad-hoc): Get continuous LocalCuFraction. Not part of the standard train_and_predict pipeline; run via separate script.

6. **Voronoi spatial smoothing** (ad-hoc, in export scripts):
   - Compute Voronoi neighbor list from target positions (12-14 avg neighbors)
   - For each atom, count neighbor phase labels
   - If ≥ min_agreement (default 0.50) of neighbors share a label, change atom to majority class
   - Single pass, using ORIGINAL labels for voting (not iteratively updated)
   - Produces SmoothedPhaseID
   - Selected by answer-agnostic sweep testing min_agreement in [0.50, 0.55, 0.60, 0.65, 0.70] and picking the value maximizing agreement with region-based ground truth (left=B, middle=A, right=C).

7. **LAMMPS export**: Read the original dump file, append columns:
   - `PhaseID` (raw classifier prediction)
   - `PhaseConfidence` (max probability)
   - `SmoothedPhaseID` (after Voronoi neighbor voting)
   - `LocalCuFraction` (regressor prediction, 0.0-1.0)
   Write as one frame with identical header + new columns.

### 8.2 Current parameters (18-feature model)
- **Smoothing min_agreement:** 0.50
- **Classifier confidence threshold:** 0.50 (below this = "Uncertain" in reporting, but all atoms get a PhaseID regardless; the LAMMPS output has no special Uncertain marker)
- **Regressor:** HistGradientBoostingRegressor, same hyperparameters as classifier

---

## 9. KNOWN LIMITATIONS AND CAVEATS

### 9.1 A/B boundary is a continuous gradient
Confirmed by direct regression (R²=0.938, MAE=0.024 on held-out reference data, R²=0.972 on polyamorphous target). The Cu fraction changes smoothly from ~0.40 to ~0.45 over ~40Å across the left→middle boundary. Discrete PhaseID labels in this region are inherently ambiguous. **Use LocalCuFraction for visualization in this region.**

### 9.2 C's residual 3.6% misclassification
After Voronoi smoothing, ~3.6% of right-region atoms are still labeled as Phase A or B. This is real local compositional heterogeneity, not classifier error: these atoms have mean LocalCuFraction=0.48 vs 0.59 for correctly-labeled Phase C atoms. The model correctly predicts their local environment — it's the local composition that genuinely differs from bulk Cu₆₄Zr₃₆.

### 9.3 q4/q6 at 3.5Å only
The Steinhardt bond order parameters are computed with `r_max=3.5Å` hardcoded in `snapshot_processor.py`. Wang et al. (2026) found 5Å carries more structural information. Adding q4_5a/q6_5a produced no measurable improvement (ΔR²=+0.0008) for this Cu-Zr system, but this may differ for other chemistries.

### 9.4 Per-atom R1 normalization
R1_SG has 13 unique values across the 36,828-atom polyamorphous target. The per-atom R1 normalization is implemented correctly in `build_descriptors.py`.

### 9.5 norm_params not saved
The pipeline does not save norm_params to disk. They are recomputed from reference CSVs at each inference run. This is fragile — if reference CSV files are moved or the whitelist changes, norm_params will silently differ.

### 9.6 Config file lists 21 features
The deployed `ref_config.yaml` lists 21 features (18 original + 3 CSRO), but the actual deployed 18-feature model excludes CSRO features. The config file has not been updated to match the deployed model.

### 9.7 process_trajectory.py has stale whitelist
The `WHITELIST` in `process_trajectory.py` was updated to include CSRO features but the 18-feature model that was validated doesn't use them. This file needs updating for consistency.
