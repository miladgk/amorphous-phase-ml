"""
build_descriptors.py — Part 1: Material-Agnostic Descriptor Layer
==================================================================
Takes ANY post-processing output CSV (per-atom temporal-averaged features)
and returns a fixed-width, unit-free feature matrix using normalization rules
that make features comparable across different element systems.

Normalization rules:
1. Length-scale features: divide by R1_SG (or R1_SG^3 for volumes)
2. Energy-scale features: z-score normalize per structure
3. Stress features: z-score normalize per structure
4. Already-dimensionless features: pass through unchanged
5. sqrt-ratio features: EXCLUDED (assume specific crystal geometries)
6. mean_2nd_/3rd_neighbor_*: EXCLUDED (known leakage in prior ablation)
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


# =============================================================================
# Features to EXCLUDE from the general framework
# =============================================================================

# sqrt-ratio features assume specific crystallographic reference ratios tied
# to particular coordination geometries — NOT material-agnostic
SQRT_RATIO_FEATURES = [
    'sqrt3_peak', 'sqrt4_peak', 'sqrt7_peak', 'sqrt12_peak',
    'sqrt3_ratio', 'sqrt4_ratio', 'sqrt7_ratio', 'sqrt12_ratio',
    'ratio_validated_sqrt3', 'ratio_validated_sqrt4',
    'ratio_validated_sqrt7', 'ratio_validated_sqrt12',
    'validated_sqrt3_r_SG', 'validated_sqrt4_r_SG',
    'validated_sqrt7_r_SG', 'validated_sqrt12_r_SG',
]

# MRO depth-1/2/3 features — known leakage from prior ablation under spatial
# holdout testing. These act as simulation-box identity proxies rather than
# genuine local structure descriptors, and should NOT be part of a general method.
# The original validated Config 2 used ONLY the 9 core baseline features + 9 thermo
# features, explicitly excluding all mean_neighbor_* (depth-1), mean_2nd_neighbor_*
# (depth-2), and mean_3rd_neighbor_* (depth-3) features.
MRO_DEPTH_123_FEATURES = [
    # Depth-1 (first-shell MRO) — these were excluded from Config 2
    'mean_neighbor_volume_temporal', 'std_neighbor_volume_temporal',
    'mean_neighbor_pentagon_fraction_temporal', 'std_neighbor_pentagon_fraction_temporal',
    'mean_neighbor_CN_temporal', 'mean_neighbor_free_volume_temporal',
    'mean_neighbor_asphericity_temporal',
    # Depth-2 (second-shell MRO)
    'mean_2nd_neighbor_volume_temporal', 'std_2nd_neighbor_volume_temporal',
    'mean_2nd_neighbor_free_volume_temporal',
    'mean_2nd_neighbor_pentagon_fraction_temporal',
    'mean_2nd_neighbor_CN_temporal',
    'cu_fraction_2nd_shell_temporal', 'zr_fraction_2nd_shell_temporal',
    # Depth-3 (third-shell MRO)
    'mean_3rd_neighbor_volume_temporal', 'std_3rd_neighbor_volume_temporal',
    'mean_3rd_neighbor_free_volume_temporal',
    'mean_3rd_neighbor_pentagon_fraction_temporal',
    'mean_3rd_neighbor_CN_temporal',
    'cu_fraction_3rd_shell_temporal', 'zr_fraction_3rd_shell_temporal',
]

# Columns to always exclude (metadata, non-feature)
META_COLUMNS = ['id', 'type', 'x', 'y', 'z', 'radius', 'group_id',
                'g_i_r_temporal_avg', 'number_of_global_peaks_SG',
                'global_peak_r_values_SG', 'global_peak_heights_SG',
                'num_targeted_peaks_second_third',
                'targeted_peak_r_second_third',
                'targeted_peak_heights_second_third',
                'Ri_over_R1_ratios_SG', 'gaussian_fit_successful',
                'gaussian_fit_r_squared']


def _is_length_scale(col: str) -> bool:
    """Check if a column is a length-scale feature needing R1_SG normalization.
    
    Length-scale features are distances, positions, or volumes that scale with
    the interatomic spacing. These need division by R1 (for distances) or R1^3
    (for volumes) to become unit-free and comparable across systems.
    """
    volume_cols = ['voronoi_volume_temporal', 'free_volume_temporal',
                   'atomic_sphere_volume',
                   'mean_neighbor_volume_temporal', 'std_neighbor_volume_temporal',
                   'mean_neighbor_free_volume_temporal']
    dist_cols = ['gaussian_peak2_center', 'gaussian_peak3_center',
                 'gaussian_peak2_sigma', 'gaussian_peak3_sigma',
                 'gaussian_peak2_fwhm', 'gaussian_peak3_fwhm',
                 'validated_sqrt3_r_SG', 'validated_sqrt4_r_SG',
                 'validated_sqrt7_r_SG', 'validated_sqrt12_r_SG',
                 'R1_SG', 'R1']
    return col in volume_cols or col in dist_cols


def _is_volume(col: str) -> bool:
    """Check if a column scales as length^3 (needs R1^3 normalization)."""
    return col in [
        'voronoi_volume_temporal', 'free_volume_temporal',
        'atomic_sphere_volume', 'volume_per_neighbor',
        'mean_neighbor_volume_temporal', 'std_neighbor_volume_temporal',
        'mean_neighbor_free_volume_temporal'
    ]


def _is_energy(col: str) -> bool:
    """Check if a column is an energy-scale feature needing z-score normalization.
    
    excess_energy_temporal and local_energy_fluctuation_temporal are already 
    per-atom local quantities — they're handled by _is_already_local_energy.
    """
    return col in [
        'pe_atom_temporal', 'pe_atom_std_temporal',
        'local_enthalpy_temporal', 'pe_volume_residual',
    ]


def _is_already_local_energy(col: str) -> bool:
    """Check if a column is already a properly local energy quantity.
    
    excess_energy_temporal = pe_atom - per-type mean within each snapshot,
    which IS local and doesn't need further normalization.
    local_energy_fluctuation_temporal is similarly local.
    mean_neighbor_pe_temporal is the mean PE of neighbors — already local.
    """
    return col in ['excess_energy_temporal', 'local_energy_fluctuation_temporal',
                   'mean_neighbor_pe_temporal']


def _is_stress(col: str) -> bool:
    """Check if a column is a stress feature needing z-score normalization."""
    return col in [
        'hydrostatic_pressure_temporal', 'von_mises_stress_temporal',
        'deviatoric_norm_temporal', 'stress_anisotropy_temporal',
    ]


def _is_already_dimensionless(col: str) -> bool:
    """Check if a column is already dimensionless and comparable across systems."""
    # Steinhardt order parameters and derived
    if col in ['q4', 'q6', 'w4', 'w6', 'q4_avg', 'q6_avg',
               'entropy', 'q4_divide_q6', 'q4_power_2', 'q6_power_2',
               'CN_density', 'log_CN', 'log_voronoi',
               'neighbor_1', 'neighbor_2', 'neighbor_1_ratio', 'neighbor_2_ratio',
               'neighbor_1_fraction_temporal', 'neighbor_2_fraction_temporal',
               'atomic_radius', 'atomic_sphere_volume',
               'volume_q6_interaction', 'volume_per_neighbor']:
        return True
    # Voronoi face counts, fractions, and coordination number
    if col in ['CN_temporal',
               'n3_temporal', 'n4_temporal', 'n5_temporal', 'n6_temporal',
               'pentagon_fraction_temporal',
               'asphericity_temporal', 'asphericity_std_temporal',
               'isb_temporal',
               'csro_unlike_temporal', 'csro_like_temporal',
               'csro_unlike_std_temporal',
               'mean_neighbor_pentagon_fraction_temporal',
               'std_neighbor_pentagon_fraction_temporal',
               'mean_neighbor_CN_temporal',
               'mean_neighbor_asphericity_temporal']:
        return True
    # s2_entropy — this is an information-theoretic quantity already unit-free.
    # Values are system-size-dependent but unit-free. Include.
    if col in ['s2_entropy_temporal', 's2_entropy_avg_temporal']:
        return True
    # Angle features are already dimensionless (degrees are unit-free)
    if col.startswith('mean_angle') or col.startswith('std_angle') or col.startswith('skewness_angle'):
        return True
    return False


# =============================================================================
# Core function: Build general descriptors
# =============================================================================

def build_general_descriptors(
    csv_path: str,
    structure_id: str,
    r1_column: str = 'R1_SG',
    verbose: bool = True,
    feature_whitelist: list = None,
    skip_normalization: bool = False,
    norm_params: dict = None
) -> pd.DataFrame:
    """
    Build a material-agnostic, unit-free feature matrix from a post-processing
    output CSV.

    Normalization:
    - Length-scale features: divided by each atom's OWN R1_SG (per-atom).
    - Already-local energy: passed through unchanged.
    - Stress/secondary-energy features: z-scored using FIXED, externally-computed 
      norm_params (pooled from all reference structures at training time).
      Never computed from the target file itself (avoids self-referential 
      contamination for mixed-phase targets).
    - Dimensionless features: passed through unchanged.

    Parameters
    ----------
    csv_path : str
        Path to the features CSV file.
    structure_id : str
        Label identifying this structure.
    r1_column : str
        Column containing PER-ATOM first-neighbor distance (default 'R1_SG').
    verbose : bool
        Print normalization details.
    skip_normalization : bool
        Deprecated/no effect.
    norm_params : dict, optional
        Frozen normalization params {feature_name: {mean, std}} pooled from 
        references. Applied to stress/secondary-energy features. If None,
        those features are passed through raw.

    Returns
    -------
    pd.DataFrame with per-atom normalized features.
    """
    df = pd.read_csv(csv_path)
    n_orig = len(df)

    if verbose:
        print(f"Loaded {csv_path}: {len(df)} atoms, {len(df.columns)} columns")

    # =========================================================================
    # Step 1: Determine per-atom R1 reference
    # =========================================================================
    # R1_SG is a PER-ATOM quantity (varies by local environment).
    # We use each atom's OWN R1_SG for normalization, NOT a file-wide median.
    if r1_column not in df.columns:
        alt = 'R1' if r1_column == 'R1_SG' else r1_column
        if alt in df.columns:
            r1_column = alt
            if verbose:
                print(f"  Using per-atom '{r1_column}' for length normalization")
        else:
            # Fallback: compute per-atom R1 estimate from gaussian_peak2_center
            if 'gaussian_peak2_center' in df.columns:
                # If no per-atom R1 available, use a simple heuristic:
                # gaussian_peak2_center / 1.7 (typical R1 ~ peak/1.7 for amorphous)
                # This is a rough local estimate
                df['R1_estimated'] = df['gaussian_peak2_center'].values / 1.7
                r1_column = 'R1_estimated'
                if verbose:
                    print(f"  WARNING: No per-atom {r1_column}. Estimated from gaussian_peak2_center/1.7.")
            else:
                raise ValueError(f"Cannot determine local length scale: no {r1_column} or gaussian_peak2_center.")

    # PER-ATOM R1 values — each atom gets its own normalization factor
    r1_per_atom = df[r1_column].values
    r1_3_per_atom = r1_per_atom ** 3

    if verbose:
        n_unique = df[r1_column].nunique()
        print(f"  Per-atom R1_SG: {n_unique} unique values across {len(df)} atoms")
        print(f"    min={r1_per_atom.min():.4f}, median={np.median(r1_per_atom):.4f}, max={r1_per_atom.max():.4f}")

    # =========================================================================
    # Step 2: Identify which features to include and how to normalize them
    # =========================================================================
    result = pd.DataFrame()
    result['type'] = df['type'].values  # Preserve atom type
    result['structure_id'] = structure_id
    
    # If a feature whitelist is provided, ONLY include those features (plus
    # apply the appropriate normalization to each). This is critical for
    # matching the behavior of the original validated Config 2 model, which
    # used exactly 18 features (9 baseline + 9 thermo) and excluded many
    # other available descriptors that would otherwise be included by the
    # general rules above.
    use_whitelist = feature_whitelist is not None

    # Track which columns were included/excluded for reporting
    included_count = 0
    excluded_meta = 0
    excluded_sqrt = 0
    excluded_mro = 0
    excluded_other = 0
    excluded_gaussian_extra = 0

    for col in df.columns:
        # Skip metadata / non-feature columns
        if col in META_COLUMNS:
            excluded_meta += 1
            continue

        # If whitelist mode: skip anything not in the whitelist
        if use_whitelist and col not in feature_whitelist:
            excluded_other += 1
            continue

        # Skip sqrt-ratio features (not material-agnostic)
        if col in SQRT_RATIO_FEATURES:
            excluded_sqrt += 1
            continue

        # Skip MRO depth-1/2/3 features (known leakage from spatial holdout ablation)
        if col in MRO_DEPTH_123_FEATURES:
            excluded_mro += 1
            continue

        # Skip Gaussian fit extras (height/sigma/fwhm are RDF-shape descriptors
        # that don't normalize cleanly across systems with different densities)
        if col in ['gaussian_peak2_height', 'gaussian_peak3_height',
                   'gaussian_peak2_sigma', 'gaussian_peak3_sigma',
                   'gaussian_peak2_fwhm', 'gaussian_peak3_fwhm']:
            excluded_gaussian_extra += 1
            continue

        # Skip object-type columns and columns that are not numeric
        if not np.issubdtype(df[col].dtype, np.number):
            excluded_other += 1
            continue

        # Skip the R1 column itself UNLESS it's in the whitelist (in which case
        # include it as a raw feature, matching the original Config 2 behavior
        # where R1 was used as an absolute length-scale feature)
        if col == r1_column or col == 'R1':
            if use_whitelist and col in feature_whitelist:
                # Whitelisted: include as raw feature
                result[col] = df[col].values
                if verbose:
                    print(f"  {col}: passed through (whitelisted length-scale reference)")
                continue
            else:
                excluded_other += 1
                continue

        # Skip combined/engineered features that assume specific element systems:
        # These are in the ML table but not the raw output, so they won't appear
        # anyway for an arbitrary new structure, but be safe
        if col in ['R3_minus_R1', 'R4_minus_R1', 'R7_minus_R1', 'R12_minus_R1',
                   'R4_minus_R3', 'R7_minus_R4', 'R12_minus_R7',
                   'R1_time_CN_temporal', 'R3_minus_R1_time_q4',
                   'sqrt3_ratio_divide_sqrt4_ratio', 'CN_time_q4', 'CN_time_q6',
                   'entropy_time_q4', 'entropy_time_q6',
                   'volume_q6_interaction']:
            excluded_other += 1
            continue

        # Now normalize based on feature type
        if _is_volume(col):
            # PER-ATOM normalization: each atom's volume divided by its OWN R1^3
            result[col + '_norm'] = df[col].values / r1_3_per_atom
            if verbose:
                print(f"  {col}: normalized by per-atom R1^3 (unit-free ratio)")

        elif col == 'gaussian_peak2_center' or col == 'gaussian_peak3_center':
            # PER-ATOM normalization: each atom's distance divided by its OWN R1
            result[col + '_norm'] = df[col].values / r1_per_atom
            if verbose:
                print(f"  {col}: normalized by per-atom R1 (unit-free ratio ~1.4-1.8)")

        elif _is_already_local_energy(col):
            # Already per-atom local: pass through directly
            result[col] = df[col].values
            if verbose:
                print(f"  {col}: passed through (already local, no re-normalization)")

        elif _is_energy(col):
            # Energy features: use norm_params if provided, raw if not
            # (norm_params come from reference training data — avoids self-referential
            #  z-scoring that would flatten phase differences in mixed targets)
            vals = df[col].values
            if norm_params and col in norm_params:
                mean_v = norm_params[col]['mean']
                std_v = norm_params[col]['std']
                if std_v > 0:
                    result[col + '_norm'] = (vals - mean_v) / std_v
                else:
                    result[col + '_norm'] = vals * 0.0
                if verbose:
                    print(f"  {col}: z-scored using external params (mean={mean_v:.4f}, std={std_v:.4f})")
            else:
                # Pass through raw — no self-referential z-score
                result[col] = vals
                if verbose:
                    print(f"  {col}: passed through raw (no self-referential z-score)")

        elif _is_stress(col):
            # Stress features: use norm_params if provided, raw if not
            vals = df[col].values
            if norm_params and col in norm_params:
                mean_v = norm_params[col]['mean']
                std_v = norm_params[col]['std']
                if std_v > 0:
                    result[col + '_norm'] = (vals - mean_v) / std_v
                else:
                    result[col + '_norm'] = vals * 0.0
                if verbose:
                    print(f"  {col}: z-scored using external params (mean={mean_v:.4f}, std={std_v:.4f})")
            else:
                # Pass through raw — no self-referential z-score
                result[col] = vals
                if verbose:
                    print(f"  {col}: passed through raw (no self-referential z-score)")

        elif _is_already_dimensionless(col):
            result[col] = df[col].values
            if verbose:
                print(f"  {col}: passed through (already dimensionless)")

        else:
            # Unknown feature: skip
            if verbose:
                print(f"  {col}: UNKNOWN TYPE, excluded")
            excluded_other += 1

    # =========================================================================
    # Step 3: Report exclusion summary
    # =========================================================================
    if verbose:
        n_final = len(result.columns) - 2  # minus type and structure_id
        print(f"\nFeature summary:")
        print(f"  Included features:         {n_final}")
        print(f"  Excluded (metadata):       {excluded_meta}")
        print(f"  Excluded (sqrt-ratios):    {excluded_sqrt}")
        print(f"  Excluded (MRO depth-2/3):  {excluded_mro}")
        print(f"  Excluded (Gaussian extra): {excluded_gaussian_extra}")
        print(f"  Excluded (other):          {excluded_other}")
        print(f"  Total columns in output:   {len(result.columns)}")
        print(f"  Non-NaN in output: {result.notna().sum().min()}/{len(result)} atoms")

    return result


def build_general_descriptors_from_df(
    df: pd.DataFrame,
    structure_id: str,
    r1_column: str = 'R1_SG',
    verbose: bool = True,
    feature_whitelist: list = None,
    skip_normalization: bool = False
) -> pd.DataFrame:
    """
    Build a material-agnostic, unit-free feature matrix from an already-loaded
    DataFrame (e.g., an ML table merged with raw output columns).
    
    This is identical to build_general_descriptors() but takes a DataFrame
    directly instead of reading from a CSV path.
    
    Parameters
    ----------
    skip_normalization : bool
        If True, features are passed through as-is (no z-score or R1 normalization).
        This is used when the input is already a pre-normalized ML table.
    """
    n_orig = len(df)

    if verbose:
        print(f"Loaded DataFrame: {len(df)} atoms, {len(df.columns)} columns")

    # =========================================================================
    # Step 1: Determine R1 reference for this structure
    # =========================================================================
    if r1_column not in df.columns:
        # Fallback: try 'R1' if R1_SG not found
        alt = 'R1' if r1_column == 'R1_SG' else r1_column
        if alt in df.columns:
            r1_column = alt
            if verbose:
                print(f"  Using '{r1_column}' instead of '{r1_column}' for length normalization")
        else:
            # Last resort: use the per-atom gaussian_peak2_center mean as R1 estimate
            if 'gaussian_peak2_center' in df.columns:
                r1_value = df['gaussian_peak2_center'].median()
                if verbose:
                    print(f"  WARNING: No {r1_column} column. Using median gaussian_peak2_center ({r1_value:.4f}) as R1 estimate.")
            else:
                raise ValueError(f"Cannot determine length scale: no {r1_column} or gaussian_peak2_center column found.")

    r1 = df[r1_column].median()  # Use per-structure median R1
    r1_3 = r1 ** 3

    if verbose:
        print(f"  R1 reference (median): {r1:.4f}, R1^3: {r1_3:.4f}")

    # =========================================================================
    # Step 2: Identify which features to include and how to normalize them
    # =========================================================================
    result = pd.DataFrame()
    result['type'] = df['type'].values  # Preserve atom type
    result['structure_id'] = structure_id
    
    use_whitelist = feature_whitelist is not None

    # Track which columns were included/excluded for reporting
    included_count = 0
    excluded_meta = 0
    excluded_sqrt = 0
    excluded_mro = 0
    excluded_other = 0
    excluded_gaussian_extra = 0

    for col in df.columns:
        # Skip metadata / non-feature columns
        if col in META_COLUMNS:
            excluded_meta += 1
            continue

        # If whitelist mode: skip anything not in the whitelist
        if use_whitelist and col not in feature_whitelist:
            excluded_other += 1
            continue

        # Skip sqrt-ratio features (not material-agnostic)
        if col in SQRT_RATIO_FEATURES:
            excluded_sqrt += 1
            continue

        # Skip MRO depth-1/2/3 features (known leakage from spatial holdout ablation)
        if col in MRO_DEPTH_123_FEATURES:
            excluded_mro += 1
            continue

        # Skip Gaussian fit extras
        if col in ['gaussian_peak2_height', 'gaussian_peak3_height',
                   'gaussian_peak2_sigma', 'gaussian_peak3_sigma',
                   'gaussian_peak2_fwhm', 'gaussian_peak3_fwhm']:
            excluded_gaussian_extra += 1
            continue

        # Skip object-type columns and columns that are not numeric
        if not np.issubdtype(df[col].dtype, np.number):
            excluded_other += 1
            continue

        # Skip the R1 column itself UNLESS it's in the whitelist
        if col == r1_column or col == 'R1':
            if use_whitelist and col in feature_whitelist:
                result[col] = df[col].values
                if verbose:
                    print(f"  {col}: passed through (whitelisted length-scale reference)")
                continue
            else:
                excluded_other += 1
                continue

        # Skip combined/engineered features
        if col in ['R3_minus_R1', 'R4_minus_R1', 'R7_minus_R1', 'R12_minus_R1',
                   'R4_minus_R3', 'R7_minus_R4', 'R12_minus_R7',
                   'R1_time_CN_temporal', 'R3_minus_R1_time_q4',
                   'sqrt3_ratio_divide_sqrt4_ratio', 'CN_time_q4', 'CN_time_q6',
                   'entropy_time_q4', 'entropy_time_q6',
                   'volume_q6_interaction']:
            excluded_other += 1
            continue

        # Now normalize based on feature type (unless skip_normalization is True)
        if skip_normalization:
            # ML table mode: pass through as-is (already normalized)
            result[col] = df[col].values
            if verbose:
                print(f"  {col}: passed through (ML table, no re-normalization)")
        elif _is_volume(col):
            result[col + '_norm'] = df[col].values / r1_3
            if verbose:
                print(f"  {col}: normalized by R1^3 ({r1_3:.4f})")

        elif _is_length_scale(col):
            result[col + '_norm'] = df[col].values / r1
            if verbose:
                print(f"  {col}: normalized by R1 ({r1:.4f})")

        elif _is_energy(col):
            vals = df[col].values
            mean_v = np.nanmean(vals)
            std_v = np.nanstd(vals)
            if std_v > 0:
                result[col + '_norm'] = (vals - mean_v) / std_v
            else:
                result[col + '_norm'] = vals * 0.0
            if verbose:
                print(f"  {col}: z-score normalized (mean={mean_v:.4f}, std={std_v:.4f})")

        elif _is_stress(col):
            vals = df[col].values
            mean_v = np.nanmean(vals)
            std_v = np.nanstd(vals)
            if std_v > 0:
                result[col + '_norm'] = (vals - mean_v) / std_v
            else:
                result[col + '_norm'] = vals * 0.0
            if verbose:
                print(f"  {col}: z-score normalized (mean={mean_v:.4f}, std={std_v:.4f})")

        elif _is_already_dimensionless(col):
            result[col] = df[col].values
            if verbose:
                print(f"  {col}: passed through (already dimensionless)")

        else:
            if verbose:
                print(f"  {col}: UNKNOWN TYPE, excluded")
            excluded_other += 1

    # =========================================================================
    # Step 3: Report exclusion summary
    # =========================================================================
    if verbose:
        n_final = len(result.columns) - 2  # minus type and structure_id
        print(f"\nFeature summary:")
        print(f"  Included features:         {n_final}")
        print(f"  Excluded (metadata):       {excluded_meta}")
        print(f"  Excluded (sqrt-ratios):    {excluded_sqrt}")
        print(f"  Excluded (MRO depth-2/3):  {excluded_mro}")
        print(f"  Excluded (Gaussian extra): {excluded_gaussian_extra}")
        print(f"  Excluded (other):          {excluded_other}")
        print(f"  Total columns in output:   {len(result.columns)}")
        print(f"  Non-NaN in output: {result.notna().sum().min()}/{len(result)} atoms")

    return result


def concatenate_structures(descriptor_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate multiple structure descriptor DataFrames into one."""
    return pd.concat(descriptor_list, ignore_index=True)
