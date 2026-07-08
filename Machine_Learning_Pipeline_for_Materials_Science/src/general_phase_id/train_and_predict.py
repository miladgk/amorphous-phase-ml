"""
train_and_predict.py — Generalized Reference-Phase Classification Framework
==============================================================================
Combines Parts 1-5:
1. Loads reference phases from a YAML config (any number of phases, any chemistry)
2. Applies material-agnostic normalization independently per reference
3. Trains a HistGB classifier with spatial holdout validation
4. Applies to a target structure with OOD warning
5. Exports OVITO-compatible output

Usage:
    python3 general_phase_id/train_and_predict.py --config general_phase_id/ref_config.yaml
"""
import sys, os, yaml, glob, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import freud

sys.path.insert(0, 'Machine_Learning_Pipeline_for_Materials_Science/src')
from general_phase_id.build_descriptors import (
    build_general_descriptors, concatenate_structures,
    SQRT_RATIO_FEATURES, MRO_DEPTH_123_FEATURES
)

# =============================================================================
# Part 1: Load reference phases from config
# =============================================================================

def load_reference_phases(config: dict, verbose: bool = True) -> tuple:
    """
    Load all reference phases listed in the config.
    Each phase gets its OWN per-structure normalization.
    
    Returns
    -------
    (descriptors_df, phase_names, phase_labels, norm_params, feature_ranges)
        descriptors_df: concatenated normalized features across all references
        phase_names: list of phase name strings (in config order)
        phase_labels: integer array (0..N-1) corresponding to each atom
        norm_params: dict of {feature_name: {mean, std}} pooled across all refs
        feature_ranges: dict of {feature_name: {min, max}} for OOD detection
    """
    ref_phases = config['reference_phases']
    n_classes = len(ref_phases)
    feature_whitelist = config.get('feature_whitelist', None)
    
    if verbose:
        print(f'Loading {n_classes} reference phases:')
        if feature_whitelist:
            print(f'  Using feature whitelist ({len(feature_whitelist)} features)')
        else:
            print(f'  Using general normalization rules (all applicable features)')
    
    descriptor_list = []
    labels = []
    
    # First pass: load raw data to compute pooled norm_params
    # Features that need external normalization (stress + secondary energy)
    norm_feature_names = [
        'hydrostatic_pressure_temporal', 'von_mises_stress_temporal',
        'deviatoric_norm_temporal', 'stress_anisotropy_temporal',
        'local_enthalpy_temporal', 'pe_atom_std_temporal', 'pe_volume_residual'
    ]
    pooled_raw = []
    raw_dfs = []
    
    for i, phase in enumerate(ref_phases):
        csv_path = phase['csv_path']
        df = pd.read_csv(csv_path)
        raw_dfs.append(df)
        # Collect raw values for norm features
        available = [c for c in norm_feature_names if c in df.columns]
        pooled_raw.append(df[available])
    
    # Compute pooled norm_params
    pooled_df = pd.concat(pooled_raw, ignore_index=True)
    norm_params = {}
    feature_ranges = {}
    for col in pooled_df.columns:
        vals = pooled_df[col].values
        norm_params[col] = {'mean': float(np.nanmean(vals)), 'std': float(np.nanstd(vals))}
        feature_ranges[col] = {'min': float(np.nanmin(vals)), 'max': float(np.nanmax(vals))}
    
    if verbose:
        print(f'\nComputed pooled norm_params from {len(pooled_df)} reference atoms:')
        for col in sorted(norm_params.keys()):
            p = norm_params[col]
            r = feature_ranges[col]
            print(f'  {col}: mean={p["mean"]:.4f}, std={p["std"]:.4f}, range=[{r["min"]:.4f}, {r["max"]:.4f}]')
    
    # Second pass: build descriptors WITH norm_params
    for i, phase in enumerate(ref_phases):
        name = phase['name']
        csv_path = phase['csv_path']
        
        if verbose:
            print(f'\n  Phase {i}: "{name}" <- {csv_path}')
        
        desc = build_general_descriptors(
            csv_path, structure_id=name, verbose=verbose,
            feature_whitelist=feature_whitelist, norm_params=norm_params)
        descriptor_list.append(desc)
        labels.append(np.full(len(desc), i, dtype=int))
    
    # Concatenate all reference phases
    descriptors = concatenate_structures(descriptor_list)
    phase_names = [p['name'] for p in ref_phases]
    phase_labels = np.concatenate(labels)
    
    if verbose:
        print(f'\nCombined training data: {len(descriptors)} atoms, {len(phase_names)} classes')
        for i, name in enumerate(phase_names):
            n = (phase_labels == i).sum()
            print(f'  {name} ({i}): {n} atoms')
    
    return descriptors, phase_names, phase_labels, norm_params, feature_ranges


def get_feature_columns(descriptors: pd.DataFrame) -> list:
    """Get feature column names, excluding metadata columns."""
    return [c for c in descriptors.columns if c not in ['structure_id', 'type']]


# =============================================================================
# Part 2: Spatial holdout split and training
# =============================================================================

def spatial_holdout_split(
    descriptors: pd.DataFrame,
    labels: np.ndarray,
    phase_names: list,
    config: dict,
    verbose: bool = True
) -> dict:
    """
    Perform spatial holdout split on each reference structure independently.
    
    For each reference phase, we need to:
    1. Get the x,y,z coordinates (they're not in the ML tables, need from raw CSV)
    2. Split at the median of the split_axis
    3. Exclude boundary atoms
    
    Returns dict with 'X_train', 'X_test', 'y_train', 'y_test', 'train_idx', 'test_idx',
    plus split diagnostics.
    """
    split_axis = config['validation'].get('split_axis', 'x')
    boundary_cutoff = config['validation'].get('boundary_cutoff', 3.5)
    ref_phases = config['reference_phases']
    
    # We need coordinates from the raw CSVs
    feature_cols = get_feature_columns(descriptors)
    X_all = descriptors[feature_cols].values
    
    # Track which atoms belong to which reference CSV
    train_mask = np.zeros(len(descriptors), dtype=bool)
    test_mask = np.zeros(len(descriptors), dtype=bool)
    
    all_train_X = []
    all_train_y = []
    all_test_X = []
    all_test_y = []
    
    total_boundary = 0
    
    if verbose:
        print(f'\nSpatial holdout split (axis={split_axis}, cutoff={boundary_cutoff}A):')
    
    for i, phase in enumerate(ref_phases):
        name = phase['name']
        # Use raw_csv_path for coordinates (ML table may not have x,y,z)
        coord_path = phase.get('raw_csv_path', phase['csv_path'])
        
        # Load raw CSV with coordinates
        raw = pd.read_csv(coord_path)
        n_atoms = len(raw)
        
        if split_axis not in raw.columns:
            raise ValueError(f"Split axis '{split_axis}' not found in {coord_path}. Available: {[c for c in raw.columns if c in ['x','y','z']]}")
        
        axis_vals = raw[split_axis].values
        axis_median = np.median(axis_vals)
        
        left = axis_vals <= axis_median
        right = axis_vals > axis_median
        
        boundary = (axis_vals >= axis_median - boundary_cutoff) & \
                   (axis_vals <= axis_median + boundary_cutoff)
        
        clean_left = left & ~boundary
        clean_right = right & ~boundary
        
        # Map to descriptor rows (they're in the same order as raw CSV rows)
        # Descriptors are built from the same CSV, so row order matches
        # Find which rows in X_all belong to this phase
        start_idx = (labels == i).argmax() if i > 0 else 0
        # Actually, labels are concatenated in order: phase 0 atoms, phase 1 atoms, etc.
        # We need to find the index range for this phase
        phase_rows = np.where(labels == i)[0]
        
        train_rows = phase_rows[clean_left]
        test_rows = phase_rows[clean_right]
        
        all_train_X.append(X_all[train_rows])
        all_train_y.append(np.full(len(train_rows), i))
        all_test_X.append(X_all[test_rows])
        all_test_y.append(np.full(len(test_rows), i))
        
        n_boundary = boundary.sum()
        total_boundary += n_boundary
        
        if verbose:
            print(f'  {name}: train={clean_left.sum()}, test={clean_right.sum()}, boundary={n_boundary}')
    
    X_train = np.concatenate(all_train_X)
    y_train = np.concatenate(all_train_y)
    X_test = np.concatenate(all_test_X)
    y_test = np.concatenate(all_test_y)
    
    if verbose:
        print(f'\nTotal: train={len(X_train)}, test={len(X_test)}, boundary={total_boundary}')
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'feature_cols': feature_cols,
        'total_boundary': total_boundary,
    }


def train_model(
    X_train, y_train, X_test, y_test,
    phase_names: list,
    model_config: dict,
    verbose: bool = True
) -> dict:
    """
    Train a classifier with spatial holdout validation.
    Returns model and performance metrics.
    """
    n_classes = len(phase_names)
    
    if verbose:
        print(f'\nTraining {model_config.get("type", "HistGradientBoostingClassifier")} '
              f'on {len(X_train)} samples, {n_classes} classes')
    
    # Build model from config
    model = HistGradientBoostingClassifier(
        random_state=model_config.get('random_state', 42),
        early_stopping=model_config.get('early_stopping', True),
        validation_fraction=model_config.get('validation_fraction', 0.1),
        n_iter_no_change=model_config.get('n_iter_no_change', 10),
        max_iter=model_config.get('max_iter', 500),
        learning_rate=model_config.get('learning_rate', 0.05),
        max_leaf_nodes=model_config.get('max_leaf_nodes', 31),
        min_samples_leaf=model_config.get('min_samples_leaf', 20),
    )
    
    # CV within training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
        if verbose:
            print(f'CV macro avg F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')
    except Exception as e:
        if verbose:
            print(f'CV failed (likely too few samples per class): {e}')
        cv_scores = np.array([np.nan])
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Test on spatial holdout
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    if verbose:
        print(f'Test accuracy (spatial holdout): {test_acc:.4f}')
        print(f'\nPer-class F1:')
        report = classification_report(y_test, y_pred, target_names=phase_names,
                                        output_dict=True)
        for name in phase_names:
            print(f'  {name}: {report[name]["f1-score"]:.4f}')
        
        print(f'\nConfusion matrix:')
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=[f'True {n}' for n in phase_names],
                             columns=[f'Pred {n}' for n in phase_names])
        print(cm_df)
    
    # Permutation importance
    if verbose:
        print(f'\nTop-10 permutation importance:')
    try:
        r = permutation_importance(model, X_test, y_test, n_repeats=10,
                                    random_state=42, n_jobs=4)
        sorted_idx = r.importances_mean.argsort()[::-1]
        feature_cols = []  # will be filled by caller
        top_features = []
        if verbose:
            print(f'{"Rank":>5s} {"Feature":45s} {"Importance":>10s}')
            print('-' * 62)
        for rank, i in enumerate(sorted_idx[:10], 1):
            if verbose:
                print(f'{rank:5d} {"feature_"+str(i):45s} {r.importances_mean[i]:10.4f}')
            top_features.append((i, r.importances_mean[i]))
    except Exception as e:
        if verbose:
            print(f'  Permutation importance skipped: {e}')
        r = None
        sorted_idx = None
    
    return {
        'model': model,
        'test_accuracy': test_acc,
        'cv_scores': cv_scores,
        'y_pred': y_pred,
        'y_test': y_test,
        'permutation_importance': r,
        'sorted_idx': sorted_idx,
    }


# =============================================================================
# Part 3: Inference on target structure
# =============================================================================

def predict_target(
    model,
    feature_cols: list,
    target_csv: str,
    phase_names: list,
    config: dict,
    verbose: bool = True,
    norm_params: dict = None
) -> dict:
    """
    Apply trained model to a target structure.
    
    Parameters
    ----------
    norm_params : dict, optional
        Normalization parameters (mean, std) for energy/stress features,
        computed from the COMBINED training data. If provided, these are
        applied to the target instead of computing self-referential statistics
        from the target file itself.
    """
    # Build descriptors for target using the real material-agnostic normalization layer
    # (per-structure R1_SG-based length scaling, per-structure z-score energy/stress scaling)
    feature_whitelist = config.get('feature_whitelist', None)
    desc = build_general_descriptors(
        target_csv, structure_id='target', verbose=verbose,
        feature_whitelist=feature_whitelist, norm_params=norm_params)
    
    # Load raw CSV for OOD check and coordinates
    raw = pd.read_csv(target_csv)
    
    # Ensure all training feature columns exist
    for c in feature_cols:
        if c not in desc.columns:
            if verbose:
                print(f'  WARNING: Feature "{c}" missing from target. Filling NaN.')
            desc[c] = np.nan
    
    X_target = desc[feature_cols].values
    
    # Predict
    preds = model.predict(X_target)
    probs = model.predict_proba(X_target)
    confidence = np.max(probs, axis=1)
    
    # OOD warning: check if target elements overlap with reference elements
    # This is a soft check -- we can detect if type labels are present
    if 'type' in raw.columns:
        target_types = sorted(raw['type'].unique())
        ref_types = set()
        for phase in config['reference_phases']:
            ref_csv = pd.read_csv(phase['csv_path'])
            ref_types.update(ref_csv['type'].unique())
        
        if not set(target_types).issubset(ref_types):
            if verbose:
                print(f'\n*** OOD WARNING ***')
                print(f'Target structure contains atom types {target_types}, '
                      f'but references only contain types {sorted(ref_types)}.')
                print(f'Predictions will be made but should be treated cautiously.')
    
    # Phase fractions at multiple thresholds
    thresholds = config['inference'].get('multiple_thresholds', [0.50, 0.60, 0.70, 0.80, 0.90])
    
    if verbose:
        print(f'\nPredictions on target ({len(preds)} atoms):')
        print(f'{"Thresh":>7s}', end='')
        for name in phase_names:
            print(f' {name:>25s}', end='')
        print(f' {"Uncertain":>12s}')
    
    results_by_threshold = {}
    for thresh in thresholds:
        phase = np.where(confidence >= thresh, preds, -1)
        total = len(phase)
        row = {'threshold': thresh}
        parts = []
        for ci in range(len(phase_names)):
            mask = phase == ci
            frac = mask.sum() / total * 100
            row[f'class_{ci}_fraction'] = frac
            row[f'class_{ci}_count'] = mask.sum()
            parts.append(f'{mask.sum():6d} ({frac:5.1f}%)')
        
        uncert = (phase == -1).sum()
        row['uncertain_count'] = uncert
        row['uncertain_fraction'] = uncert / total * 100
        parts.append(f'{uncert:6d} ({uncert/total*100:5.1f}%)')
        
        if verbose:
            print(f'{thresh:7.2f}  {"  ".join(parts)}')
        
        results_by_threshold[thresh] = row
    
    return {
        'preds': preds,
        'probs': probs,
        'confidence': confidence,
        'results_by_threshold': results_by_threshold,
        'desc': desc,
    }


# =============================================================================
# Part 5: OVITO export
# =============================================================================

def export_ovito(
    preds: np.ndarray,
    confidence: np.ndarray,
    phase_names: list,
    target_csv: str,
    config: dict,
    output_dir: str,
    verbose: bool = True
) -> dict:
    """
    Export predictions in OVITO-compatible format:
    1. Extended XYZ file with positions, phase label (int), and confidence
    2. Phase mapping CSV (integer -> phase name)
    
    Uses coordinates from the raw post-processing CSV (which has x, y, z columns)
    OR from a LAMMPS dump file if specified in config.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get atom positions from raw CSV
    raw = pd.read_csv(target_csv)
    
    if all(c in raw.columns for c in ['x', 'y', 'z']):
        x, y, z = raw['x'].values, raw['y'].values, raw['z'].values
    else:
        # Try loading from dump file
        dump_path = config['target_structure'].get('dump_path')
        if dump_path and os.path.exists(dump_path):
            if verbose:
                print(f'Loading coordinates from LAMMPS dump: {dump_path}')
            with open(dump_path) as f:
                lines = f.readlines()
            cols = lines[8].strip().replace('ITEM: ATOMS', '').strip().split()
            dump_df = pd.DataFrame([l.strip().split() for l in lines[9:] if l.strip()],
                                    columns=cols)
            dump_df['id'] = dump_df['id'].astype(int)
            # Sort by id to match feature order
            dump_df = dump_df.sort_values('id').reset_index(drop=True)
            x = dump_df['x'].astype(float).values
            y = dump_df['y'].astype(float).values
            z = dump_df['z'].astype(float).values
        else:
            # Fallback: use dummy coordinates
            if verbose:
                print('WARNING: No coordinates available. Using dummy (index-based) positions.')
            x = np.arange(len(preds))
            y = np.zeros(len(preds))
            z = np.zeros(len(preds))
    
    # Write extended XYZ with actual atom types preserved
    xyz_path = os.path.join(output_dir, 'predicted_phases.xyz')
    n_atoms = len(preds)
    
    # Get atom types from raw CSV (preserve actual element identities)
    type_to_element = {1: 'Cu', 2: 'Zr', 3: 'Al'}  # generic map
    if 'type' in raw.columns:
        elements = [type_to_element.get(t, 'X') for t in raw['type'].values]
    else:
        elements = ['A'] * n_atoms
    
    with open(xyz_path, 'w') as f:
        f.write(f'{n_atoms}\n')
        f.write(f'Lattice="1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0" '
                f'Properties=species:S:1:pos:R:3:phase:I:1:confidence:R:1\n')
        for i in range(n_atoms):
            # Preserve actual atom type (Cu/Zr) from the raw data
            f.write(f'{elements[i]} {x[i]:.6f} {y[i]:.6f} {z[i]:.6f} {preds[i]} {confidence[i]:.6f}\n')
    
    # Write phase mapping
    mapping_path = os.path.join(output_dir, 'phase_mapping.csv')
    with open(mapping_path, 'w') as f:
        f.write('phase_index,phase_name\n')
        for i, name in enumerate(phase_names):
            f.write(f'{i},{name}\n')
    
    if verbose:
        print(f'\nOVITO export:')
        print(f'  XYZ file: {os.path.abspath(xyz_path)} ({n_atoms} atoms)')
        print(f'  Columns: species, x, y, z, phase (integer), confidence')
        print(f'  Phase mapping: {os.path.abspath(mapping_path)}')
        print(f'    To color by phase in OVITO:')
        print(f'    Load {xyz_path} -> select "phase" column')
    
    return {'xyz_path': xyz_path, 'mapping_path': mapping_path}


# =============================================================================
# Main pipeline
# =============================================================================

def run(config_path: str, verbose: bool = True):
    """
    Run the complete generalized reference-phase classification pipeline.
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if verbose:
        print('=' * 70)
        print('GENERALIZED REFERENCE-PHASE CLASSIFICATION FRAMEWORK')
        print('=' * 70)
        print(f'Config: {config_path}')
        print(f'Reference phases ({len(config["reference_phases"])}):')
        for p in config['reference_phases']:
            print(f'  - {p["name"]}: {p["csv_path"]}')
        print(f'Target: {config["target_structure"]["csv_path"]}')
    
    output_dir = config.get('output_dir', 'general_phase_id_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # =====================================================================
    # Part 1: Load and normalize reference phases
    # =====================================================================
    if verbose:
        print('\n' + '=' * 70)
        print('PART 1: Loading reference phases with per-structure normalization')
        print('=' * 70)
    
    descriptors, phase_names, labels, norm_params, feature_ranges = load_reference_phases(config, verbose=verbose)
    feature_cols = get_feature_columns(descriptors)
    n_classes = len(phase_names)
    
    if verbose:
        print(f'\nFeature set: {len(feature_cols)} normalized, unit-free features')
        print(f'No hardcoded class count: detected N={n_classes} classes from config')
    
    # =====================================================================
    # Part 2: Spatial holdout + training
    # =====================================================================
    if verbose:
        print('\n' + '=' * 70)
        print('PART 2: Spatial holdout validation and training')
        print('=' * 70)
    
    split = spatial_holdout_split(descriptors, labels, phase_names, config, verbose=verbose)
    
    result = train_model(
        split['X_train'], split['y_train'],
        split['X_test'], split['y_test'],
        phase_names, config.get('model', {}),
        verbose=verbose
    )
    result['feature_cols'] = split['feature_cols']
    
    # Save model and config for provenance
    model_path = os.path.join(output_dir, 'model.pkl')
    
    # Save training metadata alongside model so inference can verify consistency
    # CRITICAL: store the EXACT feature column ORDER, not just feature names.
    # Tree-based models index features by POSITION, not name.
    training_meta = {
        'model_type': result['model'].__class__.__name__,
        'n_features': result['model'].n_features_in_,
        'n_classes': len(phase_names),
        'class_names': phase_names,
        'mro_depth': config.get('model', {}).get('mro_depth', 3),
        'feature_whitelist': config.get('feature_whitelist', None),
        'feature_order': split['feature_cols'],  # EXACT column order used for X
    }
    joblib.dump({'model': result['model'], 'training_meta': training_meta}, model_path)
    
    # Save config copy
    import shutil
    config_copy_path = os.path.join(output_dir, 'config_used.yaml')
    shutil.copy2(config_path, config_copy_path)
    
    if verbose:
        print(f'\nModel saved to {model_path}')
        print(f'Config saved to {config_copy_path}')
    
    # =====================================================================
    # Part 3: Predict on target structure
    # =====================================================================
    if verbose:
        print('\n' + '=' * 70)
        print('PART 3: Inference on target structure')
        print('=' * 70)
    
    target_csv = config['target_structure']['csv_path']
    pred_result = predict_target(
        result['model'], split['feature_cols'],
        target_csv, phase_names, config, verbose=verbose,
        norm_params=norm_params
    )
    
    # OOD check: compare target raw feature ranges against reference ranges
    if verbose and norm_params:
        target_raw = pd.read_csv(target_csv)
        ood_triggered = False
        for col in norm_params:
            if col in target_raw.columns:
                t_min = target_raw[col].min()
                t_max = target_raw[col].max()
                r_min = feature_ranges[col]['min']
                r_max = feature_ranges[col]['max']
                margin = 0.2
                if t_min < r_min - margin * abs(r_min) or t_max > r_max + margin * abs(r_max):
                    if not ood_triggered:
                        print(f'\n*** OOD WARNING: Feature range mismatch ***')
                        ood_triggered = True
                    print(f'  {col}: target=[{t_min:.2f}, {t_max:.2f}], ref=[{r_min:.2f}, {r_max:.2f}]')
        if ood_triggered:
            print(f'  Target feature ranges extend beyond reference training ranges.')
            print(f'  Predictions may be unreliable for this chemistry.')
    
    # =====================================================================
    # Part 5: Export for OVITO
    # =====================================================================
    if verbose:
        print('\n' + '=' * 70)
        print('PART 5: OVITO export')
        print('=' * 70)
    
    export_ovito(
        pred_result['preds'],
        pred_result['confidence'],
        phase_names,
        target_csv,
        config,
        output_dir,
        verbose=verbose
    )
    
    # Save full prediction results as CSV
    raw = pd.read_csv(target_csv)
    out_df = raw.copy()
    out_df['predicted_phase_index'] = pred_result['preds']
    phase_map = {i: name for i, name in enumerate(phase_names)}
    out_df['predicted_phase_name'] = [phase_map[p] for p in pred_result['preds']]
    out_df['confidence'] = pred_result['confidence']
    
    results_csv = os.path.join(output_dir, 'predictions.csv')
    out_df.to_csv(results_csv, index=False)
    if verbose:
        print(f'\nFull predictions saved to {results_csv}')
    
    # =====================================================================
    # Summary
    # =====================================================================
    if verbose:
        print('\n' + '=' * 70)
        print('SUMMARY')
        print('=' * 70)
        print(f'Reference phases: {n_classes}')
        for i, name in enumerate(phase_names):
            print(f'  [{i}] {name}')
        print(f'Spatial holdout test accuracy: {result["test_accuracy"]:.4f}')
        print(f'CV macro F1: {np.nanmean(result["cv_scores"]):.4f}')
        print(f'Target predictions saved to: {output_dir}/')
        print(f'OVITO-ready XYZ: {output_dir}/predicted_phases.xyz')
        
        # Print threshold-0.50 summary
        r = pred_result['results_by_threshold'][0.50]
        print(f'\nPhase fractions at threshold 0.50:')
        for ci in range(n_classes):
            print(f'  {phase_names[ci]}: {r[f"class_{ci}_fraction"]:.1f}%')
        print(f'  Uncertain: {r["uncertain_fraction"]:.1f}%')
    
    return result, pred_result


def load_model_with_metadata(model_path: str, verbose: bool = True) -> tuple:
    """
    Load model and check training metadata for consistency.
    
    Returns
    -------
    (model, training_meta)
        model: the trained classifier
        training_meta: dict of training-time configuration
    """
    data = joblib.load(model_path)
    
    # Handle both old format (just model) and new format (dict with metadata)
    if isinstance(data, dict) and 'model' in data and 'training_meta' in data:
        model = data['model']
        training_meta = data['training_meta']
        if verbose:
            print(f'  Model: {training_meta["model_type"]}')
            print(f'  Features: {training_meta["n_features"]}')
            print(f'  Classes: {training_meta["n_classes"]} ({training_meta["class_names"]})')
            print(f'  Training MRO_DEPTH: {training_meta.get("mro_depth", "unknown")}')
        return model, training_meta
    else:
        # Legacy format: just the model object
        if verbose:
            print(f'  Model: {type(data).__name__}')
            print(f'  Note: no training metadata found (legacy format)')
        return data, {}


def check_inference_consistency(training_meta: dict, inference_config: dict, 
                                 inference_feature_order: list = None,
                                 verbose: bool = True) -> list:
    """
    Check that inference settings match training-time settings.
    
    Parameters
    ----------
    inference_feature_order : list, optional
        The column order produced by build_general_descriptors at inference time.
        If provided, checked against training feature_order to catch positional 
        index mismatches (tree models index features by POSITION, not name).
    
    Returns list of warning strings (empty if all consistent).
    """
    warnings = []
    
    if not training_meta:
        return warnings  # Can't check without metadata
    
    # Check MRO_DEPTH
    train_mro = training_meta.get('mro_depth', None)
    inf_mro = inference_config.get('model', {}).get('mro_depth', None)
    if train_mro is not None and inf_mro is not None and train_mro != inf_mro:
        msg = (f'MRO_DEPTH MISMATCH: model trained with MRO_DEPTH={train_mro}, '
               f'inference config has MRO_DEPTH={inf_mro}. '
               f'This will silently produce wrong predictions!')
        warnings.append(msg)
        if verbose:
            print(f'\n*** WARNING: {msg} ***\n')
    
    # Check feature whitelist
    train_whitelist = training_meta.get('feature_whitelist', None)
    inf_whitelist = inference_config.get('feature_whitelist', None)
    if train_whitelist is not None and inf_whitelist is not None:
        if set(train_whitelist) != set(inf_whitelist):
            msg = 'WHITELIST MISMATCH: inference whitelist differs from training whitelist'
            warnings.append(msg)
            if verbose:
                print(f'\n*** WARNING: {msg} ***\n')
                print(f'  Training: {train_whitelist}')
                print(f'  Inference: {inf_whitelist}')
    
    # Check feature ORDER (CRITICAL: tree models index features by POSITION)
    train_order = training_meta.get('feature_order', None)
    if train_order is not None and inference_feature_order is not None:
        if list(train_order) != list(inference_feature_order):
            if set(train_order) == set(inference_feature_order):
                msg = (f'FEATURE ORDER MISMATCH: training used order {list(train_order)}, '
                       f'inference produced order {list(inference_feature_order)}. '
                       f'Features are the same but in DIFFERENT POSITIONS. '
                       f'Tree-based models use positional indexing — this produces '
                       f'confidently WRONG predictions with no error!')
            else:
                msg = (f'FEATURE SET MISMATCH: training had {len(train_order)} features, '
                       f'inference produced {len(inference_feature_order)} features.')
            warnings.append(msg)
            if verbose:
                print(f'\n*** CRITICAL WARNING: {msg} ***\n')
    
    return warnings


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generalized reference-phase classification framework')
    parser.add_argument('--config', type=str,
                        default='Machine_Learning_Pipeline_for_Materials_Science/src/general_phase_id/ref_config.yaml',
                        help='Path to config YAML')
    args = parser.parse_args()
    
    run(args.config)


if __name__ == '__main__':
    main()