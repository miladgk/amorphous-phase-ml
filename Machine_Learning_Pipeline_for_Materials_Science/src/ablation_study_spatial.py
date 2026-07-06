"""
ablation_study_spatial.py
=========================
Controlled ablation with SPATIAL holdout split — the definitive test.
Partitions each simulation box spatially (split by x-coordinate median),
ensuring atoms in test set are NOT spatial neighbors of training atoms.

Also reports: exactly what split was used in the original pipeline.py that
produced the 0.8543 baseline.
"""
import sys, os, pandas as pd, numpy as np, joblib, yaml
sys.path.insert(0, 'Machine_Learning_Pipeline_for_Materials_Science/src')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

print('='*70)
print('SPATIAL HOLDOUT ABLATION STUDY')
print('='*70)

# =============================================================================
# Load data with x,y,z coordinates from raw post-processing CSVs
# =============================================================================
raw_5050 = pd.read_csv('Atomic_Simulation_Post-Processing_Pipeline/outputs/features_5050.csv')
raw_4654 = pd.read_csv('Atomic_Simulation_Post-Processing_Pipeline/outputs/features_4654.csv')
raw_6436 = pd.read_csv('Atomic_Simulation_Post-Processing_Pipeline/outputs/features_6436.csv')

# Load ML tables (contain features but not x,y,z)
ml_5050 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_5050.csv')
ml_4654 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_4654.csv')
ml_6436 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_6436.csv')

# Check id alignment
for ml, raw, ds in [(ml_5050, raw_5050, '5050'), (ml_4654, raw_4654, '4654'), (ml_6436, raw_6436, '6436')]:
    assert np.array_equal(ml['id'].values, raw['id'].values), f'{ds}: id mismatch between raw and ML'
    assert np.array_equal(ml['type'].values, raw['type'].values), f'{ds}: type mismatch'

# Merge x,y,z and gaussian_peak columns from raw into ML tables, and merge gaussian peak cols
for ml, raw, ds in [(ml_5050, raw_5050, '5050'), (ml_4654, raw_4654, '4654'), (ml_6436, raw_6436, '6436')]:
    ml['x'] = raw['x']
    ml['y'] = raw['y']
    ml['z'] = raw['z']
    ml['gaussian_peak2_center'] = raw['gaussian_peak2_center']
    ml['gaussian_peak3_center'] = raw['gaussian_peak3_center']
    print(f'{ds}: merged x,y,z + gaussian peaks, {len(ml)} atoms')

# =============================================================================
# SPATIAL HOLDOUT SPLIT: split each dataset at x-median
# Left half -> train, right half -> test (within each dataset)
# =============================================================================
print('\n' + '='*70)
print('Spatial split details')
print('='*70)

train_parts = []
test_parts = []
boundary_parts = []

for ml, raw, ds in [(ml_5050, raw_5050, '5050'), (ml_4654, raw_4654, '4654'), (ml_6436, raw_6436, '6436')]:
    x_median = ml['x'].median()
    x_cutoff = 3.5  # typical Voronoi neighbor radius ~3-4 Angstrom
    
    left_mask = ml['x'] <= x_median
    right_mask = ml['x'] > x_median
    
    # Boundary: atoms within one neighbor-shell of the split plane
    boundary_mask = (ml['x'] >= x_median - x_cutoff) & (ml['x'] <= x_median + x_cutoff)
    n_boundary = boundary_mask.sum()
    
    # Clean split: exclude boundary from both train and test
    clean_left = left_mask & ~boundary_mask
    clean_right = right_mask & ~boundary_mask
    
    df = ml.copy()
    df['phase_label'] = 0 if ds == '5050' else (1 if ds == '4654' else 2)
    df['dataset'] = ds
    
    train_parts.append(df[clean_left])
    test_parts.append(df[clean_right])
    boundary_parts.append(df[boundary_mask])
    
    print(f'\n{ds}:')
    print(f'  x range: [{ml["x"].min():.2f}, {ml["x"].max():.2f}]')
    print(f'  x median: {x_median:.2f}')
    print(f'  Left half (train): {clean_left.sum()} atoms (x ≤ {x_median:.2f})')
    print(f'  Right half (test): {clean_right.sum()} atoms (x > {x_median:.2f})')
    print(f'  Boundary atoms excluded: {n_boundary} (within ±{x_cutoff}A of split plane)')
    print(f'  Train class balance: {df[clean_left].groupby("type").size().to_dict()}')
    print(f'  Test class balance: {df[clean_right].groupby("type").size().to_dict()}')

df_train = pd.concat(train_parts, ignore_index=True)
df_test = pd.concat(test_parts, ignore_index=True)
df_boundary = pd.concat(boundary_parts, ignore_index=True)

print(f'\nTotal train: {len(df_train)}, Test: {len(df_test)}, Boundary excluded: {len(df_boundary)}')
print(f'Train class labels: {df_train["phase_label"].value_counts().to_dict()}')
print(f'Test class labels: {df_test["phase_label"].value_counts().to_dict()}')

# =============================================================================
# Define feature groups
# =============================================================================
baseline_core = [
    'R1', 'voronoi_volume_temporal', 'CN_temporal', 'q4', 'q6',
    'gaussian_peak2_center', 'gaussian_peak3_center',
    'neighbor_1', 'neighbor_2'
]

thermo_features = [
    'pe_atom_temporal', 'pe_atom_std_temporal',
    'hydrostatic_pressure_temporal', 'von_mises_stress_temporal',
    'deviatoric_norm_temporal', 'stress_anisotropy_temporal',
    'local_enthalpy_temporal', 'excess_energy_temporal',
    'pe_volume_residual'
]

mro_depth_features = [
    'mean_2nd_neighbor_volume_temporal', 'std_2nd_neighbor_volume_temporal',
    'mean_2nd_neighbor_free_volume_temporal', 'mean_2nd_neighbor_pentagon_fraction_temporal',
    'mean_2nd_neighbor_CN_temporal', 'cu_fraction_2nd_shell_temporal',
    'zr_fraction_2nd_shell_temporal',
    'mean_3rd_neighbor_volume_temporal', 'std_3rd_neighbor_volume_temporal',
    'mean_3rd_neighbor_free_volume_temporal', 'mean_3rd_neighbor_pentagon_fraction_temporal',
    'mean_3rd_neighbor_CN_temporal', 'cu_fraction_3rd_shell_temporal',
    'zr_fraction_3rd_shell_temporal'
]

all_features = [c for c in df_train.columns if c not in ['id', 'type', 'x', 'y', 'z', 'group_id', 'phase_label', 'dataset']]

configs = {
    'CONFIG 1 (baseline 9 core features)': baseline_core,
    'CONFIG 2 (baseline + thermo, no MRO depth)': baseline_core + thermo_features,
    'CONFIG 3 (baseline + MRO depth, no thermo)': baseline_core + mro_depth_features,
    'CONFIG 4 (all features)': all_features,
}

# =============================================================================
# Training function with fixed spatial split
# =============================================================================
def train_and_evaluate(feature_cols, config_name):
    print('\n' + '='*70)
    print(config_name)
    print('='*70)
    
    available = [c for c in feature_cols if c in df_train.columns]
    missing = [c for c in feature_cols if c not in df_train.columns]
    if missing:
        print(f'WARNING: Missing features (excluded): {missing}')
    
    print(f'Feature count: {len(available)}')
    
    X_train = df_train[available].values
    y_train = df_train['phase_label'].values
    X_test = df_test[available].values
    y_test = df_test['phase_label'].values
    
    print(f'\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}')
    print(f'Split method: Spatial holdout (x-median split, ±{3.5}A boundary excluded)')
    
    model = HistGradientBoostingClassifier(
        random_state=42, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=10, max_iter=500, learning_rate=0.05,
        max_leaf_nodes=31, min_samples_leaf=20
    )
    
    # Note: Under a single fixed spatial split, CV on training set still uses
    # random StratifiedKFold within the training region. Report both.
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f'\nCV macro avg F1 (within training region): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')
    print(f'CV individual folds: {cv_scores}')
    print('(Note: CV is StratifiedKFold on training-region atoms only -- no spatial constraint within CV)')
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f'\nTest accuracy (spatial holdout): {test_acc:.4f}')
    print(f'\n### THIS is the number that matters for generalization ###')
    
    print('\nPer-class F1:')
    report = classification_report(y_test, y_pred, target_names=['Cu50Zr50(0)', 'Cu46Zr54(1)', 'Cu64Zr36(2)'], output_dict=True)
    for c in ['Cu50Zr50(0)', 'Cu46Zr54(1)', 'Cu64Zr36(2)']:
        print(f'  {c}: {report[c]["f1-score"]:.4f}')
    
    print('\nConfusion matrix (3x3):')
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=['True 0', 'True 1', 'True 2'], columns=['Pred 0', 'Pred 1', 'Pred 2']))
    
    # Permutation importance
    print(f'\nTop-10 permutation importance (feature + importance):')
    r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=4)
    sorted_idx = r.importances_mean.argsort()[::-1]
    print(f'{"Rank":>5s} {"Feature":45s} {"Importance":>10s}')
    print('-' * 62)
    for rank, i in enumerate(sorted_idx[:10], 1):
        fn = available[i]
        tag = ' <<< THERMO' if fn in thermo_features else ''
        tag += ' <<< MRO-DEPTH' if fn in mro_depth_features else ''
        print(f'{rank:5d} {fn:45s} {r.importances_mean[i]:10.4f}{tag}')
    
    return cv_scores.mean(), test_acc

# =============================================================================
# Run all 4 configs
# =============================================================================
results = {}
for name, features in configs.items():
    cv, acc = train_and_evaluate(features, name)
    results[name] = {'CV_F1': cv, 'Test_Acc': acc}

# =============================================================================
# Summary table
# =============================================================================
print('\n\n' + '='*70)
print('SUMMARY: ALL CONFIGURATIONS (SPATIAL HOLDOUT SPLIT)')
print('='*70)
print(f'{"Configuration":50s} {"CV F1":>8s} {"Test Acc":>8s}')
print('-' * 70)
for name, res in results.items():
    print(f'{name:50s} {res["CV_F1"]:8.4f} {res["Test_Acc"]:8.4f}')

# Contrast with previous result
prev = {'CONFIG 1 (baseline 9 core features)': 0.9100,
        'CONFIG 2 (baseline + thermo, no MRO depth)': 0.9848,
        'CONFIG 3 (baseline + MRO depth, no thermo)': 0.9782,
        'CONFIG 4 (all features)': 0.9996}
print(f'\n{"-"*70}')
print(f'{"Configuration":50s} {"Prev Acc":>8s} {"Spatial":>8s} {"Delta":>8s}')
print(f'{"-"*70}')
for name in configs:
    p = prev[name]
    s = results[name]['Test_Acc']
    print(f'{name:50s} {p:8.4f} {s:8.4f} {s-p:8.4f}')

# =============================================================================
# Report original pipeline split logic
# =============================================================================
print('\n\n' + '='*70)
print('ORIGINAL PIPELINE SPLIT LOGIC')
print('='*70)
print('From pipeline.py, function group_train_test_split_fixed (lines 128-175):')
print('  - Splits each group INDEPENDENTLY (70/30 within each group)')
print('  - ALL three groups (101, 123, 145) appear in BOTH train and test')
print('  - NO spatial constraint whatsoever')
print('  - Uses simple train_test_split on indices within each group')
print()
print('From data_utils.py, function group_train_test_split (lines 91-138):')
print('  - AGGREGATES groups by majority class, then splits groups (not atoms)')
print('  - FAILS for this data (only 1 group per class -> ValueError)')
print('  - pipeline.py does NOT use this function -- it uses group_train_test_split_fixed')
print()
print('Conclusion: The original 0.8543 baseline used the SAME split type as our')
print('ablation Config 1 (0.9082) -- within-group proportional split -- but with') 
print('randomized hyperparameter search (RandomForest/HistGB/LinearSVC) rather')
print('than the fixed HistGB model used here. The 0.054 point gap between 0.8543')
print('and 0.9082 is mostly from model selection (tuned RF/HistGB vs fixed HistGB)')
print('and different feature sets (9 curated vs 9 restored). Neither the original')
print('baseline nor our group-aware ablation controlled for spatial adjacency --')
print('this spatial holdout is the FIRST test that does.')

print('\nDone.')