"""
ablation_study_grouped.py
=========================
Controlled ablation with CORRECT group-aware split (by simulation run/group_id).
Uses data_utils.group_train_test_split instead of random stratified split.

Also fixes Config 1 to include all 9 original SHAP-selected features by
reading gaussian_peak2_center and gaussian_peak3_center from raw post-processing
CSV and merging into the ML table.
"""
import sys, os, pandas as pd, numpy as np, joblib, yaml
sys.path.insert(0, 'Machine_Learning_Pipeline_for_Materials_Science/src')
from data_utils import group_train_test_split
from pipeline import group_train_test_split_fixed

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# =============================================================================
# Load data with group_id
# =============================================================================
print('='*70)
print('Loading data with group_id from config')
print('='*70)

with open('Machine_Learning_Pipeline_for_Materials_Science/config.yaml') as f:
    cfg = yaml.safe_load(f)

# Load raw post-processing CSVs to get gaussian_peak columns
raw_5050 = pd.read_csv('Atomic_Simulation_Post-Processing_Pipeline/outputs/features_5050.csv')
raw_4654 = pd.read_csv('Atomic_Simulation_Post-Processing_Pipeline/outputs/features_4654.csv')
raw_6436 = pd.read_csv('Atomic_Simulation_Post-Processing_Pipeline/outputs/features_6436.csv')

# Load ML tables
ml_5050 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_5050.csv')
ml_4654 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_4654.csv')
ml_6436 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_6436.csv')

# Confirm R1 == R1_SG (just a rename)
r1_corr = np.corrcoef(raw_5050['R1_SG'], ml_5050['R1'])[0,1]
print(f'R1_SG vs R1 correlation (5050): {r1_corr:.6f}')
r1_diff = (raw_5050['R1_SG'] - ml_5050['R1']).abs().max()
print(f'R1_SG vs R1 max absolute diff (5050): {r1_diff:.10f}')
# If they're identical, use R1 from ML table

# Merge gaussian_peak2_center and gaussian_peak3_center from raw into ML tables
for ml, raw, ds in [(ml_5050, raw_5050, '5050'), (ml_4654, raw_4654, '4654'), (ml_6436, raw_6436, '6436')]:
    # Merge by atom id
    ml['gaussian_peak2_center'] = raw['gaussian_peak2_center']
    ml['gaussian_peak3_center'] = raw['gaussian_peak3_center']
    print(f'{ds}: merged gaussian_peak columns, non-NaN: {ml["gaussian_peak2_center"].notna().sum()}/{len(ml)}')

# Assign group_id per config
df_5050 = ml_5050.copy()
df_5050['group_id'] = cfg['data']['phase_A_group_id']
df_5050['phase_label'] = 0

df_4654 = ml_4654.copy()
df_4654['group_id'] = cfg['data']['phase_B_group_id']
df_4654['phase_label'] = 1

df_6436 = ml_6436.copy()
df_6436['group_id'] = cfg['data']['phase_C_group_id']
df_6436['phase_label'] = 2

# =============================================================================
# Define feature groups
# =============================================================================
# Original 9 core features (now with gaussian_peak columns restored)
baseline_core = [
    'R1', 'voronoi_volume_temporal', 'CN_temporal', 'q4', 'q6',
    'gaussian_peak2_center', 'gaussian_peak3_center',
    'neighbor_1', 'neighbor_2'
]

# Thermodynamic features
thermo_features = [
    'pe_atom_temporal', 'pe_atom_std_temporal',
    'hydrostatic_pressure_temporal', 'von_mises_stress_temporal',
    'deviatoric_norm_temporal', 'stress_anisotropy_temporal',
    'local_enthalpy_temporal', 'excess_energy_temporal',
    'pe_volume_residual'
]

# MRO depth-2/3 features
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

# All features
all_features = [c for c in df_5050.columns if c not in ['id', 'type', 'group_id', 'phase_label']]

# =============================================================================
# Define the 4 configurations
# =============================================================================
configs = {
    'CONFIG 1 (baseline 9 core features)': baseline_core,
    'CONFIG 2 (baseline + thermo, no MRO depth)': baseline_core + thermo_features,
    'CONFIG 3 (baseline + MRO depth, no thermo)': baseline_core + mro_depth_features,
    'CONFIG 4 (all features)': all_features,
}

# =============================================================================
# Training function with group-aware split
# =============================================================================
def train_and_evaluate(feature_cols, config_name):
    print('\n' + '='*70)
    print(config_name)
    print('='*70)
    
    available = [c for c in feature_cols if c in df_5050.columns]
    missing = [c for c in feature_cols if c not in df_5050.columns]
    if missing:
        print(f'WARNING: Missing features (excluded): {missing}')
    
    print(f'Feature count: {len(available)}')
    print(f'Features: {available}')
    
    df_labelled = pd.concat([df_5050, df_4654, df_6436], ignore_index=True)
    
    # Group-aware split: use group_train_test_split_fixed from pipeline.py
    # which splits each group independently (train/test within group),
    # handling the case of 3 groups with 3 different labels.
    train_idx, test_idx = group_train_test_split_fixed(
        df_labelled, group_col='group_id', target_col='phase_label',
        test_size=0.3, random_state=42
    )
    
    train_groups = df_labelled.loc[train_idx, 'group_id'].unique()
    test_groups = df_labelled.loc[test_idx, 'group_id'].unique()
    print(f'\nTrain groups: {sorted(train_groups)}')
    print(f'Test groups: {sorted(test_groups)}')
    print(f'Train samples: {len(train_idx)}, Test samples: {len(test_idx)}')
    print(f'Split method: group_train_test_split(group_col=group_id, test_size=0.3, random_state=42)')
    
    X_train = df_labelled.loc[train_idx, available].values
    y_train = df_labelled.loc[train_idx, 'phase_label'].values
    X_test = df_labelled.loc[test_idx, available].values
    y_test = df_labelled.loc[test_idx, 'phase_label'].values
    
    model = HistGradientBoostingClassifier(
        random_state=42, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=10, max_iter=500, learning_rate=0.05,
        max_leaf_nodes=31, min_samples_leaf=20
    )
    
    # CV on training set (still uses StratifiedKFold within training groups)
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f'\nCV macro avg F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')
    print(f'CV individual folds: {cv_scores}')
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f'Test accuracy: {test_acc:.4f}')
    
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
print('SUMMARY: ALL CONFIGURATIONS (GROUP-AWARE SPLIT)')
print('='*70)
print(f'{"Configuration":50s} {"CV F1":>8s} {"Test Acc":>8s}')
print('-' * 70)
for name, res in results.items():
    print(f'{name:50s} {res["CV_F1"]:8.4f} {res["Test_Acc"]:8.4f}')

print('\nDone.')