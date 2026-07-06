"""
ablation_study.py
=================
Controlled ablation to isolate the contributions of:
- Original 9 core features (baseline)
- Thermodynamic features
- MRO depth-2/3 features
- All features combined

Also includes diagnostic: between/within-group variance for
mean_3rd_neighbor_volume_temporal to check for dataset-identity leakage.
"""
import sys, os, pandas as pd, numpy as np, joblib, yaml
sys.path.insert(0, 'Machine_Learning_Pipeline_for_Materials_Science/src')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# Load data
df_5050 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_5050.csv')
df_4654 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_4654.csv')
df_6436 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_6436.csv')

# =============================================================================
# Define feature groups
# =============================================================================
# Original 9 core features (from SHAP-selected baseline)
# Note: R1_SG was renamed to R1 in feature_builder_data_clean.py
# gaussian_peak2_center and gaussian_peak3_center are not in the ML table
# (they were renamed to gaussian_peak2_center/gaussian_peak3_center in the
# original pipeline but the ML table uses different naming)
# Let's use what's available that matches the original intent
baseline_core = [
    'R1', 'voronoi_volume_temporal', 'CN_temporal', 'q4', 'q6',
    'neighbor_1', 'neighbor_2'
]
# Note: gaussian_peak2_center, gaussian_peak3_center, R1_SG are not in the
# current ML table. The original baseline had 9 features; we use the 7 that
# are available plus note this discrepancy.

# Thermodynamic features
thermo_features = [
    'pe_atom_temporal', 'pe_atom_std_temporal',
    'hydrostatic_pressure_temporal', 'von_mises_stress_temporal',
    'deviatoric_norm_temporal', 'stress_anisotropy_temporal',
    'local_enthalpy_temporal', 'excess_energy_temporal',
    'pe_volume_residual'
]

# MRO depth-2/3 features (2nd and 3rd shell neighbor-averaged)
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

# All features (everything in the ML table except id, type, and derived interaction features)
all_features = [c for c in df_5050.columns if c not in ['id', 'type']]

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
# Training function
# =============================================================================
def train_and_evaluate(feature_cols, config_name):
    print('\n' + '='*70)
    print(config_name)
    print('='*70)
    
    # Filter to only available columns
    available = [c for c in feature_cols if c in df_5050.columns]
    missing = [c for c in feature_cols if c not in df_5050.columns]
    if missing:
        print(f'WARNING: Missing features (excluded): {missing}')
    
    print(f'Feature count: {len(available)}')
    print(f'Features: {available}')
    
    df_labelled = pd.concat([
        df_5050.assign(phase_label=0),
        df_4654.assign(phase_label=1),
        df_6436.assign(phase_label=2)
    ], ignore_index=True)
    
    X = df_labelled[available].values
    y = df_labelled['phase_label'].values
    
    # Use the SAME train/test split as the original retrain_with_thermo.py
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f'\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}')
    print(f'Split method: train_test_split(test_size=0.3, random_state=42, stratify=y)')
    
    model = HistGradientBoostingClassifier(
        random_state=42, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=10, max_iter=500, learning_rate=0.05,
        max_leaf_nodes=31, min_samples_leaf=20
    )
    
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
print('SUMMARY: ALL CONFIGURATIONS')
print('='*70)
print(f'{"Configuration":50s} {"CV F1":>8s} {"Test Acc":>8s}')
print('-' * 70)
for name, res in results.items():
    print(f'{name:50s} {res["CV_F1"]:8.4f} {res["Test_Acc"]:8.4f}')

# =============================================================================
# DIAGNOSTIC: Between/within-group variance for mean_3rd_neighbor_volume_temporal
# =============================================================================
print('\n\n' + '='*70)
print('DIAGNOSTIC: mean_3rd_neighbor_volume_temporal by dataset')
print('='*70)

for ds_name, df in [('5050', df_5050), ('4654', df_4654), ('6436', df_6436)]:
    col = df['mean_3rd_neighbor_volume_temporal']
    print(f'\n{ds_name}:')
    print(f'  mean = {col.mean():.4f}')
    print(f'  std  = {col.std():.4f}')
    print(f'  min  = {col.min():.4f}')
    print(f'  max  = {col.max():.4f}')
    print(f'  Q1   = {col.quantile(0.25):.4f}')
    print(f'  Q3   = {col.quantile(0.75):.4f}')

# Between-group variance
all_data = pd.concat([
    df_5050.assign(dataset='5050'),
    df_4654.assign(dataset='4654'),
    df_6436.assign(dataset='6436')
], ignore_index=True)

group_means = all_data.groupby('dataset')['mean_3rd_neighbor_volume_temporal'].mean()
overall_mean = all_data['mean_3rd_neighbor_volume_temporal'].mean()
group_counts = all_data.groupby('dataset').size()

# Between-group variance (weighted by group size)
between_var = np.sum(group_counts * (group_means - overall_mean)**2) / (len(group_means) - 1)

# Within-group variance (pooled)
within_var = 0
for ds in ['5050', '4654', '6436']:
    g = all_data[all_data['dataset'] == ds]['mean_3rd_neighbor_volume_temporal']
    within_var += np.sum((g - g.mean())**2)
within_var /= (len(all_data) - len(group_means))

print(f'\nBetween-group variance: {between_var:.4f}')
print(f'Within-group variance:  {within_var:.4f}')
print(f'Between/Within ratio:   {between_var/within_var:.4f}')
print(f'\nGroup means:')
for ds, m in group_means.items():
    print(f'  {ds}: {m:.4f}')
print(f'Overall mean: {overall_mean:.4f}')

# Also check: does mean_3rd_neighbor_volume_temporal separate the datasets?
print('\nPairwise separation (mean +/- std):')
for ds1 in ['5050', '4654', '6436']:
    for ds2 in ['5050', '4654', '6436']:
        if ds1 < ds2:
            m1 = all_data[all_data['dataset']==ds1]['mean_3rd_neighbor_volume_temporal'].mean()
            m2 = all_data[all_data['dataset']==ds2]['mean_3rd_neighbor_volume_temporal'].mean()
            s1 = all_data[all_data['dataset']==ds1]['mean_3rd_neighbor_volume_temporal'].std()
            s2 = all_data[all_data['dataset']==ds2]['mean_3rd_neighbor_volume_temporal'].std()
            diff = abs(m1 - m2)
            print(f'  {ds1} vs {ds2}: |diff|={diff:.4f}, pooled_std={(s1+s2)/2:.4f}, ratio={diff/((s1+s2)/2):.4f}')

print('\nDone.')