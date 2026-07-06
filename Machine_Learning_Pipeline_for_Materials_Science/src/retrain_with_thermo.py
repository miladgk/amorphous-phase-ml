"""
retrain_with_thermo.py
======================
Retrain 3-class model including thermodynamic features (pe_atom, stress invariants)
derived from per-atom potential energy and stress tensor parsed from LAMMPS dumps.

Steps:
1. Rebuild ML tables from post-processing pipeline outputs (now with thermo columns)
2. Load all four datasets (5050, 4654, 6436, polyamorphous)
3. Sanity check thermodynamic columns (Step 5)
4. Train 3-class HistGB model
5. Report CV F1, test accuracy, per-class F1, confusion matrix, feature importance
6. Apply to polyamorphous and report phase fractions + low-confidence atom analysis
"""
import sys, os, pandas as pd, numpy as np, joblib, yaml
sys.path.insert(0, 'Machine_Learning_Pipeline_for_Materials_Science/src')
from feature_builder_data_clean import build_ml_table

with open('Machine_Learning_Pipeline_for_Materials_Science/config.yaml') as f:
    cfg = yaml.safe_load(f)

# =============================================================================
# Step 1: Rebuild ML tables from post-processing pipeline outputs
# =============================================================================
# Rebuild ML tables for datasets that have thermodynamic data in raw post-processing output
# Polyamorphous raw dump files don't contain thermodynamic columns (no c_pe_atom etc),
# so we use the existing ML table (which has all columns filled with NaN for thermo).
print('='*70)
print('STEP 1: Rebuilding ML tables with thermodynamic features')
print('='*70)
for ds in ['5050', '4654', '6436']:
    in_path = f'Atomic_Simulation_Post-Processing_Pipeline/outputs/features_{ds}.csv'
    out_path = f'Machine_Learning_Pipeline_for_Materials_Science/data/features_{ds}.csv'
    df = build_ml_table(in_path, tolerance=0.2, atomic_radii=cfg.get('atomic_radii'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'{ds}: {len(df)} rows, {len(df.columns)} cols')
    # Check for thermo columns
    all_thermo_cols = ['pe_atom_temporal', 'pe_atom_std_temporal',
        'hydrostatic_pressure_temporal', 'von_mises_stress_temporal',
        'deviatoric_norm_temporal', 'stress_anisotropy_temporal',
        'local_enthalpy_temporal', 'excess_energy_temporal',
        'local_energy_fluctuation_temporal', 'mean_neighbor_pe_temporal',
        'pe_volume_residual']
    thermo_cols_present = [c for c in all_thermo_cols if c in df.columns]
    print(f'  Thermodynamic columns found: {len(thermo_cols_present)}/{len(all_thermo_cols)}')
    for tc in thermo_cols_present:
        non_nan = df[tc].notna().sum()
        print(f'    {tc}: {non_nan} non-NaN values')

df_5050 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_5050.csv')
df_4654 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_4654.csv')
df_6436 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_6436.csv')
df_poly = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_polyamorphous.csv')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# =============================================================================
# Step 5: Sanity check thermodynamic columns
# =============================================================================
print('\n' + '='*70)
print('STEP 5: Thermodynamic column sanity check')
print('='*70)

thermo_cols = ['pe_atom_temporal', 'pe_atom_std_temporal',
               'hydrostatic_pressure_temporal', 'von_mises_stress_temporal',
               'deviatoric_norm_temporal', 'stress_anisotropy_temporal',
               'local_enthalpy_temporal', 'excess_energy_temporal',
               'local_energy_fluctuation_temporal', 'mean_neighbor_pe_temporal',
               'pe_volume_residual']

for phase_name, df in [('5050', df_5050), ('4654', df_4654), ('6436', df_6436)]:
    print(f'\n=== {phase_name} ===')
    available = [c for c in thermo_cols if c in df.columns]
    if not available:
        print('  No thermodynamic columns found.')
        continue
    stats = df[available].describe().loc[['mean', 'std', 'min', 'max']]
    print(stats.round(4).to_string())

# Key overlap check: 5050 vs 4654 for potential energy
if 'pe_atom_temporal' in df_5050.columns and 'pe_atom_temporal' in df_4654.columns:
    pe_5050 = df_5050['pe_atom_temporal']
    pe_4654 = df_4654['pe_atom_temporal']
    q5 = pe_4654.quantile(0.05)
    q95 = pe_4654.quantile(0.95)
    overlap_pe = ((pe_5050 >= q5) & (pe_5050 <= q95)).mean()
    print(f'\nPotential energy overlap (5050 vs 4654): {overlap_pe:.3f}')
    print(f'5050 pe mean: {pe_5050.mean():.4f}, std: {pe_5050.std():.4f}')
    print(f'4654 pe mean: {pe_4654.mean():.4f}, std: {pe_4654.std():.4f}')
    print(f'Signal (difference in means): {abs(pe_5050.mean() - pe_4654.mean()):.4f} eV')
else:
    overlap_pe = np.nan
    print('\nPotential energy columns not available.')

# Von Mises overlap
if 'von_mises_stress_temporal' in df_5050.columns and 'von_mises_stress_temporal' in df_4654.columns:
    vm_5050 = df_5050['von_mises_stress_temporal']
    vm_4654 = df_4654['von_mises_stress_temporal']
    q5v = vm_4654.quantile(0.05)
    q95v = vm_4654.quantile(0.95)
    overlap_vm = ((vm_5050 >= q5v) & (vm_5050 <= q95v)).mean()
    print(f'\nVon Mises stress overlap (5050 vs 4654): {overlap_vm:.3f}')
    print(f'5050 vm mean: {vm_5050.mean():.4f}, std: {vm_5050.std():.4f}')
    print(f'4654 vm mean: {vm_4654.mean():.4f}, std: {vm_4654.std():.4f}')
    print(f'Signal: {abs(vm_5050.mean() - vm_4654.mean()):.4f} bar')
else:
    overlap_vm = np.nan
    print('\nVon Mises stress columns not available.')

if overlap_pe is not np.nan:
    if overlap_pe < 0.50:
        print('\n>>> LOW OVERLAP (<0.50): pe_atom_temporal ALONE provides significant separation!')
    elif overlap_pe < 0.70:
        print('\n>>> MODERATE OVERLAP (0.50-0.70): pe_atom_temporal provides some signal.')
    else:
        print('\n>>> HIGH OVERLAP (>0.70): Potential energy alone is not sufficient. Stuctural/thermo combination needed.')

# =============================================================================
# Step 6: Retrain 3-class model with ALL features including thermodynamic
# =============================================================================
print('\n' + '='*70)
print('STEP 6: 3-CLASS MODEL WITH THERMODYNAMIC FEATURES')
print('='*70)

# Use all features from config
feature_cols = [c for c in cfg['features'] if c in df_5050.columns]
print(f'Total features in config: {len(cfg["features"])}')
print(f'Features available in data: {len(feature_cols)}')
thermo_in_features = [c for c in thermo_cols if c in feature_cols]
print(f'Thermodynamic features included: {len(thermo_in_features)}')

df_labelled = pd.concat([
    df_5050.assign(phase_label=0),
    df_4654.assign(phase_label=1),
    df_6436.assign(phase_label=2)
], ignore_index=True)

print(f'\nSamples: {len(df_labelled)}, Features: {len(feature_cols)}')
print('Features:', feature_cols)

X = df_labelled[feature_cols].values
y = df_labelled['phase_label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = HistGradientBoostingClassifier(
    random_state=42, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=10, max_iter=500, learning_rate=0.05,
    max_leaf_nodes=31, min_samples_leaf=20
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
print(f'\nCV macro avg F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')

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
print('\nTop-10 permutation importance (feature + importance):')
r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=4)
sorted_idx = r.importances_mean.argsort()[::-1]
print(f'{"Rank":>5s} {"Feature":45s} {"Importance":>10s}')
print('-' * 62)
for rank, i in enumerate(sorted_idx[:10], 1):
    fn = feature_cols[i]
    tag = ' <<< THERMO' if fn in thermo_cols else ''
    print(f'{rank:5d} {fn:45s} {r.importances_mean[i]:10.4f}{tag}')

# Save model
joblib.dump(model, 'Machine_Learning_Pipeline_for_Materials_Science/outputs/models/model_3class_with_thermo.pkl')
print('\nModel saved to outputs/models/model_3class_with_thermo.pkl')

# =============================================================================
# Apply to polyamorphous
# =============================================================================
print('\n' + '='*70)
print('3-CLASS PREDICTION ON POLYAMORPHOUS (with thermo features)')
print('='*70)

# Polyamorphous data doesn't have thermodynamic columns (raw dump files lack them)
# Add missing columns as NaN
for c in feature_cols:
    if c not in df_poly.columns:
        df_poly[c] = np.nan
X_poly = df_poly[feature_cols].values
probs = model.predict_proba(X_poly)
preds = model.predict(X_poly)
confidence = np.max(probs, axis=1)

# Load polyamorphous atom positions for x-coordinate analysis
# Polyamorphous dump files are in the ML pipeline's data directory
import glob
poly_dump_files = glob.glob('Machine_Learning_Pipeline_for_Materials_Science/data/polyamorphous/anneal_combined_*.lammpstrj')
if poly_dump_files:
    # Use the last timestep dump (most annealed)
    poly_dump_path = sorted(poly_dump_files)[-1]
else:
    poly_dump_path = 'Machine_Learning_Pipeline_for_Materials_Science/data/polyamorphous/anneal_combined_0.lammpstrj'
with open(poly_dump_path) as f:
    lines = f.readlines()
cols = lines[8].strip().replace('ITEM: ATOMS', '').strip().split()
atoms = pd.DataFrame([l.strip().split() for l in lines[9:] if l.strip()], columns=cols)
atoms['id'] = atoms['id'].astype(int)
atoms['x'] = atoms['x'].astype(float)
atoms_sorted = atoms.sort_values('id').reset_index(drop=True)

# Phase fractions at multiple thresholds
print(f'\n{"Thresh":>7s} {"Class 0":>55s} {"Class 1":>55s} {"Class 2":>55s} {"Uncert":>10s}')
for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
    phase = np.where(confidence >= thresh, preds, -1)
    total = len(phase)
    def _fmt(mask):
        if mask.sum() == 0:
            return f'{0:6d} {"0.0%":>5s} {"-":>7s} {"-":>7s}'
        x = atoms_sorted.loc[mask, 'x'].values
        return f'{mask.sum():6d} {mask.sum()/total*100:5.1f}% {np.mean(x):7.1f} {np.std(x):7.1f}'
    thresh_str = f'{thresh:.2f}'
    print(f'{thresh_str:>7s}   {_fmt(phase==0)}   {_fmt(phase==1)}   {_fmt(phase==2)}   {_fmt(phase==-1)}')

# Low-confidence atom analysis
low_conf = (confidence >= 0.50) & (confidence < 0.70)
print(f'\nLow-confidence atoms (0.50 < confidence < 0.70):')
print(f'  Count: {low_conf.sum()}')
if low_conf.sum() > 0:
    x_low = atoms_sorted.loc[low_conf, 'x'].values
    print(f'  x mean: {np.mean(x_low):.2f}, x std: {np.std(x_low):.2f}')
    print(f'  x range: [{np.min(x_low):.2f}, {np.max(x_low):.2f}]')
else:
    print('  No low-confidence atoms found.')

# Summary
print('\n' + '='*70)
print('SUMMARY')
print('='*70)
print(f'Potential energy overlap 5050 vs 4654: {overlap_pe:.3f}' if overlap_pe is not np.nan else 'Potential energy: N/A')
print(f'Von Mises overlap 5050 vs 4654: {overlap_vm:.3f}' if overlap_vm is not np.nan else 'Von Mises: N/A')
print(f'CV macro avg F1: {cv_scores.mean():.4f}')
print(f'Test accuracy: {test_acc:.4f}')
print('\nDone.')