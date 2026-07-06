"""
trustworthy_polyamorphous_inference.py
========================================
Train a model using ONLY the spatially-validated feature set (Config 2 from ablation:
9 baseline + thermo, NO MRO depth-2/3 features), using the spatial holdout split 
(x-median within each dataset, ±3.5A boundary excluded). Apply to polyamorphous.

Then compare against the leaky 85-feature model's polyamorphous predictions.
"""
import sys, os, pandas as pd, numpy as np, joblib
sys.path.insert(0, 'Machine_Learning_Pipeline_for_Materials_Science/src')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

print('='*70)
print('TRUSTWORTHY CONFIG 2 MODEL (9 baseline + thermo, no MRO depth)')
print('Training with SPATIAL HOLDOUT SPLIT')
print('='*70)

# =============================================================================
# Load data with x,y,z coordinates from raw post-processing CSVs
# =============================================================================
raw_5050 = pd.read_csv('Atomic_Simulation_Post-Processing_Pipeline/outputs/features_5050.csv')
raw_4654 = pd.read_csv('Atomic_Simulation_Post-Processing_Pipeline/outputs/features_4654.csv')
raw_6436 = pd.read_csv('Atomic_Simulation_Post-Processing_Pipeline/outputs/features_6436.csv')

ml_5050 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_5050.csv')
ml_4654 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_4654.csv')
ml_6436 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_6436.csv')

# Load polyamorphous (now with real thermo data)
ml_poly = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_polyamorphous.csv')

# Merge x,y,z and gaussian_peak columns from raw into ML tables
for ml, raw, ds in [(ml_5050, raw_5050, '5050'), (ml_4654, raw_4654, '4654'), (ml_6436, raw_6436, '6436')]:
    ml['x'] = raw['x'].values
    ml['y'] = raw['y'].values
    ml['z'] = raw['z'].values
    ml['gaussian_peak2_center'] = raw['gaussian_peak2_center'].values
    ml['gaussian_peak3_center'] = raw['gaussian_peak3_center'].values
    print(f'{ds}: merged x,y,z + gaussian peaks, {len(ml)} atoms')

# Also add gaussian_peak columns to polyamorphous ML table from raw
raw_poly = pd.read_csv('Atomic_Simulation_Post-Processing_Pipeline/outputs/features_polyamorphous.csv')
ml_poly['gaussian_peak2_center'] = raw_poly['gaussian_peak2_center'].values
ml_poly['gaussian_peak3_center'] = raw_poly['gaussian_peak3_center'].values
print(f'polyamorphous: merged gaussian peaks, {len(ml_poly)} atoms')

# =============================================================================
# Define Config 2 features: 9 baseline + thermo, NO MRO depth
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

config2_features = baseline_core + thermo_features
print(f'\nConfig 2 feature set: {len(config2_features)} features')
for f in config2_features:
    print(f'  {f}')

# =============================================================================
# Spatial holdout split (x-median, ±3.5A boundary excluded)
# =============================================================================
print('\n' + '='*70)
print('Spatial holdout split')
print('='*70)

train_parts = []
test_parts = []

for ml, raw, ds in [(ml_5050, raw_5050, '5050'), (ml_4654, raw_4654, '4654'), (ml_6436, raw_6436, '6436')]:
    x_median = ml['x'].median()
    x_cutoff = 3.5
    
    left_mask = ml['x'] <= x_median
    right_mask = ml['x'] > x_median
    boundary_mask = (ml['x'] >= x_median - x_cutoff) & (ml['x'] <= x_median + x_cutoff)
    clean_left = left_mask & ~boundary_mask
    clean_right = right_mask & ~boundary_mask
    
    df = ml.copy()
    df['phase_label'] = 0 if ds == '5050' else (1 if ds == '4654' else 2)
    
    train_parts.append(df[clean_left])
    test_parts.append(df[clean_right])
    
    print(f'{ds}: train={clean_left.sum()}, test={clean_right.sum()}, boundary={boundary_mask.sum()}')

df_train = pd.concat(train_parts, ignore_index=True)
df_test = pd.concat(test_parts, ignore_index=True)

print(f'\nTotal train: {len(df_train)}, Test: {len(df_test)}')
print(f'Train labels: {df_train["phase_label"].value_counts().to_dict()}')
print(f'Test labels: {df_test["phase_label"].value_counts().to_dict()}')

# =============================================================================
# Train CONFIG 2 model with spatial holdout
# =============================================================================
print('\n' + '='*70)
print('Training CONFIG 2 model with spatial holdout split')
print('='*70)

available = [c for c in config2_features if c in df_train.columns]
missing = [c for c in config2_features if c not in df_train.columns]
if missing:
    print(f'WARNING: Missing features (excluded): {missing}')
print(f'Feature count: {len(available)}')

X_train = df_train[available].values
y_train = df_train['phase_label'].values
X_test = df_test[available].values
y_test = df_test['phase_label'].values

model = HistGradientBoostingClassifier(
    random_state=42, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=10, max_iter=500, learning_rate=0.05,
    max_leaf_nodes=31, min_samples_leaf=20
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f'\nSpatial holdout test accuracy: {test_acc:.4f}')

print('\nPer-class F1:')
report = classification_report(y_test, y_pred, target_names=['Cu50Zr50(0)', 'Cu46Zr54(1)', 'Cu64Zr36(2)'], output_dict=True)
for c in ['Cu50Zr50(0)', 'Cu46Zr54(1)', 'Cu64Zr36(2)']:
    print(f'  {c}: {report[c]["f1-score"]:.4f}')

print('\nConfusion matrix:')
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=['True 0', 'True 1', 'True 2'], columns=['Pred 0', 'Pred 1', 'Pred 2']))

# Permutation importance
print('\nTop-10 permutation importance:')
r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=4)
sorted_idx = r.importances_mean.argsort()[::-1]
print(f'{"Rank":>5s} {"Feature":45s} {"Importance":>10s}')
print('-' * 62)
for rank, i in enumerate(sorted_idx[:10], 1):
    fn = available[i]
    tag = ' <<< THERMO' if fn in thermo_features else ''
    print(f'{rank:5d} {fn:45s} {r.importances_mean[i]:10.4f}{tag}')

# =============================================================================
# Apply to polyamorphous
# =============================================================================
print('\n' + '='*70)
print('CONFIG 2 PREDICTION ON POLYAMORPHOUS (with real thermo features)')
print('='*70)

# Ensure all feature columns exist in polyamorphous table
for c in available:
    if c not in ml_poly.columns:
        print(f'WARNING: {c} missing from polyamorphous! Filling NaN.')
        ml_poly[c] = np.nan

X_poly = ml_poly[available].values
probs_trust = model.predict_proba(X_poly)
preds_trust = model.predict(X_poly)
confidence_trust = np.max(probs_trust, axis=1)

# Get polyamorphous x-coordinates
# Need to load them from raw output since ML table may not have x,y,z
raw_poly = pd.read_csv('Atomic_Simulation_Post-Processing_Pipeline/outputs/features_polyamorphous.csv')
x_poly = raw_poly['x'].values

print(f'\n{"Thresh":>7s} {"Class 0":>55s} {"Class 1":>55s} {"Class 2":>55s} {"Uncert":>10s}')
for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
    phase = np.where(confidence_trust >= thresh, preds_trust, -1)
    total = len(phase)
    def _fmt(mask):
        if mask.sum() == 0:
            return f'{0:6d} {"0.0%":>5s} {"-":>7s} {"-":>7s}'
        x_mean = x_poly[mask].mean()
        x_std = x_poly[mask].std()
        return f'{mask.sum():6d} {mask.sum()/total*100:5.1f}% {x_mean:7.1f} {x_std:7.1f}'
    thresh_str = f'{thresh:.2f}'
    print(f'{thresh_str:>7s}   {_fmt(phase==0)}   {_fmt(phase==1)}   {_fmt(phase==2)}   {_fmt(phase==-1)}')

low_conf = (confidence_trust >= 0.50) & (confidence_trust < 0.70)
print(f'\nLow-confidence atoms (0.50 < confidence < 0.70):')
print(f'  Count: {low_conf.sum()}')
if low_conf.sum() > 0:
    x_low = x_poly[low_conf]
    print(f'  x mean: {np.mean(x_low):.2f}, x std: {np.std(x_low):.2f}')
    print(f'  x range: [{np.min(x_low):.2f}, {np.max(x_low):.2f}]')

# =============================================================================
# Load the leaky 85-feature model's predictions for comparison
# =============================================================================
print('\n\n' + '='*70)
print('COMPARISON: Trustworthy Config 2 vs Leaky 85-feature Model')
print('='*70)

# The 85-feature model was just saved to model_3class_with_thermo.pkl
leaky_model_path = 'Machine_Learning_Pipeline_for_Materials_Science/outputs/models/model_3class_with_thermo.pkl'
if os.path.exists(leaky_model_path):
    leaky_model = joblib.load(leaky_model_path)
    
    # Get all features used by leaky model
    import yaml
    with open('Machine_Learning_Pipeline_for_Materials_Science/config.yaml') as f:
        cfg = yaml.safe_load(f)
    leaky_features = [c for c in cfg['features'] if c in ml_poly.columns]
    
    X_poly_all = ml_poly[leaky_features].values
    probs_leaky = leaky_model.predict_proba(X_poly_all)
    preds_leaky = leaky_model.predict(X_poly_all)
    confidence_leaky = np.max(probs_leaky, axis=1)
    
    # Comparison at threshold 0.50
    print('\n--- Phase fractions at threshold 0.50 ---')
    print(f'{"Model":40s} {"Class 0":>12s} {"Class 1":>12s} {"Class 2":>12s} {"Uncert":>10s}')
    print('-' * 86)
    for name, preds, conf, probs in [
        ('Trustworthy Config 2', preds_trust, confidence_trust, probs_trust),
        ('Leaky 85-feature', preds_leaky, confidence_leaky, probs_leaky)
    ]:
        phase = np.where(conf >= 0.50, preds, -1)
        parts = []
        for cls in [0, 1, 2]:
            frac = (phase == cls).sum() / len(phase) * 100
            x_m = x_poly[phase == cls].mean() if (phase == cls).sum() > 0 else 0
            parts.append(f'{frac:5.1f}% (x={x_m:7.1f})')
        uncert = (phase == -1).sum() / len(phase) * 100
        print(f'{name:40s} {parts[0]:12s} {parts[1]:12s} {parts[2]:12s} {uncert:5.1f}%')
    
    # Direct disagreement
    agree = (preds_trust == preds_leaky).sum()
    total = len(preds_trust)
    disagree = total - agree
    print(f'\n--- Direct label disagreement ---')
    print(f'Total atoms: {total}')
    print(f'Agree: {agree} ({agree/total*100:.1f}%)')
    print(f'Disagree: {disagree} ({disagree/total*100:.1f}%)')
    
    # Disagreement contingency
    print(f'\n--- Disagreement cross-tabulation ---')
    print(f'Trustworthy Config 2 predictions vs Leaky 85-feature predictions:')
    ct = pd.crosstab(
        pd.Series(preds_trust, name='Trustworthy Config 2'),
        pd.Series(preds_leaky, name='Leaky 85-feature')
    )
    print(ct)
    
    # Per-class comparison (at threshold 0.50, excluding uncertain)
    print(f'\n--- Per-class spatial comparison at threshold 0.50 ---')
    for name, preds, conf in [
        ('Trustworthy Config 2', preds_trust, confidence_trust),
        ('Leaky 85-feature', preds_leaky, confidence_leaky)
    ]:
        phase = np.where(conf >= 0.50, preds, -1)
        print(f'\n{name}:')
        for cls in [0, 1, 2]:
            mask = phase == cls
            n = mask.sum()
            if n > 0:
                print(f'  Class {cls}: {n:6d} atoms ({n/total*100:5.1f}%), x={x_poly[mask].mean():7.1f}±{x_poly[mask].std():5.1f}')
            else:
                print(f'  Class {cls}: {0:6d} atoms (0.0%), x=N/A')

else:
    print('Leaky model not found at', leaky_model_path)
    print('Skipping comparison.')

# Save trustworthy model
os.makedirs('Machine_Learning_Pipeline_for_Materials_Science/outputs/models', exist_ok=True)
joblib.dump(model, 'Machine_Learning_Pipeline_for_Materials_Science/outputs/models/model_config2_trustworthy.pkl')
print(f'\nTrustworthy model saved to outputs/models/model_config2_trustworthy.pkl')

print('\nDone.')