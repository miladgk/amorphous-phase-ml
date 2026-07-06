"""
run_validation.py — Part 3: Validate on Existing Cu-Zr Data
==============================================================
Runs the full general phase-ID pipeline on:
1. The three labeled datasets combined (5050, 4654, 6436) → test if unsupervised 
   methods discover ~3 clusters that align with known compositions.
2. The polyamorphous dataset → compare unsupervised clusters vs supervised model.

No new simulations or structures — uses only existing output CSVs.
"""
import sys, os, joblib
sys.path.insert(0, 'Machine_Learning_Pipeline_for_Materials_Science/src')

from general_phase_id.build_descriptors import build_general_descriptors, concatenate_structures
from general_phase_id.clustering import gmm_clustering, umap_hdbscan_clustering, evaluate_clustering

import pandas as pd
import numpy as np

print('=' * 70)
print('PART 3: VALIDATION ON EXISTING CU-ZR DATA')
print('=' * 70)

# =============================================================================
# Build descriptors for all 4 datasets
# =============================================================================
print('\n' + '=' * 70)
print('Building general descriptors for all datasets')
print('=' * 70)

base = 'Atomic_Simulation_Post-Processing_Pipeline/outputs'

# The three labeled datasets
desc_5050 = build_general_descriptors(
    f'{base}/features_5050.csv', structure_id='5050')
desc_4654 = build_general_descriptors(
    f'{base}/features_4654.csv', structure_id='4654')
desc_6436 = build_general_descriptors(
    f'{base}/features_6436.csv', structure_id='6436')

# Polyamorphous
desc_poly = build_general_descriptors(
    f'{base}/features_polyamorphous.csv', structure_id='poly')

# Concatenate labeled
combined_labeled = concatenate_structures([desc_5050, desc_4654, desc_6436])

print(f'\nCombined labeled: {len(combined_labeled)} atoms')
print(f'  Structure IDs: {combined_labeled["structure_id"].value_counts().to_dict()}')

# Create ground-truth labels (0 for 5050, 1 for 4654, 2 for 6436)
def _true_label(sid):
    return {'5050': 0, '4654': 1, '6436': 2}.get(sid, -1)
true_labels = combined_labeled['structure_id'].map(_true_label).values

# =============================================================================
# Part 3.1: GMM on labeled datasets
# =============================================================================
print('\n' + '=' * 70)
print('Part 3.1a: GMM on combined labeled datasets')
print('=' * 70)

gmm_result = gmm_clustering(combined_labeled, min_components=2, max_components=8)

print(f'\nBest n_components (by BIC): {gmm_result["best_n_components"]}')
print(f'\nBIC vs n_components:')
print(gmm_result['bic_table'].to_string(index=False))

gmm_eval = evaluate_clustering(true_labels, gmm_result['labels'])
print(f'\nGMM vs ground truth:')
print(f'  NMI: {gmm_eval["nmi"]:.4f}')
print(f'  ARI: {gmm_eval["ari"]:.4f}')
print(f'  Valid atoms: {gmm_eval["n_valid"]}/{gmm_eval["total"]}')

print(f'\nCross-tabulation: GMM cluster vs true composition:')
ct_gmm = pd.crosstab(
    pd.Series(gmm_result['labels'], name='GMM Cluster'),
    pd.Series(true_labels, name='True Composition (group_id)')
)
print(ct_gmm)

# =============================================================================
# Part 3.1b: UMAP+HDBSCAN on labeled datasets
# =============================================================================
print('\n' + '=' * 70)
print('Part 3.1b: UMAP+HDBSCAN on combined labeled datasets')
print('=' * 70)

uh_result = umap_hdbscan_clustering(combined_labeled)

if uh_result['labels'] is not None:
    print(f'\nClusters found: {uh_result["n_clusters"]}')
    print(f'Noise fraction: {uh_result["noise_fraction"]*100:.1f}%')
    
    uh_eval = evaluate_clustering(true_labels, uh_result['labels'])
    print(f'\nUMAP+HDBSCAN vs ground truth:')
    print(f'  NMI: {uh_eval["nmi"]:.4f}')
    print(f'  ARI: {uh_eval["ari"]:.4f}')
    print(f'  Valid atoms: {uh_eval["n_valid"]}/{uh_eval["total"]}')
    
    print(f'\nCross-tabulation: UMAP+HDBSCAN cluster vs true composition:')
    ct_uh = pd.crosstab(
        pd.Series(uh_result['labels'], name='UMAP+HDBSCAN Cluster'),
        pd.Series(true_labels, name='True Composition (group_id)')
    )
    print(ct_uh)

# =============================================================================
# Part 3.2: Check comparison against trustworthy supervised model
# =============================================================================
print('\n' + '=' * 70)
print('Part 3.2: Polyamorphous — unsupervised vs supervised comparison')
print('=' * 70)

# Load the trustworthy supervised model's saved predictions
# They were saved in trustworthy_polyamorphous_inference.py output
# We need to regenerate them from the saved model
model_path = 'Machine_Learning_Pipeline_for_Materials_Science/outputs/models/model_config2_trustworthy.pkl'
if os.path.exists(model_path):
    supervised_model = joblib.load(model_path)
    
    # Build the features the model expects (same as Config 2)
    baseline_core = [
        'R1', 'voronoi_volume_temporal', 'CN_temporal', 'q4', 'q6',
        'gaussian_peak2_center', 'gaussian_peak3_center',
        'neighbor_1', 'neighbor_2']
    thermo_features = [
        'pe_atom_temporal', 'pe_atom_std_temporal',
        'hydrostatic_pressure_temporal', 'von_mises_stress_temporal',
        'deviatoric_norm_temporal', 'stress_anisotropy_temporal',
        'local_enthalpy_temporal', 'excess_energy_temporal',
        'pe_volume_residual']
    config2_features = baseline_core + thermo_features
    
    # Load polyamorphous ML table (has all columns)
    ml_poly = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_polyamorphous.csv')
    raw_poly = pd.read_csv(f'{base}/features_polyamorphous.csv')
    # Add gaussian peaks from raw
    ml_poly['gaussian_peak2_center'] = raw_poly['gaussian_peak2_center'].values
    ml_poly['gaussian_peak3_center'] = raw_poly['gaussian_peak3_center'].values
    
    # Ensure all features present
    available = [c for c in config2_features if c in ml_poly.columns]
    X_poly = ml_poly[available].values
    sup_preds = supervised_model.predict(X_poly)
    sup_probs = supervised_model.predict_proba(X_poly)
    sup_conf = np.max(sup_probs, axis=1)
    
    print(f'\nLoaded trustworthy supervised model: {model_path}')
    print(f'  Polyamorphous predictions: {len(sup_preds)} atoms')
    print(f'  Class distribution at conf>=0.50:')
    phase = np.where(sup_conf >= 0.50, sup_preds, -1)
    for cls in [0, 1, 2, -1]:
        n = (phase == cls).sum()
        label = {0: 'Class 0 (Cu50Zr50)', 1: 'Class 1 (Cu46Zr54)', 
                 2: 'Class 2 (Cu64Zr36)', -1: 'Uncertain'}[cls]
        print(f'    {label}: {n} ({n/len(phase)*100:.1f}%)')
    
    # GMM on polyamorphous
    print(f'\n{"-"*60}')
    print('GMM on polyamorphous descriptors')
    print(f'{"-"*60}')
    gmm_poly = gmm_clustering(desc_poly, min_components=2, max_components=8)
    print(f'\nBest n_components: {gmm_poly["best_n_components"]}')
    print(f'BIC vs n_components:')
    print(gmm_poly['bic_table'].to_string(index=False))
    
    print(f'\nCross-tabulation: GMM cluster vs supervised class (conf>=0.50):')
    ct_poly_gmm = pd.crosstab(
        pd.Series(gmm_poly['labels'], name='GMM Cluster'),
        pd.Series(np.where(sup_conf >= 0.50, sup_preds, -1), name='Supervised Class')
    )
    print(ct_poly_gmm)
    
    poly_eval_gmm = evaluate_clustering(
        np.where(sup_conf >= 0.50, sup_preds, -1), gmm_poly['labels'])
    print(f'\nGMM vs supervised (conf>=0.50):')
    print(f'  NMI: {poly_eval_gmm["nmi"]:.4f}')
    print(f'  ARI: {poly_eval_gmm["ari"]:.4f}')
    
    # UMAP+HDBSCAN on polyamorphous
    print(f'\n{"-"*60}')
    print('UMAP+HDBSCAN on polyamorphous descriptors')
    print(f'{"-"*60}')
    uh_poly = umap_hdbscan_clustering(desc_poly)
    
    if uh_poly['labels'] is not None:
        print(f'\nClusters found: {uh_poly["n_clusters"]}')
        print(f'Noise fraction: {uh_poly["noise_fraction"]*100:.1f}%')
        
        print(f'\nCross-tabulation: UMAP+HDBSCAN vs supervised class (conf>=0.50):')
        ct_poly_uh = pd.crosstab(
            pd.Series(uh_poly['labels'], name='UMAP+HDBSCAN Cluster'),
            pd.Series(np.where(sup_conf >= 0.50, sup_preds, -1), name='Supervised Class')
        )
        print(ct_poly_uh)
        
        poly_eval_uh = evaluate_clustering(
            np.where(sup_conf >= 0.50, sup_preds, -1), uh_poly['labels'])
        print(f'\nUMAP+HDBSCAN vs supervised (conf>=0.50):')
        print(f'  NMI: {poly_eval_uh["nmi"]:.4f}')
        print(f'  ARI: {poly_eval_uh["ari"]:.4f}')
else:
    print(f'\nTrustworthy model not found at {model_path}')
    print('Run trustworthy_polyamorphous_inference.py first.')

# =============================================================================
# Summary
# =============================================================================
print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print('\nNormalization rules applied:')
print('  1. Length-scale (voronoi_volume): divided by R1_SG^3')
print('  2. Distance (gaussian_peak_center): divided by R1_SG')
print('  3. Energy features: z-score normalized per structure')
print('  4. Stress features: z-score normalized per structure')
print('  5. Dimensionless (CN, q4, q6, etc.): passed through unchanged')
print('  6. EXCLUDED: sqrt-ratio features (not material-agnostic)')
print('  7. EXCLUDED: MRO depth-2/3 features (known leakage)')
print('  8. EXCLUDED: Gaussian fit extras (height/sigma/FWHM)')

print('\nData used:')
print(f'  5050: {len(desc_5050)} atoms')
print(f'  4654: {len(desc_4654)} atoms')
print(f'  6436: {len(desc_6436)} atoms')
print(f'  Polyamorphous: {len(desc_poly)} atoms')

print('\nDone.')