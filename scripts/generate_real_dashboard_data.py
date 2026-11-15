#!/usr/bin/env python3
"""
generate_real_dashboard_data.py
Extracts authentic scientific data from:
  1. general_phase_id_output/predictions.csv (36,828 atoms)
  2. Atomic_Simulation_Post-Processing_Pipeline/outputs/features_polyamorphous_recomputed.csv (atom 35 RDF)
Outputs:
  - docs/assets/canvas_atoms.json & assets/canvas_atoms.json (3,000 stratified sampled atoms with real coords, phase, conf, vm, hp, enthalpy, voronoi_vol)
  - docs/assets/dashboard_real_data.json & assets/dashboard_real_data.json (real X-density bins, real stress/pressure profiles, real RDF curve, real cumulative atom positions)
"""

import os
import json
import math
import ast
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def main():
    pred_path = 'general_phase_id_output/predictions.csv'
    feat_path = 'Atomic_Simulation_Post-Processing_Pipeline/outputs/features_polyamorphous_recomputed.csv'

    print(f"Loading predictions from {pred_path} ...")
    df = pd.read_csv(pred_path)
    total_atoms = len(df)
    print(f"Total atoms loaded: {total_atoms}")

    # Stress columns are in bar (PIPELINE.md). 1 MPa = 10 bar.
    df['vm_mpa'] = df['von_mises_stress_temporal'] / 10.0
    df['hp_mpa'] = df['hydrostatic_pressure_temporal'] / 10.0

    # =========================================================================
    # 1. Generate Stratified Sample of 3,000 Real Atoms for Canvas
    # =========================================================================
    total_target = 3000
    sample = df.groupby('predicted_phase_index', group_keys=False).apply(
        lambda grp: grp.sample(int(round(total_target * len(grp) / total_atoms)), random_state=42)
    ).reset_index(drop=True)

    if len(sample) > total_target:
        sample = sample.sample(total_target, random_state=42).reset_index(drop=True)
    elif len(sample) < total_target:
        diff = total_target - len(sample)
        extra = df.drop(sample.index).sample(diff, random_state=42)
        sample = pd.concat([sample, extra]).reset_index(drop=True)

    canvas_records = []
    for _, row in sample.iterrows():
        canvas_records.append({
            'x': round(float(row['x']), 2),
            'y': round(float(row['y']), 2),
            'phase': int(row['predicted_phase_index']),
            'conf': round(float(row['confidence']), 3),
            'vm': round(float(row['vm_mpa']), 1),
            'hp': round(float(row['hp_mpa']), 1),
            'enthalpy': round(float(row['local_enthalpy_temporal']), 3),
            'voronoi_vol': round(float(row['voronoi_volume_temporal']), 3)
        })

    # =========================================================================
    # 2. Real X-profile Bins (60 bins across X: 1.9 to 183.1 A)
    # =========================================================================
    num_bins = 60
    bins = np.linspace(df['x'].min(), df['x'].max(), num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    df['bin'] = pd.cut(df['x'], bins=bins, labels=False, include_lowest=True)

    density_bins = []
    for b in range(num_bins):
        sub = df[df['bin'] == b]
        c0 = int((sub['predicted_phase_name'] == 'Cu50Zr50').sum())
        c1 = int((sub['predicted_phase_name'] == 'Cu46Zr54').sum())
        c2 = int((sub['predicted_phase_name'] == 'Cu64Zr36').sum())
        unc = int((sub['confidence'] < 0.50).sum())
        vm = round(float(sub['vm_mpa'].mean()), 1) if len(sub) > 0 else 0.0
        hp = round(float(sub['hp_mpa'].mean()), 1) if len(sub) > 0 else 0.0
        conf = round(float(sub['confidence'].mean()), 3) if len(sub) > 0 else 0.0

        density_bins.append({
            'b': b,
            'xA': round(float(bin_centers[b]), 2),
            'c0': c0,
            'c1': c1,
            'c2': c2,
            'unc': unc,
            'vm': vm,
            'hp': hp,
            'conf': conf
        })

    # =========================================================================
    # 3. Fine-grained 0.5 A Histogram for Interactive Holdout Split Simulator
    # =========================================================================
    # 370 bins from 0 to 185 A (0.5 A width each)
    fine_bins = np.arange(0, 185.0, 0.5)
    fine_counts, _ = np.histogram(df['x'], bins=fine_bins)
    fine_hist = [int(c) for c in fine_counts]

    # =========================================================================
    # 4. Extract Real RDF for Sample Atom ID 35
    # =========================================================================
    print(f"Loading RDF from {feat_path} ...")
    feat_df = pd.read_csv(feat_path)
    row35 = feat_df[feat_df['id'] == 35].iloc[0]
    orig_rdf = np.array(ast.literal_eval(row35['g_i_r_temporal_avg']))
    r_min, r_max, num_r_bins = 0.01, 12.0, 100
    bin_edges = np.linspace(r_min, r_max, num_r_bins + 1)
    r_values = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Savitzky-Golay filter with degree 3 clamped to strictly non-negative
    smooth_rdf = np.maximum(0.0, savgol_filter(orig_rdf, 11, 5))
    # Before the first physical atomic contact, g(r) is strictly zero (no unphysical penetration)
    first_contact = np.where(orig_rdf > 0)[0]
    if len(first_contact) > 0:
        smooth_rdf[:first_contact[0]] = 0.0

    r1_val = float(row35['R1_SG'])
    h1 = float(row35['gaussian_peak2_height'])
    c1 = float(row35['gaussian_peak2_center'])
    fwhm1 = float(row35['gaussian_peak2_fwhm'])
    w1 = fwhm1 / (2 * np.sqrt(2 * np.log(2)))

    h2 = float(row35['gaussian_peak3_height'])
    c2 = float(row35['gaussian_peak3_center'])
    fwhm2 = float(row35['gaussian_peak3_fwhm'])
    w2 = fwhm2 / (2 * np.sqrt(2 * np.log(2)))

    def double_gaussian(x, h1, c1, w1, h2, c2, w2):
        g1 = h1 * np.exp(-((x - c1) ** 2) / (2 * w1 ** 2))
        g2 = h2 * np.exp(-((x - c2) ** 2) / (2 * w2 ** 2))
        return g1 + g2

    fit_r_min = r1_val * 1.4
    fit_r_max = r1_val * 2.2
    fit_indices = np.where((r_values >= fit_r_min) & (r_values <= fit_r_max))[0]
    fit_x = [round(float(v), 3) for v in r_values[fit_indices]]
    fit_y = [round(float(v), 3) for v in double_gaussian(r_values[fit_indices], h1, c1, w1, h2, c2, w2)]

    rdf_data = {
        'r': [round(float(v), 3) for v in r_values],
        'raw': [round(float(v), 3) for v in orig_rdf],
        'smooth': [round(float(v), 3) for v in smooth_rdf],
        'fit_x': fit_x,
        'fit_y': fit_y,
        'r1': round(r1_val, 3),
        'c1': round(c1, 3),
        'c2': round(c2, 3)
    }

    # =========================================================================
    # 5. Threshold Cutoff Summary Records
    # =========================================================================
    cutoffs = [0.50, 0.60, 0.70, 0.80, 0.90]
    cutoff_records = []
    for th in cutoffs:
        p0 = int(((df['predicted_phase_name'] == 'Cu50Zr50') & (df['confidence'] >= th)).sum())
        p1 = int(((df['predicted_phase_name'] == 'Cu46Zr54') & (df['confidence'] >= th)).sum())
        p2 = int(((df['predicted_phase_name'] == 'Cu64Zr36') & (df['confidence'] >= th)).sum())
        unc = total_atoms - (p0 + p1 + p2)
        cutoff_records.append({
            'threshold': th,
            'p0_pct': round((p0 / total_atoms) * 100, 1),
            'p1_pct': round((p1 / total_atoms) * 100, 1),
            'p2_pct': round((p2 / total_atoms) * 100, 1),
            'unc_pct': round((unc / total_atoms) * 100, 1),
            'p0_count': p0,
            'p1_count': p1,
            'p2_count': p2,
            'unc_count': unc
        })

    dashboard_bundle = {
        'density_bins': density_bins,
        'fine_hist_05A': fine_hist,
        'rdf_atom35': rdf_data,
        'cutoff_records': cutoff_records,
        # Sorted atom x positions in units of 0.001 A, for exact holdout counts
        'sorted_x_mm': [int(round(v * 1000)) for v in np.sort(df['x'].values)],
        'domain': {
            'x_min': round(float(df['x'].min()), 2),
            'x_max': round(float(df['x'].max()), 2),
            'y_min': round(float(df['y'].min()), 2),
            'y_max': round(float(df['y'].max()), 2),
            'z_min': round(float(df['z'].min()), 2),
            'z_max': round(float(df['z'].max()), 2),
            'x_median': round(float(df['x'].median()), 2),
            'total_atoms': total_atoms
        }
    }

    # =========================================================================
    # 6. Save JSON Files
    # =========================================================================
    target_dirs = ['docs/assets', 'assets']
    for d in target_dirs:
        os.makedirs(d, exist_ok=True)
        # canvas_atoms.json
        c_path = os.path.join(d, 'canvas_atoms.json')
        with open(c_path, 'w', encoding='utf-8') as f:
            json.dump(canvas_records, f, separators=(',', ':'))
        print(f"Saved: {c_path} ({len(canvas_records)} atoms, {os.path.getsize(c_path)/1024:.1f} KB)")

        # dashboard_real_data.json
        d_path = os.path.join(d, 'dashboard_real_data.json')
        with open(d_path, 'w', encoding='utf-8') as f:
            json.dump(dashboard_bundle, f, separators=(',', ':'))
        print(f"Saved: {d_path} ({os.path.getsize(d_path)/1024:.1f} KB)")

    print("\nAll authentic dashboard data successfully generated!")

if __name__ == '__main__':
    main()
