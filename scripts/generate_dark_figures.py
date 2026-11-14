#!/usr/bin/env python3
"""
generate_dark_figures.py
Regenerates all 6 scientific figures in dark mode matching the site UI:
- phase_spatial_distribution.png
- phase_composition_fractions.png
- stress_contours.png
- confusion_matrix.png
- feature_importance.png
- sample_peak_polyamorphous.png

Configured rcParams:
  figure.facecolor: '#070b14'
  axes.facecolor: '#0d1324'
  axes.edgecolor: '#334155'
  text.color, axes.labelcolor, xtick.color, ytick.color: '#e2e8f0'
  grid.color: '#1e293b'
  grid.alpha: 0.6
  legend.facecolor: '#0f172a'
  legend.edgecolor: '#334155'
  Phase colors:
    Cu50Zr50 = '#38bdf8'
    Cu46Zr54 = '#a855f7'
    Cu64Zr36 = '#f97316'
    Uncertain = '#64748b'
Saves at 220 DPI.
"""

import os
import ast
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Set dark theme rcParams
plt.rcParams['figure.facecolor'] = '#070b14'
plt.rcParams['axes.facecolor'] = '#0d1324'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['text.color'] = '#e2e8f0'
plt.rcParams['axes.labelcolor'] = '#e2e8f0'
plt.rcParams['xtick.color'] = '#e2e8f0'
plt.rcParams['ytick.color'] = '#e2e8f0'
plt.rcParams['grid.color'] = '#1e293b'
plt.rcParams['grid.alpha'] = 0.6
plt.rcParams['legend.facecolor'] = '#0f172a'
plt.rcParams['legend.edgecolor'] = '#334155'

target_dirs = ['docs/assets', 'general_phase_id_output/plots']
for d in target_dirs:
    os.makedirs(d, exist_ok=True)

def save_all_targets(fig, filename):
    for d in target_dirs:
        path = os.path.join(d, filename)
        fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print(f"Saved: {path}")

# Load predictions
print("Loading predictions dataset...")
df = pd.read_csv('general_phase_id_output/predictions.csv')

phase_colors = {
    'Cu50Zr50': '#38bdf8',
    'Cu46Zr54': '#a855f7',
    'Cu64Zr36': '#f97316'
}

# -------------------------------------------------------------------------
# Figure 1: Spatial Phase Distribution & Phase Profile along X
# -------------------------------------------------------------------------
print("Generating Fig 1: phase_spatial_distribution.png ...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1]})

for phase_name, col in phase_colors.items():
    mask = df['predicted_phase_name'] == phase_name
    ax1.scatter(df.loc[mask, 'x'], df.loc[mask, 'y'], c=col, label=phase_name,
                alpha=0.65, s=9, edgecolors='none')

ax1.set_title('Polyamorphous Metallic Glass: Spatial Phase Identification (X-Y Projection)',
              fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('X (Å)', fontsize=12)
ax1.set_ylabel('Y (Å)', fontsize=12)
ax1.legend(title='Identified Phase', loc='upper right', frameon=True, fontsize=10, title_fontsize=11)
ax1.set_xlim(df['x'].min() - 2, df['x'].max() + 2)
ax1.set_ylim(df['y'].min() - 2, df['y'].max() + 2)
ax1.grid(True, color='#1e293b', alpha=0.6, linestyle='--')

# Density profile along X
bins = np.linspace(df['x'].min(), df['x'].max(), 65)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
for phase_name, col in phase_colors.items():
    mask = df['predicted_phase_name'] == phase_name
    counts, _ = np.histogram(df.loc[mask, 'x'], bins=bins)
    ax2.plot(bin_centers, counts, label=phase_name, color=col, lw=2.6)

ax2.set_title('Per-Phase Atomic Density Profile Along Long Axis (X)', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('X (Å)', fontsize=12)
ax2.set_ylabel('Atom Count / Bin', fontsize=12)
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.set_xlim(df['x'].min() - 2, df['x'].max() + 2)
ax2.grid(True, color='#1e293b', alpha=0.6, linestyle='--')

save_all_targets(fig, 'phase_spatial_distribution.png')
plt.close(fig)

# -------------------------------------------------------------------------
# Figure 2: Local Stress Tensor & Pressure Contours
# -------------------------------------------------------------------------
print("Generating Fig 2: stress_contours.png ...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

# Von Mises Stress
sc1 = ax1.scatter(df['x'], df['y'], c=df['von_mises_stress_temporal'] / 1e3,
                  cmap='magma', s=8, alpha=0.85, edgecolors='none')
cbar1 = plt.colorbar(sc1, ax=ax1, pad=0.015)
cbar1.set_label('Von Mises Shear Stress (MPa)', fontsize=11, color='#e2e8f0')
cbar1.ax.yaxis.set_tick_params(color='#e2e8f0')
plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='#e2e8f0')
cbar1.outline.set_edgecolor('#334155')

ax1.set_title('Spatial Stress Field: Local Von Mises Shear Stress Distribution', fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('X (Å)', fontsize=11)
ax1.set_ylabel('Y (Å)', fontsize=11)
ax1.grid(True, color='#1e293b', alpha=0.5, linestyle=':')

# Hydrostatic Pressure
sc2 = ax2.scatter(df['x'], df['y'], c=df['hydrostatic_pressure_temporal'] / 1e3,
                  cmap='coolwarm', s=8, alpha=0.85, edgecolors='none')
cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.015)
cbar2.set_label('Hydrostatic Pressure (MPa)', fontsize=11, color='#e2e8f0')
cbar2.ax.yaxis.set_tick_params(color='#e2e8f0')
plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='#e2e8f0')
cbar2.outline.set_edgecolor('#334155')

ax2.set_title('Spatial Stress Field: Local Hydrostatic Pressure Distribution', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('X (Å)', fontsize=11)
ax2.set_ylabel('Y (Å)', fontsize=11)
ax2.grid(True, color='#1e293b', alpha=0.5, linestyle=':')

save_all_targets(fig, 'stress_contours.png')
plt.close(fig)

# -------------------------------------------------------------------------
# Figure 3: Spatial Holdout Confusion Matrix Heatmap
# -------------------------------------------------------------------------
print("Generating Fig 3: confusion_matrix.png ...")
cm_data = np.array([
    [3102, 1869, 161],
    [1901, 3201, 35],
    [149, 10, 5805]
])
labels = ['Cu50Zr50', 'Cu46Zr54', 'Cu64Zr36']
cm_norm = cm_data.astype('float') / cm_data.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Normalized Fraction', fontsize=11, color='#e2e8f0')
cbar.ax.yaxis.set_tick_params(color='#e2e8f0')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#e2e8f0')
cbar.outline.set_edgecolor('#334155')

for i in range(len(labels)):
    for j in range(len(labels)):
        text_color = '#ffffff' if cm_norm[i, j] > 0.45 else '#94a3b8'
        ax.text(j, i, f"{cm_data[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center", va="center", color=text_color, fontsize=11.5, fontweight='bold')

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels([f"Pred {l}" for l in labels], fontsize=11, fontweight='semibold')
ax.set_yticklabels([f"True {l}" for l in labels], fontsize=11, fontweight='semibold')
ax.set_xlabel('Predicted Structural Phase', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('True Reference Phase', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Spatial Holdout Confusion Matrix\n(Accuracy: 74.59% | Cu64Zr36 F1: 97.03%)',
             fontsize=13, fontweight='bold', pad=14)

save_all_targets(fig, 'confusion_matrix.png')
plt.close(fig)

# -------------------------------------------------------------------------
# Figure 4: Feature Importance Ranking (Permutation Importance)
# -------------------------------------------------------------------------
print("Generating Fig 4: feature_importance.png ...")
features = [
    'local_enthalpy_temporal',
    'excess_energy_temporal',
    'pe_volume_residual',
    'voronoi_volume_temporal',
    'hydrostatic_pressure_temporal',
    'R1_SG',
    'deviatoric_norm_temporal',
    'CN_temporal',
    'gaussian_peak3_center',
    'neighbor_1'
]
importances = [0.2450, 0.2158, 0.2150, 0.1186, 0.0456, 0.0363, 0.0288, 0.0055, 0.0053, 0.0032]

fig, ax = plt.subplots(figsize=(10.5, 6.5))
y_pos = np.arange(len(features))
bars = ax.barh(y_pos, importances[::-1], color='#38bdf8', edgecolor='#334155', alpha=0.9, height=0.68)

# Subtle highlight for top 3 features
for idx in range(len(bars)-3, len(bars)):
    bars[idx].set_color('#0ea5e9')
    bars[idx].set_edgecolor('#38bdf8')

ax.set_yticks(y_pos)
ax.set_yticklabels(features[::-1], fontsize=10.5)
ax.set_xlabel('Permutation Feature Importance (Mean Accuracy Drop)', fontsize=11.5, fontweight='bold', labelpad=10)
ax.set_title('Top Structural & Thermodynamic Descriptors for Phase Classification', fontsize=13, fontweight='bold', pad=12)
for i, v in enumerate(importances[::-1]):
    ax.text(v + 0.004, i, f"{v:.4f}", va='center', fontsize=10, fontweight='bold', color='#f8fafc')
ax.set_xlim(0, max(importances) * 1.16)
ax.grid(True, axis='x', color='#1e293b', alpha=0.7, linestyle='--')

save_all_targets(fig, 'feature_importance.png')
plt.close(fig)

# -------------------------------------------------------------------------
# Figure 5: Phase Fractions vs Confidence Thresholds
# -------------------------------------------------------------------------
print("Generating Fig 5: phase_composition_fractions.png ...")
thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
cu50 = [32.04, 20.8, 10.2, 2.8, 0.2]
cu46 = [24.74, 17.8, 11.7, 6.2, 1.8]
cu64 = [37.08, 34.4, 31.7, 28.0, 22.5]
uncertain = [6.14, 26.9, 46.5, 63.1, 75.4]

fig, ax = plt.subplots(figsize=(10.5, 6))
bar_w = 0.17
r = np.arange(len(thresholds))

ax.bar(r - 1.5*bar_w, cu50, width=bar_w, label='Cu50Zr50', color='#38bdf8', edgecolor='#334155', alpha=0.95)
ax.bar(r - 0.5*bar_w, cu46, width=bar_w, label='Cu46Zr54', color='#a855f7', edgecolor='#334155', alpha=0.95)
ax.bar(r + 0.5*bar_w, cu64, width=bar_w, label='Cu64Zr36', color='#f97316', edgecolor='#334155', alpha=0.95)
ax.bar(r + 1.5*bar_w, uncertain, width=bar_w, label='Uncertain', color='#64748b', edgecolor='#334155', alpha=0.95)

ax.set_xlabel('Classification Confidence Threshold', fontsize=11.5, fontweight='bold', labelpad=10)
ax.set_ylabel('Atomic Fraction (%)', fontsize=11.5, fontweight='bold', labelpad=10)
ax.set_title('Polyamorphous Target Phase Fractions Across Confidence Thresholds', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(r)
ax.set_xticklabels([f"≥ {t:.2f}" for t in thresholds], fontsize=10.5, fontweight='semibold')
ax.legend(title='Phase Classification', loc='upper right', frameon=True, fontsize=10, title_fontsize=11)
ax.set_ylim(0, 85)
ax.grid(True, axis='y', color='#1e293b', alpha=0.7, linestyle='--')

save_all_targets(fig, 'phase_composition_fractions.png')
plt.close(fig)

# -------------------------------------------------------------------------
# Figure 6: Sample Atom RDF Peak Analysis (Atom ID 35)
# -------------------------------------------------------------------------
print("Generating Fig 6: sample_peak_polyamorphous.png ...")
feat_df = pd.read_csv('Atomic_Simulation_Post-Processing_Pipeline/outputs/features_polyamorphous_csro_fixed.csv')
row35 = feat_df[feat_df['id'] == 35].iloc[0]

original_rdf = np.array(ast.literal_eval(row35['g_i_r_temporal_avg']))
r_min, r_max, num_bins = 0.01, 12.0, 100
bin_edges = np.linspace(r_min, r_max, num_bins + 1)
r_values = 0.5 * (bin_edges[:-1] + bin_edges[1:])
smoothed_rdf = savgol_filter(original_rdf, 11, 5)

global_peaks_r = [2.5878, 4.5062, 4.866, 5.1058, 5.8252, 6.6644, 7.264, 7.9834, 8.7028, 9.542, 9.9018, 10.5012, 11.4605]
global_peaks_h = [2.9947, 1.2757, 1.2519, 1.3614, 1.0929, 1.1204, 1.1576, 1.4255, 0.7252, 1.0183, 1.0081, 1.1416, 0.7534]
targeted_peaks_r = [4.5062, 5.1058]
targeted_peaks_h = [2.2359, 1.3062]
r1_val = 2.5878

expected_ratios_map = {
    "sqrt3": math.sqrt(3),
    "sqrt4": math.sqrt(4),
    "sqrt7": math.sqrt(7),
    "sqrt12": math.sqrt(12),
}
validation_results = {
    "sqrt3": 4.5062,
    "sqrt4": 5.1058,
    "sqrt7": 6.6644,
    "sqrt12": 8.7028,
}

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(r_values, original_rdf, label="Original Avg RDF", alpha=0.45, color="#64748b", lw=1.2, zorder=1)
ax.plot(r_values, smoothed_rdf, label="Smoothed (w=11, p=5)", color="#38bdf8", lw=2.2, zorder=2)
ax.plot(global_peaks_r, global_peaks_h, "x", color="#f87171", markersize=8, mew=2, label="Global Peaks", zorder=3)

if len(targeted_peaks_r) > 0:
    ax.plot(targeted_peaks_r, targeted_peaks_h, "o", color="#ec4899", markersize=8, mfc="none", mew=1.8,
            label="Targeted Peaks (2nd)", zorder=4)

if not np.isnan(r1_val):
    ax.axvline(r1_val, color="#34d399", linestyle="--", linewidth=1.2, label=f"Identified R1 ({r1_val:.3f} Å)", zorder=0)

    for label, ratio in expected_ratios_map.items():
        r_exp = r1_val * ratio
        val_r = validation_results.get(label)
        color = "#a855f7" if val_r is None else "#fb923c"
        linestyle = ":" if val_r is None else "-."
        linewidth = 1.0 if val_r is None else 1.4
        if r_exp <= r_max:
            ax.axvline(r_exp, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.75,
                       label=f"{label}*R1 ({r_exp:.3f} Å)", zorder=0)
        if val_r is not None:
            idx_closest = np.argmin(np.abs(r_values - val_r))
            ax.plot(val_r, original_rdf[idx_closest], "s", color="#fb923c", markersize=7, mfc="none", mew=1.8,
                    label=f"Validated {label}", zorder=4)

# Double Gaussian Fit
def double_gaussian(x, h1, c1, w1, h2, c2, w2):
    g1 = h1 * np.exp(-((x - c1) ** 2) / (2 * w1 ** 2))
    g2 = h2 * np.exp(-((x - c2) ** 2) / (2 * w2 ** 2))
    return g1 + g2

fit_h1, fit_c1, fit_w1 = 1.1801, 4.5194, 0.6188
fit_h2, fit_c2, fit_w2 = 0.7345, 5.3760, 0.3123
fit_r_min = r1_val * 1.5
fit_r_max = r1_val * 2.2
fit_indices = np.where((r_values >= fit_r_min) & (r_values <= fit_r_max))[0]
x_fit_plot = r_values[fit_indices]
y_fit_plot = double_gaussian(x_fit_plot, fit_h1, fit_c1, fit_w1, fit_h2, fit_c2, fit_w2)

ax.plot(x_fit_plot, y_fit_plot, color="#22d3ee", linewidth=2.4, linestyle="--",
        label=f"Double Gaussian Fit (R²=0.820)", zorder=5)
ax.axvline(fit_c1, color="#22d3ee", linestyle=":", alpha=0.75, label=f"Fit C1 ({fit_c1:.3f} Å)")
ax.axvline(fit_c2, color="#22d3ee", linestyle=":", alpha=0.75, label=f"Fit C2 ({fit_c2:.3f} Å)")

ax.set_xlabel("r (Å)", fontsize=12, fontweight='bold', labelpad=8)
ax.set_ylabel("g(r)", fontsize=12, fontweight='bold', labelpad=8)
ax.set_title("Peak Analysis & Crystallographic Validation for Sample Atom ID 35",
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=9, loc="upper right", frameon=True, ncol=2)
ax.grid(True, alpha=0.6, color='#1e293b')
ax.set_ylim(bottom=0, top=max(original_rdf) * 1.12)
ax.set_xlim(left=0, right=r_max)
ax.set_xticks(np.arange(0, r_max + 1, 1))

save_all_targets(fig, 'sample_peak_polyamorphous.png')
# Also save to Atomic_Simulation_Post-Processing_Pipeline/outputs/
fig.savefig('Atomic_Simulation_Post-Processing_Pipeline/outputs/sample_peak_polyamorphous.png',
            dpi=220, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
plt.close(fig)

print("All 6 dark-mode scientific figures successfully rendered and saved at 220 DPI!")
