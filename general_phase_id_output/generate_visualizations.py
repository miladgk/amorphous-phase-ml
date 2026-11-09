#!/usr/bin/env python3
"""
generate_visualizations.py
Generates comprehensive visual outputs and evaluation plots from the
trained model and polyamorphous prediction pipeline.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Seaborn removed - using pure matplotlib styling

output_dir = 'general_phase_id_output/plots'
os.makedirs(output_dir, exist_ok=True)

# 1. Load predictions
df = pd.read_csv('general_phase_id_output/predictions.csv')

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
colors = {'Cu50Zr50': '#3498db', 'Cu46Zr54': '#9b59b6', 'Cu64Zr36': '#e67e22'}

# -------------------------------------------------------------
# Figure 1: Spatial Phase Distribution & Phase Profile along X
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1]})

for phase_name, col in colors.items():
    mask = df['predicted_phase_name'] == phase_name
    ax1.scatter(df.loc[mask, 'x'], df.loc[mask, 'y'], c=col, label=phase_name,
                alpha=0.6, s=8, edgecolors='none')

ax1.set_title('Polyamorphous Metallic Glass: Spatial Phase Identification (X-Y Slice)', fontsize=14, fontweight='bold')
ax1.set_xlabel('X (Å)', fontsize=12)
ax1.set_ylabel('Y (Å)', fontsize=12)
ax1.legend(title='Identified Phase', loc='upper right', frameon=True, fontsize=10)
ax1.set_xlim(df['x'].min() - 2, df['x'].max() + 2)
ax1.set_ylim(df['y'].min() - 2, df['y'].max() + 2)

# Density profile along X
bins = np.linspace(df['x'].min(), df['x'].max(), 60)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
for phase_name, col in colors.items():
    mask = df['predicted_phase_name'] == phase_name
    counts, _ = np.histogram(df.loc[mask, 'x'], bins=bins)
    ax2.plot(bin_centers, counts, label=phase_name, color=col, lw=2.5)

ax2.set_title('Per-Phase Atomic Density Profile Along Long Axis (X)', fontsize=12, fontweight='bold')
ax2.set_xlabel('X (Å)', fontsize=12)
ax2.set_ylabel('Atom Count / Bin', fontsize=12)
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.set_xlim(df['x'].min() - 2, df['x'].max() + 2)

plt.tight_layout()
fig_path1 = os.path.join(output_dir, 'phase_spatial_distribution.png')
plt.savefig(fig_path1, dpi=200)
plt.close()
print(f"Generated: {fig_path1}")

# -------------------------------------------------------------
# Figure 2: Stress Field Contours (Von Mises & Hydrostatic)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

# Von Mises Stress
sc1 = ax1.scatter(df['x'], df['y'], c=df['von_mises_stress_temporal'] / 1e3, cmap='viridis',
                  s=7, alpha=0.8, edgecolors='none')
cbar1 = plt.colorbar(sc1, ax=ax1, pad=0.01)
cbar1.set_label('Von Mises Stress (MPa / units)', fontsize=11)
ax1.set_title('Spatial Stress Field: Von Mises Stress Distribution', fontsize=13, fontweight='bold')
ax1.set_xlabel('X (Å)', fontsize=11)
ax1.set_ylabel('Y (Å)', fontsize=11)

# Hydrostatic Pressure
sc2 = ax2.scatter(df['x'], df['y'], c=df['hydrostatic_pressure_temporal'] / 1e3, cmap='coolwarm',
                  s=7, alpha=0.8, edgecolors='none')
cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.01)
cbar2.set_label('Hydrostatic Pressure (MPa / units)', fontsize=11)
ax2.set_title('Spatial Stress Field: Hydrostatic Pressure Distribution', fontsize=13, fontweight='bold')
ax2.set_xlabel('X (Å)', fontsize=11)
ax2.set_ylabel('Y (Å)', fontsize=11)

plt.tight_layout()
fig_path2 = os.path.join(output_dir, 'stress_contours.png')
plt.savefig(fig_path2, dpi=200)
plt.close()
print(f"Generated: {fig_path2}")

# -------------------------------------------------------------
# Figure 3: Spatial Holdout Confusion Matrix Heatmap
# -------------------------------------------------------------
cm_data = np.array([
    [3102, 1869, 161],
    [1901, 3201, 35],
    [149, 10, 5805]
])
labels = ['Cu50Zr50', 'Cu46Zr54', 'Cu64Zr36']
cm_norm = cm_data.astype('float') / cm_data.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for i in range(len(labels)):
    for j in range(len(labels)):
        text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
        ax.text(j, i, f"{cm_data[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center", va="center", color=text_color, fontsize=11, fontweight='semibold')

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels([f"Pred {l}" for l in labels], fontsize=11)
ax.set_yticklabels([f"True {l}" for l in labels], fontsize=11)
ax.set_xlabel('Predicted Structural Phase', fontsize=12, fontweight='bold')
ax.set_ylabel('True Reference Phase', fontsize=12, fontweight='bold')
ax.set_title('Spatial Holdout Confusion Matrix\n(Accuracy: 74.6% | Cu64Zr36 F1: 97.0%)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig_path3 = os.path.join(output_dir, 'confusion_matrix.png')
plt.savefig(fig_path3, dpi=200)
plt.close()
print(f"Generated: {fig_path3}")

# -------------------------------------------------------------
# Figure 4: Feature Importance Ranking (Permutation Importance)
# -------------------------------------------------------------
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

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(features))
ax.barh(y_pos, importances[::-1], color='#2980b9', edgecolor='black', alpha=0.85, height=0.65)
ax.set_yticks(y_pos)
ax.set_yticklabels(features[::-1], fontsize=10.5)
ax.set_xlabel('Permutation Feature Importance (Mean Accuracy Drop)', fontsize=11, fontweight='bold')
ax.set_title('Top Structural & Thermodynamic Descriptors for Phase Classification', fontsize=13, fontweight='bold')
for i, v in enumerate(importances[::-1]):
    ax.text(v + 0.003, i, f"{v:.4f}", va='center', fontsize=9.5, fontweight='bold')
ax.set_xlim(0, max(importances) * 1.15)
plt.tight_layout()
fig_path4 = os.path.join(output_dir, 'feature_importance.png')
plt.savefig(fig_path4, dpi=200)
plt.close()
print(f"Generated: {fig_path4}")

# -------------------------------------------------------------
# Figure 5: Phase Fractions vs Confidence Thresholds
# -------------------------------------------------------------
thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
cu50 = [32.0, 20.8, 10.2, 2.8, 0.2]
cu46 = [24.7, 17.8, 11.7, 6.2, 1.8]
cu64 = [37.1, 34.4, 31.7, 28.0, 22.5]
uncertain = [6.1, 26.9, 46.5, 63.1, 75.4]

fig, ax = plt.subplots(figsize=(10, 5.5))
bar_w = 0.16
r = np.arange(len(thresholds))

ax.bar(r - 1.5*bar_w, cu50, width=bar_w, label='Cu50Zr50', color='#3498db', edgecolor='black', alpha=0.85)
ax.bar(r - 0.5*bar_w, cu46, width=bar_w, label='Cu46Zr54', color='#9b59b6', edgecolor='black', alpha=0.85)
ax.bar(r + 0.5*bar_w, cu64, width=bar_w, label='Cu64Zr36', color='#e67e22', edgecolor='black', alpha=0.85)
ax.bar(r + 1.5*bar_w, uncertain, width=bar_w, label='Uncertain', color='#7f8c8d', edgecolor='black', alpha=0.85)

ax.set_xlabel('Classification Confidence Threshold', fontsize=11, fontweight='bold')
ax.set_ylabel('Atomic Fraction (%)', fontsize=11, fontweight='bold')
ax.set_title('Polyamorphous Target Phase Fractions Across Confidence Thresholds', fontsize=13, fontweight='bold')
ax.set_xticks(r)
ax.set_xticklabels([f"≥ {t:.2f}" for t in thresholds], fontsize=10.5)
ax.legend(title='Phase Classification', loc='upper right', frameon=True, fontsize=10)
ax.set_ylim(0, 85)
plt.tight_layout()
fig_path5 = os.path.join(output_dir, 'phase_composition_fractions.png')
plt.savefig(fig_path5, dpi=200)
plt.close()
print(f"Generated: {fig_path5}")

print("All visual output plots successfully generated.")
