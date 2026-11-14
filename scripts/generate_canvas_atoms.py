#!/usr/bin/env python3
"""
generate_canvas_atoms.py
Samples 3,000 stratified random atoms across the 183 Å domain from predictions.csv
and outputs assets/canvas_atoms.json with keys: x, y, phase, conf, enthalpy, voronoi_vol
rounded to 3 decimal places to keep the file size compact.
"""

import os
import json
import pandas as pd

input_path = 'general_phase_id_output/predictions.csv'
if not os.path.exists(input_path):
    raise FileNotFoundError(f"{input_path} not found.")

print(f"Reading {input_path} ...")
usecols = [
    'x', 'y', 'predicted_phase_index', 'confidence',
    'local_enthalpy_temporal', 'voronoi_volume_temporal'
]
df = pd.read_csv(input_path, usecols=usecols)

total_target = 3000
total_len = len(df)

# Stratified random sample across predicted_phase_index
print(f"Taking stratified sample of {total_target} atoms from {total_len} total atoms...")
sample = df.groupby('predicted_phase_index', group_keys=False).apply(
    lambda grp: grp.sample(int(round(total_target * len(grp) / total_len)), random_state=42)
).reset_index(drop=True)

# Adjust to exactly 3000 if rounding caused +/- 1
if len(sample) > total_target:
    sample = sample.sample(total_target, random_state=42).reset_index(drop=True)
elif len(sample) < total_target:
    diff = total_target - len(sample)
    extra = df.drop(sample.index).sample(diff, random_state=42)
    sample = pd.concat([sample, extra]).reset_index(drop=True)

records = []
for _, row in sample.iterrows():
    records.append({
        'x': round(float(row['x']), 3),
        'y': round(float(row['y']), 3),
        'phase': int(row['predicted_phase_index']),
        'conf': round(float(row['confidence']), 3),
        'enthalpy': round(float(row['local_enthalpy_temporal']), 3),
        'voronoi_vol': round(float(row['voronoi_volume_temporal']), 3)
    })

# Format as compact JSON
json_str = json.dumps(records, separators=(',', ':'))
size_kb = len(json_str.encode('utf-8')) / 1024

target_dirs = ['docs/assets', 'assets']
for d in target_dirs:
    os.makedirs(d, exist_ok=True)
    out_file = os.path.join(d, 'canvas_atoms.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print(f"Exported: {out_file} ({len(records)} atoms, {size_kb:.1f} KB)")

print("Done generating canvas_atoms.json!")
