"""
add_energy_volume_residual.py
=============================
Standalone downstream script that adds a 'pe_volume_residual' column to a
per-dataset temporally-averaged CSV.

The residual is computed per atom type within a single dataset:
    pe_volume_residual = pe_atom_temporal - polyfit(voronoi_volume_temporal, degree=2)

This captures the component of potential energy NOT explained by local volume,
which may be sensitive to chemical ordering or phase beyond what volume alone
predicts.

IMPORTANT:
- Run ONCE per dataset CSV (5050, 4654, 6436, polyamorphous) — never on a
  combined/pooled CSV. Pooling would let the larger-population phase dominate
  the fit and bias residuals for the smaller phase.
- Fit is per atom_type within each dataset (via groupby).
- Writes IN PLACE by default (overwrites input CSV with new column added).
  Use --output to write to a different file instead.
"""

import numpy as np
import pandas as pd


def add_energy_volume_residual(df, atom_type_col='type', degree=2):
    """
    Add pe_volume_residual column to a per-dataset DataFrame.

    For each atom type within the dataset, fits a degree-2 polynomial
    of pe_atom_temporal vs voronoi_volume_temporal, then computes the
    residual (actual - predicted).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'type', 'voronoi_volume_temporal', 'pe_atom_temporal'
    atom_type_col : str
        Column name for atom type. Default 'type' (matches temporal_averaging.py output).
    degree : int
        Polynomial degree for the fit. Default 2.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with new 'pe_volume_residual' column added.
    """
    df = df.copy()
    df['pe_volume_residual'] = np.nan

    for atype, group in df.groupby(atom_type_col):
        x = group['voronoi_volume_temporal'].values
        y = group['pe_atom_temporal'].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 10:
            continue
        coeffs = np.polyfit(x[mask], y[mask], degree)
        predicted = np.polyval(coeffs, x)
        df.loc[group.index, 'pe_volume_residual'] = y - predicted

    return df


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Add pe_volume_residual to a per-dataset temporally-averaged CSV."
    )
    parser.add_argument('input_csv', help='Path to input CSV (e.g., features_5050.csv)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV path. If not provided, overwrites input CSV in place.')
    args = parser.parse_args()

    input_csv = args.input_csv
    output_csv = args.output if args.output else input_csv

    df = pd.read_csv(input_csv)
    print(f"Loaded {input_csv}: {len(df)} rows, {len(df.columns)} cols")

    # Verify required columns exist
    required = ['type', 'voronoi_volume_temporal', 'pe_atom_temporal']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        sys.exit(1)

    df = add_energy_volume_residual(df)
    df.to_csv(output_csv, index=False)

    # Sanity check
    resid = df['pe_volume_residual']
    print(f"\npe_volume_residual stats for {input_csv}:")
    print(f"  mean={resid.mean():.6e}, std={resid.std():.4f}")
    print(f"  min={resid.min():.4f}, max={resid.max():.4f}")
    print(f"  non-NaN count: {resid.notna().sum()} / {len(resid)}")

    # Per-type means should be ~0
    for atype, group in df.groupby('type'):
        r = group['pe_volume_residual']
        print(f"  type={int(atype)}: mean={r.mean():.6e}, std={r.std():.4f}")

    if output_csv == input_csv:
        print(f"\nWrote pe_volume_residual to {input_csv} (in-place)")
    else:
        print(f"\nWrote pe_volume_residual to {output_csv}")