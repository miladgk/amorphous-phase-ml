#!/usr/bin/env python3
"""
process_trajectory.py — Phase-label a multi-frame LAMMPS trajectory
=====================================================================
Usage:
    conda run -n materials-sim-ml python3 \\
        Machine_Learning_Pipeline_for_Materials_Science/src/process_trajectory.py \\
        [--snapshots SNAPSHOT_DIR_OR_GLOB]

If --snapshots is omitted, processes all .lammpstrj files in
Machine_Learning_Pipeline_for_Materials_Science/data/polyamorphous/.

Output:
    Machine_Learning_Pipeline_for_Materials_Science/outputs/phase_labeled_trajectory/
        phase_labeled_trajectory.lammpstrj   (combined multi-frame dump)
        phase_id_to_name.json                (phase ID mapping)
        per_frame_summary.csv                (phase fractions per frame)

Performance notes:
    Each snapshot requires an independent post-processing pipeline run (the
    pipeline produces one temporally-averaged output CSV per run). For N
    snapshots, the full run takes ~30 min per snapshot × N.
    
    Use --lite for inference-only mode: sets MRO_DEPTH=1 (skips expensive
    depth-2/3 neighbor averaging — those features are excluded from the
    classifier's whitelist anyway). This reduces per-frame compute.
    Use --n-workers to parallelize independent frames across CPU cores
    for faster wall-clock turnaround (reduces wall time, not total compute).
"""
import sys, os, shutil, json, tempfile, glob, re, yaml, logging, subprocess, argparse
import numpy as np, pandas as pd, joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
    'Atomic_Simulation_Post-Processing_Pipeline', 'src'))
from general_phase_id.build_descriptors import build_general_descriptors
from general_phase_id.train_and_predict import load_model_with_metadata, check_inference_consistency

# =============================================================================
# Spatial smoothing functions
# =============================================================================

def smooth_labels(labels_orig, probs, features_csv, min_agreement_fraction=0.6, 
                  n_classes=3, verbose=True):
    """
    Single-pass neighbor-majority-vote smoothing using Voronoi neighbor lists.
    
    For each atom: if fewer than min_agreement_fraction of its Voronoi neighbors
    share its label, relabel it to the neighbor majority class. If neighbors 
    are split with no clear majority, leave unchanged.
    
    Uses the existing Voronoi neighbor information from the features CSV.
    The CSV must contain neighbor indent lists (e.g. columns like neighbor_1, 
    neighbor_2) OR we compute from positions using freud.
    
    Parameters
    ----------
    labels_orig : np.ndarray
        Original per-atom phase predictions (0..n_classes-1, or -1 for Uncertain)
    probs : np.ndarray
        Prediction probabilities (n_atoms, n_classes)
    features_csv : str
        Path to features CSV with x,y,z coordinates and atom data
    min_agreement_fraction : float
        Minimum fraction of neighbors agreeing to keep or assign a label
    n_classes : int
        Number of phase classes
    
    Returns
    -------
    smoothed_labels : np.ndarray
        Labels after single-pass smoothing
    n_changed : int
        Number of atoms whose label changed
    n_uncertain_resolved : int
        Number of Uncertain atoms that got assigned a confident label
    """
    import freud
    import numpy as np
    import pandas as pd
    
    raw = pd.read_csv(features_csv)
    pos = raw[['x', 'y', 'z']].values.astype(np.float64)
    n_atoms = len(pos)
    
    # Get box dimensions from coordinate ranges
    box_size = np.array([
        pos[:, 0].max() - pos[:, 0].min(),
        pos[:, 1].max() - pos[:, 1].min(),
        pos[:, 2].max() - pos[:, 2].min()
    ]) + 1.0  # add small margin for periodic boundary
    
    # Build Voronoi neighbor list using freud
    box = freud.box.Box(*box_size)
    voro = freud.locality.Voronoi()
    voro.compute(system=(box, pos))
    nlist = voro.nlist
    
    # Get neighbor list per atom
    # nlist.point_indices (source) and nlist.query_point_indices (neighbor) are parallel arrays
    neighbors = [[] for _ in range(n_atoms)]
    for i, j in zip(nlist.point_indices, nlist.query_point_indices):
        neighbors[i].append(j)
    
    if verbose:
        avg_nn = np.mean([len(n) for n in neighbors])
        print(f'  Voronoi neighbors: avg={avg_nn:.1f} per atom')
    
    # Single pass: count neighbor labels for each atom
    labels = labels_orig.copy()
    n_changed = 0
    n_uncertain_resolved = 0
    
    # Use ORIGINAL labels for neighbor voting (not iteratively updated)
    for i in range(n_atoms):
        if len(neighbors[i]) == 0:
            continue
        
        neighbor_labels = labels_orig[neighbors[i]]
        
        # Count occurrences of each class among neighbors (excluding Uncertain=-1)
        valid = neighbor_labels >= 0
        if valid.sum() == 0:
            continue
        
        counts = np.bincount(neighbor_labels[valid], minlength=n_classes)[:n_classes]
        majority_count = counts.max()
        majority_class = counts.argmax()
        agreement = majority_count / valid.sum()
        
        if agreement >= min_agreement_fraction:
            new_label = majority_class
        else:
            new_label = labels_orig[i]
        
        if new_label != labels[i]:
            n_changed += 1
            if labels[i] == -1 and new_label >= 0:
                n_uncertain_resolved += 1
            labels[i] = new_label
    
    if verbose:
        print(f'  Smoothed: {n_changed}/{n_atoms} atoms changed '
              f'({n_uncertain_resolved} Uncertain resolved)')
    
    return labels, n_changed, n_uncertain_resolved


def compute_transition_width(x_coords, labels_smooth, labels_orig, 
                              boundary_positions=(60, 120), verbose=True):
    """
    Measure transition-zone width before and after smoothing.
    For each boundary position, find the distance over which the dominant 
    phase changes between classes.
    """
    import numpy as np
    
    results = {}
    for bp, left_class, right_class in [(60, 1, 0), (120, 0, 2)]:
        # Find atoms within 10A of the boundary
        mask = (x_coords >= bp - 10) & (x_coords <= bp + 10)
        
        xx = x_coords[mask]
        
        # For each label set, compute fraction of left_class vs right_class
        # in sliding windows
        for label_set_name, ls in [('original', labels_orig), ('smoothed', labels_smooth)]:
            ls_masked = ls[mask]
            
            # Bin by 2A windows
            bins = np.arange(bp - 10, bp + 10, 2)
            widths = []
            for bi in range(len(bins) - 1):
                bin_mask = (xx >= bins[bi]) & (xx < bins[bi+1])
                if bin_mask.sum() < 10:
                    continue
                bin_labels = ls_masked[bin_mask]
                frac_left = (bin_labels == left_class).sum() / bin_mask.sum()
                frac_right = (bin_labels == right_class).sum() / bin_mask.sum()
                widths.append({
                    'x_center': (bins[bi] + bins[bi+1]) / 2,
                    'frac_left': frac_left,
                    'frac_right': frac_right
                })
            
            if verbose and len(widths) > 1:
                # Estimate transition width: distance between 20% and 80% crossing
                # of the dominant class
                f_left = np.array([w['frac_left'] for w in widths])
                x_cent = np.array([w['x_center'] for w in widths])
                
                # Crude estimate: where does left_class fall below 0.5?
                above_half = f_left > 0.4
                if above_half.any() and (~above_half).any():
                    cross = np.where(above_half)[0][-1] if above_half.any() else 0
                    if cross < len(x_cent) - 1:
                        width = x_cent[cross + 1] - x_cent[cross]
                    else:
                        width = 0
                else:
                    width = 0
                
                results[f'{label_set_name}_bp_{bp}'] = width
    
    return results

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PP_CFG = os.path.join(ROOT, 'Atomic_Simulation_Post-Processing_Pipeline', 'config.yaml')
PP_SRC = os.path.join(ROOT, 'Atomic_Simulation_Post-Processing_Pipeline', 'src')
DEFAULT_POLY = os.path.join(ROOT, 'Machine_Learning_Pipeline_for_Materials_Science', 'data', 'polyamorphous')
OUT = os.path.join(ROOT, 'Machine_Learning_Pipeline_for_Materials_Science', 'outputs', 'phase_labeled_trajectory')
MODEL = os.path.join(ROOT, 'general_phase_id_output', 'model.pkl')
NP = os.path.join(ROOT, 'general_phase_id_output', 'norm_params.pkl')
FR = os.path.join(ROOT, 'general_phase_id_output', 'feature_ranges.pkl')

# Feature whitelist — must match what the model was trained on
WHITELIST = [
    'R1_SG','voronoi_volume_temporal','CN_temporal','q4','q6',
    'gaussian_peak2_center','gaussian_peak3_center','neighbor_1','neighbor_2',
    'excess_energy_temporal','local_energy_fluctuation_temporal',
    'hydrostatic_pressure_temporal','von_mises_stress_temporal',
    'deviatoric_norm_temporal','stress_anisotropy_temporal',
    'local_enthalpy_temporal','pe_atom_std_temporal','pe_volume_residual',
    # New features (CSRO + neighbor PE, confirmed non-leaky via spatial holdout)
    'csro_unlike_temporal','csro_like_temporal','mean_neighbor_pe_temporal'
]

# Fixed phase ID mapping (from trained model: classes [0, 1, 2])
PHASE_MAP = {0: 'Cu50Zr50', 1: 'Cu46Zr54', 2: 'Cu64Zr36'}


def ts_from_name(fn: str) -> int:
    """Extract timestep from filename like anneal_combined_1000000.lammpstrj"""
    m = re.search(r'(\d+)\.lammpstrj', os.path.basename(fn))
    return int(m.group(1)) if m else 0


def read_header(fp: str) -> dict:
    """Read LAMMPS dump header: timestep, natoms, box bounds, column names."""
    info = {'timestep': 0, 'natoms': 0, 'box_bounds': None, 'columns': [], 'raw_lines': []}
    with open(fp) as f:
        info['raw_lines'] = f.readlines()
    for i, line in enumerate(info['raw_lines']):
        s = line.strip()
        if s == 'ITEM: TIMESTEP':
            info['timestep'] = int(info['raw_lines'][i+1].strip())
        elif s == 'ITEM: NUMBER OF ATOMS':
            info['natoms'] = int(info['raw_lines'][i+1].strip())
        elif s.startswith('ITEM: BOX BOUNDS'):
            parts = s.split()
            info['box_bounds'] = {
                'tilt': parts[3:] if len(parts) > 3 else None,
                'bounds': [[float(x) for x in info['raw_lines'][i+1+j].strip().split()] for j in range(3)]
            }
        elif s.startswith('ITEM: ATOMS'):
            info['columns'] = s.replace('ITEM: ATOMS', '').strip().split()
            info['data_start'] = i + 1
            break
    return info


def run_pp_single(snap_path: str, out_csv: str, lite: bool = False) -> bool:
    """
    Run the post-processing pipeline on a SINGLE snapshot.
    The pipeline produces one temporally-averaged output CSV per run,
    so each snapshot must be processed independently.
    """
    tmp = tempfile.mkdtemp(prefix='pp_single_')
    try:
        shutil.copy2(snap_path, os.path.join(tmp, os.path.basename(snap_path)))

        # Config: only this one snapshot
        with open(PP_CFG) as f:
            cfg = yaml.safe_load(f)
        cfg['SNAPSHOT_DIRECTORY'] = tmp
        cfg['output_csv'] = out_csv
        cfg['PLOT_OUTPUT_FILE'] = '/dev/null'
        cfg['ENABLE_GAUSSIAN_FIT'] = True
        cfg['ENABLE_TARGETED_SEARCH'] = True
        cfg['ENABLE_VALIDATION'] = True
        if lite:
            # Inference-only mode: skip features not in the 18-feature whitelist
            cfg['MRO_DEPTH'] = 1           # Skip depth-2/3 MRO (leakage-driven, excluded)
            cfg['COMPUTE_SECOND_LEVEL_MRO'] = False
            # Keep: ENABLE_GAUSSIAN_FIT (needs for peak2/3 features), 
            # COMPUTE_VORONOI (needs for volumes), temporal averaging
        else:
            cfg['COMPUTE_SECOND_LEVEL_MRO'] = True
            cfg['MRO_DEPTH'] = 3

        tc = os.path.join(tmp, 'config.yaml')
        with open(tc, 'w') as f:
            yaml.dump(cfg, f)

        log.info(f'Launching PP pipeline for {os.path.basename(snap_path)}...')
        r = subprocess.run(
            [sys.executable, os.path.join(PP_SRC, 'pipeline_orchestrator.py'), '--config', tc],
            capture_output=True, text=True, timeout=3600,
            cwd=os.path.join(ROOT, 'Atomic_Simulation_Post-Processing_Pipeline')
        )

        if r.returncode != 0:
            log.error(f'Pipeline failed (exit {r.returncode}): {r.stderr[:500]}')
            return False

        if os.path.exists(out_csv):
            log.info(f'Batch post-processing complete: {out_csv} ({os.path.getsize(out_csv)} bytes)')
            return True
        else:
            log.error(f'Output CSV not found after pipeline run: {out_csv}')
            return False

    except subprocess.TimeoutExpired:
        log.error('Pipeline timed out after 3600s')
        return False
    except Exception as e:
        log.error(f'Pipeline error: {e}')
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def classify(fcsv: str, model, npd: dict) -> tuple:
    """Build descriptors and apply classifier. Returns (preds, confidence)."""
    desc = build_general_descriptors(
        fcsv, structure_id='f', verbose=False,
        feature_whitelist=WHITELIST, norm_params=npd
    )
    fcols = [c for c in desc.columns if c not in ['type', 'structure_id']]
    # Build feature matrix in the exact column order the model expects
    X = np.column_stack([desc[c].values if c in desc.columns else np.full(len(desc), np.nan)
                         for c in fcols])
    preds = model.predict(X)
    probs = model.predict_proba(X)
    return preds, np.max(probs, axis=1)


def write_frame(f, hdr: dict, preds: np.ndarray, conf: np.ndarray,
                smoothed_preds: np.ndarray = None):
    """Write one LAMMPS dump frame with added PhaseID, PhaseConfidence, 
    and SmoothedPhaseID columns."""
    lines = hdr['raw_lines']
    cols = hdr['columns']
    ds = hdr['data_start']
    n = hdr['natoms']
    ts = hdr['timestep']

    f.write(f'ITEM: TIMESTEP\n{ts}\nITEM: NUMBER OF ATOMS\n{n}\n')
    bb = hdr['box_bounds']
    f.write(f'ITEM: BOX BOUNDS {" ".join(bb["tilt"]) if bb["tilt"] else "pp pp pp"}\n')
    for b in bb['bounds']:
        f.write(f'{b[0]:.10f} {b[1]:.10f}\n')
    
    new_cols = cols + ["PhaseID", "PhaseConfidence"]
    if smoothed_preds is not None:
        new_cols.append("SmoothedPhaseID")
    f.write(f'ITEM: ATOMS {" ".join(new_cols)}\n')
    
    for i in range(n):
        line = f'{lines[ds + i].strip()} {preds[i]} {conf[i]:.6f}'
        if smoothed_preds is not None:
            line += f' {smoothed_preds[i]}'
        f.write(line + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Phase-label a multi-frame LAMMPS trajectory using the trained classifier.')
    parser.add_argument('--snapshots', '-s', type=str, default=None,
                        help='Glob pattern or directory of .lammpstrj files to process. '
                             'Default: all .lammpstrj in data/polyamorphous/')
    parser.add_argument('--lite', action='store_true',
                        help='Inference-only mode: skip MRO depth-2/3 features '
                             'not used by the classifier. Reduces compute per frame.')
    parser.add_argument('--n-workers', type=int, default=1,
                        help='Number of frames to process in parallel (default=1). '
                             'Reduces wall-clock time, not total compute.')
    args = parser.parse_args()

    # Resolve snapshot files
    if args.snapshots:
        if os.path.isdir(args.snapshots):
            snap_files = sorted(glob.glob(os.path.join(args.snapshots, '*.lammpstrj')),
                                key=ts_from_name)
        else:
            snap_files = sorted(glob.glob(args.snapshots), key=ts_from_name)
    else:
        snap_files = sorted(glob.glob(os.path.join(DEFAULT_POLY, '*.lammpstrj')),
                            key=ts_from_name)

    if not snap_files:
        log.error(f'No .lammpstrj files found. Check --snapshots path.')
        sys.exit(1)

    log.info(f'Found {len(snap_files)} snapshot files to process')
    for sf in snap_files:
        log.info(f'  {os.path.basename(sf)} (ts={ts_from_name(sf)})')

    # Load model and normalization parameters (once)
    log.info('Loading trained model and norm_params...')
    model, training_meta = load_model_with_metadata(MODEL, verbose=True)
    npd = joblib.load(NP)
    frd = joblib.load(FR)
    log.info(f'Phase mapping (FIXED across all frames): {PHASE_MAP}')
    
    # Check consistency between training-time settings and current inference config
    # We don't know the inference feature order yet (it depends on the specific CSV),
    # but the feature_order in the model metadata will be checked per-frame below.
    warnings = check_inference_consistency(training_meta, {}, verbose=True)
    if warnings:
        log.warning(f'{len(warnings)} configuration mismatch(es) detected!')
        for w in warnings:
            log.warning(f'  {w}')

    # Create output directory
    os.makedirs(OUT, exist_ok=True)

    # Save phase mapping once
    with open(os.path.join(OUT, 'phase_id_to_name.json'), 'w') as f:
        json.dump(PHASE_MAP, f, indent=2)

    # -----------------------------------------------------------------------
    # Per-snapshot post-processing + classification
    # -----------------------------------------------------------------------
    # Determine per-frame work
    frame_args = [(dp, os.path.join(OUT, f'_features_{ts_from_name(dp)}.csv'),
                   args.lite) for dp in snap_files]

    if args.n_workers > 1:
        # Parallel execution: reduces wall-clock time only
        log.info(f'Processing {len(frame_args)} frames with {args.n_workers} workers (parallel)')
        from multiprocessing import Pool
        with Pool(args.n_workers) as pool:
            success_flags = pool.starmap(run_pp_single, frame_args)
    else:
        # Sequential execution
        success_flags = []
        for dp, fcsv, lite in frame_args:
            success_flags.append(run_pp_single(dp, fcsv, lite))

    combined_path = os.path.join(OUT, 'phase_labeled_trajectory.lammpstrj')
    rows = []

    with open(combined_path, 'w') as of:
        for idx, dp in enumerate(snap_files):
            if not success_flags[idx]:
                log.error(f'Skipping ts={ts_from_name(dp)} due to PP failure')
                continue
            
            ts = ts_from_name(dp)
            fcsv = os.path.join(OUT, f'_features_{ts}.csv')
            log.info(f'[{idx+1}/{len(snap_files)}] Classifying ts={ts}...')

            # Step 2: Classify
            preds, conf = classify(fcsv, model, npd)
            
            # Check feature ORDER against training metadata
            desc = build_general_descriptors(fcsv, structure_id='_check', verbose=False,
                feature_whitelist=WHITELIST, norm_params=npd)
            inference_order = [c for c in desc.columns if c not in ['type', 'structure_id']]
            order_warnings = check_inference_consistency(training_meta, {}, 
                inference_feature_order=inference_order, verbose=False)
            if order_warnings:
                for w in order_warnings:
                    log.warning(f'FEATURE ORDER CHECK for ts={ts}: {w}')

            # Step 3: OOD check
            raw = pd.read_csv(fcsv)
            ood = []
            for c in npd:
                if c in raw.columns:
                    tmin, tmax = raw[c].min(), raw[c].max()
                    rmin, rmax = frd[c]['min'], frd[c]['max']
                    margin = 0.2
                    if tmin < rmin - margin * abs(rmin) or tmax > rmax + margin * abs(rmax):
                        ood.append(f'{c}: t=[{tmin:.1f},{tmax:.1f}] r=[{rmin:.1f},{rmax:.1f}]')
            if ood:
                log.warning(f'OOD warnings for ts={ts}:')
                for w in ood:
                    log.warning(f'  {w}')
            else:
                log.info(f'No OOD warnings for ts={ts}')

            # Step 4: Write frame
            hdr = read_header(dp)
            write_frame(of, hdr, preds, conf)

            # Step 5: Phase fractions
            total = len(preds)
            fracs = {PHASE_MAP[ci]: (preds == ci).sum() / total * 100 for ci in range(3)}
            unc = (conf < 0.50).sum() / total * 100
            rows.append({
                'timestep': ts, 'natoms': total,
                'Cu50Zr50_pct': round(fracs['Cu50Zr50'], 1),
                'Cu46Zr54_pct': round(fracs['Cu46Zr54'], 1),
                'Cu64Zr36_pct': round(fracs['Cu64Zr36'], 1),
                'Uncertain_pct': round(unc, 1)
            })
            log.info(f'  Cu50Zr50={fracs["Cu50Zr50"]:.1f}%  '
                     f'Cu46Zr54={fracs["Cu46Zr54"]:.1f}%  '
                     f'Cu64Zr36={fracs["Cu64Zr36"]:.1f}%  '
                     f'Uncertain={unc:.1f}%')

            # Clean up temp features CSV
            if os.path.exists(fcsv):
                os.remove(fcsv)

    # Save summary
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(OUT, 'per_frame_summary.csv'), index=False)

    log.info(f'\n{"=" * 60}')
    log.info(f'Trajectory processing complete!')
    log.info(f'Combined dump: {combined_path}')
    log.info(f'Phase mapping: {os.path.join(OUT, "phase_id_to_name.json")}')
    log.info(f'Per-frame summary: {os.path.join(OUT, "per_frame_summary.csv")}')
    log.info(f'Total frames: {len(rows)}')
    log.info(f'Phase ID mapping is FIXED across all frames: {PHASE_MAP}')
    log.info(f'{"=" * 60}')

    print('\nPer-frame phase fraction summary:')
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
