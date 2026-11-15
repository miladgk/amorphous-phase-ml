# Atomic Simulation Post-Processing Pipeline

Computes per-atom structural descriptors from LAMMPS trajectory dumps: Voronoi volumes and coordination,
Steinhardt q4/q6, radial distribution functions (RDF) with peak analysis, stress and energy averages,
and temporal/spatial averages across snapshots. Its output CSVs are the inputs of the ML pipeline in
`../Machine_Learning_Pipeline_for_Materials_Science/`.

The 17 features used by the deployed model are listed in `../PIPELINE.md` and in
`../Machine_Learning_Pipeline_for_Materials_Science/src/general_phase_id/ref_config.yaml`.

> **Note:** `src/spatial_analysis_levels.py` is not in the current tree. It was removed in commit `eb06523`
> and is needed for `ANALYSIS_LEVEL: temporal_first_spatial`, which produced the current feature tables.
> Restore it with:
> `git show eb06523^:Atomic_Simulation_Post-Processing_Pipeline/src/spatial_analysis_levels.py > src/spatial_analysis_levels.py`
> (`temporal_only` runs without it). See `outputs/README_recomputed.md`.

---

## Features

- Parallel processing of snapshots (`concurrent.futures`, chunked; `WORKER_TIMEOUT` in `config.yaml`).
- Voronoi tessellation with periodic boundaries (`voronoi.py`, PyVoro; freud for neighbour and bond-order calculations).
- Steinhardt bond-order parameters q4 and q6 (`snapshot_processor.py`, cutoff 3.5 Å).
- RDF per atom with Savitzky–Golay smoothing, global and targeted peak search, Gaussian fits and validation against crystallographic ratios (`rdf.py`, `peak_analysis.py`).
- Temporal averaging over snapshots (`temporal_averaging.py`) and spatial averaging over Voronoi neighbours (`ANALYSIS_LEVEL: temporal_first_spatial`).
- Medium-range order (MRO) and CSRO features are computed (`COMPUTE_SECOND_LEVEL_MRO: true`). The deployed model excludes them because they leak simulation-box identity (see `../PIPELINE.md` §8.2).

---

## Project Files

| File | Description |
|------|-------------|
| `src/pipeline_orchestrator.py` | Entry point: loads the configuration, processes snapshots, averages, runs peak analysis and writes the CSV. |
| `src/snapshot_processor.py` | Per-snapshot processing: parses dumps, Voronoi, RDF, q4/q6. |
| `src/io_module.py` | LAMMPS dump reader (box bounds, types, radii; radii come from `atomic_radii` in the config). |
| `src/voronoi.py` | Weighted Voronoi tessellation in periodic domains (PyVoro). |
| `src/rdf.py` | Per-atom RDF with KD-tree neighbour search and minimum-image convention. |
| `src/peak_analysis.py` | RDF smoothing, peak detection, Gaussian fits and ratio validation. |
| `src/temporal_averaging.py` | Averages per-snapshot values per atom. |
| `src/implement_s2_isb.py` | S2 pair-entropy and ISB feature functions. The script also contains an older binary-training routine that is not part of the current pipeline. |
| `config.yaml` | All pipeline parameters: analysis level, peak thresholds, atomic radii, snapshot directory, output CSV. |
| `config/freud.ini` | freud library settings. |
| `files/snapshots_{5050,4654,6436}/` | Reference LAMMPS dumps (11 snapshots each). |
| `outputs/` | Feature tables and plots. See `outputs/README_recomputed.md` and `../DATA.md`. |

Dependencies: `requirements.txt` (and `../environment.yml`). Verified versions for the current tables: freud-analysis 3.2, pyvoro 1.3.2.

---

## Usage

Run from this folder. The `SNAPSHOT_DIRECTORY` and `output_csv` paths in `config.yaml` are resolved relative to the config file.

```bash
cd Atomic_Simulation_Post-Processing_Pipeline
python src/pipeline_orchestrator.py --config config.yaml
```

To compute one reference dataset, set `SNAPSHOT_DIRECTORY` (e.g. `files/snapshots_5050/`), `output_csv`, and `composition.type_1_fraction` / `type_2_fraction` in a copy of `config.yaml`. The settings used for the current tables are in `outputs/README_recomputed.md`.

The worker count is taken from `multiprocessing.cpu_count()`. Runtime: about 35 minutes for the 36,828-atom target with one worker. Eight workers ran out of memory on a 7 GB machine, so limit the number of CPUs available to the process on small machines, and raise `WORKER_TIMEOUT` for the large target (`outputs/README_recomputed.md`).

---

## License

MIT (see the root `LICENSE`).
