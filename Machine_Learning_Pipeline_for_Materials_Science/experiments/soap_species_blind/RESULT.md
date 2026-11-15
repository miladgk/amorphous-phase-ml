# Experimental result: species-blind SOAP (NOT adopted)

Status: exploratory. Not part of the production 17-feature model.

## Setup
- DScribe SOAP, species-blind (one species channel), periodic cell from the dump bounds,
  r_cut = 5.0 Å, n_max = 8, l_max = 6, sigma = 0.4 Å → 252 components per atom.
- Averaged over snapshots per atom id: 11 frames for references, 9 for the target.
- Model: HistGB, same spatial holdout as production (|x − median| < 3.5 Å removed).
- Timing between snapshots: timestep index step 1e6 with dt = 2 fs, so frames are ~2 ns apart.
  This is far above the picosecond vibrational regime, so dynamic descriptors were not used.

## Results (spatial holdout, test n = 16,227)

| Model | CV macro F1 | Holdout acc | F1 Cu50 | F1 Cu46 | F1 Cu64 |
|---|---|---|---|---|---|
| Production: 17 features | 0.670 | 67.8 % | 0.504 | 0.590 | 0.900 |
| 17 + SOAP | 0.776 | 77.3 % | 0.644 | 0.657 | 0.982 |
| 17 + SOAP (L2-normalised per atom) | 0.778 | 77.5 % | 0.647 | 0.662 | 0.983 |
| SOAP only | 0.774 | 77.2 % | 0.642 | 0.657 | 0.981 |

## Interpretation and why this is not adopted
1. SOAP alone reproduces the gain; the 17 production features add essentially nothing on top.
2. L2 normalisation does not remove the gain, so it is not a simple density-scale effect.
3. Confound not excluded: each reference phase is one simulation box. Box densities differ:
   Cu50 0.0569, Cu46 0.0556, Cu64 0.0628 atoms/Å³ (target 0.0585). A holdout that splits atoms
   within one box cannot tell phase structure from a box-specific signature. The same failure
   previously caused the MRO leak documented in PROJECT_BRIEF §8.2.
4. Target sanity check (no ground truth for Cu50/Cu46): in x ≥ 130 Å the SOAP model assigns
   97.9 % Cu64 vs 92.6 % for the production model. Consistent with a real Cu64 signal, but not
   evidence for the Cu50/Cu46 split.

Conclusion: SOAP is a promising candidate. The Cu50/Cu46 improvement is not established until
it survives a test across independent simulation boxes (not available at present).

## Files
- Scripts: soap_frame.py, run_soap.py (this folder). Paths inside run_soap.py are absolute to the workstation.
- Averaged SOAP vectors and training logs: /home/m/claude_work/soap and /home/m/claude_work/soapml (not in repo, ~100 MB).
- Repo change accompanying this experiment: build_descriptors.py passes through columns starting with `soap_`
  (in `_is_already_dimensionless`). Production config does not whitelist them.
