#!/usr/bin/env python3
"""
Deprecated wrapper. The old version of this script hard-coded results from an
earlier model run (e.g. importances from the previous feature set). All figures
are now generated from the current data files by scripts/generate_dark_figures.py,
which reads predictions.csv, docs/assets/validation_summary.json,
docs/assets/dashboard_real_data.json and general_phase_id_output/permutation_importance.json.

Run from the repository root:
    python scripts/generate_dark_figures.py
"""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TARGET = os.path.join(ROOT, 'scripts', 'generate_dark_figures.py')

if __name__ == '__main__':
    sys.exit(subprocess.call([sys.executable, TARGET], cwd=ROOT))
