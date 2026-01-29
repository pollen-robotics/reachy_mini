#!/usr/bin/env python3
"""Regenerate summary figure from saved results."""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from check_head_torques import create_summary_figure

# Load results
results_file = Path("head_torque_check_results/mass_test_results.npy")

if not results_file.exists():
    print(f"Error: Results file not found: {results_file}")
    print("Please run check_head_torques.py first to generate results.")
    sys.exit(1)

print("Loading results...")
all_results = np.load(results_file, allow_pickle=True)

print(f"Loaded {len(all_results)} test results")

# Create output directory
output_dir = Path("summary_reports")
output_dir.mkdir(exist_ok=True)

# Regenerate summary
output_path = output_dir / "mass_test_summary.png"
print(f"Regenerating summary figure: {output_path}")

create_summary_figure(all_results, output_path)

print(f"✓ Summary figure regenerated successfully!")
print(f"✓ Saved to: {output_path}")
