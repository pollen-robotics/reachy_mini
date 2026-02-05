#!/usr/bin/env python3
"""
Analyze Real Robot Test Data

This script analyzes the time-series data collected from real robot pose tests.
It creates visualizations showing:
- Current over time for each motor
- Torque over time for each motor
- Maximum torque vs payload weight
- Motor comparison charts

Usage:
    python analyze_real_robot_data.py <csv_file>
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Safety limits (same as in test script)
STALL_TORQUE_NM = 0.6
SAFE_LIMIT_NM = 0.45


def plot_time_series_for_test(df, pose_key, payload_g, output_dir):
    """Plot time-series data for a specific test."""
    # Filter data for this test
    test_data = df[(df['pose_key'] == pose_key) & (df['payload_g'] == payload_g)]

    if test_data.empty:
        print(f"No data found for {pose_key} with {payload_g}g")
        return

    pose_name = test_data['pose_name'].iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Current over time
    ax = axes[0]
    for i in range(1, 7):
        col = f'current_m{i}_ma'
        if col in test_data.columns:
            ax.plot(test_data['sample_time_s'], test_data[col],
                   label=f'Motor {i}', linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Current (mA)', fontsize=12)
    ax.set_title(f'Motor Current Over Time - {pose_name} ({payload_g}g)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', ncol=3)
    ax.grid(alpha=0.3)

    # Plot 2: Torque over time
    ax = axes[1]
    for i in range(1, 7):
        col = f'torque_m{i}_nm'
        if col in test_data.columns:
            ax.plot(test_data['sample_time_s'], test_data[col],
                   label=f'Motor {i}', linewidth=1.5, alpha=0.8)

    ax.axhline(y=SAFE_LIMIT_NM, color='red', linestyle='--',
              linewidth=2, label=f'Safe Limit ({SAFE_LIMIT_NM} N·m)')
    ax.axhline(y=-SAFE_LIMIT_NM, color='red', linestyle='--', linewidth=2)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Torque (N·m)', fontsize=12)
    ax.set_title(f'Motor Torque Over Time - {pose_name} ({payload_g}g)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', ncol=3)
    ax.grid(alpha=0.3)

    # Plot 3: Maximum absolute values over time
    ax = axes[2]
    ax.plot(test_data['sample_time_s'], test_data['max_abs_current_ma'],
           label='Max Current (mA)', linewidth=2, color='blue', alpha=0.7)

    # Add second y-axis for torque
    ax2 = ax.twinx()
    ax2.plot(test_data['sample_time_s'], test_data['max_abs_torque_nm'],
            label='Max Torque (N·m)', linewidth=2, color='red', alpha=0.7)
    ax2.axhline(y=SAFE_LIMIT_NM, color='darkred', linestyle='--',
               linewidth=2, label=f'Safe Limit')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Max Current (mA)', fontsize=12, color='blue')
    ax2.set_ylabel('Max Torque (N·m)', fontsize=12, color='red')
    ax.set_title(f'Maximum Values Over Time - {pose_name} ({payload_g}g)',
                fontsize=14, fontweight='bold')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save figure
    filename = f"{pose_key}_{payload_g}g_timeseries.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {filepath}")


def plot_torque_vs_payload(df, pose_key, output_dir):
    """Plot maximum torque vs payload weight for a specific pose."""
    pose_data = df[df['pose_key'] == pose_key]

    if pose_data.empty:
        print(f"No data found for pose: {pose_key}")
        return

    pose_name = pose_data['pose_name'].iloc[0]

    # Group by payload and compute max torque
    payload_summary = pose_data.groupby('payload_g').agg({
        'max_abs_torque_nm': 'max',
        'max_abs_current_ma': 'max',
    }).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Torque vs Payload
    ax1.plot(payload_summary['payload_g'], payload_summary['max_abs_torque_nm'],
            'o-', linewidth=2, markersize=8, color='blue', label='Max Torque')
    ax1.axhline(y=SAFE_LIMIT_NM, color='red', linestyle='--',
               linewidth=2, label=f'Safe Limit ({SAFE_LIMIT_NM} N·m)')
    ax1.axhline(y=STALL_TORQUE_NM, color='darkred', linestyle='--',
               linewidth=2, label=f'Stall ({STALL_TORQUE_NM} N·m)')

    ax1.set_xlabel('Payload (g)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Max Torque (N·m)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Torque vs Payload - {pose_name}', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Plot 2: Current vs Payload
    ax2.plot(payload_summary['payload_g'], payload_summary['max_abs_current_ma'],
            'o-', linewidth=2, markersize=8, color='green', label='Max Current')

    ax2.set_xlabel('Payload (g)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Max Current (mA)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Current vs Payload - {pose_name}', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save figure
    filename = f"{pose_key}_torque_vs_payload.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {filepath}")


def plot_all_poses_comparison(df, output_dir):
    """Compare maximum torque across all poses and payloads."""
    # Compute max torque for each pose/payload combination
    summary = df.groupby(['pose_key', 'pose_name', 'payload_g']).agg({
        'max_abs_torque_nm': 'max',
    }).reset_index()

    # Get unique payloads
    payloads = sorted(summary['payload_g'].unique())

    fig, axes = plt.subplots(len(payloads), 1, figsize=(14, 4 * len(payloads)))

    if len(payloads) == 1:
        axes = [axes]

    for ax, payload in zip(axes, payloads):
        payload_data = summary[summary['payload_g'] == payload].sort_values('max_abs_torque_nm', ascending=False)

        bars = ax.barh(payload_data['pose_name'], payload_data['max_abs_torque_nm'],
                      color='steelblue', edgecolor='black', alpha=0.7)

        # Color bars based on safety
        for i, torque in enumerate(payload_data['max_abs_torque_nm']):
            if torque >= SAFE_LIMIT_NM:
                bars[i].set_color('red')
            elif torque >= SAFE_LIMIT_NM * 0.8:
                bars[i].set_color('orange')

        ax.axvline(x=SAFE_LIMIT_NM, color='red', linestyle='--',
                  linewidth=2, label=f'Safe Limit ({SAFE_LIMIT_NM} N·m)')
        ax.axvline(x=STALL_TORQUE_NM, color='darkred', linestyle='--',
                  linewidth=2, label=f'Stall ({STALL_TORQUE_NM} N·m)')

        ax.set_xlabel('Max Torque (N·m)', fontsize=12, fontweight='bold')
        ax.set_title(f'Torque Comparison - {payload}g Payload', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.legend()

    plt.tight_layout()

    # Save figure
    filename = "all_poses_comparison.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {filepath}")


def generate_summary_report(df, output_dir):
    """Generate a text summary report."""
    report = []
    report.append("=" * 80)
    report.append("REAL ROBOT TEST SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")

    # Overall statistics
    report.append("OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"Total samples collected: {len(df)}")
    report.append(f"Total tests: {df.groupby(['pose_key', 'payload_g']).ngroups}")
    report.append(f"Poses tested: {df['pose_key'].nunique()}")
    report.append(f"Payload weights tested: {sorted(df['payload_g'].unique())}")
    report.append("")

    # Maximum values
    max_row = df.loc[df['max_abs_torque_nm'].idxmax()]
    report.append("MAXIMUM TORQUE OBSERVED")
    report.append("-" * 80)
    report.append(f"Pose: {max_row['pose_name']}")
    report.append(f"Payload: {max_row['payload_g']}g")
    report.append(f"Torque: {max_row['max_abs_torque_nm']:.4f} N·m ({max_row['max_abs_torque_nm']/STALL_TORQUE_NM*100:.1f}% of stall)")
    report.append(f"Current: {max_row['max_abs_current_ma']:.1f} mA")
    report.append(f"Time: {max_row['sample_time_s']:.1f}s into test")
    report.append("")

    # Safety violations
    violations = df[df['max_abs_torque_nm'] >= SAFE_LIMIT_NM]
    if len(violations) > 0:
        report.append(f"SAFETY LIMIT VIOLATIONS ({len(violations)} samples)")
        report.append("-" * 80)
        violation_summary = violations.groupby(['pose_name', 'payload_g']).size().reset_index(name='count')
        for _, row in violation_summary.iterrows():
            report.append(f"  • {row['pose_name']} with {row['payload_g']}g: {row['count']} samples")
        report.append("")
    else:
        report.append("✓ NO SAFETY LIMIT VIOLATIONS")
        report.append("")

    # Per-pose summary
    report.append("PER-POSE MAXIMUM TORQUES")
    report.append("-" * 80)
    pose_summary = df.groupby(['pose_name', 'payload_g']).agg({
        'max_abs_torque_nm': 'max',
    }).reset_index()

    for pose in sorted(df['pose_name'].unique()):
        report.append(f"\n{pose}:")
        pose_data = pose_summary[pose_summary['pose_name'] == pose].sort_values('payload_g')
        for _, row in pose_data.iterrows():
            pct = row['max_abs_torque_nm'] / STALL_TORQUE_NM * 100
            status = "⚠" if row['max_abs_torque_nm'] >= SAFE_LIMIT_NM else "✓"
            report.append(f"  {row['payload_g']:3.0f}g: {row['max_abs_torque_nm']:.4f} N·m ({pct:5.1f}%) {status}")

    report.append("")
    report.append("=" * 80)

    # Write to file
    report_file = output_dir / "summary_report.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"✓ Saved: {report_file}")

    # Also print to console
    print("\n" + '\n'.join(report))


def main():
    """Main analysis routine."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_real_robot_data.py <csv_file>")
        print("\nExample:")
        print("  python analyze_real_robot_data.py real_robot_test_results/time_series_20240101_120000.csv")
        sys.exit(1)

    csv_file = Path(sys.argv[1])

    if not csv_file.exists():
        print(f"ERROR: File not found: {csv_file}")
        sys.exit(1)

    print("=" * 80)
    print("ANALYZING REAL ROBOT TEST DATA")
    print("=" * 80)
    print(f"\nReading data from: {csv_file}")

    # Load data
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded {len(df)} samples")
    except Exception as e:
        print(f"ERROR: Failed to load CSV: {e}")
        sys.exit(1)

    # Create output directory
    output_dir = csv_file.parent / f"analysis_{csv_file.stem}"
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Analysis outputs will be saved to: {output_dir}\n")

    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df, output_dir)

    # Plot time series for each test
    print("\nGenerating time-series plots...")
    tests = df.groupby(['pose_key', 'payload_g']).size().reset_index()[['pose_key', 'payload_g']]
    for _, row in tests.iterrows():
        plot_time_series_for_test(df, row['pose_key'], row['payload_g'], output_dir)

    # Plot torque vs payload for each pose
    print("\nGenerating torque vs payload plots...")
    for pose_key in df['pose_key'].unique():
        plot_torque_vs_payload(df, pose_key, output_dir)

    # Plot comparison across all poses
    print("\nGenerating comparison plots...")
    plot_all_poses_comparison(df, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
