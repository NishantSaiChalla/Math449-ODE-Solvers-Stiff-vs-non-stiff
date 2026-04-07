"""
Console table printer and CSV exporter for benchmark results.

References:
  Ashino, Nagase & Vaillancourt (2000). Behind and Beyond the MATLAB ODE Suite.
    Comput. Math. Appl., 40, 491-512.
"""

from pathlib import Path
import pandas as pd
from tabulate import tabulate


def print_summary(comparison_df: pd.DataFrame, output_dir: Path):
    """Print a formatted results table and export to results_summary.csv."""
    rows = []
    for _, r in comparison_df.iterrows():
        rows.append({
            "Problem":  r["problem"],
            "Solver":   r["solver"],
            "RelTol":   f"{r['rtol']:.0e}",
            "Steps":    int(r["steps"])        if pd.notna(r["steps"])   else "—",
            "nfev":     int(r["nfev"])         if pd.notna(r["nfev"])    else "—",
            "Time (s)": f"{r['time_s']:.4f}"  if pd.notna(r["time_s"])  else "—",
            "Status":   r["status"],
        })

    display_df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(tabulate(display_df, headers="keys", tablefmt="grid", showindex=False))
    print()

    csv_path = output_dir / "results_summary.csv"
    display_df.to_csv(csv_path, index=False)
    print(f"  Results saved to: {csv_path}")
