"""
Entry point for the ODE solver benchmarking project.

Usage:
    python main.py

References:
  Shampine & Reichelt (1997). The MATLAB ODE Suite. SIAM J. Sci. Comput., 18(1), 1-22.
  Ashino, Nagase & Vaillancourt (2000). Behind and Beyond the MATLAB ODE Suite.
    Comput. Math. Appl., 40, 491-512.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).parent
FIGURES_DIR = PROJECT_DIR / "figures"


def main():
    print()
    print("=" * 65)
    print("  ODE SOLVER BENCHMARKING: STIFF vs NON-STIFF")
    print("  MATH 549 — University of Victoria, 2026")
    print("  SciPy solve_ivp | RK45 (ode45) vs Radau (ode15s)")
    print("=" * 65)
    print()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Running solver comparison...")
    from solver_comparison import run_solver_comparison
    comparison_df = run_solver_comparison()

    print("\nRunning Jacobian comparison...")
    from jacobian_comparison import run_jacobian_comparison
    jacobian_df = run_jacobian_comparison()

    print("\nGenerating figures...")
    from plotting import generate_all_figures
    generate_all_figures(comparison_df, jacobian_df, FIGURES_DIR)

    print("\nPrinting summary...")
    from summary import print_summary
    print_summary(comparison_df, PROJECT_DIR)

    print()
    print(f"Done. Figures saved to {FIGURES_DIR}")
    print()


if __name__ == "__main__":
    main()
