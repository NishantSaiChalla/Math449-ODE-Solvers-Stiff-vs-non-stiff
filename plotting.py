"""
Figure generation for the ODE solver benchmarking project.

Saves seven PNG figures to the figures/ directory.

References:
  Shampine & Reichelt (1997). The MATLAB ODE Suite. SIAM J. Sci. Comput., 18(1), 1-22.
  Ashino, Nagase & Vaillancourt (2000). Behind and Beyond the MATLAB ODE Suite.
    Comput. Math. Appl., 40, 491-512.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

from problems import (
    lotka_volterra, LV_Y0, LV_T_SPAN, LV_T_EVAL,
    robertson,      ROB_Y0, ROB_T_SPAN, ROB_T_EVAL,
)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")

COLOR_RK45  = "#2196F3"
COLOR_RADAU = "#E53935"
COLOR_BDF   = "#43A047"

DPI       = 150
FIGSIZE   = (10, 6)
FIGSIZE_7 = (12, 5)
LABEL_FS  = 12
TITLE_FS  = 14
TICK_FS   = 10

TOL_LABELS = ["Loose", "Medium", "Tight"]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _bar_labels(ax, bars, fmt="{:.0f}"):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2.0, h * 1.05,
                    fmt.format(h), ha="center", va="bottom", fontsize=8)


def fig1_lotka_timeseries(figures_dir):
    res = solve_ivp(lotka_volterra, LV_T_SPAN, LV_Y0,
                    method="RK45", t_eval=LV_T_EVAL, rtol=1e-6, atol=1e-9)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(res.t, res.y[0], color=COLOR_RK45,  lw=1.5, label="Prey ($y_1$)")
    ax.plot(res.t, res.y[1], color=COLOR_RADAU, lw=1.5, label="Predator ($y_2$)")
    ax.set_xlabel("Time", fontsize=LABEL_FS)
    ax.set_ylabel("Population", fontsize=LABEL_FS)
    ax.set_title("Lotka-Volterra: Time Series (RK45)", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=TICK_FS)
    ax.grid(True)
    _save(fig, figures_dir / "fig1_lotka_timeseries.png")


def fig2_lotka_phase(figures_dir):
    res = solve_ivp(lotka_volterra, LV_T_SPAN, LV_Y0,
                    method="RK45", t_eval=LV_T_EVAL, rtol=1e-6, atol=1e-9)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(res.y[0], res.y[1], color=COLOR_RK45, lw=1.2, alpha=0.8)
    ax.plot(res.y[0, 0], res.y[1, 0], "o", color=COLOR_RADAU,
            markersize=8, label="Start $(y_1^0, y_2^0)$", zorder=5)
    ax.set_xlabel("Prey ($y_1$)", fontsize=LABEL_FS)
    ax.set_ylabel("Predator ($y_2$)", fontsize=LABEL_FS)
    ax.set_title("Lotka-Volterra: Phase Portrait (RK45)", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=TICK_FS)
    ax.grid(True)
    _save(fig, figures_dir / "fig2_lotka_phase.png")


def fig3_robertson_solution(figures_dir):
    res = solve_ivp(robertson, ROB_T_SPAN, ROB_Y0,
                    method="Radau", t_eval=ROB_T_EVAL, rtol=1e-6, atol=1e-9)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogx(res.t, res.y[0],       color=COLOR_RK45,  lw=1.5, label="$y_1$")
    ax.semilogx(res.t, res.y[1] * 1e4, color=COLOR_RADAU, lw=1.5, label=r"$y_2 \times 10^4$")
    ax.semilogx(res.t, res.y[2],       color=COLOR_BDF,   lw=1.5, label="$y_3$")
    ax.set_xlabel("Time (log scale)", fontsize=LABEL_FS)
    ax.set_ylabel("Concentration", fontsize=LABEL_FS)
    ax.set_title("Robertson Chemical Kinetics: Solution (Radau)", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=TICK_FS)
    ax.grid(True)
    _save(fig, figures_dir / "fig3_robertson_solution.png")


def _grouped_bar_chart(df, metric, ylabel, title, log_scale, figures_dir, fname):
    problems = ["Lotka-Volterra", "Robertson"]
    solvers  = ["RK45", "Radau"]
    colors   = {"RK45": COLOR_RK45, "Radau": COLOR_RADAU}

    group_labels = [f"{'LV' if 'Lotka' in p else 'Rob'}\n{t}"
                    for p in problems for t in TOL_LABELS]

    x         = np.arange(len(group_labels))
    bar_width = 0.35
    fig, ax   = plt.subplots(figsize=FIGSIZE)

    for i, solver in enumerate(solvers):
        values = []
        for prob in problems:
            for tol in TOL_LABELS:
                row = df[(df["problem"] == prob) & (df["solver"] == solver) & (df["tol_label"] == tol)]
                val = row.iloc[0][metric] if not row.empty else None
                values.append(float(val) if val is not None and val == val else 0.0)

        offset = (i - 0.5) * bar_width
        bars   = ax.bar(x + offset, values, bar_width,
                        label=solver, color=colors[solver], alpha=0.85,
                        edgecolor="white", linewidth=0.5)
        _bar_labels(ax, bars)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=TICK_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.legend(fontsize=TICK_FS)
    ax.grid(True, axis="y", alpha=0.4)
    _save(fig, figures_dir / fname)


def fig4_steps_comparison(df, figures_dir):
    _grouped_bar_chart(df, "steps", "Accepted Steps (log scale)",
                       "Accepted Steps: RK45 vs Radau", True,
                       figures_dir, "fig4_steps_comparison.png")


def fig5_nfevals_comparison(df, figures_dir):
    _grouped_bar_chart(df, "nfev", "RHS Evaluations (log scale)",
                       "Function Evaluations: RK45 vs Radau", True,
                       figures_dir, "fig5_nfevals_comparison.png")


def fig6_time_comparison(df, figures_dir):
    _grouped_bar_chart(df, "time_s", "Wall-Clock Time (s)",
                       "Wall-Clock Time: RK45 vs Radau", False,
                       figures_dir, "fig6_time_comparison.png")


def fig7_jacobian_comparison(jac_df, figures_dir):
    labels = jac_df["label"].tolist()
    njev   = jac_df["njev"].fillna(0).astype(float).tolist()
    times  = jac_df["time_s"].fillna(0).astype(float).tolist()
    colors = [COLOR_RADAU, COLOR_RK45, COLOR_BDF]
    x      = np.arange(len(labels))
    bw     = 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_7)

    bars1 = ax1.bar(x, njev, bw, color=colors, edgecolor="white", linewidth=0.5)
    _bar_labels(ax1, bars1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=TICK_FS)
    ax1.set_ylabel("Jacobian Evaluations (njev)", fontsize=LABEL_FS)
    ax1.set_title("Jacobian Evaluations", fontsize=TITLE_FS)
    ax1.tick_params(axis="y", labelsize=TICK_FS)
    ax1.grid(True, axis="y", alpha=0.4)

    bars2 = ax2.bar(x, times, bw, color=colors, edgecolor="white", linewidth=0.5)
    _bar_labels(ax2, bars2, fmt="{:.3f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=TICK_FS)
    ax2.set_ylabel("Wall-Clock Time (s)", fontsize=LABEL_FS)
    ax2.set_title("Computation Time", fontsize=TITLE_FS)
    ax2.tick_params(axis="y", labelsize=TICK_FS)
    ax2.grid(True, axis="y", alpha=0.4)

    fig.suptitle("Robertson Problem: Jacobian Supply Comparison (Radau / BDF)",
                 fontsize=TITLE_FS, y=1.01)
    _save(fig, figures_dir / "fig7_jacobian_comparison.png")


def generate_all_figures(comparison_df, jacobian_df, figures_dir: Path):
    """Generate and save all seven benchmark figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("  fig1: Lotka-Volterra time series ...")
    fig1_lotka_timeseries(figures_dir)

    print("  fig2: Lotka-Volterra phase portrait ...")
    fig2_lotka_phase(figures_dir)

    print("  fig3: Robertson solution ...")
    fig3_robertson_solution(figures_dir)

    print("  fig4: Steps comparison ...")
    fig4_steps_comparison(comparison_df, figures_dir)

    print("  fig5: Function evaluations comparison ...")
    fig5_nfevals_comparison(comparison_df, figures_dir)

    print("  fig6: Time comparison ...")
    fig6_time_comparison(comparison_df, figures_dir)

    print("  fig7: Jacobian comparison ...")
    fig7_jacobian_comparison(jacobian_df, figures_dir)
