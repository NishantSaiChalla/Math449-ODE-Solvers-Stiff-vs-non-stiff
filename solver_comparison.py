"""
Solver comparison: RK45 (ode45) vs Radau (ode15s) on two test problems.

Each run is capped at 60 s to prevent RK45 from hanging on the stiff
Robertson problem.

References:
  Shampine & Reichelt (1997). The MATLAB ODE Suite. SIAM J. Sci. Comput., 18(1), 1-22.
  Ashino, Nagase & Vaillancourt (2000). Behind and Beyond the MATLAB ODE Suite.
    Comput. Math. Appl., 40, 491-512.
"""

import time
import signal
import platform
import threading
import pandas as pd
from scipy.integrate import solve_ivp

from problems import (
    lotka_volterra, LV_Y0, LV_T_SPAN,
    robertson,      ROB_Y0, ROB_T_SPAN,
)

TOLERANCE_SETS = [
    {"label": "Loose",  "rtol": 1e-3,  "atol": 1e-6},
    {"label": "Medium", "rtol": 1e-6,  "atol": 1e-9},
    {"label": "Tight",  "rtol": 1e-9,  "atol": 1e-12},
]

SOLVERS         = ["RK45", "Radau"]
TIMEOUT_SECONDS = 60


class _TimeoutError(Exception):
    pass


def _run_with_timeout(fn, timeout):
    """Run fn() and raise _TimeoutError if it does not finish within timeout seconds."""
    if platform.system() != "Windows":
        def _handler(signum, frame):
            raise _TimeoutError()
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(timeout) + 1)
        try:
            return fn()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        result_holder = [None]
        exc_holder    = [None]

        def _target():
            try:
                result_holder[0] = fn()
            except Exception as e:
                exc_holder[0] = e

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise _TimeoutError()
        if exc_holder[0] is not None:
            raise exc_holder[0]
        return result_holder[0]


def _run_one(problem_name, problem_fn, y0, t_span, solver, tol):
    """Run a single solve_ivp benchmark and return a result dict."""
    record = {
        "problem":   problem_name,
        "solver":    solver,
        "tol_label": tol["label"],
        "rtol":      tol["rtol"],
        "atol":      tol["atol"],
        "steps":     None,
        "nfev":      None,
        "time_s":    None,
        "status":    None,
        "final_y":   None,
    }

    def _solve():
        # t_eval omitted so res.t contains every accepted step point
        return solve_ivp(
            problem_fn, t_span, y0,
            method=solver,
            rtol=tol["rtol"],
            atol=tol["atol"],
            dense_output=False,
        )

    try:
        t0  = time.perf_counter()
        res = _run_with_timeout(_solve, TIMEOUT_SECONDS)
        t1  = time.perf_counter()

        record["time_s"]  = round(t1 - t0, 6)
        record["nfev"]    = res.nfev
        record["steps"]   = res.t.size
        record["final_y"] = res.y[:, -1].tolist()
        record["status"]  = "success" if res.success else "failed"

    except _TimeoutError:
        record["status"] = "timeout"
        record["time_s"] = TIMEOUT_SECONDS

    except Exception as exc:
        record["status"] = f"error: {exc}"

    return record


def run_solver_comparison():
    """Run RK45 and Radau on both problems across all three tolerances.

    Returns a DataFrame with one row per (problem, solver, tolerance) run.
    """
    problems = [
        {"name": "Lotka-Volterra", "fn": lotka_volterra, "y0": LV_Y0,  "t_span": LV_T_SPAN},
        {"name": "Robertson",      "fn": robertson,      "y0": ROB_Y0, "t_span": ROB_T_SPAN},
    ]

    records = []
    for prob in problems:
        for solver in SOLVERS:
            for tol in TOLERANCE_SETS:
                print(f"    {prob['name']:20s} | {solver:6s} | {tol['label']:6s} ...", end="", flush=True)
                rec = _run_one(prob["name"], prob["fn"], prob["y0"], prob["t_span"], solver, tol)
                time_str = f"{rec['time_s']:.3f}s" if rec["time_s"] is not None else "N/A"
                print(f" {rec['status']}  ({time_str})")
                records.append(rec)

    return pd.DataFrame(records)
