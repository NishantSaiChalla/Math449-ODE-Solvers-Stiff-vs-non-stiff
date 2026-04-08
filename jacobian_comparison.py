"""
Jacobian supply benchmark on the Robertson problem.

Compares three solver configurations at medium tolerance (rtol=1e-6, atol=1e-9):
  (a) Radau, no Jacobian supplied
  (b) Radau, analytical Jacobian supplied
  (c) BDF,   analytical Jacobian supplied

References:
  Shampine & Reichelt (1997). The MATLAB ODE Suite. SIAM J. Sci. Comput., 18(1), 1-22.
  Ashino, Nagase & Vaillancourt (2000). Behind and Beyond the MATLAB ODE Suite.
    Comput. Math. Appl., 40, 491-512.
"""

import time
import pandas as pd
from scipy.integrate import solve_ivp

from problems import robertson, jac_robertson, ROB_Y0, ROB_T_SPAN, ROB_T_EVAL

RTOL = 1e-6
ATOL = 1e-9

CONFIGS = [
    {"label": "No Jacobian",    "method": "Radau", "jac": None},
    {"label": "Analytical Jac", "method": "Radau", "jac": jac_robertson},
    {"label": "BDF + Jac",      "method": "BDF",   "jac": jac_robertson},
]


def run_jacobian_comparison():
    """Solve the Robertson problem under three Jacobian configurations.

    Returns a DataFrame with columns: label, method, nfev, njev, time_s, status.
    njev (Jacobian evaluations) is the primary metric of interest here.
    """
    records = []

    for cfg in CONFIGS:
        print(f"    Robertson | {cfg['method']:6s} | {cfg['label']:16s} ...", end="", flush=True)

        kwargs = dict(method=cfg["method"], t_eval=ROB_T_EVAL,
                      rtol=RTOL, atol=ATOL, dense_output=False)
        if cfg["jac"] is not None:
            kwargs["jac"] = cfg["jac"]

        try:
            t0  = time.perf_counter()
            res = solve_ivp(robertson, ROB_T_SPAN, ROB_Y0, **kwargs)
            t1  = time.perf_counter()

            status = "success" if res.success else "failed"
            nfev   = res.nfev
            njev   = getattr(res, "njev", None)
            time_s = round(t1 - t0, 6)

        except Exception as exc:
            status = f"error: {exc}"
            nfev = njev = time_s = None

        print(f" {status}  (nfev={nfev}, {time_s:.3f}s)" if time_s else f" {status}")

        records.append({
            "label":  cfg["label"],
            "method": cfg["method"],
            "nfev":   nfev,
            "njev":   njev,
            "time_s": time_s,
            "status": status,
        })

    return pd.DataFrame(records)
