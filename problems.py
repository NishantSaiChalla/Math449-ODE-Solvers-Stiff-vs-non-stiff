"""
ODE problem definitions: Lotka-Volterra (non-stiff) and Robertson (stiff).

References:
  Shampine & Reichelt (1997). The MATLAB ODE Suite. SIAM J. Sci. Comput., 18(1), 1-22.
  Ashino, Nagase & Vaillancourt (2000). Behind and Beyond the MATLAB ODE Suite.
    Comput. Math. Appl., 40, 491-512.
"""

import numpy as np

# Lotka-Volterra parameters
LV_ALPHA = 1.0
LV_BETA  = 0.1
LV_DELTA = 0.075
LV_GAMMA = 0.075

LV_Y0     = [10.0, 5.0]
LV_T_SPAN = (0.0, 100.0)
LV_T_EVAL = np.linspace(0.0, 100.0, 5000)


def lotka_volterra(t, y):
    """Lotka-Volterra predator-prey system (non-stiff)."""
    y1, y2 = y
    dy1 =  LV_ALPHA * y1 - LV_BETA  * y1 * y2
    dy2 = -LV_DELTA * y2 + LV_GAMMA * y1 * y2
    return [dy1, dy2]


# Robertson rate constants
ROB_K1 = 0.04
ROB_K2 = 1e4
ROB_K3 = 3e7

ROB_Y0     = [1.0, 0.0, 0.0]
ROB_T_SPAN = (1e-5, 1e11)
ROB_T_EVAL = np.logspace(-5, 11, 500)  # log-spaced: solution evolves over 16 decades


def robertson(t, y):
    """Robertson chemical kinetics system (stiff)."""
    y1, y2, y3 = y
    dy1 = -ROB_K1 * y1 + ROB_K2 * y2 * y3
    dy2 =  ROB_K1 * y1 - ROB_K2 * y2 * y3 - ROB_K3 * y2**2
    dy3 =  ROB_K3 * y2**2
    return [dy1, dy2, dy3]


def jac_robertson(t, y):
    """Analytical Jacobian of the Robertson system."""
    y1, y2, y3 = y
    return np.array([
        [-ROB_K1,                          ROB_K2 * y3,         ROB_K2 * y2],
        [ ROB_K1, -ROB_K2 * y3 - 2 * ROB_K3 * y2,             -ROB_K2 * y2],
        [    0.0,                  2 * ROB_K3 * y2,                      0.0],
    ])
