# =============================================================================
#  local_density.py
#
#  Area density of the extremal surface inside the region y < y_0.
#
#  For the bulk solution of ER_bridge.py, the extremal surface anchored at
#  y = y_0 has area
#
#     Area = 4 pi l^3 Int_0^y_0 dy sinh^2(y) sigma(y),
#     sigma = (B/v^3) sqrt( B + A v'^2/(A - v^2) ),
#
#  so sigma(y) is the area per unit comoving volume of H^3.  Where the surface
#  lies on the horizon v_1 (v -> v_1, v' -> 0), sigma equals the horizon area
#  density
#
#     s(k) = (B(v_1)/v_1^2)^(3/2).
#
#  For k = 0 the horizon is at v = 1 and s(0) = 1.  The surfaces are computed
#  with the shooting method of ER_bridge.py, which is imported unchanged.
#
#  Run:      python3 local_density.py             (from the repository folder)
#  Requires: numpy, scipy, sympy
#  Runtime:  about 30 s
# =============================================================================
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, ".")
import ER_bridge  # noqa: E402

V_CUTOFF = 1e-4

# (k, y_0).  For k = 1 the shooting in ER_bridge.py fails at y_0 = 3.5.
CASES = [(0.0, 3.5), (0.2, 3.5), (0.5, 3.5), (1.0, 3.0)]


def horizon_density(k):
    """Horizon position v_1 and horizon area density s(k) = (B(v_1)/v_1^2)^(3/2)."""
    v_horizon = ER_bridge.v_horizon(k)
    return v_horizon, ((1 + k * v_horizon**4) / v_horizon**2) ** 1.5


def area_density_profile(y0, k, y_values):
    """Depth v(y) and area density sigma(y) of the extremal surface at the given y."""
    v_turning = ER_bridge.find_vc(y0, k, V_CUTOFF)
    surface = ER_bridge._surface(v_turning, k, V_CUTOFF)
    profile = []
    for y in y_values:
        v = brentq(lambda v_trial: surface.sol(v_trial)[0] - y,
                   V_CUTOFF * 1.0001, v_turning * (1 - 1e-12))
        dv_dy = surface.sol(v)[1]
        # weight = B/v^3 and slope_factor = A/(A - v^2), as defined in ER_bridge.fns
        _, _, B, _, weight, _, slope_factor, _ = ER_bridge.fns(v, k)
        profile.append((y, v, weight * np.sqrt(B + slope_factor * dv_dy**2)))
    return v_turning, profile


if __name__ == "__main__":
    for k, y0 in CASES:
        v_horizon, s = horizon_density(k)
        y_values = [0.5, 1.0, 1.5, 2.0, 2.5] if y0 > 3 else [0.5, 1.0, 1.5, 2.0]
        v_turning, profile = area_density_profile(y0, k, y_values)
        print(f"\nk = {k}, y_0 = {y0}:  horizon v_1 = {v_horizon:.6f},  "
              f"turning point v_c = {v_turning:.6f},  s(k) = {s:.5f}")
        print("     y      v(y)      v/v_1     sigma(y)   sigma/s(k)")
        for y, v, sigma in profile:
            print(f"   {y:4.1f}   {v:.6f}   {v/v_horizon:.6f}   {sigma:9.5f}   {sigma/s:9.5f}")
