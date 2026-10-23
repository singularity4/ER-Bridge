# =============================================================================
#  edge_term.py
#
#  Global ratio of the extremal-surface area to the horizon area.
#
#  For the region y < y_0 the script computes
#
#     Delta A = A_reg(k) - A_reg(0)      difference of the regularised
#                                        extremal-surface areas,
#     A_T     = s(k) V(y_0)              horizon area density times the
#                                        comoving volume of H^3,
#     V(y_0)  = 2 pi (sinh y_0 cosh y_0 - y_0),
#
#  and prints Delta A / A_T and (A_T - Delta A) / (4 pi sinh^2 y_0), the
#  difference per unit boundary area.  Since V(y_0)/(4 pi sinh^2 y_0) tends to
#  1/2 in H^3, the second quantity tends to a constant whenever the first does,
#  so it does not by itself identify an edge term.
#
#  Run:      python3 edge_term.py             (from the repository folder)
#  Requires: numpy, scipy, sympy
#  Runtime:  about 30 s on two cores
# =============================================================================
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, ".")
import ER_bridge  # noqa: E402

Y0_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
K_VALUES = [0.2, 0.5, 1.0]

# The shooting in ER_bridge.py fails for k = 1 at y_0 = 3.5.
Y0_MAX = {1.0: 3.0}


def regularised_area(case):
    y0, k = case
    return case, ER_bridge.area_reg(y0, k)[0]


def comoving_volume(y0):
    return 2 * np.pi * (np.sinh(y0) * np.cosh(y0) - y0)


def boundary_area(y0):
    return 4 * np.pi * np.sinh(y0) ** 2


if __name__ == "__main__":
    cases = [(y0, k) for y0 in Y0_VALUES for k in [0.0] + K_VALUES if y0 <= Y0_MAX.get(k, y0)]
    with Pool(2) as pool:
        areas = dict(pool.map(regularised_area, cases))

    for k in K_VALUES:
        v_horizon = ER_bridge.v_horizon(k)
        s = ((1 + k * v_horizon**4) / v_horizon**2) ** 1.5
        print(f"\nk = {k}:  horizon v_1 = {v_horizon:.6f},  s(k) = {s:.5f}")
        print("    y_0     Delta A        A_T     Delta A/A_T   (A_T - Delta A)/(4 pi sinh^2 y_0)")
        for y0 in [y0 for y0 in Y0_VALUES if y0 <= Y0_MAX.get(k, y0)]:
            area_difference = areas[(y0, k)] - areas[(y0, 0.0)]
            horizon_area = s * comoving_volume(y0)
            print(f"   {y0:4.1f}  {area_difference:10.4f}  {horizon_area:10.4f}"
                  f"   {area_difference / horizon_area:9.5f}"
                  f"   {(horizon_area - area_difference) / boundary_area(y0):10.5f}")
