"""area_law_data.py - data for the area-law plot (Fig. 2 of the paper): for every k and y_0, the excess area
dA = A(k) - A(0) of the extremal surface, the turning point v_c(y_0), and the transverse area of the slice
v = v_c(y_0) over the region, A_T = l^3 (B(v_c)/v_c^2)^(3/2) V_com(y_0). Uses the solver of ER_bridge.py."""
import sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from ER_bridge import area_reg, v_horizon

rows = []
start = time.time()
for k in (0.2, 0.5, 1.0):
    v1 = v_horizon(k)
    for y0 in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        area_k, vc = area_reg(y0, k)
        area_0, _ = area_reg(y0, 0.0)
        volume = 2 * np.pi * (np.sinh(y0) * np.cosh(y0) - y0)          # comoving volume, 4 pi times the integral of sinh^2 y from 0 to y_0
        transverse = ((1 + k * vc**4) / vc**2) ** 1.5 * volume           # slice v = v_c(y_0)
        horizon = ((1 + k * v1**4) / v1**2) ** 1.5 * volume              # slice v = v_1
        rows.append({"k": k, "y0": y0, "v_c": vc, "v_1": v1, "dA": area_k - area_0, "V_com": volume,
                     "A_T": transverse, "A_horizon": horizon})
        print(f"k={k} y0={y0}: v_c={vc:.6f} dA={area_k - area_0:.3f} A_T={transverse:.3f} "
              f"({time.time() - start:.0f} s)", flush=True)
pd.DataFrame(rows).to_csv("area_law_data.csv", index=False)
