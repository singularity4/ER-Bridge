"""area_law_plot.py - the area-law plot (Fig. 2 of the paper): excess area of the extremal surface against the
transverse area of the slice v = v_c(y_0) over the same region. Equal areas (the dashed line) would be the area
law S = A/4."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("area_law_data.csv")
# The solver fails at k = 1, y_0 = 3.5 (the surface lies too close to the horizon); such points are left out.
failed = (data.dA <= 0) | (data.dA > 2 * data.A_T)
if failed.any():
    print("left out (solver failure):", data.loc[failed, ["k", "y0"]].to_dict("records"))
data = data[~failed]
colours = {0.2: "#2a78d6", 0.5: "#eb6834", 1.0: "#1baf7a"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "legend.frameon": False})
figure, (left, right) = plt.subplots(1, 2, figsize=(6.4, 2.7))
for k, part in data.groupby("k"):
    left.plot(part.A_T, part.dA, "-o", color=colours[k], markersize=3.5, linewidth=1.4, label=f"$k = {k:g}$")
    right.plot(part.y0, part.dA / part.A_T, "-o", color=colours[k], markersize=3.5, linewidth=1.4,
               label=f"$k = {k:g}$")
low, high = data.A_T.min() * 0.8, data.A_T.max() * 1.2
left.plot([low, high], [low, high], "--", color="#52514e", linewidth=1, label="area law, $\\Delta\\mathcal{A} = A_T$")
left.set_xscale("log")
left.set_yscale("log")
left.set_xlabel("transverse area $A_T(y_0)$ at $v = v_c(y_0)$  [$\\ell^3$]")
left.set_ylabel("excess area $\\Delta\\mathcal{A}$  [$\\ell^3$]")
left.legend(loc="upper left", fontsize=7)
right.axhline(1, linestyle="--", color="#52514e", linewidth=1)
right.set_xlabel("$y_0$")
right.set_ylabel("$\\Delta\\mathcal{A}\\,/\\,A_T$")
right.set_ylim(0, 1.1)
figure.tight_layout()
for suffix in ("pdf", "png"):
    figure.savefig(f"area_law.{suffix}", dpi=200, bbox_inches="tight")
print(data.assign(ratio=data.dA / data.A_T)[["k", "y0", "v_c", "dA", "A_T", "ratio"]].round(4).to_string(index=False))
