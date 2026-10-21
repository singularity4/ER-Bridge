# ER-Bridge

Numerical calculations for "*An Einstein–Rosen bridge in the holographic dual of a spherically expanding fluid*"
(N. Bilić and D. Tolic, in preparation): horizons, throat, extremal surfaces and the area law.

<img src="figures/bridge.png" alt="The Einstein–Rosen bridge: transverse radius against proper distance from the throat, and the surface of revolution" width="760">

## Usage

```bash
pip install numpy scipy sympy pandas matplotlib
python ER_bridge.py           # all sections; python ER_bridge.py 1 3 7 for selected sections
python area_law_data.py       # data for Fig. 2
python area_law_plot.py       # Fig. 2
python bridge_animation.py    # the animation below
```

## Animation

<img src="figures/bridge_animation.gif" alt="The bridge in the bulk turning while the boundary fluid expands and cools" width="760">

The fluid on the boundary expands and cools, while the Einstein–Rosen bridge in the bulk stays the same.

## License

MIT
