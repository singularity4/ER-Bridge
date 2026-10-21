"""bridge_animation.py - animation of the Einstein-Rosen bridge of the expanding fluid (k = 1, l = 1).

Left: the bridge in the bulk, drawn as a surface of revolution and seen from a slowly turning camera. The axis
is the proper distance from the throat, l ln(z / z_*), and the radius is the transverse radius
R = l sqrt(B(v)) / v with v = z / tau and B = 1 + k v^4. Both depend on v alone, so the bridge is the same at
every tau: the throat (green, v_* = k^(-1/4)) and the horizons (red, v_1 and v_2) keep their radii.

Right: the boundary fluid. Comoving fluid elements of the region y < y_0 move apart as r = tau sinh y, and the
temperature falls as T = sqrt(2) k^(1/4) / (pi tau).

Output: figures/bridge_animation.gif and figures/bridge_animation.mp4 (the MP4 needs ffmpeg).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

# geometry, in the notation of the paper
k = 1.0
v_throat = k ** -0.25                          # throat, z_* = tau k^(-1/4)
v_1 = 0.69320546                               # inner horizon at k = 1, eq. (17) of the paper
v_2 = 1 / (np.sqrt(k) * v_1)                   # outer horizon, from v_1 v_2 = k^(-1/2)
v_min, v_max = 0.15, 6.7                       # range of v drawn; an inversion pair, v_min v_max ~ k^(-1/2)
proper_times = np.geomspace(1.0, 3.0, 72)      # one frame per proper time tau

# boundary fluid
region_rapidity = 1.0                          # comoving region y < y_0
element_count = 260
temperature = np.sqrt(2) * k**0.25 / (np.pi * proper_times)

# colours: text, bridge, throat, horizons, fluid (darker = hotter)
text_colour, caption_colour = "#2b2b2b", "#6b6b6b"
surface_colour, mesh_colour = "#9fb6cc", "#7f97ad"
throat_colour, horizon_colour = "#1b8a5a", "#c0392b"
fluid_colours = plt.get_cmap("Oranges")


def transverse_radius(v):
    """Transverse radius R / l of the constant-v surface, sqrt(B(v)) / v; independent of tau."""
    return np.sqrt(1 + k * v**4) / v


v_grid = np.geomspace(v_min, v_max, 90)
angle = np.linspace(0, 2 * np.pi, 48)
random = np.random.default_rng(1)
# random comoving positions inside y < y_0 (uniform in a ball of radius y_0, for display only)
element_rapidity = region_rapidity * random.random(element_count) ** (1 / 3)
element_direction = random.normal(size=(element_count, 3))
element_direction /= np.linalg.norm(element_direction, axis=1, keepdims=True)

figure = plt.figure(figsize=(8.4, 4.0), dpi=100)
bulk = figure.add_subplot(1, 2, 1, projection="3d")
boundary = figure.add_subplot(1, 2, 2, projection="3d")


def draw(frame):
    """Draw both panels at proper time proper_times[frame]."""
    tau = proper_times[frame]
    camera_azimuth = -100 + 60 * frame / proper_times.size
    for panel in (bulk, boundary):
        panel.cla()
        panel.set_axis_off()
        panel.set_box_aspect((1.6, 1, 1) if panel is bulk else (1, 1, 1))

    # bulk: the bridge as a surface of revolution about the axis l ln(z / z_*)
    depth_mesh, angle_mesh = np.meshgrid(np.log(v_grid / v_throat), angle)
    radius_mesh = np.tile(transverse_radius(v_grid), (angle.size, 1))
    surface = (depth_mesh, radius_mesh * np.cos(angle_mesh), radius_mesh * np.sin(angle_mesh))
    bulk.plot_surface(*surface, color=surface_colour, alpha=0.30, linewidth=0, shade=False)
    bulk.plot_wireframe(*surface, rstride=4, cstride=9, color=mesh_colour, linewidth=0.35)
    for v, colour, style, width in ((v_throat, throat_colour, "-", 2.6), (v_1, horizon_colour, "--", 1.6),
                                    (v_2, horizon_colour, "--", 1.6)):
        ring_radius = transverse_radius(v)
        bulk.plot(np.full(angle.size, np.log(v / v_throat)), ring_radius * np.cos(angle),
                  ring_radius * np.sin(angle), style, color=colour, linewidth=width)
    largest_radius = transverse_radius(v_max)
    bulk.set_xlim(np.log(v_min / v_throat), np.log(v_max / v_throat))
    bulk.set_ylim(-largest_radius, largest_radius)
    bulk.set_zlim(-largest_radius, largest_radius)
    bulk.view_init(elev=18, azim=camera_azimuth)
    bulk.set_title("Bulk: the Einstein–Rosen bridge", color=text_colour, fontsize=10, pad=0)
    bulk.text2D(0.5, 0.02, "throat radius $\\sqrt{2}\\,\\ell k^{1/4}$ stays constant",
                transform=bulk.transAxes, ha="center", color=caption_colour, fontsize=8)

    # boundary: comoving fluid elements at r = tau sinh y, coloured by temperature
    positions = (tau * np.sinh(element_rapidity))[:, None] * element_direction
    heat = (temperature[frame] - temperature[-1]) / (temperature[0] - temperature[-1])   # 1 hot, 0 cool
    boundary.scatter(*positions.T, s=7, color=fluid_colours(0.35 + 0.6 * heat), depthshade=False)
    region_radius = tau * np.sinh(region_rapidity)
    azimuth, polar = np.meshgrid(np.linspace(0, 2 * np.pi, 13), np.linspace(0, np.pi, 8))
    boundary.plot_wireframe(region_radius * np.cos(azimuth) * np.sin(polar),
                            region_radius * np.sin(azimuth) * np.sin(polar),
                            region_radius * np.cos(polar), color="#d9d9d9", linewidth=0.5)
    limit = 1.05 * proper_times[-1] * np.sinh(region_rapidity)
    for set_limits in (boundary.set_xlim, boundary.set_ylim, boundary.set_zlim):
        set_limits(-limit, limit)
    boundary.view_init(elev=18, azim=camera_azimuth)
    boundary.set_title("Boundary: the expanding fluid", color=text_colour, fontsize=10, pad=0)
    boundary.text2D(0.5, 0.02, f"$\\tau = {tau:.2f}$,   temperature $T \\propto 1/\\tau$ = "
                    f"{temperature[frame] / temperature[0]:.2f} of the initial value",
                    transform=boundary.transAxes, ha="center", color=caption_colour, fontsize=8)
    figure.subplots_adjust(left=0, right=1, top=0.93, bottom=0.02, wspace=0)


animation = FuncAnimation(figure, draw, frames=proper_times.size, interval=70)
animation.save("figures/bridge_animation.gif", writer=PillowWriter(fps=14))
animation.save("figures/bridge_animation.mp4", writer=FFMpegWriter(fps=14, bitrate=1800))
