"""Reproduce the IMA polar-coordinate panels and add a Swiss-roll OCT.

The polar panels follow figures/Figure_radial_map.ipynb in
lgresele/independent-mechanism-analysis. No JAX dependency is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import FixedLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

BLUE = "#1A85FF"
MAGENTA = "#D41159"
RANGE_R = (1.5 * np.pi, 4.5 * np.pi)
RANGE_V = (-5.0, 5.0)
STYLE = {
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def polar_map(s: np.ndarray) -> np.ndarray:
    """Map (..., 2) sources (s1, s2) = (radius, angle) to (x1, x2)."""
    r, theta = np.moveaxis(np.asarray(s), -1, 0)
    return np.stack((r * np.cos(theta), r * np.sin(theta)), axis=-1)


def swiss_roll(s: np.ndarray) -> np.ndarray:
    """Global OCT R^2 -> R^3, with r = exp(s_1) > 0."""
    s1, s2 = np.moveaxis(np.asarray(s), -1, 0)
    r = np.exp(s1)
    return np.stack((r * np.cos(r), s2, r * np.sin(r)), axis=-1)


def polar_panel(ax, *, transformed: bool = False):
    """Reproduce the original 15-by-15 grid and its polar image."""
    r, theta = np.meshgrid(np.linspace(0.5, 1.5, 15), np.linspace(-0.5, 0.5, 15))
    points = np.stack((r, theta), axis=-1)
    if transformed:
        points = polar_map(points)
    for segments, color in ((points, BLUE), (points.transpose(1, 0, 2), MAGENTA)):
        ax.add_collection(LineCollection(segments, colors=color, linewidths=1.5))
        ax.autoscale()
    if transformed:
        ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$")
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(FixedLocator([0.5, 1.0, 1.5]))
    else:
        ax.set(xlabel=r"$s_1$", ylabel=r"$s_2$")
        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
        ax.xaxis.set_major_locator(FixedLocator(np.linspace(0.5, 1.5, 5)))
        ax.yaxis.set_major_locator(FixedLocator([-np.pi / 8, 0, np.pi / 8]))
        ax.set_yticklabels([r"$-\pi/8$", r"$0$", r"$\pi/8$"])
    return ax


def _visible_segments(points: np.ndarray, view: np.ndarray,
                      height_range=RANGE_V) -> np.ndarray:
    """Hide curves behind the opaque roll using orthographic ray intersections.

    The surface is an extrusion of an Archimedean spiral. Intersecting a
    viewing ray with that spiral in the x1-x3 plane, and checking x2's
    bounds, therefore determines visibility. A dense polygonal spiral
    approximates the exact surface only for this visibility calculation;
    all displayed points are evaluated on the exact smooth surface.
    """
    radius = np.linspace(*RANGE_R, 2400)
    curve = np.column_stack((radius * np.cos(radius), radius * np.sin(radius)))
    edge = np.diff(curve, axis=0)
    direction = view[[0, 2]]
    cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    denominator = cross(direction, edge)
    valid = np.abs(denominator) > 1e-12
    starts, edge, denominator = curve[:-1][valid], edge[valid], denominator[valid]
    hidden = np.zeros(len(points), dtype=bool)
    for i in range(0, len(points), 128):
        batch = points[i:i + 128]
        delta = starts[None] - batch[:, None, [0, 2]]
        distance = cross(delta, edge[None]) / denominator
        fraction = cross(delta, direction) / denominator
        y = batch[:, None, 1] + distance * view[1]
        hits = ((distance > 0.004) & (fraction >= 0) & (fraction <= 1)
                & (y >= height_range[0] - 1e-7) & (y <= height_range[1] + 1e-7))
        hidden[i:i + len(batch)] = np.any(hits, axis=1)
    visible = ~hidden
    # Keep adjacent visible samples as short line segments; never join across
    # an occluded interval. Both endpoints must be visible.
    pairs = np.stack((points[:-1], points[1:]), axis=1)
    return pairs[visible[:-1] & visible[1:]]


def swiss_panel(ax, *, elev=22.0, azim=-62.0, roll=0.0,
                depth_scale=2.0, blue_curves=9):
    """Render the roll with its near end lower-left and far end upper-right."""
    if depth_scale <= 0 or blue_curves < 2:
        raise ValueError("Depth must be positive and at least two blue curves are required.")
    height_range = tuple(v * depth_scale for v in RANGE_V)
    elev_rad, azim_rad = np.deg2rad([elev, azim])
    view = np.array([np.cos(elev_rad) * np.cos(azim_rad),
                     np.cos(elev_rad) * np.sin(azim_rad), np.sin(elev_rad)])
    radius = np.linspace(*RANGE_R, 1000)
    height = np.linspace(*height_range, 500)
    # The spiral is much longer than the extrusion is deep. Sample rulings
    # by arc length to keep the visible grid cells reasonably balanced.
    arc_length = (radius * np.sqrt(1 + radius**2) + np.arcsinh(radius)) / 2
    ruling_radii = np.interp(np.linspace(arc_length[0], arc_length[-1], 41),
                            arc_length, radius)
    families = [
        (BLUE, [swiss_roll(np.column_stack((np.log(radius), np.full_like(radius, v))))
                for v in np.linspace(*height_range, blue_curves)]),
        (MAGENTA, [swiss_roll(np.column_stack((np.full_like(height, np.log(r)), height)))
                   for r in ruling_radii]),
    ]
    for color, curves in families:
        segments = np.concatenate([_visible_segments(points, view, height_range)
                                   for points in curves])
        ax.add_collection3d(Line3DCollection(segments, colors=color, linewidths=1.5))
    ax.set(xlim=(-11.5, 14.5), ylim=height_range, zlim=(-12.5, 15.0))
    ax.set_box_aspect((26, height_range[1] - height_range[0], 27.5), zoom=1.02)
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_proj_type("ortho")
    ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$")
    # Unlike a 3D z-label, text2D is included reliably in tight export bounds.
    ax.text2D(1.15, 0.59, r"$x_3$", transform=ax.transAxes, rotation=90,
              fontsize=14, ha="center", va="center")
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks(height_range)
    ax.set_zticks([-10, 0, 10])
    ax.tick_params(labelsize=13, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("white")
        axis.line.set_linewidth(0.8)
        axis._axinfo["grid"]["linewidth"] = 0
    return ax


def _save(fig, directory: Path, name: str):
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", pad_inches=0.1)
    fig.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.1)


def make_figures(output: Path, *, elev=22.0, azim=-62.0, roll=0.0,
                 depth_scale=2.0, blue_curves=9):
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for name, transformed in (("source_grid", False), ("observations_grid", True)):
            fig, ax = plt.subplots()
            polar_panel(ax, transformed=transformed)
            _save(fig, output, name)
            plt.close(fig)
        fig = plt.figure(figsize=(4.9, 4.4))
        ax = fig.add_subplot(projection="3d")
        swiss_panel(ax, elev=elev, azim=azim, roll=roll,
                    depth_scale=depth_scale, blue_curves=blue_curves)
        _save(fig, output, "swiss_roll")
        plt.close(fig)

        fig = plt.figure(figsize=(13.4, 4.6))
        grid = fig.add_gridspec(1, 3, width_ratios=[1, 0.83, 1.24],
                               left=0.065, right=0.96, bottom=0.16, top=0.88,
                               wspace=0.30)
        axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
                fig.add_subplot(grid[0, 2], projection="3d")]
        polar_panel(axes[0])
        polar_panel(axes[1], transformed=True)
        swiss_panel(axes[2], elev=elev, azim=azim, roll=roll,
                    depth_scale=depth_scale, blue_curves=blue_curves)
        for ax, title in zip(axes, ("Source coordinates", "Polar coordinates", "Swiss-roll OCT")):
            ax.set_title(title, fontsize=14, pad=16)
        _save(fig, output, "coordinate_panels")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/pdf"))
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-62.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--depth-scale", type=float, default=2.0)
    parser.add_argument("--blue-curves", type=int, default=9)
    args = parser.parse_args()
    make_figures(args.output, elev=args.elev, azim=args.azim, roll=args.roll,
                 depth_scale=args.depth_scale, blue_curves=args.blue_curves)


if __name__ == "__main__":
    main()
