"""
pareto_plotter.py (3D-only)

Save a 3D Pareto set/front figure (PNG/JPG) from multi-objective points.

- Assumes *minimization* objectives.
- Expects Nx3 points (f1, f2, f3).
- Highlights the non-dominated set (Pareto-optimal points) with a different marker.

Typical usage (after your solver run):
    points = solver.archive.as_points()   # list[tuple[float,float,float]]
    save_pareto_figures(points, out_prefix="pareto_front", dpi=600, fmt="png")
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

# Ensures the "3d" projection is registered in older Matplotlib versions.
# (Not strictly required on modern Matplotlib, but harmless and improves portability.)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def pareto_filter(points: Sequence[Sequence[float]]) -> List[Tuple[float, float, float]]:
    """Return the non-dominated subset (minimization) for 3 objectives."""
    pts = [tuple(map(float, p)) for p in points]
    if any(len(p) != 3 for p in pts):
        raise ValueError("pareto_filter expects 3D objective vectors (Nx3).")

    keep: List[Tuple[float, float, float]] = []
    for i, a in enumerate(pts):
        dominated = False
        for j, b in enumerate(pts):
            if i == j:
                continue
            if all(x <= y for x, y in zip(b, a)) and any(x < y for x, y in zip(b, a)):
                dominated = True
                break
        if not dominated:
            keep.append(a)

    # unique (stable order)
    uniq: List[Tuple[float, float, float]] = []
    seen = set()
    for p in keep:
        key = tuple(round(float(x), 12) for x in p)
        if key not in seen:
            seen.add(key)
            uniq.append((float(p[0]), float(p[1]), float(p[2])))
    return uniq


def _ensure_array(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.array(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"3D-only plotter expects Nx3 points; got shape={arr.shape}")
    return arr


def _axis_limits(values: np.ndarray, *, pad_frac: float = 0.05) -> Tuple[float, float]:
    """Compute stable axis limits for a 1D array.

    Matplotlib's 3D axes auto-scaling can look odd when all values are identical
    (e.g., all zeros -> ticks like [-0.04, 0.04]). We compute a padded range
    ourselves and, when the data are non-negative, clamp the lower bound to 0.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        return 0.0, 1.0

    vmin = float(np.nanmin(v))
    vmax = float(np.nanmax(v))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0

    if abs(vmax - vmin) < 1e-12:
        # Single-valued axis: use a pad based on magnitude (fallback to 1.0 if ~0).
        base = abs(vmin) if abs(vmin) > 1e-12 else 1.0
        pad = float(pad_frac) * base
        lo, hi = vmin - pad, vmax + pad
    else:
        pad = float(pad_frac) * (vmax - vmin)
        lo, hi = vmin - pad, vmax + pad

    # If data are non-negative, avoid negative lower bounds in the plot.
    if vmin >= 0.0:
        lo = max(0.0, lo)

    # Guard against accidental inversion (shouldn't happen, but be safe).
    if hi <= lo:
        hi = lo + 1.0

    return float(lo), float(hi)


def save_pareto_3d(
    points: Sequence[Sequence[float]],
    out_path: str = "pareto_front_3d.png",
    labels: Tuple[str, str, str] = ("Objective 1", "Objective 2", "Objective 3"),
    title: str = "Pareto set (3D, minimization)",
    dpi: int = 600,
) -> str:
    """Save a 3D scatter (all points + highlighted non-dominated).

    This version fixes the label/tick overlap by:
      - using positive label padding (never negative)
      - increasing margins slightly
      - forcing Matplotlib to include 3D axis labels in the tight bounding box
        via bbox_extra_artists (prevents clipping when using bbox_inches="tight")
    """
    arr = _ensure_array(points)
    nd = np.array(pareto_filter(arr), dtype=float)

    # Scale figure size with the global font size (helps if you increased rcParams["font.size"])
    base_fs = float(plt.rcParams.get("font.size", 12))
    scale = max(1.0, base_fs / 12.0)

    fig = plt.figure(figsize=(7.0 * scale, 6.0 * scale))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], marker="o", alpha=0.35, label="All points")
    if len(nd) > 0:
        ax.scatter(nd[:, 0], nd[:, 1], nd[:, 2], marker="x", label="Non-dominated")

    # --- Stable axis limits (prevents "tiny +/- 0.04" ranges when an axis is constant) ---
    xlo, xhi = _axis_limits(arr[:, 0])
    ylo, yhi = _axis_limits(arr[:, 1])
    zlo, zhi = _axis_limits(arr[:, 2])
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_zlim(zlo, zhi)

    # --- Label/tick spacing: keep axis labels away from tick numbers ---
    # (Negative labelpad causes overlap, especially on the z-axis.)
    lab_pad_xy = max(10.0, 0.9 * base_fs)
    lab_pad_z = max(12.0, 1.1 * base_fs)
    tick_pad = max(2.0, 0.25 * base_fs)

    ax.set_xlabel(labels[0], labelpad=lab_pad_xy)
    ax.set_ylabel(labels[1], labelpad=lab_pad_xy)
    ax.set_zlabel(labels[2], labelpad=lab_pad_z)

    # Push tick labels slightly away from the axis spine
    ax.xaxis.set_tick_params(pad=tick_pad)
    ax.yaxis.set_tick_params(pad=tick_pad)
    ax.zaxis.set_tick_params(pad=tick_pad)

    title_text = ax.set_title(title, pad=max(10.0, 0.8 * base_fs))
    leg = ax.legend(loc="upper right")

    # Avoid tight_layout() for 3D; it often behaves poorly with z-labels.
    # Instead use a gentle manual margin, and rely on bbox_inches='tight' for final output.
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.88)

    # IMPORTANT: draw first so text extents are known
    fig.canvas.draw()

    # CRITICAL: 3D + bbox_inches='tight' often ignores the z-label unless included explicitly
    extra_artists = [ax.xaxis.label, ax.yaxis.label, ax.zaxis.label, title_text, leg]

    fig.savefig(
        out_path,
        dpi=int(dpi),
        bbox_inches="tight",
        pad_inches=0.30,
        bbox_extra_artists=extra_artists,
    )
    plt.close(fig)
    return out_path


def save_pareto_figures(
    points: Sequence[Sequence[float]],
    out_prefix: str = "pareto_front",
    labels: Optional[Tuple[str, str, str]] = None,
    dpi: int = 600,
    fmt: str = "png",
) -> List[str]:
    """Save ONLY the 3D Pareto figure.

    Returns a list with one filename: f"{out_prefix}_3d.<fmt>"
    """
    fmt = str(fmt).lstrip(".").lower()
    if labels is None:
        labels = ("Objective 1", "Objective 2", "Objective 3")
    out_path = f"{out_prefix}_3d.{fmt}"
    return [save_pareto_3d(points, out_path=out_path, labels=labels, dpi=dpi)]


def read_points_csv(path: str) -> List[Tuple[float, float, float]]:
    """Read CSV with numeric rows (expects 3 columns per row)."""
    pts: List[Tuple[float, float, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            try:
                a, b, c = float(r[0]), float(r[1]), float(r[2])
            except Exception:
                # skip headers / malformed rows
                continue
            pts.append((a, b, c))
    if not pts:
        raise ValueError("No numeric Nx3 points found in CSV.")
    return pts


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python pareto_plotter.py <points.csv> [out.png|out.jpg]")
    csv_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else "pareto_front_3d.png"

    pts = read_points_csv(csv_path)
    out_prefix, ext = os.path.splitext(out)
    fmt = ext.lstrip(".") or "png"

    files = save_pareto_figures(pts, out_prefix=out_prefix, dpi=600, fmt=fmt)
    print("Saved:", ", ".join(files))