from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def pareto_filter(points: Sequence[Sequence[float]]) -> List[Tuple[float, ...]]:
    """Return the non-dominated subset (minimization)."""
    pts = [tuple(map(float, p)) for p in points]
    keep = []
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
    # unique
    uniq = []
    seen = set()
    for p in keep:
        key = tuple(round(x, 12) for x in p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def hypervolume_2d(points: Iterable[Tuple[float, float]], ref: Tuple[float, float]) -> float:
    """Exact 2D hypervolume for minimization.

    Requirements:
      - points should be non-dominated in 2D (use pareto_filter if unsure).
      - ref must be worse than all points.

    Implementation:
      Sort by x ascending and accumulate rectangular slices in y.
    """
    pts = sorted(points, key=lambda p: p[0])
    hv = 0.0
    ref_x, ref_y = ref
    best_y = ref_y
    for x, y in pts:
        if y < best_y:
            hv += (ref_x - x) * (best_y - y)
            best_y = y
    return float(hv)


def hypervolume_3d(points: Iterable[Tuple[float, float, float]], ref: Tuple[float, float, float]) -> float:
    """Exact 3D hypervolume for minimization using x-slicing + exact 2D HV.

    Assumptions:
      - points are non-dominated in 3D (we will filter internally).
      - ref dominates (is worse than) every point.

    Algorithm:
      - Sort unique x values ascending.
      - For each slab [x_i, x_{i+1}), compute 2D HV in (y,z) of points with x <= x_i.
      - Sum slab_width * area.

    This is exact for 3D with non-dominated points.
    """
    pts = pareto_filter(points)
    if not pts:
        return 0.0
    rx, ry, rz = map(float, ref)

    # sort by x
    pts_sorted = sorted(pts, key=lambda p: p[0])
    xs = sorted({p[0] for p in pts_sorted})
    xs.append(rx)

    hv = 0.0
    # cumulative set as x increases
    cum = []
    k = 0
    for i in range(len(xs) - 1):
        x_i = xs[i]
        x_next = xs[i + 1]
        # add all points with x == x_i (and any that are less due to duplicates in xs)
        while k < len(pts_sorted) and pts_sorted[k][0] <= x_i + 1e-15:
            cum.append(pts_sorted[k])
            k += 1
        width = float(x_next - x_i)
        if width <= 0:
            continue
        # project to yz and filter non-dominated in 2D
        yz = [(p[1], p[2]) for p in cum]
        yz_nd = pareto_filter(yz)  # works for 2D too
        area = hypervolume_2d([(a, b) for a, b in yz_nd], ref=(ry, rz))
        hv += width * area
    return float(hv)


def igd(approx: Sequence[Sequence[float]], reference: Sequence[Sequence[float]], normalize: bool = True) -> float:
    """Inverted Generational Distance (IGD).

    IGD( A, R ) = (1/|R|) * sum_{r in R} min_{a in A} ||r - a||_2

    If normalize=True, objectives are min-max normalized using the reference set R.
    """
    A = np.array(approx, dtype=float)
    R = np.array(reference, dtype=float)
    if len(A) == 0 or len(R) == 0:
        return float("nan")

    if normalize:
        mins = R.min(axis=0)
        maxs = R.max(axis=0)
        denom = np.where(maxs - mins == 0, 1.0, maxs - mins)
        A = (A - mins) / denom
        R = (R - mins) / denom

    # compute distances efficiently
    dists = []
    for r in R:
        diff = A - r[None, :]
        dd = np.sqrt(np.sum(diff * diff, axis=1))
        dists.append(float(dd.min()))
    return float(np.mean(dists))