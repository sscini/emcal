"""Resample curves and profiles onto a fixed arclength grid.

A simulator's shape output (a contour, or a profile such as curvature along a curve)
usually has a different number of points from run to run. Resampling it onto ``n``
points evenly spaced in arclength gives an output vector with the same length and
meaning in every run, which is what the emulator methods need.
"""
import numpy as np


def resample_curve(points, n, closed=False):
    """
    Points evenly spaced in arclength along a curve.

    Parameters
    ----------
    points: array-like, shape (m, d)
        Curve points in order
    n: int
        Number of points to return
    closed: bool, default False
        If True, the curve returns from the last point to the first. A first point
        repeated at the end is ignored.

    Returns
    -------
    resampled: np.ndarray, shape (n, d)
        Open curve: from the first to the last point. Closed curve: starts at the first
        point and does not repeat it.
    """
    points, _ = _clean_curve(points, None, closed)
    path, s = _path_and_arclength(points, closed)
    s_new = arclength_grid(n, closed)

    n_coords = points.shape[1]
    resampled = np.empty((n, n_coords))
    for j in range(n_coords):
        resampled[:, j] = np.interp(s_new, s, path[:, j])
    return resampled


def resample_values(points, values, n, closed=False):
    """
    Values given at each curve point (e.g. curvature), resampled evenly in arclength.

    Parameters
    ----------
    points: array-like, shape (m, d)
        Curve points in order
    values: array-like, shape (m,)
        One value per point
    n: int
        Number of values to return
    closed: bool, default False
        See resample_curve

    Returns
    -------
    resampled: np.ndarray, shape (n,)
        Values at arclength_grid(n, closed)
    """
    points, values = _clean_curve(points, values, closed)
    path, s = _path_and_arclength(points, closed)
    if closed:
        # The path ends back at the first point, so it ends with the first value too
        values = np.append(values, values[0])
    s_new = arclength_grid(n, closed)
    return np.interp(s_new, s, values)


def arclength_grid(n, closed=False):
    """
    Normalized arclength (0 to 1) of the points returned by resample_curve.

    Open curve: n points from 0 to 1. Closed curve: n points from 0, stopping one step
    before 1 (which is the first point again).
    """
    assert int(n) == n, "n must be an integer"
    n = int(n)
    if closed:
        assert n >= 3, "a closed curve needs n >= 3"
        return np.arange(n) / n
    assert n >= 2, "an open curve needs n >= 2"
    return np.linspace(0.0, 1.0, n)


def curve_to_vector(points):
    """
    Flatten resampled points into one output vector: all x values, then all y values, ...

    The order matches curve_x_grid.
    """
    points = np.asarray(points, dtype=float)
    assert points.ndim == 2, "points must be an (n, d) array"
    pieces = []
    for j in range(points.shape[1]):
        pieces.append(points[:, j])
    return np.concatenate(pieces)


def curve_x_grid(n, n_coords=2, closed=False):
    """
    x grid for curve_to_vector(resample_curve(points, n, closed)): one row per output.

    Returns
    -------
    x_grid: np.ndarray, shape (n * n_coords, 2)
        Columns: coordinate index (0 = x, 1 = y, ...) and normalized arclength s
    """
    s = arclength_grid(n, closed)
    rows = []
    for j in range(n_coords):
        for s_value in s:
            rows.append([j, s_value])
    return np.array(rows, dtype=float)


# ---- internals ----
def _clean_curve(points, values, closed):
    """Check the curve, drop repeated consecutive points (and their values)."""
    points = np.asarray(points, dtype=float)
    assert points.ndim == 2 and points.shape[0] >= 2, "points must be an (m, d) array with m >= 2"
    assert np.all(np.isfinite(points)), "points must be finite"
    if values is not None:
        values = np.asarray(values, dtype=float).reshape(-1)
        assert values.size == points.shape[0], "values must have one entry per point"
        assert np.all(np.isfinite(values)), "values must be finite"

    # Zero-length segments would make the arclength repeat
    keep = [0]
    for i in range(1, len(points)):
        if not np.array_equal(points[i], points[keep[-1]]):
            keep.append(i)

    # A closed curve given with its first point repeated at the end
    if closed and len(keep) > 2 and np.allclose(points[keep[-1]], points[0]):
        keep.pop()

    assert len(keep) >= 2, "the curve needs at least 2 distinct points"
    points = points[keep]
    if values is not None:
        values = values[keep]
    return points, values


def _path_and_arclength(points, closed):
    """The path to interpolate along, and its normalized arclength at each path point."""
    path = points
    if closed:
        path = np.vstack([points, points[:1]])
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]
    return path, cumulative / total_length
