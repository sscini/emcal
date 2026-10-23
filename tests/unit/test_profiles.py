"""Arclength resampling of curves and profiles (emcal.external.profiles)."""
import numpy as np
import pytest

from emcal.external.profiles import (arclength_grid, curve_to_vector, curve_x_grid,
                                     resample_curve, resample_values)


def test_open_curve_points_become_evenly_spaced():
    # Uneven points on a straight line from (0, 0) to (4, 0)
    points = [[0, 0], [0.5, 0], [3, 0], [4, 0]]
    out = resample_curve(points, 5)
    np.testing.assert_allclose(out, [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])


def test_open_curve_keeps_both_end_points():
    points = [[0, 0], [1, 1], [3, 0]]
    out = resample_curve(points, 7)
    np.testing.assert_allclose(out[0], [0, 0])
    np.testing.assert_allclose(out[-1], [3, 0])


def test_closed_square_starts_at_first_point_and_goes_around():
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    out = resample_curve(square, 8, closed=True)
    expected = [[0, 0], [0.5, 0], [1, 0], [1, 0.5], [1, 1], [0.5, 1], [0, 1], [0, 0.5]]
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_closed_curve_ignores_a_repeated_first_point():
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    with_repeat = square + [[0, 0]]
    np.testing.assert_allclose(resample_curve(with_repeat, 8, closed=True),
                               resample_curve(square, 8, closed=True))


def test_repeated_consecutive_points_are_ignored():
    points = [[0, 0], [1, 0], [1, 0], [2, 0]]
    np.testing.assert_allclose(resample_curve(points, 3), [[0, 0], [1, 0], [2, 0]])


def test_circle_resamples_onto_the_circle_with_even_spacing():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False) ** 1.3 / (2 * np.pi) ** 0.3  # uneven
    circle = np.column_stack([np.cos(t), np.sin(t)])
    out = resample_curve(circle, 50, closed=True)
    radius = np.linalg.norm(out, axis=1)
    assert np.all(np.abs(radius - 1) < 1e-3)
    steps = np.linalg.norm(np.diff(np.vstack([out, out[:1]]), axis=0), axis=1)
    assert steps.max() / steps.min() < 1.01


def test_values_are_resampled_along_the_curve():
    points = [[0, 0], [1, 0], [4, 0]]
    values = [0.0, 1.0, 4.0]  # value = distance from the start
    np.testing.assert_allclose(resample_values(points, values, 5), [0, 1, 2, 3, 4])


def test_closed_values_wrap_back_to_the_first_value():
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    values = [0.0, 1.0, 2.0, 1.0]
    out = resample_values(square, values, 8, closed=True)
    np.testing.assert_allclose(out, [0, 0.5, 1, 1.5, 2, 1.5, 1, 0.5])


def test_vector_and_x_grid_line_up():
    points = resample_curve([[0, 0], [2, 4]], 3)
    y = curve_to_vector(points)
    x_grid = curve_x_grid(3)
    np.testing.assert_allclose(y, [0, 1, 2, 0, 2, 4])  # all x, then all y
    np.testing.assert_allclose(x_grid, [[0, 0], [0, 0.5], [0, 1], [1, 0], [1, 0.5], [1, 1]])


def test_arclength_grid_open_and_closed():
    np.testing.assert_allclose(arclength_grid(3), [0, 0.5, 1])
    np.testing.assert_allclose(arclength_grid(4, closed=True), [0, 0.25, 0.5, 0.75])


def test_bad_input_is_rejected():
    with pytest.raises(AssertionError):
        resample_curve([[0, 0], [0, 0]], 3)  # zero length
    with pytest.raises(AssertionError):
        resample_curve([[0, 0], [np.nan, 1]], 3)
    with pytest.raises(AssertionError):
        resample_curve([[0, 0], [1, 0]], 1)
    with pytest.raises(AssertionError):
        resample_curve([[0, 0], [1, 0], [1, 1]], 2, closed=True)
    with pytest.raises(AssertionError):
        resample_values([[0, 0], [1, 0]], [1.0], 3)
