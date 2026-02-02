"""Tests for feature extraction on analytic profiles."""

import numpy as np
import pytest
from features.extract import gradient, laplacian, chi, max_abs_gradient, zero_crossings, energy_content, extract_all


@pytest.fixture
def grid():
    return np.linspace(0, 1, 101)


def test_gradient_parabola(grid):
    """T = 1 - r^2 => dT/dr = -2r."""
    T = 1.0 - grid**2
    dTdr = gradient(T, grid)
    expected = -2.0 * grid
    np.testing.assert_allclose(dTdr[1:-1], expected[1:-1], atol=1e-3)


def test_laplacian_parabola(grid):
    """T = 1 - r^2 => d²T/dr² = -2."""
    T = 1.0 - grid**2
    d2T = laplacian(T, grid)
    np.testing.assert_allclose(d2T[1:-1], -2.0, atol=1e-2)


def test_chi_linear():
    dTdr = np.array([0.0, 1.0, 2.0])
    result = chi(dTdr, alpha=0.5)
    np.testing.assert_allclose(result, [1.0, 1.5, 2.0])


def test_max_abs_gradient(grid):
    T = 1.0 - grid**2
    mag = max_abs_gradient(T, grid)
    assert mag == pytest.approx(2.0, abs=0.05)


def test_zero_crossings_monotone(grid):
    """Monotone profile has 0 crossings."""
    T = 1.0 - grid
    assert zero_crossings(T, grid) == 0


def test_zero_crossings_oscillating(grid):
    """Oscillating profile has crossings."""
    T = np.sin(4 * np.pi * grid)
    nc = zero_crossings(T, grid)
    assert nc >= 4  # cos(4πr) changes sign at r=1/8, 3/8, 5/8, 7/8


def test_energy_content(grid):
    """T=1-r^2: integral of (1-r^2)*r dr from 0 to 1 = 1/4."""
    T = 1.0 - grid**2
    E = energy_content(T, grid)
    assert E == pytest.approx(0.25, abs=0.01)


def test_extract_all_keys(grid):
    T = 1.0 - grid**2
    feats = extract_all(T, grid, alpha=0.0)
    expected_keys = {"max_abs_gradient", "zero_crossings", "energy_content",
                     "max_chi", "min_chi", "max_laplacian", "T_center", "T_edge"}
    assert set(feats.keys()) == expected_keys
