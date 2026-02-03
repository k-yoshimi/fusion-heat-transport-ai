"""Tests for feature extraction on analytic profiles."""

import numpy as np
import pytest
from features.extract import (
    gradient, laplacian, chi, max_abs_gradient, zero_crossings, energy_content,
    extract_all, half_max_radius, profile_centroid, gradient_slope, profile_width,
    extract_initial_features,
)


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


def test_chi_nonlinear():
    dTdr = np.array([0.0, 0.3, 1.0, 2.0])
    result = chi(dTdr, alpha=1.0)
    # |T'|=0.0 <= 0.5 -> 0.1
    # |T'|=0.3 <= 0.5 -> 0.1
    # |T'|=1.0 > 0.5 -> (1.0-0.5)^1 + 0.1 = 0.6
    # |T'|=2.0 > 0.5 -> (2.0-0.5)^1 + 0.1 = 1.6
    np.testing.assert_allclose(result, [0.1, 0.1, 0.6, 1.6])


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


# --- Tests for new profile shape features ---


def test_half_max_radius_parabola(grid):
    """T = 1 - r^2: T = 0.5 when r^2 = 0.5, so r = sqrt(0.5) ≈ 0.707."""
    T = 1.0 - grid**2
    r_half = half_max_radius(T, grid)
    assert r_half == pytest.approx(np.sqrt(0.5), abs=0.02)


def test_half_max_radius_linear(grid):
    """T = 1 - r: T = 0.5 when r = 0.5."""
    T = 1.0 - grid
    r_half = half_max_radius(T, grid)
    assert r_half == pytest.approx(0.5, abs=0.02)


def test_half_max_radius_constant_zero():
    """If T is constant zero, should return 1.0."""
    r = np.linspace(0, 1, 101)
    T = np.zeros_like(r)
    r_half = half_max_radius(T, r)
    assert r_half == 1.0


def test_profile_centroid_parabola(grid):
    """Test centroid calculation for parabolic profile."""
    T = 1.0 - grid**2
    centroid = profile_centroid(T, grid)
    # Centroid should be somewhere between 0 and 1
    assert 0 < centroid < 1


def test_profile_centroid_linear(grid):
    """For linear profile, centroid is closer to center."""
    T = 1.0 - grid
    centroid = profile_centroid(T, grid)
    assert 0.3 < centroid < 0.7


def test_gradient_slope_parabola(grid):
    """T = 1 - r^2 => dT/dr = -2r, |dT/dr| = 2r which increases with r."""
    T = 1.0 - grid**2
    slope = gradient_slope(T, grid)
    # Slope should be positive since |gradient| increases with r
    assert slope > 0


def test_gradient_slope_linear(grid):
    """T = 1 - r => dT/dr = -1, constant gradient."""
    T = 1.0 - grid
    slope = gradient_slope(T, grid)
    # Slope should be near zero for constant gradient
    assert abs(slope) < 0.1


def test_profile_width_parabola(grid):
    """Test width calculation for parabolic profile."""
    T = 1.0 - grid**2
    width = profile_width(T, grid)
    # Width should be between 0 and 1
    assert 0 < width < 1


def test_profile_width_different_shapes():
    """Profile width distinguishes different shapes.

    Note: 1-r^4 is flatter in center than 1-r^2, so it has more mass at
    larger radii, leading to a larger effective width.
    """
    r = np.linspace(0, 1, 101)
    T_parabola = 1.0 - r**2  # n=2
    T_flat_center = 1.0 - r**4  # n=4, flatter center, steeper edge
    # The r^4 profile is flatter near center, so its width is larger
    assert profile_width(T_flat_center, r) > profile_width(T_parabola, r)


def test_extract_initial_features_has_16_keys(grid):
    """extract_initial_features should return 16 features."""
    T0 = 1.0 - grid**2
    feats = extract_initial_features(T0, grid, alpha=0.5, nr=len(grid),
                                     dt=0.001, t_end=0.1)
    assert len(feats) == 16
    # Check new keys are present
    new_keys = {"half_max_radius", "profile_centroid", "gradient_slope", "profile_width"}
    assert new_keys.issubset(set(feats.keys()))
