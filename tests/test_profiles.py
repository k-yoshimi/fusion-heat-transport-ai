"""Tests for initial temperature profile factory."""

import numpy as np
import pytest
from features.profiles import (
    parabolic, gaussian, flat_top, cosine, linear,
    make_profile, parse_profile_params, get_available_profiles,
    PROFILE_REGISTRY, DEFAULT_PARAMS,
)


@pytest.fixture
def grid():
    return np.linspace(0, 1, 101)


class TestParabolicProfile:
    def test_boundary_conditions(self, grid):
        T = parabolic(grid)
        assert T[0] == pytest.approx(1.0)
        assert T[-1] == pytest.approx(0.0, abs=1e-10)

    def test_default_is_1_minus_r_squared(self, grid):
        T = parabolic(grid)
        expected = 1.0 - grid**2
        np.testing.assert_allclose(T, expected)

    def test_amplitude_parameter(self, grid):
        T = parabolic(grid, A=2.0)
        assert T[0] == pytest.approx(2.0)
        assert T[-1] == pytest.approx(0.0, abs=1e-10)

    def test_power_parameter(self, grid):
        T = parabolic(grid, n=4.0)
        expected = 1.0 - grid**4
        np.testing.assert_allclose(T, expected)


class TestGaussianProfile:
    def test_boundary_conditions(self, grid):
        T = gaussian(grid)
        assert T[0] == pytest.approx(1.0, rel=1e-6)
        assert T[-1] == pytest.approx(0.0, abs=1e-10)

    def test_sigma_parameter(self, grid):
        T_narrow = gaussian(grid, sigma=0.3)
        T_wide = gaussian(grid, sigma=0.7)
        # Narrow gaussian should decay faster
        mid_idx = len(grid) // 2
        assert T_narrow[mid_idx] < T_wide[mid_idx]

    def test_monotonic_decay(self, grid):
        T = gaussian(grid)
        assert np.all(np.diff(T) <= 0)


class TestFlatTopProfile:
    def test_boundary_conditions(self, grid):
        T = flat_top(grid)
        assert T[0] == pytest.approx(1.0, rel=1e-6)
        assert T[-1] == pytest.approx(0.0, abs=1e-10)

    def test_flat_center_region(self, grid):
        T = flat_top(grid, w=0.8, n=8, m=2)
        # Center region should be relatively flat
        center_vals = T[:20]
        assert np.std(center_vals) < 0.05

    def test_width_parameter(self, grid):
        T_narrow = flat_top(grid, w=0.5)
        T_wide = flat_top(grid, w=0.9)
        mid_idx = len(grid) // 2
        assert T_narrow[mid_idx] < T_wide[mid_idx]


class TestCosineProfile:
    def test_boundary_conditions(self, grid):
        T = cosine(grid)
        assert T[0] == pytest.approx(1.0)
        assert T[-1] == pytest.approx(0.0, abs=1e-10)

    def test_formula(self, grid):
        T = cosine(grid)
        expected = np.cos(np.pi * grid / 2.0)
        np.testing.assert_allclose(T, expected)

    def test_amplitude_parameter(self, grid):
        T = cosine(grid, A=3.0)
        assert T[0] == pytest.approx(3.0)


class TestLinearProfile:
    def test_boundary_conditions(self, grid):
        T = linear(grid)
        assert T[0] == pytest.approx(1.0)
        assert T[-1] == pytest.approx(0.0, abs=1e-10)

    def test_formula(self, grid):
        T = linear(grid)
        expected = 1.0 - grid
        np.testing.assert_allclose(T, expected)

    def test_constant_gradient(self, grid):
        T = linear(grid)
        grad = np.diff(T)
        np.testing.assert_allclose(grad, grad[0], rtol=1e-10)


class TestMakeProfile:
    def test_default_is_parabolic(self, grid):
        T = make_profile(grid)
        T_expected = parabolic(grid)
        np.testing.assert_allclose(T, T_expected)

    def test_with_profile_name(self, grid):
        T = make_profile(grid, "gaussian")
        T_expected = gaussian(grid)
        np.testing.assert_allclose(T, T_expected)

    def test_with_params(self, grid):
        T = make_profile(grid, "parabolic", {"n": 4.0})
        T_expected = parabolic(grid, n=4.0)
        np.testing.assert_allclose(T, T_expected)

    def test_unknown_profile_raises(self, grid):
        with pytest.raises(ValueError, match="Unknown profile"):
            make_profile(grid, "unknown_profile")


class TestParseProfileParams:
    def test_empty_string(self):
        assert parse_profile_params("") == {}

    def test_single_float(self):
        params = parse_profile_params("sigma=0.3")
        assert params == {"sigma": 0.3}

    def test_single_int(self):
        params = parse_profile_params("n=4")
        assert params == {"n": 4}

    def test_multiple_params(self):
        params = parse_profile_params("A=2.0,n=4")
        assert params == {"A": 2.0, "n": 4}

    def test_with_spaces(self):
        params = parse_profile_params("sigma = 0.5 , A = 1.5")
        assert params == {"sigma": 0.5, "A": 1.5}


class TestRegistry:
    def test_all_profiles_registered(self):
        expected = {"parabolic", "gaussian", "flat_top", "cosine", "linear"}
        assert set(PROFILE_REGISTRY.keys()) == expected

    def test_all_profiles_have_defaults(self):
        assert set(DEFAULT_PARAMS.keys()) == set(PROFILE_REGISTRY.keys())

    def test_get_available_profiles(self):
        profiles = get_available_profiles()
        assert set(profiles) == set(PROFILE_REGISTRY.keys())


class TestAllProfilesCommon:
    """Common tests that should pass for all profile types."""

    @pytest.mark.parametrize("profile_name", PROFILE_REGISTRY.keys())
    def test_boundary_value_at_origin(self, grid, profile_name):
        T = make_profile(grid, profile_name)
        # All profiles should have T(0) = A (default A=1.0)
        assert T[0] == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.parametrize("profile_name", PROFILE_REGISTRY.keys())
    def test_boundary_value_at_edge(self, grid, profile_name):
        T = make_profile(grid, profile_name)
        # All profiles should have T(1) = 0
        assert T[-1] == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("profile_name", PROFILE_REGISTRY.keys())
    def test_positive_in_interior(self, grid, profile_name):
        T = make_profile(grid, profile_name)
        # All profiles should be non-negative in interior
        assert np.all(T[:-1] >= 0)

    @pytest.mark.parametrize("profile_name", PROFILE_REGISTRY.keys())
    def test_shape_preserved(self, grid, profile_name):
        T = make_profile(grid, profile_name)
        assert T.shape == grid.shape
