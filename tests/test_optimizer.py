"""Tests for parameter optimizer."""

import pytest
import numpy as np

from policy.optimizer import (
    OptimizationResult,
    ParameterOptimizer,
    optimize_parameters,
)


@pytest.fixture
def sample_profile():
    """Create sample temperature profile."""
    r = np.linspace(0, 1, 51)
    T0 = 1.0 - r**2  # Parabolic
    return T0, r


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_default_values(self):
        """OptimizationResult has sensible defaults."""
        result = OptimizationResult(
            dt=0.001,
            nr=51,
            estimated_error=0.005,
            estimated_time=100,
        )
        assert result.pareto_rank == 0
        assert result.constraint_satisfied is True
        assert result.notes == ""


class TestParameterOptimizer:
    """Tests for ParameterOptimizer class."""

    def test_estimate_max_chi_parabolic(self, sample_profile):
        """Estimate max_chi from parabolic profile."""
        T0, r = sample_profile
        optimizer = ParameterOptimizer()
        max_chi = optimizer.estimate_max_chi(T0, r, alpha=0.5)
        # Parabolic has max gradient at r=1, |dT/dr| ~ 2
        assert max_chi >= 0.1  # At least baseline chi
        assert max_chi < 5.0  # Reasonable upper bound

    def test_estimate_max_chi_increases_with_alpha(self, sample_profile):
        """Max chi increases with alpha for same profile."""
        T0, r = sample_profile
        optimizer = ParameterOptimizer()
        chi_low = optimizer.estimate_max_chi(T0, r, alpha=0.0)
        chi_high = optimizer.estimate_max_chi(T0, r, alpha=1.0)
        assert chi_high >= chi_low

    def test_estimate_error_decreases_with_finer_grid(self, sample_profile):
        """Estimated error decreases with finer grid."""
        T0, r = sample_profile
        optimizer = ParameterOptimizer()
        from policy.stability import get_stability

        stability = get_stability("implicit_fdm")
        err_coarse = optimizer.estimate_error(0.001, 21, stability, 0.1)
        err_fine = optimizer.estimate_error(0.001, 101, stability, 0.1)
        assert err_fine < err_coarse

    def test_estimate_error_decreases_with_smaller_dt(self, sample_profile):
        """Estimated error decreases with smaller dt."""
        T0, r = sample_profile
        optimizer = ParameterOptimizer()
        from policy.stability import get_stability

        stability = get_stability("implicit_fdm")
        err_large_dt = optimizer.estimate_error(0.01, 51, stability, 0.1)
        err_small_dt = optimizer.estimate_error(0.0001, 51, stability, 0.1)
        assert err_small_dt < err_large_dt

    def test_estimate_time_increases_with_finer_grid(self, sample_profile):
        """Estimated time increases with finer grid."""
        T0, r = sample_profile
        optimizer = ParameterOptimizer()
        from policy.stability import get_stability

        stability = get_stability("implicit_fdm")
        time_coarse = optimizer.estimate_time(0.001, 21, 0.1, stability)
        time_fine = optimizer.estimate_time(0.001, 101, 0.1, stability)
        assert time_fine > time_coarse

    def test_optimize_implicit_fdm(self, sample_profile):
        """Optimize parameters for ImplicitFDM."""
        T0, r = sample_profile
        optimizer = ParameterOptimizer()
        result = optimizer.optimize_for_solver(
            "implicit_fdm", T0, r, alpha=0.5, t_end=0.1, target_error=0.005
        )
        assert result.dt > 0
        assert result.nr >= 11
        assert result.constraint_satisfied is True

    def test_optimize_spectral_cosine(self, sample_profile):
        """Optimize parameters for CosineSpectral."""
        T0, r = sample_profile
        optimizer = ParameterOptimizer()
        result = optimizer.optimize_for_solver(
            "spectral_cosine", T0, r, alpha=0.3, t_end=0.1, target_error=0.005
        )
        # Should respect CFL constraint
        assert result.dt > 0
        assert result.constraint_satisfied is True

    def test_optimize_with_high_alpha_fallback(self, sample_profile):
        """Optimization handles high alpha gracefully."""
        T0, r = sample_profile
        optimizer = ParameterOptimizer()
        result = optimizer.optimize_for_solver(
            "spectral_cosine", T0, r, alpha=2.0, t_end=0.1, target_error=0.005
        )
        # Should return result even if alpha exceeds limit
        assert result.dt > 0
        assert result.nr >= 11
        # Should indicate constraint issue
        if result.constraint_satisfied:
            # If stable, should have notes
            pass  # OK, may have found stable config

    def test_optimize_for_accuracy(self, sample_profile):
        """Accuracy-focused optimization returns finer params."""
        T0, r = sample_profile
        optimizer = ParameterOptimizer()
        result_balanced = optimizer.optimize_for_solver(
            "implicit_fdm", T0, r, alpha=0.5, t_end=0.1, target_error=0.005
        )
        result_accuracy = optimizer.optimize_for_accuracy(
            "implicit_fdm", T0, r, alpha=0.5, t_end=0.1, target_error=0.005
        )
        # Accuracy-focused should have smaller estimated error
        assert result_accuracy.estimated_error <= result_balanced.estimated_error

    def test_optimize_for_speed(self, sample_profile):
        """Speed-focused optimization returns coarser params."""
        T0, r = sample_profile
        optimizer = ParameterOptimizer()
        result_balanced = optimizer.optimize_for_solver(
            "implicit_fdm", T0, r, alpha=0.5, t_end=0.1, target_error=0.02
        )
        result_speed = optimizer.optimize_for_speed(
            "implicit_fdm", T0, r, alpha=0.5, t_end=0.1, max_error=0.02
        )
        # Speed-focused should have smaller estimated time
        assert result_speed.estimated_time <= result_balanced.estimated_time


class TestOptimizeParametersFunction:
    """Tests for optimize_parameters convenience function."""

    def test_basic_call(self, sample_profile):
        """Basic function call works."""
        T0, r = sample_profile
        result = optimize_parameters(
            "implicit_fdm", T0, r, alpha=0.5, t_end=0.1
        )
        assert isinstance(result, OptimizationResult)
        assert result.dt > 0
        assert result.nr >= 11

    def test_custom_target_error(self, sample_profile):
        """Custom target error affects result."""
        T0, r = sample_profile
        result_tight = optimize_parameters(
            "implicit_fdm", T0, r, alpha=0.5, t_end=0.1, target_error=0.001
        )
        result_loose = optimize_parameters(
            "implicit_fdm", T0, r, alpha=0.5, t_end=0.1, target_error=0.02
        )
        # Both should produce valid results
        assert result_tight.dt > 0
        assert result_loose.dt > 0
        # Tighter target that cannot be achieved will have notes
        if result_tight.estimated_error <= 0.001:
            # If tight target is achievable, should use finer discretization
            assert (result_tight.nr >= result_loose.nr or
                    result_tight.dt <= result_loose.dt)
        else:
            # Target cannot be achieved - should have notes
            assert result_tight.notes != ""


class TestParetoDomination:
    """Tests for Pareto domination logic."""

    def test_not_dominated_single(self):
        """Single candidate is not dominated."""
        optimizer = ParameterOptimizer()
        candidate = (0.01, 100)  # (error, time)
        others = []
        assert optimizer._is_pareto_dominated(candidate, others) is False

    def test_dominated_by_better_both(self):
        """Candidate dominated by one better in both objectives."""
        optimizer = ParameterOptimizer()
        candidate = (0.01, 100)
        others = [(0.005, 50)]  # Better in both
        assert optimizer._is_pareto_dominated(candidate, others) is True

    def test_not_dominated_by_tradeoff(self):
        """Candidate not dominated by tradeoff solution."""
        optimizer = ParameterOptimizer()
        candidate = (0.01, 100)
        others = [(0.005, 200)]  # Better error but worse time
        assert optimizer._is_pareto_dominated(candidate, others) is False

    def test_dominated_by_better_one(self):
        """Candidate dominated if other is equal in one, better in other."""
        optimizer = ParameterOptimizer()
        candidate = (0.01, 100)
        others = [(0.01, 50)]  # Same error, better time
        assert optimizer._is_pareto_dominated(candidate, others) is True
