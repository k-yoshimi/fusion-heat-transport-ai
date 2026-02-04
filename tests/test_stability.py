"""Tests for stability constraints and metadata."""

import pytest
import numpy as np

from policy.stability import (
    StabilityConstraints,
    SOLVER_STABILITY,
    get_stability,
    is_solver_stable,
    suggest_stable_dt,
)


class TestStabilityConstraints:
    """Tests for StabilityConstraints dataclass."""

    def test_unconditionally_stable_max_dt(self):
        """Unconditionally stable methods return inf max_dt."""
        sc = StabilityConstraints(is_unconditionally_stable=True)
        assert sc.compute_max_dt(dr=0.02, max_chi=1.0) == float("inf")

    def test_cfl_max_dt(self):
        """CFL-limited methods compute correct max_dt."""
        sc = StabilityConstraints(
            is_unconditionally_stable=False,
            cfl_coefficient=0.5,
        )
        dr = 0.02
        max_chi = 0.5
        expected = 0.5 * dr * dr / max_chi
        assert sc.compute_max_dt(dr, max_chi) == pytest.approx(expected)

    def test_is_stable_within_limit(self):
        """Configurations within CFL limit are stable."""
        sc = StabilityConstraints(
            is_unconditionally_stable=False,
            cfl_coefficient=0.5,
        )
        dr = 0.02
        max_chi = 0.5
        dt = 0.0003  # Well below CFL limit
        assert sc.is_stable(dt, dr, max_chi) is True

    def test_is_stable_exceeds_limit(self):
        """Configurations exceeding CFL limit are unstable."""
        sc = StabilityConstraints(
            is_unconditionally_stable=False,
            cfl_coefficient=0.5,
        )
        dr = 0.02
        max_chi = 0.5
        dt = 0.01  # Well above CFL limit
        assert sc.is_stable(dt, dr, max_chi) is False

    def test_validate_alpha_no_limit(self):
        """No max_alpha means all alpha values are valid."""
        sc = StabilityConstraints(max_alpha=None)
        assert sc.validate_alpha(10.0) is True

    def test_validate_alpha_within_limit(self):
        """Alpha within max_alpha is valid."""
        sc = StabilityConstraints(max_alpha=0.5)
        assert sc.validate_alpha(0.3) is True

    def test_validate_alpha_exceeds_limit(self):
        """Alpha exceeding max_alpha is invalid."""
        sc = StabilityConstraints(max_alpha=0.5)
        assert sc.validate_alpha(1.0) is False


class TestSolverStabilityRegistry:
    """Tests for SOLVER_STABILITY registry."""

    def test_implicit_fdm_unconditionally_stable(self):
        """ImplicitFDM is unconditionally stable."""
        sc = SOLVER_STABILITY["implicit_fdm"]
        assert sc.is_unconditionally_stable is True
        assert sc.cfl_coefficient is None

    def test_spectral_cosine_has_cfl(self):
        """CosineSpectral has CFL constraint."""
        sc = SOLVER_STABILITY["spectral_cosine"]
        assert sc.is_unconditionally_stable is False
        assert sc.cfl_coefficient == 0.5
        assert sc.max_alpha == 0.5

    def test_imex_fdm_has_cfl(self):
        """IMEXFDM has CFL constraint."""
        sc = SOLVER_STABILITY["imex_fdm"]
        assert sc.is_unconditionally_stable is False
        assert sc.cfl_coefficient == 0.4

    def test_all_solvers_have_entries(self):
        """All expected solvers have stability entries."""
        expected_solvers = [
            "implicit_fdm",
            "compact4_fdm",
            "p2_fem",
            "cell_centered_fvm",
            "chebyshev_spectral",
            "spectral_cosine",
            "imex_fdm",
            "pinn_stub",
        ]
        for name in expected_solvers:
            assert name in SOLVER_STABILITY


class TestGetStability:
    """Tests for get_stability function."""

    def test_known_solver(self):
        """get_stability returns constraints for known solver."""
        sc = get_stability("implicit_fdm")
        assert sc.is_unconditionally_stable is True

    def test_unknown_solver(self):
        """get_stability raises KeyError for unknown solver."""
        with pytest.raises(KeyError):
            get_stability("nonexistent_solver")


class TestIsSolverStable:
    """Tests for is_solver_stable function."""

    def test_implicit_fdm_always_stable(self):
        """ImplicitFDM is stable with any dt."""
        is_stable, msg = is_solver_stable(
            "implicit_fdm",
            dt=0.1,
            nr=51,
            max_chi=1.0,
            alpha=2.0,
        )
        assert is_stable is True
        assert msg == "OK"

    def test_spectral_cosine_unstable_with_large_dt(self):
        """CosineSpectral is unstable with large dt."""
        is_stable, msg = is_solver_stable(
            "spectral_cosine",
            dt=0.01,  # Large dt
            nr=51,
            max_chi=0.5,
            alpha=0.3,
        )
        assert is_stable is False
        assert "CFL" in msg

    def test_spectral_cosine_unstable_with_large_alpha(self):
        """CosineSpectral is unstable with large alpha."""
        is_stable, msg = is_solver_stable(
            "spectral_cosine",
            dt=0.0001,  # Small dt
            nr=51,
            max_chi=0.5,
            alpha=1.0,  # Large alpha
        )
        assert is_stable is False
        assert "alpha" in msg.lower()

    def test_spectral_cosine_stable_with_small_params(self):
        """CosineSpectral is stable with appropriate params."""
        is_stable, msg = is_solver_stable(
            "spectral_cosine",
            dt=0.0001,
            nr=51,
            max_chi=0.2,
            alpha=0.3,
        )
        assert is_stable is True
        assert msg == "OK"

    def test_unknown_solver(self):
        """Unknown solver returns False with error message."""
        is_stable, msg = is_solver_stable(
            "nonexistent",
            dt=0.001,
            nr=51,
            max_chi=0.5,
            alpha=0.0,
        )
        assert is_stable is False
        assert "Unknown" in msg


class TestSuggestStableDt:
    """Tests for suggest_stable_dt function."""

    def test_implicit_fdm_suggestion(self):
        """ImplicitFDM gets heuristic-based suggestion."""
        dt = suggest_stable_dt("implicit_fdm", nr=51, max_chi=0.5)
        assert dt > 0
        assert dt <= 0.001  # Heuristic default

    def test_spectral_cosine_suggestion(self):
        """CosineSpectral gets CFL-based suggestion."""
        nr = 51
        max_chi = 0.5
        dr = 1.0 / (nr - 1)
        cfl_limit = 0.5 * dr * dr / max_chi

        dt = suggest_stable_dt("spectral_cosine", nr=nr, max_chi=max_chi)
        assert dt > 0
        assert dt < cfl_limit  # With safety factor

    def test_safety_factor(self):
        """Safety factor reduces suggested dt."""
        dt_default = suggest_stable_dt("spectral_cosine", nr=51, max_chi=0.5)
        dt_safe = suggest_stable_dt(
            "spectral_cosine", nr=51, max_chi=0.5, safety_factor=0.5
        )
        assert dt_safe < dt_default
