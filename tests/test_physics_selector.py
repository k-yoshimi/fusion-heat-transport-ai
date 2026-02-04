"""Tests for physics-only solver selector."""

import pytest
import numpy as np
import os
import tempfile

from policy.physics_selector import (
    PHYSICS_FEATURE_NAMES,
    extract_physics_features,
    PhysicsSolverSelector,
    select_with_physics,
)


@pytest.fixture
def sample_profile():
    """Create sample temperature profile."""
    r = np.linspace(0, 1, 51)
    T0 = 1.0 - r**2  # Parabolic
    return T0, r


@pytest.fixture
def gaussian_profile():
    """Create gaussian temperature profile."""
    r = np.linspace(0, 1, 51)
    T0 = np.exp(-r**2 / (2 * 0.3**2))
    return T0, r


class TestPhysicsFeatureNames:
    """Tests for physics feature names."""

    def test_feature_count(self):
        """13 physics features defined."""
        assert len(PHYSICS_FEATURE_NAMES) == 13

    def test_excludes_numerical_params(self):
        """Physics features exclude dt, nr, t_end."""
        assert "dt" not in PHYSICS_FEATURE_NAMES
        assert "nr" not in PHYSICS_FEATURE_NAMES
        assert "t_end" not in PHYSICS_FEATURE_NAMES

    def test_includes_alpha(self):
        """Alpha is included in physics features."""
        assert "alpha" in PHYSICS_FEATURE_NAMES


class TestExtractPhysicsFeatures:
    """Tests for extract_physics_features function."""

    def test_returns_all_features(self, sample_profile):
        """Returns dict with all expected features."""
        T0, r = sample_profile
        feats = extract_physics_features(T0, r, alpha=0.5)
        for name in PHYSICS_FEATURE_NAMES:
            assert name in feats

    def test_alpha_in_features(self, sample_profile):
        """Alpha is correctly included in features."""
        T0, r = sample_profile
        feats = extract_physics_features(T0, r, alpha=1.5)
        assert feats["alpha"] == 1.5

    def test_positive_gradients(self, sample_profile):
        """Gradient-based features are non-negative."""
        T0, r = sample_profile
        feats = extract_physics_features(T0, r, alpha=0.5)
        assert feats["max_abs_gradient"] >= 0
        assert feats["max_chi"] >= 0.1  # Minimum chi is 0.1

    def test_profile_shape_features(self, sample_profile):
        """Profile shape features are in valid ranges."""
        T0, r = sample_profile
        feats = extract_physics_features(T0, r, alpha=0.5)
        assert 0 <= feats["half_max_radius"] <= 1
        assert 0 <= feats["profile_centroid"] <= 1
        assert 0 <= feats["profile_width"] <= 1

    def test_different_profiles_different_features(self, sample_profile, gaussian_profile):
        """Different profiles produce different features."""
        T0_para, r_para = sample_profile
        T0_gauss, r_gauss = gaussian_profile

        feats_para = extract_physics_features(T0_para, r_para, alpha=0.5)
        feats_gauss = extract_physics_features(T0_gauss, r_gauss, alpha=0.5)

        # At least some features should differ
        differences = [
            feats_para[k] != feats_gauss[k]
            for k in PHYSICS_FEATURE_NAMES
        ]
        assert any(differences)


class TestPhysicsSolverSelector:
    """Tests for PhysicsSolverSelector class."""

    def test_initialization_without_model(self):
        """Selector initializes without model file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "nonexistent.npz")
            selector = PhysicsSolverSelector(model_path)
            assert selector.tree is None

    def test_predict_without_model(self, sample_profile):
        """Predict falls back to rule-based without model."""
        T0, r = sample_profile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "nonexistent.npz")
            selector = PhysicsSolverSelector(model_path)
            prediction = selector.predict(T0, r, alpha=0.5)
            assert isinstance(prediction, str)
            assert prediction != ""

    def test_rule_based_high_alpha(self, sample_profile):
        """Rule-based selector chooses implicit for high alpha."""
        T0, r = sample_profile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "nonexistent.npz")
            selector = PhysicsSolverSelector(model_path)
            prediction = selector._rule_based_fallback(T0, r, alpha=2.0)
            assert prediction == "implicit_fdm"

    def test_rule_based_low_alpha_smooth(self, sample_profile):
        """Rule-based selector can choose spectral for smooth low-alpha."""
        T0, r = sample_profile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "nonexistent.npz")
            selector = PhysicsSolverSelector(model_path)
            prediction = selector._rule_based_fallback(T0, r, alpha=0.0)
            # Should be spectral or implicit (both valid)
            assert prediction in [
                "spectral_cosine",
                "chebyshev_spectral",
                "implicit_fdm",
            ]

    def test_predict_with_confidence(self, sample_profile):
        """predict_with_confidence returns dict."""
        T0, r = sample_profile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "nonexistent.npz")
            selector = PhysicsSolverSelector(model_path)
            confidence = selector.predict_with_confidence(T0, r, alpha=0.5)
            assert isinstance(confidence, dict)
            assert len(confidence) >= 1
            assert sum(confidence.values()) == pytest.approx(1.0)


class TestSelectWithPhysics:
    """Tests for select_with_physics convenience function."""

    def test_basic_call(self, sample_profile):
        """Basic function call works."""
        T0, r = sample_profile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "nonexistent.npz")
            prediction = select_with_physics(T0, r, alpha=0.5, model_path=model_path)
            assert isinstance(prediction, str)

    def test_returns_valid_solver(self, sample_profile):
        """Returns a known solver name."""
        T0, r = sample_profile
        known_solvers = [
            "implicit_fdm",
            "compact4_fdm",
            "p2_fem",
            "cell_centered_fvm",
            "chebyshev_spectral",
            "spectral_cosine",
            "imex_fdm",
            "pinn_stub",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "nonexistent.npz")
            prediction = select_with_physics(T0, r, alpha=0.5, model_path=model_path)
            assert prediction in known_solvers


class TestPhysicsSelectorIntegration:
    """Integration tests for physics selector with optimizer."""

    def test_two_stage_workflow(self, sample_profile):
        """Two-stage workflow produces valid result."""
        from policy.optimizer import ParameterOptimizer

        T0, r = sample_profile
        alpha = 0.5
        t_end = 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "nonexistent.npz")

            # Stage 1: Physics selector
            selector = PhysicsSolverSelector(model_path)
            solver_name = selector.predict(T0, r, alpha)

            # Stage 2: Parameter optimizer
            optimizer = ParameterOptimizer()
            result = optimizer.optimize_for_solver(
                solver_name, T0, r, alpha, t_end, target_error=0.005
            )

            assert result.dt > 0
            assert result.nr >= 11
            assert isinstance(result.constraint_satisfied, bool)

    def test_multiple_alpha_values(self, sample_profile):
        """Selector works for multiple alpha values."""
        T0, r = sample_profile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "nonexistent.npz")
            selector = PhysicsSolverSelector(model_path)

            for alpha in [0.0, 0.5, 1.0, 1.5, 2.0]:
                prediction = selector.predict(T0, r, alpha)
                assert isinstance(prediction, str)
                assert prediction != ""
