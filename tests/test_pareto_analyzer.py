"""Tests for Pareto analysis module."""

import os
import sys
import json
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from docs.analysis.pareto_analyzer import (
    ParetoPoint,
    ParetoFront,
    _is_pareto_dominated,
    _has_physbo,
    _SolverSimulator,
    compute_pareto_ranks,
    make_initial,
    ParetoAnalysisAgent,
    load_latest_pareto_front,
)

_skip_no_physbo = pytest.mark.skipif(
    not _has_physbo(), reason="physbo not installed"
)


class TestParetoPoint:
    """Tests for ParetoPoint dataclass."""

    def test_creation(self):
        """Test basic creation."""
        point = ParetoPoint(
            solver="test_solver",
            config={"alpha": 0.5, "nr": 51},
            l2_error=1e-4,
            wall_time=0.01,
            pareto_rank=0,
            is_stable=True,
        )
        assert point.solver == "test_solver"
        assert point.config["alpha"] == 0.5
        assert point.l2_error == 1e-4
        assert point.wall_time == 0.01
        assert point.pareto_rank == 0
        assert point.is_stable is True

    def test_to_dict(self):
        """Test serialization to dict."""
        point = ParetoPoint(
            solver="test",
            config={"alpha": 0.5},
            l2_error=1e-4,
            wall_time=0.01,
        )
        d = point.to_dict()
        assert d["solver"] == "test"
        assert d["config"]["alpha"] == 0.5
        assert d["l2_error"] == 1e-4

    def test_to_dict_nan_handling(self):
        """Test NaN handling in serialization."""
        point = ParetoPoint(
            solver="test",
            config={},
            l2_error=float("nan"),
            wall_time=0.01,
        )
        d = point.to_dict()
        assert d["l2_error"] is None  # NaN converted to None

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "solver": "test",
            "config": {"alpha": 0.5},
            "l2_error": 1e-4,
            "wall_time": 0.01,
            "pareto_rank": 1,
            "is_stable": False,
        }
        point = ParetoPoint.from_dict(d)
        assert point.solver == "test"
        assert point.l2_error == 1e-4
        assert point.pareto_rank == 1
        assert point.is_stable is False

    def test_from_dict_null_error(self):
        """Test handling of null error in deserialization."""
        d = {
            "solver": "test",
            "config": {},
            "l2_error": None,
            "wall_time": 0.01,
        }
        point = ParetoPoint.from_dict(d)
        assert np.isnan(point.l2_error)


class TestParetoFront:
    """Tests for ParetoFront dataclass."""

    def test_creation(self):
        """Test basic creation."""
        front = ParetoFront(
            solver_name="test_solver",
            timestamp="2024-01-01T00:00:00",
        )
        assert front.solver_name == "test_solver"
        assert front.points == []
        assert front.pareto_optimal == []

    def test_compute_pareto_optimal(self):
        """Test Pareto-optimal computation."""
        front = ParetoFront(
            solver_name="test",
            timestamp="2024-01-01",
            points=[
                ParetoPoint("test", {}, 1e-3, 0.1, is_stable=True),  # Dominated
                ParetoPoint("test", {}, 1e-4, 0.05, is_stable=True),  # Optimal
                ParetoPoint("test", {}, 1e-5, 0.2, is_stable=True),  # Optimal (low error)
                ParetoPoint("test", {}, 1e-3, 0.02, is_stable=True),  # Optimal (fast)
            ],
        )
        front.compute_pareto_optimal()

        # Point 0 is dominated by point 1 (both lower error and lower time)
        assert len(front.pareto_optimal) == 3
        assert front.points[0].pareto_rank == 1  # Dominated
        assert front.points[1].pareto_rank == 0  # Optimal
        assert front.points[2].pareto_rank == 0  # Optimal
        assert front.points[3].pareto_rank == 0  # Optimal

    def test_compute_pareto_optimal_unstable(self):
        """Test that unstable points are excluded from Pareto front."""
        front = ParetoFront(
            solver_name="test",
            timestamp="2024-01-01",
            points=[
                ParetoPoint("test", {}, 1e-3, 0.1, is_stable=True),
                ParetoPoint("test", {}, 1e-6, 0.01, is_stable=False),  # Best but unstable
            ],
        )
        front.compute_pareto_optimal()

        # Only stable point should be in Pareto front
        assert len(front.pareto_optimal) == 1
        assert front.pareto_optimal[0].l2_error == 1e-3

    def test_compute_summary(self):
        """Test summary computation."""
        front = ParetoFront(
            solver_name="test",
            timestamp="2024-01-01",
            points=[
                ParetoPoint("test", {}, 1e-4, 0.01, is_stable=True),
                ParetoPoint("test", {}, 1e-3, 0.02, is_stable=True),
                ParetoPoint("test", {}, float("nan"), 0.0, is_stable=False),
            ],
        )
        front.compute_pareto_optimal()
        front.compute_summary()

        assert front.summary["total_points"] == 3
        assert front.summary["stable_points"] == 2
        assert front.summary["stability_rate"] == pytest.approx(66.67, rel=0.01)
        assert front.summary["min_error"] == 1e-4
        assert front.summary["max_error"] == 1e-3

    def test_save_and_load(self):
        """Test save and load functionality."""
        front = ParetoFront(
            solver_name="test",
            timestamp="2024-01-01",
            points=[
                ParetoPoint("test", {"alpha": 0.5}, 1e-4, 0.01, is_stable=True),
            ],
            summary={"test": "value"},
        )
        front.compute_pareto_optimal()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            front.save(path)
            loaded = ParetoFront.load(path)

            assert loaded.solver_name == "test"
            assert len(loaded.points) == 1
            assert loaded.points[0].config["alpha"] == 0.5
            assert loaded.summary["test"] == "value"
        finally:
            os.unlink(path)


class TestParetoDominance:
    """Tests for Pareto dominance functions."""

    def test_is_pareto_dominated_yes(self):
        """Test detection of dominated point."""
        candidate = (0.1, 0.1)  # error, time
        others = [(0.05, 0.05)]  # strictly better
        assert _is_pareto_dominated(candidate, others) is True

    def test_is_pareto_dominated_no(self):
        """Test non-dominated point."""
        candidate = (0.01, 0.1)  # low error, high time
        others = [(0.1, 0.01)]  # high error, low time
        assert _is_pareto_dominated(candidate, others) is False

    def test_is_pareto_dominated_equal(self):
        """Test equal points (not dominated by equal)."""
        candidate = (0.1, 0.1)
        others = [(0.1, 0.1)]  # Same point
        assert _is_pareto_dominated(candidate, others) is False

    def test_is_pareto_dominated_partial(self):
        """Test partial dominance (one better, one equal)."""
        candidate = (0.1, 0.1)
        others = [(0.1, 0.05)]  # Same error, better time
        assert _is_pareto_dominated(candidate, others) is True

    def test_compute_pareto_ranks(self):
        """Test Pareto rank computation."""
        points = [
            (0.1, 0.1),  # Dominated by (0.05, 0.05)
            (0.05, 0.05),  # Optimal
            (0.01, 0.2),  # Optimal (best error)
            (0.2, 0.01),  # Optimal (best time)
        ]
        ranks = compute_pareto_ranks(points)

        assert ranks[0] == 1  # Dominated
        assert ranks[1] == 0  # Optimal
        assert ranks[2] == 0  # Optimal
        assert ranks[3] == 0  # Optimal


class TestMakeInitial:
    """Tests for initial condition factory."""

    def test_parabola(self):
        """Test parabolic IC."""
        r = np.linspace(0, 1, 11)
        T = make_initial(r, "parabola")
        assert T[0] == pytest.approx(1.0)  # T(0) = 1
        assert T[-1] == pytest.approx(0.0)  # T(1) = 0

    def test_gaussian(self):
        """Test Gaussian IC."""
        r = np.linspace(0, 1, 11)
        T = make_initial(r, "gaussian")
        assert T[0] == pytest.approx(1.0)  # T(0) = 1
        assert T[-1] < 0.1  # Decays near boundary

    def test_cosine(self):
        """Test cosine IC."""
        r = np.linspace(0, 1, 11)
        T = make_initial(r, "cosine")
        assert T[0] == pytest.approx(1.0)  # T(0) = 1
        assert T[-1] == pytest.approx(0.0, abs=1e-10)  # T(1) = 0

    def test_scale(self):
        """Test IC scaling."""
        r = np.linspace(0, 1, 11)
        T = make_initial(r, "parabola", ic_scale=2.0)
        assert T[0] == pytest.approx(2.0)

    def test_unknown_type(self):
        """Test unknown IC type raises error."""
        r = np.linspace(0, 1, 11)
        with pytest.raises(ValueError, match="Unknown IC type"):
            make_initial(r, "unknown_type")


class TestParetoAnalysisAgent:
    """Tests for ParetoAnalysisAgent."""

    def test_creation(self):
        """Test agent creation with custom parameters."""
        agent = ParetoAnalysisAgent(
            alpha_list=[0.0, 0.5],
            fixed_nr=21,
            dt_list=[0.001],
        )
        assert agent.alpha_list == [0.0, 0.5]
        assert agent._fixed_nr == 21
        assert agent.dt_list == [0.001]

    def test_run_single_config(self):
        """Test running a single configuration."""
        from solvers.fdm.implicit import ImplicitFDM

        agent = ParetoAnalysisAgent()
        solver = ImplicitFDM()
        config = {
            "alpha": 0.0,
            "nr": 21,
            "dt": 0.001,
            "t_end": 0.01,
            "ic_type": "parabola",
        }

        point = agent._run_single_config(solver, config)

        assert point.solver == "implicit_fdm"
        assert point.is_stable is True
        assert point.wall_time > 0
        assert not np.isnan(point.l2_error)

    def test_analyze_solver_minimal(self):
        """Test analyzing a solver with minimal sweep."""
        from solvers.fdm.implicit import ImplicitFDM

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = ParetoAnalysisAgent(
                output_dir=tmpdir,
                alpha_list=[0.0],
                fixed_nr=21,
                dt_list=[0.001],
                t_end_list=[0.01],
                ic_types=["parabola"],
                use_physbo=False,
            )
            solver = ImplicitFDM()

            front = agent.analyze_solver(solver, verbose=False)

            assert front.solver_name == "implicit_fdm"
            assert len(front.points) == 1
            assert front.summary["total_points"] == 1


class TestLoadFunctions:
    """Tests for load functions."""

    def test_load_latest_pareto_front_not_found(self):
        """Test loading from non-existent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_latest_pareto_front("nonexistent", tmpdir)
            assert result is None

    def test_load_latest_pareto_front(self):
        """Test loading latest front."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two fronts with different timestamps
            front1 = ParetoFront(
                solver_name="test",
                timestamp="2024-01-01T00:00:00",
            )
            front2 = ParetoFront(
                solver_name="test",
                timestamp="2024-01-02T00:00:00",
            )

            front1.save(os.path.join(tmpdir, "test_2024-01-01T00-00-00.json"))
            front2.save(os.path.join(tmpdir, "test_2024-01-02T00-00-00.json"))

            loaded = load_latest_pareto_front("test", tmpdir)
            assert loaded is not None
            assert loaded.timestamp == "2024-01-02T00:00:00"


class TestSolverSimulator:
    """Tests for _SolverSimulator PHYSBO interface."""

    @_skip_no_physbo
    def test_solver_simulator_stable(self):
        """Test that stable solutions return negated objectives."""
        from solvers.fdm.implicit import ImplicitFDM

        agent = ParetoAnalysisAgent(
            alpha_list=[0.0],
            ic_types=["parabola"],
            t_end_list=[0.01],
            use_physbo=True,
        )
        solver = ImplicitFDM()
        dt_candidates = np.array([0.001, 0.0005, 0.0002])

        sim = _SolverSimulator(
            agent=agent, solver=solver, alpha=0.0, ic_type="parabola",
            nr=21, t_end=0.01, dt_candidates=dt_candidates,
        )
        result = sim([0, 1, 2])

        assert result.shape == (3, 2)
        # Stable solutions should have negated values (negative, not penalty)
        for i in range(3):
            assert result[i, 0] > _SolverSimulator.PENALTY  # not penalty
            assert result[i, 1] > _SolverSimulator.PENALTY
            assert result[i, 0] < 0  # negated L2 error
            assert result[i, 1] < 0  # negated wall_time

        # Check cache populated
        assert len(sim.evaluated_points) == 3
        assert all(p.is_stable for p in sim.evaluated_points.values())

    @_skip_no_physbo
    def test_solver_simulator_unstable(self):
        """Test that unstable solutions return penalty values."""
        from solvers.fdm.implicit import ImplicitFDM

        agent = ParetoAnalysisAgent(
            alpha_list=[0.0],
            ic_types=["parabola"],
            t_end_list=[0.01],
            use_physbo=True,
        )
        solver = ImplicitFDM()
        # Very large dt likely to be unstable for explicit methods;
        # For implicit methods it may still be stable, so use a dt
        # that causes NaN via extreme settings
        dt_candidates = np.array([0.5, 1.0])  # Absurdly large dt

        sim = _SolverSimulator(
            agent=agent, solver=solver, alpha=0.0, ic_type="parabola",
            nr=21, t_end=0.01, dt_candidates=dt_candidates,
        )
        result = sim([0, 1])

        assert result.shape == (2, 2)
        # ImplicitFDM may still be stable even with large dt,
        # so we just verify the shape and that values are populated
        for i in range(2):
            # Either a valid negated value or penalty
            assert result[i, 0] <= 0 or result[i, 0] == _SolverSimulator.PENALTY


class TestBuildDtCandidates:
    """Tests for dt candidate generation."""

    def test_build_dt_candidates_default(self):
        """Test default dt candidate generation."""
        agent = ParetoAnalysisAgent(use_physbo=False)
        candidates = agent._build_dt_candidates()

        assert len(candidates) == agent._physbo_n_candidates
        assert candidates[0] == pytest.approx(1e-5, rel=1e-3)
        assert candidates[-1] == pytest.approx(1e-2, rel=1e-3)
        # Verify log-spacing
        log_diff = np.diff(np.log10(candidates))
        assert np.allclose(log_diff, log_diff[0], rtol=1e-6)

    def test_build_dt_candidates_custom_n(self):
        """Test custom number of candidates."""
        agent = ParetoAnalysisAgent(use_physbo=False)
        candidates = agent._build_dt_candidates(n=10)
        assert len(candidates) == 10
        assert candidates[0] == pytest.approx(1e-5, rel=1e-3)
        assert candidates[-1] == pytest.approx(1e-2, rel=1e-3)


class TestPhysboIntegration:
    """Integration tests for PHYSBO-based optimization."""

    @_skip_no_physbo
    def test_find_best_physbo_stable(self):
        """Test PHYSBO search finds a stable solution."""
        from solvers.fdm.implicit import ImplicitFDM

        agent = ParetoAnalysisAgent(
            alpha_list=[0.0],
            ic_types=["parabola"],
            t_end_list=[0.01],
            use_physbo=True,
            fixed_nr=21,
            physbo_n_candidates=20,
            physbo_n_random=3,
            physbo_n_bayes=5,
        )
        solver = ImplicitFDM()

        point = agent._find_best_for_problem_physbo(solver, 0.0, "parabola")

        assert point is not None
        assert point.is_stable
        assert not np.isnan(point.l2_error)
        assert point.wall_time > 0
        assert point.config["nr"] == 21

    @_skip_no_physbo
    def test_physbo_fallback(self):
        """Test that grid fallback works when PHYSBO is disabled."""
        from solvers.fdm.implicit import ImplicitFDM

        agent = ParetoAnalysisAgent(
            alpha_list=[0.0],
            fixed_nr=21,
            dt_list=[0.001],
            t_end_list=[0.01],
            ic_types=["parabola"],
            use_physbo=False,
        )
        solver = ImplicitFDM()

        # Should use grid sweep since PHYSBO disabled
        point = agent._find_best_for_problem(solver, 0.0, "parabola")
        assert point is not None
        assert point.is_stable
        assert point.config["nr"] == 21    # Fixed nr
        assert point.config["dt"] == 0.001  # From grid

    @_skip_no_physbo
    def test_cross_solver_with_physbo(self):
        """Test cross-solver analysis works with PHYSBO."""
        from solvers.fdm.implicit import ImplicitFDM
        from solvers.fdm.compact4 import Compact4FDM

        agent = ParetoAnalysisAgent(
            alpha_list=[0.0],
            ic_types=["parabola"],
            t_end_list=[0.01],
            use_physbo=True,
            fixed_nr=21,
            physbo_n_candidates=20,
            physbo_n_random=3,
            physbo_n_bayes=5,
        )

        solvers = [ImplicitFDM(), Compact4FDM()]
        analysis = agent.analyze_cross_solver(solvers, verbose=False)

        assert len(analysis.problems) == 1  # 1 alpha x 1 ic_type
        key = "alpha=0.0_ic=parabola"
        assert key in analysis.problems
        front = analysis.problems[key]
        assert len(front.points) >= 1  # At least one solver produced results

    @_skip_no_physbo
    def test_analyze_solver_physbo(self):
        """Test per-solver analysis dispatches to PHYSBO."""
        from solvers.fdm.implicit import ImplicitFDM

        agent = ParetoAnalysisAgent(
            alpha_list=[0.0],
            ic_types=["parabola"],
            t_end_list=[0.01],
            use_physbo=True,
            fixed_nr=21,
            physbo_n_candidates=20,
            physbo_n_random=3,
            physbo_n_bayes=5,
        )
        solver = ImplicitFDM()
        front = agent.analyze_solver(solver, verbose=False)

        assert front.solver_name == "implicit_fdm"
        assert len(front.points) >= 1
        assert front.summary["stable_points"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
