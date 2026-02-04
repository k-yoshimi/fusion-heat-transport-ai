"""Tests for Method Improvement Cycle module."""

import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from docs.analysis.pareto_analyzer import (
    ParetoPoint, ParetoFront, CrossSolverFront, CrossSolverAnalysis,
)
from docs.analysis.improvement_agents import (
    MethodProposal,
    Bottleneck,
    EvaluationResult,
    BottleneckAnalysisAgent,
    ProposalGenerationAgent,
    EvaluationAgent,
    ReportAgent,
)
from docs.analysis.method_improvement_cycle import (
    ImprovementCycleState,
    ImprovementHistory,
    CycleCoordinator,
)


class TestMethodProposal:
    """Tests for MethodProposal dataclass."""

    def test_creation(self):
        """Test basic creation."""
        proposal = MethodProposal(
            proposal_id="P001",
            proposal_type="parameter_tuning",
            title="Test Proposal",
            description="A test proposal",
            rationale="For testing",
            expected_benefit="Better tests",
            implementation_sketch="# Code here",
            cycle_id=1,
        )
        assert proposal.proposal_id == "P001"
        assert proposal.proposal_type == "parameter_tuning"
        assert proposal.status == "proposed"
        assert proposal.cycle_id == 1

    def test_to_dict(self):
        """Test serialization."""
        proposal = MethodProposal(
            proposal_id="P001",
            proposal_type="algorithm_tweak",
            title="Test",
            description="Desc",
            rationale="Reason",
            expected_benefit="Benefit",
            implementation_sketch="Code",
        )
        d = proposal.to_dict()
        assert d["proposal_id"] == "P001"
        assert d["status"] == "proposed"

    def test_from_dict(self):
        """Test deserialization."""
        d = {
            "proposal_id": "P002",
            "proposal_type": "new_solver",
            "title": "New Solver",
            "description": "Desc",
            "rationale": "Reason",
            "expected_benefit": "Benefit",
            "implementation_sketch": "Code",
            "status": "approved",
            "created_by": "TestAgent",
            "cycle_id": 2,
            "evaluation_scores": {"accuracy": 4.0},
            "evaluation_notes": ["Note 1"],
        }
        proposal = MethodProposal.from_dict(d)
        assert proposal.proposal_id == "P002"
        assert proposal.status == "approved"
        assert proposal.evaluation_scores["accuracy"] == 4.0


class TestBottleneck:
    """Tests for Bottleneck dataclass."""

    def test_creation(self):
        """Test basic creation."""
        bottleneck = Bottleneck(
            bottleneck_id="B001",
            category="stability",
            severity="high",
            description="Stability issue",
            affected_solvers=["solver1", "solver2"],
            evidence={"rate": 50.0},
        )
        assert bottleneck.bottleneck_id == "B001"
        assert bottleneck.category == "stability"
        assert len(bottleneck.affected_solvers) == 2

    def test_to_dict(self):
        """Test serialization."""
        bottleneck = Bottleneck(
            bottleneck_id="B001",
            category="accuracy_gap",
            severity="medium",
            description="Accuracy gap",
            affected_solvers=["solver1"],
            evidence={"gap": 10.0},
            suggested_actions=["Action 1", "Action 2"],
        )
        d = bottleneck.to_dict()
        assert d["bottleneck_id"] == "B001"
        assert len(d["suggested_actions"]) == 2


class TestImprovementCycleState:
    """Tests for ImprovementCycleState dataclass."""

    def test_creation(self):
        """Test basic creation."""
        state = ImprovementCycleState(
            cycle_id=1,
            started_at="2024-01-01T00:00:00",
        )
        assert state.cycle_id == 1
        assert state.phase == "pareto"
        assert state.pareto_fronts == {}
        assert state.completed_at is None

    def test_to_dict(self):
        """Test serialization."""
        state = ImprovementCycleState(
            cycle_id=1,
            started_at="2024-01-01T00:00:00",
            phase="evaluation",
            proposals=["P001", "P002"],
        )
        d = state.to_dict()
        assert d["cycle_id"] == 1
        assert d["phase"] == "evaluation"
        assert len(d["proposals"]) == 2

    def test_from_dict(self):
        """Test deserialization."""
        d = {
            "cycle_id": 2,
            "started_at": "2024-01-01",
            "completed_at": "2024-01-02",
            "phase": "complete",
            "pareto_fronts": {"solver1": "/path/to/front.json"},
            "bottlenecks": ["B001"],
            "proposals": ["P001"],
            "approved_proposals": ["P001"],
            "notes": "Test notes",
        }
        state = ImprovementCycleState.from_dict(d)
        assert state.cycle_id == 2
        assert state.phase == "complete"
        assert "solver1" in state.pareto_fronts


class TestImprovementHistory:
    """Tests for ImprovementHistory dataclass."""

    def test_creation(self):
        """Test basic creation."""
        history = ImprovementHistory()
        assert history.current_cycle == 0
        assert history.cycles == []
        assert history.all_proposals == {}

    def test_save_and_load(self):
        """Test save and load functionality."""
        history = ImprovementHistory(
            current_cycle=2,
            cycles=[
                ImprovementCycleState(
                    cycle_id=1,
                    started_at="2024-01-01",
                    completed_at="2024-01-01",
                    phase="complete",
                ),
                ImprovementCycleState(
                    cycle_id=2,
                    started_at="2024-01-02",
                    phase="proposal",
                ),
            ],
            all_proposals={"P001": {"proposal_id": "P001", "title": "Test"}},
            implemented_methods=["P001"],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            history.save(path)
            loaded = ImprovementHistory.load(path)

            assert loaded.current_cycle == 2
            assert len(loaded.cycles) == 2
            assert loaded.cycles[0].phase == "complete"
            assert "P001" in loaded.all_proposals
            assert "P001" in loaded.implemented_methods
        finally:
            os.unlink(path)

    def test_load_nonexistent(self):
        """Test loading from non-existent file."""
        history = ImprovementHistory.load("/nonexistent/path.json")
        assert history.current_cycle == 0
        assert history.cycles == []


class TestBottleneckAnalysisAgent:
    """Tests for BottleneckAnalysisAgent."""

    def create_mock_fronts(self):
        """Create mock Pareto fronts for testing."""
        return {
            "solver1": ParetoFront(
                solver_name="solver1",
                timestamp="2024-01-01",
                points=[
                    ParetoPoint("solver1", {"alpha": 0.0}, 1e-4, 0.01, 0, True),
                    ParetoPoint("solver1", {"alpha": 0.5}, 2e-4, 0.02, 0, True),
                ],
                pareto_optimal=[
                    ParetoPoint("solver1", {"alpha": 0.0}, 1e-4, 0.01, 0, True),
                ],
                summary={
                    "stability_rate": 100,
                    "total_points": 2,
                    "stable_points": 2,
                    "pareto_optimal_count": 1,
                },
            ),
            "solver2": ParetoFront(
                solver_name="solver2",
                timestamp="2024-01-01",
                points=[
                    ParetoPoint("solver2", {"alpha": 0.0}, 5e-5, 0.005, 0, True),
                    ParetoPoint("solver2", {"alpha": 0.5}, float("nan"), 0.0, 0, False),
                    ParetoPoint("solver2", {"alpha": 1.0}, float("nan"), 0.0, 0, False),
                ],
                pareto_optimal=[
                    ParetoPoint("solver2", {"alpha": 0.0}, 5e-5, 0.005, 0, True),
                ],
                summary={
                    "stability_rate": 33.3,
                    "total_points": 3,
                    "stable_points": 1,
                    "pareto_optimal_count": 1,
                },
            ),
        }

    def test_analyze_empty(self):
        """Test analysis with empty input."""
        agent = BottleneckAnalysisAgent()
        bottlenecks = agent.analyze({})
        assert bottlenecks == []

    def test_analyze_stability(self):
        """Test stability bottleneck detection."""
        agent = BottleneckAnalysisAgent()
        fronts = self.create_mock_fronts()

        bottlenecks = agent.analyze({"pareto_fronts": fronts})

        # Should find stability issue for solver2
        stability_bottlenecks = [b for b in bottlenecks if b.category == "stability"]
        assert len(stability_bottlenecks) >= 1
        assert "solver2" in stability_bottlenecks[0].affected_solvers

    def test_reasoning_log(self):
        """Test that reasoning is logged."""
        agent = BottleneckAnalysisAgent()
        fronts = self.create_mock_fronts()

        agent.analyze({"pareto_fronts": fronts})

        assert len(agent.reasoning_log) > 0
        assert any("Analyzing" in log for log in agent.reasoning_log)


class TestProposalGenerationAgent:
    """Tests for ProposalGenerationAgent."""

    def test_analyze_empty(self):
        """Test with no bottlenecks."""
        agent = ProposalGenerationAgent()
        proposals = agent.analyze({"bottlenecks": [], "cycle_id": 1})
        assert proposals == []

    def test_analyze_stability_bottleneck(self):
        """Test proposal generation for stability bottleneck."""
        agent = ProposalGenerationAgent()
        bottleneck = Bottleneck(
            bottleneck_id="B001",
            category="stability",
            severity="high",
            description="Low stability rate",
            affected_solvers=["test_solver"],
            evidence={"stability_rate": 50.0, "alpha_pattern": {"min_unstable": 0.5}},
        )

        proposals = agent.analyze({"bottlenecks": [bottleneck], "cycle_id": 1})

        assert len(proposals) >= 1
        # Should generate adaptive time-stepping proposal
        assert any("adaptive" in p.title.lower() for p in proposals)

    def test_unique_proposal_ids(self):
        """Test that proposal IDs are unique."""
        agent = ProposalGenerationAgent()
        bottlenecks = [
            Bottleneck("B001", "stability", "high", "Issue 1", ["s1"], {}),
            Bottleneck("B002", "accuracy_gap", "medium", "Issue 2", ["s2"], {"best_error": 1e-4, "gap_ratio": 100}),
        ]

        proposals = agent.analyze({"bottlenecks": bottlenecks, "cycle_id": 1})

        ids = [p.proposal_id for p in proposals]
        assert len(ids) == len(set(ids))  # All unique


class TestEvaluationAgent:
    """Tests for EvaluationAgent."""

    def test_analyze_empty(self):
        """Test with no proposals."""
        agent = EvaluationAgent()
        results = agent.analyze({"proposals": []})
        assert results == []

    def test_analyze_single_proposal(self):
        """Test evaluation of single proposal."""
        agent = EvaluationAgent()
        proposal = MethodProposal(
            proposal_id="P001",
            proposal_type="parameter_tuning",
            title="Increase resolution",
            description="Increase nr for better accuracy",
            rationale="Accuracy gap exists",
            expected_benefit="Reduce accuracy gap",
            implementation_sketch="nr = nr * 1.5",
        )

        results = agent.analyze({"proposals": [proposal]})

        assert len(results) == 1
        assert results[0].proposal_id == "P001"
        assert results[0].overall_score > 0
        assert "accuracy" in results[0].scores
        assert "speed" in results[0].scores
        assert "stability" in results[0].scores
        assert "complexity" in results[0].scores

    def test_ranking(self):
        """Test that proposals are ranked correctly."""
        agent = EvaluationAgent()
        proposals = [
            MethodProposal(
                "P001", "algorithm_tweak", "Adaptive stepping",
                "Add adaptive dt", "Stability issues", "Better stability",
                "# Complex code\n" * 30,  # Long sketch = more complex
            ),
            MethodProposal(
                "P002", "parameter_tuning", "Simple param change",
                "Change one param", "Easy fix", "Quick improvement",
                "dt = dt * 0.5",  # Short sketch = simpler
            ),
        ]

        results = agent.analyze({"proposals": proposals})

        assert len(results) == 2
        assert results[0].ranking == 1
        assert results[1].ranking == 2
        # Rankings should be in order of overall_score (descending)
        assert results[0].overall_score >= results[1].overall_score


class TestReportAgent:
    """Tests for ReportAgent."""

    def test_analyze_empty(self):
        """Test report with minimal input."""
        agent = ReportAgent()
        report = agent.analyze({"cycle_id": 1})

        assert "Cycle 1" in report
        assert "Executive Summary" in report

    def test_analyze_full(self):
        """Test report with full context."""
        agent = ReportAgent()
        context = {
            "cycle_id": 1,
            "pareto_fronts": {
                "solver1": ParetoFront(
                    "solver1", "2024-01-01",
                    summary={"total_points": 10, "stable_points": 8,
                             "stability_rate": 80, "pareto_optimal_count": 3,
                             "min_error": 1e-4, "max_error": 1e-2},
                ),
            },
            "bottlenecks": [
                Bottleneck("B001", "stability", "high", "Test bottleneck",
                          ["solver1"], {}, ["Action 1"]),
            ],
            "proposals": [
                MethodProposal("P001", "parameter_tuning", "Test proposal",
                              "Desc", "Rationale", "Benefit", "Code", "approved"),
            ],
            "evaluations": [
                EvaluationResult("P001", {"accuracy": 4, "speed": 3}, 3.5, 1, "approve"),
            ],
        }

        report = agent.analyze(context)

        assert "solver1" in report
        assert "B001" in report
        assert "P001" in report
        assert "Per-Solver Pareto Analysis" in report
        assert "Bottlenecks Identified" in report
        assert "Multi-Agent Evaluation" in report


class TestCycleCoordinator:
    """Tests for CycleCoordinator."""

    def test_creation(self):
        """Test coordinator creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = os.path.join(tmpdir, "history.json")
            coordinator = CycleCoordinator(
                history_path=history_path,
                pareto_dir=os.path.join(tmpdir, "pareto"),
                reports_dir=os.path.join(tmpdir, "reports"),
            )
            assert coordinator.history.current_cycle == 0

    def test_start_new_cycle(self):
        """Test starting a new cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = os.path.join(tmpdir, "history.json")
            coordinator = CycleCoordinator(history_path=history_path)

            state = coordinator.start_new_cycle()

            assert state.cycle_id == 1
            assert state.phase == "pareto"
            assert coordinator.history.current_cycle == 1

    def test_get_current_cycle(self):
        """Test getting current cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = os.path.join(tmpdir, "history.json")
            coordinator = CycleCoordinator(history_path=history_path)

            # No cycle yet
            assert coordinator.get_current_cycle() is None

            # Start cycle
            coordinator.start_new_cycle()
            state = coordinator.get_current_cycle()
            assert state is not None
            assert state.cycle_id == 1

    def test_history_persistence(self):
        """Test that history is persisted correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = os.path.join(tmpdir, "history.json")

            # First coordinator
            coordinator1 = CycleCoordinator(history_path=history_path)
            coordinator1.start_new_cycle()

            # Second coordinator loads same history
            coordinator2 = CycleCoordinator(history_path=history_path)
            assert coordinator2.history.current_cycle == 1

    def test_generate_summary_report(self):
        """Test summary report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = os.path.join(tmpdir, "history.json")
            coordinator = CycleCoordinator(history_path=history_path)

            # Add some history manually
            coordinator.history.cycles.append(
                ImprovementCycleState(
                    cycle_id=1,
                    started_at="2024-01-01",
                    completed_at="2024-01-01",
                    phase="complete",
                    bottlenecks=["B001"],
                    proposals=["P001"],
                    approved_proposals=["P001"],
                )
            )
            coordinator.history.all_proposals["P001"] = {
                "proposal_id": "P001",
                "title": "Test Proposal",
                "proposal_type": "parameter_tuning",
                "expected_benefit": "Better performance",
            }
            coordinator.history.implemented_methods.append("P001")
            coordinator.history.all_bottlenecks["B001"] = {"category": "stability"}

            report = coordinator.generate_summary_report()

            assert "Summary Report" in report
            assert "Total cycles completed:** 1" in report
            assert "P001" in report


class TestCrossSolverBottlenecks:
    """Tests for cross-solver bottleneck analysis."""

    def _make_cross_solver(self, coverage_gaps=None, overall_rankings=None,
                           solver_win_counts=None, problems=None):
        """Helper to create CrossSolverAnalysis."""
        analysis = CrossSolverAnalysis(timestamp="2024-01-01")
        if problems:
            analysis.problems = problems
        if coverage_gaps:
            analysis.coverage_gaps = coverage_gaps
        if overall_rankings:
            analysis.overall_rankings = overall_rankings
        if solver_win_counts:
            analysis.solver_win_counts = solver_win_counts
        return analysis

    def test_no_stable_solver_bottleneck(self):
        """Test detection of problems with no stable solver."""
        agent = BottleneckAnalysisAgent()
        cross = self._make_cross_solver(
            coverage_gaps=[{
                "problem_key": "alpha=1.0_ic=step",
                "problem": {"alpha": 1.0, "ic_type": "step"},
                "issue": "no_stable_solver",
                "description": "No solver produces stable result for alpha=1.0_ic=step",
            }],
            problems={"alpha=1.0_ic=step": CrossSolverFront("alpha=1.0_ic=step", {"alpha": 1.0})},
        )
        bottlenecks = agent.analyze({"cross_solver": cross})
        assert any(b.category == "no_stable_solver" for b in bottlenecks)

    def test_cross_solver_accuracy_gap(self):
        """Test detection of high-error problems."""
        agent = BottleneckAnalysisAgent()
        cross = self._make_cross_solver(
            coverage_gaps=[{
                "problem_key": "alpha=0.5_ic=parabola",
                "problem": {"alpha": 0.5, "ic_type": "parabola"},
                "issue": "high_error",
                "best_error": 0.8,
                "best_solver": "implicit_fdm",
                "description": "Best L2 error is 0.8",
            }],
            problems={"alpha=0.5_ic=parabola": CrossSolverFront("alpha=0.5_ic=parabola", {"alpha": 0.5})},
        )
        bottlenecks = agent.analyze({"cross_solver": cross})
        assert any(b.category == "cross_solver_accuracy_gap" for b in bottlenecks)

    def test_solver_instability_bottleneck(self):
        """Test detection of solvers with low stability rate."""
        agent = BottleneckAnalysisAgent()
        cross = self._make_cross_solver(
            overall_rankings=[
                {"solver": "bad_solver", "stability_rate": 40,
                 "problems_stable": 2, "problems_total": 5},
                {"solver": "good_solver", "stability_rate": 100,
                 "problems_stable": 5, "problems_total": 5},
            ],
            problems={"p1": CrossSolverFront("p1", {})},
        )
        bottlenecks = agent.analyze({"cross_solver": cross})
        instability = [b for b in bottlenecks if b.category == "solver_instability"]
        assert len(instability) == 1
        assert "bad_solver" in instability[0].affected_solvers

    def test_solver_dominance_bottleneck(self):
        """Test detection of single-solver dominance."""
        agent = BottleneckAnalysisAgent()
        cross = self._make_cross_solver(
            solver_win_counts={
                "best_accuracy": {"solver_a": 9, "solver_b": 0, "solver_c": 1},
            },
            problems={f"p{i}": CrossSolverFront(f"p{i}", {}) for i in range(10)},
        )
        bottlenecks = agent.analyze({"cross_solver": cross})
        dominance = [b for b in bottlenecks if b.category == "solver_dominance"]
        assert len(dominance) == 1
        assert "solver_b" in dominance[0].affected_solvers


class TestCrossSolverProposals:
    """Tests for cross-solver proposal generation."""

    def test_no_stable_solver_proposals(self):
        """Test proposals for no-stable-solver bottleneck."""
        agent = ProposalGenerationAgent()
        bottleneck = Bottleneck(
            "B001", "no_stable_solver", "high",
            "No stable solver for alpha=1.0", [],
            {"problem": {"alpha": 1.0, "ic_type": "step"}},
        )
        proposals = agent.analyze({"bottlenecks": [bottleneck], "cycle_id": 1})
        assert len(proposals) >= 1
        assert any("adaptive" in p.title.lower() for p in proposals)

    def test_cross_solver_accuracy_proposals(self):
        """Test proposals for cross-solver accuracy gap."""
        agent = ProposalGenerationAgent()
        bottleneck = Bottleneck(
            "B002", "cross_solver_accuracy_gap", "medium",
            "Best L2 error is 0.8", ["implicit_fdm"],
            {"best_solver": "implicit_fdm", "best_error": 0.8},
        )
        proposals = agent.analyze({"bottlenecks": [bottleneck], "cycle_id": 1})
        assert len(proposals) >= 1
        assert any("resolution" in p.title.lower() for p in proposals)

    def test_solver_instability_proposals(self):
        """Test proposals for solver instability."""
        agent = ProposalGenerationAgent()
        bottleneck = Bottleneck(
            "B003", "solver_instability", "high",
            "spectral_cosine stable on 40%", ["spectral_cosine"],
            {"stability_rate": 40, "problems_stable": 2, "problems_total": 5},
        )
        proposals = agent.analyze({"bottlenecks": [bottleneck], "cycle_id": 1})
        assert len(proposals) >= 1
        assert any("spectral_cosine" in p.title for p in proposals)

    def test_solver_dominance_proposals(self):
        """Test proposals for solver dominance."""
        agent = ProposalGenerationAgent()
        bottleneck = Bottleneck(
            "B004", "solver_dominance", "low",
            "solver_a dominates", ["solver_b", "solver_c"],
            {"dominant_solver": "solver_a", "win_ratio": 0.9,
             "zero_win_solvers": ["solver_b", "solver_c"]},
        )
        proposals = agent.analyze({"bottlenecks": [bottleneck], "cycle_id": 1})
        assert len(proposals) >= 1


class TestCrossSolverReport:
    """Tests for cross-solver report generation."""

    def test_report_with_cross_solver(self):
        """Test report includes cross-solver analysis section."""
        agent = ReportAgent()

        # Create minimal cross-solver analysis
        front = CrossSolverFront(
            problem_key="alpha=0.5_ic=parabola",
            problem={"alpha": 0.5, "ic_type": "parabola"},
        )
        front.points = [
            ParetoPoint("solver_a", {"alpha": 0.5}, 0.01, 0.005, 0, True),
            ParetoPoint("solver_b", {"alpha": 0.5}, 0.02, 0.003, 1, True),
        ]
        front.compute()

        cross = CrossSolverAnalysis(timestamp="2024-01-01")
        cross.problems["alpha=0.5_ic=parabola"] = front
        cross.compute_summary()

        report = agent.analyze({
            "cycle_id": 1,
            "pareto_fronts": {},
            "cross_solver": cross,
            "bottlenecks": [],
            "proposals": [],
            "evaluations": [],
        })

        assert "Cross-Solver Analysis" in report
        assert "Overall Solver Rankings" in report
        assert "solver_a" in report
        assert "solver_b" in report

    def test_report_without_cross_solver(self):
        """Test report works without cross-solver data."""
        agent = ReportAgent()
        report = agent.analyze({
            "cycle_id": 1,
            "pareto_fronts": {},
            "bottlenecks": [],
            "proposals": [],
            "evaluations": [],
        })
        assert "Cross-Solver Analysis" not in report
        assert "Cycle 1" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
