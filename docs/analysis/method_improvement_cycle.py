"""Multi-Agent Method Improvement Cycle Orchestrator.

This module provides the main entry point for running iterative
method improvement cycles using multi-agent analysis.

Workflow:
    1. Pareto Analysis - Run parameter sweeps, compute Pareto fronts
    2. Bottleneck Analysis - Identify performance gaps and issues
    3. Proposal Generation - Generate improvement proposals
    4. Multi-Agent Evaluation - Evaluate proposals from multiple perspectives
    5. Human Review (optional) - Interactive approval
    6. Implementation & Verification - Apply and test improvements
    7. Report & Archive - Generate reports, save state for restart

Usage:
    python docs/analysis/method_improvement_cycle.py --cycles 3
    python docs/analysis/method_improvement_cycle.py --resume
    python docs/analysis/method_improvement_cycle.py --interactive --cycles 3
    python docs/analysis/method_improvement_cycle.py --report
    python docs/analysis/method_improvement_cycle.py --fresh --cycles 3
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from docs.analysis.pareto_analyzer import (
    ParetoAnalysisAgent,
    ParetoFront,
    CrossSolverAnalysis,
    load_all_pareto_fronts,
)
from docs.analysis.improvement_agents import (
    BottleneckAnalysisAgent,
    ProposalGenerationAgent,
    EvaluationAgent,
    ReportAgent,
    MethodProposal,
    Bottleneck,
    EvaluationResult,
)

# Default paths
DEFAULT_HISTORY_PATH = os.path.join(PROJECT_ROOT, "data", "improvement_history.json")
DEFAULT_PROPOSALS_PATH = os.path.join(PROJECT_ROOT, "data", "method_proposals.json")
DEFAULT_PARETO_DIR = os.path.join(PROJECT_ROOT, "data", "pareto_fronts")
DEFAULT_REPORTS_DIR = os.path.join(PROJECT_ROOT, "data", "cycle_reports")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ImprovementCycleState:
    """State of a single improvement cycle.

    Attributes:
        cycle_id: Cycle number (1-indexed)
        started_at: ISO timestamp when cycle started
        completed_at: ISO timestamp when cycle completed (or None)
        phase: Current phase (pareto, bottleneck, proposal, evaluation, review, implementation, report, complete)
        pareto_fronts: Dict mapping solver name to file path
        bottlenecks: List of bottleneck IDs
        proposals: List of proposal IDs
        approved_proposals: List of approved proposal IDs
        notes: Free-form notes
    """
    cycle_id: int
    started_at: str
    completed_at: Optional[str] = None
    phase: str = "pareto"
    pareto_fronts: Dict[str, str] = field(default_factory=dict)
    bottlenecks: List[str] = field(default_factory=list)
    proposals: List[str] = field(default_factory=list)
    approved_proposals: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImprovementCycleState":
        return cls(**d)


@dataclass
class ImprovementHistory:
    """History of all improvement cycles.

    Attributes:
        current_cycle: Current cycle number (0 if not started)
        cycles: List of cycle states
        all_proposals: Dict mapping proposal ID to proposal data
        all_bottlenecks: Dict mapping bottleneck ID to bottleneck data
        implemented_methods: List of implemented proposal IDs
    """
    current_cycle: int = 0
    cycles: List[ImprovementCycleState] = field(default_factory=list)
    all_proposals: Dict[str, Dict] = field(default_factory=dict)
    all_bottlenecks: Dict[str, Dict] = field(default_factory=dict)
    implemented_methods: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_cycle": self.current_cycle,
            "cycles": [c.to_dict() for c in self.cycles],
            "all_proposals": self.all_proposals,
            "all_bottlenecks": self.all_bottlenecks,
            "implemented_methods": self.implemented_methods,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImprovementHistory":
        return cls(
            current_cycle=d.get("current_cycle", 0),
            cycles=[ImprovementCycleState.from_dict(c) for c in d.get("cycles", [])],
            all_proposals=d.get("all_proposals", {}),
            all_bottlenecks=d.get("all_bottlenecks", {}),
            implemented_methods=d.get("implemented_methods", []),
        )

    def save(self, path: str):
        """Save history to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ImprovementHistory":
        """Load history from JSON file."""
        if not os.path.isfile(path):
            return cls()
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Cycle Coordinator
# =============================================================================

class CycleCoordinator:
    """Orchestrates the method improvement cycle.

    Manages the workflow through all phases and handles
    state persistence for restarts.
    """

    PHASES = [
        "pareto",
        "bottleneck",
        "proposal",
        "evaluation",
        "review",
        "implementation",
        "report",
        "complete",
    ]

    def __init__(
        self,
        history_path: str = None,
        pareto_dir: str = None,
        reports_dir: str = None,
        solvers: List[str] = None,
        interactive: bool = False,
        auto_approve: bool = True,
    ):
        """Initialize the cycle coordinator.

        Args:
            history_path: Path to history JSON file
            pareto_dir: Directory for Pareto front files
            reports_dir: Directory for cycle reports
            solvers: List of solver names to analyze (None = all)
            interactive: Whether to prompt for human review
            auto_approve: Whether to auto-approve top proposals
        """
        self.history_path = history_path or DEFAULT_HISTORY_PATH
        self.pareto_dir = pareto_dir or DEFAULT_PARETO_DIR
        self.reports_dir = reports_dir or DEFAULT_REPORTS_DIR
        self.solvers_filter = solvers
        self.interactive = interactive
        self.auto_approve = auto_approve

        # Load or create history
        self.history = ImprovementHistory.load(self.history_path)

        # Initialize agents
        self.pareto_agent = ParetoAnalysisAgent(output_dir=self.pareto_dir)
        self.bottleneck_agent = BottleneckAnalysisAgent()
        self.proposal_agent = ProposalGenerationAgent()
        self.evaluation_agent = EvaluationAgent()
        self.report_agent = ReportAgent()

        # Runtime state
        self.current_pareto_fronts: Dict[str, ParetoFront] = {}
        self.current_cross_solver: Optional[CrossSolverAnalysis] = None
        self.current_bottlenecks: List[Bottleneck] = []
        self.current_proposals: List[MethodProposal] = []
        self.current_evaluations: List[EvaluationResult] = []

    def _get_solvers(self):
        """Get list of solver instances to analyze."""
        from app.run_benchmark import SOLVERS

        if self.solvers_filter:
            return [s for s in SOLVERS if s.name in self.solvers_filter]
        return SOLVERS

    def _save_history(self):
        """Save current history to file."""
        self.history.save(self.history_path)

    def start_new_cycle(self) -> ImprovementCycleState:
        """Start a new improvement cycle.

        Returns:
            New cycle state
        """
        cycle_id = self.history.current_cycle + 1
        state = ImprovementCycleState(
            cycle_id=cycle_id,
            started_at=datetime.now().isoformat(),
            phase="pareto",
        )
        self.history.cycles.append(state)
        self.history.current_cycle = cycle_id
        self._save_history()

        print(f"\n{'='*60}")
        print(f"STARTING IMPROVEMENT CYCLE {cycle_id}")
        print(f"{'='*60}")

        return state

    def get_current_cycle(self) -> Optional[ImprovementCycleState]:
        """Get the current cycle state."""
        if not self.history.cycles:
            return None
        return self.history.cycles[-1]

    def run_phase(self, phase: str, state: ImprovementCycleState):
        """Run a specific phase of the cycle.

        Args:
            phase: Phase name
            state: Current cycle state
        """
        print(f"\n[Phase: {phase.upper()}]")
        print("-" * 40)

        if phase == "pareto":
            self._run_pareto_phase(state)
        elif phase == "bottleneck":
            self._run_bottleneck_phase(state)
        elif phase == "proposal":
            self._run_proposal_phase(state)
        elif phase == "evaluation":
            self._run_evaluation_phase(state)
        elif phase == "review":
            self._run_review_phase(state)
        elif phase == "implementation":
            self._run_implementation_phase(state)
        elif phase == "report":
            self._run_report_phase(state)
        elif phase == "complete":
            self._complete_cycle(state)

    def _run_pareto_phase(self, state: ImprovementCycleState):
        """Run Pareto analysis phase (per-solver + cross-solver)."""
        solvers = self._get_solvers()
        print(f"Analyzing {len(solvers)} solvers...")

        # Per-solver analysis
        self.current_pareto_fronts = self.pareto_agent.run_quick_analysis(
            solvers, verbose=True
        )

        # Record file paths
        for name, front in self.current_pareto_fronts.items():
            filename = f"{name}_{front.timestamp.replace(':', '-')}.json"
            state.pareto_fronts[name] = os.path.join(self.pareto_dir, filename)

        # Cross-solver analysis
        print("\nRunning cross-solver analysis...")
        self.current_cross_solver = self.pareto_agent.run_quick_cross_solver(
            solvers, verbose=True
        )

        # Save cross-solver results
        cross_path = os.path.join(self.pareto_dir, "cross_solver_analysis.json")
        self.current_cross_solver.save(cross_path)

        # Advance phase
        state.phase = "bottleneck"
        self._save_history()

    def _run_bottleneck_phase(self, state: ImprovementCycleState):
        """Run bottleneck analysis phase."""
        # Load Pareto fronts if needed
        if not self.current_pareto_fronts:
            self.current_pareto_fronts = load_all_pareto_fronts(self.pareto_dir)

        # Load cross-solver results if needed
        if self.current_cross_solver is None:
            cross_path = os.path.join(self.pareto_dir, "cross_solver_analysis.json")
            if os.path.isfile(cross_path):
                self.current_cross_solver = CrossSolverAnalysis.load(cross_path)

        # Run analysis
        self.current_bottlenecks = self.bottleneck_agent.analyze({
            "pareto_fronts": self.current_pareto_fronts,
            "cross_solver": self.current_cross_solver,
        })

        # Record bottlenecks
        for b in self.current_bottlenecks:
            state.bottlenecks.append(b.bottleneck_id)
            self.history.all_bottlenecks[b.bottleneck_id] = b.to_dict()

        print(f"Found {len(self.current_bottlenecks)} bottlenecks:")
        for b in self.current_bottlenecks:
            print(f"  - [{b.severity}] {b.description}")

        # Advance phase
        state.phase = "proposal"
        self._save_history()

    def _run_proposal_phase(self, state: ImprovementCycleState):
        """Run proposal generation phase."""
        # Reconstruct bottlenecks if needed
        if not self.current_bottlenecks:
            self.current_bottlenecks = [
                Bottleneck(**self.history.all_bottlenecks[bid])
                for bid in state.bottlenecks
                if bid in self.history.all_bottlenecks
            ]

        # Generate proposals
        self.current_proposals = self.proposal_agent.analyze({
            "bottlenecks": self.current_bottlenecks,
            "cycle_id": state.cycle_id,
        })

        # Record proposals
        for p in self.current_proposals:
            state.proposals.append(p.proposal_id)
            self.history.all_proposals[p.proposal_id] = p.to_dict()

        print(f"Generated {len(self.current_proposals)} proposals:")
        for p in self.current_proposals:
            print(f"  - {p.proposal_id}: {p.title}")

        # Advance phase
        state.phase = "evaluation"
        self._save_history()

    def _run_evaluation_phase(self, state: ImprovementCycleState):
        """Run multi-agent evaluation phase."""
        # Reconstruct proposals if needed
        if not self.current_proposals:
            self.current_proposals = [
                MethodProposal.from_dict(self.history.all_proposals[pid])
                for pid in state.proposals
                if pid in self.history.all_proposals
            ]

        # Run evaluation
        self.current_evaluations = self.evaluation_agent.analyze({
            "proposals": self.current_proposals,
        })

        print(f"Evaluation results:")
        print(f"{'Rank':<6}{'ID':<8}{'Score':<8}{'Recommendation':<15}")
        print("-" * 40)
        for e in self.current_evaluations:
            print(f"{e.ranking:<6}{e.proposal_id:<8}{e.overall_score:<8.2f}{e.recommendation:<15}")

        # Advance phase
        state.phase = "review"
        self._save_history()

    def _run_review_phase(self, state: ImprovementCycleState):
        """Run review phase (interactive or auto)."""
        if self.interactive:
            self._interactive_review(state)
        elif self.auto_approve:
            self._auto_approve(state)
        else:
            print("No review mode selected. Skipping approvals.")

        # Advance phase
        state.phase = "implementation"
        self._save_history()

    def _interactive_review(self, state: ImprovementCycleState):
        """Interactive proposal review."""
        print("\nProposal Review (Interactive Mode)")
        print("=" * 40)

        for e in self.current_evaluations:
            proposal = next(
                (p for p in self.current_proposals if p.proposal_id == e.proposal_id),
                None
            )
            if not proposal:
                continue

            print(f"\n{proposal.proposal_id}: {proposal.title}")
            print(f"  Type: {proposal.proposal_type}")
            print(f"  Score: {e.overall_score:.2f}")
            print(f"  Rationale: {proposal.rationale}")
            print(f"  Expected: {proposal.expected_benefit}")

            try:
                response = input("  Approve? [Y/n/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                response = "q"

            if response == "q":
                print("Review cancelled.")
                break
            elif response in ("", "y", "yes"):
                state.approved_proposals.append(proposal.proposal_id)
                proposal.status = "approved"
                self.history.all_proposals[proposal.proposal_id] = proposal.to_dict()
                print("  -> Approved")
            else:
                proposal.status = "rejected"
                self.history.all_proposals[proposal.proposal_id] = proposal.to_dict()
                print("  -> Rejected")

    def _auto_approve(self, state: ImprovementCycleState):
        """Auto-approve top proposals."""
        print("\nAuto-approving proposals...")

        for e in self.current_evaluations:
            proposal = next(
                (p for p in self.current_proposals if p.proposal_id == e.proposal_id),
                None
            )
            if not proposal:
                continue

            if e.recommendation == "approve":
                state.approved_proposals.append(proposal.proposal_id)
                proposal.status = "approved"
                print(f"  Approved: {proposal.proposal_id}")
            else:
                proposal.status = "rejected"
                print(f"  Rejected: {proposal.proposal_id}")

            self.history.all_proposals[proposal.proposal_id] = proposal.to_dict()

    def _run_implementation_phase(self, state: ImprovementCycleState):
        """Run implementation phase.

        For parameter_tuning proposals, this could auto-generate config changes.
        For algorithm_tweak and new_solver, just log the implementation sketch.
        """
        print(f"\nApproved proposals for implementation: {len(state.approved_proposals)}")

        for pid in state.approved_proposals:
            if pid not in self.history.all_proposals:
                continue

            proposal = MethodProposal.from_dict(self.history.all_proposals[pid])

            print(f"\n{pid}: {proposal.title}")
            print(f"  Type: {proposal.proposal_type}")

            if proposal.proposal_type == "parameter_tuning":
                print("  -> Auto-implementable (parameter changes)")
                print("  Implementation sketch:")
                for line in proposal.implementation_sketch.strip().split("\n")[:5]:
                    print(f"    {line}")
            else:
                print("  -> Requires manual implementation")
                print("  Implementation sketch preview:")
                for line in proposal.implementation_sketch.strip().split("\n")[:5]:
                    print(f"    {line}")

            # Mark as implemented (for tracking)
            proposal.status = "implemented"
            self.history.all_proposals[pid] = proposal.to_dict()
            self.history.implemented_methods.append(pid)

        # Advance phase
        state.phase = "report"
        self._save_history()

    def _run_report_phase(self, state: ImprovementCycleState):
        """Generate and save cycle report."""
        # Load Pareto fronts if needed
        if not self.current_pareto_fronts:
            self.current_pareto_fronts = load_all_pareto_fronts(self.pareto_dir)

        # Reconstruct data
        bottlenecks = [
            Bottleneck(**self.history.all_bottlenecks[bid])
            for bid in state.bottlenecks
            if bid in self.history.all_bottlenecks
        ]
        proposals = [
            MethodProposal.from_dict(self.history.all_proposals[pid])
            for pid in state.proposals
            if pid in self.history.all_proposals
        ]

        # Load cross-solver results if needed
        if self.current_cross_solver is None:
            cross_path = os.path.join(self.pareto_dir, "cross_solver_analysis.json")
            if os.path.isfile(cross_path):
                self.current_cross_solver = CrossSolverAnalysis.load(cross_path)

        # Generate report
        report = self.report_agent.analyze({
            "pareto_fronts": self.current_pareto_fronts,
            "cross_solver": self.current_cross_solver,
            "bottlenecks": bottlenecks,
            "proposals": proposals,
            "evaluations": self.current_evaluations,
            "cycle_id": state.cycle_id,
        })

        # Save report
        os.makedirs(self.reports_dir, exist_ok=True)
        report_path = os.path.join(
            self.reports_dir,
            f"cycle_{state.cycle_id:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

        # Advance phase
        state.phase = "complete"
        self._save_history()

    def _complete_cycle(self, state: ImprovementCycleState):
        """Complete the cycle."""
        state.completed_at = datetime.now().isoformat()
        self._save_history()

        print(f"\n{'='*60}")
        print(f"CYCLE {state.cycle_id} COMPLETE")
        print(f"{'='*60}")
        print(f"  Bottlenecks found: {len(state.bottlenecks)}")
        print(f"  Proposals generated: {len(state.proposals)}")
        print(f"  Proposals approved: {len(state.approved_proposals)}")

    def run_cycle(self) -> ImprovementCycleState:
        """Run a complete improvement cycle.

        Returns:
            Completed cycle state
        """
        state = self.start_new_cycle()

        for phase in self.PHASES:
            self.run_phase(phase, state)
            if phase == "complete":
                break

        return state

    def resume_cycle(self) -> Optional[ImprovementCycleState]:
        """Resume an incomplete cycle.

        Returns:
            Resumed cycle state, or None if no incomplete cycle
        """
        state = self.get_current_cycle()
        if not state:
            print("No cycle to resume. Starting new cycle.")
            return self.run_cycle()

        if state.phase == "complete":
            print(f"Cycle {state.cycle_id} already complete. Starting new cycle.")
            return self.run_cycle()

        print(f"\nResuming cycle {state.cycle_id} from phase: {state.phase}")

        # Find current phase index and continue from there
        phase_idx = self.PHASES.index(state.phase)
        for phase in self.PHASES[phase_idx:]:
            self.run_phase(phase, state)
            if phase == "complete":
                break

        return state

    def run_cycles(self, n_cycles: int):
        """Run multiple improvement cycles.

        Args:
            n_cycles: Number of cycles to run
        """
        print(f"\n{'='*60}")
        print(f"RUNNING {n_cycles} IMPROVEMENT CYCLES")
        print(f"{'='*60}")

        for i in range(n_cycles):
            print(f"\n{'~'*60}")
            print(f"Cycle {i+1} of {n_cycles}")
            print(f"{'~'*60}")
            self.run_cycle()

        print(f"\n{'='*60}")
        print(f"COMPLETED {n_cycles} CYCLES")
        print(f"{'='*60}")
        print(f"Total proposals: {len(self.history.all_proposals)}")
        print(f"Implemented: {len(self.history.implemented_methods)}")

    def generate_summary_report(self) -> str:
        """Generate a summary report of all cycles.

        Returns:
            Markdown report string
        """
        lines = []
        lines.append("# Method Improvement Summary Report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        lines.append("## Overview\n")
        lines.append(f"- **Total cycles completed:** {len([c for c in self.history.cycles if c.phase == 'complete'])}")
        lines.append(f"- **Total proposals generated:** {len(self.history.all_proposals)}")
        lines.append(f"- **Proposals implemented:** {len(self.history.implemented_methods)}")
        lines.append(f"- **Bottlenecks identified:** {len(self.history.all_bottlenecks)}")
        lines.append("")

        lines.append("## Cycle History\n")
        lines.append("| Cycle | Started | Completed | Bottlenecks | Proposals | Approved |")
        lines.append("|-------|---------|-----------|-------------|-----------|----------|")
        for c in self.history.cycles:
            started = c.started_at[:10] if c.started_at else "N/A"
            completed = c.completed_at[:10] if c.completed_at else "In progress"
            lines.append(
                f"| {c.cycle_id} | {started} | {completed} | "
                f"{len(c.bottlenecks)} | {len(c.proposals)} | {len(c.approved_proposals)} |"
            )
        lines.append("")

        lines.append("## Implemented Improvements\n")
        if self.history.implemented_methods:
            for pid in self.history.implemented_methods:
                if pid in self.history.all_proposals:
                    p = self.history.all_proposals[pid]
                    lines.append(f"- **{pid}**: {p['title']}")
                    lines.append(f"  - Type: {p['proposal_type']}")
                    lines.append(f"  - Expected: {p['expected_benefit']}")
        else:
            lines.append("*No implementations yet.*")
        lines.append("")

        lines.append("## Bottleneck Summary\n")
        categories = {}
        for bid, b in self.history.all_bottlenecks.items():
            cat = b.get("category", "unknown")
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        for cat, count in categories.items():
            lines.append(f"- **{cat}:** {count} identified")
        lines.append("")

        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Method Improvement Cycle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run 3 improvement cycles
    python docs/analysis/method_improvement_cycle.py --cycles 3

    # Resume from previous state
    python docs/analysis/method_improvement_cycle.py --resume

    # Run with interactive review
    python docs/analysis/method_improvement_cycle.py --interactive --cycles 2

    # Generate summary report only
    python docs/analysis/method_improvement_cycle.py --report

    # Fresh start (clear history)
    python docs/analysis/method_improvement_cycle.py --fresh --cycles 3

    # Analyze specific solvers
    python docs/analysis/method_improvement_cycle.py --solvers implicit_fdm,spectral_cosine
        """
    )

    parser.add_argument(
        "--cycles", "-c", type=int, default=0,
        help="Number of improvement cycles to run"
    )
    parser.add_argument(
        "--resume", "-r", action="store_true",
        help="Resume from previous state"
    )
    parser.add_argument(
        "--phase", "-p", type=str, default=None,
        choices=CycleCoordinator.PHASES,
        help="Run only a specific phase"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Interactive mode for proposal review"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate summary report only"
    )
    parser.add_argument(
        "--fresh", "-f", action="store_true",
        help="Start fresh (clear history)"
    )
    parser.add_argument(
        "--solvers", "-s", type=str, default=None,
        help="Comma-separated list of solvers to analyze"
    )
    parser.add_argument(
        "--auto", "-a", action="store_true",
        help="Full auto mode: run cycles and generate report"
    )
    parser.add_argument(
        "--no-approve", action="store_true",
        help="Disable auto-approval of proposals"
    )
    parser.add_argument(
        "--history-path", type=str, default=None,
        help="Path to history JSON file"
    )

    args = parser.parse_args()

    # Parse solvers
    solvers = None
    if args.solvers:
        solvers = [s.strip() for s in args.solvers.split(",")]

    # Fresh start
    if args.fresh:
        history_path = args.history_path or DEFAULT_HISTORY_PATH
        if os.path.isfile(history_path):
            os.remove(history_path)
            print(f"Cleared history: {history_path}")

    # Create coordinator
    coordinator = CycleCoordinator(
        history_path=args.history_path,
        solvers=solvers,
        interactive=args.interactive,
        auto_approve=not args.no_approve,
    )

    # Handle modes
    if args.report:
        # Generate report only
        report = coordinator.generate_summary_report()
        report_path = os.path.join(DEFAULT_REPORTS_DIR, "summary_report.md")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Summary report saved to: {report_path}")
        print("\n" + report)

    elif args.resume:
        # Resume previous cycle
        coordinator.resume_cycle()

    elif args.phase:
        # Run specific phase
        state = coordinator.get_current_cycle()
        if not state:
            state = coordinator.start_new_cycle()
        coordinator.run_phase(args.phase, state)

    elif args.cycles > 0:
        # Run N cycles
        coordinator.run_cycles(args.cycles)

        # Generate summary report
        report = coordinator.generate_summary_report()
        report_path = os.path.join(DEFAULT_REPORTS_DIR, "summary_report.md")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nSummary report saved to: {report_path}")

    elif args.auto:
        # Auto mode: run 3 cycles and report
        coordinator.run_cycles(3)
        report = coordinator.generate_summary_report()
        print("\n" + report)

    else:
        # Default: run 1 cycle
        print("No action specified. Running 1 cycle...")
        coordinator.run_cycle()


if __name__ == "__main__":
    main()
