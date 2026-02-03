"""Iterative Hypothesis-Driven Experiment Framework.

This framework allows:
1. Define experiments with custom parameters
2. Run experiments and store results in database
3. Analyze results automatically
4. Test hypotheses against new data
5. Suggest next experiments based on findings
6. Track hypothesis memos and verification history
7. Generate comprehensive final reports

Usage:
    python docs/analysis/experiment_framework.py --interactive
    python docs/analysis/experiment_framework.py --run-experiment stability_map
    python docs/analysis/experiment_framework.py --analyze
    python docs/analysis/experiment_framework.py --cycles 3  # Run 3 verification cycles
    python docs/analysis/experiment_framework.py --report    # Generate final report
"""

import os
import sys
import csv
import json
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# Default paths (relative to project root)
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "experiments.csv")
DEFAULT_MEMO_PATH = os.path.join(PROJECT_ROOT, "data", "hypotheses_memo.json")
DEFAULT_REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "experiment_report.md")

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from solvers.fdm.implicit import ImplicitFDM
from solvers.spectral.cosine import CosineSpectral
from metrics.accuracy import compute_errors


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str
    alpha_list: List[float]
    nr_list: List[int]
    dt_list: List[float]
    t_end_list: List[float]
    ic_type: str = "parabola"  # parabola, gaussian, cosine, scaled
    ic_scale: float = 1.0
    solvers: List[str] = field(default_factory=lambda: ["implicit_fdm", "spectral_cosine"])
    lambda_cost: float = 0.1


@dataclass
class ExperimentResult:
    """Result from a single run."""
    experiment_name: str
    timestamp: str
    alpha: float
    nr: int
    dt: float
    t_end: float
    ic_type: str
    ic_scale: float
    solver: str
    l2_error: float
    linf_error: float
    wall_time: float
    max_T: float
    min_T: float
    is_stable: bool
    is_nan: bool
    score: float  # L2 + λ * time


@dataclass
class HypothesisMemo:
    """A hypothesis with its verification history and notes."""
    hypothesis_id: str
    statement: str
    created_at: str
    status: str  # "untested", "confirmed", "rejected", "inconclusive"
    confidence: float  # 0.0 to 1.0
    notes: List[str] = field(default_factory=list)
    verification_history: List[Dict[str, Any]] = field(default_factory=list)
    related_experiments: List[str] = field(default_factory=list)


# =============================================================================
# Hypothesis Tracker
# =============================================================================

class HypothesisTracker:
    """Manages hypotheses, memos, and verification history."""

    def __init__(self, memo_path: str = None):
        if memo_path is None:
            memo_path = DEFAULT_MEMO_PATH
        self.memo_path = memo_path
        self.hypotheses: Dict[str, HypothesisMemo] = {}
        self._load_memos()

    def _load_memos(self):
        """Load hypotheses from file."""
        if os.path.isfile(self.memo_path):
            with open(self.memo_path) as f:
                data = json.load(f)
                for hid, hdata in data.items():
                    self.hypotheses[hid] = HypothesisMemo(**hdata)

    def _save_memos(self):
        """Save hypotheses to file."""
        os.makedirs(os.path.dirname(self.memo_path), exist_ok=True)
        data = {hid: asdict(h) for hid, h in self.hypotheses.items()}
        with open(self.memo_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_hypothesis(self, hypothesis_id: str, statement: str) -> HypothesisMemo:
        """Add a new hypothesis."""
        memo = HypothesisMemo(
            hypothesis_id=hypothesis_id,
            statement=statement,
            created_at=datetime.now().isoformat(),
            status="untested",
            confidence=0.0,
        )
        self.hypotheses[hypothesis_id] = memo
        self._save_memos()
        return memo

    def add_note(self, hypothesis_id: str, note: str):
        """Add a note to a hypothesis."""
        if hypothesis_id in self.hypotheses:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.hypotheses[hypothesis_id].notes.append(f"[{timestamp}] {note}")
            self._save_memos()

    def record_verification(self, hypothesis_id: str, result: Dict[str, Any],
                           experiment_name: Optional[str] = None):
        """Record a verification attempt."""
        if hypothesis_id not in self.hypotheses:
            return

        h = self.hypotheses[hypothesis_id]
        h.verification_history.append({
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "experiment": experiment_name,
        })

        # Update status based on result
        if result.get("confirmed"):
            h.status = "confirmed"
            h.confidence = min(1.0, h.confidence + 0.2)
        elif result.get("confirmed") is False:
            h.status = "rejected"
            h.confidence = max(0.0, h.confidence - 0.1)
        else:
            h.status = "inconclusive"

        if experiment_name and experiment_name not in h.related_experiments:
            h.related_experiments.append(experiment_name)

        self._save_memos()

    def get_summary(self) -> Dict[str, List[str]]:
        """Get summary of hypotheses by status."""
        summary = {"confirmed": [], "rejected": [], "untested": [], "inconclusive": []}
        for hid, h in self.hypotheses.items():
            summary[h.status].append(f"{hid}: {h.statement}")
        return summary

    def list_hypotheses(self):
        """Print all hypotheses."""
        print("\n" + "=" * 60)
        print("HYPOTHESIS TRACKER")
        print("=" * 60)

        if not self.hypotheses:
            print("\nNo hypotheses registered yet.")
            print("Use 'hypo add <ID> <statement>' to add one.")
            return

        for hid, h in sorted(self.hypotheses.items()):
            status_icon = {"confirmed": "[O]", "rejected": "[X]",
                          "untested": "[ ]", "inconclusive": "[?]"}.get(h.status, "[?]")
            print(f"\n{status_icon} {hid}: {h.statement}")
            print(f"    Status: {h.status}, Confidence: {h.confidence:.0%}")
            print(f"    Verifications: {len(h.verification_history)}")
            if h.notes:
                print(f"    Latest note: {h.notes[-1][:50]}...")

    def generate_markdown_report(self, analysis: Dict, output_path: str = None):
        if output_path is None:
            output_path = DEFAULT_REPORT_PATH
        """Generate a comprehensive markdown report."""
        lines = []
        lines.append("# Experiment Framework Report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Executive Summary
        lines.append("## Executive Summary\n")
        lines.append(f"- **Total experiments:** {analysis.get('total_runs', 0)} runs")
        lines.append(f"- **Hypotheses tracked:** {len(self.hypotheses)}")
        summary = self.get_summary()
        lines.append(f"- **Confirmed:** {len(summary['confirmed'])}")
        lines.append(f"- **Rejected:** {len(summary['rejected'])}")
        lines.append(f"- **Inconclusive:** {len(summary['inconclusive'])}")

        # Solver Performance
        lines.append("\n## Solver Performance\n")
        lines.append("| Solver | Runs | Stable % | Avg L2 Error | Avg Time |")
        lines.append("|--------|------|----------|--------------|----------|")
        for solver, stats in analysis.get("solvers", {}).items():
            lines.append(f"| {solver} | {stats['total']} | {stats['stable_pct']:.1f}% | "
                        f"{stats['avg_l2']:.6f} | {stats['avg_time']*1000:.2f}ms |")

        # Stability Analysis
        lines.append("\n## Stability by Alpha (Spectral Solver)\n")
        lines.append("| Alpha | Stable | Total | Rate |")
        lines.append("|-------|--------|-------|------|")
        spec_stab = analysis.get("stability_analysis", {}).get("spectral_cosine", {})
        for alpha, stats in sorted(spec_stab.items()):
            lines.append(f"| {alpha} | {stats['stable']} | {stats['total']} | {stats['pct']:.0f}% |")

        # Winner Analysis
        lines.append("\n## Winner Distribution\n")
        winners = analysis.get("winner_analysis", {})
        total = sum(winners.values())
        for solver, count in winners.items():
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"- **{solver}:** {count} wins ({pct:.1f}%)")

        # Hypotheses
        lines.append("\n## Hypothesis Verification Results\n")
        for hid, h in sorted(self.hypotheses.items()):
            status_emoji = {"confirmed": "**CONFIRMED**", "rejected": "~~REJECTED~~",
                          "untested": "_UNTESTED_", "inconclusive": "_INCONCLUSIVE_"}.get(h.status)
            lines.append(f"### {hid}: {h.statement}\n")
            lines.append(f"- Status: {status_emoji}")
            lines.append(f"- Confidence: {h.confidence:.0%}")
            lines.append(f"- Verifications: {len(h.verification_history)}")

            if h.notes:
                lines.append("\n**Notes:**")
                for note in h.notes[-3:]:  # Last 3 notes
                    lines.append(f"- {note}")

            if h.verification_history:
                lines.append("\n**Verification History:**")
                for v in h.verification_history[-3:]:  # Last 3 verifications
                    ts = v["timestamp"][:10]
                    result = "Confirmed" if v["result"].get("confirmed") else "Not confirmed"
                    lines.append(f"- [{ts}] {result}")

            lines.append("")

        # Conclusions
        lines.append("\n## Conclusions\n")
        if summary["confirmed"]:
            lines.append("### Confirmed Hypotheses\n")
            for h in summary["confirmed"]:
                lines.append(f"- {h}")

        if summary["rejected"]:
            lines.append("\n### Rejected Hypotheses\n")
            for h in summary["rejected"]:
                lines.append(f"- {h}")

        # Next Steps
        lines.append("\n## Recommended Next Steps\n")
        if summary["untested"]:
            lines.append("1. Test remaining untested hypotheses:")
            for h in summary["untested"][:3]:
                lines.append(f"   - {h}")

        if len(summary["inconclusive"]) > 0:
            lines.append("2. Gather more data for inconclusive hypotheses")

        lines.append("3. Consider new hypotheses based on findings")
        lines.append("")

        # Write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        return output_path


# Default hypotheses to track
DEFAULT_HYPOTHESES = {
    "H1": "Smaller dt improves spectral solver stability",
    "H3": "FDM is unconditionally stable for any dt",
    "H4": "Different initial conditions lead to different optimal solvers",
    "H5": "In linear regime (|dT/dr| < 0.5), both solvers perform equally well",
    "H6": "Cost function parameter lambda > 5 favors spectral solver",
    "H7": "Spectral solver fails with NaN for alpha >= 0.2",
}


# =============================================================================
# Initial Condition Factory
# =============================================================================

def make_initial(r: np.ndarray, ic_type: str, scale: float = 1.0) -> np.ndarray:
    """Create initial condition based on type."""
    if ic_type == "parabola":
        return scale * (1 - r**2)
    elif ic_type == "gaussian":
        return scale * np.exp(-10 * r**2)
    elif ic_type == "cosine":
        return scale * np.cos(np.pi * r / 2)
    elif ic_type == "step":
        return scale * np.where(r < 0.3, 1.0, 0.0)
    elif ic_type == "sine":
        return scale * np.sin(np.pi * (1 - r))
    else:
        raise ValueError(f"Unknown IC type: {ic_type}")


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """Runs experiments and stores results."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self.db_path = db_path
        self.results: List[ExperimentResult] = []
        self.solvers = {
            "implicit_fdm": ImplicitFDM(),
            "spectral_cosine": CosineSpectral(),
        }

    def run_experiment(self, config: ExperimentConfig, verbose: bool = True) -> List[ExperimentResult]:
        """Run a full experiment with given configuration."""
        results = []
        timestamp = datetime.now().isoformat()

        total = (len(config.alpha_list) * len(config.nr_list) *
                 len(config.dt_list) * len(config.t_end_list) * len(config.solvers))
        count = 0

        if verbose:
            print(f"\n{'='*60}")
            print(f"Experiment: {config.name}")
            print(f"Description: {config.description}")
            print(f"Total runs: {total}")
            print(f"{'='*60}")

        for alpha in config.alpha_list:
            for nr in config.nr_list:
                for dt in config.dt_list:
                    for t_end in config.t_end_list:
                        r = np.linspace(0, 1, nr)
                        T0 = make_initial(r, config.ic_type, config.ic_scale)

                        # Compute reference
                        T_ref = self._compute_reference(T0, r, dt, t_end, alpha)

                        for solver_name in config.solvers:
                            count += 1
                            result = self._run_single(
                                config.name, timestamp, alpha, nr, dt, t_end,
                                config.ic_type, config.ic_scale, solver_name,
                                T0, r, T_ref, config.lambda_cost
                            )
                            results.append(result)

                            if verbose and count % 20 == 0:
                                print(f"  Progress: {count}/{total} ({100*count/total:.0f}%)")

        self.results.extend(results)
        self._save_results(results)

        if verbose:
            print(f"\nCompleted {len(results)} runs. Saved to {self.db_path}")

        return results

    def _compute_reference(self, T0, r, dt, t_end, alpha):
        """Compute reference solution with 4x refinement."""
        nr_fine = 4 * len(r) - 3
        r_fine = np.linspace(0, 1, nr_fine)
        T0_fine = np.interp(r_fine, r, T0)
        solver = ImplicitFDM()
        T_hist = solver.solve(T0_fine, r_fine, dt/4, t_end, alpha)
        indices = np.linspace(0, nr_fine - 1, len(r)).astype(int)
        return T_hist[:, indices]

    def _run_single(self, exp_name, timestamp, alpha, nr, dt, t_end,
                    ic_type, ic_scale, solver_name, T0, r, T_ref, lam) -> ExperimentResult:
        """Run a single solver configuration."""
        solver = self.solvers[solver_name]

        t0 = time.perf_counter()
        T_hist = solver.solve(T0.copy(), r, dt, t_end, alpha)
        wall_time = time.perf_counter() - t0

        # Check stability
        is_nan = bool(np.any(np.isnan(T_hist)))
        max_T = float(np.max(np.abs(T_hist))) if not is_nan else float('inf')
        min_T = float(np.min(T_hist)) if not is_nan else float('-inf')
        is_stable = not is_nan and max_T < 100

        # Compute errors
        if is_stable:
            nt_ref, nt_sol = T_ref.shape[0], T_hist.shape[0]
            if nt_sol != nt_ref:
                T_cmp = np.stack([T_hist[0], T_hist[-1]])
                T_ref_cmp = np.stack([T_ref[0], T_ref[-1]])
            else:
                T_cmp, T_ref_cmp = T_hist, T_ref
            errs = compute_errors(T_cmp, T_ref_cmp, r)
            l2_error = errs["l2"]
            linf_error = errs["linf"]
        else:
            l2_error = float('nan')
            linf_error = float('nan')

        score = l2_error + lam * wall_time if is_stable else float('inf')

        return ExperimentResult(
            experiment_name=exp_name,
            timestamp=timestamp,
            alpha=alpha,
            nr=nr,
            dt=dt,
            t_end=t_end,
            ic_type=ic_type,
            ic_scale=ic_scale,
            solver=solver_name,
            l2_error=l2_error,
            linf_error=linf_error,
            wall_time=wall_time,
            max_T=max_T,
            min_T=min_T,
            is_stable=is_stable,
            is_nan=is_nan,
            score=score,
        )

    def _save_results(self, results: List[ExperimentResult]):
        """Append results to CSV database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        file_exists = os.path.isfile(self.db_path)
        fieldnames = list(asdict(results[0]).keys())

        with open(self.db_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))

    def load_results(self) -> List[Dict]:
        """Load all results from database."""
        if not os.path.isfile(self.db_path):
            return []
        with open(self.db_path) as f:
            return list(csv.DictReader(f))


# =============================================================================
# Analyzer
# =============================================================================

class ExperimentAnalyzer:
    """Analyzes experiment results and tests hypotheses."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self.db_path = db_path

    def load_data(self) -> List[Dict]:
        """Load experiment data."""
        if not os.path.isfile(self.db_path):
            return []
        with open(self.db_path) as f:
            return list(csv.DictReader(f))

    def analyze(self, experiment_name: Optional[str] = None) -> Dict:
        """Analyze results from experiments."""
        data = self.load_data()
        if experiment_name:
            data = [d for d in data if d["experiment_name"] == experiment_name]

        if not data:
            return {"error": "No data found"}

        # Convert types
        for d in data:
            d["alpha"] = float(d["alpha"])
            d["nr"] = int(d["nr"])
            d["dt"] = float(d["dt"])
            d["is_stable"] = d["is_stable"] == "True"
            d["l2_error"] = float(d["l2_error"]) if d["l2_error"] != "nan" else float("nan")
            d["wall_time"] = float(d["wall_time"])

        analysis = {
            "total_runs": len(data),
            "experiments": list(set(d["experiment_name"] for d in data)),
            "solvers": {},
            "stability_analysis": {},
            "winner_analysis": {},
        }

        # Per-solver analysis
        for solver in ["implicit_fdm", "spectral_cosine"]:
            solver_data = [d for d in data if d["solver"] == solver]
            stable = [d for d in solver_data if d["is_stable"]]

            analysis["solvers"][solver] = {
                "total": len(solver_data),
                "stable": len(stable),
                "stable_pct": len(stable) / len(solver_data) * 100 if solver_data else 0,
                "avg_l2": np.nanmean([d["l2_error"] for d in stable]) if stable else float("nan"),
                "avg_time": np.mean([d["wall_time"] for d in stable]) if stable else 0,
            }

        # Stability by alpha
        alphas = sorted(set(d["alpha"] for d in data))
        for solver in ["implicit_fdm", "spectral_cosine"]:
            analysis["stability_analysis"][solver] = {}
            for alpha in alphas:
                subset = [d for d in data if d["solver"] == solver and d["alpha"] == alpha]
                stable = sum(1 for d in subset if d["is_stable"])
                analysis["stability_analysis"][solver][alpha] = {
                    "stable": stable,
                    "total": len(subset),
                    "pct": stable / len(subset) * 100 if subset else 0,
                }

        # Winner analysis (per configuration)
        configs = set((d["alpha"], d["nr"], d["dt"], d["t_end"], d["ic_type"]) for d in data)
        winners = {"implicit_fdm": 0, "spectral_cosine": 0, "tie": 0}

        for config in configs:
            alpha, nr, dt, t_end, ic_type = config
            subset = [d for d in data if
                      d["alpha"] == alpha and d["nr"] == nr and
                      d["dt"] == dt and d["t_end"] == float(t_end) and
                      d["ic_type"] == ic_type and d["is_stable"]]

            if len(subset) >= 2:
                best = min(subset, key=lambda x: x["l2_error"] + 0.1 * x["wall_time"])
                winners[best["solver"]] += 1
            elif len(subset) == 1:
                winners[subset[0]["solver"]] += 1

        analysis["winner_analysis"] = winners

        return analysis

    def test_hypothesis(self, hypothesis_id: str, data: Optional[List[Dict]] = None) -> Dict:
        """Test a specific hypothesis against the data."""
        if data is None:
            data = self.load_data()

        # Convert types
        for d in data:
            d["alpha"] = float(d["alpha"])
            d["dt"] = float(d["dt"])
            d["is_stable"] = d["is_stable"] == "True"

        results = {"hypothesis": hypothesis_id, "tested": True}

        if hypothesis_id == "H1":
            # Spectral stability improves with smaller dt
            spectral = [d for d in data if d["solver"] == "spectral_cosine"]
            dt_stability = {}
            for d in spectral:
                dt = d["dt"]
                if dt not in dt_stability:
                    dt_stability[dt] = {"stable": 0, "total": 0}
                dt_stability[dt]["total"] += 1
                if d["is_stable"]:
                    dt_stability[dt]["stable"] += 1

            results["dt_stability"] = {
                dt: s["stable"]/s["total"]*100 if s["total"] > 0 else 0
                for dt, s in sorted(dt_stability.items())
            }
            results["confirmed"] = any(
                dt_stability.get(dt, {}).get("stable", 0) > 0
                for dt in sorted(dt_stability.keys())[:2]  # smallest dt values
            )

        elif hypothesis_id == "H7":
            # Spectral fails with overflow for high alpha
            spectral = [d for d in data if d["solver"] == "spectral_cosine"]
            alpha_stability = {}
            for d in spectral:
                alpha = d["alpha"]
                if alpha not in alpha_stability:
                    alpha_stability[alpha] = {"stable": 0, "unstable": 0}
                if d["is_stable"]:
                    alpha_stability[alpha]["stable"] += 1
                else:
                    alpha_stability[alpha]["unstable"] += 1

            results["alpha_stability"] = alpha_stability
            # Find threshold where stability drops
            threshold = None
            for alpha in sorted(alpha_stability.keys()):
                s = alpha_stability[alpha]
                if s["unstable"] > s["stable"]:
                    threshold = alpha
                    break
            results["instability_threshold"] = threshold
            results["confirmed"] = threshold is not None

        elif hypothesis_id == "H4":
            # Different ICs affect solver performance
            ic_winners = {}
            for ic_type in set(d["ic_type"] for d in data):
                subset = [d for d in data if d["ic_type"] == ic_type and d["is_stable"]]
                if subset:
                    fdm = [d for d in subset if d["solver"] == "implicit_fdm"]
                    spec = [d for d in subset if d["solver"] == "spectral_cosine"]
                    fdm_errors = [float(d["l2_error"]) for d in fdm if d["l2_error"] != "nan"]
                    spec_errors = [float(d["l2_error"]) for d in spec if d["l2_error"] != "nan"]
                    fdm_avg = np.mean(fdm_errors) if fdm_errors else float("inf")
                    spec_avg = np.mean(spec_errors) if spec_errors else float("inf")
                    ic_winners[ic_type] = {
                        "winner": "spectral" if spec_avg < fdm_avg else "fdm",
                        "fdm_l2": fdm_avg,
                        "spectral_l2": spec_avg,
                    }

            results["ic_winners"] = ic_winners
            results["confirmed"] = len(set(w["winner"] for w in ic_winners.values())) > 1

        return results

    def print_report(self, analysis: Dict):
        """Print analysis report."""
        print("\n" + "=" * 60)
        print("EXPERIMENT ANALYSIS REPORT")
        print("=" * 60)

        print(f"\nTotal runs: {analysis['total_runs']}")
        print(f"Experiments: {', '.join(analysis['experiments'])}")

        print("\n## Solver Summary")
        for solver, stats in analysis["solvers"].items():
            print(f"\n  {solver}:")
            print(f"    Runs: {stats['total']}")
            print(f"    Stable: {stats['stable']} ({stats['stable_pct']:.1f}%)")
            print(f"    Avg L2: {stats['avg_l2']:.6f}")
            print(f"    Avg Time: {stats['avg_time']*1000:.2f}ms")

        print("\n## Stability by Alpha (Spectral)")
        spec_stab = analysis["stability_analysis"].get("spectral_cosine", {})
        for alpha, stats in sorted(spec_stab.items()):
            bar = "█" * int(stats["pct"] / 10) + "░" * (10 - int(stats["pct"] / 10))
            print(f"  α={alpha}: {bar} {stats['pct']:.0f}% ({stats['stable']}/{stats['total']})")

        print("\n## Winner Distribution")
        winners = analysis["winner_analysis"]
        total = sum(winners.values())
        for solver, count in winners.items():
            pct = count / total * 100 if total > 0 else 0
            print(f"  {solver}: {count} ({pct:.1f}%)")


# =============================================================================
# Predefined Experiments
# =============================================================================

EXPERIMENTS = {
    "stability_map": ExperimentConfig(
        name="stability_map",
        description="Map spectral stability across (alpha, dt) space",
        alpha_list=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0],
        nr_list=[51],
        dt_list=[0.002, 0.001, 0.0005, 0.0002, 0.0001],
        t_end_list=[0.1],
        ic_type="parabola",
    ),

    "ic_comparison": ExperimentConfig(
        name="ic_comparison",
        description="Compare solver performance across different ICs",
        alpha_list=[0.0, 0.5],
        nr_list=[51],
        dt_list=[0.0005],  # Use stable dt for spectral
        t_end_list=[0.1],
        ic_type="parabola",  # Will be overridden
    ),

    "linear_regime": ExperimentConfig(
        name="linear_regime",
        description="Test solvers in purely linear regime (scaled IC)",
        alpha_list=[0.0],
        nr_list=[31, 51, 71],
        dt_list=[0.001, 0.0005],
        t_end_list=[0.1],
        ic_type="parabola",
        ic_scale=0.2,  # max|dT/dr| = 0.4 < 0.5
    ),

    "fine_sweep": ExperimentConfig(
        name="fine_sweep",
        description="Fine-grained parameter sweep",
        alpha_list=[0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
        nr_list=[31, 41, 51, 61, 71],
        dt_list=[0.002, 0.001, 0.0005],
        t_end_list=[0.05, 0.1, 0.2],
        ic_type="parabola",
    ),
}


def run_ic_comparison(runner: ExperimentRunner):
    """Special experiment comparing multiple ICs."""
    all_results = []

    for ic_type in ["parabola", "gaussian", "cosine", "sine"]:
        config = ExperimentConfig(
            name=f"ic_comparison_{ic_type}",
            description=f"IC comparison: {ic_type}",
            alpha_list=[0.0, 0.2, 0.5],
            nr_list=[51],
            dt_list=[0.0005],
            t_end_list=[0.1],
            ic_type=ic_type,
        )
        results = runner.run_experiment(config, verbose=False)
        all_results.extend(results)
        print(f"  Completed IC: {ic_type}")

    return all_results


# =============================================================================
# Interactive Mode
# =============================================================================

def interactive_mode():
    """Run interactive experiment session."""
    runner = ExperimentRunner()
    analyzer = ExperimentAnalyzer()
    tracker = HypothesisTracker()

    # Initialize default hypotheses if tracker is empty
    if not tracker.hypotheses:
        print("Initializing default hypotheses...")
        for hid, statement in DEFAULT_HYPOTHESES.items():
            tracker.add_hypothesis(hid, statement)

    print("\n" + "=" * 60)
    print("INTERACTIVE EXPERIMENT FRAMEWORK")
    print("=" * 60)
    print("""
Commands:
  list            - List predefined experiments
  run <name>      - Run a predefined experiment
  run ic          - Run IC comparison experiment
  analyze         - Analyze all results
  analyze <exp>   - Analyze specific experiment
  test <H1-H10>   - Test a hypothesis
  cycle <n>       - Run n hypothesis verification cycles
  custom          - Create custom experiment

Hypothesis Management:
  hypo            - List all hypotheses
  hypo add <ID> <statement> - Add new hypothesis
  hypo note <ID> <note>     - Add note to hypothesis
  hypo status <ID> <status> - Update status (confirmed/rejected/inconclusive)

Reporting:
  report          - Generate final markdown report
  clear           - Clear database
  quit            - Exit

Auto Mode:
  auto            - Run full analysis: generate data, run cycles, generate report
  auto <n>        - Run n verification cycles with auto data generation
  fresh <n>       - Run n cycles with fresh database each time (more accurate)
""")

    while True:
        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if cmd == "quit" or cmd == "exit":
            break

        elif cmd == "list":
            print("\nPredefined experiments:")
            for name, config in EXPERIMENTS.items():
                total = (len(config.alpha_list) * len(config.nr_list) *
                         len(config.dt_list) * len(config.t_end_list) * 2)
                print(f"  {name}: {config.description} ({total} runs)")

        elif cmd.startswith("run "):
            exp_name = cmd[4:].strip()
            if exp_name == "ic":
                print("\nRunning IC comparison experiment...")
                run_ic_comparison(runner)
            elif exp_name in EXPERIMENTS:
                runner.run_experiment(EXPERIMENTS[exp_name])
            else:
                print(f"Unknown experiment: {exp_name}")

        elif cmd.startswith("analyze"):
            parts = cmd.split()
            exp_name = parts[1] if len(parts) > 1 else None
            analysis = analyzer.analyze(exp_name)
            if "error" in analysis:
                print(f"Error: {analysis['error']}")
            else:
                analyzer.print_report(analysis)

        elif cmd.startswith("test "):
            hyp_id = cmd[5:].strip().upper()
            result = analyzer.test_hypothesis(hyp_id)
            print(f"\nHypothesis {hyp_id}:")
            print(f"  Confirmed: {result.get('confirmed', 'N/A')}")
            for k, v in result.items():
                if k not in ["hypothesis", "tested", "confirmed"]:
                    print(f"  {k}: {v}")

        elif cmd == "custom":
            print("\nCustom experiment creator:")
            name = input("  Name: ").strip()
            desc = input("  Description: ").strip()
            alphas = input("  Alpha values (comma-sep): ").strip()
            dts = input("  dt values (comma-sep): ").strip()

            config = ExperimentConfig(
                name=name,
                description=desc,
                alpha_list=[float(x) for x in alphas.split(",")],
                nr_list=[51],
                dt_list=[float(x) for x in dts.split(",")],
                t_end_list=[0.1],
            )
            runner.run_experiment(config)

        elif cmd == "clear":
            if os.path.exists(runner.db_path):
                os.remove(runner.db_path)
                print("Database cleared.")

        elif cmd == "hypo":
            tracker.list_hypotheses()

        elif cmd.startswith("hypo add "):
            parts = cmd[9:].strip().split(" ", 1)
            if len(parts) >= 2:
                hid, statement = parts
                tracker.add_hypothesis(hid.upper(), statement)
                print(f"Added hypothesis {hid.upper()}: {statement}")
            else:
                print("Usage: hypo add <ID> <statement>")

        elif cmd.startswith("hypo note "):
            parts = cmd[10:].strip().split(" ", 1)
            if len(parts) >= 2:
                hid, note = parts
                tracker.add_note(hid.upper(), note)
                print(f"Added note to {hid.upper()}")
            else:
                print("Usage: hypo note <ID> <note>")

        elif cmd.startswith("hypo status "):
            parts = cmd[12:].strip().split()
            if len(parts) >= 2:
                hid, status = parts[0].upper(), parts[1].lower()
                if hid in tracker.hypotheses and status in ["confirmed", "rejected", "inconclusive"]:
                    tracker.hypotheses[hid].status = status
                    tracker._save_memos()
                    print(f"Updated {hid} status to {status}")
                else:
                    print("Invalid hypothesis ID or status")
            else:
                print("Usage: hypo status <ID> <confirmed|rejected|inconclusive>")

        elif cmd.startswith("cycle"):
            parts = cmd.split()
            n_cycles = int(parts[1]) if len(parts) > 1 else 3

            print(f"\n{'='*60}")
            print(f"RUNNING {n_cycles} HYPOTHESIS VERIFICATION CYCLES")
            print(f"{'='*60}")

            for cycle in range(1, n_cycles + 1):
                print(f"\n--- Cycle {cycle}/{n_cycles} ---")

                # Run experiments
                print("Running stability map experiment...")
                runner.run_experiment(EXPERIMENTS["stability_map"], verbose=False)

                # Test all registered hypotheses
                data = analyzer.load_data()
                for hid in tracker.hypotheses.keys():
                    result = analyzer.test_hypothesis(hid, data)
                    tracker.record_verification(hid, result, "stability_map")
                    status = "Confirmed" if result.get("confirmed") else "Not confirmed"
                    print(f"  {hid}: {status}")

                # Summary
                summary = tracker.get_summary()
                print(f"\nCycle {cycle} Summary:")
                print(f"  Confirmed: {len(summary['confirmed'])}")
                print(f"  Rejected: {len(summary['rejected'])}")
                print(f"  Inconclusive: {len(summary['inconclusive'])}")

            print(f"\nCompleted {n_cycles} cycles. Use 'report' to generate final report.")

        elif cmd == "report":
            analysis = analyzer.analyze()
            if "error" in analysis:
                print(f"Error: {analysis['error']}")
            else:
                report_path = tracker.generate_markdown_report(analysis)
                print(f"\nReport generated: {report_path}")

                # Also print summary to console
                analyzer.print_report(analysis)
                tracker.list_hypotheses()

        elif cmd.startswith("auto"):
            parts = cmd.split()
            n_cycles = int(parts[1]) if len(parts) > 1 else 3

            print(f"\n{'='*60}")
            print("AUTO MODE: FULL ANALYSIS PIPELINE")
            print(f"{'='*60}")

            # Step 1: Check if data exists, generate if not
            data = analyzer.load_data()
            if not data or len(data) < 50:
                print("\n[Step 1/4] Generating initial dataset...")
                print("Running stability_map experiment...")
                runner.run_experiment(EXPERIMENTS["stability_map"], verbose=False)
                print("Running IC comparison experiment...")
                run_ic_comparison(runner)
            else:
                print(f"\n[Step 1/4] Using existing data ({len(data)} samples)")

            # Step 2: Run verification cycles
            print(f"\n[Step 2/4] Running {n_cycles} verification cycles...")
            for cycle in range(1, n_cycles + 1):
                print(f"\n  Cycle {cycle}/{n_cycles}:")
                runner.run_experiment(EXPERIMENTS["stability_map"], verbose=False)

                data = analyzer.load_data()
                for hid in tracker.hypotheses.keys():
                    result = analyzer.test_hypothesis(hid, data)
                    tracker.record_verification(hid, result, "auto_cycle")

                summary = tracker.get_summary()
                print(f"    Confirmed: {len(summary['confirmed'])}, "
                      f"Rejected: {len(summary['rejected'])}, "
                      f"Inconclusive: {len(summary['inconclusive'])}")

            # Step 3: Analyze
            print("\n[Step 3/4] Analyzing all data...")
            analysis = analyzer.analyze()
            analyzer.print_report(analysis)

            # Step 4: Generate report
            print("\n[Step 4/4] Generating final report...")
            report_path = tracker.generate_markdown_report(analysis)
            print(f"Report saved to: {report_path}")

            tracker.list_hypotheses()
            print(f"\n{'='*60}")
            print("AUTO MODE COMPLETE")
            print(f"{'='*60}")

        elif cmd.startswith("fresh"):
            # Fresh mode: regenerate database each cycle for accurate verification
            parts = cmd.split()
            n_cycles = int(parts[1]) if len(parts) > 1 else 3

            print(f"\n{'='*60}")
            print("FRESH MODE: INDEPENDENT VERIFICATION CYCLES")
            print("(Database regenerated for each cycle)")
            print(f"{'='*60}")

            cycle_results = []

            for cycle in range(1, n_cycles + 1):
                print(f"\n--- Cycle {cycle}/{n_cycles} ---")

                # Clear database
                if os.path.exists(runner.db_path):
                    os.remove(runner.db_path)

                # Generate fresh reference data
                print("  Generating fresh reference data...")
                runner.run_experiment(EXPERIMENTS["stability_map"], verbose=False)
                run_ic_comparison(runner)

                # Test hypotheses
                data = analyzer.load_data()
                print(f"  Testing against {len(data)} fresh samples...")

                cycle_result = {}
                for hid in tracker.hypotheses.keys():
                    result = analyzer.test_hypothesis(hid, data)
                    tracker.record_verification(hid, result, f"fresh_cycle_{cycle}")
                    cycle_result[hid] = result.get("confirmed", False)
                    status = "Confirmed" if result.get("confirmed") else "Not confirmed"
                    print(f"    {hid}: {status}")

                cycle_results.append(cycle_result)

            # Summary across cycles
            print(f"\n{'='*60}")
            print("CROSS-CYCLE CONSISTENCY")
            print(f"{'='*60}")

            for hid in tracker.hypotheses.keys():
                confirmations = sum(1 for cr in cycle_results if cr.get(hid, False))
                consistency = confirmations / n_cycles * 100
                print(f"  {hid}: {confirmations}/{n_cycles} confirmed ({consistency:.0f}%)")

            # Generate report
            analysis = analyzer.analyze()
            report_path = tracker.generate_markdown_report(analysis)
            print(f"\nReport saved to: {report_path}")

            tracker.list_hypotheses()

        else:
            print("Unknown command. Type 'list' for available experiments.")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment Framework")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--run-experiment", "-r", type=str,
                        help="Run a specific experiment")
    parser.add_argument("--analyze", "-a", action="store_true",
                        help="Analyze all results")
    parser.add_argument("--test", "-t", type=str,
                        help="Test a hypothesis (H1-H10)")
    parser.add_argument("--cycles", "-c", type=int, default=0,
                        help="Run N hypothesis verification cycles")
    parser.add_argument("--report", action="store_true",
                        help="Generate final markdown report")
    parser.add_argument("--init-hypotheses", action="store_true",
                        help="Initialize default hypotheses")
    parser.add_argument("--auto", type=int, nargs="?", const=3, default=None,
                        help="Auto mode: generate data, run N cycles (default 3), generate report")
    parser.add_argument("--fresh", action="store_true",
                        help="Regenerate reference database for each verification cycle")
    args = parser.parse_args()

    tracker = HypothesisTracker()

    if args.init_hypotheses or not tracker.hypotheses:
        print("Initializing default hypotheses...")
        for hid, statement in DEFAULT_HYPOTHESES.items():
            if hid not in tracker.hypotheses:
                tracker.add_hypothesis(hid, statement)

    if args.interactive:
        interactive_mode()
    elif args.auto is not None:
        # Auto mode: full analysis pipeline
        runner = ExperimentRunner()
        analyzer = ExperimentAnalyzer()
        n_cycles = args.auto
        fresh_mode = args.fresh

        print(f"\n{'='*60}")
        print("AUTO MODE: FULL ANALYSIS PIPELINE")
        if fresh_mode:
            print("(Fresh mode: regenerating database each cycle)")
        print(f"{'='*60}")

        # Step 1: Initial data generation
        if fresh_mode:
            print("\n[Step 1/4] Fresh mode - will regenerate data each cycle")
        else:
            data = analyzer.load_data()
            if not data or len(data) < 50:
                print("\n[Step 1/4] Generating initial dataset...")
                print("Running stability_map experiment...")
                runner.run_experiment(EXPERIMENTS["stability_map"], verbose=False)
                print("Running IC comparison experiment...")
                run_ic_comparison(runner)
            else:
                print(f"\n[Step 1/4] Using existing data ({len(data)} samples)")

        # Step 2: Run verification cycles
        print(f"\n[Step 2/4] Running {n_cycles} verification cycles...")
        for cycle in range(1, n_cycles + 1):
            print(f"\n  Cycle {cycle}/{n_cycles}:")

            if fresh_mode:
                # Clear database and regenerate fresh data
                if os.path.exists(runner.db_path):
                    os.remove(runner.db_path)
                print("    Generating fresh reference data...")
                runner.run_experiment(EXPERIMENTS["stability_map"], verbose=False)
                run_ic_comparison(runner)
            else:
                runner.run_experiment(EXPERIMENTS["stability_map"], verbose=False)

            data = analyzer.load_data()
            print(f"    Testing hypotheses against {len(data)} samples...")
            for hid in tracker.hypotheses.keys():
                result = analyzer.test_hypothesis(hid, data)
                tracker.record_verification(hid, result, f"cycle_{cycle}")

            summary = tracker.get_summary()
            print(f"    Confirmed: {len(summary['confirmed'])}, "
                  f"Rejected: {len(summary['rejected'])}, "
                  f"Inconclusive: {len(summary['inconclusive'])}")

        # Step 3: Analyze
        print("\n[Step 3/4] Analyzing all data...")
        analysis = analyzer.analyze()
        analyzer.print_report(analysis)

        # Step 4: Generate report
        print("\n[Step 4/4] Generating final report...")
        report_path = tracker.generate_markdown_report(analysis)
        print(f"Report saved to: {report_path}")

        tracker.list_hypotheses()
        print(f"\n{'='*60}")
        print("AUTO MODE COMPLETE")
        print(f"{'='*60}")

    elif args.cycles > 0:
        # Run N verification cycles
        runner = ExperimentRunner()
        analyzer = ExperimentAnalyzer()
        fresh_mode = args.fresh

        print(f"\n{'='*60}")
        print(f"RUNNING {args.cycles} HYPOTHESIS VERIFICATION CYCLES")
        if fresh_mode:
            print("(Fresh mode: regenerating database each cycle)")
        print(f"{'='*60}")

        for cycle in range(1, args.cycles + 1):
            print(f"\n--- Cycle {cycle}/{args.cycles} ---")

            if fresh_mode:
                # Clear database and regenerate fresh data
                if os.path.exists(runner.db_path):
                    os.remove(runner.db_path)
                print("Generating fresh reference data...")
                runner.run_experiment(EXPERIMENTS["stability_map"], verbose=False)
                run_ic_comparison(runner)
            else:
                # Run experiments (append to existing)
                print("Running stability map experiment...")
                runner.run_experiment(EXPERIMENTS["stability_map"], verbose=False)

            # Test all registered hypotheses
            data = analyzer.load_data()
            print(f"Testing hypotheses against {len(data)} samples...")
            for hid in tracker.hypotheses.keys():
                result = analyzer.test_hypothesis(hid, data)
                tracker.record_verification(hid, result, f"cycle_{cycle}")
                status = "Confirmed" if result.get("confirmed") else "Not confirmed"
                print(f"  {hid}: {status}")

            # Summary
            summary = tracker.get_summary()
            print(f"\nCycle {cycle} Summary:")
            print(f"  Confirmed: {len(summary['confirmed'])}")
            print(f"  Rejected: {len(summary['rejected'])}")
            print(f"  Inconclusive: {len(summary['inconclusive'])}")

        # Generate report after cycles
        analysis = analyzer.analyze()
        report_path = tracker.generate_markdown_report(analysis)
        print(f"\nFinal report generated: {report_path}")

    elif args.report:
        analyzer = ExperimentAnalyzer()
        analysis = analyzer.analyze()
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
        else:
            report_path = tracker.generate_markdown_report(analysis)
            print(f"Report generated: {report_path}")
            analyzer.print_report(analysis)
            tracker.list_hypotheses()

    elif args.run_experiment:
        runner = ExperimentRunner()
        if args.run_experiment == "ic":
            run_ic_comparison(runner)
        elif args.run_experiment in EXPERIMENTS:
            runner.run_experiment(EXPERIMENTS[args.run_experiment])
        else:
            print(f"Unknown experiment: {args.run_experiment}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
    elif args.analyze:
        analyzer = ExperimentAnalyzer()
        analysis = analyzer.analyze()
        analyzer.print_report(analysis)
    elif args.test:
        analyzer = ExperimentAnalyzer()
        result = analyzer.test_hypothesis(args.test.upper())
        tracker.record_verification(args.test.upper(), result)
        print(json.dumps(result, indent=2, default=str))
    else:
        # Default: run a quick demo
        print("Running quick stability map experiment...")
        runner = ExperimentRunner()
        config = ExperimentConfig(
            name="quick_demo",
            description="Quick demo of stability mapping",
            alpha_list=[0.0, 0.2, 0.5, 1.0],
            nr_list=[51],
            dt_list=[0.001, 0.0005],
            t_end_list=[0.1],
        )
        runner.run_experiment(config)

        analyzer = ExperimentAnalyzer()
        analysis = analyzer.analyze("quick_demo")
        analyzer.print_report(analysis)


if __name__ == "__main__":
    main()
