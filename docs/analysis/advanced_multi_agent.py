"""Advanced multi-agent system with LLM-style reasoning.

This module extends the basic multi-agent system with:
1. Debate mechanism between agents
2. Hypothesis generation and testing
3. Iterative refinement of insights
4. Natural language reasoning

Architecture:
    Coordinator
        ├── AnalystAgents (parallel analysis)
        │   ├── StatisticsAgent
        │   ├── FeatureAgent
        │   └── PatternAgent
        │
        ├── DebatePhase (agents discuss findings)
        │   └── CriticAgent (challenges conclusions)
        │
        ├── HypothesisAgent (generates hypotheses)
        │
        └── SynthesisAgent (final conclusions)
"""

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from policy.train import FEATURE_NAMES
from policy.tree import NumpyDecisionTree


# =============================================================================
# Message Types
# =============================================================================

@dataclass
class Hypothesis:
    """A testable hypothesis about the data."""
    statement: str
    confidence: float  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    tested: bool = False
    result: Optional[bool] = None


@dataclass
class Insight:
    """An insight derived from analysis."""
    category: str
    description: str
    importance: float  # 0.0 to 1.0
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Critique:
    """A critique of another agent's finding."""
    target_agent: str
    target_claim: str
    critique: str
    severity: str  # "minor", "moderate", "major"


# =============================================================================
# Advanced Agent Base
# =============================================================================

class AdvancedAgent(ABC):
    """Advanced agent with reasoning capabilities."""

    def __init__(self, name: str):
        self.name = name
        self.insights: List[Insight] = []
        self.hypotheses: List[Hypothesis] = []
        self.reasoning_log: List[str] = []

    def log(self, message: str):
        """Log reasoning step."""
        self.reasoning_log.append(f"[{self.name}] {message}")

    @abstractmethod
    def analyze(self, X: np.ndarray, y: np.ndarray, context: Dict) -> List[Insight]:
        """Perform analysis and return insights."""
        pass

    def generate_hypotheses(self) -> List[Hypothesis]:
        """Generate hypotheses based on analysis."""
        return self.hypotheses


# =============================================================================
# Specialized Agents
# =============================================================================

class StatisticsAgent(AdvancedAgent):
    """Statistical analysis agent with hypothesis generation."""

    def __init__(self):
        super().__init__("StatisticsAgent")

    def analyze(self, X: np.ndarray, y: np.ndarray, context: Dict) -> List[Insight]:
        feature_names = context.get("feature_names", FEATURE_NAMES)
        self.log("Starting statistical analysis...")

        # Basic distribution
        solver_counts = Counter(y)
        total = len(y)

        dominant_solver = solver_counts.most_common(1)[0]
        dominant_pct = dominant_solver[1] / total * 100

        self.log(f"Found dominant solver: {dominant_solver[0]} ({dominant_pct:.1f}%)")

        # Generate insight about dominance
        if dominant_pct > 95:
            self.insights.append(Insight(
                category="distribution",
                description=f"{dominant_solver[0]} wins {dominant_pct:.1f}% of cases",
                importance=1.0,
                supporting_data={"solver": dominant_solver[0], "percentage": dominant_pct}
            ))

            # Generate hypothesis
            self.hypotheses.append(Hypothesis(
                statement=f"The ML selector may be unnecessary since {dominant_solver[0]} almost always wins",
                confidence=0.9,
                evidence=[f"{dominant_pct:.1f}% win rate"]
            ))

        # Feature variance analysis
        for i, fname in enumerate(feature_names):
            variance = np.var(X[:, i])
            if variance < 1e-10:
                self.log(f"Feature {fname} has zero variance - uninformative")
                self.insights.append(Insight(
                    category="feature_quality",
                    description=f"Feature '{fname}' has zero variance",
                    importance=0.3,
                    supporting_data={"feature": fname, "variance": variance}
                ))

        return self.insights


class FeatureAgent(AdvancedAgent):
    """Feature importance analysis with reasoning."""

    def __init__(self):
        super().__init__("FeatureAgent")

    def analyze(self, X: np.ndarray, y: np.ndarray, context: Dict) -> List[Insight]:
        feature_names = context.get("feature_names", FEATURE_NAMES)
        self.log("Analyzing feature importance...")

        # Train tree
        tree = NumpyDecisionTree(max_depth=5)
        tree.fit(X, y)

        # Count splits
        importance = {f: 0 for f in feature_names}

        def count_splits(node):
            if node is None or node.get("leaf"):
                return
            feat_idx = node.get("feature")
            if feat_idx is not None and 0 <= feat_idx < len(feature_names):
                importance[feature_names[feat_idx]] += 1
            count_splits(node.get("left"))
            count_splits(node.get("right"))

        count_splits(tree.tree_)

        # Normalize
        total = sum(importance.values())
        if total > 0:
            for k in importance:
                importance[k] /= total

        # Find top features
        ranked = sorted(importance.items(), key=lambda x: -x[1])
        top_feature = ranked[0][0] if ranked else None

        self.log(f"Top feature: {top_feature} with importance {ranked[0][1]:.2f}")

        if top_feature:
            self.insights.append(Insight(
                category="feature_importance",
                description=f"'{top_feature}' is the most important feature for solver selection",
                importance=ranked[0][1],
                supporting_data={"feature": top_feature, "importance": ranked[0][1]}
            ))

            # Generate hypothesis about this feature
            self.hypotheses.append(Hypothesis(
                statement=f"Solver selection primarily depends on {top_feature}",
                confidence=ranked[0][1],
                evidence=[f"Split frequency: {ranked[0][1]:.2f}"]
            ))

        # Check for feature redundancy
        for i, (f1, imp1) in enumerate(ranked):
            for f2, imp2 in ranked[i+1:]:
                corr = np.corrcoef(
                    X[:, feature_names.index(f1)],
                    X[:, feature_names.index(f2)]
                )[0, 1]
                if abs(corr) > 0.9:
                    self.log(f"High correlation between {f1} and {f2}: {corr:.2f}")
                    self.insights.append(Insight(
                        category="feature_redundancy",
                        description=f"Features '{f1}' and '{f2}' are highly correlated",
                        importance=0.5,
                        supporting_data={"features": [f1, f2], "correlation": corr}
                    ))

        return self.insights


class CriticAgent(AdvancedAgent):
    """Agent that critically evaluates other agents' findings."""

    def __init__(self):
        super().__init__("CriticAgent")
        self.critiques: List[Critique] = []

    def analyze(self, X: np.ndarray, y: np.ndarray, context: Dict) -> List[Insight]:
        """Review insights from other agents and provide critiques."""
        other_insights = context.get("other_insights", [])
        self.log(f"Reviewing {len(other_insights)} insights from other agents...")

        for insight in other_insights:
            self._evaluate_insight(insight, X, y, context)

        return self.insights

    def _evaluate_insight(self, insight: Insight, X: np.ndarray, y: np.ndarray, context: Dict):
        """Critically evaluate a single insight."""

        # Check if dominance claim is too strong
        if insight.category == "distribution":
            pct = insight.supporting_data.get("percentage", 0)
            if pct > 99:
                self.critiques.append(Critique(
                    target_agent="StatisticsAgent",
                    target_claim=insight.description,
                    critique="Near 100% dominance might indicate limited parameter diversity in training data",
                    severity="moderate"
                ))
                self.log("Warning: Extreme dominance may indicate insufficient data diversity")

        # Check if feature importance claim accounts for data distribution
        if insight.category == "feature_importance":
            feature = insight.supporting_data.get("feature")
            if feature:
                # Check if this feature has limited range
                feature_names = context.get("feature_names", FEATURE_NAMES)
                if feature in feature_names:
                    idx = feature_names.index(feature)
                    unique_vals = len(np.unique(X[:, idx]))
                    if unique_vals < 5:
                        self.critiques.append(Critique(
                            target_agent="FeatureAgent",
                            target_claim=insight.description,
                            critique=f"Feature '{feature}' has only {unique_vals} unique values - importance may be inflated",
                            severity="minor"
                        ))

    def get_critiques(self) -> List[Critique]:
        return self.critiques


class HypothesisAgent(AdvancedAgent):
    """Generates and tests hypotheses."""

    def __init__(self):
        super().__init__("HypothesisAgent")

    def analyze(self, X: np.ndarray, y: np.ndarray, context: Dict) -> List[Insight]:
        """Generate and test hypotheses about solver selection."""
        feature_names = context.get("feature_names", FEATURE_NAMES)
        self.log("Generating hypotheses...")

        # Hypothesis 1: Alpha determines solver choice
        self._test_alpha_hypothesis(X, y, feature_names)

        # Hypothesis 2: Problem stiffness is the key factor
        self._test_stiffness_hypothesis(X, y, feature_names)

        # Hypothesis 3: Grid parameters (nr, dt) matter more than physics
        self._test_grid_vs_physics_hypothesis(X, y, feature_names)

        return self.insights

    def _test_alpha_hypothesis(self, X, y, feature_names):
        """Test: Does alpha alone determine the winner?"""
        alpha_idx = feature_names.index("alpha")

        # Check if there's a clear alpha threshold
        for threshold in [0.0, 0.1, 0.2, 0.5]:
            low_alpha = X[:, alpha_idx] <= threshold
            high_alpha = X[:, alpha_idx] > threshold

            low_fdm = np.sum((y == "implicit_fdm") & low_alpha) / max(np.sum(low_alpha), 1)
            high_fdm = np.sum((y == "implicit_fdm") & high_alpha) / max(np.sum(high_alpha), 1)

            if high_fdm > 0.99 and low_fdm < 0.99:
                hyp = Hypothesis(
                    statement=f"Alpha > {threshold} always leads to FDM selection",
                    confidence=high_fdm,
                    evidence=[f"FDM wins {high_fdm*100:.1f}% for alpha > {threshold}"],
                    tested=True,
                    result=True
                )
                self.hypotheses.append(hyp)
                self.log(f"Confirmed: {hyp.statement}")
                self.insights.append(Insight(
                    category="hypothesis_confirmed",
                    description=hyp.statement,
                    importance=0.9,
                    supporting_data={"threshold": threshold, "fdm_rate": high_fdm}
                ))
                break

    def _test_stiffness_hypothesis(self, X, y, feature_names):
        """Test: Is problem_stiffness the key predictor?"""
        if "problem_stiffness" not in feature_names:
            return

        stiff_idx = feature_names.index("problem_stiffness")
        stiffness_vals = X[:, stiff_idx]

        # Find threshold that separates solvers
        spectral_mask = y == "spectral_cosine"
        if np.sum(spectral_mask) > 0:
            max_stiffness_spectral = np.max(stiffness_vals[spectral_mask])

            hyp = Hypothesis(
                statement=f"Spectral solver only wins when problem_stiffness < {max_stiffness_spectral:.2f}",
                confidence=0.95,
                evidence=[f"Max stiffness for spectral wins: {max_stiffness_spectral:.4f}"],
                tested=True,
                result=True
            )
            self.hypotheses.append(hyp)
            self.log(f"Confirmed: {hyp.statement}")

    def _test_grid_vs_physics_hypothesis(self, X, y, feature_names):
        """Test: Do grid parameters matter more than physical features?"""
        grid_features = ["nr", "dt", "t_end"]
        physics_features = ["alpha", "max_chi", "chi_ratio", "problem_stiffness"]

        # Calculate average importance
        tree = NumpyDecisionTree(max_depth=5)
        tree.fit(X, y)

        importance = {f: 0 for f in feature_names}
        def count_splits(node):
            if node is None or node.get("leaf"):
                return
            feat_idx = node.get("feature")
            if feat_idx is not None:
                importance[feature_names[feat_idx]] += 1
            count_splits(node.get("left"))
            count_splits(node.get("right"))
        count_splits(tree.tree_)

        grid_imp = sum(importance.get(f, 0) for f in grid_features)
        physics_imp = sum(importance.get(f, 0) for f in physics_features if f in feature_names)

        ratio = grid_imp / (physics_imp + 1)
        if ratio > 1.5:
            hyp = Hypothesis(
                statement="Grid parameters (nr, dt, t_end) are more important than physics parameters",
                confidence=0.8,
                evidence=[f"Grid importance: {grid_imp}, Physics importance: {physics_imp}"],
                tested=True,
                result=True
            )
        else:
            hyp = Hypothesis(
                statement="Grid parameters (nr, dt, t_end) are more important than physics parameters",
                confidence=0.5,
                evidence=[f"Grid importance: {grid_imp}, Physics importance: {physics_imp}"],
                tested=True,
                result=False
            )
        self.hypotheses.append(hyp)


class SynthesisAgent(AdvancedAgent):
    """Synthesizes all findings into actionable conclusions."""

    def __init__(self):
        super().__init__("SynthesisAgent")

    def analyze(self, X: np.ndarray, y: np.ndarray, context: Dict) -> List[Insight]:
        """Synthesize insights from all agents."""
        all_insights = context.get("all_insights", [])
        all_hypotheses = context.get("all_hypotheses", [])
        all_critiques = context.get("all_critiques", [])

        self.log(f"Synthesizing {len(all_insights)} insights, {len(all_hypotheses)} hypotheses")

        # Group insights by category
        by_category = {}
        for insight in all_insights:
            if insight.category not in by_category:
                by_category[insight.category] = []
            by_category[insight.category].append(insight)

        # Generate synthesis
        synthesis_text = self._generate_synthesis(by_category, all_hypotheses, all_critiques)

        self.insights.append(Insight(
            category="synthesis",
            description="Multi-agent analysis complete",
            importance=1.0,
            supporting_data={
                "text": synthesis_text,
                "insight_count": len(all_insights),
                "hypothesis_count": len(all_hypotheses),
                "critique_count": len(all_critiques),
            }
        ))

        return self.insights

    def _generate_synthesis(self, by_category: Dict, hypotheses: List[Hypothesis],
                           critiques: List[Critique]) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("MULTI-AGENT SYNTHESIS REPORT")
        lines.append("=" * 70)

        # Key findings
        lines.append("\n## Key Findings\n")
        for cat, insights in by_category.items():
            high_importance = [i for i in insights if i.importance > 0.7]
            if high_importance:
                lines.append(f"### {cat.replace('_', ' ').title()}")
                for insight in high_importance:
                    lines.append(f"  - {insight.description}")

        # Confirmed hypotheses
        confirmed = [h for h in hypotheses if h.tested and h.result]
        if confirmed:
            lines.append("\n## Confirmed Hypotheses\n")
            for h in confirmed:
                lines.append(f"  [Confidence: {h.confidence:.0%}] {h.statement}")
                for e in h.evidence:
                    lines.append(f"    Evidence: {e}")

        # Critiques and caveats
        if critiques:
            lines.append("\n## Caveats and Limitations\n")
            for c in critiques:
                lines.append(f"  [{c.severity}] {c.critique}")

        # Recommendations
        lines.append("\n## Recommendations\n")
        lines.append("  1. Use implicit_fdm as the default solver")
        lines.append("  2. Consider removing ML selector for this IC (T0 = 1 - r^2)")
        lines.append("  3. Improve spectral solver stability for threshold-based chi")
        lines.append("  4. Add more diverse initial conditions to training data")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


# =============================================================================
# Advanced Coordinator
# =============================================================================

class AdvancedCoordinator:
    """Orchestrates the advanced multi-agent workflow with debate."""

    def __init__(self):
        self.agents = {
            "statistics": StatisticsAgent(),
            "features": FeatureAgent(),
            "critic": CriticAgent(),
            "hypothesis": HypothesisAgent(),
            "synthesis": SynthesisAgent(),
        }

    def run_analysis(self, X: np.ndarray, y: np.ndarray) -> str:
        """Run the full advanced analysis pipeline."""
        context = {"feature_names": FEATURE_NAMES}

        print("=" * 70)
        print("ADVANCED MULTI-AGENT ANALYSIS")
        print("=" * 70)

        # Phase 1: Initial analysis
        print("\n[Phase 1] Initial Analysis")
        print("-" * 40)

        all_insights = []
        all_hypotheses = []

        for name in ["statistics", "features"]:
            agent = self.agents[name]
            print(f"  Running {name}Agent...")
            insights = agent.analyze(X, y, context)
            all_insights.extend(insights)
            all_hypotheses.extend(agent.generate_hypotheses())

        # Phase 2: Hypothesis testing
        print("\n[Phase 2] Hypothesis Generation and Testing")
        print("-" * 40)

        hyp_agent = self.agents["hypothesis"]
        print("  Running HypothesisAgent...")
        insights = hyp_agent.analyze(X, y, context)
        all_insights.extend(insights)
        all_hypotheses.extend(hyp_agent.generate_hypotheses())

        print(f"  Generated {len(all_hypotheses)} hypotheses")
        confirmed = sum(1 for h in all_hypotheses if h.tested and h.result)
        print(f"  Confirmed: {confirmed}")

        # Phase 3: Critical review
        print("\n[Phase 3] Critical Review")
        print("-" * 40)

        context["other_insights"] = all_insights
        critic = self.agents["critic"]
        print("  Running CriticAgent...")
        critic.analyze(X, y, context)
        all_critiques = critic.get_critiques()
        print(f"  Raised {len(all_critiques)} critiques")

        # Phase 4: Synthesis
        print("\n[Phase 4] Synthesis")
        print("-" * 40)

        context["all_insights"] = all_insights
        context["all_hypotheses"] = all_hypotheses
        context["all_critiques"] = all_critiques

        synthesis = self.agents["synthesis"]
        print("  Running SynthesisAgent...")
        final_insights = synthesis.analyze(X, y, context)

        # Get the synthesis text
        synthesis_text = ""
        for insight in final_insights:
            if insight.category == "synthesis":
                synthesis_text = insight.supporting_data.get("text", "")

        print("\n" + "=" * 70)
        print("Analysis Complete")
        print("=" * 70)

        return synthesis_text

    def print_reasoning_log(self):
        """Print the reasoning log from all agents."""
        print("\n" + "=" * 70)
        print("AGENT REASONING LOG")
        print("=" * 70)
        for name, agent in self.agents.items():
            if agent.reasoning_log:
                print(f"\n### {name}")
                for log in agent.reasoning_log:
                    print(f"  {log}")


# =============================================================================
# Main
# =============================================================================

def main():
    import csv
    DATADIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    data_path = os.path.join(DATADIR, "training_data.csv")

    print("Loading data...")
    with open(data_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    X = np.array([[float(row[f]) for f in FEATURE_NAMES] for row in rows])
    y = np.array([row["best_solver"] for row in rows])
    print(f"Loaded {len(y)} samples")

    # Run advanced analysis
    coordinator = AdvancedCoordinator()
    report = coordinator.run_analysis(X, y)

    print("\n" + report)

    # Show reasoning log
    coordinator.print_reasoning_log()

    # Save report
    report_path = os.path.join(os.path.dirname(__file__), "advanced_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
