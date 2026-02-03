"""Multi-agent system for solver selection analysis.

This module defines specialized agents that collaborate to analyze
benchmark data from different perspectives.

Architecture:
    Coordinator
        ├── StatisticsAgent    (statistical analysis)
        ├── FeatureAgent       (feature importance analysis)
        ├── PatternAgent       (pattern detection)
        ├── VisualizationAgent (create charts)
        └── ReportAgent        (synthesize findings)
"""

import os
import sys
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from policy.train import FEATURE_NAMES
from policy.tree import NumpyDecisionTree

DATADIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


# =============================================================================
# Base Agent Class
# =============================================================================

@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """Abstract base class for analysis agents."""

    def __init__(self, name: str):
        self.name = name
        self.messages_received: List[AgentMessage] = []
        self.messages_sent: List[AgentMessage] = []

    @abstractmethod
    def analyze(self, X: np.ndarray, y: np.ndarray, **kwargs) -> AgentMessage:
        """Perform analysis and return findings."""
        pass

    def receive(self, message: AgentMessage):
        """Receive a message from another agent."""
        self.messages_received.append(message)

    def send(self, content: Dict[str, Any], metadata: Optional[Dict] = None) -> AgentMessage:
        """Create and record an outgoing message."""
        msg = AgentMessage(sender=self.name, content=content, metadata=metadata)
        self.messages_sent.append(msg)
        return msg


# =============================================================================
# Specialized Agents
# =============================================================================

class StatisticsAgent(BaseAgent):
    """Analyzes statistical properties of the data."""

    def __init__(self):
        super().__init__("StatisticsAgent")

    def analyze(self, X: np.ndarray, y: np.ndarray, **kwargs) -> AgentMessage:
        feature_names = kwargs.get("feature_names", FEATURE_NAMES)

        # Solver distribution
        solver_counts = Counter(y)
        total = len(y)

        # Per-solver statistics
        solver_stats = {}
        for solver in np.unique(y):
            mask = y == solver
            solver_stats[solver] = {
                "count": int(np.sum(mask)),
                "percentage": float(np.sum(mask) / total * 100),
            }

        # Feature statistics by solver
        feature_stats = {}
        for i, fname in enumerate(feature_names):
            feature_stats[fname] = {
                "global_mean": float(np.mean(X[:, i])),
                "global_std": float(np.std(X[:, i])),
                "by_solver": {}
            }
            for solver in np.unique(y):
                mask = y == solver
                if np.sum(mask) > 0:
                    feature_stats[fname]["by_solver"][solver] = {
                        "mean": float(np.mean(X[mask, i])),
                        "std": float(np.std(X[mask, i])),
                    }

        # Correlation matrix
        corr_matrix = np.corrcoef(X.T)

        findings = {
            "solver_distribution": solver_stats,
            "feature_statistics": feature_stats,
            "correlation_matrix": corr_matrix.tolist(),
            "sample_count": total,
            "feature_count": len(feature_names),
        }

        return self.send(findings, metadata={"analysis_type": "statistical"})


class FeatureAgent(BaseAgent):
    """Analyzes feature importance and relationships."""

    def __init__(self):
        super().__init__("FeatureAgent")

    def analyze(self, X: np.ndarray, y: np.ndarray, **kwargs) -> AgentMessage:
        feature_names = kwargs.get("feature_names", FEATURE_NAMES)

        # Train decision tree
        tree = NumpyDecisionTree(max_depth=5)
        tree.fit(X, y)

        # Count split frequency
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

        # Rank features
        ranked_features = sorted(importance.items(), key=lambda x: -x[1])

        # Compute variance ratio for each feature (between-solver vs within-solver)
        variance_ratios = {}
        global_mean = np.mean(X, axis=0)
        for i, fname in enumerate(feature_names):
            between_var = 0
            within_var = 0
            for solver in np.unique(y):
                mask = y == solver
                n_k = np.sum(mask)
                mean_k = np.mean(X[mask, i])
                between_var += n_k * (mean_k - global_mean[i]) ** 2
                within_var += np.sum((X[mask, i] - mean_k) ** 2)
            variance_ratios[fname] = float(between_var / (within_var + 1e-10))

        findings = {
            "feature_importance": dict(ranked_features),
            "variance_ratios": variance_ratios,
            "top_features": [f[0] for f in ranked_features[:5]],
            "training_accuracy": float(np.mean(tree.predict(X) == y)),
        }

        return self.send(findings, metadata={"analysis_type": "feature_analysis"})


class PatternAgent(BaseAgent):
    """Detects patterns and decision rules."""

    def __init__(self):
        super().__init__("PatternAgent")

    def analyze(self, X: np.ndarray, y: np.ndarray, **kwargs) -> AgentMessage:
        feature_names = kwargs.get("feature_names", FEATURE_NAMES)

        # Train decision tree
        tree = NumpyDecisionTree(max_depth=5)
        tree.fit(X, y)

        # Extract decision rules
        rules = []

        def traverse(node, conditions, depth):
            if node is None:
                return
            if node.get("leaf"):
                if node.get("class") is not None:
                    rules.append({
                        "conditions": conditions.copy(),
                        "solver": node["class"],
                        "depth": depth,
                    })
                return

            feat_idx = node.get("feature")
            thresh = node.get("threshold")
            if feat_idx is not None:
                fname = feature_names[feat_idx]

                conditions.append(f"{fname} <= {thresh:.4f}")
                traverse(node.get("left"), conditions, depth + 1)
                conditions.pop()

                conditions.append(f"{fname} > {thresh:.4f}")
                traverse(node.get("right"), conditions, depth + 1)
                conditions.pop()

        traverse(tree.tree_, [], 0)

        # Identify winning conditions for minority solvers
        minority_rules = [r for r in rules if r["solver"] != "implicit_fdm"]

        # Find feature thresholds that separate solvers
        thresholds = {}
        for solver in np.unique(y):
            mask = y == solver
            if np.sum(mask) > 0:
                thresholds[solver] = {}
                for i, fname in enumerate(feature_names):
                    thresholds[solver][fname] = {
                        "min": float(np.min(X[mask, i])),
                        "max": float(np.max(X[mask, i])),
                    }

        findings = {
            "decision_rules": rules,
            "minority_rules": minority_rules,
            "solver_thresholds": thresholds,
            "rule_count": len(rules),
        }

        return self.send(findings, metadata={"analysis_type": "pattern_detection"})


class ReportAgent(BaseAgent):
    """Synthesizes findings from all agents into a coherent report."""

    def __init__(self):
        super().__init__("ReportAgent")

    def analyze(self, X: np.ndarray, y: np.ndarray, **kwargs) -> AgentMessage:
        """Generate report from received messages."""
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("MULTI-AGENT SOLVER SELECTION ANALYSIS REPORT")
        report_lines.append("=" * 70)

        # Process messages from other agents
        stats_msg = None
        feature_msg = None
        pattern_msg = None

        for msg in self.messages_received:
            if msg.metadata and msg.metadata.get("analysis_type") == "statistical":
                stats_msg = msg
            elif msg.metadata and msg.metadata.get("analysis_type") == "feature_analysis":
                feature_msg = msg
            elif msg.metadata and msg.metadata.get("analysis_type") == "pattern_detection":
                pattern_msg = msg

        # Section 1: Statistical Summary
        report_lines.append("\n## 1. Statistical Summary (from StatisticsAgent)")
        if stats_msg:
            dist = stats_msg.content.get("solver_distribution", {})
            for solver, info in dist.items():
                report_lines.append(f"   {solver}: {info['count']} samples ({info['percentage']:.1f}%)")
            report_lines.append(f"   Total samples: {stats_msg.content.get('sample_count')}")
            report_lines.append(f"   Features: {stats_msg.content.get('feature_count')}")

        # Section 2: Feature Analysis
        report_lines.append("\n## 2. Feature Analysis (from FeatureAgent)")
        if feature_msg:
            report_lines.append(f"   Training accuracy: {feature_msg.content.get('training_accuracy', 0):.1%}")
            report_lines.append("   Top 5 features by importance:")
            for i, fname in enumerate(feature_msg.content.get("top_features", [])[:5], 1):
                imp = feature_msg.content.get("feature_importance", {}).get(fname, 0)
                report_lines.append(f"      {i}. {fname}: {imp:.2f}")

        # Section 3: Pattern Detection
        report_lines.append("\n## 3. Decision Patterns (from PatternAgent)")
        if pattern_msg:
            rules = pattern_msg.content.get("decision_rules", [])
            report_lines.append(f"   Extracted {len(rules)} decision rules")

            minority = pattern_msg.content.get("minority_rules", [])
            if minority:
                report_lines.append("\n   Conditions for non-FDM solvers:")
                for rule in minority[:3]:
                    conds = " AND ".join(rule["conditions"])
                    report_lines.append(f"      IF {conds}")
                    report_lines.append(f"         THEN {rule['solver']}")

        # Section 4: Synthesis and Recommendations
        report_lines.append("\n## 4. Synthesis and Recommendations")
        report_lines.append("   Based on multi-agent analysis:")

        if stats_msg:
            dist = stats_msg.content.get("solver_distribution", {})
            if dist.get("implicit_fdm", {}).get("percentage", 0) > 95:
                report_lines.append("   - Implicit FDM dominates (>95% win rate)")
                report_lines.append("   - Consider using FDM as default without ML selection")

        if feature_msg:
            top = feature_msg.content.get("top_features", [])
            if top:
                report_lines.append(f"   - Key decision factor: {top[0]}")

        if pattern_msg:
            minority = pattern_msg.content.get("minority_rules", [])
            if len(minority) < 3:
                report_lines.append("   - Spectral solver only wins in edge cases")
                report_lines.append("   - Recommend improving spectral solver stability")

        report_lines.append("\n" + "=" * 70)

        report_text = "\n".join(report_lines)

        findings = {
            "report_text": report_text,
            "agents_consulted": len(self.messages_received),
        }

        return self.send(findings, metadata={"analysis_type": "synthesis"})


# =============================================================================
# Coordinator (Orchestrates the multi-agent workflow)
# =============================================================================

class Coordinator:
    """Orchestrates the multi-agent analysis workflow."""

    def __init__(self):
        self.agents = {
            "statistics": StatisticsAgent(),
            "features": FeatureAgent(),
            "patterns": PatternAgent(),
            "report": ReportAgent(),
        }
        self.messages: List[AgentMessage] = []

    def run_analysis(self, X: np.ndarray, y: np.ndarray,
                     feature_names: List[str] = FEATURE_NAMES) -> str:
        """Run the full multi-agent analysis pipeline."""
        print("=" * 60)
        print("Starting Multi-Agent Analysis")
        print("=" * 60)

        # Phase 1: Individual agent analysis
        print("\nPhase 1: Running specialized agents...")

        print("  [1/3] StatisticsAgent analyzing...")
        stats_msg = self.agents["statistics"].analyze(X, y, feature_names=feature_names)
        self.messages.append(stats_msg)

        print("  [2/3] FeatureAgent analyzing...")
        feature_msg = self.agents["features"].analyze(X, y, feature_names=feature_names)
        self.messages.append(feature_msg)

        print("  [3/3] PatternAgent analyzing...")
        pattern_msg = self.agents["patterns"].analyze(X, y, feature_names=feature_names)
        self.messages.append(pattern_msg)

        # Phase 2: Synthesis
        print("\nPhase 2: Synthesizing findings...")

        # Send all messages to ReportAgent
        for msg in self.messages:
            self.agents["report"].receive(msg)

        report_msg = self.agents["report"].analyze(X, y, feature_names=feature_names)
        self.messages.append(report_msg)

        print("\nPhase 3: Report generated")
        print("=" * 60)

        return report_msg.content.get("report_text", "")

    def get_agent_insights(self, agent_name: str) -> Optional[AgentMessage]:
        """Get the most recent message from a specific agent."""
        for msg in reversed(self.messages):
            if msg.sender == agent_name:
                return msg
        return None


# =============================================================================
# Main
# =============================================================================

def load_data(path: str):
    """Load training data from CSV."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    X = np.array([[float(row[f]) for f in FEATURE_NAMES] for row in rows])
    y = np.array([row["best_solver"] for row in rows])
    return X, y


def main():
    data_path = os.path.join(DATADIR, "training_data.csv")

    print("Loading data...")
    X, y = load_data(data_path)
    print(f"Loaded {len(y)} samples with {len(FEATURE_NAMES)} features")

    # Create and run coordinator
    coordinator = Coordinator()
    report = coordinator.run_analysis(X, y)

    print("\n" + report)

    # Save report
    report_path = os.path.join(os.path.dirname(__file__), "multi_agent_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Example: Access individual agent insights
    print("\n--- Individual Agent Insights ---")

    feature_insights = coordinator.get_agent_insights("FeatureAgent")
    if feature_insights:
        print(f"\nFeatureAgent top features: {feature_insights.content.get('top_features')}")

    pattern_insights = coordinator.get_agent_insights("PatternAgent")
    if pattern_insights:
        print(f"PatternAgent rules found: {pattern_insights.content.get('rule_count')}")


if __name__ == "__main__":
    main()
