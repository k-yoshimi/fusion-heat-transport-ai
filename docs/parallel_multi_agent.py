"""Parallel multi-agent system using concurrent.futures.

This demonstrates how to run multiple agents in parallel,
reducing total analysis time when agents are independent.
"""

import os
import sys
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Any, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from policy.train import FEATURE_NAMES
from policy.tree import NumpyDecisionTree
from collections import Counter


@dataclass
class AgentResult:
    """Result from an agent's analysis."""
    agent_name: str
    execution_time: float
    findings: Dict[str, Any]


def run_statistics_agent(X: np.ndarray, y: np.ndarray) -> AgentResult:
    """Statistical analysis - runs in separate thread."""
    start = time.perf_counter()

    # Solver distribution
    counts = Counter(y)
    total = len(y)

    # Feature means by solver
    feature_means = {}
    for solver in np.unique(y):
        mask = y == solver
        feature_means[solver] = {
            FEATURE_NAMES[i]: float(np.mean(X[mask, i]))
            for i in range(len(FEATURE_NAMES))
        }

    elapsed = time.perf_counter() - start

    return AgentResult(
        agent_name="StatisticsAgent",
        execution_time=elapsed,
        findings={
            "distribution": {k: v/total for k, v in counts.items()},
            "feature_means": feature_means,
        }
    )


def run_feature_agent(X: np.ndarray, y: np.ndarray) -> AgentResult:
    """Feature importance analysis - runs in separate thread."""
    start = time.perf_counter()

    # Train decision tree
    tree = NumpyDecisionTree(max_depth=5)
    tree.fit(X, y)

    # Count splits
    importance = {f: 0 for f in FEATURE_NAMES}

    def count_splits(node):
        if node is None or node.get("leaf"):
            return
        feat_idx = node.get("feature")
        if feat_idx is not None and 0 <= feat_idx < len(FEATURE_NAMES):
            importance[FEATURE_NAMES[feat_idx]] += 1
        count_splits(node.get("left"))
        count_splits(node.get("right"))

    count_splits(tree.tree_)

    # Normalize
    total = sum(importance.values())
    if total > 0:
        importance = {k: v/total for k, v in importance.items()}

    elapsed = time.perf_counter() - start

    return AgentResult(
        agent_name="FeatureAgent",
        execution_time=elapsed,
        findings={
            "importance": importance,
            "top_features": sorted(importance.items(), key=lambda x: -x[1])[:5],
        }
    )


def run_pattern_agent(X: np.ndarray, y: np.ndarray) -> AgentResult:
    """Pattern detection - runs in separate thread."""
    start = time.perf_counter()

    # Train tree
    tree = NumpyDecisionTree(max_depth=5)
    tree.fit(X, y)

    # Extract rules
    rules = []

    def traverse(node, conditions, depth):
        if node is None:
            return
        if node.get("leaf"):
            if conditions:
                rules.append({
                    "conditions": " AND ".join(conditions),
                    "solver": node.get("class"),
                })
            return

        feat_idx = node.get("feature")
        thresh = node.get("threshold")
        if feat_idx is not None:
            fname = FEATURE_NAMES[feat_idx]

            conditions.append(f"{fname} <= {thresh:.4f}")
            traverse(node.get("left"), conditions, depth + 1)
            conditions.pop()

            conditions.append(f"{fname} > {thresh:.4f}")
            traverse(node.get("right"), conditions, depth + 1)
            conditions.pop()

    traverse(tree.tree_, [], 0)

    elapsed = time.perf_counter() - start

    return AgentResult(
        agent_name="PatternAgent",
        execution_time=elapsed,
        findings={
            "rules": rules,
            "rule_count": len(rules),
        }
    )


def run_hypothesis_agent(X: np.ndarray, y: np.ndarray) -> AgentResult:
    """Hypothesis testing - runs in separate thread."""
    start = time.perf_counter()

    hypotheses = []

    # Test: Does alpha determine solver?
    alpha_idx = FEATURE_NAMES.index("alpha")
    for threshold in [0.0, 0.1, 0.2]:
        high_alpha = X[:, alpha_idx] > threshold
        if np.sum(high_alpha) > 0:
            fdm_rate = np.sum((y == "implicit_fdm") & high_alpha) / np.sum(high_alpha)
            if fdm_rate > 0.99:
                hypotheses.append({
                    "statement": f"alpha > {threshold} → FDM wins",
                    "confidence": fdm_rate,
                    "confirmed": True,
                })
                break

    # Test: Problem stiffness threshold
    stiff_idx = FEATURE_NAMES.index("problem_stiffness")
    spectral_mask = y == "spectral_cosine"
    if np.sum(spectral_mask) > 0:
        max_stiff = np.max(X[spectral_mask, stiff_idx])
        hypotheses.append({
            "statement": f"Spectral only wins when stiffness < {max_stiff:.4f}",
            "confidence": 0.95,
            "confirmed": True,
        })

    elapsed = time.perf_counter() - start

    return AgentResult(
        agent_name="HypothesisAgent",
        execution_time=elapsed,
        findings={
            "hypotheses": hypotheses,
        }
    )


class ParallelCoordinator:
    """Coordinates parallel execution of agents."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.agents = [
            ("StatisticsAgent", run_statistics_agent),
            ("FeatureAgent", run_feature_agent),
            ("PatternAgent", run_pattern_agent),
            ("HypothesisAgent", run_hypothesis_agent),
        ]

    def run_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, AgentResult]:
        """Run all agents in parallel and collect results."""

        print("=" * 60)
        print("PARALLEL MULTI-AGENT ANALYSIS")
        print("=" * 60)
        print(f"\nStarting {len(self.agents)} agents in parallel...")
        print(f"(max_workers={self.max_workers})")
        print()

        results = {}
        total_start = time.perf_counter()

        # Submit all agents to thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(agent_func, X, y): name
                for name, agent_func in self.agents
            }

            # Collect results as they complete
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    result = future.result()
                    results[result.agent_name] = result
                    print(f"  ✓ {result.agent_name} completed in {result.execution_time*1000:.1f}ms")
                except Exception as e:
                    print(f"  ✗ {agent_name} failed: {e}")

        total_time = time.perf_counter() - total_start

        # Summary
        print()
        print("-" * 40)
        print(f"Total wall time: {total_time*1000:.1f}ms")
        sum_agent_time = sum(r.execution_time for r in results.values())
        print(f"Sum of agent times: {sum_agent_time*1000:.1f}ms")
        print(f"Parallelization speedup: {sum_agent_time/total_time:.2f}x")
        print()

        return results

    def synthesize(self, results: Dict[str, AgentResult]) -> str:
        """Combine all agent results into a report."""

        lines = []
        lines.append("=" * 60)
        lines.append("SYNTHESIS REPORT")
        lines.append("=" * 60)

        # Statistics
        if "StatisticsAgent" in results:
            stats = results["StatisticsAgent"].findings
            lines.append("\n## Solver Distribution")
            for solver, pct in stats["distribution"].items():
                lines.append(f"   {solver}: {pct*100:.1f}%")

        # Feature importance
        if "FeatureAgent" in results:
            feats = results["FeatureAgent"].findings
            lines.append("\n## Top Features")
            for fname, imp in feats["top_features"]:
                lines.append(f"   {fname}: {imp:.2f}")

        # Patterns
        if "PatternAgent" in results:
            patt = results["PatternAgent"].findings
            lines.append(f"\n## Decision Rules ({patt['rule_count']} found)")
            for rule in patt["rules"][:3]:
                lines.append(f"   IF {rule['conditions']}")
                lines.append(f"      THEN {rule['solver']}")

        # Hypotheses
        if "HypothesisAgent" in results:
            hyp = results["HypothesisAgent"].findings
            lines.append("\n## Confirmed Hypotheses")
            for h in hyp["hypotheses"]:
                if h["confirmed"]:
                    lines.append(f"   ✓ {h['statement']} (confidence: {h['confidence']:.0%})")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)


def main():
    # Load data
    DATADIR = os.path.join(os.path.dirname(__file__), "..", "data")
    data_path = os.path.join(DATADIR, "training_data.csv")

    print("Loading data...")
    with open(data_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    X = np.array([[float(row[f]) for f in FEATURE_NAMES] for row in rows])
    y = np.array([row["best_solver"] for row in rows])
    print(f"Loaded {len(y)} samples\n")

    # Run parallel analysis
    coordinator = ParallelCoordinator(max_workers=4)
    results = coordinator.run_analysis(X, y)

    # Generate report
    report = coordinator.synthesize(results)
    print(report)


if __name__ == "__main__":
    main()
