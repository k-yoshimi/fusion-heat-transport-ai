"""Analyze feature-cost function relationships for solver selection.

This script generates training data, analyzes which features predict
solver selection, and creates visualizations for understanding when
each solver is optimal.
"""

import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

from policy.train import generate_training_data, train_model, FEATURE_NAMES
from policy.tree import NumpyDecisionTree

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
DATADIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_training_data(path):
    """Load training data from CSV."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    X = np.array([[float(row[f]) for f in FEATURE_NAMES] for row in rows])
    y = np.array([row["best_solver"] for row in rows])
    return X, y, rows


def analyze_solver_distribution(y):
    """Analyze which solver wins most often."""
    counts = Counter(y)
    total = len(y)
    print("\n=== Solver Win Distribution ===")
    for solver, count in counts.most_common():
        print(f"  {solver}: {count}/{total} ({100*count/total:.1f}%)")
    return counts


def analyze_feature_by_solver(X, y, feature_names):
    """Compute feature statistics grouped by winning solver."""
    solvers = np.unique(y)
    stats = {}

    print("\n=== Feature Statistics by Winning Solver ===")
    for i, fname in enumerate(feature_names):
        stats[fname] = {}
        print(f"\n{fname}:")
        for solver in solvers:
            mask = y == solver
            vals = X[mask, i]
            if len(vals) > 0:
                stats[fname][solver] = {
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "min": np.min(vals),
                    "max": np.max(vals),
                }
                print(f"  {solver}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                      f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")
    return stats


def compute_feature_importance(tree, feature_names):
    """Compute feature importance based on split frequency in decision tree."""
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

    return importance


def extract_decision_rules(tree, feature_names, max_depth=3):
    """Extract human-readable decision rules from tree."""
    rules = []

    def traverse(node, conditions, depth):
        if node is None:
            return
        if node.get("leaf") or depth >= max_depth:
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

            # Left branch: feature <= threshold
            conditions.append(f"{fname} <= {thresh:.4f}")
            traverse(node.get("left"), conditions, depth + 1)
            conditions.pop()

            # Right branch: feature > threshold
            conditions.append(f"{fname} > {thresh:.4f}")
            traverse(node.get("right"), conditions, depth + 1)
            conditions.pop()

    traverse(tree.tree_, [], 0)
    return rules


def plot_solver_distribution(counts, output_path):
    """Bar chart of solver win counts."""
    fig, ax = plt.subplots(figsize=(8, 5))

    solvers = list(counts.keys())
    values = [counts[s] for s in solvers]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    bars = ax.bar(solvers, values, color=colors[:len(solvers)])
    ax.set_ylabel("Number of Wins")
    ax.set_title("Solver Win Distribution (score = L2_error + 0.1 × wall_time)")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=12)

    ax.set_ylim(0, max(values) * 1.15)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_feature_importance(importance, output_path):
    """Horizontal bar chart of feature importance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by importance
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color="#4a90d9")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Relative Importance (split frequency)")
    ax.set_title("Feature Importance in Decision Tree")

    # Add value labels
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_feature_scatter(X, y, feature_names, feat1_idx, feat2_idx, output_path):
    """2D scatter plot colored by winning solver."""
    fig, ax = plt.subplots(figsize=(8, 6))

    solver_colors = {
        "implicit_fdm": "#1f77b4",
        "spectral_cosine": "#ff7f0e",
        "pinn_stub": "#2ca02c",
    }

    for solver, color in solver_colors.items():
        mask = y == solver
        if np.any(mask):
            ax.scatter(X[mask, feat1_idx], X[mask, feat2_idx],
                      c=color, label=solver, alpha=0.6, edgecolors="white", s=50)

    ax.set_xlabel(feature_names[feat1_idx])
    ax.set_ylabel(feature_names[feat2_idx])
    ax.set_title(f"Solver Selection: {feature_names[feat1_idx]} vs {feature_names[feat2_idx]}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_alpha_analysis(X, y, feature_names, output_path):
    """Analyze solver selection vs alpha parameter."""
    alpha_idx = feature_names.index("alpha")

    # Get unique alpha values
    alphas = np.unique(X[:, alpha_idx])

    solver_colors = {
        "implicit_fdm": "#1f77b4",
        "spectral_cosine": "#ff7f0e",
        "pinn_stub": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    # Count wins per alpha
    solvers = list(solver_colors.keys())
    width = 0.25
    x = np.arange(len(alphas))

    for i, solver in enumerate(solvers):
        counts = []
        for alpha in alphas:
            mask = (X[:, alpha_idx] == alpha) & (y == solver)
            counts.append(np.sum(mask))
        if max(counts) > 0:
            ax.bar(x + i * width, counts, width, label=solver, color=solver_colors[solver])

    ax.set_xlabel("Alpha (nonlinearity parameter)")
    ax.set_ylabel("Number of Wins")
    ax.set_title("Solver Performance vs Alpha")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_dt_nr_heatmap(X, y, feature_names, output_path):
    """Heatmap showing solver wins by dt and nr."""
    dt_idx = feature_names.index("dt")
    nr_idx = feature_names.index("nr")

    dts = np.unique(X[:, dt_idx])
    nrs = np.unique(X[:, nr_idx])

    # Count implicit_fdm wins as fraction
    heatmap = np.zeros((len(dts), len(nrs)))

    for i, dt in enumerate(dts):
        for j, nr in enumerate(nrs):
            mask = (X[:, dt_idx] == dt) & (X[:, nr_idx] == nr)
            total = np.sum(mask)
            if total > 0:
                fdm_wins = np.sum((y == "implicit_fdm") & mask)
                heatmap[i, j] = fdm_wins / total

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap, cmap="RdYlBu", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(nrs)))
    ax.set_yticks(np.arange(len(dts)))
    ax.set_xticklabels([f"{int(nr)}" for nr in nrs])
    ax.set_yticklabels([f"{dt:.4f}" for dt in dts])
    ax.set_xlabel("nr (grid points)")
    ax.set_ylabel("dt (time step)")
    ax.set_title("Implicit FDM Win Rate (blue=100%, red=0%)")

    # Add text annotations
    for i in range(len(dts)):
        for j in range(len(nrs)):
            val = heatmap[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", color=color, fontsize=11)

    fig.colorbar(im, ax=ax, label="Win Rate")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_cost_breakdown(rows, output_path):
    """Show L2 error vs wall time trade-off."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    solver_colors = {
        "implicit_fdm": "#1f77b4",
        "spectral_cosine": "#ff7f0e",
        "pinn_stub": "#2ca02c",
    }

    # We need to get actual L2 errors and wall times from benchmark runs
    # For now, show alpha vs problem_stiffness colored by winner
    ax = axes[0]
    for row in rows:
        solver = row["best_solver"]
        alpha = float(row["alpha"])
        stiffness = float(row["problem_stiffness"])
        ax.scatter(alpha, stiffness, c=solver_colors.get(solver, "gray"),
                  alpha=0.5, s=30)

    ax.set_xlabel("Alpha")
    ax.set_ylabel("Problem Stiffness (alpha × max_gradient)")
    ax.set_title("Solver Selection in Parameter Space")

    # Legend
    for solver, color in solver_colors.items():
        ax.scatter([], [], c=color, label=solver)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right plot: chi_ratio vs gradient_sharpness
    ax = axes[1]
    for row in rows:
        solver = row["best_solver"]
        chi_ratio = float(row["chi_ratio"])
        grad_sharp = float(row["gradient_sharpness"])
        ax.scatter(grad_sharp, chi_ratio, c=solver_colors.get(solver, "gray"),
                  alpha=0.5, s=30)

    ax.set_xlabel("Gradient Sharpness")
    ax.set_ylabel("Chi Ratio (max_chi / min_chi)")
    ax.set_title("Solver Selection by Diffusivity Properties")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_analysis_report(counts, importance, rules, stats):
    """Generate text report summarizing the analysis."""
    report = []
    report.append("=" * 60)
    report.append("SOLVER SELECTION ANALYSIS REPORT")
    report.append("=" * 60)

    report.append("\n## 1. Solver Win Distribution")
    total = sum(counts.values())
    for solver, count in counts.most_common():
        report.append(f"   {solver}: {count}/{total} ({100*count/total:.1f}%)")

    report.append("\n## 2. Feature Importance (Decision Tree Splits)")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for fname, imp in sorted_imp[:5]:
        report.append(f"   {fname}: {imp:.2f}")

    report.append("\n## 3. Key Decision Rules")
    for rule in rules[:5]:
        conds = " AND ".join(rule["conditions"])
        report.append(f"   IF {conds}")
        report.append(f"      THEN use {rule['solver']}")

    report.append("\n## 4. Insights")
    # Add insights based on the data
    if "implicit_fdm" in counts:
        fdm_pct = 100 * counts["implicit_fdm"] / total
        report.append(f"   - Implicit FDM wins {fdm_pct:.1f}% of cases")

    if sorted_imp[0][1] > 0.3:
        report.append(f"   - {sorted_imp[0][0]} is the most important feature")

    report.append("\n" + "=" * 60)

    return "\n".join(report)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(DATADIR, exist_ok=True)

    data_path = os.path.join(DATADIR, "training_data.csv")
    model_path = os.path.join(DATADIR, "solver_model.npz")

    # Step 1: Generate fresh training data
    print("=" * 60)
    print("Step 1: Generating training data...")
    print("=" * 60)
    if not os.path.exists(data_path):
        generate_training_data(data_path)
    else:
        print(f"  Using existing data: {data_path}")

    # Step 2: Load data
    print("\n" + "=" * 60)
    print("Step 2: Loading and analyzing data...")
    print("=" * 60)
    X, y, rows = load_training_data(data_path)
    print(f"  Loaded {len(y)} samples with {len(FEATURE_NAMES)} features")

    # Step 3: Basic analysis
    counts = analyze_solver_distribution(y)
    stats = analyze_feature_by_solver(X, y, FEATURE_NAMES)

    # Step 4: Train decision tree and analyze
    print("\n" + "=" * 60)
    print("Step 3: Training decision tree...")
    print("=" * 60)
    tree = NumpyDecisionTree(max_depth=5)
    tree.fit(X, y)
    preds = tree.predict(X)
    acc = np.mean(preds == y)
    print(f"  Training accuracy: {acc:.1%}")
    tree.save(model_path)

    # Feature importance
    importance = compute_feature_importance(tree, FEATURE_NAMES)
    print("\n  Feature Importance:")
    for fname, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
        print(f"    {fname}: {imp:.2f}")

    # Decision rules
    rules = extract_decision_rules(tree, FEATURE_NAMES, max_depth=3)
    print(f"\n  Extracted {len(rules)} decision rules")

    # Step 5: Generate visualizations
    print("\n" + "=" * 60)
    print("Step 4: Generating visualizations...")
    print("=" * 60)

    plot_solver_distribution(counts, os.path.join(FIGDIR, "solver_distribution.png"))
    plot_feature_importance(importance, os.path.join(FIGDIR, "feature_importance.png"))
    plot_alpha_analysis(X, y, FEATURE_NAMES, os.path.join(FIGDIR, "alpha_analysis.png"))
    plot_dt_nr_heatmap(X, y, FEATURE_NAMES, os.path.join(FIGDIR, "dt_nr_heatmap.png"))
    plot_cost_breakdown(rows, os.path.join(FIGDIR, "cost_breakdown.png"))

    # Feature scatter plots for key feature pairs
    alpha_idx = FEATURE_NAMES.index("alpha")
    stiff_idx = FEATURE_NAMES.index("problem_stiffness")
    plot_feature_scatter(X, y, FEATURE_NAMES, alpha_idx, stiff_idx,
                        os.path.join(FIGDIR, "alpha_vs_stiffness.png"))

    # Step 6: Generate report
    print("\n" + "=" * 60)
    print("Step 5: Generating report...")
    print("=" * 60)
    report = generate_analysis_report(counts, importance, rules, stats)
    print(report)

    report_path = os.path.join(os.path.dirname(__file__), "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved to: {report_path}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return counts, importance, rules, stats, X, y


if __name__ == "__main__":
    main()
