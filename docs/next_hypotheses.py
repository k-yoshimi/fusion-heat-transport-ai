"""Generate next hypotheses and verification methods.

Based on current analysis findings, this script:
1. Identifies gaps in understanding
2. Proposes new hypotheses
3. Suggests verification experiments
4. Prioritizes by expected impact
"""

import os
import sys
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from policy.train import FEATURE_NAMES
from collections import Counter


class Priority(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Hypothesis:
    """A hypothesis to be tested."""
    id: str
    statement: str
    rationale: str
    priority: Priority
    verification_method: str
    expected_outcome: str
    code_snippet: Optional[str] = None
    status: str = "proposed"  # proposed, testing, confirmed, rejected


@dataclass
class Gap:
    """An identified gap in understanding."""
    description: str
    importance: str
    related_hypotheses: List[str] = field(default_factory=list)


def load_current_findings():
    """Load and summarize current analysis findings."""
    DATADIR = os.path.join(os.path.dirname(__file__), "..", "data")
    data_path = os.path.join(DATADIR, "training_data.csv")

    with open(data_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    X = np.array([[float(row[f]) for f in FEATURE_NAMES] for row in rows])
    y = np.array([row["best_solver"] for row in rows])

    counts = Counter(y)
    total = len(y)

    findings = {
        "total_samples": total,
        "solver_distribution": {k: v/total for k, v in counts.items()},
        "fdm_dominance": counts.get("implicit_fdm", 0) / total,
        "spectral_wins": counts.get("spectral_cosine", 0),
        "alpha_range": (float(np.min(X[:, 0])), float(np.max(X[:, 0]))),
        "spectral_conditions": None,
    }

    # Find conditions where spectral wins
    spectral_mask = y == "spectral_cosine"
    if np.sum(spectral_mask) > 0:
        findings["spectral_conditions"] = {
            "alpha_max": float(np.max(X[spectral_mask, 0])),
            "stiffness_max": float(np.max(X[spectral_mask, FEATURE_NAMES.index("problem_stiffness")])),
        }

    return findings


def identify_gaps(findings: Dict) -> List[Gap]:
    """Identify gaps in current understanding."""
    gaps = []

    # Gap 1: Why does FDM dominate so strongly?
    if findings["fdm_dominance"] > 0.95:
        gaps.append(Gap(
            description="FDM wins 99.5% - Is this due to spectral instability or genuine superiority?",
            importance="HIGH",
            related_hypotheses=["H1", "H2", "H3"]
        ))

    # Gap 2: Limited initial condition
    gaps.append(Gap(
        description="Only one IC tested (T₀ = 1-r²). Results may not generalize.",
        importance="HIGH",
        related_hypotheses=["H4", "H5"]
    ))

    # Gap 3: Cost function sensitivity
    gaps.append(Gap(
        description="λ=0.1 fixed. Unknown how results change with different λ.",
        importance="MEDIUM",
        related_hypotheses=["H6"]
    ))

    # Gap 4: Spectral failure mode
    if findings["spectral_wins"] < 5:
        gaps.append(Gap(
            description="Spectral rarely wins. Unclear if it's error or instability.",
            importance="HIGH",
            related_hypotheses=["H2", "H7"]
        ))

    # Gap 5: Grid parameter sensitivity
    gaps.append(Gap(
        description="Limited grid parameter combinations (3×3×3). May miss optimal regions.",
        importance="MEDIUM",
        related_hypotheses=["H8"]
    ))

    return gaps


def generate_hypotheses() -> List[Hypothesis]:
    """Generate hypotheses based on current findings and gaps."""

    hypotheses = []

    # H1: Spectral instability hypothesis
    hypotheses.append(Hypothesis(
        id="H1",
        statement="Spectral solver fails due to numerical instability from threshold-based χ, not accuracy issues",
        rationale="The threshold χ formula creates discontinuities that spectral methods handle poorly",
        priority=Priority.HIGH,
        verification_method="""
1. Run spectral solver with very small dt (dt=0.0001)
2. Check if solution remains bounded
3. Compare error growth rate vs FDM
4. Analyze Fourier coefficients for spurious modes
""",
        expected_outcome="Spectral shows exponential error growth for α > 0.1",
        code_snippet="""
# Test spectral stability
from solvers.spectral.cosine import CosineSpectral
import numpy as np

r = np.linspace(0, 1, 101)
T0 = 1 - r**2
solver = CosineSpectral()

for dt in [0.001, 0.0005, 0.0001]:
    T_hist = solver.solve(T0.copy(), r, dt, t_end=0.1, alpha=0.5)
    max_T = np.max(np.abs(T_hist))
    print(f"dt={dt}: max|T|={max_T:.2e}, stable={max_T < 10}")
"""
    ))

    # H2: Spectral explicit step hypothesis
    hypotheses.append(Hypothesis(
        id="H2",
        statement="Spectral fails because nonlinear term is treated explicitly",
        rationale="CosineSpectral uses operator splitting with explicit nonlinear step",
        priority=Priority.HIGH,
        verification_method="""
1. Examine spectral solver code for time-stepping scheme
2. Implement semi-implicit or fully implicit variant
3. Compare stability regions
""",
        expected_outcome="Implicit treatment of χ improves spectral stability",
        code_snippet="""
# Check current spectral implementation
# solvers/spectral/cosine.py line ~70-90
# Look for explicit vs implicit treatment of chi term
"""
    ))

    # H3: FDM superiority for stiff problems
    hypotheses.append(Hypothesis(
        id="H3",
        statement="Crank-Nicolson (FDM) is unconditionally stable for any α and dt",
        rationale="Implicit methods are A-stable for parabolic PDEs",
        priority=Priority.MEDIUM,
        verification_method="""
1. Run FDM with very large dt (dt=0.01, 0.05)
2. Verify solution remains bounded
3. Compute stability factor vs spectral
""",
        expected_outcome="FDM stable for all tested dt, spectral fails for large dt",
        code_snippet="""
from solvers.fdm.implicit import ImplicitFDM
import numpy as np

r = np.linspace(0, 1, 51)
T0 = 1 - r**2
solver = ImplicitFDM()

for dt in [0.01, 0.02, 0.05]:
    T_hist = solver.solve(T0.copy(), r, dt, t_end=0.1, alpha=1.0)
    print(f"dt={dt}: max|T|={np.max(np.abs(T_hist)):.4f}")
"""
    ))

    # H4: Different IC hypothesis
    hypotheses.append(Hypothesis(
        id="H4",
        statement="With smoother IC (e.g., Gaussian), spectral may outperform FDM",
        rationale="Spectral methods excel for smooth solutions; T₀=1-r² has |T'|=2 at boundary",
        priority=Priority.HIGH,
        verification_method="""
1. Add new IC: T₀ = exp(-10r²) (Gaussian, already implemented before)
2. Add new IC: T₀ = cos(πr/2) (very smooth)
3. Regenerate training data with multiple ICs
4. Compare solver performance per IC
""",
        expected_outcome="Spectral wins more often for smooth ICs with small gradients",
        code_snippet="""
import numpy as np

def make_initial_gaussian(r):
    return np.exp(-10 * r**2)

def make_initial_cosine(r):
    return np.cos(np.pi * r / 2)

def make_initial_parabola(r):
    return 1 - r**2

# Compare max|dT/dr| for each IC
r = np.linspace(0, 1, 101)
dr = r[1] - r[0]

for name, ic_func in [("gaussian", make_initial_gaussian),
                       ("cosine", make_initial_cosine),
                       ("parabola", make_initial_parabola)]:
    T0 = ic_func(r)
    dTdr = np.gradient(T0, dr)
    print(f"{name}: max|dT/dr|={np.max(np.abs(dTdr)):.3f}")
"""
    ))

    # H5: Gradient threshold hypothesis
    hypotheses.append(Hypothesis(
        id="H5",
        statement="Spectral performs better when max|dT/dr| < 0.5 (below χ threshold)",
        rationale="Below threshold, χ=0.1 (constant) - linear problem favors spectral",
        priority=Priority.MEDIUM,
        verification_method="""
1. Create IC with small gradients: T₀ = 0.2*(1-r²)
2. Test both solvers in purely linear regime
3. Compare accuracy and speed
""",
        expected_outcome="Spectral wins when χ is constant (linear diffusion)",
        code_snippet="""
import numpy as np
from solvers.fdm.implicit import ImplicitFDM
from solvers.spectral.cosine import CosineSpectral

r = np.linspace(0, 1, 51)
T0 = 0.2 * (1 - r**2)  # max|dT/dr| = 0.4 < 0.5

# Run both solvers
fdm = ImplicitFDM()
spectral = CosineSpectral()

T_fdm = fdm.solve(T0.copy(), r, 0.001, 0.1, alpha=0.0)
T_spec = spectral.solve(T0.copy(), r, 0.001, 0.1, alpha=0.0)

print(f"FDM final T[0]: {T_fdm[-1, 0]:.6f}")
print(f"Spectral final T[0]: {T_spec[-1, 0]:.6f}")
"""
    ))

    # H6: Cost function sensitivity
    hypotheses.append(Hypothesis(
        id="H6",
        statement="With λ=1.0 (speed-focused), solver ranking may change",
        rationale="Current λ=0.1 heavily favors accuracy; higher λ may favor faster solver",
        priority=Priority.MEDIUM,
        verification_method="""
1. Regenerate results with λ = 0.0, 0.1, 0.5, 1.0, 2.0
2. Plot solver win rate vs λ
3. Find crossover point
""",
        expected_outcome="Higher λ may favor spectral (if faster) or reveal no change",
        code_snippet="""
# Modify policy/select.py or use select_best with different lam
from policy.select import select_best

results = [
    {"name": "fdm", "l2_error": 0.01, "wall_time": 0.005},
    {"name": "spectral", "l2_error": 0.02, "wall_time": 0.002},
]

for lam in [0.0, 0.1, 0.5, 1.0, 2.0]:
    best = select_best(results, lam=lam)
    print(f"λ={lam}: best={best['name']}")
"""
    ))

    # H7: Error vs instability
    hypotheses.append(Hypothesis(
        id="H7",
        statement="Spectral 'failures' are NaN/Inf (instability), not large but finite errors",
        rationale="Need to distinguish numerical blowup from poor accuracy",
        priority=Priority.HIGH,
        verification_method="""
1. Run spectral for all parameter combinations
2. Classify outcomes: stable+accurate, stable+inaccurate, unstable (NaN/Inf)
3. Map instability regions in parameter space
""",
        expected_outcome="Most spectral 'losses' are due to NaN, not finite large errors",
        code_snippet="""
import numpy as np
from solvers.spectral.cosine import CosineSpectral

r = np.linspace(0, 1, 51)
T0 = 1 - r**2
solver = CosineSpectral()

outcomes = {"stable": 0, "nan": 0, "inf": 0}

for alpha in [0.0, 0.5, 1.0, 1.5, 2.0]:
    T_hist = solver.solve(T0.copy(), r, 0.001, 0.1, alpha)
    if np.any(np.isnan(T_hist)):
        outcomes["nan"] += 1
    elif np.any(np.isinf(T_hist)):
        outcomes["inf"] += 1
    else:
        outcomes["stable"] += 1

print(outcomes)
"""
    ))

    # H8: Finer parameter sweep
    hypotheses.append(Hypothesis(
        id="H8",
        statement="Finer grid in (dt, nr) space reveals optimal spectral region",
        rationale="Current sweep is coarse (3×3); may miss sweet spots",
        priority=Priority.LOW,
        verification_method="""
1. Create finer sweep: nr ∈ [21, 31, 41, 51, 61, 71, 81]
2. dt ∈ [0.0001, 0.0002, 0.0005, 0.001, 0.002]
3. Generate heatmap of solver wins
""",
        expected_outcome="Find parameter region where spectral is competitive",
    ))

    # H9: Reference solution accuracy
    hypotheses.append(Hypothesis(
        id="H9",
        statement="4x refinement for reference may be insufficient for high α",
        rationale="Nonlinear problems may need finer reference for accurate evaluation",
        priority=Priority.MEDIUM,
        verification_method="""
1. Compute reference with 4x, 8x, 16x refinement
2. Check convergence of reference solution
3. Re-evaluate solver errors with refined reference
""",
        expected_outcome="Higher refinement may change relative solver rankings",
        code_snippet="""
import numpy as np
from solvers.fdm.implicit import ImplicitFDM

r = np.linspace(0, 1, 51)
T0 = 1 - r**2

for refine in [4, 8, 16]:
    nr_fine = refine * len(r) - (refine - 1)
    r_fine = np.linspace(0, 1, nr_fine)
    T0_fine = np.interp(r_fine, r, T0)
    dt_fine = 0.001 / refine

    solver = ImplicitFDM()
    T_ref = solver.solve(T0_fine, r_fine, dt_fine, 0.1, alpha=1.0)
    print(f"Refinement {refine}x: T_ref[0]={T_ref[-1, 0]:.6f}")
"""
    ))

    # H10: PINN potential
    hypotheses.append(Hypothesis(
        id="H10",
        statement="A properly trained PINN could outperform both solvers for high α",
        rationale="PINNs can learn complex nonlinear dynamics without explicit discretization",
        priority=Priority.LOW,
        verification_method="""
1. Implement full PINN (not stub) with PyTorch
2. Train on representative problems
3. Compare accuracy and inference time
""",
        expected_outcome="PINN may excel for strongly nonlinear cases (α > 1)",
    ))

    return hypotheses


def prioritize_experiments(hypotheses: List[Hypothesis]) -> List[Hypothesis]:
    """Sort hypotheses by priority and feasibility."""
    priority_order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
    return sorted(hypotheses, key=lambda h: priority_order[h.priority])


def generate_report(findings: Dict, gaps: List[Gap], hypotheses: List[Hypothesis]) -> str:
    """Generate a comprehensive report."""
    lines = []

    lines.append("=" * 70)
    lines.append("NEXT HYPOTHESES AND VERIFICATION METHODS")
    lines.append("=" * 70)

    # Current state summary
    lines.append("\n## Current State Summary\n")
    lines.append(f"- Total samples: {findings['total_samples']}")
    lines.append(f"- FDM dominance: {findings['fdm_dominance']*100:.1f}%")
    lines.append(f"- Spectral wins: {findings['spectral_wins']} cases")
    if findings['spectral_conditions']:
        lines.append(f"- Spectral win conditions: α ≤ {findings['spectral_conditions']['alpha_max']}, "
                    f"stiffness < {findings['spectral_conditions']['stiffness_max']:.4f}")

    # Gaps
    lines.append("\n## Identified Gaps\n")
    for i, gap in enumerate(gaps, 1):
        lines.append(f"{i}. [{gap.importance}] {gap.description}")
        if gap.related_hypotheses:
            lines.append(f"   Related: {', '.join(gap.related_hypotheses)}")

    # Hypotheses by priority
    lines.append("\n## Proposed Hypotheses\n")

    for priority in [Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
        priority_hyps = [h for h in hypotheses if h.priority == priority]
        if priority_hyps:
            lines.append(f"\n### {priority.value} Priority\n")
            for h in priority_hyps:
                lines.append(f"**[{h.id}] {h.statement}**")
                lines.append(f"")
                lines.append(f"Rationale: {h.rationale}")
                lines.append(f"")
                lines.append(f"Verification:")
                for line in h.verification_method.strip().split("\n"):
                    lines.append(f"  {line}")
                lines.append(f"")
                lines.append(f"Expected: {h.expected_outcome}")
                if h.code_snippet:
                    lines.append(f"")
                    lines.append(f"```python")
                    lines.append(h.code_snippet.strip())
                    lines.append(f"```")
                lines.append("")

    # Recommended next steps
    lines.append("\n## Recommended Execution Order\n")
    lines.append("1. **H7** - Classify spectral failures (NaN vs finite error)")
    lines.append("2. **H1** - Test spectral stability with small dt")
    lines.append("3. **H4** - Add new initial conditions")
    lines.append("4. **H5** - Test below-threshold regime (linear)")
    lines.append("5. **H6** - Vary cost function weight λ")
    lines.append("6. **H2** - Investigate implicit spectral variant")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


def main():
    print("Loading current findings...")
    findings = load_current_findings()

    print("Identifying gaps...")
    gaps = identify_gaps(findings)

    print("Generating hypotheses...")
    hypotheses = generate_hypotheses()
    hypotheses = prioritize_experiments(hypotheses)

    print("Generating report...\n")
    report = generate_report(findings, gaps, hypotheses)

    print(report)

    # Save report
    report_path = os.path.join(os.path.dirname(__file__), "next_hypotheses_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Also save as JSON for programmatic access
    import json
    json_path = os.path.join(os.path.dirname(__file__), "hypotheses.json")
    with open(json_path, "w") as f:
        json.dump({
            "findings": {k: v for k, v in findings.items() if not isinstance(v, np.ndarray)},
            "gaps": [{"description": g.description, "importance": g.importance} for g in gaps],
            "hypotheses": [{
                "id": h.id,
                "statement": h.statement,
                "priority": h.priority.value,
                "verification_method": h.verification_method,
            } for h in hypotheses],
        }, f, indent=2)
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
