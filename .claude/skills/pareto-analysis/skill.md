---
name: pareto-analysis
description: Run Pareto analysis to visualize error vs time trade-offs for solvers
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Pareto Analysis

Run Pareto front analysis to visualize and understand trade-offs between accuracy (L2 error) and computation time for different solver configurations.

## Quick Commands

```bash
# Run full Pareto analysis on all solvers
python docs/analysis/pareto_analyzer.py

# Run as part of improvement cycle (includes bottleneck analysis)
python docs/analysis/method_improvement_cycle.py --phase pareto
```

## What is Pareto Analysis?

Pareto analysis identifies the **Pareto-optimal** configurations - those where you cannot improve one objective (error or time) without worsening the other.

```
    Error
      ^
      |   x dominated
      |     \
      |      o-o-o  Pareto front
      |           \
      |            o
      +--------------> Time
```

Points on the Pareto front are "non-dominated" - optimal trade-offs.

## Output

### Pareto Front Files

Location: `data/pareto_fronts/{solver}_{timestamp}.json`

Structure:
```json
{
  "solver_name": "implicit_fdm",
  "timestamp": "2024-01-01T00:00:00",
  "points": [
    {
      "solver": "implicit_fdm",
      "config": {"alpha": 0.5, "nr": 51, "dt": 0.001, "t_end": 0.1, "ic_type": "parabola"},
      "l2_error": 0.0001,
      "wall_time": 0.01,
      "pareto_rank": 0,
      "is_stable": true
    }
  ],
  "pareto_optimal": [...],
  "summary": {
    "total_points": 12,
    "stable_points": 12,
    "pareto_optimal_count": 3,
    "stability_rate": 100.0,
    "min_error": 5.43e-02,
    "max_error": 1.20e+00
  }
}
```

## Programmatic Usage

```python
from docs.analysis.pareto_analyzer import (
    ParetoAnalysisAgent,
    load_all_pareto_fronts,
    load_latest_pareto_front,
)
from app.run_benchmark import SOLVERS

# Run analysis
agent = ParetoAnalysisAgent(
    alpha_list=[0.0, 0.5, 1.0],
    nr_list=[31, 51, 71],
    dt_list=[0.002, 0.001, 0.0005],
)

# Analyze single solver
front = agent.analyze_solver(SOLVERS[0], verbose=True)
print(f"Pareto-optimal: {len(front.pareto_optimal)} points")

# Analyze all solvers
results = agent.analyze_all_solvers(SOLVERS)

# Load existing results
fronts = load_all_pareto_fronts()
for name, front in fronts.items():
    print(f"{name}: {front.summary['pareto_optimal_count']} optimal points")
```

## Custom Parameter Sweep

```python
# Fine-grained sweep for specific solver
agent = ParetoAnalysisAgent(
    alpha_list=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    nr_list=[21, 31, 41, 51, 61, 71, 81, 91, 101],
    dt_list=[0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001],
    t_end_list=[0.1],
    ic_types=["parabola"],
)
```

## Interpreting Results

### Stability Rate

- **100%**: Solver is unconditionally stable in swept range
- **<90%**: Solver has stability issues - check alpha/dt combinations
- **0%**: Solver fails for all configurations (check implementation)

### Pareto-Optimal Count

- **High count**: Good coverage of trade-off space
- **Low count**: Solver may be dominated by others, or narrow operating range

### Error Range

- **Wide range**: Sensitive to parameters, need careful tuning
- **Narrow range**: Robust but may lack flexibility

## Steps (If Running Manually)

1. Ask user for analysis scope (all solvers or specific)
2. Choose parameter ranges based on goal:
   - Quick analysis: 3 values per parameter
   - Full analysis: 5+ values per parameter
3. Run analysis
4. Review results and identify:
   - Which solvers have best Pareto fronts
   - Where stability issues occur
   - Optimal configurations for user's needs
5. Recommend specific configurations
