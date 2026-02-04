---
name: improvement-cycle
description: Run multi-agent method improvement cycle to identify bottlenecks and generate proposals
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Improvement Cycle

Run the multi-agent method improvement cycle to analyze solver performance, identify bottlenecks, and generate improvement proposals.

## Quick Commands

```bash
# Run 1 improvement cycle
python docs/analysis/method_improvement_cycle.py --cycles 1

# Run 3 cycles (recommended)
python docs/analysis/method_improvement_cycle.py --cycles 3

# Resume from previous state
python docs/analysis/method_improvement_cycle.py --resume

# Interactive mode (human review of proposals)
python docs/analysis/method_improvement_cycle.py --interactive --cycles 1

# Generate summary report only
python docs/analysis/method_improvement_cycle.py --report

# Fresh start (clear history)
python docs/analysis/method_improvement_cycle.py --fresh --cycles 3

# Analyze specific solvers only
python docs/analysis/method_improvement_cycle.py --solvers implicit_fdm,spectral_cosine
```

## Workflow (7 Phases)

1. **Pareto Analysis**: Run parameter sweeps, compute Pareto fronts (error vs time)
2. **Bottleneck Analysis**: Identify stability, accuracy, speed, coverage gaps
3. **Proposal Generation**: Generate parameter_tuning and algorithm_tweak proposals
4. **Multi-Agent Evaluation**: Evaluate from 4 perspectives (accuracy, speed, stability, complexity)
5. **Human Review**: (Optional) Interactive approval/rejection
6. **Implementation**: Apply approved proposals
7. **Report & Archive**: Generate markdown report, save state

## Output Files

- `data/pareto_fronts/`: Pareto front JSON files per solver
- `data/cycle_reports/`: Markdown reports per cycle
- `data/improvement_history.json`: Cycle history and proposals

## Interpreting Results

### Bottleneck Categories

| Category | Description | Typical Action |
|----------|-------------|----------------|
| stability | Solver fails at certain params | Reduce dt, adaptive stepping |
| accuracy_gap | Large error variation | Increase resolution |
| speed_gap | Large time variation | Optimize implementation |
| coverage_gap | Single solver dominates | Find solver niches |

### Proposal Types

| Type | Description | Auto-implementable |
|------|-------------|-------------------|
| parameter_tuning | Adjust dt/nr | Yes |
| algorithm_tweak | Modify algorithm | Sketch provided |
| new_solver | New implementation | Sketch provided |

## Steps (If User Wants Custom Analysis)

1. Ask user about scope (all solvers or specific ones)
2. Run improvement cycle with appropriate flags
3. Review generated report
4. If proposals approved, guide implementation
5. Re-run to verify improvements
