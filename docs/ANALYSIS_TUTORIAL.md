# Analysis Tutorial: Hypothesis-Driven Experimentation

This tutorial covers the experiment framework for systematic solver analysis through hypothesis-driven iteration.

## Overview

The framework provides:
1. **Predefined experiments** - Parameter sweeps for stability, IC comparison, etc.
2. **Hypothesis tracking** - Register, test, and track hypotheses over time
3. **Automated verification cycles** - Run experiments → test hypotheses → generate reports
4. **Multi-agent analysis** - Parallel data analysis with specialized agents

---

## Part 1: Quick Start

### 1.1 Run a simple experiment

```bash
# Run stability mapping experiment
python docs/analysis/experiment_framework.py -r stability_map
```

This runs 80 solver configurations across different (α, dt) combinations and stores results in `data/experiments.csv`.

### 1.2 Analyze results

```bash
python docs/analysis/experiment_framework.py --analyze
```

Output shows:
- Solver performance summary
- Stability rates by α value
- Winner distribution

### 1.3 Test a hypothesis

```bash
python docs/analysis/experiment_framework.py --test H1
```

Tests hypothesis H1: "Smaller dt improves spectral solver stability"

---

## Part 2: Predefined Experiments

| Experiment | Description | Runs |
|------------|-------------|------|
| `stability_map` | Map spectral stability across (α, dt) space | 80 |
| `ic_comparison` | Compare solvers with different initial conditions | 24 |
| `linear_regime` | Test in purely linear regime (scaled IC) | 12 |
| `fine_sweep` | Comprehensive parameter sweep | 810 |

Run an experiment:

```bash
python docs/analysis/experiment_framework.py -r stability_map
python docs/analysis/experiment_framework.py -r ic_comparison
```

---

## Part 3: Hypothesis Management

### 3.1 Default hypotheses

The framework tracks these hypotheses by default:

| ID | Hypothesis |
|----|------------|
| H1 | Smaller dt improves spectral solver stability |
| H3 | FDM is unconditionally stable for any dt |
| H4 | Different initial conditions lead to different optimal solvers |
| H5 | In linear regime (|dT/dr| < 0.5), both solvers perform equally well |
| H6 | Cost function parameter λ > 5 favors spectral solver |
| H7 | Spectral solver fails with NaN for α >= 0.2 |

### 3.2 Interactive hypothesis management

```bash
python docs/analysis/experiment_framework.py --interactive
```

Commands:
```
hypo              # List all hypotheses
hypo add H8 "Your hypothesis statement"
hypo note H1 "Observation: confirmed with dt=0.0001"
hypo status H1 confirmed
```

### 3.3 Hypothesis memo storage

Hypotheses are stored in `data/hypotheses_memo.json`:

```json
{
  "H1": {
    "hypothesis_id": "H1",
    "statement": "Smaller dt improves spectral solver stability",
    "status": "confirmed",
    "confidence": 0.4,
    "notes": ["[2026-02-03] Tested with stability_map"],
    "verification_history": [...]
  }
}
```

---

## Part 4: Verification Cycles

### 4.1 Run multiple cycles

```bash
# Run 3 verification cycles (appending data)
python docs/analysis/experiment_framework.py --cycles 3

# Fresh mode: regenerate database each cycle (more accurate)
python docs/analysis/experiment_framework.py --cycles 3 --fresh
```

Each cycle:
1. Runs the stability_map experiment
2. Tests all registered hypotheses against new data
3. Updates hypothesis status and confidence
4. Prints cycle summary

### 4.2 Fresh mode vs Append mode

| Mode | Command | Data handling | Use case |
|------|---------|---------------|----------|
| Append | `--cycles 3` | Accumulates data | Fast, but may have historical bias |
| Fresh | `--cycles 3 --fresh` | Regenerates each cycle | More accurate, independent verification |

Fresh mode is recommended for:
- Final verification before publishing results
- Checking reproducibility of findings
- Avoiding bias from accumulated historical data

### 4.2 Confidence scoring

- Each confirmed verification adds +20% confidence
- Each failed verification subtracts -10% confidence
- Status changes: untested → confirmed/rejected/inconclusive

### 4.3 Generate final report

```bash
python docs/analysis/experiment_framework.py --report
```

Generates `data/experiment_report.md` with:
- Executive summary
- Solver performance tables
- All hypothesis results with history
- Conclusions and recommended next steps

---

## Part 5: Interactive Mode

```bash
python docs/analysis/experiment_framework.py -i
```

Full command reference:

| Command | Description |
|---------|-------------|
| `list` | List predefined experiments |
| `run <name>` | Run experiment (stability_map, ic_comparison, etc.) |
| `run ic` | Run IC comparison across 4 initial conditions |
| `analyze` | Analyze all results |
| `analyze <exp>` | Analyze specific experiment |
| `test H1` | Test a specific hypothesis |
| `cycle 5` | Run 5 verification cycles |
| `custom` | Create custom experiment interactively |
| `hypo` | List all hypotheses |
| `hypo add <ID> <text>` | Add new hypothesis |
| `hypo note <ID> <text>` | Add note to hypothesis |
| `report` | Generate markdown report |
| `clear` | Clear experiment database |
| `quit` | Exit |

---

## Part 6: Multi-Agent Analysis

### 6.1 Basic multi-agent system

```bash
python docs/analysis/multi_agent_analysis.py
```

Runs 4 specialized agents:
- **StatisticsAgent**: Solver distribution, feature means
- **FeatureAgent**: Feature importance via decision tree
- **PatternAgent**: Extract decision rules
- **ReportAgent**: Generate synthesis report

### 6.2 Parallel execution

```bash
python docs/analysis/parallel_multi_agent.py
```

Runs agents in parallel using ThreadPoolExecutor for ~2x speedup.

### 6.3 Advanced analysis

```bash
python docs/analysis/advanced_multi_agent.py
```

Includes:
- **HypothesisAgent**: Generate testable hypotheses
- **CriticAgent**: Challenge findings with edge cases
- **SynthesisAgent**: Create final recommendations

---

## Part 7: Custom Experiments

### 7.1 Define a custom experiment

In interactive mode:

```
> custom
  Name: my_experiment
  Description: Test high alpha values
  Alpha values (comma-sep): 1.0,1.5,2.0,3.0
  dt values (comma-sep): 0.001,0.0005
```

### 7.2 Programmatic usage

```python
from docs.analysis.experiment_framework import ExperimentConfig, ExperimentRunner

config = ExperimentConfig(
    name="custom_sweep",
    description="My custom parameter sweep",
    alpha_list=[0.0, 0.5, 1.0, 2.0],
    nr_list=[51, 101],
    dt_list=[0.001, 0.0005],
    t_end_list=[0.1],
    ic_type="parabola",
    lambda_cost=0.1,
)

runner = ExperimentRunner()
results = runner.run_experiment(config)
```

---

## Part 8: Database Schema

### experiments.csv columns

| Column | Type | Description |
|--------|------|-------------|
| experiment_name | str | Name of the experiment |
| timestamp | str | ISO format timestamp |
| alpha | float | Nonlinearity parameter |
| nr | int | Number of grid points |
| dt | float | Time step |
| t_end | float | Final time |
| ic_type | str | Initial condition type |
| ic_scale | float | IC scaling factor |
| solver | str | Solver name |
| l2_error | float | L2 error vs reference |
| linf_error | float | L∞ error |
| wall_time | float | Execution time (seconds) |
| max_T | float | Maximum temperature |
| min_T | float | Minimum temperature |
| is_stable | bool | Whether solution is stable |
| is_nan | bool | Whether NaN occurred |
| score | float | L2 + λ × time |

---

## Summary of Commands

| Command | Description |
|---------|-------------|
| `python docs/analysis/experiment_framework.py` | Run quick demo |
| `python docs/analysis/experiment_framework.py -i` | Interactive mode |
| `python docs/analysis/experiment_framework.py -r stability_map` | Run experiment |
| `python docs/analysis/experiment_framework.py --cycles 3` | Run 3 cycles |
| `python docs/analysis/experiment_framework.py --analyze` | Analyze data |
| `python docs/analysis/experiment_framework.py --test H1` | Test hypothesis |
| `python docs/analysis/experiment_framework.py --report` | Generate report |
| `python docs/analysis/multi_agent_analysis.py` | Multi-agent analysis |
| `python docs/analysis/parallel_multi_agent.py` | Parallel analysis |
| `python docs/analysis/verify_hypotheses.py` | Quick verification |

---

## Example Workflow

```bash
# Day 1: Initial exploration
python docs/analysis/experiment_framework.py -r stability_map
python docs/analysis/experiment_framework.py --analyze

# Day 2: Hypothesis verification
python docs/analysis/experiment_framework.py --cycles 3
python docs/analysis/experiment_framework.py --report

# Day 3: Deep analysis
python docs/analysis/multi_agent_analysis.py
python docs/analysis/advanced_multi_agent.py

# Review findings
cat data/experiment_report.md
```
