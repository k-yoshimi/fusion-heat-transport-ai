# Fusion Heat Transport PDE Benchmark

Benchmark suite for comparing numerical solvers of the 1D radial heat transport equation in fusion plasmas:

```
∂T/∂t = (1/r) ∂/∂r (r χ(|∂T/∂r|) ∂T/∂r)
```

where the nonlinear thermal diffusivity is:
- χ(|T'|) = (|T'| - 0.5)^α + 0.1  if |T'| > 0.5
- χ = 0.1  otherwise

**Boundary conditions:** Neumann ∂T/∂r = 0 at r=0, Dirichlet T=0 at r=1.

## Solvers

| Solver | Method | Notes |
|--------|--------|-------|
| `implicit_fdm` | Crank-Nicolson FDM | L'Hôpital at r=0, scipy banded solver |
| `spectral_cosine` | Cosine expansion | cos((k+0.5)πr) basis, operator splitting |
| `pinn_stub` | Physics-Informed NN | Requires PyTorch (optional) |

## Quick Start

```bash
pip install -e ".[dev]"
# Requires: numpy>=1.24, scipy>=1.10, pytest>=7.0

# Run tests
make test

# Run benchmark
make benchmark
# or
python -m app.run_benchmark --alpha 0.0 0.5 1.0
```

## CLI Options

### Benchmark mode

```
--alpha    Nonlinearity parameters (default: 0.0 0.5 1.0)
--nr       Radial grid points (default: 51)
--dt       Time step (default: 0.001)
--t_end    Final time (default: 0.1)
```

### ML solver selector

```bash
# Generate training data (~432 parameter combinations)
python -m app.run_benchmark --generate-data

# Train decision tree model
make train
# or: python -m policy.train --generate

# Use ML selector (runs only the predicted best solver)
python -m app.run_benchmark --use-ml-selector --alpha 1.5

# Incremental learning: run benchmark and update model with results
python -m app.run_benchmark --alpha 0.5 1.0 --update
```

Additional ML options:
```
--generate-data    Generate training data via parameter sweep
--use-ml-selector  Predict best solver with trained model, run only that one
--update           Append results to training data and retrain model
--model-path       Path to trained model (default: data/solver_model.npz)
--data-path        Path to training CSV (default: data/training_data.csv)
```

See [docs/MANUAL.md](docs/MANUAL.md) for detailed ML selector documentation.

## Solver Selection Policy

The best solver is selected by minimizing:

```
score = L2_error + λ × wall_time
```

where λ is configurable (default 0.1). The ML selector uses a numpy-only decision tree trained on 12 features (problem parameters + initial condition properties) to predict the best solver before running any computation. See `policy/select.py`.

## Hypothesis-Driven Analysis

An interactive experiment framework for systematic solver analysis:

```bash
# Run 3 hypothesis verification cycles
python docs/analysis/experiment_framework.py --cycles 3

# Interactive mode
python docs/analysis/experiment_framework.py --interactive

# Generate final report
python docs/analysis/experiment_framework.py --report
```

Features:
- Predefined experiments (stability_map, ic_comparison, linear_regime, fine_sweep)
- Hypothesis tracking with verification history
- Automatic report generation in Markdown
- Multi-agent parallel analysis

## Project Structure

```
app/                  CLI entrypoint
features/             PDE feature extraction (gradients, energy, etc.)
solvers/              Solver implementations (FDM, spectral, PINN)
metrics/              Error metrics (L2, L∞)
policy/               Solver selection policy + ML decision tree
reports/              CSV + markdown report generation
tests/                Unit tests
outputs/              Generated benchmark results (gitignored)
data/                 Training data + trained model (gitignored)
docs/
  ├── analysis/       Hypothesis-driven experiment tools
  │   ├── experiment_framework.py   # Main experiment framework
  │   ├── multi_agent_analysis.py   # Multi-agent analysis system
  │   ├── parallel_multi_agent.py   # Parallel execution
  │   └── verify_hypotheses.py      # Quick hypothesis verification
  ├── figures/        Generated figures for documentation
  └── TUTORIAL.md     Step-by-step guide
```

## Documentation

- [Tutorial](docs/TUTORIAL.md) — Step-by-step guide from first benchmark to ML selector
- [Manual](docs/MANUAL.md) — Detailed reference for all features
- [Analysis Tutorial](docs/ANALYSIS_TUTORIAL.md) — Hypothesis-driven experiment framework guide

## Reference Solution

The reference is computed using the implicit FDM solver with 4x grid refinement and 4x smaller time step.
