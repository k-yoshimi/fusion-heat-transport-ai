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

| Solver | Method | Time (ms) | L2 Error | Notes |
|--------|--------|-----------|----------|-------|
| `cell_centered_fvm` | Finite Volume | 1.9 | 0.058 | Fastest, conservative form |
| `implicit_fdm` | Crank-Nicolson FDM | 2.2 | 0.083 | Baseline, banded solver |
| `compact4_fdm` | 4th-order Compact FDM | 2.5 | 0.115 | Higher spatial accuracy |
| `imex_fdm` | IMEX FDM | 3.4 | 0.492 | Split operator approach |
| `chebyshev_spectral` | Chebyshev Spectral | 3.6 | 0.799 | Spectral accuracy |
| `p2_fem` | P2 Finite Element | 39.4 | 0.108 | Quadratic elements, high accuracy |
| `cosine_spectral` | Cosine Expansion | 2.5 | varies | Unstable for α > 0.5 |
| `pinn_stub` | Physics-Informed NN | - | - | Requires PyTorch (optional) |

*Performance measured at nr=51, dt=0.001, α=0.5, t_end=0.1*

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
solvers/              Solver implementations
  ├── fdm/            FDM solvers (implicit, compact4, imex)
  ├── spectral/       Spectral solvers (cosine, chebyshev)
  ├── fem/            Finite Element (P2)
  ├── fvm/            Finite Volume (cell-centered)
  └── pinn/           Physics-Informed Neural Networks
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

## Recent Optimizations

All solvers have been optimized for performance:

| Solver | Before | After | Speedup |
|--------|--------|-------|---------|
| Compact4 FDM | 25.3ms | 2.5ms | **10.3x** |
| Cell-Centered FVM | 14.5ms | 1.9ms | **7.6x** |
| P2 FEM | 285.0ms | 39.4ms | **7.2x** |
| Chebyshev Spectral | 8.1ms | 3.6ms | **2.3x** |

Key optimizations:
- Replaced `spsolve` with `solve_banded` for tridiagonal systems
- Vectorized matrix assembly (P2 FEM, Chebyshev)
- Precomputed geometric factors outside time loops
