# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fusion heat transport PDE benchmark — compares numerical solvers (FDM, FEM, FVM, spectral, PINN) for the 1D radial heat equation with nonlinear diffusivity χ(|T'|) = (|T'|-0.5)^α + 0.1 if |T'|>0.5, else χ = 0.1.

## Build & Test Commands

```bash
pip install -e ".[dev]"                    # Install with dev dependencies
make test                                   # Run all tests (22 tests)
python -m pytest tests/test_solvers.py -v  # Run single test file
python -m pytest tests/ -k "test_implicit" # Run tests matching pattern
make benchmark                              # Run benchmark (alpha 0.0, 0.5, 1.0)
python -m app.run_benchmark --alpha 1.5 --nr 101 --dt 0.0005  # Custom params
make clean                                  # Remove generated outputs
```

### ML Training Workflow

```bash
python -m app.run_benchmark --generate-data  # Generate training data (216 instances)
make train                                    # Train decision tree model
python -m app.run_benchmark --use-ml-selector --alpha 1.5  # Use ML to pick solver
python -m app.run_benchmark --alpha 0.5 --update           # Incremental learning
```

### Hypothesis-Driven Analysis

```bash
python docs/analysis/experiment_framework.py --cycles 3    # Run verification cycles
python docs/analysis/experiment_framework.py --interactive # Interactive mode
```

## Architecture

- **solvers/**: Each solver extends `SolverBase` (ABC) with `solve(T0, r, dt, t_end, alpha) -> T_history`
  - Solvers registered in `app/run_benchmark.py` `SOLVERS` list
- **features/extract.py**: Feature extraction (gradient, laplacian, chi, energy)
- **metrics/accuracy.py**: L2 (cylindrical-weighted) and L∞ error vs reference
- **policy/**: Solver selection — `select.py` (rule-based), `tree.py` (numpy decision tree), `train.py` (ML training)
- **reports/generate.py**: CSV + markdown output
- **app/run_benchmark.py**: CLI entrypoint

## Adding a New Solver

1. Create solver class extending `SolverBase` in appropriate `solvers/` subdirectory
2. Implement `solve(T0, r, dt, t_end, alpha) -> T_history` with shape `(nt+1, nr)`
3. Set class attribute `name` to unique identifier
4. Import and add instance to `SOLVERS` list in `app/run_benchmark.py`
5. Handle r=0 singularity via L'Hôpital rule: `(1/r)∂r(rχ∂T/∂r) → 2χ∂²T/∂r²`

## Key Design Decisions

- r=0 singularity handled via L'Hôpital: (1/r)∂r(rχ∂T/∂r) → 2χ∂²T/∂r²
- Reference solution: ImplicitFDM with 4x grid refinement and 4x smaller dt
- Performance: use `scipy.linalg.solve_banded` for tridiagonal systems, not Python loops
- No heavy deps beyond numpy/scipy; torch optional for PINN
