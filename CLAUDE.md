# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fusion heat transport PDE benchmark — compares numerical solvers (FDM, spectral, PINN) for the 1D radial heat equation with nonlinear diffusivity χ(|T'|) = (|T'|-0.5)^α + 0.1 if |T'|>0.5, else χ = 0.1.

## Build & Test Commands

- `make test` — run all tests with pytest
- `make benchmark` — run benchmark with default alpha values
- `python -m app.run_benchmark --alpha 0.0 0.5 1.0` — CLI with custom params
- `make clean` — remove generated outputs

## Architecture

- **solvers/**: Each solver extends `SolverBase` (ABC) with `solve(T0, r, dt, t_end, alpha) -> T_history`
- **features/extract.py**: Feature extraction from temperature profiles
- **metrics/accuracy.py**: L2/L∞ error vs reference solution
- **policy/select.py**: Solver selection (score = error + λ*time)
- **reports/generate.py**: CSV + markdown output
- **app/run_benchmark.py**: CLI entrypoint

## Key Design Decisions

- r=0 singularity handled via L'Hôpital: (1/r)∂r(rχ∂T/∂r) → 2χ∂²T/∂r²
- Reference solution: same FDM with 4x refinement
- No heavy deps beyond numpy; torch optional for PINN
