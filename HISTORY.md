# Development History

## Step 1: Scaffold — pyproject.toml, Makefile, module init files
- Created pyproject.toml with numpy/pytest dependencies
- Created Makefile with test/benchmark/clean targets
- Created .gitignore for outputs/ and __pycache__/
- Created all package __init__.py files

## Step 2: features/extract.py
- Implemented gradient, laplacian, chi, max_abs_gradient, zero_crossings, energy_content
- extract_all() returns full feature dict

## Step 3: solvers/base.py — abstract interface
- SolverBase ABC with solve(), make_grid(), chi() methods

## Step 4: solvers/fdm/implicit.py — Crank-Nicolson
- Handles r=0 singularity via L'Hôpital rule
- Thomas algorithm for tridiagonal system
- Neumann at r=0, Dirichlet T=0 at r=1

## Step 5: solvers/spectral/cosine.py — cosine expansion
- Pseudo-spectral with cos((k+0.5)πr) basis
- Semi-implicit time stepping

## Step 6: solvers/pinn/stub.py — optional PyTorch PINN
- Tiny MLP if torch available, NaN with warning otherwise

## Step 7: metrics/accuracy.py
- L2 and L∞ error vs reference solution

## Step 8: policy/select.py
- Score = error + λ * time, pick lowest

## Step 9-10: reports/generate.py
- CSV and markdown report generation

## Step 11: README.md, CLAUDE.md updates
- Full README with solver table, CLI usage, project structure
- CLAUDE.md updated with build commands and architecture overview

## Step 12: Benchmark run
- Ran `python -m app.run_benchmark --alpha 0.0 0.5 1.0`
- implicit_fdm wins for all alpha values (lowest L2 error)
- Spectral solver stable but less accurate (limited modes, Cartesian basis in cylindrical geometry)
- PINN stub returns NaN (no PyTorch installed)

## Step 13: docs/MANUAL.md — detailed user manual
- Reference solution generation process explained
- CLI arguments, output files, error definitions, PDE formulation

## Step 14: Claude Code skills & hooks
- `/run-benchmark`: Execute benchmark and report results
- `/add-solver`: Guided workflow for adding new solvers
- `/run-tests`: Run tests with auto-fix on failure
- `/analyze-results`: Detailed benchmark result analysis
- `/refine-reference`: Verify/improve reference solution accuracy
- PostToolUse hook: Auto syntax check on Python edits

## Step 15: Translate all file contents to English
- Converted MANUAL.md, HISTORY.md, and all skill files from Japanese to English

## Step 16: Vectorize solvers for performance
- ImplicitFDM: vectorized tridiagonal construction, precompute geometric factors, preallocate arrays (~2.6x speedup)
- CosineSpectral: matrix-vector forward/inverse transform, precompute decay/weights/norms, vectorized nonlinear flux (~23x speedup)
- Thomas algorithm: use np.empty instead of np.zeros, reduce redundant division

## Bug fixes
- Fixed spectral solver instability: switched from 1/(1+dt*lam) to exp(-lam*dt) decay
- Fixed zero_crossings test expectation (4 crossings, not 6)
