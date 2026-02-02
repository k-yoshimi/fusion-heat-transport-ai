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
