---
name: add-solver
description: Add a new solver to the project
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Add Solver

Add a new numerical solver to the project.

## Steps

1. Ask the user for the solver type/method (e.g., explicit FDM, MOL, FEM, etc.)
2. Read `solvers/base.py` to review the `SolverBase` interface
3. Create an appropriate subdirectory (e.g., `solvers/fem/`)
4. Implement the new solver inheriting from `SolverBase`:
   - Set the `name` class attribute
   - Implement `solve(T0, r, dt, t_end, alpha) -> T_history`
   - Handle r=0 singularity via L'Hopital's rule
   - Apply Neumann BC (r=0) and Dirichlet BC (r=1)
5. Add tests in `tests/test_solvers.py`:
   - Basic operation test (temperature decreases via diffusion)
   - Boundary condition test
   - Nonlinear (alpha>0) test
6. Register the solver in `app/run_benchmark.py` `run()` function
7. Verify all tests pass with `python -m pytest tests/ -v`
8. Update HISTORY.md
