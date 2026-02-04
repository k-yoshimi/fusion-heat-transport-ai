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
   - Use scipy.linalg.solve_banded for tridiagonal systems (performance)
5. Add tests in `tests/test_solvers.py`:
   - Basic operation test (temperature decreases via diffusion)
   - Boundary condition test
   - Nonlinear (alpha>0) test
6. Register the solver in `app/run_benchmark.py` SOLVERS list
7. Add stability constraints in `policy/stability.py`:
   - Determine if unconditionally stable or has CFL condition
   - Set temporal_order, spatial_order
   - Set alpha_max if there are nonlinearity limits
8. Verify all tests pass with `python -m pytest tests/ -v`
9. Run quick benchmark: `python -m app.run_benchmark --alpha 0.5`
10. Update HISTORY.md

## Implementation Checklist

- [ ] Inherits from SolverBase
- [ ] name attribute set (lowercase, underscore-separated)
- [ ] solve() returns shape (nt+1, nr)
- [ ] r=0 singularity handled (L'Hopital)
- [ ] BC: T'(0)=0, T(1)=0
- [ ] Nonlinear chi computed correctly
- [ ] Registered in SOLVERS list
- [ ] Stability info in policy/stability.py
- [ ] Tests pass
