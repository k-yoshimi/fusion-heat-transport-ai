---
name: speedup-solver
description: Analyze and optimize solver performance for speed improvements
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Speedup Solver

Analyze a solver's performance and apply optimization techniques to improve computation speed.

## Steps

1. Ask the user which solver to optimize
2. Profile the solver to identify bottlenecks:
   ```bash
   python -c "
   import cProfile
   import pstats
   from solvers.fdm.implicit import ImplicitFDM  # Change to target solver
   import numpy as np

   solver = ImplicitFDM()
   r = np.linspace(0, 1, 101)
   T0 = 1 - r**2

   pr = cProfile.Profile()
   pr.enable()
   for _ in range(10):
       solver.solve(T0.copy(), r, 0.001, 0.1, 0.5)
   pr.disable()

   stats = pstats.Stats(pr)
   stats.sort_stats('cumulative')
   stats.print_stats(20)
   "
   ```
3. Read the solver implementation and identify optimization opportunities
4. Apply optimizations based on solver type

## Optimization Techniques by Solver Type

### FDM/FEM (Tridiagonal Systems)

**Before** (slow):
```python
# Solving Ax = b with numpy.linalg.solve
A = build_matrix(...)
T_new = np.linalg.solve(A, b)
```

**After** (fast):
```python
# Using banded solver (10-100x faster)
from scipy.linalg import solve_banded
# ab[0] = upper diagonal, ab[1] = main diagonal, ab[2] = lower diagonal
ab = np.zeros((3, n))
ab[0, 1:] = upper_diag
ab[1, :] = main_diag
ab[2, :-1] = lower_diag
T_new = solve_banded((1, 1), ab, b)
```

### Spectral Methods

**Optimization targets**:
- Pre-compute DCT/FFT matrices
- Use scipy.fft instead of numpy.fft
- Cache transform coefficients

### General Optimizations

1. **Pre-compute constants**: Move invariant calculations outside loops
2. **Vectorize**: Replace Python loops with numpy operations
3. **Avoid allocations**: Reuse arrays with `out=` parameter
4. **Numba JIT**: Add `@numba.jit(nopython=True)` to hot functions

## Verification

After optimization, verify:

1. **Correctness**: Run tests
   ```bash
   python -m pytest tests/test_solvers.py -v -k "target_solver"
   ```

2. **Speed improvement**: Time comparison
   ```bash
   python -c "
   import time
   from solvers.TARGET import TargetSolver
   import numpy as np

   solver = TargetSolver()
   r = np.linspace(0, 1, 101)
   T0 = 1 - r**2

   start = time.perf_counter()
   for _ in range(100):
       solver.solve(T0.copy(), r, 0.001, 0.1, 0.5)
   elapsed = time.perf_counter() - start
   print(f'Time per solve: {elapsed/100*1000:.2f} ms')
   "
   ```

3. **Accuracy preserved**: Run benchmark
   ```bash
   python -m app.run_benchmark --alpha 0.5
   ```

## Common Bottlenecks

| Symptom | Cause | Fix |
|---------|-------|-----|
| Slow matrix solve | Dense solver | Use banded/sparse solver |
| Slow chi computation | Python loop | Vectorize with numpy |
| Memory allocation | Array creation in loop | Pre-allocate, reuse |
| FFT overhead | Repeated planning | Cache FFT plans |

## Example: Implicit FDM Optimization

Read `solvers/fdm/implicit.py` for a reference implementation that uses:
- `scipy.linalg.solve_banded` for tridiagonal systems
- Pre-computed band structure
- Vectorized chi calculation
