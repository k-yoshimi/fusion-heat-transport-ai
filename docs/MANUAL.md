# User Manual

## 1. Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# (Optional) To use the PINN solver
pip install -e ".[torch]"
```

Required: `numpy>=1.24`, `pytest>=7.0`

---

## 2. Running the Benchmark

### Basic execution

```bash
python -m app.run_benchmark
```

Default parameters: `--alpha 0.0 0.5 1.0 --nr 51 --dt 0.001 --t_end 0.1 --init gaussian`

### Customizing parameters

```bash
# Vary the nonlinearity parameter
python -m app.run_benchmark --alpha 0.0 0.5 1.0 2.0

# Increase grid resolution (slower)
python -m app.run_benchmark --nr 101 --dt 0.0005

# Use a sharp initial condition
python -m app.run_benchmark --init sharp

# Longer simulation
python -m app.run_benchmark --t_end 0.5 --dt 0.001
```

### Makefile shortcuts

```bash
make benchmark   # Run with defaults
make test        # Run tests
make clean       # Remove outputs/
```

---

## 3. Reference Solution Generation

### Overview

The reference solution is produced by **running the same Implicit FDM solver at 4x higher resolution**. This allows quantitative evaluation of how close each solver is to the "true" solution.

### Generation process in detail

Handled by `compute_reference()` in `app/run_benchmark.py`:

```
Benchmark grid               Reference grid
nr = 51 points               nr_fine = 4*51 - 3 = 201 points
dt = 0.001                   dt_fine = 0.001 / 4 = 0.00025
dr = 1/50 = 0.02             dr_fine = 1/200 = 0.005
```

**Steps:**

1. **Grid refinement**: `nr_fine = 4 * nr - 3` gives 4x spatial resolution
2. **Initial condition interpolation**: `np.interp` maps T0 onto the fine grid
3. **Time step refinement**: `dt_fine = dt / 4` gives 4x temporal resolution
4. **ImplicitFDM solve**: Crank-Nicolson on the fine grid
5. **Downsample**: Map the fine-grid result back to the original grid points

### Why this approach

- Crank-Nicolson is **2nd order in both space and time**
- Refining the grid by 4x reduces error by roughly (1/4)^2 = **1/16**
- The reference is therefore ~16x more accurate than the benchmark solve
- Works even for PDEs without known analytical solutions (Richardson extrapolation principle)

### Generating the reference standalone

Call directly from Python:

```python
import numpy as np
from app.run_benchmark import compute_reference, make_initial

nr = 51
r = np.linspace(0, 1, nr)
T0 = make_initial(r, "gaussian")  # or "sharp"

# Generate reference for alpha=0.5
T_ref = compute_reference(T0, r, dt=0.001, t_end=0.1, alpha=0.5)

print(T_ref.shape)   # (401, 51) — (time_steps+1, spatial_points)
print(T_ref[-1])      # Temperature profile at final time
```

---

## 4. Error Metric Definitions

### L2 error (relative, cylindrical-weighted)

```
L2 = sqrt( integral((T - T_ref)^2 * r dr) / integral(T_ref^2 * r dr) )
```

The `r` weighting reflects the cylindrical volume element. Errors near the edge (r->1) are weighted more heavily than near the center (r=0).

### L-infinity error (maximum absolute error)

```
Linf = max |T - T_ref|
```

Worst-case pointwise error across the entire domain.

---

## 5. Solver Selection Policy

```
score = L2_error + lambda * wall_time
```

- `lambda = 0.1` (default): accuracy-focused
- `lambda = 1.0`: more emphasis on speed
- `lambda = 0.0`: pure accuracy selection

Configurable in `policy/select.py` via `select_best()`.

---

## 6. Output Files

After a benchmark run, generated in `outputs/`:

| File | Contents |
|------|----------|
| `outputs/benchmark.csv` | Full results table (all solvers x all alpha) |
| `outputs/benchmark.md` | Markdown summary |

### CSV columns

| Column | Description |
|--------|-------------|
| `name` | Solver name |
| `alpha` | Nonlinearity parameter |
| `l2_error` | L2 error (vs reference) |
| `linf_error` | L-infinity error |
| `wall_time` | Wall-clock time [seconds] |
| `max_abs_gradient` | max\|dT/dr\| |
| `zero_crossings` | Number of zero crossings of dT/dr |
| `energy_content` | integral(T*r*dr) (thermal energy) |
| `max_chi` / `min_chi` | Max/min thermal diffusivity |
| `max_laplacian` | max\|d2T/dr2\| |
| `T_center` / `T_edge` | Temperature at center/edge |

---

## 7. Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run individual test files
python -m pytest tests/test_features.py -v   # Feature extraction
python -m pytest tests/test_solvers.py -v    # Solvers
python -m pytest tests/test_policy.py -v     # Selection policy
```

All 17 tests should pass:
- `test_features.py` (8 tests): Gradient, Laplacian, energy on analytic profiles (T=1-r^2)
- `test_solvers.py` (5 tests): Basic operation and boundary condition checks
- `test_policy.py` (4 tests): Selection logic correctness

---

## 8. Target PDE

### Equation

```
dT/dt = (1/r) d/dr (r chi dT/dr)
```

### Nonlinear diffusivity

```
chi(|dT/dr|) = 1 + alpha * |dT/dr|
```

- `alpha = 0`: Linear diffusion (analytical solution exists)
- `alpha > 0`: Enhanced diffusion in steep-gradient regions (anomalous transport model for plasmas)

### Boundary conditions

- `r = 0`: Neumann condition `dT/dr = 0` (symmetry)
- `r = 1`: Dirichlet condition `T = 0` (fixed wall temperature)

### Singularity treatment at r=0

`(1/r) d/dr(r chi dT/dr)` is an indeterminate form (0/0) at r=0.
Applying L'Hopital's rule:

```
lim_{r->0} (1/r) d/dr(r chi dT/dr) = 2 chi d2T/dr2
```

This allows stable computation at r=0.
