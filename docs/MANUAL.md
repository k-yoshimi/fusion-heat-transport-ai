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

### Post-hoc selection (default)

```
score = L2_error + lambda * wall_time
```

- `lambda = 0.1` (default): accuracy-focused
- `lambda = 1.0`: more emphasis on speed
- `lambda = 0.0`: pure accuracy selection

Configurable in `policy/select.py` via `select_best()`.

### ML-based selection

A decision tree model can predict the best solver **before** running all solvers, based only on problem parameters and initial condition features. This avoids redundant computation.

#### Workflow

```
1. Generate training data  →  2. Train model  →  3. Predict & run one solver
```

#### Step 1: Generate training data

Run a parameter sweep over multiple alpha, grid size, time step, initial condition combinations. All three solvers are benchmarked for each combination, and the best solver (by score) is labeled.

```bash
# Full sweep (~432 instances, may take several minutes)
python -m app.run_benchmark --generate-data

# Data saved to data/training_data.csv
```

The sweep covers:
- `alpha`: 0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0
- `init`: gaussian, sharp
- `nr`: 31, 51, 71
- `dt`: 0.0005, 0.001, 0.002
- `t_end`: 0.05, 0.1, 0.2

#### Step 2: Train the model

```bash
python -m policy.train --data data/training_data.csv --model data/solver_model.npz

# Or generate + train in one command
python -m policy.train --generate

# Makefile shortcut
make train
```

Options:
- `--data PATH`: Input CSV path (default: `data/training_data.csv`)
- `--model PATH`: Output model path (default: `data/solver_model.npz`)
- `--max-depth N`: Decision tree max depth (default: 5)
- `--generate`: Generate training data before training

Training accuracy is printed after fitting. The model is a numpy-only CART decision tree using Gini impurity — no sklearn dependency.

#### Step 3: Use the ML selector

```bash
# Predict the best solver and run only that one
python -m app.run_benchmark --use-ml-selector --alpha 1.5

# Custom model path
python -m app.run_benchmark --use-ml-selector --model-path data/solver_model.npz --alpha 0.5 1.0
```

The ML selector extracts 14 features from the initial condition and problem parameters, feeds them to the decision tree, and runs only the predicted solver.

#### Incremental update (`--update`)

The `--update` flag appends the current benchmark results to the training data and retrains the model. This enables incremental learning: each benchmark run improves future predictions.

```bash
# Run benchmark and update the model with the results
python -m app.run_benchmark --alpha 0.5 1.0 --update

# With custom paths
python -m app.run_benchmark --alpha 2.0 --update --data-path data/training_data.csv --model-path data/solver_model.npz
```

What `--update` does:
1. Runs the normal benchmark (all solvers, all alpha values)
2. For each alpha value, extracts initial features and determines the best solver
3. Appends these samples to `data/training_data.csv`
4. Retrains the decision tree on all accumulated data
5. Saves the updated model to `data/solver_model.npz`

Typical workflow for incremental improvement:

```bash
# Initial setup
python -m app.run_benchmark --generate-data
python -m policy.train --data data/training_data.csv

# Day-to-day use: run benchmarks and accumulate data
python -m app.run_benchmark --alpha 0.3 0.7 --init sharp --update
python -m app.run_benchmark --alpha 1.2 --nr 71 --update

# Use the improved model
python -m app.run_benchmark --use-ml-selector --alpha 1.5
```

#### Feature list (14 features)

| Category | Feature | Description |
|----------|---------|-------------|
| Problem param | `alpha` | Nonlinearity parameter |
| Problem param | `nr` | Number of grid points |
| Problem param | `dt` | Time step |
| Problem param | `t_end` | Final simulation time |
| Problem param | `init_gaussian` | 1 if gaussian IC, 0 otherwise |
| Problem param | `init_sharp` | 1 if sharp IC, 0 otherwise |
| Physical | `max_abs_gradient` | max\|dT₀/dr\| |
| Physical | `energy_content` | ∫T₀·r·dr |
| Physical | `max_chi` | max(1 + α\|dT₀/dr\|) |
| Physical | `max_laplacian` | max\|d²T₀/dr²\| |
| Physical | `T_center` | T₀(r=0) |
| Derived | `gradient_sharpness` | max_abs_gradient / T_center |
| Derived | `chi_ratio` | max_chi / min_chi |
| Derived | `problem_stiffness` | α × max_abs_gradient |

#### Model files

| File | Description |
|------|-------------|
| `data/training_data.csv` | Training data (features + best_solver label) |
| `data/solver_model.npz` | Serialized decision tree model |

Both are gitignored. Regenerate with `make train`.

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

All 22 tests should pass:
- `test_features.py` (8 tests): Gradient, Laplacian, energy on analytic profiles (T=1-r^2)
- `test_solvers.py` (5 tests): Basic operation and boundary condition checks
- `test_policy.py` (9 tests): Selection logic, ML feature extraction, decision tree, incremental update

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
