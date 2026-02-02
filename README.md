# Fusion Heat Transport PDE Benchmark

Benchmark suite for comparing numerical solvers of the 1D radial heat transport equation in fusion plasmas:

```
∂T/∂t = (1/r) ∂/∂r (r χ(|∂T/∂r|) ∂T/∂r)
```

where χ = 1 + α|∂T/∂r| is a nonlinear thermal diffusivity.

**Boundary conditions:** Neumann ∂T/∂r = 0 at r=0, Dirichlet T=0 at r=1.

## Solvers

| Solver | Method | Notes |
|--------|--------|-------|
| `implicit_fdm` | Crank-Nicolson FDM | L'Hôpital at r=0, Thomas algorithm |
| `spectral_cosine` | Cosine expansion | cos((k+0.5)πr) basis, operator splitting |
| `pinn_stub` | Physics-Informed NN | Requires PyTorch (optional) |

## Quick Start

```bash
pip install -e ".[dev]"

# Run tests
make test

# Run benchmark
make benchmark
# or
python -m app.run_benchmark --alpha 0.0 0.5 1.0
```

## CLI Options

```
--alpha    Nonlinearity parameters (default: 0.0 0.5 1.0)
--nr       Radial grid points (default: 51)
--dt       Time step (default: 0.001)
--t_end    Final time (default: 0.1)
--init     Initial condition: gaussian | sharp (default: gaussian)
```

## Solver Selection Policy

The best solver is selected by minimizing:

```
score = L2_error + λ × wall_time
```

where λ is configurable (default 0.1). See `policy/select.py`.

## Project Structure

```
app/              CLI entrypoint
features/         PDE feature extraction (gradients, energy, etc.)
solvers/          Solver implementations (FDM, spectral, PINN)
metrics/          Error metrics (L2, L∞)
policy/           Solver selection policy
reports/          CSV + markdown report generation
tests/            Unit tests
outputs/          Generated benchmark results (gitignored)
```

## Reference Solution

The reference is computed using the implicit FDM solver with 4x grid refinement and 4x smaller time step.
