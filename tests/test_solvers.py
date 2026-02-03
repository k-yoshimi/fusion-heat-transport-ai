"""Sanity checks for solvers."""

import numpy as np
import pytest
from solvers.fdm.implicit import ImplicitFDM
from solvers.spectral.cosine import CosineSpectral
from solvers.pinn.stub import PINNStub


@pytest.fixture
def setup():
    nr = 51
    r = np.linspace(0, 1, nr)
    T0 = 1.0 - r**2
    return T0, r


def test_fdm_basic(setup):
    T0, r = setup
    solver = ImplicitFDM()
    T_hist = solver.solve(T0, r, dt=0.001, t_end=0.01, alpha=0.0)
    assert T_hist.shape[0] == 11  # 10 steps + initial
    assert T_hist.shape[1] == len(r)
    # Temperature should decrease (diffusion)
    assert T_hist[-1, 0] < T_hist[0, 0]
    # BC: T(r=1) = 0
    np.testing.assert_allclose(T_hist[-1, -1], 0.0, atol=1e-10)


def test_fdm_neumann_bc(setup):
    """Check dT/dr ≈ 0 at r=0."""
    T0, r = setup
    solver = ImplicitFDM()
    # Run longer to allow diffusion (chi=0.1 baseline is slow)
    T_hist = solver.solve(T0, r, dt=0.001, t_end=0.1, alpha=0.0)
    dr = r[1] - r[0]
    dTdr_0 = (T_hist[-1, 1] - T_hist[-1, 0]) / dr
    assert abs(dTdr_0) < 0.5  # approximately zero gradient at center


def test_spectral_basic(setup):
    T0, r = setup
    solver = CosineSpectral(n_modes=16)
    T_hist = solver.solve(T0, r, dt=0.001, t_end=0.01, alpha=0.0)
    assert T_hist.shape[0] == 11
    assert T_hist[-1, 0] < T_hist[0, 0]
    np.testing.assert_allclose(T_hist[-1, -1], 0.0, atol=1e-10)


def test_fdm_nonlinear(setup):
    T0, r = setup
    solver = ImplicitFDM()
    T_hist = solver.solve(T0, r, dt=0.001, t_end=0.01, alpha=1.0)
    assert T_hist[-1, 0] < T_hist[0, 0]
    np.testing.assert_allclose(T_hist[-1, -1], 0.0, atol=1e-10)


def test_pinn_stub_runs(setup):
    T0, r = setup
    solver = PINNStub(epochs=5)
    T_hist = solver.solve(T0, r, dt=0.01, t_end=0.01, alpha=0.0)
    assert T_hist.shape[0] == 2  # 1 step + initial
    assert T_hist[0, 0] == pytest.approx(T0[0])
