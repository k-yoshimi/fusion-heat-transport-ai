"""Chebyshev pseudo-spectral solver for radial heat transport."""

import numpy as np
from scipy.linalg import solve
from solvers.base import SolverBase


class ChebyshevSpectral(SolverBase):
    """Chebyshev pseudo-spectral solver in radial coordinates.

    Uses Chebyshev-Gauss-Lobatto points for collocation.
    Optimized with vectorized matrix construction.

    Time stepping: Backward Euler for stability with nonlinear chi.
    """

    name = "chebyshev_spectral"

    def __init__(self, n_modes: int = 32):
        """Initialize Chebyshev spectral solver."""
        self.n_modes = n_modes

    def _chebyshev_grid(self, N):
        """Generate Chebyshev-Gauss-Lobatto points on [0, 1]."""
        j = np.arange(N)
        x = np.cos(j * np.pi / (N - 1))
        return (1.0 - x) / 2.0

    def _chebyshev_diff_matrix(self, N):
        """Compute Chebyshev differentiation matrix on [-1, 1]."""
        if N == 1:
            return np.array([[0.0]])

        x = np.cos(np.arange(N) * np.pi / (N - 1))
        c = np.ones(N)
        c[0] = 2.0
        c[-1] = 2.0
        c = c * ((-1.0) ** np.arange(N))

        X = np.tile(x, (N, 1))
        dX = X - X.T

        D = np.outer(c, 1.0 / c) / (dX + np.eye(N))
        D = D - np.diag(np.sum(D, axis=1))

        return D

    def solve(self, T0, r, dt, t_end, alpha):
        """Solve using Chebyshev spectral method with vectorized operations."""
        nr_orig = len(r)
        nt = int(round(t_end / dt))

        N = min(self.n_modes, nr_orig)
        r_cheb = self._chebyshev_grid(N)

        T = np.interp(r_cheb, r, T0)

        D_ref = self._chebyshev_diff_matrix(N)
        D = -2.0 * D_ref
        D2 = D @ D

        T_history = np.zeros((nt + 1, nr_orig))
        T_history[0] = T0.copy()

        I = np.eye(N)

        # Precompute safe r for interior points (avoid division by zero)
        r_safe = np.where(r_cheb > 1e-10, r_cheb, 1e-10)

        for n in range(nt):
            dTdr = D @ T
            dTdr[0] = 0.0  # Neumann BC at r=0

            chi = self.chi(dTdr, alpha)

            # Vectorized L matrix construction
            # L[i,:] = chi[i] * D2[i,:] + (chi[i]/r[i]) * D[i,:] for interior
            # L[0,:] = 2*chi[0] * D2[0,:] for r=0 (L'Hopital)

            # Interior points (i=1 to N-2): vectorized
            chi_over_r = chi / r_safe
            chi_over_r[0] = 0.0  # Will be overwritten by L'Hopital

            # Build L using broadcasting
            L = chi[:, np.newaxis] * D2 + chi_over_r[:, np.newaxis] * D

            # Override r=0 row with L'Hopital limit
            L[0, :] = 2.0 * chi[0] * D2[0, :]

            # Backward Euler: (I - dt*L) T^{n+1} = T^n
            A = I - dt * L
            rhs = T.copy()

            # Dirichlet BC at r=1 (index N-1)
            A[-1, :] = 0.0
            A[-1, -1] = 1.0
            rhs[-1] = 0.0

            T = solve(A, rhs)
            T = np.clip(T, -10, 10)

            T_history[n + 1] = np.interp(r, r_cheb, T)

        return T_history
