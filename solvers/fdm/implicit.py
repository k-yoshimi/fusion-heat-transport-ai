"""Implicit Crank-Nicolson FDM solver for radial heat transport."""

import numpy as np
from solvers.base import SolverBase


class ImplicitFDM(SolverBase):
    """Crank-Nicolson finite difference solver in radial coordinates.

    Handles the r=0 singularity via L'Hôpital's rule:
        (1/r) ∂/∂r(r χ ∂T/∂r) → 2χ ∂²T/∂r² at r=0.
    """

    name = "implicit_fdm"

    def solve(self, T0, r, dt, t_end, alpha):
        nr = len(r)
        dr = r[1] - r[0]
        nt = int(round(t_end / dt))
        dr2 = dr * dr
        half_dt = 0.5 * dt

        T_history = np.zeros((nt + 1, nr))
        T_history[0] = T0.copy()
        T = T0.copy()

        # Precompute geometric factors for interior points (i=1..nr-2)
        ri = r[1:-1]                         # (nr-2,)
        rp = ri + 0.5 * dr                   # r at i+1/2
        rm = ri - 0.5 * dr                   # r at i-1/2
        inv_ri_dr2 = 1.0 / (ri * dr2)        # 1/(r_i * dr^2)

        # Preallocate tridiagonal arrays
        a = np.zeros(nr)
        b = np.zeros(nr)
        c = np.zeros(nr)
        d = np.zeros(nr)

        # Dirichlet BC at r=1 (constant across time steps)
        b[-1] = 1.0

        for n in range(nt):
            # Gradient via central differences (vectorized)
            dTdr = np.empty(nr)
            dTdr[0] = 0.0
            dTdr[1:-1] = (T[2:] - T[:-2]) / (2.0 * dr)
            dTdr[-1] = (T[-1] - T[-2]) / dr
            chi = self.chi(dTdr, alpha)

            # Interior points: vectorized tridiagonal construction
            chi_ip = 0.5 * (chi[1:-1] + chi[2:])    # chi at i+1/2
            chi_im = 0.5 * (chi[1:-1] + chi[:-2])    # chi at i-1/2

            Ap = rp * chi_ip * inv_ri_dr2
            Am = rm * chi_im * inv_ri_dr2

            # Explicit RHS
            d[1:-1] = T[1:-1] + half_dt * (Ap * (T[2:] - T[1:-1]) - Am * (T[1:-1] - T[:-2]))

            # Implicit coefficients
            a[1:-1] = -half_dt * Am
            b[1:-1] = 1.0 + half_dt * (Ap + Am)
            c[1:-1] = -half_dt * Ap

            # r=0: L'Hôpital → 2χ d²T/dr²
            coeff = 2.0 * chi[0] / dr2
            rhs0 = 2.0 * coeff * (T[1] - T[0])
            d[0] = T[0] + half_dt * rhs0
            a[0] = 0.0
            b[0] = 1.0 + dt * coeff
            c[0] = -dt * coeff

            # r=1: Dirichlet T=0 (a[-1], b[-1], c[-1], d[-1] stay 0/1/0/0)
            d[-1] = 0.0

            # Thomas algorithm (in-place)
            T = _thomas(a, b, c, d)
            T_history[n + 1] = T

        return T_history


def _thomas(a, b, c, d):
    """Solve tridiagonal system using Thomas algorithm."""
    n = len(d)
    c_ = np.empty(n)
    d_ = np.empty(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * c_[i - 1]
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i] * d_[i - 1]) / denom

    x = np.empty(n)
    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x
