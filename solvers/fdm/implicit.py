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
        T_history = np.zeros((nt + 1, nr))
        T_history[0] = T0.copy()
        T = T0.copy()

        for n in range(nt):
            # Compute chi at current time using current gradient
            dTdr = np.zeros(nr)
            dTdr[1:-1] = (T[2:] - T[:-2]) / (2 * dr)
            dTdr[0] = 0.0  # Neumann BC
            chi = self.chi(dTdr, alpha)

            # Build tridiagonal system for Crank-Nicolson
            # Interior: (1/r_i) d/dr(r chi dT/dr) ≈ finite differences
            a = np.zeros(nr)  # sub-diagonal
            b = np.zeros(nr)  # diagonal
            c = np.zeros(nr)  # super-diagonal
            d = np.zeros(nr)  # RHS

            for i in range(1, nr - 1):
                ri = r[i]
                chi_ip = 0.5 * (chi[i] + chi[i + 1])  # chi at i+1/2
                chi_im = 0.5 * (chi[i] + chi[i - 1])  # chi at i-1/2

                rp = ri + 0.5 * dr  # r at i+1/2
                rm = ri - 0.5 * dr  # r at i-1/2

                Ap = rp * chi_ip / (ri * dr**2)
                Am = rm * chi_im / (ri * dr**2)

                # Explicit part (known)
                rhs = T[i] + 0.5 * dt * (Ap * (T[i + 1] - T[i]) - Am * (T[i] - T[i - 1]))

                # Implicit part (unknown)
                a[i] = -0.5 * dt * Am
                b[i] = 1.0 + 0.5 * dt * (Ap + Am)
                c[i] = -0.5 * dt * Ap
                d[i] = rhs

            # r=0: L'Hôpital → 2 chi d²T/dr²
            chi0 = chi[0]
            coeff = 2.0 * chi0 / dr**2
            d[0] = T[0] + 0.5 * dt * coeff * (T[1] - T[0])
            # Neumann: T[-1] = T[1], so d²T/dr² = 2(T[1]-T[0])/dr²
            # For implicit: T_new[0](1 + dt*coeff) - dt*coeff*T_new[1] = rhs
            # But using symmetry T[-1]=T[1]:
            # d²T/dr² ≈ (T[1] - 2T[0] + T[-1])/dr² = 2(T[1]-T[0])/dr²
            rhs0_explicit = 2.0 * coeff * (T[1] - T[0])
            d[0] = T[0] + 0.5 * dt * rhs0_explicit
            a[0] = 0.0
            b[0] = 1.0 + dt * coeff
            c[0] = -dt * coeff

            # r=1: Dirichlet T=0
            a[-1] = 0.0
            b[-1] = 1.0
            c[-1] = 0.0
            d[-1] = 0.0

            # Thomas algorithm
            T = _thomas(a, b, c, d)
            T_history[n + 1] = T

        return T_history


def _thomas(a, b, c, d):
    """Solve tridiagonal system using Thomas algorithm."""
    n = len(d)
    c_ = np.zeros(n)
    d_ = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n):
        m = a[i] / (b[i] - a[i] * c_[i - 1])
        c_[i] = c[i] / (b[i] - a[i] * c_[i - 1])
        d_[i] = (d[i] - a[i] * d_[i - 1]) / (b[i] - a[i] * c_[i - 1])

    x = np.zeros(n)
    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x
