"""Implicit Crank-Nicolson FDM solver for radial heat transport."""

import numpy as np
from scipy.linalg import solve_banded
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

        # Banded matrix layout for solve_banded((1, 1), ab, d):
        #   ab[0, j] = A[j-1, j]  (super-diagonal, c[j-1])
        #   ab[1, j] = A[j, j]    (diagonal, b[j])
        #   ab[2, j] = A[j+1, j]  (sub-diagonal, a[j+1])
        ab = np.zeros((3, nr))
        d = np.zeros(nr)

        for n in range(nt):
            # Gradient via central differences (vectorized)
            dTdr = np.empty(nr)
            dTdr[0] = 0.0
            dTdr[1:-1] = (T[2:] - T[:-2]) / (2.0 * dr)
            dTdr[-1] = (T[-1] - T[-2]) / dr
            chi = self.chi(dTdr, alpha)

            # Interior points i=1..nr-2 (vectorized)
            chi_ip = 0.5 * (chi[1:-1] + chi[2:])    # chi at i+1/2
            chi_im = 0.5 * (chi[1:-1] + chi[:-2])    # chi at i-1/2

            Ap = rp * chi_ip * inv_ri_dr2
            Am = rm * chi_im * inv_ri_dr2

            # RHS
            d[1:-1] = T[1:-1] + half_dt * (Ap * (T[2:] - T[1:-1]) - Am * (T[1:-1] - T[:-2]))

            # Diagonal b[i] -> ab[1, i]
            ab[1, 1:-1] = 1.0 + half_dt * (Ap + Am)
            # Super-diagonal c[i] -> ab[0, i+1]  (c[i] = -half_dt * Ap[i-1])
            ab[0, 2:nr - 1] = -half_dt * Ap[:-1]
            # Sub-diagonal a[i] -> ab[2, i-1]  (a[i] = -half_dt * Am[i-1])
            ab[2, 0:nr - 2] = -half_dt * Am

            # r=0 (i=0): L'Hopital -> 2 chi d^2T/dr^2
            coeff = 2.0 * chi[0] / dr2
            rhs0 = 2.0 * coeff * (T[1] - T[0])
            d[0] = T[0] + half_dt * rhs0
            ab[1, 0] = 1.0 + dt * coeff       # b[0]
            ab[0, 1] = -dt * coeff             # c[0] -> ab[0, 1]

            # r=1 (i=nr-1): Dirichlet T=0
            d[-1] = 0.0
            ab[1, -1] = 1.0                    # b[nr-1]
            ab[0, -1] = 0.0                    # no super-diag at last col
            ab[2, -2] = 0.0                    # a[nr-1] -> ab[2, nr-2]

            # Solve banded system (LAPACK dgbsv)
            T = solve_banded((1, 1), ab, d, overwrite_ab=False, overwrite_b=False)
            T_history[n + 1] = T

        return T_history
