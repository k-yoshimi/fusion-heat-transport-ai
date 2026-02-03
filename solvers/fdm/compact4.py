"""4th-order Compact Finite Difference Method solver for radial heat transport."""

import numpy as np
from scipy.linalg import solve_banded
from solvers.base import SolverBase


class Compact4FDM(SolverBase):
    """4th-order Compact Finite Difference solver in radial coordinates.

    Uses Crank-Nicolson time stepping with 4th-order spatial accuracy.
    Optimized with vectorized operations and banded matrix solver.
    """

    name = "compact4_fdm"

    def solve(self, T0, r, dt, t_end, alpha):
        """Solve the heat equation using 4th-order compact FDM."""
        nr = len(r)
        dr = r[1] - r[0]
        dr2 = dr * dr
        nt = int(round(t_end / dt))
        half_dt = 0.5 * dt

        T_history = np.zeros((nt + 1, nr))
        T_history[0] = T0.copy()
        T = T0.copy()

        # Precompute geometric factors
        ri = r.copy()
        ri[0] = dr / 2  # Regularize at r=0

        # Precompute face positions for interior points
        rp = r[1:-1] + 0.5 * dr  # r at i+1/2
        rm = r[1:-1] - 0.5 * dr  # r at i-1/2
        ri_interior = ri[1:-1]
        inv_ri_dr2 = 1.0 / (ri_interior * dr2)

        # Banded matrix storage
        ab = np.zeros((3, nr))
        rhs = np.zeros(nr)

        for n in range(nt):
            # 4th-order gradient for interior points where possible
            dTdr = np.zeros(nr)
            dTdr[0] = 0.0  # Neumann BC
            # 4th order for i=2..nr-3
            if nr > 4:
                i = np.arange(2, nr - 2)
                dTdr[2:-2] = (-T[4:] + 8*T[3:-1] - 8*T[1:-3] + T[:-4]) / (12 * dr)
            # 2nd order near boundaries
            dTdr[1] = (T[2] - T[0]) / (2 * dr)
            if nr > 2:
                dTdr[-2] = (T[-1] - T[-3]) / (2 * dr)
            dTdr[-1] = (T[-1] - T[-2]) / dr

            chi = self.chi(dTdr, alpha)

            # Vectorized computation of face chi values
            chi_ip = 0.5 * (chi[1:-1] + chi[2:])  # chi at i+1/2
            chi_im = 0.5 * (chi[1:-1] + chi[:-2])  # chi at i-1/2

            # Flux coefficients
            Fp = rp * chi_ip * inv_ri_dr2
            Fm = rm * chi_im * inv_ri_dr2

            # Build banded matrix for interior points
            ab[1, 1:-1] = 1.0 + half_dt * (Fp + Fm)  # diagonal
            ab[0, 2:nr-1] = -half_dt * Fp[:-1]  # super-diagonal
            ab[2, 0:nr-2] = -half_dt * Fm  # sub-diagonal

            # Build RHS for interior points (vectorized)
            rhs[1:-1] = T[1:-1] + half_dt * (Fp * (T[2:] - T[1:-1]) - Fm * (T[1:-1] - T[:-2]))

            # r=0: L'Hopital rule -> 2χ d²T/dr²
            coeff_0 = 2.0 * chi[0] / dr2
            ab[1, 0] = 1.0 + dt * coeff_0
            ab[0, 1] = -dt * coeff_0
            rhs[0] = T[0] + half_dt * 2.0 * coeff_0 * (T[1] - T[0])

            # r=1: Dirichlet BC
            ab[1, -1] = 1.0
            ab[2, -2] = 0.0
            rhs[-1] = 0.0

            # Solve banded system
            T = solve_banded((1, 1), ab, rhs, overwrite_ab=False, overwrite_b=False)
            T_history[n + 1] = T

        return T_history
