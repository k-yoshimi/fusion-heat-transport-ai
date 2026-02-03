"""Cell-Centered Finite Volume Method solver for radial heat transport."""

import numpy as np
from scipy.linalg import solve_banded
from solvers.base import SolverBase


class CellCenteredFVM(SolverBase):
    """Cell-centered Finite Volume Method solver in radial coordinates.

    Uses implicit time stepping with cell-centered discretization.
    Optimized with vectorized operations and banded matrix solver.

    Conservation form ensures strict conservation of integral quantities.
    """

    name = "cell_centered_fvm"

    def __init__(self, flux_scheme: str = "harmonic"):
        """Initialize cell-centered FVM solver.

        Args:
            flux_scheme: "harmonic" or "arithmetic" for face diffusivity.
        """
        self.flux_scheme = flux_scheme

    def solve(self, T0, r, dt, t_end, alpha):
        """Solve the heat equation using cell-centered FVM."""
        nr = len(r)
        dr = r[1] - r[0]
        nt = int(round(t_end / dt))

        # Cell face positions
        r_face = np.zeros(nr + 1)
        r_face[0] = 0.0
        r_face[1:-1] = 0.5 * (r[:-1] + r[1:])
        r_face[-1] = r[-1]

        # Control volumes: V_i = (r_{i+1/2}² - r_{i-1/2}²) / 2
        V = 0.5 * (r_face[1:]**2 - r_face[:-1]**2)
        V[0] = max(V[0], 1e-15)

        # Precompute face radii for flux computation
        r_face_inner = r_face[1:-1]  # Internal faces

        # Initialize
        T_history = np.zeros((nt + 1, nr))
        T_history[0] = T0.copy()
        T = T0.copy()

        # Banded matrix storage
        ab = np.zeros((3, nr))
        b = np.zeros(nr)

        for n in range(nt):
            # Compute gradient (vectorized)
            dTdr = np.zeros(nr)
            dTdr[0] = 0.0
            dTdr[1:-1] = (T[2:] - T[:-2]) / (2 * dr)
            dTdr[-1] = (T[-1] - T[-2]) / dr

            chi = self.chi(dTdr, alpha)

            # Face chi using harmonic or arithmetic mean (vectorized)
            chi_L = chi[:-1]
            chi_R = chi[1:]
            if self.flux_scheme == "harmonic":
                chi_face = 2 * chi_L * chi_R / (chi_L + chi_R + 1e-15)
            else:
                chi_face = 0.5 * (chi_L + chi_R)

            # Face coefficients: r_face * chi_face / dr
            face_coeff = r_face_inner * chi_face / dr

            # Build coefficients for interior cells (vectorized)
            # coeff_im[i] = face_coeff[i-1] / V[i] for i=1..nr-2
            # coeff_ip[i] = face_coeff[i] / V[i] for i=1..nr-2
            coeff_im = face_coeff[:-1] / V[1:-1]  # (nr-2,)
            coeff_ip = face_coeff[1:] / V[1:-1]   # (nr-2,)

            # Interior cells: diagonal and off-diagonals
            ab[1, 1:-1] = 1.0 + dt * (coeff_im + coeff_ip)
            ab[0, 2:-1] = -dt * coeff_ip[:-1]  # super-diagonal
            ab[2, 1:-2] = -dt * coeff_im[1:]   # sub-diagonal

            # Cell 0 (r=0): Neumann BC
            coeff_0p = face_coeff[0] / V[0]
            ab[1, 0] = 1.0 + dt * coeff_0p
            ab[0, 1] = -dt * coeff_0p

            # Cell nr-1 (r=1): Dirichlet BC
            ab[1, -1] = 1.0
            ab[0, -1] = 0.0
            ab[2, -2] = 0.0

            # RHS
            b[:] = T
            b[-1] = 0.0

            # Solve banded system
            T = solve_banded((1, 1), ab, b, overwrite_ab=False, overwrite_b=False)
            T_history[n + 1] = T

        return T_history
