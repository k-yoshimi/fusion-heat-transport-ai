"""IMEX (Implicit-Explicit) solver for radial heat transport."""

import numpy as np
from scipy.linalg import solve_banded
from solvers.base import SolverBase


class IMEXFDM(SolverBase):
    """IMEX (Implicit-Explicit) finite difference solver in radial coordinates.

    Splits the diffusion operator into linear and nonlinear parts:
    - Linear part (chi_base): treated implicitly for stability
    - Nonlinear part (chi - chi_base): treated explicitly for efficiency

    Uses adaptive chi_base to ensure stability of the explicit part.
    Based on ImplicitFDM structure with split operator approach.

    Stability:
        - CFL constraint: dt <= 0.4 * dr^2 / max_chi_nl for explicit part
        - Adaptive chi_base can handle any alpha by increasing implicit treatment
    """

    name = "imex_fdm"

    # CFL coefficient for explicit nonlinear correction
    cfl_coefficient = 0.4

    def __init__(self, chi_base: float = 0.1, adaptive_base: bool = True):
        """Initialize IMEX solver.

        Args:
            chi_base: Base diffusivity for implicit treatment (default 0.1)
            adaptive_base: If True, adapt chi_base based on current chi values
        """
        self.chi_base_init = chi_base
        self.adaptive_base = adaptive_base

    def solve(self, T0, r, dt, t_end, alpha):
        """Solve using IMEX method."""
        nr = len(r)
        dr = r[1] - r[0]
        nt = int(round(t_end / dt))
        dr2 = dr * dr
        half_dt = 0.5 * dt

        T_history = np.zeros((nt + 1, nr))
        T_history[0] = T0.copy()
        T = T0.copy()

        # Geometric factors for interior points (i=1..nr-2)
        ri = r[1:-1]
        rp = ri + 0.5 * dr
        rm = ri - 0.5 * dr
        inv_ri_dr2 = 1.0 / (ri * dr2)

        # Banded matrix for implicit part
        ab = np.zeros((3, nr))
        d = np.zeros(nr)

        for n in range(nt):
            # Compute gradient
            dTdr = np.empty(nr)
            dTdr[0] = 0.0
            dTdr[1:-1] = (T[2:] - T[:-2]) / (2.0 * dr)
            dTdr[-1] = (T[-1] - T[-2]) / dr

            # Total chi
            chi = self.chi(dTdr, alpha)

            # Adaptive chi_base: use mean chi to balance implicit/explicit parts
            # This ensures chi_nl is small enough for stability
            if self.adaptive_base:
                chi_base = max(np.mean(chi), self.chi_base_init)
            else:
                chi_base = self.chi_base_init

            # Nonlinear part (clamped to non-negative)
            chi_nl = np.maximum(chi - chi_base, 0.0)

            # CFL check for explicit stability
            max_chi_nl = np.max(chi_nl)
            cfl_explicit = dt * max_chi_nl / dr2
            if cfl_explicit > 0.4:
                # Increase chi_base to reduce explicit contribution
                chi_base = chi_base + max_chi_nl
                chi_nl = np.maximum(chi - chi_base, 0.0)

            # Linear coefficients (constant chi_base)
            Ap0 = rp * chi_base * inv_ri_dr2
            Am0 = rm * chi_base * inv_ri_dr2

            # Nonlinear coefficients at cell faces
            chi_nl_ip = 0.5 * (chi_nl[1:-1] + chi_nl[2:])
            chi_nl_im = 0.5 * (chi_nl[1:-1] + chi_nl[:-2])
            Ap_nl = rp * chi_nl_ip * inv_ri_dr2
            Am_nl = rm * chi_nl_im * inv_ri_dr2

            # Explicit nonlinear flux (interior points)
            nl_flux = np.zeros(nr)
            nl_flux[1:-1] = Ap_nl * (T[2:] - T[1:-1]) - Am_nl * (T[1:-1] - T[:-2])
            # r=0: L'Hopital
            nl_flux[0] = 2.0 * chi_nl[0] / dr2 * (T[1] - T[0])

            # Build RHS: Crank-Nicolson for linear part + explicit nonlinear
            # Interior points: T + (dt/2)*L_0*T + dt*L_nl*T
            lin_flux = Ap0 * (T[2:] - T[1:-1]) - Am0 * (T[1:-1] - T[:-2])
            d[1:-1] = T[1:-1] + half_dt * lin_flux + dt * nl_flux[1:-1]

            # Build implicit matrix for linear part (Crank-Nicolson)
            # (I - dt/2 * L_0) T^{n+1} = RHS
            ab[1, 1:-1] = 1.0 + half_dt * (Ap0 + Am0)
            ab[0, 2:nr - 1] = -half_dt * Ap0[:-1]
            ab[2, 1:nr - 1] = -half_dt * Am0

            # r=0: L'Hopital with Crank-Nicolson
            coeff_0 = 2.0 * chi_base / dr2
            lin_flux_0 = coeff_0 * (T[1] - T[0])
            d[0] = T[0] + half_dt * lin_flux_0 + dt * nl_flux[0]
            ab[1, 0] = 1.0 + half_dt * coeff_0
            ab[0, 1] = -half_dt * coeff_0

            # r=1: Dirichlet T=0
            d[-1] = 0.0
            ab[1, -1] = 1.0
            ab[0, -1] = 0.0
            ab[2, -2] = 0.0

            # Solve banded system
            T = solve_banded((1, 1), ab, d, overwrite_ab=False, overwrite_b=False)

            # Clamp for stability (should not be needed if CFL is satisfied)
            T = np.clip(T, -100.0, 100.0)

            T_history[n + 1] = T

        return T_history
