"""Cosine expansion pseudo-spectral solver for radial heat transport."""

import numpy as np
from solvers.base import SolverBase


class CosineSpectral(SolverBase):
    """Pseudo-spectral solver using cosine basis.

    Uses basis phi_k(r) = cos((k+0.5)*pi*r) which satisfies:
    - phi_k'(0) = 0 (Neumann at r=0)
    - phi_k(1) = 0 (Dirichlet at r=1)

    Time stepping: implicit linear diffusion in spectral space,
    explicit nonlinear correction in physical space.
    """

    name = "spectral_cosine"

    def __init__(self, n_modes: int = 16):
        self.n_modes = n_modes

    def solve(self, T0, r, dt, t_end, alpha):
        nr = len(r)
        nt = int(round(t_end / dt))
        T_history = np.zeros((nt + 1, nr))
        T_history[0] = T0.copy()
        T = T0.copy()
        dr = r[1] - r[0]

        N = min(self.n_modes, nr // 2)
        ks = np.arange(N)
        lam = ((ks + 0.5) * np.pi) ** 2

        # Basis matrix: phi[k, j] = cos((k+0.5)*pi*r_j)
        phi = np.cos(np.outer((ks + 0.5) * np.pi, r))  # (N, nr)

        # Use proper L2 inner product via trapezoidal rule
        # <phi_k, phi_l> = int_0^1 cos((k+0.5)*pi*r) cos((l+0.5)*pi*r) dr = 0.5 delta_kl
        # Numerically:
        phi_norm = np.zeros(N)
        for k in range(N):
            phi_norm[k] = np.trapz(phi[k] ** 2, r)

        for n in range(nt):
            # Forward transform: a_k = <T, phi_k> / <phi_k, phi_k>
            a = np.zeros(N)
            for k in range(N):
                a[k] = np.trapz(T * phi[k], r) / phi_norm[k]

            # Decay each mode: implicit linear diffusion
            # dT/dt = d²T/dr² for chi=1 has eigenvalue -lam_k for phi_k
            a_new = a * np.exp(-lam * dt)

            # Reconstruct in physical space
            T = np.dot(a_new, phi)

            # Nonlinear correction (explicit, operator splitting)
            if alpha > 0:
                dTdr = np.zeros(nr)
                dTdr[1:-1] = (T[2:] - T[:-2]) / (2 * dr)
                chi_nl = alpha * np.abs(dTdr)  # chi - 1

                nl_flux = np.zeros(nr)
                for i in range(1, nr - 1):
                    ri = r[i]
                    cnp = 0.5 * (chi_nl[i] + chi_nl[i + 1])
                    cnm = 0.5 * (chi_nl[i] + chi_nl[i - 1])
                    rp = ri + 0.5 * dr
                    rm = ri - 0.5 * dr
                    nl_flux[i] = (rp * cnp * (T[i + 1] - T[i]) - rm * cnm * (T[i] - T[i - 1])) / (ri * dr**2)
                nl_flux[0] = 2.0 * chi_nl[0] * (T[1] - T[0]) / dr**2

                T = T + dt * nl_flux

            T[-1] = 0.0
            T_history[n + 1] = T

        return T_history
