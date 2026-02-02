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
        dr2 = dr * dr

        N = min(self.n_modes, nr // 2)
        ks = np.arange(N)
        lam = ((ks + 0.5) * np.pi) ** 2

        # Basis matrix: phi[k, j] = cos((k+0.5)*pi*r_j), shape (N, nr)
        phi = np.cos(np.outer((ks + 0.5) * np.pi, r))

        # Precompute norms via trapezoidal rule (vectorized)
        # trapz(f, r) with uniform spacing = dr * (0.5*f[0] + f[1] + ... + f[-2] + 0.5*f[-1])
        phi_sq = phi ** 2  # (N, nr)
        phi_norm = np.trapz(phi_sq, r, axis=1)  # (N,)

        # Precompute weights for forward transform: phi * trapz_weight / norm
        # For trapz: weight = dr everywhere, except dr/2 at endpoints
        w = np.full(nr, dr)
        w[0] = 0.5 * dr
        w[-1] = 0.5 * dr
        phi_w = phi * w[np.newaxis, :]  # (N, nr) weighted basis
        phi_w_norm = phi_w / phi_norm[:, np.newaxis]  # (N, nr)

        # Precompute exponential decay factor (constant across time steps)
        decay = np.exp(-lam * dt)  # (N,)

        # Precompute geometric factors for nonlinear flux (interior points)
        ri = r[1:-1]
        rp = ri + 0.5 * dr
        rm = ri - 0.5 * dr
        inv_ri_dr2 = 1.0 / (ri * dr2)

        for n in range(nt):
            # Forward transform: a = phi_w_norm @ T (matrix-vector multiply)
            a = phi_w_norm @ T  # (N,)

            # Decay each mode
            a *= decay

            # Reconstruct: T = phi^T @ a
            T = phi.T @ a  # (nr,)

            # Nonlinear correction (vectorized)
            if alpha > 0:
                dTdr = np.empty(nr)
                dTdr[0] = 0.0
                dTdr[1:-1] = (T[2:] - T[:-2]) / (2.0 * dr)
                dTdr[-1] = (T[-1] - T[-2]) / dr
                chi_nl = alpha * np.abs(dTdr)

                cnp = 0.5 * (chi_nl[1:-1] + chi_nl[2:])
                cnm = 0.5 * (chi_nl[1:-1] + chi_nl[:-2])

                nl_flux = np.zeros(nr)
                nl_flux[1:-1] = (rp * cnp * (T[2:] - T[1:-1]) - rm * cnm * (T[1:-1] - T[:-2])) * inv_ri_dr2
                nl_flux[0] = 2.0 * chi_nl[0] * (T[1] - T[0]) / dr2

                T = T + dt * nl_flux

            T[-1] = 0.0
            T_history[n + 1] = T

        return T_history
