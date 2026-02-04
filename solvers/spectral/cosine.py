"""Cosine expansion pseudo-spectral solver for radial heat transport."""

import numpy as np
from solvers.base import SolverBase


class CosineSpectral(SolverBase):
    """Pseudo-spectral solver using cosine basis.

    Uses basis phi_k(r) = cos((k+0.5)*pi*r) which satisfies:
    - phi_k'(0) = 0 (Neumann at r=0)
    - phi_k(1) = 0 (Dirichlet at r=1)

    Time stepping: implicit linear diffusion (chi=0.1 baseline) in spectral space,
    explicit nonlinear correction in physical space.

    Stability:
        - CFL constraint: dt <= 0.5 * dr^2 / max_chi_nl
        - Recommended max_alpha: 0.5 (explicit nonlinear part limits stability)
    """

    name = "spectral_cosine"

    # CFL coefficient for explicit nonlinear correction
    cfl_coefficient = 0.5
    max_alpha_recommended = 0.5

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
        phi_sq = phi ** 2  # (N, nr)
        phi_norm = np.trapz(phi_sq, r, axis=1)  # (N,)

        # Precompute weights for forward transform
        w = np.full(nr, dr)
        w[0] = 0.5 * dr
        w[-1] = 0.5 * dr
        phi_w = phi * w[np.newaxis, :]  # (N, nr) weighted basis
        phi_w_norm = phi_w / phi_norm[:, np.newaxis]  # (N, nr)

        # Baseline diffusivity is 0.1, so linear decay uses 0.1 * lam
        decay = np.exp(-0.1 * lam * dt)  # (N,)

        # Precompute geometric factors for nonlinear flux (interior points)
        ri = r[1:-1]
        rp = ri + 0.5 * dr
        rm = ri - 0.5 * dr
        inv_ri_dr2 = 1.0 / (ri * dr2)

        for n in range(nt):
            # Forward transform: a = phi_w_norm @ T (matrix-vector multiply)
            a = phi_w_norm @ T  # (N,)

            # Decay each mode (linear baseline chi=0.1)
            a *= decay

            # Reconstruct: T = phi^T @ a
            T = phi.T @ a  # (nr,)

            # Nonlinear correction: chi_nl = chi_total - 0.1 (baseline)
            dTdr = np.empty(nr)
            dTdr[0] = 0.0
            dTdr[1:-1] = (T[2:] - T[:-2]) / (2.0 * dr)
            dTdr[-1] = (T[-1] - T[-2]) / dr
            abs_dTdr = np.abs(dTdr)
            chi_nl = np.where(abs_dTdr > 0.5,
                              (abs_dTdr - 0.5) ** alpha,
                              0.0)
            # Clamp to prevent numerical blowup in explicit correction
            np.clip(chi_nl, 0.0, 1.0 / (dt + 1e-30), out=chi_nl)
            chi_nl = np.nan_to_num(chi_nl, nan=0.0, posinf=0.0, neginf=0.0)

            cnp = 0.5 * (chi_nl[1:-1] + chi_nl[2:])
            cnm = 0.5 * (chi_nl[1:-1] + chi_nl[:-2])

            nl_flux = np.zeros(nr)
            nl_flux[1:-1] = (rp * cnp * (T[2:] - T[1:-1]) - rm * cnm * (T[1:-1] - T[:-2])) * inv_ri_dr2
            nl_flux[0] = 2.0 * chi_nl[0] * (T[1] - T[0]) / dr2

            T = T + dt * nl_flux
            T = np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)

            T[-1] = 0.0
            T_history[n + 1] = T

        return T_history
