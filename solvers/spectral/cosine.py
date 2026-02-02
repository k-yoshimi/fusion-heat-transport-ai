"""Cosine expansion pseudo-spectral solver for radial heat transport."""

import numpy as np
from solvers.base import SolverBase


class CosineSpectral(SolverBase):
    """Pseudo-spectral solver using cosine basis.

    Expands T(r,t) = Σ a_k(t) cos(k π r), which naturally satisfies
    Neumann BC at r=0 (dT/dr=0) and with appropriate modes, Dirichlet at r=1.

    Uses basis cos((k+0.5)π r) for k=0,1,...,N-1 which gives T(1)=0
    when coefficients are chosen appropriately via DCT.

    Time stepping: semi-implicit — linear diffusion implicit, nonlinear
    correction explicit.
    """

    name = "spectral_cosine"

    def __init__(self, n_modes: int = 32):
        self.n_modes = n_modes

    def solve(self, T0, r, dt, t_end, alpha):
        nr = len(r)
        nt = int(round(t_end / dt))
        T_history = np.zeros((nt + 1, nr))
        T_history[0] = T0.copy()
        T = T0.copy()
        dr = r[1] - r[0]

        N = min(self.n_modes, nr)
        # Basis: phi_k(r) = cos((k+0.5)*pi*r), k=0,...,N-1
        # This gives phi_k(1) = cos((k+0.5)*pi) = 0 for all k
        # and phi_k'(0) = 0 for all k
        ks = np.arange(N)
        lam = ((ks + 0.5) * np.pi) ** 2  # eigenvalues of -d²/dr²

        # Build basis matrix: phi[k, j] = cos((k+0.5)*pi*r_j)
        phi = np.cos(np.outer((ks + 0.5) * np.pi, r))  # (N, nr)

        # Projection: use least squares or simple inner product
        # <phi_k, phi_k> = integral of cos²((k+0.5)πr) dr ≈ 0.5 for normalized
        phi_norm = np.sum(phi**2, axis=1) * dr  # (N,)

        for n in range(nt):
            # Forward transform: get coefficients
            a = np.sum(phi * T[np.newaxis, :], axis=1) * dr / phi_norm  # (N,)

            # Compute nonlinear diffusivity in physical space
            dTdr = np.zeros(nr)
            dTdr[1:-1] = (T[2:] - T[:-2]) / (2 * dr)
            chi_vals = self.chi(dTdr, alpha)

            # Nonlinear flux divergence in physical space
            # (1/r) d/dr(r chi dT/dr)
            flux = np.zeros(nr)
            for i in range(1, nr - 1):
                ri = r[i]
                chip = 0.5 * (chi_vals[i] + chi_vals[i + 1])
                chim = 0.5 * (chi_vals[i] + chi_vals[i - 1])
                rp = ri + 0.5 * dr
                rm = ri - 0.5 * dr
                flux[i] = (rp * chip * (T[i + 1] - T[i]) - rm * chim * (T[i] - T[i - 1])) / (ri * dr**2)
            # r=0: L'Hôpital
            flux[0] = 2.0 * chi_vals[0] * (T[1] - T[0]) * 2.0 / dr**2
            flux[-1] = 0.0

            # Project nonlinear residual: NL = flux - (-lam * a) in spectral space
            # The linear part would give flux = -lam_k * a_k for chi=1
            flux_hat = np.sum(phi * flux[np.newaxis, :], axis=1) * dr / phi_norm
            nl_corr = flux_hat + lam * a  # nonlinear correction

            # Semi-implicit time step: (1 + dt*lam) a_new = a + dt*(- lam*a + flux_hat)
            # = a + dt*flux_hat
            # Rearranged: a_new = (a + dt*nl_corr) / (1 + dt*lam)  [linear part implicit]
            # Actually: a_new = (a + dt*flux_hat) / (1 + dt*lam)
            a_new = (a + dt * flux_hat) / (1.0 + dt * lam)

            # Back to physical space
            T = np.dot(a_new, phi)  # (nr,)
            T[-1] = 0.0  # enforce Dirichlet BC
            T_history[n + 1] = T

        return T_history
