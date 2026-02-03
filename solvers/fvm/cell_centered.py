"""Cell-Centered Finite Volume Method solver for radial heat transport."""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from solvers.base import SolverBase


class CellCenteredFVM(SolverBase):
    """Cell-centered Finite Volume Method solver in radial coordinates.

    Uses implicit time stepping with cell-centered discretization.

    Conservation form of the radial heat equation:
        ∂T/∂t = (1/r) ∂/∂r (r χ ∂T/∂r)

    Integrating over control volume [r_{i-1/2}, r_{i+1/2}]:
        V_i dT_i/dt = r_{i+1/2} χ_{i+1/2} (T_{i+1}-T_i)/Δr
                    - r_{i-1/2} χ_{i-1/2} (T_i-T_{i-1})/Δr

    where V_i = (r_{i+1/2}² - r_{i-1/2}²)/2 is the control volume.

    This method ensures strict conservation of the integral quantity.
    """

    name = "cell_centered_fvm"

    def __init__(self, flux_scheme: str = "harmonic"):
        """Initialize cell-centered FVM solver.

        Args:
            flux_scheme: Scheme for face diffusivity interpolation.
                        "harmonic" - harmonic mean (better for discontinuous chi)
                        "arithmetic" - arithmetic mean
        """
        self.flux_scheme = flux_scheme

    def _compute_face_chi(self, chi, scheme="harmonic"):
        """Compute diffusivity at cell faces.

        Args:
            chi: Diffusivity at cell centers (nr,)
            scheme: Interpolation scheme

        Returns:
            chi_face: Diffusivity at faces (nr-1,)
                     chi_face[i] = chi at face between cell i and i+1
        """
        chi_L = chi[:-1]
        chi_R = chi[1:]

        if scheme == "harmonic":
            # Harmonic mean - better for discontinuous diffusivity
            # Avoids issues when chi varies significantly
            chi_face = 2 * chi_L * chi_R / (chi_L + chi_R + 1e-15)
        else:
            # Arithmetic mean
            chi_face = 0.5 * (chi_L + chi_R)

        return chi_face

    def solve(self, T0, r, dt, t_end, alpha):
        """Solve the heat equation using cell-centered FVM.

        Args:
            T0: Initial temperature profile (nr,)
            r: Radial grid (nr,)
            dt: Time step
            t_end: Final time
            alpha: Nonlinearity parameter

        Returns:
            T_history: Temperature history (nt+1, nr)
        """
        nr = len(r)
        dr = r[1] - r[0]
        nt = int(round(t_end / dt))

        # Cell face positions
        # r_face[i] is the face between cell i and i+1
        r_face = np.zeros(nr + 1)
        r_face[0] = 0.0  # Left boundary
        r_face[1:-1] = 0.5 * (r[:-1] + r[1:])
        r_face[-1] = r[-1]  # Right boundary

        # Control volumes: V_i = (r_{i+1/2}² - r_{i-1/2}²) / 2
        # This is the radial integral weight
        V = 0.5 * (r_face[1:]**2 - r_face[:-1]**2)
        V[0] = max(V[0], 1e-15)  # Avoid division by zero at r=0

        # Initialize
        T_history = np.zeros((nt + 1, nr))
        T_history[0] = T0.copy()
        T = T0.copy()

        # Time stepping (implicit)
        for n in range(nt):
            # Compute gradient for chi (cell-centered)
            dTdr = np.zeros(nr)
            dTdr[0] = 0.0  # Neumann BC at r=0
            dTdr[1:-1] = (T[2:] - T[:-2]) / (2 * dr)
            dTdr[-1] = (T[-1] - T[-2]) / dr

            # Compute chi at cell centers
            chi = self.chi(dTdr, alpha)

            # Compute chi at faces
            chi_face = self._compute_face_chi(chi, self.flux_scheme)

            # Build coefficient matrix for implicit scheme
            # Flux from cell i to i+1: F_{i+1/2} = r_{i+1/2} χ_{i+1/2} (T_{i+1} - T_i) / dr

            # Coefficient for T_i from face i+1/2: -r_{i+1/2} χ_{i+1/2} / (V_i * dr)
            # Coefficient for T_i from face i-1/2: -r_{i-1/2} χ_{i-1/2} / (V_i * dr)

            # Face coefficients (internal faces only: indices 1 to nr-1)
            # r_face[1:nr] are the internal faces
            face_coeff = r_face[1:-1] * chi_face / dr  # (nr-1,)

            # Build tridiagonal matrix
            # Main diagonal
            diag = np.ones(nr)
            # Lower diagonal (coupling to T_{i-1})
            lower = np.zeros(nr - 1)
            # Upper diagonal (coupling to T_{i+1})
            upper = np.zeros(nr - 1)

            # Interior cells (i = 1 to nr-2)
            for i in range(1, nr - 1):
                # From face i-1/2 (index i-1 in face_coeff)
                coeff_im = face_coeff[i - 1] / V[i]
                # From face i+1/2 (index i in face_coeff)
                coeff_ip = face_coeff[i] / V[i]

                diag[i] = 1.0 + dt * (coeff_im + coeff_ip)
                lower[i - 1] = -dt * coeff_im
                upper[i] = -dt * coeff_ip

            # Cell 0 (r=0): Neumann BC ∂T/∂r = 0
            # Flux at r=0 face is zero (by symmetry)
            # Only flux from face 0+1/2
            coeff_0p = face_coeff[0] / V[0]
            diag[0] = 1.0 + dt * coeff_0p
            upper[0] = -dt * coeff_0p

            # Cell nr-1 (r=1): Dirichlet BC T = 0
            diag[-1] = 1.0
            lower[-1] = 0.0

            # Assemble sparse matrix
            A = diags([lower, diag, upper], [-1, 0, 1], format='csr')

            # Right-hand side
            b = T.copy()
            b[-1] = 0.0  # Dirichlet BC

            # Solve
            T = spsolve(A, b)
            T_history[n + 1] = T

        return T_history
