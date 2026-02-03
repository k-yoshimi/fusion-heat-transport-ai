"""P2 (Quadratic) Finite Element Method solver for radial heat transport."""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from solvers.base import SolverBase


class P2FEM(SolverBase):
    """Quadratic (P2) Finite Element solver in radial coordinates.

    Uses Crank-Nicolson time stepping with quadratic shape functions.

    The weak form of the radial heat equation:
        ∫ r ∂T/∂t v dr = -∫ r χ ∂T/∂r ∂v/∂r dr

    Handles r=0 singularity naturally through the weak formulation.
    """

    name = "p2_fem"

    def __init__(self, n_gauss: int = 3):
        """Initialize P2 FEM solver.

        Args:
            n_gauss: Number of Gauss quadrature points per element.
        """
        self.n_gauss = n_gauss
        # Gauss quadrature points and weights on [-1, 1]
        if n_gauss == 2:
            self.gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
            self.gauss_wts = np.array([1.0, 1.0])
        elif n_gauss == 3:
            self.gauss_pts = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
            self.gauss_wts = np.array([5/9, 8/9, 5/9])
        else:
            # Default to 3-point
            self.gauss_pts = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
            self.gauss_wts = np.array([5/9, 8/9, 5/9])

    def _shape_functions(self, xi):
        """P2 shape functions on reference element [-1, 1].

        Args:
            xi: Local coordinate in [-1, 1]

        Returns:
            N: Shape function values [N1, N2, N3]
            dN: Shape function derivatives [dN1/dxi, dN2/dxi, dN3/dxi]
        """
        N = np.array([
            xi * (xi - 1) / 2,    # Left node
            (1 - xi) * (1 + xi),  # Center node
            xi * (xi + 1) / 2     # Right node
        ])
        dN = np.array([
            xi - 0.5,             # dN1/dxi
            -2 * xi,              # dN2/dxi
            xi + 0.5              # dN3/dxi
        ])
        return N, dN

    def _create_p2_mesh(self, r):
        """Create P2 mesh from original grid.

        For P2 elements, we need midpoint nodes. If original grid has nr points,
        we create (nr-1) elements, each with 3 nodes (2 corners + 1 midpoint).
        Total P2 nodes = 2*(nr-1) + 1 = 2*nr - 1.

        Args:
            r: Original radial grid (nr,)

        Returns:
            r_p2: P2 mesh nodes (2*nr-1,)
            elements: Element connectivity, shape (n_elem, 3)
        """
        nr = len(r)
        n_elem = nr - 1
        n_nodes = 2 * nr - 1

        # Create P2 nodes (add midpoints)
        r_p2 = np.zeros(n_nodes)
        for i in range(nr - 1):
            r_p2[2*i] = r[i]
            r_p2[2*i + 1] = 0.5 * (r[i] + r[i+1])
        r_p2[-1] = r[-1]

        # Element connectivity (local to global node mapping)
        # Element e has nodes: [2*e, 2*e+1, 2*e+2]
        elements = np.zeros((n_elem, 3), dtype=int)
        for e in range(n_elem):
            elements[e] = [2*e, 2*e+1, 2*e+2]

        return r_p2, elements

    def _assemble_matrices(self, r_p2, elements, chi_nodal):
        """Assemble mass and stiffness matrices.

        Args:
            r_p2: P2 mesh nodes
            elements: Element connectivity
            chi_nodal: Diffusivity at each node

        Returns:
            M: Mass matrix (sparse)
            K: Stiffness matrix (sparse)
        """
        n_nodes = len(r_p2)
        n_elem = len(elements)

        M = lil_matrix((n_nodes, n_nodes))
        K = lil_matrix((n_nodes, n_nodes))

        for e in range(n_elem):
            nodes = elements[e]
            r_e = r_p2[nodes]  # [r_left, r_mid, r_right]
            chi_e = chi_nodal[nodes]

            # Element length and Jacobian
            h_e = r_e[2] - r_e[0]
            J = h_e / 2  # Jacobian of mapping from [-1,1] to element

            # Local matrices
            M_e = np.zeros((3, 3))
            K_e = np.zeros((3, 3))

            # Gauss quadrature
            for q, (xi, w) in enumerate(zip(self.gauss_pts, self.gauss_wts)):
                N, dN = self._shape_functions(xi)

                # Physical coordinate and derivatives
                r_q = np.dot(N, r_e)
                dNdr = dN / J  # dN/dr = dN/dxi * dxi/dr = dN/dxi / J

                # Interpolate chi at quadrature point
                chi_q = np.dot(N, chi_e)

                # Handle r=0 singularity: use r=epsilon for stability
                r_eff = max(r_q, 1e-10)

                # Mass matrix: ∫ r N_i N_j dr
                M_e += w * J * r_eff * np.outer(N, N)

                # Stiffness matrix: ∫ r χ dN_i/dr dN_j/dr dr
                K_e += w * J * r_eff * chi_q * np.outer(dNdr, dNdr)

            # Assemble into global matrices
            for i in range(3):
                for j in range(3):
                    M[nodes[i], nodes[j]] += M_e[i, j]
                    K[nodes[i], nodes[j]] += K_e[i, j]

        return csr_matrix(M), csr_matrix(K)

    def _interpolate_to_p2(self, T, r, r_p2):
        """Interpolate temperature from original grid to P2 nodes."""
        return np.interp(r_p2, r, T)

    def _restrict_to_original(self, T_p2, r, r_p2):
        """Restrict P2 solution back to original grid nodes."""
        return np.interp(r, r_p2, T_p2)

    def solve(self, T0, r, dt, t_end, alpha):
        """Solve the heat equation using P2 FEM.

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
        nt = int(round(t_end / dt))

        # Create P2 mesh
        r_p2, elements = self._create_p2_mesh(r)
        n_nodes = len(r_p2)

        # Initialize
        T_history = np.zeros((nt + 1, nr))
        T_history[0] = T0.copy()

        # Interpolate initial condition to P2 mesh
        T = self._interpolate_to_p2(T0, r, r_p2)

        # Time stepping (Crank-Nicolson)
        for n in range(nt):
            # Compute gradient for chi
            # Use finite differences on P2 mesh
            dTdr = np.zeros(n_nodes)
            dTdr[0] = 0.0  # Neumann BC at r=0
            dTdr[1:-1] = (T[2:] - T[:-2]) / (r_p2[2:] - r_p2[:-2])
            dTdr[-1] = (T[-1] - T[-2]) / (r_p2[-1] - r_p2[-2])

            # Compute chi at each node
            chi_nodal = self.chi(dTdr, alpha)

            # Assemble matrices
            M, K = self._assemble_matrices(r_p2, elements, chi_nodal)

            # Crank-Nicolson: (M + dt/2 K) T^{n+1} = (M - dt/2 K) T^n
            A = M + 0.5 * dt * K
            b = (M - 0.5 * dt * K) @ T

            # Apply Dirichlet BC at r=1 (last node)
            A = A.tolil()
            A[-1, :] = 0
            A[-1, -1] = 1.0
            b[-1] = 0.0
            A = A.tocsr()

            # Solve
            T = spsolve(A, b)

            # Restrict to original grid and save
            T_history[n + 1] = self._restrict_to_original(T, r, r_p2)

        return T_history
