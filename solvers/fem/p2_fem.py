"""P2 (Quadratic) Finite Element Method solver for radial heat transport."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from solvers.base import SolverBase


class P2FEM(SolverBase):
    """Quadratic (P2) Finite Element solver in radial coordinates.

    Uses Crank-Nicolson time stepping with quadratic shape functions.
    Optimized with vectorized assembly over all elements.

    The weak form of the radial heat equation:
        ∫ r ∂T/∂t v dr = -∫ r χ ∂T/∂r ∂v/∂r dr
    """

    name = "p2_fem"

    def __init__(self, n_gauss: int = 3):
        """Initialize P2 FEM solver."""
        self.n_gauss = n_gauss
        if n_gauss == 2:
            self.gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
            self.gauss_wts = np.array([1.0, 1.0])
        else:
            self.gauss_pts = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
            self.gauss_wts = np.array([5/9, 8/9, 5/9])

    def _shape_functions_vectorized(self, xi):
        """P2 shape functions for multiple xi values at once.

        Args:
            xi: Local coordinates, shape (n_gauss,)

        Returns:
            N: Shape functions, shape (n_gauss, 3)
            dN: Shape function derivatives, shape (n_gauss, 3)
        """
        N = np.column_stack([
            xi * (xi - 1) / 2,
            (1 - xi) * (1 + xi),
            xi * (xi + 1) / 2
        ])
        dN = np.column_stack([
            xi - 0.5,
            -2 * xi,
            xi + 0.5
        ])
        return N, dN

    def _create_p2_mesh(self, r):
        """Create P2 mesh from original grid."""
        nr = len(r)
        n_elem = nr - 1
        n_nodes = 2 * nr - 1

        r_p2 = np.zeros(n_nodes)
        r_p2[::2] = r
        r_p2[1::2] = 0.5 * (r[:-1] + r[1:])

        elements = np.zeros((n_elem, 3), dtype=int)
        elements[:, 0] = np.arange(0, 2*n_elem, 2)
        elements[:, 1] = np.arange(1, 2*n_elem + 1, 2)
        elements[:, 2] = np.arange(2, 2*n_elem + 2, 2)

        return r_p2, elements

    def _assemble_matrices_vectorized(self, r_p2, elements, chi_nodal):
        """Vectorized assembly of mass and stiffness matrices."""
        n_nodes = len(r_p2)
        n_elem = len(elements)

        # Get element node coordinates and chi values: shape (n_elem, 3)
        r_e = r_p2[elements]
        chi_e = chi_nodal[elements]

        # Element sizes and Jacobians
        h_e = r_e[:, 2] - r_e[:, 0]  # (n_elem,)
        J = h_e / 2  # (n_elem,)

        # Shape functions at Gauss points: N (n_gauss, 3), dN (n_gauss, 3)
        N, dN = self._shape_functions_vectorized(self.gauss_pts)
        n_gauss = len(self.gauss_pts)

        # Initialize local matrices storage: (n_elem, 3, 3)
        M_local = np.zeros((n_elem, 3, 3))
        K_local = np.zeros((n_elem, 3, 3))

        # Loop over Gauss points (small loop, 2-3 iterations)
        for q in range(n_gauss):
            w = self.gauss_wts[q]
            Nq = N[q]  # (3,)
            dNq = dN[q]  # (3,)

            # Physical coordinates at quad point: r_q[e] = sum_i N_i * r_e[e,i]
            r_q = np.dot(r_e, Nq)  # (n_elem,)

            # dN/dr = dN/dxi / J
            dNdr = dNq / J[:, np.newaxis]  # (n_elem, 3)

            # chi at quad points
            chi_q = np.dot(chi_e, Nq)  # (n_elem,)

            # Effective r (avoid r=0)
            r_eff = np.maximum(r_q, 1e-10)

            # Weight factor: w * J * r_eff
            wJr = w * J * r_eff  # (n_elem,)

            # Mass: wJr * outer(N, N)
            M_local += wJr[:, np.newaxis, np.newaxis] * np.outer(Nq, Nq)

            # Stiffness: wJr * chi * outer(dNdr, dNdr)
            # dNdr is (n_elem, 3), need outer product for each element
            K_contrib = (wJr * chi_q)[:, np.newaxis, np.newaxis] * (
                dNdr[:, :, np.newaxis] * dNdr[:, np.newaxis, :]
            )
            K_local += K_contrib

        # Assemble into global sparse matrix
        # Use COO format for efficient assembly
        rows = elements[:, :, np.newaxis].repeat(3, axis=2).flatten()
        cols = elements[:, np.newaxis, :].repeat(3, axis=1).flatten()

        M_data = M_local.flatten()
        K_data = K_local.flatten()

        M = csr_matrix((M_data, (rows, cols)), shape=(n_nodes, n_nodes))
        K = csr_matrix((K_data, (rows, cols)), shape=(n_nodes, n_nodes))

        return M, K

    def solve(self, T0, r, dt, t_end, alpha):
        """Solve the heat equation using P2 FEM."""
        nr = len(r)
        nt = int(round(t_end / dt))

        r_p2, elements = self._create_p2_mesh(r)
        n_nodes = len(r_p2)

        T_history = np.zeros((nt + 1, nr))
        T_history[0] = T0.copy()

        T = np.interp(r_p2, r, T0)

        # Precompute dr for gradient calculation
        dr_p2 = np.diff(r_p2)

        for n in range(nt):
            # Compute gradient (vectorized)
            dTdr = np.zeros(n_nodes)
            dTdr[0] = 0.0
            dTdr[1:-1] = (T[2:] - T[:-2]) / (r_p2[2:] - r_p2[:-2])
            dTdr[-1] = (T[-1] - T[-2]) / (r_p2[-1] - r_p2[-2])

            chi_nodal = self.chi(dTdr, alpha)

            # Assemble matrices
            M, K = self._assemble_matrices_vectorized(r_p2, elements, chi_nodal)

            # Crank-Nicolson: (M + dt/2 K) T^{n+1} = (M - dt/2 K) T^n
            A = M + 0.5 * dt * K
            b = (M - 0.5 * dt * K) @ T

            # Dirichlet BC at r=1
            A = A.tolil()
            A[-1, :] = 0
            A[-1, -1] = 1.0
            b[-1] = 0.0
            A = A.tocsr()

            T = spsolve(A, b)
            T_history[n + 1] = np.interp(r, r_p2, T)

        return T_history
