"""Abstract base class for heat transport solvers."""

from abc import ABC, abstractmethod
import numpy as np


class SolverBase(ABC):
    """Base class for 1D radial heat transport solvers.

    Solves: ∂T/∂t = (1/r) ∂/∂r (r χ(|∂T/∂r|) ∂T/∂r)
    where χ(|T'|) = (|T'| - 0.5)^α + 0.1  if |T'| > 0.5, else 0.1

    BCs: Neumann ∂T/∂r=0 at r=0, Dirichlet T=0 at r=1.
    """

    name: str = "base"

    @abstractmethod
    def solve(
        self,
        T0: np.ndarray,
        r: np.ndarray,
        dt: float,
        t_end: float,
        alpha: float,
    ) -> np.ndarray:
        """Solve the PDE.

        Args:
            T0: Initial temperature profile, shape (nr,).
            r: Radial grid, shape (nr,), from 0 to 1.
            dt: Time step.
            t_end: Final time.
            alpha: Nonlinearity parameter for χ.

        Returns:
            T_history: shape (nt+1, nr), temperature at each saved time step.
        """

    @staticmethod
    def make_grid(nr: int) -> np.ndarray:
        """Create uniform radial grid from 0 to 1 with nr points."""
        return np.linspace(0, 1, nr)

    @staticmethod
    def chi(dTdr: np.ndarray, alpha: float) -> np.ndarray:
        """Nonlinear diffusivity χ(|T'|) = (|T'|-0.5)^α + 0.1 if |T'|>0.5, else 0.1."""
        abs_dTdr = np.abs(dTdr)
        result = np.full_like(abs_dTdr, 0.1)
        mask = abs_dTdr > 0.5
        result[mask] = (abs_dTdr[mask] - 0.5) ** alpha + 0.1
        return result
