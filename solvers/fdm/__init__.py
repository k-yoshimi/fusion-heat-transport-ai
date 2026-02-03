"""Finite Difference Method solvers."""

from solvers.fdm.implicit import ImplicitFDM
from solvers.fdm.compact4 import Compact4FDM

__all__ = ["ImplicitFDM", "Compact4FDM"]
