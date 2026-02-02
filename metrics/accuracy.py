"""Error metrics for solver comparison."""

import numpy as np


def l2_error(T: np.ndarray, T_ref: np.ndarray, r: np.ndarray) -> float:
    """L2 error between T and T_ref, weighted by r for cylindrical geometry."""
    diff = T - T_ref
    return float(np.sqrt(np.trapz(diff**2 * r, r) / np.trapz(T_ref**2 * r + 1e-30, r)))


def linf_error(T: np.ndarray, T_ref: np.ndarray) -> float:
    """L-infinity (max absolute) error."""
    return float(np.max(np.abs(T - T_ref)))


def compute_errors(T_history: np.ndarray, T_ref: np.ndarray, r: np.ndarray) -> dict:
    """Compute errors at the final time step."""
    T_final = T_history[-1]
    T_ref_final = T_ref[-1]
    return {
        "l2": l2_error(T_final, T_ref_final, r),
        "linf": linf_error(T_final, T_ref_final),
    }
