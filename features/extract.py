"""PDE feature extraction for radial temperature profiles."""

import numpy as np


def gradient(T: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Compute dT/dr using central differences (forward/backward at edges)."""
    dr = r[1] - r[0]
    dTdr = np.zeros_like(T)
    dTdr[1:-1] = (T[2:] - T[:-2]) / (2 * dr)
    dTdr[0] = (T[1] - T[0]) / dr
    dTdr[-1] = (T[-1] - T[-2]) / dr
    return dTdr


def laplacian(T: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Compute d²T/dr² using central differences."""
    dr = r[1] - r[0]
    d2T = np.zeros_like(T)
    d2T[1:-1] = (T[2:] - 2 * T[1:-1] + T[:-2]) / dr**2
    # Forward difference approximation at r=0
    if len(T) >= 3:
        d2T[0] = (T[2] - 2 * T[1] + T[0]) / dr**2
    d2T[-1] = d2T[-2]
    return d2T


def chi(dTdr: np.ndarray, alpha: float) -> np.ndarray:
    """Nonlinear thermal diffusivity: chi = 1 + alpha * |dT/dr|."""
    return 1.0 + alpha * np.abs(dTdr)


def max_abs_gradient(T: np.ndarray, r: np.ndarray) -> float:
    """Maximum absolute temperature gradient."""
    return float(np.max(np.abs(gradient(T, r))))


def zero_crossings(T: np.ndarray, r: np.ndarray) -> int:
    """Number of zero crossings of dT/dr."""
    dTdr = gradient(T, r)
    signs = np.sign(dTdr)
    crossings = np.sum(np.abs(np.diff(signs)) > 1)
    return int(crossings)


def energy_content(T: np.ndarray, r: np.ndarray) -> float:
    """Integral of T * r * dr (proportional to thermal energy in cylinder)."""
    return float(np.trapz(T * r, r))


def extract_all(T: np.ndarray, r: np.ndarray, alpha: float = 0.0) -> dict:
    """Extract all features from a temperature profile."""
    dTdr = gradient(T, r)
    d2T = laplacian(T, r)
    chi_vals = chi(dTdr, alpha)
    return {
        "max_abs_gradient": float(np.max(np.abs(dTdr))),
        "zero_crossings": zero_crossings(T, r),
        "energy_content": energy_content(T, r),
        "max_chi": float(np.max(chi_vals)),
        "min_chi": float(np.min(chi_vals)),
        "max_laplacian": float(np.max(np.abs(d2T))),
        "T_center": float(T[0]),
        "T_edge": float(T[-1]),
    }
