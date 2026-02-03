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
    """Nonlinear thermal diffusivity: chi = (|T'|-0.5)^alpha + 0.1 if |T'|>0.5, else 0.1."""
    abs_dTdr = np.abs(dTdr)
    result = np.full_like(abs_dTdr, 0.1)
    mask = abs_dTdr > 0.5
    result[mask] = (abs_dTdr[mask] - 0.5) ** alpha + 0.1
    return result


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


def extract_initial_features(
    T0: np.ndarray, r: np.ndarray, alpha: float,
    nr: int, dt: float, t_end: float, init_kind: str,
) -> dict:
    """Extract features from initial condition and problem parameters for ML selector.

    Returns dict with 14 features suitable for solver prediction.
    """
    feats = extract_all(T0, r, alpha)
    t_center = feats["T_center"]
    max_grad = feats["max_abs_gradient"]
    max_chi_val = feats["max_chi"]
    min_chi_val = feats["min_chi"]

    return {
        # Problem parameters (6)
        "alpha": alpha,
        "nr": nr,
        "dt": dt,
        "t_end": t_end,
        "init_gaussian": 1.0 if init_kind == "gaussian" else 0.0,
        "init_sharp": 1.0 if init_kind == "sharp" else 0.0,
        # Physical features from T0 (5)
        "max_abs_gradient": max_grad,
        "energy_content": feats["energy_content"],
        "max_chi": max_chi_val,
        "max_laplacian": feats["max_laplacian"],
        "T_center": t_center,
        # Derived (3)
        "gradient_sharpness": max_grad / t_center if t_center > 0 else 0.0,
        "chi_ratio": max_chi_val / min_chi_val if min_chi_val > 0 else 1.0,
        "problem_stiffness": alpha * max_grad,
    }
