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


def half_max_radius(T: np.ndarray, r: np.ndarray) -> float:
    """Find radius where T(r) = T_max / 2.

    Returns the interpolated r value where the profile crosses half its maximum.
    If no crossing exists, returns 1.0.
    """
    T_max = np.max(T)
    if T_max <= 0:
        return 1.0
    threshold = T_max / 2.0
    # Find first crossing from above
    above = T >= threshold
    if not np.any(above) or np.all(above):
        return 1.0
    # Find transition index
    transitions = np.where(np.diff(above.astype(int)) == -1)[0]
    if len(transitions) == 0:
        return 1.0
    idx = transitions[0]
    # Linear interpolation
    if idx + 1 < len(r):
        t0, t1 = T[idx], T[idx + 1]
        r0, r1 = r[idx], r[idx + 1]
        if t0 != t1:
            return float(r0 + (threshold - t0) * (r1 - r0) / (t1 - t0))
    return float(r[idx])


def profile_centroid(T: np.ndarray, r: np.ndarray) -> float:
    """Compute profile centroid: integral(T*r^2*dr) / integral(T*r*dr).

    Represents the "center of mass" radial position of the temperature profile.
    """
    numerator = np.trapz(T * r**2, r)
    denominator = np.trapz(T * r, r)
    if denominator <= 0:
        return 0.5
    return float(numerator / denominator)


def gradient_slope(T: np.ndarray, r: np.ndarray) -> float:
    """Compute slope of |dT/dr| vs r using linear regression.

    Positive slope indicates gradient magnitude increases with r.
    Negative slope indicates gradient magnitude decreases with r.
    """
    dTdr = gradient(T, r)
    abs_dTdr = np.abs(dTdr)
    # Linear regression: slope = cov(r, |dT/dr|) / var(r)
    r_mean = np.mean(r)
    g_mean = np.mean(abs_dTdr)
    cov = np.mean((r - r_mean) * (abs_dTdr - g_mean))
    var = np.var(r)
    if var < 1e-12:
        return 0.0
    return float(cov / var)


def profile_width(T: np.ndarray, r: np.ndarray) -> float:
    """Compute effective profile width: sqrt(integral(T*r^2*dr) / integral(T*dr)).

    A measure of the radial extent of the temperature profile.
    """
    numerator = np.trapz(T * r**2, r)
    denominator = np.trapz(T, r)
    if denominator <= 0 or numerator < 0:
        return 0.5
    return float(np.sqrt(numerator / denominator))


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
    nr: int, dt: float, t_end: float,
) -> dict:
    """Extract features from initial condition and problem parameters for ML selector.

    Returns dict with 16 features suitable for solver prediction.
    """
    feats = extract_all(T0, r, alpha)
    t_center = feats["T_center"]
    max_grad = feats["max_abs_gradient"]
    max_chi_val = feats["max_chi"]
    min_chi_val = feats["min_chi"]

    return {
        # Problem parameters (4)
        "alpha": alpha,
        "nr": nr,
        "dt": dt,
        "t_end": t_end,
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
        # Profile shape features (4)
        "half_max_radius": half_max_radius(T0, r),
        "profile_centroid": profile_centroid(T0, r),
        "gradient_slope": gradient_slope(T0, r),
        "profile_width": profile_width(T0, r),
    }
