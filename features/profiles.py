"""Initial temperature profile factory for radial heat equation."""

import numpy as np
from typing import Callable, Dict, Any


def parabolic(r: np.ndarray, A: float = 1.0, n: float = 2.0) -> np.ndarray:
    """Parabolic profile: T(r) = A * (1 - r^n).

    Args:
        r: Radial coordinate array [0, 1]
        A: Peak amplitude at r=0
        n: Power exponent (n=2 gives standard parabola)

    Returns:
        Temperature profile with T(0)=A, T(1)=0
    """
    return A * (1.0 - r**n)


def gaussian(r: np.ndarray, A: float = 1.0, sigma: float = 0.5) -> np.ndarray:
    """Gaussian profile normalized to satisfy T(1)=0.

    T(r) = A * (exp(-r^2/sigma^2) - exp(-1/sigma^2)) / (1 - exp(-1/sigma^2))

    Args:
        r: Radial coordinate array [0, 1]
        A: Peak amplitude at r=0
        sigma: Width parameter

    Returns:
        Temperature profile with T(0)=A, T(1)=0
    """
    exp_r = np.exp(-r**2 / sigma**2)
    exp_1 = np.exp(-1.0 / sigma**2)
    return A * (exp_r - exp_1) / (1.0 - exp_1)


def flat_top(r: np.ndarray, A: float = 1.0, w: float = 0.8,
             n: int = 4, m: int = 2) -> np.ndarray:
    """Flat-top profile with steep edge: T(r) = A * (1 - (r/w)^n)^m for r < w, else 0.

    Modified to ensure T(1)=0 by subtracting the value at r=1.

    Args:
        r: Radial coordinate array [0, 1]
        A: Peak amplitude at r=0
        w: Width parameter (transition point)
        n: Steepness of transition
        m: Edge smoothness

    Returns:
        Temperature profile with T(0)=A, T(1)=0
    """
    ratio = r / w
    base = np.maximum(0.0, 1.0 - ratio**n)
    profile = base**m
    # Normalize to ensure T(1)=0 and T(0)=A
    profile_at_1 = max(0.0, 1.0 - (1.0/w)**n)**m
    if profile_at_1 < 1.0:
        profile = A * (profile - profile_at_1) / (1.0 - profile_at_1)
    else:
        profile = A * profile
    return profile


def cosine(r: np.ndarray, A: float = 1.0) -> np.ndarray:
    """Cosine profile: T(r) = A * cos(pi*r/2).

    Args:
        r: Radial coordinate array [0, 1]
        A: Peak amplitude at r=0

    Returns:
        Temperature profile with T(0)=A, T(1)=0
    """
    return A * np.cos(np.pi * r / 2.0)


def linear(r: np.ndarray, A: float = 1.0) -> np.ndarray:
    """Linear profile: T(r) = A * (1 - r).

    Args:
        r: Radial coordinate array [0, 1]
        A: Peak amplitude at r=0

    Returns:
        Temperature profile with T(0)=A, T(1)=0
    """
    return A * (1.0 - r)


# Registry of available profiles
PROFILE_REGISTRY: Dict[str, Callable] = {
    "parabolic": parabolic,
    "gaussian": gaussian,
    "flat_top": flat_top,
    "cosine": cosine,
    "linear": linear,
}


# Default parameters for each profile
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "parabolic": {"A": 1.0, "n": 2.0},
    "gaussian": {"A": 1.0, "sigma": 0.5},
    "flat_top": {"A": 1.0, "w": 0.8, "n": 4, "m": 2},
    "cosine": {"A": 1.0},
    "linear": {"A": 1.0},
}


def make_profile(r: np.ndarray, name: str = "parabolic",
                 params: Dict[str, Any] = None) -> np.ndarray:
    """Create initial temperature profile by name.

    Args:
        r: Radial coordinate array [0, 1]
        name: Profile type name (parabolic, gaussian, flat_top, cosine, linear)
        params: Optional parameter overrides for the profile

    Returns:
        Temperature profile array

    Raises:
        ValueError: If profile name is not recognized
    """
    if name not in PROFILE_REGISTRY:
        available = ", ".join(PROFILE_REGISTRY.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")

    profile_fn = PROFILE_REGISTRY[name]
    default = DEFAULT_PARAMS[name].copy()
    if params:
        default.update(params)

    return profile_fn(r, **default)


def parse_profile_params(param_str: str) -> Dict[str, Any]:
    """Parse profile parameters from CLI string.

    Args:
        param_str: String like "sigma=0.3,A=2.0" or "n=4"

    Returns:
        Dictionary of parameter name -> value
    """
    if not param_str:
        return {}

    params = {}
    for item in param_str.split(","):
        if "=" in item:
            key, val = item.split("=", 1)
            key = key.strip()
            val = val.strip()
            # Try to parse as number
            try:
                if "." in val:
                    params[key] = float(val)
                else:
                    params[key] = int(val)
            except ValueError:
                params[key] = val
    return params


def get_available_profiles() -> list:
    """Return list of available profile names."""
    return list(PROFILE_REGISTRY.keys())
