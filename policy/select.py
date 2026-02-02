"""Solver selection policy.

Picks the solver with the lowest combined score:
    score = error + λ * wall_time

where error is the L2 error vs a reference solution and wall_time
is in seconds. λ controls the time penalty (default 0.1).
"""


def select_best(results: list[dict], lam: float = 0.1) -> dict:
    """Select the best solver from benchmark results.

    Args:
        results: List of dicts with keys 'name', 'l2_error', 'wall_time'.
            Entries with NaN errors are excluded.
        lam: Time penalty weight.

    Returns:
        The result dict of the best solver.
    """
    import math

    valid = [r for r in results if not math.isnan(r.get("l2_error", float("nan")))]
    if not valid:
        raise ValueError("No valid solver results to select from.")

    scored = [(r, r["l2_error"] + lam * r["wall_time"]) for r in valid]
    scored.sort(key=lambda x: x[1])
    return scored[0][0]


def select_with_ml(
    T0, r, alpha: float, nr: int, dt: float, t_end: float,
    init_kind: str, model_path: str = "data/solver_model.npz",
) -> str:
    """Predict best solver using trained decision tree.

    Returns solver name string (e.g. 'implicit_fdm').
    """
    import numpy as np
    from features.extract import extract_initial_features
    from policy.tree import NumpyDecisionTree
    from policy.train import FEATURE_NAMES

    feats = extract_initial_features(T0, r, alpha, nr, dt, t_end, init_kind)
    X = np.array([[feats[f] for f in FEATURE_NAMES]])

    tree = NumpyDecisionTree()
    tree.load(model_path)
    return tree.predict(X)[0]
