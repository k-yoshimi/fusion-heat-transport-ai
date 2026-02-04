"""Physics-only solver selector.

Selects the optimal solver based only on physical features of the problem,
excluding numerical parameters (dt, nr, t_end).
"""

import csv
import os
from typing import Dict, List, Optional
import numpy as np

from features.extract import (
    extract_all,
    half_max_radius,
    profile_centroid,
    gradient_slope,
    profile_width,
)
from policy.tree import NumpyDecisionTree


# Physics-only features (13 total, excludes dt, nr, t_end)
PHYSICS_FEATURE_NAMES = [
    "alpha",
    "max_abs_gradient",
    "energy_content",
    "max_chi",
    "max_laplacian",
    "T_center",
    "gradient_sharpness",
    "chi_ratio",
    "problem_stiffness",
    "half_max_radius",
    "profile_centroid",
    "gradient_slope",
    "profile_width",
]


def extract_physics_features(
    T0: np.ndarray,
    r: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    """Extract physics-only features from initial condition.

    Args:
        T0: Initial temperature profile
        r: Radial grid
        alpha: Nonlinearity parameter

    Returns:
        Dictionary with 13 physics features
    """
    feats = extract_all(T0, r, alpha)
    t_center = feats["T_center"]
    max_grad = feats["max_abs_gradient"]
    max_chi_val = feats["max_chi"]
    min_chi_val = feats["min_chi"]

    return {
        # Primary physical parameter
        "alpha": alpha,
        # Physical features from T0
        "max_abs_gradient": max_grad,
        "energy_content": feats["energy_content"],
        "max_chi": max_chi_val,
        "max_laplacian": feats["max_laplacian"],
        "T_center": t_center,
        # Derived physical features
        "gradient_sharpness": max_grad / t_center if t_center > 0 else 0.0,
        "chi_ratio": max_chi_val / min_chi_val if min_chi_val > 0 else 1.0,
        "problem_stiffness": alpha * max_grad,
        # Profile shape features
        "half_max_radius": half_max_radius(T0, r),
        "profile_centroid": profile_centroid(T0, r),
        "gradient_slope": gradient_slope(T0, r),
        "profile_width": profile_width(T0, r),
    }


class PhysicsSolverSelector:
    """Solver selector using only physics features.

    Uses a decision tree trained on physics features to predict
    the best solver for a given problem configuration.
    """

    def __init__(self, model_path: str = "data/physics_model.npz"):
        """Initialize selector.

        Args:
            model_path: Path to trained model file
        """
        self.model_path = model_path
        self.tree: Optional[NumpyDecisionTree] = None
        self._load_model()

    def _load_model(self) -> bool:
        """Load trained model if available.

        Returns:
            True if model loaded successfully
        """
        if os.path.isfile(self.model_path):
            try:
                self.tree = NumpyDecisionTree()
                self.tree.load(self.model_path)
                return True
            except Exception:
                self.tree = None
        return False

    def predict(
        self,
        T0: np.ndarray,
        r: np.ndarray,
        alpha: float,
    ) -> str:
        """Predict best solver using physics features.

        Args:
            T0: Initial temperature profile
            r: Radial grid
            alpha: Nonlinearity parameter

        Returns:
            Name of predicted best solver
        """
        if self.tree is None:
            return self._rule_based_fallback(T0, r, alpha)

        feats = extract_physics_features(T0, r, alpha)
        X = np.array([[feats[f] for f in PHYSICS_FEATURE_NAMES]])
        return self.tree.predict(X)[0]

    def _rule_based_fallback(
        self,
        T0: np.ndarray,
        r: np.ndarray,
        alpha: float,
    ) -> str:
        """Rule-based solver selection when model unavailable.

        Uses heuristics based on problem characteristics.

        Args:
            T0: Initial temperature profile
            r: Radial grid
            alpha: Nonlinearity parameter

        Returns:
            Name of recommended solver
        """
        feats = extract_physics_features(T0, r, alpha)

        # High nonlinearity or stiffness -> implicit methods
        if alpha > 1.0 or feats["problem_stiffness"] > 1.5:
            return "implicit_fdm"

        # Smooth profiles with low alpha -> spectral methods
        if alpha <= 0.5 and feats["chi_ratio"] < 2.0:
            if feats["max_abs_gradient"] < 2.0:
                return "spectral_cosine"
            return "chebyshev_spectral"

        # Sharp gradients -> high-order FDM
        if feats["gradient_sharpness"] > 3.0:
            return "compact4_fdm"

        # Conservation important -> FVM
        if feats["energy_content"] > 0.3:
            return "cell_centered_fvm"

        # Default: robust implicit FDM
        return "implicit_fdm"

    def predict_with_confidence(
        self,
        T0: np.ndarray,
        r: np.ndarray,
        alpha: float,
    ) -> Dict[str, float]:
        """Predict solver with confidence scores.

        Args:
            T0: Initial temperature profile
            r: Radial grid
            alpha: Nonlinearity parameter

        Returns:
            Dictionary mapping solver names to confidence scores
        """
        if self.tree is None:
            solver = self._rule_based_fallback(T0, r, alpha)
            return {solver: 1.0}

        feats = extract_physics_features(T0, r, alpha)
        X = np.array([[feats[f] for f in PHYSICS_FEATURE_NAMES]])

        # Get prediction and leaf info
        prediction = self.tree.predict(X)[0]

        # Simple confidence: 1.0 for predicted, 0.0 for others
        # Could enhance with tree probability estimates
        return {prediction: 1.0}


def train_physics_model(
    data_path: str = "data/training_data.csv",
    output_path: str = "data/physics_model.npz",
    max_depth: int = 5,
) -> NumpyDecisionTree:
    """Train physics-only solver selector model.

    Args:
        data_path: Path to training data CSV
        output_path: Path to save trained model
        max_depth: Maximum tree depth

    Returns:
        Trained decision tree
    """
    # Load CSV
    with open(data_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No training data in {data_path}")

    # Extract physics features only
    X = np.array([
        [float(row[f]) for f in PHYSICS_FEATURE_NAMES]
        for row in rows
    ])
    y = np.array([row["best_solver"] for row in rows])

    # Train decision tree
    tree = NumpyDecisionTree(max_depth=max_depth)
    tree.fit(X, y)

    # Training accuracy
    preds = tree.predict(X)
    acc = np.mean(preds == y)
    print(f"Physics model training accuracy: {acc:.1%} ({int(acc * len(y))}/{len(y)})")

    # Save model
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tree.save(output_path)
    print(f"Physics model saved to {output_path}")

    return tree


def select_with_physics(
    T0: np.ndarray,
    r: np.ndarray,
    alpha: float,
    model_path: str = "data/physics_model.npz",
) -> str:
    """Convenience function to select solver using physics features.

    Args:
        T0: Initial temperature profile
        r: Radial grid
        alpha: Nonlinearity parameter
        model_path: Path to trained model

    Returns:
        Name of predicted best solver
    """
    selector = PhysicsSolverSelector(model_path)
    return selector.predict(T0, r, alpha)


def generate_physics_training_data(
    output_path: str = "data/physics_training_data.csv",
    alpha_list: Optional[List[float]] = None,
    profile_list: Optional[List[tuple]] = None,
):
    """Generate training data for physics-only model.

    Runs full benchmark for each configuration and labels with best solver.

    Args:
        output_path: Path to save training CSV
        alpha_list: List of alpha values
        profile_list: List of (profile_name, params) tuples
    """
    import time
    from solvers.fdm.implicit import ImplicitFDM
    from solvers.spectral.cosine import CosineSpectral
    from solvers.pinn.stub import PINNStub
    from solvers.fdm.compact4 import Compact4FDM
    from solvers.fdm.imex import IMEXFDM
    from solvers.fem.p2_fem import P2FEM
    from solvers.fvm.cell_centered import CellCenteredFVM
    from solvers.spectral.chebyshev import ChebyshevSpectral
    from metrics.accuracy import compute_errors
    from features.profiles import make_profile
    from policy.select import select_best

    if alpha_list is None:
        alpha_list = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]

    if profile_list is None:
        profile_list = [
            ("parabolic", {"n": 2.0}),
            ("parabolic", {"n": 4.0}),
            ("gaussian", {"sigma": 0.3}),
            ("gaussian", {"sigma": 0.5}),
            ("flat_top", {"w": 0.8, "n": 4, "m": 2}),
            ("cosine", {}),
            ("linear", {}),
        ]

    # Fixed numerical parameters for fair comparison
    nr = 51
    dt = 0.001
    t_end = 0.1

    solvers = [
        ImplicitFDM(),
        CosineSpectral(),
        PINNStub(),
        Compact4FDM(),
        IMEXFDM(),
        P2FEM(),
        CellCenteredFVM(),
        ChebyshevSpectral(),
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    rows = []

    total = len(alpha_list) * len(profile_list)
    count = 0

    for profile_name, profile_params in profile_list:
        for alpha in alpha_list:
            count += 1
            r = np.linspace(0, 1, nr)
            T0 = make_profile(r, profile_name, profile_params)

            feats = extract_physics_features(T0, r, alpha)

            # Compute reference solution
            try:
                nr_fine = 4 * nr - 3
                r_fine = np.linspace(0, 1, nr_fine)
                T0_fine = np.interp(r_fine, r, T0)
                ref_solver = ImplicitFDM()
                T_ref_full = ref_solver.solve(T0_fine, r_fine, dt / 4.0, t_end, alpha)
                indices = np.linspace(0, nr_fine - 1, nr).astype(int)
                T_ref = T_ref_full[:, indices]
            except Exception:
                continue

            # Benchmark all solvers
            results = []
            for s in solvers:
                try:
                    t0 = time.perf_counter()
                    T_hist = s.solve(T0.copy(), r, dt, t_end, alpha)
                    wall = time.perf_counter() - t0

                    nt_ref = T_ref.shape[0]
                    nt_sol = T_hist.shape[0]
                    if nt_sol != nt_ref:
                        T_hist_cmp = np.stack([T_hist[0], T_hist[-1]])
                        T_ref_cmp = np.stack([T_ref[0], T_ref[-1]])
                    else:
                        T_hist_cmp = T_hist
                        T_ref_cmp = T_ref

                    errs = compute_errors(T_hist_cmp, T_ref_cmp, r)
                    results.append({
                        "name": s.name,
                        "l2_error": errs["l2"],
                        "wall_time": wall,
                    })
                except Exception:
                    results.append({
                        "name": s.name,
                        "l2_error": float("nan"),
                        "wall_time": 0.0,
                    })

            try:
                best = select_best(results, lam=0.1)
                label = best["name"]
            except ValueError:
                continue

            row = {**feats, "best_solver": label}
            rows.append(row)

            if count % 5 == 0:
                print(f"  [{count}/{total}] profile={profile_name} "
                      f"alpha={alpha} -> {label}")

    # Write CSV
    fieldnames = PHYSICS_FEATURE_NAMES + ["best_solver"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} physics training samples to {output_path}")
    return output_path
