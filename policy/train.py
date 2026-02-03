"""Training data generation and model training for ML solver selector."""

import csv
import os
import time
import numpy as np

from solvers.fdm.implicit import ImplicitFDM
from solvers.spectral.cosine import CosineSpectral
from solvers.pinn.stub import PINNStub
from metrics.accuracy import compute_errors
from features.extract import extract_initial_features
from features.profiles import make_profile, PROFILE_REGISTRY, DEFAULT_PARAMS
from policy.select import select_best
from policy.tree import NumpyDecisionTree


FEATURE_NAMES = [
    "alpha", "nr", "dt", "t_end",
    "max_abs_gradient", "energy_content", "max_chi", "max_laplacian",
    "T_center", "gradient_sharpness", "chi_ratio", "problem_stiffness",
    "half_max_radius", "profile_centroid", "gradient_slope", "profile_width",
]


def _make_initial(r, profile_name="parabolic", profile_params=None):
    """Create initial temperature profile using the profile factory."""
    return make_profile(r, profile_name, profile_params)


def _compute_reference(T0, r, dt, t_end, alpha):
    nr_fine = 4 * len(r) - 3
    r_fine = np.linspace(0, 1, nr_fine)
    T0_fine = np.interp(r_fine, r, T0)
    solver = ImplicitFDM()
    T_hist = solver.solve(T0_fine, r_fine, dt / 4.0, t_end, alpha)
    indices = np.linspace(0, nr_fine - 1, len(r)).astype(int)
    return T_hist[:, indices]


def _get_profile_variations():
    """Return list of (profile_name, params) tuples for training data generation."""
    variations = [
        ("parabolic", {"n": 2.0}),
        ("parabolic", {"n": 4.0}),
        ("gaussian", {"sigma": 0.3}),
        ("gaussian", {"sigma": 0.5}),
        ("flat_top", {"w": 0.8, "n": 4, "m": 2}),
        ("cosine", {}),
        ("linear", {}),
    ]
    return variations


def generate_training_data(
    output_path: str = "data/training_data.csv",
    lam: float = 0.1,
    alpha_list=None,
    nr_list=None,
    dt_list=None,
    t_end_list=None,
    profile_list=None,
):
    """Run parameter sweep, label best solver, save CSV."""
    if alpha_list is None:
        alpha_list = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
    if nr_list is None:
        nr_list = [31, 51, 71]
    if dt_list is None:
        dt_list = [0.0005, 0.001, 0.002]
    if t_end_list is None:
        t_end_list = [0.05, 0.1, 0.2]
    if profile_list is None:
        profile_list = _get_profile_variations()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    solvers = [ImplicitFDM(), CosineSpectral(), PINNStub()]
    rows = []
    total = (len(alpha_list) * len(nr_list) * len(dt_list)
             * len(t_end_list) * len(profile_list))
    count = 0

    for profile_name, profile_params in profile_list:
        for alpha in alpha_list:
            for nr in nr_list:
                for dt in dt_list:
                    for t_end in t_end_list:
                        count += 1
                        r = np.linspace(0, 1, nr)
                        T0 = _make_initial(r, profile_name, profile_params)

                        feats = extract_initial_features(T0, r, alpha, nr, dt, t_end)

                        # Reference
                        try:
                            T_ref = _compute_reference(T0, r, dt, t_end, alpha)
                        except Exception:
                            continue

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
                            best = select_best(results, lam=lam)
                            label = best["name"]
                        except ValueError:
                            continue

                        row = {**feats, "best_solver": label}
                        rows.append(row)

                        if count % 10 == 0:
                            print(f"  [{count}/{total}] profile={profile_name} "
                                  f"alpha={alpha} nr={nr} dt={dt} -> {label}")

    # Write CSV (append or overwrite)
    fieldnames = FEATURE_NAMES + ["best_solver"]
    _write_rows(output_path, rows, fieldnames, append=False)

    print(f"Saved {len(rows)} training samples to {output_path}")
    return output_path


def append_training_sample(
    feats: dict, best_solver: str,
    output_path: str = "data/training_data.csv",
):
    """Append a single training sample to the CSV file."""
    fieldnames = FEATURE_NAMES + ["best_solver"]
    row = {f: feats[f] for f in FEATURE_NAMES}
    row["best_solver"] = best_solver
    _write_rows(output_path, [row], fieldnames, append=True)


def _write_rows(path: str, rows: list, fieldnames: list, append: bool):
    """Write rows to CSV, optionally appending to existing file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.isfile(path) and os.path.getsize(path) > 0

    mode = "a" if (append and file_exists) else "w"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w" or not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def train_model(
    data_path: str = "data/training_data.csv",
    output_path: str = "data/solver_model.npz",
    max_depth: int = 5,
):
    """Load CSV, train decision tree, save model."""
    # Load CSV
    with open(data_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No training data in {data_path}")

    X = np.array([[float(row[f]) for f in FEATURE_NAMES] for row in rows])
    y = np.array([row["best_solver"] for row in rows])

    tree = NumpyDecisionTree(max_depth=max_depth)
    tree.fit(X, y)

    # Training accuracy
    preds = tree.predict(X)
    acc = np.mean(preds == y)
    print(f"Training accuracy: {acc:.1%} ({int(acc * len(y))}/{len(y)})")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree.save(output_path)
    print(f"Model saved to {output_path}")
    return tree


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train ML solver selector")
    parser.add_argument("--data", default="data/training_data.csv")
    parser.add_argument("--model", default="data/solver_model.npz")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--generate", action="store_true",
                        help="Generate training data first")
    args = parser.parse_args()

    if args.generate:
        generate_training_data(args.data)

    train_model(args.data, args.model, args.max_depth)


if __name__ == "__main__":
    main()
