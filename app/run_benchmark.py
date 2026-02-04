"""CLI entrypoint for running the heat transport benchmark."""

import argparse
import time
import numpy as np

from solvers.fdm.implicit import ImplicitFDM
from solvers.spectral.cosine import CosineSpectral
from solvers.pinn.stub import PINNStub
from solvers.fdm.compact4 import Compact4FDM
from solvers.fdm.imex import IMEXFDM
from solvers.fem.p2_fem import P2FEM
from solvers.fvm.cell_centered import CellCenteredFVM
from solvers.spectral.chebyshev import ChebyshevSpectral
from metrics.accuracy import compute_errors
from features.extract import extract_all, extract_initial_features
from features.profiles import make_profile, parse_profile_params, get_available_profiles
from policy.select import select_best
from reports.generate import write_csv, write_markdown


# All available solvers
SOLVERS = [
    ImplicitFDM(),
    CosineSpectral(),
    PINNStub(),
    Compact4FDM(),
    IMEXFDM(),
    P2FEM(),
    CellCenteredFVM(),
    ChebyshevSpectral(),
]


def get_solver_map():
    """Return dict mapping solver name to instance."""
    return {s.name: s for s in SOLVERS}


def make_initial(r: np.ndarray, profile_name: str = "parabolic",
                 profile_params: dict = None) -> np.ndarray:
    """Create initial temperature profile.

    Args:
        r: Radial coordinate array
        profile_name: Profile type (parabolic, gaussian, flat_top, cosine, linear)
        profile_params: Optional parameter overrides for the profile

    Returns:
        Initial temperature profile T0(r)
    """
    return make_profile(r, profile_name, profile_params)


def compute_reference(T0, r, dt, t_end, alpha):
    """Compute reference solution with 4x refinement."""
    nr_fine = 4 * len(r) - 3
    r_fine = np.linspace(0, 1, nr_fine)
    T0_fine = np.interp(r_fine, r, T0)
    dt_fine = dt / 4.0
    solver = ImplicitFDM()
    T_hist = solver.solve(T0_fine, r_fine, dt_fine, t_end, alpha)
    # Downsample back to original grid
    indices = np.linspace(0, nr_fine - 1, len(r)).astype(int)
    return T_hist[:, indices]


def run(alpha_list, nr=51, dt=0.001, t_end=0.1,
        profile_name="parabolic", profile_params=None):
    """Run benchmark for given alpha values."""
    all_results = []

    for alpha in alpha_list:
        r = np.linspace(0, 1, nr)
        T0 = make_initial(r, profile_name, profile_params)

        # Reference solution
        print(f"Computing reference for alpha={alpha}...")
        T_ref = compute_reference(T0, r, dt, t_end, alpha)

        solvers = [ImplicitFDM(), CosineSpectral(), PINNStub()]

        for s in solvers:
            print(f"  Running {s.name} (alpha={alpha})...")
            t0 = time.perf_counter()
            T_hist = s.solve(T0.copy(), r, dt, t_end, alpha)
            wall = time.perf_counter() - t0

            # Ensure same shape for error computation
            nt_ref = T_ref.shape[0]
            nt_sol = T_hist.shape[0]
            if nt_sol != nt_ref:
                # Use final profiles only
                T_hist_cmp = np.stack([T_hist[0], T_hist[-1]])
                T_ref_cmp = np.stack([T_ref[0], T_ref[-1]])
            else:
                T_hist_cmp = T_hist
                T_ref_cmp = T_ref

            errs = compute_errors(T_hist_cmp, T_ref_cmp, r)
            feats = extract_all(T_hist[-1], r, alpha)

            result = {
                "name": s.name,
                "alpha": alpha,
                "l2_error": errs["l2"],
                "linf_error": errs["linf"],
                "wall_time": wall,
                **feats,
            }
            all_results.append(result)
            print(f"    L2={errs['l2']:.6g}, Linf={errs['linf']:.6g}, time={wall:.4f}s")

    # Select best per alpha
    for alpha in alpha_list:
        subset = [r for r in all_results if r["alpha"] == alpha]
        try:
            best = select_best(subset)
            print(f"\nBest for alpha={alpha}: {best['name']}")
        except ValueError:
            best = {"name": "none"}

    # Write reports
    write_csv(all_results, "outputs/benchmark.csv")
    best_overall = select_best([r for r in all_results if not np.isnan(r.get("l2_error", float("nan")))])
    write_markdown(all_results, best_overall["name"], "outputs/benchmark.md")
    print("\nReports written to outputs/")


def run_ml_selector(alpha_list, nr=51, dt=0.001, t_end=0.1,
                    model_path="data/solver_model.npz",
                    profile_name="parabolic", profile_params=None):
    """Run benchmark using ML-predicted best solver only."""
    from policy.select import select_with_ml

    solver_map = get_solver_map()

    for alpha in alpha_list:
        r = np.linspace(0, 1, nr)
        T0 = make_initial(r, profile_name, profile_params)

        predicted = select_with_ml(T0, r, alpha, nr, dt, t_end, model_path)
        print(f"ML predicted best solver for alpha={alpha}: {predicted}")

        if predicted not in solver_map:
            print(f"  Warning: unknown solver '{predicted}', falling back to implicit_fdm")
            predicted = "implicit_fdm"

        solver = solver_map[predicted]
        t0 = time.perf_counter()
        T_hist = solver.solve(T0.copy(), r, dt, t_end, alpha)
        wall = time.perf_counter() - t0

        T_ref = compute_reference(T0, r, dt, t_end, alpha)
        nt_ref, nt_sol = T_ref.shape[0], T_hist.shape[0]
        if nt_sol != nt_ref:
            T_hist_cmp = np.stack([T_hist[0], T_hist[-1]])
            T_ref_cmp = np.stack([T_ref[0], T_ref[-1]])
        else:
            T_hist_cmp, T_ref_cmp = T_hist, T_ref

        errs = compute_errors(T_hist_cmp, T_ref_cmp, r)
        print(f"  L2={errs['l2']:.6g}, Linf={errs['linf']:.6g}, time={wall:.4f}s")


def run_physics_selector(alpha_list, t_end=0.1, target_error=0.005,
                         profile_name="parabolic", profile_params=None,
                         optimize_params=True,
                         physics_model_path="data/physics_model.npz"):
    """Run benchmark using physics-based solver selection and parameter optimization.

    Two-stage workflow:
    1. Physics selector chooses best solver based on physical features
    2. Parameter optimizer finds optimal (dt, nr) for that solver
    """
    from policy.physics_selector import PhysicsSolverSelector
    from policy.optimizer import ParameterOptimizer

    selector = PhysicsSolverSelector(physics_model_path)
    optimizer = ParameterOptimizer()
    solver_map = get_solver_map()

    for alpha in alpha_list:
        # Initial grid for feature extraction
        r_init = np.linspace(0, 1, 51)
        T0_init = make_initial(r_init, profile_name, profile_params)

        # Stage 1: Physics selector
        predicted_solver = selector.predict(T0_init, r_init, alpha)
        print(f"\nPhysics selector chose: {predicted_solver}")

        if predicted_solver not in solver_map:
            print(f"  Warning: unknown solver '{predicted_solver}', falling back to implicit_fdm")
            predicted_solver = "implicit_fdm"

        # Stage 2: Parameter optimization
        if optimize_params:
            opt_result = optimizer.optimize_for_solver(
                predicted_solver, T0_init, r_init, alpha, t_end, target_error
            )
            nr = opt_result.nr
            dt = opt_result.dt
            print(f"Optimizer recommends: dt={dt:.6g}, nr={nr}")
            if opt_result.notes:
                print(f"  Note: {opt_result.notes}")
        else:
            nr, dt = 51, 0.001
            print(f"Using default: dt={dt}, nr={nr}")

        # Run with optimized parameters
        r = np.linspace(0, 1, nr)
        T0 = make_initial(r, profile_name, profile_params)

        solver = solver_map[predicted_solver]
        t0 = time.perf_counter()
        T_hist = solver.solve(T0.copy(), r, dt, t_end, alpha)
        wall = time.perf_counter() - t0

        # Compute reference for comparison
        T_ref = compute_reference(T0, r, dt, t_end, alpha)
        nt_ref, nt_sol = T_ref.shape[0], T_hist.shape[0]
        if nt_sol != nt_ref:
            T_hist_cmp = np.stack([T_hist[0], T_hist[-1]])
            T_ref_cmp = np.stack([T_ref[0], T_ref[-1]])
        else:
            T_hist_cmp, T_ref_cmp = T_hist, T_ref

        errs = compute_errors(T_hist_cmp, T_ref_cmp, r)
        print(f"Running benchmark...")
        print(f"  L2={errs['l2']:.6g}, Linf={errs['linf']:.6g}, time={wall:.4f}s")


def _update_model(alpha_list, nr, dt, t_end, data_path, model_path,
                  profile_name="parabolic", profile_params=None):
    """Append current benchmark results to training data and retrain."""
    from policy.train import append_training_sample, train_model

    solvers = [ImplicitFDM(), CosineSpectral(), PINNStub()]
    count = 0

    for alpha in alpha_list:
        r = np.linspace(0, 1, nr)
        T0 = make_initial(r, profile_name, profile_params)
        feats = extract_initial_features(T0, r, alpha, nr, dt, t_end)

        T_ref = compute_reference(T0, r, dt, t_end, alpha)
        results = []
        for s in solvers:
            try:
                t0 = time.perf_counter()
                T_hist = s.solve(T0.copy(), r, dt, t_end, alpha)
                wall = time.perf_counter() - t0
                nt_ref, nt_sol = T_ref.shape[0], T_hist.shape[0]
                if nt_sol != nt_ref:
                    T_hist_cmp = np.stack([T_hist[0], T_hist[-1]])
                    T_ref_cmp = np.stack([T_ref[0], T_ref[-1]])
                else:
                    T_hist_cmp, T_ref_cmp = T_hist, T_ref
                errs = compute_errors(T_hist_cmp, T_ref_cmp, r)
                results.append({"name": s.name, "l2_error": errs["l2"], "wall_time": wall})
            except Exception:
                results.append({"name": s.name, "l2_error": float("nan"), "wall_time": 0.0})

        try:
            best = select_best(results)
            append_training_sample(feats, best["name"], data_path)
            count += 1
        except ValueError:
            pass

    print(f"\nAppended {count} samples to {data_path}")

    import os
    if os.path.isfile(data_path):
        print("Retraining model...")
        train_model(data_path, model_path)
    else:
        print("No training data file found, skipping retrain.")


def main():
    parser = argparse.ArgumentParser(description="Fusion heat transport PDE benchmark")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--nr", type=int, default=51)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--t_end", type=float, default=0.1)
    parser.add_argument("--profile", type=str, default="parabolic",
                        choices=get_available_profiles(),
                        help="Initial temperature profile type")
    parser.add_argument("--profile-params", type=str, default="",
                        help='Profile parameters as "key=val,key2=val2"')
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate training data via parameter sweep")
    parser.add_argument("--use-ml-selector", action="store_true",
                        help="Use ML model to predict best solver")
    parser.add_argument("--model-path", default="data/solver_model.npz",
                        help="Path to trained model")
    parser.add_argument("--update", action="store_true",
                        help="Append benchmark results to training data and retrain model")
    parser.add_argument("--data-path", default="data/training_data.csv",
                        help="Path to training data CSV")

    # Phase 2: Physics selector and optimizer arguments
    parser.add_argument("--train-physics-model", action="store_true",
                        help="Train physics-only model from existing training data")
    parser.add_argument("--physics-selector", action="store_true",
                        help="Use physics-only selector (two-stage workflow)")
    parser.add_argument("--optimize-params", action="store_true",
                        help="Use parameter optimizer to determine dt/nr")
    parser.add_argument("--target-error", type=float, default=0.005,
                        help="Target L2 error for parameter optimization")
    parser.add_argument("--physics-model-path", default="data/physics_model.npz",
                        help="Path to physics-only model")

    args = parser.parse_args()

    profile_params = parse_profile_params(args.profile_params)

    # Training modes
    if args.generate_data:
        from policy.train import generate_training_data
        generate_training_data(args.data_path)
        return

    if args.train_physics_model:
        from policy.train import train_physics_model
        train_physics_model(args.data_path, args.physics_model_path)
        return

    # Selector modes
    if args.physics_selector:
        run_physics_selector(
            args.alpha, args.t_end, args.target_error,
            args.profile, profile_params,
            args.optimize_params, args.physics_model_path
        )
        return

    if args.use_ml_selector:
        run_ml_selector(args.alpha, args.nr, args.dt, args.t_end,
                        args.model_path, args.profile, profile_params)
        return

    # Default benchmark mode
    run(args.alpha, args.nr, args.dt, args.t_end, args.profile, profile_params)

    if args.update:
        _update_model(args.alpha, args.nr, args.dt, args.t_end,
                      args.data_path, args.model_path,
                      args.profile, profile_params)


if __name__ == "__main__":
    main()
