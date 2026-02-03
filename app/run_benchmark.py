"""CLI entrypoint for running the heat transport benchmark."""

import argparse
import time
import numpy as np

from solvers.fdm.implicit import ImplicitFDM
from solvers.spectral.cosine import CosineSpectral
from solvers.pinn.stub import PINNStub
from metrics.accuracy import compute_errors
from features.extract import extract_all, extract_initial_features
from policy.select import select_best
from reports.generate import write_csv, write_markdown


def make_initial(r: np.ndarray) -> np.ndarray:
    """Create initial temperature profile: T₀(r) = 1 - r²."""
    return 1.0 - r**2


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


def run(alpha_list, nr=51, dt=0.001, t_end=0.1):
    """Run benchmark for given alpha values."""
    all_results = []

    for alpha in alpha_list:
        r = np.linspace(0, 1, nr)
        T0 = make_initial(r)

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
                    model_path="data/solver_model.npz"):
    """Run benchmark using ML-predicted best solver only."""
    from policy.select import select_with_ml

    solver_map = {s.name: s for s in [ImplicitFDM(), CosineSpectral(), PINNStub()]}

    for alpha in alpha_list:
        r = np.linspace(0, 1, nr)
        T0 = make_initial(r)

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


def _update_model(alpha_list, nr, dt, t_end, data_path, model_path):
    """Append current benchmark results to training data and retrain."""
    from policy.train import append_training_sample, train_model

    solvers = [ImplicitFDM(), CosineSpectral(), PINNStub()]
    count = 0

    for alpha in alpha_list:
        r = np.linspace(0, 1, nr)
        T0 = make_initial(r)
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
    args = parser.parse_args()

    if args.generate_data:
        from policy.train import generate_training_data
        generate_training_data(args.data_path)
        return

    if args.use_ml_selector:
        run_ml_selector(args.alpha, args.nr, args.dt, args.t_end,
                        args.model_path)
        return

    run(args.alpha, args.nr, args.dt, args.t_end)

    if args.update:
        _update_model(args.alpha, args.nr, args.dt, args.t_end,
                      args.data_path, args.model_path)


if __name__ == "__main__":
    main()
