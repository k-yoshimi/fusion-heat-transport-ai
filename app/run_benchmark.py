"""CLI entrypoint for running the heat transport benchmark."""

import argparse
import time
import numpy as np

from solvers.fdm.implicit import ImplicitFDM
from solvers.spectral.cosine import CosineSpectral
from solvers.pinn.stub import PINNStub
from metrics.accuracy import compute_errors
from features.extract import extract_all
from policy.select import select_best
from reports.generate import write_csv, write_markdown


def make_initial(r: np.ndarray, kind: str = "gaussian") -> np.ndarray:
    """Create initial temperature profile."""
    if kind == "gaussian":
        return np.exp(-10 * r**2)
    elif kind == "sharp":
        return np.where(r < 0.3, 1.0, 0.0) * (1.0 - r / 0.3)
    else:
        raise ValueError(f"Unknown initial condition: {kind}")


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


def run(alpha_list, nr=51, dt=0.001, t_end=0.1, init="gaussian"):
    """Run benchmark for given alpha values."""
    all_results = []

    for alpha in alpha_list:
        r = np.linspace(0, 1, nr)
        T0 = make_initial(r, init)

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


def main():
    parser = argparse.ArgumentParser(description="Fusion heat transport PDE benchmark")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--nr", type=int, default=51)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--t_end", type=float, default=0.1)
    parser.add_argument("--init", choices=["gaussian", "sharp"], default="gaussian")
    args = parser.parse_args()
    run(args.alpha, args.nr, args.dt, args.t_end, args.init)


if __name__ == "__main__":
    main()
