"""Test and compare PINN variants.

Usage:
    # Quick test (fewer epochs)
    python tests/test_pinn_variants.py --quick

    # Full comparison
    python tests/test_pinn_variants.py

    # Test specific variant
    python tests/test_pinn_variants.py --variant improved

    # Verbose output
    python tests/test_pinn_variants.py --verbose
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed. Install with: pip install torch")
    print("Exiting.")
    sys.exit(1)

from solvers.fdm.implicit import ImplicitFDM
from solvers.pinn.stub import PINNStub
from solvers.pinn.simple import SimplePINN, NonlinearPINN
from metrics.accuracy import compute_errors

# Try to import advanced variants
try:
    from solvers.pinn.improved import ImprovedPINN, AdaptivePINN
    HAS_IMPROVED = True
except Exception:
    HAS_IMPROVED = False

try:
    from solvers.pinn.variants import (
        TransferPINN, CurriculumPINN, EnsemblePINN, FNOPINN
    )
    HAS_VARIANTS = True
except Exception:
    HAS_VARIANTS = False


def make_initial(r: np.ndarray) -> np.ndarray:
    """Create initial temperature profile."""
    return 1.0 - r**2


def compute_reference(T0, r, dt, t_end, alpha):
    """Compute reference solution with 4x refinement."""
    nr_fine = 4 * len(r) - 3
    r_fine = np.linspace(0, 1, nr_fine)
    T0_fine = np.interp(r_fine, r, T0)

    solver = ImplicitFDM()
    T_hist = solver.solve(T0_fine, r_fine, dt/4, t_end, alpha)

    indices = np.linspace(0, nr_fine - 1, len(r)).astype(int)
    return T_hist[:, indices]


def get_pinn_variants(quick: bool = False, verbose: bool = False):
    """Get dictionary of PINN variants to test."""
    if quick:
        # Reduced epochs for quick testing
        variants = {
            "stub": PINNStub(hidden=32, epochs=100, lr=1e-3),
            "simple": SimplePINN(
                hidden_dim=32, n_layers=3, epochs=500,
                n_collocation=500, verbose=verbose
            ),
            "nonlinear": NonlinearPINN(
                hidden_dim=32, n_layers=3, epochs=500,
                n_collocation=500, verbose=verbose
            ),
        }
        if HAS_IMPROVED:
            variants["improved"] = ImprovedPINN(
                hidden_dim=32, num_blocks=2, epochs=500,
                n_collocation=500, verbose=verbose
            )
            variants["adaptive"] = AdaptivePINN(
                hidden_dim=32, num_blocks=2, epochs=500,
                n_collocation=500, resample_interval=200, verbose=verbose
            )
        if HAS_VARIANTS:
            variants["curriculum"] = CurriculumPINN(
                hidden_dim=32, num_blocks=2, epochs_per_stage=200,
                n_stages=3, n_collocation=500, verbose=verbose
            )
            variants["ensemble"] = EnsemblePINN(
                n_models=2, hidden_dim=32, num_blocks=2, epochs=300,
                n_collocation=500, verbose=verbose
            )
            variants["fno"] = FNOPINN(
                hidden_channels=16, modes=8, n_layers=2, epochs=500,
                n_time_samples=20, verbose=verbose
            )
        return variants
    else:
        # Full training
        variants = {
            "stub": PINNStub(hidden=32, epochs=200, lr=1e-3),
            "simple": SimplePINN(
                hidden_dim=64, n_layers=4, epochs=2000,
                n_collocation=1000, verbose=verbose
            ),
            "nonlinear": NonlinearPINN(
                hidden_dim=64, n_layers=4, epochs=2000,
                n_collocation=1000, verbose=verbose
            ),
        }
        if HAS_IMPROVED:
            variants["improved"] = ImprovedPINN(
                hidden_dim=64, num_blocks=4, epochs=5000,
                n_collocation=2000, verbose=verbose
            )
            variants["adaptive"] = AdaptivePINN(
                hidden_dim=64, num_blocks=4, epochs=5000,
                n_collocation=2000, resample_interval=500, verbose=verbose
            )
        if HAS_VARIANTS:
            variants["transfer"] = TransferPINN(
                hidden_dim=64, num_blocks=4,
                pretrain_epochs=2000, finetune_epochs=3000,
                n_collocation=2000, verbose=verbose
            )
            variants["curriculum"] = CurriculumPINN(
                hidden_dim=64, num_blocks=4, epochs_per_stage=1000,
                n_stages=5, n_collocation=2000, verbose=verbose
            )
            variants["ensemble"] = EnsemblePINN(
                n_models=3, hidden_dim=64, num_blocks=4, epochs=3000,
                n_collocation=1500, verbose=verbose
            )
            variants["fno"] = FNOPINN(
                hidden_channels=32, modes=16, n_layers=4, epochs=3000,
                n_time_samples=50, verbose=verbose
            )
        return variants


def _test_single_variant(name, solver, T0, r, dt, t_end, alpha, T_ref):
    """Test a single PINN variant (helper function)."""
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"{'='*50}")

    start_time = time.perf_counter()
    T_hist = solver.solve(T0.copy(), r, dt, t_end, alpha)
    wall_time = time.perf_counter() - start_time

    # Check for NaN
    if np.any(np.isnan(T_hist)):
        print(f"  Result: FAILED (NaN)")
        return {
            "name": name,
            "l2_error": np.nan,
            "linf_error": np.nan,
            "wall_time": wall_time,
            "stable": False,
        }

    # Compute errors
    T_cmp = np.stack([T_hist[0], T_hist[-1]])
    T_ref_cmp = np.stack([T_ref[0], T_ref[-1]])
    errors = compute_errors(T_cmp, T_ref_cmp, r)

    print(f"  L2 Error:   {errors['l2']:.6f}")
    print(f"  L∞ Error:   {errors['linf']:.6f}")
    print(f"  Wall Time:  {wall_time:.2f}s")

    return {
        "name": name,
        "l2_error": errors["l2"],
        "linf_error": errors["linf"],
        "wall_time": wall_time,
        "stable": True,
        "T_final": T_hist[-1],
    }


def compare_all(args):
    """Compare all PINN variants."""
    print("=" * 60)
    print("PINN Variant Comparison")
    print("=" * 60)

    # Setup
    nr = 51
    r = np.linspace(0, 1, nr)
    T0 = make_initial(r)
    dt = 0.001
    t_end = 0.1
    alpha = args.alpha

    print(f"\nParameters:")
    print(f"  nr = {nr}, dt = {dt}, t_end = {t_end}, α = {alpha}")

    # Compute reference
    print("\nComputing reference solution (FDM with 4x refinement)...")
    T_ref = compute_reference(T0, r, dt, t_end, alpha)

    # Get variants to test
    variants = get_pinn_variants(quick=args.quick, verbose=args.verbose)

    if args.variant:
        if args.variant not in variants:
            print(f"Unknown variant: {args.variant}")
            print(f"Available: {list(variants.keys())}")
            return
        variants = {args.variant: variants[args.variant]}

    # Test each variant
    results = []
    for name, solver in variants.items():
        result = _test_single_variant(name, solver, T0, r, dt, t_end, alpha, T_ref)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Variant':<15} {'L2 Error':>12} {'L∞ Error':>12} {'Time (s)':>10} {'Status':>10}")
    print("-" * 60)

    for res in sorted(results, key=lambda x: x["l2_error"] if not np.isnan(x["l2_error"]) else 1e10):
        status = "OK" if res["stable"] else "FAILED"
        l2 = f"{res['l2_error']:.6f}" if not np.isnan(res["l2_error"]) else "NaN"
        linf = f"{res['linf_error']:.6f}" if not np.isnan(res["linf_error"]) else "NaN"
        print(f"{res['name']:<15} {l2:>12} {linf:>12} {res['wall_time']:>10.2f} {status:>10}")

    # Find best
    valid_results = [r for r in results if r["stable"]]
    if valid_results:
        best = min(valid_results, key=lambda x: x["l2_error"])
        print(f"\nBest variant: {best['name']} (L2 = {best['l2_error']:.6f})")

    # Plot comparison if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Final temperature profile
        ax = axes[0]
        ax.plot(r, T_ref[-1], 'k-', linewidth=2, label='Reference (FDM 4x)')
        for res in results:
            if res["stable"] and "T_final" in res:
                ax.plot(r, res["T_final"], '--', label=res["name"], alpha=0.7)
        ax.set_xlabel("r")
        ax.set_ylabel("T")
        ax.set_title(f"Final Temperature Profile (α={alpha})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Error comparison
        ax = axes[1]
        names = [r["name"] for r in valid_results]
        l2_errors = [r["l2_error"] for r in valid_results]
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(names)))
        bars = ax.bar(names, l2_errors, color=colors)
        ax.set_ylabel("L2 Error")
        ax.set_title("Error Comparison")
        ax.tick_params(axis='x', rotation=45)
        for bar, err in zip(bars, l2_errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{err:.4f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig("pinn_comparison.png", dpi=150)
        print(f"\nPlot saved to: pinn_comparison.png")
        plt.close()

    except ImportError:
        print("\nMatplotlib not available for plotting.")


def main():
    parser = argparse.ArgumentParser(description="Test PINN variants")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Quick test with fewer epochs")
    parser.add_argument("--variant", "-v", type=str, default=None,
                       help="Test specific variant only")
    parser.add_argument("--alpha", "-a", type=float, default=0.5,
                       help="Nonlinearity parameter (default: 0.5)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show training progress")
    args = parser.parse_args()

    compare_all(args)


if __name__ == "__main__":
    main()
