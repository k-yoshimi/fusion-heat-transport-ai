"""Generate tutorial figures for docs/figures/."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.run_benchmark import make_initial, compute_reference
from solvers.fdm.implicit import ImplicitFDM
from solvers.spectral.cosine import CosineSpectral
from metrics.accuracy import compute_errors
from features.extract import extract_all

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
plt.rcParams.update({"figure.dpi": 150, "figure.figsize": (7, 4.5)})


def fig1_initial_condition():
    """Plot the initial condition T₀(r) = 1 - r²."""
    r = np.linspace(0, 1, 201)
    T0 = make_initial(r)

    fig, ax = plt.subplots()
    ax.plot(r, T0, linewidth=2, label="T₀ = 1 − r²")
    ax.set_xlabel("r")
    ax.set_ylabel("T(r, t=0)")
    ax.set_title("Initial Condition")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "initial_conditions.png"))
    plt.close(fig)
    print("  initial_conditions.png")


def fig2_time_evolution():
    """Plot temperature evolution for alpha=0 and alpha=1."""
    r = np.linspace(0, 1, 201)
    T0 = make_initial(r)
    dt, t_end = 0.0002, 0.2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, alpha in zip(axes, [0.0, 1.0]):
        solver = ImplicitFDM()
        T_hist = solver.solve(T0.copy(), r, dt, t_end, alpha)
        nsteps = T_hist.shape[0] - 1

        for frac, ls, lw in [(0, "-", 2), (0.25, "--", 1.5), (0.5, "-.", 1.5), (1.0, ":", 1.5)]:
            idx = int(frac * nsteps)
            t_val = frac * t_end
            ax.plot(r, T_hist[idx], ls, linewidth=lw, label=f"t={t_val:.3f}")

        ax.set_xlabel("r")
        ax.set_ylabel("T(r, t)")
        ax.set_title(f"α = {alpha}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Temperature Evolution (Implicit FDM)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "time_evolution.png"), bbox_inches="tight")
    plt.close(fig)
    print("  time_evolution.png")


def fig3_solver_comparison():
    """Compare solver final profiles and errors for alpha=1.0."""
    r = np.linspace(0, 1, 101)
    T0 = make_initial(r)
    alpha, dt, t_end = 1.0, 0.0005, 0.1

    T_ref = compute_reference(T0, r, dt, t_end, alpha)

    solvers = [ImplicitFDM(), CosineSpectral()]
    colors = ["#1f77b4", "#ff7f0e"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: final profiles
    ax = axes[0]
    ax.plot(r, T_ref[-1], "k-", linewidth=2, label="Reference")
    for s, c in zip(solvers, colors):
        T_hist = s.solve(T0.copy(), r, dt, t_end, alpha)
        ax.plot(r, T_hist[-1], "--", color=c, linewidth=1.5, label=s.name)
    ax.set_xlabel("r")
    ax.set_ylabel("T(r, t_end)")
    ax.set_title("Final Temperature Profile (α=1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: pointwise error
    ax = axes[1]
    for s, c in zip(solvers, colors):
        T_hist = s.solve(T0.copy(), r, dt, t_end, alpha)
        err = np.abs(T_hist[-1] - T_ref[-1])
        ax.plot(r, err, color=c, linewidth=1.5, label=s.name)
    ax.set_xlabel("r")
    ax.set_ylabel("|T - T_ref|")
    ax.set_title("Pointwise Error (α=1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "solver_comparison.png"))
    plt.close(fig)
    print("  solver_comparison.png")


def fig4_alpha_sweep():
    """Bar chart of L2 errors and wall times across alpha values."""
    r = np.linspace(0, 1, 51)
    dt, t_end = 0.001, 0.1
    alphas = [0.0, 0.5, 1.0, 2.0]

    solver_classes = [ImplicitFDM, CosineSpectral]
    results = {s().name: {"l2": [], "time": []} for s in solver_classes}

    for alpha in alphas:
        T0 = make_initial(r)
        T_ref = compute_reference(T0, r, dt, t_end, alpha)
        for SC in solver_classes:
            s = SC()
            t0 = time.perf_counter()
            T_hist = s.solve(T0.copy(), r, dt, t_end, alpha)
            wall = time.perf_counter() - t0

            nt_ref, nt_sol = T_ref.shape[0], T_hist.shape[0]
            if nt_sol != nt_ref:
                T_cmp = np.stack([T_hist[0], T_hist[-1]])
                R_cmp = np.stack([T_ref[0], T_ref[-1]])
            else:
                T_cmp, R_cmp = T_hist, T_ref
            errs = compute_errors(T_cmp, R_cmp, r)
            results[s.name]["l2"].append(errs["l2"])
            results[s.name]["time"].append(wall)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(alphas))
    width = 0.35

    # L2 error
    ax = axes[0]
    for i, (name, data) in enumerate(results.items()):
        ax.bar(x + i * width, data["l2"], width, label=name)
    ax.set_xlabel("α")
    ax.set_ylabel("L2 Error")
    ax.set_title("L2 Error vs α")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # Wall time
    ax = axes[1]
    for i, (name, data) in enumerate(results.items()):
        ax.bar(x + i * width, [t * 1000 for t in data["time"]], width, label=name)
    ax.set_xlabel("α")
    ax.set_ylabel("Wall Time [ms]")
    ax.set_title("Wall Time vs α")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "alpha_sweep.png"))
    plt.close(fig)
    print("  alpha_sweep.png")


def fig5_nonlinear_diffusivity():
    """Show chi profile for different alpha values."""
    r = np.linspace(0, 1, 201)
    T0 = make_initial(r)
    from features.extract import gradient, chi

    dTdr = gradient(T0, r)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(r, T0, linewidth=2)
    ax.set_xlabel("r")
    ax.set_ylabel("T₀(r)")
    ax.set_title("Initial Condition: T₀ = 1 − r²")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for alpha in [0.0, 0.5, 1.0, 2.0]:
        chi_vals = chi(dTdr, alpha)
        ax.plot(r, chi_vals, linewidth=1.5, label=f"α={alpha}")
    ax.set_xlabel("r")
    ax.set_ylabel("χ(|dT/dr|)")
    ax.set_title("Nonlinear Diffusivity χ(|T'|)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "nonlinear_diffusivity.png"))
    plt.close(fig)
    print("  nonlinear_diffusivity.png")


if __name__ == "__main__":
    print("Generating tutorial figures...")
    fig1_initial_condition()
    fig2_time_evolution()
    fig3_solver_comparison()
    fig4_alpha_sweep()
    fig5_nonlinear_diffusivity()
    print("Done.")
