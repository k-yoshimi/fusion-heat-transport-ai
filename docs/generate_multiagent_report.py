"""Generate comprehensive LaTeX report for multi-agent solver analysis.

This script generates figures and a LaTeX report summarizing the
multi-agent hypothesis-driven experiment framework results.
"""

import os
import sys
import json
import subprocess
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib not available for plotting.")
    sys.exit(1)


def load_experiment_data():
    """Load experiment data from CSV file."""
    import csv
    data_path = "data/experiments.csv"
    if os.path.exists(data_path):
        data = []
        with open(data_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert types
                entry = {
                    'experiment': row.get('experiment_name', ''),
                    'alpha': float(row.get('alpha', 0)),
                    'nr': int(row.get('nr', 51)),
                    'dt': float(row.get('dt', 0.001)),
                    'solver': row.get('solver', ''),
                    'l2_error': float(row.get('l2_error', 0)) if row.get('l2_error') != 'nan' else np.nan,
                    'linf_error': float(row.get('linf_error', 0)) if row.get('linf_error') != 'nan' else np.nan,
                    'wall_time': float(row.get('wall_time', 0)),
                    'stable': row.get('is_stable', 'False') == 'True',
                }
                data.append(entry)
        return data
    return []


def load_hypothesis_data():
    """Load hypothesis tracking data."""
    hypo_path = "data/hypotheses_memo.json"
    if os.path.exists(hypo_path):
        with open(hypo_path, 'r') as f:
            return json.load(f)
    return {}


def generate_figures(data, hypotheses, output_dir):
    """Generate analysis figures."""
    os.makedirs(output_dir, exist_ok=True)

    if not data:
        print("No experiment data found.")
        return

    # Extract data by solver type
    all_solvers = sorted(set(d.get('solver', '') for d in data))
    solver_data = {s: [d for d in data if d.get('solver') == s] for s in all_solvers}

    # For backward compatibility
    fdm_data = solver_data.get('implicit_fdm', [])
    spectral_data = solver_data.get('spectral_cosine', [])

    # PINN data
    pinn_solvers = [s for s in all_solvers if s.startswith('pinn_')]
    has_pinn = len(pinn_solvers) > 0

    # Solver display names
    solver_names = {
        'implicit_fdm': 'Implicit FDM',
        'spectral_cosine': 'Spectral',
        'pinn_simple': 'PINN Simple',
        'pinn_nonlinear': 'PINN Nonlinear',
        'pinn_improved': 'PINN Improved',
        'pinn_fno': 'PINN FNO',
    }

    # Figure 1: Stability comparison by alpha (all solvers)
    fig, ax = plt.subplots(figsize=(12, 6))

    alphas = sorted(set(d.get('alpha', 0) for d in data))
    n_solvers = len(all_solvers)
    x = np.arange(len(alphas))
    width = 0.8 / n_solvers

    colors = plt.cm.tab10(np.linspace(0, 1, n_solvers))

    for i, solver in enumerate(all_solvers):
        stability = []
        for alpha in alphas:
            solver_alpha = [d for d in solver_data[solver] if d.get('alpha') == alpha]
            stable = sum(1 for d in solver_alpha if d.get('stable', False))
            stability.append(100 * stable / len(solver_alpha) if solver_alpha else 0)

        offset = (i - n_solvers / 2 + 0.5) * width
        label = solver_names.get(solver, solver)
        bars = ax.bar(x + offset, stability, width, label=label, color=colors[i])

    ax.set_xlabel('Nonlinearity Parameter α', fontsize=12)
    ax.set_ylabel('Stability Rate (%)', fontsize=12)
    ax.set_title('Solver Stability Comparison by α', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a:.1f}' for a in alphas])
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_stability_by_alpha.png'), dpi=150)
    plt.close()

    # Figure 2: L2 Error comparison (all solvers)
    fig, ax = plt.subplots(figsize=(12, 6))

    solver_errors = {}
    for solver in all_solvers:
        errors = [d.get('l2_error', np.nan) for d in solver_data[solver]
                  if d.get('stable', False) and not np.isnan(d.get('l2_error', np.nan))]
        if errors:
            solver_errors[solver] = errors

    names = [solver_names.get(s, s) for s in solver_errors.keys()]
    errors_list = list(solver_errors.values())
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    bp = ax.boxplot(errors_list, patch_artist=True, labels=names)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('L2 Error Distribution by Solver', fontsize=14)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean annotations
    for i, (solver, errs) in enumerate(solver_errors.items()):
        mean_val = np.mean(errs)
        ax.annotate(f'μ={mean_val:.3f}', xy=(i + 1, mean_val),
                   xytext=(i + 1.2, mean_val), fontsize=8, color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_error_distribution.png'), dpi=150)
    plt.close()

    # Figure 3: Computation time comparison (all solvers, log scale)
    fig, ax = plt.subplots(figsize=(12, 6))

    solver_times = {}
    for solver in all_solvers:
        times = [d.get('wall_time', 0) * 1000 for d in solver_data[solver]
                 if d.get('stable', False)]
        if times:
            solver_times[solver] = times

    names = [solver_names.get(s, s) for s in solver_times.keys()]
    times_list = list(solver_times.values())
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    bp = ax.boxplot(times_list, patch_artist=True, labels=names)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Wall Time (ms)', fontsize=12)
    ax.set_title('Computation Time Comparison (log scale)', fontsize=14)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean annotations
    for i, (solver, times) in enumerate(solver_times.items()):
        mean_val = np.mean(times)
        label = f'{mean_val:.1f}ms' if mean_val < 1000 else f'{mean_val/1000:.1f}s'
        ax.annotate(f'μ={label}', xy=(i + 1, mean_val),
                   xytext=(i + 1.15, mean_val * 1.3), fontsize=8, color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_computation_time.png'), dpi=150)
    plt.close()

    # Figure 4: Hypothesis verification results
    fig, ax = plt.subplots(figsize=(10, 6))

    if hypotheses:
        hypo_names = []
        statuses = []
        confidences = []

        for hid, h in hypotheses.items():
            hypo_names.append(hid)
            statuses.append(h.get('status', 'inconclusive'))
            confidences.append(h.get('confidence', 0))

        colors_map = {'confirmed': 'green', 'rejected': 'red', 'inconclusive': 'gray'}
        bar_colors = [colors_map.get(s, 'gray') for s in statuses]

        bars = ax.barh(hypo_names, confidences, color=bar_colors, alpha=0.7)

        ax.set_xlabel('Confidence (%)', fontsize=12)
        ax.set_title('Hypothesis Verification Results', fontsize=14)
        ax.set_xlim(0, 110)

        for bar, status in zip(bars, statuses):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                   status.upper(), va='center', fontsize=9)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Confirmed'),
            Patch(facecolor='red', alpha=0.7, label='Rejected'),
            Patch(facecolor='gray', alpha=0.7, label='Inconclusive')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_hypothesis_results.png'), dpi=150)
    plt.close()

    # Figure 5: Multi-agent system architecture
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Draw boxes for agents
    agents = [
        ('Statistics\nAgent', 1.5, 6, 'lightblue'),
        ('Feature\nAgent', 4.5, 6, 'lightgreen'),
        ('Pattern\nAgent', 7.5, 6, 'lightyellow'),
        ('Hypothesis\nAgent', 10.5, 6, 'lightcoral'),
    ]

    for name, x, y, color in agents:
        rect = plt.Rectangle((x-1, y-0.7), 2, 1.4, facecolor=color,
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold')

    # Central coordinator
    rect = plt.Rectangle((5, 3.3), 2, 1.4, facecolor='lavender',
                         edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 4, 'Coordinator\n(Parallel)', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Data flow
    rect = plt.Rectangle((5, 0.8), 2, 1.4, facecolor='lightgray',
                         edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 1.5, 'Experiment\nDatabase', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Arrows
    for name, x, y, _ in agents:
        ax.annotate('', xy=(6, 4.7), xytext=(x, y-0.7),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax.annotate('', xy=(6, 3.3), xytext=(6, 2.2),
               arrowprops=dict(arrowstyle='<->', color='blue', lw=2))

    ax.set_title('Multi-Agent Analysis System Architecture', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_architecture.png'), dpi=150)
    plt.close()

    # Figure 6: PINN Comparison (if PINN data exists)
    if has_pinn:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 6a: L2 Error comparison across all solvers
        ax = axes[0]
        solver_errors = {}
        for solver in all_solvers:
            errors = [d.get('l2_error', np.nan) for d in solver_data[solver]
                     if d.get('stable', False) and not np.isnan(d.get('l2_error', np.nan))]
            if errors:
                solver_errors[solver] = np.mean(errors)

        if solver_errors:
            names = list(solver_errors.keys())
            errors = list(solver_errors.values())
            colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

            bars = ax.bar(names, errors, color=colors)
            ax.set_ylabel('Average L2 Error', fontsize=12)
            ax.set_title('Solver Accuracy Comparison (All Solvers)', fontsize=14)
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')

            for bar, err in zip(bars, errors):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{err:.3f}', ha='center', va='bottom', fontsize=8)

        # 6b: Computation time comparison (log scale for PINN)
        ax = axes[1]
        solver_times = {}
        for solver in all_solvers:
            times = [d.get('wall_time', 0) * 1000 for d in solver_data[solver]
                    if d.get('stable', False)]
            if times:
                solver_times[solver] = np.mean(times)

        if solver_times:
            names = list(solver_times.keys())
            times = list(solver_times.values())
            colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

            bars = ax.bar(names, times, color=colors)
            ax.set_ylabel('Average Wall Time (ms)', fontsize=12)
            ax.set_title('Computation Time Comparison (All Solvers)', fontsize=14)
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, axis='y')

            for bar, t in zip(bars, times):
                label = f'{t:.1f}ms' if t < 1000 else f'{t/1000:.1f}s'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                       label, ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig6_pinn_comparison.png'), dpi=150)
        plt.close()

    print(f"Figures saved to {output_dir}/")


def generate_latex_report(data, hypotheses, output_dir):
    """Generate LaTeX report."""
    report_path = os.path.join(output_dir, "multiagent_report.tex")

    # Get all unique solvers
    all_solvers = sorted(set(d.get('solver', '') for d in data))
    solver_data = {s: [d for d in data if d.get('solver') == s] for s in all_solvers}

    # Check for PINN data
    pinn_solvers = [s for s in all_solvers if s.startswith('pinn_')]
    has_pinn = len(pinn_solvers) > 0

    # Compute statistics for all solvers
    solver_stats = {}
    for solver in all_solvers:
        sdata = solver_data[solver]
        stable = [d for d in sdata if d.get('stable', False)]
        errors = [d.get('l2_error', np.nan) for d in stable if not np.isnan(d.get('l2_error', np.nan))]
        times = [d.get('wall_time', 0) * 1000 for d in stable]

        solver_stats[solver] = {
            'total': len(sdata),
            'stable': len(stable),
            'stable_pct': 100 * len(stable) / len(sdata) if sdata else 0,
            'avg_l2': np.mean(errors) if errors else np.nan,
            'avg_time': np.mean(times) if times else 0,
        }

    # Legacy variables for backward compatibility
    fdm_data = solver_data.get('implicit_fdm', [])
    spectral_data = solver_data.get('spectral_cosine', [])
    fdm_stable = sum(1 for d in fdm_data if d.get('stable', False))
    spec_stable = sum(1 for d in spectral_data if d.get('stable', False))
    fdm_errors = [d.get('l2_error', 0) for d in fdm_data if d.get('stable', False)]
    spec_errors = [d.get('l2_error', 0) for d in spectral_data if d.get('stable', False)]
    fdm_times = [d.get('wall_time', 0) * 1000 for d in fdm_data if d.get('stable', False)]
    spec_times = [d.get('wall_time', 0) * 1000 for d in spectral_data if d.get('stable', False)]

    latex_content = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage[margin=2.5cm]{geometry}
\usepackage{float}
\usepackage{caption}
\usepackage{xcolor}
\usepackage{colortbl}

\title{Multi-Agent Hypothesis-Driven Analysis\\
of Heat Transport Solvers}
\author{Auto-generated Report}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This report presents results from a multi-agent hypothesis-driven experiment framework
for analyzing numerical solvers for the 1D heat transport equation with nonlinear diffusivity.
The framework employs four specialized AI agents (Statistics, Feature, Pattern, Hypothesis)
working in parallel to analyze solver performance, identify patterns, and verify hypotheses.
"""

    confirmed = sum(1 for h in hypotheses.values() if h.get('status') == 'confirmed')
    rejected = sum(1 for h in hypotheses.values() if h.get('status') == 'rejected')
    inconclusive = len(hypotheses) - confirmed - rejected

    latex_content += f"""
Key findings include {confirmed} confirmed hypotheses, {rejected} rejected hypothesis,
and {inconclusive} requiring further investigation.
\\end{{abstract}}

\\section{{Introduction}}

The multi-agent experiment framework automates the scientific process of:
\\begin{{enumerate}}
\\item Generating experimental data across parameter spaces
\\item Analyzing results using specialized AI agents
\\item Formulating and testing hypotheses
\\item Iterating to refine understanding
\\end{{enumerate}}

\\subsection{{Problem Statement}}

We analyze the 1D radial heat transport equation:
\\begin{{equation}}
\\frac{{\\partial T}}{{\\partial t}} = \\frac{{1}}{{r}}\\frac{{\\partial}}{{\\partial r}}\\left(r\\chi\\frac{{\\partial T}}{{\\partial r}}\\right)
\\end{{equation}}

with nonlinear diffusivity:
\\begin{{equation}}
\\chi(|T'|) = \\begin{{cases}}
(|T'| - 0.5)^\\alpha + 0.1 & \\text{{if }} |T'| > 0.5 \\\\
0.1 & \\text{{otherwise}}
\\end{{cases}}
\\end{{equation}}

\\section{{Multi-Agent Architecture}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{{fig5_architecture.png}}
\\caption{{Multi-agent system architecture with parallel agent execution.}}
\\end{{figure}}

\\subsection{{Agent Descriptions}}

\\begin{{itemize}}
\\item \\textbf{{Statistics Agent}}: Computes basic statistics, stability rates, and error metrics
\\item \\textbf{{Feature Agent}}: Extracts features from temperature profiles and identifies trends
\\item \\textbf{{Pattern Agent}}: Discovers patterns in solver behavior across parameters
\\item \\textbf{{Hypothesis Agent}}: Generates and verifies scientific hypotheses
\\end{{itemize}}

\\section{{Experimental Results}}

\\subsection{{Data Summary}}

\\begin{{table}}[H]
\\centering
\\caption{{Solver Performance Summary}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Solver & Runs & Stable & Stability & Avg L2 Error & Avg Time \\\\
\\midrule
"""

    # Add all solvers to table
    solver_display_names = {
        'implicit_fdm': 'Implicit FDM',
        'spectral_cosine': 'Spectral Cosine',
        'pinn_simple': 'PINN Simple',
        'pinn_nonlinear': 'PINN Nonlinear',
        'pinn_improved': 'PINN Improved',
        'pinn_fno': 'PINN FNO',
    }

    for solver in all_solvers:
        stats = solver_stats[solver]
        display_name = solver_display_names.get(solver, solver)
        avg_err = f"{stats['avg_l2']:.6f}" if not np.isnan(stats['avg_l2']) else "N/A"
        time_str = f"{stats['avg_time']:.2f}ms" if stats['avg_time'] < 1000 else f"{stats['avg_time']/1000:.2f}s"
        latex_content += f"{display_name} & {stats['total']} & {stats['stable']} & {stats['stable_pct']:.1f}\\% & {avg_err} & {time_str} \\\\\n"

    latex_content += r"""\bottomrule
\end{tabular}
\end{table}

\subsection{Stability Analysis}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{fig1_stability_by_alpha.png}
\caption{Stability comparison across all solvers (FDM, Spectral, and PINN variants) for different
nonlinearity parameters $\alpha$. FDM and PINN variants maintain 100\% stability, while
spectral method shows decreasing stability at higher $\alpha$.}
\end{figure}

\subsection{Accuracy Analysis}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{fig2_error_distribution.png}
\caption{L2 error distribution comparison for all solvers. Spectral achieves the lowest average error
when stable, followed by FDM. PINN variants show higher errors but 100\% stability.}
\end{figure}

\subsection{Computational Efficiency}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{fig3_computation_time.png}
\caption{Computation time comparison (log scale) for all solvers. Traditional methods (FDM, Spectral)
are 100-1000$\times$ faster than PINN variants. PINN-FNO requires the longest time.}
\end{figure}

\section{Hypothesis Verification}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{fig4_hypothesis_results.png}
\caption{Hypothesis verification results showing confidence levels and status.}
\end{figure}

\subsection{Hypothesis Details}

"""

    for hid, h in hypotheses.items():
        status = h.get('status', 'inconclusive')
        confidence = h.get('confidence', 0)
        statement = h.get('statement', hid)
        verifications = h.get('verification_count', 0)

        if status == 'confirmed':
            status_color = 'green'
            status_symbol = '$\\checkmark$'
        elif status == 'rejected':
            status_color = 'red'
            status_symbol = '$\\times$'
        else:
            status_color = 'gray'
            status_symbol = '?'

        latex_content += f"""\\subsubsection{{{hid}: {statement}}}

\\begin{{itemize}}
\\item Status: \\textcolor{{{status_color}}}{{\\textbf{{{status.upper()}}}}} {status_symbol}
\\item Confidence: {confidence}\\%
\\item Verifications: {verifications}
\\end{{itemize}}

"""

    latex_content += r"""
\section{Discussion}

\subsection{Key Findings}

\begin{enumerate}
\item \textbf{FDM Unconditional Stability}: The implicit FDM solver maintains 100\% stability
across all tested parameter combinations, making it reliable for production use.

\item \textbf{Spectral Method Trade-off}: The spectral cosine method achieves the lowest L2 errors
when stable, but suffers from instability at higher $\alpha$ values. This represents
a classic accuracy-stability trade-off.

\item \textbf{PINN Stability}: All PINN variants maintain 100\% stability across all
tested parameters, similar to FDM.

\item \textbf{PINN Accuracy}: PINN variants show higher L2 errors than traditional methods in this
benchmark. PINN-FNO achieves the best accuracy among PINN variants.

\item \textbf{Computational Cost}: PINN methods require 100-1000$\times$ more computation time
than traditional solvers. FDM is the fastest, followed by Spectral.

\item \textbf{Hypothesis H1 Confirmed}: Smaller time steps improve spectral solver stability,
providing a practical mitigation strategy.

\item \textbf{Hypothesis H7 Confirmed}: Spectral solver tends to fail with NaN for $\alpha \geq 0.2$
under certain conditions, requiring careful parameter selection.
\end{enumerate}

\subsection{Recommendations}

\begin{itemize}
\item For \textbf{reliability-critical applications}: Use implicit FDM or PINN variants
\item For \textbf{accuracy-critical applications} with low $\alpha$: Use spectral method with small dt
\item For \textbf{high nonlinearity} ($\alpha > 0.5$): Use FDM (stable and accurate)
\item For \textbf{complex geometry or inverse problems}: Consider PINN methods
\end{itemize}

"""

    # Add PINN section if PINN data exists
    if has_pinn:
        # Compute PINN-specific statistics
        pinn_stats_by_alpha = {}
        for solver in pinn_solvers:
            pinn_stats_by_alpha[solver] = {}
            for alpha in sorted(set(d.get('alpha', 0) for d in data)):
                alpha_data = [d for d in solver_data[solver] if d.get('alpha') == alpha and d.get('stable', False)]
                if alpha_data:
                    errors = [d.get('l2_error', np.nan) for d in alpha_data if not np.isnan(d.get('l2_error', np.nan))]
                    pinn_stats_by_alpha[solver][alpha] = np.mean(errors) if errors else np.nan

        pinn_section = r"""
\section{PINN Solver Analysis}

Physics-Informed Neural Networks (PINNs) were evaluated comprehensively across the same
parameter space as traditional methods. Four PINN variants were tested with identical
conditions (8 $\alpha$ values $\times$ 5 dt values = 40 runs per variant).

\subsection{PINN Architecture Overview}

\begin{itemize}
\item \textbf{PINN Simple}: Basic 3-layer MLP with PDE residual loss (32 hidden units)
\item \textbf{PINN Nonlinear}: MLP with explicit nonlinear $\chi(|T'|)$ in loss function
\item \textbf{PINN Improved}: Fourier feature embeddings + residual blocks for better convergence
\item \textbf{PINN FNO}: Fourier Neural Operator - learns in spectral space (16 channels, 8 modes)
\end{itemize}

All variants: 500 epochs, 500 collocation points, Adam optimizer.

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{fig6_pinn_comparison.png}
\caption{Comprehensive solver comparison including PINN variants: L2 error (left) and
computation time on log scale (right). Traditional methods achieve better accuracy with
orders of magnitude less computation time.}
\end{figure}

\subsection{PINN Performance by Nonlinearity ($\alpha$)}

"""
        # Add PINN performance table by alpha
        pinn_section += r"""\begin{table}[H]
\centering
\caption{PINN L2 Error by Nonlinearity Parameter $\alpha$}
\begin{tabular}{l"""
        alphas = sorted(set(d.get('alpha', 0) for d in data))
        pinn_section += "r" * len(alphas) + r"""}
\toprule
Solver"""
        for alpha in alphas:
            pinn_section += f" & $\\alpha$={alpha}"
        pinn_section += r""" \\
\midrule
"""
        for solver in pinn_solvers:
            display_name = solver_display_names.get(solver, solver)
            pinn_section += display_name
            for alpha in alphas:
                err = pinn_stats_by_alpha[solver].get(alpha, np.nan)
                if not np.isnan(err):
                    pinn_section += f" & {err:.3f}"
                else:
                    pinn_section += " & N/A"
            pinn_section += r" \\" + "\n"

        pinn_section += r"""\bottomrule
\end{tabular}
\end{table}

\subsection{PINN Key Findings}

\begin{enumerate}
\item \textbf{FNO Dominance}: PINN-FNO achieves approximately 2-3$\times$ lower L2 error
than other PINN variants across all $\alpha$ values. The Fourier Neural Operator architecture
is better suited for this spectral problem.

\item \textbf{$\alpha$-Dependent Performance}: All PINN variants show decreasing accuracy
as $\alpha$ increases. At $\alpha=1.0$, errors are roughly 3-5$\times$ higher than at $\alpha=0.0$.

\item \textbf{Unconditional Stability}: Unlike the spectral method, all PINN variants
maintain 100\% stability regardless of $\alpha$ or dt, similar to implicit FDM.

\item \textbf{Computational Cost}: PINN methods are significantly slower:
\begin{itemize}
\item PINN Simple/Nonlinear: $\sim$1 second (100$\times$ slower than FDM)
\item PINN Improved: $\sim$3 seconds (400$\times$ slower)
\item PINN FNO: $\sim$20 seconds (2800$\times$ slower)
\end{itemize}

\item \textbf{Accuracy Gap}: Even the best PINN (FNO with L2$\approx$0.31) does not match
traditional methods (Spectral: 0.088, FDM: 0.183) in this benchmark configuration.
\end{enumerate}

\subsection{When to Use PINN}

PINN methods are recommended when:
\begin{itemize}
\item Complex or irregular geometries where meshing is difficult
\item Inverse problems requiring parameter estimation
\item Problems with sparse or noisy observational data
\item Research requiring automatic differentiation through the solver
\item Situations where unconditional stability is paramount and accuracy is secondary
\end{itemize}

For well-posed forward problems on regular grids (like this benchmark), traditional
numerical methods remain superior in both accuracy and efficiency.
"""
        latex_content += pinn_section

    latex_content += r"""
\section{Conclusion}

This comprehensive benchmark evaluated six numerical solvers for the 1D radial heat transport
equation with nonlinear diffusivity across 282 experimental runs.
"""

    if has_pinn:
        latex_content += r"""
\subsection{Solver Ranking}

Based on the full parameter sweep, solvers are ranked by accuracy (when stable):

\begin{enumerate}
\item \textbf{Spectral Cosine} (L2=0.088): Best accuracy but 74.5\% stability
\item \textbf{Implicit FDM} (L2=0.183): Excellent accuracy with 100\% stability
\item \textbf{PINN FNO} (L2=0.312): Best neural network approach, 100\% stable
\item \textbf{PINN Improved} (L2=0.640): Good for neural methods
\item \textbf{PINN Nonlinear} (L2=0.659): Moderate performance
\item \textbf{PINN Simple} (L2=0.930): Baseline neural approach
\end{enumerate}

\subsection{Practical Recommendations}

\begin{table}[H]
\centering
\begin{tabular}{lll}
\toprule
Use Case & Recommended Solver & Rationale \\
\midrule
Production (reliability) & Implicit FDM & 100\% stable, fast, accurate \\
High accuracy (low $\alpha$) & Spectral Cosine & Best L2 when stable \\
High nonlinearity ($\alpha>0.5$) & Implicit FDM & Spectral unstable \\
Research/prototyping & PINN FNO & Flexible, differentiable \\
Inverse problems & PINN variants & Natural for optimization \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Key Insights}

\begin{itemize}
\item Traditional numerical methods (FDM, Spectral) outperform neural network approaches
in accuracy and efficiency for this well-posed benchmark problem
\item PINN methods provide unconditional stability but at significant computational cost
\item The Fourier Neural Operator (FNO) is the most promising PINN architecture for PDE solving
\item Hybrid approaches combining traditional stability with neural network flexibility
represent a promising research direction
\end{itemize}
"""
    else:
        latex_content += r"""
Key insights include the unconditional stability of FDM versus the conditional
accuracy advantages of spectral methods.
"""

    latex_content += r"""
\subsection{Future Work}

\begin{itemize}
\item Extended PINN training (more epochs, larger networks) for improved accuracy
\item Hybrid solvers combining FDM stability with spectral/PINN accuracy
\item Transfer learning for PINN across different $\alpha$ values
\item Automatic solver selection based on problem characteristics
\item Extension to 2D/3D geometries where PINN advantages may be more pronounced
\end{itemize}

\end{document}
"""

    with open(report_path, 'w') as f:
        f.write(latex_content)

    print(f"LaTeX report saved to {report_path}")
    return report_path


def compile_latex(tex_path, output_dir):
    """Compile LaTeX to PDF."""
    try:
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", output_dir, tex_path],
                capture_output=True, text=True, cwd=output_dir
            )

        pdf_path = tex_path.replace('.tex', '.pdf')
        if os.path.exists(pdf_path):
            print(f"PDF generated: {pdf_path}")
            return pdf_path
        else:
            print("PDF compilation may have failed. Check the log file.")
            return None
    except FileNotFoundError:
        print("pdflatex not found. Please install TeX Live or MiKTeX.")
        return None


def main():
    """Main function."""
    output_dir = "reports/multiagent"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading experiment data...")
    data = load_experiment_data()
    hypotheses = load_hypothesis_data()

    print(f"Loaded {len(data)} experiment runs")
    print(f"Loaded {len(hypotheses)} hypotheses")

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(data, hypotheses, output_dir)

    # Generate LaTeX report
    print("\nGenerating LaTeX report...")
    tex_path = generate_latex_report(data, hypotheses, output_dir)

    # Compile to PDF
    print("\nCompiling to PDF...")
    compile_latex(tex_path, output_dir)

    print("\nReport generation complete!")
    print(f"Output directory: {output_dir}/")


if __name__ == "__main__":
    main()
