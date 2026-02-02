---
name: analyze-results
description: Analyze benchmark results in detail
user-invocable: true
allowed-tools: Bash, Read, Glob
---

# Analyze Results

Perform a detailed analysis of the latest benchmark results in outputs/.

## Steps

1. Read `outputs/benchmark.csv`
2. Analyze from the following perspectives:
   - **Accuracy comparison**: L2/L-infinity error per solver per alpha
   - **Speed comparison**: Wall-clock time comparison
   - **Feature analysis**: Physical meaning of max_abs_gradient, energy_content, etc.
   - **Nonlinearity effects**: How accuracy changes as alpha increases
   - **Trade-offs**: Pareto front of accuracy vs speed
3. Provide improvement recommendations:
   - Recommended grid resolution
   - Optimal solver for each alpha value
   - Parameter tuning suggestions for better accuracy
4. Summarize findings clearly
