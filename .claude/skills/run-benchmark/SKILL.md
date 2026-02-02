---
name: run-benchmark
description: Run the benchmark suite and report results
user-invocable: true
allowed-tools: Bash, Read
---

# Run Benchmark

Execute the benchmark suite and report results.

## Steps

1. First verify tests pass with `python -m pytest tests/ -v`
2. If the user specified arguments, use those; otherwise run with defaults:
   ```
   python -m app.run_benchmark --alpha 0.0 0.5 1.0
   ```
3. Read `outputs/benchmark.csv` and `outputs/benchmark.md`
4. Report results to the user covering:
   - L2/L-infinity error for each solver
   - Wall-clock time
   - Best solver and why
   - Differences from previous run (if any)
