---
name: refine-reference
description: Verify and improve reference solution accuracy
user-invocable: true
allowed-tools: Bash, Read, Edit, Write
---

# Refine Reference

Verify the accuracy of the reference solution and improve it if needed.

## Steps

1. Read `compute_reference()` in `app/run_benchmark.py`
2. Check current settings (grid refinement factor, time step factor)
3. Convergence test: compare results at different refinement levels
   ```python
   # Compare reference at 4x, 8x, 16x refinement
   ```
4. Report findings:
   - Estimated accuracy of the current reference
   - Recommended settings for higher accuracy
   - Trade-off with computation cost
5. If the user approves, update `compute_reference()`
6. Update HISTORY.md
