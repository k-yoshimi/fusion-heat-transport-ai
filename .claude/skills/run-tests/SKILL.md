---
name: run-tests
description: Run tests and fix failures if any
user-invocable: true
allowed-tools: Bash, Read, Edit, Write, Glob, Grep
---

# Run Tests

Execute the test suite and fix any failures.

## Steps

1. Run `python -m pytest tests/ -v`
2. If all tests pass, report results and finish
3. If any tests fail:
   - Analyze the error messages of failed tests
   - Read the relevant source code
   - Explain the fix to the user before applying it
   - Re-run tests to confirm the fix
4. Append bug fix details to HISTORY.md
