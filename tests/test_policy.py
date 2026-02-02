"""Tests for solver selection policy."""

import math
import pytest
from policy.select import select_best


def test_select_lowest_score():
    results = [
        {"name": "A", "l2_error": 0.01, "wall_time": 1.0},
        {"name": "B", "l2_error": 0.001, "wall_time": 0.5},
    ]
    best = select_best(results, lam=0.1)
    assert best["name"] == "B"


def test_select_time_penalty():
    results = [
        {"name": "fast_bad", "l2_error": 0.1, "wall_time": 0.01},
        {"name": "slow_good", "l2_error": 0.001, "wall_time": 10.0},
    ]
    # With high lambda, fast solver wins
    best = select_best(results, lam=1.0)
    assert best["name"] == "fast_bad"

    # With low lambda, accurate solver wins
    best = select_best(results, lam=0.001)
    assert best["name"] == "slow_good"


def test_select_skips_nan():
    results = [
        {"name": "nan_solver", "l2_error": float("nan"), "wall_time": 0.0},
        {"name": "ok_solver", "l2_error": 0.01, "wall_time": 1.0},
    ]
    best = select_best(results)
    assert best["name"] == "ok_solver"


def test_select_all_nan_raises():
    results = [
        {"name": "bad", "l2_error": float("nan"), "wall_time": 0.0},
    ]
    with pytest.raises(ValueError):
        select_best(results)
