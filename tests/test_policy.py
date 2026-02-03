"""Tests for solver selection policy."""

import math
import numpy as np
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


# --- ML selector tests ---


def test_extract_initial_features_keys():
    from features.extract import extract_initial_features
    r = np.linspace(0, 1, 51)
    T0 = 1.0 - r ** 2
    feats = extract_initial_features(T0, r, alpha=0.5, nr=51, dt=0.001,
                                     t_end=0.1)
    expected_keys = {
        "alpha", "nr", "dt", "t_end",
        "max_abs_gradient", "energy_content", "max_chi", "max_laplacian",
        "T_center", "gradient_sharpness", "chi_ratio", "problem_stiffness",
        "half_max_radius", "profile_centroid", "gradient_slope", "profile_width",
    }
    assert set(feats.keys()) == expected_keys
    assert len(feats) == 16
    assert feats["alpha"] == 0.5


def test_numpy_decision_tree_fit_predict():
    from policy.tree import NumpyDecisionTree
    # Simple 2-class problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array(["A", "A", "B", "B"])
    tree = NumpyDecisionTree(max_depth=3)
    tree.fit(X, y)
    preds = tree.predict(X)
    assert list(preds) == list(y)


def test_numpy_decision_tree_save_load(tmp_path):
    from policy.tree import NumpyDecisionTree
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array(["A", "A", "B", "B"])
    tree = NumpyDecisionTree(max_depth=3)
    tree.fit(X, y)

    path = str(tmp_path / "model.npz")
    tree.save(path)

    tree2 = NumpyDecisionTree()
    tree2.load(path)
    preds = tree2.predict(X)
    assert list(preds) == list(y)


def test_select_with_ml(tmp_path):
    from policy.tree import NumpyDecisionTree
    from policy.train import FEATURE_NAMES
    from policy.select import select_with_ml

    # Train a trivial model that always predicts "implicit_fdm"
    n = 20
    X = np.random.rand(n, len(FEATURE_NAMES))
    y = np.array(["implicit_fdm"] * n)
    tree = NumpyDecisionTree(max_depth=2)
    tree.fit(X, y)
    model_path = str(tmp_path / "test_model.npz")
    tree.save(model_path)

    r = np.linspace(0, 1, 51)
    T0 = 1.0 - r ** 2
    result = select_with_ml(T0, r, 0.5, 51, 0.001, 0.1, model_path)
    assert result == "implicit_fdm"


def test_append_training_sample(tmp_path):
    import csv
    from policy.train import append_training_sample, FEATURE_NAMES

    csv_path = str(tmp_path / "data.csv")
    feats = {f: 0.0 for f in FEATURE_NAMES}
    feats["alpha"] = 1.0

    # First append creates the file with header
    append_training_sample(feats, "implicit_fdm", csv_path)
    # Second append adds a row
    append_training_sample(feats, "cosine_spectral", csv_path)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["best_solver"] == "implicit_fdm"
    assert rows[1]["best_solver"] == "cosine_spectral"
