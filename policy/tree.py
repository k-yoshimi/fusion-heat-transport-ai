"""Numpy-only decision tree classifier for solver selection."""

import numpy as np


class NumpyDecisionTree:
    """Simple CART decision tree using Gini impurity. No sklearn needed."""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit tree on feature matrix X (n_samples, n_features) and labels y."""
        self.classes_ = np.unique(y)
        self.tree_ = self._build(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class for each row in X."""
        if self.tree_ is None:
            raise RuntimeError("Tree not fitted yet.")
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def save(self, path: str):
        """Save tree to .npz file."""
        np.savez(path, tree=np.array(self.tree_, dtype=object),
                 classes=self.classes_)

    def load(self, path: str):
        """Load tree from .npz file."""
        data = np.load(path, allow_pickle=True)
        self.tree_ = data["tree"].item()
        self.classes_ = data["classes"]
        return self

    # -- private --

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        best_gain = -1.0
        best_feat = None
        best_thr = None

        parent_gini = self._gini(y)

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, feat], np.linspace(0, 100, 21))

            for thr in thresholds:
                left = y[X[:, feat] <= thr]
                right = y[X[:, feat] > thr]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = parent_gini - (
                    len(left) / n_samples * self._gini(left)
                    + len(right) / n_samples * self._gini(right)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thr = thr

        return best_feat, best_thr, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        # Leaf conditions
        if (depth >= self.max_depth
                or len(y) < self.min_samples_split
                or len(np.unique(y)) == 1):
            vals, counts = np.unique(y, return_counts=True)
            return {"leaf": True, "class": vals[np.argmax(counts)]}

        feat, thr, gain = self._best_split(X, y)
        if feat is None or gain <= 0:
            vals, counts = np.unique(y, return_counts=True)
            return {"leaf": True, "class": vals[np.argmax(counts)]}

        mask = X[:, feat] <= thr
        return {
            "leaf": False,
            "feature": feat,
            "threshold": thr,
            "left": self._build(X[mask], y[mask], depth + 1),
            "right": self._build(X[~mask], y[~mask], depth + 1),
        }

    def _predict_one(self, x: np.ndarray, node: dict):
        if node["leaf"]:
            return node["class"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])
