"""Exact-split CART regression tree in NumPy.

Rebuilt from a straightforward textbook implementation for three reasons that
matter to the study:

1.  The original scanned candidate thresholds using a moving average over the
    *unique* values of a feature and compared each split's MSE against the
    parent's MSE (``mse_base``), updating ``mse_base`` inside the loop.  That
    makes the accepted split depend on the order in which candidates happen to
    be visited.  Here the criterion is the standard weighted-child MSE
    reduction, evaluated for every candidate before any is accepted.

2.  Splits are found with a prefix-sum sweep over the sorted feature rather
    than by materialising the two child arrays for every candidate, which turns
    an O(n^2) inner loop into O(n log n) per feature.  The permutation study in
    RQ5 fits ~100x more trees than a normal benchmark, so this is not premature.

3.  ``feature_importances_`` is exposed, because RQ3 measures how importance
    mass is redistributed when correlated features are removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["RegressionTree"]


@dataclass
class _Node:
    value: float
    n: int
    mse: float
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = field(default=None, repr=False)
    right: "_Node | None" = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class RegressionTree:
    """CART with the squared-error criterion.

    Parameters mirror ``sklearn.tree.DecisionTreeRegressor`` so that the two can
    be compared directly: ``max_depth``, ``min_samples_split``,
    ``min_samples_leaf``.  Thresholds are midpoints between consecutive distinct
    sorted values, matching sklearn's convention.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    # ---------------------------------------------------------------- fitting

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        self._importances = np.zeros(self.n_features_in_)
        self.root_ = self._build(X, y)
        total = self._importances.sum()
        self.feature_importances_ = (
            self._importances / total if total > 0 else self._importances
        )
        return self

    @staticmethod
    def _leaf(y):
        n = y.size
        value = float(y.mean()) if n else 0.0
        mse = float(((y - value) ** 2).mean()) if n else 0.0
        return _Node(value=value, n=n, mse=mse)

    def _build(self, X, y):
        """Grow the tree with an explicit stack rather than recursion.

        ``max_depth=None`` is one of the tuned levels in the CART grid, and with
        ``min_samples_leaf=1`` a split may peel off a single sample, so tree
        depth can approach n. On the large datasets in this corpus (n up to
        36,733) that overran CPython's 1000-frame recursion limit and killed a
        worker mid-run.

        Nodes are expanded in pre-order -- push right, then left, so left is
        popped first -- which is the same order the recursive implementation
        used. That matters: ``_importances`` is accumulated by summing floats as
        nodes are expanded, so preserving the order keeps the result
        bit-identical to trees grown before this change, and results already on
        disk stay comparable with results produced after it.
        """
        root = self._leaf(y)
        stack = [(root, np.arange(y.size), 0)]

        while stack:
            node, idx, depth = stack.pop()
            n = idx.size

            depth_ok = self.max_depth is None or depth < self.max_depth
            if not depth_ok or n < self.min_samples_split or node.mse <= 0.0:
                continue

            # Materialise this node's view only for as long as the split search
            # needs it. What goes ON the stack is an index array, never a copy
            # of the data: an entry costs 8 bytes per row instead of 8*p, and
            # the pending right siblings of a deep tree are what would otherwise
            # exhaust memory. At depth 1000 on a 21,263 x 81 dataset the stack
            # would hold ~13.5 GB of float64 slices; as indices it is ~170 MB.
            Xn, yn = X[idx], y[idx]

            feature, threshold, gain = self._best_split(Xn, yn)
            if feature is None:
                continue

            mask = Xn[:, feature] <= threshold
            node.feature, node.threshold = feature, float(threshold)
            # Weighted impurity decrease, the same quantity sklearn accumulates.
            self._importances[feature] += gain * n

            # Boolean masking preserves ascending order, so idx stays sorted and
            # X[idx] holds exactly the rows, in exactly the order, that repeated
            # X[mask] slicing produced. The trees are identical, not merely
            # equivalent.
            left_idx, right_idx = idx[mask], idx[~mask]
            node.left = self._leaf(y[left_idx])
            node.right = self._leaf(y[right_idx])
            stack.append((node.right, right_idx, depth + 1))
            stack.append((node.left, left_idx, depth + 1))

        return root

    def _best_split(self, X, y):
        n = y.size
        parent_sse = float(((y - y.mean()) ** 2).sum())
        best = (None, None, 0.0)
        best_sse = parent_sse
        m = self.min_samples_leaf

        for j in range(X.shape[1]):
            col = X[:, j]
            order = np.argsort(col, kind="mergesort")
            xs, ys = col[order], y[order]

            csum = np.cumsum(ys)
            csq = np.cumsum(ys ** 2)
            total, total_sq = csum[-1], csq[-1]

            # Split after position i (0-based): left = xs[:i+1], right = rest.
            k = np.arange(1, n)                       # left sizes
            left_sum, left_sq = csum[: n - 1], csq[: n - 1]
            right_sum, right_sq = total - left_sum, total_sq - left_sq
            right_k = n - k

            sse = (left_sq - left_sum ** 2 / k) + (right_sq - right_sum ** 2 / right_k)

            # Valid only where the value actually changes and both leaves are
            # large enough; a split between two equal values is not a split.
            valid = (xs[:-1] < xs[1:]) & (k >= m) & (right_k >= m)
            if not valid.any():
                continue

            sse = np.where(valid, sse, np.inf)
            i = int(np.argmin(sse))
            if sse[i] < best_sse - 1e-12:
                best_sse = float(sse[i])
                thr = (xs[i] + xs[i + 1]) / 2.0
                best = (j, thr, (parent_sse - sse[i]) / n)

        return best

    # ------------------------------------------------------------- prediction

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._descend(self.root_, row) for row in X])

    @staticmethod
    def _descend(node, row):
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def get_params(self, deep=True):
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
