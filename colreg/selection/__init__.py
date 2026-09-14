import numpy as np

from .threshold import CorrelationThresholdSelector, VIFSelector

__all__ = [
    "CorrelationThresholdSelector",
    "VIFSelector",
    "PassthroughSelector",
    "PCASelector",
]


class PassthroughSelector:
    """The control arm.  Does nothing, on purpose.

    Every claim in the paper is relative to this.  A benchmark of selection
    methods that omits 'no selection at all' cannot answer whether selection
    helps, which is the question.
    """

    def fit(self, X, y=None, **kw):
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = X.shape[1]
        self.support_ = np.ones(X.shape[1], dtype=bool)
        self.selection_order_ = []
        self.target_was_selected_for_removal_ = False
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)

    def fit_transform(self, X, y=None, **kw):
        return self.fit(X, y, **kw).transform(X)


class PCASelector:
    """Retain enough principal components to explain ``var_target`` variance.

    The other standard answer to collinearity, and a useful upper reference:
    PCA removes collinearity by construction, at the cost of interpretable
    coefficients.  If pruning never beats PCA on accuracy and never beats
    'no selection' on stability, pruning has no remaining justification.

    Implemented on the SVD of the centred, scaled matrix so that it is
    comparable across datasets with different units.
    """

    def __init__(self, var_target: float = 0.95):
        self.var_target = var_target

    def fit(self, X, y=None, **kw):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        Z = (X - self.mean_) / self.scale_
        _, s, vt = np.linalg.svd(Z, full_matrices=False)
        ratio = s ** 2 / np.sum(s ** 2)
        k = int(np.searchsorted(np.cumsum(ratio), self.var_target) + 1)
        k = max(1, min(k, vt.shape[0]))
        self.components_ = vt[:k]
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = k
        self.support_ = np.ones(X.shape[1], dtype=bool)   # PCA drops no columns
        self.selection_order_ = []
        self.target_was_selected_for_removal_ = False
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return ((X - self.mean_) / self.scale_) @ self.components_.T

    def fit_transform(self, X, y=None, **kw):
        return self.fit(X, y, **kw).transform(X)
