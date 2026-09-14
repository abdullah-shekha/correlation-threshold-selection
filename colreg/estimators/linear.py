"""Penalised linear regressors in NumPy.

Every estimator here is written so that its objective is stated explicitly in the
docstring and its solution matches the corresponding scikit-learn estimator to
within ``PARITY_TOL`` (see ``tests/test_parity.py``).  That parity assertion is
what turns a from-scratch implementation from a redundant exercise into a
verifiable reference implementation.

Objectives (all on column-centred X and y; the intercept is never penalised):

    OLS         (1/2n)||y - Xw||^2
    Ridge             ||y - Xw||^2 + alpha ||w||^2
    Lasso       (1/2n)||y - Xw||^2 + alpha ||w||_1
    ElasticNet  (1/2n)||y - Xw||^2 + alpha*l1_ratio ||w||_1
                                   + 0.5*alpha*(1-l1_ratio) ||w||^2

The Ridge scaling follows sklearn.linear_model.Ridge (no 1/2n factor); the
Lasso and ElasticNet scalings follow sklearn.linear_model.Lasso / ElasticNet.
Mixing these conventions is the single most common reason two "identical"
implementations disagree, so they are pinned here deliberately.
"""

from __future__ import annotations

import numpy as np

from ._cd import HAVE_NUMBA, enet_path, solve_enet, warm_up

__all__ = [
    "HAVE_NUMBA",
    "enet_path",
    "solve_enet",
    "warm_up",
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "ElasticNetRegression",
]


class _BaseLinear:
    """Shared intercept handling.

    Centring X and y, fitting without an intercept, then recovering
    ``b = mean(y) - mean(X) @ w`` is exactly equivalent to fitting an
    unpenalised intercept, and it is the only way to keep the intercept out of
    the penalty term.  The original notebooks penalised nothing but also
    updated the intercept with a gradient of the wrong sign; both problems
    disappear with this formulation.
    """

    fit_intercept: bool = True

    def _center(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if self.fit_intercept:
            x_off = X.mean(axis=0)
            y_off = y.mean()
            return X - x_off, y - y_off, x_off, y_off
        return X, y, np.zeros(X.shape[1]), 0.0

    def _set_intercept(self, x_off, y_off):
        self.intercept_ = float(y_off - x_off @ self.coef_) if self.fit_intercept else 0.0

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in self._param_names}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class LinearRegression(_BaseLinear):
    """Ordinary least squares via the pseudo-inverse.

    Solved in closed form rather than by gradient descent.  Gradient descent on
    an unscaled design matrix with a hand-picked learning rate is the reason the
    original notebook's weights never left the neighbourhood of zero; it adds
    two free parameters (lr, n_iter) that have nothing to do with the research
    question and everything to do with whether the numbers mean anything.
    """

    _param_names = ("fit_intercept",)

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        Xc, yc, x_off, y_off = self._center(X, y)
        self.coef_, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        self._set_intercept(x_off, y_off)
        return self


class RidgeRegression(_BaseLinear):
    """L2-penalised least squares, closed form.

    w = (X'X + alpha I)^-1 X'y on centred data.  Uses the dual form when p > n,
    which is both faster and better conditioned in the wide regime that the
    high-collinearity strata of this study deliberately include.
    """

    _param_names = ("alpha", "fit_intercept")

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        Xc, yc, x_off, y_off = self._center(X, y)
        n, p = Xc.shape
        if p <= n:
            A = Xc.T @ Xc + self.alpha * np.eye(p)
            self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        else:  # dual form: w = X' (XX' + alpha I)^-1 y
            A = Xc @ Xc.T + self.alpha * np.eye(n)
            self.coef_ = Xc.T @ np.linalg.solve(A, yc)
        self._set_intercept(x_off, y_off)
        return self


def _soft_threshold(rho: float, thresh: float) -> float:
    if rho > thresh:
        return rho - thresh
    if rho < -thresh:
        return rho + thresh
    return 0.0


class ElasticNetRegression(_BaseLinear):
    """Cyclic coordinate descent for the elastic net.

    Coordinate update, derived from the subgradient of the objective above:

        rho_j = x_j' (y - X w + x_j w_j)
        w_j   = S(rho_j / n, alpha * l1_ratio)
                / (z_j / n + alpha * (1 - l1_ratio))

    where z_j = x_j'x_j and S is the soft-thresholding operator.  The
    ``l1_ratio`` term is what the original notebook's Lasso was missing: it
    added a bare ``+ alpha`` to the gradient, which shifts every coordinate by
    the same constant regardless of sign and therefore never sets a coefficient
    to zero.  Without ``np.sign``, an L1 penalty is not an L1 penalty.
    """

    _param_names = ("alpha", "l1_ratio", "fit_intercept", "max_iter", "tol")

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 10_000,
        tol: float = 1e-8,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y, coef_init=None):
        """Fit by cyclic coordinate descent.

        ``coef_init`` warm-starts from a previous solution.  Passing the
        solution for a neighbouring alpha typically cuts the iteration count by
        an order of magnitude; the harness exploits this when sweeping the
        tuning grid.  The result is identical either way -- only the path taken
        to it differs.
        """
        Xc, yc, x_off, y_off = self._center(X, y)
        self.coef_, self.n_iter_ = solve_enet(
            Xc, yc,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
            coef_init=coef_init,
        )
        self._set_intercept(x_off, y_off)
        return self


class LassoRegression(ElasticNetRegression):
    """Pure L1: the elastic net at ``l1_ratio = 1``."""

    _param_names = ("alpha", "fit_intercept", "max_iter", "tol")

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 10_000,
        tol: float = 1e-8,
    ):
        super().__init__(
            alpha=alpha,
            l1_ratio=1.0,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
        )
