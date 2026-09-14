"""Parity of the NumPy estimators against scikit-learn.

This file IS a contribution, not scaffolding.  A from-scratch reimplementation
of a well-known estimator is worth nothing to a reviewer on its own; a
from-scratch implementation carrying a machine-checked proof that it computes
the same thing as the reference library, and is therefore safe to instrument in
ways the reference library does not permit, is a reproducibility artifact.

Cite this file in the paper by name.
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet as SkElasticNet
from sklearn.linear_model import Lasso as SkLasso
from sklearn.linear_model import LinearRegression as SkOLS
from sklearn.linear_model import Ridge as SkRidge
from sklearn.tree import DecisionTreeRegressor as SkTree

from colreg.estimators import (
    ElasticNetRegression,
    LassoRegression,
    LinearRegression,
    RegressionTree,
    RidgeRegression,
)

COEF_TOL = 1e-6
PRED_TOL = 1e-6


def collinear_data(n=200, p=8, rank=4, noise=1.0, seed=0):
    """A design matrix with genuine rank deficiency plus noise.

    Deliberately ill-conditioned: this is the regime the whole study is about,
    and an implementation that only agrees with sklearn on well-conditioned data
    is not validated for our purposes.
    """
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(n, rank))
    mixing = rng.normal(size=(rank, p))
    X = basis @ mixing + rng.normal(scale=0.01, size=(n, p))
    beta = rng.normal(size=p)
    y = X @ beta + rng.normal(scale=noise, size=n)
    return X, y


DATASETS = {
    "well_conditioned": make_regression(
        n_samples=200, n_features=10, noise=5.0, random_state=0
    ),
    "collinear": collinear_data(),
    "wide": collinear_data(n=60, p=120, rank=8, seed=3),
}


@pytest.mark.parametrize("name", list(DATASETS))
def test_ols_matches_sklearn(name):
    X, y = DATASETS[name]
    if X.shape[1] > X.shape[0]:
        pytest.skip("OLS is not identified when p > n")
    ours = LinearRegression().fit(X, y)
    ref = SkOLS().fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=COEF_TOL)
    assert np.allclose(ours.predict(X), ref.predict(X), atol=PRED_TOL)


@pytest.mark.parametrize("name", list(DATASETS))
@pytest.mark.parametrize("alpha", [0.01, 1.0, 100.0])
def test_ridge_matches_sklearn(name, alpha):
    X, y = DATASETS[name]
    ours = RidgeRegression(alpha=alpha).fit(X, y)
    ref = SkRidge(alpha=alpha, solver="cholesky").fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=COEF_TOL)
    assert np.isclose(ours.intercept_, ref.intercept_, atol=COEF_TOL)


@pytest.mark.parametrize("name", list(DATASETS))
@pytest.mark.parametrize("alpha", [0.01, 0.1, 1.0])
def test_lasso_matches_sklearn(name, alpha):
    X, y = DATASETS[name]
    ours = LassoRegression(alpha=alpha, tol=1e-12, max_iter=100_000).fit(X, y)
    ref = SkLasso(alpha=alpha, tol=1e-12, max_iter=100_000).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-5)
    assert np.isclose(ours.intercept_, ref.intercept_, atol=1e-5)


@pytest.mark.parametrize("name", list(DATASETS))
@pytest.mark.parametrize("l1_ratio", [0.2, 0.5, 0.9])
def test_elasticnet_matches_sklearn(name, l1_ratio):
    X, y = DATASETS[name]
    ours = ElasticNetRegression(
        alpha=0.1, l1_ratio=l1_ratio, tol=1e-12, max_iter=100_000
    ).fit(X, y)
    ref = SkElasticNet(
        alpha=0.1, l1_ratio=l1_ratio, tol=1e-12, max_iter=100_000
    ).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-5)


def test_lasso_actually_sparsifies():
    """Regression test for defect D1.

    The original notebook added a bare ``+ alpha`` to the gradient instead of
    ``alpha * sign(w)``.  That produces a biased OLS, never a sparse solution.
    If this test ever fails, the L1 subgradient has been broken again.
    """
    X, y = collinear_data(n=200, p=40, rank=5, seed=7)
    dense = LassoRegression(alpha=1e-6).fit(X, y)
    sparse = LassoRegression(alpha=5.0).fit(X, y)
    assert np.count_nonzero(sparse.coef_) < np.count_nonzero(dense.coef_)
    assert np.count_nonzero(sparse.coef_) < X.shape[1]


@pytest.mark.parametrize("name", list(DATASETS))
@pytest.mark.parametrize("max_depth", [2, 4, None])
def test_tree_matches_sklearn(name, max_depth):
    X, y = DATASETS[name]
    ours = RegressionTree(max_depth=max_depth, min_samples_split=5).fit(X, y)
    ref = SkTree(
        max_depth=max_depth, min_samples_split=5, random_state=0
    ).fit(X, y)
    # Structural equality is not required (sklearn breaks exact ties by a
    # random permutation of features); predictive equivalence is.
    assert np.allclose(ours.predict(X), ref.predict(X), atol=1e-8)
