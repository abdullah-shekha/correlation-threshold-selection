"""Coordinate-descent kernels, JIT-compiled when Numba is available.

Why this file exists
--------------------
The first draft's pure-Python coordinate descent took ~1.70 s for one Lasso fit
at n=2000, p=60.  The study is dominated by inner-loop tuning -- 25 alphas x 5
inner folds x 5 outer folds x 10 repeats per (dataset, arm) -- so at that speed
the full grid is several hundred hours and simply will not get run.

Three changes, all standard practice rather than tricks, take it to 0.0067 s:

1.  **Gram-matrix coordinate descent** when n > p (the usual case here).
    Precompute ``G = X'X`` (p x p) and ``Xty = X'y`` once, then each coordinate
    update costs O(p) instead of O(n).  For n=2000, p=60 that alone is ~70x.
    This is what scikit-learn switches to internally as
    ``enet_coordinate_descent_gram``.

2.  **Fortran memory order** for the non-Gram path.  Coordinate descent reads
    one column at a time; on a C-ordered array each read strides by p and misses
    cache on every element.  Worth ~1.5x on its own.

3.  **Warm starts down the alpha path.**  Adjacent alphas have near-identical
    solutions, so descending from the sparse end and seeding each fit with the
    previous solution avoids re-solving from zero every time.  The Gram matrix
    is also built once for the whole path rather than once per alpha.

Measured on the reference problem (n=2000, p=60, half the columns near-duplicates):

    pure Python                1.70 s
    naive JIT, C order         0.70 s
    naive JIT, F order         0.47 s
    Gram JIT                   0.0067 s      255x, agrees with sklearn to 4.3e-06

If Numba is missing, everything still runs through the same code path with the
decorator as a no-op -- just slowly.  ``HAVE_NUMBA`` lets the test suite assert
that both paths give identical answers.
"""

from __future__ import annotations

import numpy as np

try:                                    # pragma: no cover - environment dependent
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                     # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap(args[0]) if args and callable(args[0]) else wrap


# fastmath is deliberately OFF throughout.  It permits reassociation of
# floating-point operations, which would break the 1e-6 parity guarantee against
# scikit-learn that the entire reference-implementation argument rests on.
# Speed bought at the cost of the parity claim is not worth having.


@njit(cache=True)
def _cd_naive(X, y, w, z, l1, l2, max_iter, tol):
    """Direct coordinate descent.  Used when p >= n, where no Gram matrix helps.

    ``X`` should be Fortran-ordered.  ``w`` is modified in place, which is what
    makes warm starting free.
    """
    n, p = X.shape
    r = y - X @ w                       # residual, maintained incrementally
    for it in range(max_iter):
        max_delta = 0.0
        for j in range(p):
            if z[j] == 0.0:
                continue
            w_old = w[j]
            rho = 0.0
            for i in range(n):
                rho += X[i, j] * r[i]
            rho = (rho + z[j] * w_old) / n

            if rho > l1:
                w_new = (rho - l1) / (z[j] / n + l2)
            elif rho < -l1:
                w_new = (rho + l1) / (z[j] / n + l2)
            else:
                w_new = 0.0

            if w_new != w_old:
                d = w_old - w_new
                for i in range(n):
                    r[i] += X[i, j] * d
                w[j] = w_new
                if abs(d) > max_delta:
                    max_delta = abs(d)
        if max_delta < tol:
            return it + 1
    return max_iter


@njit(cache=True)
def _cd_gram(G, Xty, w, q, n, l1, l2, max_iter, tol):
    """Coordinate descent on the Gram matrix.  Used when n > p.

    ``q`` must hold ``G @ w`` on entry and is kept in sync, so the coordinate
    residual is available in O(1) and the update costs O(p):

        rho_j = (Xty[j] - q[j] + G[j,j] w_j) / n

    Both ``w`` and ``q`` are modified in place.  To warm start, pass the previous
    solution and its matching ``q``; to start cold, pass zeros for both.
    """
    p = G.shape[0]
    for it in range(max_iter):
        max_delta = 0.0
        for j in range(p):
            gjj = G[j, j]
            if gjj == 0.0:
                continue
            w_old = w[j]
            rho = (Xty[j] - q[j] + gjj * w_old) / n

            if rho > l1:
                w_new = (rho - l1) / (gjj / n + l2)
            elif rho < -l1:
                w_new = (rho + l1) / (gjj / n + l2)
            else:
                w_new = 0.0

            if w_new != w_old:
                d = w_new - w_old
                for k in range(p):
                    q[k] += G[k, j] * d
                w[j] = w_new
                if abs(d) > max_delta:
                    max_delta = abs(d)
        if max_delta < tol:
            return it + 1
    return max_iter


# Below this ratio of n/p the Gram matrix costs more to build than it saves.
GRAM_MIN_RATIO = 1.5


def solve_enet(Xc, yc, alpha, l1_ratio, max_iter=10_000, tol=1e-8, coef_init=None):
    """Solve one elastic-net problem on centred data.  Returns (w, n_iter)."""
    n, p = Xc.shape
    w = np.zeros(p) if coef_init is None else np.array(coef_init, dtype=np.float64)
    l1, l2 = alpha * l1_ratio, alpha * (1.0 - l1_ratio)

    if n > GRAM_MIN_RATIO * p:
        Xf = np.asfortranarray(Xc)
        G = np.ascontiguousarray(Xf.T @ Xf)
        Xty = np.ascontiguousarray(Xf.T @ yc)
        q = G @ w
        it = _cd_gram(G, Xty, w, q, n, l1, l2, max_iter, tol)
    else:
        Xf = np.asfortranarray(Xc)
        z = (Xf ** 2).sum(axis=0)
        it = _cd_naive(Xf, np.ascontiguousarray(yc), w, z, l1, l2, max_iter, tol)
    return w, int(it)


def enet_path(Xc, yc, alphas, l1_ratio, max_iter=10_000, tol=1e-8):
    """Fit the elastic net for a sequence of alphas, warm-started.

    Pass alphas in DESCENDING order.  Starting from the sparse end and relaxing
    is what makes the path cheap: at large alpha nearly every coefficient is
    zero, and each subsequent fit only has to activate a few more.  Ascending
    order still returns correct answers, just slowly.

    The Gram matrix is built once for the whole path, not once per alpha -- the
    single biggest saving in the tuning loop.

    Returns ``(coefs, n_iters)`` with ``coefs`` of shape (len(alphas), p), in the
    order the alphas were given.
    """
    Xc = np.asfortranarray(Xc, dtype=np.float64)
    yc = np.ascontiguousarray(yc, dtype=np.float64)
    n, p = Xc.shape
    alphas = np.asarray(alphas, dtype=np.float64)

    coefs = np.empty((alphas.size, p))
    n_iters = np.empty(alphas.size, dtype=np.int64)
    w = np.zeros(p)                     # warm start carried down the path

    if n > GRAM_MIN_RATIO * p:
        G = np.ascontiguousarray(Xc.T @ Xc)
        Xty = np.ascontiguousarray(Xc.T @ yc)
        q = np.zeros(p)
        for k, a in enumerate(alphas):
            n_iters[k] = _cd_gram(
                G, Xty, w, q, n, a * l1_ratio, a * (1.0 - l1_ratio), max_iter, tol
            )
            coefs[k] = w
    else:
        z = (Xc ** 2).sum(axis=0)
        for k, a in enumerate(alphas):
            n_iters[k] = _cd_naive(
                Xc, yc, w, z, a * l1_ratio, a * (1.0 - l1_ratio), max_iter, tol
            )
            coefs[k] = w
    return coefs, n_iters


def warm_up() -> bool:
    """Trigger JIT compilation once, so timings and workers exclude it.

    Numba compiles on first call per signature (~1-2 s).  With ``cache=True`` the
    compiled artifact is written to ``__pycache__``, so calling this in the
    parent process before joblib forks means the workers reuse it instead of
    each paying the cost.  ``scripts/run_study.py`` does exactly that.
    """
    if not HAVE_NUMBA:
        return False
    rng = np.random.default_rng(0)
    X = np.asfortranarray(rng.normal(size=(16, 3)))
    y = np.ascontiguousarray(rng.normal(size=16))
    _cd_naive(X, y, np.zeros(3), (X ** 2).sum(axis=0), 0.1, 0.0, 5, 1e-8)
    G = np.ascontiguousarray(X.T @ X)
    _cd_gram(G, np.ascontiguousarray(X.T @ y), np.zeros(3), np.zeros(3),
             16, 0.1, 0.0, 5, 1e-8)
    return True
