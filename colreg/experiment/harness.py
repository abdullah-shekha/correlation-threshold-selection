"""Nested repeated cross-validation harness.

Design decisions that a reviewer will check, made explicit:

*   **Nested, not flat.**  The inner loop tunes alpha (and tau, where the arm
    calls for it); the outer loop estimates generalisation.  Tuning and
    evaluating on the same folds is the standard way benchmark papers overstate
    their winner, and it favours whichever arm has more knobs -- which here is
    the pruning arm.  Nesting removes that bias.

*   **Scaling inside the fold.**  Standardisation is fitted on the training fold
    only.  L1 and L2 penalties are not scale-invariant, so an unscaled design
    makes a single alpha mean something different for each feature; and fitting
    the scaler on all rows leaks the test fold's location and spread.

*   **Selection inside the fold.**  Same argument, and it is the object of RQ4:
    the `fit_scope` parameter deliberately breaks this rule in the leakage arms
    so the cost can be measured rather than assumed.

*   **One results row per (dataset, arm, estimator, repeat, fold).**  Long
    format, written incrementally, resumable.  Aggregation happens in analysis,
    never during the run, so results can be re-aggregated without re-running the
    grid.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass

import numpy as np

from ..analysis.metrics import (
    mae,
    r2_score,
    r2_squared_correlation,
    rmse,
)
from .arms import PARAM_GRIDS, TAU_GRID

__all__ = ["RunResult", "evaluate_arm", "kfold_indices"]


@dataclass
class RunResult:
    dataset: str
    selection: str
    estimator: str
    tie_break: str
    fit_scope: str
    repeat: int
    fold: int
    perm: int
    n_features_in: int
    n_features_out: int
    support: str          # hex bitmask of surviving columns, LSB = column 0
    best_params: str
    rmse: float
    mae: float
    r2: float
    r2_sqcorr: float          # the wrong metric, recorded for the D3 note
    target_dropped: bool
    fit_seconds: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def kfold_indices(n: int, k: int, seed: int):
    """Plain shuffled k-fold.

    Deliberately not stratified: the target is continuous.  Some benchmark
    papers stratify on binned y, which changes the sampling distribution and
    makes the Friedman test's independence assumption harder to defend.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [
        (np.setdiff1d(idx, part, assume_unique=False), part)
        for part in np.array_split(idx, k)
    ]


def _encode_support(mask, p: int) -> str:
    """Pack a boolean support mask into a hex string.

    One short field per row instead of p booleans, and directly comparable as a
    dict key when counting distinct selected subsets across column orders.
    PCA has no column-wise support, so it encodes as an empty string.
    """
    if mask is None:
        return ""
    bits = 0
    for j, keep in enumerate(np.asarray(mask, dtype=bool)[:p]):
        if keep:
            bits |= (1 << j)
    return format(bits, "x")


def _standardise(X_tr, X_te):
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X_tr - mu) / sd, (X_te - mu) / sd


def _param_combos(grid: dict):
    if not grid:
        yield {}
        return
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def evaluate_arm(
    X,
    y,
    dataset_key: str,
    selection_arm,
    estimator_arm,
    *,
    tie_break: str = "position",
    fit_scope: str = "train",
    n_repeats: int = 10,
    n_outer: int = 5,
    n_inner: int = 5,
    perm: int = 0,
    seed: int = 0,
):
    """Run one cell of the grid and yield one RunResult per outer fold.

    The full study calls this once per (dataset, selection, estimator) triple
    for the main grid, then again with ``tie_break`` varied for RQ5 and
    ``fit_scope`` varied for RQ4.
    """
    import time

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = X.shape[0]
    grid = PARAM_GRIDS[estimator_arm.key]
    tune_tau = selection_arm.params.get("tau", "absent") is None

    for rep in range(n_repeats):
        for f, (tr, te) in enumerate(kfold_indices(n, n_outer, seed=seed + rep)):
            X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]

            # ---------------- inner loop: choose params (and tau) ------------
            taus = TAU_GRID if tune_tau else [selection_arm.params.get("tau")]
            best, best_err = None, np.inf
            for tau in taus:
                for params in _param_combos(grid):
                    errs = []
                    for tr2, va in kfold_indices(len(tr), n_inner, seed=seed + rep * 97 + f):
                        e = _fit_score(
                            X_tr[tr2], y_tr[tr2], X_tr[va], y_tr[va],
                            selection_arm, estimator_arm, params, tau,
                            tie_break, "train",          # never leak in the inner loop
                            X_full=X, y_full=y,
                        )
                        errs.append(e[0])
                    m = float(np.mean(errs))
                    if m < best_err:
                        best_err, best = m, (params, tau)

            params, tau = best
            t0 = time.perf_counter()
            err, extra = _fit_score(
                X_tr, y_tr, X_te, y_te,
                selection_arm, estimator_arm, params, tau,
                tie_break, fit_scope,
                X_full=X, y_full=y, want_extra=True,
            )
            dt = time.perf_counter() - t0

            yield RunResult(
                dataset=dataset_key,
                selection=selection_arm.key,
                estimator=estimator_arm.key,
                tie_break=tie_break,
                fit_scope=fit_scope,
                repeat=rep,
                fold=f,
                perm=perm,
                n_features_in=X.shape[1],
                n_features_out=extra["n_out"],
                support=extra["support"],
                best_params=json.dumps({**params, "tau": tau}, default=str),
                rmse=err,
                mae=extra["mae"],
                r2=extra["r2"],
                r2_sqcorr=extra["r2_sqcorr"],
                target_dropped=extra["target_dropped"],
                fit_seconds=dt,
            )


def _fit_score(
    X_tr, y_tr, X_te, y_te,
    selection_arm, estimator_arm, params, tau,
    tie_break, fit_scope,
    *, X_full=None, y_full=None, want_extra=False,
):
    """Fit selector then estimator on the training data; score on held-out.

    Order matters and is fixed here: standardise -> select -> standardise again.
    The second pass matters because dropping columns changes nothing about the
    remaining ones, but PCA output is on a different scale entirely, and the
    penalty must see comparable columns whichever arm produced them.
    """
    Xs_tr, Xs_te = _standardise(X_tr, X_te)

    kw = {}
    if selection_arm.key.startswith("corr"):
        kw = {"tau": tau, "tie_break": tie_break, "fit_scope": fit_scope,
              "random_state": 0}
    sel = selection_arm.build(**kw)

    scope_kw = {}
    if fit_scope != "train":
        # The leak, made visible at the call site rather than hidden in state.
        mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        scope_kw = {"X_full": (X_full - mu) / sd, "y_full": y_full}

    sel.fit(Xs_tr, y_tr, **scope_kw)
    Xt_tr, Xt_te = sel.transform(Xs_tr), sel.transform(Xs_te)
    Xt_tr, Xt_te = _standardise(Xt_tr, Xt_te)

    model = estimator_arm.build(**params).fit(Xt_tr, y_tr)
    pred = model.predict(Xt_te)

    err = rmse(y_te, pred)
    if not want_extra:
        return err, None
    return err, {
        "mae": mae(y_te, pred),
        "r2": r2_score(y_te, pred),
        "r2_sqcorr": r2_squared_correlation(y_te, pred),
        "n_out": int(getattr(sel, "n_features_out_", Xt_tr.shape[1])),
        "support": _encode_support(getattr(sel, "support_", None), X_tr.shape[1]),
        "target_dropped": bool(getattr(sel, "target_was_selected_for_removal_", False)),
    }
