"""Metrics, stated as formulas so that the numbers are comparable.

The single most common source of non-comparable results in this literature is
R^2 reported without saying which R^2.  ``r2_score`` below is the residual-sum
definition; ``r2_squared_correlation`` is the definition the original notebooks
used, kept here *only* so the paper can quantify the gap between them.  Showing
that gap on real data is a small but genuinely useful methodological note.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rmse",
    "mae",
    "r2_score",
    "r2_squared_correlation",
    "nogueira_stability",
    "sign_consistency",
]


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination, 1 - SS_res / SS_tot.

    Can be negative; a model worse than the training mean *should* score below
    zero, and suppressing that is how bad models get reported as adequate.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def r2_squared_correlation(y_true, y_pred) -> float:
    """Squared Pearson correlation -- the WRONG metric, retained for measurement.

    Invariant to any affine rescaling of the predictions: a model that is wrong
    by a constant factor of three still scores 1.0.  Equals ``r2_score`` only for
    an unbiased least-squares fit evaluated in sample.  The paper reports the
    distribution of ``r2_squared_correlation - r2_score`` across all runs to put
    a number on how much optimism this substitution buys.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1] ** 2)


def nogueira_stability(supports) -> float:
    """Selection stability index of Nogueira, Sechidis & Brown (JMLR 2018).

        Phi = 1 - (1/p) sum_j s_j^2 / ( (k/p)(1 - k/p) )

    where s_j^2 is the unbiased sample variance of the indicator for feature j
    across the M selection sets and k is the mean number selected.  Fully
    corrected for chance: 1.0 means every resample picked the same features,
    0.0 means no better than random subsets of the same average size.

    This is the measure for RQ3.  It is the reason the study can produce a
    finding even if every accuracy comparison comes out null -- 'pruning does not
    improve accuracy but does improve stability by X' is a publishable result,
    and so is its negation.
    """
    Z = np.asarray(supports, dtype=float)
    M, p = Z.shape
    if M < 2:
        return float("nan")
    p_hat = Z.mean(axis=0)
    s2 = (M / (M - 1.0)) * p_hat * (1.0 - p_hat)
    k_bar = Z.sum(axis=1).mean()
    denom = (k_bar / p) * (1.0 - k_bar / p)
    if denom == 0:
        return 1.0
    return float(1.0 - s2.mean() / denom)


def sign_consistency(coef_matrix) -> float:
    """Fraction of coefficients holding one sign across resamples.

    For each feature, the share of the majority sign among the runs in which it
    was non-zero, averaged over features.  1.0 means no feature ever flipped.
    A model whose coefficient signs flip between folds tells a different causal
    story every time it is refitted, which is the practical harm of collinearity
    that accuracy metrics never surface.
    """
    C = np.asarray(coef_matrix, dtype=float)
    if C.ndim != 2 or C.shape[0] < 2:
        return float("nan")
    scores = []
    for j in range(C.shape[1]):
        col = C[:, j]
        nz = col[col != 0]
        if nz.size == 0:
            continue
        pos = np.sum(nz > 0)
        scores.append(max(pos, nz.size - pos) / nz.size)
    return float(np.mean(scores)) if scores else float("nan")
