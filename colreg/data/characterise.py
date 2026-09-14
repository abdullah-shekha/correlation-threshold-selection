"""Collinearity characterisation -- this produces Table 1 of the paper.

Every result in the study is reported conditional on these numbers.  Averaging
performance across a corpus without stratifying by collinearity is precisely the
mistake that makes existing comparisons uninformative: the effect of pruning has
opposite signs in different regimes, so the grand mean is close to zero and
means nothing.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["characterise", "assign_stratum", "CHARACTERISATION_FIELDS"]

CHARACTERISATION_FIELDS = (
    "n", "p", "n_over_p",
    "condition_number", "max_vif", "mean_vif",
    "mean_abs_r", "max_abs_r", "n_pairs_above_80",
    "effective_rank", "n_constant_columns",
    "numerical_rank", "rank_deficiency", "condition_number_trimmed",
    "condition_index", "condition_index_trimmed",
    "stratum",
)


def characterise(X, y=None) -> dict:
    """Describe the collinearity structure of a design matrix.

    Returns
    -------
    dict with:
        condition_number
            kappa of the *correlation* matrix (equivalently, of the standardised
            design).  Scale-free, unlike kappa(X'X) on raw columns, so it is
            comparable across datasets in different units.  This is the variable
            the strata are cut on.
        max_vif, mean_vif
            Variance inflation factors from the diagonal of the inverse
            correlation matrix.  Catches multi-way collinearity that no pairwise
            correlation reveals -- the reason the VIF arm is in the design.
        max_abs_r, n_pairs_above_80
            What the pruning rule actually sees.  A dataset can have max_abs_r
            below 0.8 (so threshold pruning does nothing at all) while max_vif
            exceeds 30.  Those datasets are the sharpest test of RQ1, because
            there the pruning arm is provably identical to the control and any
            apparent difference is noise.
        effective_rank
            exp(entropy of the normalised eigenvalue spectrum).  A continuous
            stand-in for 'how many genuinely independent directions are here'.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape

    sd = X.std(axis=0)
    degenerate = sd == 0                      # constant columns
    sd = np.where(degenerate, 1.0, sd)
    Z = (X - X.mean(axis=0)) / sd

    # Correlation computed directly from the standardised matrix rather than via
    # np.corrcoef, which divides by its own standard deviation and therefore
    # raises "invalid value encountered in divide" (and returns NaN) for any
    # constant column. Here a constant column simply gets zero correlation with
    # everything and a 1.0 on the diagonal, so kappa stays finite and the
    # dataset is not mis-assigned to the high-collinearity stratum.
    # Divide by n, not n-1: sd above is the population standard deviation
    # (np.std defaults to ddof=0), so sum(z^2) == n exactly and Z'Z/n has a unit
    # diagonal and off-diagonals equal to Pearson r to machine precision.
    # Using n-1 against a ddof=0 scaling inflates every off-diagonal by
    # n/(n-1) -- small, but the smallest eigenvalue of an ill-conditioned
    # correlation matrix is extremely sensitive to it: on the diabetes data it
    # moved kappa from 470.1 to 638.6, a 36% error.
    corr = (Z.T @ Z) / n
    corr = np.atleast_2d(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    n_degenerate = int(degenerate.sum())

    eig = np.linalg.eigvalsh(corr)
    eig = np.clip(eig, 0.0, None)
    kappa = float(eig.max() / eig.min()) if eig.min() > 1e-12 else float("inf")

    # Rank diagnostics.  Several real datasets are EXACTLY rank-deficient: a
    # feature is an exact linear combination of others (Energy Efficiency's
    # building geometry, Facebook's "total interactions" against its
    # components).  There kappa is genuinely infinite -- correct, but useless as
    # a continuous stratifying variable, and it collapses every such dataset
    # into one bucket.  So also report the condition number of the well-
    # determined part of the spectrum, which stays finite and lets the paper
    # regress the effect on log(kappa) instead of relying on 3 coarse strata.
    tol = eig.max() * max(n, p) * np.finfo(float).eps
    keep = eig[eig > tol]
    numerical_rank = int(keep.size)
    kappa_trimmed = float(keep.max() / keep.min()) if keep.size else float("inf")

    # The condition INDEX, which is what Belsley/Kuh/Welsch's 30 and 100 refer
    # to: the ratio of SINGULAR values of the scaled design matrix. Eigenvalues
    # of a cross-product are squared singular values, so
    #     condition_index = sqrt(condition_number of the correlation matrix).
    # Comparing kappa itself against 30/100 applies the thresholds to the square
    # and flags an index of 5.5 as "moderate" and 10 as "severe" -- far too
    # aggressive. On this corpus that error moved 13 of 25 datasets into the
    # wrong stratum and made the split look like 15 high / 5 / 5 when it is
    # really 7 high / 4 medium / 14 low.
    cond_index = math.sqrt(kappa) if math.isfinite(kappa) else float("inf")
    cond_index_trimmed = (
        math.sqrt(kappa_trimmed) if math.isfinite(kappa_trimmed) else float("inf")
    )

    vifs = np.diag(np.linalg.pinv(corr))
    off = corr[~np.eye(p, dtype=bool)] if p > 1 else np.array([0.0])

    spectrum = eig / eig.sum() if eig.sum() > 0 else eig
    nz = spectrum[spectrum > 0]
    eff_rank = float(np.exp(-np.sum(nz * np.log(nz)))) if nz.size else 1.0

    out = {
        "n": int(n),
        "p": int(p),
        "n_over_p": float(n / p),
        "condition_number": kappa,
        "max_vif": float(np.max(vifs)),
        "mean_vif": float(np.mean(vifs)),
        "mean_abs_r": float(np.mean(np.abs(off))),
        "max_abs_r": float(np.max(np.abs(off))),
        "n_pairs_above_80": int(np.sum(np.abs(np.triu(corr, k=1)) > 0.8)),
        "effective_rank": eff_rank,
        "n_constant_columns": n_degenerate,
        "numerical_rank": numerical_rank,
        "rank_deficiency": int(p - numerical_rank),
        "condition_number_trimmed": kappa_trimmed,
        "condition_index": cond_index,
        "condition_index_trimmed": cond_index_trimmed,
    }
    out["stratum"] = assign_stratum(out["condition_index"])
    return out


def assign_stratum(condition_index: float) -> str:
    """Cut points from Belsley, Kuh & Welsch (1980), still the standard.

    Takes the condition INDEX (a singular-value ratio), not the condition
    number of the correlation matrix -- the two differ by a square, and BKW
    state their thresholds for the index. Below 30 is unproblematic, 30-100
    moderate, above 100 severe.

    Published cut points rather than tertiles of this corpus: tertiles would make
    the strata depend on which datasets happen to be included, and a
    reviewer will say so.
    """
    if condition_index < 30:
        return "low"
    if condition_index < 100:
        return "medium"
    return "high"
