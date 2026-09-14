"""Correlation-threshold feature pruning, made into an object of study.

This module is the centre of the paper.  Everything else -- the estimators, the
harness, the datasets -- exists so that the parameters exposed here can be varied
and measured.  The three parameters below are not implementation details; each
one corresponds to a research question:

    tau         -> RQ1/RQ2  is the folklore 0.8 a good threshold, and for whom
    tie_break   -> RQ5      the idiom breaks ties by column position
    fit_scope   -> RQ4      where the correlation matrix is computed from

The literature and the tutorials both treat all three as fixed: tau = 0.8,
tie_break = position (implicitly, by writing ``for j in range(i)``), fit_scope =
full dataset (implicitly, by running ``df.corr()`` before splitting).  Turning
three silent constants into three measured factors is the contribution.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CorrelationThresholdSelector", "VIFSelector"]

TIE_BREAKS = ("position", "position_reversed", "target_corr", "mean_corr", "variance", "random")
FIT_SCOPES = ("train", "full", "with_target")


class CorrelationThresholdSelector:
    """Drop one feature from every pair whose |r| exceeds ``tau``.

    Parameters
    ----------
    tau : float
        Absolute Pearson correlation above which a pair is considered redundant.
        The near-universal default in applied work is 0.8, which appears to have
        no derivation behind it, so it is swept here.

    tie_break : str
        Which member of a redundant pair to discard.

        ``position``
            Drop the later column.  This is the idiom in the wild::

                for i in range(len(cols)):
                    for j in range(i):
                        if abs(corr[i, j]) > tau:
                            drop.add(cols[i])

            It has no criterion beyond position in the dataframe, which means
            the fitted model depends on the column order of the input file.
            RQ5 measures how much.

        ``position_reversed``
            Drop the earlier column.  Included as the cheapest possible probe of
            order sensitivity: any difference between this and ``position`` on
            the same data is pure artefact.

        ``target_corr``
            Keep whichever member correlates more strongly with y.  Requires y,
            and therefore must be fitted inside the training fold -- a selector
            that looks at the target is a model component, not preprocessing.

        ``mean_corr``
            Drop whichever member has the higher mean |r| against all remaining
            features, i.e. the more redundant one overall.  Target-free.

        ``variance``
            Keep the higher-variance member.  Target-free, scale-dependent, and
            included because it is what several popular tutorials do.

        ``random``
            Uniform choice, seeded.  The null model for tie-breaking.

    fit_scope : str
        ``train``
            Correlations computed from the training fold only.  Correct.
        ``full``
            Correlations computed from the whole dataset before splitting.  This
            is the leak found in a large fraction of published pipelines, and
            it is enabled here deliberately as a study arm.
        ``with_target``
            The target column is left in the correlation matrix, so the target
            itself can be selected for removal.  Pipelines that call
            ``df.corr()`` on the full frame do this without noticing.

    Notes
    -----
    Greedy pairwise removal is order-dependent by construction: removing a
    feature changes which later pairs still both survive.  ``selection_order_``
    records the sequence so that this can be reported rather than hidden.
    """

    def __init__(
        self,
        tau: float = 0.8,
        tie_break: str = "position",
        fit_scope: str = "train",
        random_state: int | None = None,
    ):
        if tie_break not in TIE_BREAKS:
            raise ValueError(f"tie_break must be one of {TIE_BREAKS}")
        if fit_scope not in FIT_SCOPES:
            raise ValueError(f"fit_scope must be one of {FIT_SCOPES}")
        self.tau = tau
        self.tie_break = tie_break
        self.fit_scope = fit_scope
        self.random_state = random_state

    # ------------------------------------------------------------------ fit

    def fit(self, X, y=None, X_full=None, y_full=None):
        """Fit the selector.

        ``X_full`` / ``y_full`` are only consulted when ``fit_scope != 'train'``.
        Passing them explicitly rather than stashing the full dataset on the
        object keeps the leak visible in the call site, so the harness code shows
        exactly where information crosses the fold boundary.
        """
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        if self.fit_scope == "train":
            source, target = X, (None if y is None else np.asarray(y, float))
        elif self.fit_scope == "full":
            if X_full is None:
                raise ValueError("fit_scope='full' requires X_full")
            source = np.asarray(X_full, dtype=np.float64)
            target = None if y_full is None else np.asarray(y_full, float)
        else:  # with_target -- append y as if it were a feature, as the notebook did
            if X_full is None or y_full is None:
                raise ValueError("fit_scope='with_target' requires X_full and y_full")
            source = np.column_stack(
                [np.asarray(X_full, float), np.asarray(y_full, float)]
            )
            target = None

        p = X.shape[1]

        # Correlation computed from the standardised matrix rather than via
        # np.corrcoef. Inside a CV fold a column can become constant (a one-hot
        # dummy that is all-zero in this training split), and corrcoef then
        # divides by a zero standard deviation: it emits
        # "invalid value encountered in divide" and returns NaN. nan_to_num
        # turns those into 0, INCLUDING the diagonal -- which is wrong, because
        # mean_abs_r below subtracts a diagonal of 1.0 that is no longer there
        # and hands the constant column a negative mean |r|. That biases the
        # 'mean_corr' tie-break (caret's rule) toward keeping dead columns.
        # Pruning decisions read only off-diagonals, so 'position' was never
        # affected; 'mean_corr' was.
        sd = source.std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        Z = (source - source.mean(axis=0)) / sd
        corr = (Z.T @ Z) / max(source.shape[0], 1)
        corr = np.atleast_2d(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)

        target_r = None
        if self.tie_break == "target_corr":
            if target is None:
                raise ValueError("tie_break='target_corr' requires y")
            target_r = np.abs(
                [np.corrcoef(source[:, j], target)[0, 1] for j in range(source.shape[1])]
            )
            target_r = np.nan_to_num(target_r, nan=0.0)

        variances = source.var(axis=0)
        mean_abs_r = (np.abs(corr).sum(axis=1) - 1.0) / max(corr.shape[0] - 1, 1)

        alive = np.ones(source.shape[1], dtype=bool)
        order: list[tuple[int, int, float]] = []

        # Visit pairs in the same nested order as the idiom being studied, so
        # that tie_break='position' reproduces it exactly.
        for i in range(source.shape[1]):
            for j in range(i):
                if not (alive[i] and alive[j]):
                    continue
                r = abs(corr[i, j])
                if r <= self.tau:
                    continue
                victim = self._choose_victim(i, j, target_r, mean_abs_r, variances, rng)
                alive[victim] = False
                order.append((i, j, r))

        # The appended target column, if any, is never a real feature.
        self.support_ = alive[:p].copy()
        self.target_was_selected_for_removal_ = bool(
            self.fit_scope == "with_target" and not alive[-1]
        )
        self.selection_order_ = order
        self.n_features_in_ = p
        self.n_features_out_ = int(self.support_.sum())
        return self

    def _choose_victim(self, i, j, target_r, mean_abs_r, variances, rng):
        tb = self.tie_break
        if tb == "position":
            return i
        if tb == "position_reversed":
            return j
        if tb == "target_corr":
            return i if target_r[i] < target_r[j] else j
        if tb == "mean_corr":
            return i if mean_abs_r[i] > mean_abs_r[j] else j
        if tb == "variance":
            return i if variances[i] < variances[j] else j
        return int(rng.choice([i, j]))

    # -------------------------------------------------------------- transform

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if not self.support_.any():
            # Degenerate but reachable at high tau on very collinear data.
            # Return a single column so downstream estimators still run; the
            # harness records n_features_out_ = 0 for the results table.
            return np.zeros((X.shape[0], 1))
        return X[:, self.support_]

    def fit_transform(self, X, y=None, **kw):
        return self.fit(X, y, **kw).transform(X)


class VIFSelector:
    """Backward elimination on the variance inflation factor.

    Repeatedly drops the feature with the largest VIF until every remaining VIF
    falls below ``threshold``.  Included as the principled comparator: unlike
    pairwise correlation, VIF accounts for multi-way collinearity, so it catches
    a feature that is a linear combination of three others while every pairwise
    correlation stays modest.

    Standard thresholds are 5 and 10, and are just as inherited as tau = 0.8.
    """

    def __init__(self, threshold: float = 10.0, max_drop: int | None = None):
        self.threshold = threshold
        self.max_drop = max_drop

    @staticmethod
    def _vif(X):
        """VIF_j = 1 / (1 - R^2_j) from regressing feature j on the others.

        Computed from the inverse correlation matrix, whose diagonal is exactly
        that quantity -- one pseudo-inverse instead of p regressions.
        """
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        if corr.ndim == 0:
            return np.array([1.0])
        return np.diag(np.linalg.pinv(corr))

    def fit(self, X, y=None, **kw):
        X = np.asarray(X, dtype=np.float64)
        p = X.shape[1]
        alive = np.ones(p, dtype=bool)
        dropped = 0
        while alive.sum() > 1:
            if self.max_drop is not None and dropped >= self.max_drop:
                break
            vifs = self._vif(X[:, alive])
            worst = int(np.argmax(vifs))
            if vifs[worst] <= self.threshold:
                break
            alive[np.flatnonzero(alive)[worst]] = False
            dropped += 1
        self.support_ = alive
        self.n_features_in_ = p
        self.n_features_out_ = int(alive.sum())
        self.selection_order_ = []
        self.target_was_selected_for_removal_ = False
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X[:, self.support_] if self.support_.any() else np.zeros((X.shape[0], 1))

    def fit_transform(self, X, y=None, **kw):
        return self.fit(X, y, **kw).transform(X)
