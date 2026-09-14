"""The experimental grid.

Factor A (selection) x Factor B (estimator) x Factor C (dataset), with two
sub-studies bolted on: the leakage arms for RQ4 and the permutation arm for RQ5.

Keeping the grid as data rather than as nested loops in a script means the
manuscript's design table and the code that ran cannot drift apart -- the table
is generated from this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..estimators import (
    ElasticNetRegression,
    LassoRegression,
    LinearRegression,
    RegressionTree,
    RidgeRegression,
)
from ..selection import (
    CorrelationThresholdSelector,
    PassthroughSelector,
    PCASelector,
    VIFSelector,
)

__all__ = ["SELECTION_ARMS", "ESTIMATORS", "PARAM_GRIDS", "Arm", "main_grid"]


@dataclass(frozen=True)
class Arm:
    key: str
    factory: Any
    params: dict = field(default_factory=dict)
    rq: str = ""

    def build(self, **override):
        return self.factory(**{**self.params, **override})


# --------------------------------------------------------------- Factor A
# tau is tuned in the inner CV loop for the 'corr_tuned' arm and held fixed at
# the folklore value for 'corr_080'.  Both are needed: the fixed arm is what
# practitioners actually do, the tuned arm is the fair comparison.

SELECTION_ARMS: tuple[Arm, ...] = (
    Arm("none", PassthroughSelector, {}, "control"),
    Arm("corr_070", CorrelationThresholdSelector, {"tau": 0.70}, "RQ1"),
    Arm("corr_080", CorrelationThresholdSelector, {"tau": 0.80}, "RQ1"),
    Arm("corr_090", CorrelationThresholdSelector, {"tau": 0.90}, "RQ1"),
    Arm("corr_095", CorrelationThresholdSelector, {"tau": 0.95}, "RQ1"),
    Arm("corr_tuned", CorrelationThresholdSelector, {"tau": None}, "RQ1"),
    Arm("vif_05", VIFSelector, {"threshold": 5.0}, "RQ1"),
    Arm("vif_10", VIFSelector, {"threshold": 10.0}, "RQ1"),
    Arm("pca_95", PCASelector, {"var_target": 0.95}, "RQ1"),
)

# Sub-study arms, run only on the datasets that reach the relevant regime.
TIE_BREAK_ARMS: tuple[str, ...] = (
    "position", "position_reversed", "target_corr", "mean_corr", "variance", "random",
)
LEAK_ARMS: tuple[str, ...] = ("train", "full", "with_target")


# --------------------------------------------------------------- Factor B

ESTIMATORS: tuple[Arm, ...] = (
    Arm("ols", LinearRegression, {}, "all"),
    Arm("ridge", RidgeRegression, {}, "all"),
    Arm("lasso", LassoRegression, {}, "all"),
    Arm("enet", ElasticNetRegression, {}, "all"),
    Arm("cart", RegressionTree, {}, "all"),
)

# Inner-loop tuning grids.  Log-spaced over six decades because the optimal
# alpha moves with n and with the collinearity level, and a grid that bottoms
# out at the edge silently reports a boundary solution.
import numpy as _np

PARAM_GRIDS: dict[str, dict] = {
    "ols": {},
    "ridge": {"alpha": _np.logspace(-4, 4, 25)},
    "lasso": {"alpha": _np.logspace(-4, 2, 25)},
    "enet": {"alpha": _np.logspace(-4, 2, 15), "l1_ratio": [0.1, 0.5, 0.9]},
    "cart": {"max_depth": [2, 3, 4, 6, 8, None], "min_samples_leaf": [1, 5, 20]},
}

TAU_GRID = [0.60, 0.70, 0.80, 0.90, 0.95, 0.99]


def main_grid(datasets):
    """Yield (dataset, selection_arm, estimator_arm) for the primary study."""
    for ds in datasets:
        for sel in SELECTION_ARMS:
            for est in ESTIMATORS:
                # PCA destroys the coefficient-to-feature mapping, so the
                # stability measures in RQ3 are undefined for it; the harness
                # records NaN rather than silently comparing incomparables.
                yield ds, sel, est
