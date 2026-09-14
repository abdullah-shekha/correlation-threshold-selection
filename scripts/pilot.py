"""Pilot run: a smoke test to run before committing to the full grid.

Two datasets, a reduced tuning grid, two repeats.  Its job is not to produce a
result; it is to prove that every arm executes, that the results table has the
shape the analysis code expects, and that the RQ5 permutation machinery detects
order-dependence where order-dependence must exist.

The full study should be run only after this passes, and after
scripts/verify_corpus.py has resolved every dataset in the registry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colreg.data import characterise
from colreg.experiment import ESTIMATORS, SELECTION_ARMS, evaluate_arm
from colreg.experiment import arms as arms_mod
from colreg.selection import CorrelationThresholdSelector

# Reduced grids so the pilot finishes in minutes, not hours.
arms_mod.PARAM_GRIDS.update({
    "ols": {},
    "ridge": {"alpha": np.logspace(-3, 3, 7)},
    "lasso": {"alpha": np.logspace(-3, 1, 5)},
    "enet": {"alpha": np.logspace(-3, 1, 4), "l1_ratio": [0.5]},
    "cart": {"max_depth": [3, 5, None], "min_samples_leaf": [5]},
})
arms_mod.TAU_GRID = [0.7, 0.8, 0.9]


def synthetic_collinear(n=400, seed=0):
    """A design where the answer is known.

    x0..x3 are independent signal.  x4..x7 are near-copies of x0..x3 with tiny
    noise, so they carry no additional information.  x8, x9 are pure noise.
    Only x0..x3 have non-zero coefficients.

    Any sensible selection method should drop four of the eight correlated
    columns and leave the signal intact.  If an arm cannot do this here, it will
    not do anything useful on real data either.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 4))
    copies = base + rng.normal(scale=0.05, size=(n, 4))
    noise = rng.normal(size=(n, 2))
    X = np.column_stack([base, copies, noise])
    beta = np.array([3.0, -2.0, 1.5, 1.0, 0, 0, 0, 0, 0, 0])
    y = X @ beta + rng.normal(scale=1.0, size=n)
    return X, y


def load_diabetes_xy():
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    return np.asarray(d.data, float), np.asarray(d.target, float)


def banner(text):
    print("\n" + text)
    print("-" * len(text))


def main():
    datasets = {
        "synthetic_collinear": synthetic_collinear(),
        "diabetes": load_diabetes_xy(),
    }

    banner("Table 1 -- collinearity characterisation")
    hdr = f"{'dataset':22s} {'n':>6s} {'p':>4s} {'kappa':>9s} {'maxVIF':>8s} {'maxR':>6s} {'>0.8':>5s}  stratum"
    print(hdr)
    for key, (X, y) in datasets.items():
        c = characterise(X, y)
        print(f"{key:22s} {c['n']:6d} {c['p']:4d} {c['condition_number']:9.1f} "
              f"{c['max_vif']:8.1f} {c['max_abs_r']:6.3f} {c['n_pairs_above_80']:5d}  {c['stratum']}")

    banner("Main grid -- mean test RMSE over 2x5 nested CV")
    rows = []
    for key, (X, y) in datasets.items():
        for sel in SELECTION_ARMS:
            for est in ESTIMATORS:
                res = list(evaluate_arm(X, y, key, sel, est, n_repeats=2, n_inner=3))
                rows.extend(res)
                r = np.mean([x.rmse for x in res])
                nf = np.mean([x.n_features_out for x in res])
                print(f"{key:22s} {sel.key:12s} {est.key:6s} "
                      f"RMSE {r:8.3f}   features kept {nf:5.1f}")

    banner("D3 check -- how much optimism the wrong R-squared buys")
    delta = np.array([x.r2_sqcorr - x.r2 for x in rows if np.isfinite(x.r2)])
    print(f"squared-correlation R2 minus true R2, over {delta.size} folds:")
    print(f"  mean {delta.mean():+.4f}   median {np.median(delta):+.4f}   max {delta.max():+.4f}")

    banner("RQ5 -- order dependence of the standard pruning idiom")
    X, y = datasets["synthetic_collinear"]
    rng = np.random.default_rng(0)
    from colreg.experiment.arms import Arm
    arm = Arm("corr_080", CorrelationThresholdSelector, {"tau": 0.80}, "RQ5")
    ridge = [e for e in ESTIMATORS if e.key == "ridge"][0]

    scores, kept_sets = [], []
    for k in range(12):
        order = rng.permutation(X.shape[1])
        res = list(evaluate_arm(X[:, order], y, "perm", arm, ridge,
                                n_repeats=1, n_inner=3, perm=k))
        scores.append(np.mean([r.rmse for r in res]))
        sel = CorrelationThresholdSelector(tau=0.8, tie_break="position").fit(X[:, order])
        kept_sets.append(tuple(sorted(order[sel.support_])))

    scores = np.array(scores)
    print("12 random column orders, identical data, identical estimator:")
    print(f"  RMSE  min {scores.min():.4f}  max {scores.max():.4f}  "
          f"spread {scores.max() - scores.min():.4f}  sd {scores.std():.4f}")
    print(f"  distinct feature subsets selected: {len(set(kept_sets))} of 12")

    banner("Tie-break comparison on the same data (single split, ridge)")
    for tb in ("position", "position_reversed", "target_corr", "mean_corr", "variance"):
        res = list(evaluate_arm(X, y, "tiebreak", arm, ridge,
                                tie_break=tb, n_repeats=2, n_inner=3))
        print(f"  {tb:18s} RMSE {np.mean([r.rmse for r in res]):8.4f}  "
              f"kept {np.mean([r.n_features_out for r in res]):4.1f}")

    banner("RQ4 -- leakage arms")
    for scope in ("train", "full"):
        res = list(evaluate_arm(X, y, "leak", arm, ridge,
                                fit_scope=scope, n_repeats=2, n_inner=3))
        print(f"  fit_scope={scope:6s} RMSE {np.mean([r.rmse for r in res]):8.4f}")

    print(f"\nTotal result rows produced: {len(rows)}")


if __name__ == "__main__":
    main()
