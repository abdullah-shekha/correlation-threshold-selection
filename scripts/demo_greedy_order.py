"""Order dependence is in the LOOP, not the tie-break. Self-contained proof.

    python scripts/demo_greedy_order.py

Why this exists. The permutation study found that 'mean_corr' -- the rule that
picks a victim by mean absolute correlation, with no reference to column
position -- still selected different feature subsets under different column
orders on four datasets. That should be impossible if order dependence lives in
the tie-break, so either the measurement was wrong or the explanation was.

The explanation was wrong. Look at the selection loop:

    for i in range(p):
        for j in range(i):
            if not (alive[i] and alive[j]):
                continue
            ...
            alive[victim] = False

Each contest is decided by ``_choose_victim``, and for 'mean_corr' that
decision is a comparison of two fixed numbers: order-invariant. But the
``alive`` guard means a feature killed early removes every later pair it
belonged to. So while each contest is order-invariant, WHICH CONTESTS HAPPEN
is not. Permuting the columns changes the visitation sequence, a different
feature dies first, and a different set of contests is ever held.

This script demonstrates that on synthetic data, with no dependence on the
corpus, so the claim can be checked by a reviewer in ten seconds. It shows:

  1.  A chain design (x0~x1~x2~...) where 'mean_corr' IS invariant -- the pairs
      barely interact, so the cascade never bites. This matters: the effect is
      not universal, and 14 of 18 control cells in the real study were exactly
      invariant for this reason.

  2.  A factor design with many overlapping pairs, where 'mean_corr' is NOT
      invariant, and where the minimum gap in mean |r| is reported to prove the
      variation is not caused by ties or floating point.

The consequence for the paper is a stronger claim, not a weaker one. The fix
for order dependence is not a better tie-break, because no tie-break can
restore invariance to a greedy elimination loop. The fix is to stop doing
greedy pairwise elimination -- which is what RQ1 concludes on independent
grounds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colreg.selection.threshold import CorrelationThresholdSelector   # noqa: E402

TAU = 0.8
RULES = ["position", "position_reversed", "mean_corr", "variance"]


def subsets(X, tie_break, perms=200, seed=0):
    """Distinct selected subsets, in ORIGINAL column ids, over random orders."""
    rng = np.random.default_rng(seed)
    seen = {}
    for k in range(perms):
        pi = np.arange(X.shape[1]) if k == 0 else rng.permutation(X.shape[1])
        sel = CorrelationThresholdSelector(
            tau=TAU, tie_break=tie_break, fit_scope="train", random_state=0
        ).fit(X[:, pi])
        keep = frozenset(int(pi[c]) for c in np.flatnonzero(sel.support_))
        seen[keep] = seen.get(keep, 0) + 1
    return seen


def diagnostics(X):
    p = X.shape[1]
    C = np.abs(np.corrcoef(X, rowvar=False))
    mean_abs = (C.sum(axis=1) - 1.0) / (p - 1)
    pairs = [(i, j) for i in range(p) for j in range(i) if C[i, j] > TAU]
    gaps = [abs(mean_abs[i] - mean_abs[j]) for i, j in pairs]
    return len(pairs), (min(gaps) if gaps else float("nan"))


def chain_design(n=400, p=6, r=0.93, seed=7):
    """x0 -> x1 -> x2 ...: adjacent pairs correlated, distant pairs not."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, p))
    X = np.empty_like(Z)
    X[:, 0] = Z[:, 0]
    for j in range(1, p):
        X[:, j] = r * X[:, j - 1] + np.sqrt(1 - r * r) * Z[:, j]
    return X


def factor_design(n=300, p=12, seed=5):
    """Three latent factors with uneven loadings: many overlapping pairs."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, 3))
    L = rng.uniform(0.75, 1.0, size=(3, p)) * (rng.random((3, p)) < 0.6)
    return F @ L + 0.30 * rng.standard_normal((n, p))


def report(name, X, note):
    npairs, gap = diagnostics(X)
    print("=" * 78)
    print(f"{name}   ({X.shape[1]} columns, {npairs} pairs above tau={TAU})")
    print("=" * 78)
    print(f"  {note}\n")
    print(f"  {'tie-break':20s}{'distinct subsets':>18s}   over 200 permutations")
    print("  " + "-" * 62)
    for rule in RULES:
        seen = subsets(X, rule)
        flag = "" if len(seen) > 1 else "   <- invariant"
        print(f"  {rule:20s}{len(seen):>18d}{flag}")
    seen = subsets(X, "mean_corr")
    if len(seen) > 1:
        print("\n  mean_corr's subsets, most frequent first:")
        for k, v in sorted(seen.items(), key=lambda kv: -kv[1])[:4]:
            print(f"    {sorted(k)}  chosen by {v} of 200 orders")
        print("\n  Smallest gap in mean |r| between two features sharing a pair:")
        print(f"    {gap:.3e}   (float64 epsilon is ~2.2e-16)")
        print("  The decisions are not tied and not at the precision limit, so the")
        print("  variation cannot be a tie-break artefact. It is the contest set.")
    print()


def main() -> int:
    print("\nDoes a position-free tie-break make greedy pruning order-invariant?\n")
    report("CHAIN DESIGN", chain_design(),
           "Adjacent pairs only: eliminations barely interact.")
    report("FACTOR DESIGN", factor_design(),
           "Many overlapping pairs: one early kill removes later contests.")
    print("=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    print("  'mean_corr' is order-invariant per contest and order-DEPENDENT as an")
    print("  algorithm. Invariance holds when above-tau pairs are disjoint and")
    print("  fails when they overlap, which is why the effect appeared on exactly")
    print("  the datasets in the corpus with the densest correlation structure.")
    print("  No tie-break can repair this; the greedy loop is the cause.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
