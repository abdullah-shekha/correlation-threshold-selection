"""Is the null control actually order-invariant? Check before publishing it.

    python scripts/audit_control.py

The findings document asserts that the order-invariant control produced "a
spread of exactly zero and a single distinct subset in all 18 cells". The
permutation stability measure contradicts that: four control cells score below
1.000, meaning the control chose different feature subsets under different
column orders.

Both cannot be true, and the control is load-bearing. It is what rules out the
alternative explanation that permuting columns perturbs something else in the
pipeline -- a seed, a fold, an accumulation order -- rather than the selection
rule specifically. If the control is not exactly flat, the RQ5 claim needs
restating, and it is far better to restate it now than to have a reviewer
notice that a "0.000%" figure sits next to a stability index of 0.797.

Three candidate explanations, which this script distinguishes:

  1.  EXACT TIES. 'mean_corr' drops whichever of a tied pair has the larger
      mean absolute correlation. If two features have *identical* mean |r| the
      rule cannot decide, and whatever tie-break sits underneath it -- position
      -- resolves the tie. The rule is then order-invariant everywhere except
      on exact ties, which is a caveat, not a failure.

  2.  FLOATING-POINT TIES. Mean |r| is a sum over columns. Permuting the
      columns changes the summation order, so two mathematically equal values
      can differ in their last bits, and a near-tie flips. This would make the
      control invariant in exact arithmetic but not in IEEE 754 -- a genuinely
      interesting finding, and one that strengthens rather than weakens the
      paper: even the principled rule is not reproducible at the bit level.

  3.  SOMETHING ELSE IS PERTURBED. If the differing subsets are not explained
      by ties at all, then column order is reaching the pipeline by some route
      other than the selection rule, and the paired design is not as clean as
      claimed. This is the one that would hurt, and it is why the script
      reports the minimum gap rather than assuming.

The script reports, for every control cell, how many distinct subsets appeared,
which columns differed between them, and the smallest gap in mean |r| among the
features involved. A gap at the 1e-15 level is explanation 2; an exact 0.0 is
explanation 1; a gap of ordinary size is explanation 3.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "results" / "shards"
CACHE = ROOT / "data" / "cache"

CONTROL_RULE = "mean_corr"
TAU = 0.8


def bits(hex_support: str, p: int) -> np.ndarray:
    v = int(hex_support, 16)
    return np.array([(v >> j) & 1 for j in range(p)], dtype=int)


def corr_matrix(X):
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (X - X.mean(axis=0)) / sd
    C = np.nan_to_num((Z.T @ Z) / X.shape[0], nan=0.0)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)


def tie_gap(ds):
    """Smallest gap in mean |r| between two features sharing an above-tau pair."""
    path = CACHE / f"{ds}.npz"
    if not path.exists():
        return None, None
    X = np.load(path)["X"]
    C = np.abs(corr_matrix(X))
    mean_abs = (C.sum(axis=1) - 1.0) / max(C.shape[0] - 1, 1)
    gaps = []
    p = C.shape[0]
    for i in range(p):
        for j in range(i):
            if C[i, j] > TAU:
                gaps.append(abs(mean_abs[i] - mean_abs[j]))
    if not gaps:
        return None, 0
    return float(min(gaps)), len(gaps)


def main() -> int:
    cells = defaultdict(lambda: defaultdict(list))
    for f in sorted(SHARDS.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("perm", 0) <= 0 or not r.get("support"):
                    continue
                cells[(r["dataset"], r["estimator"], r["tie_break"])][r["perm"]].append(
                    (r["support"], r["n_features_in"], r["rmse"]))
    if not cells:
        print("no permutation results found")
        return 1

    print("=" * 78)
    print(f"CONTROL AUDIT  (rule = {CONTROL_RULE})")
    print("=" * 78)
    print(f"  {'dataset':18s}{'perms':>6s}{'subsets':>9s}{'spread%':>9s}"
          f"{'min tie gap':>14s}   verdict")
    print("  " + "-" * 74)

    n_flat, offenders = 0, []
    for (ds, est, tb), perms in sorted(cells.items()):
        if tb != CONTROL_RULE:
            continue
        modal, means = [], []
        for entries in perms.values():
            counts = defaultdict(int)
            for sup_hex, p_in, _ in entries:
                counts[(sup_hex, p_in)] += 1
            modal.append(max(counts, key=counts.get))
            means.append(st.fmean(x[2] for x in entries))
        distinct = len(set(modal))
        mu = st.fmean(means)
        spread = 100.0 * (max(means) - min(means)) / mu if mu else float("nan")
        gap, _ = tie_gap(ds)

        if distinct == 1:
            verdict, n_flat = "invariant", n_flat + 1
        elif gap is None:
            verdict = "NOT invariant (no cache to check)"
            offenders.append((ds, est, modal, gap))
        elif gap == 0.0:
            verdict = "exact tie -- explanation 1"
            offenders.append((ds, est, modal, gap))
        elif gap < 1e-12:
            verdict = "float tie -- explanation 2"
            offenders.append((ds, est, modal, gap))
        else:
            verdict = "UNEXPLAINED -- explanation 3"
            offenders.append((ds, est, modal, gap))

        gs = "--" if gap is None else f"{gap:.3e}"
        print(f"  {ds:18s}{len(perms):>6d}{distinct:>9d}{spread:>9.3f}"
              f"{gs:>14s}   {verdict}")

    print(f"\n  {n_flat} cells perfectly invariant; {len(offenders)} not.")

    if offenders:
        print("\n" + "=" * 78)
        print("WHICH COLUMNS DIFFER")
        print("=" * 78)
        for ds, est, modal, gap in offenders:
            uniq = sorted(set(modal))
            p = uniq[0][1]
            vecs = [bits(h, pp) for h, pp in uniq]
            union = np.any(vecs, axis=0).astype(int)
            inter = np.all(vecs, axis=0).astype(int)
            unstable = np.where(union != inter)[0]
            print(f"  {ds} / {est}: {len(uniq)} subsets over {p} columns; "
                  f"columns that come and go: {list(unstable)}")
            print(f"    sizes {[int(v.sum()) for v in vecs]}, "
                  f"min mean-|r| gap {'--' if gap is None else f'{gap:.3e}'}")

    print("\n" + "=" * 78)
    print("WHAT TO WRITE")
    print("=" * 78)
    if not offenders:
        print("  The control is exactly invariant. The document's claim stands.")
    elif all(g is not None and g < 1e-12 for _, _, _, g in offenders):
        print("  The control is invariant except where mean |r| ties to within")
        print("  floating-point precision. Restate the claim as: the control")
        print("  produced a single subset in "
              f"{n_flat} of {n_flat + len(offenders)} cells, and in the")
        print("  remainder the rule met an exact or near-exact tie that it")
        print("  cannot resolve. That is a caveat about ties, not a failure of")
        print("  the design, and it is worth a sentence because it shows the")
        print("  positional fallback is present even inside the 'principled' rule.")
    else:
        print("  At least one cell is NOT explained by ties. Do not publish the")
        print("  0.000% control figure until this is understood: column order is")
        print("  reaching the result by a route other than the selection rule,")
        print("  and the paired design cannot be described as clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
