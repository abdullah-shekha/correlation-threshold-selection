"""Audit the RQ5 result before it is relied upon.

    python scripts/audit_rq5.py

The permutation run produced spreads of 126% (facebook), 94% (gas_turbine) and
42% (cpu_perf). Those are far too large to be ordinary redundancy, and a
reviewer will say so. There is an obvious alternative explanation that must be
ruled out first:

    A dataset can contain a feature that is very nearly the target itself.

UCI's Computer Hardware carries ERP, the original authors' own regression
estimate of PRP. Facebook's "Total Interactions" is the arithmetic sum of its
comment/like/share columns. Gas Turbine's CDP tracks turbine energy yield almost
perfectly. When such a column exists, whether the pruning rule happens to keep
it or drop it does not shift the model slightly -- it switches between "reads
the answer off the input" and "actually has to predict", and RMSE swings by an
order of magnitude.

That IS still order dependence, and the mechanism is genuine. But the magnitude
is then a property of a leaky dataset rather than of the rule, and quoting 126%
as the cost of tie-breaking would be indefensible.

This script measures max |corr(x_j, y)| per dataset, joins it to the observed
spread, and recomputes the headline figure with the leaky datasets removed. The
number that survives is the one to put in the abstract.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
SHARDS = ROOT / "results" / "shards"

# Above this, a single feature explains almost all of the target on its own.
LEAK_R = 0.95


def feature_target_correlations(X, y):
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Zx = (X - X.mean(axis=0)) / sd
    sy = y.std() or 1.0
    zy = (y - y.mean()) / sy
    r = (Zx.T @ zy) / len(y)
    return np.nan_to_num(r, nan=0.0)


def spreads_by_dataset():
    by = defaultdict(lambda: defaultdict(list))
    for f in SHARDS.glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("perm", 0) > 0 and r.get("tie_break") == "position":
                    by[r["dataset"]][r["perm"]].append(r["rmse"])
    out = {}
    for ds, perms in by.items():
        if len(perms) < 5:
            continue
        means = [st.fmean(v) for v in perms.values()]
        mu = st.fmean(means)
        out[ds] = 100.0 * (max(means) - min(means)) / mu if mu else float("nan")
    return out


def main() -> int:
    spreads = spreads_by_dataset()
    if not spreads:
        print("no permutation results found")
        return 1

    rows = []
    for ds in sorted(spreads):
        path = CACHE / f"{ds}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        X, y = d["X"], d["y"]
        r = np.abs(feature_target_correlations(X, y))
        order = np.argsort(r)[::-1]
        rows.append({
            "dataset": ds,
            "spread": spreads[ds],
            "max_r": float(r[order[0]]),
            "second_r": float(r[order[1]]) if r.size > 1 else 0.0,
            "n_above_95": int((r > LEAK_R).sum()),
            "p": X.shape[1],
        })

    rows.sort(key=lambda x: -x["spread"])

    print("=" * 76)
    print("Does a single feature nearly equal the target?")
    print("=" * 76)
    print(f"  {'dataset':18s}{'spread%':>9s}{'max|r(x,y)|':>13s}{'2nd':>8s}"
          f"{'#>0.95':>8s}   flag")
    print("  " + "-" * 66)
    for r in rows:
        flag = "LEAKY" if r["max_r"] > LEAK_R else ""
        print(f"  {r['dataset']:18s}{r['spread']:>9.2f}{r['max_r']:>13.3f}"
              f"{r['second_r']:>8.3f}{r['n_above_95']:>8d}   {flag}")

    leaky = [r for r in rows if r["max_r"] > LEAK_R]
    clean = [r for r in rows if r["max_r"] <= LEAK_R]

    print()
    print("=" * 76)
    print("HEADLINE FIGURE")
    print("=" * 76)
    all_s = [r["spread"] for r in rows]
    print(f"  all {len(rows)} datasets      median spread {st.median(all_s):6.2f}%"
          f"   max {max(all_s):7.2f}%")
    if clean:
        cs = [r["spread"] for r in clean]
        print(f"  {len(clean)} clean datasets   median spread {st.median(cs):6.2f}%"
              f"   max {max(cs):7.2f}%   <- quote THIS")
    if leaky:
        ls = [r["spread"] for r in leaky]
        print(f"  {len(leaky)} leaky datasets   median spread {st.median(ls):6.2f}%"
              f"   max {max(ls):7.2f}%")
        print("\n  Leaky datasets carry a feature with |r| > 0.95 against the target.")
        print("  Their spreads are real but confounded: the rule is switching between")
        print("  keeping and dropping a near-copy of y. Report them separately, or")
        print("  exclude them and say why. Do not put them in the headline number.")

    if clean and leaky:
        print("\n  The finding survives either way. What changes is the size of")
        print("  the effect that can be defended, so the leaky datasets are")
        print("  reported separately rather than folded into the headline figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
