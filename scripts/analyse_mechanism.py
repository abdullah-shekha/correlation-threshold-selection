"""What decides whether the tie-break matters? And does target_corr fix it?

    python scripts/analyse_mechanism.py

Motivation. Two datasets in the corpus have identical pruning structure -- one
pair above 0.8, two distinct subsets across permutations -- and wildly different
consequences:

    servo      spread 15.97%
    diabetes   spread  0.26%

Sixty-fold, from the same structure. Something other than "how much collinearity
is present" is driving this, and the paper needs to name it.

The hypothesis: the choice matters when the tied features differ in how well
they predict y. Swapping two interchangeable predictors costs nothing; swapping
a strong predictor for a weak one costs a great deal. Formally, for each pair
(a, b) with |corr(x_a, x_b)| > tau, define

    predictive asymmetry  =  | |corr(x_a, y)| - |corr(x_b, y)| |

If the hypothesis holds, dataset-level asymmetry should predict the observed
permutation spread, and 'target_corr' -- which breaks ties by keeping whichever
feature correlates more strongly with y -- should win precisely where asymmetry
is high.

CAVEAT on the proxy, established by simulation before trusting it: the
semi-partial estimate is itself unstable for a tied pair, because dividing by
sqrt(1 - r_ab^2) amplifies the estimate ~6x at r_ab = 0.99 and amplifies its
sampling noise equally. Over 200 replicates of a fixed design it separated
asymmetric from symmetric at 0.114 +/- 0.027 vs 0.034 +/- 0.026 -- informative
in expectation, but a SINGLE dataset's value can land anywhere in the overlap.
Read the Spearman across all datasets, never one dataset's number.

That second prediction is the falsifiable one, and it currently looks shaky:
target_corr won only 6 of 36 cells in the tie-break run. Either the mechanism is
wrong, or target_corr's advantage is concentrated in the high-asymmetry subset
and the overall win count hides it. This script decides which.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
SHARDS = ROOT / "results" / "shards"

TAU = 0.8
LEAK_R = 0.95


def corr_matrix(X):
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (X - X.mean(axis=0)) / sd
    C = np.nan_to_num((Z.T @ Z) / X.shape[0], nan=0.0)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)


def target_corr(X, y):
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Zx = (X - X.mean(axis=0)) / sd
    zy = (y - y.mean()) / (y.std() or 1.0)
    return np.nan_to_num((Zx.T @ zy) / len(y), nan=0.0)


def describe(ds):
    d = np.load(CACHE / f"{ds}.npz")
    X, y = d["X"], d["y"]
    C = corr_matrix(X)
    ry = np.abs(target_corr(X, y))

    rs = target_corr(X, y)          # signed, needed for the partialling
    asym, semi = [], []
    for i in range(X.shape[1]):
        for j in range(i):
            rij = C[i, j]
            if abs(rij) <= TAU:
                continue
            asym.append(abs(ry[i] - ry[j]))

            # Semi-partial correlations: each feature's UNIQUE contribution to y
            # once the other is partialled out.
            #     sp_i = (r_iy - r_jy * r_ij) / sqrt(1 - r_ij^2)
            # The marginal difference |r_iy| - |r_jy| is nearly useless for a tied
            # pair: at r_ij = 0.99 each feature inherits almost all of the
            # other's correlation with y, so two features with completely
            # different true coefficients look nearly identical marginally.
            # On a controlled fixture the marginal measure separated a
            # symmetric from an asymmetric design by 0.003 vs 0.016; the
            # semi-partial separated the same pair by 0.040 vs 0.144.
            den = np.sqrt(max(1.0 - rij * rij, 1e-12))
            sp_i = (rs[i] - rs[j] * rij) / den
            sp_j = (rs[j] - rs[i] * rij) / den
            semi.append(abs(abs(sp_i) - abs(sp_j)))
    if not asym:
        return None
    return {
        "n": X.shape[0], "p": X.shape[1], "n_pairs": len(asym),
        "max_asym": float(max(asym)),
        "mean_asym": float(st.fmean(asym)),
        "max_semi": float(max(semi)),
        "mean_semi": float(st.fmean(semi)),
        "max_ry": float(ry.max()),
    }


def perm_spreads():
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
        m = [st.fmean(v) for v in perms.values()]
        mu = st.fmean(m)
        out[ds] = 100.0 * (max(m) - min(m)) / mu if mu else float("nan")
    return out


def tiebreak_means():
    """Per dataset: mean RMSE of each tie-break rule, averaged over estimators."""
    by = defaultdict(lambda: defaultdict(list))
    for f in SHARDS.glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("perm", 0) == 0 and r.get("selection") == "corr_080":
                    by[r["dataset"]][r["tie_break"]].append(r["rmse"])
    return {ds: {k: st.fmean(v) for k, v in rules.items()} for ds, rules in by.items()}


def main() -> int:
    spreads = perm_spreads()
    if not spreads:
        print("no permutation results found")
        return 1

    rows = []
    for ds in sorted(spreads):
        if not (CACHE / f"{ds}.npz").exists():
            continue
        info = describe(ds)
        if info is None:
            continue
        info.update(dataset=ds, spread=spreads[ds],
                    leaky=info["max_ry"] > LEAK_R)
        rows.append(info)

    clean = [r for r in rows if not r["leaky"]]
    clean.sort(key=lambda r: -r["spread"])

    # ------------------------------------------------- does asymmetry predict?
    print("=" * 78)
    print("H: the tie-break matters when the tied features differ in predicting y")
    print("=" * 78)
    print(f"  {'dataset':18s}{'n':>7s}{'pairs':>7s}{'spread%':>9s}"
          f"{'max semi':>10s}{'mean semi':>11s}{'max marg':>10s}")
    print("  " + "-" * 72)
    for r in clean:
        print(f"  {r['dataset']:18s}{r['n']:>7d}{r['n_pairs']:>7d}"
              f"{r['spread']:>9.2f}{r['max_semi']:>10.3f}{r['mean_semi']:>11.3f}"
              f"{r['max_asym']:>10.3f}")

    if len(clean) >= 5:
        s = [r["spread"] for r in clean]
        for label, key in (("max semi-partial", "max_semi"),
                           ("mean semi-partial", "mean_semi"),
                           ("max marginal asym", "max_asym"),
                           ("mean marginal asym", "mean_asym"),
                           ("log n", "n"),
                           ("n pairs > tau", "n_pairs")):
            v = [np.log(r[key]) if key == "n" else r[key] for r in clean]
            rho, p = stats.spearmanr(v, s)
            supports = key.startswith(("max_semi", "mean_semi", "max_asym", "mean_asym"))
            verdict = "SUPPORTS" if (supports and rho > 0.5 and p < 0.05) else ""
            print(f"\n  Spearman(spread, {label:16s}) rho = {rho:+.3f}   p = {p:.4f}  {verdict}")

    # ------------------------------------------ the servo / diabetes contrast
    pair = {r["dataset"]: r for r in rows if r["dataset"] in ("servo", "diabetes")}
    if len(pair) == 2:
        print("\n" + "=" * 78)
        print("THE DECISIVE CONTRAST  (same structure, 60x different outcome)")
        print("=" * 78)
        for ds in ("servo", "diabetes"):
            r = pair[ds]
            print(f"  {ds:10s} n={r['n']:5d} pairs={r['n_pairs']}  "
                  f"spread={r['spread']:6.2f}%  max semi-partial={r['max_semi']:.3f}"
                  f"  max marginal={r['max_asym']:.3f}")
        hi = "servo" if pair["servo"]["max_semi"] > pair["diabetes"]["max_semi"] else "diabetes"
        big = "servo" if pair["servo"]["spread"] > pair["diabetes"]["spread"] else "diabetes"
        print(f"\n  higher asymmetry: {hi};  larger spread: {big}"
              f"   -> {'CONSISTENT' if hi == big else 'INCONSISTENT -- hypothesis fails here'}")

    # --------------------------------------------- does target_corr exploit it?
    print("\n" + "=" * 78)
    print("Does target_corr win where asymmetry is high?")
    print("=" * 78)
    means = tiebreak_means()
    ranked = []
    for r in clean:
        m = means.get(r["dataset"])
        if not m or "target_corr" not in m:
            continue
        order = sorted(m, key=m.get)                     # lower RMSE = better
        rank = order.index("target_corr") + 1
        pos_rank = order.index("position") + 1 if "position" in order else float("nan")
        ranked.append({**r, "tc_rank": rank, "pos_rank": pos_rank, "n_rules": len(order)})

    if ranked:
        # Split on the OBSERVED spread rather than the asymmetry proxy: the
        # question is whether target_corr helps where the choice demonstrably
        # matters, and spread measures that directly with no proxy in between.
        ranked.sort(key=lambda r: -r["spread"])
        print(f"  {'dataset':18s}{'max semi':>10s}{'spread%':>9s}"
              f"{'target_corr rank':>18s}{'position rank':>15s}")
        print("  " + "-" * 70)
        for r in ranked:
            print(f"  {r['dataset']:18s}{r['max_semi']:>10.3f}{r['spread']:>9.2f}"
                  f"{r['tc_rank']:>13d}/{r['n_rules']:<4d}{r['pos_rank']:>10.0f}/{r['n_rules']:<4d}")

        half = len(ranked) // 2
        hi_a = [r["tc_rank"] for r in ranked[:half]]
        lo_a = [r["tc_rank"] for r in ranked[half:]]
        po_hi = [r["pos_rank"] for r in ranked[:half]]
        print(f"\n  high-spread half: target_corr mean rank {st.fmean(hi_a):.2f}, "
              f"position {st.fmean(po_hi):.2f}")
        print(f"  low-spread  half: target_corr mean rank {st.fmean(lo_a):.2f}")
        if st.fmean(hi_a) < st.fmean(lo_a) - 0.5:
            print("\n  SUPPORTS the mechanism: target_corr does better precisely where the")
            print("  choice demonstrably matters. Recommend it as the rule, and report")
            print("  that its advantage is concentrated rather than uniform -- which is")
            print("  why its overall win count looked unimpressive.")
        else:
            print("\n  DOES NOT support it. target_corr is no better where the spread is")
            print("  large, so 'keep the more predictive feature' is not the fix. Then the")
            print("  honest conclusion is stronger and simpler: no tie-break rule is")
            print("  reliably right, and the practitioner should stop pruning and let the")
            print("  penalty handle collinearity. Check the corr_080-vs-none comparison")
            print("  in the main grid before committing to that claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
