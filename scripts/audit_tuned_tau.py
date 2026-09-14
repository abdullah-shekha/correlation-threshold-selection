"""Why is corr_tuned the least stable arm? Separate the two possible causes.

    python scripts/audit_tuned_tau.py

Figure 4 places corr_tuned at the bottom of the stability axis (Phi ~ 0.82)
while it is the one arm that costs nothing in accuracy, and the findings
document leans on that pairing. But corr_tuned is not like the other arms, and
the difference matters for what its Phi means.

Every fixed-threshold arm has ONE source of fold-to-fold variation: which
features happen to clear a constant tau in this particular training split.
corr_tuned has TWO, because the inner cross-validation re-chooses tau on every
outer fold. A fold that settles on tau = 0.99 prunes almost nothing; one that
settles on 0.70 prunes aggressively. Those two selections disagree heavily, and
the Nogueira index counts the disagreement as instability without caring where
it came from.

So the question is not whether corr_tuned is unstable -- it measurably is --
but whether it is unstable in the same SENSE as the others. This script
answers that by decomposing:

    Phi_overall     computed over all 50 outer folds, as Figure 4 reports it.

    Phi_within_tau  computed separately within each group of folds that chose
                    the SAME tau, then averaged over groups weighted by group
                    size. This is the fixed-threshold quantity: selection
                    instability with the threshold held constant.

If Phi_within_tau is close to the fixed arms' values while Phi_overall is far
below them, the low score is an artefact of threshold selection rather than a
property comparable to the other arms, and the paper must say so. If the two
are similar, corr_tuned is genuinely unstable at a fixed threshold too, and the
comparison in Figure 4 stands as drawn.

The tuning grid is TAU_GRID = [0.60, 0.70, 0.80, 0.90, 0.95, 0.99]; the level
that prunes nothing on most datasets is 0.99.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colreg.analysis.metrics import nogueira_stability          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "results" / "shards"
MANIFEST = ROOT / "data" / "corpus_manifest.json"
EXCLUDED = ROOT / "data" / "excluded_datasets.json"
STABILITY = ROOT / "results" / "stability.json"

ARM = "corr_tuned"
EST_ORDER = ["ols", "ridge", "lasso", "enet", "cart"]


def bits(hex_support: str, p: int) -> np.ndarray:
    v = int(hex_support, 16)
    return np.array([(v >> j) & 1 for j in range(p)], dtype=float)


def prunable():
    man = json.loads(MANIFEST.read_text())["datasets"]
    drop = set()
    if EXCLUDED.exists():
        drop = set(json.loads(EXCLUDED.read_text()).get("datasets", {}))
    return {k for k, v in man.items()
            if k not in drop and v["actual"].get("n_pairs_above_80", 0) > 0}


missing: dict = defaultdict(int)


def load():
    """(estimator, dataset) -> list of (tau, support_vector) over outer folds."""
    out = defaultdict(list)
    for f in sorted(SHARDS.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if (r.get("selection") != ARM or r.get("perm", 0) != 0
                        or r.get("fit_scope") != "train"
                        or r.get("tie_break") != "position"
                        or not r.get("support")):
                    continue
                try:
                    tau = json.loads(r.get("best_params") or "{}").get("tau")
                except json.JSONDecodeError:
                    tau = None
                if tau is None:
                    # A tuned fold must record the tau it settled on. If it does
                    # not, the decomposition below cannot be computed for that
                    # fold, and silently treating a missing value as a group of
                    # its own would invent stability that was never measured.
                    missing[(r["estimator"], r["dataset"])] += 1
                    continue
                out[(r["estimator"], r["dataset"])].append(
                    (float(tau), bits(r["support"], r["n_features_in"])))
    return out


def within_tau_phi(entries):
    """Phi computed inside each same-tau group, weighted by group size."""
    groups = defaultdict(list)
    for tau, vec in entries:
        groups[tau].append(vec)
    vals, weights = [], []
    for tau, vecs in groups.items():
        if len(vecs) < 2:
            continue
        if len({v.size for v in vecs}) != 1:
            continue
        vals.append(nogueira_stability(np.vstack(vecs)))
        weights.append(len(vecs))
    if not vals:
        return float("nan"), 0
    total = sum(weights)
    return sum(v * w for v, w in zip(vals, weights)) / total, len(vals)


def main() -> int:
    data = load()
    if not data:
        print(f"no {ARM} results with support masks found")
        return 1
    pr = prunable()

    print("=" * 78)
    print(f"HOW OFTEN DOES THE INNER LOOP CHANGE tau?  (arm = {ARM})")
    print("=" * 78)
    print(f"  {'dataset':18s}{'est':7s}{'folds':>6s}{'taus':>6s}"
          f"{'modal tau':>11s}{'modal share':>13s}")
    print("  " + "-" * 62)

    n_multi, n_cells = 0, 0
    for (e, ds), entries in sorted(data.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if ds not in pr:
            continue
        taus = [t for t, _ in entries]
        counts = defaultdict(int)
        for t in taus:
            counts[t] += 1
        modal = max(counts, key=counts.get)
        share = counts[modal] / len(taus)
        n_cells += 1
        n_multi += len(counts) > 1
        print(f"  {ds:18s}{e:7s}{len(taus):>6d}{len(counts):>6d}"
              f"{modal:>11.2f}{100*share:>12.0f}%")

    print(f"\n  {n_multi} of {n_cells} cells used more than one tau across folds.")
    if missing:
        total = sum(missing.values())
        print(f"  WARNING: {total} folds across {len(missing)} cells recorded no "
              f"tau in best_params")
        print("  and were dropped. Investigate before trusting the decomposition.")

    # ------------------------------------------------ the decomposition
    print("\n" + "=" * 78)
    print("DOES THE INSTABILITY SURVIVE HOLDING tau FIXED?")
    print("=" * 78)
    # The band to compare against is measured, not asserted: read the fixed
    # arms' own Phi from analyse_stability.py's output rather than quoting a
    # remembered range.
    band = {}
    if STABILITY.exists():
        S = json.loads(STABILITY.read_text())
        tbl, comparable = S["phi_by_arm_estimator"], S["comparable_arms"]
        for e in EST_ORDER:
            vals = [tbl[a][e] for a in comparable
                    if a != ARM and e in tbl.get(a, {})]
            if vals:
                band[e] = (min(vals), max(vals))
    else:
        print("  (results/stability.json absent -- run analyse_stability.py to")
        print("   compare against the fixed arms' measured range)\n")

    print(f"  {'est':7s}{'Phi overall':>14s}{'Phi within tau':>17s}"
          f"{'fixed arms':>20s}")
    print("  " + "-" * 60)

    summary = {}
    for e in EST_ORDER:
        overall, within = [], []
        for (est, ds), entries in data.items():
            if est != e or ds not in pr:
                continue
            if len({v.size for _, v in entries}) != 1 or len(entries) < 2:
                continue
            overall.append(nogueira_stability(np.vstack([v for _, v in entries])))
            w, _ = within_tau_phi(entries)
            if not np.isnan(w):
                within.append(w)
        if not overall:
            continue
        o, w = st.fmean(overall), (st.fmean(within) if within else float("nan"))
        summary[e] = (o, w)
        b = f"{band[e][0]:.3f} - {band[e][1]:.3f}" if e in band else "--"
        print(f"  {e:7s}{o:>14.3f}{w:>17.3f}{b:>20s}")

    print("\n" + "=" * 78)
    print("DECOMPOSITION")
    print("=" * 78)
    if not summary:
        print("  not enough data to decompose")
        return 0
    o = st.fmean(v[0] for v in summary.values())
    w = st.fmean(v[1] for v in summary.values() if not np.isnan(v[1]))
    print(f"  mean Phi overall     {o:.3f}")
    print(f"  mean Phi within tau  {w:.3f}")
    if np.isnan(w):
        print("\n  Every fold chose the same tau, so there is nothing to decompose:")
        print("  the low score is ordinary selection instability and Figure 4 stands.")
    elif w - o > 0.05:
        lows = [band[e][0] for e in band]
        reaches = bool(lows) and w >= min(lows)
        print(f"\n  Holding tau fixed recovers {w - o:.3f} of stability.")
        if lows and reaches:
            print(f"  That brings corr_tuned to {w:.3f}, within the range the")
            print(f"  fixed-threshold arms occupy (lowest {min(lows):.3f}). Its low")
            print("  score in Figure 4 is therefore driven by the threshold moving")
            print("  between folds rather than by the selection being noisier at a")
            print("  given threshold.")
        elif lows:
            print(f"  But {w:.3f} is still below the fixed arms' lowest ({min(lows):.3f}),")
            print("  so threshold movement explains part of the gap and not all of it.")
            print("  Both causes are present; neither alone accounts for the score.")
        else:
            print(f"  Whether {w:.3f} reaches the fixed arms' range cannot be judged")
            print("  here -- run analyse_stability.py first so the band is measured.")
        print("\n  Either way this is a real property of tuning: a practitioner who")
        print("  tunes tau does get a feature set that changes on resampling.")
        print("  It is not the same quantity the other arms' Phi measures, which")
        print("  is why the figure and the text distinguish the two.")
    else:
        print("\n  Holding tau fixed changes little, so corr_tuned is genuinely")
        print("  unstable at a fixed threshold as well. The comparison in Figure 4")
        print("  is like-for-like and needs no caveat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
