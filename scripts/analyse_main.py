"""RQ1: does correlation-threshold pruning beat not pruning at all?

    python scripts/analyse_main.py

Works with whatever is on disk. The CART cells for the large datasets are still
outstanding, so coverage is reported per estimator and any estimator with
incomplete coverage is analysed only over the datasets where every arm finished.
That is the correct handling anyway: comparing arms across different dataset
sets would confound the arm with the corpus.

Design notes that the numbers depend on:

*   Comparisons are made on per-dataset mean RMSE, ranked WITHIN each dataset.
    RMSE is not comparable across datasets -- different units, different scales
    -- so averaging it across a corpus is meaningless. Ranks are.

*   The test against 'none' is Wilcoxon signed-rank with Holm correction, not
    all-pairs Nemenyi. The question is "does anything beat doing nothing",
    which is a control-vs-rest design; Holm is more powerful there and easier
    to defend than a post-hoc that spends power on comparisons nobody asked
    about.

*   Results are split by whether the pruning rule actually acts. On the seven
    inert datasets no feature clears the threshold, so the pruning arms are
    PROVABLY identical to the control and any difference is pure noise. That
    subset is the measurement floor for reading the rest.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "results" / "shards"
MANIFEST = ROOT / "data" / "corpus_manifest.json"

CONTROL = "none"


def load_main():
    """Per (estimator, dataset, selection): mean RMSE over folds."""
    acc = defaultdict(list)
    for f in SHARDS.glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("perm", 0) != 0:
                    continue
                if r.get("fit_scope") != "train" or r.get("tie_break") != "position":
                    continue
                acc[(r["estimator"], r["dataset"], r["selection"])].append(r["rmse"])
    return {k: st.fmean(v) for k, v in acc.items()}


def prunable_sets():
    if not MANIFEST.exists():
        return None
    man = json.loads(MANIFEST.read_text())["datasets"]
    pr = {k for k, v in man.items() if v["actual"].get("n_pairs_above_80", 0) > 0}
    inert = set(man) - pr
    return pr, inert


def analyse(means, estimator, datasets, arms, label):
    complete = [d for d in datasets
                if all((estimator, d, a) in means for a in arms)]
    if len(complete) < 5:
        print(f"    {label}: only {len(complete)} complete datasets -- skipped")
        return

    M = np.array([[means[(estimator, d, a)] for a in arms] for d in complete])
    ranks = np.apply_along_axis(stats.rankdata, 1, M)      # lower RMSE = rank 1
    avg = ranks.mean(axis=0)

    ctrl = arms.index(CONTROL)
    print(f"    {label}  ({len(complete)} datasets)")
    order = np.argsort(avg)
    for i in order:
        star = "  <- control" if i == ctrl else ""
        print(f"      {arms[i]:12s} mean rank {avg[i]:5.2f}{star}")

    # Wilcoxon vs the control, Holm-corrected.
    raw = []
    for j, a in enumerate(arms):
        if j == ctrl:
            continue
        diff = M[:, j] - M[:, ctrl]
        if np.allclose(diff, 0):
            p = 1.0
        else:
            p = stats.wilcoxon(M[:, j], M[:, ctrl]).pvalue
        raw.append([a, p, float(np.median(100 * diff / M[:, ctrl]))])
    raw.sort(key=lambda r: r[1])
    m, running = len(raw), 0.0
    sig = []
    for i, rec in enumerate(raw):
        running = max(running, min(1.0, (m - i) * rec[1]))
        rec.append(running)
        if running < 0.05:
            sig.append(rec)
    if sig:
        print(f"      significant vs '{CONTROL}' after Holm:")
        for a, _, delta, ph in sig:
            direction = "better" if delta < 0 else "WORSE"
            print(f"        {a:12s} p={ph:.4f}  median RMSE {delta:+.2f}%  ({direction})")
    else:
        print(f"      nothing beats '{CONTROL}' after Holm correction")


def main() -> int:
    means = load_main()
    if not means:
        print("no main-stage results found")
        return 1

    estimators = sorted({k[0] for k in means})
    arms = sorted({k[2] for k in means})
    if CONTROL not in arms:
        print(f"control arm '{CONTROL}' missing")
        return 1

    print("=" * 74)
    print("COVERAGE")
    print("=" * 74)
    for e in estimators:
        ds = {k[1] for k in means if k[0] == e}
        full = [d for d in ds if all((e, d, a) in means for a in arms)]
        flag = "" if len(full) == len(ds) else "   <- incomplete, still running"
        print(f"  {e:8s} {len(ds):3d} datasets seen, {len(full):3d} with all "
              f"{len(arms)} arms{flag}")

    sets = prunable_sets()
    print("\n" + "=" * 74)
    print("RQ1  DOES PRUNING BEAT NOT PRUNING?")
    print("=" * 74)
    for e in estimators:
        print(f"\n  {e.upper()}")
        ds = sorted({k[1] for k in means if k[0] == e})
        analyse(means, e, ds, arms, "all datasets")
        if sets:
            pr, inert = sets
            analyse(means, e, [d for d in ds if d in pr], arms, "prunable only")
            analyse(means, e, [d for d in ds if d in inert], arms,
                    "INERT (floor: any difference here is noise)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
