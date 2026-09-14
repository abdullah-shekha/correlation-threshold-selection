"""Re-derive every table in Findings.docx from the shards and diff.

    python scripts/verify_findings.py

Why this exists. Each number in the findings document reached the document by
being printed on this machine, pasted into a chat window and retyped into a
generator script. That path has no checksum. It has been right so far as far as
anyone can tell, but "as far as anyone can tell" is doing a lot of work in that
sentence, and the numbers are about to go into a manuscript.

So this script recomputes the tables from results/shards/ and compares them to
the values hard-coded below, which were copied OUT of the document. Any
disagreement is printed as a MISMATCH line. A clean run means the document and
the data agree; it does not mean either is correct, only that the transcription
step introduced nothing.

The recomputation deliberately duplicates the logic in analyse_main.py rather
than importing it. Importing would test that the document matches the analysis
script; duplicating tests that the document matches the DATA, and would catch a
bug in the analysis script that had been baked into both.

One known wrinkle it reports rather than hides: the document's Table 3 takes its
Spread % from audit_rq5.py, which pools estimators before computing the range,
and its Subsets count from analyse_rq5.py, which groups by dataset AND
estimator. Both are computed here so the difference is visible.
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
CACHE = ROOT / "data" / "cache"
MANIFEST = ROOT / "data" / "corpus_manifest.json"
EXCLUDED = ROOT / "data" / "excluded_datasets.json"

CONTROL = "none"
LEAK_R = 0.95
EST_ORDER = ["ols", "ridge", "lasso", "enet", "cart"]

FAILURES: list[str] = []


def check(label, got, want, tol=0.005):
    """Compare a computed number to the documented one."""
    if want is None:
        return
    if got is None:
        FAILURES.append(f"{label}: document says {want}, data yields nothing")
        return
    if abs(got - want) > tol:
        FAILURES.append(f"{label}: document says {want}, data yields {got:.4f}")


# --------------------------------------------------------------- documented
# Table 1: mean rank per arm, prunable datasets. Columns follow EST_ORDER.
DOC_RANKS = {
    "none":       [2.25, 2.39, 2.14, 2.31, 3.50],
    "corr_tuned": [2.75, 2.28, 2.61, 2.44, 4.24],
    "corr_095":   [3.42, 3.72, 3.58, 3.81, 4.18],
    "corr_090":   [4.50, 4.53, 4.50, 4.56, 4.56],
    "vif_10":     [5.31, 5.36, 5.28, 5.17, 3.71],
    "pca_95":     [6.17, 6.22, 6.17, 6.06, 7.35],
    "corr_080":   [6.39, 6.33, 6.44, 6.33, 5.76],
    "vif_05":     [7.14, 7.06, 7.11, 7.17, 5.18],
    "corr_070":   [7.08, 7.11, 7.17, 7.17, 6.53],
}

# Table 2: median % RMSE change vs control where Holm-significant, else None.
DOC_HARM = {
    "vif_05":     [11.67, 11.68, 10.52, 10.68, None],
    "corr_070":   [8.80, 10.00, 8.60, 9.19, None],
    "pca_95":     [9.29, 9.74, 9.05, 9.78, None],
    "corr_080":   [7.20, 7.34, 6.70, 6.57, None],
    "corr_090":   [3.25, 3.21, 3.41, 3.48, None],
    "vif_10":     [1.62, 1.84, 1.48, 1.45, None],
    "corr_095":   [None, None, None, None, None],
    "corr_tuned": [None, None, None, None, None],
}

# Table 3: dataset -> (spread %, distinct subsets).
DOC_SPREADS = {
    "auto_mpg": (17.31, 4), "servo": (15.97, 2), "energy_y1": (15.10, 3),
    "energy_y2": (13.13, 3), "automobile": (11.95, 25), "superconduct": (6.97, 30),
    "abalone": (6.86, 6), "real_estate": (3.46, 2), "appliances": (3.12, 24),
    "istanbul_se": (2.98, 3), "communities": (1.97, 30),
    "parkinsons_motor": (0.84, 17), "california": (0.72, 4),
    "bike_hour": (0.43, 4), "diabetes": (0.26, 2),
}

# Table 5: the excluded datasets.
DOC_LEAKY = {"facebook": (0.998, 124.38), "gas_turbine": (0.989, 93.59),
             "cpu_perf": (0.966, 40.48)}

# Limitations: the inert-floor illustration.
DOC_INERT_RIDGE = {"corr_080": 4.83, "none": 3.25}


# ------------------------------------------------------------------ loading
def load_rows():
    if not SHARDS.exists():
        raise SystemExit("no results/shards directory found")
    rows = []
    for f in sorted(SHARDS.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("results/shards is empty")
    return rows


def main_means(rows):
    """Per (estimator, dataset, arm): mean RMSE over folds, main stage only."""
    acc = defaultdict(list)
    for r in rows:
        if r.get("perm", 0) != 0:
            continue
        if r.get("fit_scope") != "train" or r.get("tie_break") != "position":
            continue
        acc[(r["estimator"], r["dataset"], r["selection"])].append(r["rmse"])
    return {k: st.fmean(v) for k, v in acc.items()}


def prunable_inert():
    man = json.loads(MANIFEST.read_text())["datasets"]
    drop = set()
    if EXCLUDED.exists():
        drop = set(json.loads(EXCLUDED.read_text()).get("datasets", {}))
    man = {k: v for k, v in man.items() if k not in drop}
    pr = {k for k, v in man.items() if v["actual"].get("n_pairs_above_80", 0) > 0}
    return pr, set(man) - pr


def ranks_and_harm(means, estimator, datasets, arms):
    complete = [d for d in datasets if all((estimator, d, a) in means for a in arms)]
    if len(complete) < 5:
        return None, None, complete
    M = np.array([[means[(estimator, d, a)] for a in arms] for d in complete])
    avg = np.apply_along_axis(stats.rankdata, 1, M).mean(axis=0)
    ranks = dict(zip(arms, avg))

    ctrl = arms.index(CONTROL)
    raw = []
    for j, a in enumerate(arms):
        if j == ctrl:
            continue
        diff = M[:, j] - M[:, ctrl]
        p = 1.0 if np.allclose(diff, 0) else stats.wilcoxon(M[:, j], M[:, ctrl]).pvalue
        raw.append([a, p, float(np.median(100 * diff / M[:, ctrl]))])
    raw.sort(key=lambda r: r[1])
    harm, running, m = {}, 0.0, len(raw)
    for i, (a, p, delta) in enumerate(raw):
        running = max(running, min(1.0, (m - i) * p))
        harm[a] = delta if running < 0.05 else None
    return ranks, harm, complete


def perm_spread_pooled(rows):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
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


def perm_by_estimator(rows):
    """Spread and modal-subset churn per (dataset, estimator), as analyse_rq5."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("perm", 0) > 0 and r.get("tie_break") == "position":
            by[(r["dataset"], r["estimator"])][r["perm"]].append(r)
    out = defaultdict(dict)
    for (ds, est), perms in by.items():
        if len(perms) < 5:
            continue
        means, subsets = [], set()
        for rs in perms.values():
            means.append(st.fmean(x["rmse"] for x in rs))
            counts = defaultdict(int)
            for x in rs:
                counts[x.get("support")] += 1
            subsets.add(max(counts, key=counts.get))
        mu = st.fmean(means)
        out[ds][est] = (100.0 * (max(means) - min(means)) / mu if mu else float("nan"),
                        len(subsets))
    return out


def max_target_corr(ds):
    path = CACHE / f"{ds}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    X, y = d["X"], d["y"]
    sd = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    Zx = (X - X.mean(axis=0)) / sd
    zy = (y - y.mean()) / (y.std() or 1.0)
    return float(np.abs(np.nan_to_num((Zx.T @ zy) / len(y), nan=0.0)).max())


# -------------------------------------------------------------------- main
def main() -> int:
    rows = load_rows()
    means = main_means(rows)
    pr, inert = prunable_inert()
    arms = sorted({k[2] for k in means})

    print(f"shard rows loaded: {len(rows):,}")
    print(f"arms: {len(arms)}  ({', '.join(arms)})")

    print("\n" + "=" * 78)
    print("COVERAGE")
    print("=" * 78)
    corpus = set(json.loads(MANIFEST.read_text())["datasets"])
    excluded = {}
    if EXCLUDED.exists():
        excluded = json.loads(EXCLUDED.read_text()).get("datasets", {})
    if excluded:
        corpus -= set(excluded)
        for d, why in excluded.items():
            print(f"  excluded from the study: {d}")
            print(f"    {why[:96]}...")
    print(f"  corpus: {len(corpus)} datasets "
          f"({len(pr)} prunable, {len(inert)} inert)")
    for e in EST_ORDER:
        ds = {k[1] for k in means if k[0] == e}
        full = [d for d in ds if all((e, d, a) in means for a in arms)]
        npr = len([d for d in full if d in pr])
        print(f"\n  {e:6s} {len(full):3d} of {len(corpus):3d} complete "
              f"({npr} prunable, {len(full) - npr} inert)")
        absent = sorted(corpus - ds)
        if absent:
            print(f"         never started: {', '.join(absent)}")
        partial = sorted(ds - set(full))
        for d in partial:
            missing = [a for a in arms if (e, d, a) not in means]
            print(f"         incomplete:    {d} -- missing "
                  f"{', '.join(missing)}")

    # Datasets complete for EVERY estimator -- the number the paper can quote
    # when it says "completed every arm for every estimator".
    everywhere = [d for d in sorted(corpus)
                  if all(all((e, d, a) in means for a in arms) for e in EST_ORDER)]
    npr = len([d for d in everywhere if d in pr])
    print(f"\n  complete for ALL five estimators: {len(everywhere)} of "
          f"{len(corpus)}  ({npr} prunable, {len(everywhere) - npr} inert)")
    print(f"         missing somewhere: "
          f"{', '.join(sorted(corpus - set(everywhere)))}")

    # ------------------------------------------------------------- table 1/2
    print("\n" + "=" * 78)
    print("TABLE 1  mean rank, prunable datasets      (doc value in brackets)")
    print("=" * 78)
    print(f"  {'arm':12s}" + "".join(f"{e:>13s}" for e in EST_ORDER))
    got_ranks, got_harm = {}, {}
    for e in EST_ORDER:
        ds = sorted({k[1] for k in means if k[0] == e and k[1] in pr})
        r, h, complete = ranks_and_harm(means, e, ds, arms)
        got_ranks[e], got_harm[e] = r, h
    for a in sorted(DOC_RANKS, key=lambda k: DOC_RANKS[k][1]):
        line = f"  {a:12s}"
        for i, e in enumerate(EST_ORDER):
            want = DOC_RANKS[a][i]
            got = got_ranks[e].get(a) if got_ranks[e] else None
            check(f"Table 1 {a}/{e}", got, want)
            line += f"{got:>7.2f}[{want:4.2f}]" if got is not None else f"{'--':>13s}"
        print(line)

    print("\n" + "=" * 78)
    print("TABLE 2  median % RMSE change vs control, Holm-corrected")
    print("=" * 78)
    print(f"  {'arm':12s}" + "".join(f"{e:>15s}" for e in EST_ORDER))
    for a in DOC_HARM:
        line = f"  {a:12s}"
        for i, e in enumerate(EST_ORDER):
            want = DOC_HARM[a][i]
            got = got_harm[e].get(a) if got_harm[e] else None
            if want is None and got is None:
                line += f"{'n.s. [n.s.]':>15s}"
            elif want is None:
                line += f"{f'{got:+.2f}[n.s.]':>15s}"
                FAILURES.append(f"Table 2 {a}/{e}: document says n.s., "
                                f"data yields {got:+.2f}% significant")
            elif got is None:
                line += f"{f'n.s.[{want:+.2f}]':>15s}"
                FAILURES.append(f"Table 2 {a}/{e}: document says {want:+.2f}%, "
                                f"data yields not significant")
            else:
                check(f"Table 2 {a}/{e}", got, want)
                line += f"{f'{got:+.2f}[{want:+.2f}]':>15s}"
        print(line)

    # ------------------------------------------------------- inert floor
    print("\n" + "=" * 78)
    print("LIMITATIONS  the inert floor is not exactly flat")
    print("=" * 78)
    ds = sorted({k[1] for k in means if k[0] == "ridge" and k[1] in inert})
    r, _, complete = ranks_and_harm(means, "ridge", ds, arms)
    if r:
        print(f"  ridge, {len(complete)} inert datasets:")
        for a, want in DOC_INERT_RIDGE.items():
            got = r.get(a)
            check(f"inert floor ridge/{a}", got, want)
            print(f"    {a:12s} mean rank {got:5.2f}  [doc {want:.2f}]")
        if r.get("corr_080", 0) <= r.get(CONTROL, 0):
            FAILURES.append("inert floor: corr_080 does NOT rank worse than the "
                            "control, contradicting the limitation as written")
    else:
        print("  too few complete inert datasets to rank")

    # ------------------------------------------------------------- table 3
    print("\n" + "=" * 78)
    print("TABLE 3  permutation spread")
    print("=" * 78)
    pooled = perm_spread_pooled(rows)
    per_est = perm_by_estimator(rows)
    print(f"  {'dataset':18s}{'pooled':>9s}{'[doc]':>9s}"
          + "".join(f"{e:>10s}" for e in ("ridge", "lasso"))
          + f"{'subsets':>9s}{'[doc]':>7s}")
    print("  " + "-" * 74)
    for ds, (want_s, want_n) in sorted(DOC_SPREADS.items(), key=lambda kv: -kv[1][0]):
        got_s = pooled.get(ds)
        check(f"Table 3 spread {ds}", got_s, want_s)
        cells = per_est.get(ds, {})
        subs = {n for _, n in cells.values()}
        line = f"  {ds:18s}"
        line += f"{got_s:>9.2f}" if got_s is not None else f"{'--':>9s}"
        line += f"{want_s:>9.2f}"
        for e in ("ridge", "lasso"):
            line += f"{cells[e][0]:>10.2f}" if e in cells else f"{'--':>10s}"
        line += f"{('/'.join(str(x) for x in sorted(subs)) or '--'):>9s}{want_n:>7d}"
        if want_n not in subs:
            FAILURES.append(f"Table 3 subsets {ds}: document says {want_n}, "
                            f"data yields {sorted(subs)}")
        print(line)

    print("\n  NOTE: 'pooled' averages estimators before taking the range, which is")
    print("  what the document quotes. The per-estimator columns are the same")
    print("  quantity computed separately; where they straddle the pooled value the")
    print("  document's figure is a blend, and the paper should say which it means.")

    # ------------------------------------------------------------- table 5
    print("\n" + "=" * 78)
    print("TABLE 5  excluded datasets")
    print("=" * 78)
    for ds, (want_r, want_s) in DOC_LEAKY.items():
        got_r, got_s = max_target_corr(ds), pooled.get(ds)
        check(f"Table 5 max|r| {ds}", got_r, want_r, tol=0.0015)
        check(f"Table 5 spread {ds}", got_s, want_s, tol=0.02)
        rs = f"{got_r:.3f}" if got_r is not None else "--"
        ss = f"{got_s:.2f}" if got_s is not None else "--"
        print(f"  {ds:14s} max|r| {rs} [doc {want_r:.3f}]   "
              f"spread {ss} [doc {want_s:.2f}]")
        if got_r is not None and got_r <= LEAK_R:
            FAILURES.append(f"Table 5 {ds}: max|r| {got_r:.3f} is below the 0.95 "
                            f"exclusion threshold -- the exclusion is unjustified")

    # ------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if not FAILURES:
        print("  All documented values reproduce from the shards.")
        print("  The transcription chain introduced no errors.")
        return 0
    print(f"  {len(FAILURES)} MISMATCH(ES). The document is the thing to correct,")
    print("  unless this script is first shown to be the one at fault.")
    for f in FAILURES:
        print(f"    - {f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
