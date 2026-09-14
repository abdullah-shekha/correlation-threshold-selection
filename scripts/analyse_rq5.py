"""RQ5 go/no-go: does the tie-break choice matter enough to carry a paper?

    python scripts/analyse_rq5.py

Reads the tiebreak and permutation shards and answers one question: when the
same data, the same estimator and the same threshold are held fixed and ONLY the
column order changes, how much does the fitted model change?

Two quantities per dataset:

    subset churn   how many DISTINCT feature subsets the permutations produce.
                   1 means the rule is effectively order-invariant on this data.
    RMSE spread    (max - min) across permutations, as a percentage of the mean.
                   This is pure artefact: nothing about the data changed.

The thresholds below were fixed before the run, so that the interpretation was
not chosen after seeing the numbers:

    stop      median spread < 0.5%  AND  median distinct subsets <= 2
    proceed   median spread >= 1%   OR   median distinct subsets >= 5
    between   real but small, and reportable only as a secondary result

What matters is that the cut-offs were set in advance, not where exactly they
sit.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "results" / "shards"


def load(stage_filter):
    rows = []
    for f in SHARDS.glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if stage_filter(r):
                    rows.append(r)
    return rows


def summarise_permutations(rows):
    """Per (dataset, estimator): subset churn and RMSE spread over permutations."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["dataset"], r["estimator"])][r["perm"]].append(r)

    out = []
    for (ds, est), perms in sorted(by.items()):
        if len(perms) < 5:
            continue
        means, subsets = [], set()
        for _, rs in perms.items():
            means.append(st.fmean(x["rmse"] for x in rs))
            # A permutation's subset is stable across folds only if selection is
            # stable; take the modal fold subset as that permutation's choice.
            counts = defaultdict(int)
            for x in rs:
                counts[x["support"]] += 1
            subsets.add(max(counts, key=counts.get))
        lo, hi, mu = min(means), max(means), st.fmean(means)
        out.append({
            "dataset": ds, "estimator": est, "n_perms": len(perms),
            "rmse_mean": mu,
            "spread_pct": 100.0 * (hi - lo) / mu if mu else float("nan"),
            "sd_pct": 100.0 * st.pstdev(means) / mu if mu else float("nan"),
            "n_subsets": len(subsets),
        })
    return out


def summarise_tiebreaks(rows):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["dataset"], r["estimator"])][r["tie_break"]].append(r["rmse"])
    out = []
    for (ds, est), tbs in sorted(by.items()):
        if len(tbs) < 2:
            continue
        means = {k: st.fmean(v) for k, v in tbs.items()}
        best = min(means, key=means.get)
        worst = max(means, key=means.get)
        mu = st.fmean(means.values())
        out.append({
            "dataset": ds, "estimator": est,
            "best": best, "worst": worst,
            "gap_pct": 100.0 * (means[worst] - means[best]) / mu if mu else float("nan"),
            "means": means,
        })
    return out


def main() -> int:
    if not SHARDS.exists():
        print("no results/shards -- run the tiebreak and permutation stages first")
        return 1

    perm_rows = load(lambda r: r.get("perm", 0) > 0 and r.get("tie_break") == "position")
    ctrl_rows = load(lambda r: r.get("perm", 0) > 0 and r.get("tie_break") == "mean_corr")
    tb_rows = load(lambda r: r.get("perm", 0) == 0 and r.get("selection") == "corr_080")

    # ------------------------------------------------------------ permutation
    print("=" * 78)
    print("RQ5a  COLUMN-ORDER SENSITIVITY  (same data, same model, order permuted)")
    print("=" * 78)
    perms = summarise_permutations(perm_rows)
    if not perms:
        print("  no permutation results yet:")
        print("    python scripts/run_study.py --stage permutation --jobs 12")
    else:
        print(f"  {'dataset':18s}{'est':7s}{'perms':>7s}{'RMSE':>10s}"
              f"{'spread%':>9s}{'sd%':>7s}{'subsets':>9s}")
        print("  " + "-" * 68)
        for r in perms:
            print(f"  {r['dataset']:18s}{r['estimator']:7s}{r['n_perms']:>7d}"
                  f"{r['rmse_mean']:>10.4f}{r['spread_pct']:>9.2f}"
                  f"{r['sd_pct']:>7.2f}{r['n_subsets']:>9d}")

        ctrl = summarise_permutations(ctrl_rows)
        if ctrl:
            c_spread = st.median([r["spread_pct"] for r in ctrl])
            c_churn = st.median([r["n_subsets"] for r in ctrl])
            print(f"\n  NULL CONTROL (mean_corr, order-invariant by construction), "
                  f"{len(ctrl)} cells:")
            print(f"    median spread {c_spread:.3f}%   median distinct subsets {c_churn:.0f}")
            print("    This is the floor. Whatever 'position' shows must clear it.")
            if c_spread > 0.25 or c_churn > 1:
                print("    WARNING: the control is NOT flat. Something other than column")
                print("    order is moving the results -- investigate before trusting RQ5.")

        spreads = [r["spread_pct"] for r in perms]
        churn = [r["n_subsets"] for r in perms]
        med_s, med_c = st.median(spreads), st.median(churn)
        print(f"\n  median spread {med_s:.2f}%   median distinct subsets {med_c:.0f}"
              f"   max spread {max(spreads):.2f}%   max subsets {max(churn)}")

    # -------------------------------------------------------------- tie-break
    print()
    print("=" * 78)
    print("RQ5b  TIE-BREAK RULE  (position = Python idiom, mean_corr = caret's rule)")
    print("=" * 78)
    tbs = summarise_tiebreaks(tb_rows)
    if not tbs:
        print("  no tie-break results yet:")
        print("    python scripts/run_study.py --stage tiebreak --jobs 12")
    else:
        print(f"  {'dataset':18s}{'est':7s}{'best rule':18s}{'worst rule':18s}{'gap%':>7s}")
        print("  " + "-" * 68)
        for r in tbs:
            print(f"  {r['dataset']:18s}{r['estimator']:7s}{r['best']:18s}"
                  f"{r['worst']:18s}{r['gap_pct']:>7.2f}")

        wins = defaultdict(int)
        for r in tbs:
            wins[r["best"]] += 1
        print("\n  times each rule was best: "
              + ", ".join(f"{k}={v}" for k, v in sorted(wins.items(), key=lambda kv: -kv[1])))
        print(f"  median best-vs-worst gap: {st.median([r['gap_pct'] for r in tbs]):.2f}%")

    # ------------------------------------------------------ pre-registered test
    print()
    print("=" * 78)
    print("RESULT AGAINST THE PRE-REGISTERED THRESHOLDS")
    print("=" * 78)
    if not perms:
        print("  permutation stage not run")
        return 0

    med_s, med_c = st.median(spreads), st.median(churn)
    if med_s < 0.5 and med_c <= 2:
        print("  Below the stop threshold.")
        print(f"  Median spread {med_s:.2f}% with {med_c:.0f} distinct subsets: the rule is")
        print("  effectively order-invariant on real data.")
    elif med_s >= 1.0 or med_c >= 5:
        print("  Above the proceed threshold.")
        print(f"  Median spread {med_s:.2f}% across permutations of identical data,")
        print(f"  with a median of {med_c:.0f} distinct feature subsets.")
    else:
        print("  Between the two thresholds.")
        print(f"  Median spread {med_s:.2f}% with {med_c:.0f} distinct subsets: the effect is")
        print("  real but modest, and concentration by collinearity or n/p decides")
        print("  how much weight it can carry.")

    if ctrl_rows:
        c = summarise_permutations(ctrl_rows)
        if c:
            cs = st.median([r["spread_pct"] for r in c])
            print(f"\n  Effect relative to control: position {med_s:.2f}% vs "
                  f"mean_corr {cs:.3f}%  (ratio {med_s / cs:.0f}x)" if cs > 0
                  else f"\n  Control spread is exactly 0; position is {med_s:.2f}%.")

    worst = max(perms, key=lambda r: r["spread_pct"])
    print(f"\n  Largest single effect: {worst['dataset']} / {worst['estimator']}, "
          f"spread {worst['spread_pct']:.2f}%, {worst['n_subsets']} subsets.")
    print("  That case is checked individually in scripts/audit_rq5.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
