"""Does pruning stabilise WHICH features get selected? (RQ3)

    python scripts/analyse_stability.py

Motivation. RQ1 established that pruning does not improve accuracy. That is a
null on the accuracy axis, and a reviewer is entitled to ask whether the
practice buys something else instead. The usual defence of correlation
filtering is not really about RMSE at all -- it is about interpretability:
"with two collinear predictors the coefficients are unstable, so drop one and
the model becomes readable." That is a claim about SELECTION stability, and it
is testable from data already on disk.

Every shard records ``support``, a hex bitmask of the surviving columns, for
each of the 50 outer folds. Nogueira, Sechidis & Brown's index (JMLR 2018)
measures how consistently a selector picks the same features across resamples,
corrected for chance: 1.0 means every fold chose identically, 0.0 means no
better than random subsets of the same average size.

    Phi = 1 - mean_j(s_j^2) / [ (k/p)(1 - k/p) ]

A TRAP that this script refuses to fall into. For the 'none' arm every fold
selects every column, so the per-feature variance is zero, the denominator is
also zero, and the implementation returns 1.0 by convention. That is not
evidence that not pruning is maximally stable -- it is the index being
undefined when nothing varies. Reporting it alongside the pruning arms would
manufacture a result out of a degenerate case. It is shown here in its own
column, marked, and excluded from every comparison and correlation.

'pca_95' is excluded outright: it emits components rather than a subset of the
original columns, so its bitmask is not commensurable with the others.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colreg.analysis.metrics import nogueira_stability      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "results" / "shards"
MANIFEST = ROOT / "data" / "corpus_manifest.json"
OUT = ROOT / "results" / "stability.json"

CONTROL = "none"
DEGENERATE = {CONTROL}          # selects everything: Phi undefined, not 1.0
INCOMMENSURABLE = {"pca_95"}    # components, not a column subset
EST_ORDER = ["ols", "ridge", "lasso", "enet", "cart"]


def bits(hex_support: str, p: int) -> np.ndarray:
    """Hex bitmask -> binary indicator vector of length p, LSB = column 0."""
    v = int(hex_support, 16)
    return np.array([(v >> j) & 1 for j in range(p)], dtype=float)


def load():
    """(estimator, dataset, arm) -> list of per-fold indicator vectors."""
    sup = defaultdict(list)
    rmse = defaultdict(list)
    for f in sorted(SHARDS.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("perm", 0) != 0 or r.get("fit_scope") != "train":
                    continue
                if r.get("tie_break") != "position":
                    continue
                key = (r["estimator"], r["dataset"], r["selection"])
                rmse[key].append(r["rmse"])
                s = r.get("support")
                if s:
                    sup[key].append(bits(s, r["n_features_in"]))
    return sup, rmse


def prunable():
    man = json.loads(MANIFEST.read_text())["datasets"]
    return {k for k, v in man.items() if v["actual"].get("n_pairs_above_80", 0) > 0}


def main() -> int:
    sup, rmse = load()
    if not sup:
        print("no support bitmasks found in the shards")
        return 1
    pr = prunable()

    phi, ragged = {}, []
    for key, vecs in sup.items():
        if len(vecs) < 2:
            continue
        widths = {v.size for v in vecs}
        if len(widths) > 1:
            # n_features_in is a property of the dataset, so this cannot happen
            # in a consistent run. If it does, the shards mix two versions of a
            # dataset and the stability index would be meaningless -- say so
            # loudly rather than averaging over incompatible vectors.
            ragged.append((key, sorted(widths)))
            continue
        phi[key] = nogueira_stability(np.vstack(vecs))

    if ragged:
        print("WARNING: inconsistent feature counts within a dataset -- these")
        print("cells are excluded and the corpus cache should be re-checked:")
        for key, widths in ragged[:10]:
            print(f"  {'|'.join(key)}: n_features_in = {widths}")
        print()

    arms = sorted({k[2] for k in phi})
    comparable = [a for a in arms
                  if a not in DEGENERATE and a not in INCOMMENSURABLE]

    print("=" * 78)
    print("SELECTION STABILITY  (Nogueira Phi, mean over prunable datasets)")
    print("=" * 78)
    print("  1.0 = every fold chose the same features; 0.0 = chance.")
    print(f"  {'arm':12s}" + "".join(f"{e:>10s}" for e in EST_ORDER) + "   note")
    print("  " + "-" * 74)

    # Restrict to the datasets that have EVERY arm for a given estimator.
    # Averaging each arm over whatever datasets happen to have it makes the
    # columns incomparable: a selector that ignores the estimator entirely
    # (vif, corr_*) must give identical numbers across estimator columns, and
    # if it does not, the corpus is varying underneath the comparison.
    complete = {}
    for e in EST_ORDER:
        ds = {d for d in pr if any((e, d, a) in phi for a in arms)}
        complete[e] = sorted(d for d in ds
                             if all((e, d, a) in phi for a in arms))

    table = {}
    for a in arms:
        row, line = {}, f"  {a:12s}"
        for e in EST_ORDER:
            vals = [phi[(e, d, a)] for d in complete[e] if (e, d, a) in phi]
            if vals:
                m = st.fmean(vals)
                row[e] = m
                line += f"{m:>10.3f}"
            else:
                line += f"{'--':>10s}"
        note = ""
        if a in DEGENERATE:
            note = "DEGENERATE - selects all, Phi undefined"
        elif a in INCOMMENSURABLE:
            note = "components, not a column subset"
        elif len({round(v, 6) for v in row.values()}) == 1 and len(row) > 1:
            note = "estimator-independent, as expected"
        table[a] = row
        print(line + "   " + note)

    print("\n  Averaged over datasets complete for every arm: "
          + ", ".join(f"{e}={len(complete[e])}" for e in EST_ORDER))
    print("  The control's 1.000 is an artefact of the index being undefined when")
    print("  nothing varies, not a finding.")
    pca = table.get("pca_95", {})
    if pca and all(v > 0.999 for v in pca.values()):
        print("  pca_95 also reaches 1.000, which means its mask is constant across")
        print("  folds -- the same artefact, not a stable selector.")
    print("  Both are excluded from every comparison below.")

    # The headline read: stability against harm, not stability alone.
    print("\n" + "=" * 78)
    print("Is stability bought at the price of accuracy?")
    print("=" * 78)
    print("  A selector is stable when its decisions are not close. Aggressive")
    print("  thresholds make them not close, so the same informative features are")
    print("  discarded in every fold -- reliably. Check the ordering below against")
    print("  the RMSE costs in Table 2 of the findings document before deciding")
    print("  which direction supports the interpretability defence.")
    for e in ("ridge",):
        ordered = sorted((v[e], a) for a, v in table.items()
                         if a in comparable and e in v)
        for phi_v, a in reversed(ordered):
            print(f"    {a:12s} Phi {phi_v:.3f}")

    # ------------------------------------------------ does stability track accuracy?
    print("\n" + "=" * 78)
    print("Does the more stable arm predict better?")
    print("=" * 78)
    for e in EST_ORDER:
        ds = sorted({k[1] for k in rmse if k[0] == e and k[1] in pr})
        full = [d for d in ds if all((e, d, a) in rmse for a in arms)]
        if len(full) < 5:
            print(f"  {e:6s} too few complete datasets")
            continue
        M = np.array([[st.fmean(rmse[(e, d, a)]) for a in arms] for d in full])
        avg_rank = np.apply_along_axis(stats.rankdata, 1, M).mean(axis=0)
        rank_of = dict(zip(arms, avg_rank))

        xs = [table[a][e] for a in comparable if e in table[a]]
        ys = [rank_of[a] for a in comparable if e in table[a]]
        if len(xs) < 4:
            continue
        rho, p = stats.spearmanr(xs, ys)
        # Lower rank is better, so a NEGATIVE rho means more stable -> better.
        verdict = ("more stable arms do predict better" if rho < -0.5 and p < 0.05
                   else "no reliable relationship")
        print(f"  {e:6s} Spearman(Phi, mean rank) = {rho:+.3f}  p = {p:.4f}   {verdict}")

    print("\n  Read with care: this correlates arms, of which there are only "
          f"{len(comparable)},")
    print("  so it is suggestive at best. It is not a test on datasets.")

    # -------------------------------------------- stability across tie-break rules
    print("\n" + "=" * 78)
    print("Stability of corr_080 across tie-break rules")
    print("=" * 78)
    tb = defaultdict(lambda: defaultdict(list))
    for f in sorted(SHARDS.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if (r.get("perm", 0) != 0 or r.get("selection") != "corr_080"
                        or r.get("fit_scope") != "train" or not r.get("support")):
                    continue
                tb[(r["dataset"], r["tie_break"])][r["estimator"]].append(
                    bits(r["support"], r["n_features_in"]))
    rules = sorted({k[1] for k in tb})
    if len(rules) > 1:
        print(f"  {'rule':20s}{'mean Phi':>10s}   (over datasets x estimators)")
        print("  " + "-" * 46)
        agg = {}
        for rule in rules:
            vals = []
            for (d, rl), by_est in tb.items():
                if rl != rule or d not in pr:
                    continue
                for vecs in by_est.values():
                    if len(vecs) >= 2:
                        vals.append(nogueira_stability(np.vstack(vecs)))
            if vals:
                agg[rule] = st.fmean(vals)
        for rule, v in sorted(agg.items(), key=lambda kv: -kv[1]):
            print(f"  {rule:20s}{v:>10.3f}")
        print("\n  CAUTION -- this measures agreement across FOLDS at one fixed")
        print("  column order, which is not the RQ5 question. 'position' is fully")
        print("  determined by the column order, so it is stable across folds by")
        print("  construction; a rule that depends on estimated correlations moves")
        print("  when the fold moves. A high score here says nothing about order")
        print("  dependence. The measurement that does is below.")
    else:
        print("  only one tie-break rule on disk; run the tiebreak stage first")

    # ------------------------------------------- stability ACROSS permutations
    print("\n" + "=" * 78)
    print("SELECTION STABILITY ACROSS COLUMN PERMUTATIONS  (the RQ5 measure)")
    print("=" * 78)
    print("  Each permutation contributes its modal fold subset; Phi is then")
    print("  computed over those. This is the chance-corrected version of the")
    print("  'distinct subsets' count, and unlike that count it is comparable")
    print("  across datasets with different numbers of features.")
    pp = defaultdict(lambda: defaultdict(list))
    for f in sorted(SHARDS.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("perm", 0) <= 0 or not r.get("support"):
                    continue
                pp[(r["dataset"], r["estimator"], r["tie_break"])][r["perm"]].append(
                    (r["support"], r["n_features_in"]))
    per_rule = defaultdict(list)
    rows = []
    for (ds, e, tb), perms in sorted(pp.items()):
        if len(perms) < 5:
            continue
        modal = []
        for entries in perms.values():
            counts = defaultdict(int)
            for sup_hex, p_in in entries:
                counts[(sup_hex, p_in)] += 1
            sup_hex, p_in = max(counts, key=counts.get)
            modal.append(bits(sup_hex, p_in))
        if len({v.size for v in modal}) != 1:
            continue
        val = nogueira_stability(np.vstack(modal))
        rows.append((ds, e, tb, len(perms), val))
        per_rule[tb].append(val)
    if rows:
        print(f"\n  {'dataset':18s}{'est':7s}{'rule':18s}{'perms':>7s}{'Phi':>8s}")
        print("  " + "-" * 60)
        for ds, e, tb, np_, val in sorted(rows, key=lambda r: (r[2], r[4])):
            print(f"  {ds:18s}{e:7s}{tb:18s}{np_:>7d}{val:>8.3f}")
        print()
        for tb, vals in sorted(per_rule.items()):
            print(f"  {tb:18s} mean Phi over permutations {st.fmean(vals):.3f}  "
                  f"({len(vals)} cells)")
        print("\n  An order-invariant rule must score exactly 1.000 here: every")
        print("  column order yields the same subset. Any shortfall for 'position'")
        print("  is order dependence measured directly on the selection sets,")
        print("  independent of any effect on error.")
    else:
        print("  no permutation results with support masks found")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "phi_by_arm_estimator": table,
        "comparable_arms": comparable,
        "degenerate_arms": sorted(DEGENERATE),
        "excluded_arms": sorted(INCOMMENSURABLE),
        "phi_per_cell": {f"{e}|{d}|{a}": v for (e, d, a), v in phi.items()},
        "complete_datasets": {e: complete[e] for e in EST_ORDER},
        "permutation_phi": [
            {"dataset": d, "estimator": e, "tie_break": t, "n_perms": n_, "phi": v}
            for d, e, t, n_, v in rows
        ],
    }, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
