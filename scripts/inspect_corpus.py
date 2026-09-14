"""Print Table 1 from the manifest, and answer the stratification question.

    python scripts/inspect_corpus.py
    python scripts/inspect_corpus.py --markdown > table1.md

Two different things are being measured here and the paper must not conflate
them:

*   The **condition index** describes GLOBAL conditioning -- how close the whole
    design matrix is to rank deficiency. It is the standard collinearity
    diagnostic and the thing Belsley's 30/100 thresholds refer to.

*   **max |r|** and the count of pairs above the threshold describe what the
    PRUNING RULE ACTUALLY SEES. The rule is pairwise: it never looks at the
    spectrum, only at one correlation at a time.

These come apart, and where they do is exactly where the study is interesting.
A dataset can have a low condition index while carrying several pairs above
0.8 (a few tightly coupled features inside an otherwise well-behaved design);
another can have a high index with no pair above 0.8 at all, because its
collinearity is multi-way and no two columns are individually redundant.

The last group is the sharpest control in the whole study: there the pruning
arm removes nothing, so it is PROVABLY IDENTICAL to the no-selection arm, and
any difference measured between them is pure noise. That gives a calibration
baseline for reading every other comparison -- which is why those datasets
belong in the corpus rather than being filtered out for being "easy".
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "corpus_manifest.json"


def fmt(v, spec=".1f", inf="inf"):
    if v is None:
        return "-"
    if isinstance(v, float) and not math.isfinite(v):
        return inf
    return format(v, spec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true", help="emit a Markdown table")
    ap.add_argument("--tau", type=float, default=0.8,
                    help="pruning threshold used for the 'acts?' column")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"{MANIFEST} not found -- run scripts/verify_corpus.py first")
        return 1

    data = json.loads(MANIFEST.read_text())["datasets"]
    rows = []
    for key, e in data.items():
        a = e["actual"]
        rows.append({
            "key": key,
            "n": a["n"], "p": a["p"], "n_over_p": a["n_over_p"],
            "index": a.get("condition_index", float("nan")),
            "index_trim": a.get("condition_index_trimmed", float("nan")),
            "max_r": a["max_abs_r"],
            "pairs": a["n_pairs_above_80"],
            "max_vif": a["max_vif"],
            "rank_def": a.get("rank_deficiency", 0),
            "stratum": a["stratum"],
        })
    rows.sort(key=lambda r: (-r["index"] if math.isfinite(r["index"]) else -float("inf")))

    hdr = ["dataset", "n", "p", "n/p", "cond.idx", "max|r|", "pairs>.8", "maxVIF", "rank def", "stratum", "prunes?"]
    if args.markdown:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join(["---"] * len(hdr)) + "|")
    else:
        print(f"{hdr[0]:18s}{hdr[1]:>7s}{hdr[2]:>5s}{hdr[3]:>7s}{hdr[4]:>10s}"
              f"{hdr[5]:>8s}{hdr[6]:>10s}{hdr[7]:>12s}{hdr[8]:>10s}  {hdr[9]:8s}{hdr[10]}")
        print("-" * 108)

    for r in rows:
        acts = "yes" if r["pairs"] > 0 else "NO"
        cells = [r["key"], str(r["n"]), str(r["p"]), fmt(r["n_over_p"]),
                 fmt(r["index"]), fmt(r["max_r"], ".3f"), str(r["pairs"]),
                 fmt(r["max_vif"], ".1f"), str(r["rank_def"]), r["stratum"], acts]
        if args.markdown:
            print("| " + " | ".join(cells) + " |")
        else:
            print(f"{cells[0]:18s}{cells[1]:>7s}{cells[2]:>5s}{cells[3]:>7s}{cells[4]:>10s}"
                  f"{cells[5]:>8s}{cells[6]:>10s}{cells[7]:>12s}{cells[8]:>10s}  {cells[9]:8s}{cells[10]}")

    if args.markdown:
        return 0

    # ---------------------------------------------------------------- summary
    acts = [r for r in rows if r["pairs"] > 0]
    inert = [r for r in rows if r["pairs"] == 0]
    print()
    print(f"{len(rows)} datasets: pruning at tau={args.tau} removes something on "
          f"{len(acts)}, nothing on {len(inert)}")

    if inert:
        print("\n  Inert (pruning arm == control arm, by construction):")
        for r in inert:
            print(f"    {r['key']:18s} index={fmt(r['index']):>8s}  max|r|={r['max_r']:.3f}  "
                  f"maxVIF={fmt(r['max_vif']):>8s}  [{r['stratum']}]")
        print("  These are the calibration baseline: any RMSE difference measured")
        print("  between the two arms here is noise, and sets the floor for reading")
        print("  every other comparison in the paper.")

    print("\n  Where the two diagnostics disagree:")
    disagree = [r for r in rows
                if (r["stratum"] == "low" and r["pairs"] > 0)
                or (r["stratum"] == "high" and r["pairs"] == 0)]
    if not disagree:
        print("    none -- global conditioning and pairwise redundancy agree throughout")
    for r in disagree:
        why = ("low index but pairwise redundancy present"
               if r["stratum"] == "low" else
               "high index but no single pair above threshold (multi-way)")
        print(f"    {r['key']:18s} index={fmt(r['index']):>8s} pairs={r['pairs']:<3d} {why}")

    by = {}
    for r in rows:
        by.setdefault(r["stratum"], []).append(r)
    print("\n  Cell counts if stratifying on condition index (Belsley 30/100):")
    for s in ("low", "medium", "high"):
        got = by.get(s, [])
        n_acts = sum(1 for r in got if r["pairs"] > 0)
        print(f"    {s:7s} {len(got):3d} datasets ({n_acts} of them prunable)")

    print("\n  Cell counts if stratifying on pairwise redundancy instead:")
    print(f"    prunable      {len(acts):3d}")
    print(f"    inert         {len(inert):3d}")

    thin = [s for s in ("low", "medium", "high") if len(by.get(s, [])) < 8]
    if thin:
        print(f"\n  WARNING: {', '.join(thin)} stratum has fewer than 8 datasets.")
        print("  A Friedman test across that few has very little power; report the")
        print("  continuous log(condition index) moderator as the primary RQ2")
        print("  analysis and treat the per-stratum tables as descriptive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
