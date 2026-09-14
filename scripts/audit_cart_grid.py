"""Does the inner CV ever choose max_depth=None? Decide before changing the grid.

    python scripts/audit_cart_grid.py

max_depth=None is one of six depth levels in the CART tuning grid. On this
implementation it costs roughly 13x more per fit than a depth-capped tree
(2.83s vs 0.22s at n=4000, p=20), and a single CART unit on superconduct works
out at about ten hours. It is why the main grid stalls on the large datasets.

Removing it is the obvious fix, but it is only free if the inner loop never
selected it. Every completed result records the winning hyper-parameters, so
that is answerable from the shards already on disk rather than by assumption:

  * If max_depth=None is NEVER selected, dropping it cannot change any tuned
    choice. The 1,033 completed units stay valid exactly as they are, and only
    the 92 outstanding units need to run.

  * If it IS sometimes selected, dropping it changes the experiment, and every
    CART unit has to be re-run under the new grid for the results to be
    comparable. That is 225 units rather than 92 -- worth knowing in advance.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "results" / "shards"


def main() -> int:
    if not SHARDS.exists():
        print("no results/shards found")
        return 1

    depth_counts = Counter()
    leaf_counts = Counter()
    by_dataset = defaultdict(Counter)
    n_rows = 0

    for f in SHARDS.glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("estimator") != "cart":
                    continue
                n_rows += 1
                try:
                    bp = json.loads(r.get("best_params") or "{}")
                except json.JSONDecodeError:
                    continue
                d = bp.get("max_depth", "missing")
                key = "None" if d is None else str(d)
                depth_counts[key] += 1
                leaf_counts[str(bp.get("min_samples_leaf", "?"))] += 1
                by_dataset[r["dataset"]][key] += 1

    if not n_rows:
        print("no CART results found in the shards")
        return 1

    print(f"CART fold results examined: {n_rows:,}")
    print("\nSelected max_depth, over all completed CART folds")
    print("-" * 52)
    total = sum(depth_counts.values())
    for k in sorted(depth_counts, key=lambda x: (x == "None", x)):
        c = depth_counts[k]
        bar = "#" * int(40 * c / total)
        print(f"  {k:>6s}  {c:>7,d}  {100*c/total:5.1f}%  {bar}")

    print("\nSelected min_samples_leaf")
    print("-" * 52)
    for k, c in sorted(leaf_counts.items()):
        print(f"  {k:>6s}  {c:>7,d}  {100*c/sum(leaf_counts.values()):5.1f}%")

    none_n = depth_counts.get("None", 0)
    print("\n" + "=" * 52)
    print("DECISION")
    print("=" * 52)
    if none_n == 0:
        print("  max_depth=None was NEVER selected in any completed fold.")
        print("  Dropping it from the grid cannot change a single tuned choice,")
        print("  so the completed units remain valid unchanged. Re-run only the")
        print("  92 outstanding units.")
    else:
        share = 100 * none_n / total
        print(f"  max_depth=None was selected in {none_n:,} folds ({share:.1f}%).")
        print("  Dropping it WOULD change the experiment, so every CART unit must")
        print("  be re-run under the new grid -- 225 units, not 92 -- for the")
        print("  completed and new results to be comparable.")
        worst = sorted(by_dataset.items(),
                       key=lambda kv: -kv[1].get("None", 0))[:5]
        print("\n  Datasets where it was chosen most often:")
        for ds, c in worst:
            if c.get("None"):
                print(f"    {ds:18s} {c['None']:>5,d} folds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
