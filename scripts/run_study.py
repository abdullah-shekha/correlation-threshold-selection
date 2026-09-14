"""Full-grid runner: parallel, checkpointed, resumable.

Usage (from the project root, venv active):

    python scripts/run_study.py --jobs 6
    python scripts/run_study.py --jobs 6 --stage main      # RQ1-RQ3
    python scripts/run_study.py --jobs 6 --stage tiebreak  # RQ5
    python scripts/run_study.py --jobs 6 --stage leakage   # RQ4
    python scripts/run_study.py --collect                  # merge shards -> results.csv

Design notes that matter on a 16 GB / 8-thread machine:

*   **One shard file per work unit.**  A unit is one (dataset, selection,
    estimator, tie_break, fit_scope, perm) cell, and it writes
    ``results/shards/<key>.jsonl`` when it finishes.  Re-running skips any unit
    whose shard already exists, so a crashed or cancelled run resumes where it
    stopped.  A run that cannot be resumed is a run that gets done twice.

*   **Threads pinned to 1 inside workers.**  NumPy's BLAS will happily spawn 8
    threads per process; with 6 joblib workers that is 48 threads fighting over
    8 hardware threads, and it is *slower* than single-threaded.  The env vars
    below must be set before NumPy is imported, which is why they are at the top
    of the file.

*   **JIT warm-up in the parent.**  Numba compiles on first call (~1-2 s per
    signature).  ``cache=True`` writes the artifact to __pycache__, so warming
    up before forking means the workers load it instead of each recompiling.

*   **loky backend, not threads.**  The kernels release the GIL only partially;
    processes are the right unit here.  On Windows loky uses spawn, so the
    ``if __name__ == "__main__"`` guard below is mandatory, not stylistic.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colreg.data import characterise                       # noqa: E402
from colreg.estimators._cd import HAVE_NUMBA, warm_up      # noqa: E402
from colreg.experiment import ESTIMATORS, SELECTION_ARMS, evaluate_arm  # noqa: E402
from colreg.experiment.arms import LEAK_ARMS, TIE_BREAK_ARMS  # noqa: E402

SHARDS = ROOT / "results" / "shards"
MANIFEST = ROOT / "data" / "corpus_manifest.json"
EXCLUDED = ROOT / "data" / "excluded_datasets.json"
CACHE = ROOT / "data" / "cache"


# --------------------------------------------------------------------- loading

def load_dataset(key: str):
    """Load a prepared dataset from the local cache written by verify_corpus.py.

    Datasets are cached as .npz so the study never re-downloads mid-run, and so
    a reviewer re-running the package gets byte-identical inputs.
    """
    path = CACHE / f"{key}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python scripts/verify_corpus.py` first"
        )
    d = np.load(path)
    return d["X"], d["y"]


def available_datasets() -> list[str]:
    if not CACHE.exists():
        return []
    return sorted(p.stem for p in CACHE.glob("*.npz"))


# ----------------------------------------------------------------- work units

def unit_key(ds, sel, est, tie_break, fit_scope, perm) -> str:
    return f"{ds}__{sel}__{est}__{tie_break}__{fit_scope}__p{perm}"


def build_units(stage: str, datasets: list[str], prunable: list[str],
                n_perms: int = 100, n_control: int = 5):
    """Enumerate the work units for a stage.

    The sub-studies run only where they CAN show an effect, and the criterion is
    not the collinearity stratum.  A dataset is included in the tie-break and
    permutation stages iff it has at least one pair above the threshold, i.e.
    iff the pruning rule actually removes something.  Those two sets differ
    sharply: on this corpus 8 datasets sit in the "low" condition-index stratum
    yet still carry pairwise redundancy (auto_mpg has 6 such pairs at an index
    of only 11.7), while solar_flare has max VIF 153 and no pair above 0.8 at
    all.  Keying off the stratum would have run the sub-studies on 7 datasets
    and missed 11 of the 18 where the rule is live.
    """
    units = []
    if stage in ("main", "all"):
        for ds in datasets:
            for sel in SELECTION_ARMS:
                for est in ESTIMATORS:
                    units.append((ds, sel.key, est.key, "position", "train", 0))

    if stage in ("tiebreak", "all"):
        for ds in prunable:
            for tb in TIE_BREAK_ARMS:
                for est in ("ridge", "lasso"):
                    units.append((ds, "corr_080", est, tb, "train", 0))

    if stage in ("permutation", "all"):
        # The CV folds are seeded from the repeat index only, never from the
        # permutation, so every permutation sees IDENTICAL train/test splits.
        # The comparison is therefore paired: any RMSE difference between two
        # permutations comes from the different feature subsets, not from
        # resampling. That is what licenses a low --repeats here without
        # inflating the apparent spread.
        for ds in prunable:
            for est in ("ridge", "lasso"):
                for perm in range(1, n_perms + 1):
                    units.append((ds, "corr_080", est, "position", "train", perm))

        # NULL CONTROL. mean_corr (caret's rule) is order-invariant by
        # construction: it drops whichever member of a pair has the higher mean
        # absolute correlation, a quantity that does not depend on column order.
        # Over 50 permutations of a synthetic design it produced 1 distinct
        # subset where 'position' produced 8. Running a handful of permutations
        # under it gives a floor: whatever spread appears here is pure numerical
        # noise, and the 'position' spread must clear it to mean anything.
        for ds in prunable:
            for perm in range(1, min(n_control, n_perms) + 1):
                units.append((ds, "corr_080", "ridge", "mean_corr", "train", perm))

    if stage in ("leakage", "all"):
        for ds in datasets:
            for scope in LEAK_ARMS:
                if scope == "train":
                    continue                       # already covered by the main grid
                for est in ("ridge", "lasso", "ols"):
                    units.append((ds, "corr_080", est, "position", scope, 0))

    return units


def run_unit(unit, n_repeats, n_outer, n_inner):
    ds, sel_key, est_key, tie_break, fit_scope, perm = unit
    key = unit_key(*unit)
    out = SHARDS / f"{key}.jsonl"
    if out.exists():
        return key, 0, 0.0                          # already done

    sel = next(a for a in SELECTION_ARMS if a.key == sel_key)
    est = next(a for a in ESTIMATORS if a.key == est_key)
    X, y = load_dataset(ds)

    order = None
    if perm:
        # Permute columns for the RQ5 arm. The seed is the permutation index, so
        # the same order is reproducible and comparable across estimators.
        order = np.random.default_rng(perm).permutation(X.shape[1])
        X = X[:, order]

    t0 = time.perf_counter()
    rows = list(evaluate_arm(
        X, y, ds, sel, est,
        tie_break=tie_break, fit_scope=fit_scope, perm=perm,
        n_repeats=n_repeats, n_outer=n_outer, n_inner=n_inner,
    ))
    dt = time.perf_counter() - t0

    if order is not None:
        # The selector reported its support in PERMUTED column space. Subsets
        # from different permutations are only comparable once mapped back to
        # the original columns -- without this, counting "distinct subsets"
        # would count relabelings rather than genuinely different choices.
        for r in rows:
            r.support = _remap_support(r.support, order)

    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r)) + "\n")
    tmp.replace(out)                                # atomic: no half-written shards
    return key, len(rows), dt


def _remap_support(hex_mask: str, order) -> str:
    """Translate a support bitmask from permuted columns to original indices."""
    if not hex_mask:
        return hex_mask
    bits = int(hex_mask, 16)
    out = 0
    for j, orig in enumerate(order):
        if bits & (1 << j):
            out |= (1 << int(orig))
    return format(out, "x")


# ------------------------------------------------------------------- collect

def collect():
    import pandas as pd
    files = sorted(SHARDS.glob("*.jsonl"))
    if not files:
        print("no shards found")
        return 1
    frames = [pd.read_json(f, lines=True) for f in files]
    df = pd.concat(frames, ignore_index=True)
    out = ROOT / "results" / "results.csv"
    df.to_csv(out, index=False)
    print(f"{len(files)} shards, {len(df):,} rows -> {out}")
    print(df.groupby(["selection", "estimator"])["rmse"].mean().round(4).to_string())
    return 0


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="worker processes; leave 2 threads for the OS")
    ap.add_argument("--stage", default="main",
                    choices=["main", "tiebreak", "permutation", "leakage", "all"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="run only the first N units")
    ap.add_argument("--perms", type=int, default=100,
                    help="column permutations per dataset in the permutation stage")
    ap.add_argument("--control-perms", type=int, default=5, dest="control_perms",
                    help="permutations run under the order-invariant mean_corr control")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--skip", default="", help=(
        "comma-separated units to leave out of THIS run, each either a dataset "
        "('wine_quality') or a dataset:estimator pair ('superconduct:cart'). "
        "Use for units that are too expensive to run now; use "
        "data/excluded_datasets.json for datasets excluded from the study."))
    args = ap.parse_args()

    SHARDS.mkdir(parents=True, exist_ok=True)
    if args.collect:
        return collect()

    datasets = available_datasets()
    if not datasets:
        print("No cached datasets. Run: python scripts/verify_corpus.py")
        return 1

    # Datasets excluded from the STUDY, with a recorded reason, as distinct
    # from units skipped in one run for cost. The file is the single source of
    # truth: the runner, the analysis scripts and the manuscript's corpus count
    # all read it, so a dataset cannot be dropped from the experiment without
    # the reason travelling with it.
    excluded = {}
    if EXCLUDED.exists():
        excluded = json.loads(EXCLUDED.read_text()).get("datasets", {})
    if excluded:
        dropped = [d for d in datasets if d in excluded]
        datasets = [d for d in datasets if d not in excluded]
        for d in dropped:
            print(f"excluded from the study: {d} -- {excluded[d]}")

    # Units skipped for this run only. Nothing is recorded, because nothing is
    # being claimed: an unrun unit shows up as missing coverage, which is what
    # it is.
    skip_ds, skip_pairs = set(), set()
    for tok in (t.strip() for t in args.skip.split(",") if t.strip()):
        if ":" in tok:
            skip_pairs.add(tuple(tok.split(":", 1)))
        else:
            skip_ds.add(tok)

    prunable = []
    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text())["datasets"]
        prunable = [k for k in datasets
                    if man.get(k, {}).get("actual", {}).get("n_pairs_above_80", 0) > 0]
    if not prunable:
        # Measure directly, so the runner works before the manifest exists
        # rather than silently running an empty sub-study.
        for k in datasets:
            X, _ = load_dataset(k)
            if characterise(X)["n_pairs_above_80"] > 0:
                prunable.append(k)

    units = build_units(args.stage, datasets, prunable,
                        n_perms=args.perms, n_control=args.control_perms)
    todo = [u for u in units if not (SHARDS / f"{unit_key(*u)}.jsonl").exists()]
    if skip_ds or skip_pairs:
        before = len(todo)
        # unit = (dataset, selection, estimator, tie_break, fit_scope, perm)
        todo = [u for u in todo
                if u[0] not in skip_ds and (u[0], u[2]) not in skip_pairs]
        print(f"--skip removed {before - len(todo)} of {before} pending units")
    if args.limit:
        todo = todo[: args.limit]

    print(f"numba: {'yes' if HAVE_NUMBA else 'NO -- expect ~250x slower'}")
    print(f"datasets: {len(datasets)}  (prunable at tau=0.8: {len(prunable)})")
    print(f"stage {args.stage}: {len(units):,} units, {len(todo):,} still to do")
    print(f"workers: {args.jobs}\n")
    if not todo:
        print("nothing to do -- run with --collect to merge results")
        return 0

    warm_up()

    from joblib import Parallel, delayed
    t0 = time.perf_counter()
    results = Parallel(n_jobs=args.jobs, backend="loky", verbose=5)(
        delayed(run_unit)(u, args.repeats, args.outer, args.inner) for u in todo
    )
    total = time.perf_counter() - t0
    rows = sum(r[1] for r in results)
    print(f"\n{len(results)} units, {rows:,} rows in {total/60:.1f} min "
          f"({total/max(len(results),1):.1f} s/unit)")
    print("next: python scripts/run_study.py --collect")
    return 0


if __name__ == "__main__":            # REQUIRED on Windows (spawn, not fork)
    raise SystemExit(main())
