"""Resolve every dataset in the registry and write a citable manifest.

Run this FIRST, before any experiment. It is the step that separates a study a
reviewer can check from one they cannot.

For each entry it records the resolved source, the actual (n, p) after the
stated preprocessing, which column was used as the target, how many rows were
dropped for missingness, and the measured collinearity characterisation. The
manifest is what the paper cites; ``registry.py`` is only a wish list until this
has run.

Usage:

    python scripts/verify_corpus.py                 # everything
    python scripts/verify_corpus.py --only wine_red # retry one entry
    python scripts/verify_corpus.py --list          # show the registry, fetch nothing

Design note: this script assumes as little as possible about the shape of what
the repository hands back. Target columns in particular are RESOLVED at runtime
and reported, never trusted from the registry -- UCI is inconsistent about
whether a dataset declares a target at all, and a hard-coded column name that
silently falls back to the wrong column would corrupt every downstream result
without ever raising.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colreg.data import CORPUS, characterise   # noqa: E402

OUT = ROOT / "data" / "corpus_manifest.json"
CACHE = ROOT / "data" / "cache"


# --------------------------------------------------------------- target choice

def resolve_target(features: pd.DataFrame, targets, wanted: str):
    """Decide which column is y, and say how the decision was made.

    Returns ``(y_series, features_without_y, how)``.

    UCI datasets fall into three shapes and the fetcher must handle all of them:

    1. ``targets`` is a DataFrame with the wanted column   -> use it by name
       (Energy Efficiency needs this: one design matrix, two targets Y1/Y2).
    2. ``targets`` is a DataFrame without that name        -> use its first
       column, and record that the registry name did not match.
    3. ``targets`` is None or empty                        -> the target lives
       in ``features``; pull it out by name, else take the last column.

    Case 3 is what produced "'NoneType' object has no attribute 'columns'":
    several datasets declare no target at all.
    """
    has_targets = targets is not None and getattr(targets, "empty", True) is False

    # 1. The registry's name, in the declared targets.
    if has_targets and wanted in targets.columns:
        col = targets.loc[:, wanted]
        if isinstance(col, pd.DataFrame):        # duplicate labels
            col = col.iloc[:, 0]
        return col, features, f"targets['{wanted}']"

    # 2. The registry's name, among the features.  This is checked BEFORE
    #    falling back to an unmatched target column, because the registry names
    #    the variable we actually intend to predict.  UCI's Automobile dataset
    #    declares 'symboling' (an insurance risk rating) as its target while
    #    'price' sits in the features -- taking the declared target there would
    #    silently change what the study is modelling.
    if wanted in features.columns:
        col = features.loc[:, wanted]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return col, features.drop(columns=[wanted]), f"features['{wanted}']"

    # 3. Whatever the repository declares, positionally.  .iloc, never [], so
    #    duplicate column labels cannot hand back a 2-D frame -- that is what
    #    produced "Data must be 1-dimensional, got ndarray of shape (536, 2)".
    if has_targets:
        name = targets.columns[0]
        return targets.iloc[:, 0], features, f"targets['{name}'] (registry said '{wanted}')"

    # 4. Last resort: the final feature column.
    name = features.columns[-1]
    return (
        features.iloc[:, -1],
        features.iloc[:, :-1],
        f"features['{name}'] (last column; registry said '{wanted}')",
    )


def prepare(features: pd.DataFrame, y) -> tuple:
    """One-hot low-cardinality categoricals, drop the rest, listwise-delete.

    Stated once and applied identically to every dataset. Per-dataset cleaning
    decisions are how benchmark corpora become unreproducible; a dataset that
    needs special handling does not belong in the corpus.
    """
    X = features.copy()
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    high_card = [c for c in cat if X[c].nunique(dropna=True) > 10]
    onehot = [c for c in cat if c not in high_card]

    X = X.drop(columns=high_card)
    if onehot:
        X = pd.get_dummies(X, columns=onehot, drop_first=True)

    frame = X.assign(__y__=pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce"))
    n_before = len(frame)
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("no complete rows after listwise deletion")

    Xdf = frame.drop(columns="__y__")
    yv = frame["__y__"].to_numpy(dtype=float)

    # Drop columns that carry no information. A zero-variance column is not a
    # feature: it inflates p, and it makes the correlation matrix singular, so
    # the condition number comes back as inf and the dataset is mis-assigned to
    # the "high collinearity" stratum. On a synthetic check, one constant column
    # moved a genuinely well-conditioned matrix from kappa=1.2 to kappa=inf.
    # These arise naturally here -- an all-zero dummy after one-hot encoding, or
    # a column that becomes constant once incomplete rows are deleted.
    variances = Xdf.var(axis=0, ddof=0)
    constant = [c for c in Xdf.columns if not np.isfinite(variances[c]) or variances[c] == 0]
    if constant:
        Xdf = Xdf.drop(columns=constant)

    # Exact duplicate columns are an exact linear dependency: same effect on the
    # condition number, and the pruning rule would see r = 1.0 between them.
    dupes = [c for c in Xdf.columns[Xdf.T.duplicated().to_numpy()]]
    if dupes:
        Xdf = Xdf.loc[:, ~Xdf.T.duplicated().to_numpy()]

    if Xdf.shape[1] == 0:
        raise ValueError("no informative columns remain after cleaning")

    info = {
        "rows_dropped_na": int(n_before - len(frame)),
        "onehot_encoded": onehot,
        "dropped_high_cardinality": high_card,
        "dropped_constant": [str(c) for c in constant],
        "dropped_duplicate": [str(c) for c in dupes],
    }
    return Xdf.to_numpy(dtype=float), yv, info


# ---------------------------------------------------------------------- fetch

def fetch(spec) -> tuple:
    """Return ``(X, y, provenance)``.  Exactly three values, always.

    The previous version built this as ``_prepare(...) + (prov,)`` -- a 3-tuple
    plus a 1-tuple, i.e. four values into a three-value unpack. That single line
    is what produced 22 of the 28 failures; every UCI entry hit it regardless of
    whether the download itself worked.
    """
    if spec.source == "sklearn":
        from sklearn import datasets as skd
        bunch = getattr(skd, spec.ref)()
        X = np.asarray(bunch.data, dtype=float)
        y = np.asarray(bunch.target, dtype=float)
        return X, y, {"resolved": f"sklearn.datasets.{spec.ref}", "target_from": "bunch.target"}

    if spec.source == "uci":
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=int(spec.ref))
        y, feats, how = resolve_target(ds.data.features, ds.data.targets, spec.target)
        X, yv, info = prepare(feats, y)
        info.update({
            "resolved": f"uci:{spec.ref}",
            "uci_name": (ds.metadata or {}).get("name"),
            "target_from": how,
        })
        return X, yv, info

    if spec.source == "openml":
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(name=spec.ref, as_frame=True, parser="auto")
        X, yv, info = prepare(ds.data, ds.target)
        info.update({"resolved": f"openml:{spec.ref}", "target_from": "openml target"})
        return X, yv, info

    raise ValueError(f"unknown source '{spec.source}'")


# ----------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", metavar="KEY",
                    help="fetch only these registry keys")
    ap.add_argument("--list", action="store_true",
                    help="print the registry and exit without fetching")
    args = ap.parse_args()

    if args.list:
        for s in CORPUS:
            print(f"{s.key:20s} {s.source:8s} {s.ref:26s} target={s.target}")
        return 0

    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Merge into an existing manifest so --only tops up rather than truncates.
    manifest, failures = {}, []
    if OUT.exists():
        try:
            manifest = json.loads(OUT.read_text()).get("datasets", {})
        except json.JSONDecodeError:
            manifest = {}

    specs = CORPUS
    if args.only:
        keys = set(args.only)
        specs = [s for s in CORPUS if s.key in keys]
        missing = keys - {s.key for s in specs}
        for k in sorted(missing):
            print(f"  no registry entry named '{k}'")
        if not specs:
            return 1

    for spec in specs:
        try:
            X, y, prov = fetch(spec)
        except Exception as exc:                        # noqa: BLE001
            failures.append({"key": spec.key, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAIL  {spec.key:20s} {type(exc).__name__}: {str(exc)[:110]}")
            continue

        np.savez_compressed(CACHE / f"{spec.key}.npz", X=X, y=y)
        char = characterise(X, y)
        manifest[spec.key] = {
            "declared": {
                "name": spec.name, "source": spec.source, "ref": spec.ref,
                "n": spec.n, "p": spec.p,
                "expected_stratum": spec.expected_stratum, "note": spec.note,
            },
            "actual": char,
            "provenance": prov,
            "stratum_matches_expectation": char["stratum"] == spec.expected_stratum,
        }
        flag = "" if char["stratum"] == spec.expected_stratum else "   <-- stratum differs"
        ci = char["condition_index"]
        ci_s = "    inf" if ci == float("inf") else f"{ci:7.1f}"
        print(f"ok    {spec.key:20s} n={char['n']:6d} p={char['p']:4d} "
              f"cond_index={ci_s} {char['stratum']:6s}{flag}")
        if char["rank_deficiency"]:
            print(f"      rank {char['numerical_rank']}/{char['p']} "
                  f"({char['rank_deficiency']} exact dependenc"
                  f"{'y' if char['rank_deficiency'] == 1 else 'ies'}), "
                  f"trimmed index={char['condition_index_trimmed']:.1f}")
        if "registry said" in prov.get("target_from", ""):
            print(f"      target resolved as {prov['target_from']}")

    OUT.write_text(json.dumps({"datasets": manifest, "failures": failures}, indent=2))

    print(f"\n{len(manifest)} datasets cached, {len(failures)} failed this run")
    print(f"manifest -> {OUT}")

    by_stratum: dict[str, int] = {}
    for entry in manifest.values():
        s = entry["actual"]["stratum"]
        by_stratum[s] = by_stratum.get(s, 0) + 1
    print("strata:", ", ".join(f"{k}={v}" for k, v in sorted(by_stratum.items())) or "none")

    if failures:
        print("\nRetry individual entries after editing the registry, e.g.:")
        print(f"    python scripts/verify_corpus.py --only {failures[0]['key']}")
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
