"""Remove regenerable clutter from the project tree. Dry-run unless --apply.

    python scripts/clean.py            # show what would go
    python scripts/clean.py --apply    # actually delete

What this removes: compiled Python bytecode (*.pyc) and the pytest cache.
Both are rebuilt automatically on the next run, so deleting them costs a few
seconds of import time and nothing else.

It does NOT remove numba's compiled kernels, which are the .nbi/.nbc files
sitting in colreg/estimators/__pycache__ alongside the bytecode. @njit(cache=True)
writes its cache next to the SOURCE file, not into the interpreter's tree, so
"skip .venv" does not protect it -- an earlier version of this script claimed
otherwise and was wrong. Those two files are the compiled _cd_gram and
_cd_naive kernels; deleting them buys back 138 KB and costs a JIT recompile
on the next run, which is a bad trade in both directions.

What this REFUSES to touch, and why it is hard-coded rather than left to a
flag:

    results/shards/   ~2,500 files, ~37 MB, and about twenty hours of wall
                      clock across the tie-break, permutation and main runs.
                      Every number in the findings document is derived from
                      these. They cannot be regenerated in an afternoon and
                      two of the datasets could not be regenerated at all
                      without re-running CART units that were abandoned for
                      cost. This is the study.

    data/cache/       25 .npz files, 9.3 MB. Nominally regenerable from UCI,
                      but a re-download is a fresh snapshot: if a repository
                      revises a file, the corpus manifest -- which records
                      row counts, dropped rows and measured collinearity --
                      stops describing the data the results came from. The
                      cache is what makes the manifest checkable.

    .venv/            The environment everything runs in.

.venv is excluded from the sweep as well: clearing site-packages/__pycache__
would force every dependency to recompile bytecode on the next import for no
benefit.

Be warned that the total recovered is trivial -- of order 100 KB against a
project whose results directory alone is 37 MB. Run it to keep the tree tidy,
not to free space.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Never walk into these, and never delete anything inside them.
PROTECTED = {"results", "data", ".venv", ".git"}


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def find_targets() -> list[Path]:
    targets: list[Path] = []

    def walk(d: Path) -> None:
        for child in sorted(d.iterdir()):
            if not child.is_dir():
                continue
            rel = child.relative_to(ROOT)
            if rel.parts[0] in PROTECTED:
                continue
            if child.name == ".pytest_cache":
                targets.append(child)
                continue
            if child.name == "__pycache__":
                # Delete the bytecode file by file, never the directory: numba
                # keeps its compiled kernels (.nbi index, .nbc code) in here
                # too, and those are worth minutes of compile time.
                targets.extend(sorted(child.glob("*.pyc")))
                continue
            walk(child)

    walk(ROOT)

    # A zero-byte wheel left behind by an interrupted pip install. Harmless,
    # but it is not a package and nothing reads it.
    for whl in (ROOT / ".venv" / "Lib" / "site-packages").glob("*.whl"):
        try:
            if whl.stat().st_size == 0:
                targets.append(whl)
        except OSError:
            pass

    # An empty output directory that was never written to.
    figures = ROOT / "results" / "figures"
    if figures.is_dir() and not any(figures.iterdir()):
        targets.append(figures)

    return targets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="delete; without it, only report")
    args = ap.parse_args()

    targets = find_targets()
    if not targets:
        print("Nothing to clean.")
        return 0

    total = 0
    print(f"{'':2s}{'bytes':>12s}  path")
    print("  " + "-" * 62)
    for t in targets:
        n = size_of(t)
        total += n
        print(f"  {n:>12,d}  {t.relative_to(ROOT).as_posix()}")
    print("  " + "-" * 62)
    print(f"  {total:>12,d}  total ({total / 1e6:.2f} MB)")

    if not args.apply:
        print("\nDry run. Re-run with --apply to delete.")
        return 0

    removed = 0
    for t in targets:
        try:
            if t.is_dir():
                shutil.rmtree(t)
            else:
                t.unlink()
            removed += 1
        except OSError as exc:
            print(f"  could not remove {t}: {exc}", file=sys.stderr)
    print(f"\nRemoved {removed} of {len(targets)} items.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
