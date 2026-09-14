"""Calibrate the study to the local machine and confirm the fast paths are live.

Run this immediately after setup:

    python scripts/benchmark.py

It reports whether Numba compiled, whether BLAS threading is pinned, how fast
one work unit runs here, and -- from that -- an estimate of the full grid's wall
time at the chosen worker count.  That estimate is what decides between the full
10x5 protocol and a reduced 5x5 one.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colreg.estimators import LassoRegression, RidgeRegression
from colreg.estimators._cd import HAVE_NUMBA, enet_path, warm_up
from colreg.experiment import ESTIMATORS, SELECTION_ARMS, evaluate_arm


def problem(n=2000, p=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, p // 2:] = X[:, : p // 2] + rng.normal(scale=0.05, size=(n, p - p // 2))
    y = X @ rng.normal(size=p) + rng.normal(size=n)
    return X, y


def main():
    print(f"{platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"logical CPUs: {os.cpu_count()}")
    print(f"numba: {'available' if HAVE_NUMBA else 'MISSING -- expect ~250x slower'}")

    try:
        import numba
        print(f"  numba {numba.__version__}")
    except ImportError:
        pass

    if HAVE_NUMBA:
        t = time.perf_counter()
        warm_up()
        print(f"JIT warm-up (once per machine, cached after): {time.perf_counter()-t:.2f}s")

    X, y = problem()
    Xc, yc = X - X.mean(0), y - y.mean()

    print("\n--- single fits, n=2000 p=60 -------------------------------------")
    t = time.perf_counter(); RidgeRegression(alpha=1.0).fit(X, y)
    print(f"ridge (closed form)     {time.perf_counter()-t:.4f}s")
    t = time.perf_counter(); m = LassoRegression(alpha=0.01, tol=1e-8).fit(X, y)
    dt = time.perf_counter() - t
    print(f"lasso (Gram CD)         {dt:.4f}s   [{m.n_iter_} iterations]")
    print("                        reference: 0.0067s on the dev machine,")
    print(f"                        1.70s before optimisation ({1.70/dt:.0f}x here)")

    alphas = np.logspace(-4, 2, 25)[::-1]
    t = time.perf_counter(); enet_path(Xc, yc, alphas, 1.0)
    print(f"25-alpha warm path      {time.perf_counter()-t:.4f}s")

    print("\n--- one work unit (2x5 nested CV, reduced grid) ------------------")
    from colreg.experiment import arms as A
    A.PARAM_GRIDS["lasso"] = {"alpha": np.logspace(-3, 1, 5)}
    sel = next(a for a in SELECTION_ARMS if a.key == "corr_080")
    est = next(a for a in ESTIMATORS if a.key == "lasso")
    t = time.perf_counter()
    rows = list(evaluate_arm(X, y, "bench", sel, est, n_repeats=2, n_inner=3))
    unit_2x5 = time.perf_counter() - t
    print(f"{len(rows)} folds in {unit_2x5:.2f}s")

    # Scale to the real protocol: 10 repeats not 2, 5 inner folds not 3,
    # 25 alphas not 5.  Warm starting means the alpha grid scales sub-linearly;
    # 0.4 is the empirically observed factor, and is deliberately pessimistic.
    unit_full = unit_2x5 * (10 / 2) * (5 / 3) * (25 / 5) ** 0.4
    n_units_main = 30 * 9 * 5
    for jobs in (4, 6, 8):
        hours = unit_full * n_units_main / jobs / 3600
        print(f"  est. main grid ({n_units_main} units) @ {jobs} workers: {hours:.1f} h")

    print("\nNote: OLS and ridge units are far cheaper than lasso/enet, so the")
    print("above over-estimates the total. Run --stage main --limit 20 first")
    print("and read the reported s/unit for a real number.")

    print("\n--- GPU ----------------------------------------------------------")
    print("Not used, and correctly so. Coordinate descent on p<=120 columns is a")
    print("sequential, branch-heavy, tiny-matrix workload: a CUDA kernel launch")
    print("(~5-10 us) costs more than an entire coordinate sweep. The RTX would")
    print("only earn its keep alongside GBM baselines (XGBoost/LightGBM), and")
    print("even then the CPU versions finish these dataset sizes in seconds.")
    print("CPU cores are the resource that matters here.")


if __name__ == "__main__":
    main()
