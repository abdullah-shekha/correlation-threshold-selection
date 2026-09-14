# correlation-threshold-selection

Replication package for a benchmark study of correlation-threshold feature
selection in penalised regression.

Applied work routinely drops one predictor from every pair whose absolute
correlation exceeds a threshold, almost always 0.8. This study asks two
questions about that practice and answers both with a controlled benchmark over
24 public regression datasets.

**Does it improve predictive accuracy?** No. Across nine selection arms and five
estimators, no arm beat doing nothing on any estimator. The control took the
best mean rank on four of the five. Six of the eight pruning arms were
significantly worse on every linear estimator after Holm correction, and the
worst raised median error by 11.68%. Harm rose steadily as the threshold
tightened.

**Is the result reproducible?** Not where the rule acts. Permuting the columns
of the input moves test error by a median of 3.46% and by as much as 17.31%,
with the data, the estimator and the folds held identical. Chance-corrected
selection stability across permutations is 0.479, against 0.981 for an
order-invariant control. The cause is the elimination loop rather than the rule
used to break ties: an early removal cancels comparisons that would otherwise
have happened, so the set of comparisons performed depends on the column order
of the file.

The study produced **80,100 fold-level results**, all of which are in this
repository.

## Reproducing the paper without re-running anything

The fold-level results are included, so every table and figure can be
regenerated in minutes rather than the day the grid takes.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
tar -xzf results/shards.tar.gz
.venv/bin/python scripts/verify_findings.py
```

`verify_findings.py` re-derives every table in the manuscript directly from the
shards, using logic written independently of the analysis code, and diffs the
result against the published values. It should end with **"All documented values
reproduce from the shards."**

Then:

```bash
.venv/bin/python scripts/analyse_main.py        # ranks, Wilcoxon-Holm
.venv/bin/python scripts/analyse_stability.py   # selection stability
.venv/bin/python scripts/make_figures.py        # all figures
```

On Windows use `.venv\Scripts\python.exe`, or run `setup.ps1` to build the
environment and run the tests in one step.

## Running the study from scratch

Only needed to rebuild the results rather than check them. Budget about a day on
12 cores.

```bash
python scripts/verify_corpus.py                                  # 10-30 min
python scripts/pilot.py                                          # 2 min
python scripts/run_study.py --stage main --jobs 12 --skip superconduct:cart
python scripts/run_study.py --stage tiebreak --jobs 12           # ~3 hours
python scripts/run_study.py --stage permutation --jobs 12        # ~8 hours
```

`verify_corpus.py` resolves each dataset from its source and writes
`data/cache/*.npz` alongside `data/corpus_manifest.json`, which records the
resolved source, the target column and why it was chosen, rows dropped for
missing values, the categorical encoding, and every collinearity measure.

Every stage is resumable. Each writes one shard per cell and skips cells whose
shard already exists, so an interrupted run continues from where it stopped.

`--skip superconduct:cart` excludes one cell of the grid on cost grounds. An
unconstrained tree costs about twelve times more per fit than one capped at the
deepest grid level, and the inner loop selects the unconstrained level in 34.1%
of folds, so a single unit on the largest design matrix (21,263 by 81) runs for
about ten hours. The manuscript reports this exclusion and shows that it does
not favour the conclusion.

## What is here

```
colreg/
  selection/threshold.py   the object of study: tau, tie_break, fit_scope
  estimators/_cd.py        Numba coordinate descent, Gram-matrix formulation
  estimators/linear.py     OLS, ridge, lasso, elastic net
  estimators/tree.py       exact-split CART regression tree
  experiment/harness.py    nested repeated cross-validation
  experiment/arms.py       the nine arms, five estimators, tuning grids
  data/registry.py         the 24-dataset corpus
  data/characterise.py     collinearity measurement
  analysis/stats.py        Friedman, Nemenyi, Wilcoxon-Holm, CD diagrams
  analysis/metrics.py      RMSE, MAE, R2, Nogueira stability

scripts/                   runners, analyses, audits, figure generation
tests/test_parity.py       39 parity cases against scikit-learn

data/
  cache/*.npz              the 24 resolved design matrices
  corpus_manifest.json     how each dataset was resolved and characterised

results/
  shards.tar.gz            80,100 fold-level results (~2,400 files, extract first)
  stability.json           selection stability per arm and estimator
  figures/                 the published figures
```

The estimators are implemented directly in NumPy so that the selection step can
be instrumented and the surviving feature set recorded for every fold. Nothing
under `colreg/` imports scikit-learn. It is a test dependency only, used as the
reference for 39 parity cases across three synthetic designs: coefficients and
predictions agree to 1e-6 for OLS and ridge, coefficients to 1e-5 for lasso and
elastic net, and tree predictions to 1e-8. That separation is what makes the
parity claim mean anything.

## Requirements

Python 3.11 or 3.12. Numba lags CPython releases, and on 3.13 or later there may
be no wheel, in which case everything still runs but roughly 250 times slower.
Internet access is needed only by `verify_corpus.py`, to fetch datasets from the
UCI Machine Learning Repository.

## Data sources

The 24 datasets come from the UCI Machine Learning Repository and from the data
bundled with scikit-learn. The cached copies in `data/cache/` are numeric
arrays derived from those sources and carry the licence of each original.
`data/corpus_manifest.json` records the provenance of every one.

## Citing

If you use this code or the results, please cite the paper. See `CITATION.cff`,
which GitHub also renders as a "Cite this repository" link.

## Licence

MIT. See `LICENSE`.
