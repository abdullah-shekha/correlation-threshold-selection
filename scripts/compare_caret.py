"""Is caret's findCorrelation order-invariant? Port it and measure.

    python scripts/compare_caret.py

This study's 'mean_corr' arm was described in an earlier draft as "caret's
rule". That description was wrong, and the difference matters enough to be
worth a script rather than a footnote.

caret::findCorrelation has two implementations, and NEITHER visits pairs in
input column order:

  findCorrelation_fast   Computes each column's mean absolute correlation once,
                         finds every pair above the cutoff in a single pass, and
                         for each such pair independently flags whichever member
                         has the larger mean. There is no 'alive' guard, so no
                         cascade: the deleted set is a union over pairs, and a
                         union does not depend on the order of the terms.

  findCorrelation_exact  SORTS the columns by mean absolute correlation,
                         descending, and only then runs the nested loop. The
                         visitation order is therefore derived from the data
                         rather than from the input file, and it recomputes the
                         means after each removal.

This study's arm applies caret's CRITERION inside the tutorial idiom's LOOP:
pairs visited in raw column order, with an alive guard that lets an early
removal cancel later comparisons. That combination is order-dependent even
though each individual comparison is not, which is what Finding 5 measures.

So the honest statement is not "no tie-break can make greedy pruning
reproducible". It is: order dependence comes from visiting pairs in INPUT
order with a cascade, and it is repaired by canonicalising the visitation
order -- which mature implementations do and the tutorial idiom does not.

CAVEAT. The two functions below are ports, transcribed from caret's R source,
not R itself. The quirk in findCorrelation_exact where mn2 averages the whole
matrix minus row j -- rather than column j -- is reproduced deliberately,
because it is what the package does. Anyone relying on this should re-run the
comparison in R; it takes five minutes and it is the kind of claim a reviewer
will check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colreg.selection.threshold import CorrelationThresholdSelector as S   # noqa: E402

TAU = 0.8


def caret_fast(C, cutoff):
    A=np.abs(C); avg=A.mean(axis=0)
    uniq=np.unique(avg); rank=np.searchsorted(uniq,avg)
    p=A.shape[0]; delete=set()
    for r in range(p):
        for c in range(r+1,p):
            if A[r,c]>cutoff:
                delete.add(c if rank[c]>rank[r] else r)
    return delete

def caret_exact(C, cutoff):
    p=C.shape[0]; x=np.abs(C).astype(float)
    tmp=x.copy(); np.fill_diagonal(tmp,np.nan)
    key=np.nanmean(tmp,axis=0)
    idx=sorted(range(p), key=lambda k:(-key[k],k))
    x=x[np.ix_(idx,idx)]; new_order=np.array(idx)
    delete=np.zeros(p,bool); x2=x.copy(); np.fill_diagonal(x2,np.nan)
    for i in range(p-1):
        f=x2[~np.isnan(x2)]
        if not (f>cutoff).any(): break
        if delete[i]: continue
        for j in range(i+1,p):
            if not delete[i] and not delete[j] and x[i,j]>cutoff:
                mn1=np.nanmean(x2[i,:]); mn2=np.nanmean(np.delete(x2,j,axis=0))
                if mn1>mn2: delete[i]=True; x2[i,:]=np.nan; x2[:,i]=np.nan
                else:       delete[j]=True; x2[j,:]=np.nan; x2[:,j]=np.nan
    return set(int(new_order[k]) for k in np.flatnonzero(delete))

def sweep(X, which, perms=80, seed=1):
    p=X.shape[1]; r=np.random.default_rng(seed); seen=set()
    for k in range(perms):
        pi=np.arange(p) if k==0 else r.permutation(p)
        Xp=X[:,pi]
        if which=="ours":
            sel=S(tau=TAU,tie_break="mean_corr",fit_scope="train",random_state=0).fit(Xp)
            keep=frozenset(int(pi[c]) for c in np.flatnonzero(sel.support_))
        else:
            C=np.corrcoef(Xp,rowvar=False); drop=which(C,TAU)
            keep=frozenset(int(pi[c]) for c in range(p) if c not in drop)
        seen.add(keep)
    return len(seen)

def make(seed, kind):
    rng=np.random.default_rng(seed); n=300
    if kind=="factor":
        p=rng.integers(8,16); nf=rng.integers(2,5)
        F=rng.standard_normal((n,nf))
        L=rng.uniform(0.7,1.0,size=(nf,p))*(rng.random((nf,p))<0.6)
        return F@L+rng.uniform(0.2,0.45)*rng.standard_normal((n,p))
    if kind=="tie":
        # Bit-identical columns, so the mean |r| deciding each contest is
        # EXACTLY equal and every rule must fall back on something else. Both
        # caret variants fall back on index position here, which is what
        # separates "invariant" from "invariant except on exact ties".
        p=10; Z=rng.standard_normal((n,p//2))
        return np.hstack([Z, Z])
    p=rng.integers(6,12); r=rng.uniform(0.88,0.96)
    Z=rng.standard_normal((n,p)); X=np.empty_like(Z); X[:,0]=Z[:,0]
    for j in range(1,p): X[:,j]=r*X[:,j-1]+np.sqrt(1-r*r)*Z[:,j]
    return X


def main() -> int:
    stats = {k: {"ours": 0, "fast": 0, "exact": 0, "n": 0}
             for k in ("factor", "chain", "tie")}
    offenders = {}
    for kind in ("factor", "chain", "tie"):
        for seed in range(40):
            X = make(seed, kind)
            p = X.shape[1]
            C = np.abs(np.corrcoef(X, rowvar=False))
            if int(((C > TAU).sum() - p) / 2) == 0:
                continue
            o = sweep(X, "ours"); f = sweep(X, caret_fast); e = sweep(X, caret_exact)
            stats[kind]["n"] += 1
            stats[kind]["ours"] += (o > 1)
            stats[kind]["fast"] += (f > 1)
            stats[kind]["exact"] += (e > 1)
            if f > 1 or e > 1:
                offenders.setdefault(kind, []).append((seed, f, e))

    print("Designs where the selected subset CHANGED with column order")
    print("(80 random column orders each; 'tie' contains bit-identical columns)\n")
    print(f"  {'design':10s}{'cases':>7s}{'this study':>12s}{'caret fast':>12s}{'caret exact':>13s}")
    print("  " + "-" * 54)
    for k, v in stats.items():
        print(f"  {k:10s}{v['n']:>7d}{v['ours']:>12d}{v['fast']:>12d}{v['exact']:>13d}")
    print(f"\n  caret variants that were not invariant: "
          f"{offenders if offenders else 'none'}")
    print("\n  Conclusion. Two distinct mechanisms, which must not be conflated:")
    print("    1. CONTEST-SET CASCADE -- visiting pairs in input order lets an early")
    print("       removal cancel comparisons that would otherwise have happened, so")
    print("       the result depends on column order with NO tie required. This is")
    print("       the tutorial idiom, and it is what this study measures.")
    print("    2. TIE FALLBACK -- when the deciding statistic is exactly equal every")
    print("       rule needs a tiebreaker, and caret's is index position. This bites")
    print("       only on exact ties: duplicate columns and the like.")
    print("  caret canonicalises its traversal and escapes (1); it remains subject to")
    print("  (2). The idiom is subject to both. Saying flatly that caret is, or is")
    print("  not, order-dependent is wrong in one direction or the other.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
