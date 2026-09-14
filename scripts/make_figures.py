"""Every figure in the paper, from the shards. Nothing here needs new compute.

    python scripts/make_figures.py

Writes PDF and PNG into results/figures/, and TIFF on request.

Resolution and format, because journals are specific about both:

  PDF   Vector, and the format to submit wherever it is accepted. Resolution
        does not apply -- the art is resampled by the printer at whatever the
        press runs. Fonts are embedded as TrueType (fonttype 42) rather than
        matplotlib's default Type 3, which several publishers reject outright
        and which cannot be edited in Illustrator.

  PNG   Raster at --dpi, default 400. Above the 300 dpi most journals set as
        the floor for line art, with headroom for a figure that gets scaled
        down in a two-column layout.

  TIFF  Raster at --dpi with LZW compression, written only with --tiff.
        Some publishers still demand it and reject PNG.

Physical size is fixed in inches by figsize, so raising the dpi adds pixels
without changing how large the text appears relative to the plot. A figure
laid out to be legible at 150 dpi is equally legible at 600.

Four figures:

  fig1_cd_<estimator>   Critical-difference diagram, RQ1. Ranks on an axis,
                        arms within one CD of each other joined. Demsar (2006)
                        is the reference to cite. Drawn only when the Friedman
                        test rejects: a CD diagram over a non-significant
                        Friedman is a picture of noise with a ruler under it.

  fig2_dose_response    Median % RMSE change against the control as a function
                        of the correlation threshold. This is the figure that
                        makes RQ1 legible at a glance -- harm rises smoothly as
                        tau falls, which is what distinguishes a structural
                        effect from a handful of unlucky datasets.

  fig3_permutation      Per-dataset spread of mean RMSE over 30 column
                        permutations, as a strip with the range marked. The
                        null control is drawn on the same axis, where it is a
                        flat line at zero. That contrast IS the RQ5 result.

  fig4_stability        Nogueira stability against mean rank, one point per
                        arm. Requires results/stability.json -- run
                        analyse_stability.py first. Skipped if absent.

A caveat carried into the caption rather than buried: energy_y1 and energy_y2
share a design matrix, so the Friedman test's independence assumption is
violated by one pair out of seventeen. The script reports the test both with
and without energy_y2 so the sensitivity is visible.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colreg.analysis.stats import friedman, nemenyi_cd, cd_diagram   # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "results" / "shards"
MANIFEST = ROOT / "data" / "corpus_manifest.json"
CACHE = ROOT / "data" / "cache"
FIGS = ROOT / "results" / "figures"
STABILITY = ROOT / "results" / "stability.json"

CONTROL = "none"
EST_ORDER = ["ols", "ridge", "lasso", "enet", "cart"]
INK, MUTE, ACCENT = "#1A1A1A", "#5A6270", "#B4482E"

DPI = 400
WANT_TIFF = False

DEFAULT_DPI = 400

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 9,
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,              # on-screen only; savefig.dpi governs output
    "savefig.dpi": DEFAULT_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # Embed TrueType rather than Type 3. Type 3 is matplotlib's default and is
    # rejected by several publishers; it also renders text as unselectable
    # outlines in some viewers.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, name):
    """PDF first (vector), then the raster formats at the configured dpi."""
    FIGS.mkdir(parents=True, exist_ok=True)
    written = []
    fig.savefig(FIGS / f"{name}.pdf")            # vector; dpi is irrelevant
    written.append("pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=DPI)
    written.append(f"png@{DPI}")
    if WANT_TIFF:
        try:
            fig.savefig(FIGS / f"{name}.tiff", dpi=DPI,
                        pil_kwargs={"compression": "tiff_lzw"})
            written.append(f"tiff@{DPI}")
        except Exception as exc:                  # Pillow missing or too old
            print(f"    TIFF skipped for {name}: {exc}")
    plt.close(fig)
    print(f"  wrote results/figures/{name}.{{{', '.join(written)}}}")


def load():
    main, perm = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for f in sorted(SHARDS.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("fit_scope") != "train":
                    continue
                if r.get("perm", 0) == 0 and r.get("tie_break") == "position":
                    main[(r["estimator"], r["dataset"], r["selection"])].append(r["rmse"])
                elif r.get("perm", 0) > 0:
                    perm[(r["dataset"], r["tie_break"])][r["perm"]].append(r["rmse"])
    return ({k: st.fmean(v) for k, v in main.items()}, perm)


def prunable():
    man = json.loads(MANIFEST.read_text())["datasets"]
    return {k for k, v in man.items() if v["actual"].get("n_pairs_above_80", 0) > 0}


LEAK_R = 0.95


def leaky(datasets):
    """Datasets where one feature is nearly the target itself.

    Their permutation spreads are real but an order of magnitude larger than
    everything else, so plotting them on a shared axis compresses the fifteen
    datasets the paper actually quotes into an unreadable band around zero.
    They are drawn on their own axis instead of being dropped: the effect is
    genuine, and hiding the datasets that show it most dramatically would be
    the wrong correction.
    """
    out = set()
    for ds in datasets:
        path = CACHE / f"{ds}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        X, y = d["X"], d["y"]
        sd = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
        Zx = (X - X.mean(axis=0)) / sd
        zy = (y - y.mean()) / (y.std() or 1.0)
        r = np.abs(np.nan_to_num((Zx.T @ zy) / len(y), nan=0.0))
        if r.size and r.max() > LEAK_R:
            out.add(ds)
    return out


# ------------------------------------------------------------------- fig 1
def fig_cd(means, pr, arms):
    print("\nFigure 1: critical-difference diagrams")
    for e in EST_ORDER:
        ds = sorted({k[1] for k in means if k[0] == e and k[1] in pr})
        full = [d for d in ds if all((e, d, a) in means for a in arms)]
        if len(full) < 5:
            print(f"  {e}: only {len(full)} complete datasets -- skipped")
            continue
        M = np.array([[means[(e, d, a)] for a in arms] for d in full])
        res = friedman(M, lower_is_better=True)

        # Sensitivity: energy_y1 and energy_y2 share a design matrix.
        alt = [d for d in full if d != "energy_y2"]
        note = ""
        if len(alt) < len(full) and len(alt) >= 5:
            Ma = np.array([[means[(e, d, a)] for a in arms] for d in alt])
            note = f"; without energy_y2 p={friedman(Ma)['p_F']:.2e}"

        print(f"  {e:6s} N={res['n_datasets']} k={res['n_arms']}  "
              f"Iman-Davenport F={res['iman_davenport_F']:.2f}  "
              f"p={res['p_F']:.2e}{note}")
        if res["p_F"] >= 0.05:
            print("         Friedman not significant -- no CD diagram drawn")
            continue
        cd = nemenyi_cd(res["n_datasets"], res["n_arms"])
        labels = [a if a != CONTROL else "none (control)" for a in arms]
        FIGS.mkdir(parents=True, exist_ok=True)
        title = f"{e.upper()} — {res['n_datasets']} prunable datasets, CD={cd:.2f}"
        exts = ["pdf", "png"] + (["tiff"] if WANT_TIFF else [])
        for ext in exts:
            cd_diagram(res["avg_ranks"], labels, cd,
                       str(FIGS / f"fig1_cd_{e}.{ext}"), title=title, dpi=DPI)
        shown = ", ".join(x if x == "pdf" else f"{x}@{DPI}" for x in exts)
        print(f"         wrote results/figures/fig1_cd_{e}.{{{shown}}}  (CD={cd:.2f})")


# ------------------------------------------------------------------- fig 2
def fig_dose(means, pr, arms):
    print("\nFigure 2: dose-response")
    taus = [("corr_070", 0.70), ("corr_080", 0.80),
            ("corr_090", 0.90), ("corr_095", 0.95)]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    drew = False
    for e in EST_ORDER:
        ds = sorted({k[1] for k in means if k[0] == e and k[1] in pr})
        full = [d for d in ds if all((e, d, a) in means for a in arms)]
        if len(full) < 5:
            continue
        xs, ys = [], []
        for arm, t in taus:
            d = [100 * (means[(e, x, arm)] - means[(e, x, CONTROL)])
                 / means[(e, x, CONTROL)] for x in full]
            xs.append(t)
            ys.append(float(np.median(d)))
        ax.plot(xs, ys, marker="o", ms=4, lw=1.4, label=e.upper())
        drew = True
    if not drew:
        print("  no complete estimator -- skipped")
        plt.close(fig)
        return
    ax.axhline(0, color=INK, lw=1.0)
    # Axes fraction, not data coordinates: in data coordinates this label lands
    # on the y-axis and collides with its title.
    ax.text(0.985, 0.045, "no pruning", transform=ax.transAxes,
            fontsize=7.5, color=MUTE, ha="right", va="bottom")
    ax.set_xlabel("correlation threshold τ  (lower = more aggressive pruning)")
    ax.set_ylabel("median RMSE change vs no pruning (%)")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Harm rises monotonically as the threshold tightens", fontsize=9.5)
    save(fig, "fig2_dose_response")


# ------------------------------------------------------------------- fig 3
def fig_perm(perm):
    print("\nFigure 3: permutation spread")

    def spreads(tie):
        out = {}
        for (ds, tb), perms in perm.items():
            if tb != tie or len(perms) < 5:
                continue
            m = [st.fmean(v) for v in perms.values()]
            mu = st.fmean(m)
            if mu:
                out[ds] = (100.0 * (min(m) - mu) / mu, 100.0 * (max(m) - mu) / mu)
        return out

    pos, ctrl = spreads("position"), spreads("mean_corr")
    if not pos:
        print("  no permutation results -- skipped")
        return

    bad = leaky(pos)
    clean = sorted([d for d in pos if d not in bad],
                   key=lambda d: pos[d][1] - pos[d][0], reverse=True)
    hot = sorted([d for d in pos if d in bad],
                 key=lambda d: pos[d][1] - pos[d][0], reverse=True)

    # Two axes with independent x-scales, heights proportional to row counts.
    # A shared axis would squeeze the fifteen quotable datasets to invisibility
    # next to a 124% range.
    if hot:
        fig, axes = plt.subplots(
            2, 1, figsize=(5.8, 0.26 * (len(clean) + len(hot)) + 2.1),
            gridspec_kw={"height_ratios": [max(len(clean), 1), max(len(hot), 1)],
                         "hspace": 0.6})
        panels = [(axes[0], clean, None), (axes[1], hot, None)]
    else:
        fig, ax = plt.subplots(figsize=(5.8, max(3.0, 0.26 * len(clean) + 1.2)))
        panels = [(ax, clean, None)]

    for ax, names, _ in panels:
        for i, ds in enumerate(names):
            lo, hi = pos[ds]
            ax.plot([lo, hi], [i, i], color=ACCENT, lw=3,
                    solid_capstyle="butt", zorder=3)
            if ds in ctrl:
                c_lo, c_hi = ctrl[ds]
                ax.plot([c_lo, c_hi], [i, i], color=INK, lw=6, alpha=0.85, zorder=4)
        ax.axvline(0, color=MUTE, lw=0.8, ls=":")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7.5)
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.set_xlabel("mean RMSE across 30 column permutations, % from the mean",
                      fontsize=8)

    axes0 = panels[0][0]
    axes0.set_title("Identical data, identical folds, different column order",
                    fontsize=9.5)
    if hot:
        panels[1][0].set_title(
            "Datasets carrying a near-copy of the target — note the axis scale",
            fontsize=8.5, color=MUTE)

    # Anchor the legend under the LAST panel: hung under the first it lands on
    # the second panel's title.
    last_ax, last_names, _ = panels[-1]
    last_ax.plot([], [], color=ACCENT, lw=3, label="positional tie-break")
    last_ax.plot([], [], color=INK, lw=6, alpha=0.85, label="order-invariant control")
    # Anchor in FIGURE coordinates: an offset expressed as a fraction of a
    # three-row panel is tiny in absolute terms and lands on its x-label.
    last_ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
                   bbox_to_anchor=(0.5, -0.015), bbox_transform=fig.transFigure)
    save(fig, "fig3_permutation")


# ------------------------------------------------------------------- fig 4
def fig_stability(means, pr, arms):
    print("\nFigure 4: stability against accuracy")
    if not STABILITY.exists():
        print("  results/stability.json absent -- run analyse_stability.py first")
        return
    S = json.loads(STABILITY.read_text())
    table, comparable = S["phi_by_arm_estimator"], S["comparable_arms"]

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    drew = False
    ranks_by_est = {}
    for e in EST_ORDER:
        ds = sorted({k[1] for k in means if k[0] == e and k[1] in pr})
        full = [d for d in ds if all((e, d, a) in means for a in arms)]
        if len(full) < 5:
            continue
        M = np.array([[means[(e, d, a)] for a in arms] for d in full])
        rank_of = dict(zip(arms, np.apply_along_axis(stats.rankdata, 1, M).mean(axis=0)))
        ranks_by_est[e] = rank_of
        xs = [table[a][e] for a in comparable if e in table.get(a, {})]
        ys = [rank_of[a] for a in comparable if e in table.get(a, {})]
        if not xs:
            continue
        ax.scatter(xs, ys, s=26, alpha=0.85, label=e.upper())
        drew = True
    if not drew:
        print("  nothing to plot -- skipped")
        plt.close(fig)
        return
    ax.set_xlabel("Nogueira selection stability Φ")
    ax.set_ylabel("mean rank (lower is better)")
    ax.invert_yaxis()
    # Room for the annotations, which sit below their points: the worst-ranked
    # arm lands on the bottom axis and its label would be clipped.
    lo, hi = ax.get_ylim()                       # inverted, so lo > hi
    pad = 0.07 * abs(lo - hi)
    ax.set_ylim(lo + pad, hi - pad)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Selection stability against predictive rank", fontsize=9.5)
    # Label the two arms the text singles out, so the figure and the prose
    # refer to visibly the same points.
    ridge_ranks = ranks_by_est.get("ridge", {})
    for arm, dx, dy in (("corr_070", 0, -13), ("corr_tuned", 0, -13)):
        if arm in table and "ridge" in table[arm] and arm in ridge_ranks:
            ax.annotate(arm, (table[arm]["ridge"], ridge_ranks[arm]),
                        textcoords="offset points", xytext=(dx, dy),
                        fontsize=7, color=MUTE, ha="center")
    # No explanatory text inside the figure. The caveats this plot needs -- the
    # control being undefined rather than 1.0, and corr_tuned mixing two sources
    # of variation -- belong in the manuscript caption, where they can be read
    # at full size and edited without regenerating the artwork.
    save(fig, "fig4_stability")


def main() -> int:
    global DPI, WANT_TIFF
    ap = argparse.ArgumentParser(description="Build every figure from the shards.")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                    help=f"raster resolution for PNG and TIFF (default {DEFAULT_DPI}; "
                         "journals typically require at least 300 for line art)")
    ap.add_argument("--tiff", action="store_true",
                    help="also write LZW-compressed TIFF, for publishers that "
                         "will not accept PNG")
    args = ap.parse_args()
    if args.dpi < 300:
        print(f"warning: {args.dpi} dpi is below the 300 dpi floor most journals "
              f"set for line art")
    DPI, WANT_TIFF = args.dpi, args.tiff
    plt.rcParams["savefig.dpi"] = DPI
    print(f"raster output at {DPI} dpi; PDF is vector\n")

    if not SHARDS.exists():
        print("no results/shards found")
        return 1
    means, perm = load()
    if not means:
        print("no main-stage results found")
        return 1
    pr = prunable()
    arms = sorted({k[2] for k in means})

    fig_cd(means, pr, arms)
    fig_dose(means, pr, arms)
    fig_perm(perm)
    fig_stability(means, pr, arms)

    kinds = "PDF (vector), PNG" + (", TIFF" if WANT_TIFF else "")
    print(f"\nAll figures written to results/figures/ as {kinds}; "
          f"raster at {DPI} dpi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
