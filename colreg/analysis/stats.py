"""Statistical comparison over multiple datasets -- the Demsar (2006) protocol.

Reviewers at statistics venues will look for exactly this and nothing else will
substitute for it.  In particular: do not run a t-test per dataset and count
wins.  Repeated pairwise testing across k arms inflates the family-wise error
rate, and averaging RMSE across datasets is meaningless because the scales
differ.  The Friedman test on per-dataset *ranks*, followed by a post-hoc that
controls for multiplicity, is the accepted answer.

Implemented here rather than pulled from a library so the paper can state the
exact form of the statistic and the exact critical values used.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["rank_matrix", "friedman", "nemenyi_cd", "wilcoxon_holm", "cd_diagram"]

# Studentised range / sqrt(2) at alpha = 0.05, indexed by number of arms k.
# Standard table from Demsar (2006), Table 5.
Q_ALPHA_05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
    16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
}


def rank_matrix(scores, lower_is_better: bool = True) -> np.ndarray:
    """Per-dataset ranks of each arm.  ``scores`` is (n_datasets, n_arms).

    Average ranks are assigned to ties, which is what the Friedman statistic
    assumes.  Rank *within* each dataset, never across -- that is the whole point
    of the protocol.
    """
    S = np.asarray(scores, dtype=float)
    if not lower_is_better:
        S = -S
    return np.apply_along_axis(stats.rankdata, 1, S)


def friedman(scores, lower_is_better: bool = True) -> dict:
    """Friedman test plus the Iman-Davenport F correction.

    The raw chi-squared statistic is known to be conservative; Iman and
    Davenport's F is the version Demsar recommends and the one to report.
    """
    R = rank_matrix(scores, lower_is_better)
    N, k = R.shape
    avg = R.mean(axis=0)
    chi2 = (12.0 * N / (k * (k + 1))) * (np.sum(avg ** 2) - k * (k + 1) ** 2 / 4.0)
    p_chi2 = 1.0 - stats.chi2.cdf(chi2, k - 1)
    denom = N * (k - 1) - chi2
    if denom <= 0:
        f_stat, p_f = np.inf, 0.0
    else:
        f_stat = (N - 1) * chi2 / denom
        p_f = 1.0 - stats.f.cdf(f_stat, k - 1, (k - 1) * (N - 1))
    return {
        "avg_ranks": avg,
        "chi2": float(chi2),
        "p_chi2": float(p_chi2),
        "iman_davenport_F": float(f_stat),
        "p_F": float(p_f),
        "n_datasets": int(N),
        "n_arms": int(k),
    }


def nemenyi_cd(n_datasets: int, n_arms: int, alpha: float = 0.05) -> float:
    """Critical difference in average ranks.

        CD = q_alpha * sqrt( k(k+1) / 6N )

    Two arms differ significantly iff their average ranks differ by more than
    this.  Report it as a number in the text and draw it on the CD diagram.
    """
    if alpha != 0.05:
        raise NotImplementedError("only the alpha = 0.05 table is included")
    q = Q_ALPHA_05[n_arms]
    return float(q * np.sqrt(n_arms * (n_arms + 1) / (6.0 * n_datasets)))


def wilcoxon_holm(scores, control_index: int = 0, lower_is_better: bool = True) -> list[dict]:
    """Each arm against a control, Wilcoxon signed-rank with Holm correction.

    Use this, not Nemenyi, when the question is 'does anything beat doing
    nothing' -- which is the shape of RQ1.  Nemenyi compares all pairs and is
    correspondingly less powerful; a control-vs-rest design with Holm is both
    more appropriate and easier to defend.
    """
    S = np.asarray(scores, dtype=float)
    k = S.shape[1]
    raw = []
    for j in range(k):
        if j == control_index:
            continue
        diff = S[:, j] - S[:, control_index]
        if np.allclose(diff, 0):
            p = 1.0
        else:
            p = stats.wilcoxon(S[:, j], S[:, control_index]).pvalue
        median_diff = float(np.median(diff))
        raw.append({"arm": j, "p_raw": float(p), "median_delta": median_diff})

    raw.sort(key=lambda d: d["p_raw"])
    m = len(raw)
    running = 0.0
    for i, rec in enumerate(raw):
        adj = min(1.0, (m - i) * rec["p_raw"])
        running = max(running, adj)          # Holm's step-down is monotone
        rec["p_holm"] = running
        rec["significant_05"] = running < 0.05
        rec["better_than_control"] = (
            rec["median_delta"] < 0 if lower_is_better else rec["median_delta"] > 0
        )
    return raw


def cd_diagram(avg_ranks, labels, cd: float, path: str, title: str = "",
               dpi=None) -> str:
    """Critical-difference diagram: ranks on an axis, cliques joined by bars.

    Matplotlib only, no seaborn, so the figure is reproducible from the
    replication package with no style side effects.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(avg_ranks)
    ranks = np.asarray(avg_ranks, dtype=float)[order]
    names = [labels[i] for i in order]
    k = len(ranks)

    lo = float(np.floor(ranks.min() * 2) / 2)
    hi = float(np.ceil(ranks.max() * 2) / 2)
    if hi - lo < 1.0:                     # keep a usable axis when ranks cluster
        mid = (lo + hi) / 2
        lo, hi = mid - 0.5, mid + 0.5
    span = hi - lo

    half = (k + 1) // 2                   # arms drawn on the left-hand side
    row_gap = 0.09
    y_top = 0.42 if title else 0.30       # headroom for the CD bar and title
    y_bottom = -(0.14 + row_gap * (half - 1) + 0.10)

    fig, ax = plt.subplots(figsize=(7.6, 1.5 + 0.30 * half))
    ax.set_xlim(lo - 0.30 * span, hi + 0.30 * span)
    ax.set_ylim(y_bottom, y_top)
    ax.axis("off")

    # Rank axis, lowest (best) rank on the left.
    ax.plot([lo, hi], [0, 0], color="black", lw=1.1, zorder=3)
    for t in np.arange(np.ceil(lo * 2) / 2, hi + 1e-9, 0.5):
        major = abs(t - round(t)) < 1e-9
        ax.plot([t, t], [0, 0.035 if major else 0.02], color="black", lw=1.0, zorder=3)
        if major:
            ax.text(t, 0.055, f"{t:.0f}", ha="center", va="bottom", fontsize=9)

    # Leaders: drop from the tick to a free row, then run out to the label.
    pad = 0.04 * span
    for i, (r, nm) in enumerate(zip(ranks, names)):
        left = i < half
        level = i if left else (k - 1 - i)
        y = -(0.14 + row_gap * level)
        x_end = (lo - 0.22 * span) if left else (hi + 0.22 * span)
        ax.plot([r, r], [0, y], color="black", lw=0.9, zorder=2)
        ax.plot([r, x_end], [y, y], color="black", lw=0.9, zorder=2)
        ax.text(
            x_end + (-pad if left else pad), y, nm,
            ha="right" if left else "left", va="center", fontsize=9,
        )

    # Cliques: maximal runs of arms whose average ranks span less than CD.
    y_bar = -0.055
    i = 0
    while i < k:
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i:
            ax.plot([ranks[i] - 0.01 * span, ranks[j] + 0.01 * span], [y_bar, y_bar],
                    color="black", lw=3.4, solid_capstyle="butt", zorder=4)
            y_bar -= 0.035
        i = j + 1 if j > i else i + 1

    # CD ruler, drawn above the axis so it cannot collide with the cliques.
    ax.plot([lo, lo + cd], [0.19, 0.19], color="black", lw=1.6)
    for x in (lo, lo + cd):
        ax.plot([x, x], [0.17, 0.21], color="black", lw=1.6)
    ax.text(lo + cd / 2, 0.225, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9)
    if title:
        ax.text((lo + hi) / 2, y_top - 0.005, title, ha="center", va="top", fontsize=10)

    # Default to whatever the caller has configured rather than a hard 200:
    # journals ask for 300 dpi or more, and a figure that silently ignores the
    # requested resolution is worse than one that is obviously too small.
    fig.savefig(path, dpi=dpi if dpi is not None else plt.rcParams["savefig.dpi"],
                bbox_inches="tight")
    plt.close(fig)
    return path
