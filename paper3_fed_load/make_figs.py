# make_figs.py -- regenerates every results figure in the paper from the saved
# CSVs, in one consistent style. Run after the sweeps; no rollouts needed.
#
#   exec(open(f"{PROJECT_DIR}/make_figs.py").read())
#
# Seven figures. Stale files from earlier versions are deleted on start.
#
#   symmetry           what a symmetric score costs        width=\columnwidth
#   fig_costopt        cost curves                         width=\columnwidth
#   bandit_arms        arm trace + per-season medians      width=0.72\textwidth
#
#   doseresponse       five static schemes vs lambda       SUPPLEMENTARY
#   regimes            fixed / matched / adaptive          SUPPLEMENTARY
#   fm_backbone        MLP vs Chronos, both scores         SUPPLEMENTARY
#   cost_regimes       holding-vs-imbalance plane          SUPPLEMENTARY
#
# TEN-PAGE REVISION. Three changes against the previous version:
#
#   1. Everything is written as PDF as well as PNG. LaTeX picks the PDF up if
#      \includegraphics is given the stem with no extension. Vector output is
#      what makes the lines and text stay sharp at the reduced sizes below.
#
#   2. symmetry and fig_costopt are drawn at COL and stack their two panels
#      vertically, so each panel keeps the full column width instead of taking
#      half of it. Previously symmetry was drawn at PAGE*0.72 = 5.16 in and
#      placed at \textwidth = 7.16 in, so LaTeX upscaled it by 1.39 and an 8 pt
#      label reached the page at 11 pt. Drawing at the placement width removes
#      that magnification and the white space that came with it.
#
#   3. Annotations are shortened, not deleted. Every number the body text or
#      the abstract quotes from a figure is still printed on it, in particular
#      the +25% cost rise against the 21% to 51% WAPE sweep in symmetry, which
#      the abstract's closing sentence depends on.
#
# The four supplementary figures are unchanged apart from the shared style
# block, since they no longer compete for page budget.
#
# Needs results_baselines.csv (from baselines.py) and results_fm_signed.csv
# (from baselines_fm.py). Figures that need them are skipped if absent.
#
# cost_regimes EXCLUDES the scheduling term ($143/day, identical for every
# scheme) so the axes carry only what calibration controls.
#
# NOTE: cost_optimize.py also writes fig_costopt with the old label.
# Run make_figs.py last, or comment out its plot_cost_curves() call.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = OUT_DIR if "OUT_DIR" in globals() else "."

for _stale in ("adaptive.png", "calibtransfer.png",
               "adaptive.pdf", "calibtransfer.pdf"):
    _p = os.path.join(OUT, _stale)
    if os.path.exists(_p):
        os.remove(_p)
        print("removed stale", _stale)

# Point sizes are absolute, so they only mean what they say when the figure is
# drawn at the width it is placed at. See note 2 in the header.
# ---- font resolution -----------------------------------------------------
# Arial first. Liberation Sans is the fallback and is metric-compatible with
# Arial, so a machine without Arial lays text out at identical widths. Colab
# ships Liberation. For real Arial, run once before this script:
#   !apt-get -qq install -y ttf-mscorefonts-installer > /dev/null && fc-cache -f
#   import matplotlib.font_manager as fm; fm._load_fontmanager(try_read_cache=False)
import matplotlib.font_manager as fm

_have = {f.name for f in fm.fontManager.ttflist}
FAMILY = next((n for n in ("Arial", "Helvetica", "Liberation Sans",
                           "Nimbus Sans", "DejaVu Sans") if n in _have),
              "DejaVu Sans")
print(f"font resolved to: {FAMILY}")

# Every piece of text is pure black. Coloured labels and annotations read as
# low contrast at print size; colour is left to carry scheme identity in the
# lines and markers, where it has something to distinguish. mathtext is bound
# to the same family so $\lambda$ and PICP$_{80}$ do not drop to the default
# serif face mid-label.
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FAMILY],
    "mathtext.fontset": "custom",
    "mathtext.rm": f"{FAMILY}:regular",
    "mathtext.it": f"{FAMILY}:italic",
    "mathtext.bf": f"{FAMILY}:bold",
    "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 9.0,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "legend.fontsize": 7.2,
    "text.color": "black", "axes.labelcolor": "black",
    "axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black",
    "axes.grid": True, "grid.color": "0.78", "grid.alpha": 1.0,
    "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.linewidth": 1.0, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "legend.borderpad": 0.2,
    "legend.labelspacing": 0.25, "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4, "legend.columnspacing": 0.9,
    "lines.markersize": 4.0, "lines.linewidth": 1.9,
    "lines.markeredgewidth": 0.0, "lines.solid_capstyle": "round",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white",
})
COL, PAGE = 3.5, 7.16          # IEEE single- and double-column widths, inches
LAM = r"degradation level $\lambda$"
INK = "black"                  # one name for every annotation colour


def save(fig, name, layout="tight"):
    """Write PDF and PNG side by side.

    The PDF is the one to place: it stays vector, so strokes and glyphs are
    resolution-independent and hold their edges at any zoom. A PNG is a raster
    and gets resampled by whatever renders the page, which is what soft edges
    look like. 600 dpi makes the raster fallback as good as a raster gets."""
    if layout == "tight":
        fig.tight_layout(pad=0.25)
    stem = os.path.splitext(name)[0]
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
        fig.savefig(os.path.join(OUT, stem + ext),
                    bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)
    print("saved", stem + ".pdf  +  " + stem + ".png")


# High-contrast palette. The previous one leaned on light greys (#9AA7B5) and
# pastels (#B0A8E0, #F0A93B) that dropped out against white at print size.
# These are all dark enough to hold a 1.9 pt stroke, and every scheme also
# carries its own marker, so the figures survive a greyscale print run.
S = {
    "integrated":       ("integrated",                    "#C1121F", "s"),
    "modular_fed":      (r"modular$_\mathrm{fed}$",       "#495057", "P"),
    "modular_local":    (r"modular$_\mathrm{local}$",     "#0B5FA5", "^"),
    "cqr_fed":          (r"CQR$_\mathrm{fed}$",           "#7B2CBF", "X"),
    "cqr_local":        (r"CQR$_\mathrm{local}$",         "#4A0D80", "D"),
    "adaptive":         (r"adaptive, symmetric",          "#00695C", "o"),
    "adaptive_cqr":     (r"adaptive$_\mathrm{CQR}$",      "#0B6E4F", "v"),
    "adaptive_signed":  (r"adaptive, signed",             "#A4036F", "*"),
    "fixed@0.0":        ("fixed@0",                       "#B35C00", "X"),
    "fixed@0":          ("fixed@0",                       "#B35C00", "X"),
    "fixed@0.5":        ("fixed@0.5",                     "#7F4F24", "s"),
    "fixed@1.0":        ("fixed@1",                       "#495057", "P"),
    "fixed@1":          ("fixed@1",                       "#495057", "P"),
    "matched":          ("matched (oracle)",              "#000000", "D"),
}

LS = {
    "fixed@0.5":       (0, (5, 2)),
    "adaptive_cqr":    (0, (4, 1.6)),
    "adaptive_signed": "-",
}


def line(ax, sub, x, y, key, **kw):
    lab, c, m = S[key]
    kw.setdefault("ls", LS.get(key, "-"))
    ax.plot(sub[x], sub[y], marker=m, color=c, label=lab, **kw)


def pick(df, col, *names):
    for n in names:
        if (df[col] == n).any():
            return n
    return None


def load(name, level_col="level"):
    d = pd.read_csv(os.path.join(OUT, name))
    if level_col not in d.columns and "test_level" in d.columns:
        d = d.rename(columns={"test_level": level_col})
    return d


def baselines(base="MLP-point"):
    """results_baselines.csv, one backbone, indexed by (scheme, level)."""
    d = load("results_baselines.csv")
    return d[d.base == base]


# ==========================================================================
# dose-response, five static schemes (all matched)   -- SUPPLEMENTARY
# ==========================================================================
def fig_dose():
    d = load("results_doseresponse.csv")
    order = ["integrated", "modular_fed", "modular_local", "cqr_fed", "cqr_local"]
    fig, ax = plt.subplots(1, 3, figsize=(PAGE, 2.3))

    w = d.groupby("level")["WAPE"].first()
    ax[0].plot(w.index, w.values, "-o", color="#444441")
    ax[0].set_xlabel(LAM); ax[0].set_ylabel("WAPE (%)")
    ax[0].annotate("shared across schemes", (0.02, w.values[-1] * 0.92),
                   fontsize=7, color="#666")

    for k in order[1:]:
        sub = d[d.scheme == k].sort_values("level")
        if sub.empty:
            print(f"   [warn] no rows for {k}")
            continue
        line(ax[1], sub, "level", "PICP80", k)
    ax[1].axhline(0.8, ls="--", c="k", lw=0.9)
    ax[1].set_ylim(0.68, 0.94)
    ax[1].text(0.02, 0.905, "nominal 0.80", fontsize=6.5)
    ax[1].annotate(r"integrated: $0.25 \rightarrow 0.14$, below axis",
                   xy=(0.02, 0.702), fontsize=6.5, color="#D4537E")
    ax[1].set_xlabel(LAM); ax[1].set_ylabel(r"PICP$_{80}$")

    for k in order:
        sub = d[d.scheme == k].sort_values("level")
        if sub.empty:
            continue
        line(ax[2], sub, "level", "MPIW80", k)
    ax[2].annotate("narrow because it under-covers,\nnot because it is sharp",
                   xy=(0.5, 18), xytext=(0.14, 62), fontsize=6, color="#D4537E",
                   arrowprops=dict(arrowstyle="->", color="#D4537E", lw=0.8))
    ax[2].set_xlabel(LAM); ax[2].set_ylabel(r"MPIW$_{80}$ (kW)")
    ax[2].legend(fontsize=6.5, loc="upper left")
    save(fig, "doseresponse")


# ==========================================================================
# calibration regimes -- fixed / matched / adaptive   -- SUPPLEMENTARY
# ==========================================================================
def fig_regimes():
    t = load("results_calibtransfer.csv")
    t = t[t.scheme == "modular_local"][["calib", "level", "PICP80", "MPIW80"]]
    t = t.rename(columns={"calib": "scheme"})

    a = load("results_adaptive.csv")
    try:
        a = pd.concat([a, load("results_adaptive_cqr.csv")], ignore_index=True)
    except FileNotFoundError:
        print("   [warn] results_adaptive_cqr.csv missing")
    a = a[a.scheme.isin(["adaptive", "adaptive_cqr"])][
        ["scheme", "level", "PICP80", "MPIW80"]]

    frames = [t, a]
    try:
        b = baselines()
        g = b[b.scheme == "acl_asym"][["level", "PICP80", "MPIW80"]].copy()
        g["scheme"] = "adaptive_signed"
        frames.append(g)
    except FileNotFoundError:
        print("   [warn] results_baselines.csv missing, no signed curve")

    d = pd.concat(frames, ignore_index=True)
    keys = [k for k in (pick(d, "scheme", "fixed@0.0", "fixed@0"),
                        pick(d, "scheme", "fixed@0.5"),
                        pick(d, "scheme", "fixed@1.0", "fixed@1"),
                        pick(d, "scheme", "matched"),
                        pick(d, "scheme", "adaptive"),
                        pick(d, "scheme", "adaptive_cqr"),
                        pick(d, "scheme", "adaptive_signed")) if k]

    LW = {"adaptive": 2.6, "adaptive_cqr": 1.4, "adaptive_signed": 2.0}

    fig, ax = plt.subplots(1, 2, figsize=(PAGE * 0.72, 2.7))
    for k in keys:
        s = d[d.scheme == k].sort_values("level")
        z = (5 if k == "adaptive_signed" else
             4 if k == "adaptive_cqr" else 3 if k == "adaptive" else 2)
        line(ax[0], s, "level", "PICP80", k, lw=LW.get(k, 1.3), zorder=z)
        line(ax[1], s, "level", "MPIW80", k, lw=LW.get(k, 1.3), zorder=z)
    ax[0].axhline(0.8, ls="--", c="k", lw=0.9)
    ax[0].set_xlabel(LAM); ax[0].set_ylabel(r"PICP$_{80}$")
    ax[1].set_xlabel(LAM); ax[1].set_ylabel(r"MPIW$_{80}$ (kW)")

    if "adaptive_signed" in keys:
        ax[1].text(0.02, 132,
                   "signed at $\\lambda{=}1$ is narrower\n"
                   "than fixed@0 at $\\lambda{=}0$",
                   fontsize=5.8, color="#B5179E", va="top")

    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.2, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.10))
    save(fig, "regimes")
    print("   drawn:", keys)


# ==========================================================================
# what a symmetric score costs -- Section IV-D          SINGLE COLUMN
# ==========================================================================
def fig_symmetry():
    """Two panels stacked, sharing the degradation axis. Top: the width a
    symmetric calibrator must hold divided by the signed one, on both
    backbones, which separates what federation costs from what degradation
    costs. Bottom: operating cost against degradation."""
    b = baselines()

    def series(scheme, col):
        s = b[b.scheme == scheme].sort_values("level")
        return s.level.values, s[col].values

    lv, w_sym = series("acl", "MPIW80")
    _, w_sig = series("acl_asym", "MPIW80")

    fig, ax = plt.subplots(2, 1, figsize=(COL, 3.55), sharex=True,
                           gridspec_kw={"height_ratios": [1.0, 1.15]})

    # ---- top: width ratio, both backbones ----
    ax[0].plot(lv, w_sym / w_sig, "-o", color="#00695C", lw=2.1,
               label="federated MLP")
    try:
        f = load("results_fm_signed.csv")
        a = f[f.scheme == "fm_adaptive_cqr"].sort_values("level")
        c = f[f.scheme == "fm_adaptive_cqr_signed"].sort_values("level")
        ax[0].plot(a.level.values, a.MPIW80.values / c.MPIW80.values,
                   "--s", color="#4A0D80", lw=2.1, label="zero-shot FM")
    except FileNotFoundError:
        print("   [warn] results_fm_signed.csv missing, no FM ratio")
    ax[0].axhline(1.0, ls=":", c="k", lw=1.0)
    ax[0].set_ylabel("symmetric / signed\nwidth", linespacing=1.25)
    ax[0].legend(loc="upper left", ncol=2)
    # the clean-input gap is the whole point of the panel, so it is labelled
    # on the axis rather than left to the caption
    ax[0].annotate(f"{w_sym[0]/w_sig[0]:.2f} before\nany degradation",
                   xy=(0.0, w_sym[0] / w_sig[0]), xytext=(0.13, 1.14),
                   fontsize=7.0, color=INK, linespacing=1.25,
                   arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))

    # ---- bottom: operating cost against degradation ----
    for scheme, key, lw in (("split", "fixed@0", 1.7),
                            ("acl", "adaptive", 1.9),
                            ("acl_debias", None, 1.7),
                            ("acl_asym", "adaptive_signed", 2.4)):
        s = b[b.scheme == scheme].sort_values("level")
        if s.empty:
            continue
        if key is None:
            ax[1].plot(s.level, s.J_B, "-d", color="#8A6100", lw=lw,
                       label="adaptive $+$ intercept")
        else:
            lab, col, mk = S[key]
            ax[1].plot(s.level, s.J_B, marker=mk, color=col, lw=lw,
                       ls=LS.get(key, "-"), label=lab)
    ax[1].set_xlabel(LAM)
    # no escaped dollar here: matplotlib is not running usetex, so a
    # backslash before $ renders literally once the string also carries a
    # mathtext pair
    ax[1].set_ylabel("operating cost $J$\n(USD/day/building)", linespacing=1.25)
    ax[1].legend(loc="upper left", ncol=2, fontsize=6.8)

    j0 = float(b[(b.scheme == "acl_asym") & (b.level == 0.0)].J_B.iloc[0])
    j1 = float(b[(b.scheme == "acl_asym") & (b.level == 1.0)].J_B.iloc[0])
    ax[1].set_ylim(bottom=j0 * 0.50)
    # the abstract's closing sentence quotes this number, so it stays on the
    # figure even at the reduced size
    ax[1].annotate(f"signed: +{100*(j1-j0)/j0:.0f}% while WAPE goes 21% to 51%",
                   xy=(0.75, 112), xytext=(0.50, j0 * 0.55), fontsize=7.0,
                   color=INK, ha="center", va="bottom",
                   arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    save(fig, "symmetry")
    print(f"   width ratio: {np.round(w_sym / w_sig, 2).tolist()}")
    print(f"   signed cost: {j0:.0f} -> {j1:.0f} "
          f"(+{100*(j1-j0)/j0:.0f}%) while WAPE goes 21% -> 51%")


# ==========================================================================
# MLP vs Chronos, both scores                         -- SUPPLEMENTARY
# ==========================================================================
def fig_fm():
    m = load("results_doseresponse.csv")
    for extra in ("results_adaptive.csv", "results_adaptive_cqr.csv"):
        try:
            m = pd.concat([m, load(extra)], ignore_index=True)
        except FileNotFoundError:
            print(f"   [warn] {extra} missing")

    f = load("results_fm.csv")
    try:
        f = pd.concat([f, load("results_fm_adaptive_cqr.csv")], ignore_index=True)
    except FileNotFoundError:
        print("   [warn] results_fm_adaptive_cqr.csv missing")
    f["scheme"] = f.scheme.map({"fm_modular_local": "modular_local",
                                "fm_cqr_local": "cqr_local",
                                "fm_adaptive": "adaptive",
                                "fm_adaptive_cqr": "adaptive_cqr"})
    conf = ["modular_local", "cqr_local", "adaptive", "adaptive_cqr"]

    fig, ax = plt.subplots(1, 2, figsize=(PAGE * 0.72, 2.7))
    for d, ls in ((m, "-"), (f, (0, (4, 2)))):
        for k in conf:
            s = d[d.scheme == k].sort_values("level")
            if s.empty:
                continue
            _, c, mk = S[k]
            ax[0].plot(s.level, s.PICP80, ls=ls, color=c, marker=mk, ms=3.2)
            ax[1].plot(s.level, s.MPIW80, ls=ls, color=c, marker=mk, ms=3.2)

    _, sc, smk = S["adaptive_signed"]
    try:
        g = baselines()
        g = g[g.scheme == "acl_asym"].sort_values("level")
        ax[0].plot(g.level, g.PICP80, "-", color=sc, marker=smk, ms=4.5)
        ax[1].plot(g.level, g.MPIW80, "-", color=sc, marker=smk, ms=4.5)
    except FileNotFoundError:
        pass
    try:
        h = load("results_fm_signed.csv")
        h = h[h.scheme == "fm_adaptive_cqr_signed"].sort_values("level")
        ax[0].plot(h.level, h.PICP80, ls=(0, (4, 2)), color=sc,
                   marker=smk, ms=4.5)
        ax[1].plot(h.level, h.MPIW80, ls=(0, (4, 2)), color=sc,
                   marker=smk, ms=4.5)
    except FileNotFoundError:
        pass

    ax[0].axhline(0.8, ls="--", c="k", lw=0.9)
    ax[0].set_ylim(0.68, 0.94)
    ax[0].text(0.02, 0.905, "nominal 0.80", fontsize=6.5)
    ax[0].set_xlabel(LAM); ax[0].set_ylabel(r"PICP$_{80}$")
    ax[1].set_xlabel(LAM); ax[1].set_ylabel(r"MPIW$_{80}$ (kW)")

    sch_h = [Line2D([], [], color=S[k][1], marker=S[k][2], ms=3.2,
                    label=S[k][0]) for k in conf]
    sch_h.append(Line2D([], [], color=sc, marker=smk, ms=4.5,
                        label=S["adaptive_signed"][0]))
    bb_h = [Line2D([], [], color="0.35", ls="-", label="MLP"),
            Line2D([], [], color="0.35", ls=(0, (4, 2)), label="Chronos")]
    ax[0].legend(handles=sch_h, fontsize=5.8, loc="lower left")
    ax[1].legend(handles=bb_h, fontsize=6.5, loc="upper left")
    save(fig, "fm_backbone")

    for k in ("adaptive", "adaptive_cqr"):
        for lvl, note in ((1.0, ""), (0.0, " on clean inputs")):
            a = m[(m.scheme == k) & (m.level == lvl)]
            b_ = f[(f.scheme == k) & (f.level == lvl)]
            if a.empty or b_.empty:
                continue
            va, vb = float(a.MPIW80.iloc[0]), float(b_.MPIW80.iloc[0])
            print(f"   {k} at lambda={lvl:.0f}:  MLP {va:.1f} kW  "
                  f"Chronos {vb:.1f} kW{note}")


# ==========================================================================
# cost curves                                           SINGLE COLUMN
# ==========================================================================
def fig_costopt():
    """Two panels stacked. Top: the cost curves, each optimum marked in its
    own colour. Bottom: Proposition 1 shown rather than asserted, the
    closed-form map from cost ratio to coverage with the empirical minima
    sitting on it. Stacked because the two panels do not share an x axis and
    neither reads at half a column."""
    d = pd.read_csv(os.path.join(OUT, "cost_curve.csv"))
    fig, ax = plt.subplots(2, 1, figsize=(COL, 3.60),
                           gridspec_kw={"height_ratios": [1.0, 1.0]})

    # dark, saturated, and ordered so the three curves separate in greyscale
    RATIO_C = ["#0B5FA5", "#C1121F", "#4A0D80", "#00695C", "#B35C00"]

    # ---- top: J(alpha), each curve with its own optimum ----
    for i, (r, g) in enumerate(d.groupby("cp_ch_ratio")):
        g = g.sort_values("coverage")
        c = RATIO_C[i % len(RATIO_C)]
        ax[0].plot(g.coverage, g.J_norm, lw=2.0, color=c,
                   label=fr"$c_p/c_h={int(r)}$")
        o = g[g.is_optimum == 1]
        if o.empty:
            continue
        cov = float(o.coverage.iloc[0])
        ax[0].plot(cov, float(o.J_norm.iloc[0]), "o", color=c, ms=5, zorder=5)
        ax[0].annotate(f"{cov:.2f}", (cov, float(o.J_norm.iloc[0])),
                       textcoords="offset points", xytext=(0, 9),
                       color=INK, fontsize=7.0, ha="center", zorder=6,
                       bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))
    ax[0].set_xlabel(r"operating coverage $1-\alpha$", labelpad=1.5)
    ax[0].set_ylabel(r"cost $J(\alpha)$ (norm.)")
    ax[0].set_xlim(0.40, 0.98)
    ax[0].set_ylim(bottom=0.90)
    ax[0].legend(loc="upper center", ncol=3, fontsize=7.0)

    # ---- bottom: the closed form, with the empirical optima on it ----
    rr = np.geomspace(2.0, 30.0, 200)
    ax[1].plot(rr, (rr - 1.0) / (rr + 1.0), "-", color="#000000", lw=2.0,
               label=r"$1-\alpha^\star = \frac{c_p-c_h}{c_p+c_h}$")
    try:
        s = pd.read_csv(os.path.join(OUT, "cost_optima_sweep.csv"))
        ax[1].plot(s.cp_ch_ratio, s.coverage_emp, "o", color="#A4036F",
                   ms=5.0, mfc="none", mew=1.6, label="empirical minimum")
        err = float((s.alpha_star_emp - s.alpha_star_theory).abs().max())
        print(f"   max |empirical - closed form| over the ratio sweep: {err:.4f}")
    except FileNotFoundError:
        print("   [warn] cost_optima_sweep.csv missing; run cost_ratio_sweep.py")
        o = d[d.is_optimum == 1]
        ax[1].plot(o.cp_ch_ratio, o.coverage, "o", color="#A4036F",
                   ms=5.0, mfc="none", mew=1.6, label="empirical minimum")
    for r_, lab in ((4, "60%"), (9, "80%"), (19, "90%")):
        cv = (r_ - 1.0) / (r_ + 1.0)
        ax[1].plot([r_, r_], [0.30, cv], ls=":", c="0.55", lw=1.0, zorder=0)
        ax[1].text(r_, cv + 0.022, lab, fontsize=7.0, ha="center", color=INK)
    ax[1].set_xscale("log")
    ax[1].set_xticks([2, 4, 9, 19, 30])
    ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1].set_xlabel(r"shortfall-to-holding ratio $c_p/c_h$", labelpad=1.5)
    ax[1].set_ylabel(r"optimal coverage $1-\alpha^\star$")
    ax[1].set_ylim(0.30, 1.02)
    ax[1].legend(loc="lower right", fontsize=7.0)
    save(fig, "fig_costopt")


# ==========================================================================
# bandit                                              0.72 * TEXTWIDTH
# ==========================================================================
def fig_bandit(days_per_season=None):
    """Left: how often each candidate level was chosen, by season, for each
    tariff segment. Right: what that selection was worth against a fixed
    target, with the known-tariff optimum as the reference. Stays double
    column: the left panel is a five-by-six heat map in two segments and does
    not survive being halved. Height is reduced instead."""
    h = np.load(os.path.join(OUT, "bandit_hist_seasons.npy"))
    n = len(h)
    d1 = days_per_season or (n // 6)
    ns = max(1, n // d1)
    ARMS = np.array([0.05, 0.10, 0.20, 0.30, 0.40])
    ORACLE = {"peak": 0.10, "off-peak": 0.40}

    fig, ax = plt.subplots(1, 2, figsize=(PAGE * 0.72, 2.65),
                           gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- left: selection frequency, two stacked strips in one axes --------
    freq = {}
    for j, seg in enumerate(("peak", "off-peak")):
        f = np.zeros((len(ARMS), ns))
        for s in range(ns):
            blk = h[s * d1:(s + 1) * d1, j]
            for i, a in enumerate(ARMS):
                f[i, s] = np.mean(np.isclose(blk, a))
        freq[seg] = f

    gap = 1                                    # blank row between the strips
    rows = 2 * len(ARMS) + gap
    M = np.full((rows, ns), np.nan)
    M[:len(ARMS)] = freq["peak"][::-1]
    M[len(ARMS) + gap:] = freq["off-peak"][::-1]
    im = ax[0].imshow(M, aspect="auto", origin="upper", cmap="BuPu",
                      vmin=0, vmax=1,
                      extent=(0.5, ns + 0.5, rows - 0.5, -0.5))

    ylab, ypos = [], []
    for i, a in enumerate(ARMS[::-1]):
        ylab.append(f"{a:.2f}"); ypos.append(i)
    ylab.append(""); ypos.append(len(ARMS))
    for i, a in enumerate(ARMS[::-1]):
        ylab.append(f"{a:.2f}"); ypos.append(len(ARMS) + gap + i)
    ax[0].set_yticks(ypos); ax[0].set_yticklabels(ylab, fontsize=7.2)
    ax[0].set_xticks(np.arange(1, ns + 1))
    ax[0].set_xlabel("replayed season", labelpad=1.5)
    ax[0].set_ylabel(r"candidate $\alpha^\star$")
    ax[0].grid(False)

    tick_lookup = {}
    for j, (seg, off) in enumerate((("peak", 0),
                                    ("off-peak", len(ARMS) + gap))):
        k = int(np.argmin(np.abs(ARMS[::-1] - ORACLE[seg])))
        ax[0].add_patch(plt.Rectangle((0.5, off + k - 0.5), ns, 1.0,
                                      fill=False, ec="#C1121F", lw=1.6,
                                      zorder=5))
        tick_lookup[off + k] = True
        ax[0].text(0.42, off - 0.42, seg, fontsize=7.4, ha="left",
                   va="bottom", color=INK, transform=ax[0].transData)
    for pos, lbl in zip(ypos, ax[0].get_yticklabels()):
        if tick_lookup.get(pos):
            lbl.set_color("#C1121F")
            lbl.set_fontweight("bold")
    cb = fig.colorbar(im, ax=ax[0], pad=0.015, fraction=0.036)
    cb.set_label("fraction of days chosen", fontsize=7.0, color=INK)
    cb.ax.tick_params(labelsize=6.6, color=INK, labelcolor=INK)
    cb.outline.set_edgecolor(INK)

    # ---- right: what the selection was worth -----------------------------
    try:
        z = np.load(os.path.join(OUT, "bandit_seasons.npz"))
        sv, orc = z["saving_pct"], float(z["oracle_saving_pct"])
        xs = np.arange(1, len(sv) + 1)
        ax[1].axhline(orc, ls="--", c="#000000", lw=1.5)
        ax[1].text(1.05, orc - 0.10, f"known-tariff optimum {orc:.1f}%",
                   fontsize=7.0, va="top", ha="left", color=INK)
        ax[1].plot(xs, sv, "-o", color="#4A0D80", lw=2.1, ms=5)
        hit = next((int(x) for x, v in zip(xs, sv) if v >= orc), None)
        if hit is not None:
            ax[1].text(len(sv) - 0.15, sv.min() + 0.55,
                       f"reaches it in season {hit},\n"
                       "without observing the rates",
                       fontsize=7.0, color=INK, ha="right", va="bottom")
        ax[1].set_xticks(xs)
        ax[1].set_xlim(0.7, len(sv) + 0.45)
        ax[1].set_xlabel("replayed season", labelpad=1.5)
        ax[1].set_ylabel("bill reduction vs.\nfixed target (%)", linespacing=1.25)
        print(f"   bandit saving by season: {np.round(sv, 1).tolist()}"
              f"   oracle {orc:.1f}%")
    except FileNotFoundError:
        ax[1].text(0.5, 0.5, "run bandit_seasons_save.py",
                   ha="center", va="center", fontsize=7, color="#C1121F",
                   transform=ax[1].transAxes)
        print("   [warn] bandit_seasons.npz missing; run bandit_seasons_save.py")

    save(fig, "bandit_arms")
    print("   late-half medians:",
          [round(float(np.median(h[n//2:, i])), 2) for i in (0, 1)])


# ==========================================================================
# the trade-off plane                                 -- SUPPLEMENTARY
# ==========================================================================
CLUSTER = ["modular_local", "cqr_local", "adaptive_modular", "adaptive_cqr"]

CLAB = {"integrated": "integrated", "fixed@0": "fixed@0",
        "modular_fed": r"modular$_\mathrm{fed}$",
        "modular_local": r"modular$_\mathrm{local}$",
        "cqr_local": r"CQR$_\mathrm{local}$",
        "adaptive_modular": r"adaptive$_\mathrm{modular}$",
        "adaptive_cqr": r"adaptive$_\mathrm{CQR}$",
        "adaptive_signed": "adaptive, signed"}
CCOL = {"integrated": "#D4537E", "fixed@0": "#F0A93B",
        "modular_fed": "#9AA7B5", "modular_local": "#378ADD",
        "cqr_local": "#7F77DD", "adaptive_modular": "#1D9E75",
        "adaptive_cqr": "#0B6E4F", "adaptive_signed": "#B5179E"}
CMKR = {"integrated": "s", "fixed@0": "X", "modular_fed": "P",
        "modular_local": "^", "cqr_local": "D",
        "adaptive_modular": "o", "adaptive_cqr": "v",
        "adaptive_signed": "*"}


def fig_cost():
    d = pd.read_csv(os.path.join(OUT, "cost_table_cp.csv"), index_col=0)
    H = {}; P = {}; C = {}
    for o in d.index:
        H[o] = float(d.loc[o, "hold_B"])
        P[o] = float(d.loc[o, "penalty"])
        C[o] = float(d.loc[o, "PICP80"])

    try:
        b = baselines()
        g = b[(b.scheme == "acl_asym") & (b.level == 1.0)]
        if not g.empty:
            g = g.iloc[0]
            H["adaptive_signed"] = float(g.hold_B)
            P["adaptive_signed"] = float(g.penalty)
            C["adaptive_signed"] = float(g.PICP80)
    except FileNotFoundError:
        print("   [warn] results_baselines.csv missing, no signed point")

    order = [o for o in ["integrated", "fixed@0", "modular_fed",
                         "modular_local", "cqr_local", "adaptive_modular",
                         "adaptive_cqr", "adaptive_signed"] if o in H]
    T = {o: H[o] + P[o] for o in order}

    fig, ax = plt.subplots(figsize=(COL, 3.5))
    xmax = max(H.values()) * 1.45
    ymax = max(P.values()) * 1.16

    for c in np.unique(np.round(
            np.linspace(min(T.values()), max(T.values()), 4) / 50) * 50):
        ax.plot([0, c], [c, 0], ls=":", c="0.78", lw=0.8, zorder=0)
        if c < ymax * 0.96:
            ax.text(xmax * 0.015, c + ymax * 0.008, f"{c:.0f} USD",
                    fontsize=5.5, color="0.55", ha="left", va="bottom")

    for o in order:
        ax.scatter(H[o], P[o], s=58 if o == "adaptive_signed" else 42,
                   color=CCOL[o], marker=CMKR[o], zorder=3, clip_on=False)

    main_off = {"integrated": ((13, -4), "left", "top"),
                "fixed@0": ((13, -4), "left", "top"),
                "modular_fed": ((13, -4), "left", "top"),
                "adaptive_signed": ((-9, 7), "right", "bottom")}
    for o, (dx, hal, val) in main_off.items():
        if o not in T:
            continue
        ax.annotate(f"{CLAB[o]} · {T[o]:.0f} USD\nPICP {C[o]:.2f}",
                    (H[o], P[o]), textcoords="offset points", xytext=dx,
                    fontsize=6.0, color=CCOL[o], ha=hal, va=val,
                    linespacing=1.35, zorder=12,
                    bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.82))

    if "integrated" in T:
        ax.annotate("under-provisioned", (H["integrated"], P["integrated"]),
                    textcoords="offset points", xytext=(-4, 11),
                    fontsize=6.4, color="#D4537E", ha="left")
    if "modular_fed" in T:
        ax.annotate("over-provisioned", (H["modular_fed"], P["modular_fed"]),
                    textcoords="offset points", xytext=(-4, 11),
                    fontsize=6.4, color="#8894A2", ha="right")

    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    ax.set_xlabel(r"holding cost $c_h$ (USD/day)")
    ax.set_ylabel(r"imbalance cost $c_p$ (USD/day)")

    cl = [o for o in CLUSTER if o in T]
    if len(cl) >= 2:
        hx = [H[o] for o in cl]; py = [P[o] for o in cl]
        xr = max(max(hx) - min(hx), 10)
        yc = 0.5 * (max(py) + min(py))
        yh = max((max(py) - min(py)) * 0.5, 2.0)
        ix = (min(hx) - 0.34 * xr, max(hx) + 0.34 * xr)
        iy = (yc - yh * 12.0, yc + yh * 9.0)

        ins = ax.inset_axes([0.545, 0.52, 0.45, 0.45])
        for c in np.arange(np.floor(min(T[o] for o in cl) / 10) * 10,
                           max(T[o] for o in cl) + 20, 20):
            ins.plot([0, c], [c, 0], ls=":", c="0.80", lw=0.7, zorder=0)
        for o in cl:
            ins.scatter(H[o], P[o], s=34, color=CCOL[o], marker=CMKR[o],
                        zorder=3)
        ins_off = {"adaptive_cqr": ((0, 17), "left"),
                   "adaptive_modular": ((0, 6), "left"),
                   "cqr_local": ((0, -13), "right"),
                   "modular_local": ((0, -25), "right")}
        for o in cl:
            dx, hal = ins_off.get(o, ((0, 10), "left"))
            ins.annotate(f"{CLAB[o]} · {T[o]:.0f} USD", (H[o], P[o]),
                         textcoords="offset points", xytext=dx,
                         fontsize=5.6, color=CCOL[o], ha=hal)
        ins.set_xlim(*ix); ins.set_ylim(*iy)
        ins.tick_params(labelsize=5.2, length=2, pad=1.5)
        ins.set_xticks([int(round(min(hx) / 10) * 10),
                        int(round(max(hx) / 10) * 10)])
        ins.set_yticks([])
        ins.grid(alpha=0.22, axis="x")
        ins.set_facecolor("white")
        for sp in ins.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.6); sp.set_color("0.6")
        ins.set_title(r"symmetric score, meets coverage",
                      fontsize=5.8, color="0.35", pad=2.5)
        ax.indicate_inset_zoom(ins, edgecolor="0.55", lw=0.7, alpha=0.85)

    save(fig, "cost_regimes")

    print("\n   calibration-attributable cost (scheduling excluded):")
    print(f"   {'scheme':<20}{'PICP':>6}{'hold':>7}{'imb':>6}"
          f"{'total':>7}{'vs best':>9}")
    best = min(T.values())
    for o in order:
        gap = "" if T[o] == best else f"{100*(T[o]-best)/T[o]:>8.0f}%"
        print(f"   {o:<20}{C[o]:>6.2f}{H[o]:>7.0f}{P[o]:>6.0f}"
              f"{T[o]:>7.0f}{gap:>9}")


for _f in (fig_dose, fig_regimes, fig_symmetry, fig_fm,
           fig_costopt, fig_bandit, fig_cost):
    try:
        _f()
    except FileNotFoundError as _e:
        print(f"   [skip] {_f.__name__}: {_e}")

print("\nfigures written to:", OUT)
print("LaTeX, in the paper:")
print("  symmetry      -> figure,  width=\\columnwidth")
print("  fig_costopt   -> figure,  width=\\columnwidth")
print("  bandit_arms   -> figure*, width=0.72\\textwidth")
print("LaTeX, supplementary:")
print("  doseresponse / regimes / fm_backbone -> figure*, width=\\textwidth")
print("  cost_regimes                         -> figure,  width=\\columnwidth")
print("Give \\includegraphics the stem with no extension so it takes the PDF.")
