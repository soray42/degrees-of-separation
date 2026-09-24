"""Main-text figures 1-5 of the paper (EPJ Data Science plan, PAPER_PLAN.md section 4).

Draws only. Nothing is re-estimated: every plotted number is either
  (i)  read from a table written by an analysis script (data/interim/*.csv / *.parquet, outputs/*.csv), with at most
       a simple aggregation (a mean over fields, a sort, a join), or
  (ii) copied from the RESULT write-up of the analysis script, for the interval estimates that exist only there
       (joint-bootstrap CIs of across-field means, cluster-bootstrap CIs of pooled within-institution betas).
       Those constants sit in PUB below, each with its source file and section. Where a point estimate can also be
       computed from a table, the script asserts that the two agree, so a published CI is only ever attached to the
       estimate it belongs to.

Figures (drawn at the printed text width, 165 mm, so font sizes are the printed sizes (at least 6 pt); vector PDF +
300-dpi PNG, one font family; series that must be told apart differ in lightness or marker shape as well as hue):
  paper_fig1  US coupling map with the control ladder per field (scripts/55, 58; outputs/expanded66_gap_map.csv)
  paper_fig2  US: (a) mean coupling ladder, (b) within-institution beta, (c) horse race, (d) c_F - c_G bracket
              (scripts/55, 56, 58)
  paper_fig3  UK: (a) subject coupling raw vs within-provider standardised, (b) share surviving by method,
              (c) within-band G vs selectivity (scripts/62)
  paper_fig4  career time: (a) US fixed cohorts y1/y5/y10, (b) UK YAG1/3/5, (c) b_F vs b_G by horizon
              (scripts/59, which reproduces scripts/52 on V4.13.0; scripts/62)
  paper_fig5  boundary condition: signed partial r(F, earn | geo) vs strict licensure share (scripts/64), and UK
              pay-scale subjects vs others, all graduates, full 33-subject sample (scripts/62)

Inputs (read-only):
  data/interim/selectivity_fields.csv   (scripts/55)      data/interim/selectivity_deep.csv       (scripts/58)
  data/interim/prestige_reliability.csv (scripts/56)      outputs/expanded66_gap_map.csv          (scripts/08/28 map)
  data/interim/pseo_refresh.csv         (scripts/59)      data/interim/uk_leo_summary.csv         (scripts/62)
  data/interim/uk_leo_subject_coupling.csv (scripts/62)   data/interim/licensing_battery.csv      (scripts/64)
  data/interim/acs_occ_anchors.parquet  (scripts/20; the strict licensure share scripts/64 uses)
  data/interim/career_time_coupling.csv (scripts/52; slope of the G coupling, two-stage CI)
Outputs: outputs/figures/paper_fig{1..5}.pdf and .png
No random numbers are drawn for the figures. PDF/PNG metadata dates are suppressed so re-runs are byte-identical.
Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
     .venv/bin/python scripts/70_paper_figures.py [--si-checks]

--si-checks (opt-in; draws no figure) prints the numbers the paper marks "SI check" that come from this script:
  (0) the canonical US sample sizes (fields, cells, institutions per field) from scripts/55's field table;
  (1) Scorecard-PSEO agreement of institution-by-field earnings levels, pooled and within field (V4.13.0, y5,
      pooled cohorts), with scripts/32's own loaders (reads data/raw/pseo/pseoe_all.csv.gz; about a minute);
  (2) per-field fixed-cohort career slopes (scripts/52, data/interim/career_time_coupling.csv and
      career_time_brand_field.csv) against the field's graduate-degree share (legacy scripts/35b,
      data/interim/er_axis_35b_deferral.csv); field bootstrap with seed 70, 4,000 draws;
  (3) UK pay-scale contrast on the 32-subject composition sample (scripts/62 subject table).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

HALO = [pe.withStroke(linewidth=2.2, foreground="white")]   # keeps call-out text legible over markers

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
FIGDIR = ROOT / "outputs" / "figures"

MM = 1 / 25.4
WIDTH = 165 * MM                      # printed text width of the manuscript (6.5 in); figures are included at 1:1

# which PSEO release the US career-time panels use (pseo_refresh.csv `variant`): "old" = V4.13.0 (2025Q4, the
# release scripts/52 used; PAPER_PLAN decision 6), "new" = V4.14.1 (2026Q2). Every statement holds in both.
PSEO_VARIANT = "old"
PSEO_LABEL = {"old": "PSEO V4.13.0", "new": "PSEO V4.14.1"}[PSEO_VARIANT]

# ------------------------------------------------------------------------------------------------------------------
# style
# ------------------------------------------------------------------------------------------------------------------
FONT = "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT, "mathtext.fontset": "dejavusans",
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "legend.frameon": False, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlepad": 4,
    "lines.linewidth": 1.0, "lines.markersize": 4, "errorbar.capsize": 0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "70",
    "savefig.dpi": 300, "figure.dpi": 100, "axes.unicode_minus": True,
})

# Okabe-Ito (colour-blind safe) for semantic series; F = blue, G = vermillion throughout
BLUE, VERM, GREEN, ORANGE, SKY, PURPLE = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7"
BLACK, DGREY, MGREY, LGREY = "#000000", "#4D4D4D", "#8C8C8C", "#D0D0D0"
# UK national-pay-scale subjects: Okabe-Ito orange, far lighter than the dark grey of the other subjects, so the two
# stay distinct under colour-vision deficiency; they also get a diamond marker (shape carries the same information)
PAY = ORANGE
PAY_TXT = "#9A6A00"                    # darker shade of PAY for text labels (legible on white)

# CIP-2 family colours (Paul Tol "muted", colour-blind safe). Families with 3+ fields in the 52-field map, plus the
# single-field families the text names (Computer/Info, Math & Stats, Education); the rest share grey.
C2NAME = {  # copied from scripts/24 (C2NAME), which scripts/28 and scripts/57 import
    "11": "Computer/Info", "27": "Math & Stats", "14": "Engineering", "40": "Physical Sci", "26": "Biological Sci",
    "45": "Social Sci", "42": "Psychology", "23": "English", "54": "History", "38": "Philosophy/Religion",
    "52": "Business", "51": "Health", "04": "Architecture/Planning", "01": "Agriculture", "03": "Nat. Resources",
    "13": "Education", "50": "Arts", "44": "Social Work", "30": "Nutrition/Interdisc.", "09": "Communication",
    "19": "Family/Consumer Sci", "16": "Foreign Languages", "31": "Parks/Kinesiology", "39": "Theology"}
FAM_COL = {"14": "#332288", "11": "#88CCEE", "27": "#44AA99", "40": "#117733", "52": "#999933", "45": "#DDCC77",
           "26": "#CC6677", "51": "#882255", "13": "#AA4499"}
FAM_ORDER = ["14", "11", "27", "40", "52", "45", "26", "51", "13"]
OTHER_COL = "#DDDDDD"                  # light enough to differ from Math & Stats (#44AA99) under protan/deutan vision


def fam_colour(c2):
    return FAM_COL.get(c2, OTHER_COL)


def panel_title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontweight="bold", fontsize=7.5)


def save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None,
                                                  "Creator": None})
    fig.savefig(FIGDIR / f"{name}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote outputs/figures/{name}.pdf and .png")


def close(a, b, tol, what):
    assert abs(a - b) <= tol, f"{what}: table gives {a:.5f}, write-up {b:.5f}"


def fmt(x, nd=2):
    return f"{x:+.{nd}f}".replace("-", "−")


# ------------------------------------------------------------------------------------------------------------------
# published interval estimates that exist only in the RESULT write-ups (point, lo, hi)
# ------------------------------------------------------------------------------------------------------------------
PUB = {
    # SELECTIVITY_DEEP_RESULT.md (scripts/58), "Key numbers": joint institution-bootstrap 95% CI of the field mean
    "raw52": (0.419, 0.355, 0.482),          # mean rho(F, Y), full sample, 52 fields
    "broad46": (0.141, 0.075, 0.208),        # mean rho(F, Y | SAT, ADM, inst. Pell, control, state level), 46 fields
    "broadG44": (0.059, 0.003, 0.114),       # mean rho(F, Y | broad + G), main field-specific spec, 44 fields
    # SELECTIVITY_DEEP_RESULT.md "Key numbers": within-institution beta, log points per within-field SD of F,
    # institution cluster-bootstrap percentile CI; main-spec cells = 2,889 programs, 168 institutions, 47 fields
    "fe_none_main": (0.0851, np.nan, np.nan),  # same cells, no institution FE ("+0.0851 (point)"; no CI reported)
    "fe_a_main": (0.0139, 0.0053, 0.0227),     # (a) on the SAT-covariate sample
    "fe_sg_main": (0.0118, 0.0037, 0.0205),    # (a) + field x {SAT, brand G}
    "fe_main": (0.0084, 0.0000, 0.0164),       # (a) + field x {SAT, ADM, inst. Pell, brand G} [main]
    # full scripts/55 sample, 3,416 programs, 198 institutions, 52 fields
    # SELECTIVITY_RESULT.md (scripts/55) within-institution table: no institution FE +0.084 (SE 0.006, CR1 by
    # institution) -> plotted as +/- 1.96 SE; SELECTIVITY_DEEP_RESULT.md Key numbers: (a) +0.0129 [+0.0048, +0.0212]
    "fe_none_full": (0.084, 0.084 - 1.96 * 0.006, 0.084 + 1.96 * 0.006),
    "fe_a_full": (0.0129, 0.0048, 0.0212),
    # SELECTIVITY_DEEP_RESULT.md section (iv) horse race: mean Shapley R2, joint-bootstrap 95% CI, 47 fields
    "hr4": {"F": (0.076, 0.058, 0.093), "S": (0.173, 0.146, 0.199), "G": (0.086, 0.066, 0.106),
            "IE": (0.201, 0.173, 0.230)},
    "hr3": {"F": (0.092, 0.071, 0.112), "S": (0.257, 0.212, 0.302), "G": (0.107, 0.081, 0.132)},
    # PRESTIGE_RELIABILITY_RESULT.md (scripts/56) Key numbers: mean ADV = c_F - c_G, institution bootstrap, 57 fields
    "adv_U": (-0.049, -0.085, -0.011),
    "adv_LB": (-0.005, -0.043, 0.044),       # F corrected with its public-edge reliability (lower bound)
    "adv_EXT": (-0.042, -0.077, -0.001),     # reliability extrapolated to the full hire count
    # SELECTIVITY_RESULT.md Answer 4 / brand table: net of selectivity (broad), 46 fields, "+0.00 [-0.05, +0.05]"
    "adv_sel": (0.00, -0.05, 0.05),
    # SELECTIVITY_DEEP_RESULT.md Answer 5: F | sel. set + G minus G | sel. set + F, 44 fields
    "adv_mirror": (-0.006, -0.120, 0.109),
}


# ------------------------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------------------------
def load_us_fields():
    s55 = pd.read_csv(INTERIM / "selectivity_fields.csv")
    s58 = pd.read_csv(INTERIM / "selectivity_deep.csv")
    gm = pd.read_csv(ROOT / "outputs" / "expanded66_gap_map.csv", dtype={"cip2": str})
    t = (s55[["field", "label", "n", "rho_raw", "se_rho_raw", "rho_raw_sat", "rho_pcs", "rho_pcs_adm",
              "rho_full_stlev", "se_rho_full_stlev", "rho_full_ie", "cG_full_stlev"]]
         .merge(s58[["field", "F_broadG", "se_F_broadG", "G_broadF", "broad", "raw",
                     "hr_sh_F", "hr_sh_S", "hr_sh_G", "hr_sh_IE", "hr_r2", "hr3_sh_F", "hr3_sh_S", "hr3_sh_G"]], on="field")
         .merge(gm[["field", "spearman", "cip2", "reliable"]], on="field"))
    t = t[t.n >= 15].reset_index(drop=True)                     # canonical US spec: n >= 15 institutions
    assert len(t) == 52 and t.rho_raw.notna().all()
    assert np.allclose(t.rho_raw, t.spearman, atol=1e-5)        # scripts/55 baseline = gap map
    assert np.allclose(t.rho_full_stlev.dropna(), t.broad.dropna(), atol=1e-5)   # scripts/58 reproduces 55
    return t


# ------------------------------------------------------------------------------------------------------------------
# Figure 1: US coupling map with the control ladder
# ------------------------------------------------------------------------------------------------------------------
def fig1(t):
    t = t.sort_values("rho_raw", ascending=True).reset_index(drop=True)   # bottom = lowest raw coupling
    k = len(t)
    close(t.rho_raw.mean(), PUB["raw52"][0], 5e-4, "fig1 mean raw")
    close(t.rho_full_stlev.mean(), PUB["broad46"][0], 5e-4, "fig1 mean broad")
    close(t.F_broadG.mean(), PUB["broadG44"][0], 5e-4, "fig1 mean broad+G")
    n_b, n_g = int(t.rho_full_stlev.notna().sum()), int(t.F_broadG.notna().sum())
    assert (n_b, n_g) == (46, 44)

    fig = plt.figure(figsize=(WIDTH, 168 * MM), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    lax = fig.add_subplot(gs[0, 1]); lax.axis("off")
    XL = (-1.0, 1.0)                            # a correlation axis; normal-approximation CIs are cut at +-1

    def ci(ax_, est, se, yy, **kw):
        lo, hi = np.clip(est - 1.96 * se, *XL), np.clip(est + 1.96 * se, *XL)
        ax_.plot([lo, hi], [yy, yy], solid_capstyle="butt", **kw)

    dz = 0.24                                   # vertical dodge within a row: raw / broad / broad + G
    ysum = k + 1.1                              # summary row above the fields
    for i, r in t.iterrows():
        col = fam_colour(r.cip2)
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F4F4F4", lw=0, zorder=0)
        ci(ax, r.rho_raw, r.se_rho_raw, i + dz, color=col, lw=0.9, zorder=2)
        ax.scatter(r.rho_raw, i + dz, s=5 + 0.33 * r.n, color=col, edgecolor="#222222", lw=0.35, zorder=4)
        if np.isfinite(r.rho_full_stlev):       # broad selectivity partial (scripts/55)
            ci(ax, r.rho_full_stlev, r.se_rho_full_stlev, i, color=MGREY, lw=0.6, zorder=2)
            ax.scatter(r.rho_full_stlev, i, s=9, marker="s", facecolor="white", edgecolor=DGREY, lw=0.6, zorder=4)
        if np.isfinite(r.F_broadG):             # selectivity + G (scripts/58)
            ci(ax, r.F_broadG, r.se_F_broadG, i - dz, color=BLACK, lw=0.6, zorder=2)
            ax.scatter(r.F_broadG, i - dz, s=8, marker="D", color=BLACK, lw=0, zorder=4)
    # summary row: published means with joint institution-bootstrap CIs (SELECTIVITY_DEEP_RESULT Key numbers).
    # Each mean gets a vertical line styled like its series (grey dashed / dark-grey dotted / black dash-dot) and a
    # short label at its foot, so the three lines can be told apart without matching positions to the top row.
    mean_style = {"raw52": (MGREY, (0, (4, 2)), "raw", -1.12), "broad46": (DGREY, (0, (1, 1.5)), "net of sel.", -0.62),
                  "broadG44": (BLACK, (0, (4, 1.5, 1, 1.5)), "+ G", -0.62)}
    for (key, yy, mk, fc, ec, sz) in [("raw52", ysum + dz, "o", MGREY, "#222222", 22),
                                      ("broad46", ysum, "s", "white", DGREY, 20),
                                      ("broadG44", ysum - dz, "D", BLACK, BLACK, 14)]:
        est, lo, hi = PUB[key]
        lc, ls, lab, ylab = mean_style[key]
        ax.plot([lo, hi], [yy] * 2, color=DGREY if key == "raw52" else ec, lw=1.2, zorder=2)
        ax.scatter(est, yy, s=sz, marker=mk, facecolor=fc, edgecolor=ec, lw=0.7, zorder=4)
        ax.plot([est, est], [-0.5, k - 0.5], color=lc, lw=0.7, ls=ls, zorder=1)
        ax.text(est + (0.012 if key != "broadG44" else -0.012), ylab, lab, fontsize=6.0, color=lc,
                ha="left" if key != "broadG44" else "right", va="top")
    ax.axhline(k + 0.35, color=DGREY, lw=0.5)
    ax.axvline(0, color=BLACK, lw=0.6, zorder=1)

    ax.set_yticks(list(range(k)) + [ysum])
    ax.set_yticklabels(list(t.label) + ["Mean across fields"], fontsize=6.3)
    for tl, rel in zip(ax.get_yticklabels(), list(t.reliable) + [True]):
        if rel:
            tl.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-1.9, ysum + 0.75)
    ax.set_xlim(XL[0] - 0.02, XL[1] + 0.02)
    xt = np.arange(-1.0, 1.01, 0.25)
    ax.set_xticks(xt)
    ax.set_xticklabels([fmt(v) if abs(v) > 1e-9 else "0" for v in xt])
    ax.set_xlabel("Within-field Spearman ρ of department prestige F with bachelor's earnings,\n"
                  "or partial ρ net of the stated controls")
    ax.spines["left"].set_visible(False)

    # legends in the side panel (the notes on controls, intervals and the bold labels are in the caption)
    hdr = {"weight": "bold", "size": 6.6}
    series = [Line2D([], [], marker="o", color=MGREY, mec="#222222", mew=0.4, ms=5, lw=0.9,
                     label=f"raw, 52 fields\nmean {fmt(PUB['raw52'][0], 3)}"),
              Line2D([], [], marker="s", color=MGREY, mfc="white", mec=DGREY, mew=0.7, ms=3.6, lw=0.6,
                     label=f"net of selectivity\n{n_b} fields\nmean {fmt(PUB['broad46'][0], 3)}"),
              Line2D([], [], marker="D", color=BLACK, ms=3.0, lw=0.6,
                     label=f"net of selectivity\nand G, {n_g} fields\nmean {fmt(PUB['broadG44'][0], 3)}")]
    l1 = lax.legend(handles=series, loc="upper left", bbox_to_anchor=(0.0, 1.0), title="Estimate (95% CI)",
                    title_fontproperties=hdr, labelspacing=0.9, alignment="left", fontsize=6.3)
    lax.add_artist(l1)
    present = set(t.cip2)
    fam = [Patch(facecolor=FAM_COL[c], edgecolor="#222222", lw=0.35,
                 label=f"{C2NAME[c]} ({int((t.cip2 == c).sum())})") for c in FAM_ORDER if c in present]
    fam.append(Patch(facecolor=OTHER_COL, edgecolor="#222222", lw=0.35,
                     label=f"other families ({int((~t.cip2.isin(FAM_ORDER)).sum())})"))
    l2 = lax.legend(handles=fam, loc="upper left", bbox_to_anchor=(0.0, 0.76), title="CIP-2 family (fields)",
                    title_fontproperties=hdr, alignment="left", handlelength=1.2, fontsize=6.3)
    lax.add_artist(l2)
    sizes = [Line2D([], [], marker="o", ls="", color=MGREY, mec="#222222", mew=0.35,
                    ms=np.sqrt(5 + 0.33 * nn), label=f"{nn}") for nn in (20, 60, 150)]
    lax.legend(handles=sizes, loc="upper left", bbox_to_anchor=(0.0, 0.44), title="Institutions (n)",
               title_fontproperties=hdr, alignment="left", labelspacing=0.9, fontsize=6.3)
    lax.text(0.0, 0.27, "Bold labels: fields\npassing the reliability\nflag (20 of 52)",
             transform=lax.transAxes, fontsize=6.2, va="top", ha="left", color=DGREY, linespacing=1.3)
    assert int(t.reliable.sum()) == 20
    save(fig, "paper_fig1")
    return t


# ------------------------------------------------------------------------------------------------------------------
# Figure 2: US decomposition
# ------------------------------------------------------------------------------------------------------------------
def fig2(t):
    fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 134 * MM), layout="constrained",
                            gridspec_kw={"width_ratios": [1.15, 1.0]})
    # (a) control ladder on the 44 fields estimable at every step
    ax = axs[0, 0]
    steps = ["rho_raw_sat", "rho_pcs", "rho_pcs_adm", "rho_full_stlev", "F_broadG"]
    labs = ["raw", "+ Pell,\ncontrol,\nstate level", "+ admit\nrate", "+ SAT", "+ G"]
    c = t.dropna(subset=steps + ["rho_full_ie"])
    assert len(c) == 44
    assert set(c.field) == set(t.dropna(subset=["F_broadG"]).field)
    means = c[steps].mean().values
    # SELECTIVITY_RESULT.md control-precision gradient (44 fields): +0.43 -> +0.31 -> +0.24 -> +0.14
    for v, p in zip(means[:4], (0.43, 0.31, 0.24, 0.14)):
        close(v, p, 5e-3, "fig2a ladder step")
    close(means[4], PUB["broadG44"][0], 5e-4, "fig2a +G step")
    x = np.arange(len(steps))
    for _, r in c.iterrows():
        ax.plot(x, r[steps].values.astype(float), color=LGREY, lw=0.45, zorder=1)
    ax.plot(x, means, color=BLACK, lw=1.4, marker="o", ms=4, zorder=3)
    est, lo, hi = PUB["broadG44"]
    ax.errorbar(x[-1], est, yerr=[[est - lo], [hi - est]], color=BLACK, lw=1.1, zorder=4)
    for xi, v in zip(x, means):
        ax.annotate(fmt(v), (xi, v), xytext=(4, 5), textcoords="offset points", fontsize=6.5, fontweight="bold")
    ie_mean = float(c.rho_full_ie.mean())      # the ladder's other last step (Table 2): + institution-wide earnings
    close(ie_mean, 0.12, 5e-3, "fig2a institution-earnings step")
    # (the caption says that Table 2's ladder ends with institution-wide earnings instead of G: ie_mean = +0.12)
    ax.axhline(0, color=BLACK, lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.2)
    ax.set_xlim(-0.3, len(steps) - 0.55)
    ax.set_ylabel("Partial ρ(F, earnings) within field")
    ax.set_ylim(-0.42, 0.95)
    ax.text(0.99, 0.99, f"{len(c)} fields estimable at every step (SAT sample)\n"
            "grey: one field; black: mean across fields;\nbar: bootstrap 95% CI of the mean (last step)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color=DGREY, linespacing=1.25)
    panel_title(ax, "a", "Mean coupling as controls are added")

    # (b) within-institution beta
    ax = axs[0, 1]
    specs = [("fe_none", "field FE\nonly"), ("fe_a", "+ inst.\nFE"),
             ("fe_sg", "+ field\nslopes:\nSAT, G"), ("fe", "main: +\nSAT, ADM,\nPell, G")]
    xs = np.arange(len(specs))
    for j, (key, _) in enumerate(specs):
        est, lo, hi = PUB[f"{key}_main"] if key != "fe" else PUB["fe_main"]
        if np.isfinite(lo):
            ax.errorbar(xs[j] + 0.1, est, yerr=[[est - lo], [hi - est]], fmt="o", color=BLACK, ms=4, lw=1.0)
        else:
            ax.plot(xs[j] + 0.1, est, "o", color=BLACK, ms=4)
            ax.annotate("point estimate\nonly", (xs[j] + 0.1, est), xytext=(6, -2), textcoords="offset points",
                        fontsize=6.0, color=DGREY, va="top")
        ax.annotate(fmt(est, 4), (xs[j] + 0.1, est), xytext=(6, 4), textcoords="offset points", fontsize=6.3,
                    fontweight="bold")
    for j, key in [(0, "fe_none_full"), (1, "fe_a_full")]:
        est, lo, hi = PUB[key]
        ax.errorbar(xs[j] - 0.12, est, yerr=[[est - lo], [hi - est]], fmt="s", mfc="white", mec=MGREY, color=MGREY,
                    ms=3.6, lw=0.9)
    ax.axhline(0, color=BLACK, lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([s[1] for s in specs], fontsize=6.1)
    ax.set_xlim(-0.45, len(specs) - 0.2)
    ax.set_ylim(-0.005, 0.1)
    ax.set_ylabel("β: change in log earnings\nper within-field SD of F")
    ax.legend(handles=[Line2D([], [], marker="o", color=BLACK, ms=4, lw=1.0,
                              label="2,889 programs,\n168 institutions, 47 fields"),
                       Line2D([], [], marker="s", color=MGREY, mfc="white", mec=MGREY, ms=3.6, lw=0.9,
                              label="3,416 programs,\n198 institutions, 52 fields")],
              loc="upper right", bbox_to_anchor=(1.0, 0.70), fontsize=6.0)
    ax.text(0.99, 0.40, "FE: fixed effects\n95% institution\ncluster-bootstrap CIs", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.0, color=DGREY, linespacing=1.25)
    panel_title(ax, "b", "Within-institution slope of F")

    # (c) horse race
    ax = axs[1, 0]
    blocks = [("F", "department\nprestige F", BLUE, "hr_sh_F", "hr3_sh_F"),
              ("S", "selectivity\n(SAT, −admit\nrate)", GREEN, "hr_sh_S", "hr3_sh_S"),
              ("G", "academia-\nwide\nprestige G", VERM, "hr_sh_G", "hr3_sh_G"),
              ("IE", "institution-\nwide\nearnings", PURPLE, "hr_sh_IE", None)]
    hr = t.dropna(subset=["hr_sh_F"])
    assert len(hr) == 47
    w = 0.36
    for j, (b, lab, col, c4, c3) in enumerate(blocks):
        est, lo, hi = PUB["hr4"][b]
        close(hr[c4].mean(), est, 6e-4, f"fig2c 4-block {b}")
        ax.bar(j - w / 2, est, w, color=col, edgecolor="#222222", lw=0.4)
        ax.errorbar(j - w / 2, est, yerr=[[est - lo], [hi - est]], color=BLACK, lw=0.8)
        if c3 is not None:
            est, lo, hi = PUB["hr3"][b]
            close(hr[c3].mean(), est, 6e-4, f"fig2c 3-block {b}")
            ax.bar(j + w / 2, est, w, color="white", edgecolor=col, hatch="//////", lw=0.8)
            ax.errorbar(j + w / 2, est, yerr=[[est - lo], [hi - est]], color=BLACK, lw=0.8)
    ax.set_xticks(range(len(blocks))); ax.set_xticklabels([b[1] for b in blocks], fontsize=6.2)
    ax.set_ylabel("Mean Shapley share of R² across fields")
    ax.set_ylim(0, 0.37)
    ax.legend(handles=[Patch(facecolor="white", edgecolor="#222222", lw=0.8,
                             label=f"solid bars: four blocks (mean R² {hr.hr_r2.mean():.2f})"),
                       Patch(facecolor="white", edgecolor="#222222", hatch="//////", lw=0.8,
                             label="hatched bars: without institution earnings")],
              loc="upper left", fontsize=6.2)
    panel_title(ax, "c", "Horse race: Shapley shares of R² (47 fields)")

    # (d) c_F - c_G with the reliability bracket
    ax = axs[1, 1]
    rel = pd.read_csv(INTERIM / "prestige_reliability.csv")
    rel = rel[rel.unit == "field"]
    assert len(rel) == 57
    close(rel.ADV_U.mean(), PUB["adv_U"][0], 5e-4, "fig2d ADV_U")
    close(rel.ADV_LB.mean(), PUB["adv_LB"][0], 5e-4, "fig2d ADV_LB")
    close(rel.ADV_EXT.mean(), PUB["adv_EXT"][0], 5e-4, "fig2d ADV_EXT")
    s = t.dropna(subset=["rho_full_stlev", "cG_full_stlev"])
    adv_sel = float((s.rho_full_stlev - s.cG_full_stlev).mean())
    close(adv_sel, PUB["adv_sel"][0], 5e-3, "fig2d ADV net of selectivity")
    m = t.dropna(subset=["F_broadG", "G_broadF"])
    close(float((m.F_broadG - m.G_broadF).mean()), PUB["adv_mirror"][0], 5e-4, "fig2d mirror")
    rows = [("adv_U", f"raw ({len(rel)} fields)", PUB["adv_U"][0]),
            ("adv_LB", "both corrected,\nlower-bound rel$_F$", PUB["adv_LB"][0]),
            ("adv_EXT", "both corrected,\nextrapolated rel$_F$", PUB["adv_EXT"][0]),
            ("adv_sel", f"net of selectivity\n({len(s)} fields)", adv_sel),
            ("adv_mirror", f"F | sel. + G minus\nG | sel. + F ({len(m)})", PUB["adv_mirror"][0])]
    ys = np.arange(len(rows))[::-1]
    ax.axvspan(PUB["adv_EXT"][0], PUB["adv_LB"][0], ymin=0.49, ymax=0.83, color=SKY, alpha=0.35, lw=0)
    ax.text((PUB["adv_EXT"][0] + PUB["adv_LB"][0]) / 2, ys[1] + 0.52, "reliability bracket", ha="center",
            va="bottom", fontsize=6.0, color=BLUE)
    for yy, (key, lab, est) in zip(ys, rows):
        _, lo, hi = PUB[key]
        nd = 2 if key == "adv_sel" else 3   # the net-of-selectivity interval is published to 2 decimals only
        shown = PUB[key][0] if key == "adv_sel" else est
        ax.errorbar(est, yy, xerr=[[est - lo], [hi - est]], fmt="o", color=BLACK, ms=3.6, lw=1.0)
        ax.annotate(f"{fmt(shown, nd)} [{fmt(lo, nd)}, {fmt(hi, nd)}]", (hi, yy), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=6.0)
    ax.axvline(0, color=BLACK, lw=0.6)
    ax.set_yticks(ys); ax.set_yticklabels([r[1] for r in rows], fontsize=6.2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.14, 0.28)
    ax.set_xticks([-0.1, 0.0, 0.1, 0.2])
    ax.set_xticklabels([fmt(v, 1) if v else "0" for v in (-0.1, 0.0, 0.1, 0.2)])
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel("c$_F$ − c$_G$: mean over fields of\nρ(F, earnings) − ρ(G, earnings)")
    ax.annotate("← G stronger", (-0.004, -0.5), fontsize=6.0, ha="right", color=DGREY)
    ax.annotate("F stronger →", (0.004, -0.5), fontsize=6.0, ha="left", color=DGREY)
    ax.text(0.99, 0.99, f"raw means: c$_F$ {fmt(rel.c_F.mean(), 3)}, c$_G$ {fmt(rel.c_G.mean(), 3)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color=DGREY)
    panel_title(ax, "d", "Field vs academia-wide prestige")
    save(fig, "paper_fig2")


# ------------------------------------------------------------------------------------------------------------------
# UK summary helper
# ------------------------------------------------------------------------------------------------------------------
def uk_summary():
    u = pd.read_csv(INTERIM / "uk_leo_summary.csv")

    def get(q, section=None):
        x = u[u.quantity == q]
        if section is not None:
            x = x[x.section == section]
        assert len(x) == 1, f"uk_leo_summary: {q!r} matched {len(x)} rows"
        r = x.iloc[0]
        return float(r.est), float(r.lo), float(r.hi), (int(r.k) if np.isfinite(r.k) else None), float(r.p)
    return get


# ------------------------------------------------------------------------------------------------------------------
# Figure 3: UK prior-attainment control
# ------------------------------------------------------------------------------------------------------------------
def fig3():
    U = uk_summary()
    sc = pd.read_csv(INTERIM / "uk_leo_subject_coupling.csv")
    d = sc.dropna(subset=["raw_c", "std_wp"]).sort_values("raw_c").reset_index(drop=True)
    assert len(d) == 32
    raw_m, std_m = U("raw coupling, composition sample"), U(
        "composition-standardised earnings (within-provider band gradients)")
    close(d.raw_c.mean(), raw_m[0], 5e-4, "fig3a raw_c mean")
    close(d.std_wp.mean(), std_m[0], 5e-4, "fig3a std_wp mean")

    fig = plt.figure(figsize=(WIDTH, 128 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    k = len(d)
    ysum = k + 1.0
    for i, r in d.iterrows():
        cat = r.setting if isinstance(r.setting, str) else ""
        pay = cat in ("core", "broad")
        col = PAY if pay else DGREY
        mk = "D" if pay else "o"
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F4F4F4", lw=0, zorder=0)
        ax.plot([r.raw_c, r.std_wp], [i, i], color=col, lw=0.7, zorder=2, alpha=0.9)
        ax.scatter(r.raw_c, i, s=16 if not pay else 14, marker=mk, color=col, edgecolor=col if not pay else "#222222",
                   lw=0.5, zorder=3)
        ax.scatter(r.std_wp, i, s=16 if not pay else 14, marker=mk, facecolor="white", edgecolor=col, lw=0.9,
                   zorder=3)
    for (est, lo, hi, kk, _), fc in [(raw_m, DGREY), (std_m, "white")]:
        ax.plot([lo, hi], [ysum + (0.18 if fc != "white" else -0.18)] * 2, color=DGREY, lw=1.1)
        ax.scatter(est, ysum + (0.18 if fc != "white" else -0.18), s=24, facecolor=fc, edgecolor=DGREY, lw=0.8,
                   zorder=3)
    ax.axhline(k - 0.35 + 0.5, color=DGREY, lw=0.5)
    ax.axvline(0, color=BLACK, lw=0.6)
    names = []
    for _, r in d.iterrows():
        nm = r.subject
        if r.setting == "core":
            nm = nm + " ◆"
        elif r.setting == "broad":
            nm = nm + " ◇"
        names.append(nm)
    ax.set_yticks(list(range(k)) + [ysum])
    ax.set_yticklabels(names + [f"Mean ({k} subjects), 95% CI"], fontsize=6.2)
    for tl, r in zip(ax.get_yticklabels(), list(d.setting) + [None]):
        if r in ("core", "broad"):
            tl.set_color(PAY_TXT)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.7, ysum + 0.8)
    ax.set_xlim(-0.12, 0.85)
    ax.set_xlabel("Spearman ρ(G, median earnings) across providers, YAG5\n(composition sample: 32 subjects)")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=DGREY, ms=4,
                               label=f"all graduates, mean {fmt(raw_m[0], 3)}"),
                        Line2D([], [], marker="o", ls="", mfc="white", mec=DGREY, ms=4,
                               label=f"standardised for attainment mix (within-provider gradients), "
                                     f"mean {fmt(std_m[0], 3)}"),
                        Line2D([], [], marker="D", ls="", color=PAY, mec="#222222", mew=0.5, ms=3.6,
                               label="national pay scales (◆), allied health (◇)")],
               loc="outside lower left", ncol=2, fontsize=6.2, handletextpad=0.3, columnspacing=1.0)
    panel_title(ax, "a", "Coupling by subject (UK)")

    # (b) share surviving by method
    ax = fig.add_subplot(gs[0, 1])
    rows = [(U("share surviving: within-band / matched"), "same band vs\nall graduates"),
            (U("share surviving, attenuation-adjusted"), "same band,\nnoise-corrected"),
            (U("share surviving: std_wp / raw_c"), "standardised,\nwithin-provider"),
            (U("share surviving: fe_std / fe_raw"), "standardised, free\ngradient (subset)"),
            (U("share surviving: std / raw_c"), "standardised, national\nmedians (over-corrects)")]
    ys = np.arange(len(rows))[::-1]
    for yy, ((est, lo, hi, kk, _), lab) in zip(ys, rows):
        floor = "national" in lab           # over-corrects: a floor, not an estimate
        col = MGREY if floor else VERM
        ax.barh(yy, 100 * est, height=0.62, color="white" if floor else col, edgecolor=col,
                hatch="//////" if floor else None, lw=0.8, alpha=1.0 if floor else 0.85)
        ax.errorbar(100 * est, yy, xerr=[[100 * (est - lo)], [100 * (hi - est)]], color=BLACK, lw=0.9)
        ax.text(104, yy, f"{100 * est:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]\n{kk} subjects",
                va="center", fontsize=6.0, linespacing=1.2)
    ax.axvline(100, color=BLACK, lw=0.5, ls=(0, (2, 2)))
    ax.set_yticks(ys); ax.set_yticklabels([r[1] for r in rows], fontsize=6.2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 136)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share surviving (%)")
    panel_title(ax, "b", "Share surviving")

    # (c) within-band G vs selectivity
    ax = fig.add_subplot(gs[1, 1])
    band = [U("within-band coupling (bands 1-5)"),
            U("within-band: Spearman(institution top-band share, band earnings)"),
            U("within-band: partial rho(G, band earnings | institution top-band share)"),
            U("within-band: partial rho(institution top-band share, band earnings | G)")]
    allg = [U("all-graduate coupling, same providers"),
            U("matched all-graduate: Spearman(institution top-band share, earnings)"),
            U("matched all-graduate: partial rho(G, earnings | institution top-band share)"),
            None]
    labs = ["ρ(G, pay)", "ρ(selectivity, pay)", "G | selectivity", "selectivity | G"]
    ys = np.arange(len(labs))[::-1]
    for yy, b, a in zip(ys, band, allg):
        est, lo, hi, _, _ = b
        ax.errorbar(est, yy + 0.13, xerr=[[est - lo], [hi - est]], fmt="o", color=VERM, ms=4, lw=1.0)
        if a is not None:
            est, lo, hi, _, _ = a
            ax.errorbar(est, yy - 0.13, xerr=[[est - lo], [hi - est]], fmt="o", mfc="white", mec=DGREY,
                        color=DGREY, ms=4, lw=1.0)
    ax.axvline(0, color=BLACK, lw=0.6)
    ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=6.3)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.1, 0.62)
    ax.set_ylim(-1.3, len(labs) + 0.45)
    ax.set_xlabel("Spearman or partial ρ, YAG5 earnings")
    ax.legend(handles=[Line2D([], [], marker="o", color=VERM, ms=4, lw=1.0,
                              label=f"same-band graduates ({band[0][3]} subjects)"),
                       Line2D([], [], marker="o", color=DGREY, mfc="white", mec=DGREY, ms=4, lw=1.0,
                              label="all graduates, same providers")],
              loc="upper left", fontsize=6.0, borderaxespad=0.1)
    # (selectivity = the institution's share of graduates with >= 360 UCAS points; A | B = partial rho net of B;
    #  both are defined in the caption)
    panel_title(ax, "c", "G vs intake selectivity")
    save(fig, "paper_fig3")


# ------------------------------------------------------------------------------------------------------------------
# Figure 4: career time
# ------------------------------------------------------------------------------------------------------------------
def fig4():
    P = pd.read_csv(INTERIM / "pseo_refresh.csv")
    P = P[P.variant == PSEO_VARIANT]

    def prow(section, stat, horizon=None, quantile=None):
        x = P[(P.section == section) & (P.stat == stat)]
        if horizon is not None:
            x = x[x.horizon == horizon]
        if quantile is not None:
            x = x[x["quantile"] == quantile]
        assert len(x) == 1, f"pseo_refresh: {stat} {horizon} {quantile} matched {len(x)}"
        r = x.iloc[0]
        return float(r.estimate), float(r.ci_lo), float(r.ci_hi), r.k

    U = uk_summary()
    TW = " [two-way field x institution cluster variance]"
    CR = " [crossed bootstrap]"
    fig = plt.figure(figsize=(WIDTH, 78 * MM), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[11, 6.6, 11])
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1], sharey=axa)
    axc = fig.add_subplot(gs[0, 2])
    yrs = {"y1": 1, "y5": 5, "y10": 10}

    # (a) US: mean coupling of F and of G with PSEO p50 earnings, crossed field x institution bootstrap CIs
    for stat, col, mk, off, lab in [("coupling_F", BLUE, "o", -0.14, "department prestige F"),
                                     ("coupling_G", VERM, "s", 0.14, "academia-wide prestige G")]:
        pts = [prow("quantile", stat + CR, h, "p50") for h in yrs]
        ks = {int(float(p[3])) for p in pts}
        assert len(ks) == 1, "US fixed-cohort horizons should share one field set"
        xs = np.array(list(yrs.values())) + off
        e = np.array([p[0] for p in pts]); lo = np.array([p[1] for p in pts]); hi = np.array([p[2] for p in pts])
        axa.errorbar(xs, e, yerr=[e - lo, hi - e], fmt=mk + "-", color=col, ms=3.8, lw=1.1,
                     mfc=col if mk == "o" else "white", mec=col, label=lab)
    k_us = ks.pop()
    sl = prow("career_time", "slope_all" + TW)
    # slope of the G coupling: scripts/52's table only (two-stage CI; scripts/59 reports no two-way CI for it)
    ct = pd.read_csv(INTERIM / "career_time_coupling.csv")
    rg = ct[(ct.record == "summary") & (ct.stat == "drhoG_all")]
    assert len(rg) == 1 and int(rg.k_fields.iloc[0]) == k_us
    rg = rg.iloc[0]
    axa.text(0.03, 0.985, f"within-cohort slope per year, {k_us} fields:\n"
             f"ρ(F): {fmt(sl[0], 3)} [{fmt(sl[1], 3)}, {fmt(sl[2], 3)}] (two-way)\n"
             f"ρ(G): {fmt(rg.estimate, 3)} [{fmt(rg.ci_lo, 3)}, {fmt(rg.ci_hi, 3)}] (two-stage)\n"
             f"bars: crossed field × institution bootstrap",
             transform=axa.transAxes, va="top", fontsize=6.0, color=DGREY, linespacing=1.3)
    axa.set_xticks([1, 5, 10]); axa.set_xlim(0, 11)
    axa.set_xlabel("Years since graduation")
    axa.set_ylabel("Mean within-field Spearman ρ(prestige, earnings)")
    axa.axhline(0, color=BLACK, lw=0.6)
    axa.legend(loc="lower right", fontsize=6.2)
    panel_title(axa, "a", "US, fixed cohorts (PSEO)")

    # (b) UK: rho(G, earnings) at YAG1/3/5 on balanced fixed-cohort panels, two-stage bootstrap CIs
    series = [("coupling at YAG{}", "all graduates", "o", "-", VERM, VERM,
               "career slope of rho(G, earnings) per year", -0.12),
              ("within-provider-standardised coupling at YAG{}", "standardised", "s", "--", VERM,
               "white", "career slope, within-provider-standardised", 0.0),
              ("within-band coupling at YAG{}", "same band", "^", ":", ORANGE, ORANGE,
               "career slope, within prior-attainment band", 0.12)]
    ks_uk = []
    for q, lab, mk, ls, col, fc, slq, off in series:
        pts = [U(q.format(y)) for y in (1, 3, 5)]
        e = np.array([p[0] for p in pts]); lo = np.array([p[1] for p in pts]); hi = np.array([p[2] for p in pts])
        s_ = U(slq)
        assert len({p[3] for p in pts}) == 1
        axb.errorbar(np.array([1, 3, 5]) + off, e, yerr=[e - lo, hi - e], fmt=mk, ls=ls, color=col, mfc=fc,
                     mec=col, ms=3.8, lw=1.1, label=f"{lab} ({pts[0][3]}):\n{fmt(s_[0], 3)}/yr")
        ks_uk.append(pts[0][3])
    axb.set_xticks([1, 3, 5]); axb.set_xlim(0, 6.2)
    axb.set_xlabel("Years since graduation")
    axb.tick_params(labelleft=False)
    axb.axhline(0, color=BLACK, lw=0.6)
    axb.legend(loc="lower right", fontsize=6.0, labelspacing=0.35, handlelength=2.0, borderaxespad=0.1,
               title="subjects in ( )", title_fontsize=6.0, frameon=True, facecolor="white", edgecolor="none",
               framealpha=1.0)
    s0 = U("career slope of rho(G, earnings) per year")
    axb.text(0.04, 0.985, f"slope, all graduates:\n{fmt(s0[0], 3)}/yr\n[{fmt(s0[1], 3)}, {fmt(s0[2], 3)}]\n"
             f"bars: two-stage\nbootstrap", transform=axb.transAxes, va="top", fontsize=6.0,
             color=VERM)
    panel_title(axb, "b", "UK (LEO)")
    axa.set_ylim(-0.24, 0.72)

    # (c) US: standardized rank-regression coefficients b_F and b_G per horizon, two-way CIs
    for coef, col, mk, off, lab in [("bF", BLUE, "o", -0.14, "b$_F$ (department prestige)"),
                                    ("bG", VERM, "s", 0.14, "b$_G$ (academia-wide prestige)")]:
        pts = [prow("career_time", f"{coef}_{h}_all" + TW) for h in yrs]
        e = np.array([p[0] for p in pts]); lo = np.array([p[1] for p in pts]); hi = np.array([p[2] for p in pts])
        axc.errorbar(np.array(list(yrs.values())) + off, e, yerr=[e - lo, hi - e], fmt=mk + "-", color=col,
                     mfc=col if mk == "o" else "white", mec=col, ms=3.8, lw=1.1, label=lab)
    dF, dG, dGF = (prow("career_time", s + TW) for s in ("dbF_all", "dbG_all", "dG_minus_dF_all"))
    axc.text(0.03, 0.985, f"slopes per year (two-way CI):\n"
             f"d$_F$ (of b$_F$) {fmt(dF[0], 3)} [{fmt(dF[1], 3)}, {fmt(dF[2], 3)}]\n"
             f"d$_G$ (of b$_G$) {fmt(dG[0], 3)} [{fmt(dG[1], 3)}, {fmt(dG[2], 3)}]\n"
             f"d$_G$ − d$_F$ {fmt(dGF[0], 3)} [{fmt(dGF[1], 3)}, {fmt(dGF[2], 3)}]\n"
             f"bars: two-way CI of b$_F$, b$_G$",
             transform=axc.transAxes, va="top", fontsize=6.0, color=DGREY, linespacing=1.3)
    axc.axhline(0, color=BLACK, lw=0.6)
    axc.set_xticks([1, 5, 10]); axc.set_xlim(0, 11)
    axc.set_ylim(-0.24, 0.72)
    axc.set_xlabel("Years since graduation")
    axc.set_ylabel("Rank-regression coefficient (standardised)")
    axc.legend(loc="lower right", fontsize=6.2)
    panel_title(axc, "c", "US: loading on F and on G")
    save(fig, "paper_fig4")
    return sl, s0, (dF, dG, dGF)


# ------------------------------------------------------------------------------------------------------------------
# Figure 5: boundary condition
# ------------------------------------------------------------------------------------------------------------------
def fig5(t):
    L = pd.read_csv(INTERIM / "licensing_battery.csv")
    SPEC, SAMPLE = "BEA region FE", "full (Scorecard state)"     # the version of the RESULT's family-means table
    pr = L[(L.section == "b_all_fields") & (L.stat == "partial_r(F,earn|geo)") & (L.spec == SPEC)
           & (L["sample"] == SAMPLE)][["field", "value", "ci_lo", "ci_hi", "n"]]
    anch = pd.read_parquet(INTERIM / "acs_occ_anchors.parquet")[["field", "licensure_strict"]]
    gm = pd.read_csv(ROOT / "outputs" / "expanded66_gap_map.csv", dtype={"cip2": str})[["field", "cip2"]]
    lab = pd.read_csv(INTERIM / "selectivity_fields.csv")[["field", "label"]]
    d = pr.merge(anch, on="field", how="left").merge(gm, on="field", how="left").merge(lab, on="field", how="left")
    d = d.dropna(subset=["value"])
    assert len(d) == 45 and d.licensure_strict.notna().all() and d.cip2.notna().all()

    def grad(stat):
        x = L[(L.section == "b_partial_gradient") & (L.field.isna()) & (L.spec == SPEC) & (L["sample"] == SAMPLE)
              & (L.stat == stat)]
        assert len(x) == 1
        return x.iloc[0]
    g_all = grad("spearman(licensure,partial_r)_signed")
    g_within = grad("spearman(licensure,partial_r)_signed_within_cip2")
    g_broad = grad("spearman(licensure,partial_r)_signed_broad")
    assert int(g_all.n) == len(d)
    # recompute the headline Spearman from the plotted points as a consistency check
    rs = pd.Series(d.licensure_strict.values).rank().corr(pd.Series(d.value.values).rank())
    close(rs, float(g_all.value), 1e-4, "fig5 Spearman(strict share, partial r)")

    fig = plt.figure(figsize=(WIDTH, 100 * MM), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    for _, r in d.iterrows():
        ax.plot([r.licensure_strict] * 2, [r.ci_lo, r.ci_hi], color=LGREY, lw=0.7, zorder=1)
    for _, r in d.iterrows():
        ax.scatter(r.licensure_strict, r.value, s=5 + 0.33 * r.n, color=fam_colour(r.cip2), edgecolor="#222222",
                   lw=0.35, zorder=3)
    offs = {"special_education": (6, 5, "left"), "accounting": (10, -13, "left"), "nursing": (-6, -10, "right"),
            "communication_disorders": (6, -10, "left"), "computer_science": (5, 6, "left"),
            "biology": (7, -5, "left")}
    for f, (dx, dy, ha) in offs.items():
        r = d[d.field == f].iloc[0]
        if f == "accounting":        # the low-share cluster is dense: place this label in the clear strip below it
            ax.annotate(r.label, (r.licensure_strict, r.value), xytext=(0.004, 0.205), textcoords="data",
                        fontsize=6.3, ha="left", va="center", fontweight="bold", path_effects=HALO,
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=DGREY, shrinkA=1, shrinkB=3))
            continue
        ax.annotate(r.label, (r.licensure_strict, r.value), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.3, ha=ha, fontweight="bold" if f in ("special_education", "accounting") else "normal",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=DGREY, shrinkA=0, shrinkB=2), path_effects=HALO)
    ax.axhline(0, color=BLACK, lw=0.6)
    ax.set_xlim(-0.02, 0.84)
    ax.set_ylim(-0.85, 1.30)                     # headroom above the data (max CI end +0.87) for the statistics
    ax.set_yticks([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Strict licensure share of the field's full-time workers\nwith a BA or higher (ACS 2023)")
    ax.set_ylabel("Partial r(F, earnings | BEA-region FE), 95% CI")
    ax.text(0.13, 0.99,                          # x in data units (strict share), y in axes units
            f"Spearman, {int(g_all.n)} fields: {fmt(g_all.value)} [{fmt(g_all.ci_lo)}, {fmt(g_all.ci_hi)}]\n"
            f"   CIP-2 family bootstrap [{fmt(g_all.ci_lo_cip2)}, {fmt(g_all.ci_hi_cip2)}]\n"
            f"within families {fmt(g_within.value)}; broad share {fmt(g_broad.value)}",
            transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=6.2,
            color=DGREY, linespacing=1.3)
    # marker-size key (area proportional to the number of institutions, as in Fig 1)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=MGREY, mec="#222222", mew=0.35,
                              ms=np.sqrt(5 + 0.33 * nn), label=f"{nn}") for nn in (20, 60, 150)],
              loc="lower left", title="institutions", title_fontsize=6.2, fontsize=6.2, ncol=3,
              handletextpad=0.2, columnspacing=0.9, borderaxespad=0.3)
    present = [c for c in FAM_ORDER if c in set(d.cip2)]
    fam = [Patch(facecolor=FAM_COL[c], edgecolor="#222222", lw=0.35, label=C2NAME[c]) for c in present]
    fam.append(Patch(facecolor=OTHER_COL, edgecolor="#222222", lw=0.35, label="other families"))
    fig.legend(handles=fam, loc="outside lower left", ncol=len(fam) // 2, fontsize=6.2, handlelength=1.0,
               columnspacing=1.0, title="Panel (a): CIP-2 family (marker area: number of institutions)",
               title_fontproperties={"weight": "bold", "size": 6.3}, alignment="left")
    panel_title(ax, "a", f"US: prestige–pay association net of geography ({len(d)} fields)")

    # (b) UK: national-pay-scale subjects vs others. One sample only: raw YAG5 coupling with G for all graduates on
    # the full 33-subject sample (column `raw`), with the group means and two-stage CIs of uk_leo_summary section e.
    # The attainment-standardised contrast (+0.053 vs +0.463) is on the 32-subject composition sample, where the raw
    # means are +0.057 and +0.543 (computed below and asserted), so it is given in the caption, not plotted here.
    ax = fig.add_subplot(gs[0, 1])
    U = uk_summary()
    sc_all = pd.read_csv(INTERIM / "uk_leo_subject_coupling.csv")
    sc = sc_all.dropna(subset=["raw"])
    assert len(sc) == 33
    core = sc[sc.setting == "core"]
    oth = sc[sc.setting != "core"]
    cm, om = U("raw coupling YAG5: setting-priced (core) mean"), U("raw coupling YAG5: others mean (vs core)")
    diff = U("raw coupling YAG5: setting-priced (core) minus others")
    close(core.raw.mean(), cm[0], 5e-4, "fig5b core mean")
    close(oth.raw.mean(), om[0], 5e-4, "fig5b others mean")
    comp = sc_all.dropna(subset=["raw_c", "std_wp"])          # the composition sample (32 subjects)
    sw_c, sw_o = (U("within-provider-standardised coupling YAG5: setting-priced (core) mean"),
                  U("within-provider-standardised coupling YAG5: others mean (vs core)"))
    close(comp[comp.setting == "core"].std_wp.mean(), sw_c[0], 5e-4, "fig5b std core mean (caption)")
    close(comp[comp.setting != "core"].std_wp.mean(), sw_o[0], 5e-4, "fig5b std others mean (caption)")
    print(f"fig5b caption: composition sample ({len(comp)} subjects) raw_c means "
          f"{comp[comp.setting == 'core'].raw_c.mean():+.3f} (core) / {comp[comp.setting != 'core'].raw_c.mean():+.3f} "
          f"(others, {int((comp.setting != 'core').sum())}); std_wp {sw_c[0]:+.3f} / {sw_o[0]:+.3f}")
    for xg, grp in [(0, core), (1, oth)]:
        g = grp.sort_values("raw").reset_index(drop=True)
        offs_ = ((np.arange(len(g)) % 5) - 2) * 0.055 if len(g) > 5 else (np.arange(len(g)) - 1) * 0.07
        for j, r in g.iterrows():
            pay = r.setting in ("core", "broad")
            ab = r.setting == "broad"
            col = PAY if pay else DGREY
            ax.scatter(xg + 0.2 + offs_[j], r.raw, s=15 if pay else 13, marker="D" if pay else "o",
                       facecolor="white" if ab else col, edgecolor="#222222" if (pay and not ab) else col,
                       lw=0.6 if not ab else 0.9, zorder=3)
            if pay:
                name = {"Nursing and midwifery": "Nursing", "Medicine and dentistry": "Medicine",
                        "Education and teaching": "Teaching", "Allied health": "Allied health"}[r.subject]
                ax.annotate(name, (xg + 0.2 + offs_[j], r.raw), xytext=(5, 0), textcoords="offset points",
                            fontsize=6.2, va="center", color=PAY_TXT, path_effects=HALO)
    for xg, (est, lo, hi, kk, _) in [(0, cm), (1, om)]:
        ax.errorbar(xg - 0.15, est, yerr=[[est - lo], [hi - est]], fmt="s", color=BLACK, ms=4, lw=1.1, zorder=4)
    ax.axhline(0, color=BLACK, lw=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"pay-scale\n({len(core)})", f"other\n({len(oth)})"], fontsize=6.4)
    ax.set_xlim(-0.5, 1.8)
    ax.set_ylim(-0.25, 0.95)
    ax.set_ylabel("Spearman ρ(G, median earnings), YAG5,\nall graduates (33 subjects)")
    ax.legend(handles=[Line2D([], [], marker="s", color=BLACK, ms=4, lw=1.1, label="group mean, 95% CI"),
                       Line2D([], [], marker="D", ls="", color=PAY, mec="#222222", mew=0.6, ms=3.6,
                              label="pay-scale subject"),
                       Line2D([], [], marker="D", ls="", mfc="white", mec=PAY, mew=0.9, ms=3.6,
                              label="allied health"),
                       Line2D([], [], marker="o", ls="", color=DGREY, ms=3.6, label="other subject")],
              loc="upper left", fontsize=6.0)
    ax.text(0.98, 0.02, f"difference {fmt(diff[0], 3)}\n[{fmt(diff[1], 3)}, {fmt(diff[2], 3)}],\n"
            f"permutation p = {diff[4]:.4f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0,
            color=DGREY)
    panel_title(ax, "b", "UK: pay-scale subjects")
    save(fig, "paper_fig5")
    return d, g_all, g_within, g_broad, cm, om, diff


def si_checks():
    """Numbers marked 'SI check' in the paper that are computed here (see the module docstring)."""
    import importlib.util
    import sys
    from scipy.stats import spearmanr
    sys.path.insert(0, str(ROOT))
    # (0) canonical US sample sizes from scripts/55's field table (n = matched institutions per field)
    sf = pd.read_csv(INTERIM / "selectivity_fields.csv")
    c15, c8 = sf[sf.n >= 15], sf[sf.n >= 8]
    print(f"(0) fields with n >= 15: {len(c15)}, cells {int(c15.n.sum())}, median n {c15.n.median():.0f}, "
          f"range {int(c15.n.min())}-{int(c15.n.max())}; n >= 8: {len(c8)} fields, {int(c8.n.sum())} cells")
    # (1) earnings-level agreement between the two US sources, scripts/32's loaders and join
    spec = importlib.util.spec_from_file_location("s32", ROOT / "scripts" / "32_pseo_gap_crossval.py")
    s32 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s32)
    t28 = s32.s28.build_table()[["inst_key", "field", "y"]].rename(columns={"y": "y_scorecard"})
    lvl = s32.load_pseo_ba("y5").merge(t28, on=["inst_key", "field"]).dropna(subset=["y_pseo", "y_scorecard"])
    pooled = spearmanr(lvl.y_pseo, lvl.y_scorecard)[0]
    wf = pd.DataFrame([(f, len(g), spearmanr(g.y_pseo, g.y_scorecard)[0])
                       for f, g in lvl.groupby("field") if len(g) >= 8], columns=["field", "n", "rho"])
    print(f"(1) level agreement: pooled Spearman {pooled:+.3f} over {len(lvl)} cells (scripts/32: +0.94, n=1535)")
    print(f"    within field, fields with >= 8 cells: {len(wf)} fields, {int(wf.n.sum())} cells; mean {wf.rho.mean():+.3f}, "
          f"median {wf.rho.median():+.3f}, range {wf.rho.min():+.3f} ({wf.loc[wf.rho.idxmin(), 'field']}) to "
          f"{wf.rho.max():+.3f} ({wf.loc[wf.rho.idxmax(), 'field']}), n-weighted {np.average(wf.rho, weights=wf.n):+.3f}")
    # (2) career slopes against the graduate-degree share
    ct = pd.read_csv(INTERIM / "career_time_coupling.csv")
    fs = ct[(ct.record == "field_slope") & (ct.source == "pseo_fixed_cohort")].copy()
    assert len(fs) == 39
    fs["seg_early"] = (fs.coupling_y5 - fs.coupling_y1) / 4
    fs["seg_late"] = (fs.coupling_y10 - fs.coupling_y5) / 5
    br = pd.read_csv(INTERIM / "career_time_brand_field.csv")
    br = br[br.record == "field_slope"][["field", "dbG", "dbF"]]
    dfr = pd.read_csv(INTERIM / "er_axis_35b_deferral.csv")[["field", "grad_degree_share", "deferral_class"]]
    m = fs.merge(br, on="field").merge(dfr, on="field").dropna(subset=["grad_degree_share"]).reset_index(drop=True)
    rng = np.random.default_rng(70)
    print(f"(2) career slopes vs graduate-degree share: {len(m)} fields")
    for col in ["slope_per_yr", "seg_early", "seg_late", "dbG", "dbF"]:
        r, p = spearmanr(m[col], m.grad_degree_share)
        bs = []
        for _ in range(4000):
            mm = m.iloc[rng.choice(len(m), len(m), replace=True)]
            if mm.grad_degree_share.nunique() > 2:
                bs.append(spearmanr(mm[col], mm.grad_degree_share)[0])
        print(f"    {col:13s} Spearman {r:+.2f} (p {p:.3f}); field bootstrap 95% "
              f"[{np.nanpercentile(bs, 2.5):+.2f}, {np.nanpercentile(bs, 97.5):+.2f}]")
    g = m.groupby("deferral_class")[["slope_per_yr", "seg_late", "dbG"]].agg(["mean", "count"])
    print(g.round(4).to_string())
    # (3) UK pay-scale contrast on the composition sample
    sc = pd.read_csv(INTERIM / "uk_leo_subject_coupling.csv").dropna(subset=["raw_c", "std_wp"])
    core = sc.setting == "core"
    print(f"(3) UK composition sample ({len(sc)} subjects): raw_c {sc[core].raw_c.mean():+.3f} (3 pay-scale) vs "
          f"{sc[~core].raw_c.mean():+.3f} ({int((~core).sum())} others); std_wp {sc[core].std_wp.mean():+.3f} vs "
          f"{sc[~core].std_wp.mean():+.3f}")


def main():
    import sys
    if "--si-checks" in sys.argv[1:]:
        si_checks()
        return
    t = load_us_fields()
    fig1(t)
    fig2(t)
    fig3()
    fig4()
    fig5(t)


if __name__ == "__main__":
    main()
