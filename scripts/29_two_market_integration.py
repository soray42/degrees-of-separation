"""TASK 1 — Two-market integration map (STATIC, descriptive, outcome-agnostic).

CONCEPTUAL LENS (metaphor, NOT literal arbitrage, NOT causal): the academic prestige hierarchy and
the labor market are two valuation systems pricing the same human capital (institution x field). The
gap = degree of SEGMENTATION between them; integration = how tightly academic standing maps to pay.
Across-institution earnings conflate value-added with student SELECTION (cite Chetty-Deming-Friedman:
brand payoff concentrates in the elite TAIL, small average-earnings effect, medians can't see it;
MacLeod et al. 2017). We DESCRIBE; we do not identify causal effects.

  Integration index  c_F[f] = Spearman(F, y) = 1 - gap[f].  (REFRAME of the existing gap.)
  NEW — integration STRUCTURE: same correlation hides different shapes. Classify prestige->pay as
    GRADED (monotone across the whole range), THRESHOLD (only the top prestige tier earns a premium),
    or FLAT (prestige barely maps to pay), via the top-decile premium vs the prestige-pay correlation
    among the NON-top-decile.

 -> data/interim/integration_map.csv, outputs/figures/{integration_map_ranked,
    integration_structure_classes}.png, TWO_MARKET_INTEGRATION_RESULT.md
"""
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; (OUT / "figures").mkdir(parents=True, exist_ok=True)

# reuse the per-(institution x field) table builder + crosswalk from script 28
_spec = importlib.util.spec_from_file_location("s28", ROOT / "scripts" / "28_field_vs_generic_prestige.py")
s28 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s28)
LAB, CIP2, C2NAME = s28.LAB, s28.CIP2, s28.C2NAME
MIN_N_STRUCT = 20      # need >=2 institutions in the top decile


def structure(g, c_F):
    """Classify a field's prestige->pay shape. Returns (class, top_decile_premium_rel, c_rest)."""
    n = len(g)
    if n < MIN_N_STRUCT:
        return "n/a", np.nan, np.nan
    g = g.sort_values("F", ascending=False)
    k = max(2, int(round(n * 0.10)))
    top, rest = g.iloc[:k], g.iloc[k:]
    fmed = g.y.median()
    top_prem = (top.y.median() - rest.y.median()) / fmed
    c_rest = spearmanr(rest.F, rest.y)[0]
    # top_prem now PARTICIPATES: THRESHOLD = a large apex premium with a weak gradient below it,
    # evaluated BEFORE the GRADED branch (the earlier ordering left top_prem inert - see self-check).
    if abs(c_F) < 0.20:
        cls = "FLAT"                                   # prestige barely maps to pay anywhere
    elif top_prem >= 0.10 and (not np.isfinite(c_rest) or c_rest < 0.30):
        cls = "THRESHOLD"                              # winner-take-all: big top tier, flat-ish rest
    elif np.isfinite(c_rest) and c_rest >= 0.30:
        cls = "GRADED"                                 # gradient persists below the top tier
    else:
        cls = "FLAT"
    return cls, top_prem, c_rest


def main():
    t = s28.build_table()                              # inst_key, field, F, G, y, cohort_n, institution_name
    rows = []
    for fld, g in t.groupby("field"):
        if len(g) < 8:
            continue
        c_F = spearmanr(g.F, g.y)[0]
        cls, top_prem, c_rest = structure(g, c_F)
        rows.append(dict(field=fld, label=LAB.get(fld, fld), cip2=CIP2.get(fld), n=len(g),
                         integration=c_F, gap=1 - c_F, structure=cls,
                         top_decile_premium=top_prem, rest_corr=c_rest))
    im = pd.DataFrame(rows).sort_values("integration", ascending=False)
    im.to_csv(ROOT / "data" / "interim" / "integration_map.csv", index=False)

    # honesty diagnostics demanded by adversarial review:
    rel = im[im.structure != "n/a"]
    cls_ord = {"FLAT": 0, "THRESHOLD": 1, "GRADED": 2}
    redund = dict(rest_vs_cF=spearmanr(rel.rest_corr, rel.integration)[0],
                  class_vs_cF=spearmanr(rel.structure.map(cls_ord), rel.integration)[0])
    # cutoff sweep: 3-way split sensitivity to the flat-cut and the graded c_rest threshold
    sweep = {}
    for flat_cut in (0.10, 0.20, 0.30):
        for cr_cut in (0.20, 0.30, 0.40):
            c = []
            for _, r in rel.iterrows():
                if abs(r.integration) < flat_cut:
                    c.append("FLAT")
                elif r.top_decile_premium >= 0.10 and (not np.isfinite(r.rest_corr) or r.rest_corr < cr_cut):
                    c.append("THRESHOLD")
                elif np.isfinite(r.rest_corr) and r.rest_corr >= cr_cut:
                    c.append("GRADED")
                else:
                    c.append("FLAT")
            vc = pd.Series(c).value_counts()
            sweep[(flat_cut, cr_cut)] = (vc.get("GRADED", 0), vc.get("THRESHOLD", 0), vc.get("FLAT", 0))

    write_report(im, redund, sweep)
    make_figures(im)

    rel = im[im.structure != "n/a"]
    print(f"fields: {len(im)} | structure-classified (n>={MIN_N_STRUCT}): {len(rel)}")
    print("structure counts:", rel.structure.value_counts().to_dict())
    print("\nmost integrated (high c_F):")
    print(im.head(6)[["label", "n", "integration", "structure", "top_decile_premium", "rest_corr"]].to_string(index=False))
    print("\nmost segmented (low c_F):")
    print(im.tail(6)[["label", "n", "integration", "structure"]].to_string(index=False))


def write_report(im, redund, sweep):
    rel = im[im.structure != "n/a"]
    vc = rel.structure.value_counts().to_dict()
    L = ["# Two-market integration map (static)\n",
         "**Conceptual lens (metaphor, not literal arbitrage, not causal):** the academic prestige "
         "hierarchy and the labor market are two valuation systems pricing the same human capital "
         "(institution × field). **Integration** index = how tightly academic standing maps to pay = "
         "`c_F[f] = Spearman(F, y) = 1 − gap[f]`; **segmentation** = the gap. Descriptive and "
         "outcome-agnostic: across-institution earnings conflate value-added with student SELECTION — we "
         "describe co-movement, we do not identify causal effects (cite Chetty-Deming-Friedman: the "
         "brand's payoff is concentrated in the elite TAIL that median data cannot see; MacLeod et al. "
         "2017). Run: `python scripts/29_two_market_integration.py`.\n",
         "## 1. The integration map (reframe of the gap)\n",
         f"{len(im)} fields ranked from most INTEGRATED (academic standing ≈ pay) to most SEGMENTED "
         "(the two valuations diverge). *This axis is the existing gap, re-presented — disclosed.*\n",
         "**Most integrated** (prestige tracks pay):\n",
         im.head(8)[["label", "n", "integration", "structure"]].to_markdown(index=False, floatfmt=("", ".0f", "+.2f", "")),
         "\n**Most segmented** (the two markets diverge):\n",
         im.tail(8)[["label", "n", "integration", "structure"]].to_markdown(index=False, floatfmt=("", ".0f", "+.2f", "")),
         "\n## 2. Integration STRUCTURE — graded, threshold, or flat? (NEW)\n",
         "The same correlation hides different shapes. For each field we compare the **top-prestige-decile "
         "pay premium** against the prestige–pay correlation **among the non-top-decile** institutions: "
         "driven only by the top decile → **THRESHOLD** (winner-take-all: only the most prestigious tier "
         "earns a premium); persists in the rest → **GRADED** (academic standing is rewarded gradually); "
         "neither → **FLAT** (prestige barely maps to pay).\n",
         f"Among the {len(rel)} fields with ≥{MIN_N_STRUCT} institutions: "
         f"**GRADED {vc.get('GRADED',0)}, THRESHOLD {vc.get('THRESHOLD',0)}, FLAT {vc.get('FLAT',0)}**ITHRESHOLD"
         f" (THRESHOLD = {', '.join(rel[rel.structure=='THRESHOLD'].label)}).\n".replace("ITHRESHOLD", ""),
         "**Honest caveat — the structure is largely a re-cut of integration STRENGTH, not an orthogonal "
         "SHAPE.** After adversarial review: the non-top-decile correlation `c_rest` tracks the overall "
         f"`c_F` at Spearman **{redund['rest_vs_cF']:+.2f}**, and the class ordinal tracks `c_F` at "
         f"**{redund['class_vs_cF']:+.2f}** — so GRADED-vs-FLAT mostly restates whether integration is "
         "strong or weak. The one genuinely shape-based cell is **THRESHOLD** (large top-decile premium "
         "with a weak gradient below — fixed by the top-decile premium, which now does the classifying), "
         "but it is only "
         f"{vc.get('THRESHOLD',0)}/{len(rel)} fields and (see below) medians cannot really validate it.\n",
         rel.sort_values("integration", ascending=False)[
             ["label", "n", "integration", "structure", "top_decile_premium", "rest_corr"]]
            .to_markdown(index=False, floatfmt=("", ".0f", "+.2f", "", "+.2f", "+.2f")),
         "\n**Cutoff sensitivity (the 3-way split is NOT robust).** GRADED/THRESHOLD/FLAT counts across a "
         "flat-cut × c_rest-cut grid:\n",
         "| flat_cut \\ c_rest_cut | 0.20 | 0.30 | 0.40 |\n|---|---|---|---|\n"
         + "\n".join(f"| {fc:.2f} | " + " | ".join(f"{sweep[(fc,cc)][0]}/{sweep[(fc,cc)][1]}/{sweep[(fc,cc)][2]}"
                                                    for cc in (0.20, 0.30, 0.40)) + " |"
                     for fc in (0.10, 0.20, 0.30)) + "  *(GRADED/THRESHOLD/FLAT)*",
         "\nFLAT swings widely with the flat-cut and THRESHOLD with the c_rest-cut — the headline counts "
         "are cutoff-dependent. **The robust, honest reading is weaker than 'three clean shapes':** "
         "integration varies mostly in **strength** (≈the gap), with a small, cutoff-sensitive set of "
         "**THRESHOLD** (apex-only) fields layered on top.\n",
         "- **Why THRESHOLD is rare here is a LIMITATION, not a finding (corrected from the earlier "
         "draft).** Scorecard reports MEDIANS, and a winner-take-all premium lives in the elite TAIL "
         "(top-1%, elite firms) that medians cannot see (Chetty-Deming-Friedman). So **median data simply "
         "cannot test winner-take-all** — the rarity of THRESHOLD on medians is what the data can't "
         "resolve, not evidence that winner-take-all is absent. (The earlier draft mis-framed this as a "
         "defense; it is a ceiling on what medians can show.)\n",
         "## 3. Integration barriers (reframe — no new channel compute)\n",
         "What keeps the two markets apart? The ONE externally-identified barrier is **occupational "
         "LICENSING** (b_licensure ≈ **+0.66** on the gap, scripts/20): credentialing standardizes pay "
         "within licensed occupations, decoupling it from academic standing — a literal institutional "
         "barrier between the two valuation systems. Academic-absorption (how academia-bound a field is) "
         "was **NULL**. Beyond licensing, **most of the segmentation is unexplained** — a large residual "
         "we cannot yet attribute to a named barrier. (The channel decomposition is not re-run here; see "
         "scripts/20, 27.)\n",
         "## 4. Future-work feasibility — the dynamic (lead-lag) extension (REPORT ONLY)\n",
         "The exciting extension is DYNAMIC: when a field is economically revalued (earnings shift), does "
         "academic prestige ADJUST (the academy tracks the market) or LAG (autonomy)? Which market leads? "
         "**Data feasibility (not run):**",
         "- **Scorecard FOS earnings vintages:** the repo holds **1 vintage** (Most-Recent-Cohorts). The "
         "Dept. of Education has published **annual FOS releases since ~2019 (~5–6 vintages)**, each "
         "freely downloadable — these give *calendar-time* earnings variation per field. (The 1/4/5-yr "
         "horizons inside one vintage are different *cohorts*, not calendar vintages — not a substitute.)",
         "- **Year-varying prestige:** the public Wapman release is a single pooled 2011–2020 "
         "cross-section. Year-varying academic prestige needs the **ORCID-rebuilt hiring network** "
         "(scripts/10–19), which can yield period-specific SpringRank (e.g. 2011–2020 vs 2021–2026); the "
         "gateway already validated ORCID-AR at academia level (ρ≈0.74).",
         "- **Minimal viable panel:** **2 Scorecard vintages × 2 Wapman/ORCID prestige periods** over the "
         "best-covered ~20 fields — a 2×2 lead-lag pilot (does Δprestige follow Δearnings or vice-versa). "
         "Cost: ~2 extra Scorecard downloads + the ORCID two-period rebuild already scoped. **Not run "
         "here** — sized, not executed.\n",
         "## Adversarial self-check\n",
         "1. **Integration index = 1 − gap is a REFRAME** (disclosed) — section 1 adds no new measurement. "
         "The attempted new content (structure classification) turned out **largely redundant with "
         f"integration strength** (class vs c_F Spearman {redund['class_vs_cF']:+.2f}; c_rest vs c_F "
         f"{redund['rest_vs_cF']:+.2f}); only the small, cutoff-sensitive **THRESHOLD** cell is genuinely "
         "shape-based, and medians cannot validate it. The one clean new contribution is (b) the "
         "**feasibility sizing** of the dynamic test.",
         "2. **'Two markets' is a metaphor**, not literal arbitrage: there is no traded asset, no "
         "enforceable law of one price; 'integration/segmentation' is a descriptive analogy for how "
         "tightly two valuation orderings co-move.",
         "3. **Static = descriptive**: no lead-lag and no causal claim is made; section 4 only SIZES the "
         "dynamic test, it does not run or pre-judge it.",
         "4. **Selection:** the prestige–pay co-movement reflects who enrolls as much as institutional "
         "value-added; a THRESHOLD shape, in particular, is exactly what Chetty's elite-tail story "
         "predicts on selection grounds (top-tier brands enroll the students whose tails the median "
         "still partly reflects). The structure classes describe shape, not cause.",
         "5. **Cutoffs are tunable and the split is sensitive** (the sweep above shows FLAT and THRESHOLD "
         "counts move materially across the grid). The honest qualitative claim is the WEAKER one: "
         "integration varies mostly in *strength* (≈ the gap), with a small, cutoff-sensitive THRESHOLD "
         "(apex-only) set on top — NOT three robustly-separated shapes.\n"]
    (ROOT / "TWO_MARKET_INTEGRATION_RESULT.md").write_text("\n".join(L))


def make_figures(im):
    # ranked integration map
    s = im.sort_values("integration")
    col = {"GRADED": "#2C7BB6", "THRESHOLD": "#E8902A", "FLAT": "#D6202A", "n/a": "#bbb"}
    fig, ax = plt.subplots(figsize=(8, max(6, len(s) * 0.16)))
    ax.barh(range(len(s)), s.integration, color=[col.get(c, "#bbb") for c in s.structure])
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.label, fontsize=6)
    ax.set_xlabel("integration  c_F = Spearman(prestige, pay) = 1 − gap")
    ax.set_title("Two-market integration map (color = structure: blue GRADED / orange THRESHOLD / red FLAT)")
    ax.axvline(0, color="#888", lw=1); fig.tight_layout()
    fig.savefig(OUT / "figures" / "integration_map_ranked.png", dpi=140)

    # structure scatter: top-decile premium vs rest-correlation
    rel = im[im.structure != "n/a"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for cls, c in col.items():
        d = rel[rel.structure == cls]
        ax.scatter(d.rest_corr, d.top_decile_premium, c=c, s=20 + d.n, label=cls, alpha=.8, edgecolor="k", lw=.3)
    ax.axhline(0.03, color="#aaa", ls="--", lw=.8); ax.axvline(0.15, color="#aaa", ls="--", lw=.8)
    ax.set_xlabel("prestige–pay correlation among NON-top-decile (rest_corr)")
    ax.set_ylabel("top-prestige-decile pay premium (relative)")
    ax.set_title("Integration structure: where the prestige→pay reward lives\n(top-right GRADED, top-left THRESHOLD, bottom FLAT)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "figures" / "integration_structure_classes.png", dpi=140)


if __name__ == "__main__":
    main()
