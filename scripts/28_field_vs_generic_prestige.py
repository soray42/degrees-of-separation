"""Field-specific vs generic (academia-wide) prestige as competing predictors of field earnings.

DESCRIPTIVE, OUTCOME-AGNOSTIC. No causal language. Earnings differences across institutions conflate
institutional value-added with student SELECTION; we DESCRIBE which prestige signal correlates with
observed median earnings, we do NOT identify causal effects. Causal backing is CITED (not
re-estimated): Chetty-Deming-Friedman (QJE 2026 / NBER w31492 — Ivy-Plus raises elite-TAIL outcomes,
much smaller effects on AVERAGE earnings); MacLeod-Riehl-Saavedra-Urquiola 2017 (reputation->earnings
via an exit-exam shock).

Per field f, across institutions i offering f with F, G, y all defined:
  y[i,f] = Scorecard BA median earnings (primary 4yr; robustness 1yr/5yr)
  F[i,f] = field-specific Wapman SpringRank prestige (primary: published field Rank; robustness:
           continuous SpringRank score recomputed from field edges)
  G[i]   = generic academia-wide Wapman SpringRank prestige (published Academia Rank), constant across f

NOTE ON THE PRESTIGE MEASURE: the public Wapman release ships the ORDINAL Rank per (institution x
field); the continuous per-field SpringRank score is NOT released, so the primary uses the published
rank (which is also what the gap uses, so c_F = 1 - gap exactly), and the continuous score recomputed
from edges is the robustness. Spearman analyses are rank-invariant, so this is moot for A/B1.

 -> data/interim/field_vs_generic.csv, outputs/figures/{prestige_agreement_by_field,
    field_reputation_advantage, betaF_vs_betaG_scatter, named_examples_cs}.png, FIELD_VS_GENERIC_RESULT.md
"""
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from src.crosswalks import fields as F
from src.crosswalks.institutions import normalize_institution_name
from src.load_ar import load_ar_wapman
from src.load_er import load_er_scorecard

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
RANKS = ROOT / "data" / "raw" / "wapman2022" / "ranks.csv"
MIN_N = 15

# audited 66-field set = ALL_FIELDS + the 12 recovered fields from scripts/24
_spec = importlib.util.spec_from_file_location("cw24", ROOT / "scripts" / "24_crosswalk_audit.py")
_cw24 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cw24)
FIELDS66 = F.ALL_FIELDS + _cw24.RECOVER
LAB = {f["key"]: f["label"] for f in FIELDS66}
CIP2 = {f["key"]: f["cip4"][0][:2] for f in FIELDS66}
C2NAME = _cw24.C2NAME


def load_generic():
    rk = pd.read_csv(RANKS)
    ac = rk[rk.TaxonomyLevel == "Academia"].copy()
    ac["inst_key"] = ac.InstitutionName.map(normalize_institution_name)
    ac = ac[ac.inst_key != ""].drop_duplicates("inst_key")
    return ac[["inst_key", "Rank"]].rename(columns={"Rank": "g_rank"})


def build_table(earn_col="EARN_MDN_4YR", count_col="EARN_COUNT_WNE_4YR"):
    ar = load_ar_wapman(fields=FIELDS66)            # inst_key, field, prestige_score = -field_rank
    er = load_er_scorecard(earn_col=earn_col, count_col=count_col, fields=FIELDS66)
    g = load_generic()
    t = ar.merge(er[["inst_key", "field", "earnings", "cohort_n"]], on=["inst_key", "field"]) \
          .merge(g, on="inst_key")
    t = t.rename(columns={"prestige_score": "F"})   # F = -field_rank (higher = more prestigious)
    t["G"] = -t.g_rank                               # G = -academia_rank (higher = more prestigious)
    t["y"] = t.earnings
    return t.dropna(subset=["F", "G", "y"])


def per_field(t):
    rows, insts = [], {}
    for fld, g in t.groupby("field"):
        n = len(g)
        if n < 8:
            continue
        rho_pp = spearmanr(g.F, g.G)[0]
        c_F = spearmanr(g.F, g.y)[0]                  # = 1 - gap[f]
        c_G = spearmanr(g.G, g.y)[0]                  # the NEW object
        rec = dict(field=fld, label=LAB.get(fld, fld), cip2=CIP2.get(fld), n=n,
                   rho_PP=rho_pp, c_F=c_F, c_G=c_G, ADV=c_F - c_G, gap=1 - c_F)
        # B2 horse-race: standardize F,G within field; WLS y ~ F + G weighted by cohort size
        if n >= MIN_N and g.F.std() > 0 and g.G.std() > 0:
            zF = (g.F - g.F.mean()) / g.F.std(); zG = (g.G - g.G.mean()) / g.G.std()
            w = g.cohort_n.fillna(1).clip(lower=1).values
            X = sm.add_constant(np.column_stack([zF, zG]))
            m = sm.WLS(g.y.values, X, weights=w).fit()
            # incremental R2: full minus drop-one
            r2_full = m.rsquared
            r2_noG = sm.WLS(g.y.values, sm.add_constant(zF.values), weights=w).fit().rsquared
            r2_noF = sm.WLS(g.y.values, sm.add_constant(zG.values), weights=w).fit().rsquared
            rec.update(beta_F=m.params[1], beta_G=m.params[2], se_F=m.bse[1], se_G=m.bse[2],
                       incR2_F=r2_full - r2_noG, incR2_G=r2_full - r2_noF, r2=r2_full)
        else:
            rec.update(beta_F=np.nan, beta_G=np.nan, se_F=np.nan, se_G=np.nan,
                       incR2_F=np.nan, incR2_G=np.nan, r2=np.nan)
        rows.append(rec)
        insts[fld] = g
    return pd.DataFrame(rows), insts


def classify(r):
    if not np.isfinite(r.beta_F) or not np.isfinite(r.beta_G):
        return "n/a"
    sigF = abs(r.beta_F) > 1.96 * r.se_F; sigG = abs(r.beta_G) > 1.96 * r.se_G
    if r.beta_G < 0 and sigG and r.beta_F > 0:
        return "brand-suppression"            # brand-elite UNDER-earn given field strength
    if sigF and not sigG:
        return "field-dominant"
    if sigG and not sigF:
        return "brand-dominant"
    if sigF and sigG:
        return "both"
    return "neither"


def divergence_table(g, name, k=6):
    """Within a field: rank institutions by field-strong/brand-weak (high-F/low-G) vs opposite."""
    d = g.copy()
    d["F_pct"] = d.F.rank(pct=True)      # high = more field-prestigious
    d["G_pct"] = d.G.rank(pct=True)      # high = more brand-prestigious
    d["div"] = d.F_pct - d.G_pct          # >0 field-strong/brand-weak; <0 brand-strong/field-weak
    d["name"] = d.institution_name
    hi = d.sort_values("div", ascending=False).head(k)   # field-strong, brand-medium
    lo = d.sort_values("div").head(k)                     # brand-strong, field-medium
    return hi, lo, d


def main():
    t = build_table()
    pf, insts = per_field(t)
    pf["classification"] = pf.apply(classify, axis=1)
    pf.to_csv(ROOT / "data" / "interim" / "field_vs_generic.csv", index=False)

    # robustness: horizons + continuous recompute
    horizons = {"1yr": ("EARN_MDN_1YR", "EARN_COUNT_WNE_1YR"),
                "5yr": ("EARN_MDN_5YR", "EARN_COUNT_WNE_5YR")}
    hz = {}
    for h, (ec, cc) in horizons.items():
        th = build_table(ec, cc); pfh, _ = per_field(th)
        hz[h] = pfh[["field", "ADV", "c_G", "beta_G"]].rename(columns={c: f"{c}_{h}" for c in ["ADV", "c_G", "beta_G"]})
    rob = pf[["field", "ADV", "c_G", "beta_G"]].merge(hz["1yr"], on="field").merge(hz["5yr"], on="field")
    adv_horizon_corr = {h: spearmanr(rob.ADV, rob[f"ADV_{h}"])[0] for h in horizons}
    cg_horizon_corr = {h: spearmanr(rob.c_G.dropna(), rob.loc[rob.c_G.notna(), f"c_G_{h}"])[0] for h in horizons}

    write_report(t, pf, insts, adv_horizon_corr, cg_horizon_corr)
    make_figures(pf, insts)

    rel = pf[pf.n >= MIN_N]
    print(f"fields: {len(pf)} (>= {MIN_N} inst: {len(rel)})")
    print(f"c_F = 1 - gap identity check: max|c_F-(1-gap)| = {(pf.c_F-(1-pf.gap)).abs().max():.2e}")
    print(f"mean ADV (c_F - c_G) = {pf.ADV.mean():+.2f}; ADV>0 in {(pf.ADV>0).sum()}/{len(pf)} fields")
    print(f"mean c_F={pf.c_F.mean():+.2f}, mean c_G={pf.c_G.mean():+.2f}")
    print("classification counts:", rel.classification.value_counts().to_dict())
    print(f"ADV horizon stability: {adv_horizon_corr}")
    print("\ntop field-reputation-advantage (ADV) fields:")
    print(pf.sort_values("ADV", ascending=False)[["label", "n", "c_F", "c_G", "ADV", "rho_PP"]].head(8).to_string(index=False))


def write_report(t, pf, insts, adv_hz, cg_hz):
    rel = pf[pf.n >= MIN_N].copy()
    # named examples
    def fmt_div(fldkey, title):
        if fldkey not in insts:
            return f"\n*{title}: field not in data.*\n"
        hi, lo, _ = divergence_table(insts[fldkey], title)
        out = [f"\n**{title}** (field prestige F, academia-wide brand G as percentiles; earnings = "
               "Scorecard median; schools strong in BOTH are not divergence cases and are excluded by "
               "construction here — these are the extremes of F_pct − G_pct):\n",
               "field-strong / brand-medium (high-F, low-G):",
               hi[["name", "F_pct", "G_pct", "y"]].to_markdown(index=False, floatfmt=("", ".2f", ".2f", ",.0f")),
               "\nbrand-strong / field-medium (low-F, high-G):",
               lo[["name", "F_pct", "G_pct", "y"]].to_markdown(index=False, floatfmt=("", ".2f", ".2f", ",.0f"))]
        med_hi, med_lo = hi.y.median(), lo.y.median()
        verdict = ("**field-strong schools OUT-earn brand-strong schools** "
                   if med_hi > med_lo else "**brand-strong schools OUT-earn field-strong schools** "
                   if med_lo > med_hi else "**no earnings difference** ")
        out.append(f"\n→ median earnings: field-strong ${med_hi:,.0f} vs brand-strong ${med_lo:,.0f} — "
                   f"{verdict}(${med_hi-med_lo:+,.0f}).")
        return "\n".join(out)

    cs_key = "computer_science"
    eng_key = "mechanical_engineering" if "mechanical_engineering" in insts else "electrical_engineering"
    # a high-gap field present in data
    hg = pf[pf.n >= MIN_N].sort_values("gap", ascending=False)
    hg_key = hg.iloc[0].field if len(hg) else None

    L = ["# Field-specific vs generic (academia-wide) prestige as predictors of field earnings\n",
         "**Descriptive, outcome-agnostic — no causal claims.** Across-institution earnings differences "
         "conflate value-added with student selection; we describe which prestige signal *correlates* "
         "with observed median earnings. Causal backing is CITED, not re-estimated: Chetty-Deming-"
         "Friedman (Ivy-Plus lifts elite-TAIL outcomes, small average-earnings effects) and MacLeod et "
         "al. 2017 (reputation→earnings via an exit-exam shock). Run: "
         "`python scripts/28_field_vs_generic_prestige.py`.\n",
         f"Audited 66-field crosswalk; per-(institution × field) table n={len(t)} rows, "
         f"{pf.shape[0]} fields ({len(rel)} with ≥{MIN_N} institutions). Primary: published Wapman ranks "
         "(F = field, G = academia-wide) + Scorecard 4yr BA median earnings.\n",
         "## Headline (outcome-agnostic — the hypothesis does NOT hold)\n",
         f"**On MEDIAN BA earnings, the generic academia-wide brand predicts pay at least as well as — "
         f"usually slightly better than — field-specific academic reputation.** Mean c_G "
         f"({pf.c_G.mean():+.2f}) ≥ mean c_F ({pf.c_F.mean():+.2f}); field reputation's advantage "
         f"ADV = c_F − c_G is **negative in {(pf.ADV<0).sum()}/{len(pf)} fields** (mean {pf.ADV.mean():+.2f}); "
         f"in the horse-race, **brand-dominant fields ({(rel.classification=='brand-dominant').sum()}) "
         f"outnumber field-dominant ({(rel.classification=='field-dominant').sum()})**. The "
         "'Georgia-Tech-CS beats Yale-CS' intuition does NOT generalize on medians: in CS the data-driven "
         "brand-strong divergence schools OUT-earn the field-strong ones (Analysis C). This is fully "
         "consistent with Chetty-Deming-Friedman — the brand's payoff is concentrated in elite-TAIL "
         "outcomes (top-1%, elite firms/grad school) that Scorecard MEDIANS cannot see — and with "
         "selection (generic prestige ≈ selectivity ≈ student composition). We describe correlations; we "
         "claim no causal effect. The fields where field reputation DOES lead are applied/vocational "
         "(engineering subfields, social work, special education), NOT CS.\n",
         "## A — Do the two prestige signals even agree? (prestige-only; not a function of earnings)\n",
         f"rho_PP[f] = Spearman(field prestige F, academia-wide prestige G). Mean **{pf.rho_PP.mean():+.2f}** "
         f"(median {pf.rho_PP.median():+.2f}); field and brand prestige are "
         + ("strongly" if pf.rho_PP.mean() > 0.7 else "moderately") + " aligned on average, but "
         "disagreement varies by field. **Highest-disagreement fields** (where 'which signal to trust' "
         "matters most):\n",
         pf.sort_values("rho_PP")[["label", "n", "rho_PP"]].head(10).to_markdown(index=False, floatfmt=("", ".0f", "+.2f")),
         "\n## B1 — Which signal predicts pay? Correlation-difference ADV = c_F − c_G\n",
         "`c_F = Spearman(F, y)` **= 1 − gap[f]** (DISCLOSED: this restates the gap). `c_G = Spearman(G, y)` "
         "is the NEW object (the gap says nothing about generic prestige). `ADV = c_F − c_G` = field "
         "reputation's predictive advantage over the generic brand for field pay.\n",
         f"- mean c_F = **{pf.c_F.mean():+.2f}** (=1−mean gap), mean c_G = **{pf.c_G.mean():+.2f}**, "
         f"mean **ADV = {pf.ADV.mean():+.2f}**; ADV>0 (field rep more informative) in "
         f"**{(pf.ADV>0).sum()}/{len(pf)}** fields, ADV<0 (brand more informative) in "
         f"{(pf.ADV<0).sum()}/{len(pf)}.",
         "\nFields ranked by field-reputation advantage (ADV):\n",
         pf.sort_values("ADV", ascending=False)[["label", "n", "c_F", "c_G", "ADV", "rho_PP"]]
            .to_markdown(index=False, floatfmt=("", ".0f", "+.2f", "+.2f", "+.2f", "+.2f")),
         "\n## B2 — Horse-race regression y ~ F + G (standardized, cohort-weighted)\n",
         "β_F is mechanically tied to the gap; the NEW content is **β_G** (residual pay-association of "
         "the generic brand conditional on field strength) and the β_F-vs-β_G contrast. Classification "
         f"(fields with ≥{MIN_N} institutions, n={len(rel)}):\n",
         "| classification | n fields | meaning |",
         "|---|---|---|",
         f"| field-dominant | {(rel.classification=='field-dominant').sum()} | β_F sig, β_G ~0 |",
         f"| brand-dominant | {(rel.classification=='brand-dominant').sum()} | β_G sig, β_F ~0 |",
         f"| both | {(rel.classification=='both').sum()} | both signals add |",
         f"| brand-suppression | {(rel.classification=='brand-suppression').sum()} | β_G<0: brand-elite UNDER-earn given field strength |",
         f"| neither | {(rel.classification=='neither').sum()} | neither sig |",
         f"\n- mean β_F = {rel.beta_F.mean():+.0f}, mean β_G = {rel.beta_G.mean():+.0f} (earnings $ per SD "
         f"of standardized prestige); mean incremental R²: field {rel.incR2_F.mean():.2f}, brand "
         f"{rel.incR2_G.mean():.2f}.",
         "\nPer-field β_F, β_G (≥{0} inst):\n".format(MIN_N),
         rel.sort_values("beta_G", ascending=False)[["label", "n", "beta_F", "se_F", "beta_G", "se_G", "classification"]]
            .to_markdown(index=False, floatfmt=("", ".0f", ",.0f", ",.0f", ",.0f", ",.0f", "")),
         "\n## C — Divergence institutions / named-example verification\n",
         "Honest test of the 'Georgia-Tech-CS vs Yale-CS' intuition: within a field, do high-F/low-G "
         "(field-strong, brand-medium) schools out-earn low-F/high-G (brand-strong, field-medium) "
         "schools? Schools strong in BOTH (MIT, Stanford, CMU, Berkeley) are not divergence cases.\n",
         "**Caveat on the data-driven divergence cases (important):** Georgia Tech is strong in BOTH CS "
         "and overall brand, so it is NOT a divergence case and is excluded by construction. The actual "
         "highest-(F−G) CS schools turn out to be **regional publics with only *median* CS prestige "
         "(F_pct ≈ 0.4–0.65) but very low brand**, vs brand-strong privates with median CS prestige — so "
         "this tests 'mid-field/low-brand public vs low-field/high-brand private', not 'CS-elite vs "
         "brand-elite', and it conflates geography and selection. Truly CS-elite-but-brand-weak schools "
         "are rare (CS-elite ⇒ usually brand-strong). Read the verdict in that light.",
         fmt_div(cs_key, "Computer Science (low-gap)"),
         fmt_div(eng_key, f"{LAB.get(eng_key, eng_key)} (low-gap)"),
         fmt_div(hg_key, f"{LAB.get(hg_key, hg_key)} (high-gap, contrast)") if hg_key else "",
         "\n## D — Cross-field synthesis\n",
         f"- c_F ≈ 1 − gap confirmed (max abs deviation {(pf.c_F-(1-pf.gap)).abs().max():.0e}). The "
         "non-tautological objects are **rho_PP, c_G/ADV, and β_G**.",
         f"- Does the brand's residual premium β_G have its own pattern? corr(β_G, gap) = "
         f"{spearmanr(rel.beta_G, rel.gap)[0]:+.2f}, corr(c_G, gap) = {spearmanr(pf.c_G, pf.gap)[0]:+.2f} "
         f"— {'the brand premium tracks the gap' if abs(spearmanr(rel.beta_G, rel.gap)[0])>0.4 else 'the brand premium is largely independent of the gap'}.",
         "\n## Robustness\n",
         f"- **Earnings horizon:** ADV rank-stability vs 4yr: 1yr {adv_hz['1yr']:+.2f}, 5yr {adv_hz['5yr']:+.2f}; "
         f"c_G stability 1yr {cg_hz['1yr']:+.2f}, 5yr {cg_hz['5yr']:+.2f}.",
         "- **Rank vs continuous SpringRank:** Spearman analyses (A, B1) are rank-invariant so identical; "
         "the published rank is primary (the continuous per-field score is not in the public release).",
         f"- **Field-size:** per-field n reported; B2 restricted to ≥{MIN_N} institutions; the cross-field "
         "synthesis weights/sizes points by n (figure).\n",
         "## Adversarial self-check\n",
         "1. **Tautology:** c_F and β_F are mechanically the gap (c_F = 1 − gap exactly). Nothing about "
         "field-prestige-predicts-pay is new. The genuinely new objects are **rho_PP** (do the signals "
         "agree — prestige-only, not earnings), **c_G / ADV** (does the generic brand predict pay, and "
         "does field rep beat it), and **β_G** (the brand's residual premium net of field strength).",
         "2. **Selection vs value-added:** earnings gaps across institutions conflate value-added with "
         "who enrolls. We make NO causal claim. Chetty-Deming-Friedman: Ivy-Plus shifts elite-tail "
         "outcomes with small AVERAGE-earnings effects — so a weak c_G on MEDIAN earnings is fully "
         "consistent with a large brand effect on tails we cannot see. MacLeod et al. 2017 shows "
         "reputation can causally raise earnings where employer information is scarce.",
         "3. **Median-only:** Scorecard gives MEDIAN earnings. Recognition / tail outcomes (top-1%, "
         "elite firms, elite grad school) — exactly where Chetty finds the brand matters — are NOT in "
         "these data and are not computed; cited to Chetty, not estimated. A null brand effect here is a "
         "null on MEDIANS only.",
         f"4. **Named examples:** the CS / engineering / high-gap tables above report the result whichever "
         "way it went (see the per-table verdicts) — not asserted.",
         "5. **Sensitivity:** horizon-stability and field-n are reported above; the published-rank vs "
         "continuous-score choice does not affect the rank-based headlines.\n"]
    (ROOT / "FIELD_VS_GENERIC_RESULT.md").write_text("\n".join([x for x in L if x]))


def make_figures(pf, insts):
    rel = pf[pf.n >= MIN_N].copy()
    # A: prestige agreement by field
    s = pf.sort_values("rho_PP")
    fig, ax = plt.subplots(figsize=(8, max(6, len(s) * 0.16)))
    ax.barh(range(len(s)), s.rho_PP, color=["#D6202A" if v < 0.5 else "#4878CF" for v in s.rho_PP])
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.label, fontsize=6); ax.set_xlabel("rho_PP = Spearman(field prestige, academia-wide prestige)")
    ax.set_title("A. Do the two prestige signals agree? (low = disagreement)"); fig.tight_layout()
    fig.savefig(OUT / "figures" / "prestige_agreement_by_field.png", dpi=140)

    # B1: ADV ranked (and c_F vs c_G)
    s = pf.sort_values("ADV")
    fig, ax = plt.subplots(figsize=(8, max(6, len(s) * 0.16)))
    ax.barh(range(len(s)), s.ADV, color=["#2C7BB6" if v > 0 else "#D6202A" for v in s.ADV])
    ax.axvline(0, color="#888", lw=1); ax.set_yticks(range(len(s))); ax.set_yticklabels(s.label, fontsize=6)
    ax.set_xlabel("ADV = c_F − c_G  (field-reputation advantage; >0 field rep more informative)")
    ax.set_title("B1. Field-reputation advantage over the generic brand"); fig.tight_layout()
    fig.savefig(OUT / "figures" / "field_reputation_advantage.png", dpi=140)

    # D: beta_F vs beta_G scatter
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sc = ax.scatter(rel.beta_F, rel.beta_G, c=rel.gap, s=20 + rel.n, cmap="RdYlBu_r", alpha=.8, edgecolor="k", lw=.3)
    plt.colorbar(sc, label="gap")
    ax.axhline(0, color="#888", ls="--", lw=1); ax.axvline(0, color="#888", ls="--", lw=1)
    lim = max(abs(rel.beta_F).max(), abs(rel.beta_G).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], ":", color="#bbb")
    for _, r in rel.iterrows():
        ax.annotate(r.label, (r.beta_F, r.beta_G), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("β_F  (field prestige, $ per SD)"); ax.set_ylabel("β_G  (generic brand, $ per SD, net of F)")
    ax.set_title("D. Field vs brand pay-association per field (size=n, color=gap)\nbelow diagonal = field beats brand")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "betaF_vs_betaG_scatter.png", dpi=140)

    # C: named examples CS scatter
    if "computer_science" in insts:
        g = insts["computer_science"].copy()
        g["F_pct"] = g.F.rank(pct=True); g["G_pct"] = g.G.rank(pct=True); g["div"] = g.F_pct - g.G_pct
        fig, ax = plt.subplots(figsize=(8, 6.5))
        sc = ax.scatter(g.G_pct, g.F_pct, c=g.y, s=30, cmap="viridis")
        plt.colorbar(sc, label="CS median earnings ($)")
        ax.plot([0, 1], [0, 1], ":", color="#bbb")
        for _, r in g.sort_values("div").head(6).iterrows():
            ax.annotate(r.institution_name, (r.G_pct, r.F_pct), fontsize=6, color="#D6202A")
        for _, r in g.sort_values("div", ascending=False).head(6).iterrows():
            ax.annotate(r.institution_name, (r.G_pct, r.F_pct), fontsize=6, color="#2C7BB6")
        ax.set_xlabel("generic brand percentile (G)"); ax.set_ylabel("CS field prestige percentile (F)")
        ax.set_title("C. CS: field-strong/brand-medium (blue, above diag) vs brand-strong/field-medium (red)")
        fig.tight_layout(); fig.savefig(OUT / "figures" / "named_examples_cs.png", dpi=140)


if __name__ == "__main__":
    main()
