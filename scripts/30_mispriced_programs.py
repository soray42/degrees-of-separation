"""TASK 2 — Mispriced programs: institution x field over-/under-valuation (DESCRIPTIVE).

Two valuation systems price the same human capital; per field we ask where they DISAGREE most.
Per-field valuation residual: within field, rank-residualize earnings y on field prestige F;
standardize within field.
  r[i,f] > 0  MARKET-OVERPERFORMING  (paid above its academic standing)
  r[i,f] < 0  ACADEMICALLY OVER-VALUED (academic standing exceeds market pay)

HARD CAVEAT (central, prominent): the residual conflates institutional value-added with student
SELECTION. A "market-overperforming" program may simply enrol abler/richer students. This is
"where the two valuations DISAGREE", NOT "this program adds value / is underrated". Cite
Chetty-Deming-Friedman (brand payoff in the elite TAIL, median data can't see it) & MacLeod 2017.
Named programs are ILLUSTRATIVE (institution x field earnings are imprecise); the DISTRIBUTION /
systematic pattern is the robust object.

 -> data/interim/valuation_residuals.csv, data/interim/institution_valuation.csv,
    outputs/figures/{over_undervalued_programs,institution_valuation_extremes}.png,
    MISPRICED_PROGRAMS_RESULT.md
"""
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
_spec = importlib.util.spec_from_file_location("s28", ROOT / "scripts" / "28_field_vs_generic_prestige.py")
s28 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s28)
LAB = s28.LAB
MIN_FIELD_N = 15          # min institutions in a field to residualize
MIN_COHORT = 30           # min Scorecard earnings-cohort size to SURFACE a named program
MIN_INST_FIELDS = 3       # min fields to rank an institution


def residualize(t, min_field_n=MIN_FIELD_N):
    """Within each field, rank-residualize y on F; standardize within field. Returns rows with r."""
    out = []
    for fld, g in t.groupby("field"):
        if len(g) < min_field_n:
            continue
        g = g.copy()
        rF = rankdata(g.F); ry = rankdata(g.y)
        b1, b0 = np.polyfit(rF, ry, 1)
        resid = ry - (b0 + b1 * rF)
        sd = resid.std(ddof=1)
        g["F_pct"] = g.F.rank(pct=True); g["G_pct"] = g.G.rank(pct=True)
        g["resid"] = (resid / sd) if sd > 0 else 0.0
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main():
    t = s28.build_table()
    r = residualize(t)
    r["label"] = r.field.map(LAB)
    r.to_csv(ROOT / "data" / "interim" / "valuation_residuals.csv", index=False)

    # institution-level aggregate (>= MIN_INST_FIELDS fields)
    inst = r.groupby("inst_key").agg(institution=("institution_name", "first"),
                                     mean_resid=("resid", "mean"), n_fields=("field", "nunique"),
                                     mean_G_pct=("G_pct", "mean")).reset_index()
    inst = inst[inst.n_fields >= MIN_INST_FIELDS].sort_values("mean_resid")
    inst.to_csv(ROOT / "data" / "interim" / "institution_valuation.csv", index=False)

    # horizon robustness: residual rank-stability across 1/4/5yr
    hz = {}
    for h, (ec, cc) in {"1yr": ("EARN_MDN_1YR", "EARN_COUNT_WNE_1YR"),
                        "5yr": ("EARN_MDN_5YR", "EARN_COUNT_WNE_5YR")}.items():
        th = s28.build_table(ec, cc); rh = residualize(th)
        hz[h] = rh.set_index(["inst_key", "field"]).resid
    base = r.set_index(["inst_key", "field"]).resid
    horizon_corr = {}
    for h, s in hz.items():
        j = pd.concat([base, s], axis=1, keys=["b", "h"]).dropna()
        horizon_corr[h] = spearmanr(j.b, j.h)[0]

    write_report(r, inst, horizon_corr)
    make_figures(r, inst)

    surf = r[r.cohort_n >= MIN_COHORT]
    print(f"programs: {len(r)} ({len(surf)} with cohort>={MIN_COHORT}); fields residualized: {r.field.nunique()}")
    print(f"institutions (>= {MIN_INST_FIELDS} fields): {len(inst)}")
    print(f"horizon residual rank-stability: {horizon_corr}")
    print("\nmost ACADEMICALLY OVER-VALUED programs (r<0, surfaced):")
    print(surf.nsmallest(6, "resid")[["institution_name", "label", "F_pct", "G_pct", "y", "resid"]].to_string(index=False))
    print("\nmost MARKET-OVERPERFORMING programs (r>0, surfaced):")
    print(surf.nlargest(6, "resid")[["institution_name", "label", "F_pct", "G_pct", "y", "resid"]].to_string(index=False))


def write_report(r, inst, hz):
    surf = r[r.cohort_n >= MIN_COHORT]
    ov = surf.nsmallest(12, "resid")[["institution_name", "label", "F_pct", "G_pct", "y", "resid", "cohort_n"]]
    op = surf.nlargest(12, "resid")[["institution_name", "label", "F_pct", "G_pct", "y", "resid", "cohort_n"]]
    L = ["# Mispriced programs — institution × field over-/under-valuation (descriptive)\n",
         "Two valuation systems price the same human capital (institution × field); per field we surface "
         "where they DISAGREE most. **Residual** = earnings rank-residualized on field prestige within "
         "field, standardized: **r>0 market-overperforming** (paid above academic standing), **r<0 "
         "academically over-valued** (academic standing exceeds market pay).\n",
         "> **HARD CAVEAT — read first.** The residual conflates institutional value-added with student "
         "**SELECTION**. A 'market-overperforming' program may simply enrol abler / richer / better-"
         "connected students — this is **where the two valuations DISAGREE**, NOT evidence that a program "
         "'adds value' or is 'underrated'. We make **no causal claim**. Chetty-Deming-Friedman: the "
         "brand's payoff is concentrated in the elite TAIL that median earnings cannot see; MacLeod et "
         "al. 2017 shows reputation can causally move pay where employer information is scarce. **Named "
         "programs below are ILLUSTRATIVE** — institution×field earnings are imprecise; the **DISTRIBUTION "
         "/ systematic pattern is the robust object**, individual names are not.\n",
         f"Run: `python scripts/30_mispriced_programs.py`. {len(r)} programs across "
         f"{r.field.nunique()} fields (≥{MIN_FIELD_N} inst/field); named programs require Scorecard "
         f"earnings-cohort ≥ {MIN_COHORT}.\n",
         "## Most ACADEMICALLY OVER-VALUED programs (r<0 — academic standing exceeds market pay)\n",
         "*Illustrative, not a verdict on any program (selection caveat above).*\n",
         ov.to_markdown(index=False, floatfmt=("", "", ".2f", ".2f", ",.0f", "+.2f", ".0f")),
         "\n## Most MARKET-OVERPERFORMING programs (r>0 — paid above academic standing)\n",
         "*Illustrative, not a verdict on any program (selection caveat above).*\n",
         op.to_markdown(index=False, floatfmt=("", "", ".2f", ".2f", ",.0f", "+.2f", ".0f")),
         "\n## Institution-level systematic valuation (the more robust object)\n",
         f"Mean standardized residual across each institution's fields (≥{MIN_INST_FIELDS} fields; "
         "single-field institutions are NOT ranked).\n",
         "**Most systematically MARKET-OVERPERFORMING institutions** (mean r>0 across their fields):\n",
         inst.nlargest(10, "mean_resid")[["institution", "n_fields", "mean_resid", "mean_G_pct"]]
            .to_markdown(index=False, floatfmt=("", ".0f", "+.2f", ".2f")),
         "\n**Most systematically ACADEMICALLY OVER-VALUED institutions** (mean r<0):\n",
         inst.nsmallest(10, "mean_resid")[["institution", "n_fields", "mean_resid", "mean_G_pct"]]
            .to_markdown(index=False, floatfmt=("", ".0f", "+.2f", ".2f")),
         f"\nNote the **mean_G_pct** column — if over-performing institutions are systematically high-"
         f"brand (or low-brand), that is the selection/brand signal, not value-added: corr(mean_resid, "
         f"mean_G_pct) = **{spearmanr(inst.mean_resid, inst.mean_G_pct)[0]:+.2f}** across institutions.\n",
         "## Robustness — does the disagreement survive across earnings horizons?\n",
         f"Program-level residual rank-stability vs the 4yr residual: **1yr {hz['1yr']:+.2f}, 5yr "
         f"{hz['5yr']:+.2f}**. " + ("Stable → the over/under-valuation is a robust disagreement, not a "
         "single-horizon artifact." if min(hz.values()) > 0.5 else "Only moderately stable → individual "
         "program residuals are partly horizon-specific noise; read the distribution, not the names.") + "\n",
         "## Selectivity netting (optional) — NOT FEASIBLE in-repo\n",
         "The optional 'residual net of selectivity' (y ~ F + admit-rate/SAT) is **not run**: no "
         "institution-level selectivity (ADM_RATE, SAT/ACT) is in the repo — the Scorecard FoS file is "
         "field-level and carries none, and IPEDS here is Completions only. So the selection confound "
         "**cannot be netted out** with current data — which makes the HARD CAVEAT load-bearing. To add "
         "it: pull the Scorecard institution file (ADM_RATE, SAT_AVG) and re-residualize y ~ F + "
         "selectivity (still descriptive; selection-on-unobservables would remain).\n",
         "## Adversarial self-check\n",
         "1. **Selection vs value-added (central):** the entire residual is contaminated by who enrols. "
         "The institution-level corr(mean_resid, mean brand percentile) above is a direct read on how "
         "much the 'mispricing' is just brand/selectivity sorting. We claim DISAGREEMENT between two "
         "valuations, never value-added or 'underrated'.",
         "2. **Named = illustrative, distribution = robust:** institution×field Scorecard earnings are "
         "noisy (small cohorts, privacy suppression); the named tables are examples, and we require "
         f"cohort ≥ {MIN_COHORT} to surface one. The robust objects are the residual DISTRIBUTION and the "
         "institution-level aggregate, not any single program.",
         "3. **Horizon stability** is reported above; unstable program residuals are noise.",
         "4. **Selectivity-netting** would change which programs top the list; it is not feasible here and "
         "is flagged, not silently skipped.",
         "5. **No causal / 'underrated' language** is used; 'over-valued' / 'over-performing' name a "
         "DESCRIPTIVE gap between two orderings, not a quality judgment.\n"]
    (ROOT / "MISPRICED_PROGRAMS_RESULT.md").write_text("\n".join(L))


def make_figures(r, inst):
    # residual distribution per field (boxplot-ish) — show spread of disagreement by field
    fig, ax = plt.subplots(figsize=(9, 5))
    order = r.groupby("label").resid.std().sort_values(ascending=False).index[:18]
    data = [r[r.label == f].resid.values for f in order]
    ax.boxplot(data, vert=True, showfliers=True, flierprops=dict(marker=".", markersize=3))
    ax.set_xticklabels(order, rotation=60, ha="right", fontsize=7)
    ax.axhline(0, color="#888", lw=1)
    ax.set_ylabel("standardized within-field valuation residual r")
    ax.set_title("Where the two valuations disagree most — residual spread by field (top-18 by spread)\n"
                 "r>0 market-overperforming, r<0 academically over-valued (SELECTION caveat applies)")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "over_undervalued_programs.png", dpi=140)

    # institution extremes
    top = inst.nlargest(12, "mean_resid"); bot = inst.nsmallest(12, "mean_resid")
    d = pd.concat([bot, top])
    fig, ax = plt.subplots(figsize=(8, 7))
    cols = ["#D6202A" if v < 0 else "#2C7BB6" for v in d.mean_resid]
    ax.barh(range(len(d)), d.mean_resid, color=cols)
    ax.set_yticks(range(len(d))); ax.set_yticklabels([f"{n[:34]} (k={int(k)})" for n, k in zip(d.institution, d.n_fields)], fontsize=6.5)
    ax.axvline(0, color="#888", lw=1)
    ax.set_xlabel("mean standardized valuation residual across the institution's fields")
    ax.set_title("Systematically over-performing (blue) vs academically over-valued (red) institutions\n"
                 "DESCRIPTIVE — conflates value-added with selection; names illustrative")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "institution_valuation_extremes.png", dpi=140)


if __name__ == "__main__":
    main()
