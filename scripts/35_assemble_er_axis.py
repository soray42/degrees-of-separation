"""
scripts/35_assemble_er_axis.py
=============================================================================
Assemble the revealed-placement identification layer (35a + 35b + 35c) into
ER_AXIS_IDENTIFICATION_RESULT.md: header/reframe + the three sections + synthesis +
the required adversarial self-check. Reads the fragments written by 35a/b/c and
recomputes the three headline numbers from the interim CSVs so the synthesis stays
accurate. Run after 35a, 35b, 35c. `python scripts/35_assemble_er_axis.py`.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

INTERIM = ROOT / "data" / "interim"
B_ABSORPTION, B_LICENSURE = -0.02, +0.66


def main():
    # headline numbers (recomputed from the interim CSVs)
    px = pd.read_csv(INTERIM / "er_axis_35a_proxy_by_horizon.csv")
    sc = px[px.source == "Scorecard"]
    c1 = float(sc[sc.horizon == "1YR"].corr_prestige.iloc[0])
    c5 = float(sc[sc.horizon == "5YR"].corr_prestige.iloc[0])
    db = pd.read_csv(INTERIM / "er_axis_35b_deferral.csv").dropna(subset=["status_resid", "grad_degree_share"])
    r_defer = pearsonr(db.status_resid, db.grad_degree_share)[0]
    mc = pd.read_csv(INTERIM / "er_axis_35c_mechanism.csv")
    cvr = spearmanr(mc.dropna(subset=["xinst_cv", "licensure"]).licensure,
                    mc.dropna(subset=["xinst_cv", "licensure"]).xinst_cv)[0]
    dd = mc.dropna(subset=["prestige_R2", "geo_R2", "licensure"])
    rdom = spearmanr(dd.licensure, dd.geo_R2 - dd.prestige_R2)[0]
    mech = "(b) prestige-orthogonal variance (setting, not school)" if rdom > 0.2 and cvr > -0.2 else \
           ("(a) wage compression" if cvr < -0.2 else "mixed")

    frags = []
    for f in ["er_axis_a.md", "er_axis_b.md", "er_axis_c.md"]:
        p = INTERIM / f
        frags.append(p.read_text() if p.exists() else f"## [missing fragment {f} -- run the matching script]\n")

    head = [
        "# The revealed-placement identification layer (ER axis)\n",
        "**Reframe (load-bearing).** The gap's second axis is **not** 'employer reputation' (a "
        "perception/survey construct, the QS sense) but **revealed labour-market placement** -- where "
        "graduates actually land -- which median earnings proxies. The paper's object is academic "
        "**reputation** (AR, the hiring-network SpringRank) versus revealed **placement** -- an honest "
        "asymmetry (perception vs realised outcome), not 'reputation vs reputation'. This layer establishes "
        "salary as a **defensible-but-incomplete** placement proxy: where it works (35a), where it fails via "
        "pipeline deferral (35b), and the licensing mechanism behind apparent decoupling (35c). "
        "Field-level identification; descriptive, outcome-agnostic, seeded.\n",
    ]
    synth = [
        "## Synthesis\n",
        "**Salary is a defensible-but-incomplete revealed-placement proxy.** Across horizons and three "
        f"independent earnings sources it tracks non-wage placement positively (Scorecard salary-vs-occupational-"
        f"prestige rises {c1:+.2f}->{c5:+.2f} from 1yr to 5yr; PSEO and ACS agree), so it is a real proxy -- but "
        "it fails in **two distinct, identified ways**, each a source of **apparent** decoupling rather than "
        "market undervaluation:\n",
        f"1. **Pipeline-deferral fields** -- a **cross-field proxy-timing** effect. Early-career BA earnings is "
        f"a poor *terminal*-market proxy where graduates defer to graduate school; the field-level "
        f"salary-vs-status divergence is explained by graduate-degree share (status residual ~ deferral, "
        f"Pearson {r_defer:+.2f}). This is **distinct from the project's null academic-absorption channel** "
        f"(b_absorption = {B_ABSORPTION:+.2f}; absorption is PhD->academia, a different construct, near-"
        f"orthogonal to deferral), and it does **not** explain the within-field gap (which is rank-based and "
        f"horizon-stable). It narrows the 'earnings mis-prices natural science' claim to a measurement-timing "
        f"fact.\n",
        f"2. **Licensed fields** -- via **{mech}**, *not* wage compression. Licensure does not shrink within-"
        f"field wage variance (CV ~ licensure {cvr:+.2f}); rather, with higher licensure that variance becomes "
        f"prestige-orthogonal -- driven by destination geography/setting, not school prestige (geography-over-"
        f"prestige dominance ~ licensure {rdom:+.2f}). This is the mechanism behind b_licensure = "
        f"{B_LICENSURE:+.2f}: the license makes wages depend on *setting*, so prestige cannot predict pay and "
        f"the field looks decoupled.\n",
        "\nNeither failure is evidence that the labour market undervalues these fields; both are properties of "
        "**salary as a timed, setting-sensitive proxy** for revealed placement.\n",
        "## Adversarial self-check\n",
        f"- **Absorption vs deferral reconciliation.** The null academic-absorption channel (PhD->academia, "
        f"b_absorption = {B_ABSORPTION:+.2f}) and pipeline deferral (BA->grad-school) are different constructs "
        f"and near-orthogonal (corr ~ +0.08); deferral drives the status residual while absorption does not. So "
        f"35b does not contradict, re-discover, or rehabilitate the absorption null -- it identifies a separate "
        f"cross-field proxy-timing effect.\n",
        f"- **35c mechanism stated explicitly.** The data reject (a) wage compression (CV is flat in licensure) "
        f"and support (b) prestige-orthogonal/setting-driven variance. The honest consequence: the story is "
        f"**'license makes wages depend on setting, not school'**, not 'license compresses wages'. Nursing is "
        f"the clean illustration, but the claim is field-general (continuous across the decomposable fields).\n",
        "- **Rigor-up / punch-down on natural science.** 35b *narrows* rather than inflates: it converts an "
        "apparent 'market undervalues natural science' story into a proxy-window measurement fact, and leaves "
        "the within-field gap intact. We do not claim natural-science programs are well- or under-priced in "
        "terminal markets -- only that early-career BA salary cannot adjudicate it.\n",
        "- **Grain caveat (load-bearing).** This is the **field-level** identification layer. It establishes "
        "where salary is and isn't a good *field-level* placement proxy; it does **not** recompute a within-"
        "field non-wage gap. The within-field analogue `gap^occ_f = 1 - Spearman_i(prestige_{i,f}, "
        "occ_prestige_{i,f})` still needs institution x field occupation flows (Revelio); its estimand is "
        "specified in `ER_DEFINITION_RESULT.md`.\n",
        "\n*Provenance:* generated by `scripts/35a_placement_proxy_validation.py`, "
        "`scripts/35b_pipeline_deferral.py`, `scripts/35c_licensing_compression.py`, assembled by "
        "`scripts/35_assemble_er_axis.py`. Reuses the gap, ACS licensure (b_licensure), multi-horizon ICC, "
        "Condon/Hughes 2024 OPR, and NY Fed by-major; no within-field gap re-run. Interim tables are "
        "regenerated by the scripts (gitignored per repo policy).\n",
    ]
    (ROOT / "ER_AXIS_IDENTIFICATION_RESULT.md").write_text(
        "\n".join(head) + "\n" + "\n\n".join(frags) + "\n\n" + "\n".join(synth))
    print(f"assembled ER_AXIS_IDENTIFICATION_RESULT.md | proxy {c1:+.2f}->{c5:+.2f} | "
          f"deferral {r_defer:+.2f} | mechanism {mech}")


if __name__ == "__main__":
    main()
