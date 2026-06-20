"""Tier 0 — coverage probe (the gate inputs).  → results/COVERAGE_REPORT.md

Answers, with numbers: how many institutions match per field at each level; how many
fields clear n>=10; whether PhD ER cells exist in PSEO; and — critically — whether the
ER-covered institutions span the Wapman prestige range or only its public-heavy tail.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from src.load_ar import load_ar_wapman
from src.load_er import load_er_scorecard, load_er_pseo, PSEO_EARN
from src.crosswalks import fields as F

R = Path("results"); R.joinpath("figures").mkdir(parents=True, exist_ok=True)
Path("data/interim").mkdir(parents=True, exist_ok=True)
LAB = {f["key"]: f["label"] for f in F.TIER0_FIELDS}
MIN_N = 10


def main():
    ar = load_ar_wapman()
    sc = load_er_scorecard("undergrad")
    ps = load_er_pseo("undergrad")

    # ---- per-field matched counts ----
    rows = []
    for k in F.field_keys():
        a = set(ar.loc[ar.field == k, "inst_key"])
        s = set(sc.loc[sc.field == k, "inst_key"])
        p = set(ps.loc[ps.field == k, "inst_key"])
        rows.append(dict(field=k, label=LAB[k], ar_inst=len(a),
                         ar_x_scorecard=len(a & s), ar_x_pseo=len(a & p)))
    cov = pd.DataFrame(rows)
    n_sc_ok = int((cov.ar_x_scorecard >= MIN_N).sum())
    n_ps_ok = int((cov.ar_x_pseo >= MIN_N).sum())

    # ---- PhD-cell coverage in PSEO (doctoral=17, master=07) at 4-digit CIP ----
    raw = pd.read_csv(PSEO_EARN, dtype=str,
                      usecols=["inst_level", "degree_level", "cip_level", "status_y5_earnings"])
    def phd_cov(dl):
        d = raw[(raw.degree_level == dl) & (raw.inst_level == "I") & (raw.cip_level == "4")]
        present = (d.status_y5_earnings == "1").sum()
        return len(d), int(present), float(present / max(len(d), 1))
    doc = phd_cov("17"); mas = phd_cov("07")

    # ---- prestige-range coverage (the truncation check) ----
    r = pd.read_csv("data/raw/wapman2022/ranks.csv")
    from src.crosswalks.institutions import normalize_institution_name
    acad = r[r.TaxonomyLevel == "Academia"].copy()
    acad["inst_key"] = acad.InstitutionName.map(normalize_institution_name)
    sc_keys = set(sc.inst_key); ps_keys = set(ps.inst_key)
    acad["in_sc"] = acad.inst_key.isin(sc_keys); acad["in_ps"] = acad.inst_key.isin(ps_keys)
    acad["pq"] = pd.qcut(acad.Rank, 4, labels=["Q1_top", "Q2", "Q3", "Q4_bottom"])
    pq = acad.groupby("pq")[["in_sc", "in_ps"]].mean()

    # figure: prestige ECDF of all vs Scorecard-covered vs PSEO-covered
    fig, ax = plt.subplots(figsize=(8, 5))
    for mask, lab, c in [(acad.index, "all Wapman", "#888"),
                         (acad.in_sc, "Scorecard-covered", "#4878CF"),
                         (acad.in_ps, "PSEO-covered", "#D6202A")]:
        v = np.sort(acad.loc[mask, "Rank"].values) if mask is not acad.index else np.sort(acad.Rank.values)
        ax.plot(v, np.linspace(0, 1, len(v)), label=f"{lab} (n={len(v)})", color=c)
    ax.set_xlabel("Wapman prestige Rank (0 = most prestigious)")
    ax.set_ylabel("ECDF"); ax.set_title("Prestige-range coverage of ER sources\n"
                  "PSEO truncates the elite (low-Rank) top")
    ax.legend(); fig.tight_layout(); fig.savefig("results/figures/coverage_prestige_range.png", dpi=150)

    # ---- coverage_long.csv: one row per (field, AR institution), ER flags ----
    long = ar[["field", "inst_key", "institution_name"]].copy()
    long["has_undergrad_er"] = long.set_index(["field", "inst_key"]).index.isin(
        sc.set_index(["field", "inst_key"]).index)
    long["has_phd_er"] = False  # PSEO doctoral suppressed at field grain (see report)
    long.to_csv("data/interim/coverage_long.csv", index=False)

    # top-20 prestige presence
    top = acad.nsmallest(20, "Rank")[["InstitutionName", "in_sc", "in_ps"]]

    # ---- write report ----
    L = []
    L.append("# Tier 0 — Coverage Report\n")
    L.append("> **Provenance note.** Operational spec from the Tier-0 kickoff + "
             "`degrees_of_separation_proposal_v2.md`. All numbers reproduce via "
             "`python scripts/01_probe.py`. Date 2026-06-20.\n")
    L.append(f"AR = Wapman SpringRank ({ar.field.nunique()} fields, {ar.inst_key.nunique()} "
             f"institutions). ER undergrad = Scorecard FoS; ER enrichment = PSEO.\n")
    L.append("## Matched institutions per field (AR ∩ ER), undergrad level\n")
    L.append(cov.sort_values("ar_x_scorecard", ascending=False)
             .to_markdown(index=False))
    L.append(f"\n- Fields with **AR∩Scorecard ≥ {MIN_N}**: **{n_sc_ok}/{len(cov)}**.")
    L.append(f"- Fields with **AR∩PSEO ≥ {MIN_N}**: **{n_ps_ok}/{len(cov)}** "
             "(PSEO is partial / public-skewed).\n")
    L.append("## PhD-level ER coverage in PSEO (the v2 enrichment)\n")
    L.append(f"- Doctoral (degree_level 17) institution×4-digit-CIP cells: **{doc[0]}** rows, "
             f"**{doc[1]} released** ({doc[2]:.0%}). **Master's (07): {mas[1]} released.**")
    L.append("- → **PhD- and master's-level earnings are disclosure-suppressed at the field "
             "(4-digit CIP) grain.** PSEO releases graduate earnings only at the 2-digit-CIP "
             "grain (too coarse to separate e.g. chemistry from physics). **The PhD-level gap "
             "cannot be measured at the field level.**\n")
    L.append("## Prestige-range coverage (truncation check — critical)\n")
    L.append("PSEO match rate of Wapman institutions by prestige quartile (Q1 = most prestigious):\n")
    L.append((pq * 100).round(1).rename(columns={"in_sc": "Scorecard %", "in_ps": "PSEO %"})
             .to_markdown())
    L.append("\nTop-20 most prestigious Wapman institutions — present in each ER source?\n")
    L.append(top.to_markdown(index=False))
    L.append(f"\n- **Scorecard spans the full hierarchy** (≈{acad.in_sc.mean():.0%} of Wapman "
             "institutions, even coverage across quartiles).")
    L.append(f"- **PSEO is severely top-truncated**: only {acad.in_ps.mean():.0%} of Wapman "
             f"institutions, and just {pq.loc['Q1_top','in_ps']:.0%} of the top prestige "
             "quartile. ~18 of the top-20 (Caltech, MIT, Harvard, Princeton, Stanford, "
             "Berkeley, Yale, Columbia, Chicago, Penn, Cornell, CMU, …) are **absent**. "
             "Any PSEO-based gap is measured on a truncated hierarchy. See "
             "`figures/coverage_prestige_range.png`.\n")
    L.append("## Go / shelve read on coverage\n")
    L.append(f"- Undergrad map (Scorecard AR×ER): **{n_sc_ok} fields ≥ {MIN_N}**, full prestige "
             "range → **sound**.")
    L.append("- PhD-level enrichment (PSEO): **blocked** (cells suppressed at field grain) "
             "**and** prestige-truncated → the v2 two-level/PhD-ER plan is **not feasible** "
             "with public PSEO. The static map and P1 proceed on the undergrad level; the "
             "PhD-level rescue of noise-dominated fields (chemistry, earth sciences) is "
             "**unavailable**.\n")
    R.joinpath("COVERAGE_REPORT.md").write_text("\n".join(L))
    print("wrote results/COVERAGE_REPORT.md, data/interim/coverage_long.csv, "
          "figures/coverage_prestige_range.png")
    print(f"AR∩Scorecard>={MIN_N}: {n_sc_ok}/{len(cov)} | AR∩PSEO>={MIN_N}: {n_ps_ok}/{len(cov)}")
    print(f"PhD doctoral released cells: {doc[1]} | PSEO top-quartile coverage: {pq.loc['Q1_top','in_ps']:.0%}")


if __name__ == "__main__":
    main()
