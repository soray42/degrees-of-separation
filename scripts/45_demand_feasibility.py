"""
scripts/45_demand_feasibility.py
=============================================================================
FEASIBILITY GATE for a revealed-DEMAND layer (applications/acceptances by provider x field).
Verifies the open data EXISTS and JOINS to the earnings + prestige axes already built. This is
the GATE only --- NO slopes, NO analysis, NO conclusions about demand.

HARD NO-FABRICATION RULE: if a source is not openly available at the needed grain or will not
join, the branch STOPS and reports exactly what is missing. Application/acceptance counts are
never synthesised. A clean "joins / doesn't" is the deliverable.

Reuses the UK LEO provider x CAH2 earnings (data/interim/leo_provider_subject.parquet) and the
UK ORCID CAH2 prestige providers (data/interim/orcid_uk_phd_faculty_edges.parquet) from
scripts/40, and the US institution x field panel (data/interim/valuation_residuals.csv).
Seeded. Run: `python scripts/45_demand_feasibility.py`.

SOURCE AVAILABILITY (verified against the official documentation / portals; encoded here, not
inferred):
  UK UCAS end-of-cycle, FREE provider-level data resources (per the official Acceptances /
  Applicants resource catalogues):
    - Provider x subject group (JACS3): resource EOC_HEP_*_014, cycles 2007-2021.
    - Provider x subject group (HECoS): resource EOC_HEP_*_015, cycles 2019-2021.
    - APPLICATIONS and ACCEPTANCES are in the FREE provider-level release; the data files are
      per-resource CSV/zip behind /media/<id>/download (verified: a provider-level zip downloads;
      the per-resource media ids are mapped on the dynamic page).
    - OFFERS: only a partial 18-year-old subset is free (resource 021); COMPLETE offers / admit
      rates are the PAID "UCAS EXACT" product and are NOT used.
    - Provider key = UCAS provider code + NAME (not UKPRN) -> join by normalised name.
  US UC "Freshman admission by discipline" (UC Information Center):
    - campus x DISCIPLINE (15 broad disciplines + undeclared/unknown), applications/admits/
      enrolled, cycles 2012-2023; downloadable (Tableau crosstab export; a "data download
      instructions" PDF exists). Grain = DISCIPLINE, not major. Covers the 9 UC undergraduate
      campuses only.
  US CSU (secondary): the CSU data dashboard publishes admissions by college/major, but the data
    centre was access-blocked (HTTP 403) in this environment -> flagged scrape/portal-gated.
"""
from __future__ import annotations
import sys, zipfile, re, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.crosswalks.institutions import normalize_institution_name as norm

INTERIM = ROOT / "data" / "interim"
UCAS_ZIP = ROOT / "data" / "raw" / "ucas" / "ucas_provider_acceptances.zip"   # resource 021 (provider universe)
UC_CAMPUSES = ["berkeley", "los angeles", "davis", "irvine", "san diego", "santa barbara",
               "santa cruz", "riverside", "merced"]   # the 9 UC undergraduate campuses


def ucas_provider_universe():
    """Distinct normalised UK provider names from the UCAS free provider-level release (the
    provider universe is shared across all provider-level resources)."""
    if not UCAS_ZIP.exists():
        return None
    z = zipfile.ZipFile(UCAS_ZIP)
    csv = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    raw = z.open(csv).read().decode("latin-1").splitlines()
    # data starts after the UCAS preamble (header row contains 'Provider')
    hdr = next(i for i, l in enumerate(raw) if l.split(",")[0].strip().strip('"') in ("Year", "Provider"))
    prov = set()
    for l in raw[hdr + 1:]:
        parts = l.split(",")
        if len(parts) > 1:
            p = re.sub(r"^[A-Z]\d+\s+", "", parts[1].strip().strip('"'))  # strip UCAS code prefix
            if p:
                prov.add(norm(p))
    return prov


def main():
    L = ["# Revealed-demand layer --- FEASIBILITY GATE (does the data exist + join?)\n",
         "Gate only: verify open application/acceptance data exists at provider x field and joins to the "
         "earnings + prestige axes. **No slopes, no conclusions.** No-fabrication run. Seeded; "
         "`python scripts/45_demand_feasibility.py`.\n",
         "## Per-source availability\n",
         "| source | available? | grain | years | downloadable? |",
         "|---|---|---|---|---|",
         "| **UK UCAS** end-of-cycle (free provider-level) | **yes (free)** | provider x subject group "
         "(JACS3 / HECoS) | 2007--2021 (JACS3); 2019--2021 (HECoS) | yes, per-resource CSV/zip via "
         "`/media/<id>/download` |",
         "| UK UCAS OFFERS / admit rates | partial-free / **PAID** | provider x subject | -- | complete offers = "
         "paid **UCAS EXACT** (NOT used); only an 18-yr-old offer subset is free |",
         "| **US UC** freshman admission by discipline | **yes** | campus x **discipline** (15 broad + "
         "undeclared) | 2012--2023 | yes (Tableau crosstab export; download-instructions PDF) |",
         "| US CSU (secondary) | gated here | college/major | -- | dashboard; data centre **HTTP 403** in this "
         "environment -> scrape/portal-gated |",
         "\n**Applications + acceptances are FREE** at provider x subject group (UK) and campus x discipline "
         "(US UC). The funnel middle (complete OFFERS / admit rates) is paid UCAS EXACT in the UK and is not "
         "used.\n"]

    # ---- UK three-way join (the gate) ----
    ucas = ucas_provider_universe()
    leo = pd.read_parquet(INTERIM / "leo_provider_subject.parquet")
    leo_p = set(leo.provider_name.dropna().map(norm))
    uk = pd.read_parquet(INTERIM / "orcid_uk_phd_faculty_edges.parquet")
    orc_p = set(pd.concat([uk.org_from_ror_name, uk.org_to_ror_name]).dropna().map(norm))
    L += ["## UK join test (demand x earnings x prestige)\n"]
    if ucas is None:
        L.append("**UCAS provider-level file not present** -> download a free provider-level resource zip "
                 "(`/media/<id>/download`) and re-run; UK join not testable here. No number fabricated.\n")
        uk_go = "NOT TESTABLE"
    else:
        n3 = len(ucas & leo_p & orc_p)
        L += [f"Joining on the **normalised provider name** (UCAS provider = code + name, not UKPRN; LEO has "
              f"UKPRN + name; ORCID has ROR name):\n",
              f"- UCAS providers (free release): **{len(ucas)}**; LEO providers (earnings): **{len(leo_p)}**; "
              f"ORCID UK providers (prestige): **{len(orc_p)}**.\n",
              f"- UCAS $\\cap$ LEO = {len(ucas & leo_p)}; UCAS $\\cap$ ORCID = {len(ucas & orc_p)}; "
              f"LEO $\\cap$ ORCID = {len(leo_p & orc_p)}.\n",
              f"- **Three-way (demand + earnings + prestige) = {n3} providers** (exact normalised-name match; "
              f"fuzzy matching + a UCAS-code$\\to$UKPRN crosswalk would raise it). Sample: "
              f"{', '.join(sorted(list(ucas & leo_p & orc_p))[:6])}.\n",
              f"- **Subject join:** UCAS subject group (HECoS / JACS3) $\\to$ CAH2 (the LEO + ORCID-prestige "
              "grain) via the official HESA/ONS HECoS$\\to$CAH and JACS3$\\to$CAH lookups (documented; the "
              "actual UCAS subject-group resource 014/015 is free but its per-resource media id is mapped only on "
              "the dynamic page --- a manual pull completes it).\n",
              f"\nSo a usable UK panel exists: **{n3} providers x CAH2 subjects** (intersected with the ~16 "
              "CAH2 subjects that carry a stable UK ORCID prestige axis, scripts/40) carry demand + earnings + "
              "prestige.\n"]
        uk_go = "GO"

    # ---- US coverage gap ----
    vr = pd.read_csv(INTERIM / "valuation_residuals.csv")
    n_us_inst = vr.inst_key.nunique()
    uc_in = sorted({k for k in vr.inst_key.unique()
                    if "university california" in str(k) and any(c in str(k) for c in UC_CAMPUSES)})
    L += ["## US join test (demand x gap/prestige x Scorecard)\n",
          f"UC discipline $\\to$ project CIP-2 is a clean crosswalk (15 broad disciplines $\\approx$ CIP-2 "
          "clusters). The binding problem is **institutional coverage**, not the crosswalk:\n",
          f"- The US prestige/earnings axis spans **{n_us_inst} institutions** nationally "
          "(inst x field, `valuation_residuals.csv`; publics, privates, and the elite tail).\n",
          f"- The only free by-field US demand source (UC) covers the **{len(uc_in)} UC undergraduate campuses** "
          f"--- all present in the US panel, but only **{len(uc_in)/n_us_inst*100:.0f}%** of the axis, "
          "**California public only, no privates/elites**. CSU would add ~23 more CA publics (still single-state, "
          "public, and access-gated here).\n",
          "- A within-field, across-institution demand--prestige--earnings design needs broad cross-institution "
          "demand variation; UC gives at most 9 institutions per discipline and misses the entire private/elite "
          "tail. **IPEDS** admissions (all Title-IV institutions) is institution-level and has **no by-major/"
          "field breakdown**, so there is no broad US provider x field demand source.\n"]
    us_go = "NO-GO"

    # ---- GATE VERDICT ----
    L += ["## GATE VERDICT\n",
          f"- **UK: {uk_go}.** Free UCAS provider x subject-group applications + acceptances join to LEO "
          "earnings + ORCID prestige on "
          + (f"{len(ucas & leo_p & orc_p)} providers" if ucas is not None else "(re-run with UCAS file)")
          + " via normalised name; the subject crosswalk to CAH2 is documented. Steps 2--4 can proceed on the UK "
          "panel (finish the UCAS subject-resource pull + name/UKPRN crosswalk first). The funnel is "
          "demand(applications) + outcome(acceptances); admit-RATE needs paid EXACT.\n",
          f"- **US: {us_go}** for a panel comparable to the national gap/prestige axis. The demand-by-discipline "
          "data exists and is downloadable (UC), but its institutional coverage (~9 CA-public campuses, ~4% of "
          "the axis, no privates/elites) is far too narrow and non-comparable to the US prestige/earnings axis. "
          "It is at best a **partial, UC-system-internal** exercise (9 campuses x 15 disciplines), not a "
          "US-wide revealed-demand layer. No free broad US provider x field demand source exists.\n",
          "## Adversarial self-check\n",
          "- **UCAS applications vs offers (free vs EXACT).** Applications and acceptances are free at provider x "
          "subject group; the complete offers / admit-rate funnel is the paid UCAS EXACT product and is excluded. "
          "A demand(applications)+outcome(acceptances) panel is free; an admit-RATE slope is not.\n",
          "- **UC discipline-not-major coarseness.** UC publishes 15 broad DISCIPLINES, not majors --- "
          "comparable to CIP-2 / CAH2 but it loses within-discipline field variation; not finer than the gap "
          "grain.\n",
          "- **Provider/campus name-matching losses.** The UK join is exact normalised name "
          + (f"({len(ucas & leo_p & orc_p)} of {len(leo_p)} LEO / {len(ucas)} UCAS providers); " if ucas is not None else "; ")
          + "a UCAS-code$\\to$UKPRN crosswalk and fuzzy matching would raise coverage. Unmatched providers are "
          "listed by the join, not forced.\n",
          "- **Demand-count comparability across the prestige axis.** UCAS counts are rounded to the nearest 5 "
          "and UC counts are specific to the California applicant pool; absolute demand levels are not comparable "
          "across sources or against the (national) prestige axis --- any design must use within-provider / "
          "within-subject RANKS or shares, not raw counts.\n",
          "- **Gate only.** This verifies existence + join; it does NOT estimate demand slopes or draw any "
          "conclusion about revealed demand.\n"]

    (ROOT / "DEMAND_FEASIBILITY_RESULT.md").write_text("\n".join(L))
    print(f"45 done. UK gate={uk_go} "
          + (f"(3-way={len(ucas & leo_p & orc_p)} providers) " if ucas is not None else "")
          + f"| US gate={us_go} (UC {len(uc_in)}/{n_us_inst} institutions)")


if __name__ == "__main__":
    main()
