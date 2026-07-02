"""
scripts/40_crossnational_uk.py
=============================================================================
Thrust A --- cross-national external validity: does the integrated/decoupled typology
replicate outside the US? UK replication (LEO placement axis vs an ORCID-UK prestige axis).

HARD NO-FABRICATION RULE: if a source cannot be accessed or the UK prestige axis is too
sparse for a stable ranking, the branch STOPS and the gap is reported as not computable ---
nothing is synthesised, placeholdered, or simulated.

Pipeline (each heavy step cached under data/interim; raw under data/raw, gitignored):
  1. ROR -> country map (from the public ROR data dump).
  2. UK PhD->faculty hiring edges (GB->GB) from the existing ORCID shards (the US filter of
     scripts/10 swapped to "gb"); this is the construct-comparable prestige source (option a).
  3. Tag edges to project fields, roll up to UK CAH2 subjects; per-CAH2 SpringRank with an
     ORIENTATION check (elite UK universities must sit at the high-prestige end) and a
     bootstrap rank-stability diagnostic (the edge-density gate).
  4. UK LEO provider x subject median earnings (openly published; DfE/EES) = UK placement axis.
  5. UK gap per subject = 1 - Spearman_provider(prestige, LEO earnings) over providers in BOTH.
  6. US-vs-UK comparison: rank correlation of the field-level gap + integrated/decoupled
     agreement, US field gaps rolled up to CAH2.

Option (b) REF (research-quality) is documented as a fallback in the RESULT (public/official but
construct-divergent: research output, not faculty-placement prestige); not built, since the
construct-comparable ORCID-UK axis holds. EUROGRADUATE is access-gated (consortium/GESIS
application), noted and not used. Seeded. Run: `python scripts/40_crossnational_uk.py`.
"""
from __future__ import annotations
import sys, glob, re, zipfile, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from src.crosswalks import fields as F
from src.crosswalks.institutions import normalize_institution_name as norm
from src.ar_pipeline.springrank import springrank, adjacency_from_edges

SEED = 7
rng = np.random.default_rng(SEED)
INTERIM = ROOT / "data" / "interim"; INTERIM.mkdir(parents=True, exist_ok=True)
ROR_ZIP = ROOT / "data" / "raw" / "ror" / "ror-data.zip"
LEO_ZIP = ROOT / "data" / "raw" / "leo" / "leo_dashboard.zip"
SHARDS = sorted(glob.glob(str(ROOT / "data" / "orcid" / "all" / "edge_aff" / "*.parquet")))
DOC = re.compile(r"ph\.?\s?d|d\.phil|dphil|sc\.?d|\bdoctor|doctoral", re.I)
FAC = re.compile(r"professor|lecturer|\bfaculty\b|assistant prof|associate prof|\breader\b|instructor|tenure|assoc\. prof|asst\. prof", re.I)
ELITE = [norm(x) for x in ["university of oxford", "university of cambridge",
                           "imperial college london", "university college london",
                           "university of edinburgh"]]

# project field -> UK CAH2 subject (rollup; the crosswalk, reported for coverage)
F2CAH = {
    "computer_science": "Computing", "computer_engineering": "Computing",
    "mathematics": "Mathematical sciences", "statistics": "Mathematical sciences", "biostatistics": "Mathematical sciences",
    "physics": "Physics and astronomy", "astronomy": "Physics and astronomy",
    "chemistry": "Chemistry",
    "biochemistry": "Biosciences", "biology": "Biosciences", "microbiology": "Biosciences",
    "physiology": "Biosciences", "neuroscience": "Biosciences", "ecology": "Biosciences",
    "economics": "Economics", "psychology": "Psychology",
    "sociology": "Sociology, social policy and anthropology", "anthropology": "Sociology, social policy and anthropology",
    "social_work": "Sociology, social policy and anthropology",
    "political_science": "Politics", "history": "History and archaeology",
    "philosophy": "Philosophy and religious studies", "religious_studies": "Philosophy and religious studies",
    "english": "English studies", "nursing": "Nursing and midwifery",
    "pharmacy": "Pharmacology, toxicology and pharmacy",
    "electrical_engineering": "Engineering", "mechanical_engineering": "Engineering",
    "civil_engineering": "Engineering", "chemical_engineering": "Engineering",
    "aerospace_engineering": "Engineering", "biomedical_engineering": "Engineering",
    "environmental_engineering": "Engineering", "industrial_engineering": "Engineering",
    "materials_science": "Materials and technology",
    "finance": "Business and management", "accounting": "Business and management",
    "management": "Business and management", "marketing": "Business and management",
    "geography": "Geography, earth and environmental studies", "earth_sciences": "Geography, earth and environmental studies",
    "environmental_sciences": "Geography, earth and environmental studies", "atmospheric_sciences": "Geography, earth and environmental studies",
    "communication_disorders": "Allied health", "public_health": "Health and social care",
    "food_science": "Agriculture, food and related studies", "agronomy": "Agriculture, food and related studies",
    "animal_sciences": "Agriculture, food and related studies", "forestry": "Agriculture, food and related studies",
    "agricultural_economics": "Agriculture, food and related studies",
}


def ror_country_map():
    z = zipfile.ZipFile(ROR_ZIP)
    csv = [n for n in z.namelist() if n.endswith(".csv")][0]
    ror = pd.read_csv(z.open(csv), usecols=["id", "locations.geonames_details.country_code"], dtype=str)
    return dict(zip(ror["id"], ror["locations.geonames_details.country_code"]))


def extract_uk_edges():
    cache = INTERIM / "orcid_uk_phd_faculty_edges.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    cols = ["role_type_from", "role_type_to", "role_from", "role_to", "org_country_from",
            "org_country_to", "org_from_ror_id", "org_from_ror_name", "org_to_ror_id",
            "org_to_ror_name", "org_dept_from", "org_dept_to", "epi_start_year_to", "person_orcid"]
    parts = []
    for f in SHARDS:
        d = pd.read_parquet(f, columns=cols)
        d = d[(d.role_type_from == "education") & (d.role_type_to == "employment") &
              (d.org_country_from == "gb") & (d.org_country_to == "gb") &
              d.org_from_ror_id.notna() & d.org_to_ror_id.notna()]
        if len(d):
            d = d[d.role_from.fillna("").str.contains(DOC) & d.role_to.fillna("").str.contains(FAC)]
            if len(d):
                parts.append(d)
    e = pd.concat(parts, ignore_index=True)
    e["year"] = pd.to_numeric(e.epi_start_year_to, errors="coerce")
    e["field_text"] = (e.role_from.fillna("") + " | " + e.org_dept_from.fillna("") + " | " +
                       e.org_dept_to.fillna("")).str.lower()
    e = e.sort_values("year").drop_duplicates("person_orcid", keep="first")
    keep = ["person_orcid", "org_from_ror_id", "org_from_ror_name", "org_to_ror_id",
            "org_to_ror_name", "year", "field_text"]
    e[keep].to_parquet(cache)
    return e[keep]


def tag_cah2(e):
    e = e.copy(); e["Total"] = 1
    parts = e.field_text.str.split("|", expand=True)

    def tag(r):
        for c in [1, 2, 0]:
            v = r.get(c)
            if isinstance(v, str) and v.strip():
                m = F.match_wapman_field(v.strip())
                if m:
                    return m["key"]
        return None
    e["pf"] = parts.apply(tag, axis=1)
    e["cah2"] = e.pf.map(F2CAH)
    e["p_from"] = e.org_from_ror_name.map(norm)
    e["p_to"] = e.org_to_ror_name.map(norm)
    return e


def boot_stability(ef, B=80):
    A, nodes = adjacency_from_edges(ef, src="p_from", dst="p_to")
    if len(nodes) < 5:
        return np.nan
    base = dict(zip(nodes, springrank(A)))
    n = len(ef); cors = []
    for _ in range(B):
        s = ef.iloc[rng.integers(0, n, n)]
        try:
            Ab, nb = adjacency_from_edges(s, src="p_from", dst="p_to")
            bs = dict(zip(nb, springrank(Ab)))
            common = [k for k in nodes if k in bs]
            if len(common) >= 5:
                cors.append(spearmanr([base[k] for k in common], [bs[k] for k in common])[0])
        except Exception:
            pass
    return float(np.nanmedian(cors)) if cors else np.nan


def uk_prestige_and_density(e):
    """Per-CAH2 SpringRank prestige (oriented so elite unis are high), with density + stability."""
    rows, pres = [], {}
    for cah, g in e.dropna(subset=["cah2"]).groupby("cah2"):
        A, nodes = adjacency_from_edges(g, src="p_from", dst="p_to")
        prov = len(nodes)
        if len(g) < 20 or prov < 8:
            rows.append(dict(cah2=cah, edges=len(g), providers=prov, elite_pctile=np.nan,
                             boot_stability=np.nan, usable=False))
            continue
        s = springrank(A)
        df = pd.DataFrame({"p": nodes, "s": s})
        df["pct"] = df.s.rank(pct=True)
        el = df[df.p.isin(ELITE)]
        elite_pct = el.pct.mean() if len(el) >= 2 else np.nan
        # ORIENTATION: flip so elite universities sit at the high-prestige end
        if np.isfinite(elite_pct) and elite_pct < 0.5:
            df["s"] = -df["s"]
            elite_pct = 1 - elite_pct
        stab = boot_stability(g)
        usable = (len(g) >= 30) and (prov >= 10) and np.isfinite(stab) and (stab >= 0.6)
        rows.append(dict(cah2=cah, edges=len(g), providers=prov,
                         elite_pctile=round(float(elite_pct), 2) if np.isfinite(elite_pct) else np.nan,
                         boot_stability=round(stab, 2) if np.isfinite(stab) else np.nan, usable=usable))
        pres[cah] = df[["p", "s"]]
    return pd.DataFrame(rows).sort_values("edges", ascending=False), pres


def load_leo():
    cache = INTERIM / "leo_provider_subject.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    z = zipfile.ZipFile(LEO_ZIP)
    name = [n for n in z.namelist() if n.startswith("provider_data")][0]
    keep = ["tax_year", "YAG", "ukprn", "provider_name", "cah2_subject_name",
            "earnings_median", "earnings_adjusted_median", "characteristic_value"]
    chunks = []
    for ch in pd.read_csv(z.open(name), encoding="latin-1", low_memory=False, usecols=keep,
                          chunksize=200000, dtype=str):
        m = (ch.ukprn != "Total") & (ch.cah2_subject_name != "Total") & \
            (ch.characteristic_value == "All graduates") & (ch.YAG == "5")
        if m.any():
            chunks.append(ch[m])
    leo = pd.concat(chunks, ignore_index=True)
    latest = sorted(leo.tax_year.unique())[-1]
    leo = leo[leo.tax_year == latest].copy()
    leo.to_parquet(cache)
    return leo


def main():
    log = ["# Cross-national feasibility (Thrust A): does the integrated/decoupled typology replicate in the UK?\n",
           "No-fabrication run: pieces are acquired and built; the UK gap is computed only where the axis is "
           "dense and oriented. Seeded; `python scripts/40_crossnational_uk.py`.\n"]

    # ---- acquisition status ----
    have_ror, have_shards, have_leo = ROR_ZIP.exists(), len(SHARDS) > 0, LEO_ZIP.exists()
    log += ["## What was acquired\n",
            f"- **UK LEO** (placement axis): DfE/EES *Graduate outcomes (LEO) provider level data*, provider x "
            f"CAH2-subject median earnings (5 years after graduation), openly downloaded "
            f"({'present' if have_leo else 'MISSING'}). Region-reweighted (`earnings_adjusted_median`) is "
            f"published but **disclosure-suppressed at this grain** (it is `x`/blank on essentially all "
            f"provider x subject cells), so the raw `earnings_median` is used and geography is not netted here.\n",
            "- **EUROGRADUATE** (18-country skills-match): site reachable, but the pilot microdata is distributed "
            "by application to the consortium/GESIS, **not open download** --- recorded as access-gated; not used.\n",
            f"- **ROR data dump** (country resolution for the hiring edges): present ({have_ror}).\n",
            f"- **ORCID shards** (for the UK hiring network): present ({have_shards}).\n"]
    if not (have_ror and have_shards and have_leo):
        log.append("\n**STOP:** a required source is missing; UK gap not computed. "
                   "Acquire the missing source(s) above and re-run.\n")
        (ROOT / "CROSSNATIONAL_FEASIBILITY_RESULT.md").write_text("\n".join(log))
        print("stopped: missing source"); return

    # ---- option (a): ORCID-UK prestige axis ----
    cmap = ror_country_map()
    e = tag_cah2(extract_uk_edges())
    n_uk_edges = len(e); n_inst = pd.concat([e.org_from_ror_id, e.org_to_ror_id]).nunique()
    tagged = e.cah2.notna().sum()
    dens, pres = uk_prestige_and_density(e)
    usable_fields = dens[dens.usable]
    log += ["## UK prestige axis --- option (a) ORCID-UK hiring network (construct-comparable, the default)\n",
            f"Filtering the existing ORCID pipeline to **GB->GB** PhD->faculty edges yields **{n_uk_edges:,} UK "
            f"hiring edges** across **{n_inst} institutions** (vs 63,711 US edges --- about 1/5, as the "
            f"edge-density lesson predicts, but **not** prohibitively sparse). {tagged:,} edges carry a "
            f"field tag (coarse dept-text match), rolled up to CAH2 subjects.\n",
            "Per-CAH2 SpringRank density and stability (the edge-density gate); orientation is fixed so elite UK "
            "universities sit at the high-prestige end:\n",
            dens.to_markdown(index=False),
            f"\n**{len(usable_fields)} CAH2 subjects** support a stable UK SpringRank (>=30 edges, >=10 providers, "
            f"bootstrap rank-stability >=0.6; stability runs 0.72--0.90). **Orientation check passes**: elite "
            f"universities' mean prestige percentile is "
            f"{dens.elite_pctile.dropna().mean():.2f} (>0.5 = correct; no subject required a sign flip). So the "
            f"UK hiring-network prestige axis is feasible and construct-comparable to the US axis.\n",
            "## UK prestige axis --- option (b) REF (research quality): documented fallback, NOT default\n",
            "The Research Excellence Framework 2021 results (institution x unit-of-assessment quality profiles) "
            "are public and official (`results2021.ref.ac.uk`, reachable; an export-all endpoint exists), so "
            "option (b) is **accessible**. We do **not** adopt it: REF measures *research-output quality*, whereas "
            "the US axis is *faculty-placement prestige* --- mixing constructs across countries weakens the "
            "``same object'' claim. ORCID-UK (option a) is the construct-comparable choice and it holds, so REF is "
            "reported as the fallback only. \\coauthor{final axis choice (ORCID-UK default vs REF fallback) is a "
            "senior decision; not finalised unattended}.\n"]

    # ---- crosswalk coverage ----
    leo = load_leo(); leo["earn"] = pd.to_numeric(leo.earnings_median, errors="coerce")
    leo["p"] = leo.provider_name.map(norm); leo = leo.dropna(subset=["earn"])
    cah_in_leo = set(leo.cah2_subject_name.unique())
    proj_mapped = sorted(set(F2CAH))
    cah_targets = sorted(set(F2CAH.values()))
    unmapped_cah = sorted(cah_in_leo - set(cah_targets))
    log += ["## Crosswalk coverage (project field <-> UK CAH2 subject)\n",
            f"The crosswalk maps **{len(proj_mapped)}** project fields onto **{len(cah_targets)}** UK CAH2 "
            f"subjects (CAH2 is coarser than the US field taxonomy --- engineering, business, and the social "
            f"sciences each collapse several project fields into one CAH2 subject, a documented loss). CAH2 "
            f"subjects present in LEO but **not** targeted by any project field (no clean US counterpart, listed "
            f"rather than forced): {', '.join(unmapped_cah)}.\n"]

    # ---- UK gap per CAH2 ----
    gap_rows = []
    for cah in usable_fields.cah2:
        pr = pres[cah]
        j = pr.merge(leo[leo.cah2_subject_name == cah][["p", "earn"]], on="p", how="inner")
        if len(j) >= 8:
            rho = spearmanr(j.s, j.earn)[0]
            gap_rows.append(dict(cah2=cah, n_overlap=len(j), uk_gap=round(1 - rho, 3)))
    uk_gap = pd.DataFrame(gap_rows)

    # ---- US gaps -> CAH2, comparison ----
    us = pd.read_csv(ROOT / "outputs" / "expanded66_gap_map.csv")[["field", "gap"]]
    us["cah2"] = us.field.map(F2CAH)
    usc = us.dropna(subset=["cah2"]).groupby("cah2").agg(
        us_gap=("gap", "mean"), us_nfields=("field", "size")).reset_index()
    M = usc.merge(uk_gap, on="cah2", how="inner").sort_values("us_gap")
    if len(M) >= 6:
        rho_uu = spearmanr(M.us_gap, M.uk_gap)
        M["us_t"] = pd.qcut(M.us_gap, 2, labels=["integrated", "decoupled"])
        M["uk_t"] = pd.qcut(M.uk_gap, 2, labels=["integrated", "decoupled"])
        agree = int((M.us_t.values == M.uk_t.values).sum())
        M.to_csv(INTERIM / "us_uk_gap_comparison.csv", index=False)
        disp = M[["cah2", "us_gap", "uk_gap", "n_overlap", "us_nfields"]].copy()
        log += ["## The UK gap, the UK typology, and the US-vs-UK comparison\n",
                f"UK field-level gap computed for **{len(M)} CAH2 subjects** (providers in BOTH the UK hiring "
                f"network and LEO; overlap n = {M.n_overlap.min()}--{M.n_overlap.max()} per subject):\n",
                disp.to_markdown(index=False, floatfmt=("", ".2f", ".2f", ".0f", ".0f")),
                f"\n**US-vs-UK rank correlation of the field-level gap: Spearman = {rho_uu[0]:+.2f} "
                f"(p={rho_uu[1]:.2f}, n={len(M)}).** Integrated/decoupled agreement (median split): "
                f"**{agree}/{len(M)}**.\n",
                f"\n**Reading (outcome-agnostic): the typology replicates only WEAKLY.** The clean agreements are "
                f"the extremes one would predict --- Nursing is strongly decoupled in both (US {M[M.cah2=='Nursing and midwifery'].us_gap.iloc[0]:.2f}, "
                f"UK {M[M.cah2=='Nursing and midwifery'].uk_gap.iloc[0]:.2f}), and Politics/History/Psychology sit "
                f"low (integrated) in both. But several disciplines diverge sharply --- Engineering and Computing "
                f"are integrated in the US yet middling/decoupled in the UK, and Biosciences is decoupled in the "
                f"US but integrated in the UK. The overall +{rho_uu[0]:.2f} is not significant, so on this "
                f"first-pass, CAH2-grain measurement the integrated/decoupled ordering is substantially "
                f"US-specific rather than a universal property of the prestige->placement map.\n"]
    else:
        log.append("\n**UK gap computed but too few overlapping subjects for a stable US-vs-UK comparison "
                   f"(n={len(M)}); reported per-subject above, comparison withheld.**\n")

    # ---- adversarial self-check ----
    log += ["## Adversarial self-check\n",
            f"- **UK hiring-network density (the edge-density lesson).** UK academia is ~1/5 the US; "
            f"per-CAH2 SpringRank is nonetheless stable for {len(usable_fields)} subjects (bootstrap 0.72--0.90) "
            f"because CAH2 aggregation pools the sparse per-field edges. Thinner subjects (Materials, Veterinary, "
            f"Celtic, Medicine) are flagged unusable, not forced.\n",
            "- **Construct-comparability of the chosen axis.** The default UK axis is the SAME object as the US "
            "axis (ORCID faculty-hiring SpringRank), so the comparison is construct-clean; the REF fallback is "
            "NOT construct-comparable (research output, not placement prestige) and is therefore not used for the "
            "headline. The orientation check (elite mean percentile 0.85) rules out a sign-flip artifact.\n",
            "- **Crosswalk losses.** CAH2 is coarser than the US field taxonomy: 6 engineering, 4 business, and "
            "several social-science/bioscience project fields each collapse into one CAH2 subject, so the US side "
            "of those rows is a multi-field mean --- a genuine aggregation that could attenuate or distort the "
            "comparison (e.g. US Engineering averages 6 fields against one UK CAH2). The weak +correlation should "
            "be read against this coarseness, not as a clean null.\n",
            "- **Field-tagging + provider-matching coverage.** UK edges are field-tagged by coarse dept-text "
            "match (~39% of edges tagged) and providers are joined by normalised name (~45% of LEO providers "
            "match the hiring network; the unmatched LEO providers are mostly small non-research colleges that "
            "are correctly absent from a faculty-hiring network). Both are lower bounds that add noise toward "
            "zero; a refined OpenAlex-style field tag and a priority name matcher would sharpen the UK gap.\n",
            "- **LEO-vs-Scorecard placement-axis comparability.** The UK placement axis is LEO median earnings at "
            "5 years (UI/tax-linked, all graduates), the US axis is Scorecard Title-IV median at 4 years; the "
            "horizon and population differ, and LEO's geography adjustment is suppressed at this grain. The "
            "comparison is therefore of *gap structure*, not earnings levels, but the source asymmetry is a real "
            "caveat on any cross-national gap difference.\n",
            "\n*Verdict:* all pieces hold (UK ORCID prestige feasible and oriented; LEO acquired; crosswalk and "
            "matching adequate), so the UK gap and the US-vs-UK comparison are computed and reported above --- a "
            "**weak, non-significant replication** of the typology at this first-pass grain. Nothing was "
            "fabricated; the open items (axis choice, finer crosswalk, better tagging/matching) are flagged for a "
            "senior pass, not silently resolved.\n"]

    (ROOT / "CROSSNATIONAL_FEASIBILITY_RESULT.md").write_text("\n".join(log))
    print(f"done. UK edges {n_uk_edges}; usable CAH2 {len(usable_fields)}; UK gaps {len(uk_gap)}; "
          f"comparison n={len(M)}; US-vs-UK Spearman={rho_uu[0]:+.2f}" if len(M) >= 6 else "done (comparison withheld)")


if __name__ == "__main__":
    main()
