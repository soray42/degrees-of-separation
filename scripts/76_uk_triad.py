"""
scripts/76_uk_triad.py
=============================================================================================
UK triad: academic reputation (REF 2021), employer reputation (LEO pay) and students' choices
(intake attainment / demand) at the university x subject level.

Public data only: REF 2021 results by institution x unit of assessment (UoA) (results2021.ref.ac.uk,
'export-all' spreadsheet), DfE LEO provider x CAH2 (as scripts/62, loaded with scripts/62's own
loader), the ORCID-derived UK hiring rank G (scripts/62's build_prestige) and the UCAS provider x
subject files already in data/raw/ucas. Descriptive, not causal. No existing script is edited;
scripts/62 is imported.

PRE-SPECIFICATION (written 2026-09-27, before any REF, UCAS or student-side data were read; the
REF file had been downloaded but not opened. Anything not listed here is labelled exploratory in the
write-up.)

Student-side data. Preferred source: the OfS Discover Uni (Unistats) course-level dataset. It is
hosted by HESA (www.hesa.ac.uk/support/tools-and-downloads/unistats); on 2026-09-27 every request from
this machine to www.hesa.ac.uk (page and file URLs) returned HTTP 403 with 'cf-mitigated: challenge'
(a Cloudflare browser challenge), and the Internet Archive returned HTTP 429, so it is NOT used.
Fallbacks, in this order and fixed now:
  S-LEO (primary student measure at CAH2): the intake attainment of the programme's graduates from
      the LEO file itself = programme share of graduates with >= 360 UCAS tariff points (LEO prior-
      attainment bands 1-2 over known bands; scripts/62's definition), pooled over cohorts
      2013/14-2016/17 at YAG5 (counts summed over cohorts). It is an entry-tariff measure of the
      students, published at provider x CAH2. Variant: mean band score (band 1 = 5 ... band 5 = 1,
      band 6 = 0, over bands 1-6).
  S-UCAS (the task's named fallback; demand): UCAS end-of-cycle provider x subject-group files in
      data/raw/ucas (granularity documented by the script; per data/raw/SOURCES.md they are CAH level 1,
      keyed by UCAS provider name, not UKPRN). Demand = main-scheme applications per accepted
      applicant, pooled over the years in the files. Used only in the across-provider designs (T3b, T4)
      on CAH1 groups that coincide with exactly one clean CAH2 subject below; UKPRN link by normalised
      name through the LEO provider names.

REF 2021. Overall quality profile of each submission; GPA = (4 %4* + 3 %3* + 2 %2* + 1 %1*) / 100;
variant = %4*. Provider x UoA score = FTE-weighted mean over the provider's submissions to that UoA
(multiple submissions). Institution score = FTE-weighted mean over all its submissions. Link to LEO by
UKPRN.

Crosswalk CAH2 (LEO subject names) -> UoA, fixed now:
  CLEAN (primary; one UoA is the research home of the subject and the subject is that UoA's principal
  teaching counterpart): Psychology 4; Biosciences 5; Chemistry 8; Physics and astronomy 9;
  Mathematical sciences 10; Computing 11; Engineering 12; Architecture, building and planning 13;
  Economics 16; Business and management 17; Law 18; Politics 19; Education and teaching 23;
  Sport and exercise sciences 24; English studies 27.
  EXTENDED (sensitivity; many-to-many; a multi-UoA subject gets the FTE-weighted GPA over the mapped
  UoAs the provider submitted to): Medicine and dentistry 1; Allied health 3; Nursing and midwifery 3;
  Pharmacology, toxicology and pharmacy 3; Agriculture, food and related studies 6; Veterinary sciences
  6; Geography, earth and environmental studies 7+14; Materials and technology 12; Sociology, social
  policy and anthropology 20+21+22; History and archaeology 15+28; Philosophy and religious studies
  30+31; Languages and area studies 25+26; Creative arts and design 32; Performing arts 33; Media,
  journalism and communications 34.
  UNMAPPED: Medical sciences; General, applied and forensic sciences; Health and social care; Combined
  and general studies; UoA 2 (Public health) and UoA 29 (Classics).
  Coverage is reported (cells, providers, graduates, REF FTE).

Common design. LEO: all graduates, YAG5, cohorts 2013/14-2016/17 (scripts/62), log median earnings.
Unit = provider x CAH2 'programme'; pay = mean of the released log medians over those cohorts.
Providers: LEO HEIs with a REF submission in the mapped UoA(s), minus scripts/62's exclusions (University
of London umbrella, Open University, Birkbeck). A subject enters when >= 15 providers (scripts/62's
NMIN) have the scores the analysis needs. Rank z = within-subject midrank of the REF GPA over the
providers of that subject's cell, standardised to SD 1 within the subject ('per SD').

  T1 external validity of G. Spearman, across scripts/62's primary providers with REF, of
     institution-level REF GPA with G. 95% CI: provider bootstrap (percentile). Variant: %4*.
  T2 department level.
     (a) Within provider: OLS  pay_ps = a_p + g_s + b z_ps + e  on the CLEAN sample (providers with
         >= 2 subjects; singletons dropped iteratively). b = log points of pay per within-subject SD
         of the REF GPA rank. CI: provider-clustered CR1 (t, G-1 df), and a provider cluster bootstrap
         (percentile) alongside. MDE (80% power, 5% two-sided) = 2.80 x CR1 SE. Compared with the US
         within-institution slope of scripts/58 (institution FE + field FE: +0.0129; main spec with
         field-specific slopes on institution traits: +0.0084), read from SELECTIVITY_DEEP_RESULT.md.
         Sensitivities: EXTENDED sample; raw GPA z instead of rank; %4*; + subject x {zG, zS} slopes
         (zS = institution share of graduates with >= 360 points, scripts/62; the analogue of the US
         main spec; primary providers only); + programme intake control (S-LEO).
     (b) Within subject, across providers: per subject, Spearman(REF GPA, pay) vs Spearman(G, pay) on
         the same providers (scripts/62 primary providers with REF); mean over subjects; difference
         REF - G; partials rho(REF, pay | G) and rho(G, pay | REF). CI: two-stage bootstrap (subjects
         outer; providers within subject inner, as scripts/62).
  T3 students: the T2 designs with the student measure as the outcome.
     (a) Within provider: the T2(a) regression with S-LEO as the outcome; for the comparison, outcome
         and pay are each divided by their within-subject SD over the same cells (SD units); test =
         b_student - b_pay (paired provider cluster bootstrap; CR1 of the stacked system alongside).
         'Does student selectivity track department research standing within a university more than pay
         does?' = this difference.
     (b) Across providers: Spearman(REF, student) vs Spearman(REF, pay), and Spearman(G, student) vs
         Spearman(G, pay), same cells; two-stage bootstrap.
  T4 triad agreement: per CLEAN subject, pairwise Spearman across providers among REF GPA (academic),
     pay (employer) and S-LEO (students), on providers with all three; averaged over subjects; 95% CI
     from a subject bootstrap (subjects resampled); pairwise differences paired; 'which pair agrees
     most' = highest mean, with the bootstrap share of replicates in which each pair is highest.
     Two-stage (subject + provider) CI alongside. S-UCAS demand in the same design on the CAH1 groups
     that coincide with one clean CAH2 subject.
  Reading rule: a pre-specified quantity or difference is 'detected' if its 95% CI excludes 0, else
  'not detected' and reported with its MDE (2.80 SE; SE from the stated CI method, bootstrap SD where
  the CI is a bootstrap). Language is descriptive; no estimate is read as causal.

Seeds: numpy SeedSequence(76).spawn, one child per named analysis in a fixed order; byte-identical on
re-run. BLAS threads fixed at 1 (the matrices are small; faster than 8 and thread-count independent);
single process. Run:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/76_uk_triad.py
Outputs: data/interim/uk_triad_*.csv, UK_TRIAD_RESULT.md (repo root, local only).

Revision 1 (after an independent verification; everything added is exploratory unless it is a
pre-specified quantity that the first version did not print): two-stage CIs for the T4 differences;
the LEO 'low' (1-2 graduates) band cells re-coded as 1 and as 2; scale-free and floor-free versions of
the T3(a) difference; alternative-UoA codings of the extensive margin and of Economics; leave-one-subject-
out ranges; REF 2014 (assessment period 2008-2013, contemporaneous with the LEO cohorts' study) as a
sensitivity for T2(a)/T4 and as a test-retest reliability of the within-provider REF axis.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"     # small matrices: one BLAS thread is faster, and results do not depend on the thread count
import io
import re
import sys
import time
import pickle
import hashlib
import zipfile
import warnings
import importlib.util
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, t as tdist, mannwhitneyu


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s62 = _load("s62", "62_uk_leo.py")          # UK LEO engine (not edited)
wspear, wpartial, norm = s62.wspear, s62.wpartial, s62.norm
NMIN, COHORTS, PA_ALL = s62.NMIN, s62.COHORTS, s62.PA_ALL

SEED = 76
B = 2000                                     # bootstrap replicates (every bootstrap in this script)
Z80 = 0.8416212335729143                     # norm.ppf(0.80)
MDE_K = 1.959963984540054 + Z80              # 2.80: MDE = 2.80 SE (80% power, 5% two-sided)

REF_XLSX = ROOT / "data" / "raw" / "ref2021" / "ref2021_results_all_2022-05-06.xlsx"
REF_MD5 = "3de425a70ac71038722d9287a39eeb06"
REF_URL = "https://results2021.ref.ac.uk/profiles/export-all"
UCAS_ACC = ROOT / "data" / "raw" / "ucas" / "z_140341.zip"    # _1: accepted applicants
UCAS_APP = ROOT / "data" / "raw" / "ucas" / "z_140351.zip"    # _3: main scheme applications
US_MD = ROOT / "SELECTIVITY_DEEP_RESULT.md"
INTERIM = ROOT / "data" / "interim"
OUT_SUM = INTERIM / "uk_triad_summary.csv"
OUT_PANEL = INTERIM / "uk_triad_panel.csv"
OUT_SUBJ = INTERIM / "uk_triad_subjects.csv"
OUT_XW = INTERIM / "uk_triad_crosswalk.csv"
OUT_MD = ROOT / "UK_TRIAD_RESULT.md"
PRESPEC_MD5 = "03bb1e32557d5350b86d6dfe9e89e017"   # md5 of the PRE-SPECIFICATION block at freeze (2026-09-27T18:14Z)
# copy (cp -p, mtime kept) of the frozen block as written to disk at the freeze; checked against PRESPEC_MD5
FROZEN = INTERIM / "uk_triad_prespec_frozen.txt"
# REF 2014 (revision 1, exploratory): results.ref.ac.uk "Download: results", accessed 2026-09-27T21:37:10Z
REF14_XLSX = ROOT / "data" / "raw" / "ref2014" / "ref2014_results_all.xlsx"
REF14_MD5 = "a201b69749b6108fde84991f8a69f948"
REF14_URL = "https://results.ref.ac.uk/DownloadFile/AllResults/xlsx"

# ---- crosswalk CAH2 -> REF UoA (fixed in the pre-specification) --------------------------------
CLEAN = {"Psychology": [4], "Biosciences": [5], "Chemistry": [8], "Physics and astronomy": [9],
         "Mathematical sciences": [10], "Computing": [11], "Engineering": [12],
         "Architecture, building and planning": [13], "Economics": [16], "Business and management": [17],
         "Law": [18], "Politics": [19], "Education and teaching": [23], "Sport and exercise sciences": [24],
         "English studies": [27]}
EXT = {"Medicine and dentistry": [1], "Allied health": [3], "Nursing and midwifery": [3],
       "Pharmacology, toxicology and pharmacy": [3], "Agriculture, food and related studies": [6],
       "Veterinary sciences": [6], "Geography, earth and environmental studies": [7, 14],
       "Materials and technology": [12], "Sociology, social policy and anthropology": [20, 21, 22],
       "History and archaeology": [15, 28], "Philosophy and religious studies": [30, 31],
       "Languages and area studies": [25, 26], "Creative arts and design": [32], "Performing arts": [33],
       "Media, journalism and communications": [34]}
UNMAPPED = ["Medical sciences", "General, applied and forensic sciences", "Health and social care",
            "Combined and general studies"]
# UCAS CAH1 groups that coincide with exactly one clean CAH2 subject
UCAS_CAH1 = {"CAH04": "Psychology", "CAH09": "Mathematical sciences", "CAH11": "Computing",
             "CAH13": "Architecture, building and planning", "CAH16": "Law",
             "CAH17": "Business and management", "CAH22": "Education and teaching"}
# ---- revision 1 (exploratory, chosen after the verification) ------------------------------------
# REF 2014 UoAs of the CLEAN subjects (REF 2014 had 36 UoAs; its engineering UoAs 12-15 were merged
# into REF 2021 UoA 12)
CLEAN14 = {"Psychology": [4], "Biosciences": [5], "Chemistry": [8], "Physics and astronomy": [9],
           "Mathematical sciences": [10], "Computing": [11], "Engineering": [12, 13, 14, 15],
           "Architecture, building and planning": [16], "Economics": [18], "Business and management": [19],
           "Law": [20], "Politics": [21], "Education and teaching": [25], "Sport and exercise sciences": [26],
           "English studies": [29]}
# alternative REF 2021 homes of a subject's research where the provider made no submission to the
# matched UoA (economists in business schools; biologists in allied health / psychology-neuroscience /
# agriculture / earth sciences; architects in engineering / art and design)
ALT_UOA = {"Economics": [17], "Biosciences": [3, 4, 6, 7], "Architecture, building and planning": [12, 32]}

# ---- seeds: one SeedSequence child per named analysis, in this fixed order ---------------------
# Names are only ever appended: SeedSequence.spawn gives child i the spawn key (i,), so appending leaves
# every earlier child, and every earlier draw, unchanged.
ANALYSES = ["t1", "t2a", "t2a_ext", "t2a_raw", "t2a_p4", "t2a_trait", "t2a_intake", "t2a_trait_base",
            "t2b_inner", "t2b_outer", "t3a", "t3a_score", "t3b_inner", "t3b_outer",
            "t4_subj", "t4_inner", "t4_outer", "t4u_subj", "t4u_inner", "t4u_outer",
            "e_margin", "e_wtriad",
            "e_gvert_inner", "e_gvert_outer",   # reserved, not run (planned G-vertical check; no code, no output)
            "t4_ext_subj",
            "t2a_yag3",                          # reserved, not run (planned YAG3 variant; no code, no output)
            "t4_score_subj", "t3bu_inner", "t3bu_outer",
            # revision 1
            "e_low1_t4", "e_low2_t4", "e_low1_t3a", "e_low2_t3a", "e_low1_t2a", "e_low2_t2a",
            "e_t3a_nofloor", "e_margin_excl", "e_margin_alt", "e_econ17_t2a", "e_econ17_t4",
            "e_ref14_t2a", "e_ref14_rel", "e_ref14_t4"]
assert len(set(ANALYSES)) == len(ANALYSES)
_KIDS = dict(zip(ANALYSES, np.random.SeedSequence(SEED).spawn(len(ANALYSES))))


def rng_of(name: str) -> np.random.Generator:
    return np.random.default_rng(_KIDS[name])


def kids_of(name: str, n: int) -> list[np.random.SeedSequence]:
    """n grandchildren of one analysis' SeedSequence (one per subject, in sorted subject order)."""
    return np.random.SeedSequence(_KIDS[name].entropy, spawn_key=_KIDS[name].spawn_key + (7,)).spawn(n)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prespec_md5() -> str:
    t = Path(__file__).read_text(encoding="utf-8")
    a, b = t.index("PRE-SPECIFICATION"), t.index("Seeds: numpy SeedSequence")
    return hashlib.md5(t[a:b].encode()).hexdigest()


# =============================================================================================
# data
# =============================================================================================
def load_ref() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """REF 2021 overall profiles. Returns submission rows, provider x UoA scores (FTE-weighted over
    multiple submissions) and institution scores (FTE-weighted over all submissions)."""
    assert md5(REF_XLSX) == REF_MD5
    d = pd.read_excel(REF_XLSX, header=6)
    d = d[d.Profile == "Overall"].copy()
    d["ukprn"] = d["Institution code (UKPRN)"].astype(str).str.strip()
    d["uoa"] = d["Unit of assessment number"].astype(int)
    d["fte"] = d["FTE of submitted staff"].astype(float)
    for c in ["4*", "3*", "2*", "1*", "Unclassified"]:
        d[c] = pd.to_numeric(d[c], errors="raise").astype(float)
    tot = d[["4*", "3*", "2*", "1*", "Unclassified"]].sum(1)
    assert (tot.sub(100).abs() <= 1.0).all(), tot.describe()
    d["gpa"] = ((4 * d["4*"] + 3 * d["3*"] + 2 * d["2*"] + 1 * d["1*"]) / 100.0).round(9)
    d["p4"] = d["4*"]
    d["joint"] = d["Joint submission"].notna()
    sub = d[["ukprn", "Institution name", "uoa", "Unit of assessment name", "Multiple submission letter",
             "joint", "fte", "gpa", "p4"]].rename(columns={"Institution name": "ref_name",
                                                         "Unit of assessment name": "uoa_name",
                                                         "Multiple submission letter": "letter"})
    assert (sub.fte > 0).all()

    def wavg(g):
        w = g.fte.values
        # rounded to 1e-9 so that equal grades stay exactly tied (weighted means carry float noise)
        return pd.Series({"gpa": round(float(np.average(g.gpa, weights=w)), 9),
                          "p4": round(float(np.average(g.p4, weights=w)), 9), "fte": w.sum(), "nsub": len(g)})
    pu = sub.groupby(["ukprn", "uoa"]).apply(wavg).reset_index()
    inst = sub.groupby("ukprn").apply(wavg).reset_index()
    inst["ref_name"] = inst.ukprn.map(sub.drop_duplicates("ukprn").set_index("ukprn").ref_name)
    inst["power"] = inst.gpa * inst.fte
    return sub, pu, inst


def load_ref14() -> tuple[pd.DataFrame, pd.DataFrame]:
    """REF 2014 overall profiles (revision 1, exploratory). The sheet has a machine-name header row
    ('UKPRN', ..., 'StaffFte', 'FourStar', ...) above the human-readable one; data follow the latter.
    Returns submission rows and provider x UoA scores (FTE-weighted over multiple submissions)."""
    assert md5(REF14_XLSX) == REF14_MD5
    raw = pd.read_excel(REF14_XLSX, header=None, dtype=object)
    k = int(raw.index[raw[0] == "UKPRN"][0])
    k2 = int(raw.index[raw[0] == "Institution code (UKPRN)"][0])
    d = raw.iloc[k2 + 1:].copy()
    d.columns = raw.iloc[k].tolist()
    d = d[d.Profile.notna()]
    assert d.Profile.value_counts().nunique() == 1          # four profiles per submission
    d = d[d.Profile == "Overall"].copy()
    d["ukprn"] = d.UKPRN.astype(str).str.strip()
    d["uoa"] = d.UOA.astype(int)
    d["fte"] = d.StaffFte.astype(float)
    for c in ["FourStar", "ThreeStar", "TwoStar", "OneStar", "Unclassified"]:
        d[c] = pd.to_numeric(d[c], errors="raise").astype(float)
    tot = d[["FourStar", "ThreeStar", "TwoStar", "OneStar", "Unclassified"]].sum(axis=1)
    assert (tot.sub(100).abs() <= 1.0).all()
    d["gpa"] = ((4 * d.FourStar + 3 * d.ThreeStar + 2 * d.TwoStar + 1 * d.OneStar) / 100.0).round(9)
    assert (d.fte > 0).all()
    sub = d[["ukprn", "Institution", "uoa", "UnitOfAssessment", "fte", "gpa"]].reset_index(drop=True)

    def wavg(g):
        return pd.Series({"gpa": round(float(np.average(g.gpa, weights=g.fte.values)), 9), "fte": g.fte.sum()})
    pu = sub.groupby(["ukprn", "uoa"]).apply(wavg).reset_index()
    return sub, pu


def load_pa_raw(ukprns: set) -> pd.DataFrame:
    """Prior-attainment band counts of LEO provider x CAH2 rows at YAG5, cohorts 2013-16, with the
    suppression code kept (scripts/62's loader turns it into NaN): counts are rounded to multiples of 5
    and 'low' marks a count that rounds to 0 but is not 0 (1 or 2 graduates); a band with no graduates
    has no row (no 'z' code occurs in these rows). Returns, per provider x subject x band, the pooled
    released count ('num') and the number of cohorts in which the band is 'low' ('nlow')."""
    keep = ["academic_year", "YAG", "ukprn", "cah2_subject_name", "grads", "characteristic_type",
            "characteristic_value"]
    z = zipfile.ZipFile(s62.LEO_ZIP)
    name = [n for n in z.namelist() if n.startswith("provider_data")][0]
    parts = []
    for ch in pd.read_csv(z.open(name), encoding="latin-1", dtype=str, usecols=keep, chunksize=200000):
        ch = ch[(ch.characteristic_type == "prior_attainment_code") & (ch.YAG == "5") & ch.ukprn.isin(ukprns)
                & (ch.cah2_subject_name != "Total")]
        coh = ch.academic_year.str[:4].astype(int)
        parts.append(ch[coh.isin(COHORTS)].assign(cohort=coh[coh.isin(COHORTS)]))
    d = pd.concat(parts, ignore_index=True)
    d["band"] = d.characteristic_value.str.replace("prior_attainment_", "PA", regex=False).replace({"Not known": "PA_NK"})
    assert not d.duplicated(["ukprn", "cah2_subject_name", "cohort", "band"]).any()
    codes = set(d.grads[~d.grads.str.fullmatch(r"\d+")])
    assert codes <= {"low"}, codes
    num = pd.to_numeric(d.grads, errors="coerce")
    assert (num.dropna() % 5 == 0).all() and (num.dropna() >= 5).all()
    d["num"] = num.fillna(0.0)
    d["nlow"] = (d.grads == "low").astype(float)
    out = d.groupby(["ukprn", "cah2_subject_name", "band"])[["num", "nlow"]].sum().reset_index()
    return out.rename(columns={"cah2_subject_name": "subject"})


def load_ucas() -> tuple[pd.DataFrame, dict]:
    """UCAS end-of-cycle provider x HECoS (CAH1) subject group: accepted applicants (_1) and main scheme
    applications (_3), cycles in the file. Returns long table (provider name, CAH1 code, year, acc, app)
    and a granularity record read from the files themselves."""
    out, info = [], {}
    for zp, suf, col in [(UCAS_ACC, "1", "acc"), (UCAS_APP, "3", "app")]:
        z = zipfile.ZipFile(zp)
        n = f"EOC_HEP_data_resource_2021_015_{suf}.csv"
        raw = z.open(n).read().decode("latin-1").splitlines()
        hdr = [l.strip('"').strip() for l in raw[:15]]
        k = next(i for i, l in enumerate(raw) if l.startswith("Year,Provider name,"))
        df = pd.read_csv(io.StringIO("\n".join(raw[k:])), dtype=str)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        meas = df.columns[3]
        info[col] = dict(file=f"{zp.name}:{n}", md5=md5(zp), measure=meas,
                         cycles=next(h for h in hdr if h.startswith("Cycles")),
                         classes=next(h for h in hdr if h.startswith("Analysis classes")).split(":", 1)[1].strip(),
                         groups=sorted(df["HECoS subject group"].unique()))
        df = df.rename(columns={"Provider name": "pname", "HECoS subject group": "grp", "Year": "year", meas: col})
        df[col] = pd.to_numeric(df[col], errors="coerce")
        out.append(df[["pname", "grp", "year", col]])
    u = out[0].merge(out[1], on=["pname", "grp", "year"], how="outer")
    u = u[(u.pname != "All") & (u.grp != "All")].copy()
    u["cah1"] = u.grp.str.extract(r"^\((CAH\d\d)\)")[0]
    return u, info


# UCAS provider name (code prefix removed) -> LEO UKPRN, where normalised names differ (checked by hand)
UCAS_ALIAS = {"Imperial College London": "10003270", "Northumbria University, Newcastle": "10001282",
              "Royal Holloway, University of London": "10005553", "Goldsmiths, University of London": "10002718",
              "SOAS University of London": "10007780", "St George's, University of London": "10007782",
              "City, University of London": "10001478", "Bristol, University of the West of England": "10007164",
              "Newman University, Birmingham": "10007832", "Rose Bruford College": "10005523",
              "SRUC Scotland's Rural College": "10005700", "UCL (University College London)": "10007784",
              "University of the Highlands and Islands (UHI)": "10007114", "The University of Law": "10039956",
              "Plymouth Marjon University": "10037449", "Solent University (Southampton)": "10006022",
              "BPP University": "10031982"}


def _ucas_keys(nm: str) -> list[str]:
    nm = re.sub(r"^[A-Z]\d{2}\s+", "", nm)
    vs = [nm, re.sub(r",\s*University of London$", "", nm), nm.split(",")[0], re.sub(r"\s+UEA$", "", nm)]
    out = []
    for v in vs:
        k = norm(v)
        for kk in [k, s62._swap(k), k.replace(" upon tyne", "")]:
            if kk and kk not in out:
                out.append(kk)
    return out


def ucas_demand(u: pd.DataFrame, prov: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Demand = pooled main-scheme applications / pooled accepted applicants, per provider x CAH1 group
    in UCAS_CAH1; UKPRN link by normalised provider name (scripts/62's 'University of X' swap, ', University
    of London' and ', <place>' suffixes dropped) plus UCAS_ALIAS."""
    x = u[u.cah1.isin(UCAS_CAH1)].groupby(["pname", "cah1"])[["app", "acc"]].sum(min_count=1).reset_index()
    x = x[(x.acc >= 10) & (x.app > 0)]
    x["demand"] = x.app / x.acc
    x["subject"] = x.cah1.map(UCAS_CAH1)
    hei = prov[prov.provider_type == "HEI"]
    keys: dict[str, set] = {}
    for up, nm in zip(hei.ukprn, hei.provider_name):
        k = norm(nm)
        for kk in {k, s62._swap(k), k.replace(" upon tyne", ""), s62._swap(k.replace(" upon tyne", ""))}:
            if kk:
                keys.setdefault(kk, set()).add(up)
    names = pd.Series(sorted(x.pname.unique()))

    def one(nm):
        base = re.sub(r"^[A-Z]\d{2}\s+", "", nm)
        if base in UCAS_ALIAS:
            return UCAS_ALIAS[base]
        for kk in _ucas_keys(nm):
            if kk in keys and len(keys[kk]) == 1:
                return next(iter(keys[kk]))
        return None
    m = dict(zip(names, names.map(one)))
    x["ukprn"] = x.pname.map(m)
    # one UCAS name per UKPRN (else the UKPRN is ambiguous and dropped)
    dup = x.dropna(subset=["ukprn"]).groupby("ukprn").pname.nunique()
    x.loc[x.ukprn.isin(dup[dup > 1].index), "ukprn"] = None
    info = dict(n_names=int(len(names)), n_matched=int(pd.Series(m).notna().sum()),
                n_dup=int((dup > 1).sum()), unmatched=sorted(k for k, v in m.items() if pd.isna(v)))
    return x.dropna(subset=["ukprn"]), info


def build_panel(leo: pd.DataFrame, prov: pd.DataFrame, pu: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    """Provider x CAH2 panel: pay (mean log median, YAG5, cohorts 2013-16), intake (pooled bands),
    REF scores through the crosswalk, G, institution selectivity."""
    hei = prov[(prov.provider_type == "HEI") & (prov.excluded == "")]
    d = leo[leo.ukprn.isin(hei.ukprn) & (leo.subject != "Total") & (leo.yag == 5) & leo.cohort.isin(COHORTS)]
    a = d[(d.grp == "ALL") & (d.earnings_median > 0)]
    pay = a.assign(ly=np.log(a.earnings_median)).groupby(["ukprn", "subject"]).agg(
        pay=("ly", "mean"), n_coh=("ly", "size"), grads=("grads", "sum"),
        n_earn=("grads_earnings_include", "sum")).reset_index()
    # intake: pooled band counts over the four cohorts (suppressed band counts -> 0, as scripts/62)
    b = d[d.grp.isin(PA_ALL[:-1])].pivot_table(index=["ukprn", "subject"], columns="grp", values="grads",
                                                 aggfunc="sum").reindex(columns=PA_ALL[:-1]).fillna(0.0)
    known = b.sum(1)
    k6 = b[["PA1", "PA2", "PA3", "PA4", "PA5", "PA6"]].sum(1)
    it = pd.DataFrame({"top": (b.PA1 + b.PA2) / known.where(known > 0),
                       "score": (5 * b.PA1 + 4 * b.PA2 + 3 * b.PA3 + 2 * b.PA4 + 1 * b.PA5) / k6.where(k6 > 0),
                       "known": known}).reset_index()
    P = pay.merge(it, on=["ukprn", "subject"], how="left")
    # REF through the crosswalk
    pr = pu.set_index(["ukprn", "uoa"])
    rows = []
    for mp, XW in (("clean", CLEAN), ("ext", EXT)):
        for s, uoas in XW.items():
            for up in P.ukprn[P.subject == s].unique():
                g = [pr.loc[(up, q)] for q in uoas if (up, q) in pr.index]
                if not g:
                    continue
                g = pd.DataFrame(g)
                rows.append(dict(ukprn=up, subject=s,
                                 ref_gpa=round(float(np.average(g.gpa, weights=g.fte)), 9),
                                 ref_p4=round(float(np.average(g.p4, weights=g.fte)), 9), ref_fte=float(g.fte.sum()),
                                 ref_nuoa=len(g)))
    R = pd.DataFrame(rows)
    P = P.merge(R, on=["ukprn", "subject"], how="left")
    P["xw"] = P.subject.map({**{s: "clean" for s in CLEAN}, **{s: "ext" for s in EXT}}).fillna("unmapped")
    pv = prov.set_index("ukprn")
    for c in ["provider_name", "G", "primary", "russell"]:
        P[c] = P.ukprn.map(pv[c])
    P["ref_inst"] = P.ukprn.isin(inst.ukprn)
    # institution selectivity (scripts/62): share of graduates with >= 360 points, provider total rows,
    # mean over cohorts 2013-16 at YAG5
    t = leo[(leo.ukprn != "Total") & (leo.subject == "Total") & (leo.yag == 5) & leo.cohort.isin(COHORTS)]
    tg = t.pivot_table(index=["ukprn", "cohort"], columns="grp", values="grads", aggfunc="first")
    kn = tg.reindex(columns=PA_ALL[:-1]).fillna(0).sum(1)
    tops = (tg.reindex(columns=["PA1", "PA2"]).fillna(0).sum(1) / kn.where(kn > 0)).groupby("ukprn").mean()
    P["inst_top"] = P.ukprn.map(tops)
    return P.sort_values(["subject", "ukprn"]).reset_index(drop=True)


def add_revision_cols(P: pd.DataFrame, pu: pd.DataFrame, pu14: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    """Revision-1 columns (exploratory): intake with 'low' band cells coded 0 (as the panel), 1 and 2;
    the reason a zero intake share is zero; REF 2014 GPA through CLEAN14; alternative-UoA submission;
    Economics scored by UoA 16, else UoA 17."""
    P = P.copy()
    bands = PA_ALL[:-1]
    num = pa.pivot_table(index=["ukprn", "subject"], columns="band", values="num", aggfunc="sum").reindex(columns=PA_ALL).fillna(0.0)
    nlow = pa.pivot_table(index=["ukprn", "subject"], columns="band", values="nlow", aggfunc="sum").reindex(columns=PA_ALL).fillna(0.0)
    key = pd.MultiIndex.from_frame(P[["ukprn", "subject"]])
    for f in (0, 1, 2):
        b = (num + f * nlow)[bands]
        kn = b.sum(axis=1)
        k6 = b[["PA1", "PA2", "PA3", "PA4", "PA5", "PA6"]].sum(axis=1)
        top = ((b.PA1 + b.PA2) / kn.where(kn > 0)).reindex(key).values
        sc = ((5 * b.PA1 + 4 * b.PA2 + 3 * b.PA3 + 2 * b.PA4 + 1 * b.PA5) / k6.where(k6 > 0)).reindex(key).values
        P[f"top_l{f}"], P[f"score_l{f}"] = top, sc
    # the f = 0 coding is the panel's own intake measure (check of this second read of the LEO file)
    both = P.top.notna() | P.top_l0.notna()
    assert np.allclose(P.top[both].values, P.top_l0[both].values, equal_nan=False), "intake re-read differs"
    t12 = num[["PA1", "PA2"]].sum(axis=1).reindex(key).values
    l12 = nlow[["PA1", "PA2"]].sum(axis=1).reindex(key).values
    P["zero_why"] = np.where(P.top != 0, "", np.where(np.nan_to_num(t12) > 0, "released>0?",
                             np.where(np.nan_to_num(l12) > 0, "low", "no rows")))
    assert not (P.zero_why == "released>0?").any()
    # REF 2014
    pr14 = pu14.set_index(["ukprn", "uoa"])
    r14 = {}
    for s, uoas in CLEAN14.items():
        for up in P.ukprn[P.subject == s].unique():
            g = [pr14.loc[(up, q)] for q in uoas if (up, q) in pr14.index]
            if g:
                g = pd.DataFrame(g)
                r14[(up, s)] = round(float(np.average(g.gpa, weights=g.fte)), 9)
    P["ref14_gpa"] = [r14.get((u, s), np.nan) for u, s in zip(P.ukprn, P.subject)]
    # alternative UoAs (extensive margin) and the Economics fallback
    pr = pu.set_index(["ukprn", "uoa"])
    P["alt_sub"] = [any((u, q) in pr.index for q in ALT_UOA.get(s, [])) for u, s in zip(P.ukprn, P.subject)]
    e17 = P.ref_gpa.copy()
    m = (P.subject == "Economics") & P.ref_gpa.isna()
    e17[m] = [pr.loc[(u, 17), "gpa"] if (u, 17) in pr.index else np.nan for u in P.ukprn[m]]
    P["ref_gpa_e17"] = e17
    return P


def load_leo_prov():
    """scripts/62's LEO loader and prestige G. UK_TRIAD_DEVCACHE (a pickle path) is a development
    shortcut only; the reported run does not set it."""
    dev = os.environ.get("UK_TRIAD_DEVCACHE")
    if dev and Path(dev).exists():
        with open(dev, "rb") as f:
            return pickle.load(f)
    leo, pattr = s62.load_leo()
    e = s62.load_edges()
    prov, G, deg, e = s62.build_prestige(e, pattr)
    keep = leo[(leo.yag == 5) & leo.cohort.isin(COHORTS)]
    out = (keep.reset_index(drop=True), prov)
    if dev:
        with open(dev, "wb") as f:
            pickle.dump(out, f)
    return out


# =============================================================================================
# statistics
# =============================================================================================
def zrank(x: np.ndarray) -> np.ndarray:
    r = rankdata(x)
    return (r - r.mean()) / r.std()


def zraw(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std()


def freq_weights(n: int, ss: np.random.SeedSequence) -> np.ndarray:
    rng = np.random.default_rng(ss)
    idx = rng.integers(n, size=(B, n))
    W = np.zeros((B, n), dtype=np.int32)
    np.add.at(W, (np.repeat(np.arange(B), n), idx.ravel()), 1)
    return W


def ci(bs) -> tuple[float, float]:
    bs = np.asarray(bs, float)
    bs = bs[np.isfinite(bs)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def two_stage(stack: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """stack (K subjects, B inner replicates): subjects resampled, each drawn subject contributes an
    independently chosen inner replicate (scripts/62's two-stage bootstrap)."""
    K, Bn = stack.shape
    fi = rng.integers(K, size=(Bn, K))
    ci_ = rng.integers(Bn, size=(Bn, K))
    return np.nanmean(stack[fi, ci_], axis=1)


def drop_singletons(df: pd.DataFrame, nmin: int = NMIN) -> pd.DataFrame:
    while True:
        n0 = len(df)
        df = df[df.groupby("ukprn").subject.transform("size") >= 2]
        df = df[df.groupby("subject").ukprn.transform("size") >= nmin]
        if len(df) == n0:
            return df


def _demean(M: np.ndarray, pid: np.ndarray, G: int) -> np.ndarray:
    """Subtract provider means (column by column)."""
    cnt = np.bincount(pid, minlength=G).astype(float)
    out = np.empty_like(M, dtype=float)
    for j in range(M.shape[1]):
        out[:, j] = M[:, j] - (np.bincount(pid, weights=M[:, j], minlength=G) / cnt)[pid]
    return out


def _others(sid: np.ndarray, C: np.ndarray | None) -> np.ndarray:
    n = len(sid)
    S = np.zeros((n, sid.max() + 1))
    S[np.arange(n), sid] = 1.0
    parts = [S[:, 1:]]
    if C is not None and C.shape[1]:
        parts.append(C)
    return np.hstack(parts)


def fe_resid(M: np.ndarray, pid, sid, C=None) -> np.ndarray:
    """Residuals of each column of M on provider FE + subject FE (+ C): provider FE partialled out by
    demeaning, then the demeaned subject dummies and controls (Frisch-Waugh-Lovell; exact)."""
    G = pid.max() + 1
    Mw = _demean(M, pid, G)
    Xw = _demean(_others(sid, C), pid, G)
    coef, *_ = np.linalg.lstsq(Xw, Mw, rcond=None)
    return Mw - Xw @ coef


def fe_point(Y: np.ndarray, z: np.ndarray, pid, sid, C=None):
    """OLS of each column of Y on z + provider FE + subject FE (+ controls C). FWL: returns beta (m,),
    z residual, Y residuals (after removing z)."""
    R = fe_resid(np.column_stack([Y, z]), pid, sid, C)
    zt = R[:, -1]
    Yt = R[:, :-1]
    beta = zt @ Yt / (zt @ zt)
    U = Yt - np.outer(zt, beta)
    return beta, zt, U


def fe_fit(df: pd.DataFrame, ycols: list[str], zcol: str, ccols: list[str] | None, ss_name: str | None,
           pair: tuple[int, int] | None = None) -> dict:
    """Two-way FE regression with provider-clustered CR1 SEs and a provider cluster bootstrap
    (clusters resampled with replacement; a drawn-twice provider gets two FE). FE nested in the
    clusters are not counted in the small-sample factor (reghdfe convention)."""
    df = df.sort_values(["ukprn", "subject"]).reset_index(drop=True)
    pid = pd.factorize(df.ukprn, sort=True)[0]
    sid = pd.factorize(df.subject, sort=True)[0]
    Y = df[ycols].values.astype(float)
    z = df[zcol].values.astype(float)
    C = df[ccols].values.astype(float) if ccols else None
    beta, zt, U = fe_point(Y, z, pid, sid, C)
    N, Gc, S = len(df), pid.max() + 1, sid.max() + 1
    K = 1 + (S - 1) + (C.shape[1] if C is not None else 0) + 1
    cf = Gc / (Gc - 1) * (N - 1) / (N - K)
    sc = np.zeros((Gc, len(ycols)))
    np.add.at(sc, pid, zt[:, None] * U)
    den = (zt @ zt) ** 2
    se = np.sqrt(cf * (sc ** 2).sum(0) / den)
    tq = float(tdist.ppf(0.975, Gc - 1))
    out = dict(beta=beta, se=se, lo=beta - tq * se, hi=beta + tq * se, N=N, G=Gc, S=S,
               p=2 * tdist.sf(np.abs(beta / se), Gc - 1), sd_z=float(z.std()), df=df)
    if pair is not None:
        i, j = pair
        sd = np.sqrt(cf * ((sc[:, i] - sc[:, j]) ** 2).sum() / den)
        out["diff"] = beta[i] - beta[j]
        out["diff_se"] = sd
        out["diff_lo"], out["diff_hi"] = out["diff"] - tq * sd, out["diff"] + tq * sd
    if ss_name is not None:
        rng = rng_of(ss_name)
        groups = [np.flatnonzero(pid == g) for g in range(Gc)]
        bs = np.full((B, len(ycols)), np.nan)
        for b in range(B):
            draw = rng.integers(Gc, size=Gc)
            rows = np.concatenate([groups[g] for g in draw])
            newp = np.concatenate([np.full(len(groups[g]), k) for k, g in enumerate(draw)])
            sidb = pd.factorize(sid[rows], sort=True)[0]
            Cb = C[rows] if C is not None else None
            bb, _, _ = fe_point(Y[rows], z[rows], newp, sidb, Cb)
            bs[b] = bb
        out["bs"] = bs
        out["blo"] = np.nanpercentile(bs, 2.5, axis=0)
        out["bhi"] = np.nanpercentile(bs, 97.5, axis=0)
        out["bse"] = np.nanstd(bs, axis=0)
        if pair is not None:
            dd = bs[:, pair[0]] - bs[:, pair[1]]
            out["diff_bs"] = dd
            out["diff_blo"], out["diff_bhi"] = ci(dd)
            out["diff_bse"] = float(np.nanstd(dd))
    return out


def subject_cells(P: pd.DataFrame, need: list[str], subjects: list[str], primary: bool) -> dict[str, pd.DataFrame]:
    x = P[P.subject.isin(subjects)].dropna(subset=need)
    if primary:
        x = x[x.primary == True]  # noqa: E712
    out = {}
    for s, g in x.groupby("subject"):
        if len(g) >= NMIN:
            out[s] = g.sort_values("ukprn").reset_index(drop=True)
    return out


def across(cells: dict[str, pd.DataFrame], stats: dict, inner: str, outer: str) -> dict:
    """Per-subject across-provider statistics with provider frequency weights (exact resampling);
    stats = {name: f(g, W) -> (B,) or (1,)}. Returns per-subject points, inner replicates, and the
    two-stage mean over subjects for each statistic and for requested differences."""
    subs = sorted(cells)
    kid = kids_of(inner, len(subs))
    pts = {k: {} for k in stats}
    reps = {k: {} for k in stats}
    for s, ss in zip(subs, kid):
        g = cells[s]
        W = freq_weights(len(g), ss)
        one = np.ones((1, len(g)))
        for k, f in stats.items():
            pts[k][s] = float(f(g, one)[0])
            reps[k][s] = f(g, W)
    rng = rng_of(outer)
    # one outer draw shared by every statistic (paired differences)
    K = len(subs)
    fi = rng.integers(K, size=(B, K))
    ci_ = rng.integers(B, size=(B, K))
    ts = {}
    for k in stats:
        st = np.vstack([reps[k][s] for s in subs])
        ts[k] = np.nanmean(st[fi, ci_], axis=1)
    mean = {k: float(np.mean([pts[k][s] for s in subs])) for k in stats}
    return dict(subs=subs, pts=pts, ts=ts, mean=mean, n={s: len(cells[s]) for s in subs})


def summ(est, bs) -> dict:
    lo, hi = ci(bs)
    se = float(np.nanstd(bs))
    return dict(est=float(est), lo=lo, hi=hi, se=se, mde=MDE_K * se)


def us_slopes() -> dict:
    """US within-institution slopes of scripts/58, read from its write-up."""
    t = US_MD.read_text(encoding="utf-8")
    out = {}
    for key, pat in [("a", r"\| within-institution β: \(a\) institution FE \+ field FE \+ β z\(F\) \[scripts/55 \(a\)\] \| ([\d,]+) programs, (\d+) institutions \| (\d+) \| ([+-]\d\.\d+) \[([+-]\d\.\d+), ([+-]\d\.\d+)\] \|"),
                     ("main", r"\| within-institution β: \(a\) \+ field × \{SAT, ADM, inst\. Pell, brand G\} \[main\] \| ([\d,]+) programs, (\d+) institutions \| (\d+) \| ([+-]\d\.\d+) \[([+-]\d\.\d+), ([+-]\d\.\d+)\] \|")]:
        m = re.search(pat, t)
        assert m, key
        out[key] = dict(n=int(m.group(1).replace(",", "")), inst=int(m.group(2)), fields=int(m.group(3)),
                        est=float(m.group(4)), lo=float(m.group(5)), hi=float(m.group(6)))
    for key, lab in [("a", r"\(a\) institution FE \+ field FE \+ β z\(F\) \[scripts/55 \(a\)\]"),
                     ("main", r"\(a\) \+ field × \{SAT, ADM, inst\. Pell, brand G\} \[main\]")]:
        m = re.search(r"\| " + lab + r" \| [\d,]+ \| \d+ \| \d+ \| \d+ \| [+-]\d\.\d+ \((\d\.\d+)\) \|", t)
        assert m, key
        out[key]["se"] = float(m.group(1))
    return out


def corr_cols(R: np.ndarray) -> np.ndarray:
    C = np.corrcoef(R, rowvar=False)
    return np.array([C[0, 1], C[0, 2], C[1, 2]])


# =============================================================================================
# main analysis
# =============================================================================================
def analyse() -> dict:
    t0 = time.time()
    assert prespec_md5() == PRESPEC_MD5, "pre-specification block changed since the freeze"
    assert md5(FROZEN) == PRESPEC_MD5, "frozen copy of the pre-specification differs from the block"
    sub, pu, inst = load_ref()
    sub14, pu14 = load_ref14()
    leo, prov = load_leo_prov()
    P = build_panel(leo, prov, pu, inst)
    u, uinfo = load_ucas()
    dem, dinfo = ucas_demand(u, prov)
    P = P.merge(dem[["ukprn", "subject", "demand", "app", "acc"]], on=["ukprn", "subject"], how="left")
    assert not P.duplicated(["ukprn", "subject"]).any()
    pa = load_pa_raw(set(P.ukprn))
    P = add_revision_cols(P, pu, pu14, pa)
    assert set(CLEAN) | set(EXT) | set(UNMAPPED) <= set(P.subject.unique()) | {"Veterinary sciences"}
    rows: list[dict] = []

    def put(sec, q, spec, est, lo=np.nan, hi=np.nan, k=np.nan, se=np.nan, mde=np.nan, p=np.nan, verdict=""):
        rows.append(dict(section=sec, quantity=q, spec=spec, est=est, lo=lo, hi=hi, se=se, mde=mde, p=p, k=k,
                         verdict=verdict))

    def verdict(lo, hi):
        return "detected" if (lo > 0 or hi < 0) else "not detected"

    def putb(sec, q, spec, s, k, pre=True):
        put(sec, q, spec, s["est"], s["lo"], s["hi"], k, s["se"], s["mde"],
            verdict=verdict(s["lo"], s["hi"]) if pre else "")

    ctx: dict = dict(P=P, prov=prov, inst=inst, pu=pu, sub=sub, uinfo=uinfo, dinfo=dinfo, sub14=sub14)

    # ------------------------------------------------------------------ coverage
    hei = prov[(prov.provider_type == "HEI") & (prov.excluded == "")]
    ref_ids = set(inst.ukprn)
    cov = {}
    cov["ref_inst"] = len(inst)
    cov["ref_fte"] = float(sub.fte.sum())
    cov["ref_inst_in_leo_hei"] = int(len(ref_ids & set(hei.ukprn)))
    cov["ref_fte_in_leo_hei"] = float(sub[sub.ukprn.isin(hei.ukprn)].fte.sum() / sub.fte.sum())
    prim = prov[prov.primary]
    cov["primary"] = int(len(prim))
    cov["primary_with_ref"] = int(prim.ukprn.isin(ref_ids).sum())
    cu = {q for v in CLEAN.values() for q in v}
    eu = cu | {q for v in EXT.values() for q in v}
    cov["ref_fte_clean_uoa"] = float(sub[sub.uoa.isin(cu)].fte.sum() / sub.fte.sum())
    cov["ref_fte_ext_uoa"] = float(sub[sub.uoa.isin(eu)].fte.sum() / sub.fte.sum())
    pp = P[P.pay.notna()]
    cov["prog"] = int(len(pp))
    cov["prog_grads"] = float(pp.grads.sum())
    for mp in ["clean", "ext", "unmapped"]:
        m = pp.xw == mp
        cov[f"prog_{mp}"] = int(m.sum())
        cov[f"prog_{mp}_ref"] = int((m & pp.ref_gpa.notna()).sum())
        cov[f"grads_{mp}"] = float(pp.grads[m].sum() / pp.grads.sum())
        cov[f"grads_{mp}_ref"] = float(pp.grads[m & pp.ref_gpa.notna()].sum() / pp.grads.sum())
    cov["n_hei"] = int(pp.ukprn.nunique())
    cov["n_hei_ref"] = int(pp.ukprn[pp.ref_inst].nunique())
    ctx["cov"] = cov
    # per subject coverage
    subj_rows = []
    for s in sorted(set(CLEAN) | set(EXT) | set(UNMAPPED)):
        g = pp[pp.subject == s]
        subj_rows.append(dict(subject=s, xw=("clean" if s in CLEAN else "ext" if s in EXT else "unmapped"),
                              uoa="+".join(str(q) for q in (CLEAN.get(s) or EXT.get(s) or [])),
                              n_prog=len(g), n_ref=int(g.ref_gpa.notna().sum()),
                              share_ref=float(g.ref_gpa.notna().mean()) if len(g) else np.nan,
                              grads_share_ref=float(g.grads[g.ref_gpa.notna()].sum() / g.grads.sum()) if len(g) else np.nan,
                              n_ref_ref_inst=int(g.ref_inst.sum())))
    SJ = pd.DataFrame(subj_rows).set_index("subject")

    # ------------------------------------------------------------------ T1
    X = prim.merge(inst, on="ukprn", how="inner").sort_values("ukprn").reset_index(drop=True)
    it = P.drop_duplicates("ukprn").set_index("ukprn").inst_top
    X["inst_top"] = X.ukprn.map(it)
    n1 = len(X)
    W = freq_weights(n1, kids_of("t1", 1)[0])
    one = np.ones((1, n1))
    Gv, gpa, p4 = X.G.values, X.gpa.values, X.p4.values
    t1 = {}
    for k, (a, b_) in {"G_gpa": (Gv, gpa), "G_p4": (Gv, p4), "G_fte": (Gv, X.fte.values),
                       "G_power": (Gv, X.power.values)}.items():
        t1[k] = summ(wspear(a, b_, one)[0], wspear(a, b_, W))
    Xs = X[X.inst_top.notna()].reset_index(drop=True)
    ms = X.inst_top.notna().values
    Ws, ones_s = W[:, ms], np.ones((1, ms.sum()))
    for k, f in {"G_sel": lambda w: wspear(Xs.G.values, Xs.inst_top.values, w),
                 "gpa_sel": lambda w: wspear(Xs.gpa.values, Xs.inst_top.values, w),
                 "G_gpa_s": lambda w: wspear(Xs.G.values, Xs.gpa.values, w),
                 "pG_gpa_sel": lambda w: wpartial(Xs.G.values, Xs.gpa.values, Xs[["inst_top"]].values, w),
                 "psel_gpa_G": lambda w: wpartial(Xs.inst_top.values, Xs.gpa.values, Xs[["G"]].values, w)}.items():
        t1[k] = summ(f(ones_s)[0], f(Ws))
    d_ = wspear(Xs.G.values, Xs.gpa.values, Ws) - wspear(Xs.inst_top.values, Xs.gpa.values, Ws)
    t1["G_minus_sel_gpa"] = summ(t1["G_gpa_s"]["est"] - t1["gpa_sel"]["est"], d_)
    rg = X.russell.values.astype(bool)
    t1["auc_gpa"] = float(mannwhitneyu(gpa[rg], gpa[~rg]).statistic / (rg.sum() * (~rg).sum()))
    t1["auc_G"] = float(mannwhitneyu(Gv[rg], Gv[~rg]).statistic / (rg.sum() * (~rg).sum()))
    t1["n"], t1["n_sel"], t1["n_rg"] = n1, int(ms.sum()), int(rg.sum())
    ctx["t1"], ctx["X1"] = t1, X

    # ------------------------------------------------------------------ T2(a)
    def fe_sample(mapset, need, primary=False, zsrc="ref_gpa", zfun=zrank, extra_z=()):
        x = P[P.xw.isin(mapset)].dropna(subset=need)
        if primary:
            x = x[x.primary == True]  # noqa: E712
        x = x[x.groupby("subject").ukprn.transform("size") >= NMIN].copy()
        x["z"] = x.groupby("subject")[zsrc].transform(lambda v: pd.Series(zfun(v.values), index=v.index))
        for c in extra_z:
            x["z_" + c] = x.groupby("subject")[c].transform(lambda v: pd.Series(zrank(v.values), index=v.index))
        return drop_singletons(x)

    def inter(df, cols):
        subs = sorted(df.subject.unique())
        out = []
        for c in cols:
            for s in subs:
                nm = f"{c}|{s}"
                df[nm] = np.where(df.subject == s, df[c], 0.0)
                out.append(nm)
        return out

    t2 = {}
    d0 = fe_sample(["clean"], ["ref_gpa", "pay"])
    t2["main"] = fe_fit(d0, ["pay"], "z", None, "t2a")
    d1 = fe_sample(["clean", "ext"], ["ref_gpa", "pay"])
    t2["ext"] = fe_fit(d1, ["pay"], "z", None, "t2a_ext")
    d2 = fe_sample(["clean"], ["ref_gpa", "pay"], zfun=zraw)
    t2["raw"] = fe_fit(d2, ["pay"], "z", None, "t2a_raw")
    d3 = fe_sample(["clean"], ["ref_p4", "pay"], zsrc="ref_p4")
    t2["p4"] = fe_fit(d3, ["pay"], "z", None, "t2a_p4")
    d4 = fe_sample(["clean"], ["ref_gpa", "pay", "G", "inst_top"], primary=True, extra_z=("G", "inst_top"))
    t2["trait_base"] = fe_fit(d4, ["pay"], "z", None, "t2a_trait_base")
    cc = inter(d4, ["z_G", "z_inst_top"])
    t2["trait"] = fe_fit(d4, ["pay"], "z", cc, "t2a_trait")
    d5 = fe_sample(["clean"], ["ref_gpa", "pay", "top"], extra_z=("top",))
    cc5 = inter(d5, ["z_top"])
    t2["intake"] = fe_fit(d5, ["pay"], "z", cc5, "t2a_intake")
    # same seed as "t2a_intake": identical cluster draws, so the two fits are paired
    t2["intake_base"] = fe_fit(d5, ["pay"], "z", None, "t2a_intake")
    bb, bi = t2["intake_base"]["bs"][:, 0], t2["intake"]["bs"][:, 0]
    t2["intake_drop"] = summ(float(t2["intake_base"]["beta"][0] - t2["intake"]["beta"][0]), bb - bi)
    t2["intake_share"] = summ(1 - float(t2["intake"]["beta"][0]) / float(t2["intake_base"]["beta"][0]), 1 - bi / bb)
    t2["us"] = us_slopes()
    ctx["t2"] = t2

    # ------------------------------------------------------------------ T2(b), T3(b)
    cells2 = subject_cells(P, ["ref_gpa", "pay", "G"], list(CLEAN), primary=True)
    st2 = {"rho_ref": lambda g, w: wspear(g.ref_gpa.values, g.pay.values, w),
           "rho_G": lambda g, w: wspear(g.G.values, g.pay.values, w),
           "rho_refG": lambda g, w: wspear(g.ref_gpa.values, g.G.values, w),
           "p_ref_G": lambda g, w: wpartial(g.ref_gpa.values, g.pay.values, g[["G"]].values, w),
           "p_G_ref": lambda g, w: wpartial(g.G.values, g.pay.values, g[["ref_gpa"]].values, w)}
    A2 = across(cells2, st2, "t2b_inner", "t2b_outer")
    ctx["A2"] = A2
    cells3 = subject_cells(P, ["ref_gpa", "pay", "G", "top", "inst_top"], list(CLEAN), primary=True)
    st3 = {"ref_top": lambda g, w: wspear(g.ref_gpa.values, g.top.values, w),
           "ref_pay": lambda g, w: wspear(g.ref_gpa.values, g.pay.values, w),
           "G_top": lambda g, w: wspear(g.G.values, g.top.values, w),
           "G_pay": lambda g, w: wspear(g.G.values, g.pay.values, w),
           "p_ref_pay_sel": lambda g, w: wpartial(g.ref_gpa.values, g.pay.values, g[["inst_top"]].values, w),
           "p_ref_top_sel": lambda g, w: wpartial(g.ref_gpa.values, g.top.values, g[["inst_top"]].values, w),
           "sel_pay": lambda g, w: wspear(g.inst_top.values, g.pay.values, w),
           "sel_top": lambda g, w: wspear(g.inst_top.values, g.top.values, w)}
    A3 = across(cells3, st3, "t3b_inner", "t3b_outer")
    ctx["A3"] = A3
    # S-UCAS in the T3(b) design (pre-specified): primary providers, CAH1 groups that are one clean CAH2 subject
    cells3u = subject_cells(P, ["ref_gpa", "pay", "G", "demand"], list(UCAS_CAH1.values()), primary=True)
    st3u = {"ref_dem": lambda g, w: wspear(g.ref_gpa.values, g.demand.values, w),
            "ref_pay": lambda g, w: wspear(g.ref_gpa.values, g.pay.values, w),
            "G_dem": lambda g, w: wspear(g.G.values, g.demand.values, w),
            "G_pay": lambda g, w: wspear(g.G.values, g.pay.values, w)}
    ctx["A3u"] = across(cells3u, st3u, "t3bu_inner", "t3bu_outer")

    # ------------------------------------------------------------------ T3(a)
    d6 = fe_sample(["clean"], ["ref_gpa", "pay", "top", "score"])
    for c in ["pay", "top", "score"]:
        d6[c + "_sd"] = d6[c] / d6.groupby("subject")[c].transform(lambda v: v.std(ddof=0))
    t3 = {"main": fe_fit(d6, ["pay_sd", "top_sd", "pay", "top"], "z", None, "t3a", pair=(1, 0)),
          "score": fe_fit(d6, ["pay_sd", "score_sd"], "z", None, "t3a_score", pair=(1, 0))}
    # exploratory: within-provider triad (FE-residualised z(REF), pay, intake; Pearson)
    dd = d6.sort_values(["ukprn", "subject"]).reset_index(drop=True)
    pid = pd.factorize(dd.ukprn, sort=True)[0]
    sid = pd.factorize(dd.subject, sort=True)[0]
    M = dd[["z", "pay", "top"]].values.astype(float)
    wt_pt = corr_cols(fe_resid(M, pid, sid))
    rng = rng_of("e_wtriad")
    groups = [np.flatnonzero(pid == g) for g in range(pid.max() + 1)]
    wbs = np.full((B, 3), np.nan)
    for b in range(B):
        draw = rng.integers(len(groups), size=len(groups))
        rr = np.concatenate([groups[g] for g in draw])
        newp = np.concatenate([np.full(len(groups[g]), k) for k, g in enumerate(draw)])
        wbs[b] = corr_cols(fe_resid(M[rr], newp, pd.factorize(sid[rr], sort=True)[0]))
    t3["wtriad"] = dict(pt=wt_pt, bs=wbs, n=len(dd), G=len(groups))
    # exploratory: extensive margin (did the provider submit to the subject's UoA?)
    em = P[(P.xw == "clean") & P.pay.notna() & P.top.notna() & P.ref_inst].copy()
    em["sub"] = em.ref_gpa.notna().astype(float)
    em = em[em.groupby("subject").ukprn.transform("size") >= NMIN]
    em = drop_singletons(em)
    for c in ["pay", "top"]:
        em[c + "_sd"] = em[c] / em.groupby("subject")[c].transform(lambda v: v.std(ddof=0))
    t3["margin"] = fe_fit(em, ["pay_sd", "top_sd", "pay", "top"], "sub", None, "e_margin", pair=(1, 0))
    t3["margin_share"] = float(em["sub"].mean())
    ctx["t3"] = t3

    # ------------------------------------------------------------------ T4
    def triad(cells, pairs, subj_name, inner, outer):
        subs = sorted(cells)
        pts = np.array([[spearmanr(cells[s][a], cells[s][b_])[0] for a, b_ in pairs] for s in subs])
        rng_ = rng_of(subj_name)
        fi = rng_.integers(len(subs), size=(B, len(subs)))
        bs = pts[fi].mean(1)                                          # (B, npairs)
        st = {f"{a}~{b_}": (lambda g, w, a=a, b_=b_: wspear(g[a].values, g[b_].values, w)) for a, b_ in pairs}
        A = across(cells, st, inner, outer) if inner else None
        return dict(subs=subs, pts=pts, bs=bs, A=A, pairs=pairs, n={s: len(cells[s]) for s in subs})

    pairs = [("ref_gpa", "pay"), ("ref_gpa", "top"), ("pay", "top")]
    c4 = subject_cells(P, ["ref_gpa", "pay", "top"], list(CLEAN), primary=False)
    T4 = triad(c4, pairs, "t4_subj", "t4_inner", "t4_outer")
    c4e = subject_cells(P, ["ref_gpa", "pay", "top"], list(CLEAN) + list(EXT), primary=False)
    T4e = triad(c4e, pairs, "t4_ext_subj", None, None)
    pairs_u = [("ref_gpa", "pay"), ("ref_gpa", "top"), ("pay", "top"), ("ref_gpa", "demand"),
               ("pay", "demand"), ("top", "demand")]
    c4u = subject_cells(P, ["ref_gpa", "pay", "top", "demand"], list(UCAS_CAH1.values()), primary=False)
    T4u = triad(c4u, pairs_u, "t4u_subj", "t4u_inner", "t4u_outer")
    c4s = subject_cells(P, ["ref_gpa", "pay", "score"], list(CLEAN), primary=False)
    T4s = triad(c4s, [("ref_gpa", "pay"), ("ref_gpa", "score"), ("pay", "score")], "t4_score_subj", None, None)
    ctx.update(T4=T4, T4e=T4e, T4u=T4u, T4s=T4s)
    # floor of the intake share: programmes with no graduate at >= 360 points
    z4 = pd.concat(c4.values())
    ctx["top_zero"] = dict(share=float((z4.top == 0).mean()), n=len(z4),
                           share_primary=float((z4[z4.primary == True].top == 0).mean()))  # noqa: E712

    # ================================================================== revision 1 (exploratory)
    rv: dict = {}
    # why a zero intake share is zero (T4 programmes): both >= 360-point bands 'low' in every cohort, or no row
    rv["zero_why"] = {k: int(v) for k, v in z4.zero_why.value_counts().items()}
    # 'low' band cells (1-2 graduates) coded as 1 and as 2 instead of 0, on the cells of the original analyses
    # (T4: c4; T3(a): d6; T2(a) intake control: d5), so that only the coding changes
    rv["low"] = {}
    for f in (1, 2):
        tc = f"top_l{f}"
        c = {s: g.copy() for s, g in c4.items()}
        Tf = triad(c, [("ref_gpa", "pay"), ("ref_gpa", tc), ("pay", tc)], f"e_low{f}_t4", None, None)
        d6f = d6.copy()
        d6f[tc + "_sd"] = d6f[tc] / d6f.groupby("subject")[tc].transform(lambda v: v.std(ddof=0))
        t3f = fe_fit(d6f, ["pay_sd", tc + "_sd"], "z", None, f"e_low{f}_t3a", pair=(1, 0))
        d5f = d5[[c_ for c_ in d5.columns if "|" not in c_]].copy()
        d5f["z_" + tc] = d5f.groupby("subject")[tc].transform(lambda v: pd.Series(zrank(v.values), index=v.index))
        c5f = inter(d5f, ["z_" + tc])
        wi = fe_fit(d5f, ["pay"], "z", c5f, f"e_low{f}_t2a")
        wo = fe_fit(d5f, ["pay"], "z", None, f"e_low{f}_t2a")     # same draws: paired
        bo, bi_ = wo["bs"][:, 0], wi["bs"][:, 0]
        rv["low"][f] = dict(T4=Tf, zero=float((z4[tc] == 0).mean()), n4=len(z4), t3=t3f, t2_with=wi, t2_without=wo,
                            t2_drop=summ(float(wo["beta"][0] - wi["beta"][0]), bo - bi_),
                            t2_share=summ(1 - float(wi["beta"][0]) / float(wo["beta"][0]), 1 - bi_ / bo))
    # T3(a) without the small-SD ('floor') subjects: within-subject SD of the intake share below a third of the
    # median subject's SD (chosen after the verification; the SD-unit scaling inflates these subjects)
    sdv = d6.groupby("subject").top.apply(lambda v: float(v.std(ddof=0)))
    fl = d6.groupby("subject").top.apply(lambda v: float((v == 0).mean()))
    floor_subj = sorted(sdv[sdv < float(np.median(sdv)) / 3].index)
    d6n = drop_singletons(d6[~d6.subject.isin(floor_subj)].copy())
    for c_ in ["pay", "top"]:
        d6n[c_ + "_sd"] = d6n[c_] / d6n.groupby("subject")[c_].transform(lambda v: v.std(ddof=0))
    rv["t3_nofloor"] = fe_fit(d6n, ["pay_sd", "top_sd"], "z", None, "e_t3a_nofloor", pair=(1, 0))
    rv["floor"] = {s: dict(zero=float(fl[s]), sd=float(sdv[s])) for s in floor_subj}
    rv["floor_sd_median"] = float(np.median(sdv))
    # extensive margin: without the subjects whose research often sits in another UoA; alternative UoAs counted
    em0 = P[(P.xw == "clean") & P.pay.notna() & P.top.notna() & P.ref_inst].copy()
    rv["margin"] = {}
    for nm, ss, dfm in [("excl", "e_margin_excl", em0[~em0.subject.isin(ALT_UOA)].copy()),
                        ("alt", "e_margin_alt", em0.copy())]:
        dfm["sub"] = (dfm.ref_gpa.notna() | (dfm.alt_sub if nm == "alt" else False)).astype(float)
        dfm = drop_singletons(dfm[dfm.groupby("subject").ukprn.transform("size") >= NMIN])
        rv["margin"][nm] = fe_fit(dfm, ["pay", "top"], "sub", None, ss)
        rv["margin"][nm]["share"] = float(dfm["sub"].mean())
    # no score is not no research: programmes of the ALT_UOA subjects without a matched-UoA score
    pp_ = P[P.pay.notna()]
    rv["noscore"] = {}
    for s in ALT_UOA:
        g = pp_[pp_.subject == s]
        ns = g[g.ref_gpa.isna()]
        rv["noscore"][s] = dict(n=len(g), n_ref=int(g.ref_gpa.notna().sum()), n_none=len(ns),
                                n_none_alt=int(ns.alt_sub.sum()), n_rg=int((g.russell == True).sum()),  # noqa: E712
                                rg_none=sorted(ns[ns.russell == True].provider_name),  # noqa: E712
                                rg_none_alt=int(ns[ns.russell == True].alt_sub.sum()))  # noqa: E712
    # leave one clean subject out: T2(a) point and CR1; T4 subject means
    loo = {}
    for s in sorted(d0.subject.unique()):
        x = drop_singletons(d0[d0.subject != s].copy())
        r_ = fe_fit(x, ["pay"], "z", None, None)
        k_ = [i for i, s2 in enumerate(T4["subs"]) if s2 != s]
        loo[s] = dict(beta=float(r_["beta"][0]), lo=float(r_["lo"][0]), hi=float(r_["hi"][0]),
                      t4=T4["pts"][k_].mean(0))
    rv["loo"] = loo
    # Economics scored by UoA 16, else UoA 17
    de = fe_sample(["clean"], ["ref_gpa_e17", "pay"], zsrc="ref_gpa_e17")
    rv["e17_t2a"] = fe_fit(de, ["pay"], "z", None, "e_econ17_t2a")
    c4x = subject_cells(P, ["ref_gpa_e17", "pay", "top"], list(CLEAN), primary=False)
    rv["e17_t4"] = triad(c4x, [("ref_gpa_e17", "pay"), ("ref_gpa_e17", "top"), ("pay", "top")], "e_econ17_t4", None, None)
    rv["e17_n"] = int(P[(P.subject == "Economics") & P.pay.notna()].ref_gpa_e17.notna().sum())
    # REF 2014: coverage, T2(a), T4, and the test-retest reliability of the within-provider REF axis
    heis = set(hei.ukprn)
    rv["r14_cov"] = dict(inst=int(sub14.ukprn.nunique()), subs=len(sub14),
                         inst_hei=int(len(set(sub14.ukprn) & heis)),
                         clean14=int((pp_.xw == "clean").sum()),
                         clean14_ref=int(((pp_.xw == "clean") & pp_.ref14_gpa.notna()).sum()),
                         clean_both=int(((pp_.xw == "clean") & pp_.ref14_gpa.notna() & pp_.ref_gpa.notna()).sum()))
    d14 = fe_sample(["clean"], ["ref14_gpa", "pay"], zsrc="ref14_gpa")
    rv["r14_t2a"] = fe_fit(d14, ["pay"], "z", None, "e_ref14_t2a")
    c414 = subject_cells(P, ["ref14_gpa", "pay", "top"], list(CLEAN), primary=False)
    rv["r14_t4"] = triad(c414, [("ref14_gpa", "pay"), ("ref14_gpa", "top"), ("pay", "top")], "e_ref14_t4", None, None)
    dc = P[P.xw == "clean"].dropna(subset=["ref_gpa", "ref14_gpa", "pay"])
    dc = dc[dc.groupby("subject").ukprn.transform("size") >= NMIN].copy()
    for zc_, src in [("z", "ref_gpa"), ("z14", "ref14_gpa")]:
        dc[zc_] = dc.groupby("subject")[src].transform(lambda v: pd.Series(zrank(v.values), index=v.index))
    dc = drop_singletons(dc).sort_values(["ukprn", "subject"]).reset_index(drop=True)
    rv["rel_across"] = float(np.mean([spearmanr(g.ref_gpa, g.ref14_gpa)[0] for _, g in dc.groupby("subject")]))
    pid_c = pd.factorize(dc.ukprn, sort=True)[0]
    sid_c = pd.factorize(dc.subject, sort=True)[0]
    Mc = dc[["pay", "z", "z14"]].values.astype(float)

    def relstats(R_):
        y_, a_, b_ = R_[:, 0], R_[:, 1], R_[:, 2]
        r_ = float(a_ @ b_ / np.sqrt((a_ @ a_) * (b_ @ b_)))
        b21, b14 = float(a_ @ y_ / (a_ @ a_)), float(b_ @ y_ / (b_ @ b_))
        return np.array([r_, b21, b14, b21 / r_, float(b_ @ y_ / (b_ @ a_))])
    rel_pt = relstats(fe_resid(Mc, pid_c, sid_c))
    rng_r = rng_of("e_ref14_rel")
    grp_c = [np.flatnonzero(pid_c == g) for g in range(pid_c.max() + 1)]
    rel_bs = np.full((B, 5), np.nan)
    for b in range(B):
        draw = rng_r.integers(len(grp_c), size=len(grp_c))
        rr = np.concatenate([grp_c[g] for g in draw])
        newp = np.concatenate([np.full(len(grp_c[g]), k) for k, g in enumerate(draw)])
        rel_bs[b] = relstats(fe_resid(Mc[rr], newp, pd.factorize(sid_c[rr], sort=True)[0]))
    rv["rel"] = dict(pt=rel_pt, bs=rel_bs, N=len(dc), G=len(grp_c), S=int(sid_c.max() + 1))
    ctx["rv"] = rv

    # per-subject table
    for s in SJ.index:
        for nm, A, keys in [("t2b", A2, ["rho_ref", "rho_G", "rho_refG", "p_ref_G", "p_G_ref"]),
                            ("t3b", A3, ["ref_top", "ref_pay", "G_top", "G_pay"])]:
            if s in A["subs"]:
                SJ.loc[s, f"{nm}_n"] = A["n"][s]
                for k in keys:
                    SJ.loc[s, f"{nm}_{k}"] = A["pts"][k][s]
        for nm, T in [("t4", T4), ("t4e", T4e), ("t4s", T4s)]:
            if s in T["subs"]:
                i = T["subs"].index(s)
                SJ.loc[s, f"{nm}_n"] = T["n"][s]
                for j, (a, b_) in enumerate(T["pairs"]):
                    SJ.loc[s, f"{nm}_{a}~{b_}"] = T["pts"][i, j]
        if s in T4u["subs"]:
            i = T4u["subs"].index(s)
            SJ.loc[s, "t4u_n"] = T4u["n"][s]
            for j, (a, b_) in enumerate(T4u["pairs"]):
                SJ.loc[s, f"t4u_{a}~{b_}"] = T4u["pts"][i, j]
    ctx["SJ"] = SJ
    ctx["rows"], ctx["put"], ctx["putb"] = rows, put, putb
    ctx["elapsed"] = time.time() - t0
    return ctx


# =============================================================================================
# key-number rows (every number in the write-up comes from here or from ctx)
# =============================================================================================
PAIR_LAB = {"ref_gpa~pay": "REF-pay", "ref_gpa~top": "REF-intake", "pay~top": "pay-intake",
            "ref_gpa~score": "REF-intake(band score)", "pay~score": "pay-intake(band score)",
            "ref_gpa~demand": "REF-demand", "pay~demand": "pay-demand", "top~demand": "intake-demand"}


def collect(ctx: dict) -> pd.DataFrame:
    rows = []

    def put(sec, q, spec, est, lo=np.nan, hi=np.nan, se=np.nan, k="", pre="", p=np.nan, ci_kind=""):
        v = ""
        if pre == "pre" and np.isfinite(lo) and np.isfinite(hi):
            v = "detected" if (lo > 0 or hi < 0) else "not detected"
        rows.append(dict(section=sec, quantity=q, spec=spec, est=est, lo=lo, hi=hi, se=se,
                         mde=MDE_K * se if np.isfinite(se) else np.nan, p=p, k=k, status=pre, verdict=v,
                         ci=ci_kind))

    def putS(sec, q, spec, s, k="", pre="", ci_kind="bootstrap"):
        put(sec, q, spec, s["est"], s["lo"], s["hi"], s["se"], k, pre, ci_kind=ci_kind)

    cov, t1, t2, t3 = ctx["cov"], ctx["t1"], ctx["t2"], ctx["t3"]
    # ---- coverage
    sec = "coverage"
    put(sec, "REF 2021 institutions / submissions", "overall profiles", cov["ref_inst"], k=f"{len(ctx['sub'])} submissions")
    put(sec, "REF institutions that are LEO HEIs (UKPRN)", "LEO HEIs minus scripts/62 exclusions", cov["ref_inst_in_leo_hei"])
    put(sec, "share of REF FTE at those institutions", "", cov["ref_fte_in_leo_hei"])
    put(sec, "scripts/62 primary providers with REF", f"of {cov['primary']}", cov["primary_with_ref"])
    put(sec, "share of REF FTE in CLEAN-mapped UoAs", "", cov["ref_fte_clean_uoa"])
    put(sec, "share of REF FTE in CLEAN or EXTENDED UoAs", "", cov["ref_fte_ext_uoa"])
    put(sec, "LEO programmes (HEI x CAH2 with a YAG5 median, cohorts 2013-16)", f"{cov['n_hei']} HEIs", cov["prog"])
    for mp, lab in [("clean", "CLEAN subjects"), ("ext", "EXTENDED subjects"), ("unmapped", "unmapped subjects")]:
        put(sec, f"programmes in {lab}", "count; share of all programmes' graduates in CI column", cov[f"prog_{mp}"],
            lo=cov[f"grads_{mp}"])
        put(sec, f"programmes in {lab} with a REF score", "count; share of all graduates in CI column",
            cov[f"prog_{mp}_ref"], lo=cov[f"grads_{mp}_ref"])
    rv = ctx["rv"]
    tz = ctx["top_zero"]
    put(sec, "share of T4 programmes with intake share 0", "revision 1; T4 cells", tz["share"], k=tz["n"])
    put(sec, "T4 programmes with intake share 0: both >= 360-point bands 'low' (1-2 graduates) in every cohort",
        "revision 1; suppressed, not zero", rv["zero_why"].get("low", 0), k=tz["n"])
    put(sec, "T4 programmes with intake share 0: no released >= 360-point band row", "revision 1",
        rv["zero_why"].get("no rows", 0), k=tz["n"])
    for s, v_ in rv["noscore"].items():
        put(sec, f"{s}: programmes with a matched-UoA score / without", f"revision 1; UoA {'+'.join(map(str, CLEAN[s]))}",
            v_["n_ref"], k=f"{v_['n_none']} without; {v_['n_none_alt']} of them at providers submitting to UoA {'/'.join(map(str, ALT_UOA[s]))}")
    c14 = rv["r14_cov"]
    put(sec, "REF 2014 institutions / submissions", "revision 1; overall profiles", c14["inst"], k=f"{c14['subs']} submissions")
    put(sec, "REF 2014 institutions that are LEO HEIs (UKPRN)", "revision 1", c14["inst_hei"])
    put(sec, "programmes in CLEAN subjects with a REF 2014 score", f"revision 1; of {c14['clean14']:,}", c14["clean14_ref"],
        k=f"{c14['clean_both']} with both REF 2014 and REF 2021")
    # ---- T1
    sec = "T1"
    n = t1["n"]
    putS(sec, "Spearman(G, institution REF GPA)", "scripts/62 primary providers with REF; provider bootstrap", t1["G_gpa"], k=n, pre="pre")
    putS(sec, "Spearman(G, institution REF %4*)", "variant", t1["G_p4"], k=n, pre="pre")
    putS(sec, "Spearman(G, REF FTE submitted)", "exploratory (size)", t1["G_fte"], k=n)
    putS(sec, "Spearman(G, REF research power = GPA x FTE)", "exploratory", t1["G_power"], k=n)
    ns = t1["n_sel"]
    putS(sec, "Spearman(G, institution selectivity)", "exploratory; providers with LEO selectivity", t1["G_sel"], k=ns)
    putS(sec, "Spearman(REF GPA, institution selectivity)", "exploratory", t1["gpa_sel"], k=ns)
    putS(sec, "Spearman(G, REF GPA), same providers", "exploratory", t1["G_gpa_s"], k=ns)
    putS(sec, "Spearman(G, REF GPA) minus Spearman(selectivity, REF GPA)", "exploratory; paired", t1["G_minus_sel_gpa"], k=ns)
    putS(sec, "partial rho(G, REF GPA | selectivity)", "exploratory", t1["pG_gpa_sel"], k=ns)
    putS(sec, "partial rho(selectivity, REF GPA | G)", "exploratory", t1["psel_gpa_G"], k=ns)
    put(sec, "AUC(Russell Group vs others): REF GPA", f"{t1['n_rg']} RG members", t1["auc_gpa"], k=n)
    put(sec, "AUC(Russell Group vs others): G", "same providers", t1["auc_G"], k=n)
    # ---- T2(a)
    sec = "T2a"
    lab = {"main": ("within-provider slope of log pay per SD of REF GPA rank", "CLEAN; provider FE + subject FE", "pre"),
           "ext": ("  sensitivity: EXTENDED crosswalk", "CLEAN + EXTENDED subjects", "sens"),
           "raw": ("  sensitivity: raw GPA (z) instead of rank", "CLEAN", "sens"),
           "p4": ("  sensitivity: %4* rank instead of GPA rank", "CLEAN", "sens"),
           "trait_base": ("  primary providers, same spec", "CLEAN; scripts/62 primary providers", "sens"),
           "trait": ("  + subject x {zG, zS} slopes (US main-spec analogue)", "same cells as the row above", "sens"),
           "intake_base": ("  intake sample, same spec", "CLEAN; programmes with LEO intake", "sens"),
           "intake": ("  + subject x programme intake (S-LEO) slopes", "same cells as the row above", "sens")}
    for key, (q, spec, st) in lab.items():
        r = t2[key]
        kk = f"{r['N']} programmes, {r['G']} providers, {r['S']} subjects"
        put(sec, q, spec + " [CR1 CI]", float(r["beta"][0]), float(r["lo"][0]), float(r["hi"][0]), float(r["se"][0]),
            kk, st, p=float(r["p"][0]), ci_kind="CR1")
        if "bs" in r:
            put(sec, q + " (cluster bootstrap)", spec, float(r["beta"][0]), float(r["blo"][0]), float(r["bhi"][0]),
                float(r["bse"][0]), kk, st, ci_kind="bootstrap")
    putS(sec, "  intake control: slope without minus with", "same cells; paired cluster bootstrap (same draws)",
         t2["intake_drop"], k="", pre="sens")
    s_ = t2["intake_share"]
    put(sec, "  intake control: share of the slope removed", "1 - with / without; paired cluster bootstrap (percentile CI; ratio, no SE)",
        s_["est"], s_["lo"], s_["hi"], np.nan, "", "sens", ci_kind="bootstrap")
    us = t2["us"]
    for key, q in [("a", "US (scripts/58): institution FE + field FE"), ("main", "US (scripts/58): main spec")]:
        u = us[key]
        put(sec, q, "read from SELECTIVITY_DEEP_RESULT.md; institution cluster bootstrap CI", u["est"], u["lo"], u["hi"],
            u["se"], f"{u['n']} programmes, {u['inst']} institutions, {u['fields']} fields", ci_kind="bootstrap")
    m = t2["main"]
    dUS = float(m["beta"][0] - us["a"]["est"])
    seUS = float(np.sqrt(m["se"][0] ** 2 + us["a"]["se"] ** 2))
    put(sec, "UK main minus US (a)", "exploratory; independent samples, normal approximation with CR1 SEs", dUS,
        dUS - 1.959963984540054 * seUS, dUS + 1.959963984540054 * seUS, seUS, ci_kind="normal")
    # revision 1: leave one subject out; Economics fallback
    loo = rv["loo"]
    smin, smax = min(loo, key=lambda s: loo[s]["beta"]), max(loo, key=lambda s: loo[s]["beta"])
    for s, q in [(smin, "leave-one-subject-out: lowest slope"), (smax, "leave-one-subject-out: highest slope")]:
        put(sec, "  " + q, "revision 1; CLEAN minus one subject [CR1 CI]", loo[s]["beta"], loo[s]["lo"], loo[s]["hi"],
            k=f"without {s}", ci_kind="CR1")
    r = rv["e17_t2a"]
    kk = f"{r['N']} programmes, {r['G']} providers, {r['S']} subjects"
    put(sec, "  Economics scored by UoA 16, else UoA 17", "revision 1; CLEAN [CR1 CI]", float(r["beta"][0]), float(r["lo"][0]),
        float(r["hi"][0]), float(r["se"][0]), kk, ci_kind="CR1")
    put(sec, "  Economics scored by UoA 16, else UoA 17 (cluster bootstrap)", "revision 1; CLEAN", float(r["beta"][0]),
        float(r["blo"][0]), float(r["bhi"][0]), float(r["bse"][0]), kk, ci_kind="bootstrap")
    # revision 1: intake control with 'low' band cells coded 1 and 2
    for f in (1, 2):
        L_ = rv["low"][f]
        for key, q in [("t2_without", "slope without intake control"), ("t2_with", "slope with subject x intake slopes")]:
            r = L_[key]
            kk = f"{r['N']} programmes, {r['G']} providers, {r['S']} subjects"
            put("T2a-low", f"'low' = {f}: {q}", "revision 1; CLEAN; programmes with LEO intake [CR1 CI]", float(r["beta"][0]),
                float(r["lo"][0]), float(r["hi"][0]), float(r["se"][0]), kk, ci_kind="CR1")
        putS("T2a-low", f"'low' = {f}: intake control: slope without minus with", "revision 1; paired cluster bootstrap",
             L_["t2_drop"])
        s_ = L_["t2_share"]
        put("T2a-low", f"'low' = {f}: intake control: share of the slope removed", "revision 1; paired cluster bootstrap (ratio, no SE)",
            s_["est"], s_["lo"], s_["hi"], np.nan, "", ci_kind="bootstrap")
    # ---- T2(b)
    sec = "T2b"
    A2 = ctx["A2"]
    K2 = len(A2["subs"])
    for key, q, st in [("rho_ref", "mean rho(REF GPA, pay) across providers", "pre"),
                       ("rho_G", "mean rho(G, pay), same providers", "pre"),
                       ("p_ref_G", "mean partial rho(REF GPA, pay | G)", "pre"),
                       ("p_G_ref", "mean partial rho(G, pay | REF GPA)", "pre"),
                       ("rho_refG", "mean rho(REF GPA, G) (descriptive)", "")]:
        putS(sec, q, "CLEAN subjects; scripts/62 primary providers with REF; two-stage bootstrap",
             summ(A2["mean"][key], A2["ts"][key]), k=K2, pre=st)
    putS(sec, "rho(REF GPA, pay) minus rho(G, pay)", "paired", summ(A2["mean"]["rho_ref"] - A2["mean"]["rho_G"],
                                                               A2["ts"]["rho_ref"] - A2["ts"]["rho_G"]), k=K2, pre="pre")
    # ---- T3(a)
    sec = "T3a"
    r = t3["main"]
    kk = f"{r['N']} programmes, {r['G']} providers, {r['S']} subjects"
    for i, (q, st) in enumerate([("within-provider slope of pay (within-subject SD units) per SD of REF rank", "pre"),
                                 ("within-provider slope of intake (within-subject SD units) per SD of REF rank", "pre"),
                                 ("within-provider slope of log pay per SD of REF rank (same cells)", ""),
                                 ("within-provider slope of intake share >= 360 points per SD of REF rank", "")]):
        put(sec, q, "CLEAN; provider FE + subject FE [CR1]", float(r["beta"][i]), float(r["lo"][i]), float(r["hi"][i]),
            float(r["se"][i]), kk, st, p=float(r["p"][i]), ci_kind="CR1")
        put(sec, q + " (cluster bootstrap)", "", float(r["beta"][i]), float(r["blo"][i]), float(r["bhi"][i]),
            float(r["bse"][i]), kk, st, ci_kind="bootstrap")
    put(sec, "intake minus pay (SD units)", "CR1 of the stacked system", r["diff"], r["diff_lo"], r["diff_hi"],
        r["diff_se"], kk, "pre", ci_kind="CR1")
    put(sec, "intake minus pay (SD units) (paired cluster bootstrap)", "", r["diff"], r["diff_blo"], r["diff_bhi"],
        r["diff_bse"], kk, "pre", ci_kind="bootstrap")
    r2 = t3["score"]
    put(sec, "  variant: mean band score (SD units) per SD of REF rank", "same cells [CR1]", float(r2["beta"][1]),
        float(r2["lo"][1]), float(r2["hi"][1]), float(r2["se"][1]), kk, "sens", ci_kind="CR1")
    put(sec, "  variant: band score minus pay (SD units) (paired cluster bootstrap)", "", r2["diff"], r2["diff_blo"],
        r2["diff_bhi"], r2["diff_bse"], kk, "sens", ci_kind="bootstrap")
    w = t3["wtriad"]
    for j, q in enumerate(["within-provider corr(REF rank, pay)", "within-provider corr(REF rank, intake)",
                           "within-provider corr(pay, intake)"]):
        bs = w["bs"][:, j]
        put(sec, q, "exploratory; provider + subject FE residuals, Pearson; cluster bootstrap", float(w["pt"][j]),
            *ci(bs), float(np.nanstd(bs)), f"{w['n']} programmes, {w['G']} providers", ci_kind="bootstrap")
    # revision 1: scale-free and floor-free versions of the T3(a) difference; 'low' codings
    putS(sec, "within-provider corr(REF rank, intake) minus corr(REF rank, pay)",
         "revision 1; scale-free; paired cluster bootstrap (same draws as the rows above)",
         summ(float(w["pt"][1] - w["pt"][0]), w["bs"][:, 1] - w["bs"][:, 0]), k=f"{w['n']} programmes, {w['G']} providers")
    rn = rv["t3_nofloor"]
    kkn = f"{rn['N']} programmes, {rn['G']} providers, {rn['S']} subjects"
    for i, q in [(0, "without floor subjects: pay (SD units) per SD of REF rank"),
                 (1, "without floor subjects: intake (SD units) per SD of REF rank")]:
        put(sec, q, "revision 1; subjects whose intake-share SD is below a third of the median subject's dropped; cluster bootstrap", float(rn["beta"][i]),
            float(rn["blo"][i]), float(rn["bhi"][i]), float(rn["bse"][i]), kkn, ci_kind="bootstrap")
    put(sec, "without floor subjects: intake minus pay (SD units)", "revision 1; paired cluster bootstrap", rn["diff"],
        rn["diff_blo"], rn["diff_bhi"], rn["diff_bse"], kkn, ci_kind="bootstrap")
    r3 = t3["margin"]
    kk3 = f"{r3['N']} programmes, {r3['G']} providers, {r3['S']} subjects; {r3['df']['sub'].mean():.3f} submitted"
    # (the within-subject SD-unit rows of the first version are dropped in revision 1: the intake SD is
    # dominated by the floor subjects, see the T3a floor rows)
    for i, q in [(2, "REF-submitted minus not: log pay (within provider)"),
                 (3, "REF-submitted minus not: intake share (within provider)")]:
        put(sec, q, "exploratory; CLEAN subjects at REF-submitting institutions; cluster bootstrap",
            float(r3["beta"][i]), float(r3["blo"][i]), float(r3["bhi"][i]), float(r3["bse"][i]), kk3, ci_kind="bootstrap")
    for nm, lab in [("excl", "without Economics, Biosciences, Architecture"),
                    ("alt", "submission to the matched or an alternative UoA counts")]:
        rm = rv["margin"][nm]
        kkm = f"{rm['N']} programmes, {rm['G']} providers, {rm['S']} subjects; {rm['share']:.3f} submitted"
        for i, q in [(0, "log pay"), (1, "intake share")]:
            put(sec, f"REF-submitted minus not: {q} ({lab})", "revision 1; cluster bootstrap", float(rm["beta"][i]),
                float(rm["blo"][i]), float(rm["bhi"][i]), float(rm["bse"][i]), kkm, ci_kind="bootstrap")
    for f in (1, 2):
        rl = rv["low"][f]["t3"]
        kkl = f"{rl['N']} programmes, {rl['G']} providers, {rl['S']} subjects"
        put("T3a-low", f"'low' = {f}: intake (SD units) per SD of REF rank", "revision 1; CR1", float(rl["beta"][1]),
            float(rl["lo"][1]), float(rl["hi"][1]), float(rl["se"][1]), kkl, ci_kind="CR1")
        put("T3a-low", f"'low' = {f}: intake minus pay (SD units)", "revision 1; paired cluster bootstrap", rl["diff"],
            rl["diff_blo"], rl["diff_bhi"], rl["diff_bse"], kkl, ci_kind="bootstrap")
    # ---- T3(b)
    sec = "T3b"
    A3 = ctx["A3"]
    K3 = len(A3["subs"])
    for key, q, st in [("ref_top", "mean rho(REF GPA, intake)", "pre"), ("ref_pay", "mean rho(REF GPA, pay)", "pre"),
                       ("G_top", "mean rho(G, intake)", "pre"), ("G_pay", "mean rho(G, pay)", "pre"),
                       ("sel_top", "mean rho(institution selectivity, intake)", ""),
                       ("sel_pay", "mean rho(institution selectivity, pay)", ""),
                       ("p_ref_top_sel", "mean partial rho(REF GPA, intake | institution selectivity)", ""),
                       ("p_ref_pay_sel", "mean partial rho(REF GPA, pay | institution selectivity)", "")]:
        putS(sec, q, "CLEAN; primary providers with REF, G, intake; two-stage bootstrap",
             summ(A3["mean"][key], A3["ts"][key]), k=K3, pre=st)
    for a, b_, q in [("ref_top", "ref_pay", "REF: rho(intake) minus rho(pay)"), ("G_top", "G_pay", "G: rho(intake) minus rho(pay)")]:
        putS(sec, q, "paired", summ(A3["mean"][a] - A3["mean"][b_], A3["ts"][a] - A3["ts"][b_]), k=K3, pre="pre")
    A3u = ctx["A3u"]
    K3u = len(A3u["subs"])
    for key, q in [("ref_dem", "mean rho(REF GPA, demand)"), ("ref_pay", "mean rho(REF GPA, pay)"),
                   ("G_dem", "mean rho(G, demand)"), ("G_pay", "mean rho(G, pay)")]:
        putS("T3b-UCAS", q, "UCAS subjects; primary providers with REF, G, demand; two-stage bootstrap",
             summ(A3u["mean"][key], A3u["ts"][key]), k=K3u, pre="pre")
    for a, b_, q in [("ref_dem", "ref_pay", "REF: rho(demand) minus rho(pay)"), ("G_dem", "G_pay", "G: rho(demand) minus rho(pay)")]:
        putS("T3b-UCAS", q, "paired", summ(A3u["mean"][a] - A3u["mean"][b_], A3u["ts"][a] - A3u["ts"][b_]), k=K3u, pre="pre")
    # ---- T4
    # T4-score: the band score is the pre-specified S-LEO variant, but it was added to T4 after the first run
    # showed the floor of the share measure, so its rows are 'post hoc' (revision 1; first version: 'sens')
    lab_x = {"ref_gpa_e17": "REF", "ref14_gpa": "REF", "top_l1": "intake", "top_l2": "intake", "ref_gpa": "REF",
             "pay": "pay", "top": "intake"}                      # the section and the row prefix name the variant
    tri = [("T4", ctx["T4"], "pre", "", None), ("T4-ext", ctx["T4e"], "sens", "", None),
           ("T4-score", ctx["T4s"], "post hoc", "", None), ("T4-UCAS", ctx["T4u"], "pre", "", None)]
    tri += [("T4-low", rv["low"][f]["T4"], "", f"'low' = {f}: ", f) for f in (1, 2)]
    tri += [("T4-econ", rv["e17_t4"], "", "Economics UoA 16 else 17: ", None), ("REF2014", rv["r14_t4"], "", "REF 2014: ", None)]
    for sec, T, st, pfx, f in tri:
        K = len(T["subs"])
        keys = [f"{a}~{b_}" for a, b_ in T["pairs"]]
        labs_ = [PAIR_LAB[k_] if not pfx else f"{lab_x[a]}-{lab_x[b_]}" for k_, (a, b_) in zip(keys, T["pairs"])]
        mean = T["pts"].mean(0)
        for j, k_ in enumerate(keys):
            putS(sec, f"{pfx}mean rho {labs_[j]}", ("revision 1; " if pfx else "") + "subject bootstrap",
                 summ(mean[j], T["bs"][:, j]), k=K, pre=st)
            if T["A"] is not None:
                putS(sec, f"mean rho {labs_[j]} (two-stage)", "subjects + providers", summ(mean[j], T["A"]["ts"][k_]), k=K, pre=st)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                qd = f"{pfx}{labs_[i]} minus {labs_[j]}"
                putS(sec, qd, ("revision 1; " if pfx else "") + "paired subject bootstrap",
                     summ(mean[i] - mean[j], T["bs"][:, i] - T["bs"][:, j]), k=K, pre=st)
                if T["A"] is not None:
                    # revision 1: the pre-specified two-stage CI, alongside, also for the differences
                    putS(sec, qd + " (two-stage)", "paired two-stage bootstrap (subjects + providers; one outer draw)",
                         summ(mean[i] - mean[j], T["A"]["ts"][keys[i]] - T["A"]["ts"][keys[j]]), k=K, pre=st)
        if f is not None:
            put(sec, f"{pfx}share of T4 programmes with intake share 0", "revision 1", rv["low"][f]["zero"], k=rv["low"][f]["n4"])
        if pfx:
            continue
        top = np.bincount(np.argmax(T["bs"], 1), minlength=len(keys)) / len(T["bs"])
        for j, k_ in enumerate(keys):
            put(sec, f"share of bootstrap replicates in which {PAIR_LAB[k_]} agrees most", "subject bootstrap",
                float(top[j]), k=K)
        if T["A"] is not None:
            TS = np.column_stack([T["A"]["ts"][k_] for k_ in keys])
            top2 = np.bincount(np.argmax(TS, 1), minlength=len(keys)) / len(TS)
            for j, k_ in enumerate(keys):
                put(sec, f"share of two-stage replicates in which {PAIR_LAB[k_]} agrees most", "revision 1; two-stage bootstrap",
                    float(top2[j]), k=K)
    # REF 2014 slopes and the reliability of the within-provider REF axis
    sec = "REF2014"
    r = rv["r14_t2a"]
    kk = f"{r['N']} programmes, {r['G']} providers, {r['S']} subjects"
    put(sec, "within-provider slope of log pay per SD of REF 2014 GPA rank", "revision 1; CLEAN14 [CR1 CI]", float(r["beta"][0]),
        float(r["lo"][0]), float(r["hi"][0]), float(r["se"][0]), kk, ci_kind="CR1")
    put(sec, "within-provider slope of log pay per SD of REF 2014 GPA rank (cluster bootstrap)", "revision 1; CLEAN14",
        float(r["beta"][0]), float(r["blo"][0]), float(r["bhi"][0]), float(r["bse"][0]), kk, ci_kind="bootstrap")
    rl_ = rv["rel"]
    kk = f"{rl_['N']} programmes, {rl_['G']} providers, {rl_['S']} subjects"
    for j, q in enumerate(["within-provider corr of REF 2021 and REF 2014 ranks (test-retest reliability)",
                           "slope of log pay per SD of REF 2021 rank, same cells",
                           "slope of log pay per SD of REF 2014 rank, same cells",
                           "REF 2021 slope / reliability (disattenuated)",
                           "IV slope: REF 2021 rank instrumented by REF 2014 rank"]):
        putS(sec, q, "revision 1; provider + subject FE; cells with both scores; cluster bootstrap (one draw for all rows)",
             summ(float(rl_["pt"][j]), rl_["bs"][:, j]), k=kk)
    putS(sec, "slope per SD: REF 2014 rank minus REF 2021 rank, same cells", "revision 1; paired cluster bootstrap",
         summ(float(rl_["pt"][2] - rl_["pt"][1]), rl_["bs"][:, 2] - rl_["bs"][:, 1]), k=kk)
    put(sec, "mean across-provider Spearman(REF 2021, REF 2014) within subject", "revision 1; same cells; descriptive",
        rv["rel_across"], k=f"{rl_['S']} subjects")
    return pd.DataFrame(rows)


# =============================================================================================
# write-up
# =============================================================================================
def f3(x, d=3):
    return f"{x:+.{d}f}"


def fci(lo, hi, d=3):
    return f"[{lo:+.{d}f}, {hi:+.{d}f}]"


def write_md(ctx: dict, R: pd.DataFrame) -> str:
    def row(sec, q):
        r = R[(R.section == sec) & (R.quantity == q)]
        assert len(r) == 1, (sec, q, len(r))
        return r.iloc[0]

    def ec(sec, q, d=3):
        r = row(sec, q)
        return f"{f3(r.est, d)} {fci(r.lo, r.hi, d)}"

    def eci(sec, q, d=3):
        r = row(sec, q)
        return fci(r.lo, r.hi, d)

    def pct(x, d=1):
        return f"{100 * x:+.{d}f}%"

    def excl0(r):
        return bool(r.lo > 0 or r.hi < 0)

    def pc(x):
        return f"{100 * x:.0f}%"

    cov, t1, t2, t3, rv = ctx["cov"], ctx["t1"], ctx["t2"], ctx["t3"], ctx["rv"]
    us = t2["us"]
    m, mt, mi, mib = t2["main"], t2["trait"], t2["intake"], t2["intake_base"]
    b0, se0 = float(m["beta"][0]), float(m["se"][0])
    r3 = t3["main"]
    T4, T4u, T4e, T4s = ctx["T4"], ctx["T4u"], ctx["T4e"], ctx["T4s"]
    k4 = [f"{a}~{b_}" for a, b_ in T4["pairs"]]
    labs = [PAIR_LAB[k] for k in k4]
    mean4 = T4["pts"].mean(0)
    order = np.argsort(-mean4)
    top_share = np.bincount(np.argmax(T4["bs"], 1), minlength=3) / B
    A2, A3 = ctx["A2"], ctx["A3"]
    v = lambda sec, q: row(sec, q).verdict  # noqa: E731
    wt = t3["wtriad"]
    # ---- claims the text makes, checked against the numbers (the script stops if one fails)
    assert labs == ["REF-pay", "REF-intake", "pay-intake"] and labs[order[-1]] == "REF-pay"
    d1, d2, d3 = (row("T4", q) for q in ["REF-pay minus REF-intake", "REF-pay minus pay-intake", "REF-intake minus pay-intake"])
    d1t, d2t, d3t = (row("T4", q + " (two-stage)") for q in ["REF-pay minus REF-intake", "REF-pay minus pay-intake",
                                                             "REF-intake minus pay-intake"])
    sb_both, ts_none = excl0(d1) and excl0(d2), not excl0(d1t) and not excl0(d2t)
    assert sb_both and ts_none and not excl0(d3) and not excl0(d3t)
    dUS = row("T2a", "UK main minus US (a)")
    assert dUS.lo < 0 < dUS.hi and dUS.hi > us["a"]["est"] and dUS.mde > max(b0, us["a"]["est"])
    dsd = row("T3a", "intake minus pay (SD units) (paired cluster bootstrap)")
    bpay = row("T3a", "within-provider slope of pay (within-subject SD units) per SD of REF rank").est
    rlo, rhi = (bpay + dsd.lo) / bpay, (bpay + dsd.hi) / bpay
    assert dsd.lo < 0 < dsd.hi
    low = {f: {q: row("T4-low", f"'low' = {f}: {q}") for q in
               ["mean rho REF-pay", "mean rho REF-intake", "mean rho pay-intake", "REF-intake minus pay-intake",
                "REF-pay minus REF-intake", "REF-pay minus pay-intake", "share of T4 programmes with intake share 0"]}
           for f in (1, 2)}
    for f in (1, 2):   # REF-pay stays lowest; pay-intake's lead over REF-intake shrinks and is not detected
        assert low[f]["mean rho REF-pay"].est < min(low[f]["mean rho REF-intake"].est, low[f]["mean rho pay-intake"].est)
        assert -low[f]["REF-intake minus pay-intake"].est < -d3.est and not excl0(low[f]["REF-intake minus pay-intake"])
    s1, s2 = (row("T2a-low", f"'low' = {f}: intake control: share of the slope removed") for f in (1, 2))
    loo = rv["loo"]
    smin, smax = min(loo, key=lambda s: loo[s]["beta"]), max(loo, key=lambda s: loo[s]["beta"])
    assert loo[smin]["lo"] > 0
    loo_order = sum(int(np.argmin(v_["t4"]) == 0 and np.argmax(v_["t4"]) == 2) for v_ in loo.values())
    assert loo_order == len(loo) and all(v_["lo"] > 0 for v_ in loo.values())
    assert np.argmin(rv["e17_t4"]["pts"].mean(0)) == 0
    fl = rv["floor"]
    rn = row("T3a", "without floor subjects: intake minus pay (SD units)")
    sfree = row("T3a", "within-provider corr(REF rank, intake) minus corr(REF rank, pay)")
    assert not excl0(rn) and not excl0(sfree)
    r14 = {q: row("REF2014", q) for q in ["within-provider slope of log pay per SD of REF 2014 GPA rank",
                                           "within-provider corr of REF 2021 and REF 2014 ranks (test-retest reliability)",
                                           "slope of log pay per SD of REF 2021 rank, same cells",
                                           "slope of log pay per SD of REF 2014 rank, same cells",
                                           "REF 2021 slope / reliability (disattenuated)",
                                           "IV slope: REF 2021 rank instrumented by REF 2014 rank",
                                           "slope per SD: REF 2014 rank minus REF 2021 rank, same cells",
                                           "mean across-provider Spearman(REF 2021, REF 2014) within subject"]}
    rel = rv["rel"]
    assert excl0(r14["slope per SD: REF 2014 rank minus REF 2021 rank, same cells"])
    assert r14["within-provider slope of log pay per SD of REF 2014 GPA rank"].est > b0
    t14 = rv["r14_t4"]
    assert np.argmin(t14["pts"].mean(0)) == 0 and np.argmax(t14["pts"].mean(0)) == 2
    ns = rv["noscore"]
    tz = ctx["top_zero"]
    zw = rv["zero_why"]
    ud = R[(R.section == "T4-UCAS") & R.quantity.str.contains(" minus ")]
    ud = ud[ud.quantity.apply(lambda q: sum("demand" in s for s in q.replace(" (two-stage)", "").split(" minus ")) == 1)]
    ud_sb, ud_ts = ud[~ud.quantity.str.endswith("(two-stage)")], ud[ud.quantity.str.endswith("(two-stage)")]
    ud_det, ud_det_ts = int((ud_sb.verdict == "detected").sum()), int((ud_ts.verdict == "detected").sum())
    ud_pos = int((ud_sb.apply(lambda r: (r.est > 0) == ("demand" in r.quantity.split(" minus ")[1]), axis=1)).sum())
    wr = [row("T3a", q).est for q in ["within-provider corr(REF rank, pay)", "within-provider corr(REF rank, intake)",
                                       "within-provider corr(pay, intake)"]]
    ctx["range_across"] = (float(mean4.min()), float(mean4.max()))
    ctx["range_within"] = (float(min(wr)), float(max(wr)))
    frozen_t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(FROZEN.stat().st_mtime))
    L = []
    L.append("# UK triad — do research standing (REF), graduates' pay and student intake rank UK university × subject programmes alike, and at which level?\n")
    L.append(f"Script: `scripts/76_uk_triad.py` (seed {SEED}; every number below is written by the script from its own computations; re-run byte-identical, see Method). Public data only: REF 2021 results by institution × unit of assessment (downloaded 2026-09-27, `data/raw/SOURCES.md` §10i), REF 2014 results (revision 1, exploratory; §10i-b), DfE LEO provider × CAH2 graduate outcomes and the ORCID-derived UK hiring rank G (both as `scripts/62`, imported, not edited), and the UCAS provider × subject-group files already on disk. Descriptive, not causal. Three rankings of the same programmes: *academic* = the department's REF 2021 grade-point average (GPA) in the unit of assessment (UoA) matched to the subject; *employer* = LEO median earnings five years after graduation (YAG5, cohorts 2013/14–2016/17, as scripts/62); *students* = the programme's intake attainment (share of its graduates who entered with ≥ 360 UCAS tariff points, from the same LEO file) and, on {len(UCAS_CAH1)} subjects, UCAS applications per acceptance. Tables: `data/interim/uk_triad_summary.csv` (all key numbers), `uk_triad_subjects.csv`, `uk_triad_panel.csv`, `uk_triad_crosswalk.csv`. This is revision 1; what changed after the independent verification is listed under Revisions.\n")
    L.append("## Answer\n")
    L.append(f"**Key question: is the UK hiring rank G externally valid, does a department's own research standing go with its graduates' pay within a university, and do students' choices follow research standing more than pay does?** G is externally valid. Within a university, higher-REF subjects pay somewhat more ({pct(b0)} per SD of REF rank); the comparison with the US estimate is too imprecise to say whether the UK slope is smaller or larger. With the programme's own intake held fixed that slope is {pc(t2['intake_share']['est'])} smaller (point estimate; {pc(s1.est)} and {pc(s2.est)} when suppressed intake cells are coded as 1 or 2 graduates). The data cannot tell apart how strongly intake and pay track a department's research standing within a university: relative to the pay slope, the interval allows an intake slope of {rlo:.1f} to {rhi:.1f} times it. Across providers within a subject the three rankings agree strongly, and REF–pay has the lowest point estimate; it is below both other pairs under the pre-specified subject bootstrap, but not once provider-level sampling error is included (two-stage bootstrap), so that ordering is marginal. Within a university the three agree weakly. REF 2014, which assessed research while these cohorts studied, gives a larger within-university pay slope ({pct(r14['within-provider slope of log pay per SD of REF 2014 GPA rank'].est)} per SD), and the within-university REF axis has a test–retest correlation of only {r14['within-provider corr of REF 2021 and REF 2014 ranks (test-retest reliability)'].est:.2f} between the two exercises (exploratory).\n")
    # 1 T1
    L.append(f"1. **T1 — G passes an external check: the hiring rank and REF 2021 order UK universities alike.** Across the {t1['n']} scripts/62 primary providers with a REF submission, Spearman(G, institution REF GPA) = {ec('T1', 'Spearman(G, institution REF GPA)')} (provider bootstrap; MDE {row('T1', 'Spearman(G, institution REF GPA)').mde:.3f}); with the 4* share instead of the GPA {ec('T1', 'Spearman(G, institution REF %4*)')}. Both are {v('T1', 'Spearman(G, institution REF GPA)')} (pre-specified). The Russell Group separates from the rest on REF GPA with AUC {t1['auc_gpa']:.3f} and on G with AUC {t1['auc_G']:.3f} (same providers). Exploratory: G and intake selectivity track REF GPA to a degree the data cannot tell apart (Spearman with REF GPA {ec('T1', 'Spearman(G, REF GPA), same providers')} for G vs {ec('T1', 'Spearman(REF GPA, institution selectivity)')} for the institution's share of graduates with ≥ 360 points; difference {ec('T1', 'Spearman(G, REF GPA) minus Spearman(selectivity, REF GPA)')}, MDE {row('T1', 'Spearman(G, REF GPA) minus Spearman(selectivity, REF GPA)').mde:.3f}), and G keeps partial ρ = {ec('T1', 'partial rho(G, REF GPA | selectivity)')} with REF GPA net of selectivity, so G is not a relabelled intake measure. G also correlates with the size of the REF submission (Spearman with submitted FTE {ec('T1', 'Spearman(G, REF FTE submitted)')}).\n")
    # 2 T2
    L.append(f"2. **T2 — Department level: within a university, a subject one SD higher in its REF rank has {pct(b0)} higher pay; how this compares with the US cannot be told; much of it goes with the programme's intake.** Provider fixed effects + subject fixed effects, {m['S']} clean subjects: β = {ec('T2a', 'within-provider slope of log pay per SD of REF GPA rank', 4)} log points per within-subject SD of the REF GPA rank (CR1, provider-clustered; cluster bootstrap {fci(float(m['blo'][0]), float(m['bhi'][0]), 4)}; {m['N']} programmes, {m['G']} providers; MDE {MDE_K * se0:.4f}); {v('T2a', 'within-provider slope of log pay per SD of REF GPA rank')} (pre-specified). The US analogue (scripts/58, institution FE + field FE, department hiring rank F) is {f3(us['a']['est'], 4)} {fci(us['a']['lo'], us['a']['hi'], 4)}; the difference, UK minus US, is {f3(dUS.est, 4)} {fci(dUS.lo, dUS.hi, 4)} (exploratory, normal approximation; MDE {dUS.mde:.4f}, larger than either slope). The two cannot be told apart, and the interval also includes {f3(us['a']['est'], 4)}, the difference at which the UK slope would be twice the US estimate. Adding subject-specific slopes on G and institution selectivity, the analogue of the US main specification, gives {ec('T2a', '  + subject x {zG, zS} slopes (US main-spec analogue)', 4)} against the US {f3(us['main']['est'], 4)} {fci(us['main']['lo'], us['main']['hi'], 4)} (the paper's {pct(us['main']['est'], 2)} per SD; the UK point estimate is {float(mt['beta'][0]) / us['main']['est']:.1f} times the US one, a difference not tested). Sensitivities: extended crosswalk ({t2['ext']['S']} subjects) {ec('T2a', '  sensitivity: EXTENDED crosswalk', 4)}; raw GPA instead of rank {ec('T2a', '  sensitivity: raw GPA (z) instead of rank', 4)}; 4* share {ec('T2a', '  sensitivity: %4* rank instead of GPA rank', 4)}. Exploratory (revision 1): leaving out one subject at a time, the slope ranges from {f3(loo[smin]['beta'], 4)} (without {smin}; CR1 {fci(loo[smin]['lo'], loo[smin]['hi'], 4)}) to {f3(loo[smax]['beta'], 4)} (without {smax}); with Economics scored by UoA 16, else by the provider's UoA 17 (Business and management) submission, {ec('T2a', '  Economics scored by UoA 16, else UoA 17', 4)}.\n\n"
             f"   The UK data add what the US data lack: the programme's own entry attainment. With subject-specific slopes on the programme's intake the slope falls from {f3(float(mib['beta'][0]), 4)} to {ec('T2a', '  + subject x programme intake (S-LEO) slopes', 4)} on the same cells (CR1; the CI {'includes' if row('T2a', '  + subject x programme intake (S-LEO) slopes').lo < 0 < row('T2a', '  + subject x programme intake (S-LEO) slopes').hi else 'excludes'} zero). The drop is {ec('T2a', '  intake control: slope without minus with', 4)}, {pc(t2['intake_share']['est'])} of the slope (95% CI {pc(t2['intake_share']['lo'])} to {pc(t2['intake_share']['hi'])}; paired cluster bootstrap). The share depends on how suppressed intake cells are coded (exploratory, same cells; see answer 4): with 'low' band cells (1–2 graduates) coded as 1 it is {pc(s1.est)} ({pc(s1.lo)} to {pc(s1.hi)}), as 2 {pc(s2.est)} ({pc(s2.lo)} to {pc(s2.hi)}). So within a university the subjects with stronger research departments also have graduates with higher prior attainment, and in point estimates more than half of the pay gradient across a university's subjects goes with that. US selectivity is observed only for the university, so this within-university sorting is not controlled in the US estimate (the programme's Pell share is, and it did not lower the US slope; paper).\n\n"
             f"   REF 2014 (exploratory, revision 1; assessment period 2008–2013, while these cohorts studied; UoAs mapped in Method) gives a larger within-university slope: {ec('REF2014', 'within-provider slope of log pay per SD of REF 2014 GPA rank', 4)} per SD of the REF 2014 rank (CR1; {row('REF2014', 'within-provider slope of log pay per SD of REF 2014 GPA rank').k}). On the {rel['N']} programmes scored in both exercises, the REF 2014 slope exceeds the REF 2021 slope by {ec('REF2014', 'slope per SD: REF 2014 rank minus REF 2021 rank, same cells', 4)} (paired cluster bootstrap), and the within-provider correlation of the two ranks (provider and subject fixed effects removed) is {ec('REF2014', 'within-provider corr of REF 2021 and REF 2014 ranks (test-retest reliability)')}, far below their across-provider agreement within subjects (mean Spearman {r14['mean across-provider Spearman(REF 2021, REF 2014) within subject'].est:.3f}). A department's standing relative to the rest of its university is therefore either measured with much noise or changed between the exercises. Read as a test–retest reliability, it gives a disattenuated REF 2021 slope of {ec('REF2014', 'REF 2021 slope / reliability (disattenuated)', 4)} (β/r, from {f3(r14['slope of log pay per SD of REF 2021 rank, same cells'].est, 4)} on these cells), and the errors-in-variables IV estimate (REF 2021 rank instrumented by the REF 2014 rank) is {ec('REF2014', 'IV slope: REF 2021 rank instrumented by REF 2014 rank', 4)}; the two differ because REF 2014 goes with pay more strongly than REF 2021 does. Both corrections assume classical measurement error and a true standing that did not change between 2014 and 2021; where it did change, r understates the reliability and the corrections overstate the slope. The US slope is not corrected in the same way, so the corrected figures are not comparable with it.\n\n"
             f"   Across providers within a subject (T2b, {len(A2['subs'])} subjects, scripts/62 primary providers), the department's REF GPA and the university's G couple with pay to a degree the data cannot tell apart: mean ρ(REF, pay) = {ec('T2b', 'mean rho(REF GPA, pay) across providers')}, ρ(G, pay) = {ec('T2b', 'mean rho(G, pay), same providers')}, difference {ec('T2b', 'rho(REF GPA, pay) minus rho(G, pay)')} ({v('T2b', 'rho(REF GPA, pay) minus rho(G, pay)')}; MDE {row('T2b', 'rho(REF GPA, pay) minus rho(G, pay)').mde:.3f}). Each keeps a partial correlation beyond the other (REF | G {ec('T2b', 'mean partial rho(REF GPA, pay | G)')}; G | REF {ec('T2b', 'mean partial rho(G, pay | REF GPA)')}); within a subject the two correlate {ec('T2b', 'mean rho(REF GPA, G) (descriptive)')}. Unlike the field-tagged ORCID axis of scripts/40, REF gives a department-level academic measure with a clear across-provider coupling with pay.\n")
    # 3 T3
    fl_txt = " and ".join(f"{s} ({fl[s]['sd']:.3f}; {pc(fl[s]['zero'])} of programmes at 0)" for s in fl)
    L.append(f"3. **T3 — Students: the data cannot tell apart how strongly intake and pay track a department's research standing.** Within a university (same regression, both outcomes in within-subject SD units, {r3['N']} programmes): pay {ec('T3a', 'within-provider slope of pay (within-subject SD units) per SD of REF rank')} SD and intake {ec('T3a', 'within-provider slope of intake (within-subject SD units) per SD of REF rank')} SD per SD of REF rank (CR1); difference intake − pay {f3(dsd.est)} {fci(dsd.lo, dsd.hi)} (paired cluster bootstrap; CR1 of the stacked system {eci('T3a', 'intake minus pay (SD units)')}), {dsd.verdict} (MDE {dsd.mde:.3f} SD). Relative to the pay slope's point estimate, the interval allows an intake slope of {rlo:.1f} to {rhi:.1f} times it. The small positive point estimate depends on the scaling (exploratory, revision 1): the SD units divide by the within-subject SD of the intake share, which is very small in {fl_txt}, against a median of {rv['floor_sd_median']:.3f} across the {r3['S']} subjects. Without them the difference is {ec('T3a', 'without floor subjects: intake minus pay (SD units)')}; the scale-free comparison, within-provider corr(REF rank, intake) − corr(REF rank, pay), is {ec('T3a', 'within-provider corr(REF rank, intake) minus corr(REF rank, pay)')}; with 'low' band cells coded as 1 or 2 the SD-unit difference is {ec('T3a-low', chr(39) + 'low' + chr(39) + ' = 1: intake minus pay (SD units)')} and {ec('T3a-low', chr(39) + 'low' + chr(39) + ' = 2: intake minus pay (SD units)')}. None excludes 0. In natural units the intake share with ≥ 360 points is {f3(100 * float(r3['beta'][3]), 1)} percentage points higher per SD of REF rank {fci(100 * float(r3['lo'][3]), 100 * float(r3['hi'][3]), 1)}. With the mean band score as the intake measure the difference is {ec('T3a', '  variant: band score minus pay (SD units) (paired cluster bootstrap)')}. Across providers within a subject (T3b, {len(A3['subs'])} subjects): ρ(REF, intake) {ec('T3b', 'mean rho(REF GPA, intake)')} vs ρ(REF, pay) {ec('T3b', 'mean rho(REF GPA, pay)')}, difference {ec('T3b', 'REF: rho(intake) minus rho(pay)')} ({v('T3b', 'REF: rho(intake) minus rho(pay)')}, MDE {row('T3b', 'REF: rho(intake) minus rho(pay)').mde:.3f}); for G, {ec('T3b', 'mean rho(G, intake)')} vs {ec('T3b', 'mean rho(G, pay)')}, difference {ec('T3b', 'G: rho(intake) minus rho(pay)')} ({v('T3b', 'G: rho(intake) minus rho(pay)')}). No pre-specified design detects a difference. Exploratory: net of institution selectivity, REF GPA keeps a similar small partial correlation with intake ({ec('T3b', 'mean partial rho(REF GPA, intake | institution selectivity)')}) and with pay ({ec('T3b', 'mean partial rho(REF GPA, pay | institution selectivity)')}), while institution selectivity itself couples with the programme's intake at {ec('T3b', 'mean rho(institution selectivity, intake)')}. With UCAS demand (applications per acceptance) as the student measure, in the T3(b) design on the {len(ctx['A3u']['subs'])} subjects where UCAS groups match one clean subject, REF couples with demand at {ec('T3b-UCAS', 'mean rho(REF GPA, demand)')} and with pay at {ec('T3b-UCAS', 'mean rho(REF GPA, pay)')} (difference {ec('T3b-UCAS', 'REF: rho(demand) minus rho(pay)')}, {v('T3b-UCAS', 'REF: rho(demand) minus rho(pay)')}); G at {ec('T3b-UCAS', 'mean rho(G, demand)')} and {ec('T3b-UCAS', 'mean rho(G, pay)')} (difference {ec('T3b-UCAS', 'G: rho(demand) minus rho(pay)')}, {v('T3b-UCAS', 'G: rho(demand) minus rho(pay)')}).\n")
    # 4 T4
    diffs = [f"{q.replace(' minus ', ' − ')} {f3(r.est)} {fci(r.lo, r.hi)} ({r.verdict}, MDE {r.mde:.3f}; two-stage {fci(rt.lo, rt.hi)}, {rt.verdict})"
             for q, r, rt in [("REF-pay minus REF-intake", d1, d1t), ("REF-pay minus pay-intake", d2, d2t),
                              ("REF-intake minus pay-intake", d3, d3t)]]
    ext_txt = ", ".join(f"{PAIR_LAB[f'{a}~{b_}']} {f3(T4e['pts'].mean(0)[j])}" for j, (a, b_) in enumerate(T4e["pairs"]))
    sc_txt = ", ".join(f"{PAIR_LAB[f'{a}~{b_}']} {f3(T4s['pts'].mean(0)[j])}" for j, (a, b_) in enumerate(T4s["pairs"]))
    e17_txt = ", ".join(f"{q} {f3(row('T4-econ', 'Economics UoA 16 else 17: mean rho ' + q).est)}" for q in labs)
    r14_txt = ", ".join(f"{q} {f3(row('REF2014', 'REF 2014: mean rho ' + q).est)}" for q in labs)
    ud_txt = "; ".join(f"{PAIR_LAB[f'{a}~{b_}']} {ec('T4-UCAS', 'mean rho ' + PAIR_LAB[f'{a}~{b_}'])}"
                       for a, b_ in T4u["pairs"] if "demand" in b_)
    uo_txt = "; ".join(f"{PAIR_LAB[f'{a}~{b_}']} {f3(T4u['pts'].mean(0)[j])}"
                       for j, (a, b_) in enumerate(T4u["pairs"]) if "demand" not in b_)
    best = int(np.argmax(top_share))
    lows = [PAIR_LAB[k] for k in ([f"{a}~{b_}" for a, b_ in T["pairs"]][int(np.argmin(T["pts"].mean(0)))] for T in (T4, T4e, T4s))]
    assert len(set(lows)) == 1
    rs = row("T4-score", "REF-intake(band score) minus pay-intake(band score)")
    lowtxt = "; ".join(f"'low' = {f}: REF-pay {f3(low[f]['mean rho REF-pay'].est)}, REF-intake {f3(low[f]['mean rho REF-intake'].est)}, pay-intake {f3(low[f]['mean rho pay-intake'].est)}" for f in (1, 2))
    L.append(f"4. **T4 — Triad: across providers within a subject all three rankings agree strongly; REF–pay has the lowest point estimate, but its gap to the other two pairs is marginal, and which of those two is highest depends on how suppressed intake cells are coded.** "
             f"Mean over {len(T4['subs'])} clean subjects (subject bootstrap): "
             + "; ".join(f"{labs[j]} {ec('T4', 'mean rho ' + labs[j])}" for j in order)
             + ". Pairwise differences (pre-specified subject bootstrap; the pre-specified two-stage CI alongside): " + "; ".join(diffs)
             + f". Under the subject bootstrap both REF–pay differences exclude 0, but with provider-level sampling error included neither does (two-stage upper bounds {f3(d1t.hi)} and {f3(d2t.hi)}): the evidence that REF and pay agree least is marginal. {labs[best]} agrees most in {pc(top_share[best])} of subject-bootstrap replicates ("
             + ", ".join(f"{labs[j]} {pc(top_share[j])}" for j in range(3)) + "). "
             + f"REF-pay is the lowest mean on the extended crosswalk ({len(T4e['subs'])} subjects: {ext_txt}), with Economics scored by UoA 16 else 17 ({e17_txt}), with REF 2014 instead of REF 2021 ({r14_txt}), when any one subject is left out (in all {len(loo)} leave-one-out runs, with pay-intake highest in each), and with the mean band score as the intake measure ({sc_txt}; post hoc: the band score is the pre-specified S-LEO variant, but it was added to T4 after the first run). "
             + f"The intake share has a floor: {pc(tz['share'])} of the T4 programmes have a share of 0, none of them a released zero. In {zw.get('low', 0)} of the {tz['n']} programmes both ≥ 360-point bands are published as 'low' (1–2 graduates, not zero) in every cohort, and in {zw.get('no rows', 0)} neither band has a released row; scripts/62's definition counts both as 0. Coding each 'low' band cell as 1 or as 2 graduates (exploratory, same cells) leaves {pc(low[1]['share of T4 programmes with intake share 0'].est)} at 0 and gives {lowtxt}: pay-intake's lead over REF-intake shrinks from {f3(-d3.est)} to {f3(-low[1]['REF-intake minus pay-intake'].est)} and {f3(-low[2]['REF-intake minus pay-intake'].est)}, so 'pay and intake agree most' is not robust to this coding, while REF-pay stays lowest. With the band score the two point estimates are also close (REF-intake minus pay-intake {f3(rs.est)} {fci(rs.lo, rs.hi)}). "
             + f"UCAS demand (applications per acceptance, cycles 2019–2021; {len(T4u['subs'])} subjects whose CAH1 group is a single clean CAH2 subject) agrees less with the other three than they agree with one another in point estimates: {ud_txt} (same cells: {uo_txt}); in {ud_pos} of the {len(ud_sb)} paired differences between a demand pair and a non-demand pair the demand pair is lower; {ud_det} of the {len(ud_sb)} are detected under the subject bootstrap and {ud_det_ts} under the two-stage bootstrap.\n\n"
             + f"   Within a university the triad is weak (exploratory; provider and subject fixed effects removed, {wt['n']} programmes, Pearson, cluster bootstrap): corr(REF rank, pay) {ec('T3a', 'within-provider corr(REF rank, pay)')}, corr(REF rank, intake) {ec('T3a', 'within-provider corr(REF rank, intake)')}, corr(pay, intake) {ec('T3a', 'within-provider corr(pay, intake)')}. Against across-provider means of {f3(mean4.min())} to {f3(mean4.max())} (Spearman, a different statistic, so the comparison is indicative), most of the agreement among the three markets is agreement about universities, not about a university's departments; part of the within-university weakness may be measurement (the REF test–retest correlation within a university is {r14['within-provider corr of REF 2021 and REF 2014 ranks (test-retest reliability)'].est:.2f}; answer 2). "
             + f"Exploratory, extensive margin: at REF-submitting universities, subjects whose department submitted to the matched UoA ({pc(t3['margin_share'])} of programmes) pay {ec('T3a', 'REF-submitted minus not: log pay (within provider)')} log points more and have a {ec('T3a', 'REF-submitted minus not: intake share (within provider)')} higher intake share than subjects without a submission. No submission to the matched UoA does not mean no research (Caveats); without Economics, Biosciences and Architecture the two differences are {ec('T3a', 'REF-submitted minus not: log pay (without Economics, Biosciences, Architecture)')} and {ec('T3a', 'REF-submitted minus not: intake share (without Economics, Biosciences, Architecture)')}, and counting a submission to an alternative UoA (Economics 17; Biosciences 3, 4, 6, 7; Architecture 12, 32) as submitted they are {ec('T3a', 'REF-submitted minus not: log pay (submission to the matched or an alternative UoA counts)')} and {ec('T3a', 'REF-submitted minus not: intake share (submission to the matched or an alternative UoA counts)')}.\n")
    L.append(f"**Reading against the working headline.** The UK REF data confirm the headline's placement of agreement: the academic, employer and student rankings of programmes agree mostly about universities (across-provider mean ρ within subjects {f3(ctx['range_across'][0])} to {f3(ctx['range_across'][1])}, and G, the hiring rank, matches REF at {f3(t1['G_gpa']['est'])} across universities), and much less about departments within a university (within-provider correlations {f3(ctx['range_within'][0])} to {f3(ctx['range_within'][1])}). REF makes the department level testable in the UK, which the ORCID field axis could not (UK_LEO_RESULT answer 6). There the department pay gradient is {pct(b0)} per SD; the intervals are too wide to rank it against the US {pct(us['a']['est'])} (institution and field FE) or {pct(us['main']['est'])} (main specification). {pc(t2['intake_share']['est'])} of it (point estimate; 95% CI {pc(t2['intake_share']['lo'])} to {pc(t2['intake_share']['hi'])}; {pc(s2.est)}–{pc(s1.est)} under the 'low' recodings) moves with the programme's own intake. The within-university REF axis is noisy (test–retest {r14['within-provider corr of REF 2021 and REF 2014 ranks (test-retest reliability)'].est:.2f} between REF 2014 and REF 2021), and the contemporaneous REF 2014 rank gives a larger slope; both suggest that the REF 2021 department-level slope understates rather than overstates the association (exploratory). For the next project on students' choice of university × field: within a university, higher-attainment students are found in the subjects with stronger research departments, and pay follows the same order; public aggregates cannot tell whether students choose these departments for their research standing, their peers or their pay, which is what admission-cutoff designs can separate.\n")
    # ---------------- pre-spec and deviations
    L.append("## Pre-specification and deviations\n")
    L.append(f"The tests T1–T4, the crosswalk, the samples and the inference were written into the script header on 2026-09-27 before the REF, UCAS or student-side data were opened (block md5 at freeze `{PRESPEC_MD5}`; the script asserts the block is unchanged, current md5 `{prespec_md5()}`). The block was also written to a file at the freeze; `data/interim/{FROZEN.name}` is a copy with the file time kept (md5 `{md5(FROZEN)}`, identical to the block; modified {frozen_t}). That is after the REF download (2026-09-27T18:07:38Z) and before the first development run of the analysis (18:21Z). The file time is the only timestamp: this workflow does not commit, so the durable record would be a git commit of the script by the author. Every row marked *pre* in the table below is a pre-specified quantity; *sens* rows are the pre-specified sensitivities; *post hoc* rows are a pre-specified variant applied to a design after seeing a result; unmarked rows are exploratory (rows added in revision 1 say so in the spec column). Deviations and choices made after the freeze:\n")
    L.append("- **Discover Uni not obtained.** The pre-specified preferred student source could not be downloaded (HESA's site returns a Cloudflare browser challenge to scripted requests; SOURCES.md §10i), as already recorded in the pre-specification; the fallbacks fixed there are used.")
    L.append(f"- **UCAS link (operational, decided after reading the file layout, before any result).** Pooled acceptances ≥ 10 per provider × group; names matched by normalised name with ', University of London' / ', <place>' suffixes dropped, plus {len(UCAS_ALIAS)} hand aliases (listed in the script). "
             f"{ctx['dinfo']['n_matched']} of {ctx['dinfo']['n_names']} UCAS provider names in the {len(UCAS_CAH1)} groups link to a LEO HEI; the unmatched names are further-education colleges, private providers and Northern Irish universities, which are not in the LEO HEI sample.")
    L.append("- **Mean band score as the intake measure in T4 (post hoc) and T3a (sensitivity).** The band score is the pre-specified S-LEO variant; it was added to T4 after the first run showed the floor of the share measure (answer 4), so its T4 rows are labelled post hoc (revision 1; the first version labelled them as pre-specified sensitivities).")
    L.append("- **Two-stage CIs of the T4 differences (revision 1).** The pre-specification asks for the two-stage CI alongside the subject bootstrap in T4; the first version printed it for the three means but not for their differences. Revision 1 adds it for every T4 and T4-UCAS difference (answer 4). Everything else added in revision 1 is exploratory.")
    L.append("- **Celtic studies** (a LEO CAH2 subject) was omitted from the crosswalk by oversight and is treated as unmapped.")
    L.append("- **Inference details not stated in the pre-specification:** the intake-control comparison in T2(a) fits both specifications on the same cluster draws (same seed), so the drop and the share removed are paired; the REF rank z is computed once on each subject's cell and held fixed in the cluster bootstrap; T3(a)'s within-subject SDs are computed on the final regression cells; the CR1 small-sample factor does not count provider fixed effects (nested in the clusters). BLAS threads are fixed at 1 instead of 8 (the matrices are small; faster, and results do not depend on the thread count).\n")
    # ---------------- key numbers
    L.append("## Key numbers\n")
    L.append(f"CI = 95%. *CI kind*: CR1 = provider-clustered, t with G − 1 df; bootstrap = percentile (provider cluster bootstrap for within-provider slopes; two-stage subject + provider bootstrap for across-provider means; subject bootstrap for T4 rows without '(two-stage)'), {B:,} replicates; normal = estimate ± 1.96 SE. SE = CR1 SE or bootstrap SD. MDE = 2.80 SE (80% power, 5% two-sided). *status*: pre = pre-specified, sens = pre-specified sensitivity, post hoc = pre-specified variant applied after seeing a result (exploratory), blank = exploratory; verdict per the pre-specified reading rule (CI excludes 0), shown for pre rows only. k = subjects, providers or programmes entering the statistic. Coverage rows give counts in *estimate* and a share in the CI column. 'low' = f: LEO band cells published as 'low' (1–2 graduates) coded as f instead of 0.\n")
    L.append("| section | quantity | spec | estimate | 95% CI | CI kind | SE | MDE | status | verdict | k |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in R.iterrows():
        if r.section == "coverage":
            est = f"{r.est:.3f}" if r.quantity.startswith("share") else f"{r.est:,.0f}"
            cis = f"share {r.lo:.3f}" if np.isfinite(r.lo) else "—"
        else:
            d = 3
            if r.section in ("T2a", "T2a-low") or (r.section == "REF2014" and "slope" in r.quantity):
                d = 4
            plain = r.quantity.startswith(("AUC", "share of bootstrap", "share of two-stage")) or "share of T4 programmes" in r.quantity
            est = f"{r.est:+.{d}f}" if not plain else f"{r.est:.3f}"
            cis = fci(r.lo, r.hi, d) if np.isfinite(r.lo) else "—"
        se = f"{r.se:.4f}" if np.isfinite(r.se) else "—"
        mde = f"{r.mde:.4f}" if np.isfinite(r.mde) else "—"
        qq, sp = r.quantity.strip().replace("|", "\\|"), str(r.spec).replace("|", "\\|")
        L.append(f"| {r.section} | {qq} | {sp} | {est} | {cis} | {r.ci or '—'} | {se} | {mde} | {r.status or '—'} | {r.verdict or '—'} | {r.k if r.k != '' else '—'} |")
    L.append("")
    # ---------------- method
    L.append("## Method\n")
    L.append(f"- **REF 2021.** `REF 2021 Results - All - 2022-05-06.xlsx` (md5 checked at run time), overall quality profiles of {len(ctx['sub'])} submissions by {cov['ref_inst']} institutions (joint submissions appear once per partner institution with its own FTE). GPA = (4·%4* + 3·%3* + 2·%2* + 1·%1*)/100; variant %4*. Provider × UoA = FTE-weighted over multiple submissions; institution = FTE-weighted over all its submissions; scores are rounded to 1e-9 so that equal grades tie exactly in the ranks. Linked to LEO by UKPRN: {cov['ref_inst_in_leo_hei']} REF institutions are LEO HEIs in the sample ({100 * cov['ref_fte_in_leo_hei']:.1f}% of REF FTE); {cov['primary_with_ref']} of the {cov['primary']} scripts/62 primary providers have a REF submission.")
    L.append(f"- **Crosswalk.** CAH2 → UoA as pre-specified (`data/interim/uk_triad_crosswalk.csv`; Appendix A). Clean subjects: {', '.join(f'{s} ({CLEAN[s][0]})' for s in CLEAN)}. The clean UoAs hold {100 * cov['ref_fte_clean_uoa']:.1f}% of REF FTE (clean + extended {100 * cov['ref_fte_ext_uoa']:.1f}%). Of {cov['prog']:,} LEO programmes (HEI × CAH2 with a released YAG5 median, cohorts 2013–16; {cov['n_hei']} HEIs), {cov['prog_clean']:,} are in clean subjects ({100 * cov['grads_clean']:.1f}% of graduates) and {cov['prog_clean_ref']:,} of those have a REF score ({100 * cov['grads_clean_ref']:.1f}% of all graduates); the extended subjects add {cov['prog_ext_ref']:,} programmes with a score. A programme has no score when its provider made no submission to the matched UoA (which does not mean its staff do no research; Caveats); the within-provider analyses use only programmes with a score (the extensive margin is an exploratory row).")
    L.append(f"- **Programme variables.** Pay = mean over cohorts 2013/14–2016/17 of log median earnings at YAG5, all graduates (released medians only; scripts/62's loader, multi-site rows resolved as there). Intake (S-LEO) = pooled over the same cohorts, (band 1 + band 2) / known bands 1–9 of graduates' prior attainment (scripts/62's definition; band cells published as 'low' count as 0, as does a band with no row); band score = (5·b1 + 4·b2 + 3·b3 + 2·b4 + 1·b5 + 0·b6)/(b1..b6). Institution selectivity = scripts/62's institution-wide share with ≥ 360 points, mean over the cohorts. G = scripts/62's academia-wide SpringRank.")
    L.append(f"- **T1.** Spearman across providers; provider bootstrap by exact frequency weights (scripts/62's weighted rank statistics), one draw shared by all T1 statistics (paired differences). AUC = Mann–Whitney U / (n₁n₂).")
    L.append(f"- **T2(a), T3(a).** OLS with provider and subject fixed effects, the provider effects removed by demeaning and the subject dummies (and any controls) partialled out (Frisch–Waugh–Lovell; checked against the dummy-variable fit). Subjects need ≥ {NMIN} providers with the variables; providers with one subject are dropped, iteratively with the subject minimum. z = within-subject midrank of the REF score over the subject's cell, standardised to SD 1. SEs clustered by provider (CR1, factor G/(G−1)·(N−1)/(N−K), K without the nested provider effects); a provider cluster bootstrap (clusters drawn with replacement, a provider drawn twice enters as two providers) alongside. The US-analogue specification adds subject × z(G) and subject × z(institution selectivity) slopes (z = within-subject standardised ranks); the intake specification adds subject × z(programme intake) slopes. T3(a) divides pay and intake by their within-subject SDs on the regression cells and fits both on the same cells; the difference uses one bootstrap draw for both (paired) and, alongside, the CR1 variance of the stacked system. Leave-one-subject-out (revision 1): the T2(a) cells minus one subject, singletons dropped again, z kept.")
    L.append(f"- **T2(b), T3(b).** Per clean subject, Spearman (and partial Spearman: ranks residualised on the ranks of the control) across scripts/62 primary providers with all variables, ≥ {NMIN} providers. Two-stage bootstrap: providers resampled within subject (exact frequency weights, one draw per subject shared by all statistics), subjects resampled in the outer stage with an independently chosen inner replicate per drawn subject, one outer draw shared by all statistics of the analysis (paired differences).")
    L.append("- **T4.** Per clean subject, pairwise Spearman among REF GPA, pay and intake on all REF-scored HEIs with the three variables; mean over subjects; subject bootstrap (point correlations, subjects resampled) as pre-specified, two-stage bootstrap alongside (for the means and, from revision 1, for the paired differences: one outer draw shared by the three pairs). UCAS demand: pooled main-scheme applications / pooled accepted applicants, 2019–2021, on the CAH1 groups (04) psychology, (09) mathematical sciences, (11) computing, (13) architecture, building and planning, (16) law, (17) business and management, (22) education and teaching, each a single clean CAH2 subject.")
    c14 = rv["r14_cov"]
    L.append(f"- **Suppressed intake cells (revision 1, exploratory).** In the LEO file graduate counts are rounded to multiples of 5 and a count that rounds to 0 but is not 0 is published as 'low' (1–2 graduates); a band with no graduates has no row (no 'z' code occurs in these rows). scripts/62's loader turns 'low' into missing, so the panel counts it as 0. A second read of the LEO file keeps the code (with 'low' = 0 it reproduces the panel's intake exactly; checked at run time); 'low' = 1 and 'low' = 2 code every such cohort × band cell as 1 or 2 graduates. They are applied on the cells of the original analyses (T4, T3(a), T2(a) intake control), so only the coding changes.")
    L.append(f"- **REF 2014 (revision 1, exploratory).** `REF2014 Results.xlsx` from results.ref.ac.uk (md5 checked at run time): overall profiles of {c14['subs']:,} submissions by {c14['inst']} institutions, GPA as for 2021, weighted by the FTE of Category A staff submitted; linked by UKPRN ({c14['inst_hei']} institutions are LEO HEIs in the sample). REF 2014 had 36 UoAs; the clean subjects map to {', '.join(f'{s} ' + '+'.join(map(str, CLEAN14[s])) for s in CLEAN14)} (the four REF 2014 engineering UoAs were merged into REF 2021 UoA 12; FTE-weighted). {c14['clean14_ref']} clean programmes have a REF 2014 score, {c14['clean_both']} both scores. Reliability: on the programmes with both scores (subjects with ≥ {NMIN} providers, singletons dropped; {rel['N']} programmes, {rel['G']} providers), both ranks standardised within subject; r = correlation of their provider + subject FE residuals; disattenuated slope = β(REF 2021)/r; IV = cov(pay, z2014)/cov(z2021, z2014) on the residuals; one cluster-bootstrap draw for all of these.")
    L.append(f"- **Alternative UoAs (revision 1, exploratory).** Economics → 17 (Business and management); Biosciences → 3, 4, 6, 7; Architecture, building and planning → 12, 32: used only to code the extensive margin ('submitted' = a submission to the matched or an alternative UoA) and, for Economics, a T2(a)/T4 variant scoring Economics by UoA 16 and, where the provider made no UoA 16 submission, by its UoA 17 submission ({rv['e17_n']} of {ns['Economics']['n']} Economics programmes then have a score).")
    L.append(f"- **Reproducibility.** `numpy.random.SeedSequence({SEED}).spawn`, one child per named analysis in a fixed list, per-subject grandchildren in sorted subject order; names added in revision 1 are appended, so every first-version draw is unchanged (checked: all first-version rows reproduce); three names are reserved for analyses that were planned but not run. Single process, BLAS threads fixed at 1. Two consecutive runs give byte-identical tables and write-up (md5-checked).\n")
    # ---------------- caveats
    L.append("## Caveats\n")
    L.append(f"- **Timing.** REF 2021 assesses research from 2014–2020 (staff in post on the 2020 census date); the LEO graduates entered around 2010–2013 and their YAG5 pay falls in tax years 2019/20–2022/23. REF 2021 therefore measures a department's standing after the graduates were taught. REF 2014 (outputs 2008–2013, census date 31 October 2013) is contemporaneous with their studies; with it the within-university pay slope is larger ({pct(r14['within-provider slope of log pay per SD of REF 2014 GPA rank'].est)} vs {pct(b0)}) and the triad ordering is the same (answer 4). UCAS demand (2019–2021 cycles) is later still.")
    L.append(f"- **The within-university REF axis is noisy.** Across providers within a subject REF 2014 and REF 2021 ranks agree at mean Spearman {r14['mean across-provider Spearman(REF 2021, REF 2014) within subject'].est:.3f}, but their within-provider residuals correlate at only {r14['within-provider corr of REF 2021 and REF 2014 ranks (test-retest reliability)'].est:.2f}. To the extent that this is measurement noise rather than real change between the exercises, the pre-specified within-provider slopes (T2a, T3a) are attenuated, and the T3(a) comparison of intake and pay shares that attenuation. The disattenuated figures in answer 2 assume a stable true standing and are exploratory.")
    L.append("- **A UoA is not a teaching department.** Even the clean pairs are approximate (UoA 4 includes psychiatry and neuroscience, UoA 24 leisure and tourism, UoA 12 serves both engineering and materials). REF GPA is research quality of submitted staff, not size (size is the exploratory FTE row).")
    e_, b_, a_ = ns["Economics"], ns["Biosciences"], ns["Architecture, building and planning"]
    assert e_["rg_none_alt"] == len(e_["rg_none"])
    L.append(f"- **Selection into the sample; no score is not no research.** Programmes whose provider made no submission to the matched UoA have no score and drop out of the rank analyses; they are disproportionately at less research-intensive providers. A missing score can also mean that the subject's researchers were submitted to another UoA. Economics has a UoA 16 score for {e_['n_ref']} of {e_['n']} programmes: {len(e_['rg_none'])} of the {e_['n_rg']} Russell Group economics programmes have none ({'; '.join(e_['rg_none'])}), all {e_['rg_none_alt']} at providers with a UoA 17 (Business and management) submission, which is where economists in business schools are returned. Biosciences has a UoA 5 score for {b_['n_ref']} of {b_['n']} programmes; {b_['n_none_alt']} of the {b_['n_none']} without one are at providers that submitted to UoA 3, 4, 6 or 7. Architecture, building and planning has a UoA 13 score for {a_['n_ref']} of {a_['n']} ({'; '.join(a_['rg_none'])} among the Russell Group without); {a_['n_none_alt']} of the {a_['n_none']} without are at providers that submitted to UoA 12 or 32. In the rank analyses this thins these subjects towards providers with a dedicated submission; the leave-one-subject-out runs and the Economics UoA 17 variant (answers 2 and 4) show that the T2(a) slope and the T4 ordering do not hinge on any one subject. The extensive margin is re-estimated with these three subjects excluded and with alternative UoAs counted as submitted (answer 4). The within-provider design compares subjects of the same provider, so provider-wide differences do not enter, but subject × provider factors other than research standing (professional accreditation, local labour markets for some subjects, London) are not controlled.")
    L.append(f"- **The intake measure is the graduates' entry attainment,** not applicants' demand or entry cutoffs: it counts only students who graduated and only those with a known band, and band cells of 1–2 graduates are published as 'low' and counted as 0 (answer 4 recodes them as 1 and 2). It has a floor: {pc(tz['share'])} of the T4 programmes have a share of 0, almost all because both ≥ 360-point bands are 'low' or unpublished; with 'low' = 1 the share at 0 is {pc(low[1]['share of T4 programmes with intake share 0'].est)}. The band-score variant covers the lower part of the distribution. Discover Uni's course-level tariff of entrants, the pre-specified preferred measure, could not be downloaded.")
    L.append(f"- **Pay contains the return to graduates' own attainment,** so part of the pay–intake agreement is composition (higher-attainment graduates earn more wherever they studied; scripts/62 finds a within-provider band gradient), not two markets agreeing. Intake and pay also come from the same LEO rows, so programme-size-related noise is shared; their correlation is not a pure market-agreement quantity. The UCAS demand ratio mixes popularity with capacity and is rounded to multiples of 5; it covers {len(UCAS_CAH1)} subjects only.")
    L.append("- **Comparison with the US** sets a UK REF rank (research quality) beside a US hiring rank (F); the constructs differ, as do the earnings horizon (UK 5 years, US 4 years) and the fields. The difference row is a rough, exploratory normal approximation that treats the two estimates as independent; its MDE exceeds both slopes, so it can neither show a difference nor support equality.")
    L.append("- **Nothing here is causal.** Positive within-provider slopes describe where higher pay and higher-attainment students are found, not what a department's research does for its graduates.\n")
    # ---------------- revisions
    L.append("## Revisions\n")
    L.append(f"Revision 1 answers an independent verification of the first version. The verifier rebuilt the panel from the LEO zip and a fresh REF download (differences < 1e-8) and regenerated the write-up byte-identically; every first-version number reproduces. Each point below was checked against the data before it was applied. Every first-version number is unchanged in this run (checked row by row against the first version's summary table; the seeds of the first-version analyses are unchanged). The only first-version rows that changed are the two extensive-margin rows in within-subject SD units, which were dropped (item 4), and the status of the six T4-score rows (item 5). The ten points, in the verifier's order:\n")
    L.append(f"1. **T4 differences lacked the pre-specified two-stage CI (major; agreed).** The pre-specification asks for the two-stage CI alongside; the first version printed it for the means only. Added for every T4 and T4-UCAS difference (and the share of two-stage replicates in which each pair is highest). REF-pay − REF-intake: subject bootstrap {eci('T4', 'REF-pay minus REF-intake')}, two-stage {eci('T4', 'REF-pay minus REF-intake (two-stage)')}; REF-pay − pay-intake: {eci('T4', 'REF-pay minus pay-intake')} and {eci('T4', 'REF-pay minus pay-intake (two-stage)')}. 'REF and pay least' is now stated as the lowest point estimate, detected under the subject bootstrap but not under the two-stage bootstrap, i.e. marginal (Answer, answer 4). T4-UCAS: {ud_det} of {len(ud_sb)} demand vs non-demand differences detected under the subject bootstrap, {ud_det_ts} under the two-stage bootstrap.")
    L.append(f"2. **'No graduate with ≥ 360 points' was wrong (major; agreed).** None of the {int(round(tz['share'] * tz['n']))} zero shares is a released zero: in {zw.get('low', 0)} programmes both bands are 'low' (1–2 graduates) in every cohort, in {zw.get('no rows', 0)} there is no released row. The sentence now says so. New exploratory recoding ('low' = 1, 2; same cells): zero share {pc(low[1]['share of T4 programmes with intake share 0'].est)}; T4 pay-intake minus REF-intake {f3(-low[1]['REF-intake minus pay-intake'].est)} / {f3(-low[2]['REF-intake minus pay-intake'].est)} (first version {f3(-d3.est)}); T3(a) difference {f3(row('T3a-low', chr(39) + 'low' + chr(39) + ' = 1: intake minus pay (SD units)').est)} / {f3(row('T3a-low', chr(39) + 'low' + chr(39) + ' = 2: intake minus pay (SD units)').est)} (first version {f3(dsd.est)}); T2(a) share removed by the intake control {pc(s1.est)} / {pc(s2.est)} (first version {pc(t2['intake_share']['est'])}). 'pay-intake highest' is now reported as not robust to this coding. Here the cells of each original analysis are held fixed, so only the coding changes. The verifier's T4 figures (REF-pay 0.581; REF-intake 0.659 / 0.663; pay-intake 0.671 / 0.667; 15% at 0) also let in programmes whose known bands are all 'low', and its share removed (54% → 51% / 46%) was computed on the T3(a) cells rather than on the T2(a) intake-control cells behind the reported {pc(t2['intake_share']['est'])}; its T3(a) figures (+0.006 / +0.003) match the ones here. The reading is the same either way.")
    L.append(f"3. **Equivalence wording (minor; agreed).** 'About the US rate', 'about as much as pay does' and 'about as closely as' are gone. UK − US {f3(dUS.est, 4)} {fci(dUS.lo, dUS.hi, 4)} (MDE {dUS.mde:.4f}) is described as not distinguishable, with an interval that includes the value at which the UK slope would be twice the US one; the UK main-spec analogue is {float(mt['beta'][0]) / us['main']['est']:.1f} times the US main estimate (not tested). T3(a): relative to the pay slope, the interval allows an intake slope of {rlo:.1f}–{rhi:.1f} times it. T1 and T2(b) comparisons are stated as not detectably different, with their MDEs.")
    L.append(f"4. **T3(a) SD scaling (minor; agreed).** The within-subject SD of the intake share is very small in {' and '.join(fl)} (median {rv['floor_sd_median']:.3f}); without them the SD-unit difference is {ec('T3a', 'without floor subjects: intake minus pay (SD units)')} (the verifier: 0.000). Added the scale-free within-provider correlation difference {ec('T3a', 'within-provider corr(REF rank, intake) minus corr(REF rank, pay)')}; the verifier's version with within-provider residual SDs as units (+0.071 [−0.089, +0.220]) likewise does not exclude 0. The 'lean towards intake' sentence is removed; no design detects a difference. The extensive margin's within-subject SD-unit rows (pay +0.189, intake +0.000) are dropped: the SD scaling multiplies each subject's intake shares by the inverse of their SD, which is tiny in the same two subjects, and the SD-unit intake row read as if it contradicted the natural-unit one (+0.016 [+0.006, +0.027]); the natural-unit rows stay.")
    L.append("5. **T4 band-score rows labelled as pre-specified sensitivities (minor; agreed).** They are now *post hoc*, and answer 4 says so where it uses them.")
    L.append(f"6. **Selective coverage of Economics, Biosciences and Architecture (minor; agreed).** The Caveats now say that no score is not no research and name the {len(ns['Economics']['rg_none'])} Russell Group economics programmes without a UoA 16 score. The extensive margin is re-estimated without the three subjects and with alternative UoAs counted as submitted (answer 4; both leave both differences positive with CIs above 0). Also added: leave-one-subject-out (T2(a) {f3(loo[smin]['beta'], 4)} to {f3(loo[smax]['beta'], 4)}, every CR1 CI above 0; T4 order unchanged in {loo_order} of {len(loo)}) and Economics scored by UoA 16 else 17 (T2(a) {f3(row('T2a', '  Economics scored by UoA 16, else UoA 17').est, 4)}). The verifier reported a leave-one-out range of 0.0155–0.0210 and 0.0196 without Economics; this script's version (z kept from the full cells, as in the main fit) gives {f3(loo[smin]['beta'], 4)} to {f3(loo[smax]['beta'], 4)} and {f3(loo['Economics']['beta'], 4)} without Economics. The source of the difference was not traced; the reading is the same.")
    L.append("7. **Reserved seed names (minor; agreed).** `e_gvert_inner`, `e_gvert_outer` and `t2a_yag3` were planned analyses that were never run (no code, no output); they are now commented as reserved, not run, and kept so that the seed order does not change. Revision-1 names are appended after them.")
    L.append("8. **REF licence and row count (minor; agreed).** SOURCES.md §10i now records the REF copyright page (https://2021.ref.ac.uk/copyright/index.html: REF submissions information, including downloadable submissions data, may be used under CC BY 4.0, which requires attribution) and the row count: 7,552 data rows (1,888 submissions × 4 profiles) plus one trailing row holding a single space. The REF 2014 download has its own provenance subsection (§10i-b).")
    L.append(f"9. **Freeze timestamp (minor; partly applied).** The frozen block's file is now kept in the repository tree as `data/interim/{FROZEN.name}` (copied with its file time; the script checks its md5 against the block) and its time is cited above. The verifier's fix, a git commit, is not done: this workflow's rules forbid commits, so that step is left to the author.")
    L.append(f"10. **Timing and reliability (minor; agreed).** REF 2014 added (exploratory): T2(a) with the REF 2014 rank {f3(r14['within-provider slope of log pay per SD of REF 2014 GPA rank'].est, 4)}; within-provider test–retest r = {f3(r14['within-provider corr of REF 2021 and REF 2014 ranks (test-retest reliability)'].est)}; disattenuated REF 2021 slope {f3(r14['REF 2021 slope / reliability (disattenuated)'].est, 4)} and IV {f3(r14['IV slope: REF 2021 rank instrumented by REF 2014 rank'].est, 4)}; T4 with REF 2014 keeps the order (answers 2 and 4, Caveats).\n")
    # ---------------- appendix
    L.append("## Appendix A — per subject\n")
    L.append("n LEO = programmes with a YAG5 median; n REF = of those, with a REF score (share of the subject's graduates in brackets). A programme without a score may still have research returned to another UoA (Caveats; Economics, Biosciences and Architecture especially). T4 columns: across-provider Spearman on REF-scored HEIs (n T4). T2b/T3b: scripts/62 primary providers with REF (n T2b). — = subject not in that analysis (fewer than 15 providers or not mapped).\n")
    SJ = ctx["SJ"]
    L.append("| subject | map | UoA | n LEO | n REF (grads share) | n T4 | REF–pay | REF–intake | pay–intake | n T2b | ρ(REF, pay) | ρ(G, pay) | ρ(REF, G) | ρ(REF, intake) | ρ(G, intake) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def g(s, c, d=3):
        x = SJ.loc[s, c] if c in SJ.columns else np.nan
        return "—" if not np.isfinite(x) else f"{x:+.{d}f}"

    def gi(s, c):
        x = SJ.loc[s, c] if c in SJ.columns else np.nan
        return "—" if not np.isfinite(x) else f"{int(x)}"
    for mp in ["clean", "ext", "unmapped"]:
        for s in SJ[SJ.xw == mp].sort_values("n_prog", ascending=False).index:
            pre = "t4" if mp == "clean" else "t4e"
            L.append(f"| {s} | {mp} | {SJ.loc[s, 'uoa'] or '—'} | {int(SJ.loc[s, 'n_prog'])} | {int(SJ.loc[s, 'n_ref'])} ({SJ.loc[s, 'grads_share_ref']:.2f}) | {gi(s, pre + '_n')} | {g(s, pre + '_ref_gpa~pay')} | {g(s, pre + '_ref_gpa~top')} | {g(s, pre + '_pay~top')} | {gi(s, 't2b_n')} | {g(s, 't2b_rho_ref')} | {g(s, 't2b_rho_G')} | {g(s, 't2b_rho_refG')} | {g(s, 't3b_ref_top')} | {g(s, 't3b_G_top')} |")
    L.append("")
    L.append("## Provenance (for SOURCES.md)\n")
    L.append(f"- **REF 2021 results** — `{REF_URL}` (official results site; accessed 2026-09-27 18:07:38 GMT; server filename 'REF 2021 Results - All - 2022-05-06.xlsx'), stored as `data/raw/ref2021/ref2021_results_all_2022-05-06.xlsx`, {REF_XLSX.stat().st_size:,} bytes, md5 `{md5(REF_XLSX)}`. REF submissions information is licensed CC BY 4.0 (attribution required; https://2021.ref.ac.uk/copyright/index.html). Recorded in `data/raw/SOURCES.md` §10i.")
    L.append(f"- **REF 2014 results** (revision 1) — `{REF14_URL}` (results.ref.ac.uk, 'Download: results'; accessed 2026-09-27 21:37:10 GMT; server filename 'REF2014 Results.xlsx'), stored as `data/raw/ref2014/ref2014_results_all.xlsx`, {REF14_XLSX.stat().st_size:,} bytes, md5 `{md5(REF14_XLSX)}`. Recorded in `data/raw/SOURCES.md` §10i-b.")
    ui = ctx["uinfo"]
    L.append(f"- **UCAS end-of-cycle provider × subject group** (already on disk, SOURCES.md §9): `{ui['acc']['file']}` ({ui['acc']['measure']}; {ui['acc']['cycles']}; analysis classes '{ui['acc']['classes']}'; zip md5 `{ui['acc']['md5']}`) and `{ui['app']['file']}` ({ui['app']['measure']}; zip md5 `{ui['app']['md5']}`). Granularity: provider name × CAH level 1 group ({len([x for x in ui['acc']['groups'] if x != 'All'])} groups), no UKPRN.")
    L.append(f"- **DfE LEO provider-level data** and the **ORCID-derived UK hiring edges** as in UK_LEO_RESULT.md (not re-downloaded; LEO zip md5 `{md5(s62.LEO_ZIP)}`; revision 1 reads the zip a second time for the prior-attainment rows with their suppression codes).")
    L.append("- **Discover Uni (Unistats)**: not obtained (HTTP 403 Cloudflare challenge at hesa.ac.uk; Internet Archive HTTP 429), 2026-09-27; SOURCES.md §10i.")
    return "\n".join(L) + "\n"


def main():
    ctx = analyse()
    R = collect(ctx)
    R.to_csv(OUT_SUM, index=False, float_format="%.6g")
    ctx["SJ"].reset_index().to_csv(OUT_SUBJ, index=False, float_format="%.6g")
    cols = ["ukprn", "provider_name", "subject", "xw", "pay", "n_coh", "grads", "top", "score", "known", "ref_gpa",
            "ref_p4", "ref_fte", "ref_nuoa", "G", "primary", "russell", "inst_top", "demand", "app", "acc",
            # revision 1
            "zero_why", "top_l1", "top_l2", "score_l1", "score_l2", "ref14_gpa", "alt_sub", "ref_gpa_e17"]
    ctx["P"][cols].to_csv(OUT_PANEL, index=False, float_format="%.10g")
    xw = [dict(subject=s, crosswalk="clean", uoa=q) for s, v in CLEAN.items() for q in v] + \
         [dict(subject=s, crosswalk="extended", uoa=q) for s, v in EXT.items() for q in v] + \
         [dict(subject=s, crosswalk="unmapped", uoa="") for s in UNMAPPED + ["Celtic studies"]]
    X = pd.DataFrame(xw)
    un = ctx["sub"].drop_duplicates("uoa").set_index("uoa").uoa_name
    X["uoa_name"] = X.uoa.map(lambda q: un.get(q, "") if q != "" else "")
    X.to_csv(OUT_XW, index=False)
    OUT_MD.write_text(write_md(ctx, R), encoding="utf-8")
    import resource
    print(f"wrote {OUT_SUM.name}, {OUT_SUBJ.name}, {OUT_PANEL.name}, {OUT_XW.name}, {OUT_MD.name}; "
          f"{ctx['elapsed']:.0f} s; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")


if __name__ == "__main__":
    main()
