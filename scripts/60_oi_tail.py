"""Elite-tail outcomes: does academic prestige (Wapman academia-wide brand) predict reaching the top of
the earnings distribution beyond institutional selectivity and parental income, and more than it
predicts the median?

Why: every coupling in this project uses MEDIAN earnings (Scorecard / PSEO). Chetty-Deming-Friedman
show elite-college effects live in the upper tail, which medians cannot see. The Opportunity Insights
Mobility Report Cards (Chetty, Friedman, Saez, Turner, Yagan 2017/2020) publish, per college
(super_opeid), the share of children reaching the top 20/10/5/1 percent of their birth cohort's
individual-earnings distribution in 2014 (1980-82 birth cohorts, ages 32-34), with parents' income
distribution for the same children. Public data only. Descriptive, not causal.

Data
  OI Table 2  (mrc_table2.csv)  college-level outcomes, 1980-82 cohorts (k_median, k_mean, k_rank, k_q5,
                                k_top10pc, k_top5pc, k_top1pc; par_rank, par_top1pc, ...; parent-quintile-
                                conditional outcomes kq5_cond_parqX, ktop1pc_cond_parqX, k_rank_cond_parqX)
  OI Table 3  (mrc_table3.csv)  the same by birth cohort 1980-1991 (earnings still measured in 2014, so
                                cohort c is observed at age 2014-c)
  OI Table 10 (mrc_table10.csv) college characteristics: sat_avg_2001/2013, scorecard_rej_rate_2013,
                                Barron's 2009 index, 2000 IPEDS major shares by broad group
  OI Table 11 (mrc_table11.csv) OPEID(6-digit) -> super_opeid crosswalk
  Wapman et al. 2022 ranks.csv  academia-wide SpringRank ordinal (G = -Rank, higher = more prestigious)
                                and field ranks (F, supplementary ecological check)
  Scorecard institution file    UNITID / OPEID6 / INSTNM / MAIN (crosswalk) + current SAT_AVG, ADM_RATE
  IPEDS C2023_a (local)         only to check what the OI "major share" groups contain (label check)

Crosswalk (Wapman name -> Scorecard UNITID -> OPEID6 -> OI super_opeid)
  Route A1: normalized-name exact match to Scorecard INSTNM (src/crosswalks/institutions, the project
            convention); several UNITIDs with the same key -> the MAIN campus.
  Route A2: explicit alias table ALIASES below (Wapman name -> exact Scorecard INSTNM + state), used only
            where names differ in surface form ("-Main Campus", renamed institutions); every alias is
            asserted to hit exactly one Scorecard row.
  Route B : fallback where Route A has no OPEID6 in OI Table 11: normalized Wapman name == normalized
            OI Table 11 institution_name (2013 DoE names).
  Validation: where both routes resolve, super_opeids must agree; token-Jaccard similarity between the
  Wapman name and the OI name is listed for the lowest-similarity pairs.
  Exclusions: OI system groups (multi = 1: outcomes are averages over several universities); OI
  "insufficient data" (super_opeid = -1); several Wapman units on one single-college super_opeid ->
  keep the Scorecard MAIN campus (drops e.g. Weill Cornell, Brite Divinity, Rutgers-Newark).
  Flag: OPEID6 shared by more than one currently operating undergraduate Scorecard unit (branch
  campuses, e.g. Penn State): OI outcomes then cover all branches -> robustness excludes them.

Analysis sample C ("common"): single-college OI units with outcomes, SAT, rejection rate, Barron's
index and parental income. Rank-based throughout (Spearman; partial correlations from OLS residuals
of within-sample ranks, as scripts/55). Joint institution bootstrap (B draws, percentile CIs; the same
draws for every statistic so differences are paired).

  (a) coupling rho(G, y) and rho(SAT, y), rho(rejection rate, y) for y from the median to the top 1%;
      paired difference tail - median.
  (b) net of parental income: partial on par_rank + par_top1pc ("par"); parent-quintile-conditional
      outcomes (share reaching top 1% / top 20%, mean rank, among children from parent quintile q).
  (c) brand vs selectivity: partial rho(G, y | selectivity [SAT + rejection rate + Barron's dummies]),
      | selectivity + parents (headline), | + wider parent distribution + public/private + region
      (strict); selectivity net of brand + parents; standardized rank OLS horse race with HC1 SEs and
      Shapley R2 over blocks {brand}, {selectivity}, {parents}; "tail beyond the median":
      partial rho(G, top-x share | k_median [+ selectivity + parents]).
  (d) field mix (ECOLOGICAL): does the brand-outcome slope vary with the institution's 2000 share of
      health + public/social-service majors ("setting-priced" mix) or STEM + business majors?
      z(rank y) ~ zG + zM + zG*zM + selectivity + parents; and rho(G, y) within mix terciles.
  (e) supplementary, ECOLOGICAL: per Wapman field, field prestige F vs brand G against the
      institution-wide outcome (not the field's graduates). NOT a test of field vs brand pricing: G is
      close to the institution's average field prestige and the outcome pools all fields, so c_F < c_G
      and a near-zero partial rho(F, y | G) are expected either way. A two-scenario calibration
      (outcome = institution-average field prestige + noise, or brand + noise, noise set so that c_G
      matches the observed value) shows how much the statistics can discriminate.
  (f) supplementary: OI Table 3 age profile (cohorts 1980-1991 observed at ages 34-23 in 2014) and
      cross-cohort reliability of each outcome (1980/81/82 -> Spearman-Brown for the 3-cohort mean).

Reproducible: SEED fixed; outputs byte-identical on re-run.
Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/60_oi_tail.py
     (single process, ~3.5 min, peak RSS ~0.4 GB)
Outputs: data/interim/oi_tail.csv (all statistics, long format), outputs/figures/oi_tail.png,
         OI_TAIL_RESULT.md (root; gitignored).
"""
from __future__ import annotations

import sys
import hashlib
import importlib.util
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.crosswalks.institutions import normalize_institution_name as norm
from src.load_ar import load_ar_wapman

_spec = importlib.util.spec_from_file_location("s28", ROOT / "scripts" / "28_field_vs_generic_prestige.py")
_s28 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s28)
FIELDS66, LAB = _s28.FIELDS66, _s28.LAB

SEED = 60
B = 1000             # joint institution bootstrap draws, main sample (VM resource cap: <= 1000)
B_ROB = 1000         # robustness samples and age profile
NMIN_FIELD = 30      # (e) min institutions per field
SIM_SEED = 600       # (e) calibration simulation
SIM_R = 1000         # (e) calibration: noise draws
SIM_CAL_R = 200      # (e) calibration: draws used to set the noise level (bisection)
OI = ROOT / "data" / "raw" / "opportunity_insights"
WAPMAN = ROOT / "data" / "raw" / "wapman2022" / "ranks.csv"
SC_INST = ROOT / "data" / "raw" / "scorecard_inst" / "Most-Recent-Cohorts-Institution.csv"
IPEDS_C = ROOT / "data" / "raw" / "ipeds" / "C2023_a.csv"
OUT_CSV = ROOT / "data" / "interim" / "oi_tail.csv"
OUT_FIG = ROOT / "outputs" / "figures" / "oi_tail.png"
OUT_MD = ROOT / "OI_TAIL_RESULT.md"

OUT = ["k_median", "k_mean", "k_rank", "k_q5", "k_top10pc", "k_top5pc", "k_top1pc"]
TAILS = ["k_top10pc", "k_top5pc", "k_top1pc"]
OLAB = {"k_median": "median earnings", "k_mean": "mean earnings", "k_rank": "mean earnings rank",
        "k_q5": "share top 20%", "k_top10pc": "share top 10%", "k_top5pc": "share top 5%",
        "k_top1pc": "share top 1%"}
OLAB_PQ = {"ktop1pc_cond_parq1": "share top 1%, parents in q1", "kq5_cond_parq1": "share top 20%, parents in q1",
           "k_rank_cond_parq1": "mean rank, parents in q1", "ktop1pc_cond_parq5": "share top 1%, parents in q5",
           "k_rank_cond_parq5": "mean rank, parents in q5"}
OSHORT = {"k_median": "median", "k_mean": "mean", "k_rank": "mean rank", "k_q5": "top 20%",
          "k_top10pc": "top 10%", "k_top5pc": "top 5%", "k_top1pc": "top 1%"}
PAR = ["par_rank", "par_top1pc"]
PAR_PLUS = ["par_rank", "par_q5", "par_top10pc", "par_top1pc", "par_toppt1pc"]
SEL = ["SAT", "REJ"]                     # + Barron's dummies (categorical BARR)
# control sets: name -> (continuous rank covariates, categorical covariates)
CS = {
    "none": ([], []),
    "par": (PAR, []),
    "sel": (SEL, ["BARR"]),
    "sel_par": (SEL + PAR, ["BARR"]),
    "strict": (SEL + PAR_PLUS, ["BARR", "PUBLIC", "region"]),
}
CS_LAB = {"none": "raw", "par": "| parents", "sel": "| selectivity",
          "sel_par": "| selectivity + parents", "strict": "| strict"}
CS_DESC = {"none": "no controls",
           "par": "par_rank + par_top1pc",
           "sel": "SAT + rejection rate 2013 + Barron's 2009 category dummies",
           "sel_par": "selectivity (as above) + par_rank + par_top1pc",
           "strict": "selectivity + par_rank, par_q5, par_top10pc, par_top1pc, par_toppt1pc + public "
                     "+ census-region dummies"}

# Wapman name -> (exact Scorecard INSTNM, state, note). Only names the normalized-key join misses.
ALIASES = {
    "Albert Einstein College of Medicine of Yeshiva University": ("Albert Einstein College of Medicine", "NY", "graduate-only"),
    "Air Force Institute of Technology": ("Air Force Institute of Technology-Graduate School of Engineering & Management", "OH", "graduate-only"),
    "Arizona State University": ("Arizona State University Campus Immersion", "AZ", "Tempe campus"),
    "Baruch College (CUNY)": ("CUNY Bernard M Baruch College", "NY", ""),
    "Binghamton University, State University of New York": ("Binghamton University", "NY", ""),
    "Bowling Green State University": ("Bowling Green State University-Main Campus", "OH", ""),
    "Brooklyn College (CUNY)": ("CUNY Brooklyn College", "NY", ""),
    "City College (CUNY)": ("CUNY City College", "NY", ""),
    "College of Optometry, State University of New York": ("SUNY College of Optometry", "NY", "graduate-only"),
    "College of William & Mary, The": ("William & Mary", "VA", ""),
    "Colorado State University": ("Colorado State University-Fort Collins", "CO", ""),
    "Columbia University": ("Columbia University in the City of New York", "NY", ""),
    "Florida A&M University": ("Florida Agricultural and Mechanical University", "FL", ""),
    "Georgia Institute of Technology": ("Georgia Institute of Technology-Main Campus", "GA", ""),
    "Indiana University-Purdue University Indianapolis": ("Indiana University-Indianapolis", "IN", "IUPUI split in 2024; IU Indianapolis kept OPEID6 001813"),
    "Jewish Theological Seminary, The": ("Jewish Theological Seminary of America", "NY", ""),
    "John Jay College of Criminal Justice (CUNY)": ("CUNY John Jay College of Criminal Justice", "NY", ""),
    "Kent State University": ("Kent State University at Kent", "OH", ""),
    "Lehman College (CUNY)": ("CUNY Lehman College", "NY", ""),
    "Louisiana State University": ("Louisiana State University and Agricultural & Mechanical College", "LA", ""),
    "Louisiana State University School of Medicine in New Orleans": ("Louisiana State University Health Sciences Center-New Orleans", "LA", "health sciences center"),
    "Maharishi University of Management": ("Maharishi International University", "IA", "renamed"),
    "Miami University": ("Miami University-Oxford", "OH", ""),
    "New Mexico State University": ("New Mexico State University-Main Campus", "NM", ""),
    "North Carolina Agricultural and Technical State University": ("North Carolina A & T State University", "NC", ""),
    "North Carolina State University": ("North Carolina State University at Raleigh", "NC", ""),
    "North Dakota State University": ("North Dakota State University-Main Campus", "ND", ""),
    "Ohio State University, The": ("Ohio State University-Main Campus", "OH", ""),
    "Ohio University": ("Ohio University-Main Campus", "OH", ""),
    "Oklahoma State University": ("Oklahoma State University-Main Campus", "OK", ""),
    "Pennsylvania State University, The": ("Pennsylvania State University-Main Campus", "PA", ""),
    "Purdue University": ("Purdue University-Main Campus", "IN", ""),
    "Queens College (CUNY)": ("CUNY Queens College", "NY", ""),
    "Rosalind Franklin University": ("Rosalind Franklin University of Medicine and Science", "IL", "graduate-only"),
    "Rutgers  - Newark": ("Rutgers University-Newark", "NJ", ""),
    "Rutgers - New Brunswick": ("Rutgers University-New Brunswick", "NJ", ""),
    "SUNY Upstate Medical University": ("Upstate Medical University", "NY", ""),
    "St. John's University": ("St. John's University-New York", "NY", ""),
    "Stony Brook University, State University of New York": ("Stony Brook University", "NY", ""),
    "Texas A&M Corpus Christi": ("Texas A & M University-Corpus Christi", "TX", ""),
    "Texas A&M Kingsville": ("Texas A&M University-Kingsville", "TX", ""),
    "Texas A&M University": ("Texas A&M University-College Station", "TX", ""),
    "Texas A&M University-Commerce": ("East Texas A&M University", "TX", "renamed 2024 (same OPEID6)"),
    "Trinity International University": ("Trinity International University-Illinois", "IL", ""),
    "Tulane University": ("Tulane University of Louisiana", "LA", ""),
    "University at Albany, State University of New York": ("University at Albany", "NY", ""),
    "University at Buffalo, State University of New York": ("University at Buffalo", "NY", ""),
    "University of Akron, The": ("University of Akron Main Campus", "OH", ""),
    "University of Cincinnati, The": ("University of Cincinnati-Main Campus", "OH", ""),
    "University of Colorado Denver": ("University of Colorado Denver/Anschutz Medical Campus", "CO", ""),
    "University of Hawaii": ("University of Hawaii at Manoa", "HI", "flagship campus"),
    "University of Massachusetts Medical School": ("University of Massachusetts Chan Medical School", "MA", "graduate-only"),
    "University of Missouri": ("University of Missouri-Columbia", "MO", ""),
    "University of Montana, Missoula": ("The University of Montana", "MT", ""),
    "University of New Hampshire": ("University of New Hampshire-Main Campus", "NH", ""),
    "University of New Mexico, The": ("University of New Mexico-Main Campus", "NM", ""),
    "University of North Texas Health Science Center at Fort Worth": ("University of North Texas Health Science Center", "TX", ""),
    "University of Oklahoma": ("University of Oklahoma-Norman Campus", "OK", ""),
    "University of Pittsburgh": ("University of Pittsburgh-Pittsburgh Campus", "PA", ""),
    "University of South Carolina": ("University of South Carolina-Columbia", "SC", ""),
    "University of Tennessee, The": ("The University of Tennessee-Knoxville", "TN", ""),
    "University of Texas Medical Branch, The": ("The University of Texas Medical Branch at Galveston", "TX", ""),
    "University of Texas Southwestern Medical Center at Dallas, The": ("University of Texas Southwestern Medical Center", "TX", "graduate-only"),
    "University of Virginia": ("University of Virginia-Main Campus", "VA", ""),
    "University of Washington": ("University of Washington-Seattle Campus", "WA", ""),
    "Wright State University": ("Wright State University-Main Campus", "OH", ""),
}
# deliberately NOT aliased: "Saint Thomas University" (the Scorecard name "St. Thomas University" differs, and
# several St. Thomas institutions exist; route B then links it by exact OI name to OI's "Saint Thomas University",
# FL, OPEID6 001468), "LIU Post" / "Long Island University Brooklyn Campus" (both campuses now one
# Scorecard unit and one OPEID6 -> the OI outcome cannot be assigned to either campus), "Texas A&M Health
# Science Center" (now a non-main unit on Texas A&M's OPEID6), graduate schools with no Scorecard record.


# ---------------------------------------------------------------------------------------------
# data + crosswalk
# ---------------------------------------------------------------------------------------------
def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jacc(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / max(1, len(ta | tb))


def load_scorecard():
    cols = ["UNITID", "OPEID6", "INSTNM", "STABBR", "MAIN", "PREDDEG", "UGDS", "CURROPER",
            "SAT_AVG", "ADM_RATE"]
    s = pd.read_csv(SC_INST, usecols=cols, dtype=str)
    s["inst_key"] = s.INSTNM.map(norm)
    s["main"] = (s.MAIN == "1").astype(int)
    for c in ["UGDS", "SAT_AVG", "ADM_RATE"]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s["op6"] = pd.to_numeric(s.OPEID6, errors="coerce")
    # undergraduate units per OPEID6 (branch-campus flag)
    ug = s[(s.CURROPER == "1") & s.PREDDEG.isin(["2", "3"]) & (s.UGDS > 0)]
    s["n_ug_units_op6"] = s.op6.map(ug.groupby("op6").size()).fillna(0).astype(int)
    return s


def build_crosswalk():
    r = pd.read_csv(WAPMAN)
    ac = r[(r.TaxonomyLevel == "Academia") & r.InstitutionName.notna()].copy()
    ac["inst_key"] = ac.InstitutionName.map(norm)
    ac = ac.drop_duplicates("inst_key").rename(columns={"InstitutionName": "wapman_name", "Rank": "g_rank"})
    ac = ac[["wapman_name", "inst_key", "g_rank"]].reset_index(drop=True)
    sc = load_scorecard()
    t11 = pd.read_csv(OI / "mrc_table11.csv")
    t2 = pd.read_csv(OI / "mrc_table2.csv")

    # Route A1: normalized-name exact match; ties -> main campus, then larger UGDS
    a1 = sc.sort_values(["inst_key", "main", "UGDS"], ascending=[True, False, False]) \
           .drop_duplicates("inst_key")
    xw = ac.merge(a1[["inst_key", "UNITID", "op6", "INSTNM", "STABBR", "main", "n_ug_units_op6",
                      "SAT_AVG", "ADM_RATE"]], on="inst_key", how="left")
    xw["route"] = np.where(xw.UNITID.notna(), "A1_name", "")
    xw["alias_note"] = ""
    # Route A2: explicit aliases
    for wname, (inst, st, note) in ALIASES.items():
        hit = sc[(sc.INSTNM == inst) & (sc.STABBR == st)]
        assert len(hit) == 1, (wname, inst, len(hit))
        i = xw.index[xw.wapman_name == wname]
        assert len(i) == 1, wname
        assert xw.loc[i[0], "route"] == "", f"alias for an already-matched name: {wname}"
        h = hit.iloc[0]
        for c in ["UNITID", "op6", "INSTNM", "STABBR", "main", "n_ug_units_op6", "SAT_AVG", "ADM_RATE"]:
            xw.loc[i[0], c] = h[c]
        xw.loc[i[0], "route"] = "A2_alias"
        xw.loc[i[0], "alias_note"] = note
    xw["in_scorecard"] = xw.route != ""
    # OPEID6 -> super_opeid
    t11m = t11[["opeid", "super_opeid", "institution_name", "multi"]].rename(
        columns={"opeid": "op6", "institution_name": "oi_inst_name", "multi": "t11_multi"})
    xw = xw.merge(t11m, on="op6", how="left")
    # Route B: Wapman name == OI Table 11 institution name (2013 DoE names) or Table 2 college name
    nm = pd.concat([t11[["super_opeid", "institution_name"]].rename(columns={"institution_name": "nm"}),
                    t2[["super_opeid", "name"]].rename(columns={"name": "nm"})])
    nm = nm[nm.nm.notna()].copy()
    nm["k"] = nm.nm.astype(str).map(norm)
    amb = nm.groupby("k").super_opeid.nunique()
    nm = nm[nm.k.map(amb) == 1].drop_duplicates("k")
    kB = dict(zip(nm.k, nm.super_opeid))
    xw["super_B"] = xw.inst_key.map(kB)
    xw["routeB_used"] = xw.super_opeid.isna() & xw.super_B.notna()
    xw.loc[xw.routeB_used, "super_opeid"] = xw.loc[xw.routeB_used, "super_B"]
    xw.loc[xw.routeB_used, "route"] = xw.loc[xw.routeB_used, "route"].map(
        lambda r: (r + "+" if r else "") + "B_oi_name")
    both = xw.super_B.notna() & ~xw.routeB_used & xw.super_opeid.notna()
    xw["route_agree"] = np.where(both, (xw.super_B == xw.super_opeid), np.nan)
    # OI Table 2 status
    t2i = t2.set_index("super_opeid")
    xw["oi_name"] = xw.super_opeid.map(t2i["name"])
    xw["multi"] = xw.super_opeid.map(t2i["multi"])
    xw["name_jacc"] = [jacc(k, norm(n)) if isinstance(n, str) else np.nan
                       for k, n in zip(xw.inst_key, xw.oi_inst_name.fillna(xw.oi_name))]
    has_out = xw.super_opeid.map(t2i["k_top1pc"]).notna() & xw.super_opeid.map(t2i["k_median"]).notna()
    status = np.select(
        [xw.super_opeid.isna() & ~xw.in_scorecard,
         xw.super_opeid.isna() & xw.in_scorecard,
         xw.super_opeid <= 0,
         xw.multi == 1,
         ~has_out],
        ["no Scorecard record, no OI name match",
         "Scorecard OPEID6 not in OI Table 11",
         "OI: insufficient data (super_opeid = -1)",
         "OI system group (multi = 1)",
         "no OI Table 2 row"], default="ok")
    xw["status"] = status
    # several Wapman units on one single-college super_opeid -> keep the main campus
    ok = xw.status == "ok"
    size = xw[ok].groupby("super_opeid").wapman_name.transform("size")
    xw["dup_group"] = False
    xw.loc[size.index[size > 1], "dup_group"] = True
    for so, g in xw[ok & xw.dup_group].groupby("super_opeid"):
        mains = g[g.main == 1]
        keep = mains.g_rank.idxmin() if len(mains) >= 1 else None
        for i in g.index:
            if i != keep:
                xw.loc[i, "status"] = "duplicate on super_opeid (non-main unit dropped)"
    xw["branch_shared"] = xw.n_ug_units_op6.fillna(0) > 1
    xw["G"] = -xw.g_rank.astype(float)
    return xw.sort_values("g_rank").reset_index(drop=True)


def load_oi(xw, statuses=("ok",)):
    t2 = pd.read_csv(OI / "mrc_table2.csv")
    t10 = pd.read_csv(OI / "mrc_table10.csv")
    keep10 = ["super_opeid", "barrons", "sat_avg_2001", "sat_avg_2013", "scorecard_rej_rate_2013",
              "pct_arthuman_2000", "pct_business_2000", "pct_health_2000", "pct_multidisci_2000",
              "pct_publicsocial_2000", "pct_stem_2000", "pct_socialscience_2000", "pct_tradepersonal_2000"]
    base = xw[xw.status.isin(statuses)].sort_values("g_rank").drop_duplicates("super_opeid")
    d = base.drop(columns=["multi"]).merge(t2, on="super_opeid", how="left") \
          .merge(t10[keep10], on="super_opeid", how="left")
    d["SAT"] = d.sat_avg_2001.fillna(d.sat_avg_2013)
    d["SAT_src"] = np.where(d.sat_avg_2001.notna(), "2001", np.where(d.sat_avg_2013.notna(), "2013", ""))
    d["REJ"] = d.scorecard_rej_rate_2013
    d["BARR"] = d.barrons.where(d.barrons.isin([1, 2, 3, 4, 5]), 6)       # 9 special / 999 non-sel -> 6
    d["PUBLIC"] = (d["type"] == 1).astype(int)
    d["SET"] = d.pct_health_2000 + d.pct_publicsocial_2000
    d["INT"] = d.pct_stem_2000 + d.pct_business_2000
    d["SC_SAT"] = d.SAT_AVG
    d["SC_REJ"] = 1 - d.ADM_RATE
    sh = ipeds_shares()
    d = d.merge(sh, left_on="UNITID", right_index=True, how="left")
    return d.sort_values("g_rank").reset_index(drop=True)


def ipeds_shares():
    """IPEDS C2023_a first-major bachelor's shares by CIP-2 per UNITID + the two 2023 mixes."""
    c = pd.read_csv(IPEDS_C, usecols=["UNITID", "CIPCODE", "MAJORNUM", "AWLEVEL", "CTOTALT"],
                    dtype={"UNITID": str, "CIPCODE": str}, encoding="utf-8-sig")
    c = c[(c.AWLEVEL == 5) & (c.MAJORNUM == 1) & (c.CIPCODE != "99")].copy()
    c["cip2"] = c.CIPCODE.str.slice(0, 2)
    tot = c.groupby("UNITID").CTOTALT.sum()
    sh = c.groupby(["UNITID", "cip2"]).CTOTALT.sum().unstack(fill_value=0).div(tot, axis=0)
    sh = sh[tot.reindex(sh.index) > 0]
    out = pd.DataFrame(index=sh.index)
    out["SET23"] = 100 * (sh.get("13", 0) + sh.get("51", 0))                  # education + health
    out["INT23"] = 100 * sum(sh.get(k, 0) for k in ["11", "14", "27", "52"])  # CS, eng., math/stat, business
    for k in sh.columns:
        out["cip_" + k] = sh[k]
    return out


# ---------------------------------------------------------------------------------------------
# rank statistics
# ---------------------------------------------------------------------------------------------
def rk(v):
    return rankdata(v)


def design(D, idx, conts, cats, extra=None):
    """Intercept + ranks of continuous covariates + dummies of categorical covariates on rows idx."""
    n = len(idx)
    cols = [np.ones((n, 1))]
    for c in conts:
        cols.append(rk(D[c][idx])[:, None])
    for c in cats:
        _, inv = np.unique(D[c][idx], return_inverse=True)
        L = inv.max() + 1
        if L > 1:
            cols.append(np.eye(L)[inv][:, 1:])
    if extra is not None:
        cols.append(extra)
    return np.hstack(cols)


def resid(Z, V):
    coef, *_ = np.linalg.lstsq(Z, V, rcond=None)
    return V - Z @ coef


def corr_cols(E, i, j):
    a, b = E[:, i], E[:, j]
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 1e-12 else np.nan


def partials(D, idx, xs, ys, conts, cats):
    """{(x, y): partial rank corr of x, y given covariates} on rows idx (ranks within the rows)."""
    names = list(dict.fromkeys(list(xs) + list(ys)))
    V = np.column_stack([rk(D[v][idx]) for v in names])
    E = resid(design(D, idx, conts, cats), V)
    pos = {v: k for k, v in enumerate(names)}
    return {(x, y): corr_cols(E, pos[x], pos[y]) for x in xs for y in ys if x != y}


def zr(v):
    r = rk(v)
    return (r - r.mean()) / r.std()


def horse(D, idx, y, with_se=False):
    """z(rank y) ~ z(rank G) + z(rank SAT) + z(rank REJ) + Barron's dummies + z(rank par_rank) +
    z(rank par_top1pc). Standardized coefficients + Shapley R2 over {brand, selectivity, parents}."""
    yy = zr(D[y][idx])
    Xc = {v: zr(D[v][idx]) for v in ["G", "SAT", "REJ", "par_rank", "par_top1pc"]}
    _, inv = np.unique(D["BARR"][idx], return_inverse=True)
    bd = np.eye(inv.max() + 1)[inv][:, 1:] if inv.max() > 0 else np.zeros((len(idx), 0))
    blocks = {"brand": [Xc["G"]], "selectivity": [Xc["SAT"], Xc["REJ"], bd], "parents": [Xc["par_rank"], Xc["par_top1pc"]]}

    def X_of(bl):
        cols = [np.ones((len(idx), 1))]
        for b in bl:
            for c in blocks[b]:
                cols.append(c if c.ndim == 2 else c[:, None])
        return np.hstack(cols)

    def r2(bl):
        if not bl:
            return 0.0
        X = X_of(bl)
        e = resid(X, yy[:, None])[:, 0]
        return 1.0 - (e @ e) / (yy @ yy)

    Bn = list(blocks)
    cache = {tuple(sorted(s)): r2(list(s)) for k in range(len(Bn) + 1) for s in combinations(Bn, k)}
    out = {"r2": cache[tuple(sorted(Bn))]}
    for b in Bn:
        others = [o for o in Bn if o != b]
        shap = 0.0
        for k in range(len(others) + 1):
            for s in combinations(others, k):
                w = factorial(k) * factorial(len(Bn) - k - 1) / factorial(len(Bn))
                shap += w * (cache[tuple(sorted(s + (b,)))] - cache[tuple(sorted(s))])
        out[f"shap_{b}"] = shap
        out[f"uniq_{b}"] = cache[tuple(sorted(Bn))] - cache[tuple(sorted(others))]
    X = X_of(Bn)
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    out.update(b_G=coef[1], b_SAT=coef[2], b_REJ=coef[3], b_par_rank=coef[-2], b_par_top1pc=coef[-1])
    if with_se:
        fit = sm.OLS(yy, X).fit(cov_type="HC1")
        out.update(se_G=fit.bse[1], se_SAT=fit.bse[2], se_REJ=fit.bse[3])
    return out


def mix_model(D, idx, y, m, controls=True):
    """z(rank y) ~ zG + zM + zG*zM [+ selectivity + parents] (ranks); returns (interaction, G main)."""
    yy = zr(D[y][idx]); g = zr(D["G"][idx]); mm = zr(D[m][idx])
    Z = design(D, idx, SEL + PAR if controls else [], ["BARR"] if controls else [],
               extra=np.column_stack([g, mm, g * mm]))
    coef, *_ = np.linalg.lstsq(Z, yy, rcond=None)
    return float(coef[-1]), float(coef[-3])


def spear(x, y):
    return partials({"x": x, "y": y}, np.arange(len(x)), ["x"], ["y"], [], [])[("x", "y")]


# ---------------------------------------------------------------------------------------------
# statistic bundle for one row set (point estimate or bootstrap draw)
# ---------------------------------------------------------------------------------------------
MIXES = ["SET", "INT", "SET23", "INT23"]
REL_PQ = ["ktop1pc_cond_parq1", "kq5_cond_parq1", "k_rank_cond_parq1", "ktop1pc_cond_parq5", "k_rank_cond_parq5"]
MIX_DESC = {"SET": "OI 2000: health + public/social services", "INT": "OI 2000: STEM + business",
            "SET23": "IPEDS 2023: education (CIP 13) + health (CIP 51)",
            "INT23": "IPEDS 2023: computer science + engineering + math/statistics + business (CIP 11, 14, 27, 52)"}
PQ_OUT = [f"ktop1pc_cond_parq{q}" for q in range(1, 6)] + [f"kq5_cond_parq{q}" for q in range(1, 6)] + \
         [f"k_rank_cond_parq{q}" for q in range(1, 6)]


def stats_main(D, idx, fields=None, full=True):
    s = {}
    # (a)-(c): brand and selectivity partials under each control set
    for cs, (conts, cats) in CS.items():
        p = partials(D, idx, ["G"], OUT, conts, cats)
        for y in OUT:
            s[f"G|{cs}|{y}"] = p[("G", y)]
    for x in ["SAT", "REJ"]:
        for cs, conts, cats in [("none", [], []), ("par", PAR, []), ("G_par", ["G"] + PAR, [])]:
            p = partials(D, idx, [x], OUT, conts, cats)
            for y in OUT:
                s[f"{x}|{cs}|{y}"] = p[(x, y)]
    # tail beyond the median
    for cs, conts, cats in [("med", ["k_median"], []), ("med_sel_par", ["k_median"] + SEL + PAR, ["BARR"])]:
        p = partials(D, idx, ["G"], TAILS, conts, cats)
        for t in TAILS:
            s[f"G|{cs}|{t}"] = p[("G", t)]
    p = partials(D, idx, ["SAT"], TAILS, ["k_median", "G"] + PAR, [])
    for t in TAILS:
        s[f"SAT|med_G_par|{t}"] = p[("SAT", t)]
    if not full:
        return s
    # (b) parent-quintile-conditional outcomes
    p0 = partials(D, idx, ["G"], PQ_OUT, [], [])
    p1 = partials(D, idx, ["G"], PQ_OUT, SEL, ["BARR"])
    for y in PQ_OUT:
        s[f"G|none|{y}"] = p0[("G", y)]
        s[f"G|sel|{y}"] = p1[("G", y)]
    # (c) horse race
    for y in OUT:
        h = horse(D, idx, y)
        for k, v in h.items():
            s[f"HR|{k}|{y}"] = v
    # (d) field mix (ecological)
    for m in MIXES:
        im = idx[np.isfinite(D[m][idx])]                  # IPEDS-2023 mixes: a few colleges missing
        for y in ["k_median", "k_top10pc", "k_top5pc", "k_top1pc"]:
            b3, bg = mix_model(D, im, y, m)
            s[f"MIX|{m}_x_G|{y}"] = b3
            s[f"MIX|{m}_G_main|{y}"] = bg
            s[f"MIX|{m}_x_G_raw|{y}"] = mix_model(D, im, y, m, controls=False)[0]
        v = D[m][im]
        q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
        lo, hi = im[v <= q1], im[v > q2]
        for y in ["k_median", "k_top1pc"]:
            s[f"MIX|{m}_T1|{y}"] = spear(D["G"][lo], D[y][lo])
            s[f"MIX|{m}_T3|{y}"] = spear(D["G"][hi], D[y][hi])
    # (e) field prestige vs brand, institution-wide outcomes (ecological)
    if fields:
        for f, mask in fields.items():
            fi = idx[mask[idx]]
            if len(fi) < NMIN_FIELD // 2:
                continue
            p = partials(D | {"F": D["F_" + f]}, fi, ["F", "G"], ["k_median", "k_top1pc"], [], [])
            q = partials(D | {"F": D["F_" + f]}, fi, ["F"], ["k_median", "k_top1pc"], ["G"], [])
            for y in ["k_median", "k_top1pc"]:
                s[f"FLD|{f}|cF|{y}"] = p[("F", y)]
                s[f"FLD|{f}|cG|{y}"] = p[("G", y)]
                s[f"FLD|{f}|F_G|{y}"] = q[("F", y)]
    return s


def add_diffs(s):
    """Paired differences tail - median for every predictor|spec family."""
    out = dict(s)
    for k, v in s.items():
        parts = k.split("|")
        if len(parts) == 3 and parts[2] in TAILS + ["k_q5"] and parts[0] in ("G", "SAT", "REJ") \
                and f"{parts[0]}|{parts[1]}|k_median" in s:
            out[f"{parts[0]}|{parts[1]}|D_{parts[2]}"] = v - s[f"{parts[0]}|{parts[1]}|k_median"]
        if len(parts) == 3 and parts[2] == "k_top1pc" and parts[0] in ("G", "SAT") \
                and f"{parts[0]}|{parts[1]}|k_rank" in s:
            out[f"{parts[0]}|{parts[1]}|R_k_top1pc"] = v - s[f"{parts[0]}|{parts[1]}|k_rank"]
        if len(parts) == 3 and parts[0] in ("HR", "MIX") and parts[2] in TAILS and \
                f"{parts[0]}|{parts[1]}|k_median" in s:
            out[f"{parts[0]}|{parts[1]}|D_{parts[2]}"] = v - s[f"{parts[0]}|{parts[1]}|k_median"]
    # within parent quintile: top-1% share vs mean rank, and top-20% vs mean rank
    for q in range(1, 6):
        for cs in ["none", "sel"]:
            a, b, c = (s.get(f"G|{cs}|ktop1pc_cond_parq{q}"), s.get(f"G|{cs}|kq5_cond_parq{q}"),
                       s.get(f"G|{cs}|k_rank_cond_parq{q}"))
            if a is not None:
                out[f"G|{cs}|D_top1_rank_parq{q}"] = a - c
                out[f"G|{cs}|D_q5_rank_parq{q}"] = b - c
    # tercile contrasts (ecological mix)
    for m in MIXES:
        for y in ["k_median", "k_top1pc"]:
            a, b = s.get(f"MIX|{m}_T3|{y}"), s.get(f"MIX|{m}_T1|{y}")
            if a is not None:
                out[f"MIX|{m}_T3-T1|{y}"] = a - b
    # field aggregates
    fk = sorted({k.split("|")[1] for k in s if k.startswith("FLD|")})
    if fk:
        for stat in ["cF", "cG", "F_G"]:
            for y in ["k_median", "k_top1pc"]:
                vals = np.array([s[f"FLD|{f}|{stat}|{y}"] for f in fk])
                out[f"FLDMEAN|{stat}|{y}"] = float(np.nanmean(vals))
        for y in ["k_median", "k_top1pc"]:
            out[f"FLDMEAN|cF-cG|{y}"] = out[f"FLDMEAN|cF|{y}"] - out[f"FLDMEAN|cG|{y}"]
        out["FLDMEAN|F_G|D_k_top1pc"] = out["FLDMEAN|F_G|k_top1pc"] - out["FLDMEAN|F_G|k_median"]
    return out


def bootstrap(D, n, nb, seed, **kw):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(nb, n))
    rows = [add_diffs(stats_main(D, np.sort(draws[b]), **kw)) for b in range(nb)]
    return pd.DataFrame(rows)


def to_arrays(d: pd.DataFrame, extra_fields=None) -> dict:
    D = {c: d[c].to_numpy() for c in d.columns if d[c].dtype != object}
    for c in ["BARR", "PUBLIC", "region"]:
        D[c] = d[c].to_numpy()
    if extra_fields:
        D.update(extra_fields)
    return D


def summarise(point: dict, boot: pd.DataFrame, sample: str, n: int, section_of) -> list:
    rows = []
    for k, v in point.items():
        bv = boot[k].to_numpy(float) if k in boot else np.array([np.nan])
        bv = bv[np.isfinite(bv)]
        lo, hi = (np.percentile(bv, [2.5, 97.5]) if len(bv) > 10 else (np.nan, np.nan))
        ple0 = float(np.mean(bv <= 0)) if len(bv) > 10 else np.nan
        parts = k.split("|")
        rows.append(dict(section=section_of(k), sample=sample, predictor=parts[0], spec=parts[1],
                         outcome="|".join(parts[2:]), n=n, est=v, lo=lo, hi=hi, boot_share_le0=ple0))
    return rows


def section_of(k):
    p = k.split("|")
    if p[0] == "HR":
        return "c_horse_race"
    if p[0] == "MIX":
        return "d_field_mix_ecological"
    if p[0] in ("FLD", "FLDMEAN"):
        return "e_field_prestige_ecological"
    if p[1] in ("med", "med_sel_par", "med_G_par"):
        return "c_tail_beyond_median"
    if "cond_parq" in k or "_parq" in k:
        return "b_parent_quintile_conditional"
    if p[1] in ("none",) and p[0] == "G":
        return "a_coupling"
    if p[1] == "par":
        return "b_net_of_parents"
    if p[1] in ("sel", "sel_par", "strict", "G_par"):
        return "c_brand_vs_selectivity"
    return "a_coupling"


# ---------------------------------------------------------------------------------------------
# (e) calibration: can field-vs-brand statistics on a pooled (all-field) outcome discriminate?
# ---------------------------------------------------------------------------------------------
def _pearson_cols(a, Bm):
    """Pearson correlation of vector a (n,) with every column of Bm (n, R)."""
    a = a - a.mean()
    Bm = Bm - Bm.mean(0)
    return (a @ Bm) / np.sqrt((a @ a) * (Bm * Bm).sum(0))


def field_stats_vec(G, farr, fields, Y):
    """(e) statistics for every column of Y (n, R) at once: means over fields of c_F = rho(F, y),
    c_G = rho(G, y), c_F - c_G, partial rho(F, y | G), and the number of fields with a positive partial.
    Same estimator as partials() (correlation of rank residuals on [1, rank G] = first-order partial
    correlation of the ranks)."""
    cF, cG, pr = [], [], []
    for f, mask in fields.items():
        rf = rankdata(farr["F_" + f][mask]); rg = rankdata(G[mask]); ry = rankdata(Y[mask], axis=0)
        a, b = _pearson_cols(rf, ry), _pearson_cols(rg, ry)
        c = _pearson_cols(rf, rg[:, None])[0]
        cF.append(a); cG.append(b); pr.append((a - c * b) / np.sqrt((1 - c * c) * (1 - b * b)))
    cF, cG, pr = np.array(cF), np.array(cG), np.array(pr)          # (fields, R)
    return {"cF": cF.mean(0), "cG": cG.mean(0), "cF-cG": (cF - cG).mean(0), "F_G": pr.mean(0),
            "npos": (pr > 0).sum(0).astype(float)}


def field_calibration(C, farr, fields):
    """Two data-generating scenarios on the real sample-C field prestiges:
      field-only: y = z(institution-average of within-field z(rank F) over the (e) fields) + sigma * eps
      brand-only: y = z(rank G) + sigma * eps
    eps iid N(0, 1), SIM_R draws; sigma set by bisection (first SIM_CAL_R draws) so that the mean over
    fields and draws of c_G equals the observed c_G for that outcome. Institutions are fixed, so the
    ranges across draws reflect outcome noise only."""
    G = C.G.to_numpy(float)
    n = len(C)
    Z = np.full((n, len(fields)), np.nan)
    for j, (f, mask) in enumerate(fields.items()):
        Z[mask, j] = zr(farr["F_" + f][mask])
    has = np.isfinite(Z).any(1)
    fbar = np.zeros(n)
    fbar[has] = np.nanmean(Z[has], axis=1)
    fbar[has] = (fbar[has] - fbar[has].mean()) / fbar[has].std()   # institutions with no field rank never
    signals = {"field_only": fbar, "brand_only": zr(G)}               # enter a field's statistics
    rho_g_fbar = spear(G[has], fbar[has])
    E = np.random.default_rng(SIM_SEED).standard_normal((n, SIM_R))
    obs, out = {}, {}
    for y in ["k_median", "k_top1pc"]:
        obs[y] = {k: float(v[0]) for k, v in field_stats_vec(G, farr, fields, C[y].to_numpy(float)[:, None]).items()}
        for sc, x in signals.items():
            lo, hi = 0.0, 5.0
            for _ in range(30):
                mid = (lo + hi) / 2
                if field_stats_vec(G, farr, fields, x[:, None] + mid * E[:, :SIM_CAL_R])["cG"].mean() > obs[y]["cG"]:
                    lo = mid
                else:
                    hi = mid
            sig = (lo + hi) / 2
            out[(sc, y)] = dict(sigma=sig, **field_stats_vec(G, farr, fields, x[:, None] + sig * E))
    rows = []
    for (sc, y), st in out.items():
        for k, v in st.items():
            if k == "sigma":
                rows.append(dict(section="e_field_prestige_calibration_sim", sample=f"C_common_sim_{sc}",
                                 predictor="SIM", spec="sigma", outcome=y, n=n, est=v, lo=np.nan, hi=np.nan,
                                 boot_share_le0=np.nan))
                continue
            lo_, hi_ = np.percentile(v, [2.5, 97.5])
            rows.append(dict(section="e_field_prestige_calibration_sim", sample=f"C_common_sim_{sc}",
                             predictor="SIM", spec=k, outcome=y, n=n, est=float(v.mean()), lo=lo_, hi=hi_,
                             boot_share_le0=float(np.mean(v <= 0))))
            rows.append(dict(section="e_field_prestige_calibration_sim", sample=f"C_common_sim_{sc}",
                             predictor="SIM", spec=k + "|share_le_observed", outcome=y, n=n,
                             est=float(np.mean(v <= obs[y][k])), lo=np.nan, hi=np.nan, boot_share_le0=np.nan))
    rows.append(dict(section="e_field_prestige_calibration_sim", sample="C_common", predictor="G",
                     spec="institution_avg_field_prestige", outcome="spearman", n=int(has.sum()), est=rho_g_fbar,
                     lo=np.nan, hi=np.nan, boot_share_le0=np.nan))
    summ = {"rho_g_fbar": rho_g_fbar, "n_fbar": int(has.sum()), "obs": obs,
            "sim": {key: {k: (float(np.mean(v)), *np.percentile(v, [2.5, 97.5]), float(np.mean(v <= obs[key[1]][k])))
                          if k != "sigma" else v for k, v in st.items()} for key, st in out.items()}}
    return pd.DataFrame(rows), summ


# ---------------------------------------------------------------------------------------------
# (f) Table 3: age profile + reliability
# ---------------------------------------------------------------------------------------------
def table3_profile(d, rng_seed):
    t3 = pd.read_csv(OI / "mrc_table3.csv",
                     usecols=["super_opeid", "cohort", "k_median", "k_top1pc", "k_top5pc", "k_top10pc",
                              "k_q5", "k_rank", "k_mean", "par_rank", "count"] + REL_PQ)
    t3 = t3[t3.super_opeid.isin(d.super_opeid)]
    g = d.set_index("super_opeid").G
    rows = []
    rng = np.random.default_rng(rng_seed)
    for c, x in t3.groupby("cohort"):
        x = x.dropna(subset=["k_median", "k_top1pc", "par_rank"]).sort_values("super_opeid")
        n = len(x)
        if n < 30:
            continue
        D = {"G": x.super_opeid.map(g).to_numpy(float), "k_median": x.k_median.to_numpy(float),
             "k_top1pc": x.k_top1pc.to_numpy(float), "par_rank": x.par_rank.to_numpy(float)}

        def st(idx):
            a = partials(D, idx, ["G"], ["k_median", "k_top1pc"], [], [])
            b = partials(D, idx, ["G"], ["k_median", "k_top1pc"], ["par_rank"], [])
            return {"raw_med": a[("G", "k_median")], "raw_top1": a[("G", "k_top1pc")],
                    "par_med": b[("G", "k_median")], "par_top1": b[("G", "k_top1pc")]}
        pt = st(np.arange(n))
        draws = rng.integers(0, n, size=(B_ROB, n))
        bs = pd.DataFrame([st(np.sort(dr)) for dr in draws])
        for k, v in pt.items():
            lo, hi = np.percentile(bs[k], [2.5, 97.5])
            rows.append(dict(cohort=int(c), age_2014=2014 - int(c), n=n, stat=k, est=v, lo=lo, hi=hi))
    prof = pd.DataFrame(rows)
    # cross-cohort reliability (1980, 1981, 1982) on the analysis institutions
    rel = {}
    w = t3[t3.cohort.isin([1980, 1981, 1982])]
    for y in OUT + REL_PQ:
        pv = w.pivot(index="super_opeid", columns="cohort", values=y).dropna()
        rs = [spear(pv[a].to_numpy(float), pv[b].to_numpy(float)) for a, b in [(1980, 1981), (1980, 1982), (1981, 1982)]]
        rbar = float(np.mean(rs))
        rel[y] = dict(n=len(pv), r_pair=rbar, R3=3 * rbar / (1 + 2 * rbar))
    return prof, rel


# ---------------------------------------------------------------------------------------------
# label check for the OI major groups (IPEDS 2023 bachelor's shares by CIP-2)
# ---------------------------------------------------------------------------------------------
OI_GROUPS = ["pct_arthuman_2000", "pct_business_2000", "pct_health_2000", "pct_multidisci_2000",
             "pct_publicsocial_2000", "pct_stem_2000", "pct_socialscience_2000", "pct_tradepersonal_2000"]
CIP_NAMES = {"13": "education", "51": "health", "43": "security/protective services",
             "44": "public admin/social service", "52": "business", "14": "engineering",
             "11": "computer science", "27": "math/statistics", "45": "social sciences", "42": "psychology"}


def major_label_check(d):
    """Which OI 2000 major group tracks each 2023 CIP-2 family (Spearman across the same colleges)."""
    x = d.dropna(subset=OI_GROUPS + ["cip_13"])
    out = {"n": len(x), "rows": []}
    for cp, name in CIP_NAMES.items():
        if "cip_" + cp not in x:
            continue
        r = {g_: spear(x[g_].to_numpy(float), x["cip_" + cp].to_numpy(float)) for g_ in OI_GROUPS}
        top = sorted(r.items(), key=lambda kv: -kv[1])[:2]
        out["rows"].append((cp, name, top))
    return out


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------
def main():
    xw = build_crosswalk()
    d_all = load_oi(xw)
    need = ["G", "k_median", "k_top1pc", "SAT", "REJ", "BARR"] + PAR_PLUS + ["SET", "INT"] + OUT + PQ_OUT
    A = d_all.dropna(subset=["G"] + OUT).reset_index(drop=True)
    C = d_all.dropna(subset=need).reset_index(drop=True)
    C = C[C.region.notna()].reset_index(drop=True)

    # field prestige (ecological, supplementary): F = -field rank on sample C
    ar = load_ar_wapman(fields=FIELDS66)
    fmasks, farr = {}, {}
    for f, g in ar.groupby("field"):
        fmap = dict(zip(g.inst_key, g.prestige_score))
        v = C.inst_key.map(fmap).to_numpy(float)
        if np.isfinite(v).sum() >= NMIN_FIELD:
            fmasks[f] = np.isfinite(v)
            farr["F_" + f] = v
    fields = dict(sorted(fmasks.items()))

    DC = to_arrays(C, farr)
    nC = len(C)
    point = add_diffs(stats_main(DC, np.arange(nC), fields=fields))
    # HC1 SEs for the horse race at the point estimate
    hr_se = {y: horse(DC, np.arange(nC), y, with_se=True) for y in OUT}
    boot = bootstrap(DC, nC, B, SEED, fields=fields)
    rows = summarise(point, boot, "C_common", nC, section_of)
    res = pd.DataFrame(rows)
    mix_n = {m: int(np.isfinite(C[m].to_numpy(float)).sum()) for m in MIXES}
    for m in MIXES:                                   # IPEDS-2023 mixes miss a few colleges
        res.loc[(res.predictor == "MIX") & res.spec.str.match(m + "_"), "n"] = mix_n[m]
    for f, m in fields.items():                       # per-field n for the (e) rows
        res.loc[(res.predictor == "FLD") & (res.spec == f), "n"] = int(m.sum())

    # sample A (all single-college units with outcomes): raw coupling and | parents
    DA = to_arrays(A)
    nA = len(A)

    def stats_A(D, idx):
        s = {}
        p = partials(D, idx, ["G"], OUT, [], [])
        q = partials(D, idx, ["G"], OUT, PAR, [])
        for y in OUT:
            s[f"G|none|{y}"] = p[("G", y)]
            s[f"G|par|{y}"] = q[("G", y)]
        return add_diffs(s)
    ptA = stats_A(DA, np.arange(nA))
    rngA = np.random.default_rng(SEED + 1)
    bA = pd.DataFrame([stats_A(DA, np.sort(dr)) for dr in rngA.integers(0, nA, size=(B, nA))])
    res = pd.concat([res, pd.DataFrame(summarise(ptA, bA, "A_all_outcomes", nA, section_of))])

    # robustness samples (headline statistics only)
    d_sys = load_oi(xw, statuses=("ok", "OI system group (multi = 1)"))
    C_sys = d_sys.dropna(subset=need).reset_index(drop=True)
    C_sys = C_sys[C_sys.region.notna()]
    rob_specs = {
        "R1_SAT2001_only": C[C.SAT_src == "2001"],
        "R2_no_branch_shared_OPEID6": C[~C.branch_shared.astype(bool)],
        "R3_scorecard_current_SAT_ADM": C.assign(SAT=C.SC_SAT, REJ=C.SC_REJ).dropna(subset=["SAT", "REJ"]),
        "R4_private_nonprofit_only": C[C.PUBLIC == 0],
        "R5_public_only": C[C.PUBLIC == 1],
        "R6_plus_OI_system_groups": C_sys,
        "R7_Barrons_tier_1_2_only": C[C.BARR.isin([1, 2])],
    }
    rob_n = {}
    for k, (name, sub) in enumerate(rob_specs.items()):
        sub = sub.reset_index(drop=True)
        rob_n[name] = len(sub)
        Dr = to_arrays(sub)
        pt = add_diffs(stats_main(Dr, np.arange(len(sub)), full=False))
        br = bootstrap(Dr, len(sub), B_ROB, SEED + 10 + k, full=False)
        res = pd.concat([res, pd.DataFrame(summarise(pt, br, name, len(sub), section_of))])

    prof, rel = table3_profile(A, SEED + 100)
    labchk = major_label_check(C)
    cal_rows, cal = field_calibration(C, farr, fields)
    for y in ["k_median", "k_top1pc"]:                # the calibration's observed values = the (e) point estimates
        for k in ["cF", "cG", "cF-cG", "F_G"]:
            assert abs(cal["obs"][y][k] - point[f"FLDMEAN|{k}|{y}"]) < 1e-9, (y, k)

    # ---- write the statistics table
    prof_rows = prof.assign(section="f_age_profile_table3", sample="A_all_outcomes_table3",
                            predictor="G", spec=prof.stat, outcome="age_" + prof.age_2014.astype(str),
                            boot_share_le0=np.nan)[["section", "sample", "predictor", "spec", "outcome", "n",
                                                   "est", "lo", "hi", "boot_share_le0"]]
    rel_rows = pd.DataFrame([dict(section="f_reliability_table3", sample="A_all_outcomes_table3",
                                  predictor="cross_cohort", spec="spearman_brown_3", outcome=y, n=v["n"],
                                  est=v["R3"], lo=np.nan, hi=np.nan, boot_share_le0=np.nan)
                             for y, v in rel.items()])
    res = pd.concat([res, prof_rows, rel_rows, cal_rows], ignore_index=True)
    res = res.sort_values(["section", "sample", "predictor", "spec", "outcome"], kind="mergesort")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False, float_format="%.6f")

    make_figure(res, C, prof)
    write_md(xw, d_all, A, C, res, point, hr_se, rel, prof, rob_n, fields, labchk, mix_n, cal)
    print(f"wrote {OUT_CSV}, {OUT_FIG}, {OUT_MD}")


# ---------------------------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------------------------
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed categorical order (slots 1-4)
MK = ["o", "s", "^", "D"]


def get(res, sample, pred, spec, outcome):
    r = res[(res["sample"] == sample) & (res.predictor == pred) & (res.spec == spec) & (res.outcome == outcome)]
    assert len(r) == 1, (sample, pred, spec, outcome, len(r))
    r = r.iloc[0]
    return float(r.est), float(r.lo), float(r.hi)


def make_figure(res, C, prof):
    ladder = ["k_median", "k_rank", "k_q5", "k_top10pc", "k_top5pc", "k_top1pc"]
    xs = np.arange(len(ladder))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    plt.rcParams.update({"font.size": 9})

    def ladder_panel(ax, pred, specs, title, leg):
        for j, (spec, lab) in enumerate(specs):
            e = np.array([get(res, "C_common", pred, spec, y) for y in ladder])
            off = (j - (len(specs) - 1) / 2) * 0.12
            ax.errorbar(xs + off, e[:, 0], yerr=[e[:, 0] - e[:, 1], e[:, 2] - e[:, 0]], color=COL[j],
                        marker=MK[j], ms=6, lw=2, capsize=0, elinewidth=1.2, label=lab)
        ax.axhline(0, color="#8a8983", lw=0.8)
        ax.set_xticks(xs); ax.set_xticklabels([OSHORT[y] for y in ladder])
        ax.set_ylabel("Spearman / partial rank correlation")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8, **leg)

    ladder_panel(axes[0, 0], "G", [("none", "raw"), ("par", "| parents"), ("sel", "| selectivity"),
                                   ("sel_par", "| selectivity + parents")],
                 f"A. Academia-wide brand G vs college outcomes (n = {len(C)})",
                 dict(loc="upper center", ncol=2))
    axes[0, 0].set_ylim(top=0.95)
    ladder_panel(axes[0, 1], "SAT", [("none", "raw"), ("par", "| parents"), ("G_par", "| brand + parents")],
                 "B. Selectivity (SAT) vs college outcomes, same colleges", dict(loc="lower left"))
    # C: tail beyond the median
    ax = axes[1, 0]
    specs = [("med", "| median"), ("med_sel_par", "| median + selectivity + parents")]
    for j, (spec, lab) in enumerate(specs):
        e = np.array([get(res, "C_common", "G", spec, t) for t in TAILS])
        off = (j - 0.5) * 0.14
        ax.errorbar(np.arange(3) + off, e[:, 0], yerr=[e[:, 0] - e[:, 1], e[:, 2] - e[:, 0]], color=COL[j],
                    marker=MK[j], ms=6, lw=0, elinewidth=1.4, label="G " + lab)
    e = np.array([get(res, "C_common", "SAT", "med_G_par", t) for t in TAILS])
    ax.errorbar(np.arange(3) + 0.3, e[:, 0], yerr=[e[:, 0] - e[:, 1], e[:, 2] - e[:, 0]], color=COL[2],
                marker=MK[2], ms=6, lw=0, elinewidth=1.4, label="SAT | median + brand + parents")
    ax.axhline(0, color="#8a8983", lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([OSHORT[t] for t in TAILS])
    ax.set_ylabel("partial rank correlation with tail share")
    ax.set_title("C. Tail beyond the median: colleges with the same median", loc="left", fontsize=10)
    ax.grid(axis="y", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(top=0.9)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    # D: age profile (Table 3)
    ax = axes[1, 1]
    for j, (st, lab) in enumerate([("raw_med", "median"), ("raw_top1", "top 1% share"),
                                   ("par_med", "median | par_rank"), ("par_top1", "top 1% | par_rank")]):
        p = prof[prof.stat == st].sort_values("age_2014")
        ax.plot(p.age_2014, p.est, color=COL[j], marker=MK[j], ms=5, lw=2 if j < 2 else 1.2,
                ls="-" if j < 2 else "--", label=lab)
        if j < 2:
            ax.fill_between(p.age_2014, p.lo, p.hi, color=COL[j], alpha=0.12, lw=0)
    ax.axhline(0, color="#8a8983", lw=0.8)
    ax.set_xlabel("age in 2014 (birth cohort 1991 ... 1980; different cohorts, same year)")
    ax.set_ylabel("rank correlation with brand G")
    ax.set_title("D. Brand coupling by age at measurement (OI Table 3)", loc="left", fontsize=10)
    ax.grid(axis="y", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Opportunity Insights college outcomes (1980-82 cohorts, earnings in 2014) vs Wapman "
                 "academia-wide prestige; 95% bootstrap CIs", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, metadata={"Software": None})
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# write-up
# ---------------------------------------------------------------------------------------------
def fmt(t, d=2):
    e, lo, hi = t
    return f"{e:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def write_md(xw, d_all, A, C, res, point, hr_se, rel, prof, rob_n, fields, labchk, mix_n, cal):
    g = lambda *a: get(res, *a)
    S = "C_common"
    L = []
    w = L.append
    nC, nA = len(C), len(A)
    # key numbers
    gm, g1 = g(S, "G", "none", "k_median"), g(S, "G", "none", "k_top1pc")
    gd = g(S, "G", "none", "D_k_top1pc")
    gsp_m, gsp_1 = g(S, "G", "sel_par", "k_median"), g(S, "G", "sel_par", "k_top1pc")
    gsp_d = g(S, "G", "sel_par", "D_k_top1pc")
    gst_m, gst_1 = g(S, "G", "strict", "k_median"), g(S, "G", "strict", "k_top1pc")
    gst_d = g(S, "G", "strict", "D_k_top1pc")
    sat_m, sat_1 = g(S, "SAT", "none", "k_median"), g(S, "SAT", "none", "k_top1pc")
    satg_m, satg_1 = g(S, "SAT", "G_par", "k_median"), g(S, "SAT", "G_par", "k_top1pc")
    tbm = g(S, "G", "med", "k_top1pc"); tbm2 = g(S, "G", "med_sel_par", "k_top1pc")
    sat_tbm = g(S, "SAT", "med_G_par", "k_top1pc")
    w("# Elite-tail outcomes vs academic prestige: Opportunity Insights college tables (scripts/60)\n")
    w("*Descriptive, institution level, not causal. Public data only. Generated by "
      "`scripts/60_oi_tail.py`; every number below is computed by that script (statistics table: "
      "`data/interim/oi_tail.csv`; figure: `outputs/figures/oi_tail.png`).*\n")
    w("## Answer\n")
    w(ANSWER_TEMPLATE.format(
        nC=nC, gm=fmt(gm), g1=fmt(g1), gd=fmt(gd), gsp_m=fmt(gsp_m), gsp_1=fmt(gsp_1), gsp_d=fmt(gsp_d),
        gst_m=fmt(gst_m), gst_1=fmt(gst_1), gst_d=fmt(gst_d), sat_m=fmt(sat_m), sat_1=fmt(sat_1),
        satg_m=fmt(satg_m), satg_1=fmt(satg_1), tbm=fmt(tbm), tbm2=fmt(tbm2), sat_tbm=fmt(sat_tbm),
        **answer_extras(res, point, hr_se, rel, prof, rob_n, fields, C, cal)))
    # ---------------- numbers
    w("\n## Numbers\n")
    w("### Key numbers\n")
    w(key_table(res, rob_n))
    w(f"Sample C (\"common\", n = {nC}): Wapman institutions linked to a single-college OI unit "
      "(multi = 0) with Table 2 outcomes, SAT (2001, else 2013; OI Table 10), 2013 rejection rate, "
      "Barron's 2009 index and parental income. Estimates: Spearman or partial rank correlation "
      "(OLS residuals of within-sample ranks); brackets: 95% percentile CIs from a joint institution "
      f"bootstrap ({B} draws; the same draws for every column, so differences are paired). "
      "\"Tail − median\" is the paired difference of the two correlations.\n")
    w("### (a)-(c) Brand G and selectivity against the outcome ladder (sample C)\n")
    w("| outcome | G raw | G \\| parents | G \\| selectivity | G \\| selectivity + parents | G \\| strict | "
      "SAT raw | SAT \\| brand + parents | rejection rate \\| brand + parents |")
    w("|---|---|---|---|---|---|---|---|---|")
    for y in OUT:
        cells = [fmt(g(S, "G", cs, y)) for cs in ["none", "par", "sel", "sel_par", "strict"]]
        cells += [fmt(g(S, "SAT", "none", y)), fmt(g(S, "SAT", "G_par", y)), fmt(g(S, "REJ", "G_par", y))]
        w(f"| {OLAB[y]} | " + " | ".join(cells) + " |")
    for t in ["k_q5"] + TAILS:
        cells = [fmt(g(S, "G", cs, "D_" + t)) for cs in ["none", "par", "sel", "sel_par", "strict"]]
        cells += [fmt(g(S, "SAT", "none", "D_" + t)), fmt(g(S, "SAT", "G_par", "D_" + t)),
                  fmt(g(S, "REJ", "G_par", "D_" + t))]
        w(f"| *{OSHORT[t]} − median* | " + " | ".join(cells) + " |")
    w("\nControl sets: " + "; ".join(f"**{CS_LAB[k]}** = {v}" for k, v in CS_DESC.items() if k != "none")
      + ". SAT/rejection-rate columns: partial on G + par_rank + par_top1pc.\n")
    cc0 = lambda a, b: spear(C[a].to_numpy(float), C[b].to_numpy(float))
    w(f"Collinearity in sample C (Spearman): G–SAT {cc0('G', 'SAT'):+.2f}, G–rejection rate {cc0('G', 'REJ'):+.2f}, "
      f"SAT–rejection rate {cc0('SAT', 'REJ'):+.2f}, G–par_rank {cc0('G', 'par_rank'):+.2f}, "
      f"G–par_top1pc {cc0('G', 'par_top1pc'):+.2f}, SAT–par_top1pc {cc0('SAT', 'par_top1pc'):+.2f}.\n")
    w(f"Sample A (all {nA} single-college units with outcomes, no selectivity requirement): "
      f"G raw vs median {fmt(g('A_all_outcomes', 'G', 'none', 'k_median'))}, vs top 1% "
      f"{fmt(g('A_all_outcomes', 'G', 'none', 'k_top1pc'))}, difference "
      f"{fmt(g('A_all_outcomes', 'G', 'none', 'D_k_top1pc'))}; | parents: median "
      f"{fmt(g('A_all_outcomes', 'G', 'par', 'k_median'))}, top 1% {fmt(g('A_all_outcomes', 'G', 'par', 'k_top1pc'))}.\n")
    w("### (c) Tail beyond the median (sample C)\n")
    w("Partial rank correlation with the tail share among colleges with the same median earnings.\n")
    w("| tail share | G \\| median | G \\| median + selectivity + parents | SAT \\| median + brand + parents |")
    w("|---|---|---|---|")
    for t in TAILS:
        w(f"| {OLAB[t]} | {fmt(g(S, 'G', 'med', t))} | {fmt(g(S, 'G', 'med_sel_par', t))} | "
          f"{fmt(g(S, 'SAT', 'med_G_par', t))} |")
    w("\n### (c) Horse race: standardized rank OLS, sample C\n")
    w("z(rank y) ~ z(rank G) + z(rank SAT) + z(rank rejection rate) + Barron's dummies + z(rank par_rank) + "
      "z(rank par_top1pc). β with bootstrap 95% CI (HC1 SE at the point estimate in parentheses); "
      "Shapley R² shares over the blocks {brand}, {selectivity: SAT, rejection rate, Barron's}, "
      "{parents}; unique R² = drop in R² when the block is removed.\n")
    w("| outcome | β brand G | β SAT | β rejection rate | R² | Shapley brand | Shapley selectivity | "
      "Shapley parents | unique R² brand |")
    w("|---|---|---|---|---|---|---|---|---|")
    for y in OUT:
        h = hr_se[y]
        w(f"| {OLAB[y]} | {fmt(g(S, 'HR', 'b_G', y))} ({h['se_G']:.2f}) | {fmt(g(S, 'HR', 'b_SAT', y))} "
          f"({h['se_SAT']:.2f}) | {fmt(g(S, 'HR', 'b_REJ', y))} ({h['se_REJ']:.2f}) | "
          f"{g(S, 'HR', 'r2', y)[0]:.2f} | {fmt(g(S, 'HR', 'shap_brand', y))} | "
          f"{fmt(g(S, 'HR', 'shap_selectivity', y))} | {fmt(g(S, 'HR', 'shap_parents', y))} | "
          f"{fmt(g(S, 'HR', 'uniq_brand', y), 3)} |")
    for t in TAILS:
        w(f"| *{OSHORT[t]} − median* | {fmt(g(S, 'HR', 'b_G', 'D_' + t))} | {fmt(g(S, 'HR', 'b_SAT', 'D_' + t))} | "
          f"{fmt(g(S, 'HR', 'b_REJ', 'D_' + t))} | | {fmt(g(S, 'HR', 'shap_brand', 'D_' + t))} | "
          f"{fmt(g(S, 'HR', 'shap_selectivity', 'D_' + t))} | {fmt(g(S, 'HR', 'shap_parents', 'D_' + t))} | "
          f"{fmt(g(S, 'HR', 'uniq_brand', 'D_' + t), 3)} |")
    w("\n### (b) Parent-quintile-conditional outcomes (sample C)\n")
    w("Among children whose parents were in quintile q: share reaching the top 1%, share reaching the "
      "top 20%, and mean earnings rank. Spearman with G, raw and net of selectivity.\n")
    w("| parent quintile | top 1% raw | top 1% \\| selectivity | top 20% raw | top 20% \\| selectivity | "
      "mean rank raw | mean rank \\| selectivity |")
    w("|---|---|---|---|---|---|---|")
    for q in range(1, 6):
        w(f"| q{q} | " + " | ".join(fmt(g(S, "G", cs, f"{y}{q}")) for y in
                                   ["ktop1pc_cond_parq", "kq5_cond_parq", "k_rank_cond_parq"]
                                   for cs in ["none", "sel"]) + " |")
    w("\nPaired differences within parent quintile (Spearman with G of the top-1% share minus that of the "
      "mean rank; raw and net of selectivity):\n")
    w("| parent quintile | top 1% − mean rank, raw | top 1% − mean rank, \\| selectivity | "
      "top 20% − mean rank, raw | top 20% − mean rank, \\| selectivity |")
    w("|---|---|---|---|---|")
    for q in range(1, 6):
        w(f"| q{q} | {fmt(g(S, 'G', 'none', f'D_top1_rank_parq{q}'))} | {fmt(g(S, 'G', 'sel', f'D_top1_rank_parq{q}'))} | "
          f"{fmt(g(S, 'G', 'none', f'D_q5_rank_parq{q}'))} | {fmt(g(S, 'G', 'sel', f'D_q5_rank_parq{q}'))} |")
    w("\n### (d) Field mix — ECOLOGICAL (sample C)\n")
    w("Institution major mix from OI Table 10 (IPEDS 2000 shares). SET = health + public/social-service "
      "majors (the setting-priced mix); INT = STEM + business. Interaction: coefficient on "
      "z(rank G) × z(rank mix) in z(rank y) ~ zG + zMix + zG·zMix + selectivity + parents. "
      "Terciles: Spearman(G, y) within the bottom (T1) and top (T3) tercile of the mix share.\n")
    w("| mix (n) | outcome | G main effect | G × mix | G × mix, no controls | ρ(G, y) in T1 | ρ(G, y) in T3 | T3 − T1 |")
    w("|---|---|---|---|---|---|---|---|")
    for m in MIXES:
        for y in ["k_median", "k_top1pc"]:
            w(f"| {m} ({mix_n[m]}) | {OLAB[y]} | {fmt(g(S, 'MIX', m + '_G_main', y))} | {fmt(g(S, 'MIX', m + '_x_G', y))} | "
              f"{fmt(g(S, 'MIX', m + '_x_G_raw', y))} | {fmt(g(S, 'MIX', m + '_T1', y))} | {fmt(g(S, 'MIX', m + '_T3', y))} | "
              f"{fmt(g(S, 'MIX', m + '_T3-T1', y))} |")
        w(f"| {m} | *top 1% − median* | | {fmt(g(S, 'MIX', m + '_x_G', 'D_k_top1pc'))} | "
          f"{fmt(g(S, 'MIX', m + '_x_G_raw', 'D_k_top1pc'))} | | | |")
    w("\nMixes: " + "; ".join(f"**{k}** = {v}" for k, v in MIX_DESC.items()) + ".\n")
    q = C.SET.quantile([1 / 3, 2 / 3]).to_numpy()
    cc = lambda a, b: spear(C.dropna(subset=[a, b])[a].to_numpy(float), C.dropna(subset=[a, b])[b].to_numpy(float))
    w(f"SET tercile cut points: {q[0]:.1f}% and {q[1]:.1f}% of majors; median SET share {C.SET.median():.1f}%. "
      f"Rank correlation with G in sample C: SET {cc('G', 'SET'):+.2f}, INT {cc('G', 'INT'):+.2f}, "
      f"SET23 {cc('G', 'SET23'):+.2f}, INT23 {cc('G', 'INT23'):+.2f}; SET vs SET23 {cc('SET', 'SET23'):+.2f}, "
      f"INT vs INT23 {cc('INT', 'INT23'):+.2f}.\n")
    w(f"Label check — which OI 2000 major group tracks each 2023 CIP-2 family (Spearman across {labchk['n']} "
      "sample-C colleges; IPEDS C2023 first-major bachelor's shares; best two groups):\n")
    for cp, name, top in labchk["rows"]:
        w(f"- CIP {cp} {name}: " + ", ".join(f"{k.replace('pct_', '').replace('_2000', '')} {r:+.2f}" for k, r in top))
    w("\n### (e) Field prestige vs brand against institution-wide outcomes — ECOLOGICAL; not a test of field vs brand pricing (sample C)\n")
    fk = list(fields)
    w(f"{len(fk)} Wapman fields with ≥ {NMIN_FIELD} sample-C institutions. The outcome is the whole "
      "college's, pooled over all fields, not the field's graduates. **These statistics cannot tell whether pay "
      "follows field prestige or brand.** Brand G is close to the average of an institution's field prestiges "
      f"(Spearman {cal['rho_g_fbar']:+.2f} across the {cal['n_fbar']} sample-C institutions with at least one of these "
      "field ranks; equal-weight mean of within-field z(rank F)), so an outcome pooled over fields is expected to "
      "track G more closely than any single field's F, and to show a partial ρ(F, y | G) near zero, whichever "
      "of the two is priced. The calibration below quantifies this. Means over fields (joint bootstrap CI):\n")
    w("| statistic | median | top 1% | top 1% − median |")
    w("|---|---|---|---|")
    w(f"| c_F = ρ(field prestige, y) | {fmt(g(S, 'FLDMEAN', 'cF', 'k_median'))} | {fmt(g(S, 'FLDMEAN', 'cF', 'k_top1pc'))} | |")
    w(f"| c_G = ρ(brand, y), same institutions | {fmt(g(S, 'FLDMEAN', 'cG', 'k_median'))} | {fmt(g(S, 'FLDMEAN', 'cG', 'k_top1pc'))} | |")
    w(f"| c_F − c_G | {fmt(g(S, 'FLDMEAN', 'cF-cG', 'k_median'))} | {fmt(g(S, 'FLDMEAN', 'cF-cG', 'k_top1pc'))} | |")
    w(f"| partial ρ(F, y \\| G) | {fmt(g(S, 'FLDMEAN', 'F_G', 'k_median'))} | {fmt(g(S, 'FLDMEAN', 'F_G', 'k_top1pc'))} | "
      f"{fmt(g(S, 'FLDMEAN', 'F_G', 'D_k_top1pc'))} |")
    fl = [(LAB.get(f, f), int(fields[f].sum()), g(S, "FLD", f, "F_G|k_top1pc"), g(S, "FLD", f, "F_G|k_median"))
          for f in fk]
    npos1 = sum(1 for x in fl if x[2][0] > 0)
    nposm = sum(1 for x in fl if x[3][0] > 0)
    nsig1 = sum(1 for x in fl if x[2][1] > 0)
    nsigm = sum(1 for x in fl if x[3][1] > 0)
    assert npos1 == int(cal["obs"]["k_top1pc"]["npos"]) and nposm == int(cal["obs"]["k_median"]["npos"])
    w(f"\nPer field, partial ρ(F, top 1% | G) is positive in {npos1} of {len(fl)} fields and its 95% CI "
      f"excludes 0 from above in {nsig1} (median: positive in {nposm}, CI excludes 0 in {nsigm}); range "
      f"{min(x[2][0] for x in fl):+.2f} to {max(x[2][0] for x in fl):+.2f}. Per-field values (with n) are in the CSV.\n")
    w(f"**Calibration.** Two scenarios on the same {nC} institutions and {len(fk)} field rankings, {SIM_R} noise draws "
      f"(seed {SIM_SEED}). *Field-only*: outcome = institution-average field prestige (as above, standardized) + noise, "
      "i.e. pay depends only on field prestige, pooled over the institution's fields. *Brand-only*: outcome = "
      "z(rank G) + noise. Noise is iid normal with its level σ set, per outcome, so that the mean c_G matches the "
      "observed c_G. Scenario cells: mean over draws [2.5th, 97.5th percentile across draws; institutions fixed, so "
      "this is outcome noise only]; observed cells: point estimate [joint bootstrap 95% CI]. \"share ≤ obs\" = share "
      "of draws at or below the observed value.\n")
    w(f"| outcome | scenario | σ | c_G | c_F − c_G | share ≤ obs | mean partial ρ(F, y \\| G) | share ≤ obs | "
      f"fields with positive partial (of {len(fk)}) | share ≤ obs |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    fr = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"
    for y in ["k_median", "k_top1pc"]:
        for sc, lab in [("field_only", "field-only"), ("brand_only", "brand-only")]:
            st = cal["sim"][(sc, y)]
            w(f"| {OSHORT[y]} | {lab} | {st['sigma']:.2f} | {fr(st['cG'])} | {fr(st['cF-cG'])} | {st['cF-cG'][3]:.3f} | "
              f"{fr(st['F_G'])} | {st['F_G'][3]:.3f} | {st['npos'][0]:.1f} [{st['npos'][1]:.0f}, {st['npos'][2]:.0f}] | "
              f"{st['npos'][3]:.3f} |")
        w(f"| {OSHORT[y]} | **observed** | | {fmt(g(S, 'FLDMEAN', 'cG', y))} | {fmt(g(S, 'FLDMEAN', 'cF-cG', y))} | | "
          f"{fmt(g(S, 'FLDMEAN', 'F_G', y))} | | {int(cal['obs'][y]['npos'])} | |")
    w("\n" + calib_text(cal, res) + "\n")
    w(f"### Robustness (headline statistics re-estimated on sub-samples, {B_ROB} bootstrap draws each)\n")
    w("| sample | n | G raw: median | G raw: top 1% | raw: top 1% − median | G \\| sel + par: median | "
      "G \\| sel + par: top 1% | sel + par: top 1% − median | top 1% \\| median | top 1% \\| median + sel + par |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for smp, n in [(S, nC)] + list(rob_n.items()):
        w(f"| {smp} | {n} | {fmt(g(smp, 'G', 'none', 'k_median'))} | {fmt(g(smp, 'G', 'none', 'k_top1pc'))} | "
          f"{fmt(g(smp, 'G', 'none', 'D_k_top1pc'))} | "
          f"{fmt(g(smp, 'G', 'sel_par', 'k_median'))} | {fmt(g(smp, 'G', 'sel_par', 'k_top1pc'))} | "
          f"{fmt(g(smp, 'G', 'sel_par', 'D_k_top1pc'))} | {fmt(g(smp, 'G', 'med', 'k_top1pc'))} | "
          f"{fmt(g(smp, 'G', 'med_sel_par', 'k_top1pc'))} |")
    w("\nR1: SAT from 2001 only (drops colleges whose 2001 SAT is missing, e.g. Duke); R2: drops colleges whose "
      "OPEID6 is shared by several current undergraduate Scorecard units (OI outcome covers branch campuses, "
      "e.g. Penn State, Pittsburgh, Ohio State); R3: selectivity = current Scorecard SAT_AVG and 1 − ADM_RATE "
      "(scripts/55 convention; measured ~20 years after these cohorts entered); R4/R5: private non-profit / "
      "public only; R6: adds the OI system groups (multi = 1), each carrying the brand of its best-ranked "
      "Wapman member and group-level outcomes, selectivity and parents (outcomes then mix several campuses); "
      "R7: Barron's 2009 tier 1-2 colleges only (the elite and highly selective range where "
      "Chetty-Deming-Friedman locate tail effects).\n")
    w("### (f) Measurement: outcome reliability and age at measurement (OI Table 3, sample A colleges)\n")
    w("Cross-cohort reliability: mean pairwise Spearman of the outcome across the 1980, 1981 and 1982 cohorts, "
      "stepped up to a 3-cohort mean (Spearman-Brown). Part of the cohort-to-cohort difference is real change, "
      "so these are lower bounds on reliability; ρ/√R is the implied disattenuated coupling with G (raw, sample C).\n")
    w("| outcome | n | R (3 cohorts) | ρ(G, y) raw | ρ / √R |")
    w("|---|---|---|---|---|")
    for y in OUT + REL_PQ:
        e = g(S, "G", "none", y)[0]
        w(f"| {OLAB.get(y, OLAB_PQ.get(y, y))} | {rel[y]['n']} | {rel[y]['R3']:.3f} | {e:+.3f} | {e / np.sqrt(rel[y]['R3']):+.3f} |")
    w("\nAge profile (different birth cohorts, all measured in 2014; not a panel):\n")
    w("| age in 2014 | n | ρ(G, median) | ρ(G, top 1%) | ρ(G, median \\| par_rank) | ρ(G, top 1% \\| par_rank) |")
    w("|---|---|---|---|---|---|")
    for a in sorted(prof.age_2014.unique(), reverse=True):
        p = prof[prof.age_2014 == a].set_index("stat")
        w(f"| {a} | {int(p.n.iloc[0])} | " + " | ".join(
            f"{p.loc[s, 'est']:+.2f} [{p.loc[s, 'lo']:+.2f}, {p.loc[s, 'hi']:+.2f}]"
            for s in ["raw_med", "raw_top1", "par_med", "par_top1"]) + " |")
    # ---------------- method + crosswalk
    w("\n## Method\n")
    w(METHOD_TEXT.format(B=B, B_ROB=B_ROB, SIM_R=SIM_R, SIM_SEED=SIM_SEED, SIM_CAL_R=SIM_CAL_R))
    w("\n### Crosswalk and match rates\n")
    ns = len(xw)
    cnt = xw.status.value_counts()
    w(f"Wapman academia-wide ranks list {ns} named institutions (two unnamed rows dropped).\n")
    w("| step | institutions | share of Wapman |")
    w("|---|---|---|")
    a1 = (xw.route.str.startswith("A1")).sum(); a2 = (xw.route.str.startswith("A2")).sum()
    w(f"| Scorecard UNITID by normalized name (A1) | {a1} | {a1 / ns:.1%} |")
    w(f"| Scorecard UNITID by explicit alias (A2) | {a2} | {a2 / ns:.1%} |")
    w(f"| any Scorecard UNITID | {a1 + a2} | {(a1 + a2) / ns:.1%} |")
    inT11 = xw.super_opeid.notna().sum()
    rb = xw.routeB_used.sum()
    w(f"| linked to an OI super_opeid (via OPEID6, or OI-name fallback B for {rb}) | {inT11} | {inT11 / ns:.1%} |")
    for st_ in ["OI: insufficient data (super_opeid = -1)", "OI system group (multi = 1)",
                "duplicate on super_opeid (non-main unit dropped)", "no OI Table 2 row",
                "Scorecard OPEID6 not in OI Table 11", "no Scorecard record, no OI name match"]:
        w(f"| − {st_} | {int(cnt.get(st_, 0))} | {cnt.get(st_, 0) / ns:.1%} |")
    w(f"| **usable single-college OI units (sample A)** | {nA} | {nA / ns:.1%} |")
    w(f"| **with selectivity + parents + mix complete (sample C)** | {nC} | {nC / ns:.1%} |")
    agree = xw.route_agree.dropna()
    w(f"\nRoute check: where both the Scorecard route and the OI-name route resolve ({len(agree)} institutions), "
      f"they agree on the super_opeid in {int(agree.sum())} ({agree.mean():.1%}). ")
    ok = xw[xw.status == "ok"].copy()
    lowj = ok.sort_values("name_jacc").head(8)
    w("Lowest name similarity among usable links (Wapman name → OI name), checked by hand: " +
      "; ".join(f"{r.wapman_name} → {r.oi_inst_name if isinstance(r.oi_inst_name, str) else r.oi_name}"
                for r in lowj.itertuples()) + ". All are the same institution under an older or longer name; "
      "UT Rio Grande Valley was formed in 2015 from UT Pan American (whose OPEID it kept), so its OI outcomes "
      "are UT Pan American's.\n")
    w(f"Branch-shared OPEID6 among sample A: {int(A.branch_shared.sum())} colleges (excluded in robustness R2). "
      f"SAT source in sample C: {int((C.SAT_src == '2001').sum())} from 2001, {int((C.SAT_src == '2013').sum())} "
      "from 2013 (2001 missing).\n")
    sysg = xw[xw.status == "OI system group (multi = 1)"]
    w(f"Excluded as OI system groups ({len(sysg)} Wapman institutions): " +
      ", ".join(f"{r.wapman_name} (rank {r.g_rank})" for r in sysg.sort_values('g_rank').itertuples()) + ".\n")
    um = xw[xw.status.isin(["no Scorecard record, no OI name match", "Scorecard OPEID6 not in OI Table 11"])]
    w(f"Unlinked ({len(um)}): " + ", ".join(sorted(um.wapman_name)) + ".\n")
    rbl = xw[xw.routeB_used]
    w(f"Linked only by OI name (route B, {len(rbl)}): " + "; ".join(
        f"{r.wapman_name} → {r.oi_name} (status: {r.status})" for r in rbl.itertuples()) + ".\n")
    mo = xw[xw.status.isin(["no OI Table 2 row", "duplicate on super_opeid (non-main unit dropped)"])]
    t3c = pd.read_csv(OI / "mrc_table3.csv", usecols=["super_opeid", "cohort", "k_median"])
    t3c = t3c[t3c.super_opeid.isin(mo.super_opeid.dropna()) & t3c.k_median.notna()]
    first_cohort = t3c.groupby("super_opeid").cohort.min().astype(int).to_dict()
    del t3c
    w("Dropped with a link (no OI Table 2 row, or non-main duplicate): " + "; ".join(
        f"{r.wapman_name} → " + (f"{r.oi_name} (non-main duplicate)" if isinstance(r.oi_name, str)
                                 else f"super_opeid {int(r.super_opeid)} (no Table 2 row)")
        for r in mo.itertuples()) + ". The colleges without a Table 2 row have no outcomes for the 1980-82 cohorts in "
      "Table 3 either (earliest birth cohort with a median in Table 3: " + "; ".join(
          f"{r.wapman_name} {first_cohort.get(int(r.super_opeid), 'none')}"
          for r in mo[mo.status == "no OI Table 2 row"].itertuples()) + ").\n")
    ins = xw[xw.status == "OI: insufficient data (super_opeid = -1)"]
    w(f"OI insufficient data ({len(ins)}): " + ", ".join(sorted(ins.wapman_name)) + ".\n")
    tier = C.tier_name.value_counts()
    w("Sample C by OI tier: " + ", ".join(f"{k} {v}" for k, v in tier.items()) + ".\n")
    w("\n## Caveats\n")
    w(CAVEATS_TEXT.format(gsat=spear(C.G.to_numpy(float), C.SAT.to_numpy(float)),
                          n7=rob_n["R7_Barrons_tier_1_2_only"], nivy=int((C.tier_name == "Ivy Plus").sum()),
                          rho_gf=cal["rho_g_fbar"]))
    w("\n## Changes in this revision\n")
    w(REVISION_TEXT)
    w("\n## Provenance (for SOURCES.md)\n")
    w(provenance())
    OUT_MD.write_text("\n".join(L) + "\n")


CAL_STATS = [("cF-cG", "c_F − c_G"), ("F_G", "mean partial"), ("npos", "number of fields with a positive partial")]


def calib_checks(cal):
    """Where each observed (e) statistic falls relative to each scenario's 95% range of draws."""
    out = []
    for y in ["k_median", "k_top1pc"]:
        for k, lab in CAL_STATS:
            o = cal["obs"][y][k]
            for sc in ["field_only", "brand_only"]:
                m_, lo, hi, share = cal["sim"][(sc, y)][k]
                out.append(dict(y=y, k=k, lab=lab, sc=sc, obs=o, lo=lo, hi=hi, share=share,
                                pos="inside" if lo <= o <= hi else ("below" if o < lo else "above")))
    return out


def calib_short(cal):
    """One sentence for the Answer: where the observed statistics sit, computed."""
    ch = calib_checks(cal)
    off_f = [c for c in ch if c["sc"] == "field_only" and c["pos"] != "inside"]
    off_b = [c for c in ch if c["sc"] == "brand_only" and c["pos"] != "inside"]
    if not off_f and not off_b:
        return "The observed values lie inside the simulated range of both scenarios."
    short = {"cF-cG": "c_F − c_G", "F_G": "mean partial", "npos": "count of positive partials"}
    desc = lambda cs: "; ".join(f"{short[c['k']]}, {OSHORT[c['y']]}: {c['share']:.3f}" for c in cs)
    side = lambda cs: "low" if all(c["pos"] == "below" for c in cs) else "outside the range"
    if off_f and not off_b:
        return (f"The observed per-field partials sit {side(off_f)} for field-only pricing (share of draws ≤ observed: "
                f"{desc(off_f)}) and inside the brand-only range, but that rests on the calibration's assumptions "
                "(equal field weights; noise unrelated to selectivity and parents).")
    if off_b and not off_f:
        return (f"The observed statistics sit {side(off_b)} for brand-only pricing (share of draws ≤ observed: "
                f"{desc(off_b)}) and inside the field-only range, but that rests on the calibration's assumptions.")
    return (f"The observed statistics sit outside part of both scenarios' ranges (field-only: {desc(off_f)}; "
            f"brand-only: {desc(off_b)}; shares of draws ≤ observed).")


def calib_text(cal, res):
    """Plain-language reading of the (e) calibration; qualitative words are computed, not hard-coded."""
    S = "C_common"
    ch = calib_checks(cal)
    f1, b1 = cal["sim"][("field_only", "k_top1pc")], cal["sim"][("brand_only", "k_top1pc")]
    txt = ("Reading. Both scenarios give c_F well below c_G (top 1%: field-only "
           f"{f1['cF-cG'][0]:+.2f}, brand-only {b1['cF-cG'][0]:+.2f}) and a mean partial near zero "
           f"(field-only {f1['F_G'][0]:+.2f}, brand-only {b1['F_G'][0]:+.2f}), as observed. ")
    for sc, lab in [("brand_only", "brand-only"), ("field_only", "field-only")]:
        off = [c for c in ch if c["sc"] == sc and c["pos"] != "inside"]
        if not off:
            txt += f"Every observed c_F − c_G, mean partial and count lies inside the {lab} range. "
        else:
            nin = sum(1 for c in ch if c["sc"] == sc) - len(off)
            txt += (f"Against the {lab} range, {len(off)} of the {len(off) + nin} observed values fall outside it: " +
                    "; ".join(f"{c['lab']} for the {OSHORT[c['y']]} ({c['pos']}; share of draws ≤ observed "
                              f"{c['share']:.3f})" for c in off) + f"; the other {nin} lie inside. ")
    cover = []
    for y in ["k_median", "k_top1pc"]:
        for k in ["cF-cG", "F_G"]:
            o = get(res, S, "FLDMEAN", k, y)
            cover.append(all(o[1] <= cal["sim"][(sc, y)][k][0] <= o[2] for sc in ["field_only", "brand_only"]))
    txt += ("The bootstrap intervals of the observed c_F − c_G and mean partial cover both scenario means for both "
            "outcomes. " if all(cover) else
            "The bootstrap intervals of the observed c_F − c_G and mean partial do not always cover both scenario "
            "means (see the tables). ")
    if any(c["sc"] == "field_only" and c["pos"] != "inside" for c in ch):
        txt += ("The mean partial and the count of positive partials summarise the same per-field partials, so they are "
                "one signal: under this noise model, the pooled outcomes carry less field-specific signal beyond brand "
                "than equal-weight field pricing would leave. That does not establish brand pricing. The field-only scenario is one specific "
                "model (equal weights across the 63 Wapman fields, not weighted by where undergraduates enrol; noise "
                "unrelated to selectivity and parents), and no other field-pricing model was examined. ")
    txt += ("Section (e) therefore does not test Phase 1 finding (2) (brand predicts pay at least as well as field "
            "prestige): with a pooled institution-wide outcome, c_F < c_G and a near-zero partial are expected by "
            "construction whichever prestige is priced. It is kept as a descriptive, ecological table only.")
    return txt


# HTTP last-modified headers, read with curl -I on 2026-09-24
LASTMOD = {"mrc_table2.csv": "Mon, 02 Apr 2018 21:20:16 GMT", "mrc_table3.csv": "Mon, 02 Apr 2018 21:22:59 GMT",
           "mrc_table10.csv": "Mon, 02 Apr 2018 21:29:43 GMT", "mrc_table11.csv": "Mon, 02 Apr 2018 21:30:31 GMT",
           "Codebook-MRC-Table-2.pdf": "Mon, 02 Apr 2018 21:20:13 GMT",
           "Codebook-MRC-Table-3.pdf": "Mon, 02 Apr 2018 21:20:52 GMT",
           "Codebook-MRC-Table-10.pdf": "Mon, 02 Apr 2018 21:29:43 GMT",
           "Codebook-MRC-Table-11.pdf": "Mon, 02 Apr 2018 21:30:32 GMT"}


def provenance():
    base = "https://opportunityinsights.org/wp-content/uploads/2018/04/"
    lines = ["Opportunity Insights, *Mobility Report Cards: The Role of Colleges in Intergenerational Mobility* "
             "(Chetty, Friedman, Saez, Turner, Yagan; NBER w23618 2017; QJE 2020) — online data tables. "
             "Release: April 2018 upload (HTTP `last-modified` 2 April 2018 for every file, listed below). "
             "Local copies accessed 2026-09-23; all eight files re-downloaded on 2026-09-24 and byte-identical "
             "(md5) to the local copies. Public, no registration. Stored under "
             "`data/raw/opportunity_insights/` (gitignored; not redistributed).\n",
             "| file | URL | bytes | md5 | server last-modified |", "|---|---|---|---|---|"]
    for f in ["mrc_table2.csv", "mrc_table3.csv", "mrc_table10.csv", "mrc_table11.csv",
              "Codebook-MRC-Table-2.pdf", "Codebook-MRC-Table-3.pdf", "Codebook-MRC-Table-10.pdf",
              "Codebook-MRC-Table-11.pdf"]:
        p = OI / f
        lines.append(f"| `{f}` | {base}{f} | {p.stat().st_size:,} | `{md5(p)}` | {LASTMOD[f]} |")
    lines.append("\nAlso read (already documented in SOURCES.md): Wapman et al. 2022 `wapman2022/ranks.csv`; "
                 "College Scorecard institution file `scorecard_inst/Most-Recent-Cohorts-Institution.csv` "
                 f"(md5 `{md5(SC_INST)}`); IPEDS C2023_a `ipeds/C2023_a.csv` (label check only). "
                 "The OI paper PDF (https://opportunityinsights.org/wp-content/uploads/2018/03/coll_mrc_paper.pdf) "
                 "was read for definitions (earnings = individual pre-tax earnings in 2014, ranked within birth "
                 "cohort; 99th percentile $197,000 at age 34 for the 1980 cohort; college-level ranks and top-1% "
                 "rates stabilize by the early thirties) and is not stored.")
    return "\n".join(lines) + "\n"


def answer_extras(res, point, hr_se, rel, prof, rob_n, fields, C, cal):
    g = lambda *a: get(res, *a)
    f = lambda *a, d=2: fmt(g(*a), d)
    S = "C_common"
    R7 = "R7_Barrons_tier_1_2_only"
    pa = lambda st, age: prof[(prof.stat == st) & (prof.age_2014 == age)].est.iloc[0]
    return dict(
        gpar_d=f(S, "G", "par", "D_k_top1pc"), gpar_1=f(S, "G", "par", "k_top1pc"), gpar_m=f(S, "G", "par", "k_median"),
        g_r=f(S, "G", "none", "R_k_top1pc"), gsp_r=f(S, "G", "sel_par", "R_k_top1pc"),
        uniq1=f(S, "HR", "uniq_brand", "k_top1pc", d=3), uniqm=f(S, "HR", "uniq_brand", "k_median", d=3),
        shap_b1=f(S, "HR", "shap_brand", "k_top1pc"), shap_s1=f(S, "HR", "shap_selectivity", "k_top1pc"),
        shap_p1=f(S, "HR", "shap_parents", "k_top1pc"), bG1=f(S, "HR", "b_G", "k_top1pc"),
        bS1=f(S, "HR", "b_SAT", "k_top1pc"),
        satg_d=f(S, "SAT", "G_par", "D_k_top1pc"),
        fg1=f(S, "FLDMEAN", "F_G", "k_top1pc"), fgm=f(S, "FLDMEAN", "F_G", "k_median"),
        cfcg1=f(S, "FLDMEAN", "cF-cG", "k_top1pc"), nfld=len(fields), rho_gf=cal["rho_g_fbar"],
        sf_d=cal["sim"][("field_only", "k_top1pc")]["cF-cG"][0], sf_p=cal["sim"][("field_only", "k_top1pc")]["F_G"][0],
        sb_d=cal["sim"][("brand_only", "k_top1pc")]["cF-cG"][0], sb_p=cal["sim"][("brand_only", "k_top1pc")]["F_G"][0],
        cal_short=calib_short(cal),
        set_x1=f(S, "MIX", "SET_x_G", "k_top1pc"), set_xd=f(S, "MIX", "SET_x_G", "D_k_top1pc"),
        set23_x1=f(S, "MIX", "SET23_x_G", "k_top1pc"), set23_xd=f(S, "MIX", "SET23_x_G", "D_k_top1pc"),
        int_x1=f(S, "MIX", "INT_x_G", "k_top1pc"), int23_x1=f(S, "MIX", "INT23_x_G", "k_top1pc"),
        set23_raw_d=f(S, "MIX", "SET23_x_G_raw", "D_k_top1pc"),
        n7=rob_n[R7], r7_1=f(R7, "G", "none", "k_top1pc"), r7_m=f(R7, "G", "none", "k_median"),
        r7_d=f(R7, "G", "none", "D_k_top1pc"), r7_sp_1=f(R7, "G", "sel_par", "k_top1pc"),
        r7_sp_m=f(R7, "G", "sel_par", "k_median"), r7_sp_d=f(R7, "G", "sel_par", "D_k_top1pc"),
        r7_tbm=f(R7, "G", "med", "k_top1pc"), r7_tbm2=f(R7, "G", "med_sel_par", "k_top1pc"),
        n47=int((C.BARR.isin([1, 2]) & (C.PUBLIC == 0)).sum()),
        n4=rob_n["R4_private_nonprofit_only"], r4_d=f("R4_private_nonprofit_only", "G", "none", "D_k_top1pc"),
        r4_sp_d=f("R4_private_nonprofit_only", "G", "sel_par", "D_k_top1pc"),
        pq1_1=f(S, "G", "sel", "ktop1pc_cond_parq1"), pq1_q5=f(S, "G", "sel", "kq5_cond_parq1"),
        pq1_r=f(S, "G", "sel", "k_rank_cond_parq1"), pq1_d=f(S, "G", "sel", "D_top1_rank_parq1"),
        pq5_1=f(S, "G", "sel", "ktop1pc_cond_parq5"), pq5_r=f(S, "G", "sel", "k_rank_cond_parq5"),
        pq5_d=f(S, "G", "sel", "D_top1_rank_parq5"), pq1q5_d=f(S, "G", "sel", "D_q5_rank_parq1"),
        a23m=pa("raw_med", 23), a34m=pa("raw_med", 34), a23t=pa("raw_top1", 23), a34t=pa("raw_top1", 34),
        p26m=pa("par_med", 26), p34m=pa("par_med", 34), p26t=pa("par_top1", 26), p34t=pa("par_top1", 34),
        rel1=rel["k_top1pc"]["R3"], relm=rel["k_median"]["R3"],
        gd5=f(S, "G", "none", "D_k_top5pc"), g_sel_r=f(S, "G", "sel", "R_k_top1pc"),
        sat_d=f(S, "SAT", "none", "D_k_top1pc"), satg_d10=f(S, "SAT", "G_par", "D_k_top10pc"),
        set23_raw_1=f(S, "MIX", "SET23_x_G_raw", "k_top1pc"), set23_t1=f(S, "MIX", "SET23_T1", "k_top1pc"),
        set23_t3=f(S, "MIX", "SET23_T3", "k_top1pc"),
        g_set23=spear(C.dropna(subset=["SET23"]).G.to_numpy(float), C.dropna(subset=["SET23"]).SET23.to_numpy(float)),
    )


def key_table(res, rob_n):
    g = lambda *a: get(res, *a)
    S, R7 = "C_common", "R7_Barrons_tier_1_2_only"
    n = int(res[res["sample"] == S].n.max())
    rows = [
        ("ρ(G, median earnings)", g(S, "G", "none", "k_median"), f"C, n = {n}", "raw Spearman"),
        ("ρ(G, share top 1%)", g(S, "G", "none", "k_top1pc"), f"C, n = {n}", "raw Spearman"),
        ("top 1% − median", g(S, "G", "none", "D_k_top1pc"), f"C, n = {n}", "paired bootstrap difference"),
        ("ρ(G, top 1%) − ρ(G, median) | parents", g(S, "G", "par", "D_k_top1pc"), f"C, n = {n}", "par_rank + par_top1pc"),
        ("ρ(G, median | selectivity + parents)", g(S, "G", "sel_par", "k_median"), f"C, n = {n}", "SAT + rejection + Barron's + par_rank + par_top1pc"),
        ("ρ(G, top 1% | selectivity + parents)", g(S, "G", "sel_par", "k_top1pc"), f"C, n = {n}", "same"),
        ("top 1% − median | selectivity + parents", g(S, "G", "sel_par", "D_k_top1pc"), f"C, n = {n}", "same"),
        ("ρ(G, top 1% | strict)", g(S, "G", "strict", "k_top1pc"), f"C, n = {n}", "+ parent distribution, public, region"),
        ("ρ(G, top 1% | median)", g(S, "G", "med", "k_top1pc"), f"C, n = {n}", "tail beyond the median"),
        ("ρ(G, top 1% | median + selectivity + parents)", g(S, "G", "med_sel_par", "k_top1pc"), f"C, n = {n}", "tail beyond the median"),
        ("ρ(SAT, top 1% | median + brand + parents)", g(S, "SAT", "med_G_par", "k_top1pc"), f"C, n = {n}", "tail beyond the median"),
        ("ρ(SAT, top 1% | brand + parents) − ρ(SAT, median | brand + parents)", g(S, "SAT", "G_par", "D_k_top1pc"), f"C, n = {n}", "paired"),
        ("unique R² of brand, top 1% share", g(S, "HR", "uniq_brand", "k_top1pc"), f"C, n = {n}", "rank OLS horse race"),
        ("G × (education+health share) on top 1%", g(S, "MIX", "SET23_x_G", "k_top1pc"),
         f"C with IPEDS 2023, n = {int(res[(res.predictor == 'MIX') & res.spec.str.match('SET23_')].n.max())}",
         "ecological, + selectivity + parents"),
        ("ρ(G, top 1%) − ρ(G, median), Barron's 1-2 only", g(R7, "G", "none", "D_k_top1pc"), f"R7, n = {rob_n[R7]}", "raw"),
        ("same, | selectivity + parents", g(R7, "G", "sel_par", "D_k_top1pc"), f"R7, n = {rob_n[R7]}", "SAT + rejection + Barron's + parents"),
    ]
    L = ["| quantity | estimate [95% CI] | sample | specification |", "|---|---|---|---|"]
    for q, t, smp, spec in rows:
        d = 3 if "R²" in q else 2
        esc = lambda x: x.replace("|", "\\|")            # literal bars inside markdown cells
        L.append(f"| {esc(q)} | {fmt(t, d)} | {smp} | {esc(spec)} |")
    return "\n".join(L) + "\n"


ANSWER_TEMPLATE = """\
**Mostly no, and no more for the tail than for the median.** At the college level, academic brand adds at most a
small, borderline association with elite-tail outcomes beyond selectivity and parental income (top-1% share
{gsp_1}; about zero with the strict control set), and that residual is no larger for the top 1% than for the median.
Raw, brand tracks the top 1% clearly more than the median only in two overlapping subsets, the most selective
colleges (Barron's tier 1-2, n = {n7}; top 1% − median {r7_d}) and private non-profits (n = {n4}; {r4_d});
{n47} colleges are in both. Net of selectivity and parental income the private non-profit difference falls to
{r4_sp_d}; the Barron's 1-2 difference stays similar in size ({r7_sp_d}) but cannot be distinguished from zero at
this sample size.

- **Tail vs median.** Across {nC} PhD-granting universities (Wapman institutions linked to single-college OI units),
  academia-wide brand G tracks the share of former students reaching the top 1% of their cohort's earnings
  (ρ = {g1}) about as closely as it tracks their median earnings (ρ = {gm}). Raw, the coupling is slightly tighter
  toward the top (top 5% − median {gd5}; top 1% − median {gd}); the difference vanishes once parents' income is held
  fixed (top 1% − median {gpar_d}).
- **Beyond selectivity and parental income.** Net of selectivity (SAT, 2013 rejection rate, Barron's tier) and
  parental income, brand keeps a small, borderline association that is no larger for the tail than for the median:
  top 1% {gsp_1}, median {gsp_m}, difference {gsp_d}. With the strict control set (adding the parent-income
  distribution, public/private and region) it is about zero (top 1% {gst_1}, median {gst_m}). In the horse race brand's
  unique R² for the top-1% share is {uniq1}; the Shapley shares of R² are {shap_b1} for brand, {shap_s1} for
  selectivity and {shap_p1} for parents.
- **The tail is visible, but it loads on selectivity, not on the academic hierarchy.** Among colleges with the same
  median, higher-brand colleges send more students to the top 1% (partial ρ = {tbm}), but not once selectivity and
  parents are held fixed ({tbm2}), while SAT still predicts the top-1% share at a given median, brand and parental
  income ({sat_tbm}). SAT's coupling also rises toward the tail (raw top 1% − median {sat_d}; net of brand and parents
  top 10% − median {satg_d10}, top 1% − median {satg_d}).
- **Most selective colleges (suggestive only).** Among Barron's tier-1/2 colleges (n = {n7}), brand couples much more
  tightly with the top-1% share ({r7_1}) than with the median ({r7_m}; difference {r7_d}) and predicts a thicker top 1%
  at a given median ({r7_tbm}). Net of selectivity and parents the difference is {r7_sp_d} (top 1% {r7_sp_1}, median
  {r7_sp_m}) and the tail-beyond-median partial is {r7_tbm2}. This is the shape a Chetty-Deming-Friedman-type
  elite-tail premium would take, but here it is not distinguishable from selectivity and parental income. Among
  private non-profits (n = {n4}) the raw tail premium is {r4_d}, and {r4_sp_d} net of selectivity and parents.
- **Low-income students: a small brand-specific signal.** For children from the bottom parent-income quintile, brand
  predicts the chance of reaching the top 20% more than it predicts their mean rank, net of selectivity (top 20% {pq1_q5},
  mean rank {pq1_r}; difference {pq1q5_d}); for reaching the top 1% the difference is {pq1_d} (top 1% {pq1_1}). In the
  top quintile the top-1% vs mean-rank difference is {pq5_d}. Unconditionally, the top-1% vs mean-rank difference is
  {g_r} raw, {g_sel_r} net of selectivity and {gsp_r} net of selectivity and parents. Within-quintile conditioning holds
  parental income fixed only to the quintile, so this is a small secondary pattern about upward mobility, not an elite-tail
  effect that medians were hiding.
- **Major mix (ecological): no consistent modification.** Without controls, colleges with more education + health
  majors (IPEDS 2023) show a weaker brand–top-1% association (interaction {set23_raw_1}; ρ(G, top 1%) {set23_t1} in
  the bottom tercile of that share vs {set23_t3} in the top tercile), which is the direction the "pay set by setting"
  reading predicts. But high education/health colleges are also lower-brand (ρ(G, share) = {g_set23:+.2f}), so the
  tercile contrast partly reflects a narrower brand range, and net of selectivity and parents the interaction is
  {set23_x1}. With OI's own 2000 mix (health + public/social services) the controlled interaction is {set_x1}, the
  opposite sign. STEM + business shares do not modify the slope ({int_x1} with OI 2000 shares, {int23_x1} with IPEDS
  2023). At the college level, major mix does not reliably change how much brand is associated with the tail.
- **Field prestige vs brand (ecological): not testable with these data.** OI outcomes pool all of a college's fields,
  and brand G is close to an institution's average field prestige (ρ = {rho_gf:+.2f}). A smaller coupling for a single
  field's prestige than for brand (c_F − c_G for the top 1%: {cfcg1}) and a near-zero partial ρ(F, y | G) (mean over
  {nfld} fields: top 1% {fg1}, median {fgm}) are therefore expected whether pay follows field prestige or brand: a
  calibration on the same fields gives c_F − c_G {sf_d:+.2f} and partial {sf_p:+.2f} when only field prestige is priced,
  and {sb_d:+.2f} and {sb_p:+.2f} when only brand is. {cal_short} Section (e) does not test Phase 1 finding (2).

**For the working headline.** The public tail data do not rescue the academic hierarchy. At the college level,
elite-tail placement is predicted by selectivity and family background, and brand adds about as little beyond them at
the top 1% as at the median. This is consistent with labour markets pricing selectivity and family background more
than the academic hierarchy; whether a department's own standing is priced for its own graduates cannot be seen in
institution-wide outcomes (section (e)). It also narrows the "medians cannot see the tail" defence of brand prestige, and
of field prestige averaged over an institution's departments (which brand closely tracks), as far as public
college-level data allow: an elite-tail premium for academic prestige of the kind
Chetty-Deming-Friedman find for Ivy-Plus attendance, if it exists, would have to sit among the ~80 most selective
colleges, where this sample cannot resolve it. The age profile (OI Table 3; different cohorts, all measured in 2014)
shows brand's coupling with the median rising from {a23m:+.2f} at age 23 to {a34m:+.2f} at 34 while the top-1% coupling
is flatter ({a23t:+.2f} to {a34t:+.2f}); net of parents the median coupling rises from {p26m:+.2f} at 26 to {p34m:+.2f}
at 34 (top 1%: {p26t:+.2f} to {p34t:+.2f}). This fits finding (3), but it is confounded by cohort and by graduate-school
enrolment at young ages. Tail shares are measured precisely at this level (3-cohort reliability {rel1:.3f} for the
top-1% share, {relm:.3f} for the median), so the null differences are not an artefact of noisier tail outcomes.
"""

METHOD_TEXT = """\
- **Outcomes.** OI Mobility Report Card Table 2 (Chetty, Friedman, Saez, Turner, Yagan): for each college
  (super_opeid), children of the 1980-82 birth cohorts assigned to the college they attended most between ages 19
  and 22; individual pre-tax earnings in 2014 (ages 32-34), ranked within birth cohort nationally (top-1% cut-off
  about $197,000 at age 34 for the 1980 cohort, per the OI paper). k_median, k_mean, k_rank (mean rank), k_q5 (share in
  the top 20%), k_top10pc, k_top5pc, k_top1pc; parents: par_rank (mean parental income rank), par_q5, par_top10pc,
  par_top1pc, par_toppt1pc; parent-quintile-conditional outcomes kq5_cond_parqX, ktop1pc_cond_parqX,
  k_rank_cond_parqX. Table 3 repeats the outcomes by cohort 1980-1991 (all measured in 2014).
- **Prestige.** G = −(Wapman et al. 2022 academia-wide SpringRank ordinal, 2011-2020 hiring network), higher = more
  prestigious; field prestige F = −(published field rank) via `src/load_ar.py` (66-field universe from scripts/28).
- **Selectivity.** SAT = OI Table 10 `sat_avg_2001` (mean of 25th/75th percentile math+verbal SAT; closest to these
  cohorts' entry), `sat_avg_2013` where 2001 is missing; REJ = OI Table 10 `scorecard_rej_rate_2013`; Barron's 2009
  selectivity index as category dummies (1, 2, 3, 4, 5, other). Robustness R3 uses current Scorecard SAT_AVG and
  1 − ADM_RATE (the scripts/55 convention).
- **Estimators.** Spearman correlations; partial correlations = correlation of OLS residuals of within-sample ranks on
  the ranked continuous covariates and category dummies (as scripts/55). Horse race: OLS of the standardized rank of
  the outcome on standardized ranks of G, SAT, REJ, par_rank, par_top1pc plus Barron's dummies; HC1 SEs; Shapley
  decomposition of R² over three blocks. Field mix: z(rank y) ~ zG + zMix + zG·zMix (+ selectivity + parents), and
  Spearman(G, y) within the bottom and top terciles of the mix share. Field side: per field with ≥ 30 sample-C
  institutions, c_F = ρ(F, y), c_G = ρ(G, y) on the same institutions, partial ρ(F, y | G); averaged over fields.
- **Field-side calibration ((e)).** Institution-average field prestige = equal-weight mean over the (e) fields of
  within-field z(rank F), standardized. Field-only scenario: y = that average + σ·ε; brand-only: y = z(rank G) + σ·ε;
  ε iid N(0, 1), {SIM_R} draws (seed {SIM_SEED}), institutions and field rankings fixed at sample C. σ is set per
  outcome by bisection on the first {SIM_CAL_R} draws so that the mean c_G equals the observed c_G. The (e) statistics
  are computed for every draw with the same estimator as the observed ones (checked to agree to 1e-9 on the data).
- **Inference.** Joint institution bootstrap (B = {B} on sample C and sample A; {B_ROB} on each robustness sample and
  each Table 3 cohort), percentile 95% CIs. Every statistic is recomputed on the same draws, so differences
  (tail − median, T3 − T1, c_F − c_G) are paired. `boot_share_le0` in the CSV is the share of draws ≤ 0.
- **Reliability.** For each outcome, the mean pairwise Spearman across the 1980, 1981 and 1982 cohorts (Table 3) on the
  analysis colleges, stepped up to a 3-cohort mean by Spearman-Brown.
- **Seeds and determinism.** SEED = 60 (main), 61 (sample A), 70-76 (robustness), 160 (Table 3), 600 ((e)
  calibration); a second run gives
  byte-identical CSV, figure and write-up.
"""

CAVEATS_TEXT = """\
- **Ecological.** Every field-related statement ((d), (e)) relates an institution-wide outcome to an institution's
  major mix or to one department's prestige. It cannot say whether brand pays more to STEM graduates than to nursing
  graduates; it says only whether colleges with more of one kind of major show a different brand-outcome slope.
  The major-mix variables are also correlated with brand (education + health share vs G: see (d)). In (e) the
  pooled outcome and the near-identity of brand and average field prestige (ρ = {rho_gf:+.2f}) make field and brand pricing
  hard to tell apart at this level; the calibration that shows this assumes equal weights across fields and noise
  unrelated to selectivity and parents, and examines one field-pricing model only.
- **Not graduates, not fields.** OI assigns children to the college they attended most at ages 19-22, whether or not
  they graduated, and pools all majors. These are not the bachelor's-graduate field medians used elsewhere in the
  project, so the coupling levels here are not comparable to ρ_f.
- **Timing.** Outcomes are for the 1980-82 cohorts (college around 1998-2004, earnings in 2014); the Wapman hierarchy is
  the 2011-2020 hiring network, and selectivity is measured in 2001/2013 (Barron's 2009). Prestige is slow-moving, but
  G postdates the cohorts' college years.
- **Coverage.** The sample is PhD-granting universities only (Wapman institutions), so the range of
  selectivity is narrower than in the OI universe. OI system groups (multi = 1) are excluded in the main sample; this
  drops several top-50 flagships (Wisconsin, Illinois, Minnesota, Indiana, Maryland, Colorado, Pittsburgh, UMass).
  R6 adds them back with group-level outcomes and does not change the picture. Branch campuses sharing an OPEID6
  are pooled by OI (R2 drops them).
- **"Beyond selectivity" is conditional on the proxies.** SAT, rejection rate and Barron's tier are institution-level
  proxies for student quality. If they miss quality that is correlated with brand, brand's residual association is
  biased upward (so the small residual is, if anything, generous to brand); if brand itself raises selectivity,
  controlling for selectivity removes part of what brand does. Public college-level data cannot separate these (the
  same two readings as in scripts/55). Brand and SAT are correlated ({gsat:+.2f} in sample C), which widens partial CIs.
- **The elite range is thin.** R7 has {n7} colleges and the Ivy-Plus tier {nivy}; tail effects confined to a
  handful of colleges (as in Chetty-Deming-Friedman) would not be resolved by rank correlations at this sample size.
- **Earnings at 32-34.** Earnings keep fanning out after the early thirties; the OI paper reports that college-level
  ranks, including top-1% shares, have stabilized by then, but top-1% outcomes of later careers are not observed.
- **Table 3 age profile** compares different birth cohorts in the same year, not the same people over time; young-age
  coupling is depressed where many graduates are still in graduate school.
- **Label check.** The OI major groups are not documented at the CIP level in the codebook; the 2023 IPEDS comparison
  in (d) shows "public and social services" tracks protective services and public administration/social work, and
  that education (CIP 13) has no clear home in the eight OI groups. The education + health mix is therefore built
  from IPEDS 2023, about 20 years after these cohorts' majors.
"""

REVISION_TEXT = """\
- Section (e) (field prestige vs brand on institution-wide outcomes) was re-scoped after review. The earlier version
  said field prestige "adds essentially nothing beyond brand ... extending Phase 1 finding (2) to the tail". That
  clause is withdrawn: because brand is close to the average of an institution's field prestiges and the OI outcome
  pools all fields, c_F < c_G and a near-zero partial are expected whether fields or brand are priced, so (e) does not
  test finding (2). Added: ρ(G, institution-average field prestige), the two-scenario calibration table and its
  reading, a method line and a caveat. The (e) row was dropped from the key-numbers table, and the working-headline
  sentence contrasting brand with "a department's academic standing" was rephrased, since that contrast rested on (e).
- Every other estimate is unchanged: the statistics in the CSV other than the new calibration rows
  (section `e_field_prestige_calibration_sim`) are identical to the previous run.
"""


if __name__ == "__main__":
    main()
