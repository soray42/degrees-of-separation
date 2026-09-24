"""Selectivity, pushed as far as public data allow: does the academy's FIELD-SPECIFIC hierarchy carry
placement information of its own (reading A), or does within-field coupling only reflect how much each
field pays for institutional selectivity / brand (reading B)?

Builds on scripts/55 (Phase 1 #1: mean coupling +0.43 -> +0.14 net of SAT/admit/Pell/state; within-
institution slope +0.084 -> +0.013). Public data only: College Scorecard Field-of-Study (FoS) +
institution files, Wapman et al. 2022 published ranks. Descriptive; no causal claim.

Objects (per field f, across institutions i teaching f; 66-field universe, n >= NMIN=15 institutions):
  F   = field prestige (-published Wapman field Rank; higher = more prestigious)
  G   = academia-wide brand (-published Wapman Academia Rank; scripts/28 load_generic)
  Y   = Scorecard FoS BA median earnings 4 yrs after completion (EARN_MDN_4YR)
  institution controls (Scorecard institution file, scripts/55 joins): SAT_AVG, ADM_RATE, PCTPELL,
      CONTROL, ST_EARN (leave-one-out state earnings level), MD_EARN_WNE_P10 (institution-wide earnings)
  PROGRAM controls (FoS file itself, aggregated over the field's CIP-4 rows like load_er_scorecard):
      PPELL   = sum EARN_COUNT_PELL_WNE_4YR / sum (PELL + NOPELL counts) over rows where both released
      PMALE   = same with EARN_COUNT_MALE_WNE_4YR / NOMALE (gender composition; secondary)
      PPELL_D = DEBT_PELL_STGP_EVAL_N / (DEBT_PELL + DEBT_NOPELL) (Pell share in the debt cohort; wider
                coverage, different population = borrowers; sensitivity only)
      E_PELL, E_NOPELL = Pell / non-Pell median earnings (EARN_PELL/NOPELL_WNE_MDN_4YR)

(i)   Program composition: rank-partial coupling rho(F, Y | SAT, ADM, PCTPELL, CONTROL, state level
      [+ PPELL] [+ PMALE]) and the same with Pell-only / non-Pell-only earnings as the outcome.
(ii)  Field-specific component: residualize rank(F) AND rank(Y) on G + selectivity (+ composition),
      correlate the residuals (= partial correlation). Per field (joint institution bootstrap SEs),
      pooled three ways (unweighted mean with joint-bootstrap CI; REML random-effects mean; cell-level
      pooled slope of standardized residuals with institution-clustered SE), empirical-Bayes shrinkage
      per field, BH-FDR counts, TOST equivalence at +-EQ_BOUND, MDE. Reliability bracket: the residual
      F_perp has reliability 1 - (1 - rel_F)/(1 - R2(F | controls)) (classical error, rel_F from
      scripts/56: public-edge lower bound LB and extrapolated EXT). Mirror test: brand net of field
      prestige and the same controls. Split-half check of whether the field-specific component has any
      reproducible cross-field variation, and whether it lines up with baseline coupling or with
      selectivity pricing rho(SAT, Y).
(iii) Within-institution cross-department design, COMMON beta with FIELD-SPECIFIC returns:
        log Y_if = alpha_i + gamma_f + beta z(F)_if + sum_f D_f [d_f zSAT_i + e_f zADM_i + p_f zPCTPELL_i
                   + t_f zG_i] (+ program PPELL / PMALE slopes) + e
      alpha_i absorbed by within-institution demeaning; SEs clustered by institution (CR1) and an
      institution cluster (pairs) bootstrap; per-field beta_f version; reliability-corrected beta.
      (scripts/55 reported beta_f for this spec but its "pooled" beta omitted the field-specific slopes.)
(iv)  Horse race per field on ranks: blocks {F}, {selectivity: SAT, -ADM}, {brand G}, {institution
      earnings MD_EARN_WNE_P10}; Shapley R2 (general dominance), complete dominance, unique R2 of F;
      joint-bootstrap CIs of the mean shares; 3-block (no institution earnings) and 5-block (+ PPELL)
      variants.
(v)   Data that could separate A from B: checks the FoS/institution files for program-level test-score
      columns and counts the released provider x subject x prior-attainment-band cells in the public
      UK LEO provider file already on disk (data/raw/leo; availability only, no UK analysis here).
(vi)  Other earnings horizons of the same FoS release (EARN_MDN_1YR / EARN_MDN_5YR; different graduating
      cohorts, README s3, so not a career-time comparison; the programs are the same, so not an independent replication):
      the main partial (F | broad + G) and the within-institution FE2 beta re-estimated with 1-yr and 5-yr
      earnings on the SAME programs as the 4-yr estimate (per-horizon samples and the common sample where
      all three horizons are released), paired differences from shared bootstrap draws, per-field beta_f
      at each horizon (shared cluster-bootstrap draws), reliability bracket per horizon. Pooled over horizons on
      the common programs: each estimate averaged over the 4-, 1- and 5-yr outcomes with the CI from averaging the
      shared bootstrap draws (partials; within-institution beta for spec (a), the spec-literal design inst. FE +
      field FE + field x SAT + field x G, and the main spec; per-field beta_f with BH-FDR); cohort-size-weighted
      (WLS) within-institution versions (each horizon's program count, or the 4-yr count).
(vii) Does the remainder differ between fields? Split-half check of the within-institution per-field beta_f (the
      model re-estimated on random halves of institutions; fields with >= 60 programs; variants without computer
      science / engineering and with field x {state level, private}), and a power simulation for both split-half
      checks (expected reliability as a function of the between-field SD tau, given the fields' bootstrap SEs).
(viii) Wild cluster bootstrap (Rademacher signs by institution) of the per-field beta_f under a common-beta null
      (heterogeneity: Cochran's Q with a null that keeps the dependence between fields sharing institutions) and
      under beta_f = 0 (counts of fields above zero); across horizons, with the same signs at every horizon, the null
      distribution of the cross-horizon correlation of beta_f, which the shared programs make positive even with one
      common beta. Every wild null also with leverage-corrected (jackknife, WCR3-type) restricted residuals.
(ix)  Robustness of the cross-field heterogeneity to NONLINEAR field-specific pricing: the main spec plus field x
      {zG^2, zG^3}, or plus field x cubic in G, SAT and admit rate (common beta, per-field beta_f, Q, wild Q,
      split-half, computer science, CS/engineering contrast). Where the pooled per-field remainder sits under equal,
      precision and program-count weights (CS/engineering, business, rest; random-label benchmark; the pooled main
      spec without the CS/engineering fields), and the cross-horizon agreement of beta_f against centred pairs-
      bootstrap draws (sampling error only).

Reproduction checks (asserted): baseline rho_f = outputs/expanded66_gap_map.csv; scripts/55 broad
partial, horse race and within-institution beta_f (specs a, c) from data/interim/selectivity_fields.csv.

Reproducible: seeds fixed; outputs byte-identical on re-run.
Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
     .venv/bin/python scripts/58_selectivity_deep.py   (10-15 min on the shared VM, peak RSS ~1.2 GB)
Outputs: data/interim/selectivity_deep.csv, outputs/figures/selectivity_deep.png,
         SELECTIVITY_DEEP_RESULT.md (root; gitignored).
"""
from __future__ import annotations

import os
# single-threaded BLAS: the machine is shared, oversubscribed BLAS threads were 4-5x slower here, and a
# fixed thread count keeps floating-point reductions identical across runs
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import hashlib
import zipfile
import importlib.util
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, chi2, norm, binom
from scipy.optimize import minimize_scalar, brentq
from scipy.linalg import cho_factor, cho_solve
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_ar import load_ar_wapman
from src.load_er import load_er_scorecard
from src.gap import compute_gap_map
from src.crosswalks import fields as F
from src.crosswalks.institutions import normalize_institution_name

# scripts/55 (joins, institution file, partial-correlation conventions); it loads FIELDS66/LAB from
# scripts/28 via importlib exactly as scripts/52 does.
_spec = importlib.util.spec_from_file_location("s55", ROOT / "scripts" / "55_selectivity.py")
s55 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s55)
FIELDS66, LAB = s55.FIELDS66, s55.LAB

SEED = 58
NMIN = 15             # min institutions in a (field, sample) cell (as scripts/55)
DFMIN = 10            # min residual df for a partial correlation (as scripts/55)
B_BOOT = 1000         # joint institution bootstrap draws
B_FE = 999            # institution cluster bootstrap draws for the within-institution common beta
SPLIT_NMIN = 60       # well-estimated fields for the split-half checks (as scripts/55)
SPLIT_K = 200         # random half-splits of institutions
N_PERM = 1000         # field-label permutations (resource cap: <= 1000 replicates)
N_PERM_CROSS = 500    # first N_PERM_CROSS of them for the cross-fits
EQ_BOUND = 0.10       # equivalence bound for the mean partial correlation (TOST, 90% CI)
REL_MIN = 0.05        # residual reliability below this -> disattenuation not identified
B_GAP = 50
FE_PF_MIN_DISTINCT = 8   # per-field bootstrap: >= 8 distinct institutions (6 field-specific parameters + 2)
CSV_TOL = 1e-5        # scripts/55 CSV is written at 6 significant digits
# split-half of the within-institution per-field beta_f (same institution half-splits for every variant)
PF_SPLIT_NMIN = 60    # fields with >= 60 programs in the main-spec sample (the n_SAT >= 60 rule of the partial check)
PF_SPLIT_NMIN2 = 40   # wider field set (sensitivity)
# power of the split-half reliability test: simulated fields with the observed bootstrap SEs
POW_NSIM = 1000       # simulated data sets per tau
POW_NSPLIT = 50       # half-splits per simulated data set
POW_GRID_P = np.round(np.arange(0.0, 0.3001, 0.0025), 4)     # tau grid, partial correlations
POW_GRID_B = np.round(np.arange(0.0, 0.0801, 0.0005), 4)     # tau grid, beta_f (log points)
# wild cluster bootstrap (Rademacher weights by institution) of the per-field beta_f under common-beta / zero-beta
# nulls: heterogeneity (Q) and cross-horizon-correlation tests that keep shared institutions and shared programs
B_WILD = 1000         # replicates (resource cap: <= 1000)

FOS_FILE = ROOT / "data" / "raw" / "scorecard_fos" / "Most-Recent-Cohorts-Field-of-Study.csv"
INST_FILE = ROOT / "data" / "raw" / "scorecard_inst" / "Most-Recent-Cohorts-Institution.csv"
LEO_ZIP = ROOT / "data" / "raw" / "leo" / "leo_dashboard.zip"
GAP_MAP = ROOT / "outputs" / "expanded66_gap_map.csv"
S55_CSV = ROOT / "data" / "interim" / "selectivity_fields.csv"
REL_CSV = ROOT / "data" / "interim" / "prestige_reliability.csv"
OUT_CSV = ROOT / "data" / "interim" / "selectivity_deep.csv"
OUT_FIG = ROOT / "outputs" / "figures" / "selectivity_deep.png"
OUT_MD = ROOT / "SELECTIVITY_DEEP_RESULT.md"

# other earnings horizons of the same FoS release (different graduating cohorts; README s3)
HORIZONS = [("earn1", "EARN_MDN_1YR", "EARN_COUNT_WNE_1YR"), ("earn5", "EARN_MDN_5YR", "EARN_COUNT_WNE_5YR")]
HZ_Y = {"y4": "Y", "y1": "Y1", "y5": "Y5"}                 # horizon tag -> per-field array key
HZ_LAB = {"y4": "4-yr", "y1": "1-yr", "y5": "5-yr"}

BROAD = ["SAT_AVG", "ADM_RATE", "PCTPELL", "ST_EARN"]      # + CONTROL dummy = scripts/55 broad spec
ADMSET = ["ADM_RATE", "PCTPELL", "ST_EARN"]                # + CONTROL = scripts/55 adm_stlev
CAT = ["CONTROL"]

# per-field arrays: key -> cells column
VARS = {"P": "prestige_score", "Y": "earnings", "G": "G", "SAT_AVG": "SAT_AVG", "ADM_RATE": "ADM_RATE",
        "NEG_ADM": "NEG_ADM", "PCTPELL": "PCTPELL", "ST_EARN": "ST_EARN", "IE": "MD_EARN_WNE_P10",
        "PPELL": "PPELL", "PMALE": "PMALE", "PPELL_D": "PPELL_D", "E_PELL": "earn_pell",
        "E_NOPELL": "earn_nopell", "Y1": "earn1", "Y5": "earn5"}

# samples: name -> (parent sample, extra variables that must be finite)
SAMPLES = {
    "all": (None, []),
    "sat": (None, None),                                     # scripts/55 SAT family (special-cased)
    "comp": ("sat", ["PPELL", "E_PELL", "E_NOPELL"]),
    "comp_m": ("comp", ["PMALE"]),
    "compd": ("sat", ["PPELL_D"]),
    "adm": (None, None),                                     # scripts/55 ADM family (special-cased)
    "adm_comp": ("adm", ["PPELL", "E_PELL", "E_NOPELL"]),
    # earnings horizons: SAT sample + the other horizon released (same programs for both outcomes)
    "h1": ("sat", ["Y1"]),
    "h5": ("sat", ["Y5"]),
    "h3": ("sat", ["Y1", "Y5"]),                             # all three horizons released
}
SAMPLE_DESC = {
    "all": "all matched cells", "sat": "SAT sample (SAT_AVG, ADM_RATE, PCTPELL, state, control known)",
    "comp": "composition sample (SAT sample + program Pell share and Pell/non-Pell medians released)",
    "comp_m": "composition sample + program male share released",
    "compd": "SAT sample + debt-cohort Pell share released",
    "adm": "ADM sample (ADM_RATE, PCTPELL, state, control known; keeps test-blind institutions)",
    "adm_comp": "ADM sample + program Pell share and Pell/non-Pell medians released",
    "h1": "SAT sample + 1-yr median released", "h5": "SAT sample + 5-yr median released",
    "h3": "SAT sample + 1-yr and 5-yr medians released (common programs)",
}

# partial-correlation specs: name -> (sample, x, y, continuous controls, categorical controls)
SPECS = {
    # full sample
    "raw": ("all", "P", "Y", [], []),
    "F_G": ("all", "P", "Y", ["G"], []),
    "G_raw": ("all", "G", "Y", [], []),
    "G_F": ("all", "G", "Y", ["P"], []),
    # SAT sample
    "raw_sat": ("sat", "P", "Y", [], []),
    "sat_earn": ("sat", "SAT_AVG", "Y", [], []),
    "sel": ("sat", "P", "Y", ["SAT_AVG", "ADM_RATE"], []),
    "F_selG": ("sat", "P", "Y", ["SAT_AVG", "ADM_RATE", "G"], []),
    "broad": ("sat", "P", "Y", BROAD, CAT),
    "F_broadG": ("sat", "P", "Y", BROAD + ["G"], CAT),
    "F_broadGie": ("sat", "P", "Y", BROAD + ["G", "IE"], CAT),
    "G_broad": ("sat", "G", "Y", BROAD, CAT),
    "G_broadF": ("sat", "G", "Y", BROAD + ["P"], CAT),
    # composition sample (program Pell share)
    "raw_comp": ("comp", "P", "Y", [], []),
    "broad_comp": ("comp", "P", "Y", BROAD, CAT),
    "F_broadP": ("comp", "P", "Y", BROAD + ["PPELL"], CAT),
    "F_broadG_comp": ("comp", "P", "Y", BROAD + ["G"], CAT),
    "F_all": ("comp", "P", "Y", BROAD + ["G", "PPELL"], CAT),
    "G_all": ("comp", "G", "Y", BROAD + ["P", "PPELL"], CAT),
    "raw_nopell": ("comp", "P", "E_NOPELL", [], []),
    "raw_pell": ("comp", "P", "E_PELL", [], []),
    "F_broadP_nopell": ("comp", "P", "E_NOPELL", BROAD + ["PPELL"], CAT),
    "F_broadP_pell": ("comp", "P", "E_PELL", BROAD + ["PPELL"], CAT),
    "F_all_nopell": ("comp", "P", "E_NOPELL", BROAD + ["G", "PPELL"], CAT),
    "F_all_pell": ("comp", "P", "E_PELL", BROAD + ["G", "PPELL"], CAT),
    # + program male share
    "F_allM_base": ("comp_m", "P", "Y", BROAD + ["G", "PPELL"], CAT),
    "F_allM": ("comp_m", "P", "Y", BROAD + ["G", "PPELL", "PMALE"], CAT),
    # debt-cohort Pell share (wider coverage)
    "broad_compd": ("compd", "P", "Y", BROAD, CAT),
    "F_broadPd": ("compd", "P", "Y", BROAD + ["PPELL_D"], CAT),
    "F_allD": ("compd", "P", "Y", BROAD + ["G", "PPELL_D"], CAT),
    # ADM sample
    "raw_adm": ("adm", "P", "Y", [], []),
    "adm": ("adm", "P", "Y", ADMSET, CAT),
    "F_admG": ("adm", "P", "Y", ADMSET + ["G"], CAT),
    "F_admGP": ("adm_comp", "P", "Y", ADMSET + ["G", "PPELL"], CAT),
}
# earnings-horizon replication: per-horizon samples (4-yr vs 1-yr on h1, 4-yr vs 5-yr on h5) and the common
# sample h3 (raw, selectivity-only, main field-specific spec and the brand mirror at all three horizons)
for _s, _hz in (("h1", ["y4", "y1"]), ("h5", ["y4", "y5"])):
    for _h in _hz:
        SPECS[f"F_broadG_{_s}{_h}"] = (_s, "P", HZ_Y[_h], BROAD + ["G"], CAT)
for _h in ("y4", "y1", "y5"):
    SPECS[f"raw_h3{_h}"] = ("h3", "P", HZ_Y[_h], [], [])
    SPECS[f"broad_h3{_h}"] = ("h3", "P", HZ_Y[_h], BROAD, CAT)
    SPECS[f"F_broadG_h3{_h}"] = ("h3", "P", HZ_Y[_h], BROAD + ["G"], CAT)
    SPECS[f"G_broadF_h3{_h}"] = ("h3", "G", HZ_Y[_h], BROAD + ["P"], CAT)
HZ_SPECS = [k for k in SPECS if "_h1y" in k or "_h5y" in k or "_h3y" in k]
# horizon specs sharing sample, x and controls are residualized in one least-squares solve (speed only)
HZ_BATCH = {}
for _k in HZ_SPECS:
    _s, _x, _y, _c, _cat = SPECS[_k]
    if _c or _cat:
        HZ_BATCH.setdefault((_s, _x, tuple(_c), tuple(_cat)), []).append(_k)
HZ_BATCH_OF = {k: grp for grp in HZ_BATCH.values() for k in grp}
SPEC_DESC = {
    "raw": "ρ(F, Y), full sample (= gap-map baseline)",
    "F_G": "ρ(F, Y | brand G), full sample",
    "G_raw": "ρ(G, Y), full sample",
    "G_F": "ρ(G, Y | F), full sample",
    "raw_sat": "ρ(F, Y), SAT sample",
    "sat_earn": "selectivity pricing ρ(SAT_AVG, Y), SAT sample",
    "sel": "ρ(F, Y | SAT, ADM) [scripts/55 sel]",
    "F_selG": "ρ(F, Y | SAT, ADM, brand G) — minimal field-specific component",
    "broad": "ρ(F, Y | SAT, ADM, inst. Pell, control, state level) [scripts/55 broad]",
    "F_broadG": "ρ(F, Y | broad + G) — field-specific component",
    "F_broadGie": "ρ(F, Y | broad + G + institution-wide earnings)",
    "G_broad": "ρ(G, Y | broad)",
    "G_broadF": "ρ(G, Y | broad + F) — brand net of field prestige",
    "raw_comp": "ρ(F, Y), composition sample",
    "broad_comp": "ρ(F, Y | broad), composition sample",
    "F_broadP": "ρ(F, Y | broad + program Pell share)",
    "F_broadG_comp": "ρ(F, Y | broad + G), composition sample",
    "F_all": "ρ(F, Y | broad + G + program Pell share) — all public controls",
    "G_all": "ρ(G, Y | broad + F + program Pell share)",
    "raw_nopell": "ρ(F, non-Pell median), composition sample",
    "raw_pell": "ρ(F, Pell median), composition sample",
    "F_broadP_nopell": "ρ(F, non-Pell median | broad + program Pell share)",
    "F_broadP_pell": "ρ(F, Pell median | broad + program Pell share)",
    "F_all_nopell": "ρ(F, non-Pell median | broad + G + program Pell share)",
    "F_all_pell": "ρ(F, Pell median | broad + G + program Pell share)",
    "F_allM_base": "ρ(F, Y | broad + G + program Pell share), male-share sample",
    "F_allM": "ρ(F, Y | broad + G + program Pell share + program male share)",
    "broad_compd": "ρ(F, Y | broad), debt-cohort sample",
    "F_broadPd": "ρ(F, Y | broad + debt-cohort Pell share)",
    "F_allD": "ρ(F, Y | broad + G + debt-cohort Pell share)",
    "raw_adm": "ρ(F, Y), ADM sample",
    "adm": "ρ(F, Y | ADM, inst. Pell, control, state level) [scripts/55 adm_stlev]",
    "F_admG": "ρ(F, Y | ADM set + G)",
    "F_admGP": "ρ(F, Y | ADM set + G + program Pell share)",
}
_HZ_SMP = {"h1": "1-yr sample", "h5": "5-yr sample", "h3": "common programs"}
for _k, (_s, _x, _y, _c, _cat) in SPECS.items():
    if _k in HZ_SPECS:
        _h = _k[-2:]
        _core = {"raw": "ρ(F, Y)", "broad": "ρ(F, Y | selectivity set)", "F_broadG": "ρ(F, Y | selectivity set + G)",
                 "G_broadF": "ρ(G, Y | selectivity set + F)"}[_k.rsplit("_", 1)[0]]
        SPEC_DESC[_k] = f"{_core}, {HZ_LAB[_h]} earnings, {_HZ_SMP[_s]}"
SHORT = {
    "raw": "F, raw", "F_G": "F | G", "G_F": "G | F", "sel": "F | SAT, ADM", "F_selG": "F | SAT, ADM, G",
    "broad": "F | selectivity set", "F_broadG": "F | selectivity set + G",
    "F_broadGie": "F | selectivity set + G + institution earnings", "G_broad": "G | selectivity set",
    "G_broadF": "G | selectivity set + F", "adm": "F | ADM set", "F_admG": "F | ADM set + G (ADM sample)",
    "F_admGP": "F | ADM set + G + program Pell share (ADM sample)", "raw_comp": "F, raw (composition sample)",
    "broad_comp": "F | selectivity set (composition sample)", "F_broadP": "F | selectivity set + program Pell share",
    "F_broadG_comp": "F | selectivity set + G (composition sample)",
    "F_all": "F | selectivity set + G + program Pell share", "G_all": "G | selectivity set + F + program Pell share",
    "F_all_pell": "same, Pell-graduate median as outcome", "F_all_nopell": "same, non-Pell median as outcome",
    "F_broadP_pell": "F | selectivity set + program Pell share, Pell median",
    "F_broadP_nopell": "F | selectivity set + program Pell share, non-Pell median",
    "F_allM_base": "F | selectivity set + G + program Pell share (male-share sample)",
    "F_allM": "F | selectivity set + G + program Pell and male share",
    "broad_compd": "F | selectivity set (debt-cohort sample)", "F_broadPd": "F | selectivity set + debt-cohort Pell share",
    "F_allD": "F | selectivity set + G + debt-cohort Pell share",
}
for _k in HZ_SPECS:
    _core = {"raw": "F, raw", "broad": "F | selectivity set", "F_broadG": "F | selectivity set + G",
             "G_broadF": "G | selectivity set + F"}[_k.rsplit("_", 1)[0]]
    SHORT[_k] = f"{_core}, {HZ_LAB[_k[-2:]]}, {_HZ_SMP[SPECS[_k][0]]}"


def _t(x: str) -> str:
    """Escape pipes for a markdown table cell."""
    return x.replace("|", "\\|")


# field-prestige residual specs that get the reliability bracket and the pooled residual slope
PERP = ["F_G", "F_selG", "F_broadG", "F_broadGie", "F_broadG_comp", "F_all", "F_all_nopell", "F_allM", "F_allD",
        "F_admG", "F_admGP", "F_broadP", "broad", "broad_comp", "adm"] + \
       [k for k in HZ_SPECS if k.startswith("F_broadG") or k.startswith("broad")]

# horse-race blocks (ranks, standardized): F, selectivity, brand, institution-wide earnings
HR_BLOCKS = {"F": ["P"], "S": ["SAT_AVG", "NEG_ADM"], "G": ["G"], "IE": ["IE"]}
HR_LAB = {"F": "department prestige F", "S": "selectivity (SAT, −ADM)", "G": "brand G",
          "IE": "institution earnings", "C": "program Pell share"}

# within-institution specs: name -> (field-specific slope covariates, extra sample requirements, outcome)
INST_SLOPES = ["SAT_AVG", "ADM_RATE", "PCTPELL"]
COMP_NEED = ["PPELL", "earn_pell", "earn_nopell"]   # same cells for all-student, Pell, non-Pell outcomes
FE_SPECS = {
    "FE0": ([], [], "earnings"),
    "FE0_s": ([], INST_SLOPES + ["G"], "earnings"),
    "FE1": (INST_SLOPES, [], "earnings"),
    "FE2": (INST_SLOPES + ["G"], [], "earnings"),
    "FE2sg": (["SAT_AVG", "G"], ["ADM_RATE", "PCTPELL"], "earnings"),
    "FE2ie": (INST_SLOPES + ["G", "MD_EARN_WNE_P10"], [], "earnings"),
    "FE0_c": ([], INST_SLOPES + ["G"] + COMP_NEED, "earnings"),
    "FE2_c": (INST_SLOPES + ["G"], COMP_NEED, "earnings"),
    "FE3": (INST_SLOPES + ["G", "PPELL"], COMP_NEED, "earnings"),
    "FE3_np": (INST_SLOPES + ["G", "PPELL"], COMP_NEED + ["earnings"], "earn_nopell"),
    "FE3_p": (INST_SLOPES + ["G", "PPELL"], COMP_NEED + ["earnings"], "earn_pell"),
    "FE2_m": (INST_SLOPES + ["G", "PPELL"], COMP_NEED + ["PMALE"], "earnings"),
    "FE3m": (INST_SLOPES + ["G", "PPELL", "PMALE"], COMP_NEED, "earnings"),
    "FE2_d": (INST_SLOPES + ["G"], ["PPELL_D"], "earnings"),
    "FE3d": (INST_SLOPES + ["G", "PPELL_D"], [], "earnings"),
    # main + field-specific returns to the state earnings level and private control (same cells as FE2; asserted)
    "FE2sc": (INST_SLOPES + ["G", "ST_EARN", "PRIVATE"], [], "earnings"),
    # main + NONLINEAR field-specific returns: field x {zG^2, zG^3} (convex / elite-tail pricing of brand), and field x
    # cubic in G, SAT and admit rate (same cells as FE2; asserted). z = within-field z-score; a cubic in z spans the
    # same functions as a cubic in G, so the standardization does not matter for the fit.
    "FE2cg": (INST_SLOPES + ["G", "G_2", "G_3"], [], "earnings"),
    "FE2ca": (INST_SLOPES + ["G", "G_2", "G_3", "SAT_AVG_2", "SAT_AVG_3", "ADM_RATE_2", "ADM_RATE_3"], [], "earnings"),
}
POLY_VARS = ["G", "SAT_AVG", "ADM_RATE"]      # cubic terms <var>_2, <var>_3 are added to the cells in compute()
NL_SPECS = ["FE2", "FE2cg", "FE2ca"]         # linear main spec and its two nonlinear variants (robustness block)
# per-field bootstrap / split-half: a field's beta_f counts in a draw only with >= (field-specific parameters + 2)
# distinct institutions: FE2 has 6 (intercept, slope, 4 trait slopes) -> 8 (FE_PF_MIN_DISTINCT); FE2cg 8 -> 10; FE2ca 12 -> 14
NL_MIN_DISTINCT = {"FE2": 8, "FE2cg": 10, "FE2ca": 14}
PF_NMIN_MID = 30         # fields with >= 30 programs: a mid-size set between 'all' and the >= 60 split-half set
BUSINESS = ["accounting", "finance", "marketing", "management", "economics"]   # post hoc group (review), see (iii)
# earnings-horizon replication of (a) and the main spec: the outcome's partner horizons enter `need`, so all
# specs of a group are estimated on identical programs (fe_group_boot asserts it)
HZ_COL = {"y4": "earnings", "y1": "earn1", "y5": "earn5"}
for _s, _hz in (("h1", ["y4", "y1"]), ("h5", ["y4", "y5"]), ("h3", ["y4", "y1", "y5"])):
    for _h in _hz:
        _other = [HZ_COL[o] for o in _hz if o != _h]
        FE_SPECS[f"FE2_{_s}{_h}"] = (INST_SLOPES + ["G"], _other, HZ_COL[_h])
        if _s == "h3":
            FE_SPECS[f"FE0_{_s}{_h}"] = ([], INST_SLOPES + ["G"] + _other, HZ_COL[_h])
HZ3 = ("y4", "y1", "y5")
# common programs: the spec-literal design of the task (institution FE + field FE + field x SAT + field x G) at each
# horizon, and cohort-size-weighted (WLS) versions of (a), the spec-literal design and the main spec, weighted by
# the program count behind each horizon's median. All are on the same cells as FE0/FE2_h3* (same `need`), so they
# join that bootstrap group and every horizon average / weighted-vs-unweighted contrast is paired.
HZ_WCOL = {"y4": "cohort_n", "y1": "earn1_n", "y5": "earn5_n"}
FE_WEIGHT = {}
for _h in HZ3:
    _other = [HZ_COL[o] for o in HZ3 if o != _h]
    FE_SPECS[f"FE2sg_h3{_h}"] = (["SAT_AVG", "G"], ["ADM_RATE", "PCTPELL"] + _other, HZ_COL[_h])
    FE_SPECS[f"FE2w_h3{_h}"] = (INST_SLOPES + ["G"], _other, HZ_COL[_h])
    FE_WEIGHT[f"FE2w_h3{_h}"] = HZ_WCOL[_h]
    # alternative weights: the 4-yr program count at every horizon (main spec only). At 4 yr this is the same
    # model as FE2w_h3y4, which is reused (aliased after fitting) rather than re-estimated.
    if _h != "y4":
        FE_SPECS[f"FE2v_h3{_h}"] = (INST_SLOPES + ["G"], _other, HZ_COL[_h])
        FE_WEIGHT[f"FE2v_h3{_h}"] = "cohort_n"
FE_ALIAS = {"FE2v_h3y4": "FE2w_h3y4"}
HZ_FE_FAM = ["FE0", "FE2sg", "FE2", "FE2w", "FE2v"]   # families averaged over the three horizons
HZ_AVG_P = ["raw", "broad", "F_broadG", "G_broadF"]           # partial-correlation families averaged likewise
for _fam in HZ_AVG_P:
    SPEC_DESC[f"{_fam}_h3avg"] = ({"raw": "ρ(F, Y)", "broad": "ρ(F, Y | selectivity set)",
                                   "F_broadG": "ρ(F, Y | selectivity set + G)",
                                   "G_broadF": "ρ(G, Y | selectivity set + F)"}[_fam]
                                  + ", average of 4-, 1- and 5-yr earnings, common programs")
    SHORT[f"{_fam}_h3avg"] = ({"raw": "F, raw", "broad": "F | selectivity set", "F_broadG": "F | selectivity set + G",
                               "G_broadF": "G | selectivity set + F"}[_fam] + ", 4/1/5-yr average, common programs")
HZ_FE_PF = ["FE2_h3y4", "FE2_h3y1", "FE2_h3y5"]           # per-field beta_f at each horizon, common programs
FE_DESC = {
    "FE0": "(a) institution FE + field FE + β z(F) [scripts/55 (a)]",
    "FE0_s": "(a) on the SAT-covariate sample",
    "FE1": "(a) + field × {SAT, ADM, inst. Pell}",
    "FE2": "(a) + field × {SAT, ADM, inst. Pell, brand G} [main]",
    "FE2sg": "(a) + field × {SAT, brand G}",
    "FE2ie": "main + field × institution-wide earnings",
    "FE0_c": "(a), composition sample",
    "FE2_c": "main, composition sample",
    "FE3": "main + field × program Pell share",
    "FE3_np": "same, log non-Pell median as outcome",
    "FE3_p": "same, log Pell median as outcome",
    "FE2_m": "main + field × program Pell share, male-share sample",
    "FE3m": "main + field × program Pell and male share",
    "FE2_d": "main, debt-cohort sample",
    "FE3d": "main + field × debt-cohort Pell share",
    "FE2sc": "main + field × {state earnings level, private}",
    "FE2cg": "main + field × {zG², zG³} (cubic brand pricing)",
    "FE2ca": "main + field × cubic in G, SAT and admit rate",
}
_FE_BASE = {"FE0": "(a)", "FE2": "main", "FE2sg": "(a) + field × {SAT, brand G}",
            "FE2w": "main, cohort-size weighted", "FE2v": "main, weighted by the 4-yr program count"}
for _k in FE_SPECS:
    if _k not in FE_DESC:
        _b, _sh = _k.split("_")
        FE_DESC[_k] = (f"{_FE_BASE[_b]}, {HZ_LAB[_sh[2:]]} earnings, "
                       f"{ {'h1': '1-yr sample', 'h5': '5-yr sample', 'h3': 'common programs'}[_sh[:2]] }")
FE_DESC["FE2v_h3y4"] = f"{_FE_BASE['FE2v']}, 4-yr earnings, common programs"
for _b in HZ_FE_FAM:
    FE_DESC[f"{_b}_h3avg"] = f"{_FE_BASE[_b]}, average of 4-, 1- and 5-yr earnings, common programs"
# specs on identical cells share cluster-bootstrap draws; paired contrasts (a, b) -> beta_a - beta_b
# (horizon groups are appended at the end so the earlier groups keep their seeds)
# (FE2sc is appended to the SAT-covariate group: the rng is re-created per spec, so the other specs' draws are unchanged)
FE_GROUPS = [["FE0"], ["FE0_s", "FE2sg", "FE1", "FE2", "FE2ie", "FE2sc", "FE2cg", "FE2ca"],
             ["FE0_c", "FE2_c", "FE3", "FE3_np", "FE3_p"],
             ["FE2_m", "FE3m"], ["FE2_d", "FE3d"],
             ["FE2_h1y4", "FE2_h1y1"], ["FE2_h5y4", "FE2_h5y5"],
             ["FE0_h3y4", "FE0_h3y1", "FE0_h3y5", "FE2_h3y4", "FE2_h3y1", "FE2_h3y5"]
             # spec-literal and weighted variants: same group (same seed, rng re-created per spec, so the draws
             # of the six specs above are unchanged and every spec in the group sees identical draws)
             + [f"{_b}_h3{_h}" for _b in ("FE2sg", "FE2w") for _h in HZ3] + ["FE2v_h3y1", "FE2v_h3y5"]]
FE_PAIRS = [("FE2", "FE0_s"), ("FE2sg", "FE0_s"), ("FE2sc", "FE2"), ("FE2cg", "FE2"), ("FE2ca", "FE2"), ("FE3", "FE2_c"), ("FE3", "FE3_np"), ("FE3", "FE3_p"), ("FE3m", "FE2_m"),
            ("FE3d", "FE2_d"),
            ("FE2_h1y1", "FE2_h1y4"), ("FE2_h5y5", "FE2_h5y4"), ("FE2_h3y1", "FE2_h3y4"), ("FE2_h3y5", "FE2_h3y4"),
            ("FE0_h3y1", "FE0_h3y4"), ("FE0_h3y5", "FE0_h3y4"),
            ("FE2sg_h3y1", "FE2sg_h3y4"), ("FE2sg_h3y5", "FE2sg_h3y4"), ("FE2w_h3y1", "FE2w_h3y4"),
            ("FE2w_h3y5", "FE2w_h3y4")] + [(f"FE2w_h3{_h}", f"FE2_h3{_h}") for _h in HZ3]


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def load_composition() -> pd.DataFrame:
    """Program composition from the FoS file, aggregated to (field, inst_key) over the same rows
    load_er_scorecard keeps (CREDLEV 3, curated CIP-4, EARN_MDN_4YR released, non-empty inst_key)."""
    num = ["EARN_MDN_4YR", "EARN_COUNT_WNE_4YR", "EARN_COUNT_PELL_WNE_4YR", "EARN_COUNT_NOPELL_WNE_4YR",
           "EARN_COUNT_MALE_WNE_4YR", "EARN_COUNT_NOMALE_WNE_4YR", "DEBT_PELL_STGP_EVAL_N",
           "DEBT_NOPELL_STGP_EVAL_N"]
    d = pd.read_csv(FOS_FILE, usecols=["INSTNM", "CIPCODE", "CREDLEV"] + num, dtype=str)
    d = d[d["CREDLEV"] == "3"].copy()
    d["field"] = d["CIPCODE"].astype(str).str.zfill(4).map(F.cip4_to_field_key(FIELDS66))
    d = d[d["field"].notna()].copy()
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["EARN_MDN_4YR"])
    d["inst_key"] = d["INSTNM"].map(normalize_institution_name)
    d = d[d["inst_key"] != ""].copy()
    d["W"] = d["EARN_COUNT_WNE_4YR"].fillna(0.0)
    agg = {"W": "sum"}
    for tag, a, b in (("pp", "EARN_COUNT_PELL_WNE_4YR", "EARN_COUNT_NOPELL_WNE_4YR"),
                      ("pm", "EARN_COUNT_MALE_WNE_4YR", "EARN_COUNT_NOMALE_WNE_4YR"),
                      ("pd", "DEBT_PELL_STGP_EVAL_N", "DEBT_NOPELL_STGP_EVAL_N")):
        ok = d[a].notna() & d[b].notna()
        d[f"{tag}_num"] = np.where(ok, d[a], 0.0)
        d[f"{tag}_den"] = np.where(ok, d[a] + d[b], 0.0)
        d[f"{tag}_w"] = np.where(ok, d["W"], 0.0)
        d[f"{tag}_rows"] = ok.astype(int)
        agg.update({f"{tag}_num": "sum", f"{tag}_den": "sum", f"{tag}_w": "sum", f"{tag}_rows": "sum"})
    d["nrows"] = 1
    agg["nrows"] = "sum"
    g = d.groupby(["field", "inst_key"]).agg(agg).reset_index()
    for tag, col in (("pp", "PPELL"), ("pm", "PMALE"), ("pd", "PPELL_D")):
        g[col] = np.where(g[f"{tag}_den"] > 0, g[f"{tag}_num"] / g[f"{tag}_den"].where(g[f"{tag}_den"] > 0, 1), np.nan)
        g[f"{col}_rowcov"] = g[f"{tag}_rows"] / g["nrows"]
    return g[["field", "inst_key", "PPELL", "PMALE", "PPELL_D", "PPELL_rowcov", "PMALE_rowcov",
              "PPELL_D_rowcov", "nrows"]]


def field_arrays(d: pd.DataFrame, inst_index: dict) -> dict:
    fd = {"n": len(d), "inst_idx": d.inst_key.map(inst_index).to_numpy(int)}
    for k, c in VARS.items():
        fd[k] = d[c].to_numpy(float)
    fd["CONTROL"] = d.CONTROL.fillna(-1).to_numpy(int)
    st_ok = (d.STABBR.fillna("") != "").to_numpy()
    fin = lambda ks: np.all([np.isfinite(fd[k]) for k in ks], axis=0) if ks else np.ones(len(d), bool)
    m = {"all": fin(["P", "Y", "G"])}
    m["sat"] = d[s55.FAMILY_NEED["sat"]].notna().all(axis=1).to_numpy() & st_ok & m["all"]
    m["adm"] = d[s55.FAMILY_NEED["adm"]].notna().all(axis=1).to_numpy() & st_ok & m["all"]
    for s, (par, extra) in SAMPLES.items():
        if s in m:
            continue
        m[s] = m[par] & fin(extra)
    fd["mask"] = m
    fd["use"] = {s: int(m[s].sum()) >= NMIN for s in SAMPLES}
    return fd


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------
def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 1e-12 else np.nan


def _partial(xr, yr, Z):
    """Residualize rank vectors xr, yr on Z (with intercept column); returns r, df, ex, ey, R2_x."""
    n = len(xr)
    coef, _, rank, _ = np.linalg.lstsq(Z, np.column_stack([xr, yr]), rcond=None)
    df = n - rank - 1
    if df < DFMIN:
        return np.nan, df, None, None, np.nan
    E = np.column_stack([xr, yr]) - Z @ coef
    ex, ey = E[:, 0], E[:, 1]
    den = np.sqrt((ex @ ex) * (ey @ ey))
    if den <= 1e-9:
        return np.nan, df, None, None, np.nan
    sx = ((xr - xr.mean()) ** 2).sum()
    r2x = 1 - (ex @ ex) / sx if sx > 0 else np.nan
    return float(ex @ ey / den), df, ex, ey, float(r2x)


def _partial_multi(xr, ys, Z):
    """_partial for several outcome rank vectors sharing x and Z (one lstsq); same return per outcome."""
    n = len(xr)
    M = np.column_stack([xr] + ys)
    coef, _, rank, _ = np.linalg.lstsq(Z, M, rcond=None)
    df = n - rank - 1
    if df < DFMIN:
        return [(np.nan, df, None, None, np.nan)] * len(ys)
    E = M - Z @ coef
    ex = E[:, 0]
    sx = ((xr - xr.mean()) ** 2).sum()
    r2x = 1 - (ex @ ex) / sx if sx > 0 else np.nan
    out = []
    for j in range(len(ys)):
        ey = E[:, j + 1]
        den = np.sqrt((ex @ ex) * (ey @ ey))
        out.append((np.nan, df, None, None, np.nan) if den <= 1e-9 else (float(ex @ ey / den), df, ex, ey, float(r2x)))
    return out


def field_stats(fd: dict, rows: np.ndarray, point: bool = False, hr: bool = True, only=None) -> dict:
    """All per-field statistics on a (possibly resampled) row set. point=True also returns residual
    vectors, R2 of F on the controls and df (for pooling / reliability)."""
    out = {}
    rk_cache, rows_cache, dum_cache = {}, {}, {}

    def sample_rows(s):
        if s not in rows_cache:
            rows_cache[s] = rows[fd["mask"][s][rows]]
        return rows_cache[s]

    def rk(s, v):
        key = (s, v)
        if key not in rk_cache:
            rk_cache[key] = rankdata(fd[v][sample_rows(s)])
        return rk_cache[key]

    done = set()
    for name, (s, x, y, conts, cats) in SPECS.items():
        if (only is not None and name not in only) or name in done:
            continue
        if not fd["use"][s]:
            continue
        r = sample_rows(s)
        if len(r) < 4:
            continue
        xr, yr = rk(s, x), rk(s, y)
        if not conts and not cats:
            out[name] = _corr(xr, yr)
            continue
        cols = [np.ones((len(r), 1))] + [rk(s, c)[:, None] for c in conts]
        for c in cats:
            if (s, c) not in dum_cache:
                _, inv = np.unique(fd[c][r], return_inverse=True)
                L = inv.max() + 1
                dum_cache[(s, c)] = np.eye(L)[inv][:, 1:] if L > 1 else None
            if dum_cache[(s, c)] is not None:
                cols.append(dum_cache[(s, c)])
        if name in HZ_BATCH_OF and only is None:
            grp = HZ_BATCH_OF[name]
            res = zip(grp, _partial_multi(xr, [rk(s, SPECS[g][2]) for g in grp], np.hstack(cols)))
            done.update(grp)
        else:
            res = [(name, _partial(xr, yr, np.hstack(cols)))]
        for nm, (rho, df, ex, ey, r2x) in res:
            out[nm] = rho
            if point:
                out[f"df_{nm}"] = df
                out[f"r2x_{nm}"] = r2x
                out[f"_res_{nm}"] = (ex, ey, fd["inst_idx"][r]) if ex is not None else None
    if hr:
        out.update(horse_race(fd, rows, point=point))
    return out


def _zr(v):
    r = rankdata(v)
    s = r.std()
    return (r - r.mean()) / s if s > 0 else r * 0.0


def shapley(r2: dict, B: list) -> dict:
    out = {}
    for b in B:
        others = [o for o in B if o != b]
        v = 0.0
        for k in range(len(others) + 1):
            for s in combinations(others, k):
                w = factorial(k) * factorial(len(B) - k - 1) / factorial(len(B))
                v += w * (r2[frozenset(s + (b,))] - r2[frozenset(s)])
        out[b] = v
    return out


def complete_dominance(r2: dict, B: list, a: str, b: str) -> int:
    """+1 if block a completely dominates b (larger increment in every subset of the other blocks),
    -1 if b dominates a, 0 otherwise."""
    rest = [o for o in B if o not in (a, b)]
    da = []
    for k in range(len(rest) + 1):
        for s in combinations(rest, k):
            s = frozenset(s)
            da.append((r2[s | {a}] - r2[s]) - (r2[s | {b}] - r2[s]))
    da = np.array(da)
    return 1 if np.all(da > 0) else (-1 if np.all(da < 0) else 0)


def horse_race(fd: dict, rows: np.ndarray, point: bool = False) -> dict:
    """SAT sample: rank(Y) on z-ranked blocks; all-subset R2 -> Shapley (4 blocks: F, S, G, IE;
    3 blocks F, S, G; scripts/55's 3 blocks F, S, IE as a reproduction check). On the composition
    sample (point only): 5 blocks + program Pell share C."""
    out = {}
    if not fd["use"]["sat"]:
        return out
    r = rows[fd["mask"]["sat"][rows] & np.isfinite(fd["IE"][rows])]
    if len(r) < NMIN:
        return out
    y = _zr(fd["Y"][r])
    Xb = {b: np.column_stack([_zr(fd[v][r]) for v in vs]) for b, vs in HR_BLOCKS.items()}
    B4 = ["F", "S", "G", "IE"]
    sst = float(y @ y)
    r2 = {}
    for k in range(len(B4) + 1):
        for s in combinations(B4, k):
            if not s:
                r2[frozenset()] = 0.0
                continue
            X = np.column_stack([np.ones(len(r))] + [Xb[b] for b in s])
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            e = y - X @ coef
            r2[frozenset(s)] = 1 - float(e @ e) / sst
    sh4 = shapley(r2, B4)
    sh3 = shapley(r2, ["F", "S", "G"])
    sh55 = shapley(r2, ["F", "S", "IE"])
    out["hr_r2"] = r2[frozenset(B4)]
    out["hr_r2_3"] = r2[frozenset(["F", "S", "G"])]
    for b in B4:
        out[f"hr_sh_{b}"] = sh4[b]
    for b in ["F", "S", "G"]:
        out[f"hr3_sh_{b}"] = sh3[b]
    for b in ["F", "S", "IE"]:
        out[f"hr55_sh_{b}"] = sh55[b]
    out["hr_uniq_F"] = r2[frozenset(B4)] - r2[frozenset(["S", "G", "IE"])]
    out["hr_uniq_G"] = r2[frozenset(B4)] - r2[frozenset(["F", "S", "IE"])]
    out["hr3_uniq_F"] = r2[frozenset(["F", "S", "G"])] - r2[frozenset(["S", "G"])]
    if point:
        out["hr_n"] = len(r)
        for a, b in (("F", "G"), ("F", "S"), ("F", "IE"), ("G", "S")):
            out[f"hr_cd_{a}_{b}"] = complete_dominance(r2, B4, a, b)
        out["hr_leader"] = max(B4, key=lambda b: sh4[b])
        out["hr3_leader"] = max(["F", "S", "G"], key=lambda b: sh3[b])
        Xf = sm.add_constant(np.column_stack([Xb[b] for b in B4]))
        fit = sm.OLS(y, Xf).fit(cov_type="HC1")
        out["hr_b_F"], out["hr_se_F"], out["hr_p_F"] = float(fit.params[1]), float(fit.bse[1]), float(fit.pvalues[1])
        out["hr_b_G"], out["hr_se_G"], out["hr_p_G"] = float(fit.params[4]), float(fit.bse[4]), float(fit.pvalues[4])
        # 5-block variant with program Pell share (composition sample)
        if fd["use"]["comp"]:
            rc = rows[fd["mask"]["comp"][rows] & np.isfinite(fd["IE"][rows])]
            if len(rc) >= NMIN:
                yc = _zr(fd["Y"][rc])
                Xc = {b: np.column_stack([_zr(fd[v][rc]) for v in vs]) for b, vs in HR_BLOCKS.items()}
                Xc["C"] = _zr(fd["PPELL"][rc])[:, None]
                B5 = B4 + ["C"]
                r25 = {}
                for k in range(len(B5) + 1):
                    for s in combinations(B5, k):
                        if not s:
                            r25[frozenset()] = 0.0
                            continue
                        X = np.column_stack([np.ones(len(rc))] + [Xc[b] for b in s])
                        e = yc - X @ np.linalg.lstsq(X, yc, rcond=None)[0]
                        r25[frozenset(s)] = 1 - float(e @ e) / float(yc @ yc)
                sh5 = shapley(r25, B5)
                out["hr5_n"] = len(rc)
                out["hr5_r2"] = r25[frozenset(B5)]
                for b in B5:
                    out[f"hr5_sh_{b}"] = sh5[b]
    return out


def reml(y, v):
    """Random-effects meta-analysis (REML): mu, se_mu, tau, Q, Q p, I2. Fields treated as independent."""
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = np.asarray(y)[ok], np.asarray(v)[ok]
    k = len(y)
    if k < 3:
        return dict(k=k, mu=np.nan, se=np.nan, tau=np.nan, Q=np.nan, p=np.nan, I2=np.nan,
                    tau_lo=np.nan, tau_hi=np.nan)

    def nll(t2):
        w = 1 / (v + t2)
        mu = np.sum(w * y) / np.sum(w)
        return 0.5 * (np.sum(np.log(v + t2)) + np.log(np.sum(w)) + np.sum(w * (y - mu) ** 2))

    res = minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-10})
    t2 = float(res.x) if nll(res.x) < nll(0.0) else 0.0
    w = 1 / (v + t2)
    mu = float(np.sum(w * y) / np.sum(w))
    wf = 1 / v
    mu_fe = np.sum(wf * y) / np.sum(wf)
    Q = float(np.sum(wf * (y - mu_fe) ** 2))

    # Q-profile 95% CI for tau (Viechtbauer 2007): generalized Q(t2) is decreasing in t2
    def qgen(t):
        ww = 1 / (v + t)
        m = np.sum(ww * y) / np.sum(ww)
        return float(np.sum(ww * (y - m) ** 2))

    def solve(target):
        if qgen(0.0) <= target:
            return 0.0
        hi = 1.0
        while qgen(hi) > target and hi < 1e4:
            hi *= 2
        return float(brentq(lambda t: qgen(t) - target, 0.0, hi, xtol=1e-12))

    t2_lo, t2_hi = solve(chi2.ppf(0.975, k - 1)), solve(chi2.ppf(0.025, k - 1))
    return dict(k=k, mu=mu, se=float(np.sqrt(1 / np.sum(w))), tau=float(np.sqrt(t2)), Q=Q,
                p=float(chi2.sf(Q, k - 1)), I2=float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0,
                tau_lo=float(np.sqrt(t2_lo)), tau_hi=float(np.sqrt(t2_hi)))


def bh(p, q=0.05):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), bool)
    ps = p[ok]
    m = len(ps)
    if m == 0:
        return out
    order = np.argsort(ps, kind="mergesort")
    thr = q * np.arange(1, m + 1) / m
    passed = ps[order] <= thr
    kmax = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    sel = np.zeros(m, bool); sel[order[:kmax]] = True
    out[np.where(ok)[0]] = sel
    return out


def pooled_resid_slope(res_list):
    """Pool standardized within-field residual pairs across fields: slope of ey on ex (both unit
    variance within field) = (n_f - 1)-weighted mean partial r; CR1 SE clustered by institution."""
    xs, ys, gs = [], [], []
    for ex, ey, g in res_list:
        xs.append(ex / ex.std(ddof=1)); ys.append(ey / ey.std(ddof=1)); gs.append(g)
    x, y, g = np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)
    sxx = float(x @ x)
    b = float(x @ y) / sxx
    e = y - b * x
    sc = pd.Series(x * e).groupby(g).sum().to_numpy()
    G = len(sc)
    V = G / (G - 1) * float(sc @ sc) / sxx ** 2
    return dict(b=b, se=float(np.sqrt(V)), n_cells=len(x), n_clusters=G)


# ---------------------------------------------------------------------------------------------
# (iii) within-institution cross-department design
# ---------------------------------------------------------------------------------------------
def fe_sample(cells, need, yvar):
    d = cells.dropna(subset=[yvar, "prestige_score"] + need).copy()
    d = d[d[yvar] > 0]
    while True:
        n0 = len(d)
        d = d[d.groupby("inst_key").field.transform("size") >= 2]
        d = d[d.groupby("field").inst_key.transform("size") >= NMIN]
        if len(d) == n0:
            break
    return d.sort_values(["field", "inst_key"]).reset_index(drop=True)


def fe_build(cells, slopes, need, yvar, per_field, wcol=None):
    """Within-institution design matrix. wcol: program weights (WLS). With weights the institution FE are
    absorbed by weighted within-institution demeaning and rows are scaled by sqrt(w), which is WLS with
    institution dummies (Frisch-Waugh-Lovell); regressors are standardized exactly as in the unweighted spec."""
    d = fe_sample(cells, slopes + need, yvar)
    fields = sorted(d.field.unique())
    code = pd.Categorical(d.field, categories=fields).codes
    D = np.eye(len(fields))[code]
    g = d.groupby("field")
    z = g.prestige_score.transform(lambda s: (s - s.mean()) / s.std()).to_numpy()
    cols, names = [D[:, 1:]], [f"g_{f}" for f in fields[1:]]
    if per_field:
        cols.append(D * z[:, None]); names += [f"b_{f}" for f in fields]
    else:
        cols.append(z[:, None]); names += ["beta"]
    for c in slopes:
        zc = g[c].transform(lambda s: (s - s.mean()) / s.std()).fillna(0).to_numpy()
        cols.append(D * zc[:, None]); names += [f"x_{c}_{f}" for f in fields]
    X = np.hstack(cols)
    y = np.log(d[yvar].to_numpy(float))
    ic = pd.factorize(d.inst_key)[0]
    cnt = np.bincount(ic).astype(float)

    def dm(M):
        s = np.zeros((cnt.size,) + M.shape[1:]); np.add.at(s, ic, M)
        return M - (s / (cnt[:, None] if M.ndim == 2 else cnt))[ic]

    if wcol is None:
        return dict(d=d, fields=fields, X=dm(X), y=dm(y), names=names, groups=ic, cnt=cnt, z=z, code=code)
    wv = d[wcol].to_numpy(float)
    assert np.all(np.isfinite(wv)), "program weights must be finite"
    wv = np.clip(wv, 1.0, None)                  # as load_er_scorecard: a missing / zero count counts as 1
    ws = np.bincount(ic, weights=wv)
    sw = np.sqrt(wv)

    def wdm(M):
        W_ = wv[:, None] if M.ndim == 2 else wv
        s = np.zeros((cnt.size,) + M.shape[1:]); np.add.at(s, ic, M * W_)
        return (M - (s / (ws[:, None] if M.ndim == 2 else ws))[ic]) * (sw[:, None] if M.ndim == 2 else sw)

    return dict(d=d, fields=fields, X=wdm(X), y=wdm(y), names=names, groups=ic, cnt=cnt, z=z, code=code, w=wv)


def fe_fit(M, rel=None):
    fit = sm.OLS(M["y"], M["X"]).fit(cov_type="cluster", cov_kwds={"groups": M["groups"]})
    p, V = np.asarray(fit.params), np.asarray(fit.cov_params())
    out = dict(n_cells=len(M["y"]), n_inst=int(M["cnt"].size), n_fields=len(M["fields"]), n_par=len(M["names"]),
               fields=M["fields"], n_by_field=M["d"].groupby("field").size())
    if "beta" in M["names"]:
        j = M["names"].index("beta")
        out.update(beta=float(p[j]), se=float(np.sqrt(V[j, j])))
        # reliability-corrected slope: classical error in z(F) with variance (1 - rel_f) per cell,
        # shrunk by within-institution demeaning (1 - 1/n_i); attenuation = 1 - err var / resid var
        others = np.delete(M["X"], j, axis=1)
        xj = M["X"][:, j]
        rj = xj - others @ np.linalg.lstsq(others, xj, rcond=None)[0]
        out["resid_var_share"] = float(rj @ rj / (xj @ xj))
        if rel is not None:
            for tag, rv in rel.items():
                rf = np.array([rv.get(f, np.nan) for f in M["fields"]])
                rf = np.where(np.isfinite(rf), rf, np.nanmean(rf))[M["code"]]
                err = np.sum((1 - rf) * (1 - 1 / M["cnt"][M["groups"]]))
                lam = 1 - err / float(rj @ rj)
                out[f"lam_{tag}"] = float(lam)
    else:
        bidx = [M["names"].index(f"b_{f}") for f in M["fields"]]
        out["beta_f"] = pd.Series(p[bidx], index=M["fields"])
        out["se_f"] = pd.Series(np.sqrt(np.diag(V)[bidx]), index=M["fields"])
        k = len(bidx)
        R = np.zeros((k - 1, len(p)))
        for r_, j in enumerate(bidx[1:]):
            R[r_, bidx[0]] = -1; R[r_, j] = 1
        rb, RV = R @ p, R @ V @ R.T
        W = float(rb @ np.linalg.pinv(RV) @ rb); rk = int(np.linalg.matrix_rank(RV))
        out.update(wald=W, wald_df=rk, wald_p=float(chi2.sf(W, rk)))
    return out


def fe_group_boot(Ms: list, seed, B: int) -> np.ndarray:
    """Institution (pairs) cluster bootstrap of the common beta for several specs estimated on the SAME
    cells: identical institution draws for every spec (so differences between specs are paired). Each
    resampled copy keeps its own FE (rows are already demeaned within institution). Uses per-cluster
    cross-products: X'X = sum_g c_g X_g'X_g, X'y = sum_g c_g X_g'y_g (c_g = times drawn). Specs with an
    identical design matrix (the same regressors, different outcome: e.g. one spec at three earnings horizons)
    share X'X and its Cholesky factor within a draw; identical inputs give identical factorizations, so each spec's
    draws are bit-for-bit those of fitting it alone."""
    M0 = Ms[0]
    for M in Ms[1:]:
        assert np.array_equal(M["groups"], M0["groups"]) and M["d"].inst_key.equals(M0["d"].inst_key) \
            and M["d"].field.equals(M0["d"].field), "grouped FE specs must share cells"
    G = int(M0["cnt"].size)
    idx = [np.where(M0["groups"] == g)[0] for g in range(G)]
    out = np.full((B, len(Ms)), np.nan)
    xgroups = []                                           # lists of spec indices with an identical X
    for m, M in enumerate(Ms):
        for gl in xgroups:
            R = Ms[gl[0]]
            if R["names"] == M["names"] and R["X"].shape == M["X"].shape and np.array_equal(R["X"], M["X"]):
                gl.append(m)
                break
        else:
            xgroups.append([m])
    for gl in xgroups:
        X, j = Ms[gl[0]]["X"], Ms[gl[0]]["names"].index("beta")
        A = np.stack([X[r].T @ X[r] for r in idx])
        bvs = [np.stack([X[r].T @ Ms[m]["y"][r] for r in idx]) for m in gl]
        for m, bv in zip(gl, bvs):
            sol_full = np.linalg.lstsq(A.sum(0), bv.sum(0), rcond=None)[0][j]
            assert abs(sol_full - np.linalg.lstsq(X, Ms[m]["y"], rcond=None)[0][j]) < 1e-8, "normal equations drifted"
        rng = np.random.default_rng(seed)                  # same draws for every spec in the group
        for bi in range(B):
            c = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
            XtX = np.tensordot(c, A, axes=1)
            keep = np.diag(XtX) > 1e-10
            if not keep[j]:
                continue
            Ak = XtX[np.ix_(keep, keep)]
            jk = int(np.cumsum(keep)[j]) - 1
            try:
                cf = cho_factor(Ak)
            except np.linalg.LinAlgError:
                cf = None
            for m, bv in zip(gl, bvs):
                bk = (c @ bv)[keep]
                sol = None
                if cf is not None:
                    sol = cho_solve(cf, bk)
                    if not np.all(np.isfinite(sol)) or np.linalg.norm(Ak @ sol - bk) > 1e-6 * max(1.0, np.linalg.norm(bk)):
                        sol = None
                if sol is None:
                    sol = np.linalg.lstsq(Ak, bk, rcond=None)[0]
                out[bi, m] = sol[jk]
        del A, bvs
    return out


def fe_boot_pf(M, seed, B: int, min_distinct: int) -> np.ndarray:
    """Institution (pairs) cluster bootstrap of the per-field beta_f of a per_field=True model.
    Returns (B, k); NaN where the field has fewer than `min_distinct` distinct institutions in the draw
    (its field-specific intercept, slope and control slopes are then barely identified)."""
    G = int(M["cnt"].size)
    idx = [np.where(M["groups"] == g)[0] for g in range(G)]
    X, y = M["X"], M["y"]
    k = len(M["fields"])
    bidx = np.array([M["names"].index(f"b_{f}") for f in M["fields"]])
    inst_field = np.zeros((G, k), bool)
    inst_field[M["groups"], M["code"]] = True
    A = np.stack([X[r].T @ X[r] for r in idx]); bv = np.stack([X[r].T @ y[r] for r in idx])
    rng = np.random.default_rng(seed)
    out = np.full((B, k), np.nan)
    p = X.shape[1]
    for bi in range(B):
        c = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
        XtX = np.tensordot(c, A, axes=1); Xty = c @ bv
        keep = np.diag(XtX) > 1e-10
        sol = np.linalg.lstsq(XtX[np.ix_(keep, keep)], Xty[keep], rcond=None)[0]
        full = np.full(p, np.nan); full[keep] = sol
        ok = (inst_field & (c > 0)[:, None]).sum(0) >= min_distinct
        out[bi] = np.where(ok, full[bidx], np.nan)
    del A, bv
    return out


def _drop_beta(M):
    """The common-beta model M without its beta column: the null beta_f = 0 for every field."""
    j = M["names"].index("beta")
    return dict(M, X=np.delete(M["X"], j, axis=1), names=[n for n in M["names"] if n != "beta"])


def _cr1_f(P, u):
    """CR1 (institution-clustered) SEs of the per-field beta_f of an OLS fit whose residuals are M_u u
    (identical to statsmodels' cluster covariance: factor G/(G-1) (n-1)/(n-p))."""
    uh = u - P["X"] @ (P["Pu"] @ u)
    n, p = P["X"].shape
    S = np.zeros((P["G"], P["Pb"].shape[0]))
    np.add.at(S, P["g"], (P["Pb"] * uh[None, :]).T)
    return np.sqrt((S ** 2).sum(0) * P["G"] / (P["G"] - 1) * (n - 1) / (n - p))


def _cv3_resid(M, r):
    """Leverage-corrected (jackknife / CV3-type) residuals of a within-institution OLS fit: per institution g,
    r_g <- (I - H_gg)^+ r_g, with H_gg = X_g (X'X)^+ X_g' + 11'/n_g (the second term is the absorbed institution FE;
    rows are demeaned within institution). These are the residuals of the WCR3x wild bootstraps of MacKinnon, Nielsen
    and Webb (2023): with few programs per field-specific parameter the plain restricted residuals are shrunk by the
    fit and the wild null is too narrow; the transformation undoes that shrinkage."""
    X, g = M["X"], M["groups"]
    A = np.linalg.pinv(X.T @ X)
    out = np.zeros_like(r)
    for gg in range(int(M["cnt"].size)):
        ix = np.where(g == gg)[0]
        ng = len(ix)
        H = X[ix] @ A @ X[ix].T + np.full((ng, ng), 1.0 / ng)
        out[ix] = np.linalg.pinv(np.eye(ng) - H, rcond=1e-10) @ r[ix]
    return out


def wild_pf(pairs, seed, B: int, cr1=False, kinds=("WCR", "WCU")) -> dict:
    """Wild cluster bootstrap (Rademacher weights by institution) of the per-field beta_f under a null imposed by
    restricted models. pairs = [(Mr, Mu), ...], one per outcome (earnings horizon): Mr = restricted model (common
    beta, or no prestige slope for the null beta_f = 0), Mu = per-field model on the same cells. Both are within-
    institution demeaned; multiplying residuals by an institution-level weight keeps them demeaned. In a replicate
    y* = X_r b_r + w_g r, where r = the restricted residuals (WCR, imposes the null on the residuals) or the
    per-field model's residuals (WCU, null imposed only through X_r b_r), or the restricted residuals with the
    leverage correction of _cv3_resid (WCR3; kinds must name it). The SAME institution weights are applied
    to every outcome in a replicate, so dependence between outcomes that share programs (persistent program-level
    residuals) is kept; the weights are drawn once, before the kinds, so adding a kind leaves the others unchanged.
    beta*_f = Pb y* = b0 + Pb (w r), Pb = the beta_f rows of pinv(X_u), b0 = Pb X_r b_r.
    Returns observed beta_f (k, m), null draws per kind: (B, k, m), residual correlations between outcomes and,
    with cr1=True, per-replicate CR1 SEs (B, k, m)."""
    Ps = []
    for Mr, Mu in pairs:
        assert Mr["d"].inst_key.equals(Mu["d"].inst_key) and Mr["d"].field.equals(Mu["d"].field) \
            and np.array_equal(Mr["y"], Mu["y"]), "wild bootstrap: restricted and per-field models must share cells"
        br = np.linalg.lstsq(Mr["X"], Mr["y"], rcond=None)[0]
        Pu = np.linalg.pinv(Mu["X"])
        bidx = np.array([Mu["names"].index(f"b_{f}") for f in Mu["fields"]])
        P = dict(X=Mu["X"], Pu=Pu, Pb=Pu[bidx], g=Mu["groups"], G=int(Mu["cnt"].size))
        P["b0"] = P["Pb"] @ (Mr["X"] @ br)
        P["obs"] = P["Pb"] @ Mu["y"]
        P["r"] = {"WCR": Mr["y"] - Mr["X"] @ br, "WCU": Mu["y"] - Mu["X"] @ (Pu @ Mu["y"])}
        if "WCR3" in kinds:
            P["r"]["WCR3"] = _cv3_resid(Mr, P["r"]["WCR"])
        assert np.ptp(P["b0"]) < 1e-10, "restricted fit must give one common slope"
        # check: Pb reproduces the per-field OLS point estimates
        assert np.allclose(P["obs"], np.linalg.lstsq(Mu["X"], Mu["y"], rcond=None)[0][bidx], atol=1e-8)
        Ps.append(P)
    g0, fl0 = Ps[0]["g"], list(pairs[0][1]["fields"])
    for (Mr, Mu), P in zip(pairs, Ps):
        assert np.array_equal(P["g"], g0) and list(Mu["fields"]) == fl0 and Mu["d"].inst_key.equals(pairs[0][1]["d"].inst_key)
    G, k, m = Ps[0]["G"], len(fl0), len(Ps)
    out = dict(fields=fl0, B=B, obs=np.column_stack([P["obs"] for P in Ps]), b0=np.array([P["b0"][0] for P in Ps]),
               n_cells=len(g0), n_inst=G, kinds=list(kinds))
    for kind in kinds:
        R = np.column_stack([P["r"][kind] for P in Ps])
        out[f"rescorr_{kind}"] = np.corrcoef(R.T) if m > 1 else None
    if cr1:
        out["cr1_obs"] = np.column_stack([_cr1_f(P, P["r"]["WCU"]) for P in Ps])
    rng = np.random.default_rng(seed)
    W = rng.choice([-1.0, 1.0], size=(B, G))                  # one set of institution weights, used for every kind
    for kind in kinds:
        nb = np.empty((B, k, m))
        ns = np.empty((B, k, m)) if cr1 else None
        for bi in range(B):
            wv = W[bi][g0]
            for j, P in enumerate(Ps):
                u = wv * P["r"][kind]
                nb[bi, :, j] = P["b0"] + P["Pb"] @ u
                if cr1:
                    ns[bi, :, j] = _cr1_f(P, u)
        out[kind] = nb
        if cr1:
            out[f"cr1_{kind}"] = ns
    del Ps
    return out


def _qfix(b, s):
    """Cochran's Q with fixed weights 1/s^2 along the last axis (same statistic as reml()'s Q)."""
    w = 1 / s ** 2
    mu = (w * b).sum(-1, keepdims=True) / w.sum(-1, keepdims=True)
    return (w * (b - mu) ** 2).sum(-1)


def wild_p(null, obs):
    """One-sided p (upper tail) of an observed statistic against its bootstrap null, (1 + #{null >= obs}) / (1 + B)."""
    null = np.asarray(null, float)
    return float((1 + np.sum(null >= obs - 1e-12)) / (1 + len(null)))


WILD_KINDS = ("WCR", "WCU", "WCR3")       # restricted, per-field (unrestricted) and leverage-corrected restricted residuals


def _is_eng(f):
    return "engineering" in f or f == "computer_science"


def wild_tests(cells, FEM, FE, B: int) -> dict:
    """(1) Cross-horizon correlation of the per-field beta_f (main spec, common programs) against its null with one
    common beta per horizon: the horizons reuse the same programs, whose residuals persist across cohorts, so the
    beta_f correlate across horizons even without field differences. Same institution weights at every horizon.
    (2) Heterogeneity (Cochran's Q) of beta_f with a wild-cluster null (common beta imposed): 4-yr main spec (all
    fields; fields with >= PF_NMIN_MID and >= PF_SPLIT_NMIN programs; >= PF_NMIN_MID without computer science and
    engineering), each horizon and the 4/1/5-yr average on the common programs. Every null is computed with plain
    restricted residuals (WCR), per-field residuals (WCU) and leverage-corrected restricted residuals (WCR3).
    (3) Counts of fields with beta_f / SE > 1.96 against a wild-cluster null with beta_f = 0 for every field."""
    out = {"B": B, "kinds": WILD_KINDS}
    pf = FE["FE2_pf"]
    # ---- 4-yr main spec, all fields ----
    s_, n_, y_ = FE_SPECS["FE2"]
    Mu = fe_build(cells, s_, n_, y_, per_field=True)
    nby = Mu["d"].groupby("field").size()
    wq = wild_pf([(FEM["FE2"], Mu)], [SEED, 22], B, cr1=True, kinds=WILD_KINDS)
    fl = wq["fields"]
    assert list(pf["beta_f"].index) == fl and np.allclose(wq["obs"][:, 0], pf["beta_f"].to_numpy(), atol=1e-10)
    assert np.allclose(wq["cr1_obs"][:, 0], pf["se_f"].to_numpy(), rtol=1e-6), "CR1 SEs do not match statsmodels"
    big = [f for f in fl if nby[f] >= PF_SPLIT_NMIN]
    mid = [f for f in fl if nby[f] >= PF_NMIN_MID]
    mid_noeng = [f for f in mid if not _is_eng(f)]
    se_fix = {"iqr": pf["se_iqr"].reindex(fl).to_numpy(float), "sd": pf["se_boot"].reindex(fl).to_numpy(float)}
    Q = {}
    for set_, fs in (("all", fl), ("mid", mid), ("big", big), ("mid_noeng", mid_noeng)):
        ix = np.array([fl.index(f) for f in fs])
        b_obs = wq["obs"][ix, 0]
        for wt in ("iqr", "sd", "cr1"):
            s_obs = se_fix[wt][ix] if wt != "cr1" else wq["cr1_obs"][ix, 0]
            q_obs = float(_qfix(b_obs, s_obs))
            r_ = dict(k=len(ix), Q=q_obs, chi2_p=float(chi2.sf(q_obs, len(ix) - 1)))
            for kind in WILD_KINDS:
                bn = wq[kind][:, ix, 0]
                sn = se_fix[wt][ix][None, :] if wt != "cr1" else wq[f"cr1_{kind}"][:, ix, 0]
                qn = _qfix(bn, sn)
                r_[kind] = dict(p=wild_p(qn, q_obs), mean=float(qn.mean()), p95=float(np.percentile(qn, 95)))
            Q[(set_, wt)] = r_
    out["Q4"], out["Q4_big"], out["Q4_mid"], out["Q4_mid_noeng"] = Q, big, mid, mid_noeng
    # how wide is each wild null per field, relative to the pairs-bootstrap spread (IQR-based SE)? by field size
    nb_ = nby.reindex(fl).to_numpy()
    ratio = {}
    for kind in WILD_KINDS:
        rr = wq[kind][:, :, 0].std(axis=0, ddof=1) / se_fix["iqr"]
        ratio[kind] = {lab: float(np.median(rr[msk])) for lab, msk in
                       (("lt30", nb_ < PF_NMIN_MID), ("30_59", (nb_ >= PF_NMIN_MID) & (nb_ < PF_SPLIT_NMIN)),
                        ("ge60", nb_ >= PF_SPLIT_NMIN))}
    out["sd_ratio"] = ratio
    out["n_size"] = {"lt30": int(np.sum(nb_ < PF_NMIN_MID)),
                     "30_59": int(np.sum((nb_ >= PF_NMIN_MID) & (nb_ < PF_SPLIT_NMIN))),
                     "ge60": int(np.sum(nb_ >= PF_SPLIT_NMIN))}
    # counts of fields with beta_f / SE_IQR > 1.96 (and < -1.96) under the null beta_f = 0 for every field
    wc = wild_pf([(_drop_beta(FEM["FE2"]), Mu)], [SEED, 23], B, kinds=WILD_KINDS)
    assert np.allclose(wc["obs"][:, 0], wq["obs"][:, 0], atol=1e-10)
    s_iqr = se_fix["iqr"]
    z_obs = wq["obs"][:, 0] / s_iqr
    cnt = {}
    for kind in WILD_KINDS:
        zn = wc[kind][:, :, 0] / s_iqr[None, :]
        cp, cn_ = (zn > 1.96).sum(1), (zn < -1.96).sum(1)
        cnt[kind] = dict(p_pos=wild_p(cp, int((z_obs > 1.96).sum())), p_neg=wild_p(cn_, int((z_obs < -1.96).sum())),
                         mean_pos=float(cp.mean()), mean_neg=float(cn_.mean()), p95_pos=float(np.percentile(cp, 95)))
    out["cnt4"] = dict(k=len(fl), n_pos=int((z_obs > 1.96).sum()), n_neg=int((z_obs < -1.96).sum()), **cnt)
    del Mu, wq, wc
    # ---- common programs: the three horizons ----
    Mus = []
    for n in HZ_FE_PF:
        s_, n_, y_ = FE_SPECS[n]
        Mus.append(fe_build(cells, s_, n_, y_, per_field=True))
    wh = wild_pf([(FEM[n], M_) for n, M_ in zip(HZ_FE_PF, Mus)], [SEED, 21], B, kinds=WILD_KINDS)
    fl3 = wh["fields"]
    for j, n in enumerate(HZ_FE_PF):
        assert list(FE[n + "_pf"]["beta_f"].index) == fl3
        assert np.allclose(wh["obs"][:, j], FE[n + "_pf"]["beta_f"].to_numpy(), atol=1e-10)
        assert abs(wh["b0"][j] - FE[n]["beta"]) < 1e-10          # restricted fit = the common-beta model
    prs = list(combinations(range(len(HZ_FE_PF)), 2))
    corr = {}
    for a, b in prs:
        key = (HZ_FE_PF[a], HZ_FE_PF[b])
        oa, ob = wh["obs"][:, a], wh["obs"][:, b]
        r_ = dict(r=float(np.corrcoef(oa, ob)[0, 1]), rs=float(spearmanr(oa, ob)[0]),
                  rescorr_WCR=float(wh["rescorr_WCR"][a, b]), rescorr_WCU=float(wh["rescorr_WCU"][a, b]))
        for kind in WILD_KINDS:
            na_, nb_ = wh[kind][:, :, a], wh[kind][:, :, b]
            rn = s55.rowwise_pearson(na_, nb_)
            rsn = s55.rowwise_spear(na_, nb_)
            r_[kind] = dict(mean=float(np.mean(rn)), p95=float(np.percentile(rn, 95)), p=wild_p(rn, r_["r"]),
                            mean_s=float(np.mean(rsn)), p95_s=float(np.percentile(rsn, 95)), p_s=wild_p(rsn, r_["rs"]))
        corr[key] = r_
    out["hz_corr"] = corr
    out["hz_cells"], out["hz_inst"], out["hz_k"] = wh["n_cells"], wh["n_inst"], len(fl3)
    # heterogeneity per horizon (IQR-based bootstrap SEs of that horizon, fixed) and of the 4/1/5-yr average
    PF = FE.get("pf_h3avg")
    Qh = {}
    for j, n in enumerate(HZ_FE_PF + ["avg"]):
        if n == "avg":
            if PF is None:
                continue
            s_ = PF["se_iqr"].reindex(fl3).to_numpy(float)
            b_obs = wh["obs"].mean(1)
            nulls = {kind: _qfix(wh[kind].mean(2), s_[None, :]) for kind in WILD_KINDS}
        else:
            s_ = FE[n + "_pf"]["se_iqr"].reindex(fl3).to_numpy(float)
            b_obs = wh["obs"][:, j]
            nulls = {kind: _qfix(wh[kind][:, :, j], s_[None, :]) for kind in WILD_KINDS}
        q_obs = float(_qfix(b_obs, s_))
        Qh[n] = dict(k=len(fl3), Q=q_obs, chi2_p=float(chi2.sf(q_obs, len(fl3) - 1)),
                     **{kind: dict(p=wild_p(v, q_obs), mean=float(v.mean()), p95=float(np.percentile(v, 95)))
                        for kind, v in nulls.items()})
    out["Qh"] = Qh
    del wh
    # counts under beta_f = 0, 4/1/5-yr average (pooled per-field test; z = average / its IQR-based bootstrap SE)
    if PF is not None:
        wc3 = wild_pf([(_drop_beta(FEM[n]), M_) for n, M_ in zip(HZ_FE_PF, Mus)], [SEED, 24], B, kinds=WILD_KINDS)
        s_ = PF["se_iqr"].reindex(fl3).to_numpy(float)
        z_obs = wc3["obs"].mean(1) / s_
        cnt = {}
        for kind in WILD_KINDS:
            zn = wc3[kind].mean(2) / s_[None, :]
            cp, cn_ = (zn > 1.96).sum(1), (zn < -1.96).sum(1)
            cnt[kind] = dict(p_pos=wild_p(cp, int((z_obs > 1.96).sum())), p_neg=wild_p(cn_, int((z_obs < -1.96).sum())),
                             mean_pos=float(cp.mean()), mean_neg=float(cn_.mean()), p95_pos=float(np.percentile(cp, 95)))
        out["cnt_avg"] = dict(k=len(fl3), n_pos=int((z_obs > 1.96).sum()), n_neg=int((z_obs < -1.96).sum()), **cnt)
        del wc3
    del Mus
    return out


def nonlinear_het(cells, FE, FEM, inst_index, splits, H_main, perms, b_pf, b_wild) -> dict:
    """Is the cross-field heterogeneity of the within-institution beta_f robust to NONLINEAR field-specific returns
    to brand and selectivity (convex, elite-tail pricing of brand is itself a reading-B mechanism that a linear
    field x G slope cannot absorb)? For the main spec (FE2) and FE2cg (+ field x {zG^2, zG^3}) / FE2ca (+ field x
    cubic in G, SAT, ADM), all on the same cells: per-field beta_f; institution pairs bootstrap with the SAME draws
    as the main spec's per-field bootstrap (seed [SEED, 5]); REML tau and chi2 Q (bootstrap-SD and IQR-based SEs);
    wild-cluster Q with the same Rademacher signs as wild_tests (seed [SEED, 22]; WCR, WCU, WCR3); split-half
    reliability on the same institution splits and field permutations as the main split-half; computer science and
    the CS/engineering-vs-other contrast. H_main / perms: the main spec's split-half estimates and permutations."""
    out = {}
    d0 = FEM["FE2"]["d"]
    base = None
    for name in NL_SPECS:
        slopes, need, yvar = FE_SPECS[name]
        Mu = fe_build(cells, slopes, need, yvar, per_field=True)
        Mr = FEM[name]
        assert Mu["d"].inst_key.equals(d0.inst_key) and Mu["d"].field.equals(d0.field), "nonlinear specs: same cells"
        fl = list(Mu["fields"])
        nby = Mu["d"].groupby("field").size().reindex(fl)
        fit = fe_fit(Mu)
        md = NL_MIN_DISTINCT[name]
        if name == "FE2":
            assert md == FE_PF_MIN_DISTINCT
            bb = FE["FE2_pf"]["boot"]                     # the main spec's per-field draws (seed [SEED, 5])
        else:
            bb = fe_boot_pf(Mu, [SEED, 5], b_pf, md)
        PB = pd.DataFrame(bb, columns=fl)
        bf = fit["beta_f"]
        se_sd = PB.std(ddof=1)
        se_iqr = (PB.quantile(0.75) - PB.quantile(0.25)) / 1.349
        r = dict(beta_f=bf, se_cr1=fit["se_f"], se_sd=se_sd, se_iqr=se_iqr, lo=PB.quantile(0.025),
                 hi=PB.quantile(0.975), fail=PB.isna().mean(), n_by_field=nby, n_par=fit["n_par"],
                 n_cells=fit["n_cells"], n_inst=fit["n_inst"], min_distinct=md, B=int(bb.shape[0]))
        # fields whose beta_f is defined in at least half of the bootstrap draws (with 12 field-specific parameters
        # the smallest fields rarely reach NL_MIN_DISTINCT distinct institutions in a draw; their spread is unknown)
        est = [f for f in fl if r["fail"][f] <= 0.5]
        r["not_est"] = [f for f in fl if f not in est]
        sets = {"all": est, "mid": [f for f in est if nby[f] >= PF_NMIN_MID],
                "big": [f for f in est if nby[f] >= PF_SPLIT_NMIN],
                "mid_noeng": [f for f in est if nby[f] >= PF_NMIN_MID and not _is_eng(f)]}
        assert sets["big"] == [f for f in fl if nby[f] >= PF_SPLIT_NMIN], "a split-half field is not estimable"
        het = {}
        for sn, fs in sets.items():
            for wt, se_ in (("sd", se_sd), ("iqr", se_iqr)):
                q = reml(bf[fs].to_numpy(float), se_[fs].to_numpy(float) ** 2)
                het[(sn, wt)] = dict(k=len(fs), tau=q["tau"], p=q["p"])
        if name == "FE2":
            # the main spec's wild-cluster Q was computed in wild_tests with the same signs: reuse it
            WQ4 = FE["wild"]["Q4"]
            for sn, key_ in (("all", None), ("mid", "Q4_mid"), ("big", "Q4_big"), ("mid_noeng", "Q4_mid_noeng")):
                assert sets[sn] == (fl if key_ is None else FE["wild"][key_])
                for wt in ("iqr", "sd"):
                    het[(sn, "wild_" + wt)] = WQ4[(sn, wt)]
        else:
            wq = wild_pf([(Mr, Mu)], [SEED, 22], b_wild, kinds=WILD_KINDS)
            assert np.allclose(wq["obs"][:, 0], bf.to_numpy(), atol=1e-10)
            for sn, fs in sets.items():
                ix = np.array([fl.index(f) for f in fs])
                for wt, se_ in (("iqr", se_iqr), ("sd", se_sd)):
                    s_obs = se_[fs].to_numpy(float)
                    q_obs = float(_qfix(bf[fs].to_numpy(float), s_obs))
                    d_ = dict(k=len(ix), Q=q_obs, chi2_p=float(chi2.sf(q_obs, len(ix) - 1)))
                    for kind in WILD_KINDS:
                        qn = _qfix(wq[kind][:, ix, 0], s_obs[None, :])
                        d_[kind] = dict(p=wild_p(qn, q_obs), mean=float(qn.mean()), p95=float(np.percentile(qn, 95)))
                    het[(sn, "wild_" + wt)] = d_
            del wq
        r["het"], r["sets"] = het, sets
        for sn in ("all", "mid"):
            fs = sets[sn]
            E = [f for f in fs if _is_eng(f)]
            O = [f for f in fs if not _is_eng(f)]
            dg = PB[E].mean(axis=1) - PB[O].mean(axis=1)
            r[f"eng_{sn}"] = dict(kE=len(E), kO=len(O), diff=float(bf[E].mean() - bf[O].mean()),
                                  lo=float(dg.quantile(0.025)), hi=float(dg.quantile(0.975)))
        # split-half on the same splits and the same field permutations as the main spec's variants
        if name == "FE2":
            H = H_main
        else:
            keys_ = pd.factorize(Mu["d"].inst_key)[1]
            inst_univ = np.array([inst_index[k_] for k_ in keys_])
            H = fe_pf_halves(Mu, inst_univ, splits, md)
        big = sets["big"]
        sp = {}
        for tag, fs in (("main", big), ("main_nocs", [f for f in big if f != "computer_science"]),
                        ("main_noeng", [f for f in big if not _is_eng(f)])):
            sel = np.array([fl.index(f) for f in fs])
            assert len(perms[tag][0]) == len(sel)
            sp[tag] = pf_split_stats(H, sel, perms[tag])
        r["split"] = sp
        del H
        if base is None:
            base = bf
        for sn in ("all", "mid"):
            fs = sets[sn]
            r[f"corr_main_{sn}"] = float(np.corrcoef(bf[fs].to_numpy(), base.reindex(fs).to_numpy())[0, 1])
        out[name] = r
        del Mu, PB
    return out


def oster(FEM) -> dict:
    """Oster (2019) coefficient-stability bound for the common within-institution beta. 'Short' models
    on the long model's cells: (s0) gamma_f + beta z (no institution FE); (s1) alpha_i + gamma_f + beta z.
    Long: FE2 (alpha_i + gamma_f + beta z + field x {SAT, ADM, inst. Pell, brand G}). R2 are total R2
    (institution FE included where present). delta* = selection on unobservables, relative to the
    observables added between short and long, that would drive beta to 0 at R_max; beta*(delta=1)."""
    L = FEM["FE2"]
    d = L["d"]
    y = np.log(d["earnings"].to_numpy(float))
    sst = float(((y - y.mean()) ** 2).sum())
    bl = np.linalg.lstsq(L["X"], L["y"], rcond=None)[0]
    jl = L["names"].index("beta")
    r_long = 1 - float(((L["y"] - L["X"] @ bl) ** 2).sum()) / sst
    S1 = FEM["FE0_s"]
    assert S1["d"].inst_key.equals(d.inst_key) and S1["d"].field.equals(d.field)
    b1 = np.linalg.lstsq(S1["X"], S1["y"], rcond=None)[0]
    r_s1 = 1 - float(((S1["y"] - S1["X"] @ b1) ** 2).sum()) / sst
    D = np.eye(len(L["fields"]))[L["code"]]
    X0 = np.column_stack([D, L["z"]])
    b0 = np.linalg.lstsq(X0, y, rcond=None)[0]
    r_s0 = 1 - float(((y - X0 @ b0) ** 2).sum()) / sst
    out = dict(beta_long=float(bl[jl]), r_long=r_long, n=len(y))
    for tag, bs, rs in (("s0", float(b0[-1]), r_s0), ("s1", float(b1[S1["names"].index("beta")]), r_s1)):
        out[f"beta_{tag}"], out[f"r_{tag}"] = bs, rs
        for rtag, rmax in (("13", min(1.0, 1.3 * r_long)), ("half", r_long + (1 - r_long) / 2)):
            den = (bs - out["beta_long"]) * (rmax - r_long)
            out[f"delta_{tag}_{rtag}"] = float(out["beta_long"] * (r_long - rs) / den) if den != 0 else np.nan
            out[f"bstar_{tag}_{rtag}"] = float(out["beta_long"] - (bs - out["beta_long"]) * (rmax - r_long) / (r_long - rs))
            out[f"rmax_{rtag}"] = rmax
    return out


def composition_balance(cells):
    """Within institution: does department prestige predict the program's Pell share?
    z(PPELL) = alpha_i + gamma_f + b z(F) (composition sample); SE clustered by institution."""
    d = fe_sample(cells, INST_SLOPES + ["G"] + COMP_NEED, "earnings")
    fields = sorted(d.field.unique())
    D = np.eye(len(fields))[pd.Categorical(d.field, categories=fields).codes]
    g = d.groupby("field")
    z = g.prestige_score.transform(lambda s: (s - s.mean()) / s.std()).to_numpy()
    zp = g.PPELL.transform(lambda s: (s - s.mean()) / s.std()).to_numpy()
    ic = pd.factorize(d.inst_key)[0]
    X = np.column_stack([D[:, 1:], z])
    Xd = X - pd.DataFrame(X).groupby(ic).transform("mean").to_numpy()
    yd = zp - pd.Series(zp).groupby(ic).transform("mean").to_numpy()
    fit = sm.OLS(yd, Xd).fit(cov_type="cluster", cov_kwds={"groups": ic})
    Xo = np.column_stack([D, z])
    fo = sm.OLS(zp, Xo).fit(cov_type="cluster", cov_kwds={"groups": ic})
    return dict(b=float(fit.params[-1]), se=float(fit.bse[-1]), b_noFE=float(fo.params[-1]),
                se_noFE=float(fo.bse[-1]), n=len(d), n_inst=int(ic.max() + 1))


# ---------------------------------------------------------------------------------------------
# split-half: is there reproducible cross-field variation in the field-specific component?
# ---------------------------------------------------------------------------------------------
def split_half(FD, fields, N, rng):
    fl = [f for f in fields if int(FD[f]["mask"]["sat"].sum()) >= SPLIT_NMIN]
    k = len(fl)
    names = ["raw", "sat_earn", "broad", "F_broadG", "G_broadF"]
    H = {n: np.full((SPLIT_K, 2, k), np.nan) for n in names}
    for s in range(SPLIT_K):
        inA = np.zeros(N, bool); inA[rng.permutation(N)[: N // 2]] = True
        for j, f in enumerate(fl):
            fd = FD[f]; a = inA[fd["inst_idx"]]
            for h, rows in enumerate((np.where(a)[0], np.where(~a)[0])):
                st = field_stats(fd, rows, hr=False, only=names)
                for n in names:
                    H[n][s, h, j] = st.get(n, np.nan)
    rs = s55.rowwise_spear
    rp = s55.rowwise_pearson
    out = dict(k=k, fields=fl)
    perms = [rng.permutation(k) for _ in range(N_PERM)]
    for n in ["broad", "F_broadG", "G_broadF"]:
        A, Bm = H[n][:, 0, :], H[n][:, 1, :]
        rel = rp(A, Bm)
        obs = float(np.nanmean(rel))
        null = np.array([np.nanmean(rp(A, Bm[:, pi])) for pi in perms])
        out[f"rel_{n}"] = obs
        out[f"rel_{n}_p"] = float((1 + np.sum(null >= obs)) / (1 + N_PERM))
        out[f"rel_{n}_null95"] = float(np.percentile(null, 95))
        out[f"undef_{n}"] = float(np.mean(~np.isfinite(A)))
        for other in ["raw", "sat_earn"]:
            OA, OB = H[other][:, 0, :], H[other][:, 1, :]
            cross = (rs(A, OB) + rs(Bm, OA)) / 2
            obs_c = float(np.nanmean(cross))
            null_c = np.array([np.nanmean((rs(A, OB[:, pi]) + rs(Bm, OA[:, pi])) / 2) for pi in perms[:N_PERM_CROSS]])
            out[f"cross_{n}_{other}"] = obs_c
            out[f"cross_{n}_{other}_p"] = float((1 + np.sum(null_c >= obs_c)) / (1 + len(null_c)))
            out[f"crossP_{n}_{other}"] = float(np.nanmean((rp(A, OB) + rp(Bm, OA)) / 2))
    for other in ["raw", "sat_earn"]:
        out[f"rel_{other}"] = float(np.nanmean(rp(H[other][:, 0, :], H[other][:, 1, :])))
    # disattenuated (true-score) cross-field correlations, Pearson throughout; only meaningful when both
    # split-half reliabilities are clearly positive
    for n in ["broad", "F_broadG", "G_broadF"]:
        for other in ["raw", "sat_earn"]:
            den = out[f"rel_{n}"] * out[f"rel_{other}"]
            out[f"ts_{n}_{other}"] = out[f"crossP_{n}_{other}"] / np.sqrt(den) if den > 0 else np.nan
    return out


def fe_pf_halves(M, inst_univ: np.ndarray, splits: np.ndarray, min_distinct: int) -> np.ndarray:
    """Per-field beta_f of a per_field=True within-institution model re-estimated on each half of each random
    split of the institution universe. Rows are already demeaned within institution, so keeping whole
    institutions keeps the institution FE exact. inst_univ: universe index of each cluster; splits: (K, N) bool
    (True = half A). Returns (K, 2, k); NaN where a field has < min_distinct institutions in that half."""
    G = int(M["cnt"].size)
    idx = [np.where(M["groups"] == g)[0] for g in range(G)]
    X, y = M["X"], M["y"]
    k = len(M["fields"])
    bidx = np.array([M["names"].index(f"b_{f}") for f in M["fields"]])
    inst_field = np.zeros((G, k), bool)
    inst_field[M["groups"], M["code"]] = True
    A = np.stack([X[r].T @ X[r] for r in idx]); bv = np.stack([X[r].T @ y[r] for r in idx])
    p = X.shape[1]
    out = np.full((splits.shape[0], 2, k), np.nan)
    for s in range(splits.shape[0]):
        inA = splits[s][inst_univ]
        for h, c in enumerate((inA, ~inA)):
            c = c.astype(float)
            XtX = np.tensordot(c, A, axes=1); Xty = c @ bv
            keep = np.diag(XtX) > 1e-10
            sol = np.linalg.lstsq(XtX[np.ix_(keep, keep)], Xty[keep], rcond=None)[0]
            full = np.full(p, np.nan); full[keep] = sol
            ok = (inst_field & (c > 0)[:, None]).sum(0) >= min_distinct
            out[s, h] = np.where(ok, full[bidx], np.nan)
    del A, bv
    return out


def pf_split_stats(H: np.ndarray, sel: np.ndarray, perms: list) -> dict:
    """Split-half reliability of per-field estimates H (K, 2, k) over the fields `sel` (column indices):
    mean over splits of the cross-field Pearson / Spearman correlation between halves; permutation p from
    relabelling the fields of half B (the same permutation applied in every split)."""
    A_, B_ = H[:, 0, sel], H[:, 1, sel]
    rp, rs = s55.rowwise_pearson, s55.rowwise_spear
    obs, obs_s = float(np.nanmean(rp(A_, B_))), float(np.nanmean(rs(A_, B_)))
    null = np.array([np.nanmean(rp(A_, B_[:, pi])) for pi in perms])
    null_s = np.array([np.nanmean(rs(A_, B_[:, pi])) for pi in perms])
    return dict(k=len(sel), rel=obs, p=float((1 + np.sum(null >= obs)) / (1 + len(perms))),
                rel_s=obs_s, p_s=float((1 + np.sum(null_s >= obs_s)) / (1 + len(perms))),
                null95=float(np.percentile(null, 95)), undef=float(np.mean(~np.isfinite(A_))))


def splithalf_power(se: np.ndarray, taus: np.ndarray, seed, n_sim=POW_NSIM, n_split=POW_NSPLIT) -> pd.DataFrame:
    """What split-half reliability a given between-field SD tau would produce, given the fields' observed
    full-sample bootstrap SEs. Model: true theta_f ~ N(0, tau^2); full-sample estimate theta_f + e_f,
    e_f ~ N(0, se_f^2); the two half-sample estimates are theta_f + e_f +- d_f with d_f ~ N(0, se_f^2), so each
    half has sampling variance 2 se_f^2 and the halves are independent given theta. The statistic is the mean
    over n_split splits of the cross-field Pearson correlation between halves, as in the data. Fields are
    treated as independent (they share institutions in the data). Common random numbers across tau.
    Returns per tau: mean, 2.5 / 97.5 percentiles and power against the simulated tau = 0 95th percentile."""
    rng = np.random.default_rng(seed)
    k = len(se)
    zt = rng.standard_normal((n_sim, 1, k))
    ze = rng.standard_normal((n_sim, 1, k))
    zd = rng.standard_normal((n_sim, n_split, k))
    base = ze * se + 0.0
    dd = zd * se

    def stat(tau):
        c = tau * zt + base
        A_, B_ = c + dd, c - dd
        A_ = A_ - A_.mean(-1, keepdims=True); B_ = B_ - B_.mean(-1, keepdims=True)
        r = (A_ * B_).sum(-1) / np.sqrt((A_ * A_).sum(-1) * (B_ * B_).sum(-1))
        return r.mean(-1)

    null = stat(0.0)
    crit = float(np.percentile(null, 95))
    rows = []
    for t in taus:
        v = null if t == 0 else stat(float(t))
        rows.append(dict(tau=float(t), mean=float(v.mean()), lo=float(np.percentile(v, 2.5)),
                         hi=float(np.percentile(v, 97.5)), power=float(np.mean(v > crit)), crit=crit))
    return pd.DataFrame(rows), null


def power_summary(tab: pd.DataFrame, null: np.ndarray, obs: float, at: dict) -> dict:
    """tau range consistent with the observed statistic (observed inside the simulated 2.5-97.5% band), smallest
    grid tau with >= 80% power, the expected reliability at named tau values (rows of `tab`), and the p-value of
    the observed statistic against the simulated tau = 0 distribution. Unlike the field-label permutation null,
    which conditions on the realized full-sample estimates, the simulated null also lets the dispersion of those
    estimates vary by chance, so it is wider and its p-value more conservative."""
    ok = tab[(tab["lo"] <= obs) & (obs <= tab["hi"])]
    p80 = tab[tab["power"] >= 0.8]
    out = dict(obs=obs, crit=float(tab["crit"].iloc[0]),
               p_sim=float((1 + np.sum(null >= obs)) / (1 + len(null))),
               tau_lo=float(ok["tau"].min()) if len(ok) else np.nan,
               tau_hi=float(ok["tau"].max()) if len(ok) else np.nan,
               tau_p80=float(p80["tau"].min()) if len(p80) else np.nan,
               grid_max=float(tab["tau"].max()))
    for name, t in at.items():
        r = tab.iloc[int(np.argmin(np.abs(tab["tau"].to_numpy() - t)))]
        out[f"exp_{name}"] = float(r["mean"])
        out[f"pow_{name}"] = float(r["power"])
        out[f"tau_{name}"] = float(t)
    return out


# ---------------------------------------------------------------------------------------------
# (v) availability of separating data
# ---------------------------------------------------------------------------------------------
def availability():
    fos_cols = pd.read_csv(FOS_FILE, nrows=0).columns.tolist()
    inst_cols = pd.read_csv(INST_FILE, nrows=0).columns.tolist()
    pat = lambda c: any(t in c for t in ("SAT", "ACT")) and not any(t in c for t in ("STATE", "SATIS"))
    out = dict(fos_ncols=len(fos_cols), fos_test_cols=[c for c in fos_cols if pat(c)],
               inst_test_cols=[c for c in inst_cols if pat(c)][:12],
               inst_test_ncols=len([c for c in inst_cols if pat(c)]))
    use = ["tax_year", "YAG", "provider_name", "cah2_code", "current_region_name", "earnings_median",
           "characteristic_type", "characteristic_value"]
    parts = []
    with zipfile.ZipFile(LEO_ZIP) as z, z.open("provider_data_20250716.csv") as fh:
        for ch in pd.read_csv(fh, encoding="latin-1", dtype=str, usecols=use, chunksize=500000):
            ch = ch[(ch.provider_name != "Total") & (ch.cah2_code != "Total")
                    & (ch.current_region_name == "Total")
                    & ch.characteristic_type.isin(["prior_attainment_code", "All graduates"])]
            ch = ch[pd.to_numeric(ch.earnings_median, errors="coerce").notna()]
            parts.append(ch[["tax_year", "YAG", "provider_name", "cah2_code", "characteristic_type",
                             "characteristic_value"]])
    L = pd.concat(parts, ignore_index=True)
    pa = L[(L.characteristic_type == "prior_attainment_code") & (L.characteristic_value != "Not known")]
    out["leo_pa_rows"] = int(len(pa))
    out["leo_bands"] = int(pa.characteristic_value.nunique())
    y5 = pa[pa.YAG == "5"]
    ty = sorted(y5.tax_year.unique())[-1]
    q = y5[y5.tax_year == ty]
    out["leo_tax_year"] = ty
    out["leo_pa_rows_y5_latest"] = int(len(q))
    out["leo_providers"] = int(q.provider_name.nunique())
    out["leo_subjects"] = int(q.cah2_code.nunique())
    cellcnt = q.groupby(["cah2_code", "characteristic_value"]).provider_name.nunique()
    out["leo_band_cells_ge15"] = int((cellcnt >= 15).sum())
    out["leo_band_cells"] = int(len(cellcnt))
    out["leo_subjects_with_ge15_band"] = int(cellcnt[cellcnt >= 15].reset_index().cah2_code.nunique())
    bands = q.groupby(["provider_name", "cah2_code"]).characteristic_value.nunique()
    out["leo_prov_subj_ge2bands"] = int((bands >= 2).sum())
    out["leo_prov_subj"] = int(len(bands))
    allg = L[(L.characteristic_type == "All graduates") & (L.YAG == "5") & (L.tax_year == ty)]
    out["leo_all_prov_subj"] = int(allg.groupby(["provider_name", "cah2_code"]).ngroups)
    return out


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------
def compute(b_boot=B_BOOT, b_fe=B_FE, split_k=None, do_avail=True):
    global SPLIT_K
    if split_k is not None:
        SPLIT_K = split_k
    rng = np.random.default_rng(SEED)
    ar = load_ar_wapman(fields=FIELDS66)
    er = load_er_scorecard("undergrad", fields=FIELDS66)

    # --- baseline reproduction ---
    gm = compute_gap_map(ar, er, "undergrad", B=B_GAP, seed=SEED)
    ref = pd.read_csv(GAP_MAP)
    chk = ref[["field", "n_institutions", "spearman", "reliable"]].merge(
        gm[["field", "n_institutions", "spearman"]], on="field", how="left", suffixes=("", "_re"))
    repro_maxdiff = float((chk.spearman - chk.spearman_re).abs().max())
    assert repro_maxdiff < 1e-9, "baseline did not reproduce"

    inst = s55.load_institutions()
    cells = s55.build_cells(ar, er, inst)
    n0 = len(cells)
    comp = load_composition()
    cells = cells.merge(comp, on=["field", "inst_key"], how="left")
    # other earnings horizons of the same release (1-yr, 5-yr medians; different graduating cohorts)
    for tag, ec, cc in HORIZONS:
        e = load_er_scorecard("undergrad", earn_col=ec, count_col=cc, fields=FIELDS66)
        cells = cells.merge(e[["inst_key", "field", "earnings", "cohort_n"]].rename(
                                columns={"earnings": tag, "cohort_n": f"{tag}_n"}),
                            on=["field", "inst_key"], how="left")
        del e
    assert len(cells) == n0 and cells.INSTNM.notna().all()
    assert "cohort_n" in cells and cells.cohort_n.notna().all()     # 4-yr program count (weights)
    # private control (CONTROL 2 or 3) for the field x {state level, private} within-institution variant
    cells["PRIVATE"] = (cells.CONTROL != 1).astype(float).where(cells.CONTROL.notna())
    # squared / cubed within-field z-scores of brand, SAT and admit rate for the nonlinear specs FE2cg / FE2ca
    for c_ in POLY_VARS:
        z_ = cells.groupby("field")[c_].transform(lambda s: (s - s.mean()) / s.std())
        cells[f"{c_}_2"], cells[f"{c_}_3"] = z_ ** 2, z_ ** 3
    universe = sorted(cells.inst_key.unique())
    inst_index = {k: i for i, k in enumerate(universe)}
    N = len(universe)
    fields = sorted(cells.field.unique())
    FD = {f: field_arrays(cells[cells.field == f], inst_index) for f in fields}

    # --- point estimates ---
    point = {f: field_stats(FD[f], np.arange(FD[f]["n"]), point=True) for f in fields}
    keys = sorted({k for f in fields for k in field_stats(FD[f], np.arange(FD[f]["n"])).keys()})

    # --- joint institution bootstrap ---
    boot = {k: np.full((b_boot, len(fields)), np.nan) for k in keys}
    pr = np.full(N, 1.0 / N)
    for b in range(b_boot):
        cnt = rng.multinomial(N, pr)
        for j, f in enumerate(fields):
            fd = FD[f]
            rows = np.repeat(np.arange(fd["n"]), cnt[fd["inst_idx"]])
            if len(rows) < 4:
                continue
            for k, v in field_stats(fd, rows).items():
                boot[k][b, j] = v
    se = {k: np.nanstd(boot[k], axis=0, ddof=1) for k in keys}

    # --- reliability of field prestige (scripts/56) ---
    R = pd.read_csv(REL_CSV)
    R = R[R.unit == "field"].set_index("field")
    rel = {"LB": R["relF_LB"].to_dict(), "EXT": R["relF_EXT"].to_dict()}

    # --- per-field table ---
    s55t = pd.read_csv(S55_CSV).set_index("field") if S55_CSV.exists() else None
    relflag = ref.set_index("field")["reliable"].to_dict()
    rows = []
    for j, f in enumerate(fields):
        fd, pt = FD[f], point[f]
        r = dict(field=f, label=LAB.get(f, f), reliable=bool(relflag.get(f, False)),
                 licensed=f in F.LICENSED_FIELDS, n=fd["n"])
        for s in SAMPLES:
            r[f"n_{s}"] = int(fd["mask"][s].sum())
        for k in keys:
            r[k] = pt.get(k, np.nan)
            r[f"se_{k}"] = se[k][j] if np.isfinite(r[k]) else np.nan
        for name in SPECS:
            if f"df_{name}" in pt:
                r[f"df_{name}"] = pt[f"df_{name}"]
                r[f"r2x_{name}"] = pt[f"r2x_{name}"]
        for k in ["hr_n", "hr_leader", "hr3_leader", "hr_b_F", "hr_se_F", "hr_p_F", "hr_b_G", "hr_se_G",
                  "hr_p_G", "hr_cd_F_G", "hr_cd_F_S", "hr_cd_F_IE", "hr_cd_G_S", "hr5_n", "hr5_r2",
                  "hr5_sh_F", "hr5_sh_S", "hr5_sh_G", "hr5_sh_IE", "hr5_sh_C"]:
            r[k] = pt.get(k, np.nan)
        r["relF_LB"], r["relF_EXT"] = rel["LB"].get(f, np.nan), rel["EXT"].get(f, np.nan)
        # program composition descriptives (composition sample)
        m = fd["mask"]["comp"]
        if m.sum() >= NMIN:
            r["ppell_mean"] = float(np.mean(fd["PPELL"][m]))
            r["rho_ppell_pctpell"] = float(spearmanr(fd["PPELL"][m], fd["PCTPELL"][m])[0])
            r["rho_F_ppell"] = float(spearmanr(fd["P"][m], fd["PPELL"][m])[0])
            r["rho_ppell_Y"] = float(spearmanr(fd["PPELL"][m], fd["Y"][m])[0])
        rows.append(r)
    T = pd.DataFrame(rows)

    # horizon-pooled partial correlations: per field, the average of the 4-, 1- and 5-yr estimates on the common
    # programs; the bootstrap average uses the same joint draws (NaN in a draw if any horizon is undefined there)
    for fam in HZ_AVG_P:
        cols = [f"{fam}_h3{h}" for h in HZ3]
        bk = np.mean(np.stack([boot[c] for c in cols]), axis=0)
        boot[f"{fam}_h3avg"] = bk
        T[f"{fam}_h3avg"] = T[cols].mean(axis=1, skipna=False)
        T[f"se_{fam}_h3avg"] = np.where(np.isfinite(T[f"{fam}_h3avg"]), np.nanstd(bk, axis=0, ddof=1), np.nan)

    # reliability bracket for the residual field-prestige specs
    for name in PERP:
        if name not in T:
            continue
        r2 = T.get(f"r2x_{name}", pd.Series(0.0, index=T.index)).fillna(0.0) if SPECS[name][3] else 0.0
        for tag in ("LB", "EXT"):
            rp = 1 - (1 - T[f"relF_{tag}"]) / (1 - r2)
            rp = rp.where(rp >= REL_MIN)
            T[f"relperp_{tag}_{name}"] = rp
            T[f"dis_{tag}_{name}"] = T[name] / np.sqrt(rp)

    # --- reproduction checks against scripts/55 ---
    repro = {"baseline_maxdiff": repro_maxdiff}
    if s55t is not None:
        a = T.set_index("field")
        for mine, theirs in (("broad", "rho_full_stlev"), ("raw_sat", "rho_raw_sat"), ("adm", "rho_adm_stlev"),
                             ("sel", "rho_sel"),
                             ("hr55_sh_F", "hr_shap_prest"), ("hr55_sh_S", "hr_shap_sel"),
                             ("hr55_sh_IE", "hr_shap_ie")):
            x = a[mine].reindex(s55t.index); y = s55t[theirs]
            ok = x.notna() & y.notna()
            repro[f"{mine}_vs_{theirs}"] = (float((x[ok] - y[ok]).abs().max()), int(ok.sum()),
                                            int((x.notna() != y.notna()).sum()))
            assert repro[f"{mine}_vs_{theirs}"][0] < CSV_TOL, f"{mine} does not reproduce scripts/55 {theirs}"

    # --- (iii) within-institution designs ---
    FE, FEM = {}, {}
    for name, (slopes, need, yvar) in FE_SPECS.items():
        wc = FE_WEIGHT.get(name)
        FEM[name] = fe_build(cells, slopes, need, yvar, per_field=False, wcol=wc)
        FE[name] = fe_fit(FEM[name], rel if wc is None else None)     # no reliability correction under weights
    for a_, b_ in FE_ALIAS.items():                  # identical model: same object, so it also gets b_'s bootstrap
        FE[a_], FEM[a_] = FE[b_], FEM[b_]
    if b_fe > 0:
        for gi, grp in enumerate(FE_GROUPS):
            bb = fe_group_boot([FEM[n] for n in grp], [SEED, 3, gi], b_fe)
            for m, n in enumerate(grp):
                FE[n]["boot"] = bb[:, m]
                FE[n]["boot_se"] = float(np.nanstd(bb[:, m], ddof=1))
                FE[n]["boot_ci"] = (float(np.nanpercentile(bb[:, m], 2.5)), float(np.nanpercentile(bb[:, m], 97.5)))
                FE[n]["boot_fail"] = int(np.sum(~np.isfinite(bb[:, m])))
                fin_ = bb[np.isfinite(bb[:, m]), m]
                FE[n]["boot_p0"] = float(np.mean(fin_ <= 0))
        for a_, b_ in FE_PAIRS:
            dlt = FE[a_]["boot"] - FE[b_]["boot"]
            FE[f"pair_{a_}_{b_}"] = dict(diff=FE[a_]["beta"] - FE[b_]["beta"], se=float(np.nanstd(dlt, ddof=1)),
                                         ci=(float(np.nanpercentile(dlt, 2.5)), float(np.nanpercentile(dlt, 97.5))))
        # horizon-pooled within-institution beta on the common programs: average of the 4-, 1- and 5-yr betas, CI
        # from the average of the shared cluster-bootstrap draws (NaN in a draw if any horizon failed there)
        for fam in HZ_FE_FAM:
            nm = [f"{fam}_h3{h}" for h in HZ3]
            assert all(FEM[n]["d"].inst_key.equals(FEM[nm[0]]["d"].inst_key) for n in nm)
            bb = np.column_stack([FE[n]["boot"] for n in nm]).mean(axis=1)
            fin_ = bb[np.isfinite(bb)]
            r0 = FE[nm[0]]
            FE[f"{fam}_h3avg"] = dict(beta=float(np.mean([FE[n]["beta"] for n in nm])), boot=bb,
                                      boot_se=float(np.nanstd(bb, ddof=1)),
                                      boot_ci=(float(np.nanpercentile(bb, 2.5)), float(np.nanpercentile(bb, 97.5))),
                                      boot_p0=float(np.mean(fin_ <= 0)), boot_fail=int(np.sum(~np.isfinite(bb))),
                                      n_cells=r0["n_cells"], n_inst=r0["n_inst"], n_fields=r0["n_fields"])
        for a_, b_ in (("FE2w_h3avg", "FE2"), ("FE2v_h3avg", "FE2")):
            dlt = FE[a_]["boot"] - FE[f"{b_}_h3avg"]["boot"]
            FE[f"pair_{a_}_{b_}_h3avg"] = dict(diff=FE[a_]["beta"] - FE[f"{b_}_h3avg"]["beta"],
                                               se=float(np.nanstd(dlt, ddof=1)),
                                               ci=(float(np.nanpercentile(dlt, 2.5)), float(np.nanpercentile(dlt, 97.5))))
    for name in ["FE0", "FE2", "FE3", "FE3_np", "FE3_p"]:
        slopes, need, yvar = FE_SPECS[name]
        Mf = fe_build(cells, slopes, need, yvar, per_field=True)
        FE[name + "_pf"] = fe_fit(Mf)
        T[f"fe_{name}_beta"] = T.field.map(FE[name + "_pf"]["beta_f"])
        T[f"fe_{name}_se"] = T.field.map(FE[name + "_pf"]["se_f"])
        T[f"fe_{name}_ncells"] = T.field.map(FE[name + "_pf"]["n_by_field"])
        if name == "FE2" and b_fe > 0:
            # cluster-bootstrap SEs for the per-field betas: with ~280 parameters and ~170 clusters the
            # CR1 covariance is rank-deficient, so per-field CR1 SEs and joint tests are fragile
            bb = fe_boot_pf(Mf, [SEED, 5], b_fe, FE_PF_MIN_DISTINCT)
            FE["FE2_pf"]["boot_fail"] = pd.Series(np.mean(~np.isfinite(bb), axis=0), index=Mf["fields"])
            FE["FE2_pf"]["se_boot"] = pd.Series(np.nanstd(bb, axis=0, ddof=1), index=Mf["fields"])
            q25, q75 = np.nanpercentile(bb, 25, axis=0), np.nanpercentile(bb, 75, axis=0)
            FE["FE2_pf"]["se_iqr"] = pd.Series((q75 - q25) / 1.349, index=Mf["fields"])
            FE["FE2_pf"]["ci_lo"] = pd.Series(np.nanpercentile(bb, 2.5, axis=0), index=Mf["fields"])
            FE["FE2_pf"]["ci_hi"] = pd.Series(np.nanpercentile(bb, 97.5, axis=0), index=Mf["fields"])
            for c_ in ("se_boot", "se_iqr", "ci_lo", "ci_hi", "boot_fail"):
                T[f"fe_FE2_{c_}"] = T.field.map(FE["FE2_pf"][c_])
            FE["FE2_pf"]["boot"] = bb                    # kept for the nonlinear-pricing comparison (same draws)
            del bb
        del Mf
    # per-field beta_f of the main spec at each earnings horizon on the common programs; one seed for all
    # three, so the institution draws are shared and per-field horizon differences are paired
    HZB, HZG = {}, {}
    for name in HZ_FE_PF:
        slopes, need, yvar = FE_SPECS[name]
        Mf = fe_build(cells, slopes, need, yvar, per_field=True)
        HZG[name] = (Mf["groups"].copy(), Mf["d"].inst_key.copy(), Mf["d"].field.copy())
        r_ = fe_fit(Mf)
        if b_fe > 0:
            bb = fe_boot_pf(Mf, [SEED, 11], b_fe, FE_PF_MIN_DISTINCT)
            HZB[name] = pd.DataFrame(bb, columns=Mf["fields"])
            r_["se_boot"] = pd.Series(np.nanstd(bb, axis=0, ddof=1), index=Mf["fields"])
            q25, q75 = np.nanpercentile(bb, 25, axis=0), np.nanpercentile(bb, 75, axis=0)
            r_["se_iqr"] = pd.Series((q75 - q25) / 1.349, index=Mf["fields"])
            r_["ci_lo"] = pd.Series(np.nanpercentile(bb, 2.5, axis=0), index=Mf["fields"])
            r_["ci_hi"] = pd.Series(np.nanpercentile(bb, 97.5, axis=0), index=Mf["fields"])
            r_["boot_fail"] = pd.Series(np.mean(~np.isfinite(bb), axis=0), index=Mf["fields"])
            for c_ in ("se_iqr", "ci_lo", "ci_hi"):
                T[f"fe_{name}_{c_}"] = T.field.map(r_[c_])
            del bb
        FE[name + "_pf"] = r_
        T[f"fe_{name}_beta"] = T.field.map(r_["beta_f"])
        T[f"fe_{name}_ncells"] = T.field.map(r_["n_by_field"])
        del Mf
    if HZB:
        f0 = list(HZB[HZ_FE_PF[0]].columns)
        assert all(list(HZB[n].columns) == f0 for n in HZ_FE_PF), "horizon per-field models differ in fields"
        for a_ in HZ_FE_PF[1:]:
            dlt = HZB[a_] - HZB[HZ_FE_PF[0]]
            FE[f"pfdiff_{a_}"] = dict(diff=FE[a_ + "_pf"]["beta_f"] - FE[HZ_FE_PF[0] + "_pf"]["beta_f"],
                                      lo=dlt.quantile(0.025), hi=dlt.quantile(0.975))
        # horizon-pooled per-field beta_f: average of the three horizons, per draw (same cells, same draws, so a
        # field drops out of a draw at all three horizons or none); two-sided percentile-bootstrap p, BH over fields
        g0 = HZG[HZ_FE_PF[0]]
        for n in HZ_FE_PF[1:]:
            assert np.array_equal(HZG[n][0], g0[0]) and HZG[n][1].equals(g0[1]) and HZG[n][2].equals(g0[2])
        nan0 = HZB[HZ_FE_PF[0]].isna().to_numpy()
        assert all(np.array_equal(HZB[n].isna().to_numpy(), nan0) for n in HZ_FE_PF)
        A_ = sum(HZB[n] for n in HZ_FE_PF) / len(HZ_FE_PF)
        pt_ = sum(FE[n + "_pf"]["beta_f"] for n in HZ_FE_PF) / len(HZ_FE_PF)
        nfin = A_.notna().sum()
        c_le, c_ge = (A_ <= 0).sum(), (A_ >= 0).sum()
        n_far = np.minimum(c_le, c_ge)                       # draws on the far side of zero
        # raw percentile-bootstrap p (0 if no draw crosses zero: at B = 999 that only means p below ~0.002, while
        # the first BH threshold for 42 fields is 0.05/42 = 0.0012) and normal p from the IQR-based bootstrap SE
        p_ = np.minimum(1.0, 2 * n_far / nfin)
        se_iqr_ = (A_.quantile(0.75) - A_.quantile(0.25)) / 1.349
        p_n = pd.Series(2 * norm.sf(np.abs(pt_ / se_iqr_)), index=pt_.index)
        eng = [f for f in A_.columns if "engineering" in f or f == "computer_science"]
        oth = [f for f in A_.columns if f not in eng]
        grp_ = {}
        for tag, fl_ in (("all", list(A_.columns)), ("eng", eng), ("other", oth)):
            bm = A_[fl_].mean(axis=1)
            grp_[tag] = dict(k=len(fl_), fields=fl_, mean=float(pt_[fl_].mean()),
                             lo=float(bm.quantile(0.025)), hi=float(bm.quantile(0.975)), p0=float((bm <= 0).mean()))
        dg = A_[eng].mean(axis=1) - A_[oth].mean(axis=1)
        grp_["eng_minus_other"] = dict(diff=grp_["eng"]["mean"] - grp_["other"]["mean"],
                                       lo=float(dg.quantile(0.025)), hi=float(dg.quantile(0.975)))
        FE["pf_h3avg"] = dict(beta_f=pt_, lo=A_.quantile(0.025), hi=A_.quantile(0.975), se=A_.std(ddof=1),
                              se_iqr=se_iqr_, p=p_, n_far=n_far, bh=pd.Series(bh(p_.to_numpy()), index=p_.index),
                              p_n=p_n, bh_n=pd.Series(bh(p_n.to_numpy()), index=p_n.index), n_fin=nfin, groups=grp_,
                              n_by_field=FE[HZ_FE_PF[0] + "_pf"]["n_by_field"])
        # percentile CI with a Bonferroni correction for the 3 fields named in advance in scripts/55 (98.33%)
        FE["pf_h3avg"]["lo3"] = A_.quantile(0.025 / 3)
        FE["pf_h3avg"]["hi3"] = A_.quantile(1 - 0.025 / 3)
        # --- where the pooled remainder sits (post hoc groups), under three weightings of the fields: equal,
        # precision (1 / variance of the field's pooled draws) and program count. Per draw, a weighted mean over the
        # fields defined in that draw. Groups: CS/engineering, business (BUSINESS), the rest, and complements.
        nbf = FE[HZ_FE_PF[0] + "_pf"]["n_by_field"].reindex(A_.columns).astype(float)
        wts = {"equal": pd.Series(1.0, index=A_.columns), "precision": 1.0 / A_.var(ddof=1), "programs": nbf}
        bus = [f for f in BUSINESS if f in A_.columns]
        rest = [f for f in oth if f not in bus]
        gsets = {"all": list(A_.columns), "eng": eng, "other": oth, "bus": bus, "rest": rest,
                 "nonbus": [f for f in A_.columns if f not in bus]}

        def _wm(fl_, w_):
            M_ = A_[fl_].to_numpy()
            W_ = np.where(np.isfinite(M_), w_[fl_].to_numpy(float)[None, :], 0.0)
            return (np.nan_to_num(M_) * W_).sum(1) / W_.sum(1), float((pt_[fl_] * w_[fl_]).sum() / w_[fl_].sum())

        grpw = {"sets": gsets, "wshare_eng": float(wts["precision"][eng].sum() / wts["precision"].sum())}
        for wn, w_ in wts.items():
            dr = {}
            for gn, fl_ in gsets.items():
                bm, m_ = _wm(fl_, w_)
                dr[gn] = bm
                grpw[(gn, wn)] = dict(k=len(fl_), mean=m_, lo=float(np.percentile(bm, 2.5)),
                                      hi=float(np.percentile(bm, 97.5)), p0=float(np.mean(bm <= 0)))
            for ga, gb in (("eng", "other"), ("bus", "rest"), ("bus", "nonbus"), ("eng", "rest")):
                dd = dr[ga] - dr[gb]
                grpw[(f"{ga}-{gb}", wn)] = dict(diff=grpw[(ga, wn)]["mean"] - grpw[(gb, wn)]["mean"],
                                               lo=float(np.percentile(dd, 2.5)), hi=float(np.percentile(dd, 97.5)))
        # random sets of field labels of the same size: how often does 'set minus the other fields' reach the
        # observed contrast? (N_PERM sets; equal and precision weights; says nothing about choosing the grouping post hoc)
        rng_l = np.random.default_rng([SEED, 25])
        ptv, wpv, kk = pt_.to_numpy(float), wts["precision"].to_numpy(float), len(pt_)
        lab_ = {}
        for gn, gc in (("eng", "other"), ("bus", "nonbus")):
            m_ = len(gsets[gn])
            ne, npr = np.empty(N_PERM), np.empty(N_PERM)
            for i_ in range(N_PERM):
                s_ = np.zeros(kk, bool)
                s_[rng_l.choice(kk, m_, replace=False)] = True
                ne[i_] = ptv[s_].mean() - ptv[~s_].mean()
                npr[i_] = (ptv[s_] * wpv[s_]).sum() / wpv[s_].sum() - (ptv[~s_] * wpv[~s_]).sum() / wpv[~s_].sum()
            oe, op = grpw[(f"{gn}-{gc}", "equal")]["diff"], grpw[(f"{gn}-{gc}", "precision")]["diff"]
            lab_[gn] = dict(m=m_, obs_eq=oe, p_eq=float((1 + np.sum(ne >= oe - 1e-12)) / (1 + N_PERM)),
                            obs_pr=op, p_pr=float((1 + np.sum(npr >= op - 1e-12)) / (1 + N_PERM)),
                            p95_eq=float(np.percentile(ne, 95)), p95_pr=float(np.percentile(npr, 95)))
        grpw["labels"] = lab_
        FE["pf_h3avg"]["groups_w"] = grpw
        FE["pf_h3avg"]["w_precision"] = wts["precision"] / wts["precision"].sum()
        # --- cross-horizon agreement vs shared sampling error (review): per field, the correlation of its bootstrap
        # draws between two horizons (the same institution draws, the same programs); and the cross-field
        # correlation of CENTRED draws (draw minus point estimate = sampling error only), whose distribution is what
        # the observed cross-field correlation of beta_f would look like from shared sampling error alone
        xd = {}
        nb0 = FE[HZ_FE_PF[0] + "_pf"]["n_by_field"].reindex(f0)
        for a_, b_ in combinations(HZ_FE_PF, 2):
            Da, Db = HZB[a_].to_numpy(), HZB[b_].to_numpy()
            ok_ = np.isfinite(Da) & np.isfinite(Db)
            rf = np.array([np.corrcoef(Da[ok_[:, j], j], Db[ok_[:, j], j])[0, 1] for j in range(len(f0))])
            pa = FE[a_ + "_pf"]["beta_f"].reindex(f0)
            pb = FE[b_ + "_pf"]["beta_f"].reindex(f0)
            Ca, Cb = Da - pa.to_numpy()[None, :], Db - pb.to_numpy()[None, :]
            res = dict(draw_r=pd.Series(rf, index=f0), draw_r_med=float(np.median(rf)),
                       draw_r_q25=float(np.percentile(rf, 25)), draw_r_q75=float(np.percentile(rf, 75)))
            for tag, fl_ in (("all", f0), ("mid", [f for f in f0 if nb0[f] >= PF_NMIN_MID])):
                jj = np.array([f0.index(f) for f in fl_])
                obs_ = float(np.corrcoef(pa[fl_], pb[fl_])[0, 1])
                nl_ = []
                for i_ in range(Ca.shape[0]):
                    ok2 = np.isfinite(Ca[i_, jj]) & np.isfinite(Cb[i_, jj])
                    nl_.append(np.corrcoef(Ca[i_, jj][ok2], Cb[i_, jj][ok2])[0, 1])
                nl_ = np.array(nl_)
                res[tag] = dict(k=len(fl_), obs=obs_, null_med=float(np.median(nl_)),
                                null95=float(np.percentile(nl_, 95)), share=float(np.mean(nl_ >= obs_)))
            xd[(a_, b_)] = res
        FE["pf_h3avg"]["xh_draws"] = xd
        for c_, v_ in (("beta", pt_), ("ci_lo", FE["pf_h3avg"]["lo"]), ("ci_hi", FE["pf_h3avg"]["hi"]),
                       ("se_iqr", se_iqr_), ("p_pct", p_), ("bh_pct", FE["pf_h3avg"]["bh"]), ("p_norm", p_n),
                       ("bh_norm", FE["pf_h3avg"]["bh_n"]), ("w_precision", FE["pf_h3avg"]["w_precision"])):
            T[f"fe_FE2_h3avg_{c_}"] = T.field.map(v_)
        del HZB, A_
        # pooled main spec (unweighted and cohort-size weighted) without the CS/engineering fields: does the
        # weighted pooled remainder survive without them? (the specs refitted on the remaining fields; own draws)
        c_ = cells[~cells.field.isin(eng)]
        Ms_ = []
        for fam in ("FE2", "FE2w"):
            for h in HZ3:
                n_ = f"{fam}_h3{h}"
                sl_, nd_, yv_ = FE_SPECS[n_]
                Ms_.append(fe_build(c_, sl_, nd_, yv_, per_field=False, wcol=FE_WEIGHT.get(n_)))
        bb_ = fe_group_boot(Ms_, [SEED, 3, 20], b_fe)
        pts_ = [fe_fit(M_)["beta"] for M_ in Ms_]
        loo = {"dropped": list(eng)}
        for i_, fam in enumerate(("FE2", "FE2w")):
            a3 = bb_[:, 3 * i_:3 * i_ + 3].mean(axis=1)
            fin_ = a3[np.isfinite(a3)]
            loo[fam] = dict(beta=float(np.mean(pts_[3 * i_:3 * i_ + 3])), per_h=pts_[3 * i_:3 * i_ + 3],
                            ci=(float(np.nanpercentile(a3, 2.5)), float(np.nanpercentile(a3, 97.5))),
                            p0=float(np.mean(fin_ <= 0)), n_cells=len(Ms_[3 * i_]["y"]),
                            n_inst=int(Ms_[3 * i_]["cnt"].size), n_fields=len(Ms_[3 * i_]["fields"]))
        FE["loo_eng"] = loo
        del Ms_, bb_, c_
    # --- wild cluster bootstrap tests that keep the dependence through shared institutions and shared programs ---
    if b_fe > 0:
        FE["wild"] = wild_tests(cells, FEM, FE, B_WILD)
    FE["oster"] = oster(FEM)
    pooled55 = s55.fe_design(cells, [])["pooled"]
    repro["fe0_common_vs_s55_pooled"] = abs(FE["FE0"]["beta"] - pooled55)
    assert repro["fe0_common_vs_s55_pooled"] < 1e-10
    if s55t is not None:
        for mine, theirs in (("fe_FE0_beta", "fe_a_beta"), ("fe_FE2_beta", "fe_c_beta")):
            x = T.set_index("field")[mine].reindex(s55t.index); y = s55t[theirs]
            ok = x.notna() & y.notna()
            repro[f"{mine}_vs_{theirs}"] = (float((x[ok] - y[ok]).abs().max()), int(ok.sum()),
                                            int((x.notna() != y.notna()).sum()))
            assert repro[f"{mine}_vs_{theirs}"][0] < CSV_TOL, f"{mine} does not reproduce scripts/55 {theirs}"
    FE["balance"] = composition_balance(cells)
    assert FEM["FE2sc"]["d"].inst_key.equals(FEM["FE2"]["d"].inst_key) and FEM["FE2sc"]["d"].field.equals(FEM["FE2"]["d"].field)

    # --- split-half of the within-institution per-field beta_f ---
    # the same random halves of the institution universe for every spec; the model is re-estimated on each half
    # (all fields), and the cross-field ordering of beta_f is compared between halves for well-estimated fields
    if b_fe > 0:
        rng_s = np.random.default_rng([SEED, 13])
        splits = np.zeros((SPLIT_K, N), bool)
        for s in range(SPLIT_K):
            splits[s, rng_s.permutation(N)[: N // 2]] = True
        PFS = {"splits_k": SPLIT_K}
        halves = {}
        for name in ["FE2", "FE2sc", "FE2sg", "FE0_s"]:
            slopes, need, yvar = FE_SPECS[name]
            Mf = fe_build(cells, slopes, need, yvar, per_field=True)
            keys_ = pd.factorize(Mf["d"].inst_key)[1]
            inst_univ = np.array([inst_index[k_] for k_ in keys_])
            assert np.array_equal(pd.factorize(Mf["d"].inst_key)[0], Mf["groups"])
            halves[name] = (Mf["fields"], fe_pf_halves(Mf, inst_univ, splits, FE_PF_MIN_DISTINCT),
                            Mf["d"].groupby("field").size())
            if name == "FE2":
                na_ = splits[:, inst_univ].sum(1)          # main-spec institutions in half A, per split
                PFS["inst_half"] = (int(na_.min()), int(np.median(na_)), int(na_.max()), int(Mf["cnt"].size))
            del Mf
        f2, _, nby = halves["FE2"]
        for name in ["FE2sc", "FE2sg", "FE0_s"]:
            assert list(halves[name][0]) == list(f2)
        eng_all = [f for f in f2 if "engineering" in f or f == "computer_science"]
        big = [f for f in f2 if nby[f] >= PF_SPLIT_NMIN]
        big2 = [f for f in f2 if nby[f] >= PF_SPLIT_NMIN2]
        variants = [("main", "FE2", big), ("main_sc", "FE2sc", big), ("main_n40", "FE2", big2),
                    ("main_nocs", "FE2", [f for f in big if f != "computer_science"]),
                    ("main_noeng", "FE2", [f for f in big if f not in eng_all]),
                    ("literal", "FE2sg", big), ("a_s", "FE0_s", big)]
        rng_p = np.random.default_rng([SEED, 14])
        perm_store = {}
        for tag, name, fl_ in variants:
            sel = np.array([list(f2).index(f) for f in fl_])
            perms = [rng_p.permutation(len(sel)) for _ in range(N_PERM)]
            perm_store[tag] = perms
            PFS[tag] = pf_split_stats(halves[name][1], sel, perms)
            PFS[tag].update(spec=name, fields=fl_)
        PFS["eng_in_big"] = [f for f in big if f in eng_all]
        FE["pf_split"] = PFS
        # --- robustness of the cross-field heterogeneity to nonlinear (cubic) field-specific pricing ---
        NL = nonlinear_het(cells, FE, FEM, inst_index, splits, halves["FE2"][1], perm_store, b_fe, B_WILD)
        for tag in ("main", "main_nocs", "main_noeng"):          # the main spec reproduces the split-half above
            assert abs(NL["FE2"]["split"][tag]["rel"] - PFS[tag]["rel"]) < 1e-12
            assert abs(NL["FE2"]["split"][tag]["p"] - PFS[tag]["p"]) < 1e-12
        if "wild" in FE:                                          # ... and the wild-cluster Q of wild_tests
            for set_ in ("all", "mid", "big", "mid_noeng"):
                for kind in WILD_KINDS:
                    assert abs(NL["FE2"]["het"][(set_, "wild_iqr")][kind]["p"]
                               - FE["wild"]["Q4"][(set_, "iqr")][kind]["p"]) < 1e-12
        for name in ("FE2cg", "FE2ca"):
            r_ = NL[name]
            for c_, v_ in (("beta", r_["beta_f"]), ("ci_lo", r_["lo"]), ("ci_hi", r_["hi"]), ("se_iqr", r_["se_iqr"])):
                T[f"fe_{name}_{c_}"] = T.field.map(v_)
        FE["nl"] = NL
        del halves, splits, perm_store

    # --- split-half ---
    SH = split_half(FD, fields, N, np.random.default_rng([SEED, 7]))

    # --- coverage of the composition sample ---
    u = cells.assign(has_pp=cells.PPELL.notna())
    has_med = cells.earn_pell.notna() & cells.earn_nopell.notna()
    cov = dict(cells=len(cells), cells_ppell=int(u.has_pp.sum()), cells_pmale=int(cells.PMALE.notna().sum()),
               cells_med=int(has_med.sum()), cells_ppell_nomed=int((u.has_pp & ~has_med).sum()),
               cells_med_noppell=int((~u.has_pp & has_med).sum()),
               cells_ppell_d=int(cells.PPELL_D.notna().sum()),
               ppell_rowcov_mean=float(cells.loc[cells.PPELL.notna(), "PPELL_rowcov"].mean()),
               multi_row_cells=int((cells.nrows > 1).sum()))
    gq = pd.qcut(u.g_rank.rank(method="first"), 4, labels=["top", "2nd", "3rd", "bottom"])
    cov["by_gq"] = u.groupby(gq, observed=True).has_pp.mean().to_dict()
    cov["by_size"] = u.groupby(pd.qcut(u.cohort_n.rank(method="first"), 4, labels=["smallest", "2nd", "3rd", "largest"]),
                               observed=True).has_pp.mean().to_dict()
    ctx = dict(cells=cells, FD=FD, fields=fields, point=point, boot=boot, se=se, keys=keys, T=T, FE=FE,
               SH=SH, rel=rel, repro=repro, cov=cov, N=N, ref=ref)
    ctx["avail"] = availability() if do_avail else None
    return ctx


# ---------------------------------------------------------------------------------------------
# cross-field summaries
# ---------------------------------------------------------------------------------------------
def summarize(ctx, col, fl=None, resid=True):
    """Cross-field summary of a per-field statistic: unweighted mean (joint-bootstrap SE), REML
    random-effects mean (bootstrap SE with fixed weights), tau, Q, per-field significance counts
    (bootstrap SE, normal), BH-FDR, empirical-Bayes counts, pooled residual slope, TOST, MDE."""
    T, boot, idx = ctx["T"].set_index("field"), ctx["boot"], ctx["idx"]
    s = T[np.isfinite(T[col])]
    if fl is not None:
        s = s[s.index.isin(fl)]
    k = len(s)
    out = dict(col=col, k=k)
    if k < 3:
        return out
    y, se = s[col].to_numpy(float), s[f"se_{col}"].to_numpy(float)
    Bm = boot[col][:, [idx[f] for f in s.index]]
    bm = np.nanmean(Bm, axis=1)
    mean, se_mean = float(y.mean()), float(np.nanstd(bm, ddof=1))
    re = reml(y, se ** 2)
    w = 1 / (se ** 2 + re["tau"] ** 2)
    fin = np.isfinite(Bm)
    bre = np.nansum(np.where(fin, Bm, 0) * w, axis=1) / np.sum(fin * w, axis=1)
    se_re = float(np.nanstd(bre, ddof=1))
    z = y / se
    p = 2 * norm.sf(np.abs(z))
    sel = bh(p)
    t2 = re["tau"] ** 2
    if t2 > 0:
        Bf = t2 / (t2 + se ** 2)
        post = re["mu"] + Bf * (y - re["mu"])
        pv = t2 * se ** 2 / (t2 + se ** 2) + (se ** 2 / (t2 + se ** 2)) ** 2 * se_re ** 2
    else:
        post, pv = np.full(k, re["mu"]), np.full(k, se_re ** 2)
    out.update(mean=mean, se_mean=se_mean, lo=mean - 1.96 * se_mean, hi=mean + 1.96 * se_mean,
               lo90=mean - 1.645 * se_mean, hi90=mean + 1.645 * se_mean,
               pct_lo=float(np.nanpercentile(bm, 2.5)), pct_hi=float(np.nanpercentile(bm, 97.5)),
               boot_bias=float(np.nanmean(bm) - mean), sd=float(y.std(ddof=1)),
               rms_se=float(np.sqrt(np.mean(se ** 2))),
               mu_re=re["mu"], se_re=se_re, lo_re=re["mu"] - 1.96 * se_re, hi_re=re["mu"] + 1.96 * se_re,
               se_re_indep=re["se"], tau=re["tau"], tau_hi=re["tau_hi"], Q=re["Q"], Qp=re["p"], I2=re["I2"],
               n_pos=int(np.sum(z > 1.96)), n_neg=int(np.sum(z < -1.96)),
               n_bh_pos=int(np.sum(sel & (z > 0))), n_bh_neg=int(np.sum(sel & (z < 0))),
               n_eb_pos=int(np.sum(post - 1.96 * np.sqrt(pv) > 0)),
               n_eb_neg=int(np.sum(post + 1.96 * np.sqrt(pv) < 0)),
               n_pos_point=int(np.sum(y > 0)), exp_fp=0.025 * k,
               mde80=2.8 * se_mean, tost=bool((mean - 1.645 * se_mean > -EQ_BOUND) and (mean + 1.645 * se_mean < EQ_BOUND)),
               eq_bound=float(max(abs(mean - 1.645 * se_mean), abs(mean + 1.645 * se_mean))),
               fields=list(s.index), post=pd.Series(post, index=s.index), post_sd=pd.Series(np.sqrt(pv), index=s.index))
    rl = s[s.reliable]
    out["k_rel"] = len(rl)
    out["mean_rel"] = float(rl[col].mean()) if len(rl) else np.nan
    if resid and SPECS.get(col) is not None:
        res = [ctx["point"][f].get(f"_res_{col}") for f in s.index]
        res = [r for r in res if r is not None]
        if len(res) >= 3:
            ps = pooled_resid_slope(res)
            out.update(pool_b=ps["b"], pool_se=ps["se"], pool_lo=ps["b"] - 1.96 * ps["se"],
                       pool_hi=ps["b"] + 1.96 * ps["se"], pool_cells=ps["n_cells"], pool_clusters=ps["n_clusters"])
    return out


def diff_summary(ctx, a, b, fl=None):
    """Mean over fields (finite in both) of a - b, with joint-bootstrap SE."""
    T, boot, idx = ctx["T"].set_index("field"), ctx["boot"], ctx["idx"]
    s = T[np.isfinite(T[a]) & np.isfinite(T[b])]
    if fl is not None:
        s = s[s.index.isin(fl)]
    jj = [idx[f] for f in s.index]
    d = (s[a] - s[b]).to_numpy()
    bd = np.nanmean(boot[a][:, jj] - boot[b][:, jj], axis=1)
    m, se_ = float(d.mean()), float(np.nanstd(bd, ddof=1))
    return dict(k=len(s), mean_a=float(s[a].mean()), mean_b=float(s[b].mean()), diff=m, se=se_,
                lo=m - 1.96 * se_, hi=m + 1.96 * se_)


def rel_summary(ctx, col):
    """Disattenuated mean for a residual field-prestige spec under the LB and EXT reliabilities."""
    T, boot, idx = ctx["T"].set_index("field"), ctx["boot"], ctx["idx"]
    out = {}
    base = T[np.isfinite(T[col])]
    out["k"] = len(base)
    r2c = f"r2x_{col}"
    out["r2x_mean"] = float(base[r2c].mean()) if r2c in base and SPECS[col][3] else 0.0
    for tag in ("LB", "EXT"):
        rp = base[f"relperp_{tag}_{col}"]
        ok = rp.notna()
        s = base[ok]
        fac = 1 / np.sqrt(s[f"relperp_{tag}_{col}"].to_numpy())
        jj = [idx[f] for f in s.index]
        bm = np.nanmean(boot[col][:, jj] * fac[None, :], axis=1)
        m = float(np.mean(s[col].to_numpy() * fac))
        se_ = float(np.nanstd(bm, ddof=1))
        out[tag] = dict(k_id=int(ok.sum()), k_not=int((~ok).sum()), relperp_med=float(rp[ok].median()) if ok.any() else np.nan,
                        relF_mean=float(base[f"relF_{tag}"].mean()),
                        mean_obs=float(s[col].mean()) if len(s) else np.nan,
                        mean=m, se=se_, lo=m - 1.96 * se_, hi=m + 1.96 * se_)
    return out


def hr_summary(ctx):
    T, boot, idx = ctx["T"].set_index("field"), ctx["boot"], ctx["idx"]
    s = T[np.isfinite(T["hr_sh_F"])]
    jj = [idx[f] for f in s.index]
    out = dict(k=len(s), k_rel=int(s.reliable.sum()), r2=float(s.hr_r2.mean()), r2_3=float(s.hr_r2_3.mean()))

    def mb(col, sub=None):
        q = s if sub is None else s[sub]
        j2 = [idx[f] for f in q.index]
        bm = np.nanmean(boot[col][:, j2], axis=1)
        m, se_ = float(q[col].mean()), float(np.nanstd(bm, ddof=1))
        return dict(mean=m, se=se_, lo=m - 1.96 * se_, hi=m + 1.96 * se_)

    for b in ["F", "S", "G", "IE"]:
        out[f"sh_{b}"] = mb(f"hr_sh_{b}")
        out[f"sh_{b}_rel"] = mb(f"hr_sh_{b}", s.reliable)
        out[f"lead_{b}"] = int((s.hr_leader == b).sum())
    for b in ["F", "S", "G"]:
        out[f"sh3_{b}"] = mb(f"hr3_sh_{b}")
        out[f"lead3_{b}"] = int((s.hr3_leader == b).sum())
    bd = np.nanmean(boot["hr_sh_F"][:, jj] - boot["hr_sh_G"][:, jj], axis=1)
    dm = float((s.hr_sh_F - s.hr_sh_G).mean())
    out["shFG"] = dict(mean=dm, se=float(np.nanstd(bd, ddof=1)))
    out["shFG"]["lo"], out["shFG"]["hi"] = dm - 1.96 * out["shFG"]["se"], dm + 1.96 * out["shFG"]["se"]
    out["n_shF_gt_shG"] = int((s.hr_sh_F > s.hr_sh_G).sum())
    out["uniq_F"] = mb("hr_uniq_F"); out["uniq_G"] = mb("hr_uniq_G"); out["uniq3_F"] = mb("hr3_uniq_F")
    for a, b in (("F", "G"), ("F", "S"), ("F", "IE"), ("G", "S")):
        c = s[f"hr_cd_{a}_{b}"]
        out[f"cd_{a}_{b}"] = (int((c == 1).sum()), int((c == -1).sum()), int((c == 0).sum()))
    out["F_pos_sig"] = int(((s.hr_p_F < 0.05) & (s.hr_b_F > 0)).sum())
    out["F_neg_sig"] = int(((s.hr_p_F < 0.05) & (s.hr_b_F < 0)).sum())
    out["G_pos_sig"] = int(((s.hr_p_G < 0.05) & (s.hr_b_G > 0)).sum())
    out["G_neg_sig"] = int(((s.hr_p_G < 0.05) & (s.hr_b_G < 0)).sum())
    q5 = s[np.isfinite(s.hr5_sh_F)]
    out["k5"] = len(q5)
    for b in ["F", "S", "G", "IE", "C"]:
        out[f"sh5_{b}"] = float(q5[f"hr5_sh_{b}"].mean()) if len(q5) else np.nan
    out["r2_5"] = float(q5.hr5_r2.mean()) if len(q5) else np.nan
    return out


def fe_pf_summary(FEr, se_key="se_f"):
    b, se = FEr["beta_f"].to_numpy(), FEr[se_key].reindex(FEr["beta_f"].index).to_numpy()
    re = reml(b, se ** 2)
    z = b / se
    sel = bh(2 * norm.sf(np.abs(z)))
    t2 = re["tau"] ** 2
    if t2 > 0:
        post = re["mu"] + t2 / (t2 + se ** 2) * (b - re["mu"])
        pv = t2 * se ** 2 / (t2 + se ** 2) + (se ** 2 / (t2 + se ** 2)) ** 2 * re["se"] ** 2
    else:
        post, pv = np.full(len(b), re["mu"]), np.full(len(b), re["se"] ** 2)
    return dict(k=len(b), mean=float(b.mean()), sd=float(b.std(ddof=1)), rms_se=float(np.sqrt(np.mean(se ** 2))),
                mu_re=re["mu"], se_re=re["se"], tau=re["tau"], tau_hi=re["tau_hi"], Qp=re["p"], wald_p=FEr["wald_p"],
                wald=FEr["wald"], wald_df=FEr["wald_df"],
                n_pos=int(np.sum(z > 1.96)), n_neg=int(np.sum(z < -1.96)),
                n_bh_pos=int(np.sum(sel & (z > 0))), n_bh_neg=int(np.sum(sel & (z < 0))),
                n_eb_pos=int(np.sum(post - 1.96 * np.sqrt(pv) > 0)), n_eb_neg=int(np.sum(post + 1.96 * np.sqrt(pv) < 0)))


SUMM_SPECS = ["raw", "F_G", "G_raw", "G_F", "raw_sat", "sat_earn", "sel", "F_selG", "broad", "F_broadG", "F_broadGie", "G_broad",
              "G_broadF", "raw_comp", "broad_comp", "F_broadP", "F_broadG_comp", "F_all", "G_all", "raw_nopell",
              "raw_pell", "F_broadP_nopell", "F_broadP_pell", "F_all_nopell", "F_all_pell", "F_allM_base",
              "F_allM", "broad_compd", "F_broadPd", "F_allD", "raw_adm", "adm", "F_admG", "F_admGP"] + HZ_SPECS + \
             [f"{_fam}_h3avg" for _fam in HZ_AVG_P]
DIFFS = [("F_selG", "sel"), ("F_broadG", "broad"), ("F_broadP", "broad_comp"), ("F_all", "F_broadG_comp"), ("F_all", "broad_comp"),
         ("F_all_nopell", "F_all"), ("F_all_pell", "F_all"), ("F_broadP_nopell", "F_broadP"), ("F_allM", "F_allM_base"),
         ("F_broadPd", "broad_compd"), ("F_broadGie", "F_broadG"), ("F_admG", "adm"), ("F_broadG", "G_broadF"),
         ("F_all", "G_all"), ("F_G", "G_F"),
         # earnings horizons (same programs, same bootstrap draws)
         ("F_broadG_h1y1", "F_broadG_h1y4"), ("F_broadG_h5y5", "F_broadG_h5y4"),
         ("raw_h3y1", "raw_h3y4"), ("raw_h3y5", "raw_h3y4"), ("broad_h3y1", "broad_h3y4"), ("broad_h3y5", "broad_h3y4"),
         ("F_broadG_h3y1", "F_broadG_h3y4"), ("F_broadG_h3y5", "F_broadG_h3y4"),
         ("G_broadF_h3y1", "G_broadF_h3y4"), ("G_broadF_h3y5", "G_broadF_h3y4"),
         ("F_broadG_h3y4", "broad_h3y4"), ("F_broadG_h3y1", "broad_h3y1"), ("F_broadG_h3y5", "broad_h3y5")]
REL_SPECS = ["F_G", "broad", "F_broadG", "F_all", "F_admG", "F_broadGie", "F_broadG_h1y4", "F_broadG_h1y1",
             "F_broadG_h5y4", "F_broadG_h5y5", "F_broadG_h3y4", "F_broadG_h3y1", "F_broadG_h3y5"]


def summaries(ctx):
    ctx["idx"] = {f: j for j, f in enumerate(ctx["fields"])}
    ctx["S"] = {c: summarize(ctx, c) for c in SUMM_SPECS}
    ctx["S_rel"] = {c: summarize(ctx, c, fl=list(ctx["T"].field[ctx["T"].reliable]), resid=False)
                    for c in ["F_broadG", "F_all", "broad", "F_admG"]}
    ctx["S_nolic"] = {c: summarize(ctx, c, fl=list(ctx["T"].field[~ctx["T"].licensed]), resid=False)
                      for c in ["F_broadG", "F_all"]}
    ctx["D"] = {(a, b): diff_summary(ctx, a, b) for a, b in DIFFS}
    ctx["REL"] = {c: rel_summary(ctx, c) for c in REL_SPECS}
    ctx["HR"] = hr_summary(ctx)
    ctx["FEPF"] = {n: fe_pf_summary(ctx["FE"][n + "_pf"]) for n in ["FE0", "FE2", "FE3", "FE3_np", "FE3_p"]}
    if "se_iqr" in ctx["FE"]["FE2_pf"]:
        ctx["FEPF"]["FE2_bootsd"] = fe_pf_summary(ctx["FE"]["FE2_pf"], "se_boot")
        ctx["FEPF"]["FE2_bootiqr"] = fe_pf_summary(ctx["FE"]["FE2_pf"], "se_iqr")
    for n in HZ_FE_PF:
        r_ = ctx["FE"][n + "_pf"]
        if "se_iqr" in r_:
            q = fe_pf_summary(r_, "se_iqr")
            q["ci_pos"] = [f for f in r_["beta_f"].index if r_["ci_lo"][f] > 0]
            q["ci_neg"] = [f for f in r_["beta_f"].index if r_["ci_hi"][f] < 0]
            ctx["FEPF"][n] = q
    if all(n in ctx["FEPF"] for n in HZ_FE_PF):
        B_ = pd.DataFrame({n: ctx["FE"][n + "_pf"]["beta_f"] for n in HZ_FE_PF})
        ctx["FEPF"]["hz_corr"] = {(a, b): float(B_[a].corr(B_[b])) for a, b in combinations(HZ_FE_PF, 2)}
        ctx["FEPF"]["hz_spear"] = {(a, b): float(B_[a].corr(B_[b], method="spearman")) for a, b in combinations(HZ_FE_PF, 2)}
    # power of the split-half reliability tests (simulated fields with the observed bootstrap SEs)
    Ti, SH, S = ctx["T"].set_index("field"), ctx["SH"], ctx["S"]
    POW = {}
    for j, n in enumerate(["F_broadG", "broad"]):
        se_ = Ti.loc[SH["fields"], f"se_{n}"].to_numpy(float)
        assert np.all(np.isfinite(se_))
        at = {"zero": 0.0, "reml": S[n]["tau"], "reml_hi": S[n]["tau_hi"]}
        if n == "F_broadG":
            at.update(mid=0.08, broad_reml=S["broad"]["tau"])
        taus = np.unique(np.concatenate([POW_GRID_P, np.round(list(at.values()), 6)]))
        tab, null = splithalf_power(se_, taus, [SEED, 15, j])
        POW[n] = power_summary(tab, null, SH[f"rel_{n}"], at)
        POW[n].update(se_mean=float(se_.mean()), se_rms=float(np.sqrt(np.mean(se_ ** 2))), k=len(se_))
        # heterogeneity test on the same fields (Q, bootstrap SEs), for comparison with the split-half p-values
        q_ = reml(Ti.loc[SH["fields"], n].to_numpy(float), se_ ** 2)
        POW[n].update(Q_tau=q_["tau"], Q_p=q_["p"])
    if "pf_split" in ctx["FE"] and "se_boot" in ctx["FE"]["FE2_pf"]:
        PFS, pf = ctx["FE"]["pf_split"], ctx["FE"]["FE2_pf"]
        fl_ = PFS["main"]["fields"]
        se_ = pf["se_boot"].reindex(fl_).to_numpy(float)
        assert np.all(np.isfinite(se_))
        at = {"zero": 0.0, "reml_sd": ctx["FEPF"]["FE2_bootsd"]["tau"], "reml_iqr": ctx["FEPF"]["FE2_bootiqr"]["tau"]}
        taus = np.unique(np.concatenate([POW_GRID_B, np.round(list(at.values()), 6)]))
        tab, null = splithalf_power(se_, taus, [SEED, 16])
        POW["beta_f"] = power_summary(tab, null, PFS["main"]["rel"], at)
        POW["beta_f"].update(se_mean=float(se_.mean()), se_rms=float(np.sqrt(np.mean(se_ ** 2))), k=len(se_))
        b_ = pf["beta_f"].reindex(fl_).to_numpy(float)
        q_sd = reml(b_, se_ ** 2)
        q_iqr = reml(b_, pf["se_iqr"].reindex(fl_).to_numpy(float) ** 2)
        POW["beta_f"].update(Q_tau=q_sd["tau"], Q_p=q_sd["p"], Q_tau_iqr=q_iqr["tau"], Q_p_iqr=q_iqr["p"])
        PFS["main"]["p_sim"] = POW["beta_f"]["p_sim"]
        # the other variants that use the main-spec model (whose per-field bootstrap SEs exist): same simulated null
        for i, tag in enumerate(["main_n40", "main_nocs", "main_noeng"]):
            se_v = pf["se_boot"].reindex(PFS[tag]["fields"]).to_numpy(float)
            assert np.all(np.isfinite(se_v))
            _, null_v = splithalf_power(se_v, np.array([0.0]), [SEED, 17, i])
            PFS[tag]["p_sim"] = float((1 + np.sum(null_v >= PFS[tag]["rel"])) / (1 + len(null_v)))
    if "wild" in ctx["FE"]:
        WD = ctx["FE"]["wild"]
        if "pf_split" in ctx["FE"]:
            assert WD["Q4_big"] == list(ctx["FE"]["pf_split"]["main"]["fields"])
        if "hz_corr" in ctx["FEPF"]:
            for key_, v_ in WD["hz_corr"].items():
                assert abs(v_["r"] - ctx["FEPF"]["hz_corr"][key_]) < 1e-10
                assert abs(v_["rs"] - ctx["FEPF"]["hz_spear"][key_]) < 1e-10
        # the chi2 Q of the full-sample main spec reproduces fe_pf_summary's Q (same statistic)
        assert abs(WD["Q4"][("all", "iqr")]["chi2_p"] - ctx["FEPF"]["FE2_bootiqr"]["Qp"]) < 1e-9
        assert abs(WD["Q4"][("all", "sd")]["chi2_p"] - ctx["FEPF"]["FE2_bootsd"]["Qp"]) < 1e-9
    ctx["POW"] = POW
    # EB posterior for the headline residual specs into the table
    T = ctx["T"]
    for c in ["F_broadG", "F_all"]:
        T[f"eb_{c}"] = T.field.map(ctx["S"][c]["post"])
        T[f"eb_sd_{c}"] = T.field.map(ctx["S"][c]["post_sd"])
    return ctx


# ---------------------------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------------------------
PAL = {"blue": "#0F4D92", "blue2": "#3775BA", "red": "#B64342", "red2": "#E9A6A1", "teal": "#42949E",
       "violet": "#9A4D8E", "neutral": "#9a9a9a", "green": "#5aa45a"}


def make_figure(ctx):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 1.0, "legend.frameon": False, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.25], hspace=0.5, wspace=0.42)
    T = ctx["T"].set_index("field")
    S = ctx["S"]

    # (a) forest: field-specific component per field
    ax = fig.add_subplot(gs[:, 0])
    c = "F_broadG"
    s = T[np.isfinite(T[c])].sort_values(c)
    ypos = np.arange(len(s))
    col = [PAL["red"] if lic else PAL["blue"] for lic in s.licensed]
    ax.errorbar(s[c], ypos, xerr=1.96 * s[f"se_{c}"], fmt="none", ecolor="#b8c4d6", elinewidth=1.1, zorder=1)
    ax.scatter(s[c], ypos, c=col, s=18, zorder=3, label="estimate ± 1.96 bootstrap SE")
    ax.scatter(s[f"eb_{c}"], ypos, facecolors="none", edgecolors="k", s=16, lw=0.7, zorder=4,
               label="empirical-Bayes (shrunk) estimate")
    su = S[c]
    ax.axvspan(su["lo"], su["hi"], color=PAL["teal"], alpha=0.18, lw=0,
               label=f"mean {su['mean']:+.3f} [{su['lo']:+.3f}, {su['hi']:+.3f}] (joint bootstrap)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([LAB.get(f, f) + (" *" if r else "") for f, r in zip(s.index, s.reliable)], fontsize=7)
    ax.set_xlabel("partial ρ(field prestige, 4-yr earnings | brand G, SAT, admit rate, inst. Pell, control, state level)")
    ax.set_title(f"(a) Field-specific component of prestige, per field (SAT sample, k={len(s)})\n"
                 f"{su['n_pos']} fields > 0 and {su['n_neg']} < 0 at 1.96 SE (≈{su['exp_fp']:.1f} per tail expected by chance); "
                 f"BH-FDR: {su['n_bh_pos']} / {su['n_bh_neg']}\n"
                 f"between-field SD τ = {su['tau']:.3f} (Q p = {su['Qp']:.2f}).  red = licensed field; * = reliable (gap map)",
                 loc="left", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.045), fontsize=7.5, ncol=1)
    ax.set_ylim(-1, len(s))

    # (b) ladder of mean partials
    ax = fig.add_subplot(gs[0, 1])
    rows = [("raw", "F: raw ρ (full sample)", PAL["blue"]),
            ("F_G", "F | brand G (full sample)", PAL["blue"]),
            ("broad", "F | selectivity set (SAT sample) [scripts/55 broad]", PAL["blue"]),
            ("F_broadG", "F | selectivity set + G", PAL["blue"]),
            ("F_broadGie", "F | selectivity set + G + inst. earnings", PAL["blue"]),
            ("F_admG", "F | ADM set + G (ADM sample, keeps test-blind)", PAL["blue2"]),
            ("broad_comp", "F | selectivity set (composition sample)", PAL["teal"]),
            ("F_broadP", "F | selectivity set + program Pell share", PAL["teal"]),
            ("F_all", "F | selectivity set + G + program Pell share", PAL["teal"]),
            ("F_all_nopell", "  same, non-Pell median as outcome", PAL["teal"]),
            ("F_allM", "  same + program male share (male-share sample)", PAL["teal"]),
            ("G_broad", "G | selectivity set (SAT sample)", PAL["red"]),
            ("G_broadF", "G | selectivity set + F", PAL["red"])]
    for i, (cc, lab, colr) in enumerate(rows):
        q = S[cc]
        ax.errorbar(q["mean"], -i, xerr=[[q["mean"] - q["lo"]], [q["hi"] - q["mean"]]], fmt="o", color=colr,
                    ms=4.5, capsize=2, lw=1.2)
        ax.text(1.02, -i, f"{q['mean']:+.3f}  (k={q['k']})", transform=ax.get_yaxis_transform(), va="center", fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(-np.arange(len(rows)))
    ax.set_yticklabels([r[1] for r in rows], fontsize=7.5)
    ax.set_xlabel("mean partial ρ across fields, 95% CI (joint institution bootstrap)")
    ax.set_title("(b) How much coupling is left as controls are added\n"
                 "blue/teal = field prestige F, red = brand G; selectivity set = SAT, admit rate, inst. Pell, control, state level",
                 loc="left", fontsize=9)

    # (c) within-institution common beta
    ax = fig.add_subplot(gs[1, 1])
    FE = ctx["FE"]
    fr = [("FE0", PAL["neutral"]), ("FE0_s", PAL["neutral"]), ("FE2sg", PAL["blue2"]), ("FE1", PAL["blue2"]),
          ("FE2", PAL["blue"]), ("FE2ie", PAL["blue"]), ("FE2cg", PAL["blue"]), ("FE2ca", PAL["blue"]),
          ("FE2_c", PAL["teal"]), ("FE3", PAL["teal"]),
          ("FE3_np", PAL["teal"]), ("FE3_p", PAL["teal"]), ("FE3m", PAL["teal"]), ("FE3d", PAL["violet"])]
    for i, (n, colr) in enumerate(fr):
        r = FE[n]
        lo, hi = r.get("boot_ci", (r["beta"] - 1.96 * r["se"], r["beta"] + 1.96 * r["se"]))
        ax.errorbar(r["beta"], -i, xerr=[[r["beta"] - lo], [hi - r["beta"]]], fmt="o", color=colr, ms=4.5, capsize=2)
        ax.text(1.02, -i, f"{r['beta']:+.4f}  ({r['n_cells']} cells, {r['n_fields']} fields)",
                transform=ax.get_yaxis_transform(), va="center", fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(-np.arange(len(fr)))
    ax.set_yticklabels([FE_DESC[n] for n, _ in fr], fontsize=7.5)
    ax.set_xlabel("common β: log 4-yr earnings per within-field SD of department prestige (95% institution cluster-bootstrap CI)")
    ax.set_title("(c) Within-institution cross-department design (institution FE + field FE + field-specific slopes)",
                 loc="left", fontsize=9)

    # (d) horse race
    ax = fig.add_subplot(gs[2, 1])
    H = ctx["HR"]
    blocks = ["F", "S", "G", "IE"]
    x = np.arange(len(blocks))
    wdt = 0.36
    m4 = [H[f"sh_{b}"]["mean"] for b in blocks]
    e4 = [1.96 * H[f"sh_{b}"]["se"] for b in blocks]
    m3 = [H[f"sh3_{b}"]["mean"] if b != "IE" else np.nan for b in blocks]
    e3 = [1.96 * H[f"sh3_{b}"]["se"] if b != "IE" else 0 for b in blocks]
    ax.bar(x - wdt / 2, m4, wdt, yerr=e4, color=PAL["blue"], capsize=3,
           label=f"4 blocks (mean R² {H['r2']:.2f}); leader in {H['lead_F']}/{H['lead_S']}/{H['lead_G']}/{H['lead_IE']} fields")
    ax.bar(x + wdt / 2, m3, wdt, yerr=e3, color=PAL["red2"], capsize=3,
           label=f"3 blocks, no inst. earnings (mean R² {H['r2_3']:.2f}); leader {H['lead3_F']}/{H['lead3_S']}/{H['lead3_G']}")
    ax.set_xticks(x)
    ax.set_xticklabels([HR_LAB[b] for b in blocks])
    ax.set_ylabel("mean Shapley R² across fields (95% CI)")
    ax.set_title(f"(d) Horse race per field on ranks (SAT sample, k={H['k']}): who predicts program earnings?",
                 loc="left", fontsize=9)
    ax.set_ylim(0, 1.3 * max(np.nanmax(np.array(m4) + np.array(e4)), np.nanmax(np.array(m3) + np.array(e3))))
    ax.legend(fontsize=7.5, loc="upper left")

    # (e) earnings-horizon replication on the common programs, with the horizon-pooled (4/1/5-yr average) estimate
    sub = gs[3, 1].subgridspec(1, 2, wspace=0.32, width_ratios=[1.2, 1.2])
    hcol = {"y4": PAL["blue"], "y1": PAL["teal"], "y5": PAL["red"], "avg": "k"}
    off = {"y4": -0.27, "y1": -0.09, "y5": 0.09, "avg": 0.27}
    hlab = dict(HZ_LAB, avg="4/1/5-yr average")
    ax = fig.add_subplot(sub[0, 0])
    grp = [("raw", "F raw"), ("broad", "F |\nsel. set"), ("F_broadG", "F | sel.\nset + G"), ("G_broadF", "G | sel.\nset + F")]
    for h in ("y4", "y1", "y5", "avg"):
        for i, (g, _) in enumerate(grp):
            q = S[f"{g}_h3{h}"]
            ax.errorbar(i + off[h], q["mean"], yerr=[[q["mean"] - q["lo"]], [q["hi"] - q["mean"]]],
                        fmt="D" if h == "avg" else "o", color=hcol[h], ms=4, capsize=2, lw=1.1,
                        label=hlab[h] + ("" if h == "avg" else " earnings") if i == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels([g[1] for g in grp], fontsize=7.5)
    ax.set_ylabel("mean partial ρ across fields (95% CI)", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"(e) Same programs, other earnings horizons (programs with all three released)\n"
                 f"left: partial correlations, k = {S['F_broadG_h3y4']['k']} fields; right: within-institution β; "
                 f"diamonds = 4/1/5-yr average (shared draws)",
                 loc="left", fontsize=9)
    ax = fig.add_subplot(sub[0, 1])
    grp2 = [("FE0", "(a) inst. +\nfield FE"), ("FE2sg", "(a) + field ×\n{SAT, G}"), ("FE2", "main"),
            ("FE2w", "main,\nweighted")]
    for h in ("y4", "y1", "y5", "avg"):
        for i, (g, _) in enumerate(grp2):
            r = FE[f"{g}_h3{h}"]
            lo, hi = r.get("boot_ci", (r["beta"] - 1.96 * r.get("se", np.nan), r["beta"] + 1.96 * r.get("se", np.nan)))
            ax.errorbar(i + off[h], r["beta"], yerr=[[r["beta"] - lo], [hi - r["beta"]]], fmt="D" if h == "avg" else "o",
                        color=hcol[h], ms=4, capsize=2, lw=1.1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(grp2)))
    ax.set_xticklabels([g[1] for g in grp2], fontsize=7)
    ax.set_ylabel(f"β, log points per SD ({FE['FE2_h3y4']['n_cells']:,} programs)", fontsize=8)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------------------------
FULL_SPECS = ["F_selG", "F_broadG", "F_broadGie", "F_admG", "F_admGP", "F_all", "F_all_nopell", "F_all_pell",
              "F_allM", "F_allD"]
NAMED = ["computer_science", "economics", "statistics", "computer_engineering", "accounting", "nursing"]


def write_csv(ctx):
    T = ctx["T"].copy()
    drop = [c for c in T.columns if c.startswith("_")]
    T = T.drop(columns=drop).sort_values("field").reset_index(drop=True)
    T.to_csv(OUT_CSV, index=False, float_format="%.6g")


def _f(x, nd=3, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _ci(lo, hi, nd=3):
    return f"[{_f(lo, nd)}, {_f(hi, nd)}]"


def _p(p):
    if p is None or not np.isfinite(p):
        return "—"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.2g}" if p < 0.01 else f"{p:.2f}"


def _est(q, nd=3):
    return f"{_f(q['mean'] if 'mean' in q else q['diff'], nd)} {_ci(q['lo'], q['hi'], nd)}"


def _fe_ci(r, nd=4):
    lo, hi = r["boot_ci"]
    return f"{_f(r['beta'], nd)} {_ci(lo, hi, nd)}"


HZ_DIFF_LIST = [  # (label, kind, key) : kind P = partial (DIFFS), F = within-institution (FE_PAIRS)
    ("partial, 5-yr sample, 5 − 4 yr", "P", ("F_broadG_h5y5", "F_broadG_h5y4")),
    ("partial, 1-yr sample, 1 − 4 yr", "P", ("F_broadG_h1y1", "F_broadG_h1y4")),
    ("partial, common programs, 5 − 4 yr", "P", ("F_broadG_h3y5", "F_broadG_h3y4")),
    ("partial, common programs, 1 − 4 yr", "P", ("F_broadG_h3y1", "F_broadG_h3y4")),
    ("within-institution β (main), 5-yr sample, 5 − 4 yr", "F", ("FE2_h5y5", "FE2_h5y4")),
    ("within-institution β (main), 1-yr sample, 1 − 4 yr", "F", ("FE2_h1y1", "FE2_h1y4")),
    ("within-institution β (main), common programs, 5 − 4 yr", "F", ("FE2_h3y5", "FE2_h3y4")),
    ("within-institution β (main), common programs, 1 − 4 yr", "F", ("FE2_h3y1", "FE2_h3y4")),
    ("within-institution β (a), common programs, 5 − 4 yr", "F", ("FE0_h3y5", "FE0_h3y4")),
    ("within-institution β (a), common programs, 1 − 4 yr", "F", ("FE0_h3y1", "FE0_h3y4")),
    ("within-institution β (field × {SAT, G}), common programs, 5 − 4 yr", "F", ("FE2sg_h3y5", "FE2sg_h3y4")),
    ("within-institution β (field × {SAT, G}), common programs, 1 − 4 yr", "F", ("FE2sg_h3y1", "FE2sg_h3y4")),
    ("within-institution β (main, weighted), common programs, 5 − 4 yr", "F", ("FE2w_h3y5", "FE2w_h3y4")),
    ("within-institution β (main, weighted), common programs, 1 − 4 yr", "F", ("FE2w_h3y1", "FE2w_h3y4")),
]


def hz_diffs(ctx):
    out = []
    for lab, kind, (a, b) in HZ_DIFF_LIST:
        if kind == "P":
            q = ctx["D"][(a, b)]
            out.append(dict(lab=lab, diff=q["diff"], lo=q["lo"], hi=q["hi"], nd=3))
        else:
            q = ctx["FE"][f"pair_{a}_{b}"]
            out.append(dict(lab=lab, diff=q["diff"], lo=q["ci"][0], hi=q["ci"][1], nd=4))
    return out


def hz_diff_sentence(ctx):
    dd = hz_diffs(ctx)
    sig = [d for d in dd if d["lo"] > 0 or d["hi"] < 0]
    s = (f"Of the {len(dd)} paired horizon differences (listed in the horizon section), "
         f"{len(sig)} exclude{'s' if len(sig) == 1 else ''} zero" + (" (" + "; ".join(f"{d['lab']} {_f(d['diff'], d['nd'])} "
                                                        f"{_ci(d['lo'], d['hi'], d['nd'])}" for d in sig) + ")"
                                       if sig else "") + " and the rest are imprecise. ")
    s += _hz_diff_verdict(sig, dd)
    if 2 * len(sig) <= len(dd):
        s += (" Whether any remainder exists is tested by the estimate pooled over horizons (an average over these "
              "cohorts), not by counting separately significant horizons: with three noisy estimates of a small "
              "quantity, 'significant at every horizon' can fail even when the quantity is the same at each.")
    return s


def _hz_diff_verdict(sig, dd):
    if not sig:
        return "So the horizons show no evidence that the remainder differs between cohorts."
    if len(sig) == 1:
        return "So the horizons show little evidence that the remainder differs between cohorts."
    if 2 * len(sig) <= len(dd):
        return ("So there is some evidence that the remainder is smaller for the 1-year or 5-year cohorts than for the "
                "4-year cohort in the comparisons named, but most differences are imprecise.")
    return "In most comparisons the remainder is significantly smaller at the other horizon than at 4 years."


def hz_diff_short(ctx):
    dd = hz_diffs(ctx)
    sig = [d for d in dd if d["lo"] > 0 or d["hi"] < 0]
    s = (f"{len(sig)} of {len(dd)} paired horizon differences exclude{'s' if len(sig) == 1 else ''} zero"
         + (" (" + "; ".join(f"{d['lab']} {_f(d['diff'], d['nd'])} {_ci(d['lo'], d['hi'], d['nd'])}" for d in sig) + ")"
            if sig else ""))
    return s + ". " + _hz_diff_verdict(sig, dd)[:-1]


def rel_hz_sentence(REL):
    """Which horizon estimates the lower-bound reliability correction makes clearly nonzero."""
    lab = {"F_broadG_h5y4": "4-year (5-year sample)", "F_broadG_h5y5": "5-year (5-year sample)",
           "F_broadG_h1y4": "4-year (1-year sample)", "F_broadG_h1y1": "1-year (1-year sample)",
           "F_broadG_h3y4": "4-year (common programs)", "F_broadG_h3y1": "1-year (common programs)",
           "F_broadG_h3y5": "5-year (common programs)"}
    yes = [lab[c] for c in lab if REL[c]["LB"]["lo"] > 0]
    no = [lab[c] for c in lab if REL[c]["LB"]["lo"] <= 0]
    out = (f"With the lower-bound correction the 95% CI lies above zero for {', '.join(yes) if yes else 'none of these'}"
           f" and includes zero for {', '.join(no) if no else 'none'}.")
    if REL["F_broadG_h5y5"]["LB"]["lo"] <= 0 and REL["F_broadG_h3y5"]["LB"]["lo"] <= 0:
        out += " The statement that the correction makes the remainder clearly nonzero does not carry over to 5-year earnings."
    return out


def write_md(ctx):
    S, D, T, FE, H, SH, REL = ctx["S"], ctx["D"], ctx["T"].set_index("field"), ctx["FE"], ctx["HR"], ctx["SH"], ctx["REL"]
    FEPF, av, cov, O = ctx["FEPF"], ctx["avail"], ctx["cov"], FE["oster"]
    L = []
    w = L.append
    main = S["F_broadG"]
    full = [c for c in FULL_SPECS if S[c]["k"] >= 3]
    excl = [c for c in full if S[c]["lo"] > 0]
    incl = [c for c in full if S[c]["lo"] <= 0 <= S[c]["hi"]]
    tost = [c for c in full if S[c]["tost"]]
    max_hi = max(S[c]["hi"] for c in full)
    pf = FE["FE2_pf"]
    ci_pos = [f for f in pf["beta_f"].index if pf["ci_lo"][f] > 0]
    ci_neg = [f for f in pf["beta_f"].index if pf["ci_hi"][f] < 0]
    fe2, fe0, fe0s = FE["FE2"], FE["FE0"], FE["FE0_s"]
    nofe = O["beta_s0"]
    dstar = [O[f"delta_{a}_{b}"] for a in ("s0", "s1") for b in ("13", "half")]
    # ---- earnings-horizon replication (same programs) ----
    P1, P1b, P5, P5b = S["F_broadG_h1y1"], S["F_broadG_h1y4"], S["F_broadG_h5y5"], S["F_broadG_h5y4"]
    PC = {h: S[f"F_broadG_h3{h}"] for h in ("y4", "y1", "y5")}
    F1, F1b, F5, F5b = FE["FE2_h1y1"], FE["FE2_h1y4"], FE["FE2_h5y5"], FE["FE2_h5y4"]
    FC = {h: FE[f"FE2_h3{h}"] for h in ("y4", "y1", "y5")}
    A0 = {h: FE[f"FE0_h3{h}"] for h in ("y4", "y1", "y5")}
    pos = lambda q: q["lo"] > 0                               # partial: normal-approximation CI excludes 0
    fpos = lambda r: r["boot_ci"][0] > 0                      # within-inst.: percentile CI excludes 0
    hzf = {h: FEPF.get(f"FE2_h3{h}") for h in ("y4", "y1", "y5")}
    cs3 = " / ".join(_f(FE[f"FE2_h3{h}_pf"]["beta_f"]["computer_science"], 3) for h in ("y4", "y1", "y5"))
    ci_pct = lambda q: _ci(q["pct_lo"], q["pct_hi"])
    # ---- horizon-pooled estimates (average of 4-, 1-, 5-yr on the common programs; shared bootstrap draws) ----
    PAV = S["F_broadG_h3avg"]
    AV = {fam: FE[f"{fam}_h3avg"] for fam in HZ_FE_FAM}
    pooled = [("partial", pos(PAV))] + [(fam, fpos(AV[fam])) for fam in ("FE0", "FE2sg", "FE2", "FE2w", "FE2v")]
    n_pool_pos = sum(b for _, b in pooled)
    dep_txt = ("is distinguishable from zero in every pooled specification" if n_pool_pos == len(pooled) else
               "is not distinguishable from zero in any pooled specification" if n_pool_pos == 0 else
               "depends on the specification and on how programs are weighted")
    PF = FE["pf_h3avg"]
    pfb = PF["beta_f"]
    pf_pos = [f for f in pfb.sort_values(ascending=False).index if PF["lo"][f] > 0]
    pf_neg = [f for f in pfb.sort_values().index if PF["hi"][f] < 0]
    gr = PF["groups"]
    eng = gr["eng"]["fields"]
    pf_pos_eng = [f for f in pf_pos if f in eng]
    # CS / engineering fields whose pooled CI is above zero and whose slope is positive at every horizon
    eng_stable = [f for f in pf_pos_eng if all(FE[f"FE2_h3{h}_pf"]["beta_f"][f] > 0 for h in HZ3)]
    concentrated = (gr["eng"]["lo"] > 0 and gr["other"]["lo"] <= 0 and len(pf_pos_eng) * 2 > len(pf_pos))
    bh_n = [f for f in pfb.index if PF["bh_n"][f]]
    bh_p = [f for f in pfb.index if PF["bh"][f]]
    bh_n_ci0 = [f for f in bh_n if PF["lo"][f] <= 0 <= PF["hi"][f]]      # pass normal-p BH, percentile CI covers 0
    h55 = [f for f in s55.HIGH if f in pfb.index]             # scripts/55 HIGH, named before this analysis
    assert "computer_science" in h55
    pp_raw = lambda f: (f"0 (<{2 / int(PF['n_fin'][f]):.3f})" if int(PF["n_far"][f]) == 0 else _p(PF["p"][f]))
    pf_lab = lambda f: (f"{LAB.get(f, f)} {_f(pfb[f], 3)} {_ci(PF['lo'][f], PF['hi'][f], 3)} "
                        f"({int(PF['n_by_field'][f])} programs)")
    betas_pool = [AV[fam]["beta"] for fam in ("FE0", "FE2sg", "FE2", "FE2w", "FE2v")]
    ex_word = lambda b: "is above zero" if b else "includes zero"
    # ---- heterogeneity of the remainder across fields: both designs ----
    PFS, POW = FE["pf_split"], ctx["POW"]
    PWF, PWB, PWS = POW["F_broadG"], POW["beta_f"], POW["broad"]
    bsd, biq = FEPF["FE2_bootsd"], FEPF["FE2_bootiqr"]
    kpf = len(pf["beta_f"])
    pbin_pf = float(binom.sf(len(ci_pos) - 1, kpf, 0.025))          # P(>= this many CIs above 0 | no effect, indep.)
    pbin_pool = float(binom.sf(len(pf_pos) - 1, len(pfb), 0.025))
    pfs_perm = PFS["main"]["p"] < 0.05                      # permutation p (conditional on the realized estimates)
    pfs_sim = PWB["p_sim"] < 0.05                            # against the simulated tau = 0 distribution
    pfs_sig = pfs_perm and pfs_sim
    # how to describe the reproducibility of the beta_f ordering across halves of institutions
    repro_vb = "reproduces" if pfs_sig else "tends to reproduce" if pfs_perm else "does not clearly reproduce"
    repro_note = ("" if pfs_sig or not pfs_perm else
                  " (significant by field-label permutation, borderline against a more conservative simulated null)")
    # ---- wild cluster bootstrap tests (shared institutions; shared programs across horizons) ----
    WD = FE["wild"]
    WQ, WC, WH, WN4, WNA = WD["Q4"], WD["hz_corr"], WD["Qh"], WD["cnt4"], WD["cnt_avg"]
    pfloor = 1 / (1 + WD["B"])
    pw = lambda p: (f"≤ {pfloor:.3f}" if p <= pfloor + 1e-12 else _p(p))              # wild p, resolution-limited
    pe = lambda p: (f"≤ {pfloor:.3f}" if p <= pfloor + 1e-12 else f"= {_p(p)}")       # same, with '=' when above floor
    k41, k45, k15 = ("FE2_h3y4", "FE2_h3y1"), ("FE2_h3y4", "FE2_h3y5"), ("FE2_h3y1", "FE2_h3y5")
    trip = lambda f_: " / ".join(f_(WC[kk]) for kk in (k41, k45, k15))
    xh_null_sig = [kk for kk in (k41, k45, k15) if WC[kk]["WCR"]["p"] < 0.05]
    xh_short = (f"r = {trip(lambda q: _f(q['r'], 2))} for 4 vs 1, 4 vs 5 and 1 vs 5 years; with one common β and the "
                f"same programs the expected r is {trip(lambda q: _f(q['WCR']['mean'], 2))}, p = "
                f"{trip(lambda q: pw(q['WCR']['p']))}")
    xh_verdict = ("so the correlation is not clearly more than reusing the same programs produces, and it is not "
                  "evidence that the field differences are stable across cohorts" if not xh_null_sig else
                  "so the correlation is partly more than reusing the same programs produces, but it is not an "
                  "independent replication")
    xh_sent = (f"The per-field slopes correlate across horizons (r = {trip(lambda q: _f(q['r'], 2))} for 4 vs 1, 4 vs 5 "
               f"and 1 vs 5 years), but this is not a replication: the three horizons use the same "
               f"{WD['hz_cells']:,} programs, prestige measure and controls, and program-level residuals persist across "
               f"cohorts (residuals of the common-β model correlate {trip(lambda q: _f(q['rescorr_WCR'], 2))}), so the "
               f"slopes would correlate even if every field had the same β. With one common β per horizon imposed (wild "
               f"cluster bootstrap, the same institution weights at every horizon, {WD['B']} replicates) the expected "
               f"correlation is {trip(lambda q: _f(q['WCR']['mean'], 2))} (95th percentile "
               f"{trip(lambda q: _f(q['WCR']['p95'], 2))}), and the observed values give p = "
               f"{trip(lambda q: pw(q['WCR']['p']))} (Spearman {trip(lambda q: pw(q['WCR']['p_s']))}; resampling the "
               f"per-field model's residuals instead, which does not impose the null on the residuals, p = "
               f"{trip(lambda q: pw(q['WCU']['p']))}); {xh_verdict}")
    wq_all, wq_big = WQ[("all", "iqr")], WQ[("big", "iqr")]
    k_big = wq_big["k"]
    wq_sent = (f"a wild-cluster Q test, which imposes one common β and keeps the dependence between fields that share "
               f"institutions, gives p {pe(wq_all['WCR']['p'])} over the {wq_all['k']} fields (weights from the IQR-based "
               f"bootstrap SEs; bootstrap-SD weights p {pe(WQ[('all', 'sd')]['WCR']['p'])}, per-replicate CR1 weights "
               f"p {pe(WQ[('all', 'cr1')]['WCR']['p'])}) and p {pe(wq_big['WCR']['p'])} over the {k_big} fields with "
               f"≥ {PF_SPLIT_NMIN} programs (SD weights {pw(WQ[('big', 'sd')]['WCR']['p'])}, CR1 weights "
               f"{pw(WQ[('big', 'cr1')]['WCR']['p'])})")
    wq_short = (f"wild-cluster Q p {pe(wq_all['WCR']['p'])} over {wq_all['k']} fields and p {pe(wq_big['WCR']['p'])} "
                f"over the {k_big} with ≥ {PF_SPLIT_NMIN} programs")
    fe_het = all(WQ[("all", wt)]["WCR"]["p"] < 0.05 for wt in ("iqr", "sd", "cr1"))
    cnt4_sent = (f"{WN4['n_pos']} fields are above zero at 1.96 IQR-based bootstrap SE; a wild cluster bootstrap with "
                 f"β_f = 0 for every field, which keeps the shared institutions, gives on average "
                 f"{WN4['WCR']['mean_pos']:.1f} such fields and P(≥ {WN4['n_pos']}) {pe(WN4['WCR']['p_pos'])}")
    cnta_sent = (f"at 1.96 IQR-based SE, a looser rule than the percentile CIs for small, heavy-tailed fields, "
                 f"{WNA['n_pos']} fields are above zero, against {WNA['WCR']['mean_pos']:.1f} on average (95th "
                 f"percentile {WNA['WCR']['p95_pos']:.0f}) under a wild cluster bootstrap with β_f = 0 for every field, "
                 f"P(≥ {WNA['n_pos']}) {pe(WNA['WCR']['p_pos'])}")
    n_eng_big = len(PFS["eng_in_big"])
    spl = lambda q: f"{_f(q['rel'], 2)} (p = {_p(q['p'])})"
    split_sent = (f"split-half reliability {_f(PFS['main']['rel'], 2)}, Spearman {_f(PFS['main']['rel_s'], 2)}, on "
                  f"the {PFS['main']['k']} fields with ≥ {PF_SPLIT_NMIN} programs: permutation p = "
                  f"{_p(PFS['main']['p'])}, p = {_p(PWB['p_sim'])} against a simulated null that also lets the fields' "
                  f"estimates disperse by chance, χ² Q test on the same fields p = {_p(PWB['Q_p'])} with bootstrap-SD SEs "
                  f"and {_p(PWB['Q_p_iqr'])} with IQR-based SEs; "
                  f"without computer science {spl(PFS['main_nocs'])}, without computer science and the "
                  f"{n_eng_big - 1} engineering fields among them {spl(PFS['main_noeng'])}")
    rng_txt = lambda P: (f"any τ from {P['tau_lo']:.3f} to {P['tau_hi']:.3f}" if np.isfinite(P["tau_lo"]) else "no τ on the grid")
    p80_txt = lambda P: (f"τ ≥ {P['tau_p80']:.3f}" if np.isfinite(P["tau_p80"]) else f"no τ up to {P['grid_max']:.2f}")
    cons_ = lambda P, t: bool(np.isfinite(P["tau_lo"]) and P["tau_lo"] <= t + 1e-9 and t - 1e-9 <= P["tau_hi"])
    part_power_sent = (f"given these {PWF['k']} fields' bootstrap SEs (mean {PWF['se_mean']:.3f}), the expected split-half "
                       f"reliability is {_f(PWF['exp_zero'], 2)} at τ = 0, {_f(PWF['exp_reml'], 2)} at τ = "
                       f"{PWF['tau_reml']:.3f} (the REML estimate), {_f(PWF['exp_mid'], 2)} at τ = 0.08 and "
                       f"{_f(PWF['exp_reml_hi'], 2)} at τ = {PWF['tau_reml_hi']:.3f} (its 95% upper bound); the test "
                       f"reaches 80% power only for {p80_txt(PWF)}, and the observed {_f(PWF['obs'], 2)} is consistent "
                       f"with {rng_txt(PWF)}")

    # ---- where the pooled remainder sits (weightings), leverage-corrected wild tests, nonlinear pricing ----
    NL, GW, XD, LOO = FE["nl"], PF["groups_w"], PF["xh_draws"], FE["loo_eng"]
    cs = "computer_science"
    nlg, nla, nl0 = NL["FE2cg"], NL["FE2ca"], NL["FE2"]
    wqa, wqm, wqb, wqn = WQ[("all", "iqr")], WQ[("mid", "iqr")], WQ[("big", "iqr")], WQ[("mid_noeng", "iqr")]
    lev = lambda q: pw(q["WCR3"]["p"])
    # computer science: pooled test, 3-field correction, the 4-yr main spec and both nonlinear variants
    cs_all_variants = bool(PF["lo"][cs] > 0 and PF["lo3"][cs] > 0 and all(NL[n]["lo"][cs] > 0 for n in NL_SPECS))
    # heterogeneity: linear pricing (wild Q with plain and leverage-corrected restricted residuals, all fields) and on
    # the well-estimated fields under cubic brand pricing (chi2 with IQR / SD SEs, wild WCR / WCR3)
    het_lin = bool(wqa["WCR"]["p"] < 0.05 and wqa["WCR3"]["p"] < 0.05)
    ps_cg_big = [nlg["het"][("big", "iqr")]["p"], nlg["het"][("big", "sd")]["p"],
                 nlg["het"][("big", "wild_iqr")]["WCR"]["p"], nlg["het"][("big", "wild_iqr")]["WCR3"]["p"]]
    het_nl_robust = max(ps_cg_big) < 0.05
    prange = lambda ps: f"{_p(min(ps))}–{_p(max(ps))}"
    eng_cg, eng_ca = nlg["eng_mid"]["lo"] > 0, nla["eng_mid"]["lo"] > 0
    ENG_ = [f for f in eng if f != cs]
    BUS_ = GW["sets"]["bus"]
    srt = lambda fl_: sorted(fl_, key=lambda f: -pfb[f])
    eng_pos = [f for f in srt(ENG_) if pfb[f] >= 0.01]
    eng_zero = [f for f in srt(ENG_) if abs(pfb[f]) < 0.01]
    eng_negf = [f for f in srt(ENG_) if pfb[f] <= -0.01]
    bus_pos = [f for f in srt(BUS_) if pfb[f] >= 0.01]
    bus_rest = [f for f in srt(BUS_) if pfb[f] < 0.01]
    vals = lambda fl_: ", ".join(f"{LAB.get(f, f)} {_f(pfb[f], 3)}" for f in fl_)
    names = lambda fl_: ", ".join(LAB.get(f, f) for f in fl_)
    gwm = lambda g, wn: f"{_f(GW[(g, wn)]['mean'], 4)} {_ci(GW[(g, wn)]['lo'], GW[(g, wn)]['hi'], 4)}"
    gwd = lambda c, wn: f"{_f(GW[(c, wn)]['diff'], 4)} {_ci(GW[(c, wn)]['lo'], GW[(c, wn)]['hi'], 4)}"
    k_rest = len(GW["sets"]["rest"])
    labp = GW["labels"]
    # cross-horizon: per-field draw correlations and the centred-draw (pure sampling error) benchmark
    xk = (k41, k45, k15)
    trip_x = lambda f_: " / ".join(f_(XD[kk]) for kk in xk)
    cs_dr = trip_x(lambda q: f"{q['draw_r'][cs]:+.2f}")
    xh_all_ns = all(WC[kk][kind]["p"] >= 0.05 for kk in xk for kind in ("WCR", "WCR3")) and \
        all(XD[kk]["all"]["share"] >= 0.05 for kk in xk)
    sh_all = trip_x(lambda q: f"{100 * q['all']['share']:.0f}%")
    sh_mid = trip_x(lambda q: f"{100 * q['mid']['share']:.0f}%")
    xd_sent = (f"Pure sampling error gives correlations of that size too: the per-field bootstrap draws of the three "
               f"horizons share the institution draws, and a field's draws correlate between horizons (median over "
               f"fields {trip_x(lambda q: _f(q['draw_r_med'], 2))}; computer science {cs_dr}); centring the draws on "
               f"the estimates leaves sampling error only, and its cross-field correlation between horizons reaches "
               f"the observed r in {sh_all} of the draws ({sh_mid} on the {XD[k41]['mid']['k']} fields with "
               f"≥ {PF_NMIN_MID} programs)")
    # leave-out: pooled main spec without the CS / engineering fields
    loo_w_pos = LOO["FE2w"]["ci"][0] > 0
    eng_prec3 = sorted(ENG_, key=lambda f: -PF["w_precision"][f])[:3]
    bus_like_eng = all(abs(GW[("bus", wn)]["mean"] - GW[("eng", wn)]["mean"]) < 0.01 for wn in ("equal", "precision", "programs"))
    eq_largest = GW[("eng-other", "equal")]["diff"] >= max(GW[("eng-other", "precision")]["diff"],
                                                          GW[("eng-other", "programs")]["diff"])

    w("# Selectivity, pushed further: does department prestige carry placement information beyond institutional status?")
    w("")
    w("Public data only (College Scorecard Field-of-Study and institution files; Wapman et al. 2022 published field and "
      "academia-wide ranks). Descriptive; no causal claim. Every number below is produced by "
      f"`scripts/58_selectivity_deep.py` (seed {SEED}; joint institution bootstrap B = {B_BOOT}; institution cluster "
      f"bootstrap B = {B_FE} for the within-institution design; {SPLIT_K} random half-splits and {N_PERM} field "
      "permutations for the split-half checks). Per-field table: `data/interim/selectivity_deep.csv`; figure: "
      "`outputs/figures/selectivity_deep.png`. It extends `scripts/55_selectivity.py` (SELECTIVITY_RESULT.md) and "
      "reproduces its broad partial, horse race and within-institution β_f (checks under Method).")
    w("")
    w("Two readings are being separated. **(A)** The academy's field-specific hierarchy carries placement information "
      "of its own. **(B)** Within-field coupling reflects how much each field pays for institutional selectivity and "
      "brand, plus whatever student quality those measures miss.")
    w("")
    w("Notation: F = department (field) prestige, −published Wapman field rank; G = academia-wide brand, −published "
      "Wapman Academia rank; Y = Scorecard median earnings 4 years after the bachelor's (1-year and 5-year medians "
      "in the horizon replication); 'selectivity set' = institution "
      "SAT_AVG, ADM_RATE, institution Pell share, control and the state earnings level (the scripts/55 broad spec). "
      "Partial correlations are on ranks within field.")
    w("")
    w("## Answer")
    w("")
    w("**Key question: after all public controls, how much field-specific prestige signal remains, in how many fields, "
      "and is it distinguishable from zero?**")
    w("")
    w("**Short answer: a small remainder is left. Averaged across fields it is borderline, and whether it is "
      f"distinguishable from zero {dep_txt}. "
      + ("Computer science is the one field whose remainder is individually clear: its within-institution premium is "
         "above zero in the per-field test pooled over the three earnings horizons, also with a correction for the "
         "three fields named in advance, and with 4-year earnings in the main specification and in both nonlinear "
         "variants (field-specific cubic pricing of brand, and of brand, SAT and admit rate). " if cs_all_variants else
         "No single field's remainder is above zero in the main specification and both nonlinear variants. ")
      + (f"Elsewhere the remainder is positive in several engineering fields ({names(eng_pos)}) and in "
         f"{'the' if len(bus_pos) == len(BUS_) else len(bus_pos)} business fields ({names(bus_pos)}), "
         + ("but none of them passes a multiple-testing correction across the "
            f"{len(pfb)} fields with percentile-bootstrap p"
            + (f" ({names([f for f in eng_pos + bus_pos if PF['bh_n'][f]])} does with a normal approximation, which "
               f"understates the spread of small fields' draws)" if any(PF["bh_n"][f] for f in eng_pos + bus_pos) else "")
            + "; " if not any(PF["bh"][f] for f in eng_pos + bus_pos) else
            f"{names([f for f in eng_pos + bus_pos if PF['bh'][f]])} after a multiple-testing correction; ")
         + f"it is near zero in {names(eng_zero)} and zero or below on average in the remaining {k_rest} fields. "
         f"These groups were read off the results, and how far the engineering fields stand out depends on how the "
         f"fields are weighted. " if eng_pos else "")
      + ("Whether the fields' remainders really differ is not settled: with linear field-specific pricing of "
         "brand and selectivity the per-field slopes are heterogeneous by wild-cluster tests (also with "
         "leverage-corrected residuals), but once each field may price brand nonlinearly the heterogeneity test on "
         f"the {nlg['het'][('big', 'iqr')]['k']} well-estimated fields gives p = {prange(ps_cg_big)} and the "
         f"split-half reliability of their ordering falls from {_f(nl0['split']['main']['rel'], 2)} to "
         f"{_f(nlg['split']['main']['rel'], 2)}. " if het_lin and not het_nl_robust else
         "The per-field slopes differ between fields, also once brand is priced nonlinearly. " if het_lin else
         "Differences between fields are not reliably detected. ")
      + ("The per-field slopes also correlate across the 4-, 1- and 5-year earnings horizons, but those horizons reuse "
         "the same programs and institutions, and the correlation is within what that shared sampling error "
         "produces, so it is not counted as evidence. " if xh_all_ns else
         "The per-field slopes also correlate across the 4-, 1- and 5-year earnings horizons, but those horizons reuse "
         "the same programs, so this is not an independent replication. ")
      + "Public data cannot tell whether the remainder reflects the department's academic standing (A) or "
      "program-level student selection that institution-level measures miss (B).**")
    w("")
    w(f"How much: with 4-year earnings on the full SAT sample, the mean within-field partial correlation of department "
      f"prestige with earnings net of the selectivity set and brand is {_est(main)} ({main['k']} fields), against "
      f"{_f(S['raw']['mean'])} raw, and the within-institution β (main spec) is {_fe_ci(fe2)} log points per SD of "
      f"department prestige. The same Scorecard release has 1-year and 5-year medians for other graduating cohorts. "
      f"On the {FC['y4']['n_cells']:,} programs released at all three horizons, averaging the 4-, 1- and 5-year "
      f"estimates (CI from the average of shared bootstrap draws, so it allows for the dependence between horizons): "
      f"mean partial correlation {_est(PAV)} ({PAV['k']} fields; percentile CI {ci_pct(PAV)}); within-institution β "
      f"{_fe_ci(AV['FE0'])} with institution and field fixed effects only (spec a), {_fe_ci(AV['FE2sg'])} with "
      f"field-specific returns to SAT and brand (the design as specified: institution FE + field FE + field × SAT + "
      f"field × G), and {_fe_ci(AV['FE2'])} when field-specific returns to admit rate and institution Pell share are "
      f"added too (main spec; CI {ex_word(fpos(AV['FE2']))}). Weighting programs by the number of graduates behind "
      f"each median changes the main-spec average to {_fe_ci(AV['FE2w'])} (weights = each horizon's own count) or "
      f"{_fe_ci(AV['FE2v'])} (4-year count). So the pooled remainder is about "
      f"{100 * min(betas_pool):.1f}–{100 * max(betas_pool):.1f}% of pay per within-field SD of department prestige, "
      f"and a partial correlation of about {_f(PAV['mean'], 2)}. Horizon by horizon the estimates are less precise: "
      f"the 95% CI of the 5-year estimate alone {ex_word(pos(PC['y5']))} in the partial-correlation design "
      f"({_est(PC['y5'])}) and {ex_word(fpos(FC['y5']))} in the unweighted main spec ({_fe_ci(FC['y5'])}); "
      f"weighted by cohort size it {ex_word(fpos(FE['FE2w_h3y5']))} ({_fe_ci(FE['FE2w_h3y5'])}). "
      + hz_diff_short(ctx) + " (Answer 3).")
    w("")
    w(f"Where: per field (within institutions, main spec, 4/1/5-year average), {len(pf_pos)} of {len(pfb)} fields have "
      f"a 95% CI above zero and {len(pf_neg)} below, against ≈{0.025 * len(pfb):.2f} per tail expected by chance "
      f"(if the fields were independent, {len(pf_pos)} or more above zero would occur with probability "
      f"{_p(pbin_pool)}; they share institutions: {cnta_sent}; with leverage-corrected residuals the null mean is "
      f"{WNA['WCR3']['mean_pos']:.1f} and P(≥ {WNA['n_pos']}) {pe(WNA['WCR3']['p_pos'])}). "
      f"Above zero: {'; '.join(pf_lab(f) for f in pf_pos) or 'none'}. Below zero: "
      f"{'; '.join(pf_lab(f) for f in pf_neg) or 'none'}. Computer science is the clearest: "
      f"{int(PF['n_far'][cs])} of {int(PF['n_fin'][cs])} bootstrap draws are at or below zero, BH-FDR across all "
      f"{len(pfb)} fields keeps it with percentile-bootstrap p"
      + ("" if PF["bh"][cs] else " (it does not)")
      + f", and it is one of the {len(s55.HIGH)} fields named in advance in scripts/55 (its 98.33% CI, the correction "
      f"for 3 fields, is {_ci(PF['lo3'][cs], PF['hi3'][cs], 3)}). The other engineering fields: {vals(srt(ENG_))}; the "
      f"business fields: {vals(srt(BUS_))} (unadjusted CI above zero: "
      f"{names([f for f in srt(ENG_ + BUS_) if PF['lo'][f] > 0]) or 'none'}; BH-FDR with percentile p keeps "
      f"{names([f for f in srt(ENG_ + BUS_) if PF['bh'][f]]) or 'none of them'}, with normal p "
      f"{names([f for f in srt(ENG_ + BUS_) if PF['bh_n'][f]]) or 'none of them'}). Grouped after seeing these "
      f"results: the {len(eng)} computer science and engineering fields average {gwm('eng', 'equal')} with equal "
      f"weights, {gwm('eng', 'precision')} when each field is weighted by the precision of its pooled estimate "
      f"(1 / bootstrap variance; the three most precisely estimated engineering fields are {vals(eng_prec3)}) and "
      f"{gwm('eng', 'programs')} weighted by program count; the other {GW[('other', 'equal')]['k']} fields "
      f"{_f(GW[('other', 'equal')]['mean'], 4)} / {_f(GW[('other', 'precision')]['mean'], 4)} / "
      f"{_f(GW[('other', 'programs')]['mean'], 4)} (difference {gwd('eng-other', 'equal')} / "
      f"{gwd('eng-other', 'precision')} / {gwd('eng-other', 'programs')}). The {len(BUS_)} business fields average "
      f"{gwm('bus', 'equal')} (precision-weighted {gwm('bus', 'precision')}; by programs {gwm('bus', 'programs')}), "
      + ("about as much as the engineering group, " if bus_like_eng else "")
      + f"and the remaining {k_rest} fields {_f(GW[('rest', 'equal')]['mean'], 4)} "
      f"/ {_f(GW[('rest', 'precision')]['mean'], 4)} / {_f(GW[('rest', 'programs')]['mean'], 4)}. Random sets of "
      f"{labp['eng']['m']} field labels reach the observed CS/engineering-minus-other contrast in "
      f"{100 * labp['eng']['p_eq']:.1f}% of {N_PERM} draws with equal weights ({100 * labp['eng']['p_pr']:.1f}% with "
      f"precision weights), and random sets of {labp['bus']['m']} reach the business-minus-other contrast in "
      f"{100 * labp['bus']['p_eq']:.1f}% ({100 * labp['bus']['p_pr']:.1f}%), before any allowance for having chosen "
      f"the groups after seeing the results. Without the {len(LOO['dropped'])} computer science and engineering "
      f"fields (models refitted on the other {LOO['FE2']['n_fields']} fields), the pooled main-spec β is "
      f"{_f(LOO['FE2']['beta'], 4)} {_ci(*LOO['FE2']['ci'], 4)} unweighted and {_f(LOO['FE2w']['beta'], 4)} "
      f"{_ci(*LOO['FE2w']['ci'], 4)} weighted by cohort size, against {_fe_ci(AV['FE2'])} and {_fe_ci(AV['FE2w'])} "
      f"with them: "
      + ("the cohort-size-weighted remainder does not rest on those fields alone. "
         + ("The two statements 'distinguishable from zero depending on weighting' and 'larger in computer science "
            "and engineering' therefore lean on different weightings: the pooled average is pulled up by weighting "
            "large programs more, while the engineering contrast is largest when every field counts equally. "
            if eq_largest else "") if loo_w_pos else
         "without them the weighted remainder is smaller and its CI "
         + ("includes zero, " if LOO["FE2w"]["ci"][0] <= 0 else "stays above zero, ")
         + "so part, but not all, of it sits in those fields. ")
      + f"Computer science's slope is {cs3} at 4 / 1 / 5 years, but that is close to one estimate seen three times "
      f"(its bootstrap draws correlate {cs_dr} between the 4 v 1, 4 v 5 and 1 v 5-year estimates), so the per-field "
      f"evidence rests on the pooled test. Fields named in advance in scripts/55 as high-coupling "
      f"({', '.join(LAB.get(f, f) for f in s55.HIGH)}), where estimable here: " + "; ".join(pf_lab(f) for f in h55)
      + ".")
    w("")
    w(f"Do the fields differ? {xh_sent}. {xd_sent}. So the correlation across horizons is not evidence that the "
      f"fields differ. The tests that do not reuse programs give a mixed answer. With 4-year earnings on the full "
      f"main-spec sample and linear field-specific slopes, the between-field SD of β_f is τ = {biq['tau']:.3f} "
      f"(χ² Q p = {_p(biq['Qp'])} with IQR-based bootstrap SEs, {_p(bsd['Qp'])} with bootstrap-SD SEs; {wq_sent}). "
      f"The plain restricted-residual null is too narrow for fields with few programs per field-specific parameter "
      f"(its per-field SD is a median {WD['sd_ratio']['WCR']['lt30']:.2f} of the pairs-bootstrap IQR-based SE for the "
      f"{WD['n_size']['lt30']} fields with < {PF_NMIN_MID} programs); with leverage-corrected residuals "
      f"({WD['sd_ratio']['WCR3']['lt30']:.2f}) the wild-cluster Q p is {lev(wqa)} over all {wqa['k']} fields, "
      f"{lev(wqm)} over the {wqm['k']} with ≥ {PF_NMIN_MID} programs, {lev(wqb)} over the {wqb['k']} with ≥ "
      f"{PF_SPLIT_NMIN} and {lev(wqn)} over the {wqn['k']} with ≥ {PF_NMIN_MID} programs outside computer science "
      f"and engineering. The ordering of β_f across fields {repro_vb} across random halves of institutions "
      f"({split_sent}). A linear field × G slope cannot absorb convex (elite-tail) pricing of brand, which is itself a "
      f"reading-B mechanism. With field × {{zG², zG³}} added, β_f correlate {_f(nlg['corr_main_mid'], 2)} with the "
      f"main spec on the fields with ≥ {PF_NMIN_MID} programs, computer science stays at "
      f"{_f(nlg['beta_f'][cs], 3)} {_ci(nlg['lo'][cs], nlg['hi'][cs], 3)}, but on the "
      f"{nlg['het'][('big', 'iqr')]['k']} fields with ≥ {PF_SPLIT_NMIN} programs τ falls from "
      f"{nl0['het'][('big', 'iqr')]['tau']:.3f} to {nlg['het'][('big', 'iqr')]['tau']:.3f} (IQR-based SEs), χ² Q p "
      f"is {_p(nlg['het'][('big', 'iqr')]['p'])} (IQR) / {_p(nlg['het'][('big', 'sd')]['p'])} (SD), the wild-cluster p "
      f"{pw(nlg['het'][('big', 'wild_iqr')]['WCR']['p'])} (restricted residuals) / "
      f"{lev(nlg['het'][('big', 'wild_iqr')])} (leverage-corrected), and the split-half reliability "
      f"{spl(nlg['split']['main'])}; over all {nlg['het'][('all', 'iqr')]['k']} fields χ² Q p is "
      f"{_p(nlg['het'][('all', 'iqr')]['p'])} and the leverage-corrected wild p "
      f"{lev(nlg['het'][('all', 'wild_iqr')])}. With cubic terms in G, SAT and admit rate, the tests on the fields "
      f"with ≥ {PF_SPLIT_NMIN} programs are mixed (χ² Q p {_p(nla['het'][('big', 'iqr')]['p'])} / "
      f"{_p(nla['het'][('big', 'sd')]['p'])}, wild p {pw(nla['het'][('big', 'wild_iqr')]['WCR']['p'])} restricted / "
      f"{lev(nla['het'][('big', 'wild_iqr')])} leverage-corrected), the split-half reliability is "
      f"{spl(nla['split']['main'])}, and the leverage-corrected wild p over the {nla['het'][('all', 'iqr')]['k']} "
      f"estimable fields is {lev(nla['het'][('all', 'wild_iqr')])}. The CS/engineering-minus-other contrast on the fields with ≥ {PF_NMIN_MID} programs is "
      f"{_f(nl0['eng_mid']['diff'], 3)} {_ci(nl0['eng_mid']['lo'], nl0['eng_mid']['hi'], 3)} (linear), "
      f"{_f(nlg['eng_mid']['diff'], 3)} {_ci(nlg['eng_mid']['lo'], nlg['eng_mid']['hi'], 3)} (cubic G) and "
      f"{_f(nla['eng_mid']['diff'], 3)} {_ci(nla['eng_mid']['lo'], nla['eng_mid']['hi'], 3)} (cubic G, SAT, ADM). "
      + ("So general cross-field heterogeneity is significant with linear pricing but not robust to nonlinear "
         "pricing of brand on the well-estimated fields; " if het_lin and not het_nl_robust else
         "So the heterogeneity survives nonlinear pricing of brand; " if het_lin else
         "So heterogeneity is not detected; ")
      + ("what survives all three pricing variants is the computer science premium. " if cs_all_variants else "")
      + "None of these tests favours A over B: program-level selection into majors (for example separately admitted "
      "engineering programs) would also differ between fields and reproduce across halves of institutions.")
    w("")
    w(f"1. **How much, with 4-year earnings: a small remainder.** Net of the selectivity set and brand G, the mean within-field partial "
      f"correlation of department prestige with earnings is {_est(main)} ({main['k']} fields). For comparison, the raw "
      f"mean is {_f(S['raw']['mean'])} ({S['raw']['k']} fields) and the mean net of the selectivity set alone is "
      f"{_f(S['broad']['mean'])} ({S['broad']['k']} fields). Adding the program's own Pell share (the {S['F_all']['k']} "
      f"fields where the FoS file releases it) gives {_est(S['F_all'])}. Pooled over cells, the slope of standardized "
      f"residual earnings on standardized residual prestige is {_f(main['pool_b'])} (SE {main['pool_se']:.3f}, "
      f"clustered by institution). Within institutions, with field-specific returns to SAT, admit rate, Pell share and "
      f"brand, one within-field SD of department prestige goes with {100 * fe2['beta']:.2f}% higher 4-year earnings "
      f"(β = {_fe_ci(fe2)} log points, institution cluster bootstrap; {fe2['n_cells']:,} programs, {fe2['n_inst']} "
      f"institutions, {fe2['n_fields']} fields). Without institution fixed effects the same cells give "
      f"{_f(nofe, 4)}, so about {100 * fe2['beta'] / nofe:.0f}% of that slope is left. Items 1, 2 and 5–7 use "
      f"4-year earnings unless stated; items 3 and 4 add the 1-year and 5-year earnings.")
    head2 = ("yes, in every specification" if len(excl) == len(full) else
             "no, in none of the specifications" if not excl else "only borderline")
    w(f"2. **Distinguishable from zero with 4-year earnings: {head2}.** Across the {len(full)} specifications that net out "
      f"selectivity and brand (with or without institution-wide earnings, program Pell share, gender share, the "
      f"ADM-only sample, Pell/non-Pell outcomes), the 95% CI of the mean excludes zero in {len(excl)} "
      f"({'; '.join(SHORT[c] + ' ' + _est(S[c]) for c in excl)}) and includes it in {len(incl)} "
      f"({'; '.join(SHORT[c] + ' ' + _est(S[c]) for c in incl)}). Point estimates range from "
      f"{_f(min(S[c]['mean'] for c in full))} to {_f(max(S[c]['mean'] for c in full))} (median "
      f"{_f(float(np.median([S[c]['mean'] for c in full])))}) and no upper bound exceeds {_f(max_hi, 2)}. The mean is "
      f"equivalent to zero within ±{EQ_BOUND:.2f} (90% CI inside the bound) in {len(tost)} of the {len(full)} "
      f"({'; '.join(SHORT[c] for c in tost) if tost else 'none'}). The random-effects mean of the main spec is "
      f"{_f(main['mu_re'])} {_ci(main['lo_re'], main['hi_re'])} (this CI treats fields as independent although "
      f"they share institutions). The within-institution β has a 95% bootstrap CI of {_ci(*fe2['boot_ci'], 4)} "
      f"({100 * fe2['boot_p0']:.1f}% of the {B_FE} cluster-bootstrap draws are ≤ 0; CR1 SE {fe2['se']:.4f}, "
      f"z = {fe2['beta'] / fe2['se']:.2f}).")
    d5p, d1f, d5f = D[("F_broadG_h5y5", "F_broadG_h5y4")], FE["pair_FE2_h1y1_FE2_h1y4"], FE["pair_FE2_h5y5_FE2_h5y4"]
    dc5 = D[("F_broadG_h3y5", "F_broadG_h3y4")]
    w(f"3. **Other earnings horizons: pooled over the three horizons the remainder is small and borderline; each "
      f"horizon alone is imprecise.** Pooled on the {FC['y4']['n_cells']:,} common programs (average of the 4-, 1- and "
      f"5-year estimates, CI from the same bootstrap draws): partial correlation {_est(PAV)} ({PAV['k']} fields; "
      f"selectivity-only partial {_est(S['broad_h3avg'])}, raw {_est(S['raw_h3avg'])}, brand net of F and the "
      f"selectivity set {_est(S['G_broadF_h3avg'])}); within-institution β, unweighted: spec (a) {_fe_ci(AV['FE0'])}, "
      f"field × {{SAT, G}} {_fe_ci(AV['FE2sg'])} (per horizon {_f(FE['FE2sg_h3y4']['beta'], 4)} "
      f"{_ci(*FE['FE2sg_h3y4']['boot_ci'], 4)} / {_f(FE['FE2sg_h3y1']['beta'], 4)} / {_f(FE['FE2sg_h3y5']['beta'], 4)}), "
      f"main {_fe_ci(AV['FE2'])}; main weighted by each horizon's program count {_fe_ci(AV['FE2w'])} (per horizon "
      f"{_f(FE['FE2w_h3y4']['beta'], 4)} ({FE['FE2w_h3y4']['se']:.4f}) / {_f(FE['FE2w_h3y1']['beta'], 4)} "
      f"({FE['FE2w_h3y1']['se']:.4f}) / {_f(FE['FE2w_h3y5']['beta'], 4)} ({FE['FE2w_h3y5']['se']:.4f}), CR1 SE; "
      f"5-year CI {_ci(*FE['FE2w_h3y5']['boot_ci'], 4)}); main weighted by the 4-year count "
      f"{_fe_ci(AV['FE2v'])} (per horizon {_f(FE['FE2v_h3y4']['beta'], 4)} / {_f(FE['FE2v_h3y1']['beta'], 4)} / "
      f"{_f(FE['FE2v_h3y5']['beta'], 4)}). Weighting changes the main-spec average by "
      f"{_f(FE['pair_FE2w_h3avg_FE2_h3avg']['diff'], 4)} {_ci(*FE['pair_FE2w_h3avg_FE2_h3avg']['ci'], 4)} ("
      + ("larger programs, whose medians are less noisy, carry a larger premium"
         if FE['pair_FE2w_h3avg_FE2_h3avg']['ci'][0] > 0 else
         "larger programs carry a smaller premium" if FE['pair_FE2w_h3avg_FE2_h3avg']['ci'][1] < 0 else
         "not distinguishable from no change, although it moves the CI " + ("away from" if fpos(AV['FE2w']) else "towards")
         + " zero") + "). "
      f"Horizon by horizon, each comparison keeps the "
      f"programs fixed and changes only the outcome, with shared bootstrap draws, so the differences are paired. "
      f"Partial correlation, main spec: on the {P5['k']} fields of the 5-year sample, 4-year {_est(P5b)} vs 5-year "
      f"{_est(P5)} (difference {_est(d5p)}); on the {P1['k']} fields of the 1-year sample, 4-year {_est(P1b)} vs "
      f"1-year {_est(P1)} (percentile CI {ci_pct(P1)}). Within institutions (main spec): 5-year sample "
      f"({F5['n_cells']:,} programs) 4-year {_fe_ci(F5b)} vs 5-year {_fe_ci(F5)} (difference "
      f"{_f(d5f['diff'], 4)} {_ci(*d5f['ci'], 4)}); 1-year sample ({F1['n_cells']:,} programs) 4-year {_fe_ci(F1b)} vs "
      f"1-year {_fe_ci(F1)} (difference {_f(d1f['diff'], 4)} {_ci(*d1f['ci'], 4)}). On the {FC['y4']['n_cells']:,} "
      f"programs released at all three horizons the within-institution β is {_f(FC['y4']['beta'], 4)} / "
      f"{_f(FC['y1']['beta'], 4)} / {_f(FC['y5']['beta'], 4)} (4 / 1 / 5 years) and the partial correlation "
      f"{_f(PC['y4']['mean'])} / {_f(PC['y1']['mean'])} / {_f(PC['y5']['mean'])} ({PC['y4']['k']} fields; 5 − 4 years "
      f"{_est(dc5)}). The 5-year earnings carry the usual prestige signal: on those programs the raw coupling is "
      f"{_f(S['raw_h3y5']['mean'])} (4-year {_f(S['raw_h3y4']['mean'])}) and the selectivity-only partial "
      f"{_f(S['broad_h3y5']['mean'])} (4-year {_f(S['broad_h3y4']['mean'])}). What falls is the field-specific "
      f"part ({_f(PC['y5']['mean'])} against {_f(PC['y4']['mean'])}), while brand net of F and the selectivity set "
      f"is {_f(S['G_broadF_h3y5']['mean'])} at 5 years against {_f(S['G_broadF_h3y4']['mean'])} at 4 years "
      f"(difference {_est(D[('G_broadF_h3y5', 'G_broadF_h3y4')])}). Even the within-institution slope without field-specific "
      f"returns (spec a, scripts/55's +0.013) is {_fe_ci(A0['y5'])} with 5-year and {_fe_ci(A0['y1'])} with 1-year "
      f"earnings on these programs (4-year {_fe_ci(A0['y4'])}). " + hz_diff_sentence(ctx))
    part_het = main["Qp"] < 0.05 or SH["rel_F_broadG_p"] < 0.05
    head3 = (("in the partial-correlation design no field stands out" if main["n_bh_pos"] == 0 else
              "in the partial-correlation design a few fields stand out")
             + (" and differences between fields are not detected, but that design has little power to detect them"
                if not part_het else " and the fields differ")
             + ("; within institutions the per-field slopes differ with linear field-specific pricing, but not "
                "robustly once brand is priced nonlinearly" if het_lin and not het_nl_robust else
                "; within institutions the per-field slopes differ, also with nonlinear pricing" if het_lin else
                "; within institutions differences are not detected either")
             + (f"; pooled over horizons {len(pf_pos)} fields have a CI above zero"
                + (", and computer science's premium holds in the linear and both nonlinear pricing variants"
                   if cs_all_variants else "")
                if pf_pos else ""))
    w(f"4. **In how many fields: {head3}.** Partial-correlation design: {main['n_pos']} of "
      f"{main['k']} fields are above zero at 1.96 bootstrap SE and {main['n_neg']} below; about {main['exp_fp']:.1f} "
      f"per tail are expected by chance, and {main['n_bh_pos']} survive BH-FDR (5%). This design "
      + ("does not detect differences between fields" if not part_het else "detects differences between fields")
      + f": between-field SD τ = {main['tau']:.3f} (95% upper bound {main['tau_hi']:.3f}; Q p = "
      f"{_p(main['Qp'])}), and the split-half reliability of the cross-field ordering of the field-specific component "
      f"is {_f(SH['rel_F_broadG'], 2)} (permutation p = {_p(SH['rel_F_broadG_p'])}; the selectivity-only partial has "
      f"{_f(SH['rel_broad'], 2)}, p = {_p(SH['rel_broad_p'])}). "
      + ("That is a weak null, not evidence that the fields do not differ: " if not part_het else "For scale: ")
      + part_power_sent + " (simulation in (ii)). "
      f"The within-institution design, which uses more of the data, "
      + ("detects differences with linear field-specific pricing, but not robustly once brand is priced "
         "nonlinearly (below). " if het_lin and not het_nl_robust else
         "detects differences. " if het_lin else "does not detect differences either. ")
      + f"Per-field bootstrap CIs are above zero in {len(ci_pos)} of {kpf} fields "
      f"({', '.join(LAB.get(f, f) + ' ' + _f(pf['beta_f'][f], 3) + ' (' + str(int(pf['n_by_field'][f])) + ' programs)' for f in ci_pos)}) "
      f"and below zero in {len(ci_neg)} "
      f"({', '.join(LAB.get(f, f) + ' ' + _f(pf['beta_f'][f], 3) + ' (' + str(int(pf['n_by_field'][f])) + ' programs)' for f in ci_neg)}); "
      f"about {0.025 * kpf:.1f} per tail are expected by chance (if the fields were independent, {len(ci_pos)} or more "
      f"above zero would occur with probability {_p(pbin_pf)}; they share institutions: {cnt4_sent}), and "
      f"with bootstrap (IQR) SEs {biq['n_bh_pos']} positive and {biq['n_bh_neg']} negative survive BH-FDR. "
      f"Between-field SD of β_f: τ = {biq['tau']:.3f} with IQR-based bootstrap SEs (χ² Q p = {_p(biq['Qp'])}) and "
      f"{bsd['tau']:.3f} with bootstrap-SD SEs (χ² Q p = {_p(bsd['Qp'])}); the χ² test treats fields as independent, "
      f"and {wq_sent}; with leverage-corrected restricted residuals, which widen the null for fields with few programs, "
      f"p = {lev(wqa)} / {lev(wqm)} / {lev(wqb)} / {lev(wqn)} (all fields / ≥ {PF_NMIN_MID} programs / ≥ "
      f"{PF_SPLIT_NMIN} programs / ≥ {PF_NMIN_MID} programs without computer science and engineering). The ordering "
      f"of β_f across fields "
      + repro_vb
      + f" across random halves of institutions: {split_sent} (details in (iii)). At the REML τ = {biq['tau']:.3f} the "
      f"expected reliability for these fields is {_f(PWB['exp_reml_iqr'], 2)}; the observed {_f(PWB['obs'], 2)} is "
      f"consistent with {rng_txt(PWB)} log points. With nonlinear field-specific pricing of brand (field × {{zG², "
      f"zG³}}) the test on the {nlg['het'][('big', 'iqr')]['k']} fields with ≥ {PF_SPLIT_NMIN} programs gives χ² Q p "
      f"{_p(nlg['het'][('big', 'iqr')]['p'])} / {_p(nlg['het'][('big', 'sd')]['p'])} (IQR / SD SEs) and wild p "
      f"{pw(nlg['het'][('big', 'wild_iqr')]['WCR']['p'])} / {lev(nlg['het'][('big', 'wild_iqr')])} (restricted / "
      f"leverage-corrected residuals), and the split-half reliability is {spl(nlg['split']['main'])}; with cubic "
      f"G, SAT and admit rate it is {spl(nla['split']['main'])} (details in (iii)). Computer science ({int(pf['n_by_field']['computer_science'])} programs) has the clearest premium "
      f"({_f(pf['beta_f']['computer_science'], 3)} {_ci(pf['ci_lo']['computer_science'], pf['ci_hi']['computer_science'], 3)} "
      f"log points per SD); its partial correlation in the main spec is {_f(T.loc['computer_science', 'F_broadG'], 2)} "
      f"(SE {T.loc['computer_science', 'se_F_broadG']:.2f}). On the "
      f"{FC['y4']['n_cells']:,} programs released at all three horizons ({hzf['y4']['k']} fields), the fields whose "
      f"within-institution CI is above zero horizon by horizon are, with 4-year earnings: "
      f"{', '.join(LAB.get(f, f) for f in hzf['y4']['ci_pos']) or 'none'}; 1-year: "
      f"{', '.join(LAB.get(f, f) for f in hzf['y1']['ci_pos']) or 'none'}; 5-year: "
      f"{', '.join(LAB.get(f, f) for f in hzf['y5']['ci_pos']) or 'none'} (below zero: "
      f"{len(hzf['y4']['ci_neg'])} / {len(hzf['y1']['ci_neg'])} / {len(hzf['y5']['ci_neg'])}). Across fields the β_f "
      f"correlate between horizons ({xh_short}; centred bootstrap draws, i.e. sampling error alone, reach the observed "
      f"r in {sh_all} of draws; the horizons reuse the same programs and institutions, so this is not a replication; "
      f"horizon section). Computer science: "
      f"{_f(FE['FE2_h3y4_pf']['beta_f']['computer_science'], 3)} / {_f(FE['FE2_h3y1_pf']['beta_f']['computer_science'], 3)} / "
      f"{_f(FE['FE2_h3y5_pf']['beta_f']['computer_science'], 3)} (4 / 1 / 5 years; 5-year CI "
      f"{_ci(FE['FE2_h3y5_pf']['ci_lo']['computer_science'], FE['FE2_h3y5_pf']['ci_hi']['computer_science'], 3)}); its "
      f"bootstrap draws correlate {cs_dr} between horizons, so these are close to one estimate seen three times, not "
      f"three confirmations. "
      f"In the partial-correlation design, fields above / below zero at 1.96 SE on the same programs: "
      f"{PC['y4']['n_pos']} / {PC['y4']['n_neg']} (4-year), {PC['y1']['n_pos']} / {PC['y1']['n_neg']} (1-year), "
      f"{PC['y5']['n_pos']} / {PC['y5']['n_neg']} (5-year), {PAV['n_pos']} / {PAV['n_neg']} (4/1/5-year average; "
      f"BH {PAV['n_bh_pos']} / {PAV['n_bh_neg']}), against ≈{PC['y4']['exp_fp']:.1f} per tail by chance. "
      f"The horizon-by-horizon counts are three noisy looks at the same fields; the per-field test is the one pooled "
      f"over horizons (average of the three slopes on shared draws). Pooled, {len(pf_pos)} fields have a within-"
      f"institution CI above zero ({', '.join(LAB.get(f, f) for f in pf_pos) or 'none'}) and {len(pf_neg)} below "
      f"({', '.join(LAB.get(f, f) for f in pf_neg) or 'none'}), against ≈{0.025 * len(pfb):.2f} per tail by chance. "
      f"BH-FDR (5%) over the {len(pfb)} fields keeps {len(bh_p)}"
      + (f" ({', '.join(LAB.get(f, f) for f in bh_p)})" if bh_p else "")
      + f" with raw percentile-bootstrap p and {len(bh_n)}"
      + (f" ({', '.join(LAB.get(f, f) for f in bh_n)})" if bh_n else "")
      + f" with normal p from the IQR-based bootstrap SE. "
      + (f"The two disagree for two reasons. First, the bootstrap resolution is coarse: computer science has "
         f"{int(PF['n_far']['computer_science'])} of {int(PF['n_fin']['computer_science'])} draws on the far side of zero "
         f"(raw p = {pp_raw('computer_science')}) but normal p = {PF['p_n']['computer_science']:.4f}; with {B_FE} "
         f"draws a percentile p below about 0.002 cannot be resolved, and the first BH threshold is 0.05/{len(pfb)} = "
         f"{0.05 / len(pfb):.4f}. "
         + (f"Second, for small fields the bootstrap distribution is heavy-tailed and the IQR-based SE understates its "
            f"spread, so normal p is too small: {', '.join(LAB.get(f, f) + ' (' + str(int(PF['n_by_field'][f])) + ' programs, percentile CI ' + _ci(PF['lo'][f], PF['hi'][f], 3) + ')' for f in bh_n_ci0)} "
            f"{'passes' if len(bh_n_ci0) == 1 else 'pass'} with normal p although the percentile CI includes zero. "
            if bh_n_ci0 else "")
         + "So whether any single field survives a multiplicity correction across all fields depends on the p-value "
         "method. " if set(bh_p) != set(bh_n) else
         f"(With {B_FE} draws a percentile p below about 0.002 cannot be resolved; the first BH threshold is "
         f"0.05/{len(pfb)} = {0.05 / len(pfb):.4f}.) ")
      + f"Computer science is one of the "
      f"{len(s55.HIGH)} fields named in advance in scripts/55 (HIGH: {', '.join(LAB.get(f, f) for f in s55.HIGH)}), so "
      f"for it a correction for {len(s55.HIGH)} tests is enough: its pooled CI is "
      f"{_ci(PF['lo']['computer_science'], PF['hi']['computer_science'], 3)} and its normal p "
      f"{PF['p_n']['computer_science']:.4f} is "
      + ("below" if PF['p_n']['computer_science'] < 0.05 / len(s55.HIGH) else "not below")
      + f" 0.05/{len(s55.HIGH)} = {0.05 / len(s55.HIGH):.4f}, its 98.33% percentile CI (the same correction on the "
      f"bootstrap draws) is {_ci(PF['lo3']['computer_science'], PF['hi3']['computer_science'], 3)} and "
      f"{int(PF['n_far']['computer_science'])} of its {int(PF['n_fin']['computer_science'])} draws are on the far side "
      f"of zero; " + "; ".join(
          f"{LAB.get(f, f)}: {_f(pfb[f], 3)} {_ci(PF['lo'][f], PF['hi'][f], 3)}" for f in h55 if f != "computer_science")
      + ".")
    fg = H["shFG"]
    head4 = ("brand ties it" if fg["lo"] <= 0 <= fg["hi"] else
             "brand out-predicts it" if fg["mean"] < 0 else "it out-predicts brand")
    w(f"5. **Selectivity and institution-wide pay out-predict department prestige; {head4}.** In the per-field "
      f"horse race on ranks ({H['k']} fields), mean Shapley R² is {H['sh_F']['mean']:.3f} for department prestige, "
      f"{H['sh_S']['mean']:.3f} for selectivity (SAT, −admit rate), {H['sh_G']['mean']:.3f} for brand and "
      f"{H['sh_IE']['mean']:.3f} for institution-wide earnings. Department prestige is the largest block in "
      f"{H['lead_F']} fields (selectivity {H['lead_S']}, brand {H['lead_G']}, institution earnings {H['lead_IE']}). "
      f"Prestige minus brand Shapley: {_est(H['shFG'])}. Mirror test: brand net of the selectivity set and F is "
      f"{_est(S['G_broadF'])}, about the same as F net of the selectivity set and G "
      f"(difference {_est(D[('F_broadG', 'G_broadF')])}).")
    dp, fp_ = D[("F_broadP", "broad_comp")], FE["pair_FE3_FE2_c"]
    head5 = ("Program composition does not explain the remainder" if dp["hi"] > 0 and fp_["ci"][1] > 0
             else "Program composition explains part of the remainder")
    w(f"6. **{head5}.** The program's own Pell share (released for "
      f"{cov['cells_ppell']:,} of {cov['cells']:,} matched programs) moves the partial by "
      f"{_est(D[('F_broadP', 'broad_comp')])} and the within-institution β by "
      f"{_f(FE['pair_FE3_FE2_c']['diff'], 4)} {_ci(*FE['pair_FE3_FE2_c']['ci'], 4)}. Within Pell graduates and within "
      f"non-Pell graduates the fully controlled partial is {_est(S['F_all_pell'])} and {_est(S['F_all_nopell'])} "
      f"({S['F_all']['k']} fields), both including zero.")
    rawc = ["raw", "raw_h3y4", "raw_h3y1", "raw_h3y5"]
    brc = ["broad", "broad_h3y4", "broad_h3y1", "broad_h3y5"]
    fgc = ["F_broadG", "F_broadG_h1y1", "F_broadG_h5y5", "F_broadG_h3y4", "F_broadG_h3y1", "F_broadG_h3y5"]
    w(f"7. **What this says about A vs B.** Reading A can claim at most a small remainder. With 4-year earnings: a "
      f"mean partial correlation of {_f(min(S[c]['mean'] for c in full), 2)} to {_f(max(S[c]['mean'] for c in full), 2)} "
      f"on the rank scale (main spec {_f(main['mean'], 3)}, against {_f(S['raw']['mean'], 2)} raw) and about "
      f"{100 * fe2['beta']:.1f}% of 4-year pay per SD of department prestige within institutions. Pooled over the "
      f"three horizons: a partial correlation of {_f(PAV['mean'], 3)} and {100 * min(betas_pool):.1f}–"
      f"{100 * max(betas_pool):.1f}% of pay per SD, "
      + ("distinguishable from zero in every pooled specification" if n_pool_pos == len(pooled) else
         "not distinguishable from zero in any pooled specification" if n_pool_pos == 0 else
         f"distinguishable from zero in {n_pool_pos} of the {len(pooled)} pooled specifications (not in the "
         + ", ".join({'partial': 'partial correlation', 'FE0': 'spec (a)', 'FE2sg': 'field × {SAT, G} spec',
                      'FE2': 'unweighted main spec', 'FE2w': 'weighted main spec', 'FE2v': 'main spec with 4-yr-count weights'}[k]
                     for k, b in pooled if not b) + ")")
      + ". "
      + ((f"Within institutions the clearest single field is computer science ({pf_lab(cs)} pooled over horizons; "
          f"{_f(nlg['beta_f'][cs], 3)} and {_f(nla['beta_f'][cs], 3)} with cubic brand and cubic brand / selectivity "
          f"pricing, 4-year earnings). " if cs_all_variants else "")
         + (f"Several engineering fields ({names(eng_pos)}) and business fields ({names(bus_pos)}) are positive, "
            f"none passing a multiple-testing correction with percentile-bootstrap p, {names(eng_zero)} are near "
            f"zero, and the remaining {k_rest} fields average "
            f"{_f(GW[('rest', 'equal')]['mean'], 4)} (equal weights) / {_f(GW[('rest', 'precision')]['mean'], 4)} "
            f"(precision weights). The CS/engineering-minus-other contrast is {gwd('eng-other', 'equal')} with equal "
            f"weights and {gwd('eng-other', 'precision')} with precision weights, and random sets of "
            f"{labp['eng']['m']} fields reach it in {100 * labp['eng']['p_eq']:.1f}% of draws; the business fields "
            f"differ from the remaining {k_rest} fields by {gwd('bus-rest', 'equal')} (equal) and "
            f"{gwd('bus-rest', 'precision')} (precision weights). "
            if eng_pos else ""))
      + f"The two designs differ on whether the remainder varies between fields. In the partial-correlation design the "
      f"field-specific component shows little reproducible cross-field variation (split-half reliability "
      f"{_f(SH['rel_F_broadG'], 2)}, p = {_p(SH['rel_F_broadG_p'])}, against {_f(SH['rel_broad'], 2)} for the "
      f"selectivity-only partial), but that check can only rule out large differences (80% power only for "
      f"{p80_txt(PWF)}; the observed value is consistent with {rng_txt(PWF)}, "
      + ("including" if cons_(PWF, PWF['tau_reml']) else "but not") + f" the REML estimate τ = {PWF['tau_reml']:.3f}), and "
      f"what variation the component has lines up with baseline coupling and with selectivity pricing (cross-fits "
      f"{_f(SH['cross_F_broadG_raw'], 2)}, p = {_p(SH['cross_F_broadG_raw_p'])}, and "
      f"{_f(SH['cross_F_broadG_sat_earn'], 2)}, p = {_p(SH['cross_F_broadG_sat_earn_p'])}). Little of F is left once "
      f"G and selectivity are removed (mean R² of F on those controls {REL['F_broadG']['r2x_mean']:.2f}), which is "
      f"one reason this design has little power. In the within-institution design with linear field-specific pricing "
      f"the department-specific slopes "
      + (f"differ across fields ({wq_short}; leverage-corrected residuals p {lev(wqa)} over all fields and {lev(wqb)} "
         f"over the {wqb['k']} with ≥ {PF_SPLIT_NMIN} programs) and their ordering {repro_vb} across independent halves "
         f"of institutions ({_f(PFS['main']['rel'], 2)}; permutation p = {_p(PFS['main']['p'])}, simulated-null p = "
         f"{_p(PWB['p_sim'])}; without computer science and engineering {spl(PFS['main_noeng'])})"
         if het_lin else
         f"are not detectably heterogeneous ({wq_short})")
      + (f". That heterogeneity does not survive nonlinear pricing of brand on the well-estimated fields: with field × "
         f"{{zG², zG³}}, p = {prange(ps_cg_big)} on the {nlg['het'][('big', 'iqr')]['k']} fields with ≥ "
         f"{PF_SPLIT_NMIN} programs and split-half reliability {spl(nlg['split']['main'])}. Convex pricing of brand is "
         f"itself a reading-B mechanism, so part of what looked like field-specific department premia may be fields "
         f"paying for the top of the brand distribution" if het_lin and not het_nl_robust else "")
      + f". Their correlation across earnings horizons is not counted, because the horizons reuse the same programs "
      f"and institutions ({xh_short}; sampling error alone reaches the observed r in {sh_all} of centred draws)"
      + ". So the evidence shows a small within-institution premium that "
      + ("is clear in computer science under all three pricing variants, and cross-field differences beyond that which are "
         "significant with linear pricing but fragile to how brand is priced. " if cs_all_variants and het_lin
         and not het_nl_robust else
         "differs between fields. " if het_lin else "cannot be ruled out. ")
      + f"What is robust is the size of the drop: at every horizon, "
      f"coupling falls from {_f(min(S[c]['mean'] for c in rawc), 2)}–{_f(max(S[c]['mean'] for c in rawc), 2)} raw to "
      f"{_f(min(S[c]['mean'] for c in brc), 2)}–{_f(max(S[c]['mean'] for c in brc), 2)} under the selectivity set and "
      f"to {_f(max(S[c]['mean'] for c in fgc + ['F_broadG_h3avg']), 2)} or less once brand is added. Most of the raw "
      f"coupling therefore goes with institution-level status, which is what B predicts for the bulk of coupling "
      f"(brand G is built from the same hiring data and partly contains the field's own hierarchy, so part of that "
      f"drop may be field-specific; Caveats). For the remainder the evidence does not favour B over A: it could be "
      f"department standing (A) or program-level student selection that institution-level measures miss (B), and "
      f"public US data cannot tell these apart. "
      + ("The computer science premium (and the positive, imprecise engineering and business remainders) fits "
         "either reading: under A, employers in these fields would read department standing; under B, selection into "
         "these majors within an institution (separately admitted engineering or business programs, capped computer "
         "science majors, sorting of stronger students into them) would track department standing. "
         if cs_all_variants or eng_pos else "")
      + f"For the 4-year within-institution β, a coefficient-stability bound (Oster 2019) "
      f"puts the selection on unobservables needed to erase it at δ* = {min(dstar):.2f} to {max(dstar):.2f} times "
      f"the selection on the observed controls, depending on the reference model and R_max; it is below the usual "
      f"benchmark of 1 in {sum(d < 1 for d in dstar)} of {len(dstar)} settings. Section (v) names the data that "
      f"could separate A from B.")
    w("")
    w("Caveat that matters most: the residual prestige measure is noisy. The published rank's reliability is uncertain "
      "(scripts/56), and after partialling out brand and selectivity only a small share of its variance is left, so "
      "measurement error attenuates the remainder. Under the public-edge lower-bound reliability the disattenuated mean "
      f"of the main spec is {_est(REL['F_broadG']['LB'])} ({REL['F_broadG']['LB']['k_id']} of {REL['F_broadG']['k']} "
      f"fields identifiable), under the extrapolated reliability {_est(REL['F_broadG']['EXT'])}. The within-institution "
      f"β would be {_f(fe2['beta'] / fe2['lam_LB'], 4)} (lower bound) or {_f(fe2['beta'] / fe2['lam_EXT'], 4)} "
      f"(extrapolated). These corrections apply to the 4-year estimates on the full SAT sample, where the "
      f"lower-bound correction would make the remainder clearly nonzero, though still well below the raw coupling "
      f"({_f(S['raw']['mean'], 3)}). By horizon, on identical programs, the lower-bound-corrected mean is: 5-year "
      f"sample, 4-year {_est(REL['F_broadG_h5y4']['LB'])} vs 5-year {_est(REL['F_broadG_h5y5']['LB'])}; 1-year "
      f"sample, 4-year {_est(REL['F_broadG_h1y4']['LB'])} vs 1-year {_est(REL['F_broadG_h1y1']['LB'])}; common "
      f"programs, 4 / 1 / 5-year {_est(REL['F_broadG_h3y4']['LB'])} / {_est(REL['F_broadG_h3y1']['LB'])} / "
      f"{_est(REL['F_broadG_h3y5']['LB'])}. " + rel_hz_sentence(REL) + " The correction rescales each field's "
      f"estimate, so it cannot turn estimates near zero into a clear signal.")
    w("")

    # ---------------- key numbers ----------------
    w("## Key numbers")
    w("")
    w("CI = 95% joint institution-bootstrap CI of the mean across fields (partial correlations) or institution "
      "cluster-bootstrap percentile CI (within-institution β, log points per within-field SD of department prestige).")
    w("")
    w("| quantity | sample / specification | k fields | estimate [95% CI] |")
    w("|---|---|---|---|")
    rows = [("raw", "full sample"), ("broad", "SAT sample; scripts/55 broad spec"),
            ("F_selG", "SAT sample; SAT, ADM, G only"), ("F_broadG", "SAT sample; **main field-specific spec**"),
            ("F_broadGie", "SAT sample"), ("F_admG", "ADM sample (keeps test-blind institutions)"),
            ("F_all", "composition sample (program Pell share released)"),
            ("F_all_pell", "composition sample"), ("F_all_nopell", "composition sample"),
            ("G_broadF", "SAT sample (brand mirror)")]
    for c, smp in rows:
        w(f"| mean {_t(SPEC_DESC[c])} | {smp} | {S[c]['k']} | {_est(S[c])} |")
    w(f"| pooled residual slope, main spec | {main['pool_cells']:,} programs, {main['pool_clusters']} institutions | "
      f"{main['k']} | {_f(main['pool_b'])} {_ci(main['pool_lo'], main['pool_hi'])} (CR1) |")
    w(f"| REML random-effects mean, main spec; τ [95% upper] | SAT sample | {main['k']} | {_f(main['mu_re'])} "
      f"{_ci(main['lo_re'], main['hi_re'])}; τ {main['tau']:.3f} [{main['tau_hi']:.3f}] |")
    w(f"| fields > 0 / < 0 at 1.96 SE; BH-FDR, main spec | SAT sample | {main['k']} | {main['n_pos']} / "
      f"{main['n_neg']}; {main['n_bh_pos']} / {main['n_bh_neg']} (chance ≈ {main['exp_fp']:.1f} per tail) |")
    for n in ["FE0", "FE0_s", "FE2sg", "FE2", "FE2ie", "FE2sc", "FE2cg", "FE2ca", "FE3"]:
        r = FE[n]
        w(f"| within-institution β: {FE_DESC[n]} | {r['n_cells']:,} programs, {r['n_inst']} institutions | "
          f"{r['n_fields']} | {_fe_ci(r)} |")
    w(f"| same cells as the main spec, no institution FE (field FE + β z only) | {O['n']:,} programs | {fe2['n_fields']} | "
      f"{_f(nofe, 4)} (point) |")
    w(f"| horse race, mean Shapley R²: F / selectivity / G / inst. earnings | SAT sample, ranks | {H['k']} | "
      f"{H['sh_F']['mean']:.3f} / {H['sh_S']['mean']:.3f} / {H['sh_G']['mean']:.3f} / {H['sh_IE']['mean']:.3f} |")
    w(f"| split-half reliability of the cross-field ordering: selectivity-only partial vs field-specific component | "
      f"fields with n_SAT ≥ {SPLIT_NMIN} | {SH['k']} | {_f(SH['rel_broad'], 2)} (p {_p(SH['rel_broad_p'])}) vs "
      f"{_f(SH['rel_F_broadG'], 2)} (p {_p(SH['rel_F_broadG_p'])}) |")
    w(f"| same check, power: expected reliability of the field-specific component at τ = 0 / {PWF['tau_reml']:.3f} "
      f"(REML) / {PWF['tau_reml_hi']:.3f} (REML upper bound); τ with 80% power | simulation with these fields' "
      f"bootstrap SEs | {PWF['k']} | {_f(PWF['exp_zero'], 2)} / {_f(PWF['exp_reml'], 2)} / {_f(PWF['exp_reml_hi'], 2)}; "
      f"{p80_txt(PWF)} |")
    w(f"| between-field heterogeneity: partial correlation (main spec) vs within-institution β_f (main spec, "
      f"IQR-based bootstrap SEs) | SAT sample / main-spec cells | {main['k']} / {kpf} | τ {main['tau']:.3f}, Q p "
      f"{_p(main['Qp'])} vs τ {biq['tau']:.3f}, Q p {_p(biq['Qp'])} (bootstrap-SD SEs: Q p {_p(bsd['Qp'])}) |")
    w(f"| within-institution β_f heterogeneity, wild-cluster Q (one common β imposed; keeps shared institutions), "
      f"restricted residuals: weights from IQR / SD bootstrap SEs / per-replicate CR1 | main-spec cells: all fields; "
      f"fields with ≥ {PF_SPLIT_NMIN} programs | {wq_all['k']} / {k_big} | p "
      + " / ".join(pw(WQ[("all", wt)]["WCR"]["p"]) for wt in ("iqr", "sd", "cr1")) + "; "
      + " / ".join(pw(WQ[("big", wt)]["WCR"]["p"]) for wt in ("iqr", "sd", "cr1")) + " |")
    w(f"| same, leverage-corrected restricted residuals (WCR3), IQR weights: all fields / ≥ {PF_NMIN_MID} programs / "
      f"≥ {PF_SPLIT_NMIN} programs / ≥ {PF_NMIN_MID} programs without computer science and engineering | main-spec "
      f"cells | {wqa['k']} / {wqm['k']} / {wqb['k']} / {wqn['k']} | p {lev(wqa)} / {lev(wqm)} / {lev(wqb)} / "
      f"{lev(wqn)} |")
    for n in NL_SPECS:
        q = NL[n]
        w(f"| β_f heterogeneity on the fields with ≥ {PF_SPLIT_NMIN} programs, {FE_DESC[n]}: χ² Q p (IQR / SD SEs); "
          f"wild p (restricted / leverage-corrected); split-half reliability (permutation p) | main-spec cells | "
          f"{q['het'][('big', 'iqr')]['k']} | {_p(q['het'][('big', 'iqr')]['p'])} / {_p(q['het'][('big', 'sd')]['p'])}; "
          f"{pw(q['het'][('big', 'wild_iqr')]['WCR']['p'])} / {lev(q['het'][('big', 'wild_iqr')])}; "
          f"{spl(q['split']['main'])} |")
    w(f"| split-half reliability of the within-institution β_f (main spec): all / without computer science / "
      f"without computer science and engineering (permutation p) | fields with ≥ {PF_SPLIT_NMIN} programs | "
      f"{PFS['main']['k']} / {PFS['main_nocs']['k']} / {PFS['main_noeng']['k']} | {spl(PFS['main'])} / "
      f"{spl(PFS['main_nocs'])} / {spl(PFS['main_noeng'])}; all fields against the simulated null p = "
      f"{_p(PWB['p_sim'])} |")
    w(f"| within-institution β_f CIs above / below zero (main spec, 4-yr) | main-spec cells | {kpf} | "
      f"{len(ci_pos)} / {len(ci_neg)} (chance ≈ {0.025 * kpf:.1f} per tail; P(≥ {len(ci_pos)} above) = "
      f"{_p(pbin_pf)} if fields were independent; at 1.96 IQR-based SE {WN4['n_pos']} above zero against "
      f"{WN4['WCR']['mean_pos']:.1f} expected under a wild-cluster null with β_f = 0, P(≥ {WN4['n_pos']}) "
      f"{pe(WN4['WCR']['p_pos'])}; leverage-corrected null {WN4['WCR3']['mean_pos']:.1f}, P "
      f"{pe(WN4['WCR3']['p_pos'])}) |")
    w(f"| Oster δ* for the within-institution β (R_max = {O['rmax_13']:.2f}) | main-spec cells; short = no institution FE / "
      f"institution FE only | {fe2['n_fields']} | {O['delta_s0_13']:.2f} / {O['delta_s1_13']:.2f} (R_max = "
      f"{O['rmax_half']:.3f}: {O['delta_s0_half']:.2f} / {O['delta_s1_half']:.2f}) |")
    w(f"| **horizon replication** — mean ρ(F, Y \\| selectivity set + G), 4-yr vs 5-yr earnings | 5-yr sample (SAT sample, 5-yr median released) | "
      f"{P5['k']} | 4-yr {_est(P5b)}; 5-yr {_est(P5)} |")
    w(f"| same, 4-yr vs 1-yr earnings | 1-yr sample | {P1['k']} | 4-yr {_est(P1b)}; 1-yr {_est(P1)} (percentile CI "
      f"{ci_pct(P1)}) |")
    w(f"| same, 4 / 1 / 5-yr earnings | common programs (all three released) | {PC['y4']['k']} | "
      f"{_est(PC['y4'])} / {_est(PC['y1'])} / {_est(PC['y5'])} |")
    w(f"| within-institution β (main), 4-yr vs 5-yr earnings | 5-yr sample: {F5['n_cells']:,} programs, {F5['n_inst']} "
      f"institutions | {F5['n_fields']} | 4-yr {_fe_ci(F5b)}; 5-yr {_fe_ci(F5)} |")
    w(f"| within-institution β (main), 4-yr vs 1-yr earnings | 1-yr sample: {F1['n_cells']:,} programs, {F1['n_inst']} "
      f"institutions | {F1['n_fields']} | 4-yr {_fe_ci(F1b)}; 1-yr {_fe_ci(F1)} |")
    w(f"| within-institution β (main), 4 / 1 / 5-yr earnings | common programs: {FC['y4']['n_cells']:,} programs, "
      f"{FC['y4']['n_inst']} institutions | {FC['y4']['n_fields']} | {_fe_ci(FC['y4'])} / {_fe_ci(FC['y1'])} / "
      f"{_fe_ci(FC['y5'])} |")
    w(f"| fields with within-institution β_f CI above / below zero, 4 / 1 / 5-yr earnings | common programs | "
      f"{hzf['y4']['k']} | {len(hzf['y4']['ci_pos'])} / {len(hzf['y4']['ci_neg'])}; {len(hzf['y1']['ci_pos'])} / "
      f"{len(hzf['y1']['ci_neg'])}; {len(hzf['y5']['ci_pos'])} / {len(hzf['y5']['ci_neg'])} (chance ≈ "
      f"{0.025 * hzf['y4']['k']:.1f} per tail) |")
    w(f"| correlation of within-institution β_f (main) across horizons, 4 v 1 / 4 v 5 / 1 v 5 yr, against its null "
      f"with one common β (same programs; wild cluster bootstrap, shared institution weights) | common programs | "
      f"{WD['hz_k']} | r {trip(lambda q: _f(q['r'], 2))}; null mean {trip(lambda q: _f(q['WCR']['mean'], 2))}; "
      f"p {trip(lambda q: pw(q['WCR']['p']))} (leverage-corrected {trip(lambda q: pw(q['WCR3']['p']))}; share of "
      f"centred pairs-bootstrap draws at or above r: {sh_all}) |")
    w(f"| **pooled over horizons** — mean ρ(F, Y \\| selectivity set + G), average of 4-, 1- and 5-yr earnings | "
      f"common programs | {PAV['k']} | {_est(PAV)} (percentile CI {ci_pct(PAV)}) |")
    for fam in ("FE0", "FE2sg", "FE2", "FE2w", "FE2v"):
        r = AV[fam]
        w(f"| within-institution β: {FE_DESC[f'{fam}_h3avg']} | {r['n_cells']:,} programs, {r['n_inst']} institutions "
          f"| {r['n_fields']} | {_fe_ci(r)} |")
    w(f"| within-institution β_f (main), 4/1/5-yr average: fields with CI above / below zero; BH-FDR with raw "
      f"percentile p / normal p | common programs | {len(pfb)} | {len(pf_pos)} / {len(pf_neg)} (chance ≈ "
      f"{0.025 * len(pfb):.2f} per tail); BH {len(bh_p)} / {len(bh_n)} |")
    w(f"| same, computer science: pooled β_f [95% CI]; 98.33% CI (3 fields named in advance); draws ≤ 0 | common "
      f"programs | 1 | {_f(pfb[cs], 4)} {_ci(PF['lo'][cs], PF['hi'][cs], 4)}; {_ci(PF['lo3'][cs], PF['hi3'][cs], 4)}; "
      f"{int(PF['n_far'][cs])} of {int(PF['n_fin'][cs])} |")
    w(f"| same, mean β_f of the computer science and engineering fields vs the other fields (grouping chosen after "
      f"seeing the results), equal weights | common programs | {gr['eng']['k']} / {gr['other']['k']} | "
      f"{_f(gr['eng']['mean'], 4)} {_ci(gr['eng']['lo'], gr['eng']['hi'], 4)} vs {_f(gr['other']['mean'], 4)} "
      f"{_ci(gr['other']['lo'], gr['other']['hi'], 4)} |")
    w(f"| same, precision-weighted (1 / bootstrap variance) / weighted by program count | common programs | "
      f"{gr['eng']['k']} / {gr['other']['k']} | {gwm('eng', 'precision')} vs {gwm('other', 'precision')}; "
      f"{gwm('eng', 'programs')} vs {gwm('other', 'programs')} |")
    w(f"| same, business fields (accounting, economics, finance, management, marketing; post hoc) vs the remaining "
      f"fields: equal / precision / program-count weights | common programs | {len(BUS_)} / {k_rest} | "
      f"{gwm('bus', 'equal')} vs {gwm('rest', 'equal')}; {_f(GW[('bus', 'precision')]['mean'], 4)} vs "
      f"{_f(GW[('rest', 'precision')]['mean'], 4)}; {_f(GW[('bus', 'programs')]['mean'], 4)} vs "
      f"{_f(GW[('rest', 'programs')]['mean'], 4)} |")
    w(f"| random sets of {labp["eng"]["m"]} ({labp["bus"]["m"]}) field labels reaching the observed "
      f"CS/engineering (business) minus other contrast, equal / precision weights | common programs, {N_PERM} sets | "
      f"{len(pfb)} | {100 * labp['eng']['p_eq']:.1f}% / {100 * labp['eng']['p_pr']:.1f}% "
      f"({100 * labp['bus']['p_eq']:.1f}% / {100 * labp['bus']['p_pr']:.1f}%) |")
    w(f"| within-institution β (main), 4/1/5-yr average, without the {len(LOO['dropped'])} CS/engineering fields: "
      f"unweighted / cohort-size weighted | {LOO['FE2']['n_cells']:,} programs, {LOO['FE2']['n_inst']} institutions | "
      f"{LOO['FE2']['n_fields']} | {_f(LOO['FE2']['beta'], 4)} {_ci(*LOO['FE2']['ci'], 4)} / "
      f"{_f(LOO['FE2w']['beta'], 4)} {_ci(*LOO['FE2w']['ci'], 4)} |")
    w("")

    # ---------------- (i) composition ----------------
    w("## (i) Program-level composition (Scorecard FoS)")
    w("")
    w(f"Program Pell share PPELL = Σ EARN_COUNT_PELL_WNE_4YR / Σ (EARN_COUNT_PELL_WNE_4YR + EARN_COUNT_NOPELL_WNE_4YR) "
      f"over the field's CIP-4 rows at the institution (rows where both counts are released). It is released for "
      f"{cov['cells_ppell']:,} of {cov['cells']:,} matched (field, institution) cells; the Pell and non-Pell medians "
      f"are released for {cov['cells_med']:,}, and {cov['cells_ppell_nomed']} cells have the share without both "
      f"medians, {cov['cells_med_noppell']} both medians without the share. Release depends strongly on program size (share released by cohort-size "
      f"quartile, smallest → largest: " + ", ".join(f"{v:.0%}" for v in cov['by_size'].values()) + ") and somewhat on "
      f"brand (top → bottom academia-wide quartile: " + ", ".join(f"{v:.0%}" for v in cov['by_gq'].values()) + "). "
      f"Mean PPELL across programs is {np.nanmean(T['ppell_mean']):.2f} (a share of federally aided completers, "
      f"because Scorecard earnings cover Title IV recipients only). Its within-field Spearman correlation with the "
      f"institution's PCTPELL averages {np.nanmean(T['rho_ppell_pctpell']):+.2f}, with department prestige "
      f"{np.nanmean(T['rho_F_ppell']):+.2f} and with program earnings {np.nanmean(T['rho_ppell_Y']):+.2f} "
      f"({int(T['rho_ppell_Y'].notna().sum())} fields).")
    w("")
    w("| specification | k | mean partial ρ [95% CI] | τ | Q p | fields >0 / <0 (1.96 SE) |")
    w("|---|---|---|---|---|---|")
    for c in ["raw_comp", "broad_comp", "F_broadP", "F_broadG_comp", "F_all", "raw_pell", "raw_nopell",
              "F_broadP_pell", "F_broadP_nopell", "F_all_pell", "F_all_nopell", "F_allM_base", "F_allM",
              "broad_compd", "F_broadPd", "F_allD", "F_admGP"]:
        q = S[c]
        w(f"| {_t(SPEC_DESC[c])} | {q['k']} | {_est(q)} | {q['tau']:.3f} | {_p(q['Qp'])} | {q['n_pos']} / {q['n_neg']} |")
    w("")
    w("Paired differences (same fields, same bootstrap draws):")
    w("")
    w("| difference | k | mean difference [95% CI] |")
    w("|---|---|---|")
    for a, b in [("F_broadP", "broad_comp"), ("F_all", "F_broadG_comp"), ("F_all_pell", "F_all"),
                 ("F_all_nopell", "F_all"), ("F_broadP_nopell", "F_broadP"), ("F_allM", "F_allM_base"),
                 ("F_broadPd", "broad_compd")]:
        q = D[(a, b)]
        w(f"| ({_t(SHORT[a])}) − ({_t(SHORT[b])}) | {q['k']} | {_f(q['diff'])} {_ci(q['lo'], q['hi'])} |")
    w("")
    bal = FE["balance"]
    w(f"Within institutions, departments with higher prestige graduate somewhat fewer Pell students: z(PPELL) on z(F) "
      f"with institution and field FE gives {_f(bal['b'])} (SE {bal['se']:.3f}; {bal['n']:,} programs, "
      f"{bal['n_inst']} institutions), {_f(bal['b_noFE'])} (SE {bal['se_noFE']:.3f}) without institution FE. So "
      f"composition does differ across departments, but controlling for it does not reduce the prestige association "
      f"(tables above and below). Debt-cohort Pell share (DEBT_PELL_STGP_EVAL_N, borrowers; released for "
      f"{cov['cells_ppell_d']:,} cells) is a wider-coverage check: it moves the partial by "
      f"{_est(D[('F_broadPd', 'broad_compd')])} and the within-institution β by "
      f"{_f(FE['pair_FE3d_FE2_d']['diff'], 4)} {_ci(*FE['pair_FE3d_FE2_d']['ci'], 4)}.")
    w("")
    w("| within-institution spec (composition samples) | programs | institutions | fields | β [95% bootstrap CI] |")
    w("|---|---|---|---|---|")
    for n in ["FE0_c", "FE2_c", "FE3", "FE3_p", "FE3_np", "FE2_m", "FE3m", "FE2_d", "FE3d"]:
        r = FE[n]
        w(f"| {FE_DESC[n]} | {r['n_cells']:,} | {r['n_inst']} | {r['n_fields']} | {_fe_ci(r)} |")
    for a, b in [("FE3", "FE2_c"), ("FE3", "FE3_np"), ("FE3", "FE3_p"), ("FE3m", "FE2_m"), ("FE3d", "FE2_d")]:
        q = FE[f"pair_{a}_{b}"]
        w(f"| paired difference: ({FE_DESC[a]}) − ({FE_DESC[b]}) | | | | {_f(q['diff'], 4)} {_ci(*q['ci'], 4)} |")
    w("")
    w(f"The non-Pell-outcome β ({_f(FE['FE3_np']['beta'], 4)}) is below the all-student β ({_f(FE['FE3']['beta'], 4)}); "
      f"the paired difference {_f(FE['pair_FE3_FE3_np']['diff'], 4)} {_ci(*FE['pair_FE3_FE3_np']['ci'], 4)} is one "
      f"of several comparisons and is read as a hint, not a finding: part of the all-student within-institution "
      f"premium may be composition within the aided population that a linear Pell-share control does not remove.")
    w("")

    # ---------------- (ii) field-specific component ----------------
    w("## (ii) Field-specific component of prestige")
    w("")
    w("Within each field, rank(F) and rank(Y) are each residualized on the controls (ranks of the continuous "
      "controls, a control dummy); the partial correlation is the correlation of the two residuals. Pooled three "
      "ways: unweighted mean with joint-bootstrap CI, REML random-effects mean (fields treated as independent; they "
      "share institutions, so the joint-bootstrap CI is the one to read), and the cell-level slope of standardized "
      "residuals with institution-clustered SE.")
    w("")
    w("| specification | k | mean [95% CI] | REML μ [95% CI] | τ [95% upper] | Q p | pooled slope (SE) | >0 / <0 at 1.96 SE | BH >0 / <0 | TOST ±0.10 | reliable fields: k, mean |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in ["raw", "F_G", "G_F", "sel", "F_selG", "broad", "F_broadG", "F_broadGie", "G_broad", "G_broadF", "adm",
              "F_admG", "F_all", "G_all"]:
        q = S[c]
        ps = f"{_f(q['pool_b'])} ({q['pool_se']:.3f})" if "pool_b" in q else "—"
        w(f"| {_t(SPEC_DESC[c])} | {q['k']} | {_est(q)} | {_f(q['mu_re'])} {_ci(q['lo_re'], q['hi_re'])} | "
          f"{q['tau']:.3f} [{q['tau_hi']:.3f}] | {_p(q['Qp'])} | {ps} | {q['n_pos']} / {q['n_neg']} | "
          f"{q['n_bh_pos']} / {q['n_bh_neg']} | {'yes' if q['tost'] else 'no'} (±{q['eq_bound']:.3f}) | "
          f"{q['k_rel']}, {_f(q['mean_rel'])} |")
    w("")
    w("TOST column: 'yes' if the 90% CI of the mean lies inside ±0.10; in brackets the smallest symmetric bound that "
      "would pass. Minimum detectable mean (80% power, 5% two-sided) for the main spec: "
      f"{main['mde80']:.3f}.")
    w("")
    w("Paired differences:")
    w("")
    w("| difference | k | mean difference [95% CI] |")
    w("|---|---|---|")
    for a, b in [("F_selG", "sel"), ("F_broadG", "broad"), ("F_broadGie", "F_broadG"), ("F_admG", "adm"),
                 ("F_G", "G_F"), ("F_broadG", "G_broadF"), ("F_all", "G_all")]:
        q = D[(a, b)]
        w(f"| ({_t(SHORT[a])}) − ({_t(SHORT[b])}) | {q['k']} | {_f(q['diff'])} {_ci(q['lo'], q['hi'])} |")
    w("")
    w("Subsets of the main spec:")
    w("")
    for tag, dd, lab in (("reliable", ctx["S_rel"], "reliable fields (gap map)"),
                         ("unlicensed", ctx["S_nolic"], "fields without occupational licensing")):
        for c in dd:
            q = dd[c]
            w(f"- {lab}, {SHORT[c]}: k = {q['k']}, mean {_est(q)}, τ {q['tau']:.3f} (Q p {_p(q['Qp'])}), "
              f"{q['n_pos']} / {q['n_neg']} fields at 1.96 SE, BH {q['n_bh_pos']}.")
    w("")
    w("**Reliability bracket.** F has reliability rel_F < 1 (scripts/56: LB = public-edge split-half, a lower bound for "
      "the published rank; EXT = extrapolated to the full hiring census). Partialling out controls that explain a "
      "share R² of F's variance leaves a residual with reliability 1 − (1 − rel_F)/(1 − R²) (classical error, "
      f"independent of the controls); it is treated as not identified below {REL_MIN}. The disattenuated partial is "
      "the observed one divided by the square root of that reliability (prestige side only).")
    w("")
    w("| specification | k | mean R²(F on controls) | LB: fields identified, median residual reliability, observed mean on them, disattenuated mean [95% CI] | EXT: same |")
    w("|---|---|---|---|---|")
    for c in ["F_G", "broad", "F_broadG", "F_broadGie", "F_admG", "F_all"]:
        r = REL[c]
        cells_ = []
        for tag in ("LB", "EXT"):
            q = r[tag]
            cells_.append(f"{q['k_id']} of {r['k']}, {q['relperp_med']:.2f}, {_f(q['mean_obs'])}, {_est(q)}")
        w(f"| {_t(SPEC_DESC[c])} | {r['k']} | {r['r2x_mean']:.2f} | {cells_[0]} | {cells_[1]} |")
    w("")
    w(f"**Is there a field-specific pattern?** Split-half check on the {SH['k']} fields with n_SAT ≥ {SPLIT_NMIN}: the "
      f"institutions are split at random into halves ({SPLIT_K} splits); the per-field statistic on one half is "
      f"correlated across fields with the other half (reliability, Pearson) and with the baseline coupling or "
      f"selectivity pricing ρ(SAT, Y) on the other half (cross-fit, Spearman; both directions averaged). p = field-label "
      f"permutation.")
    w("")
    w("| per-field statistic | split-half reliability (p) | cross-fit with baseline coupling (p) | cross-fit with selectivity pricing (p) | true-score r with baseline / pricing (Pearson) |")
    w("|---|---|---|---|---|")
    for n, lab in (("broad", "ρ(F, Y \\| selectivity set) [scripts/55 broad]"), ("F_broadG", "ρ(F, Y \\| selectivity set + G)"),
                   ("G_broadF", "ρ(G, Y \\| selectivity set + F)")):
        ts = ((f"{_f(SH[f'ts_{n}_raw'], 2)} / {_f(SH[f'ts_{n}_sat_earn'], 2)}"
               + ("" if SH[f"rel_{n}_p"] < 0.05 else " (very imprecise: reliability not significant)"))
              if SH[f"rel_{n}"] > 0 and np.isfinite(SH[f"ts_{n}_raw"]) else "— (reliability ≤ 0)")
        w(f"| {lab} | {_f(SH[f'rel_{n}'], 2)} ({_p(SH[f'rel_{n}_p'])}) | {_f(SH[f'cross_{n}_raw'], 2)} "
          f"({_p(SH[f'cross_{n}_raw_p'])}) | {_f(SH[f'cross_{n}_sat_earn'], 2)} ({_p(SH[f'cross_{n}_sat_earn_p'])}) | "
          f"{ts} |")
    w(f"| baseline coupling ρ(F, Y) | {_f(SH['rel_raw'], 2)} | | | |")
    w(f"| selectivity pricing ρ(SAT, Y) | {_f(SH['rel_sat_earn'], 2)} | | | |")
    w("")
    fg_rel_sig = SH["rel_F_broadG_p"] < 0.05
    cf_sig = SH["cross_F_broadG_raw_p"] < 0.05 and SH["cross_F_broadG_sat_earn_p"] < 0.05
    w(("The field-specific component's split-half reliability is " + _f(SH['rel_F_broadG'], 2) +
       f" (p = {_p(SH['rel_F_broadG_p'])}): its cross-field ordering " + ("reproduces" if fg_rel_sig else
                                                                         "does not reproduce clearly") +
       f" across halves. Its cross-fits with baseline coupling ({_f(SH['cross_F_broadG_raw'], 2)}, p = "
       f"{_p(SH['cross_F_broadG_raw_p'])}) and with selectivity pricing ({_f(SH['cross_F_broadG_sat_earn'], 2)}, p = "
       f"{_p(SH['cross_F_broadG_sat_earn_p'])}) are " + ("both above the permutation null" if cf_sig else "small") +
       f". With a reliability this low a cross-half correlation cannot exceed about √(reliability × partner's "
       f"reliability) ≈ {np.sqrt(max(SH['rel_F_broadG'], 0) * SH['rel_raw']):.2f} (baseline) and "
       f"{np.sqrt(max(SH['rel_F_broadG'], 0) * SH['rel_sat_earn']):.2f} (pricing), so the cross-fits are near that "
       f"ceiling; disattenuated, they correspond to true-score correlations of {_f(SH['ts_F_broadG_raw'], 2)} "
       f"(baseline) and {_f(SH['ts_F_broadG_sat_earn'], 2)} (pricing). Those true-score values divide by "
       f"√({_f(SH['rel_F_broadG'], 2)} × the partner's reliability) and are very imprecise. What they say is that "
       f"the reproducible variation the component has lines up with both baseline coupling and selectivity pricing; "
       f"since those two are nearly collinear across fields (true-score r +0.94 in scripts/55), this does not "
       f"discriminate between A and B. The selectivity-only partial's ordering reproduces, as in scripts/55, with "
       f"true-score correlation {_f(SH['ts_broad_raw'], 2)} with baseline coupling and "
       f"{_f(SH['ts_broad_sat_earn'], 2)} with selectivity pricing."))
    w("")
    w(f"**How much can the split-half check detect?** Simulation with the observed bootstrap SEs of the same fields "
      f"({POW_NSIM} simulated data sets per τ, {POW_NSPLIT} half-splits each): true field values θ_f ~ N(0, τ²), each "
      f"half estimates θ_f with sampling variance 2·SE_f² (half the institutions), and the statistic is the mean "
      f"over splits of the cross-field correlation between halves, as in the data. Fields are treated as "
      f"independent. Power is against the simulated τ = 0 95th percentile (the permutation null's 95th percentile "
      f"from the data is shown for comparison). 'τ consistent' = the τ grid values whose simulated 2.5–97.5% band "
      f"contains the observed reliability.")
    w("")
    w("| per-field statistic | fields | RMS bootstrap SE | observed reliability (permutation p) | p against simulated τ = 0 | Q p, same fields | null 95th pct: permutation / simulated | expected reliability at τ = 0 / REML τ / REML upper bound | τ with ≥ 80% power | τ consistent with observed |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for key, lab, perm95, unit in (("F_broadG", "ρ(F, Y \\| selectivity set + G)", SH["rel_F_broadG_null95"], ""),
                                   ("broad", "ρ(F, Y \\| selectivity set) [scripts/55 broad]", SH["rel_broad_null95"], ""),
                                   ("beta_f", "within-institution β_f, main spec (4-yr; bootstrap-SD SEs)",
                                    PFS["main"]["null95"], " log points")):
        P = POW[key]
        if key == "beta_f":
            ex = (f"{_f(P['exp_zero'], 2)} / {_f(P['exp_reml_sd'], 2)} (τ = {P['tau_reml_sd']:.3f}, SD SEs) / "
                  f"{_f(P['exp_reml_iqr'], 2)} (τ = {P['tau_reml_iqr']:.3f}, IQR SEs)")
            obs_p = PFS["main"]["p"]
        else:
            ex = (f"{_f(P['exp_zero'], 2)} / {_f(P['exp_reml'], 2)} (τ = {P['tau_reml']:.3f}) / "
                  f"{_f(P['exp_reml_hi'], 2)} (τ = {P['tau_reml_hi']:.3f})")
            obs_p = SH[f"rel_{key}_p"]
        w(f"| {lab} | {P['k']} | {P['se_rms']:.3f} | {_f(P['obs'], 2)} ({_p(obs_p)}) | {_p(P['p_sim'])} | "
          f"{_p(P['Q_p'])} | {_f(perm95, 2)} / {_f(P['crit'], 2)} | {ex} | {p80_txt(P)}{unit} | {rng_txt(P)}{unit} |")
    w("")
    w(f"So the partial-correlation check is weak: a reliability of {_f(PWF['obs'], 2)} is consistent with "
      f"{rng_txt(PWF)} (expected {_f(PWF['exp_reml'], 2)} at the REML estimate τ = {PWF['tau_reml']:.3f}), the check "
      f"has 80% power only for {p80_txt(PWF)}, and it cannot tell τ = 0 from τ ≈ {PWF['tau_reml']:.3f}. Its null result is "
      f"not evidence that the department-specific part has no cross-field variation. The fall from "
      f"{_f(SH['rel_broad'], 2)} (selectivity-only partial; expected {_f(PWS['exp_reml'], 2)} at its REML τ = "
      f"{PWS['tau_reml']:.3f}) to {_f(SH['rel_F_broadG'], 2)} "
      + ("is what the smaller between-field SD after partialling out G would produce: each observed reliability lies "
         "inside the simulated band at its own REML τ " if cons_(PWS, PWS['tau_reml']) and cons_(PWF, PWF['tau_reml'])
         else "is not fully accounted for by the REML τ values: at least one observed reliability lies outside the "
              "simulated band at its own REML τ ")
      + f"(τ {PWS['tau_reml']:.3f} → {PWF['tau_reml']:.3f}; RMS SE {PWS['se_rms']:.3f} → {PWF['se_rms']:.3f}). The "
      f"within-institution per-field slopes, estimated from more of the data, give the second test ((iii)): "
      f"{split_sent}.")
    w("")
    w("**Per field** (SAT sample; fields with an estimate for the main spec; SE = joint bootstrap; within-inst. β_f = "
      "within-institution slope with field-specific returns, 95% institution cluster-bootstrap CI).")
    w("")
    w("| field | licensed | reliable | n_SAT | raw ρ | ρ \\| selectivity set | ρ \\| selectivity set + G (SE) | ρ \\| + program Pell (SE) | brand ρ(G, Y \\| selectivity set + F) | residual reliability LB / EXT | within-inst. β_f [95% CI] |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    tt = T[np.isfinite(T["F_broadG"])].sort_values("F_broadG", ascending=False)
    for f, r in tt.iterrows():
        fa = f"{_f(r['F_all'], 2)} ({r['se_F_all']:.2f})" if np.isfinite(r["F_all"]) else "—"
        fe = (f"{_f(r['fe_FE2_beta'], 3)} {_ci(r['fe_FE2_ci_lo'], r['fe_FE2_ci_hi'], 3)}"
              if np.isfinite(r.get("fe_FE2_beta", np.nan)) else "—")
        w(f"| {r['label']} | {'yes' if r['licensed'] else ''} | {'yes' if r['reliable'] else ''} | {int(r['n_sat'])} | "
          f"{_f(r['raw'], 2)} | {_f(r['broad'], 2)} | {_f(r['F_broadG'], 2)} ({r['se_F_broadG']:.2f}) | {fa} | "
          f"{_f(r['G_broadF'], 2)} | {_f(r['relperp_LB_F_broadG'], 2, False)} / {_f(r['relperp_EXT_F_broadG'], 2, False)} | {fe} |")
    w("")

    # ---------------- (iii) within-institution ----------------
    w("## (iii) Within-institution cross-department design with field-specific returns")
    w("")
    w("log Y_if = α_i + γ_f + β·z(F)_if + Σ_f D_f·(d_f·zSAT_i + a_f·zADM_i + p_f·zPCTPELL_i + t_f·zG_i) + e_if. "
      "α_i absorbs each institution's level (its average selectivity, brand, location, composition); the field × "
      "institution-trait slopes let every field price SAT, admit rate, Pell share and brand at its own rate, so β is "
      "identified only from departments whose prestige differs from what their institution's traits imply, compared "
      "with other departments of the same institution. z(F) and the traits are standardized within field. Institutions "
      "with one field and fields with fewer than 15 programs are dropped iteratively. SEs: CR1 clustered by "
      "institution, and an institution (pairs) cluster bootstrap in which all specifications on the same cells share "
      "the draws, so their differences are paired.")
    w("")
    w("| specification | programs | institutions | fields | parameters | β (CR1 SE) | 95% bootstrap CI | residual variance share of z(F) | β / λ: LB, EXT |")
    w("|---|---|---|---|---|---|---|---|---|")
    for n in ["FE0", "FE0_s", "FE2sg", "FE1", "FE2", "FE2ie", "FE2sc"]:
        r = FE[n]
        w(f"| {FE_DESC[n]} | {r['n_cells']:,} | {r['n_inst']} | {r['n_fields']} | {r['n_par']} | {_f(r['beta'], 4)} "
          f"({r['se']:.4f}) | {_ci(*r['boot_ci'], 4)} | {r['resid_var_share']:.2f} | {_f(r['beta'] / r['lam_LB'], 4)}, "
          f"{_f(r['beta'] / r['lam_EXT'], 4)} |")
    for a, b in [("FE2", "FE0_s"), ("FE2sg", "FE0_s"), ("FE2sc", "FE2")]:
        q = FE[f"pair_{a}_{b}"]
        w(f"| paired difference: ({FE_DESC[a]}) − ({FE_DESC[b]}) | | | | | {_f(q['diff'], 4)} ({q['se']:.4f}) | {_ci(*q['ci'], 4)} | | |")
    w("")
    w("β / λ is the slope corrected for classical measurement error in z(F): λ = 1 − (error variance left after "
      "within-institution demeaning) / (residual variance of z(F) given the other regressors), with the field error "
      "variance 1 − rel_F from scripts/56 (LB or EXT).")
    w("")
    w(f"**Per-field β_f** (main spec with β_f by field; {len(pf['beta_f'])} fields). CR1 SEs are fragile here "
      f"({FE['FE2_pf']['n_par']} parameters, {FE['FE2_pf']['n_inst']} clusters), so heterogeneity is also computed "
      f"with bootstrap SEs (draws where a field has fewer than {FE_PF_MIN_DISTINCT} distinct institutions are "
      "dropped for that field).")
    w("")
    w("| SEs used | mean β_f | REML μ (SE) | τ [95% upper] | Q p | β_f > 0 / < 0 at 1.96 SE | BH > 0 / < 0 |")
    w("|---|---|---|---|---|---|---|")
    for tag, lab in (("FE2", "CR1 (clustered)"), ("FE2_bootsd", "bootstrap SD"), ("FE2_bootiqr", "bootstrap IQR/1.349")):
        q = FEPF[tag]
        w(f"| {lab} | {_f(q['mean'], 4)} | {_f(q['mu_re'], 4)} ({q['se_re']:.4f}) | {q['tau']:.4f} [{q['tau_hi']:.4f}] | "
          f"{_p(q['Qp'])} | {q['n_pos']} / {q['n_neg']} | {q['n_bh_pos']} / {q['n_bh_neg']} |")
    w(f"| (a) without field-specific slopes (CR1), for reference | {_f(FEPF['FE0']['mean'], 4)} | "
      f"{_f(FEPF['FE0']['mu_re'], 4)} ({FEPF['FE0']['se_re']:.4f}) | {FEPF['FE0']['tau']:.4f} "
      f"[{FEPF['FE0']['tau_hi']:.4f}] | {_p(FEPF['FE0']['Qp'])} | {FEPF['FE0']['n_pos']} / {FEPF['FE0']['n_neg']} | "
      f"{FEPF['FE0']['n_bh_pos']} / {FEPF['FE0']['n_bh_neg']} |")
    w("")
    w(f"Wald test that all β_f are equal (CR1, pseudo-inverse): χ²({FEPF['FE2']['wald_df']}) = {FEPF['FE2']['wald']:.1f}, "
      f"p = {_p(FEPF['FE2']['wald_p'])}; with a rank-deficient cluster covariance this test may over-reject and is not "
      "relied on.")
    w("")
    w(f"**Wild-cluster heterogeneity tests.** The χ² reference of Q treats fields as independent, but fields share "
      f"institutions. Here the null (one common β for all fields) is imposed and its distribution simulated by a wild "
      f"cluster bootstrap ({WD['B']} replicates, one Rademacher sign per institution): the common-β model's residuals "
      f"(restricted residuals) are multiplied by the institution's sign and added back to its fitted values, the "
      f"per-field model is re-estimated, and Q is recomputed with the same weights (CR1 weights are recomputed in "
      f"every replicate, a studentized Q). The same draw applies to all fields, so their dependence through shared "
      f"institutions is kept. Plain restricted residuals are shrunk by the fit: a field with 15–30 programs has 6 "
      f"field-specific parameters, so its residuals understate its errors and the null is too narrow for it. The "
      f"per-field SD of the null β_f, as a share of the pairs-bootstrap IQR-based SE (median by field size: < "
      f"{PF_NMIN_MID} / {PF_NMIN_MID}–{PF_SPLIT_NMIN - 1} / ≥ {PF_SPLIT_NMIN} programs, {WD['n_size']['lt30']} / "
      f"{WD['n_size']['30_59']} / {WD['n_size']['ge60']} fields), is "
      + " / ".join(f"{WD['sd_ratio']['WCR'][c_]:.2f}" for c_ in ("lt30", "30_59", "ge60"))
      + " with restricted residuals and "
      + " / ".join(f"{WD['sd_ratio']['WCR3'][c_]:.2f}" for c_ in ("lt30", "30_59", "ge60"))
      + " with leverage-corrected restricted residuals (each institution's residuals multiplied by (I − H_gg)⁺, the "
      f"jackknife transformation of the WCR3x bootstraps in MacKinnon, Nielsen and Webb 2023, H_gg including the "
      f"absorbed institution FE). "
      + ("The earlier text called restricted residuals conservative here; with this much leverage they are not, so "
         "the leverage-corrected column is the one to read. " if WD["sd_ratio"]["WCR"]["lt30"] < 0.95 else
         "Here the leverage correction changes little. ")
      + f"Per-field residuals do not impose the null on the residuals (a check). p = (1 + replicates with Q at least as large) / (1 + {WD['B']}), so p ≤ "
      f"{pfloor:.3f} means no replicate reached the observed Q.")
    w("")
    w("| per-field β_f (main spec) | fields | weights | Q | χ² p (fields independent) | wild p, restricted residuals (null mean Q) | wild p, leverage-corrected restricted residuals (null mean Q) | wild p, per-field residuals |")
    w("|---|---|---|---|---|---|---|---|")
    wlab = {"iqr": "IQR-based bootstrap SE (fixed)", "sd": "bootstrap-SD SE (fixed)", "cr1": "CR1, per replicate"}
    for set_, lab in (("all", f"4-yr, all fields ({FE['FE2_pf']['n_cells']:,} programs)"),
                      ("mid", f"4-yr, fields with ≥ {PF_NMIN_MID} programs"),
                      ("big", f"4-yr, fields with ≥ {PF_SPLIT_NMIN} programs (split-half set)"),
                      ("mid_noeng", f"4-yr, fields with ≥ {PF_NMIN_MID} programs, without computer science and "
                                    f"engineering")):
        for wt in ("iqr", "sd", "cr1"):
            q = WQ[(set_, wt)]
            w(f"| {lab} | {q['k']} | {wlab[wt]} | {q['Q']:.1f} | {_p(q['chi2_p'])} | {pw(q['WCR']['p'])} "
              f"({q['WCR']['mean']:.1f}) | {pw(q['WCR3']['p'])} ({q['WCR3']['mean']:.1f}) | {pw(q['WCU']['p'])} |")
    for n in HZ_FE_PF + ["avg"]:
        q = WH[n]
        lab = (f"common programs, {HZ_LAB[n[-2:]]} earnings" if n != "avg" else
               "common programs, 4/1/5-yr average (pooled test)")
        w(f"| {lab} | {q['k']} | {wlab['iqr']} | {q['Q']:.1f} | {_p(q['chi2_p'])} | {pw(q['WCR']['p'])} "
          f"({q['WCR']['mean']:.1f}) | {pw(q['WCR3']['p'])} ({q['WCR3']['mean']:.1f}) | {pw(q['WCU']['p'])} |")
    w("")
    w(f"Counts under the null β_f = 0 for every field (wild cluster bootstrap from the model without a prestige "
      f"slope; fields counted at |β_f| > 1.96 × the IQR-based bootstrap SE): 4-yr main spec, "
      f"{WN4['n_pos']} of {WN4['k']} fields above zero (restricted residuals: null mean {WN4['WCR']['mean_pos']:.1f}, "
      f"95th percentile {WN4['WCR']['p95_pos']:.0f}, P(≥ {WN4['n_pos']}) {pe(WN4['WCR']['p_pos'])}; leverage-corrected: "
      f"null mean {WN4['WCR3']['mean_pos']:.1f}, 95th percentile {WN4['WCR3']['p95_pos']:.0f}, P(≥ {WN4['n_pos']}) "
      f"{pe(WN4['WCR3']['p_pos'])}) and {WN4['n_neg']} below (P(≥ {WN4['n_neg']}) {pe(WN4['WCR']['p_neg'])} / "
      f"{pe(WN4['WCR3']['p_neg'])}); 4/1/5-yr average on the common programs, {WNA['n_pos']} of {WNA['k']} above zero "
      f"(null mean {WNA['WCR']['mean_pos']:.1f} / {WNA['WCR3']['mean_pos']:.1f}, P(≥ {WNA['n_pos']}) "
      f"{pe(WNA['WCR']['p_pos'])} / {pe(WNA['WCR3']['p_pos'])}) and {WNA['n_neg']} below (P(≥ {WNA['n_neg']}) "
      f"{pe(WNA['WCR']['p_neg'])} / {pe(WNA['WCR3']['p_neg'])}). With per-field residuals the four probabilities are "
      f"{pw(WN4['WCU']['p_pos'])}, {pw(WN4['WCU']['p_neg'])}, {pw(WNA['WCU']['p_pos'])} and "
      f"{pw(WNA['WCU']['p_neg'])}.")
    w("")
    ih = PFS["inst_half"]
    w(f"**Does the cross-field ordering of β_f reproduce?** Split-half check: the institution universe is split at random "
      f"into halves ({PFS['splits_k']} splits, the same splits for every row; {ih[0]}–{ih[2]} of the {ih[3]} main-spec "
      f"institutions fall in half A, median {ih[1]}); on each half the whole model is re-estimated (institution FE, "
      f"field FE, the field-specific slopes, β_f for every field), and the β_f of well-estimated fields are correlated "
      f"across fields between the halves. Reliability = mean over splits of that correlation; p = {N_PERM} field-label "
      f"permutations of half B. A field's β_f is dropped from a half with fewer than {FE_PF_MIN_DISTINCT} of its "
      f"institutions.")
    w("")
    w("| specification | fields | split-half reliability, Pearson (permutation p) | Spearman (permutation p) | p against simulated τ = 0 | permutation null 95th pct |")
    w("|---|---|---|---|---|---|")
    sp_lab = {"main": f"main (field × {{SAT, ADM, inst. Pell, G}}), fields with ≥ {PF_SPLIT_NMIN} programs",
              "main_sc": f"main + field × {{state earnings level, private}}, ≥ {PF_SPLIT_NMIN} programs",
              "main_n40": f"main, fields with ≥ {PF_SPLIT_NMIN2} programs",
              "main_nocs": f"main, ≥ {PF_SPLIT_NMIN} programs, without computer science",
              "main_noeng": f"main, ≥ {PF_SPLIT_NMIN} programs, without computer science and engineering "
                            f"({', '.join(LAB.get(f, f) for f in PFS['eng_in_big'])})",
              "literal": f"(a) + field × {{SAT, G}} (the design as specified), ≥ {PF_SPLIT_NMIN} programs",
              "a_s": f"(a) without field-specific slopes (same cells), ≥ {PF_SPLIT_NMIN} programs"}
    for tag in ["main", "main_sc", "main_n40", "main_nocs", "main_noeng", "literal", "a_s"]:
        q = PFS[tag]
        w(f"| {sp_lab[tag]} | {q['k']} | {_f(q['rel'], 3)} ({_p(q['p'])}) | {_f(q['rel_s'], 3)} ({_p(q['p_s'])}) | "
          f"{_p(q.get('p_sim', np.nan))} | {_f(q['null95'], 3)} |")
    w("")
    w(f"Main-spec reading: the ordering of the department-specific slopes across fields {repro_vb} across "
      f"independent halves of institutions ({_f(PFS['main']['rel'], 2)}; permutation p = {_p(PFS['main']['p'])}). The "
      f"permutation null conditions on the realized full-sample estimates; against the simulated τ = 0 null of (ii), "
      f"which also lets their dispersion vary by chance, p = {_p(PWB['p_sim'])} (simulated 95th percentile "
      f"{_f(PWB['crit'], 2)} against the permutation 95th percentile {_f(PFS['main']['null95'], 2)}). The Q test on "
      f"the same {PFS['main']['k']} fields gives p = {_p(PWB['Q_p'])} (bootstrap-SD SEs; τ = {PWB['Q_tau']:.3f}) and "
      f"{_p(PWB['Q_p_iqr'])} (IQR-based SEs; τ = {PWB['Q_tau_iqr']:.3f}). On the {PFS['main_n40']['k']} fields with "
      f"≥ {PF_SPLIT_NMIN2} programs the reliability is {_f(PFS['main_n40']['rel'], 2)} (permutation p = "
      f"{_p(PFS['main_n40']['p'])}, simulated-null p = {_p(PFS['main_n40']['p_sim'])}). Removing computer science gives "
      f"{spl(PFS['main_nocs'])}, and removing computer science and the engineering fields gives "
      f"{spl(PFS['main_noeng'])} (permutation p; simulated-null p {_p(PFS['main_nocs']['p_sim'])} and "
      f"{_p(PFS['main_noeng']['p_sim'])}), so the reproducible part is "
      + ("carried mainly by those fields. " if PFS['main_noeng']['p'] >= 0.05 else "not confined to those fields. ")
      + f"The engineering grouping was chosen after seeing the per-field results. Without field-specific slopes "
      f"(spec a, same cells) the ordering is far more reliable ({_f(PFS['a_s']['rel'], 2)}): most of the cross-field "
      f"variation in the raw within-institution slopes is each field's own pricing of institution traits, which the "
      f"field-specific slopes remove; the design as specified (field × {{SAT, G}}) gives {spl(PFS['literal'])}. A "
      f"reproducible ordering would say that the field differences are not sampling noise; it would not say what "
      f"they are: department standing (A) and program-level selection into majors that institution-level measures "
      f"miss (B) would both reproduce across halves of institutions.")
    w("")
    w(f"**Nonlinear field-specific pricing of brand and selectivity.** The main spec lets each field price brand G, "
      f"SAT, admit rate and Pell share linearly. If some fields pay mostly for the top of the brand distribution "
      f"(convex pricing, itself a reading-B mechanism), a linear field × G slope leaves that in the residual, where it "
      f"can load on department prestige differently by field. Two variants on the same {nl0['n_cells']:,} programs: "
      f"field × {{zG², zG³}} added (cubic brand pricing, {nlg['n_par']} parameters) and field × cubic terms in G, SAT "
      f"and admit rate ({nla['n_par']} parameters). Per-field bootstrap: the same {nl0['B']} institution draws as the "
      f"main spec; a field's β_f counts in a draw only with at least (its field-specific parameters + 2) distinct "
      f"institutions ({nl0['min_distinct']} / {nlg['min_distinct']} / {nla['min_distinct']}), and fields defined in "
      f"fewer than half the draws are left out of the heterogeneity tests ("
      + (f"{names(nla['not_est'])} in the cubic G / SAT / ADM spec" if nla["not_est"] else "none")
      + (f"; {names(nlg['not_est'])} with cubic G" if nlg["not_est"] else "")
      + f"). Wild-cluster Q: the same Rademacher signs as above, IQR-based SEs of each spec as fixed weights. "
      f"Split-half: the same {PFS['splits_k']} splits and field permutations as the main spec.")
    w("")
    w("| specification | common β [95% CI] | β_f corr. with main (≥ 30 / all) | CS β_f [95% CI] | CS/eng − other, ≥ 30 programs [95% CI] | τ, ≥ 60 programs (IQR SEs) | χ² Q p ≥ 60: IQR / SD SEs | wild p ≥ 60: restricted / leverage-corrected | χ² Q p ≥ 30; leverage-corrected wild p ≥ 30 | all estimable fields: k, χ² Q p, leverage-corrected wild p | ≥ 30 without CS/eng: leverage-corrected wild p | split-half ≥ 60 (p); without CS; without CS/eng |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in NL_SPECS:
        q, h_ = NL[n], NL[n]["het"]
        w(f"| {FE_DESC[n]} | {_fe_ci(FE[n])} | {_f(q['corr_main_mid'], 2)} / {_f(q['corr_main_all'], 2)} | "
          f"{_f(q['beta_f'][cs], 3)} {_ci(q['lo'][cs], q['hi'][cs], 3)} | {_f(q['eng_mid']['diff'], 3)} "
          f"{_ci(q['eng_mid']['lo'], q['eng_mid']['hi'], 3)} | {h_[('big', 'iqr')]['tau']:.4f} | "
          f"{_p(h_[('big', 'iqr')]['p'])} / {_p(h_[('big', 'sd')]['p'])} | {pw(h_[('big', 'wild_iqr')]['WCR']['p'])} / "
          f"{lev(h_[('big', 'wild_iqr')])} | {_p(h_[('mid', 'iqr')]['p'])}; {lev(h_[('mid', 'wild_iqr')])} | "
          f"{h_[('all', 'iqr')]['k']}, {_p(h_[('all', 'iqr')]['p'])}, {lev(h_[('all', 'wild_iqr')])} | "
          f"{lev(h_[('mid_noeng', 'wild_iqr')])} | {spl(q['split']['main'])}; {spl(q['split']['main_nocs'])}; "
          f"{spl(q['split']['main_noeng'])} |")
    w("")
    w(f"Paired differences of the common β (same draws): cubic G − main {_f(FE['pair_FE2cg_FE2']['diff'], 4)} "
      f"{_ci(*FE['pair_FE2cg_FE2']['ci'], 4)}; cubic G / SAT / ADM − main {_f(FE['pair_FE2ca_FE2']['diff'], 4)} "
      f"{_ci(*FE['pair_FE2ca_FE2']['ci'], 4)}. Fields with a per-field CI above zero: "
      + "; ".join(f"{lab_}: {names([f for f in NL[n]['beta_f'].index if NL[n]['lo'][f] > 0 and NL[n]['fail'][f] <= 0.5]) or 'none'}"
                  for n, lab_ in (("FE2", "main"), ("FE2cg", "cubic G"), ("FE2ca", "cubic G / SAT / ADM")))
      + ". Reading: the common β hardly moves and computer science keeps its premium in both variants"
      + ("" if cs_all_variants else " (not in every variant: see the table)")
      + ". The CS/engineering contrast holds with cubic brand pricing"
      + ("" if eng_cg else " only as a point estimate (CI includes zero)")
      + (" and its CI includes zero once SAT and admit rate are cubic too" if not eng_ca else
         " and with cubic SAT and admit rate too")
      + ". The general heterogeneity test on the well-estimated fields "
      + ("does not survive cubic brand pricing" if not het_nl_robust else "survives cubic brand pricing")
      + f" (p = {prange(ps_cg_big)}), and the split-half reliability falls to {_f(nlg['split']['main']['rel'], 2)}. "
      f"With cubic SAT and admit rate as well, the tests on those fields are mixed (χ² Q p "
      f"{_p(nla['het'][('big', 'iqr')]['p'])} / {_p(nla['het'][('big', 'sd')]['p'])}, wild p "
      f"{pw(nla['het'][('big', 'wild_iqr')]['WCR']['p'])} / {lev(nla['het'][('big', 'wild_iqr')])}) and the split-half "
      f"reliability is {_f(nla['split']['main']['rel'], 2)}; so "
      f"much of the cross-field spread of β_f in the linear spec goes with how fields price brand, not with a stable "
      f"ordering of department premia. These are two of many possible functional forms; they show the heterogeneity "
      f"result is fragile, not that the fields are the same.")
    w("")
    w("| field | programs | β_f | CR1 SE | bootstrap SE (SD / IQR) | 95% bootstrap CI | draws dropped |")
    w("|---|---|---|---|---|---|---|")
    order = pf["beta_f"].sort_values(ascending=False).index
    for f in order:
        w(f"| {LAB.get(f, f)} | {int(pf['n_by_field'][f])} | {_f(pf['beta_f'][f], 3)} | {pf['se_f'][f]:.3f} | "
          f"{pf['se_boot'][f]:.3f} / {pf['se_iqr'][f]:.3f} | {_ci(pf['ci_lo'][f], pf['ci_hi'][f], 3)} | "
          f"{pf['boot_fail'][f]:.1%} |")
    w("")
    w(f"**Coefficient stability (Oster 2019).** Total R² (institution FE included where present) on the main-spec cells: "
      f"short model without institution FE (field FE + β z) R² = {O['r_s0']:.3f}, β = {_f(O['beta_s0'], 4)}; with "
      f"institution FE (spec a) R² = {O['r_s1']:.3f}, β = {_f(O['beta_s1'], 4)}; main spec R² = {O['r_long']:.3f}, β = "
      f"{_f(O['beta_long'], 4)}. δ* is how strong selection on unobservables would have to be, relative to selection "
      f"on the controls added between the short model and the main spec, for the true β to be zero; β*(δ = 1) is the bias-adjusted "
      f"β under equal selection.")
    w("")
    w("| short model | R_max | δ* | β*(δ = 1) |")
    w("|---|---|---|---|")
    for tag, lab in (("s0", "field FE only (no institution FE)"), ("s1", "institution FE + field FE (spec a)")):
        for rtag, rl in (("13", f"min(1, 1.3 R²_main) = {O['rmax_13']:.3f}"),
                         ("half", f"R²_main + half the unexplained = {O['rmax_half']:.3f}")):
            w(f"| {lab} | {rl} | {O[f'delta_{tag}_{rtag}']:.2f} | {_f(O[f'bstar_{tag}_{rtag}'], 4)} |")
    w("")
    w(f"Program medians carry sampling noise, so R² = 1 is not attainable; the second R_max is a less demanding "
      f"alternative. Against the model without institution FE the remainder would vanish with unobserved selection "
      f"{O['delta_s0_13']:.2f}–{O['delta_s0_half']:.2f} times as strong as the observed; against the institution-FE "
      f"model, with the field-specific slopes as the only added controls, it takes {O['delta_s1_13']:.2f}–"
      f"{O['delta_s1_half']:.2f}. The remainder is fragile to unobserved program-level selection of a size comparable "
      f"to the selection already controlled.")
    w("")

    # ---------------- horizon replication ----------------
    w("## Replication with 1-year and 5-year earnings")
    w("")
    w("The FoS file also releases median earnings 1 year (EARN_MDN_1YR) and 5 years (EARN_MDN_5YR) after the "
      "bachelor's. In one release the three horizons describe different graduating cohorts, measured in different "
      "years (README §3), so this is a replication on other cohorts and at other career points, not a career-time "
      "comparison. It is not an independent replication: the programs, the prestige measure and the controls are "
      "the same, and a program's residual earnings persist across its cohorts (see the cross-horizon benchmark "
      "below). Each outcome's program-level median is aggregated over the field's CIP-4 rows exactly as the "
      "4-year one. Every comparison holds the programs fixed: the 1-year sample and 5-year sample are the SAT sample "
      "restricted to programs with that horizon released, and the common sample has all three. Specifications, "
      "controls and bootstrap draws are the same as for the 4-year estimates, so the horizon differences are paired "
      f"(joint institution bootstrap B = {B_BOOT} for partial correlations; institution cluster bootstrap B = {B_FE} "
      "for the within-institution β, one draw set per sample).")
    w("")
    w("**Partial correlations** (mean across fields; 95% CI = joint bootstrap, normal approximation; percentile CI in "
      "the last column; fields > 0 / < 0 at 1.96 SE).")
    w("")
    w("| specification | sample | k | mean [95% CI] | fields > 0 / < 0 | percentile CI |")
    w("|---|---|---|---|---|---|")
    for c in ["F_broadG_h5y4", "F_broadG_h5y5", "F_broadG_h1y4", "F_broadG_h1y1",
              "raw_h3y4", "raw_h3y1", "raw_h3y5", "broad_h3y4", "broad_h3y1", "broad_h3y5",
              "F_broadG_h3y4", "F_broadG_h3y1", "F_broadG_h3y5", "G_broadF_h3y4", "G_broadF_h3y1", "G_broadF_h3y5"]:
        q = S[c]
        core, smp = SPEC_DESC[c].rsplit(", ", 1)
        w(f"| {_t(core)} | {smp} | {q['k']} | {_est(q)} | {q['n_pos']} / {q['n_neg']} | {ci_pct(q)} |")
    w(f"| for reference: {_t(SPEC_DESC['F_broadG'])}, 4-yr | SAT sample (all programs) | {main['k']} | {_est(main)} | "
      f"{main['n_pos']} / {main['n_neg']} | {ci_pct(main)} |")
    w("")
    w("Paired differences (same fields, same draws):")
    w("")
    w("| difference | k | mean difference [95% CI] |")
    w("|---|---|---|")
    for a, b in [("F_broadG_h5y5", "F_broadG_h5y4"), ("F_broadG_h1y1", "F_broadG_h1y4"), ("raw_h3y5", "raw_h3y4"),
                 ("raw_h3y1", "raw_h3y4"), ("broad_h3y5", "broad_h3y4"), ("broad_h3y1", "broad_h3y4"),
                 ("F_broadG_h3y5", "F_broadG_h3y4"), ("F_broadG_h3y1", "F_broadG_h3y4"),
                 ("G_broadF_h3y5", "G_broadF_h3y4"), ("G_broadF_h3y1", "G_broadF_h3y4"),
                 ("F_broadG_h3y4", "broad_h3y4"), ("F_broadG_h3y1", "broad_h3y1"), ("F_broadG_h3y5", "broad_h3y5")]:
        q = D[(a, b)]
        w(f"| ({_t(SHORT[a])}) − ({_t(SHORT[b])}) | {q['k']} | {_f(q['diff'])} {_ci(q['lo'], q['hi'])} |")
    w("")
    hc = T[[c for c in ["F_broadG_h3y4", "F_broadG_h3y1", "F_broadG_h3y5"]]].dropna()
    w(f"Across the {len(hc)} common-sample fields, the per-field main-spec partials correlate at "
      f"r = {hc.F_broadG_h3y4.corr(hc.F_broadG_h3y1):+.2f} (4 vs 1 year) and "
      f"{hc.F_broadG_h3y4.corr(hc.F_broadG_h3y5):+.2f} (4 vs 5 years); they are computed on the same programs, so "
      "part of this correlation is shared program-level noise (no null benchmark is computed for this design; see the "
      "within-institution benchmark below). Reliability bracket (as in (ii)), disattenuated means:")
    w("")
    w("| specification | k | LB: fields identified, disattenuated mean [95% CI] | EXT: same |")
    w("|---|---|---|---|")
    for c in ["F_broadG_h5y4", "F_broadG_h5y5", "F_broadG_h1y4", "F_broadG_h1y1", "F_broadG_h3y4", "F_broadG_h3y1",
              "F_broadG_h3y5"]:
        r = REL[c]
        w(f"| {_t(SPEC_DESC[c])} | {r['k']} | {r['LB']['k_id']}, {_est(r['LB'])} | {r['EXT']['k_id']}, "
          f"{_est(r['EXT'])} |")
    w("")
    w("**Within-institution β** (log points per within-field SD of department prestige; 95% institution "
      "cluster-bootstrap percentile CI; P(β ≤ 0) = share of bootstrap draws ≤ 0).")
    w("")
    w("| specification | programs | institutions | fields | β (CR1 SE) | 95% bootstrap CI | P(β ≤ 0) |")
    w("|---|---|---|---|---|---|---|")
    for n in ["FE2_h5y4", "FE2_h5y5", "FE2_h1y4", "FE2_h1y1", "FE0_h3y4", "FE0_h3y1", "FE0_h3y5", "FE2_h3y4",
              "FE2_h3y1", "FE2_h3y5"] + [f"{b}_h3{h}" for b in ("FE2sg", "FE2w") for h in HZ3] + \
             ["FE2v_h3y1", "FE2v_h3y5", "FE2"]:
        r = FE[n]
        lab = FE_DESC[n] + (" (for reference: SAT-covariate sample, 4-yr)" if n == "FE2" else "")
        w(f"| {lab} | {r['n_cells']:,} | {r['n_inst']} | {r['n_fields']} | {_f(r['beta'], 4)} ({r['se']:.4f}) | "
          f"{_ci(*r['boot_ci'], 4)} | {r['boot_p0']:.3f} |")
    for a, b in [("FE2_h5y5", "FE2_h5y4"), ("FE2_h1y1", "FE2_h1y4"), ("FE0_h3y5", "FE0_h3y4"), ("FE0_h3y1", "FE0_h3y4"),
                 ("FE2_h3y5", "FE2_h3y4"), ("FE2_h3y1", "FE2_h3y4"), ("FE2sg_h3y5", "FE2sg_h3y4"),
                 ("FE2sg_h3y1", "FE2sg_h3y4"), ("FE2w_h3y5", "FE2w_h3y4"), ("FE2w_h3y1", "FE2w_h3y4")]:
        q = FE[f"pair_{a}_{b}"]
        w(f"| paired difference: ({FE_DESC[a]}) − ({FE_DESC[b]}) | | | | {_f(q['diff'], 4)} ({q['se']:.4f}) | "
          f"{_ci(*q['ci'], 4)} | |")
    w("")
    w("β / λ (reliability-corrected, as in (iii)): " + "; ".join(
        f"{FE_DESC[n]} {_f(FE[n]['beta'] / FE[n]['lam_LB'], 4)} (LB), {_f(FE[n]['beta'] / FE[n]['lam_EXT'], 4)} (EXT)"
        for n in ["FE2_h3y4", "FE2_h3y1", "FE2_h3y5"]) + ".")
    w("")
    w("**Pooled over the three horizons** (common programs). Each estimate is the average of its 4-, 1- and 5-year "
      "versions; the 95% CI comes from averaging the three estimates within each bootstrap draw (the same institution "
      "draws for all horizons and specifications), so it allows for the dependence between horizons that share "
      "programs, the prestige measure and the controls. This replaces counting how many horizons are separately "
      "significant. Weighted rows: WLS with each program weighted by the number of graduates behind its median (that "
      "horizon's count, or the 4-year count in the last row), institution FE absorbed by weighted demeaning; "
      "regressors as in the unweighted spec. CR1 SE in brackets for single horizons.")
    w("")
    w("| estimate | k fields | 4-yr | 1-yr | 5-yr | 4/1/5-yr average [95% CI] | percentile CI or P(β ≤ 0) |")
    w("|---|---|---|---|---|---|---|")
    for fam in HZ_AVG_P:
        q = S[f"{fam}_h3avg"]
        w(f"| mean {_t(SPEC_DESC[f'{fam}_h3avg'].rsplit(', average', 1)[0])} | {q['k']} | "
          + " | ".join(_f(S[f"{fam}_h3{h}"]["mean"]) for h in HZ3) + f" | {_est(q)} | {ci_pct(q)} |")
    for fam in HZ_FE_FAM:
        r = AV[fam]
        w(f"| within-institution β: {FE_DESC[f'{fam}_h3avg'].rsplit(', average', 1)[0]} | {r['n_fields']} | "
          + " | ".join(f"{_f(FE[f'{fam}_h3{h}']['beta'], 4)} ({FE[f'{fam}_h3{h}']['se']:.4f})" for h in HZ3)
          + f" | {_fe_ci(r)} | P = {r['boot_p0']:.3f} |")
    for a_, b_ in (("FE2w_h3avg", "FE2"), ("FE2v_h3avg", "FE2")):
        q = FE[f"pair_{a_}_{b_}_h3avg"]
        w(f"| paired difference: ({FE_DESC[a_].rsplit(', average', 1)[0]}) − ({FE_DESC[f'{b_}_h3avg'].rsplit(', average', 1)[0]}), "
          f"4/1/5-yr average | | | | | {_f(q['diff'], 4)} {_ci(*q['ci'], 4)} | |")
    w("")
    w(f"Per horizon, the weighted main spec has 95% bootstrap CIs "
      + " / ".join(_ci(*FE[f"FE2w_h3{h}"]["boot_ci"], 4) for h in HZ3) + " (4 / 1 / 5 years; with 4-year-count weights "
      + " / ".join(_ci(*FE[f"FE2v_h3{h}"]["boot_ci"], 4) for h in HZ3) + "), against "
      + " / ".join(_ci(*FE[f"FE2_h3{h}"]["boot_ci"], 4) for h in HZ3) + " unweighted. The spec-literal design "
      "(field × {SAT, G}): " + " / ".join(_ci(*FE[f"FE2sg_h3{h}"]["boot_ci"], 4) for h in HZ3) + ".")
    w("")
    w(f"**Per-field β_f at each horizon** (main spec with β_f by field on the common programs; one set of "
      f"{B_FE} institution draws for all three horizons; draws where a field has fewer than {FE_PF_MIN_DISTINCT} "
      "distinct institutions are dropped for that field).")
    w("")
    w("| horizon | fields | mean β_f | REML μ (SE) | τ [95% upper] | Q p: χ² / wild cluster | 95% CI above / below zero | BH > 0 / < 0 (IQR SEs) |")
    w("|---|---|---|---|---|---|---|---|")
    for h in ("y4", "y1", "y5"):
        q = hzf[h]
        w(f"| {HZ_LAB[h]} | {q['k']} | {_f(q['mean'], 4)} | {_f(q['mu_re'], 4)} ({q['se_re']:.4f}) | {q['tau']:.4f} "
          f"[{q['tau_hi']:.4f}] | {_p(q['Qp'])} / {pw(WH[f'FE2_h3{h}']['WCR']['p'])} | {len(q['ci_pos'])} / "
          f"{len(q['ci_neg'])} | {q['n_bh_pos']} / "
          f"{q['n_bh_neg']} |")
    re_av = reml(pfb.to_numpy(), PF["se_iqr"].reindex(pfb.index).to_numpy() ** 2)
    assert abs(re_av["p"] - WH["avg"]["chi2_p"]) < 1e-9
    assert all(abs(hzf[h]["Qp"] - WH[f"FE2_h3{h}"]["chi2_p"]) < 1e-9 for h in HZ3)
    zav = pfb / PF["se_iqr"].reindex(pfb.index)
    w(f"| 4/1/5-yr average (pooled test) | {len(pfb)} | {_f(float(pfb.mean()), 4)} | {_f(re_av['mu'], 4)} "
      f"({re_av['se']:.4f}) | {re_av['tau']:.4f} [{re_av['tau_hi']:.4f}] | {_p(re_av['p'])} / {pw(WH['avg']['WCR']['p'])} | "
      f"{len(pf_pos)} / "
      f"{len(pf_neg)} | {int(sum(PF['bh_n'][f] and zav[f] > 0 for f in pfb.index))} / "
      f"{int(sum(PF['bh_n'][f] and zav[f] < 0 for f in pfb.index))} |")
    w("")
    w(f"Mean β_f of the pooled test with a CI from the same draws, under three weightings of the fields: equal; "
      f"precision (1 / variance of the field's pooled bootstrap draws, so precisely estimated fields count more; the "
      f"{len(eng)} CS/engineering fields carry {100 * GW['wshare_eng']:.0f}% of the precision weight); and program "
      f"count. Groups (chosen after seeing the per-field results, so they describe where the remainder sits rather "
      f"than test a hypothesis): computer science and engineering ({names(eng)}), business ({names(BUS_)}), the "
      f"remaining {k_rest} fields, and complements.")
    w("")
    w("| group | k | equal weights [95% CI] | precision weights [95% CI] | program-count weights [95% CI] |")
    w("|---|---|---|---|---|")
    for g_, lab_ in (("all", "all fields"), ("eng", "computer science and engineering"), ("other", "all other fields"),
                     ("bus", "business"), ("rest", "neither CS/engineering nor business")):
        w(f"| {lab_} | {GW[(g_, 'equal')]['k']} | {gwm(g_, 'equal')} | {gwm(g_, 'precision')} | {gwm(g_, 'programs')} |")
    for c_, lab_ in (("eng-other", "difference: CS/engineering − all other"), ("bus-rest", "difference: business − neither"),
                     ("bus-nonbus", "difference: business − all other"), ("eng-rest", "difference: CS/engineering − neither")):
        w(f"| {lab_} | | {gwd(c_, 'equal')} | {gwd(c_, 'precision')} | {gwd(c_, 'programs')} |")
    w("")
    w(f"The equal-weight CS/engineering row reproduces the earlier contrast ({_f(gr['eng_minus_other']['diff'], 4)}). "
      f"Random sets of field labels of the same size (N = {N_PERM}): 'set minus the other fields' reaches the observed "
      f"CS/engineering contrast in {100 * labp['eng']['p_eq']:.1f}% of sets (equal weights; 95th percentile "
      f"{_f(labp['eng']['p95_eq'], 4)}) and {100 * labp['eng']['p_pr']:.1f}% (precision weights), and the business "
      f"contrast in {100 * labp['bus']['p_eq']:.1f}% / {100 * labp['bus']['p_pr']:.1f}%. These shares take the "
      f"grouping as given; choosing it after seeing the results would make them larger. Per field, the engineering "
      f"fields are {vals(srt(ENG_))} and the business fields {vals(srt(BUS_))}.")
    w("")
    w("Fields whose CI excludes zero at any single horizon or in the pooled test, fields kept by either BH-FDR "
      "version, and the fields named in advance in scripts/55 (HIGH):")
    w("")
    w("| field | programs | β_f 4-yr [95% CI] | β_f 1-yr [95% CI] | β_f 5-yr [95% CI] | 5 − 4 yr [95% CI] | "
      "4/1/5-yr average [95% CI] | p: normal / raw percentile |")
    w("|---|---|---|---|---|---|---|---|")
    pfs = {h: FE[f"FE2_h3{h}_pf"] for h in ("y4", "y1", "y5")}
    anyf = sorted(set(hzf["y4"]["ci_pos"] + hzf["y4"]["ci_neg"] + hzf["y1"]["ci_pos"] + hzf["y1"]["ci_neg"]
                      + hzf["y5"]["ci_pos"] + hzf["y5"]["ci_neg"] + pf_pos + pf_neg + h55 + bh_n + bh_p),
                  key=lambda f: -pfb[f])
    d5 = FE["pfdiff_FE2_h3y5"]
    for f in anyf:
        cells_ = []
        for h in ("y4", "y1", "y5"):
            r = pfs[h]
            cells_.append(f"{_f(r['beta_f'][f], 3)} {_ci(r['ci_lo'][f], r['ci_hi'][f], 3)}")
        w(f"| {LAB.get(f, f)} | {int(pfs['y4']['n_by_field'][f])} | {' | '.join(cells_)} | {_f(d5['diff'][f], 3)} "
          f"{_ci(d5['lo'][f], d5['hi'][f], 3)} | {_f(pfb[f], 3)} {_ci(PF['lo'][f], PF['hi'][f], 3)} | "
          f"{_p(PF['p_n'][f])} / {pp_raw(f)} |")
    w("")
    w(f"Raw percentile p = 2 × share of the {B_FE} draws on the far side of zero (0 when no draw crosses zero, which at "
      f"this B means only p below about 0.002); normal p uses the IQR-based bootstrap SE. Full per-field results: "
      f"columns `fe_FE2_h3avg_*` of the CSV.")
    w("")
    w(f"**Is the correlation of β_f across horizons more than reusing the same programs produces?** The three horizons "
      f"are estimated on the same {WD['hz_cells']:,} programs ({WD['hz_inst']} institutions) with the same prestige "
      f"measure and controls, and a program's residual earnings persist across its cohorts. So the per-field slopes "
      f"would correlate across horizons even if every field had the same β. Null benchmark: the common-β model is fitted "
      f"at each horizon, its residuals are multiplied by one Rademacher sign per institution, the same signs at every "
      f"horizon (so each program's residuals keep their correlation across horizons), and the per-field model is "
      f"re-estimated at each horizon ({WD['B']} replicates; restricted residuals as in (iii), plain and "
      f"leverage-corrected; the per-field model's residuals as a check). p = (1 + replicates with a correlation at "
      f"least as large) / (1 + {WD['B']}). A second benchmark needs no model of the null: the per-field pairs-bootstrap "
      f"draws of the three horizons use the same institution draws, so each field's draws correlate across horizons; "
      f"centred on the point estimates they are sampling error only, and their cross-field correlation between two "
      f"horizons, draw by draw, is the correlation that shared sampling error alone produces.")
    w("")
    w("| horizons | r (Spearman) | residual correlation across horizons: common-β / per-field model | null, restricted residuals: mean r / 95th pct / p (Spearman p) | null, leverage-corrected restricted residuals: mean r / 95th pct / p | null, per-field residuals: mean r / 95th pct / p (Spearman p) |")
    w("|---|---|---|---|---|---|")
    for kk, lab in ((k41, "4 vs 1 yr"), (k45, "4 vs 5 yr"), (k15, "1 vs 5 yr")):
        q = WC[kk]
        w(f"| {lab} | {_f(q['r'], 2)} ({_f(q['rs'], 2)}) | {_f(q['rescorr_WCR'], 2)} / {_f(q['rescorr_WCU'], 2)} | "
          f"{_f(q['WCR']['mean'], 2)} / {_f(q['WCR']['p95'], 2)} / {pw(q['WCR']['p'])} ({pw(q['WCR']['p_s'])}) | "
          f"{_f(q['WCR3']['mean'], 2)} / {_f(q['WCR3']['p95'], 2)} / {pw(q['WCR3']['p'])} | "
          f"{_f(q['WCU']['mean'], 2)} / {_f(q['WCU']['p95'], 2)} / {pw(q['WCU']['p'])} ({pw(q['WCU']['p_s'])}) |")
    w("")
    w("| horizons | per-field correlation of bootstrap draws: median [IQR] over fields; computer science | centred draws, all fields: observed r / median / 95th pct / share ≥ observed | centred draws, fields with ≥ 30 programs: k, observed r / median / 95th pct / share ≥ observed |")
    w("|---|---|---|---|")
    for kk, lab in ((k41, "4 vs 1 yr"), (k45, "4 vs 5 yr"), (k15, "1 vs 5 yr")):
        q = XD[kk]
        w(f"| {lab} | {_f(q['draw_r_med'], 2)} [{_f(q['draw_r_q25'], 2)}, {_f(q['draw_r_q75'], 2)}]; "
          f"{_f(q['draw_r'][cs], 2)} | {_f(q['all']['obs'], 2)} / {_f(q['all']['null_med'], 2)} / "
          f"{_f(q['all']['null95'], 2)} / {q['all']['share']:.3f} | {q['mid']['k']}, {_f(q['mid']['obs'], 2)} / "
          f"{_f(q['mid']['null_med'], 2)} / {_f(q['mid']['null95'], 2)} / {q['mid']['share']:.3f} |")
    w("")
    n_sig_p = len(xh_null_sig)
    n_sig_s = sum(WC[kk]["WCR"]["p_s"] < 0.05 for kk in (k41, k45, k15))
    n_sig_u = sum(WC[kk]["WCU"]["p"] < 0.05 for kk in (k41, k45, k15))
    cnt_w = lambda n_: "none" if n_ == 0 else ("all three" if n_ == 3 else str(n_))
    w(f"For 4 vs 1 year most of the observed correlation is what shared programs produce under one common β "
      f"({_f(WC[k41]['WCR']['mean'], 2)} of {_f(WC[k41]['r'], 2)}). For the two pairs with 5-year earnings the null "
      f"mean is small ({_f(WC[k45]['WCR']['mean'], 2)} and {_f(WC[k15]['WCR']['mean'], 2)}) but the null is wide (95th "
      f"percentile {_f(WC[k45]['WCR']['p95'], 2)} and {_f(WC[k15]['WCR']['p95'], 2)}). With restricted residuals the "
      f"observed Pearson r exceeds the null at the 5% level for {cnt_w(n_sig_p)} of the 3 pairs (p "
      f"{trip(lambda q: pw(q['WCR']['p']))}) and the Spearman r for {cnt_w(n_sig_s)} (p "
      f"{trip(lambda q: pw(q['WCR']['p_s']))}); with the per-field model's residuals, whose null means are lower "
      f"mainly because they exclude the field differences that restricted residuals carry into every horizon, the "
      f"Pearson r exceeds it for {cnt_w(n_sig_u)} (p {trip(lambda q: pw(q['WCU']['p']))}); with leverage-corrected "
      f"restricted residuals for {cnt_w(sum(WC[kk]['WCR3']['p'] < 0.05 for kk in xk))} (p "
      f"{trip(lambda q: pw(q['WCR3']['p']))}). Centred pairs-bootstrap draws reach the observed r in {sh_all} of draws "
      f"({sh_mid} on the fields with ≥ {PF_NMIN_MID} programs). So the cross-horizon correlation cannot be told apart "
      f"from shared sampling error; whether a variant passes the 5% level depends on the bootstrap and the correlation "
      f"measure, and it is not used as evidence against sampling noise. Computer science's three slopes are close to "
      f"one estimate seen three times (draw correlations {cs_dr}); its evidence is the pooled test "
      f"({int(PF['n_far'][cs])} of {int(PF['n_fin'][cs])} draws at or below zero). Whether the fields differ is "
      f"judged by the wild-cluster Q tests and the across-institution split-half in (iii), which do not reuse "
      f"programs, and there the answer depends on how brand is priced.")
    w("")
    w("Reading: horizon by horizon, the 95% CI of the unweighted main-spec β "
      + ", ".join(f"{ex_word(fpos(FC[h]))} with {HZ_LAB[h]} earnings" for h in HZ3)
      + "; weighted by cohort size, it " + ", ".join(f"{ex_word(fpos(FE[f'FE2w_h3{h}']))} at {HZ_LAB[h]}" for h in HZ3)
      + f". Pooled over the three horizons the unweighted main-spec β is {_fe_ci(AV['FE2'])}, the spec-literal design gives "
      f"{_fe_ci(AV['FE2sg'])} and spec (a) {_fe_ci(AV['FE0'])}; weighting by cohort size gives {_fe_ci(AV['FE2w'])} "
      f"for the main spec. The pooled per-field test finds {len(pf_pos)} fields above zero "
      f"({', '.join(LAB.get(f, f) for f in pf_pos) or 'none'}), {len(pf_pos_eng)} of them computer science or "
      f"engineering, against ≈{0.025 * len(pfb):.2f} expected by chance ({cnta_sent}). The per-field slopes correlate "
      f"across horizons ({xh_short}), but because the horizons reuse the same programs and institutions this is not a "
      f"replication across cohorts and is not counted as evidence against noise (tables above). "
      + (lambda dd, sg: f"Of the {len(dd)} paired horizon differences (partial correlations and common β), "
                        f"{len(sg)} exclude{'s' if len(sg) == 1 else ''} zero"
         + (" (" + "; ".join(d['lab'] for d in sg) + "), so the premium may be smaller for the 1-year or 5-year "
            "cohorts in some designs; " if sg else "; ")
         + "because the horizons are different cohorts, this cannot be read as change over the early career.")(
          hz_diffs(ctx), [d for d in hz_diffs(ctx) if d["lo"] > 0 or d["hi"] < 0]))
    w("")

    # ---------------- (iv) horse race ----------------
    w("## (iv) Horse race: department prestige vs selectivity vs brand vs institution-wide earnings")
    w("")
    w("Per field, SAT sample: rank(Y) on standardized ranks of F, {SAT_AVG, −ADM_RATE}, G and MD_EARN_WNE_P10; all-"
      "subset R² → Shapley (general dominance) shares, complete dominance, unique R².")
    w("")
    w("| blocks | k | mean R² | mean Shapley: F | selectivity | G | inst. earnings | program Pell share | largest block: F / S / G / IE |")
    w("|---|---|---|---|---|---|---|---|---|")
    w(f"| F, S, G, IE | {H['k']} | {H['r2']:.3f} | {_est(H['sh_F'])} | {_est(H['sh_S'])} | {_est(H['sh_G'])} | "
      f"{_est(H['sh_IE'])} | — | {H['lead_F']} / {H['lead_S']} / {H['lead_G']} / {H['lead_IE']} |")
    w(f"| F, S, G (no inst. earnings) | {H['k']} | {H['r2_3']:.3f} | {_est(H['sh3_F'])} | {_est(H['sh3_S'])} | "
      f"{_est(H['sh3_G'])} | — | — | {H['lead3_F']} / {H['lead3_S']} / {H['lead3_G']} / — |")
    w(f"| F, S, G, IE, program Pell share (composition sample, point only) | {H['k5']} | {H['r2_5']:.3f} | "
      f"{H['sh5_F']:.3f} | {H['sh5_S']:.3f} | {H['sh5_G']:.3f} | {H['sh5_IE']:.3f} | {H['sh5_C']:.3f} | — |")
    w(f"| reliable fields, 4 blocks | {H['k_rel']} | — | {_est(H['sh_F_rel'])} | {_est(H['sh_S_rel'])} | "
      f"{_est(H['sh_G_rel'])} | {_est(H['sh_IE_rel'])} | — | — |")
    w("")
    w(f"- Shapley F − G: {_est(H['shFG'])}; F > G in {H['n_shF_gt_shG']} of {H['k']} fields.")
    w(f"- Unique R² (added last): F {_est(H['uniq_F'])}, G {_est(H['uniq_G'])}; F in the 3-block model "
      f"{_est(H['uniq3_F'])}.")
    for a, b in (("F", "G"), ("F", "S"), ("F", "IE"), ("G", "S")):
        c = H[f"cd_{a}_{b}"]
        w(f"- Complete dominance {HR_LAB[a]} vs {HR_LAB[b]}: {a} dominates in {c[0]}, {b} in {c[1]}, undetermined in "
          f"{c[2]} fields.")
    w(f"- Coefficient on F in the 4-block regression (HC1): positive at p < .05 in {H['F_pos_sig']} of {H['k']} "
      f"fields, negative in {H['F_neg_sig']}; on G: {H['G_pos_sig']} positive, {H['G_neg_sig']} negative.")
    w("")
    w("| field | n | R² | Shapley F / S / G / IE | largest | complete dominance F vs G | β_F (SE, HC1) | β_G (SE) |")
    w("|---|---|---|---|---|---|---|---|")
    hh = T[np.isfinite(T["hr_sh_F"]) & (T["reliable"] | T.index.isin(NAMED))].sort_values("hr_sh_F", ascending=False)
    cdlab = {1: "F", -1: "G", 0: "—"}
    for f, r in hh.iterrows():
        w(f"| {r['label']} | {int(r['hr_n'])} | {r['hr_r2']:.2f} | {r['hr_sh_F']:.3f} / {r['hr_sh_S']:.3f} / "
          f"{r['hr_sh_G']:.3f} / {r['hr_sh_IE']:.3f} | {r['hr_leader']} | {cdlab[int(r['hr_cd_F_G'])]} | "
          f"{_f(r['hr_b_F'], 2)} ({r['hr_se_F']:.2f}) | {_f(r['hr_b_G'], 2)} ({r['hr_se_G']:.2f}) |")
    w("")
    w("Institution-wide earnings include the field's own graduates, so its lead is partly mechanical; the 3-block "
      "row shows the ranking without it.")
    w("")

    # ---------------- (v) separating A from B ----------------
    w("## (v) What would separate A from B")
    w("")
    w("What each reading predicts:")
    w("")
    w("- **A (department standing carries placement information):** department prestige predicts earnings within "
      "institutions after each field's own returns to institution selectivity and brand are allowed for; the "
      "effect survives controls for the students' own entry attainment; and it moves when a department's standing "
      "changes while its institution's selectivity does not.")
    w("- **B (coupling = field-specific pricing of institutional status, plus unmeasured student quality):** the "
      "within-institution slope goes to zero once program-level student quality is controlled; any remainder is "
      "ordered across fields like selectivity pricing; changes in department standing without changes in student "
      "intake do not move earnings.")
    w("")
    w("What public US data can and cannot do:")
    w("")
    w(f"- Done here: institution-level selectivity with field-specific prices, brand with field-specific prices, "
      f"program Pell and gender composition, Pell / non-Pell outcomes, a replication on 1-year and 5-year "
      f"earnings pooled over horizons, and cohort-size weighting. They leave a small remainder (Answers 1–4): "
      f"borderline on average, with a distance from zero that depends on specification and weighting, and "
      + ("clear within institutions only in computer science; positive but individually imprecise in several "
         "engineering and business fields. " if cs_all_variants and eng_pos else
         "not concentrated in an identifiable group of fields. ")
      + (f"With linear field-specific pricing the per-field slopes differ ({wq_short}; leverage-corrected {lev(wqa)} / "
         f"{lev(wqb)}) and their cross-field ordering {repro_vb} across halves of institutions (split-half reliability "
         f"of β_f {_f(PFS['main']['rel'], 2)}; permutation p = {_p(PFS['main']['p'])}, simulated-null p = "
         f"{_p(PWB['p_sim'])}); with nonlinear pricing of brand that is no longer clear (p = {prange(ps_cg_big)}, "
         f"split-half {_f(nlg['split']['main']['rel'], 2)}). " if het_lin and not het_nl_robust else
         f"Within institutions the per-field slopes differ ({wq_short}). " if het_lin else "")
      + f"The partial-correlation design's split-half check ({_f(SH['rel_F_broadG'], 2)}, p = "
      f"{_p(SH['rel_F_broadG_p'])}) is too weak to show either way (80% power only for {p80_txt(PWF)}).")
    w(f"- Not possible: program-level entry attainment. The Scorecard FoS file has {av['fos_ncols']} columns and "
      f"{len(av['fos_test_cols'])} with SAT/ACT information; the institution file has {av['inst_test_ncols']} "
      f"SAT/ACT columns, all institution-wide. Within-institution selection into majors (separately admitted "
      f"engineering, business or nursing programs; sorting of stronger students into some majors) is therefore "
      f"unobserved, and it is exactly the confounder that could produce a small positive remainder correlated with "
      f"department prestige. The Oster bound says selection of that kind (δ* ≈ {min(dstar):.1f}–"
      f"{max(dstar):.1f} of the observed selection, depending on the reference model) would suffice.")
    w("")
    w("Data that could separate them:")
    w("")
    w(f"1. **UK LEO by prior-attainment band (public, on disk).** The DfE LEO provider-level file "
      f"(`data/raw/leo/leo_dashboard.zip`, `provider_data_20250716.csv`) releases median earnings by provider × subject "
      f"× prior-attainment band. For tax year {av['leo_tax_year']} at 5 years after graduation it has "
      f"{av['leo_pa_rows_y5_latest']:,} released band rows covering {av['leo_providers']} providers and "
      f"{av['leo_subjects']} subjects; {av['leo_band_cells_ge15']} of {av['leo_band_cells']} subject × band cells have "
      f"≥ 15 providers ({av['leo_subjects_with_ge15_band']} subjects have at least one such cell), and "
      f"{av['leo_prov_subj_ge2bands']:,} of {av['leo_prov_subj']:,} provider × subject pairs with any band release have "
      f"≥ 2 bands (band rows exist for {av['leo_prov_subj']:,} of the {av['leo_all_prov_subj']:,} provider × subject "
      f"pairs with an all-graduates release; 'Not known' prior attainment excluded, {av['leo_bands']} bands). Holding the graduates' own prior attainment fixed within subject, reading B predicts that "
      f"subject-level department prestige (e.g. the ORCID-rebuilt UK hiring network) stops predicting earnings; "
      f"reading A predicts it does not. This is the most direct public test (ROADMAP, November); it needs a UK "
      f"subject-level prestige measure, and small cells are suppressed.")
    w("2. **Changes in department standing over time (public, not assembled).** Wapman's ranks are one 2011–2020 "
      "snapshot. Department-level ratings from the National Research Council assessments (1995 and the 2010 "
      "data-based assessment) are published (availability of machine-readable tables not checked in this run); paired with PSEO graduation-cohort earnings, a within-department design "
      "with institution × cohort fixed effects would ask whether earnings follow changes in a department's standing "
      "when its institution's selectivity is held fixed. Threats: major-level selection can move with standing, and "
      "the NRC field taxonomy and cohort overlap would need checking.")
    w("3. **Program-level admission statistics where they are published.** Some state systems publish admit rates "
      "or entering GPA by major (not checked in this run). Such data would likely cover few institutions per field: "
      "enough for a case study of program-level selection, not for a within-field correlation across many "
      "institutions. A cheaper version targets the fields where the remainder sits: whether an institution admits "
      "students directly into its computer science or engineering major, or caps entry to it, can be coded from "
      "public admissions pages (not assembled here). Under B the computer science and engineering premium should sit "
      "where entry to the major is selective; under A it should not depend on how students enter the major. The same "
      "coding for business schools with separate admission would test the business remainder.")
    w("4. **Student-level records (not available under the public-data constraint).** Linked transcripts with entry "
      "test scores, major and earnings (state longitudinal data systems, restricted federal surveys) would allow "
      "comparisons of students with the same entry attainment across departments of different standing. This is "
      "the decisive design and is parked (ROADMAP §3).")
    w("")

    # ---------------- method ----------------
    w("## Method")
    w("")
    rp = ctx["repro"]
    w("- **Samples.** Matched (field, institution) cells from scripts/55 (`build_cells`): Wapman field prestige × "
      "Scorecard FoS BA median earnings (EARN_MDN_4YR) joined on UNITID to the institution file. SAT sample = SAT_AVG, "
      "ADM_RATE, PCTPELL, state level, control and state known; ADM sample drops the SAT requirement (keeps "
      "test-blind institutions); composition sample = SAT sample + program Pell share and Pell / non-Pell medians "
      f"released. A field enters a sample with n ≥ {NMIN} institutions; partials need residual df ≥ {DFMIN}.")
    w("- **Program composition.** Aggregated over the same CIP-4 rows that `load_er_scorecard` keeps (CREDLEV 3, "
      "curated CIP-4 map, earnings released): PPELL as defined in (i); PMALE = same with "
      "EARN_COUNT_MALE_WNE_4YR / EARN_COUNT_NOMALE_WNE_4YR; PPELL_D = DEBT_PELL_STGP_EVAL_N / (DEBT_PELL + "
      "DEBT_NOPELL). Pell / non-Pell medians = EARN_PELL_WNE_MDN_4YR / EARN_NOPELL_WNE_MDN_4YR via "
      "`load_er_scorecard` (scripts/55).")
    w("- **Partial correlations** (as scripts/55): ranks of prestige, earnings and continuous controls; both "
      "residualized by OLS on [1, control ranks, control dummies]; Pearson correlation of residuals.")
    w(f"- **Joint institution bootstrap** (B = {B_BOOT}): each draw resamples institutions and recomputes every "
      "field's statistics on the same draw, so cross-field means, paired differences and the horse-race summaries "
      "keep the dependence through shared institutions. Per-field SE = bootstrap SD.")
    w("- **Cross-field summaries.** Unweighted mean (joint-bootstrap SE); REML random-effects mean with bootstrap "
      "SEs as within-field variances; τ with a Q-profile 95% CI; per-field z = estimate / SE, BH-FDR at 5%; TOST at "
      f"±{EQ_BOUND}; MDE = 2.8 × SE of the mean. Empirical-Bayes shrunken estimates are in the CSV (`eb_*`) and "
      "figure; with τ near zero they collapse to the pooled mean and are not used to count fields.")
    w("- **Pooled residual slope**: within each field the two rank residuals are standardized; the pooled OLS slope "
      "equals the (n_f − 1)-weighted mean partial correlation; CR1 SE clustered by institution.")
    w(f"- **Within-institution design**: see (iii). Institution FE absorbed by within-institution demeaning (exact). "
      f"Cluster bootstrap B = {B_FE} (common β, shared draws within groups of specs on identical cells) and B = "
      f"{B_FE} for the per-field β_f of the main spec.")
    w("- **Earnings horizons**: EARN_MDN_1YR / EARN_MDN_5YR (counts EARN_COUNT_WNE_1YR / _5YR) via "
      "`load_er_scorecard`, aggregated over CIP-4 rows like the 4-year median. Per-horizon samples require the other "
      "horizon to be released, so each outcome pair is compared on identical programs; the within-institution specs "
      "put the partner horizons into the sample requirement for the same reason (asserted before the shared-draw "
      "bootstrap). Per-field β_f at the three horizons share one set of institution draws, so their differences are "
      "paired.")
    w("- **Pooled over horizons** (common programs): point = mean of the 4-, 1- and 5-year estimates; CI = percentiles "
      "(partial correlations: normal approximation and percentiles) of the same mean computed within each bootstrap "
      "draw, which keeps the dependence between horizons. Per-field pooled β_f: the same, per field; two-sided "
      "p from the bootstrap (2 × share of draws on the far side of zero) and from a normal approximation with the "
      "IQR-based bootstrap SE; BH-FDR at 5% over the fields for each. The spec-literal design (institution FE + field "
      "FE + field × SAT + field × G) and the weighted specs are estimated on the same cells and bootstrap draws as "
      "spec (a) and the main spec.")
    w("- **Cohort-size weighting**: WLS with weight = number of graduates behind the program's median at that horizon "
      "(EARN_COUNT_WNE_4YR / _1YR / _5YR summed over the CIP-4 rows, at least 1), or the 4-year count at every "
      "horizon; institution FE absorbed exactly by weighted within-institution demeaning; CR1 SE clustered by "
      "institution and the same institution cluster bootstrap.")
    w(f"- **Split-half** ({SPLIT_K} splits of the institution universe into halves; fields with n_SAT ≥ "
      f"{SPLIT_NMIN}); reliability = mean Pearson correlation across fields between the halves; permutation p from "
      f"{N_PERM} field relabellings ({N_PERM_CROSS} for the cross-fits).")
    w(f"- **Split-half of the within-institution β_f** ({SPLIT_K} splits of the institution universe into halves, the "
      f"same for every specification; seed [{SEED}, 13]): the per-field model is re-estimated on the institutions of "
      f"each half from per-institution cross-products (rows are demeaned within institution, so dropping whole "
      f"institutions keeps the FE exact); fields with ≥ {PF_SPLIT_NMIN} programs in the main-spec sample "
      f"(≥ {PF_SPLIT_NMIN2} as a sensitivity check); reliability = mean over splits of the Pearson (and Spearman) "
      f"correlation across fields between halves; p from {N_PERM} relabellings of the fields in half B.")
    w(f"- **Power of the split-half checks**: simulation described in (ii) ({POW_NSIM} data sets × {POW_NSPLIT} splits "
      f"per τ; τ grid step {POW_GRID_P[1]:.4f} for partial correlations and {POW_GRID_B[1]:.4f} for β_f; per-field "
      f"SEs = joint-bootstrap SDs for partial correlations, cluster-bootstrap SDs for β_f).")
    w(f"- **Wild cluster bootstrap** ({B_WILD} replicates; Rademacher signs by institution; seeds [{SEED}, 21–24]): "
      f"the restricted model (one common β, or no prestige slope for the null β_f = 0) is fitted on the within-"
      f"institution-demeaned data, its residuals are multiplied by the institution's sign (which keeps them demeaned) "
      f"and added to its fitted values, and the per-field model is re-estimated by the rows of pinv(X) that give the "
      f"β_f (exact OLS; the per-replicate CR1 SEs use the statsmodels factor G/(G−1)·(n−1)/(n−p), checked against "
      f"statsmodels). Across horizons the same signs are used at every horizon. Leverage-corrected variant (WCR3; "
      f"MacKinnon, Nielsen and Webb 2023, J. Applied Econometrics 38(5), 671–694, doi:10.1002/jae.2969): each "
      f"institution's restricted residuals are multiplied by (I − H_gg)⁺, H_gg = X_g (X'X)⁺ X_g' + 11'/n_g (the second "
      f"term is the absorbed institution FE), with the same signs. Check variant: the per-field model's "
      f"residuals instead of the restricted ones. Statistics: Cochran's Q with fixed weights (the observed IQR-based "
      f"or SD bootstrap SEs) or per-replicate CR1 weights; the Pearson / Spearman correlation of β_f between "
      f"horizons; the number of fields with β_f above (below) 1.96 IQR-based SE. One-sided p = (1 + #replicates ≥ "
      f"observed) / (1 + {B_WILD}).")
    w(f"- **Nonlinear field-specific pricing** (robustness of the heterogeneity): the main spec plus field × {{zG², "
      f"zG³}}, or plus field × {{zG², zG³, zSAT², zSAT³, zADM², zADM³}} (z = within-field z-score; same cells, "
      f"asserted); common β in the same cluster-bootstrap group as the main spec (paired differences); per-field β_f "
      f"with the main spec's per-field institution draws (seed [{SEED}, 5], B = {B_FE}; a field counts in a draw with "
      f"≥ its field-specific parameters + 2 distinct institutions; fields defined in < 50% of draws are left out of "
      f"the tests); REML τ and χ² Q; wild-cluster Q with the signs above; split-half with the main spec's splits and "
      f"permutations.")
    w(f"- **Where the pooled remainder sits**: groups of fields (CS/engineering; business = {names(BUS_)}; the rest) "
      f"read off the per-field results; per bootstrap draw, the weighted mean of the pooled β_f over the fields defined "
      f"in that draw (weights: equal, 1 / variance of the field's pooled draws, or program count); random-label "
      f"benchmark: {N_PERM} random sets of the same number of fields (seed [{SEED}, 25]), 'set minus the rest' on the "
      f"point estimates. Leave-out: the pooled main spec (unweighted and cohort-size weighted) refitted without the "
      f"CS/engineering fields, {B_FE} cluster-bootstrap draws (seed [{SEED}, 3, 20]).")
    w("- **Cross-horizon benchmark from the pairs bootstrap**: the per-field β_f draws of the three horizons share "
      "their institution draws; per field, the correlation of its draws between two horizons; and, per draw, the "
      "cross-field correlation between two horizons of the draws minus the point estimates (sampling error only).")
    w("- **Horse race**: all-subset OLS R² of rank(Y) on the z-scored rank blocks; Shapley = average marginal R² "
      "over block orderings; complete dominance = one block adds more R² than the other in every subset of the "
      "remaining blocks.")
    w("- **Reproduction checks (asserted in the script).** Baseline ρ_f = `outputs/expanded66_gap_map.csv` "
      f"(max |diff| {rp['baseline_maxdiff']:.1e}); against `data/interim/selectivity_fields.csv` (written at 6 "
      "significant digits): broad partial, SAT-sample raw ρ, SAT+ADM partial, ADM-sample partial, the scripts/55 "
      "3-block Shapley shares, and the within-institution β_f of specs (a) and (c) (max |diff| "
      + ", ".join(f"{v[0]:.1e}" for k, v in rp.items() if isinstance(v, tuple)) +
      f"); the pooled spec-(a) β equals scripts/55's `fe_design` pooled slope (|diff| "
      f"{rp['fe0_common_vs_s55_pooled']:.1e}).")
    w("")

    # ---------------- caveats ----------------
    w("## Caveats")
    w("")
    w("- Descriptive associations. Median earnings of federally aided completers (Scorecard covers Title IV "
      "recipients only), 4 years after the bachelor's unless stated; the elite tail is invisible.")
    w("- The horizon replication compares different graduating cohorts measured in different years (README §3), and "
      "the 5-year and 1-year medians are released for fewer programs than the 4-year one. A remainder that appears "
      "for one cohort-horizon and not for others could be cohort-specific, horizon-specific or noise; these data "
      "cannot say which. The Oster bound, the reliability-corrected β and the composition, horse-race and split-half "
      "analyses use 4-year earnings only.")
    w("- The pooled estimates average three cohort-specific estimates, so they measure the average remainder over "
      "these cohorts and horizons, not the remainder at one career stage. Weighted and unweighted estimates target "
      "different averages when the premium differs with program size; neither is the single right answer, so both "
      "are reported.")
    w("- Institution-level selectivity (recent entering class) stands in for the student quality of a program's "
      "earlier graduates. Whatever it misses stays in every partial, is correlated with prestige, and is priced more "
      "where selectivity pays more. No public US file measures program-level entry attainment.")
    w("- Brand G is built from the same faculty-hiring data pooled over all fields, so for large fields it contains "
      "part of the field's own hierarchy. Partialling G removes some genuinely field-specific variation too; the "
      "field-specific component is a conservative (low) estimate of reading A in that respect.")
    w("- The residual prestige measure is noisy (reliability bracket above). The LB bracket rests on public-edge "
      "split-half reliability, which understates the published rank's reliability; EXT extrapolates it. The "
      "correction assumes classical error independent of the controls.")
    w("- Program Pell share is released for about half of the matched programs, mostly large ones, so the "
      "composition specs cover fewer and larger fields.")
    w("- Fields share institutions; the χ² Q tests and REML τ treat fields as independent (indicative). The "
      "joint-bootstrap CIs of means and differences and the wild-cluster tests in (iii) do not make that assumption. "
      "The wild-cluster tests with plain restricted residuals are too narrow for small fields (leverage); the "
      "leverage-corrected version is the one to read. Even so, the heterogeneity of β_f holds with linear "
      "field-specific pricing and is fragile to nonlinear pricing of brand; cubic polynomials are two of many possible "
      "functional forms, so this shows fragility, not that the fields are the same.")
    w("- The 1-year, 4-year and 5-year estimates reuse the same programs, prestige measure and controls, and program-"
      "level residuals persist across cohorts, so per-field estimates at different horizons are not independent "
      "replications; their correlation is compared with a wild-cluster null in the horizon section. A field whose "
      "slope is positive at all three horizons (e.g. computer science) is one field measured three times on the same "
      "programs, not three confirmations.")
    w(f"- The wild cluster bootstrap assumes independence between institutions and uses Rademacher weights; with "
      f"{WD['B']} replicates its p cannot resolve below {pfloor:.3f}. For the cross-horizon correlation the two residual "
      f"choices give different answers (p {trip(lambda q: pw(q['WCR']['p']))} with restricted residuals, "
      f"{trip(lambda q: pw(q['WCU']['p']))} with per-field residuals, {trip(lambda q: pw(q['WCR3']['p']))} with "
      f"leverage-corrected residuals; centred pairs-bootstrap draws reach the observed r in {sh_all} of draws), so "
      f"that comparison is inconclusive rather than evidence either way; it is not used to support the heterogeneity "
      f"claim.")
    small = [f"{LAB.get(f, f)} ({int(n)})" for f, n in pf["n_by_field"].items() if n < 20]
    w(f"- Within-institution per-field β_f: fields with fewer than 20 programs ({', '.join(small)}) have few "
      f"observations relative to their six field-specific parameters; their β_f are unstable, and CR1-based "
      f"heterogeneity tests may over-reject.")
    w("- Oster's δ* is a sensitivity heuristic that assumes the unobserved selection is proportional to the observed "
      "one; it depends on R_max, which is not known for noisy program medians.")
    w("- The power simulation for the split-half checks treats fields as independent with normal errors and a "
      "half-sample variance of twice the full-sample bootstrap variance. Fields share institutions, so the real "
      "statistic may be somewhat noisier; the simulated τ = 0 95th percentile can be compared with the permutation "
      "null from the data (table in (ii)). Conversely, the field-label permutation p-values of all split-half "
      "statistics here (reliabilities and cross-fits, as in scripts/55) condition on the realized full-sample "
      "estimates and are narrower than the simulated null, so they are likely somewhat optimistic.")
    w("- The split-half check of β_f and the computer science / engineering and business contrasts are post hoc "
      "relative to the per-field results; the size of the group contrasts depends on how fields are weighted (equal, "
      f"precision, program count), and random label sets of the same size reach them in "
      f"{100 * min(labp[g_][k_] for g_ in ('eng', 'bus') for k_ in ('p_eq', 'p_pr')):.0f}–"
      f"{100 * max(labp[g_][k_] for g_ in ('eng', 'bus') for k_ in ('p_eq', 'p_pr')):.0f}% of draws before any "
      f"allowance for choosing the groups. A reproducible ordering is equally expected under A and under "
      f"program-level selection (B).")
    w("- Many specifications are reported; individual significant results among them (e.g. a few per-field β_f, the "
      "Pell/non-Pell β difference) should be read with the expected false-positive counts in mind.")
    w("")

    # ---------------- changes ----------------
    w("## Changes in this revision")
    w("")
    w(f"- **Corrected after a fourth independent check (three points).** (1) *Where the remainder sits.* The previous "
      f"headline said it is 'concentrated in computer science and engineering fields' and compared those 9 fields "
      f"(equal weights, {_f(gr['eng']['mean'], 4)}) with 'the other 33 fields' ({_f(gr['other']['mean'], 4)}). That "
      f"depends on the weighting and hides a business group of the same size: weighted by precision the 9 fields "
      f"average {gwm('eng', 'precision')}, the precisely estimated {names(eng_zero) or 'engineering fields'} being near "
      f"zero; the 5 business fields average {gwm('bus', 'equal')}; random sets of 9 labels reach the "
      f"CS/engineering contrast in {100 * labp['eng']['p_eq']:.1f}% of draws; and without the 9 fields the "
      f"cohort-size-weighted pooled β is {_f(LOO['FE2w']['beta'], 4)} {_ci(*LOO['FE2w']['ci'], 4)} (with them "
      f"{_fe_ci(AV['FE2w'])})" + (", so the weighted pooled remainder is not mainly an engineering effect"
                                  if LOO["FE2w"]["beta"] > 0.5 * AV["FE2w"]["beta"] else "") + ". The text now "
      f"says: clear in computer science; positive but not individually clear in several engineering and business "
      f"fields; near zero in {names(eng_zero) or 'some engineering fields'}; zero or below in the rest; with a "
      f"group table under three weightings. (2) *Cross-horizon agreement.* The remaining uses of 'positive at each of "
      f"the three horizons' as support (the 'Where' paragraph, Answer 7) are removed; computer science's three slopes "
      f"are close to one estimate seen three times (draw correlations {cs_dr}), and centred bootstrap draws reach "
      f"the observed cross-field correlations in {sh_all} of draws. Per-field claims rest on the pooled test "
      f"(computer science: {int(PF['n_far'][cs])} of {int(PF['n_fin'][cs])} draws ≤ 0; 98.33% CI "
      f"{_ci(PF['lo3'][cs], PF['hi3'][cs], 3)}). (3) *Heterogeneity.* The previous version rested 'the fields "
      f"differ' on a wild-cluster Q with p ≤ {pfloor:.3f} and called restricted residuals conservative. With 6 "
      f"field-specific parameters and 15–30 programs, restricted residuals are shrunk and the null is too narrow; with "
      f"leverage-corrected residuals (WCR3) Q p is {lev(wqa)} (all fields), {lev(wqm)} (≥ {PF_NMIN_MID} programs), "
      f"{lev(wqb)} (≥ {PF_SPLIT_NMIN}) and {lev(wqn)} (≥ {PF_NMIN_MID}, without CS/engineering). With nonlinear "
      f"(cubic) field-specific pricing of brand, β_f correlate {_f(nlg['corr_main_mid'], 2)} with the main spec but "
      f"the test on the well-estimated fields gives p = {prange(ps_cg_big)} and the split-half reliability falls to "
      f"{_f(nlg['split']['main']['rel'], 2)}. 'p ≤ {pfloor:.3f}' is no longer the headline evidence; the text says "
      f"heterogeneity is significant with linear pricing and not robust to nonlinear pricing, while the computer "
      f"science premium holds under linear and both nonlinear pricing variants"
      + ("" if eng_ca else " and the CS/engineering contrast's CI includes zero once SAT and admit rate are cubic too")
      + ".")
    w(f"- **Added:** leverage-corrected wild bootstrap (WCR3) next to every wild test, with the per-field width of each "
      f"null against the pairs-bootstrap SE; mid-size field sets (≥ {PF_NMIN_MID} programs, with and without "
      f"CS/engineering); the nonlinear-pricing specs (common β, per-field β_f, heterogeneity, split-half, CS and the "
      f"CS/engineering contrast; figure panel (c) and CSV columns `fe_FE2cg_*`, `fe_FE2ca_*`); group means under three "
      f"weightings with a business group, random-label benchmarks and the leave-out pooled β; cross-horizon draw "
      f"correlations and the centred-draw benchmark; CS's 98.33% CI; precision weights per field in the CSV "
      f"(`fe_FE2_h3avg_w_precision`).")
    w("- **Unchanged:** every earlier estimate. The new analyses use their own seeds or reuse existing draws (the "
      "nonlinear specs join the SAT-covariate bootstrap group, whose draws are regenerated identically for every spec; "
      "adding a residual type to the wild bootstrap does not change the signs). The group bootstrap now shares X'X "
      "between specs with the same regressors (checked: bit-for-bit the same draws as before). The CSV gains columns; "
      "the earlier columns are unchanged.")
    w("")
    w("Previous revision (cross-horizon correlation):")
    w("")
    w(f"- **Corrected after a third independent check: the correlation of β_f across earnings horizons is not evidence "
      f"of stability or against noise.** The previous version cited the correlation of the per-field slopes across the "
      f"4-, 1- and 5-year earnings (r = {trip(lambda q: _f(q['r'], 2))}) as showing that 'the pattern is stable across "
      f"cohorts', that 'the heterogeneity and its recurrence across cohorts argue against sampling noise' (also "
      f"'recurrence across cohorts argues against noise' in the horizon section), and that the computer science / "
      f"engineering remainder 'persists across all three cohort horizons'. None of this follows: the three horizons "
      f"use the same programs, prestige measure and controls, and program-level residuals persist across cohorts "
      f"(common-β residuals correlate {trip(lambda q: _f(q['rescorr_WCR'], 2))}), so the slopes correlate even when "
      f"every field has the same β. A wild cluster bootstrap with one common β and the same institution signs at "
      f"every horizon puts the null mean at {trip(lambda q: _f(q['WCR']['mean'], 2))} and gives p = "
      f"{trip(lambda q: pw(q['WCR']['p']))}; this reproduces, with different random signs, the checker's benchmark "
      f"(null mean +0.45 / +0.13 / +0.11, p = 0.063 / 0.10 / 0.073, stated in the review). Those phrases, and the "
      f"similar 'stable across cohorts' in Answer 7 and 'where it recurs across cohorts' in (v), are removed. "
      f"The claim that the fields differ now rests on wild-cluster Q tests, which keep the dependence between fields "
      f"that share institutions ({wq_short}), and on the across-institution split-half; neither reuses programs. The "
      f"headline was otherwise unchanged then: a small remainder that, within institutions, differs between fields and "
      f"sits mostly in computer science and engineering (both revised in the current revision), and that public data "
      f"cannot attribute to A or B.")
    w(f"- **Added:** wild-cluster tests ({B_WILD} replicates): Q for the heterogeneity of β_f (4-yr main spec, all "
      f"fields and the split-half fields; each horizon and the 4/1/5-yr average) with fixed and per-replicate CR1 "
      f"weights; the null benchmark for the cross-horizon correlation of β_f (Pearson and Spearman; restricted and "
      f"per-field residuals); counts of fields above zero under β_f = 0, next to the binomial tails that assume "
      f"independent fields; a Q column for the wild-cluster p in the per-horizon table; a caveat on reusing programs "
      f"across horizons.")
    w("- **Unchanged (that revision):** no previously reported estimate changed; the wild-cluster tests used their "
      "own seeds and left the existing bootstrap draws untouched. The CSV and the figure were byte-identical to the "
      "version before.")
    w("")
    w("Previous revision (differences between fields):")
    w("")
    w("- **Corrected after a second independent check: the evidence on differences between fields.** The previous "
      "version (Answer 7, Answer 4 and the summary written from it) said that the reproducible cross-field ordering "
      "of coupling comes from status the department shares with its institution's brand, that the "
      "department-specific part 'shows none', and read this as support for reading B. That rested on one "
      f"low-powered null (split-half reliability {_f(SH['rel_F_broadG'], 2)}, p = {_p(SH['rel_F_broadG_p'])}, in the "
      "partial-correlation design), while the same outputs showed significant cross-fits of that component with "
      "baseline coupling and selectivity pricing (true-score values were computed but suppressed as 'not "
      f"interpretable'), and, within institutions, heterogeneous β_f (Q p = {_p(biq['Qp'])}), "
      f"{len(ci_pos)} fields with a CI above zero against ≈{0.025 * kpf:.1f} by chance, and β_f that correlate "
      "across horizons (this last point was withdrawn in a later revision). The summary also presented that count as if it were close to chance and left out the "
      "horizon-pooled and computer science / engineering results. The text now separates the two designs: the "
      "partial-correlation check is weak (it rules out only large between-field differences), and within "
      f"institutions a small field-specific premium differs between fields and its ordering {repro_vb} across halves "
      "of institutions, mostly through computer science and engineering. 'Supports B' is no longer "
      "drawn from this evidence; A and program-level selection (B) remain indistinguishable for the remainder. "
      f"The checker's own split-half of β_f (+0.27, permutation p ≈ 0.01, stated in the review) is reproduced here "
      f"with different random splits ({_f(PFS['main']['rel'], 2)}, permutation p = {_p(PFS['main']['p'])}); against "
      f"a simulated null that also lets the full-sample estimates disperse by chance it is "
      + ("also significant" if pfs_sim else "borderline")
      + f" (p = {_p(PWB['p_sim'])}), so the text says the ordering '{repro_vb}'.")
    w("- **Added:** a split-half check of the within-institution β_f (main spec; + field × {state level, private}; "
      f"fields with ≥ {PF_SPLIT_NMIN2} programs; without computer science; without computer science and engineering; "
      "the spec-literal design and spec (a) for reference); a power simulation for the split-half checks (expected "
      "reliability by τ, τ with 80% power, τ consistent with the observed value, and a p-value against the simulated "
      "τ = 0 distribution, which is more conservative than the field-label permutation p), with the permutation-null "
      "95th percentiles and Q tests on the same fields for comparison; the true-score cross-fits of the field-specific component, flagged as imprecise; "
      "binomial tail probabilities for the counts of per-field CIs above zero; the within-institution spec with "
      "field-specific returns to the state earnings level and private control (common β and paired difference); "
      "both designs' Q p in the key numbers.")
    w("- **Unchanged (that revision):** no previously reported estimate changed. The new split-half and simulation "
      "used their own seeds, and the new within-institution spec joined an existing bootstrap group whose draws are "
      "regenerated identically for every spec. The CSV and the figure were byte-identical to the version before.")
    w("")
    w("Previous revision (horizon pooling):")
    w("")
    w("- **Headline revised after an independent check.** The previous headline said that public US data show no "
      "field-specific prestige signal robustly different from zero, that the 4-year remainder does not replicate at "
      "5 years in either design, and that it is not a stable field-specific signal. Those statements rested on "
      "counting horizons one at a time (a result had to be significant at every horizon, and a field at all three) "
      "and on weighting small, noisy program medians equally. Three noisy estimates of a small quantity can easily "
      "fail an 'every horizon' rule even when the quantity is the same at each, and the check that the per-field "
      "slopes correlate across horizons was set aside rather than tested (that correlation is benchmarked in a "
      "later revision: it is mostly what reusing the same programs produces). The headline then rested on pooled "
      "tests: a small remainder, borderline on average, with a distance from zero that depends on specification and "
      "weighting, " + ("concentrated in computer science and engineering fields (that revision added 'where it "
                       "recurs across cohorts', withdrawn two revisions later; 'concentrated' itself is revised in the "
                       "current revision)"
                       if concentrated else "not concentrated in an identifiable group of fields")
      + "; public data cannot attribute it to A or to program-level selection.")
    w("- **Added:** horizon-pooled estimates on the common programs (partial correlations; within-institution β for "
      "spec (a), the spec-literal design institution FE + field FE + field × SAT + field × G, and the main spec) with "
      "CIs from averaging the shared bootstrap draws; a per-field pooled test of β_f with BH-FDR counts under two "
      "p-value methods and the resolution limit of a 999-draw percentile p; cohort-size-weighted (WLS) versions "
      "with each horizon's own program count and with the 4-year count; the computer science / engineering versus "
      "other fields contrast (post hoc); a public-data test aimed at those fields in (v); pooled markers in figure "
      "panel (e); per-field pooled columns `fe_FE2_h3avg_*` and pooled partial columns `*_h3avg` in the CSV. The "
      "count of paired horizon differences now also covers spec (a), the spec-literal design and the weighted main "
      "spec, not only the main spec.")
    w("- **Removed:** the all-horizons rules (a common estimate had to exclude zero at every horizon; a field had to "
      "have a CI above zero at all three horizons).")
    w("- **Unchanged:** no previously reported estimate changes. The new analyses reuse the existing bootstrap draws (the "
      "spec-literal and weighted specs join the existing common-program bootstrap group, whose random draws are "
      "regenerated identically for every spec), so no existing estimate moves. Checked against the previous outputs: "
      "every previous CSV column is identical, and previous table rows are unchanged except the per-field horizon "
      "table, which gains two columns and rows for the fields flagged by the pooled test.")
    w("- Previous revision: added the replication on 1-year and 5-year earnings.")
    w("")

    # ---------------- provenance ----------------
    w("## Provenance (for SOURCES.md)")
    w("")
    w("No new raw data were downloaded for this script; it reads files already documented in `data/raw/SOURCES.md` "
      "(entries 1, 2, the College Scorecard institution-file entry, and the UK LEO dashboard entry). md5 of the "
      "files as read on the run date:")
    w("")
    w("| file | bytes | md5 | source / vintage (from SOURCES.md) |")
    w("|---|---|---|---|")
    for p, src in ctx["prov"]:
        w(f"| `{p.relative_to(ROOT)}` | {p.stat().st_size:,} | `{ctx['md5'][str(p)]}` | {src} |")
    w("")
    w("Derived inputs: `outputs/expanded66_gap_map.csv` (baseline check), `data/interim/selectivity_fields.csv` "
      "(scripts/55, reproduction checks), `data/interim/prestige_reliability.csv` (scripts/56, rel_F LB / EXT).")
    w("")
    txt = "\n".join(L) + "\n"
    txt = txt.replace("p = <1e-4", "p < 1e-4").replace("Q p = <1e-4", "Q p < 1e-4")
    OUT_MD.write_text(txt, encoding="utf-8")


PROV = [
    (ROOT / "data" / "raw" / "scorecard_fos" / "Most-Recent-Cohorts-Field-of-Study.csv",
     "College Scorecard Field-of-Study, 'Most Recent' release updated 2026-06-10 "
     "(https://ed-public-download.scorecard.network/downloads/Most-Recent-Cohorts-Field-of-Study_06102026.zip), accessed 2026-06-20"),
    (ROOT / "data" / "raw" / "scorecard_inst" / "Most-Recent-Cohorts-Institution.csv",
     "College Scorecard institution 'Most Recent Cohorts' file, build 2026-05-27 (https://collegescorecard.ed.gov/data/), accessed 2026-09-23"),
    (ROOT / "data" / "raw" / "wapman2022" / "ranks.csv",
     "Wapman et al. 2022, Zenodo record 6941651 v1 (us-faculty-hiring-networks.zip), accessed 2026-06-20"),
    (ROOT / "data" / "raw" / "leo" / "leo_dashboard.zip",
     "DfE LEO graduate outcomes, provider-level dashboard data (explore-education-statistics release a13c6267-…), accessed 2026-06-22"),
]


def main():
    ctx = compute()
    ctx = summaries(ctx)
    ctx["prov"] = PROV
    ctx["md5"] = {str(p): md5(p) for p, _ in PROV}
    write_csv(ctx)
    make_figure(ctx)
    write_md(ctx)
    S, FE = ctx["S"], ctx["FE"]
    m = S["F_broadG"]
    print(f"field-specific component (selectivity set + G): mean {m['mean']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}], "
          f"k={m['k']}, {m['n_pos']}/{m['n_neg']} fields at 1.96 SE, BH {m['n_bh_pos']}, tau {m['tau']:.3f} (Q p {m['Qp']:.2f})")
    a = S["F_all"]
    print(f"+ program Pell share: mean {a['mean']:+.3f} [{a['lo']:+.3f}, {a['hi']:+.3f}], k={a['k']}")
    r = FE["FE2"]
    print(f"within-institution beta (FE2): {r['beta']:+.4f} [{r['boot_ci'][0]:+.4f}, {r['boot_ci'][1]:+.4f}], "
          f"{r['n_cells']} programs, {r['n_inst']} institutions, {r['n_fields']} fields")
    H = ctx["HR"]
    print(f"horse race Shapley F/S/G/IE: {H['sh_F']['mean']:.3f} / {H['sh_S']['mean']:.3f} / {H['sh_G']['mean']:.3f} / "
          f"{H['sh_IE']['mean']:.3f} (k={H['k']})")
    for c in ("F_broadG_h5y4", "F_broadG_h5y5", "F_broadG_h1y4", "F_broadG_h1y1", "F_broadG_h3avg"):
        q = S[c]
        print(f"{c}: mean {q['mean']:+.3f} [{q['lo']:+.3f}, {q['hi']:+.3f}] (pct [{q['pct_lo']:+.3f}, {q['pct_hi']:+.3f}]), k={q['k']}")
    for n in ("FE2_h5y4", "FE2_h5y5", "FE2_h1y4", "FE2_h1y1", "FE2_h3y4", "FE2_h3y1", "FE2_h3y5"):
        r = FE[n]
        print(f"{n}: {r['beta']:+.4f} [{r['boot_ci'][0]:+.4f}, {r['boot_ci'][1]:+.4f}], {r['n_cells']} programs")
    for fam in HZ_FE_FAM:
        r = FE[f"{fam}_h3avg"]
        print(f"{fam}_h3avg: {r['beta']:+.4f} [{r['boot_ci'][0]:+.4f}, {r['boot_ci'][1]:+.4f}], P(<=0) {r['boot_p0']:.3f}")
    PF = FE["pf_h3avg"]
    print("pooled beta_f CI > 0:", [f for f in PF["beta_f"].index if PF["lo"][f] > 0],
          "< 0:", [f for f in PF["beta_f"].index if PF["hi"][f] < 0],
          "BH raw pct:", [f for f in PF["beta_f"].index if PF["bh"][f]],
          "BH normal:", [f for f in PF["beta_f"].index if PF["bh_n"][f]])
    for tag, q in FE["pf_split"].items():
        if isinstance(q, dict):
            print(f"beta_f split-half {tag} ({q['spec']}, k={q['k']}): rel {q['rel']:+.3f} (p {q['p']:.4f}), "
                  f"Spearman {q['rel_s']:+.3f} (p {q['p_s']:.4f}), null95 {q['null95']:+.3f}")
    for key, P in ctx["POW"].items():
        print(f"power {key}: obs {P['obs']:+.3f}, p_sim {P['p_sim']:.4f}, Q p {P['Q_p']:.4f}, crit {P['crit']:+.3f}, "
              f"tau consistent [{P['tau_lo']}, {P['tau_hi']}], "
              f"tau80 {P['tau_p80']}, " + ", ".join(f"{k}={v:+.3f}" for k, v in P.items() if k.startswith("exp_")))
    print(f"FE2sc: {FE['FE2sc']['beta']:+.4f} {FE['FE2sc']['boot_ci']}")
    WD = FE["wild"]
    for key, q in WD["hz_corr"].items():
        print(f"wild cross-horizon {key}: r {q['r']:+.3f} (rs {q['rs']:+.3f}), resid corr {q['rescorr_WCR']:+.3f}; "
              f"WCR null mean {q['WCR']['mean']:+.3f} p95 {q['WCR']['p95']:+.3f} p {q['WCR']['p']:.3f} (Spearman p "
              f"{q['WCR']['p_s']:.3f}); WCU null mean {q['WCU']['mean']:+.3f} p {q['WCU']['p']:.3f}")
    for key, q in WD["Q4"].items():
        print(f"wild Q 4-yr {key}: k={q['k']} Q {q['Q']:.1f} chi2 p {q['chi2_p']:.5f} WCR p {q['WCR']['p']:.4f} "
              f"(null mean {q['WCR']['mean']:.1f}) WCU p {q['WCU']['p']:.4f}")
    for key, q in WD["Qh"].items():
        print(f"wild Q {key}: Q {q['Q']:.1f} chi2 p {q['chi2_p']:.5f} WCR p {q['WCR']['p']:.4f} WCU p {q['WCU']['p']:.4f}")
    print("wild counts 4-yr:", WD["cnt4"])
    print("wild counts 4/1/5 avg:", WD["cnt_avg"])
    print("wild null SD / pairs IQR SE by field size:", WD["sd_ratio"])
    for key, q in WD["Q4"].items():
        print(f"wild Q 4-yr {key}: k={q['k']} WCR3 p {q['WCR3']['p']:.4f} (null mean {q['WCR3']['mean']:.1f})")
    NL = FE["nl"]
    for n in NL_SPECS:
        q = NL[n]
        h = q["het"]
        print(f"{n}: common {FE[n]['beta']:+.4f} {FE[n]['boot_ci']}, corr main (mid/all) {q['corr_main_mid']:+.3f} / "
              f"{q['corr_main_all']:+.3f}, CS {q['beta_f']['computer_science']:+.3f} [{q['lo']['computer_science']:+.3f}, "
              f"{q['hi']['computer_science']:+.3f}], eng-other mid {q['eng_mid']['diff']:+.4f} [{q['eng_mid']['lo']:+.4f}, "
              f"{q['eng_mid']['hi']:+.4f}], not estimable {q['not_est']}")
        for sn in ("all", "mid", "big", "mid_noeng"):
            print(f"   {sn}: k={h[(sn, 'iqr')]['k']} tau {h[(sn, 'iqr')]['tau']:.4f} chi2 p iqr {h[(sn, 'iqr')]['p']:.4f} "
                  f"sd {h[(sn, 'sd')]['p']:.4f}; wild p WCR {h[(sn, 'wild_iqr')]['WCR']['p']:.4f} "
                  f"WCR3 {h[(sn, 'wild_iqr')]['WCR3']['p']:.4f} WCU {h[(sn, 'wild_iqr')]['WCU']['p']:.4f}")
        print("   split:", {k: (round(v["rel"], 3), round(v["p"], 4)) for k, v in q["split"].items()})
    GW = FE["pf_h3avg"]["groups_w"]
    for k, v in GW.items():
        if isinstance(k, tuple):
            print("group", k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
    print("labels", GW["labels"])
    for k, v in FE["pf_h3avg"]["xh_draws"].items():
        print("xh draws", k, round(v["draw_r_med"], 3), round(v["draw_r"]["computer_science"], 3),
              {t: {a: round(b, 3) for a, b in v[t].items()} for t in ("all", "mid")})
    print("CS pooled 98.33% CI", FE["pf_h3avg"]["lo3"]["computer_science"], FE["pf_h3avg"]["hi3"]["computer_science"])
    print("leave-out CS/eng:", {k: v for k, v in FE["loo_eng"].items() if k != "dropped"})
    print(f"wrote {OUT_CSV.relative_to(ROOT)}, {OUT_FIG.relative_to(ROOT)}, {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
