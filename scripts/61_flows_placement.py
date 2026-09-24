"""PSEO Flows as a NON-WAGE placement axis and a cell-level mechanism test.

QUESTION. (1) Does placement into market-priced sectors couple with academic prestige, academia-wide
brand and selectivity, within field families? (2) Does the pay-for-status gradient (prestige/brand/
selectivity -> graduate earnings) shrink where more of a program's graduates work in sectors whose
pay is set by the setting (public budgets, pay scales, reimbursement)?

DATA (public only; nothing restricted, no resume/profile data)
  * PSEO Flows `data/raw/pseo/pseof_all.csv.gz` (Census LEHD, release V4.13.0 / 2025Q4): counts of
    employed bachelor's graduates by institution x CIP-2 x NAICS sector (agg_level_pseo 88 pooled
    cohorts, 94 by 3-year graduation cohort) and the all-sector totals incl. non/marginally employed
    (agg 40 / 46). Flows are released only at CIP-2 (no CIP-4). A graduate's sector is that of the
    main (highest-earning) job in the reference year; graduates below the PSEO attachment threshold
    are "non- or marginally employed" (PSEO technical documentation).
  * PSEO Earnings `data/raw/pseo/pseoe_all.csv.gz` (same release): bachelor's median (p50) earnings
    for the SAME institution x CIP-2 x cohort x horizon cells (agg 40 / 46), and IPEDS completion
    counts per institution x CIP-4 (agg 42) used as weights.
  * Wapman et al. field ranks (src/load_ar.py, FIELDS66 via scripts/28) and the academia-wide rank
    (scripts/28 load_generic); College Scorecard institution file (SAT_AVG, ADM_RATE, PCTPELL, CONTROL,
    MD_EARN_WNE_P10) joined on the 8-digit OPEID that PSEO uses as its institution id.
  * ACS PUMS 2023 person files (data/raw/acs) only to document the ex-ante sector classes.

SECTOR CLASSES (fixed before looking at any coupling result)
  setting-priced  S = NAICS 61 educational services + 62 health care & social assistance
                      + 92 public administration: employers are mostly governments or non-profits,
                      pay follows salary schedules, public budgets or reimbursement rates.
  market-priced   M = NAICS 51 information + 52 finance & insurance + 54 professional, scientific &
                      technical services: private, high-pay knowledge services with individually set pay
                      (the elite-recruiting sectors: banks, consulting, software, law/accounting firms).
  M+ (sensitivity) = M + 55 management of companies (corporate headquarters).
  Everything else (manufacturing, retail, construction, ...) is neither.
  L (post hoc, added in the second revision after an independent verifier used it) = low-pay private
      sectors 44-45 retail, 56 admin & support, 71 arts & recreation, 72 accommodation & food, 81 other
      services; L2 (sensitivity) = 44-45, 56, 72 only (71 and 81 have large non-profit shares in the ACS). The ACS table in the report
  documents the ownership split (government / non-profit / for-profit) and the median wage of young
  bachelor's-degree workers in each class.

CELLS AND FAMILIES
  cell = institution x CIP-2 family at horizon h (pooled cohorts "0000"; y5 is the primary horizon;
  y1, y10 and the fixed 2010-2012 cohort are robustness). Employed-graduate total >= EMP_MIN and a
  released PSEO p50 are required. Project families = CIP-2 families holding FIELDS66 fields whose
  CIP-4 codes cover >= FAM_COVER of the family's PSEO bachelor's graduates nationally (drops CIP-30
  multi/interdisciplinary and CIP-39 theology).
  family prestige P_{i,c} = IPEDS-count-weighted mean, over the FIELDS66 fields in family c ranked at
  institution i, of the within-field Wapman percentile 1 - Rank/max(Rank) (weights: PSEO pooled
  y1_ipeds_count of the field's CIP-4 codes at i). P_u = unweighted mean (robustness).
  coverage_{i,c} = ranked-field graduates / all graduates of the CIP-2 cell.
  brand G = -academia-wide Wapman rank; selectivity SAT_AVG and -ADM_RATE (Scorecard).

ANALYSES
  (a) sector shares per cell; national sector mix; ACS documentation of the classes; test-retest
      reliability of the shares across graduation cohorts (2013 vs 2016 at y1).
  (b) placement coupling within family: Spearman of P / G / SAT / -ADM with the market share M (and
      the setting share S), next to wage coupling Spearman(., log p50) on the same cells; joint
      institution bootstrap (B_BOOT) for cross-family means; partial versions net of SAT, -ADM, Pell,
      control and state earnings level (the "broad" spec of scripts/55); split-half cross-fit for the
      cross-family ordering (placement vs wage coupling); within-institution design (institution FE).
  (c) mechanism test at cell level:
        log p50_{ic} = gamma_c + beta_c z_c(status) + delta_c e_{ic} + kappa_c e_{ic}^2 + theta z_c(status) x e_{ic} + u
      e = setting-priced share in 10-percentage-point units, centred within family; z_c = within-family
      standardisation; family-specific slopes on status and exposure and family-specific curvature in
      exposure, so theta is identified only from variation within families. The kappa_c e^2 term was added
      after independent verification (log p50 is curved in e and z is correlated with e, so with a linear
      exposure main effect z*e partly stands in for e^2); every row is also fitted with a linear e, and a
      functional-form battery (FF_SPECS) is run on all horizons / cohorts and on V4.14.1. SEs clustered by
      institution (CR1); institution pairs-cluster bootstrap and cluster jackknife; random-intercept
      (institution) mixed model; many robustness variants; median split with and without S partialled.
  (c') joint exposure models (second revision, after independent verification): the setting share S and
      the market share M are strongly negatively related within families, so a flatter status gradient
      where S is high may just be a steeper one where M is high. Models with z*e_S AND z*e_M (and the
      low-pay private share L = NAICS 44-45, 56, 71, 72, 81), every exposure with its own family-specific
      main-effect shape, under 10 functional forms (JFF_SPECS) on every horizon / cohort sample and on
      V4.14.1; robustness rows (state / institution FE, selectivity controls, WS cells, WLS, ...); Wald
      tests (theta_S = theta_L; theta_S = theta_L = 0); a Gelbach decomposition of the S-only theta; and a
      shape-free conditional median split (S split within M halves, M split within S halves).
  (c'') timing of the exposure (third revision, after independent verification): the joint models of (c')
      measure the shares at the same horizon as the outcome (y5), but y5 shares are partly an outcome of
      status (sorting into market sectors grows with career time). The same y5 outcome cells are refitted
      with the exposure measured at year 1: the year-1 sector shares of the same institution x family (pooled
      cohorts, and the 2010-12, 2013-15 and 2016-18 graduation cohorts; y1 cell >= EMP_MIN employed), next
      to the y5 shares on exactly the same subsamples; full functional-form batteries, robustness rows
      (linear / cubic / status^2 / state FE / institution FE / selectivity x share controls / WS / WLS /
      rank outcome / y5 shares also in the model / L ...), wild-cluster restricted bootstrap (WCR) p-values,
      an ex-ante sector-wage-mix index C = log sum_k share_k x ACS young-graduate median wage of sector k,
      implied slopes, a Gelbach decomposition and the conditional median split on y1 shares.
      Fourth revision (after independent verification): the third revision wrongly stated that the 2013-15 and
      2016-18 cohorts do not reach y5 in V4.13.0. They do (flows agg 46 and earnings y5 are released for them),
      and the pooled y5 cell sums every cohort through 2016-18, so the pooled y1 cell holds the same graduates
      plus the 2019-21 cohort. Added: each cohort's own y1 shares -> own y5 pay for 2013-15 and 2016-18 (the
      2010-12 one already existed), the pooled y1 shares with the 2019-21 cohort removed (pooled minus that
      cohort's released counts), and the cohort composition of the pooled cells.
  (d) selectivity-adjusted versions: family-specific slopes on SAT, -ADM, Pell, control and state
      earnings level and their interactions with e.
  (e) release check: the primary pooled-y5 statistics re-estimated on PSEO V4.14.1 (2026Q2; flows in
      data/raw/pseo_flows_2026q2/, earnings + institutions in data/raw/pseo_2026q2/).

Descriptive, not causal: across-institution differences mix value-added and selection, and the
sector mix of a CIP-2 cell also reflects its CIP-4 program mix and its state's labour market.
Seeded; outputs are byte-identical on re-run.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/61_flows_placement.py
Outputs: data/interim/flows_placement.csv, outputs/figures/flows_placement.png,
         FLOWS_PLACEMENT_RESULT.md (repo root, local only).
"""
from __future__ import annotations

import os
import sys
import zlib
import hashlib
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pcm
from scipy.stats import rankdata, norm, chi2
import scipy.sparse as sps
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pa.set_cpu_count(2)            # shared VM: cap arrow's CSV-parsing threads
pa.set_io_thread_count(2)

from src.load_ar import load_ar_wapman
from src.crosswalks import fields as F
from src.crosswalks.institutions import normalize_institution_name

_spec = importlib.util.spec_from_file_location("s28", ROOT / "scripts" / "28_field_vs_generic_prestige.py")
_s28 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s28)
FIELDS66, LAB, C2NAME = _s28.FIELDS66, _s28.LAB, _s28.C2NAME

SEED = 61
EMP_MIN = 50           # employed graduates in a cell (primary); 100 as robustness
NMIN = 15              # cells per family for a within-family statistic
NMIN_HALF = 8          # cells per family per half in the split-half cross-fit
DFMIN = 10             # residual df for a partial correlation
FAM_COVER = 0.25       # national ranked-field share needed for a project family
B_BOOT = 1000          # joint institution bootstrap draws (family statistics)
B_CLU = 999            # pairs-cluster bootstrap draws (interaction coefficients)
K_SPLIT = 200          # random half-splits of institutions (cross-family ordering)
N_PERM = 1000          # family-label permutations
B_WCR = 999            # wild-cluster restricted bootstrap draws (Rademacher weights by institution)
if os.environ.get("FLOWS61_QUICK"):   # development only: small resampling counts (outputs not for use)
    B_BOOT, B_CLU, K_SPLIT, N_PERM, B_WCR = 40, 39, 6, 40, 39
H_MAIN = "y5"
HZ = ["y1", "y5", "y10"]
COHORTS = ["0000", "2010", "2013", "2016", "2019"]   # 2019 (-21): only to remove it from the pooled y1 cell
EXCL_COHORT = "2019"   # fourth revision: pooled y1 minus the 2019-21 cohort (the only cohort not in pooled y5)
SETTING = ["61", "62", "92"]
MARKET = ["51", "52", "54"]
MARKET_PLUS = MARKET + ["55"]
LOWPAY = ["44-45", "56", "71", "72", "81"]      # post hoc (second revision): low-pay private sectors
LOWPAY_FP = ["44-45", "56", "72"]               # sensitivity: without 71 and 81 (large non-profit shares)
SECTOR_NAME = {"11": "Agriculture", "21": "Mining", "22": "Utilities", "23": "Construction",
               "31-33": "Manufacturing", "42": "Wholesale", "44-45": "Retail", "48-49": "Transport",
               "51": "Information", "52": "Finance & insurance", "53": "Real estate",
               "54": "Professional/scientific/technical", "55": "Management of companies",
               "56": "Admin & support", "61": "Educational services", "62": "Health care & social asst.",
               "71": "Arts & recreation", "72": "Accommodation & food", "81": "Other services",
               "92": "Public administration"}
SECTORS = list(SECTOR_NAME)
SEL_CONT = ["SAT", "NEG_ADM", "PELL", "ST_EARN"]     # broad selectivity controls (scripts/55 full_stlev)
SEL_CAT = ["CONTROL"]
# Shape of the exposure main effect in the interaction model. Log p50 is curved in the setting share and
# status is correlated with it within families, so with a linear exposure main effect the product z*e can
# stand in for the omitted e^2 (spurious interaction). PRIMARY = family-specific e^2 (revision after
# independent verification; the first version used 'lin'). FF_SPECS = functional-form battery.
ECURVE_MAIN = "e2"
FF_SPECS = [("lin", "linear e only (first-version primary)", dict(ecurve="lin")),
            ("e2c", "+ common e²", dict(ecurve="e2c")),
            ("e2", "+ family-specific e² (primary)", dict(ecurve="e2")),
            ("e3", "+ family-specific e², e³", dict(ecurve="e3")),
            ("eq5", "+ family × e-quintile dummies", dict(ecurve="eq5")),
            ("eq5zq5", "+ family × e-quintile and family × status-quintile dummies", dict(ecurve="eq5", zq5=True)),
            ("e2z2", "+ family-specific e² and status²", dict(ecurve="e2", quad=True)),
            ("rr_lin", "rank × rank (within-family standardised ranks of status and S), linear", dict(ecurve="lin", rank_x=True)),
            ("rr", "rank × rank, + family-specific (rank S)²", dict(ecurve="e2", rank_x=True))]
N_FF = len(FF_SPECS)
N_FF_NR = sum(not kw.get("rank_x") for _, _, kw in FF_SPECS)   # non-rank forms
FF_SAMPLES = [("pooled_y5", "y5 pooled (main)"), ("pooled_y1", "y1 pooled"), ("pooled_y10", "y10 pooled"),
              ("c2010_y5", "cohort 2010-12, y5"), ("c2010_y10", "cohort 2010-12, y10"), ("v4141", "V4.14.1, y5 pooled")]
# Joint exposure models (second revision). Every exposure gets the same family-specific main-effect shape;
# 'e2x' adds family-specific cross products of the exposures (with one exposure it equals 'e2').
JSETS = [("SM", ("M",), "S + M"), ("SL", ("L",), "S + L"), ("SML", ("M", "L"), "S + M + L")]
JFF_SPECS = FF_SPECS + [("e2x", "+ family-specific e² of each exposure and their cross products", dict(ecurve="e2x"))]
N_JFF = len(JFF_SPECS)
# Timing of the exposure (third revision; own-cohort designs for 2013-15 and 2016-18 added in the fourth): the y5
# outcome cells with the sector shares of the same institution x family measured at year 1. key, label, source of
# the shares (y1 cell table), base = the outcome cell table. Three designs:
#   pooled: pooled-cohort y1 shares -> pooled-cohort y5 pay (same cohorts 2001-18, plus 2019-21 at y1);
#   own:    one graduation cohort's own y1 shares -> the same cohort's own y5 pay;
#   mixed:  one cohort's y1 shares -> the pooled-cohort y5 pay (that cohort is one of several in the outcome).
X_SAMPLES = [("x_pooled_y1", "pooled cohorts: y1 shares, y5 pay", "pooled_y1", "pooled_y5"),
             ("x_c2010_same", "2010–12 cohort: own y1 shares, own y5 pay", "c2010_y1", "c2010_y5"),
             ("x_c2013_same", "2013–15 cohort: own y1 shares, own y5 pay", "c2013_y1", "c2013_y5"),
             ("x_c2016_same", "2016–18 cohort: own y1 shares, own y5 pay", "c2016_y1", "c2016_y5"),
             ("x_c2010_y1", "2010–12 cohort y1 shares, pooled y5 pay", "c2010_y1", "pooled_y5"),
             ("x_c2013_y1", "2013–15 cohort y1 shares, pooled y5 pay", "c2013_y1", "pooled_y5"),
             ("x_c2016_y1", "2016–18 cohort y1 shares, pooled y5 pay", "c2016_y1", "pooled_y5")]
X_MAIN = "x_pooled_y1"
X_OWN = [k for k, _, _, b in X_SAMPLES if b != "pooled_y5"]
X_MIXED = [k for k, _, _, b in X_SAMPLES if k != X_MAIN and b == "pooled_y5"]
# Fourth-revision checks (not counted with the timing samples): pooled y1 shares with the 2019-21 cohort removed.
# 'strict': cells whose 2019-21 y1 count is suppressed are dropped; 'all': a suppressed 2019-21 count is taken as 0.
X_CHECKS = [("x_p1no19", "pooled cohorts without 2019–21: y1 shares, y5 pay (strict)", "pooled_y1_no19", "pooled_y5"),
            ("x_p1no19all", "pooled cohorts without 2019–21: y1 shares, y5 pay (suppressed 2019–21 counts as 0)",
             "pooled_y1_no19all", "pooled_y5")]
X_ALL = X_SAMPLES + X_CHECKS
SHARE_COLS = ["S", "M", "Mplus", "L", "L2", "C"] + [f"sh_{k}" for k in SECTORS]

PSEOF = ROOT / "data" / "raw" / "pseo" / "pseof_all.csv.gz"
PSEOE = ROOT / "data" / "raw" / "pseo" / "pseoe_all.csv.gz"
PSEO_INST = ROOT / "data" / "raw" / "pseo" / "pseo_all_institutions.csv"
PSEO_VER = ROOT / "data" / "raw" / "pseo" / "version_pseo.txt"
# newer release V4.14.1 (2026Q2): earnings + institutions downloaded for scripts/59 (data/raw/pseo_2026q2/),
# flows downloaded for this script (data/raw/pseo_flows_2026q2/). Release sensitivity only.
PSEOF_NEW = ROOT / "data" / "raw" / "pseo_flows_2026q2" / "pseof_all.csv.gz"
PSEOF_NEW_VER = ROOT / "data" / "raw" / "pseo_flows_2026q2" / "version_pseo.txt"
PSEOE_NEW = ROOT / "data" / "raw" / "pseo_2026q2" / "pseoe_all.csv.gz"
PSEO_INST_NEW = ROOT / "data" / "raw" / "pseo_2026q2" / "pseo_all_institutions.csv"
PSEO_VER_NEW = ROOT / "data" / "raw" / "pseo_2026q2" / "version_pseo.txt"
SC_INST = ROOT / "data" / "raw" / "scorecard_inst" / "Most-Recent-Cohorts-Institution.csv"
ACS = [ROOT / "data" / "raw" / "acs" / "psam_pusa.csv", ROOT / "data" / "raw" / "acs" / "psam_pusb.csv"]
WAPMAN = ROOT / "data" / "raw" / "wapman2022" / "ranks.csv"
OUT_CSV = ROOT / "data" / "interim" / "flows_placement.csv"
OUT_FIG = ROOT / "outputs" / "figures" / "flows_placement.png"
OUT_MD = ROOT / "FLOWS_PLACEMENT_RESULT.md"

warnings.filterwarnings("ignore", category=RuntimeWarning)


def rng_for(tag: str) -> np.random.Generator:
    """Independent, order-free stream per analysis (deterministic)."""
    return np.random.default_rng([SEED, zlib.crc32(tag.encode())])


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


# =============================================================================================
# data
# =============================================================================================
def read_flows(path: Path = PSEOF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bachelor's institution x CIP-2 rows of PSEO Flows. Returns (sector rows, total rows);
    counts are NaN unless status == 1. Streams the ~42M-row file in 8 MB blocks with typed columns
    and keeps only agg 40/46/88/94 (all national geography, see label_agg_level_pseo), degree 05 and
    the analysis cohorts, so the peak memory stays small. The all-sector totals by cohort (agg 46) are kept for
    every cohort (fourth revision: cohort composition of the pooled cells); a row with a NaN count is a released
    row whose status is not 1 (e.g. suppressed), a missing row means the cohort has no cell."""
    scols = ["agg_level_pseo", "institution", "degree_level", "cipcode", "grad_cohort", "industry"]
    vcols = [f"{h}_grads_{k}" for k in ("emp", "nme") for h in HZ]
    fcols = [f"status_{c}" for c in vcols]
    types = {**{c: pa.string() for c in scols}, **{c: pa.float64() for c in vcols},
             **{c: pa.int8() for c in fcols}}
    co = pacsv.ConvertOptions(include_columns=scols + vcols + fcols, column_types=types)
    rd = pacsv.open_csv(path, convert_options=co, read_options=pacsv.ReadOptions(block_size=1 << 23, use_threads=False))
    aggs, cohs, keep = pa.array(["40", "46", "88", "94"]), pa.array(COHORTS), []
    for b in rd:
        t = pa.Table.from_batches([b])
        m = pcm.and_(pcm.and_(pcm.is_in(t.column("agg_level_pseo"), value_set=aggs),
                              pcm.equal(t.column("degree_level"), "05")),
                     pcm.or_(pcm.is_in(t.column("grad_cohort"), value_set=cohs),
                             pcm.equal(t.column("agg_level_pseo"), "46")))
        t = t.filter(m)
        if t.num_rows == 0:
            continue
        arrs = {c: t.column(c) for c in ["agg_level_pseo", "institution", "cipcode", "grad_cohort", "industry"]}
        for h in HZ:
            for k in ("emp", "nme"):
                v, f = t.column(f"{h}_grads_{k}"), t.column(f"status_{h}_grads_{k}")
                ok = pcm.fill_null(pcm.equal(f, 1), False)
                arrs[f"{k}_{h}"] = pcm.if_else(ok, v, pa.scalar(None, pa.float64()))
        keep.append(pa.table(arrs))
        del t, b
    df = pa.concat_tables(keep).to_pandas()
    del keep
    df["cip2"] = df["cipcode"].str.replace(".", "", regex=False).str.zfill(2)
    df = df.rename(columns={"grad_cohort": "cohort"})
    keepc = ["institution", "cip2", "cohort", "industry"] + [f"emp_{h}" for h in HZ] + [f"nme_{h}" for h in HZ]
    sec = df[df.agg_level_pseo.isin(["88", "94"])][keepc].reset_index(drop=True)
    tot = df[df.agg_level_pseo.isin(["40", "46"])][keepc].drop(columns="industry").reset_index(drop=True)
    return sec, tot


def read_earnings(path: Path = PSEOE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(CIP-2 cell earnings by cohort/horizon, CIP-4 IPEDS counts, CIP-2 IPEDS counts); bachelor's.
    Read in chunks with usecols; filtered to institution-level bachelor's rows early."""
    use = ["agg_level_pseo", "inst_level", "institution", "degree_level", "cipcode", "grad_cohort",
           "y1_ipeds_count"]
    use += [f"{h}_p50_earnings" for h in HZ] + [f"{h}_grads_earn" for h in HZ] + [f"status_{h}_earnings" for h in HZ]
    parts = []
    for ch in pd.read_csv(path, dtype=str, usecols=use, chunksize=100_000):
        ch = ch[(ch.inst_level == "I") & (ch.degree_level == "05")]
        ch = ch[(ch.agg_level_pseo.isin(["40", "46"]) & ch.grad_cohort.isin(COHORTS)) |
                (ch.agg_level_pseo.isin(["40", "42"]) & (ch.grad_cohort == "0000"))]
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    del parts
    e2 = df[df.agg_level_pseo.isin(["40", "46"]) & df.grad_cohort.isin(COHORTS)].copy()
    e2["cip2"] = e2.cipcode.str.replace(".", "", regex=False).str.zfill(2)
    out = e2[["institution", "cip2", "grad_cohort"]].rename(columns={"grad_cohort": "cohort"})
    for h in HZ:
        ok = e2[f"status_{h}_earnings"] == "1"
        out[f"p50_{h}"] = pd.to_numeric(e2[f"{h}_p50_earnings"], errors="coerce").where(ok)
        out[f"nearn_{h}"] = pd.to_numeric(e2[f"{h}_grads_earn"], errors="coerce").where(ok)
    pooled = df[df.grad_cohort == "0000"]
    c4 = pooled[pooled.agg_level_pseo == "42"].copy()
    c4["cip4"] = c4.cipcode.str.replace(".", "", regex=False).str.zfill(4)
    c4["n_grad"] = pd.to_numeric(c4.y1_ipeds_count, errors="coerce")
    c2 = pooled[pooled.agg_level_pseo == "40"].copy()
    c2["cip2"] = c2.cipcode.str.replace(".", "", regex=False).str.zfill(2)
    c2["n_grad_c2"] = pd.to_numeric(c2.y1_ipeds_count, errors="coerce")
    return (out.reset_index(drop=True), c4[["institution", "cip4", "n_grad"]].reset_index(drop=True),
            c2[["institution", "cip2", "n_grad_c2"]].reset_index(drop=True))


def pseo_institutions(path: Path = PSEO_INST) -> pd.DataFrame:
    ins = pd.read_csv(path, dtype=str)
    ins.columns = [c.strip().lstrip("\ufeff") for c in ins.columns]
    ins["inst_key"] = ins["label"].map(normalize_institution_name)
    return ins[["institution", "label", "institution_state", "inst_key"]].drop_duplicates("institution")


def scorecard_institutions() -> pd.DataFrame:
    """Scorecard institution covariates keyed by 8-digit OPEID (= PSEO institution id). Several
    UNITIDs can share an OPEID: keep the main campus, then the largest undergraduate enrolment.
    ST_EARN = leave-one-out mean log MD_EARN_WNE_P10 over predominantly-bachelor's institutions of the
    same state (the 'state earnings level' of scripts/55, recomputed here the same way)."""
    cols = ["UNITID", "OPEID", "INSTNM", "STABBR", "MAIN", "UGDS", "CONTROL", "PREDDEG", "ADM_RATE",
            "SAT_AVG", "PCTPELL", "MD_EARN_WNE_P10"]
    d = pd.read_csv(SC_INST, usecols=cols, dtype=str)
    for c in ["MAIN", "UGDS", "CONTROL", "PREDDEG", "ADM_RATE", "SAT_AVG", "PCTPELL", "MD_EARN_WNE_P10"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["_le"] = np.where((d.PREDDEG == 3) & (d.MD_EARN_WNE_P10 > 0), np.log(d.MD_EARN_WNE_P10), np.nan)
    d["_ok"] = d["_le"].notna().astype(float)
    s = d.groupby("STABBR")["_le"].transform(lambda x: x.fillna(0).sum())
    k = d.groupby("STABBR")["_ok"].transform("sum")
    denom = k - d["_ok"]
    d["ST_EARN"] = np.where(denom > 0, (s - d["_le"].fillna(0.0)) / denom.where(denom > 0, 1), np.nan)
    d = d.sort_values(["OPEID", "MAIN", "UGDS", "UNITID"], ascending=[True, False, False, True])
    d = d.drop_duplicates("OPEID")
    d = d.rename(columns={"OPEID": "institution", "SAT_AVG": "SAT", "PCTPELL": "PELL"})
    d["NEG_ADM"] = -d["ADM_RATE"]
    return d[["institution", "UNITID", "INSTNM", "STABBR", "CONTROL", "SAT", "ADM_RATE", "NEG_ADM",
              "PELL", "ST_EARN"]]


def family_prestige(c4: pd.DataFrame, c2n: pd.DataFrame, ins: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per (institution, cip2): weighted / unweighted mean within-field Wapman percentile over ranked
    FIELDS66 fields, ranked-field coverage, number of ranked fields. Also the national ranked share."""
    cmap = F.cip4_to_field_key(FIELDS66)                       # cip4 -> field (resolved as elsewhere)
    ar = load_ar_wapman(fields=FIELDS66)                       # inst_key, field, prestige_score = -Rank
    ar["maxrank"] = ar.groupby("field").prestige_score.transform(lambda s: -s.min())
    ar["pct"] = 1.0 + ar.prestige_score / ar.maxrank           # 1 = top, 0 = bottom of the field
    fam_fields = {}
    for c4c, f in cmap.items():
        fam_fields.setdefault(c4c[:2], set()).add(f)
    x = c4.merge(ins[["institution", "inst_key"]], on="institution", how="left")
    x["field"] = x.cip4.map(cmap)
    x["cip2"] = x.cip4.str[:2]
    x = x.merge(ar[["inst_key", "field", "pct"]], on=["inst_key", "field"], how="left")
    # national ranked-field share of each family (all PSEO institutions; ranked = in FIELDS66 map)
    nat = (x[x.field.notna()].groupby("cip2").n_grad.sum() / x.groupby("cip2").n_grad.sum())
    # weighted prestige: ranked fields present in PSEO with a positive IPEDS count
    r = x[x.pct.notna()].copy()
    w = r.groupby(["institution", "cip2", "field"]).agg(n=("n_grad", "sum"), pct=("pct", "first")).reset_index()
    w["n"] = w["n"].fillna(0.0)
    wp = w[w.n > 0].assign(wp=lambda t: t.n * t.pct).groupby(["institution", "cip2"]).agg(
        wp=("wp", "sum"), n_rank=("n", "sum"), k_w=("field", "size")).reset_index()
    wp["P"] = wp.wp / wp.n_rank
    # unweighted: every ranked FIELDS66 field of the family at the institution (Wapman only)
    rows = []
    ins_k = ins[["institution", "inst_key"]]
    arf = ar[["inst_key", "field", "pct"]]
    for c2c, flds in sorted(fam_fields.items()):
        a = arf[arf.field.isin(flds)].groupby("inst_key").agg(P_u=("pct", "mean"), k_u=("field", "size"))
        a = a.reset_index().merge(ins_k, on="inst_key")
        a["cip2"] = c2c
        rows.append(a)
    un = pd.concat(rows, ignore_index=True)[["institution", "cip2", "P_u", "k_u"]]
    out = un.merge(wp[["institution", "cip2", "P", "n_rank", "k_w"]], on=["institution", "cip2"], how="outer")
    out = out.merge(c2n, on=["institution", "cip2"], how="left")
    out["coverage"] = out.n_rank.fillna(0.0) / out.n_grad_c2.where(out.n_grad_c2 > 0)
    return out.drop(columns=["n_grad_c2"]), nat.to_dict()


def generic_brand() -> pd.DataFrame:
    g = _s28.load_generic()                                     # inst_key, g_rank (academia-wide)
    g["G"] = -g["g_rank"].astype(float)
    return g[["inst_key", "G"]]


def build_cells(sec, tot, earn, prest, ins, sc, brand, cohort: str, h: str, fams: list[str]) -> pd.DataFrame:
    """One row per institution x CIP-2 (cohort, horizon) with sector shares, earnings and covariates."""
    s = sec[(sec.cohort == cohort) & sec[f"emp_{h}"].notna()]
    wide = s.pivot_table(index=["institution", "cip2"], columns="industry", values=f"emp_{h}",
                         aggfunc="sum").reindex(columns=SECTORS).fillna(0.0)
    emp = wide.sum(axis=1)
    cells = pd.DataFrame({"emp": emp})
    for k in SECTORS:
        cells[f"sh_{k}"] = wide[k] / emp.where(emp > 0)
    cells["S"] = wide[SETTING].sum(axis=1) / emp.where(emp > 0)
    cells["M"] = wide[MARKET].sum(axis=1) / emp.where(emp > 0)
    cells["Mplus"] = wide[MARKET_PLUS].sum(axis=1) / emp.where(emp > 0)
    cells["L"] = wide[LOWPAY].sum(axis=1) / emp.where(emp > 0)
    cells["L2"] = wide[LOWPAY_FP].sum(axis=1) / emp.where(emp > 0)
    cells = cells.reset_index()
    t = tot[tot.cohort == cohort][["institution", "cip2", f"emp_{h}", f"nme_{h}"]]
    cells = cells.merge(t.rename(columns={f"emp_{h}": "emp_tot", f"nme_{h}": "nme"}),
                        on=["institution", "cip2"], how="left")
    cells["nme_share"] = cells.nme / (cells.emp_tot + cells.nme)
    e = earn[earn.cohort == cohort][["institution", "cip2", f"p50_{h}", f"nearn_{h}"]]
    cells = cells.merge(e.rename(columns={f"p50_{h}": "p50", f"nearn_{h}": "n_earn"}),
                        on=["institution", "cip2"], how="left")
    cells = cells.merge(ins, on="institution", how="left")
    cells = cells.merge(prest, on=["institution", "cip2"], how="left")
    cells = cells.merge(brand, on="inst_key", how="left")
    cells = cells.merge(sc, on="institution", how="left")
    cells["log_earn"] = np.log(cells.p50.where(cells.p50 > 0))
    cells["cohort"], cells["horizon"] = cohort, h
    cells["project_family"] = cells.cip2.isin(fams)
    cells["family"] = cells.cip2.map(lambda c: C2NAME.get(c, f"CIP-{c}"))
    return cells.sort_values(["cip2", "institution"]).reset_index(drop=True)


def sec_pooled_minus(sec: pd.DataFrame, tot: pd.DataFrame, h: str, excl: str = EXCL_COHORT,
                     strict: bool = True) -> tuple[pd.DataFrame, dict]:
    """Fourth revision: sector rows of a pseudo-cohort '0000-<excl>' = pooled-cohort counts minus the released counts
    of cohort `excl` (the 2019-21 cohort, the only one in the pooled y1 cell that does not reach y5) at horizon h,
    clipped at 0 (both are noise-protected). The cohort's cell is released (subtract its sector counts), absent (no
    row: nothing to subtract) or present but not released (suppressed). strict: suppressed cells are dropped;
    otherwise their count is taken as 0. Returns (sector rows, counts of the three cases among pooled cells)."""
    v = f"emp_{h}"
    p = sec[(sec.cohort == "0000") & sec[v].notna()][["institution", "cip2", "industry", v]]
    x = sec[(sec.cohort == excl) & sec[v].notna()][["institution", "cip2", "industry", v]].rename(columns={v: "x"})
    t = tot[tot.cohort == excl][["institution", "cip2", v]]
    key = p.institution + "|" + p.cip2
    k_rel = set((t.institution + "|" + t.cip2)[t[v].notna()])
    k_sup = set((t.institution + "|" + t.cip2)[t[v].isna()])
    cells = set(key)
    n = dict(released=len(cells & k_rel), absent=len(cells - k_rel - k_sup), suppressed=len(cells & k_sup))
    j = p.merge(x, on=["institution", "cip2", "industry"], how="left")
    j[v] = np.clip(j[v] - j["x"].fillna(0.0), 0.0, None)
    if strict:
        j = j[~(j.institution + "|" + j.cip2).isin(k_sup)]
    j = j.drop(columns="x").assign(cohort=f"0000-{excl}")
    return j.reset_index(drop=True), n


def acs_sectors() -> pd.DataFrame:
    """ACS 2023 PUMS: bachelor's+ wage/salary employees aged 22-40, full-time full-year, by NAICS
    sector: government share, government + non-profit share, weighted median wage."""
    num = ["PWGTP", "AGEP", "COW", "SCHL", "WAGP", "WKHP", "WKWN"]
    types = {**{c: pa.float64() for c in num}, "NAICSP": pa.string()}
    parts = []
    for p in ACS:                      # streamed in 8 MB blocks; filtered per block
        co = pacsv.ConvertOptions(include_columns=num + ["NAICSP"], column_types=types)
        rd = pacsv.open_csv(p, convert_options=co, read_options=pacsv.ReadOptions(block_size=1 << 23, use_threads=False))
        for b in rd:
            t = pa.Table.from_batches([b])
            c = {k: t.column(k) for k in num}
            m = pcm.and_(pcm.and_(pcm.and_(pcm.greater_equal(c["AGEP"], 22), pcm.less_equal(c["AGEP"], 40)),
                                  pcm.and_(pcm.greater_equal(c["SCHL"], 21),
                                           pcm.and_(pcm.greater_equal(c["COW"], 1), pcm.less_equal(c["COW"], 5)))),
                         pcm.and_(pcm.and_(pcm.greater_equal(c["WKHP"], 35), pcm.greater_equal(c["WKWN"], 50)),
                                  pcm.and_(pcm.greater(c["WAGP"], 0), pcm.is_valid(t.column("NAICSP")))))
            parts.append(t.filter(pcm.fill_null(m, False)).to_pandas())
            del t, b
    d = pd.concat(parts, ignore_index=True)
    del parts
    n = d.NAICSP.str.strip()
    d = d[~n.str.startswith("92811") & (n != "999920")].copy()
    two = d.NAICSP.str.strip().str[:2]
    d["sector"] = two.replace({"31": "31-33", "32": "31-33", "33": "31-33", "3M": "31-33",
                               "44": "44-45", "45": "44-45", "4M": "44-45", "48": "48-49", "49": "48-49"})
    d = d[d.sector.isin(SECTORS)]
    rows = []
    for k, g in d.groupby("sector"):
        w = g.PWGTP.to_numpy(float)
        o = np.argsort(g.WAGP.to_numpy(float), kind="mergesort")
        cw = np.cumsum(w[o])
        med = float(g.WAGP.to_numpy(float)[o][np.searchsorted(cw, cw[-1] / 2)])
        rows.append(dict(sector=k, n_obs=len(g), weight=w.sum(),
                         gov=float(w[g.COW.between(3, 5).to_numpy()].sum() / w.sum()),
                         gov_np=float(w[g.COW.between(2, 5).to_numpy()].sum() / w.sum()),
                         med_wage=med))
    out = pd.DataFrame(rows)
    out["cls"] = np.where(out.sector.isin(SETTING), "setting", np.where(out.sector.isin(MARKET), "market", "other"))
    # class-level aggregates
    agg = []
    for cls, g in d.assign(cls=np.where(d.sector.isin(SETTING), "setting",
                                        np.where(d.sector.isin(MARKET), "market", "other"))).groupby("cls"):
        w = g.PWGTP.to_numpy(float)
        o = np.argsort(g.WAGP.to_numpy(float), kind="mergesort"); cw = np.cumsum(w[o])
        agg.append(dict(sector=f"[{cls}]", n_obs=len(g), weight=w.sum(),
                        gov=float(w[g.COW.between(3, 5).to_numpy()].sum() / w.sum()),
                        gov_np=float(w[g.COW.between(2, 5).to_numpy()].sum() / w.sum()),
                        med_wage=float(g.WAGP.to_numpy(float)[o][np.searchsorted(cw, cw[-1] / 2)]), cls=cls))
    return pd.concat([out, pd.DataFrame(agg)], ignore_index=True)


# =============================================================================================
# rank statistics
# =============================================================================================
def _rk(a):
    return rankdata(a)


def _pear(x, y):
    xc, yc = x - x.mean(), y - y.mean()
    den = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    return float((xc * yc).sum() / den) if den > 0 else np.nan


def spear(x, y):
    if len(x) < 5:
        return np.nan
    return _pear(_rk(x), _rk(y))


def partial_spear(x, y, Z):
    """Rank x, y and the continuous covariates (Z columns: ranked if continuous, dummies as given),
    residualise both on [1, Z], correlate residuals. NaN when residual df < DFMIN."""
    n = len(x)
    X = np.column_stack([np.ones(n), Z]) if Z.size else np.ones((n, 1))
    rkx = np.linalg.matrix_rank(X)
    if n - rkx - 1 < DFMIN:
        return np.nan
    rx, ry = _rk(x), _rk(y)
    bx = np.linalg.lstsq(X, rx, rcond=None)[0]; by = np.linalg.lstsq(X, ry, rcond=None)[0]
    return _pear(rx - X @ bx, ry - X @ by)


# statistic specs: name -> (x, y, partial?)
STATS_W = {  # Wapman sample (family prestige defined)
    "rP_M": ("P", "M", False), "rG_M": ("G", "M", False), "rP_Mplus": ("P", "Mplus", False),
    "rP_S": ("P", "S", False), "rG_S": ("G", "S", False),
    "rP_E": ("P", "log_earn", False), "rG_E": ("G", "log_earn", False),
    "rM_E": ("M", "log_earn", False), "rS_E": ("S", "log_earn", False),
    "rP_nme": ("P", "nme_share", False),
}
STATS_WS = {  # Wapman x Scorecard-selectivity sample
    "rP_M_ws": ("P", "M", False), "rG_M_ws": ("G", "M", False), "rSAT_M_ws": ("SAT", "M", False),
    "rP_E_ws": ("P", "log_earn", False), "rG_E_ws": ("G", "log_earn", False),
    "rSAT_E_ws": ("SAT", "log_earn", False),
    "pP_M": ("P", "M", True), "pG_M": ("G", "M", True), "pP_S": ("P", "S", True),
    "pP_E": ("P", "log_earn", True), "pG_E": ("G", "log_earn", True),
}
STATS_B = {  # broad selectivity sample (all PSEO institutions with Scorecard covariates)
    "rSAT_M_b": ("SAT", "M", False), "rADM_M_b": ("NEG_ADM", "M", False),
    "rSAT_S_b": ("SAT", "S", False), "rADM_S_b": ("NEG_ADM", "S", False),
    "rSAT_E_b": ("SAT", "log_earn", False), "rADM_E_b": ("NEG_ADM", "log_earn", False),
    "rM_E_b": ("M", "log_earn", False), "rS_E_b": ("S", "log_earn", False),
}
STAT_LABEL = {
    "rP_M": "ρ(field-family prestige P, market share M)", "rG_M": "ρ(brand G, M)",
    "rP_Mplus": "ρ(P, M+55)", "rP_S": "ρ(P, setting share S)", "rG_S": "ρ(G, S)",
    "rP_E": "wage coupling ρ(P, log p50)", "rG_E": "ρ(G, log p50)", "rM_E": "ρ(M, log p50)",
    "rS_E": "ρ(S, log p50)", "rP_nme": "ρ(P, non/marginal-employment share)",
    "rP_M_ws": "ρ(P, M)", "rG_M_ws": "ρ(G, M)", "rSAT_M_ws": "ρ(SAT, M)",
    "rP_E_ws": "ρ(P, log p50)", "rG_E_ws": "ρ(G, log p50)", "rSAT_E_ws": "ρ(SAT, log p50)",
    "pP_M": "partial ρ(P, M), net of sel.", "pG_M": "partial ρ(G, M), net of sel.", "pP_S": "partial ρ(P, S), net of sel.",
    "pP_E": "partial ρ(P, log p50), net of sel.", "pG_E": "partial ρ(G, log p50), net of sel.",
    "rSAT_M_b": "ρ(SAT, M)", "rADM_M_b": "ρ(−ADM, M)", "rSAT_S_b": "ρ(SAT, S)", "rADM_S_b": "ρ(−ADM, S)",
    "rSAT_E_b": "ρ(SAT, log p50)", "rADM_E_b": "ρ(−ADM, log p50)", "rM_E_b": "ρ(M, log p50)",
    "rS_E_b": "ρ(S, log p50)",
}


class FamStats:
    """Per-family arrays of one sample + all its statistics on a row index (with repetition)."""

    def __init__(self, d: pd.DataFrame, specs: dict, inst_index: dict):
        self.specs = specs
        self.fams = [c for c, g in d.groupby("cip2", sort=True) if len(g) >= NMIN]
        self.data = {}
        for c in self.fams:
            g = d[d.cip2 == c].reset_index(drop=True)
            A = {k: g[k].to_numpy(float) for k in set(sum([[x, y] for x, y, _ in specs.values()], []))}
            A["inst"] = g.institution.map(inst_index).to_numpy(int)
            A["S_raw"] = g.S.to_numpy(float)
            A["M_raw"] = g.M.to_numpy(float)
            if any(p for _, _, p in specs.values()):
                A["CONTROL"] = g.CONTROL.to_numpy(float)
                for cc in SEL_CONT:
                    A[cc] = g[cc].to_numpy(float)
            self.data[c] = A

    def stats(self, c, rows=None) -> dict:
        A = self.data[c]
        n = len(A["inst"])
        rows = np.arange(n) if rows is None else rows
        out = {}
        Z = None
        for name, (x, y, part) in self.specs.items():
            xs, ys = A[x][rows], A[y][rows]
            if part:
                if Z is None:
                    # re-rank covariates on the drawn rows; control dummies from the drawn rows
                    cols = [_rk(A[cc][rows]) for cc in SEL_CONT]
                    ctl = A["CONTROL"][rows]
                    for v in sorted(np.unique(ctl))[1:]:
                        cols.append((ctl == v).astype(float))
                    Z = np.column_stack(cols)
                out[name] = partial_spear(xs, ys, Z)
            else:
                out[name] = spear(xs, ys)
        return out

    def split_stat(self, c, x, y, rows=None, partial=False) -> float:
        """Coupling in the high-S half minus the low-S half (split at the family median of S).
        partial=True: within each half, partial Spearman of x and y given (ranked) S, so that the
        remaining variation of S inside a half does not drive either coupling."""
        A = self.data[c]
        rows = np.arange(len(A["inst"])) if rows is None else rows
        s = A["S_raw"][rows]
        med = np.median(s)
        hi, lo = rows[s > med], rows[s <= med]
        if len(hi) < 6 or len(lo) < 6:
            return np.nan
        if partial:
            if len(hi) < 8 or len(lo) < 8:
                return np.nan
            return _pspear_s(A[x][hi], A[y][hi], A["S_raw"][hi]) - _pspear_s(A[x][lo], A[y][lo], A["S_raw"][lo])
        return spear(A[x][hi], A[y][hi]) - spear(A[x][lo], A[y][lo])

    def csplit_stat(self, c, x, y, rows=None, by="S", cond=None) -> float:
        """Shape-free conditional median split (second revision). by: the share whose family-median split
        defines high / low; cond: None, or the other share: the rows are first split at the family median of
        `cond`, and the high-minus-low coupling difference by `by` is computed inside each `cond` half and
        averaged over the two halves (so the `by` contrast is taken at roughly fixed `cond`)."""
        A = self.data[c]
        rows = np.arange(len(A["inst"])) if rows is None else rows
        parts = [rows]
        if cond is not None:
            v = A[f"{cond}_raw"][rows]
            med = np.median(v)
            parts = [rows[v > med], rows[v <= med]]
        diffs = []
        for g in parts:
            s = A[f"{by}_raw"][g]
            med = np.median(s)
            hi, lo = g[s > med], g[s <= med]
            if len(hi) < 6 or len(lo) < 6:
                return np.nan
            diffs.append(spear(A[x][hi], A[y][hi]) - spear(A[x][lo], A[y][lo]))
        return float(np.mean(diffs))


CSPLIT_KINDS = [("msplit", "M", None), ("csplitS", "S", "M"), ("csplitM", "M", "S")]


def _pspear_s(x, y, s):
    """Partial Spearman of x and y given one covariate s (all ranked)."""
    X = np.column_stack([np.ones(len(x)), _rk(s)])
    rx, ry = _rk(x), _rk(y)
    bx = np.linalg.lstsq(X, rx, rcond=None)[0]; by = np.linalg.lstsq(X, ry, rcond=None)[0]
    return _pear(rx - X @ bx, ry - X @ by)


def joint_bootstrap(fs: FamStats, n_inst: int, tag: str, split_pairs=()) -> dict:
    """Resample institutions with replacement (jointly for all families); per draw, every family's
    statistics on the rows of the drawn institutions (with multiplicity)."""
    rng = rng_for(tag)
    names = (list(fs.specs) + [f"split_{x}_{y}" for x, y in split_pairs]
             + [f"psplit_{x}_{y}" for x, y in split_pairs]
             + [f"{kd}_{x}_{y}" for kd, _, _ in CSPLIT_KINDS for x, y in split_pairs])
    K = len(fs.fams)
    out = {k: np.full((B_BOOT, K), np.nan) for k in names}
    for b in range(B_BOOT):
        cnt = np.bincount(rng.integers(n_inst, size=n_inst), minlength=n_inst)
        for j, c in enumerate(fs.fams):
            A = fs.data[c]
            rows = np.repeat(np.arange(len(A["inst"])), cnt[A["inst"]])
            if len(rows) < 5:
                continue
            st = fs.stats(c, rows)
            for k, v in st.items():
                out[k][b, j] = v
            for x, y in split_pairs:
                out[f"split_{x}_{y}"][b, j] = fs.split_stat(c, x, y, rows)
                out[f"psplit_{x}_{y}"][b, j] = fs.split_stat(c, x, y, rows, partial=True)
                for kd, by, cond in CSPLIT_KINDS:
                    out[f"{kd}_{x}_{y}"][b, j] = fs.csplit_stat(c, x, y, rows, by=by, cond=cond)
    return out


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def summarise(point: pd.DataFrame, boot: dict, fams: list, name: str, mask=None) -> dict:
    """Mean over families with a finite point estimate, bootstrap CI of that mean (same families),
    counts of families with bootstrap CI above / below zero."""
    p = point.set_index("cip2").loc[fams, name].to_numpy(float)
    use = np.isfinite(p) if mask is None else (np.isfinite(p) & mask)
    B = boot[name][:, use]
    mean_b = np.nanmean(B, axis=1)
    lo_f = np.nanpercentile(B, 2.5, axis=0); hi_f = np.nanpercentile(B, 97.5, axis=0)
    return dict(k=int(use.sum()), mean=float(np.mean(p[use])) if use.any() else np.nan,
                ci=ci(mean_b), n_pos=int((lo_f > 0).sum()), n_neg=int((hi_f < 0).sum()),
                n_gt0=int((p[use] > 0).sum()), median=float(np.median(p[use])) if use.any() else np.nan,
                sd=float(np.std(p[use], ddof=1)) if use.sum() > 1 else np.nan)


def paired(point, boot, fams, a, b) -> dict:
    pa_ = point.set_index("cip2").loc[fams, a].to_numpy(float)
    pb_ = point.set_index("cip2").loc[fams, b].to_numpy(float)
    use = np.isfinite(pa_) & np.isfinite(pb_)
    d = np.nanmean(boot[a][:, use] - boot[b][:, use], axis=1)
    return dict(k=int(use.sum()), diff=float(np.mean(pa_[use] - pb_[use])), ci=ci(d),
                n_a_gt=int((pa_[use] > pb_[use]).sum()))


def split_half_order(d: pd.DataFrame, fams: list, pairs: list, tag: str) -> dict:
    """Cross-family Spearman between two statistics estimated on independent halves of the
    institutions (A->B and B->A, averaged over K_SPLIT splits); permutation p from relabelling
    families (same permutation in every split)."""
    rng = rng_for(tag)
    insts = np.array(sorted(d.institution.unique()))
    res = {}
    halves = []
    for _ in range(K_SPLIT):
        perm = rng.permutation(len(insts))
        halves.append((set(insts[perm[: len(insts) // 2]]), set(insts[perm[len(insts) // 2:]])))
    fam_d = {c: d[d.cip2 == c] for c in fams}
    for sx, sy in pairs:
        XA = np.full((K_SPLIT, 2, len(fams)), np.nan); YB = np.full((K_SPLIT, 2, len(fams)), np.nan)
        (x1, y1), (x2, y2) = STATS_W[sx][:2], STATS_W[sy][:2]
        for s, (A, B) in enumerate(halves):
            for o, (h1, h2) in enumerate([(A, B), (B, A)]):
                for j, c in enumerate(fams):
                    g = fam_d[c]
                    g1, g2 = g[g.institution.isin(h1)], g[g.institution.isin(h2)]
                    if len(g1) >= NMIN_HALF and len(g2) >= NMIN_HALF:
                        XA[s, o, j] = spear(g1[x1].to_numpy(float), g1[y1].to_numpy(float))
                        YB[s, o, j] = spear(g2[x2].to_numpy(float), g2[y2].to_numpy(float))

        def cross(pi):
            vals = []
            for s in range(K_SPLIT):
                for o in range(2):
                    xa, yb = XA[s, o], YB[s, o][pi]
                    m = np.isfinite(xa) & np.isfinite(yb)
                    if m.sum() >= 6:
                        vals.append(spear(xa[m], yb[m]))
            return float(np.mean(vals)) if vals else np.nan, len(vals)
        obs, nv = cross(np.arange(len(fams)))
        prng = rng_for(tag + sx + sy)
        null = np.array([cross(prng.permutation(len(fams)))[0] for _ in range(N_PERM)])
        full_x = np.array([spear(fam_d[c][x1].to_numpy(float), fam_d[c][y1].to_numpy(float)) for c in fams])
        full_y = np.array([spear(fam_d[c][x2].to_numpy(float), fam_d[c][y2].to_numpy(float)) for c in fams])
        res[(sx, sy)] = dict(cross=obs, n_rep=nv, p=float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + len(null))),
                             null_mean=float(np.mean(null)), same=spear(full_x, full_y), k=len(fams))
    return res


# =============================================================================================
# cell-level regressions
# =============================================================================================
def zc(d: pd.DataFrame, col: str) -> np.ndarray:
    g = d.groupby("cip2")[col]
    return ((d[col] - g.transform("mean")) / g.transform("std")).fillna(0.0).to_numpy()


def _fam_quintile(d: pd.DataFrame, v: np.ndarray) -> np.ndarray:
    """Within-family quintile (0..4) of v; ties broken by row order (d is sorted by cip2, institution)."""
    r = pd.Series(v).groupby(d.cip2.to_numpy()).rank(method="first")
    n = pd.Series(v).groupby(d.cip2.to_numpy()).transform("size")
    return np.minimum(((r - 1) * 5 // n).to_numpy(int), 4)


def zr(d: pd.DataFrame, col: str) -> np.ndarray:
    """Within-family standardised rank (mean 0, SD 1 within family)."""
    r = d.groupby("cip2")[col].rank(method="average")
    g = r.groupby(d.cip2)
    return ((r - g.transform("mean")) / g.transform("std")).fillna(0.0).to_numpy()


def design(d: pd.DataFrame, status: list, expo: str = "S", extras: tuple = (), extras_int: bool = True,
           quad: bool = False, state_fe: bool = False, inst_fe: bool = False, fam_theta: bool = False,
           rank_y: bool = False, ecurve: str | None = None, zq5: bool = False, rank_x: bool = False,
           expo2: tuple = ()):
    """X, y, names for  y = gamma_c + sum_s beta_{s,c} z_c(s) + delta_c e + [curvature in e] + theta_s z_c(s) e + ...
    e = (expo - family mean)/0.10. extras: family-specific slopes on z_c(x) (+ z_c(x) e if extras_int).
    ecurve (exposure main-effect shape): 'lin' = family-specific linear only; 'e2c' = + one common e^2;
    'e2' = + family-specific e^2 (PRIMARY, ECURVE_MAIN); 'e3' = + family-specific e^2 and e^3; 'eq5' = +
    family x within-family e-quintile dummies; 'e2x' = 'e2' + family-specific cross products of the
    exposures (= 'e2' with one exposure). zq5: + family x within-family z-quintile dummies of status[0].
    rank_x: status and exposure replaced by within-family standardised ranks (e then in SD-of-rank units).
    expo2: further exposures (second revision), each with the same main-effect shape and its own
    theta_{s}[x] = z_c(s) e_x term (names suffixed '[x]'); extras_int adds z_c(x) e_x for them too. With
    expo2 = () the columns and their order are exactly those of the single-exposure model."""
    ecurve = ECURVE_MAIN if ecurve is None else ecurve
    d = d.sort_values(["cip2", "institution"]).reset_index(drop=True)
    fams = sorted(d.cip2.unique())
    D = (d.cip2.to_numpy()[:, None] == np.array(fams)[None, :]).astype(float)

    def _expo(col):
        if rank_x:
            return zr(d, col)
        return ((d[col] - d.groupby("cip2")[col].transform("mean")) / 0.10).to_numpy()
    e = _expo(expo)
    E = [("", e)] + [(f"[{x}]", _expo(x)) for x in expo2]      # (name suffix, exposure)
    cols, names = [], []
    for j, c in enumerate(fams):
        if not (inst_fe and j == 0):
            cols.append(D[:, j]); names.append(f"g_{c}")
    Zs = {s: (zr(d, s) if rank_x else zc(d, s)) for s in status}
    for s in status:
        for j, c in enumerate(fams):
            cols.append(D[:, j] * Zs[s]); names.append(f"b_{s}_{c}")
    for sf, ex in E:
        for j, c in enumerate(fams):
            cols.append(D[:, j] * ex); names.append(f"d{sf}_{c}")
    for sf, ex in E:
        if ecurve == "e2c":
            cols.append(ex ** 2); names.append("e2c" + sf)
        elif ecurve in ("e2", "e3", "e2x"):
            for j, c in enumerate(fams):
                cols.append(D[:, j] * ex ** 2); names.append(f"e2{sf}_{c}")
            if ecurve == "e3":
                for j, c in enumerate(fams):
                    cols.append(D[:, j] * ex ** 3); names.append(f"e3{sf}_{c}")
        elif ecurve == "eq5":
            qe = _fam_quintile(d, ex)
            for j, c in enumerate(fams):
                for q in range(1, 5):
                    cols.append(D[:, j] * (qe == q)); names.append(f"eq{q}{sf}_{c}")
        elif ecurve != "lin":
            raise ValueError(ecurve)
    if ecurve == "e2x":
        for a in range(len(E)):
            for b_ in range(a + 1, len(E)):
                for j, c in enumerate(fams):
                    cols.append(D[:, j] * E[a][1] * E[b_][1]); names.append(f"ex{E[a][0]}{E[b_][0]}_{c}")
    if zq5:
        qz = _fam_quintile(d, Zs[status[0]])
        for j, c in enumerate(fams):
            for q in range(1, 5):
                cols.append(D[:, j] * (qz == q)); names.append(f"zq{q}_{c}")
    if quad:
        for j, c in enumerate(fams):
            cols.append(D[:, j] * Zs[status[0]] ** 2); names.append(f"q_{c}")
    for x in extras:
        if x == "CONTROL":
            zx = (d.CONTROL.to_numpy(float) > 1).astype(float)            # private (2, 3) vs public
            zx = zx - pd.Series(zx).groupby(d.cip2.to_numpy()).transform("mean").to_numpy()
        else:
            zx = zc(d, x)
        for j, c in enumerate(fams):
            cols.append(D[:, j] * zx); names.append(f"x_{x}_{c}")
        if extras_int:
            cols.append(zx * e); names.append(f"eta_{x}")
            for sf, ex in E[1:]:
                cols.append(zx * ex); names.append(f"eta_{x}{sf}")
    for s in status:
        if fam_theta:
            for j, c in enumerate(fams):
                cols.append(D[:, j] * Zs[s] * e); names.append(f"theta_{s}_{c}")
        else:
            cols.append(Zs[s] * e); names.append(f"theta_{s}")
            for sf, ex in E[1:]:
                cols.append(Zs[s] * ex); names.append(f"theta_{s}{sf}")
    if state_fe:
        st = sorted(d.institution_state.unique())
        for v in st[1:]:
            cols.append((d.institution_state == v).to_numpy(float)); names.append(f"st_{v}")
    X = np.column_stack(cols)
    if rank_y:
        y = d.groupby("cip2").log_earn.rank(pct=True).to_numpy()
    else:
        y = d.log_earn.to_numpy(float)
    groups = pd.factorize(d.institution)[0]
    return X, y, names, groups, d, e


def demean(M, groups):
    cnt = np.bincount(groups).astype(float)
    if M.ndim == 1:
        s = np.bincount(groups, weights=M)
        return M - (s / cnt)[groups]
    s = np.zeros((cnt.size, M.shape[1])); np.add.at(s, groups, M)
    return M - (s / cnt[:, None])[groups]


def wcr_pvalues(X: np.ndarray, y: np.ndarray, groups: np.ndarray, idx: dict, tag: str) -> dict:
    """Wild-cluster restricted bootstrap (WCR) p-values for single coefficients (third revision). For each
    coefficient j in idx: the model is refitted with coefficient j restricted to 0; B_WCR bootstrap samples
    y* = X_r b_r + u_r * w_g (Rademacher w_g drawn per institution) are refitted unrestricted, and the CR1 t
    statistic of coefficient j is compared with the observed one: p = (1 + #{|t*| >= |t|}) / (1 + B_WCR).
    X, y are the (possibly within-transformed or sqrt-weighted) estimation arrays; idx: key -> column index."""
    N = X.shape[0]
    XtXi = np.linalg.pinv(X.T @ X)
    K = int(np.linalg.matrix_rank(X))
    G = int(groups.max()) + 1
    Gm = sps.csr_matrix((np.ones(N), (groups, np.arange(N))), shape=(G, N))
    cfac = G / (G - 1) * (N - 1) / (N - K)          # CR1 factor (as statsmodels' cluster covariance)
    out = {}
    for key in sorted(idx, key=str):
        j = idx[key]
        h = X @ XtXi[j]                                # beta_j = h'y

        def tstat(Y):
            U = Y - X @ (XtXi @ (X.T @ Y))
            Sg = np.asarray(Gm @ (h[:, None] * U))
            return (h @ Y) / np.sqrt(cfac * (Sg ** 2).sum(axis=0))
        t0 = float(tstat(y[:, None])[0])
        Xr = np.delete(X, j, axis=1)
        yr = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
        ur = y - yr
        rng = rng_for(f"{tag}|{key}")
        exceed, done = 0, 0
        while done < B_WCR:
            b = min(250, B_WCR - done)
            wg = rng.choice(np.array([-1.0, 1.0]), size=(G, b))
            ts = tstat(yr[:, None] + ur[:, None] * wg[groups])
            exceed += int(np.sum(np.abs(ts) >= abs(t0) - 1e-12))
            done += b
        out[key] = float((1 + exceed) / (1 + B_WCR))
    return out


def fit_interaction(d: pd.DataFrame, status: list, tag: str, expo: str = "S", boot: bool = False,
                    weights=None, jack: bool = False, wcr: bool = False, **kw) -> dict:
    X, y, names, groups, dd, e = design(d, status, expo=expo, **kw)
    inst_fe = kw.get("inst_fe", False)
    Xf, yf = (demean(X, groups), demean(y, groups)) if inst_fe else (X, y)
    if weights is not None:
        w = dd[weights].to_numpy(float)
        fit = sm.WLS(yf, Xf, weights=w).fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        fit = sm.OLS(yf, Xf).fit(cov_type="cluster", cov_kwds={"groups": groups})
    par, V = np.asarray(fit.params), np.asarray(fit.cov_params())
    fams = sorted(dd.cip2.unique())
    nc = dd.groupby("cip2").size().reindex(fams).to_numpy(float)
    out = dict(tag=tag, n_cells=len(dd), n_inst=int(dd.institution.nunique()), k=len(fams),
               sd_e=float(np.std(e, ddof=1)), names=names, params=par, V=V,
               ecurve=kw.get("ecurve") or ECURVE_MAIN, e=e, fams=fams, nc=nc, expo=expo)
    expo2 = tuple(kw.get("expo2", ()))

    def _tdict(j):
        return dict(theta=float(par[j]), se=float(np.sqrt(V[j, j])),
                    ci=(float(par[j] - 1.96 * np.sqrt(V[j, j])), float(par[j] + 1.96 * np.sqrt(V[j, j]))),
                    p=float(2 * norm.sf(abs(par[j] / np.sqrt(V[j, j])))))
    TH = {}                                      # (status, exposure) -> theta dict; primary entries shared with out[s]
    for s in status:
        if kw.get("fam_theta"):
            continue
        j = names.index(f"theta_{s}")
        c = np.zeros(len(names))
        for i, f in enumerate(fams):
            c[names.index(f"b_{s}_{f}")] = nc[i] / nc.sum()
        beta, beta_se = float(c @ par), float(np.sqrt(c @ V @ c))
        if kw.get("zq5"):                       # with z-quintile dummies, b_ is not the slope at the mean
            beta, beta_se = np.nan, np.nan
        out[s] = dict(**_tdict(j), beta=beta, beta_se=beta_se)
        TH[(s, expo)] = out[s]
        for x in expo2:
            TH[(s, x)] = _tdict(names.index(f"theta_{s}[{x}]"))
    out["TH"] = TH
    tidx = {k: names.index(f"theta_{k[0]}" if k[1] == expo else f"theta_{k[0]}[{k[1]}]") for k in TH}
    if wcr:                                      # wild-cluster restricted bootstrap p for every theta
        if weights is not None:
            sw = np.sqrt(dd[weights].to_numpy(float))
            Xw, yw = Xf * sw[:, None], yf * sw
        else:
            Xw, yw = Xf, yf
        ps = wcr_pvalues(Xw, yw, groups, tidx, "wcr" + tag)
        for k in TH:
            TH[k]["wcr_p"] = ps[k]
    if jack:                                     # delete-one-institution cluster jackknife
        G = groups.max() + 1
        th = {k: np.full(G, np.nan) for k in TH}
        for g in range(G):
            keep = groups != g
            Xg, yg = Xf[keep], yf[keep]
            if weights is not None:
                sw = np.sqrt(w[keep]); Xg, yg = Xg * sw[:, None], yg * sw
            coef = np.linalg.lstsq(Xg, yg, rcond=None)[0]
            for k in TH:
                th[k][g] = coef[tidx[k]]
        for k in TH:
            se_j = float(np.sqrt((G - 1) / G * np.sum((th[k] - th[k].mean()) ** 2)))
            TH[k]["jack_se"] = se_j
            TH[k]["jack_ci"] = (TH[k]["theta"] - 1.96 * se_j, TH[k]["theta"] + 1.96 * se_j)
    if boot:
        rng = rng_for("clu" + tag)
        G = groups.max() + 1
        idx_by_g = [np.flatnonzero(groups == g) for g in range(G)]
        th = {k: np.full(B_CLU, np.nan) for k in TH}
        for b in range(B_CLU):
            gs = rng.integers(G, size=G)
            rows = np.concatenate([idx_by_g[g] for g in gs])
            newg = np.repeat(np.arange(G), [len(idx_by_g[g]) for g in gs])
            Xb, yb = X[rows], y[rows]
            if inst_fe:
                Xb, yb = demean(Xb, newg), demean(yb, newg)
            if weights is not None:
                sw = np.sqrt(dd[weights].to_numpy(float)[rows])
                Xb, yb = Xb * sw[:, None], yb * sw
            coef = np.linalg.lstsq(Xb, yb, rcond=None)[0]
            for k in TH:
                th[k][b] = coef[tidx[k]]
        for k in TH:
            TH[k]["boot_ci"] = ci(th[k])
            TH[k]["boot_p"] = float(min(1.0, 2 * min(np.mean(th[k] <= 0), np.mean(th[k] >= 0))))
    return out


def wald(r: dict, restr: list) -> dict:
    """Wald test of linear restrictions on a fitted model's coefficients (CR1 covariance).
    restr: list of rows, each a list of (coefficient name, weight); H0: every row = 0."""
    names, par, V = r["names"], r["params"], r["V"]
    R = np.zeros((len(restr), len(names)))
    for i, terms in enumerate(restr):
        for nm, wt in terms:
            R[i, names.index(nm)] = wt
    rb, RV = R @ par, R @ V @ R.T
    W = float(rb @ np.linalg.pinv(RV) @ rb)
    dfw = int(np.linalg.matrix_rank(RV))
    se = np.sqrt(np.diag(RV))
    return dict(W=W, df=dfw, p=float(chi2.sf(W, dfw)), est=[float(v) for v in rb], se=[float(v) for v in se],
                ci=[(float(a - 1.96 * b), float(a + 1.96 * b)) for a, b in zip(rb, se)])


def _tn(s: str, x: str, expo: str = "S") -> str:
    """Coefficient name of the status x exposure product term."""
    return f"theta_{s}" if x == expo else f"theta_{s}[{x}]"


def joint_tests(r: dict, s: str, expo: str = "S") -> dict:
    """For a fit with exposures (expo, M, L): theta_S - theta_L and joint theta_S = theta_L = 0; for
    (expo, M): theta_S + theta_M (a pure 'non-market vs market' contrast would make them opposite)."""
    xs = [expo] + [x for (s_, x) in r["TH"] if s_ == s and x != expo]
    out = {}
    if "L" in xs or "L2" in xs:
        lx = "L" if "L" in xs else "L2"
        out["S_minus_L"] = wald(r, [[(_tn(s, expo, expo), 1.0), (_tn(s, lx, expo), -1.0)]])
        out["S_L_zero"] = wald(r, [[(_tn(s, expo, expo), 1.0)], [(_tn(s, lx, expo), 1.0)]])
    if "M" in xs:
        out["S_plus_M"] = wald(r, [[(_tn(s, expo, expo), 1.0), (_tn(s, "M", expo), 1.0)]])
    return out


def _lean(r: dict) -> dict:
    """Drop the large per-fit arrays (coefficients, covariance, exposure vector)."""
    for x in ("params", "V", "e", "names"):
        r.pop(x, None)
    return r


def joint_battery(cw: pd.DataFrame, cb: pd.DataFrame, tagp: str) -> dict:
    """Second revision: theta for S jointly with M and / or L (JSETS), for P and G (W-type sample) and SAT
    (B-type sample), under every functional form in JFF_SPECS. Keyed (status key, set, form); lean."""
    out = {}
    for key, d, s in [("P", cw, "P"), ("G", cw, "G"), ("SAT_b", cb, "SAT")]:
        if d is None or len(d) == 0 or d.cip2.nunique() < 2:
            continue
        for sk, xs, _ in JSETS:
            for k, _, kw in JFF_SPECS:
                r = fit_interaction(d, [s], f"{tagp}jb{key}{sk}{k}", expo2=xs, **kw)
                r["tests"] = joint_tests(r, s)
                out[(key, sk, k)] = _lean(r)
    return out


def interaction_collinearity(d: pd.DataFrame, s: str, others=("M", "L")) -> dict:
    """Pooled within-family correlations between exposures and between the status x exposure products."""
    dd = d.sort_values(["cip2", "institution"]).reset_index(drop=True)
    z = zc(dd, s)
    E = {x: ((dd[x] - dd.groupby("cip2")[x].transform("mean")) / 0.10).to_numpy() for x in ("S",) + tuple(others)}
    out = {}
    for x in others:
        out[f"e_S_{x}"] = float(np.corrcoef(E["S"], E[x])[0, 1])
        out[f"ze_S_{x}"] = float(np.corrcoef(z * E["S"], z * E[x])[0, 1])
        out[f"b_{x}_on_S"] = float(np.polyfit(E["S"], E[x], 1)[0])       # within-family slope of e_x on e_S
    return out


def gelbach(d: pd.DataFrame, s: str, expo2=("M",), **kw) -> dict:
    """Exact OLS decomposition (Gelbach 2016) of theta_S(alone) - theta_S(joint) into the contributions of the
    columns the joint model adds: the other exposures' main effects (family-specific linear + shape) and
    their status x exposure products. contribution_k = delta_k * gamma_k, delta_k = coefficient on the
    theta_S column when added column k is regressed on the S-only design."""
    Xa, y, na, _, _, _ = design(d, [s], **kw)
    Xj, yj, nj, groups, _, _ = design(d, [s], expo2=expo2, **kw)
    assert np.allclose(y, yj)
    fj = sm.OLS(yj, Xj).fit(cov_type="cluster", cov_kwds={"groups": groups})
    gam = np.asarray(fj.params)
    fa = np.linalg.lstsq(Xa, y, rcond=None)[0]
    add = [i for i, n in enumerate(nj) if n not in na]
    jt = na.index(f"theta_{s}")
    delta = np.linalg.lstsq(Xa, Xj[:, add], rcond=None)[0][jt]
    contrib = delta * gam[add]
    groups_ = {}
    for i, k in enumerate(add):
        n = nj[k]
        if n.startswith("theta_"):
            lab = "product " + n.split("[")[1].rstrip("]")
        elif n.startswith("ex"):
            lab = "cross products"
        else:
            lab = "main effects " + n.split("[")[1].split("]")[0]
        groups_[lab] = groups_.get(lab, 0.0) + float(contrib[i])
    th_a = float(fa[jt]); th_j = float(gam[nj.index(f"theta_{s}")])
    return dict(theta_alone=th_a, theta_joint=th_j, diff=th_a - th_j, parts=groups_,
                check=float(sum(groups_.values()) - (th_a - th_j)))


def ff_battery(cw: pd.DataFrame, cb: pd.DataFrame, tagp: str, lean: bool = True) -> dict:
    """theta for P and G (W-type sample) and SAT (B-type sample) under every exposure/status functional
    form in FF_SPECS. lean: drop the coefficient vectors (only theta, CI, beta are kept)."""
    out = {}
    for key, d, s in [("P", cw, "P"), ("G", cw, "G"), ("SAT_b", cb, "SAT")]:
        if d is None or len(d) == 0 or d.cip2.nunique() < 2:
            continue
        for k, _, kw in FF_SPECS:
            r = fit_interaction(d, [s], f"{tagp}ff{key}{k}", **kw)
            if lean:
                for x in ("params", "V", "e", "names"):
                    r.pop(x, None)
            out[(key, k)] = r
    return out


def implied_slopes(r: dict, s: str) -> dict:
    """Cell-weighted status slope at the within-family 10th / 90th percentile of e (beta_bar + theta e),
    the decline between them, and where the slope reaches zero. Valid for specs in which the status slope
    is beta_c + theta e at z = 0 (linear, e^2, e^3, e-quintile, and e^2 + z^2 at the family-mean status)."""
    names, par, V, e = r["names"], r["params"], r["V"], r["e"]
    fams_, nc = r["fams"], r["nc"]
    p10, p90 = float(np.percentile(e, 10)), float(np.percentile(e, 90))
    c0 = np.zeros(len(names))
    for i, f in enumerate(fams_):
        c0[names.index(f"b_{s}_{f}")] = nc[i] / nc.sum()
    jt = names.index(f"theta_{s}")
    sl = {}
    for lab, x in [("p10", p10), ("p90", p90)]:
        c = c0.copy(); c[jt] += x
        sl[lab] = (float(c @ par), float(np.sqrt(c @ V @ c)), x * 10)
    v = r[s]
    zero = -v["beta"] / v["theta"] if v["theta"] < 0 else np.nan
    return dict(slope=sl, decline=float(1 - sl["p90"][0] / sl["p10"][0]),
                zero_pp=zero * 10 if np.isfinite(zero) else np.nan,
                share_beyond=float(np.mean(e > zero)) if np.isfinite(zero) else np.nan)


def implied_joint(r: dict, d: pd.DataFrame, s: str, x: str = "M") -> dict:
    """Joint S + x model: cell-weighted status slope at the within-family 10th / 90th percentile of e_S with
    e_x at its family mean, and at the 10th / 90th percentile of e_x with e_S at its family mean."""
    dd = d.sort_values(["cip2", "institution"]).reset_index(drop=True)
    ex = ((dd[x] - dd.groupby("cip2")[x].transform("mean")) / 0.10).to_numpy()
    names, par, V, es = r["names"], r["params"], r["V"], r["e"]
    c0 = np.zeros(len(names))
    for i, f in enumerate(r["fams"]):
        c0[names.index(f"b_{s}_{f}")] = r["nc"][i] / r["nc"].sum()
    out = {}
    for lab, jn, ev in [("S", f"theta_{s}", es), (x, f"theta_{s}[{x}]", ex)]:
        for q in (10, 90):
            val = float(np.percentile(ev, q))
            c = c0.copy(); c[names.index(jn)] += val
            out[(lab, q)] = (float(c @ par), float(np.sqrt(c @ V @ c)), val * 10)
    return out


def curvature_diag(d: pd.DataFrame, s: str, r_e2c: dict, r_e2: dict) -> dict:
    """Why the exposure main effect matters: within-family corr(z, e) and corr(z*e, e^2) (pooled over
    families), the common e^2 coefficient, and a Wald test that all family-specific e^2 terms are 0."""
    dd = d.sort_values(["cip2", "institution"]).reset_index(drop=True)
    z = zc(dd, s)
    e = ((dd.S - dd.groupby("cip2").S.transform("mean")) / 0.10).to_numpy()
    out = dict(corr_ze=float(np.corrcoef(z, e)[0, 1]), corr_zee2=float(np.corrcoef(z * e, e ** 2)[0, 1]))
    j = r_e2c["names"].index("e2c")
    b, se = float(r_e2c["params"][j]), float(np.sqrt(r_e2c["V"][j, j]))
    out.update(e2c=b, e2c_ci=(b - 1.96 * se, b + 1.96 * se))
    idx = [i for i, n in enumerate(r_e2["names"]) if n.startswith("e2_")]
    pb, Vb = r_e2["params"][idx], r_e2["V"][np.ix_(idx, idx)]
    W = float(pb @ np.linalg.pinv(Vb) @ pb); dfw = int(np.linalg.matrix_rank(Vb))
    out.update(wald_e2=W, df_e2=dfw, p_e2=float(chi2.sf(W, dfw)), n_pos_e2=int((pb > 0).sum()), k_e2=len(idx))
    return out


def fit_mixed(d: pd.DataFrame, status: str) -> dict:
    """Random intercept for institution; family intercepts, family-specific slopes, common theta."""
    X, y, names, groups, dd, e = design(d, [status])
    j = names.index(f"theta_{status}")
    # lbfgs can stop at a degenerate boundary (log-likelihood inf) on these designs; use bfgs and fall
    # back to powell unless the fit converged with a finite log-likelihood.
    for meth in ("bfgs", "powell"):
        try:
            m = sm.MixedLM(y, X, groups=groups).fit(reml=True, method=meth, maxiter=2000)
        except Exception:                                      # pragma: no cover
            continue
        if m.converged and np.isfinite(m.llf):
            return dict(theta=float(m.params[j]), se=float(m.bse[j]), converged=True, method=meth,
                        re_var=float(np.asarray(m.cov_re)[0, 0]), resid_var=float(m.scale), llf=float(m.llf))
    return dict(theta=np.nan, se=np.nan, converged=False, method="none")


def fam_theta_table(d: pd.DataFrame, status: str) -> tuple[pd.DataFrame, dict]:
    """Family-specific theta_c (same model, theta varying by family) + Wald test of equality."""
    X, y, names, groups, dd, e = design(d, [status], fam_theta=True)
    fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    par, V = np.asarray(fit.params), np.asarray(fit.cov_params())
    fams = sorted(dd.cip2.unique())
    idx = [names.index(f"theta_{status}_{c}") for c in fams]
    rows = [dict(cip2=c, theta=par[i], se=np.sqrt(V[i, i]), n=int((dd.cip2 == c).sum())) for c, i in zip(fams, idx)]
    R = np.zeros((len(idx) - 1, len(names)))
    for r, i in enumerate(idx[1:]):
        R[r, idx[0]] = -1; R[r, i] = 1
    rb = R @ par; RV = R @ V @ R.T
    W = float(rb @ np.linalg.pinv(RV) @ rb); dfw = int(np.linalg.matrix_rank(RV))
    return pd.DataFrame(rows), dict(wald=W, df=dfw, p=float(chi2.sf(W, dfw)))


def within_inst(d: pd.DataFrame, ycol: str, xcol: str = "P", scale: float = 1.0) -> dict:
    """Pooled y = gamma_c + beta z_c(x) (+ institution FE); SE clustered by institution."""
    d = d.dropna(subset=[ycol, xcol]).copy()
    d = d[d.groupby("institution").cip2.transform("size") >= 2]
    d = d[d.groupby("cip2").institution.transform("size") >= NMIN]
    d = d.sort_values(["cip2", "institution"]).reset_index(drop=True)
    fams = sorted(d.cip2.unique())
    D = (d.cip2.to_numpy()[:, None] == np.array(fams)[None, :]).astype(float)
    z = zc(d, xcol)
    y = d[ycol].to_numpy(float) * scale
    groups = pd.factorize(d.institution)[0]
    out = dict(n_cells=len(d), n_inst=int(d.institution.nunique()), k=len(fams))
    for fe in (False, True):
        if fe:
            X = demean(np.column_stack([D[:, 1:], z]), groups); yy = demean(y, groups)
        else:
            X = np.column_stack([D, z]); yy = y
        if fe and xcol == "G":                                 # G is constant within institution
            out["fe"] = (np.nan, np.nan)
            continue
        f = sm.OLS(yy, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
        out["fe" if fe else "nofe"] = (float(f.params[-1]), float(f.bse[-1]))
    return out


def samp_W(c, emp_min=EMP_MIN, cover_min=None, pcol="P"):
    """W: Wapman-matched cells with family prestige, brand, a released p50 and >= emp_min employed."""
    x = c[c.project_family & (c.emp >= emp_min) & c.log_earn.notna() & c[pcol].notna() &
          c.G.notna() & c.S.notna()]
    if cover_min is not None:
        x = x[x.coverage >= cover_min]
    x = x[x.groupby("cip2").institution.transform("size") >= NMIN]
    return x.reset_index(drop=True)


def samp_WS(c, **kw):
    """WS: W with all Scorecard selectivity covariates."""
    x = samp_W(c, **kw)
    x = x.dropna(subset=SEL_CONT + SEL_CAT)
    return x[x.groupby("cip2").institution.transform("size") >= NMIN].reset_index(drop=True)


def samp_B(c, emp_min=EMP_MIN, project=True):
    """B: every PSEO institution with Scorecard SAT and admit rate (no Wapman requirement)."""
    x = c[(c.emp >= emp_min) & c.log_earn.notna() & c.S.notna() & c.SAT.notna() & c.NEG_ADM.notna()]
    if project:
        x = x[x.project_family]
    x = x[x.groupby("cip2").institution.transform("size") >= NMIN]
    return x.reset_index(drop=True)


def add_wage_mix(c: pd.DataFrame, med: pd.Series) -> None:
    """In place: C = log of the pay level of the cell's sector mix, sum_k share_k x ACS young-graduate median
    wage of sector k (20 sectors; ex-ante summary of the whole placement mix, third revision)."""
    v = sum(c[f"sh_{k}"].to_numpy(float) * float(med[k]) for k in SECTORS)
    c["C"] = np.log(np.where(v > 0, v, np.nan))


def expo_frame(d: pd.DataFrame, src: pd.DataFrame, emp_min: int = EMP_MIN, which: str = "x") -> pd.DataFrame:
    """Third revision: the y5 sample d restricted to cells whose institution x family also has >= emp_min employed
    graduates in the source cell table src (e.g. pooled y1, or one cohort at y1). which='x': the exposure shares
    (SHARE_COLS) are replaced by the source's shares; which='5': the y5 shares are kept (same subsample, for
    comparison). The y5 shares are always kept with suffix '5' (S5, M5, ...). Families need >= NMIN cells."""
    x = src[["institution", "cip2", "emp"] + SHARE_COLS].rename(
        columns={"emp": "emp_x", **{c: c + "_x" for c in SHARE_COLS}})
    j = d.merge(x, on=["institution", "cip2"], how="left")
    j = j[j.emp_x >= emp_min].copy()
    for c in SHARE_COLS:
        j[c + "5"] = j[c]
        if which == "x":
            j[c] = j[c + "_x"]
    j = j[j.groupby("cip2").institution.transform("size") >= NMIN]
    return j.sort_values(["cip2", "institution"]).reset_index(drop=True)


def timing_diag(d: pd.DataFrame, s: str = "SAT") -> dict:
    """Pooled within-family correlations (x-frame from expo_frame): y1 vs y5 shares, S1 vs M1, and status with
    the y1 shares, the y5 shares and the y1 -> y5 change (is the y5 share partly an outcome of status?)."""
    dd = d.sort_values(["cip2", "institution"]).reset_index(drop=True)
    z = zc(dd, s)

    def cen(v):
        return (dd[v] - dd.groupby("cip2")[v].transform("mean")).to_numpy(float)

    def r(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    S1, S5, M1, M5 = cen("S"), cen("S5"), cen("M"), cen("M5")
    return dict(S1_S5=r(S1, S5), M1_M5=r(M1, M5), S1_M1=r(S1, M1), S5_M5=r(S5, M5), z_S1=r(z, S1), z_S5=r(z, S5),
                z_M1=r(z, M1), z_M5=r(z, M5), z_dS=r(z, S5 - S1), z_dM=r(z, M5 - M1),
                sd_dS=float(np.std(S5 - S1, ddof=1) * 100), sd_dM=float(np.std(M5 - M1, ddof=1) * 100), n=len(dd))


def family_block(d: pd.DataFrame, specs: dict, tag: str, splits=()) -> dict:
    """Point estimates per family + joint institution bootstrap for one sample."""
    insts = sorted(d.institution.unique())
    iidx = {v: i for i, v in enumerate(insts)}
    fs = FamStats(d, specs, iidx)
    rows = []
    for c in fs.fams:
        st = fs.stats(c)
        st.update({f"split_{x}_{y}": fs.split_stat(c, x, y) for x, y in splits})
        st.update({f"psplit_{x}_{y}": fs.split_stat(c, x, y, partial=True) for x, y in splits})
        st.update({f"{kd}_{x}_{y}": fs.csplit_stat(c, x, y, by=by, cond=cond)
                   for kd, by, cond in CSPLIT_KINDS for x, y in splits})
        st.update(cip2=c, n=len(fs.data[c]["inst"]),
                  mean_S=float(d[d.cip2 == c].S.mean()), mean_M=float(d[d.cip2 == c].M.mean()))
        rows.append(st)
    point = pd.DataFrame(rows)
    bt = joint_bootstrap(fs, len(insts), "boot" + tag, splits)
    return dict(point=point, boot=bt, fams=fs.fams, n_inst=len(insts), n_cells=len(d))


REL_STATS_W = {k: STATS_W[k] for k in ["rP_M", "rG_M", "rP_S", "rP_E", "rG_E"]}
REL_STATS_WS = {k: STATS_WS[k] for k in ["rSAT_M_ws", "pP_M", "pG_M", "pP_E"]}
REL_STATS_B = {k: STATS_B[k] for k in ["rSAT_M_b", "rSAT_S_b", "rSAT_E_b"]}


def release_sensitivity(fams: list, sc: pd.DataFrame, brand: pd.DataFrame, W_old: pd.DataFrame) -> dict:
    """Primary pooled-y5 statistics on the V4.14.1 (2026Q2) release: flows, earnings and institutions all
    from V4.14.1; same project families, samples, statistics and random-stream tags (prefixed 'rel')."""
    sec, tot = read_flows(PSEOF_NEW)
    earn, c4, c2n = read_earnings(PSEOE_NEW)
    ins = pseo_institutions(PSEO_INST_NEW)
    prest, _ = family_prestige(c4, c2n, ins)
    cells = build_cells(sec, tot, earn, prest, ins, sc, brand, "0000", H_MAIN, fams)
    n_inst_flows = int(pd.Index(sorted(set(sec[sec.cohort != EXCL_COHORT].institution) | set(
        tot[tot.cohort.isin(COHORTS) & (tot.cohort != EXCL_COHORT)].institution))).nunique())
    del sec, tot, earn, c4, c2n
    cells["sample"] = "v4141_pooled_y5"
    W, WS, Bs = samp_W(cells), samp_WS(cells), samp_B(cells)
    out = dict(cells=cells, W=W, WS=WS, Bs=Bs, n_inst_flows=n_inst_flows,
               new_W_inst=sorted(set(W.institution) - set(W_old.institution)),
               lost_W_inst=sorted(set(W_old.institution) - set(W.institution)))
    out["new_W_labels"] = sorted(W[W.institution.isin(out["new_W_inst"])].label.unique())
    out["res_b"] = {"W": family_block(W, REL_STATS_W, "relW"), "WS": family_block(WS, REL_STATS_WS, "relWS"),
                    "B": family_block(Bs, REL_STATS_B, "relB")}
    out["SUM"] = {(tag, n): summarise(R["point"], R["boot"], R["fams"], n)
                  for tag, R in out["res_b"].items() for n in R["boot"]}
    for key_ri, ec in [("RI", ECURVE_MAIN), ("RIL", "lin")]:     # primary (family e^2) and linear-e versions
        out[key_ri] = {"P": fit_interaction(W, ["P"], "relP", ecurve=ec), "G": fit_interaction(W, ["G"], "relG", ecurve=ec),
                       "SAT_b": fit_interaction(Bs, ["SAT"], "relSAT_b", ecurve=ec),
                       "P_sel": fit_interaction(WS, ["P"], "relP_sel", extras=tuple(SEL_CONT + SEL_CAT), ecurve=ec),
                       "joint": fit_interaction(WS, ["P", "G", "SAT"], "reljoint", ecurve=ec),
                       "SAT_b_sel": fit_interaction(Bs.dropna(subset=SEL_CONT + SEL_CAT), ["SAT"], "relSAT_b_sel",
                                                    extras=("NEG_ADM", "PELL", "ST_EARN", "CONTROL"), ecurve=ec)}
    out["FF"] = ff_battery(W, Bs, "rel")
    # second revision: joint exposure models on the newer release
    out["JB"] = joint_battery(W, Bs, "rel")
    out["RJ"] = {}
    for key, d, s, xs in [("SAT_b_SM", Bs, "SAT", ("M",)), ("SAT_b_SML", Bs, "SAT", ("M", "L")),
                          ("P_SM", W, "P", ("M",)), ("P_SML", W, "P", ("M", "L")), ("G_SM", W, "G", ("M",))]:
        r = fit_interaction(d, [s], "rel" + key, expo2=xs)
        r["tests"] = {s: joint_tests(r, s)}
        out["RJ"][key] = _lean(r)
    return out


# =============================================================================================
# main
# =============================================================================================
def cohort_composition(tot: pd.DataFrame, earn: pd.DataFrame, Bs: pd.DataFrame, CELLS: dict) -> dict:
    """Fourth revision: which graduation cohorts the pooled cells hold (V4.13.0, bachelor's, institution x CIP-2).
    y5: released (status 1) all-sector y5 counts and y5 p50 by cohort; on the B cells, the pooled y5 employed total
    against the sum of the released cohort totals (all cohorts, and without 2013-15 and 2016-18), and the share of
    the pooled y5 employed total in the 2013-15 and 2016-18 cohorts. y1: share of the pooled y1 employed total in the
    2019-21 cohort (the only pooled-y1 cohort without y5), on the B cells of the pooled-y1 timing sample. Cohort
    counts that are not released count as 0 in the sums and shares."""
    out = {}
    coh = sorted(c for c in tot.cohort.unique() if c != "0000")
    out["cohorts"] = coh
    for h in ("y1", "y5"):
        v = f"emp_{h}"
        out[f"flows_rel_{h}"] = {c: int(tot[(tot.cohort == c)][v].notna().sum()) for c in coh}
        out[f"earn_rel_{h}"] = {c: int(earn[(earn.cohort == c)][f"p50_{h}"].notna().sum())
                                for c in sorted(earn.cohort.unique()) if c != "0000"}
        wide = tot.pivot_table(index=["institution", "cip2"], columns="cohort", values=v, aggfunc="sum")
        out[f"wide_{h}"] = wide
    w5 = out.pop("wide_y5")
    b = Bs[["institution", "cip2"]].merge(w5.reset_index(), on=["institution", "cip2"], how="left")
    b = b[b["0000"].notna() & (b["0000"] > 0)]
    cc = [c for c in coh if c in b.columns]
    s_all = b[cc].fillna(0.0).sum(axis=1)
    s_x = b[[c for c in cc if c not in ("2013", "2016")]].fillna(0.0).sum(axis=1)
    sh = b[[c for c in ("2013", "2016") if c in b.columns]].fillna(0.0).sum(axis=1) / b["0000"]
    out["y5_B"] = dict(n=len(b), n_B=len(Bs), ratio_all=float((b["0000"] / s_all).median()),
                       ratio_x=float((b["0000"] / s_x).median()), share_1316=float(sh.median()),
                       share_1316_p10=float(sh.quantile(0.1)), share_1316_p90=float(sh.quantile(0.9)),
                       n_both=int((b["2013"].notna() & b["2016"].notna()).sum()))
    w1 = out.pop("wide_y1")
    p1 = CELLS["pooled_y1"]                              # the cells of the pooled-y1 timing sample (as expo_frame)
    xb = Bs[["institution", "cip2"]].merge(p1[p1.emp >= EMP_MIN][["institution", "cip2"]], on=["institution", "cip2"])
    xb = xb[xb.groupby("cip2").institution.transform("size") >= NMIN]
    b1 = xb.merge(w1.reset_index(), on=["institution", "cip2"], how="left")
    b1 = b1[b1["0000"].notna() & (b1["0000"] > 0)]
    sh1 = b1[EXCL_COHORT].fillna(0.0) / b1["0000"] if EXCL_COHORT in b1.columns else pd.Series(0.0, index=b1.index)
    out["y1_x"] = dict(n=len(b1), share_19=float(sh1.median()), share_19_p10=float(sh1.quantile(0.1)),
                       share_19_p90=float(sh1.quantile(0.9)),
                       n_rel=int(b1[EXCL_COHORT].notna().sum()) if EXCL_COHORT in b1.columns else 0)
    return out


def _mem(msg: str) -> None:
    """Progress line with the process's peak resident memory so far (MB)."""
    import resource
    print(f"{msg} [peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB]", flush=True)


def main():
    _mem("reading PSEO flows ...")
    sec, tot = read_flows()
    _mem(f"flows read: {len(sec):,} sector rows, {len(tot):,} total rows")
    earn, c4, c2n = read_earnings()
    ins = pseo_institutions()
    sc = scorecard_institutions()
    brand = generic_brand()
    prest, nat_cover = family_prestige(c4, c2n, ins)
    fl_inst = sorted(set(sec[sec.cohort != EXCL_COHORT].institution) |       # the analysis cohorts, as before
                     set(tot[tot.cohort.isin(COHORTS) & (tot.cohort != EXCL_COHORT)].institution))
    opeid_match = (len(fl_inst), int(pd.Index(fl_inst).isin(sc.institution).sum()))
    fams = sorted(c for c, v in nat_cover.items() if v >= FAM_COVER)
    dropped_fams = sorted(c for c, v in nat_cover.items() if v < FAM_COVER)

    # consistency: sector counts sum to the released cell totals
    chk = {}
    for h in HZ:
        s = sec[sec.cohort == "0000"].groupby(["institution", "cip2"])[f"emp_{h}"].sum(min_count=1)
        t = tot[tot.cohort == "0000"].set_index(["institution", "cip2"])[f"emp_{h}"]
        j = pd.concat([s.rename("s"), t.rename("t")], axis=1).dropna()
        chk[h] = (len(j), float((j.s == j.t).mean()))

    # --- cells for every (cohort, horizon) -----------------------------------------------------
    samples = {"pooled_y1": ("0000", "y1"), "pooled_y5": ("0000", "y5"), "pooled_y10": ("0000", "y10"),
               "c2010_y1": ("2010", "y1"), "c2010_y5": ("2010", "y5"), "c2010_y10": ("2010", "y10"),
               "c2013_y1": ("2013", "y1"), "c2016_y1": ("2016", "y1"),
               "c2013_y5": ("2013", "y5"), "c2016_y5": ("2016", "y5")}      # fourth revision: own-cohort y5 pay
    CELLS = {}
    for k, (co, h) in samples.items():
        c = build_cells(sec, tot, earn, prest, ins, sc, brand, co, h, fams)
        c["sample"] = k
        CELLS[k] = c
    # fourth revision: pooled y1 shares without the 2019-21 cohort (exposure source only; no earnings attached)
    NO19 = {}
    for k, strict in [("pooled_y1_no19", True), ("pooled_y1_no19all", False)]:
        sx, NO19[k] = sec_pooled_minus(sec, tot, "y1", strict=strict)
        c = build_cells(sx, tot, earn, prest, ins, sc, brand, f"0000-{EXCL_COHORT}", "y1", fams)
        c["sample"] = k
        CELLS[k] = c
        del sx
    main_cells = CELLS["pooled_y5"]

    W, WS, Bs = samp_W(main_cells), samp_WS(main_cells), samp_B(main_cells)
    Ball = samp_B(main_cells, project=False)
    # flows employed total vs earnings-file count of graduates with earnings (same population by definition)
    cnt = {}
    for lab, d_ in [("all", main_cells[main_cells.emp > 0]), ("W", W)]:
        r_ = (d_.n_earn / d_.emp).dropna()
        cnt[lab] = (len(r_), float(r_.median()), float(r_.quantile(0.05)), float(r_.quantile(0.95)))
    chk["count_ratio"] = cnt
    COMP = cohort_composition(tot, earn, Bs, CELLS)
    print(f"W: {len(W)} cells / {W.institution.nunique()} inst / {W.cip2.nunique()} fam; "
          f"WS: {len(WS)} / {WS.institution.nunique()}; B: {len(Bs)} / {Bs.institution.nunique()}", flush=True)

    # --- national sector mix (all PSEO bachelor's cells, pooled y5) -----------------------------
    s5 = sec[(sec.cohort == "0000")].groupby("industry")["emp_y5"].sum()
    nat_mix = (s5 / s5.sum()).reindex(SECTORS)
    s5W = sec[(sec.cohort == "0000") & sec.set_index(["institution", "cip2"]).index.isin(
        W.set_index(["institution", "cip2"]).index)].groupby("industry")["emp_y5"].sum()
    W_mix = (s5W / s5W.sum()).reindex(SECTORS)
    del sec, tot, earn, s5, s5W                     # everything below works on the cell tables
    _mem("cells built")
    acs = acs_sectors()
    _mem("ACS read")
    # third revision: ex-ante pay level of each cell's sector mix (ACS young-graduate sector medians)
    med = acs[acs.sector.isin(SECTORS)].set_index("sector").med_wage.reindex(SECTORS)
    for c_ in list(CELLS.values()) + [W, WS, Bs, Ball]:
        add_wage_mix(c_, med)

    # --- (a) test-retest reliability of placement shares: cohort 2013 vs 2016 at y1 -------------
    a13, a16 = CELLS["c2013_y1"], CELLS["c2016_y1"]
    rr = a13[a13.project_family & (a13.emp >= EMP_MIN)].merge(
        a16[(a16.emp >= EMP_MIN)][["institution", "cip2", "S", "M", "log_earn"]],
        on=["institution", "cip2"], suffixes=("_a", "_b"))
    rel = {}
    for v in ["S", "M", "log_earn"]:
        vals, ks = [], 0
        for c, g in rr.groupby("cip2"):
            g = g.dropna(subset=[f"{v}_a", f"{v}_b"])
            if len(g) >= NMIN:
                vals.append(spear(g[f"{v}_a"].to_numpy(float), g[f"{v}_b"].to_numpy(float))); ks += 1
        rel[v] = (float(np.mean(vals)), float(np.min(vals)), ks, int(rr.dropna(subset=[f"{v}_a", f"{v}_b"]).shape[0]))
    # between-cell vs within-family: share of variance of S across cells explained by family
    vS = main_cells[main_cells.project_family & (main_cells.emp >= EMP_MIN)]
    eta2_S = float(1 - vS.groupby("cip2").S.transform(lambda s: s - s.mean()).var() / vS.S.var())
    eta2_M = float(1 - vS.groupby("cip2").M.transform(lambda s: s - s.mean()).var() / vS.M.var())

    # --- (b) placement coupling ---------------------------------------------------------------
    _mem("family statistics + bootstrap ...")
    res_b = {}
    for tag, d, specs, splits in [("W", W, STATS_W, [("P", "log_earn"), ("G", "log_earn")]),
                                  ("WS", WS, STATS_WS, []), ("B", Bs, STATS_B, [("SAT", "log_earn")]),
                                  ("Ball", Ball, STATS_B, [])]:
        res_b[tag] = family_block(d, specs, tag, splits)

    SUM = {}
    for tag in res_b:
        R = res_b[tag]
        for name in list(R["boot"]):
            SUM[(tag, name)] = summarise(R["point"], R["boot"], R["fams"], name)
    PAIR = {
        "W: rP_E - rP_M": paired(res_b["W"]["point"], res_b["W"]["boot"], res_b["W"]["fams"], "rP_E", "rP_M"),
        "W: rG_M - rP_M": paired(res_b["W"]["point"], res_b["W"]["boot"], res_b["W"]["fams"], "rG_M", "rP_M"),
        "W: rG_E - rP_E": paired(res_b["W"]["point"], res_b["W"]["boot"], res_b["W"]["fams"], "rG_E", "rP_E"),
        "WS: rSAT_M - rP_M": paired(res_b["WS"]["point"], res_b["WS"]["boot"], res_b["WS"]["fams"], "rSAT_M_ws", "rP_M_ws"),
        "WS: pP_E - pP_M": paired(res_b["WS"]["point"], res_b["WS"]["boot"], res_b["WS"]["fams"], "pP_E", "pP_M"),
        "WS: rP_M - pP_M": paired(res_b["WS"]["point"], res_b["WS"]["boot"], res_b["WS"]["fams"], "rP_M_ws", "pP_M"),
        "B: rSAT_E - rSAT_M": paired(res_b["B"]["point"], res_b["B"]["boot"], res_b["B"]["fams"], "rSAT_E_b", "rSAT_M_b"),
    }
    # cross-family ordering: placement coupling vs wage coupling (independent halves)
    _mem("split-half ordering ...")
    ORDER = split_half_order(W, res_b["W"]["fams"], [("rP_M", "rP_E"), ("rG_M", "rG_E"), ("rP_S", "rP_E")], "order")
    # legacy grain: family wage coupling vs family mean setting share
    pw = res_b["W"]["point"]
    legacy = dict(k=len(pw), rho=spear(pw.mean_S.to_numpy(), pw.rP_E.to_numpy()),
                  rho_G=spear(pw.mean_S.to_numpy(), pw.rG_E.to_numpy()),
                  rho_M=spear(pw.mean_M.to_numpy(), pw.rP_E.to_numpy()),
                  rho_SM=spear(pw.mean_S.to_numpy(), pw.mean_M.to_numpy()))
    pb = res_b["B"]["point"]
    legacy["k_b"], legacy["rho_b"] = len(pb), spear(pb.mean_S.to_numpy(), pb.rSAT_E_b.to_numpy())
    legacy["rho_b_M"] = spear(pb.mean_M.to_numpy(), pb.rSAT_E_b.to_numpy())
    # within-institution design
    # within-family collinearity of the status measures (pooled Pearson of within-family z-scores, WS cells)
    wz = WS.sort_values(["cip2", "institution"]).reset_index(drop=True)
    zP, zG, zS_ = zc(wz, "P"), zc(wz, "G"), zc(wz, "SAT")
    corr_zz = {"P_SAT": float(np.corrcoef(zP, zS_)[0, 1]), "P_G": float(np.corrcoef(zP, zG)[0, 1]),
               "G_SAT": float(np.corrcoef(zG, zS_)[0, 1])}
    WI = {"M": within_inst(W, "M", scale=100), "S": within_inst(W, "S", scale=100),
          "log_earn": within_inst(W, "log_earn"),
          "M_G": within_inst(W, "M", "G", 100), "E_G": within_inst(W, "log_earn", "G")}

    # --- (c)/(d) cell-level interaction models --------------------------------------------------
    _mem("interaction models ...")
    # every row is fitted twice: primary exposure shape (family-specific e^2, RI) and linear e (RIL, the
    # first version's primary); the report shows both so the reader can see which rows depend on the shape.
    RI, RIL = {}, {}

    def fit2(key, d, status, boot=False, jack=False, **kw):
        RI[key] = fit_interaction(d, status, key, boot=boot, jack=jack, **kw)
        RIL[key] = fit_interaction(d, status, key + "_lin", boot=boot, ecurve="lin", **kw)

    fit2("P", W, ["P"], boot=True, jack=True)
    fit2("G", W, ["G"], boot=True, jack=True)
    fit2("SAT_ws", WS, ["SAT"])
    fit2("SAT_b", Bs, ["SAT"], boot=True, jack=True)
    fit2("ADM_b", Bs, ["NEG_ADM"])
    fit2("SAT_ball", Ball, ["SAT"])
    fit2("joint", WS, ["P", "G", "SAT"])
    fit2("P_sel", WS, ["P"], extras=tuple(SEL_CONT + SEL_CAT), boot=True)
    fit2("P_sel_noint", WS, ["P"], extras=tuple(SEL_CONT + SEL_CAT), extras_int=False)
    fit2("G_sel", WS, ["G"], extras=tuple(SEL_CONT + SEL_CAT))
    fit2("P_ws", WS, ["P"])
    fit2("SAT_b_sel", Bs.dropna(subset=SEL_CONT + SEL_CAT), ["SAT"], extras=("NEG_ADM", "PELL", "ST_EARN", "CONTROL"))
    fit2("P_instfe", W, ["P"], inst_fe=True)
    fit2("SAT_b_instfe", Bs, ["SAT"], inst_fe=True)
    fit2("P_state", W, ["P"], state_fe=True)
    fit2("SAT_b_state", Bs, ["SAT"], state_fe=True)
    fit2("P_quad", W, ["P"], quad=True)
    fit2("P_wt", W, ["P"], weights="emp")
    fit2("P_rank", W, ["P"], rank_y=True)
    fit2("P_emp100", samp_W(main_cells, emp_min=100), ["P"])
    fit2("P_cov50", samp_W(main_cells, cover_min=0.5), ["P"])
    Wu = samp_W(main_cells, pcol="P_u").assign(P=lambda t: t.P_u)
    fit2("Pu", Wu, ["P"])
    fit2("P_satonly", WS, ["P"], extras=("SAT",))
    fit2("P_compM", W, ["P"], extras=("M",), extras_int=False)
    fit2("SAT_b_compM", Bs, ["SAT"], extras=("M",), extras_int=False)
    fit2("SAT_b_wt", Bs, ["SAT"], weights="emp")
    fit2("SAT_b_rank", Bs, ["SAT"], rank_y=True)
    fit2("SAT_b_emp100", samp_B(main_cells, emp_min=100), ["SAT"])
    fit2("SAT_b_quad", Bs, ["SAT"], quad=True)
    fit2("P_M", W, ["P"], expo="M")
    fit2("G_M", W, ["G"], expo="M")
    fit2("SAT_b_M", Bs, ["SAT"], expo="M")
    for k in SETTING:
        fit2(f"P_s{k}", W, ["P"], expo=f"sh_{k}")
        fit2(f"SAT_b_s{k}", Bs, ["SAT"], expo=f"sh_{k}")
    # functional-form battery on every horizon / cohort sample and (below) on V4.14.1
    FF = {}
    for k in ["pooled_y5", "pooled_y1", "pooled_y10", "c2010_y1", "c2010_y5", "c2010_y10"]:
        cw = samp_W(CELLS[k]); cb = samp_B(CELLS[k])
        FF[k] = ff_battery(cw, cb, "" if k == "pooled_y5" else k, lean=(k != "pooled_y5"))
        if k != "pooled_y5":
            RI[f"P_{k}"], RIL[f"P_{k}"] = FF[k].get(("P", ECURVE_MAIN)), FF[k].get(("P", "lin"))
            RI[f"SAT_b_{k}"], RIL[f"SAT_b_{k}"] = FF[k].get(("SAT_b", ECURVE_MAIN)), FF[k].get(("SAT_b", "lin"))
    DIAG = {key: curvature_diag(d, s_, FF["pooled_y5"][(key, "e2c")], FF["pooled_y5"][(key, "e2")])
            for key, d, s_ in [("P", W, "P"), ("G", W, "G"), ("SAT_b", Bs, "SAT")]}
    MIX = {"P": fit_mixed(W, "P"), "SAT_b": fit_mixed(Bs, "SAT")}
    # implied slope at within-family p10 / p90 of the setting share; zero crossing; by exposure shape
    IMPL = {key: {k: implied_slopes(FF["pooled_y5"][(key, k)], s_) for k in ("lin", "e2", "e3", "eq5", "e2z2")}
            for key, s_ in [("P", "P"), ("G", "G"), ("SAT_b", "SAT")]}
    for r in FF["pooled_y5"].values():            # free the coefficient vectors once used
        for x in ("V", "e"):
            r.pop(x, None)
    # --- (c') joint exposure models (second revision) ------------------------------------------
    _mem("joint exposure models ...")
    RJ = {}

    def fitj(key, d, status, expo2, boot=False, jack=False, wcr=True, **kw):
        r = fit_interaction(d, status, "J" + key, expo2=expo2, boot=boot, jack=jack, wcr=wcr, **kw)
        r["tests"] = {s: joint_tests(r, s, kw.get("expo", "S")) for s in status}
        RJ[key] = r

    fitj("SAT_b_SM", Bs, ["SAT"], ("M",), boot=True, jack=True)
    fitj("SAT_b_SL", Bs, ["SAT"], ("L",))
    fitj("SAT_b_SML", Bs, ["SAT"], ("M", "L"), boot=True)
    fitj("SAT_b_SML2", Bs, ["SAT"], ("M", "L2"))
    fitj("SAT_b_SM_state", Bs, ["SAT"], ("M",), state_fe=True)
    fitj("SAT_b_SM_instfe", Bs, ["SAT"], ("M",), inst_fe=True)
    fitj("SAT_b_SM_sel", Bs.dropna(subset=SEL_CONT + SEL_CAT), ["SAT"], ("M",),
         extras=("NEG_ADM", "PELL", "ST_EARN", "CONTROL"))
    fitj("SAT_ws_SM", WS, ["SAT"], ("M",))
    fitj("SAT_b_SM_wt", Bs, ["SAT"], ("M",), weights="emp")
    fitj("SAT_b_SM_rank", Bs, ["SAT"], ("M",), rank_y=True)
    fitj("SAT_b_SM_emp100", samp_B(main_cells, emp_min=100), ["SAT"], ("M",))
    fitj("SAT_ball_SM", Ball, ["SAT"], ("M",))
    fitj("ADM_b_SM", Bs, ["NEG_ADM"], ("M",))
    fitj("SAT_b_SMplus", Bs, ["SAT"], ("Mplus",))
    fitj("P_SM", W, ["P"], ("M",), boot=True, jack=True)
    fitj("P_SL", W, ["P"], ("L",))
    fitj("P_SML", W, ["P"], ("M", "L"))
    fitj("P_SM_instfe", W, ["P"], ("M",), inst_fe=True)
    fitj("P_SM_state", W, ["P"], ("M",), state_fe=True)
    fitj("P_sel_SM", WS, ["P"], ("M",), extras=tuple(SEL_CONT + SEL_CAT))
    fitj("G_SM", W, ["G"], ("M",))
    fitj("G_SML", W, ["G"], ("M", "L"))
    fitj("joint_SM", WS, ["P", "G", "SAT"], ("M",))
    RJ["joint_SM"]["tests_all"] = {
        "S_all": wald(RJ["joint_SM"], [[(f"theta_{s}", 1.0)] for s in ("P", "G", "SAT")]),
        "M_all": wald(RJ["joint_SM"], [[(f"theta_{s}[M]", 1.0)] for s in ("P", "G", "SAT")]),
        "S_sum": wald(RJ["joint_SM"], [[(f"theta_{s}", 1.0) for s in ("P", "G", "SAT")]]),
        "M_sum": wald(RJ["joint_SM"], [[(f"theta_{s}[M]", 1.0) for s in ("P", "G", "SAT")]])}
    for k in SETTING:
        fitj(f"SAT_b_s{k}_M", Bs, ["SAT"], ("M",), expo=f"sh_{k}")
        fitj(f"P_s{k}_M", W, ["P"], ("M",), expo=f"sh_{k}")
    JB = {}
    for k, _ in FF_SAMPLES:
        if k in CELLS:
            JB[k] = joint_battery(samp_W(CELLS[k]), samp_B(CELLS[k]), "" if k == "pooled_y5" else k)
    COL = {"SAT_b": interaction_collinearity(Bs, "SAT"), "P": interaction_collinearity(W, "P"),
           "G": interaction_collinearity(W, "G")}
    GB = {"SAT_b_M": gelbach(Bs, "SAT", ("M",)), "SAT_b_L": gelbach(Bs, "SAT", ("L",)),
          "SAT_b_ML": gelbach(Bs, "SAT", ("M", "L")), "P_M": gelbach(W, "P", ("M",)), "G_M": gelbach(W, "G", ("M",))}
    IMPLJ = {"SAT_b": implied_joint(RJ["SAT_b_SM"], Bs, "SAT"), "P": implied_joint(RJ["P_SM"], W, "P")}
    fitj("SAT_b_SC", Bs, ["SAT"], ("C",))           # third revision: S with the ACS sector-wage-mix index C (y5)

    # --- (c'') timing of the exposure: y5 outcome, sector shares measured at year 1 (third revision) ---------
    _mem("exposure timing (y1 shares) ...")
    JBX, JB5X, FFX, FF5X, RX, XN = {}, {}, {}, {}, {}, {}

    def fitx(key, d, status, expo2, boot=False, jack=False, wcr=True, **kw):
        r = fit_interaction(d, status, "X" + key, expo2=expo2, boot=boot, jack=jack, wcr=wcr, **kw)
        r["tests"] = {s: joint_tests(r, s, kw.get("expo", "S")) for s in status}
        RX[key] = r
    XFR = {}
    for k, _, src, base in X_ALL:
        if base == "pooled_y5":
            b_B, b_W, b_WS = Bs, W, WS
        else:                                      # fixed cohort: that cohort's own y5 cells as the outcome
            b_B, b_W, b_WS = samp_B(CELLS[base]), samp_W(CELLS[base]), samp_WS(CELLS[base])
        xb, xw = expo_frame(b_B, CELLS[src]), expo_frame(b_W, CELLS[src])
        xb5, xw5 = expo_frame(b_B, CELLS[src], which="5"), expo_frame(b_W, CELLS[src], which="5")
        xws = expo_frame(b_WS, CELLS[src])
        XN[k] = dict(B=(len(xb), xb.institution.nunique(), xb.cip2.nunique()),
                     W=(len(xw), xw.institution.nunique(), xw.cip2.nunique()),
                     WS=(len(xws), xws.institution.nunique(), xws.cip2.nunique()),
                     B_base=len(b_B), B_drop=len(b_B) - len(xb), W_drop=len(b_W) - len(xw))
        del b_B, b_W, b_WS
        JBX[k], JB5X[k] = joint_battery(xw, xb, k), joint_battery(xw5, xb5, k + "_5")
        FFX[k], FF5X[k] = ff_battery(xw, xb, k), ff_battery(xw5, xb5, k + "_5")
        fitx(f"SAT_SM_{k}", xb, ["SAT"], ("M",), boot=(k == X_MAIN), jack=(k == X_MAIN))
        fitx(f"SAT_SM5_{k}", xb5, ["SAT"], ("M",))
        fitx(f"SAT_SML_{k}", xb, ["SAT"], ("M", "L"))
        fitx(f"SAT_SML5_{k}", xb5, ["SAT"], ("M", "L"))
        fitx(f"SAT_SC_{k}", xb, ["SAT"], ("C",))
        fitx(f"SAT_SC5_{k}", xb5, ["SAT"], ("C",))
        fitx(f"SAT_SM_sel_{k}", xb.dropna(subset=SEL_CONT + SEL_CAT), ["SAT"], ("M",),
             extras=("NEG_ADM", "PELL", "ST_EARN", "CONTROL"))
        fitx(f"SAT_SM5_sel_{k}", xb5.dropna(subset=SEL_CONT + SEL_CAT), ["SAT"], ("M",),
             extras=("NEG_ADM", "PELL", "ST_EARN", "CONTROL"))
        fitx(f"P_SM_{k}", xw, ["P"], ("M",))
        fitx(f"G_SM_{k}", xw, ["G"], ("M",))
        XFR[k] = dict(xb=xb, xw=xw, xws=xws, xb5=xb5)
    # fourth revision: the pooled y1 shares WITH the 2019-21 cohort on exactly the cells of the strict no-2019 check
    kc = X_CHECKS[0][0]
    kc_keys = set(XFR[kc]["xb"].institution + "|" + XFR[kc]["xb"].cip2)
    fitx(f"SAT_SMP1_{kc}", expo_frame(Bs[(Bs.institution + "|" + Bs.cip2).isin(kc_keys)], CELLS["pooled_y1"]),
         ["SAT"], ("M",))
    # robustness rows on the pooled-y1 exposure (B cells unless stated); '5' = the y5 shares on the same cells
    xb, xw, xws, xb5 = (XFR[X_MAIN][c] for c in ("xb", "xw", "xws", "xb5"))
    k0 = X_MAIN
    for form in ("lin", "e3", "eq5", "e2z2", "rr"):
        fitx(f"SAT_SM_{form}_{k0}", xb, ["SAT"], ("M",), **dict(FF_SPECS[[f[0] for f in FF_SPECS].index(form)][2]))
    fitx(f"SAT_SM_state_{k0}", xb, ["SAT"], ("M",), state_fe=True)
    fitx(f"SAT_SM_instfe_{k0}", xb, ["SAT"], ("M",), inst_fe=True)
    fitx(f"SAT_SM5_instfe_{k0}", xb5, ["SAT"], ("M",), inst_fe=True)
    fitx(f"SAT_SM_ws_{k0}", xws, ["SAT"], ("M",))
    fitx(f"SAT_SM_wt_{k0}", xb, ["SAT"], ("M",), weights="emp")
    fitx(f"SAT_SM_rank_{k0}", xb, ["SAT"], ("M",), rank_y=True)
    fitx(f"SAT_SM_emp100_{k0}", expo_frame(samp_B(main_cells, emp_min=100), CELLS["pooled_y1"], emp_min=100),
         ["SAT"], ("M",))
    fitx(f"SAT_SM_ball_{k0}", expo_frame(Ball, CELLS["pooled_y1"]), ["SAT"], ("M",))
    fitx(f"ADM_SM_{k0}", xb, ["NEG_ADM"], ("M",))
    fitx(f"SAT_SMplus_{k0}", xb, ["SAT"], ("Mplus",))
    fitx(f"SAT_SM_y5also_{k0}", xb, ["SAT"], ("M", "S5", "M5"))
    fitx(f"SAT_SL_{k0}", xb, ["SAT"], ("L",))
    fitx(f"SAT_SML2_{k0}", xb, ["SAT"], ("M", "L2"))
    fitx(f"P_SM_instfe_{k0}", xw, ["P"], ("M",), inst_fe=True)
    fitx(f"P_SM_sel_{k0}", xws, ["P"], ("M",), extras=tuple(SEL_CONT + SEL_CAT))
    fitx(f"P_SML_{k0}", xw, ["P"], ("M", "L"))
    fitx(f"G_SML_{k0}", xw, ["G"], ("M", "L"))
    fitx(f"joint_SM_{k0}", xws, ["P", "G", "SAT"], ("M",))
    RX[f"joint_SM_{k0}"]["tests_all"] = {
        "S_all": wald(RX[f"joint_SM_{k0}"], [[(f"theta_{s}", 1.0)] for s in ("P", "G", "SAT")]),
        "M_all": wald(RX[f"joint_SM_{k0}"], [[(f"theta_{s}[M]", 1.0)] for s in ("P", "G", "SAT")]),
        "S_sum": wald(RX[f"joint_SM_{k0}"], [[(f"theta_{s}", 1.0) for s in ("P", "G", "SAT")]]),
        "M_sum": wald(RX[f"joint_SM_{k0}"], [[(f"theta_{s}[M]", 1.0) for s in ("P", "G", "SAT")]])}
    for k in SETTING:
        fitx(f"SAT_s{k}_M_{k0}", xb, ["SAT"], ("M",), expo=f"sh_{k}")
    RXA = {"SAT": fit_interaction(xb, ["SAT"], "XA_SAT", wcr=True),            # S1 alone (non-lean: implied slopes)
           "SAT5": fit_interaction(xb5, ["SAT"], "XA_SAT5", wcr=True)}
    IMPLX = {"alone": implied_slopes(RXA["SAT"], "SAT"), "alone5": implied_slopes(RXA["SAT5"], "SAT"),
             "joint": implied_joint(RX[f"SAT_SM_{k0}"], xb, "SAT")}
    GBX = {"SAT_M": gelbach(xb, "SAT", ("M",)), "SAT_M5": gelbach(xb5, "SAT", ("M",))}
    TDX = {k: timing_diag(XFR[k]["xb"]) for k, _, _, _ in X_ALL}
    # shape-free conditional median split with y1 shares, and with y5 shares on the same cells
    res_bx = {"x1": family_block(xb, {"rSAT_E_b": STATS_B["rSAT_E_b"]}, "Bx1", [("SAT", "log_earn")]),
              "x5": family_block(xb5, {"rSAT_E_b": STATS_B["rSAT_E_b"]}, "Bx5", [("SAT", "log_earn")])}
    SUMX = {(tag, n): summarise(R["point"], R["boot"], R["fams"], n) for tag, R in res_bx.items() for n in R["boot"]}
    for r in RX.values():                          # keep only what the report and CSV use
        for x_ in ("V", "e", "params", "names"):
            r.pop(x_, None)
    del XFR, xb, xw, xws, xb5
    _mem("exposure timing done")

    FT_P, FT_P_w = fam_theta_table(W, "P")
    FT_S, FT_S_w = fam_theta_table(Bs, "SAT")
    # leave-one-family-out
    LOFO = {}
    for key, d, s in [("P", W, "P"), ("G", W, "G"), ("SAT_b", Bs, "SAT")]:
        vals = []
        for c in sorted(d.cip2.unique()):
            r = fit_interaction(d[d.cip2 != c], [s], f"lofo{key}{c}")
            vals.append((c, r[s]["theta"], r[s]["se"], r[s]["ci"][1] < 0))
        LOFO[key] = vals
    # median-split coupling difference (high-S half minus low-S half), raw and with S partialled in each half
    SPLIT = {k: SUM[k] for k in SUM if k[1].startswith("split_") or k[1].startswith("psplit_")}

    # horizon/cohort robustness of placement coupling means (point estimates only)
    HROB = {}
    for k in ["pooled_y1", "pooled_y5", "pooled_y10", "c2010_y1", "c2010_y5", "c2010_y10"]:
        cw = samp_W(CELLS[k]); cb = samp_B(CELLS[k])
        pr = []
        for c, g in cw.groupby("cip2"):
            pr.append(dict(rP_M=spear(g.P.to_numpy(float), g.M.to_numpy(float)),
                           rG_M=spear(g.G.to_numpy(float), g.M.to_numpy(float)),
                           rP_E=spear(g.P.to_numpy(float), g.log_earn.to_numpy(float)),
                           rP_S=spear(g.P.to_numpy(float), g.S.to_numpy(float))))
        pb_ = [dict(rSAT_M=spear(g.SAT.to_numpy(float), g.M.to_numpy(float)),
                    rSAT_E=spear(g.SAT.to_numpy(float), g.log_earn.to_numpy(float))) for _, g in cb.groupby("cip2")]
        pr, pb_ = pd.DataFrame(pr), pd.DataFrame(pb_)
        HROB[k] = dict(kW=len(pr), nW=len(cw), iW=cw.institution.nunique(),
                       **{c: float(pr[c].mean()) for c in pr}, kB=len(pb_), nB=len(cb),
                       **{c: float(pb_[c].mean()) for c in pb_})

    _mem("release sensitivity (V4.14.1) ...")
    REL = release_sensitivity(fams, sc, brand, W)
    add_wage_mix(REL["cells"], med)                # CSV column only
    FF["v4141"] = REL["FF"]
    JB["v4141"] = REL["JB"]
    _mem("release sensitivity done")

    ctx = dict(REL=REL, chk=chk, opeid_match=opeid_match, nat_cover=nat_cover, fams=fams, dropped_fams=dropped_fams,
               CELLS=CELLS, W=W, WS=WS, Bs=Bs, Ball=Ball, nat_mix=nat_mix, W_mix=W_mix, acs=acs,
               rel=rel, eta2_S=eta2_S, eta2_M=eta2_M, res_b=res_b, SUM=SUM, PAIR=PAIR, ORDER=ORDER,
               legacy=legacy, WI=WI, RI=RI, RIL=RIL, FF=FF, DIAG=DIAG, MIX=MIX, FT_P=FT_P, FT_P_w=FT_P_w,
               FT_S=FT_S, FT_S_w=FT_S_w, LOFO=LOFO, SPLIT=SPLIT, HROB=HROB, IMPL=IMPL, corr_zz=corr_zz,
               RJ=RJ, JB=JB, COL=COL, GB=GB, IMPLJ=IMPLJ,
               RX=RX, RXA=RXA, JBX=JBX, JB5X=JB5X, FFX=FFX, FF5X=FF5X, XN=XN, IMPLX=IMPLX, GBX=GBX, TDX=TDX,
               res_bx=res_bx, SUMX=SUMX, med=med, COMP=COMP, NO19=NO19)
    write_csv(ctx)
    make_figure(ctx)
    write_md(ctx)
    print_summary(ctx)


# =============================================================================================
# outputs
# =============================================================================================
def write_csv(ctx):
    keep = ["level", "release", "sample", "cohort", "horizon", "institution", "label", "institution_state", "inst_key",
            "UNITID", "cip2", "family", "project_family", "emp", "emp_tot", "nme", "nme_share", "S", "M", "Mplus",
            "L", "L2", "C"]
    keep += [f"sh_{k}" for k in SECTORS]
    keep += ["p50", "n_earn", "log_earn", "P", "P_u", "k_w", "k_u", "coverage", "G", "SAT", "ADM_RATE",
             "PELL", "CONTROL", "ST_EARN", "in_W", "in_WS", "in_B"]
    parts = []
    REL = ctx["REL"]
    items = [(k, c, "V4.13.0", ctx) for k, c in ctx["CELLS"].items()]
    items.append(("v4141_pooled_y5", REL["cells"], "V4.14.1", REL))
    for k, c, rel, src in items:
        c = c.copy()
        c["level"], c["release"] = "cell", rel
        key = c.institution + "|" + c.cip2
        if k in ("pooled_y5", "v4141_pooled_y5"):
            c["in_W"] = key.isin(src["W"].institution + "|" + src["W"].cip2)
            c["in_WS"] = key.isin(src["WS"].institution + "|" + src["WS"].cip2)
            c["in_B"] = key.isin(src["Bs"].institution + "|" + src["Bs"].cip2)
        else:
            c["in_W"] = c["in_WS"] = c["in_B"] = np.nan
        c = c[c.emp.notna() & (c.emp > 0)]
        parts.append(c.reindex(columns=keep))
    cells = pd.concat(parts, ignore_index=True)
    # family rows (pooled y5): point estimates + bootstrap SE for every statistic
    fam_rows = []
    for lev, rel, smp, resb in [("family_", "V4.13.0", "pooled_y5", ctx["res_b"]),
                                ("family_rel_", "V4.14.1", "v4141_pooled_y5", REL["res_b"])]:
        for tag, R in resb.items():
            p = R["point"].set_index("cip2")
            for j, c in enumerate(R["fams"]):
                row = dict(level=f"{lev}{tag}", release=rel, sample=smp, cohort="0000", horizon="y5", cip2=c,
                           family=C2NAME.get(c, f"CIP-{c}"), n_cells=int(p.loc[c, "n"]),
                           mean_S=p.loc[c, "mean_S"], mean_M=p.loc[c, "mean_M"])
                for name in R["boot"]:
                    row[name] = p.loc[c, name]
                    row[f"{name}_se"] = float(np.nanstd(R["boot"][name][:, j], ddof=1))
                fam_rows.append(row)
    # third revision: the conditional-split family statistics on the pooled-y1 exposure cells (x1) and on the
    # y5 shares of the same cells (x5)
    for tag, R in ctx["res_bx"].items():
        p = R["point"].set_index("cip2")
        for j, c in enumerate(R["fams"]):
            row = dict(level=f"family_B{tag}", release="V4.13.0", sample="pooled_y5",
                       expo_source="pooled_y1" if tag == "x1" else "pooled_y5", cohort="0000", horizon="y5", cip2=c,
                       family=C2NAME.get(c, f"CIP-{c}"), n_cells=int(p.loc[c, "n"]),
                       mean_S=p.loc[c, "mean_S"], mean_M=p.loc[c, "mean_M"])
            for name in R["boot"]:
                row[name] = p.loc[c, name]
                row[f"{name}_se"] = float(np.nanstd(R["boot"][name][:, j], ddof=1))
            fam_rows.append(row)
    fam = pd.DataFrame(fam_rows)
    # theta rows (second revision): every interaction coefficient of the functional-form batteries (single
    # exposure: set 'S'; joint: 'SM', 'SL', 'SML') and of the joint robustness rows, with CR1 CIs; third
    # revision: 'expo_source' = cell table the shares come from (y1 shares for the x_* samples, whose outcome is
    # y5), the x5_* samples = y5 shares on the same cells as x_*, and the WCR p where computed
    th_rows = []
    smp_rel = {k: ("V4.14.1" if k == "v4141" else "V4.13.0") for k, _ in FF_SAMPLES}
    src_of = {k: k for k, _ in FF_SAMPLES}
    src_of["v4141"] = "v4141_pooled_y5"
    base_of = {}
    for k, _, src, base in X_ALL:
        src_of[k], src_of["x5" + k[1:]] = src, base
        base_of[k] = base
    stat_s = {"P": "P", "G": "G", "SAT_b": "SAT"}
    ff_all = {**ctx["FF"], **ctx["FFX"], **{"x5" + k[1:]: v for k, v in ctx["FF5X"].items()}}
    jb_all = {**ctx["JB"], **ctx["JBX"], **{"x5" + k[1:]: v for k, v in ctx["JB5X"].items()}}
    for smp, ff in ff_all.items():
        for (key, form), r in ff.items():
            v = r[stat_s[key]]
            th_rows.append(dict(level="theta_battery", release=smp_rel.get(smp, "V4.13.0"), sample=smp,
                                expo_source=src_of.get(smp, smp), spec=key,
                                status=stat_s[key], expo_set="S", form=form, exposure="S", n_cells=r["n_cells"],
                                n_inst=r["n_inst"], theta=v["theta"], theta_se=v["se"], theta_lo=v["ci"][0],
                                theta_hi=v["ci"][1], theta_p=v["p"]))
    for smp, jb in jb_all.items():
        for (key, sk, form), r in jb.items():
            for (s_, x), v in r["TH"].items():
                th_rows.append(dict(level="theta_battery", release=smp_rel.get(smp, "V4.13.0"), sample=smp,
                                    expo_source=src_of.get(smp, smp), spec=key,
                                    status=s_, expo_set=sk, form=form, exposure=x, n_cells=r["n_cells"],
                                    n_inst=r["n_inst"], theta=v["theta"], theta_se=v["se"], theta_lo=v["ci"][0],
                                    theta_hi=v["ci"][1], theta_p=v["p"]))
    rows_j = [("theta_joint_rows", key, r, "pooled_y5", "pooled_y5") for key, r in ctx["RJ"].items()]
    for key, r in ctx["RX"].items():                     # key ends with the x-sample; '5' rows use y5 shares
        k_ = next(k for k, _, _, _ in X_ALL if key.endswith(k))
        y5sh = key.split("_")[1].endswith("5")
        src_ = "pooled_y1" if key.startswith("SAT_SMP1_") else (base_of[k_] if y5sh else src_of[k_])
        rows_j.append(("theta_timing_rows", key, r, base_of[k_], src_))
    for key, r in ctx["RXA"].items():
        rows_j.append(("theta_timing_rows", "S_alone_" + key + "_" + X_MAIN, r, "pooled_y5",
                       "pooled_y5" if key.endswith("5") else "pooled_y1"))
    for lev, key, r, smp, src in rows_j:
        for (s_, x), v in r["TH"].items():
            th_rows.append(dict(level=lev, release="V4.13.0", sample=smp, expo_source=src, spec=key, status=s_,
                                expo_set="+".join([r["expo"]] + [x_ for (ss, x_) in r["TH"] if ss == s_ and x_ != r["expo"]]),
                                form=r["ecurve"], exposure=x, n_cells=r["n_cells"], n_inst=r["n_inst"],
                                theta=v["theta"], theta_se=v["se"], theta_lo=v["ci"][0], theta_hi=v["ci"][1],
                                theta_p=v["p"], theta_wcr_p=v.get("wcr_p", np.nan)))
    th = pd.DataFrame(th_rows)
    out = pd.concat([cells, fam, th], ignore_index=True)
    out = out.sort_values(["level", "sample", "cip2", "institution", "spec", "expo_set", "form", "status", "exposure"],
                          na_position="last", kind="mergesort").reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.8g")


def make_figure(ctx):
    W, R = ctx["W"], ctx["res_b"]["W"]
    p = R["point"].copy()
    p["label"] = p.cip2.map(lambda c: C2NAME.get(c, c))
    se = {k: np.nanstd(R["boot"][k], axis=0, ddof=1) for k in ["rP_M", "rP_E", "rG_M"]}
    C_SET, C_MKT, INK, MUTED = "#eb6834", "#2a78d6", "#0b0b0b", "#8a8984"
    fig = plt.figure(figsize=(13.5, 33.0))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.0, 1.0, 0.62, 0.95, 1.25])
    axs = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                    [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
    ax_e = fig.add_subplot(gs[2, :])
    ax_f, ax_g = fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])
    ax_h = fig.add_subplot(gs[4, :])
    # (a) sector mix per family (W sample, pooled y5)
    ax = axs[0, 0]
    q = p.sort_values("rP_E")
    y = np.arange(len(q))
    ax.barh(y, q.mean_S, color=C_SET, height=0.7, label="setting-priced (61+62+92)")
    ax.barh(y, q.mean_M, left=q.mean_S + 0.004, color=C_MKT, height=0.7, label="market-priced (51+52+54)")
    ax.set_yticks(y); ax.set_yticklabels(q.label, fontsize=8)
    ax.set_xlabel("mean share of employed graduates (y5, pooled cohorts)")
    ax.set_title("(a) sector exposure by field family\n(sorted by wage coupling ρ(P, log p50), top = highest)",
                 fontsize=10, loc="left")
    ax.set_xlim(0, 1.3); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(fontsize=8, loc="center right", frameon=False)
    # (b) placement vs wage coupling
    ax = axs[0, 1]
    ax.errorbar(p.rP_E, p.rP_M, xerr=1.96 * se["rP_E"], yerr=1.96 * se["rP_M"], fmt="o", ms=5,
                color=C_MKT, ecolor="#b9cfee", elinewidth=1, label="field-family prestige P")
    ax.plot(p.rG_E, p.rG_M, "s", ms=4, mfc="white", mec=INK, mew=0.8,
            label="academia-wide brand G (same cells; ρ(G, ·) on both axes)")
    lab_fams = {"Health", "Education", "Computer/Info", "Business", "Engineering", "Nat. Resources", "History"}
    for _, r in p.iterrows():
        if r.label in lab_fams:
            ax.annotate(r.label, (r.rP_E, r.rP_M), fontsize=7.5, xytext=(4, -9), textcoords="offset points", color=INK)
    lim = [-0.7, 0.9]
    ax.plot(lim, lim, color=MUTED, lw=0.7, ls="--"); ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlim(lim); ax.set_ylim(-0.35, 0.9)
    ax.set_xlabel("wage coupling ρ(status, log p50) within family")
    ax.set_ylabel("placement coupling ρ(status, market-sector share M)")
    ax.set_title("(b) non-wage placement coupling vs wage coupling (per family, 95% CI)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    # (c) marginal status slope vs setting exposure: primary (family e^2, solid + band) and linear e (dashed)
    ax = axs[1, 0]
    for key, s, col, lab in [("P", "P", C_MKT, "field-family prestige P (Wapman sample)"),
                             ("SAT_b", "SAT", C_SET, "SAT (all PSEO institutions with SAT)")]:
        dd = ctx["W"] if key == "P" else ctx["Bs"]
        e_ = (dd.S - dd.groupby("cip2").S.transform("mean")) / 0.10
        xs = np.linspace(np.percentile(e_, 5), np.percentile(e_, 95), 41)   # e in 10-pp units
        for src, ls, band in [("RI", "-", True), ("RIL", "--", False)]:
            r = ctx[src][key]
            names = r["names"]; par = r["params"]; V = r["V"]
            fams = r["fams"]; nc = r["nc"]
            c0 = np.zeros(len(names))
            for i, f in enumerate(fams):
                c0[names.index(f"b_{s}_{f}")] = nc[i] / nc.sum()
            jt = names.index(f"theta_{s}")
            est, lo, hi = [], [], []
            for x in xs:
                c = c0.copy(); c[jt] += x
                m, v = c @ par, c @ V @ c
                est.append(m); lo.append(m - 1.96 * np.sqrt(v)); hi.append(m + 1.96 * np.sqrt(v))
            ax.plot(xs * 10, est, color=col, lw=2 if band else 1.1, ls=ls,
                    label=lab + (" — primary (family e²)" if band else " — linear e (first version)"))
            if band:
                ax.fill_between(xs * 10, lo, hi, color=col, alpha=0.15, lw=0)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("setting-priced share, deviation from family mean (pp; 5th-95th percentile of cells)")
    ax.set_ylabel("log-earnings slope per within-family SD of status")
    ax.set_title("(c) pay-for-status gradient vs exposure to setting-priced sectors\n(cell-level model; band = CR1 95% CI of the primary fit)",
                 fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    # (d) theta across specifications (primary exposure shape; open square = same row with linear e)
    ax = axs[1, 1]
    specs = spec_rows(ctx)
    yy = np.arange(len(specs))[::-1]
    for yv, (lab, th, lo, hi, grp, thl) in zip(yy, specs):
        col = C_MKT if grp == "P" else (C_SET if grp == "SAT" else INK)
        ax.plot([lo, hi], [yv, yv], color=col, lw=1.4)
        ax.plot(th, yv, "o", color=col, ms=4.5, mfc=col if grp != "G" else "white")
        if np.isfinite(thl):
            ax.plot(thl, yv, "s", ms=3.5, mfc="none", mec=MUTED, mew=0.8)
    ax.set_yticks(yy); ax.set_yticklabels([s[0] for s in specs], fontsize=7)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_xlabel("θ: change in log-earnings slope per SD status, per +10 pp setting-priced share")
    ax.set_title("(d) θ across specifications, primary exposure shape (95% CI)\n(open grey square: same row with a linear e main effect)",
                 fontsize=10, loc="left")
    # (e) functional-form battery, y5 pooled
    ax = ax_e
    FFm = ctx["FF"]["pooled_y5"]
    xk = np.arange(len(FF_SPECS))
    for off, (key, s, col, lab, mfc) in zip([-0.2, 0.0, 0.2], [("P", "P", C_MKT, "prestige P (W)", C_MKT),
                                                              ("G", "G", INK, "brand G (W)", "white"),
                                                              ("SAT_b", "SAT", C_SET, "SAT (B)", C_SET)]):
        for i, (k, _, _) in enumerate(FF_SPECS):
            r = FFm.get((key, k))
            if r is None:
                continue
            v = r[s]
            ax.plot([i + off, i + off], v["ci"], color=col, lw=1.4)
            ax.plot(i + off, v["theta"], "o", color=col, mfc=mfc, ms=5, label=lab if i == 0 else None)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.axvline(len(FF_SPECS) - 2.5, color=MUTED, lw=0.6, ls=":")
    short = {"lin": "linear e\n(1st version)", "e2c": "+ common e²", "e2": "+ family e²\n(primary)",
             "e3": "+ family\ne², e³", "eq5": "+ family ×\ne-quintile", "eq5zq5": "+ family ×\ne- and status-\nquintile",
             "e2z2": "+ family e²\n+ status²", "rr_lin": "rank × rank,\nlinear", "rr": "rank × rank,\n+ (rank S)²"}
    ax.set_xticks(xk); ax.set_xticklabels([short[k] for k, _, _ in FF_SPECS], fontsize=8)
    ax.set_ylabel("θ (95% CR1 CI)")
    ax.set_title("(e) θ by functional form of the exposure / status main effects (y5 pooled). Right of the dotted line: "
                 "status and S as within-family standardised ranks (θ per SD of rank × SD of rank)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    # (f) second revision: SAT x S alone vs jointly with SAT x M (and SAT x L), by functional form, y5 pooled
    ax = ax_f
    JBm = ctx["JB"]["pooled_y5"]
    forms = [k for k, _, _ in JFF_SPECS]
    series = [("S alone", lambda k: FFm.get(("SAT_b", k), {}).get("SAT"), C_SET, "white", -0.27),
              ("S | M", lambda k: JBm[("SAT_b", "SM", k)]["TH"][("SAT", "S")], C_SET, C_SET, -0.09),
              ("M | S", lambda k: JBm[("SAT_b", "SM", k)]["TH"][("SAT", "M")], C_MKT, C_MKT, 0.09),
              ("L | S, M", lambda k: JBm[("SAT_b", "SML", k)]["TH"][("SAT", "L")], MUTED, MUTED, 0.27)]
    for lab, get, col, mfc, off in series:
        for i, k in enumerate(forms):
            v = get(k)
            if v is None:
                continue
            ax.plot([i + off, i + off], v["ci"], color=col, lw=1.2)
            ax.plot(i + off, v["theta"], "o", color=col, mfc=mfc, ms=4.5, label=f"θ(SAT × {lab})" if i == 0 else None)
    ax.axhline(0, color=MUTED, lw=0.7)
    short_f = {"lin": "linear e", "e2c": "+ common e²", "e2": "+ family e² (primary)", "e3": "+ family e², e³",
               "eq5": "+ family × e-quintile", "eq5zq5": "+ family × e-, status-quintile", "e2z2": "+ family e², status²",
               "rr_lin": "rank × rank, linear", "rr": "rank × rank, + (rank S)²", "e2x": "+ family e², e_S·e_M"}
    ax.set_xticks(np.arange(len(forms)))
    ax.set_xticklabels([short_f[k] for k in forms], fontsize=7.5, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("θ (95% CR1 CI)")
    ax.set_title("(f) SAT × setting share, alone and jointly with SAT × market share\n(B cells, y5 pooled; "
                 "'| M' = model also has SAT × M and M's own main-effect shape)", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.30))
    # (g) joint S + M model across samples and specifications (primary shape)
    ax = ax_g
    rows_g = joint_forest_rows(ctx)
    yy = np.arange(len(rows_g))[::-1]
    for yv, (lab, vS, vM) in zip(yy, rows_g):
        for v, col, dy in [(vS, C_SET, 0.14), (vM, C_MKT, -0.14)]:
            ax.plot(v["ci"], [yv + dy, yv + dy], color=col, lw=1.3)
            ax.plot(v["theta"], yv + dy, "o", color=col, ms=4)
    ax.plot([], [], "o-", color=C_SET, ms=4, label="θ(status × S | M)")
    ax.plot([], [], "o-", color=C_MKT, ms=4, label="θ(status × M | S)")
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows_g], fontsize=7)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_xlabel("θ per SD status, per +10 pp share (95% CR1 CI)")
    ax.set_title("(g) joint S + M model across samples and specifications\n(primary shape: family e² for each share)",
                 fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    # (h) third revision: when the shares are measured (y5 outcome; shares at y5 vs at year 1)
    ax = ax_h
    rows_h = timing_forest_rows(ctx)
    yy = np.arange(len(rows_h))[::-1]
    for yv, (lab, vS, vM, y5sh) in zip(yy, rows_h):
        if y5sh:
            ax.axhspan(yv - 0.45, yv + 0.45, color="#efefec", lw=0)
        for v, col, dy in [(vS, C_SET, 0.15), (vM, C_MKT, -0.15)]:
            ax.plot(v["ci"], [yv + dy, yv + dy], color=col, lw=1.3)
            ax.plot(v["theta"], yv + dy, "o", color=col, ms=4, mfc=col if not y5sh else "white")
    ax.plot([], [], "o-", color=C_SET, ms=4, label="θ(SAT × S | M)")
    ax.plot([], [], "o-", color=C_MKT, ms=4, label="θ(SAT × M | S)")
    ax.plot([], [], "s", color="#efefec", ms=9, label="shares measured at y5 (open markers)")
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows_h], fontsize=7.5)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_xlabel("θ per SD of SAT, per +10 pp share (95% CR1 CI); outcome = log p50 at y5, B cells")
    DC = _design_counts(ctx)
    t1 = ctx["RX"][f"SAT_SM_{X_MAIN}"]["TH"]
    ax.set_title("(h) y5 earnings cells with the sector shares measured at y5 (shaded rows, open markers) or at year 1 "
                 "(filled markers)\n"
                 f"primary shape; S | M < 0 / M | S > 0 in: y5 shares {DC['n5S']} / {DC['n5M']} of {DC['n5']}; pooled y1 "
                 f"shares {int(DC['pooledS'])} / {int(DC['pooledM'])} of 1; own-cohort y1 → y5 {DC['ownS']} / "
                 f"{DC['ownM']} of {DC['n_own']}; one cohort's y1 shares, pooled y5 pay {DC['mixS']} / {DC['mixM']} of "
                 f"{DC['n_mix']}", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for a in list(axs.flat) + [ax_e, ax_f, ax_g, ax_h]:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def spec_rows(ctx):
    RI, RIL = ctx["RI"], ctx["RIL"]
    rows = []
    def add(lab, key, s, grp):
        r = RI.get(key)
        if r is None or s not in r:
            return
        rl = RIL.get(key)
        thl = rl[s]["theta"] if rl is not None and s in rl else np.nan
        rows.append((lab, r[s]["theta"], r[s]["ci"][0], r[s]["ci"][1], grp, thl))
    add("P × S (primary, Wapman cells)", "P", "P", "P")
    add("P × S, selectivity-adjusted", "P_sel", "P", "P")
    add("P × S, institution FE", "P_instfe", "P", "P")
    add("P × S, state FE", "P_state", "P", "P")
    add("P × S, + P² by family", "P_quad", "P", "P")
    add("P × S, weighted by employed", "P_wt", "P", "P")
    add("P × S, rank outcome", "P_rank", "P", "P")
    add("P × S, cells ≥100 employed", "P_emp100", "P", "P")
    add("P × S, ranked coverage ≥0.5", "P_cov50", "P", "P")
    add("P × S, y1 pooled", "P_pooled_y1", "P", "P")
    add("P × S, y10 pooled", "P_pooled_y10", "P", "P")
    add("P × S, cohort 2010 y5", "P_c2010_y5", "P", "P")
    add("P × S, + M main effects only (no P × M)", "P_compM", "P", "P")
    add("joint P, G, SAT: P × S", "joint", "P", "P")
    add("G × S (brand)", "G", "G", "G")
    add("G × S, selectivity-adjusted", "G_sel", "G", "G")
    add("SAT × S (all PSEO inst.)", "SAT_b", "SAT", "SAT")
    add("SAT × S, + ADM/Pell/control/state", "SAT_b_sel", "SAT", "SAT")
    add("SAT × S, + M main effects only (no SAT × M)", "SAT_b_compM", "SAT", "SAT")
    add("joint P, G, SAT: SAT × S (WS cells)", "joint", "SAT", "SAT")
    add("SAT × S, institution FE", "SAT_b_instfe", "SAT", "SAT")
    add("SAT × S, state FE", "SAT_b_state", "SAT", "SAT")
    add("SAT × S, y1 pooled", "SAT_b_pooled_y1", "SAT", "SAT")
    add("SAT × S, y10 pooled", "SAT_b_pooled_y10", "SAT", "SAT")
    add("SAT × S, cohort 2010 y5", "SAT_b_c2010_y5", "SAT", "SAT")
    add("−ADM × S (all PSEO inst.)", "ADM_b", "NEG_ADM", "SAT")
    return rows


def joint_forest_rows(ctx):
    """(label, theta_S | M, theta_M | S) for the joint S + M model (primary shape) across samples / specs."""
    RJ, JB = ctx["RJ"], ctx["JB"]
    rows = []

    def add(lab, r, s):
        if r is not None and (s, "S") in r["TH"] and (s, "M") in r["TH"]:
            rows.append((lab, r["TH"][(s, "S")], r["TH"][(s, "M")]))
    add("SAT, y5 pooled (B, primary)", RJ.get("SAT_b_SM"), "SAT")
    add("SAT, state FE", RJ.get("SAT_b_SM_state"), "SAT")
    add("SAT, institution FE", RJ.get("SAT_b_SM_instfe"), "SAT")
    add("SAT, + ADM/Pell/state/control (× S, × M)", RJ.get("SAT_b_SM_sel"), "SAT")
    add("SAT, WS cells (Wapman-matched)", RJ.get("SAT_ws_SM"), "SAT")
    add("SAT, WLS by employed", RJ.get("SAT_b_SM_wt"), "SAT")
    add("SAT, cells ≥ 100 employed", RJ.get("SAT_b_SM_emp100"), "SAT")
    add("SAT, all CIP-2 families", RJ.get("SAT_ball_SM"), "SAT")
    for k, lab in FF_SAMPLES[1:]:
        add(f"SAT, {lab}", JB.get(k, {}).get(("SAT_b", "SM", ECURVE_MAIN)), "SAT")
    add("P, y5 pooled (W)", RJ.get("P_SM"), "P")
    add("P, institution FE", RJ.get("P_SM_instfe"), "P")
    add("P, selectivity-adjusted (WS)", RJ.get("P_sel_SM"), "P")
    for k, lab in FF_SAMPLES[1:]:
        add(f"P, {lab}", JB.get(k, {}).get(("P", "SM", ECURVE_MAIN)), "P")
    add("G, y5 pooled (W)", RJ.get("G_SM"), "G")
    return rows


TIMING_ROBUST = [("lin", "linear shares"), ("e3", "+ family e², e³"), ("eq5", "+ family × share-quintile"),
                 ("e2z2", "+ family SAT²"), ("rr", "rank × rank + (rank share)²"), ("state", "+ state FE"),
                 ("instfe", "+ institution FE"), ("ws", "WS cells (Wapman-matched)"), ("wt", "WLS by employed"),
                 ("rank", "outcome = within-family rank"), ("emp100", "cells ≥ 100 employed (y1 and y5)"),
                 ("ball", "all CIP-2 families"), ("y5also", "+ y5 S, M and SAT × y5 S, M")]


def timing_forest_rows(ctx):
    """(label, theta_S | M, theta_M | S, shares-at-y5?) for figure panel (h)."""
    RJ, RX = ctx["RJ"], ctx["RX"]
    rows = [("y5 shares, all B cells (section (c') primary)", RJ["SAT_b_SM"]["TH"][("SAT", "S")],
             RJ["SAT_b_SM"]["TH"][("SAT", "M")], True)]
    for k, lab, _, _ in [X_SAMPLES[0]] + X_CHECKS[:1] + X_SAMPLES[1:]:
        chk_ = k in [c[0] for c in X_CHECKS]
        pairs = [(f"SAT_SM_{k}", "check: " + lab.replace(": y1 shares, y5 pay", ""), False)] if chk_ else \
            [(f"SAT_SM5_{k}", f"y5 shares, cells of '{lab}'", True), (f"SAT_SM_{k}", lab, False)]
        for key, l_, y5 in pairs:
            rows.append((l_, RX[key]["TH"][("SAT", "S")], RX[key]["TH"][("SAT", "M")], y5))
    for key, l_, y5 in [(f"SAT_SM5_sel_{X_MAIN}", "y5 shares, + −ADM, Pell, state, control (× shares)", True),
                        (f"SAT_SM_sel_{X_MAIN}", "pooled y1 shares, + −ADM, Pell, state, control (× shares)", False),
                        (f"SAT_SM5_instfe_{X_MAIN}", "y5 shares, + institution FE", True)]:
        rows.append((l_, RX[key]["TH"][("SAT", "S")], RX[key]["TH"][("SAT", "M")], y5))
    for f_, lab in TIMING_ROBUST:
        r = RX[f"SAT_SM_{f_}_{X_MAIN}"]
        rows.append((f"pooled y1 shares, {lab}" + (" (rank scale)" if f_ == "rank" else ""), r["TH"][("SAT", "S")],
                     r["TH"][("SAT", "M")], False))
    return rows


def _f(x, nd=2, sign=True):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _ci(c, nd=2):
    return f"[{_f(c[0], nd)}, {_f(c[1], nd)}]"


def _p(p):
    if p is None or not np.isfinite(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _peq(p):
    return "p < 0.001" if (p is not None and np.isfinite(p) and p < 0.001) else f"p = {_p(p)}"


def write_md(ctx):
    SUM, PAIR, RI, W, WS, Bs = ctx["SUM"], ctx["PAIR"], ctx["RI"], ctx["W"], ctx["WS"], ctx["Bs"]
    res_b = ctx["res_b"]
    def S(tag, name):
        return SUM[(tag, name)]
    def th(key, s):
        return RI[key][s]
    rP, rG, rPE = S("W", "rP_M"), S("W", "rG_M"), S("W", "rP_E")
    rSb, rSEb = S("B", "rSAT_M_b"), S("B", "rSAT_E_b")
    tP, tG, tS = th("P", "P"), th("G", "G"), th("SAT_b", "SAT")
    tPs = th("P_sel", "P")
    sdS_W = RI["P"]["sd_e"] * 10
    sdS_B = RI["SAT_b"]["sd_e"] * 10
    L = []
    L.append("# PSEO Flows as a placement axis: sector placement and the pay-for-status gradient\n")
    L.append("Public data only (Census PSEO Flows and Earnings, Wapman faculty-hiring ranks, College Scorecard "
             "institution file, ACS PUMS 2023). Descriptive; no causal claim. Every number below is produced by "
             f"`scripts/61_flows_placement.py` (seed {SEED}; joint institution bootstrap B = {B_BOOT} for family "
             f"statistics, institution pairs-cluster bootstrap B = {B_CLU} for key interactions, wild-cluster "
             f"restricted bootstrap B = {B_WCR} for the joint and timing rows, {N_PERM} family-label "
             "permutations). Cell and family table: "
             "`data/interim/flows_placement.csv`; figure: `outputs/figures/flows_placement.png`.\n")
    # ------------------------------------------------------------------ answer
    L.append("## Answer\n")
    L.append(_answer1(ctx))
    L.append(_answer2(ctx))
    L.append(_answer_detail(ctx))
    L.append(_revision_md(ctx))
    # ------------------------------------------------------------------ numbers
    L.append("## Key numbers\n")
    L.append("Samples (pooled cohorts, y5 unless stated; cells need ≥ "
             f"{EMP_MIN} employed graduates, a released PSEO p50, and families ≥ {NMIN} cells). "
             f"**W** = Wapman-matched PSEO institutions with family prestige P and brand G: {len(W)} cells, "
             f"{W.institution.nunique()} institutions, {W.cip2.nunique()} families. **WS** = W with Scorecard SAT, "
             f"admit rate, Pell, control, state level: {len(WS)} cells, {WS.institution.nunique()} institutions, "
             f"{WS.cip2.nunique()} families. **B** = all PSEO institutions with Scorecard SAT and admit rate in "
             f"project families: {len(Bs)} cells, {Bs.institution.nunique()} institutions, {Bs.cip2.nunique()} "
             "families. CIs for means across families: joint institution bootstrap (institutions resampled once "
             "per draw for all families). 'sig. +/−' = families whose own bootstrap 95% CI excludes 0.\n")
    L.append("### (b) Placement coupling vs wage coupling, within family\n")
    L.append("| statistic | sample | k fam. | mean | 95% CI | median | fam. >0 | sig. + / − |")
    L.append("|---|---|---|---|---|---|---|---|")
    for tag, names in [("W", ["rP_M", "rG_M", "rP_Mplus", "rP_S", "rG_S", "rP_E", "rG_E", "rM_E", "rS_E", "rP_nme"]),
                       ("WS", ["rP_M_ws", "rG_M_ws", "rSAT_M_ws", "rP_E_ws", "rG_E_ws", "rSAT_E_ws",
                               "pP_M", "pG_M", "pP_S", "pP_E", "pG_E"]),
                       ("B", ["rSAT_M_b", "rADM_M_b", "rSAT_S_b", "rADM_S_b", "rSAT_E_b", "rADM_E_b", "rM_E_b", "rS_E_b"]),
                       ("Ball", ["rSAT_M_b", "rSAT_S_b", "rSAT_E_b"])]:
        for n in names:
            s = S(tag, n)
            samp = {"W": "W", "WS": "WS", "B": "B", "Ball": "B, all CIP-2 families"}[tag]
            L.append(f"| {STAT_LABEL[n]} | {samp} | {s['k']} | {_f(s['mean'])} | {_ci(s['ci'])} | {_f(s['median'])} | "
                     f"{s['n_gt0']}/{s['k']} | {s['n_pos']} / {s['n_neg']} |")
    L.append("\n'sel.' = partial Spearman net of ranked SAT_AVG, −ADM_RATE, Pell share, state earnings level and "
             "control dummies (the 'broad' spec of scripts/55).\n")
    L.append("| paired contrast (mean over families) | k | difference | 95% CI | families where first > second |")
    L.append("|---|---|---|---|---|")
    for k, v in PAIR.items():
        L.append(f"| {k} | {v['k']} | {_f(v['diff'])} | {_ci(v['ci'])} | {v['n_a_gt']}/{v['k']} |")
    L.append("")
    L.append("Cross-family ordering (does a family with stronger placement coupling also have stronger wage "
             f"coupling?). Independent halves of the W institutions ({K_SPLIT} random splits × 2 directions; "
             f"families need ≥ {NMIN_HALF} cells per half); two-sided p from {N_PERM} family-label permutations.\n")
    L.append("| pair | k fam. | same-sample Spearman | split-half Spearman | null mean | p |")
    L.append("|---|---|---|---|---|---|")
    for (a, b), v in ctx["ORDER"].items():
        L.append(f"| {STAT_LABEL[a]} vs {STAT_LABEL[b]} | {v['k']} | {_f(v['same'])} | {_f(v['cross'])} | "
                 f"{_f(v['null_mean'])} | {_p(v['p'])} |")
    WI = ctx["WI"]
    L.append("\nWithin-institution design (pooled over W families; y = β z_c(status) + family FE, ± institution FE; "
             "SE clustered by institution; cells in institutions with ≥ 2 families):\n")
    L.append("| outcome | status | cells / inst. | β without inst. FE (SE) | β with inst. FE (SE) |")
    L.append("|---|---|---|---|---|")
    for key, lab, stat, unit in [("M", "market share M (pp)", "P", ""), ("S", "setting share S (pp)", "P", ""),
                                 ("log_earn", "log p50", "P", ""), ("M_G", "market share M (pp)", "G", ""),
                                 ("E_G", "log p50", "G", "")]:
        v = WI[key]
        fe = (f"{_f(v['fe'][0], 3)} ({v['fe'][1]:.3f})" if np.isfinite(v['fe'][0])
              else "n/a (G constant within institution)")
        L.append(f"| {lab} | {stat} | {v['n_cells']} / {v['n_inst']} | {_f(v['nofe'][0], 3)} ({v['nofe'][1]:.3f}) | {fe} |")
    L.append("")
    # ------------------------------------------------------------------ (c)
    L.append(_interaction_md(ctx))
    L.append(_joint_md(ctx))
    L.append(_timing_md(ctx))
    lg = ctx["legacy"]
    L.append(f"\nFamily grain (the old ~20-unit test, for comparison): Spearman across families of family mean S with "
             f"wage coupling ρ(P, log p50) = {_f(lg['rho'])} (k = {lg['k']}), with ρ(G, log p50) = {_f(lg['rho_G'])}; "
             f"with ρ(SAT, log p50) on B = {_f(lg['rho_b'])} (k = {lg['k_b']}). Family mean M with ρ(P, log p50): "
             f"{_f(lg['rho_M'])}; with ρ(SAT, log p50) on B: {_f(lg['rho_b_M'])}; family mean S with family mean M: "
             f"{_f(lg['rho_SM'])}, so at this grain S and M cannot be told apart either. No inference attached (k small, "
             "shared institutions).\n")
    # horizon robustness of coupling means
    L.append("Placement and wage coupling by horizon / cohort (point estimates, mean over families):\n")
    L.append("| sample | W cells / inst. / fam. | ρ(P, M) | ρ(G, M) | ρ(P, S) | ρ(P, log p50) | B cells / fam. | ρ(SAT, M) | ρ(SAT, log p50) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, v in ctx["HROB"].items():
        L.append(f"| {k} | {v['nW']} / {v['iW']} / {v['kW']} | {_f(v.get('rP_M'))} | {_f(v.get('rG_M'))} | {_f(v.get('rP_S'))} | "
                 f"{_f(v.get('rP_E'))} | {v['nB']} / {v['kB']} | {_f(v.get('rSAT_M'))} | {_f(v.get('rSAT_E'))} |")
    L.append("")
    L.append(_release_md(ctx))
    # ------------------------------------------------------------------ (a)
    L.append("### (a) The placement measure\n")
    acs = ctx["acs"].set_index("sector")
    L.append("Sector classes, fixed before any coupling was computed. ACS 2023 PUMS: wage/salary employees aged "
             "22–40 with a bachelor's degree or more, full-time (≥ 35 h) full-year (≥ 50 weeks), military excluded; "
             "person-weighted. PSEO columns: share of employed bachelor's graduates at y5 (pooled cohorts) in the "
             "sector, all PSEO institutions × CIP-2, and in the W cells.\n")
    L.append("| NAICS | sector | class | ACS: government employer | ACS: government or non-profit | ACS: median wage | ACS n | PSEO share (all) | PSEO share (W) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k in SECTORS + ["[setting]", "[market]", "[other]"]:
        if k not in acs.index:
            continue
        a = acs.loc[k]
        nm = SECTOR_NAME.get(k, k.strip("[]") + " class")
        cls_sec = {"[setting]": SETTING, "[market]": MARKET,
                   "[other]": [x for x in SECTORS if x not in SETTING + MARKET]}.get(k)
        if cls_sec is not None:
            ps, pw_ = f"{ctx['nat_mix'][cls_sec].sum():.3f}", f"{ctx['W_mix'][cls_sec].sum():.3f}"
        else:
            ps = f"{ctx['nat_mix'][k]:.3f}" if k in ctx["nat_mix"].index else ""
            pw_ = f"{ctx['W_mix'][k]:.3f}" if k in ctx["W_mix"].index else ""
        L.append(f"| {k} | {nm} | {a['cls']} | {a['gov']:.2f} | {a['gov_np']:.2f} | ${a['med_wage']:,.0f} | {int(a['n_obs']):,} | {ps} | {pw_} |")
    rel = ctx["rel"]
    a_s, a_m = acs.loc["[setting]"], acs.loc["[market]"]
    L.append(f"\nWhy these classes: among young full-time bachelor's-degree employees (ACS), "
             f"{a_s['gov_np']*100:.0f}% of those in the setting-priced class work for a government or non-profit "
             f"employer ({a_s['gov']*100:.0f}% government), against {a_m['gov_np']*100:.0f}% in the market-priced class; "
             f"median wages are ${a_s['med_wage']:,.0f} and ${a_m['med_wage']:,.0f}. The premise (not tested here) is "
             "that pay in 61 and 92 mostly follows public salary schedules and pay in 62 follows reimbursement rates and "
             "institutional pay scales, while 51, 52 and 54 are private, high-wage knowledge services where employers "
             "set pay individually. The split was set by the task specification (61/62/92 vs 51/52/54) before any "
             "coupling was computed; 55 (management of companies) is added in the M+ sensitivity.")
    L.append(f"\nReliability of the shares. Test–retest across graduation cohorts (2013–15 vs 2016–18, both at y1; "
             f"same institution × family, both cells ≥ {EMP_MIN} employed; all PSEO institutions in project families): "
             f"mean within-family Spearman S {rel['S'][0]:+.2f} (min {rel['S'][1]:+.2f}, {rel['S'][2]} families, "
             f"{rel['S'][3]} cell pairs), M {rel['M'][0]:+.2f} (min {rel['M'][1]:+.2f}), log p50 {rel['log_earn'][0]:+.2f} "
             f"(min {rel['log_earn'][1]:+.2f}). Family membership explains {ctx['eta2_S']:.2f} of the cell-level "
             f"variance of S and {ctx['eta2_M']:.2f} of M (pooled y5, cells ≥ {EMP_MIN}); the rest is within-family "
             f"variation, which is what the cell-level tests use. Within-family SD of S: {sdS_W:.1f} pp (W), "
             f"{sdS_B:.1f} pp (B).\n")
    chk = ctx["chk"]
    cr = chk["count_ratio"]
    L.append("Consistency: in the pooled bachelor's cells the 20 sector counts sum exactly to the released "
             "employed total in " + ", ".join(f"{h}: {chk[h][1]*100:.0f}% of {chk[h][0]:,} cells" for h in HZ) +
             ". Flows 'employed' and Earnings 'graduates with earnings' are the same population by definition (PSEO "
             "attachment rule), and both released counts are noise-protected; they agree at the median: earnings "
             "count / flows employed total at y5 has "
             f"median {cr['all'][1]:.2f} (5th–95th percentile {cr['all'][2]:.2f}–{cr['all'][3]:.2f}) over "
             f"{cr['all'][0]:,} pooled cells and {cr['W'][1]:.2f} ({cr['W'][2]:.2f}–{cr['W'][3]:.2f}) over the "
             f"{cr['W'][0]:,} W cells.\n")
    L.append("Project families (national ranked-field share of PSEO bachelor's graduates ≥ "
             f"{FAM_COVER}): " + ", ".join(f"{C2NAME.get(c, c)} ({c}, {ctx['nat_cover'][c]:.2f})" for c in ctx["fams"]) +
             ". Dropped: " + ", ".join(f"CIP-{c} ({ctx['nat_cover'][c]:.2f})" for c in ctx["dropped_fams"]) + ".\n")
    Wc = ctx["W"]
    L.append(f"Ranked-field coverage of W cells (share of the CIP-2 cell's graduates in ranked FIELDS66 programs): "
             f"median {Wc.coverage.median():.2f}, IQR {Wc.coverage.quantile(.25):.2f}–{Wc.coverage.quantile(.75):.2f}.\n")
    # per-family table
    L.append("Per-family estimates (W sample, pooled y5; bootstrap SE in parentheses):\n")
    L.append("| family | CIP-2 | cells | mean S | mean M | ρ(P, M) | ρ(G, M) | ρ(P, S) | ρ(P, log p50) | ρ(G, log p50) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    R = res_b["W"]
    se = {k: np.nanstd(R["boot"][k], axis=0, ddof=1) for k in ["rP_M", "rG_M", "rP_S", "rP_E", "rG_E"]}
    for j, c in enumerate(R["fams"]):
        p = R["point"].set_index("cip2").loc[c]
        L.append(f"| {C2NAME.get(c, c)} | {c} | {int(p.n)} | {p.mean_S:.2f} | {p.mean_M:.2f} | "
                 + " | ".join(f"{_f(p[k])} ({se[k][j]:.2f})" for k in ["rP_M", "rG_M", "rP_S", "rP_E", "rG_E"]) + " |")
    L.append("")
    L.append(_method(ctx))
    L.append(_caveats(ctx))
    L.append(_provenance(ctx))
    OUT_MD.write_text("\n".join(L))


FF_STATUS = [("P", "P", "P × S", "W"), ("G", "G", "G × S", "W"), ("SAT_b", "SAT", "SAT × S", "B")]


def _nsig(ff: dict, key: str, s: str, specs=None) -> tuple[int, int]:
    """(number of functional forms with CI excluding 0 on the negative side, number fitted)."""
    ks = [k for k, _, _ in FF_SPECS] if specs is None else specs
    rs = [ff[(key, k)][s] for k in ks if (key, k) in ff]
    return sum(v["ci"][1] < 0 for v in rs), len(rs)


def _theta_range(ff: dict, key: str, s: str, rank=False) -> tuple[float, float, str, str]:
    """min / max theta over the functional forms in 10-pp units (rank versions excluded unless rank)."""
    ks = [k for k, _, kw in FF_SPECS if bool(kw.get("rank_x")) == rank and (key, k) in ff]
    th = {k: ff[(key, k)][s]["theta"] for k in ks}
    lo, hi = min(th, key=th.get), max(th, key=th.get)
    return th[lo], th[hi], lo, hi


FF_SHORT = {"lin": "linear e", "e2c": "common e²", "e2": "family e²", "e3": "family e², e³", "eq5": "family × e-quintile",
            "eq5zq5": "family × e- and status-quintile", "e2z2": "family e² + status²", "rr_lin": "linear rank × rank",
            "rr": "rank × rank + (rank S)²"}


def _interaction_md(ctx):
    RI, RIL, FF, DIAG, IMPL, res_b = ctx["RI"], ctx["RIL"], ctx["FF"], ctx["DIAG"], ctx["IMPL"], ctx["res_b"]
    L = ["### (c)/(d) Cell-level interaction: does the pay-for-status gradient shrink with setting-priced exposure?\n"]
    L.append("Model (primary): log p50_{ic} = γ_c + β_c z_c(status) + δ_c e_{ic} + κ_c e_{ic}² + θ z_c(status)·e_{ic} "
             "(+ controls), e = setting-priced share in 10-pp units, centred within family; status standardised within "
             "family; γ, β, δ, κ family-specific. β̄ = cell-weighted mean of β_c (log points per within-family SD of "
             "status at the family-mean share). θ < 0 means a flatter gradient where more graduates work in "
             "setting-priced sectors. CR1 SEs clustered by institution. The first version of this script had no κ_c e² "
             "term; that 'linear e' fit is shown next to every row.\n")
    # ---- functional form
    L.append("#### Functional form of the exposure main effect\n")
    dP, dG, dS = DIAG["P"], DIAG["G"], DIAG["SAT_b"]
    L.append(f"Why it matters. Within families, log p50 is curved in the setting share: the {dP['k_e2']} family-specific e² terms "
             f"are jointly non-zero (Wald χ² = {dP['wald_e2']:.0f} on {dP['df_e2']} df in the P model on W, "
             f"{dS['wald_e2']:.0f} on {dS['df_e2']} df in the SAT model on B, both {_peq(max(dP['p_e2'], dS['p_e2']))}; "
             f"{dP['n_pos_e2']}/{dP['k_e2']} and {dS['n_pos_e2']}/{dS['k_e2']} positive), and a single common e² term is "
             f"{_f(dP['e2c'], 3)} {_ci(dP['e2c_ci'], 3)} (W, P model) and {_f(dS['e2c'], 3)} {_ci(dS['e2c_ci'], 3)} (B, SAT model). "
             f"Status is correlated with e within families (pooled within-family corr(z, e) = {_f(dP['corr_ze'])} for P, "
             f"{_f(dG['corr_ze'])} for G, {_f(dS['corr_ze'])} for SAT), so the product z·e is correlated with e² "
             f"({_f(dP['corr_zee2'])}, {_f(dG['corr_zee2'])}, {_f(dS['corr_zee2'])}). With a linear e main effect the "
             "interaction can therefore pick up omitted curvature in e. The table refits θ with more flexible main "
             "effects.\n")
    L.append("θ by functional form (y5 pooled; CR1 95% CI; p). Rank rows: status and S replaced by within-family "
             "standardised ranks, so θ is per SD of status rank per SD of S rank (one within-family SD of S is "
             f"{RI['P']['sd_e']*10:.1f} pp on W and {RI['SAT_b']['sd_e']*10:.1f} pp on B), not per 10 pp.\n")
    L.append("| exposure / status main effects | P × S (W) | G × S (W) | SAT × S (B) |")
    L.append("|---|---|---|---|")
    ffm = FF["pooled_y5"]
    for k, lab, _ in FF_SPECS:
        cells_ = []
        for key, s, _, _ in FF_STATUS:
            r = ffm.get((key, k))
            cells_.append("—" if r is None else f"{_f(r[s]['theta'], 3)} {_ci(r[s]['ci'], 3)} ({_peq(r[s]['p'])})")
        L.append(f"| {lab} | " + " | ".join(cells_) + " |")
    L.append("\n| | P × S | G × S | SAT × S |\n|---|---|---|---|")
    L.append(f"| forms (of {N_FF}) with CI below 0 | " + " | ".join(f"{_nsig(ffm, key, s)[0]}/{_nsig(ffm, key, s)[1]}"
                                                         for key, s, _, _ in FF_STATUS) + " |")
    L.append(f"| θ range over the {N_FF_NR} non-rank forms | " + " | ".join(
        "{} ({}) to {} ({})".format(_f(a, 3), FF_SHORT[la], _f(b, 3), FF_SHORT[lb])
        for a, b, la, lb in [_theta_range(ffm, key, s) for key, s, _, _ in FF_STATUS]) + " |")
    L.append("\nSame comparison on other horizons, cohorts and the newer release (cells / institutions of the W or B "
             f"sample; θ [CR1 95% CI]; last column: forms, of {N_FF}, with CI below 0):\n")
    L.append("| sample | status | cells / inst. | linear e | family e² (primary) | family × e-quintile | family e² + status² | CI < 0 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for smp, slab in FF_SAMPLES:
        ff = FF.get(smp, {})
        for key, s, lab, _ in FF_STATUS:
            r0 = ff.get((key, "e2"))
            if r0 is None:
                continue
            cs = [f"{_f(ff[(key, k)][s]['theta'], 3)} {_ci(ff[(key, k)][s]['ci'], 3)}" if (key, k) in ff else "—"
                  for k in ("lin", "e2", "eq5", "e2z2")]
            ns, nt = _nsig(ff, key, s)
            L.append(f"| {slab} | {lab} | {r0['n_cells']} / {r0['n_inst']} | " + " | ".join(cs) + f" | {ns}/{nt} |")
    L.append("\nImplied status slope (cell-weighted β̄ + θ e, at the family-mean status) at the within-family 10th and "
             "90th percentile of the setting share, by exposure shape (y5 pooled; CR1 SE):\n")
    L.append("| status | exposure / status main effects | e at p10 / p90 (pp) | slope at p10 (SE) | slope at p90 (SE) | "
             "decline p10 → p90 | slope = 0 at (pp above family mean) | cells beyond |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key, s, lab, smp in FF_STATUS:
        for k in ("lin", "e2", "e3", "eq5", "e2z2"):
            im = IMPL[key][k]
            a, b = im["slope"]["p10"], im["slope"]["p90"]
            z = "never (θ ≥ 0)" if not np.isfinite(im["zero_pp"]) else f"{im['zero_pp']:+.0f}"
            sb = "—" if not np.isfinite(im["share_beyond"]) else f"{im['share_beyond']*100:.1f}%"
            L.append(f"| {s} ({smp}) | {FF_SHORT[k]}{' (primary)' if k == ECURVE_MAIN else ''} | {a[2]:+.0f} / {b[2]:+.0f} | "
                     f"{_f(a[0], 3)} ({a[1]:.3f}) | {_f(b[0], 3)} ({b[1]:.3f}) | {im['decline']*100:.0f}% | {z} | {sb} |")
    # ---- all specifications
    L.append("\n#### All specifications (primary exposure shape; linear-e fit alongside)\n")
    L.append("'boot' = institution pairs-cluster bootstrap percentile CI (B = %d, primary shape). Last column: the same "
             "row with a linear e main effect (the first version's specification)." % B_CLU + "\n")
    L.append("| specification | sample | cells / inst. / fam. | β̄ (SE) | θ | 95% CI (CR1) | p | boot 95% CI | θ with linear e [CR1 CI] |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    rows = [("P × S (primary)", "P", "P", "W"), ("G × S (brand)", "G", "G", "W"),
            ("P × S on WS sample", "P_ws", "P", "WS"), ("SAT × S", "SAT_ws", "SAT", "WS"),
            ("joint: P × S", "joint", "P", "WS"), ("joint: G × S", "joint", "G", "WS"), ("joint: SAT × S", "joint", "SAT", "WS"),
            ("P × S, selectivity-adjusted (family slopes on SAT, −ADM, Pell, state level, control + their × S)", "P_sel", "P", "WS"),
            ("P × S, selectivity slopes by family, no selectivity × S", "P_sel_noint", "P", "WS"),
            ("G × S, selectivity-adjusted", "G_sel", "G", "WS"),
            ("SAT × S", "SAT_b", "SAT", "B"), ("−ADM × S", "ADM_b", "NEG_ADM", "B"),
            ("SAT × S, + family slopes on −ADM, Pell, state level, control (+ × S)", "SAT_b_sel", "SAT", "B"),
            ("SAT × S, all CIP-2 families", "SAT_ball", "SAT", "B all"),
            ("P × S, institution FE", "P_instfe", "P", "W"), ("SAT × S, institution FE", "SAT_b_instfe", "SAT", "B"),
            ("P × S, state FE", "P_state", "P", "W"), ("SAT × S, state FE", "SAT_b_state", "SAT", "B"),
            ("P × S, + family-specific P²", "P_quad", "P", "W"), ("P × S, WLS by employed graduates", "P_wt", "P", "W"),
            ("P × S, outcome = within-family percentile rank of p50", "P_rank", "P", "W"),
            ("P × S, cells ≥ 100 employed", "P_emp100", "P", "W"), ("P × S, ranked-field coverage ≥ 0.5", "P_cov50", "P", "W"),
            ("P_u (unweighted family prestige) × S", "Pu", "P", "W"),
            ("P × S, + family slopes on SAT and SAT × S only", "P_satonly", "P", "WS"),
            ("P × S, + family slopes on market share M only, no P × M (first-revision composition control)", "P_compM", "P", "W"),
            ("SAT × S, + family slopes on market share M only, no SAT × M (first-revision composition control; joint model below)", "SAT_b_compM", "SAT", "B"),
            ("SAT × S, + family-specific SAT²", "SAT_b_quad", "SAT", "B"),
            ("SAT × S, WLS by employed graduates", "SAT_b_wt", "SAT", "B"),
            ("SAT × S, outcome = within-family percentile rank of p50", "SAT_b_rank", "SAT", "B"),
            ("SAT × S, cells ≥ 100 employed", "SAT_b_emp100", "SAT", "B"),
            ("P × M (market share instead of S; e and e² of M)", "P_M", "P", "W"), ("G × M", "G_M", "G", "W"),
            ("SAT × M", "SAT_b_M", "SAT", "B")]
    rows += [(f"P × share NAICS {k} ({SECTOR_NAME[k]}; e and e² of that share)", f"P_s{k}", "P", "W") for k in SETTING]
    rows += [(f"SAT × share NAICS {k} ({SECTOR_NAME[k]})", f"SAT_b_s{k}", "SAT", "B") for k in SETTING]
    for k, lab in [("pooled_y1", "y1 pooled"), ("pooled_y10", "y10 pooled"), ("c2010_y1", "cohort 2010-12, y1"),
                   ("c2010_y5", "cohort 2010-12, y5"), ("c2010_y10", "cohort 2010-12, y10")]:
        rows += [(f"P × S, {lab}", f"P_{k}", "P", "W"), (f"SAT × S, {lab}", f"SAT_b_{k}", "SAT", "B")]
    for lab, key, s, samp in rows:
        r = RI.get(key)
        if r is None or s not in r:
            L.append(f"| {lab} | {samp} | — | — | — | — | — | — | — |")
            continue
        v = r[s]
        bc = _ci(v["boot_ci"], 3) if "boot_ci" in v else "—"
        bt = ("n/a (constant within institution)" if key.endswith("instfe") and s in ("SAT", "G", "NEG_ADM")
              else f"{_f(v['beta'], 3)} ({v['beta_se']:.3f})")
        rl = RIL.get(key)
        lin = f"{_f(rl[s]['theta'], 3)} {_ci(rl[s]['ci'], 3)}" if rl is not None and s in rl else "—"
        L.append(f"| {lab} | {samp} | {r['n_cells']} / {r['n_inst']} / {r['k']} | {bt} | "
                 f"{_f(v['theta'], 3)} | {_ci(v['ci'], 3)} | {_p(v['p'])} | {bc} | {lin} |")
    jk = [(lab, RI[k][s]) for lab, k, s in [("P", "P", "P"), ("G", "G", "G"), ("SAT (B)", "SAT_b", "SAT")]]
    L.append("\nDelete-one-institution cluster jackknife, primary rows: " + "; ".join(
        f"θ_{lab} {_f(v['theta'], 3)}, jackknife 95% CI {_ci(v['jack_ci'], 3)}" for lab, v in jk) + ".")
    bl = [(lab, RIL[k][s]) for lab, k, s in [("P", "P", "P"), ("G", "G", "G"), ("SAT (B)", "SAT_b", "SAT")]]
    L.append("Pairs-cluster bootstrap of the linear-e fits (first-version spec): " + "; ".join(
        f"θ_{lab} {_f(v['theta'], 3)} {_ci(v['boot_ci'], 3)}" for lab, v in bl) + ".")
    MIX = ctx["MIX"]
    L.append(f"\nMultilevel check (random institution intercept, REML, same fixed part as the primary model; model-based SE): "
             f"θ_P = {_f(MIX['P']['theta'], 3)} (SE {MIX['P']['se']:.3f}; institution variance {MIX['P']['re_var']:.4f}, "
             f"residual {MIX['P']['resid_var']:.4f}); θ_SAT (B) = {_f(MIX['SAT_b']['theta'], 3)} (SE {MIX['SAT_b']['se']:.3f}; "
             f"institution variance {MIX['SAT_b']['re_var']:.4f}, residual {MIX['SAT_b']['resid_var']:.4f}). The random "
             "intercept pulls the estimate toward the within-institution (fixed-effects) value; the model-based SE "
             "ignores within-institution correlation of the residuals across families"
             + (" and is smaller than the CR1 SE.\n" if MIX['P']['se'] < RI['P']['P']['se'] and MIX['SAT_b']['se'] < RI['SAT_b']['SAT']['se']
                else ".\n"))
    for key, lab in [("P", "P × S (W)"), ("G", "G × S (W)"), ("SAT_b", "SAT × S (B)")]:
        v = ctx["LOFO"][key]
        ths = [t[1] for t in v]
        lo_c = min(v, key=lambda t: t[1]); hi_c = max(v, key=lambda t: t[1])
        L.append(f"Leave-one-family-out, {lab} (primary shape): θ ranges {_f(min(ths), 3)} (without "
                 f"{C2NAME.get(lo_c[0], lo_c[0])}) to {_f(max(ths), 3)} (without {C2NAME.get(hi_c[0], hi_c[0])}); "
                 f"{sum(t < 0 for t in ths)}/{len(ths)} point estimates negative, {sum(t[3] for t in v)}/{len(v)} with CI below 0.\n")
    L.append("Family-specific θ_c (primary model with θ varying by family; CR1 SE). Wald test of equal θ_c: "
             f"P × S χ² = {ctx['FT_P_w']['wald']:.1f} on {ctx['FT_P_w']['df']} df, {_peq(ctx['FT_P_w']['p'])}; "
             f"SAT × S (B) χ² = {ctx['FT_S_w']['wald']:.1f} on {ctx['FT_S_w']['df']} df, {_peq(ctx['FT_S_w']['p'])}.\n")
    L.append("| family | CIP-2 | cells (W) | θ_c P × S (SE) | cells (B) | θ_c SAT × S (SE) | mean S (W) | wage coupling ρ(P, log p50) |")
    L.append("|---|---|---|---|---|---|---|---|")
    ft = ctx["FT_P"].set_index("cip2"); fs = ctx["FT_S"].set_index("cip2")
    pw = res_b["W"]["point"].set_index("cip2")
    for c in sorted(set(ft.index) | set(fs.index)):
        a = f"{_f(ft.loc[c, 'theta'], 3)} ({ft.loc[c, 'se']:.3f})" if c in ft.index else "—"
        b = f"{_f(fs.loc[c, 'theta'], 3)} ({fs.loc[c, 'se']:.3f})" if c in fs.index else "—"
        L.append(f"| {C2NAME.get(c, c)} | {c} | {int(ft.loc[c, 'n']) if c in ft.index else '—'} | {a} | "
                 f"{int(fs.loc[c, 'n']) if c in fs.index else '—'} | {b} | "
                 f"{_f(pw.loc[c, 'mean_S'], 2, False) if c in pw.index else '—'} | {_f(pw.loc[c, 'rP_E']) if c in pw.index else '—'} |")
    L.append("")
    sp = ctx["SPLIT"]
    L.append("Median split within family (coupling among cells above the family's median setting share minus "
             "coupling among cells at or below it; mean over families, joint bootstrap CI). This needs no functional "
             "form for S. 'S partialled' = partial Spearman given S inside each half, so that the remaining spread of S "
             "within a half does not drive either coupling:\n")
    L.append("| contrast | sample | k | mean difference | 95% CI |")
    L.append("|---|---|---|---|---|")
    labs = {"P_log_earn": "ρ(P, log p50)", "G_log_earn": "ρ(G, log p50)", "SAT_log_earn": "ρ(SAT, log p50)"}
    for (tag, n), v in sorted(sp.items(), key=lambda t: (t[0][1].split("_", 1)[1], t[0][1].startswith("p"), t[0][0])):
        kind, var = n.split("_", 1)
        lab = labs[var] + ": high-S − low-S" + (", S partialled in each half" if kind == "psplit" else "")
        L.append(f"| {lab} | {tag} | {v['k']} | {_f(v['mean'])} | {_ci(v['ci'])} |")
    return "\n".join(L)


JSET_LAB = {"SM": "S + M", "SL": "S + L", "SML": "S + M + L"}
JFF_SHORT = {**FF_SHORT, "e2x": "family e² + cross products"}
JCOLS = [("SM", "S", "S \\| M"), ("SM", "M", "M \\| S"), ("SL", "S", "S \\| L"), ("SL", "L", "L \\| S"),
         ("SML", "S", "S \\| M, L"), ("SML", "M", "M \\| S, L"), ("SML", "L", "L \\| S, M")]


def _vth(v, nd=3):
    return f"{_f(v['theta'], nd)} {_ci(v['ci'], nd)}"


def _jcount(jb: dict, key: str, sk: str, s: str, x: str, side: str) -> tuple[int, int]:
    """(forms with CI below 0 (side '-') or above 0 (side '+'), forms fitted) for one joint theta."""
    n = t = 0
    for k, _, _ in JFF_SPECS:
        r = jb.get((key, sk, k))
        if r is None:
            continue
        v = r["TH"][(s, x)]
        t += 1
        n += int(v["ci"][1] < 0) if side == "-" else int(v["ci"][0] > 0)
    return n, t


def _scount(JB: dict, key: str, sk: str, s: str, x: str, side: str) -> tuple[int, int]:
    """Same count over the FF_SAMPLES at the primary shape."""
    n = t = 0
    for smp, _ in FF_SAMPLES:
        r = JB.get(smp, {}).get((key, sk, ECURVE_MAIN))
        if r is None:
            continue
        v = r["TH"][(s, x)]
        t += 1
        n += int(v["ci"][1] < 0) if side == "-" else int(v["ci"][0] > 0)
    return n, t


def _form_table(jb: dict, ff: dict, key: str, s: str) -> list:
    """Markdown rows: theta alone and in the joint share sets under every functional form, plus the counts."""
    L = ["| exposure / status main effects | " + s + " × S alone | " + " | ".join(f"{s} × {lab}" for _, _, lab in JCOLS) + " |",
         "|---" * (len(JCOLS) + 2) + "|"]
    for k, lab, _ in JFF_SPECS:
        r0 = ff.get((key, k))
        alone = "= family e²" if k == "e2x" else (_vth(r0[s]) if r0 is not None else "—")
        cs = []
        for sk, x, _ in JCOLS:
            r = jb.get((key, sk, k))
            cs.append(_vth(r["TH"][(s, x)]) if r is not None else "—")
        L.append(f"| {lab} | {alone} | " + " | ".join(cs) + " |")
    na = _nsig(ff, key, s)
    cnt = []
    for sk, x, _ in JCOLS:
        neg, t = _jcount(jb, key, sk, s, x, "-")
        pos, _ = _jcount(jb, key, sk, s, x, "+")
        cnt.append(f"{neg} / {pos} of {t}")
    L.append(f"| forms with CI below 0 / above 0 | {na[0]} / "
             f"{sum(ff[(key, k)][s]['ci'][0] > 0 for k, _, _ in FF_SPECS if (key, k) in ff)} of {na[1]} | "
             + " | ".join(cnt) + " |\n")
    return L


def _wp(v) -> str:
    """WCR p of a theta dict, formatted ('—' if not computed)."""
    p = v.get("wcr_p") if isinstance(v, dict) else None
    if p is None or not np.isfinite(p):
        return "—"
    return f"≤ {1 / (1 + B_WCR):.3f}" if p <= 1 / (1 + B_WCR) + 1e-12 else f"{p:.3f}"


def _vw(v, nd=3) -> str:
    """theta [CR1 CI] (WCR p)."""
    return f"{_vth(v, nd)} (p {_wp(v)})" if "wcr_p" in v else _vth(v, nd)


def _timing_md(ctx):
    """Section (c''): the y5 gradient with the sector shares measured at year 1 (third revision)."""
    RJ, RX, RXA, JBX, JB5X, FFX, FF5X = (ctx[k] for k in ("RJ", "RX", "RXA", "JBX", "JB5X", "FFX", "FF5X"))
    XN, IMPLX, GBX, TDX, SUMX, RI = ctx["XN"], ctx["IMPLX"], ctx["GBX"], ctx["TDX"], ctx["SUMX"], ctx["RI"]
    k0 = X_MAIN
    td = TDX[k0]
    COMP, NO19 = ctx["COMP"], ctx["NO19"]
    c5, c1 = COMP["y5_B"], COMP["y1_x"]
    fr5, er5 = COMP["flows_rel_y5"], COMP["earn_rel_y5"]
    L = ["\n### (c'') When are the shares measured? The y5 gradient with year-1 sector shares (third and fourth "
         "revisions)\n"]
    L.append("Why. The joint models in (c') measure S and M at y5 in the same pooled cells whose y5 p50 is the outcome. "
             "Those shares can be partly an outcome of status: sorting into market sectors by status grows with career "
             "time (section 'Horizon' below), and within families higher SAT goes with a "
             + ("slightly " if td["z_dM"] < 0.2 else "") + "larger y1 → y5 rise of the market share (pooled "
             f"within-family corr(z_SAT, M5 − M1) = {_f(td['z_dM'])}; for the setting share {_f(td['z_dS'])}; B cells of the "
             f"pooled-y1 sample; SD of the y1 → y5 change {td['sd_dM']:.1f} pp for M and {td['sd_dS']:.1f} pp for S). This "
             "section keeps y5 outcome cells and measures the exposure at year 1 of the same institution × family "
             f"(the y1 cell needs ≥ {EMP_MIN} employed graduates, which drops {XN[k0]['B_drop']} of {len(ctx['Bs'])} B "
             "cells for the pooled y1 shares). Within families the y1 and y5 shares of a cell are correlated but not the "
             f"same ({_f(td['S1_S5'])} for S, {_f(td['M1_M5'])} for M; S and M {_f(td['S1_M1'])} at y1 and "
             f"{_f(td['S5_M5'])} at y5). The y1 shares are measured four years before the y5 pay but are not exogenous: "
             f"status already predicts them (corr(z_SAT, S1) {_f(td['z_S1'])}, corr(z_SAT, M1) {_f(td['z_M1'])}; at y5 "
             f"{_f(td['z_S5'])} and {_f(td['z_M5'])}).\n")
    L.append("Which graduates the cells hold (fourth revision; the third revision stated wrongly that the 2013–15 and "
             "2016–18 cohorts do not reach y5 in this release, and that the pooled y1 shares therefore describe largely "
             "other graduates than those whose y5 pay is the outcome). In V4.13.0 every 3-year graduation cohort from "
             "2001–03 to 2016–18 reaches y5: PSEO Flows release y5 employed counts (status 1, all-sector totals) for "
             f"the 2013–15 cohort in {fr5.get('2013', 0):,} institution × CIP-2 cells and for the 2016–18 cohort in "
             f"{fr5.get('2016', 0):,}, and PSEO Earnings releases their y5 p50 in {er5.get('2013', 0):,} and "
             f"{er5.get('2016', 0):,} cells (the 2019–21 cohort: {fr5.get(EXCL_COHORT, 0)} and "
             f"{er5.get(EXCL_COHORT, 0)}). On the B cells ({c5['n']} of {c5['n_B']} with a released pooled total) the "
             "pooled ('0000') y5 employed count equals the sum of the released cohort counts at the median (ratio "
             f"{c5['ratio_all']:.2f}; {c5['ratio_x']:.2f} without the 2013–15 and 2016–18 cohorts), and a median "
             f"{c5['share_1316'] * 100:.0f}% of a cell's pooled-y5 employed graduates are in those two cohorts "
             f"(10th–90th percentile {c5['share_1316_p10'] * 100:.0f}–{c5['share_1316_p90'] * 100:.0f}%). The pooled y1 "
             "cell holds the same cohorts plus the 2019–21 cohort, which has no y5 yet (median "
             f"{c1['share_19'] * 100:.0f}% of a cell's pooled-y1 employed graduates; 10th–90th percentile "
             f"{c1['share_19_p10'] * 100:.0f}–{c1['share_19_p90'] * 100:.0f}%; {c1['n']} B cells of the pooled-y1 sample). "
             "Unreleased cohort counts are taken as 0 in these sums. The timing samples therefore follow three designs: "
             "(i) *pooled*: pooled-cohort y1 shares with the pooled-cohort y5 pay, largely the same graduates at y1 and "
             "y5; a check removes the 2019–21 cohort from the y1 shares (pooled minus that cohort's released counts; "
             f"of the {sum(NO19['pooled_y1_no19'].values()):,} pooled y1 cells, the 2019–21 count is released in "
             f"{NO19['pooled_y1_no19']['released']:,}, absent in {NO19['pooled_y1_no19']['absent']:,} and suppressed in "
             f"{NO19['pooled_y1_no19']['suppressed']:,}; the strict version drops the suppressed ones); (ii) *own "
             "cohort*: each cohort's own y1 shares and its own y5 p50 (B cells with a y1 cell of ≥ "
             f"{EMP_MIN} employed: " + ", ".join(f"{_coh(k)} {XN[k]['B'][0]} of {XN[k]['B_base']}" for k in X_OWN)
             + "); (iii) *mixed*: one cohort's y1 shares with the pooled y5 pay, which combines that cohort's placement "
             "with the pay of all cohorts. Every y1 row is shown next to the same model with the y5 shares on exactly "
             "the same cells. 'p' = wild-cluster restricted bootstrap p (WCR: null imposed, Rademacher weights by "
             f"institution, B = {B_WCR}); CIs are CR1.\n")
    # --- T1: SAT, primary shape, by exposure timing
    L.append("SAT × share, primary shape (family e² for each share), B cells, outcome log p50 at y5. 'forms' = functional "
             f"forms (of {N_JFF}) with a CI below 0 for SAT × S | M and above 0 for SAT × M | S:\n")
    L.append("| shares measured | cells / inst. | SAT × S alone | SAT × S \\| M | SAT × M \\| S | forms S \\| M < 0 | forms M \\| S > 0 |")
    L.append("|---|---|---|---|---|---|---|")
    ff5 = ctx["FF"]["pooled_y5"]
    jb5 = ctx["JB"]["pooled_y5"]
    L.append(f"| y5, all B cells (section (c')) | {RJ['SAT_b_SM']['n_cells']} / {RJ['SAT_b_SM']['n_inst']} | "
             f"{_vth(ff5[('SAT_b', ECURVE_MAIN)]['SAT'])} | {_vw(RJ['SAT_b_SM']['TH'][('SAT', 'S')])} | "
             f"{_vw(RJ['SAT_b_SM']['TH'][('SAT', 'M')])} | {_jcount(jb5, 'SAT_b', 'SM', 'SAT', 'S', '-')[0]}/{N_JFF} | "
             f"{_jcount(jb5, 'SAT_b', 'SM', 'SAT', 'M', '+')[0]}/{N_JFF} |")
    kc = X_CHECKS[0][0]
    t1_order = [X_SAMPLES[0]] + X_CHECKS + X_SAMPLES[1:]
    for k, lab, _, _ in t1_order:
        for jb, ff, key, l_ in [(JB5X[k], FF5X[k], f"SAT_SM5_{k}", f"y5, same cells as next row"),
                                (JBX[k], FFX[k], f"SAT_SM_{k}", f"**{lab}**" if k not in dict(
                                    (c[0], 1) for c in X_CHECKS) else f"check: {lab}")]:
            r = RX[key]
            L.append(f"| {l_} | {r['n_cells']} / {r['n_inst']} | {_vth(ff[('SAT_b', ECURVE_MAIN)]['SAT'])} | "
                     f"{_vw(r['TH'][('SAT', 'S')])} | {_vw(r['TH'][('SAT', 'M')])} | "
                     f"{_jcount(jb, 'SAT_b', 'SM', 'SAT', 'S', '-')[0]}/{N_JFF} | "
                     f"{_jcount(jb, 'SAT_b', 'SM', 'SAT', 'M', '+')[0]}/{N_JFF} |")
        if k == kc:
            r = RX[f"SAT_SMP1_{kc}"]
            L.append(f"| pooled y1 shares with the 2019–21 cohort, same cells as previous row | {r['n_cells']} / "
                     f"{r['n_inst']} | — | {_vw(r['TH'][('SAT', 'S')])} | {_vw(r['TH'][('SAT', 'M')])} | — | — |")
    # --- T2: S + M + L and S + C
    L.append("\nThe low-pay private share L (post hoc) and the ex-ante sector-wage-mix index C (primary shape; B cells; "
             "y5 outcome). C = log Σ_k share_k × ACS 2023 median wage of full-time employees aged 22–40 with a bachelor's "
             "degree or more in sector k "
             "(the 20 sectors of the table in (a)), centred within family, in units of 0.1 log points; it summarises the "
             "pay level of the whole sector mix without choosing classes:\n")
    L.append("| shares measured | SAT × S \\| M, L | SAT × M \\| S, L | SAT × L \\| S, M | θ_S − θ_L | SAT × S \\| C | SAT × C \\| S |")
    L.append("|---|---|---|---|---|---|---|")
    r3, rc = RJ["SAT_b_SML"], RJ["SAT_b_SC"]
    t3 = r3["tests"]["SAT"]["S_minus_L"]
    L.append(f"| y5, all B cells | {_vw(r3['TH'][('SAT', 'S')])} | {_vw(r3['TH'][('SAT', 'M')])} | "
             f"{_vw(r3['TH'][('SAT', 'L')])} | {_f(t3['est'][0], 3)} {_ci(t3['ci'][0], 3)} | "
             f"{_vw(rc['TH'][('SAT', 'S')])} | {_vw(rc['TH'][('SAT', 'C')])} |")
    for k, lab, _, _ in X_SAMPLES:
        for sfx, l_ in [("5", "y5, same cells as next row"), ("", f"**{lab}**")]:
            r3, rc = RX[f"SAT_SML{sfx}_{k}"], RX[f"SAT_SC{sfx}_{k}"]
            t3 = r3["tests"]["SAT"]["S_minus_L"]
            L.append(f"| {l_} | {_vw(r3['TH'][('SAT', 'S')])} | {_vw(r3['TH'][('SAT', 'M')])} | "
                     f"{_vw(r3['TH'][('SAT', 'L')])} | {_f(t3['est'][0], 3)} {_ci(t3['ci'][0], 3)} | "
                     f"{_vw(rc['TH'][('SAT', 'S')])} | {_vw(rc['TH'][('SAT', 'C')])} |")
    # --- T3: selectivity controls by timing
    L.append("\nWith the broad selectivity controls (family-specific slopes on −ADM, Pell, state earnings level and "
             "control, and each of them × S and × M; primary shape; B cells with all covariates):\n")
    L.append("| shares measured | cells / inst. | SAT × S \\| M | SAT × M \\| S |")
    L.append("|---|---|---|---|")
    r = RJ["SAT_b_SM_sel"]
    L.append(f"| y5, all B cells | {r['n_cells']} / {r['n_inst']} | {_vw(r['TH'][('SAT', 'S')])} | {_vw(r['TH'][('SAT', 'M')])} |")
    for k, lab, _, _ in X_SAMPLES + X_CHECKS:
        chk_ = k in [c[0] for c in X_CHECKS]
        for key, l_ in [(f"SAT_SM5_sel_{k}", "y5, same cells as next row"),
                        (f"SAT_SM_sel_{k}", f"check: {lab}" if chk_ else f"**{lab}**")]:
            r = RX[key]
            L.append(f"| {l_} | {r['n_cells']} / {r['n_inst']} | {_vw(r['TH'][('SAT', 'S')])} | {_vw(r['TH'][('SAT', 'M')])} |")
    # --- T4: robustness, pooled y1 shares
    L.append(f"\nPooled y1 shares, other specifications (y5 outcome; primary shape unless stated; CR1 95% CI (WCR p); "
             f"'boot' = institution pairs-cluster bootstrap, B = {B_CLU}; 'jack' = delete-one-institution jackknife):\n")
    L.append("| specification | sample | cells / inst. | θ(status × S1 \\| M1) | θ(status × M1 \\| S1) | other |")
    L.append("|---|---|---|---|---|---|")
    rowsx = [("SAT × S1, SAT × M1 (primary)", f"SAT_SM_{k0}", "SAT", "B", "M")]
    rowsx += [(f"SAT, {lab}", f"SAT_SM_{f_}_{k0}", "SAT", "WS" if f_ == "ws" else ("B all" if f_ == "ball" else "B"), "M")
              for f_, lab in TIMING_ROBUST]
    rowsx += [("SAT, y5 shares on the same cells, + institution FE", f"SAT_SM5_instfe_{k0}", "SAT", "B", "M"),
              ("−ADM × S1, −ADM × M1", f"ADM_SM_{k0}", "NEG_ADM", "B", "M"),
              ("SAT, M+ (M + NAICS 55) instead of M", f"SAT_SMplus_{k0}", "SAT", "B", "Mplus"),
              ("P × S1, P × M1", f"P_SM_{k0}", "P", "W", "M"),
              ("P, + institution FE", f"P_SM_instfe_{k0}", "P", "W", "M"),
              ("P, selectivity-adjusted (family slopes on SAT, −ADM, Pell, state level, control + × S1, × M1)",
               f"P_SM_sel_{k0}", "P", "WS", "M"),
              ("G × S1, G × M1", f"G_SM_{k0}", "G", "W", "M"),
              ("joint status P, G, SAT: SAT", f"joint_SM_{k0}", "SAT", "WS", "M"),
              ("joint status P, G, SAT: P", f"joint_SM_{k0}", "P", "WS", "M"),
              ("joint status P, G, SAT: G", f"joint_SM_{k0}", "G", "WS", "M")]
    for lab, key, s, samp, mx in rowsx:
        r = RX[key]
        vS, vM = r["TH"][(s, "S")], r["TH"][(s, mx)]
        extra = []
        for nm, v in [("S1", vS), ("M1", vM)]:
            if "boot_ci" in v:
                extra.append(f"{nm}: boot {_ci(v['boot_ci'], 3)}")
            if "jack_ci" in v:
                extra.append(f"{nm}: jack {_ci(v['jack_ci'], 3)}")
        if key == f"SAT_SM_y5also_{k0}":
            extra.append(f"SAT × S5 {_vw(r['TH'][('SAT', 'S5')])}; SAT × M5 {_vw(r['TH'][('SAT', 'M5')])}")
        L.append(f"| {lab} | {samp} | {r['n_cells']} / {r['n_inst']} | {_vw(vS)} | {_vw(vM)} | {'; '.join(extra) or '—'} |")
    ta = RX[f"joint_SM_{k0}"]["tests_all"]
    L.append(f"\nThree-status model with pooled y1 shares (WS cells): all three × S1 | M1 = 0: χ² = {ta['S_all']['W']:.2f} on "
             f"{ta['S_all']['df']} df, {_peq(ta['S_all']['p'])}; sum {_f(ta['S_sum']['est'][0], 3)} {_ci(ta['S_sum']['ci'][0], 3)}; "
             f"all three × M1 | S1 = 0: χ² = {ta['M_all']['W']:.2f} on {ta['M_all']['df']} df, {_peq(ta['M_all']['p'])}; sum "
             f"{_f(ta['M_sum']['est'][0], 3)} {_ci(ta['M_sum']['ci'][0], 3)}.")
    rsl, rsl2 = RX[f"SAT_SL_{k0}"], RX[f"SAT_SML2_{k0}"]
    L.append(f"\nWith S1 and L1 only: SAT × S1 {_vw(rsl['TH'][('SAT', 'S')])}, SAT × L1 {_vw(rsl['TH'][('SAT', 'L')])}. "
             f"S1 + M1 + L2₁ (L2 without 71, 81): SAT × S1 {_vw(rsl2['TH'][('SAT', 'S')])}, SAT × L2₁ "
             f"{_vw(rsl2['TH'][('SAT', 'L2')])}; θ_S − θ_L2 {_f(rsl2['tests']['SAT']['S_minus_L']['est'][0], 3)} "
             f"{_ci(rsl2['tests']['SAT']['S_minus_L']['ci'][0], 3)}. Setting-sector components with M1 (pooled y1 shares; "
             "each component and M1 with their own family e²): "
             + "; ".join(f"{SECTOR_NAME[k].lower()} {_vw(RX[f'SAT_s{k}_M_{k0}']['TH'][('SAT', f'sh_{k}')])} (SAT × M1 given "
                         f"it {_vth(RX[f'SAT_s{k}_M_{k0}']['TH'][('SAT', 'M')])})" for k in SETTING) + ".")
    # --- T5: battery, pooled y1 shares, SAT; P and G summaries
    L.append(f"\nθ by functional form, SAT, pooled y1 shares (B cells, y5 outcome; CR1 95% CI; '|' = other y1 shares held fixed):\n")
    L += _form_table(JBX[k0], FFX[k0], "SAT_b", "SAT")
    L.append("Prestige and brand with y1 shares (W cells, y5 outcome, primary shape; last two columns: the same S + M "
             "model with the y5 shares of the same cells):\n")
    L.append("| shares measured | status | cells / inst. | × S alone | × S \\| M | × M \\| S | forms S \\| M < 0 / > 0 | "
             "forms M \\| S > 0 | y5 shares: × S \\| M | y5 shares: × M \\| S |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, lab, _, _ in X_SAMPLES:
        for key, s in [("P", "P"), ("G", "G")]:
            r = JBX[k][(key, "SM", ECURVE_MAIN)]
            r5 = JB5X[k][(key, "SM", ECURVE_MAIN)]
            L.append(f"| {lab} | {s} | {r['n_cells']} / {r['n_inst']} | {_vth(FFX[k][(key, ECURVE_MAIN)][s])} | "
                     f"{_vth(r['TH'][(s, 'S')])} | {_vth(r['TH'][(s, 'M')])} | "
                     f"{_jcount(JBX[k], key, 'SM', s, 'S', '-')[0]} / {_jcount(JBX[k], key, 'SM', s, 'S', '+')[0]} of {N_JFF} | "
                     f"{_jcount(JBX[k], key, 'SM', s, 'M', '+')[0]}/{N_JFF} | {_vth(r5['TH'][(s, 'S')])} | {_vth(r5['TH'][(s, 'M')])} |")
    # --- implied slopes, Gelbach, conditional split, timing diagnostics
    ia, ia5, ij = IMPLX["alone"], IMPLX["alone5"], IMPLX["joint"]
    L.append("\nImplied SAT slope (cell-weighted β̄ + θ e at the family-mean SAT; y5 outcome, primary shape; CR1 SE), "
             "pooled-y1 cells:\n")
    L.append("| model | share moved | share at p10 / p90 (pp from family mean) | slope at p10 (SE) | slope at p90 (SE) | "
             "change p10 → p90 | slope = 0 at (pp) | cells beyond |")
    L.append("|---|---|---|---|---|---|---|---|")
    for lab, im in [("S alone, y5 shares (same cells)", ia5), ("S1 alone, y1 shares", ia)]:
        a, b = im["slope"]["p10"], im["slope"]["p90"]
        z = "never (θ ≥ 0)" if not np.isfinite(im["zero_pp"]) else f"{im['zero_pp']:+.0f}"
        sb = "—" if not np.isfinite(im["share_beyond"]) else f"{im['share_beyond']*100:.1f}%"
        L.append(f"| {lab} | S | {a[2]:+.0f} / {b[2]:+.0f} | {_f(a[0], 3)} ({a[1]:.3f}) | {_f(b[0], 3)} ({b[1]:.3f}) | "
                 f"{(b[0] / a[0] - 1) * 100:+.0f}% | {z} | {sb} |")
    for x in ("S", "M"):
        a, b = ij[(x, 10)], ij[(x, 90)]
        L.append(f"| S1 + M1 (other share at its family mean) | {x}1 | {a[2]:+.0f} / {b[2]:+.0f} | {_f(a[0], 3)} ({a[1]:.3f}) | "
                 f"{_f(b[0], 3)} ({b[1]:.3f}) | {(b[0] / a[0] - 1) * 100:+.0f}% | — | — |")
    L.append(f"\nθ(SAT × S1) alone {_vw(RXA['SAT']['SAT'])}; with the y5 shares of the same cells "
             f"{_vw(RXA['SAT5']['SAT'])}.")
    L.append("\nWhere the S-only θ goes when M is added (Gelbach decomposition, pooled-y1 cells, primary shape):\n")
    L.append("| shares | θ × S alone | θ × S joint | difference | from M's main effects | from the SAT × M product |")
    L.append("|---|---|---|---|---|---|")
    for lab, k in [("y5 (same cells)", "SAT_M5"), ("y1", "SAT_M")]:
        g = GBX[k]
        me = sum(v for n, v in g["parts"].items() if not n.startswith("product"))
        pr = sum(v for n, v in g["parts"].items() if n.startswith("product"))
        L.append(f"| {lab} | {_f(g['theta_alone'], 4)} | {_f(g['theta_joint'], 4)} | {_f(g['diff'], 4)} | {_f(me, 4)} | {_f(pr, 4)} |")
    L.append(f"\nShape-free conditional median splits of ρ(SAT, log p50) on the pooled-y1 cells (mean over families of high "
             f"half minus low half; joint institution bootstrap, B = {B_BOOT}):\n")
    L.append("| shares measured | high − low S | high − low M | S within M halves | M within S halves |")
    L.append("|---|---|---|---|---|")
    for tag, lab in [("x5", "y5 (same cells)"), ("x1", "y1")]:
        cs = []
        for kd in ("split", "msplit", "csplitS", "csplitM"):
            v = SUMX.get((tag, f"{kd}_SAT_log_earn"))
            cs.append(f"{_f(v['mean'])} {_ci(v['ci'])} (k = {v['k']})" if v is not None else "—")
        L.append(f"| {lab} | " + " | ".join(cs) + " |")
    L.append("\nTiming diagnostics (pooled within-family correlations on the B cells of each sample):\n")
    L.append("| y1 shares from | cells / inst. | r(S1, S5) | r(M1, M5) | r(z_SAT, S1) | r(z_SAT, S5) | r(z_SAT, M1) | "
             "r(z_SAT, M5) | r(z_SAT, S5 − S1) | r(z_SAT, M5 − M1) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, lab, _, _ in X_SAMPLES:
        t = TDX[k]
        L.append(f"| {lab} | {XN[k]['B'][0]} / {XN[k]['B'][1]} | {_f(t['S1_S5'])} | {_f(t['M1_M5'])} | {_f(t['z_S1'])} | "
                 f"{_f(t['z_S5'])} | {_f(t['z_M1'])} | {_f(t['z_M5'])} | {_f(t['z_dS'])} | {_f(t['z_dM'])} |")
    return "\n".join(L) + "\n"


def _joint_md(ctx):
    """Section (c'): joint exposure models (second revision)."""
    RI, RJ, JB, FF, COL, GB, SUM = ctx["RI"], ctx["RJ"], ctx["JB"], ctx["FF"], ctx["COL"], ctx["GB"], ctx["SUM"]
    jb5, ff5 = JB["pooled_y5"], FF["pooled_y5"]
    cS, cP = COL["SAT_b"], COL["P"]
    L = ["\n### (c') Setting share or market share? Joint exposure models (second revision)\n"]
    L.append("Why. Within families the setting share S and the market share M move against each other (pooled "
             f"within-family correlation of the centred shares {_f(cS['e_S_M'])} on B, {_f(cP['e_S_M'])} on W; of the "
             f"status × share products z·e_S and z·e_M {_f(cS['ze_S_M'])} for SAT, {_f(cP['ze_S_M'])} for P). A flatter "
             "status gradient where S is high can therefore be a steeper one where M is high. The models below put "
             "status × S and status × M (and, post hoc, status × L, the share in low-pay private sectors NAICS 44-45, 56, "
             "71, 72, 81) in one model; every share has its own family-specific main-effect shape (the same form for "
             "each share), and θ for a share is per SD of status per +10 pp of that share with the other shares in the "
             "model held fixed. The omitted category is everything else (11, 21, 22, 23, 31-33, 42, 48-49, 53, 55, and "
             "L when L is not in the model). All shares in this section are measured at the outcome's horizon (y5 "
             "unless stated); section (c'') measures them at year 1 instead, which changes which share carries the "
             "SAT moderation. Collinearity: the CR1 SE of θ(SAT × S) is "
             f"{RI['SAT_b']['SAT']['se']:.4f} alone and {RJ['SAT_b_SM']['TH'][('SAT', 'S')]['se']:.4f} with M (P: "
             f"{RI['P']['P']['se']:.4f} and {RJ['P_SM']['TH'][('P', 'S')]['se']:.4f}; with M and L, SAT "
             f"{RJ['SAT_b_SML']['TH'][('SAT', 'S')]['se']:.4f}). "
             + ("The SE barely changes, so the S term loses significance because its point estimate moves toward 0 "
                "(SAT) or changes sign (P, G), not because precision is lost."
                if RJ['SAT_b_SM']['TH'][('SAT', 'S')]['se'] < 1.25 * RI['SAT_b']['SAT']['se'] else
                "The SE rises noticeably, so part of the loss of significance is a loss of precision.")
             + "\n")
    # --- by functional form, y5
    for key, s, samp in [("SAT_b", "SAT", "B"), ("P", "P", "W"), ("G", "G", "W")]:
        L.append(f"θ by functional form, {s} ({samp} cells, y5 pooled; CR1 95% CI; '|' = other shares held fixed):\n")
        L += _form_table(jb5, ff5, key, s)
    # --- across samples, primary shape
    L.append("Across samples, primary shape (family e² for each share; CR1 95% CI). Last two columns: functional forms "
             f"(of {N_JFF}) in that sample with a CI below 0 for status × S | M and above 0 for status × M | S.\n")
    L.append("| sample | status | cells / inst. | × S alone | × S \\| M | × M \\| S | × S \\| M, L | × M \\| S, L | × L \\| S, M | "
             "θ_S − θ_L (S + M + L) | Wald θ_S = θ_L = 0 | forms S \\| M < 0 | forms M \\| S > 0 |")
    L.append("|---" * 14 + "|")
    for smp, slab in FF_SAMPLES:
        jb, ff = JB.get(smp, {}), FF.get(smp, {})
        for key, s, _, _ in FF_STATUS:
            rSM, rSML = jb.get((key, "SM", ECURVE_MAIN)), jb.get((key, "SML", ECURVE_MAIN))
            if rSM is None or rSML is None:
                continue
            a = ff.get((key, ECURVE_MAIN))
            t = rSML["tests"]
            dl = t["S_minus_L"]
            L.append(f"| {slab} | {s} | {rSM['n_cells']} / {rSM['n_inst']} | {_vth(a[s]) if a is not None else '—'} | "
                     f"{_vth(rSM['TH'][(s, 'S')])} | {_vth(rSM['TH'][(s, 'M')])} | {_vth(rSML['TH'][(s, 'S')])} | "
                     f"{_vth(rSML['TH'][(s, 'M')])} | {_vth(rSML['TH'][(s, 'L')])} | "
                     f"{_f(dl['est'][0], 3)} {_ci(dl['ci'][0], 3)} | {_peq(t['S_L_zero']['p'])} | "
                     f"{_jcount(jb, key, 'SM', s, 'S', '-')[0]}/{_jcount(jb, key, 'SM', s, 'S', '-')[1]} | "
                     f"{_jcount(jb, key, 'SM', s, 'M', '+')[0]}/{_jcount(jb, key, 'SM', s, 'M', '+')[1]} |")
    # --- robustness rows
    L.append("\nJoint S + M model, other specifications (y5 pooled, primary shape; CR1 95% CI (p = wild-cluster restricted "
             f"bootstrap p, B = {B_WCR}; added in the third revision); 'boot' = institution "
             f"pairs-cluster bootstrap, B = {B_CLU}; 'jack' = delete-one-institution jackknife):\n")
    L.append("| specification | sample | cells / inst. | θ(status × S \\| M) | θ(status × M \\| S) | other |")
    L.append("|---|---|---|---|---|---|")
    jrows = [("SAT × S, SAT × M (primary joint)", "SAT_b_SM", "SAT", "B"),
             ("SAT, + state FE", "SAT_b_SM_state", "SAT", "B"),
             ("SAT, + institution FE", "SAT_b_SM_instfe", "SAT", "B"),
             ("SAT, + family slopes on −ADM, Pell, state level, control (+ × S, × M)", "SAT_b_SM_sel", "SAT", "B"),
             ("SAT, WS cells", "SAT_ws_SM", "SAT", "WS"), ("SAT, WLS by employed graduates", "SAT_b_SM_wt", "SAT", "B"),
             ("SAT, outcome = within-family percentile rank of p50", "SAT_b_SM_rank", "SAT", "B"),
             ("SAT, cells ≥ 100 employed", "SAT_b_SM_emp100", "SAT", "B"),
             ("SAT, all CIP-2 families", "SAT_ball_SM", "SAT", "B all"),
             ("−ADM × S, −ADM × M", "ADM_b_SM", "NEG_ADM", "B"),
             ("SAT, M+ (M + NAICS 55) instead of M", "SAT_b_SMplus", "SAT", "B"),
             ("P × S, P × M", "P_SM", "P", "W"), ("P, + institution FE", "P_SM_instfe", "P", "W"),
             ("P, + state FE", "P_SM_state", "P", "W"),
             ("P, selectivity-adjusted (family slopes on SAT, −ADM, Pell, state level, control + × S, × M)", "P_sel_SM", "P", "WS"),
             ("G × S, G × M", "G_SM", "G", "W"),
             ("joint status P, G, SAT: SAT", "joint_SM", "SAT", "WS"), ("joint status P, G, SAT: P", "joint_SM", "P", "WS"),
             ("joint status P, G, SAT: G", "joint_SM", "G", "WS")]
    for lab, key, s, samp in jrows:
        r = RJ[key]
        vS, vM = r["TH"][(s, "S")], r["TH"][(s, "Mplus" if key == "SAT_b_SMplus" else "M")]
        extra = []
        for nm, v in [("S", vS), ("M", vM)]:
            if "boot_ci" in v:
                extra.append(f"{nm}: boot {_ci(v['boot_ci'], 3)}")
            if "jack_ci" in v:
                extra.append(f"{nm}: jack {_ci(v['jack_ci'], 3)}")
        L.append(f"| {lab} | {samp} | {r['n_cells']} / {r['n_inst']} | {_vw(vS)} | {_vw(vM)} | {'; '.join(extra) or '—'} |")
    ta = RJ["joint_SM"]["tests_all"]
    L.append(f"\nThree-status model on the WS cells (last three rows; P, G and SAT are correlated within families): all "
             f"three × S | M = 0: χ² = {ta['S_all']['W']:.2f} on {ta['S_all']['df']} df, {_peq(ta['S_all']['p'])}; sum of "
             f"the three × S | M terms {_f(ta['S_sum']['est'][0], 3)} {_ci(ta['S_sum']['ci'][0], 3)}; all three × M | S = 0: "
             f"χ² = {ta['M_all']['W']:.2f} on {ta['M_all']['df']} df, {_peq(ta['M_all']['p'])}; sum of the three × M | S "
             f"terms {_f(ta['M_sum']['est'][0], 3)} {_ci(ta['M_sum']['ci'][0], 3)}.")
    L.append("\nSetting-sector components jointly with M (y5 pooled, primary shape; each component share and M with their "
             "own family-specific e²):\n")
    L.append("| component | SAT × component \\| M (B) | SAT × M \\| component | P × component \\| M (W) | P × M \\| component | "
             "SAT × component alone | P × component alone |")
    L.append("|---|---|---|---|---|---|---|")
    for k in SETTING:
        rs, rp = RJ[f"SAT_b_s{k}_M"], RJ[f"P_s{k}_M"]
        L.append(f"| NAICS {k} {SECTOR_NAME[k]} | {_vth(rs['TH'][('SAT', f'sh_{k}')])} | {_vth(rs['TH'][('SAT', 'M')])} | "
                 f"{_vth(rp['TH'][('P', f'sh_{k}')])} | {_vth(rp['TH'][('P', 'M')])} | {_vth(RI[f'SAT_b_s{k}']['SAT'])} | "
                 f"{_vth(RI[f'P_s{k}']['P'])} |")
    # --- S + M + L, tests
    r3, r3b, r3p = RJ["SAT_b_SML"], RJ["SAT_b_SML2"], RJ["P_SML"]
    t3, t3b, t3p = r3["tests"]["SAT"], r3b["tests"]["SAT"], r3p["tests"]["P"]
    L.append("\nS, M and the low-pay private share together (y5 pooled, primary shape). L is post hoc: it was defined by "
             "the independent verifier after seeing the S + M result. L2 drops 71 and 81, which have large non-profit "
             "shares in the ACS table below:\n")
    L.append("| model | sample | θ(× S) | θ(× M) | θ(× L) | θ_S − θ_L | Wald θ_S = θ_L = 0 |")
    L.append("|---|---|---|---|---|---|---|")
    for lab, r, s, t, lx, smp in [("SAT, S + M + L", r3, "SAT", t3, "L", "B"), ("SAT, S + M + L2", r3b, "SAT", t3b, "L2", "B"),
                                  ("P, S + M + L", r3p, "P", t3p, "L", "W"), ("G, S + M + L", RJ["G_SML"], "G", RJ["G_SML"]["tests"]["G"], "L", "W")]:
        dl = t["S_minus_L"]
        bt = (f"; boot θ_S {_ci(r['TH'][(s, 'S')]['boot_ci'], 3)}, θ_M {_ci(r['TH'][(s, 'M')]['boot_ci'], 3)}"
              if "boot_ci" in r["TH"][(s, "S")] else "")
        L.append(f"| {lab}{bt} | {smp} | {_vw(r['TH'][(s, 'S')])} | {_vw(r['TH'][(s, 'M')])} | {_vw(r['TH'][(s, lx)])} | "
                 f"{_f(dl['est'][0], 3)} {_ci(dl['ci'][0], 3)} | χ² = {t['S_L_zero']['W']:.2f} on {t['S_L_zero']['df']} df, "
                 f"{_peq(t['S_L_zero']['p'])} |")
    rSL = RJ["SAT_b_SL"]
    L.append(f"\nWith S and L only (no M): SAT × S {_vth(rSL['TH'][('SAT', 'S')])}, SAT × L {_vth(rSL['TH'][('SAT', 'L')])}; "
             f"P × S {_vth(RJ['P_SL']['TH'][('P', 'S')])}, P × L {_vth(RJ['P_SL']['TH'][('P', 'L')])}.")
    # --- Gelbach
    L.append("\nWhere the S-only θ goes when M is added (exact OLS decomposition, Gelbach 2016; y5 pooled, primary shape; "
             "θ(alone) − θ(joint) = sum of the contributions of the added columns; point estimates):\n")
    L.append("| status, added shares | θ × S alone | θ × S joint | difference | from the added shares' main effects | "
             "from the added status × share products |")
    L.append("|---|---|---|---|---|---|")
    for lab, k in [("SAT (B), + M", "SAT_b_M"), ("SAT (B), + L", "SAT_b_L"), ("SAT (B), + M, L", "SAT_b_ML"),
                   ("P (W), + M", "P_M"), ("G (W), + M", "G_M")]:
        g = GB[k]
        me = sum(v for n, v in g["parts"].items() if n.startswith("main effects") or n == "cross products")
        pr = sum(v for n, v in g["parts"].items() if n.startswith("product"))
        L.append(f"| {lab} | {_f(g['theta_alone'], 4)} | {_f(g['theta_joint'], 4)} | {_f(g['diff'], 4)} | {_f(me, 4)} | "
                 f"{_f(pr, 4)} |")
    # --- implied slopes
    IJ = ctx["IMPLJ"]
    L.append("\nImplied status slope in the joint S + M model (cell-weighted β̄ + θ_S e_S + θ_M e_M, at the family-mean "
             "status; y5 pooled, primary shape; CR1 SE): moving one share from its within-family 10th to its 90th "
             "percentile with the other share at its family mean.\n")
    L.append("| status | share moved | share at p10 / p90 (pp from family mean) | slope at p10 (SE) | slope at p90 (SE) |")
    L.append("|---|---|---|---|---|")
    for key, s in [("SAT_b", "SAT"), ("P", "P")]:
        for x in ("S", "M"):
            a, b = IJ[key][(x, 10)], IJ[key][(x, 90)]
            L.append(f"| {s} | {x} | {a[2]:+.0f} / {b[2]:+.0f} | {_f(a[0], 3)} ({a[1]:.3f}) | {_f(b[0], 3)} ({b[1]:.3f}) |")
    # --- conditional median split
    L.append("\nShape-free check: conditional median splits (mean over families of coupling in the high half minus the low "
             f"half; joint institution bootstrap CI, B = {B_BOOT}). 'S within M halves' = cells are first split at the "
             "family median of M, the S split is made inside each M half, and the two differences are averaged; likewise "
             "'M within S halves'.\n")
    L.append("| coupling | sample | high − low S (unconditional) | high − low M (unconditional) | S within M halves | M within S halves |")
    L.append("|---|---|---|---|---|---|")
    for tag, x, lab in [("B", "SAT", "ρ(SAT, log p50)"), ("W", "P", "ρ(P, log p50)"), ("W", "G", "ρ(G, log p50)")]:
        cells_ = []
        for kd in ("split", "msplit", "csplitS", "csplitM"):
            v = SUM.get((tag, f"{kd}_{x}_log_earn"))
            cells_.append(f"{_f(v['mean'])} {_ci(v['ci'])} (k = {v['k']})" if v is not None else "—")
        L.append(f"| {lab} | {tag} | " + " | ".join(cells_) + " |")
    return "\n".join(L) + "\n"


def _release_md(ctx):
    """Section: the primary statistics on the V4.14.1 (2026Q2) release."""
    REL, SUM, RI = ctx["REL"], ctx["SUM"], ctx["RI"]
    W, WS, Bs = ctx["W"], ctx["WS"], ctx["Bs"]
    L = ["### Release check: PSEO V4.14.1 (2026Q2)\n",
         "Flows, earnings and the institutions file all from the August 2026 release (V4.14.1; flows downloaded "
         "for this script, earnings and institutions from `data/raw/pseo_2026q2/`). Same project families, sample "
         f"rules, statistics and seeds; joint institution bootstrap ({B_BOOT} draws) for family means; θ with CR1 "
         f"CIs only. Flows institutions: {REL['n_inst_flows']:,} (V4.13.0: {ctx['opeid_match'][0]:,}). "
         f"W: {len(REL['W'])} cells / {REL['W'].institution.nunique()} institutions / {REL['W'].cip2.nunique()} "
         f"families (V4.13.0: {len(W)} / {W.institution.nunique()} / {W.cip2.nunique()}); WS: {len(REL['WS'])} / "
         f"{REL['WS'].institution.nunique()} (V4.13.0: {len(WS)} / {WS.institution.nunique()}); B: {len(REL['Bs'])} / "
         f"{REL['Bs'].institution.nunique()} (V4.13.0: {len(Bs)} / {Bs.institution.nunique()}). "
         f"W institutions added: {len(REL['new_W_inst'])}"
         + (f" ({'; '.join(REL['new_W_labels'])})" if REL["new_W_labels"] else "")
         + f"; dropped: {len(REL['lost_W_inst'])}. The two columns share most institutions, so they are not "
         "independent and their difference is not tested.\n",
         "| statistic | sample | V4.13.0 mean [95% CI] | V4.14.1 mean [95% CI] | V4.14.1 k fam. |", "|---|---|---|---|---|"]
    for tag, names in [("W", list(REL_STATS_W)), ("WS", list(REL_STATS_WS)), ("B", list(REL_STATS_B))]:
        for n in names:
            a, b = SUM[(tag, n)], REL["SUM"][(tag, n)]
            L.append(f"| {STAT_LABEL[n]} | {tag} | {_f(a['mean'])} {_ci(a['ci'])} | {_f(b['mean'])} {_ci(b['ci'])} | {b['k']} |")
    L.append("\n| interaction θ (per SD status, per +10 pp setting share) | sample | V4.13.0 θ, primary [CR1 95% CI] | "
             "V4.14.1 θ, primary [CR1 95% CI] | V4.14.1 θ, linear e [CR1 95% CI] | V4.14.1 cells / inst. |")
    L.append("|---|---|---|---|---|---|")
    RIL_ = ctx["RIL"]
    for lab, key, st, smp in [("P × S", "P", "P", "W"), ("G × S", "G", "G", "W"),
                              ("SAT × S", "SAT_b", "SAT", "B"),
                              ("SAT × S, + ADM/Pell/state/control (+ × S)", "SAT_b_sel", "SAT", "B"),
                              ("P × S, selectivity-adjusted", "P_sel", "P", "WS"),
                              ("joint: P × S", "joint", "P", "WS"), ("joint: G × S", "joint", "G", "WS"),
                              ("joint: SAT × S", "joint", "SAT", "WS")]:
        a, b, c = RI[key][st], REL["RI"][key][st], REL["RIL"][key][st]
        L.append(f"| {lab} | {smp} | {_f(a['theta'], 3)} {_ci(a['ci'], 3)} | {_f(b['theta'], 3)} {_ci(b['ci'], 3)} | "
                 f"{_f(c['theta'], 3)} {_ci(c['ci'], 3)} | {REL['RI'][key]['n_cells']} / {REL['RI'][key]['n_inst']} |")
    RJ_ = ctx["RJ"]
    L.append("\nJoint exposure models on V4.14.1 (second revision; primary shape; CR1 95% CI):\n")
    L.append("| θ | sample | V4.13.0 | V4.14.1 | V4.14.1 cells / inst. |")
    L.append("|---|---|---|---|---|")
    for key, s, smp in [("SAT_b_SM", "SAT", "B"), ("SAT_b_SML", "SAT", "B"), ("P_SM", "P", "W"), ("P_SML", "P", "W"),
                        ("G_SM", "G", "W")]:
        a, b = RJ_[key], REL["RJ"][key]
        sk = key.split("_")[-1]
        for x in [x_ for (s_, x_) in a["TH"] if s_ == s]:
            L.append(f"| {s} × {x}, model {JSET_LAB[sk]} | {smp} | {_vth(a['TH'][(s, x)])} | {_vth(b['TH'][(s, x)])} | "
                     f"{b['n_cells']} / {b['n_inst']} |")
    L.append("\nThe full functional-form battery on V4.14.1 is in the functional-form table above (row 'V4.14.1').\n")
    return "\n".join(L)


def _sig(c):
    """'excludes 0' test on a CI tuple."""
    return np.isfinite(c[0]) and (c[0] > 0 or c[1] < 0)


def _answer1(ctx):
    SUM, PAIR, WI, res_b = ctx["SUM"], ctx["PAIR"], ctx["WI"], ctx["res_b"]
    rP, rG, rPE, rGE = SUM[("W", "rP_M")], SUM[("W", "rG_M")], SUM[("W", "rP_E")], SUM[("W", "rG_E")]
    rSw, pPM, pPE, pGM = SUM[("WS", "rSAT_M_ws")], SUM[("WS", "pP_M")], SUM[("WS", "pP_E")], SUM[("WS", "pG_M")]
    rSb, rSEb, rPS, rSS = SUM[("B", "rSAT_M_b")], SUM[("B", "rSAT_E_b")], SUM[("W", "rP_S")], SUM[("B", "rSAT_S_b")]
    d = PAIR["W: rP_E - rP_M"]
    same = not _sig(d["ci"])
    d_sel = PAIR["WS: rP_M - pP_M"]                  # raw minus selectivity-partialled placement coupling
    d_sat = PAIR["WS: rSAT_M - rP_M"]                # SAT vs prestige placement coupling, same cells
    d_pe = PAIR["WS: pP_E - pP_M"]                   # partialled wage vs partialled placement coupling
    # "mostly institutional status": selectivity removes most of the prestige-placement coupling (CI of the
    # drop above 0 and the partial less than half the raw value) and the within-institution slope is under
    # half the across-institution slope.
    via_status = (d_sel["ci"][0] > 0 and abs(pPM["mean"]) < 0.5 * abs(SUM[("WS", "rP_M_ws")]["mean"])
                  and abs(WI["M"]["fe"][0]) < 0.5 * abs(WI["M"]["nofe"][0]))
    if rP["ci"][0] > 0:
        head = ("Yes, about as strongly as pay does" if same else
                "Yes, but with a strength that differs from wage coupling (see the paired contrast)")
        head += (", and in the same way: mostly through institutional status rather than a department's own "
                 "standing." if via_status else ".")
    else:
        head = "No clear coupling."
    sat_more = (f"Selectivity couples with market placement more than prestige does (paired difference on the "
                f"same WS cells {_f(d_sat['diff'])} {_ci(d_sat['ci'])}): " if d_sat["ci"][0] > 0 else
                f"Selectivity and prestige couple with market placement to a similar degree (paired difference on "
                f"the same WS cells {_f(d_sat['diff'])} {_ci(d_sat['ci'])}): ")
    fall_like = ("about where wage coupling falls" if not _sig(d_pe["ci"]) else
                 "not where wage coupling falls")
    fewer_S = ("Higher-status programs also send fewer graduates into setting-priced sectors"
               if rPS["ci"][1] < 0 and rSS["ci"][1] < 0 else
               "The association of status with the setting-priced share is")
    return (f"**Key question 1: does placement into market-priced sectors couple with prestige or brand?** {head} "
            f"Within field families (y5, pooled cohorts; W sample: {res_b['W']['n_cells']} institution × family cells, "
            f"{res_b['W']['n_inst']} Wapman-matched PSEO institutions, {rP['k']} families), the mean Spearman of "
            f"field-family prestige P with the share of employed graduates in information, finance and professional "
            f"services (M) is {_f(rP['mean'])} {_ci(rP['ci'])}; for academia-wide brand G it is {_f(rG['mean'])} "
            f"{_ci(rG['ci'])}. On the same cells wage coupling ρ(P, log p50) is {_f(rPE['mean'])} {_ci(rPE['ci'])} "
            f"(difference wage − placement {_f(d['diff'])} {_ci(d['ci'])}). {sat_more}"
            f"ρ(SAT, M) {_f(rSw['mean'])} {_ci(rSw['ci'])} on the WS cells, and "
            f"{_f(rSb['mean'])} {_ci(rSb['ci'])} across all {res_b['B']['n_inst']} PSEO institutions with an SAT "
            f"average (B sample; wage coupling ρ(SAT, log p50) there {_f(rSEb['mean'])} {_ci(rSEb['ci'])}). Net of "
            f"SAT, admit rate, Pell share, control and state earnings level, placement coupling with prestige falls "
            f"to {_f(pPM['mean'])} {_ci(pPM['ci'])} (brand {_f(pGM['mean'])} {_ci(pGM['ci'])}), {fall_like} "
            f"({_f(pPE['mean'])} {_ci(pPE['ci'])}; partial wage − partial placement {_f(d_pe['diff'])} {_ci(d_pe['ci'])}). "
            f"Within institutions (institution fixed effects), "
            f"one within-family SD of department prestige goes with {_f(WI['M']['fe'][0], 2)} pp (SE {WI['M']['fe'][1]:.2f}) "
            f"of market share, against {_f(WI['M']['nofe'][0], 2)} pp (SE {WI['M']['nofe'][1]:.2f}) across "
            f"institutions. {fewer_S} "
            f"(ρ(P, S) {_f(rPS['mean'])} {_ci(rPS['ci'])}; ρ(SAT, S) {_f(rSS['mean'])} {_ci(rSS['ci'])}).\n")


def _th(v, nd=3):
    return f"{_f(v['theta'], nd)} {_ci(v['ci'], nd)}"


def _carries_th(t: dict, s: str) -> str:
    """Which share a joint S + M fit's moderation loads on (CI below 0 for S | M, above 0 for M | S)."""
    s_, m_ = t[(s, "S")]["ci"][1] < 0, t[(s, "M")]["ci"][0] > 0
    return "both shares" if s_ and m_ else ("S" if s_ else ("M" if m_ else "neither share"))


COH_LAB = {"2010": "2010–12", "2013": "2013–15", "2016": "2016–18"}


def _coh(k: str) -> str:
    """Graduation-cohort label of a single-cohort timing sample ('2013–15')."""
    src = next(src for kk, _, src, _ in X_ALL if kk == k)
    return COH_LAB.get(src[1:5], src)


def _design_counts(ctx) -> dict:
    """Fourth revision: SAT x S | M and SAT x M | S (primary shape, B cells, y5 pay) counted by design. 'S' = CI of
    S | M below 0, 'M' = CI of M | S above 0. y5: the y5 shares on the cells of each timing sample; own: each
    cohort's own y1 shares and own y5 pay; mixed: one cohort's y1 shares with the pooled y5 pay; sel: the broad
    selectivity x share controls (all B cells at y5, and y5 and y1 shares on every timing sample)."""
    RJ, RX = ctx["RJ"], ctx["RX"]
    xs = [k for k, _, _, _ in X_SAMPLES]

    def sneg(t):
        return bool(t[("SAT", "S")]["ci"][1] < 0)

    def mpos(t):
        return bool(t[("SAT", "M")]["ci"][0] > 0)
    th = {k: RX[f"SAT_SM_{k}"]["TH"] for k, _, _, _ in X_ALL}
    th5 = [RX[f"SAT_SM5_{k}"]["TH"] for k in xs]
    sel = [RJ["SAT_b_SM_sel"]["TH"]] + [RX[f"SAT_SM{t_}_sel_{k}"]["TH"] for k in xs for t_ in ("5", "")]
    d = dict(n5=len(th5), n5S=sum(map(sneg, th5)), n5M=sum(map(mpos, th5)),
             n_own=len(X_OWN), ownS=sum(sneg(th[k]) for k in X_OWN), ownM=sum(mpos(th[k]) for k in X_OWN),
             own_neither=sum(not sneg(th[k]) and not mpos(th[k]) for k in X_OWN),
             n_mix=len(X_MIXED), mixS=sum(sneg(th[k]) for k in X_MIXED), mixM=sum(mpos(th[k]) for k in X_MIXED),
             n_sel=len(sel), selS=sum(map(sneg, sel)), selM=sum(mpos(t) for t in sel),
             n_sel_own=len(X_OWN), selS_own=sum(sneg(RX[f"SAT_SM_sel_{k}"]["TH"]) for k in X_OWN),
             selM_own=sum(mpos(RX[f"SAT_SM_sel_{k}"]["TH"]) for k in X_OWN),
             pooledS=sneg(th[X_MAIN]), pooledM=mpos(th[X_MAIN]),
             y5S=sneg(RJ["SAT_b_SM"]["TH"]), y5M=mpos(RJ["SAT_b_SM"]["TH"]))
    d["mix_both"] = sum(sneg(th[k]) and mpos(th[k]) for k in X_MIXED)
    for k, _, _, _ in X_CHECKS:
        d[f"{k}_S"], d[f"{k}_M"] = sneg(th[k]), mpos(th[k])
    # the all-cells check keeps (almost) every pooled-y1 cell; the strict one drops cells with a suppressed 2019-21
    # count, so it is compared with the pooled y1 shares on its own cells (SAT_SMP1)
    ka, ks = X_CHECKS[1][0], X_CHECKS[0][0]
    d["no19_same"] = d[f"{ka}_S"] == d["pooledS"] and d[f"{ka}_M"] == d["pooledM"]
    tp1 = RX[f"SAT_SMP1_{ks}"]["TH"]
    d["no19_strict_as_p1"] = d[f"{ks}_S"] == sneg(tp1) and d[f"{ks}_M"] == mpos(tp1)
    # only the pooled-cohort y1 design loads on S; every other design leans to M or to neither share
    d["lean_M"] = (d["pooledS"] and not d["pooledM"] and d["n5S"] == 0 and d["ownS"] == 0 and d["selS"] == 0
                   and d["ownM"] > 0 and d["n5M"] > 0 and d["selM"] > 0)
    return d


def _s_support(DC: dict) -> str:
    """Which designs give SAT x S | M a CI below 0 (all of them set y1 shares against the pooled-cohort y5 pay)."""
    parts = (["the pooled-cohort y1 shares"] if DC["pooledS"] else []) + (
        [f"one cohort's y1 shares in {DC['mixS']} of {DC['n_mix']} cohorts"
         + (f" ({DC['mix_both']} of them together with the market share)" if DC["mix_both"] else "")]
        if DC["mixS"] else [])
    return " and ".join(parts) if parts else "no design"


def _cell_sizes(ctx) -> dict:
    """Median employed graduates per B cell (y1 exposure cell and y5 outcome cell) of each timing sample."""
    out = {}
    for k, _, src, base in X_SAMPLES:
        b = ctx["Bs"] if base == "pooled_y5" else samp_B(ctx["CELLS"][base])
        x = expo_frame(b, ctx["CELLS"][src])
        out[k] = (float(x.emp_x.median()), float(x.emp.median()))
    return out


# the fourth-round verifier's reported numbers (quoted, not computed): own-cohort designs, the pooled y1 shares
# without the 2019-21 cohort, and the cohort composition of the pooled cells
VERIFIER4 = dict(x_c2013_same=dict(S=-0.0043, S_ci=(-0.0107, 0.0021), S_p=0.235, M=0.0175, M_ci=(0.0085, 0.0264),
                                   M_p=0.002, fS=2, fM=10),
                 x_c2016_same=dict(S=-0.0052, S_ci=(-0.0117, 0.0013), S_p=0.135, M=0.0088, M_ci=(-0.0003, 0.0180),
                                   M_p=0.084),
                 x_c2010_same=dict(S=-0.0063, M=0.0101),
                 no19=dict(S=-0.0093, S_ci=(-0.0151, -0.0034), M=0.0033, fS=10, fM=2),
                 sel13_M=0.018, sel13_M_ci=(0.006, 0.030),
                 flows_y5={"2013": 5858, "2016": 5846}, earn_y5={"2013": 6082, "2016": 6135},
                 ratio_all=1.00, ratio_x=1.81, share_1316=0.41, share_19=0.14)


def _answer2(ctx):
    RI, RIL, IMPL, FF, DIAG = ctx["RI"], ctx["RIL"], ctx["IMPL"], ctx["FF"], ctx["DIAG"]
    RJ, JB, COL, SUM = ctx["RJ"], ctx["JB"], ctx["COL"], ctx["SUM"]
    ffm, jb5 = FF["pooled_y5"], JB["pooled_y5"]
    tP, tG, tS, tPs = RI["P"]["P"], RI["G"]["G"], RI["SAT_b"]["SAT"], RI["P_sel"]["P"]
    lP, lG, lS = RIL["P"]["P"], RIL["G"]["G"], RIL["SAT_b"]["SAT"]
    tPn = RI["P_sel_noint"]["P"]
    jP, jG, jS = RI["joint"]["P"], RI["joint"]["G"], RI["joint"]["SAT"]
    nS, nSt = _nsig(ffm, "SAT_b", "SAT"); nP, nPt = _nsig(ffm, "P", "P"); nG, nGt = _nsig(ffm, "G", "G")
    s_lo, s_hi, _, _ = _theta_range(ffm, "SAT_b", "SAT")
    sdW, sdB = RI["P"]["sd_e"] * 10, RI["SAT_b"]["sd_e"] * 10
    iS = IMPL["SAT_b"]
    dec = {k: v["decline"] for k, v in iS.items()}
    zer = {k: v["zero_pp"] for k, v in iS.items() if np.isfinite(v["zero_pp"])}
    # ---- joint S + M: shares measured at y5 (second revision) and at y1 (third revision)
    RX, RXA, JBX, JB5X, FFX, TDX, SUMX = (ctx[k] for k in ("RX", "RXA", "JBX", "JB5X", "FFX", "TDX", "SUMX"))
    IMPLX = ctx["IMPLX"]
    k0 = X_MAIN
    jSM = RJ["SAT_b_SM"]["TH"]
    vS, vM = jSM[("SAT", "S")], jSM[("SAT", "M")]
    fS, fM = _jcount(jb5, "SAT_b", "SM", "SAT", "S", "-"), _jcount(jb5, "SAT_b", "SM", "SAT", "M", "+")
    z2 = jb5[("SAT_b", "SM", "e2z2")]["TH"]
    col = COL["SAT_b"]
    x1 = RX[f"SAT_SM_{k0}"]["TH"]
    xS, xM = x1[("SAT", "S")], x1[("SAT", "M")]
    gS1, gM1 = _jcount(JBX[k0], "SAT_b", "SM", "SAT", "S", "-"), _jcount(JBX[k0], "SAT_b", "SM", "SAT", "M", "+")
    a1 = RXA["SAT"]["SAT"]
    nA1 = _nsig(FFX[k0], "SAT_b", "SAT")
    DC = _design_counts(ctx)
    COMP = ctx["COMP"]
    kc, kca = X_CHECKS[0][0], X_CHECKS[1][0]

    def sm_txt(k):
        t = RX[f"SAT_SM_{k}"]["TH"]
        return (f"S | M {_vw(t[('SAT', 'S')])}, M | S {_vw(t[('SAT', 'M')])} "
                f"({_jcount(JBX[k], 'SAT_b', 'SM', 'SAT', 'S', '-')[0]} and "
                f"{_jcount(JBX[k], 'SAT_b', 'SM', 'SAT', 'M', '+')[0]} of {N_JFF} forms)")

    def carries(k):
        t = RX[f"SAT_SM_{k}"]["TH"]
        s_, m_ = t[("SAT", "S")]["ci"][1] < 0, t[("SAT", "M")]["ci"][0] > 0
        return "both shares" if s_ and m_ else ("S" if s_ else ("M" if m_ else "neither share"))
    xs_all = [k for k, _, _, _ in X_SAMPLES]
    n5S = sum(RX[f"SAT_SM5_{k}"]["TH"][("SAT", "S")]["ci"][1] < 0 for k in xs_all)
    n5M = sum(RX[f"SAT_SM5_{k}"]["TH"][("SAT", "M")]["ci"][0] > 0 for k in xs_all)
    n1S = sum(RX[f"SAT_SM_{k}"]["TH"][("SAT", "S")]["ci"][1] < 0 for k in xs_all)
    n1M = sum(RX[f"SAT_SM_{k}"]["TH"][("SAT", "M")]["ci"][0] > 0 for k in xs_all)
    # which pooled-y1 robustness rows keep / lose the S1 loading
    rob_keep = [lab for f_, lab in TIMING_ROBUST if RX[f"SAT_SM_{f_}_{k0}"]["TH"][("SAT", "S")]["ci"][1] < 0]
    rob_lose = [(lab, RX[f"SAT_SM_{f_}_{k0}"]["TH"][("SAT", "S")]) for f_, lab in TIMING_ROBUST
                if not RX[f"SAT_SM_{f_}_{k0}"]["TH"][("SAT", "S")]["ci"][1] < 0]
    ysel = RX[f"SAT_SM_sel_{k0}"]["TH"]
    yalso = RX[f"SAT_SM_y5also_{k0}"]["TH"]
    # broad selectivity x share controls under both timings (all B cells at y5 + y5 / y1 on each subsample)
    sel_rows = [RJ["SAT_b_SM_sel"]["TH"]] + [RX[f"SAT_SM{t_}_sel_{k}"]["TH"] for k in xs_all for t_ in ("5", "")]
    selS = sum(t[("SAT", "S")]["ci"][1] < 0 for t in sel_rows)
    selM = sum(t[("SAT", "M")]["ci"][0] > 0 for t in sel_rows)
    # sector-wage-mix index C and the low-pay share L by timing
    c5 = RJ["SAT_b_SC"]["TH"]
    cS1 = {k: RX[f"SAT_SC_{k}"]["TH"][("SAT", "S")] for k in xs_all}
    cS5n = sum(RX[f"SAT_SC5_{k}"]["TH"][("SAT", "S")]["ci"][1] < 0 for k in xs_all)
    cS1n = sum(v["ci"][1] < 0 for v in cS1.values())
    dL5 = RJ["SAT_b_SML"]["tests"]["SAT"]["S_minus_L"]
    dL1 = {k: RX[f"SAT_SML_{k}"]["tests"]["SAT"]["S_minus_L"] for k in xs_all}
    dL5n = sum(RX[f"SAT_SML5_{k}"]["tests"]["SAT"]["S_minus_L"]["ci"][0][1] < 0 for k in xs_all)
    dL1n = sum(v["ci"][0][1] < 0 for v in dL1.values())
    cs5, cm5 = SUMX[("x5", "csplitS_SAT_log_earn")], SUMX[("x5", "csplitM_SAT_log_earn")]
    cs1, cm1 = SUMX[("x1", "csplitS_SAT_log_earn")], SUMX[("x1", "csplitM_SAT_log_earn")]
    pSM, gSM = RJ["P_SM"]["TH"], RJ["G_SM"]["TH"]
    pX, gX = RX[f"P_SM_{k0}"]["TH"], RX[f"G_SM_{k0}"]["TH"]
    pgneg = sum(_jcount(JBX[k], key, "SM", s_, "S", "-")[0] for k in xs_all for key, s_ in (("P", "P"), ("G", "G")))
    pgtot = sum(_jcount(JBX[k], key, "SM", s_, "S", "-")[1] for k in xs_all for key, s_ in (("P", "P"), ("G", "G")))
    pgpos = sum(RX[f"{s_}_SM_{k}"]["TH"][(s_, "S")]["theta"] > 0 for k in xs_all for s_ in ("P", "G"))
    pga_n = {s_: [k for k in xs_all if FFX[k][(s_, ECURVE_MAIN)][s_]["ci"][1] < 0] for s_ in ("P", "G")}
    lab_s = {k: lab for k, lab, _, _ in X_SAMPLES}
    ia, ij, i5 = IMPLX["alone"], IMPLX["joint"], iS[ECURVE_MAIN]
    td = TDX[k0]
    tdo = {k: TDX[k] for k in X_OWN}
    s_robust = (tS["ci"][1] < 0 and a1["ci"][1] < 0 and nS == nSt and nA1[0] == nA1[1]
                and RI["SAT_b_sel"]["SAT"]["ci"][1] < 0)
    dec5 = (1 - i5["slope"]["p90"][0] / i5["slope"]["p10"][0]) * 100
    dec1 = (1 - ia["slope"]["p90"][0] / ia["slope"]["p10"][0]) * 100
    pg_robust = (nP == nPt) or (nG == nGt)
    if DC["lean_M"]:
        design_txt = ("The setting share carries it only when year-1 shares are set against the pooled-cohort y5 pay: "
                      + _s_support(DC) + "."
                      + (" The pooled result is unchanged when the 2019–21 cohort, which has no y5 yet, is removed "
                         "from the y1 shares." if DC["no19_same"] else "")
                      + " The y5 shares, each cohort's own y1 → y5 design (market share in "
                      f"{DC['ownM']} of {DC['n_own']} cohorts, setting share in none) and every fit with the broad "
                      f"selectivity × share controls (setting share in none of {DC['n_sel']}) lean toward the market "
                      "share or neither.")
    else:
        design_txt = ("The attribution depends on the design: y1 shares pooled over cohorts load on "
                      f"{_carries_th(x1, 'SAT')}; y5 shares give S | M below 0 in {DC['n5S']} and M | S above 0 in "
                      f"{DC['n5M']} of {DC['n5']} samples; each cohort's own y1 → y5 design {DC['ownS']} and "
                      f"{DC['ownM']} of {DC['n_own']}; the broad selectivity × share controls {DC['selS']} and "
                      f"{DC['selM']} of {DC['n_sel']} fits.")
    head = ("For selectivity, the SAT–earnings gradient is flatter in cells that send more graduates into setting-priced "
            "sectors" + (" (CI below 0 under every functional form, with y5 and with y1 shares)" if s_robust else "")
            + "; for prestige and brand it is " + ("" if pg_robust else "not robustly ") + "flatter. Which placement "
            "carries the SAT flattening, setting-priced or market-priced sectors, is not identified by public data. "
            + design_txt + " In magnitude the SAT "
            f"slope is {dec5:.0f}% lower at the within-family 90th than at the 10th percentile of the setting share "
            f"({dec1:.0f}% with y1 shares; S alone) and stays above zero for "
            f"{(1 - max(i5['share_beyond'], ia['share_beyond'])) * 100:.1f}% or more of cells.")
    t_alone = (f"With the setting share alone, θ for SAT is {_th(tS)} per SD of SAT per +10 pp with shares measured at "
               f"y5 (CI below 0 under {nS} of {nSt} functional forms; with the broad selectivity controls and their × S "
               f"terms {_th(RI['SAT_b_sel']['SAT'])}) and {_vw(a1)} with the year-1 shares of the same institution × "
               f"family (pooled cohorts; {nA1[0]} of {nA1[1]} forms).")
    c19, c19a, cP1 = RX[f"SAT_SM_{kc}"], RX[f"SAT_SM_{kca}"], RX[f"SAT_SMP1_{kc}"]
    g19 = (_jcount(JBX[kc], "SAT_b", "SM", "SAT", "S", "-")[0], _jcount(JBX[kc], "SAT_b", "SM", "SAT", "M", "+")[0])
    g19a = (_jcount(JBX[kca], "SAT_b", "SM", "SAT", "S", "-")[0], _jcount(JBX[kca], "SAT_b", "SM", "SAT", "M", "+")[0])
    dz_own = [tdo[k]["z_dM"] for k in X_OWN]
    dzS_own = [tdo[k]["z_dS"] for k in X_OWN]
    t_time = (f"Within families the two shares move against each other (pooled within-family correlation "
              f"{_f(col['e_S_M'])} at y5, {_f(td['S1_M1'])} at y1). One model with SAT × S and SAT × M, each share "
              f"with its own family-specific linear and quadratic terms (B cells, y5 pay, {N_JFF} functional forms), "
              "under four designs:\n"
              f"  - *y5 shares* (measured at the outcome horizon): loads on {_carries_th(jSM, 'SAT')}. θ(SAT × S | M) = "
              f"{_vw(vS)}, θ(SAT × M | S) = {_vw(vM)} (y5 pooled; CI below 0 for S | M in {fS[0]} of {fS[1]} forms, "
              f"above 0 for M | S in {fM[0]}). On the cells of the {DC['n5']} timing samples, the y5 shares give S | M "
              f"below 0 in {DC['n5S']} and M | S above 0 in {DC['n5M']}.\n"
              f"  - *y1 shares pooled over cohorts, y5 pay pooled over cohorts*: loads on {_carries_th(x1, 'SAT')}. "
              f"θ(SAT × S1 | M1) = {_vw(xS)}, θ(SAT × M1 | S1) = {_vw(xM)} ({gS1[0]} and {gM1[0]} of {N_JFF} forms). "
              "These are largely the same graduates at y1 and y5: the pooled y5 cell sums every graduation cohort "
              "through 2016–18, and the pooled y1 cell holds the same cohorts plus 2019–21 (median "
              f"{COMP['y1_x']['share_19'] * 100:.0f}% of its employed graduates). With the 2019–21 cohort removed "
              f"from the y1 shares ({c19a['n_cells']} cells; a suppressed 2019–21 count taken as 0): S1 | M1 "
              f"{_vw(c19a['TH'][('SAT', 'S')])}, M1 | S1 {_vw(c19a['TH'][('SAT', 'M')])} ({g19a[0]} and {g19a[1]} of "
              f"{N_JFF} forms). Dropping the cells whose 2019–21 count is suppressed ({c19['n_cells']} cells left): "
              f"{_vw(c19['TH'][('SAT', 'S')])} and {_vw(c19['TH'][('SAT', 'M')])} ({g19[0]} and {g19[1]} of {N_JFF} "
              f"forms); the pooled y1 shares, 2019–21 included, give {_vth(cP1['TH'][('SAT', 'S')])} and "
              f"{_vth(cP1['TH'][('SAT', 'M')])} on the same cells"
              + (", so the weaker result there comes from the smaller sample, not from removing the cohort"
                 if DC["no19_strict_as_p1"] else "") + ".\n"
              "  - *each cohort's own y1 shares and own y5 pay*: "
              + "; ".join(f"{_coh(k)} {sm_txt(k)}, i.e. {carries(k)}" for k in X_OWN)
              + f". S | M has a CI below 0 in {DC['ownS']} of {DC['n_own']} and M | S a CI above 0 in "
              f"{DC['ownM']}.\n"
              "  - *one cohort's y1 shares with the pooled y5 pay* (that cohort's placement against the pay of all "
              "cohorts): " + "; ".join(f"{_coh(k)} {sm_txt(k)}, i.e. {carries(k)}" for k in X_MIXED) + ".\n"
              f"  Within families, higher SAT goes with a change of the market share between y1 and y5 of r = "
              f"{_f(td['z_dM'])} (pooled; " + ", ".join(f"{_coh(k)} {_f(tdo[k]['z_dM'])}" for k in X_OWN)
              + f" in the own-cohort samples) and of the setting share of r = {_f(td['z_dS'])} ("
              + ", ".join(f"{_f(v)}" for v in dzS_own) + ")"
              + ("; the association is weak, but in the direction expected if the y5 market share is partly an "
                 "outcome of status." if td["z_dM"] > 0 and min(dz_own) > 0 else
                 "; the association is weak and its sign differs across samples."))
    ysel_keep = ysel[("SAT", "S")]["ci"][1] < 0
    z2_lose = not (z2[("SAT", "M")]["ci"][0] > 0)
    fe_lose = not (RJ["SAT_b_SM_instfe"]["TH"][("SAT", "M")]["ci"][0] > 0)
    neither = (not ysel_keep or bool(rob_lose)) and (z2_lose or fe_lose)
    t_notall = (("Neither attribution holds in every specification. " if neither else "")
                + "The pooled-y1 S loading keeps a CI below 0 with " + ", ".join(rob_keep)
                + f" (with the y5 shares and their products added: θ(SAT × S1) {_vw(yalso[('SAT', 'S')])}, SAT × S5 "
                f"{_vw(yalso[('SAT', 'S5')])}, SAT × M5 {_vw(yalso[('SAT', 'M5')])})"
                + (f"; it does not with {', '.join(f'{lab} ({_vth(v)})' for lab, v in rob_lose)}" if rob_lose else "")
                + (", and it keeps it " if ysel_keep else ", and it does not ")
                + f"with the broad selectivity controls and their × S1 and × M1 terms ({_vw(ysel[('SAT', 'S')])}). "
                + ("The y5 M loading " + ("disappears" if (z2_lose and fe_lose) else "does not hold in every row")
                   + f" with a family-specific SAT² term ({_vth(z2[('SAT', 'M')])}) or institution fixed effects "
                   f"({_vth(RJ['SAT_b_SM_instfe']['TH'][('SAT', 'M')])}). ")
                + "With the broad selectivity × share "
                f"controls, S | M has a CI below 0 in {selS} of {len(sel_rows)} fits (y5 and y1 shares, all samples) and "
                f"M | S a CI above 0 in {selM}.")
    def _neg(v):
        return v["ci"][1] < 0
    c5_abs = (not _neg(c5[("SAT", "S")])) and c5[("SAT", "C")]["ci"][0] > 0
    c1_keep = _neg(cS1[k0])
    l5_same, l1_diff = not dL5["ci"][0][1] < 0, dL1[k0]["ci"][0][1] < 0
    same_way = c5_abs and c1_keep and l5_same and l1_diff
    cS1own = sum(cS1[k]["ci"][1] < 0 for k in X_OWN)
    dL1own = sum(dL1[k]["ci"][0][1] < 0 for k in X_OWN)
    t_cl = ("Two checks that do not choose between S and M. "
            + f"The pay level of each cell's whole sector mix (C, the log of Σ_k share_k × the ACS median wage of young "
            f"graduates in sector k; ex ante, no class choice) "
            + ("absorbs S" if c5_abs else "does not absorb S") + " when both are measured at y5 (θ(SAT × S | C) "
            f"{_vw(c5[('SAT', 'S')])}, θ(SAT × C | S) {_vw(c5[('SAT', 'C')])}; S keeps a CI below 0 in {cS5n} of "
            f"{len(xs_all)} y5 subsamples); with y1 shares S "
            + ("keeps a CI below 0" if c1_keep else "does not keep a CI below 0")
            + f" ({_vw(cS1[k0])} for pooled cohorts; CI below 0 in {cS1n} of {len(xs_all)} y1 samples, {cS1own} of "
            f"{len(X_OWN)} own-cohort ones). Against the "
            f"low-pay private sectors (L, post hoc) θ_S − θ_L is {_f(dL5['est'][0], 3)} {_ci(dL5['ci'][0], 3)} at y5 "
            f"(CI below 0 in {dL5n} of {len(xs_all)} y5 subsamples) and {_f(dL1[k0]['est'][0], 3)} "
            f"{_ci(dL1[k0]['ci'][0], 3)} with pooled y1 shares (CI below 0 in {dL1n} of {len(xs_all)} y1 samples, "
            f"{dL1own} of {len(X_OWN)} own-cohort ones). "
            f"The shape-free conditional median split (pooled-y1 cells): splitting on S within halves of M changes "
            f"SAT–earnings coupling by {_f(cs5['mean'])} {_ci(cs5['ci'])} with y5 shares and {_f(cs1['mean'])} "
            f"{_ci(cs1['ci'])} with y1 shares; splitting on M within halves of S by {_f(cm5['mean'])} {_ci(cm5['ci'])} "
            f"and {_f(cm1['mean'])} {_ci(cm1['ci'])}.")
    t_pg = (f"For raw prestige and brand, the S-only flattening already depended on the shape of the exposure main "
            f"effect ({nP} of {nPt} forms with a CI below 0 for prestige, {nG} of {nGt} for brand; primary shape "
            f"{_th(tP)} and {_th(tG)}). With M in the model the prestige and brand × S terms are "
            + ("positive" if min(pSM[('P', 'S')]['theta'], gSM[('G', 'S')]['theta'], pX[('P', 'S')]['theta'],
                                 gX[('G', 'S')]['theta']) > 0 else "mixed in sign")
            + f" under both timings: θ(P × S | M) {_vth(pSM[('P', 'S')])} with y5 shares and {_vth(pX[('P', 'S')])} with "
            f"pooled y1 shares; θ(G × S | M) {_vth(gSM[('G', 'S')])} and {_vth(gX[('G', 'S')])}. Over the "
            f"{len(xs_all)} y1 samples × {N_JFF} forms × 2 status measures, "
            + (f"none of the {pgtot} fits has a CI below 0" if pgneg == 0 else
               f"{pgneg} of the {pgtot} fits have a CI below 0")
            + f", and the primary-shape point estimate is positive in {pgpos} of {2 * len(xs_all)}. "
            + f"With S alone (no M in the model) and y1 shares, θ has a CI below 0 for prestige in {len(pga_n['P'])} "
            f"and for brand in {len(pga_n['G'])} of the {len(xs_all)} timing samples"
            + (" (" + "; ".join(f"{lab_s[k]}: {s_} {_vth(FFX[k][(s_, ECURVE_MAIN)][s_])}"
                                 for s_ in ("P", "G") for k in pga_n[s_]) + ")" if pga_n["P"] or pga_n["G"] else "")
            + ". "
            + ("So once M is in the model there is no setting-specific flattening for department prestige or brand "
               "under either timing. "
               if pgneg <= 0.05 * pgtot and pSM[("P", "S")]["theta"] > 0 and gSM[("G", "S")]["theta"] > 0 else
               "So a setting-specific flattening for department prestige or brand is not established under either "
               "timing once M is in the model. ")
            + f"For the part of department prestige that selectivity leaves over (S only), θ = {_th(tPs)}.")
    t_mag = (f"Magnitude. With y1 shares the SAT slope is {_f(ia['slope']['p10'][0], 3)} at the within-family 10th "
             f"percentile of the setting share and {_f(ia['slope']['p90'][0], 3)} (SE {ia['slope']['p90'][1]:.3f}) at the "
             f"90th ({(ia['slope']['p90'][0] / ia['slope']['p10'][0] - 1) * 100:+.0f}%; S alone), and "
             f"{_f(ij[('S', 10)][0], 3)} and {_f(ij[('S', 90)][0], 3)} (SE {ij[('S', 90)][1]:.3f}) with M1 at its family "
             f"mean; it would reach 0 only {ia['zero_pp']:+.0f} pp above the family mean, which "
             f"{ia['share_beyond'] * 100:.1f}% of cells exceed (y5 shares: {_f(i5['slope']['p10'][0], 3)} → "
             f"{_f(i5['slope']['p90'][0], 3)}, zero at {i5['zero_pp']:+.0f} pp, {i5['share_beyond'] * 100:.1f}% of cells).")
    mix_txt = ", ".join(f"{_coh(k)} {carries(k)}" for k in X_MIXED)
    t_sum = ("In short: the S-only SAT flattening is " + ("robust" if s_robust else "not robust in every form")
             + ". Its attribution to setting-priced versus market-priced placement is not identified. Year-1 shares of "
             f"the same graduates load on {carries(k0)} when pooled over cohorts ({gS1[0]} of {N_JFF} forms for "
             f"S | M" + (", also with the 2019–21 cohort removed from the y1 shares" if DC["no19_same"] else "")
             + "), but on M in "
             f"{DC['ownM']} of {DC['n_own']} cohorts, on S in {DC['ownS']} and on neither in {DC['own_neither']} in each "
             f"cohort's own y1 → y5 design; y5 shares give S | M below 0 in {DC['n5S']} and M | S above 0 in "
             f"{DC['n5M']} of {DC['n5']} samples; with the broad selectivity × share controls S | M has a CI below 0 in "
             f"{DC['selS']} and M | S a CI above 0 in {DC['selM']} of {DC['n_sel']} fits"
             + (" (with these controls only the market-share term ever has a CI that excludes 0)"
                if DC["selS"] == 0 < DC["selM"] else "")
             + f". One cohort's y1 shares with the pooled y5 pay, which mix that cohort's placement with all cohorts' "
             f"pay, give: {mix_txt}. "
             + ("The setting share has a CI below 0 only when y1 shares are set against the pooled-cohort y5 pay; "
                "the other designs lean toward the market share or neither. " if DC["lean_M"] else "")
             + "The clause 'largely vanishes' is not supported in magnitude. θ is the change in the "
             "log-earnings slope per within-family SD of status for each +10 percentage points of a share, estimated "
             "within families.")
    L = [f"**Key question 2: does the pay-for-status gradient shrink with exposure to setting-priced sectors at the "
         f"cell level?** {head}\n",
         f"- **Setting share alone.** {t_alone}",
         f"- **Setting or market share: depends on the design (third and fourth revisions).** {t_time}",
         f"- **Robustness of each attribution.** {t_notall}",
         f"- **Sector-wage mix, low-pay sectors, median splits.** {t_cl}",
         f"- **Prestige and brand.** {t_pg}",
         f"- **{t_mag.split('. ', 1)[0]}.** {t_mag.split('. ', 1)[1]}",
         f"- {t_sum}\n"]
    L.append("- **What this means for the working headline (third and fourth revisions).** This analysis does not "
             "support 'it largely "
             "vanishes where pay is set by setting' as written. In magnitude: within families the SAT slope is "
             f"{dec5:.0f}% lower ({dec1:.0f}% with y1 shares) at the 90th than at the 10th percentile of setting exposure, "
             f"and at the 90th percentile it is still {_f(i5['slope']['p90'][0], 3)} (SE {i5['slope']['p90'][1]:.3f}) log "
             "points per SD of SAT. In attribution: whether the flattening reflects pay set by setting (status priced "
             "less inside schools, hospitals and government) or status sorting graduates into market-priced sectors "
             "(information, finance, professional services) is not identified, because the attribution changes with "
             "the design"
             + (" (the setting share carries it only when y1 shares are set against the pooled-cohort y5 pay; the "
                "y5 shares, each cohort's own y1 → y5 design and the selectivity-controlled fits lean toward the "
                "market share or neither)"
                if DC["lean_M"] else "")
             + " and PSEO does not release earnings by sector. What the "
             "data support is narrower: the pay-for-selectivity gradient is flatter in programs that place more of "
             "their graduates in education, health care and public administration"
             + (", and department prestige and brand show no robust flattening." if not pg_robust else "."))
    dP, dS = DIAG["P"], DIAG["SAT_b"]
    L.append(f"- **Specification of the exposure main effect (first revision).** Log p50 is curved in the setting share within families "
             f"(common e² term {_f(dP['e2c'], 3)} {_ci(dP['e2c_ci'], 3)} on W, {_f(dS['e2c'], 3)} {_ci(dS['e2c_ci'], 3)} "
             f"on B), and status is correlated with the share (within-family corr(z, e) {_f(DIAG['P']['corr_ze'])} for P, "
             f"{_f(DIAG['G']['corr_ze'])} for G, {_f(dS['corr_ze'])} for SAT), so with a linear exposure main effect "
             f"the product z·e partly stands in for the missing e² (corr(z·e, e²) {_f(dP['corr_zee2'])}, "
             f"{_f(DIAG['G']['corr_zee2'])}, {_f(dS['corr_zee2'])}). The primary model therefore has family-specific e² "
             f"terms; the linear-e fit is reported next to every row, nine functional forms are compared for the "
             f"S-only model and ten for the joint models.")
    rob = [("admit rate, Pell, control and state level and their × S terms", "SAT_b_sel"),
           ("state fixed effects", "SAT_b_state"), ("institution fixed effects", "SAT_b_instfe")]
    rob_s = "; ".join(f"{lab} {_th(RI[k]['SAT'])}" for lab, k in rob)
    hz = [(lab, FF[k][("SAT_b", ECURVE_MAIN)]["SAT"]) for k, lab in FF_SAMPLES[1:] if ("SAT_b", ECURVE_MAIN) in FF.get(k, {})]
    hz_s = "; ".join(f"{lab} {_th(v)}" for lab, v in hz)
    L.append(f"- **Setting share alone, SAT (B sample: {RI['SAT_b']['n_cells']} cells, {RI['SAT_b']['n_inst']} institutions, "
             f"{RI['SAT_b']['k']} families).** The SAT–earnings slope is {_f(tS['beta'], 3)} log points per SD "
             f"(SE {tS['beta_se']:.3f}) at the family-mean setting share and θ = {_th(tS)} (pairs-cluster bootstrap "
             f"{_ci(tS['boot_ci'], 3)}, cluster jackknife {_ci(tS['jack_ci'], 3)}; linear e: {_th(lS)}). "
             f"At the within-family 10th / 90th percentile of the setting share ({iS[ECURVE_MAIN]['slope']['p10'][2]:+.0f} / "
             f"{iS[ECURVE_MAIN]['slope']['p90'][2]:+.0f} pp from the family mean) the slope is "
             f"{_f(iS[ECURVE_MAIN]['slope']['p10'][0], 3)} / {_f(iS[ECURVE_MAIN]['slope']['p90'][0], 3)}, "
             f"{dec[ECURVE_MAIN]*100:.0f}% lower; it would reach zero at {iS[ECURVE_MAIN]['zero_pp']:+.0f} pp, which "
             f"{iS[ECURVE_MAIN]['share_beyond']*100:.1f}% of cells exceed. Across exposure shapes (linear, e², e³, "
             f"e-quintiles, e² + SAT²) the p10 → p90 decline ranges {min(dec.values())*100:.0f}%–{max(dec.values())*100:.0f}% "
             f"and the zero crossing +{min(zer.values()):.0f} to +{max(zer.values()):.0f} pp. These S-only figures describe "
             "cells with more setting-sector graduates, which are also cells with fewer market-sector graduates (see the "
             f"joint model above). With the primary shape: {rob_s}; {hz_s}. All of these rows have S only; the "
             "same rows with M added are in section (c') below.")
    hzp = [(lab, FF[k][("P", ECURVE_MAIN)]["P"]) for k, lab in FF_SAMPLES[1:] if ("P", ECURVE_MAIN) in FF.get(k, {})]
    n_hzp = sum(v["ci"][1] < 0 for _, v in hzp)
    sP, sG = SUM[("W", "split_P_log_earn")], SUM[("W", "split_G_log_earn")]
    L.append(f"- **Setting share alone, prestige and brand (W sample: {RI['P']['n_cells']} cells, {RI['P']['n_inst']} institutions).** "
             f"Prestige slope {_f(tP['beta'], 3)} (SE {tP['beta_se']:.3f}); θ = {_th(tP)} (bootstrap "
             f"{_ci(tP['boot_ci'], 3)}, jackknife {_ci(tP['jack_ci'], 3)}); linear e: {_th(lP)}. Brand: "
             f"θ = {_th(tG)} (linear e: {_th(lG)}). Other horizons, cohorts and the newer release with the primary shape "
             f"(prestige): " + "; ".join(f"{lab} {_th(v)}" for lab, v in hzp)
             + f" ({n_hzp} of {len(hzp)} with CI below 0). The median split (no functional form for S): coupling in the high-S "
             f"half minus the low-S half is {_f(sP['mean'])} {_ci(sP['ci'])} for ρ(P, log p50) and {_f(sG['mean'])} "
             f"{_ci(sG['ci'])} for ρ(G, log p50)" + (" (both CIs include 0)." if not (_sig(sP['ci']) or _sig(sG['ci'])) else "."))
    jj = RJ["joint_SM"]["TH"]
    ta = RJ["joint_SM"]["tests_all"]
    sat_rows = [k for k, r in RJ.items() if ("SAT", "S") in r["TH"] and ("SAT", "M") in r["TH"]
                and r["ecurve"] == ECURVE_MAIN]
    sat_neg = [k for k in sat_rows if RJ[k]["TH"][("SAT", "S")]["ci"][1] < 0]
    L.append(f"- **Which status measure carries it (primary shape).** In a horse race on the WS cells with S only, θ is {_th(jS)} "
             f"for SAT, {_th(jP)} for prestige and {_th(jG)} for brand; with S and M, θ(× S | M) is "
             f"{_vth(jj[('SAT', 'S')])} for SAT, {_vth(jj[('P', 'S')])} for prestige and {_vth(jj[('G', 'S')])} for brand, "
             f"and θ(× M | S) {_vth(jj[('SAT', 'M')])}, {_vth(jj[('P', 'M')])} and {_vth(jj[('G', 'M')])}"
             + (f". Of the {len(sat_rows)} y5 joint rows with SAT at the primary shape (section (c')), "
                + ("this is the only one" if sat_neg == ["joint_SM"] else f"{len(sat_neg)} ({', '.join(sat_neg)})")
                + " with a CI below 0 for SAT × S | M. It comes with a brand × S | M term of the opposite sign on "
                f"strongly correlated regressors (r(G, SAT) {_f(ctx['corr_zz']['G_SAT'])}); the three × S | M terms sum to "
                f"{_f(ta['S_sum']['est'][0], 3)} {_ci(ta['S_sum']['ci'][0], 3)} but are jointly non-zero "
                f"({_peq(ta['S_all']['p'])}), while the three × M | S terms are "
                + ("not jointly distinguishable from 0" if ta["M_all"]["p"] >= 0.05 else "jointly non-zero")
                + f" ({_peq(ta['M_all']['p'])}; sum {_f(ta['M_sum']['est'][0], 3)} {_ci(ta['M_sum']['ci'][0], 3)}). So on "
                f"the {RJ['joint_SM']['n_inst']} WS institutions, once all three status measures compete, the M loading "
                "does not show up and the S terms offset each other; the single-status SAT fit on the same WS cells gives "
                f"θ(SAT × S | M) = {_vth(RJ['SAT_ws_SM']['TH'][('SAT', 'S')])} and θ(SAT × M | S) = "
                f"{_vth(RJ['SAT_ws_SM']['TH'][('SAT', 'M')])}" if jj[("SAT", "S")]["ci"][1] < 0 else "")
             + ". With "
             f"family-specific selectivity slopes and their interactions with S, θ for prestige is {_th(tPs)}: the "
             f"department-prestige gradient that remains after "
             + ("selectivity shrinks with setting exposure" if tPs["ci"][1] < 0 else
                "selectivity does not shrink with setting exposure"
                + (" (it is steeper where the setting share is higher: the CI is above 0)" if tPs["ci"][0] > 0 else ""))
             + f"; with selectivity slopes by family but without their × S terms it is {_th(tPn)}. P and SAT are "
             f"correlated within families (pooled within-family Pearson r of the standardised scores "
             f"{_f(ctx['corr_zz']['P_SAT'])} on the WS cells; P and G {_f(ctx['corr_zz']['P_G'])}, G and SAT "
             f"{_f(ctx['corr_zz']['G_SAT'])}), so this split is less certain than the SAT result.")
    L.append(f"- One within-family SD of the setting share is {sdW:.1f} pp (W) and {sdB:.1f} pp (B).\n")
    return "\n".join(L)


VERIFIER3 = dict(S=-0.0095, M=+0.0023, n=4661, inst=348, S_ci=(-0.0154, -0.0036), M_ci=(-0.0052, +0.0098),
                 S_p=0.004, M_p=0.57)   # the third-round verifier's reported pooled-y1 numbers (quoted, not computed)


def _revision3_md(ctx) -> list:
    """Third revision: timing of the exposure (numbers from this run; the verifier's reported values are quoted)."""
    RJ, RX, JB, JBX, JB5X = ctx["RJ"], ctx["RX"], ctx["JB"], ctx["JBX"], ctx["JB5X"]
    k0 = X_MAIN
    r5, r55, r1 = RJ["SAT_b_SM"], RX[f"SAT_SM5_{k0}"], RX[f"SAT_SM_{k0}"]
    xS, xM = r1["TH"][("SAT", "S")], r1["TH"][("SAT", "M")]
    rep = (round(xS["theta"], 4) == VERIFIER3["S"] and round(xM["theta"], 4) == VERIFIER3["M"]
           and r1["n_cells"] == VERIFIER3["n"] and r1["n_inst"] == VERIFIER3["inst"]
           and all(round(a, 4) == b for a, b in zip(xS["ci"] + xM["ci"], VERIFIER3["S_ci"] + VERIFIER3["M_ci"])))
    L = ["### Third revision: when are the shares measured?\n",
         "(The cohort-coverage premise of this revision was wrong; see the fourth revision above.) "
         "Independent verification of the second revision found that its S-versus-M attribution had not been tested "
         "for the timing of the exposure. The second revision measured S and M at y5, in the same pooled cells whose "
         "y5 p50 is the outcome, and y5 shares can be partly an outcome of status (sorting into market sectors grows with "
         "career time). With the year-1 shares of the same institution × family as the exposure, the verifier found "
         "the SAT moderation loading on S, not M (reported: θ(SAT × S1 | M1) = -0.0095 [-0.0154, -0.0036], "
         "wild-cluster restricted bootstrap p = 0.004; θ(SAT × M1 | S1) = +0.0023 [-0.0052, +0.0098], p = 0.57; "
         f"4661 B cells, 348 institutions). This run: θ(SAT × S1 | M1) = {_vw(xS, 4)}, θ(SAT × M1 | S1) = "
         f"{_vw(xM, 4)} on {r1['n_cells']} cells / {r1['n_inst']} institutions"
         + (f" (the verifier's point estimates, CR1 CIs and sample reproduce to 4 decimals; the WCR p-values, "
            f"{_wp(xS)} and {_wp(xM)} against the verifier's {VERIFIER3['S_p']} and {VERIFIER3['M_p']}, differ by "
            "bootstrap draw)." if rep else
            " (this differs from the verifier's reported values; see section (c'') for the full set of fits).")
         + " Changes:\n",
         "1. New section (c''): the y5 outcome cells with the sector shares measured at year 1 (pooled cohorts, and "
         "the 2010–12, 2013–15 and 2016–18 graduation cohorts), plus one fixed-cohort sample (the 2010–12 graduates' "
         "own y1 shares and own y5 pay). Every y1 fit is shown next to the same model with the y5 shares of exactly "
         f"the same cells. For each sample: the {N_FF}-form S-only and {N_JFF}-form joint batteries (S + M, S + L, "
         "S + M + L; SAT, prestige, brand), the S + M + L and selectivity-control rows, and an ex-ante sector-wage-mix "
         "index C (log Σ_k share_k × ACS median wage of sector k). For pooled y1 shares also: the robustness rows of "
         "(c'), a model with the y5 shares and their products added, setting-sector components, the three-status "
         "model, implied slopes, a Gelbach decomposition, a conditional median split and timing diagnostics. Figure "
         "panel (h) is new.",
         f"2. Wild-cluster restricted bootstrap p-values (Rademacher weights by institution, B = {B_WCR}) for every θ "
         "in the joint rows of (c') and the timing rows of (c'').",
         "3. Key answer 2 was rewritten. Withdrawn from the second revision, because they hold only for shares "
         "measured at y5: 'in a joint model the moderation loads on the market-sector share'; 'setting-priced sectors "
         "behave like the other non-market sectors'; 'whether the other graduates go to schools, hospitals and "
         "government or to retail … makes no detectable difference'; 'it is not what carries the S-only result'; "
         "'gives no support to the headline clause'. Also rewritten: 'What this means for the working headline', "
         "'What else the data say' items 1–2, the caveats 'Composition vs pricing' and 'Many specifications', and "
         "the method (item 7c).",
         "4. Unchanged: part (b), every S-only row and every point estimate and CR1 CI of section (c'); the (c') "
         "joint rows gain the WCR p. The CSV gains the column `expo_source` (the cell table the shares come from), "
         "`theta_wcr_p`, the level `theta_timing_rows`, the theta_battery samples `x_*` (y1 shares) and `x5_*` (y5 "
         "shares on the same cells), the family levels `family_Bx1` / `family_Bx5` and the cell column C.\n",
         "| row (B cells, outcome log p50 at y5, primary shape; CR1 CI (WCR p)) | second revision: y5 shares, all B "
         "cells | y5 shares, cells with a y1 share | third revision and this version: y1 shares (pooled cohorts), "
         "same cells |",
         "|---|---|---|---|",
         f"| cells / institutions | {r5['n_cells']} / {r5['n_inst']} | {r55['n_cells']} / {r55['n_inst']} | "
         f"{r1['n_cells']} / {r1['n_inst']} |",
         f"| SAT × S \\| M | {_vw(r5['TH'][('SAT', 'S')])} | {_vw(r55['TH'][('SAT', 'S')])} | {_vw(xS)} |",
         f"| SAT × M \\| S | {_vw(r5['TH'][('SAT', 'M')])} | {_vw(r55['TH'][('SAT', 'M')])} | {_vw(xM)} |"]
    jbs = [JB["pooled_y5"], JB5X[k0], JBX[k0]]
    L.append(f"| forms (of {N_JFF}) with a CI below 0 for SAT × S \\| M / above 0 for SAT × M \\| S | "
             + " | ".join(f"{_jcount(j, 'SAT_b', 'SM', 'SAT', 'S', '-')[0]} / {_jcount(j, 'SAT_b', 'SM', 'SAT', 'M', '+')[0]}"
                          for j in jbs) + " |")
    sels = [RJ["SAT_b_SM_sel"], RX[f"SAT_SM5_sel_{k0}"], RX[f"SAT_SM_sel_{k0}"]]
    L.append("| + −ADM, Pell, state level, control and their × S, × M: SAT × S \\| M; SAT × M \\| S | "
             + " | ".join(f"{_vw(r['TH'][('SAT', 'S')])}; {_vw(r['TH'][('SAT', 'M')])}" for r in sels) + " |")
    sml = [RJ["SAT_b_SML"], RX[f"SAT_SML5_{k0}"], RX[f"SAT_SML_{k0}"]]
    L.append("| S + M + L: θ_S − θ_L | " + " | ".join(
        f"{_f(r['tests']['SAT']['S_minus_L']['est'][0], 3)} {_ci(r['tests']['SAT']['S_minus_L']['ci'][0], 3)}" for r in sml) + " |")
    L.append("| prestige (W cells): P × S \\| M; P × M \\| S | "
             + " | ".join(f"{_vth(r['TH'][('P', 'S')])}; {_vth(r['TH'][('P', 'M')])}"
                          for r in [RJ["P_SM"], JB5X[k0][("P", "SM", ECURVE_MAIN)], JBX[k0][("P", "SM", ECURVE_MAIN)]])
             + " |")
    L.append("")
    return L


def _rep4(r: dict, v: dict) -> bool:
    """Do this run's SAT x S | M and SAT x M | S (and CIs where quoted) match the verifier's to 4 decimals?"""
    t = r["TH"]
    ok = round(t[("SAT", "S")]["theta"], 4) == v["S"] and round(t[("SAT", "M")]["theta"], 4) == v["M"]
    if "S_ci" in v:
        ok = ok and all(round(a, 4) == b for a, b in zip(t[("SAT", "S")]["ci"], v["S_ci"]))
    if "M_ci" in v:
        ok = ok and all(round(a, 4) == b for a, b in zip(t[("SAT", "M")]["ci"], v["M_ci"]))
    return bool(ok)


def _revision4_md(ctx) -> list:
    """Fourth revision: cohort coverage of the pooled cells and the own-cohort designs (numbers from this run; the
    verifier's reported values are quoted)."""
    RX, JBX, COMP, NO19 = ctx["RX"], ctx["JBX"], ctx["COMP"], ctx["NO19"]
    DC = _design_counts(ctx)
    V = VERIFIER4
    c5, c1 = COMP["y5_B"], COMP["y1_x"]
    fr5, er5 = COMP["flows_rel_y5"], COMP["earn_rel_y5"]
    kc, kca = X_CHECKS[0][0], X_CHECKS[1][0]
    rep = {k: _rep4(RX[f"SAT_SM_{k}"], V[k]) for k in ("x_c2013_same", "x_c2016_same")}
    rep10 = _rep4(RX["SAT_SM_x_c2010_same"], V["x_c2010_same"])
    rep19 = {k: _rep4(RX[f"SAT_SM_{k}"], V["no19"]) for k in (kc, kca)}
    f13 = (_jcount(JBX["x_c2013_same"], "SAT_b", "SM", "SAT", "S", "-")[0],
           _jcount(JBX["x_c2013_same"], "SAT_b", "SM", "SAT", "M", "+")[0])
    sel13 = RX["SAT_SM_sel_x_c2013_same"]["TH"][("SAT", "M")]
    comp_ok = (fr5.get("2013") == V["flows_y5"]["2013"] and fr5.get("2016") == V["flows_y5"]["2016"]
               and er5.get("2013") == V["earn_y5"]["2013"] and er5.get("2016") == V["earn_y5"]["2016"])

    def vq(k):
        v = V[k]
        return (f"S | M {v['S']:+.4f} [{v['S_ci'][0]:+.4f}, {v['S_ci'][1]:+.4f}] (p {v['S_p']:.3f}), M | S "
                f"{v['M']:+.4f} [{v['M_ci'][0]:+.4f}, {v['M_ci'][1]:+.4f}] (p {v['M_p']:.3f})")

    def vt(k):
        t = RX[f"SAT_SM_{k}"]["TH"]
        return f"S | M {_vw(t[('SAT', 'S')], 4)}, M | S {_vw(t[('SAT', 'M')], 4)}"
    L = ["### Fourth revision (this version): which graduates the cells hold, and the own-cohort designs\n",
         "Independent verification of the third revision found a false premise in section (c''): it said that the "
         "2013–15 and 2016–18 cohorts do not reach y5 in this release, so that the y1 shares describe 'largely other "
         "graduates than those whose y5 pay is the outcome' (repeated in the caveat 'Composition vs pricing'). That "
         "is wrong for V4.13.0. The verifier reported: Flows agg 46 has status-1 y5 counts for the 2013 and 2016 "
         f"cohorts in {V['flows_y5']['2013']} and {V['flows_y5']['2016']} cells, Earnings releases their y5 p50 in "
         f"{V['earn_y5']['2013']} and {V['earn_y5']['2016']} bachelor's cells, the pooled y5 count equals the sum "
         f"over all cohorts (median ratio {V['ratio_all']:.2f}; {V['ratio_x']:.2f} without 2013 and 2016), and a "
         f"median {V['share_1316'] * 100:.0f}% of pooled-y5 employed graduates in the B cells come from those two "
         f"cohorts; the pooled y1 cell adds 2019–21 (median {V['share_19'] * 100:.0f}% of y1 employed). This run "
         f"reads {fr5.get('2013', 0)} and {fr5.get('2016', 0)} flows cells and {er5.get('2013', 0)} and "
         f"{er5.get('2016', 0)} earnings cells" + (" (the same counts)" if comp_ok else " (these differ)")
         + f", a median ratio of {c5['ratio_all']:.2f} ({c5['ratio_x']:.2f} without 2013 and 2016), a median share "
         f"of {c5['share_1316'] * 100:.0f}% for the two cohorts on {c5['n']} B cells and a median 2019–21 share of "
         f"{c1['share_19'] * 100:.0f}% of pooled-y1 employed on {c1['n']} B cells of the pooled-y1 sample (unreleased "
         "cohort counts as 0; the verifier's exact denominators are not known, so small differences in these shares "
         "are expected). So the pooled y1-share design compares largely the same graduates at y1 and y5, and each "
         "cohort's own y1 → y5 design was feasible for 2013–15 and 2016–18; the third revision ran it only for "
         f"2010–12. The verifier reported, for the own-cohort designs (primary shape, B cells, WCR p): 2013–15 "
         f"{vq('x_c2013_same')} ({V['x_c2013_same']['fS']} and {V['x_c2013_same']['fM']} of {N_JFF} forms; with the "
         f"selectivity × share controls M | S +{V['sel13_M']:.3f} [+{V['sel13_M_ci'][0]:.3f}, "
         f"+{V['sel13_M_ci'][1]:.3f}]); 2016–18 {vq('x_c2016_same')}; and with the 2019–21 cohort removed from the "
         f"pooled y1 shares S | M {V['no19']['S']:+.4f} [{V['no19']['S_ci'][0]:+.4f}, {V['no19']['S_ci'][1]:+.4f}], "
         f"M | S {V['no19']['M']:+.4f} ({V['no19']['fS']} and {V['no19']['fM']} of {N_JFF} forms). This run: 2013–15 "
         f"{vt('x_c2013_same')} ({f13[0]} and {f13[1]} of {N_JFF} forms; with the selectivity × share controls "
         f"M | S {_vth(sel13)}); 2016–18 {vt('x_c2016_same')}"
         + (" (point estimates and CIs reproduce to 4 decimals; WCR p-values differ by bootstrap draw)"
            if all(rep.values()) else " (not all of these reproduce the verifier's values to 4 decimals)")
         + f"; 2010–12 (unchanged from the third revision) {vt('x_c2010_same')}"
         + (" (the verifier's -0.0063 / +0.0101)" if rep10 else "")
         + f"; without the 2019–21 cohort (strict) {vt(kc)}, (suppressed 2019–21 counts as 0) {vt(kca)}"
         + (" (the all-cells version reproduces the verifier's values)" if rep19[kca] and not rep19[kc] else
            " (the strict version reproduces the verifier's values)" if rep19[kc] else
            " (neither version reproduces the verifier's values to 4 decimals; their exact construction is not known)")
         + (f"; on the strict cells the pooled y1 shares with the 2019–21 cohort give "
            f"{_vw(RX[f'SAT_SMP1_{kc}']['TH'][('SAT', 'S')], 4)} and {_vw(RX[f'SAT_SMP1_{kc}']['TH'][('SAT', 'M')], 4)}, "
            "so the strict version differs through its smaller sample, not through the removed cohort"
            if DC["no19_strict_as_p1"] else "")
         + ". Changes:\n",
         "1. Corrected the cohort-coverage statement in section (c'') and in the caveat 'Composition vs pricing'. "
         "Section (c'') now reports which cohorts the pooled cells hold (flows and earnings release counts by "
         "cohort, pooled-to-summed ratio, cohort shares); the cohort totals are read for every cohort (agg 46).",
         "2. New timing samples: the 2013–15 and 2016–18 cohorts' own y1 shares with their own y5 pay, run through "
         f"the whole timing battery ({N_FF}-form S-only and {N_JFF}-form joint batteries for SAT, prestige and brand; "
         "S + M + L; C; the broad selectivity × share controls; WCR p). The timing samples are grouped by design: "
         f"pooled (1), own cohort ({len(X_OWN)}), one cohort's y1 shares with the pooled y5 pay ({len(X_MIXED)}).",
         "3. New check: the pooled y1 shares with the 2019–21 cohort removed (pooled minus that cohort's released "
         "sector counts, clipped at 0; strict: cells whose 2019–21 count is suppressed are dropped; all cells: a "
         "suppressed count is taken as 0), with the full battery and, on the strict cells, the pooled y1 shares for "
         "comparison. Checks are not counted with the timing samples.",
         "4. Rewritten: Key answer 2 (the lead, 'Setting or market share', 'Prestige and brand' (adds the S-alone "
         "θ with y1 shares over the timing samples), 'In short', 'What this means for the working headline'), "
         "'What else the data say' item 2, the caveats 'Composition vs pricing' and 'Many "
         "specifications', method 7c, the figure panel (h). Withdrawn: 'the 2013–15 and 2016–18 cohorts do not reach "
         "y5 in this release'; 'these y1 shares describe the program's placement pattern, largely of other graduates "
         "than those whose y5 pay is the outcome' and 'the pooled y1 cell describes largely other cohorts than the y5 "
         "outcome'; 'shares measured at year 1 (pre-determined) load on S' as a general statement; 'the answer "
         "changes with when the shares are measured' (it changes with the design"
         + (": S only where y1 shares are set against the pooled-cohort y5 pay" if DC["lean_M"] else "")
         + "). The single-cohort row '2013–15 cohort, y1 shares … i.e. S' is kept but relabelled: it "
         "combines that cohort's y1 shares with the pooled y5 pay (a mixed design); with the cohort's own y5 pay it "
         f"loads on {_carries_th(RX['SAT_SM_x_c2013_same']['TH'], 'SAT')}.",
         "5. Unchanged: every sample, seed and random-stream tag of the earlier rows (the new samples and checks have "
         "their own tags); the timing samples of the third revision keep their CSV keys, with new labels. Checked "
         "against the third-revision CSV: all 64,873 of its rows reappear in this CSV with identical values (string "
         "match). The CSV "
         "gains the cell tables `c2013_y5`, `c2016_y5`, `pooled_y1_no19`, `pooled_y1_no19all` and the timing samples "
         "`x_c2013_same`, `x_c2016_same`, `x_p1no19`, `x_p1no19all` (and their `x5_*` y5-share versions).\n",
         "| design (SAT × share, primary shape, B cells, y5 pay; CR1 CI (WCR p)) | cells / inst. | SAT × S \\| M | "
         "SAT × M \\| S | forms S \\| M < 0 / M \\| S > 0 | loads on | + selectivity × share controls: S \\| M; "
         "M \\| S |",
         "|---|---|---|---|---|---|---|"]
    rows = [(X_MAIN, "pooled cohorts: y1 shares (third revision)")] + \
           [(k, f"check: {lab}") for k, lab, _, _ in X_CHECKS] + \
           [(k, lab + (" (third revision)" if k == "x_c2010_same" else " (new)")) for k in X_OWN
            for kk, lab, _, _ in X_SAMPLES if kk == k] + \
           [(k, lab + " (third revision)") for k in X_MIXED for kk, lab, _, _ in X_SAMPLES if kk == k]
    for k, lab in rows:
        r, rs = RX[f"SAT_SM_{k}"], RX[f"SAT_SM_sel_{k}"]
        t, ts = r["TH"], rs["TH"]
        L.append(f"| {lab} | {r['n_cells']} / {r['n_inst']} | {_vw(t[('SAT', 'S')])} | {_vw(t[('SAT', 'M')])} | "
                 f"{_jcount(JBX[k], 'SAT_b', 'SM', 'SAT', 'S', '-')[0]} / "
                 f"{_jcount(JBX[k], 'SAT_b', 'SM', 'SAT', 'M', '+')[0]} | {_carries_th(t, 'SAT')} | "
                 f"{_vw(ts[('SAT', 'S')])}; {_vw(ts[('SAT', 'M')])} |")
    L.append(f"\nCounts over the {len(X_SAMPLES)} timing samples (checks excluded): S | M has a CI below 0 in "
             f"{DC['ownS']} of {DC['n_own']} own-cohort designs and M | S a CI above 0 in {DC['ownM']}; with the y5 shares "
             f"of the same cells {DC['n5S']} and {DC['n5M']} of {DC['n5']}; with the broad selectivity × share controls "
             f"{DC['selS']} and {DC['selM']} of {DC['n_sel']} fits (all B cells at y5, and y5 and y1 shares on every "
             "timing sample).\n")
    return L


def _revision_md(ctx):
    """What changed relative to the first version of this analysis (numbers from this run)."""
    RI, RIL, IMPL = ctx["RI"], ctx["RIL"], ctx["IMPL"]
    iS = IMPL["SAT_b"]
    rows = [("P × S (W)", "P", "P"), ("G × S (W)", "G", "G"), ("SAT × S (B)", "SAT_b", "SAT"),
            ("P × S selectivity-adjusted (WS)", "P_sel", "P"), ("joint: SAT × S (WS)", "joint", "SAT"),
            ("SAT × S + ADM/Pell/state/control (B)", "SAT_b_sel", "SAT"), ("P × S, institution FE (W)", "P_instfe", "P"),
            ("P × S, + family slopes on M only (W)", "P_compM", "P"), ("SAT × S, + family slopes on M only (B)", "SAT_b_compM", "SAT"),
            ("P × M (W)", "P_M", "P"), ("SAT × M (B)", "SAT_b_M", "SAT")]
    RJ, JB = ctx["RJ"], ctx["JB"]
    jb5 = JB["pooled_y5"]
    vS, vM = RJ["SAT_b_SM"]["TH"][("SAT", "S")], RJ["SAT_b_SM"]["TH"][("SAT", "M")]
    L = ["## What changed\n"]
    L += _revision4_md(ctx)
    L += _revision3_md(ctx)
    L += ["### Second revision: setting share or market share\n",
         "Independent verification of the first revision found that its composition control (family-specific slopes "
         "on the market share M, without a status × M product; `extras=('M',), extras_int=False` in the code) could "
         "not test the competing explanation, namely that the status gradient is steeper where more graduates enter "
         "NAICS 51/52/54. In a model with both SAT × S and SAT × M the moderation loads on M. The verifier also found "
         "that the gradient flattens about as much toward low-pay private sectors. This run reproduces the verifier's "
         f"numbers (SAT, B, y5 pooled, family e² for each share: θ(SAT × S | M) = {_vth(vS, 4)}, θ(SAT × M | S) = "
         f"{_vth(vM, 4)}). Changes:\n",
         "1. Joint exposure models (status × S together with status × M, status × L, or both; each share with its own "
         f"family-specific main-effect shape) under {N_JFF} functional forms (the 9 of the S-only battery plus family e² "
         "with cross products of the shares), for SAT, prestige and brand, on all six samples, with robustness rows, "
         "bootstrap and jackknife CIs for the primary joint rows, Wald tests, a Gelbach decomposition and a shape-free "
         "conditional median split (section (c')). Figure panels (f) and (g) are new.",
         "2. Deleted: 'For SAT the flattening is not just the market share.' The row it rested on (SAT × S with "
         f"family-specific slopes on M, θ = {_th(RI['SAT_b_compM']['SAT'])}) holds M's main effect fixed but not the "
         f"SAT × M product. With the product in the model θ(SAT × S | M) = {_vth(vS)}.",
         "3. Key answer 2 reworded: the flattening was read as loading on the market share, so that public data "
         "could not tie it to setting-priced pay. The third revision withdraws the M reading as stated (it holds only "
         "for shares measured at y5) and the wording 'gives no support' / 'makes no detectable difference'. Also "
         "rewritten then: the caveat 'Composition vs pricing', 'What else the data say' items 1–4, the "
         "functional-form and many-specifications caveats, and the method.",
         "4. The low-pay private share L (NAICS 44-45, 56, 71, 72, 81) is post hoc; it is the verifier's definition. "
         "L2 (without 71 and 81, which have large non-profit shares) is a sensitivity check.",
         "5. The CSV gains the cell shares L and L2, the conditional-split family statistics, and one row per "
         "interaction coefficient of the functional-form batteries and joint rows (levels `theta_battery`, "
         "`theta_joint_rows`).\n",
         "| row (y5 pooled, family e²) | first revision | this version |", "|---|---|---|",
         f"| SAT × S (B) | S only: {_th(RI['SAT_b']['SAT'])}; with M main effects only: {_th(RI['SAT_b_compM']['SAT'])} | "
         f"with SAT × M: {_vth(vS)} |",
         f"| SAT × M (B) | M only: {_th(RI['SAT_b_M']['SAT'])} | with SAT × S: {_vth(vM)} |",
         f"| P × S (W) | S only: {_th(RI['P']['P'])}; with M main effects only: {_th(RI['P_compM']['P'])} | "
         f"with P × M: {_vth(RJ['P_SM']['TH'][('P', 'S')])} |",
         f"| P × M (W) | M only: {_th(RI['P_M']['P'])} | with P × S: {_vth(RJ['P_SM']['TH'][('P', 'M')])} |",
         f"| forms with CI below 0 for SAT × S (y5) | S only: {_nsig(ctx['FF']['pooled_y5'], 'SAT_b', 'SAT')[0]} of "
         f"{_nsig(ctx['FF']['pooled_y5'], 'SAT_b', 'SAT')[1]} | given M: {_jcount(jb5, 'SAT_b', 'SM', 'SAT', 'S', '-')[0]} of "
         f"{_jcount(jb5, 'SAT_b', 'SM', 'SAT', 'S', '-')[1]} |",
         "\n### First revision: exposure main-effect shape\n",
         "Independent verification found that the first version's cell-level interaction model had only a linear "
         "main effect of the setting share e, while log p50 is curved in e and status is correlated with e within "
         "families, so θ partly absorbed the omitted e² (a spurious-interaction pattern). The first version added "
         "status² as a check but not e². Changes:\n",
         "1. The primary interaction model now includes family-specific e² terms; every robustness row uses it, and "
         "the linear-e fit is shown alongside.",
         "2. New functional-form battery (linear; common e²; family e²; family e², e³; family × e-quintile dummies, "
         "with and without family × status-quintile dummies; family e² + status²; rank × rank with and without "
         "(rank S)²) at y5, y1, y10, the 2010–12 cohort and on V4.14.1; cluster-jackknife CIs for the primary rows; "
         "leave-one-family-out for brand; a median split with S partialled within each half; implied-slope "
         "figures by exposure shape.",
         "3. Key answer 2 was reworded: on the main sample the prestige and brand flattening does not survive "
         "curvature in S; the SAT result holds in sign but its size is specification-dependent. Part (b) (placement "
         "coupling) was unchanged; its statistics and bootstrap draws were identical to the first version.\n",
         "| row | θ, first version (linear e) [CR1 CI] | θ, first revision and this version (family e²) [CR1 CI] |",
         "|---|---|---|"]
    for lab, k, s in rows:
        L.append(f"| {lab} | {_th(RIL[k][s])} | {_th(RI[k][s])} |")
    L.append(f"\nSAT slope decline from the 10th to the 90th percentile of S: {iS['lin']['decline']*100:.0f}% (linear e) → "
             f"{iS[ECURVE_MAIN]['decline']*100:.0f}% (family e²); zero crossing {iS['lin']['zero_pp']:+.0f} pp → "
             f"{iS[ECURVE_MAIN]['zero_pp']:+.0f} pp; cells beyond it {iS['lin']['share_beyond']*100:.1f}% → "
             f"{iS[ECURVE_MAIN]['share_beyond']*100:.1f}%.\n")
    return "\n".join(L)


def _answer_detail(ctx):
    SUM, RI, ORDER, HROB, lg = ctx["SUM"], ctx["RI"], ctx["ORDER"], ctx["HROB"], ctx["legacy"]
    o = ORDER[("rP_M", "rP_E")]
    sP, sS = SUM[("W", "split_P_log_earn")], SUM[("B", "split_SAT_log_earn")]
    names = {"61": "education", "62": "health care", "92": "public administration"}
    comp = {k: RI["P_s" + k]["P"] for k in SETTING}; compS = {k: RI["SAT_b_s" + k]["SAT"] for k in SETTING}
    cM, cMS = RI["P_compM"]["P"], RI["SAT_b_compM"]["SAT"]
    h1, h10 = HROB["c2010_y1"], HROB["c2010_y10"]

    def neg(d):
        return [names[k] for k in SETTING if d[k]["ci"][1] < 0]

    def fmt(d):
        return ", ".join(f"{names[k]} {_f(d[k]['theta'], 3)} {_ci(d[k]['ci'], 3)}" for k in SETTING)
    nS, nP = neg(compS), neg(comp)
    L = ["**What else the data say.**\n"]
    RIL = ctx["RIL"]
    mP, mS = RI["P_M"]["P"], RI["SAT_b_M"]["SAT"]
    mir = ("The mirror image holds for market exposure" if mP["ci"][0] > 0 and mS["ci"][0] > 0 else
           "For market exposure the mirror image holds for SAT only" if mS["ci"][0] > 0 else
           "For market exposure there is no clear mirror image")
    RJ = ctx["RJ"]
    compSM = {k: RJ[f"SAT_b_s{k}_M"]["TH"][("SAT", f"sh_{k}")] for k in SETTING}
    compPM = {k: RJ[f"P_s{k}_M"]["TH"][("P", f"sh_{k}")] for k in SETTING}
    nSM, nPM = neg(compSM), neg(compPM)
    RX = ctx["RX"]
    k0 = X_MAIN
    compSM1 = {k: RX[f"SAT_s{k}_M_{k0}"]["TH"][("SAT", f"sh_{k}")] for k in SETTING}
    nSM1 = neg(compSM1)
    L.append(f"1. **Sector components** (each share entered alone with its own family-specific e², θ per +10 pp). "
             f"SAT: {fmt(compS)}; CI below 0 for {', '.join(nS) if nS else 'none'}. Prestige: {fmt(comp)}; CI below 0 "
             f"for {', '.join(nP) if nP else 'none'}. With M and SAT × M in the model, shares at y5 (second revision): SAT "
             f"{fmt(compSM)} (CI below 0 for {', '.join(nSM) if nSM else 'none'}); prestige {fmt(compPM)} (CI below 0 "
             f"for {', '.join(nPM) if nPM else 'none'}). The same SAT model with pooled y1 shares (third revision): "
             f"{fmt(compSM1)} (CI below 0 for {', '.join(nSM1) if nSM1 else 'none'}). {mir}: θ for the market share M "
             f"alone is {_th(mP)} (prestige; linear e: {_th(RIL['P_M']['P'])}) and {_th(mS)} (SAT; linear e: "
             f"{_th(RIL['SAT_b_M']['SAT'])}).")
    jS_, jM_ = RJ["SAT_b_SM"]["TH"][("SAT", "S")], RJ["SAT_b_SM"]["TH"][("SAT", "M")]
    xS_, xM_ = RX[f"SAT_SM_{k0}"]["TH"][("SAT", "S")], RX[f"SAT_SM_{k0}"]["TH"][("SAT", "M")]
    gb = ctx["GB"]["SAT_b_M"]
    gme = sum(v for n, v in gb["parts"].items() if not n.startswith("product"))
    gpr = sum(v for n, v in gb["parts"].items() if n.startswith("product"))
    gx = ctx["GBX"]["SAT_M"]
    gxme = sum(v for n, v in gx["parts"].items() if not n.startswith("product"))
    gxpr = sum(v for n, v in gx["parts"].items() if n.startswith("product"))
    DC = _design_counts(ctx)
    own_txt = "; ".join(f"{_coh(k)} S | M {_vth(RX[f'SAT_SM_{k}']['TH'][('SAT', 'S')])}, M | S "
                        f"{_vth(RX[f'SAT_SM_{k}']['TH'][('SAT', 'M')])}" for k in X_OWN)
    L.append(f"2. **Composition (rewritten in the second, third and fourth revisions).** PSEO does not release earnings "
             f"by sector, so a flatter gradient in high-S cells can come from flatter pay-for-status inside setting-priced "
             f"employers or from status-based sorting into high-paying market sectors. The first revision held M fixed "
             f"only through family-specific main effects (θ(SAT × S) {_th(cMS)}, prestige {_th(cM)}) and read the SAT "
             f"result as 'not just the market share'; that was wrong, because it did not include the SAT × M product. "
             f"The second revision added the product with y5 shares: θ(SAT × S | M) = {_vw(jS_)} and θ(SAT × M | S) = "
             f"{_vw(jM_)}; of the change in θ(SAT × S) from {_f(gb['theta_alone'], 3)} (S alone) to "
             f"{_f(gb['theta_joint'], 3)}, {_f(gme, 3)} comes from M's main effects and {_f(gpr, 3)} from the SAT × M "
             f"product. With the y1 shares of the same institution × family pooled over cohorts (third revision) the "
             f"same model gives θ(SAT × S1 | M1) = {_vw(xS_)} and θ(SAT × M1 | S1) = {_vw(xM_)}; adding M1 moves "
             f"θ(SAT × S1) from {_f(gx['theta_alone'], 3)} to {_f(gx['theta_joint'], 3)} ({_f(gxme, 3)} from M1's main "
             f"effects, {_f(gxpr, 3)} from the SAT × M1 product). With each cohort's own y1 shares and own y5 pay "
             f"(fourth revision): {own_txt} (S | M below 0 in {DC['ownS']} of {DC['n_own']}, M | S above 0 in "
             f"{DC['ownM']}). So "
             + ("S carries it only in the designs that set y1 shares against the pooled-cohort y5 pay (section "
                "(c''))" if DC["lean_M"] else
                "the share that carries the flattening depends on the design (section (c''))")
             + ", and no design separates the channels: status already predicts the y1 shares, the y5 shares are "
             "partly an outcome of status, and pay inside sectors is not observed.")
    sG = SUM[("W", "split_G_log_earn")]
    qS, qP, qG = SUM[("B", "psplit_SAT_log_earn")], SUM[("W", "psplit_P_log_earn")], SUM[("W", "psplit_G_log_earn")]
    L.append(f"3. **Median split** (coupling in cells above the family median of S minus coupling at or below it; no "
             f"functional form for S needed): ρ(SAT, log p50) {_f(sS['mean'])} {_ci(sS['ci'])} ({sS['k']} families); "
             f"ρ(P, log p50) {_f(sP['mean'])} {_ci(sP['ci'])}; ρ(G, log p50) {_f(sG['mean'])} {_ci(sG['ci'])} "
             f"({sP['k']} families). With S partialled within each half: SAT {_f(qS['mean'])} {_ci(qS['ci'])} "
             f"(k = {qS['k']}), P {_f(qP['mean'])} {_ci(qP['ci'])} (k = {qP['k']}), G {_f(qG['mean'])} {_ci(qG['ci'])} "
             f"(k = {qG['k']})"
             + (". The split supports the S-only SAT result and not the prestige or brand one. Split on S within halves of "
                f"M, the SAT difference is {_f(SUM[('B', 'csplitS_SAT_log_earn')]['mean'])} "
                f"{_ci(SUM[('B', 'csplitS_SAT_log_earn')]['ci'])}; split on M within halves of S it is "
                f"{_f(SUM[('B', 'csplitM_SAT_log_earn')]['mean'])} {_ci(SUM[('B', 'csplitM_SAT_log_earn')]['ci'])} "
                "(section (c'))." if sS["ci"][1] < 0 and not sP["ci"][1] < 0
                and not sG["ci"][1] < 0 else "."))
    L.append(f"4. **Family grain.** Across the {lg['k']} families, family mean S and wage coupling ρ(P, log p50) have "
             f"Spearman {_f(lg['rho'])} (SAT coupling on B: {_f(lg['rho_b'])}, k = {lg['k_b']}), the same direction as "
             f"the S-only cell-level θ, which uses within-family variation only; family mean M has Spearman "
             f"{_f(lg['rho_M'])} with ρ(P, log p50) ({_f(lg['rho_b_M'])} with SAT coupling on B), and family mean S and M "
             f"have {_f(lg['rho_SM'])}, so the family grain cannot separate S from M. Wald tests of equal family-specific "
             f"θ_c (S only): "
             f"SAT {_peq(ctx['FT_S_w']['p'])}, prestige {_peq(ctx['FT_P_w']['p'])} (per-family table below). "
             f"Leaving out one family at a time, the SAT θ has a CI below 0 in "
             f"{sum(t[3] for t in ctx['LOFO']['SAT_b'])}/{len(ctx['LOFO']['SAT_b'])} fits, the prestige θ in "
             f"{sum(t[3] for t in ctx['LOFO']['P'])}/{len(ctx['LOFO']['P'])} and the brand θ in "
             f"{sum(t[3] for t in ctx['LOFO']['G'])}/{len(ctx['LOFO']['G'])} (ranges below).")
    if o["p"] < 0.05:
        otxt = (f"families with stronger prestige–placement coupling also have stronger prestige–wage coupling: "
                f"split-half Spearman {_f(o['cross'])} (null mean {_f(o['null_mean'])}, {_peq(o['p'])}, k = {o['k']}).")
    else:
        otxt = (f"the association between prestige–placement coupling and prestige–wage coupling is "
                f"{_f(o['cross'])} (null mean {_f(o['null_mean'])}, {_peq(o['p'])}, k = {o['k']}); not distinguishable "
                f"from zero at {o['k']} families.")
    oS = ORDER[("rP_S", "rP_E")]
    otxt += (f" Families where higher prestige goes with a lower setting share have stronger prestige–wage coupling "
             f"(split-half Spearman of ρ(P, S) with ρ(P, log p50) {_f(oS['cross'])}, null mean {_f(oS['null_mean'])}, "
             f"{_peq(oS['p'])}).")
    L.append("5. **Placement ordering vs wage ordering across families.** On independent halves of institutions, " + otxt)
    early = h1["rP_M"] > h1["rP_E"] and h1["rSAT_M"] > h1["rSAT_E"]
    L.append(f"6. **Horizon.** On the fixed 2010–12 cohort, placement coupling ρ(P, M) is {_f(h1['rP_M'])} at y1 "
             f"and {_f(h10['rP_M'])} at y10 while wage coupling ρ(P, log p50) goes {_f(h1['rP_E'])} → {_f(h10['rP_E'])} "
             f"(SAT: placement {_f(h1['rSAT_M'])} → {_f(h10['rSAT_M'])}, wage {_f(h1['rSAT_E'])} → {_f(h10['rSAT_E'])}). "
             + ("Sorting into market sectors by status is already there in year 1, when the pay gradient is still "
                "small; the pay gradient grows faster afterwards. " if early else "")
             + "Point estimates without CIs; one cohort.")
    REL = ctx["REL"]
    rw, rb = REL["SUM"][("W", "rP_M")], REL["SUM"][("B", "rSAT_M_b")]
    tp, ts, tg = REL["RI"]["P"]["P"], REL["RI"]["SAT_b"]["SAT"], REL["RI"]["G"]["G"]
    lp, ls = REL["RIL"]["P"]["P"], REL["RIL"]["SAT_b"]["SAT"]
    L.append(f"7. **Newer release.** On PSEO V4.14.1 (2026Q2; W {REL['W'].institution.nunique()} institutions, "
             f"B {REL['Bs'].institution.nunique()}): ρ(P, M) {_f(rw['mean'])} {_ci(rw['ci'])}, ρ(SAT, M) {_f(rb['mean'])} "
             f"{_ci(rb['ci'])}; primary shape θ(P × S) {_th(tp)}, θ(G × S) {_th(tg)}, θ(SAT × S) {_th(ts)}; with a "
             f"linear e, θ(P × S) {_th(lp)} and θ(SAT × S) {_th(ls)}. "
             + ("The significance pattern is the same as on V4.13.0, including the dependence of the prestige result "
                "on the exposure shape " if ((lp["ci"][1] < 0) == (RIL["P"]["P"]["ci"][1] < 0)
                                             and (tp["ci"][1] < 0) == (RI["P"]["P"]["ci"][1] < 0)
                                             and (ts["ci"][1] < 0) == (RI["SAT_b"]["SAT"]["ci"][1] < 0))
                else "The significance pattern differs from V4.13.0 in at least one of these rows ")
             + "(release-check and functional-form tables below).")
    return "\n".join(L) + "\n"


def _method(ctx):
    W, Bs = ctx["W"], ctx["Bs"]
    return ("## Method\n\n"
            "1. **Flows.** `pseof_all.csv.gz` streamed with pyarrow; bachelor's (degree_level 05), institution rows, "
            "agg_level_pseo 88 (institution × CIP-2 × NAICS sector, pooled cohorts), 94 (same by 3-year graduation "
            "cohort), 40/46 (all-sector totals incl. non/marginally employed). Counts used only when the status flag "
            "is 1. Shares are sector counts over the sum of the 20 sectors (which equals the released employed total). "
            "S = 61+62+92, M = 51+52+54, M+ = M+55.\n"
            "2. **Earnings.** `pseoe_all.csv.gz`, agg 40/46: median (p50) annual earnings of the same institution × "
            "CIP-2 × cohort × horizon cell, released cells only; log p50 is the outcome. Primary horizon y5, pooled "
            "cohorts (\"0000\" pools the cohorts that reach the horizon; flows and earnings pool the same ones). "
            "Robustness: y1, y10 pooled; fixed 2010–12 cohort at y1/y5/y10.\n"
            "3. **Institutions.** PSEO institution id = 8-digit OPEID → name → `normalize_institution_name` → "
            "Wapman ranks (field and academia-wide) by normalised name, as in scripts/28/52; Scorecard institution "
            "file joined on OPEID (main campus, then largest enrolment, when several UNITIDs share an OPEID; "
            f"{ctx['opeid_match'][1]} of {ctx['opeid_match'][0]} bachelor's flows institutions match an OPEID).\n"
            "4. **Family prestige P.** For institution i and CIP-2 family c: IPEDS-weighted mean (PSEO pooled "
            "y1_ipeds_count of the field's CIP-4 codes at i) of the within-field Wapman percentile 1 − Rank/max(Rank) "
            "over the FIELDS66 fields of c ranked at i (CIP-4 → field map `cip4_to_field_key(FIELDS66)`). P_u = "
            "unweighted mean over all ranked fields of c at i. Within-family Spearman statistics are invariant to "
            "monotone transformations, so only the within-family ordering of P matters there.\n"
            f"5. **Samples.** Cells need ≥ {EMP_MIN} employed graduates at the horizon and a released p50; families "
            f"need ≥ {NMIN} cells. W: P and G defined. WS: W plus Scorecard SAT_AVG, ADM_RATE, PCTPELL, CONTROL, state "
            "earnings level. B: every PSEO institution with SAT_AVG and ADM_RATE (no Wapman requirement).\n"
            f"6. **Family statistics.** Spearman within family; partial versions rank all variables and residualise on "
            f"ranked SAT, −ADM, Pell, state level and control dummies (residual df ≥ {DFMIN}). Means across families "
            f"with a joint institution bootstrap ({B_BOOT} draws; institutions resampled once per draw for all "
            f"families, so the dependence between families that share institutions is kept). Cross-family ordering by "
            f"split-half cross-fit ({K_SPLIT} splits × 2 directions, {N_PERM} family-label permutations).\n"
            "7. **Interaction model.** OLS of log p50 on family intercepts, family-specific slopes on z_c(status), on "
            "e (setting share, 10-pp units, centred within family) and on e², and a common θ on z_c(status)·e. "
            "Because the main effects are family-specific, θ is identified from within-family variation only. The "
            "family-specific e² is the primary exposure shape since this revision (the first version had a linear e "
            "only; see 'What changed'); the functional-form battery replaces it by a common e², family e² and e³, "
            "family × within-family e-quintile dummies (with and without family × status-quintile dummies), family "
            "e² plus family status², and a rank × rank version (status and S as within-family standardised ranks, "
            "with and without family (rank S)²). SEs clustered by institution (CR1); pairs-cluster bootstrap "
            f"({B_CLU} draws, design held fixed) and delete-one-institution jackknife for the key rows; "
            "random-intercept mixed model (REML) as a multilevel check; family-specific θ_c with a Wald test; "
            "leave-one-family-out. Selectivity adjustment: family-specific slopes on z_c(SAT), z_c(−ADM), z_c(Pell), "
            "z_c(state level) and a private-control indicator, plus each of these × e. Implied slopes at the "
            "within-family 10th / 90th percentile of e are β̄ + θ e (cell-weighted β̄), at the family-mean status.\n"
            "7b. **Joint exposure models (second revision).** The same model with further shares x ∈ {M, L} (L2, M+ as "
            "sensitivities): each share centred within family in 10-pp units, with its own family-specific linear term "
            "and the same family-specific shape (the form under test), and its own common product θ_x z_c(status)·e_x. "
            f"Battery of {N_JFF} forms: the 9 S-only forms applied to every share, plus family e² of every share with "
            "family-specific cross products of the shares. Share sets S + M, S + L, S + M + L; statuses P, G (W) and SAT "
            "(B); samples y5, y1, y10 pooled, 2010–12 cohort at y5 and y10, and V4.14.1 at y5. Robustness rows at y5 "
            "(state FE, institution FE, selectivity controls with their × S and × M terms, WS cells, WLS, rank outcome, "
            f"cells ≥ 100, all CIP-2 families, −ADM, M+); pairs-cluster bootstrap ({B_CLU} draws) for the primary S + M "
            "and S + M + L rows and delete-one-institution jackknife for the primary S + M rows. Wald tests (CR1) of "
            "θ_S = θ_L and θ_S = θ_L = 0. Gelbach (2016) decomposition: θ_S(alone) − θ_S(joint) = Σ_k δ_k γ_k over the "
            "columns the joint model adds (γ_k their joint-model coefficients, δ_k the coefficient on z·e_S when column k "
            "is regressed on the S-only design); exact for OLS. Conditional median split: within each family, cells are "
            "split at the family median of M, the S split is made inside each M half, and the high − low S coupling "
            "differences are averaged over the two halves (and the same with M and S swapped); joint institution "
            "bootstrap as in (b).\n"
            "7c. **Timing of the exposure (third and fourth revisions).** Outcome: log p50 at y5 of B, W and WS cells. "
            "Exposure: every share (S, M, M+, L, L2, the 20 sector shares and C) of the same institution × CIP-2 cell "
            "measured at y1. Three designs: (i) pooled: the pooled-cohort y1 cell with the pooled-cohort y5 outcome "
            "cells; the pooled y5 cell sums every 3-year graduation cohort that reaches y5 (2001–03 to 2016–18) and the "
            "pooled y1 cell the same cohorts plus 2019–21; (ii) own cohort: one cohort's own y1 cell with the same "
            "cohort's y5 cells (2010–12, and since the fourth revision 2013–15 and 2016–18); (iii) mixed: one cohort's "
            "y1 cell (2010–12, 2013–15, 2016–18) with the pooled y5 cells. The y1 cell needs ≥ "
            f"{EMP_MIN} employed graduates and families need ≥ {NMIN} cells after this restriction. Every model is "
            "refitted with the y5 shares on exactly the same cells. Check (fourth revision): pooled y1 sector counts "
            "minus the released counts of the 2019–21 cohort, sector by sector, clipped at 0; cells without a 2019–21 "
            "row keep the pooled counts, cells whose 2019–21 count is suppressed are dropped (strict) or kept with that "
            "cohort taken as 0. Cohort composition: released all-sector cohort totals (agg 46, status 1; unreleased = 0) "
            "against the pooled totals (agg 40) on the B cells. C = log Σ_k share_k × w_k, with w_k the ACS 2023 median "
            "wage of full-time full-year "
            "wage/salary employees aged 22–40 with a bachelor's degree or more in sector k (table in (a)), centred within family, in 0.1-log-point units (the "
            "'10-pp' scaling of the other shares). Wild-cluster restricted bootstrap p (Cameron, Gelbach and Miller "
            f"2008): for each θ the model is refitted with that θ set to 0, {B_WCR} samples y* = X_r b_r + u_r w_g "
            "with Rademacher w_g drawn per institution are refitted unrestricted, and the CR1 t statistic of θ is "
            "compared with the observed one; with institution fixed effects on the within-transformed design, with "
            "WLS on the √w-weighted design.\n"
            "8. **Reliability.** Within-family Spearman of the same cell's S (and M, log p50) between the 2013–15 and "
            "2016–18 graduation cohorts at y1.\n"
            "9. **ACS.** 2023 PUMS person files; bachelor's+ (SCHL ≥ 21), age 22–40, wage/salary employees (COW 1–5), "
            "≥ 35 h/week, ≥ 50 weeks, WAGP > 0, military (NAICSP 92811*) excluded; sector = first two NAICSP "
            "characters; person weights.\n"
            f"10. **Release check.** Flows, earnings and institutions from PSEO V4.14.1 (2026Q2); pooled y5 cells; "
            f"the same families, sample rules and statistics; joint bootstrap ({B_BOOT} draws, own stream tags) "
            "for family means and CR1 CIs for θ.\n"
            "11. **Determinism.** Every random stream is `default_rng([61, crc32(tag)])`; two runs of the script give "
            "byte-identical CSV, PNG and report (md5-checked).\n")


def _ff_ratio(ctx) -> float:
    """max / min |theta_SAT| over the non-rank functional forms (y5 pooled)."""
    lo, hi, _, _ = _theta_range(ctx["FF"]["pooled_y5"], "SAT_b", "SAT")
    return float(max(abs(lo), abs(hi)) / min(abs(lo), abs(hi)))


def _caveats(ctx):
    acs = ctx["acs"].set_index("sector")
    W, RI = ctx["W"], ctx["RI"]
    rel = ctx["rel"]
    DC, sz = _design_counts(ctx), _cell_sizes(ctx)
    def t(key, s_):
        r = RI.get(key)
        return f"{_f(r[s_]['theta'], 3)} {_ci(r[s_]['ci'], 3)}" if r is not None and s_ in r else "—"
    return ("## Caveats\n\n"
            "- **Descriptive.** Status measures and sector placement are both outcomes of who enrols; nothing here "
            "separates value-added from selection. S is itself lower where status is higher, so θ describes "
            "heterogeneity in the gradient across cells, not a causal moderation.\n"
            "- **Composition vs pricing (rewritten in the second, third and fourth revisions).** Earnings are not "
            "released by sector. A flatter gradient in high-S cells fits 'pay set by setting' (status is priced less "
            "inside schools, hospitals and government) and also 'status sorts graduates into high-pay market sectors'. "
            "Models with both status × S and status × M do not separate the two. With shares measured at y5 the SAT "
            f"moderation loads on {_carries_th(ctx['RJ']['SAT_b_SM']['TH'], 'SAT')} (section (c')). With y1 shares it "
            "depends on the design (section (c'')): the y1 shares of largely the same graduates, pooled over cohorts, "
            f"load on {_carries_th(ctx['RX'][f'SAT_SM_{X_MAIN}']['TH'], 'SAT')}"
            + (" (also with the 2019–21 cohort removed)" if DC["no19_same"] else "")
            + (f", as do one cohort's y1 shares set against the pooled y5 pay in {DC['mixS']} of {DC['n_mix']} cohorts"
               if DC["mixS"] else "")
            + f"; each cohort's own y1 shares with its own y5 pay give S | M below 0 in {DC['ownS']} and M | S above 0 "
            f"in {DC['ownM']} of {DC['n_own']} cohorts. With the broad selectivity controls and their × share terms "
            f"{_sel_phrase(ctx)}. No timing gives an exogenous exposure: status already predicts the y1 shares "
            f"(within-family r(z_SAT, S1) {_f(ctx['TDX'][X_MAIN]['z_S1'])}, r(z_SAT, M1) "
            f"{_f(ctx['TDX'][X_MAIN]['z_M1'])}), and S and M move together within families (correlation "
            f"{_f(ctx['COL']['SAT_b']['e_S_M'])} at y5, {_f(ctx['TDX'][X_MAIN]['S1_M1'])} at y1, B cells). The designs "
            "also differ in sample and noise (median employed graduates per B cell at y1 / y5: "
            + ", ".join(f"{'pooled' if k == X_MAIN else _coh(k) + ' own'} {sz[k][0]:.0f} / {sz[k][1]:.0f}"
                        for k in [X_MAIN] + X_OWN)
            + "), so their disagreement cannot be put down to timing alone. This "
            "analysis should not be cited for 'it largely vanishes where pay is set by setting': the magnitude is not "
            "'largely', and the attribution to setting-priced pay is not identified. It does not rule that mechanism "
            "out either.\n"
            "- **What S also measures.** Within a CIP-2 family, the setting share also reflects the CIP-4 program "
            "mix (e.g. nursing vs health administration within CIP-51; teacher preparation inside subject families) "
            "and the state's public sector. Institution and state fixed effects do not remove program mix.\n"
            f"- **Sector is the employer's industry, not the job.** A hospital accountant counts as health care. "
            f"Health care (62) is mostly a non-government employer (ACS government share {acs.loc['62','gov']:.2f}, "
            f"government or non-profit {acs.loc['62','gov_np']:.2f}); its classification as setting-priced rests on "
            f"pay scales and reimbursement, which this script does not test. Components are reported separately.\n"
            f"- **Coverage.** Only {W.institution.nunique()} PSEO institutions match Wapman (PSEO is public-skewed and "
            "misses most elite privates), so the prestige and brand results come from a top-truncated set. The "
            "project's canonical name join also misses at least 19 Wapman institutions that are in PSEO (e.g. Stony "
            "Brook, Buffalo, Iowa State, Missouri; listed in PSEO_REFRESH_RESULT.md), which this script inherits. The "
            f"selectivity results use {ctx['Bs'].institution.nunique()} institutions, most of them not in Wapman. "
            "PSEO covers UI-covered and "
            "federal (OPM) jobs. Independent contractors, the unincorporated self-employed, salespeople paid mainly "
            "on commission, workers of some non-profits, the armed forces and the Postal Service are not observed "
            "(PSEO technical documentation), and graduates without attached employment (including those in "
            "graduate school) are outside the shares.\n"
            f"- **Noise.** Flows counts carry differentially private noise (geometric mechanism, ε = 1.5, drawn at "
            f"state × sector and aggregated, with post-processing for negative counts). Cells below {EMP_MIN} employed "
            f"are dropped; with ≥ 100 the prestige θ is {t('P_emp100', 'P')} and the SAT θ {t('SAT_b_emp100', 'SAT')}. "
            f"Test–retest reliability of S across cohorts is "
            f"{rel['S'][0]:+.2f} (a lower bound on reliability, since programs also change between cohorts). "
            "Classical measurement error in S attenuates the product term, but it also leaves part of the curvature "
            "of earnings in S uncontrolled, and status is correlated with S; so noise in S does not only pull θ "
            "toward 0 (the same mechanism as under 'Functional form' below).\n"
            "- **Pooled cohorts.** Institutions contribute different cohort mixes to the pooled cell (partners joined "
            f"PSEO at different times); on the fixed 2010–12 cohort at y5, θ is {t('P_c2010_y5', 'P')} (prestige) and "
            f"{t('SAT_b_c2010_y5', 'SAT')} (SAT).\n"
            f"- **Family prestige is a composite at CIP-2.** Median ranked-field coverage of W cells is "
            f"{W.coverage.median():.2f}; θ is {t('P_cov50', 'P')} on cells with coverage ≥ 0.5 and {t('Pu', 'P')} "
            f"with the unweighted P_u.\n"
            "- **Functional form.** θ is a product term in a model whose main effects must be specified. Log p50 is "
            "curved in the setting share and status is correlated with the share, so θ depends on how flexibly the "
            "exposure main effect is modelled. The first version's linear-e θ for prestige and brand "
            + ("does not survive" if not (ctx['RI']['P']['P']['ci'][1] < 0 or ctx['RI']['G']['G']['ci'][1] < 0) else "partly survives")
            + f" family-specific curvature; the SAT θ has a CI below 0 under {_nsig(ctx['FF']['pooled_y5'], 'SAT_b', 'SAT')[0]} "
            f"of {_nsig(ctx['FF']['pooled_y5'], 'SAT_b', 'SAT')[1]} forms but its size varies about {_ff_ratio(ctx):.1f}-fold "
            f"across the {N_FF_NR} non-rank forms (S only). With M and SAT × M in the model, θ(SAT × S | M) has a CI "
            f"below 0 in {_jcount(ctx['JB']['pooled_y5'], 'SAT_b', 'SM', 'SAT', 'S', '-')[0]} of "
            f"{_jcount(ctx['JB']['pooled_y5'], 'SAT_b', 'SM', 'SAT', 'S', '-')[1]} forms at y5 and θ(SAT × M | S) a CI "
            f"above 0 in {_jcount(ctx['JB']['pooled_y5'], 'SAT_b', 'SM', 'SAT', 'M', '+')[0]}. None of the forms is known "
            "to be right; the median splits, which need no form for S or M, are the checks that assume least.\n"
            "- **Many specifications.** The primary rows are P × S (W) and SAT × S (B) at y5, pooled cohorts, ≥ 50 "
            "employed, family-specific e² (changed from linear e after verification); everything else is robustness "
            "and is reported in full, including the rows that go the other way (prestige net of selectivity). The "
            f"second revision adds {N_JFF} forms × 3 share sets × 3 statuses × {len(FF_SAMPLES)} samples of joint fits "
            "plus about 30 joint robustness rows; the low-pay share L was defined post hoc (by the verifier, after the "
            f"S + M result was known), so tests involving L are exploratory. The third and fourth revisions add "
            f"{len(X_SAMPLES)} exposure-timing samples ({len(X_SAMPLES) - 2} in the third, the 2013–15 and 2016–18 "
            f"own-cohort designs in the fourth) and {len(X_CHECKS)} checks, each with y1 and y5 shares ({N_FF} S-only and "
            f"{N_JFF} × 3 × 3 joint fits per sample and timing), and about {_n_timing_rows(ctx)} timing rows with "
            "wild-cluster p-values; the timing check was suggested by the verifier after the y5 result was known, and "
            "the own-cohort designs after the pooled-y1 result was known. With this many fits, single "
            "rows whose CI excludes 0 are expected by chance; the report counts how many rows agree rather than "
            "picking one.\n")


def _n_timing_rows(ctx) -> int:
    """Number of distinct timing-row fits (y1 and y5-share rows of section (c''))."""
    return len(ctx["RX"]) + len(ctx["RXA"])


def _sel_phrase(ctx) -> str:
    """How the S | M and M | S loadings fare with the broad selectivity x share controls, both timings."""
    RJ, RX = ctx["RJ"], ctx["RX"]
    xs_all = [k for k, _, _, _ in X_SAMPLES]
    rows = [RJ["SAT_b_SM_sel"]["TH"]] + [RX[f"SAT_SM{t_}_sel_{k}"]["TH"] for k in xs_all for t_ in ("5", "")]
    nS = sum(t[("SAT", "S")]["ci"][1] < 0 for t in rows)
    nM = sum(t[("SAT", "M")]["ci"][0] > 0 for t in rows)
    y1 = RX[f"SAT_SM_sel_{X_MAIN}"]["TH"]
    return (f"SAT × S | M has a CI below 0 in {nS} and SAT × M | S a CI above 0 in {nM} of {len(rows)} fits (y5 and "
            f"y1 shares, all samples; pooled y1 shares: S | M {_vth(y1[('SAT', 'S')])}, M | S {_vth(y1[('SAT', 'M')])})")


def _provenance(ctx):
    files = [PSEOF, PSEOE, PSEO_INST, PSEO_VER, PSEOF_NEW, PSEOF_NEW_VER, PSEOE_NEW, PSEO_INST_NEW, PSEO_VER_NEW,
             SC_INST, WAPMAN] + ACS
    L = ["## Provenance (for SOURCES.md)\n",
         "New raw data downloaded for this analysis: `data/raw/pseo_flows_2026q2/pseof_all.csv.gz` and its "
         "`version_pseo.txt` (PSEO Flows, release V4.14.1 2026Q2), fetched with curl from "
         "https://lehd.ces.census.gov/data/pseo/latest_release/all/ on 2026-09-23 (UTC; version file at 23:17, "
         "flows file complete at 23:29). HEAD request at download, repeated on 2026-09-24 with the same result: "
         "Content-Length 195,565,573 bytes, Last-Modified Tue, 25 Aug 2026 21:07:46 GMT; the local size "
         f"is {PSEOF_NEW.stat().st_size:,} bytes"
         + (" (equal)" if PSEOF_NEW.stat().st_size == 195_565_573 else " (DIFFERENT: incomplete download?)")
         + "; `gzip -t` passed (at download and again on 2026-09-24) and the header matches the V4.13.0 flows "
         "file. The `latest_release` URLs now serve "
         "V4.14.1; the V4.13.0 rows below were fetched from the same URLs when V4.13.0 was the latest release. "
         "All other inputs were already under `data/raw/` (the V4.14.1 earnings "
         "and institutions files were downloaded for scripts/59, see PSEO_REFRESH_RESULT.md). Documentation "
         "fetched on 2026-09-23 (not stored in the repo): LEHD public-use schema "
         "https://lehd.ces.census.gov/data/schema/latest/lehd_public_use_schema.html (PSEOF variables §4.3.7, "
         "aggregation levels `label_agg_level_pseo.csv`: agg 40/46/88/94 are the institution × CIP-2 levels; flows "
         "have no CIP-4 level) and the PSEO technical documentation "
         "https://lehd.ces.census.gov/doc/PSEOTechnicalDocumentation.pdf (main-job industry assignment, "
         "non/marginal employment, geometric-mechanism noise with ε = 1.5).\n",
         "| file | source URL | release / vintage | bytes | md5 | accessed |", "|---|---|---|---|---|---|"]
    ver = PSEO_VER.read_text().strip().splitlines()
    rel = "; ".join(v.split(" pseopu")[0] for v in ver)
    vnew = [v for v in PSEOF_NEW_VER.read_text().strip().splitlines() if " US 00 " in v]
    rel_new = "; ".join(v.split(" pseopu")[0] for v in vnew)
    LR = "https://lehd.ces.census.gov/data/pseo/latest_release/all/"
    meta = {PSEOF: ("https://lehd.ces.census.gov/data/pseo/latest_release/all/pseof_all.csv.gz", rel, "2026-06-21"),
            PSEOE: ("https://lehd.ces.census.gov/data/pseo/latest_release/all/pseoe_all.csv.gz", rel, "2026-06-20"),
            PSEO_INST: ("https://lehd.ces.census.gov/data/pseo/latest_release/all/pseo_all_institutions.csv", rel, "2026-06-20"),
            PSEO_VER: ("https://lehd.ces.census.gov/data/pseo/latest_release/all/version_pseo.txt", rel, "2026-06-20"),
            PSEOF_NEW: (LR + "pseof_all.csv.gz", rel_new, "2026-09-23 (this script)"),
            PSEOF_NEW_VER: (LR + "version_pseo.txt", rel_new + " (+ state lines)", "2026-09-23 (this script)"),
            PSEOE_NEW: (LR + "pseoe_all.csv.gz", rel_new, "2026-09-23 (scripts/59)"),
            PSEO_INST_NEW: (LR + "pseo_all_institutions.csv", rel_new, "2026-09-23 (scripts/59)"),
            PSEO_VER_NEW: (LR + "version_pseo.txt", rel_new + " (+ state lines)", "2026-09-23 (scripts/59)"),
            SC_INST: ("College Scorecard Most-Recent-Cohorts-Institution (see SOURCES.md)", "Most Recent, build 2026-05-27", "2026-09-23"),
            WAPMAN: ("Wapman et al. 2022 ranks.csv (see SOURCES.md)", "2011-2020", "see SOURCES.md"),
            ACS[0]: ("https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_pus.zip", "ACS 2023 1-year PUMS", "2026-06-20"),
            ACS[1]: ("https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_pus.zip", "ACS 2023 1-year PUMS", "2026-06-20")}
    for f in files:
        u, r, a = meta[f]
        L.append(f"| `{f.relative_to(ROOT)}` | {u} | {r} | {f.stat().st_size:,} | `{md5(f)}` | {a} |")
    L.append("\nAccess dates for existing files are those recorded in `data/raw/SOURCES.md` (PSEO §7, ACS, Scorecard "
             "institution file).\n")
    return "\n".join(L)


def print_summary(ctx):
    SUM, RI = ctx["SUM"], ctx["RI"]
    for k, v in sorted(SUM.items()):
        print(f"{k[0]:5s} {k[1]:22s} k={v['k']:2d} mean={v['mean']:+.3f} ci=[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] "
              f"pos={v['n_pos']} neg={v['n_neg']}")
    for k, v in ctx["PAIR"].items():
        print(f"PAIR {k}: {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] k={v['k']}")
    for k, v in ctx["ORDER"].items():
        print("ORDER", k, {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()})
    for k, r in RI.items():
        if r is None:
            print("RI", k, None); continue
        for s in ["P", "G", "SAT", "NEG_ADM"]:
            if s in r:
                v = r[s]
                print(f"RI {k:16s} {s:7s} n={r['n_cells']} inst={r['n_inst']} k={r['k']} beta={v['beta']:+.4f}({v['beta_se']:.4f}) "
                      f"theta={v['theta']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] p={v['p']:.3f}"
                      + (f" boot=[{v['boot_ci'][0]:+.4f},{v['boot_ci'][1]:+.4f}]" if "boot_ci" in v else ""))
    for k, r in ctx["RIL"].items():
        if r is None:
            continue
        for s in ["P", "G", "SAT", "NEG_ADM"]:
            if s in r:
                v = r[s]
                print(f"RIL {k:16s} {s:7s} theta={v['theta']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] p={v['p']:.3f}"
                      + (f" boot=[{v['boot_ci'][0]:+.4f},{v['boot_ci'][1]:+.4f}]" if "boot_ci" in v else ""))
    for smp, ff in ctx["FF"].items():
        for (key, k), r in ff.items():
            s = {"P": "P", "G": "G", "SAT_b": "SAT"}[key]
            v = r[s]
            print(f"FF {smp:10s} {key:6s} {k:7s} n={r['n_cells']} inst={r['n_inst']} theta={v['theta']:+.4f} "
                  f"[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] p={v['p']:.3f}")
    for k in ("P", "G", "SAT_b"):
        v = ctx["RI"][k][{"P": "P", "G": "G", "SAT_b": "SAT"}[k]]
        print(f"JACK {k} theta={v['theta']:+.4f} jack_ci=[{v['jack_ci'][0]:+.4f},{v['jack_ci'][1]:+.4f}]")
    print("DIAG", ctx["DIAG"])
    print("MIX", ctx["MIX"])
    print("IMPL", ctx["IMPL"])
    print("WI", ctx["WI"])
    print("corr_zz", ctx["corr_zz"])
    print("rel", ctx["rel"], "eta2", ctx["eta2_S"], ctx["eta2_M"])
    print("legacy", ctx["legacy"])
    print("HROB", ctx["HROB"])
    print("LOFO", ctx["LOFO"])
    print("FT_P_w", ctx["FT_P_w"], "FT_S_w", ctx["FT_S_w"])
    REL = ctx["REL"]
    for k, v in sorted(REL["SUM"].items()):
        print(f"REL {k[0]:3s} {k[1]:12s} k={v['k']:2d} mean={v['mean']:+.3f} ci=[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")
    for k, r in REL["RI"].items():
        for s_ in ["P", "G", "SAT"]:
            if s_ in r:
                v = r[s_]
                print(f"REL RI {k:10s} {s_:4s} n={r['n_cells']} inst={r['n_inst']} theta={v['theta']:+.4f} "
                      f"[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]")
    print("REL new W institutions", REL["new_W_labels"], "lost", REL["lost_W_inst"])
    # second revision: joint exposure models
    for k, r in ctx["RJ"].items():
        print(f"RJ {k:18s} n={r['n_cells']} inst={r['n_inst']} " + "  ".join(
            f"{s_}x{x}={v['theta']:+.4f}[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]"
            + (f"boot[{v['boot_ci'][0]:+.4f},{v['boot_ci'][1]:+.4f}]" if "boot_ci" in v else "")
            + (f"jack[{v['jack_ci'][0]:+.4f},{v['jack_ci'][1]:+.4f}]" if "jack_ci" in v else "")
            for (s_, x), v in r["TH"].items()))
        for s_, t in r.get("tests", {}).items():
            for nm, w in t.items():
                print(f"   TEST {k} {s_} {nm}: est={w['est']} W={w['W']:.3f} df={w['df']} p={w['p']:.4f}")
    for smp, jb in ctx["JB"].items():
        for (key, sk, form), r in jb.items():
            print(f"JB {smp:10s} {key:5s} {sk:4s} {form:7s} n={r['n_cells']} " + "  ".join(
                f"{x}={v['theta']:+.4f}[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]" for (s_, x), v in r["TH"].items()))
    print("COL", ctx["COL"])
    print("GB", ctx["GB"])
    print("IMPLJ", ctx["IMPLJ"])
    for key, r in REL["RJ"].items():
        print(f"REL RJ {key:10s} n={r['n_cells']} inst={r['n_inst']} " + "  ".join(
            f"{s_}x{x}={v['theta']:+.4f}[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]" for (s_, x), v in r["TH"].items()))
    # third revision: timing of the exposure
    for k, r in list(ctx["RX"].items()) + [("S_alone_" + k, r) for k, r in ctx["RXA"].items()]:
        print(f"RX {k:28s} n={r['n_cells']} inst={r['n_inst']} " + "  ".join(
            f"{s_}x{x}={v['theta']:+.4f}[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]p{v.get('wcr_p', np.nan):.3f}"
            + (f"boot[{v['boot_ci'][0]:+.4f},{v['boot_ci'][1]:+.4f}]" if "boot_ci" in v else "")
            for (s_, x), v in r["TH"].items()))
        for s_, t in r.get("tests", {}).items():
            for nm, w in t.items():
                print(f"   TEST {k} {s_} {nm}: est={w['est']} ci={w['ci']} p={w['p']:.4f}")
    for lab, jbd in [("JBX", ctx["JBX"]), ("JB5X", ctx["JB5X"])]:
        for smp, jb in jbd.items():
            for (key, sk, form), r in jb.items():
                print(f"{lab} {smp:12s} {key:5s} {sk:4s} {form:7s} n={r['n_cells']} " + "  ".join(
                    f"{x}={v['theta']:+.4f}[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]" for (s_, x), v in r["TH"].items()))
    print("XN", ctx["XN"])
    print("IMPLX", ctx["IMPLX"])
    print("GBX", ctx["GBX"])
    print("TDX", ctx["TDX"])
    for k, v in sorted(ctx["SUMX"].items()):
        print(f"SUMX {k} k={v['k']} mean={v['mean']:+.3f} ci=[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")
    # fourth revision: cohort composition of the pooled cells; pooled y1 minus the 2019-21 cohort
    print("COMP", ctx["COMP"])
    print("NO19", ctx["NO19"])


if __name__ == "__main__":
    main()
