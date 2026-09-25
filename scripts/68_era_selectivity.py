"""Era-matched selectivity: do the selectivity controls of scripts/55 and scripts/58 change the answer when they
describe the graduates' own entering classes instead of the 2026 file's (test-optional era) entering classes?

REFEREE PROBLEM. scripts/55 and scripts/58 take SAT_AVG, ADM_RATE, PCTPELL (and CONTROL, STABBR) from the 2026 College
Scorecard "Most Recent" institution file, where SAT_AVG and ADM_RATE describe the fall-2024 entering class and PCTPELL
academic year 2023-24 (data dictionary, Most_Recent_Inst_Cohort_Map; asserted at run time). The earnings are for
graduates who entered about a decade earlier. This script rebuilds the controls from the Scorecard historical
institution files (MERGED<yyyy>_<yy>_PP.csv inside the "All Data Files" zip, read with usecols UNITID, SAT_AVG,
ADM_RATE, PCTPELL, CONTROL, STABBR) for the entry years of the graduates behind each earnings measure, and re-runs the
scripts/55 level ladder, the within-institution department slope of scripts/55 / 58 and the career-time slope of
scripts/52 / 59 on the same programs with both sets of controls. Descriptive, not causal.

PRE-SPECIFICATION (written before any estimate of this script was computed; md5 of this block printed in the result)
ENTRY-YEAR DEFINITIONS
  Scorecard Field-of-Study EARN_MDN_4YR (Most Recent file, June 2026 release): "Treasury AY2017-2018, AY2018-2019
  pooled earnings cohort measured in CY2022, CY2023" (data dictionary, FieldOfStudy_Cohort_Map, FieldOfStudyMostRecent
  datafile; asserted at run time). A bachelor's completer of award year 2017-18 (2018-19) who took four years entered
  in fall 2014 (fall 2015). Era-matched controls (version E):
    SAT_AVG, ADM_RATE = mean of the fall-2014 and fall-2015 values (MERGED2014_15, MERGED2015_16), available values;
    PCTPELL = mean of academic years 2014-15 and 2015-16 (MERGED2015_16, MERGED2016_17);
    CONTROL, STABBR = MERGED2015_16 (MERGED2014_15 where missing).
  Wide window (version W; sensitivity for 5-6-year completers): SAT_AVG, ADM_RATE over falls 2012-2015, PCTPELL over
    academic years 2012-13 to 2015-16; CONTROL, STABBR as E.
  2026 controls (version 26) = the scripts/55 columns (fall-2024 SAT_AVG / ADM_RATE, AY 2023-24 PCTPELL).
  The state earnings level ST_EARN of scripts/55 (leave-one-out mean log MD_EARN_WNE_P10 of PREDDEG=3 institutions in
  the state, 2026 file) is a geography control, not a selectivity measure; every version uses it.
  PSEO bachelor's fixed cohorts: graduation cohort c = graduates of calendar years c..c+2 (PSEO documentation). Entry
  = falls c-4..c-2. SAT_AVG and ADM_RATE exist in the Scorecard files from fall 2001 and PCTPELL from AY 2007-08, so:
    cohorts 2004 (falls 2000-2002, of which 2001 and 2002 exist), 2007 (2003-2005), 2010 (2006-2008): mean over the
      available falls of the entry window; CONTROL from the file of fall c-2 (earlier falls of the window if missing);
    cohort 2001 (falls 1997-1999, none exist): only in a sensitivity, with fall-2001 values ("nearest available",
      2-4 years after entry).
  PCTPELL is not used for PSEO (not in the Scorecard files for the 2004 and 2007 entry windows).
PRE-SPECIFIED TESTS
  (T1) Scorecard 4-year earnings on "common programs": programs whose institution has the scripts/55 SAT-family
    (SAT_AVG, ADM_RATE, PCTPELL, CONTROL, STABBR, ST_EARN) complete under BOTH the 26 and the E controls.
    (T1a) scripts/55 level ladder on a fixed field set (n >= 15 and residual df >= 10 in pcs, pcs_adm, broad and sel
      under 26, E and W): mean over fields of the raw rho_f and of the partial rho_f given
        pcs     : PCTPELL + CONTROL + ST_EARN
        pcs_adm : ADM_RATE + PCTPELL + CONTROL + ST_EARN
        broad   : SAT_AVG + ADM_RATE + PCTPELL + CONTROL + ST_EARN   [PRIMARY: the scripts/55 net coupling]
        sel     : SAT_AVG + ADM_RATE
      and strict (broad with state FE in place of ST_EARN; own field set, df >= 10 under 26 and E), each under the
      26, E (and W) controls. Reported: means, E - 26 and W - 26 differences.
    (T1b) within-institution cross-department design on common programs (scripts/58 fe_build: institution FE absorbed,
      z-scored prestige and field-specific slopes; programs also need brand G): pooled beta (log points per within-field
      SD of department prestige) of
        field FE only (no institution FE) -> (a) institution FE -> (b) + field x {SAT, ADM, Pell}
        -> main (scripts/58) + field x {SAT, ADM, Pell, brand G}   [PRIMARY: main]
      under 26, E and W controls; E - 26 and W - 26 differences.
  (T2) stability: Spearman across institutions of SAT_AVG(E) with SAT_AVG(26) [PRIMARY], and of ADM_RATE and PCTPELL,
    on the institutions of the Scorecard analysis (every institution of the scripts/55 cells with both values); the
    same for W; and for each PSEO entry window with 26 on the T3 institutions. 95% CI: institution bootstrap
    (percentile, B = 1000).
  (T3) PSEO V4.14.1 balanced fixed-cohort panel (scripts/52 engine via scripts/59, canonical join), cohorts 2004, 2007,
    2010, institutions with SAT_AVG and ADM_RATE under both versions (cells re-formed, n >= 15): per cell the OLS slope
    over y1/y5/y10 of the partial Spearman of field prestige F and median earnings given SAT_AVG + ADM_RATE (ranks,
    scripts/55 convention) under the E controls [PRIMARY] and under the 26 controls, and the raw coupling slope on the
    same cells; field value = mean over its cohorts; statistic = mean over fields. Reported: E, 26, raw, E - 26, E - raw.
  Secondary (pre-specified sensitivities): W controls in T1; cohort 2001 with fall-2001 controls added to T3; T3 with
    SAT_AVG + ADM_RATE + CONTROL + ST_EARN; T1a on the era-complete sample (programs with complete E controls,
    including institutions without a fall-2024 SAT_AVG) against scripts/55's own 26 sample.
  Inference: two-way (field, institution) cluster variance of scripts/59 (twoway(): V_field + V_inst - V_field x inst)
    from (i) the between-field variance / K (a delete-one-field jackknife for the pooled beta), (ii) B = 1000
    institution multinomial draws shared by every field (fields fixed), (iii) B = 1000 institution draws independent
    per field. Normal 95% CI est +/- 1.96 SE; two-sided p from z. Differences use the same draws (paired). For a null,
    the minimum detectable difference at 80% power (two-sided 5%) = 2.80 SE.
  Reading rule for each primary difference D = E - 26: "differs" if the 95% CI excludes 0; "equivalent within
    +/- delta" if the 90% CI lies inside +/- delta, delta = one third of the 2026-control estimate in scripts/55 / 58 / 59
    (0.05 for the mean partial rho_f, 0.003 log points per SD for beta, 0.010 per year for the slope); otherwise
    "inconclusive". Everything else is exploratory and labelled so.
END PRE-SPECIFICATION

REVISION NOTE (2026-09-25, after an independent verification; the pre-specification block above is unchanged and its
md5 is printed). (1) The reading rule's text ("one third of the 2026-control estimate") and its T3 number (0.010/yr,
one third of scripts/59's raw slope; scripts/59 has no selectivity-controlled slope) disagree. Both readings are now
reported for every primary difference: at the numeric margins as written and at delta = one third of the 2026-control
estimate on the same programs / cells computed here; the T3 90% CI is also stated relative to the 2026-control partial
slope. (2) For many statistics the Cameron-Gelbach-Miller two-way variance is below one of its one-way parts
(V_field or V_institution); a conservative variance max(V_two-way, V_field, V_institution) is reported next to the
pre-specified one for every row, with the readings under it. (3) A reading whose 90% (or 95%) CI endpoint lies
within 2 Monte Carlo SEs of its threshold is flagged as on the margin. (4) STABBR is backfilled with the most recent
value in every MERGED file (data dictionary, asserted), so it is not re-dated; CONTROL is. (5) Every number of the
write-up, including counts and the ratios computed in the text, is written to the CSV (sections coverage, derived);
each row carries a status (primary / pre-specified / pre-specified sensitivity / exploratory / check / ...).
(6) Statistics not in the pre-specification (T1b (a) - main, T3 26 - raw, T3 per-horizon couplings, T1a raw - v)
are labelled exploratory. (7) The start of SAT_AVG / ADM_RATE is cited from the data dictionary and checked in
MERGED2000_01 (no values); the institution documentation p. 12 supports the SAT start only.

Reuse (no existing file edited): scripts/55 (load_institutions, build_cells, spear, partial_rank, NMIN, DFMIN),
scripts/58 (fe_build, fe_fit, FOS_FILE), scripts/59 (twoway, wrank, wcorr, set_release, institutions, NEW_EARN) and
through it scripts/52 (load_fixed, fixed_panel, cells_of, W_SLOPE, corr_rows). For T1b the un-demeaned fe_build design
is rebuilt here and checked against fe_build (demeaning it reproduces fe_build's X and y), so that every replicate
re-absorbs the institution FE under its own row weights (weighted within-institution demeaning = WLS with institution
dummies): an institution drawn twice counts twice, and a field deleted by the jackknife no longer enters the other
fields' institution means.
Reproducible: one seed per bootstrap replicate from numpy SeedSequence([68, tag(, field)]).spawn(B); replicates are
computed in fixed chunks of 50 in worker processes with single-threaded BLAS, so the outputs are byte-identical on
re-run and for any number of workers (env ERA68_WORKERS, default 2: peak total PSS about 680 MB with 2 workers;
2 at most if MemAvailable < 2 GB, 1 if < 1.5 GB).
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/68_era_selectivity.py
Outputs: data/interim/era_selectivity.csv (every number), ERA_SELECTIVITY_RESULT.md (root; local only).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import sys
import gc
import time
import zlib
import hashlib
import zipfile
import importlib.util
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s58 = _load("s58", "58_selectivity_deep.py")      # within-institution design (fe_build); loads scripts/55 as s58.s55
s55 = s58.s55
s59 = _load("s59", "59_pseo_refresh.py")          # two-way variance, weighted ranks, release plumbing; loads scripts/52
s52 = s59.s52
FIELDS66, LAB = s55.FIELDS66, s55.LAB
NMIN, DFMIN = s55.NMIN, s55.DFMIN
HZ, W_SLOPE = s52.HZ, s52.W_SLOPE

SEED = 68
B = 1000                      # replicates per institution draw (shared / independent) and for T2
CHUNK = 50                    # fixed replicate chunk (floating-point results do not depend on the worker count)
Z975 = 1.959963984540054
Z95 = 1.6448536269514722
Z80 = 0.8416212335729143
DELTA = {"t1a": 0.05, "t1b": 0.003, "t3": 0.010}      # equivalence margins (pre-specified)
OUT_CSV = ROOT / "data" / "interim" / "era_selectivity.csv"
OUT_MD = ROOT / "ERA_SELECTIVITY_RESULT.md"
S55_CSV = ROOT / "data" / "interim" / "selectivity_fields.csv"
S58_CSV = ROOT / "data" / "interim" / "selectivity_deep.csv"
S59_CSV = ROOT / "data" / "interim" / "pseo_refresh.csv"
INST_FILE = s55.INST_FILE

# ---- provenance of the files this script downloaded (checked at run time) --------------------------------------
HIST_DIR = ROOT / "data" / "raw" / "scorecard_hist"
ZIP = HIST_DIR / "College_Scorecard_Raw_Data_06102026.zip"
ZIP_URL = "https://ed-public-download.scorecard.network/downloads/College_Scorecard_Raw_Data_06102026.zip"
ZIP_MD5, ZIP_SIZE = "38f26ca8776518f77836ddfa1e219de8", 469_515_074
ZIP_LASTMOD = "Tue, 09 Jun 2026 16:50:50 GMT"     # HTTP Last-Modified; the S3 ETag equals the md5 (single-part)
ZIP_DIR = "College_Scorecard_Raw_Data_06032026"   # folder name inside the zip
DOC_URL = "https://collegescorecard.ed.gov/files/"
DOCS = {"FieldOfStudyDataDocumentation.pdf": ("6a273d6bbca8659593a5a1fc85f0a371", 891_575),
        "InstitutionDataDocumentation.pdf": ("2a51d6b969e8b1be765e18024e363e04", 531_031),
        "CollegeScorecardDataDictionary.xlsx": ("06edabe318ce7686abeb967d7a290e05", 727_278)}
DOC_LASTMOD = "Fri, 11 Sep 2026 02:33:17 GMT"
ACCESSED = "2026-09-24, 20:31-20:33 UTC"
DATA_PAGE = "https://collegescorecard.ed.gov/data/"
DOC_PAGE = "https://collegescorecard.ed.gov/data/data-documentation/"
PSEO_DOC = "https://lehd.ces.census.gov/data/pseo_documentation.html"
HIST_COLS = ["UNITID", "SAT_AVG", "ADM_RATE", "PCTPELL", "CONTROL", "STABBR"]

# ---- era definitions (pre-specified) ----------------------------------------------------------------------------
ERA_FOS = dict(falls=[2014, 2015], pell=[2014, 2015], ctl=[2015, 2014])            # version E
ERA_FOS_W = dict(falls=[2012, 2013, 2014, 2015], pell=[2012, 2013, 2014, 2015], ctl=[2015, 2014])   # version W
T3_COH = ["2004", "2007", "2010"]                  # primary PSEO cohorts
T3_COH4 = ["2001", "2004", "2007", "2010"]         # sensitivity (2001: fall-2001 stand-in)
SAT_FIRST_FALL = 2001


def era_pseo(c: str) -> dict:
    c = int(c)
    if c == 2001:
        return dict(falls=[2001], pell=[], ctl=[2001], window=[1997, 1998, 1999], standin=True)
    w = [c - 4, c - 3, c - 2]
    return dict(falls=[y for y in w if y >= SAT_FIRST_FALL], pell=[], ctl=[c - 2, c - 3, c - 4], window=w,
                standin=False)


# T1a partial-correlation specs: name -> (continuous covariates, categorical covariates); version-dependent columns
# are SAT_AVG, ADM_RATE, PCTPELL, CONTROL, STABBR; ST_EARN is common.
SPEC1 = {"raw": ([], []),
         "pcs": (["PCTPELL", "ST_EARN"], ["CONTROL"]),
         "pcs_adm": (["ADM_RATE", "PCTPELL", "ST_EARN"], ["CONTROL"]),
         "broad": (["SAT_AVG", "ADM_RATE", "PCTPELL", "ST_EARN"], ["CONTROL"]),
         "sel": (["SAT_AVG", "ADM_RATE"], []),
         "strict": (["SAT_AVG", "ADM_RATE", "PCTPELL"], ["CONTROL", "STABBR"])}
SPEC1_DESC = {"raw": "raw rho_f (no controls)", "pcs": "PCTPELL + CONTROL + state earnings level",
              "pcs_adm": "ADM_RATE + PCTPELL + CONTROL + state earnings level",
              "broad": "SAT_AVG + ADM_RATE + PCTPELL + CONTROL + state earnings level (scripts/55 broad)",
              "sel": "SAT_AVG + ADM_RATE", "strict": "SAT_AVG + ADM_RATE + PCTPELL + CONTROL + state FE"}
SPEC1_SHORT = {"raw": "raw", "pcs": "Pell + control + state level", "pcs_adm": "ADM + Pell + control + state level",
               "broad": "broad (SAT + ADM + Pell + control + state level)", "sel": "SAT + ADM only",
               "strict": "strict (SAT + ADM + Pell + control + state FE)"}
LADDER = ["raw", "pcs", "pcs_adm", "broad", "sel"]
VERS = ["26", "E", "W"]
VLAB = {"26": "2026 file (fall 2024)", "E": "era-matched (falls 2014-15)", "W": "wide window (falls 2012-15)",
        "-": "none"}
VCOLS = ["SAT_AVG", "ADM_RATE", "PCTPELL", "CONTROL", "STABBR"]
FAM_NEED = ["SAT_AVG", "ADM_RATE", "PCTPELL", "ST_EARN", "CONTROL", "STABBR"]      # scripts/55 SAT family
# T1b specs: name -> field-specific slope covariates (None = no institution FE)
FE_SPECS = {"nofe": None, "a": [], "b": ["SAT_AVG", "ADM_RATE", "PCTPELL"], "main": ["SAT_AVG", "ADM_RATE", "PCTPELL", "G"]}
FE_DESC = {"nofe": "field FE only, no institution FE (scripts/55 comparator)", "a": "(a) institution FE + field FE",
           "b": "(b) (a) + field x {SAT, ADM, inst. Pell}",
           "main": "main (scripts/58): (a) + field x {SAT, ADM, inst. Pell, brand G}"}
# T3 specs
SPEC3 = {"raw": ([], []), "sel": (["SAT", "ADM"], []), "sec": (["SAT", "ADM", "ST_EARN"], ["CTL"])}

ROWS: list[dict] = []
# build-history times of this lane (file modification times of the interrupted build's saved copies; UTC, the
# machine's local time was CEST = UTC+2)
BUILD_BACKUP_UTC = "2026-09-24 21:53 UTC (23:53 CEST)"
BUILD_DEV_UTC = "21:57 UTC (23:57 CEST)"


def rec(section, stat, estimate, spec="", version="", sample="", field="", lo=np.nan, hi=np.nan, se=np.nan,
        p=np.nan, k=np.nan, n=np.nan, note="", status="", se_c=np.nan, lo_c=np.nan, hi_c=np.nan, reading=""):
    ROWS.append(dict(section=section, stat=stat, spec=spec, version=version, sample=sample, field=field,
                     estimate=estimate, ci_lo=lo, ci_hi=hi, se=se, p_value=p, k=k, n=n, status=status,
                     se_maxvar=se_c, ci_lo_maxvar=lo_c, ci_hi_maxvar=hi_c, reading=reading, note=note))


def drv(stat, value, note="", sample="", n=np.nan, k=np.nan, status="derived (from the rows above)"):
    """A number computed in the write-up from other results: recorded in the CSV section 'derived'."""
    rec("derived", stat, value, sample=sample, n=n, k=k, note=note, status=status)
    return value


def stage(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def prespec_md5() -> str:
    doc = __doc__
    a, b = doc.index("PRE-SPECIFICATION ("), doc.index("END PRE-SPECIFICATION")
    return hashlib.md5(doc[a:b].encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# provenance, documentation checks, historical files
# ---------------------------------------------------------------------------------------------------------------
def fname_fall(y: int) -> str:
    return f"MERGED{y}_{(y + 1) % 100:02d}_PP.csv"


def provenance() -> dict:
    out = {}
    assert ZIP.stat().st_size == ZIP_SIZE, "zip size differs from the server Content-Length"
    out["zip_md5"] = md5(ZIP)
    assert out["zip_md5"] == ZIP_MD5, "zip md5 differs from the recorded (= S3 ETag) value"
    for f, (m, sz) in DOCS.items():
        p = HIST_DIR / f
        assert p.stat().st_size == sz and md5(p) == m, f
    zf = zipfile.ZipFile(ZIP)
    info = {i.filename: i for i in zf.infolist()}
    # the zip's Most-Recent institution file is byte-identical to the one scripts/55 reads
    h = hashlib.md5()
    with zf.open(f"{ZIP_DIR}/Most-Recent-Cohorts-Institution.csv") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    out["inzip_mr_md5"] = h.hexdigest()
    out["disk_mr_md5"] = md5(INST_FILE)
    assert out["inzip_mr_md5"] == out["disk_mr_md5"], "zip Most-Recent file differs from data/raw/scorecard_inst"
    h = hashlib.md5()
    with zf.open(f"{ZIP_DIR}/Most-Recent-Cohorts-Field-of-Study.csv") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    out["inzip_fos_md5"] = h.hexdigest()
    out["disk_fos_md5"] = md5(s58.FOS_FILE)
    assert out["inzip_fos_md5"] == out["disk_fos_md5"], "zip Most-Recent FoS file differs from data/raw/scorecard_fos"
    out["members"] = {n: (info[n].file_size, f"{info[n].CRC:08x}", "%04d-%02d-%02d %02d:%02d" % info[n].date_time[:5])
                      for n in sorted(info) if "/MERGED" in n and not n.startswith("__MACOSX")}
    out["n_merged"] = len(out["members"])
    rec("provenance", "zip_md5", np.nan, note=out["zip_md5"], n=ZIP_SIZE)
    rec("provenance", "zip_most_recent_inst_md5_equals_scorecard_inst", 1.0, note=out["inzip_mr_md5"])
    rec("provenance", "zip_most_recent_fos_md5_equals_scorecard_fos", 1.0, note=out["inzip_fos_md5"])
    for n, (sz, crc, dt) in out["members"].items():
        rec("provenance", "zip_member", np.nan, note=f"{n}; crc32 {crc}; {dt}", n=sz)
    for f, (m, sz) in DOCS.items():
        rec("provenance", "doc_md5", np.nan, note=f"{f}; {m}", n=sz)
    return out


def dictionary_checks(years_fall, years_pell, years_ctl=()) -> dict:
    """The cohort definitions this script relies on, read from the data dictionary and asserted."""
    x = pd.ExcelFile(HIST_DIR / "CollegeScorecardDataDictionary.xlsx")
    out = {}
    readme = pd.read_excel(x, "README", header=None, dtype=str)
    out["release"] = str(readme.iloc[1, 0]).strip()
    fos = pd.read_excel(x, "FieldOfStudy_Cohort_Map", header=None, dtype=str)
    col = list(fos.iloc[0]).index("FieldOfStudyMostRecent datafile")
    row = fos[fos[0] == "EARN_MDN_4YR"].iloc[0]
    out["fos_4yr"] = str(row[col])
    assert "AY2017-2018, AY2018-2019" in out["fos_4yr"] and "CY2022, CY2023" in out["fos_4yr"], out["fos_4yr"]
    mr = pd.read_excel(x, "Most_Recent_Inst_Cohort_Map", header=None, dtype=str).set_index(0)
    out["mr"] = {v: (str(mr.loc[v, 1]), str(mr.loc[v, 2])) for v in ("SAT_AVG", "ADM_RATE", "PCTPELL", "CONTROL", "STABBR")}
    assert out["mr"]["SAT_AVG"][0].startswith("Fall 2024") and out["mr"]["ADM_RATE"][0].startswith("Fall 2024")
    assert out["mr"]["PCTPELL"][0].startswith("AcadYr 2023-24")
    ic = pd.read_excel(x, "Institution_Cohort_Map", header=None, dtype=str)
    hdr = list(ic.iloc[0])
    icm = ic.set_index(0)
    out["hist"] = {}
    for y in years_fall:
        c = hdr.index(f"MERGED_{y}-{(y + 1) % 100:02d} datafile")
        for v in ("SAT_AVG", "ADM_RATE"):
            s = str(icm.loc[v].iloc[c - 1])
            assert s.startswith(f"Fall {y},"), (v, y, s)
            out["hist"][(v, y)] = s
    for y in years_pell:                                   # academic year y-(y+1) -> file y+1
        c = hdr.index(f"MERGED_{y + 1}-{(y + 2) % 100:02d} datafile")
        s = str(icm.loc["PCTPELL"].iloc[c - 1])
        assert s.startswith(f"AcadYr {y}-{(y + 1) % 100:02d},"), (y, s)
        out["hist"][("PCTPELL", y)] = s
    # first file with SAT_AVG / ADM_RATE / PCTPELL (the MERGED_2000-01 entries of SAT_AVG and ADM_RATE are empty)
    for v in ("SAT_AVG", "ADM_RATE", "PCTPELL"):
        vals = icm.loc[v].iloc[:len(hdr) - 1]
        first = next(h for h, s in zip(hdr[1:], vals) if isinstance(s, str) and s != "nan")
        out[f"first_{v}"] = first
    assert out["first_SAT_AVG"] == "MERGED_2001-02 datafile" and out["first_ADM_RATE"] == "MERGED_2001-02 datafile"
    assert out["first_PCTPELL"] == "MERGED_2008-09 datafile"
    c00 = hdr.index("MERGED_2000-01 datafile")
    for v in ("SAT_AVG", "ADM_RATE"):
        assert str(icm.loc[v].iloc[c00 - 1]) == "nan", v
    out["adm_2001"] = str(icm.loc["ADM_RATE"].iloc[hdr.index("MERGED_2001-02 datafile") - 1])
    assert out["adm_2001"].startswith("Fall 2001")
    # STABBR is backfilled with the most recent IPEDS value in every MERGED file; CONTROL is dated (no backfill clause)
    backfill = "unless more recent data are available. Then the most recent data reported to IPEDS"
    out["stabbr_backfill"], out["control_dated"] = {}, {}
    for y in sorted(set(years_fall) | set(years_ctl)):
        c = hdr.index(f"MERGED_{y}-{(y + 1) % 100:02d} datafile")
        s_st, s_ct = str(icm.loc["STABBR"].iloc[c - 1]), str(icm.loc["CONTROL"].iloc[c - 1])
        assert backfill in s_st and s_st.startswith(f"AcadYr {y}-"), (y, s_st)
        assert backfill not in s_ct and s_ct.startswith(f"AcadYr {y}-"), (y, s_ct)
        out["stabbr_backfill"][y], out["control_dated"][y] = s_st, s_ct
    return out


# (document, 0-based PDF page, printed page, phrase) -- asserted in the PDF text (whitespace collapsed)
DOC_QUOTES = [
    ("FieldOfStudyDataDocumentation.pdf", 3, "3",
     "The cohort of evaluated graduates for earnings metrics consists of those individuals who received federal "
     "financial aid, but excludes"),
    ("FieldOfStudyDataDocumentation.pdf", 3, "3", "were subsequently enrolled in school during the measurement year"),
    ("FieldOfStudyDataDocumentation.pdf", 3, "3", "excluded those who received a higher-level credential than the "
     "credential level of the field of study measured"),
    ("FieldOfStudyDataDocumentation.pdf", 8, "8", "the \u201cmost recent\u201d data file will have data for all "
     "earnings metrics, pulling that information from the last time it was calculated"),
    ("FieldOfStudyDataDocumentation.pdf", 9, "9", "Scorecard combines students into 2-year cohorts to increase cohort "
     "cell sizes"),
    ("InstitutionDataDocumentation.pdf", 10, "11", "Colleges report to IPEDS their Fall admissions rate"),
    ("InstitutionDataDocumentation.pdf", 11, "12", "SAT and ACT data are available from 2001-02 on"),
    ("InstitutionDataDocumentation.pdf", 11, "12", "Starting with the 2017-18 Scorecard data, the SAT scores reported "
     "were under the new (post-March 2016) scoring system"),
    ("InstitutionDataDocumentation.pdf", 17, "18", "describe the share of degree/certificate-seeking undergraduate "
     "students who received Pell Grants in a given year"),
]


def doc_checks() -> dict:
    """The documentation statements this script relies on, found on the cited pages (asserted)."""
    import re
    from pypdf import PdfReader
    out, texts = {}, {}
    for f, pg, printed, q in DOC_QUOTES:
        if f not in texts:
            r = PdfReader(str(HIST_DIR / f))
            texts[f] = [re.sub(r"\s+", " ", (x.extract_text() or "")) for x in r.pages]
            out[f"{f}:version"] = re.search(r"Version: ([A-Za-z]+ \d{4})", texts[f][0]).group(1)
            out[f"{f}:pages"] = len(r.pages)
        assert q in texts[f][pg], (f, pg, q)
        out.setdefault("quotes", []).append((f, pg + 1, printed, q))
    return out


def read_hist(file_years) -> dict:
    """file year y -> MERGEDy_(y+1)_PP.csv, UNITID-indexed, numeric (NULL / PrivacySuppressed -> NaN)."""
    zf = zipfile.ZipFile(ZIP)
    out = {}
    for y in sorted(set(file_years)):
        with zf.open(f"{ZIP_DIR}/{fname_fall(y)}") as fh:
            d = pd.read_csv(fh, usecols=HIST_COLS, dtype=str)
        assert d.UNITID.is_unique, y
        for c in ("SAT_AVG", "ADM_RATE", "PCTPELL", "CONTROL"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        out[y] = d.set_index("UNITID")
    return out


def era_table(H: dict, falls, pell, ctl) -> pd.DataFrame:
    """Per UNITID: mean SAT_AVG / ADM_RATE over the given falls (fall y is in file y), mean PCTPELL over the given
    academic years (AY y-(y+1) is in file y+1), CONTROL / STABBR from the first file of `ctl` that has them."""
    ids = sorted(set().union(*[set(H[y].index) for y in set(falls) | {a + 1 for a in pell} | set(ctl)]))
    out = pd.DataFrame(index=pd.Index(ids, name="UNITID"))
    for v in ("SAT_AVG", "ADM_RATE"):
        M = pd.concat([H[y][v].reindex(ids) for y in falls], axis=1)
        out[v] = M.mean(axis=1, skipna=True)
        out[f"n_{v}"] = M.notna().sum(axis=1)
    if pell:
        M = pd.concat([H[a + 1]["PCTPELL"].reindex(ids) for a in pell], axis=1)
        out["PCTPELL"] = M.mean(axis=1, skipna=True)
    else:
        out["PCTPELL"] = np.nan
    ctl_v = pd.Series(np.nan, index=ids)
    st_v = pd.Series(np.nan, index=ids, dtype=object)
    for y in ctl[::-1]:                                   # earlier choices overwrite later ones
        c = H[y]["CONTROL"].reindex(ids)
        s = H[y]["STABBR"].reindex(ids)
        ctl_v = c.where(c.notna(), ctl_v)
        st_v = s.where(s.notna(), st_v)
    out["CONTROL"] = ctl_v
    out["STABBR"] = st_v
    return out.reset_index()


# ---------------------------------------------------------------------------------------------------------------
# weighted rank (partial) correlations: institution multinomial weights == resampling with replacement
# ---------------------------------------------------------------------------------------------------------------
def _dummies(labels) -> np.ndarray | None:
    _, inv = np.unique(np.asarray(labels), return_inverse=True)
    L = int(inv.max()) + 1
    return np.eye(L)[inv][:, 1:] if L > 1 else None


def wpartial(P, Ys, conts, cats, W):
    """Rank partial correlations of P with each column of Ys given covariates under institution weights.
    P (n,), Ys (n, m), conts: list of (n,), cats: list of (n,) labels, W (B, n) (copies of each row). Ranks are the
    average ranks in the resample (scripts/59 wrank), continuous covariates enter as ranks and categorical ones as
    dummies (scripts/55 partial_rank), the residualization is weighted least squares (= OLS on the resample).
    Returns (B, m); NaN where a residual variance is 0."""
    Bn, n = W.shape
    m = Ys.shape[1]
    rP = s59.wrank(P, W)
    rY = np.stack([s59.wrank(Ys[:, k], W) for k in range(m)], -1)          # (B, n, m)
    if not conts and not cats:
        return np.stack([s59.wcorr(rP, rY[..., k], W) for k in range(m)], -1)
    Z = np.stack([np.ones((Bn, n))] + [s59.wrank(c, W) for c in conts], -1)
    dm = [d for d in (_dummies(c) for c in cats) if d is not None]
    if dm:
        D = np.hstack(dm)
        Z = np.concatenate([Z, np.broadcast_to(D, (Bn,) + D.shape)], -1)
    V = np.concatenate([rP[..., None], rY], -1)
    ZW = Z * W[..., None]
    A = np.matmul(ZW.transpose(0, 2, 1), Z)
    R = np.matmul(ZW.transpose(0, 2, 1), V)
    s = np.sqrt(np.clip(np.einsum("bii->bi", A), 1e-300, None))           # column scaling before the pseudo-inverse
    As = A / s[:, :, None] / s[:, None, :]
    coef = np.matmul(np.linalg.pinv(As, rcond=1e-10, hermitian=True), R / s[:, :, None]) / s[:, :, None]
    E = V - np.matmul(Z, coef)
    e0, ek = E[..., 0], E[..., 1:]
    num = np.einsum("bn,bn,bnm->bm", W, e0, ek)
    den = np.sqrt(np.einsum("bn,bn->b", W, e0 * e0)[:, None] * np.einsum("bn,bnm->bm", W, ek * ek))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    scale = np.sqrt(np.einsum("bn,bn->b", W, (rP - (W * rP).sum(1, keepdims=True) / W.sum(1, keepdims=True)) ** 2))
    bad = den <= 1e-9 * (scale[:, None] ** 2 + 1e-300)
    return np.where(bad, np.nan, out)


def rng_rep(tag: str, b0: int, b1: int, j: int | None = None) -> list:
    base = np.random.SeedSequence([SEED, zlib.crc32(tag.encode())] + ([j] if j is not None else []))
    return [np.random.default_rng(s) for s in base.spawn(B)[b0:b1]]


def draw_counts(gens, n_units: int) -> np.ndarray:
    p = np.full(n_units, 1.0 / n_units)
    return np.stack([g.multinomial(n_units, p) for g in gens]).astype(float)


# ---------------------------------------------------------------------------------------------------------------
# worker tasks (fork; data in the module global _D, set before the pool is created)
# ---------------------------------------------------------------------------------------------------------------
_D: dict = {}


def _t1_field_stats(fd: dict, W: np.ndarray) -> np.ndarray:
    """All estimable T1a statistics of one field under weights W (B, n_field) -> (B, n_stats)."""
    out = np.full((W.shape[0], len(fd["keys"])), np.nan)
    for i, (smp, spec, ver) in enumerate(fd["keys"]):
        r = fd["rows"][smp]
        Wr = W[:, r]
        conts, cats = SPEC1[spec]
        v = "26" if ver == "-" else ver
        out[:, i] = wpartial(fd["P"][r], fd["Y"][r][:, None], [fd[f"{c}_{v}" if c in VCOLS else c][r] for c in conts],
                             [fd[f"{c}_{v}"][r] for c in cats], Wr)[:, 0]
    return out


def task_t1(args):
    kind, group, j, b0, b1 = args
    with threadpool_limits(1):
        D = _D["t1"]
        fl = D["fields"] if kind == "shared" else [j]
        tag = f"t1_{kind}_{group}"
        gens = rng_rep(tag, b0, b1, None if kind == "shared" else zlib.crc32(j.encode()))   # j = field name
        Wu = draw_counts(gens, D["N"])
        res = {f: None for f in fl}
        redrawn = 0
        todo = np.arange(b1 - b0)
        for _ in range(200):
            bad = np.zeros(len(todo), bool)
            for f in fl:
                fd = D["FD"][f][group]
                if not fd["keys"]:
                    continue
                v = _t1_field_stats(fd, Wu[todo][:, fd["inst_idx"]])
                if res[f] is None:
                    res[f] = np.full((b1 - b0, len(fd["keys"])), np.nan)
                res[f][todo] = v
                bad |= ~np.isfinite(v).all(1)
            if not bad.any():
                return dict(res=res, redrawn=redrawn, b0=b0)
            todo = todo[bad]
            redrawn += len(todo)
            for t in todo:
                Wu[t] = draw_counts([gens[t]], D["N"])[0]
        raise AssertionError(f"T1 draws stayed degenerate ({tag}, {j})")


def fe_demean(Xr, yr, S, groups, w):
    """Weighted within-institution demeaning (institution FE absorbed; Frisch-Waugh-Lovell). S: sparse (G, n)
    institution indicator. Rows with w = 0 do not enter the means."""
    ws = S @ w
    inv = np.divide(1.0, ws, out=np.zeros_like(ws), where=ws > 0)
    mx = (S @ (Xr * w[:, None])) * inv[:, None]
    my = (S @ (yr * w)) * inv
    return Xr - mx[groups], yr - my[groups]


def fe_solve_w(D, key, w):
    """Pooled beta of design `key` under row weights w: WLS with institution dummies (weighted demeaning) when the
    design absorbs institution FE, plain WLS otherwise. Equals OLS on the resample in which row r appears w[r]
    times with each institution's FE shared across its rows."""
    Xr, yr, j = D["X"][key], D["y"][key], D["j"][key]
    if D["absorb"][key]:
        X, y = fe_demean(Xr, yr, D["S"], D["groups"], w)
    else:
        X, y = Xr, yr
    return fe_solve(X, y, w, j)


def fe_solve(X, y, w, j):
    Xw = X * w[:, None]
    XtX = X.T @ Xw
    Xty = Xw.T @ y
    keep = np.diag(XtX) > 1e-10 * max(1.0, float(np.max(np.diag(XtX))))
    if not keep[j]:
        return np.nan
    Ak, bk = XtX[np.ix_(keep, keep)], Xty[keep]
    jk = int(np.cumsum(keep)[j]) - 1
    sol = None
    try:
        cf = cho_factor(Ak)
        sol = cho_solve(cf, bk)
        if not np.all(np.isfinite(sol)) or np.linalg.norm(Ak @ sol - bk) > 1e-6 * max(1.0, np.linalg.norm(bk)):
            sol = None
    except np.linalg.LinAlgError:
        sol = None
    if sol is None:
        sol = np.linalg.lstsq(Ak, bk, rcond=None)[0]
    return float(sol[jk])


def task_fe(args):
    kind, b0, b1 = args
    with threadpool_limits(1):
        D = _D["fe"]
        G, groups, code, K = D["G"], D["groups"], D["code"], D["K"]
        if kind == "shared":
            gens = rng_rep("fe_shared", b0, b1)
        else:
            gens = [rng_rep("fe_ind", b0, b1, f) for f in range(K)]
        out = np.full((b1 - b0, len(D["keys"])), np.nan)
        redrawn = 0
        for t in range(b1 - b0):
            for _ in range(200):
                if kind == "shared":
                    c = draw_counts([gens[t]], G)[0]
                    w = c[groups]
                else:
                    C = np.stack([draw_counts([gens[f][t]], G)[0] for f in range(K)])   # (K, G)
                    w = C[code, groups]
                v = np.array([fe_solve_w(D, key, w) for key in D["keys"]])
                if np.isfinite(v).all():
                    out[t] = v
                    break
                redrawn += 1
            else:
                raise AssertionError("FE draws stayed degenerate")
        return dict(res=out, redrawn=redrawn, b0=b0)


def _t3_cell_stats(cd: dict, W: np.ndarray, grp: str) -> np.ndarray:
    """(B, n_keys, 3) coupling per horizon for each (spec, version) of one cell in bootstrap group grp."""
    out = np.full((W.shape[0], len(cd["gkeys"][grp]), 3), np.nan)
    for i, (spec, ver) in enumerate(cd["gkeys"][grp]):
        conts, cats = SPEC3[spec]
        out[:, i, :] = wpartial(cd["P"], cd["E"], [cd[f"{c}_{ver}" if c != "ST_EARN" else c] for c in conts],
                                [cd[f"{c}_{ver}"] for c in cats], W)
    return out


def task_t3(args):
    kind, grp, j, b0, b1 = args
    with threadpool_limits(1):
        D = _D["t3"]
        cl = D["cells_in"][grp] if kind == "shared" else D["cells_of_field"][grp][j]
        gens = rng_rep(f"t3_{kind}_{grp}", b0, b1, None if kind == "shared" else j)
        Wu = draw_counts(gens, D["NI"])
        res = {c: None for c in cl}
        redrawn = 0
        todo = np.arange(b1 - b0)
        for _ in range(200):
            bad = np.zeros(len(todo), bool)
            for c in cl:
                cd = D["cells"][c]
                v = _t3_cell_stats(cd, Wu[todo][:, cd["inst_idx"]], grp)
                if res[c] is None:
                    res[c] = np.full((b1 - b0,) + v.shape[1:], np.nan)
                res[c][todo] = v
                bad |= ~np.isfinite(v.reshape(len(todo), -1)).all(1)
            if not bad.any():
                return dict(res=res, redrawn=redrawn, b0=b0)
            todo = todo[bad]
            redrawn += len(todo)
            for t in todo:
                Wu[t] = draw_counts([gens[t]], D["NI"])[0]
        raise AssertionError("T3 draws stayed degenerate")


def run_pool(fn, tasks, workers):
    if workers <= 1:
        return [fn(t) for t in tasks]
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        return list(ex.map(fn, tasks, chunksize=1))


def chunks():
    return [(b, min(b + CHUNK, B)) for b in range(0, B, CHUNK)]


def n_workers() -> int:
    w = max(1, min(6, int(os.environ.get("ERA68_WORKERS", "2"))))
    avail = None
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable:"):
                avail = int(ln.split()[1]) / 1024
    except OSError:
        pass
    if avail is not None and avail < 1536:
        w = 1
    elif avail is not None and avail < 2048:
        w = min(w, 2)
    print(f"[workers] {w} (MemAvailable {avail:.0f} MB)" if avail else f"[workers] {w}", flush=True)
    return w


# ---------------------------------------------------------------------------------------------------------------
# inference helpers
# ---------------------------------------------------------------------------------------------------------------
def tw(est, v_field, cb, ib) -> dict:
    """scripts/59 two-way variance + p, MDE and the equivalence 90% CI."""
    r = s59.twoway(est, v_field, np.asarray(cb, float), np.asarray(ib, float))
    se = r["tse"]
    r.update(est=float(est), p=float(2 * norm.sf(abs(est / se))) if se > 0 else np.nan, mde=(Z975 + Z80) * se,
             lo90=est - Z95 * se, hi90=est + Z95 * se, mc90=r["tmcse"] * Z95 / Z975)
    # conservative variance (added after verification, not pre-specified): the CGM two-way variance can fall below
    # one of its one-way parts when V_field x inst exceeds V_field or V_inst; max(V_two-way, V_field, V_inst)
    vc = max(se * se, r["tvf"], r["tvs"])
    sc = float(np.sqrt(vc))
    r.update(cse=sc, clo=est - Z975 * sc, chi=est + Z975 * sc, clo90=est - Z95 * sc, chi90=est + Z95 * sc,
             cp=float(2 * norm.sf(abs(est / sc))) if sc > 0 else np.nan, cbind=bool(sc > se * (1 + 1e-9)),
             cwhich=("two-way" if not sc > se * (1 + 1e-9) else "V_field" if r["tvf"] >= r["tvs"] else "V_inst"))
    return r


def sig2(x: float) -> str:
    """Two significant digits, trailing zeros kept (0.0030, 0.0048, 0.043)."""
    return f"{float(f'{x:.2g}'):#.2g}"


def reading(r: dict, delta: float, cons: bool = False, lab: str | None = None) -> str:
    lo, hi, lo90, hi90 = ((r["clo"], r["chi"], r["clo90"], r["chi90"]) if cons else
                          (r["tlo"], r["thi"], r["lo90"], r["hi90"]))
    if lo > 0 or hi < 0:
        return "differs"
    if lo90 > -delta and hi90 < delta:
        return f"equivalent within ±{lab if lab is not None else f'{delta:g}'}"
    return "inconclusive"


def near_margin(r: dict, delta: float, lab: str | None = None) -> str:
    """Empty unless the (two-way) reading changes when the SE moves by 2 Monte Carlo SEs (scripts/59 tmcse / 1.96 is
    the Monte Carlo SE of the SE), i.e. a deciding CI endpoint lies within 2 Monte Carlo SEs of its threshold."""
    base = reading(r, delta, lab=lab)
    m = r["tmcse"] / Z975
    alt = []
    for s2 in (r["tse"] - 2 * m, r["tse"] + 2 * m):
        e = r["est"]
        rr = dict(r, tlo=e - Z975 * s2, thi=e + Z975 * s2, lo90=e - Z95 * s2, hi90=e + Z95 * s2)
        x = reading(rr, delta, lab=lab)
        if x != base and short(x) not in alt:
            alt.append(short(x))
    if not alt:
        return ""
    return (f"{short(base)} becomes {' or '.join(alt)} if the SE moves by 2 Monte Carlo SEs; 95% CI "
            f"[{r['tlo']:+.4f}, {r['thi']:+.4f}], 90% CI [{r['lo90']:+.4f}, {r['hi90']:+.4f}] vs ±{lab or f'{delta:g}'}; "
            f"Monte Carlo SE of a 90% bound {r['mc90']:.2g}")


def jack_var(vals_minus) -> float:
    th = np.asarray(vals_minus, float)
    K = len(th)
    return float((K - 1) / K * ((th - th.mean()) ** 2).sum())


def boot_spearman(x, y, tag):
    """Institution bootstrap (percentile) of Spearman(x, y); per-replicate seeds."""
    n = len(x)
    gens = rng_rep(tag, 0, B)
    idx = np.stack([g.integers(0, n, n) for g in gens])
    rx, ry = rankdata(x[idx], axis=1), rankdata(y[idx], axis=1)
    bs = s52.corr_rows(rx, ry)
    est = float(s52.corr_rows(rankdata(x)[None, :], rankdata(y)[None, :])[0])
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ---------------------------------------------------------------------------------------------------------------
# T1: Scorecard 4-year earnings
# ---------------------------------------------------------------------------------------------------------------
def build_t1(H, inst26):
    ar = s55.load_ar_wapman(fields=FIELDS66)
    er = s55.load_er_scorecard("undergrad", fields=FIELDS66)
    cells = s55.build_cells(ar, er, inst26)
    del er
    assert cells.INSTNM.notna().all()
    E = era_table(H, **ERA_FOS)
    Wd = era_table(H, **ERA_FOS_W)
    for tag, T in (("E", E), ("W", Wd)):
        cells = cells.merge(T[["UNITID"] + VCOLS].rename(columns={c: f"{c}_{tag}" for c in VCOLS}), on="UNITID",
                            how="left")
    for c in VCOLS:
        cells[f"{c}_26"] = cells[c]
    for v in VERS:
        cells[f"STABBR_{v}"] = cells[f"STABBR_{v}"].fillna("")
        need = [f"{c}_{v}" for c in ("SAT_AVG", "ADM_RATE", "PCTPELL", "CONTROL")] + ["ST_EARN"]
        cells[f"ok_{v}"] = cells[need].notna().all(axis=1) & (cells[f"STABBR_{v}"] != "")
    # E-complete implies W-complete (W's years contain E's)
    assert not (cells.ok_E & ~cells.ok_W).any()
    cells["ok_C"] = cells.ok_26 & cells.ok_E
    st_diff = int((cells.ok_C & (cells.STABBR_E != cells.STABBR_26)).sum())
    uC = cells[cells.ok_C].drop_duplicates("UNITID")
    ctl_diff = int((uC.CONTROL_E != uC.CONTROL_26).sum())
    return ar, cells, dict(stabbr_differs_cells=st_diff, control_differs_insts=ctl_diff)


def t1_point(cells):
    """Point estimates (scripts/55 spear / partial_rank) for every (field, sample, spec, version); estimability."""
    universe = sorted(cells.inst_key.unique())
    iidx = {k: i for i, k in enumerate(universe)}
    fields = sorted(cells.field.unique())
    samples = {"C": "ok_C", "S26": "ok_26", "SE": "ok_E"}
    keys_all = ([("C", "raw", "-")] + [("C", s, v) for s in ("pcs", "pcs_adm", "broad", "sel") for v in VERS]
                + [("S26", s, "26" if s != "raw" else "-") for s in LADDER]
                + [("SE", s, "E" if s != "raw" else "-") for s in LADDER])
    keys_strict = [("C", "strict", v) for v in ("26", "E")]
    FD, point, dfs, nn = {}, {}, {}, {}
    for f in fields:
        d = cells[cells.field == f]
        base = dict(P=d.prestige_score.to_numpy(float), Y=d.earnings.to_numpy(float),
                    ST_EARN=d.ST_EARN.to_numpy(float), inst_idx=d.inst_key.map(iidx).to_numpy(int))
        for v in VERS:
            for c in VCOLS:
                col = d[f"{c}_{v}"]
                base[f"{c}_{v}"] = (col.fillna("").to_numpy(str) if c == "STABBR" else
                                    col.fillna(-1).to_numpy(int) if c == "CONTROL" else col.to_numpy(float))
        rows = {s: np.flatnonzero(d[col].to_numpy()) for s, col in samples.items()}
        for s, r in rows.items():
            nn[(f, s)] = len(r)
        est = {}
        for key in keys_all + keys_strict:
            smp, spec, ver = key
            r = rows[smp]
            if len(r) < NMIN:
                continue
            v = "26" if ver == "-" else ver
            conts, cats = SPEC1[spec]
            if spec == "raw":
                est[key] = (s55.spear(base["P"][r], base["Y"][r]), len(r) - 2)
            else:
                est[key] = s55.partial_rank(base["P"][r], base["Y"][r],
                                            [base[f"{c}_{v}" if c in VCOLS else c][r] for c in conts],
                                            [base[f"{c}_{v}"][r] for c in cats])
        for key, (val, df) in est.items():
            point[(f, key)] = val
            dfs[(f, key)] = df
        FD[f] = dict(base=base, rows=rows)
    return universe, fields, FD, point, dfs, nn, keys_all, keys_strict


def t1_fieldsets(fields, point, dfs, nn):
    ok = lambda f, key: (f, key) in point and np.isfinite(point[(f, key)]) and (key[1] == "raw" or dfs[(f, key)] >= DFMIN)
    fs = {}
    fs["ladder_C"] = [f for f in fields if nn[(f, "C")] >= NMIN and ok(f, ("C", "raw", "-"))
                      and all(ok(f, ("C", s, v)) for s in ("pcs", "pcs_adm", "broad", "sel") for v in VERS)]
    fs["strict_C"] = [f for f in fields if all(ok(f, ("C", "strict", v)) for v in ("26", "E"))]
    for smp, v in (("S26", "26"), ("SE", "E")):
        for s in LADDER:
            fs[f"{smp}_{s}"] = [f for f in fields if ok(f, (smp, s, "-" if s == "raw" else v))]
    # era-complete vs scripts/55 sample: fields estimable in both, per spec
    for s in LADDER:
        fs[f"SEvS26_{s}"] = [f for f in fs[f"S26_{s}"] if f in fs[f"SE_{s}"]]
    return fs, ok


def check_engine(FD, fields, point, keys):
    """wpartial at unit weights == scripts/55 partial_rank / spear (all computed point estimates)."""
    worst = 0.0
    for f in fields:
        base, rows = FD[f]["base"], FD[f]["rows"]
        for key in keys:
            if (f, key) not in point or not np.isfinite(point[(f, key)]):
                continue
            smp, spec, ver = key
            r = rows[smp]
            v = "26" if ver == "-" else ver
            conts, cats = SPEC1[spec]
            w = wpartial(base["P"][r], base["Y"][r][:, None], [base[f"{c}_{v}" if c in VCOLS else c][r] for c in conts],
                         [base[f"{c}_{v}"][r] for c in cats], np.ones((1, len(r))))[0, 0]
            worst = max(worst, abs(w - point[(f, key)]))
    assert worst < 1e-9, worst
    return worst


def check_s55(fields, point, fs):
    ref = pd.read_csv(S55_CSV).set_index("field")
    worst, n = 0.0, 0
    for f in fields:
        a = point.get((f, ("S26", "broad", "26")), np.nan)
        b = ref.loc[f, "rho_full_stlev"] if f in ref.index else np.nan
        est_mine = f in fs["S26_broad"]
        assert est_mine == bool(np.isfinite(b)), f
        if est_mine:
            worst = max(worst, abs(a - b)); n += 1
    assert worst < 1e-5, worst
    for s, col in (("raw", "rho_raw_sat"), ("sel", "rho_sel"), ("pcs", "rho_pcs"), ("pcs_adm", "rho_pcs_adm")):
        for f in fs[f"S26_{s}"]:
            assert abs(point[(f, ("S26", s, "-" if s == "raw" else "26"))] - ref.loc[f, col]) < 1e-5, (f, s)
    return worst, n


T1_GROUPS = ("ladder", "strict", "sec")


def t1_groups(fs, keys_all, keys_strict) -> dict:
    """Bootstrap groups: group -> {field: keys}. A replicate is redrawn only when a statistic of its OWN group is not
    finite, so the primary ladder is not conditioned on the estimability of the strict or secondary-sample statistics
    (or of fields outside the fixed ladder field set). ladder = the common-program ladder keys for the fixed T1a field
    set; strict = strict keys + raw for the strict field set; sec = the era-complete / scripts/55-sample keys for their
    own field sets."""
    G = {"ladder": {f: [k for k in keys_all if k[0] == "C"] for f in fs["ladder_C"]},
         "strict": {f: [("C", "raw", "-")] + list(keys_strict) for f in fs["strict_C"]},
         "sec": {}}
    for s in LADDER:
        for smp, v in (("SE", "E"), ("S26", "26")):
            key = (smp, s, "-" if s == "raw" else v)
            for f in fs[f"{smp}_{s}"]:
                G["sec"].setdefault(f, []).append(key)
    return G


def t1_boot(FD, fields, universe, groups, point, workers):
    """Shared and independent-per-field institution draws for the T1a statistics, one set per group."""
    D = {"N": len(universe), "fields": fields, "FD": {}}
    for f in fields:
        base, rows = FD[f]["base"], FD[f]["rows"]
        D["FD"][f] = {}
        for grp in T1_GROUPS:
            kk = groups[grp].get(f, [])
            assert all(np.isfinite(point[(f, k)]) for k in kk)
            D["FD"][f][grp] = dict(keys=kk, rows=rows, inst_idx=base["inst_idx"], **{k: v for k, v in base.items()
                                                                                    if k != "inst_idx"})
    _D["t1"] = D
    out = {}
    for grp in T1_GROUPS:
        tasks = [("shared", grp, None, b0, b1) for b0, b1 in chunks()]
        out[("shared", grp)] = run_pool(task_t1, tasks, workers)
        tasks = [("ind", grp, f, b0, b1) for f in fields if groups[grp].get(f) for b0, b1 in chunks()]
        out[("ind", grp)] = run_pool(task_t1, tasks, workers)
        stage(f"T1a bootstrap {grp} done")
    # assemble: arrays (B,) per (group, draw type, key, field)
    A = {}
    for (kind, grp), res in out.items():
        for f in fields:
            for key in D["FD"][f][grp]["keys"]:
                A.setdefault((grp, kind, key), {})[f] = np.full(B, np.nan)
        for r in res:
            for f, v in r["res"].items():
                if v is None:
                    continue
                for i, key in enumerate(D["FD"][f][grp]["keys"]):
                    A[(grp, kind, key)][f][r["b0"]:r["b0"] + v.shape[0]] = v[:, i]
    for k, d in A.items():
        assert all(np.isfinite(x).all() for x in d.values()), k
    red = {f"{kind}_{grp}": int(sum(r["redrawn"] for r in res)) for (kind, grp), res in out.items()}
    return A, red


def t1_stat(A, point, fl, key_a, key_b=None, grp="ladder"):
    """Mean over fields fl of key_a (minus key_b): estimate + two-way inference (replicates of bootstrap group grp)."""
    pa = np.array([point[(f, key_a)] for f in fl])
    ca = np.mean([A[(grp, "shared", key_a)][f] for f in fl], axis=0)
    ia = np.mean([A[(grp, "ind", key_a)][f] for f in fl], axis=0)
    if key_b is not None:
        pb = np.array([point[(f, key_b)] for f in fl])
        pa = pa - pb
        ca = ca - np.mean([A[(grp, "shared", key_b)][f] for f in fl], axis=0)
        ia = ia - np.mean([A[(grp, "ind", key_b)][f] for f in fl], axis=0)
    r = tw(float(pa.mean()), float(pa.var(ddof=1) / len(pa)), ca, ia)
    r["k"] = len(fl)
    return r


def t1_share(A, point, fl, v, v0=None):
    """Exploratory: 1 - mean broad(v) / mean raw over fields fl (minus the same for v0); two-way variance with the
    field part from a delete-one-field jackknife."""
    raw = ("C", "raw", "-")

    def sh(pr, pb):
        return 1.0 - pb.mean(axis=0) / pr.mean(axis=0)

    def val(idx, src):
        pr = np.array([src(f, raw) for f in np.asarray(fl)[idx]])
        out = sh(pr, np.array([src(f, ("C", "broad", v)) for f in np.asarray(fl)[idx]]))
        if v0 is not None:
            out = out - sh(pr, np.array([src(f, ("C", "broad", v0)) for f in np.asarray(fl)[idx]]))
        return out
    allidx = np.arange(len(fl))
    est = float(val(allidx, lambda f, k: point[(f, k)]))
    jk = [float(val(np.delete(allidx, i), lambda f, k: point[(f, k)])) for i in allidx]
    cb = val(allidx, lambda f, k: A[("ladder", "shared", k)][f])
    ib = val(allidx, lambda f, k: A[("ladder", "ind", k)][f])
    r = tw(est, jack_var(jk), cb, ib)
    r["k"] = len(fl)
    return r


def t1_stat_x(A, point, fl_a, key_a, fl_b, key_b):
    """Difference of means over two field sets (secondary: era-complete vs scripts/55 sample); V_field by jackknife
    over the union of fields."""
    U = sorted(set(fl_a) | set(fl_b))

    def val(fs_a, fs_b):
        return np.mean([point[(f, key_a)] for f in fs_a]) - np.mean([point[(f, key_b)] for f in fs_b])
    est = float(val(fl_a, fl_b))
    jk = [val([g for g in fl_a if g != f], [g for g in fl_b if g != f]) for f in U]
    cb = (np.mean([A[("sec", "shared", key_a)][f] for f in fl_a], 0)
          - np.mean([A[("sec", "shared", key_b)][f] for f in fl_b], 0))
    ib = np.mean([A[("sec", "ind", key_a)][f] for f in fl_a], 0) - np.mean([A[("sec", "ind", key_b)][f] for f in fl_b], 0)
    r = tw(est, jack_var(jk), cb, ib)
    r["k"] = f"{len(fl_a)} vs {len(fl_b)}"
    return r


# ---------------------------------------------------------------------------------------------------------------
# T1b: within-institution design
# ---------------------------------------------------------------------------------------------------------------
def fe_raw(cv, slopes):
    """scripts/58 fe_build (per_field=False, no weights) WITHOUT the within-institution demeaning, so that the
    institution FE can be re-absorbed under any row weights. Checked against fe_build: demeaning this design
    reproduces fe_build's X and y (asserted)."""
    m = s58.fe_build(cv, slopes, [], "earnings", per_field=False)
    d = m["d"]
    fields = m["fields"]
    code = pd.Categorical(d.field, categories=fields).codes
    Dm = np.eye(len(fields))[code]
    g = d.groupby("field")
    z = g.prestige_score.transform(lambda s: (s - s.mean()) / s.std()).to_numpy()
    cols = [Dm[:, 1:], z[:, None]]
    for c in slopes:
        zc = g[c].transform(lambda s: (s - s.mean()) / s.std()).fillna(0).to_numpy()
        cols.append(Dm * zc[:, None])
    Xr = np.hstack(cols)
    yr = np.log(d["earnings"].to_numpy(float))
    ic, cnt = m["groups"], m["cnt"]
    S = csr_matrix((np.ones(len(ic)), (ic, np.arange(len(ic)))), shape=(cnt.size, len(ic)))
    Xd, yd = fe_demean(Xr, yr, S, ic, np.ones(len(ic)))
    assert Xd.shape == m["X"].shape and np.max(np.abs(Xd - m["X"])) < 1e-10 and np.max(np.abs(yd - m["y"])) < 1e-10
    assert np.max(np.abs(z - m["z"])) == 0
    return dict(X=Xr, y=yr, names=m["names"], groups=ic, cnt=cnt, code=code, fields=fields, d=d, z=z, S=S,
                Xdm=m["X"], ydm=m["y"], absorb=True)


def build_fe(cells):
    """T1b designs on the common programs (SAT family complete under the 26 and E controls, hence under W) with
    brand G; every spec and version on the same cells (asserted)."""
    need = [f"{c}_{v}" for c in ("SAT_AVG", "ADM_RATE", "PCTPELL") for v in VERS] + ["G"]
    fc = cells[cells.ok_C].dropna(subset=need).copy()
    assert fc.ok_W.all()
    M = {}
    ref = None
    for spec, sl in FE_SPECS.items():
        if sl is None:
            continue
        for v in (VERS if sl else ["-"]):
            # every version's selectivity columns are required in fc, so fe_sample keeps the same cells
            cv = fc if v in ("-", "26") else fc.assign(**{c: fc[f"{c}_{v}"] for c in ("SAT_AVG", "ADM_RATE", "PCTPELL")})
            if v == "26":
                assert np.allclose(fc["SAT_AVG"], fc["SAT_AVG_26"]) and np.allclose(fc["PCTPELL"], fc["PCTPELL_26"])
            m = fe_raw(cv, sl)
            if ref is None:
                ref = m
            assert m["d"].inst_key.equals(ref["d"].inst_key) and m["d"].field.equals(ref["d"].field), (spec, v)
            M[(spec, v)] = m
    # field FE only (no institution FE), same cells
    d = ref["d"]
    K = len(ref["fields"])
    Dm = np.eye(K)[ref["code"]]
    M[("nofe", "-")] = dict(X=np.column_stack([Dm, ref["z"]]), y=np.log(d.earnings.to_numpy(float)),
                            names=[f"g_{f}" for f in ref["fields"]] + ["beta"], groups=ref["groups"], code=ref["code"],
                            cnt=ref["cnt"], fields=ref["fields"], d=d, S=ref["S"], absorb=False)
    return M, ref


def fe_design_dict(M, ref):
    keys = sorted(M, key=lambda k: (list(FE_SPECS).index(k[0]), k[1]))
    return dict(G=int(ref["cnt"].size), groups=ref["groups"], code=ref["code"], K=len(ref["fields"]), keys=keys,
                S=ref["S"], X={k: M[k]["X"] for k in keys}, y={k: M[k]["y"] for k in keys},
                j={k: M[k]["names"].index("beta") for k in keys}, absorb={k: M[k]["absorb"] for k in keys})


def fe_point(M, D):
    P = {}
    for key, m in M.items():
        j = m["names"].index("beta")
        P[key] = fe_solve_w(D, key, np.ones(len(m["y"])))
        # the weighted solver reproduces plain least squares on the (fe_build-demeaned) design
        X, y = (m["Xdm"], m["ydm"]) if m["absorb"] else (m["X"], m["y"])
        assert abs(P[key] - np.linalg.lstsq(X, y, rcond=None)[0][j]) < 1e-9, key
    return P


def fe_jack(M, ref, D):
    """Delete-one-field jackknife of each spec's pooled beta: the field's rows get weight 0 and the institution FE
    are re-absorbed on the remaining rows (standardization of the other fields' regressors unchanged)."""
    code = ref["code"]
    return {key: np.array([fe_solve_w(D, key, (code != f).astype(float)) for f in range(len(ref["fields"]))])
            for key in M}


def check_s58(cells):
    """scripts/58 main-spec per-field beta_f on its own sample (fe_build + fe_fit) == data/interim/selectivity_deep.csv,
    and its pooled beta (the +0.0084 of SELECTIVITY_DEEP_RESULT.md)."""
    Mf = s58.fe_build(cells, s58.INST_SLOPES + ["G"], [], "earnings", per_field=True)
    fit = s58.fe_fit(Mf)
    ref = pd.read_csv(S58_CSV).set_index("field")["fe_FE2_beta"]
    diff = (fit["beta_f"] - ref.reindex(fit["beta_f"].index)).abs().max()
    assert diff < 1e-5, diff
    Mp = s58.fe_build(cells, s58.INST_SLOPES + ["G"], [], "earnings", per_field=False)
    bp = fe_solve(Mp["X"], Mp["y"], np.ones(len(Mp["y"])), Mp["names"].index("beta"))
    return dict(maxdiff=float(diff), k=len(fit["beta_f"]), pooled=bp, n_cells=len(Mp["y"]), n_inst=int(Mp["cnt"].size),
                n_fields=len(Mp["fields"]))


def fe_boot(M, ref, workers):
    _D["fe"] = fe_design_dict(M, ref)
    keys = _D["fe"]["keys"]
    out, red = {}, {}
    for kind in ("shared", "ind"):
        res = run_pool(task_fe, [(kind, b0, b1) for b0, b1 in chunks()], workers)
        arr = np.full((B, len(keys)), np.nan)
        for r in res:
            arr[r["b0"]:r["b0"] + r["res"].shape[0]] = r["res"]
        out[kind] = {k: arr[:, i] for i, k in enumerate(keys)}
        red[kind] = int(sum(r["redrawn"] for r in res))
        stage(f"T1b bootstrap {kind} done")
    return out, red


def fe_stat(P, J, BT, a, b=None):
    est = P[a] - (P[b] if b else 0.0)
    jk = J[a] - (J[b] if b is not None else 0.0)
    cb = BT["shared"][a] - (BT["shared"][b] if b else 0.0)
    ib = BT["ind"][a] - (BT["ind"][b] if b else 0.0)
    r = tw(float(est), jack_var(jk), cb, ib)
    return r


# ---------------------------------------------------------------------------------------------------------------
# T3: PSEO fixed cohorts
# ---------------------------------------------------------------------------------------------------------------
def pseo_unitid() -> pd.DataFrame:
    """PSEO institution id (8-digit OPEID) -> UNITID via the 2026 institution file; where one OPEID has several
    UNITIDs, the main campus (MAIN == 1)."""
    ins = s59.institutions("new")
    mr = pd.read_csv(INST_FILE, usecols=["UNITID", "OPEID", "MAIN", "INSTNM"], dtype=str)
    m = ins.merge(mr, left_on="institution", right_on="OPEID", how="left")
    m["_main"] = (m.MAIN == "1").astype(int)
    m = m.sort_values(["institution", "_main", "UNITID"], ascending=[True, False, True]).drop_duplicates("institution")
    return m[["institution", "label", "inst_key", "UNITID", "INSTNM"]]


def build_t3(H, inst26, ar):
    s59.set_release("new")
    er = {h: s52.load_fixed(h, s52.COHORTS) for h in HZ}
    gen_raw = s52._s28.load_generic()
    gen = gen_raw.assign(G=-gen_raw["g_rank"].astype(float))[["inst_key", "G"]]
    panel = s52.fixed_panel(er, ar, gen)
    del er
    # reproduction: the scripts/59 V4.14.1 all-field raw slope
    full = s52.cells_of(panel)
    per_f = {}
    for (f, c), g in full:
        E = g[[f"e_{h}" for h in HZ]].to_numpy(float)
        rho = wpartial(g.P.to_numpy(float), E, [], [], np.ones((1, len(g))))[0]
        per_f.setdefault(f, []).append(rho @ W_SLOPE)
    slope_full = float(np.mean([np.mean(v) for v in per_f.values()]))
    ref = pd.read_csv(S59_CSV, low_memory=False)
    ref = ref[(ref.section == "career_time") & (ref.variant == "new") & (ref.stat == "slope_all")].estimate.iloc[0]
    assert abs(slope_full - ref) < 1e-6, (slope_full, ref)
    # institution -> UNITID
    pu = pseo_unitid()
    multi = pu[pu.inst_key.isin(panel.inst_key)].groupby("inst_key").institution.nunique()
    assert (multi == 1).all(), "a panel institution maps to several PSEO ids"
    k2u = pu.drop_duplicates("inst_key").set_index("inst_key").UNITID
    panel["UNITID"] = panel.inst_key.map(k2u)
    assert panel.UNITID.notna().all()
    # audit trail of the OPEID -> UNITID match (PSEO label next to the Scorecard name), one row per panel institution
    pm = pu[pu.inst_key.isin(panel.inst_key)].drop_duplicates("inst_key").sort_values("inst_key")
    for r in pm.itertuples():
        rec("t3_map", "pseo_institution_to_unitid", np.nan, sample=r.institution, field=r.inst_key,
            note=f"PSEO: {r.label} -> UNITID {r.UNITID}: {r.INSTNM}")
    i26 = inst26.set_index("UNITID")
    panel["SAT_26"] = panel.UNITID.map(i26.SAT_AVG)
    panel["ADM_26"] = panel.UNITID.map(i26.ADM_RATE)
    panel["CTL_26"] = panel.UNITID.map(i26.CONTROL)
    panel["ST_EARN"] = panel.UNITID.map(i26.ST_EARN)
    eras = {}
    for c in T3_COH4:
        e = era_pseo(c)
        eras[c] = e
        T = era_table(H, e["falls"], e["pell"], e["ctl"]).set_index("UNITID")
        msk = panel.grad_cohort == c
        panel.loc[msk, "SAT_E"] = panel.loc[msk, "UNITID"].map(T.SAT_AVG)
        panel.loc[msk, "ADM_E"] = panel.loc[msk, "UNITID"].map(T.ADM_RATE)
        panel.loc[msk, "CTL_E"] = panel.loc[msk, "UNITID"].map(T.CONTROL)
        panel.loc[msk, "nfall_E"] = panel.loc[msk, "UNITID"].map(T.n_SAT_AVG)
    # PSEO bachelor's cohorts are 3-year graduation windows: grad_cohort_years on the rows of the four cohorts
    cy = set()
    for ch in pd.read_csv(s59.NEW_EARN, usecols=["degree_level", "grad_cohort", "grad_cohort_years"], dtype=str,
                          chunksize=200_000):
        ch = ch[(ch.degree_level == s52.PSEO_DEGREE_LEVEL["undergrad"]) & ch.grad_cohort.isin(T3_COH4)]
        cy |= set(ch.grad_cohort_years.unique())
    assert cy == {"3"}, cy
    info = dict(cohort_years=3, panel_rows=len(panel), panel_insts=panel.inst_key.nunique(), full_cells=len(full),
                full_fields=len(per_f), slope_full=slope_full, slope_ref=float(ref))
    return panel, eras, info


def t3_cells(panel):
    need = ["SAT_26", "ADM_26", "SAT_E", "ADM_E"]
    p = panel.dropna(subset=need)
    cells = s52.cells_of(p)
    univ = sorted(set().union(*[set(g.inst_key) for _, g in cells]))
    iid = {k: i for i, k in enumerate(univ)}
    fields = sorted({f for (f, _), _ in cells})
    out, point, dfs = [], {}, {}
    for ci_, ((f, c), g) in enumerate(cells):
        cd = dict(field=f, cohort=c, n=len(g), P=g.P.to_numpy(float), E=g[[f"e_{h}" for h in HZ]].to_numpy(float),
                  ST_EARN=g.ST_EARN.to_numpy(float), inst_idx=np.array([iid[k] for k in g.inst_key]),
                  SAT_26=g.SAT_26.to_numpy(float), ADM_26=g.ADM_26.to_numpy(float),
                  SAT_E=g.SAT_E.to_numpy(float), ADM_E=g.ADM_E.to_numpy(float),
                  CTL_26=g.CTL_26.fillna(-1).to_numpy(int), CTL_E=g.CTL_E.fillna(-1).to_numpy(int))
        sec_ok = bool(g[["ST_EARN", "CTL_26", "CTL_E"]].notna().all(axis=None))
        keys = [("raw", "-")]
        for spec in ("sel", "sec"):
            for v in ("26", "E"):
                conts, cats = SPEC3[spec]
                if spec == "sec" and not sec_ok:
                    continue
                vals = []
                for k in range(3):
                    r_, df = s55.partial_rank(cd["P"], cd["E"][:, k],
                                              [cd[f"{x}_{v}" if x != "ST_EARN" else x] for x in conts],
                                              [cd[f"{x}_{v}"] for x in cats])
                    vals.append(r_)
                dfs[(ci_, spec, v)] = df
                if df >= DFMIN and np.all(np.isfinite(vals)):
                    keys.append((spec, v))
                    point[(ci_, spec, v)] = np.array(vals)
        raw = np.array([s55.spear(cd["P"], cd["E"][:, k]) for k in range(3)])
        point[(ci_, "raw", "-")] = raw
        # engine check at unit weights
        for (spec, v) in keys:
            conts, cats = SPEC3[spec]
            vv = "26" if v == "-" else v
            w = wpartial(cd["P"], cd["E"], [cd[f"{x}_{vv}" if x != "ST_EARN" else x] for x in conts],
                         [cd[f"{x}_{vv}"] for x in cats], np.ones((1, cd["n"])))[0]
            assert np.max(np.abs(w - point[(ci_, spec, v)])) < 1e-9, (f, c, spec, v)
        cd["keys"] = keys
        # bootstrap groups (a replicate is redrawn only for non-finite statistics of its own group): p3 = primary
        # (cohorts 2004/2007/2010), p4 = + cohort 2001 (sensitivity), sec = SAT + ADM + control + state level
        both = lambda sp: (sp, "26") in keys and (sp, "E") in keys
        cd["gkeys"] = {"p3": [("raw", "-"), ("sel", "26"), ("sel", "E")] if c in T3_COH and both("sel") else [],
                       "p4": [("raw", "-"), ("sel", "26"), ("sel", "E")] if c in T3_COH4 and both("sel") else [],
                       "sec": [("raw", "-"), ("sec", "26"), ("sec", "E")] if c in T3_COH and both("sec") else []}
        out.append(cd)
    return out, fields, univ, point, dfs


T3_GROUPS = ("p3", "p4", "sec")


def t3_boot(cells, fields, univ, workers):
    cin = {g: [i for i, cd in enumerate(cells) if cd["gkeys"][g]] for g in T3_GROUPS}
    cof = {g: {j: [i for i in cin[g] if cells[i]["field"] == f] for j, f in enumerate(fields)} for g in T3_GROUPS}
    _D["t3"] = dict(cells=cells, NI=len(univ), cells_in=cin, cells_of_field=cof)
    A, red = {}, {}
    for g in T3_GROUPS:
        out = {}
        out["shared"] = run_pool(task_t3, [("shared", g, None, b0, b1) for b0, b1 in chunks()], workers)
        out["ind"] = run_pool(task_t3, [("ind", g, j, b0, b1) for j in range(len(fields)) if cof[g][j]
                                        for b0, b1 in chunks()], workers)
        A[g] = {}
        for kind in ("shared", "ind"):
            red[f"{kind}_{g}"] = int(sum(r["redrawn"] for r in out[kind]))
            arr = {i: np.full((B, len(cells[i]["gkeys"][g]), 3), np.nan) for i in cin[g]}
            for r in out[kind]:
                for i, v in r["res"].items():
                    if v is not None:
                        arr[i][r["b0"]:r["b0"] + v.shape[0]] = v
            assert all(np.isfinite(x).all() for x in arr.values()), (g, kind)
            A[g][kind] = arr
        stage(f"T3 bootstrap {g} done")
    return A, red


def t3_values(cells, point, A, grp, spec, ver, horizon=None):
    """Per-field value (mean over the field's cells in bootstrap group grp) of the slope (or of the coupling at one
    horizon) of (spec, ver): point, shared, independent replicates."""
    per = {}
    for i, cd in enumerate(cells):
        if not cd["gkeys"][grp]:
            continue
        ki = cd["gkeys"][grp].index((spec, ver))
        pv = point[(i, spec, ver)]
        wv = W_SLOPE if horizon is None else np.eye(3)[horizon]
        per.setdefault(cd["field"], []).append((pv @ wv, A[grp]["shared"][i][:, ki, :] @ wv,
                                                A[grp]["ind"][i][:, ki, :] @ wv))
    fl = sorted(per)
    pt = np.array([np.mean([x[0] for x in per[f]]) for f in fl])
    cb = np.array([np.mean([x[1] for x in per[f]], axis=0) for f in fl])
    ib = np.array([np.mean([x[2] for x in per[f]], axis=0) for f in fl])
    ncell = sum(len(per[f]) for f in fl)
    return fl, pt, cb, ib, ncell


def t3_stat(cells, point, A, grp, a, b, horizon=None):
    fl, pa, ca, ia, nc = t3_values(cells, point, A, grp, *a, horizon)
    if b is not None:
        fl2, pb, cb_, ib_, _ = t3_values(cells, point, A, grp, *b, horizon)
        assert fl2 == fl
        pa, ca, ia = pa - pb, ca - cb_, ia - ib_
    r = tw(float(pa.mean()), float(pa.var(ddof=1) / len(pa)), ca.mean(0), ia.mean(0))
    r.update(k=len(fl), ncell=nc, fields=fl)
    return r


# ---------------------------------------------------------------------------------------------------------------
# T2: stability
# ---------------------------------------------------------------------------------------------------------------
def t2(cells, inst26, H, panel):
    out = {}
    u = cells.drop_duplicates("UNITID")
    for v, tag in (("E", "E"), ("W", "W")):
        for c in ("SAT_AVG", "ADM_RATE", "PCTPELL"):
            x, y = u[f"{c}_{v}"].to_numpy(float), u[f"{c}_26"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            est, lo, hi = boot_spearman(x[ok], y[ok], f"t2_{c}_{v}")
            out[("fos", c, v)] = dict(est=est, lo=lo, hi=hi, n=int(ok.sum()))
    # coverage of SAT_AVG among the Scorecard analysis institutions and programs
    cov = dict(insts=len(u), sat26=int(u.SAT_AVG_26.notna().sum()), satE=int(u.SAT_AVG_E.notna().sum()),
               both=int((u.SAT_AVG_26.notna() & u.SAT_AVG_E.notna()).sum()),
               onlyE=int((u.SAT_AVG_E.notna() & u.SAT_AVG_26.isna()).sum()),
               only26=int((u.SAT_AVG_26.notna() & u.SAT_AVG_E.isna()).sum()),
               ok26=int(u.ok_26.sum()), okE=int(u.ok_E.sum()), okC=int(u.ok_C.sum()),
               cells=len(cells), cells26=int(cells.ok_26.sum()), cellsE=int(cells.ok_E.sum()),
               cellsC=int(cells.ok_C.sum()))
    # mean level change (exploratory, descriptive)
    both = u[u.SAT_AVG_26.notna() & u.SAT_AVG_E.notna()]
    cov["sat_mean26"], cov["sat_meanE"] = float(both.SAT_AVG_26.mean()), float(both.SAT_AVG_E.mean())
    b2 = u[u.ADM_RATE_26.notna() & u.ADM_RATE_E.notna()]
    cov["adm_mean26"], cov["adm_meanE"] = float(b2.ADM_RATE_26.mean()), float(b2.ADM_RATE_E.mean())
    # exploratory: year-by-year Spearman of X(fall y) with X(fall 2024), X = SAT_AVG or ADM_RATE, on the Scorecard
    # analysis institutions ("fos") and on the PSEO panel institutions ("pseo"); institution bootstrap percentile CIs
    curve = []
    id_sets = {"fos": u.UNITID.to_numpy(str), "pseo": np.array(sorted(panel.UNITID.unique()), dtype=str)}
    for iset, var in (("fos", "SAT_AVG"), ("fos", "ADM_RATE"), ("pseo", "SAT_AVG"), ("pseo", "ADM_RATE")):
        ids = id_sets[iset]
        y24 = H[2024][var].reindex(ids).to_numpy(float)
        for y in range(2001, 2024):
            x = H[y][var].reindex(ids).to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y24)
            est, lo, hi = boot_spearman(x[ok], y24[ok], f"t2curve_{iset}_{var}_{y}") if ok.sum() > 5 else (np.nan,) * 3
            curve.append(dict(iset=iset, var=var, y=y, n=int(ok.sum()), est=est, lo=lo, hi=hi))
    # exploratory: did admission rates diverge by selectivity? Spearman(ADM_RATE fall 2024 - ADM_RATE fall y0,
    # SAT_AVG fall y0): negative = institutions with higher SAT in y0 lowered their admission rate more
    div = {}
    for iset, y0 in (("pseo", 2003), ("fos", 2014)):
        ids = id_sets[iset]
        d_adm = H[2024]["ADM_RATE"].reindex(ids).to_numpy(float) - H[y0]["ADM_RATE"].reindex(ids).to_numpy(float)
        sat0 = H[y0]["SAT_AVG"].reindex(ids).to_numpy(float)
        ok = np.isfinite(d_adm) & np.isfinite(sat0)
        est, lo, hi = boot_spearman(d_adm[ok], sat0[ok], f"t2div_{iset}_{y0}")
        div[(iset, y0)] = dict(est=est, lo=lo, hi=hi, n=int(ok.sum()),
                               mean_d=float(np.mean(d_adm[ok])))
    # the 2026 file's SAT_AVG / ADM_RATE / PCTPELL are the MERGED2024_25 values
    m = inst26.set_index("UNITID")
    agree = {}
    for c in ("SAT_AVG", "ADM_RATE", "PCTPELL"):
        a = m[c].reindex(H[2024].index)
        bb = H[2024][c]
        both_ = a.notna() & bb.notna()
        agree[c] = (int(both_.sum()), int((np.abs(a[both_] - bb[both_]) < 1e-9).sum()),
                    int((a.notna() != bb.notna()).sum()))
    # PSEO eras (T3 institutions: SAT and ADM under both versions)
    pse = {}
    for c in T3_COH4:
        g = panel[(panel.grad_cohort == c)].drop_duplicates("inst_key")
        for v26, vE, lab in (("SAT_26", "SAT_E", "SAT_AVG"), ("ADM_26", "ADM_E", "ADM_RATE")):
            x, y = g[vE].to_numpy(float), g[v26].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            est, lo, hi = boot_spearman(x[ok], y[ok], f"t2_pseo_{lab}_{c}")
            pse[(c, lab)] = dict(est=est, lo=lo, hi=hi, n=int(ok.sum()), n_panel=len(g))
    return out, cov, curve, agree, pse, div


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------
def main():
    workers = n_workers()
    stage(f"pre-specification md5 {prespec_md5()}")
    prov = provenance()
    stage("provenance checked")
    fos_years = sorted(set(ERA_FOS_W["falls"]) | set(ERA_FOS["falls"]))
    pell_years = sorted(set(ERA_FOS_W["pell"]))
    t3_falls = sorted({y for c in T3_COH4 for y in era_pseo(c)["falls"]})
    ctl_years = sorted(set(ERA_FOS["ctl"]) | set(ERA_FOS_W["ctl"]) | {y for c in T3_COH4 for y in era_pseo(c)["ctl"]})
    dic = dictionary_checks(sorted(set(fos_years) | set(t3_falls) | {2024}), pell_years, ctl_years)
    docs = doc_checks()
    stage(f"data dictionary ({dic['release']}) and documentation pages checked")
    need_files = (set(fos_years) | {a + 1 for a in pell_years} | set(ERA_FOS["ctl"]) | set(t3_falls)
                  | set(ctl_years) | set(range(2000, 2025)))
    H = read_hist(need_files)
    # fall 2000: the file exists but carries no SAT_AVG / ADM_RATE (the dictionary's MERGED_2000-01 entries are empty)
    dic["m2000"] = dict(rows=len(H[2000]), sat=int(H[2000].SAT_AVG.notna().sum()),
                        adm=int(H[2000].ADM_RATE.notna().sum()))
    assert dic["m2000"]["sat"] == 0 and dic["m2000"]["adm"] == 0 and dic["m2000"]["rows"] > 0
    dic["n_hist_files"] = len(H)
    stage(f"read {len(H)} historical files")
    inst26 = s55.load_institutions()

    # ---------------- T1 ----------------
    ar, cells, t1info = build_t1(H, inst26)
    universe, fields, FD, point, dfs, nn, keys_all, keys_strict = t1_point(cells)
    fs, _ok = t1_fieldsets(fields, point, dfs, nn)
    eng = check_engine(FD, fields, point, keys_all + keys_strict)
    s55chk = check_s55(fields, point, fs)
    stage(f"T1 point estimates; engine max|diff| {eng:.1e}; scripts/55 broad reproduced ({s55chk[1]} fields, "
          f"max|diff| {s55chk[0]:.1e})")
    s58chk = check_s58(cells)
    stage(f"scripts/58 main-spec beta_f reproduced (max|diff| {s58chk['maxdiff']:.1e}); pooled {s58chk['pooled']:+.4f}")
    M, ref = build_fe(cells)
    FD_fe = fe_design_dict(M, ref)
    FP = fe_point(M, FD_fe)
    FJ = fe_jack(M, ref, FD_fe)
    stage(f"T1b designs: {len(ref['y'])} programs, {int(ref['cnt'].size)} institutions, {len(ref['fields'])} fields")

    # ---------------- T3 data ----------------
    panel, eras, t3info = build_t3(H, inst26, ar)
    stage(f"T3 panel: {t3info['panel_rows']} rows, {t3info['panel_insts']} institutions; scripts/59 slope reproduced "
          f"({t3info['slope_full']:+.6f})")
    cells3, fields3, univ3, point3, dfs3 = t3_cells(panel)
    stage(f"T3 cells {len(cells3)}, fields {len(fields3)}, institutions {len(univ3)}")
    t2res, t2cov, t2curve, t2agree, t2pseo, t2div = t2(cells, inst26, H, panel)
    del H
    gc.collect()

    # ---------------- bootstrap ----------------
    G1 = t1_groups(fs, keys_all, keys_strict)
    A1, red1 = t1_boot(FD, fields, universe, G1, point, workers)
    BT, redfe = fe_boot(M, ref, workers)
    A3, red3 = t3_boot(cells3, fields3, univ3, workers)

    # ---------------- T1a statistics ----------------
    L = fs["ladder_C"]
    T1 = {}
    T1[("raw", "-")] = t1_stat(A1, point, L, ("C", "raw", "-"))
    for s in ("pcs", "pcs_adm", "broad", "sel"):
        for v in VERS:
            T1[(s, v)] = t1_stat(A1, point, L, ("C", s, v))
        for v in ("E", "W"):
            T1[(s, f"{v}-26")] = t1_stat(A1, point, L, ("C", s, v), ("C", s, "26"))
        for v in VERS:
            T1[(s, f"raw-{v}")] = t1_stat(A1, point, L, ("C", "raw", "-"), ("C", s, v))
    Ls = fs["strict_C"]
    for v in ("26", "E"):
        T1[("strict", v)] = t1_stat(A1, point, Ls, ("C", "strict", v), grp="strict")
    T1[("strict", "E-26")] = t1_stat(A1, point, Ls, ("C", "strict", "E"), ("C", "strict", "26"), grp="strict")
    T1[("raw_strict", "-")] = t1_stat(A1, point, Ls, ("C", "raw", "-"), grp="strict")
    # exploratory: share of the raw level removed by the broad controls, 26 vs E (ratio; V_field by jackknife)
    shares = {v: t1_share(A1, point, L, v) for v in VERS}
    shares["E-26"] = t1_share(A1, point, L, "E", "26")
    shares["W-26"] = t1_share(A1, point, L, "W", "26")
    # secondary: era-complete sample (E controls) vs scripts/55's own sample (26 controls)
    SEC = {}
    for s in LADDER:
        vE, v26 = ("-", "-") if s == "raw" else ("E", "26")
        fa, fb = fs[f"SE_{s}"], fs[f"S26_{s}"]
        SEC[(s, "SE")] = t1_stat(A1, point, fa, ("SE", s, vE), grp="sec")
        SEC[(s, "S26")] = t1_stat(A1, point, fb, ("S26", s, v26), grp="sec")
        SEC[(s, "SE-S26")] = t1_stat_x(A1, point, fa, ("SE", s, vE), fb, ("S26", s, v26))
    n_rows_C = {f: nn[(f, "C")] for f in L}
    # programs behind each statistic (sum over its fields of the sample's programs)
    nsum = lambda fl, smp: int(sum(nn[(f, smp)] for f in fl))
    for (s, v), r in T1.items():
        r["n"] = nsum(Ls, "C") if s in ("strict", "raw_strict") else nsum(L, "C")
    for r in shares.values():
        r["n"] = nsum(L, "C")
    for s in LADDER:
        SEC[(s, "SE")]["n"] = nsum(fs[f"SE_{s}"], "SE")
        SEC[(s, "S26")]["n"] = nsum(fs[f"S26_{s}"], "S26")
        SEC[(s, "SE-S26")]["n"] = np.nan
        SEC[(s, "SE-S26")]["n_note"] = f"programs {SEC[(s, 'SE')]['n']} vs {SEC[(s, 'S26')]['n']}"

    # ---------------- T1b statistics ----------------
    FE = {}
    for key in M:
        FE[key] = fe_stat(FP, FJ, BT, key)
    for spec in ("b", "main"):
        for v in ("E", "W"):
            FE[(spec, f"{v}-26")] = fe_stat(FP, FJ, BT, (spec, v), (spec, "26"))
    FE[("a-main", "26")] = fe_stat(FP, FJ, BT, ("a", "-"), ("main", "26"))
    FE[("a-main", "E")] = fe_stat(FP, FJ, BT, ("a", "-"), ("main", "E"))
    fe_n = dict(cells=len(ref["y"]), insts=int(ref["cnt"].size), fields=len(ref["fields"]))
    for r in FE.values():
        r["n"], r["k"] = fe_n["cells"], fe_n["fields"]

    # ---------------- T3 statistics ----------------
    T3 = {}
    for cset, g in (("P3", "p3"), ("P4", "p4")):
        T3[(cset, "raw")] = t3_stat(cells3, point3, A3, g, ("raw", "-"), None)
        T3[(cset, "sel26")] = t3_stat(cells3, point3, A3, g, ("sel", "26"), None)
        T3[(cset, "selE")] = t3_stat(cells3, point3, A3, g, ("sel", "E"), None)
        T3[(cset, "E-26")] = t3_stat(cells3, point3, A3, g, ("sel", "E"), ("sel", "26"))
        T3[(cset, "E-raw")] = t3_stat(cells3, point3, A3, g, ("sel", "E"), ("raw", "-"))
        T3[(cset, "26-raw")] = t3_stat(cells3, point3, A3, g, ("sel", "26"), ("raw", "-"))
        for hi, h in enumerate(HZ):
            for nm, a in (("raw", ("raw", "-")), ("sel26", ("sel", "26")), ("selE", ("sel", "E"))):
                T3[(cset, f"{nm}_{h}")] = t3_stat(cells3, point3, A3, g, a, None, horizon=hi)
    T3[("P3", "sec_raw")] = t3_stat(cells3, point3, A3, "sec", ("raw", "-"), None)
    T3[("P3", "sec26")] = t3_stat(cells3, point3, A3, "sec", ("sec", "26"), None)
    T3[("P3", "secE")] = t3_stat(cells3, point3, A3, "sec", ("sec", "E"), None)
    T3[("P3", "secE-26")] = t3_stat(cells3, point3, A3, "sec", ("sec", "E"), ("sec", "26"))
    for r in T3.values():
        r["n"] = r["ncell"]
    t3n = {}
    for cset, g, cl in (("P3", "p3", T3_COH), ("P4", "p4", T3_COH4), ("SEC", "sec", T3_COH)):
        cc = [cd for cd in cells3 if cd["gkeys"][g]]
        insts = sorted(set().union(*[set(univ3[i] for i in cd["inst_idx"]) for cd in cc]))
        t3n[cset] = dict(cells=len(cc), insts=len(insts), n_min=min(cd["n"] for cd in cc), n_max=max(cd["n"] for cd in cc),
                         by_coh={c: sum(cd["cohort"] == c for cd in cc) for c in cl})
    # panel coverage by cohort: institutions in the balanced panel vs with both versions' SAT/ADM
    covp = {}
    for c in T3_COH4:
        g = panel[panel.grad_cohort == c].drop_duplicates("inst_key")
        covp[c] = dict(panel=len(g), sat26=int(g[["SAT_26", "ADM_26"]].notna().all(axis=1).sum()),
                       satE=int(g[["SAT_E", "ADM_E"]].notna().all(axis=1).sum()),
                       both=int(g[["SAT_26", "ADM_26", "SAT_E", "ADM_E"]].notna().all(axis=1).sum()),
                       nfall=g.loc[g.SAT_E.notna(), "nfall_E"].value_counts().sort_index().to_dict())

    ctx = dict(prespec=prespec_md5(), prov=prov, dic=dic, docs=docs, t1info=t1info, fs=fs, T1=T1, shares=shares, SEC=SEC, red1=red1, s55chk=s55chk,
               s58chk=s58chk, FE=FE, FP=FP, fe_n=fe_n, redfe=redfe, T3=T3, t3n=t3n, t3info=t3info, covp=covp,
               red3=red3, eras=eras, t2res=t2res, t2cov=t2cov, t2curve=t2curve, t2agree=t2agree, t2pseo=t2pseo, t2div=t2div,
               universe=universe, fields=fields, point=point, nC=n_rows_C, cells3=cells3, point3=point3,
               workers=workers, n_univ3=len(univ3), fields3=fields3)
    record(ctx)
    md = render(ctx)          # records the numbers it derives (CSV section 'derived') before the CSV is written
    write_csv()
    OUT_MD.write_text(md)
    print(f"[md] {OUT_MD}")
    stage("done")


# ---------------------------------------------------------------------------------------------------------------
# records, csv, markdown
# ---------------------------------------------------------------------------------------------------------------
EXPL = "exploratory (not pre-specified)"
FE_LEV_KEYS = [("nofe", "-"), ("a", "-")] + [(sp, v) for sp in ("b", "main") for v in ("26", "E", "W")]
FE_LAB = {("nofe", "-"): "field FE only", ("a", "-"): "(a)",
          **{(sp, v): f"{'(b)' if sp == 'b' else 'main'} {({'26': '2026', 'E': 'era-matched', 'W': 'wide window'})[v]}"
             for sp in ("b", "main") for v in ("26", "E", "W")}}


def st_t1(s, v):
    if (s, v) == ("broad", "E-26"):
        return "primary"
    if v.startswith("raw-"):
        return EXPL
    if v in ("W", "W-26"):
        return "pre-specified sensitivity"
    return "pre-specified"


def st_fe(spec, v):
    if (spec, v) == ("main", "E-26"):
        return "primary"
    if spec == "a-main":
        return EXPL
    if v in ("W", "W-26"):
        return "pre-specified sensitivity"
    return "pre-specified"


def st_t3(cset, nm):
    if (cset, nm) == ("P3", "E-26"):
        return "primary"
    if nm == "26-raw" or "_y" in nm:
        return EXPL
    if cset == "P4" or nm.startswith("sec"):
        return "pre-specified sensitivity"
    return "pre-specified"


def short(lab: str) -> str:
    return lab.split(" within")[0]


def reading_grid(ctx) -> dict:
    """Readings of every pre-specified E - 26 / W - 26 difference: at the pre-set numeric margin (the pre-specified
    reading) and at one third of the 2026-control estimate on the same programs / cells, each under the two-way and
    under the conservative (max) variance; plus the Monte Carlo margin flags."""
    T1, FE, T3 = ctx["T1"], ctx["FE"], ctx["T3"]
    G = {}

    def add(key, r, delta, ref26):
        alt = abs(ref26) / 3
        lab = sig2(alt)
        G[key] = dict(r=r, delta=delta, alt=alt, alt_lab=lab, ref26=ref26,
                      num=reading(r, delta), num_c=reading(r, delta, cons=True),
                      alt_r=reading(r, alt, lab=lab), alt_c=reading(r, alt, cons=True, lab=lab),
                      flag=near_margin(r, delta), flag_alt=near_margin(r, alt, lab))
    for s in ("pcs", "pcs_adm", "broad", "sel"):
        for v in ("E", "W"):
            add(("T1", s, f"{v}-26"), T1[(s, f"{v}-26")], DELTA["t1a"], T1[(s, "26")]["est"])
    add(("T1", "strict", "E-26"), T1[("strict", "E-26")], DELTA["t1a"], T1[("strict", "26")]["est"])
    for sp in ("b", "main"):
        for v in ("E", "W"):
            add(("FE", sp, f"{v}-26"), FE[(sp, f"{v}-26")], DELTA["t1b"], FE[(sp, "26")]["est"])
    add(("T3", "P3", "E-26"), T3[("P3", "E-26")], DELTA["t3"], T3[("P3", "sel26")]["est"])
    add(("T3", "P4", "E-26"), T3[("P4", "E-26")], DELTA["t3"], T3[("P4", "sel26")]["est"])
    add(("T3", "P3", "secE-26"), T3[("P3", "secE-26")], DELTA["t3"], T3[("P3", "sec26")]["est"])
    return G


def reading_str(g, md: bool = False, flags: bool = True) -> str:
    """Pre-set margin reading first (the pre-specified one), then the reading at one third of the 2026 estimate."""
    s = f"±{g['delta']:g}: {short(g['num'])}"
    if g["num_c"] != g["num"]:
        s += f" (max-var: {short(g['num_c'])})"
    if g["flag"] and flags:
        s += " [at the margin within Monte Carlo error]" if md else f" [at the margin within Monte Carlo error: {g['flag']}]"
    s += f"; ±{g['alt_lab']}{' (1/3 of the 2026 estimate)' if not md else ''}: {short(g['alt_r'])}"
    if g["alt_c"] != g["alt_r"]:
        s += f" (max-var: {short(g['alt_c'])})"
    if g["flag_alt"] and flags:
        s += " [at the margin within Monte Carlo error]" if md else f" [at the margin within Monte Carlo error: {g['flag_alt']}]"
    return s


def _rec_tw(section, stat, r, spec="", version="", sample="", status="", reading_="", note=""):
    k = r.get("k", np.nan)
    rec(section, stat, r["est"], spec, version, sample, lo=r["tlo"], hi=r["thi"], se=r["tse"], p=r["p"],
        k=k if not isinstance(k, str) else np.nan, n=r.get("n", np.nan), status=status, se_c=r["cse"], lo_c=r["clo"],
        hi_c=r["chi"], reading=reading_,
        note=(f"two-way SE={r['tse']:.4g}; V_field={r['tvf']:.3g}, V_inst={r['tvs']:.3g}, V_fxi={r['tvi']:.3g}"
              + ("; V<=0, larger one-way variance used" if r["tfallback"] else "")
              + f"; MC SE of 95% CI endpoints={r['tmcse']:.2g}, of 90% CI endpoints={r['mc90']:.2g}; 90% CI "
                f"[{r['lo90']:.4g}, {r['hi90']:.4g}]; MDE80={r['mde']:.4g}; max-var SE={r['cse']:.4g} "
                f"(binding: {r['cwhich']}), 90% CI [{r['clo90']:.4g}, {r['chi90']:.4g}], p={r['cp']:.3g}"
              + (f"; k={k}" if isinstance(k, str) else "") + (f"; {r['n_note']}" if r.get("n_note") else "")
              + (f"; {note}" if note else "")))


T3_SAMPLE = {"P3": "cohorts 2004/2007/2010", "P4": "cohorts 2001-2010 (2001: fall-2001 stand-in)",
             "SEC": "cohorts 2004/2007/2010, cells with control and state level"}


def record(ctx):
    T1, FE, T3 = ctx["T1"], ctx["FE"], ctx["T3"]
    RG = ctx["RG"] = reading_grid(ctx)
    for (s, v), r in T1.items():
        g = RG.get(("T1", s, v))
        _rec_tw("T1a", "mean_rho", r, s, v,
                "common programs, strict field set" if s in ("strict", "raw_strict") else "common programs",
                status=st_t1(s, v), reading_=reading_str(g) if g else "")
    for (s, smp), r in ctx["SEC"].items():
        _rec_tw("T1a_secondary", "mean_rho", r, s, smp, {"SE": "era-complete programs (E controls)",
                                                         "S26": "scripts/55 SAT sample (26 controls)",
                                                         "SE-S26": "era-complete minus scripts/55 sample"}[smp],
                status="pre-specified sensitivity")
    for v, r in ctx["shares"].items():
        _rec_tw("T1a_exploratory", "share_of_raw_removed_by_broad", r, "broad", v, "common programs",
                status="exploratory")
    for f in ctx["fs"]["ladder_C"]:
        for s in ("raw", "broad"):
            for v in (["-"] if s == "raw" else VERS):
                rec("T1a_field", "rho", ctx["point"][(f, ("C", s, v))], s, v, "common programs", field=f,
                    n=ctx["nC"][f], status="exploratory (per-field values)")
        rec("T1a_field", "rho_E_minus_26", ctx["point"][(f, ("C", "broad", "E"))] - ctx["point"][(f, ("C", "broad", "26"))],
            "broad", "E-26", "common programs", field=f, n=ctx["nC"][f], status="exploratory (per-field values)")
    for key, r in FE.items():
        spec, v = key
        g = RG.get(("FE", spec, v))
        _rec_tw("T1b", "pooled_beta", r, spec, v, "common programs (FE)", status=st_fe(spec, v),
                reading_=reading_str(g) if g else "")
    for key, r in T3.items():
        cset, nm = key
        g = RG.get(("T3", cset, nm))
        _rec_tw("T3", nm, r, "", "", T3_SAMPLE["SEC" if nm.startswith("sec") else cset], status=st_t3(cset, nm),
                reading_=reading_str(g) if g else "")
    for key, x in ctx["t2res"].items():
        rec("T2", "spearman_across_institutions", x["est"], key[1], key[2], "Scorecard analysis institutions",
            lo=x["lo"], hi=x["hi"], n=x["n"], note="institution bootstrap percentile CI",
            status="primary" if (key[1], key[2]) == ("SAT_AVG", "E") else "pre-specified")
    for key, x in ctx["t2pseo"].items():
        rec("T2", "spearman_across_institutions", x["est"], key[1], f"PSEO cohort {key[0]} entry window",
            "T3 panel institutions", lo=x["lo"], hi=x["hi"], n=x["n"],
            note=f"institution bootstrap percentile CI; panel institutions {x['n_panel']}",
            status="pre-specified sensitivity" if key[0] == "2001" else "pre-specified")
    for c in ctx["t2curve"]:
        rec("T2_exploratory", "spearman_fall_y_vs_fall2024", c["est"], c["var"], str(c["y"]),
            {"fos": "Scorecard analysis institutions", "pseo": "PSEO panel institutions"}[c["iset"]], lo=c["lo"],
            hi=c["hi"], n=c["n"], note="institution bootstrap percentile CI", status="exploratory")
    for (iset, y0), x in ctx["t2div"].items():
        rec("T2_exploratory", "spearman_adm_change_to_2024_vs_sat", x["est"], "ADM_RATE", str(y0),
            {"fos": "Scorecard analysis institutions", "pseo": "PSEO panel institutions"}[iset], lo=x["lo"],
            hi=x["hi"], n=x["n"], note="institution bootstrap percentile CI", status="exploratory")
        rec("T2_exploratory", "mean_adm_change_to_2024", x["mean_d"], "ADM_RATE", str(y0),
            {"fos": "Scorecard analysis institutions", "pseo": "PSEO panel institutions"}[iset], n=x["n"],
            status="exploratory")
    # ---- coverage and sample counts
    cv = lambda k_, v_, note="", sample="": rec("coverage", k_, v_, sample=sample, note=note, status="coverage")
    for k_, v_ in ctx["t2cov"].items():
        cv(k_, v_)
    cv("t1a_ladder_fields", len(ctx["fs"]["ladder_C"]))
    cv("t1a_ladder_programs", ctx["T1"][("raw", "-")]["n"], note="common programs of the T1a ladder field set")
    cv("t1a_strict_fields", len(ctx["fs"]["strict_C"]))
    cv("t1a_strict_programs", ctx["T1"][("strict", "E")]["n"])
    cv("common_programs_state_differs_2026_vs_era_file", ctx["t1info"]["stabbr_differs_cells"],
       note="STABBR is backfilled with the most recent value in every MERGED file (dictionary), so 0 by construction")
    cv("common_institutions_control_differs_2026_vs_era_file", ctx["t1info"]["control_differs_insts"],
       note="CONTROL is dated in the MERGED files (dictionary)")
    for k_ in ("cells", "insts", "fields"):
        cv(f"t1b_{k_}", ctx["fe_n"][k_])
    for cset, x in ctx["t3n"].items():
        smp = T3_SAMPLE[cset]
        cv("t3_cells", x["cells"], sample=smp)
        cv("t3_institutions", x["insts"], sample=smp)
        cv("t3_cell_n_min", x["n_min"], sample=smp)
        cv("t3_cell_n_max", x["n_max"], sample=smp)
        for c, n_ in x["by_coh"].items():
            cv(f"t3_cells_cohort_{c}", n_, sample=smp)
    for c, x in ctx["covp"].items():
        for k_ in ("panel", "sat26", "satE", "both"):
            cv(f"pseo_cohort_{c}_institutions_{k_}", x[k_],
               note={"panel": "balanced panel", "sat26": "SAT_AVG and ADM_RATE, 2026 file",
                     "satE": "SAT_AVG and ADM_RATE, entry window", "both": "both versions"}[k_])
        for nf, n_ in x["nfall"].items():
            cv(f"pseo_cohort_{c}_institutions_with_{int(nf)}_falls_of_SAT", n_)
    ti = ctx["t3info"]
    for k_ in ("panel_rows", "panel_insts", "full_cells", "full_fields", "cohort_years"):
        cv(f"t3_{k_}", ti[k_], note="scripts/59 V4.14.1 balanced fixed-cohort panel (all fields, cohorts 2001-2010)")
    m0 = ctx["dic"]["m2000"]
    cv("merged2000_01_rows", m0["rows"])
    cv("merged2000_01_sat_avg_values", m0["sat"])
    cv("merged2000_01_adm_rate_values", m0["adm"])
    cv("historical_files_read", ctx["dic"]["n_hist_files"])
    # ---- checks
    ck = lambda k_, v_, **kw: rec("check", k_, v_, status="check", **kw)
    for c, (nb, same, mism) in ctx["t2agree"].items():
        ck(f"2026_file_equals_MERGED2024_25_{c}", same, n=nb, note=f"presence differs for {mism} UNITIDs")
    ck("scripts55_broad_max_abs_diff", ctx["s55chk"][0], k=ctx["s55chk"][1])
    ck("scripts58_main_beta_f_max_abs_diff", ctx["s58chk"]["maxdiff"], k=ctx["s58chk"]["k"])
    ck("scripts58_main_pooled_beta", ctx["s58chk"]["pooled"], n=ctx["s58chk"]["n_cells"], k=ctx["s58chk"]["n_fields"])
    ck("scripts59_slope_all_new_reproduced", ti["slope_full"], k=ti["full_fields"], n=ti["full_cells"])
    ck("scripts59_slope_all_new_reference", ti["slope_ref"], note="data/interim/pseo_refresh.csv")
    for k_, v_ in {**{f"t1_{a}": b for a, b in ctx["red1"].items()}, **{f"fe_{a}": b for a, b in ctx["redfe"].items()},
                   **{f"t3_{a}": b for a, b in ctx["red3"].items()}}.items():
        ck(f"redrawn_replicates_{k_}", v_, n=B)
    dic = ctx["dic"]
    ck("dictionary_release", np.nan, note=dic["release"])
    ck("dictionary_fos_EARN_MDN_4YR", np.nan, note=dic["fos_4yr"])
    for v, (a, b) in dic["mr"].items():
        ck(f"dictionary_most_recent_{v}", np.nan, note=f"{a}; drawn from {b}")
    for v in ("SAT_AVG", "ADM_RATE", "PCTPELL"):
        ck(f"dictionary_first_file_{v}", np.nan, note=dic[f"first_{v}"])
    ck("dictionary_ADM_RATE_MERGED_2001-02", np.nan, note=dic["adm_2001"])
    for y, s_ in dic["stabbr_backfill"].items():
        ck("dictionary_STABBR_backfilled", np.nan, version=str(y), note=s_)
    for y, s_ in dic["control_dated"].items():
        ck("dictionary_CONTROL_dated", np.nan, version=str(y), note=s_)
    rec("provenance", "prespecification_md5", np.nan, note=ctx["prespec"], status="provenance")
    rec("provenance", "n_merged_members", ctx["prov"]["n_merged"], status="provenance")
    for f, pdfp, printed, q in ctx["docs"]["quotes"]:
        rec("provenance", "doc_quote", np.nan, spec=f, version=f"p. {printed} / PDF p. {pdfp}", note=q,
            status="provenance")
    for f in DOCS:
        if f"{f}:pages" not in ctx["docs"]:
            continue
        rec("provenance", "doc_version_pages", ctx["docs"][f"{f}:pages"], spec=f, note=ctx["docs"][f"{f}:version"],
            status="provenance")


def write_csv():
    df = pd.DataFrame(ROWS)
    df.loc[df.status == "", "status"] = "provenance"
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"[csv] {OUT_CSV} ({len(df)} rows)")


def f_(x, d=3, sign=True):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def ci_(r, d=3):
    return f"[{f_(r['tlo'], d)}, {f_(r['thi'], d)}]"


def cci_(r, d=3):
    return f"[{f_(r['clo'], d)}, {f_(r['chi'], d)}]"


def p_(p):
    if not np.isfinite(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def pct(x):
    return f"{100 * x:+.0f}%"


STATUS_SHORT = {"primary": "**primary**", "pre-specified": "pre-specified",
                "pre-specified sensitivity": "pre-specified sensitivity", EXPL: "*exploratory (not pre-specified)*",
                "exploratory": "*exploratory*"}


def _row(q, sample, r, d=3, kcol=None, status="", reading_=""):
    k = kcol if kcol is not None else r.get("k", "")
    n = r.get("n", np.nan)
    ncol = str(int(n)) if isinstance(n, (int, np.integer, float)) and np.isfinite(n) else r.get("n_note", "")
    st = STATUS_SHORT.get(status, status)
    return (f"| {q} | {sample} | {f_(r['est'], d)} | {ci_(r, d)} | {cci_(r, d)} | {p_(r['p'])} | "
            f"{f_(r['mde'], d, False)} | {k} | {ncol} | {st}{'; ' + reading_ if reading_ else ''} |")


def render(ctx) -> str:
    T1, FE, T3, SEC, SH, RG = ctx["T1"], ctx["FE"], ctx["T3"], ctx["SEC"], ctx["shares"], ctx["RG"]
    fs, cov, t3n, covp = ctx["fs"], ctx["t2cov"], ctx["t3n"], ctx["covp"]
    L = fs["ladder_C"]
    kL, kS = len(L), len(fs["strict_C"])
    d1, db, d3 = DELTA["t1a"], DELTA["t1b"], DELTA["t3"]
    g1, gb, g3 = RG[("T1", "broad", "E-26")], RG[("FE", "main", "E-26")], RG[("T3", "P3", "E-26")]
    t2 = ctx["t2res"]
    sat = t2[("fos", "SAT_AVG", "E")]
    adm = t2[("fos", "ADM_RATE", "E")]
    pell = t2[("fos", "PCTPELL", "E")]
    ps = ctx["t2pseo"]
    ps_sat = [ps[(c, "SAT_AVG")]["est"] for c in T3_COH]
    ps_adm = [ps[(c, "ADM_RATE")]["est"] for c in T3_COH]
    fe_n = ctx["fe_n"]
    P3n, P4n = t3n["P3"], t3n["P4"]
    dic = ctx["dic"]
    fosdoc = "FieldOfStudyDataDocumentation.pdf"
    insdoc = "InstitutionDataDocumentation.pdf"
    ver_f = ctx["docs"][f"{fosdoc}:version"]
    CV = {(c["iset"], c["var"], c["y"]): c for c in ctx["t2curve"]}
    nprog = T1[("raw", "-")]["n"]

    def pg(f, phrase_start):
        for (ff, pdfp, printed, qq) in ctx["docs"]["quotes"]:
            if ff == f and phrase_start in qq:
                return f"p. {printed} / PDF p. {pdfp}"
        raise KeyError(phrase_start)

    def ci90(r, d=3):
        return f"[{f_(r['lo90'], d)}, {f_(r['hi90'], d)}]"

    def cci90(r, d=3):
        return f"[{f_(r['clo90'], d)}, {f_(r['chi90'], d)}]"

    # ---- primary readings under the four combinations (margin x variance)
    prim = [("T1a", "T1a level", g1), ("T1b", "T1b within-institution slope", gb), ("T3", "T3 career-time slope", g3)]
    eq = lambda lab: lab.startswith("equivalent")
    combos = (("num", "pre-set numeric margins, two-way variance [pre-specified]"),
              ("alt_r", "one third of the 2026 estimate on the same programs / cells, two-way variance"),
              ("num_c", "pre-set numeric margins, conservative variance"),
              ("alt_c", "one third of the 2026 estimate on the same programs / cells, conservative variance"))
    cnt = {c: sum(eq(g[c]) for _, _, g in prim) for c, _ in combos}
    ndf = {c: sum(g[c] == "differs" for _, _, g in prim) for c, _ in combos}
    for c, lab in combos:
        drv("primary_differences_equivalent", cnt[c], note=lab, k=3)
        drv("primary_differences_differ", ndf[c], note=lab, k=3)
    for t, _, g in prim:
        drv("primary_same_cells_margin", g["alt"], sample=t, note=f"one third of |2026-control estimate| {g['ref26']:+.6g}")

    def names(c, want):
        return [lab for _, lab, g in prim if (eq(g[c]) if want == "eq" else short(g[c]) == want)]

    def andj(xs):
        return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1] if xs else "none"

    def cnt_txt(c):
        e, i = names(c, "eq"), names(c, "inconclusive")
        return (f"{cnt[c]} of 3 {'is' if cnt[c] == 1 else 'are'} equivalent ({andj(e) if e else 'none'})"
                + (f" and {andj(i)} {'is' if len(i) == 1 else 'are'} inconclusive" if i else ""))

    # ---- derived numbers of the text
    r3 = T3[("P3", "E-26")]
    s26 = T3[("P3", "sel26")]["est"]
    ratio_t3 = drv("t3_selE_over_raw_slope", T3[("P3", "selE")]["est"] / T3[("P3", "raw")]["est"],
                   sample=T3_SAMPLE["P3"], note="era-matched partial slope / raw slope, same cells")
    rel_pt = drv("t3_E_minus_26_relative_to_sel26", r3["est"] / s26, sample=T3_SAMPLE["P3"],
                 note="(E - 26) / 2026-control partial slope (point estimate of the denominator)")
    rel_lo = drv("t3_E_minus_26_90ci_lower_relative_to_sel26", r3["lo90"] / s26, sample=T3_SAMPLE["P3"])
    rel_hi = drv("t3_E_minus_26_90ci_upper_relative_to_sel26", r3["hi90"] / s26, sample=T3_SAMPLE["P3"])
    delta_share = drv("t3_preset_margin_over_scripts59_raw_slope", d3 / ctx["t3info"]["slope_full"],
                      note="0.010 / scripts/59 all-field raw slope")
    delta_share26 = drv("t3_preset_margin_over_sel26", d3 / s26, sample=T3_SAMPLE["P3"],
                        note="0.010 / 2026-control partial slope on the T3 cells")
    fos_adm_08 = [CV[("fos", "ADM_RATE", y)]["est"] for y in range(2001, 2009)]
    pse_adm_08 = [CV[("pseo", "ADM_RATE", y)]["est"] for y in range(2001, 2009)]
    drv("adm_rate_spearman_vs_2024_falls_2001_2008_min", min(fos_adm_08), sample="Scorecard analysis institutions")
    drv("adm_rate_spearman_vs_2024_falls_2001_2008_max", max(fos_adm_08), sample="Scorecard analysis institutions")
    drv("adm_rate_spearman_vs_2024_falls_2001_2008_min", min(pse_adm_08), sample="PSEO panel institutions")
    drv("adm_rate_spearman_vs_2024_falls_2001_2008_max", max(pse_adm_08), sample="PSEO panel institutions")
    pay_E = drv("t1b_main_E_percent_of_pay_per_sd", 100 * (np.exp(FE[("main", "E")]["est"]) - 1),
                note="100 (exp(beta) - 1)")
    pf_d = sorted(((f, ctx["point"][(f, ("C", "broad", "E"))] - ctx["point"][(f, ("C", "broad", "26"))]) for f in L),
                  key=lambda t: (t[1], t[0]))
    drv("t1a_field_E_minus_26_min", pf_d[0][1], note=pf_d[0][0], k=kL)
    drv("t1a_field_E_minus_26_max", pf_d[-1][1], note=pf_d[-1][0], k=kL)
    n_big = drv("t1a_fields_abs_E_minus_26_above_0.10", sum(abs(v) > 0.10 for _, v in pf_d), k=kL)
    sd_pf = drv("t1a_field_E_minus_26_sd", float(np.std([v for _, v in pf_d], ddof=1)), k=kL)
    no_sat26 = drv("scorecard_institutions_without_fall2024_sat", cov["insts"] - cov["sat26"])
    no_satE = drv("scorecard_institutions_without_sat_falls_2014_15", cov["insts"] - cov["satE"])
    # conservative variance: in how many statistics does it bind?
    allr = ([("T1a", r) for r in T1.values()] + [("T1a_secondary", r) for r in SEC.values()]
            + [("T1a_exploratory", r) for r in SH.values()] + [("T1b", r) for r in FE.values()]
            + [("T3", r) for r in T3.values()])
    nbind = drv("statistics_where_conservative_variance_binds", sum(r["cbind"] for _, r in allr), n=len(allr),
                note="max(V_two-way, V_field, V_inst) > V_two-way")
    nbind_vs = drv("statistics_where_V_inst_binds", sum(r["cwhich"] == "V_inst" for _, r in allr), n=len(allr))
    nbind_vf = drv("statistics_where_V_field_binds", sum(r["cwhich"] == "V_field" for _, r in allr), n=len(allr))
    fe_lev = [FE[k_] for k_ in FE_LEV_KEYS]
    n_fe_vi = drv("t1b_level_betas_with_V_inst_below_V_fxi", sum(r["tvs"] < r["tvi"] for r in fe_lev), n=len(fe_lev))

    steps_diff = [s for s in ("pcs", "pcs_adm", "broad", "sel") if reading(T1[(s, "E-26")], d1) == "differs"]
    o = []
    o.append("# Era-matched selectivity — do the 2026 selectivity controls misdate the graduates?")
    o.append("")
    o.append("Script: `scripts/68_era_selectivity.py` (seed 68; one seed per bootstrap replicate; byte-identical on re-run "
             "and for any number of workers). Every number below is printed by that script from its own computation "
             "and is also in `data/interim/era_selectivity.csv` (estimates and intervals in sections T1a-T3, counts in "
             "`coverage`, ratios and counts computed for this text in `derived`; each row has a `status`). coupling = "
             "Spearman, across institutions offering a bachelor's field, of field prestige F (−Wapman field rank) and "
             "graduates' median earnings; partial coupling = the same after rank-residualizing both sides on the "
             "stated controls. Descriptive, not causal. Public data only (College Scorecard, Census PSEO, Wapman et al.). "
             "Revised after an independent verification (see Revisions).")
    o.append("")
    o.append("## Answer")
    o.append("")
    head = ("**Key question: scripts/55 and scripts/58 net out selectivity with SAT_AVG and ADM_RATE of the fall-2024 "
            "entering class and the AY 2023-24 Pell share (the 2026 \"Most Recent\" file; test-optional era), but the "
            "earnings belong to graduates who entered about a decade earlier. Do controls dated to the graduates' own "
            "entry years give a different answer on the same programs?** ")
    if all(v == 0 for v in ndf.values()):
        head += ("None of the three primary differences (era-matched minus 2026 controls) has a 95% CI that excludes 0, "
                 "under the pre-specified two-way variance or under a conservative one (the largest of the two-way and "
                 "the one-way variances). ")
    else:
        head += (f"{ndf['num']} of the 3 primary differences (era-matched minus 2026 controls) have a 95% CI that "
                 f"excludes 0 under the pre-specified variance ({ndf['num_c']} under the conservative one). ")
    head += ("Whether they are equivalent to 0 depends on the margin. The pre-specified reading rule defines the "
             "margin as one third of the 2026-control estimate, but its number for T3 (±0.010/yr) is one third of "
             f"scripts/59's raw career-time slope ({f_(ctx['t3info']['slope_full'], 4)}/yr), not of a "
             f"selectivity-controlled one; it is {100 * delta_share26:.0f}% of the 2026-control partial slope on the T3 "
             f"cells ({f_(s26, 4)}/yr). At the pre-set numeric margins, {cnt_txt('num')}. At one third of the 2026-control "
             f"estimate on the same programs or cells (±{g1['alt_lab']}, ±{gb['alt_lab']} log points, "
             f"±{g3['alt_lab']}/yr), {cnt_txt('alt_r')}. With the conservative variance the counts are {cnt['num_c']} "
             f"and {cnt['alt_c']} of 3. ")
    head += (f"The firm T3 result is the comparison with the raw slope: net of era-matched selectivity the career-time "
             f"slope of the coupling is {ratio_t3:.2f} of the raw slope on the same cells (E − raw "
             f"{f_(T3[('P3', 'E-raw')]['est'], 4)}/yr {ci_(T3[('P3', 'E-raw')], 4)}) and "
             + ("stays above 0" if T3[("P3", "selE")]["tlo"] > 0 and T3[("P3", "selE")]["clo"] > 0 else
                "its CI " + ("excludes" if T3[("P3", "selE")]["tlo"] > 0 else "includes") + " 0")
             + f" ({f_(T3[('P3', 'selE')]['est'], 4)}/yr {ci_(T3[('P3', 'selE')], 4)}). Against the 2026-control "
             f"partial slope, the era-matched one is {abs(100 * rel_pt):.0f}% {'lower' if rel_pt < 0 else 'higher'} "
             f"(point estimate), and the 90% CI of the difference runs from {pct(rel_lo)} to {pct(rel_hi)} of it. ")
    if steps_diff:
        head += ("Of the other pre-specified ladder steps, " + ", ".join(SPEC1_SHORT[x] for x in steps_diff)
                 + f" {'differs' if len(steps_diff) == 1 else 'differ'} (E − 26 "
                 + ", ".join(f"{f_(T1[(x, 'E-26')]['est'])} {ci_(T1[(x, 'E-26')])}" for x in steps_diff)
                 + "; unadjusted for the three non-primary ladder steps tested). ")
    head += (f"Era-matched and 2026 SAT_AVG rank the institutions of the Scorecard analysis "
             f"{'almost identically' if sat['est'] >= 0.9 else 'similarly' if sat['est'] >= 0.7 else 'differently'} "
             f"(Spearman {f_(sat['est'], 2, False)}); admission rates rank them "
             f"{'less' if adm['est'] < sat['est'] else 'no less'} stably ({f_(adm['est'], 2, False)}). On the PSEO panel "
             f"institutions the admission-rate ranking of the 2001-2008 entry windows is lower still "
             f"({f_(min(ps_adm), 2, False)} to {f_(max(ps_adm), 2, False)}); part of that gap is the institution set, not "
             f"the era: single falls 2001-2008 give {f_(min(pse_adm_08), 2, False)} to {f_(max(pse_adm_08), 2, False)} "
             f"on the PSEO institutions and {f_(min(fos_adm_08), 2, False)} to {f_(max(fos_adm_08), 2, False)} on the "
             "Scorecard institutions. ")
    head += ("None of the three primary quantities moves detectably when the controls are re-dated; where a reading is "
             "inconclusive, the interval cannot rule out a change of the size of the margin (details below)."
             if ndf["num"] == 0 else "The differences are stated below with their intervals.")
    o.append(head)
    o.append("")
    # 1. T2
    o.append(f"1. **How similarly the era-matched and 2026 controls rank institutions (T2).** Across the {sat['n']} "
             f"institutions of the Scorecard analysis with both values, Spearman(SAT_AVG falls 2014-15, SAT_AVG fall 2024) "
             f"= {f_(sat['est'])} [{f_(sat['lo'])}, {f_(sat['hi'])}]; ADM_RATE {f_(adm['est'])} [{f_(adm['lo'])}, "
             f"{f_(adm['hi'])}] (n={adm['n']}); PCTPELL (mean of AY 2014-15 and 2015-16 vs AY 2023-24) {f_(pell['est'])} "
             f"[{f_(pell['lo'])}, {f_(pell['hi'])}] (n={pell['n']}). For the PSEO entry windows of cohorts "
             f"2004/2007/2010, on the PSEO panel institutions: SAT_AVG {', '.join(f_(x, 2, False) for x in ps_sat)}; "
             f"ADM_RATE {', '.join(f_(x, 2, False) for x in ps_adm)} (95% CIs in the table; the lower ADM_RATE values "
             f"reflect both the earlier falls and the public-heavy PSEO institution set, see answer 6). The 2026 file "
             f"has a fall-2024 SAT_AVG for {cov['sat26']} of the {cov['insts']} analysis institutions and the "
             f"era-matched window for {cov['satE']}; {cov['onlyE']} have an era-matched SAT but none in fall 2024 and "
             f"{cov['only26']} the reverse.")
    # 2. T1a
    r2 = T1[("broad", "E-26")]
    o.append(f"2. **Level ladder (T1a; {kL} fields, {nprog} common programs).** Mean coupling: raw "
             f"{f_(T1[('raw', '-')]['est'])} {ci_(T1[('raw', '-')])}; broad net coupling (SAT + ADM + Pell + control + "
             f"state earnings level) {f_(T1[('broad', '26')]['est'])} {ci_(T1[('broad', '26')])} with the 2026 controls "
             f"and {f_(T1[('broad', 'E')]['est'])} {ci_(T1[('broad', 'E')])} with era-matched controls. Difference "
             f"E − 26 = {f_(r2['est'])} {ci_(r2)} (90% CI {ci90(r2)}; MDE at 80% power {f_(r2['mde'], 3, False)}): "
             f"{short(g1['num'])} at the pre-set margin of ±{d1:g} and {short(g1['alt_r'])} at ±{g1['alt_lab']} (one "
             f"third of the 2026-control estimate on these programs); with the conservative variance (SE "
             f"{r2['cse']:.4f}, from {r2['cwhich']}; 90% CI {cci90(r2)}) {short(g1['num_c'])} at ±{d1:g} and "
             f"{short(g1['alt_c'])} at ±{g1['alt_lab']}. The share of the raw mean removed by the broad controls is "
             f"{f_(SH['26']['est'], 2, False)} (2026) and {f_(SH['E']['est'], 2, False)} (era-matched) (exploratory; "
             f"E − 26 {f_(SH['E-26']['est'])} {ci_(SH['E-26'])}). Along the ladder, E − 26 is "
             + "; ".join(f"{SPEC1_SHORT[s]} {f_(T1[(s, 'E-26')]['est'])} {ci_(T1[(s, 'E-26')])}"
                         for s in ("pcs", "pcs_adm", "broad", "sel")) + ". "
             + (("Steps whose E − 26 CI excludes 0: " + ", ".join(
                 f"{SPEC1_SHORT[s]} ({'era-matched controls remove more' if T1[(s, 'E-26')]['est'] < 0 else '2026 controls remove more'}; "
                 f"conservative CI {cci_(T1[(s, 'E-26')])})" for s in steps_diff)
                 + (". The steps with SAT_AVG or ADM_RATE do not differ." if
                    not any(s in steps_diff for s in ("pcs_adm", "broad", "sel")) else "."))
                if steps_diff else "No step's E − 26 CI excludes 0.")
             + ((" STABBR is not re-dated: in every MERGED file it is backfilled with the most recent IPEDS value (data "
                 "dictionary, Institution_Cohort_Map), so the state is the same for every common program by "
                 f"construction ({ctx['t1info']['stabbr_differs_cells']} differ). CONTROL is dated, and it is the same "
                 f"for all {cov['okC']} common institutions ({ctx['t1info']['control_differs_insts']} differ). So among "
                 "the re-dated variables only PCTPELL changes in that step: the Pell share of AY 2014-16 removes more of "
                 "the coupling than that of AY 2023-24.")
                if "pcs" in steps_diff and ctx["t1info"]["stabbr_differs_cells"] == 0
                and ctx["t1info"]["control_differs_insts"] == 0 and T1[("pcs", "E-26")]["est"] < 0 else ""))
    # 3. T1b
    rb = FE[("main", "E-26")]
    m26, mE = FE[("main", "26")], FE[("main", "E")]
    hw_b = drv("t1b_main_E_minus_26_90ci_half_width", Z95 * rb["tse"], note="1.645 x two-way SE")
    inc = lambda lo, hi: lo <= 0 <= hi
    lev_txt = (f"Under the pre-specified two-way variance the main-spec 95% CI "
               + ("includes 0 with both sets of controls" if inc(m26["tlo"], m26["thi"]) and inc(mE["tlo"], mE["thi"])
                  else "includes 0 with the 2026 controls and not with era-matched controls" if inc(m26["tlo"], m26["thi"])
                  else "includes 0 with era-matched controls and not with the 2026 controls" if inc(mE["tlo"], mE["thi"])
                  else "excludes 0 with both sets of controls") + ". ")
    if n_fe_vi:
        vi_names = [FE_LAB[k_] for k_ in FE_LEV_KEYS if FE[k_]["tvs"] < FE[k_]["tvi"]]
        lev_txt += (f"For {n_fe_vi} of the {len(fe_lev)} T1b level betas ({andj(vi_names)}) the shared-institution "
                    "variance is below the independent-per-field one, so their two-way variance falls below the "
                    f"field-jackknife variance alone (main spec, era-matched: V_field {mE['tvf']:.3g}, V_inst "
                    f"{mE['tvs']:.3g}, V_field×inst {mE['tvi']:.3g}; 2026: {m26['tvf']:.3g}, {m26['tvs']:.3g}, "
                    f"{m26['tvi']:.3g}). With the conservative variance the main-spec CIs are {cci_(m26, 4)} (2026) and "
                    f"{cci_(mE, 4)} (era-matched)"
                    + (", both including 0" if inc(m26["clo"], m26["chi"]) and inc(mE["clo"], mE["chi"]) else "")
                    + ". ")
    both_inc_any = (inc(m26["clo"], m26["chi"]) or inc(m26["tlo"], m26["thi"])) and \
                   (inc(mE["clo"], mE["chi"]) or inc(mE["tlo"], mE["thi"]))
    if both_inc_any:
        lev_txt += ("Neither main-spec level CI clearly excludes 0"
                    + ("; the two-way crossing of 0 is borderline, not a change between the two sets of controls"
                       if inc(m26["tlo"], m26["thi"]) != inc(mE["tlo"], mE["thi"]) else "")
                    + f" (paired difference {f_(rb['est'], 4)}). ")
    o.append(f"3. **Within-institution department slope (T1b; {fe_n['cells']} programs, {fe_n['insts']} institutions, "
             f"{fe_n['fields']} fields).** Pooled beta (log points per within-field SD of department prestige), main "
             f"spec (institution FE + field × {{SAT, ADM, Pell, brand G}}): {f_(m26['est'], 4)} {ci_(m26, 4)} with the "
             f"2026 controls, {f_(mE['est'], 4)} {ci_(mE, 4)} era-matched. " + lev_txt
             + f"E − 26 = {f_(rb['est'], 4)} {ci_(rb, 4)} (90% CI {ci90(rb, 4)}; MDE {f_(rb['mde'], 4, False)}): "
             f"{short(gb['num'])} at the pre-set margin of ±{db:g} log points"
             + (f" (the 90% CI half-width, {hw_b:.4f}, exceeds the margin, so this design cannot establish "
                f"equivalence at ±{db:g})" if hw_b >= db else "")
             + f", and {short(gb['alt_r'])} at ±{gb['alt_lab']} (one third of the 2026-control estimate on these "
             f"programs); "
             + (f"the conservative variance ({cci_(rb, 4)}) does not change either reading. "
                if (gb["num_c"], gb["alt_c"]) == (gb["num"], gb["alt_r"]) else
                f"with the conservative variance ({cci_(rb, 4)}) the readings are {short(gb['num_c'])} and "
                f"{short(gb['alt_c'])}. ")
             + f"Without institution FE the slope is {f_(FE[('nofe', '-')]['est'], 4)}; with institution FE alone "
             f"{f_(FE[('a', '-')]['est'], 4)}. scripts/58's own pooled estimate, reproduced here: "
             f"{f_(ctx['s58chk']['pooled'], 4)} on {ctx['s58chk']['n_cells']} programs. The era-matched main-spec slope "
             f"is about {pay_E:.1f}% of pay per SD of department prestige.")
    # 4. T3
    o.append(f"4. **Career-time slope of the partial coupling (T3; PSEO V4.14.1 fixed cohorts 2004/2007/2010, "
             f"{P3n['cells']} field × cohort cells, {T3[('P3', 'selE')]['k']} fields, {P3n['insts']} institutions).** "
             f"Mean per-field OLS slope over y1/y5/y10 of the coupling given SAT_AVG + ADM_RATE with era-matched "
             f"controls: {f_(T3[('P3', 'selE')]['est'], 4)}/yr {ci_(T3[('P3', 'selE')], 4)}, {ratio_t3:.2f} of the raw "
             f"slope on the same cells ({f_(T3[('P3', 'raw')]['est'], 4)}/yr {ci_(T3[('P3', 'raw')], 4)}); E − raw = "
             f"{f_(T3[('P3', 'E-raw')]['est'], 4)} {ci_(T3[('P3', 'E-raw')], 4)}"
             + (f" (conservative {cci_(T3[('P3', 'E-raw')], 4)})" if T3[("P3", "E-raw")]["cbind"] else "")
             + f". With the 2026 controls the partial slope is {f_(s26, 4)}/yr "
             f"{ci_(T3[('P3', 'sel26')], 4)}. E − 26 = {f_(r3['est'], 4)} {ci_(r3, 4)} (90% CI {ci90(r3, 4)}; MDE "
             f"{f_(r3['mde'], 4, False)}): {short(g3['num'])} at the pre-set margin of ±{d3:g}/yr, which is "
             f"{100 * delta_share:.0f}% of scripts/59's raw slope and {100 * delta_share26:.0f}% of the 2026-control "
             f"partial slope compared here. Relative to that partial slope the point difference is {pct(rel_pt)} and "
             f"the 90% CI {pct(rel_lo)} to {pct(rel_hi)}; at ±{g3['alt_lab']}/yr (one third of it) the reading is "
             f"{short(g3['alt_r'])}. With the conservative variance (90% CI {cci90(r3, 4)}) the readings are "
             f"{short(g3['num_c'])} at ±{d3:g} and {short(g3['alt_c'])} at ±{g3['alt_lab']}. An equivalence reading "
             "bounds the difference by the margin; it does not show that the two slopes are equal.")
    # 5. sensitivities
    sec_items = [("broad W − 26 (T1a)", RG[("T1", "broad", "W-26")], 3),
                 (f"strict (state FE) E − 26 (T1a, {kS} fields)", RG[("T1", "strict", "E-26")], 3),
                 ("main W − 26 (T1b)", RG[("FE", "main", "W-26")], 4),
                 ("slope E − 26 incl. cohort 2001 with fall-2001 stand-in (T3)", RG[("T3", "P4", "E-26")], 4),
                 ("slope E − 26, SAT + ADM + control + state level (T3)", RG[("T3", "P3", "secE-26")], 4)]

    def sens(lab, g, d):
        r = g["r"]
        s = f"{lab}: {f_(r['est'], d)} {ci_(r, d)} (90% CI {ci90(r, d)}), {reading_str(g, md=True, flags=False)}"
        if g["flag"]:
            s += (f" — the ±{g['delta']:g} reading sits on the margin within Monte Carlo error ({g['flag']}), so it "
                  "is not a firm contrast with the primary")
        if g["flag_alt"]:
            s += f" — the ±{g['alt_lab']} reading sits on the margin within Monte Carlo error ({g['flag_alt']})"
        return s
    o.append("5. **Pre-specified sensitivities** (readings at the pre-set margin, then at one third of the 2026 "
             "estimate; max-var = conservative variance where it changes the reading). "
             + "; ".join(sens(*x) for x in sec_items)
             + f". Era-complete sample with era-matched controls vs scripts/55's own sample with 2026 controls "
             f"(different programs; broad): {f_(SEC[('broad', 'SE')]['est'])} ({SEC[('broad', 'SE')]['k']} fields) vs "
             f"{f_(SEC[('broad', 'S26')]['est'])} ({SEC[('broad', 'S26')]['k']} fields), difference "
             f"{f_(SEC[('broad', 'SE-S26')]['est'])} {ci_(SEC[('broad', 'SE-S26')])}.")

    def cv_(iset, var, y):
        c = CV[(iset, var, y)]
        return f"{f_(c['est'], 2, False)} [{f_(c['lo'], 2, False)}, {f_(c['hi'], 2, False)}]"

    def cv_min(iset, var):
        cs = [c for c in ctx["t2curve"] if c["iset"] == iset and c["var"] == var and np.isfinite(c["est"])]
        c = min(cs, key=lambda c: (c["est"], c["y"]))
        return f"{cv_(iset, var, c['y'])} (fall {c['y']})"
    dv_p, dv_f = ctx["t2div"][("pseo", 2003)], ctx["t2div"][("fos", 2014)]
    o.append(f"6. **Exploratory.** Spearman of SAT_AVG in fall y with fall 2024 on the Scorecard analysis institutions: "
             f"lowest {cv_min('fos', 'SAT_AVG')}, fall 2014 {cv_('fos', 'SAT_AVG', 2014)}, fall 2023 "
             f"{cv_('fos', 'SAT_AVG', 2023)}. ADM_RATE drifts much more: fall 2003 {cv_('fos', 'ADM_RATE', 2003)}, fall "
             f"2014 {cv_('fos', 'ADM_RATE', 2014)} on the same institutions, and on the PSEO panel institutions fall "
             f"2003 {cv_('pseo', 'ADM_RATE', 2003)} and fall 2014 {cv_('pseo', 'ADM_RATE', 2014)} (SAT_AVG there: fall "
             f"2003 {cv_('pseo', 'SAT_AVG', 2003)}); over falls 2001-2008 the ADM_RATE values are "
             f"{f_(min(pse_adm_08), 2, False)}-{f_(max(pse_adm_08), 2, False)} on the PSEO institutions and "
             f"{f_(min(fos_adm_08), 2, False)}-{f_(max(fos_adm_08), 2, False)} on the Scorecard institutions, so the "
             "institution set accounts for much of the gap. The drift is ordered by selectivity: across the PSEO panel "
             f"institutions, Spearman(change in ADM_RATE from fall 2003 to fall 2024, SAT_AVG fall 2003) = "
             f"{f_(dv_p['est'], 2)} [{f_(dv_p['lo'], 2)}, {f_(dv_p['hi'], 2)}] (n={dv_p['n']}; mean change "
             f"{f_(dv_p['mean_d'], 3)}), i.e. institutions with higher SAT in 2003 lowered their admission rate more; "
             f"from fall 2014 on the Scorecard analysis institutions {f_(dv_f['est'], 2)} [{f_(dv_f['lo'], 2)}, "
             f"{f_(dv_f['hi'], 2)}] (n={dv_f['n']}). The mean SAT_AVG on the same {cov['both']} institutions is "
             f"{cov['sat_meanE']:.0f} (falls 2014-15) and {cov['sat_mean26']:.0f} (fall 2024); the level is not "
             "comparable across the March-2016 SAT redesign (see caveats), ranks are. Per field (broad partial, common "
             f"programs, {kL} fields), E − 26 ranges from {f_(pf_d[0][1])} ({LAB.get(pf_d[0][0], pf_d[0][0])}) to "
             f"{f_(pf_d[-1][1])} ({LAB.get(pf_d[-1][0], pf_d[-1][0])}); {n_big} fields move by more than 0.10 and the "
             f"across-field SD of the per-field difference is {sd_pf:.3f}. The mean over fields is stable; single "
             "fields' partial coupling depends on how the controls are dated (per-field values in the CSV, section "
             "T1a_field).")
    o.append("")
    # ---------------- primary readings grid ----------------
    o.append("Readings of the three primary differences (E − 26) under each margin and variance. The pre-specified "
             "reading is the first column; the others were added after verification.")
    o.append("")
    o.append("| primary difference | pre-set margin, two-way variance | ⅓ of 2026 estimate, two-way | pre-set margin, "
             "conservative variance | ⅓ of 2026 estimate, conservative |")
    o.append("|---|---|---|---|---|")
    for t, lab, g in prim:
        o.append(f"| {lab} | {g['num']} | {g['alt_r']} | {g['num_c']} | {g['alt_c']} |")
    o.append(f"| equivalent (of 3) | {cnt['num']} | {cnt['alt_r']} | {cnt['num_c']} | {cnt['alt_c']} |")
    o.append("")
    # ---------------- key numbers ----------------
    o.append("## Key numbers")
    o.append("")
    o.append("Two-way (field, institution) cluster variance of scripts/59 (Cameron-Gelbach-Miller: V_field + "
             f"V_institution − V_field×institution; {B} replicates for each institution draw) unless stated; normal "
             "95% CI = estimate ± 1.96 SE; p two-sided from z; MDE = minimum detectable difference at 80% power, "
             "two-sided 5% (2.80 SE). Max-var CI = the same with the conservative variance max(V_two-way, V_field, "
             "V_institution) (added after verification; equal to the two-way CI where the two-way variance is the "
             "largest). Differences use the same replicates (paired). k = fields; n = programs (T1), field × cohort "
             "cells (T3). Reading (E − 26 and W − 26 rows): at the pre-set margin (±0.05 for mean coupling, ±0.003 log "
             "points for beta, ±0.01/yr for the slope; the pre-specified reading), then at one third of the "
             "corresponding 2026-control estimate on the same programs / cells; max-var in brackets where it changes "
             "the reading. Status: primary, pre-specified, pre-specified sensitivity, or exploratory (not in the "
             "pre-specification).")
    o.append("")
    o.append("| quantity | sample / spec | estimate | 95% CI | max-var 95% CI | p | MDE80 | k | n | status; reading |")
    o.append("|---|---|---|---|---|---|---|---|---|---|")
    o.append(_row("T1a mean coupling, raw", "Scorecard FoS BA EARN_MDN_4YR; common programs", T1[("raw", "-")],
                  status=st_t1("raw", "-")))
    for s in ("pcs", "pcs_adm", "broad", "sel"):
        for v in VERS:
            o.append(_row(f"T1a mean partial coupling, {SPEC1_SHORT[s]}", f"common programs; {VLAB[v]} controls",
                          T1[(s, v)], status=st_t1(s, v)))
        for v in ("E", "W"):
            o.append(_row(f"T1a {SPEC1_SHORT[s]}: {v} − 26", "common programs; paired", T1[(s, f"{v}-26")],
                          status=st_t1(s, f"{v}-26"), reading_=reading_str(RG[("T1", s, f"{v}-26")], md=True)))
    for v in ("26", "E"):
        o.append(_row("T1a mean partial coupling, strict (state FE)", f"common programs, strict field set; {VLAB[v]}",
                      T1[("strict", v)], status=st_t1("strict", v)))
    o.append(_row("T1a strict: E − 26", "common programs, strict field set; paired", T1[("strict", "E-26")],
                  status=st_t1("strict", "E-26"), reading_=reading_str(RG[("T1", "strict", "E-26")], md=True)))
    o.append(_row("T1a mean coupling, raw, strict field set", "common programs, strict field set", T1[("raw_strict", "-")],
                  status=st_t1("raw_strict", "-")))
    for v in VERS:
        o.append(_row("T1a share of raw mean removed by broad", f"common programs; {VLAB[v]}", SH[v], status="exploratory"))
    for v in ("E-26", "W-26"):
        o.append(_row(f"T1a share removed: {v.replace('-', ' − ')}", "common programs; paired", SH[v],
                      status="exploratory"))
    for s in LADDER:
        for smp, lab in (("SE", "era-complete programs, E controls"), ("S26", "scripts/55 SAT sample, 2026 controls"),
                         ("SE-S26", "era-complete minus scripts/55 sample (different programs)")):
            o.append(_row(f"T1a secondary: mean {SPEC1_SHORT[s]}", lab, SEC[(s, smp)], status="pre-specified sensitivity"))
    for key, lab in ((("nofe", "-"), "field FE only (no institution FE)"), (("a", "-"), "(a) institution FE + field FE"),
                     *[((sp, v), f"{'(b) + field × {SAT, ADM, Pell}' if sp == 'b' else 'main: + field × {SAT, ADM, Pell, G}'}; {VLAB[v]}")
                       for sp in ("b", "main") for v in VERS]):
        o.append(_row("T1b pooled beta", f"common programs with G; {lab}", FE[key], d=4, status=st_fe(*key)))
    for sp in ("b", "main"):
        for v in ("E", "W"):
            o.append(_row(f"T1b beta {sp}: {v} − 26", "common programs with G; paired", FE[(sp, f"{v}-26")], d=4,
                          status=st_fe(sp, f"{v}-26"), reading_=reading_str(RG[("FE", sp, f"{v}-26")], md=True)))
    for v in ("26", "E"):
        o.append(_row("T1b beta (a) − main", f"common programs with G; {VLAB[v]}", FE[("a-main", v)], d=4,
                      status=st_fe("a-main", v)))
    for cset in ("P3", "P4"):
        lab = T3_SAMPLE[cset]
        for nm, ql in (("raw", "raw coupling slope"), ("sel26", "partial slope given SAT+ADM, 2026 controls"),
                       ("selE", "partial slope given SAT+ADM, era-matched"), ("E-26", "partial slope: E − 26"),
                       ("E-raw", "partial slope E − raw"), ("26-raw", "partial slope 26 − raw")):
            g = RG.get(("T3", cset, nm))
            o.append(_row(f"T3 {ql} (/yr)", f"PSEO V4.14.1 fixed cohorts, {lab}", T3[(cset, nm)], d=4,
                          status=st_t3(cset, nm), reading_=reading_str(g, md=True) if g else ""))
        for h in HZ:
            for nm, ql in (("raw", "raw coupling"), ("sel26", "partial coupling, 2026"), ("selE", "partial coupling, era-matched")):
                o.append(_row(f"T3 {ql} at {h}", f"PSEO fixed cohorts, {lab}", T3[(cset, f"{nm}_{h}")],
                              status=st_t3(cset, f"{nm}_{h}")))
    for nm, ql in (("sec_raw", "raw coupling slope"), ("sec26", "partial slope given SAT+ADM+control+state level, 2026"),
                   ("secE", "partial slope given SAT+ADM+control+state level, era-matched"),
                   ("secE-26", "partial slope (SAT+ADM+control+state level): E − 26")):
        g = RG.get(("T3", "P3", nm))
        o.append(_row(f"T3 {ql} (/yr)", f"PSEO fixed cohorts, {T3_SAMPLE['SEC']}", T3[("P3", nm)], d=4,
                      status=st_t3("P3", nm), reading_=reading_str(g, md=True) if g else ""))
    o.append("")
    o.append(f"T2 (institution bootstrap, percentile 95% CI, {B} replicates; n = institutions):")
    o.append("")
    o.append("| quantity | institutions | estimate | 95% CI | n | status |")
    o.append("|---|---|---|---|---|---|")
    for (src, c, v), x in ctx["t2res"].items():
        vl = ({"E": "AY 2014-15 and 2015-16", "W": "AY 2012-13 to 2015-16"}[v] if c == "PCTPELL" else
              {"E": "falls 2014-15", "W": "falls 2012-15"}[v])
        o.append(f"| Spearman({c} {vl}, {c} 2026 file) | Scorecard analysis institutions | {f_(x['est'])} | "
                 f"[{f_(x['lo'])}, {f_(x['hi'])}] | {x['n']} | "
                 f"{'**primary**' if (c, v) == ('SAT_AVG', 'E') else 'pre-specified'} |")
    for (c, lab), x in ctx["t2pseo"].items():
        e = ctx["eras"][c]
        o.append(f"| Spearman({lab} falls {'-'.join(str(y) for y in (e['falls'][0], e['falls'][-1])) if len(e['falls']) > 1 else e['falls'][0]}"
                 f"{' (stand-in; entry window ' + str(e['window'][0]) + '-' + str(e['window'][-1]) + ')' if e['standin'] else ''}, "
                 f"{lab} fall 2024) | PSEO cohort {c} panel institutions ({x['n_panel']}) | {f_(x['est'])} | "
                 f"[{f_(x['lo'])}, {f_(x['hi'])}] | {x['n']} | "
                 f"{'pre-specified sensitivity' if c == '2001' else 'pre-specified'} |")
    o.append("")
    o.append("Samples and coverage (counts; CSV section `coverage`):")
    o.append("")
    o.append("| item | value |")
    o.append("|---|---|")
    for k_, lab in (("insts", "Scorecard analysis institutions (scripts/55 cells)"), ("cells", "programs (field × institution)"),
                    ("sat26", "institutions with fall-2024 SAT_AVG"), ("satE", "institutions with SAT_AVG in falls 2014-15"),
                    ("both", "institutions with both"), ("onlyE", "era-matched SAT_AVG only"), ("only26", "fall-2024 SAT_AVG only"),
                    ("ok26", "institutions with the SAT family complete, 2026 controls (scripts/55 sample)"),
                    ("okE", "institutions with the SAT family complete, era-matched"), ("okC", "institutions, common (both)"),
                    ("cells26", "programs, scripts/55 SAT sample"), ("cellsE", "programs, era-complete"),
                    ("cellsC", "programs, common")):
        o.append(f"| {lab} | {cov[k_]} |")
    o.append(f"| T1a ladder fields (estimable under 26, E and W in every spec) / their common programs | {kL} / {nprog} |")
    o.append(f"| T1a strict fields / their common programs | {kS} / {T1[('strict', 'E')]['n']} |")
    o.append(f"| T1b programs / institutions / fields | {fe_n['cells']} / {fe_n['insts']} / {fe_n['fields']} |")
    for cset in ("P3", "P4", "SEC"):
        x = t3n[cset]
        lab_ = {"P3": "primary, cohorts 2004/2007/2010", "P4": "with cohort 2001",
                "SEC": "control + state-level sensitivity"}[cset]
        o.append(f"| T3 {lab_}: cells (by cohort) / institutions / cell n range | {x['cells']} "
                 f"({', '.join(f'{c}: {n}' for c, n in x['by_coh'].items())}) / {x['insts']} / {x['n_min']}-{x['n_max']} |")
    for c, x in covp.items():
        o.append(f"| PSEO cohort {c}: panel institutions / SAT+ADM 2026 / SAT+ADM era / both | {x['panel']} / "
                 f"{x['sat26']} / {x['satE']} / {x['both']} |")
    o.append(f"| 2026-file values equal MERGED2024_25 (SAT_AVG, ADM_RATE, PCTPELL; UNITIDs with both) | "
             + ", ".join(f"{c}: {same}/{nb}" for c, (nb, same, mism) in ctx["t2agree"].items()) + " |")
    o.append(f"| common programs whose state differs between the 2026 and era files (0 by construction: STABBR is "
             f"backfilled) / common institutions whose control differs | {ctx['t1info']['stabbr_differs_cells']} / "
             f"{ctx['t1info']['control_differs_insts']} |")
    o.append(f"| statistics (T1a, T1b, T3 rows) where the conservative variance exceeds the two-way variance (binding "
             f"part V_institution / V_field) | {nbind} of {len(allr)} ({nbind_vs} / {nbind_vf}) |")
    o.append("")
    o.append("Exploratory: Spearman(X in fall y, X in fall 2024) across institutions, X = SAT_AVG or ADM_RATE; "
             f"institution bootstrap percentile 95% CI ({B} replicates); n = institutions with both values. Scorecard "
             f"= the {cov['insts']} institutions of the Scorecard analysis; PSEO = the institutions of the PSEO "
             "fixed-cohort panel (public-heavy; a different institution set).")
    o.append("")
    o.append("| fall y | SAT_AVG, Scorecard | ADM_RATE, Scorecard | SAT_AVG, PSEO | ADM_RATE, PSEO |")
    o.append("|---|---|---|---|---|")
    for y in range(2001, 2024):
        o.append(f"| {y} | " + " | ".join(
            (lambda c: f"{f_(c['est'], 3, False)} [{f_(c['lo'], 3, False)}, {f_(c['hi'], 3, False)}] (n={c['n']})")(
                CV[(iset, var, y)]) for iset, var in (("fos", "SAT_AVG"), ("fos", "ADM_RATE"), ("pseo", "SAT_AVG"),
                                                      ("pseo", "ADM_RATE"))) + " |")
    o.append("")
    # ---------------- method ----------------
    o.append("## Method")
    o.append("")
    o.append(f"- **Whose entry years.** Scorecard field-of-study EARN_MDN_4YR in the Most-Recent file (data "
             f"dictionary README: \"{dic['release']}\") is, per the dictionary (FieldOfStudy_Cohort_Map), \"{dic['fos_4yr']}\". The "
             f"field-of-study technical documentation ({ver_f} version) defines the earnings cohort as completers who "
             f"received federal aid, excluding those enrolled, deceased or not working in the measurement year "
             f"({pg(fosdoc, 'The cohort of evaluated')}), states that the Most-Recent file takes each earnings "
             f"metric from the last time it was calculated ({pg(fosdoc, 'data file will have data for all earnings metrics')}) "
             f"and that \"Scorecard combines students into 2-year cohorts\" ({pg(fosdoc, 'Scorecard combines students')}), "
             f"i.e. two pooled award years. A bachelor's completer of award "
             f"year 2017-18 (2018-19) who took four years entered in fall 2014 (fall 2015). Era-matched controls (E): "
             f"SAT_AVG and ADM_RATE = mean of falls 2014 and 2015 (files MERGED2014_15, MERGED2015_16); PCTPELL = mean "
             f"of AY 2014-15 and 2015-16 (files MERGED2015_16, MERGED2016_17); CONTROL and STABBR from MERGED2015_16. "
             f"CONTROL is dated in the MERGED files (\"{dic['control_dated'][2015]}\"); STABBR is not: every MERGED "
             f"file carries \"{dic['stabbr_backfill'][2015]}\" (Institution_Cohort_Map, asserted for each file used), "
             f"so it equals the 2026 value by construction. Wide window (W, for 5-6-year completers): falls 2012-15, AY "
             f"2012-13 to 2015-16. The 2026 controls (26) are those of scripts/55: SAT_AVG and ADM_RATE "
             f"\"{dic['mr']['SAT_AVG'][0]}\", PCTPELL \"{dic['mr']['PCTPELL'][0]}\", CONTROL/STABBR "
             f"\"{dic['mr']['CONTROL'][0]}\" (Most_Recent_Inst_Cohort_Map). Each file-year assignment is asserted "
             f"against the dictionary's Institution_Cohort_Map.")
    m0 = dic["m2000"]
    o.append(f"- **PSEO cohorts.** A PSEO bachelor's graduation cohort c pools graduates of calendar years c to c+2 "
             f"(grad_cohort_years = {ctx['t3info']['cohort_years']} on every bachelor's row of cohorts "
             f"{'/'.join(T3_COH4)} in the V4.14.1 earnings file; {PSEO_DOC}); entry = falls c−4 to c−2: cohort 2004 → "
             f"falls 2001-2002, 2007 → 2003-2005, 2010 → 2006-2008; mean over the available falls. Fall 2000 has no "
             f"SAT_AVG or ADM_RATE in the Scorecard files: MERGED2000_01 exists ({m0['rows']} institutions) but has "
             f"{m0['sat']} SAT_AVG and {m0['adm']} ADM_RATE values, and the data dictionary (Institution_Cohort_Map) "
             f"leaves both empty for MERGED_2000-01 and first dates them in {dic['first_ADM_RATE'].replace(' datafile', '')} "
             f"(ADM_RATE: \"{dic['adm_2001']}\"); the institution documentation states that \"SAT and ACT data are "
             f"available from 2001-02 on\" ({pg(insdoc, 'SAT and ACT data are available')}). Cohort 2001 (entry "
             f"1997-1999) has no SAT data and enters only a sensitivity with fall-2001 values. PSEO institutions (8-digit "
             f"OPEID) are matched to UNITID through the 2026 institution file (main campus where one OPEID has several).")
    o.append("- **T1a.** scripts/55's partial coupling (ranks of prestige, earnings and continuous controls; categorical "
             "controls as dummies; OLS residualization; estimate kept when n ≥ 15 and residual df ≥ 10), on the "
             "common programs, i.e. institutions with the SAT family complete under both the 2026 and the E controls "
             "(the state earnings level of scripts/55, a geography control from the 2026 file, is used in every "
             "version). Fixed field set: fields estimable in every ladder spec under 26, E and W. Statistic = "
             "unweighted mean over fields. Reproduction: the 2026-control broad partial on scripts/55's own sample "
             f"equals data/interim/selectivity_fields.csv in {ctx['s55chk'][1]} fields (max |diff| "
             f"{ctx['s55chk'][0]:.1e}, CSV precision).")
    o.append("- **T1b.** scripts/58's within-institution design (`fe_build`): log earnings on z(prestige) within "
             "field, field FE, institution FE absorbed, field-specific slopes on the listed institution traits "
             "(standardized within field), pooled beta. The same programs under every spec and version (asserted). "
             f"Reproduction: scripts/58's main-spec per-field beta_f on its own sample (max |diff| "
             f"{ctx['s58chk']['maxdiff']:.1e}; pooled beta {f_(ctx['s58chk']['pooled'], 4)} on "
             f"{ctx['s58chk']['n_cells']} programs). Replicates re-absorb the institution FE under the replicate's "
             "row weights (weighted within-institution demeaning), so an institution drawn twice counts twice and a "
             "field dropped by the jackknife leaves the other fields' institution means.")
    o.append(f"- **T3.** scripts/52's balanced fixed-cohort panel on PSEO V4.14.1 via scripts/59 (canonical join); "
             f"reproduction: the all-field raw slope {f_(ctx['t3info']['slope_full'], 6)} equals scripts/59's "
             f"{f_(ctx['t3info']['slope_ref'], 6)}. Institutions with SAT_AVG and ADM_RATE under both versions; cells "
             "re-formed with n ≥ 15. Per cell, the partial Spearman of F and median earnings given rank SAT_AVG + "
             "rank ADM_RATE at y1, y5, y10 and its OLS slope on years {1, 5, 10}; field value = mean over its cohorts; "
             "statistic = mean over fields; raw slope on the same cells.")
    shared_red = {**{f"T1a {g}": ctx["red1"][f"shared_{g}"] for g in T1_GROUPS},
                  **{f"T3 {g}": ctx["red3"][f"shared_{g}"] for g in T3_GROUPS}}
    shared_red = {{"T1a ladder": "the T1a primary ladder", "T1a strict": "the T1a strict set",
                   "T1a sec": "the T1a secondary samples", "T3 p3": "the T3 primary", "T3 p4": "T3 with cohort 2001",
                   "T3 sec": "the T3 control + state-level sensitivity"}[k]: v for k, v in shared_red.items()}
    max_red = drv("max_share_of_shared_draws_redrawn", max(shared_red.values()) / B,
                  note=max(shared_red, key=shared_red.get))
    o.append("- **Inference.** scripts/59 `twoway()`: V_field (between-field variance / k; delete-one-field jackknife "
             "for the pooled beta and the exploratory share) + V_institution (one institution multinomial draw shared "
             f"by all fields) − V_field×institution (an independent institution draw per field), {B} replicates each; "
             "a replicate in which a statistic of its bootstrap group is undefined is redrawn from its own seed stream "
             "(typically a small cell whose resample has few distinct institutions, which prestige and a control, "
             "e.g. SAT_AVG, put in exactly the same order, so the prestige ranks have zero residual; the materialized "
             "resample gives the same undefined value). The n ≥ 15 / df ≥ 10 rule applies to the point estimates; "
             "replicates are kept whenever the statistic is defined. Groups: T1a primary ladder "
             "(common programs, fixed field set), strict, secondary samples; T1b; T3 primary (cohorts 2004/2007/2010), "
             "with cohort 2001, and the control + state-level sensitivity; so a primary statistic is conditioned only "
             f"on its own group's estimability. Redrawn of {B} (shared/independent; the independent counts sum over "
             f"fields): T1a ladder {ctx['red1']['shared_ladder']}/{ctx['red1']['ind_ladder']}, strict "
             f"{ctx['red1']['shared_strict']}/{ctx['red1']['ind_strict']}, secondary {ctx['red1']['shared_sec']}/"
             f"{ctx['red1']['ind_sec']}; T1b {ctx['redfe']['shared']}/{ctx['redfe']['ind']}; T3 primary "
             f"{ctx['red3']['shared_p3']}/{ctx['red3']['ind_p3']}, with 2001 {ctx['red3']['shared_p4']}/"
             f"{ctx['red3']['ind_p4']}, control + state level {ctx['red3']['shared_sec']}/{ctx['red3']['ind_sec']}. "
             "T2: institution bootstrap percentile CI. Weighted ranks and weighted residualization reproduce "
             "resampling with replacement exactly; at unit weights they equal the point estimates (asserted, "
             "max |diff| < 1e-9). Added after verification (not pre-specified): (i) the conservative variance "
             "max(V_two-way, V_field, V_institution), reported for every row; (ii) a reading is flagged as on the "
             "margin when it would change if the SE moved by 2 Monte Carlo SEs, i.e. when a deciding CI endpoint lies "
             "within 2 Monte Carlo SEs of its threshold (Monte Carlo SE of the SE = scripts/59's `tmcse` / 1.96).")
    o.append(f"- **Pre-specification** (in the script docstring, md5 `{ctx['prespec']}`). The three tests (T1 level "
             "ladder and within-institution slope, T2 SAT_AVG rank stability, T3 career-time partial slope) were fixed "
             "on 2026-09-24 before any result. The operational details in the docstring block (samples, entry-year "
             "windows, margins, reading rule, sensitivities) were written during an earlier build of this lane that "
             "was interrupted before it reported; no later run changed the block. The earliest saved copy of that "
             f"build's script ({BUILD_BACKUP_UTC}) already contains the block with this md5, and the build's first "
             f"saved estimates (a 100-replicate development run) are from {BUILD_DEV_UTC}; its earlier scratch scripts "
             "computed file coverage and the institution-id match only. Unsaved interactive checks cannot be ruled "
             "out. Reading rule for a primary difference D = E − 26: \"differs\" if the 95% CI excludes 0; "
             "\"equivalent within ±δ\" if the 90% CI lies inside ±δ; otherwise \"inconclusive\". The block defines δ "
             "as \"one third of the 2026-control estimate in scripts/55 / 58 / 59\" and gives the numbers 0.05, 0.003 "
             "and 0.010. The first two are about one third of scripts/55's broad mean on its own sample "
             f"({f_(SEC[('broad', 'S26')]['est'])}, reproduced here) and of scripts/58's pooled main-spec beta "
             f"({f_(ctx['s58chk']['pooled'], 4)}). The third is one third of scripts/59's all-field raw career-time "
             f"slope ({f_(ctx['t3info']['slope_full'], 4)}/yr); scripts/59 has no selectivity-controlled slope, and the "
             f"2026-control partial slope computed here ({f_(s26, 4)}/yr) is smaller, so the text and the number of the "
             "rule disagree for T3. Both readings are reported: at the numbers as written (the pre-specified reading) "
             "and at one third of the 2026-control estimate on the same programs / cells "
             f"(±{g1['alt_lab']}, ±{gb['alt_lab']}, ±{g3['alt_lab']}). The share-removed rows, the fall-by-fall "
             "SAT_AVG / ADM_RATE curves, the ADM_RATE divergence, the per-field differences, the level means, "
             "T1b (a) − main, T3 26 − raw and the T3 per-horizon couplings are exploratory.")
    o.append("")
    # ---------------- caveats ----------------
    o.append("## Caveats")
    o.append("")
    o.append("- **Entry year is inferred, not observed.** The dating assumes four years from fall entry to completion. "
             "Completers who took five or six years entered one or two falls earlier (window W covers them). Transfer "
             "students and completers who first enrolled elsewhere are not described by either file's first-time "
             "entering class. The Scorecard earnings cohort covers only federally aided completers "
             f"({pg(fosdoc, 'The cohort of evaluated')}); SAT_AVG and ADM_RATE describe an institution's first-time "
             "applicants and admits, not its completers.")
    o.append(f"- **SAT scale.** The SAT was redesigned in March 2016; Scorecard SAT data from 2017-18 on are on the "
             f"new scale ({pg(insdoc, 'Starting with the 2017-18')}). Falls 2012-2015 and 2001-2008 are on the old "
             "scale and fall 2024 on the new one. All controls enter as within-field ranks, so the scale change does "
             "not affect T1 or T3; the mean SAT levels in answer 6 are not comparable.")
    o.append("- **Institution-level controls.** Both versions describe an institution's entering class, not a "
             "department's graduates. Era-matching fixes the dating of the proxy but not the gap between an "
             "institution's entrants and a department's completers. The selectivity-pricing caveat of scripts/55 "
             "(imperfect proxies leave a partial ordered like the baseline) applies unchanged.")
    o.append(f"- **PSEO sample.** Era-matched controls exist for fewer early-cohort institutions (e.g. cohort 2004: "
             f"{covp['2004']['both']} of {covp['2004']['panel']} panel institutions have both versions). T3 uses "
             f"{P3n['insts']} of the panel's {ctx['t3info']['panel_insts']} institutions; with cohort 2001 it has "
             f"{P4n['cells']} of the {ctx['t3info']['full_cells']} field × cohort cells behind scripts/59's slope "
             f"({P3n['cells']} in the primary cohorts 2004-2010). CONTROL (of each version) and the 2026 state earnings "
             "level enter only the control + state-level sensitivity. The PSEO panel institutions are a different, "
             "public-heavy set, so rank-stability numbers on them are not comparable with those on the Scorecard "
             "institutions.")
    o.append("- **Margins.** The equivalence readings depend on δ (see Method, Pre-specification). At the numbers as "
             "written, T3 uses a margin set from the raw slope, which is lenient for a comparison of partial slopes; "
             "at one third of the 2026-control partial slope, T3 is inconclusive. An equivalence reading bounds the "
             "difference by δ; it does not show that the two sets of controls give the same answer.")
    o.append(f"- **Variance.** If the bootstrap variance of a field's value matches its sampling variance and shared "
             "institutions make fields covary non-negatively, each one-way part of the CGM variance is at least the "
             "field × institution part and the two-way variance is at least either one-way variance. In these data "
             f"the estimated V_field×institution exceeds V_field or V_institution for {nbind} of the {len(allr)} "
             f"T1a/T1b/T3 statistics (so V_institution is the largest part for {nbind_vs} of them and V_field for "
             f"{nbind_vf}), and the two-way variance is then below a one-way variance. The pre-specified readings use "
             "the two-way variance; the max-var columns show what changes under the conservative one.")
    o.append("- **Multiplicity.** Three differences are primary (T1a broad, T1b main, T3 slope; all E − 26). The "
             "other E − 26 and W − 26 rows (ladder steps, strict, wide window, cohort 2001, control + state level) "
             "are pre-specified sensitivities, reported without adjustment for multiple comparisons; "
             f"{len([x for x in steps_diff if x != 'broad'])} of the three non-primary ladder steps (E − 26) "
             f"{'has' if len([x for x in steps_diff if x != 'broad']) == 1 else 'have'} a 95% CI that excludes 0.")
    o.append("- **Bootstrap conditioning.** Resamples in which a partial correlation is undefined (in a small cell, "
             "the few drawn institutions ordered identically by prestige and a control) are redrawn, so the variances "
             "are conditional on definability within each bootstrap group (redraw counts in Method; at most "
             f"{max_red:.1%} of the shared draws of any group, in {max(shared_red, key=shared_red.get)}). "
             "Near-degenerate resamples that remain (few residual degrees of freedom) give extreme partial "
             "correlations and widen the intervals rather than narrow them. Monte Carlo error: with 1000 replicates "
             "per draw type the CI endpoints carry a Monte Carlo SE (CSV notes); readings within 2 Monte Carlo SEs of "
             "a threshold are flagged.")
    o.append("- **Undated geography.** The state earnings level (scripts/55) comes from the 2026 file in every "
             "version, STABBR is backfilled with the most recent value in every historical file, and the "
             "PSEO-to-UNITID match uses the 2026 file's OPEIDs; only SAT_AVG, ADM_RATE, PCTPELL and CONTROL are "
             "re-dated.")
    o.append("- **Scope.** A null here says only that re-dating the same three proxies does not change the "
             "estimates beyond the stated MDEs. It does not show that the proxies capture student quality. "
             f"The common sample keeps the {cov['okC']} of {cov['insts']} Scorecard analysis institutions with the "
             f"SAT family complete under both versions; {no_sat26} have no fall-2024 SAT_AVG (and are outside "
             f"scripts/55's sample too) and {no_satE} none in falls 2014-15.")
    o.append("")
    # ---------------- provenance ----------------
    pv = ctx["prov"]
    o.append("## Data and provenance")
    o.append("")
    o.append(f"- `data/raw/scorecard_hist/College_Scorecard_Raw_Data_06102026.zip` ({ZIP_SIZE:,} bytes, md5 "
             f"`{pv['zip_md5']}` = server ETag), from `{ZIP_URL}` (HTTP Last-Modified {ZIP_LASTMOD}), accessed "
             f"{ACCESSED}. Read member by member with usecols {', '.join(HIST_COLS)}; never extracted. Its "
             f"Most-Recent institution and field-of-study files are byte-identical to the ones scripts/55 reads "
             f"(md5 `{pv['inzip_mr_md5']}`, `{pv['inzip_fos_md5']}`). {pv['n_merged']} MERGED members; CRC-32 and "
             "sizes in the CSV (section provenance).")
    o.append(f"- Documentation from `{DOC_URL}` (Last-Modified {DOC_LASTMOD}): "
             + "; ".join(f"`{f}` ({sz:,} bytes, md5 `{m}`)" for f, (m, sz) in DOCS.items())
             + ". Quoted pages are asserted at run time (CSV rows `doc_quote`). Entries in `data/raw/SOURCES.md` "
               "(section 10d).")
    o.append("- Reused without edits: scripts/55 (samples, partial_rank), scripts/58 (fe_build, fe_sample), scripts/59 "
             "(twoway, weighted ranks, V4.14.1 release plumbing), scripts/52 (fixed-cohort panel).")
    o.append("")
    # ---------------- revisions ----------------
    g_sec = RG[("T3", "P3", "secE-26")]
    r_sec = g_sec["r"]
    o.append("## Revisions")
    o.append("")
    o.append("Changes after an independent verification of the first version (2026-09-25). The verifier re-computed "
             "the T1a/T1b/T3 point estimates and the bootstrap with its own code and seeds; its checks reproduced "
             "the points exactly. Each item below was checked against the data before it was applied; none changes "
             "a point estimate, the pre-specification block, or a pre-specified (two-way, numeric-margin) reading.")
    o.append("")
    o.append(f"1. **T3 margin (major).** The reading rule's text (one third of the 2026-control estimate) and its T3 "
             f"number (0.010/yr, one third of scripts/59's raw slope) disagree; the first version kept the number and "
             f"called the text loose wording. Now both are reported: ±{d3:g}/yr gives \"{short(g3['num'])}\", "
             f"±{g3['alt_lab']}/yr (one third of the 2026-control partial slope {f_(s26, 4)}) gives "
             f"\"{short(g3['alt_r'])}\"; relative to that slope the 90% CI of E − 26 is {pct(rel_lo)} to "
             f"{pct(rel_hi)}. The first version's headline gave only the numeric-margin count ({cnt['num']} of 3 "
             f"equivalent); the answer now gives {cnt['num']} of 3 at the numeric margins and {cnt['alt_r']} of 3 at "
             f"the same-cells margins, and the "
             f"T3 answer now leads with E − raw ({f_(T3[('P3', 'E-raw')]['est'], 4)} {ci_(T3[('P3', 'E-raw')], 4)}; "
             f"partial slope {ratio_t3:.2f} of raw). The same two margins are reported for every E − 26 / W − 26 row.")
    o.append(f"2. **Two-way variance below a one-way part.** The verifier found that for the T1b level betas V_inst < "
             f"V_field×inst, so the two-way variance is below the field-jackknife variance. In this run that holds for "
             f"{n_fe_vi} of the {len(fe_lev)} level betas ({andj([FE_LAB[k_] for k_ in FE_LEV_KEYS if FE[k_]['tvs'] < FE[k_]['tvi']])}), "
             + (f". The era-matched main-spec CI {ci_(mE, 4)} excludes 0 only under the two-way variance (conservative "
                f"{cci_(mE, 4)})" if mE["tlo"] > 0 and mE["clo"] <= 0 else f". Era-matched main spec: two-way "
                f"{ci_(mE, 4)}, conservative {cci_(mE, 4)}")
             + f". Checking all rows showed "
             f"the conservative variance binds in {nbind} of {len(allr)} statistics, mostly through V_institution "
             f"({nbind_vs}); it is now reported for every row (CSV columns `*_maxvar`, table column max-var 95% CI). "
             "Answer 3 now says that neither main-spec level CI clearly excludes 0.")
    o.append(f"3. **Knife-edge sensitivity.** The T3 control + state-level E − 26 reading at ±{d3:g} depends on a 90% "
             f"lower bound of {r_sec['lo90']:+.4f} against −{d3:g} with a Monte Carlo SE of {r_sec['mc90']:.2g}; the "
             "verifier's independent bootstrap put it inside the margin. It is now flagged as on the margin within "
             "Monte Carlo error rather than presented as a contrast with the primary; every reading is checked the "
             "same way.")
    o.append("4. **STABBR is not re-dated.** The data dictionary marks STABBR in every MERGED file as backfilled with "
             "the most recent IPEDS value (asserted in the script), so \"0 programs whose state differs\" holds by "
             "construction. STABBR was dropped from the list of re-dated variables; CONTROL is dated, and its 0 "
             "differences are an empirical result. The conclusion that only PCTPELL changes in the Pell + control + "
             "state step is unchanged.")
    o.append(f"5. **ADM_RATE stability across institution sets.** The first version compared the Scorecard "
             f"institutions ({f_(adm['est'], 2, False)}) with the PSEO entry windows ({f_(min(ps_adm), 2, False)}-{f_(max(ps_adm), 2, False)}) "
             f"as if the gap were the era. On the same falls 2001-2008 the Scorecard institutions give "
             f"{f_(min(fos_adm_08), 2, False)}-{f_(max(fos_adm_08), 2, False)} and the PSEO institutions "
             f"{f_(min(pse_adm_08), 2, False)}-{f_(max(pse_adm_08), 2, False)}; the text now says so.")
    o.append("6. **CSV completeness.** The first version's header claimed every number was in the CSV, but sample "
             "counts, cohort splits, panel coverage, the state/control difference counts and the ratios computed in "
             "the text were not, and n / k were empty on the T1a, T1b and T3 rows. These are now in sections "
             "`coverage` and `derived`, n and k are filled, and every row has a `status`.")
    o.append("7. **Exploratory labels.** T1b (a) − main, T3 26 − raw, the T3 per-horizon couplings (y1/y5/y10) and "
             "the T1a raw − v rows are not in the pre-specification and are now labelled exploratory in the table and "
             "the CSV. The W − 26 rows stay pre-specified sensitivities (the block lists \"W controls in T1\" and "
             "\"E - 26 and W - 26 differences\"). \"Equivalent\" is no longer paraphrased as \"no difference\".")
    o.append("8. **Start of SAT_AVG / ADM_RATE.** The institution documentation (p. 12) supports only the SAT start "
             "(2001-02). The ADM_RATE start is now cited from the data dictionary (empty for MERGED_2000-01, "
             f"\"{dic['adm_2001']}\" for MERGED_2001-02) and checked in MERGED2000_01, which exists but has "
             f"{m0['sat']} SAT_AVG and {m0['adm']} ADM_RATE values.")
    o.append("9. **Time zones.** The build-history times are now in UTC (the machine's local time was CEST); the "
             "download times were already in UTC.")
    o.append("")
    return "\n".join(o) + "\n"


def write_md(ctx):
    OUT_MD.write_text(render(ctx))
    print(f"[md] {OUT_MD}")


if __name__ == "__main__":
    main()
