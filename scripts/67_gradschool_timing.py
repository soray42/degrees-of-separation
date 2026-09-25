"""Graduate-school timing -- is the career-time rise in coupling a mechanical by-product of graduate study?

QUESTION (referee problem). Within-field coupling rho_f(h) = Spearman over institutions of field prestige
and graduates' median earnings h years after the bachelor's rises with h on fixed PSEO cohorts
(scripts/52, scripts/59: +0.030/yr on V4.14.1). Graduates of higher-status institutions more often enter
PhD / JD / MD / MBA programmes. Enrollment lowers (or removes) their UI earnings at y1, and at y5 for PhD
students and medical residents, and completion then raises them. That alone would produce a rise in
coupling with years since graduation. This script runs five pre-specified checks of how much of the rise
such timing could account for, with public data only. Descriptive, not causal.

PRE-SPECIFICATION (fixed before any estimate below was computed; set down 2026-09-24 in this docstring;
the lane had seen only the earlier scripts' published numbers and, for T5, how many panel cells are
admissible under each reliability scenario -- no estimate of any test statistic)
  Panel. PSEO V4.14.1 (data/raw/pseo_2026q2/), bachelor's fixed cohorts 2001/2004/2007/2010, the balanced
  scripts/52 panel (same institutions at y1, y5, y10; n>=15 per field x cohort), built with scripts/52's
  functions through scripts/59's release switch (neither edited). Field career slope s_f = mean over the
  field's cohorts of the OLS slope of coupling on years {1, 5, 10}. Reproduction check: all-field mean
  equals scripts/59's V4.14.1 value 0.030382 (k=40).
  Field graduate-degree share g_f (ACS 2023 1-year PUMS persons, psam_pusa + psam_pusb): persons aged
  30-45 with SCHL >= 21 (bachelor's or higher) whose FOD1P (first field of bachelor's degree) maps to f;
  g_f = PWGTP-weighted share with SCHL in {22, 23, 24} (master's, professional, doctorate). FOD1P -> field
  map: src/crosswalks.fields.fod1p_by_key_all() (the project's collision-free map) for the fields it
  covers, and EXTRA_FOD1P below for the six recovered fields it does not cover (documented in the report,
  with a flag where the FOD1P code is broader than the field or shared by two fields). Standard errors of
  g_f from the 80 ACS replicate weights (successive-difference formula, 4/80 * sum of squared deviations).
  Inference standard: the two-way (field, institution) cluster variance of scripts/59 (V_field + V_inst -
  V_field x inst; scripts/59.twoway, imported), from the between-field variance (jackknife over fields for
  non-linear statistics) and two institution multinomial draws of NB replicates each (one shared by every
  cell of every field; one independent per field). Normal 95% CI est +/- 1.96 SE. Status 'holds' / 'edge'
  / 'does not hold' as in scripts/59 (edge = deciding endpoint within 2 Monte Carlo SE of 0).
  MDE = (1.96 + 0.8416) x SE (80% power, 5% two-sided); power at the stated benchmark from the normal
  approximation.

  T1 (field level). OLS slope beta of s_f on g_f across the fields with a share. Pre-specified intervals:
  field bootstrap (NB_FIELD resamples of fields, percentile CI) and a permutation p (NPERM shuffles of g
  across fields, two-sided); the two-way CI is reported beside them. Secondary: Spearman(s_f, g_f) with a
  permutation p. Benchmark: beta_full = mean(s) / mean(g), the slope if every field's rise were
  proportional to its graduate-degree share (no rise at share 0); power also at beta_full / 2.
  Verdict: 'confound signature present' if the field-bootstrap CI lies above 0 and permutation p < 0.05;
  'opposite' if the CI lies below 0; otherwise 'not detected' when MDE <= beta_full / 2 and 'underpowered'
  when MDE > beta_full / 2.
  T2 (institution level). Partial Spearman coupling given same-horizon PSEO coverage (share of graduates
  with released earnings; scripts/52's definition) on the coverage-complete sample; slope of partial
  coupling vs slope of raw coupling on the same sample, their difference and the share removed
  (1 - partial/raw). Must reproduce scripts/59's V4.14.1 partial slope 0.024574. PhD production per
  bachelor's graduate: used only if public NCSES tables give doctorate counts by baccalaureate institution
  for the panel's institutions (see amendment A1 for the check and the construction).
  Verdict: 'rise survives the control' if the partial slope's CI lies above 0.
  T3. The y5->y10 segment slope (rho(y10) - rho(y5))/5, mean over the lower half of the fields by g_f
  (the floor(K/2) fields with the lowest share; ties by field key), next to the upper half and the
  difference. Benchmark: the all-field y5->y10 slope. Verdict: 'survives' if the lower-half CI lies above
  0; 'underpowered' if the CI includes 0 and MDE > benchmark; 'not detected' if the CI includes 0 and
  MDE <= benchmark.
  T4. The calendar-matched contrast D_cal = (rho(c, y10) - rho(c+9, y1)) / 9 on triple-matched
  institutions (scripts/52/59 construction) recomputed without cells whose earnings years fall in
  2008-2010 or 2020-2022. Earnings years of cohort c (graduated c..c+2): y1 c+1..c+3, y10 c+10..c+12.
  Primary: drop cohort pairs whose two contrast cells (c, y10), (c+9, y1) have earnings in those years
  (drops c = 2010, whose cells are 2020-22) -> pairs 2001, 2004, 2007. Strict: also drop pairs whose third
  triple cell (c, y1) does (drops c = 2007, y1 in 2008-10) -> pairs 2001, 2004, with the within-cohort and
  cross-cohort contrasts and the bound. Benchmark: D_cal on all four pairs. Verdict: 'lower end survives'
  if the D_cal CI lies above 0.
  T5. The F/G decomposition of the rise (per cell, standardized rank regression of earnings on F and G;
  slopes dF, dG of the two coefficients over years; scripts/52 item 5) with F and G disattenuated by the
  per-field reliabilities of scripts/56 (data/interim/prestige_reliability.csv): LB (public-edge
  split-half, lower bound for the published rank) and EXT (extrapolated to the full hire count).
  rho_F* = rho_F / sqrt(rel_F), rho_G* = rho_G / sqrt(rel_G), r_FG* = r_FG / sqrt(rel_F rel_G);
  b_F* = (rho_F* - r_FG* rho_G*) / (1 - r_FG*^2), b_G* symmetric. (a) Per-cell form (as scripts/52): cells
  with r_FG*^2 >= 1 are inadmissible (the assumed reliabilities cannot hold there) and are dropped; the
  uncorrected decomposition on the same admissible cells is shown beside it. In a replicate, an admissible
  cell whose replicate r_FG*^2 >= 1 is missing for that replicate (field value = mean over its remaining
  cells). (b) Pooled form: the decomposition of the across-field mean corrected correlations
  (mean_f rho_F*, mean_f rho_G*, mean_f r_FG* at each horizon), defined whenever the mean r_FG* < 1.
  Both forms are reported for LB and EXT; neither is designated the single primary. Verdict per
  scenario and form: dG* CI > 0, dF* CI, dG* - dF* CI > 0.

AMENDMENT A1 (2026-09-24, made while building T2 and before any T2 estimate was computed). The first
  draft of this docstring (interrupted run) said the PhD control could not be built because the NSF 22-321
  tables list only the top 50 baccalaureate institutions. The NCSES Interactive Data Tool (Survey of Earned
  Doctorates; library measure nTPpz 'Doctorate Recipients by Baccalaureate Institution' = Sum(mRecpBacInst); label
  as read from the tool's engine, corrected in A3) releases research-doctorate counts by baccalaureate institution
  (IPEDS UnitID) and doctorate year for every institution; counts of 1 are released, so small cells are not
  suppressed. The query fetches doctorate years 2000-2024. The control is therefore built, as T2 pre-specified:
    D_i(c) = research doctorates awarded in doctorate years c+5 .. c+12 to people whose bachelor's
             institution is i (all doctorate fields; SED counts summed over the IPEDS UnitIDs that the
             College Scorecard institution file lists under i's 8-digit OPEID, which is PSEO's institution id);
    BA_i(c) = PSEO V4.14.1 y1_ipeds_count of i's all-programme bachelor's row for cohort c (graduation years
             c..c+2; the same IPEDS count scripts/52 uses for coverage);
    PhD rate r_i(c) = (D_i(c) / 8) / (BA_i(c) / 3) = doctorates per year per bachelor's graduate per year.
  The window starts 3 years after the cohort's last graduation year and ends 10 years after its first
  (SED medians of years from bachelor's to doctorate lie in this range); it is an institution-level
  proxy, cohort-matched, not a count of this cohort's own doctorates. Sample: the coverage-complete panel
  rows with a defined rate (cells re-formed, n>=15). Statistics: slope of raw coupling, of partial coupling
  given r (all horizons), and of partial coupling given coverage and r together (second-order partial
  Spearman), each with raw minus partial and the share removed. Verdict as for coverage: 'rise survives
  the control' if the partial slope's CI lies above 0. Professional degrees (JD, MD, MBA) have no public
  institution-level counts and are covered only through the ACS field shares of T1/T3.

AMENDMENT A2 (2026-09-25, made when the run was resumed after the interruption and before any estimate of any
  test statistic had been computed or seen). The lane's task statement words T5 as "the F/G decomposition of the
  rise with F disattenuated at the reliability bracket of scripts/56". The docstring above corrects F and G
  (G by its own scripts/56 reliability, 0.97-1.00 by field) and listed the F-only correction as exploratory E6.
  To match the task wording, the F-only correction (LBFo, EXTFo: F corrected, G taken as measured) is reported as
  part of T5, beside the F-and-G correction; neither is designated the single primary, and E6 is withdrawn as a
  separate label. The md5 values of the SED query files are pinned in SED_MD5 (re-fetch record: amendment A3).
  The FOD1P flag of teacher_ed_subjects is changed from 'broader' to 'clean' (the ten FOD1P teacher-education
  codes are the CIP 13.12/13.13 programmes; the shares are unchanged).

AMENDMENT A3 (2026-09-25, made when this script was resumed a second time: items (i)-(iii) before this session
  computed any estimate, item (iv) after a first run whose test statistics were not looked at. A previous session had
  run an earlier version of the script (outputs written 2026-09-25 00:26 local time); those outputs were not read,
  and every number in the report comes from the re-run).
  (i) The PSEO all-programme bachelor's row (agg_level_pseo 44; cip_level A) that gives BA_i(c) carries an IPEDS
  count status flag (LEHD schema label_flags_ipeds_count.csv: 1 = as reported, 2 = edited for consistency with PSEO,
  3 = not available, 4 = partially missing). With flag 4 the count is a partial sum and the PhD rate is overstated.
  The primary construction of A1 is unchanged (flag-3 rows have no count, hence no rate). A sensitivity re-forms
  the PhD-rate cells on institution x cohort rows with flag 1 only (labelled 'sensitivity' in T2; same statistics
  and verdict rule); the flag distribution of the panel rows is reported. (ii) The re-fetch record: the SED files
  were fetched at 2026-09-24 22:11:30 UTC (sed_bacc_origin_query.json); a re-fetch by this run at 22:46:32 UTC gave
  byte-identical rows and totals files (md5 in SED_MD5). An earlier note of a re-fetch at 22:18 UTC could not be
  verified and is withdrawn. (iii) The PSEO Flows rows used in E4 are the cohort-specific aggregation levels 46
  (institution x CIP-2 x cohort, all sectors) and 94 (the same by NAICS sector); uniqueness is asserted.
  (iv) Institution draws are made per cell type (fixed-cohort, coverage, NAICS-61, PhD-rate, PhD-rate flag-1,
  calendar), each shared by every cell of that type in every field (and, for the independent draw, by the field's
  cells of that type); a replicate is redrawn only when a statistic of that type is undefined. Every statistic
  combines cells of one type, so each is conditioned only on its own cells. (A first run of this version with one
  draw for all six types redrew 391 of 2000 shared replicates, against 88 of 1000 for scripts/59's three types; its
  test statistics were not looked at before this change.) (v) A1's name of the SED measure is corrected to the label
  read from the tool's engine on 2026-09-24 ~23:20 UTC, and A1's unverified year range and table code are replaced
  by the years actually fetched (2000-2024).

EXPLORATORY (not pre-specified; labelled as such in the report)
  E1 T1 with the professional-or-doctorate share (SCHL 23-24) and the doctorate share (SCHL 24).
  E2 T1 on the two segments (y1->y5, y5->y10) separately.
  E3 across fields, Spearman(g_f, y1 coupling level).
  E4 T2 with a second time-varying control: the share of employed graduates working in educational
     services (NAICS 61) at the same horizon, from PSEO Flows V4.14.1 (institution x CIP-2 family x cohort;
     the family of the field's first CIP-4 code), alone and together with coverage.
  E5 the y1->y5 segment in the lower half of fields by g_f.
  E6 (withdrawn by A2: the F-only correction is part of T5.)
  E7 (added with A1) the institution-level footprint of graduate study: mean within-cell Spearman of the
     PhD rate with earnings at y1, y5, y10 and its slope over years; Spearman of the PhD rate with F and G.
  E8 (added after the T5 estimates were seen) IQR-based two-way intervals (scripts/59.twoway tse_r) for the
     per-cell corrected T5 statistics, whose replicates are heavy-tailed; the pre-specified normal intervals are
     reported as computed.

REVISION 1 (2026-09-25, after an independent verification of the first version; every result had been seen, so
  everything added here is exploratory and labelled so). No pre-specified statistic, sample, seed or verdict rule is
  changed.
  E2 (extended) the y5->y10 segment slope on the professional-or-doctorate share (SCHL 23-24) and on the doctorate
     share (SCHL 24): the y5 depression the referee describes comes from doctoral study and residency.
  E9 what T3 can and cannot separate. Proportional mechanism: the late rise proportional to g_f with no rise at
     g = 0, slope b_late = mean(late)/mean(g) (E2's benchmark); predicted lower-half, upper-half and upper-minus-lower
     late slopes = b_late x the half's mean g. Observed minus predicted with the T3 two-way SEs, and the power of a
     5% two-sided test to separate the proportional mechanism from a uniform late rise (lower half: distance
     between the all-field late slope and the proportional prediction; difference: the predicted difference vs 0).
  E10 the IPEDS count flag of the bachelor's denominator. (a) Within institutions: OLS of log BA_i(c) on institution
     and cohort indicators and 1[flag 4] over the panel institutions' institution x cohort rows with a count
     (institution-clustered CR1 SE; variant with institution indicators only). (b) Stress test on the full PhD-rate
     sample: flag-4 counts divided by exp(b_lo), b_lo = b - 1.96 SE (the largest shortfall of flag-4 counts inside
     the 95% interval), rate recomputed, the T2 PhD-rate statistics re-estimated with their own institution draws
     (cell type 'phda'). A first draft of this revision used exp(b) itself; b turned out close to 0 (seen in the
     run log before any E10(b) statistic was looked at), which would have made (b) uninformative, so the interval
     endpoint is used instead. (c) The flag-1 rows keep the full-sample rate (checked: number of rows whose flag-1
     rate differs from the full-sample rate).
  E11 T1 with the two fields that share FOD1P 4101 (Kinesiology, Health/PE/Recreation; same g): beta without
     Kinesiology, without Health/PE/Recreation, and with the two merged into one point (mean career slope), each with
     the field bootstrap, the permutation p and the two-way interval of T1, and the T1 verdict rule applied.
  Averaging: the within-cell Spearman correlations of prestige with coverage (T2) and with the NAICS-61 share (E4)
  are now means over fields of the field's mean over cells, as every other statistic here (the first version
  averaged over cells). The T1 field-bootstrap row reports the MDE and power the verdict uses (larger SE).
  Provenance: A3(ii) withdrew a note of a re-fetch at 22:18 UTC as unverifiable. The copy of that fetch was found
  (session scratch folder; query record accessed_utc 2026-09-24 22:18:45; rows and totals md5 equal to SED_MD5), so
  both re-fetches (22:18:45 and 22:46:32 UTC) are recorded as byte-identical.

Seeds: every replicate draws its randomness from its own child of numpy SeedSequence([SEED, crc32(tag)])
(.spawn), so results do not depend on the number of worker processes. Outputs byte-identical on re-run.
Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 python scripts/67_gradschool_timing.py
Outputs: data/interim/gradschool_timing.csv (every number), data/interim/gradschool_field_share.csv (field
         table), data/interim/gradschool_phd_rate.csv (institution x cohort PhD rates),
         outputs/figures/gradschool_timing.png, GRADSCHOOL_TIMING_RESULT.md (repo root, local only).
"""
from __future__ import annotations

import os
import sys
import gc
import json
import zlib
import hashlib
import warnings
import resource
import importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.crosswalks import fields as FX


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SEED = 67
s59 = _load("s59", "59_pseo_refresh.py")      # release switch, weighted ranks, two-way variance (not edited)
s52 = s59.s52                                # fixed-cohort panel builders (not edited)
FIELDS66, LAB = s52.FIELDS66, s52.LAB
HZ = ["y1", "y5", "y10"]
W_SLOPE = s52.W_SLOPE
COHORTS, LATE, NMIN = s52.COHORTS, s52.LATE, s52.NMIN
wrank, wcorr, twoway, jack_var = s59.wrank, s59.wcorr, s59.twoway, s59.jack_var
Z975 = s59.Z975
Z80 = 0.8416212335729143
MDE_K = Z975 + Z80
NB = 2000              # institution multinomial draws per kind (shared; independent per field)
NB_FIELD = 10000       # field bootstrap (T1)
NPERM = 10000          # field-label permutations (T1)
WORKERS = int(os.environ.get("GST_WORKERS", "4"))
AGE_LO, AGE_HI = 30, 45
SCHL_BA, SCHL_GRAD, SCHL_LONG, SCHL_PHD = ["21", "22", "23", "24"], ["22", "23", "24"], ["23", "24"], ["24"]
REF_CSV = ROOT / "data" / "interim" / "pseo_refresh.csv"            # scripts/59 output (V4.14.1 = variant 'new')
REF_STATS = {"slope_all": "slope_all", "late_all": "late_slope_all", "early_all": "early_slope_all",
             "dbF_all": "dbF_all", "dbG_all": "dbG_all", "D_cal_all": "D_cal_all", "D_within_all": "D_within_all",
             "D_cross_all": "D_cross_all", "cov_partial_all": "cov_partial_all"}
EXCL_YEARS = set(range(2008, 2011)) | set(range(2020, 2023))

ACS = [ROOT / "data" / "raw" / "acs" / "psam_pusa.csv", ROOT / "data" / "raw" / "acs" / "psam_pusb.csv"]
ACS_DICT = ROOT / "data" / "raw" / "acs" / "PUMS_Data_Dictionary_2023.txt"
REL_CSV = ROOT / "data" / "interim" / "prestige_reliability.csv"
FLOWS = ROOT / "data" / "raw" / "pseo_flows_2026q2" / "pseof_all.csv.gz"
SC_INST = ROOT / "data" / "raw" / "scorecard_inst" / "Most-Recent-Cohorts-Institution.csv"
# NCSES Interactive Data Tool, Survey of Earned Doctorates (Qlik Sense engine, anonymous access)
SED_APP = "ba83ce5c-1a4a-4b53-a54d-febd3b048a3f"
SED_WSS = f"wss://ncsesdata.nsf.gov/app/{SED_APP}"
SED_PAGE = "https://ncsesdata.nsf.gov/builder/sed"
SED_MEASURE = 'Sum({<Year={">=2000"}>} mRecpBacInst)'      # library measure nTPpz = Sum(mRecpBacInst), years >= 2000
SED_DIMS = ["IPEDSUnitID", "instName", "Year"]
SED_DIR = ROOT / "data" / "raw" / "ncses_sed_bacc"
SED_ROWS = SED_DIR / "sed_bacc_origin_unitid_year.csv"
SED_TOT = SED_DIR / "sed_bacc_origin_year_totals.csv"
SED_QUERY = SED_DIR / "sed_bacc_origin_query.json"
SED_MD5 = {"rows": "ed0fd0147d5ba5ef21146ec9188a4bc1",      # pinned; re-fetches at 2026-09-24 22:18:45 UTC and
           "totals": "7e4d5b43e40b4961b0912f29cc2b6c71"}    # 22:46:32 UTC were byte-identical (SOURCES.md 10g)
SED_LAG = (5, 12)                        # doctorate years c+5 .. c+12 for bachelor's cohort c (c..c+2)
SED_ACCESS = ("2026-09-24 22:11:30 UTC", "2026-09-24 22:18:45 UTC and 22:46:32 UTC")   # fetch; verified re-fetches
BUSINESS = ["accounting", "finance", "management", "marketing"]      # business fields (MBA timing note, Revision 1)
SHARED_4101 = ("kinesiology", "hper")                                 # fields sharing FOD1P 4101 (E11)
LEHD_FLAGS = ROOT / "data" / "raw" / "lehd_schema" / "label_flags_ipeds_count.csv"   # IPEDS count status labels
OUT_CSV = ROOT / "data" / "interim" / "gradschool_timing.csv"
OUT_FIELD = ROOT / "data" / "interim" / "gradschool_field_share.csv"
OUT_PHD = ROOT / "data" / "interim" / "gradschool_phd_rate.csv"
OUT_FIG = ROOT / "outputs" / "figures" / "gradschool_timing.png"
OUT_MD = ROOT / "GRADSCHOOL_TIMING_RESULT.md"

# FOD1P codes for the six panel fields that fod1p_by_key_all() does not cover (recovered fields of scripts/24).
# flag: 'clean' = the FOD1P code is the field; 'broader' = the code covers more than the field; 'shared' = the
# same code is also used for another panel field (both then get the same share).
EXTRA_FOD1P = {
    "kinesiology": (["4101"], "shared",
                    "CIP 31.05 kinesiology lies in FOD1P 4101 'Physical Fitness Parks Recreation And Leisure' (CIP 31); "
                    "shared with Health/PE/Recreation"),
    "hper": (["4101"], "shared",
             "CIP 31.01/31.03 parks and recreation = FOD1P 4101; shared with Kinesiology"),
    "human_dev": (["2901"], "broader",
                  "CIP 19.07 lies in FOD1P 2901 'Family And Consumer Sciences' (all of CIP 19 except nutrition)"),
    "nutrition": (["4002"], "clean", "FOD1P 4002 'Nutrition Sciences'"),
    "spanish": (["2602"], "broader",
                "CIP 16.09 lies in FOD1P 2602 'French German Latin And Other Common Foreign Language Studies'"),
    "teacher_ed_subjects": (["2304", "2305", "2306", "2307", "2308", "2309", "2311", "2312", "2313", "2314"], "clean",
                            "CIP 13.12/13.13 (levels and subject areas) = the teacher-education FOD1P codes 2304-2309 "
                            "and 2311-2314; excluded: 2310 special needs (the Special Education field) and 2300/2301/"
                            "2303/2399 (general education, administration, counseling, miscellaneous)"),
}


def stage(msg: str):
    print(f"[67] {msg} (maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB)", flush=True)


def seeds(tag: str, n: int) -> list:
    """One child SeedSequence per replicate (independent of how replicates are split across workers)."""
    return np.random.SeedSequence([SEED, zlib.crc32(tag.encode())]).spawn(n)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


RES: list[dict] = []


def rec(section, stat, est, sample="", se=np.nan, lo=np.nan, hi=np.nan, ci_kind="", z=np.nan, p=np.nan,
        p_kind="", k=np.nan, mde=np.nan, power=np.nan, bench=np.nan, mcse=np.nan, status="", note=""):
    RES.append(dict(section=section, stat=stat, sample=sample, estimate=est, se=se, ci_lo=lo, ci_hi=hi,
                    ci_kind=ci_kind, z=z, p_value=p, p_kind=p_kind, k=k, mde80=mde, power=power, benchmark=bench,
                    mc_se_endpoint=mcse, status=status, note=note))


def power_at(b: float, se: float) -> float:
    if not (np.isfinite(b) and np.isfinite(se) and se > 0):
        return np.nan
    return float(norm.cdf(abs(b) / se - Z975) + norm.cdf(-abs(b) / se - Z975))


def status_pos(lo, hi, mcse) -> str:
    tol = 2 * mcse if np.isfinite(mcse) else 0.0
    if lo > tol:
        return "holds (CI > 0)"
    if abs(lo) <= tol:
        return "edge (lower end within 2 MC SE of 0)"
    if hi < -tol:
        return "opposite (CI < 0)"
    return "does not hold (CI includes 0)"


def z_p(z: float) -> float:
    return float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan


def ols_slope(x, y) -> float:
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


# =============================================================================================
# (1) ACS 2023 field graduate-degree shares
# =============================================================================================
def acs_file(path: str):
    """Worker: weighted counts (PWGTP and the 80 replicate weights) by FOD1P x SCHL for persons aged 30-45
    with a bachelor's degree or higher. Streams the file in 4 MB blocks (only 84 columns parsed)."""
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.compute as pcm
    wcols = ["PWGTP"] + [f"PWGTP{i}" for i in range(1, 81)]
    cols = ["AGEP", "SCHL", "FOD1P"] + wcols
    types = {c: pa.int64() for c in ["AGEP"] + wcols}
    types.update(SCHL=pa.string(), FOD1P=pa.string())
    rd = pacsv.open_csv(path, convert_options=pacsv.ConvertOptions(include_columns=cols, column_types=types),
                        read_options=pacsv.ReadOptions(block_size=1 << 22, use_threads=False))
    parts, nrows = [], 0
    ba = pa.array(SCHL_BA)
    for b in rd:
        t = pa.Table.from_batches([b])
        nrows += t.num_rows
        m = pcm.and_(pcm.and_(pcm.greater_equal(t.column("AGEP"), AGE_LO), pcm.less_equal(t.column("AGEP"), AGE_HI)),
                     pcm.is_in(t.column("SCHL"), value_set=ba))
        t = t.filter(m)
        if t.num_rows:
            d = t.to_pandas()
            d["n"] = 1
            parts.append(d.groupby(["FOD1P", "SCHL"], dropna=False)[["n"] + wcols].sum())
        del t, b
    out = pd.concat(parts).groupby(level=[0, 1], dropna=False).sum().reset_index()
    return out, nrows


def fod1p_labels() -> dict:
    lines = ACS_DICT.read_text(encoding="latin-1").splitlines()
    i = next(j for j, ln in enumerate(lines) if ln.startswith("FOD1P "))
    lab = {}
    for ln in lines[i + 2:]:
        if not ln.strip():
            break
        code, _, rest = ln.strip().partition(" ")
        lab[code] = rest.strip().lstrip(".")
    return lab


def field_fod1p_map(fields: list[str]) -> dict:
    base = FX.fod1p_by_key_all()
    out = {}
    for f in fields:
        if f in base:
            out[f] = (list(base[f]), "project map", "src/crosswalks.fields.fod1p_by_key_all()")
        elif f in EXTRA_FOD1P:
            out[f] = EXTRA_FOD1P[f]
    used = {}
    for f, (codes, _, _) in out.items():
        for c in codes:
            used.setdefault(c, []).append(f)
    for c, fs in used.items():
        if len(fs) > 1:
            assert all(out[f][1] == "shared" for f in fs), (c, fs)
    return out


def acs_aggregate():
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(acs_file, [str(p) for p in ACS]))
    agg = pd.concat([r[0] for r in res]).groupby(["FOD1P", "SCHL"], dropna=False).sum().reset_index()
    agg = agg.sort_values(["FOD1P", "SCHL"]).reset_index(drop=True)
    info = dict(acs_person_rows=int(sum(r[1] for r in res)), acs_sample_ba_plus_30_45=int(agg["n"].sum()),
                acs_weighted_ba_plus_30_45=float(agg["PWGTP"].sum()),
                acs_missing_fod1p=int(agg.loc[agg.FOD1P.isna(), "n"].sum()))
    return agg, info


def acs_shares(agg: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    fm = field_fod1p_map(fields)
    lab = fod1p_labels()
    wcols = ["PWGTP"] + [f"PWGTP{i}" for i in range(1, 81)]
    rows = []
    for f in fields:
        if f not in fm:
            continue
        codes, flag, note = fm[f]
        a = agg[agg.FOD1P.isin(codes)]
        tot = a[wcols].sum().to_numpy(float)
        r = dict(field=f, label=LAB.get(f, f), fod1p=" ".join(codes),
                 fod1p_labels="; ".join(f"{c} {lab.get(c, '?')}" for c in codes), map_flag=flag, map_note=note,
                 n_ba_plus=int(a["n"].sum()), w_ba_plus=float(tot[0]))
        for name, codes_s in (("grad", SCHL_GRAD), ("long", SCHL_LONG), ("phd", SCHL_PHD), ("ma", ["22"])):
            num = a[a.SCHL.isin(codes_s)][wcols].sum().to_numpy(float)
            sh = num / tot
            r[f"share_{name}"] = float(sh[0])
            r[f"se_{name}"] = float(np.sqrt(4.0 / 80.0 * ((sh[1:] - sh[0]) ** 2).sum()))
            r[f"n_{name}"] = int(a[a.SCHL.isin(codes_s)]["n"].sum())
        rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================================
# (2) PSEO Flows: share of employed graduates in educational services (NAICS 61)   [exploratory E4]
# =============================================================================================
def flows_worker(path: str) -> pd.DataFrame:
    s61 = _load("s61", "61_flows_placement.py")      # flows reader (not edited); cohorts set for this read
    s61.COHORTS = list(COHORTS)
    sec, tot = s61.read_flows(Path(path))
    ed = sec[(sec.industry == "61") & sec.cohort.isin(COHORTS)].drop(columns="industry")
    tt = tot[tot.cohort.isin(COHORTS)]
    # cohort-specific rows are aggregation levels 94 (by sector) and 46 (all sectors) only: one row per key (A3)
    assert not ed.duplicated(["institution", "cip2", "cohort"]).any()
    assert not tt.duplicated(["institution", "cip2", "cohort"]).any()
    m = ed.merge(tt, on=["institution", "cip2", "cohort"], suffixes=("_61", "_all"), how="inner")
    keep = ["institution", "cip2", "cohort"] + [f"emp_{h}_61" for h in HZ] + [f"emp_{h}_all" for h in HZ]
    return m[keep].reset_index(drop=True)


def flows_shares(ins: pd.DataFrame) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("fork")) as ex:
        m = ex.submit(flows_worker, str(FLOWS)).result()
    m = m.merge(ins, on="institution", how="left")
    m = m[m.inst_key.notna() & (m.inst_key != "")]
    out = None
    for h in HZ:
        a, t = f"emp_{h}_61", f"emp_{h}_all"
        x = m[m[a].notna() & m[t].notna() & (m[t] > 0)].groupby(["inst_key", "cip2", "cohort"])[[a, t]].sum()
        x = x.assign(**{f"n61_{h}": x[a] / x[t]})[[f"n61_{h}"]].reset_index()
        out = x if out is None else out.merge(x, on=["inst_key", "cip2", "cohort"], how="outer")
    return out.rename(columns={"cohort": "grad_cohort"}).sort_values(["inst_key", "cip2", "grad_cohort"]).reset_index(drop=True)


# =============================================================================================
# (3) NCSES SED: research doctorates by baccalaureate institution (Interactive Data Tool)
# =============================================================================================
def sed_fetch():
    """One-time download through the Qlik Sense engine API behind the public NCSES table builder
    (anonymous session): hypercube IPEDSUnitID x instName x Year of SED_MEASURE, zero rows suppressed,
    plus the year totals of the same measure (consistency check). Writes SED_ROWS, SED_TOT, SED_QUERY."""
    import datetime
    import websocket
    SED_DIR.mkdir(parents=True, exist_ok=True)
    ws = websocket.create_connection(SED_WSS, header=["Origin: https://ncsesdata.nsf.gov"], timeout=300)
    hello = json.loads(ws.recv())
    ctr = [0]

    def call(method, handle, params):
        ctr[0] += 1
        ws.send(json.dumps(dict(jsonrpc="2.0", id=ctr[0], method=method, handle=handle, params=params)))
        while True:
            r = json.loads(ws.recv())
            if r.get("id") == ctr[0]:
                if "error" in r:
                    raise RuntimeError(r["error"])
                return r["result"]

    doc = call("OpenDoc", -1, [SED_APP])["qReturn"]["qHandle"]
    lib = call("GetMeasure", doc, ["nTPpz"])["qReturn"]["qHandle"]
    lib_def = call("GetProperties", lib, [])["qProp"]["qMeasure"]["qDef"]

    def cube(dims, meas):
        W = len(dims) + 1
        hc = {"qDimensions": [{"qDef": {"qFieldDefs": [d]}, "qNullSuppression": False} for d in dims],
              "qMeasures": [{"qDef": {"qDef": meas}}], "qSuppressZero": True, "qSuppressMissing": False,
              "qInitialDataFetch": [{"qTop": 0, "qLeft": 0, "qHeight": 0, "qWidth": W}]}
        h = call("CreateSessionObject", doc, [{"qInfo": {"qType": "hc"}, "qHyperCubeDef": hc}])["qReturn"]["qHandle"]
        ny = call("GetLayout", h, [])["qLayout"]["qHyperCube"]["qSize"]["qcy"]
        H = 10000 // W
        rows = []
        for top in range(0, ny, H):
            pg = call("GetHyperCubeData", h, ["/qHyperCubeDef", [{"qTop": top, "qLeft": 0, "qHeight": H, "qWidth": W}]])
            for r in pg["qDataPages"][0]["qMatrix"]:
                rows.append([c.get("qText") for c in r[:-1]] + [r[-1].get("qNum")])
        return pd.DataFrame(rows, columns=dims + ["n"]), ny

    t0 = datetime.datetime.now(datetime.timezone.utc)
    d, ny = cube(SED_DIMS, SED_MEASURE)
    tot, _ = cube(["Year"], SED_MEASURE)
    ws.close()
    assert len(d) == ny
    d["Year"] = d["Year"].astype(int); d["n"] = d["n"].round().astype(int)
    tot["Year"] = tot["Year"].astype(int); tot["n"] = tot["n"].round().astype(int)
    d = d.sort_values(SED_DIMS).reset_index(drop=True)
    tot = tot.sort_values("Year").reset_index(drop=True)
    assert (d.groupby("Year").n.sum().reindex(tot.Year).to_numpy() == tot.n.to_numpy()).all()
    d.to_csv(SED_ROWS, index=False, lineterminator="\n")
    tot.to_csv(SED_TOT, index=False, lineterminator="\n")
    SED_QUERY.write_text(json.dumps(dict(
        page=SED_PAGE, engine=SED_WSS, app=SED_APP, session_user=hello.get("params", {}).get("userDirectory"),
        library_measure_id="nTPpz", library_measure_def=lib_def, measure=SED_MEASURE, dimensions=SED_DIMS,
        suppress_zero=True, accessed_utc=t0.strftime("%Y-%m-%d %H:%M:%S"), rows=len(d)), indent=1) + "\n")


def sed_load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not (SED_ROWS.exists() and SED_TOT.exists()):
        sed_fetch()
    info = dict(rows_md5=md5(SED_ROWS), totals_md5=md5(SED_TOT), rows_bytes=SED_ROWS.stat().st_size,
                totals_bytes=SED_TOT.stat().st_size)
    if SED_MD5["rows"]:
        assert info["rows_md5"] == SED_MD5["rows"], info
        assert info["totals_md5"] == SED_MD5["totals"], info
    d = pd.read_csv(SED_ROWS, dtype={"IPEDSUnitID": str, "instName": str})
    tot = pd.read_csv(SED_TOT)
    assert (d.groupby("Year").n.sum().reindex(tot.Year).to_numpy() == tot.n.to_numpy()).all()
    info["query"] = json.loads(SED_QUERY.read_text()) if SED_QUERY.exists() else {}
    return d, tot, info


def pseo_ba_counts() -> pd.DataFrame:
    """PSEO V4.14.1 institution-level all-programme bachelor's rows (cip_level A, agg_level_pseo 44):
    y1_ipeds_count by cohort and its IPEDS count status flag (A3)."""
    use = ["agg_level_pseo", "inst_level", "institution", "degree_level", "cip_level", "grad_cohort",
           "y1_ipeds_count", "status_y1_ipeds_count"]
    parts = []
    for ch in pd.read_csv(s59.NEW_EARN, dtype=str, usecols=use, chunksize=200_000):
        parts.append(ch[(ch.inst_level == "I") & (ch.degree_level == "05") & (ch.cip_level == "A")
                        & ch.grad_cohort.isin(COHORTS)])
    d = pd.concat(parts, ignore_index=True)
    assert (d.agg_level_pseo == "44").all()
    d["BA"] = pd.to_numeric(d["y1_ipeds_count"], errors="coerce")
    d["BA_status"] = d["status_y1_ipeds_count"].astype(str)
    assert not d.duplicated(["institution", "grad_cohort"]).any()
    assert d.loc[d.BA_status == "3", "BA"].isna().all() and d.loc[d.BA_status != "3", "BA"].notna().all()
    return d[["institution", "grad_cohort", "BA", "BA_status"]]


def phd_rates(ins: pd.DataFrame, keys: set) -> tuple[pd.DataFrame, dict]:
    """PhD rate r_i(c) per panel institution x cohort (amendment A1)."""
    sed, tot, sinfo = sed_load()
    sc = pd.read_csv(SC_INST, usecols=["UNITID", "OPEID", "INSTNM"], dtype=str)
    ik = ins[ins.inst_key.isin(keys)][["institution", "inst_key"]].drop_duplicates()
    assert not ik.inst_key.duplicated().any() and not ik.institution.duplicated().any()
    u = ik.merge(sc, left_on="institution", right_on="OPEID", how="left")
    u6 = sed[sed.IPEDSUnitID.str.fullmatch(r"\d{6}")].copy()
    have = set(u6.IPEDSUnitID)
    ba = pseo_ba_counts()
    rows = []
    for (inst, key), g in u.groupby(["institution", "inst_key"], sort=True):
        ids = sorted(x for x in g.UNITID.dropna())
        s = u6[u6.IPEDSUnitID.isin(ids)]
        for c in COHORTS:
            y0, y1 = int(c) + SED_LAG[0], int(c) + SED_LAG[1]
            D = int(s[(s.Year >= y0) & (s.Year <= y1)].n.sum())
            bb = ba[(ba.institution == inst) & (ba.grad_cohort == c)]
            B = float(bb.BA.iloc[0]) if len(bb) and np.isfinite(bb.BA.iloc[0]) else np.nan
            st = str(bb.BA_status.iloc[0]) if len(bb) else "no row"
            rate = (D / (y1 - y0 + 1)) / (B / 3) if B > 0 else np.nan
            rows.append(dict(inst_key=key, institution=inst, unitids=" ".join(ids), n_unitid_in_sed=sum(i in have for i in ids),
                             sed_names="; ".join(sorted(set(s.instName))), grad_cohort=c, doc_years=f"{y0}-{y1}",
                             D=D, BA=B, BA_status=st, phd_rate=rate, phd_rate_s1=rate if st == "1" else np.nan))
    out = pd.DataFrame(rows)
    info = dict(sed=sinfo, sed_rows=len(sed), sed_years=f"{sed.Year.min()}-{sed.Year.max()}",
                sed_total=int(tot.n.sum()), sed_nonus=int(sed[sed.IPEDSUnitID == "Non-U.S. institution"].n.sum()),
                sed_unknown=int(sed[sed.IPEDSUnitID == "BA institution unknown or not reported"].n.sum()),
                sed_unitids=len(have), n_inst=out.inst_key.nunique(),
                n_inst_unitid=int(u.groupby("inst_key").UNITID.apply(lambda x: x.notna().any()).sum()),
                n_inst_in_sed=int(out.groupby("inst_key").n_unitid_in_sed.max().gt(0).sum()),
                multi_unitid=sorted(out[out.unitids.str.contains(" ")].inst_key.unique()))
    return out, info


def _ols_first(y, X, cl):
    """OLS coefficient on the first column of X, with conventional and cluster-robust (CR1, clusters cl) SEs."""
    Xi = np.linalg.pinv(X.T @ X)
    bet = Xi @ X.T @ y
    e = y - X @ bet
    n, rk = X.shape[0], int(np.linalg.matrix_rank(X))
    se_c = float(np.sqrt(Xi[0, 0] * (e @ e) / (n - rk)))
    meat = np.zeros((X.shape[1], X.shape[1]))
    for k_ in np.unique(cl):
        m_ = cl == k_
        u = X[m_].T @ e[m_]
        meat += np.outer(u, u)
    G_ = len(np.unique(cl))
    V = Xi @ meat @ Xi * (G_ / (G_ - 1)) * ((n - 1) / (n - rk))
    return float(bet[0]), float(np.sqrt(V[0, 0])), se_c


def flag4_shortfall(phd: pd.DataFrame) -> dict:
    """E10(a) (Revision 1): within-institution log difference of the bachelor's counts flagged 4 ('partially
    missing') against the same institution's other cohorts. OLS of log BA on 1[flag 4], institution and cohort
    indicators (and, as a variant, institution indicators only), over the panel institutions' institution x cohort
    rows with a count; conventional and institution-clustered (CR1) SE. b_lo = b - 1.96 CR1 SE, the largest shortfall
    of flag-4 counts within the 95% interval (used by E10(b))."""
    d = phd[phd.BA.notna() & (phd.BA > 0)].sort_values(["inst_key", "grad_cohort"]).reset_index(drop=True)
    y = np.log(d.BA.to_numpy(float))
    f4 = (d.BA_status == "4").to_numpy(float)
    I = pd.get_dummies(d.inst_key).to_numpy(float)
    C = pd.get_dummies(d.grad_cohort, drop_first=True).to_numpy(float)
    cl = d.inst_key.to_numpy()
    b, se_r, se_c = _ols_first(y, np.column_stack([f4, I, C]), cl)
    b_nc, se_nc, se_nc_c = _ols_first(y, np.column_stack([f4, I]), cl)
    both = d.groupby("inst_key").BA_status.agg(lambda x: {"1", "4"} <= set(x)).sum()
    flags = phd.BA_status.value_counts().sort_index().to_dict()
    return dict(b=b, se=se_r, se_conv=se_c, b_lo=b - Z975 * se_r, b_nc=b_nc, se_nc=se_nc, se_nc_conv=se_nc_c,
                n=len(d), n_inst=int(d.inst_key.nunique()), n_both=int(both), flags=flags, rows_all=len(phd))


# =============================================================================================
# (4) cells and weighted statistics
# =============================================================================================
def partial(rxy, rxz, ryz):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


CELLS: dict = {}          # kind -> list of dict(field, coh, arrays..., ix); set in main (inherited by fork workers)
CTX: dict = {}            # rel, pair_sets (set in main before any fork)


def ev_cell(kind: str, c: dict, W: np.ndarray) -> dict:
    Wc = W[:, c["ix"]]
    if kind == "fix":                                   # = scripts/59.w_fixed
        rP, rG = wrank(c["P"], Wc), wrank(c["G"], Wc)
        rE = [wrank(c["E"][:, k], Wc) for k in range(3)]
        rhoF = np.stack([wcorr(rP, r, Wc) for r in rE], -1)
        rhoG = np.stack([wcorr(rG, r, Wc) for r in rE], -1)
        rFG = wcorr(rP, rG, Wc)
        d = (1 - rFG ** 2)[:, None]
        d = np.where(d > 1e-12, d, np.nan)
        return dict(rhoF=rhoF, rhoG=rhoG, rFG=rFG, bF=(rhoF - rFG[:, None] * rhoG) / d,
                    bG=(rhoG - rFG[:, None] * rhoF) / d)
    if kind == "cov":                                   # = scripts/59.w_cov
        rP = wrank(c["P"], Wc)
        raw, par = [], []
        for k in range(3):
            rE, rC = wrank(c["E"][:, k], Wc), wrank(c["C"][:, k], Wc)
            a, b, cc = wcorr(rP, rE, Wc), wcorr(rP, rC, Wc), wcorr(rE, rC, Wc)
            raw.append(a); par.append(partial(a, b, cc))
        return dict(raw=np.stack(raw, -1), par=np.stack(par, -1))
    if kind in ("edu", "phd", "phd1", "phda"):         # second control N: NAICS-61 share (edu) or PhD rate (phd, phd1, phda)
        rP = wrank(c["P"], Wc)
        rN0 = wrank(c["N"], Wc) if c["N"].ndim == 1 else None
        out = {k: [] for k in ("raw", "pc", "pn", "pb", "rNE")}
        for k in range(3):
            rE, rC = wrank(c["E"][:, k], Wc), wrank(c["C"][:, k], Wc)
            rN = rN0 if rN0 is not None else wrank(c["N"][:, k], Wc)
            pe, pc_, pn_ = wcorr(rP, rE, Wc), wcorr(rP, rC, Wc), wcorr(rP, rN, Wc)
            ec, en, cn = wcorr(rE, rC, Wc), wcorr(rE, rN, Wc), wcorr(rC, rN, Wc)
            pe_c = partial(pe, pc_, ec)
            out["raw"].append(pe); out["pc"].append(pe_c); out["pn"].append(partial(pe, pn_, en))
            out["pb"].append(partial(pe_c, partial(pn_, pc_, cn), partial(en, ec, cn)))
            out["rNE"].append(en)
        return {k: np.stack(v, -1) for k, v in out.items()}
    if kind == "cal":                                   # = scripts/59.w_cal
        rP = wrank(c["P"], Wc)
        return dict(rho=np.stack([wcorr(rP, wrank(c["E"][:, k], Wc), Wc) for k in range(3)], -1))
    raise ValueError(kind)


def ev_all(W: np.ndarray, field=None, kinds=None) -> dict:
    return {kind: [ev_cell(kind, c, W) if (field is None or c["field"] == field) else None for c in cl]
            for kind, cl in CELLS.items() if kinds is None or kind in kinds}


def _bad(res: dict) -> np.ndarray:
    m = None
    for cl in res.values():
        for r in cl:
            if r is None:
                continue
            for x in r.values():
                b = ~np.isfinite(x.reshape(x.shape[0], -1)).all(1)
                m = b if m is None else (m | b)
    return m


def draw_kind(tag: str, n_inst: int, kind: str, field=None) -> tuple[list, int]:
    """NB institution multinomial draws for the cells of one cell type (kind); replicate b uses its own child
    seed. A replicate in which a cell statistic of this type (of the field's cells, or of all cells) is undefined is
    redrawn from the same child stream (as scripts/52 boot_valid / scripts/59 draw_valid). One draw per cell type
    (A3 iv): every statistic combines cells of one type only, so each is conditioned on its own cells."""
    gens = [np.random.default_rng(s) for s in seeds(f"{tag}_{kind}", NB)]
    p = np.full(n_inst, 1.0 / n_inst)
    W = np.stack([g.multinomial(n_inst, p) for g in gens]).astype(float)
    res = ev_all(W, field, (kind,))
    if all(r is None for r in res[kind]):
        return res[kind], 0
    redrawn = 0
    for _ in range(200):
        bad = np.flatnonzero(_bad(res))
        if not len(bad):
            return res[kind], redrawn
        redrawn += len(bad)
        W[bad] = np.stack([gens[b].multinomial(n_inst, p) for b in bad]).astype(float)
        new = ev_all(W[bad], field, (kind,))
        for r, rn in zip(res[kind], new[kind]):
            if r is not None:
                for k in r:
                    r[k][bad] = rn[k]
    raise AssertionError(f"could not draw non-degenerate replicates for {tag} {kind}")


def draw(tag: str, n_inst: int, field=None) -> tuple[dict, dict]:
    """One draw_kind per cell type; returns the cell results by type and the redraw count by type."""
    res, red = {}, {}
    for kind in CELLS:
        res[kind], red[kind] = draw_kind(tag, n_inst, kind, field)
    return res, red


def _draw_field(args):
    """Worker: independent institution draw for one field; returns that field's per-field statistics."""
    f, n_inst = args
    res, rd = draw(f"indep_{f}", n_inst, field=f)
    return f, per_field(res), rd


# =============================================================================================
# (5) per-field statistics from cell results
# =============================================================================================
def _nanmean(x, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(x, axis)


def per_field(res: dict) -> dict:
    """res: kind -> list of cell dicts aligned with CELLS[kind] (entries may be None). Returns stat -> {field:
    array (B,) or (B,3)}."""
    rel, pair_sets = CTX["rel"], CTX["pair_sets"]
    S: dict = {}

    def add(stat, f, v):
        S.setdefault(stat, {}).setdefault(f, []).append(v)

    for c, r in zip(CELLS["fix"], res["fix"]):
        if r is None:
            continue
        f = c["field"]
        rF, rG, q = r["rhoF"], r["rhoG"], r["rFG"]
        add("slope", f, rF @ W_SLOPE); add("early", f, (rF[:, 1] - rF[:, 0]) / 4)
        add("late", f, (rF[:, 2] - rF[:, 1]) / 5); add("y1lvl", f, rF[:, 0])
        add("dF_U", f, r["bF"] @ W_SLOPE); add("dG_U", f, r["bG"] @ W_SLOPE)
        add("rF_h", f, rF); add("rG_h", f, rG); add("rFG", f, q)
        for s in ("LB", "EXT", "LBFo", "EXTFo"):
            relF, relG = rel[s][f]
            if not c["adm"][s]:
                continue
            a, b = rF / np.sqrt(relF), rG / np.sqrt(relG)
            qs = q / np.sqrt(relF * relG)
            d = (1 - qs ** 2)[:, None]
            d = np.where(d > 0, d, np.nan)
            bF, bG = (a - qs[:, None] * b) / d, (b - qs[:, None] * a) / d
            add(f"dF_{s}", f, bF @ W_SLOPE); add(f"dG_{s}", f, bG @ W_SLOPE)
            add(f"bF_{s}_h", f, bF); add(f"bG_{s}_h", f, bG)
            add(f"dF_Uadm{s}", f, r["bF"] @ W_SLOPE); add(f"dG_Uadm{s}", f, r["bG"] @ W_SLOPE)
    for c, r in zip(CELLS["cov"], res["cov"]):
        if r is not None:
            add("cov_raw", c["field"], r["raw"] @ W_SLOPE); add("cov_par", c["field"], r["par"] @ W_SLOPE)
    for kind in ("edu", "phd", "phd1", "phda"):
        for c, r in zip(CELLS[kind], res[kind]):
            if r is not None:
                for k in ("raw", "pc", "pn", "pb", "rNE"):
                    add(f"{kind}_{k}", c["field"], r[k] @ W_SLOPE)
                add(f"{kind}_rNE_h", c["field"], r["rNE"])
    for c, r in zip(CELLS["cal"], res["cal"]):
        if r is None:
            continue
        rho = r["rho"]
        for name, ps in pair_sets.items():
            if c["coh"] in ps:
                add(f"Dw_{name}", c["field"], (rho[:, 1] - rho[:, 0]) / 9)
                add(f"Dc_{name}", c["field"], (rho[:, 1] - rho[:, 2]) / 9)
                add(f"Dx_{name}", c["field"], (rho[:, 2] - rho[:, 0]) / 9)
    out = {}
    for stat, d in S.items():
        out[stat] = {}
        for f, lst in d.items():
            x = np.stack(lst)                       # (cells, B, ...)
            if stat.startswith(("dF_LB", "dG_LB", "dF_EXT", "dG_EXT", "bF_LB", "bG_LB", "bF_EXT", "bG_EXT")):
                out[stat][f] = _nanmean(x, 0) if np.isfinite(x).any() else np.full(x.shape[1:], np.nan)
            else:
                out[stat][f] = x.mean(0)
    return out


# =============================================================================================
# (6) two-way inference helpers
# =============================================================================================
def fb_idx(tag: str, K: int, B: int = NB_FIELD) -> np.ndarray:
    return np.stack([np.random.default_rng(s).integers(K, size=K) for s in seeds(tag, B)])


def two_way_mean(so, sb, si, fields, tag, h=None, nan_ok=False) -> dict:
    """Mean over fields of a per-field statistic. so/sb/si: {field: array} (observed (1,)/(1,3); shared and
    independent draws (B,)/(B,3)); h: horizon index for (.,3) statistics."""
    g = (lambda a: a[..., h]) if h is not None else (lambda a: a)
    pf = np.array([float(g(so[f])[0]) for f in fields])
    X = np.stack([g(sb[f]) for f in fields]); Xi = np.stack([g(si[f]) for f in fields])
    if nan_ok:
        cb, ib = _nanmean(X), _nanmean(Xi)
        nan_frac = float(np.mean(~np.isfinite(X)))
        cb, ib = cb[np.isfinite(cb)], ib[np.isfinite(ib)]
    else:
        cb, ib, nan_frac = X.mean(0), Xi.mean(0), 0.0
    K = len(fields)
    fb = pf[fb_idx(tag, K, 2000)].mean(1)
    tw = twoway(float(pf.mean()), float(pf.var(ddof=1) / K), cb, ib, fb)
    tw.update(est=float(pf.mean()), k=K, pf=pf, cb=cb, ib=ib, nan_frac=nan_frac)
    return tw


def two_way_fun(fun, cols_o, cols_b, cols_i, tag) -> dict:
    """Two-way variance for a non-linear function of per-field arrays: fun(*cols) with each col (K, ...)
    observed; cols_b / cols_i the same with a leading replicate axis: list of (B, K, ...)."""
    est = float(fun(*cols_o))
    vf = jack_var(fun, *cols_o)
    cb = np.array([fun(*[c[b] for c in cols_b]) for b in range(cols_b[0].shape[0])])
    ib = np.array([fun(*[c[b] for c in cols_i]) for b in range(cols_i[0].shape[0])])
    K = cols_o[0].shape[0]
    fi = fb_idx(tag, K, 2000)
    fb = np.array([fun(*[c[ix] for c in cols_o]) for ix in fi])
    nan_frac = float(np.mean(~np.isfinite(cb)))
    cb, ib = cb[np.isfinite(cb)], ib[np.isfinite(ib)]
    tw = twoway(est, vf, cb, ib, fb[np.isfinite(fb)])
    tw.update(est=est, k=K, cb=cb, ib=ib, nan_frac=nan_frac)
    return tw


def put_tw(section, stat, tw, sample, bench=np.nan, note="", verdict=None):
    se = tw["tse"]
    mcse = tw.get("tmcse", np.nan)
    st = verdict if verdict is not None else status_pos(tw["tlo"], tw["thi"], mcse)
    if np.isfinite(mcse) and 2 * mcse > (tw["thi"] - tw["tlo"]) / 2:
        st += "; undetermined: 2 Monte Carlo SE of an endpoint exceed the CI half-width"
    rec(section, stat, tw["est"], sample, se, tw["tlo"], tw["thi"], "two-way normal", tw["tz"], z_p(tw["tz"]),
        "two-way z", tw["k"], MDE_K * se, power_at(bench, se), bench, mcse, st,
        note + (f"; robust (IQR) SE {tw['tse_r']:.4g}" if np.isfinite(tw.get("tse_r", np.nan)) else "")
        + ("; V<=0, larger one-way variance used" if tw.get("tfallback") else ""))
    tw["status"] = st
    tw["mde"] = MDE_K * se
    tw["power"] = power_at(bench, se)
    tw["bench"] = bench
    return tw


def robust_row(stat, tw, sample) -> tuple:
    """Exploratory E8: the per-cell corrected statistics have heavy-tailed replicates (cells whose replicate r_FG* is
    close to 1), so beside the pre-specified normal interval the IQR-based two-way SE of scripts/59.twoway (tse_r) is
    reported as est +/- 1.96 tse_r (not pre-specified)."""
    se = tw.get("tse_r", np.nan)
    lo, hi = tw["est"] - Z975 * se, tw["est"] + Z975 * se
    st = ("CI > 0" if lo > 0 else "CI < 0" if hi < 0 else "CI includes 0") if np.isfinite(se) else "robust SE undefined"
    rec("E8", stat + " [robust IQR interval]", tw["est"], sample + " (exploratory)", se, lo, hi,
        "two-way normal with IQR-based variances (exploratory)", k=tw["k"], status=st)
    return (lo, hi, st)


def surv(tw) -> str:
    """Verdict for 'rise survives' rules: CI above 0, with the edge rule."""
    if tw["tlo"] > 2 * tw["tmcse"]:
        return "survives"
    if abs(tw["tlo"]) <= 2 * tw["tmcse"]:
        return "edge"
    return "not distinguishable from 0" if tw["thi"] >= -2 * tw["tmcse"] else "opposite"


# =============================================================================================
def main():
    RES.clear()
    stage("start")
    # ---------------- ACS aggregation first (small parent before forking) ----------------
    acs_agg, acs_info = acs_aggregate()
    stage(f"ACS aggregated: {acs_info}")
    ins = s59.institutions("new").drop_duplicates("institution")[["institution", "inst_key"]]
    n61 = flows_shares(ins)
    stage(f"flows: {len(n61)} institution x CIP-2 x cohort rows")

    # ---------------- PSEO fixed-cohort panel (V4.14.1) ----------------
    ar = s52.load_ar_wapman(fields=FIELDS66)
    gen = s52._s28.load_generic()
    gen = gen.assign(G=-gen["g_rank"].astype(float))[["inst_key", "G"]]
    s59.set_release("new")
    er = {"y1": s52.load_fixed("y1", COHORTS + [LATE[c] for c in COHORTS]),
          "y5": s52.load_fixed("y5", COHORTS), "y10": s52.load_fixed("y10", COHORTS)}
    panel = s52.fixed_panel(er, ar, gen)
    assert panel["G"].notna().all()
    fix_cells = s52.cells_of(panel)
    fields = sorted({k[0] for k, _ in fix_cells})
    cal_cells = s59.calendar_cells(er, ar)
    del er
    gc.collect()
    stage(f"panel: {len(panel)} rows, {panel.inst_key.nunique()} institutions, {len(fix_cells)} cells, "
          f"{len(fields)} fields; calendar cells {len(cal_cells)}")

    share = acs_shares(acs_agg, fields)
    assert sorted(share.field) == fields, set(fields) - set(share.field)

    # ---------------- PhD rates (A1) ----------------
    phd, phd_info = phd_rates(ins, set(panel.inst_key))
    stage(f"PhD rates: {phd_info['n_inst']} institutions, {phd.phd_rate.notna().sum()} institution x cohort rates")
    # E10 (Revision 1): flag-4 shortfall within institutions; adjusted-denominator rate
    f4 = flag4_shortfall(phd)
    phd["phd_rate_adj"] = np.where(phd.BA_status == "4", phd.phd_rate * np.exp(f4["b_lo"]), phd.phd_rate)
    f4["n_s1_diff"] = int((phd.phd_rate_s1.notna() & (phd.phd_rate_s1 != phd.phd_rate)).sum())
    stage(f"flag-4 shortfall b={f4['b']:.4f} (CR1 SE {f4['se']:.4f})")

    # ---------------- reliabilities (scripts/56) ----------------
    relt = pd.read_csv(REL_CSV)
    relt = relt[relt.unit == "field"].set_index("field")
    rel = {"LB": {f: (float(relt.loc[f, "relF_LB"]), float(relt.loc[f, "relG_LB"])) for f in fields},
           "EXT": {f: (float(relt.loc[f, "relF_EXT"]), float(relt.loc[f, "relG_EXT"])) for f in fields},
           "LBFo": {f: (float(relt.loc[f, "relF_LB"]), 1.0) for f in fields},
           "EXTFo": {f: (float(relt.loc[f, "relF_EXT"]), 1.0) for f in fields}}

    # ---------------- cells ----------------
    fam = {f["key"]: f["cip4"][0][:2] for f in FIELDS66}
    panel["cip2"] = panel.field.map(fam)
    panel = panel.merge(n61, on=["inst_key", "cip2", "grad_cohort"], how="left")
    panel = panel.merge(phd[["inst_key", "grad_cohort", "phd_rate", "phd_rate_s1", "phd_rate_adj", "BA_status"]],
                        on=["inst_key", "grad_cohort"], how="left")
    covp = panel.dropna(subset=[f"cov_{h}" for h in HZ])
    cov_cells = s52.cells_of(covp)
    edu_cells = s52.cells_of(covp.dropna(subset=[f"n61_{h}" for h in HZ]))
    phd_cells = s52.cells_of(covp.dropna(subset=["phd_rate"]))
    phd1_cells = s52.cells_of(covp.dropna(subset=["phd_rate_s1"]))
    phda_cells = s52.cells_of(covp.dropna(subset=["phd_rate_adj"]))
    # BA count status flags of the coverage-complete panel rows (A3)
    ba_flags = covp.BA_status.fillna("no row").value_counts().sort_index().to_dict()
    # institution universe: every institution in a fixed-cohort, coverage or calendar cell (as scripts/59)
    univ = sorted(set().union(*[set(g.inst_key) for _, g in fix_cells + cov_cells + cal_cells]))
    assert set().union(*[set(g.inst_key) for _, g in edu_cells + phd_cells + phd1_cells + phda_cells]) <= set(univ)
    iid = {k: i for i, k in enumerate(univ)}
    NI = len(univ)
    ixs = lambda g: np.array([iid[k] for k in g.inst_key])
    CELLS.clear()
    CELLS["fix"] = []
    for (f, c), g in fix_cells:
        P, G = g.P.to_numpy(float), g.G.to_numpy(float)
        rfg = float(spearmanr(P, G)[0])
        adm = {s: rfg ** 2 < rel[s][f][0] * rel[s][f][1] for s in rel}
        CELLS["fix"].append(dict(field=f, coh=c, n=len(g), P=P, G=G, E=g[[f"e_{h}" for h in HZ]].to_numpy(float),
                                 ix=ixs(g), rFG=rfg, adm=adm))
    CELLS["cov"] = [dict(field=f, coh=c, n=len(g), P=g.P.to_numpy(float), E=g[[f"e_{h}" for h in HZ]].to_numpy(float),
                         C=g[[f"cov_{h}" for h in HZ]].to_numpy(float), ix=ixs(g)) for (f, c), g in cov_cells]
    CELLS["edu"] = [dict(field=f, coh=c, n=len(g), P=g.P.to_numpy(float), E=g[[f"e_{h}" for h in HZ]].to_numpy(float),
                         C=g[[f"cov_{h}" for h in HZ]].to_numpy(float), N=g[[f"n61_{h}" for h in HZ]].to_numpy(float),
                         ix=ixs(g)) for (f, c), g in edu_cells]
    CELLS["phd"] = [dict(field=f, coh=c, n=len(g), P=g.P.to_numpy(float), E=g[[f"e_{h}" for h in HZ]].to_numpy(float),
                         C=g[[f"cov_{h}" for h in HZ]].to_numpy(float), N=g["phd_rate"].to_numpy(float),
                         G=g.G.to_numpy(float), ix=ixs(g)) for (f, c), g in phd_cells]
    CELLS["phd1"] = [dict(field=f, coh=c, n=len(g), P=g.P.to_numpy(float), E=g[[f"e_{h}" for h in HZ]].to_numpy(float),
                          C=g[[f"cov_{h}" for h in HZ]].to_numpy(float), N=g["phd_rate_s1"].to_numpy(float),
                          G=g.G.to_numpy(float), ix=ixs(g)) for (f, c), g in phd1_cells]
    CELLS["phda"] = [dict(field=f, coh=c, n=len(g), P=g.P.to_numpy(float), E=g[[f"e_{h}" for h in HZ]].to_numpy(float),
                          C=g[[f"cov_{h}" for h in HZ]].to_numpy(float), N=g["phd_rate_adj"].to_numpy(float),
                          G=g.G.to_numpy(float), ix=ixs(g)) for (f, c), g in phda_cells]
    CELLS["cal"] = [dict(field=f, coh=c, n=len(g), P=g.P.to_numpy(float), E=g[["e0", "e1", "e2"]].to_numpy(float),
                         ix=ixs(g)) for (f, c), g in cal_cells]
    # T4 pair sets from the earnings-year rule (not hard-coded)
    ey = lambda c, h: set(range(int(c) + {"y1": 1, "y10": 10}[h], int(c) + {"y1": 1, "y10": 10}[h] + 3))
    pair_sets = {"all": list(COHORTS),
                 "t4p": [c for c in COHORTS if not ((ey(c, "y10") | ey(LATE[c], "y1")) & EXCL_YEARS)],
                 "t4s": [c for c in COHORTS if not ((ey(c, "y1") | ey(c, "y10") | ey(LATE[c], "y1")) & EXCL_YEARS)]}
    assert pair_sets["t4p"] == ["2001", "2004", "2007"] and pair_sets["t4s"] == ["2001", "2004"], pair_sets
    CTX.update(rel=rel, pair_sets=pair_sets)
    n_cells = {k: len(v) for k, v in CELLS.items()}
    n_cov_rows, n_phd_rows = len(covp), int(covp.phd_rate.notna().sum())
    n_phd1_rows = int(covp.phd_rate_s1.notna().sum())
    del covp, cov_cells, edu_cells, phd_cells, phd1_cells, phda_cells
    gc.collect()
    stage(f"cells: {n_cells}; universe {NI} institutions")

    # ---------------- observed; reproduction of scripts/59 (V4.14.1) ----------------
    obs = per_field(ev_all(np.ones((1, NI))))
    ref = pd.read_csv(REF_CSV)
    ref = ref[(ref.variant == "new")]
    refv = {k: float(ref[(ref.section == "career_time") & (ref.stat == v)].estimate.iloc[0]) for k, v in REF_STATS.items()}
    mean_of = lambda st: float(np.mean([obs[st][f][0] for f in sorted(obs[st])]))
    chk = dict(slope_all=mean_of("slope"), late_all=mean_of("late"), early_all=mean_of("early"),
               dbF_all=mean_of("dF_U"), dbG_all=mean_of("dG_U"), D_cal_all=mean_of("Dc_all"),
               D_within_all=mean_of("Dw_all"), D_cross_all=mean_of("Dx_all"), cov_partial_all=mean_of("cov_par"))
    for k, v in chk.items():
        assert np.isclose(v, refv[k], rtol=6e-6, atol=1e-9), (k, v, refv[k])
    rf = ref[(ref.section == "career_field_slope") & (ref.stat == "slope_per_yr")].set_index("field").estimate
    assert sorted(rf.index) == fields
    fsl_dev = max(abs(obs["slope"][f][0] - rf[f]) for f in fields)
    assert all(np.isclose(obs["slope"][f][0], rf[f], rtol=6e-6, atol=1e-9) for f in fields)
    k59 = dict(slope=int(ref[(ref.section == "career_time") & (ref.stat == "slope_all")].k.iloc[0]))
    assert k59["slope"] == len(fields)
    stage(f"observed; scripts/59 values reproduced (max per-field slope deviation {fsl_dev:.2e})")

    # ---------------- institution draws ----------------
    free_mb = None
    try:
        free_mb = int([ln.split()[1] for ln in open("/proc/meminfo") if ln.startswith("MemAvailable")][0]) // 1024
    except Exception:
        pass
    nw = WORKERS if (free_mb is None or free_mb >= 1536) else 1
    shared, red_s = draw("shared", NI)
    sb = per_field(shared)
    del shared
    gc.collect()
    stage(f"shared draw ({NB}; redrawn {red_s}); {nw} workers for per-field draws (MemAvailable {free_mb} MB)")
    all_fields = sorted({c["field"] for cl in CELLS.values() for c in cl})
    si, red_i = {}, {k: 0 for k in CELLS}
    with ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("fork")) as ex:
        for f, pf_, rd in ex.map(_draw_field, [(f, NI) for f in all_fields]):
            for k_, v_ in rd.items():
                red_i[k_] += v_
            for stat, d in pf_.items():
                si.setdefault(stat, {}).update(d)
    stage(f"independent per-field draws ({len(all_fields)} fields; redrawn {red_i})")

    # ---------------- setup numbers ----------------
    for k_ in CELLS:
        rec("setup", f"draws_redrawn_shared_{k_}", red_s[k_],
            note=f"of {NB} replicates of the {k_} cells (a replicate with an undefined statistic in a cell of this type "
                 f"is redrawn)")
        rec("setup", f"draws_redrawn_independent_{k_}", red_i[k_],
            note=f"of {NB} x {len({c['field'] for c in CELLS[k_]})} field replicates of the {k_} cells")
    rec("setup", "panel_rows", len(panel)); rec("setup", "panel_institutions", panel.inst_key.nunique())
    rec("setup", "panel_cells", n_cells["fix"]); rec("setup", "panel_fields", len(fields))
    rec("setup", "universe_institutions", NI)
    for kind in ("cov", "phd", "phd1", "phda", "edu", "cal"):
        rec("setup", f"{kind}_cells", n_cells[kind], k=len({c["field"] for c in CELLS[kind]}),
            note=f"{len(set().union(*[set(c['ix']) for c in CELLS[kind]]))} institutions")
    rec("setup", "coverage_complete_rows", n_cov_rows); rec("setup", "coverage_complete_rows_with_phd_rate", n_phd_rows)
    rec("setup", "coverage_complete_rows_with_phd_rate_flag1", n_phd1_rows,
        note="BA count with IPEDS count status 1 (as reported)")
    flab = pd.read_csv(LEHD_FLAGS, dtype=str).set_index("flag")["label"].str.strip().to_dict()
    for fl_, n_ in ba_flags.items():
        rec("setup", f"coverage_complete_rows_BA_status_{fl_}", n_, note=flab.get(fl_, "no all-programme row"))
    for k, v in acs_info.items():
        rec("setup", k, v)
    for k, v in chk.items():
        rec("setup", f"reproduced_{k}", v, note=f"scripts/59 V4.14.1 value {refv[k]:.6g} (data/interim/pseo_refresh.csv)")
    rec("setup", "max_abs_dev_per_field_slope_vs_scripts59", fsl_dev, k=len(fields))
    for k in ("sed_rows", "sed_total", "sed_nonus", "sed_unknown", "sed_unitids", "n_inst", "n_inst_unitid", "n_inst_in_sed"):
        rec("setup", f"phd_{k}", phd_info[k], note=phd_info["sed_years"] if k == "sed_rows" else "")
    rec("setup", "phd_multi_unitid_institutions", len(phd_info["multi_unitid"]), note="; ".join(phd_info["multi_unitid"]))
    R = {"setup": dict(NI=NI, n_cells=n_cells, n_cov_rows=n_cov_rows, n_phd_rows=n_phd_rows, fsl_dev=fsl_dev,
                       red_s=red_s, red_i=red_i, chk=chk, refv=refv, acs=acs_info, phd=phd_info,
                       n_panel=len(panel), n_inst=panel.inst_key.nunique(), fields=fields, nw=nw,
                       n_phd1_rows=n_phd1_rows, ba_flags=ba_flags, flab=flab,
                       n_inst_kind={k: len(set().union(*[set(c["ix"]) for c in CELLS[k]])) for k in CELLS},
                       k_kind={k: len({c["field"] for c in CELLS[k]}) for k in CELLS})}

    # =====================================================================================
    # T1
    # =====================================================================================
    sh = share.set_index("field")
    t1f = [f for f in fields if f in sh.index]
    g = sh.loc[t1f, "share_grad"].to_numpy(float)
    s_o = np.array([obs["slope"][f][0] for f in t1f])
    beta = ols_slope(g, s_o)
    beta_full = float(s_o.mean() / g.mean())
    w = (g - g.mean()) / ((g - g.mean()) ** 2).sum()
    Sb = np.stack([sb["slope"][f] for f in t1f], 1); Si = np.stack([si["slope"][f] for f in t1f], 1)
    tw = twoway(beta, jack_var(lambda a, b: ols_slope(a, b), g, s_o), Sb @ w, Si @ w)
    tw.update(est=beta, k=len(t1f))
    FI = fb_idx("t1_fieldboot", len(t1f))
    fbb = np.array([ols_slope(g[ix], s_o[ix]) if np.ptp(g[ix]) > 0 else np.nan for ix in FI])
    fbb = fbb[np.isfinite(fbb)]
    pg = np.stack([np.random.default_rng(s).permutation(len(g)) for s in seeds("t1_perm", NPERM)])
    null = np.array([ols_slope(g[p], s_o) for p in pg])
    p_perm = float((1 + np.sum(np.abs(null) >= abs(beta) - 1e-15)) / (1 + NPERM))
    fb_lo, fb_hi = (float(x) for x in np.percentile(fbb, [2.5, 97.5]))
    se_fb = float(np.std(fbb, ddof=1))
    mde1 = MDE_K * max(se_fb, tw["tse"])
    if fb_lo > 0 and p_perm < 0.05:
        v1 = "confound signature present"
    elif fb_hi < 0:
        v1 = "opposite"
    elif mde1 <= beta_full / 2:
        v1 = "not detected (powered against beta_full/2)"
    else:
        v1 = "underpowered"
    samp1 = f"{len(t1f)} fields; s_f = V4.14.1 fixed-cohort career slope; g_f = ACS 2023 share with a graduate degree, ages 30-45"
    se_v = max(se_fb, tw["tse"])
    rec("T1", "beta: slope of career slope on graduate-degree share [field bootstrap]", beta, samp1, se_fb, fb_lo, fb_hi,
        "field bootstrap percentile", np.nan, p_perm, "permutation of g across fields", len(t1f), mde1,
        power_at(beta_full, se_v), beta_full, np.nan, v1,
        f"MDE80 and power use 2.80 x max(field-bootstrap SE, two-way SE) = the SE the verdict uses ({se_v:.4g}); power at "
        f"beta_full/2: {power_at(beta_full / 2, se_v):.3f}; with the field-bootstrap SE alone ({se_fb:.4g}): MDE80 "
        f"{MDE_K * se_fb:.4g}, power {power_at(beta_full, se_fb):.3f}; {NB_FIELD} field resamples, {NPERM} permutations")
    put_tw("T1", "beta: slope of career slope on graduate-degree share [two-way]", tw, samp1, beta_full,
           f"power at beta_full/2: {power_at(beta_full / 2, tw['tse']):.3f}")
    rs = float(spearmanr(g, s_o)[0])
    nullr = np.array([spearmanr(g[p], s_o)[0] for p in pg])
    p_rs = float((1 + np.sum(np.abs(nullr) >= abs(rs) - 1e-12)) / (1 + NPERM))
    fbr = np.array([spearmanr(g[ix], s_o[ix])[0] for ix in FI[:2000]])
    fbr = fbr[np.isfinite(fbr)]
    rs_lo, rs_hi = (float(x) for x in np.percentile(fbr, [2.5, 97.5]))
    rec("T1", "Spearman(career slope, graduate-degree share) [secondary]", rs, samp1, np.nan, rs_lo, rs_hi,
        "field bootstrap percentile (2000)", np.nan, p_rs, "permutation", len(t1f))
    rec("T1", "benchmark beta_full = mean(s)/mean(g)", beta_full, samp1, note="slope if every field's rise were proportional to its share")
    rec("T1", "mean career slope over T1 fields", float(s_o.mean()), samp1, k=len(t1f))
    rec("T1", "mean graduate-degree share over T1 fields", float(g.mean()), samp1, k=len(t1f))
    rec("T1", "SD of graduate-degree share over T1 fields", float(g.std(ddof=1)), samp1, k=len(t1f))
    rec("T1", "min graduate-degree share", float(g.min()), samp1, note=t1f[int(np.argmin(g))])
    rec("T1", "max graduate-degree share", float(g.max()), samp1, note=t1f[int(np.argmax(g))])
    rec("T1", "implied change in career slope per +10 pp share", beta * 0.10, samp1)
    rec("T1", "fitted career slope at the lowest share", float(s_o.mean() + beta * (g.min() - g.mean())), samp1)
    se_g = sh.loc[t1f, "se_grad"].to_numpy(float)
    relg = float(1 - (se_g ** 2).mean() / g.var(ddof=1))
    rec("T1", "reliability of g across fields (1 - mean ACS var / var g)", relg, samp1, note="ACS sampling error only")
    R["T1"] = dict(beta=beta, fb=(fb_lo, fb_hi), se_fb=se_fb, p=p_perm, tw=tw, beta_full=beta_full, verdict=v1, rs=rs,
                   rs_ci=(rs_lo, rs_hi), p_rs=p_rs, k=len(t1f), g=g, s=s_o, fields=t1f, gmean=float(g.mean()),
                   smean=float(s_o.mean()), mde=mde1, pow_full=power_at(beta_full, max(se_fb, tw["tse"])),
                   pow_half=power_at(beta_full / 2, max(se_fb, tw["tse"])), relg=relg, gmin=float(g.min()),
                   gmax=float(g.max()), fmin=t1f[int(np.argmin(g))], fmax=t1f[int(np.argmax(g))],
                   fit_min=float(s_o.mean() + beta * (g.min() - g.mean())))

    # E11 (Revision 1): the two fields that share FOD1P 4101 (same g) as one point / either one dropped
    def t1_variant(groups, tag):
        gg = np.array([float(sh.loc[grp[0], "share_grad"]) for grp in groups])
        assert all(float(sh.loc[f, "share_grad"]) == gg[i] for i, grp in enumerate(groups) for f in grp)
        so = np.array([np.mean([obs["slope"][f][0] for f in grp]) for grp in groups])
        SB = np.stack([np.mean([sb["slope"][f] for f in grp], 0) for grp in groups], 1)
        SI = np.stack([np.mean([si["slope"][f] for f in grp], 0) for grp in groups], 1)
        b_ = ols_slope(gg, so)
        ww = (gg - gg.mean()) / ((gg - gg.mean()) ** 2).sum()
        t_ = twoway(b_, jack_var(lambda a, c: ols_slope(a, c), gg, so), SB @ ww, SI @ ww)
        t_.update(est=b_, k=len(groups))
        fi = fb_idx(f"{tag}_fieldboot", len(groups))
        fbv = np.array([ols_slope(gg[ix], so[ix]) if np.ptp(gg[ix]) > 0 else np.nan for ix in fi])
        fbv = fbv[np.isfinite(fbv)]
        pm = np.stack([np.random.default_rng(s_).permutation(len(gg)) for s_ in seeds(f"{tag}_perm", NPERM)])
        nl_ = np.array([ols_slope(gg[p_], so) for p_ in pm])
        pp_ = float((1 + np.sum(np.abs(nl_) >= abs(b_) - 1e-15)) / (1 + NPERM))
        lo_, hi_ = (float(x) for x in np.percentile(fbv, [2.5, 97.5]))
        sefb = float(np.std(fbv, ddof=1))
        bf = float(so.mean() / gg.mean())
        sev = max(sefb, t_["tse"])
        if lo_ > 0 and pp_ < 0.05:
            vv = "confound signature present"
        elif hi_ < 0:
            vv = "opposite"
        elif MDE_K * sev <= bf / 2:
            vv = "not detected (powered against beta_full/2)"
        else:
            vv = "underpowered"
        return dict(beta=b_, fb=(lo_, hi_), se_fb=sefb, p=pp_, tw=t_, bench=bf, mde=MDE_K * sev,
                    pow_full=power_at(bf, sev), verdict=vv, k=len(groups))
    ka, kb = SHARED_4101
    assert sh.loc[ka, "share_grad"] == sh.loc[kb, "share_grad"] and sh.loc[ka, "fod1p"] == sh.loc[kb, "fod1p"]
    R["E11"] = {}
    for tg, nm, groups in (("e11_drop_a", f"without {LAB.get(ka, ka)}", [[f] for f in t1f if f != ka]),
                           ("e11_drop_b", f"without {LAB.get(kb, kb)}", [[f] for f in t1f if f != kb]),
                           ("e11_merged", f"{LAB.get(ka, ka)} and {LAB.get(kb, kb)} merged into one point",
                            [[f] for f in t1f if f not in (ka, kb)] + [[ka, kb]])):
        v_ = t1_variant(groups, tg)
        rec("E11", f"beta {nm}", v_["beta"], f"{v_['k']} points (exploratory, Revision 1)", v_["se_fb"], v_["fb"][0],
            v_["fb"][1], "field bootstrap percentile", np.nan, v_["p"], "permutation of g across points", v_["k"],
            v_["mde"], v_["pow_full"], v_["bench"], np.nan, v_["verdict"] + " (T1 rule)",
            f"two-way {fci(v_['tw'])}; MDE80 and power from the larger of the two SEs")
        R["E11"][nm] = v_

    # E1, E2: alternative shares and segments (exploratory)
    for sname, col in (("professional or doctorate share (SCHL 23-24)", "share_long"), ("doctorate share (SCHL 24)", "share_phd")):
        x = sh.loc[t1f, col].to_numpy(float)
        b_ = ols_slope(x, s_o)
        wx = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
        tw_ = twoway(b_, jack_var(lambda a, b: ols_slope(a, b), x, s_o), Sb @ wx, Si @ wx)
        tw_.update(est=b_, k=len(t1f))
        nl = np.array([ols_slope(x[p], s_o) for p in pg])
        pp = float((1 + np.sum(np.abs(nl) >= abs(b_) - 1e-15)) / (1 + NPERM))
        put_tw("E1", f"beta on {sname}", tw_, samp1 + " (exploratory)", float(s_o.mean() / x.mean()),
               f"permutation p {pp:.4f}; benchmark = mean(s)/mean(share)")
        R.setdefault("E1", {})[col] = dict(tw=tw_, p=pp, bench=float(s_o.mean() / x.mean()))
    for seg in ("early", "late"):
        y = np.array([obs[seg][f][0] for f in t1f])
        b_ = ols_slope(g, y)
        Yb = np.stack([sb[seg][f] for f in t1f], 1); Yi = np.stack([si[seg][f] for f in t1f], 1)
        tw_ = twoway(b_, jack_var(lambda a, b: ols_slope(a, b), g, y), Yb @ w, Yi @ w)
        tw_.update(est=b_, k=len(t1f))
        nl = np.array([ols_slope(g[p], y) for p in pg])
        pp = float((1 + np.sum(np.abs(nl) >= abs(b_) - 1e-15)) / (1 + NPERM))
        put_tw("E2", f"beta of the {'y1->y5' if seg == 'early' else 'y5->y10'} segment slope on graduate-degree share",
               tw_, samp1 + " (exploratory)", float(y.mean() / g.mean()), f"permutation p {pp:.4f}")
        R.setdefault("E2", {})[seg] = dict(tw=tw_, p=pp, bench=float(y.mean() / g.mean()))
    y = np.array([obs["late"][f][0] for f in t1f])
    Yb = np.stack([sb["late"][f] for f in t1f], 1); Yi = np.stack([si["late"][f] for f in t1f], 1)
    for sname, col in (("professional-or-doctorate share (SCHL 23-24)", "share_long"), ("doctorate share (SCHL 24)", "share_phd")):
        x = sh.loc[t1f, col].to_numpy(float)
        b_ = ols_slope(x, y)
        wx = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
        tw_ = twoway(b_, jack_var(lambda a, b: ols_slope(a, b), x, y), Yb @ wx, Yi @ wx)
        tw_.update(est=b_, k=len(t1f))
        nl = np.array([ols_slope(x[p], y) for p in pg])
        pp = float((1 + np.sum(np.abs(nl) >= abs(b_) - 1e-15)) / (1 + NPERM))
        put_tw("E2", f"beta of the y5->y10 segment slope on the {sname} (Revision 1)", tw_, samp1 + " (exploratory)",
               float(y.mean() / x.mean()), f"permutation p {pp:.4f}; benchmark = mean(late)/mean(share)")
        R["E2"][f"late_{col}"] = dict(tw=tw_, p=pp, bench=float(y.mean() / x.mean()))
    y1l = np.array([obs["y1lvl"][f][0] for f in t1f])
    r3 = float(spearmanr(g, y1l)[0])
    nl = np.array([spearmanr(g[p], y1l)[0] for p in pg])
    p3 = float((1 + np.sum(np.abs(nl) >= abs(r3) - 1e-12)) / (1 + NPERM))
    rec("E3", "Spearman(graduate-degree share, y1 coupling level)", r3, samp1 + " (exploratory)", p=p3,
        p_kind="permutation", k=len(t1f))
    R["E3"] = dict(r=r3, p=p3)
    stage("T1 done")

    # =====================================================================================
    # T2
    # =====================================================================================
    cf = sorted(obs["cov_raw"])
    samp2 = f"coverage-complete balanced panel (coverage at y1, y5, y10), {n_cells['cov']} cells"
    twr = put_tw("T2", "slope of raw coupling, coverage-complete sample", two_way_mean(obs["cov_raw"], sb["cov_raw"],
                 si["cov_raw"], cf, "t2_raw"), samp2)
    twp = two_way_mean(obs["cov_par"], sb["cov_par"], si["cov_par"], cf, "t2_par")
    put_tw("T2", "slope of partial coupling given same-horizon coverage", twp, samp2, twr["est"] / 2,
           "benchmark = half the raw slope", verdict="partial rise " + surv(twp))
    dfo = {f: obs["cov_raw"][f] - obs["cov_par"][f] for f in cf}
    twd = two_way_mean(dfo, {f: sb["cov_raw"][f] - sb["cov_par"][f] for f in cf},
                       {f: si["cov_raw"][f] - si["cov_par"][f] for f in cf}, cf, "t2_diff")
    put_tw("T2", "raw minus partial slope (removed by the coverage control)", twd, samp2, twr["est"] / 2,
           "benchmark = half the raw slope")

    def share_removed(key_raw, key_par, fs, tag, sample, sec="T2", label=""):
        RO = np.array([obs[key_raw][f][0] for f in fs]); PO = np.array([obs[key_par][f][0] for f in fs])
        RB = np.stack([sb[key_raw][f] for f in fs], 1); PB = np.stack([sb[key_par][f] for f in fs], 1)
        RI = np.stack([si[key_raw][f] for f in fs], 1); PI = np.stack([si[key_par][f] for f in fs], 1)
        t_ = two_way_fun(lambda a, b: 1 - b.mean() / a.mean(), [RO, PO], [RB, PB], [RI, PI], tag)
        return put_tw(sec, f"share of the slope removed (1 - partial/raw){label}", t_, sample, 0.5,
                      verdict=("CI excludes 0.5" if (t_["thi"] < 0.5 or t_["tlo"] > 0.5) else "CI includes 0.5"))

    tws = share_removed("cov_raw", "cov_par", cf, "t2_share", samp2, label=", coverage")
    R["T2"] = dict(raw=twr, par=twp, diff=twd, share=tws, verdict_cov=surv(twp), k=len(cf))
    covc = pd.DataFrame([dict(field=c["field"], h=h, rho=float(spearmanr(c["P"], c["C"][:, i])[0]))
                         for c in CELLS["cov"] for i, h in enumerate(HZ)])
    R["T2"]["rhoC"] = {h: float(covc[covc.h == h].groupby("field").rho.mean().mean()) for h in HZ}
    for h in HZ:
        rec("T2", f"Spearman(prestige, coverage) at {h}, mean over fields of the field's mean over cells",
            R["T2"]["rhoC"][h], samp2, k=int(covc.field.nunique()))
    # PhD rate (A1)
    pf_ = sorted(obs["phd_raw"])
    samp2d = (f"coverage-complete panel rows with a PhD rate (SED doctorates c+5..c+12 by baccalaureate institution per "
              f"PSEO bachelor's graduate), {n_cells['phd']} cells")
    R["T2"]["phd"] = {}
    for key, lab_ in (("raw", "slope of raw coupling, PhD-rate sample"), ("pn", "slope of partial coupling given the PhD rate"),
                      ("pc", "slope of partial coupling given coverage, PhD-rate sample"),
                      ("pb", "slope of partial coupling given coverage and the PhD rate")):
        tw_ = two_way_mean(obs[f"phd_{key}"], sb[f"phd_{key}"], si[f"phd_{key}"], pf_, f"t2_phd_{key}")
        rawest = R["T2"]["phd"]["raw"]["est"] if key != "raw" else np.nan
        put_tw("T2", lab_, tw_, samp2d, rawest / 2 if key != "raw" else np.nan,
               "benchmark = half the raw slope on the same sample" if key != "raw" else "",
               verdict=("partial rise " + surv(tw_)) if key in ("pn", "pb") else None)
        R["T2"]["phd"][key] = tw_
    for key, lab_ in (("pn", "PhD rate"), ("pb", "coverage and PhD rate")):
        tw_ = two_way_mean({f: obs["phd_raw"][f] - obs[f"phd_{key}"][f] for f in pf_},
                           {f: sb["phd_raw"][f] - sb[f"phd_{key}"][f] for f in pf_},
                           {f: si["phd_raw"][f] - si[f"phd_{key}"][f] for f in pf_}, pf_, f"t2_phd_d{key}")
        put_tw("T2", f"raw minus partial slope (removed by the {lab_} control)", tw_, samp2d,
               R["T2"]["phd"]["raw"]["est"] / 2, "benchmark = half the raw slope")
        R["T2"]["phd"][f"d{key}"] = tw_
        R["T2"]["phd"][f"s{key}"] = share_removed("phd_raw", f"phd_{key}", pf_, f"t2_phd_share_{key}", samp2d,
                                                  label=f", {lab_}")
    R["T2"]["phd"]["k"] = len(pf_)
    R["T2"]["verdict_phd"] = surv(R["T2"]["phd"]["pn"])
    R["T2"]["verdict_both"] = surv(R["T2"]["phd"]["pb"])
    # A3 sensitivity: PhD-rate cells re-formed on rows whose bachelor's count is 'as reported' (IPEDS status 1)
    p1_ = sorted(obs["phd1_raw"])
    samp2s = (f"sensitivity (A3): as the PhD-rate sample, rows with IPEDS count status 1 only, {n_cells['phd1']} cells")
    R["T2"]["phd1"] = {"k": len(p1_)}
    for key, lab_ in (("raw", "slope of raw coupling, PhD-rate flag-1 sample"),
                      ("pn", "slope of partial coupling given the PhD rate, flag-1 sample"),
                      ("pb", "slope of partial coupling given coverage and the PhD rate, flag-1 sample")):
        tw_ = two_way_mean(obs[f"phd1_{key}"], sb[f"phd1_{key}"], si[f"phd1_{key}"], p1_, f"t2_phd1_{key}")
        rawest = R["T2"]["phd1"]["raw"]["est"] if key != "raw" else np.nan
        put_tw("T2", lab_, tw_, samp2s, rawest / 2 if key != "raw" else np.nan,
               "benchmark = half the raw slope on the same sample" if key != "raw" else "",
               verdict=("partial rise " + surv(tw_)) if key != "raw" else None)
        R["T2"]["phd1"][key] = tw_
    for key, lab_ in (("pn", "PhD rate"), ("pb", "coverage and PhD rate")):
        R["T2"]["phd1"][f"s{key}"] = share_removed("phd1_raw", f"phd1_{key}", p1_, f"t2_phd1_share_{key}", samp2s,
                                                   label=f", {lab_}, flag-1 sample")
    # E10 (Revision 1): denominator flag. (a) within-institution shortfall; (b) adjusted denominator; (c) flag-1 rates
    samp10 = (f"panel institutions' institution x cohort rows with a bachelor's count ({f4['n']} rows, {f4['n_inst']} "
              f"institutions, {f4['n_both']} with both flag 1 and flag 4) (exploratory, Revision 1)")
    zb = f4["b"] / f4["se"]
    rec("E10", "log bachelor's count, flag 4 minus other flags, within institution (institution + cohort indicators)",
        f4["b"], samp10, f4["se"], f4["b"] - Z975 * f4["se"], f4["b"] + Z975 * f4["se"], "normal, institution-clustered CR1",
        zb, z_p(zb), "z (CR1)", f4["n"],
        note=f"exp(b) - 1 = {np.expm1(f4['b']):+.4f}; conventional SE {f4['se_conv']:.4g}; flags of all panel institution x "
             f"cohort rows: " + ", ".join(f"{k_} {v_}" for k_, v_ in f4["flags"].items()))
    znc = f4["b_nc"] / f4["se_nc"]
    rec("E10", "log bachelor's count, flag 4 minus other flags, within institution (institution indicators only)",
        f4["b_nc"], samp10, f4["se_nc"], f4["b_nc"] - Z975 * f4["se_nc"], f4["b_nc"] + Z975 * f4["se_nc"],
        "normal, institution-clustered CR1", znc, z_p(znc), "z (CR1)", f4["n"],
        note=f"exp(b) - 1 = {np.expm1(f4['b_nc']):+.4f}; conventional SE {f4['se_nc_conv']:.4g}; without cohort "
             f"indicators the flag also picks up cohort differences in class size")
    pa_ = sorted(obs["phda_raw"])
    samp10b = (f"stress test: as the PhD-rate sample, flag-4 bachelor's counts divided by exp(b_lo) = "
               f"{np.exp(f4['b_lo']):.4f} (b_lo = lower 95% endpoint of b), {n_cells['phda']} cells (exploratory, "
               f"Revision 1)")
    R["E10"] = dict(f4=f4, k=len(pa_))
    for key, lab_ in (("raw", "slope of raw coupling, adjusted-denominator sample"),
                      ("pn", "slope of partial coupling given the adjusted PhD rate"),
                      ("pb", "slope of partial coupling given coverage and the adjusted PhD rate")):
        tw_ = two_way_mean(obs[f"phda_{key}"], sb[f"phda_{key}"], si[f"phda_{key}"], pa_, f"e10_phda_{key}")
        put_tw("E10", lab_, tw_, samp10b)
        R["E10"][key] = tw_
    for key, lab_ in (("pn", "adjusted PhD rate"), ("pb", "coverage and adjusted PhD rate")):
        R["E10"][f"s{key}"] = share_removed("phda_raw", f"phda_{key}", pa_, f"e10_phda_share_{key}", samp10b,
                                            sec="E10", label=f", {lab_}")
    rec("E10", "flag-1 rows whose rate differs from the full-sample rate", f4["n_s1_diff"],
        "PhD-rate table (exploratory, Revision 1)", note="the flag-1 sensitivity keeps each row's full-sample rate; it "
        "changes only which rows (fields, institutions) enter")
    # E7: footprint of graduate study at the institution level
    rows7 = []
    for c in CELLS["phd"]:
        for i, h in enumerate(HZ):
            rows7.append(dict(field=c["field"], coh=c["coh"], h=h, rDE=float(spearmanr(c["N"], c["E"][:, i])[0]),
                              rPD=float(spearmanr(c["P"], c["N"])[0]), rGD=float(spearmanr(c["G"], c["N"])[0])))
    e7 = pd.DataFrame(rows7)
    fm7 = e7.groupby(["field", "h"]).mean(numeric_only=True).reset_index()
    R["E7"] = dict(rDE={h: float(fm7[fm7.h == h].rDE.mean()) for h in HZ}, rPD=float(fm7[fm7.h == "y1"].rPD.mean()),
                   rGD=float(fm7[fm7.h == "y1"].rGD.mean()))
    for h in HZ:
        tw_ = two_way_mean(obs["phd_rNE_h"], sb["phd_rNE_h"], si["phd_rNE_h"], pf_, f"e7_rde_{h}", h=HZ.index(h))
        put_tw("E7", f"Spearman(PhD rate, earnings) at {h}, mean over fields of the field's mean over cells", tw_,
               samp2d + " (exploratory)")
        R["E7"][f"tw_{h}"] = tw_
    tw_ = two_way_mean(obs["phd_rNE"], sb["phd_rNE"], si["phd_rNE"], pf_, "e7_rde_slope")
    put_tw("E7", "slope over years of Spearman(PhD rate, earnings)", tw_, samp2d + " (exploratory)")
    R["E7"]["slope"] = tw_
    rec("E7", "Spearman(field prestige F, PhD rate), mean over fields of the field's mean over cells", R["E7"]["rPD"],
        samp2d + " (exploratory)", k=len(pf_))
    rec("E7", "Spearman(academia-wide rank G, PhD rate), mean over fields of the field's mean over cells", R["E7"]["rGD"],
        samp2d + " (exploratory)", k=len(pf_))
    pr = phd.dropna(subset=["phd_rate"])
    R["E7"]["rate_q"] = {c: tuple(float(x) for x in np.percentile(pr[pr.grad_cohort == c].phd_rate, [10, 50, 90])) for c in COHORTS}
    for c in COHORTS:
        q = R["E7"]["rate_q"][c]
        rec("E7", f"PhD rate p10 / p50 / p90, cohort {c}", q[1], f"{int((pr.grad_cohort == c).sum())} panel institutions",
            lo=q[0], hi=q[2], ci_kind="p10 / p90 (not a CI)", note="doctorates per year / bachelor's graduates per year")
    # E4: NAICS 61 share (exploratory)
    ef = sorted(obs["edu_raw"])
    samp4 = f"coverage-complete rows with a PSEO Flows NAICS-61 share at y1, y5, y10, {n_cells['edu']} cells (exploratory)"
    R["E4"] = {}
    for k, lab_ in (("raw", "raw coupling"), ("pc", "partial given coverage"), ("pn", "partial given NAICS-61 share"),
                    ("pb", "partial given coverage and NAICS-61 share")):
        tw_ = two_way_mean(obs[f"edu_{k}"], sb[f"edu_{k}"], si[f"edu_{k}"], ef, f"e4_{k}")
        put_tw("E4", f"slope of {lab_}", tw_, samp4)
        R["E4"][k] = tw_
    for k in ("pn", "pb"):
        tw_ = two_way_mean({f: obs["edu_raw"][f] - obs[f"edu_{k}"][f] for f in ef},
                           {f: sb["edu_raw"][f] - sb[f"edu_{k}"][f] for f in ef},
                           {f: si["edu_raw"][f] - si[f"edu_{k}"][f] for f in ef}, ef, f"e4_d{k}")
        put_tw("E4", f"raw minus {'NAICS-61' if k == 'pn' else 'both-controls'} partial slope", tw_, samp4)
        R["E4"][f"d{k}"] = tw_
    n61c = pd.DataFrame([dict(field=c["field"], h=h, rho=float(spearmanr(c["P"], c["N"][:, i])[0]))
                         for c in CELLS["edu"] for i, h in enumerate(HZ)])
    R["E4"]["rhoN"] = {h: float(n61c[n61c.h == h].groupby("field").rho.mean().mean()) for h in HZ}
    for h in HZ:
        rec("E4", f"Spearman(prestige, NAICS-61 share) at {h}, mean over fields of the field's mean over cells",
            R["E4"]["rhoN"][h], samp4, k=int(n61c.field.nunique()))
    R["E4"]["k"] = len(ef)
    stage("T2 done")

    # =====================================================================================
    # T3
    # =====================================================================================
    order = sorted(t1f, key=lambda f: (sh.loc[f, "share_grad"], f))
    half = len(order) // 2
    LO, HI = order[:half], order[half:]
    thr = (float(sh.loc[LO[-1], "share_grad"]), float(sh.loc[HI[0], "share_grad"]))
    samp3 = lambda nm, fs: f"{nm} half of fields by graduate-degree share ({len(fs)} fields)"
    tw_all = two_way_mean(obs["late"], sb["late"], si["late"], fields, "t3_all")
    put_tw("T3", "y5->y10 segment slope, all fields", tw_all, f"all {len(fields)} fields")
    twl = two_way_mean(obs["late"], sb["late"], si["late"], LO, "t3_lo")
    se_ = twl["tse"]
    if twl["tlo"] > 2 * twl["tmcse"]:
        v3 = "survives"
    elif abs(twl["tlo"]) <= 2 * twl["tmcse"]:
        v3 = "edge"
    elif twl["thi"] < 0:
        v3 = "opposite"
    else:
        v3 = "underpowered" if MDE_K * se_ > tw_all["est"] else "not detected"
    put_tw("T3", "y5->y10 segment slope, lower half by graduate-degree share", twl, samp3("lower", LO), tw_all["est"],
           "benchmark = all-field y5->y10 slope", verdict=v3)
    twh = two_way_mean(obs["late"], sb["late"], si["late"], HI, "t3_hi")
    put_tw("T3", "y5->y10 segment slope, upper half by graduate-degree share", twh, samp3("upper", HI))
    lo_pf, hi_pf = twl["pf"], twh["pf"]
    twdh = twoway(float(hi_pf.mean() - lo_pf.mean()), float(lo_pf.var(ddof=1) / len(LO) + hi_pf.var(ddof=1) / len(HI)),
                  twh["cb"] - twl["cb"], twh["ib"] - twl["ib"])
    twdh.update(est=float(hi_pf.mean() - lo_pf.mean()), k=f"{len(HI)} vs {len(LO)}")
    put_tw("T3", "y5->y10 segment slope, upper minus lower half", twdh, "upper vs lower half")
    rec("T3", "share threshold (highest share in the lower half, lowest in the upper half)", thr[0], "lower / upper half",
        lo=thr[0], hi=thr[1], ci_kind="threshold (not a CI)")
    R["T3"] = dict(all=tw_all, lo=twl, hi=twh, diff=twdh, verdict=v3, LO=LO, HI=HI, thr=thr)
    R["T3"]["biz"] = {f: (float(obs["late"][f][0]), "lower" if f in LO else "upper") for f in BUSINESS}
    # E9 (Revision 1): what T3 can separate. Proportional mechanism: late rise = b_late x g (no rise at g = 0).
    assert t1f == fields
    b_late = R["E2"]["late"]["bench"]
    gl_, gh_ = float(sh.loc[LO, "share_grad"].mean()), float(sh.loc[HI, "share_grad"].mean())
    pred = {"lo": b_late * gl_, "hi": b_late * gh_}
    pred["diff"] = pred["hi"] - pred["lo"]
    samp9 = "T3 halves; proportional mechanism with slope b_late = mean(y5->y10)/mean(g) (exploratory, Revision 1)"
    E9 = dict(b_late=b_late, g_lo=gl_, g_hi=gh_, pred=pred)
    for key, nm, tw_ in (("lo", "lower half", twl), ("hi", "upper half", twh), ("diff", "upper minus lower", twdh)):
        pr = pred[key]
        rec("E9", f"proportional-mechanism prediction, y5->y10 slope, {nm}", pr, samp9,
            note=(f"b_late {b_late:.4f} x mean g {gl_ if key == 'lo' else gh_:.4f}" if key != "diff"
                  else f"b_late {b_late:.4f} x (mean g upper {gh_:.4f} - lower {gl_:.4f})"))
        inside = tw_["tlo"] <= pr <= tw_["thi"]
        zz = (tw_["est"] - pr) / tw_["tse"]
        rec("E9", f"observed minus proportional prediction, y5->y10 slope, {nm}", tw_["est"] - pr, samp9, tw_["tse"],
            tw_["tlo"] - pr, tw_["thi"] - pr, "two-way normal (the T3 SE)", zz, z_p(zz), "two-way z",
            status="prediction inside the observed CI" if inside else "prediction outside the observed CI")
        E9[f"inside_{key}"] = inside
    sep_lo = tw_all["est"] - pred["lo"]
    E9["sep_lo"], E9["pow_lo"] = sep_lo, power_at(sep_lo, twl["tse"])
    E9["pow_diff"] = power_at(pred["diff"], twdh["tse"])
    rec("E9", "separation, lower-half level: uniform late rise (all-field slope) minus proportional prediction", sep_lo,
        samp9, twl["tse"], mde=MDE_K * twl["tse"], power=E9["pow_lo"], bench=sep_lo,
        note="power of a 5% two-sided test, with the lower-half SE, to tell the two predictions apart")
    rec("E9", "separation, upper minus lower: proportional prediction vs no difference", pred["diff"], samp9,
        twdh["tse"], mde=MDE_K * twdh["tse"], power=E9["pow_diff"], bench=pred["diff"],
        note="power of a 5% two-sided test, with the T3 difference SE, to tell the two predictions apart")
    E9["ratio_e2"] = R["E2"]["late"]["tw"]["est"] / b_late
    rec("E9", "E2's y5->y10 slope on g as a multiple of the proportional benchmark b_late", E9["ratio_e2"], samp9,
        note=f"E2 y5->y10 beta {R['E2']['late']['tw']['est']:.4f} / b_late {b_late:.4f}")
    R["E9"] = E9
    for nm, fs in (("lower", LO), ("upper", HI)):
        tw_ = two_way_mean(obs["early"], sb["early"], si["early"], fs, f"e5_{nm}")
        put_tw("E5", f"y1->y5 segment slope, {nm} half by graduate-degree share", tw_, samp3(nm, fs) + " (exploratory)")
        R.setdefault("E5", {})[nm] = tw_
        tw_ = two_way_mean(obs["slope"], sb["slope"], si["slope"], fs, f"e5s_{nm}")
        put_tw("E5", f"career slope (y1, y5, y10 OLS), {nm} half by graduate-degree share", tw_, samp3(nm, fs) + " (exploratory)")
        R["E5"][f"slope_{nm}"] = tw_
    stage("T3 done")

    # =====================================================================================
    # T4
    # =====================================================================================
    R["T4"] = {}
    for name, lab_ in (("all", "all four cohort pairs (reference)"), ("t4p", "pairs 2001, 2004, 2007 (primary)"),
                       ("t4s", "pairs 2001, 2004 (strict)")):
        fs = sorted(obs[f"Dc_{name}"])
        for key, nm in (("Dc", "calendar-matched contrast D_cal"), ("Dw", "within-cohort (y10-y1)/9"),
                        ("Dx", "cross-cohort drift at y1")):
            if name == "t4p" and key != "Dc":
                continue
            tw_ = two_way_mean(obs[f"{key}_{name}"], sb[f"{key}_{name}"], si[f"{key}_{name}"], fs, f"t4_{name}_{key}")
            bench = R["T4"][("all", "Dc")]["est"] if (key == "Dc" and name != "all") else np.nan
            put_tw("T4", f"{nm}, {lab_}", tw_, (f"triple-matched institutions; {lab_}; excluded earnings years 2008-10, "
                   f"2020-22" if name != "all" else "triple-matched institutions; all pairs"), bench,
                   "benchmark = D_cal on all four pairs" if key == "Dc" and name != "all" else "")
            R["T4"][(name, key)] = tw_
    R["T4"]["verdict"] = surv(R["T4"][("t4p", "Dc")])
    R["T4"]["verdict_strict"] = surv(R["T4"][("t4s", "Dc")])
    R["T4"]["pairs"] = pair_sets
    stage("T4 done")

    # =====================================================================================
    # T5
    # =====================================================================================
    R["T5"] = {}
    for key, nm in (("dF_U", "dF uncorrected"), ("dG_U", "dG uncorrected")):
        tw_ = two_way_mean(obs[key], sb[key], si[key], fields, f"t5_{key}")
        put_tw("T5", f"{nm}, per-cell form, all cells", tw_, f"all {n_cells['fix']} cells, {len(fields)} fields")
        R["T5"][key] = tw_
    tw_ = two_way_mean({f: obs["dG_U"][f] - obs["dF_U"][f] for f in fields}, {f: sb["dG_U"][f] - sb["dF_U"][f] for f in fields},
                       {f: si["dG_U"][f] - si["dF_U"][f] for f in fields}, fields, "t5_dGmF_U")
    put_tw("T5", "dG - dF uncorrected, per-cell form, all cells", tw_, "all cells")
    R["T5"]["dGmF_U"] = tw_
    RFo = np.stack([obs["rF_h"][f][0] for f in fields]); RGo = np.stack([obs["rG_h"][f][0] for f in fields])
    Qo = np.array([obs["rFG"][f][0] for f in fields])
    RFb = np.stack([sb["rF_h"][f] for f in fields], 1); RGb = np.stack([sb["rG_h"][f] for f in fields], 1)
    Qb = np.stack([sb["rFG"][f] for f in fields], 1)
    RFi = np.stack([si["rF_h"][f] for f in fields], 1); RGi = np.stack([si["rG_h"][f] for f in fields], 1)
    Qi = np.stack([si["rFG"][f] for f in fields], 1)

    def pooled(RF, RG, Q, rf, rg):
        a = (RF / np.sqrt(rf)[:, None]).mean(0); b = (RG / np.sqrt(rg)[:, None]).mean(0)
        q = (Q / np.sqrt(rf * rg)).mean()
        dd = 1 - q ** 2
        if not dd > 0:
            return np.full(3, np.nan), np.full(3, np.nan), q
        return (a - q * b) / dd, (b - q * a) / dd, q

    for s in ("LB", "EXT", "LBFo", "EXTFo"):
        sec = "T5"                                   # A2: F-only (LBFo, EXTFo) reported in T5 beside F-and-G
        adm_cells = [c for c in CELLS["fix"] if c["adm"][s]]
        fs = sorted({c["field"] for c in adm_cells})
        n_inad = n_cells["fix"] - len(adm_cells)
        samp5 = (f"per-cell form, {s} reliabilities; {len(adm_cells)} of {n_cells['fix']} cells admissible "
                 f"(r_FG*^2 < 1), {len(fs)} fields")
        rec(sec, f"inadmissible cells at {s}", n_inad, samp5,
            note="; ".join(f"{c['field']} {c['coh']}" for c in CELLS["fix"] if not c["adm"][s]))
        out = {}
        for key, nm in ((f"dF_{s}", "dF*"), (f"dG_{s}", "dG*")):
            tw_ = two_way_mean(obs[key], sb[key], si[key], fs, f"t5_{key}", nan_ok=True)
            put_tw(sec, f"{nm} corrected ({s}), per-cell form", tw_, samp5,
                   note=f"share of field x replicate values missing (shared draw) {tw_['nan_frac']:.3f}")
            out[key[:2]] = tw_
            out[key[:2] + "_rob"] = robust_row(f"{nm} corrected ({s}), per-cell form", tw_, samp5)
        o_ = {f: obs[f"dG_{s}"][f] - obs[f"dF_{s}"][f] for f in fs}
        tw_ = two_way_mean(o_, {f: sb[f"dG_{s}"][f] - sb[f"dF_{s}"][f] for f in fs},
                           {f: si[f"dG_{s}"][f] - si[f"dF_{s}"][f] for f in fs}, fs, f"t5_dGmF_{s}", nan_ok=True)
        put_tw(sec, f"dG* - dF* corrected ({s}), per-cell form", tw_, samp5, note=f"share missing {tw_['nan_frac']:.3f}")
        out["dGmF_rob"] = robust_row(f"dG* - dF* corrected ({s}), per-cell form", tw_, samp5)
        out["dGmF"] = tw_
        for key, nm in ((f"dF_Uadm{s}", "dF"), (f"dG_Uadm{s}", "dG")):
            tw_ = two_way_mean(obs[key], sb[key], si[key], fs, f"t5_{key}")
            put_tw(sec, f"{nm} uncorrected on the cells admissible at {s}", tw_, samp5)
            out[f"{nm}_Uadm"] = tw_
        for key in (f"bF_{s}_h", f"bG_{s}_h"):
            lv = _nanmean(np.stack([obs[key][f][0] for f in fs]), 0)
            out[key[:2] + "_h"] = lv
            for i, h in enumerate(HZ):
                rec(sec, f"{key[:2]}* at {h} ({s}), per-cell form, mean over fields", float(lv[i]), samp5)
        relF = np.array([rel[s][f][0] for f in fields]); relG = np.array([rel[s][f][1] for f in fields])
        bFo, bGo, qo = pooled(RFo, RGo, Qo, relF, relG)
        out["pooled_q"] = float(qo)
        out["pooled_levels"] = (bFo, bGo)
        rec(sec, f"pooled form: mean corrected r_FG* ({s})", float(qo), f"all {len(fields)} fields")
        for nm, fun in (("dF*", lambda RF, RG, Q, rf, rg: pooled(RF, RG, Q, rf, rg)[0] @ W_SLOPE),
                        ("dG*", lambda RF, RG, Q, rf, rg: pooled(RF, RG, Q, rf, rg)[1] @ W_SLOPE),
                        ("dG* - dF*", lambda RF, RG, Q, rf, rg: (pooled(RF, RG, Q, rf, rg)[1] - pooled(RF, RG, Q, rf, rg)[0]) @ W_SLOPE)):
            tw_ = two_way_fun(fun, [RFo, RGo, Qo, relF, relG],
                              [RFb, RGb, Qb, np.broadcast_to(relF, (NB, len(fields))), np.broadcast_to(relG, (NB, len(fields)))],
                              [RFi, RGi, Qi, np.broadcast_to(relF, (NB, len(fields))), np.broadcast_to(relG, (NB, len(fields)))],
                              f"t5_pooled_{s}_{nm}")
            put_tw(sec, f"{nm} corrected ({s}), pooled form", tw_, f"all {len(fields)} fields; decomposition of mean corrected correlations",
                   note=f"share of replicates undefined (shared draw) {tw_['nan_frac']:.3f}")
            out[f"pooled_{nm}"] = tw_
        for i, h in enumerate(HZ):
            rec(sec, f"pooled b_F* at {h} ({s})", float(bFo[i]), f"all {len(fields)} fields")
            rec(sec, f"pooled b_G* at {h} ({s})", float(bGo[i]), f"all {len(fields)} fields")
        out.update(n_inad=n_inad, fs=fs, n_adm=len(adm_cells), relF_mean=float(relF.mean()), relG_mean=float(relG.mean()))
        R["T5"][s] = out
    R["T5"]["U_pooled"] = {}
    for nm, fun in (("dF", lambda RF, RG, Q, rf, rg: pooled(RF, RG, Q, rf, rg)[0] @ W_SLOPE),
                    ("dG", lambda RF, RG, Q, rf, rg: pooled(RF, RG, Q, rf, rg)[1] @ W_SLOPE),
                    ("dG - dF", lambda RF, RG, Q, rf, rg: (pooled(RF, RG, Q, rf, rg)[1] - pooled(RF, RG, Q, rf, rg)[0]) @ W_SLOPE)):
        one = np.ones(len(fields))
        tw_ = two_way_fun(fun, [RFo, RGo, Qo, one, one], [RFb, RGb, Qb, np.broadcast_to(one, (NB, len(fields))),
                          np.broadcast_to(one, (NB, len(fields)))], [RFi, RGi, Qi, np.broadcast_to(one, (NB, len(fields))),
                          np.broadcast_to(one, (NB, len(fields)))], f"t5_pooledU_{nm}")
        put_tw("T5", f"{nm} uncorrected, pooled form", tw_, f"all {len(fields)} fields")
        R["T5"]["U_pooled"][nm] = tw_
    rec("T5", "pooled form: mean r_FG uncorrected", float(Qo.mean()), f"all {len(fields)} fields")
    R["T5"]["U_q"] = float(Qo.mean())
    R["T5"]["U_levels"] = (pooled(RFo, RGo, Qo, np.ones(len(fields)), np.ones(len(fields)))[:2])
    stage("T5 done")

    # ---------------- outputs ----------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(RES).to_csv(OUT_CSV, index=False, float_format="%.6g", lineterminator="\n")
    fs_tab = share.copy()
    fs_tab["career_slope"] = fs_tab.field.map(lambda f: float(obs["slope"][f][0]))
    fs_tab["late_slope"] = fs_tab.field.map(lambda f: float(obs["late"][f][0]))
    fs_tab["early_slope"] = fs_tab.field.map(lambda f: float(obs["early"][f][0]))
    fs_tab["half"] = fs_tab.field.map(lambda f: "lower" if f in LO else "upper")
    fs_tab.to_csv(OUT_FIELD, index=False, float_format="%.6g", lineterminator="\n")
    phd.to_csv(OUT_PHD, index=False, float_format="%.6g", lineterminator="\n")
    make_fig(R)
    write_md(R, fs_tab)
    stage("written")


# =============================================================================================
def make_fig(R):
    t1 = R["T1"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.8), gridspec_kw=dict(width_ratios=[1.05, 1, 1]))
    a = ax[0]
    a.scatter(t1["g"], t1["s"], s=18, color="#2a78d6", zorder=3)
    # label only the extremes (three lowest / highest shares and slopes) to keep the panel legible
    order_g, order_s = np.argsort(t1["g"]), np.argsort(t1["s"])
    lab_ix = sorted(set(order_g[:3]) | set(order_g[-3:]) | set(order_s[:3]) | set(order_s[-3:]))
    for i_ in lab_ix:
        a.annotate(LAB.get(t1["fields"][i_], t1["fields"][i_]), (t1["g"][i_], t1["s"][i_]), fontsize=6.5, color="#444",
                   xytext=(3, 2), textcoords="offset points")
    xs = np.linspace(0, t1["g"].max(), 10)
    xo = np.linspace(t1["g"].min(), t1["g"].max(), 10)
    a.plot(xo, t1["smean"] + t1["beta"] * (xo - t1["gmean"]), color="#eb6834", lw=1.2, label=f"OLS slope {t1['beta']:+.3f}")
    a.plot(xs, t1["beta_full"] * xs, color="#999", lw=1, ls="--", label=f"proportional benchmark {t1['beta_full']:+.3f}")
    a.axhline(0, color="#ccc", lw=0.8)
    a.set_xlabel("graduate-degree share of the field's bachelor's holders\naged 30-45 (ACS 2023)", fontsize=8)
    a.set_ylabel("career slope of coupling per year (PSEO fixed cohorts)", fontsize=8)
    a.set_title("(a) T1: career slope vs share (extremes labelled)", fontsize=10)
    a.legend(fontsize=7, frameon=False)

    def forest(b, items, xlabel, title, preds=None):
        labs = [x[0] for x in items]
        est = np.array([x[1]["est"] for x in items]); lo = np.array([x[1]["tlo"] for x in items])
        hi = np.array([x[1]["thi"] for x in items])
        y = np.arange(len(labs))[::-1]
        col = ["#eb6834" if l.startswith("T") else "#2a78d6" for l in labs]
        for yy, e, l_, h_, c_ in zip(y, est, lo, hi, col):
            b.errorbar([e], [yy], xerr=[[e - l_], [h_ - e]], fmt="o", color=c_, ms=4, capsize=2, lw=1)
        if preds:
            for k_, v_ in preds.items():
                b.plot([v_], [y[k_]], marker="D", ms=5, mfc="none", mec="#222", mew=1, ls="none",
                       label="proportional-mechanism prediction (E9)" if k_ == min(preds) else None)
            b.legend(fontsize=7, frameon=False, loc="upper left")
        b.axvline(0, color="#ccc", lw=0.8)
        b.set_yticks(y); b.set_yticklabels(labs, fontsize=7)
        b.set_xlabel(xlabel, fontsize=8)
        b.set_title(title, fontsize=10)

    items = [("raw slope, coverage sample", R["T2"]["raw"]), ("T2: partial | coverage", R["T2"]["par"]),
             ("T2: partial | PhD rate", R["T2"]["phd"]["pn"]), ("T2: partial | coverage + PhD rate", R["T2"]["phd"]["pb"]),
             ("y5->y10, all fields", R["T3"]["all"]), ("T3: y5->y10, lower-share half", R["T3"]["lo"]),
             ("y5->y10, upper-share half", R["T3"]["hi"]), ("y5->y10, upper minus lower half", R["T3"]["diff"]),
             ("D_cal, 4 pairs", R["T4"][("all", "Dc")]),
             ("T4: D_cal, excl. 2008-10/2020-22", R["T4"][("t4p", "Dc")]), ("T4: D_cal, strict", R["T4"][("t4s", "Dc")])]
    pr = R["E9"]["pred"]
    forest(ax[1], items, "change in coupling per year\n(two-way 95% CI)",
           "(b) T2-T4 (orange: pre-specified statistic)", preds={5: pr["lo"], 6: pr["hi"], 7: pr["diff"]})
    T5 = R["T5"]
    items5 = [("dF uncorrected, pooled", T5["U_pooled"]["dF"]), ("dG uncorrected, pooled", T5["U_pooled"]["dG"]),
              ("dG - dF uncorrected, pooled", T5["U_pooled"]["dG - dF"]),
              ("T5: dF*, LB-F, pooled", T5["LBFo"]["pooled_dF*"]), ("T5: dG*, LB-F, pooled", T5["LBFo"]["pooled_dG*"]),
              ("T5: dG* - dF*, LB-F, pooled", T5["LBFo"]["pooled_dG* - dF*"]),
              ("T5: dF*, EXT-F, pooled", T5["EXTFo"]["pooled_dF*"]), ("T5: dG*, EXT-F, pooled", T5["EXTFo"]["pooled_dG*"]),
              ("T5: dG* - dF*, EXT-F, pooled", T5["EXTFo"]["pooled_dG* - dF*"]),
              ("dG - dF uncorrected, per-cell", T5["dGmF_U"])]
    forest(ax[2], items5, "change per year in the standardized\nrank-regression coefficient on F or G (two-way 95% CI;\n"
           "per-cell corrected intervals undetermined, not shown)", "(c) T5: F/G slopes, F disattenuated")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=160, metadata={"Software": None})
    plt.close(fig)


# =============================================================================================
def fnum(x, d=4, sign=True) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def fci(tw, d=4) -> str:
    return f"[{fnum(tw['tlo'], d)}, {fnum(tw['thi'], d)}]"


def ec(tw, d=4) -> str:
    return f"{fnum(tw['est'], d)} {fci(tw, d)}"


def fpv(p) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def tab_num(x, section: str) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "—"
    if x.is_integer() and (abs(x) >= 1 or x == 0):
        return f"{int(x):,}"
    if section == "setup":
        return f"{x:.6g}"
    return f"{x:+.4f}" if abs(x) < 1000 else f"{x:,.1f}"


def lab(f: str) -> str:
    return LAB.get(f, f)


SLAB = {"LBFo": "LB-F", "EXTFo": "EXT-F", "LB": "LB-FG", "EXT": "EXT-FG"}
SLAB_DEF = ("LB / EXT = F's reliability at the public-edge lower bound / extrapolated to the full hire count "
            "(scripts/56); -F = F corrected, G as measured; -FG = F and G corrected")


def grp_status(d: dict) -> str:
    out = {}
    for s_ in ("LBFo", "EXTFo", "LB", "EXT"):
        out.setdefault(d[s_], []).append(SLAB[s_])
    return "; ".join(f"{v} at " + (" and ".join(ls) if len(ls) <= 2 else ", ".join(ls[:-1]) + " and " + ls[-1])
                     for v, ls in out.items())


def write_md(R, fs_tab):
    S, T1, T2, T3, T4, T5 = R["setup"], R["T1"], R["T2"], R["T3"], R["T4"], R["T5"]
    E1, E2, E3, E4, E5, E7 = R["E1"], R["E2"], R["E3"], R["E4"], R["E5"], R["E7"]
    E9, E10, E11 = R["E9"], R["E10"], R["E11"]
    ph, ph1 = T2["phd"], T2["phd1"]
    f4 = E10["f4"]
    k = len(S["fields"])
    L = []
    w = L.append
    w("# Graduate-school timing: is the career-time rise in coupling a by-product of graduate study?")
    w("")
    w("Script: `scripts/67_gradschool_timing.py` (seeded; re-run gives byte-identical outputs). Tables: "
      "`data/interim/gradschool_timing.csv` (every number), `data/interim/gradschool_field_share.csv` (field table), "
      "`data/interim/gradschool_phd_rate.csv` (institution x cohort PhD rates); figure "
      "`outputs/figures/gradschool_timing.png`. Public data only (PSEO V4.14.1, ACS 2023 PUMS, NCSES Survey of Earned "
      "Doctorates by baccalaureate institution, College Scorecard institution file, Wapman et al. 2022). coupling = "
      "Spearman(field prestige F, graduates' median earnings) across institutions within a field. Descriptive, not "
      "causal.")
    w("")
    # ------------------------------------------------------------------ answer
    w("## Answer")
    w("")
    w("**Key question: could the rise of coupling with years since graduation be a mechanical by-product of graduate "
      "study?** The referee's mechanism: graduates of higher-status institutions more often enter PhD, JD, MD or MBA "
      "programmes; while enrolled they have low or no UI-covered earnings (at y1, and at y5 for PhD students and "
      "medical residents), and after completion they earn more. Institution medians would then understate coupling "
      "early and not late, and coupling would rise with years since graduation even if the pay gradient on status "
      "did not change. Five checks were fixed on 2026-09-24 before any estimate (docstring; amendments A1-A3 are "
      "listed in Method, the Revision 1 changes in Revisions). Panel: PSEO V4.14.1 fixed cohorts 2001/2004/2007/2010, balanced over y1/y5/y10, "
      f"{k} fields, {S['n_cells']['fix']} field x cohort cells, {S['n_inst']} institutions; all-field career slope "
      f"{fnum(S['chk']['slope_all'])}/yr (reproduces scripts/59's V4.14.1 value {fnum(S['refv']['slope_all'])}). "
      "Intervals are two-way (field, institution) cluster intervals as in scripts/59 unless stated.")
    w("")
    ANSWER_SLOT = len(L)
    w("")
    n = 0

    def item(title, body):
        nonlocal n
        n += 1
        w(f"{n}. **{title}** {body}")

    # T1
    t1tw = T1["tw"]
    t1title = {"confound signature present": "T1, field level: the career slope is larger in fields where more "
                                             "bachelor's holders have a graduate degree.",
               "opposite": "T1, field level: the career slope is smaller in fields where more bachelor's holders "
                           "have a graduate degree.",
               "underpowered": "T1, field level: underpowered. The field-level data cannot tell a rise proportional "
                               "to the graduate-degree share from no share gradient at all.",
               }.get(T1["verdict"], "T1, field level: no share gradient detected at the pre-set power.")
    item(t1title,
         f"Across the {T1['k']} fields, the OLS slope of the fixed-cohort career slope s_f on the field's graduate-degree "
         f"share g_f (ACS 2023, bachelor's holders aged {AGE_LO}-{AGE_HI}, share with a master's, professional or doctoral "
         f"degree) is beta = {fnum(T1['beta'])} per unit of share, i.e. {fnum(T1['beta'] * 0.1)}/yr per 10 percentage "
         f"points [field bootstrap {fnum(T1['fb'][0])}, {fnum(T1['fb'][1])}; permutation p = {fpv(T1['p'])}; two-way "
         f"{fci(t1tw)}]. Spearman(s_f, g_f) = {fnum(T1['rs'], 3)} [{fnum(T1['rs_ci'][0], 3)}, {fnum(T1['rs_ci'][1], 3)}] "
         f"(permutation p = {fpv(T1['p_rs'])}). Shares run from {T1['gmin']:.3f} ({lab(T1['fmin'])}) to "
         f"{T1['gmax']:.3f} ({lab(T1['fmax'])}), mean {T1['gmean']:.3f}. Benchmark: if every field's rise were "
         f"proportional to its share (no rise at share 0), beta would be mean(s)/mean(g) = {fnum(T1['beta_full'])}. "
         f"The minimum detectable slope (80% power, 5% two-sided, larger of the field-bootstrap and two-way SE) is "
         f"{T1['mde']:.4f}; power is {T1['pow_full']:.2f} against the benchmark and {T1['pow_half']:.2f} against half "
         f"of it. Verdict (pre-set rule): **{T1['verdict']}**. The point estimate is "
         f"{T1['beta'] / T1['beta_full']:.2f} of the benchmark, but the interval contains both 0 and the benchmark. The "
         f"fitted career slope at the lowest share is {fnum(T1['fit_min'])}/yr (all-field mean {fnum(T1['smean'])}). "
         f"The share has little sampling error (reliability across fields {T1['relg']:.3f}, from the ACS replicate "
         f"weights), but it describes 30-45-year-olds in 2023, not the PSEO graduates, and it counts degrees held, not "
         f"enrolment at y1 or y5. {lab(SHARED_4101[0])} (career slope {fnum(T1['s'][T1['fields'].index(SHARED_4101[0])])}, "
         f"the largest of all fields) and {lab(SHARED_4101[1])} share FOD1P 4101 and so the same g, which the field "
         f"bootstrap and the permutation treat as two independent points (exploratory E11, Revision 1): "
         + "; ".join(f"{nm_}: beta {fnum(v_['beta'])} [field bootstrap {fnum(v_['fb'][0])}, {fnum(v_['fb'][1])}; "
                     f"permutation p {fpv(v_['p'])}], {v_['verdict']} by the T1 rule" for nm_, v_ in E11.items())
         + f". The point estimates are at most {max(v_['beta'] / v_['bench'] for v_ in E11.values()):.2f} of their "
           f"benchmarks, and the verdict is "
         + ("unchanged." if all(v_["verdict"] == T1["verdict"] for v_ in E11.values()) else "not the same in every version."))
    # T2
    tr, tp, ts = T2["raw"], T2["par"], T2["share"]
    rc = T2["rhoC"]
    allsurv = T2["verdict_cov"] == "survives" and T2["verdict_phd"] == "survives" and T2["verdict_both"] == "survives"
    item((f"T2, institution level: the rise survives both controls, but the PhD-rate control absorbs a share "
          f"{fnum(ph['spn']['est'], 2, False)} of it." if allsurv else "T2, institution level: verdicts differ by control."),
         f"(a) Coverage (share of IPEDS graduates with PSEO earnings at the same horizon, scripts/52's control; "
         f"higher-status institutions have lower coverage: within-cell Spearman(prestige, coverage), mean over fields of "
         f"the field's mean over cells, {fnum(rc['y1'], 3)} at y1, {fnum(rc['y5'], 3)} at y5, {fnum(rc['y10'], 3)} at "
         f"y10). On the coverage-complete "
         f"panel ({T2['k']} fields, {S['n_cells']['cov']} cells) the raw slope is {ec(tr)} and the slope of partial "
         f"coupling given coverage is {ec(tp)}; the control removes {fnum(ts['est'], 3)} {fci(ts, 3)} of the slope. "
         f"Verdict: **partial rise {T2['verdict_cov']}**. (b) PhD rate (research doctorates awarded in years c+5..c+12 "
         f"to people whose bachelor's institution is i, per bachelor's graduate of cohort c; NCSES SED): defined for "
         f"{S['n_phd_rows']} of {S['n_cov_rows']} coverage-complete rows ({ph['k']} fields, {S['n_cells']['phd']} "
         f"cells), so the sample is the coverage sample. Given the PhD rate the slope is {ec(ph['pn'])}; the control "
         f"removes {fnum(ph['spn']['est'], 3)} {fci(ph['spn'], 3)} of it; verdict **partial rise {T2['verdict_phd']}**. "
         f"Given coverage and the PhD rate together: {ec(ph['pb'])}, share removed {fnum(ph['spb']['est'], 3)} "
         f"{fci(ph['spb'], 3)}; verdict **partial rise {T2['verdict_both']}**. The PhD rate is not a clean timing "
         f"variable: within cells it tracks prestige (Spearman with F {fnum(E7['rPD'], 3)}, with G "
         f"{fnum(E7['rGD'], 3)}; means over fields), and its own association with earnings rises from {fnum(E7['rDE']['y1'], 3)} at y1 to "
         f"{fnum(E7['rDE']['y5'], 3)} at y5 and {fnum(E7['rDE']['y10'], 3)} at y10 (exploratory E7). That rise fits "
         f"graduate-school timing, and it equally fits a status gradient in pay that the PhD rate shares with prestige. "
         f"The share removed therefore mixes the two and is not an estimate of the timing effect. Denominator: "
         f"{S['ba_flags'].get('4', 0)} of the {S['n_cov_rows']} rows have a bachelor's count flagged 'partially "
         f"missing' (IPEDS flag 4), which could overstate their PhD rate. Within institutions (institution and cohort "
         f"indicators) the flag-4 counts differ from the same institution's other counts by a log {fnum(f4['b'])} "
         f"[{fnum(f4['b'] - Z975 * f4['se'])}, {fnum(f4['b'] + Z975 * f4['se'])}] (institution-clustered; "
         f"{f4['n_both']} institutions have both flags; exploratory E10), so there is "
         + ("no sign of a shortfall" if f4["b"] >= 0 else "at most a small shortfall")
         + f". Raising the flag-4 counts by the largest shortfall inside that interval "
         f"({100 * -np.expm1(f4['b_lo']):.1f}%) changes the share removed by the PhD rate from "
         f"{fnum(ph['spn']['est'], 3)} to {fnum(E10['spn']['est'], 3)} {fci(E10['spn'], 3)} (both controls: "
         f"{fnum(ph['spb']['est'], 3)} to {fnum(E10['spb']['est'], 3)}); the control enters only through within-cell "
         f"ranks. The A3 sensitivity on the {S['n_phd1_rows']} "
         f"rows flagged 'as reported' ({ph1['k']} fields, {S['n_cells']['phd1']} cells) gives a raw slope "
         f"{ec(ph1['raw'])}, given the PhD rate {ec(ph1['pn'])} (share removed {fnum(ph1['spn']['est'], 3)} "
         f"{fci(ph1['spn'], 3)}), given both {ec(ph1['pb'])} (share removed {fnum(ph1['spb']['est'], 3)} "
         f"{fci(ph1['spb'], 3)}). Each of these rows keeps its full-sample rate ({f4['n_s1_diff']} rows differ), so the "
         f"smaller share removed there reflects which fields and institutions enter, not a cleaner denominator. JD, MD "
         f"and MBA enrolment has no public institution-level count and enters only through coverage and the field "
         f"shares of T1/T3.")
    # T3
    tl, th, td, ta = T3["lo"], T3["hi"], T3["diff"], T3["all"]
    pr9 = E9["pred"]
    t3sep = not (E9["inside_lo"] and E9["inside_diff"])
    item(f"T3, the y5->y10 segment in the fields with the least graduate study: {T3['verdict']} by the pre-set rule, "
         + ("but T3 cannot tell the confound from its absence." if not t3sep else "and the proportional mechanism lies "
            "outside at least one T3 interval."),
         f"Lower half by graduate-degree share ({len(T3['LO'])} fields, share <= {T3['thr'][0]:.3f}, mean "
         f"{E9['g_lo']:.3f}): {ec(tl)}/yr; upper half ({len(T3['HI'])} fields, share >= {T3['thr'][1]:.3f}, mean "
         f"{E9['g_hi']:.3f}): {ec(th)}; upper minus lower {ec(td)}; all {k} fields {ec(ta)}. Verdict (pre-set rule: "
         f"lower-half CI above 0): **{T3['verdict']}**. The rule's power ({tl['power']:.2f}, MDE {tl['mde']:.4f}) is "
         f"against a zero late rise, not against the confound, and the lower half is not free of graduate study (shares "
         f"{min(fs_tab[fs_tab.half == 'lower'].share_grad):.2f}-{max(fs_tab[fs_tab.half == 'lower'].share_grad):.2f}). "
         f"If the late rise were fully proportional to the share (no rise at share 0; slope b_late = mean late slope / "
         f"mean share = {fnum(E9['b_late'])}), the lower half would show {fnum(pr9['lo'])}/yr, the upper half "
         f"{fnum(pr9['hi'])} and the difference {fnum(pr9['diff'])}; the lower-half and difference predictions are "
         f"{'both inside' if (E9['inside_lo'] and E9['inside_diff']) else 'not both inside'} the observed intervals. "
         f"The power to tell that mechanism from a uniform late rise is {E9['pow_lo']:.2f} with the lower-half level and "
         f"{E9['pow_diff']:.2f} with the difference (exploratory E9, Revision 1). Across all fields, the late-segment "
         f"slope on the share is {ec(E2['late']['tw'])} (permutation p {fpv(E2['late']['p'])}; exploratory E2), "
         f"{E9['ratio_e2']:.2f} times the proportional benchmark; on the professional-or-doctorate share (SCHL 23-24, "
         f"the degrees whose study can still depress y5) it is {ec(E2['late_share_long']['tw'])}, "
         f"{E2['late_share_long']['tw']['est'] / E2['late_share_long']['bench']:.2f} times its own benchmark (exploratory "
         f"E2, Revision 1). "
         + (("The lower half's late rise is therefore not distinguishable from the upper half's, and the observed "
             "pattern is not distinguishable from the proportional mechanism. ")
            if (td["tlo"] <= 0 <= td["thi"] and E9["inside_lo"] and E9["inside_diff"]) else "")
         + f"The lower half is: "
         + ", ".join(f"{lab(f)}" for f in T3["LO"]) + ".")
    # T4
    d_all, d_p, d_s = T4[("all", "Dc")], T4[("t4p", "Dc")], T4[("t4s", "Dc")]
    dw_s, dx_s = T4[("t4s", "Dw")], T4[("t4s", "Dx")]
    dw_a, dx_a = T4[("all", "Dw")], T4[("all", "Dx")]
    item(f"T4, calendar-matched contrast without recession and pandemic earnings years: lower end {T4['verdict']}.",
         f"D_cal = (rho(c, y10) - rho(c+9, y1))/9 on triple-matched institutions: all four cohort pairs {ec(d_all)} "
         f"(k={d_all['k']}); without pairs whose contrast cells have earnings in 2008-2010 or 2020-2022 (drops c=2010; "
         f"pairs {', '.join(T4['pairs']['t4p'])}) {ec(d_p)} (k={d_p['k']}), verdict **lower end {T4['verdict']}**; strict "
         f"(also drops c=2007, whose y1 cell has earnings in 2008-2010; pairs {', '.join(T4['pairs']['t4s'])}) "
         f"{ec(d_s)} (k={d_s['k']}), verdict **{T4['verdict_strict']}**. On the strict pairs the within-cohort change "
         f"(y10-y1)/9 is {ec(dw_s)} and the cross-cohort drift at y1 is {ec(dx_s)} (all pairs: {ec(dw_a)} and "
         f"{ec(dx_a)}). Under scripts/52's sign assumption (calendar and cohort trends both >= 0) the career-time "
         f"slope lies between D_cal and the within-cohort change: [{fnum(d_s['est'])}, {fnum(dw_s['est'])}] on the "
         f"strict pairs, [{fnum(d_all['est'])}, {fnum(dw_a['est'])}] on all pairs. T4 answers whether the lower end "
         f"depends on recession and pandemic years, when graduate enrolment is counter-cyclical; it does not remove "
         f"graduate-school timing, because the y1 cell of cohort c+9 is exposed to it as much as any y1 cell.")
    # T5
    def sst(st):
        return "undetermined" if "undetermined" in st else st.split(" (")[0]

    def sc(s):
        o = T5[s]
        return (f"dF* {ec(o['dF'])}, dG* {ec(o['dG'])}, dG*-dF* {ec(o['dGmF'])} ({o['n_adm']} of "
                f"{S['n_cells']['fix']} cells admissible, {len(o['fs'])} fields; mean rel_F {o['relF_mean']:.3f}, "
                f"rel_G {o['relG_mean']:.3f})")

    def pc(s):
        o = T5[s]
        return (f"dF* {ec(o['pooled_dF*'])}, dG* {ec(o['pooled_dG*'])}, dG*-dF* {ec(o['pooled_dG* - dF*'])} "
                f"(mean corrected r_FG {o['pooled_q']:.3f})")

    def rob(s, key):
        lo_, hi_, st_ = T5[s][key]
        return f"[{fnum(lo_)}, {fnum(hi_)}]"
    uf, ug, um = T5["dF_U"], T5["dG_U"], T5["dGmF_U"]
    up = T5["U_pooled"]
    t5_pattern = (sst(T5["EXTFo"]["pooled_dG*"]["status"]) == "holds"
                  and sst(T5["LBFo"]["pooled_dG*"]["status"]) != "holds"
                  and all(sst(T5[s]["pooled_dG* - dF*"]["status"]) != "holds" for s in ("LB", "EXT", "LBFo", "EXTFo"))
                  and sst(up["dG - dF"]["status"]) != "holds" and sst(um["status"]) == "holds"
                  and all(sst(T5[s]["dGmF"]["status"]) == "undetermined" for s in ("LB", "EXT", "LBFo", "EXTFo")))
    item("T5, F/G decomposition with field prestige disattenuated: the corrected G slope holds at the extrapolated "
         "reliability (EXT), not at the public-edge lower bound (LB). In the pooled form G's lead over F is not "
         "distinguishable from zero whether or not F is corrected; the per-cell uncorrected lead is above zero but has "
         "no determinable corrected counterpart, so the data cannot say whether correcting F removes it."
         if t5_pattern else "T5, F/G decomposition with field prestige disattenuated.",
         f"Uncorrected, per-cell form (all {S['n_cells']['fix']} cells, the scripts/52/59 statistic): dF {ec(uf)}, dG "
         f"{ec(ug)}, dG-dF {ec(um)}; pooled form: dF {ec(up['dF'])}, dG {ec(up['dG'])}, dG-dF {ec(up['dG - dF'])}. "
         f"Scenarios: {SLAB_DEF}. Pooled form: LB-F {pc('LBFo')}; EXT-F {pc('EXTFo')}; LB-FG {pc('LB')}; EXT-FG "
         f"{pc('EXT')}. Per-cell form: LB-F {sc('LBFo')}; EXT-F {sc('EXTFo')}; LB-FG {sc('LB')}; EXT-FG {sc('EXT')}. "
         f"The per-cell corrected "
         f"intervals are undetermined: in some replicates a cell's corrected r_FG* comes close to 1, the corrected "
         f"coefficients explode, and the Monte Carlo error of the interval endpoints exceeds the interval (status "
         f"'undetermined' in the table). With IQR-based variances (exploratory E8) the per-cell dG* intervals are "
         f"LB-F {rob('LBFo', 'dG_rob')}, EXT-F {rob('EXTFo', 'dG_rob')} and the dG*-dF* intervals LB-F "
         f"{rob('LBFo', 'dGmF_rob')}, EXT-F {rob('EXTFo', 'dGmF_rob')}. Status of the pooled form by the pre-set rule: "
         f"dG* "
         + "; ".join(f"{SLAB[s]} {sst(T5[s]['pooled_dG*']['status'])}" for s in ("LBFo", "EXTFo", "LB", "EXT"))
         + "; dG*-dF* "
         + "; ".join(f"{SLAB[s]} {sst(T5[s]['pooled_dG* - dF*']['status'])}" for s in ("LBFo", "EXTFo", "LB", "EXT"))
         + (f". By the pre-specified intervals, the corrected G slope dG* is above zero only if the published field "
            f"rank is close to its extrapolated reliability; at the public-edge lower bound it is not distinguishable "
            f"from zero (pooled uncorrected dG {ec(up['dG'])}). G's lead over F is a separate question. In the pooled "
            f"form it is not distinguishable from zero even uncorrected ({ec(up['dG - dF'])}), so its absence in the "
            f"corrected pooled intervals comes from the estimator form, not from the correction. In the per-cell form "
            f"the uncorrected lead is {ec(um)}, but every corrected per-cell interval is undetermined, so these data "
            f"cannot say whether correcting F would remove it. The " if t5_pattern else ". The ")
         + "exploratory per-cell IQR intervals are more favourable to G (dG* above zero in "
         + f"{sum(T5[s]['dG_rob'][2] == 'CI > 0' for s in SLAB)} of 4 scenarios, dG*-dF* above zero in "
         + f"{sum(T5[s]['dGmF_rob'][2] == 'CI > 0' for s in SLAB)} of 4), but they were chosen after the pre-specified "
           "intervals were seen.")
    # confound
    w("")
    w("### If the graduate-study confound were real: what would and would not survive")
    w("")
    w("")
    w("### Exploratory (not pre-specified)")
    w("")
    e1l, e1p = E1["share_long"], E1["share_phd"]
    w(f"- E1. T1 with the professional-or-doctorate share (SCHL 23-24): beta {ec(e1l['tw'])} (permutation p "
      f"{fpv(e1l['p'])}); with the doctorate share (SCHL 24): beta {ec(e1p['tw'])} (permutation p {fpv(e1p['p'])}).")
    w(f"- E2. T1 on the segments: y1->y5 slope on share {ec(E2['early']['tw'])} (p {fpv(E2['early']['p'])}); y5->y10 "
      f"{ec(E2['late']['tw'])} (p {fpv(E2['late']['p'])}; proportional benchmark {fnum(E2['late']['bench'])}). "
      f"Revision 1: the y5->y10 slope on the professional-or-doctorate share (SCHL 23-24; doctoral study and residency "
      f"are what can depress y5) is {ec(E2['late_share_long']['tw'])} (p {fpv(E2['late_share_long']['p'])}; benchmark "
      f"{fnum(E2['late_share_long']['bench'])}), on the doctorate share (SCHL 24) {ec(E2['late_share_phd']['tw'])} (p "
      f"{fpv(E2['late_share_phd']['p'])}; benchmark {fnum(E2['late_share_phd']['bench'])}).")
    w(f"- E3. Spearman(graduate-degree share, y1 coupling level) across fields = {fnum(E3['r'], 3)} (permutation p "
      f"{fpv(E3['p'])}).")
    w(f"- E4. PSEO Flows share of employed graduates working in educational services (NAICS 61), CIP-2 family of the "
      f"field ({E4['k']} fields, {S['n_cells']['edu']} cells): within-cell Spearman(prestige, NAICS-61 share), mean "
      f"over fields, "
      f"{fnum(E4['rhoN']['y1'], 3)} / {fnum(E4['rhoN']['y5'], 3)} / {fnum(E4['rhoN']['y10'], 3)} at y1/y5/y10; slope "
      f"raw {ec(E4['raw'])}, given coverage {ec(E4['pc'])}, given the NAICS-61 share {ec(E4['pn'])}, given both "
      f"{ec(E4['pb'])}.")
    w(f"- E5. y1->y5 segment: lower half {ec(E5['lower'])}, upper half {ec(E5['upper'])}; whole career slope: lower "
      f"half {ec(E5['slope_lower'])}, upper half {ec(E5['slope_upper'])}.")
    w(f"- E7. Within-cell Spearman(PhD rate, earnings), mean over fields: y1 {ec(E7['tw_y1'], 3)}, y5 {ec(E7['tw_y5'], 3)}, y10 "
      f"{ec(E7['tw_y10'], 3)}; slope over years {ec(E7['slope'])}. PhD rate (doctorates per year per bachelor's "
      f"graduate per year) p10 / p50 / p90 by cohort: "
      + "; ".join(f"{c}: {q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}" for c, q in E7["rate_q"].items()) + ".")
    w("- E8. IQR-based two-way intervals for the per-cell corrected T5 statistics (added after the T5 estimates were "
      "seen, because their replicates are heavy-tailed): "
      + "; ".join(f"{SLAB[s_]}: dF* [{fnum(T5[s_]['dF_rob'][0])}, {fnum(T5[s_]['dF_rob'][1])}], dG* "
                  f"[{fnum(T5[s_]['dG_rob'][0])}, {fnum(T5[s_]['dG_rob'][1])}], dG*-dF* "
                  f"[{fnum(T5[s_]['dGmF_rob'][0])}, {fnum(T5[s_]['dGmF_rob'][1])}]" for s_ in ("LBFo", "EXTFo", "LB", "EXT"))
      + ".")
    pr9 = E9["pred"]
    w(f"- E9 (Revision 1). What T3 can separate: proportional mechanism with b_late = {fnum(E9['b_late'])} (mean g "
      f"lower half {E9['g_lo']:.4f}, upper half {E9['g_hi']:.4f}) predicts {fnum(pr9['lo'])} (lower), "
      f"{fnum(pr9['hi'])} (upper), {fnum(pr9['diff'])} (difference); observed {ec(tl)}, {ec(th)}, {ec(td)}. Power to "
      f"separate it from a uniform late rise: {E9['pow_lo']:.3f} (lower-half level, distance {fnum(E9['sep_lo'])}), "
      f"{E9['pow_diff']:.3f} (difference). E2's y5->y10 slope on g is {E9['ratio_e2']:.2f} x b_late.")
    w(f"- E10 (Revision 1). IPEDS count flag of the bachelor's denominator: within institutions, log count flag 4 minus "
      f"others {fnum(f4['b'])} (CR1 SE {f4['se']:.4f}, conventional SE {f4['se_conv']:.4f}; {f4['n']} rows, "
      f"{f4['n_inst']} institutions, {f4['n_both']} with both flags 1 and 4); with institution indicators only "
      f"{fnum(f4['b_nc'])} (CR1 SE {f4['se_nc']:.4f}). Stress test (flag-4 counts / exp(b_lo), b_lo = "
      f"{fnum(f4['b_lo'])}): raw {ec(E10['raw'])}, given the adjusted PhD rate {ec(E10['pn'])} (share removed "
      f"{fnum(E10['spn']['est'], 3)} {fci(E10['spn'], 3)}), given coverage and it {ec(E10['pb'])} (share removed "
      f"{fnum(E10['spb']['est'], 3)} {fci(E10['spb'], 3)}); primary: {ec(ph['pn'])} ({fnum(ph['spn']['est'], 3)}) and "
      f"{ec(ph['pb'])} ({fnum(ph['spb']['est'], 3)}).")
    w("- E11 (Revision 1). T1 with the two fields sharing FOD1P 4101: "
      + "; ".join(f"{nm_}: beta {fnum(v_['beta'])} [field bootstrap {fnum(v_['fb'][0])}, {fnum(v_['fb'][1])}; two-way "
                  f"{fci(v_['tw'])}; permutation p {fpv(v_['p'])}; MDE {v_['mde']:.4f}, benchmark {fnum(v_['bench'])}] "
                  f"{v_['verdict']}" for nm_, v_ in E11.items()) + ".")
    w("")
    # ------------------------------------------------------------------ key numbers
    w("## Key numbers")
    w("")
    w("Every number the script computes (also in `data/interim/gradschool_timing.csv`). Slopes are change in coupling "
      "per year since graduation. Interval: two-way normal = est +/- 1.96 SE with SE from the two-way (field, "
      "institution) cluster variance V_field + V_inst - V_field x inst (scripts/59.twoway; 2000 institution draws of "
      "each kind); field bootstrap percentile (T1, 10000 resamples); p: two-way z or field-label permutation (10000). "
      "MDE80 = 2.80 x SE (80% power, 5% two-sided); power = power of the two-sided 5% test at the benchmark. Status "
      "rule (scripts/59): 'holds' = CI above 0, 'edge' = the lower end within 2 Monte Carlo SE of 0; the qualifier "
      "'undetermined' is appended when 2 Monte Carlo SE of an endpoint exceed the CI half-width (added after the T5 "
      "estimates were seen; it does not change any rule outcome). Rows marked E are exploratory. Scenario codes in T5 "
      "rows: LBFo / EXTFo = LB-F / EXT-F, LB / EXT = LB-FG / EXT-FG (" + SLAB_DEF + ").")
    w("")
    w("| section | quantity | sample / spec | estimate | 95% interval | interval | p | p from | k | MDE80 | benchmark | "
      "power | status | note |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in RES:
        iv = (f"[{tab_num(r['ci_lo'], r['section'])}, {tab_num(r['ci_hi'], r['section'])}]"
              if np.isfinite(r["ci_lo"]) else "—")
        w("| " + " | ".join([
            r["section"], r["stat"], r["sample"] or "—", tab_num(r["estimate"], r["section"]), iv, r["ci_kind"] or "—",
            fpv(r["p_value"]) if np.isfinite(r["p_value"]) else "—", r["p_kind"] or "—",
            str(r["k"]) if not isinstance(r["k"], float) or np.isfinite(r["k"]) else "—",
            f"{r['mde80']:.4f}" if np.isfinite(r["mde80"]) else "—",
            tab_num(r["benchmark"], r["section"]) if np.isfinite(r["benchmark"]) else "—",
            f"{r['power']:.3f}" if np.isfinite(r["power"]) else "—", r["status"] or "—",
            (r["note"] or "—").replace("|", "/")]) + " |")
    w("")
    # ------------------------------------------------------------------ method
    w("## Method")
    w("")
    w("- **Pre-specification.** The five tests, their statistics, benchmarks and verdict rules were fixed in the "
      "script's docstring on 2026-09-24 before any estimate was computed. Amendments: A1 (2026-09-24, before any T2 "
      "estimate) builds the PhD control from the NCSES table builder after the first draft had judged it infeasible; "
      "A2 (2026-09-25) reports the F-only correction in T5 beside the F-and-G correction, to match the lane's task "
      "wording, and withdraws E6; A3 (2026-09-25) adds the IPEDS-count flag sensitivity for the PhD rate, replaces an "
      "unverifiable re-fetch note by a verified re-fetch and asserts the uniqueness of the Flows rows (before any "
      "estimate of this session), and makes the institution draws per cell type (after a first run whose test "
      "statistics were not looked at; with one draw for all six cell types a replicate was redrawn whenever any cell "
      "of any type was degenerate). A previous session had run an earlier version of the script; its outputs were not "
      "read, and every number here comes from this run. Anything not in the docstring's T1-T5 is labelled "
      "exploratory; E8 and the 'undetermined' status qualifier were added after the T5 estimates were seen. Revision 1 "
      "(after an independent verification; all results seen) adds the exploratory rows E2 (late segment on the "
      "professional/doctorate shares), E9, E10 and E11, makes two descriptive averages field-weighted and rewrites the "
      "reading of T3, T5 and the confound list; no pre-specified statistic, sample, seed or rule changed (Revisions).")
    w(f"- **Panel.** scripts/52's functions through scripts/59's release switch (neither edited): "
      f"`load_er_pseo('undergrad', h, fields=FIELDS66, grad_cohort=...)` on PSEO V4.14.1, bachelor's cohorts 2001, "
      f"2004, 2007, 2010 (3-year graduation windows), balanced over y1/y5/y10 within field x cohort, cells with n >= "
      f"{NMIN}. F = -(Wapman field rank), G = -(Wapman academia-wide rank). Field career slope s_f = mean over the "
      f"field's cohorts of the OLS slope of coupling on years (1, 5, 10). Reproduction of scripts/59 (V4.14.1, "
      f"data/interim/pseo_refresh.csv): all-field slope {S['chk']['slope_all']:.6f} vs {S['refv']['slope_all']:.6f}, "
      f"y5->y10 {S['chk']['late_all']:.6f} vs {S['refv']['late_all']:.6f}, D_cal {S['chk']['D_cal_all']:.6f} vs "
      f"{S['refv']['D_cal_all']:.6f}, coverage partial {S['chk']['cov_partial_all']:.6f} vs "
      f"{S['refv']['cov_partial_all']:.6f}, dF {S['chk']['dbF_all']:.6f} vs {S['refv']['dbF_all']:.6f}, dG "
      f"{S['chk']['dbG_all']:.6f} vs {S['refv']['dbG_all']:.6f}; largest per-field slope difference "
      f"{S['fsl_dev']:.1e} ({k} fields).")
    a = S["acs"]
    w(f"- **Graduate-degree share (T1, T3).** ACS 2023 1-year PUMS person files psam_pusa.csv and psam_pusb.csv "
      f"({a['acs_person_rows']:,} person records): persons aged {AGE_LO}-{AGE_HI} with SCHL >= 21 (bachelor's or "
      f"higher; {a['acs_sample_ba_plus_30_45']:,} records, {a['acs_weighted_ba_plus_30_45']:,.0f} weighted); g_f = "
      f"PWGTP-weighted share with SCHL 22-24 among those whose first bachelor's field FOD1P maps to f. SE from the 80 "
      f"replicate weights (successive-difference formula 4/80 x sum of squared deviations). FOD1P -> field: "
      f"src/crosswalks.fields.fod1p_by_key_all() (the project's collision-free map) and six codes added here for "
      f"fields that map does not cover (Appendix B; kinesiology and Health/PE/Recreation share FOD1P 4101 and so "
      f"get the same share).")
    p_ = S["phd"]
    w(f"- **PhD rate (T2, A1).** NCSES Survey of Earned Doctorates, public table builder "
      f"({SED_PAGE}): research doctorates by baccalaureate institution (IPEDS UnitID) and doctorate year, "
      f"{p_['sed_years']}, {p_['sed_rows']:,} institution x year rows with a non-zero count, {p_['sed_total']:,} "
      f"doctorates in total ({p_['sed_nonus']:,} with a non-U.S. and {p_['sed_unknown']:,} with an unknown "
      f"baccalaureate institution). Each PSEO institution (8-digit OPEID) is linked to the IPEDS UnitIDs that the "
      f"College Scorecard institution file lists under that OPEID ({p_['n_inst_unitid']} of {p_['n_inst']} panel "
      f"institutions have a UnitID, {p_['n_inst_in_sed']} appear in SED; {len(p_['multi_unitid'])} have more than one "
      f"UnitID). D_i(c) = doctorates in years c+{SED_LAG[0]}..c+{SED_LAG[1]}; BA_i(c) = PSEO y1_ipeds_count of the "
      f"institution's all-programme bachelor's row for cohort c; rate = (D/8)/(BA/3). The window is a cohort-matched "
      f"institution-level proxy, not a count of the cohort's own doctorates. IPEDS count flags of the coverage-"
      f"complete rows: "
      + ", ".join(f"{fl_} ({S['flab'].get(fl_, 'no all-programme row')}): {n_}" for fl_, n_ in S["ba_flags"].items())
      + ".")
    w("- **Partial coupling (T2).** Per cell and horizon, partial Spearman coupling of prestige and earnings given "
      "same-horizon coverage (scripts/52), given the PhD rate, and given both (second-order partial correlation of "
      "ranks); slope over (1, 5, 10) as for s_f; share removed = 1 - mean partial slope / mean raw slope.")
    w("- **T3.** Segment slope (rho(y10) - rho(y5))/5 averaged over the field's cohorts; lower half = the floor(K/2) "
      "fields with the lowest g_f (ties by field key).")
    w("- **T4.** Calendar cells (c, y1), (c, y10), (c+9, y1) on institutions present in all three (scripts/59's "
      "calendar_cells); earnings years of window c: y1 c+1..c+3, y10 c+10..c+12. The excluded pairs follow from these "
      "years by rule (asserted: primary 2001/2004/2007, strict 2001/2004).")
    w("- **T5.** Per cell, standardized rank regression of earnings on F and G: b_F = (rho_F - r_FG rho_G)/(1 - r_FG^2), "
      "b_G symmetric; dF, dG = slopes over years. Disattenuation with scripts/56's per-field reliabilities "
      "(data/interim/prestige_reliability.csv): LB = split-half reliability of the public-edge SpringRank (a lower "
      "bound for the published rank), EXT = extrapolated to the full hire count; rho_F* = rho_F/sqrt(rel_F), "
      "rho_G* = rho_G/sqrt(rel_G), r_FG* = r_FG/sqrt(rel_F rel_G) (F-only variants: rel_G = 1). Cells with "
      "r_FG*^2 >= 1 are inadmissible (the assumed reliabilities cannot hold there) and are dropped, with the "
      "uncorrected decomposition on the same cells beside them; in a replicate an admissible cell whose replicate "
      "r_FG*^2 >= 1 is left out for that replicate. Pooled form: the same algebra on the across-field means of the "
      "corrected correlations at each horizon.")
    w(f"- **Inference.** Two-way (field, institution) cluster variance of scripts/59 (imported): V_field (between-field "
      f"variance / K, or the jackknife over fields for non-linear statistics) + V_inst (one institution multinomial "
      f"draw per cell type, shared by every cell of that type in every field; {NB} replicates) - V_field x inst (an "
      f"independent draw per field and cell type, shared by that field's cells of the type; {NB} replicates); normal "
      f"CI. Replicates with an undefined statistic in a cell of the type are redrawn from the same child seed "
      f"(shared / per-field redraws by type: "
      + ", ".join(f"{k_} {S['red_s'][k_]} / {S['red_i'][k_]}" for k_ in S["red_s"])
      + f"). T1 adds a field bootstrap "
      f"({NB_FIELD}) and a permutation of g across fields ({NPERM}). Seeds: one numpy SeedSequence child per "
      f"replicate, so results do not depend on the number of worker processes. Institution universe "
      f"{S['NI']}.")
    w("")
    # ------------------------------------------------------------------ caveats
    w("## Caveats")
    w("")
    w("- **Proxies, not enrolment.** No public source gives graduate enrolment by bachelor's institution, field and "
      "cohort. The field share counts degrees held by 30-45-year-olds in 2023 (cohorts partly different from the PSEO "
      "graduates; people who moved field between degrees are counted under their bachelor's field). The PhD rate is "
      "an institution-wide rate, the same for every field of an institution, and counts research doctorates only; "
      "law, medicine and business have no public institution-level counts. Coverage mixes graduate enrolment with "
      "out-of-state work, self-employment and non-employment.")
    w("- **Partial correlation is linear in ranks.** A control that enters the earnings of a subset of graduates "
      "(those enrolled) is only approximately removed by a partial Spearman correlation on institution medians.")
    w(f"- **IPEDS counts.** The all-programme bachelor's count is flagged 'partially missing' (flag 4) for "
      f"{S['ba_flags'].get('4', 0)} of the {S['n_cov_rows']} coverage-complete rows. Within institutions the log "
      f"difference of these counts from the same institution's other counts is {fnum(E10['f4']['b'])} (95% CI "
      f"[{fnum(E10['f4']['b'] - Z975 * E10['f4']['se'])}, {fnum(E10['f4']['b'] + Z975 * E10['f4']['se'])}]; "
      f"identified from the {E10['f4']['n_both']} institutions with both flags); raising them by the largest shortfall "
      f"in that interval moves the T2 shares removed from {T2['phd']['spn']['est']:.3f} to {E10['spn']['est']:.3f} "
      f"(PhD rate) and from {T2['phd']['spb']['est']:.3f} to {E10['spb']['est']:.3f} (both controls) (E10). A shortfall "
      f"that differs across institutions within a cell is not tested. The flag-1 subsample keeps each row's rate and "
      f"differs only in which fields and institutions enter.")
    w("- **Field-level power.** T1 and T3 rest on the fields of the panel; the MDEs and powers are stated with each "
      "result. At this power neither can exclude a late or whole-career rise proportional to the share (E9, T1), and "
      "a mechanism that operates through professional degrees in high-share and low-share fields alike would not "
      "show in either.")
    w("- **Calendar years.** T4 removes cells whose earnings years fall in 2008-2010 or 2020-2022 but keeps the "
      "surrounding years; the strict version rests on two cohort pairs.")
    w("- **Reliability bracket.** LB and EXT bracket the reliability of the published field rank from public data "
      "(scripts/56); neither is the true value. With F and G collinear, the corrected coefficients are sensitive to "
      "the assumed reliabilities, and some cells are inadmissible under LB.")
    w("- **Descriptive.** Institution medians of UI-covered earnings; no individual histories; PSEO partner "
      "institutions are public-skewed.")
    w("")
    # ------------------------------------------------------------------ revisions
    w("## Revisions")
    w("")
    w("### Revision 1 (2026-09-25): answers to an independent verification")
    w("")
    w("The verifier rebuilt the ACS shares, the panel statistics, the halves, the PhD-rate table and the two-way "
      "variances and reproduced the first version's point estimates and intervals. Every change below is in reading, "
      "labelling or added exploratory rows; no pre-specified statistic, sample, seed or verdict rule changed. The "
      "first version's CSV was compared row by row with this run's (outside the script): every row with an unchanged "
      "label has identical values except the T1 field-bootstrap row (MDE80, power and note, changed on purpose, item "
      "7); of the relabelled rows, the E7 correlations have identical values and the coverage (T2) and NAICS-61 (E4) "
      "correlations changed because they are now field-weighted (item 7).")
    w("")
    up_, um_ = T5["U_pooled"], T5["dGmF_U"]
    w(f"1. **T3 was read as surviving the confound (major).** The pre-set verdict is kept ('{T3['verdict']}': the "
      f"lower-half CI lies above 0), but the item moved from 'would survive' to 'not informative about the confound'. "
      f"The lower half is not free of graduate study (mean share {E9['g_lo']:.3f}). A fully proportional late rise "
      f"predicts {fnum(E9['pred']['lo'])} (lower half) and {fnum(E9['pred']['diff'])} (difference), "
      f"{'both inside' if (E9['inside_lo'] and E9['inside_diff']) else 'not both inside'} the observed intervals, and "
      f"the power to tell it from a uniform rise is {E9['pow_lo']:.2f} / {E9['pow_diff']:.2f} (new exploratory E9). "
      f"The first version's 'power 1.00' is against a zero rise, which is now stated. E2's late-segment slope on the "
      f"share ({fnum(E2['late']['tw']['est'])}, {E9['ratio_e2']:.2f} x the proportional benchmark) is now reported "
      f"in T3, and E2 adds the late segment on the professional/doctorate shares. 'As large as', 'no smaller' and "
      f"'equal to ... within' were replaced by 'not distinguishable from'.")
    w(f"2. **The loss of G's lead over F was attributed to the correction of F (major).** In the pooled form the "
      f"uncorrected lead is already not distinguishable from zero ({ec(up_['dG - dF'])}); only the per-cell uncorrected "
      f"lead ({ec(um_)}) is above zero, and its corrected counterparts are undetermined. T5's title, the 'In short' "
      f"paragraph and the G-loading bullet now say so; 'depends on how reliable the published field rank is' now "
      f"refers to the corrected G slope dG* only (pooled uncorrected dG {ec(up_['dG'])}, LB-F "
      f"{ec(T5['LBFo']['pooled_dG*'])}, EXT-F {ec(T5['EXTFo']['pooled_dG*'])}).")
    w("3. **'Would survive' overstated the proxy-adjusted rise (minor).** The heading is now 'Survives the available "
      "institution-level proxies (the confound itself is not excluded)', and the bullet names what the proxies miss: "
      "field-specific doctoral study, professional degrees, and study while working, which lowers earnings without "
      "removing them from coverage.")
    bz_ = T3["biz"]
    w(f"4. **MBA timing (minor).** MBA programmes were listed as ending before y5. Full-time MBA students often enrol "
      f"after several years of work, so y5 can fall in the programme; the text now says so and reports the business "
      f"fields' late slopes ({fnum(min(v for v, _ in bz_.values()))} to {fnum(max(v for v, _ in bz_.values()))}).")
    w(f"5. **'Flag 4 overstates the PhD rate' was asserted, not tested (minor).** New exploratory E10: within "
      f"institutions (institution and cohort indicators) the log difference of flag-4 counts is {fnum(f4['b'])} "
      f"(CR1 SE {f4['se']:.4f}; with institution indicators only {fnum(f4['b_nc'])}, SE {f4['se_nc']:.4f}). "
      + ("The verifier's within-institution estimate had the other sign; both are small and not distinguishable from "
         "zero. " if (f4["b"] > 0 and abs(f4["b"] / f4["se"]) < Z975) else "")
      + f"Raising flag-4 counts by the largest shortfall inside the interval ({100 * -np.expm1(f4['b_lo']):.1f}%) moves "
      f"the share removed by the PhD rate from {ph['spn']['est']:.3f} to {E10['spn']['est']:.3f}. The flag-1 "
      f"sensitivity keeps every row's rate ({f4['n_s1_diff']} rows differ), so its smaller share removed reflects "
      f"composition; the text now says so.")
    e11m = [v_ for nm_, v_ in E11.items() if "merged" in nm_][0]
    w(f"6. **Kinesiology and Health/PE/Recreation share one FOD1P code (minor).** New exploratory E11: beta "
      + ", ".join(f"{fnum(v_['beta'])} ({nm_})" for nm_, v_ in E11.items())
      + f"; with the two merged the T1 rule gives '{e11m['verdict']}'.")
    w("7. **Averages and the T1 row (minor).** The within-cell correlations of prestige with coverage (T2) and with "
      "the NAICS-61 share (E4) are now means over fields of the field's mean over cells, like E7 and every two-way "
      "statistic (the first version averaged over cells); labels say so. The T1 field-bootstrap row's MDE80 and power "
      "now use the larger of the field-bootstrap and two-way SEs, as the verdict does (field-bootstrap-only values in "
      "its note).")
    w("8. **SED re-fetch record (minor).** The SED_MD5 comment contradicted A3(ii). The copy of the 22:18 UTC fetch "
      "was found (query record accessed_utc 2026-09-24 22:18:45; rows and totals md5 equal to the pinned values), so "
      "the script, the Provenance line and data/raw/SOURCES.md 10g now record both re-fetches, 22:18:45 and 22:46:32 "
      "UTC, as byte-identical.")
    w("9. **Figure (minor).** Three panels: (a) T1 with only the extreme fields labelled; (b) T2-T4 coupling slopes "
      "with the E9 proportional predictions marked; (c) T5 coefficient slopes (x-axis: change per year in the "
      "standardized rank-regression coefficient), including pooled dG - dF uncorrected and dG* - dF* at LB-F and "
      "EXT-F.")
    w("")
    # ------------------------------------------------------------------ appendix
    w("## Appendix A: fields")
    w("")
    w("g = share of bachelor's holders aged 30-45 with a graduate degree (ACS 2023; SE from replicate weights); "
      "career slope, y1->y5 and y5->y10 segment slopes from the V4.14.1 fixed-cohort panel.")
    w("")
    w("| field | half | g (grad) | SE | prof/doc share | doctorate share | ACS records | career slope | y1->y5 | "
      "y5->y10 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in fs_tab.sort_values(["share_grad", "field"]).iterrows():
        w(f"| {r['label']} | {r['half']} | {r['share_grad']:.3f} | {r['se_grad']:.3f} | {r['share_long']:.3f} | "
          f"{r['share_phd']:.3f} | {int(r['n_ba_plus']):,} | {fnum(r['career_slope'])} | {fnum(r['early_slope'])} | "
          f"{fnum(r['late_slope'])} |")
    w("")
    w("## Appendix B: FOD1P -> field map")
    w("")
    w("| field | FOD1P codes (ACS labels) | source / flag | note |")
    w("|---|---|---|---|")
    for _, r in fs_tab.sort_values("field").iterrows():
        w(f"| {r['label']} | {r['fod1p_labels']} | {r['map_flag']} | {r['map_note']} |")
    w("")
    w("## Provenance")
    w("")
    w(f"- NCSES SED baccalaureate-origin tables (data/raw/ncses_sed_bacc/): fetched {SED_ACCESS[0]} through the "
      f"public table builder's engine (anonymous session), re-fetched {SED_ACCESS[1]} byte-identical; md5 rows "
      f"{p_['sed']['rows_md5']}, totals {p_['sed']['totals_md5']} (data/raw/SOURCES.md).")
    w("- ACS 2023 PUMS, PSEO V4.14.1, College Scorecard institution file, Wapman et al. 2022 and scripts/56's "
      "reliabilities: as documented in data/raw/SOURCES.md and the scripts they come from. LEHD schema label files "
      "(data/raw/lehd_schema/) give the IPEDS count flags.")
    w("")
    L[ANSWER_SLOT:ANSWER_SLOT] = summary_lines(R)
    ci_ = L.index("### If the graduate-study confound were real: what would and would not survive") + 2
    L[ci_:ci_] = confound_lines(R)
    OUT_MD.write_text("\n".join(L) + "\n")


def summary_lines(R) -> list:
    T1, T2, T3, T4, T5, E9 = R["T1"], R["T2"], R["T3"], R["T4"], R["T5"], R["E9"]
    ph = T2["phd"]
    st = lambda x: "undetermined" if "undetermined" in x else x.split(" (")[0]
    t5g = {s: st(T5[s]["pooled_dG*"]["status"]) for s in ("LBFo", "EXTFo", "LB", "EXT")}
    t5d = {s: st(T5[s]["pooled_dG* - dF*"]["status"]) for s in ("LBFo", "EXTFo", "LB", "EXT")}
    up, um = T5["U_pooled"], T5["dGmF_U"]
    pr = E9["pred"]
    both_in = E9["inside_lo"] and E9["inside_diff"]
    pooled_u_holds = st(up["dG - dF"]["status"]) == "holds"
    pc_und = all(st(T5[s]["dGmF"]["status"]) == "undetermined" for s in ("LBFo", "EXTFo", "LB", "EXT"))
    return [
        f"**In short.** By the pre-set rules: T1 (field level) is **{T1['verdict']}** (beta {fnum(T1['beta'])} per unit "
        f"of graduate-degree share, MDE {T1['mde']:.3f} against a proportional benchmark of {fnum(T1['beta_full'])}); "
        f"T2 (institution level) the partial rise **{T2['verdict_cov']}** coverage ({ec(T2['par'])}/yr), "
        f"**{T2['verdict_phd']}** the PhD rate ({ec(ph['pn'])}) and **{T2['verdict_both']}** both "
        f"({ec(ph['pb'])}), against a raw {ec(T2['raw'])}; T3 the y5->y10 rise in the lower half of fields by "
        f"graduate-degree share **{T3['verdict']}** ({ec(T3['lo'])}), but T3 "
        + ("does not separate the confound from its absence: " if both_in else "is informative only in part: ")
        + f"a late rise fully proportional to the share predicts {fnum(pr['lo'])}/yr in the lower half and an "
        f"upper-minus-lower difference of {fnum(pr['diff'])}, "
        + ("both inside the intervals" if both_in else "not both inside the intervals")
        + f" (observed difference {ec(T3['diff'])}; power to tell the two apart {E9['pow_lo']:.2f} with the level and "
        f"{E9['pow_diff']:.2f} with the difference); T4 the calendar-matched lower end without 2008-2010 and 2020-2022 "
        f"earnings years **{T4['verdict']}** ({ec(T4[('t4p', 'Dc')])}; strict {ec(T4[('t4s', 'Dc')])}); T5 (pooled "
        f"form, F disattenuated; {SLAB_DEF}) the corrected G slope {grp_status(t5g)}, and G's lead over F "
        f"{grp_status(t5d)}"
        + (f", as it already does not uncorrected in the same pooled form ({ec(up['dG - dF'])})" if not pooled_u_holds
           else f", although uncorrected in the same pooled form it holds ({ec(up['dG - dF'])})")
        + (f"; the per-cell uncorrected lead {ec(um)} has no determinable corrected counterpart, so the data cannot say "
           f"whether correcting F removes it" if pc_und else "")
        + f". The rise is not removed by any pre-specified check, but its size is not robust: the institution PhD-rate "
        f"control absorbs a share {fnum(ph['spn']['est'], 2, False)} [{fnum(ph['spn']['tlo'], 2, False)}, "
        f"{fnum(ph['spn']['thi'], 2, False)}] of the slope, and the data cannot say how much of that is graduate-school "
        f"timing and how much is a status gradient the PhD rate shares with prestige. The field-level tests (T1, T3) "
        f"have too little power to rule the mechanism in or out, and whether the corrected G slope is above zero "
        f"depends on how reliable the published field rank is.",
    ]


def confound_lines(R) -> list:
    T1, T2, T3, T4, T5, E5, E7, E9 = R["T1"], R["T2"], R["T3"], R["T4"], R["T5"], R["E5"], R["E7"], R["E9"]
    ph = T2["phd"]
    st = lambda x: "undetermined" if "undetermined" in x else x.split(" (")[0]
    up, um = T5["U_pooled"], T5["dGmF_U"]
    pr = E9["pred"]
    both_in = E9["inside_lo"] and E9["inside_diff"]
    biz = T3["biz"]
    halves = sorted({h for _, h in biz.values()})
    biz_l = ", ".join(f"{LAB.get(f, f)} {fnum(v, 4)}" for f, (v, _) in sorted(biz.items(), key=lambda kv: kv[1][0]))
    pc_st = sorted({st(T5[s]["dGmF"]["status"]) for s in ("LBFo", "EXTFo", "LB", "EXT")})
    return [
        "Suppose graduate study does lower the y1 medians (and, through doctoral study and medical residency, the y5 "
        "medians) of higher-status institutions more than those of lower-status ones. Then:",
        "",
        "Survives the available institution-level proxies (the confound itself is not excluded):",
        "",
        f"- **A positive rise net of coverage and the PhD rate**: {ec(ph['pb'])}/yr, "
        f"{ph['pb']['est'] / T2['raw']['est']:.2f} of the raw {fnum(T2['raw']['est'])}. This treats everything that "
        f"moves with the PhD rate as timing, which removes status signal as well. It does not exclude a remaining "
        f"timing effect through professional degrees or field-specific doctoral study. The PhD rate counts the "
        f"institution's research doctorates in all fields, not this cohort's or this field's doctorates. Law, medicine "
        f"and business enter only through coverage, and coverage records only the absence of UI earnings, so study "
        f"while working (part-time MBA or nursing master's programmes, for example), which lowers earnings without "
        f"removing them, is not captured. No public institution x field x cohort data measure these routes.",
        "",
        "Would not survive, or is not informative about the confound:",
        "",
        f"- **The size of the career slope** ({fnum(T2['raw']['est'])}/yr) as a measure of how the status gradient in "
        f"pay grows with experience: a share {fnum(ph['spb']['est'], 2, False)} [{fnum(ph['spb']['tlo'], 2, False)}, "
        f"{fnum(ph['spb']['thi'], 2, False)}] of it moves with coverage and the institution's PhD production together, "
        f"and these data cannot split that part into timing and status.",
        f"- **The y5->y10 rise where graduate degrees are least common**: {ec(T3['lo'])}/yr in the lower half of fields "
        f"by share ({T3['verdict']} by the pre-set rule), against {ec(T3['hi'])} in the upper half (difference "
        f"{ec(T3['diff'])}). A late rise fully proportional to the share predicts {fnum(pr['lo'])} and "
        f"{fnum(pr['diff'])}, " + ("both inside" if both_in else "not both inside") + " these intervals; the power to "
        f"tell that mechanism from a uniform late rise is {E9['pow_lo']:.2f} (level) and {E9['pow_diff']:.2f} "
        f"(difference), and across all fields the late slope on the share is {E9['ratio_e2']:.2f} times the "
        f"proportional benchmark (E2, E9). Master's and law programmes usually end before y5, but full-time MBA "
        f"students often enrol after several years of work, so y5 can fall in the programme. The business fields ("
        + ("all in the " + halves[0] + " half" if len(halves) == 1 else "in both halves")
        + f"; {biz_l}) have y5->y10 slopes from {fnum(min(v for v, _ in biz.values()))} to "
        f"{fnum(max(v for v, _ in biz.values()))}, so this route did not produce a large late rise there. Doctoral "
        f"study and residency can depress y5 in every field.",
        f"- **y1 coupling as coupling at entry, and the size of the y1->y5 segment**: y1 is where enrolment depresses UI "
        f"earnings. The y1->y5 segment is positive in both halves of fields (lower {ec(E5['lower'])}, upper "
        f"{ec(E5['upper'])}; exploratory E5), but its size is exposed to the confound.",
        f"- **The calendar-matched contrast as a lower bound on career time**: its y1 cell (cohort c+9) is as exposed "
        f"to enrolment as any y1 cell, so under the confound D_cal is biased upward too. T4 shows only that the lower "
        f"end does not depend on the recession and pandemic years ({ec(T4[('t4p', 'Dc')])}).",
        f"- **The G loading as evidence against the confound**: the PhD rate is an institution-wide trait that tracks "
        f"the academia-wide rank G (within-cell Spearman {fnum(E7['rGD'], 3)}; with field prestige F "
        f"{fnum(E7['rPD'], 3)}; means over fields), so graduate-school timing would load on G as well. G's lead over F "
        f"is itself not established. Pooled form: uncorrected {ec(up['dG - dF'])} ({st(up['dG - dF']['status'])}), "
        + "; ".join(f"{SLAB[s]} {ec(T5[s]['pooled_dG* - dF*'])} ({st(T5[s]['pooled_dG* - dF*']['status'])})"
                    for s in ("LBFo", "EXTFo"))
        + f". Per-cell form: uncorrected {ec(um)} ({st(um['status'])}), corrected intervals "
        + " / ".join(pc_st)
        + f" (exploratory IQR-based interval at EXT-F [{fnum(T5['EXTFo']['dGmF_rob'][0])}, "
          f"{fnum(T5['EXTFo']['dGmF_rob'][1])}]). Even where G leads, that is what timing would produce.",
        f"- **A field-level verdict either way**: T1's power is {T1['pow_full']:.2f} against a rise proportional to "
        f"the share and {T1['pow_half']:.2f} against half of that; both that mechanism and no share gradient lie "
        f"inside the interval.",
    ]


if __name__ == "__main__":
    main()
