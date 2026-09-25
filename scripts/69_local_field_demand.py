"""Local field-specific labour demand: does the within-institution department-prestige slope survive a control for
how well the institution's local labour market pays the department's field?

Referee problem. Institution fixed effects remove institution-wide averages (and so every state-wide or metro-wide
earnings level), but not FIELD-SPECIFIC local demand: a department ranked above its institution's other departments
may sit in a local market that pays that field well (computer science near a tech hub, petroleum engineering in
Texas). This script builds field x state (and field x metro) earnings levels for young bachelor's holders from the
public ACS 2023 1-year PUMS and adds them to the scripts/55/58 within-institution design. Public data only;
descriptive, not causal.

Objects
  cells     program = (field f, institution i): Scorecard FoS bachelor's median earnings 4 yrs after completion
            (EARN_MDN_4YR), Wapman field prestige F (-field rank), academia-wide brand G, institution SAT_AVG,
            ADM_RATE, PCTPELL (scripts/55 build_cells; scripts/58 conventions, reused by import).
  ACS       2023 1-year PUMS persons (data/raw/acs/psam_pus{a,b}.csv, read in chunks with usecols): ages 23-35,
            bachelor's as highest degree (SCHL 21), full-time full-year (WKHP >= 35, WKWN >= 50), civilian employed
            (ESR 1/2), not enrolled (SCH 1), earnings PERNP > 0 with implied hourly earnings >= half the federal
            minimum wage (PERNP / (WKHP * WKWN) >= 3.625). y = log(PERNP * ADJINC / 1e6). Person weight PWGTP; the 80
            successive-difference replicate weights PWGTP1..80 carry the ACS sampling error.
  FOD1P map src/crosswalks/fields.py fod1p_by_key_all() (52 fields) + this script's map for 11 of the 12 fields
            scripts/24 recovered (RECOVER_FOD1P; urban planning is left unmapped: the ACS has no planning
            category of its own, and the field has fewer than 15 programs) + the two "gap-only" fields whose ACS
            category is shared (biostatistics -> 3702, religious studies -> 4801). Each field is tagged exclusive /
            coarse / shared.
  L_fs      field x state earnings level (state of residence). Per cell (f, s): weighted mean of y (m_fs), sampling
            variance v_fs = deff * sigma2_f / n_eff_fs (pooled within-field variance, Kish effective n, one design
            effect deff = median SDR-replicate / model variance ratio over cells with n >= 30). Additive fit
            m_fs = a_f + b_s + u_fs with weights 1/(v_fs + tau2); tau2 = Paule-Mandel moment estimate (one value over
            all fields). Empirical-Bayes field-specific local component U_fs = tau2 / (tau2 + v_fs) * u_fs (0 where the
            ACS has no person in the cell). L_fs = a_f + b_s + U_fs. With institution FE (which absorb b_s) and field
            FE (a_f), a regression on L and a regression on U give the same coefficients; U is what identifies.
  metro     field x metro (CBSA) version: 2020 PUMAs -> counties (Census 2020 tract-to-PUMA relationship file,
            majority of tracts) -> March 2020 OMB metropolitan statistical areas; institutions -> county (IPEDS HD2023
            COUNTYCD; HD2023 CBSA where the county code is not in the 2020 delineation, i.e. the Connecticut planning
            regions) -> metro. Units = metros plus one non-metro remainder per state. Hierarchical EB: the metro's
            field-specific deviation from the state-level U of its principal state is shrunk with its own
            Paule-Mandel variance; programs outside a metro keep the state-level U.

Within-institution design (scripts/58 main spec "FE2", 4-yr earnings; reused by import, s58.fe_build / fe_fit):
    log Y_if = alpha_i + gamma_f + beta z(F)_if + sum_f D_f [d_f zSAT_i + e_f zADM_i + p_f zPELL_i + t_f zG_i]
               (+ delta U_{f, s(i)})  + e
  before = without U, after = with U, on the SAME programs (the sample requires U; every FE2 field has an ACS
  category, so the programs are exactly scripts/58's 2,889).

PRE-SPECIFIED TESTS (fixed before any estimate was looked at; everything else is exploratory and labelled so)
  T1  Slope change: Delta = beta_after - beta_before (FE2, 4-yr earnings, state-level U, common delta), with its
      95% CI; also beta_before, beta_after, delta, and the share of beta removed. Null reported with the minimum
      detectable |Delta| at 80% power (two-sided 5%): 2.80 x SE(Delta).
  T2  Computer-science remainder before/after: CS beta_f in the per-field version of FE2 (common delta on U),
      (a) 4-yr earnings on the T1 programs, (b) average of the 4-, 1- and 5-yr estimates on the programs released at
      all three horizons (scripts/58's headline CS remainder). Before, after, paired difference, CIs.
  T3  (exploratory) How much of the cross-field ordering of the baseline coupling rho_f tracks field-state earnings
      dispersion: across fields, Spearman(rho_f, tau_f) (tau_f = field-specific SD of the true field-state
      component, per-field Paule-Mandel on the ACS cells), Spearman(rho_f, SD of U over the field's programs),
      Spearman(rho_f, rho(F, U)_f), and the ordering and mean of rho_f against the partial rho(F, Y | U).
  Also reported (asked for): SD of the residualised z(F) within institution (how much department-rank variation
  the design uses), before and after U, in within-field SD units and in field-rank positions.

Inference
  * scripts/58 standard: CR1 SEs clustered by institution and the institution (pairs) cluster bootstrap percentile
    CI (B = 999). The scripts/58 FE2 CI is reproduced with s58.fe_group_boot and scripts/58's own seed (asserted).
  * Project standard where fields and institutions both recur (all pooled slopes and cross-field statistics): the
    two-way (field, institution) cluster variance of scripts/59 (s59.twoway, reused by import):
    V = V_field + V_inst - V_field x inst, from the jackknife over fields, the replicates of one institution draw
    shared by all fields, and the replicates of an independent institution draw per field (B = 999 each).
    Cross-check: the analytic Cameron-Gelbach-Miller sandwich (CR1 by field + CR1 by institution - HC1).
  * ACS sampling error of the generated regressor U: the whole EB pipeline is re-run with each of the 80 ACS
    replicate weights and the model refitted; V_acs is added to V for every statistic that uses U in T1/T2/T3 (the
    ACS and Scorecard samples are independent). Since the revision (see REVISION NOTE) V_acs is centred at the
    replicate mean, 4/80 sum_r (theta_r - mean_r theta_r)^2; the theta0-centred form 4/80 sum_r (theta_r - theta)^2
    of the first version is reported beside it.
  Seeds: one numpy SeedSequence child per replicate (SeedSequence([SEED, crc32(specset), kind]).spawn(B)), so results
  are identical for any number of worker processes; replicate work runs with BLAS limited to one thread per process.
  The scripts/58 bootstraps keep scripts/58's seeds ([58, 3, 1], [58, 5], [58, 11]) so before/after draws are paired.

Exploratory sensitivities (labelled so in the write-up): unshrunk U; field-specific tau2_f shrinkage; delta fixed
at 1 (offset); field-specific delta_f; state of work (POWSP) instead of residence; exclusive-mapping fields only;
metro-level U; a leave-field-out occupation-mix index (national occupation mix of field f's young graduates x
state occupation earnings of the other fields' graduates, shrunk toward the national occupation mean with a prior
of 10 persons); specs (a) FE0, FE1 and FE2sg of scripts/58 before/after.

BUILD NOTE (2026-09-25; implementation decisions taken while finishing the interrupted build, before any T1/T2/T3
estimate had been computed; none changes a test above)
  * EB on ACS CATEGORIES, not project fields: fields that share an ACS category (kinesiology / hper -> 4101,
    statistics / biostatistics -> 3702, philosophy / religious studies -> 4801) would otherwise enter the additive
    fit and the Paule-Mandel moment twice with identical data. The EB runs once per distinct FOD1P code set and each
    field takes its category's U (tau_f = its category's tau).
  * Mapping kinds: "exclusive" now requires that no other project field uses the category, so statistics and
    philosophy are "shared" (the partial build left their canonical tag). Only the exclusive-only sensitivity uses it.
  * Additive fit solved from its normal equations (field and geography dummies are sparse); same estimator as the
    weighted least squares of the partial build, fast enough for the ~440 metro units.
  * Leave-one-field-out replicates: when the reference field is the one left out, the first remaining field's dummy
    is dropped (otherwise the remaining field dummies are collinear with the institution FE).
  * Primary interval for T1 and the pooled sensitivities: two-way (field, institution) variance + V_acs (project
    standard); the scripts/58 institution cluster bootstrap percentile CI is reported beside it. For T2 (one field's
    coefficient, a single field cluster) the scripts/58 institution cluster bootstrap is the standard; the normal CI
    with V_boot + V_acs is reported beside the percentile CI.
  * Metro rule: the prior of a metro unit is the state-level U of the state with most sample persons in the unit; an
    institution whose metro has no ACS person of its field takes that prior; institutions outside a metro, or in a
    metro absent from the ACS units, keep their own state's U.
  * V_acs is not computed for the occupation-mix index (its replicate rebuild is not implemented); that row says so.

VERIFICATION NOTE (2026-09-25, second session; the interrupted build had run once with 19 replicates for debugging,
and its outputs were not read before these changes; no test above changed)
  * Bug fix, metro level: the partial build coded a person outside every MSA as unit -STATE, and cell_sums drops
    negative geography codes, so the per-state non-metro remainders the docstring describes never entered the
    metro-level fit. They are now coded NONMETRO_BASE + STATE (CBSA codes are < NONMETRO_BASE; asserted).
  * Added (exploratory, labelled so): computer-science beta_f with the metro-level U in place of the state-level U
    (same programs, same scripts/58 bootstrap draws, so paired with T2); descriptive statistics of U on the T1
    programs (SD over programs, SD within institution, within-institution correlation with z(F), and the slope pi of
    U on the residualised z(F), with the omitted-variable identity beta_after - beta_before = -delta * pi asserted);
    the computer-science U by state; and spec (a) (institution FE + field FE + beta z(F)) on its OWN programs, which
    keeps institutions without SAT_AVG (the main spec drops them, e.g. the test-blind University of California
    campuses): pooled beta and CS beta_f before / after (state U) / metro U, two-way + ACS for the pooled slope and an
    institution cluster bootstrap (this script's seeds [69, 20] and [69, 21], paired) for both.
  * Provenance of the three downloaded files is in data/raw/SOURCES.md under the subsection titled
    SOURCES_SECTION (the partial build's "10d" label had meanwhile been taken by scripts/68).
  * Replicate engines free the parent's large temporaries before forking (memory cap of the lane).
  * Metro level: the Paule-Mandel moment for the metro deviation d now uses df = cells - rank of the unit-level
    additive fit (the partial build used cells - 1, which biases tau2_d toward 0). Found because tau_d came out 0 in
    the first full run; it is 0 under either df (sum d^2 / v is below both), so the metro-level U still equals the
    principal state's U, and the write-up says so.

REVISION NOTE (2026-09-25, after an independent verification of the first full run; T1/T2/T3 and every estimator of
a point estimate unchanged)
  * ACS replicate variance centred at the replicate mean. Each replicate re-runs Paule-Mandel with v held at its
    main-weight value, but replicate-weighted cell means carry about v/4 of extra perturbation variance, so tau2 is
    larger in (nearly) every replicate and statistics that depend on the shrinkage are shifted in every replicate.
    The theta0-centred form then adds 4 x (mean shift)^2 to the sampling variance. Primary V_acs is now
    4/80 sum_r (theta_r - mean_r theta_r)^2 (the "mse = FALSE" form of replicate variance); the theta0-centred value,
    the replicate shift mean_r theta_r - theta0 and the replicate tau distribution are reported beside it. Added as
    a check: V_acs with tau2 (and tau2_g) held at the main-weight values in every replicate (T1, T2, T3).
  * Correlation-type T3 statistics get a Fisher-z interval (normal interval on atanh scale, SE by the delta method,
    back-transformed); the normal interval stays in the CSV.
  * Added (exploratory, post hoc, labelled so): T3 diagnostics of what the SD of the shrunk U over a field's programs
    measures: Spearman(SD of U, ACS persons of the category), Spearman(SD of U, tau_f), Spearman(rho_f, ACS persons),
    and the rank-partial Spearman(rho_f, SD of U | ACS persons).
  * The design effect stays the pre-specified median ratio; the mean ratio and the ratio of sums over the same cells
    are reported, and a sensitivity model (E_dsum) rebuilds U with deff = ratio of sums.
  * Write-up: T1 delta statement, T2 wording as a bounded change, T3 verdict, build history and a Revisions section.

Reproducible: seeded; outputs byte-identical on re-run (any worker count).
Run: OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
     .venv/bin/python scripts/69_local_field_demand.py [--workers N]
Outputs: data/interim/local_field_demand.csv (every reported number), data/interim/local_field_demand_cells.csv
         (field x state L, U, n, lambda), data/interim/local_field_demand_fields.csv (T3 per-field table),
         data/interim/local_field_demand_acs.parquet (+ _counts.csv; the ACS extract, rebuilt when missing),
         LOCAL_FIELD_DEMAND_RESULT.md (root; gitignored).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import sys
import gc
import zlib
import time
import hashlib
import zipfile
import argparse
import importlib.util
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import rankdata, spearmanr
import scipy.sparse as sps
from scipy.sparse.csgraph import connected_components
import statsmodels.api as sm
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_ar import load_ar_wapman
from src.load_er import load_er_scorecard
from src.crosswalks import fields as F


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s58 = _load("s58", "58_selectivity_deep.py")     # within-institution design (fe_sample / fe_build / fe_fit / boot)
s55 = s58.s55                                     # cells, institution file, rank partials
s59 = _load("s59", "59_pseo_refresh.py")         # two-way (field, institution) cluster variance
FIELDS66, LAB = s58.FIELDS66, s58.LAB
twoway, Z975 = s59.twoway, s59.Z975

SEED = 69
B = 999                  # replicates per bootstrap kind (as scripts/58 B_FE)
B_REP = B                # two-way replicate engines (this script's seeds); = B
N_REP = 80               # ACS successive-difference replicate weights
NMIN = s58.NMIN          # 15
AGE_MIN, AGE_MAX = 23, 35
FT_HOURS, FY_WEEKS = 35, 50
HOURLY_FLOOR = 7.25 / 2  # half the federal minimum wage
DEFF_NMIN = 30           # cells used to calibrate the design effect
SIG_DFMIN = 30           # within-field df below which the pooled within-cell variance is used
OCC_PRIOR = 10.0         # occupation-mix index: prior strength (persons) toward the national occupation mean
PF_MIN_DISTINCT = s58.FE_PF_MIN_DISTINCT
CHUNK = 50_000
MDE_K = 2.80             # 80% power, two-sided 5%: (1.96 + 0.84) x SE
MEM_MIN_MB = 1500        # fall back to one worker below this much available memory

ACS_DIR = ROOT / "data" / "raw" / "acs"
ACS_FILES = ["psam_pusa.csv", "psam_pusb.csv"]
ACS_DICT = ACS_DIR / "PUMS_Data_Dictionary_2023.txt"
REPW = [f"PWGTP{r}" for r in range(1, N_REP + 1)]
ACS_KEEP = ["STATE", "PUMA", "POWSP", "AGEP", "SCHL", "SCH", "ESR", "WKHP", "WKWN", "PERNP", "ADJINC", "FOD1P",
            "OCCP", "PWGTP"] + REPW
GEO_DIR = ROOT / "data" / "raw" / "local_demand_geo"
TRACT_PUMA = GEO_DIR / "2020_Census_Tract_to_2020_PUMA.txt"
CBSA_XLS = GEO_DIR / "list1_2020.xls"
HD_ZIP = GEO_DIR / "HD2023.zip"
INST_FILE = s55.INST_FILE
EXTRACT = ROOT / "data" / "interim" / "local_field_demand_acs.parquet"
EXTRACT_COUNTS = ROOT / "data" / "interim" / "local_field_demand_acs_counts.csv"
OUT_CSV = ROOT / "data" / "interim" / "local_field_demand.csv"
OUT_CELLS = ROOT / "data" / "interim" / "local_field_demand_cells.csv"
OUT_FIELDS = ROOT / "data" / "interim" / "local_field_demand_fields.csv"
OUT_MD = ROOT / "LOCAL_FIELD_DEMAND_RESULT.md"
DEEP_CSV = s58.OUT_CSV
GAP_MAP = s58.GAP_MAP

NONMETRO_BASE = 99000   # metro-level unit code of a state's non-metro remainder: NONMETRO_BASE + state FIPS

# provenance of the files this script downloaded (data/raw/SOURCES.md, subsection SOURCES_SECTION); md5 asserted
SOURCES_SECTION = "Local field demand — geography files for ACS field × state / metro earnings (scripts/69)"
PROV = {TRACT_PUMA: "466d01af6c9a04e960fac4f14c321005", CBSA_XLS: "8c7e7feb95afe5bb43d1612d33007bf3",
        HD_ZIP: "a6ffbba5496eed4c5ec3f6b513b8fea9"}
PROV_URL = {
    TRACT_PUMA: "https://www2.census.gov/geo/docs/maps-data/data/rel2020/2020_Census_Tract_to_2020_PUMA.txt",
    CBSA_XLS: "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2020/"
              "delineation-files/list1_2020.xls",
    HD_ZIP: "https://nces.ed.gov/ipeds/datacenter/data/HD2023.zip"}

# scripts/58 reproduction targets
FE2_SLOPES = list(s58.INST_SLOPES) + ["G"]
S58_FE2_GI = next(i for i, g in enumerate(s58.FE_GROUPS) if "FE2" in g)
S58_FE2_SEED = [s58.SEED, 3, S58_FE2_GI]           # scripts/58 compute(): fe_group_boot(..., [SEED, 3, gi])
S58_PF_SEED = [s58.SEED, 5]                         # per-field FE2 bootstrap (4-yr)
S58_H3_SEED = [s58.SEED, 11]                        # per-field FE2 at each horizon, common programs
SEED_A_POOL = [SEED, 20]                            # exploratory spec (a) on its own sample: pooled, paired draws
SEED_A_PF = [SEED, 21]                              # exploratory spec (a) on its own sample: per-field, paired draws
PF_MIN_DISTINCT_A = 4                               # spec (a) per field: intercept + slope (2 parameters) + 2
S58_FE2_PUB = (0.0084, 0.0000, 0.0164)              # SELECTIVITY_DEEP_RESULT.md Key numbers (main spec, 4 dp)
CSV_TOL = s58.CSV_TOL
HZ3 = s58.HZ3
HZ_COL = s58.HZ_COL
CS = "computer_science"

# FOD1P for the 12 fields recovered by scripts/24 (codes checked against the 2023 PUMS dictionary at run time).
# kind: exclusive = one ACS category that no other project field uses; coarse = the ACS category is broader than the
# field; shared = the category is also used by another project field (both get the same local earnings level).
RECOVER_FOD1P = {
    "linguistics": (["2601"], "exclusive"),               # Linguistics And Comparative Language And Literature
    "spanish": (["2602"], "coarse"),                      # French German Latin And Other Common Foreign Language Studies
    "nutrition": (["4002"], "exclusive"),                 # Nutrition Sciences
    "theology": (["4901"], "exclusive"),                  # Theology And Religious Vocations
    "ag_engineering": (["2402"], "coarse"),               # Biological Engineering
    "education_general": (["2300"], "exclusive"),         # General Education
    "education_admin": (["2301"], "exclusive"),           # Educational Administration And Supervision
    "teacher_ed_subjects": (["2304", "2305", "2306", "2307", "2308", "2309", "2311", "2312", "2313", "2314"],
                            "coarse"),                    # level- and subject-specific teacher education
    "kinesiology": (["4101"], "shared"),                  # Physical Fitness Parks Recreation And Leisure
    "hper": (["4101"], "shared"),                         # same category
    "human_dev": (["2901"], "coarse"),                    # Family And Consumer Sciences
}
SHARED_FOD1P = {"biostatistics": ["3702"], "religious_studies": ["4801"]}   # categories of statistics / philosophy

# U variants (program-level columns) built from the ACS; "U" is the pre-specified state-of-residence EB component
UVARS = ["U", "U_raw", "U_ftau", "U_pow", "U_met", "U_dsum", "L_occ"]
UVAR_ACS = ["U", "U_raw", "U_ftau", "U_pow", "U_met", "U_dsum"]          # re-built for every ACS replicate weight

# models of the main specset (identical programs: the FE2 sample, which requires every U variant)
#   label: (slopes, extra common columns, extra field-specific columns, offset, description)
MAIN_MODELS = {
    "before": (FE2_SLOPES, (), (), None, "FE2 [scripts/58 main], no local control"),
    "after": (FE2_SLOPES, ("U",), (), None, "FE2 + common δ × U (state of residence, EB) [T1]"),
    "E_raw": (FE2_SLOPES, ("U_raw",), (), None, "FE2 + raw (unshrunk) log field-state mean"),
    "E_ftau": (FE2_SLOPES, ("U_ftau",), (), None, "FE2 + U shrunk with field-specific τ²_f"),
    "E_offset": (FE2_SLOPES, (), (), "U", "FE2 with U as an offset (δ fixed at 1)"),
    "E_pf": (FE2_SLOPES, (), ("U",), None, "FE2 + field-specific δ_f × U"),
    "E_pow": (FE2_SLOPES, ("U_pow",), (), None, "FE2 + U by state of WORK (POWSP)"),
    "E_met": (FE2_SLOPES, ("U_met",), (), None, "FE2 + metro-level U (hierarchical EB)"),
    "E_occ": (FE2_SLOPES, ("L_occ",), (), None, "FE2 + leave-field-out occupation-mix index"),
    "E_dsum": (FE2_SLOPES, ("U_dsum",), (), None, "FE2 + U with deff = ratio of sums (revision)"),
    "FE0_b": ([], (), (), None, "(a) institution FE + field FE + β z(F), FE2 programs"),
    "FE0_a": ([], ("U",), (), None, "(a) + common δ × U"),
    "FE1_b": (list(s58.INST_SLOPES), (), (), None, "(a) + field × {SAT, ADM, inst. Pell}"),
    "FE1_a": (list(s58.INST_SLOPES), ("U",), (), None, "same + common δ × U"),
    "FE2sg_b": (["SAT_AVG", "G"], (), (), None, "(a) + field × {SAT, brand G}"),
    "FE2sg_a": (["SAT_AVG", "G"], ("U",), (), None, "same + common δ × U"),
}
# (after, before) pairs: Delta = beta_after - beta_before on identical programs
MAIN_PAIRS = [("after", "before"), ("E_raw", "before"), ("E_ftau", "before"), ("E_offset", "before"),
              ("E_pf", "before"), ("E_pow", "before"), ("E_met", "before"), ("E_occ", "before"), ("E_dsum", "before"),
              ("FE0_a", "FE0_b"), ("FE1_a", "FE1_b"), ("FE2sg_a", "FE2sg_b")]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage(msg: str, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def mem_available_mb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


# =============================================================================================
# ACS extract
# =============================================================================================
def _extract_file(fname: str):
    """Chunked read of one PUMS person file (usecols only); returns the analysis sample and attrition counts."""
    cnt = dict(file=fname, all=0, age=0, ba=0, ftfy=0, employed=0, not_enrolled=0, pos_earn=0, floor=0)
    out = []
    for ch in pd.read_csv(ACS_DIR / fname, usecols=ACS_KEEP, dtype=np.float64, chunksize=CHUNK):
        cnt["all"] += len(ch)
        ch = ch[(ch.AGEP >= AGE_MIN) & (ch.AGEP <= AGE_MAX)]; cnt["age"] += len(ch)
        ch = ch[ch.SCHL == 21]; cnt["ba"] += len(ch)
        ch = ch[(ch.WKHP >= FT_HOURS) & (ch.WKWN >= FY_WEEKS)]; cnt["ftfy"] += len(ch)
        ch = ch[ch.ESR.isin([1, 2])]; cnt["employed"] += len(ch)
        ch = ch[ch.SCH == 1]; cnt["not_enrolled"] += len(ch)
        ch = ch[ch.PERNP > 0]; cnt["pos_earn"] += len(ch)
        ch = ch[ch.PERNP / (ch.WKHP * ch.WKWN) >= HOURLY_FLOOR]; cnt["floor"] += len(ch)
        out.append(ch[ACS_KEEP].copy())
    cnt["bytes"] = int((ACS_DIR / fname).stat().st_size)
    cnt["md5"] = md5(ACS_DIR / fname)
    return pd.concat(out, ignore_index=True), cnt


def acs_extract(workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if EXTRACT.exists() and EXTRACT_COUNTS.exists():
        return pd.read_parquet(EXTRACT), pd.read_csv(EXTRACT_COUNTS)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("fork")) as ex:
            res = list(ex.map(_extract_file, ACS_FILES))
    else:
        res = [_extract_file(f) for f in ACS_FILES]
    A = pd.concat([r[0] for r in res], ignore_index=True)
    assert A.FOD1P.notna().all() and A.PWGTP.gt(0).all()
    intc = [c for c in ACS_KEEP if c not in ("POWSP",)]
    assert A[intc].notna().all().all()
    A["POWSP"] = A.POWSP.fillna(-1)
    A = A.astype({c: np.int32 for c in intc + ["POWSP"]})
    A.to_parquet(EXTRACT, index=False)
    C = pd.DataFrame([r[1] for r in res])
    C.to_csv(EXTRACT_COUNTS, index=False)
    return A, C


def acs_y(A: pd.DataFrame) -> np.ndarray:
    """log earnings in 2023 dollars (float arithmetic: PERNP * ADJINC overflows int32)."""
    return np.log(A.PERNP.to_numpy(np.float64) * A.ADJINC.to_numpy(np.float64) / 1e6)


def fod1p_codes_in_dictionary() -> dict:
    lines = ACS_DICT.read_text(encoding="latin-1").splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.startswith("FOD1P "))
    out = {}
    for l in lines[i0 + 2:]:
        s = l.strip()
        if not s:
            break
        code, _, lab = s.partition(".")
        code = code.strip()
        if code.isdigit() and len(code) == 4:
            out[code] = lab.strip()
    return out


def field_map() -> tuple[dict, dict]:
    """field -> FOD1P codes and field -> mapping kind, over the 66-field universe."""
    m = {k: list(v) for k, v in F.fod1p_by_key_all().items()}
    kind = {k: "exclusive" for k in m}
    for k, (codes, kd) in RECOVER_FOD1P.items():
        assert k not in m
        m[k], kind[k] = list(codes), kd
    for k, codes in SHARED_FOD1P.items():
        assert k not in m
        m[k], kind[k] = list(codes), "shared"
    # a canonical ("exclusive") field whose category another project field also uses is "shared" (statistics,
    # philosophy); build note
    users = defaultdict(set)
    for k, codes in m.items():
        for c in codes:
            users[c].add(k)
    for k in m:
        if kind[k] == "exclusive" and any(len(users[c]) > 1 for c in m[k]):
            kind[k] = "shared"
    D = fod1p_codes_in_dictionary()
    assert len(D) == 174, len(D)
    for k, codes in m.items():
        for c in codes:
            assert c in D, (k, c)
    keys66 = {f["key"] for f in FIELDS66}
    assert set(m) <= keys66, set(m) - keys66
    return m, kind


def acs_groups(fmap: dict) -> tuple[dict, dict]:
    """Distinct ACS categories (FOD1P code sets): field -> group key, group key -> codes. Code sets of different
    fields are either identical or disjoint (asserted)."""
    f2g, g2c = {}, {}
    for f in sorted(fmap):
        g = "+".join(sorted(fmap[f]))
        f2g[f] = g
        g2c[g] = sorted(fmap[f])
    allc = [c for cs in g2c.values() for c in cs]
    assert len(allc) == len(set(allc)), "overlapping FOD1P code sets"
    return f2g, g2c


# =============================================================================================
# geography
# =============================================================================================
def state_fips() -> dict:
    d = pd.read_csv(INST_FILE, usecols=["STABBR", "ST_FIPS"], dtype=str).dropna().drop_duplicates()
    d = d[d.ST_FIPS.str.fullmatch(r"\d+")]
    d["ST_FIPS"] = d.ST_FIPS.astype(int)
    g = d.groupby("STABBR").ST_FIPS.nunique()
    assert (g == 1).all()
    return dict(zip(d.STABBR, d.ST_FIPS))


def metro_maps() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """PUMA -> metro (majority of tracts, March 2020 MSAs); UNITID -> metro. Returns (puma_map, inst_map, info)."""
    for p, h in PROV.items():
        assert md5(p) == h, f"{p.name}: md5 changed"
    src_md = (ROOT / "data" / "raw" / "SOURCES.md").read_text(encoding="utf-8")
    assert SOURCES_SECTION in src_md and all(h in src_md for h in PROV.values()), "provenance entry missing"
    x = pd.read_excel(CBSA_XLS, header=2, dtype=str)
    x = x[x["FIPS State Code"].notna() & x["FIPS County Code"].notna()]
    x = x[x["Metropolitan/Micropolitan Statistical Area"] == "Metropolitan Statistical Area"]
    x["county"] = x["FIPS State Code"].str.zfill(2) + x["FIPS County Code"].str.zfill(3)
    c2m = dict(zip(x.county, x["CBSA Code"].astype(int)))
    mtitle = dict(zip(x["CBSA Code"].astype(int), x["CBSA Title"]))
    t = pd.read_csv(TRACT_PUMA, dtype=str, encoding="utf-8-sig")
    t["county"] = t.STATEFP + t.COUNTYFP
    t["cbsa"] = t.county.map(c2m).fillna(-1).astype(int)
    t["puma"] = t.STATEFP.astype(int) * 100000 + t.PUMA5CE.astype(int)
    g = t.groupby(["puma", "cbsa"]).size().rename("k").reset_index()
    tot = g.groupby("puma").k.transform("sum")
    g["share"] = g.k / tot
    g = g.sort_values(["puma", "share", "cbsa"], ascending=[True, False, True]).drop_duplicates("puma")
    g["cbsa"] = np.where(g.share > 0.5, g.cbsa, -1)
    pm = g[["puma", "cbsa", "share"]].reset_index(drop=True)
    hd = pd.read_csv(zipfile.ZipFile(HD_ZIP).open("HD2023.csv"), dtype=str, encoding="utf-8-sig",
                     usecols=["UNITID", "STABBR", "COUNTYCD", "CBSA", "CBSATYPE"])
    hd["county"] = hd.COUNTYCD.str.zfill(5)
    hd["cbsa"] = hd.county.map(c2m)
    via_hd = hd.cbsa.isna() & (hd.CBSATYPE == "1") & pd.to_numeric(hd.CBSA, errors="coerce").isin(set(mtitle))
    hd.loc[via_hd, "cbsa"] = pd.to_numeric(hd.loc[via_hd, "CBSA"], errors="coerce")
    hd["cbsa"] = hd.cbsa.fillna(-1).astype(int)
    hd["via_hd"] = via_hd
    info = dict(n_msa=len(mtitle), n_puma=len(pm), n_puma_metro=int((pm.cbsa > 0).sum()),
                puma_share_min=float(pm[pm.cbsa > 0].share.min()), puma_split=int((g.share < 1).sum()),
                mtitle=mtitle)
    return pm, hd[["UNITID", "cbsa", "via_hd"]], info


# =============================================================================================
# field x geography cell sums and the empirical-Bayes local component
# =============================================================================================
def cell_sums(A: pd.DataFrame, g2c: dict, geo: str) -> pd.DataFrame:
    """Per (ACS group, geo): n, SW2 = sum w^2, SWY2 = sum w y^2 (main weight), and for r = 0..80 (0 = main weight)
    W_r = sum w_r, WY_r = sum w_r y. Columns W0..W80, Y0..Y80. Persons with geo < 0 are left out."""
    wc = ["PWGTP"] + REPW
    Wm = A[wc].to_numpy(np.float64)
    y = A["y"].to_numpy(np.float64)
    gv = A[geo].to_numpy()
    fod = A["FOD1P"].to_numpy()
    rows = []
    for grp in sorted(g2c):
        codes = np.array([int(c) for c in g2c[grp]])
        m = np.isin(fod, codes) & (gv >= 0)
        if not m.any():
            continue
        g_, inv = np.unique(gv[m], return_inverse=True)
        Wf, yf = Wm[m], y[m]
        k = len(g_)
        n = np.bincount(inv, minlength=k)
        S = np.zeros((k, 2 * len(wc) + 2))
        np.add.at(S, inv, np.hstack([Wf, Wf * yf[:, None], (Wf[:, 0] ** 2)[:, None], (Wf[:, 0] * yf ** 2)[:, None]]))
        df = pd.DataFrame(S, columns=[f"W{r}" for r in range(len(wc))] + [f"Y{r}" for r in range(len(wc))]
                          + ["SW2", "SWY2"])
        df.insert(0, "n", n)
        df.insert(0, "geo", g_)
        df.insert(0, "grp", grp)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _additive_fit(m, w, fi, gi, nf, ng):
    """Weighted LS of m on group + geography dummies (first geography dropped), from the normal equations.
    Returns fitted, a_f, b_g."""
    p = nf + ng - 1
    N = np.zeros((p, p))
    Wf = np.bincount(fi, weights=w, minlength=nf)
    Wg = np.bincount(gi, weights=w, minlength=ng)
    N[np.arange(nf), np.arange(nf)] = Wf
    N[nf + np.arange(ng - 1), nf + np.arange(ng - 1)] = Wg[1:]
    cross = np.zeros((nf, ng))
    np.add.at(cross, (fi, gi), w)
    N[:nf, nf:] = cross[:, 1:]
    N[nf:, :nf] = cross[:, 1:].T
    rhs = np.concatenate([np.bincount(fi, weights=w * m, minlength=nf),
                          np.bincount(gi, weights=w * m, minlength=ng)[1:]])
    try:
        coef = np.linalg.solve(N, rhs)
        if not np.all(np.isfinite(coef)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(N, rhs, rcond=None)[0]
    a = coef[:nf]
    b = np.concatenate([[0.0], coef[nf:]])
    return a[fi] + b[gi], a, b


def _design_rank(fi, gi, nf, ng) -> int:
    """Rank of the group + geography dummy design (connected components: nf + ng - #components)."""
    adj = sps.coo_matrix((np.ones(len(fi)), (fi, nf + gi)), shape=(nf + ng, nf + ng))
    ncomp = connected_components(adj + adj.T, directed=False)[0]
    return nf + ng - ncomp


def _pm_tau2(u_of, v, df):
    """Paule-Mandel: tau2 >= 0 solving sum u(tau2)^2 / (v + tau2) = df (u may depend on tau2)."""
    g = lambda t: float(np.sum(u_of(t) ** 2 / (v + t))) - df
    if g(0.0) <= 0:
        return 0.0
    hi = 1.0
    while g(hi) > 0:
        hi *= 4
    return float(brentq(g, 0.0, hi, xtol=1e-12, rtol=1e-12))


def eb_level(C: pd.DataFrame, r: int = 0, deff: float | None = None, tau2_fix: float | None = None,
             tau2_f_fix: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Empirical-Bayes group-specific local component from cell sums C (one geography level) with weight set r.
    Variance model (sigma2_g, n_eff, deff) always from the main weight; the cell means from weight set r.
    Returns per cell: U (one tau2 over all groups), U_ftau (group-specific tau2_g where the group has >= 5 cells),
    U_raw (unshrunk residual u). tau2_fix / tau2_f_fix (revision check): use these variances instead of the
    Paule-Mandel estimates (the replicate-variance variant with the shrinkage held at its main-weight value)."""
    C = C.copy()
    W0, Y0 = C["W0"].to_numpy(), C["Y0"].to_numpy()
    mean0 = Y0 / W0
    info = {}
    C["neff"] = W0 ** 2 / C["SW2"]
    within = C["SWY2"] - Y0 ** 2 / W0                      # sum w (y - mean)^2 within cell
    # pooled within-cell variance per group (weighted, df-corrected by persons - cells); groups with fewer than
    # SIG_DFMIN within-cell degrees of freedom use the pooled value over all groups
    g_ = C.groupby("grp")
    wf, Wf_, nf_, kf_ = within.groupby(C.grp).sum(), g_.W0.sum(), g_.n.sum(), g_.size()
    dff = nf_ - kf_
    s_all = float(within.sum() / C.W0.sum() * C.n.sum() / (C.n.sum() - len(C)))
    s_f = (wf / Wf_ * nf_ / dff.where(dff > 0, 1)).where(dff >= SIG_DFMIN, s_all)
    sig2 = C.grp.map(s_f).to_numpy(np.float64)
    info["sig2_all"] = s_all
    info["sig2_pooled_groups"] = sorted(s_f.index[dff < SIG_DFMIN])
    vmod = sig2 / C["neff"].to_numpy()
    if deff is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            mr = C[[f"Y{k}" for k in range(1, N_REP + 1)]].to_numpy() / C[[f"W{k}" for k in range(1, N_REP + 1)]].to_numpy()
        vsdr = 4.0 / N_REP * ((mr - mean0[:, None]) ** 2).sum(1)
        big = (C["n"].to_numpy() >= DEFF_NMIN) & (vmod > 0) & np.all(np.isfinite(mr), axis=1)
        ratio = vsdr[big] / vmod[big]
        deff = float(np.median(ratio))
        info.update(deff_cells=int(big.sum()), deff_q=(float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))),
                    deff_mean=float(np.mean(ratio)), deff_sum=float(vsdr[big].sum() / vmod[big].sum()))
    v = deff * vmod
    Wr = C[f"W{r}"].to_numpy()
    # replicate weights can be zero; where a replicate's cell weight is <= 0 the replicate mean is undefined and the
    # main-weight mean is used (tiny cells only, which are shrunk almost to 0 anyway)
    m = np.where(Wr > 0, C[f"Y{r}"].to_numpy() / np.where(Wr > 0, Wr, 1.0), mean0)
    info["rep_cells_undefined"] = int((Wr <= 0).sum())
    fl = sorted(C.grp.unique()); gl = sorted(C.geo.unique())
    fi = pd.Categorical(C.grp, categories=fl).codes.astype(int)
    gi = pd.Categorical(C.geo, categories=gl).codes.astype(int)
    nf, ng = len(fl), len(gl)
    rank = _design_rank(fi, gi, nf, ng)
    cache = {}

    def u_of(t):
        key = float(t)
        if key not in cache:
            fit, a, b = _additive_fit(m, 1.0 / (v + t), fi, gi, nf, ng)
            cache[key] = (m - fit, a, b)
        return cache[key][0]

    tau2 = _pm_tau2(u_of, v, len(m) - rank) if tau2_fix is None else float(tau2_fix)
    u = u_of(tau2)
    _, a, b = cache[float(tau2)]
    # per-group tau2_g (Paule-Mandel within group, a_g and b_geo held at the global fit; df = cells - 1)
    tau2_f = {}
    for j, f in enumerate(fl):
        mk = fi == j
        if tau2_f_fix is not None:
            tau2_f[f] = tau2_f_fix.get(f, np.nan)
        else:
            tau2_f[f] = _pm_tau2(lambda t, mk=mk: u[mk], v[mk], int(mk.sum()) - 1) if mk.sum() >= 5 else np.nan
    C["m"], C["v"], C["u"] = m, v, u
    C["a_g"] = a[fi]
    C["b_geo"] = b[gi]
    C["lam"] = tau2 / (tau2 + v)
    t2 = np.array([tau2_f[f] if np.isfinite(tau2_f[f]) else tau2 for f in C.grp])
    C["lam_f"] = t2 / (t2 + v)
    C["U"] = C["lam"] * u
    C["U_ftau"] = C["lam_f"] * u
    C["U_raw"] = u
    info.update(tau2=tau2, tau2_f=tau2_f, deff=deff, n_cells=len(C), df=len(m) - rank, rank=rank,
                n_groups=nf, n_geo=ng)
    keep = ["grp", "geo", "n", "neff", "m", "v", "u", "a_g", "b_geo", "lam", "lam_f", "U", "U_ftau", "U_raw"]
    return C[keep].reset_index(drop=True), info


def eb_metro(Cm: pd.DataFrame, unit_state: dict, Us: pd.DataFrame, deff: float, r: int = 0) -> tuple[pd.DataFrame, dict]:
    """Hierarchical metro level: units = metros and state non-metro remainders. The unit-level group component u'
    (additive fit on the unit cells, its own Paule-Mandel tau2 for the weights) is compared with the state-level U of
    the unit's principal state; the deviation d is shrunk with its own Paule-Mandel variance:
    U_metro = prior + tau2_d / (tau2_d + v) * d."""
    E, info = eb_level(Cm, r=r, deff=deff)
    su = Us.set_index(["grp", "geo"])["U"]
    prior = su.reindex(pd.MultiIndex.from_arrays([E.grp.to_numpy(), E.geo.map(unit_state).to_numpy()])).fillna(0.0)
    prior = prior.to_numpy()
    d = E["u"].to_numpy() - prior
    v = E["v"].to_numpy()
    # df = cells - rank of the unit-level additive fit (u carries that many degrees of freedom; verification fix)
    t2 = _pm_tau2(lambda t: d, v, info["df"])
    lam = t2 / (t2 + v) if t2 > 0 else np.zeros_like(v)
    E["prior"] = prior
    E["lam_d"] = lam
    E["U"] = prior + lam * d
    info.update(tau2_dev=t2, dev_q=float(np.sum(d ** 2 / v)))
    return E, info


def occ_index(A: pd.DataFrame, fmap: dict) -> pd.DataFrame:
    """Leave-field-out occupation-mix index per (field, state of residence): sum_o pi_fo * e_{o,s,-f}, where pi_fo is
    the national occupation distribution (weights) of field f's graduates in the sample and e_{o,s,-f} the mean log
    earnings in occupation o and state s of the sample's graduates of every OTHER field (all FOD1P codes outside f),
    shrunk toward the national leave-field-out occupation mean with a prior of OCC_PRIOR persons."""
    occ = A.OCCP.to_numpy(); st = A.STATE.to_numpy()
    ol, oi = np.unique(occ, return_inverse=True)
    sl, si = np.unique(st, return_inverse=True)
    O, S = len(ol), len(sl)
    w = A.PWGTP.to_numpy(np.float64); y = A.y.to_numpy(np.float64)
    fod = A.FOD1P.to_numpy()

    def sums(mask):
        W_ = np.zeros((O, S)); Y_ = np.zeros((O, S)); N_ = np.zeros((O, S))
        np.add.at(W_, (oi[mask], si[mask]), w[mask]); np.add.at(Y_, (oi[mask], si[mask]), w[mask] * y[mask])
        np.add.at(N_, (oi[mask], si[mask]), 1.0)
        return W_, Y_, N_

    Wall, Yall, Nall = sums(np.ones(len(A), bool))
    rows = []
    for f in sorted(fmap):
        mf = np.isin(fod, np.array([int(c) for c in fmap[f]]))
        if not mf.any():
            continue
        Wf, Yf, Nf = sums(mf)
        pi = Wf.sum(1) / Wf.sum()
        Wo, Yo = Wall.sum(1) - Wf.sum(1), Yall.sum(1) - Yf.sum(1)
        okn = Wo > 0
        e_nat = np.where(okn, Yo / np.where(okn, Wo, 1.0), 0.0)
        Wos, Yos, Nos = Wall - Wf, Yall - Yf, Nall - Nf
        e_os = np.where(Wos > 0, Yos / np.where(Wos > 0, Wos, 1.0), 0.0)
        Nos = np.where(Wos > 0, Nos, 0.0)
        blend = (Nos * e_os + OCC_PRIOR * e_nat[:, None]) / (Nos + OCC_PRIOR)
        pv = np.where(okn, pi, 0.0)
        L = (pv[:, None] * blend).sum(0) / pv.sum()
        rows.append(pd.DataFrame(dict(field=f, geo=sl, L_occ=L, pi_cov=pv.sum())))
    return pd.concat(rows, ignore_index=True)


# =============================================================================================
# program-level U variants
# =============================================================================================
def _lookup(tab: pd.DataFrame, grp: np.ndarray, geo: np.ndarray, col: str) -> np.ndarray:
    s = tab.set_index(["grp", "geo"])[col]
    return s.reindex(pd.MultiIndex.from_arrays([grp, geo])).to_numpy(np.float64)


class Geo:
    """Everything needed to build the program-level U variants from the cell sums for one ACS weight set r."""

    def __init__(self, cells, f2g, Cs, Cp, Cm, unit_state, prog_unit, deffs):
        self.grp = cells.field.map(f2g).to_numpy(object)
        self.mapped = cells.field.isin(f2g).to_numpy()
        self.fips = cells.fips.to_numpy(np.int64)
        self.unit = prog_unit                                   # metro unit of the program (-1: not in a metro)
        self.Cs, self.Cp, self.Cm = Cs, Cp, Cm
        self.unit_state, self.deffs = unit_state, deffs
        self.fix = None                                         # main-weight (tau2, tau2_g) of the state level

    def _prog(self, tab: pd.DataFrame, src: str) -> np.ndarray:
        v = _lookup(tab, self.grp, self.fips, src)
        return np.where(self.mapped, np.nan_to_num(v, nan=0.0), np.nan)

    def build_fixed(self, r: int) -> tuple[np.ndarray, dict]:
        """State-level U with weight set r and the shrinkage variances held at their main-weight values (revision:
        replicate-variance check). Returns the program-level U and the (fixed) tau2_g."""
        Es, info = eb_level(self.Cs, r, self.deffs.get("state"), tau2_fix=self.fix["tau2"], tau2_f_fix=self.fix["tau2_f"])
        return self._prog(Es, "U"), info["tau2_f"]

    def build(self, r: int) -> tuple[dict, dict]:
        Es, info_s = eb_level(self.Cs, r, self.deffs.get("state"))
        Ep, info_p = eb_level(self.Cp, r, self.deffs.get("pow"))
        Em, info_m = eb_metro(self.Cm, self.unit_state, Es, self.deffs.get("metro"), r)
        Ed, info_d = eb_level(self.Cs, r, self.deffs.get("state_sum"))      # revision: deff = ratio of sums
        out = {}
        for col, tab, src in (("U", Es, "U"), ("U_raw", Es, "U_raw"), ("U_ftau", Es, "U_ftau"), ("U_pow", Ep, "U"),
                              ("U_dsum", Ed, "U")):
            out[col] = self._prog(tab, src)
        # metro: in a metro unit present in the ACS -> the unit's U (its prior where the unit has no person of the
        # field); otherwise the program's own state-level U
        um = _lookup(Em, self.grp, self.unit, "U")
        pri_state = np.array([self.unit_state.get(u, -1) for u in self.unit])
        prior = _lookup(Es, self.grp, pri_state, "U")
        prior = np.nan_to_num(prior, nan=0.0)
        in_unit = (self.unit > 0) & (pri_state > 0)
        um = np.where(np.isfinite(um), um, prior)
        out["U_met"] = np.where(self.mapped, np.where(in_unit, um, out["U"]), np.nan)
        info = dict(state=info_s, pow=info_p, metro=info_m, dsum=info_d, Es=Es, Ep=Ep, Em=Em,
                    n_in_metro_unit=int((in_unit & self.mapped).sum()))
        return out, info


# =============================================================================================
# within-institution design (scripts/58 fe_build conventions) and its replicate solver
# =============================================================================================
def design(cells, slopes, need, yvar, per_field, extra=(), extra_pf=(), offset=None):
    """The scripts/58 fe_build design (same sample rule s58.fe_sample, same within-field z-scores, same column
    order), NOT demeaned, plus common extra columns (appended after the field-specific slopes) and field-specific
    slopes on raw extra_pf columns (appended last). offset: a column subtracted from log y (coefficient fixed at 1)."""
    d = s58.fe_sample(cells, list(slopes) + list(need), yvar)
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
    ucols = {}
    for c in extra:
        ucols[len(names)] = (c, None)
        cols.append(d[c].to_numpy(float)[:, None]); names.append(f"c_{c}")
    for c in extra_pf:
        for k, f in enumerate(fields):
            ucols[len(names) + k] = (c, k)
        cols.append(D * d[c].to_numpy(float)[:, None]); names += [f"xp_{c}_{f}" for f in fields]
    X = np.hstack(cols)
    logy = np.log(d[yvar].to_numpy(float))
    y = logy - d[offset].to_numpy(float) if offset is not None else logy
    ic = pd.factorize(d.inst_key)[0]
    return dict(d=d, fields=fields, code=np.asarray(code), X=X, y=y, logy=logy, names=names, ic=ic,
                G=int(ic.max()) + 1, z=z, ucols=ucols, offset=offset, rows=d["_row"].to_numpy(),
                sdP=g.prestige_score.transform("std").to_numpy())


def to_fe(Mr: dict) -> dict:
    """Demeaned model in the s58.fe_build format (same arithmetic as fe_build's dm), for s58.fe_fit."""
    ic = Mr["ic"]
    cnt = np.bincount(ic).astype(float)

    def dm(M):
        s = np.zeros((cnt.size,) + M.shape[1:]); np.add.at(s, ic, M)
        return M - (s / (cnt[:, None] if M.ndim == 2 else cnt))[ic]

    return dict(d=Mr["d"], fields=Mr["fields"], X=dm(Mr["X"]), y=dm(Mr["y"]), names=Mr["names"], groups=ic, cnt=cnt,
                z=Mr["z"], code=Mr["code"])


def with_u(M: dict, uv: dict) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) of raw design M with every U-variant column / offset rebuilt from program-level arrays uv (indexed by
    cells row)."""
    X, y = M["X"], M["y"]
    if M["ucols"]:
        X = X.copy()
        for j, (c, k) in M["ucols"].items():
            v = uv[c][M["rows"]]
            X[:, j] = v if k is None else np.where(M["code"] == k, v, 0.0)
    if M["offset"] is not None:
        y = M["logy"] - uv[M["offset"]][M["rows"]]
    return X, y


def wsolve(X, y, ic, G, w, tidx, code=None):
    """WLS with institution FE absorbed by WEIGHTED within-institution demeaning (rows with w = 0 drop out; a field
    whose rows all have w = 0 loses its columns). Returns the coefficients at tidx (NaN if not identified). The first
    K - 1 columns are the dummies of fields[1:]; if the reference field has no weight, the dummy of the first field
    with weight is dropped as well (collinear with the institution FE otherwise)."""
    n, p = X.shape
    P = sps.csr_matrix((w, (ic, np.arange(n))), shape=(G, n))
    sw = np.asarray(P.sum(1)).ravel()
    inv = np.where(sw > 0, 1.0 / np.where(sw > 0, sw, 1.0), 0.0)
    Xd = X - (P @ X * inv[:, None])[ic]
    yd = y - (P @ y * inv)[ic]
    Xw = Xd * w[:, None]
    A_ = Xd.T @ Xw
    b_ = Xw.T @ yd
    keep = np.diag(A_) > 1e-10
    if code is not None:
        fw = np.bincount(code, weights=w, minlength=int(code.max()) + 1)
        if fw[0] <= 0:
            k1 = int(np.argmax(fw > 0))
            if k1 >= 1:
                keep[k1 - 1] = False
    Ak, bk = A_[np.ix_(keep, keep)], b_[keep]
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
    full = np.full(p, np.nan)
    full[keep] = sol
    return full[tidx]


# ---- replicate engine (fork workers; one SeedSequence child per replicate) ----
_SS: dict = {}          # key -> dict(ic, G, code, K, specs=[(X, y, tidx, pf_fields or None)])
KIND_ID = {"inst": 1, "fxi": 2, "jack": 3}


def _weights(S, kind, b, children):
    ic, G, code, K = S["ic"], S["G"], S["code"], S["K"]
    if kind == "jack":
        return (code != b).astype(np.float64)
    rng = np.random.default_rng(children[b])
    if kind == "inst":
        c = np.bincount(rng.integers(0, G, G), minlength=G).astype(np.float64)
        return c[ic]
    w = np.empty(len(ic))
    for k in range(K):                               # independent institution draw per field
        c = np.bincount(rng.integers(0, G, G), minlength=G).astype(np.float64)
        mk = code == k
        w[mk] = c[ic[mk]]
    return w


def _eval(S, w, specs=None):
    out = []
    for X, y, tidx, pf in (S["specs"] if specs is None else specs):
        v = wsolve(X, y, S["ic"], S["G"], w, tidx, S["code"])
        if pf is not None:                           # per-field slopes: need >= PF_MIN_DISTINCT institutions drawn
            for j, k in enumerate(pf):
                mk = (S["code"] == k) & (w > 0)
                if len(np.unique(S["ic"][mk])) < PF_MIN_DISTINCT:
                    v[j] = np.nan
        out.append(v)
    return np.concatenate(out)


def _rep_chunk(args):
    key, kind, idx = args
    S = _SS[key]
    n_b = S["K"] if kind == "jack" else B_REP
    children = None if kind == "jack" else np.random.SeedSequence([SEED, zlib.crc32(key.encode()), KIND_ID[kind]]).spawn(n_b)
    with threadpool_limits(1):
        return np.stack([_eval(S, _weights(S, kind, b, children)) for b in idx])


def run_reps(key: str, workers: int) -> dict:
    """Replicates of every target of specset `key`: inst (shared institution draw), fxi (independent draw per
    field), jack (leave one field out); plus the full-sample point values."""
    S = _SS[key]
    out = {}
    gc.collect()
    with threadpool_limits(1):
        out["point"] = _eval(S, np.ones(len(S["ic"])))
    tasks = []
    for kind in ("inst", "fxi", "jack"):
        n_b = S["K"] if kind == "jack" else B_REP
        step = 25 if kind != "jack" else 4
        tasks += [(key, kind, list(range(i, min(i + step, n_b)))) for i in range(0, n_b, step)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
            res = list(ex.map(_rep_chunk, tasks))
    else:
        res = [_rep_chunk(t) for t in tasks]
    for kind in ("inst", "fxi", "jack"):
        out[kind] = np.vstack([r for t, r in zip(tasks, res) if t[1] == kind])
    return out


def register(key: str, models: list[tuple[str, dict, list[str]]]) -> list[str]:
    """Register a specset: models = [(label, raw design, target names)], all on identical rows. Returns target
    labels 'label:name'."""
    M0 = models[0][1]
    for _, M, _ in models[1:]:
        assert M["d"].inst_key.equals(M0["d"].inst_key) and M["d"].field.equals(M0["d"].field), "specset rows differ"
    specs, labels = [], []
    for lab, M, tg in models:
        tidx = np.array([M["names"].index(t) for t in tg])
        pf = [M["fields"].index(t[2:]) for t in tg] if all(t.startswith("b_") for t in tg) else None
        specs.append((M["X"], M["y"], tidx, pf))
        labels += [f"{lab}:{t}" for t in tg]
    _SS[key] = dict(ic=M0["ic"], G=M0["G"], code=M0["code"], K=len(M0["fields"]), specs=specs,
                    models=[(lab, M, np.array([M["names"].index(t) for t in tg])) for lab, M, tg in models])
    return labels


# ---- ACS replicate refits (fork workers; no randomness) ----
_ACS: dict = {}


def _acs_task(r: int):
    """Everything that uses U, recomputed with ACS replicate weight set r (r = 0: main weights)."""
    A = _ACS
    with threadpool_limits(1):
        uv, info = A["geo"].build(r)
        uv["L_occ"] = A["L_occ"]
        out = {}
        for key in A["specsets"]:
            S = _SS[key]
            specs = []
            for lab, M, tidx in S["models"]:
                X, y = with_u(M, uv)
                pf = [M["fields"].index(M["names"][j][2:]) for j in tidx] if all(
                    M["names"][j].startswith("b_") for j in tidx) else None
                specs.append((X, y, tidx, pf))
            out[key] = _eval(S, np.ones(len(S["ic"])), specs)
        # T2 per-field models (institution-cluster standard): CS beta_f
        for key, M in A["pf_models"].items():
            X, y = with_u(M, uv)
            j = M["names"].index(f"b_{CS}")
            out[key] = float(wsolve(X, y, M["ic"], M["G"], np.ones(len(M["ic"])), np.array([j]), M["code"])[0])
        # T3 per-field statistics
        tau_g = info["state"]["tau2_f"]
        out["T3"] = t3_point(A["T3"], uv["U"], tau_g)
        out["tau2"] = dict(state=info["state"]["tau2"], pow=info["pow"]["tau2"], dsum=info["dsum"]["tau2"],
                           metro_dev=info["metro"]["tau2_dev"])
        # revision check: the same statistics with the state-level shrinkage (tau2, tau2_g) held at the main-weight
        # values (U linear in the replicate cell means)
        Ufix, tau_g_fix = A["geo"].build_fixed(r)
        uvf = dict(uv)
        uvf["U"] = Ufix
        fix = {}
        for lab, M, tidx in _SS["main"]["models"]:
            if lab == "after":
                X, y = with_u(M, uvf)
                fix["main_after"] = wsolve(X, y, M["ic"], M["G"], np.ones(len(M["ic"])), tidx, M["code"])
        for key, M in A["pf_models"].items():
            if key.endswith("_after"):
                X, y = with_u(M, uvf)
                j = M["names"].index(f"b_{CS}")
                fix[key] = float(wsolve(X, y, M["ic"], M["G"], np.ones(len(M["ic"])), np.array([j]), M["code"])[0])
        fix["T3"] = t3_point(A["T3"], Ufix, tau_g_fix)
        out["fix"] = fix
    return r, out


def run_acs(workers: int) -> dict:
    rs = list(range(0, N_REP + 1))
    gc.collect()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
            res = list(ex.map(_acs_task, rs))
    else:
        res = [_acs_task(r) for r in rs]
    return dict(res)


def acs_var(th0: float, reps) -> dict:
    """ACS successive-difference replicate variance of a scalar statistic (80 replicate values, NaN dropped).
    v: primary since the revision, centred at the replicate mean, 4/80 sum_r (theta_r - mean_r theta_r)^2;
    v_th0: the first version's form, centred at the full-sample value, 4/80 sum_r (theta_r - theta0)^2
    (= v + 4/80 x n_rep x bias^2); bias: mean_r theta_r - theta0."""
    x = np.asarray(reps, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(v=0.0, v_th0=0.0, bias=0.0, n=0)
    mu = float(x.mean())
    return dict(v=4.0 / N_REP * float(np.sum((x - mu) ** 2)), v_th0=4.0 / N_REP * float(np.sum((x - th0) ** 2)),
                bias=mu - float(th0), n=int(len(x)))


# =============================================================================================
# T3: per-field statistics and cross-field summaries
# =============================================================================================
def t3_setup(cells: pd.DataFrame, fields: list, f2g: dict, acs_n_group: dict) -> dict:
    universe = sorted(cells.inst_key.unique())
    idx = {k: i for i, k in enumerate(universe)}
    FD = {}
    for f in fields:
        d = cells[cells.field == f]
        FD[f] = dict(P=d.prestige_score.to_numpy(float), Y=d.earnings.to_numpy(float),
                     rows=d["_row"].to_numpy(), inst=d.inst_key.map(idx).to_numpy(int), grp=f2g[f])
    # ACS persons of each field's category (fixed across replicates; revision diagnostics)
    nacs = np.array([float(acs_n_group[f2g[f]]) for f in fields])
    return dict(FD=FD, fields=list(fields), N=len(universe), nacs=nacs)


def t3_field(P, Y, U, rows=None) -> tuple:
    if rows is not None:
        P, Y, U = P[rows], Y[rows], U[rows]
    if len(P) < 4:
        return (np.nan,) * 4
    rho = s55.spear(P, Y)
    rfu = s55.spear(P, U)
    sdu = float(np.std(U, ddof=1))
    prt = s55.partial_rank(P, Y, [U], [])[0] if np.std(U) > 0 else s55.partial_rank(P, Y, [], [])[0]
    return rho, rfu, sdu, prt


T3_STATS = ["S_rho_tau", "S_rho_sdU", "S_rho_rhoFU", "mean_rho", "mean_partial", "diff_partial_minus_rho",
            "S_rho_partial", "mean_rhoFU",
            # revision (exploratory, post hoc): what the SD of the shrunk U over a field's programs measures
            "S_sdU_nacs", "S_sdU_tau", "S_rho_nacs", "P_rho_sdU_nacs"]
T3_CORR = {nm for nm in T3_STATS if nm.startswith(("S_", "P_"))}    # correlations: Fisher-z interval


def _sp(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    return float(spearmanr(a[ok], b[ok])[0])


def _psp(a, b, c):
    """Rank-partial Spearman correlation of a and b given c (scripts/55 partial_rank), finite rows only."""
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    if ok.sum() < 5:
        return np.nan
    return float(s55.partial_rank(a[ok], b[ok], [c[ok]], [])[0])


def t3_cross(rho, tau, rfu, sdu, prt, nacs) -> np.ndarray:
    okp = np.isfinite(rho) & np.isfinite(prt)
    return np.array([_sp(rho, tau), _sp(rho, sdu), _sp(rho, rfu), float(np.nanmean(rho)),
                     float(np.mean(prt[okp])), float(np.mean(prt[okp] - rho[okp])), _sp(rho, prt),
                     float(np.nanmean(rfu)),
                     _sp(sdu, nacs), _sp(sdu, tau), _sp(rho, nacs), _psp(rho, sdu, nacs)])


def t3_point(T: dict, U: np.ndarray, tau2_g: dict, counts=None, drop=None) -> np.ndarray:
    """Cross-field T3 statistics; counts: (K, N) institution counts per field (None: all rows once); drop: field
    index left out (jackknife). Returns the statistics followed by the per-field values (rho, rfu, sdu, prt, tau)."""
    K = len(T["fields"])
    vals = np.full((K, 5), np.nan)
    for k, f in enumerate(T["fields"]):
        if drop is not None and k == drop:
            continue
        fd = T["FD"][f]
        rows = None if counts is None else np.repeat(np.arange(len(fd["P"])), counts[k][fd["inst"]])
        vals[k, :4] = t3_field(fd["P"], fd["Y"], U[fd["rows"]], rows)
        t2 = tau2_g.get(fd["grp"], np.nan)
        vals[k, 4] = np.sqrt(t2) if np.isfinite(t2) else np.nan
    rho, rfu, sdu, prt, tau = vals.T
    nacs = np.where(np.isfinite(rho), T["nacs"], np.nan)
    return np.concatenate([t3_cross(rho, tau, rfu, sdu, prt, nacs), vals.ravel()])


def _t3_chunk(args):
    kind, idx = args
    T = _ACS["T3"]
    K, N = len(T["fields"]), T["N"]
    U, tau2_g = _ACS["U0"], _ACS["tau2_g0"]
    ns = len(T3_STATS)
    out = []
    with threadpool_limits(1):
        if kind == "jack":
            for b in idx:
                out.append(t3_point(T, U, tau2_g, drop=b)[:ns])
        else:
            ch = np.random.SeedSequence([SEED, zlib.crc32(b"T3"), KIND_ID[kind]]).spawn(B_REP)
            for b in idx:
                rng = np.random.default_rng(ch[b])
                if kind == "inst":
                    c = np.bincount(rng.integers(0, N, N), minlength=N)
                    counts = np.tile(c, (K, 1))
                else:
                    counts = np.stack([np.bincount(rng.integers(0, N, N), minlength=N) for _ in range(K)])
                out.append(t3_point(T, U, tau2_g, counts=counts)[:ns])
    return kind, np.array(out)


def run_t3(workers: int) -> dict:
    K = len(_ACS["T3"]["fields"])
    tasks = [("inst", list(range(i, min(i + 50, B_REP)))) for i in range(0, B_REP, 50)] \
        + [("fxi", list(range(i, min(i + 50, B_REP)))) for i in range(0, B_REP, 50)] \
        + [("jack", list(range(i, min(i + 10, K)))) for i in range(0, K, 10)]
    gc.collect()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
            res = list(ex.map(_t3_chunk, tasks))
    else:
        res = [_t3_chunk(t) for t in tasks]
    return {k: np.vstack([r for kk, r in res if kk == k]) for k in ("inst", "fxi", "jack")}


# =============================================================================================
# inference helpers
# =============================================================================================
def jack_var(th: np.ndarray) -> float:
    th = np.asarray(th, float)
    th = th[np.isfinite(th)]
    K = len(th)
    return float((K - 1) / K * ((th - th.mean()) ** 2).sum())


def fisher_ci(est: float, se: float) -> tuple[float, float]:
    """95% interval of a correlation on the Fisher-z scale (SE by the delta method), back-transformed."""
    if not (np.isfinite(est) and np.isfinite(se)) or abs(est) >= 1:
        return np.nan, np.nan
    z0, sz = np.arctanh(est), se / (1.0 - est ** 2)
    return float(np.tanh(z0 - Z975 * sz)), float(np.tanh(z0 + Z975 * sz))


def tw(est: float, jack: np.ndarray, inst: np.ndarray, fxi: np.ndarray, acs: dict | None = None,
       corr: bool = False) -> dict:
    """scripts/59 two-way variance (+ the ACS replicate variance of U, independent sample) -> SE, normal 95% CI.
    acs: acs_var() of the statistic (primary: replicate-mean-centred; the theta0-centred CI is kept beside it).
    corr: the statistic is a correlation -> also the Fisher-z interval (fz_lo, fz_hi)."""
    inst, fxi = inst[np.isfinite(inst)], fxi[np.isfinite(fxi)]
    r = twoway(est, jack_var(jack), inst, fxi)
    acs = acs or dict(v=0.0, v_th0=0.0, bias=0.0, n=0)
    se = float(np.sqrt(r["tse"] ** 2 + acs["v"]))
    se0 = float(np.sqrt(r["tse"] ** 2 + acs["v_th0"]))
    out = dict(est=float(est), se_tw=r["tse"], vf=r["tvf"], vs=r["tvs"], vi=r["tvi"], fallback=r["tfallback"],
               mcse=r["tmcse"], vacs=float(acs["v"]), vacs_th0=float(acs["v_th0"]), acs_bias=float(acs["bias"]),
               se=se, lo=est - Z975 * se, hi=est + Z975 * se, se_th0=se0, lo_th0=est - Z975 * se0,
               hi_th0=est + Z975 * se0, z=est / se if se > 0 else np.nan, mde=MDE_K * se,
               inst_lo=float(np.percentile(inst, 2.5)), inst_hi=float(np.percentile(inst, 97.5)),
               n_fail=int(len(jack) + 2 * B_REP - np.isfinite(jack).sum() - len(inst) - len(fxi)), corr=corr)
    out["fz_lo"], out["fz_hi"] = fisher_ci(est, se) if corr else (np.nan, np.nan)
    out["fz_lo_th0"], out["fz_hi_th0"] = fisher_ci(est, se0) if corr else (np.nan, np.nan)
    # reported interval: Fisher-z for correlations, normal otherwise
    out["rlo"], out["rhi"] = (out["fz_lo"], out["fz_hi"]) if corr else (out["lo"], out["hi"])
    return out


def cgm_se(Ms: list, idx: list, signs: list) -> dict:
    """Analytic two-way (field, institution) sandwich of a linear combination sum_s sign_s beta_{model s, idx s} of
    OLS coefficients of demeaned models on identical rows: CR1(field) + CR1(institution) - HC1 (programs are the
    field x institution intersections)."""
    IF = None
    p_max = 0
    for M, j, sg in zip(Ms, idx, signs):
        X, y = M["X"], M["y"]
        XtXi = np.linalg.pinv(X.T @ X)
        u = y - X @ (XtXi @ (X.T @ y))
        row = (X @ XtXi[:, j]) * u * sg
        IF = row if IF is None else IF + row
        p_max = max(p_max, np.linalg.matrix_rank(X))
    n = len(IF)
    out = {}
    for tag, g in (("field", Ms[0]["code"]), ("inst", Ms[0]["groups"])):
        S = np.bincount(g, weights=IF)
        G = len(S)
        out[tag] = float(G / (G - 1) * (n - 1) / (n - p_max) * (S ** 2).sum())
    out["hc1"] = float(n / (n - p_max) * (IF ** 2).sum())
    v = out["field"] + out["inst"] - out["hc1"]
    out["se"] = float(np.sqrt(v)) if v > 0 else float(np.sqrt(max(out["field"], out["inst"])))
    out["fallback"] = not v > 0
    out["se_inst"] = float(np.sqrt(out["inst"]))
    return out


def cr1(Mfe: dict, name: str) -> tuple[float, float]:
    fit = sm.OLS(Mfe["y"], Mfe["X"]).fit(cov_type="cluster", cov_kwds={"groups": Mfe["groups"]})
    j = Mfe["names"].index(name)
    return float(np.asarray(fit.params)[j]), float(np.sqrt(np.asarray(fit.cov_params())[j, j]))


def resid_z(Mfe: dict, sdP: np.ndarray) -> dict:
    """SD of z(F) after within-institution demeaning, and of its residual on every other regressor (the identifying
    variation of beta), in within-field SD units and in field-rank positions (x the field's SD of -rank)."""
    j = Mfe["names"].index("beta")
    xj = Mfe["X"][:, j]
    oth = np.delete(Mfe["X"], j, axis=1)
    rj = xj - oth @ np.linalg.lstsq(oth, xj, rcond=None)[0]
    n = len(xj)
    cs = Mfe["d"].field.to_numpy() == CS
    return dict(sd_within=float(np.sqrt(xj @ xj / n)), sd_resid=float(np.sqrt(rj @ rj / n)),
                share=float(rj @ rj / (xj @ xj)), sd_resid_rank=float(np.sqrt(np.mean((rj * sdP) ** 2))),
                sd_within_rank=float(np.sqrt(np.mean((xj * sdP) ** 2))),
                sd_resid_cs=float(np.sqrt(np.mean(rj[cs] ** 2))) if cs.any() else np.nan,
                sd_resid_rank_cs=float(np.sqrt(np.mean((rj[cs] * sdP[cs]) ** 2))) if cs.any() else np.nan,
                sd_raw_rank=float(np.sqrt(np.mean(sdP ** 2))))


# =============================================================================================
# main computation
# =============================================================================================
def build_cells() -> pd.DataFrame:
    ar = load_ar_wapman(fields=FIELDS66)
    er = load_er_scorecard("undergrad", fields=FIELDS66)
    inst = s55.load_institutions()
    cells = s55.build_cells(ar, er, inst)
    n0 = len(cells)
    for tag, ec, cc in s58.HORIZONS:                      # other horizons of the same release (as scripts/58)
        e = load_er_scorecard("undergrad", earn_col=ec, count_col=cc, fields=FIELDS66)
        cells = cells.merge(e[["inst_key", "field", "earnings", "cohort_n"]].rename(
            columns={"earnings": tag, "cohort_n": f"{tag}_n"}), on=["field", "inst_key"], how="left")
    assert len(cells) == n0 and cells.INSTNM.notna().all()
    cells["fips"] = cells.STABBR.map(state_fips())
    assert cells.fips.notna().all()
    cells["fips"] = cells.fips.astype(np.int64)
    cells["_row"] = np.arange(len(cells))
    return cells


def compute(workers: int) -> dict:
    R: dict = {}
    fmap, kind = field_map()
    f2g, g2c = acs_groups(fmap)
    R["fmap"], R["kind"], R["f2g"] = fmap, kind, f2g

    # ---------------- ACS ----------------
    stage("ACS extract")
    A, counts = acs_extract(workers)
    R["acs_counts"] = counts
    A["y"] = acs_y(A)
    stage(f"ACS sample: {len(A)} persons")
    pm, hdm, minfo = metro_maps()
    R["metro_info"] = {k: v for k, v in minfo.items() if k != "mtitle"}
    A["puma"] = A.STATE.astype(np.int64) * 100000 + A.PUMA.astype(np.int64)
    A = A.merge(pm[["puma", "cbsa"]], on="puma", how="left", validate="many_to_one")
    assert A.cbsa.notna().all(), "PUMA not in the 2020 tract-to-PUMA file"
    assert A.cbsa.max() < NONMETRO_BASE and A.STATE.max() < 100
    A["unit"] = np.where(A.cbsa > 0, A.cbsa, NONMETRO_BASE + A.STATE).astype(np.int64)
    A["POW_ST"] = np.where((A.POWSP >= 1) & (A.POWSP <= 56), A.POWSP, -1).astype(np.int64)
    A["STATE"] = A.STATE.astype(np.int64)
    mapped_codes = {int(c) for cs in g2c.values() for c in cs}
    Am = A[A.FOD1P.isin(mapped_codes)]
    R["acs_n_mapped"] = int(len(Am))
    R["acs_n_all"] = int(len(A))
    R["acs_n_group"] = {g: int(Am.FOD1P.isin([int(c) for c in cs]).sum()) for g, cs in g2c.items()}
    Cs = cell_sums(Am, g2c, "STATE")
    Cp = cell_sums(Am, g2c, "POW_ST")
    Cm = cell_sums(Am, g2c, "unit")
    # principal state of each metro unit: most sample persons
    us = Am.groupby(["unit", "STATE"]).size().rename("k").reset_index().sort_values(
        ["unit", "k", "STATE"], ascending=[True, False, True]).drop_duplicates("unit")
    unit_state = dict(zip(us.unit, us.STATE))
    R["n_units"] = len(unit_state)
    R["n_metro_units"] = int(sum(u < NONMETRO_BASE for u in unit_state))
    assert min(unit_state) > 0
    L_occ_tab = occ_index(A, fmap)
    del A, Am
    gc.collect()

    # ---------------- Scorecard cells ----------------
    stage("Scorecard cells")
    cells = build_cells()
    hd_ = hdm.copy()
    cells = cells.merge(hd_.rename(columns={"cbsa": "inst_cbsa", "via_hd": "cbsa_via_hd"}), on="UNITID", how="left",
                        validate="many_to_one")
    assert cells._row.is_monotonic_increasing and len(cells) == cells._row.max() + 1
    R["inst_no_hd"] = int(cells.drop_duplicates("UNITID").inst_cbsa.isna().sum())
    cells["inst_cbsa"] = cells.inst_cbsa.fillna(-1).astype(np.int64)
    # program metro unit (only metros present as ACS units count)
    prog_unit = np.where(cells.inst_cbsa > 0, cells.inst_cbsa, -1).astype(np.int64)
    R["inst_in_metro"] = int(cells.loc[cells.inst_cbsa > 0].UNITID.nunique())
    R["inst_metro_absent"] = int(cells.loc[(cells.inst_cbsa > 0) & ~cells.inst_cbsa.isin(list(unit_state))].UNITID.nunique())
    R["inst_cbsa_via_hd"] = int(cells.loc[cells.cbsa_via_hd.fillna(False).astype(bool)].UNITID.nunique())
    R["n_inst_all"] = int(cells.UNITID.nunique())

    # main-weight EB (deff estimated here, reused by every replicate)
    Es0, info_s = eb_level(Cs, 0, None)
    Ep0, info_p = eb_level(Cp, 0, None)
    Em_tmp, info_m0 = eb_level(Cm, 0, None)
    # state_sum (revision sensitivity): deff = ratio of sums over the same calibration cells
    deffs = dict(state=info_s["deff"], pow=info_p["deff"], metro=info_m0["deff"], state_sum=info_s["deff_sum"])
    geo = Geo(cells, f2g, Cs, Cp, Cm, unit_state, prog_unit, deffs)
    geo.fix = dict(tau2=info_s["tau2"], tau2_f=dict(info_s["tau2_f"]))
    uv0, info0 = geo.build(0)
    for c in UVAR_ACS:
        cells[c] = uv0[c]
    Ufix0, _ = geo.build_fixed(0)
    assert np.array_equal(Ufix0, uv0["U"], equal_nan=True), "fixed-shrinkage U differs at the main weight"
    R["info_dsum"] = info0["dsum"]
    lo = L_occ_tab.set_index(["field", "geo"]).L_occ
    cells["L_occ"] = lo.reindex(pd.MultiIndex.from_arrays([cells.field, cells.fips])).to_numpy()
    R["info0"] = info0
    R["deffs"] = deffs
    R["info_s"], R["info_p"], R["info_m"] = info_s, info_p, info0["metro"]
    R["Es0"] = Es0
    # cell-level table
    has = _lookup(Es0, cells.field.map(f2g).to_numpy(object), cells.fips.to_numpy(), "n")
    cells["acs_n_state"] = np.nan_to_num(has, nan=0.0)
    R["cells"] = cells

    # ---------------- models ----------------
    stage("within-institution models")
    need_all = FE2_SLOPES + UVARS
    MM = {}
    for lab, (slopes, extra, extra_pf, offset, _) in MAIN_MODELS.items():
        need = [c for c in need_all if c not in slopes]
        MM[lab] = design(cells, slopes, need, "earnings", False, extra, extra_pf, offset)
    Mb = MM["before"]
    # scripts/58 reproduction: same programs, same design, same bootstrap
    ref = s58.fe_build(cells, FE2_SLOPES, [], "earnings", per_field=False)
    assert ref["d"].inst_key.equals(Mb["d"].inst_key) and ref["d"].field.equals(Mb["d"].field), \
        "FE2 programs differ from scripts/58 (a program lacks U)"
    fb = to_fe(Mb)
    assert np.array_equal(ref["X"], fb["X"]) and np.array_equal(ref["y"], fb["y"])
    FEB = s58.fe_fit(fb)
    assert abs(FEB["beta"] - s58.fe_fit(ref)["beta"]) < 1e-12
    R["n_prog"], R["n_inst"], R["k_fields"] = FEB["n_cells"], FEB["n_inst"], FEB["n_fields"]
    R["n_prog_noacs"] = int((cells.loc[Mb["rows"], "acs_n_state"] == 0).sum())
    R["n_prog_noacs_cs"] = int(((cells.loc[Mb["rows"], "acs_n_state"] == 0) & (cells.loc[Mb["rows"], "field"] == CS)).sum())
    FEF = {lab: s58.fe_fit(to_fe(M)) for lab, M in MM.items()}
    FEZ = {lab: to_fe(M) for lab, M in MM.items()}
    for lab, M in MM.items():
        if lab != "before":
            w1 = wsolve(M["X"], M["y"], M["ic"], M["G"], np.ones(len(M["ic"])), np.array([M["names"].index("beta")]), M["code"])[0]
            assert abs(w1 - FEF[lab]["beta"]) < 1e-9, (lab, w1, FEF[lab]["beta"])
    R["FEF"] = FEF
    R["delta"] = {lab: cr1(FEZ[lab], f"c_{MAIN_MODELS[lab][1][0]}") for lab in MM if MAIN_MODELS[lab][1]}
    # scripts/58 institution cluster bootstrap (its seed), paired before/after
    bb = s58.fe_group_boot([ref], S58_FE2_SEED, B)[:, 0]
    bb2 = s58.fe_group_boot([FEZ["before"], FEZ["after"]], S58_FE2_SEED, B)
    assert np.array_equal(bb, bb2[:, 0], equal_nan=True)
    R["s58boot"] = bb2
    ci58 = (float(np.nanpercentile(bb, 2.5)), float(np.nanpercentile(bb, 97.5)))
    for a_, b_ in zip((FEB["beta"],) + ci58, S58_FE2_PUB):
        assert abs(round(a_, 4) - b_) < 1e-9, ("scripts/58 FE2 does not reproduce", a_, b_)
    R["s58_ci"] = ci58
    # SD of the residualised z(F)
    R["rz"] = {lab: resid_z(FEZ[lab], MM[lab]["sdP"]) for lab in ("FE0_b", "before", "after", "E_met", "E_pf")}
    # analytic two-way cross-check
    R["cgm"] = {"before": cgm_se([FEZ["before"]], [FEZ["before"]["names"].index("beta")], [1]),
                "after": cgm_se([FEZ["after"]], [FEZ["after"]["names"].index("beta")], [1]),
                "delta": cgm_se([FEZ["after"], FEZ["before"]], [FEZ["after"]["names"].index("beta"),
                                                                 FEZ["before"]["names"].index("beta")], [1, -1])}
    # descriptive (added in verification): how U varies on the T1 programs and how it lines up with z(F)
    Za = FEZ["after"]
    jb, ju = Za["names"].index("beta"), Za["names"].index("c_U")
    xz, xu = Za["X"][:, jb], Za["X"][:, ju]
    oth = np.delete(Za["X"], [jb, ju], axis=1)
    coef = np.linalg.lstsq(oth, np.column_stack([xz, xu]), rcond=None)[0]
    rz_, ru_ = (np.column_stack([xz, xu]) - oth @ coef).T
    u_prog = MM["after"]["d"]["U"].to_numpy(float)
    cs_ = MM["after"]["d"].field.to_numpy() == CS
    R["U_desc"] = dict(sd_prog=float(np.std(u_prog, ddof=1)), sd_within=float(np.sqrt(np.mean(xu ** 2))),
                       sd_resid=float(np.sqrt(np.mean(ru_ ** 2))),
                       r_within=float(xz @ xu / np.sqrt((xz @ xz) * (xu @ xu))),
                       r_resid=float(rz_ @ ru_ / np.sqrt((rz_ @ rz_) * (ru_ @ ru_))),
                       sd_prog_cs=float(np.std(u_prog[cs_], ddof=1)),
                       r_within_cs=float(xz[cs_] @ xu[cs_] / np.sqrt((xz[cs_] @ xz[cs_]) * (xu[cs_] @ xu[cs_]))),
                       r_resid_cs=float(rz_[cs_] @ ru_[cs_] / np.sqrt((rz_[cs_] @ rz_[cs_]) * (ru_[cs_] @ ru_[cs_]))))
    # omitted-variable identity (Frisch-Waugh-Lovell): beta_after - beta_before = -delta * pi, pi = slope of U on z(F)
    # net of every other regressor
    pi_ = float(rz_ @ ru_ / (rz_ @ rz_))
    R["U_desc"]["pi"] = pi_
    ovb = (FEF["after"]["beta"] - FEF["before"]["beta"]) + R["delta"]["after"][0] * pi_
    assert abs(ovb) < 1e-10, ("omitted-variable identity fails", ovb)
    R["unmapped_all"] = {f: int(v) for f, v in cells[~cells.field.isin(list(f2g))].groupby("field").size().items()}
    dcs = MM["after"]["d"][cs_]
    cst = dcs.groupby("STABBR").agg(U=("U", "first"), n_programs=("U", "size"), acs_n=("acs_n_state", "first"))
    assert (dcs.groupby("STABBR").U.nunique() == 1).all()
    R["cs_U_states"] = cst.sort_index().sort_values(["U", "n_programs"], ascending=[False, False], kind="mergesort")
    del FEZ, Za, oth, ref, fb, bb
    gc.collect()

    # exclusive-mapping fields only (own programs)
    cex = cells[cells.field.map(kind).eq("exclusive")]
    MX = {lab: design(cex, FE2_SLOPES, ["U"], "earnings", False, extra) for lab, extra in
          (("before", ()), ("after", ("U",)))}
    R["excl_fit"] = {lab: s58.fe_fit(to_fe(M)) for lab, M in MX.items()}
    R["excl_fields_dropped"] = sorted(set(Mb["fields"]) - set(MX["before"]["fields"]))

    # ---------------- T2 per-field models ----------------
    stage("T2 per-field models")
    PF = {}
    # before / after (T2, state-level U) / met (exploratory: metro-level U); `need` holds both U columns so the three
    # versions sit on identical programs (every mapped program has both)
    for lab, extra in (("before", ()), ("after", ("U",)), ("met", ("U_met",))):
        PF[f"y4_{lab}"] = design(cells, FE2_SLOPES, ["U", "U_met"], "earnings", True, extra)
        for h in HZ3:
            other = [HZ_COL[o] for o in HZ3 if o != h]
            PF[f"h3{h}_{lab}"] = design(cells, FE2_SLOPES, other + ["U", "U_met"], HZ_COL[h], True, extra)
    for k in PF:
        base = PF[k.rsplit("_", 1)[0] + "_before"]
        assert base["d"].inst_key.equals(PF[k]["d"].inst_key) and base["d"].field.equals(PF[k]["d"].field)
    PFF = {k: s58.fe_fit(to_fe(M)) for k, M in PF.items()}
    PFB = {}
    for k, M in PF.items():
        seed = S58_PF_SEED if k.startswith("y4") else S58_H3_SEED
        PFB[k] = pd.DataFrame(s58.fe_boot_pf(to_fe(M), seed, B, PF_MIN_DISTINCT), columns=M["fields"])
        gc.collect()
    deep = pd.read_csv(DEEP_CSV).set_index("field")
    # reproduction of scripts/58 per-field beta_f and CIs (4-yr and the 4/1/5-yr average)
    fl = PF["y4_before"]["fields"]
    chk = pd.DataFrame(dict(b=PFF["y4_before"]["beta_f"], lo=PFB["y4_before"].quantile(0.025),
                            hi=PFB["y4_before"].quantile(0.975)))
    for mine, theirs in (("b", "fe_FE2_beta"), ("lo", "fe_FE2_ci_lo"), ("hi", "fe_FE2_ci_hi")):
        dd = (chk[mine] - deep[theirs].reindex(fl)).abs().max()
        assert dd < CSV_TOL, (mine, theirs, dd)
    avg_b = sum(PFF[f"h3{h}_before"]["beta_f"] for h in HZ3) / 3
    avg_B = sum(PFB[f"h3{h}_before"] for h in HZ3) / 3
    fl3 = PF["h3y4_before"]["fields"]
    for mine, theirs in ((avg_b, "fe_FE2_h3avg_beta"), (avg_B.quantile(0.025), "fe_FE2_h3avg_ci_lo"),
                         (avg_B.quantile(0.975), "fe_FE2_h3avg_ci_hi")):
        dd = (mine.reindex(fl3) - deep[theirs].reindex(fl3)).abs().max()
        assert dd < CSV_TOL, (theirs, dd)
    R["PF"] = {k: dict(n=PFF[k]["n_cells"], n_inst=PFF[k]["n_inst"], k=PFF[k]["n_fields"],
                       n_cs=int(PFF[k]["n_by_field"][CS]), b_cs=float(PFF[k]["beta_f"][CS]),
                       boot_cs=PFB[k][CS].to_numpy()) for k in PF}
    R["PF_delta"] = {k: cr1(to_fe(PF[k]), "c_U" if k.endswith("_after") else "c_U_met")
                     for k in PF if not k.endswith("_before")}
    del PFB, PFF
    for k in [k for k in PF if k.endswith("_before")]:
        del PF[k]                                          # the replicate engines need only the U models
    gc.collect()

    # ---------------- exploratory: spec (a) on its OWN sample (no SAT requirement) ----------------
    # The main spec needs SAT_AVG, which drops test-blind institutions; spec (a) (institution FE + field FE + beta z(F))
    # keeps them. Pooled beta and the CS beta_f, before / after (state U) / metro U, on scripts/58's FE0 programs.
    stage("exploratory: spec (a) on its own sample")
    MA = {lab: design(cells, [], ["U", "U_met"], "earnings", False, extra)
          for lab, extra in (("before", ()), ("after", ("U",)), ("met", ("U_met",)))}
    refA = s58.fe_build(cells, [], [], "earnings", per_field=False)
    assert refA["d"].inst_key.equals(MA["before"]["d"].inst_key) and refA["d"].field.equals(MA["before"]["d"].field), \
        "spec (a) programs differ from scripts/58 FE0"
    assert abs(s58.fe_fit(refA)["beta"] - s58.fe_fit(to_fe(MA["before"]))["beta"]) < 1e-12
    del refA
    FA = {lab: s58.fe_fit(to_fe(M)) for lab, M in MA.items()}
    bbA = s58.fe_group_boot([to_fe(M) for M in MA.values()], SEED_A_POOL, B)
    PA = {lab: design(cells, [], ["U", "U_met"], "earnings", True, extra)
          for lab, extra in (("before", ()), ("after", ("U",)), ("met", ("U_met",)))}
    PAF = {lab: s58.fe_fit(to_fe(M)) for lab, M in PA.items()}
    dA = (PAF["before"]["beta_f"] - deep["fe_FE0_beta"].reindex(PA["before"]["fields"])).abs().max()
    assert dA < CSV_TOL, ("spec (a) per-field beta_f does not reproduce scripts/58", dA)
    PAB = {lab: s58.fe_boot_pf(to_fe(M), SEED_A_PF, B, PF_MIN_DISTINCT_A)[:, M["fields"].index(CS)]
           for lab, M in PA.items()}
    dcsA = MA["before"]["d"][MA["before"]["d"].field == CS]
    R["A"] = dict(n=FA["before"]["n_cells"], n_inst=FA["before"]["n_inst"], k=FA["before"]["n_fields"],
                  beta={lab: FA[lab]["beta"] for lab in FA}, se_cr1={lab: FA[lab]["se"] for lab in FA},
                  boot=bbA, b_cs={lab: float(PAF[lab]["beta_f"][CS]) for lab in PAF},
                  boot_cs=PAB, n_cs=int(PAF["before"]["n_by_field"][CS]),
                  cs_states=dcsA.STABBR.value_counts().sort_index(),
                  delta={lab: cr1(to_fe(MA[lab]), "c_U" if lab == "after" else "c_U_met") for lab in ("after", "met")},
                  delta_cs={lab: cr1(to_fe(PA[lab]), "c_U" if lab == "after" else "c_U_met") for lab in ("after", "met")})
    for lab in ("after", "met"):
        PF[f"a_{lab}"] = PA[lab]
    del PA, PAF, PAB, bbA, FA
    gc.collect()

    # ---------------- T3 setup ----------------
    gm = pd.read_csv(GAP_MAP).set_index("field")
    cnt_f = cells.groupby("field").size()
    t3f = sorted(f for f in cnt_f.index if cnt_f[f] >= NMIN and f in f2g)
    R["t3_unmapped"] = sorted(f for f in cnt_f.index if cnt_f[f] >= NMIN and f not in f2g)
    T3 = t3_setup(cells, t3f, f2g, R["acs_n_group"])
    rho_chk = np.array([s55.spear(T3["FD"][f]["P"], T3["FD"][f]["Y"]) for f in t3f])
    R["t3_repro"] = float(np.nanmax(np.abs(rho_chk - gm.spearman.reindex(t3f).to_numpy())))
    assert R["t3_repro"] < 1e-9, "baseline rho_f does not reproduce the gap map"
    R["t3_fields"] = t3f

    # ---------------- replicate engines ----------------
    tg_main = register("main", [(lab, MM[lab], ["beta"] + (["c_U"] if lab == "after" else [])
                                 + (["c_U_dsum"] if lab == "E_dsum" else [])) for lab in MM])
    tg_excl = register("excl", [(lab, MX[lab], ["beta"]) for lab in MX])
    tg_a = register("a", [(lab, MA[lab], ["beta"]) for lab in MA])
    _ACS.update(geo=geo, L_occ=cells["L_occ"].to_numpy(float), specsets=["main", "excl", "a"],
                pf_models=dict(PF), T3=T3,
                U0=cells["U"].to_numpy(float), tau2_g0=info_s["tau2_f"])
    stage(f"ACS replicate refits (81 weight sets, {workers} workers)")
    AC = run_acs(workers)
    # main weights through the replicate path must equal the point estimates
    for key, lab in (("main", tg_main), ("excl", tg_excl), ("a", tg_a)):
        with threadpool_limits(1):
            pt = _eval(_SS[key], np.ones(len(_SS[key]["ic"])))
        assert np.allclose(AC[0][key], pt, rtol=0, atol=1e-10), key
    # fixed-shrinkage check at the main weight reproduces the point estimates
    ia = [tg_main.index("after:beta"), tg_main.index("after:c_U")]
    assert np.allclose(AC[0]["fix"]["main_after"], AC[0]["main"][ia], rtol=0, atol=1e-10)
    for kk, vv in AC[0]["fix"].items():
        if kk not in ("main_after", "T3"):
            assert abs(vv - AC[0][kk]) < 1e-10, kk
    assert np.allclose(AC[0]["fix"]["T3"], AC[0]["T3"], rtol=0, atol=1e-12, equal_nan=True)
    R["AC"] = AC
    stage(f"two-way replicates, main specset ({len(tg_main)} targets)")
    R["reps_main"] = run_reps("main", workers)
    R["tg_main"] = tg_main
    stage("two-way replicates, exclusive-only specset")
    R["reps_excl"] = run_reps("excl", workers)
    R["tg_excl"] = tg_excl
    stage("two-way replicates, spec (a) own-sample specset (exploratory)")
    R["reps_a"] = run_reps("a", workers)
    R["tg_a"] = tg_a
    stage("T3 replicates")
    R["T3_point"] = t3_point(T3, cells["U"].to_numpy(float), info_s["tau2_f"])
    R["T3_reps"] = run_t3(workers)
    return R


# =============================================================================================
# summaries
# =============================================================================================
def acs_variants(est: float, se_other: float, full, fix, corr: bool = False) -> dict:
    """Revision table: the total SE / 95% CI of a statistic under four ACS replicate-variance versions:
    full EB re-run per replicate (full) or shrinkage held at its main-weight value (fix), each centred at the
    replicate mean (mean; primary = full_mean) or at the full-sample value (th0; first version = full_th0).
    se_other: the Scorecard part (two-way SE, or the institution-bootstrap SD for one field's coefficient)."""
    out = {}
    for tag, reps in (("full", full), ("fix", fix)):
        a = acs_var(est, reps)
        out[f"{tag}_bias"] = a["bias"]
        for c, v in (("mean", a["v"]), ("th0", a["v_th0"])):
            se = float(np.sqrt(se_other ** 2 + v))
            nlo, nhi = est - Z975 * se, est + Z975 * se
            lo, hi = fisher_ci(est, se) if corr else (nlo, nhi)
            out[f"{tag}_{c}"] = dict(se_acs=float(np.sqrt(v)), se=se, lo=lo, hi=hi, nlo=nlo, nhi=nhi)
    out.update(est=float(est), se_other=float(se_other), corr=corr)
    return out


def summarize(R: dict) -> dict:
    S: dict = {}
    AC = R["AC"]
    RR = range(1, N_REP + 1)
    reps = R["reps_main"]
    tg = R["tg_main"]
    ti = {t: i for i, t in enumerate(tg)}
    acs_main = np.stack([AC[r]["main"] for r in RR])
    th0 = AC[0]["main"]

    def comb(fn, corr=False):
        """two-way + ACS summary of a scalar function of the target vector."""
        est = fn(reps["point"])
        assert abs(fn(th0) - est) < 1e-10
        jk = np.array([fn(v) for v in reps["jack"]])
        ib = np.array([fn(v) for v in reps["inst"]])
        xb = np.array([fn(v) for v in reps["fxi"]])
        ac = np.array([fn(v) for v in acs_main])
        return tw(est, jk, ib, xb, acs_var(est, ac), corr)

    S["beta"] = {lab: comb(lambda v, i=ti[f"{lab}:beta"]: v[i]) for lab in MAIN_MODELS}
    S["pair"] = {a: comb(lambda v, i=ti[f"{a}:beta"], j=ti[f"{b}:beta"]: v[i] - v[j]) for a, b in MAIN_PAIRS}
    S["share"] = comb(lambda v: (v[ti["before:beta"]] - v[ti["after:beta"]]) / v[ti["before:beta"]])
    S["delta_U"] = comb(lambda v: v[ti["after:c_U"]])
    S["delta_dsum"] = comb(lambda v: v[ti["E_dsum:c_U_dsum"]])
    # revision: ACS replicate variance with the shrinkage held at the main-weight value (T1 after / Delta / delta)
    fx = np.stack([AC[r]["fix"]["main_after"] for r in RR])          # columns: after beta, after c_U
    b_before = th0[ti["before:beta"]]
    rev = {}
    for nm, r_, full_col, fix_col in (("delta_U", S["delta_U"], ti["after:c_U"], fx[:, 1]),
                                      ("Delta_T1", S["pair"]["after"], None, fx[:, 0] - b_before),
                                      ("beta_after", S["beta"]["after"], ti["after:beta"], fx[:, 0])):
        full = acs_main[:, full_col] if full_col is not None else acs_main[:, ti["after:beta"]] - acs_main[:, ti["before:beta"]]
        rev[nm] = acs_variants(r_["est"], r_["se_tw"], full, fix_col)
    # replicate tau of the state level (the source of the replicate shift)
    t2r = np.array([AC[r]["tau2"]["state"] for r in RR])
    t20 = AC[0]["tau2"]["state"]
    assert abs(t20 - R["info_s"]["tau2"]) < 1e-15
    S["tau_reps"] = dict(tau0=float(np.sqrt(t20)), med=float(np.median(np.sqrt(t2r))), lo=float(np.sqrt(t2r.min())),
                         hi=float(np.sqrt(t2r.max())), n_above=int((t2r > t20).sum()), n=int(len(t2r)),
                         mean_shift_tau2=float(t2r.mean() - t20),
                         tau_pow_med=float(np.median(np.sqrt([AC[r]["tau2"]["pow"] for r in RR]))),
                         tau_dsum_med=float(np.median(np.sqrt([AC[r]["tau2"]["dsum"] for r in RR]))),
                         tau_dev_max=float(np.sqrt(max(AC[r]["tau2"]["metro_dev"] for r in RR))))
    # scripts/58 institution-cluster bootstrap of T1 (its seed): before / after / difference
    bb = R["s58boot"]
    S["s58"] = {"before": np.nanpercentile(bb[:, 0], [2.5, 97.5]), "after": np.nanpercentile(bb[:, 1], [2.5, 97.5]),
                "diff": np.nanpercentile(bb[:, 1] - bb[:, 0], [2.5, 97.5]),
                "share": np.nanpercentile((bb[:, 0] - bb[:, 1]) / bb[:, 0], [2.5, 97.5]),
                "se_diff": float(np.nanstd(bb[:, 1] - bb[:, 0], ddof=1)),
                "p0_after": float(np.mean(bb[np.isfinite(bb[:, 1]), 1] <= 0))}
    # exclusive-only
    rx, tx = R["reps_excl"], R["tg_excl"]
    tix = {t: i for i, t in enumerate(tx)}
    acs_x = np.stack([AC[r]["excl"] for r in RR])

    def combx(fn):
        est = fn(rx["point"])
        return tw(est, np.array([fn(v) for v in rx["jack"]]), np.array([fn(v) for v in rx["inst"]]),
                  np.array([fn(v) for v in rx["fxi"]]), acs_var(est, [fn(v) for v in acs_x]))

    S["excl"] = {"before": combx(lambda v: v[tix["before:beta"]]), "after": combx(lambda v: v[tix["after:beta"]]),
                 "diff": combx(lambda v: v[tix["after:beta"]] - v[tix["before:beta"]])}
    # exploratory: spec (a) on its own sample
    ra, ta = R["reps_a"], R["tg_a"]
    tia = {t: i for i, t in enumerate(ta)}
    acs_a = np.stack([AC[r]["a"] for r in RR])

    def comba(fn):
        est = fn(ra["point"])
        return tw(est, np.array([fn(v) for v in ra["jack"]]), np.array([fn(v) for v in ra["inst"]]),
                  np.array([fn(v) for v in ra["fxi"]]), acs_var(est, [fn(v) for v in acs_a]))

    RA = R["A"]
    SA = {"pool": {}, "cs": {}}
    for nm, fn in (("before", lambda v: v[tia["before:beta"]]), ("after", lambda v: v[tia["after:beta"]]),
                   ("met", lambda v: v[tia["met:beta"]]),
                   ("diff", lambda v: v[tia["after:beta"]] - v[tia["before:beta"]]),
                   ("diff_met", lambda v: v[tia["met:beta"]] - v[tia["before:beta"]])):
        SA["pool"][nm] = comba(fn)
    bA = RA["boot"]
    for nm, dr in (("before", bA[:, 0]), ("after", bA[:, 1]), ("met", bA[:, 2]), ("diff", bA[:, 1] - bA[:, 0]),
                   ("diff_met", bA[:, 2] - bA[:, 0])):
        fin = dr[np.isfinite(dr)]
        SA["pool"][nm]["pb_lo"], SA["pool"][nm]["pb_hi"] = float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))
    bc, pc = RA["boot_cs"], RA["b_cs"]
    for nm, est, dr, akey in (("before", pc["before"], bc["before"], None), ("after", pc["after"], bc["after"], "a_after"),
                              ("met", pc["met"], bc["met"], "a_met"),
                              ("diff", pc["after"] - pc["before"], bc["after"] - bc["before"], "a_after"),
                              ("diff_met", pc["met"] - pc["before"], bc["met"] - bc["before"], "a_met")):
        fin = dr[np.isfinite(dr)]
        a = dict(v=0.0, v_th0=0.0, bias=0.0)
        if akey is not None:
            assert abs(AC[0][akey] - pc[akey[2:]]) < 1e-10
            a = acs_var(pc[akey[2:]], [AC[r][akey] for r in RR])
        se = float(np.sqrt(np.std(fin, ddof=1) ** 2 + a["v"]))
        se0 = float(np.sqrt(np.std(fin, ddof=1) ** 2 + a["v_th0"]))
        SA["cs"][nm] = dict(est=est, lo=float(np.percentile(fin, 2.5)), hi=float(np.percentile(fin, 97.5)), vacs=a["v"],
                            vacs_th0=a["v_th0"], acs_bias=a["bias"], se=se, nlo=est - Z975 * se, nhi=est + Z975 * se,
                            nlo_th0=est - Z975 * se0, nhi_th0=est + Z975 * se0, mde=MDE_K * se, n_fin=int(len(fin)))
    S["A"] = SA
    # T2
    T2 = {}
    for tag, keys in (("y4", ["y4"]), ("h3avg", [f"h3{h}" for h in HZ3])):
        bdr = {v: np.mean([R["PF"][f"{k}_{v}"]["boot_cs"] for k in keys], axis=0) for v in ("before", "after", "met")}
        pt_ = {v: float(np.mean([R["PF"][f"{k}_{v}"]["b_cs"] for k in keys])) for v in ("before", "after", "met")}
        va_ = {}
        for v in ("after", "met"):
            ac = np.array([np.mean([AC[r][f"{k}_{v}"] for k in keys]) for r in RR])
            ac0 = float(np.mean([AC[0][f"{k}_{v}"] for k in keys]))
            assert abs(ac0 - pt_[v]) < 1e-10
            va_[v] = acs_var(pt_[v], ac)
        acf = np.array([np.mean([AC[r]["fix"][f"{k}_after"] for k in keys]) for r in RR])
        pb, pa, pm = pt_["before"], pt_["after"], pt_["met"]
        bef, aft, met = bdr["before"], bdr["after"], bdr["met"]
        out = {}
        for nm, est, dr, a in (("before", pb, bef, dict(v=0.0, v_th0=0.0, bias=0.0)), ("after", pa, aft, va_["after"]),
                               ("diff", pa - pb, aft - bef, va_["after"]), ("met", pm, met, va_["met"]),
                               ("diff_met", pm - pb, met - bef, va_["met"])):
            fin = dr[np.isfinite(dr)]
            se_b = float(np.std(fin, ddof=1))
            se = float(np.sqrt(se_b ** 2 + a["v"]))
            se0 = float(np.sqrt(se_b ** 2 + a["v_th0"]))
            out[nm] = dict(est=est, lo=float(np.percentile(fin, 2.5)), hi=float(np.percentile(fin, 97.5)),
                           se_boot=se_b, vacs=a["v"], vacs_th0=a["v_th0"], acs_bias=a["bias"], se=se,
                           nlo=est - Z975 * se, nhi=est + Z975 * se, nlo_th0=est - Z975 * se0, nhi_th0=est + Z975 * se0,
                           mde=MDE_K * se, n_fin=int(len(fin)), p_le0=float(np.mean(fin <= 0)))
        # bound on the change relative to the before-remainder (percentile CI, the T2 standard)
        out["diff"]["bound_share"] = max(abs(out["diff"]["lo"]), abs(out["diff"]["hi"])) / abs(pb)
        rev[f"T2_{tag}"] = acs_variants(pa - pb, out["diff"]["se_boot"],
                                        np.array([np.mean([AC[r][f"{k}_after"] for k in keys]) for r in RR]) - pb,
                                        acf - pb)
        shd = (bef - aft) / bef
        shd = shd[np.isfinite(shd)]
        out["share"] = dict(est=(pb - pa) / pb, lo=float(np.percentile(shd, 2.5)), hi=float(np.percentile(shd, 97.5)))
        k0 = keys[0]
        out.update(n=R["PF"][f"{k0}_before"]["n"], n_inst=R["PF"][f"{k0}_before"]["n_inst"],
                   k=R["PF"][f"{k0}_before"]["k"], n_cs=R["PF"][f"{k0}_before"]["n_cs"],
                   hz={k: tuple(R["PF"][f"{k}_{v}"]["b_cs"] for v in ("before", "after", "met")) for k in keys},
                   delta={k: R["PF_delta"][f"{k}_after"] for k in keys},
                   delta_met={k: R["PF_delta"][f"{k}_met"] for k in keys})
        T2[tag] = out
    S["T2"] = T2
    # T3
    ns = len(T3_STATS)
    pt = R["T3_point"]
    ac = np.stack([AC[r]["T3"][:ns] for r in RR])
    acf = np.stack([AC[r]["fix"]["T3"][:ns] for r in RR])
    assert np.allclose(AC[0]["T3"], pt, equal_nan=True)
    T3S = {}
    for j, nm in enumerate(T3_STATS):
        T3S[nm] = tw(pt[j], R["T3_reps"]["jack"][:, j], R["T3_reps"]["inst"][:, j], R["T3_reps"]["fxi"][:, j],
                     acs_var(pt[j], ac[:, j]), nm in T3_CORR)
        rev[f"T3_{nm}"] = acs_variants(pt[j], T3S[nm]["se_tw"], ac[:, j], acf[:, j], nm in T3_CORR)
    S["T3"] = T3S
    S["rev"] = rev
    vals = pt[ns:].reshape(len(R["t3_fields"]), 5)
    S["T3_fields"] = pd.DataFrame(vals, columns=["rho", "rho_FU", "sd_U", "partial", "tau"], index=R["t3_fields"])
    S["n_tau0"] = int((S["T3_fields"].tau == 0).sum())
    return S


# =============================================================================================
# output
# =============================================================================================
def _f(x, nd=4, sign=True):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _ci(lo, hi, nd=4):
    return f"[{_f(lo, nd)}, {_f(hi, nd)}]"


def _status(lo, hi):
    if lo > 0:
        return "above 0"
    if hi < 0:
        return "below 0"
    return "includes 0"


ROWS: list = []


def row(section, quantity, sample, est, lo=np.nan, hi=np.nan, se=np.nan, ci_type="", k=np.nan, n=np.nan, n_inst=np.nan,
        note="", **kw):
    r = dict(section=section, quantity=quantity, sample=sample, k_fields=k, n_programs=n, n_institutions=n_inst,
             estimate=est, se=se, ci_lo=lo, ci_hi=hi, ci_type=ci_type, note=note)
    r.update(kw)
    ROWS.append(r)


def write_outputs(R: dict, S: dict):
    cells = R["cells"]
    n, ni, k = R["n_prog"], R["n_inst"], R["k_fields"]
    smp = f"FE2 programs ({n:,} programs, {ni} institutions, {k} fields)"
    TW = "two-way (field, institution) + ACS replicate variance (replicate-mean-centred); normal 95% CI"
    # ---- CSV rows ----
    for lab, (sl, ex, expf, off, desc) in MAIN_MODELS.items():
        r = S["beta"][lab]
        row("T1/sens: slope", f"beta: {desc}", smp, r["est"], r["lo"], r["hi"], r["se"], TW, k, n, ni,
            se_twoway_scorecard=r["se_tw"], v_acs=r["vacs"], v_acs_theta0=r["vacs_th0"], acs_bias=r["acs_bias"],
            ci_lo_theta0=r["lo_th0"], ci_hi_theta0=r["hi_th0"], se_cr1=R["FEF"][lab]["se"],
            inst_boot_lo=r["inst_lo"], inst_boot_hi=r["inst_hi"], v_field=r["vf"], v_inst=r["vs"],
            v_field_x_inst=r["vi"], twoway_fallback=r["fallback"], mcse_ci_endpoint=r["mcse"],
            resid_var_share=R["FEF"][lab].get("resid_var_share", np.nan))
    for a, b in MAIN_PAIRS:
        r = S["pair"][a]
        row("T1/sens: change", f"Delta beta = [{MAIN_MODELS[a][4]}] - [{MAIN_MODELS[b][4]}]", smp, r["est"], r["lo"],
            r["hi"], r["se"], TW, k, n, ni, se_twoway_scorecard=r["se_tw"], v_acs=r["vacs"], v_acs_theta0=r["vacs_th0"],
            acs_bias=r["acs_bias"], ci_lo_theta0=r["lo_th0"], ci_hi_theta0=r["hi_th0"], mde80=r["mde"],
            inst_boot_lo=r["inst_lo"], inst_boot_hi=r["inst_hi"], twoway_fallback=r["fallback"],
            note="T1 (pre-specified)" if a == "after" else "exploratory")
    r = S["share"]
    row("T1", "share of beta removed (beta_before - beta_after) / beta_before", smp, r["est"], r["lo"], r["hi"], r["se"],
        TW, k, n, ni, inst_boot_lo=r["inst_lo"], inst_boot_hi=r["inst_hi"])
    r = S["delta_U"]
    row("T1", "delta (coefficient on U, log points per log point)", smp, r["est"], r["lo"], r["hi"], r["se"], TW, k, n, ni,
        se_cr1=R["delta"]["after"][1], se_twoway_scorecard=r["se_tw"], v_acs=r["vacs"], v_acs_theta0=r["vacs_th0"],
        acs_bias=r["acs_bias"], ci_lo_theta0=r["lo_th0"], ci_hi_theta0=r["hi_th0"],
        inst_boot_lo=r["inst_lo"], inst_boot_hi=r["inst_hi"])
    r = S["delta_dsum"]
    row("sens: deff", "delta on U built with deff = ratio of sums (revision; exploratory)", smp, r["est"], r["lo"], r["hi"],
        r["se"], TW, k, n, ni, se_twoway_scorecard=r["se_tw"], v_acs=r["vacs"], v_acs_theta0=r["vacs_th0"],
        acs_bias=r["acs_bias"])
    for kk, vv in (("deff (median ratio; pre-specified, used)", R["deffs"]["state"]),
                   ("deff (mean ratio, same cells)", R["info_s"]["deff_mean"]),
                   ("deff (ratio of sums, same cells; E_dsum)", R["info_s"]["deff_sum"]),
                   ("tau, state level (deff median)", np.sqrt(R["info_s"]["tau2"])),
                   ("tau, state level (deff ratio of sums)", np.sqrt(R["info_dsum"]["tau2"]))):
        row("sens: deff", kk, f"ACS category x state cells (n >= {DEFF_NMIN} for deff)", vv,
            n=R["info_s"]["deff_cells"] if kk.startswith("deff") else R["info_s"]["n_cells"])
    tr = S["tau_reps"]
    for kk, vv in (("tau main weight", tr["tau0"]), ("tau replicate median", tr["med"]), ("tau replicate min", tr["lo"]),
                   ("tau replicate max", tr["hi"]), ("replicates with tau above main", tr["n_above"]),
                   ("mean replicate tau2 minus main tau2", tr["mean_shift_tau2"]),
                   ("state of work: tau replicate median", tr["tau_pow_med"]),
                   ("metro deviation: tau_d replicate max", tr["tau_dev_max"])):
        row("revision: ACS replicates", kk, f"{tr['n']} ACS replicate weight sets, state level", vv)
    for key, rv in S["rev"].items():
        for var in ("full_th0", "full_mean", "fix_th0", "fix_mean"):
            x = rv[var]
            row("revision: ACS variance versions", f"{key} [{var}]", "as the row of the same statistic", rv["est"],
                x["lo"], x["hi"], x["se"], ("Fisher-z" if rv["corr"] else "normal") + " 95% CI; SE = sqrt(se_other^2 + V_acs)",
                se_other=rv["se_other"], se_acs=x["se_acs"],
                acs_bias=rv["full_bias"] if var.startswith("full") else rv["fix_bias"],
                note={"full_th0": "first version (theta0-centred)", "full_mean": "primary (replicate-mean-centred)",
                      "fix_th0": "check: shrinkage fixed, theta0-centred",
                      "fix_mean": "check: shrinkage fixed, replicate-mean-centred"}[var])
    s5 = S["s58"]
    row("T1", "beta before, scripts/58 institution cluster bootstrap (seed [58,3,1])", smp, S["beta"]["before"]["est"],
        s5["before"][0], s5["before"][1], R["FEF"]["before"]["se"], "institution pairs bootstrap percentile (B=999); SE = CR1",
        k, n, ni)
    row("T1", "beta after, same draws", smp, S["beta"]["after"]["est"], s5["after"][0], s5["after"][1],
        R["FEF"]["after"]["se"], "institution pairs bootstrap percentile (B=999); SE = CR1", k, n, ni)
    row("T1", "Delta beta, same draws (paired)", smp, S["pair"]["after"]["est"], s5["diff"][0], s5["diff"][1],
        s5["se_diff"], "institution pairs bootstrap percentile (B=999); SE = bootstrap SD", k, n, ni)
    row("T1", "share of beta removed, same draws", smp, S["share"]["est"], s5["share"][0], s5["share"][1], np.nan,
        "institution pairs bootstrap percentile (B=999)", k, n, ni)
    for tag, c in R["cgm"].items():
        row("T1 cross-check", f"analytic CGM two-way SE ({tag})", smp, np.nan, se=c["se"], ci_type="CR1 field + CR1 inst - HC1",
            k=k, n=n, n_inst=ni, se_inst_only=c["se_inst"], cgm_fallback=c["fallback"])
    for lab, rz in R["rz"].items():
        for kk, vv in rz.items():
            row("residualised z(F)", f"{kk} [{MAIN_MODELS[lab][4]}]", smp, vv, k=k, n=n, n_inst=ni)
    for nm in ("before", "after", "diff"):
        r = S["excl"][nm]
        fx = R["excl_fit"]["before"]
        row("sens: exclusive-only", f"{nm}", f"exclusive-mapping fields ({fx['n_cells']} programs, {fx['n_inst']} inst.)",
            r["est"], r["lo"], r["hi"], r["se"], TW, fx["n_fields"], fx["n_cells"], fx["n_inst"],
            inst_boot_lo=r["inst_lo"], inst_boot_hi=r["inst_hi"], mde80=r["mde"], v_acs=r["vacs"],
            v_acs_theta0=r["vacs_th0"], note="exploratory")
    for tag, T in S["T2"].items():
        for nm in ("before", "after", "diff", "met", "diff_met"):
            r = T[nm]
            row(f"T2 CS {tag}", {"met": "after, metro-level U (exploratory)",
                                 "diff_met": "change, metro-level U (exploratory)"}.get(nm, nm),
                f"{T['n']:,} programs ({T['n_cs']} CS), {T['n_inst']} institutions, {T['k']} fields",
                r["est"], r["lo"], r["hi"], r["se_boot"], "institution pairs bootstrap percentile (scripts/58 seed)",
                T["k"], T["n"], T["n_inst"], normal_lo_with_acs=r["nlo"], normal_hi_with_acs=r["nhi"],
                se_with_acs=r["se"], v_acs=r["vacs"], v_acs_theta0=r["vacs_th0"], acs_bias=r["acs_bias"],
                normal_lo_theta0=r["nlo_th0"], normal_hi_theta0=r["nhi_th0"], mde80=r["mde"], share_draws_le0=r["p_le0"],
                bound_share_of_before=r.get("bound_share", np.nan),
                note="pre-specified" if nm in ("before", "after", "diff") else "exploratory")
        r = T["share"]
        row(f"T2 CS {tag}", "share of CS beta_f removed by U", f"{T['n']:,} programs ({T['n_cs']} CS)", r["est"], r["lo"],
            r["hi"], ci_type="institution pairs bootstrap percentile (scripts/58 seed)", k=T["k"], n=T["n"],
            n_inst=T["n_inst"])
        for h, (b_, a_, m_) in T["hz"].items():
            row(f"T2 CS {tag}", f"CS beta_f at horizon {h}: before / after / metro", "point estimates", a_,
                k=T["k"], n=T["n"], n_inst=T["n_inst"], before=b_, metro=m_, delta_U=T["delta"][h][0],
                delta_U_se_cr1=T["delta"][h][1], delta_Umet=T["delta_met"][h][0], delta_Umet_se_cr1=T["delta_met"][h][1])
    RA, SA = R["A"], S["A"]
    smpA = f"spec (a) own sample ({RA['n']:,} programs, {RA['n_inst']} institutions, {RA['k']} fields)"
    for nm in ("before", "after", "met", "diff", "diff_met"):
        r = SA["pool"][nm]
        row("exploratory: spec (a) own sample, pooled", nm, smpA, r["est"], r["lo"], r["hi"], r["se"], TW, RA["k"],
            RA["n"], RA["n_inst"], inst_boot_lo=r["pb_lo"], inst_boot_hi=r["pb_hi"], mde80=r["mde"],
            v_acs=r["vacs"], twoway_fallback=r["fallback"], note="exploratory; institution bootstrap seed [69, 20]")
        r = SA["cs"][nm]
        row("exploratory: spec (a) own sample, CS beta_f", nm, f"{smpA}; {RA['n_cs']} CS programs", r["est"], r["lo"],
            r["hi"], r["se"], "institution pairs bootstrap percentile (seed [69, 21]); SE adds V_acs", RA["k"], RA["n"],
            RA["n_inst"], normal_lo_with_acs=r["nlo"], normal_hi_with_acs=r["nhi"], v_acs=r["vacs"],
            v_acs_theta0=r["vacs_th0"], normal_lo_theta0=r["nlo_th0"], normal_hi_theta0=r["nhi_th0"], mde80=r["mde"],
            note="exploratory")
    for lab in ("after", "met"):
        row("exploratory: spec (a) own sample, pooled", f"delta on {'U' if lab == 'after' else 'U_met'}", smpA,
            RA["delta"][lab][0], se=RA["delta"][lab][1], ci_type="CR1 (institution)", k=RA["k"], n=RA["n"],
            n_inst=RA["n_inst"])
    for st, v in RA["cs_states"].items():
        row("exploratory: spec (a) own sample, CS beta_f", f"CS programs in {st}", smpA, float(v))
    for kk, vv in R["U_desc"].items():
        row("U descriptive (exploratory)", kk, smp, vv, k=k, n=n, n_inst=ni)
    for st, r in R["cs_U_states"].iterrows():
        row("U descriptive (exploratory)", f"CS U in {st}", smp, r["U"], n=r["n_programs"], acs_persons_cell=r["acs_n"])
    for nm, r in S["T3"].items():
        row("T3 (exploratory)", nm, f"fields with >= {NMIN} programs and an ACS category", r["est"], r["rlo"], r["rhi"],
            r["se"], ("two-way (field, institution) + ACS replicate variance (replicate-mean-centred); Fisher-z 95% CI"
                      if r["corr"] else TW), len(R["t3_fields"]),
            note="exploratory" + ("; added in revision (post hoc)" if nm in ("S_sdU_nacs", "S_sdU_tau", "S_rho_nacs",
                                                                             "P_rho_sdU_nacs") else ""),
            inst_boot_lo=r["inst_lo"], inst_boot_hi=r["inst_hi"], normal_lo=r["lo"], normal_hi=r["hi"],
            se_twoway_scorecard=r["se_tw"], twoway_fallback=r["fallback"], v_acs=r["vacs"], v_acs_theta0=r["vacs_th0"],
            acs_bias=r["acs_bias"], ci_lo_theta0=r["fz_lo_th0"] if r["corr"] else r["lo_th0"],
            ci_hi_theta0=r["fz_hi_th0"] if r["corr"] else r["hi_th0"], mde80=r["mde"])
    row("T3 (exploratory)", "fields with tau_f = 0", f"fields with >= {NMIN} programs and an ACS category", S["n_tau0"],
        k=len(R["t3_fields"]))
    out = pd.DataFrame(ROWS)
    out.to_csv(OUT_CSV, index=False, float_format="%.10g")
    # per-field table (T3 fields) and cell table
    FT = S["T3_fields"].copy()
    FT["kind"] = FT.index.map(R["kind"])
    FT["acs_category"] = FT.index.map(R["f2g"])
    FT["acs_persons"] = FT.acs_category.map(R["acs_n_group"])
    FT["n_programs"] = cells.groupby("field").size().reindex(FT.index)
    FT.index.name = "field"
    FT.to_csv(OUT_FIELDS, float_format="%.10g")
    Es = R["Es0"].copy()
    inv = defaultdict(list)
    for f, g in R["f2g"].items():
        inv[g].append(f)
    Es.insert(1, "fields", Es.grp.map(lambda g: "|".join(inv[g])))
    Es.to_csv(OUT_CELLS, index=False, float_format="%.10g")
    write_md(R, S)


def _moved(r):
    """Conditional wording for an estimate with a CI (never asserts more than the interval supports)."""
    st = _status(r["lo"], r["hi"])
    return "the interval includes zero" if st == "includes 0" else f"the interval is {st}"


def _rci(r, nd=2):
    """reported interval (Fisher-z for correlations, normal otherwise) of a tw() result"""
    return _ci(r["rlo"], r["rhi"], nd)


def _rst(r):
    return _status(r["rlo"], r["rhi"])


def _pct(x):
    return f"{100 * x:.0f}%"


def write_md(R: dict, S: dict):
    L = []
    A = L.append
    n, ni, k = R["n_prog"], R["n_inst"], R["k_fields"]
    b0, b1 = S["beta"]["before"], S["beta"]["after"]
    d1 = S["pair"]["after"]
    sh = S["share"]
    dl = S["delta_U"]
    s5 = S["s58"]
    T2 = S["T2"]
    t3 = S["T3"]
    rev = S["rev"]
    tr = S["tau_reps"]
    info_s = R["info_s"]
    cnt = R["acs_counts"]
    tot = cnt[["all", "age", "ba", "ftfy", "employed", "not_enrolled", "pos_earn", "floor"]].sum()
    rz0, rz1, rzA = R["rz"]["before"], R["rz"]["after"], R["rz"]["FE0_b"]
    ud = R["U_desc"]
    t4, th = T2["y4"], T2["h3avg"]
    ex = S["excl"]
    fx = R["excl_fit"]["before"]
    nT3 = len(R["t3_fields"])
    cst = R["cs_U_states"]
    n_ca, n_wa = int(cst.n_programs.get("CA", 0)), int(cst.n_programs.get("WA", 0))
    csa = R["A"]["cs_states"]
    n_ca_a, n_wa_a = int(csa.get("CA", 0)), int(csa.get("WA", 0))
    SA = S["A"]
    cad = SA["cs"]["diff"]
    st_dl = _status(dl["lo"], dl["hi"])
    tau_dev0 = R["info_m"]["tau2_dev"] == 0

    A("# Local field-specific labour demand: does the within-institution department slope survive field × state earnings?")
    A("")
    A("Script: `scripts/69_local_field_demand.py` (seed 69; byte-identical on re-run, any worker count). Public data "
      "only: College Scorecard Field-of-Study and institution files, Wapman et al. 2022 ranks, ACS 2023 1-year PUMS, "
      "Census 2020 tract-to-PUMA file, OMB 2020 metro delineation, IPEDS HD2023. Descriptive, not causal. Every "
      "number below is printed by the script from its own computation; the full table is "
      "`data/interim/local_field_demand.csv`, the field × state cells `data/interim/local_field_demand_cells.csv`, the "
      "per-field table `data/interim/local_field_demand_fields.csv`. Revised on 2026-09-25 after an independent "
      "verification (section Revisions): the ACS replicate variance is now centred at the replicate mean, which "
      "changes intervals that involve the shrinkage (δ, T3), not point estimates.")
    A("")
    A("Design. The within-institution specification is scripts/58's main spec (institution FE + field FE + β·z(F) + "
      "field-specific slopes on SAT, admit rate, institution Pell share and brand G; log 4-year Scorecard earnings). "
      "The control is U, the empirical-Bayes field-specific component of log earnings of young bachelor's holders in "
      "the institution's state (ACS 2023: ages 23–35, bachelor's highest, full-time full-year, civilian employed, not "
      "enrolled; field = first field of degree). With institution and field fixed effects, adding U and adding the "
      "full log field × state earnings level L = a_f + b_s + U give the same β, because a_f and b_s are absorbed. "
      "\"Before\" and \"after\" are fitted on the same programs and, for every interval, on the same resamples.")
    A("")
    A("## Answer")
    A("")
    # summary paragraph: every clause is conditional on the computed intervals
    st_d, st_b0, st_b1 = _status(d1["lo"], d1["hi"]), _status(b0["lo"], b0["hi"]), _status(b1["lo"], b1["hi"])
    s_t1 = (f"the slope is {_f(b0['est'])} before and {_f(b1['est'])} after (log points per within-field SD of "
            f"department prestige), a change of {_f(d1['est'])} {_ci(d1['lo'], d1['hi'])}, {_moved(d1)}")
    if st_d == "includes 0":
        s_t1 += (f"; a change of {d1['mde']:.4f} ({d1['mde'] / abs(b0['est']):.0%} of the before-slope) would have "
                 f"been detected with 80% power")
    s_after = (f"the before-slope's interval {_ci(b0['lo'], b0['hi'])} {'includes zero' if st_b0 == 'includes 0' else 'is ' + st_b0}"
               f" and the after-slope's {_ci(b1['lo'], b1['hi'])} "
               f"{'includes zero' if st_b1 == 'includes 0' else 'is ' + st_b1} (two-way (field, institution) + ACS "
               f"replicate variance; with scripts/58's institution-only bootstrap they are {_ci(*s5['before'])} and "
               f"{_ci(*s5['after'])})")
    if st_dl == "above 0":
        s_delta = (f"The control itself is not empty: its coefficient is δ = {_f(dl['est'], 3)} "
                   f"{_ci(dl['lo'], dl['hi'], 3)} (above 0), so within an institution the programs in fields that the "
                   f"state pays well earn more; β does not move because U is almost uncorrelated with the residualised "
                   f"department rank (π = {_f(ud['pi'], 4)}, the slope of U on z(F) net of every other regressor; "
                   f"within-institution correlation {_f(ud['r_within'], 3)}), and Δβ = −δ × π exactly.")
    else:
        s_delta = (f"The coefficient on U is δ = {_f(dl['est'], 3)} {_ci(dl['lo'], dl['hi'], 3)} ({st_dl}); β does "
                   f"not move because Δβ = −δ × π exactly and U is almost uncorrelated with the residualised "
                   f"department rank (π = {_f(ud['pi'], 4)}; within-institution correlation {_f(ud['r_within'], 3)}).")
    st_cs = _status(t4["after"]["lo"], t4["after"]["hi"])
    st_csh = _status(th["after"]["lo"], th["after"]["hi"])
    s_t2 = (f"For computer science (T2) the state-level control changes the remainder by {_f(t4['diff']['est'])} "
            f"{_ci(t4['diff']['lo'], t4['diff']['hi'])} with 4-year earnings ({_f(t4['before']['est'])} → "
            f"{_f(t4['after']['est'])}); the interval bounds the change at {_pct(t4['diff']['bound_share'])} of the "
            f"before-remainder (MDE {t4['diff']['mde']:.4f}), and the after-control interval "
            f"{_ci(t4['after']['lo'], t4['after']['hi'])} is {st_cs}. Averaged over 4-, 1- and 5-year earnings the "
            f"remainder goes from {_f(th['before']['est'])} to {_f(th['after']['est'])} (change {_f(th['diff']['est'])} "
            f"{_ci(th['diff']['lo'], th['diff']['hi'])}; after-control interval {_ci(th['after']['lo'], th['after']['hi'])}, "
            f"{st_csh}). The main-spec sample holds only {n_ca} California and {n_wa} Washington CS programs"
            + (" and the metro-level U adds nothing (τ_d = 0)" if tau_dev0 else "")
            + f", so the Bay Area and Seattle markets are examined only in exploratory spec (a) on its own programs "
            f"({n_ca_a} California, {n_wa_a} Washington CS programs; answer 7), where the CS change is "
            f"{_f(cad['est'])} {_ci(cad['lo'], cad['hi'])} ({_moved(cad)}).")
    r_tau, r_part, r_sdu = t3["S_rho_tau"], t3["S_rho_partial"], t3["S_rho_sdU"]
    r_n, r_nt, r_pn = t3["S_sdU_nacs"], t3["S_sdU_tau"], t3["P_rho_sdU_nacs"]
    s_t3 = (f"Across fields (T3, exploratory), Spearman(ρ_f, τ_f) between baseline coupling and the SD of the field's "
            f"true field × state earnings component is {_f(r_tau['est'], 2)} {_rci(r_tau)}"
            + (f", uninformative rather than a null (MDE {r_tau['mde']:.2f})" if _rst(r_tau) == "includes 0"
               else f" ({_rst(r_tau)})")
            + f"; partialling U out of each field's coupling keeps the cross-field ordering (Spearman(ρ_f, partial ρ_f) "
            f"= {_f(r_part['est'], 2)} {_rci(r_part)}); ρ_f correlates {_f(r_sdu['est'], 2)} {_rci(r_sdu)} with the "
            f"SD of the shrunk U over a field's programs, a statistic that ")
    if _rst(r_n) == "above 0" and _rst(r_nt) == "includes 0":
        s_t3 += (f"tracks the ACS sample of the field's category (Spearman {_f(r_n['est'], 2)}) rather than τ_f "
                 f"(Spearman {_f(r_nt['est'], 2)}); net of ACS persons the correlation is {_f(r_pn['est'], 2)} "
                 f"{_rci(r_pn)} (answer 5).")
    else:
        s_t3 += (f"correlates {_f(r_n['est'], 2)} with the ACS sample of the field's category and {_f(r_nt['est'], 2)} "
                 f"with τ_f; net of ACS persons the correlation is {_f(r_pn['est'], 2)} {_rci(r_pn)} (answer 5).")
    A(f"**Key question: once each department is compared with its own field's local labour market, how much of the "
      f"within-institution department-prestige slope is left?** With the ACS field × state control (T1), {s_t1}; "
      f"{s_after}. {s_delta} {s_t2} {s_t3} All of this is descriptive; U mixes local demand with who lives in a "
      f"state, and it partly contains the programs' own graduates (Caveats).")
    A("")
    # 1. T1
    if st_dl == "above 0":
        s_d1 = (f"The coefficient on U is δ = {_f(dl['est'], 3)} {_ci(dl['lo'], dl['hi'], 3)} log points of program "
                f"earnings per log point of the field-specific local level (above 0; CR1 SE {R['delta']['after'][1]:.3f}; "
                f"institution-bootstrap percentile {_ci(dl['inst_lo'], dl['inst_hi'], 3)}): U predicts program "
                f"earnings within institution, yet Δ ≈ 0 because U is almost orthogonal to the residualised z(F) "
                f"(π = {_f(ud['pi'], 4)}; Δβ = −δπ). The ACS part of this interval is approximate (Caveats): with "
                f"the shrinkage held at its main-weight value in every replicate the δ interval is "
                f"{_ci(rev['delta_U']['fix_mean']['lo'], rev['delta_U']['fix_mean']['hi'], 3)} (replicate-mean-centred) "
                f"or {_ci(rev['delta_U']['fix_th0']['lo'], rev['delta_U']['fix_th0']['hi'], 3)} (θ0-centred), and the "
                f"first version's θ0-centred full re-run gave {_ci(dl['lo_th0'], dl['hi_th0'], 3)} (Revisions).")
    else:
        s_d1 = (f"The coefficient on U is δ = {_f(dl['est'], 3)} {_ci(dl['lo'], dl['hi'], 3)} log points of program "
                f"earnings per log point of the field-specific local level ({_moved(dl)}; CR1 SE "
                f"{R['delta']['after'][1]:.3f}; first version's θ0-centred ACS variance "
                f"{_ci(dl['lo_th0'], dl['hi_th0'], 3)}; Revisions).")
    A(f"1. **T1 (pre-specified): slope change.** On the {n:,} programs of scripts/58's main spec ({ni} institutions, "
      f"{k} fields), β = {_f(b0['est'])} {_ci(b0['lo'], b0['hi'])} before and {_f(b1['est'])} "
      f"{_ci(b1['lo'], b1['hi'])} after adding U. Change Δ = {_f(d1['est'])} {_ci(d1['lo'], d1['hi'])} (two-way + "
      f"ACS, SE {d1['se']:.4f}); {_moved(d1)}. The share of β removed is {_f(sh['est'], 2)} "
      f"{_ci(sh['lo'], sh['hi'], 2)} (a ratio; its interval is wide whenever β_before's interval comes near zero). "
      f"Minimum detectable |Δ| at 80% power (two-sided 5%): {d1['mde']:.4f} log points, "
      f"{d1['mde'] / abs(b0['est']):.0%} of β_before. {s_d1} With scripts/58's own institution cluster bootstrap "
      f"(its seed, paired draws): before {_f(b0['est'])} {_ci(*s5['before'])} (reproduces scripts/58), after "
      f"{_f(b1['est'])} {_ci(*s5['after'])}, Δ {_ci(*s5['diff'])}.")
    # 2. T2
    A(f"2. **T2 (pre-specified): computer-science remainder.** Per-field version of the main spec with a common δ on "
      f"U. (a) 4-year earnings, {t4['n']:,} programs ({t4['n_cs']} CS): CS β_f {_f(t4['before']['est'])} "
      f"{_ci(t4['before']['lo'], t4['before']['hi'])} before, {_f(t4['after']['est'])} "
      f"{_ci(t4['after']['lo'], t4['after']['hi'])} after; paired change {_f(t4['diff']['est'])} "
      f"{_ci(t4['diff']['lo'], t4['diff']['hi'])}; share removed {_f(t4['share']['est'], 2)} "
      f"{_ci(t4['share']['lo'], t4['share']['hi'], 2)} (institution cluster bootstrap percentile, scripts/58 seed; "
      f"with the ACS replicate variance added, normal CI after {_ci(t4['after']['nlo'], t4['after']['nhi'])}, change "
      f"{_ci(t4['diff']['nlo'], t4['diff']['nhi'])}). (b) Average of 4-, 1- and 5-year earnings on the "
      f"{th['n']:,} programs released at all three horizons ({th['n_cs']} CS; scripts/58's headline CS remainder): "
      f"{_f(th['before']['est'])} {_ci(th['before']['lo'], th['before']['hi'])} before, {_f(th['after']['est'])} "
      f"{_ci(th['after']['lo'], th['after']['hi'])} after; change {_f(th['diff']['est'])} "
      f"{_ci(th['diff']['lo'], th['diff']['hi'])} (with ACS variance: after {_ci(th['after']['nlo'], th['after']['nhi'])},"
      f" change {_ci(th['diff']['nlo'], th['diff']['nhi'])}). Minimum detectable CS change at 80% power: "
      f"{t4['diff']['mde']:.4f} (4-yr), {th['diff']['mde']:.4f} (average). Per horizon (before → after): "
      + "; ".join(f"{h[2:]} {_f(v[0])} → {_f(v[1])}" for h, v in th["hz"].items()) + ". "
      f"As a bounded change: the state-level control changes the CS remainder by "
      f"{_f(t4['diff']['est'])} {_ci(t4['diff']['lo'], t4['diff']['hi'])}, at most about "
      f"{_pct(t4['diff']['bound_share'])} of β_f ({_pct(th['diff']['bound_share'])} for the average), and the "
      f"after-control interval is {st_cs} ({st_csh} for the average). This sample has only {n_ca} California and "
      f"{n_wa} Washington CS programs"
      + (", and the metro-level U equals the state-level U (τ_d = 0; answer 6)" if tau_dev0 else "")
      + f"; in exploratory spec (a), which keeps the test-blind University of California and University of "
      f"Washington campuses ({n_ca_a} California, {n_wa_a} Washington CS programs), the change is {_f(cad['est'])} "
      f"{_ci(cad['lo'], cad['hi'])} ({_moved(cad)}; answer 7).")
    # 3. residual z
    A(f"3. **Identifying variation (asked for).** z(F) is standardised within field (SD 1; in rank positions the "
      f"program-weighted RMS of the fields' SDs is {rz0['sd_raw_rank']:.1f} places). After within-institution "
      f"demeaning its SD is {rz0['sd_within']:.3f} ({rz0['sd_within_rank']:.1f} rank places). Net of the field FE and "
      f"the field-specific trait slopes of the main spec, the residual SD is {rz0['sd_resid']:.3f} "
      f"({rz0['sd_resid_rank']:.1f} rank places; {rz0['share']:.2f} of the demeaned variance). Adding U leaves "
      f"{rz1['sd_resid']:.3f} ({rz1['sd_resid_rank']:.1f} places; {rz1['share']:.2f}). In spec (a) (no trait slopes) it "
      f"is {rzA['sd_resid']:.3f} ({rzA['sd_resid_rank']:.1f} places). For computer science alone: "
      f"{rz0['sd_resid_cs']:.3f} → {rz1['sd_resid_cs']:.3f} within-field SD ({rz0['sd_resid_rank_cs']:.1f} → "
      f"{rz1['sd_resid_rank_cs']:.1f} rank places).")
    # 4. U itself
    top = ", ".join(f"{s} {_f(r['U'], 3)} ({int(r['n_programs'])})" for s, r in cst.head(3).iterrows())
    bot = ", ".join(f"{s} {_f(r['U'], 3)} ({int(r['n_programs'])})" for s, r in cst.tail(3).iterrows())
    A(f"4. **What the control varies with (descriptive).** U has one value per ACS category × state "
      f"({info_s['n_cells']:,} populated cells, {info_s['n_groups']} categories, {info_s['n_geo']} states). The SD of "
      f"its true component is τ = {np.sqrt(info_s['tau2']):.3f} log points (Paule–Mandel), against a median cell "
      f"sampling SD of {np.sqrt(np.median(R['Es0'].v)):.3f}; the median shrinkage weight is "
      f"{np.median(R['Es0'].lam):.2f}. On the {n:,} T1 programs U has SD {ud['sd_prog']:.3f}, "
      f"{ud['sd_within']:.3f} within institution and {ud['sd_resid']:.3f} net of every other regressor; its "
      f"within-institution correlation with z(F) is {_f(ud['r_within'], 3)} ({_f(ud['r_resid'], 3)} net of the other "
      f"regressors). Computer science: SD of U over the {int(cst.n_programs.sum())} CS programs {ud['sd_prog_cs']:.3f}; "
      f"within-institution correlation with z(F) {_f(ud['r_within_cs'], 3)} ({_f(ud['r_resid_cs'], 3)} net); highest "
      f"CS U (state, programs): {top}; lowest: {bot} (Appendix C). The main-spec sample requires SAT_AVG, so it has "
      f"{n_ca} CS programs in California and {n_wa} in Washington. {R['n_prog_noacs']} of the {n:,} programs sit in a "
      f"field × state cell with no ACS person (U = 0 there; {R['n_prog_noacs_cs']} of them CS). δ fixed at 1 moves β "
      f"by −π = {_f(-ud['pi'], 4)} (the offset row in answer 6).")
    # 5. T3
    r_rfu, r_mfu = t3["S_rho_rhoFU"], t3["mean_rhoFU"]
    r_rn = t3["S_rho_nacs"]
    if _rst(r_tau) == "includes 0":
        s5a = (f"{_rst(r_tau)}; with a minimum detectable correlation of {r_tau['mde']:.2f} this is uninformative "
               f"rather than a null")
    else:
        s5a = _rst(r_tau)
    A(f"5. **T3 (exploratory): does the cross-field ordering of coupling track field × state dispersion?** Across "
      f"{nT3} fields (baseline ρ_f reproduces the gap map, max |diff| {R['t3_repro']:.1e}). (i) Against the "
      f"dispersion of the field's true state-level component: Spearman(ρ_f, τ_f) = {_f(r_tau['est'], 2)} "
      f"{_rci(r_tau)} ({s5a}); {S['n_tau0']} of the {nT3} fields have τ_f = 0 (tied). (ii) Partialling U out of each "
      f"field's coupling moves the mean from {_f(t3['mean_rho']['est'], 3)} to {_f(t3['mean_partial']['est'], 3)} "
      f"(difference {_f(t3['diff_partial_minus_rho']['est'], 3)} "
      f"{_ci(t3['diff_partial_minus_rho']['lo'], t3['diff_partial_minus_rho']['hi'], 3)}) and keeps the ordering: "
      f"Spearman(ρ_f, partial ρ_f) = {_f(r_part['est'], 2)} {_rci(r_part)} (institution bootstrap with fields fixed "
      f"{_ci(r_part['inst_lo'], r_part['inst_hi'], 2)}). (iii) Spearman(ρ_f, SD of U over the field's programs) = "
      f"{_f(r_sdu['est'], 2)} {_rci(r_sdu)} ({_rst(r_sdu)}). The SD of the shrunk U is not a measure of field × state "
      f"dispersion alone: by construction it scales with the shrinkage weights τ²/(τ² + v), and so with the ACS "
      f"sample of the field's category. Across fields, Spearman(SD of U, ACS persons) = {_f(r_n['est'], 2)} "
      f"{_rci(r_n)} and Spearman(SD of U, τ_f) = {_f(r_nt['est'], 2)} {_rci(r_nt)}; ρ_f itself correlates "
      f"{_f(r_rn['est'], 2)} {_rci(r_rn)} with ACS persons, and net of ACS persons the rank-partial "
      f"Spearman(ρ_f, SD of U) is {_f(r_pn['est'], 2)} {_rci(r_pn)} ({_rst(r_pn)}; these four diagnostics were added "
      f"in the revision, post hoc). (iv) Spearman(ρ_f, ρ(F, U)_f) = {_f(r_rfu['est'], 2)} {_rci(r_rfu)} "
      f"({_rst(r_rfu)}); mean ρ(F, U)_f {_f(r_mfu['est'], 3)} {_ci(r_mfu['lo'], r_mfu['hi'], 3)}. Two-way + ACS "
      f"intervals, Fisher-z for correlations; T3 was named exploratory in advance, and its several statistics are not "
      f"corrected for multiplicity.")

    # 6. sensitivities
    def sens(lab):
        r, p = S["beta"][lab], S["pair"][lab]
        return f"{_f(r['est'])} (Δ {_f(p['est'])} {_ci(p['lo'], p['hi'])})"
    dd = S["delta_dsum"]
    A(f"6. **Exploratory sensitivities (β after; Δ against the same-program β before; two-way + ACS).** Raw unshrunk "
      f"log field × state mean {sens('E_raw')}; field-specific τ²_f shrinkage {sens('E_ftau')}; δ fixed at 1 (offset) "
      f"{sens('E_offset')}; field-specific δ_f {sens('E_pf')}; state of work instead of residence {sens('E_pow')}; "
      f"metro-level U {sens('E_met')}; leave-field-out occupation-mix index {sens('E_occ')} (no ACS replicate "
      f"variance for this row); design effect as the ratio of sums ({R['info_s']['deff_sum']:.3f} instead of the "
      f"median {R['deffs']['state']:.3f}; τ = {np.sqrt(R['info_dsum']['tau2']):.4f}; added in the revision) "
      f"{sens('E_dsum')}, with δ = {_f(dd['est'], 3)} {_ci(dd['lo'], dd['hi'], 3)}. Exclusive-mapping fields only "
      f"({fx['n_cells']:,} programs, {fx['n_inst']} institutions, "
      f"{fx['n_fields']} fields): {_f(ex['before']['est'])} → {_f(ex['after']['est'])} (Δ {_f(ex['diff']['est'])} "
      f"{_ci(ex['diff']['lo'], ex['diff']['hi'])}). Other specifications of scripts/58 on the same programs: (a) "
      f"{_f(S['beta']['FE0_b']['est'])} → {sens('FE0_a')}; (a) + field × {{SAT, ADM, Pell}} "
      f"{_f(S['beta']['FE1_b']['est'])} → {sens('FE1_a')}; (a) + field × {{SAT, G}} {_f(S['beta']['FE2sg_b']['est'])} → "
      f"{sens('FE2sg_a')}. Computer science with the metro-level U in place of the state-level U (same programs and "
      f"draws as T2): {_f(t4['met']['est'])} {_ci(t4['met']['lo'], t4['met']['hi'])} (4-yr; change "
      f"{_f(t4['diff_met']['est'])} {_ci(t4['diff_met']['lo'], t4['diff_met']['hi'])}) and {_f(th['met']['est'])} "
      f"{_ci(th['met']['lo'], th['met']['hi'])} (4/1/5-yr average; change {_f(th['diff_met']['est'])} "
      f"{_ci(th['diff_met']['lo'], th['diff_met']['hi'])})."
      + (" The metro-level field deviations are not distinguishable from sampling noise in the 1-year ACS (τ_d = 0; "
         "Method), so the metro-level U equals the state-level U of each metro's principal state: the metro rows "
         "are not a finer-grained control." if tau_dev0 else ""))
    RA = R["A"]
    pa_, ca_ = SA["pool"], SA["cs"]
    A(f"7. **Exploratory: spec (a) on its own sample, which keeps institutions without a reported SAT average.** "
      f"Institution FE + field FE + β·z(F) on scripts/58's spec-(a) programs ({RA['n']:,} programs, {RA['n_inst']} "
      f"institutions, {RA['k']} fields; {RA['n_cs']} CS programs, {n_ca_a} of them in California and "
      f"{n_wa_a} in Washington). Pooled β {_f(pa_['before']['est'])} {_ci(pa_['before']['lo'], pa_['before']['hi'])} "
      f"before, {_f(pa_['after']['est'])} {_ci(pa_['after']['lo'], pa_['after']['hi'])} with U (change "
      f"{_f(pa_['diff']['est'])} {_ci(pa_['diff']['lo'], pa_['diff']['hi'])}, {_moved(pa_['diff'])}), "
      f"{_f(pa_['met']['est'])} with the metro-level U (change {_f(pa_['diff_met']['est'])} "
      f"{_ci(pa_['diff_met']['lo'], pa_['diff_met']['hi'])}); two-way + ACS. CS β_f {_f(ca_['before']['est'])} "
      f"{_ci(ca_['before']['lo'], ca_['before']['hi'])} before, {_f(ca_['after']['est'])} "
      f"{_ci(ca_['after']['lo'], ca_['after']['hi'])} with U (change {_f(ca_['diff']['est'])} "
      f"{_ci(ca_['diff']['lo'], ca_['diff']['hi'])}, {_moved(ca_['diff'])}), {_f(ca_['met']['est'])} "
      f"{_ci(ca_['met']['lo'], ca_['met']['hi'])} with the metro-level U (change {_f(ca_['diff_met']['est'])} "
      f"{_ci(ca_['diff_met']['lo'], ca_['diff_met']['hi'])}); institution cluster bootstrap percentile (this "
      f"script's seed; with the ACS variance added the change intervals are "
      f"{_ci(ca_['diff']['nlo'], ca_['diff']['nhi'])} and {_ci(ca_['diff_met']['nlo'], ca_['diff_met']['nhi'])}). "
      f"Spec (a) has no selectivity or brand controls, so its levels are not comparable with the main spec's; only "
      f"the before/after change is of interest here.")
    A("")

    # ---- Key numbers ----
    A("## Key numbers")
    A("")
    A("β = within-institution slope, log points of median earnings per within-field SD of department prestige. "
      "Two-way = scripts/59 `twoway` (V_field from the jackknife over fields + V_inst from 999 shared institution "
      "draws − V_field×inst from 999 independent draws per field) plus the ACS replicate variance of U (80 SDR "
      "weights, whole EB re-run, centred at the replicate mean); normal 95% CI, Fisher-z for correlations (T3). "
      "Institution bootstrap = scripts/58 pairs cluster bootstrap percentile CI (B = 999; for T1 before/after/Δ the "
      "scripts/58 draws, seed [58, 3, 1]; for other rows the shared-draw replicates of the two-way engine). T2 rows: "
      "institution bootstrap percentile CI is the primary interval (one field); the normal CI adds the ACS replicate "
      "variance. MDE = minimum detectable effect at 80% power, two-sided 5% (2.80 × SE).")
    A("")
    A("| quantity | sample / specification | k fields | estimate | 95% CI (two-way + ACS) | institution-bootstrap CI | status |")
    A("|---|---|---|---|---|---|---|")
    smp = f"{n:,} programs, {ni} inst."
    for lab in MAIN_MODELS:
        r = S["beta"][lab]
        ib = _ci(*s5["before"]) if lab == "before" else (_ci(*s5["after"]) if lab == "after" else _ci(r["inst_lo"], r["inst_hi"]))
        tag = "**T1** " if lab in ("before", "after") else ""
        A(f"| {tag}β: {MAIN_MODELS[lab][4]} | {smp} | {k} | {_f(r['est'])} | {_ci(r['lo'], r['hi'])} | {ib} | "
          f"{_status(r['lo'], r['hi'])} |")
    for a_, b_ in MAIN_PAIRS:
        r = S["pair"][a_]
        ib = _ci(*s5["diff"]) if a_ == "after" else _ci(r["inst_lo"], r["inst_hi"])
        tag = "**T1** " if a_ == "after" else ""
        A(f"| {tag}Δβ: {MAIN_MODELS[a_][4]} minus {MAIN_MODELS[b_][4]} | {smp} | {k} | {_f(r['est'])} | "
          f"{_ci(r['lo'], r['hi'])} (MDE {r['mde']:.4f}) | {ib} | {_status(r['lo'], r['hi'])} |")
    A(f"| **T1** share of β removed | {smp} | {k} | {_f(sh['est'], 3)} | {_ci(sh['lo'], sh['hi'], 3)} | "
      f"{_ci(*s5['share'], 3)} | |")
    A(f"| **T1** δ on U | {smp} | {k} | {_f(dl['est'], 3)} | {_ci(dl['lo'], dl['hi'], 3)} (first version, θ0-centred "
      f"ACS variance: {_ci(dl['lo_th0'], dl['hi_th0'], 3)}; CR1 SE {R['delta']['after'][1]:.3f}) | "
      f"{_ci(dl['inst_lo'], dl['inst_hi'], 3)} | {_status(dl['lo'], dl['hi'])} |")
    A(f"| δ on U with deff = ratio of sums (revision, exploratory) | {smp} | {k} | {_f(dd['est'], 3)} | "
      f"{_ci(dd['lo'], dd['hi'], 3)} | {_ci(dd['inst_lo'], dd['inst_hi'], 3)} | {_status(dd['lo'], dd['hi'])} |")
    for nm, lab in (("before", "β"), ("after", "β + U"), ("diff", "Δβ")):
        r = S["excl"][nm]
        A(f"| exclusive-mapping fields only: {lab} | {fx['n_cells']:,} programs, {fx['n_inst']} inst. | {fx['n_fields']} | "
          f"{_f(r['est'])} | {_ci(r['lo'], r['hi'])}" + (f" (MDE {r['mde']:.4f})" if nm == "diff" else "")
          + f" | {_ci(r['inst_lo'], r['inst_hi'])} | {_status(r['lo'], r['hi'])} |")
    for tag, T in T2.items():
        lab = "4-yr" if tag == "y4" else "4/1/5-yr average, common programs"
        for nm, nml in (("before", "**T2** CS β_f before"), ("after", "**T2** CS β_f after (state U)"),
                        ("diff", "**T2** CS change (state U)"), ("met", "CS β_f after, metro U (exploratory)"),
                        ("diff_met", "CS change, metro U (exploratory)")):
            r = T[nm]
            mde = f" (MDE {r['mde']:.4f})" if nm.startswith("diff") else ""
            bnd = f"; bound {_pct(r['bound_share'])} of before" if nm == "diff" else ""
            A(f"| {nml} ({lab}) | {T['n']:,} programs ({T['n_cs']} CS), {T['n_inst']} inst. | {T['k']} | {_f(r['est'])} | "
              f"normal with ACS: {_ci(r['nlo'], r['nhi'])}{mde} | {_ci(r['lo'], r['hi'])}{bnd} | {_status(r['lo'], r['hi'])} |")
        r = T["share"]
        A(f"| **T2** share of CS β_f removed ({lab}) | {T['n']:,} programs ({T['n_cs']} CS) | {T['k']} | "
          f"{_f(r['est'], 3)} | | {_ci(r['lo'], r['hi'], 3)} | |")
    smpA = f"spec (a) own sample: {RA['n']:,} programs, {RA['n_inst']} inst."
    for nm, lab in (("before", "β"), ("after", "β + U"), ("met", "β + metro U"), ("diff", "Δβ (U)"),
                    ("diff_met", "Δβ (metro U)")):
        r = SA["pool"][nm]
        mde = f" (MDE {r['mde']:.4f})" if nm.startswith("diff") else ""
        A(f"| exploratory, spec (a) own sample, pooled: {lab} | {smpA} | {RA['k']} | {_f(r['est'])} | "
          f"{_ci(r['lo'], r['hi'])}{mde} | {_ci(r['pb_lo'], r['pb_hi'])} | {_status(r['lo'], r['hi'])} |")
    for nm, lab in (("before", "CS β_f"), ("after", "CS β_f + U"), ("met", "CS β_f + metro U"),
                    ("diff", "CS change (U)"), ("diff_met", "CS change (metro U)")):
        r = SA["cs"][nm]
        mde = f" (MDE {r['mde']:.4f})" if nm.startswith("diff") else ""
        A(f"| exploratory, spec (a) own sample: {lab} | {smpA} ({RA['n_cs']} CS) | {RA['k']} | {_f(r['est'])} | "
          f"normal with ACS: {_ci(r['nlo'], r['nhi'])}{mde} | {_ci(r['lo'], r['hi'])} | {_status(r['lo'], r['hi'])} |")
    T3LAB = (("S_rho_tau", "Spearman(ρ_f, τ_f)"), ("S_rho_sdU", "Spearman(ρ_f, SD of U over programs)"),
             ("S_rho_rhoFU", "Spearman(ρ_f, ρ(F, U)_f)"), ("mean_rhoFU", "mean ρ(F, U)_f"),
             ("mean_rho", "mean ρ_f (baseline)"), ("mean_partial", "mean ρ(F, Y | U)_f"),
             ("diff_partial_minus_rho", "mean partial − mean baseline"),
             ("S_rho_partial", "Spearman(ρ_f, ρ(F, Y | U)_f)"),
             ("S_sdU_nacs", "Spearman(SD of U, ACS persons of the category) [revision, post hoc]"),
             ("S_sdU_tau", "Spearman(SD of U, τ_f) [revision, post hoc]"),
             ("S_rho_nacs", "Spearman(ρ_f, ACS persons) [revision, post hoc]"),
             ("P_rho_sdU_nacs", "rank-partial Spearman(ρ_f, SD of U | ACS persons) [revision, post hoc]"))
    for nm, lab in T3LAB:
        r = t3[nm]
        A(f"| T3 (exploratory): {lab} | all matched programs, fields ≥ {NMIN} programs | {nT3} | {_f(r['est'], 3)} | "
          f"{_ci(r['rlo'], r['rhi'], 3)}{' Fisher-z' if r['corr'] else ''} (MDE {r['mde']:.3f}) | "
          f"{_ci(r['inst_lo'], r['inst_hi'], 3)} (fields fixed) | {_rst(r)} |")
    A(f"| T3: fields with τ_f = 0 | fields ≥ {NMIN} programs | {nT3} | {S['n_tau0']} | | | |")
    for lab, nm in (("FE0_b", "(a)"), ("before", "main (T1 before)"), ("after", "main + U (T1 after)"),
                    ("E_met", "main + metro U"), ("E_pf", "main + field-specific δ_f U")):
        rz = R["rz"][lab]
        A(f"| SD of residualised z(F) within institution, {nm} | {smp} | {k} | {rz['sd_resid']:.3f} "
          f"({rz['sd_resid_rank']:.1f} rank places; CS {rz['sd_resid_cs']:.3f}) | | | |")
    for kk, lab in (("sd_prog", "SD of U over programs"), ("sd_within", "SD of U within institution"),
                    ("sd_resid", "SD of U net of all other regressors"),
                    ("r_within", "within-institution corr(z(F), U)"), ("r_resid", "corr(z(F), U) net of other regressors"),
                    ("sd_prog_cs", "SD of U over CS programs"), ("r_within_cs", "within-institution corr(z(F), U), CS"),
                    ("r_resid_cs", "corr(z(F), U) net of other regressors, CS"),
                    ("pi", "π: slope of U on z(F) net of other regressors (Δβ = −δπ)")):
        A(f"| {lab} (descriptive) | {smp} | {k} | {_f(ud[kk], 4 if kk == 'pi' else 3, kk.startswith('r_') or kk == 'pi')} | | | |")
    A("")
    A("Analytic cross-check (Cameron–Gelbach–Miller sandwich: CR1 by field + CR1 by institution − HC1): SE of β before "
      f"{R['cgm']['before']['se']:.4f}, after {R['cgm']['after']['se']:.4f}, Δ {R['cgm']['delta']['se']:.4f}; the "
      f"replicate two-way SEs (Scorecard part only) are {b0['se_tw']:.4f}, {b1['se_tw']:.4f}, {d1['se_tw']:.4f}; the "
      f"ACS replicate variance adds {np.sqrt(b1['vacs']):.4f} (after) and {np.sqrt(d1['vacs']):.4f} (Δ) in SE units "
      f"(θ0-centred, first version: {np.sqrt(b1['vacs_th0']):.4f} and {np.sqrt(d1['vacs_th0']):.4f}). For δ the "
      f"Scorecard two-way SE is {dl['se_tw']:.4f} and the ACS SE {np.sqrt(dl['vacs']):.4f} (first version "
      f"{np.sqrt(dl['vacs_th0']):.4f}). Institution-only CR1 SEs (scripts/55/58 standard): before "
      f"{R['FEF']['before']['se']:.4f}, after {R['FEF']['after']['se']:.4f}.")
    A("")

    # ---- Method ----
    A("## Method")
    A("")
    A(f"- **ACS sample.** 2023 1-year PUMS person files (psam_pusa/psam_pusb, read in {CHUNK:,}-row chunks, needed "
      f"columns only): {int(tot['all']):,} persons → ages {AGE_MIN}–{AGE_MAX} {int(tot['age']):,} → bachelor's highest "
      f"(SCHL 21) {int(tot['ba']):,} → full-time full-year (WKHP ≥ {FT_HOURS}, WKWN ≥ {FY_WEEKS}) {int(tot['ftfy']):,} → "
      f"civilian employed (ESR 1–2) {int(tot['employed']):,} → not enrolled (SCH 1) {int(tot['not_enrolled']):,} → "
      f"PERNP > 0 {int(tot['pos_earn']):,} → implied hourly earnings ≥ ${HOURLY_FLOOR:.3f} {int(tot['floor']):,}. "
      f"{R['acs_n_mapped']:,} of them have a first field of degree in one of the {len(set(R['f2g'].values()))} ACS "
      f"categories mapped to project fields. y = log(PERNP × ADJINC / 10⁶), 2023 dollars.")
    unm = R["unmapped_all"]
    A(f"- **Field map.** src/crosswalks fod1p_by_key_all() plus this script's map for {len(RECOVER_FOD1P)} of the "
      f"fields recovered by scripts/24 and two fields that share a category (biostatistics → statistics' 3702, "
      f"religious studies → philosophy's 4801). Kinds over the {len(R['fmap'])} mapped fields: "
      + ", ".join(f"{kk} {sum(1 for v in R['kind'].values() if v == kk)}" for kk in ("exclusive", "coarse", "shared"))
      + ". Fields sharing a category get the same U. Fields left unmapped (programs in the matched cells): "
      + (", ".join(f"{LAB.get(f, f)} ({v})" for f, v in unm.items()) if unm else "none")
      + ("; none has ≥ " + str(NMIN) + " programs" if all(v < NMIN for v in unm.values()) else "")
      + "; the T1 programs are exactly scripts/58's (asserted).")
    A(f"- **Empirical Bayes.** Per ACS category × state cell: weighted mean m, sampling variance v = deff · σ²_g / n_eff "
      f"(pooled within-cell variance of the category; Kish effective n). deff = median ratio of the 80-replicate SDR "
      f"variance to the model variance over cells with n ≥ {DEFF_NMIN}: {R['deffs']['state']:.3f} (IQR "
      f"{info_s['deff_q'][0]:.2f}–{info_s['deff_q'][1]:.2f}, {info_s['deff_cells']} cells). The ratio is right-skewed, "
      f"so the median is below the mean ratio ({info_s['deff_mean']:.3f}) and the ratio of sums "
      f"({info_s['deff_sum']:.3f}); the median was the pre-specified choice and is kept, which makes v about "
      f"{1 - R['deffs']['state'] / info_s['deff_sum']:.0%} smaller than with the ratio of sums (slightly weaker "
      f"shrinkage); the ratio-of-sums version is sensitivity E_dsum (answer 6). Additive fit m = a_g + b_s + u "
      f"weighted by 1/(v + τ²); τ² by Paule–Mandel ({info_s['df']} df): τ = {np.sqrt(info_s['tau2']):.4f}. "
      f"U = τ²/(τ² + v) · u; 0 in cells without persons. State of work (POWSP): deff {R['deffs']['pow']:.3f}, "
      f"τ = {np.sqrt(R['info_p']['tau2']):.4f}. Metro: PUMAs assigned to 2020 MSAs by majority of tracts "
      f"({R['metro_info']['n_puma_metro']:,} of {R['metro_info']['n_puma']:,} PUMAs; smallest majority "
      f"{R['metro_info']['puma_share_min']:.2f}); {R['n_metro_units']} metro units and "
      f"{R['n_units'] - R['n_metro_units']} state non-metro remainders in the metro-level fit; deff "
      f"{R['deffs']['metro']:.3f}; the metro deviation d from its principal state's U has τ_d = "
      f"{np.sqrt(R['info_m']['tau2_dev']):.4f} (Paule–Mandel, Σ d²/v = {R['info_m']['dev_q']:,.0f} on "
      f"{R['info_m']['df']:,} df). Of the {R['n_inst_all']} institutions with programs, "
      f"{R['inst_in_metro']} are in a 2020 MSA (IPEDS HD2023 county; {R['inst_cbsa_via_hd']} via the HD2023 CBSA code "
      f"for Connecticut planning regions; {R['inst_no_hd']} not found in HD2023), {R['inst_metro_absent']} of them in "
      f"a metro with no ACS unit. Programs outside a metro keep their state-level U.")
    A("- **Occupation-mix index (exploratory).** For field f and state s: Σ_o π_fo · ē_os,−f, with π_fo the national "
      "occupation distribution of f's young graduates and ē_os,−f the mean log earnings of the other fields' "
      f"graduates in occupation o and state s, shrunk to the national mean with a prior of {OCC_PRIOR:.0f} persons.")
    A("- **Within-institution design.** scripts/58 fe_sample / fe_build conventions (institutions with ≥ 2 fields, "
      "fields with ≥ 15 programs, iterated; z-scores within field), imported. The main-spec β, its CR1 SE and its "
      "institution-bootstrap CI (seed [58, 3, 1]) are reproduced exactly (asserted), as are scripts/58's per-field "
      "β_f and CIs at 4 years (seed [58, 5]) and for the 4/1/5-year average (seed [58, 11]) against "
      "data/interim/selectivity_deep.csv. The after models use the same draws, so before/after differences are paired.")
    A(f"- **Two-way variance.** One specset holds every pooled model on the same programs; for each replicate "
      f"({B_REP} shared institution draws, {B_REP} independent institution draws per field, {k} leave-one-field-out "
      f"fits; one SeedSequence child per replicate) all models are refitted by weighted within-institution least "
      f"squares, so every Δ is paired. scripts/59 `twoway` combines them; the ACS replicate variance is added "
      f"(independent samples).")
    A(f"- **ACS replicate variance.** The whole EB pipeline (cell means, additive fit, Paule–Mandel τ², shrinkage; "
      f"state, state-of-work and metro levels) is re-run with each of the 80 replicate weights, U is rebuilt, and "
      f"every model that uses U is refitted. The design effect and the variance model v are held at their "
      f"main-weight values. V_acs = 4/80 Σ_r (θ_r − θ̄)² with θ̄ the mean of the 80 replicate values (revision; the "
      f"first version centred at the full-sample θ0, see Revisions). Replicate-weighted cell means carry about v/4 "
      f"of extra perturbation variance, so the replicate τ is above the main τ ({tr['tau0']:.4f}) in "
      f"{tr['n_above']} of {tr['n']} replicates (median {tr['med']:.4f}, range {tr['lo']:.4f}–{tr['hi']:.4f}); "
      f"centring at θ̄ removes that common shift. Check: the same with τ² and τ²_g held at their main-weight values "
      f"in every replicate (U linear in the replicate cell means), both centrings (Revisions table).")
    A("- **T3.** Per field (all matched programs, ≥ 15): ρ_f = Spearman(F, Y) (= gap map), ρ(F, U)_f, SD of U over "
      "the field's programs, the rank-partial ρ(F, Y | U)_f (scripts/55 partial_rank), and τ_f = √τ²_g of the field's "
      "ACS category (per-category Paule–Mandel with the global a_g, b_s; categories with < 5 cells have no τ_f). "
      "Cross-field Spearman correlations and means; two-way variance from the jackknife over fields, shared and "
      "per-field institution draws (fields fixed), plus the ACS replicate variance. Correlations get a Fisher-z "
      "interval: est ± 1.96 SE on the atanh scale with SE_z = SE / (1 − est²), back-transformed (revision). The "
      "revision's diagnostics use the ACS persons of the field's category (fixed across replicates) and the "
      "rank-partial correlation of scripts/55.")
    A("- **Pre-specification and build history.** T1 and T2 (and T3 as exploratory) were fixed on 2026-09-24 in the "
      "script docstring, before any estimate. That date is self-reported: the script is untracked in git (lanes do "
      "not commit), so no version-control record backs it. The docstring's build note lists implementation "
      "decisions taken while finishing the interrupted build (EB on ACS categories, mapping kinds, metro rule, "
      "interval roles). That build then ran once with 19 replicates for debugging and was interrupted; its outputs "
      "were not used. The verification note lists the changes made afterwards, without reading those outputs: a bug "
      "fix at the metro level (the state non-metro remainders had been dropped from the metro-level fit) and the "
      "exploratory additions marked as such (metro-level CS remainder, the descriptive statistics of U, spec (a) on "
      "its own sample). The slope π with its omitted-variable identity, and spec (a) on its own sample, were added "
      "after this session's own 19-replicate development run (which showed that the main-spec sample holds few "
      "California CS programs); they are exploratory for that reason. A full run was then read, and after it the "
      "Paule–Mandel df of the metro deviation was changed from cells − 1 to cells − rank (exploratory metro rows "
      "only; τ_d = 0 under either df, so no number changed). The revision after the independent verification "
      "changed the ACS variance estimator, added the labelled diagnostics and the E_dsum sensitivity, and rewrote "
      "the wording (Revisions). No test and no point estimate changed.")
    A("")
    # ---- Caveats ----
    A("## Caveats")
    A("")
    A("- **Residence, not the graduates' own labour market.** U describes the institution's state; Scorecard "
      "earnings are national and many graduates move. U is a proxy for local field demand, not the market the "
      "program's graduates actually face. The state-of-work and metro versions are exploratory checks of the same "
      "idea, not a fix.")
    A("- **U contains the programs' own graduates.** In-state ACS graduates of field f include graduates of the "
      "sample programs. If a stronger department raises in-state field earnings, U partly absorbs the department's "
      "own premium, and the after-control slope then understates the part not explained by local demand. The "
      "leave-field-out occupation index avoids the program's own field but not its occupations.")
    A("- **Composition, not demand.** Field × state earnings differences also reflect who stays in or moves into a "
      "state (sorting on ability, cost of living), so U mixes demand and composition. Nothing here separates them.")
    A("- **Replicate variance of an EB-generated regressor.** The SDR replicates perturb each cell mean by about a "
      "quarter of its sampling variance; a statistic that re-estimates τ² from ~2,500 cells turns this into a shift "
      "common to all replicates rather than noise. Centring at the replicate mean removes the shift but still treats "
      "the replicate spread of the re-estimated τ² as sampling variability; holding the shrinkage fixed gives a "
      "second version (Revisions table). Neither is an exact variance of an EB-generated regressor, and for "
      "statistics that depend on the amount of shrinkage (δ, the T3 rows built on U) the ACS part of the interval "
      "is approximate.")
    A("- **Coarse and shared categories.** ACS first field of degree is coarser than CIP for some fields (Spanish, "
      "teacher education, agricultural engineering, human development) and shared by two project fields in three "
      "cases; the exclusive-only sensitivity drops them.")
    A("- **Timing.** ACS 2023 earnings of 23–35-year-olds against Scorecard 4-year medians for earlier cohorts; U "
      "is treated as a persistent local level.")
    A("- **Metro assignment** is by majority of tracts per PUMA, so PUMAs that straddle a metro boundary are "
      "assigned whole; small metros without a majority PUMA fall into the state remainder.")
    A("- **Single-field intervals.** T2 is one field's coefficient; the two-way variance does not apply (one field "
      "cluster), so the scripts/58 institution cluster bootstrap is used, with a normal interval that adds the ACS "
      "replicate variance beside it.")
    A("- **Normal intervals.** Two-way intervals are normal (est ± 1.96 SE). Correlations (T3) use the Fisher-z "
      "interval, which stays inside ±1; the normal interval is in the CSV. Ratios (the shares of β removed) can "
      "still cross natural bounds.")
    A("- **Sample.** The main-spec programs require SAT_AVG in the 2026 Scorecard institution file, which drops "
      "institutions that report no SAT average (the University of California campuses and the University of "
      "Washington–Seattle among them), so T1/T2 speak to the California and Washington markets only through the "
      "programs that remain (Appendix C). Answer 7 repeats the check in spec (a), which keeps them but has no "
      "selectivity or brand controls.")
    A("- **Ratios.** The shares of β removed are ratios of noisy slopes; their intervals are wide and asymmetric and "
      "should be read with the Δ rows, not instead of them.")
    A("")
    # ---- Revisions ----
    A("## Revisions (2026-09-25, after an independent verification)")
    A("")
    A("An independent verifier re-derived the first full run (ACS extract, EB cells, T1/T2/T3 point estimates and "
      "Scorecard-side SEs) and raised seven points. Point estimates did not change; what changed is listed below, "
      "with every number printed by this script.")
    A("")
    rd, rD, rb = rev["delta_U"], rev["Delta_T1"], rev["beta_after"]
    A(f"1. **ACS replicate variance (major).** The first version computed V_acs = 4/80 Σ_r (θ_r − θ0)². Every "
      f"replicate re-runs Paule–Mandel with v held at its main-weight value, while the replicate-weighted cell means "
      f"carry about v/4 of extra perturbation variance, so the replicate τ exceeds the main τ in {tr['n_above']} of "
      f"{tr['n']} replicates (mean τ² shift {tr['mean_shift_tau2']:+.4f}). Statistics that depend on the amount of "
      f"shrinkage are then shifted in every replicate (for δ the mean replicate shift is {_f(rd['full_bias'], 3)}), "
      f"and the θ0-centred form adds 4 × shift² to the variance. V_acs is now centred at the replicate mean. For δ "
      f"the ACS SE goes from {rd['full_th0']['se_acs']:.3f} to {rd['full_mean']['se_acs']:.3f} (Scorecard two-way SE "
      f"{rd['se_other']:.3f}), and the δ interval from {_ci(rd['full_th0']['lo'], rd['full_th0']['hi'], 3)} to "
      f"{_ci(rd['full_mean']['lo'], rd['full_mean']['hi'], 3)} ({_status(rd['full_mean']['lo'], rd['full_mean']['hi'])}); "
      f"with the shrinkage held fixed it is {_ci(rd['fix_mean']['lo'], rd['fix_mean']['hi'], 3)} (replicate-mean) "
      f"or {_ci(rd['fix_th0']['lo'], rd['fix_th0']['hi'], 3)} (θ0-centred). For β_after and Δ the change is "
      f"negligible (ACS SE of Δ {rD['full_th0']['se_acs']:.6f} → {rD['full_mean']['se_acs']:.6f}), so the T1 Δ and "
      f"its MDE stand. The δ statements in the Answer and Key numbers were rewritten from the new intervals.")
    A("")
    A("| statistic | estimate | Scorecard SE | ACS SE, θ0-centred (first version) | ACS SE, replicate-mean (now) | replicate shift | ACS SE, shrinkage fixed: θ0 / mean | 95% CI first version | 95% CI now |")
    A("|---|---|---|---|---|---|---|---|---|")
    RVLAB = [("delta_U", "T1 δ on U", 3), ("Delta_T1", "T1 Δβ", 5), ("beta_after", "T1 β after", 5),
             ("T2_y4", "T2 CS change, 4-yr (SE: institution bootstrap)", 5),
             ("T2_h3avg", "T2 CS change, 4/1/5-yr average (SE: institution bootstrap)", 5)] + \
            [(f"T3_{nm}", f"T3 {lab}", 3) for nm, lab in T3LAB]
    added = {f"T3_{nm}" for nm in ("S_sdU_nacs", "S_sdU_tau", "S_rho_nacs", "P_rho_sdU_nacs")}
    for key, lab, nd in RVLAB:
        x = rev[key]
        first = "n/a (added in revision)" if key in added else _ci(x['full_th0']['nlo'], x['full_th0']['nhi'], nd)
        A(f"| {lab} | {_f(x['est'], nd)} | {x['se_other']:.{nd + 1}f} | {x['full_th0']['se_acs']:.{nd + 1}f} | "
          f"{x['full_mean']['se_acs']:.{nd + 1}f} | {_f(x['full_bias'], nd + 1)} | {x['fix_th0']['se_acs']:.{nd + 1}f} / "
          f"{x['fix_mean']['se_acs']:.{nd + 1}f} | {first} | {_ci(x['full_mean']['lo'], x['full_mean']['hi'], nd)} |")
    A("")
    A("T2 CIs in this table are normal intervals with the institution-bootstrap SD plus V_acs; the primary T2 interval "
      "is the bootstrap percentile interval, which has no ACS part. \"First version\" intervals are normal, as "
      "reported then; \"now\" intervals are Fisher-z for correlations (T3 Spearman rows) and normal otherwise.")
    A("")
    ch = [f"{lab} ({_status(t3[nm]['lo_th0'], t3[nm]['hi_th0'])} → {_rst(t3[nm])})" for nm, lab in T3LAB[:8]
          if _status(t3[nm]["lo_th0"], t3[nm]["hi_th0"]) != _rst(t3[nm])]
    A(f"2. **T3 verdict (major).** The first version's report summarised T3 as a null (no cross-field association "
      f"with an interval excluding 0). That was wrong as worded: the ordering row Spearman(ρ_f, partial ρ_f) always excluded 0, "
      f"Spearman(ρ_f, τ_f) has an MDE of {r_tau['mde']:.2f} and is uninformative rather than null, and the SD of the "
      f"shrunk U depends on the shrinkage. With the revised V_acs the status of "
      + (", ".join(ch) if ch else "no original T3 row")
      + f" changes. Answer 5 now reports the three parts separately and adds four diagnostics, labelled post hoc: "
      f"Spearman(SD of U, ACS persons) {_f(r_n['est'], 2)} {_rci(r_n)}, Spearman(SD of U, τ_f) {_f(r_nt['est'], 2)} "
      f"{_rci(r_nt)}, Spearman(ρ_f, ACS persons) {_f(r_rn['est'], 2)} {_rci(r_rn)}, and the rank-partial "
      f"Spearman(ρ_f, SD of U | ACS persons) {_f(r_pn['est'], 2)} {_rci(r_pn)}. T3 stays exploratory, with no "
      f"multiplicity correction.")
    A(f"3. **T2 wording (minor).** \"No effect\" is replaced by a bounded change: {_f(t4['diff']['est'])} "
      f"{_ci(t4['diff']['lo'], t4['diff']['hi'])}, at most about {_pct(t4['diff']['bound_share'])} of β_f (MDE "
      f"{t4['diff']['mde']:.4f}), with the sample's {n_ca} California / {n_wa} Washington CS programs and the "
      f"exploratory spec (a) change {_f(cad['est'])} {_ci(cad['lo'], cad['hi'])} stated beside it.")
    A(f"4. **Correlation intervals (minor).** The first version's normal interval for Spearman(ρ_f, partial ρ_f) "
      f"crossed +1 ({_ci(r_part['lo_th0'], r_part['hi_th0'], 3)}). Correlation rows now use Fisher-z intervals "
      f"({_ci(r_part['rlo'], r_part['rhi'], 3)} for that row; institution-bootstrap percentile "
      f"{_ci(r_part['inst_lo'], r_part['inst_hi'], 3)} beside it).")
    A("5. **Build history (minor).** The Method now says that a full run was read before the metro Paule–Mandel df "
      "was changed (exploratory rows; no numeric consequence).")
    A("6. **Pre-specification date (minor).** The Method now says the 2026-09-24 date is self-reported and not "
      "backed by version control (the script is untracked; lanes do not commit).")
    A(f"7. **Design effect (minor).** The median of the right-skewed replicate/model variance ratio "
      f"({R['deffs']['state']:.3f}) is below the mean ratio ({info_s['deff_mean']:.3f}) and the ratio of sums "
      f"({info_s['deff_sum']:.3f}). The pre-specified median is kept; the Method states the choice, and sensitivity "
      f"E_dsum rebuilds U with the ratio of sums (τ {np.sqrt(R['info_dsum']['tau2']):.4f} instead of "
      f"{np.sqrt(info_s['tau2']):.4f}): β after {_f(S['beta']['E_dsum']['est'])}, Δ {_f(S['pair']['E_dsum']['est'])} "
      f"{_ci(S['pair']['E_dsum']['lo'], S['pair']['E_dsum']['hi'])}, δ {_f(dd['est'], 3)} {_ci(dd['lo'], dd['hi'], 3)}.")
    A("")
    # ---- Appendix ----
    A("## Appendix A — T3 per-field values (exploratory)")
    A("")
    A("| field | kind | ACS persons | programs | ρ_f | τ_f | SD of U | ρ(F, U) | ρ(F, Y \\| U) |")
    A("|---|---|---|---|---|---|---|---|---|")
    FT = S["T3_fields"]
    cnt_f = R["cells"].groupby("field").size()
    for f in FT.sort_values(["rho"], ascending=False, kind="mergesort").index:
        r = FT.loc[f]
        A(f"| {LAB.get(f, f)} | {R['kind'][f]} | {R['acs_n_group'][R['f2g'][f]]:,} | {int(cnt_f[f])} | "
          f"{_f(r['rho'], 3)} | {_f(r['tau'], 3, False)} | {_f(r['sd_U'], 3, False)} | {_f(r['rho_FU'], 3)} | "
          f"{_f(r['partial'], 3)} |")
    A("")
    A("## Appendix B — ACS extraction")
    A("")
    A("| file | bytes | md5 | persons | ages 23–35 | BA highest | FTFY | employed | not enrolled | PERNP > 0 | hourly floor |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in cnt.iterrows():
        A(f"| {r['file']} | {int(r['bytes']):,} | `{r['md5']}` | {int(r['all']):,} | {int(r['age']):,} | {int(r['ba']):,} | "
          f"{int(r['ftfy']):,} | {int(r['employed']):,} | {int(r['not_enrolled']):,} | {int(r['pos_earn']):,} | "
          f"{int(r['floor']):,} |")
    A("")
    A("## Appendix C — computer-science U by state (T1 programs; descriptive)")
    A("")
    A("| state | U (log points) | CS programs | ACS persons in the CS × state cell |")
    A("|---|---|---|---|")
    for s, r in cst.iterrows():
        A(f"| {s} | {_f(r['U'], 3)} | {int(r['n_programs'])} | {int(r['acs_n'])} |")
    A("")
    A("## Provenance")
    A("")
    A(f"Downloaded for this script (data/raw/SOURCES.md, subsection \"{SOURCES_SECTION}\"); md5 asserted at run time:")
    A("")
    for p, h in PROV.items():
        A(f"- `{p.relative_to(ROOT)}` ← {PROV_URL[p]} ({p.stat().st_size:,} bytes, md5 `{h}`)")
    A("")
    A("Already on disk: ACS 2023 PUMS (SOURCES.md §9), Scorecard FoS / institution files (§2, §9), Wapman ranks (§1).")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)   # 4 workers peaked at ~840 MB PSS (lane cap ~800 MB)
    a = ap.parse_args()
    workers = max(1, min(6, a.workers))
    avail = mem_available_mb()
    if np.isfinite(avail) and avail < MEM_MIN_MB:
        workers = 1
    stage(f"workers {workers} (MemAvailable {avail:.0f} MB)")
    R = compute(workers)
    S = summarize(R)
    write_outputs(R, S)
    stage("done")


if __name__ == "__main__":
    main()
