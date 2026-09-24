"""Cell-level hierarchical model of prestige-earnings coupling, and power / equivalence of the
project's field-level nulls.

WHY. Field-level tests in this project run on 14-20 reliable fields (up to ~50 with noisy ones), so a
null there may only mean "too few fields". Two questions:
  (a) Does the licensing / "pay set by setting" moderation of coupling survive when the model uses
      every institution x field cell, lets the prestige slope vary by field, adds institution random
      effects, and controls institution selectivity with field-specific slopes?
  (b) Which of the field-level nulls are informative (rule out a moderate effect) and which are
      underpowered?

(a) CELL-LEVEL HIERARCHICAL MODEL. Cells = (institution, field) with Wapman field prestige
    (-published field Rank, via src.load_ar) and College Scorecard FoS bachelor's EARN_MDN_4YR
    (src.load_er), FIELDS66 universe (scripts/28), fields with >= NMIN=15 institutions (the same 52
    fields and 3,444 cells as the expanded66 gap map; asserted). Within each field, earnings and
    prestige are turned into standardized normal scores (van der Waerden), so the within-field OLS
    slope of y on P is the normal-scores correlation, i.e. approximately the field's Spearman rho_f.
        y_if = a_f + (b + g * M_f + u_f) P_if + [controls] + v_i + e_if
    u_f ~ N(0, tau_P^2) random field slopes; v_i ~ N(0, s_I^2) institution random intercepts (crossed);
    a_f field fixed intercepts. M_f = field-level moderator:
        lic         ex-ante licensed field (src.crosswalks.fields.LICENSED_FIELDS: nursing,
                    communication disorders, accounting, civil engineering)
        acs_strict  ACS PUMS 2023 share of a field's employed graduates in near-universally licensed
                    occupations (data/interim/mixture_anchors.csv, built by scripts/20; its primary
                    anchor); acs_broad = the broad list (adds accountants, engineers, health techs,
                    counselors/social workers).
    Specs (normal-score scale unless stated):
        A1 raw, all fields                       A2 A1 + institution random intercepts
        A3 raw on the selectivity sample         A4 A3 + field-specific slopes on institution
                                                    SAT_AVG, -ADM_RATE, PCTPELL, state earnings
                                                    level, private control (scripts/55 "broad")
        A4B A4 + field-specific slope on academia-wide brand G (-Wapman academia rank)
        A4E A4B + field-specific slope on institution-wide earnings (Scorecard MD_EARN_WNE_P10)
        A5 A4 + institution random intercepts    A6 A4 + institution fixed effects
        A7 status model: random field slopes on P, SAT_AVG and brand G, each interacted with M_f;
           other controls field-specific
        L1/L2/L3/L4/L4B/L5 = A1/A2/A3/A4/A4B/A5 with log earnings (within-field demeaned) and z-scored
           prestige (log points per SD; comparable to the within-institution design of scripts/55);
        C1L/C2L = C1/C2 on log earnings
    Institution selectivity as a cell-level moderator of the prestige slope: P x SAT added to A4/A5.
    Estimation: REML with crossed independent variance components (own implementation, profiled
    REML via a q x q Cholesky; validated against statsmodels MixedLM on A1), optimized over the
    variance ratios theta^2 with five starts and a boundary scan (a theta-parametrized optimizer stops
    at spurious theta = 0 points because d dev / d theta = 0 there; audited in reml_checks).
    Heterogeneity: tau_P with a profile-likelihood 95% CI and a boundary LRT (0.5 chi2_0 + 0.5 chi2_1).
    Inference on the cross-level interaction g: model-based Wald SE AND a field-label permutation test
    (moderator values permuted across fields; statistic = GLS t of g with the variance parameters fixed
    at the REML estimates of the model WITHOUT the moderator, which do not depend on the labels, so the
    test is exact under exchangeability of fields; it is robust to institution dependence across
    fields). Raw vs controlled g on the same sample: field-cluster bootstrap (fields resampled; the
    relevant interval for a field-level moderator) and institution bootstrap (institutions resampled
    within the realized fields, i.e. conditional on them; used for differences only); variance
    parameters fixed at each spec's REML estimates.
    Field-level comparison on the same fields: (i) two-step (per-field OLS slope, then REML random-
    effects meta-regression with Knapp-Hartung SE and a field-label permutation p); (ii) scripts/20-
    style precision-weighted WLS of the gap-map rho_f on the moderator (se floor 0.08 as in the file);
    (iii) the published scripts/20 value (README Result 2: gap b_lic = +0.66 [+0.33, +0.97], i.e.
    coupling -0.66 per unit strict share, with academic absorption in the model).
    Leverage of the ACS shares (leverage_robustness): nursing (~0.77) and special education (~0.60) lie
    far above the other fields (<= 0.37), so the shares also enter as winsorized, square-root, log and
    rank-normal-score moderators (per SD), as a field-level Spearman of per-field slopes with the share,
    and with both fields dropped; leave-one-field-out also for A4B and L4B. The broad share gets the same checks.
    Field type (field_type_checks): the strict share is lowest in every engineering / CS / statistics / business
    field, so the moderation is re-tested with an engineering/CS/math/business dummy (CIP-2 11/14/27/52) and other
    field-level controls as second cross-level moderators (Kennedy residual permutation), with CIP-2 fixed slopes and
    the share permuted within CIP-2, and within the non-bloc and the bloc fields separately.

(b) POWER AND EQUIVALENCE of the field-level nulls (alpha = 0.05; 80% power; moderate effect =
    true-score correlation 0.30, large = 0.50, in the direction each theory predicts):
      B1 within-occupation dispersion law (scripts/53): Spearman(tail_hi, rho_f), reliable 16 / all 50
         fields (data/interim/dispersion_law_fields.csv); plus all six dispersion measures.
      B2 occupation concentration (scripts/50): Spearman(occ_hhi, rho_f), reliable 16 / all 50
         (data/interim/occupation_channel_fields.csv); theory predicts a NEGATIVE correlation.
      B3 kinship Moran's I (scripts/51): flow-kinship W on the 14 reliable fields in the faculty-flow
         network (functions imported from scripts/51); effect = SAR parameter lambda, reported as the
         implied correlation between a field's true coupling and its kin-weighted neighbours'.
      B4 integrated-minus-others career-time slope (scripts/52; data/interim/career_time_coupling.csv):
         moderate = half the all-field slope, large = the whole all-field slope ("others flat").
    B1/B2: rho_f is measured with error (se from the gap-map bootstrap CI). Power and confidence bounds
    are simulated on the TRUE-score scale: latent coupling correlated r with the moderator (Gaussian
    copula on the moderator's normal scores), observed = latent + N(0, se_f^2); the one-sided 95%
    bound is found by inverting P(observed Spearman <= s_obs | r). The analytic observed-score version
    (Fisher z, var 1.06/(n-3)) is reported next to it. B3: SAR(lambda) latent + measurement noise;
    power of the right-tail permutation test used by scripts/51; bound by inverting P(I <= I_obs |
    lambda). B4: SE from scripts/52's two-stage bootstrap CI (normal approximation), plus simulated
    power of its field-label permutation test. The same checks are run on the cell-level versions of
    B1/B2 (moderator x prestige in A1/A4, per SD of the moderator; r-scale = g / tau_P of the model
    without the moderator).
    Classification: "informative" if the one-sided 95% bound in the theory's direction excludes the
    moderate effect; "rules out large only" if it excludes the large but not the moderate effect;
    "underpowered" otherwise.

Public data only. Descriptive, not causal. Seeded (SEED=63; one independent stream per analysis);
outputs byte-identical on re-run.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/63_hierarchical_power.py
Outputs: data/interim/hierarchical_power.csv, outputs/figures/hierarchical_power.png,
         HIERARCHICAL_POWER_RESULT.md (root; gitignored).
"""
from __future__ import annotations

import os
# single-threaded BLAS: faster for the many small dense solves here on a shared machine, and the
# optimizer path (hence every printed digit) does not depend on the thread count
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
import zlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import solve_triangular, cho_solve, qr
from scipy.optimize import minimize, brentq, minimize_scalar
from scipy.stats import rankdata, norm, chi2, spearmanr, t as tdist
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_ar import load_ar_wapman
from src.load_er import load_er_scorecard
from src.crosswalks import fields as F

SEED = 63
NMIN = 15              # min institutions per field (as scripts/55 and the gap map)
DFMIN = 10             # min residual df for the field-specific controls (as scripts/55)
N_PERM = 1000          # field-label permutations (cross-level interactions); replicate cap 1000
N_BOOT = 1000          # joint institution bootstrap draws (raw vs controlled)
N_SIM = 1000           # simulations per grid point (B1/B2 power and bound inversion; B3 inversion)
N_SIM_MORAN = 1000     # simulations per lambda for the Moran permutation-test power
N_PERM_MORAN = 499     # permutations inside each Moran power simulation
N_SIM_CT = 1000        # simulations per delta, career-time permutation-test power
N_PERM_OBS = 1000      # permutations for the observed Moran's I p-value (scripts/51 used 10000)
N_PERM_CT = 999        # permutations inside each career-time power simulation
MODERATE, LARGE = 0.30, 0.50
Z80 = norm.ppf(0.80)
PSI_STARTS = (0.0004, 0.0025, 0.01, 0.04, 1.0)      # REML starts, variance ratios (theta 0.02 ... 1)
PSI_SCAN = (1e-4, 1e-3, 4e-3, 0.016, 0.064)         # boundary scan (theta 0.01 ... 0.25)
INST_FILE = ROOT / "data" / "raw" / "scorecard_inst" / "Most-Recent-Cohorts-Institution.csv"
GAP_MAP = ROOT / "outputs" / "expanded66_gap_map.csv"
ANCHORS = ROOT / "data" / "interim" / "mixture_anchors.csv"
DISP = ROOT / "data" / "interim" / "dispersion_law_fields.csv"
OCC = ROOT / "data" / "interim" / "occupation_channel_fields.csv"
CAREER = ROOT / "data" / "interim" / "career_time_coupling.csv"
OUT_CSV = ROOT / "data" / "interim" / "hierarchical_power.csv"
OUT_FIG = ROOT / "outputs" / "figures" / "hierarchical_power.png"
OUT_MD = ROOT / "HIERARCHICAL_POWER_RESULT.md"
PUBLISHED_LIC = dict(b_gap=0.66, lo=0.33, hi=0.97)   # README.md Result 2 (scripts/20), gap units
# figure colours: dataviz reference categorical slots 1-4 (validated) + reserved status colours
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
S_GOOD, S_WARN, S_CRIT = "#0ca30c", "#fab219", "#d03b3b"

# FIELDS66 + label map + academia-wide brand loader, exactly as scripts/52 and 55 obtain them
_spec = importlib.util.spec_from_file_location("s28", ROOT / "scripts" / "28_field_vs_generic_prestige.py")
_s28 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s28)
FIELDS66, LAB = _s28.FIELDS66, _s28.LAB
# Moran's I helpers of scripts/51 (reused, not re-implemented)
_spec = importlib.util.spec_from_file_location("s51", ROOT / "scripts" / "51_moran_kinship_coupling.py")
_s51 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s51)

MODS = {"lic": "ex-ante licensed (dummy)", "acs_strict": "ACS licensed-occupation share, strict",
        "acs_broad": "ACS licensed-occupation share, broad"}
# bachelor's-only versions (SCHL == 21): drop graduate-degree occupations (physicians, lawyers, ...)
MODS_BA = {"acs_strict_ba": "ACS strict licensed share, bachelor's-only holders",
           "acs_k12_ba": "ACS K-12 teacher share (SOC 25-2), bachelor's-only holders",
           "acs_health_ba": "ACS health-practitioner share (SOC 29-1), bachelor's-only holders"}
ACS_GLOB = "psam_pus*.csv"
LIC_STRICT = ("291", "231", "252", "171011", "193031")   # = scripts/20 LIC_STRICT
LIC_BROAD_EXTRA = ("292", "132011", "172", "211")         # = scripts/20 LIC_BROAD_EXTRA
BRIDGE = {"tail_hi": "dispersion index tail_hi (scripts/53), per SD",
          "occ_hhi": "occupation concentration occ_hhi (scripts/50), per SD"}
CONTROLS = ["SAT", "NADM", "PELL", "STE", "PRIV"]
# field-specific control slopes by spec type (G = academia-wide brand, IE = institution-wide earnings)
CTRL = {None: [], "field": CONTROLS, "field_g": CONTROLS + ["G"], "field_ge": CONTROLS + ["G", "IE"],
        "status": ["NADM", "PELL", "STE", "PRIV"]}
SPECS = {
    "A1": dict(sample="all", y="ns", ctrl=None, inst=None, label="raw, all fields"),
    "A2": dict(sample="all", y="ns", ctrl=None, inst="re", label="raw + institution RE"),
    "A3": dict(sample="sat", y="ns", ctrl=None, inst=None, label="raw, selectivity sample"),
    "A4": dict(sample="sat", y="ns", ctrl="field", inst=None,
               label="selectivity-controlled (field-specific slopes)"),
    "A4B": dict(sample="sat", y="ns", ctrl="field_g", inst=None,
                label="A4 + field-specific brand slope"),
    "A4E": dict(sample="sat", y="ns", ctrl="field_ge", inst=None,
                label="A4B + field-specific institution-wide earnings slope"),
    "A5": dict(sample="sat", y="ns", ctrl="field", inst="re", label="A4 + institution RE"),
    "A6": dict(sample="sat", y="ns", ctrl="field", inst="fe", label="A4 + institution FE"),
    "A7": dict(sample="sat", y="ns", ctrl="status", inst=None, slopes=["P", "SAT", "G"],
               label="status model: P, SAT, brand random slopes"),
    "C1": dict(sample="sat", y="ns", ctrl=None, inst=None, slopes=["SAT"],
               label="selectivity pricing: earnings ~ SAT_AVG only"),
    "C2": dict(sample="sat", y="ns", ctrl=None, inst=None, slopes=["G"],
               label="brand pricing: earnings ~ academia-wide rank only"),
    "L1": dict(sample="all", y="log", ctrl=None, inst=None, label="log earnings, raw"),
    "L2": dict(sample="all", y="log", ctrl=None, inst="re", label="log earnings, raw + institution RE"),
    "L3": dict(sample="sat", y="log", ctrl=None, inst=None, label="log earnings, raw, selectivity sample"),
    "L4": dict(sample="sat", y="log", ctrl="field", inst=None, label="log earnings, selectivity-controlled"),
    "L4B": dict(sample="sat", y="log", ctrl="field_g", inst=None,
                label="log earnings, selectivity + field-specific brand slope"),
    "L5": dict(sample="sat", y="log", ctrl="field", inst="re",
               label="log earnings, selectivity-controlled + institution RE"),
    "C1L": dict(sample="sat", y="log", ctrl=None, inst=None, slopes=["SAT"],
                label="log earnings ~ SAT_AVG only"),
    "C2L": dict(sample="sat", y="log", ctrl=None, inst=None, slopes=["G"],
                label="log earnings ~ academia-wide rank only"),
}
for _k, _v in SPECS.items():
    _v.setdefault("slopes", ["P"])
# specs that also get the bachelor's-only shares
BA_SPECS = ("A1", "A3", "A4", "A4B", "A4E", "A5", "A7", "C1", "C2", "L3", "L4", "L4B", "L5", "C1L", "C2L")
BOOT_SPECS = ("A3", "A4", "A4B", "A4E", "A5")          # raw vs controlled, bootstrapped (A3 first)
LOFO = [("A1", "acs_strict"), ("A4", "acs_strict"), ("A1", "acs_strict_ba"), ("A4", "acs_strict_ba"),
        ("A4B", "acs_strict"), ("A4B", "acs_strict_ba"), ("L4B", "acs_strict"), ("L4B", "acs_strict_ba")]
DROP_SET = ["nursing", "communication_disorders"]
# leverage checks for the two ACS shares: nursing and special education lie far above the other fields
LEV_MODS = ("acs_strict", "acs_strict_ba", "acs_broad")   # the broad share was added in the field-type revision
STRICT2 = ("acs_strict", "acs_strict_ba")                    # the two strict-list shares
LEV_SPECS = ("A3", "A4", "A4B", "A4E", "A7", "L3", "L4", "L4B", "L5", "C1", "C2", "C1L", "C2L")
SP_SPECS = ("A3", "A4", "A4B", "A4E", "L3", "L4", "L4B", "L5")
DROP2 = ["nursing", "special_education"]
WINS = 0.30
# ---- field-type check (field_type_checks). The strict share is lowest in every engineering, computer-science,
# mathematics/statistics and business field, so its moderation of the prestige slope could be a field-type contrast
# rather than a dose-response in licensing. CIP-2 of each field as in scripts/28 / 57 (first cip4 of FIELDS66).
CIP2 = _s28.CIP2
BLOC = ("11", "14", "27", "52")          # computer/info, engineering, math & statistics, business
FT_SETS = {                               # name -> label; dummies are built from CIP-2 sets
    "bloc": "engineering/CS/math/business dummy (CIP-2 11, 14, 27, 52)",
    "bloc_nomath": "engineering/CS/business dummy (CIP-2 11, 14, 52)",
    "integrated": "README 'integrated' clusters dummy (CIP-2 11, 14, 27, 45)",
    "integrated_bus": "'integrated' clusters + business dummy (CIP-2 11, 14, 27, 45, 52)",
    "log_earn": "log median Scorecard earnings of the field's cells, per SD",
    "log_n": "log institutions per field (gap map; scripts/57 field size), per SD",
    "absorption": "ACS academic absorption share (scripts/20), per SD",
    "prestige_range": "log range of published Wapman ranks across the field's cells, per SD",
    "cip2_fe": "CIP-2 fixed slopes; share permuted within CIP-2 (stratified)",
}
FT_DUMMY = {"bloc": BLOC, "bloc_nomath": ("11", "14", "52"), "integrated": ("11", "14", "27", "45"),
            "integrated_bus": ("11", "14", "27", "45", "52")}
FT_SPECS = ("A3", "A4", "A4B", "A5", "A7", "L4")   # full ladder for none / bloc / cip2_fe and the subsets
FT_SPECS_SHORT = ("A4", "A4B")                  # the other covariate sets
FT_FORMS = ("linear", "square root", "rank normal score")


def rng_for(tag: str) -> np.random.Generator:
    return np.random.default_rng([SEED, zlib.crc32(tag.encode())])


# =============================================================================================
# data
# =============================================================================================
def load_institutions() -> pd.DataFrame:
    """Scorecard institution covariates + leave-one-out state earnings level (mirrors scripts/55)."""
    cols = ["UNITID", "INSTNM", "STABBR", "CONTROL", "PREDDEG", "ADM_RATE", "SAT_AVG", "PCTPELL",
            "MD_EARN_WNE_P10"]
    d = pd.read_csv(INST_FILE, usecols=cols, dtype=str)
    for c in ["ADM_RATE", "SAT_AVG", "PCTPELL", "MD_EARN_WNE_P10", "CONTROL", "PREDDEG"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["_le"] = np.where((d.PREDDEG == 3) & (d.MD_EARN_WNE_P10 > 0), np.log(d.MD_EARN_WNE_P10), np.nan)
    d["_ok"] = d["_le"].notna().astype(float)
    s = d.groupby("STABBR")["_le"].transform(lambda x: x.fillna(0).sum())
    k = d.groupby("STABBR")["_ok"].transform("sum")
    denom = k - d["_ok"]
    d["ST_EARN"] = np.where(denom > 0, (s - d["_le"].fillna(0.0)) / denom.where(denom > 0, 1), np.nan)
    return d.drop(columns=["_le", "_ok"])


def build_cells() -> pd.DataFrame:
    ar = load_ar_wapman(fields=FIELDS66)
    er = load_er_scorecard("undergrad", fields=FIELDS66)
    m = ar[["inst_key", "field", "prestige_score"]].merge(
        er[["inst_key", "field", "earnings", "cohort_n", "institution_id"]],
        on=["inst_key", "field"], how="inner").dropna(subset=["prestige_score", "earnings"])
    m = m[m.earnings > 0].rename(columns={"institution_id": "UNITID"})
    m["UNITID"] = m["UNITID"].astype(str)
    m = m.merge(load_institutions(), on="UNITID", how="left")
    m = m.merge(_s28.load_generic(), on="inst_key", how="left")
    m["G"] = -m["g_rank"]
    m["NEG_ADM"] = -m["ADM_RATE"]
    m["PRIVATE"] = (m["CONTROL"] != 1).astype(float).where(m["CONTROL"].notna())
    return m.sort_values(["field", "inst_key"]).reset_index(drop=True)


def field_moderators(fields: list[str], acs: pd.DataFrame) -> pd.DataFrame:
    gm = pd.read_csv(GAP_MAP).set_index("field")
    an = pd.read_csv(ANCHORS).set_index("field")
    dp = pd.read_csv(DISP).set_index("field")
    oc = pd.read_csv(OCC).set_index("field")
    t = pd.DataFrame(index=pd.Index(fields, name="field"))
    t["lic"] = [float(f in F.LICENSED_FIELDS) for f in fields]
    t["acs_strict"] = an["licensure_strict"].reindex(fields)
    t["acs_broad"] = an["licensure_broad"].reindex(fields)
    t["absorption"] = an["absorption_acs"].reindex(fields)
    for c in ["acs_strict_ba", "acs_k12_ba", "acs_health_ba", "share_ba", "n_acs_ba"]:
        t[c] = acs[c].reindex(fields)
    t["tail_hi"] = dp["tail_hi"].reindex(fields)
    t["occ_hhi"] = oc["occ_hhi"].reindex(fields)
    for c in ["spearman", "se", "reliable", "n_institutions", "ci_lo", "ci_hi"]:
        t[c] = gm[c].reindex(fields)
    return t


def acs_shares() -> pd.DataFrame:
    """Licensed-occupation shares from ACS PUMS 2023 exactly as scripts/20 build_anchors (all degree
    holders, SCHL >= 21) -- asserted equal to data/interim/mixture_anchors.csv -- plus the same shares
    for bachelor's-only holders (SCHL == 21) and two components of the strict list."""
    need = ["FOD1P", "SOCP", "SCHL", "WKHP", "ESR", "PERNP", "PWGTP"]
    code_map = F.fod1p_to_field_key(F.fod1p_by_key_all())
    frames = []
    for fp in sorted((ROOT / "data" / "raw" / "acs").glob(ACS_GLOB)):
        for ch in pd.read_csv(fp, usecols=need, dtype={"SOCP": str}, chunksize=500_000):
            for c in ["FOD1P", "SCHL", "WKHP", "ESR", "PERNP", "PWGTP"]:
                ch[c] = pd.to_numeric(ch[c], errors="coerce")
            ch = ch[(ch.SCHL >= 21) & ch.FOD1P.notna() & (ch.WKHP >= 35) & ch.ESR.isin([1, 2]) & (ch.PERNP > 0)]
            frames.append(ch[["FOD1P", "SOCP", "SCHL", "PWGTP"]].copy())
    a = pd.concat(frames, ignore_index=True)
    a["field"] = a.FOD1P.astype(int).astype(str).str.zfill(4).map(code_map)
    a = a[a.field.notna()].copy()
    soc = a.SOCP.fillna("").str.replace("X", "0", regex=False)
    a["strict"] = soc.str.startswith(LIC_STRICT)
    a["broad"] = soc.str.startswith(LIC_STRICT + LIC_BROAD_EXTRA)
    a["k12"] = soc.str.startswith("252"); a["health"] = soc.str.startswith("291")
    rows = []
    for fld, g in a.groupby("field"):
        ba = g[g.SCHL == 21]
        w = lambda d, c: float(np.average(d[c], weights=d.PWGTP)) if len(d) else np.nan
        rows.append(dict(field=fld, n_acs=len(g), n_acs_ba=len(ba), strict_all=w(g, "strict"),
                         broad_all=w(g, "broad"), acs_strict_ba=w(ba, "strict"), acs_k12_ba=w(ba, "k12"),
                         acs_health_ba=w(ba, "health"), share_ba=float(ba.PWGTP.sum() / g.PWGTP.sum())))
    out = pd.DataFrame(rows).set_index("field")
    an = pd.read_csv(ANCHORS).set_index("field")
    chk = max(float((out.strict_all - an.licensure_strict.reindex(out.index)).abs().max()),
              float((out.broad_all - an.licensure_broad.reindex(out.index)).abs().max()))
    assert chk < 1e-9 and set(out.index) == set(an.index), f"ACS shares did not reproduce scripts/20 ({chk})"
    return out


def ns(s: pd.Series) -> pd.Series:
    """Within-field standardized normal scores (van der Waerden; average ranks for ties)."""
    v = s.to_numpy(float)
    ok = np.isfinite(v)
    out = np.full(len(v), np.nan)
    if ok.sum() >= 2:
        r = rankdata(v[ok])
        z = norm.ppf((r - 0.5) / ok.sum())
        sd = z.std()
        out[ok] = (z - z.mean()) / sd if sd > 0 else 0.0
    return pd.Series(out, index=s.index)


def zs(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def rank_ns(v: pd.Series) -> pd.Series:
    """Normal scores of a field-level moderator across fields (van der Waerden)."""
    return pd.Series(norm.ppf((rankdata(v.to_numpy(float)) - 0.5) / len(v)), index=v.index)


# forms of a share moderator; all but 'linear' limit the pull of the two high-share fields
FORMS = {"linear": lambda v: v,
         f"winsorized at {WINS:.2f}": lambda v: v.clip(upper=WINS),
         "square root": lambda v: np.sqrt(v),
         "log(share + 0.01)": lambda v: np.log(v + 0.01),
         "rank normal score": rank_ns}


def transform(S: pd.DataFrame, sat: bool) -> pd.DataFrame:
    S = S.copy()
    g = S.groupby("field")
    S["y_ns"] = g.earnings.transform(ns); S["P_ns"] = g.prestige_score.transform(ns)
    S["G_ns"] = g.G.transform(ns)
    S["y_log"] = np.log(S.earnings) - g.earnings.transform(lambda e: np.log(e).mean())
    S["P_log"] = g.prestige_score.transform(zs); S["G_log"] = g.G.transform(zs)
    if sat:
        for c, src in [("SAT", "SAT_AVG"), ("NADM", "NEG_ADM"), ("PELL", "PCTPELL"), ("STE", "ST_EARN")]:
            S[c + "_ns"] = g[src].transform(ns)
            S[c + "_log"] = g[src].transform(zs)
        S["PRIV_ns"] = S.PRIVATE - g.PRIVATE.transform("mean")
        S["PRIV_log"] = S["PRIV_ns"]
        # institution-wide median earnings 10 yrs after entry (all fields; MD_EARN_WNE_P10)
        S["IE_ns"] = g.MD_EARN_WNE_P10.transform(ns)
        S["IE_log"] = g.MD_EARN_WNE_P10.transform(lambda e: zs(np.log(e)))
    return S


# =============================================================================================
# REML for crossed, independent variance components
# =============================================================================================
class REML:
    """y = X b + sum_k Z_k u_k + e, u_k ~ N(0, s2 th_k^2 I), e ~ N(0, s2 I). Profiled REML; the
    marginal covariance is handled through the q x q matrix A = I + Lam Z'Z Lam."""

    def __init__(self, y, X, blocks):
        self.y = np.asarray(y, float); self.X = np.asarray(X, float)
        n = len(self.y); mats, self.kidx, start = [], [], 0
        for codes, vals in blocks:
            codes = np.asarray(codes, int); q = int(codes.max()) + 1
            mats.append(sp.csr_matrix((np.asarray(vals, float), (np.arange(n), codes)), shape=(n, q)))
            self.kidx.append(np.arange(start, start + q)); start += q
        self.Z = sp.hstack(mats).tocsr(); self.q = start; self.K = len(blocks)
        self.ZtZ = (self.Z.T @ self.Z).toarray()
        self.ZtX = np.asarray(self.Z.T @ self.X)
        self.Zty = np.asarray(self.Z.T @ self.y).ravel()
        self.XtX = self.X.T @ self.X; self.Xty = self.X.T @ self.y; self.yty = float(self.y @ self.y)
        self.n, self.p = self.X.shape

    def lam(self, th):
        l = np.empty(self.q)
        for k, idx in enumerate(self.kidx):
            l[idx] = th[k]
        return l

    def parts(self, th):
        l = self.lam(th)
        A = np.eye(self.q) + (l[:, None] * self.ZtZ) * l[None, :]
        L = np.linalg.cholesky(A)
        CX = solve_triangular(L, l[:, None] * self.ZtX, lower=True)
        cy = solve_triangular(L, l * self.Zty, lower=True)
        XVX = self.XtX - CX.T @ CX
        XVy = self.Xty - CX.T @ cy
        yVy = self.yty - cy @ cy
        Lx = np.linalg.cholesky(XVX)
        b = cho_solve((Lx, True), XVy)
        return dict(L=L, Lx=Lx, b=b, r2=float(yVy - b @ XVy), l=l)

    def dev(self, th):
        try:
            P = self.parts(th)
        except np.linalg.LinAlgError:
            return 1e30
        npp = self.n - self.p
        s2 = P["r2"] / npp
        if s2 <= 0:
            return 1e30
        return (npp * (1 + np.log(2 * np.pi * s2)) + 2 * np.log(np.diag(P["L"])).sum()
                + 2 * np.log(np.diag(P["Lx"])).sum())

    def fit(self, fixed=None, starts=PSI_STARTS):
        """REML optimum. `fixed` holds theta values (SD ratios) of components kept fixed; the free
        components are optimized over psi = theta^2 (variance ratios), bounds [0, 900].
        Why psi and not theta: the deviance depends on theta only through Lam Z'Z Lam, so d dev / d theta
        is exactly 0 at theta = 0 and a theta-parametrized L-BFGS-B that reaches the bound stops there
        even when the optimum is interior (an earlier version of this script did so, e.g. A4 x acs_strict).
        In psi the deviance is smooth at 0 (log det(I + Lam Z'Z Lam) = log det(I + Z'Z Lam^2)) with a
        non-zero slope. Several starts (psi 0.0004 ... 1, i.e. theta 0.02 ... 1), then a boundary scan: a
        free component at 0 is set to small positive values and the fit restarted if that lowers the
        deviance. `starts` entries are scalars (all free components) or arrays of psi."""
        fixed = fixed or {}
        free = [k for k in range(self.K) if k not in fixed]

        def full(t):
            th = np.zeros(self.K)
            for k, v in fixed.items():
                th[k] = v
            th[free] = np.sqrt(np.clip(t, 0.0, None))
            return th

        def obj(t):
            return self.dev(full(t))

        def run(x0):
            return minimize(obj, x0, method="L-BFGS-B", bounds=[(0.0, 900.0)] * len(free),
                            options=dict(ftol=1e-11, maxiter=1000))
        if free:
            best = None
            for s0 in starts:
                x0 = np.full(len(free), s0, float) if np.isscalar(s0) else np.asarray(s0, float)
                r = run(x0)
                if best is None or r.fun < best.fun - 1e-9:
                    best = r
            for _ in range(5):                      # boundary scan
                improved = False
                for j in range(len(free)):
                    if best.x[j] > 1e-8:
                        continue
                    for v in PSI_SCAN:
                        x1 = best.x.copy(); x1[j] = v
                        if obj(x1) < best.fun - 1e-7:
                            r = run(x1)
                            if r.fun < best.fun - 1e-9:
                                best, improved = r, True
                            break
                if not improved:
                    break
            th = full(best.x)
        else:
            th = full(np.zeros(0))
        P = self.parts(th)
        s2 = P["r2"] / (self.n - self.p)
        cov = s2 * cho_solve((P["Lx"], True), np.eye(self.p))
        return dict(theta=th, s2=s2, sd=np.sqrt(s2) * th, beta=P["b"], cov=cov, dev=self.dev(th), P=P)

    def profile(self, k, fit, level=0.95):
        """Profile-likelihood CI for the SD of component k and the boundary LRT for SD_k = 0."""
        crit = chi2.ppf(level, 1)
        cache = {}
        others = np.delete(fit["theta"], k)

        def pdev(v):
            key = round(float(v), 10)
            if key not in cache:
                cache[key] = self.fit(fixed={k: float(v)},
                                      starts=(others ** 2,) + PSI_STARTS if len(others) else (0.04,))
            return cache[key]
        th_hat = fit["theta"][k]
        d0 = pdev(0.0)["dev"] - fit["dev"]
        lrt_p = float(0.5 * chi2.sf(max(d0, 0.0), 1)) if d0 > 1e-12 else 0.5
        g = lambda v: pdev(v)["dev"] - fit["dev"] - crit
        lo = 0.0 if d0 < crit else brentq(g, 0.0, th_hat, xtol=1e-5)
        hi_b = max(th_hat, 1e-3) * 2
        while g(hi_b) < 0 and hi_b < 30:
            hi_b *= 2
        hi = brentq(g, th_hat, hi_b, xtol=1e-5)
        to_sd = lambda v: float(v * np.sqrt(pdev(v)["s2"])) if v > 0 else 0.0
        return dict(lo=to_sd(lo), hi=to_sd(hi), lrt=float(max(d0, 0.0)), lrt_p=lrt_p)

    # ---- fixed-variance GLS for added columns (permutation statistic) ----
    def base(self, th):
        P = self.parts(th)
        l, L = P["l"], P["L"]

        def vinv(M):
            M = np.asarray(M, float)
            ZtM = np.asarray(self.Z.T @ M)
            w = cho_solve((L, True), (l[:, None] * ZtM) if M.ndim == 2 else l * ZtM)
            return M - self.Z @ ((l[:, None] * w) if M.ndim == 2 else l * w)
        VX0 = vinv(self.X)
        XVX0 = self.X.T @ VX0
        Lx = np.linalg.cholesky(XVX0)
        b0 = cho_solve((Lx, True), VX0.T @ self.y)
        r0 = float(self.y @ vinv(self.y) - b0 @ (VX0.T @ self.y))
        return dict(vinv=vinv, VX0=VX0, Lx=Lx, b0=b0, r0=r0)

    def add_t(self, B, W):
        """GLS estimate, SE and t of added columns W (n x m) given the base columns, fixed variances."""
        W = np.atleast_2d(np.asarray(W, float).T).T
        VW = B["vinv"](W)
        X0VW = B["VX0"].T @ W
        S = W.T @ VW - X0VW.T @ cho_solve((B["Lx"], True), X0VW)
        rhs = VW.T @ self.y - X0VW.T @ B["b0"]
        g = np.linalg.solve(S, rhs)
        r2 = B["r0"] - g @ S @ g
        s2 = r2 / (self.n - self.p - W.shape[1])
        se = np.sqrt(s2 * np.diag(np.linalg.inv(S)))
        return g, se, g / se


def gls_fixed(y, X, blocks, th):
    """GLS with fixed variance ratios; pseudo-inverse in case a resample makes X rank deficient."""
    R = REML.__new__(REML)
    REML.__init__(R, y, X, blocks)
    l = R.lam(th)
    A = np.eye(R.q) + (l[:, None] * R.ZtZ) * l[None, :]
    L = np.linalg.cholesky(A)
    CX = solve_triangular(L, l[:, None] * R.ZtX, lower=True)
    cy = solve_triangular(L, l * R.Zty, lower=True)
    XVX = R.XtX - CX.T @ CX
    XVy = R.Xty - CX.T @ cy
    return np.linalg.pinv(XVX, rcond=1e-10) @ XVy


# =============================================================================================
# design matrices
# =============================================================================================
def drop_dependent(X, names, keep):
    """Drop all-zero and linearly dependent columns (pivoted QR); columns in `keep` must survive."""
    nz = np.abs(X).sum(0) > 1e-12
    X, names = X[:, nz], [n for n, z in zip(names, nz) if z]
    _, R, piv = qr(X, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    rank = int((d > d.max() * 1e-9).sum())
    sel = np.sort(piv[:rank])
    out = [names[i] for i in sel]
    assert all(k in out for k in keep), f"key column dropped: {set(keep) - set(out)}"
    return X[:, sel], out


def design(S: pd.DataFrame, spec: dict, mod: pd.Series | None, extra: str | None = None,
           fcov: pd.DataFrame | None = None):
    """Returns y, X0 (base), names0, W (interaction columns), wnames, blocks, field list. `fcov` = field-level
    covariates (index = field): each enters the base as covariate x every random slope (cross-level controls)."""
    suf = "_" + spec["y"]
    fields = sorted(S.field.unique())
    fc = pd.Categorical(S.field, categories=fields).codes
    k = len(fields)
    D = np.eye(k)[fc]
    y = S["y" + suf].to_numpy(float)
    cols, names = [np.ones(len(S))], ["const"]
    cols += [D[:, j] for j in range(1, k)]; names += [f"a_{f}" for f in fields[1:]]
    slopes = spec["slopes"]
    for s_ in slopes:
        cols.append(S[s_ + suf].to_numpy(float)); names.append(f"b_{s_}")
    for c in CTRL[spec["ctrl"]]:
        v = S[c + suf].to_numpy(float)
        for j, f in enumerate(fields):
            cols.append(D[:, j] * v); names.append(f"x_{c}_{f}")
    if extra == "PxSAT":
        cols.append(S["P" + suf].to_numpy(float) * S["SAT" + suf].to_numpy(float)); names.append("b_PxSAT")
    if fcov is not None:
        for c in fcov.columns:
            cv = fcov[c].reindex(fields).to_numpy(float)[fc]
            for s_ in slopes:
                cols.append(cv * S[s_ + suf].to_numpy(float)); names.append(f"h_{c}_{s_}")
    blocks = [(fc, S[s_ + suf].to_numpy(float)) for s_ in slopes]
    ic = pd.factorize(S.inst_key)[0]
    if spec["inst"] == "re":
        blocks.append((ic, np.ones(len(S))))
    elif spec["inst"] == "fe":
        Di = np.eye(ic.max() + 1)[ic]
        cols += [Di[:, j] for j in range(1, Di.shape[1])]; names += [f"i_{j}" for j in range(1, Di.shape[1])]
    X0 = np.column_stack(cols)
    keep = [f"b_{s_}" for s_ in slopes] + (["b_PxSAT"] if extra == "PxSAT" else [])
    X0, names = drop_dependent(X0, names, keep)
    W, wnames = None, []
    if mod is not None:
        mv = mod.reindex(fields).to_numpy(float)[fc]
        W = np.column_stack([mv * S[s_ + suf].to_numpy(float) for s_ in slopes])
        wnames = [f"g_{s_}" for s_ in slopes]
    return dict(y=y, X0=X0, names=names, W=W, wnames=wnames, blocks=blocks, fields=fields, fc=fc,
                slopes=slopes, n=len(S), n_inst=int(ic.max() + 1))


# =============================================================================================
# part (a): hierarchical fits
# =============================================================================================
def comp_names(spec):
    return list(spec["slopes"]) + (["inst"] if spec["inst"] == "re" else [])


def fit_null(S, spec, extra=None):
    d = design(S, spec, None, extra)
    R = REML(d["y"], d["X0"], d["blocks"])
    f = R.fit()
    return d, R, f


def fit_mod(S, spec, mod: pd.Series, null_cache: dict, key: str, rng, extra=None):
    """Full REML fit with moderator; permutation p with the null model's variance parameters."""
    d = design(S, spec, mod, extra)
    X = np.column_stack([d["X0"], d["W"]])
    R = REML(d["y"], X, d["blocks"])
    f = R.fit()
    if key not in null_cache:
        R0 = REML(d["y"], d["X0"], d["blocks"])
        null_cache[key] = (R0, R0.fit())
    R0, f0 = null_cache[key]
    B = R0.base(f0["theta"])
    _, _, t_obs = R0.add_t(B, d["W"])
    mv = mod.reindex(d["fields"]).to_numpy(float)
    slopes_v = np.column_stack([S[s_ + "_" + spec["y"]].to_numpy(float) for s_ in d["slopes"]])
    tn = np.empty((N_PERM, len(d["slopes"])))
    for b in range(N_PERM):
        pm = rng.permutation(mv)[d["fc"]]
        _, _, tn[b] = R0.add_t(B, pm[:, None] * slopes_v)
    p_perm = (1 + (np.abs(tn) >= np.abs(t_obs) - 1e-12).sum(0)) / (1 + N_PERM)
    p_perm_1 = (1 + (tn <= t_obs + 1e-12).sum(0)) / (1 + N_PERM)     # one-sided, negative direction
    p0 = len(d["names"])
    out = []
    comps = comp_names(spec)
    for j, s_ in enumerate(d["slopes"]):
        g, se = f["beta"][p0 + j], np.sqrt(f["cov"][p0 + j, p0 + j])
        bi = d["names"].index(f"b_{s_}")
        out.append(dict(term=f"{s_} x moderator", slope=s_, g=g, se=se,
                        lo=g - 1.96 * se, hi=g + 1.96 * se, p_wald=float(2 * norm.sf(abs(g / se))),
                        p_perm=float(p_perm[j]), p_perm_neg=float(p_perm_1[j]), t_fixed=float(t_obs[j]),
                        b_main=f["beta"][bi], se_main=np.sqrt(f["cov"][bi, bi]),
                        tau=f["sd"][comps.index(s_)], tau_null=f0["sd"][comps.index(s_)]))
    info = dict(k=len(d["fields"]), n=d["n"], n_inst=d["n_inst"], s_resid=float(np.sqrt(f["s2"])),
                s_inst=float(f["sd"][comps.index("inst")]) if "inst" in comps else np.nan,
                fields=d["fields"])
    return out, info, f, d


def two_step(S, spec, mod: pd.Series):
    """Per-field OLS slope of y on P (+ the spec's controls), then REML meta-regression on mod."""
    suf = "_" + spec["y"]
    rows = []
    for fld, g in S.groupby("field"):
        Xc = [np.ones(len(g)), g["P" + suf].to_numpy(float)]
        Xc += [g[c + suf].to_numpy(float) for c in CTRL[spec["ctrl"]]]
        Xf = np.column_stack(Xc)
        Xf = Xf[:, np.r_[0, 1, 2 + np.where(np.abs(Xf[:, 2:]).sum(0) > 1e-12)[0]]] if Xf.shape[1] > 2 else Xf
        o = sm.OLS(g["y" + suf].to_numpy(float), Xf).fit()
        rows.append(dict(field=fld, b=o.params[1], se=o.bse[1], n=len(g)))
    T = pd.DataFrame(rows).set_index("field")
    T["m"] = mod.reindex(T.index)
    return T


def meta_reg(b, se, m):
    """REML random-effects meta-regression b ~ 1 + m; Knapp-Hartung SE; returns dict."""
    X = np.column_stack([np.ones(len(b)), m]) if m is not None else np.ones((len(b), 1))
    v = se ** 2

    def nll(t2):
        w = 1 / (v + t2)
        XtW = X.T * w
        M = XtW @ X
        beta = np.linalg.solve(M, XtW @ b)
        r = b - X @ beta
        return 0.5 * (np.sum(np.log(v + t2)) + np.linalg.slogdet(M)[1] + np.sum(w * r ** 2))
    t2 = float(minimize_scalar(nll, bounds=(0, 2), method="bounded", options=dict(xatol=1e-8)).x)
    if nll(0.0) <= nll(t2):
        t2 = 0.0
    w = 1 / (v + t2)
    XtW = X.T * w
    M = XtW @ X
    beta = np.linalg.solve(M, XtW @ b)
    r = b - X @ beta
    k, p = X.shape
    qkh = max(1.0, float(np.sum(w * r ** 2) / (k - p)))
    cov = qkh * np.linalg.inv(M)
    return dict(beta=beta, se=np.sqrt(np.diag(cov)), tau=np.sqrt(t2), df=k - p, w=w)


def perm_spearman(x, y, tag):
    """Spearman of x with y and its two-sided field-label permutation p (N_PERM permutations of x)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = float(spearmanr(x, y)[0])
    rng = rng_for(tag)
    X = np.array([rng.permutation(x) for _ in range(N_PERM)])
    null = spearman_rows(rankdata(y), X)
    return r, float((1 + np.sum(np.abs(null) >= abs(r) - 1e-12)) / (1 + N_PERM))


def leverage_robustness(samples, fmod, null_cache):
    """The two ACS shares are skewed: nursing (about 0.77) and special education (about 0.60) lie far above
    the other fields of the selectivity sample (at most 0.37), so a moderator entered linearly gives those two
    fields most of the leverage. Three checks, on the same fields and specs as table (b0):
      (i)  forms of the moderator: linear, winsorized at WINS, square root, log(share + 0.01), rank normal score;
           each z-scored across the fields in the fit, so g = change in the slope per SD of the transformed
           moderator; Wald and field-label permutation p exactly as in fit_mod (permuting a transformed moderator
           = permuting field labels);
      (ii) field level: Spearman of the per-field OLS slope (two_step, same variables as the spec) with the share,
           field-label permutation p; on all fields and without nursing and special education;
      (iii) nursing and special education dropped together (linear, per unit share), with permutation p."""
    lv, spr, d2 = [], [], []
    for mk in LEV_MODS:
        for sk in LEV_SPECS:
            spec = SPECS[sk]; S = samples[spec["sample"]]
            mv = fmod[mk].reindex(sorted(S.field.unique())).dropna()
            Sm = S[S.field.isin(mv.index)]
            key = (sk, tuple(sorted(mv.index)))
            for fn, fx in FORMS.items():
                z = zs(fx(mv))
                out, info, _, _ = fit_mod(Sm, spec, z, null_cache, key, rng_for(f"lev_{sk}_{mk}_{fn}"))
                for o in out:
                    lv.append(dict(spec=sk, moderator=mk, form=fn, k=info["k"], n=info["n"],
                                   sd_share=float(mv.std(ddof=0)), **o))
            mv2 = mv.drop(DROP2, errors="ignore")
            out, info, _, _ = fit_mod(S[S.field.isin(mv2.index)], spec, mv2, null_cache,
                                      (sk, tuple(sorted(mv2.index))), rng_for(f"drop2_{sk}_{mk}"))
            for o in out:
                d2.append(dict(spec=sk, moderator=mk, drop="nursing + special education", k=info["k"],
                               n=info["n"], **o))
        for sk in SP_SPECS:
            spec = SPECS[sk]; S = samples[spec["sample"]]
            T = two_step(S, spec, fmod[mk]).dropna(subset=["m"])
            r, p = perm_spearman(T.m, T.b, f"fspear_{sk}_{mk}")
            T2 = T.drop(DROP2, errors="ignore")
            r2, p2 = perm_spearman(T2.m, T2.b, f"fspear2_{sk}_{mk}")
            spr.append(dict(spec=sk, moderator=mk, k=len(T), spearman=r, p_perm=p, k_drop2=len(T2),
                            spearman_drop2=r2, p_perm_drop2=p2))
        print(f"[leverage] {mk} done", flush=True)
    return pd.DataFrame(lv), pd.DataFrame(spr), pd.DataFrame(d2)


def fit_cov(S, spec, m: pd.Series, C: pd.DataFrame | None, null_cache: dict, key, rng,
            strata: pd.Series | None = None):
    """Cross-level moderator m with field-level controls C (each C column x every random slope in the base).
    REML fit with m; permutation p by Kennedy residual permutation: m is regressed on [1, C] across the fields
    (OLS) and the residuals are permuted across fields (within `strata` if given); statistic = GLS t of the
    residual x slope columns given the base, with the variance parameters of the model without m (they do not
    depend on the labels). With C = None this is exactly the test of fit_mod. Returns one dict per random slope
    plus the fitted model and design."""
    d = design(S, spec, m, fcov=C)
    X = np.column_stack([d["X0"], d["W"]])
    R = REML(d["y"], X, d["blocks"])
    f = R.fit()
    if key not in null_cache:
        R0 = REML(d["y"], d["X0"], d["blocks"])
        null_cache[key] = (R0, R0.fit())
    R0, f0 = null_cache[key]
    B = R0.base(f0["theta"])
    mv = m.reindex(d["fields"]).to_numpy(float)
    if C is None or C.shape[1] == 0:
        r = mv
    else:
        Cm = np.column_stack([np.ones(len(mv)), C.reindex(d["fields"]).to_numpy(float)])
        r = mv - Cm @ np.linalg.lstsq(Cm, mv, rcond=None)[0]
    slopes_v = np.column_stack([S[s_ + "_" + spec["y"]].to_numpy(float) for s_ in d["slopes"]])
    _, _, t_obs = R0.add_t(B, r[d["fc"]][:, None] * slopes_v)
    if strata is not None:
        sv = strata.reindex(d["fields"]).to_numpy()
        groups = [np.where(sv == s_)[0] for s_ in sorted(set(sv))]
    tn = np.empty((N_PERM, len(d["slopes"])))
    for b in range(N_PERM):
        if strata is None:
            rp = rng.permutation(r)
        else:
            rp = r.copy()
            for gi in groups:
                if len(gi) > 1:
                    rp[gi] = r[rng.permutation(gi)]
        _, _, tn[b] = R0.add_t(B, rp[d["fc"]][:, None] * slopes_v)
    p_perm = (1 + (np.abs(tn) >= np.abs(t_obs) - 1e-12).sum(0)) / (1 + N_PERM)
    p0 = len(d["names"])
    comps = comp_names(spec)
    out = []
    for j, s_ in enumerate(d["slopes"]):
        g, se = f["beta"][p0 + j], np.sqrt(f["cov"][p0 + j, p0 + j])
        bi = d["names"].index(f"b_{s_}")
        out.append(dict(term=f"{s_} x moderator", slope=s_, g=g, se=se, lo=g - 1.96 * se, hi=g + 1.96 * se,
                        p_wald=float(2 * norm.sf(abs(g / se))), p_perm=float(p_perm[j]), t_fixed=float(t_obs[j]),
                        b_main=f["beta"][bi], tau=f["sd"][comps.index(s_)], tau_null=f0["sd"][comps.index(s_)],
                        k=len(d["fields"]), n=d["n"]))
    return out, f, d


def field_type_frame(S: pd.DataFrame, fmod: pd.DataFrame) -> pd.DataFrame:
    """Field-level covariates of the field-type check on the fields of sample S (continuous ones raw here;
    z-scored across the fields of each fit)."""
    flds = sorted(S.field.unique())
    t = pd.DataFrame(index=pd.Index(flds, name="field"))
    t["cip2"] = [CIP2[f] for f in flds]
    for nm, codes in FT_DUMMY.items():
        t[nm] = t.cip2.isin(codes).astype(float)
    g = S.groupby("field")
    t["log_earn"] = np.log(g.earnings.median().reindex(flds))
    t["log_n"] = np.log(fmod["n_institutions"].reindex(flds).astype(float))
    t["absorption"] = fmod["absorption"].reindex(flds)
    t["prestige_range"] = np.log((g.prestige_score.max() - g.prestige_score.min()).reindex(flds))
    return t


def field_type_checks(samples, fmod, null_cache):
    """Is the licensed-share moderation of the prestige slope a dose-response in licensing or a field-type contrast?
    The strict share is lowest in every engineering, CS, math/statistics and business field. On the selectivity
    sample (fields with an ACS share), per SD of the share in three forms (linear, square root, rank normal score):
      (i)   field-level controls, each x every random slope, Kennedy residual permutation of the share (fit_cov):
            none; the engineering/CS/math/business dummy ('bloc'); CIP-2 fixed slopes with the share permuted within
            CIP-2 (stratified; a within-discipline test) -- specs FT_SPECS; and, rank form only, specs A4/A4B: the bloc
            without math, the README 'integrated' clusters with and without business, log field earnings, log field
            size, the academic absorption share and the log prestige range. For 'bloc' the bloc coefficient net of the
            share gets its own Kennedy permutation p (the bloc residualized on the share).
      (ii)  the bloc dummy alone as the moderator;
      (iii) the fits within the non-bloc fields and within the bloc fields (share re-standardized / re-ranked
            within the subset; plain field-label permutation), the field-level Spearman of per-field slopes with
            the share there (A4, A4B) and a field-cluster bootstrap of the rank form (A4, A4B).
    'none' uses the rng streams of leverage_robustness, so it reproduces table (b0) for the specs run there."""
    S = samples["sat"]
    FC = field_type_frame(S, fmod)
    RNS = "rank normal score"
    rows, sprs, boots, coll = [], [], [], []
    ncache = {}                 # null models with field-level controls / on subsets (local: freed on return)

    def add(out, **kw):
        for o in out:
            rows.append(dict(**kw, **o))

    for mk in LEV_MODS:
        mv0 = fmod[mk].reindex(sorted(S.field.unique())).dropna()
        flds = list(mv0.index)
        Sm = S[S.field.isin(flds)]
        F0 = FC.loc[flds]
        bl = F0["bloc"]
        coll.append(dict(moderator=mk, k=len(flds), k_bloc=int(bl.sum()),
                         spearman_bloc=float(spearmanr(mv0, bl)[0]),
                         pearson_rank_bloc=float(np.corrcoef(zs(rank_ns(mv0)), bl)[0, 1]),
                         bloc_lo=float(mv0[bl == 1].min()), bloc_hi=float(mv0[bl == 1].max()),
                         other_lo=float(mv0[bl == 0].min()), other_hi=float(mv0[bl == 0].max()),
                         n_strata=int(F0.cip2.nunique()),
                         n_strata2=int((F0.cip2.value_counts() >= 2).sum()),
                         k_in_strata2=int(F0.cip2.map(F0.cip2.value_counts()).ge(2).sum())))
        for fn in FT_FORMS:
            z = zs(FORMS[fn](mv0))
            for cs in ["none", "bloc", "cip2_fe"] + [c for c in FT_SETS if c not in ("bloc", "cip2_fe")]:
                if cs not in ("none", "bloc", "cip2_fe") and fn != RNS:
                    continue
                for sk in (FT_SPECS if cs in ("none", "bloc", "cip2_fe") else FT_SPECS_SHORT):
                    spec = SPECS[sk]
                    strata = None
                    if cs == "none":
                        C = None
                        key = (sk, tuple(sorted(flds)))
                        tag = f"lev_{sk}_{mk}_{fn}" if sk in LEV_SPECS else f"ft_none_{sk}_{mk}_{fn}"
                    elif cs == "cip2_fe":
                        codes = sorted(F0.cip2.unique())
                        C = pd.DataFrame({f"cip{c}": (F0.cip2 == c).astype(float) for c in codes[1:]}, index=F0.index)
                        strata = F0.cip2
                        key = (sk, tuple(sorted(flds)), cs)
                        tag = f"ft_{cs}_{sk}_{mk}_{fn}"
                    else:
                        v = F0[cs].dropna()         # absorption is missing for environmental engineering
                        C = pd.DataFrame({cs: v if cs in FT_DUMMY else zs(v)}, index=v.index)
                        key = (sk, tuple(sorted(v.index)), cs)
                        tag = f"ft_{cs}_{sk}_{mk}_{fn}"
                    fl_c = list(C.index) if C is not None else flds
                    out, f, d = fit_cov(Sm[Sm.field.isin(fl_c)], spec, z.reindex(fl_c), C,
                                        null_cache if cs == "none" else ncache, key, rng_for(tag), strata=strata)
                    extra = {}
                    if C is not None and cs != "cip2_fe":
                        j = d["names"].index(f"h_{cs}_P")
                        cg, cse = f["beta"][j], np.sqrt(f["cov"][j, j])
                        extra = dict(cov_g=cg, cov_se=cse, cov_p_wald=float(2 * norm.sf(abs(cg / cse))),
                                     corr_cov=float(np.corrcoef(z.reindex(fl_c), C[cs].reindex(fl_c))[0, 1]))
                    if cs == "bloc":
                        # the bloc net of the share: Kennedy permutation with the roles swapped
                        ob, _, _ = fit_cov(Sm, spec, bl, pd.DataFrame({"share": z}), {}, "swap",
                                           rng_for(f"ftbloc_{sk}_{mk}_{fn}"))
                        extra["cov_p_perm"] = ob[0]["p_perm"]
                        assert abs(ob[0]["g"] - extra["cov_g"]) < 1e-5, (sk, mk, fn, ob[0]["g"], extra["cov_g"])
                    add(out, spec=sk, moderator=mk, form=fn, control=cs, sample="all fields with a share", **extra)
            print(f"[field-type] {mk} {fn} done", flush=True)
        # (ii) the bloc dummy alone (same fields)
        for sk in FT_SPECS:
            out, _, _ = fit_cov(Sm, SPECS[sk], bl, None, null_cache, (sk, tuple(sorted(flds))),
                                rng_for(f"ftblocalone_{sk}_{mk}"))
            add(out, spec=sk, moderator="bloc", form="dummy", control=f"none (fields with {mk})",
                sample="all fields with a share")
        # (iii) within the non-bloc and within the bloc fields
        for sub, inb in (("non-bloc fields", 0.0), ("bloc fields", 1.0)):
            mv = mv0[bl == inb]
            Ss = S[S.field.isin(mv.index)]
            for fn in ("linear", RNS):
                z = zs(FORMS[fn](mv))
                for sk in FT_SPECS:
                    out, f, d = fit_cov(Ss, SPECS[sk], z, None, ncache, (sk, tuple(sorted(mv.index))),
                                        rng_for(f"ftsub_{sub}_{sk}_{mk}_{fn}"))
                    add(out, spec=sk, moderator=mk, form=fn, control="none", sample=sub,
                        share_lo=float(mv.min()), share_hi=float(mv.max()))
                    if fn == RNS and sk in ("A4", "A4B"):
                        fg = FieldGLS(Ss, SPECS[sk], z, f["theta"])
                        base = fg.stats()
                        g0 = fg.solve(*base)[1]
                        assert abs(g0 - out[0]["g"]) < 1e-6, (sub, sk, mk, g0, out[0]["g"])
                        rng = rng_for(f"ftboot_{sub}_{sk}_{mk}")
                        D = np.empty(N_BOOT)
                        for b in range(N_BOOT):
                            cnt = np.bincount(rng.integers(0, len(mv), size=len(mv)), minlength=len(mv))
                            D[b] = FieldGLS.solve(*base, counts=cnt)[1]
                        boots.append(dict(sample=sub, spec=sk, moderator=mk, form=fn, g=out[0]["g"], n_draws=N_BOOT,
                                          boot_lo=np.percentile(D, 2.5), boot_hi=np.percentile(D, 97.5),
                                          boot_se=D.std(ddof=1)))
            for sk in ("A4", "A4B"):
                T = two_step(Ss, SPECS[sk], mv).dropna(subset=["m"])
                r_, p_ = perm_spearman(T.m, T.b, f"ftsp_{sub}_{sk}_{mk}")
                sprs.append(dict(sample=sub, spec=sk, moderator=mk, k=len(T), spearman=r_, p_perm=p_))
        print(f"[field-type] {mk} subsets done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(sprs), pd.DataFrame(boots), pd.DataFrame(coll), FC


def run_part_a(cells: pd.DataFrame, fmod: pd.DataFrame):
    nf = cells.groupby("field").size()
    S_all = cells[cells.field.map(nf) >= NMIN].copy()
    need = ["SAT_AVG", "ADM_RATE", "PCTPELL", "ST_EARN", "CONTROL", "STABBR"]
    ok = S_all[need].notna().all(1) & (S_all.STABBR.fillna("") != "")
    S_sat = S_all[ok].copy()
    nfs = S_sat.groupby("field").size()
    S_sat = S_sat[S_sat.field.map(nfs) >= NMIN + 2].copy()        # NMIN and residual df >= DFMIN
    S_all = transform(S_all, sat=False)
    S_sat = transform(S_sat, sat=True)
    samples = {"all": S_all, "sat": S_sat}

    # --- reproduce the gap map on S_all (the baseline object) ---
    rep = S_all.groupby("field").apply(lambda g: spearmanr(g.prestige_score, g.earnings)[0],
                                       include_groups=False)
    ref = fmod["spearman"].reindex(rep.index)
    repro = float((rep - ref).abs().max())
    assert repro < 1e-9, "gap-map baseline did not reproduce"
    nref = fmod["n_institutions"].reindex(rep.index)
    assert (S_all.groupby("field").size() == nref).all(), "gap-map n did not reproduce"

    # --- validation of the REML engine against statsmodels MixedLM (A1 null model) ---
    d, R, f = fit_null(S_all, SPECS["A1"])
    md = sm.MixedLM(d["y"], d["X0"], groups=d["fc"], exog_re=d["blocks"][0][1][:, None]).fit(reml=True)
    bi = d["names"].index("b_P")
    val = dict(tau_mine=float(f["sd"][0]), tau_sm=float(np.sqrt(np.asarray(md.cov_re).ravel()[0])),
               sig_mine=float(np.sqrt(f["s2"])), sig_sm=float(np.sqrt(md.scale)),
               b_mine=float(f["beta"][bi]), b_sm=float(np.asarray(md.fe_params)[bi]))
    assert abs(val["tau_mine"] - val["tau_sm"]) < 1e-3 and abs(val["b_mine"] - val["b_sm"]) < 1e-4, val

    rows, hetero, blups, null_cache = [], [], {}, {}
    for sk, spec in SPECS.items():
        S = samples[spec["sample"]]
        # heterogeneity of field slopes: model without moderator
        d0, R0, f0 = fit_null(S, spec)
        comps = comp_names(spec)
        null_cache[(sk, tuple(d0["fields"]))] = (R0, f0)
        s0 = spec["slopes"][0]
        bi = d0["names"].index(f"b_{s0}")
        hrow = dict(spec=sk, label=spec["label"], main=s0, k=len(d0["fields"]), n=d0["n"],
                    n_inst=d0["n_inst"], b_main=f0["beta"][bi], se_b_main=np.sqrt(f0["cov"][bi, bi]),
                    s_resid=float(np.sqrt(f0["s2"])))
        for j, c in enumerate(comps):
            hrow[f"sd_{c}"] = float(f0["sd"][j])
        if spec["inst"] != "re" or sk in ("A2", "A5"):
            for c in [c for c in comps if c != "inst"]:
                pr = R0.profile(comps.index(c), f0)
                hrow.update({f"sd_{c}_lo": pr["lo"], f"sd_{c}_hi": pr["hi"], f"lrt_{c}": pr["lrt"],
                             f"lrt_p_{c}": pr["lrt_p"]})
        # BLUP field slopes (first slope)
        P_ = f0["P"]
        u = P_["l"] * cho_solve((P_["L"], True), P_["l"] * (R0.Z.T @ (d0["y"] - d0["X0"] @ f0["beta"])))
        blups[sk] = pd.Series(f0["beta"][bi] + u[R0.kidx[0]], index=d0["fields"])
        hetero.append(hrow)
        # moderators
        mods = dict(MODS)
        if sk in BA_SPECS:
            mods.update(MODS_BA)
        if sk in ("A1", "A4"):
            mods.update(BRIDGE)
        for mk in mods:
            mv = fmod[mk].reindex(sorted(S.field.unique()))
            fl = list(mv.dropna().index)
            m_use = mv.loc[fl]
            if mk in BRIDGE:
                m_use = (m_use - m_use.mean()) / m_use.std(ddof=0)
            out, info, f, d = fit_mod(S[S.field.isin(fl)], spec, m_use, null_cache, (sk, tuple(sorted(fl))),
                                      rng_for(f"perm_{sk}_{mk}"))
            for o in out:
                rows.append(dict(spec=sk, spec_label=spec["label"], moderator=mk, mod_label=mods[mk],
                                 **{k: v for k, v in info.items() if k != "fields"}, **o))
        # institution selectivity as a cell-level moderator of the prestige slope
        if sk in ("A4", "A5"):
            d1 = design(S, spec, None, extra="PxSAT")
            R1 = REML(d1["y"], d1["X0"], d1["blocks"]); f1 = R1.fit()
            j = d1["names"].index("b_PxSAT"); g, se = f1["beta"][j], np.sqrt(f1["cov"][j, j])
            bi1 = d1["names"].index("b_P")
            rows.append(dict(spec=sk, spec_label=spec["label"], moderator="PxSAT",
                             mod_label="institution SAT (within-field normal score) x prestige",
                             k=len(d1["fields"]), n=d1["n"], n_inst=d1["n_inst"],
                             s_resid=float(np.sqrt(f1["s2"])),
                             s_inst=float(f1["sd"][-1]) if spec["inst"] == "re" else np.nan,
                             term="P x SAT", slope="P", g=g, se=se, lo=g - 1.96 * se, hi=g + 1.96 * se,
                             p_wald=float(2 * norm.sf(abs(g / se))), p_perm=np.nan, p_perm_neg=np.nan,
                             t_fixed=np.nan, b_main=f1["beta"][bi1], se_main=np.sqrt(f1["cov"][bi1, bi1]),
                             tau=f1["sd"][0], tau_null=f0["sd"][0]))
        print(f"[A] {sk} done: k={hrow['k']} n={hrow['n']} b_{s0}={hrow['b_main']:+.3f} "
              f"sd_{s0}={hrow['sd_' + s0]:.3f}", flush=True)
    A = pd.DataFrame(rows)
    H = pd.DataFrame(hetero)

    # --- robustness of the licensing interaction: drop nursing + communication disorders; LOFO ---
    rob = []
    for sk in ("A1", "A4", "A4B", "A5"):
        spec = SPECS[sk]; S = samples[spec["sample"]]
        for mk in list(MODS) + list(MODS_BA):
            mv = fmod[mk].reindex(sorted(S.field.unique())).dropna().drop(DROP_SET, errors="ignore")
            if mv.nunique() < 2:
                continue
            d = design(S[S.field.isin(mv.index)], spec, mv)
            X = np.column_stack([d["X0"], d["W"]])
            f = REML(d["y"], X, d["blocks"]).fit()
            g, se = f["beta"][-1], np.sqrt(f["cov"][-1, -1])
            rob.append(dict(spec=sk, moderator=mk, drop="nursing + communication disorders",
                            k=len(d["fields"]), g=g, se=se, lo=g - 1.96 * se, hi=g + 1.96 * se,
                            p_wald=float(2 * norm.sf(abs(g / se))), tau=f["sd"][0]))
    for sk, mk in LOFO:
        spec = SPECS[sk]; S = samples[spec["sample"]]
        mv_all = fmod[mk].reindex(sorted(S.field.unique())).dropna()
        for fd in mv_all.index:
            mv = mv_all.drop(fd)
            d = design(S[S.field.isin(mv.index)], spec, mv)
            X = np.column_stack([d["X0"], d["W"]])
            f = REML(d["y"], X, d["blocks"]).fit()
            g, se = f["beta"][-1], np.sqrt(f["cov"][-1, -1])
            rob.append(dict(spec=sk, moderator=mk, drop=fd, k=len(d["fields"]), g=g, se=se,
                            lo=g - 1.96 * se, hi=g + 1.96 * se, p_wald=float(2 * norm.sf(abs(g / se))),
                            tau=f["sd"][0]))
        print(f"[lofo] {sk} {mk} done", flush=True)
    RB = pd.DataFrame(rob)
    # --- leverage of the two high-share fields: moderator forms, field-level Spearman, drop both ---
    LV, SPR, D2 = leverage_robustness(samples, fmod, null_cache)
    # --- field type: engineering/CS/math/business contrast vs licensing dose-response ---
    FT, FTS, FTB, FTC, FC = field_type_checks(samples, fmod, null_cache)
    chk = FT[(FT.control == "none") & (FT["sample"] == "all fields with a share") & FT.spec.isin(LEV_SPECS)]
    for r in chk.itertuples():          # the no-control rows are table (b0) re-run on the same rng streams
        l_ = LV[(LV.spec == r.spec) & (LV.moderator == r.moderator) & (LV.form == r.form) & (LV.slope == r.slope)].iloc[0]
        assert abs(l_.g - r.g) < 1e-8 and abs(l_.p_perm - r.p_perm) < 1e-12, (r.spec, r.moderator, r.form)

    # --- field-level comparisons on the same fields ---
    fl_rows, fs_rows = [], {}
    for sk in ("A1", "A3", "A4", "A4B", "L1", "L3", "L4", "L4B"):
        spec = SPECS[sk]; S = samples[spec["sample"]]
        mks = list(MODS) + (list(MODS_BA) if sk in BA_SPECS else []) + (list(BRIDGE) if sk in ("A1", "A4") else [])
        for mk in mks:
            T = two_step(S, spec, fmod[mk])
            if mk == "lic":
                fs_rows[sk] = T[["b", "se", "n"]]
            T = T.dropna(subset=["m"])
            m = T.m.to_numpy(float)
            if mk in BRIDGE:
                m = (m - m.mean()) / m.std(ddof=0)
            b = T.b.to_numpy(float); se_b = T.se.to_numpy(float)
            mr = meta_reg(b, se_b, m)
            w0 = meta_reg(b, se_b, None)["w"]          # null-model weights: label-free statistic

            def wls_slope(mm):
                Xp = np.column_stack([np.ones(len(b)), mm])
                return np.linalg.solve((Xp.T * w0) @ Xp, (Xp.T * w0) @ b)[1]
            obs = wls_slope(m)
            rng = rng_for(f"meta_perm_{sk}_{mk}")
            null = np.array([wls_slope(rng.permutation(m)) for _ in range(N_PERM)])
            p_perm = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + N_PERM)
            g, se = mr["beta"][1], mr["se"][1]
            q = tdist.ppf(0.975, mr["df"])
            fl_rows.append(dict(spec=sk, moderator=mk, method="two-step REML meta-regression (Knapp-Hartung)",
                                k=len(T), g=g, se=se, lo=g - q * se, hi=g + q * se,
                                p=float(2 * tdist.sf(abs(g / se), mr["df"])), p_perm=float(p_perm),
                                tau=mr["tau"]))
            if sk in ("A1", "A3") and mk in MODS:
                G = fmod.loc[T.index]
                y_ = G.spearman.to_numpy(float); w_ = 1 / G.se.to_numpy(float) ** 2
                o = sm.WLS(y_, sm.add_constant(T.m.to_numpy(float)), weights=w_).fit(cov_type="HC1")
                fl_rows.append(dict(spec=sk, moderator=mk,
                                    method="gap-map rho_f ~ moderator, WLS 1/se^2, HC1 (scripts/20 style)",
                                    k=len(T), g=o.params[1], se=o.bse[1], lo=o.conf_int()[1][0],
                                    hi=o.conf_int()[1][1], p=float(o.pvalues[1]), p_perm=np.nan, tau=np.nan))
    FL = pd.DataFrame(fl_rows)
    # per-field slope table (two-step OLS and shrunken BLUPs)
    FS = fmod[["lic", "acs_strict", "acs_strict_ba", "acs_broad", "acs_k12_ba", "acs_health_ba", "spearman",
               "reliable"]].copy()
    FS["cip2"] = FS.index.map(CIP2)
    FS["bloc"] = FS.cip2.isin(BLOC).astype(float)
    for sk in ("A1", "A3", "A4", "A4B", "L4"):
        FS[f"b_{sk}"] = fs_rows[sk]["b"]; FS[f"se_{sk}"] = fs_rows[sk]["se"]; FS[f"n_{sk}"] = fs_rows[sk]["n"]
    for sk in ("A1", "A3", "A4", "A5", "C1", "C2"):
        FS[f"blup_{sk}"] = blups[sk]
    FS = FS[FS.b_A1.notna()].reset_index()
    FS.insert(1, "label", FS.field.map(LAB))

    # --- joint institution bootstrap: raw vs selectivity-controlled interaction, same draws ---
    BT = boot_raw_vs_controlled(S_sat, fmod)
    # --- optimizer audit near the theta = 0 boundary ---
    RC = reml_checks(S_sat, fmod)
    return dict(A=A, H=H, FL=FL, BT=BT, RB=RB, FS=FS, RC=RC, LV=LV, SPR=SPR, D2=D2, val=val, repro=repro,
                FT=FT, FTS=FTS, FTB=FTB, FTC=FTC, FC=FC,
                samples=samples, blups=blups, n_all=len(S_all), n_sat=len(S_sat))


class FieldGLS:
    """GLS with fixed variance ratio for specs whose only random effect is the field slope on P:
    y_f = A_f a_f + [p_f, m_f p_f] (b, g)' + e_f, V_f = I + theta^2 p_f p_f' (Sherman-Morrison inverse),
    A_f = field-specific intercept and control slopes. Fields are independent blocks with their own
    nuisance columns, so each field reduces to 2 x 2 / 2 x 1 partialled statistics: a field-cluster draw
    is a count-weighted sum of them and an institution draw a row-weighted recomputation. Same estimate
    as gls_fixed(design_boot(...)) (asserted in boot_raw_vs_controlled)."""

    def __init__(self, S, spec, mv, theta):
        suf = "_" + spec["y"]
        fields = sorted(S.field.unique())
        fc = pd.Categorical(S.field, categories=fields).codes
        p = S["P" + suf].to_numpy(float)
        A = [np.ones(len(S))] + [S[c + suf].to_numpy(float) for c in CTRL[spec["ctrl"]]]
        m = mv.reindex(fields).to_numpy(float)[fc]
        B = np.column_stack(A + [p, m * p, S["y" + suf].to_numpy(float)])
        self.na, self.k, self.m = len(A), len(fields), B.shape[1]
        self.t2 = float(theta[0]) ** 2
        self.BB = np.einsum("ni,nj->nij", B, B).reshape(len(S), -1)
        self.Bp = B * p[:, None]
        self.pp = p * p
        self.H = sp.csr_matrix((np.ones(len(S)), (fc, np.arange(len(S)))), shape=(self.k, len(S)))

    def stats(self, w=None):
        H = self.H if w is None else self.H @ sp.diags(np.asarray(w, float))
        k, m, na = self.k, self.m, self.na
        G = np.asarray(H @ self.BB).reshape(k, m, m)
        q = np.asarray(H @ self.Bp); s = np.asarray(H @ self.pp).ravel()
        c = self.t2 / (1.0 + self.t2 * s)
        G = G - c[:, None, None] * q[:, :, None] * q[:, None, :]
        Gab = G[:, :na, na:]
        R = G[:, na:, na:] - np.transpose(Gab, (0, 2, 1)) @ np.linalg.pinv(G[:, :na, :na], rcond=1e-10) @ Gab
        return R[:, :2, :2], R[:, :2, 2]

    @staticmethod
    def solve(Scc, Scy, counts=None):
        cnt = np.ones(len(Scc)) if counts is None else np.asarray(counts, float)
        return np.linalg.pinv(np.tensordot(cnt, Scc, 1)) @ (cnt @ Scy)


def boot_raw_vs_controlled(S, fmod):
    """Raw (A3) vs controlled (A4, A4B, A4E, A5) cross-level interaction under two resampling schemes;
    each draw re-estimated by GLS with the spec's REML variance ratios held fixed.
      field-cluster: fields drawn with replacement (each copy its own field, random slope and controls);
        carries the between-field sampling variance, which is the main uncertainty for a moderator that
        is measured once per field.
      institution: institutions drawn with replacement, the realized fields kept; conditional on the
        realized field slopes u_f, so it reflects within-field sampling error only (too narrow for the
        level of g; reported for the raw-vs-controlled difference only).
    Moderators: the dummy, the two ACS shares (linear, per unit share) and the two shares' rank normal scores
    (per SD; `*_rns`; specs without institution random effects only), whose scores are fixed at their values
    on the observed fields."""
    out = []
    for mk in ("lic", "acs_strict", "acs_strict_ba", "acs_strict_rns", "acs_strict_ba_rns", "acs_broad_rns"):
        if mk.endswith("_rns"):         # rank-normal-score form of the share, per SD (fast specs only)
            mv = zs(rank_ns(fmod[mk[:-4]].reindex(sorted(S.field.unique())).dropna()))
            specs = [s_ for s_ in BOOT_SPECS if SPECS[s_]["inst"] is None]
        else:
            mv = fmod[mk].reindex(sorted(S.field.unique())).dropna()
            specs = list(BOOT_SPECS)
        Sm = S[S.field.isin(mv.index)].reset_index(drop=True)
        th, dd = {}, {}
        for sk in specs:
            d = design(Sm, SPECS[sk], mv)
            X = np.column_stack([d["X0"], d["W"]])
            R = REML(d["y"], X, d["blocks"]); f = R.fit()
            th[sk] = f["theta"]
            dd[sk] = f["beta"][X.shape[1] - 1]
        fast = {sk: FieldGLS(Sm, SPECS[sk], mv, th[sk]) for sk in specs if SPECS[sk]["inst"] is None}
        base = {sk: fg.stats() for sk, fg in fast.items()}
        for sk, fg in fast.items():                  # fast path == generic GLS == REML estimate
            db = design_boot(Sm, SPECS[sk], mv)
            g_gen = gls_fixed(db["y"], np.column_stack([db["X0"], db["W"]]), db["blocks"], th[sk])[-1]
            g_fast = fg.solve(*base[sk])[1]
            assert abs(g_fast - g_gen) < 1e-7 and abs(g_fast - dd[sk]) < 1e-6, (mk, sk, g_fast, g_gen, dd[sk])
        draws = {}
        # (i) field-cluster bootstrap
        flds = np.array(sorted(mv.index))
        mvals = mv.reindex(flds).to_numpy(float)
        rows_f = Sm.groupby("field").indices
        rng = rng_for(f"fboot_{mk}")
        D = np.full((N_BOOT, len(specs)), np.nan)
        for b in range(N_BOOT):
            pick = rng.integers(0, len(flds), size=len(flds))
            if np.unique(mvals[pick]).size < 2:        # moderator constant in the draw (dummy only)
                continue
            cnt = np.bincount(pick, minlength=len(flds))
            Sb = None
            for j, sk in enumerate(specs):
                if sk in fast:
                    D[b, j] = FieldGLS.solve(*base[sk], counts=cnt)[1]
                    continue
                if Sb is None:
                    idx = np.concatenate([rows_f[flds[i]] for i in pick])
                    labs = [f"{flds[i]}#{j2:03d}" for j2, i in enumerate(pick)]
                    Sb = Sm.iloc[idx].copy()
                    Sb["field"] = np.concatenate([np.full(len(rows_f[flds[i]]), lb) for i, lb in zip(pick, labs)])
                    mvb = pd.Series(mvals[pick], index=labs)
                db = design_boot(Sb, SPECS[sk], mvb)
                D[b, j] = gls_fixed(db["y"], np.column_stack([db["X0"], db["W"]]), db["blocks"], th[sk])[-1]
                if b == 0:                              # fast path on a resample == generic GLS
                    for sk2 in fast:
                        db2 = design_boot(Sb, SPECS[sk2], mvb)
                        g2 = gls_fixed(db2["y"], np.column_stack([db2["X0"], db2["W"]]), db2["blocks"], th[sk2])[-1]
                        assert abs(g2 - D[b, specs.index(sk2)]) < 1e-6, (mk, sk2, "field draw 0")
        draws["field-cluster"] = D
        # (ii) institution bootstrap (conditional on the realized fields)
        insts = np.array(sorted(Sm.inst_key.unique()))
        icode = pd.Categorical(Sm.inst_key, categories=insts).codes
        rows_of = Sm.groupby("inst_key").indices
        rng = rng_for(f"boot_{mk}")
        D = np.full((N_BOOT, len(specs)), np.nan)
        for b in range(N_BOOT):
            pick = rng.integers(0, len(insts), size=len(insts))
            w = np.bincount(pick, minlength=len(insts))[icode]
            Sb = None
            for j, sk in enumerate(specs):
                if sk in fast:
                    D[b, j] = FieldGLS.solve(*fast[sk].stats(w))[1]
                    continue
                if Sb is None:
                    idx = np.concatenate([rows_of[insts[i]] for i in pick])
                    Sb = Sm.iloc[idx].copy()
                    Sb["inst_key"] = np.concatenate([np.full(len(rows_of[insts[i]]), str(j2))
                                                     for j2, i in enumerate(pick)])
                db = design_boot(Sb, SPECS[sk], mv)
                D[b, j] = gls_fixed(db["y"], np.column_stack([db["X0"], db["W"]]), db["blocks"], th[sk])[-1]
                if b == 0:
                    for sk2 in fast:
                        db2 = design_boot(Sb, SPECS[sk2], mv)
                        g2 = gls_fixed(db2["y"], np.column_stack([db2["X0"], db2["W"]]), db2["blocks"], th[sk2])[-1]
                        assert abs(g2 - D[b, specs.index(sk2)]) < 1e-6, (mk, sk2, "institution draw 0")
        draws["institution (conditional on fields)"] = D
        for scheme, D in draws.items():
            ok = np.isfinite(D).all(1)
            Dk = D[ok]
            for j, sk in enumerate(specs):
                out.append(dict(moderator=mk, scheme=scheme, spec=sk, g=dd[sk], n_draws=int(ok.sum()),
                                boot_lo=np.percentile(Dk[:, j], 2.5), boot_hi=np.percentile(Dk[:, j], 97.5),
                                boot_se=Dk[:, j].std(ddof=1)))
            for j, sk in enumerate(specs[1:], start=1):
                diff = Dk[:, j] - Dk[:, 0]
                out.append(dict(moderator=mk, scheme=scheme, spec=f"{sk}-A3", g=dd[sk] - dd["A3"],
                                n_draws=int(ok.sum()), boot_lo=np.percentile(diff, 2.5),
                                boot_hi=np.percentile(diff, 97.5), boot_se=diff.std(ddof=1),
                                ratio=dd[sk] / dd["A3"] if dd["A3"] != 0 else np.nan))
        print(f"[boot] {mk} done", flush=True)
    return pd.DataFrame(out)


def reml_checks(S, fmod):
    """Optimizer audit on fits whose REML optimum is near theta = 0. The fit used in this script vs
    (i) the earlier theta-parametrized two-start L-BFGS-B (starts 0.2, 1.0), (ii) Nelder-Mead on |theta|
    from five starts, (iii) a grid over the prestige-slope theta with the other components at the REML
    optimum. Asserts that the fit used here is at least as good as (ii) and (iii)."""
    cases = [("A4", "acs_strict", ()), ("A4", "acs_strict_ba", ("nursing",)), ("A4B", "acs_strict_ba", ()),
             ("A4B", "acs_strict", tuple(DROP_SET)), ("A7", "acs_strict", ())]
    rows = []
    for sk, mk, drop in cases:
        mv = fmod[mk].reindex(sorted(S.field.unique())).dropna()
        drop = tuple(f for f in drop if f in mv.index)      # fields actually present
        if drop:
            mv = mv.drop(list(drop))
        d = design(S[S.field.isin(mv.index)], SPECS[sk], mv)
        X = np.column_stack([d["X0"], d["W"]])
        R = REML(d["y"], X, d["blocks"]); f = R.fit()
        K = R.K
        jg = X.shape[1] - len(d["slopes"])            # g of the first slope (prestige)
        old = None
        for s0 in (0.2, 1.0):
            r = minimize(R.dev, np.full(K, s0), method="L-BFGS-B", bounds=[(0.0, 30.0)] * K)
            if old is None or r.fun < old.fun - 1e-9:
                old = r
        fo = R.fit(fixed={k: float(old.x[k]) for k in range(K)})
        nm = None
        for s0 in (0.02, 0.05, 0.1, 0.2, 0.5):
            r = minimize(lambda t: R.dev(np.abs(t)), np.full(K, s0), method="Nelder-Mead",
                         options=dict(xatol=1e-8, fatol=1e-10, maxiter=5000, maxfev=10000))
            if nm is None or r.fun < nm.fun - 1e-9:
                nm = r
        grid = np.round(np.arange(0.0, 0.4001, 0.0005), 5)
        gd = np.array([R.dev(np.r_[v, f["theta"][1:]]) for v in grid])
        assert f["dev"] <= min(nm.fun, gd.min()) + 1e-6, (sk, mk, f["dev"], nm.fun, gd.min())
        rows.append(dict(spec=sk, moderator=mk, drop=" + ".join(drop), k=len(d["fields"]), n=d["n"],
                         theta=float(f["theta"][0]), sd=float(f["sd"][0]), dev=float(f["dev"]),
                         g=float(f["beta"][jg]), se=float(np.sqrt(f["cov"][jg, jg])),
                         dev_theta0=float(R.dev(np.r_[0.0, f["theta"][1:]])),
                         old_theta=float(old.x[0]), old_sd=float(fo["sd"][0]), old_dev=float(old.fun),
                         old_g=float(fo["beta"][jg]), old_se=float(np.sqrt(fo["cov"][jg, jg])),
                         nm_theta=float(abs(nm.x[0])), nm_dev=float(nm.fun),
                         grid_theta=float(grid[int(np.argmin(gd))]), grid_dev=float(gd.min()),
                         theta_all=" ".join(f"{v:.4f}" for v in f["theta"]),
                         old_theta_all=" ".join(f"{v:.4f}" for v in old.x),
                         nm_theta_all=" ".join(f"{abs(v):.4f}" for v in nm.x)))
    print("[check] REML optimizer audit done", flush=True)
    return pd.DataFrame(rows)


def design_boot(Sb, spec, mv):
    """design() without the rank-reduction assertion (resamples can lose columns); pinv handles it."""
    suf = "_" + spec["y"]
    fields = sorted(Sb.field.unique())
    fc = pd.Categorical(Sb.field, categories=fields).codes
    k = len(fields); D = np.eye(k)[fc]
    cols = [np.ones(len(Sb))] + [D[:, j] for j in range(1, k)] + [Sb["P" + suf].to_numpy(float)]
    for c in CTRL[spec["ctrl"]]:
        v = Sb[c + suf].to_numpy(float)
        cols += [D[:, j] * v for j in range(k)]
    blocks = [(fc, Sb["P" + suf].to_numpy(float))]
    if spec["inst"] == "re":
        blocks.append((pd.factorize(Sb.inst_key)[0], np.ones(len(Sb))))
    W = (mv.reindex(fields).to_numpy(float)[fc] * Sb["P" + suf].to_numpy(float))[:, None]
    return dict(y=Sb["y" + suf].to_numpy(float), X0=np.column_stack(cols), W=W, blocks=blocks)


# =============================================================================================
# part (b): power and equivalence of the field-level nulls
# =============================================================================================
def spearman_rows(x_rank, Y):
    """Spearman of a fixed rank vector with each row of Y."""
    ry = rankdata(Y, axis=1)
    xc = x_rank - x_rank.mean(); yc = ry - ry.mean(1, keepdims=True)
    return (yc @ xc) / np.sqrt((yc ** 2).sum(1) * (xc ** 2).sum())


def p_two_sided_spearman(s, n):
    """Two-sided p of Spearman's rho, t approximation (scipy.stats.spearmanr's default)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = s * np.sqrt((n - 2) / np.clip(1 - s ** 2, 1e-12, None))
    return 2 * tdist.sf(np.abs(t), n - 2)


def interp_cross(xg, yg, level):
    """First x at which yg crosses `level` (linear interpolation); nan if never."""
    for i in range(1, len(xg)):
        if (yg[i - 1] - level) * (yg[i] - level) <= 0 and yg[i] != yg[i - 1]:
            return float(xg[i - 1] + (level - yg[i - 1]) * (xg[i] - xg[i - 1]) / (yg[i] - yg[i - 1]))
    return np.nan


def corr_null(x, obs, se, direction, tag):
    """Power / equivalence for a field-level Spearman null with a noisy outcome (rho_f)."""
    x, obs, se = map(lambda a: np.asarray(a, float), (x, obs, se))
    n = len(x)
    s_obs, p_obs = spearmanr(x, obs)
    s_obs = float(s_obs)
    # analytic (observed-score scale): Fisher z with var 1.06/(n-3)
    sdz = np.sqrt(1.06 / (n - 3))
    z = np.arctanh(s_obs)
    a90 = (np.tanh(z - 1.645 * sdz), np.tanh(z + 1.645 * sdz))
    mde_a = float(np.tanh((1.96 + Z80) * sdz))
    bound_a = a90[1] if direction > 0 else a90[0]
    # simulation (true-score scale)
    var_obs = obs.var(ddof=1); noise = np.mean(se ** 2)
    rel = 1 - noise / var_obs
    sT = np.sqrt(max(var_obs - noise, 1e-6))
    zx = norm.ppf((rankdata(x) - 0.5) / n); zx = (zx - zx.mean()) / zx.std()
    xr = rankdata(x)
    grid = np.round(np.arange(-0.95, 0.9501, 0.025), 4)
    rng = rng_for(tag)
    E = rng.standard_normal((N_SIM, n)); U = rng.standard_normal((N_SIM, n))  # common random numbers
    power, cdf, mean_s = [], [], []
    for r in grid:
        T = sT * (r * zx[None, :] + np.sqrt(1 - r ** 2) * E)
        Y = T + se[None, :] * U
        s = spearman_rows(xr, Y)
        power.append(np.mean(p_two_sided_spearman(s, n) < 0.05))
        cdf.append(np.mean(s <= s_obs))
        mean_s.append(s.mean())
    power, cdf, mean_s = map(np.array, (power, cdf, mean_s))
    # MDE in the theory's direction (two-sided 5% test, 80% power)
    pos = grid >= 0 if direction > 0 else grid <= 0
    gd = grid[pos] if direction > 0 else grid[pos][::-1]
    pw = power[pos] if direction > 0 else power[pos][::-1]
    mde_t = interp_cross(gd, pw, 0.80)
    pow_mod = float(np.interp(direction * MODERATE, grid, power))
    pow_large = float(np.interp(direction * LARGE, grid, power))
    # confidence bounds by inversion: upper = smallest r with P(S <= s_obs | r) <= 0.05, etc.
    up = interp_cross(grid, cdf, 0.05)          # cdf decreases in r
    lo = interp_cross(grid, cdf, 0.95)
    cens_up, cens_lo = not np.isfinite(up), not np.isfinite(lo)
    up = up if np.isfinite(up) else (grid[-1] if cdf[-1] > 0.05 else grid[0])
    lo = lo if np.isfinite(lo) else (grid[0] if cdf[0] < 0.95 else grid[-1])
    up95 = interp_cross(grid, cdf, 0.025); lo95 = interp_cross(grid, cdf, 0.975)
    bound_t = up if direction > 0 else lo       # one-sided 95% bound in the theory's direction
    return dict(n=n, estimate=s_obs, p=float(p_obs), reliability=float(rel),
                ci90_obs_lo=float(a90[0]), ci90_obs_hi=float(a90[1]), bound_obs=float(bound_a),
                mde_obs=mde_a, ci90_true_lo=float(lo), ci90_true_hi=float(up),
                ci95_true_lo=float(lo95) if np.isfinite(lo95) else np.nan,
                ci95_true_hi=float(up95) if np.isfinite(up95) else np.nan,
                bound_true=float(bound_t), mde_true=float(mde_t) if np.isfinite(mde_t) else np.nan,
                power_moderate=pow_mod, power_large=pow_large,
                tost_bound_true=float(max(abs(lo), abs(up))),
                censored=bool(cens_up if direction > 0 else cens_lo),
                curve=pd.DataFrame(dict(effect=grid, power=power, cdf=cdf, mean_obs=mean_s)))


def classify(bound, direction, thr_mod, thr_large):
    """bound = one-sided 95% bound in the theory's direction (same units as the thresholds)."""
    b = direction * bound
    if b < thr_mod:
        return "informative (rules out moderate)"
    if b < thr_large:
        return "rules out large only"
    return "underpowered"


def moran_null():
    gap, net, rel, rel14, missing = _s51.load()
    nodes = rel14.field.tolist()
    x = rel14.spearman.to_numpy(float)
    se = ((rel14.ci_hi - rel14.ci_lo) / 3.92).to_numpy(float)
    W, iso, _ = _s51.build_W_flow(net, nodes)
    n = len(x); S0 = W.sum()
    I_obs = float(_s51.morans_I(x, W))
    assert abs(I_obs - 0.010789263252700108) < 1e-9, "Moran's I of scripts/51 did not reproduce"
    rng = rng_for("moran_obs")
    perms = np.array([rng.permutation(n) for _ in range(N_PERM_OBS)])

    def I_rows(Xm):
        Zc = Xm - Xm.mean(-1, keepdims=True)
        return (n / S0) * np.einsum("...i,ij,...j->...", Zc, W, Zc) / (Zc ** 2).sum(-1)
    null = I_rows(x[perms])
    p_two = (1 + np.sum(np.abs(null) >= abs(I_obs))) / (1 + len(null))
    p_right = (1 + np.sum(null >= I_obs)) / (1 + len(null))
    var_obs = x.var(ddof=1); noise = np.mean(se ** 2)
    sT = np.sqrt(max(var_obs - noise, 1e-6))
    Iden = np.eye(n)

    def latent(lam, E):
        M = np.linalg.inv(Iden - lam * W)
        Sig = M @ M.T
        sc = np.sqrt(np.mean(np.diag(Sig)))
        return sT * (E @ M.T) / sc
    grid = np.round(np.arange(-0.9, 0.9501, 0.025), 4)
    rng = rng_for("moran_inv")
    E = rng.standard_normal((N_SIM, n)); U = rng.standard_normal((N_SIM, n))
    cdf, kin, I_lat = [], [], []
    for lam in grid:
        T = latent(lam, E)
        Y = T + se[None, :] * U
        cdf.append(np.mean(I_rows(Y) <= I_obs))
        WT = T @ W.T
        Tc = T - T.mean(1, keepdims=True); Wc = WT - WT.mean(1, keepdims=True)
        kin.append(np.mean((Tc * Wc).sum(1) / np.sqrt((Tc ** 2).sum(1) * (Wc ** 2).sum(1))))
        I_lat.append(np.mean(I_rows(T)))
    cdf, kin, I_lat = map(np.array, (cdf, kin, I_lat))
    # power of the right-tail permutation test (as in scripts/51), coarser grid
    pgrid = np.round(np.arange(0.0, 0.9501, 0.05), 4)
    rng = rng_for("moran_power")
    E2 = rng.standard_normal((N_SIM_MORAN, n)); U2 = rng.standard_normal((N_SIM_MORAN, n))
    pp = np.array([rng.permutation(n) for _ in range(N_PERM_MORAN)])
    power = []
    for lam in pgrid:
        Y = latent(lam, E2) + se[None, :] * U2
        Io = I_rows(Y)
        In = I_rows(Y[:, pp])                                  # (sims, perms)
        pr = (1 + (In >= Io[:, None] - 1e-12).sum(1)) / (1 + N_PERM_MORAN)
        power.append(np.mean(pr < 0.05))
    power = np.array(power)
    kin_p = np.interp(pgrid, grid, kin)
    lam_mod = interp_cross(grid[grid >= 0], kin[grid >= 0], MODERATE)
    lam_large = interp_cross(grid[grid >= 0], kin[grid >= 0], LARGE)
    lam_mde = interp_cross(pgrid, power, 0.80)
    kin0 = float(np.interp(0.0, grid, kin))
    Cn = Iden - np.ones((n, n)) / n

    def centred_share(lam):
        """share of the latent SAR variance that is not common to all fields (what Moran's I can see)"""
        M = np.linalg.inv(Iden - lam * W); Sg = M @ M.T
        return float(np.trace(Cn @ Sg @ Cn) / np.trace(Sg))
    eff_nb = 1 / (W ** 2).sum(1)
    lam_mod_x = interp_cross(grid[grid >= 0], kin[grid >= 0] - kin0, MODERATE)   # +0.30 above the lambda = 0 baseline
    lam_up = interp_cross(grid, cdf, 0.05)
    lam_lo = interp_cross(grid, cdf, 0.95)
    kin_at = lambda l: float(np.interp(l, grid, kin)) if np.isfinite(l) else np.nan
    censored = not np.isfinite(lam_up)          # P(I <= I_obs | lambda) never falls below 0.05 on the grid
    kin_up = kin_at(lam_up) if not censored else float(kin.max())
    jmax = int(np.argmax(power))
    return dict(n=n, estimate=I_obs, p=float(p_two), p_right=float(p_right), reliability=float(1 - noise / var_obs),
                lam_mod=lam_mod, lam_large=lam_large, lam_mde=lam_mde, lam_up=lam_up, lam_lo=lam_lo,
                kin_mde=kin_at(lam_mde), kin_up=kin_up, kin_lo=kin_at(lam_lo), censored=censored,
                max_power=float(power[jmax]), lam_max_power=float(pgrid[jmax]), kin_max_power=float(kin_p[jmax]),
                min_cdf=float(cdf.min()), lam_min_cdf=float(grid[int(np.argmin(cdf))]),
                kin0=kin_at(0.0),
                power_moderate=float(np.interp(lam_mod, pgrid, power)) if np.isfinite(lam_mod) else np.nan,
                power_large=float(np.interp(lam_large, pgrid, power)) if np.isfinite(lam_large) else np.nan,
                lam_mod_x=lam_mod_x, cshare0=centred_share(0.0), cshare95=centred_share(0.95),
                eff_nb_lo=float(eff_nb.min()), eff_nb_hi=float(eff_nb.max()),
                nnz_lo=int((W > 0).sum(1).min()), nnz_hi=int((W > 0).sum(1).max()),
                power_moderate_x=float(np.interp(lam_mod_x, pgrid, power)) if np.isfinite(lam_mod_x) else np.nan,
                I_at_mod=float(np.interp(lam_mod, grid, I_lat)), iso=iso,
                curve=pd.DataFrame(dict(effect=kin_p, lam=pgrid, power=power)),
                inv=pd.DataFrame(dict(lam=grid, kin=kin, cdf=cdf, I_latent=I_lat)))


def career_null():
    d = pd.read_csv(CAREER)
    fs = d[(d.record == "field_slope") & (d.source == "pseo_fixed_cohort")].copy()
    sm_ = d[d.record == "summary"].set_index("stat")
    est = float(sm_.loc["slope_int_minus_others", "estimate"])
    lo, hi = float(sm_.loc["slope_int_minus_others", "ci_lo"]), float(sm_.loc["slope_int_minus_others", "ci_hi"])
    p = float(sm_.loc["slope_int_minus_others", "p_value"])
    all_slope = float(sm_.loc["slope_all", "estimate"])
    isA = (fs.group == "integrated").to_numpy()
    s = fs.slope_per_yr.to_numpy(float)
    est_re = s[isA].mean() - s[~isA].mean()
    assert abs(est_re - est) < 1e-6, "career-time contrast did not reproduce"
    se_boot = (hi - lo) / (2 * 1.96)
    se_welch = float(np.sqrt(s[isA].var(ddof=1) / isA.sum() + s[~isA].var(ddof=1) / (~isA).sum()))
    thr_mod, thr_large = 0.5 * all_slope, 1.0 * all_slope
    # simulated power of scripts/52's field-label permutation test (two-sided)
    resid = np.where(isA, s - s[isA].mean(), s - s[~isA].mean())
    nA, n = int(isA.sum()), len(s)
    rng = rng_for("career_power")
    lab = np.zeros((N_PERM_CT, n))
    for i in range(N_PERM_CT):
        m = np.zeros(n, bool); m[rng.choice(n, nA, replace=False)] = True
        lab[i] = np.where(m, 1 / nA, -1 / (n - nA))
    base_lab = np.where(np.arange(n) < nA, 1 / nA, -1 / (n - nA))
    Rd = resid[rng.integers(n, size=(N_SIM_CT, n))]
    grid = np.round(np.arange(0.0, 0.0601, 0.0025), 5)
    power = []
    for dl in grid:
        Y = Rd + dl * (np.arange(n) < nA)[None, :]
        obs = Y @ base_lab
        nul = Y @ lab.T
        pr = (1 + (np.abs(nul) >= np.abs(obs)[:, None] - 1e-12).sum(1)) / (1 + N_PERM_CT)
        power.append(np.mean(pr < 0.05))
    power = np.array(power)
    mde_sim = interp_cross(grid, power, 0.80)
    return dict(n=f"{nA} vs {n - nA}", estimate=est, p=p, ci95_lo=lo, ci95_hi=hi, se_boot=se_boot,
                se_welch=se_welch, all_slope=all_slope, thr_mod=thr_mod, thr_large=thr_large,
                ci90_lo=est - 1.645 * se_boot, ci90_hi=est + 1.645 * se_boot,
                bound=est + 1.645 * se_boot, mde_analytic=(1.96 + Z80) * se_boot, mde_sim=mde_sim,
                power_moderate_analytic=float(norm.sf(1.96 - thr_mod / se_boot) + norm.cdf(-1.96 - thr_mod / se_boot)),
                power_large_analytic=float(norm.sf(1.96 - thr_large / se_boot) + norm.cdf(-1.96 - thr_large / se_boot)),
                power_moderate_sim=float(np.interp(thr_mod, grid, power)),
                power_large_sim=float(np.interp(thr_large, grid, power)),
                curve=pd.DataFrame(dict(effect=grid, power=power)))


def cell_bridge(A, H, mk, sk, direction):
    """Cell-level version of a field-level null: moderator (per SD) x prestige in spec sk."""
    r = A[(A.spec == sk) & (A.moderator == mk)].iloc[0]
    tau0 = r.tau_null
    g, se = r.g, r.se
    to_r = lambda v: v / tau0
    bound = g + 1.645 * se if direction > 0 else g - 1.645 * se
    mde = (1.96 + Z80) * se
    pw = lambda thr: float(norm.sf(1.96 - thr * tau0 / se) + norm.cdf(-1.96 - thr * tau0 / se))
    return dict(n=int(r.k), estimate=to_r(g), g=g, se=se, p=r.p_wald, p_perm=r.p_perm, tau0=tau0,
                ci90_lo=to_r(g - 1.645 * se), ci90_hi=to_r(g + 1.645 * se), bound=to_r(bound),
                mde=to_r(mde) * direction, power_moderate=pw(MODERATE), power_large=pw(LARGE),
                tost=max(abs(to_r(g - 1.645 * se)), abs(to_r(g + 1.645 * se))))


def run_part_b(A, H):
    dp = pd.read_csv(DISP)
    gm = pd.read_csv(GAP_MAP).set_index("field")
    se_u = lambda fl: ((gm.loc[fl, "ci_hi"] - gm.loc[fl, "ci_lo"]) / 3.92).to_numpy(float)
    oc = pd.read_csv(OCC).dropna(subset=["rho_f"])
    res, curves = {}, {}
    for setname, sel in (("reliable", dp.reliable.astype(bool)), ("all", np.ones(len(dp), bool))):
        s = dp[sel.to_numpy() if hasattr(sel, "to_numpy") else sel]
        for meas in ["tail_hi", "sd", "iqr", "tail_lo", "p90_10", "cv_level"]:
            r = corr_null(s[meas], s.coupling, se_u(s.field), +1, f"disp_{meas}_{setname}")
            res[("B1", meas, setname)] = r
        o = oc[oc.reliable.astype(bool)] if setname == "reliable" else oc
        res[("B2", "occ_hhi", setname)] = corr_null(o.occ_hhi, o.rho_f, se_u(o.field), -1, f"occ_{setname}")
    res[("B3", "moran_flow", "reliable14")] = moran_null()
    res[("B4", "int_minus_others", "39 fields")] = career_null()
    for mk, dirn in (("tail_hi", +1), ("occ_hhi", -1)):
        for sk in ("A1", "A4"):
            res[("cell", mk, sk)] = cell_bridge(A, H, mk, sk, dirn)
    return res


# =============================================================================================
# outputs
# =============================================================================================
def power_table(res):
    rows = []
    lab = {("B1", "reliable"): "B1 dispersion law (tail_hi), reliable fields",
           ("B1", "all"): "B1 dispersion law (tail_hi), all fields",
           ("B2", "reliable"): "B2 occupation concentration (occ_hhi), reliable fields",
           ("B2", "all"): "B2 occupation concentration (occ_hhi), all fields"}
    for (blk, meas, st), r in res.items():
        if blk in ("B1", "B2"):
            dirn = +1 if blk == "B1" else -1
            rows.append(dict(test=blk, measure=meas, sample=st,
                             label=lab.get((blk, st), f"{blk} {meas}, {st}") if meas in ("tail_hi", "occ_hhi")
                             else f"B1 dispersion law ({meas}), {st} fields",
                             scale="Spearman r (true-score)", direction=dirn, n=r["n"], estimate=r["estimate"],
                             p=r["p"], reliability=r["reliability"], mde=r["mde_true"], mde_check=r["mde_obs"],
                             censored=r["censored"],
                             power_moderate=r["power_moderate"], power_large=r["power_large"],
                             ci90_lo=r["ci90_true_lo"], ci90_hi=r["ci90_true_hi"], bound=r["bound_true"],
                             bound_check=r["bound_obs"], tost=r["tost_bound_true"],
                             thr_mod=dirn * MODERATE, thr_large=dirn * LARGE,
                             verdict=classify(r["bound_true"], dirn, MODERATE, LARGE)))
        elif blk == "B3":
            rows.append(dict(test="B3", measure="Moran's I, flow kinship", sample=st,
                             label="B3 kinship Moran's I (flow W), 14 reliable fields",
                             scale="kin correlation corr(T, WT) under SAR", direction=1, n=r["n"],
                             estimate=r["estimate"], p=r["p"], reliability=r["reliability"],
                             mde=r["kin_mde"], power_moderate=r["power_moderate"], power_large=r["power_large"],
                             ci90_lo=r["kin_lo"], ci90_hi=r["kin_up"], bound=r["kin_up"], tost=np.nan,
                             censored=r["censored"], max_power=r["max_power"], kin_max_power=r["kin_max_power"],
                             thr_mod=MODERATE, thr_large=LARGE,
                             verdict=classify(r["kin_up"], +1, MODERATE, LARGE)))
        elif blk == "B4":
            rows.append(dict(test="B4", measure="integrated minus others", sample=st,
                             label="B4 career-time slope, integrated minus others",
                             scale="coupling per year", direction=1, n=r["n"], estimate=r["estimate"], p=r["p"],
                             reliability=np.nan, mde=r["mde_analytic"], mde_check=r["mde_sim"], censored=False,
                             se=r["se_boot"], se_check=r["se_welch"],
                             power_moderate=r["power_moderate_analytic"], power_large=r["power_large_analytic"],
                             ci90_lo=r["ci90_lo"], ci90_hi=r["ci90_hi"], bound=r["bound"],
                             tost=max(abs(r["ci90_lo"]), abs(r["ci90_hi"])),
                             thr_mod=r["thr_mod"], thr_large=r["thr_large"],
                             verdict=classify(r["bound"], +1, r["thr_mod"], r["thr_large"])))
        elif blk == "cell":
            dirn = +1 if meas == "tail_hi" else -1
            rows.append(dict(test="cell", measure=meas, sample=st,
                             label=f"cell-level {meas} x prestige ({st})",
                             scale="g / tau_P (true-score r)", direction=dirn, n=r["n"], estimate=r["estimate"],
                             p=r["p"], p_perm=r["p_perm"], reliability=np.nan, mde=r["mde"], censored=False,
                             power_moderate=r["power_moderate"], power_large=r["power_large"],
                             ci90_lo=r["ci90_lo"], ci90_hi=r["ci90_hi"], bound=r["bound"], tost=r["tost"],
                             thr_mod=dirn * MODERATE, thr_large=dirn * LARGE,
                             verdict=classify(r["bound"], dirn, MODERATE, LARGE)))
    return pd.DataFrame(rows)


def write_csv(a, PT):
    A, H, FL, BT = a["A"], a["H"], a["FL"], a["BT"]
    parts = [A.assign(record="cell_interaction"), H.assign(record="heterogeneity"),
             FL.assign(record="field_level"), BT.assign(record="bootstrap_raw_vs_controlled"),
             a["RB"].assign(record="robustness_drop_fields"), a["FS"].assign(record="field_slopes"),
             a["RC"].assign(record="reml_optimizer_check"), a["LV"].assign(record="leverage_moderator_forms"),
             a["SPR"].assign(record="leverage_field_spearman"), a["D2"].assign(record="leverage_drop_nursing_special_ed"),
             a["FT"].assign(record="field_type_fits"), a["FTS"].assign(record="field_type_subset_spearman"),
             a["FTB"].assign(record="field_type_subset_bootstrap"), a["FTC"].assign(record="field_type_collinearity"),
             PT.assign(record="power_equivalence")]
    out = pd.concat(parts, ignore_index=True, sort=False)
    front = ["record", "spec", "spec_label", "moderator", "mod_label", "form", "control", "method", "scheme", "term",
             "drop", "field", "test", "measure", "sample", "label"]
    out = out[[c for c in front if c in out.columns] + [c for c in out.columns if c not in front]]
    out.to_csv(OUT_CSV, index=False, float_format="%.6g")


def make_figure(a, res, PT):
    A, BT, blups = a["A"], a["BT"], a["blups"]
    fig = plt.figure(figsize=(15, 25))
    gs = fig.add_gridspec(4, 2, hspace=0.42, wspace=0.55)
    ax = np.empty((4, 2), dtype=object)
    for i in range(4):
        for j in range(2):
            if (i, j) != (2, 1):
                ax[i, j] = fig.add_subplot(gs[i, j])
    sub_f = gs[2, 1].subgridspec(1, 2, wspace=0.3, width_ratios=[1, 1])
    f_ns, f_log = fig.add_subplot(sub_f[0]), fig.add_subplot(sub_f[1])
    # (a) licensing x status slope across specs
    specs = ["A1", "A2", "A3", "A4", "A4B", "A4E", "A5", "A6", "A7", "C1", "C2"]
    mods = ["lic", "acs_strict", "acs_strict_ba"]
    col = {"lic": C_ORANGE, "acs_strict": C_BLUE, "acs_strict_ba": C_AQUA}
    main_of = {sk: SPECS[sk]["slopes"][0] for sk in specs}
    for j, mk in enumerate(mods):
        sub = A[(A.moderator == mk) & (A.spec.isin(specs))]
        sub = sub[sub.apply(lambda r: r.slope == main_of[r.spec], axis=1)].set_index("spec").reindex(specs)
        y = np.arange(len(specs)) + (j - 1) * 0.25
        ax[0, 0].errorbar(sub.g, y, xerr=[sub.g - sub.lo, sub.hi - sub.g], fmt="o", ms=4, color=col[mk],
                          capsize=2, label=MODS.get(mk, MODS_BA.get(mk)) + (" (per unit share)" if mk != "lic" else ""))
    FL = a["FL"]
    for j, mk in enumerate(["lic", "acs_strict"]):
        r = FL[(FL.spec == "A1") & (FL.moderator == mk) & FL.method.str.startswith("gap-map")].iloc[0]
        ax[0, 0].errorbar(r.g, len(specs) + (j - 1) * 0.25, xerr=[[r.g - r.lo], [r.hi - r.g]], fmt="s", ms=4,
                          color=col[mk], mfc="white", capsize=2)
    ax[0, 0].axvline(0, color="k", lw=0.6)
    ax[0, 0].axvline(-PUBLISHED_LIC["b_gap"], color=C_BLUE, lw=0.8, ls=":")
    ax[0, 0].set_yticks(list(range(len(specs) + 1)))
    ax[0, 0].set_yticklabels([f"{s_}: {SPECS[s_]['label']}" for s_ in specs] + ["field level: gap-map rho_f (WLS)"],
                             fontsize=7)
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_xlabel("moderator x slope (change in within-field slope), 95% Wald CI (too narrow between fields;\n"
                        "use the permutation p, table b0); squares: field level", fontsize=8)
    ax[0, 0].set_title("(a) licensing moderation of the prestige slope (C1/C2: SAT / brand slope)\n"
                       "dotted: scripts/20 field-level value (-0.66 per unit strict share)", fontsize=9)
    ax[0, 0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=1, frameon=False)
    # (b) field slopes, raw vs selectivity-controlled (BLUPs on the selectivity sample)
    b3, b4 = blups["A3"], blups["A4"]
    fl = b3.index.intersection(b4.index)
    lic = [f in F.LICENSED_FIELDS for f in fl]
    ax[0, 1].scatter(b3[fl], b4[fl], s=26, c=np.where(lic, C_ORANGE, C_GREY), edgecolor="white", lw=0.6, zorder=3)
    offs = {"civil_engineering": (4, -9), "accounting": (-10, 6), "nursing": (4, 4), "economics": (4, -8),
            "computer_science": (-40, 6), "biology": (4, -8)}
    for f in fl:
        if f in offs:
            ax[0, 1].annotate(LAB.get(f, f), (b3[f], b4[f]), fontsize=7, xytext=offs[f], textcoords="offset points",
                              color="#52514e")
    lim = [min(b3.min(), b4.min()) - 0.05, max(b3.max(), b4.max()) + 0.05]
    ax[0, 1].plot(lim, lim, color="k", lw=0.5, ls=":")
    ax[0, 1].axhline(0, color="k", lw=0.5); ax[0, 1].axvline(0, color="k", lw=0.5)
    ax[0, 1].set_xlabel("field prestige slope, raw (A3; shrunken estimate)", fontsize=8)
    ax[0, 1].set_ylabel("field prestige slope, selectivity-controlled (A4; shrunken)", fontsize=8)
    ax[0, 1].set_title(f"(b) field slopes before / after selectivity controls, {len(fl)} fields\n"
                       "orange = ex-ante licensed; dotted = no change", fontsize=9)
    # (c) power curves on a common scale: effect / moderate threshold
    c = ax[1, 0]
    for (blk, meas, st), r in res.items():
        if blk == "B1" and meas == "tail_hi":
            cv = r["curve"]; m = cv.effect >= 0
            c.plot(cv.effect[m] / MODERATE, cv.power[m], label=f"B1 dispersion, {st} (k={r['n']})",
                   color=C_BLUE, ls="-" if st == "all" else "--")
        if blk == "B2":
            cv = r["curve"]; m = cv.effect <= 0
            c.plot(-cv.effect[m] / MODERATE, cv.power[m], label=f"B2 occupation conc., {st} (k={r['n']})",
                   color=C_ORANGE, ls="-" if st == "all" else "--")
        if blk == "B3":
            cv = r["curve"]
            c.plot(cv.effect / MODERATE, cv.power, label=f"B3 kinship Moran (k={r['n']})", color=C_AQUA)
        if blk == "B4":
            cv = r["curve"]
            c.plot(cv.effect / r["thr_mod"], cv.power, label="B4 career time, integrated - others (1 = half the slope)",
                   color=C_YELLOW)
    c.axhline(0.8, color="k", lw=0.5, ls=":"); c.axvline(1, color="k", lw=0.6)
    c.set_xlim(0, 3); c.set_ylim(0, 1)
    c.set_xlabel("true effect / moderate effect (large = 1.67; B4: 2)", fontsize=8)
    c.set_ylabel("power (two-sided 5% test as used)")
    c.set_title("(c) power of the project's field-level null tests", fontsize=9)
    c.legend(fontsize=7, loc="lower right")
    # (d) equivalence: one-sided 95% bound in the theory's direction, in units of the moderate effect
    d = ax[1, 1]
    rows = PT[PT.measure.isin(["tail_hi", "occ_hhi", "Moran's I, flow kinship", "integrated minus others"])]
    rows = rows.reset_index(drop=True)
    for i, r in rows.iterrows():
        sc = abs(r.thr_mod)
        est = r.direction * (r.estimate if r.test != "B3" else np.nan) / sc
        lo, hi = sorted([r.direction * r.ci90_lo / sc, r.direction * r.ci90_hi / sc])
        colr = {"informative (rules out moderate)": S_GOOD, "rules out large only": S_WARN,
                "underpowered": S_CRIT}[r.verdict]
        d.plot([lo, hi], [i, i], color=colr, lw=2.5)
        d.plot(r.direction * r.bound / sc, i, marker="|", color="k", ms=12)
        d.plot(abs(r.thr_large) / sc, i, marker="d", color="#555555", ms=5)
        if np.isfinite(est):
            d.plot(est, i, "o", color=colr)
        d.annotate(r.verdict.split(" (")[0], (1.0, i), xycoords=("axes fraction", "data"), xytext=(4, 0),
                   textcoords="offset points", va="center", fontsize=6.5, color="#52514e")
    d.axvline(1, color="k", lw=0.6)
    d.axvline(0, color="k", lw=0.4, ls=":")
    d.set_yticks(range(len(rows))); d.set_yticklabels(rows.label, fontsize=7)
    d.invert_yaxis()
    d.set_xlabel("effect in the theory's direction / moderate effect", fontsize=8)
    d.set_title("(d) 90% intervals; | = one-sided 95% bound; diamond = large effect\n"
                "verdict at right (B1-B3 on the true-score scale)", fontsize=9)
    # (e) leverage: per-field controlled slope vs the bachelor's-only share
    e = ax[2, 0]
    FS = a["FS"].set_index("field")
    sp_ = a["SPR"].set_index(["spec", "moderator"])
    T = FS[["acs_strict_ba", "b_A4B", "se_A4B"]].dropna()
    hi2 = T.index.isin(DROP2)
    e.errorbar(T.acs_strict_ba[~hi2], T.b_A4B[~hi2], yerr=1.96 * T.se_A4B[~hi2], fmt="o", ms=4, color=C_GREY,
               elinewidth=0.5, capsize=0, zorder=3, label="other fields")
    e.errorbar(T.acs_strict_ba[hi2], T.b_A4B[hi2], yerr=1.96 * T.se_A4B[hi2], fmt="o", ms=6, color=C_ORANGE,
               elinewidth=0.8, capsize=0, zorder=4, label="nursing, special education")
    for f in T.index[hi2]:
        e.annotate(LAB.get(f, f), (T.acs_strict_ba[f], T.b_A4B[f]), fontsize=7, xytext=(-6, 8),
                   textcoords="offset points", ha="right", color="#52514e")
    xs = np.linspace(0, T.acs_strict_ba.max(), 50)
    for msk, colr, ls, lab_ in ((np.ones(len(T), bool), C_GREY, "--", "OLS line, all fields"),
                                (~hi2, C_BLUE, "-", "OLS line without the two")):
        c_ = np.polyfit(T.acs_strict_ba[msk], T.b_A4B[msk], 1)
        xx = xs if msk.all() else xs[xs <= T.acs_strict_ba[msk].max()]
        e.plot(xx, np.polyval(c_, xx), color=colr, ls=ls, lw=1.2, label=f"{lab_} (slope {c_[0]:+.2f})")
    e.axhline(0, color="k", lw=0.5)
    r_ = sp_.loc[("A4B", "acs_strict_ba")]
    e.set_xlabel("ACS strict licensed share, bachelor's-only holders", fontsize=8)
    e.set_ylabel("per-field OLS prestige slope, selectivity + brand controlled (A4B), 95% CI", fontsize=8)
    e.set_title(f"(e) two fields carry the linear fit ({len(T)} fields)\nfield-level Spearman "
                f"{r_.spearman:+.2f} (perm. p {fp_(r_.p_perm)}); without the two {r_.spearman_drop2:+.2f} "
                f"(p {fp_(r_.p_perm_drop2)})", fontsize=9)
    e.legend(fontsize=7, loc="lower left", frameon=False)
    # (f) moderator forms: g per SD of the (transformed) share; filled = permutation p < 0.05
    LV = a["LV"][a["LV"].slope == "P"]
    colm = {"acs_strict": C_BLUE, "acs_strict_ba": C_AQUA, "acs_broad": C_YELLOW}
    labm = {"acs_strict": "strict, all-degree", "acs_strict_ba": "strict, bachelor's-only", "acs_broad": "broad, all-degree"}
    for axf, sks, unit in ((f_ns, ("A3", "A4", "A4B", "A4E", "A7"), "normal-score slope"),
                           (f_log, ("L3", "L4", "L4B", "L5"), "log points per SD of prestige")):
        for i, sk in enumerate(sks):
            for j, mk in enumerate(LEV_MODS):
                s_ = LV[(LV.spec == sk) & (LV.moderator == mk)].set_index("form")
                y_ = i + (j - 1) * 0.27
                rob = s_.drop("linear")
                axf.plot([rob.g.min(), rob.g.max()], [y_, y_], color=colm[mk], lw=1.2, alpha=0.7)
                for fn, mkr, ms in (("linear", "o", 5), ("rank normal score", "D", 5)):
                    r = s_.loc[fn]
                    axf.plot(r.g, y_, marker=mkr, ms=ms, color=colm[mk], ls="none",
                             mfc=colm[mk] if r.p_perm < 0.05 else "white", mew=1.2,
                             label=(f"{labm[mk]}: {fn}" if i == 0 else None))
        axf.axvline(0, color="k", lw=0.6)
        axf.set_yticks(range(len(sks))); axf.set_yticklabels(sks, fontsize=8)
        axf.invert_yaxis()
        axf.set_xlabel(f"g per SD of the share\n({unit})", fontsize=8)
    f_ns.set_title("(f) moderation of the prestige slope by form of the share\n"
                   "o linear, D rank normal score, line = range of the 4 leverage-resistant forms;\n"
                   "filled = field-label permutation p < 0.05", fontsize=9, loc="left")
    f_ns.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(1.0, -0.2), ncol=2, frameon=False)
    # (g) field-type check: rank-form moderation of the prestige slope under field-type controls / within subsets
    g_ = ax[3, 0]
    FTP = a["FT"][a["FT"].slope == "P"]
    RN = "rank normal score"
    conds = [("none", "all fields with a share", "no field-type control"),
             ("bloc", "all fields with a share", "+ eng/CS/math/business dummy"),
             ("cip2_fe", "all fields with a share", "within CIP-2 (stratified)"),
             ("none", "non-bloc fields", "non-bloc fields only"),
             ("none", "bloc fields", "eng/CS/math/business fields only")]
    ylab = []
    for i, (sk, (cs, smp, lab_)) in enumerate([(sk, c) for sk in ("A4", "A4B") for c in conds]):
        ylab.append(f"{sk}: {lab_}")
        for j, mk in enumerate(LEV_MODS):
            r = FTP[(FTP.spec == sk) & (FTP.moderator == mk) & (FTP.control == cs) & (FTP.form == RN)
                    & (FTP["sample"] == smp)].iloc[0]
            y_ = i + (j - 1) * 0.25
            g_.plot([r.lo, r.hi], [y_, y_], color=colm[mk], lw=1.2)
            g_.plot(r.g, y_, marker=("o", "s", "D")[j], ms=5, color=colm[mk], ls="none",
                    mfc=colm[mk] if r.p_perm < 0.05 else "white", mew=1.2, label=labm[mk] if i == 0 else None)
    g_.axhline(len(conds) - 0.5, color="#c9c8c4", lw=0.8)
    g_.axvline(0, color="k", lw=0.6)
    g_.set_yticks(range(len(ylab))); g_.set_yticklabels(ylab, fontsize=7.5)
    g_.invert_yaxis()
    g_.set_xlabel("change in the prestige slope per SD of the share (rank normal score), 95% Wald CI", fontsize=8)
    g_.set_title("(g) licensed-share moderation vs a field-type contrast (A4: selectivity; A4B: + brand)\n"
                 "filled = permutation p < 0.05 (Kennedy residual / within-CIP-2 / within-subset)", fontsize=9)
    g_.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=3, frameon=False)
    # (h) per-field controlled slopes vs the strict share, bloc vs other fields
    h_ = ax[3, 1]
    T = a["FS"].set_index("field")[["acs_strict", "b_A4B", "se_A4B", "bloc"]].dropna()
    for inb, colr, lab_ in ((1.0, C_ORANGE, "engineering, CS, math/statistics, business (CIP-2 11/14/27/52)"),
                            (0.0, C_GREY, "other fields")):
        t_ = T[T.bloc == inb]
        h_.errorbar(t_.acs_strict, t_.b_A4B, yerr=1.96 * t_.se_A4B, fmt="o", ms=5, color=colr, elinewidth=0.5,
                    capsize=0, zorder=3, label=f"{lab_} ({len(t_)}; Spearman {spearmanr(t_.acs_strict, t_.b_A4B)[0]:+.2f})")
    for f in ("nursing", "special_education", "mathematics", "statistics", "architecture", "music"):
        if f in T.index:
            h_.annotate(LAB.get(f, f), (T.acs_strict[f], T.b_A4B[f]), fontsize=7,
                        xytext={"music": (5, -11), "mathematics": (5, 6)}.get(f, (5, 4)),
                        textcoords="offset points", color="#52514e")
    h_.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    xt = [0.02, 0.05, 0.1, 0.2, 0.5]
    h_.xaxis.set_major_locator(FixedLocator(xt)); h_.xaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in xt]))
    h_.xaxis.set_minor_locator(NullLocator())
    h_.axhline(0, color="k", lw=0.5)
    h_.set_xlabel("ACS strict licensed share, all degree holders (log scale)", fontsize=8)
    h_.set_ylabel("per-field OLS prestige slope, selectivity + brand controlled (A4B), 95% CI", fontsize=8)
    h_.set_title(f"(h) the share separates field types ({len(T)} fields)\nSpearman of the share with the "
                 f"field-type dummy {spearmanr(T.acs_strict, T.bloc)[0]:+.2f}", fontsize=9)
    h_.legend(fontsize=7, loc="lower left", frameon=False)
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


# =============================================================================================
# RESULT.md
# =============================================================================================
PROV_FILES = [
    ("data/raw/scorecard_fos/Most-Recent-Cohorts-Field-of-Study.csv",
     "College Scorecard FoS, 'Most Recent' release updated 2026-06-10; "
     "https://ed-public-download.scorecard.network/downloads/Most-Recent-Cohorts-Field-of-Study_06102026.zip; "
     "accessed 2026-06-20 (SOURCES.md s.2)"),
    ("data/raw/scorecard_inst/Most-Recent-Cohorts-Institution.csv",
     "College Scorecard institution 'Most Recent Cohorts' file, build 2026-05-27; https://collegescorecard.ed.gov/data/ "
     "(Most-Recent-Cohorts-Institution zip); accessed 2026-09-23 (SOURCES.md, scripts/55 entry)"),
    ("data/raw/wapman2022/ranks.csv", "Wapman et al. 2022 ranks, Zenodo 10.5281/zenodo.6941651 v1 (2022-07-29); "
     "accessed 2026-06-20 (SOURCES.md s.1)"),
    ("data/raw/acs/psam_pusa.csv", "ACS PUMS 2023 1-year persons, part a; "
     "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_pus.zip; accessed 2026-06-20 (SOURCES.md s.9)"),
    ("data/raw/acs/psam_pusb.csv", "ACS PUMS 2023 1-year persons, part b; same archive (SOURCES.md s.9)"),
    ("outputs/expanded66_gap_map.csv", "gap map (scripts/08, 22)"),
    ("data/interim/mixture_anchors.csv", "ACS licensure anchors (scripts/20)"),
    ("data/interim/dispersion_law_fields.csv", "dispersion index (scripts/53)"),
    ("data/interim/occupation_channel_fields.csv", "occupation concentration (scripts/50)"),
    ("data/interim/career_time_coupling.csv", "career-time slopes (scripts/52)"),
    ("data/interim/faculty_flow_network.parquet", "faculty-flow kinship network (scripts/49)"),
]


def provenance():
    import hashlib
    rows = []
    for rel, what in PROV_FILES:
        fp = ROOT / rel
        h = hashlib.md5()
        with open(fp, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        rows.append(dict(file=rel, what=what, bytes=fp.stat().st_size, md5=h.hexdigest()))
    return pd.DataFrame(rows)


def f3(v, nd=3, sign=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def fp_(v):
    if not np.isfinite(v):
        return "n/a"
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def write_md(a, res, PT, prov):
    A, H, FL, BT, RB, FS, RC = a["A"], a["H"], a["FL"], a["BT"], a["RB"], a["FS"], a["RC"]
    Hs = H.set_index("spec")

    def ga(sk, mk, slope=None):
        sub = A[(A.spec == sk) & (A.moderator == mk)]
        if slope is not None:
            sub = sub[sub.slope == slope]
        return sub.iloc[0]

    def gfl(sk, mk, method):
        return FL[(FL.spec == sk) & (FL.moderator == mk) & FL.method.str.startswith(method)].iloc[0]

    def gbt(mk, sk, scheme="field-cluster"):
        return BT[(BT.moderator == mk) & (BT.spec == sk) & BT.scheme.str.startswith(scheme)].iloc[0]

    def lofo(sk, mk):
        g = RB[(RB.spec == sk) & (RB.moderator == mk) & (RB["drop"] != "nursing + communication disorders")]
        i, j = g.g.idxmin(), g.g.idxmax()
        return dict(lo=g.g.min(), lo_f=LAB.get(g.loc[i, "drop"], g.loc[i, "drop"]), hi=g.g.max(),
                    hi_f=LAB.get(g.loc[j, "drop"], g.loc[j, "drop"]), pmax=g.p_wald.max(), k=len(g),
                    tau_lo=g.tau.min(), tau_hi=g.tau.max(), nsig=int((g.p_wald < 0.05).sum()))

    def dropset(sk, mk):
        return RB[(RB.spec == sk) & (RB.moderator == mk) & (RB["drop"] == "nursing + communication disorders")].iloc[0]

    P = {(r.test, r.measure, r.sample): r for r in PT.itertuples()}
    b1r, b1a = P[("B1", "tail_hi", "reliable")], P[("B1", "tail_hi", "all")]
    b2r, b2a = P[("B2", "occ_hhi", "reliable")], P[("B2", "occ_hhi", "all")]
    b3 = res[("B3", "moran_flow", "reliable14")]; b3r = P[("B3", "Moran's I, flow kinship", "reliable14")]
    b4 = res[("B4", "int_minus_others", "39 fields")]; b4r = P[("B4", "integrated minus others", "39 fields")]
    c1a, c1b = P[("cell", "tail_hi", "A1")], P[("cell", "tail_hi", "A4")]
    c2a, c2b = P[("cell", "occ_hhi", "A1")], P[("cell", "occ_hhi", "A4")]
    b1_all6 = PT[(PT.test == "B1") & (PT["sample"] == "all")]
    b1_rel6 = PT[(PT.test == "B1") & (PT["sample"] == "reliable")]

    l1, l3, l4, l4b, l4e, l5, l6 = (ga(k, "lic") for k in ("A1", "A3", "A4", "A4B", "A4E", "A5", "A6"))
    l7 = ga("A7", "lic", "P")
    s1, s2, s3, s4, s4b, s4e, s5, s6 = (ga(k, "acs_strict") for k in ("A1", "A2", "A3", "A4", "A4B", "A4E", "A5", "A6"))
    ba1, ba3, ba4, ba4b, ba4e, ba5 = (ga(k, "acs_strict_ba") for k in ("A1", "A3", "A4", "A4B", "A4E", "A5"))
    h3, h4, h4b = ga("A3", "acs_health_ba"), ga("A4", "acs_health_ba"), ga("A4B", "acs_health_ba")
    k3, k4, k4b = ga("A3", "acs_k12_ba"), ga("A4", "acs_k12_ba"), ga("A4B", "acs_k12_ba")
    a7p, a7s, a7g = (ga("A7", "acs_strict", sl) for sl in ("P", "SAT", "G"))
    a7bp, a7bs, a7bg = (ga("A7", "acs_strict_ba", sl) for sl in ("P", "SAT", "G"))
    a7hp, a7kp = ga("A7", "acs_health_ba", "P"), ga("A7", "acs_k12_ba", "P")
    c1s, c2s = ga("C1", "acs_strict"), ga("C2", "acs_strict")
    c1ba, c2ba = ga("C1", "acs_strict_ba"), ga("C2", "acs_strict_ba")
    c1l, c2l = ga("C1", "lic"), ga("C2", "lic")
    c1Ls, c2Ls = ga("C1L", "acs_strict"), ga("C2L", "acs_strict")
    c1Lb, c2Lb = ga("C1L", "acs_strict_ba"), ga("C2L", "acs_strict_ba")
    gl3s, gl4s, gl5s = (ga(k, "acs_strict") for k in ("L3", "L4", "L5"))
    gl3b, gl4b, gl5b = (ga(k, "acs_strict_ba") for k in ("L3", "L4", "L5"))
    gl1s = ga("L1", "acs_strict")
    fl_wls, fl_2s = gfl("A1", "acs_strict", "gap-map"), gfl("A1", "acs_strict", "two-step")
    fl_2s4, fl_2s4b = gfl("A4", "acs_strict", "two-step"), gfl("A4B", "acs_strict", "two-step")
    fl_2b4b = gfl("A4B", "acs_strict_ba", "two-step")
    fl_wls_l = gfl("A1", "lic", "gap-map")
    fb = lambda mk, sk: gbt(mk, sk, "field")
    ib = lambda mk, sk: gbt(mk, sk, "institution")
    lo_s4, lo_b4, lo_s1, lo_b1 = lofo("A4", "acs_strict"), lofo("A4", "acs_strict_ba"), lofo("A1", "acs_strict"), lofo("A1", "acs_strict_ba")
    ds_l1, ds_s4, ds_b4, ds_b1 = dropset("A1", "lic"), dropset("A4", "acs_strict"), dropset("A4", "acs_strict_ba"), dropset("A1", "acs_strict_ba")
    ds_s4b, ds_b4b = dropset("A4B", "acs_strict"), dropset("A4B", "acs_strict_ba")
    rc = RC.set_index(["spec", "moderator", "drop"])
    rc_s4 = rc.loc[("A4", "acs_strict", "")]
    rc_b4n = rc.loc[("A4", "acs_strict_ba", "nursing")]
    rc_a7 = rc.loc[("A7", "acs_strict", "")]
    rc_b4bn = rc.loc[("A4B", "acs_strict", " + ".join(f for f in DROP_SET if f in set(a["samples"]["sat"].field)))]
    fsi = FS.set_index("field")
    nurse_ba = fsi.loc["nursing", "acs_strict_ba"]; nurse_s = fsi.loc["nursing", "acs_strict"]
    sat_lic = sorted(set(a["samples"]["sat"].field.unique()) & F.LICENSED_FIELDS)
    pm = lambda r: f"{f3(r.g, 2)} (SE {r.se:.2f}; Wald p {fp_(r.p_wald)}; permutation p {fp_(r.p_perm)})"
    pp = lambda r: f"{f3(r.g, 2)} (permutation p {fp_(r.p_perm)})"
    pp3 = lambda r: f"{f3(r.g, 3)} (permutation p {fp_(r.p_perm)})"
    pm3 = lambda r: f"{f3(r.g, 3)} (SE {r.se:.3f}; Wald p {fp_(r.p_wald)}; permutation p {fp_(r.p_perm)})"
    wald_p = lambda g, se: float(2 * norm.sf(abs(g / se)))
    ci = lambda lo, hi, nd=2: f"[{f3(lo, nd)}, {f3(hi, nd)}]"
    val = a["val"]
    at0 = lambda v: v < 1e-4
    # ---- leverage checks (moderator forms, field-level Spearman, nursing + special education dropped) ----
    main_sl = lambda T: T[T.slope == T.spec.map(lambda k: SPECS[k]["slopes"][0])]     # first slope of each spec
    LVp, SPR, D2 = main_sl(a["LV"]), a["SPR"], main_sl(a["D2"])
    RNS, WIN = "rank normal score", f"winsorized at {WINS:.2f}"
    ROBF = [f for f in FORMS if f != "linear"]
    lv = lambda sk, mk, fn=RNS: LVp[(LVp.spec == sk) & (LVp.moderator == mk) & (LVp.form == fn)].iloc[0]
    spr = lambda sk, mk: SPR[(SPR.spec == sk) & (SPR.moderator == mk)].iloc[0]
    d2 = lambda sk, mk: D2[(D2.spec == sk) & (D2.moderator == mk)].iloc[0]
    lofo1 = lambda sk, mk, fd: RB[(RB.spec == sk) & (RB.moderator == mk) & (RB["drop"] == fd)].iloc[0]

    def rob(sk, mk, forms=None):
        s_ = LVp[(LVp.spec == sk) & (LVp.moderator == mk) & LVp.form.isin(forms or ROBF)]
        return dict(pmin=s_.p_perm.min(), pmax=s_.p_perm.max(), nsig=int((s_.p_perm < 0.05).sum()), n=len(s_),
                    gmin=s_.g.min(), gmax=s_.g.max())
    prr = lambda r: f"{f3(r.g, 3)} (permutation p {fp_(r.p_perm)})"
    psp = lambda r: f"{f3(r.spearman, 2)} (permutation p {fp_(r.p_perm)})"
    prange = lambda o: (f"permutation p {fp_(o['pmin'])} to {fp_(o['pmax'])}, {o['nsig']} of {o['n']} below 0.05"
                        if o["n"] > 1 else f"permutation p {fp_(o['pmin'])}")
    NSS, LOGS = ("A3", "A4", "A4B", "A4E", "A7"), ("L4", "L4B", "L5")
    rns_ns = LVp[LVp.spec.isin(NSS) & (LVp.form == RNS) & LVp.moderator.isin(STRICT2)]
    l3_forms = LVp[(LVp.spec == "L3") & LVp.moderator.isin(STRICT2)]
    l3_none = bool((l3_forms.p_perm >= 0.05).all()) and not any(spr("L3", mk).p_perm < 0.05 for mk in STRICT2)
    se_ba = fsi.loc["special_education", "acs_strict_ba"] if "special_education" in fsi.index else np.nan
    oth = fsi[fsi.index.isin(a["samples"]["sat"].field.unique()) & fsi.acs_strict_ba.notna()
              & ~fsi.index.isin(DROP2)]
    oth_rng = {mk: (oth[mk].min(), oth[mk].max()) for mk in STRICT2}
    fbr = lambda mk, sk: gbt(f"{mk}_rns", sk, "field")
    nf = lambda sk, mk: rob(sk, mk)["nsig"]

    def weak_forms(sk, mk):
        s_ = LVp[(LVp.spec == sk) & (LVp.moderator == mk) & LVp.form.isin(ROBF) & (LVp.p_perm >= 0.05)]
        return ", ".join(f"{r.form} p {fp_(r.p_perm)}" for r in s_.itertuples())

    def strong(sk, mk):
        """rank form, >= 3 of the 4 leverage-resistant forms, the field-level Spearman (where run) and the linear fit
        without nursing and special education all at permutation p < 0.05"""
        has_sp = bool(((SPR.spec == sk) & (SPR.moderator == mk)).any())
        return bool(lv(sk, mk).p_perm < 0.05 and nf(sk, mk) >= 3 and d2(sk, mk).p_perm < 0.05
                    and (not has_sp or spr(sk, mk).p_perm < 0.05))

    def claim(cond, what):
        """the prose below states these; a re-run that breaks one must fail rather than print a false sentence"""
        assert bool(cond), f"prose claim no longer holds: {what}"

    # ---- field-type check (field_type_checks) ----
    FTP = a["FT"][a["FT"].slope == a["FT"].spec.map(lambda k: SPECS[k]["slopes"][0])]
    FTS_, FTB_, FTC_ = a["FTS"], a["FTB"], a["FTC"].set_index("moderator")
    ALLF, NB, BF, BR = "all fields with a share", "non-bloc fields", "bloc fields", "acs_broad"

    def ft(sk, mk, cs="bloc", fn=RNS, smp=ALLF):
        return FTP[(FTP.spec == sk) & (FTP.moderator == mk) & (FTP.control == cs) & (FTP.form == fn)
                   & (FTP["sample"] == smp)].iloc[0]

    def ftalone(sk, mk="acs_strict"):
        return FTP[(FTP.spec == sk) & (FTP.moderator == "bloc") & (FTP.control == f"none (fields with {mk})")].iloc[0]
    fts = lambda smp, sk, mk: FTS_[(FTS_["sample"] == smp) & (FTS_.spec == sk) & (FTS_.moderator == mk)].iloc[0]
    ftbt = lambda smp, sk, mk: FTB_[(FTB_["sample"] == smp) & (FTB_.spec == sk) & (FTB_.moderator == mk)].iloc[0]
    pk = lambda r: f"{f3(r.g, 3)} (p {fp_(r.p_perm)})"                         # g (permutation p)
    pc = lambda r: f"{f3(r.cov_g, 3)} (p {fp_(r.cov_p_perm)})"                 # bloc coefficient (Kennedy p)
    fsi_sat = FS.set_index("field")
    fsi_sat = fsi_sat[fsi_sat.index.isin(set(a["samples"]["sat"].field.unique())) & fsi_sat.acs_strict.notna()]
    lead_bloc = {mk: int(np.argmax(fsi_sat.sort_values(mk).bloc.to_numpy() == 0)) for mk in STRICT2}
    bloc_ct = lambda sk, mk, fn=RNS: ft(sk, mk, "bloc", fn)
    FTN = [s_ for s_ in FT_SPECS if not s_.startswith("L")]          # normal-score specs of the field-type ladder
    cip_sig = {mk: [s_ for s_ in FTN if ft(s_, mk, "cip2_fe").p_perm < 0.05] for mk in LEV_MODS}
    nb_all = [ft(s_, mk, "none", fn, NB) for s_ in FTN for mk in STRICT2 for fn in ("linear", RNS)]
    br_ctrl = [ft(s_, BR, "none", fn) for s_ in ("A4", "A4B", "A7") for fn in FT_FORMS]
    br_a5 = {fn: ft("A5", BR, "none", fn) for fn in FT_FORMS}
    unl_A4 = float(fsi_sat.loc[fsi_sat.lic == 0, "b_A4"].mean())
    L = []
    L.append("# Cell-level hierarchical model of coupling, and power of the field-level nulls\n")
    L.append("Public data only (College Scorecard FoS + institution files, Wapman field and academia-wide ranks, ACS PUMS 2023, "
             "and the interim tables of scripts/20, 49-53). Descriptive, not causal. Every number below is produced by "
             "`scripts/63_hierarchical_power.py` (seed 63; single-threaded BLAS; byte-identical on re-run). Long table: "
             "`data/interim/hierarchical_power.csv` (column `record` separates the blocks); figure: "
             "`outputs/figures/hierarchical_power.png` (panels e-f: the leverage of the two high-share fields; g-h: the field-type check). Claims "
             "withdrawn from earlier local drafts of this file are listed under Revision notes at the end.\n")
    # ------------------------------------------------------------------ answer
    L.append("## Answer\n")
    pred = lambda r, x: r.b_main + r.g * x
    INF, LRG = "informative (rules out moderate)", "rules out large only"
    sv = lambda r: {INF: "informative", LRG: "rules out large only"}.get(r.verdict, "underpowered")

    def sig(r):
        if r.p_wald < 0.05 and r.p_perm < 0.05:
            return "significant by both tests"
        if r.p_wald < 0.05:
            return "Wald-significant but not by the permutation test"
        if r.p_perm < 0.05:
            return "significant by the permutation test only"
        return "not significant"
    versions = [(b1r, f"dispersion law, {b1r.n} reliable fields"), (b1a, f"dispersion law, all {b1a.n} fields (unweighted)"),
                (c1a, f"dispersion law, cell level ({c1a.n} fields)"),
                (c1b, f"dispersion law, cell level after selectivity controls ({c1b.n} fields)"),
                (b2r, f"occupation concentration, {b2r.n} reliable fields"),
                (b2a, f"occupation concentration, all {b2a.n} fields (unweighted)"),
                (c2a, f"occupation concentration, cell level ({c2a.n} fields)"),
                (c2b, f"occupation concentration, cell level after selectivity controls ({c2b.n} fields)"),
                (b3r, "kinship Moran's I"), (b4r, "career-time contrast")]
    ok2 = lambda r: r.p_perm < 0.05
    sat_fields = set(a["samples"]["sat"].field.unique())
    ba_top = (fsi[fsi.index.isin(sat_fields) & fsi.acs_strict_ba.notna()].acs_strict_ba
              .sort_values(ascending=False).head(4))
    ba_top_txt = ", ".join(f"{LAB.get(f, f).lower()} ({v:.2f})" for f, v in ba_top.items())
    unl_mean = float(fsi.loc[fsi.index.isin(sat_fields) & (fsi.lic == 0), "blup_A3"].mean())
    lic_slopes = ", ".join(f"{LAB[f]} {f3(fsi.loc[f, 'blup_A3'], 2)}" for f in sat_lic)
    het_A4 = Hs.loc["A4", "lrt_p_P"] < 0.05
    share_expl = 1 - (s4.tau / s4.tau_null) ** 2
    fb_s4, fb_s4b, fb_b4, fb_b4b = fb("acs_strict", "A4"), fb("acs_strict", "A4B"), fb("acs_strict_ba", "A4"), fb("acs_strict_ba", "A4B")
    fb_b4e, fb_s4e = fb("acs_strict_ba", "A4E"), fb("acs_strict", "A4E")
    strict_ladder_ok = all(ok2(r) for r in (s4, s4b, s4e, a7p))
    ba_ladder_ns = not any(ok2(r) for r in (ba4b, ba4e, a7bp))
    all_v = [r for r, _ in versions]
    mde_abs = [abs(r.mde) for r in all_v if r.test in ("B1", "B2", "cell")]
    pw_mod = [r.power_moderate for r in all_v]
    inf_rows = [(r, nm) for r, nm in versions if r.verdict == INF]
    claim(all(m > MODERATE for m in mde_abs) and not np.isfinite(b3r.mde) and b4["mde_analytic"] > b4["thr_mod"]
          and max(pw_mod) < 0.8, "no version of any null has 80% power at a moderate effect")
    claim(all(r.direction * r.estimate < 0 for r, _ in inf_rows), "informative versions: estimate opposite to the theory")
    claim(c1a.verdict != INF and c2a.verdict != INF, "cell-level versions do not exclude a moderate effect")
    claim(b3r.verdict == "underpowered" and b3["censored"], "kinship null unbounded")
    inf_txt = "; ".join(f"{nm}: Spearman {f3(r.estimate, 2)}, bound {f3(r.bound, 2)}, power at moderate "
                        f"{r.power_moderate:.0%}" for r, nm in inf_rows)
    tost_r = [r.tost for r in all_v if r.test in ("B1", "B2", "cell")]
    claim(min(tost_r) > MODERATE and b4r.tost > b4r.thr_mod and not np.isfinite(b3r.tost), "no TOST equivalence at the moderate margin")
    L.append(f"**Short answer.** (1) No version of any of the four nulls has 80% power against a moderate effect. The minimum "
             f"detectable true-score correlation is {min(mde_abs):.2f}-{max(mde_abs):.2f} (moderate = 0.30) for the dispersion "
             f"law and occupation concentration at field and cell level; the kinship Moran test never reaches 80% power (at "
             f"most {b3['max_power']:.0%} anywhere); the career-time contrast needs {b4['mde_analytic']:.4f}/yr against a "
             f"moderate {b4['thr_mod']:.4f}/yr. Power at the moderate effect is {min(pw_mod):.0%}-{max(pw_mod):.0%} across "
             f"the {len(all_v)} versions. Judged by equivalence instead (the one-sided 95% bound in the theory's direction "
             f"excludes the moderate effect), {len(inf_rows)} versions are informative: {inf_txt}. In "
             f"{'both' if len(inf_rows) == 2 else 'each'} the point estimate has the sign opposite to the theory, which is what "
             f"puts the bound below 0.30 despite low power, and the precision-weighted cell-level versions of the same two nulls "
             f"do not exclude a moderate effect (bounds {f3(c1a.bound, 2)} and {f3(c2a.bound, 2)}). By two-sided TOST no version "
             f"is equivalent to zero within the moderate margin: the smallest margin TOST at 5% accepts is "
             f"{min(tost_r):.2f}-{max(tost_r):.2f} on the correlation scale, {b4r.tost:.4f}/yr for the career-time contrast "
             f"({b4r.tost / b4r.thr_mod:.1f} times the moderate contrast), and none for kinship (table f). Verdict by null: dispersion "
             f"law (scripts/53): {sv(b1r)} on the {b1r.n} reliable fields, {sv(b1a)} on all {b1a.n} (unweighted), {sv(c1a)} at "
             f"the cell level ({c1a.n} fields), {sv(c1b)} there after selectivity controls ({c1b.n} fields). Occupation "
             f"concentration (scripts/50): {sv(b2r)} on the reliable fields, {sv(b2a)} on all fields, {sv(c2a)} at the cell "
             f"level, {sv(c2b)} after selectivity controls. Kinship Moran's I (scripts/51, {b3r.n} fields): {sv(b3r)}; no upper "
             f"bound can be set. Integrated-minus-others career-time slope (scripts/52): {sv(b4r)}; it excludes a contrast as "
             f"large as the all-field slope ({b4['thr_large']:.4f}/yr), not half of it. So none of the four nulls excludes a "
             f"moderate effect in every version, and the kinship null says nothing either way.")
    rs4b, rb4b = lv("A4B", "acs_strict"), lv("A4B", "acs_strict_ba")
    fr_s, fr_b = fbr("acs_strict", "A4B"), fbr("acs_strict_ba", "A4B")
    same_rank = abs(rs4b.g - rb4b.g) < max(rs4b.se, rb4b.se)
    ctrl_keep = all(0.8 <= fbr(mk, "A4B-A3").ratio <= 1.25 for mk in STRICT2)
    rl3 = {mk: lv("L3", mk) for mk in LEV_MODS}
    rl4 = {mk: lv("L4", mk) for mk in LEV_MODS}
    rl4b = {mk: lv("L4B", mk) for mk in LEV_MODS}
    n_se = int(fsi.loc["special_education", "n_A3"])
    S_, B_ = "acs_strict", "acs_strict_ba"
    claim(all(strong(sk, mk) for sk in ("A4", "A4B", "A4E", "A7") for mk in STRICT2), "both shares strong at A4-A7")
    claim(strong("L4", S_) and nf("L4", S_) == 4, "all-degree share: every check at L4")
    claim(all(lv("L4", B_).p_perm < 0.05 and spr("L4", B_).p_perm < 0.05 and d2("L4", B_).p_perm < 0.05
              and nf("L4", B_) < 4 for _ in [0]), "bachelor's-only share at L4: rank, Spearman, drop-2 yes; not every form")
    claim(all(lv("L4B", mk).p_perm > lv("L4", mk).p_perm for mk in STRICT2)
          and sum(nf("L4B", mk) for mk in STRICT2) < sum(nf("L4", mk) for mk in STRICT2), "L4B weaker than L4")
    claim(l3_none, "no form of either share significant at L3")
    claim(not (ok2(l4) and l4.p_wald < 0.05), "dummy not significant")
    claim(abs(s1.g - fl_wls.g) < 0.1 and abs(s1.g - fl_2s.g) < 0.1 and abs(s1.g + PUBLISHED_LIC["b_gap"]) < 0.1,
          "raw cell-level share interaction matches the field-level estimates")
    claim(ok2(s4) and fl_2s4.p_perm < 0.05 and not ok2(l1), "A4 share significant, field-level too; dummy not")
    rbk = {mk: {s_: bloc_ct(s_, mk) for s_ in FT_SPECS} for mk in LEV_MODS}
    claim(all(rbk[mk][s_].p_perm >= 0.05 for mk in STRICT2 for s_ in FTN), "strict shares not significant with the bloc")
    claim(all(rbk[mk][s_].cov_p_perm >= 0.05 for mk in STRICT2 for s_ in ("A4", "A4B")),
          "bloc not significant net of the rank share at A4/A4B")
    claim(all(ftalone(s_).p_perm < 0.05 for s_ in FTN), "bloc alone significant")
    claim(all(r.p_perm >= 0.05 for r in nb_all), "no strict-share moderation within the non-bloc fields")
    inci = lambda s_, mk: ftbt(NB, s_, mk).boot_lo < lv(s_, mk).g < ftbt(NB, s_, mk).boot_hi
    claim(all(inci(s_, S_) for s_ in ("A4", "A4B")), "non-bloc field-cluster CIs include the full-sample estimate (all-degree)")
    claim(inci("A4B", B_) and ft("A4", B_, "none", RNS, NB).lo < lv("A4", B_).g, "bachelor's-only: A4B bootstrap CI and A4 Wald CI include it")
    ba_edge = inci("A4", B_) or abs(ftbt(NB, "A4", B_).boot_lo - lv("A4", B_).g) < 0.005
    claim(ba_edge, "bachelor's-only A4 bootstrap CI includes the full-sample estimate or ends within 0.005 of it")
    claim(all(r.p_perm >= 0.05 for r in br_ctrl), "broad share not significant net of selectivity")
    claim(not cip_sig["acs_strict"], "all-degree share not significant within CIP-2")
    claim(all(ft("A4", mk, c).p_perm < 0.05 for mk in STRICT2 for c in ("log_earn", "log_n", "absorption",
                                                                        "prestige_range", "integrated")),
          "survives the continuous field-level controls and the integrated clusters")
    nbr = lambda s_, mk: ft(s_, mk, "none", RNS, NB)
    cipt = lambda mk: ", ".join(f"{s_} {pk(ft(s_, mk, 'cip2_fe'))}" for s_ in FTN)
    # broad share on the log scale (answer 2d)
    brL3, brL4, brL4B, brL5 = (lv(k_, BR) for k_ in ("L3", "L4", "L4B", "L5"))
    brL4_bloc, brL4_cip, brL4_inb = ft("L4", BR, "bloc"), ft("L4", BR, "cip2_fe"), ft("L4", BR, "none", RNS, BF)
    claim(brL3.p_perm < 0.05 and brL4.p_perm < 0.05 and brL5.p_perm < 0.05 and brL4_bloc.p_perm < 0.05,
          "broad share moderates the log slope raw, at L4, at L5 and at L4 with the bloc")
    claim(brL4B.p_perm >= 0.05 and brL4_cip.p_perm >= 0.05, "broad share log moderation gone with brand and within CIP-2")
    claim(all(rbk[mk]["L4"].p_perm >= 0.05 for mk in STRICT2), "strict shares' L4 moderation gone with the bloc")
    claim(ft("A4", S_, "integrated").p_perm < 0.05 and ft("A4B", S_, "integrated").p_perm < 0.05,
          "README integrated clusters leave the all-degree moderation in place")
    L.append(f"(2) Not identifiably as a licensing (pay-set-by-setting) effect. The strict-list share (health practitioners, "
             f"lawyers, K-12 teachers, architects, psychologists) does moderate the prestige slope net of selectivity, brand and "
             f"institution controls, but three checks cannot tell it apart from an engineering/CS/math/business-versus-other "
             f"contrast and find no dose-response in licensing, and the two licensure measures that also count licensed "
             f"engineers and accountants show no moderation of the status ordering net of selectivity (the broad share does "
             f"moderate the log premium net of selectivity, but not once brand is controlled; check d). What survives: raw, the cell-level interaction reproduces the field-level "
             f"estimate ({f3(s1.g, 2)} per unit all-degree share, SE {s1.se:.2f}, permutation p {fp_(s1.p_perm)}, on the "
             f"{int(s1.k)} fields with an ACS share, against {f3(fl_wls.g, 2)} from the scripts/20-style field-level WLS, "
             f"{f3(fl_2s.g, 2)} from a two-step meta-regression and {f3(-PUBLISHED_LIC['b_gap'], 2)} published); with "
             f"field-specific selectivity slopes (A4, {int(s4.k)} fields) it is {pm(s4)} (two-step field level "
             f"{f3(fl_2s4.g, 2)}, permutation p {fp_(fl_2s4.p_perm)}). Because nursing and special education hold most of the "
             f"share's range, the leverage-resistant rank form is the one to read: per SD of the share, with selectivity and "
             f"brand slopes (A4B), {prr(rs4b)} (all-degree) and {prr(rb4b)} (bachelor's-only), field-cluster 95% CIs "
             f"{ci(fr_s.boot_lo, fr_s.boot_hi, 3)} and {ci(fr_b.boot_lo, fr_b.boot_hi, 3)}, against an average-field "
             f"controlled slope of {f3(rs4b.b_main, 3)}; from raw (A3) to the status model (A7) it stays between "
             f"{f3(rns_ns.g.min(), 3)} and {f3(rns_ns.g.max(), 3)} (permutation p <= {fp_(rns_ns.p_perm.max())}). The checks:\n")
    L.append(f"   (a) **Field type.** The {lead_bloc['acs_strict']} lowest all-degree shares of the {int(FTC_.loc['acs_strict', 'k'])} "
             f"fields all belong to engineering, CS, statistics or business, and the share's Spearman with a dummy for CIP-2 "
             f"11/14/27/52 ({int(FTC_.loc['acs_strict', 'k_bloc'])} fields; the 'bloc') is "
             f"{f3(FTC_.loc['acs_strict', 'spearman_bloc'], 2)} (bachelor's-only {f3(FTC_.loc['acs_strict_ba', 'spearman_bloc'], 2)}). "
             f"With the bloc as a second cross-level moderator (Kennedy residual permutation, {N_PERM} draws) the rank-form "
             f"moderation is {pk(rbk[S_]['A3'])} raw, {pk(rbk[S_]['A4'])} with selectivity (A4), {pk(rbk[S_]['A4B'])} with "
             f"brand (A4B), {pk(rbk[S_]['A5'])} with institution RE (A5), {pk(rbk[S_]['A7'])} in the status model (A7); "
             f"bachelor's-only {pk(rbk[B_]['A4'])} (A4) and {pk(rbk[B_]['A4B'])} (A4B). In the same models the bloc is not "
             f"significant either ({pc(rbk[S_]['A4'])} at A4, {pc(rbk[S_]['A4B'])} at A4B), so the two cannot be separated; "
             f"alone the bloc steepens the controlled prestige slope by {pk(ftalone('A4'))} (A4). The bloc without math gives "
             f"the same ({pk(ft('A4', S_, 'bloc_nomath'))}); the README's 'integrated' clusters (CIP-2 11/14/27/45, social "
             f"sciences instead of business) leave the moderation in place ({pk(ft('A4', S_, 'integrated'))}; A4B "
             f"{pk(ft('A4B', S_, 'integrated'))}), and adding business to them weakens it ({pk(ft('A4', S_, 'integrated_bus'))}; "
             f"bachelor's-only {pk(ft('A4', B_, 'integrated_bus'))}). Controls for log field earnings "
             f"({pk(ft('A4', S_, 'log_earn'))}), field size ({pk(ft('A4', S_, 'log_n'))}), the academic absorption share "
             f"({pk(ft('A4', S_, 'absorption'))}) and the prestige range ({pk(ft('A4', S_, 'prestige_range'))}) leave it "
             f"intact (all A4, all-degree).\n")
    L.append(f"   (b) **Within the non-bloc fields** ({int(nbr('A4', S_).k)} fields; all-degree share "
             f"{nbr('A4', S_).share_lo:.2f}-{nbr('A4', S_).share_hi:.2f}, including nursing and special education) the "
             f"rank-form moderation is small and not significant: A4 {pk(nbr('A4', S_))}, A4B {pk(nbr('A4B', S_))}; "
             f"bachelor's-only {pk(nbr('A4', B_))} and {pk(nbr('A4B', B_))}; field-level Spearman of per-field A4B slopes with "
             f"the share {f3(fts(NB, 'A4B', S_).spearman, 2)} (p {fp_(fts(NB, 'A4B', S_).p_perm)}) and "
             f"{f3(fts(NB, 'A4B', B_).spearman, 2)} (p {fp_(fts(NB, 'A4B', B_).p_perm)}). These fits are imprecise: the "
             f"field-cluster 95% CIs ({ci(ftbt(NB, 'A4', S_).boot_lo, ftbt(NB, 'A4', S_).boot_hi, 3)} at A4 and "
             f"{ci(ftbt(NB, 'A4B', S_).boot_lo, ftbt(NB, 'A4B', S_).boot_hi, 3)} at A4B, all-degree) include the full-sample "
             f"estimates ({f3(lv('A4', S_).g, 3)} and {f3(lv('A4B', S_).g, 3)}); for the bachelor's-only share the A4B interval "
             f"{ci(ftbt(NB, 'A4B', B_).boot_lo, ftbt(NB, 'A4B', B_).boot_hi, 3)} includes its full-sample {f3(lv('A4B', B_).g, 3)}, "
             + (f"and so does the A4 interval {ci(ftbt(NB, 'A4', B_).boot_lo, ftbt(NB, 'A4', B_).boot_hi, 3)} "
                f"(full-sample {f3(lv('A4', B_).g, 3)}). " if inci("A4", B_) else
                f"while the A4 interval {ci(ftbt(NB, 'A4', B_).boot_lo, ftbt(NB, 'A4', B_).boot_hi, 4)} ends at its "
                f"full-sample {f3(lv('A4', B_).g, 4)} (the Wald 95% CI {ci(nbr('A4', B_).lo, nbr('A4', B_).hi, 3)} includes "
                f"it). ")
             + "So among these fields a moderation of the full-sample size is neither shown nor ruled out.\n")
    L.append(f"   (c) **Within CIP-2** (CIP-2-specific prestige slopes, share permuted within CIP-2; "
             f"{int(FTC_.loc['acs_strict', 'k_in_strata2'])} of the {int(FTC_.loc['acs_strict', 'k'])} fields sit in the "
             f"{int(FTC_.loc['acs_strict', 'n_strata2'])} CIP-2 groups with two or more fields and carry the test) the "
             f"all-degree share shows no moderation in any spec ({cipt(S_)}); the bachelor's-only share is mixed, "
             f"{len(cip_sig[B_])} of {len(FTN)} specs below p 0.05 ({cipt(B_)}).\n")
    br4, br4b = ft("A4", BR, "none"), ft("A4B", BR, "none")
    brl4, brl4b = ga("A4", BR), ga("A4B", BR)
    acc4, civ4 = fsi_sat.loc["accounting", "b_A4"], fsi_sat.loc["civil_engineering", "b_A4"]
    L.append(f"   (d) **Licensure measured broadly: no moderation of the status ordering net of selectivity; some of the log "
             f"premium, gone with brand.** The broad ACS share adds health "
             f"technicians, accountants, licensed engineers and counselors/social workers to the strict list; it is not skewed "
             f"({fsi_sat[BR].min():.2f}-{fsi_sat[BR].max():.2f}) and hardly related to the bloc (Spearman "
             f"{f3(FTC_.loc[BR, 'spearman_bloc'], 2)}). Raw it moderates the prestige slope ({pm(ga('A1', BR))} on all fields; "
             f"rank form on the selectivity sample {pk(ft('A3', BR, 'none'))}); with field-specific selectivity slopes it "
             f"does not, in any form: rank A4 {pk(br4)}, A4B {pk(br4b)}, A7 {pk(ft('A7', BR, 'none'))}; square root A4 "
             f"{pk(ft('A4', BR, 'none', 'square root'))}; linear per unit share A4 {f3(brl4.g, 2)} (p {fp_(brl4.p_perm)}), "
             f"A4B {f3(brl4b.g, 2)} (p {fp_(brl4b.p_perm)}) ({len(br_ctrl)} of {len(br_ctrl)} fits at A4, A4B and A7 with "
             f"p >= 0.05); with institution random intercepts instead (A5) it is borderline (rank {pk(br_a5[RNS])}, square "
             f"root {pk(br_a5['square root'])}, linear per SD {pk(br_a5['linear'])}). On log earnings it behaves differently: "
             f"raw (L3, rank form) {pk(brL3)}; net of selectivity (L4) {pk(brL4)} (the four leverage-resistant forms "
             f"{prange(rob('L4', BR))}; linear per unit share {f3(ga('L4', BR).g, 3)}, p {fp_(ga('L4', BR).p_perm)}; field-level "
             f"Spearman {psp(spr('L4', BR))}; without nursing and special education {f3(d2('L4', BR).g, 3)} per unit share, "
             f"p {fp_(d2('L4', BR).p_perm)}); with institution RE (L5) {pk(brL5)}; with the bloc added still "
             f"{pk(brL4_bloc)}; but not with the brand slope (L4B {pk(brL4B)}) nor within CIP-2 (L4 {pk(brL4_cip)}). "
             f"The ex-ante licensed dummy fails as well ({f3(l4.g, 2)}, permutation p {fp_(l4.p_perm)}, at A4; on log "
             f"earnings at L4 {f3(ga('L4', 'lic').g, 4)}, p {fp_(ga('L4', 'lic').p_perm)}); "
             f"two of its three licensed fields in the selectivity sample, accounting and civil engineering, have steep "
             f"controlled slopes (per-field OLS at A4 {f3(acc4, 2)} and {f3(civ4, 2)} against {f3(unl_A4, 2)} for the "
             f"average unlicensed field with a share).\n")
    L.append(f"   So these data give no licensing dose-response behind the 'pay set by setting' reading: the moderation of the "
             f"status ordering is specific to the strict list, cannot be separated from field type, is not detectable within the "
             f"non-bloc fields or (all-degree share) within CIP-2, and does not hold for licensure measured broadly or ex ante. "
             f"One qualification: the README's own 'integrated' grouping (CIP-2 11/14/27/45, fixed before this check) does "
             f"not remove it (A4 {pk(ft('A4', S_, 'integrated'))}); groupings that put business with engineering and CS do. "
             f"What survives at the cell level with selectivity controlled is a "
             f"contrast: fields whose graduates work in strict-list licensed occupations have a looser status-earnings "
             f"ordering than engineering, CS, math and business fields; these data do not say whether that is pay set by "
             f"setting or anything else that separates the two groups of fields. On log earnings (the premium) the strict "
             f"shares in rank form give L4 (selectivity) {prr(rl4[S_])} and {prr(rl4[B_])}, weaker with brand (L4B "
             f"{prr(rl4b[S_])} and {prr(rl4b[B_])}), nothing before controls, and nothing once the bloc is added (L4 "
             f"{pk(rbk[S_]['L4'])} and {pk(rbk[B_]['L4'])}); the broad share's log-scale moderation (L4 {pk(brL4)}) survives "
             f"the bloc ({pk(brL4_bloc)}) but not the brand slope (L4B {pk(brL4B)}) or the within-CIP-2 test "
             f"({pk(brL4_cip)}). So no licensure measure has a log-scale moderation that passes both the field-type and the "
             f"brand checks run here (L4B was not fitted with the bloc). Selectivity and brand pricing also fall with the "
             f"strict share (table b).")
    rt4 = {mk: lv("A4", mk) for mk in LEV_MODS}
    sdf = lambda v: "0 (at the boundary)" if at0(v) else f"{v:.3f}"
    px4, px5 = ga("A4", "PxSAT"), ga("A5", "PxSAT")
    claim(px5.p_wald >= 0.05, "P x SAT not significant with institution RE")
    L.append(f"(3) The between-field variance of the prestige slope (normal-score units, about a Spearman) is "
             f"{Hs.loc['A1', 'sd_P'] ** 2:.4f} raw on all {Hs.loc['A1', 'k']} fields (SD {Hs.loc['A1', 'sd_P']:.3f} "
             f"[{Hs.loc['A1', 'sd_P_lo']:.3f}, {Hs.loc['A1', 'sd_P_hi']:.3f}]) and {Hs.loc['A3', 'sd_P'] ** 2:.4f} raw on the "
             f"selectivity sample ({Hs.loc['A3', 'k']} fields). Field prestige slopes "
             f"{'stay heterogeneous' if het_A4 else 'are no longer distinguishable from equal'} "
             f"after selectivity controls (SD {Hs.loc['A4', 'sd_P']:.3f}, variance {Hs.loc['A4', 'sd_P'] ** 2:.4f}, LRT p "
             f"{fp_(Hs.loc['A4', 'lrt_p_P'])}; raw SD {Hs.loc['A3', 'sd_P']:.3f} on the same sample); adding the field-specific "
             f"brand slope takes the SD to {Hs.loc['A4B', 'sd_P']:.3f}, "
             f"{'still distinguishable from zero' if Hs.loc['A4B', 'lrt_p_P'] < 0.05 else 'no longer distinguishable from zero'} "
             f"(LRT p {fp_(Hs.loc['A4B', 'lrt_p_P'])}). In-sample, adding the "
             f"all-degree licensed share as a linear moderator takes the controlled slope SD from {s4.tau_null:.3f} to {s4.tau:.3f} on "
             f"its {int(s4.k)} fields, i.e. it accounts for about {share_expl:.0%} of the remaining between-field slope variance"
             f"{' (the estimate is at the boundary)' if at0(s4.tau) else ', not all of it'}; the bachelor's-only share takes it to "
             f"{ba4.tau:.3f}. In the rank form the two shares take it to {rt4[S_].tau:.3f} and {rt4[B_].tau:.3f} (about "
             f"{1 - (rt4[S_].tau / rt4[S_].tau_null) ** 2:.0%} and {1 - (rt4[B_].tau / rt4[B_].tau_null) ** 2:.0%}); with the "
             f"brand slope (A4B) to {sdf(rs4b.tau)} and {sdf(rb4b.tau)}. The engineering/CS/math/business dummy alone "
             f"takes it to {ftalone('A4').tau:.3f} (A4; from {ftalone('A4').tau_null:.3f}), about "
             f"{1 - (ftalone('A4').tau / ftalone('A4').tau_null) ** 2:.0%} of that variance (rank-form strict shares: "
             f"{1 - (rt4[S_].tau / rt4[S_].tau_null) ** 2:.0%} and {1 - (rt4[B_].tau / rt4[B_].tau_null) ** 2:.0%}). Institution selectivity as a cell-level moderator "
             f"(prestige x within-field SAT normal score) is {f3(px4.g, 3)} per SD of SAT in A4 (Wald p {fp_(px4.p_wald)}; "
             f"no institution effects, so this SE ignores clustering by institution) and {f3(px5.g, 3)} (Wald p "
             f"{fp_(px5.p_wald)}) with institution random intercepts (A5): no robust sign that field prestige pays more at "
             f"more selective institutions.\n")
    # ------------------------------------------------------------------ revision notes (printed at the end)
    R = []
    R.append("### Latest revision: field type and the broad share\n")
    R.append("Independent verification found (1) that the rank-form licensing moderation, described in the previous version "
             "as surviving 'clearly', cannot be told apart from an engineering/CS/business-versus-other contrast, and (2) that "
             "the broad ACS licensed share, fitted in tables (b) and (c), failed net of selectivity and was left out of the "
             "answer, the caveats and table (k0). Fixed here; every number of the previous version is reproduced unchanged "
             "(the field-type rows without a control re-run table b0 on the same random streams, asserted).\n")
    R.append(f"- **Withdrawn: 'Yes for both ACS licensed-occupation shares' and 'the controls hardly shrink it' as evidence "
             f"for pay set by setting.** The strict-share moderation survives status controls but not a field-type control: "
             f"with the engineering/CS/math/business dummy as a second cross-level moderator (Kennedy residual permutation) "
             f"the rank form gives {pk(rbk[S_]['A4'])} at A4 and {pk(rbk[S_]['A4B'])} at A4B (previously {prr(lv('A4', S_))} "
             f"and {prr(rs4b)}); within the non-bloc fields {pk(nbr('A4', S_))} and {pk(nbr('A4B', S_))}; within CIP-2 "
             f"{pk(ft('A4', S_, 'cip2_fe'))} and {pk(ft('A4B', S_, 'cip2_fe'))}. The verification's figures (A4 -0.053, "
             f"p 0.107; A4B -0.065, p 0.104; non-bloc A4 -0.030, p 0.42) are reproduced up to permutation noise. Answer (2) "
             f"now reads 'not identifiably as a licensing effect'. Added tables (k1), (k1-ii), (k2), figure panels (g) and (h), the "
             f"within-CIP-2 stratified test the verification suggested, the bloc's own Kennedy p, the within-bloc fits and "
             f"field-cluster intervals for the subsets (which show the non-bloc null is imprecise, not a demonstrated zero).")
    R.append(f"- **Added: the broad share next to the dummy.** The broad share now runs through every leverage check of table "
             f"(b0) (winsorized, square root, log, rank; field-level Spearman; nursing and special education dropped), the "
             f"field-cluster bootstrap of its rank form (table d), table (k0) and answer (2d). Net of selectivity it shows no "
             f"moderation at A4, A4B or A7 in any form (rank A4 {pk(br4)}, A4B {pk(br4b)}) and is borderline with "
             f"institution RE (A5 rank {pk(br_a5[RNS])}); the previous answer reported 'both ACS shares' while three were "
             f"fitted. The answer now says the moderation of the status ordering is specific to the strict list. On log "
             f"earnings the broad share does moderate the prestige slope net of selectivity (L4 rank {pk(brL4)}, with the bloc "
             f"{pk(brL4_bloc)}), not with brand (L4B {pk(brL4B)}); that is reported in answer (2d) and the caveats (an earlier "
             f"draft of this revision said the broad share 'shows nothing net of selectivity', which was true only of the "
             f"normal-score specs).\n")
    R.append("### Second revision: leverage of the two high-share fields\n")
    R.append("Independent verification found that the two narrowings of answer (2) in the previous version came from "
             "entering the licensed share linearly while two fields hold most of its range. Fixed here; every number is from "
             "the re-run, and the numbers of the previous version that are not discussed below are reproduced unchanged.\n")
    lb_n, lb_s = lofo1("A4B", B_, "nursing"), lofo1("A4B", B_, "special_education")
    lo_bb = lofo("A4B", B_)
    R.append(f"- **Added leverage checks.** For both ACS shares and specs {', '.join(LEV_SPECS)}: the share entered "
             f"winsorized at {WINS:.2f}, as a square root, as log(share + 0.01) and as rank normal scores (each per SD, with the "
             f"same field-label permutation test), a field-level Spearman of per-field slopes with the share ({N_PERM} label "
             f"permutations), the linear fit with nursing and special education dropped together, leave-one-field-out for A4B "
             f"and L4B, a field-cluster bootstrap of the rank form, and a log-scale spec with the brand slope (L4B). Table (b0) "
             f"now shows linear and leverage-resistant results side by side; the old per-unit table is (b0-ii); panels (e) and "
             f"(f) of the figure show the leverage.")
    R.append(f"- **Withdrawn: \"only the all-degree share survives the brand control\".** The previous version reported the "
             f"bachelor's-only share at A4B as {pp(ba4b)} and in A7 as {pp(a7bp)}, and read that as not surviving. The linear "
             f"coefficient is set mainly by nursing and special education: without nursing it is {f3(lb_n.g, 2)} (Wald p "
             f"{fp_(lb_n.p_wald)}), without special education {f3(lb_s.g, 2)} (Wald p {fp_(lb_s.p_wald)}), without both "
             f"{f3(d2('A4B', B_).g, 2)} (permutation p {fp_(d2('A4B', B_).p_perm)}); {lo_bb['nsig']} of the {lo_bb['k']} "
             f"single-field drops reach Wald p < 0.05, so the leave-one-out check of the previous version, run only for A4, "
             f"could not show this. With leverage-resistant forms the bachelor's-only moderation holds at A4B ({prange(rob('A4B', B_))}; "
             f"rank form {prr(rb4b)}), A4E ({prange(rob('A4E', B_))}) and A7 ({prange(rob('A7', B_))}), as does the "
             f"field-level Spearman at A4B ({psp(spr('A4B', B_))}).")
    R.append(f"- **Withdrawn: \"the moderation is of fit, not of premium\".** It rested on linear log-scale coefficients (L4: "
             f"{pp3(gl4s)} all-degree, {pp3(gl4b)} bachelor's-only). With the leverage handled the log-scale moderation is there "
             f"net of selectivity (L4 rank form {prr(rl4[S_])} and {prr(rl4[B_])}; field-level Spearman {psp(spr('L4', S_))} "
             f"and {psp(spr('L4', B_))}), weaker once the brand slope is added (L4B, below) and absent before controls (L3).")
    R.append("- Unchanged: answer (1) on the nulls, the dummy result, the heterogeneity estimates and the previously reported "
             "linear coefficients.")
    R.append("- Presentation only (no number changed): the answer now leads with the power of each null and the cell- vs "
             "field-level licensing comparison, states the institution-selectivity moderator, table (f) gives the smallest "
             "TOST margin, a compact table (k0) was added, and these notes moved to the end.\n")
    R.append("### First revision\n")
    R.append("The first round fixed four problems; its numbers are all reproduced above and below. Its second item led to "
             "the narrowing that the second revision withdraws.\n")
    R.append(f"- **REML optimizer stopped at spurious boundary points.** The deviance depends on the SD ratio theta only "
             f"through theta^2, so its derivative in theta is exactly 0 at theta = 0, and the previous two-start L-BFGS-B "
             f"(starts 0.2 and 1.0, in theta) could stop at the bound when the optimum is interior; the outcome also depended "
             f"on the BLAS thread count. The optimizer now works on theta^2 (where the deviance has a non-zero slope at 0), "
             f"uses five starts, and scans any component left at 0 (table g). For A4 x all-degree share the earlier "
             f"optimizer, re-run in this script, gives slope SD {rc_s4.old_sd:.3f} (deviance {rc_s4.old_dev:.3f}, g "
             f"{f3(rc_s4.old_g, 3)}, SE {rc_s4.old_se:.3f}); the corrected fit gives {rc_s4.sd:.3f} (deviance {rc_s4.dev:.3f}, "
             f"g {f3(rc_s4.g, 3)}, SE {rc_s4.se:.3f}), confirmed by Nelder-Mead ({rc_s4.nm_dev:.3f}) and a theta grid "
             f"(minimum at theta {rc_s4.grid_theta:.4f}). So the previous \"SD 0.099 to 0.000\" is {s4.tau_null:.3f} to "
             f"{s4.tau:.3f} (the share accounts for about {share_expl:.0%} of the remaining slope variance, not all of it), and "
             f"\"the REML estimate is at the boundary\" is withdrawn. Other numbers that moved: A4 bachelor's-only share without "
             f"nursing {f3(rc_b4n.old_g, 2)} (earlier optimizer) to {f3(rc_b4n.g, 2)}, so the leave-one-out range is "
             f"{f3(lo_b4['lo'], 2)} to {f3(lo_b4['hi'], 2)}; A7 prestige x share {f3(rc_a7.old_g, 3)} (SE {rc_a7.old_se:.3f}, "
             f"Wald p {fp_(wald_p(rc_a7.old_g, rc_a7.old_se))}) to {f3(rc_a7.g, 3)} (SE {rc_a7.se:.3f}, Wald p "
             f"{fp_(wald_p(rc_a7.g, rc_a7.se))}) and SAT x share now {f3(a7s.g, 2)} (Wald p {fp_(a7s.p_wald)}); the slope "
             f"SDs of the A4 leave-one-out fits are {lo_s4['tau_lo']:.3f}-{lo_s4['tau_hi']:.3f} (all-degree share) and "
             f"{lo_b4['tau_lo']:.3f}-{lo_b4['tau_hi']:.3f} (bachelor's-only), no longer 0. "
             + (f"The slope SDs still at 0 are genuine boundaries: the brand slope in A7 (0 with and without moderators; "
                f"Nelder-Mead in table g also puts it at {float(rc_a7.nm_theta_all.split()[-1]):.4f}) and A4B x all-degree share "
                f"in the drop-set row of table (e) (Nelder-Mead and grid deviances {rc_b4bn.nm_dev:.3f} and "
                f"{rc_b4bn.grid_dev:.3f} vs {rc_b4bn.dev:.3f}). "
                if (at0(rc_b4bn.sd) and min(rc_b4bn.nm_dev, rc_b4bn.grid_dev) >= rc_b4bn.dev - 1e-6
                    and float(rc_a7.nm_theta_all.split()[-1]) < 1e-3 and rc_a7.nm_dev >= rc_a7.dev - 1e-6) else "")
             + f"Permutation p-values were not "
             f"affected (they use the moderator-free models, whose fits are unchanged), nor was the direction of any conclusion.")
    R.append(f"- **The bachelor's-only share had no status controls.** Added: the status model A7 with the bachelor's-only share "
             f"and its two components, and two new specs, A4B (A4 + a field-specific slope on academia-wide brand) and A4E "
             f"(A4B + a field-specific slope on institution-wide median earnings, MD_EARN_WNE_P10). The bachelor's-only "
             f"moderation does not survive them ({f3(ba4b.g, 2)}, permutation p {fp_(ba4b.p_perm)}, with brand; A7 "
             f"{f3(a7bp.g, 2)}, p {fp_(a7bp.p_perm)}), while the all-degree share does ({f3(s4b.g, 2)}, p {fp_(s4b.p_perm)}; A7 "
             f"{f3(a7p.g, 2)}, p {fp_(a7p.p_perm)}). Answer (2) was narrowed accordingly; the leverage checks of this "
             f"revision withdraw that narrowing.")
    R.append(f"- **The institution bootstrap was conditional on the realized fields.** Re-sampling institutions inside fixed "
             f"fields keeps each field's realized slope deviation, so it measures within-field sampling error only and "
             f"misses the between-field sampling variance that dominates for a moderator measured once per field. The "
             f"previous table (d) therefore gave the dummy's raw interaction a CI excluding 0 while its permutation p was "
             f"{fp_(l3.p_perm)}. Table (d) now leads with a field-cluster bootstrap; the institution bootstrap is kept, labelled "
             f"as conditional, for the raw-vs-controlled differences only. For the dummy the field-cluster CIs are "
             f"{ci(fb('lic', 'A3').boot_lo, fb('lic', 'A3').boot_hi)} (A3), {ci(fb('lic', 'A4').boot_lo, fb('lic', 'A4').boot_hi)} "
             f"(A4) and {ci(fb('lic', 'A4-A3').boot_lo, fb('lic', 'A4-A3').boot_hi)} for the A4 - A3 difference: controls "
             f"move the dummy's interaction toward 0 by more than resampling noise (the difference is paired on the same "
             f"fields, so it is estimated more tightly than either level), but neither level differs from 0. For A4 x all-degree share the SEs are now Wald "
             f"{s4.se:.3f} (slope SD {s4.tau:.3f}), two-step Knapp-Hartung {fl_2s4.se:.3f}, field-cluster bootstrap "
             f"{fb_s4.boot_se:.3f}; inference rests on the field-label permutation test.")
    R.append(f"- **Scale dependence was not reported.** Added L3 (raw log earnings on the selectivity sample), the "
             f"bachelor's-only share on L3-L5, and log-scale selectivity and brand pricing (C1L, C2L). On log points the "
             f"licensing moderation of the prestige slope is weak before any control (see answer 2).\n")
    # ------------------------------------------------------------------ details
    L.append("**1. Which field-level nulls are informative?** Moderate effect = true-score correlation 0.30 in the direction the "
             "theory predicts (large = 0.50); for the career-time contrast, moderate = half the all-field slope, large = all of it. "
             "\"Informative\" means the one-sided 95% bound in the theory's direction excludes the moderate effect; \"rules out "
             "large only\" means it excludes the large but not the moderate effect; otherwise \"underpowered\". This is a "
             "one-sided criterion; two-sided equivalence (TOST) is stricter, and table (f) gives the smallest margin it "
             "accepts for each version.\n")
    n_b1_below = int((b1_all6.bound < MODERATE).sum())
    L.append(f"- **Within-occupation dispersion law (scripts/53).** Reliable {b1r.n} fields: Spearman {f3(b1r.estimate, 2)}, "
             f"true-score upper bound {f3(b1r.bound, 2)}, MDE {b1r.mde:.2f}: {sv(b1r)}. All {b1a.n} fields: Spearman "
             f"{f3(b1a.estimate, 2)}, true-score upper bound {f3(b1a.bound, 2)}: {sv(b1a)}; the six dispersion measures give "
             f"bounds {f3(b1_all6.bound.min(), 2)} to {f3(b1_all6.bound.max(), 2)} ({n_b1_below} of 6 below +0.30). The "
             f"precision-weighted cell-level version (tail_hi x prestige, {c1a.n} fields) gives r = {f3(c1a.estimate, 2)} with "
             f"upper bound {f3(c1a.bound, 2)}: {sv(c1a)}; after selectivity controls ({c1b.n} fields) the bound is "
             f"{f3(c1b.bound, 2)}: {sv(c1b)}. So the law is ruled out at moderate size only by the unweighted all-field "
             f"Spearman, whose point estimate has the opposite sign to the cell-level one." if (b1a.verdict == INF and c1a.verdict != INF
             and np.sign(b1a.estimate) != np.sign(c1a.estimate)) else
             f"- **Within-occupation dispersion law (scripts/53).** Reliable {b1r.n} fields: Spearman {f3(b1r.estimate, 2)}, "
             f"true-score upper bound {f3(b1r.bound, 2)}, MDE {b1r.mde:.2f}: {sv(b1r)}. All {b1a.n} fields: Spearman "
             f"{f3(b1a.estimate, 2)}, upper bound {f3(b1a.bound, 2)}: {sv(b1a)} ({n_b1_below} of 6 measures below +0.30). Cell "
             f"level ({c1a.n} fields): r = {f3(c1a.estimate, 2)}, bound {f3(c1a.bound, 2)}: {sv(c1a)}; after selectivity "
             f"controls {f3(c1b.bound, 2)}: {sv(c1b)}.")
    L.append(f"- **Occupation concentration (scripts/50; theory predicts a negative correlation).** Reliable {b2r.n}: Spearman "
             f"{f3(b2r.estimate, 2)}, true-score lower bound {f3(b2r.bound, 2)}, MDE {f3(b2r.mde, 2)}: {sv(b2r)}. All {b2a.n}: "
             f"{f3(b2a.estimate, 2)}, lower bound {f3(b2a.bound, 2)}: {sv(b2a)}. Cell level ({c2a.n} fields): r = "
             f"{f3(c2a.estimate, 2)}, lower bound {f3(c2a.bound, 2)}: {sv(c2a)}; after selectivity controls {f3(c2b.bound, 2)}: "
             f"{sv(c2b)}.")
    moran_bound = (f"No upper bound can be set: P(I <= observed | lambda) never falls below 0.05 for lambda <= 0.95 "
                   f"(minimum {b3['min_cdf']:.3f})." if b3["censored"] else
                   f"One-sided 95% upper bound on the kin correlation: {f3(b3['kin_up'], 2)}.")
    L.append(f"- **Kinship Moran's I (scripts/51): {sv(b3r)}.** With {b3['n']} fields and the faculty-flow weights, the "
             f"permutation test has {b3['power_moderate']:.0%} power at a kin correlation of 0.30 and at most "
             f"{b3['max_power']:.0%} anywhere on the grid (at kin correlation {b3['kin_max_power']:+.2f}). Because the kin "
             f"correlation is {b3['kin0']:+.2f} under no autocorrelation, a rise of 0.30 above that baseline is a smaller "
             f"alternative, with power {b3['power_moderate_x']:.0%}. {moran_bound} The pilot called this \"a genuine null, "
             "not a power artifact of a weak test\" because its bootstrap interval for I, [-0.176, +0.187], straddles zero "
             "(outputs/MORAN_KINSHIP_COUPLING.md). That reading does not hold: the interval re-sampled only the measurement "
             "error of rho_f at fixed values; it is not a sampling distribution under an alternative, so it says nothing "
             "about power.")
    L.append(f"- **Career-time slope, integrated minus others (scripts/52): {sv(b4r)}.** Estimate "
             f"{f3(b4['estimate'], 4)}/yr, one-sided 95% upper bound {f3(b4['bound'], 4)}/yr; MDE (80%) {b4['mde_analytic']:.4f}/yr "
             f"(simulated permutation test: {b4['mde_sim']:.4f}). The bound "
             f"{'excludes' if b4['bound'] < b4['thr_large'] else 'does not exclude'} integrated fields rising by the whole "
             f"all-field slope ({b4['thr_large']:.4f}/yr) more than the others, and "
             f"{'excludes' if b4['bound'] < b4['thr_mod'] else 'does not exclude'} half of it ({b4['thr_mod']:.4f}/yr; power "
             f"{b4['power_moderate_analytic']:.0%}).")
    L.append("- I could not find the \"0 of 240 specifications\" count of README/ROADMAP in any script or result file (it "
             "appears only in README.md and ROADMAP.md, added in commit 0d98605); scripts/53 runs 6 dispersion measures x 2 "
             "field sets, and those 12 tests are in table (f).\n")
    L.append("**2. Does the licensing / pay-set-by-setting moderation survive at the cell level with selectivity controlled?** "
             f"Not identifiably as licensing. The two strict-list shares survive every status control (selectivity, brand, "
             f"institution-wide earnings, institution effects, the status model) once the share does not enter linearly, but "
             f"they cannot be told apart from a control for engineering/CS/math/business fields; they are not detectable "
             f"within the other fields and, for the all-degree share, not within CIP-2; the broad share and the ex-ante dummy "
             f"show no moderation of the normal-score slope net of selectivity. On log earnings the strict shares hold net of "
             f"selectivity (fully for the all-degree share, not in every form for the bachelor's-only share), weaker with "
             f"brand added, not before controls and not with the bloc; the broad share moderates the log slope raw and net "
             f"of selectivity (also with the bloc), not with brand or within CIP-2. Tables (k1), (k1-ii) and (k2) have the field-type checks; (b0) the "
             f"status-control ladder with every form of the three shares; (b0-ii) the linear details.\n")
    bl_lin, bl_sq = bloc_ct("A4", S_, "linear"), bloc_ct("A4", S_, "square root")
    inb = lambda s_, mk: ft(s_, mk, "none", RNS, BF)
    k12_math = fsi.loc["mathematics", "acs_k12_ba"] if "mathematics" in fsi.index else np.nan
    L.append(f"- **Field type, details (tables k1, k1-ii, k2).** In the linear and square-root forms, which give nursing "
             f"and special education more weight, the bloc keeps an effect net of the share while the share loses its own: "
             f"A4, all-degree, linear share {pk(bl_lin)} with bloc {pc(bl_lin)}; square root {pk(bl_sq)} with bloc "
             f"{pc(bl_sq)}; in the rank form neither is significant net of the other (answer 2a). Collinearity with the "
             f"bloc: Pearson of the rank-form share with the dummy {f3(FTC_.loc[S_, 'pearson_rank_bloc'], 2)} (all-degree) and "
             f"{f3(FTC_.loc[B_, 'pearson_rank_bloc'], 2)} (bachelor's-only). Inside the bloc ({int(inb('A4', S_).k)} fields) "
             f"the all-degree share only runs from {inb('A4', S_).share_lo:.2f} to {inb('A4', S_).share_hi:.2f}, the top "
             f"being mathematics (bachelor's-only K-12 teacher share {k12_math:.2f}); there the rank-form moderation is "
             f"{pk(inb('A4', S_))} at A4 and {pk(inb('A4B', S_))} at A4B (field-cluster 95% CI "
             f"{ci(ftbt(BF, 'A4B', S_).boot_lo, ftbt(BF, 'A4B', S_).boot_hi, 3)}), bachelor's-only {pk(inb('A4B', B_))} at "
             f"A4B, and the field-level Spearman at A4B {f3(fts(BF, 'A4B', S_).spearman, 2)} (p {fp_(fts(BF, 'A4B', S_).p_perm)}) "
             f"and {f3(fts(BF, 'A4B', B_).spearman, 2)} (p {fp_(fts(BF, 'A4B', B_).p_perm)}): at most a borderline negative "
             f"association inside the bloc, over a share range of a few percent. Within the non-bloc fields no spec or form "
             f"of either strict share reaches p < 0.05 ({len(nb_all)} fits, A3-A7, linear and rank; permutation p "
             f"{fp_(min(r.p_perm for r in nb_all))}-{fp_(max(r.p_perm for r in nb_all))}).")
    brb = fbr(BR, "A4B")
    L.append(f"- **Broad share (added to the leverage table b0 in this revision).** Linear per unit share (table b): raw "
             f"{pm(ga('A3', BR))} on the selectivity sample, {pm(ga('A4', BR))} with selectivity controls, {pm(ga('A4B', BR))} "
             f"with brand, {pm(ga('A7', BR, 'P'))} in the status model. Rank form (table b0): A3 {prr(lv('A3', BR))}, A4 "
             f"{prr(lv('A4', BR))}, A4B {prr(lv('A4B', BR))} (field-cluster 95% CI {ci(brb.boot_lo, brb.boot_hi, 3)}), A7 "
             f"{prr(lv('A7', BR))}; field-level Spearman of per-field A4B slopes with the broad share "
             f"{psp(spr('A4B', BR))}. For comparison, rank form: selectivity pricing C1 {prr(lv('C1', BR))}, brand pricing "
             f"C2 {prr(lv('C2', BR))}, raw log slope L3 {prr(lv('L3', BR))}"
             + ("; so the broad share goes with less status pricing in general, but not with a looser status ordering once "
                "selectivity is held fixed" if all(lv(k_, BR).p_perm < 0.05 for k_ in ("C1", "C2")) else "")
             + f". On log earnings it does go with a smaller premium net of selectivity (rank form L4 {prr(brL4)}, L5 "
             f"{prr(brL5)}), also within the 17 bloc fields alone, where the broad list adds licensed engineers and "
             f"accountants (L4 {pk(brL4_inb)}; normal-score A4 there {pk(ft('A4', BR, 'none', RNS, BF))}), but not once "
             f"brand is controlled (L4B {prr(brL4B)}; linear L4B {pm3(ga('L4B', BR))}).")
    L.append(f"- **Ex-ante licensed dummy: {'yes' if ok2(l4) and l4.p_wald < 0.05 else 'no'}.** Raw on the selectivity sample "
             f"{pm(l3)}; with field-specific selectivity slopes {pm(l4)}; + brand {f3(l4b.g, 2)}; + institution RE "
             f"{f3(l5.g, 2)}, + institution FE {f3(l6.g, 2)}. Only {len(sat_lic)} licensed fields reach the selectivity sample; "
             f"their raw (shrunken) prestige slopes are {lic_slopes}, against a mean of {f3(unl_mean, 2)} for the unlicensed "
             f"fields there. On all fields the raw dummy effect {f3(l1.g, 2)} (Wald p {fp_(l1.p_wald)}, permutation p "
             f"{fp_(l1.p_perm)}) becomes {f3(ds_l1.g, 2)} when nursing and communication disorders are dropped; at the field "
             f"level on the same fields the gap-map WLS gives {f3(fl_wls_l.g, 2)} (p {fp_(fl_wls_l.p)}).")
    cell_vs_2s = s1.se / fl_2s.se
    L.append(f"- **ACS licensed-occupation share, all degree holders (the scripts/20 anchor): "
             f"{'survives every status control used here, not the field-type control (answer 2a)' if strict_ladder_ok else 'not by every control'}.** Per unit share: raw "
             f"{pm(s3)}; selectivity-controlled {pm(s4)}; + brand {pm(s4b)}; + institution-wide earnings {pm(s4e)}; + "
             f"institution RE {f3(s5.g, 2)}; + institution FE {f3(s6.g, 2)}; status model A7 {pm(a7p)}. For A4 the "
             f"slope SD with the moderator is {s4.tau:.3f}{' (at the boundary)' if at0(s4.tau) else ''}; the two-step "
             f"Knapp-Hartung SE on the same fields is {fl_2s4.se:.2f} and the field-cluster bootstrap SE {fb_s4.boot_se:.2f} "
             f"(95% CI {ci(fb_s4.boot_lo, fb_s4.boot_hi)}; skewed because resamples without nursing give steeper "
             f"estimates), so the Wald p understates the uncertainty and the permutation p is the one to use. The "
             f"controlled estimate keeps {fb('acs_strict', 'A4-A3').ratio:.0%} of the raw one (difference "
             f"{f3(fb('acs_strict', 'A4-A3').g, 2)}, field-cluster 95% CI {ci(fb('acs_strict', 'A4-A3').boot_lo, fb('acs_strict', 'A4-A3').boot_hi)}). "
             f"Leaving out one field at a time gives {f3(lo_s4['lo'], 2)} (without {lo_s4['lo_f']}) to "
             f"{f3(lo_s4['hi'], 2)} (without {lo_s4['hi_f']}), largest Wald p {fp_(lo_s4['pmax'])}; for A4B "
             f"{f3(lofo('A4B', S_)['lo'], 2)} to {f3(lofo('A4B', S_)['hi'], 2)}, {lofo('A4B', S_)['nsig']} of "
             f"{lofo('A4B', S_)['k']} with Wald p < 0.05. In the rank form (per SD of the share): A3 {prr(lv('A3', S_))}, A4 "
             f"{prr(lv('A4', S_))}, A4B {prr(lv('A4B', S_))}, A4E {prr(lv('A4E', S_))}, A7 {prr(lv('A7', S_))}; the other "
             f"leverage-resistant forms at A4B give {prange(rob('A4B', S_))}; field-level Spearman at A4B "
             f"{psp(spr('A4B', S_))}; without nursing and special education {f3(d2('A4B', S_).g, 2)} per unit share "
             f"(permutation p {fp_(d2('A4B', S_).p_perm)}). At the raw level the "
             f"cell-level estimate on all fields ({f3(s1.g, 2)}, SE {s1.se:.2f}) is close to the field-level ones on the same "
             f"{int(s1.k)} fields (gap-map WLS {f3(fl_wls.g, 2)}, SE {fl_wls.se:.2f}; two-step meta-regression {f3(fl_2s.g, 2)}, "
             f"SE {fl_2s.se:.2f}) and to the published scripts/20 value (-0.66 in coupling units). The cell-level SE is "
             f"{cell_vs_2s:.2f} times the two-step SE: a moderator measured once per field is still identified from the fields; "
             f"what the cell model adds is control of selectivity inside each field.")
    fsl = lambda f, c: fsi.loc[f, c] if f in fsi.index else np.nan
    pr_line = lambda sk, f: d2(sk, B_).b_main + d2(sk, B_).g * fsl(f, "acs_strict_ba")
    L.append(f"- **Two fields carry the linear fit.** In the selectivity sample the bachelor's-only share is {nurse_ba:.2f} in "
             f"nursing and {se_ba:.2f} in special education and {oth_rng[B_][0]:.2f}-{oth_rng[B_][1]:.2f} in the other "
             f"{len(oth)} fields (all-degree share: {nurse_s:.2f}, {fsl('special_education', S_):.2f} and "
             f"{oth_rng[S_][0]:.2f}-{oth_rng[S_][1]:.2f}). A share entered linearly is fitted mostly to those two fields, and they "
             f"do not sit on the line through the others: special education ({n_se} cells) has a steep positive prestige slope "
             f"({f3(fsl('special_education', 'b_A3'), 2)} raw, {f3(fsl('special_education', 'b_A4'), 2)} selectivity-controlled, "
             f"{f3(fsl('special_education', 'b_A4B'), 2)} with brand; per-field OLS), nursing a flat one "
             f"({f3(fsl('nursing', 'b_A3'), 2)}, {f3(fsl('nursing', 'b_A4'), 2)}, {f3(fsl('nursing', 'b_A4B'), 2)}), whereas "
             f"the linear A4B fit to the bachelor's-only share on the other {len(oth)} fields ({f3(d2('A4B', B_).g, 2)} per unit "
             f"share) extrapolates to {f3(pr_line('A4B', 'nursing'), 2)} at nursing's share and "
             f"{f3(pr_line('A4B', 'special_education'), 2)} at special education's"
             f"{', values a normal-score slope (a correlation) cannot take' if max(pr_line('A4B', f_) for f_ in DROP2) < -1 else ''}. "
             f"So the per-unit coefficient on all "
             f"{int(ba4b.k)} fields mixes a steep trend among the other fields with two fields off that trend; it is not a "
             f"usable dose-response slope above a share of about {max(oth_rng[B_][1], oth_rng[S_][1]):.1f}, and whether it is "
             f"significant depends on the two fields together: in A4B, of the {lo_bb['k']} single-field drops {lo_bb['nsig']} reach Wald "
             f"p < 0.05 for the bachelor's-only share (without nursing {f3(lb_n.g, 2)}, p {fp_(lb_n.p_wald)}; without special "
             f"education {f3(lb_s.g, 2)}, p {fp_(lb_s.p_wald)}), while dropping both gives {f3(d2('A4B', B_).g, 2)} "
             f"(permutation p {fp_(d2('A4B', B_).p_perm)}).")
    L.append(f"- **ACS licensed-occupation share, bachelor's-only holders: survives the same status controls once leverage "
             f"is handled; not the field-type control (answer 2a).** Linear, per unit share (as in the previous version): {pm(ba3)} raw, {pm(ba4)} "
             f"selectivity-controlled ({f3(ba5.g, 2)} with institution RE), {pm(ba4b)} with the brand slope, {pm(ba4e)} with "
             f"institution-wide earnings as well, {pm(a7bp)} in the status model A7; field-cluster bootstrap 95% CIs "
             f"{ci(fb_b4.boot_lo, fb_b4.boot_hi)} (A4) and {ci(fb_b4b.boot_lo, fb_b4b.boot_hi)} (A4B). Rank form, per SD of the "
             f"share: A3 {prr(lv('A3', B_))}, A4 {prr(lv('A4', B_))}, A4B {prr(lv('A4B', B_))}, A4E {prr(lv('A4E', B_))}, A7 "
             f"{prr(lv('A7', B_))}; field-cluster bootstrap 95% CI at A4B {ci(fr_b.boot_lo, fr_b.boot_hi, 3)}. Winsorized, "
             f"square-root and log forms: {prange(rob('A4B', B_))} at A4B, {prange(rob('A4E', B_))} at A4E, "
             f"{prange(rob('A7', B_))} at A7. Field-level Spearman of per-field slopes with the share: A4 {psp(spr('A4', B_))}, "
             f"A4B {psp(spr('A4B', B_))}, A4E {psp(spr('A4E', B_))}; without nursing and special education "
             f"{f3(spr('A4B', B_).spearman_drop2, 2)} at A4B (permutation p {fp_(spr('A4B', B_).p_perm_drop2)}). The "
             f"health-practitioner component carries the linear version raw ({f3(h3.g, 2)}, SE {h3.se:.2f}) more than after "
             f"controls ({f3(h4.g, 2)}, SE {h4.se:.2f}, permutation p {fp_(h4.p_perm)}; with brand {f3(h4b.g, 2)}, p "
             f"{fp_(h4b.p_perm)}); the K-12 teacher component is uninformative ({f3(k3.g, 2)}, SE {k3.se:.2f} raw); these two "
             f"components were not re-run in leverage-resistant forms. In the status model A7 (per unit share) the health "
             f"component gives {pp(a7hp)} and the K-12 component {pp(a7kp)} on the prestige slope.")
    L.append(f"- **The two shares are not separated by these data.** The all-degree share counts physicians, lawyers and "
             f"psychologists, so professional-school pipeline fields score high: biology {fsi.loc['biology', S_]:.2f}, history "
             f"{fsi.loc['history', S_]:.2f}, chemistry {fsi.loc['chemistry', S_]:.2f} among all degree holders against "
             f"{fsi.loc['biology', B_]:.2f}, {fsi.loc['history', B_]:.2f} and {fsi.loc['chemistry', B_]:.2f} among "
             f"bachelor's-only holders; lower coupling there could reflect who is left in the Scorecard earnings sample rather "
             f"than pay set by setting. That concern was the reason for the bachelor's-only share. On the rank scale the two "
             f"shares give {'the same' if same_rank else 'different'} moderation at A4B ({f3(rs4b.g, 3)} vs {f3(rb4b.g, 3)} per SD; "
             f"field-cluster CIs {ci(fr_s.boot_lo, fr_s.boot_hi, 3)} and {ci(fr_b.boot_lo, fr_b.boot_hi, 3)}), and their linear "
             f"coefficients differ through the two high-share fields. So these data do not tell a pipeline reading from a "
             f"setting reading; since the bachelor's-only share, which leaves out most pipeline occupations, gives "
             f"{'the same' if same_rank else 'a similar'} rank-scale moderation, the moderation is not carried by pipeline "
             f"fields alone.")
    L.append(f"- **Scale: net of selectivity the moderation is also of the log premium; before controls it is not visible in "
             f"log points.** Linear, per unit share (as in the previous version; same {int(gl3s.k)} fields, log points per SD "
             f"of prestige): all-degree {pm3(gl3s)} raw, {pm3(gl4s)} selectivity-controlled, {f3(gl5s.g, 3)} (permutation p "
             f"{fp_(gl5s.p_perm)}) with institution RE; on all {int(gl1s.k)} fields without controls {f3(gl1s.g, 3)} "
             f"(permutation p {fp_(gl1s.p_perm)}); bachelor's-only {pm3(gl3b)} raw and {pm3(gl4b)} controlled. These have the "
             f"same leverage problem. Rank form, per SD of the share (all-degree, bachelor's-only): L3 {prr(rl3[S_])}, "
             f"{prr(rl3[B_])}; L4 {prr(rl4[S_])}, {prr(rl4[B_])}; L4B (+ brand) {prr(rl4b[S_])}, {prr(rl4b[B_])}; L5 (+ "
             f"institution RE) {prr(lv('L5', S_))}, {prr(lv('L5', B_))}; the average-field L4 slope is "
             f"{f3(rl4[S_].b_main, 4)}. Field-level Spearman of per-field log slopes with the share: L3 {psp(spr('L3', S_))}, "
             f"{psp(spr('L3', B_))}; L4 {psp(spr('L4', S_))}, {psp(spr('L4', B_))}; L4B {psp(spr('L4B', S_))}, "
             f"{psp(spr('L4B', B_))}. Without nursing and special education, linear per unit share: L3 {f3(d2('L3', S_).g, 3)} "
             f"and {f3(d2('L3', B_).g, 3)} (permutation p {fp_(d2('L3', S_).p_perm)}, {fp_(d2('L3', B_).p_perm)}); L4 "
             f"{f3(d2('L4', S_).g, 3)} and {f3(d2('L4', B_).g, 3)} (p {fp_(d2('L4', S_).p_perm)}, {fp_(d2('L4', B_).p_perm)}); "
             f"L4B {f3(d2('L4B', S_).g, 3)} and {f3(d2('L4B', B_).g, 3)} (p {fp_(d2('L4B', S_).p_perm)}, "
             f"{fp_(d2('L4B', B_).p_perm)}). Winsorized, square-root and log forms: L4 {prange(rob('L4', S_))} and "
             f"{prange(rob('L4', B_))}; L4B {prange(rob('L4B', S_))} and {prange(rob('L4B', B_))}. Read together: net of "
             f"selectivity (L4) a higher licensed share goes with a smaller log premium per SD of status as well as a looser "
             f"status ordering of earnings, for the all-degree share under every check and for the bachelor's-only share in the "
             f"rank form, the field-level Spearman and the fit without the two fields but not in every transform "
             f"({weak_forms('L4', B_)}). With the brand slope also controlled (L4B) the log-scale evidence is weaker: rank-form "
             f"permutation p {fp_(rl4b[S_].p_perm)} and {fp_(rl4b[B_].p_perm)}, field-level Spearman p "
             f"{fp_(spr('L4B', S_).p_perm)} and {fp_(spr('L4B', B_).p_perm)}, {nf('L4B', S_)} and {nf('L4B', B_)} of 4 "
             f"transforms below 0.05, fit without the two fields p {fp_(d2('L4B', S_).p_perm)} and {fp_(d2('L4B', B_).p_perm)}. "
             f"In raw log slopes (L3) there is none. The previous \"fit, not premium\" reading is withdrawn; the log-premium "
             f"result is less secure than the rank-scale one.")
    dummy_ns = c1l.p_wald >= 0.05 and c2l.p_wald >= 0.05
    a7_ns = a7s.p_wald >= 0.05 and a7g.p_wald >= 0.05
    L.append(f"- **Status pricing in general is lower where the licensed share is high.** Selectivity pricing (earnings on SAT_AVG "
             f"alone): {f3(c1s.g, 2)} per unit all-degree share (SE {c1s.se:.2f}, permutation p {fp_(c1s.p_perm)}), "
             f"{f3(c1ba.g, 2)} per unit bachelor's-only share (p {fp_(c1ba.p_perm)}); on log earnings {f3(c1Ls.g, 3)} (p "
             f"{fp_(c1Ls.p_perm)}) and {f3(c1Lb.g, 3)} (p {fp_(c1Lb.p_perm)}). Brand pricing (academia-wide rank alone): "
             f"{f3(c2s.g, 2)} (p {fp_(c2s.p_perm)}) and {f3(c2ba.g, 2)} (p {fp_(c2ba.p_perm)}); on log earnings {f3(c2Ls.g, 3)} "
             f"(p {fp_(c2Ls.p_perm)}) and {f3(c2Lb.g, 3)} (p {fp_(c2Lb.p_perm)}). The ex-ante dummy gives "
             f"{f3(c1l.g, 2)} (Wald p {fp_(c1l.p_wald)}) and {f3(c2l.g, 2)} (Wald p {fp_(c2l.p_wald)})"
             f"{', neither significant' if dummy_ns else ''}. With prestige, SAT and brand in one model (A7), the all-degree "
             f"share moderates the prestige slope by {f3(a7p.g, 2)} (permutation p {fp_(a7p.p_perm)}) and the SAT and brand "
             f"slopes by {f3(a7s.g, 2)} (Wald p {fp_(a7s.p_wald)}) and {f3(a7g.g, 2)} (Wald p {fp_(a7g.p_wald)})"
             f"{'; the three are collinear within fields, so the split between them is weakly identified' if a7_ns else ''}; "
             f"the bachelor's-only share gives {f3(a7bp.g, 2)}, {f3(a7bs.g, 2)} and {f3(a7bg.g, 2)} (permutation p "
             f"{fp_(a7bp.p_perm)}, {fp_(a7bs.p_perm)}, {fp_(a7bg.p_perm)}). These are linear per-unit coefficients; in the rank "
             f"form (per SD of the share; all-degree, bachelor's-only) selectivity pricing falls by {prr(lv('C1', S_))} and "
             f"{prr(lv('C1', B_))} (log earnings: {prr(lv('C1L', S_))} and {prr(lv('C1L', B_))}) and brand pricing by "
             f"{prr(lv('C2', S_))} and {prr(lv('C2', B_))} (log: {prr(lv('C2L', S_))} and {prr(lv('C2L', B_))}).\n")
    L.append(f"**3. Heterogeneity of field slopes.** Between-field SD of the prestige slope (normal-score units, about a "
             f"Spearman): raw {Hs.loc['A1', 'sd_P']:.3f} [{Hs.loc['A1', 'sd_P_lo']:.3f}, {Hs.loc['A1', 'sd_P_hi']:.3f}] "
             f"(variance {Hs.loc['A1', 'sd_P'] ** 2:.4f}) on all {Hs.loc['A1', 'k']} fields; {Hs.loc['A3', 'sd_P']:.3f} on the "
             f"selectivity sample; {Hs.loc['A4', 'sd_P']:.3f} [{Hs.loc['A4', 'sd_P_lo']:.3f}, {Hs.loc['A4', 'sd_P_hi']:.3f}] "
             f"(variance {Hs.loc['A4', 'sd_P'] ** 2:.4f}) with selectivity controls (boundary LRT p "
             f"{fp_(Hs.loc['A4', 'lrt_p_P'])}); {Hs.loc['A4B', 'sd_P']:.3f} adding the brand slope (LRT p "
             f"{fp_(Hs.loc['A4B', 'lrt_p_P'])}); {Hs.loc['A5', 'sd_P']:.3f} with institution RE. Selectivity controls "
             f"remove about {1 - (Hs.loc['A4', 'sd_P'] / Hs.loc['A3', 'sd_P']) ** 2:.0%} of the between-field variance on the "
             f"same sample; the rest is {'still' if het_A4 else 'not'} distinguishable from zero. In the controlled model the "
             f"all-degree share leaves a slope SD of {s4.tau:.3f} (from {s4.tau_null:.3f} on the same {int(s4.k)} fields, about "
             f"{share_expl:.0%} of that variance accounted for{'; at the boundary' if at0(s4.tau) else ''}; the two-step "
             f"meta-regression leaves a residual tau of {fl_2s4.tau:.3f}) and the bachelor's-only share {ba4.tau:.3f}. The "
             f"mean slope falls from {f3(Hs.loc['A3', 'b_main'], 3)} to {f3(Hs.loc['A4', 'b_main'], 3)} (A3 to A4; "
             f"{f3(Hs.loc['A4B', 'b_main'], 3)} with brand), in line with the +0.43 to +0.14 of scripts/55. Institution random "
             f"intercepts alone take the mean slope from {f3(Hs.loc['A1', 'b_main'], 3)} to {f3(Hs.loc['A2', 'b_main'], 3)} "
             f"(log scale: {f3(Hs.loc['L1', 'b_main'], 3)} to {f3(Hs.loc['L2', 'b_main'], 3)} log points per SD of prestige; "
             f"scripts/55's within-institution design: +0.084 to +0.013), while the between-field SD of the slope changes less "
             f"({Hs.loc['A1', 'sd_P']:.3f} to {Hs.loc['A2', 'sd_P']:.3f}).\n")
    # ------------------------------------------------------------------ tables
    L.append("## Key numbers\n")
    ctrl_lab = {"A3": "none", "A4": "selectivity", "A4B": "selectivity + brand",
                "A4E": "selectivity + brand + institution-wide earnings", "A5": "selectivity + institution RE",
                "A7": "status model (P, SAT, brand random slopes)", "L3": "none (log earnings)",
                "L4": "selectivity (log earnings)", "L4B": "selectivity + brand (log earnings)",
                "L5": "selectivity + institution RE (log earnings)", "C1": "none; slope on SAT_AVG (selectivity pricing)",
                "C2": "none; slope on brand (brand pricing)", "C1L": "none; slope on SAT_AVG (log earnings)",
                "C2L": "none; slope on brand (log earnings)"}
    ctrl_lab["A1"] = "none (all fields)"
    L.append("### (k0) At a glance: licensing moderation of the prestige slope, cell level vs field level\n")
    L.append(f"Cell level: g = change in the within-field prestige slope per unit share (linear; dummy: licensed vs not) or per SD "
             f"of the share (rank normal score), with the field-label permutation p ({N_PERM} permutations) in brackets. "
             f"Normal-score specs in slope units (about a Spearman); L-specs in log points per SD of prestige. '+ field type' = "
             f"the same rank-form fit with an engineering/CS/math/business dummy (CIP-2 11/14/27/52) as a second cross-level "
             f"moderator, Kennedy residual permutation p (table k1). Field level: two-step (per-field OLS slope with the same "
             f"controls, then REML meta-regression; per unit share; permutation p). Fields / cells = the fits with an ACS "
             f"share (the dummy uses every field of the sample: {Hs.loc['A1', 'k']} on all fields, {Hs.loc['A3', 'k']} on the "
             f"selectivity sample). n/a = not run for that spec. Published field-level value (scripts/20, README Result 2): "
             f"about {f3(-PUBLISHED_LIC['b_gap'], 2)} per unit strict share. Linear per-unit coefficients lean on nursing and "
             f"special education; read them with the rank columns.\n")
    L.append("| spec | controls | fields | cells | strict all-degree, linear | strict all-degree, rank | strict all-degree, rank, "
             "+ field type | strict bachelor's-only, linear | strict bachelor's-only, rank | broad, linear | broad, rank | "
             "ex-ante dummy | field level (two-step), strict all-degree, linear |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def rank_or_ft(sk, mk):
        if sk in LEV_SPECS:
            return lv(sk, mk)
        hit = FTP[(FTP.spec == sk) & (FTP.moderator == mk) & (FTP.control == "none") & (FTP.form == RNS)
                  & (FTP["sample"] == ALLF)]
        return hit.iloc[0] if len(hit) else None
    for sk in ("A1", "A3", "A4", "A4B", "A4E", "A5", "A7", "L3", "L4", "L4B"):
        nd = 4 if sk.startswith("L") else 3
        cg = lambda r: f"{f3(r.g, nd)} ({fp_(r.p_perm)})" if r is not None else "n/a"
        f2 = FL[(FL.spec == sk) & (FL.moderator == S_) & FL.method.str.startswith("two-step")]
        r_s = ga(sk, S_, "P")
        rb_ = bloc_ct(sk, S_) if sk in FT_SPECS else None
        L.append(f"| {sk} | {ctrl_lab[sk]} | {int(r_s.k)} | {int(r_s.n):,} | {cg(r_s)} | {cg(rank_or_ft(sk, S_))} | "
                 f"{cg(rb_)} | {cg(ga(sk, B_, 'P'))} | {cg(rank_or_ft(sk, B_))} | {cg(ga(sk, BR, 'P'))} | "
                 f"{cg(rank_or_ft(sk, BR))} | {cg(ga(sk, 'lic', 'P'))} | {cg(f2.iloc[0]) if len(f2) else 'n/a'} |")
    L.append("")
    # ---- (k1) field-type controls
    L.append("### (k1) Field-type check: licensed-share moderation of the prestige slope with field-level controls\n")
    L.append(f"Selectivity sample, fields with an ACS share. Share in rank normal scores, per SD; g (permutation p, {N_PERM} "
             f"draws). Control = a field-level covariate entered as covariate x every random slope; the share's p is a Kennedy "
             f"residual permutation (share residualized on the control across fields, residuals permuted); for 'within "
             f"CIP-2' the prestige slope gets CIP-2 fixed effects and the share is permuted within CIP-2. Control coefficient "
             f"= the control x prestige coefficient in the all-degree model (dummies: 0/1; continuous controls: per SD), "
             f"with its Kennedy p for the bloc and Wald p otherwise. 'bloc alone' = the dummy as the only moderator on the "
             f"same fields. Collinearity with the bloc (Spearman of the share with the dummy): strict all-degree "
             f"{f3(FTC_.loc[S_, 'spearman_bloc'], 2)}, strict bachelor's-only {f3(FTC_.loc[B_, 'spearman_bloc'], 2)}, broad "
             f"{f3(FTC_.loc[BR, 'spearman_bloc'], 2)}; share ranges in the bloc / other fields: strict all-degree "
             f"{FTC_.loc[S_, 'bloc_lo']:.3f}-{FTC_.loc[S_, 'bloc_hi']:.3f} / {FTC_.loc[S_, 'other_lo']:.3f}-"
             f"{FTC_.loc[S_, 'other_hi']:.3f}, bachelor's-only {FTC_.loc[B_, 'bloc_lo']:.3f}-{FTC_.loc[B_, 'bloc_hi']:.3f} / "
             f"{FTC_.loc[B_, 'other_lo']:.3f}-{FTC_.loc[B_, 'other_hi']:.3f}, broad {FTC_.loc[BR, 'bloc_lo']:.3f}-"
             f"{FTC_.loc[BR, 'bloc_hi']:.3f} / {FTC_.loc[BR, 'other_lo']:.3f}-{FTC_.loc[BR, 'other_hi']:.3f}.\n")
    L.append("| spec | control | k | strict all-degree | strict bachelor's-only | broad | control coefficient (strict all-degree model) |")
    L.append("|---|---|---|---|---|---|---|")
    for cs in ["none", "bloc", "cip2_fe"] + [c for c in FT_SETS if c not in ("bloc", "cip2_fe")]:
        for sk in (FT_SPECS if cs in ("none", "bloc", "cip2_fe") else FT_SPECS_SHORT):
            nd = 4 if sk.startswith("L") else 3
            r3 = [ft(sk, mk, cs) for mk in LEV_MODS]
            c_ = r3[0]
            if cs == "bloc":
                cc = f"{f3(c_.cov_g, nd)} (Kennedy p {fp_(c_.cov_p_perm)})"
            elif cs in ("none", "cip2_fe"):
                cc = ""
            else:
                cc = f"{f3(c_.cov_g, nd)} (Wald p {fp_(c_.cov_p_wald)})"
            lab_ = "none" if cs == "none" else ("within CIP-2 (stratified)" if cs == "cip2_fe" else FT_SETS[cs])
            L.append(f"| {sk} | {lab_} | {int(c_.k)} | " + " | ".join(f"{f3(r.g, nd)} ({fp_(r.p_perm)})" for r in r3)
                     + f" | {cc} |")
    for sk in FT_SPECS:
        nd = 4 if sk.startswith("L") else 3
        r_ = ftalone(sk)
        L.append(f"| {sk} | bloc alone (dummy as the moderator) | {int(r_.k)} | | | | {f3(r_.g, nd)} (p {fp_(r_.p_perm)}) |")
    L.append("")
    # ---- (k1-ii) forms under the bloc and within CIP-2
    L.append("### (k1-ii) Field-type check by form of the share\n")
    L.append("Same fits as (k1) for the three forms (each per SD of the transformed share); share g (permutation p). Bloc "
             "coefficient = the bloc x prestige coefficient net of the share in that form (Kennedy p).\n")
    L.append("| spec | share | control | linear | square root | rank normal score | bloc coefficient: linear / square root / rank |")
    L.append("|---|---|---|---|---|---|---|")
    for mk in LEV_MODS:
        for cs in ("none", "bloc", "cip2_fe"):
            for sk in FT_SPECS:
                nd = 4 if sk.startswith("L") else 3
                rr = [ft(sk, mk, cs, fn) for fn in FT_FORMS]
                bc = " / ".join(f"{f3(r.cov_g, nd)} ({fp_(r.cov_p_perm)})" for r in rr) if cs == "bloc" else ""
                L.append(f"| {sk} | {mk} | {'within CIP-2' if cs == 'cip2_fe' else cs} | "
                         + " | ".join(f"{f3(r.g, nd)} ({fp_(r.p_perm)})" for r in rr) + f" | {bc} |")
    L.append("")
    # ---- (k2) within the non-bloc and the bloc fields
    L.append("### (k2) Field-type check: fits within the non-bloc fields and within the bloc fields\n")
    L.append(f"Share re-standardized (linear, per SD) or re-ranked (rank normal score, per SD) within the subset; plain "
             f"field-label permutation p ({N_PERM}) within the subset; Wald 95% CI; field-cluster CI = {N_BOOT}-draw "
             f"bootstrap of the rank form (A4, A4B; variance ratio fixed). Field-level Spearman = per-field OLS slope (same "
             f"controls) vs the share within the subset ({N_PERM} label permutations). The full-sample rank-form estimates "
             f"are in table (k0).\n")
    L.append("| subset | spec | share | k | share range | rank: g [Wald 95% CI] (p) | rank: field-cluster 95% CI | linear per SD: g (p) | field-level Spearman (p) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for smp in (NB, BF):
        for mk in LEV_MODS:
            for sk in FT_SPECS:
                nd = 4 if sk.startswith("L") else 3
                rr, rl_ = ft(sk, mk, "none", RNS, smp), ft(sk, mk, "none", "linear", smp)
                bt_ = FTB_[(FTB_["sample"] == smp) & (FTB_.spec == sk) & (FTB_.moderator == mk)]
                sp_ = FTS_[(FTS_["sample"] == smp) & (FTS_.spec == sk) & (FTS_.moderator == mk)]
                L.append(f"| {smp} | {sk} | {mk} | {int(rr.k)} | {rr.share_lo:.3f}-{rr.share_hi:.3f} | {f3(rr.g, nd)} "
                         f"[{f3(rr.lo, nd)}, {f3(rr.hi, nd)}] ({fp_(rr.p_perm)}) | "
                         f"{ci(bt_.boot_lo.iloc[0], bt_.boot_hi.iloc[0], nd) if len(bt_) else ''} | "
                         f"{f3(rl_.g, nd)} ({fp_(rl_.p_perm)}) | "
                         f"{f3(sp_.spearman.iloc[0], 2) + ' (' + fp_(sp_.p_perm.iloc[0]) + ')' if len(sp_) else ''} |")
    L.append("")
    L.append("### (b0) Licensing moderation of the status slopes: linear vs leverage-resistant forms of the share (same 40 fields; no field-type control)\n")
    sd_sh = {mk: float(LVp[LVp.moderator == mk].sd_share.iloc[0]) for mk in LEV_MODS}
    L.append(f"Each cell: g (field-label permutation p, {N_PERM} permutations). Linear, per unit = the per-unit coefficient of "
             f"table (b0-ii). All other columns are per SD of the (transformed) share, so forms and shares are comparable: linear "
             f"per SD = per unit x SD of the share ({sd_sh['acs_strict']:.3f} strict all-degree, {sd_sh['acs_strict_ba']:.3f} "
             f"strict bachelor's-only, {sd_sh['acs_broad']:.3f} broad); winsorized at {WINS:.2f}, square root, log(share + 0.01) and rank normal score are each "
             f"z-scored across the 40 fields. Normal-score specs in slope units (about a Spearman); L-specs in log points per SD "
             f"of prestige, 4 decimals. Rank CI = {N_BOOT}-draw field-cluster bootstrap of the rank form (variance ratios fixed; "
             f"specs without institution RE). Field-level Spearman = per-field OLS slope (same variables) vs the share, "
             f"{N_PERM} label permutations (no two-step version of A7). Without nursing + special education = linear, per unit "
             f"share, 38 fields. C-specs moderate the SAT or brand slope instead of the prestige slope.\n")
    L.append("| spec | controls | share | linear, per unit | linear, per SD | winsorized, per SD | square root, per SD | "
             "log, per SD | rank normal score, per SD | rank: field-cluster 95% CI | field-level Spearman | without nursing + special ed., per unit |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mk in LEV_MODS:
        for sk in LEV_SPECS:
            nd = 4 if sk.startswith("L") else 3
            cl = lambda r: f"{f3(r.g, nd)} ({fp_(r.p_perm)})"
            r0 = ga(sk, mk, SPECS[sk]["slopes"][0])
            bt_ = BT[(BT.moderator == f"{mk}_rns") & (BT.spec == sk) & BT.scheme.str.startswith("field")]
            bci = ci(bt_.boot_lo.iloc[0], bt_.boot_hi.iloc[0], nd) if len(bt_) else "n/a"
            sp_ = SPR[(SPR.spec == sk) & (SPR.moderator == mk)]
            spc = f"{f3(sp_.spearman.iloc[0], 2)} ({fp_(sp_.p_perm.iloc[0])})" if len(sp_) else "n/a"
            L.append(f"| {sk} | {ctrl_lab[sk]} | {mk} | {cl(r0)} | " + " | ".join(cl(lv(sk, mk, fn)) for fn in FORMS)
                     + f" | {bci} | {spc} | {cl(d2(sk, mk))} |")
    L.append("")
    L.append("### (b0-ii) Linear moderator: details (as in the previous version, plus L4B)\n")
    L.append("g per unit share (dummy: licensed vs not). Normal-score specs in slope units (about a Spearman); L-specs in log "
             "points per SD of prestige. Permutation p = field-label permutation. Field-cluster CI = "
             f"{N_BOOT}-draw field-cluster bootstrap with variance ratios fixed (BOOT specs only). The linear share coefficients "
             "are dominated by nursing and special education; read them with table (b0).\n")
    L.append("| spec | controls | moderator | k | g | Wald SE | perm. p | field-cluster 95% CI | slope SD with / without |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    ladder = [(k_, ctrl_lab[k_]) for k_ in ("A3", "A4", "A4B", "A4E", "A5", "A7", "L3", "L4", "L4B", "L5")]
    for mk in ("acs_strict", "acs_strict_ba", "lic"):
        for sk, lab_ in ladder:
            r = ga(sk, mk, "P")
            bt_ = BT[(BT.moderator == mk) & (BT.spec == sk) & BT.scheme.str.startswith("field")]
            bci = ci(bt_.boot_lo.iloc[0], bt_.boot_hi.iloc[0], 3) if len(bt_) else ""
            L.append(f"| {sk} | {lab_} | {mk} | {int(r.k)} | {f3(r.g)} | {r.se:.3f} | {fp_(r.p_perm)} | {bci} | "
                     f"{r.tau:.3f}{' (at 0)' if at0(r.tau) else ''} / {r.tau_null:.3f} |")
    L.append("")
    L.append("### (a) Heterogeneity of within-field slopes (models without moderators)\n")
    L.append("Slope units: normal-score regression coefficient (about a Spearman correlation); L-specs: log points per SD of "
             "prestige (C1L/C2L: per SD of SAT / brand). SD = REML between-field SD of the slope, 95% profile-likelihood CI, "
             "boundary LRT p (0.5 chi2_0 + 0.5 chi2_1); the profile CI and LRT were not computed for L2 and L5 (n/a).\n")
    L.append("| spec | sample / model | k fields | cells | institutions | mean slope (SE) | slope SD [95% CI] | slope variance | LRT p | institution SD | residual SD |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for sk, r in Hs.iterrows():
        c = r["main"]
        lo_, hi_ = r.get(f"sd_{c}_lo", np.nan), r.get(f"sd_{c}_hi", np.nan)
        L.append(f"| {sk} | {r['label']} (slope on {c}) | {r['k']} | {r['n']} | {r['n_inst']} | {f3(r['b_main'])} ({r['se_b_main']:.3f}) | "
                 f"{r['sd_' + c]:.3f} [{f3(lo_, 3, False)}, {f3(hi_, 3, False)}] | {r['sd_' + c] ** 2:.4f} | "
                 f"{fp_(r.get('lrt_p_' + c, np.nan))} | "
                 f"{f3(r.get('sd_inst', np.nan), 3, False)} | {r['s_resid']:.3f} |")
    if "sd_SAT" in Hs.columns:
        r = Hs.loc["A7"]
        L.append(f"\nA7 (P, SAT, brand together): slope SD for SAT {r['sd_SAT']:.3f} [{f3(r['sd_SAT_lo'], 3, False)}, "
                 f"{f3(r['sd_SAT_hi'], 3, False)}] (LRT p {fp_(r['lrt_p_SAT'])}), for brand {r['sd_G']:.3f} "
                 f"[{f3(r['sd_G_lo'], 3, False)}, {f3(r['sd_G_hi'], 3, False)}] (LRT p {fp_(r['lrt_p_G'])}).\n")
    L.append("### (b) Moderator x slope interactions (cell level)\n")
    L.append("g = change in the within-field slope per unit of the moderator (0 to 1 share; dummy 0/1; bridge moderators per SD). "
             f"Wald CI from REML (understates between-field uncertainty; see table d). Permutation p: moderator values permuted "
             f"across fields ({N_PERM} draws), statistic = GLS t with the variance parameters of the model without the moderator. "
             "SD with / without = slope SD with and without the moderator on the same fields; \"(at 0)\" marks a REML estimate "
             "at the boundary after the boundary scan.\n")
    L.append("| spec | moderator | term | k | cells | g | 95% CI | Wald p | perm. p | slope at moderator 0 | slope SD with / without |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    order = ["lic", "acs_strict", "acs_broad", "acs_strict_ba", "acs_k12_ba", "acs_health_ba", "tail_hi", "occ_hhi", "PxSAT"]
    for sk in SPECS:
        for mk in order:
            sub = A[(A.spec == sk) & (A.moderator == mk)]
            for r in sub.itertuples():
                L.append(f"| {sk} | {mk} | {r.term} | {int(r.k)} | {int(r.n)} | {f3(r.g)} | [{f3(r.lo)}, {f3(r.hi)}] | "
                         f"{fp_(r.p_wald)} | {fp_(r.p_perm)} | {f3(r.b_main)} | {r.tau:.3f}{' (at 0)' if at0(r.tau) else ''} "
                         f"/ {r.tau_null:.3f} |")
    L.append("\n### (c) Same moderators at the field level (same fields)\n")
    L.append("| spec | moderator | method | k | g | 95% CI | p | perm. p | residual tau |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in FL.itertuples():
        L.append(f"| {r.spec} | {r.moderator} | {r.method} | {r.k} | {f3(r.g)} | [{f3(r.lo)}, {f3(r.hi)}] | {fp_(r.p)} | "
                 f"{fp_(r.p_perm)} | {f3(r.tau, 3, False)} |")
    L.append(f"\nPublished field-level reference (README Result 2, scripts/20): gap on strict licensure share b = +0.66 "
             f"[+0.33, +0.97] with academic absorption in the model, i.e. about -0.66 per unit share in coupling units.\n")
    L.append("### (d) Raw vs controlled interaction on the same sample: bootstrap\n")
    n_fb = BT[BT.scheme.str.startswith("field")].groupby("moderator").n_draws.min()
    L.append(f"{N_BOOT} draws per scheme; each spec re-estimated by GLS with its REML variance ratios held fixed (so variance-"
             "component uncertainty is ignored). **Field-cluster**: fields drawn with replacement, each copy its own field; "
             "this carries the between-field sampling variance and is the relevant interval for a field-level moderator. "
             "**Institution**: institutions drawn with replacement inside the realized fields; it is conditional on the "
             "realized field slopes (within-field sampling error only), so it is shown for the raw-vs-controlled "
             "differences only, not for the levels. `A4-A3` = controlled minus raw; ratio = controlled / raw point "
             f"estimate. For the dummy, draws with no licensed field are dropped (valid field-cluster draws: "
             f"{int(n_fb.get('lic', N_BOOT))}). `*_rns` = the share as rank normal scores, per SD (specs without institution "
             f"RE only).\n")
    L.append("| moderator | spec | g | field-cluster 95% CI | field-cluster SE | institution 95% CI (conditional on fields) | institution SE | ratio |")
    L.append("|---|---|---|---|---|---|---|---|")
    for mk in ("lic", "acs_strict", "acs_strict_ba", "acs_strict_rns", "acs_strict_ba_rns", "acs_broad_rns"):
        present = set(BT[BT.moderator == mk].spec)
        for sk in [s_ for s_ in list(BOOT_SPECS) + [f"{s_}-A3" for s_ in BOOT_SPECS[1:]] if s_ in present]:
            f_ = gbt(mk, sk, "field"); i_ = gbt(mk, sk, "institution")
            is_diff = "-" in sk
            L.append(f"| {mk} | {sk} | {f3(f_.g)} | [{f3(f_.boot_lo)}, {f3(f_.boot_hi)}] | {f_.boot_se:.3f} | "
                     + (f"[{f3(i_.boot_lo)}, {f3(i_.boot_hi)}] | {i_.boot_se:.3f} | " if is_diff else "not shown | not shown | ")
                     + f"{f3(getattr(f_, 'ratio', np.nan), 2, False) if is_diff else ''} |")
    L.append("\n### (e) Dropping fields\n")
    miss = [LAB.get(f, f) for f in DROP_SET if f not in sat_fields]
    if miss:
        L.append(f"{' and '.join(miss)} {'is' if len(miss) == 1 else 'are'} not in the selectivity sample, so for the A4, A4B "
                 f"and A5 rows only the other field is dropped (k falls by one).\n")
    L.append("| spec | moderator | fields dropped | k | g | 95% CI | Wald p | slope SD |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in RB[RB["drop"] == "nursing + communication disorders"].itertuples():
        L.append(f"| {r.spec} | {r.moderator} | nursing + communication disorders | {r.k} | {f3(r.g)} | [{f3(r.lo)}, {f3(r.hi)}] | "
                 f"{fp_(r.p_wald)} | {r.tau:.3f}{' (at 0)' if at0(r.tau) else ''} |")
    for r in D2.itertuples():
        L.append(f"| {r.spec} | {r.moderator} | nursing + special education (perm. p {fp_(r.p_perm)}) | {r.k} | {f3(r.g)} | "
                 f"[{f3(r.lo)}, {f3(r.hi)}] | {fp_(r.p_wald)} | {r.tau:.3f}{' (at 0)' if at0(r.tau) else ''} |")
    for sk, mk in LOFO:
        o = lofo(sk, mk)
        L.append(f"| {sk} | {mk} | each field in turn ({o['k']} fits; {o['nsig']} with Wald p < 0.05) | | {f3(o['lo'])} "
                 f"(without {o['lo_f']}) to {f3(o['hi'])} (without {o['hi_f']}) | | max {fp_(o['pmax'])} | "
                 f"{o['tau_lo']:.3f}-{o['tau_hi']:.3f} |")
    L.append("\n### (f) Power and equivalence of the field-level nulls\n")
    L.append("Direction = sign the theory predicts. MDE = smallest true effect with 80% power for the two-sided 5% test as run. "
             "Bound = one-sided 95% confidence bound in the theory's direction (true-score scale for B1-B3; see Method). "
             "90% interval = two one-sided tests at 5% each (TOST). Smallest TOST margin = the smallest symmetric equivalence "
             "margin +/-D that TOST at 5% would accept (D = the larger absolute end of the 90% interval, same scale as the "
             "estimate); a null is equivalent to zero within D, not within anything smaller.\n")
    L.append("| test | k | estimate (p) | scale | MDE | power at moderate / large | 90% interval | bound | smallest TOST margin | check (observed-score / alt.) | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in PT.itertuples():
        if r.test == "B1" and r.measure != "tail_hi":
            continue
        est = f"{f3(r.estimate, 3)} ({fp_(r.p)})"
        cen = " (censored: not bounded)" if bool(r.censored) else ""
        chk = ""
        if r.test in ("B1", "B2"):
            chk = f"obs.-score bound {f3(r.bound_check, 2)}, MDE {r.mde_check:.2f}; reliability of rho_f {r.reliability:.2f}"
        elif r.test == "B4":
            chk = f"simulated-permutation MDE {r.mde_check:.4f}; Welch SE {r.se_check:.4f} vs bootstrap SE {r.se:.4f}"
        elif r.test == "B3":
            chk = f"max power {r.max_power:.2f} at kin r {r.kin_max_power:+.2f}; Moran's I is the estimate"
        elif r.test == "cell":
            chk = f"perm. p {fp_(r.p_perm)}"
        mde = "not reached" if not np.isfinite(r.mde) else f3(r.mde, 3)
        L.append(f"| {r.label} | {r.n} | {est} | {r.scale} | {mde} | {r.power_moderate:.2f} / {r.power_large:.2f} | "
                 f"[{f3(r.ci90_lo, 3)}, {f3(r.ci90_hi, 3)}] | {f3(r.bound, 3)}{cen} | "
                 f"{f3(r.tost, 3, False) if np.isfinite(r.tost) else 'none (not bounded)'} | {chk} | {r.verdict} |")
    L.append("\nAll six dispersion measures (scripts/53), true-score bound in the theory's (positive) direction:\n")
    L.append("| measure | reliable 16: Spearman, bound, verdict | all 50: Spearman, bound, verdict |")
    L.append("|---|---|---|")
    for meas in ["tail_hi", "sd", "iqr", "tail_lo", "p90_10", "cv_level"]:
        rr = b1_rel6[b1_rel6.measure == meas].iloc[0]; ra = b1_all6[b1_all6.measure == meas].iloc[0]
        L.append(f"| {meas} | {f3(rr.estimate, 2)}, {f3(rr.bound, 2)}, {rr.verdict} | {f3(ra.estimate, 2)}, {f3(ra.bound, 2)}, {ra.verdict} |")
    L.append("\n### (g) REML optimizer audit (fits with the prestige-slope SD near 0)\n")
    L.append("This fit = the optimizer used throughout (theta^2 parametrization, five starts, boundary scan). Earlier optimizer = "
             "the previous version's two-start L-BFGS-B in theta, re-run here with this script's thread setting. Nelder-Mead = "
             "on |theta| from five starts. Grid = prestige-slope theta on a 0.0005 grid over [0, 0.4] with the other components "
             "at this fit's values. Deviance = REML deviance (lower is better).\n")
    L.append("| spec | moderator | dropped | k | this fit: slope SD, deviance | g (SE) | deviance at theta_P = 0 | earlier optimizer: slope SD, deviance, g (SE) | Nelder-Mead: theta_P, deviance | grid: theta_P, deviance | all theta: this fit / earlier / Nelder-Mead |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in RC.itertuples():
        L.append(f"| {r.spec} | {r.moderator} | {r.drop or '-'} | {r.k} | {r.sd:.3f}, {r.dev:.3f} | {f3(r.g)} ({r.se:.3f}) | "
                 f"{r.dev_theta0:.3f} | {r.old_sd:.3f}, {r.old_dev:.3f}, {f3(r.old_g)} ({r.old_se:.3f}) | "
                 f"{r.nm_theta:.4f}, {r.nm_dev:.3f} | {r.grid_theta:.4f}, {r.grid_dev:.3f} | "
                 f"{r.theta_all} / {r.old_theta_all} / {r.nm_theta_all} |")
    L.append("")
    # ------------------------------------------------------------------ method
    L.append("## Method\n")
    L.append(f"- **Cells.** (institution, field) pairs with Wapman field prestige (-published field Rank) and Scorecard FoS "
             f"bachelor's EARN_MDN_4YR, FIELDS66, fields with >= {NMIN} institutions: {a['n_all']} cells, {Hs.loc['A1', 'k']} fields, reproducing the "
             f"expanded66 gap map exactly (max |Spearman difference| {a['repro']:.1e}; same n per field). Selectivity sample: "
             f"cells with SAT_AVG, ADM_RATE, PCTPELL, state earnings level, CONTROL and state; fields with >= {NMIN + 2} such "
             f"cells so the field-specific controls of A4 leave >= {DFMIN} residual df: {a['n_sat']} cells, {Hs.loc['A3', 'k']} "
             f"fields (as the broad spec of scripts/55). A4B and A4E add one and two field-specific slopes on the same sample, "
             f"so their smallest field keeps {NMIN + 2 - 8} and {NMIN + 2 - 9} residual df.")
    L.append("- **Scale.** Within each field (and sample), earnings, prestige and the continuous covariates become standardized "
             "van der Waerden normal scores; the private-control dummy is demeaned within field. The within-field OLS slope of "
             "earnings on prestige is then the normal-scores correlation, close to the field's Spearman rho_f: it measures how "
             "closely status orders earnings, not the size of the pay gap. L-specs (and C1L/C2L) use log earnings demeaned "
             "within field and z-scored regressors (log points per SD), i.e. the premium.")
    L.append("- **Model.** y_if = a_f + (b + g M_f + u_f) x_if + controls + v_i + e_if with field fixed intercepts, random field "
             "slopes u_f, crossed institution random intercepts v_i (A2, A5, L2, L5) or institution fixed effects (A6). Controls in "
             "A4-A6, L4-L5: field-specific slopes on SAT_AVG, -ADM_RATE, PCTPELL, leave-one-out state earnings level (as scripts/55) "
             "and private control. A4B (L4B on log earnings) adds a field-specific slope on academia-wide brand G (-Wapman "
             "academia rank); A4E adds "
             "one on institution-wide median earnings 10 years after entry (Scorecard MD_EARN_WNE_P10, all fields). A7: random "
             "field slopes on prestige, SAT_AVG and G, each interacted with M_f, other controls field-specific. C1/C2 (C1L/C2L "
             "on log earnings): earnings on SAT_AVG or G alone. Variance components are independent (no slope-intercept "
             "covariance; the within-field centring makes the intercept covariance moot).")
    L.append(f"- **Estimation.** Profiled REML through the q x q matrix I + Lambda Z'Z Lambda (own code), optimized over the "
             f"variance ratios theta^2 (L-BFGS-B, starts theta = 0.02, 0.05, 0.1, 0.2, 1), then a boundary scan that moves any "
             f"component at 0 to theta = 0.01-0.25 and restarts if the deviance falls; table (g) audits this against "
             f"Nelder-Mead and a grid. Check against statsmodels MixedLM on the A1 null model: slope SD {val['tau_mine']:.5f} vs "
             f"{val['tau_sm']:.5f}, residual SD {val['sig_mine']:.5f} vs {val['sig_sm']:.5f}, mean slope {val['b_mine']:.5f} vs "
             f"{val['b_sm']:.5f}. Profile-likelihood CIs for slope SDs; LRT for SD = 0 against the 50:50 chi-square mixture.")
    L.append(f"- **Inference on g.** Wald from REML, plus a field-label permutation test ({N_PERM} permutations of the moderator "
             "values across fields; statistic = GLS t of g with the variance parameters of the moderator-free model, which do not "
             "depend on the labels, so the test is valid under exchangeable fields and unaffected by institution dependence "
             f"across fields). Bootstrap ({N_BOOT} draws per scheme, variance ratios fixed at each spec's REML estimate): "
             "field-cluster (fields resampled) and institution (institutions resampled within the realized fields, conditional on "
             "them). For specs whose only random effect is the field slope, each field is an independent block with its own "
             "nuisance columns, so the bootstrap GLS is computed from per-field partialled statistics (checked equal to the "
             "full GLS on the data and on the first draw of each scheme).")
    L.append("- **Moderators.** Ex-ante dummy = src.crosswalks.fields.LICENSED_FIELDS. ACS shares: PWGTP-weighted share of a "
             "field's full-time employed graduates (FOD1P) with positive earnings in licensed SOC codes, exactly as scripts/20 "
             "build_anchors (strict: 29-1, 23-1, 25-2, 17-1011, 19-3031; broad adds 29-2, 13-2011, 17-2, 21-1); the all-degree "
             "version is asserted equal to data/interim/mixture_anchors.csv. Bachelor's-only versions restrict to SCHL = 21. "
             "Components: K-12 teachers (25-2) and health practitioners (29-1). Bridge moderators tail_hi (scripts/53) and "
             "occ_hhi (scripts/50) are z-scored across the fields in the fit.")
    L.append(f"- **Leverage checks (the three ACS shares; the broad share was added in the field-type revision; specs "
             f"{', '.join(LEV_SPECS)}).** (i) The share enters linearly, winsorized "
             f"at {WINS:.2f}, as a square root, as log(share + 0.01) and as rank normal scores across the fields "
             f"(van der Waerden), each z-scored across the fields in the fit; the same REML fit and field-label permutation "
             f"test as the main table (permuting a transformed moderator is permuting field labels). (ii) Field-level "
             f"Spearman of the per-field OLS prestige slope (two-step, same variables as the spec) with the share; two-sided "
             f"p from {N_PERM} label permutations; also without nursing and special education. (iii) Linear fit with nursing "
             f"and special education dropped together, with its own moderator-free model for the permutation test. (iv) "
             f"Leave-one-field-out for A4B and L4B as well as A1 and A4. (v) Field-cluster bootstrap of the rank form (scores "
             f"fixed at their values on the observed fields). Winsorizing point {WINS:.2f} (largest share outside the two "
             f"fields: {oth_rng[B_][1]:.2f} bachelor's-only, {oth_rng[S_][1]:.2f} all-degree); it was taken from the "
             f"verification, not tuned here.")
    L.append("- **Field-level comparison.** Two-step: per-field OLS slope (same variables), then REML random-effects meta-"
             "regression with Knapp-Hartung SE and a field-label permutation p (weights from the moderator-free meta-model). "
             "scripts/20 style: WLS of the gap-map rho_f on the moderator with weights 1/se^2 (se as stored, floor 0.08), HC1 SE.")
    L.append(f"- **Field-type check (field_type_checks).** On the {int(FTC_.loc[S_, 'k'])} selectivity-sample fields with "
             f"an ACS share. Field-level controls: a dummy for CIP-2 11/14/27/52 (engineering, CS, math/statistics, "
             f"business; CIP-2 = first CIP-4 family of each field in FIELDS66, as scripts/28 and 57), the same without math, "
             f"the README 'integrated' clusters (11/14/27/45) with and without business, log median Scorecard earnings of "
             f"the field's cells, log institutions per field (gap map), the ACS academic absorption share (scripts/20; "
             f"missing for environmental engineering, so 39 fields) and the log range of published Wapman ranks across the "
             f"field's cells (continuous controls z-scored across fields). Each control enters as control x every random "
             f"slope of the spec. Test for the share: Kennedy residual permutation ({N_PERM} draws) - the share (in the given "
             f"form) is regressed by OLS on [1, control] across fields, the residuals are permuted across fields, and the "
             f"statistic is the GLS t of residual x slope given the base (which holds control x slope), with the variance "
             f"parameters of the model without the share. Within CIP-2: CIP-2 fixed effects on every random slope and the "
             f"share's residuals (deviations from CIP-2 means) permuted within CIP-2; singleton CIP-2 groups drop out. The "
             f"bloc net of the share: the same with the roles swapped. Subsets: non-bloc and bloc fields separately, share "
             f"re-standardized / re-ranked within the subset, plain field-label permutation, field-cluster bootstrap (A4, "
             f"A4B) and field-level Spearman. The rows without a control reproduce table (b0) (same random streams; "
             f"asserted in the script).")
    L.append(f"- **B1/B2 power and bounds.** Moderator values fixed; latent coupling T = s_T (r z_x + sqrt(1 - r^2) e), z_x = "
             f"normal scores of the moderator, s_T^2 = var(observed rho_f) - mean(se^2); observed = T + se_f u with se_f = gap-map "
             f"bootstrap CI width / 3.92. {N_SIM} simulations per r on a 0.025 grid (common random numbers). Power: two-sided "
             "Spearman t-test at 5% as run in scripts/50/53. Bound: r at which P(simulated Spearman <= observed) = 0.05 (upper) or "
             "P(>=) = 0.05 (lower). Check column: Fisher z with var 1.06/(n-3), no measurement error.")
    L.append(f"- **B3.** Latent T = s_T (I - lambda W)^-1 e rescaled to unit mean marginal variance, W = the row-normalized "
             f"faculty-flow matrix of scripts/51 ({b3['n']} reliable fields), plus measurement noise. Power of the right-tail permutation "
             f"test ({N_PERM_MORAN} permutations, {N_SIM_MORAN} simulations per lambda); bound by inverting P(I <= I_obs | lambda) "
             f"({N_SIM} simulations per lambda, grid -0.90 to 0.95). Effect scale: mean correlation between a field's latent "
             f"coupling and its flow-weighted neighbours' (kin correlation); at lambda = 0 it is {b3['kin0']:+.2f}, not 0: across "
             f"only {b3['n']} fields, centring plus a zero diagonal make the expected correlation negative under independence (the "
             f"same small-sample effect that puts E[I] at -1/(n-1)). W is sparse and concentrated ({b3['nnz_lo']}-{b3['nnz_hi']} "
             f"non-zero neighbours per field, effective number of neighbours 1/sum(w^2) {b3['eff_nb_lo']:.1f}-"
             f"{b3['eff_nb_hi']:.1f}). Moderate (0.30) is at "
             f"lambda {b3['lam_mod']:.2f}, large (0.50) at {b3['lam_large']:.2f}; a rise of 0.30 above the lambda = 0 value "
             f"is at lambda {b3['lam_mod_x']:.2f}.")
    L.append(f"- **B4.** SE from the two-stage bootstrap 95% CI of scripts/52 ((hi - lo) / 3.92, normal approximation); analytic "
             f"MDE (1.96 + 0.84) SE. Check: simulated power of scripts/52's two-sided field-label permutation test "
             f"({N_PERM_CT} permutations, {N_SIM_CT} simulations per effect; field slopes resampled from the group-centred "
             f"residuals, effect added to the {b4['n'].split(' vs ')[0]} integrated fields).")
    L.append("- **Cell-level bridge (tail_hi, occ_hhi).** r-scale = g / (slope SD of the moderator-free model on the same fields), "
             "i.e. the correlation of the moderator with the true field slope; bound = g -/+ 1.645 SE on that scale (slope SD held "
             "fixed).\n")
    # ------------------------------------------------------------------ caveats
    L.append("## Caveats\n")
    L.append(f"- **Field-level moderators still have field-level power.** The cell model uses {int(A.n.min()):,}-{int(A.n.max()):,} "
             f"cells, but a moderator measured once per field is identified from {int(A.k.min())}-{int(A.k.max())} fields; its SE "
             f"is close to the two-step SE ({s1.se:.2f} vs {fl_2s.se:.2f} for the strict share, raw, all fields). The cell model "
             "changes what is held fixed (selectivity inside the field, institution effects), not how many independent units "
             "there are.")
    L.append(f"- **Wald SEs are too narrow for a field-level moderator; use the permutation p.** For A4 x all-degree share: Wald "
             f"SE {s4.se:.3f}, two-step Knapp-Hartung {fl_2s4.se:.3f}, field-cluster bootstrap {fb_s4.boot_se:.3f}. The "
             f"field-cluster bootstrap is itself erratic here because a few high-share fields (nursing, special education) carry "
             f"much of the leverage, and it holds variance ratios fixed. The institution bootstrap in table (d) is conditional on "
             f"the realized fields and must not be read as a CI for g itself.")
    L.append("- **What the licensed share measures.** The ACS share describes where a field's graduates (all or bachelor's-only, "
             "nationally) work, not the graduates of the institutions in the cell. The all-degree share also rises with "
             "professional-school pipelines (medicine, law), where Scorecard 4-year earnings (working, not enrolled) are measured "
             "on graduates who did not continue, so lower coupling there can reflect who is left in the earnings sample rather "
             f"than pay set by setting. The bachelor's-only share avoids most of that. Both shares are highly skewed (highest "
             f"in the selectivity sample: {ba_top_txt}, bachelor's-only), so a linear per-unit coefficient is not a dose-response "
             f"slope above a share of about 0.4, where only nursing and special education lie; special education "
             f"({int(fsi.loc['special_education', 'n_A3'])} cells) has a steep positive slope ({f3(fsi.loc['special_education', 'b_A3'], 2)} "
             f"raw, {f3(fsi.loc['special_education', 'b_A4'], 2)} controlled, unshrunken).")
    ns_forms = LVp[LVp.spec.isin(("A4", "A4B", "A4E", "A7")) & LVp.form.isin(ROBF) & LVp.moderator.isin(STRICT2)]
    ns_weak = ns_forms[ns_forms.p_perm >= 0.05]
    L.append(f"- **Field type and licensing are confounded across these fields.** The strict share is lowest in every "
             f"engineering, CS, statistics and business field, so across the {int(FTC_.loc[S_, 'k'])} fields it works almost "
             f"as a field-type indicator; a licensing (setting) mechanism would need licensing to vary within field types, "
             f"and within CIP-2 only {int(FTC_.loc[S_, 'k_in_strata2'])} fields in {int(FTC_.loc[S_, 'n_strata2'])} groups "
             f"carry that variation. The within-subset fits have 17-23 fields and wide intervals, so their nulls do not rule "
             f"out a moderation of the full-sample size. The bloc was defined in the verification from the data pattern "
             f"(the fields at the bottom of the share), not in advance; the README 'integrated' clusters (engineering, CS, "
             f"math and social sciences), fixed before this check, leave the moderation in place (A4 "
             f"{pk(ft('A4', S_, 'integrated'))}), and adding business to them makes it borderline (A4 "
             f"{pk(ft('A4', S_, 'integrated_bus'))}, A4B {pk(ft('A4B', S_, 'integrated_bus'))}; bachelor's-only A4 "
             f"{pk(ft('A4', B_, 'integrated_bus'))}). So the field-type reading hinges on business fields being grouped with "
             f"engineering and CS, and the README's own, earlier grouping does not remove the moderation.")
    L.append(f"- **Forms of the moderator are a choice.** All four leverage-resistant forms are reported, none was picked "
             f"after the fact. From A4 to A7 they agree in sign everywhere (g {f3(ns_forms.g.min(), 3)} to "
             f"{f3(ns_forms.g.max(), 3)} per SD) and {len(ns_forms) - len(ns_weak)} of {len(ns_forms)} have permutation p < 0.05"
             + (f"; the exceptions are " + "; ".join(f"{r.spec} {'all-degree' if r.moderator == S_ else 'bachelor' + chr(39) + 's-only'} "
                                                     f"{r.form} (p {fp_(r.p_perm)})" for r in ns_weak.itertuples())
                if len(ns_weak) else "") + ". On the log scale the forms disagree more (table b0).")
    L.append("- **Brand and institution-wide earnings controls.** Brand (academia-wide rank) is correlated with field prestige "
             "and SAT within fields, so A4B and A7 partly compete for the same variation; institution-wide earnings include the "
             "earnings of the field's own graduates, so A4E partly controls for the outcome and is a lower bound on what status "
             "adds, not a clean estimate.")
    L.append("- **Selectivity proxies are institution-level** (SAT_AVG and ADM_RATE of entrants, Pell share, state earnings); "
             "program-level student quality not captured by them stays in every 'controlled' slope (see SELECTIVITY_RESULT.md).")
    L.append("- **Scale.** Normal-score slopes measure rank agreement between status and earnings; log slopes measure the premium. "
             "The two need not move together (a field can have a tight ordering with a small spread of pay). The strict-share "
             "moderation shows on the first at every status-control level (not with the field-type control, answer 2a); on "
             "the second it shows net of selectivity, weaker with brand, not in the raw log slopes and not with the "
             "field-type control. The broad share shows the reverse pattern across scales: nothing on the first net of "
             "selectivity, a log-premium moderation net of selectivity that survives the field-type control but not brand "
             "(answer 2d). Which scale a 'setting' mechanism should show on is not settled here.")
    L.append("- **Fixed-variance shortcuts.** The permutation statistic and both bootstraps hold variance ratios fixed; the "
             "permutation test is still exact for its statistic, the bootstrap intervals ignore variance-component uncertainty.")
    L.append("- **B1/B2 true-score bounds** assume noise in rho_f independent of the moderator, normal on the rho scale, and treat "
             f"the ACS-derived moderators as error-free (moderator error would widen the bounds). The unweighted all-field "
             f"Spearman and the precision-weighted cell version have {'opposite' if np.sign(b1a.estimate) != np.sign(c1a.estimate) else 'the same'} "
             f"signs for B1 ({f3(b1a.estimate, 2)} vs {f3(c1a.estimate, 2)}) and "
             f"{'opposite' if np.sign(b2a.estimate) != np.sign(c2a.estimate) else 'the same'} signs for B2 "
             f"({f3(b2a.estimate, 2)} vs {f3(c2a.estimate, 2)}); the all-field test gives every field equal weight, including "
             "noisy ones (reliability of rho_f on all fields "
             f"{b1a.reliability:.2f} vs {b1r.reliability:.2f} on the reliable ones).")
    L.append("- **B3** uses one alternative (a SAR process on the flow graph); other alternatives could be easier to detect, but "
             f"the test's power is bounded by n = {b3['n']}. Power falls again at high lambda (panel c) because the SAR process "
             f"then loads mainly on the component common to all fields, which Moran's I removes by centring: the field-specific "
             f"share of the latent variance is {b3['cshare0']:.2f} at lambda = 0 and {b3['cshare95']:.2f} at lambda = 0.95.")
    L.append("- **B4** uses a normal approximation to the two-stage bootstrap; the 'moderate' and 'large' contrasts (half / all of "
             "the all-field slope) are conventions chosen here, as are 0.30 / 0.50 for correlations.")
    L.append("- Scorecard medians only; no tail outcomes. Bachelor's level only.\n")
    # ------------------------------------------------------------------ provenance
    L.append("## Revision notes (earlier local drafts of this file)\n")
    L.extend(R)
    L.append("## Provenance (for SOURCES.md)\n")
    L.append("No new data were downloaded for this analysis, so nothing needs adding to SOURCES.md. Inputs read (raw files as "
             "documented in data/raw/SOURCES.md, whose URLs, vintages and access dates are repeated here; interim files produced "
             "by the scripts named); size and md5 computed at run time:\n")
    L.append("| file | what | bytes | md5 |")
    L.append("|---|---|---|---|")
    for r in prov.itertuples():
        L.append(f"| `{r.file}` | {r.what} | {r.bytes:,} | `{r.md5}` |")
    OUT_MD.write_text("\n".join(L) + "\n")


def main():
    cells = build_cells()
    nf = cells.groupby("field").size()
    fields = sorted(nf[nf >= NMIN].index)
    acs = acs_shares()
    fmod = field_moderators(sorted(cells.field.unique()), acs)
    a = run_part_a(cells, fmod)
    res = run_part_b(a["A"], a["H"])
    PT = power_table(res)
    write_csv(a, PT)
    make_figure(a, res, PT)
    write_md(a, res, PT, provenance())
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(a["H"].round(4).to_string())
    print(a["A"][["spec", "moderator", "term", "k", "n", "g", "se", "p_wald", "p_perm", "b_main", "tau",
                  "tau_null"]].round(4).to_string())
    print(a["FL"].round(4).to_string())
    print(a["BT"].round(4).to_string())
    print(PT[["label", "n", "estimate", "p", "mde", "power_moderate", "ci90_lo", "ci90_hi", "bound",
              "verdict"]].round(3).to_string())
    return a, res, PT


if __name__ == "__main__":
    main()
