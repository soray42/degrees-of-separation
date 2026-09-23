"""Selectivity confound test: is field coupling rho_f just "how much institutional SELECTIVITY
pays in field f"?

Referee objection (#1): program median earnings reflect who enrolls, so Spearman(prestige,
earnings) within a field may only measure how strongly selectivity is priced in that field.
This script tests that objection with public data only (College Scorecard institution file +
Scorecard Field-of-Study + Wapman prestige). Descriptive, not causal.

Baseline (reproduced first, must equal outputs/expanded66_gap_map.csv `spearman`):
    rho_f = Spearman_i( Wapman field prestige (-Rank) , Scorecard FoS BA EARN_MDN_4YR )

Per field f (66-field universe; estimates need >= NMIN=15 institutions in the cell):
 (i)   baseline rho_f (full sample) and rho_f on each covariate-complete subsample
 (ii)  selectivity coupling Spearman(SAT_AVG, earn), Spearman(-ADM_RATE, earn) and the
       prestige-selectivity correlation Spearman(prestige, SAT_AVG / -ADM_RATE)
 (iii) PARTIAL coupling: rank-transform prestige, earnings and continuous covariates within the
       field sample, OLS-residualize both sides on the covariates, correlate residuals.
         sel         : SAT_AVG + ADM_RATE                                   (SAT sample)
         full_state  : SAT_AVG + ADM_RATE + PCTPELL + CONTROL + state FE    (SAT sample)  [strict]
         full_stlev  : SAT_AVG + ADM_RATE + PCTPELL + CONTROL + state earnings level (SAT sample) [broad]
         adm_state   : ADM_RATE + PCTPELL + CONTROL + state FE               (ADM sample)
         adm_stlev   : ADM_RATE + PCTPELL + CONTROL + state earnings level  (ADM sample)
       control-precision gradient on the SAT sample (raw -> pcs -> pcs_adm -> full_stlev -> full_ie):
         pcs         : PCTPELL + CONTROL + state earnings level              (SAT sample)
         pcs_adm     : ADM_RATE + PCTPELL + CONTROL + state earnings level   (SAT sample)
         full_ie     : full_stlev + institution-wide MD_EARN_WNE_P10         (SAT sample)
       "state earnings level" = leave-one-out mean log MD_EARN_WNE_P10 over all predominantly-
       bachelor's (PREDDEG=3) Scorecard institutions in the same state: a 1-df geography control
       for fields too small to carry ~40 state dummies. A partial estimate is reported only when
       the residual df (n - rank(Z) - 1) >= DFMIN=10.
 (iv)  horse race (SAT sample): rank(earn) ~ z rank(prestige) + z rank(SAT) + z rank(-ADM)
       + z rank(MD_EARN_WNE_P10); HC1 SEs; unique R2 and Shapley R2 over three blocks
       {prestige}, {selectivity: SAT, ADM}, {institution-wide earnings}.
 (v)   SES stratification: Spearman(prestige, EARN_PELL_WNE_MDN_4YR) and
       Spearman(prestige, EARN_NOPELL_WNE_MDN_4YR) on the common sample where all-student, Pell
       and non-Pell medians are all released. NOT a selectivity control.
 (vi)  within-institution cross-department design pooling all fields:
         log earn_{i,f} = alpha_i + gamma_f + beta_f * z(prestige)_{i,f} + e
       (a) as written; (b) + field-specific slopes on institution SAT_AVG, ADM_RATE, PCTPELL
       (so "selectivity pays differently by field" cannot load on beta_f); (c) + field-specific
       slope on the academia-wide Wapman brand rank. alpha_i absorbed by within-institution
       demeaning; SEs clustered by institution; beta_f heterogeneity by Wald test and a
       DerSimonian-Laird tau.

Sampling noise: a joint institution-level bootstrap (resample the 227 matched institutions with
replacement, recompute every field's statistics on the same draw) -> per-field SEs and CIs for
cross-field contrasts (named high-minus-low contrast, brand ADV). Heterogeneity vs noise: DL tau,
Cochran Q (treats fields as independent, which they are not -- they share institutions -- so Q
p-values are indicative only).

Ordering test (does the cross-field ranking of the partial rho_f track the baseline ranking?).
Baseline and partial rho_f are estimated from the same institutions, so their errors are positively
correlated and Spearman(baseline, partial) is positive even when the true partial coupling is the
same in every field. A bootstrap percentile CI of that Spearman keeps the shared noise inside every
draw and cannot address this. Three checks are used instead:
  (1) shared-noise null: true partial rho constant across fields; each bootstrap draw supplies the
      per-field noise pair (baseline, partial) (deviation from the field's bootstrap mean; joint
      draws keep cross-field noise dependence). Null Spearman = Spearman(truth_raw + e_raw,
      e_partial). truth_raw is either the observed baseline (variant A) or the observed baseline
      shrunk so its cross-field variance equals the moment estimate of the true variance
      (variant B, constrained-Bayes shrinkage). p = share of null draws >= observed.
  (2) split-half cross-fit on well-estimated fields (n >= SPLIT_NMIN in the spec's sample): split
      the matched institutions at random into halves A/B; baseline rho on A vs partial on B (and B
      vs A) have independent noise. Mean over SPLIT_K splits; permutation p from relabelling fields
      (same permutation in every split). Also: same-half value, split-half reliabilities and the
      disattenuated true-score correlation r(raw_A, partial_B) / sqrt(r(raw_A, raw_B) r(partial_A,
      partial_B)).
  (3) moment decomposition: cross-field sample (co)variances minus the expected (co)variances of the
      bootstrap noise vectors -> true between-field variance and true-score correlation, by field
      size.
The same issue affects Spearman(beta_f, rho_f) in the FE design (split-half cross-fit: beta_f on one
half of institutions, rho_f on the other; specs a, b, c) and Spearman(broad, strict) (shared-noise
null + a paired test of strict - broad differences).

Does an ordering that beats the shared-noise null answer the referee? That null holds the true
partial rho constant across fields. The referee's hypothesis does not imply a constant partial when
selectivity is measured imperfectly (institution-level SAT_AVG / ADM_RATE of recent entrants standing
in for the student quality of a program's graduates): the part of student quality the proxies miss
stays in the partial, and it is priced more where selectivity pays more, so the hypothesis predicts
a positive partial ordered like the baseline. Two checks:
  (4) control-precision gradient on the SAT sample, fixed field set (estimable at every step):
      raw -> + Pell/control/state level -> + ADM_RATE -> + SAT_AVG (= broad) -> + institution-wide
      earnings. Per step: mean, Spearman vs baseline (+ shared-noise null); on the n_SAT >= 60 fields,
      the split-half true-score correlation of the step's partial with the baseline and with
      selectivity pricing rho(SAT_AVG, earn).
  (5) split-half cross-field regression on the same random halvings as the broad cross-fit in (2):
      across fields, z(rank outcome on half B) ~ z(rank baseline on half A) + z(rank pricing on half A),
      and B/A swapped; coefficients averaged over the 2*SPLIT_K replicates. Pricing = rho(SAT_AVG, earn)
      or rho(-ADM_RATE, earn) on the same half. One-sided Freedman-Lane permutation p per coefficient
      (field labels, same permutation in every replicate). Calibration row: outcome = the baseline
      itself, where the true weights are (1, 0) by construction. Errors-in-variables version: true-score
      correlations from cross-half moments -> standardized true-score regression (descriptive; no p).

Brand vs field (FIELD_VS_GENERIC_RESULT.md): c_F = Spearman(field prestige, earn),
c_G = Spearman(-academia-wide Wapman rank, earn); reproduced raw, then recomputed net of the
same covariates.

Reproducible: SEED fixed; outputs byte-identical on re-run.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/55_selectivity.py
Outputs: data/interim/selectivity_fields.csv, outputs/figures/selectivity_coupling.png,
         SELECTIVITY_RESULT.md (root; gitignored).
"""
from __future__ import annotations

import sys
import importlib.util
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, chi2
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

SEED = 55
NMIN = 15            # min institutions in a (field, sample) cell
DFMIN = 10           # min residual df for a partial correlation
B_BOOT = 1000        # joint institution bootstrap draws
B_GAP = 50           # bootstrap draws inside compute_gap (only spearman/n are used here)
SPLIT_NMIN = 60      # "well-estimated" field: n >= 60 institutions in the spec's sample
SPLIT_K = 300        # random half-splits of institutions (partial-coupling cross-fit)
SPLIT_FE_K = 200     # random half-splits for the FE-design cross-fit
N_PERM = 2000        # field-label permutations for the cross-fit tests
ORDER_SPECS = ["sel", "full_stlev", "full_state", "adm_stlev", "adm_state"]
NO_NULL = {"rho_raw", "rho_raw_sat", "rho_raw_adm", "rho_all_common", "cG_raw", "cG_raw_sat"}
INST_FILE = ROOT / "data" / "raw" / "scorecard_inst" / "Most-Recent-Cohorts-Institution.csv"
GAP_MAP = ROOT / "outputs" / "expanded66_gap_map.csv"
OUT_CSV = ROOT / "data" / "interim" / "selectivity_fields.csv"
OUT_FIG = ROOT / "outputs" / "figures" / "selectivity_coupling.png"
OUT_MD = ROOT / "SELECTIVITY_RESULT.md"
HIGH = ["computer_science", "statistics", "economics"]
LOW = ["nursing", "communication_disorders"]

# FIELDS66 + label map, exactly as scripts/52 builds them
_spec = importlib.util.spec_from_file_location("s28", ROOT / "scripts" / "28_field_vs_generic_prestige.py")
_s28 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s28)
FIELDS66, LAB = _s28.FIELDS66, _s28.LAB

# partial-correlation specs: name -> (sample family, continuous covariates, categorical covariates)
SPECS = {
    "sel":        ("sat", ["SAT_AVG", "ADM_RATE"], []),
    "full_state": ("sat", ["SAT_AVG", "ADM_RATE", "PCTPELL"], ["CONTROL", "STABBR"]),
    "full_stlev": ("sat", ["SAT_AVG", "ADM_RATE", "PCTPELL", "ST_EARN"], ["CONTROL"]),
    "adm_state":  ("adm", ["ADM_RATE", "PCTPELL"], ["CONTROL", "STABBR"]),
    "adm_stlev":  ("adm", ["ADM_RATE", "PCTPELL", "ST_EARN"], ["CONTROL"]),
    # control-precision gradient (SAT sample)
    "pcs":        ("sat", ["PCTPELL", "ST_EARN"], ["CONTROL"]),
    "pcs_adm":    ("sat", ["ADM_RATE", "PCTPELL", "ST_EARN"], ["CONTROL"]),
    "full_ie":    ("sat", ["SAT_AVG", "ADM_RATE", "PCTPELL", "ST_EARN", "MD_EARN_WNE_P10"], ["CONTROL"]),
}
SPEC_DESC = {
    "sel": "SAT_AVG + ADM_RATE (SAT sample)",
    "full_state": "SAT_AVG + ADM_RATE + PCTPELL + CONTROL + state FE (SAT sample)",
    "full_stlev": "SAT_AVG + ADM_RATE + PCTPELL + CONTROL + state earnings level (SAT sample)",
    "adm_state": "ADM_RATE + PCTPELL + CONTROL + state FE (ADM sample)",
    "adm_stlev": "ADM_RATE + PCTPELL + CONTROL + state earnings level (ADM sample)",
    "pcs": "PCTPELL + CONTROL + state earnings level (SAT sample)",
    "pcs_adm": "ADM_RATE + PCTPELL + CONTROL + state earnings level (SAT sample)",
    "full_ie": "SAT_AVG + ADM_RATE + PCTPELL + CONTROL + state earnings level + MD_EARN_WNE_P10 (SAT sample)",
}
# control-precision gradient: steps on the SAT sample, from no controls to the broad spec + inst. earnings
GRAD = ["rho_raw_sat", "rho_pcs", "rho_pcs_adm", "rho_full_stlev", "rho_full_ie"]
GRAD_LAB = {"rho_raw_sat": "raw (no controls)", "rho_pcs": "+ Pell share, control, state level",
            "rho_pcs_adm": "+ ADM_RATE", "rho_full_stlev": "+ SAT_AVG (= broad spec)",
            "rho_full_ie": "+ institution-wide earnings"}
# half-sample partials kept for the split-half cross-field regression
SR_SPECS = ["pcs", "pcs_adm", "full_stlev", "full_ie", "sel", "adm_stlev"]
# (outcome, pricing measure, field family, label) rows of the split-half cross-field regression
REG_ROWS = [("raw", "sat_earn", "sat", "baseline ρ_f itself (calibration: true weights 1, 0)"),
            ("raw_sat", "sat_earn", "sat", "raw ρ_f, SAT sample (gradient step 0)"),
            ("pcs", "sat_earn", "sat", "partial given Pell, control, state level (step 1)"),
            ("pcs_adm", "sat_earn", "sat", "partial given + ADM_RATE (step 2)"),
            ("full_stlev", "sat_earn", "sat", "partial given + SAT_AVG = broad spec (step 3)"),
            ("full_ie", "sat_earn", "sat", "partial given + institution-wide earnings (step 4)"),
            ("sel", "sat_earn", "sat", "partial given SAT_AVG + ADM_RATE only"),
            ("full_stlev", "adm_earn", "sat", "broad partial, ADM pricing"),
            ("adm_stlev", "sat_earn", "adm", "partial given ADM + Pell, control, state level (ADM sample)"),
            ("adm_stlev", "adm_earn", "adm", "partial given ADM + Pell, control, state level (ADM sample), ADM pricing")]
PRICE_LAB = {"sat_earn": "ρ(SAT_AVG, earn)", "adm_earn": "ρ(−ADM_RATE, earn)"}
FAMILY_NEED = {
    "sat": ["SAT_AVG", "ADM_RATE", "PCTPELL", "ST_EARN", "CONTROL", "STABBR"],
    "adm": ["ADM_RATE", "PCTPELL", "ST_EARN", "CONTROL", "STABBR"],
}


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def load_institutions() -> pd.DataFrame:
    cols = ["UNITID", "INSTNM", "STABBR", "CONTROL", "PREDDEG", "ADM_RATE", "SAT_AVG",
            "SATMTMID", "SATVRMID", "ACTCMMID", "PCTPELL", "MD_EARN_WNE_P10"]
    d = pd.read_csv(INST_FILE, usecols=cols, dtype=str)
    for c in ["ADM_RATE", "SAT_AVG", "SATMTMID", "SATVRMID", "ACTCMMID", "PCTPELL",
              "MD_EARN_WNE_P10", "CONTROL", "PREDDEG"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")         # 'NULL' / 'PrivacySuppressed' -> NaN
    # leave-one-out state earnings level over predominantly-bachelor's institutions
    d["_le"] = np.where((d.PREDDEG == 3) & (d.MD_EARN_WNE_P10 > 0), np.log(d.MD_EARN_WNE_P10), np.nan)
    d["_ok"] = d["_le"].notna().astype(float)
    s = d.groupby("STABBR")["_le"].transform(lambda x: x.fillna(0).sum())
    k = d.groupby("STABBR")["_ok"].transform("sum")
    own = d["_le"].fillna(0.0)
    denom = k - d["_ok"]
    d["ST_EARN"] = np.where(denom > 0, (s - own) / denom.where(denom > 0, 1), np.nan)
    return d.drop(columns=["_le", "_ok"])


def build_cells(ar, er, inst):
    """One row per (field, institution): prestige, earnings, institution covariates, brand, Pell."""
    m = ar[["inst_key", "field", "prestige_score"]].merge(
        er[["inst_key", "field", "earnings", "cohort_n", "institution_id", "institution_name"]],
        on=["inst_key", "field"], how="inner").dropna(subset=["prestige_score", "earnings"])
    m = m.rename(columns={"institution_id": "UNITID"})
    m["UNITID"] = m["UNITID"].astype(str)
    m = m.merge(inst, on="UNITID", how="left")
    g = _s28.load_generic()                                # inst_key, g_rank (academia-wide)
    m = m.merge(g, on="inst_key", how="left")
    m["G"] = -m["g_rank"]
    for tag, (ec, cc) in {"pell": ("EARN_PELL_WNE_MDN_4YR", "EARN_COUNT_PELL_WNE_4YR"),
                          "nopell": ("EARN_NOPELL_WNE_MDN_4YR", "EARN_COUNT_NOPELL_WNE_4YR")}.items():
        e = load_er_scorecard("undergrad", earn_col=ec, count_col=cc, fields=FIELDS66)
        m = m.merge(e[["inst_key", "field", "earnings"]].rename(columns={"earnings": f"earn_{tag}"}),
                    on=["inst_key", "field"], how="left")
    m["NEG_ADM"] = -m["ADM_RATE"]
    return m.sort_values(["field", "inst_key"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------
def spear(x, y):
    if len(x) < 4:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def partial_rank(x, y, conts, cats):
    """Rank-residualized partial correlation of x,y given covariates. Returns (r, df)."""
    n = len(x)
    cols = [np.ones((n, 1))] + [rankdata(c)[:, None] for c in conts]
    for c in cats:
        _, inv = np.unique(c, return_inverse=True)
        L = inv.max() + 1
        if L > 1:
            cols.append(np.eye(L)[inv][:, 1:])
    Z = np.hstack(cols)
    Yv = np.column_stack([rankdata(x), rankdata(y)])
    coef, _, rank, _ = np.linalg.lstsq(Z, Yv, rcond=None)
    df = n - rank - 1
    if df < DFMIN:
        return np.nan, df
    E = Yv - Z @ coef
    ex, ey = E[:, 0], E[:, 1]
    den = np.sqrt((ex @ ex) * (ey @ ey))
    if den <= 1e-9:
        return np.nan, df
    return float(ex @ ey / den), df


def field_arrays(d: pd.DataFrame, inst_index: dict) -> dict:
    fd = {"n": len(d), "inst_idx": d.inst_key.map(inst_index).to_numpy(int),
          "P": d.prestige_score.to_numpy(float), "Y": d.earnings.to_numpy(float),
          "G": d.G.to_numpy(float),
          "PELL_E": d.earn_pell.to_numpy(float), "NOPELL_E": d.earn_nopell.to_numpy(float)}
    for c in ["SAT_AVG", "ADM_RATE", "NEG_ADM", "PCTPELL", "ST_EARN", "MD_EARN_WNE_P10"]:
        fd[c] = d[c].to_numpy(float)
    fd["CONTROL"] = d.CONTROL.fillna(-1).to_numpy(int)
    fd["STABBR"] = d.STABBR.fillna("").to_numpy(str)
    for fam, need in FAMILY_NEED.items():
        fd["ok_" + fam] = d[need].notna().all(axis=1).to_numpy() & (d.STABBR.fillna("") != "").to_numpy()
    fd["ok_pell"] = (d.earn_pell.notna() & d.earn_nopell.notna()).to_numpy()
    fd["ok_g"] = d.G.notna().to_numpy()
    # which samples clear NMIN at the point estimate (fixed for the bootstrap)
    fd["use"] = {k: int(fd["ok_" + k].sum()) >= NMIN for k in ("sat", "adm", "pell")}
    fd["use"]["raw"] = fd["n"] >= NMIN
    return fd


def field_stats(fd: dict, rows: np.ndarray) -> dict:
    """All per-field correlation statistics on a (possibly resampled) row set."""
    out = {}
    if fd["use"]["raw"]:
        out["rho_raw"] = spear(fd["P"][rows], fd["Y"][rows])
        okg = rows[fd["ok_g"][rows]]
        out["cG_raw"] = spear(fd["G"][okg], fd["Y"][okg])
    for fam in ("sat", "adm"):
        if not fd["use"][fam]:
            continue
        r = rows[fd["ok_" + fam][rows]]
        P, Y = fd["P"][r], fd["Y"][r]
        out[f"rho_raw_{fam}"] = spear(P, Y)
        for name, (sfam, conts, cats) in SPECS.items():
            if sfam != fam:
                continue
            out[f"rho_{name}"], out[f"df_{name}"] = partial_rank(
                P, Y, [fd[c][r] for c in conts], [fd[c][r] for c in cats])
        if fam == "sat":
            out["rho_sat_earn"] = spear(fd["SAT_AVG"][r], Y)
            out["rho_prest_sat"] = spear(P, fd["SAT_AVG"][r])
            # brand, raw and net of the same covariates (same SAT sample)
            out["cG_raw_sat"] = spear(fd["G"][r], Y)
            for name in ("full_stlev", "full_state"):
                _, conts, cats = SPECS[name]
                out[f"cG_{name}"], _ = partial_rank(fd["G"][r], Y, [fd[c][r] for c in conts],
                                                    [fd[c][r] for c in cats])
        else:
            out["rho_adm_earn"] = spear(fd["NEG_ADM"][r], Y)
            out["rho_prest_adm"] = spear(P, fd["NEG_ADM"][r])
            out["rho_ie_earn"] = spear(fd["MD_EARN_WNE_P10"][r], Y)
            out["rho_prest_ie"] = spear(P, fd["MD_EARN_WNE_P10"][r])
    if fd["use"]["pell"]:
        r = rows[fd["ok_pell"][rows]]
        out["rho_all_common"] = spear(fd["P"][r], fd["Y"][r])
        out["rho_pell"] = spear(fd["P"][r], fd["PELL_E"][r])
        out["rho_nopell"] = spear(fd["P"][r], fd["NOPELL_E"][r])
    return out


def horse_race(fd: dict) -> dict:
    """SAT sample: rank(earn) ~ z(prestige) + z(SAT) + z(-ADM) + z(inst-wide earnings)."""
    if not fd["use"]["sat"]:
        return {}
    r = np.where(fd["ok_sat"] & np.isfinite(fd["MD_EARN_WNE_P10"]))[0]
    if len(r) < NMIN:
        return {}
    z = lambda v: (rankdata(v) - rankdata(v).mean()) / rankdata(v).std()
    y = z(fd["Y"][r])
    X = {"prest": z(fd["P"][r]), "sat": z(fd["SAT_AVG"][r]), "adm": z(fd["NEG_ADM"][r]),
         "ie": z(fd["MD_EARN_WNE_P10"][r])}
    blocks = {"prest": ["prest"], "sel": ["sat", "adm"], "ie": ["ie"]}
    names = ["prest", "sat", "adm", "ie"]
    Xf = sm.add_constant(np.column_stack([X[k] for k in names]))
    fit = sm.OLS(y, Xf).fit(cov_type="HC1")
    out = {"hr_n": len(r), "hr_r2": float(fit.rsquared)}
    for j, k in enumerate(names):
        out[f"hr_b_{k}"] = float(fit.params[j + 1]); out[f"hr_se_{k}"] = float(fit.bse[j + 1])

    def r2(bl):
        if not bl:
            return 0.0
        cols = [X[c] for b in bl for c in blocks[b]]
        return float(sm.OLS(y, sm.add_constant(np.column_stack(cols))).fit().rsquared)

    B = list(blocks)
    cache = {tuple(sorted(s)): r2(list(s)) for k in range(4) for s in combinations(B, k)}
    full = cache[tuple(sorted(B))]
    for b in B:
        others = [o for o in B if o != b]
        out[f"hr_uniq_{b}"] = full - cache[tuple(sorted(others))]
        shap = 0.0
        for k in range(len(others) + 1):
            for s in combinations(others, k):
                wgt = factorial(k) * factorial(len(B) - k - 1) / factorial(len(B))
                shap += wgt * (cache[tuple(sorted(s + (b,)))] - cache[tuple(sorted(s))])
        out[f"hr_shap_{b}"] = shap
    out["hr_leader"] = max(B, key=lambda b: out[f"hr_shap_{b}"])
    return out


# ---------------------------------------------------------------------------------------------
# ordering tests: shared-noise null, split-half cross-fit, moment decomposition
# ---------------------------------------------------------------------------------------------
def _joint(X, Y):
    m = np.isfinite(X) & np.isfinite(Y)
    return np.where(m, X, np.nan), np.where(m, Y, np.nan), m


def rowwise_spear(X, Y):
    """Spearman per row of two (K, k) arrays on jointly finite entries (average ranks; NaN if <5)."""
    X, Y, m = _joint(np.atleast_2d(X), np.atleast_2d(Y))
    rx = pd.DataFrame(X).rank(axis=1).to_numpy(); ry = pd.DataFrame(Y).rank(axis=1).to_numpy()
    return rowwise_pearson(rx, ry)


def rowwise_pearson(X, Y):
    X, Y, m = _joint(np.atleast_2d(X), np.atleast_2d(Y))
    n = m.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dx = X - np.nanmean(X, 1, keepdims=True); dy = Y - np.nanmean(Y, 1, keepdims=True)
        num = np.nansum(dx * dy, 1); den = np.sqrt(np.nansum(dx ** 2, 1) * np.nansum(dy ** 2, 1))
        return np.where((n >= 5) & (den > 0), num / np.where(den > 0, den, 1.0), np.nan)


def rowwise_cov(X, Y):
    X, Y, m = _joint(np.atleast_2d(X), np.atleast_2d(Y))
    n = m.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dx = X - np.nanmean(X, 1, keepdims=True); dy = Y - np.nanmean(Y, 1, keepdims=True)
        return np.where(n >= 3, np.nansum(dx * dy, 1) / (n - 1), np.nan)


def noise_null(boot, base, idx, xc, yc, fl, keep=False):
    """Spearman(x, y) across fields vs a null with constant true y and the joint-bootstrap noise
    pairs (x, y) of each field; plus the moment decomposition of the cross-field (co)variances."""
    fl = [f for f in fl if np.isfinite(base.loc[f, xc]) and np.isfinite(base.loc[f, yc])]
    k = len(fl)
    if k < 5:
        return dict(k=k)
    jj = [idx[f] for f in fl]
    X, Y = base.loc[fl, xc].to_numpy(float), base.loc[fl, yc].to_numpy(float)
    bx, by = boot[xc][:, jj], boot[yc][:, jj]
    ex, ey = bx - np.nanmean(bx, 0), by - np.nanmean(by, 0)
    ex, ey, jm = _joint(ex, ey)
    ncorr = np.array([np.corrcoef(ex[jm[:, j], j], ey[jm[:, j], j])[0, 1] for j in range(k)])
    obs = float(spearmanr(X, Y)[0])
    # moments: observed cross-field (co)variances and expected ones of the pure-noise vectors
    Sxy, Sxx, Syy = float(np.cov(X, Y)[0, 1]), float(np.var(X, ddof=1)), float(np.var(Y, ddof=1))
    Nxy = float(np.nanmean(rowwise_cov(ex, ey)))
    Nxx = float(np.nanmean(rowwise_cov(ex, ex))); Nyy = float(np.nanmean(rowwise_cov(ey, ey)))
    tvx, tvy = Sxx - Nxx, Syy - Nyy
    r_true = (Sxy - Nxy) / np.sqrt(tvx * tvy) if (tvx > 0 and tvy > 0) else np.nan
    se_x = np.nanstd(bx, 0, ddof=1)
    v = max(tvx, 0.0)
    truth = {"A": X, "B": X.mean() + (X - X.mean()) * np.sqrt(v / (v + se_x ** 2))}
    out = dict(k=k, obs=obs, ncorr_min=float(ncorr.min()), ncorr_med=float(np.median(ncorr)),
               ncorr_max=float(ncorr.max()), Sxy=Sxy, Nxy=Nxy, share=Nxy / Sxy if Sxy != 0 else np.nan,
               Sxx=Sxx, Nxx=Nxx, Syy=Syy, Nyy=Nyy, tvx=tvx, tvy=tvy, r_true=float(r_true))
    # variant C: where the bootstrap noise variance of y exceeds y's observed cross-field variance
    # (impossible under the null, where the observed variance is all noise), shrink the part of y's
    # noise that is unrelated to x's noise until the null's expected cross-field variance equals the
    # observed one. Shrinking the unrelated part raises the noise correlation -> a stricter null.
    bj = np.array([np.cov(ex[jm[:, j], j], ey[jm[:, j], j])[0, 1] / np.var(ex[jm[:, j], j], ddof=1)
                   for j in range(k)])
    Aa, Uu = bj[None, :] * ex, ey - bj[None, :] * ex
    lam = 1.0
    if Nyy > Syy:
        qa = float(np.nanmean(rowwise_cov(Uu, Uu))); qb = 2 * float(np.nanmean(rowwise_cov(Aa, Uu)))
        qc = float(np.nanmean(rowwise_cov(Aa, Aa))) - Syy
        disc = qb ** 2 - 4 * qa * qc
        lam = float(np.clip((-qb + np.sqrt(disc)) / (2 * qa), 0.0, 1.0)) if (qa > 0 and disc >= 0) else 0.0
    out["lamC"] = lam
    for t, tx, ey_ in (("A", truth["A"], ey), ("B", truth["B"], ey), ("C", truth["B"], Aa + lam * Uu)):
        nul = rowwise_spear(tx[None, :] + ex, ey_)
        nul = nul[np.isfinite(nul)]
        out[f"null{t}_mean"] = float(nul.mean()); out[f"null{t}_q95"] = float(np.percentile(nul, 95))
        out[f"null{t}_p"] = float((1 + np.sum(nul >= obs)) / (1 + len(nul)))
        if keep:
            out[f"null{t}"] = nul
    return out


def crossfit_perm(XA, XB, YA, YB, rng):
    """Mean over splits of [Spearman(X_A, Y_B) + Spearman(X_B, Y_A)]/2 and its permutation p
    (field labels of Y permuted, same permutation in every split)."""
    cross = (rowwise_spear(XA, YB) + rowwise_spear(XB, YA)) / 2
    obs = float(np.nanmean(cross))
    k, cnt = XA.shape[1], 0
    for _ in range(N_PERM):
        pi = rng.permutation(k)
        v = np.nanmean((rowwise_spear(XA, YB[:, pi]) + rowwise_spear(XB, YA[:, pi])) / 2)
        cnt += v >= obs
    return cross, float((1 + cnt) / (1 + N_PERM))


def split_half(FD, fields, N, spec, rng, point):
    """Split-half cross-fit of baseline rho vs a partial spec on fields with n_fam >= SPLIT_NMIN."""
    fam, conts, cats = SPECS[spec]
    fl = [f for f in fields if np.isfinite(point[f].get(f"rho_{spec}", np.nan))
          and int(FD[f]["ok_" + fam].sum()) >= SPLIT_NMIN]
    k = len(fl)
    RA, RB, QA, QB = (np.full((SPLIT_K, k), np.nan) for _ in range(4))

    def half(fd, rows):
        raw = spear(fd["P"][rows], fd["Y"][rows]) if len(rows) >= NMIN else np.nan
        r = rows[fd["ok_" + fam][rows]]
        if len(r) < NMIN:
            return raw, np.nan
        return raw, partial_rank(fd["P"][r], fd["Y"][r], [fd[c][r] for c in conts],
                                 [fd[c][r] for c in cats])[0]

    for s in range(SPLIT_K):
        inA = np.zeros(N, bool); inA[rng.permutation(N)[: N // 2]] = True
        for j, f in enumerate(fl):
            fd = FD[f]; a = inA[fd["inst_idx"]]
            RA[s, j], QA[s, j] = half(fd, np.where(a)[0])
            RB[s, j], QB[s, j] = half(fd, np.where(~a)[0])
    cross, p_perm = crossfit_perm(RA, RB, QA, QB, rng)
    same = (rowwise_spear(RA, QA) + rowwise_spear(RB, QB)) / 2
    crossP = (rowwise_pearson(RA, QB) + rowwise_pearson(RB, QA)) / 2
    relR, relQ = rowwise_pearson(RA, RB), rowwise_pearson(QA, QB)
    mR, mQ, mC = np.nanmean(relR), np.nanmean(relQ), np.nanmean(crossP)
    ratio = np.where((relR > 0) & (relQ > 0), crossP / np.sqrt(np.clip(relR * relQ, 1e-12, None)), np.nan)
    return dict(spec=spec, fam=fam, k=k, fields=fl, cross=cross, same=same, p_perm=p_perm,
                cross_mean=float(np.nanmean(cross)), cross_min=float(np.nanmin(cross)),
                cross_q05=float(np.nanpercentile(cross, 5)), cross_le0=int(np.sum(cross <= 0)), K=SPLIT_K,
                same_mean=float(np.nanmean(same)), relR=float(mR), relQ=float(mQ),
                relQ_s=float(np.nanmean(rowwise_spear(QA, QB))), crossP=float(mC),
                dis=float(mC / np.sqrt(mR * mQ)) if (mR > 0 and mQ > 0) else np.nan,
                dis_med=float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else np.nan,
                undef=float(np.mean(~np.isfinite(QA)) / 2 + np.mean(~np.isfinite(QB)) / 2))


def split_panel(FD, fields, N, rng):
    """Half-sample statistics for the split-half cross-field regression, on fields with n_SAT or n_ADM
    >= SPLIT_NMIN. Called with default_rng([SEED, 101]) it draws the same halvings, in the same order,
    as split_half(..., 'full_stlev', ...), so the two analyses use identical splits."""
    fl = [f for f in fields if max(int(FD[f]["ok_sat"].sum()), int(FD[f]["ok_adm"].sum())) >= SPLIT_NMIN]
    names = ["raw", "raw_sat", "sat_earn", "adm_earn"] + SR_SPECS
    H = {n: np.full((SPLIT_K, 2, len(fl)), np.nan) for n in names}
    for s in range(SPLIT_K):
        inA = np.zeros(N, bool); inA[rng.permutation(N)[: N // 2]] = True
        for j, f in enumerate(fl):
            fd = FD[f]; a = inA[fd["inst_idx"]]
            for h, rows in enumerate((np.where(a)[0], np.where(~a)[0])):
                if len(rows) >= NMIN:
                    H["raw"][s, h, j] = spear(fd["P"][rows], fd["Y"][rows])
                for fam in ("sat", "adm"):
                    r = rows[fd["ok_" + fam][rows]]
                    if len(r) < NMIN:
                        continue
                    if fam == "sat":
                        H["raw_sat"][s, h, j] = spear(fd["P"][r], fd["Y"][r])
                        H["sat_earn"][s, h, j] = spear(fd["SAT_AVG"][r], fd["Y"][r])
                    else:
                        H["adm_earn"][s, h, j] = spear(fd["NEG_ADM"][r], fd["Y"][r])
                    for sp in SR_SPECS:
                        sf, conts, cats = SPECS[sp]
                        if sf == fam:
                            H[sp][s, h, j] = partial_rank(fd["P"][r], fd["Y"][r], [fd[c][r] for c in conts],
                                                          [fd[c][r] for c in cats])[0]
    return fl, H


def _zrows(M, rank):
    if rank:
        M = np.apply_along_axis(rankdata, 1, M)
    return (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)


def _stack(Mh, sel, pred):
    """(K, 2, k) half arrays -> (2K, k) replicates: predictors from half A then B, outcome from B then A."""
    a, b = Mh[:, 0][:, sel], Mh[:, 1][:, sel]
    return np.concatenate([a, b]) if pred else np.concatenate([b, a])


def cf_regress(Yh, X1h, X2h, sel, rng, rank=True):
    """Across fields, per split and direction: z(outcome on one half) ~ z(X1 on the other half) +
    z(X2 on that other half). Mean standardized coefficients over the replicates; one-sided
    Freedman-Lane permutation p per coefficient (reduced model = the other predictor; its residuals
    permuted over fields with the same permutation in every replicate)."""
    Y, X1, X2 = _stack(Yh, sel, False), _stack(X1h, sel, True), _stack(X2h, sel, True)
    ok = np.isfinite(Y).all(1) & np.isfinite(X1).all(1) & np.isfinite(X2).all(1)
    Y, X1, X2 = _zrows(Y[ok], rank), _zrows(X1[ok], rank), _zrows(X2[ok], rank)
    R, k = Y.shape
    perms = np.array([rng.permutation(k) for _ in range(N_PERM)])
    b = np.zeros((R, 2)); bnull = np.zeros((2, N_PERM))
    for r in range(R):
        X = np.column_stack([X1[r], X2[r]])
        Hm = np.linalg.solve(X.T @ X, X.T)
        b[r] = Hm @ Y[r]
        for jf, other in ((0, X2[r]), (1, X1[r])):
            fit = (other @ Y[r] / (other @ other)) * other
            bnull[jf] += ((fit[None, :] + (Y[r] - fit)[perms]) @ Hm.T)[:, jf]
    bo, bnull = b.mean(0), bnull / R
    p = [float((1 + np.sum(bnull[j] >= bo[j])) / (1 + N_PERM)) for j in (0, 1)]
    return dict(k=int(k), R=int(R), R_drop=int((~ok).sum()), b_base=float(bo[0]), b_price=float(bo[1]),
                p_base=p[0], p_price=p[1], q95_base=float(np.percentile(bnull[0], 95)),
                q95_price=float(np.percentile(bnull[1], 95)))


def eiv_decomp(Yh, X1h, X2h, sel):
    """Errors-in-variables version of cf_regress on the Pearson scale: split-half reliabilities and
    cross-half correlations (independent noise) -> true-score correlations -> standardized true-score
    regression of the outcome on (X1, X2). Mean over splits of each moment, then the ratios."""
    g = lambda M, h: M[:, h][:, sel]
    xc = lambda A, B: float(np.nanmean(rowwise_pearson(A, B)))
    rel = lambda M: xc(g(M, 0), g(M, 1))
    cr = lambda M1, M2: (xc(g(M1, 0), g(M2, 1)) + xc(g(M1, 1), g(M2, 0))) / 2
    r11, r22, ryy = rel(X1h), rel(X2h), rel(Yh)
    ok = min(r11, r22, ryy) > 0
    t12 = cr(X1h, X2h) / np.sqrt(r11 * r22) if ok else np.nan
    t1y = cr(X1h, Yh) / np.sqrt(r11 * ryy) if ok else np.nan
    t2y = cr(X2h, Yh) / np.sqrt(r22 * ryy) if ok else np.nan
    bt = (np.linalg.solve(np.array([[1.0, t12], [t12, 1.0]]), np.array([t1y, t2y]))
          if ok and abs(t12) < 1 else np.array([np.nan, np.nan]))
    return dict(rel_base=r11, rel_price=r22, rel_y=ryy, t12=float(t12), t1y=float(t1y), t2y=float(t2y),
                eb_base=float(bt[0]), eb_price=float(bt[1]),
                cf_base=float(np.nanmean((rowwise_spear(g(X1h, 0), g(Yh, 1)) + rowwise_spear(g(X1h, 1), g(Yh, 0))) / 2)),
                cf_price=float(np.nanmean((rowwise_spear(g(X2h, 0), g(Yh, 1)) + rowwise_spear(g(X2h, 1), g(Yh, 0))) / 2)))


def dl_tau(y, se):
    """DerSimonian-Laird between-field SD, Cochran Q, p, I2 (fields treated as independent)."""
    ok = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y, v = y[ok], se[ok] ** 2
    k = len(y)
    if k < 3:
        return dict(k_dl=k, tau=np.nan, Q=np.nan, p=np.nan, I2=np.nan)
    w = 1 / v
    mu = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - mu) ** 2))
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / c)
    return dict(k_dl=k, tau=float(np.sqrt(tau2)), Q=Q, p=float(chi2.sf(Q, k - 1)),
                I2=float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0)


# ---------------------------------------------------------------------------------------------
# (vi) within-institution cross-department design
# ---------------------------------------------------------------------------------------------
def fe_design(cells: pd.DataFrame, extra: list[str], absorb: bool = True, restrict=None) -> dict:
    d = cells.dropna(subset=["earnings", "prestige_score"] + extra).copy()
    d = d[d.earnings > 0]
    if restrict is not None:
        d = d[d.inst_key.isin(restrict)]
    while True:                                          # singletons carry no within-inst info
        n0 = len(d)
        ic = d.groupby("inst_key").field.transform("size")
        d = d[ic >= 2]
        fc = d.groupby("field").inst_key.transform("size")
        d = d[fc >= NMIN]
        if len(d) == n0:
            break
    d = d.sort_values(["field", "inst_key"]).reset_index(drop=True)
    fields = sorted(d.field.unique())
    d["y"] = np.log(d.earnings)
    d["z"] = d.groupby("field").prestige_score.transform(lambda s: (s - s.mean()) / s.std())
    cols, names = [], []
    for j, f in enumerate(fields):
        D = (d.field == f).to_numpy(float)
        if j > 0 or not absorb:
            cols.append(D); names.append(f"g_{f}")
        cols.append(D * d.z.to_numpy()); names.append(f"b_{f}")
        for c in extra:
            zc = d.groupby("field")[c].transform(lambda s: (s - s.mean()) / s.std()).fillna(0).to_numpy()
            cols.append(D * zc); names.append(f"x_{c}_{f}")
    X = pd.DataFrame(np.column_stack(cols), columns=names)
    y = d.y.copy()
    if absorb:
        X = X - X.groupby(d.inst_key.to_numpy()).transform("mean")
        y = y - y.groupby(d.inst_key.to_numpy()).transform("mean")
    groups = pd.factorize(d.inst_key)[0]
    fit = sm.OLS(y.to_numpy(), X.to_numpy()).fit(cov_type="cluster", cov_kwds={"groups": groups})
    V = np.asarray(fit.cov_params())
    bidx = [names.index(f"b_{f}") for f in fields]
    beta = np.asarray(fit.params)[bidx]; se = np.sqrt(np.diag(V)[bidx])
    # Wald: all beta_f equal
    k = len(bidx)
    R = np.zeros((k - 1, len(names)))
    for r_, j in enumerate(bidx[1:]):
        R[r_, bidx[0]] = -1; R[r_, j] = 1
    rb = R @ np.asarray(fit.params); RV = R @ V @ R.T
    rk = np.linalg.matrix_rank(RV)
    W = float(rb @ np.linalg.pinv(RV) @ rb)
    # named contrast mean(HIGH) - mean(LOW)
    hi = [f for f in HIGH if f in fields]; lo = [f for f in LOW if f in fields]
    c = np.zeros(len(names))
    for f in hi:
        c[names.index(f"b_{f}")] += 1 / len(hi)
    for f in lo:
        c[names.index(f"b_{f}")] -= 1 / len(lo)
    con = float(c @ np.asarray(fit.params)); con_se = float(np.sqrt(c @ V @ c))
    # pooled common slope (same sample): y ~ alpha_i + gamma_f + beta z
    Xp = [(d.field == f).to_numpy(float) for f in fields[(1 if absorb else 0):]] + [d.z.to_numpy()]
    Xp = pd.DataFrame(np.column_stack(Xp))
    if absorb:
        Xp = Xp - Xp.groupby(d.inst_key.to_numpy()).transform("mean")
    fp = sm.OLS(y.to_numpy(), Xp.to_numpy()).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return dict(fields=fields, beta=pd.Series(beta, index=fields), se=pd.Series(se, index=fields),
                n_cells=len(d), n_inst=d.inst_key.nunique(), n_par=len(names),
                wald=W, wald_df=int(rk), wald_p=float(chi2.sf(W, rk)),
                contrast=con, contrast_se=con_se, hi=hi, lo=lo,
                pooled=float(fp.params[-1]), pooled_se=float(fp.bse[-1]),
                n_by_field=d.groupby("field").size())


def fe_beta_fast(cells: pd.DataFrame, extra: list[str], restrict) -> pd.Series:
    """beta_f of fe_design (same sample rules, institution FE absorbed), point estimates only."""
    d = cells.dropna(subset=["earnings", "prestige_score"] + extra)
    d = d[(d.earnings > 0) & d.inst_key.isin(restrict)]
    while True:
        n0 = len(d)
        d = d[d.groupby("inst_key").field.transform("size") >= 2]
        d = d[d.groupby("field").inst_key.transform("size") >= NMIN]
        if len(d) == n0:
            break
    if d.empty:
        return pd.Series(dtype=float)
    d = d.sort_values(["field", "inst_key"]).reset_index(drop=True)
    fields = sorted(d.field.unique())
    g = d.groupby("field")
    z = g.prestige_score.transform(lambda s: (s - s.mean()) / s.std()).to_numpy()
    D = np.eye(len(fields))[pd.Categorical(d.field, categories=fields).codes]
    cols = [D[:, 1:], D * z[:, None]]
    for c in extra:
        zc = g[c].transform(lambda s: (s - s.mean()) / s.std()).fillna(0).to_numpy()
        cols.append(D * zc[:, None])
    X, y = np.hstack(cols), np.log(d.earnings.to_numpy())
    ic = pd.factorize(d.inst_key)[0]
    cnt = np.bincount(ic).astype(float)

    def dm(M):
        s = np.zeros((cnt.size,) + M.shape[1:]); np.add.at(s, ic, M)
        return M - (s / (cnt[:, None] if M.ndim == 2 else cnt))[ic]

    coef = np.linalg.lstsq(dm(X), dm(y), rcond=None)[0]
    k = len(fields)
    return pd.Series(coef[k - 1: 2 * k - 1], index=fields)


def fe_split_half(cells, FD, universe, fe_fields, extra, rng):
    """Cross-fit Spearman(beta_f on half A, baseline rho_f on half B) and vice versa."""
    N, k = len(universe), len(fe_fields)
    uni = np.array(universe)
    BA, BB, RA, RB = (np.full((SPLIT_FE_K, k), np.nan) for _ in range(4))
    for s in range(SPLIT_FE_K):
        inA = np.zeros(N, bool); inA[rng.permutation(N)[: N // 2]] = True
        for tag, mask, Bm, Rm in (("A", inA, BA, RA), ("B", ~inA, BB, RB)):
            bt = fe_beta_fast(cells, extra, set(uni[mask]))
            for j, f in enumerate(fe_fields):
                Bm[s, j] = bt.get(f, np.nan)
                fd = FD[f]; rows = np.where(mask[fd["inst_idx"]])[0]
                Rm[s, j] = spear(fd["P"][rows], fd["Y"][rows]) if len(rows) >= NMIN else np.nan
    cross, p_perm = crossfit_perm(BA, BB, RA, RB, rng)
    same = (rowwise_spear(BA, RA) + rowwise_spear(BB, RB)) / 2
    kk = (np.sum(np.isfinite(BA) & np.isfinite(RB), 1) + np.sum(np.isfinite(BB) & np.isfinite(RA), 1)) / 2
    return dict(k_all=k, k_mean=float(kk.mean()), cross=cross, same=same, p_perm=p_perm,
                cross_mean=float(np.nanmean(cross)), cross_min=float(np.nanmin(cross)),
                cross_q05=float(np.nanpercentile(cross, 5)), cross_le0=int(np.sum(cross <= 0)), K=SPLIT_FE_K,
                same_mean=float(np.nanmean(same)))


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    ar = load_ar_wapman(fields=FIELDS66)
    er = load_er_scorecard("undergrad", fields=FIELDS66)

    # --- reproduce the paper's baseline ---
    gm = compute_gap_map(ar, er, "undergrad", B=B_GAP, seed=SEED)
    ref = pd.read_csv(GAP_MAP)
    chk = ref[["field", "n_institutions", "spearman", "reliable"]].merge(
        gm[["field", "n_institutions", "spearman"]], on="field", how="left", suffixes=("", "_re"))
    repro_maxdiff = float((chk.spearman - chk.spearman_re).abs().max())
    repro_n_ok = bool((chk.n_institutions.fillna(0) == chk.n_institutions_re.fillna(0)).all())
    assert repro_maxdiff < 1e-9 and repro_n_ok, "baseline did not reproduce"
    print(f"[baseline] reproduced {chk.spearman.notna().sum()} fields; max|diff|={repro_maxdiff:.1e}")

    inst = load_institutions()
    cells = build_cells(ar, er, inst)
    assert cells.INSTNM.notna().all(), "UNITID join failed for some cells"
    universe = sorted(cells.inst_key.unique())
    inst_index = {k: i for i, k in enumerate(universe)}
    N = len(universe)

    # coverage (institution level)
    u = cells.drop_duplicates("inst_key")
    cov = {c: float(u[c].notna().mean()) for c in ["SAT_AVG", "ADM_RATE", "PCTPELL", "ST_EARN",
                                                    "MD_EARN_WNE_P10", "CONTROL", "STABBR"]}
    cov["sat_family"] = float(u[FAMILY_NEED["sat"]].notna().all(axis=1).mean())
    cov["adm_family"] = float(u[FAMILY_NEED["adm"]].notna().all(axis=1).mean())
    u = u.assign(gq=pd.qcut(u.g_rank.rank(method="first"), 4, labels=["top", "2nd", "3rd", "bottom"]))
    cov_q = u.groupby("gq", observed=True).agg(n=("inst_key", "size"),
                                               sat=("SAT_AVG", lambda s: s.notna().mean()),
                                               adm=("ADM_RATE", lambda s: s.notna().mean()))
    miss_sat_ca = int(((u.SAT_AVG.isna()) & (u.STABBR == "CA")).sum())
    miss_sat = int(u.SAT_AVG.isna().sum())

    fields = sorted(cells.field.unique())
    FD = {f: field_arrays(cells[cells.field == f], inst_index) for f in fields}

    # point estimates
    point = {f: field_stats(FD[f], np.arange(FD[f]["n"])) for f in fields}
    hr = {f: horse_race(FD[f]) for f in fields}
    stat_keys = sorted({k for f in fields for k in point[f] if not k.startswith("df_")})

    # joint institution bootstrap
    boot = {k: np.full((B_BOOT, len(fields)), np.nan) for k in stat_keys}
    p = np.full(N, 1.0 / N)
    for b in range(B_BOOT):
        cnt = rng.multinomial(N, p)
        for j, f in enumerate(fields):
            fd = FD[f]
            rows = np.repeat(np.arange(fd["n"]), cnt[fd["inst_idx"]])
            if len(rows) < 4:
                continue
            for k, v in field_stats(fd, rows).items():
                if k in boot:
                    boot[k][b, j] = v
    se = {k: np.nanstd(boot[k], axis=0, ddof=1) for k in stat_keys}
    bfail = {k: np.mean(~np.isfinite(boot[k]), axis=0) for k in stat_keys}

    # ---- per-field table ----
    rel = ref.set_index("field")["reliable"].to_dict()
    rows = []
    for j, f in enumerate(fields):
        fd = FD[f]
        r = dict(field=f, label=LAB.get(f, f), reliable=bool(rel.get(f, False)),
                 licensed=f in F.LICENSED_FIELDS, n=fd["n"],
                 n_sat=int(fd["ok_sat"].sum()), n_adm=int(fd["ok_adm"].sum()),
                 n_pell_common=int(fd["ok_pell"].sum()),
                 n_states_sat=int(len(np.unique(fd["STABBR"][fd["ok_sat"]]))))
        for k in stat_keys:
            r[k] = point[f].get(k, np.nan)
            r[f"se_{k}"] = se[k][j] if np.isfinite(r[k]) else np.nan
        for name in SPECS:
            r[f"df_{name}"] = point[f].get(f"df_{name}", np.nan)
            r[f"bootfail_{name}"] = bfail[f"rho_{name}"][j] if np.isfinite(r[f"rho_{name}"]) else np.nan
            # within-field bootstrap correlation of the (baseline, partial) estimation errors
            bx, by = boot["rho_raw"][:, j], boot[f"rho_{name}"][:, j]
            ok = np.isfinite(bx) & np.isfinite(by)
            r[f"noisecorr_{name}"] = (float(np.corrcoef(bx[ok], by[ok])[0, 1])
                                      if np.isfinite(r[f"rho_{name}"]) and ok.sum() > 10 else np.nan)
        r.update(hr[f])
        rows.append(r)
    T = pd.DataFrame(rows)
    T["rho_baseline_ref"] = T.field.map(ref.set_index("field")["spearman"])

    # ---- (vi) within-institution FE design ----
    FE = {}
    FE["a"] = fe_design(cells, [])
    FE["a_ols"] = fe_design(cells, [], absorb=False)
    sat_insts = set(cells.loc[cells.SAT_AVG.notna() & cells.ADM_RATE.notna() & cells.PCTPELL.notna(), "inst_key"])
    FE["b"] = fe_design(cells, ["SAT_AVG", "ADM_RATE", "PCTPELL"])
    FE["a_bs"] = fe_design(cells, [], restrict=sat_insts)        # (a) on (b)'s institution set
    FE["c"] = fe_design(cells, ["SAT_AVG", "ADM_RATE", "PCTPELL", "G"])
    for tag in ["a", "a_ols", "b", "a_bs", "c"]:
        T[f"fe_{tag}_beta"] = T.field.map(FE[tag]["beta"])
        T[f"fe_{tag}_se"] = T.field.map(FE[tag]["se"])
        T[f"fe_{tag}_ncells"] = T.field.map(FE[tag]["n_by_field"])

    # ---- cross-field summaries ----
    idx = {f: j for j, f in enumerate(fields)}
    base = T.set_index("field")

    def fset(col, reliable_only=False):
        s = base[np.isfinite(base[col])]
        if reliable_only:
            s = s[s.reliable]
        return list(s.index)

    def contrast(col, fl=None):
        hi = [f for f in HIGH if np.isfinite(base.loc[f, col])]
        lo = [f for f in LOW if np.isfinite(base.loc[f, col])]
        if not hi or not lo:
            return dict(hi=hi, lo=lo, est=np.nan, lo95=np.nan, hi95=np.nan)
        est = base.loc[hi, col].mean() - base.loc[lo, col].mean()
        bk = col if col in boot else None
        if bk is None:
            return dict(hi=hi, lo=lo, est=est, lo95=np.nan, hi95=np.nan)
        bh = np.nanmean(boot[bk][:, [idx[f] for f in hi]], axis=1)
        bl = np.nanmean(boot[bk][:, [idx[f] for f in lo]], axis=1)
        d = bh - bl
        d = d[np.isfinite(d)]
        return dict(hi=hi, lo=lo, est=est, lo95=np.percentile(d, 2.5), hi95=np.percentile(d, 97.5))

    summ = {}
    stat_list = ["rho_raw", "rho_raw_sat", "rho_raw_adm", "rho_sel", "rho_full_state",
                 "rho_full_stlev", "rho_adm_state", "rho_adm_stlev", "rho_pcs", "rho_pcs_adm", "rho_full_ie",
                 "rho_sat_earn", "rho_adm_earn", "rho_ie_earn",
                 "rho_all_common", "rho_pell", "rho_nopell", "cG_raw", "cG_raw_sat",
                 "cG_full_stlev", "cG_full_state"]
    for col in stat_list:
        for relonly in (False, True):
            fl = fset(col, relonly)
            y = base.loc[fl, col].to_numpy(float); s_ = base.loc[fl, f"se_{col}"].to_numpy(float)
            h = dl_tau(y, s_)
            same = {"rho_sel": "rho_raw_sat", "rho_full_state": "rho_raw_sat",
                    "rho_full_stlev": "rho_raw_sat", "rho_adm_state": "rho_raw_adm",
                    "rho_adm_stlev": "rho_raw_adm", "cG_full_stlev": "cG_raw_sat", "rho_pcs": "rho_raw_sat",
                    "rho_pcs_adm": "rho_raw_sat", "rho_full_ie": "rho_raw_sat",
                    "cG_full_state": "cG_raw_sat", "rho_pell": "rho_all_common",
                    "rho_nopell": "rho_all_common", "rho_sat_earn": "rho_raw_sat",
                    "rho_adm_earn": "rho_raw_adm", "rho_ie_earn": "rho_raw_adm"}.get(col, "rho_raw")
            s_full = spearmanr(base.loc[fl, "rho_raw"], y)[0] if len(fl) >= 5 else np.nan
            s_same = spearmanr(base.loc[fl, same], y)[0] if len(fl) >= 5 else np.nan
            # shared-noise null for the cross-field Spearman (not for the raw-vs-raw reproductions)
            nn_full = (noise_null(boot, base, idx, "rho_raw", col, fl) if col not in NO_NULL else {})
            nn_same = (noise_null(boot, base, idx, same, col, fl) if col not in NO_NULL and col != same else {})
            # analytic noise model on the Fisher-z scale: var = 1.06/(df-1), df = n-2 for a raw
            # Spearman and n-rank(Z)-1 for a partial (Fieller et al. 1957 factor for rank data)
            dfc = f"df_{col[4:]}" if col[4:] in SPECS else None
            ncol = {"rho_raw": "n", "rho_raw_sat": "n_sat", "rho_raw_adm": "n_adm", "rho_sat_earn": "n_sat",
                    "rho_adm_earn": "n_adm", "rho_ie_earn": "n_adm", "rho_all_common": "n_pell_common",
                    "rho_pell": "n_pell_common", "rho_nopell": "n_pell_common", "cG_raw": "n",
                    "cG_raw_sat": "n_sat"}.get(col)
            if dfc:
                dfv = base.loc[fl, dfc].to_numpy(float)
            elif ncol:
                dfv = base.loc[fl, ncol].to_numpy(float) - 2
            else:
                dfv = np.full(len(fl), np.nan)
            zse = np.sqrt(1.06 / np.clip(dfv - 1, 1, None))
            hz = dl_tau(np.arctanh(np.clip(y, -0.999, 0.999)), zse)
            summ[(col, relonly)] = dict(
                tau_z=hz["tau"], p_z=hz["p"],
                n_pos=int(np.sum(y - 1.96 * s_ > 0)), n_neg=int(np.sum(y + 1.96 * s_ < 0)),
                k=len(fl), mean=float(np.mean(y)) if len(y) else np.nan,
                sd=float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
                rms_se=float(np.sqrt(np.nanmean(s_ ** 2))) if len(s_) else np.nan,
                mean_same=float(base.loc[fl, same].mean()) if len(fl) else np.nan, same=same,
                s_full=s_full, nn_full=nn_full, s_same=s_same, nn_same=nn_same, **h)
    contr = {col: contrast(col) for col in ["rho_raw", "rho_raw_sat", "rho_raw_adm", "rho_sel",
                                            "rho_full_state", "rho_full_stlev", "rho_adm_state",
                                            "rho_adm_stlev", "rho_pell", "rho_nopell", "rho_all_common"]}

    # fixed field set estimable in every spec: {CS, economics} vs {nursing}
    FIX_HI, FIX_LO = ["computer_science", "economics"], ["nursing"]
    contr_fix = {}
    for col in contr:
        est = base.loc[FIX_HI, col].mean() - base.loc[FIX_LO, col].mean()
        d = (np.nanmean(boot[col][:, [idx[f] for f in FIX_HI]], axis=1)
             - np.nanmean(boot[col][:, [idx[f] for f in FIX_LO]], axis=1))
        d = d[np.isfinite(d)]
        contr_fix[col] = dict(est=est, lo95=np.percentile(d, 2.5), hi95=np.percentile(d, 97.5))

    # same field set: broad vs strict partial on the fields where strict is estimable
    ff = fset("rho_full_state")
    same_set = {}
    for col in ("rho_raw_sat", "rho_full_stlev", "rho_full_state"):
        y = base.loc[ff, col].to_numpy(float); s_ = base.loc[ff, f"se_{col}"].to_numpy(float)
        same_set[col] = dict(k=len(ff), mean=float(y.mean()), sd=float(y.std(ddof=1)),
                             rms_se=float(np.sqrt(np.mean(s_ ** 2))), **dl_tau(y, s_))
    same_set["k"] = len(ff)
    same_set["s_broad_strict"] = spearmanr(base.loc[ff, "rho_full_stlev"], base.loc[ff, "rho_full_state"])[0]
    same_set["bootfail_strict"] = float(base.loc[ff, "bootfail_full_state"].mean())
    # Spearman(broad, strict) shares estimation noise too: test it against the shared-noise null
    same_set["nn_broad_strict"] = noise_null(boot, base, idx, "rho_full_stlev", "rho_full_state", ff)
    # paired test: do strict and broad partials differ across fields by more than noise?
    jj = [idx[f] for f in ff]
    dd = boot["rho_full_state"][:, jj] - boot["rho_full_stlev"][:, jj]
    d_obs = (base.loc[ff, "rho_full_state"] - base.loc[ff, "rho_full_stlev"]).to_numpy(float)
    d_se = np.nanstd(dd, axis=0, ddof=1)
    same_set["diff"] = dict(mean=float(d_obs.mean()), sd=float(d_obs.std(ddof=1)),
                            rms_se=float(np.sqrt(np.mean(d_se ** 2))), **dl_tau(d_obs, d_se))

    # ---- ordering tests: shared-noise null, moment decomposition, split-half cross-fit ----
    order = {}
    for spec in ORDER_SPECS:
        col = f"rho_{spec}"
        ncol_ = "n_" + SPECS[spec][0]
        fl = fset(col)
        sets = {"all": fl, "large": [f for f in fl if base.loc[f, ncol_] >= SPLIT_NMIN],
                "small": [f for f in fl if base.loc[f, ncol_] < SPLIT_NMIN],
                "reliable": [f for f in fl if base.loc[f, "reliable"]]}
        for sname, sl in sets.items():
            order[(spec, sname)] = noise_null(boot, base, idx, "rho_raw", col, sl,
                                              keep=(spec == "full_stlev" and sname == "all"))
    split = {}
    for i, spec in enumerate(ORDER_SPECS):
        split[spec] = split_half(FD, fields, N, spec, np.random.default_rng([SEED, 100 + i]), point)
        print(f"[split] {spec}: k={split[spec]['k']} cross={split[spec]['cross_mean']:+.3f} "
              f"p_perm={split[spec]['p_perm']:.4f} undefined={split[spec]['undef']:.1%}")
    # ---- does the surviving ordering answer the referee? split-half cross-field regression ----
    sp_fl, H = split_panel(FD, fields, N, np.random.default_rng([SEED, 100 + ORDER_SPECS.index("full_stlev")]))
    n_fam = {"sat": np.array([int(FD[f]["ok_sat"].sum()) for f in sp_fl]),
             "adm": np.array([int(FD[f]["ok_adm"].sum()) for f in sp_fl])}
    reg = {}
    for i, (out, pr, fam, lab) in enumerate(REG_ROWS):
        col = {"raw": "rho_raw", "raw_sat": "rho_raw_sat"}.get(out, f"rho_{out}")
        sel_ = (n_fam[fam] >= SPLIT_NMIN) & np.array([np.isfinite(base.loc[f, col]) for f in sp_fl])
        reg[(out, pr)] = dict(lab=lab, fam=fam, fields=[f for f, s_ in zip(sp_fl, sel_) if s_],
                              rank=cf_regress(H[out], H["raw"], H[pr], sel_, np.random.default_rng([SEED, 500 + i])),
                              val=cf_regress(H[out], H["raw"], H[pr], sel_, np.random.default_rng([SEED, 600 + i]),
                                             rank=False),
                              **eiv_decomp(H[out], H["raw"], H[pr], sel_))
        q_ = reg[(out, pr)]
        print(f"[reg] {out:10s} {pr:8s} k={q_['rank']['k']} b_base={q_['rank']['b_base']:+.3f} "
              f"(p={q_['rank']['p_base']:.4f}) b_price={q_['rank']['b_price']:+.3f} (p={q_['rank']['p_price']:.4f}) "
              f"t12={q_['t12']:+.3f} eiv={q_['eb_base']:+.2f}/{q_['eb_price']:+.2f}")
    # consistency: same splits and fields as the broad split-half cross-fit above
    q_, e_ = split["full_stlev"], reg[("full_stlev", "sat_earn")]
    assert e_["fields"] == q_["fields"], "split-half field sets differ"
    assert abs(e_["cf_base"] - q_["cross_mean"]) < 1e-9 and abs(e_["t1y"] - q_["dis"]) < 1e-9, "splits differ"
    # control-precision gradient on a fixed SAT-sample field set
    grad_fl = [f for f in fields if all(np.isfinite(base.loc[f, c]) for c in GRAD)]
    grad = {}
    for c in GRAD:
        y = base.loc[grad_fl, c].to_numpy(float)
        grad[c] = dict(k=len(grad_fl), mean=float(y.mean()),
                       s_base=float(spearmanr(base.loc[grad_fl, "rho_raw"], y)[0]),
                       nn=(noise_null(boot, base, idx, "rho_raw", c, grad_fl) if c != "rho_raw_sat" else {}),
                       reg=reg[(c[4:], "sat_earn")])

    fe_split = {}
    for i, (tag, extra) in enumerate((("a", []), ("b", ["SAT_AVG", "ADM_RATE", "PCTPELL"]),
                                      ("c", ["SAT_AVG", "ADM_RATE", "PCTPELL", "G"]))):
        fe_split[tag] = fe_split_half(cells, FD, universe, FE[tag]["fields"], extra,
                                      np.random.default_rng([SEED, 200 + i]))
        print(f"[fe split] {tag}: k~{fe_split[tag]['k_mean']:.1f} cross={fe_split[tag]['cross_mean']:+.3f} "
              f"p_perm={fe_split[tag]['p_perm']:.4f}")

    # brand vs field
    brand = {}
    # reproduce FIELD_VS_GENERIC on n>=8 fields (point estimates computed here without NMIN gate)
    rep = []
    for f in fields:
        fd = FD[f]
        if fd["n"] >= 8:
            okg = fd["ok_g"]
            rep.append((f, spear(fd["P"][okg], fd["Y"][okg]), spear(fd["G"][okg], fd["Y"][okg])))
    rep = pd.DataFrame(rep, columns=["field", "cF", "cG"])
    brand["repro"] = dict(k=len(rep), cF=rep.cF.mean(), cG=rep.cG.mean(), neg=int((rep.cF < rep.cG).sum()))
    for tag, fcol, gcol in [("raw_sat", "rho_raw_sat", "cG_raw_sat"),
                            ("full_stlev", "rho_full_stlev", "cG_full_stlev"),
                            ("full_state", "rho_full_state", "cG_full_state")]:
        s = base[np.isfinite(base[fcol]) & np.isfinite(base[gcol])]
        adv = s[fcol] - s[gcol]
        jj = [idx[f] for f in s.index]
        bd = np.nanmean(boot[fcol][:, jj] - boot[gcol][:, jj], axis=1)
        brand[tag] = dict(k=len(s), cF=s[fcol].mean(), cG=s[gcol].mean(), adv=adv.mean(),
                          neg=int((adv < 0).sum()), adv_ci=np.percentile(bd[np.isfinite(bd)], [2.5, 97.5]),
                          k_rel=int(s.reliable.sum()), neg_rel=int((adv[s.reliable] < 0).sum()))

    # FE heterogeneity summaries
    fe_summ = {}
    for tag in ["a", "a_ols", "b", "a_bs", "c"]:
        bt, st = FE[tag]["beta"], FE[tag]["se"]
        h = dl_tau(bt.to_numpy(), st.to_numpy())
        common = [f for f in bt.index if np.isfinite(base.loc[f, "rho_raw"])]
        rel_f = [f for f in common if base.loc[f, "reliable"]]
        fe_summ[tag] = dict(k=len(bt), sd=float(bt.std(ddof=1)), rms_se=float(np.sqrt(np.mean(st ** 2))),
                            mean=float(bt.mean()), s_raw=spearmanr(bt[common], base.loc[common, "rho_raw"])[0],
                            s_raw_rel=spearmanr(bt[rel_f], base.loc[rel_f, "rho_raw"])[0] if len(rel_f) >= 5 else np.nan,
                            k_rel=len(rel_f), n_pos_sig=int(((bt - 1.96 * st) > 0).sum()),
                            n_neg_sig=int(((bt + 1.96 * st) < 0).sum()), **h)

    # ---- write table ----
    T = T.sort_values("field").reset_index(drop=True)
    front = ["field", "label", "reliable", "licensed", "n", "rho_baseline_ref", "rho_raw", "se_rho_raw",
             "n_sat", "rho_raw_sat", "rho_sel", "rho_full_state", "df_full_state", "rho_full_stlev",
             "rho_pcs", "rho_pcs_adm", "rho_full_ie", "n_adm", "rho_raw_adm", "rho_adm_state", "df_adm_state", "rho_adm_stlev",
             "rho_sat_earn", "rho_adm_earn", "rho_ie_earn", "rho_prest_sat", "rho_prest_adm", "rho_prest_ie"]
    T = T[front + [c for c in T.columns if c not in front]]
    T.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"[table] {OUT_CSV}  ({len(T)} fields)")

    ctx = dict(T=T, base=base, summ=summ, contr=contr, contr_fix=contr_fix, same_set=same_set,
               brand=brand, FE=FE, fe_summ=fe_summ, order=order, split=split, fe_split=fe_split,
               cov=cov, cov_q=cov_q, miss_sat=miss_sat, miss_sat_ca=miss_sat_ca, N=N,
               repro_maxdiff=repro_maxdiff, n_cells=len(cells), hr=hr, reg=reg, grad=grad)
    _figure(ctx)
    _write_md(ctx)
    _print(ctx)


# ---------------------------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------------------------
NAMED = HIGH + LOW + ["computer_engineering"]
OFFS = {"computer_engineering": (4, -11), "communication_disorders": (-10, -12)}


def _figure(ctx):
    base, FE = ctx["base"], ctx["FE"]
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 4, figsize=(20, 14.6))
    C_REL, C_OTH, C_NAMED = "#1f5f99", "#9aa5b1", "#c2410c"

    def scatter(ax, xcol, ycol, xlab, ylab, title, xerr=None, yerr=None):
        s = base[np.isfinite(base[xcol]) & np.isfinite(base[ycol])]
        for relv, col, lab in ((False, C_OTH, "not reliable"), (True, C_REL, "reliable (gap map)")):
            q = s[s.reliable == relv]
            ax.errorbar(q[xcol], q[ycol], yerr=q[yerr] if yerr else None, xerr=q[xerr] if xerr else None,
                        fmt="o", ms=4.2, color=col, ecolor=col, elinewidth=0.6, alpha=0.85, label=f"{lab} (k={len(q)})")
        for f in NAMED:
            if f in s.index:
                ax.annotate(LAB.get(f, f), (s.loc[f, xcol], s.loc[f, ycol]), fontsize=7.5, color=C_NAMED,
                            xytext=OFFS.get(f, (4, 3)), textcoords="offset points")
        lo = min(s[xcol].min(), s[ycol].min(), -0.2) - 0.05
        hi = max(s[xcol].max(), s[ycol].max(), 0.8) + 0.05
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.6, ls="--")
        ax.axhline(0, color="k", lw=0.4, ls=":"); ax.axvline(0, color="k", lw=0.4, ls=":")
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title, fontsize=9.5, loc="left")
        ax.legend(fontsize=7, loc="upper left", frameon=False)

    sm_ = ctx["summ"]
    od = ctx["order"]
    o1, o2 = od[("full_stlev", "all")], od[("full_state", "all")]
    scatter(axes[0, 0], "rho_raw", "rho_full_stlev", "baseline ρ_f (full sample)",
            "partial ρ_f | SAT, ADM, Pell, control, state earnings level",
            f"(a) Coupling net of selectivity [broad spec]\nSpearman across fields {o1['obs']:+.2f}; "
            f"shared-noise null mean {o1['nullB_mean']:+.2f}, p={o1['nullB_p']:.3f}",
            yerr="se_rho_full_stlev")
    scatter(axes[0, 1], "rho_raw", "rho_full_state", "baseline ρ_f (full sample)",
            "partial ρ_f | SAT, ADM, Pell, control, state FE",
            f"(b) Net of selectivity [strict spec, state FE; df≥{DFMIN}]\nSpearman across fields "
            f"{o2['obs']:+.2f}; shared-noise null mean {o2['nullB_mean']:+.2f}, p={o2['nullB_p']:.3f}",
            yerr="se_rho_full_state")
    scatter(axes[0, 2], "rho_raw_sat", "rho_sat_earn", "prestige coupling ρ(prestige, earn) (SAT sample)",
            "selectivity coupling ρ(SAT_AVG, earn)",
            "(c) Selectivity vs prestige as predictors of field earnings")

    # (d) ordering test, broad spec: shared-noise null vs observed; split-half cross-fit
    ax = axes[0, 3]
    sp_ = ctx["split"]["full_stlev"]
    bins = np.linspace(-0.4, 1.0, 57)
    ax.hist(o1["nullB"], bins=bins, color="#9aa5b1", alpha=0.75, density=True,
            label=f"shared-noise null, {o1['k']} fields (constant true partial)")
    ax.hist(sp_["cross"], bins=bins, color="#1f5f99", alpha=0.75, density=True,
            label=f"cross-fit: raw on half A vs partial on half B ({sp_['k']} fields n_SAT≥{SPLIT_NMIN})")
    ax.hist(sp_["same"], bins=bins, histtype="step", color="#c2410c", lw=1.2, density=True,
            label="same half: raw and partial on the same half (shares noise)")
    ax.axvline(o1["obs"], color="k", lw=1.0, label=f"observed, {o1['k']} fields: {o1['obs']:+.2f}")
    ax.axvline(0, color="k", lw=0.4, ls=":")
    ax.set_xlabel("Spearman across fields: baseline ρ_f vs broad partial ρ_f")
    ax.set_ylabel("density")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.set_title("(d) Does the ordering survive beyond shared noise? [broad spec]", fontsize=9.5, loc="left")
    ax.legend(fontsize=6.5, loc="upper left", frameon=False)

    # (e) horse race Shapley shares for reliable fields
    ax = axes[1, 0]
    h = base[base.reliable & np.isfinite(base.get("hr_r2", np.nan))].sort_values("rho_raw")
    yy = np.arange(len(h))
    left = np.zeros(len(h))
    for b, col, lab in (("prest", "#1f5f99", "field prestige"), ("sel", "#c2410c", "selectivity (SAT, ADM)"),
                        ("ie", "#6b7280", "institution-wide earnings")):
        v = h[f"hr_shap_{b}"].to_numpy()
        ax.barh(yy, v, left=left, color=col, label=lab, height=0.72)
        left += v
    ax.set_yticks(yy); ax.set_yticklabels([LAB.get(f, f) for f in h.index], fontsize=7)
    ax.set_xlabel("Shapley share of R² (rank earnings; SAT sample)")
    ax.set_title("(e) Horse race, reliable fields (sorted by baseline ρ_f)", fontsize=9.5, loc="left")
    ax.legend(fontsize=7, loc="lower right", frameon=False)

    # (f) within-institution beta_f vs baseline rho
    ax = axes[1, 1]
    for tag, col, mk, lab in (("a", "#1f5f99", "o", "(a) α_i + γ_f + β_f z"),
                              ("b", "#c2410c", "s", "(b) + field-specific SAT/ADM/Pell slopes")):
        bt, st = FE[tag]["beta"], FE[tag]["se"]
        x = base.loc[bt.index, "rho_raw"]
        ax.errorbar(x, bt, yerr=1.96 * st, fmt=mk, ms=3.8, color=col, ecolor=col, elinewidth=0.5,
                    alpha=0.8, label=f"{lab} (k={len(bt)})")
    for f in NAMED:
        if f in FE["a"]["beta"].index:
            ax.annotate(LAB.get(f, f), (base.loc[f, "rho_raw"], FE["a"]["beta"][f]), fontsize=7.5,
                        color=C_NAMED, xytext=OFFS.get(f, (4, 3)), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("baseline ρ_f (full sample)")
    ax.set_ylabel("β_f: Δ log earnings per SD field prestige (95% CI, clustered)")
    ax.set_title("(f) Within-institution cross-department design", fontsize=9.5, loc="left")
    ax.legend(fontsize=7, loc="upper left", frameon=False)

    # (g) Pell vs non-Pell coupling
    ax = axes[1, 2]
    s = base[np.isfinite(base.rho_pell) & np.isfinite(base.rho_nopell)]
    for relv, col, lab in ((False, C_OTH, "not reliable"), (True, C_REL, "reliable")):
        q = s[s.reliable == relv]
        ax.scatter(q.rho_nopell, q.rho_pell, s=18, color=col, label=f"{lab} (k={len(q)})")
    for f in NAMED:
        if f in s.index:
            ax.annotate(LAB.get(f, f), (s.loc[f, "rho_nopell"], s.loc[f, "rho_pell"]), fontsize=7.5,
                        color=C_NAMED, xytext=OFFS.get(f, (4, 3)), textcoords="offset points")
    lo, hi = -0.5, 1.0
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.6, ls="--")
    ax.axhline(0, color="k", lw=0.4, ls=":"); ax.axvline(0, color="k", lw=0.4, ls=":")
    ax.set_xlabel("ρ(prestige, non-Pell median earnings)"); ax.set_ylabel("ρ(prestige, Pell median earnings)")
    ax.set_title("(g) SES stratification (common sample; not a selectivity control)", fontsize=9.5, loc="left")
    ax.legend(fontsize=7, loc="upper left", frameon=False)

    # (h) FE design: Spearman(beta_f, rho_f) with independent halves vs the same sample
    ax = axes[1, 3]
    fs_, fsum = ctx["fe_split"], ctx["fe_summ"]
    bins = np.linspace(-0.4, 1.0, 57)
    for tag, col, lab in (("a", "#1f5f99", "(a) α_i + γ_f + β_f z"),
                          ("b", "#c2410c", "(b) + field-specific SAT/ADM/Pell slopes"),
                          ("c", "#6b7280", "(c) (b) + field-specific brand slope")):
        ax.hist(fs_[tag]["cross"], bins=bins, color=col, alpha=0.45, density=True,
                label=f"{lab}: cross-fit (β on half A, ρ on half B)")
        ax.axvline(fsum[tag]["s_raw"], color=col, lw=1.2, ls="--",
                   label=f"{lab}: full sample, shared noise ({fsum[tag]['s_raw']:+.2f})")
    ax.axvline(0, color="k", lw=0.4, ls=":")
    ax.set_xlabel("Spearman across fields: β_f vs baseline ρ_f")
    ax.set_ylabel("density")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.7)
    ax.set_title(f"(h) FE design ordering, {SPLIT_FE_K} random half-splits", fontsize=9.5, loc="left")
    ax.legend(fontsize=6.0, loc="upper left", frameon=False)

    # ---- row 3: does the surviving ordering answer the referee? ----
    grad, reg = ctx["grad"], ctx["reg"]
    xs = np.arange(len(GRAD))
    xl = ["0\nraw", "1\n+ Pell,\ncontrol,\nstate lvl", "2\n+ ADM", "3\n+ SAT\n(= broad)", "4\n+ inst.\nearnings"]
    # (i) control-precision gradient: level and ordering
    ax = axes[2, 0]
    kf, kl = grad[GRAD[0]]["k"], grad[GRAD[0]]["reg"]["rank"]["k"]
    ax.plot(xs, [grad[c]["mean"] for c in GRAD], "o-", color="#1f5f99", label=f"mean partial ρ_f ({kf} fields)")
    ax.plot(xs, [grad[c]["s_base"] for c in GRAD], "s-", color="#c2410c",
            label=f"Spearman(baseline, partial), {kf} fields")
    nm = [grad[c]["nn"].get("nullB_mean", np.nan) for c in GRAD]
    ax.plot(xs, nm, "s--", color="#c2410c", mfc="none", label="  shared-noise null mean (constant true partial)")
    ax.plot(xs, [grad[c]["reg"]["t1y"] for c in GRAD], "^-", color="#111827",
            label=f"true-score r(partial, baseline), {kl} fields n_SAT≥{SPLIT_NMIN}")
    ax.plot(xs, [grad[c]["reg"]["t2y"] for c in GRAD], "v-", color="#6b7280",
            label=f"true-score r(partial, selectivity pricing ρ(SAT,earn)), {kl} fields")
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels(xl, fontsize=7)
    ax.set_ylim(-0.1, 1.35)
    ax.set_title("(i) Control-precision gradient\n(SAT sample; controls added left to right)", fontsize=9.5, loc="left")
    ax.legend(fontsize=6.5, loc="upper right", frameon=False)
    # (j) split-half cross-field regression coefficients along the gradient (incl. calibration)
    rows_j = [("raw", "sat_earn")] + [(c[4:], "sat_earn") for c in GRAD]
    xj = np.arange(len(rows_j))
    xlj = ["calib.\nbaseline\nitself"] + xl
    ax = axes[2, 1]
    for key, col, mk, lab in (("base", "#111827", "o", "baseline ρ_f (half A)"),
                              ("price", "#c2410c", "s", "selectivity pricing ρ(SAT, earn) (half A)")):
        v = np.array([reg[r]["rank"][f"b_{key}"] for r in rows_j])
        pv = np.array([reg[r]["rank"][f"p_{key}"] for r in rows_j])
        ax.plot(xj, v, "-", color=col, lw=1.0)
        ax.scatter(xj[pv < 0.05], v[pv < 0.05], marker=mk, color=col, s=34, zorder=3,
                   label=f"{lab}; filled: Freedman–Lane p<0.05")
        ax.scatter(xj[pv >= 0.05], v[pv >= 0.05], marker=mk, facecolors="white", edgecolors=col, s=34, zorder=3)
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set_xticks(xj); ax.set_xticklabels(xlj, fontsize=7)
    ax.set_ylabel("standardized coefficient (ranks across fields)")
    ax.set_ylim(-0.15, 0.85)
    ax.set_title(f"(j) Split-half regression, {kl} fields: outcome (half B)\n"
                 "on baseline ρ_f and pricing (half A), ranks", fontsize=9.5, loc="left")
    ax.legend(fontsize=6.5, loc="upper right", frameon=False)
    # (k) errors-in-variables version
    ax = axes[2, 2]
    for key, col, mk, lab in (("base", "#111827", "o", "baseline ρ_f"),
                              ("price", "#c2410c", "s", "selectivity pricing ρ(SAT, earn)")):
        ax.plot(xj, [reg[r][f"eb_{key}"] for r in rows_j], mk + "-", color=col, label=lab)
    t12 = reg[("full_stlev", "sat_earn")]["t12"]
    vmax = max(max(reg[r]["eb_base"], reg[r]["eb_price"]) for r in rows_j)
    ax.set_ylim(top=vmax + 0.45)
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set_xticks(xj); ax.set_xticklabels(xlj, fontsize=7)
    ax.set_ylabel("standardized true-score coefficient")
    ax.legend(fontsize=6.5, loc="upper center", frameon=False)
    ax.set_title(f"(k) Same, corrected for unreliability (EIV)\npredictors' true-score r = {t12:+.2f}: "
                 "unstable, descriptive", fontsize=9.5, loc="left")
    # (l) pricing-measure dependence
    ax = axes[2, 3]
    rows_l = [("full_stlev", "sat_earn"), ("full_stlev", "adm_earn"), ("sel", "sat_earn"),
              ("adm_stlev", "sat_earn"), ("adm_stlev", "adm_earn")]
    lab_l = ["broad\n\nSAT\npricing", "broad\n\nADM\npricing", "SAT+ADM\nonly\nSAT\npricing",
             "ADM +\nstate lvl\nSAT\npricing", "ADM +\nstate lvl\nADM\npricing"]
    xl_ = np.arange(len(rows_l)); w = 0.36
    for off, key, col, lab in ((-w / 2, "base", "#111827", "baseline ρ_f"),
                               (w / 2, "price", "#c2410c", "pricing measure")):
        v = [reg[r]["rank"][f"b_{key}"] for r in rows_l]
        q95 = [reg[r]["rank"][f"q95_{key}"] for r in rows_l]
        ax.bar(xl_ + off, v, width=w, color=col, alpha=0.8, label=lab)
        ax.scatter(xl_ + off, q95, marker="_", color="k", s=120, zorder=3)
    ax.scatter([], [], marker="_", color="k", s=120, label="95th pct of Freedman–Lane null")
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set_xticks(xl_); ax.set_xticklabels(lab_l, fontsize=7)
    ax.set_ylabel("standardized coefficient (ranks across fields)")
    ax.set_title("(l) Split-half regression by outcome and pricing measure\n(ranks across fields)", fontsize=9.5, loc="left")
    ax.legend(fontsize=6.5, loc="upper right", frameon=False)

    fig.suptitle("Is field coupling just the price of institutional selectivity? "
                 "(Scorecard FoS BA 4-yr medians × Wapman field prestige; fields with n≥15)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    print(f"[fig] {OUT_FIG}")


# ---------------------------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------------------------
def _f(x, nd=2, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _ci(ci, nd=2):
    return f"[{_f(ci[0], nd)}, {_f(ci[1], nd)}]" if np.all(np.isfinite(ci)) else "—"


def _nn(nn):
    if not nn or "obs" not in nn:
        return ""
    return f" ({_f(nn['nullB_mean'])}; {_fmt_p(nn['nullB_p'])})"


def _tv(v):
    return f"{v:.4f} (SD {np.sqrt(v):.3f})" if np.isfinite(v) and v > 0 else f"{v:.4f} (none)"


def _rt(r):
    if not np.isfinite(r):
        return "not identified (corrected variance ≤ 0)"
    return f"{r:+.2f}" if abs(r) <= 1 else f"not identified ({r:+.2f})"


SET_LAB = {"all": "all estimable", "large": f"n ≥ {SPLIT_NMIN}", "small": f"n < {SPLIT_NMIN}",
           "reliable": "reliable"}
SPEC_SHORT = {"sel": "SAT + ADM", "full_stlev": "broad", "full_state": "strict (state FE)",
              "adm_stlev": "ADM + state level", "adm_state": "ADM + state FE"}


def _write_order(L, ctx):
    order, split, fsp, fsum = ctx["order"], ctx["split"], ctx["fe_split"], ctx["fe_summ"]
    L.append("**Ordering tests: does the partial ρ_f ranking track the baseline ranking beyond shared "
             "noise?** Baseline and partial ρ_f are estimated on the same institutions. `n` in the field-set "
             "column is the field's institution count in the spec's sample (n_SAT or n_ADM). Null A: baseline "
             "truth = observed baseline; B: baseline shrunk to its estimated true spread; C: as B, with the part of "
             "the partial's bootstrap noise that is unrelated to the baseline's noise scaled by λ so that the "
             "null's cross-field variance of the partial equals the observed one (λ = 1: no rescaling needed).\n")
    L.append("| spec | field set | k | observed Spearman(baseline, partial) | within-field noise corr min / median / max | "
             "shared noise / observed cross-field cov | null A: mean [95th pct], p | null B: mean [95th pct], p | "
             "null C: mean, p (λ) | true between-field variance of partial ρ_f (moment) | true-score r (moment) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for spec in ORDER_SPECS:
        for sname in ("all", "large", "small", "reliable"):
            o = order[(spec, sname)]
            if o.get("k", 0) < 5:
                L.append(f"| {SPEC_SHORT[spec]} | {SET_LAB[sname]} | {o.get('k', 0)} | — | — | — | — | — | — | — | — |")
                continue
            L.append(f"| {SPEC_SHORT[spec]} | {SET_LAB[sname]} | {o['k']} | {_f(o['obs'])} | "
                     f"{_f(o['ncorr_min'])} / {_f(o['ncorr_med'])} / {_f(o['ncorr_max'])} | {o['share']:.0%} | "
                     f"{_f(o['nullA_mean'])} [{_f(o['nullA_q95'])}], {_fmt_p(o['nullA_p'])} | "
                     f"{_f(o['nullB_mean'])} [{_f(o['nullB_q95'])}], {_fmt_p(o['nullB_p'])} | "
                     f"{_f(o['nullC_mean'])}, {_fmt_p(o['nullC_p'])} ({o['lamC']:.2f}) | "
                     f"{_tv(o['tvy'])} | {_rt(o['r_true'])} |")
    L.append("")
    L.append(f"**Split-half cross-fit** (fields with n ≥ {SPLIT_NMIN} in the spec's sample; {SPLIT_K} random "
             "halvings of the institutions; baseline ρ_f on one half vs partial ρ_f on the other).\n")
    L.append("| spec | k fields | cross-fit Spearman: mean | 5th pct / min over splits | splits ≤ 0 | permutation p | "
             "same-half Spearman: mean | split-half reliability raw / partial (Pearson) | partial reliability (Spearman) | "
             "disattenuated true-score r: ratio of means / median over splits | half-sample partials undefined |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for spec in ORDER_SPECS:
        q = split[spec]
        L.append(f"| {SPEC_SHORT[spec]} | {q['k']} | {_f(q['cross_mean'])} | {_f(q['cross_q05'])} / "
                 f"{_f(q['cross_min'])} | {q['cross_le0']}/{q['K']} | {_fmt_p(q['p_perm'])} | {_f(q['same_mean'])} | "
                 f"{_f(q['relR'])} / {_f(q['relQ'])} | {_f(q['relQ_s'])} | {_f(q['dis'])} / {_f(q['dis_med'])} | "
                 f"{q['undef']:.0%} |")
    L.append("")
    L.append(f"**FE design: Spearman(β_f, baseline ρ_f) with independent halves** ({SPLIT_FE_K} random halvings; "
             "β_f on one half, ρ_f on the other).\n")
    L.append("| spec | fields in the full FE fit | fields per split (mean) | full-sample Spearman (shared noise) | "
             "cross-fit mean | 5th pct / min over splits | splits ≤ 0 | permutation p | same-half mean |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for tag, lab in (("a", "(a) α_i + γ_f + β_f z"), ("b", "(b) + field-specific SAT/ADM/Pell slopes"),
                     ("c", "(c) (b) + field-specific brand slope")):
        q = fsp[tag]
        L.append(f"| {lab} | {q['k_all']} | {q['k_mean']:.1f} | {_f(fsum[tag]['s_raw'])} | {_f(q['cross_mean'])} | "
                 f"{_f(q['cross_q05'])} / {_f(q['cross_min'])} | {q['cross_le0']}/{q['K']} | {_fmt_p(q['p_perm'])} | "
                 f"{_f(q['same_mean'])} |")
    L.append("")


def _write_referee(L, ctx):
    """Tables for: does an ordering that beats the shared-noise null answer the referee?"""
    grad, reg = ctx["grad"], ctx["reg"]
    g0 = grad[GRAD[0]]
    kl = g0["reg"]["rank"]["k"]
    L.append("### Does the surviving ordering answer the referee?\n")
    L.append("The shared-noise null above holds the true partial ρ_f constant across fields. The referee's "
             "hypothesis does not predict a constant partial when selectivity is measured imperfectly: institution-level "
             "SAT_AVG and ADM_RATE of recent entrants stand in for the student quality of a program's graduates, the part "
             "they miss stays in the partial, and it is priced more in fields where selectivity pays more. Under that "
             "hypothesis the partial is positive and ordered like the baseline. The two tables below try to separate that "
             "pattern from a field-specific prestige effect.\n")
    L.append(f"**(1) Control-precision gradient** (SAT sample; the {g0['k']} fields estimable at every step; controls "
             "added cumulatively from top to bottom). The full-sample Spearman shares noise with the baseline (bracket: "
             f"shared-noise null mean and p, variant B). Split-half columns: the {kl} fields with n_SAT ≥ {SPLIT_NMIN} "
             f"and the same {SPLIT_K} halvings as the broad cross-fit; true-score r = mean cross-half correlation / "
             "sqrt(product of mean split-half reliabilities), Pearson.\n")
    L.append("| step | controls | mean ρ_f | Spearman vs baseline (null mean; p) | cross-fit Spearman(baseline half A, step half B) | "
             "split-half reliability of the step | true-score r with baseline ρ_f | true-score r with pricing ρ(SAT_AVG, earn) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(GRAD):
        g, r, nn = grad[c], grad[c]["reg"], grad[c]["nn"]
        sb = (f"{_f(g['s_base'])} ({_f(nn['nullB_mean'])}; {_fmt_p(nn['nullB_p'])})" if nn else
              f"{_f(g['s_base'])} (re-estimate of the baseline; no null)")
        L.append(f"| {i} | {GRAD_LAB[c]} | {_f(g['mean'])} | {sb} | {_f(r['cf_base'])} | {_f(r['rel_y'])} | "
                 f"{_f(r['t1y'])} | {_f(r['t2y'])} |")
    L.append("")
    rc = reg[("raw", "sat_earn")]
    L.append(f"**(2) Split-half cross-field regression** ({SPLIT_K} halvings × 2 directions; fields with n ≥ "
             f"{SPLIT_NMIN} in the outcome's sample). Across fields: outcome on half B ~ baseline ρ_f on half A + "
             "pricing measure on half A (and A/B swapped), all three standardized across fields; coefficients averaged "
             "over replicates. `rank` = on ranks across fields (primary), `value` = on the estimates themselves. p = one-sided "
             f"Freedman–Lane permutation p ({N_PERM} field relabellings, same relabelling in every replicate). The "
             "calibration row regresses the baseline on itself (other half), where the true weights are 1 and 0; the "
             "weights it returns show how the two predictors share credit when nothing but the baseline matters. EIV = "
             "errors-in-variables version from cross-half moments (true-score correlations, then the standardized "
             "regression of the true outcome on the two true predictors); it has no p-value, and with predictor "
             "true-score correlations near +0.9 it is unstable.\n")
    L.append("| outcome (half B) | pricing (half A) | k | cross-fit Spearman with outcome: baseline / pricing | rank: β baseline (p) | "
             "rank: β pricing (p) | value: β baseline (p) | value: β pricing (p) | split-half reliability: baseline / pricing / "
             "outcome | true-score r(baseline, pricing) | EIV β baseline / pricing |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for out, pr, fam, lab in REG_ROWS:
        q = reg[(out, pr)]; a, v = q["rank"], q["val"]
        L.append(f"| {lab} | {PRICE_LAB[pr]} | {a['k']} | {_f(q['cf_base'])} / {_f(q['cf_price'])} | "
                 f"{_f(a['b_base'])} ({_fmt_p(a['p_base'])}) | {_f(a['b_price'])} ({_fmt_p(a['p_price'])}) | "
                 f"{_f(v['b_base'])} ({_fmt_p(v['p_base'])}) | {_f(v['b_price'])} ({_fmt_p(v['p_price'])}) | "
                 f"{q['rel_base']:.2f} / {q['rel_price']:.2f} / {q['rel_y']:.2f} | {_f(q['t12'])} | "
                 f"{_f(q['eb_base'])} / {_f(q['eb_price'])} |")
    drops = [(q["lab"], q["rank"]["R_drop"]) for q in reg.values() if q["rank"]["R_drop"] > 0]
    L.append("")
    L.append(f"Replicates used per row: {rc['rank']['R']} of {2 * SPLIT_K}"
             + ("" if not drops else "; rows with replicates dropped for an undefined half-sample estimate: "
                + ", ".join(f"{l_} ({d_})" for l_, d_ in drops))
             + ". The two predictors are measured with different reliability (baseline "
             f"{rc['rel_base']:.2f}, ρ(SAT_AVG, earn) {rc['rel_price']:.2f}, ρ(−ADM_RATE, earn) "
             f"{reg[('full_stlev', 'adm_earn')]['rel_price']:.2f} on the SAT-sample fields), which tilts the plain "
             "regression toward the more reliable one; the calibration row and the EIV column show the size of that tilt.\n")


def _write_md(ctx):
    T, base, summ, contr, brand, FE, fe_summ = (ctx[k] for k in
                                                ("T", "base", "summ", "contr", "brand", "FE", "fe_summ"))
    cov = ctx["cov"]
    L = []
    s_b = summ[("rho_full_stlev", False)]; s_s = summ[("rho_full_state", False)]
    s_sel = summ[("rho_sel", False)]; s_raw = summ[("rho_raw", False)]
    s_as = summ[("rho_adm_state", False)]; s_al = summ[("rho_adm_stlev", False)]
    fa, fb, fc = fe_summ["a"], fe_summ["b"], fe_summ["c"]

    L.append("# Selectivity confound test — is field coupling just the price of selectivity?\n")
    L.append("Public data only (College Scorecard institution + field-of-study files, Wapman field "
             "prestige). Descriptive; no causal claim. Every number below is produced by "
             "`scripts/55_selectivity.py` (seed 55; joint institution bootstrap, "
             f"B={B_BOOT}). Per-field table: `data/interim/selectivity_fields.csv`; figure: "
             "`outputs/figures/selectivity_coupling.png`.\n")

    # ---------------- answer ----------------
    L.append("## Answer\n")
    L.append("KEY_ANSWER_PLACEHOLDER\n")

    # ---------------- what changed ----------------
    L.append("## What changed in this revision\n")
    L.append("REVISION_PLACEHOLDER\n")

    # ---------------- key numbers ----------------
    L.append("## Key numbers\n")
    L.append("Cross-field statistics are over fields with a defined estimate (n≥15 institutions; "
             f"partial specs also need residual df≥{DFMIN}). SE = joint institution-bootstrap SE. "
             "τ = DerSimonian–Laird between-field SD (fields treated as independent; they share "
             "institutions, so τ and Q are indicative). `Spearman vs baseline` = Spearman across "
             "fields between the full-sample baseline ρ_f and the row's estimate; in brackets the mean "
             "of the shared-noise null (true value of the row's statistic constant across fields, "
             "joint-bootstrap noise pairs, variant B) and the share of null draws ≥ the observed value.\n")
    L.append("| quantity | sample / spec | k fields | mean | SD across fields | RMS SE | τ (DL) | Q p | τ_z, Q p (analytic Fisher-z SEs) | fields sig. >0 / <0 | Spearman vs baseline (null mean; p) | Spearman vs same-sample raw (null mean; p) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    rowsdef = [("rho_raw", "baseline ρ_f, full sample (= expanded66_gap_map `spearman`)"),
               ("rho_raw_sat", "raw ρ_f on SAT sample"),
               ("rho_sel", "partial given " + SPEC_DESC["sel"]),
               ("rho_full_stlev", "partial given " + SPEC_DESC["full_stlev"] + " **[broad]**"),
               ("rho_full_state", "partial given " + SPEC_DESC["full_state"] + " **[strict]**"),
               ("rho_raw_adm", "raw ρ_f on ADM sample"),
               ("rho_adm_stlev", "partial given " + SPEC_DESC["adm_stlev"]),
               ("rho_adm_state", "partial given " + SPEC_DESC["adm_state"]),
               ("rho_pcs", "partial given " + SPEC_DESC["pcs"] + " (gradient step 1)"),
               ("rho_pcs_adm", "partial given " + SPEC_DESC["pcs_adm"] + " (gradient step 2)"),
               ("rho_full_ie", "partial given " + SPEC_DESC["full_ie"] + " (gradient step 4)"),
               ("rho_sat_earn", "selectivity coupling ρ(SAT_AVG, earn), SAT sample"),
               ("rho_adm_earn", "selectivity coupling ρ(−ADM_RATE, earn), ADM sample"),
               ("rho_ie_earn", "ρ(institution MD_EARN_WNE_P10, field earn), ADM sample"),
               ("rho_all_common", "raw ρ_f, Pell/non-Pell common sample"),
               ("rho_pell", "ρ(prestige, Pell median earn), common sample"),
               ("rho_nopell", "ρ(prestige, non-Pell median earn), common sample")]
    for relonly in (False, True):
        if relonly:
            L.append("| *reliable fields only (expanded66 `reliable`)* | | | | | | | | | | | |")
        for col, lab in rowsdef:
            s = summ[(col, relonly)]
            sf = "—" if col == "rho_raw" else f"{_f(s['s_full'])}{_nn(s['nn_full'])}"
            ss = "—" if col == s["same"] else f"{_f(s['s_same'])}{_nn(s['nn_same'])}"
            L.append(f"| {col} | {lab} | {s['k']} | {_f(s['mean'])} | {_f(s['sd'], 3, False)} | "
                     f"{_f(s['rms_se'], 3, False)} | {_f(s['tau'], 3, False)} | {_fmt_p(s['p'])} | "
                     f"{_f(s['tau_z'], 3, False)}, {_fmt_p(s['p_z'])} | {s['n_pos']} / {s['n_neg']} | {sf} | {ss} |")
    L.append("")
    L.append("_Reading the table._ `fields sig. >0 / <0` counts fields whose estimate is more than 1.96 "
             "bootstrap SEs from zero. τ_z repeats the DL heterogeneity estimate on the Fisher-z scale with "
             "analytic variances 1.06/(df−1) instead of bootstrap SEs (τ_z is in z units, not ρ units). "
             "Rows compared with the baseline share its institutions, so a positive Spearman is expected "
             "from shared estimation noise alone; the bracketed null mean shows how large. No null is "
             "given for the raw-vs-raw rows (they re-estimate the same quantity on a subsample).\n")
    _write_order(L, ctx)
    _write_referee(L, ctx)
    ss_ = ctx["same_set"]
    L.append(f"**Broad vs strict on the same {ss_['k']} fields** (the fields where the state-FE spec is "
             "estimable): is the strict spec's weaker heterogeneity a different answer or more noise?\n")
    L.append("| column | mean | SD across fields | RMS SE | τ (DL) | Q p |")
    L.append("|---|---|---|---|---|---|")
    for col in ("rho_raw_sat", "rho_full_stlev", "rho_full_state"):
        q = ss_[col]
        L.append(f"| {col} | {_f(q['mean'])} | {q['sd']:.3f} | {q['rms_se']:.3f} | {q['tau']:.3f} | {_fmt_p(q['p'])} |")
    dq, nb = ss_["diff"], ss_["nn_broad_strict"]
    L.append(f"| strict − broad (paired) | {_f(dq['mean'])} | {dq['sd']:.3f} | {dq['rms_se']:.3f} | "
             f"{dq['tau']:.3f} | {_fmt_p(dq['p'])} |")
    L.append(f"\nSpearman(broad, strict) across these {ss_['k']} fields = {_f(ss_['s_broad_strict'])}; the two "
             f"specs share almost all their data, so under a constant true strict partial the shared-noise "
             f"null gives {_f(nb['nullB_mean'])} on average (95th pct {_f(nb['nullB_q95'])}; p = "
             f"{_fmt_p(nb['nullB_p'])}; variant A p = {_fmt_p(nb['nullA_p'])}; noise-calibrated variant C: mean "
             f"{_f(nb['nullC_mean'])}, p = {_fmt_p(nb['nullC_p'])}). The paired row tests whether "
             "strict and broad differ across fields by more than their (paired bootstrap) noise. Mean share "
             f"of bootstrap draws in which the strict estimate was not computable (df<{DFMIN}) = "
             f"{ss_['bootfail_strict']:.1%}.\n")

    L.append("**Named contrast** mean ρ(HIGH) − mean ρ(LOW), HIGH = {computer science, statistics, "
             "economics}, LOW = {nursing, communication disorders}; only fields with a defined estimate "
             "enter; bootstrap 95% CI.\n")
    L.append("| column | HIGH fields used | LOW fields used | contrast [95% CI] | fixed set {CS, economics} − {nursing} [95% CI] |")
    L.append("|---|---|---|---|---|")
    cf = ctx["contr_fix"]
    for col, c in contr.items():
        L.append(f"| {col} | {', '.join(c['hi']) or '—'} | {', '.join(c['lo']) or '—'} | "
                 f"{_f(c['est'])} {_ci((c['lo95'], c['hi95']))} | {_f(cf[col]['est'])} "
                 f"{_ci((cf[col]['lo95'], cf[col]['hi95']))} |")
    L.append("")

    L.append("**Named fields** (SE in parentheses; — = not estimable: n<15 in that sample or df<10).\n")
    L.append("| field | n | baseline ρ | n_SAT | raw ρ (SAT) | partial sel | partial broad | partial strict (df) | n_ADM | partial ADM+state-level | partial ADM+state FE (df) | ρ(SAT,earn) | ρ(prestige,SAT) | FE β_f (a) | FE β_f (b) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in HIGH + ["computer_engineering"] + LOW:
        r = base.loc[f]
        def v(c):
            return f"{_f(r[c])} ({_f(r['se_' + c], 2, False)})" if np.isfinite(r[c]) else "—"
        def fev(t):
            return f"{_f(r[f'fe_{t}_beta'], 3)} ({_f(r[f'fe_{t}_se'], 3, False)})" if np.isfinite(r[f"fe_{t}_beta"]) else "—"
        dfs = f" ({int(r['df_full_state'])})" if np.isfinite(r["df_full_state"]) else ""
        dfa = f" ({int(r['df_adm_state'])})" if np.isfinite(r["df_adm_state"]) else ""
        L.append(f"| {LAB.get(f, f)} | {int(r.n)} | {v('rho_raw')} | {int(r.n_sat)} | {v('rho_raw_sat')} | "
                 f"{v('rho_sel')} | {v('rho_full_stlev')} | {v('rho_full_state')}{dfs} | {int(r.n_adm)} | "
                 f"{v('rho_adm_stlev')} | {v('rho_adm_state')}{dfa} | {v('rho_sat_earn')} | "
                 f"{v('rho_prest_sat')} | {fev('a')} | {fev('b')} |")
    L.append("")

    L.append("**Brand vs field** (c_F = Spearman(field prestige, earn), c_G = Spearman(academia-wide "
             "Wapman rank, earn); ADV = c_F − c_G).\n")
    L.append("| sample / spec | k fields | mean c_F | mean c_G | mean ADV [95% CI] | ADV<0 (brand ≥ field) | reliable: ADV<0 / k |")
    L.append("|---|---|---|---|---|---|---|")
    r0 = brand["repro"]
    L.append(f"| reproduction of FIELD_VS_GENERIC (n≥8, full sample, raw) | {r0['k']} | {_f(r0['cF'])} | "
             f"{_f(r0['cG'])} | {_f(r0['cF'] - r0['cG'])} | {r0['neg']}/{r0['k']} | — |")
    for tag, lab in (("raw_sat", "raw, SAT sample (n≥15)"),
                     ("full_stlev", "partial, broad spec (SAT sample)"),
                     ("full_state", "partial, strict spec, state FE (SAT sample)")):
        b = brand[tag]
        L.append(f"| {lab} | {b['k']} | {_f(b['cF'])} | {_f(b['cG'])} | {_f(b['adv'])} {_ci(b['adv_ci'])} | "
                 f"{b['neg']}/{b['k']} | {b['neg_rel']}/{b['k_rel']} |")
    L.append("")

    L.append("**Within-institution cross-department design** (log earn = α_i + γ_f + β_f·z(prestige) + e; "
             "institution FE absorbed; SEs clustered by institution; institutions with one field and "
             f"fields with <{NMIN} cells dropped iteratively).\n")
    L.append("| spec | cells | institutions | fields | pooled common β (SE) | mean β_f | SD β_f | RMS SE | τ (DL) | Q p | Wald all β_f equal: χ²(df), p | Spearman(β_f, baseline ρ_f) all / reliable | HIGH−LOW β contrast (SE) | β_f>0 sig / <0 sig |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    fe_lab = {"a_ols": "no institution FE (field FE only) — comparator",
              "a": "(a) α_i + γ_f + β_f z", "a_bs": "(a) on (b)'s institution set",
              "b": "(b) (a) + field-specific slopes on SAT_AVG, ADM_RATE, PCTPELL",
              "c": "(c) (b) + field-specific slope on academia-wide brand"}
    for tag in ["a_ols", "a", "a_bs", "b", "c"]:
        e, s = FE[tag], fe_summ[tag]
        L.append(f"| {fe_lab[tag]} | {e['n_cells']} | {e['n_inst']} | {s['k']} | {_f(e['pooled'], 3)} "
                 f"({_f(e['pooled_se'], 3, False)}) | {_f(s['mean'], 3)} | {_f(s['sd'], 3, False)} | "
                 f"{_f(s['rms_se'], 3, False)} | {_f(s['tau'], 3, False)} | {_fmt_p(s['p'])} | "
                 f"{e['wald']:.1f}({e['wald_df']}), {_fmt_p(e['wald_p'])} | {_f(s['s_raw'])} / {_f(s['s_raw_rel'])} (k={s['k_rel']}) | "
                 f"{_f(e['contrast'], 3)} ({_f(e['contrast_se'], 3, False)}) | {s['n_pos_sig']} / {s['n_neg_sig']} |")
    L.append("")

    # horse race summary
    h = base[np.isfinite(base.get("hr_r2", pd.Series(dtype=float)))]
    L.append("**Horse race** (SAT sample; rank(earn) on z-ranked prestige, SAT_AVG, −ADM_RATE, "
             "MD_EARN_WNE_P10; OLS, HC1). Shapley R² shares over blocks {prestige}, {selectivity}, "
             "{institution-wide earnings}.\n")
    L.append("| fields | k | mean R² | mean Shapley prestige | mean Shapley selectivity | mean Shapley inst. earnings | leader = prestige / selectivity / inst. earnings | prestige β>0 with p<.05 (HC1) | mean prestige β |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for lab, q in (("all n≥15", h), ("reliable", h[h.reliable])):
        lead = q.hr_leader.value_counts()
        sig = int(((q.hr_b_prest - 1.96 * q.hr_se_prest) > 0).sum())
        L.append(f"| {lab} | {len(q)} | {q.hr_r2.mean():.2f} | {q.hr_shap_prest.mean():.3f} | "
                 f"{q.hr_shap_sel.mean():.3f} | {q.hr_shap_ie.mean():.3f} | "
                 f"{lead.get('prest', 0)} / {lead.get('sel', 0)} / {lead.get('ie', 0)} | {sig}/{len(q)} | "
                 f"{_f(q.hr_b_prest.mean(), 3)} |")
    L.append("")
    L.append("| field (reliable or named) | n | β prestige (SE) | β SAT (SE) | β −ADM (SE) | β inst. earn (SE) | R² | Shapley P / S / IE | unique R² P / S / IE |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    show = h[h.reliable | h.index.isin(HIGH + LOW)].sort_values("rho_raw", ascending=False)
    for f, r in show.iterrows():
        L.append(f"| {LAB.get(f, f)} | {int(r.hr_n)} | {_f(r.hr_b_prest)} ({r.hr_se_prest:.2f}) | "
                 f"{_f(r.hr_b_sat)} ({r.hr_se_sat:.2f}) | {_f(r.hr_b_adm)} ({r.hr_se_adm:.2f}) | "
                 f"{_f(r.hr_b_ie)} ({r.hr_se_ie:.2f}) | {r.hr_r2:.2f} | {r.hr_shap_prest:.2f} / "
                 f"{r.hr_shap_sel:.2f} / {r.hr_shap_ie:.2f} | {r.hr_uniq_prest:.3f} / {r.hr_uniq_sel:.3f} / "
                 f"{r.hr_uniq_ie:.3f} |")
    L.append("")

    # ---------------- method ----------------
    L.append("## Method\n")
    L.append(f"- **Baseline reproduced**: `compute_gap_map(load_ar_wapman(FIELDS66), load_er_scorecard(FIELDS66))` "
             f"returns the `spearman` and `n_institutions` of `outputs/expanded66_gap_map.csv` exactly "
             f"(max |diff| = {ctx['repro_maxdiff']:.1e}; all n equal). `reliable` flags are taken from that file.")
    L.append(f"- **Join**: `load_er_scorecard` keeps `institution_id` = Scorecard UNITID; the institution file "
             f"(`data/raw/scorecard_inst/Most-Recent-Cohorts-Institution.csv`) is joined on UNITID; all "
             f"{ctx['n_cells']} (field, institution) cells joined ({ctx['N']} distinct institutions).")
    L.append(f"- **Coverage** (share of the {ctx['N']} matched institutions): SAT_AVG {cov['SAT_AVG']:.1%}, "
             f"ADM_RATE {cov['ADM_RATE']:.1%}, PCTPELL {cov['PCTPELL']:.1%}, MD_EARN_WNE_P10 "
             f"{cov['MD_EARN_WNE_P10']:.1%}, state earnings level {cov['ST_EARN']:.1%}; complete SAT-family "
             f"covariates {cov['sat_family']:.1%}, ADM-family {cov['adm_family']:.1%}. "
             f"{ctx['miss_sat_ca']} of the {ctx['miss_sat']} institutions without SAT_AVG are in California "
             "(test-blind UC campuses among them), so the SAT sample under-represents CA public research "
             "universities. SAT coverage by academia-wide prestige quartile: " +
             ", ".join(f"{q} {r.sat:.0%} (ADM {r.adm:.0%}, n={int(r.n)})" for q, r in ctx["cov_q"].iterrows()) + ".")
    L.append("- **Partial coupling**: within each field sample, prestige, earnings and continuous covariates "
             "are converted to average ranks; prestige-rank and earnings-rank are each OLS-residualized on "
             "[1, covariate ranks, CONTROL dummy, state dummies or state earnings level]; partial ρ = "
             f"Pearson correlation of the two residual vectors. Reported only when n≥{NMIN} and "
             f"n − rank(Z) − 1 ≥ {DFMIN}. With state FE, institutions alone in their state within a field "
             "contribute nothing; that is why the strict spec is only estimable for larger fields and why "
             "the broad spec replaces state dummies by one covariate (leave-one-out mean log "
             "MD_EARN_WNE_P10 of PREDDEG=3 institutions in the state).")
    L.append("- **Joint institution bootstrap**: each draw resamples the matched institutions with "
             "replacement and recomputes every field's statistics on the same draw, so between-field "
             "dependence through shared institutions is kept in the CIs of cross-field contrasts (named "
             "contrasts, brand ADV) and in the noise pairs used by the shared-noise null.")
    L.append("- **Shared-noise null** (ordering test): for each field, the bootstrap deviations of (baseline ρ_f, "
             "partial ρ_f) from their bootstrap means are one draw of their joint estimation error. Null: the true "
             "partial ρ is the same in every field; simulated baseline = truth_raw + e_raw, simulated partial = "
             "e_partial, same draw for all fields; null Spearman over "
             f"{B_BOOT} draws. truth_raw = observed baseline (variant A) or the observed baseline shrunk toward "
             "its mean by sqrt(v/(v+SE_f²)), v = moment estimate of its true cross-field variance, so that the "
             "simulated baseline is not over-dispersed (variant B, reported by default). Variant C: as B, but where "
             "the bootstrap noise variance of the partial exceeds its observed cross-field variance (impossible "
             "under the null, where the observed spread is all noise; this happens for the state-FE partials), the "
             "component of the partial's noise that is unrelated to the baseline's noise (residual of a per-field "
             "regression of e_partial on e_raw over draws) is scaled by a common λ < 1 until the two match. p = "
             "(1 + #null ≥ observed)/(1 + #draws).")
    L.append("- **Moment decomposition**: observed cross-field sample (co)variances of (baseline, partial) minus the "
             "average cross-field sample (co)variances of the bootstrap noise vectors (this subtracts both the "
             "within-field noise and the part shared across fields through common institutions). True-score "
             "correlation = corrected covariance / sqrt(product of corrected variances); not identified when a "
             "corrected variance is ≤ 0 or the ratio falls outside [−1, 1].")
    L.append(f"- **Split-half cross-fit**: fields with n ≥ {SPLIT_NMIN} institutions in the spec's sample; the "
             f"{ctx['N']} matched institutions are split at random into two halves ({SPLIT_K} splits); baseline "
             "ρ_f on half A is correlated across fields with the partial ρ_f on half B and vice versa (independent "
             "noise), then averaged. Permutation p: the same random relabelling of fields applied to the partial "
             f"vector in every split ({N_PERM} permutations). Split-half reliabilities are Pearson correlations "
             "across fields between the two halves' estimates; the disattenuated true-score correlation is "
             "mean r(raw_A, partial_B) / sqrt(mean r(raw_A, raw_B) · mean r(partial_A, partial_B)). Half-sample "
             f"estimates need n ≥ {NMIN} and df ≥ {DFMIN} like the full-sample ones.")
    L.append("- **Control-precision gradient**: partial ρ_f on the SAT sample with cumulatively more institution-level "
             "controls (none; PCTPELL + CONTROL + state earnings level; + ADM_RATE; + SAT_AVG = broad spec; + "
             "MD_EARN_WNE_P10), on the fields where every step is estimable. Each step is in the joint bootstrap, so its "
             "Spearman with the baseline is tested against the same shared-noise null.")
    L.append(f"- **Split-half cross-field regression**: uses the same {SPLIT_K} halvings as the broad split-half "
             "cross-fit (the script asserts that its broad-row cross-fit Spearman and true-score r equal those of the "
             "cross-fit table). Within each half and field: baseline ρ_f (all rows), pricing ρ(SAT_AVG, earn) or "
             "ρ(−ADM_RATE, earn) and the partials (same n ≥ 15 / df ≥ 10 rules). Across fields, the outcome from one "
             "half is regressed on the baseline and pricing from the other half (both directions), after standardizing "
             "each vector across fields (on ranks for the `rank` columns). Freedman–Lane: residuals of the outcome on "
             "the other predictor are permuted over fields and added back to its fitted values; the focal "
             "coefficient's replicate-averaged value under each permutation forms the null. EIV: split-half "
             "reliabilities r_XX = mean corr(X_A, X_B) and cross-half correlations r_XY = mean of corr(X_A, Y_B) and "
             "corr(X_B, Y_A) give true-score correlations r_XY / sqrt(r_XX r_YY); the standardized regression of the "
             "outcome's true score on the two predictors' true scores follows.")
    L.append(f"- **FE-design cross-fit**: β_f of spec (a), (b) or (c) estimated on one random half of institutions "
             f"(same sample rules, fields with ≥{NMIN} cells in that half) and baseline ρ_f on the other half; "
             f"{SPLIT_FE_K} splits; permutation p as above.")
    L.append("- **Horse race**: SAT sample, all variables rank-transformed and standardized; Shapley R² "
             "averages each block's marginal R² over all block orderings (sums to total R²).")
    L.append("- **Pell / non-Pell**: `EARN_PELL_WNE_MDN_4YR` / `EARN_NOPELL_WNE_MDN_4YR` loaded with the "
             "same `load_er_scorecard` aggregation; compared with the all-student median on the common "
             "sample where all three are released.")
    L.append("- **Within-institution design**: pooled (field, institution) cells, prestige z-scored within "
             "field, log earnings; α_i absorbed by within-institution demeaning (exact for one-way FE); "
             "one field's γ is the reference; cluster-robust SEs by institution (FE nested in clusters, "
             "no df correction for α_i). Spec (b) adds field-specific slopes on the institution's "
             "z-scored SAT_AVG, ADM_RATE and PCTPELL, so field-specific pricing of institution-level "
             "selectivity cannot load onto β_f; spec (c) also adds a field-specific slope on the "
             "academia-wide brand rank. Wald test uses the clustered VCV (pseudo-inverse; df = its rank).")

    # ---------------- caveats ----------------
    L.append("\n## Caveats\n")
    L.append("- Selectivity is measured at the institution level for recent entering cohorts; earnings are "
             "for completers of earlier cohorts. Major-specific selection (e.g., separately-admitted "
             "engineering or nursing programs, sorting of stronger students into some majors) is not "
             "observed in public data, and no design here removes it. The institution-FE design absorbs only "
             "each institution's average level; spec (b) adds field-specific slopes on institution-level selectivity.")
    L.append("- Institution selectivity and field prestige are strongly correlated in most fields "
             "(see ρ(prestige, SAT) above), so partialling removes much of the prestige variation as well. "
             "A drop from raw to partial ρ is the share of the prestige–earnings association that cannot "
             "be separated from selectivity with these data; it is not proof that selectivity causes it.")
    L.append("- **Residual confounding from imperfect selectivity measures.** SAT_AVG and ADM_RATE describe an "
             "institution's recent entering class, not the students who completed a given major there years earlier. "
             "Whatever student quality these proxies miss stays in every partial ρ_f, correlated with prestige, and "
             "it is priced more in fields where selectivity pays more. So the referee's hypothesis, with these "
             "controls, predicts a positive partial ρ_f whose ordering across fields resembles the baseline's. "
             "An ordering that beats the constant-partial (shared-noise) null is therefore expected under both the "
             "referee's reading and a field-specific prestige reading. The gradient and split-half regression "
             "tables try to separate the two and cannot do so decisively.")
    t12s = [q["t12"] for q in ctx["reg"].values()]
    ks = [q["rank"]["k"] for q in ctx["reg"].values()]
    L.append("- The pricing measures are not pure prices of selectivity: SAT_AVG and ADM_RATE are correlated with "
             "prestige (see ρ(prestige, SAT)), so ρ(SAT_AVG, earn) and ρ(−ADM_RATE, earn) also carry some prestige "
             f"coupling. Across the large fields their true-score correlation with the baseline is {_f(min(t12s))} to "
             f"{_f(max(t12s))} (regression table), so the baseline ordering and the selectivity-pricing ordering are "
             f"nearly the same ordering, and {min(ks)}–{max(ks)} fields give little room to separate them.")
    L.append("- MD_EARN_WNE_P10 (institution-wide, all entrants, 10 yrs after entry) contains the field's own "
             "graduates, so its horse-race advantage is partly mechanical.")
    L.append("- Pell/non-Pell stratification controls for family income at entry, not for selectivity; "
             "Pell/non-Pell medians are released for fewer cells.")
    L.append("- The SAT sample excludes test-blind/test-optional institutions without SAT_AVG (most UC "
             "campuses, Caltech); the ADM-only specs keep them.")
    L.append("- Fields share institutions, so cross-field Q-tests and DL τ treat dependent estimates as "
             "independent; they are indicative only.")
    L.append("- The shared-noise null takes the joint-bootstrap deviations as the sampling error of each field's "
             "(baseline, partial) pair. For the state-FE partials the bootstrap overstates the noise (see below). If "
             "that excess is unrelated to the baseline's noise it dilutes the noise correlation and makes nulls A/B "
             "too lenient for those specs; variant C corrects for this under that assumption, so for the state-FE "
             "specs C is the p-value to read. The field-"
             "permutation p of the cross-fit tests (and the Freedman–Lane p of the split-half regression) treats "
             "fields as exchangeable under the null; fields differ in size, so it is approximate. The disattenuated correlations have no confidence interval here; the "
             "spread over random splits is split-to-split variation, not sampling uncertainty.")
    L.append(f"- The n ≥ {SPLIT_NMIN} threshold for 'well-estimated' fields follows the review that raised the "
             "shared-noise issue; other thresholds were not tried.")
    est = base[np.isfinite(base.rho_full_state)]
    nest = base[np.isfinite(base.rho_full_stlev) & ~np.isfinite(base.rho_full_state)]
    L.append(f"- Small fields have wide SEs. The strict state-FE spec is estimable only for fields with n_SAT ≥ "
             f"{int(est.n_sat.min())} (the largest field it fails for has n_SAT = {int(nest.n_sat.max())}); "
             f"statistics (n_SAT = {int(base.loc['statistics', 'n_sat'])}, df = "
             f"{int(base.loc['statistics', 'df_full_state'])}) is excluded there, and communication disorders "
             f"(n_SAT = {int(base.loc['communication_disorders', 'n_sat'])}) is excluded from every SAT spec.")
    L.append("- Bootstrap SEs for the state-FE partials are larger than analytic ones (resampled draws contain "
             "fewer distinct institutions per state, so they lose more df); the table reports both noise "
             "models (τ from bootstrap SEs, τ_z from analytic Fisher-z SEs) and they disagree on whether "
             "state-FE heterogeneity is detectable.")
    L.append("- Within-institution spec (a): with field-specific β_f and a common α_i, part of the "
             "cross-field β_f differences is identified from fields loading differently on institution-level "
             "traits correlated with prestige; spec (b) removes that channel for SAT/ADM/Pell, spec (c) also "
             "for the academia-wide brand. Spec (c) leaves only department prestige that deviates from the "
             "institution's brand, which is a small share of the variation, so its nulls have low power.")
    L.append("- Descriptive associations only; Scorecard medians cannot see the elite tail.")

    txt = "\n".join(L) + "\n"
    txt = txt.replace("KEY_ANSWER_PLACEHOLDER", _answer(ctx))
    txt = txt.replace("REVISION_PLACEHOLDER", _revision(ctx))
    OUT_MD.write_text(txt)
    print(f"[result] {OUT_MD}")


def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "—"
    return f"{p:.2g}" if p >= 1e-4 else "<1e-4"


def _p(p):
    """p-value for prose."""
    return f"p = {_fmt_p(p)}" if p >= 1e-4 else "p < 1e-4"


def _answer(ctx):
    """Plain-language answer assembled from computed numbers only."""
    summ, contr, brand, FE, fe_summ, base = (ctx[k] for k in ("summ", "contr", "brand", "FE", "fe_summ", "base"))
    order, split, fsp = ctx["order"], ctx["split"], ctx["fe_split"]
    sb, ss, sr = summ[("rho_full_stlev", False)], summ[("rho_full_state", False)], summ[("rho_raw", False)]
    sbs = summ[("rho_raw_sat", False)]
    sbr, ssr = summ[("rho_full_stlev", True)], summ[("rho_full_state", True)]
    sa = summ[("rho_adm_stlev", False)]
    cb, cs, ca = contr["rho_full_stlev"], contr["rho_full_state"], contr["rho_adm_stlev"]
    fa, fb = fe_summ["a"], fe_summ["b"]
    ob, os_, oa = order[("full_stlev", "all")], order[("full_state", "all")], order[("adm_stlev", "all")]
    obr, osr = order[("full_stlev", "reliable")], order[("full_state", "reliable")]
    A = []
    A.append(
        f"1. **Level: coupling shrinks substantially once selectivity is netted out.** Across the "
        f"{sb['k']} fields estimable in the broad spec (SAT, ADM, Pell share, control, state earnings "
        f"level; SAT sample), mean partial ρ_f = {_f(sb['mean'])} vs {_f(sb['mean_same'])} raw on the same "
        f"sample; in the strict spec (state FE, {ss['k']} fields) {_f(ss['mean'])} vs {_f(ss['mean_same'])}. "
        f"Selectivity itself couples with field earnings at least as strongly as prestige does: mean "
        f"ρ(SAT, earn) = {_f(summ[('rho_sat_earn', False)]['mean'])} vs mean ρ(prestige, earn) = "
        f"{_f(sbs['mean'])} on the same SAT sample ({summ[('rho_sat_earn', False)]['k']} fields).")
    A.append(
        f"2. **Cross-field ordering (the paper's object), against the shared-noise null** (constant true "
        f"partial ρ; p for variants B / A / C): Spearman(baseline, partial) = {_f(ob['obs'])} (broad, {ob['k']} "
        f"fields; null mean {_f(ob['nullB_mean'])}, {_p(ob['nullB_p'])} / {_p(ob['nullA_p'])} / {_p(ob['nullC_p'])}), "
        f"{_f(os_['obs'])} (strict, {os_['k']}; null {_f(os_['nullB_mean'])}, {_p(os_['nullB_p'])} / "
        f"{_p(os_['nullA_p'])} / {_p(os_['nullC_p'])}), {_f(oa['obs'])} (ADM-only + state level, {oa['k']}; null "
        f"{_f(oa['nullB_mean'])}, {_p(oa['nullB_p'])} / {_p(oa['nullA_p'])} / {_p(oa['nullC_p'])}). Reliable fields only: "
        f"{_f(obr['obs'])} (broad, k={obr['k']}; null {_f(obr['nullB_mean'])}, {_p(obr['nullB_p'])}), "
        f"{_f(osr['obs'])} (strict, k={osr['k']}; null {_f(osr['nullB_mean'])}, {_p(osr['nullB_p'])}). "
        f"Between-field SD of partial ρ: {sb['sd']:.3f} (broad) vs RMS bootstrap SE {sb['rms_se']:.3f}, "
        f"DL τ = {sb['tau']:.3f} (Q p = {_fmt_p(sb['p'])}); strict: SD {ss['sd']:.3f} vs RMS SE "
        f"{ss['rms_se']:.3f}, τ = {ss['tau']:.3f} (Q p = {_fmt_p(ss['p'])}). For comparison the baseline "
        f"has SD {sr['sd']:.3f}, RMS SE {sr['rms_se']:.3f}, τ = {sr['tau']:.3f}. The null here holds the true "
        "partial constant across fields; that is not what the referee's hypothesis predicts with imperfect "
        "selectivity proxies (see 'Does the surviving ordering answer the referee?' below).")
    A.append(
        f"3. **Named contrast** (mean HIGH − mean LOW): baseline {_f(contr['rho_raw']['est'])} "
        f"{_ci((contr['rho_raw']['lo95'], contr['rho_raw']['hi95']))}; broad partial {_f(cb['est'])} "
        f"{_ci((cb['lo95'], cb['hi95']))} (LOW = {', '.join(cb['lo']) or '—'}); strict partial "
        f"{_f(cs['est'])} {_ci((cs['lo95'], cs['hi95']))} (HIGH = {', '.join(cs['hi']) or '—'}; LOW = "
        f"{', '.join(cs['lo']) or '—'}); ADM-only + state level {_f(ca['est'])} "
        f"{_ci((ca['lo95'], ca['hi95']))} (LOW = {', '.join(ca['lo']) or '—'}). "
        f"Within-institution FE design (a): HIGH−LOW β contrast {_f(FE['a']['contrast'], 3)} "
        f"(SE {FE['a']['contrast_se']:.3f}); with field-specific selectivity slopes (b): "
        f"{_f(FE['b']['contrast'], 3)} (SE {FE['b']['contrast_se']:.3f}).")
    b0, b1, b2 = brand["repro"], brand["full_stlev"], brand["full_state"]
    A.append(
        f"4. **Brand ≥ field:** reproduced raw (mean c_F {_f(b0['cF'])}, c_G {_f(b0['cG'])}; ADV<0 in "
        f"{b0['neg']}/{b0['k']}). Net of selectivity (broad): mean c_F {_f(b1['cF'])}, c_G {_f(b1['cG'])}, "
        f"ADV {_f(b1['adv'])} {_ci(b1['adv_ci'])}, ADV<0 in {b1['neg']}/{b1['k']}; strict: c_F {_f(b2['cF'])}, "
        f"c_G {_f(b2['cG'])}, ADV {_f(b2['adv'])} {_ci(b2['adv_ci'])}, ADV<0 in {b2['neg']}/{b2['k']}.")
    A.append(
        f"5. **Within-institution design** (institution FE absorb each institution's average level: brand, "
        f"selectivity, geography, Pell composition): pooled β = {_f(FE['a']['pooled'], 3)} log points per SD of field prestige "
        f"(SE {FE['a']['pooled_se']:.3f}; {FE['a']['n_cells']} cells, {FE['a']['n_inst']} institutions); "
        f"without institution FE the same sample gives {_f(FE['a_ols']['pooled'], 3)} "
        f"(SE {FE['a_ols']['pooled_se']:.3f}). β_f heterogeneity: SD {fa['sd']:.3f} vs RMS SE "
        f"{fa['rms_se']:.3f}, τ = {fa['tau']:.3f}, Wald equal-β p = {_fmt_p(FE['a']['wald_p'])}. "
        f"Spearman(β_f, baseline ρ_f) on the full sample = {_f(fa['s_raw'])} (k={fa['k']}), but β_f and ρ_f "
        f"share their data; with β_f on one random half of institutions and ρ_f on the other it is "
        f"{_f(fsp['a']['cross_mean'])} on average over {SPLIT_FE_K} splits (≤0 in {fsp['a']['cross_le0']}/{SPLIT_FE_K}; "
        f"permutation {_p(fsp['a']['p_perm'])}). With field-specific selectivity/Pell slopes (b): pooled β "
        f"{_f(FE['b']['pooled'], 3)} (SE {FE['b']['pooled_se']:.3f}), SD {fb['sd']:.3f} vs RMS SE "
        f"{fb['rms_se']:.3f}, τ = {fb['tau']:.3f}; Spearman(β_f, ρ_f) {_f(fb['s_raw'])} full sample, "
        f"{_f(fsp['b']['cross_mean'])} cross-fit (≤0 in {fsp['b']['cross_le0']}/{SPLIT_FE_K}; permutation "
        f"{_p(fsp['b']['p_perm'])}). Adding field-specific brand slopes (c): Spearman(β_f, ρ_f) "
        f"{_f(fe_summ['c']['s_raw'])} full sample, {_f(fsp['c']['cross_mean'])} cross-fit (≤0 in "
        f"{fsp['c']['cross_le0']}/{SPLIT_FE_K}; permutation {_p(fsp['c']['p_perm'])}); HIGH−LOW β contrast "
        f"{_f(FE['c']['contrast'], 3)} (SE {FE['c']['contrast_se']:.3f}).")
    sp, snp, sac = summ[("rho_pell", False)], summ[("rho_nopell", False)], summ[("rho_all_common", False)]
    A.append(
        f"6. **Pell / non-Pell (SES stratification, not a selectivity control):** on the common sample "
        f"({sac['k']} fields) mean ρ = {_f(sac['mean'])} (all students), {_f(sp['mean'])} (Pell), "
        f"{_f(snp['mean'])} (non-Pell); Spearman across fields vs the all-student ρ on that sample: "
        f"Pell {_f(sp['s_same'])} (shared-noise null mean {_f(sp['nn_same']['nullB_mean'])}, "
        f"{_p(sp['nn_same']['nullB_p'])}), non-Pell {_f(snp['s_same'])} (null "
        f"{_f(snp['nn_same']['nullB_mean'])}, {_p(snp['nn_same']['nullB_p'])}).")
    return _verdict(ctx) + "\n\n**Detail.**\n\n" + "\n".join(A)


def _verdict(ctx):
    summ, contr_fix, FE, fe_summ, brand, base, ss_ = (ctx[k] for k in (
        "summ", "contr_fix", "FE", "fe_summ", "brand", "base", "same_set"))
    order, split, fsp, reg, grad = ctx["order"], ctx["split"], ctx["fe_split"], ctx["reg"], ctx["grad"]
    sb, ss, sbs = (summ[(c, False)] for c in ("rho_full_stlev", "rho_full_state", "rho_raw_sat"))
    ob, os_, oa, oas = (order[(s, "all")] for s in ("full_stlev", "full_state", "adm_stlev", "adm_state"))
    obl, osm, obr = (order[("full_stlev", s)] for s in ("large", "small", "reliable"))
    spb, sps = split["full_stlev"], split["full_state"]
    rc, rb, rba = reg[("raw", "sat_earn")], reg[("full_stlev", "sat_earn")], reg[("full_stlev", "adm_earn")]
    ras, raa = reg[("adm_stlev", "sat_earn")], reg[("adm_stlev", "adm_earn")]
    fa, fb, fcs = fe_summ["a"], fe_summ["b"], fe_summ["c"]
    drop = 1 - sb["mean"] / sb["mean_same"]
    fe_drop = 1 - FE["a"]["pooled"] / FE["a_ols"]["pooled"]
    pmax = lambda o: max(o["nullA_p"], o["nullB_p"], o["nullC_p"])
    arrow = lambda vals: " → ".join(_f(v) for v in vals)
    G = [grad[c] for c in GRAD]
    kg, kl = G[0]["k"], G[0]["reg"]["rank"]["k"]
    t1 = [g["reg"]["t1y"] for g in G]; t2 = [g["reg"]["t2y"] for g in G]
    ib = GRAD.index("rho_full_stlev")
    a_c, a_b = rc["rank"], rb["rank"]
    price_wins = a_b["p_price"] < 0.05 and a_b["p_base"] >= 0.05
    base_wins = a_b["p_base"] < 0.05 and a_b["p_price"] >= 0.05
    cfb, cfs = contr_fix["rho_full_stlev"], contr_fix["rho_full_state"]
    b1 = brand["full_stlev"]
    fec_zero = abs(FE["c"]["contrast"]) < 1.96 * FE["c"]["contrast_se"]
    cs, ec, nu = (base.loc[f] for f in ("computer_science", "economics", "nursing"))
    sat_m = summ[("rho_sat_earn", False)]["mean"]
    q_ = base[np.isfinite(base.rho_full_stlev)]
    sig_fields = list(q_[(q_.rho_full_stlev - 1.96 * q_.se_rho_full_stlev) > 0]
                      .sort_values("rho_full_stlev", ascending=False).index)
    hq = base[np.isfinite(base["hr_r2"])]
    shap = {b: float(hq[f"hr_shap_{b}"].mean()) for b in ("prest", "sel", "ie")}
    hk, lead_p = len(hq), int((hq.hr_leader == "prest").sum())
    large_carries = (obl["tvy"] > 0) and not (osm["tvy"] > 0.25 * obl["tvy"])
    V = []
    V.append(
        "**Short answer.**\n\n"
        f"1. **Level: most of the average coupling cannot be separated from institutional selectivity.** Mean partial "
        f"ρ_f net of SAT, admission rate, Pell share, control and state earnings level is {_f(sb['mean'])} vs "
        f"{_f(sb['mean_same'])} raw on the same SAT sample ({sb['k']} fields; {drop:.0%} lower). In the "
        f"within-institution design the pooled prestige slope falls {fe_drop:.0%} once institution fixed effects are "
        "added.\n"
        f"2. **Ordering: a cross-field ordering remains, but public data cannot tell it apart from field-specific "
        f"pricing of selectivity, so it does not answer the referee's objection.** The broad partial still ranks "
        f"fields like the baseline beyond shared estimation noise (Spearman {_f(ob['obs'])} vs a null mean of "
        f"{_f(ob['nullB_mean'])}, {_p(ob['nullB_p'])}). That null assumes the true partial is the same in every "
        "field. With institution-level SAT/ADM standing in for the student quality of a program's graduates, the "
        "referee's hypothesis predicts something else: a positive partial that is larger where selectivity is priced "
        "more, i.e. ordered like the baseline. "
        + ("The checks that try to separate the two point toward selectivity pricing, but not decisively. "
           if price_wins else "The checks that try to separate the two do not settle it. ")
        + f"Across the {kl} large fields, baseline coupling and selectivity pricing "
        f"ρ(SAT_AVG, earnings) have a true-score correlation of {_f(rb['t12'])}. "
        + (f"In a split-half regression the broad partial loads on selectivity pricing ({_f(a_b['b_price'])}, "
           f"{_p(a_b['p_price'])}) and not on baseline coupling ({_f(a_b['b_base'])}, {_p(a_b['p_base'])}). "
           if price_wins else
           f"In a split-half regression the broad partial loads on baseline coupling ({_f(a_b['b_base'])}, "
           f"{_p(a_b['p_base'])}) rather than on selectivity pricing ({_f(a_b['b_price'])}, {_p(a_b['p_price'])}). "
           if base_wins else
           f"In a split-half regression neither predictor is clearly ahead (baseline {_f(a_b['b_base'])}, "
           f"{_p(a_b['p_base'])}; pricing {_f(a_b['b_price'])}, {_p(a_b['p_price'])}). ")
        + (f"(With no controls the same regression splits the weight {_f(G[0]['reg']['rank']['b_base'])} baseline / "
           f"{_f(G[0]['reg']['rank']['b_price'])} pricing, so pricing's weight stays about the same while the "
           "baseline's goes to about zero.) " if price_wins else "")
        + f"The within-institution design with field-specific selectivity and brand slopes leaves "
        f"a HIGH−LOW β contrast of {_f(FE['c']['contrast'], 3)} (SE {FE['c']['contrast_se']:.3f}), although its "
        f"cross-field ordering of β_f still correlates with the baseline across independent halves "
        f"({_f(fsp['c']['cross_mean'])}, {_p(fsp['c']['p_perm'])}).\n"
        f"3. **Named contrasts shrink.** {{CS, economics}} − {{nursing}} is {_f(contr_fix['rho_raw']['est'])} at "
        f"baseline, {_f(cfb['est'])} {_ci((cfb['lo95'], cfb['hi95']))} in the broad spec and {_f(cfs['est'])} "
        f"{_ci((cfs['lo95'], cfs['hi95']))} in the strict state-FE spec.\n"
        f"4. **Brand ≥ field does not survive adjustment**: net of selectivity, mean ADV = c_F − c_G = "
        f"{_f(b1['adv'])} {_ci(b1['adv_ci'])}.")
    V.append("")
    V.append(
        f"- Level (detail): {sb['n_pos']}/{sb['k']} fields keep a broad partial ρ_f more than 1.96 SE above zero "
        f"({', '.join(LAB.get(f, f) for f in sig_fields)}) vs {sbs['n_pos']}/{sbs['k']} raw. In the within-institution "
        f"design the pooled slope goes from {_f(FE['a_ols']['pooled'], 3)} to {_f(FE['a']['pooled'], 3)} log points per "
        f"SD. Mean ρ(SAT, earnings) {_f(sat_m)} {'exceeds' if sat_m > sbs['mean'] else 'does not exceed'} mean "
        f"ρ(prestige, earnings) {_f(sbs['mean'])} on the same sample. In the horse race the mean Shapley R² share is "
        f"{shap['prest']:.3f} for prestige, {shap['sel']:.3f} for selectivity and {shap['ie']:.3f} for institution-wide "
        f"earnings ({hk} fields; prestige is the largest block in {lead_p} of them).")
    V.append(
        f"- Ordering vs shared noise. Baseline and partial ρ_f are estimated on the same institutions, so their errors "
        f"are correlated (within-field bootstrap correlation median {_f(ob['ncorr_med'])}, broad spec). Shared noise is "
        f"{ob['share']:.0%} of the observed cross-field covariance. Against the constant-partial null: broad "
        f"{_f(ob['obs'])} ({_p(ob['nullB_p'])}; largest p over the three null variants {_fmt_p(pmax(ob))}), strict state FE "
        f"{_f(os_['obs'])} (null mean {_f(os_['nullB_mean'])}, {_p(os_['nullB_p'])}; noise-calibrated C "
        f"{_p(os_['nullC_p'])}), ADM + state level {_f(oa['obs'])} (null {_f(oa['nullB_mean'])}, {_p(oa['nullB_p'])}), "
        f"ADM + state FE {_f(oas['obs'])} (null {_f(oas['nullB_mean'])}, {_p(oas['nullB_p'])}). With independent halves "
        f"({spb['k']} fields with n_SAT ≥ {SPLIT_NMIN}) the baseline-vs-broad-partial Spearman is {_f(spb['cross_mean'])} "
        f"(field-permutation {_p(spb['p_perm'])}; true-score r {_f(spb['dis'])}). The full-sample rank test on the "
        f"{obl['k']} large fields alone gives {_f(obl['obs'])} (null mean {_f(obl['nullB_mean'])}, {_p(obl['nullB_p'])}), "
        f"and on the {obr['k']} reliable fields {_f(obr['obs'])} (null mean {_f(obr['nullB_mean'])}, {_p(obr['nullB_p'])}).")
    V.append(
        "- Why beating that null does not settle the question: see the caveat on residual confounding. Imperfect "
        "institution-level proxies leave part of student quality in the partial, and fields that price selectivity "
        "more will show a larger partial. The referee's reading and a field-specific prestige reading both predict an "
        "ordering that beats the constant-partial null.")
    V.append(
        f"- Control-precision gradient (SAT sample, {kg} fields estimable at every step; controls added in the order "
        f"none → Pell share, control, state level → + ADM_RATE → + SAT_AVG → + institution-wide earnings): the mean "
        f"partial goes {arrow([g['mean'] for g in G])} and its Spearman with the baseline goes "
        f"{arrow([g['s_base'] for g in G])} (shared-noise null means {arrow([g['nn']['nullB_mean'] for g in G[1:]])} for "
        f"steps 1–4). On the {kl} fields with n_SAT ≥ {SPLIT_NMIN}, the true-score correlation of the step's partial "
        f"with the baseline goes {arrow(t1)}, and with selectivity pricing ρ(SAT_AVG, earn) {arrow(t2)}. Up to the SAT "
        f"step, each added control removes more of the level and more of the baseline-like ordering"
        + (f", and more of the alignment with the baseline ({_f(t1[0])} → {_f(t1[ib])}) than of the alignment with "
           f"selectivity pricing ({_f(t2[0])} → {_f(t2[ib])})" if (t1[0] - t1[ib]) > (t2[0] - t2[ib]) else "")
        + f". The last step changes little (mean {_f(G[ib]['mean'])} → {_f(G[-1]['mean'])}, Spearman "
        f"{_f(G[ib]['s_base'])} → {_f(G[-1]['s_base'])}). A stable field-specific component would produce that, and so "
        "would institution-wide earnings adding little information on program-level student quality beyond SAT and "
        "ADM. The gradient cannot separate the two.")
    V.append(
        f"- Split-half cross-field regression ({kl} fields, {SPLIT_K} halvings × 2 directions, ranks, one-sided "
        f"Freedman–Lane p): for the broad partial on one half, the other half's selectivity pricing ρ(SAT_AVG, earn) "
        f"gets {_f(a_b['b_price'])} ({_p(a_b['p_price'])}) and the other half's baseline {_f(a_b['b_base'])} "
        f"({_p(a_b['p_base'])}). The method is not neutral. In the calibration row the outcome is the baseline itself "
        f"(true weights 1 and 0), and the regression still gives {_f(a_c['b_price'])} ({_p(a_c['p_price'])}) to pricing "
        f"and {_f(a_c['b_base'])} to the baseline. The reason is that the two predictors' true scores correlate "
        f"{_f(rc['t12'])} and pricing is measured more reliably (split-half reliability {rc['rel_price']:.2f} vs "
        f"{rc['rel_base']:.2f}). What carries information is the shift from the calibration row to the broad row: "
        "weight moves from the baseline to pricing as selectivity is controlled. Along the gradient, with the "
        "sample-matched step 0 (raw ρ_f on the SAT sample) as the reference, pricing's coefficient goes "
        f"{arrow([g['reg']['rank']['b_price'] for g in G])} and the baseline's "
        f"{arrow([g['reg']['rank']['b_base'] for g in G])}. The result depends on the pricing "
        f"measure. With ρ(−ADM_RATE, earn) (reliability {rba['rel_price']:.2f}) the broad partial gives the baseline "
        f"{_f(rba['rank']['b_base'])} ({_p(rba['rank']['p_base'])}) and pricing {_f(rba['rank']['b_price'])} "
        f"({_p(rba['rank']['p_price'])}). For the ADM-sample partial (ADM + Pell, control, state level; "
        f"{ras['rank']['k']} fields), ρ(SAT_AVG, earn) gets {_f(ras['rank']['b_price'])} ({_p(ras['rank']['p_price'])}) "
        f"vs baseline {_f(ras['rank']['b_base'])} ({_p(ras['rank']['p_base'])}), while ρ(−ADM_RATE, earn) gets "
        f"{_f(raa['rank']['b_price'])} ({_p(raa['rank']['p_price'])}) vs baseline {_f(raa['rank']['b_base'])} "
        f"({_p(raa['rank']['p_base'])}). SAT_AVG is the stronger selectivity measure (mean ρ(SAT, earn) {_f(sat_m)} vs "
        f"ρ(−ADM, earn) {_f(summ[('rho_adm_earn', False)]['mean'])})"
        + (", and it favours pricing in both samples, but it is also the more reliable predictor, which the "
           "calibration row shows helps it. " if (price_wins and ras['rank']['p_price'] < 0.05) else
           "; the SAT-based rows do not consistently favour pricing. ")
        + f"The errors-in-variables version gives the broad "
        f"partial true-score coefficients {_f(rb['eb_base'])} (baseline) and {_f(rb['eb_price'])} (pricing), against "
        f"{_f(rc['eb_base'])} / {_f(rc['eb_price'])} in the calibration row. The direction is the same, but at a "
        f"predictor true-score correlation of {_f(rb['t12'])} these coefficients are unstable"
        + (" (values outside [−1, 1])" if max(abs(rb['eb_base']), abs(rb['eb_price'])) > 1 else "")
        + " and are not evidence on their own.")
    V.append(
        f"- Within-institution design, next to the ordering claim (β_f on one random half of institutions, baseline ρ_f "
        f"on the other; {SPLIT_FE_K} splits): Spearman(β_f, ρ_f) is {_f(fsp['a']['cross_mean'])} in spec (a) "
        f"(permutation {_p(fsp['a']['p_perm'])}), {_f(fsp['b']['cross_mean'])} with field-specific SAT/ADM/Pell slopes "
        f"(b) ({_p(fsp['b']['p_perm'])}), and {_f(fsp['c']['cross_mean'])} with field-specific brand slopes as well (c) "
        f"({_p(fsp['c']['p_perm'])}). The HIGH−LOW β contrast goes {_f(FE['a']['contrast'], 3)} (SE "
        f"{FE['a']['contrast_se']:.3f}) → {_f(FE['b']['contrast'], 3)} ({FE['b']['contrast_se']:.3f}) → "
        f"{_f(FE['c']['contrast'], 3)} ({FE['c']['contrast_se']:.3f})"
        + (", and the last value is not distinguishable from zero. " if fec_zero else ". ")
        + "Institution fixed effects absorb each institution's average level. They do not absorb field-specific "
        "pricing of institution-level selectivity (that is what (b) adds), and they do not touch selection into "
        "majors within an institution.")
    V.append(
        f"- Where the remaining broad-spec heterogeneity sits: moment estimate of the true between-field variance "
        f"{_tv(obl['tvy'])} across the {obl['k']} fields with n_SAT ≥ {SPLIT_NMIN}, {_tv(osm['tvy'])} across the "
        f"{osm['k']} smaller fields"
        + (" (so the 46-field DL τ " if large_carries else " (the 46-field DL τ ")
        + f"{sb['tau']:.3f}, Q p = {_fmt_p(sb['p'])}, "
        + ("is carried by the large fields)." if large_carries else "is not concentrated in the large fields).")
        + f" Strict spec (state FE, {ss['k']} fields): SD {ss['sd']:.3f} vs RMS bootstrap SE {ss['rms_se']:.3f}, τ "
        f"{ss['tau']:.3f} (Q p = {_fmt_p(ss['p'])}; analytic Fisher-z τ_z {ss['tau_z']:.3f}, Q p = {_fmt_p(ss['p_z'])}); "
        f"its ordering beats the shared-noise null ({_p(os_['nullB_p'])}; C {_p(os_['nullC_p'])}), and the paired "
        f"strict − broad differences show no disagreement beyond noise (Q p = {_fmt_p(ss_['diff']['p'])}). The bootstrap "
        f"overstates state-FE noise (noise variance {os_['Nyy']:.4f} vs observed {os_['Syy']:.4f}); split-half is not "
        f"usable there ({sps['undef']:.0%} of half-sample estimates undefined).")
    V.append(
        f"- Named fields: computer science keeps a positive partial in every spec (broad {_f(cs.rho_full_stlev)} ± "
        f"{cs.se_rho_full_stlev:.2f} SE; strict {_f(cs.rho_full_state)} ± {cs.se_rho_full_state:.2f}); economics is "
        f"the most selectivity-sensitive of the high fields (broad {_f(ec.rho_full_stlev)} ± {ec.se_rho_full_stlev:.2f}; "
        f"strict {_f(ec.rho_full_state)} ± {ec.se_rho_full_state:.2f}); nursing stays near zero (broad "
        f"{_f(nu.rho_full_stlev)}, strict {_f(nu.rho_full_state)}). Statistics is too small for the state-FE specs (broad "
        f"partial {_f(base.loc['statistics', 'rho_full_stlev'])}, SE {base.loc['statistics', 'se_rho_full_stlev']:.2f}); "
        "communication disorders is too small for every SAT spec (ADM + state level: "
        f"{_f(base.loc['communication_disorders', 'rho_adm_stlev'])}, SE "
        f"{base.loc['communication_disorders', 'se_rho_adm_stlev']:.2f}).")
    V.append(
        f"- Brand ≥ field (detail): before adjustment (same SAT sample) mean ADV = "
        f"{_f(brand['raw_sat']['adv'])} {_ci(brand['raw_sat']['adv_ci'])}, brand ahead in "
        f"{brand['raw_sat']['neg']}/{brand['raw_sat']['k']} fields; net of selectivity {_f(b1['adv'])} "
        f"{_ci(b1['adv_ci'])}, brand ahead in {b1['neg']}/{b1['k']} ({b1['neg_rel']}/{b1['k_rel']} reliable). Brand's "
        "raw edge over field prestige is the part of brand that tracks selectivity; net of it the two are tied on average.")
    sp, snp, sac = (summ[(c, False)] for c in ("rho_pell", "rho_nopell", "rho_all_common"))
    V.append(
        f"- SES strata (not a selectivity control): coupling persists within Pell and within non-Pell graduates (mean "
        f"ρ {_f(sp['mean'])} and {_f(snp['mean'])} vs {_f(sac['mean'])} for all students, {sac['k']}-field common sample) "
        f"and keeps the all-student ordering beyond shared noise (Spearman {_f(sp['s_same'])} and {_f(snp['s_same'])}; "
        f"null means {_f(sp['nn_same']['nullB_mean'])} and {_f(snp['nn_same']['nullB_mean'])}, "
        f"{_p(sp['nn_same']['nullB_p'])} and {_p(snp['nn_same']['nullB_p'])}). Pell status is not selectivity, so this "
        "does not bear on the objection.")
    return "\n".join(V)


def _revision(ctx):
    reg, grad, fsp, FE = ctx["reg"], ctx["grad"], ctx["fe_split"], ctx["FE"]
    rb, rc = reg[("full_stlev", "sat_earn")], reg[("raw", "sat_earn")]
    ob = ctx["order"][("full_stlev", "all")]
    R = []
    R.append("Revision 3 (this version). A reviewer pointed out that the previous headline presented the ordering "
             "that survives the shared-noise test as a partial answer to the referee, and the data do not support "
             "that reading. The constant-partial null is not the referee's null. SAT_AVG and ADM_RATE are "
             "institution-level proxies for program-level student quality, and with such imperfect proxies the "
             "referee's hypothesis itself predicts a positive, baseline-correlated partial. Changes:")
    R.append("- The headline is restated. Most of the level is not separable from selectivity, and the surviving ordering "
             "cannot be distinguished from field-specific pricing of selectivity with public data. It is no longer "
             "described as surviving adjustment in the sense of answering the objection.")
    R.append(f"- Added a control-precision gradient with three new partial specs on the SAT sample: Pell/control/state "
             f"level; + ADM_RATE; broad + institution-wide earnings. They are in the joint bootstrap and the per-field CSV "
             f"(`rho_pcs`, `rho_pcs_adm`, `rho_full_ie`). Mean partial "
             f"{' → '.join(_f(grad[c]['mean']) for c in GRAD)}; Spearman with the baseline "
             f"{' → '.join(_f(grad[c]['s_base']) for c in GRAD)} ({grad[GRAD[0]]['k']} fields).")
    R.append(f"- Added a split-half cross-field regression (outcome on half B ~ baseline + selectivity pricing on half A) "
             f"with one-sided Freedman–Lane permutation p, run on the same halvings as the existing cross-fit. Broad "
             f"partial: pricing {_f(rb['rank']['b_price'])} ({_p(rb['rank']['p_price'])}), baseline "
             f"{_f(rb['rank']['b_base'])} ({_p(rb['rank']['p_base'])}). It also has a calibration row (outcome = "
             f"baseline itself). That row shows the plain regression already gives pricing {_f(rc['rank']['b_price'])} when "
             "only the baseline matters, so only the shift away from the calibration row is read as evidence. It also "
             "has an ADM-based pricing measure, which does not favour pricing, and an errors-in-variables version, "
             "which points the same way but is unstable.")
    R.append(f"- Added FE spec (c) to the FE-design cross-fit ({_f(fsp['c']['cross_mean'])}, {_p(fsp['c']['p_perm'])}), "
             f"and moved the FE (b)/(c) results (cross-fit {_f(fsp['b']['cross_mean'])}, {_p(fsp['b']['p_perm'])}; "
             f"contrast {_f(FE['c']['contrast'], 3)}, SE {FE['c']['contrast_se']:.3f}) next to the ordering claim.")
    R.append("- Added a caveat on residual confounding from imperfect selectivity measures, and one on the pricing "
             "measures not being pure prices.")
    R.append(f"- Unchanged: every number reported in revision 2. The bootstrap draws, split seeds and point estimates are "
             f"the same (e.g. broad Spearman {_f(ob['obs'])}, null mean {_f(ob['nullB_mean'])}, {_p(ob['nullB_p'])}).")
    R.append("")
    R.append("Revision 2 (previous). It replaced a bootstrap percentile CI of Spearman(baseline, partial), which kept "
             "the shared estimation noise in every draw, with the shared-noise null, the split-half cross-fit and the "
             "moment decomposition. It also re-tested the strict spec's ordering, replaced the same-sample "
             "Spearman(β_f, ρ_f) with a cross-fit, and raised the bootstrap from 500 to 1000 draws.")
    return "\n".join(R)


def _print(ctx):
    summ, contr, brand, fe_summ, FE = (ctx[k] for k in ("summ", "contr", "brand", "fe_summ", "FE"))
    print("\n=== cross-field summaries (all fields) ===")
    for col in ["rho_raw", "rho_raw_sat", "rho_sel", "rho_full_stlev", "rho_full_state", "rho_raw_adm",
                "rho_adm_stlev", "rho_adm_state", "rho_sat_earn", "rho_adm_earn", "rho_ie_earn",
                "rho_all_common", "rho_pell", "rho_nopell"]:
        s = summ[(col, False)]
        nf = s["nn_full"]
        print(f"  {col:15s} k={s['k']:2d} mean={s['mean']:+.3f} (same {s['mean_same']:+.3f}) sd={s['sd']:.3f} "
              f"rmsSE={s['rms_se']:.3f} tau={s['tau']:.3f} Qp={_fmt_p(s['p'])} S_full={s['s_full']:+.2f}"
              + (f" null={nf['nullB_mean']:+.2f} p={nf['nullB_p']:.4f}" if nf else ""))
    print("\n=== ordering: shared-noise null ===")
    for (spec, sname), o in ctx["order"].items():
        if o.get("k", 0) >= 5:
            print(f"  {spec:10s} {sname:8s} k={o['k']:2d} obs={o['obs']:+.3f} nullA={o['nullA_mean']:+.3f} "
                  f"pA={o['nullA_p']:.4f} nullB={o['nullB_mean']:+.3f} pB={o['nullB_p']:.4f} pC={o['nullC_p']:.4f} lam={o['lamC']:.2f} share={o['share']:.2f} "
                  f"tvy={o['tvy']:+.4f} r_true={o['r_true']:+.2f}")
    print("\n=== ordering: split-half ===")
    for spec, q in ctx["split"].items():
        print(f"  {spec:10s} k={q['k']} cross={q['cross_mean']:+.3f} min={q['cross_min']:+.3f} le0={q['cross_le0']:.3f} "
              f"p={q['p_perm']:.4f} same={q['same_mean']:+.3f} relR={q['relR']:+.3f} relQ={q['relQ']:+.3f} "
              f"dis={q['dis']:+.3f} undef={q['undef']:.3f}")
    print("\n=== named contrasts ===")
    for col, c in contr.items():
        print(f"  {col:15s} {c['est']:+.3f} [{c['lo95']:+.3f},{c['hi95']:+.3f}] hi={c['hi']} lo={c['lo']}")
    print("\n=== brand ===")
    for k, b in brand.items():
        print(" ", k, {kk: (np.round(v, 3) if isinstance(v, (float, np.floating)) else v) for kk, v in b.items()})
    print("\n=== FE ===")
    for tag in ["a_ols", "a", "a_bs", "b", "c"]:
        e, s = FE[tag], fe_summ[tag]
        print(f"  {tag:5s} cells={e['n_cells']} inst={e['n_inst']} k={s['k']} pooled={e['pooled']:+.3f}({e['pooled_se']:.3f}) "
              f"mean={s['mean']:+.3f} sd={s['sd']:.3f} rmsSE={s['rms_se']:.3f} tau={s['tau']:.3f} "
              f"wald={e['wald']:.1f}({e['wald_df']}) p={_fmt_p(e['wald_p'])} S={s['s_raw']:+.2f} "
              f"contrast={e['contrast']:+.3f}({e['contrast_se']:.3f}) +sig={s['n_pos_sig']} -sig={s['n_neg_sig']}")
    for tag, q in ctx["fe_split"].items():
        print(f"  split {tag}: k~{q['k_mean']:.1f} cross={q['cross_mean']:+.3f} le0={q['cross_le0']:.3f} "
              f"p={q['p_perm']:.4f} same={q['same_mean']:+.3f}")
    print("\n=== control-precision gradient ===")
    for c in GRAD:
        g = ctx["grad"][c]; r = g["reg"]
        print(f"  {c:15s} k={g['k']} mean={g['mean']:+.3f} S_base={g['s_base']:+.3f} "
              + (f"null={g['nn']['nullB_mean']:+.3f} p={g['nn']['nullB_p']:.4f} " if g["nn"] else "")
              + f"| large k={r['rank']['k']} cf={r['cf_base']:+.3f} rel={r['rel_y']:.3f} t1y={r['t1y']:+.3f} "
              f"t2y={r['t2y']:+.3f}")


if __name__ == "__main__":
    main()
