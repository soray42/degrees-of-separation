"""Earnings-noise reliability of within-field coupling: how much of the field heterogeneity (I2 = 0.69),
the field ordering and the field classification survives once the sampling noise of each program's
earnings median is propagated.

REFEREE PROBLEM. Q and I2 (scripts/57) treat each coupling's sampling variance as 1/(n_f - 3), which
covers the sampling of institutions only. The program earnings medians are themselves estimates from
finite graduate cohorts (Scorecard: as few as 16 earners). Noise in them (i) attenuates rho_f toward 0 by a
field-specific amount (fields with small cohorts or wide within-program spread look less coupled), which can
create spurious between-field heterogeneity and reorder fields, and (ii) adds variance the 1/(n-3) model may
not carry. This script estimates the noise, the per-field reliability of the earnings ranks, disattenuates
coupling and refits the heterogeneity models of scripts/57 with the noise propagated.

PRE-SPECIFIED (fixed in this docstring before any number of this script was computed)
  Data / sample. scripts/57's 57 non-degenerate fields (outputs/expanded66_gap_map.csv; Scorecard FoS
    EARN_MDN_4YR at credential level 3, Wapman field prestige), rebuilt program by program and asserted to
    reproduce n_f and rho_f. z_f = atanh(rho_f).
  Program SE (sampling SE of a median). SE(median) ~ 1.2533 sigma / sqrt(n) (the large-sample SE of a median
    under a normal distribution, sqrt(pi/2) sigma / sqrt(n)), sigma = IQR / 1.349 (the normal IQR).
    PSEO rows: released p25/p75 and y{h}_grads_earn of the row itself. Scorecard rows (no spread released):
    n = EARN_COUNT_WNE_4YR, sigma = d_f x median / 1.349, d_f = the field's median relative IQR
    (p75 - p25)/p50 over PSEO V4.14.1 bachelor's institution x CIP-4 rows at y5 (pooled cohorts), with the
    CIP-2 pool (then all rows) when a field has fewer than 10 such rows. A program (institution x field) that
    pools several CIP-4 rows has earnings = count-weighted mean of the row medians (as src.load_er does), so
    SE^2 = sum w_j^2 SE_j^2 / (sum w_j)^2. Relative SE r_i = SE_i / y_i is the SE on the log scale.
  Reliability. R_f = 1 - mean_i(r_i^2) / Var_i(log y_i) over the field's matched institutions (the share of
    the cross-institution variance of log earnings that is not sampling noise; Spearman coupling is invariant
    to the log). Rank reliability index = (6/pi) asin(sqrt(R_f)/2), the expected Spearman correlation between
    observed and noise-free earnings under bivariate normality.
  Disattenuation. Spearman's correction on the normal-score scale: r = 2 sin(pi rho/6), r_c = r / sqrt(R_f),
    rho_c = (6/pi) asin(r_c/2) (|r_c| capped at 0.999, flagged).
  Noise propagation. 2000 draws: every program median perturbed by its SE (y* = y + SE x N(0,1)), rho_f
    recomputed. v_noise_f = variance of atanh(rho*_f) over draws. In each draw the disattenuated coupling is
    recomputed with the reliability of the twice-noisy data, (V - m)/(V + m).
  Variants. V0 original (z, 1/(n-3)); V1 noise-aware variance (z, 1/(n-3) + v_noise); V2 noise-corrected
    = PRIMARY (z_c = atanh(rho_c), v_c = J^2 [1/(n-3) + v_noise] + K^2 Var(R_f), J = dz_c/dz, K = dz_c/dR,
    Var(R_f) from a bootstrap over the field's institutions and over the PSEO programs that give d_f).
    V2a disattenuation only (v = J^2/(n-3) + K^2 Var R) splits the two mechanisms.
  Recomputed: Cochran Q, I2 (Higgins-Thompson, as scripts/57), field-only REML tau2, and the three-level
    model of scripts/57 (tau2_cluster, tau2_field, ICC_true with profile CIs, LR_cluster with mixture p and
    the size-preserving label permutation p on scripts/57's first 2000 permutations), unadjusted and with
    log n (scripts/57's primary adjustment).
  Inference standard. For statistics over fields (I2, ratios, Spearman ordering, mean coupling) the two-way
    (field, institution) cluster variance of scripts/59 (twoway(), imported): V_field (jackknife over fields)
    + V_institution (one institution multinomial draw shared by all fields, fields fixed) - V_field x inst
    (independent draw per field), 1000 draws each, normal 95% CI. Q-profile CIs for tau2 / I2 beside it.
  T1 (how much of I2 = 0.69 remains): ratio = I2(V2) / I2(V0), 57 fields, independent errors, two-way CI.
    Reading: ratio >= 0.75 "most of I2 remains"; 0.25 <= ratio < 0.75 "part remains"; < 0.25 "most of I2 is
    earnings noise". Cochran Q of V2 reported with its p.
  T2 (does the field ordering survive): Spearman(z_c, z) across the 57 fields. "Survives" if the estimate is
    >= 0.80 and its two-way 95% lower bound is >= 0.80.
  T3 (does any field's classification flip): fields classified by scripts/52's rule (Scorecard 4YR baseline
    from compute_gap_map, B=250, seed 52; reliable_flag and n >= 15; integrated rho >= 0.45, decoupled
    < 0.25, middle otherwise). A field flips if its disattenuated coupling falls in another band; each flip is
    reported with the share of the 2000 noise draws in which it lands in the new band ("robust" if >= 95%).
  AMENDMENT (made after the per-field reliabilities were computed and before any T1-T3 statistic was
  computed). The z-scale plug-in correction above is undefined for one of the 57 fields (biomedical
  engineering: r / sqrt(R) = 1.01 > 1); the cap would give it z_c = 3.8 with J = K = 0, i.e. a sampling variance
  of about 0 and an unbounded weight. The PRIMARY noise-corrected variant V2 therefore uses the Hunter-Schmidt
  linear correction on the normal-score correlation scale r = 2 sin(pi rho / 6), with no cap:
      y_c = r / sqrt(R_f),   v_c = G^2 [1/(n-3) + v_noise] / R_f + r^2 / (4 R_f^3) Var(R_f),   G = dr/dz,
  (a corrected value above 1 is a noisy estimate of a true value <= 1 and is kept as such). V0 and V1 are refit
  on the same scale (V0r: G^2/(n-3); V1r: G^2 [1/(n-3) + v_noise]) so that T1 compares like with like:
  T1 ratio = I2(V2) / I2(V0r). scripts/57's z-scale V0 (I2 = 0.694) is reproduced beside them, and the z-scale
  V2 without the undefined field is a sensitivity. T2 = Spearman(y_c, r) (= Spearman(y_c, z)). T3 uses the
  same rho bands mapped to r (0.45 -> 0.4669, 0.25 -> 0.2611). In the two-way institution draws R_f is held at
  its estimate (its uncertainty enters through v_c); separate draws from the per-field reliability bootstrap
  show how much the reliability uncertainty moves I2 and the ordering.
  Everything else is EXPLORATORY and labelled so: the national-quartile, lognormal and program-specific-PSEO
  SE variants, SIMEX, correlated errors, the reliable-20 subset, and the PSEO V4.14.1 fixed-cohort career-time
  panel (does the +0.03/yr rise survive disattenuation?).
  LATER ADDITIONS (2026-09-25, made after the three-level fits of V0-V2 had been printed; exploratory or
  diagnostic, none changes T1-T3): (i) |rho| capped at 0.999 inside the two-way institution replicates (a
  resampled small field can reach rho = 1, z infinite), as scripts/57's joint bootstrap does; (ii) power of the
  cluster LR test for V0r and V2 by scripts/57's parametric bootstrap; (iii) Spearman associations of R_f with
  rho_f, n_f and cohort size, and the largest rank moves; (iv) PSEO panel: 3 cells have an observed R <= 0 and
  the first build's rule (every resampled cell keeps R > 0.05) is met by almost no institution draw, so the
  correction uses the cells with observed R >= R_GATE = 0.30 at every horizon, resampled R floored at
  R_FLOOR = 0.10, and only degenerate draws are redrawn (scripts/59's rule).
  RESTART ADDITIONS (2026-09-25, after an interrupted first build had printed every T1-T3 number; none changes
  the definition of T1-T3): (v) exploratory SE variants dp_mid / dp_hi that add the differential-privacy noise
  of the Scorecard medians documented in FieldOfStudyDataDocumentation.pdf (Version September 2025, printed page
  7, Exhibit 2: share of released medians by |perturbed - unperturbed| / unperturbed; the text calls the second
  column four-year, the exhibit header says '5 years'): relative SD sqrt(sum p_k c_k^2) with c_k the bucket
  midpoints (dp_mid) or upper edges (dp_hi), added in quadrature to each CIP-4 row's sampling SE; (vi) the
  Q-profile CI of I2 is printed beside its matching point, the model-based I2 = tau2_REML / (tau2_REML + s~^2),
  and the Higgins-Thompson I2 = (Q - df)/Q keeps the two-way CI; (vii) the 'R needed for the next band' of a
  field with r <= 0 is undefined (disattenuation cannot move it up); (viii) scripts/57's joint bootstrap runs
  under one BLAS thread (as in scripts/57); (ix) the report states the level shift of the noise-draw and
  reliability-bootstrap distributions (each draw adds a second dose of noise), which are not CIs of the point;
  (x) the 'least stable' T3 fields are ranked over all classified fields by the share of noise draws with a band
  change, and the largest such share among integrated fields is printed.

Descriptive, not causal. Public data only, all already on disk (no download). Seeded (SEED=73; one
SeedSequence child per replicate, so results do not depend on the worker count); outputs byte-identical on
re-run. BLAS: 8 threads in the parent per the resource rule, but every model fit runs under
threadpool_limits(1) so parent and worker fits are bit-identical.
Run:     PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/73_earnings_reliability.py
Outputs: data/interim/earnings_reliability.csv (every number), data/interim/earnings_reliability_fields.csv
         (per field), EARNINGS_RELIABILITY_RESULT.md (repo root, local/gitignored).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "8"
import sys
sys.dont_write_bytecode = True
import gc
import hashlib
import importlib.util
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2, rankdata, spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.load_ar import load_ar_wapman                                    # noqa: E402
from src.load_er import load_er_scorecard, SCORECARD_FOS, PSEO_DEGREE_LEVEL  # noqa: E402
from src.gap import compute_gap_map                                       # noqa: E402
from src.crosswalks import fields as F                                    # noqa: E402
from src.crosswalks.institutions import normalize_institution_name        # noqa: E402


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s57 = _load("s57", "57_icc_threelevel.py")     # three-level REML, data rebuild (not edited)
s59 = _load("s59", "59_pseo_refresh.py")       # two-way cluster variance, V4.14.1 plumbing (not edited)
s52 = s59.s52                                  # fixed-cohort constants + classification (not edited)
FIELDS66, LAB, CIP2, C2NAME = s57.FIELDS66, s57.LAB, s57.CIP2, s57.C2NAME
HZ, COHORTS, W_SLOPE, NMIN, YRS = s52.HZ, s52.COHORTS, s52.W_SLOPE, s52.NMIN, s52.YRS
twoway = s59.twoway

SEED = 73
N_DRAW = 2000            # noise-perturbation draws (pre-specified)
N_BOOT_R = 2000          # per-field bootstrap of the reliability (institutions + PSEO programs)
N_TW = 1000              # institution draws of each kind for the two-way variance (as scripts/59)
N_PERM = 2000            # label permutations (scripts/57's first 2000)
N_PB = 1000              # parametric-bootstrap null draws for the power of the cluster test
N_POW = 500              # draws per power scenario
POWER_ICC = (0.25, 0.50, 0.75)
SIMEX_L = (0.5, 1.0, 1.5, 2.0)
MED_SE = float(np.sqrt(np.pi / 2))       # 1.2533: SE(median) = MED_SE sigma / sqrt(n) under normality
IQR_Z = 1.3489795003921634               # IQR of a standard normal
MIN_PSEO_ROWS = 10       # a field's own PSEO dispersion needs at least this many y5 pooled rows
CAP = 0.999              # |r_c| cap in the disattenuation
R_GATE = 0.30            # PSEO panel (exploratory): a cell enters the correction if its observed R >= this at y1, y5, y10
R_FLOOR = 0.10           # PSEO panel: resampled reliabilities of gated cells floored here (count reported)
INT_T, DEC_T = 0.45, 0.25                # scripts/52 bands
# Scorecard FoS documentation (Version September 2025, printed p. 7, Exhibit 2), four-year medians: share of released
# medians whose differential-privacy perturbation is < 1%, 1-4% and 5-9% of the unperturbed value (exploratory use)
DP_SHARES = np.array([0.64, 0.31, 0.05])
DP_MID = np.array([0.005, 0.03, 0.075])     # bucket midpoints ([0,1), [1,5), [5,10) percent)
DP_HI = np.array([0.01, 0.05, 0.10])        # bucket upper edges (upper bound of the second moment)
DP_SD = {"dp_mid": float(np.sqrt((DP_SHARES * DP_MID ** 2).sum())),
         "dp_hi": float(np.sqrt((DP_SHARES * DP_HI ** 2).sum()))}
NPROC = int(os.environ.get("ER73_NPROC", "4"))
Z975 = s59.Z975

OUT_CSV = ROOT / "data" / "interim" / "earnings_reliability.csv"
OUT_FIELDS = ROOT / "data" / "interim" / "earnings_reliability_fields.csv"
OUT_MD = ROOT / "EARNINGS_RELIABILITY_RESULT.md"
REF57 = ROOT / "data" / "interim" / "icc_threelevel.csv"
REF59 = ROOT / "data" / "interim" / "pseo_refresh.csv"

ROWS: list[dict] = []


def rec(section, stat, estimate, variant="", sample="", lo=np.nan, hi=np.nan, p=np.nan, k=np.nan,
        field="", note=""):
    ROWS.append(dict(section=section, stat=stat, variant=variant, sample=sample, field=field,
                     estimate=estimate, ci_lo=lo, ci_hi=hi, p_value=p, k=k, note=note))


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def streams(names):
    ss = np.random.SeedSequence(SEED)
    return dict(zip(names, ss.spawn(len(names))))


SS = streams(["noise", "simex", "bootR", "tw_sc_shared", "tw_sc_indep", "tw_ps_shared", "tw_ps_indep", "power"])


# ------------------------------------------------------------------ deterministic parallel map ----------
_W: dict = {}


def _init_worker():
    threadpool_limits(1)


def _pw(i):
    return _W["fn"](_W["items"][i])


def pmap(fn, items, nproc=None):
    """Order-preserving map over forked workers (concurrent.futures, fork context). Random draws are
    made inside fn from a per-item SeedSequence child, so the result does not depend on NPROC."""
    items = list(items)
    nproc = NPROC if nproc is None else nproc
    if nproc <= 1 or len(items) < 16:
        with threadpool_limits(1):
            return [fn(x) for x in items]
    _W["fn"], _W["items"] = fn, items
    try:
        with ProcessPoolExecutor(max_workers=nproc, mp_context=mp.get_context("fork"),
                                 initializer=_init_worker) as ex:
            out = list(ex.map(_pw, range(len(items)), chunksize=max(1, len(items) // (nproc * 8))))
    finally:
        _W.clear()
    return out


# ------------------------------------------------------------------ small helpers ------------------------
def corr_rows(x, y):
    xc = x - x.mean(-1, keepdims=True); yc = y - y.mean(-1, keepdims=True)
    den = np.sqrt((xc ** 2).sum(-1) * (yc ** 2).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (xc * yc).sum(-1) / den


def disatt(rho, R):
    """Spearman's correction on the normal-score scale; returns (rho_c, capped)."""
    rho = np.asarray(rho, float); R = np.asarray(R, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rc = 2 * np.sin(np.pi * rho / 6) / np.sqrt(R)
    cap = np.abs(rc) >= CAP
    rc = np.clip(rc, -CAP, CAP)
    return (6 / np.pi) * np.arcsin(rc / 2), cap


def zc_of(z, R):
    return np.arctanh(disatt(np.tanh(z), R)[0])


def rank_rel(R):
    return (6 / np.pi) * np.arcsin(np.sqrt(np.clip(R, 0, 1)) / 2)


def I2Q(z, v):
    """Higgins-Thompson I2 = (Q - df)/Q with fixed-effect weights (= scripts/57 cochran() for diagonal V)."""
    w = 1 / v; mu = (w * z).sum() / w.sum(); Q = float((w * (z - mu) ** 2).sum()); df = len(z) - 1
    return max(0.0, (Q - df) / Q), Q


def qprofile(z, v, Ve=None):
    """Q-profile 95% CI for tau2 (Viechtbauer 2007); generalized Q with a full V_e if given."""
    k = len(z); df = k - 1
    Vb = np.diag(v) if Ve is None else Ve

    def Qg(t):
        Vi = np.linalg.inv(Vb + t * np.eye(k)); one = np.ones(k)
        mu = (one @ Vi @ z) / (one @ Vi @ one); r = z - mu
        return float(r @ Vi @ r)
    lo_t, hi_t = chi2.ppf(0.975, df), chi2.ppf(0.025, df)
    lo = 0.0 if Qg(0.0) < lo_t else brentq(lambda t: Qg(t) - lo_t, 0.0, 20.0, xtol=1e-12)
    hi = 0.0 if Qg(0.0) < hi_t else brentq(lambda t: Qg(t) - hi_t, 0.0, 20.0, xtol=1e-12)
    return lo, hi


def v_typ(v):
    w = 1 / v; k = len(v)
    return (k - 1) * w.sum() / (w.sum() ** 2 - (w ** 2).sum())


def jackknife(fn, K):
    th = np.array([fn(np.delete(np.arange(K), i)) for i in range(K)])
    return float((K - 1) / K * ((th - th.mean(0)) ** 2).sum(0)) if th.ndim == 1 else \
        (K - 1) / K * ((th - th.mean(0)) ** 2).sum(0)


# ================================================================== A. PSEO rows (V4.14.1) ===============
def pseo_rows() -> pd.DataFrame:
    """Bachelor's, institution-level, CIP-4 rows of V4.14.1, long by horizon, exactly the rows
    src.load_er.load_er_pseo keeps (status_y{h}_earnings == '1', p50 present, institution with inst_key)."""
    cols = ["inst_level", "institution", "degree_level", "cip_level", "cipcode", "grad_cohort"]
    for h in HZ:
        cols += [f"{h}_p25_earnings", f"{h}_p50_earnings", f"{h}_p75_earnings", f"{h}_grads_earn",
                 f"status_{h}_earnings"]
    fmap = F.cip4_to_field_key(FIELDS66)
    out = []
    for ch in pd.read_csv(s59.NEW_EARN, dtype=str, usecols=cols, chunksize=200_000):
        ch = ch[(ch.inst_level == "I") & (ch.cip_level == "4")
                & (ch.degree_level == PSEO_DEGREE_LEVEL["undergrad"])]
        cip4 = ch.cipcode.astype(str).str.replace(".", "", regex=False).str.zfill(4)
        ch = ch.assign(cip4=cip4, field=cip4.map(fmap))
        ch = ch[ch.field.notna()]
        for h in HZ:
            x = ch[ch[f"status_{h}_earnings"] == "1"]
            y = pd.DataFrame(dict(field=x.field.values, institution=x.institution.values, cip4=x.cip4.values,
                                  grad_cohort=x.grad_cohort.values, horizon=h,
                                  p25=pd.to_numeric(x[f"{h}_p25_earnings"], errors="coerce").values,
                                  p50=pd.to_numeric(x[f"{h}_p50_earnings"], errors="coerce").values,
                                  p75=pd.to_numeric(x[f"{h}_p75_earnings"], errors="coerce").values,
                                  n=pd.to_numeric(x[f"{h}_grads_earn"], errors="coerce").values))
            out.append(y.dropna(subset=["p50"]))
    df = pd.concat(out, ignore_index=True)
    inst = s59.institutions("new").drop_duplicates("institution")[["institution", "inst_key"]]
    df = df.merge(inst, on="institution", how="left")
    df = df[df.inst_key.notna() & (df.inst_key != "")].reset_index(drop=True)
    df["cip2"] = df.cip4.str[:2]
    df["w"] = df.n.fillna(1.0).clip(lower=1.0)
    ok = df.p25.notna() & df.p75.notna() & (df.p50 > 0) & (df.p75 >= df.p25) & (df.n > 0)
    df["q_ok"] = ok
    df["rel_iqr"] = np.where(ok, (df.p75 - df.p25) / df.p50, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["log_iqr"] = np.where(ok & (df.p25 > 0), np.log(df.p75 / df.p25), np.nan)
    return df


def field_dispersion(ps: pd.DataFrame) -> pd.DataFrame:
    """Field-specific within-program spread from PSEO y5 pooled-cohort rows: median relative IQR and
    median log IQR; fallback to the CIP-2 pool, then all rows, when < MIN_PSEO_ROWS rows."""
    x = ps[(ps.horizon == "y5") & (ps.grad_cohort == "0000") & ps.q_ok]
    allf = sorted({f["key"] for f in FIELDS66})
    rows = []
    for f in allf:
        own = x[x.field == f]
        if len(own) >= MIN_PSEO_ROWS:
            src, pool = "field", own
        else:
            c2 = x[x.cip4.str[:2] == CIP2[f]]
            src, pool = ("cip2", c2) if len(c2) >= MIN_PSEO_ROWS else ("all", x)
        rows.append(dict(field=f, d_src=src, d_rows=len(pool), d_rows_own=len(own),
                         d_insts=pool.inst_key.nunique(), d_rel=float(pool.rel_iqr.median()),
                         d_log=float(pool.log_iqr.dropna().median()) / IQR_Z,
                         d_rel_p25=float(pool.rel_iqr.quantile(.25)), d_rel_p75=float(pool.rel_iqr.quantile(.75))))
    return pd.DataFrame(rows).set_index("field")


def dispersion_pool(ps, disp, f):
    x = ps[(ps.horizon == "y5") & (ps.grad_cohort == "0000") & ps.q_ok]
    s = disp.loc[f, "d_src"]
    pool = x[x.field == f] if s == "field" else x[x.cip4.str[:2] == CIP2[f]] if s == "cip2" else x
    return pool[["inst_key", "rel_iqr"]].reset_index(drop=True)


def pseo_prog_disp(ps: pd.DataFrame) -> pd.DataFrame:
    """Program-specific relative IQR (y5 pooled), grads-weighted over the program's CIP-4 rows."""
    x = ps[(ps.horizon == "y5") & (ps.grad_cohort == "0000") & ps.q_ok].copy()
    x["wd"] = x.rel_iqr * x.w
    g = x.groupby(["field", "inst_key"], as_index=False)[["wd", "w"]].sum()
    g["d_prog"] = g.wd / g.w
    return g[["field", "inst_key", "d_prog"]]


def pseo_program_se(ps: pd.DataFrame, disp: pd.DataFrame) -> pd.DataFrame:
    """Row SE from the row's own p25/p75/n; aggregated to (field, inst_key, grad_cohort, horizon) with the
    same weights as load_er_pseo."""
    x = ps.copy()
    sig = np.where(x.q_ok, (x.p75 - x.p25) / IQR_Z, x.field.map(disp.d_rel).to_numpy() * x.p50 / IQR_Z)
    x["own_q"] = x.q_ok
    with np.errstate(invalid="ignore", divide="ignore"):
        x["se"] = MED_SE * sig / np.sqrt(x.n.where(x.n > 0))
    x["we"] = x.p50 * x.w; x["wse2"] = (x.w * x.se) ** 2
    g = x.groupby(["field", "inst_key", "grad_cohort", "horizon"], as_index=False).agg(
        we=("we", "sum"), w=("w", "sum"), wse2=("wse2", lambda s: s.sum(min_count=len(s))),
        n_rows=("p50", "size"), n_own_q=("own_q", "sum"), n_earn=("n", "sum"))
    g["earnings"] = g.we / g.w
    g["se"] = np.sqrt(g.wse2) / g.w
    return g.drop(columns=["we", "wse2"])


# ================================================================== B. Scorecard programs ================
def scorecard_programs(disp: pd.DataFrame, prog: pd.DataFrame) -> pd.DataFrame:
    use = ["UNITID", "INSTNM", "CIPCODE", "CREDLEV", "EARN_MDN_4YR", "EARN_COUNT_WNE_4YR",
           "EARN_MDN_4YR_NAT", "EARN_P25_4YR_NAT", "EARN_P75_4YR_NAT"]
    df = pd.read_csv(SCORECARD_FOS, usecols=use, dtype=str)
    df = df[df.CREDLEV == "3"].copy()
    df["cip4"] = df.CIPCODE.astype(str).str.zfill(4)
    df["field"] = df.cip4.map(F.cip4_to_field_key(FIELDS66))
    df = df[df.field.notna()].copy()
    df["earn"] = pd.to_numeric(df.EARN_MDN_4YR, errors="coerce")
    df["cohort"] = pd.to_numeric(df.EARN_COUNT_WNE_4YR, errors="coerce")
    df = df.dropna(subset=["earn"])
    df["inst_key"] = df.INSTNM.map(normalize_institution_name)
    df = df[df.inst_key != ""].copy()
    assert df.cohort.notna().all() and (df.cohort > 0).all(), "released median without an earnings count"
    df["w"] = df.cohort.fillna(1.0).clip(lower=1.0)
    for c in ("EARN_MDN_4YR_NAT", "EARN_P25_4YR_NAT", "EARN_P75_4YR_NAT"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    d_rel = df.field.map(disp.d_rel).to_numpy()
    d_log = df.field.map(disp.d_log).to_numpy()
    nat = ((df.EARN_P75_4YR_NAT - df.EARN_P25_4YR_NAT) / df.EARN_MDN_4YR_NAT).to_numpy()
    df = df.merge(prog, on=["field", "inst_key"], how="left")
    d_prog = np.where(df.d_prog.notna(), df.d_prog, d_rel)
    sq = np.sqrt(df.cohort.to_numpy())
    e = df.earn.to_numpy()
    df["se_main"] = MED_SE * d_rel * e / IQR_Z / sq          # primary
    df["se_lognormal"] = MED_SE * d_log * e / sq             # lognormal density at the median
    df["se_national"] = MED_SE * nat * e / IQR_Z / sq        # national CIP-4 IQR (includes between-program spread)
    df["se_pseoprog"] = MED_SE * d_prog * e / IQR_Z / sq     # program's own PSEO spread where available
    for v in ("dp_mid", "dp_hi"):                            # + documented differential-privacy noise of the median
        df[f"se_{v}"] = np.sqrt(df.se_main ** 2 + (DP_SD[v] * e) ** 2)
    df["has_prog"] = df.d_prog.notna()
    df["we"] = df.earn * df.w
    agg = dict(institution_id=("UNITID", "first"), we=("we", "sum"), wsum=("w", "sum"),
               cohort_n=("cohort", "sum"), n_rows=("earn", "size"), has_prog=("has_prog", "max"),
               nat_missing=("EARN_P25_4YR_NAT", lambda s: int(s.isna().sum())))
    for v in SE_VARIANTS:
        df[f"w2{v}"] = (df.w * df[f"se_{v}"]) ** 2
        agg[f"w2{v}"] = (f"w2{v}", lambda s: s.sum(min_count=len(s)))
    g = df.groupby(["field", "inst_key"]).agg(**agg).reset_index()
    g["earnings"] = g.we / g.wsum
    for v in SE_VARIANTS:
        g[f"se_{v}"] = np.sqrt(g[f"w2{v}"]) / g.wsum
    # reproduction check against the loader every other script uses
    ref = load_er_scorecard(fields=FIELDS66)
    chk = g.merge(ref[["field", "inst_key", "earnings", "cohort_n"]], on=["field", "inst_key"], how="outer",
                  suffixes=("", "_ref"), indicator=True)
    assert (chk._merge == "both").all(), "program set differs from load_er_scorecard"
    assert np.allclose(chk.earnings, chk.earnings_ref, rtol=1e-12, atol=0), "earnings differ from load_er_scorecard"
    return g.drop(columns=["we", "wsum"] + [f"w2{v}" for v in SE_VARIANTS])


SE_VARIANTS = ("main", "lognormal", "national", "pseoprog", "dp_mid", "dp_hi")
EXPL_VARIANTS = (("lognormal", "lognormal SE"), ("national", "national CIP-4 IQR"),
                 ("pseoprog", "program's own PSEO IQR"), ("dp_mid", "+ DP noise (bucket midpoints)"),
                 ("dp_hi", "+ DP noise (bucket upper edges)"))


# ================================================================== C. matched fields ====================
def matched(ar: pd.DataFrame, sc: pd.DataFrame) -> dict:
    out = {}
    for f in sorted(set(ar.field) & set(sc.field)):
        a = ar[ar.field == f][["inst_key", "prestige_score"]]
        m = a.merge(sc[sc.field == f], on="inst_key", how="inner").dropna(subset=["prestige_score", "earnings"])
        if len(m) < 8:
            continue
        m = m.sort_values("inst_key").reset_index(drop=True)
        out[f] = dict(P=m.prestige_score.to_numpy(float), y=m.earnings.to_numpy(float),
                      keys=m.inst_key.to_numpy(str), cohort=m.cohort_n.to_numpy(float),
                      **{f"se_{v}": m[f"se_{v}"].to_numpy(float) for v in SE_VARIANTS})
    return out


def rel_parts(y, se):
    ly = np.log(y); V = float(np.var(ly, ddof=1)); m = float(np.mean((se / y) ** 2))
    return V, m, 1.0 - m / V


def field_core(M: dict, var="main") -> pd.DataFrame:
    rows = []
    for f, d in M.items():
        P, y, se = d["P"], d["y"], d[f"se_{var}"]
        rho = float(spearmanr(P, y)[0]); V, m, R = rel_parts(y, se)
        rows.append(dict(field=f, n=len(y), rho=rho, z=np.arctanh(rho), V=V, m=m, R=R,
                         rank_rel=float(rank_rel(R)), med_cohort=float(np.median(d["cohort"])),
                         min_cohort=float(np.min(d["cohort"])), med_relse=float(np.median(se / y))))
    return pd.DataFrame(rows).set_index("field")


def noise_draws(M: dict, fields: list, var: str, stream, lam=1.0):
    """N_DRAW perturbation draws; one SeedSequence child per draw generating the standard normals of every
    program of every field (fixed field order). Returns z* (N_DRAW, K)."""
    n = np.array([len(M[f]["y"]) for f in fields]); off = np.concatenate([[0], np.cumsum(n)])
    kids = stream.spawn(N_DRAW)
    E = np.empty((N_DRAW, off[-1]))
    for i, k in enumerate(kids):
        E[i] = np.random.default_rng(k).standard_normal(off[-1])
    Z = np.empty((N_DRAW, len(fields)))
    for j, f in enumerate(fields):
        d = M[f]; y, se = d["y"], d[f"se_{var}"]
        Ys = y[None, :] + np.sqrt(lam) * se[None, :] * E[:, off[j]:off[j + 1]]
        Z[:, j] = np.arctanh(corr_rows(rankdata(d["P"])[None, :], rankdata(Ys, axis=1)))
    del E
    return Z




def boot_R(M, fields, pools, var="main"):
    """Per-field bootstrap of the reliability: institutions resampled (V, m, rho) and, for the primary SE,
    the PSEO programs behind d_f resampled by institution (m scales with d_f^2). One SeedSequence child per
    field. Returns per field: R replicates, Var(R), percentile CI, share <= 0, bootstrap variance of y_c."""
    kids = dict(zip(fields, SS["bootR"].spawn(len(fields))))
    out = {}
    for f in fields:
        d = M[f]; rng = np.random.default_rng(kids[f])
        P, y, se = d["P"], d["y"], d[f"se_{var}"]; n = len(y)
        vals, ik = pools[f]
        ui, inv = np.unique(ik, return_inverse=True)
        order = np.argsort(inv, kind="stable"); starts = np.searchsorted(inv[order], np.arange(len(ui)))
        ends = np.append(starts[1:], len(inv))
        d0 = float(np.median(vals))
        idx = rng.integers(n, size=(N_BOOT_R, n))
        pick = rng.integers(len(ui), size=(N_BOOT_R, len(ui)))
        ly, r2 = np.log(y), (se / y) ** 2
        Vb = np.var(ly[idx], axis=1, ddof=1); mb = r2[idx].mean(1)
        if var == "main":
            sv = vals[order]
            db = np.array([np.median(np.concatenate([sv[starts[i]:ends[i]] for i in pk])) for pk in pick])
            mb = mb * (db / d0) ** 2
        Rb = 1 - mb / Vb
        rho_b = corr_rows(rankdata(P[idx], axis=1), rankdata(y[idx], axis=1))
        rb = 2 * np.sin(np.pi * rho_b / 6)
        with np.errstate(invalid="ignore", divide="ignore"):
            ycb = np.where(Rb > 0, rb / np.sqrt(np.where(Rb > 0, Rb, 1.0)), np.nan)
        ok = np.isfinite(ycb)
        out[f] = dict(Rb=Rb, varR=float(np.var(Rb, ddof=1)), R_lo=float(np.percentile(Rb, 2.5)),
                      R_hi=float(np.percentile(Rb, 97.5)), R_le0=float(np.mean(Rb <= 0)),
                      yc_boot_var=float(np.var(ycb[ok], ddof=1)), n_bad=int((~ok).sum()))
    return out


R_HI = 2 * np.sin(np.pi * INT_T / 6)     # 0.4654
R_LO = 2 * np.sin(np.pi * DEC_T / 6)     # 0.2611


def band_r(x):
    x = np.asarray(x, float)
    return np.where(x >= R_HI, 2, np.where(x < R_LO, 0, 1))


BAND = {2: "integrated", 1: "middle", 0: "decoupled"}


def rho_ext(yc):
    """rho scale of a normal-score correlation; monotone extension above 1 (asin(yc/2) defined to yc = 2)."""
    return (6 / np.pi) * np.arcsin(np.clip(np.asarray(yc, float), -2, 2) / 2)


def field_table(core, fields, Zn, BR):
    """Per-field quantities of every variant (fields in the order of the noise-draw matrix Zn)."""
    T = core.loc[fields].copy()
    T["v0z"] = 1.0 / (T.n - 3)
    T["v_noise"] = np.var(Zn, axis=0, ddof=1)
    T["noise_share_z"] = T.v_noise / (T.v0z + T.v_noise)
    T["r"] = 2 * np.sin(np.pi * T.rho / 6)
    T["G"] = (np.pi / 3) * np.cos(np.pi * T.rho / 6) * (1 - T.rho ** 2)       # dr/dz
    T["v0r"] = T.G ** 2 * T.v0z
    T["v1r"] = T.G ** 2 * (T.v0z + T.v_noise)
    T["varR"] = [BR[f]["varR"] for f in fields]
    T["R_lo"] = [BR[f]["R_lo"] for f in fields]; T["R_hi"] = [BR[f]["R_hi"] for f in fields]
    T["R_le0"] = [BR[f]["R_le0"] for f in fields]
    T["yc"] = T.r / np.sqrt(T.R)
    T["rho_c"] = rho_ext(T.yc)
    T["v2"] = T.v1r / T.R + T.r ** 2 / (4 * T.R ** 3) * T.varR
    T["v2a"] = T.v0r / T.R + T.r ** 2 / (4 * T.R ** 3) * T.varR
    T["v2_bootvar"] = [BR[f]["yc_boot_var"] for f in fields]
    T["v2b"] = T.v2_bootvar + T.G ** 2 * T.v_noise / T.R          # sensitivity: bootstrap variance of y_c
    # z-scale sensitivity (fields where the plug-in is defined)
    T["zc_defined"] = T.yc.abs() < CAP
    zc, J, K = [], [], []
    h = 1e-6
    for z, R, ok in zip(T.z, T.R, T.zc_defined):
        if not ok:
            zc.append(np.nan); J.append(np.nan); K.append(np.nan); continue
        zc.append(float(zc_of(z, R)))
        J.append(float((zc_of(z + h, R) - zc_of(z - h, R)) / (2 * h)))
        K.append(float((zc_of(z, R + h) - zc_of(z, R - h)) / (2 * h)))
    T["zc"], T["J"], T["K"] = zc, J, K
    T["v2z"] = T.J ** 2 * (T.v0z + T.v_noise) + T.K ** 2 * T.varR
    return T


# ------------------------------------------------------------------ three-level fits ---------------------
def fit3(y, s2=None, Ve=None, g=None, C=None, X=None, prof=False):
    with threadpool_limits(1):
        m = s57.ThreeLevel(np.asarray(y, float), s2=None if s2 is None else np.asarray(s2, float), Ve=Ve, X=X)
        null, conly, full = s57.core_fit(m, g, C)
        h = m.cochran()
        tc, tf, vt = full["tc"], full["tf"], h["v_typ"]
        o = dict(k=len(y), Q=h["Q"], df=h["df"], pQ=h["p"], I2=h["I2"], vtyp=vt, tau2=null["tf"],
                 I2_model=null["tf"] / (null["tf"] + vt), tc=tc, tf=tf, icc=full["icc"], lr=full["lr"],
                 p_mix=s57.mixture_p(full["lr"]), I2c=tc / (tc + tf + vt), I2f=tf / (tc + tf + vt))
        if X is None:
            o["mu"] = m.gls(tc, tf, g, C)[0]
        else:
            b, bc = m.beta(tc, tf, g, C); o["b1"] = float(b[1]); o["b1_se"] = float(np.sqrt(bc[1, 1]))
        if prof:
            prof_c = lambda t: m.max1d(lambda x: m.loglik(t, x, g, C))[1]
            prof_f = lambda t: m.max1d(lambda x: m.loglik(x, t, g, C))[1]
            prof_icc = lambda r: m.max1d(lambda T_: m.loglik(r * T_, (1 - r) * T_, g, C))[1]
            o["tc_ci"] = s57.profile_ci(prof_c, tc, full["ll"], 0.0, s57.UB)
            o["tf_ci"] = s57.profile_ci(prof_f, tf, full["ll"], 0.0, s57.UB)
            o["icc_ci"] = s57.profile_ci(prof_icc, full["icc"] if np.isfinite(full["icc"]) else 0.0,
                                         full["ll"], 0.0, 1.0)
            if Ve is None and X is None:
                lo, hi = qprofile(np.asarray(y, float), np.asarray(s2, float))
                o["tau2_ci"] = (lo, hi); o["I2_ci"] = (lo / (lo + vt), hi / (hi + vt))
    return o, m, null


def perm_lr(m, null, g, C, perms):
    def one(q):
        with threadpool_limits(1):
            return m.fit(g[q], C, null)["lr"]
    return np.array(pmap(one, perms))


def lr_power(y, s2, g, C, stream):
    """scripts/57's parametric bootstrap of the cluster LR (null = fitted field-only model) and its power at the
    fitted variance split and at ICC_true = 0.25 / 0.50 / 0.75 of the fitted total; one SeedSequence child per
    simulated data set, drawn in the parent."""
    y = np.asarray(y, float); s2 = np.asarray(s2, float); k = len(y)
    with threadpool_limits(1):
        m = s57.ThreeLevel(y, s2)
        null, _, full = s57.core_fit(m, g, C)
        mu0 = m.gls(0.0, null["tf"], g, C)[0]
    kids = iter(stream.spawn(N_PB + N_POW * (1 + len(POWER_ICC))))

    def draws(tc_s, tf_s, N):
        out = []
        for _ in range(N):
            rng = np.random.default_rng(next(kids))
            out.append(mu0 + np.sqrt(tc_s) * rng.standard_normal(C)[g] + np.sqrt(tf_s) * rng.standard_normal(k)
                       + np.sqrt(s2) * rng.standard_normal(k))
        return out

    def fit_sim(ys):
        with threadpool_limits(1):
            ms = s57.ThreeLevel(ys, s2)
            return ms.fit(g, C, ms.fit_null())["lr"]
    pb = np.array(pmap(fit_sim, draws(0.0, null["tf"], N_PB)))
    crit = float(np.quantile(pb, 0.95))
    tot = full["tc"] + full["tf"]
    scen = [("point estimate", full["tc"], full["tf"])] + [(f"ICC={r_:.2f}", r_ * tot, (1 - r_) * tot)
                                                           for r_ in POWER_ICC]
    pw = []
    for name, a_, b_ in scen:
        sims = np.array(pmap(fit_sim, draws(a_, b_, N_POW)))
        pw.append(dict(scenario=name, tc=float(a_), tf=float(b_), power=float(np.mean(sims > crit))))
    return dict(crit=crit, p_pboot=s57.perm_p(pb, full["lr"]), power=pw, n_pb=N_PB, n_pow=N_POW)


# ------------------------------------------------------------------ two-way replicates (Scorecard) -------
def _valid(P, y, idx):
    return len(idx) >= 4 and len(np.unique(P[idx])) >= 3 and len(np.unique(y[idx])) >= 3


def sc_rep(args):
    """One institution draw (kind 'shared': one multinomial over the union of the fields' institutions;
    'indep': one draw per field) -> rho per field. Degenerate draws are redrawn (counted)."""
    kind, child = args
    D = _W["SC"]; fields = D["fields"]; rng = np.random.default_rng(child); red = 0
    while True:
        rho = np.empty(len(fields)); ok = True
        if kind == "shared":
            cnt = np.bincount(rng.integers(0, D["NU"], D["NU"]), minlength=D["NU"])
        for j, f in enumerate(fields):
            P, y, pos = D["P"][j], D["y"][j], D["pos"][j]; n = len(P)
            if kind == "shared":
                idx = np.repeat(np.arange(n), cnt[pos])
                if not _valid(P, y, idx):
                    ok = False; break
            else:
                while True:
                    idx = np.repeat(np.arange(n), np.bincount(rng.integers(0, n, n), minlength=n))
                    if _valid(P, y, idx):
                        break
                    red += 1
            rho[j] = corr_rows(rankdata(P[idx]), rankdata(y[idx]))
        if ok:
            return rho, red
        red += 1


def sc_stats(rho, T, sub=None):
    """Statistics over fields from a vector of couplings (R, variances fixed at their estimates)."""
    ix = np.arange(len(rho)) if sub is None else sub
    # |rho| capped at 0.999 as in scripts/57's joint bootstrap: a resampled small field with 3-4 distinct
    # institutions can reach rho = +-1 (z infinite); no observed coupling is near the cap
    rho = np.clip(rho[ix], -CAP, CAP); t = {c: T[c].to_numpy()[ix] for c in ("R", "v0z", "v0r", "v1r", "v2", "v2a")}
    z = np.arctanh(rho); r = 2 * np.sin(np.pi * rho / 6); yc = r / np.sqrt(t["R"])
    i0z, i0, i1, i2, i2a = (np.float64(I2Q(a_, b_)[0]) for a_, b_ in
                            ((z, t["v0z"]), (r, t["v0r"]), (r, t["v1r"]), (yc, t["v2"]), (yc, t["v2a"])))
    return np.array([i0z, i0, i1, i2, i2a, i2 / i0, i2 / i0z, i2 - i0, i1 / i0, i2a / i0,
                     spearmanr(yc, r)[0], np.mean(rho_ext(yc)) - np.mean(rho), np.mean(yc) - np.mean(r)])


SC_STATS = ["I2 V0 (z scale, scripts/57)", "I2 V0r (r scale)", "I2 V1r (noise-aware variance)",
            "I2 V2 (noise-corrected, primary)", "I2 V2a (disattenuation only)",
            "T1 ratio I2(V2)/I2(V0r)", "ratio I2(V2)/I2(V0, z scale)", "I2(V2) - I2(V0r)",
            "ratio I2(V1r)/I2(V0r)", "ratio I2(V2a)/I2(V0r)", "T2 Spearman(y_c, r)",
            "mean rho_c - mean rho", "mean y_c - mean r"]


def twoway_sc(T, M, fields, subsets):
    P = [M[f]["P"] for f in fields]; y = [M[f]["y"] for f in fields]
    U = sorted(set().union(*(set(M[f]["keys"]) for f in fields))); pos_u = {u: i for i, u in enumerate(U)}
    pos = [np.array([pos_u[k] for k in M[f]["keys"]]) for f in fields]
    _W["SC"] = dict(fields=fields, P=P, y=y, pos=pos, NU=len(U))
    items = [("shared", c) for c in SS["tw_sc_shared"].spawn(N_TW)] + \
            [("indep", c) for c in SS["tw_sc_indep"].spawn(N_TW)]
    res = pmap(sc_rep, items)
    RHO = np.array([r_ for r_, _ in res]); red = np.array([k for _, k in res])
    out = {}
    rho0 = T.rho.to_numpy()
    for name, sub in subsets.items():
        est = sc_stats(rho0, T, sub)
        K = len(sub)
        jk = np.array([sc_stats(rho0, T, np.delete(sub, i)) for i in range(K)])
        vfield = (K - 1) / K * ((jk - jk.mean(0)) ** 2).sum(0)
        B = np.array([sc_stats(r_, T, sub) for r_ in RHO])
        cb, ib = B[:N_TW], B[N_TW:]
        out[name] = [dict(stat=s_, est=float(est[i]), **twoway(float(est[i]), float(vfield[i]), cb[:, i], ib[:, i]),
                          cb_lo=float(np.percentile(cb[:, i], 2.5)), cb_hi=float(np.percentile(cb[:, i], 97.5)))
                     for i, s_ in enumerate(SC_STATS)]
    return out, dict(NU=len(U), red_shared=int(red[:N_TW].sum()), red_indep=int(red[N_TW:].sum()),
                     capped=int((np.abs(RHO) >= CAP).sum()), cells=int(RHO.size))


# ------------------------------------------------------------------ PSEO fixed-cohort panel --------------
def pseo_panel(pse, ar):
    x = pse[pse.grad_cohort.isin(COHORTS)]
    wide = None
    for h in HZ:
        xh = x[x.horizon == h][["field", "inst_key", "grad_cohort", "earnings", "se"]].rename(
            columns={"earnings": f"e_{h}", "se": f"se_{h}"})
        wide = xh if wide is None else wide.merge(xh, on=["field", "inst_key", "grad_cohort"], how="inner")
    wide = wide.merge(ar[["field", "inst_key", "prestige_score"]].rename(columns={"prestige_score": "P"}),
                      on=["field", "inst_key"], how="inner")
    wide = wide.sort_values(["field", "grad_cohort", "inst_key"]).reset_index(drop=True)
    cells = [(k, g.reset_index(drop=True)) for k, g in wide.groupby(["field", "grad_cohort"], sort=True)
             if len(g) >= NMIN]
    return wide, cells


def cell_eval(P, E, S, idx):
    """P (n,), E, S (n,3) earnings / SE; idx index vector -> rho (3,), R (3,)."""
    rP = rankdata(P[idx]); rho = np.empty(3); R = np.empty(3)
    for k in range(3):
        e = E[idx, k]; rho[k] = corr_rows(rP, rankdata(e))
        R[k] = 1 - np.mean((S[idx, k] / e) ** 2) / np.var(np.log(e), ddof=1)
    return rho, R


def panel_stats(cellres, cell_field, fields, gate, gfields):
    """cellres: list of (rho (3,), R (3,)) per cell -> per-field statistics (dict of arrays) and the number of
    gated cell x horizon reliabilities below R_FLOOR. All-cell statistics use every cell of the fields in `fields`;
    the correction uses only the cells in `gate` (observed R >= R_GATE at every horizon; fixed from the observed
    data, so every replicate uses the same cells) of the fields in `gfields`, with R floored at R_FLOOR."""
    rho = np.array([c[0] for c in cellres]); R = np.array([c[1] for c in cellres])
    r = 2 * np.sin(np.pi * rho / 6)
    Rf = np.maximum(R, R_FLOOR); yc = r / np.sqrt(Rf); rc = rho_ext(yc)
    fl = np.array(cell_field)
    fm = lambda arr, f, sel=None: arr[(fl == f) if sel is None else ((fl == f) & sel)]
    sl = lambda arr, fs, sel=None: np.array([(fm(arr, f, sel) @ W_SLOPE).mean() for f in fs])
    Rm = np.array([fm(R, f).mean(0) for f in fields])                               # (K, 3) mean over cohorts
    vals = dict(slope_rho=sl(rho, fields), R_y1=Rm[:, 0], R_y5=Rm[:, 1], R_y10=Rm[:, 2],
                R_y10_minus_y1=Rm[:, 2] - Rm[:, 0],
                slope_rho_g=sl(rho, gfields, gate), slope_rhoc_g=sl(rc, gfields, gate),
                slope_r_g=sl(r, gfields, gate), slope_yc_g=sl(yc, gfields, gate))
    vals["slope_rhoc_minus_rho_g"] = vals["slope_rhoc_g"] - vals["slope_rho_g"]
    vals["slope_yc_minus_r_g"] = vals["slope_yc_g"] - vals["slope_r_g"]
    for key, arr in (("rho", rho), ("rhoc", rc)):
        m_ = np.array([fm(arr, f, gate).mean(0) for f in gfields])
        vals[f"{key}_y1_g"], vals[f"{key}_y10_g"] = m_[:, 0], m_[:, 2]
    return vals, int((R[gate] < R_FLOOR).sum())


PANEL_STATS = ["slope_rho", "R_y1", "R_y5", "R_y10", "R_y10_minus_y1", "slope_rho_g", "slope_rhoc_g",
               "slope_rhoc_minus_rho_g", "slope_r_g", "slope_yc_g", "slope_yc_minus_r_g", "rho_y1_g", "rho_y10_g",
               "rhoc_y1_g", "rhoc_y10_g"]


def ps_rep(args):
    """One institution draw for the PSEO panel ('shared': one multinomial over every panel institution; 'indep':
    one draw per field, shared by that field's cohorts). Only degenerate cells (fewer than 3 distinct prestige or
    earnings values) are redrawn, as in scripts/59."""
    kind, child = args
    D = _W["PS"]; rng = np.random.default_rng(child); red = 0

    def ok_cell(c, idx):
        if len(idx) < 4 or len(np.unique(c["P"][idx])) < 3:
            return False
        return all(len(np.unique(c["E"][idx, k])) >= 3 for k in range(3))
    if kind == "shared":
        while True:
            cnt = np.bincount(rng.integers(0, D["NU"], D["NU"]), minlength=D["NU"])
            idxs = [np.repeat(np.arange(len(c["P"])), cnt[c["pos"]]) for c in D["cells"]]
            if all(ok_cell(c, ix) for c, ix in zip(D["cells"], idxs)):
                break
            red += 1
    else:
        idxs = [None] * len(D["cells"])
        for f, cis in D["field_cells"].items():
            nu = D["field_nu"][f]
            while True:
                cnt = np.bincount(rng.integers(0, nu, nu), minlength=nu)
                cand = [np.repeat(np.arange(len(D["cells"][ci]["P"])), cnt[D["cells"][ci]["fpos"]]) for ci in cis]
                if all(ok_cell(D["cells"][ci], ix) for ci, ix in zip(cis, cand)):
                    break
                red += 1
            for ci, ix in zip(cis, cand):
                idxs[ci] = ix
    res = [cell_eval(c["P"], c["E"], c["S"], ix) for c, ix in zip(D["cells"], idxs)]
    v, nfl = panel_stats(res, D["cell_field"], D["fields"], D["gate"], D["gfields"])
    return np.array([v[k].mean() for k in PANEL_STATS]), red, nfl


def panel_section(pse, ar, R, lg):
    """E. PSEO fixed-cohort career-time panel (exploratory): reliability by horizon and the career-time slope
    of the disattenuated coupling, two-way CIs."""
    wide, cells = pseo_panel(pse, ar)
    pfields = sorted({k[0] for k, _ in cells})
    cell_field = [k[0] for k, _ in cells]
    CD = []
    for (f, coh), gcell in cells:
        CD.append(dict(P=gcell.P.to_numpy(float), E=gcell[[f"e_{h}" for h in HZ]].to_numpy(float),
                       S=gcell[[f"se_{h}" for h in HZ]].to_numpy(float), keys=gcell.inst_key.to_numpy(str)))
    se_nan = int(sum(int(np.isnan(c["S"]).sum()) for c in CD))
    assert se_nan == 0, f"{se_nan} PSEO panel program SEs are missing"
    res0 = [cell_eval(c["P"], c["E"], c["S"], np.arange(len(c["P"]))) for c in CD]
    cellR = np.array([r_[1] for r_ in res0])
    assert np.isfinite(cellR).all()
    gate = (cellR >= R_GATE).all(1)
    gfields = sorted({f for f, g_ in zip(cell_field, gate) if g_})
    pv, nfl0 = panel_stats(res0, cell_field, pfields, gate, gfields)
    assert nfl0 == 0
    ref59 = pd.read_csv(REF59)
    s59v = float(ref59[(ref59.section == "career_time") & (ref59.variant == "new") &
                       (ref59.stat == "slope_all")].estimate.iloc[0])
    # pseo_refresh.csv stores 6 significant digits
    assert abs(pv["slope_rho"].mean() - s59v) <= 5e-7, "PSEO panel does not reproduce scripts/59"
    U = sorted(set().union(*(set(c["keys"]) for c in CD))); pos_u = {u: i for i, u in enumerate(U)}
    field_cells, field_nu = {}, {}
    for ci, (c, f) in enumerate(zip(CD, cell_field)):
        c["pos"] = np.array([pos_u[k] for k in c["keys"]])
        field_cells.setdefault(f, []).append(ci)
    for f, cis in field_cells.items():
        uf = sorted(set().union(*(set(CD[ci]["keys"]) for ci in cis))); pf = {u: i for i, u in enumerate(uf)}
        field_nu[f] = len(uf)
        for ci in cis:
            CD[ci]["fpos"] = np.array([pf[k] for k in CD[ci]["keys"]])
    _W["PS"] = dict(cells=CD, cell_field=cell_field, fields=pfields, NU=len(U), field_cells=field_cells,
                    field_nu=field_nu, gate=gate, gfields=gfields)
    items = [("shared", c) for c in SS["tw_ps_shared"].spawn(N_TW)] + \
            [("indep", c) for c in SS["tw_ps_indep"].spawn(N_TW)]
    pr = pmap(ps_rep, items)
    _W.clear()
    PB = np.array([a_ for a_, _, _ in pr]); pred = np.array([b_ for _, b_, _ in pr])
    pfl = np.array([c_ for _, _, c_ in pr])
    PAN = []
    for i, st in enumerate(PANEL_STATS):
        Kst = len(pv[st])
        est = float(pv[st].mean()); vf = float(np.var(pv[st], ddof=1) / Kst)
        PAN.append(dict(stat=st, est=est, k=Kst, **twoway(est, vf, PB[:N_TW, i], PB[N_TW:, i])))
    R["PAN"] = pd.DataFrame(PAN).set_index("stat")
    R["pan_info"] = dict(fields=len(pfields), cells=len(cells), insts=len(U), s59=s59v,
                         red_shared=int(pred[:N_TW].sum()), red_indep=int(pred[N_TW:].sum()),
                         gated_cells=int(gate.sum()), gated_fields=len(gfields),
                         cells_Rle0=int((cellR.min(1) <= 0).sum()), floored_share=float(pfl.sum() / (gate.sum() * 3 * len(pr))),
                         cellR=cellR, cell_n=np.array([len(c["P"]) for c in CD]),
                         dropped=sorted({f"{f} {k_}" for (f, k_), g_ in zip([k for k, _ in cells], gate) if not g_}))
    pf_ = {k: v for k, v in pv.items() if len(v) == len(pfields)}
    R["pan_fields"] = pd.DataFrame(pf_, index=pfields)
    for k, v in pv.items():
        if len(v) == len(gfields):
            R["pan_fields"][k] = pd.Series(v, index=gfields)
    lg(f"PSEO panel: {len(pfields)} fields, {len(cells)} cells ({int(gate.sum())} gated, {len(gfields)} fields); "
       f"slope {pv['slope_rho'].mean():+.4f}; gated rho {pv['slope_rho_g'].mean():+.4f} -> rho_c "
       f"{pv['slope_rhoc_g'].mean():+.4f}; R y1/y5/y10 {pv['R_y1'].mean():.3f}/{pv['R_y5'].mean():.3f}/"
       f"{pv['R_y10'].mean():.3f}; floored share {R['pan_info']['floored_share']:.4f}")


# ================================================================== main ==================================
def free_mb():
    try:
        with open("/proc/meminfo") as fh:
            kv = {l.split(":")[0]: int(l.split()[1]) for l in fh}
        return kv["MemAvailable"] // 1024
    except Exception:
        return -1


def perms57(k, which):
    """scripts/57's own label permutations (SeedSequence(57), stream perm57 / perm20), first N_PERM."""
    kid = np.random.SeedSequence(57).spawn(6)
    prng = np.random.default_rng(kid[1 if which == "57" else 2])
    return [prng.permutation(k) for _ in range(N_PERM)]


def groups(cip2):
    cl, g = np.unique(np.asarray(cip2), return_inverse=True)
    return g, len(cl)


def main():
    global NPROC
    t0 = time.time()
    fm = free_mb()
    if 0 <= fm < 1500:              # resource rule: one worker when less than 1.5 GB is available
        NPROC = 1
    elif 0 <= fm < 2500:
        NPROC = min(NPROC, 2)
    print(f"available memory {fm} MB; workers {NPROC}", flush=True)
    lg = lambda s_: print(f"[{time.time() - t0:7.1f}s] {s_}", flush=True)
    R = {}                                                 # everything the report uses

    # ---------------- A/B. programs, SEs, reliabilities ----------------
    ps = pseo_rows()
    disp = field_dispersion(ps)
    prog = pseo_prog_disp(ps)
    pse = pseo_program_se(ps, disp)
    x5 = ps[(ps.horizon == "y5") & ps.q_ok]
    a_ = x5[x5.grad_cohort == "0000"].groupby("field").rel_iqr.median()
    b_ = x5[x5.grad_cohort != "0000"].groupby("field").rel_iqr.median()
    ratio_fixed = (b_ / a_).dropna()
    R["pseo"] = dict(rows=len(ps), rows_y5_pooled=int(((ps.horizon == "y5") & (ps.grad_cohort == "0000")).sum()),
                     q_ok=float(ps.q_ok.mean()), programs=len(pse), se_nan=int(pse.se.isna().sum()),
                     fixed_pooled_ratio=float(ratio_fixed.median()),
                     fixed_pooled_q=(float(ratio_fixed.quantile(.1)), float(ratio_fixed.quantile(.9))),
                     fixed_pooled_k=len(ratio_fixed), md5_earn=md5(s59.NEW_EARN), md5_inst=md5(s59.NEW_INST))
    sc = scorecard_programs(disp, prog)
    R["sc"] = dict(programs=len(sc), md5=md5(SCORECARD_FOS))
    lg(f"PSEO rows {len(ps)}; PSEO programs {len(pse)}; Scorecard programs {len(sc)}")
    ar = load_ar_wapman(fields=FIELDS66)
    M = matched(ar, sc)
    pools = {}
    for f in M:
        pl = dispersion_pool(ps, disp, f); pools[f] = (pl.rel_iqr.to_numpy(float), pl.inst_key.to_numpy(str))
    del ps, x5, a_, b_                      # the long PSEO frame is not needed again (pse / pools keep what is)
    gc.collect()
    d57, meta = s57.load_fields()
    fields57 = list(d57.field)
    core = field_core(M)
    for f, n, rho in zip(d57.field, d57.n, d57.rho):
        assert core.loc[f, "n"] == n and np.isclose(core.loc[f, "rho"], rho), f"57 rebuild mismatch {f}"
    coh = np.concatenate([M[f]["cohort"] for f in fields57])
    rse = np.concatenate([M[f]["se_main"] / M[f]["y"] for f in fields57])
    R["prog"] = dict(n_programs=float(len(coh)), cohort_med=float(np.median(coh)), cohort_min=float(coh.min()),
                     cohort_p10=float(np.percentile(coh, 10)), relse_med=float(np.median(rse)),
                     dp_sd_mid=DP_SD["dp_mid"], dp_sd_hi=DP_SD["dp_hi"],
                     relse_p10=float(np.percentile(rse, 10)), relse_p90=float(np.percentile(rse, 90)),
                     share_cohort_lt30=float(np.mean(coh < 30)))
    dsrc = disp.loc[fields57, "d_src"].value_counts()
    R["disp_med"] = float(disp.loc[fields57, "d_rel"].median())
    R["disp_src"] = (f"own-field pool for {int(dsrc.get('field', 0))} of 57 fields, CIP-2 pool for "
                     f"{int(dsrc.get('cip2', 0))}, all rows for {int(dsrc.get('all', 0))}; pools of "
                     f"{int(disp.loc[fields57, 'd_rows'].min())} to {int(disp.loc[fields57, 'd_rows'].max())} rows")

    # ---------------- T3 baseline classification (scripts/52's rule) ----------------
    er4 = load_er_scorecard(fields=FIELDS66)
    gm4 = compute_gap_map(ar, er4, "undergrad", B=s52.B_GAP, seed=s52.SEED)
    cls = s52.classify(gm4).set_index("field")
    del er4
    classified = [f for f in cls.index if cls.loc[f, "group"] != "unclassified"]
    for f in classified:
        assert f in M and core.loc[f, "n"] == cls.loc[f, "baseline_n"] and \
            np.isclose(core.loc[f, "rho"], cls.loc[f, "baseline_rho"]), f"classification rebuild mismatch {f}"
    FA = sorted(set(fields57) | set(classified))
    lg(f"classified fields {len(classified)} "
       f"({cls.group.value_counts().to_dict()}); analysis fields {len(FA)}")

    # ---------------- C. noise draws, reliability bootstrap, per-field table ----------------
    Zn = noise_draws(M, FA, "main", SS["noise"])
    BR = boot_R(M, FA, pools, "main")
    T = field_table(core, FA, Zn, BR)
    T["cip2"] = [CIP2[f] for f in FA]
    T["d_rel"] = [disp.loc[f, "d_rel"] for f in FA]; T["d_src"] = [disp.loc[f, "d_src"] for f in FA]
    T["group52"] = [cls.loc[f, "group"] if f in cls.index else "not in gm4" for f in FA]
    lg("noise draws + reliability bootstrap done")
    j57 = np.array([FA.index(f) for f in fields57])
    T57 = T.loc[fields57]
    g, C = groups(d57.cip2)
    rel20 = np.where(d57.reliable.to_numpy())[0]
    R["T"] = T; R["T57"] = T57; R["cls"] = cls; R["disp"] = disp; R["meta"] = meta

    # noise-draw quantities on the r scale (twice-noisy reliability)
    rho_d = np.tanh(Zn); r_d = 2 * np.sin(np.pi * rho_d / 6)
    Rstar = ((T.V - T.m) / (T.V + T.m)).to_numpy()
    yc_d = r_d / np.sqrt(Rstar)[None, :]

    # ---------------- D. heterogeneity: primary sample (57, independent errors) ----------------
    y_of = dict(V0z=T57.z, V0r=T57.r, V1r=T57.r, V2=T57.yc, V2a=T57.yc, V2b=T57.yc)
    v_of = dict(V0z=T57.v0z, V0r=T57.v0r, V1r=T57.v1r, V2=T57.v2, V2a=T57.v2a, V2b=T57.v2b)
    FIT, PERM = {}, {}
    P57 = perms57(57, "57")
    Xn = s57.design(pd.DataFrame(dict(log_n=np.log(d57.n.to_numpy(float)))), ("log_n",))
    for vn in ("V0z", "V0r", "V1r", "V2", "V2a", "V2b"):
        FIT[vn], m, null = fit3(y_of[vn].to_numpy(), v_of[vn].to_numpy(), g=g, C=C, prof=True)
        if vn in ("V0z", "V0r", "V1r", "V2"):
            lr = perm_lr(m, null, g, C, P57)
            FIT[vn]["p_perm"] = s57.perm_p(lr, FIT[vn]["lr"]); FIT[vn]["perm_q95"] = float(np.quantile(lr, .95))
            FIT[vn + "+logn"], mx, nx = fit3(y_of[vn].to_numpy(), v_of[vn].to_numpy(), g=g, C=C, X=Xn)
            lrx = perm_lr(mx, nx, g, C, P57)
            FIT[vn + "+logn"]["p_perm"] = s57.perm_p(lrx, FIT[vn + "+logn"]["lr"])
        lg(f"fit {vn}: I2={FIT[vn]['I2']:.4f} tau2={FIT[vn]['tau2']:.5f} tc={FIT[vn]['tc']:.5f} "
           f"tf={FIT[vn]['tf']:.5f} ICC={FIT[vn]['icc']:.3f} LR={FIT[vn]['lr']:.3f} p_mix={FIT[vn]['p_mix']:.4f}"
           + (f" p_perm={FIT[vn]['p_perm']:.4f}" if "p_perm" in FIT[vn] else "")
           + (f" | +log n LR={FIT[vn + '+logn']['lr']:.3f} p_mix={FIT[vn + '+logn']['p_mix']:.4f} "
              f"p_perm={FIT[vn + '+logn']['p_perm']:.4f}" if vn + "+logn" in FIT else ""))
    # z-scale V2 sensitivity without the undefined field(s)
    zd = T57.zc_defined.to_numpy()
    gz, Cz = groups(d57.cip2[zd])
    FIT["V2z"], _, _ = fit3(T57.zc[zd].to_numpy(), T57.v2z[zd].to_numpy(), g=gz, C=Cz)
    FIT["V0z_same"], _, _ = fit3(T57.z[zd].to_numpy(), T57.v0z[zd].to_numpy(), g=gz, C=Cz)
    R["zc_undefined"] = list(T57.index[~zd])
    # reproduction of scripts/57
    ref = pd.read_csv(REF57)
    rsel = lambda st, note=None: float(ref[(ref["sample"] == "nondegen57") & (ref.var_spec == "fisher") &
                                           (ref.adjust == "none") & (ref.statistic == st)].estimate.iloc[0])
    for key, st in (("I2", "I2_total"), ("tc", "tau2_cluster"), ("lr", "LR_cluster")):
        assert np.isclose(FIT["V0z"][key], rsel(st), rtol=1e-5), f"scripts/57 reproduction failed: {st}"
    R["FIT"] = FIT
    kp = SS["power"].spawn(2)
    R["POW"] = {"V0r": lr_power(T57.r, T57.v0r, g, C, kp[0]), "V2": lr_power(T57.yc, T57.v2, g, C, kp[1])}
    lg("power: " + "; ".join(f"{k_} crit={v_['crit']:.3f} p_pboot={v_['p_pboot']:.4f} "
                              + ",".join(f"{s_['power']:.3f}" for s_ in v_['power']) for k_, v_ in R["POW"].items()))
    sp_ = lambda a_, b_: tuple(float(x) for x in spearmanr(a_, b_))
    R["assoc"] = dict(R_vs_rho=sp_(T57.R, T57.rho), R_vs_n=sp_(T57.R, T57.n), R_vs_cohort=sp_(T57.R, T57.med_cohort),
                      yc_vs_n=sp_(T57.yc, T57.n), rho_vs_n=sp_(T57.rho, T57.n))
    rk0 = rankdata(-T57.rho.to_numpy(), method="ordinal"); rk1 = rankdata(-T57.yc.to_numpy(), method="ordinal")
    dmv = rk0 - rk1                                   # > 0: moves up the ordering
    o_ = np.argsort(-dmv, kind="stable")
    R["moves"] = dict(up=[(T57.index[i], rk0[i], rk1[i]) for i in o_[:3]],
                      down=[(T57.index[i], rk0[i], rk1[i]) for i in o_[::-1][:3]],
                      mean_abs=float(np.mean(np.abs(dmv))))

    # reliable 20 and correlated errors (exploratory)
    gr, Cr = groups(d57.cip2.iloc[rel20])
    for vn in ("V0z", "V0r", "V1r", "V2"):
        FIT["r20_" + vn], _, _ = fit3(y_of[vn].to_numpy()[rel20], v_of[vn].to_numpy()[rel20], g=gr, C=Cr)
    kid = np.random.SeedSequence(57).spawn(6)
    tables, _ = s57.rebuild_fields(d57)
    with threadpool_limits(1):                           # scripts/57 runs it single-threaded
        JB = s57.joint_bootstrap(d57, np.random.default_rng(kid[0]), tables)
    Rc = JB["R"]
    sz = np.sqrt(T57.v0z.to_numpy())
    Ve = dict(V0z=Rc * np.outer(sz, sz))
    Gd = np.diag(T57.G.to_numpy())
    Ve["V0r"] = Gd @ Ve["V0z"] @ Gd
    Ve["V1r"] = Ve["V0r"] + np.diag((T57.G ** 2 * T57.v_noise).to_numpy())
    Sd = np.diag(1 / np.sqrt(T57.R.to_numpy()))
    Ve["V2"] = Sd @ Ve["V1r"] @ Sd + np.diag((T57.r ** 2 / (4 * T57.R ** 3) * T57.varR).to_numpy())
    for vn in ("V0z", "V0r", "V1r", "V2"):
        FIT["corr_" + vn], _, _ = fit3(y_of[vn].to_numpy(), Ve=Ve[vn], g=g, C=C)
    assert np.isclose(FIT["corr_V0z"]["lr"], float(ref[(ref["sample"] == "nondegen57") & (ref.var_spec == "fisher_corr")
                      & (ref.adjust == "none") & (ref.statistic == "LR_cluster")].estimate.iloc[0]), rtol=1e-4)
    R["JB"] = dict(n_ok=JB["n_ok"])
    del tables, JB
    lg("reliable-20 and correlated-error fits done")

    # ---------------- noise-propagation and reliability-uncertainty distributions (57) ----------------
    ycd57, rd57 = yc_d[:, j57], r_d[:, j57]
    r57, v1, v2 = T57.r.to_numpy(), T57.v1r.to_numpy(), T57.v2.to_numpy()
    ND = dict(I2_V2=np.array([I2Q(ycd57[i], v2)[0] for i in range(N_DRAW)]),
              I2_V1r=np.array([I2Q(rd57[i], v1)[0] for i in range(N_DRAW)]),
              sp_yc=np.array([spearmanr(ycd57[i], r57)[0] for i in range(N_DRAW)]),
              sp_r=np.array([spearmanr(rd57[i], r57)[0] for i in range(N_DRAW)]))
    ND["I2_ratio"] = ND["I2_V2"] / FIT["V0r"]["I2"]

    def one_draw(i):
        o, _, _ = fit3(ycd57[i], v2, g=g, C=C)
        return o["tc"], o["tf"], (o["icc"] if np.isfinite(o["icc"]) else 0.0), o["lr"], o["tau2"]
    fd = np.array(pmap(one_draw, range(N_DRAW)))
    for i_, k_ in enumerate(("tc", "tf", "icc", "lr", "tau2")):
        ND[k_] = fd[:, i_]
    Rb57 = np.stack([BR[f]["Rb"] for f in fields57], 1)          # (N_BOOT_R, 57)
    Rb57f = np.maximum(Rb57, 0.01)
    ycb57 = r57[None, :] / np.sqrt(Rb57f)
    RD = dict(I2_V2=np.array([I2Q(ycb57[i], v2)[0] for i in range(N_BOOT_R)]),
              sp_yc=np.array([spearmanr(ycb57[i], r57)[0] for i in range(N_BOOT_R)]),
              floored=int((Rb57 < 0.01).sum()))
    RD["I2_ratio"] = RD["I2_V2"] / FIT["V0r"]["I2"]
    R["ND"], R["RD"] = ND, RD
    R["lr_crit_mix"] = float(chi2.ppf(0.90, 1))       # 0.05 critical value of the 0.5 chi2_0 + 0.5 chi2_1 mixture
    R["nd_lr_sig"] = float(np.mean(ND["lr"] > R["lr_crit_mix"]))
    lg("noise-propagation draws fitted")

    # ---------------- two-way (field, institution) cluster variance ----------------
    TW, twinfo = twoway_sc(T57, M, fields57, {"57": np.arange(57), "reliable20": rel20})
    R["TW"], R["twinfo"] = TW, twinfo
    lg(f"two-way replicates done ({twinfo})")

    # ---------------- T3 ----------------
    jc = np.array([FA.index(f) for f in classified])
    base_band = np.array([{"integrated": 2, "middle": 1, "decoupled": 0}[cls.loc[f, "group"]] for f in classified])
    new_band = band_r(T.loc[classified, "yc"].to_numpy())
    draw_band = band_r(yc_d[:, jc])
    Rbc = np.maximum(np.stack([BR[f]["Rb"] for f in classified], 1), 0.01)
    rb_band = band_r(T.loc[classified, "r"].to_numpy()[None, :] / np.sqrt(Rbc))
    t3 = []
    for i, f in enumerate(classified):
        r_ = T.loc[f, "r"]; b0 = base_band[i]
        # reliability at which r / sqrt(R) reaches the next band up; undefined for r <= 0 (the correction only
        # moves a coupling away from 0)
        need = np.nan if (b0 == 2 or r_ <= 0) else (r_ / R_HI) ** 2 if b0 == 1 else (r_ / R_LO) ** 2
        t3.append(dict(field=f, n=int(T.loc[f, "n"]), rho=T.loc[f, "rho"], R=T.loc[f, "R"], R_lo=T.loc[f, "R_lo"],
                       R_hi=T.loc[f, "R_hi"], rho_c=T.loc[f, "rho_c"], base=BAND[b0], new=BAND[int(new_band[i])],
                       flip=bool(new_band[i] != b0), share_draws_new=float(np.mean(draw_band[:, i] == new_band[i])),
                       share_draws_changed=float(np.mean(draw_band[:, i] != b0)),
                       share_Rboot_changed=float(np.mean(rb_band[:, i] != b0)), R_needed=need))
    T3 = pd.DataFrame(t3).set_index("field")
    # rho-only classification of the 57 fields with n >= 15 (secondary)
    ro = T57[T57.n >= NMIN]
    ro_b0 = band_r(ro.r.to_numpy()); ro_b1 = band_r(ro.yc.to_numpy())
    jro = np.array([FA.index(f) for f in ro.index])
    ro_draw = band_r(yc_d[:, jro])
    RO = pd.DataFrame(dict(rho=ro.rho, R=ro.R, rho_c=ro.rho_c, base=[BAND[int(b)] for b in ro_b0],
                           new=[BAND[int(b)] for b in ro_b1],
                           share_draws_new=[float(np.mean(ro_draw[:, i] == ro_b1[i])) for i in range(len(ro))]))
    RO["flip"] = RO.base != RO.new
    R["T3"], R["RO"] = T3, RO
    lg(f"T3: {int(T3.flip.sum())} of {len(T3)} classified fields flip; rho-only: {int(RO.flip.sum())} of {len(RO)}")

    # ---------------- SIMEX (exploratory) ----------------
    lam_kids = SS["simex"].spawn(len(SIMEX_L))
    rbar = {0.0: T.r.to_numpy()}
    for lam, kk in zip(SIMEX_L, lam_kids):
        Zl = noise_draws(M, FA, "main", kk, lam)
        rbar[lam] = (2 * np.sin(np.pi * np.tanh(Zl) / 6)).mean(0)
        del Zl
    L = np.array([0.0] + list(SIMEX_L)); Yl = np.stack([rbar[l_] for l_ in L])
    coef = np.polyfit(L, Yl, 2)                                  # (3, K)
    T["r_simex"] = coef[0] * 1 - coef[1] + coef[2]               # quadratic extrapolation to lambda = -1
    T["r_simex_lin"] = np.array([np.polyval(np.polyfit(L, Yl[:, j], 1), -1.0) for j in range(len(FA))])
    R["simex"] = dict(sp=float(spearmanr(T.loc[fields57, "r_simex"], T57.yc)[0]),
                      mean_simex=float(T.loc[fields57, "r_simex"].mean()), mean_hs=float(T57.yc.mean()),
                      mean_r=float(T57.r.mean()), sp_r=float(spearmanr(T.loc[fields57, "r_simex"], T57.r)[0]))
    lg("SIMEX done")

    # ---------------- SE variants (exploratory) ----------------
    VAR = {}
    for var, _ in EXPL_VARIANTS:
        cv = field_core(M, var).loc[FA]
        Zv = noise_draws(M, FA, var, np.random.SeedSequence([SEED, 1 + SE_VARIANTS.index(var)]))
        BRv = boot_R(M, FA, pools, var)
        Tv = field_table(cv, FA, Zv, BRv)
        del Zv
        t57 = Tv.loc[fields57]; ok = (t57.R > 0).to_numpy()
        gv, Cv = groups(d57.cip2[ok])
        f0, _, _ = fit3(t57.r[ok].to_numpy(), t57.v0r[ok].to_numpy(), g=gv, C=Cv)
        f2, _, _ = fit3(t57.yc[ok].to_numpy(), t57.v2[ok].to_numpy(), g=gv, C=Cv)
        okc = (Tv.loc[classified, "R"] > 0).to_numpy()
        nb = band_r(Tv.loc[classified, "yc"].to_numpy())
        flips = [f for i, f in enumerate(classified) if okc[i] and nb[i] != base_band[i]]
        VAR[var] = dict(k=int(ok.sum()), R_le0=list(t57.index[~ok]), R_med=float(t57.R.median()),
                        R_min=float(t57.R.min()), I2_0=f0["I2"], I2_2=f2["I2"], ratio=f2["I2"] / f0["I2"],
                        tc=f2["tc"], tf=f2["tf"], lr=f2["lr"], p_mix=f2["p_mix"],
                        sp=float(spearmanr(t57.yc[ok], t57.r[ok])[0]), flips=flips,
                        cls_undefined=[f for i, f in enumerate(classified) if not okc[i]])
        lg(f"variant {var}: k={VAR[var]['k']} ratio={VAR[var]['ratio']:.3f} sp={VAR[var]['sp']:.3f} flips={flips}")
    R["VAR"] = VAR
    del Zn, yc_d, r_d, rho_d
    gc.collect()

    # ---------------- E. PSEO fixed-cohort career-time panel (exploratory) ----------------
    panel_section(pse, ar, R, lg)

    if os.environ.get("ER73_DUMP"):                   # development aid: keep R to re-render the report
        import pickle
        with open(os.environ["ER73_DUMP"], "wb") as fh:
            pickle.dump(R, fh)
    write_outputs(R)
    write_report(R)
    lg("wrote " + ", ".join(str(p.relative_to(ROOT)) for p in (OUT_CSV, OUT_FIELDS, OUT_MD)))


# ================================================================== outputs ===============================
VLAB = {"V0z": "V0 z = atanh(rho), 1/(n-3) (scripts/57)",
        "V0r": "V0r r = 2 sin(pi rho/6), G^2/(n-3)",
        "V1r": "V1r r, G^2 [1/(n-3) + v_noise] (noise-aware variance)",
        "V2": "V2 y_c = r/sqrt(R), noise-corrected (PRIMARY)",
        "V2a": "V2a y_c, disattenuation only (G^2/(n-3)/R + K term)",
        "V2b": "V2b y_c, bootstrap variance of y_c + noise term",
        "V2z": "V2z z_c = atanh(rho_c), z-scale plug-in (fields where defined)",
        "V0z_same": "V0 z on the same fields as V2z"}
SCALE = {"V0z": "z", "V0r": "r", "V1r": "r", "V2": "y_c", "V2a": "y_c", "V2b": "y_c", "V2z": "z_c", "V0z_same": "z"}


def _f(x, d=3, sign=False):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(x):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _ci(lo, hi, d=3, sign=False):
    return f"[{_f(lo, d, sign)}, {_f(hi, d, sign)}]"


def _p(p):
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(p):
        return "—"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.4f}"


def _q(x, qs=(2.5, 50, 97.5)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, q)) for q in qs]


def _lab(f):
    return LAB.get(f, f)


def t1_reading(x):
    return ("most of I2 remains" if x >= 0.75 else "part of I2 remains" if x >= 0.25
            else "most of I2 is earnings noise")


def t1_band_of_ci(lo, hi):
    cuts = [c for c in (0.25, 0.75) if lo < c <= hi]
    return ("the two-way 95% CI lies inside that band" if not cuts else
            "the two-way 95% CI crosses the " + " and ".join(f"{c:.2f}" for c in cuts) + " boundary")


def t2_verdict(est, lo):
    if est >= 0.80 and lo >= 0.80:
        return "survives"
    if est >= 0.80:
        return "not established (estimate >= 0.80, two-way lower bound < 0.80)"
    return "does not survive (estimate < 0.80)"


def build_rows(R):
    """Every number of the report goes through rec() into data/interim/earnings_reliability.csv."""
    T, T57, FIT, TW = R["T"], R["T57"], R["FIT"], R["TW"]
    ps_, sc_ = R["pseo"], R["sc"]
    rec("data", "pseo_rows_bachelor_cip4_long", ps_["rows"], sample="PSEO V4.14.1, institution x CIP-4 x cohort x horizon")
    rec("data", "pseo_rows_y5_pooled", ps_["rows_y5_pooled"], sample="PSEO V4.14.1, y5, grad_cohort 0000")
    rec("data", "pseo_share_rows_with_p25_p75_n", ps_["q_ok"])
    rec("data", "pseo_programs", ps_["programs"], sample="field x institution x cohort x horizon")
    rec("data", "pseo_program_se_missing", ps_["se_nan"])
    rec("data", "pseo_fixed_over_pooled_rel_iqr_median", ps_["fixed_pooled_ratio"], lo=ps_["fixed_pooled_q"][0],
        hi=ps_["fixed_pooled_q"][1], k=ps_["fixed_pooled_k"], note="per-field ratio of median rel. IQR, fixed "
        "cohorts / pooled 0000, y5; lo/hi = 10th/90th percentile over fields")
    rec("data", "md5_pseoe_all", np.nan, note=ps_["md5_earn"]); rec("data", "md5_pseo_institutions", np.nan, note=ps_["md5_inst"])
    rec("data", "scorecard_programs", sc_["programs"], sample="Scorecard FoS credential level 3, FIELDS66")
    rec("data", "md5_scorecard_fos", np.nan, note=sc_["md5"])
    pr = R["prog"]
    for k_, v_ in pr.items():
        rec("programs", k_, v_, sample="matched programs of the 57 fields (Scorecard, primary SE)")
    for f in T.index:
        t = T.loc[f]
        for c in ("n", "rho", "z", "r", "V", "m", "R", "R_lo", "R_hi", "R_le0", "rank_rel", "med_cohort", "min_cohort",
                  "med_relse", "d_rel", "v0z", "v_noise", "noise_share_z", "varR", "yc", "rho_c", "v0r", "v1r", "v2",
                  "v2a", "v2b", "zc", "v2z", "r_simex", "r_simex_lin"):
            rec("field", c, t[c], field=f, sample="57 fields" if f in T57.index else "classified only",
                note=f"d_src={t.d_src}; group52={t.group52}" if c == "R" else "")
    for vn, o in FIT.items():
        for key in ("k", "Q", "df", "pQ", "I2", "vtyp", "tau2", "I2_model", "tc", "tf", "icc", "lr", "p_mix",
                    "p_perm", "perm_q95", "mu", "b1", "b1_se", "I2c", "I2f"):
            if key not in o:
                continue
            lo = hi = np.nan
            if key == "tc" and "tc_ci" in o:
                lo, hi = o["tc_ci"][:2]
            if key == "tf" and "tf_ci" in o:
                lo, hi = o["tf_ci"][:2]
            if key == "icc" and "icc_ci" in o:
                lo, hi = o["icc_ci"][:2]
            if key == "tau2" and "tau2_ci" in o:
                lo, hi = o["tau2_ci"]
            if key == "I2" and "I2_ci" in o:
                lo, hi = o["I2_ci"]
            rec("threelevel", key, o[key], variant=vn, lo=lo, hi=hi, k=o["k"],
                note=VLAB.get(vn.split("+")[0].replace("r20_", "").replace("corr_", ""), vn)
                + (" + log n" if "+logn" in vn else "") + ("; reliable 20" if vn.startswith("r20_") else "")
                + ("; correlated errors (joint-bootstrap correlation)" if vn.startswith("corr_") else "")
                + ("; tau2/I2 CI = Q-profile; tc/tf/icc CI = profile likelihood" if key in ("tau2", "I2", "tc", "tf", "icc") else ""))
    for sub, rows in TW.items():
        for r_ in rows:
            rec("twoway", r_["stat"], r_["est"], sample=sub, lo=r_["tlo"], hi=r_["thi"],
                note=f"SE={r_['tse']:.4g}; V_field={r_['tvf']:.3g}; V_inst={r_['tvs']:.3g}; V_fxi={r_['tvi']:.3g}"
                     + ("; V<=0 fallback" if r_["tfallback"] else "") + f"; MC SE endpoint={r_['tmcse']:.2g}; "
                     f"shared-draw percentile [{r_['cb_lo']:.4g}, {r_['cb_hi']:.4g}]")
    for k_, x in R["ND"].items():
        q = _q(x)
        rec("noise_draws", k_, q[1], lo=q[0], hi=q[2], k=N_DRAW, sample="57 fields",
            note="median and 2.5/97.5 percentiles over noise-perturbation draws")
    for k_, x in R["RD"].items():
        if k_ == "floored":
            rec("reliability_draws", k_, x, note="field x draw reliabilities below 0.01, floored at 0.01")
            continue
        q = _q(x)
        rec("reliability_draws", k_, q[1], lo=q[0], hi=q[2], k=N_BOOT_R, sample="57 fields",
            note="median and 2.5/97.5 percentiles over reliability-bootstrap draws")
    for f, t in R["T3"].iterrows():
        for c in ("n", "rho", "R", "rho_c", "share_draws_new", "share_draws_changed", "share_Rboot_changed", "R_needed"):
            rec("T3", c, t[c], field=f, lo=t.R_lo if c == "R" else np.nan, hi=t.R_hi if c == "R" else np.nan,
                note=f"base={t.base}; new={t.new}; flip={t.flip}")
    for f, t in R["RO"].iterrows():
        for c in ("rho", "R", "rho_c", "share_draws_new"):
            rec("rho_only_classification", c, t[c], field=f, note=f"base={t.base}; new={t.new}; flip={t.flip}")
    for k_, v_ in R["simex"].items():
        rec("simex", k_, v_, sample="57 fields")
    for var, o in R["VAR"].items():
        for k_ in ("k", "R_med", "R_min", "I2_0", "I2_2", "ratio", "tc", "tf", "lr", "p_mix", "sp"):
            rec("se_variant", k_, o[k_], variant=var, note=f"R<=0: {o['R_le0']}; flips: {o['flips']}; "
                f"classification undefined: {o['cls_undefined']}")
    for k_, v_ in R["assoc"].items():
        rec("assoc", k_, v_[0], p=v_[1], k=57, sample="57 fields", note="Spearman across fields")
    for vn, o in R["POW"].items():
        rec("power", "pboot_crit_LR", o["crit"], variant=vn, k=o["n_pb"])
        rec("power", "p_pboot_LR", o["p_pboot"], variant=vn, k=o["n_pb"])
        for s_ in o["power"]:
            rec("power", "power_LR_cluster", s_["power"], variant=vn, k=o["n_pow"],
                note=f"{s_['scenario']}: tau2_cluster={s_['tc']:.5g}, tau2_field={s_['tf']:.5g}")
    for st, r_ in R["PAN"].iterrows():
        rec("pseo_panel", st, r_["est"], lo=r_["tlo"], hi=r_["thi"], k=r_["k"],
            sample=f"PSEO V4.14.1 fixed cohorts {'/'.join(COHORTS)}, balanced, n>={NMIN}"
                   + (f"; gated cells (observed R >= {R_GATE} at every horizon)" if st.endswith("_g") else "; all cells"),
            note=f"two-way SE={r_['tse']:.4g}" + ("; V<=0 fallback" if r_["tfallback"] else ""))
    for k_ in ("fields", "cells", "insts", "s59", "red_shared", "red_indep", "gated_cells", "gated_fields",
               "cells_Rle0", "floored_share"):
        rec("pseo_panel_info", k_, R["pan_info"][k_])
    rec("pseo_panel_info", "cells_dropped_by_gate", len(R["pan_info"]["dropped"]), note="; ".join(R["pan_info"]["dropped"]))
    for f, t in R["pan_fields"].iterrows():
        for c in R["pan_fields"].columns:
            rec("pseo_panel_field", c, t[c], field=f)
    cr = R["pan_info"]["cellR"]
    for i, h in enumerate(HZ):
        q = _q(cr[:, i], (0, 50, 100))
        rec("pseo_panel_info", f"cell_R_{h}", q[1], lo=q[0], hi=q[2], k=len(cr), note="median; lo/hi = min/max over cells")
    for k_ in ("NU", "red_shared", "red_indep"):
        rec("twoway_info", k_, R["twinfo"][k_])
    rec("jb", "joint_bootstrap_usable", R["JB"]["n_ok"])


def write_outputs(R):
    build_rows(R)
    df = pd.DataFrame(ROWS)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.8g")
    T = R["T"].copy()
    T.insert(0, "label", [_lab(f) for f in T.index])
    T["in57"] = [f in R["T57"].index for f in T.index]
    T3 = R["T3"]
    T["t3_base"] = [T3.loc[f, "base"] if f in T3.index else "" for f in T.index]
    T["t3_new"] = [T3.loc[f, "new"] if f in T3.index else "" for f in T.index]
    T.drop(columns=[c for c in ("G", "J", "K", "v2_bootvar") if c in T.columns]).to_csv(
        OUT_FIELDS, index_label="field", float_format="%.8g")


def write_report(R):
    T, T57, FIT, TW, ND, RD, T3, RO = (R[k] for k in ("T", "T57", "FIT", "TW", "ND", "RD", "T3", "RO"))
    tw = {r_["stat"]: r_ for r_ in TW["57"]}
    tw20 = {r_["stat"]: r_ for r_ in TW["reliable20"]}
    L = []
    a = L.append
    t1 = tw["T1 ratio I2(V2)/I2(V0r)"]; t1z = tw["ratio I2(V2)/I2(V0, z scale)"]
    t2 = tw["T2 Spearman(y_c, r)"]
    i0z, i0r, i1r, i2, i2a = (tw[s_] for s_ in ("I2 V0 (z scale, scripts/57)", "I2 V0r (r scale)",
                                                "I2 V1r (noise-aware variance)", "I2 V2 (noise-corrected, primary)",
                                                "I2 V2a (disattenuation only)"))
    r1, r2a = tw["ratio I2(V1r)/I2(V0r)"], tw["ratio I2(V2a)/I2(V0r)"]
    mdiff = tw["mean rho_c - mean rho"]
    flips = T3[T3.flip]; robust = flips[flips.share_draws_new >= 0.95]
    v2t = t2_verdict(t2["est"], t2["tlo"])
    pr = R["prog"]
    F2, F0r, F0z = FIT["V2"], FIT["V0r"], FIT["V0z"]
    nd_i2, nd_ratio, nd_sp = _q(ND["I2_V2"]), _q(ND["I2_ratio"]), _q(ND["sp_yc"])
    rd_i2, rd_ratio, rd_sp = _q(RD["I2_V2"]), _q(RD["I2_ratio"]), _q(RD["sp_yc"])
    Rq = _q(T57.R, (0, 25, 50, 75, 100))
    lowR = T57.sort_values("R").head(5)
    a("# Earnings-noise reliability of within-field coupling")
    a("")
    a("Script: `scripts/73_earnings_reliability.py` (seed 73; one SeedSequence child per replicate, byte-identical on "
      "re-run and for any worker count). coupling rho_f = Spearman, across the institutions that offer bachelor's field f, "
      "of Wapman field prestige and the program's median earnings (Scorecard FoS EARN_MDN_4YR, credential level 3). "
      "Descriptive, not causal. The pre-specified tests T1-T3 and their reading rules were fixed in the script docstring on "
      "2026-09-24 before any T1-T3 statistic was computed; one amendment (primary correction on the normal-score scale, "
      "made after the per-field reliabilities were computed and before any T1-T3 statistic) is stated in the docstring and "
      "in Method. An interrupted first build printed T1-T3 once; this re-run keeps their definitions and reading rules "
      "unchanged, and the restart additions listed in the docstring (differential-privacy SE variants, the matched point "
      "for the Q-profile I2 interval, wording of the draw distributions) are exploratory or presentational. Everything else "
      "is labelled exploratory. Public data already on disk; nothing was downloaded for this analysis.")
    a("")
    a("## Answer")
    a("")
    a(f"**Key question: once the sampling noise of each program's earnings median is propagated, how much of the "
      f"between-field heterogeneity of coupling (I2 = {_f(F0z['I2'])}), the field ordering and the field classification "
      f"remains?** Program medians are estimated from a median of {_f(pr['cohort_med'], 0)} earners (10th percentile "
      f"{_f(pr['cohort_p10'], 0)}), with an approximate relative SE of {_f(100 * pr['relse_med'], 1)}% of the median. "
      f"The per-field reliability of log earnings across institutions has median {_f(Rq[2], 2)} (range {_f(Rq[0], 2)} "
      f"to {_f(Rq[4], 2)}, 57 fields). "
      f"T1: the noise-corrected I2 is {_f(i2['est'])} against {_f(i0r['est'])} on the same scale, ratio {_f(t1['est'])} "
      f"{_ci(t1['tlo'], t1['thi'])} (two-way 95% CI), so by the pre-specified reading **{t1_reading(t1['est'])}**; "
      f"{t1_band_of_ci(t1['tlo'], t1['thi'])}. The primary variant counts the earnings-noise variance on top of "
      f"1/(n-3), which is conservative; with disattenuation alone the ratio is {_f(r2a['est'])} "
      f"{_ci(r2a['tlo'], r2a['thi'])}. At the two-way lower bound of the primary ratio, the share of I2 lost when earnings noise is propagated is at "
      f"most {_f(100 * (1 - t1['tlo']), 0)}%. "
      f"T2: Spearman(corrected, original) across fields = {_f(t2['est'])} {_ci(t2['tlo'], t2['thi'])}; the field ordering "
      f"**{v2t}** by the pre-specified rule. Re-drawing every median once more around its observed value gives a median "
      f"Spearman with the original ordering of {_f(nd_sp[1])} after correction and {_f(_q(ND['sp_r'])[1])} without it "
      f"(exploratory; each draw adds a second dose of noise). "
      f"T3: {len(flips)} of {len(T3)} classified fields change band after disattenuation"
      + (f" ({', '.join(f'{_lab(f)}: {t.base} -> {t.new}' for f, t in flips.iterrows())}); "
         f"{len(robust)} of them in at least 95% of the noise draws." if len(flips) else
         "; the least stable (largest share of noise draws with a band change) are " + ", ".join(
             f"{_lab(f)} ({t.base}; {_f(100 * t.share_draws_changed, 1)}%)"
             for f, t in T3.sort_values("share_draws_changed", ascending=False,
                                        kind="stable").head(2).iterrows()) + ".")
      + (" Under the exploratory SE variants: " + "; ".join(
          f"{nm_} changes the band of {', '.join(_lab(x) for x in R['VAR'][v_]['flips'])}"
          for v_, nm_ in EXPL_VARIANTS if R["VAR"][v_]["flips"]) + "."
         if any(R["VAR"][v_]["flips"] for v_, _ in EXPL_VARIANTS) else
         " No field flips under the exploratory SE variants.")
      + f" In scripts/57's three-level model (r scale), the CIP-2 cluster variance is {_f(F0r['tc'], 4)} before and "
      f"{_f(F2['tc'], 4)} {_ci(*F2['tc_ci'][:2], 4)} after the correction, while the within-cluster field variance "
      f"goes from {_f(F0r['tf'], 4)} to {_f(F2['tf'], 4)} {_ci(*F2['tf_ci'][:2], 4)}; ICC_true {_f(F0r['icc'], 2)} -> "
      f"{_f(F2['icc'], 2)}, LR_c {_f(F0r['lr'], 2)} -> {_f(F2['lr'], 2)} (label-permutation p {_p(F0r['p_perm'])} -> "
      f"{_p(F2['p_perm'])}; with log n {_p(FIT['V0r+logn']['p_perm'])} -> {_p(FIT['V2+logn']['p_perm'])}).")
    a("")
    # 1. T1
    a(f"1. **T1 — how much of I2 remains (pre-specified).** 57 fields (scripts/57's non-degenerate set), independent "
      f"errors, Higgins-Thompson I2. On the normal-score scale r = 2 sin(pi rho / 6): V0r (sampling of institutions only) "
      f"I2 = {_f(i0r['est'])} {_ci(i0r['tlo'], i0r['thi'])}; V1r (plus the earnings-noise variance of each coupling) "
      f"{_f(i1r['est'])} {_ci(i1r['tlo'], i1r['thi'])}; V2 (disattenuated coupling y_c = r / sqrt(R_f) with its full "
      f"variance, PRIMARY) {_f(i2['est'])} {_ci(i2['tlo'], i2['thi'])}. T1 ratio I2(V2)/I2(V0r) = {_f(t1['est'])} "
      f"{_ci(t1['tlo'], t1['thi'])}; reading: {t1_reading(t1['est'])} ({t1_band_of_ci(t1['tlo'], t1['thi'])}). "
      f"Against scripts/57's z-scale I2 = {_f(i0z['est'])}: ratio {_f(t1z['est'])} {_ci(t1z['tlo'], t1z['thi'])}. "
      f"Cochran Q of V2 = {_f(F2['Q'], 1)} on {F2['df']} df (p = {_p(F2['pQ'])}); the model-based I2(V2) = "
      f"tau2_REML / (tau2_REML + typical within-field variance) = {_f(F2['I2_model'])}, Q-profile 95% CI "
      f"{_ci(*F2['I2_ci'])} (V0r: {_f(F0r['I2_model'])} {_ci(*F0r['I2_ci'])}). At the two-way lower bound of the T1 "
      f"ratio, the share of I2 lost is at most {_f(100 * (1 - t1['tlo']), 0)}%; the upper bound "
      f"({_f(t1['thi'])}) does not exclude that none is lost. The two mechanisms separately: adding the noise "
      f"variance alone gives ratio {_f(r1['est'])} {_ci(r1['tlo'], r1['thi'])}; disattenuation alone (V2a) gives "
      f"{_f(r2a['est'])} {_ci(r2a['tlo'], r2a['thi'])}. "
      f"Propagation (not CIs of the point estimate): across the {N_DRAW} noise-perturbation draws, in which every median "
      f"is re-drawn around its observed value and the coupling re-corrected, I2(V2) has median {_f(nd_i2[1])} "
      f"(2.5-97.5%: {_f(nd_i2[0])} to {_f(nd_i2[2])}; ratio {_f(nd_ratio[1])}, {_f(nd_ratio[0])} to "
      f"{_f(nd_ratio[2])}); across the {N_BOOT_R} reliability-bootstrap draws, median {_f(rd_i2[1])} ({_f(rd_i2[0])} "
      f"to {_f(rd_i2[2])}; ratio {_f(rd_ratio[1])}, {_f(rd_ratio[0])} to {_f(rd_ratio[2])}). Each noise draw adds a "
      f"second dose of noise that the fixed V2 variances do not carry, and resampled reliabilities spread the "
      f"corrected couplings, so the level of both distributions sits "
      f"{'above' if nd_i2[1] > i2['est'] and rd_i2[1] > i2['est'] else 'away from'} the point estimate; their width "
      f"is the quantity of interest.")
    # 2. T2
    a(f"2. **T2 — field ordering (pre-specified).** Spearman(y_c, r) across the 57 fields = {_f(t2['est'])} "
      f"{_ci(t2['tlo'], t2['thi'])} (two-way 95% CI; SE {_f(t2['tse'], 4)}); rule: survives if the estimate and the lower "
      f"bound are both >= 0.80 -> **{v2t}**. Propagation distributions (as in 1; exploratory): Spearman of the re-drawn "
      f"and re-corrected coupling with the original, noise-perturbation draws median {_f(nd_sp[1])} ({_f(nd_sp[0])} to "
      f"{_f(nd_sp[2])}); reliability-bootstrap draws: median {_f(rd_sp[1])} ({_f(rd_sp[0])} to {_f(rd_sp[2])}). "
      f"Mean coupling rises from {_f(T57.rho.mean())} to {_f(T57.rho_c.mean())} on the rho scale (difference "
      f"{_f(mdiff['est'], 3, True)} {_ci(mdiff['tlo'], mdiff['thi'], 3, True)}). Largest moves up the ordering: "
      + ", ".join(f"{_lab(f)} (rank {int(a_)} -> {int(b_)})" for f, a_, b_ in R["moves"]["up"])
      + "; largest moves down: "
      + ", ".join(f"{_lab(f)} (rank {int(a_)} -> {int(b_)})" for f, a_, b_ in R["moves"]["down"])
      + f" (rank 1 = highest coupling; mean absolute rank change {_f(R['moves']['mean_abs'], 1)} of 57).")
    # 3. T3
    t3txt = (f"3. **T3 — classification flips (pre-specified).** scripts/52's rule on the Scorecard 4YR baseline "
             f"(reliable_flag and n >= 15; integrated rho >= 0.45, decoupled < 0.25): {len(T3)} classified fields "
             f"({(T3.base == 'integrated').sum()} integrated, {(T3.base == 'middle').sum()} middle, "
             f"{(T3.base == 'decoupled').sum()} decoupled). ")
    if len(flips):
        t3txt += (f"{len(flips)} change band after disattenuation: "
                  + "; ".join(f"{_lab(f)} {t.base} -> {t.new} (rho {_f(t.rho)} -> {_f(t.rho_c)}, R = {_f(t.R, 2)} "
                              f"{_ci(t.R_lo, t.R_hi, 2)}; in the new band in {_f(100 * t.share_draws_new, 1)}% of noise draws, "
                              f"band changed in {_f(100 * t.share_Rboot_changed, 1)}% of reliability-bootstrap draws; "
                              f"{'robust' if t.share_draws_new >= 0.95 else 'not robust'})" for f, t in flips.iterrows())
                  + ". ")
    else:
        t3txt += "No field changes band. "
    stay = T3[(~T3.flip) & (T3.base != "integrated")]
    if len(stay):
        t3txt += ("Distance to a flip for the non-integrated fields that stay (the reliability at which the corrected "
                  "coupling would reach the next band up, against the estimate and its bootstrap 95% CI): "
                  + "; ".join((f"{_lab(f)} ({t.base}) needs R <= {_f(t.R_needed, 2)}, has {_f(t.R, 2)} "
                               if np.isfinite(t.R_needed) else
                               f"{_lab(f)} ({t.base}; rho <= 0, cannot move up) has R = {_f(t.R, 2)} ")
                              + f"{_ci(t.R_lo, t.R_hi, 2)}, band changed in {_f(100 * t.share_draws_changed, 1)}% of noise "
                              f"draws" for f, t in stay.iterrows()) + ". ")
    ig = T3[T3.base == "integrated"].sort_values("share_draws_changed", ascending=False, kind="stable")
    t3txt += ("Integrated fields cannot move down at the point estimate: disattenuation only raises a positive "
              f"coupling. In the noise draws they leave their band in at most "
              f"{_f(100 * ig.share_draws_changed.iloc[0], 1)}% of draws ({_lab(ig.index[0])}). ")
    rof = RO[RO.flip]
    t3txt += (f"Secondary, rho-only bands over the {len(RO)} of 57 fields with n >= 15 (no reliability flag): "
              f"{len(rof)} change band"
              + (": " + "; ".join(f"{_lab(f)} {t.base} -> {t.new} ({_f(100 * t.share_draws_new, 1)}% of draws)"
                                 for f, t in rof.iterrows()) if len(rof) else "") + ".")
    a(t3txt)
    # 4. noise / reliability
    a(f"4. **How noisy the program medians are.** Matched Scorecard programs of the 57 fields: {pr['n_programs']:.0f} "
      f"(institution x field), earners per program median {_f(pr['cohort_med'], 0)} (min {_f(pr['cohort_min'], 0)}, "
      f"10th percentile {_f(pr['cohort_p10'], 0)}); approximate SE of the median {_f(100 * pr['relse_med'], 1)}% of the "
      f"median (10th-90th percentile {_f(100 * pr['relse_p10'], 1)}% to {_f(100 * pr['relse_p90'], 1)}%). Borrowed PSEO "
      f"within-program relative IQR: median over fields {_f(R['disp_med'], 3)} ({R['disp_src']}). Per-field reliability "
      f"R_f of log earnings across institutions: quartiles {_f(Rq[1], 2)} / {_f(Rq[2], 2)} / {_f(Rq[3], 2)}, range "
      f"{_f(Rq[0], 2)} to {_f(Rq[4], 2)}; rank-reliability index (expected Spearman of observed with noise-free "
      f"earnings) median {_f(T57.rank_rel.median(), 3)}. Lowest R_f: "
      + ", ".join(f"{_lab(f)} {_f(t.R, 2)} {_ci(t.R_lo, t.R_hi, 2)} (n={int(t.n)})" for f, t in lowR.iterrows())
      + f". Share of a coupling's z-scale variance that is earnings noise (V1): median "
      f"{_f(T57.noise_share_z.median(), 3)}. Exploratory associations across the 57 fields: Spearman(R_f, rho_f) = "
      f"{_f(R['assoc']['R_vs_rho'][0], 3, True)} (p = {_p(R['assoc']['R_vs_rho'][1])}), Spearman(R_f, n_f) = "
      f"{_f(R['assoc']['R_vs_n'][0], 3, True)} (p = {_p(R['assoc']['R_vs_n'][1])}), Spearman(R_f, median cohort) = "
      f"{_f(R['assoc']['R_vs_cohort'][0], 3, True)} (p = {_p(R['assoc']['R_vs_cohort'][1])}).")
    # 5. three-level
    Fl0, Fl2 = FIT["V0r+logn"], FIT["V2+logn"]
    pw = R["POW"]
    a(f"5. **scripts/57's three-level model with the noise propagated.** V0z reproduces scripts/57 (I2 {_f(F0z['I2'])}, "
      f"tau2_cluster {_f(F0z['tc'], 4)}, LR_c {_f(F0z['lr'], 2)}). On the r scale, V0r: tau2_cluster {_f(F0r['tc'], 4)} "
      f"{_ci(*F0r['tc_ci'][:2], 4)}, tau2_field {_f(F0r['tf'], 4)}, ICC_true {_f(F0r['icc'], 2)} {_ci(*F0r['icc_ci'][:2], 2)}, "
      f"LR_c {_f(F0r['lr'], 2)} (mixture p {_p(F0r['p_mix'])}, label-permutation p {_p(F0r['p_perm'])}). V2: tau2_cluster "
      f"{_f(F2['tc'], 4)} {_ci(*F2['tc_ci'][:2], 4)}, tau2_field {_f(F2['tf'], 4)} {_ci(*F2['tf_ci'][:2], 4)}, ICC_true "
      f"{_f(F2['icc'], 2)} {_ci(*F2['icc_ci'][:2], 2)}, LR_c {_f(F2['lr'], 2)} (mixture p {_p(F2['p_mix'])}, permutation p "
      f"{_p(F2['p_perm'])}, parametric-bootstrap p {_p(pw['V2']['p_pboot'])}). Field-only tau on each scale: V0r "
      f"{_f(np.sqrt(F0r['tau2']), 3)}, V2 {_f(np.sqrt(F2['tau2']), 3)}. With log n (scripts/57's primary adjustment): "
      f"V0r LR_c {_f(Fl0['lr'], 2)} (mixture p {_p(Fl0['p_mix'])}, permutation p {_p(Fl0['p_perm'])}), V2 LR_c "
      f"{_f(Fl2['lr'], 2)} (mixture p {_p(Fl2['p_mix'])}, permutation p {_p(Fl2['p_perm'])}); log-n slope V0r "
      f"{_f(Fl0['b1'], 3, True)} (SE {_f(Fl0['b1_se'], 3)}), V2 {_f(Fl2['b1'], 3, True)} (SE {_f(Fl2['b1_se'], 3)}). "
      f"Across the noise draws (V2 refit {N_DRAW} times): ICC_true median {_f(_q(ND['icc'])[1], 2)} "
      f"({_f(_q(ND['icc'])[0], 2)} to {_f(_q(ND['icc'])[2], 2)}), LR_c median {_f(_q(ND['lr'])[1], 2)} "
      f"({_f(_q(ND['lr'])[0], 2)} to {_f(_q(ND['lr'])[2], 2)}), share of draws with LR_c above the 0.05 mixture "
      f"critical value {_f(R['lr_crit_mix'], 2)}: {_f(100 * R['nd_lr_sig'], 1)}%. Power of the cluster test (parametric bootstrap, "
      f"{pw['V2']['n_pow']} draws per scenario) at a true ICC of 0.50 with the fitted total variance: V0r "
      + _f([s_['power'] for s_ in pw['V0r']['power'] if s_['scenario'] == 'ICC=0.50'][0], 2)
      + ", V2 " + _f([s_['power'] for s_ in pw['V2']['power'] if s_['scenario'] == 'ICC=0.50'][0], 2) + ".")
    # 6. exploratory checks
    V = R["VAR"]; sx = R["simex"]
    a(f"6. **Checks (exploratory).** SE variants (T1 ratio / T2 Spearman / T3 flips / median R_f): "
      + "; ".join(f"{nm} {_f(V[v_]['ratio'])} / {_f(V[v_]['sp'])} / {len(V[v_]['flips'])}"
                  + (f" ({', '.join(_lab(x) for x in V[v_]['flips'])})" if V[v_]['flips'] else "")
                  + f" / {_f(V[v_]['R_med'], 2)}"
                  + (f", {len(V[v_]['R_le0'])} field(s) with R <= 0 dropped" if V[v_]['R_le0'] else "")
                  for v_, nm in EXPL_VARIANTS)
      + f". The DP variants add a relative SD of {_f(100 * pr['dp_sd_mid'], 2)}% (midpoints) or "
      f"{_f(100 * pr['dp_sd_hi'], 2)}% (upper edges) of the median, from the distribution of the perturbation "
      f"published in the Scorecard documentation, against a median sampling SE of {_f(100 * pr['relse_med'], 1)}%"
      + f". SIMEX (quadratic extrapolation to lambda = -1): mean corrected r {_f(sx['mean_simex'])} vs "
      f"{_f(sx['mean_hs'])} (primary) and {_f(sx['mean_r'])} (observed); Spearman with the primary correction "
      f"{_f(sx['sp'])}. Correlated errors (scripts/57's joint-bootstrap correlation): I2 V0r {_f(FIT['corr_V0r']['I2'])} "
      f"-> V2 {_f(FIT['corr_V2']['I2'])}, LR_c {_f(FIT['corr_V0r']['lr'], 2)} -> {_f(FIT['corr_V2']['lr'], 2)} "
      f"(mixture p {_p(FIT['corr_V2']['p_mix'])}). Reliable 20: I2 V0r {_f(FIT['r20_V0r']['I2'])} -> V2 "
      f"{_f(FIT['r20_V2']['I2'])} (T1-type ratio {_f(tw20['T1 ratio I2(V2)/I2(V0r)']['est'])} "
      f"{_ci(tw20['T1 ratio I2(V2)/I2(V0r)']['tlo'], tw20['T1 ratio I2(V2)/I2(V0r)']['thi'])}; ordering "
      f"{_f(tw20['T2 Spearman(y_c, r)']['est'])} {_ci(tw20['T2 Spearman(y_c, r)']['tlo'], tw20['T2 Spearman(y_c, r)']['thi'])}). "
      f"z-scale plug-in V2z without the {len(R['zc_undefined'])} field(s) where it is undefined "
      f"({', '.join(_lab(x) for x in R['zc_undefined'])}): I2 {_f(FIT['V0z_same']['I2'])} -> {_f(FIT['V2z']['I2'])} "
      f"(ratio {_f(FIT['V2z']['I2'] / FIT['V0z_same']['I2'])}).")
    # 7. PSEO panel
    PAN, pi = R["PAN"], R["pan_info"]
    a(f"7. **PSEO fixed-cohort career-time panel (exploratory).** {pi['fields']} fields, {pi['cells']} field x cohort "
      f"cells (V4.14.1, cohorts {'/'.join(COHORTS)}, balanced, n >= {NMIN}; reproduces scripts/59's slope "
      f"{_f(pi['s59'], 4, True)}/yr). "
      f"PSEO releases p25/p75 per row, so each program's SE is its own. Mean reliability across cells y1 / y5 / y10: "
      f"{_f(PAN.loc['R_y1', 'est'], 3)} / {_f(PAN.loc['R_y5', 'est'], 3)} / {_f(PAN.loc['R_y10', 'est'], 3)} (y10 - y1 "
      f"{_f(PAN.loc['R_y10_minus_y1', 'est'], 3, True)} {_ci(PAN.loc['R_y10_minus_y1', 'tlo'], PAN.loc['R_y10_minus_y1', 'thi'], 3, True)}), "
      + ("so the reliability of the medians shows no clear change with years since graduation (the two-way CI of the "
         "y10 - y1 difference includes 0). "
         if PAN.loc['R_y10_minus_y1', 'tlo'] <= 0 <= PAN.loc['R_y10_minus_y1', 'thi'] else
         f"so reliability is {'higher' if PAN.loc['R_y10_minus_y1', 'est'] > 0 else 'lower'} at y10 than at y1 "
         "(two-way CI excludes 0). ")
      + f"{pi['cells_Rle0']} cells have an observed R <= 0 at "
      f"some horizon; the correction therefore uses the {pi['gated_cells']} cells ({pi['gated_fields']} fields) with "
      f"observed R >= {R_GATE} at every horizon, with resampled R floored at {R_FLOOR} "
      f"({_f(100 * pi['floored_share'], 2)}% of resampled cell-horizon reliabilities). On those cells the career-time "
      f"slope of coupling is {_f(PAN.loc['slope_rho_g', 'est'], 4, True)}/yr "
      f"{_ci(PAN.loc['slope_rho_g', 'tlo'], PAN.loc['slope_rho_g', 'thi'], 4, True)} (all cells "
      f"{_f(PAN.loc['slope_rho', 'est'], 4, True)}); of the disattenuated coupling "
      f"{_f(PAN.loc['slope_rhoc_g', 'est'], 4, True)}/yr {_ci(PAN.loc['slope_rhoc_g', 'tlo'], PAN.loc['slope_rhoc_g', 'thi'], 4, True)}; "
      f"difference {_f(PAN.loc['slope_rhoc_minus_rho_g', 'est'], 4, True)} "
      f"{_ci(PAN.loc['slope_rhoc_minus_rho_g', 'tlo'], PAN.loc['slope_rhoc_minus_rho_g', 'thi'], 4, True)} (two-way 95% CIs; "
      f"normal-score scale {_f(PAN.loc['slope_r_g', 'est'], 4, True)} -> {_f(PAN.loc['slope_yc_g', 'est'], 4, True)}). "
      f"The gate and the floor were set after the per-cell reliabilities were seen (post hoc); the gated slope of the "
      f"uncorrected coupling is shown beside the all-cell slope so that the effect of the gate itself is visible.")
    a("")
    # ---------------- key numbers ----------------
    a("## Key numbers")
    a("")
    a("Two-way CI = normal 95% interval from the two-way (field, institution) cluster variance of scripts/59 "
      "(V_field jackknife over fields + V_institution one shared institution draw - V_field x institution independent "
      f"draws per field; {N_TW} draws of each kind), reliabilities and variances held at their estimates. Q-profile / "
      "profile-likelihood CIs where stated. Noise-draw and reliability-bootstrap columns give 2.5-97.5 percentiles.")
    a("")
    a("| quantity | sample / spec | estimate | 95% CI | p | k |")
    a("|---|---|---|---|---|---|")
    row = lambda q, s_, e, c="—", p="—", k="57": a(f"| {q} | {s_} | {e} | {c} | {p} | {k} |")
    for st, lab_ in (("I2 V0 (z scale, scripts/57)", "I2, V0 z scale (scripts/57)"),
                     ("I2 V0r (r scale)", "I2, V0r"), ("I2 V1r (noise-aware variance)", "I2, V1r"),
                     ("I2 V2 (noise-corrected, primary)", "I2, V2 (primary)"),
                     ("I2 V2a (disattenuation only)", "I2, V2a"),
                     ("T1 ratio I2(V2)/I2(V0r)", "**T1** ratio I2(V2)/I2(V0r)"),
                     ("ratio I2(V2)/I2(V0, z scale)", "ratio I2(V2)/I2(V0 z)"),
                     ("I2(V2) - I2(V0r)", "I2(V2) - I2(V0r)"),
                     ("ratio I2(V1r)/I2(V0r)", "ratio I2(V1r)/I2(V0r)"),
                     ("ratio I2(V2a)/I2(V0r)", "ratio I2(V2a)/I2(V0r)"),
                     ("T2 Spearman(y_c, r)", "**T2** Spearman(y_c, r)"),
                     ("mean rho_c - mean rho", "mean rho_c - mean rho"),
                     ("mean y_c - mean r", "mean y_c - mean r")):
        r_ = tw[st]
        sg = st in ("I2(V2) - I2(V0r)", "mean rho_c - mean rho", "mean y_c - mean r")
        row(lab_, "57 fields, two-way CI", _f(r_["est"], 3, sg), _ci(r_["tlo"], r_["thi"], 3, sg))
    row("share of I2 lost when earnings noise is propagated, upper bound = 1 - two-way lower bound of T1", "57 fields",
        _f(1 - tw["T1 ratio I2(V2)/I2(V0r)"]["tlo"], 3))
    for st in ("T1 ratio I2(V2)/I2(V0r)", "T2 Spearman(y_c, r)", "I2 V2 (noise-corrected, primary)"):
        r_ = tw20[st]
        row(st, "reliable 20, two-way CI", _f(r_["est"]), _ci(r_["tlo"], r_["thi"]), k="20")
    for vn in ("V0z", "V0r", "V1r", "V2", "V2a", "V2b"):
        o = FIT[vn]
        row(f"Cochran Q, {vn}", VLAB[vn], f"{_f(o['Q'], 1)} ({o['df']} df)", "—", _p(o["pQ"]))
        row(f"I2 = (Q - df)/Q, {vn}", VLAB[vn] + " (two-way CI above)", _f(o["I2"]))
        row(f"I2 model-based tau2/(tau2 + s~^2), {vn}", "REML tau2, typical variance s~^2; Q-profile CI",
            _f(o["I2_model"]), _ci(*o["I2_ci"]))
        row(f"tau2 field-only REML, {vn}", f"{VLAB[vn]}; scale {SCALE[vn]}", _f(o["tau2"], 5), _ci(*o["tau2_ci"], 5))
        row(f"tau2_cluster, {vn}", "three-level REML, profile CI", _f(o["tc"], 5), _ci(*o["tc_ci"][:2], 5))
        row(f"tau2_field, {vn}", "three-level REML, profile CI", _f(o["tf"], 5), _ci(*o["tf_ci"][:2], 5))
        row(f"ICC_true, {vn}", "three-level REML, profile CI", _f(o["icc"], 3), _ci(*o["icc_ci"][:2], 3))
        row(f"LR_cluster, {vn}", "mixture p" + ("; label-permutation p (2000)" if "p_perm" in o else ""),
            _f(o["lr"], 3), "—", _p(o["p_mix"]) + (f"; {_p(o['p_perm'])}" if "p_perm" in o else ""))
    for vn in ("V0z", "V0r", "V1r", "V2"):
        o = FIT[vn + "+logn"]
        row(f"LR_cluster + log n, {vn}", "mixture p; label-permutation p (2000, n attached)", _f(o["lr"], 3), "—",
            f"{_p(o['p_mix'])}; {_p(o['p_perm'])}")
        row(f"log-n slope, {vn}", "three-level REML, GLS", _f(o["b1"], 4, True),
            _ci(o["b1"] - Z975 * o["b1_se"], o["b1"] + Z975 * o["b1_se"], 4, True))
    for vn in ("V0r", "V2"):
        o = pw[vn]
        row(f"parametric-bootstrap p of LR_cluster, {vn}", f"{o['n_pb']} draws from the field-only fit", "—", "—",
            _p(o["p_pboot"]))
        for s_ in o["power"]:
            row(f"power of the cluster test, {vn}, {s_['scenario']}", f"{o['n_pow']} draws; 5% level (pboot critical "
                f"value {_f(o['crit'], 3)})", _f(s_["power"], 3))
    for key, lab_ in (("I2_V2", "I2(V2) over noise draws"), ("I2_ratio", "T1 ratio over noise draws"),
                      ("sp_yc", "T2 Spearman over noise draws"), ("I2_V1r", "I2(V1r) over noise draws"),
                      ("sp_r", "Spearman(r*, r) over noise draws"), ("icc", "ICC_true(V2) over noise draws"),
                      ("lr", "LR_cluster(V2) over noise draws"), ("tc", "tau2_cluster(V2) over noise draws"),
                      ("tf", "tau2_field(V2) over noise draws"), ("tau2", "tau2 field-only (V2) over noise draws")):
        q = _q(ND[key])
        row(lab_, f"{N_DRAW} noise-perturbation draws; median", _f(q[1], 4), _ci(q[0], q[2], 4))
    for key, lab_ in (("I2_V2", "I2(V2) over reliability draws"), ("I2_ratio", "T1 ratio over reliability draws"),
                      ("sp_yc", "T2 Spearman over reliability draws")):
        q = _q(RD[key])
        row(lab_, f"{N_BOOT_R} reliability-bootstrap draws; median", _f(q[1], 4), _ci(q[0], q[2], 4))
    row("T3 flips (primary)", "classified fields, scripts/52 rule", f"{len(flips)} of {len(T3)}", k=str(len(T3)))
    row("T3 flips robust (>= 95% of noise draws)", "classified fields", f"{len(robust)} of {len(flips)}", k=str(len(T3)))
    row("rho-only bands, flips", "57 fields with n >= 15", f"{int(RO.flip.sum())} of {len(RO)}", k=str(len(RO)))
    row("median R_f", "57 fields; range in CI column", _f(Rq[2], 3), _ci(Rq[0], Rq[4], 3))
    row("median relative SE of a program median", "matched programs of the 57 fields; 10th-90th pct in CI column",
        _f(pr["relse_med"], 4), _ci(pr["relse_p10"], pr["relse_p90"], 4), k=f"{pr['n_programs']:.0f} programs")
    row("median earners per program", "matched programs; min and 10th pct in CI column", _f(pr["cohort_med"], 0),
        f"[{_f(pr['cohort_min'], 0)}, {_f(pr['cohort_p10'], 0)}]", k=f"{pr['n_programs']:.0f} programs")
    row("DP relative SD of a Scorecard median, midpoints / upper edges", "Scorecard FoS documentation Exhibit 2 "
        "(4-year column), exploratory", f"{_f(pr['dp_sd_mid'], 4)} / {_f(pr['dp_sd_hi'], 4)}", k="—")
    row("median rank-reliability index", "57 fields", _f(T57.rank_rel.median(), 3))
    row("median noise share of z variance (V1)", "57 fields", _f(T57.noise_share_z.median(), 3))
    row("mean rho / mean rho_c", "57 fields", f"{_f(T57.rho.mean())} / {_f(T57.rho_c.mean())}")
    for k_, lab_ in (("R_vs_rho", "Spearman(R_f, rho_f)"), ("R_vs_n", "Spearman(R_f, n_f)"),
                     ("R_vs_cohort", "Spearman(R_f, median earners)"), ("yc_vs_n", "Spearman(y_c, n_f)"),
                     ("rho_vs_n", "Spearman(rho_f, n_f)")):
        row(lab_, "57 fields (exploratory)", _f(R["assoc"][k_][0], 3, True), "—", _p(R["assoc"][k_][1]))
    for v_, _ in EXPL_VARIANTS:
        o = V[v_]
        row(f"SE variant {v_}: I2 V0r -> V2, ratio", f"{o['k']} fields with R > 0 (exploratory)",
            f"{_f(o['I2_0'])} -> {_f(o['I2_2'])}, {_f(o['ratio'])}", k=str(o["k"]))
        row(f"SE variant {v_}: Spearman(y_c, r); LR_c", "exploratory", f"{_f(o['sp'])}; {_f(o['lr'], 2)}", "—",
            _p(o["p_mix"]), k=str(o["k"]))
    row("SIMEX mean corrected r", "57 fields, quadratic, lambda = -1 (exploratory)", _f(sx["mean_simex"]))
    row("SIMEX Spearman with primary y_c", "57 fields (exploratory)", _f(sx["sp"]))
    for st, lab_ in (("slope_rho", "career slope of rho (/yr), all cells"), ("R_y1", "reliability y1"),
                     ("R_y5", "reliability y5"), ("R_y10", "reliability y10"), ("R_y10_minus_y1", "reliability y10 - y1"),
                     ("slope_rho_g", "career slope of rho (/yr), gated cells"),
                     ("slope_rhoc_g", "career slope of rho_c (/yr), gated cells"),
                     ("slope_rhoc_minus_rho_g", "rho_c slope - rho slope, gated cells"),
                     ("slope_r_g", "career slope of r (/yr), gated cells"),
                     ("slope_yc_g", "career slope of y_c (/yr), gated cells"),
                     ("slope_yc_minus_r_g", "y_c slope - r slope, gated cells"),
                     ("rho_y1_g", "rho at y1, gated"), ("rho_y10_g", "rho at y10, gated"),
                     ("rhoc_y1_g", "rho_c at y1, gated"), ("rhoc_y10_g", "rho_c at y10, gated")):
        r_ = PAN.loc[st]
        d_ = 4 if "slope" in st else 3
        smp = (f"{pi['gated_cells']} gated cells" if st.endswith("_g") else f"{pi['cells']} cells")
        row(f"PSEO panel: {lab_}", f"{int(r_['k'])} fields, {smp}, two-way CI (exploratory)",
            _f(r_["est"], d_, True), _ci(r_["tlo"], r_["thi"], d_, True), k=str(int(r_["k"])))
    a("")
    # ---------------- method ----------------
    a("## Method")
    a("")
    a(f"- **Sample.** scripts/57's 57 non-degenerate fields of `outputs/expanded66_gap_map.csv`, rebuilt program by program "
      f"from `load_ar_wapman(FIELDS66)` and Scorecard FoS (EARN_MDN_4YR, credential level 3, count-weighted over a field's "
      f"CIP-4 rows as `load_er_scorecard` does); n_f and rho_f are asserted to reproduce scripts/57, and V0z reproduces its "
      f"I2, tau2_cluster and LR_c. T3 uses scripts/52's {len(T3)} classified fields "
      f"({'all' if all(f in T57.index for f in T3.index) else 'not all'} inside the 57).")
    a(f"- **SE of a program median.** SE ~ {MED_SE:.4f} sigma / sqrt(n) (large-sample SE of a median of a normal "
      f"distribution, sqrt(pi/2) sigma / sqrt(n)), sigma = IQR / {IQR_Z:.3f}. Scorecard releases no within-program spread, so "
      f"sigma = d_f x median / 1.349 with d_f the field's median relative IQR (p75 - p25)/p50 over PSEO V4.14.1 bachelor's "
      f"institution x CIP-4 rows at y5, pooled cohorts ({R['disp_src']}); n = EARN_COUNT_WNE_4YR. A program pooling "
      f"several CIP-4 rows gets SE^2 = sum w^2 SE_j^2 / (sum w)^2 with the loader's weights. PSEO rows use their own "
      f"p25/p75 and y{{h}}_grads_earn ({_f(100 * R['pseo']['q_ok'], 1)}% of released rows have both). Definitions "
      f"(Scorecard data dictionary, FieldOfStudy_Data_Dictionary): EARN_MDN_4YR = median earnings of graduates working "
      f"and not enrolled 4 years after completing; EARN_COUNT_WNE_4YR = the number of those graduates.")
    a(f"- **Inputs (md5).** Scorecard FoS `data/raw/scorecard_fos/Most-Recent-Cohorts-Field-of-Study.csv` "
      f"`{R['sc']['md5']}`; PSEO V4.14.1 `data/raw/pseo_2026q2/pseoe_all.csv.gz` `{R['pseo']['md5_earn']}` and "
      f"`pseo_all_institutions.csv` `{R['pseo']['md5_inst']}` (provenance: data/raw/SOURCES.md sections 2 and 10a); "
      f"Scorecard FoS documentation for the DP variants: SOURCES.md section 10d.")
    a("- **Reliability.** R_f = 1 - mean_i (SE_i / y_i)^2 / Var_i(log y_i) over the field's matched institutions (share of "
      "the cross-institution variance of log earnings that is not sampling noise). Rank-reliability index "
      "(6/pi) asin(sqrt(R_f)/2). Uncertainty of R_f: bootstrap over the field's institutions and, jointly, over the PSEO "
      f"institutions behind d_f ({N_BOOT_R} draws, one SeedSequence child per field).")
    a("- **Disattenuation (after the amendment).** On the normal-score scale r = 2 sin(pi rho / 6): y_c = r / sqrt(R_f), "
      "no cap (a value above 1 is a noisy estimate of a true value <= 1 and is kept); rho_c = (6/pi) asin(y_c / 2) for "
      "display (extended monotonically above 1). The docstring's original z-scale plug-in is undefined where "
      f"r / sqrt(R) >= {CAP} ({', '.join(_lab(x) for x in R['zc_undefined'])}); it is kept as the V2z sensitivity "
      "without those fields.")
    a(f"- **Noise propagation.** {N_DRAW} draws; every program median perturbed y* = y + SE x N(0,1), rho recomputed; "
      "v_noise_f = variance of atanh(rho*) over draws. In each draw the corrected coupling uses the reliability of the "
      "twice-noisy data (V - m)/(V + m). The noise draws are refit through the full three-level model.")
    a("- **Variants.** V0z = scripts/57 (z, 1/(n-3)). V0r = r with G^2/(n-3), G = dr/dz. V1r = r with G^2[1/(n-3) + "
      "v_noise]. V2 (PRIMARY) = y_c with G^2[1/(n-3) + v_noise]/R + r^2 Var(R)/(4 R^3). V2a = y_c with G^2/(n-3)/R + the "
      "same Var(R) term. V2b = y_c with the reliability-bootstrap variance of y_c + the noise term. T1 = I2(V2)/I2(V0r) "
      "(same scale); reading >= 0.75 most remains, 0.25-0.75 part remains, < 0.25 most is noise. T2 = Spearman(y_c, r) "
      "(= Spearman(y_c, z)); survives if the estimate and the two-way lower bound are both >= 0.80. T3 = scripts/52 bands "
      f"mapped to r ({INT_T} -> {R_HI:.4f}, {DEC_T} -> {R_LO:.4f}); a flip is robust if the field is in the new band in "
      ">= 95% of noise draws.")
    a(f"- **Three-level model.** scripts/57's REML (imported, not edited): z_f = mu + u_CIP2 + v_field + e_f with the "
      f"variant's outcome and known variance; profile-likelihood CIs; LR_c against the field-only model with the 0.5 "
      f"chi2_0 + 0.5 chi2_1 mixture and scripts/57's first {N_PERM} size-preserving label permutations; log n as a fixed "
      f"moderator; power from a parametric bootstrap of the field-only fit ({pw['V2']['n_pb']} null draws, "
      f"{pw['V2']['n_pow']} per scenario). Correlated errors: scripts/57's joint-bootstrap correlation carried to each "
      f"scale ({R['JB']['n_ok']} usable joint draws).")
    a(f"- **Two-way inference.** scripts/59's `twoway()` (imported): V_field = jackknife over the 57 fields; V_inst = "
      f"variance over {N_TW} draws of one institution multinomial shared by all fields ({R['twinfo']['NU']} institutions); "
      f"V_field x inst = variance over {N_TW} draws with an independent institution draw per field; normal 95% CI. "
      f"Degenerate draws (fewer than 3 distinct values) were redrawn: {R['twinfo']['red_shared']} shared, "
      f"{R['twinfo']['red_indep']} independent.")
    a("- **Exploratory.** SE variants: lognormal (sigma of log earnings from the PSEO log IQR), national (Scorecard's "
      "national CIP-4 p25/p75, which includes between-program spread), pseoprog (the program's own PSEO relative IQR where "
      "the program is in PSEO), dp_mid / dp_hi (the primary SE plus, in quadrature, the differential-privacy perturbation "
      "of the released median: relative SD sqrt(sum p_k c_k^2) from the documentation's Exhibit 2 shares p = 0.64 / "
      "0.31 / 0.05 of medians perturbed by < 1%, 1-4% and 5-9%, c_k = bucket midpoints or upper edges; the reliability "
      "bootstrap of these variants resamples institutions only). SIMEX: noise added at lambda = 0.5/1/1.5/2 and a quadratic in lambda extrapolated to -1. "
      "PSEO panel: scripts/59's balanced fixed-cohort panel, each program's SE from its own p25/p75/count, reliability "
      f"and disattenuated coupling per cell x horizon; the correction uses cells with observed R >= {R_GATE} at every "
      f"horizon, resampled R floored at {R_FLOOR}; two-way CI (V_field = variance of the per-field values / K; shared "
      "and per-field institution draws; only degenerate draws redrawn).")
    a("")
    # ---------------- caveats ----------------
    a("## Caveats")
    a("")
    a(f"- **The SE of a median is an approximation.** {MED_SE:.4f} sigma / sqrt(n) holds for a large sample from a normal "
      "distribution; earnings are right-skewed, so with the IQR-based sigma it is a first-order approximation (the "
      "lognormal variant is one check). Scorecard medians carry no released spread; the field's PSEO spread is borrowed, "
      "and PSEO covers a public-skewed subset of institutions and a different earnings concept (annual earnings of "
      "graduates with covered employment, vs Scorecard's working-not-enrolled medians).")
    a("- **Only sampling noise is modelled in the primary SE.** The Scorecard FoS medians are released with "
      "differential-privacy noise (documentation, 'Privacy protection'); the dp variants add it using the published "
      "distribution of the perturbation, which is given in buckets and pooled over all credential levels, so its second "
      "moment is approximate. Medians whose perturbation exceeded an undisclosed threshold were suppressed, which "
      "removes the noisiest cells from the sample. PSEO's own disclosure protection, cohort-composition differences and the "
      "earnings-count definition are not modelled; if they add noise, the reliabilities here are upper bounds and the "
      "correction is too small.")
    a("- **V1/V2 partly double-count.** 1/(n-3) is the sampling variance of an observed correlation whose earnings "
      "already contain the noise; adding v_noise treats the institutions as fixed and the medians as re-drawn. This makes "
      "V1r and V2 conservative (lower I2). V2a omits the added term; both are reported.")
    a("- **Disattenuation assumes classical errors.** Independent of prestige and of the true median, bivariate normality "
      "on the normal-score scale. Rank-based coupling is corrected through its Pearson counterpart; with n_f as small as "
      f"{int(T57.n.min())} the correction is itself noisy (the Var(R) term carries this).")
    a("- **Two-way intervals hold reliabilities at their estimates.** Reliability uncertainty enters through the V2 "
      "variance and is shown separately by the reliability-bootstrap distributions; it is not added to the two-way CI.")
    a("- **Descriptive.** Nothing here identifies why coupling differs across fields; it only bounds how much of the "
      "observed difference sampling noise in earnings medians could produce.")
    a("")
    # ---------------- appendices ----------------
    a("## Appendix A — per field (57 fields, sorted by observed coupling)")
    a("")
    a("R CI = reliability bootstrap 95%. rho_c = disattenuated coupling on the rho scale. noise share = v_noise / "
      "(1/(n-3) + v_noise) on z. group = scripts/52 class (T3).")
    a("")
    a("| field | CIP-2 | n | med. earners | med. rel. SE | d_f | R_f | R 95% CI | rank rel. | rho | rho_c | y_c | noise share | group |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f, t in T57.sort_values("rho", ascending=False).iterrows():
        a(f"| {_lab(f)} | {t.cip2} | {int(t.n)} | {_f(t.med_cohort, 0)} | {_f(100 * t.med_relse, 1)}% | {_f(t.d_rel, 3)} "
          f"| {_f(t.R, 3)} | {_ci(t.R_lo, t.R_hi, 3)} | {_f(t.rank_rel, 3)} | {_f(t.rho, 3, True)} | {_f(t.rho_c, 3, True)} "
          f"| {_f(t.yc, 3, True)} | {_f(t.noise_share_z, 3)} | {t.group52} |")
    a("")
    a("## Appendix B — T3 classified fields")
    a("")
    a("| field | n | rho | R_f | R 95% CI | rho_c | base | new | flip | share of noise draws in new band | share of reliability draws with a band change | R needed for next band up |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f, t in T3.iterrows():
        a(f"| {_lab(f)} | {int(t.n)} | {_f(t.rho, 3, True)} | {_f(t.R, 3)} | {_ci(t.R_lo, t.R_hi, 3)} | {_f(t.rho_c, 3, True)} "
          f"| {t.base} | {t.new} | {'yes' if t.flip else 'no'} | {_f(t.share_draws_new, 3)} | {_f(t.share_Rboot_changed, 3)} "
          f"| {_f(t.R_needed, 3)} |")
    a("")
    a("## Appendix C — PSEO fixed-cohort panel per field (exploratory; mean over the field's cohorts)")
    a("")
    a(f"All-cell columns use every balanced cell; gated columns use cells with observed R >= {R_GATE} at every horizon "
      f"(n/a: no such cell). Cells dropped by the gate: {', '.join(pi['dropped']) if pi['dropped'] else 'none'}.")
    a("")
    a("| field | career slope rho (all) | R y1 | R y5 | R y10 | career slope rho (gated) | career slope rho_c (gated) |")
    a("|---|---|---|---|---|---|---|")
    for f, t in R["pan_fields"].iterrows():
        a(f"| {_lab(f)} | {_f(t.slope_rho, 4, True)} | {_f(t.R_y1, 3)} | {_f(t.R_y5, 3)} | {_f(t.R_y10, 3)} "
          f"| {_f(t.slope_rho_g, 4, True)} | {_f(t.slope_rhoc_g, 4, True)} |")
    a("")
    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
