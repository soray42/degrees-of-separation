"""Licensing battery: how robust is "prestige carries no pay information beyond geography in
licensed / local fields" once overfitting and selectivity are handled?

The licensing channel is the paper's one clean field-level channel. This script reproduces the
published numbers and puts every specification side by side. Descriptive; nothing here is causal.

(a) b_licensure side by side, plus a one-change-at-a-time bridge between the specifications
    - scripts/20 primary: WLS (1/SE^2, SE floor 0.08) gap ~ licensure_strict + absorption_acs,
      field bootstrap (B=2000, seed 11)                                   published +0.66 [+0.33,+0.97]
    - scripts/20 broad licensure                                          published +0.45 [+0.05,+0.72]
    - scripts/13 multilevel: REML gap ~ z(dispersion) + license + (1|CIP-2), SE floor 0.05,
      license = hand-coded 4-field binary tag (src/crosswalks/fields.LICENSED_FIELDS)
                                                                          published +0.063 [-0.05,+0.18]
    The bridge changes one thing per step (sample/SE floor, dispersion covariate, CIP-2 random
    intercept, regressor), so the reader can see which step moves the coefficient. A common scale
    is reported: the implied nursing-minus-computer-science gap difference b * (x_nursing - x_CS).
(b) scripts/35c incremental-R^2 decomposition of within-field earnings on field prestige F vs
    geography, redone with ADJUSTED R^2 and leave-one-out CROSS-VALIDATED R^2, three geography codings
    (state FE as in 35c; BEA-region FE from the Scorecard REGION column; a 1-df state earnings level,
    ST_EARN from scripts/55) and two samples (35c's PSEO-listed institutions; all matched
    institutions with the Scorecard state). Headline statistic: Spearman(licensure_strict,
    incremental prestige R^2) across fields, 35c's bootstrap (B=5000, seed 7). Nursing's partial
    r(F, earnings | geography) is reported in all 6 sample x coding versions, with its rank among fields
    (signed, |r|, and among fields with >= 10 residual df), so no single coding carries the answer, and
    compared with biology / CS / economics / accounting by Fisher-z tests (levels, log and rank scales).
    Because incremental R^2 = partial r^2 x (1 - R^2(geo)) is sign-blind and shrinks where geography
    explains a lot, the cross-field gradient is also computed on the signed partial r:
    Spearman(licensure, partial r), all 6 versions, with drop-field subsets in (c).
(c) robustness: drop nursing and communication disorders; leave-one-field-out ranges; accounting
    (licensed, CPA) and special education (licensed, K-12) as counter-cases.
(d) NEW: does the licensing gradient survive selectivity adjustment? Outcomes = per-field partial
    couplings / within-institution slopes from scripts/55 (data/interim/selectivity_fields.csv).
    Licensure is an external ACS field attribute, so its slope on a partial coupling is not inflated
    by the shared-noise problem that affects Spearman(baseline, partial).
(e) UK nursing (scripts/41) redone with adjusted and LOO-CV R^2.

Revision (verifier issues): (1) nursing is also compared after selectivity adjustment (scripts/55 partial
couplings and within-institution slopes; nursing_after_selectivity); (2)/(5) "pay set by setting" is tested with a
measured proxy, the government / nonprofit employment share from the raw ACS PUMS COW column (acs_setting_shares,
section_setting), incl. rank partial correlations with the strict share; (3) every field-level Spearman in (b)-(d)
also gets a CIP-2 cluster-bootstrap CI (B=5000), the WLS gap slopes a CIP-2 cluster bootstrap, and the key
gradients a within-CIP-2 rank association with a within-family permutation test, leave-one-family-out ranges and a
without-engineering version (cluster_diag); (4) broad-share results are reported next to the strict share.

Inputs (all public, all already in the repo; no new downloads): Wapman ranks, College Scorecard
Field-of-Study + Institution files, ACS PUMS 2023 anchors (data/interim/acs_occ_anchors.parquet,
scripts/20), data/interim/valuation_residuals.csv (scripts/30), PSEO institution list, UK LEO +
ORCID UK hiring edges (scripts/40-41 caches), data/interim/selectivity_fields.csv (scripts/55).

Seeded; outputs byte-identical on re-run.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/64_licensing_battery.py
Outputs: data/interim/licensing_battery.csv, outputs/figures/licensing_battery.png,
         LICENSING_BATTERY_RESULT.md (repo root, local only).
"""
from __future__ import annotations

import os
import sys
import hashlib
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, norm
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)            # scripts/13 builds its output path relative to the cwd at import time
warnings.filterwarnings("ignore")

from src.crosswalks import fields as F
from src.crosswalks.institutions import normalize_institution_name
from src.load_ar import load_ar_wapman
from src.load_er import load_er_scorecard
from src.gap import compute_gap_map
from src import dispersion as DSP


def _imp(name: str, fn: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fn)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s28 = _imp("s28", "28_field_vs_generic_prestige.py")
s20 = _imp("s20", "20_mixture_decomp.py")
s13 = _imp("s13", "13_granularity.py")
s35c = _imp("s35c", "35c_licensing_compression.py")
s55 = _imp("s55", "55_selectivity.py")
s41 = _imp("s41", "41_crossnational_licensing.py")      # loads scripts/40 as s41.s40
FIELDS66, LAB = s28.FIELDS66, s28.LAB

SEED = 64
B_BRIDGE = 1000          # field-bootstrap draws for the bridge rows (rows 1-2 reuse scripts/20's B=2000)
B_SLOPE = 1000           # field-bootstrap draws for WLS slopes in (d)
# Replicate counts above 1000 come from the published procedures this battery reproduces draw-for-draw:
# scripts/20's bootstrap (B=2000, seed 11; rows A1-A2 only) and scripts/35c's Spearman bootstrap (B=5000,
# seed 7), which bs() uses for EVERY field-level Spearman CI so that all correlations share one procedure
# and one set of draws. Both run on <= 47 field-level values (vectorised, a few MB). bs() also records the CI
# from the first B_MC = 1000 draws of the same seed; main() reports how much that would change (MC list).
B_MC = 1000
MC: list[tuple] = []     # (r, lo, hi, lo_1000, hi_1000) for every bs() call
# CIP-2 cluster bootstrap (revision): the licensing gradient is a contrast between disciplines, so fields in
# the same CIP-2 family are not independent draws. Every field-level Spearman that is given field keys also
# gets a CI from resampling whole CIP-2 families (B_CL draws, seed SEED_CL), plus a within-CIP-2 rank
# association with a within-family permutation test and leave-one-family-out ranges (cluster_diag).
B_CL = 5000             # raised from 1000: at B = 1000 a second seed moved cluster CI bounds by up to 0.12 and
#                          flipped "excludes 0" in 38 of 646 CIs (first revision run), too noisy for the counts the
#                          report relies on. The vectorised Spearman version holds a 5000 x k matrix (k <= 47).
B_PERM = 1000            # within-family permutation tests
SEED_CL = 64
ENG = "14"               # CIP-2 engineering family
MC_CL: list[tuple] = []  # (r, clo, chi, clo_seed2, chi_seed2) for every cluster-bootstrap CI
ACS_RAW = sorted((ROOT / "data" / "raw" / "acs").glob("psam_pus*.csv"))
MIN_INST = s35c.MIN_INST  # 15, as scripts/35c
THIN = s35c.THIN_INST     # 20
INTERIM = ROOT / "data" / "interim"
OUT_CSV = INTERIM / "licensing_battery.csv"
OUT_FIG = ROOT / "outputs" / "figures" / "licensing_battery.png"
OUT_MD = ROOT / "LICENSING_BATTERY_RESULT.md"
INST_FILE = ROOT / "data" / "raw" / "scorecard_inst" / "Most-Recent-Cohorts-Institution.csv"
PSEO_INST = ROOT / "data" / "raw" / "pseo" / "pseo_all_institutions.csv"
CIP2 = {f["key"]: f["cip4"][0][:2] for f in F.ALL_FIELDS}
DROP_NCD = ["nursing", "communication_disorders"]
PREMED = ["biology", "biochemistry", "microbiology", "physiology", "neuroscience"]   # strict share inflated by MDs
PUBLISHED = {  # the numbers this battery must reproduce (source in brackets)
    "s20_strict": (0.66, 0.33, 0.97),     # outputs/MIXTURE_DECOMP_RESULT.md
    "s20_broad": (0.45, 0.05, 0.72),      # outputs/MIXTURE_DECOMP_RESULT.md
    "s13_binary": (0.063, -0.051, 0.178),  # outputs/GRANULARITY_RESULT.md
    "35c_corr": (-0.59, -0.80, -0.30),    # data/interim/er_axis_c.md
}
PAL = {"blue": "#0F4D92", "blue2": "#3775BA", "red": "#B64342", "red2": "#E9A6A1",
       "green": "#8BCF8B", "teal": "#42949E", "violet": "#9A4D8E", "grey": "#9A9A9A",
       "light": "#CFCECE"}

ROWS: list[dict] = []    # long-format output table


def rec(section, spec, sample, stat, value, lo=np.nan, hi=np.nan, n=np.nan, field="", note="",
        clo=np.nan, chi=np.nan):
    ROWS.append(dict(section=section, spec=spec, sample=sample, field=field, stat=stat,
                     value=float(value) if value is not None else np.nan,
                     ci_lo=float(lo), ci_hi=float(hi), ci_lo_cip2=float(clo), ci_hi_cip2=float(chi),
                     n=float(n) if n is not None else np.nan, note=note))


def recb(section, spec, sample, stat, r, field="", note=""):
    """Record a bs() result: field-bootstrap CI in ci_lo/ci_hi, CIP-2 cluster-bootstrap CI in *_cip2."""
    rec(section, spec, sample, stat, r[0], r[1], r[2], r[3], field=field, note=note,
        clo=getattr(r, "clo", np.nan), chi=getattr(r, "chi", np.nan))


def fmt(x, d=2, sign=True):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def fci(lo, hi, d=2):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    return f"[{lo:+.{d}f}, {hi:+.{d}f}]"


class BSR(tuple):
    """(r, lo, hi, n) from the field bootstrap. Attributes: .clo/.chi = 95% CI of the same Spearman from a
    CIP-2 cluster bootstrap (NaN when bs() got no field keys), .g = number of CIP-2 families."""
    def __new__(cls, r, lo, hi, n, clo=np.nan, chi=np.nan, g=0):
        t = super().__new__(cls, (r, lo, hi, n))
        t.clo, t.chi, t.g = clo, chi, g
        return t

    def __getnewargs__(self):
        return tuple(self) + (self.clo, self.chi, self.g)


def cip2_of(fields):
    return np.asarray([CIP2.get(f, f) for f in np.asarray(fields, dtype=object)], dtype=object)


def bs(x, y, B=5000, seed=7, fields=None):
    """Spearman with scripts/35c's field bootstrap (B=5000, seed 7), vectorised. Same draws, same
    >2-distinct-values filter and the same average-rank Spearman as s35c.boot_spearman, so the result
    equals it to floating-point precision (checked in main()). With `fields` (the field keys, aligned with
    x and y) it also returns the CIP-2 cluster-bootstrap CI. Returns BSR (r, lo, hi, n)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]; n = len(x)
    if n < 5:
        return BSR(np.nan, np.nan, np.nan, n)
    r = spearmanr(x, y)[0]
    lo, hi = _bs_ci(x, y, B, seed)
    if B > B_MC:                    # Monte Carlo check: the same procedure with B_MC draws
        MC.append((r, lo, hi) + _bs_ci(x, y, B_MC, seed))
    if fields is None:
        return BSR(r, lo, hi, n)
    cl = cip2_of(fields)[ok]
    clo, chi, g = _bs_cluster_ci(x, y, cl)
    MC_CL.append((r, clo, chi) + _bs_cluster_ci(x, y, cl, B_CL, SEED_CL + 1)[:2])   # Monte Carlo check
    return BSR(r, lo, hi, n, clo, chi, g)


def _spear(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / den) if den > 0 else np.nan


def _bs_cluster_ci_loop(x, y, cl, B=B_CL, seed=SEED_CL):
    """Reference (loop) version of the CIP-2 cluster bootstrap; used only to check _bs_cluster_ci."""
    labs, inv = np.unique(cl.astype(str), return_inverse=True)
    G = len(labs)
    members = [np.flatnonzero(inv == j) for j in range(G)]
    D = np.random.default_rng(seed).integers(0, G, (B, G))
    out = np.full(B, np.nan)
    for b in range(B):
        idx = np.concatenate([members[j] for j in D[b]])
        xb, yb = x[idx], y[idx]
        if len(np.unique(xb)) <= 2 or len(np.unique(yb)) <= 2:
            continue
        out[b] = _spear(xb, yb)
    return out


def _wranks(v, W):
    """Average ranks of each field's value in the expanded sample in which field i appears W[b, i] times."""
    lt = (v[None, :] < v[:, None]).astype(float)       # lt[i, j] = v_j < v_i
    eq = (v[None, :] == v[:, None]).astype(float)
    return W @ lt.T + (W @ eq.T + 1.0) / 2.0


def _bs_cluster_draws(x, y, cl, B=B_CL, seed=SEED_CL):
    """CIP-2 cluster bootstrap of Spearman, vectorised: draw G families with replacement and keep all their
    fields. A draw is represented by each field's multiplicity; Spearman on the expanded sample is the
    multiplicity-weighted Pearson correlation of the expanded-sample average ranks. Draws with <= 2 distinct x or
    y values are skipped (NaN), as in the field bootstrap. Same draws as _bs_cluster_ci_loop."""
    labs, inv = np.unique(cl.astype(str), return_inverse=True)
    G = len(labs)
    D = np.random.default_rng(seed).integers(0, G, (B, G))
    cnt = np.zeros((B, G))
    np.add.at(cnt, (np.repeat(np.arange(B), G), D.ravel()), 1.0)
    W = cnt[:, inv]
    RX, RY = _wranks(x, W), _wranks(y, W)
    sw = W.sum(1, keepdims=True)
    mx, my = (W * RX).sum(1, keepdims=True) / sw, (W * RY).sum(1, keepdims=True) / sw
    cov = (W * (RX - mx) * (RY - my)).sum(1)
    den = np.sqrt((W * (RX - mx) ** 2).sum(1) * (W * (RY - my) ** 2).sum(1))
    r = np.where(den > 0, cov / np.where(den > 0, den, 1.0), np.nan)
    ux = np.unique(x, return_inverse=True)[1]; uy = np.unique(y, return_inverse=True)[1]
    dx = ((W @ np.eye(ux.max() + 1)[ux]) > 0).sum(1); dy = ((W @ np.eye(uy.max() + 1)[uy]) > 0).sum(1)
    r[(dx <= 2) | (dy <= 2)] = np.nan
    return r, G


def _bs_cluster_ci(x, y, cl, B=B_CL, seed=SEED_CL):
    out, G = _bs_cluster_draws(x, y, cl, B, seed)
    if np.isfinite(out).sum() < 100:
        return np.nan, np.nan, G
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5)), G


def cluster_diag(x, y, fields, B=B_PERM, seed=SEED_CL):
    """Discipline structure of a field-level Spearman(x, y).
    - within: Pearson correlation of the across-field ranks of x and y after demeaning both within CIP-2
      families (= rank regression with family fixed effects); only families with >= 2 fields contribute.
      p from permuting x within families (B draws, two-sided, (1 + #|r*| >= |r|) / (1 + B)).
    - share_within: share of the rank variance of x that lies within families.
    - leave-one-family-out: range of Spearman over dropping each family; without engineering (CIP-2 14)
      with field- and cluster-bootstrap CIs."""
    x = np.asarray(x, float); y = np.asarray(y, float); f = np.asarray(fields, dtype=object)
    ok = np.isfinite(x) & np.isfinite(y); x, y, f = x[ok], y[ok], f[ok]
    cl = cip2_of(f).astype(str)
    rx, ry = rankdata(x), rankdata(y)
    labs, inv = np.unique(cl, return_inverse=True)
    size = np.bincount(inv)
    multi = size[inv] >= 2
    mx = np.bincount(inv, rx) / size; my = np.bincount(inv, ry) / size
    wx, wy = (rx - mx[inv])[multi], (ry - my[inv])[multi]
    den = np.sqrt((wx @ wx) * (wy @ wy))
    r_w = float(wx @ wy / den) if den > 0 else np.nan
    share_within = float((wx @ wx) / (((rx - rx.mean()) ** 2).sum()))
    inv_mm = inv[multi]
    contrib = {str(labs[j]): float((wx[inv_mm == j] ** 2).sum() / (wx @ wx)) for j in np.unique(inv_mm)} \
        if (wx @ wx) > 0 else {}
    p_w = np.nan
    if np.isfinite(r_w):
        rng = np.random.default_rng(seed)
        inv_m = inv[multi]
        groups = [np.flatnonzero(inv_m == j) for j in np.unique(inv_m)]
        cnt = 0
        for _ in range(B):
            wp = wx.copy()
            for gi in groups:
                wp[gi] = wx[gi][rng.permutation(len(gi))]
            rp = float(wp @ wy / den)
            cnt += abs(rp) >= abs(r_w) - 1e-12
        p_w = (1 + cnt) / (1 + B)
    loco = []
    for c in labs:
        k = cl != c
        if k.sum() >= 5:
            loco.append((spearmanr(x[k], y[k])[0], c))
    lv = np.array([v for v, _ in loco])
    j0, j1 = int(np.nanargmin(lv)), int(np.nanargmax(lv))
    ne = cl != ENG
    no_eng = bs(x[ne], y[ne], fields=f[ne])
    return dict(r_within=r_w, p_within=p_w, k_within=int(multi.sum()), g_within=int((size >= 2).sum()),
                share_within=share_within, contrib=contrib, loco_min=float(lv[j0]), loco_min_c=loco[j0][1],
                loco_max=float(lv[j1]), loco_max_c=loco[j1][1], no_eng=no_eng, k=len(x), g=len(labs))


def rec_diag(section, spec, sample, name, d, field=""):
    rec(section, spec, sample, f"{name}_within_cip2", d["r_within"], n=d["k_within"], field=field,
        note=f"rank association after demeaning ranks within CIP-2 families ({d['g_within']} families with >= 2 "
             f"fields); within-family permutation p={d['p_within']:.4f} (B={B_PERM}); share of x rank variance "
             f"within families={d['share_within']:.3f}")
    rec(section, spec, sample, f"{name}_leave_one_cip2_out_min", d["loco_min"], n=d["k"], field=d["loco_min_c"],
        note="field column = CIP-2 family left out")
    rec(section, spec, sample, f"{name}_leave_one_cip2_out_max", d["loco_max"], n=d["k"], field=d["loco_max_c"],
        note="field column = CIP-2 family left out")
    recb(section, spec, sample, f"{name}_without_engineering_cip14", d["no_eng"], field=field,
         note="field bootstrap CI in ci_lo/ci_hi; CIP-2 cluster bootstrap CI in *_cip2")


def _bs_ci(x, y, B, seed):
    n = len(x)
    I = np.random.default_rng(seed).integers(0, n, (B, n))   # = B sequential integers(0, n, n) draws
    X, Y = x[I], y[I]
    Xs, Ys = np.sort(X, 1), np.sort(Y, 1)
    keep = ((Xs[:, 1:] != Xs[:, :-1]).sum(1) + 1 > 2) & ((Ys[:, 1:] != Ys[:, :-1]).sum(1) + 1 > 2)
    rx = pd.DataFrame(X[keep]).rank(axis=1).to_numpy(); ry = pd.DataFrame(Y[keep]).rank(axis=1).to_numpy()
    rx = rx - rx.mean(1, keepdims=True); ry = ry - ry.mean(1, keepdims=True)
    den = np.sqrt((rx * rx).sum(1) * (ry * ry).sum(1))
    b = np.where(den > 0, (rx * ry).sum(1) / np.where(den > 0, den, 1.0), np.nan)
    return (float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5)))


def wls_slope_boot(x, y, se, B, seed):
    """Field bootstrap of the one-regressor WLS slope (weights 1/SE^2), vectorised. Draws with fewer
    than 3 distinct x values are skipped. Returns the array of slopes."""
    I = np.random.default_rng(seed).integers(0, len(x), (B, len(x)))
    X, Y, W = x[I], y[I], 1.0 / se[I] ** 2
    Xs = np.sort(X, 1)
    keep = (Xs[:, 1:] != Xs[:, :-1]).sum(1) + 1 > 2
    X, Y, W = X[keep], Y[keep], W[keep]
    xm = (W * X).sum(1) / W.sum(1); ym = (W * Y).sum(1) / W.sum(1)
    return (W * (X - xm[:, None]) * (Y - ym[:, None])).sum(1) / (W * (X - xm[:, None]) ** 2).sum(1)


def young_licensure():
    """Strict licensure share among 22-27-year-old full-time BA+ workers (closest to the early-career
    population behind Scorecard's 4-year earnings), from scripts/35's cached ACS occupation extract
    (data/interim/acs_occ_dist.parquet), with scripts/20's strict SOC prefixes."""
    o = pd.read_parquet(INTERIM / "acs_occ_dist.parquet")
    soc = o.soc6.fillna("").str.replace("X", "0", regex=False)
    lic = np.zeros(len(o), bool)
    for pre in s20.LIC_STRICT:
        lic |= soc.str.startswith(pre).to_numpy()
    o = o.assign(lic=lic.astype(float))
    g = o.groupby("field", sort=True)
    return pd.DataFrame({"lic_young": g.apply(lambda d: np.average(d.lic, weights=d.PWGTP)),
                         "n_young": g.size()}).reset_index()


def acs_setting_shares(anch):
    """Employer-sector shares per field from the raw ACS PUMS 2023 person files (COW = class of worker), on
    exactly scripts/20's anchor population (BA+, full-time >= 35 h, employed, PERNP > 0, FOD1P mapped with
    scripts/20's code map). Read in chunks with usecols; only filtered rows are kept.
      gov   = COW 3-5 (local, state, federal government): the closest public measure of pay set by a public
              pay scale / salary schedule
      npo   = COW 2 (private not-for-profit)
      govnp = COW 2-5
    Weighted by PWGTP, all ages and ages 22-27. Checks that the per-field row counts equal scripts/20's n_acs."""
    code_map = F.fod1p_to_field_key(F.fod1p_by_key_all())
    use = ["FOD1P", "SCHL", "WKHP", "ESR", "PERNP", "PWGTP", "COW", "AGEP"]
    keep = []
    for fpath in ACS_RAW:
        for ch in pd.read_csv(fpath, usecols=use, dtype="float64", chunksize=400_000):
            ch = ch[(ch.SCHL >= 21) & ch.FOD1P.notna() & (ch.WKHP >= 35) & ch.ESR.isin([1, 2]) & (ch.PERNP > 0)]
            ch = ch.assign(field=ch.FOD1P.astype(int).astype(str).str.zfill(4).map(code_map))
            keep.append(ch[ch.field.notna()][["field", "PWGTP", "COW", "AGEP"]].copy())
            del ch
    a = pd.concat(keep, ignore_index=True)
    del keep
    a["gov"] = a.COW.isin([3, 4, 5]).astype(float)
    a["npo"] = (a.COW == 2).astype(float)
    a["govnp"] = a.COW.isin([2, 3, 4, 5]).astype(float)
    out = []
    for fld, g in a.groupby("field", sort=True):
        y = g[(g.AGEP >= 22) & (g.AGEP <= 27)]
        w = g.PWGTP.to_numpy(); wy = y.PWGTP.to_numpy()
        out.append(dict(field=fld, n_acs_check=len(g), n_young_sector=len(y),
                        **{c: float(np.average(g[c], weights=w)) for c in ("gov", "npo", "govnp")},
                        **{f"{c}_young": (float(np.average(y[c], weights=wy)) if len(y) >= 30 else np.nan)
                           for c in ("gov", "npo", "govnp")}))
    del a
    S = pd.DataFrame(out).merge(anch[["field", "n_acs"]], on="field", how="left")
    ok = bool((S.n_acs_check == S.n_acs).all())
    return S.drop(columns=["n_acs"]), ok


# =============================================================================================
# (a) b_licensure side by side + bridge
# =============================================================================================
def build_gap_frames():
    ar = load_ar_wapman(fields=F.ALL_FIELDS)
    er = load_er_scorecard(fields=F.ALL_FIELDS)
    gm = compute_gap_map(ar, er, "undergrad")
    excl = (gm.ci_hi < gm.synth_lo) | (gm.ci_lo > gm.synth_hi)
    gm["reliable"] = (gm.n_institutions >= 10) & (gm.signal_frac >= 0.5) & excl
    gm["cip2"] = gm.field.map(CIP2)
    anch = s20.build_anchors()                                   # cached ACS PUMS 2023 anchors
    # scripts/20 sample
    a = gm.copy()
    a["se"] = ((a.ci_hi - a.ci_lo) / (2 * 1.96)).clip(lower=0.08)
    degen = (a.gap <= 0.02) | (a.gap >= 1.6) | (a.n_institutions < 8) | ((a.ci_hi - a.ci_lo) < 0.02)
    a = a[~degen].dropna(subset=["gap"]).merge(anch, on="field", how="left")
    a = a.dropna(subset=["gap", "licensure_strict", "absorption_acs"]).copy()
    # scripts/13 sample
    b = gm.copy()
    b["se"] = ((b.ci_hi - b.ci_lo) / (2 * 1.96)).clip(lower=0.05)
    w = DSP.load_acs_workers(expanded=True)
    disp = DSP.field_dispersion_acs(w, 22, 27).rename(columns={"disp_acs": "disp"})[["field", "disp"]]
    b = b.merge(disp, on="field", how="left").dropna(subset=["gap", "disp"]).copy()
    b["z"] = (b.disp - b.disp.mean()) / b.disp.std()
    b = b.merge(anch, on="field", how="left")
    for d in (a, b):
        d["lic_bin"] = d.field.map(F.is_licensed).astype(int)
        d["lic_bin50"] = (d.licensure_strict >= 0.5).astype(int)
        d["label"] = d.field.map(LAB)
    return gm, a, b, anch


def fit_wls(d, xcols):
    m = sm.WLS(d.gap.values, sm.add_constant(d[xcols].values, has_constant="add"),
               weights=1 / d.se.values ** 2).fit(cov_type="HC1")
    return float(m.params[-1]), float(m.bse[-1])


def fit_reml(d, xcols):
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in xcols])
    beta, se, tau2, _ = s13.reml_metareg(d.gap.values, X, d.se.values, d.cip2.values)
    return float(beta[-1]), float(se[-1]), float(tau2)


def fit(d, xcols, est):
    if est == "wls":
        return fit_wls(d, xcols)
    b, s, _ = fit_reml(d, xcols)
    return b, s


def boot_fit(d, xcols, est, B, seed):
    rng = np.random.default_rng(seed)
    out, fail = [], 0
    for _ in range(B):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        if s[xcols[-1]].nunique() < 2:
            fail += 1
            continue
        try:
            out.append(fit(s, xcols, est)[0])
        except Exception:
            fail += 1
    arr = np.array(out)
    return np.percentile(arr, 2.5), np.percentile(arr, 97.5), fail


def boot_fit_cluster(d, xcols, B=B_CL, seed=SEED_CL):
    """CIP-2 cluster bootstrap of the WLS licensure coefficient (resample whole CIP-2 families)."""
    cl = d.cip2.astype(str).to_numpy()
    labs = np.unique(cl)
    members = {c: np.flatnonzero(cl == c) for c in labs}
    D = np.random.default_rng(seed).integers(0, len(labs), (B, len(labs)))
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in xcols])
    y, sw = d.gap.to_numpy(float), 1.0 / d.se.to_numpy(float)       # sqrt of the WLS weights 1/SE^2
    xl = d[xcols[-1]].to_numpy(float)
    out = []
    for b in range(B):
        idx = np.concatenate([members[labs[j]] for j in D[b]])
        if len(np.unique(xl[idx])) < 3:
            continue
        c, _, rk, _ = np.linalg.lstsq(X[idx] * sw[idx, None], y[idx] * sw[idx], rcond=None)
        if rk == X.shape[1]:
            out.append(c[-1])          # WLS coefficient (same as fit_wls's params[-1])
    arr = np.array(out)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), len(labs)


BRIDGE = [
    # key, label, sample, regressors (licensure last), estimator
    ("A1", "scripts/20 primary: WLS, strict share + absorption", "s20", ["absorption_acs", "licensure_strict"], "wls"),
    ("A2", "scripts/20 broad: WLS, broad share + absorption", "s20", ["absorption_acs", "licensure_broad"], "wls"),
    ("A3", "step 1: drop absorption (strict share only)", "s20", ["licensure_strict"], "wls"),
    ("A4", "step 2: scripts/13 sample (SE floor 0.05, dispersion non-missing)", "s13", ["licensure_strict"], "wls"),
    ("A5", "step 3: + z(dispersion)", "s13", ["z", "licensure_strict"], "wls"),
    ("A6", "step 4: + CIP-2 random intercept (REML) = scripts/13 model, strict share", "s13", ["z", "licensure_strict"], "reml"),
    ("A7", "scripts/13 model, broad share", "s13", ["z", "licensure_broad"], "reml"),
    ("A8", "step 5: swap to the 4-field binary tag = scripts/13 published", "s13", ["z", "lic_bin"], "reml"),
    ("A9", "binary tag, WLS + z(dispersion), no random intercept", "s13", ["z", "lic_bin"], "wls"),
    ("A10", "binary tag in scripts/20 WLS (+ absorption)", "s20", ["absorption_acs", "lic_bin"], "wls"),
    ("A11", "scripts/13 model, ACS-binarised tag (strict share >= 0.5)", "s13", ["z", "lic_bin50"], "reml"),
]


def section_a(gm, a, b):
    frames = {"s20": a, "s13": b}
    nurse_cs_obs = float(gm.set_index("field").gap["nursing"] - gm.set_index("field").gap["computer_science"])
    res = []
    for i, (key, lab, samp, xc, est) in enumerate(BRIDGE):
        d = frames[samp]
        bhat, se = fit(d, xc, est)
        if key in ("A1", "A2"):             # scripts/20's own bootstrap, for exact reproduction
            A = s20.boot_coefs(d, xc[::-1])   # its column order is (licensure, absorption)
            lo, hi = np.nanpercentile(A[:, 1], [2.5, 97.5]); fail = 0
            B_used = 2000
        else:
            lo, hi, fail = boot_fit(d, xc, est, B_BRIDGE, SEED + i)
            B_used = B_BRIDGE
        clo = chi = np.nan
        if est == "wls" and not xc[-1].startswith("lic_bin"):   # CIP-2 cluster bootstrap (continuous shares)
            clo, chi, _ = boot_fit_cluster(d, xc, B_CL, SEED_CL)
        xcol = xc[-1]
        xv = d.set_index("field")[xcol]
        dx = float(xv.get("nursing", np.nan) - xv.get("computer_science", np.nan))
        sdx = float(d[xcol].std())
        # robustness: drop nursing + communication disorders; leave-one-field-out range
        dd = d[~d.field.isin(DROP_NCD)]
        try:
            b_drop = fit(dd, xc, est)[0] if dd[xcol].nunique() > 1 else np.nan
        except Exception:
            b_drop = np.nan
        loo = []
        for f in d.field:
            s = d[d.field != f]
            loo.append((fit(s, xc, est)[0] if s[xcol].nunique() > 1 else np.nan, f))
        loo_v = np.array([v for v, _ in loo], float)
        jmin = int(np.nanargmin(loo_v)); jmax = int(np.nanargmax(loo_v))
        r = dict(key=key, label=lab, sample=samp, regressor=xcol, est=est, n=len(d),
                 n_lic=int(d[xcol].sum()) if xcol.startswith("lic_bin") else np.nan,
                 b=bhat, se=se, lo_model=bhat - 1.96 * se, hi_model=bhat + 1.96 * se,
                 lo=lo, hi=hi, clo=clo, chi=chi, B=B_used, boot_fail=fail, dx=dx, sdx=sdx,
                 implied=bhat * dx, implied_lo=lo * dx, implied_hi=hi * dx, std=bhat * sdx,
                 b_drop=b_drop, n_drop=len(dd), loo_min=loo_v[jmin], loo_min_f=loo[jmin][1],
                 loo_max=loo_v[jmax], loo_max_f=loo[jmax][1],
                 b_no_accounting=dict((f, v) for v, f in loo).get("accounting", np.nan))
        res.append(r)
        sp = f"{key} {lab}"
        rec("a_blicensure", sp, samp, "b", bhat, lo, hi, len(d), note=f"{est}; regressor {xcol}; field bootstrap B={B_used}"
            + (f"; CIP-2 cluster bootstrap B={B_CL} in *_cip2" if np.isfinite(clo) else ""), clo=clo, chi=chi)
        rec("a_blicensure", sp, samp, "b_model_ci", bhat, bhat - 1.96 * se, bhat + 1.96 * se, len(d), note="HC1 (WLS) or GLS (REML) SE")
        rec("a_blicensure", sp, samp, "implied_nursing_minus_cs_gap", bhat * dx, lo * dx, hi * dx, len(d), note=f"b*(x_nursing-x_CS), dx={dx:.3f}")
        rec("a_blicensure", sp, samp, "b_per_sd_regressor", bhat * sdx, n=len(d), note=f"SD(x)={sdx:.3f}")
        rec("c_robust", sp, samp, "b_drop_nursing_commdis", b_drop, n=len(dd))
        rec("c_robust", sp, samp, "b_leave_one_field_out_min", loo_v[jmin], n=len(d) - 1, field=loo[jmin][1])
        rec("c_robust", sp, samp, "b_leave_one_field_out_max", loo_v[jmax], n=len(d) - 1, field=loo[jmax][1])
        rec("c_robust", sp, samp, "b_without_accounting", r["b_no_accounting"], n=len(d) - 1, field="accounting")
    R = pd.DataFrame(res)
    rec("a_blicensure", "observed nursing - computer science gap", "gap map", "gap_difference", nurse_cs_obs)
    # which fields carry the binary tag, and their measured licensure shares
    tagged = b[b.lic_bin == 1][["field", "gap", "se", "cip2", "licensure_strict", "licensure_broad"]]
    corr_tag = float(np.corrcoef(b.lic_bin, b.licensure_strict)[0, 1])
    rec("a_blicensure", "corr(binary tag, strict share)", "s13", "pearson", corr_tag, n=len(b))
    # CIP-2 sibling contrast of each tagged field (unweighted mean gap of untagged fields in its CIP-2)
    sib = []
    for _, t in tagged.iterrows():
        s = b[(b.cip2 == t.cip2) & (b.lic_bin == 0)]
        sib.append(dict(field=t.field, cip2=t.cip2, gap=t.gap, se=t.se, lic_strict=t.licensure_strict,
                        lic_broad=t.licensure_broad, sib_gap=s.gap.mean(), n_sib=len(s),
                        sib_fields=", ".join(LAB.get(x, x) for x in s.field)))
        rec("a_blicensure", "binary-tag field vs untagged CIP-2 siblings", "s13", "gap_minus_sibling_mean",
            t.gap - s.gap.mean(), n=len(s), field=t.field)
    return R, pd.DataFrame(sib), corr_tag, nurse_cs_obs


# =============================================================================================
# (b) incremental R^2 decomposition: raw / adjusted / LOO-CV, three geography codings, two samples
# =============================================================================================
def load_cells():
    vr = pd.read_csv(INTERIM / "valuation_residuals.csv")          # scripts/30: inst x field F, earnings
    er = load_er_scorecard(fields=FIELDS66)
    v = vr.merge(er[["inst_key", "field", "institution_id"]].rename(columns={"institution_id": "UNITID"}),
                 on=["inst_key", "field"], how="left")
    v["UNITID"] = v.UNITID.astype(str)
    inst = s55.load_institutions()                                   # STABBR, ST_EARN (leave-one-out)
    reg = pd.read_csv(INST_FILE, usecols=["UNITID", "STABBR", "REGION"], dtype=str)
    s2r = (reg[reg.REGION != "0"].groupby("STABBR").REGION
           .agg(lambda x: x.value_counts().sort_index().idxmax()).to_dict())   # BEA region per state
    v = v.merge(inst[["UNITID", "STABBR", "ST_EARN", "SAT_AVG"]], on="UNITID", how="left")
    v["REGION"] = v.STABBR.map(s2r)
    ins = pd.read_csv(PSEO_INST, dtype=str)
    ins.columns = [c.strip().lstrip("﻿") for c in ins.columns]
    ins["inst_key"] = ins.label.map(normalize_institution_name)
    st = ins.dropna(subset=["institution_state"]).drop_duplicates("inst_key")[["inst_key", "institution_state"]]
    v = v.merge(st, on="inst_key", how="left")
    anch = pd.read_parquet(INTERIM / "acs_occ_anchors.parquet")[["field", "licensure_strict", "licensure_broad"]]
    v = v.merge(anch, on="field", how="left")
    both = v.dropna(subset=["STABBR", "institution_state"])
    agree = float((both.STABBR == both.institution_state).mean()) if len(both) else np.nan
    # consistency: valuation_residuals.csv equals a fresh scripts/28 build on its (inst, field) cells
    t = s28.build_table()[["inst_key", "field", "F", "earnings"]]
    chk = vr[["inst_key", "field", "F", "earnings"]].merge(t, on=["inst_key", "field"], how="left",
                                                          suffixes=("", "_new"))
    fresh_ok = bool(np.allclose(chk.F, chk.F_new) and np.allclose(chk.earnings, chk.earnings_new))
    return v, dict(cells=len(v), stabbr_missing=int(v.STABBR.isna().sum()),
                   pseo_state=int(v.institution_state.notna().sum()), state_agree=agree,
                   st_earn_missing=int(v.ST_EARN.isna().sum()), fresh_rebuild_match=fresh_ok)


def _design(geo, kind):
    if kind == "cat":
        lev, inv = np.unique(geo, return_inverse=True)
        return np.eye(len(lev))[inv]                    # one-hot, all levels (spans the constant)
    if kind == "cont":
        return np.column_stack([np.ones(len(geo)), geo.astype(float)])
    return np.ones((len(geo), 1))


def ols_fit(y, X):
    c, _, rk, _ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ c
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(e @ e) / sst
    dfr = len(y) - rk
    adj = 1 - (1 - r2) * (len(y) - 1) / dfr if dfr > 0 else np.nan
    return r2, adj, int(rk), int(dfr)


def loo_r2(y, Fv, geo, use_F, kind):
    """Leave-one-out R^2. Categorical geography: one-hot on the training fold; a held-out unit whose
    level is absent from the training fold gets the count-weighted mean of the training intercepts."""
    n = len(y)
    pred = np.empty(n)
    idx = np.arange(n)
    for i in range(n):
        tr = idx != i
        yt = y[tr]
        if kind == "cat":
            lev, inv = np.unique(geo[tr], return_inverse=True)
            X = np.eye(len(lev))[inv]
        elif kind == "cont":
            X = np.column_stack([np.ones(n - 1), geo[tr].astype(float)])
        else:
            X = np.ones((n - 1, 1))
        if use_F:
            X = np.column_stack([X, Fv[tr]])
        c = np.linalg.lstsq(X, yt, rcond=None)[0]
        if kind == "cat":
            L = len(lev)
            j = int(np.searchsorted(lev, geo[i]))
            if j < L and lev[j] == geo[i]:
                base = c[j]
            else:
                cnt = np.bincount(inv, minlength=L)
                base = float((c[:L] * cnt).sum() / cnt.sum())
            p = base + (c[L] * Fv[i] if use_F else 0.0)
        else:
            x = [1.0] + ([float(geo[i])] if kind == "cont" else []) + ([Fv[i]] if use_F else [])
            p = float(np.dot(c, x))
        pred[i] = p
    return 1 - float(((y - pred) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def partial_r(y, Fv, geo, kind):
    """Partial correlation of F and earnings given the geography design; Fisher-z 95% CI. Undefined
    (NaN) when the full model (geography + F) has no residual df: the two residual vectors then live in
    a 1-dimensional space and correlate exactly +-1 by construction."""
    G = _design(geo, kind)
    ey = y - G @ np.linalg.lstsq(G, y, rcond=None)[0]
    eF = Fv - G @ np.linalg.lstsq(G, Fv, rcond=None)[0]
    q = np.linalg.matrix_rank(G) - 1
    den = np.sqrt((ey @ ey) * (eF @ eF))
    if den <= 0 or len(y) - (q + 1) - 1 < 1:
        return np.nan, np.nan, np.nan, q
    r = float(ey @ eF / den)
    dfz = len(y) - q - 3
    if dfz < 1:
        return r, np.nan, np.nan, q
    h = 1.96 / np.sqrt(dfz)
    return r, float(np.tanh(np.arctanh(r) - h)), float(np.tanh(np.arctanh(r) + h)), q


CODINGS = {"state FE": ("geo_state", "cat"), "BEA region FE": ("REGION", "cat"),
           "state earnings level (1 df)": ("ST_EARN", "cont")}
MEASURES = ["raw", "adjusted", "LOO-CV"]


def decompose(v):
    """Per field x sample x geography coding x R^2 measure."""
    out = []
    samples = {"35c sample (PSEO-listed)": "institution_state", "full (Scorecard state)": "STABBR"}
    for samp, stcol in samples.items():
        for fld, g in v.groupby("field", sort=True):
            lic = g.licensure_strict.iloc[0]
            d0 = g.dropna(subset=["earnings", "F", stcol]).copy()
            d0["geo_state"] = d0[stcol]
            # field eligibility exactly as scripts/35c: n >= 15, >= 3 states, finite licensure
            if len(d0) < MIN_INST or d0.geo_state.nunique() < 3 or not np.isfinite(lic):
                continue
            for cname, (gcol, kind) in CODINGS.items():
                d = d0.dropna(subset=[gcol])
                y = d.earnings.to_numpy(float); Fv = d.F.to_numpy(float)
                geo = d[gcol].to_numpy(float) if kind == "cont" else d[gcol].astype(str).to_numpy()
                Xg = _design(geo, kind)
                Xp = np.column_stack([np.ones(len(y)), Fv])
                Xf = np.column_stack([Xg, Fv])
                rp, rg, rf = ols_fit(y, Xp), ols_fit(y, Xg), ols_fit(y, Xf)
                ok = rf[3] >= 1                                    # full model not saturated
                cv = {}
                if ok:
                    cv = {"p": loo_r2(y, Fv, geo, True, None), "g": loo_r2(y, Fv, geo, False, kind),
                          "f": loo_r2(y, Fv, geo, True, kind)}
                pr = partial_r(y, Fv, geo, kind)
                # scale checks: log earnings, and ranks of earnings and F (rank-based partial r)
                pr_log = partial_r(np.log(y), Fv, geo, kind) if np.all(y > 0) else (np.nan,) * 4
                pr_rk = partial_r(rankdata(y), rankdata(Fv), geo, kind)
                base = dict(sample=samp, coding=cname, field=fld, label=LAB.get(fld, fld),
                            licensure=lic, lic_broad=g.licensure_broad.iloc[0], n=len(y),
                            n_geo=(len(np.unique(geo)) if kind == "cat" else 1), dfr_full=rf[3],
                            thin=len(y) < THIN, partial_r=pr[0], partial_lo=pr[1], partial_hi=pr[2],
                            dfz=len(y) - pr[3] - 3,                     # Fisher-z df of the partial r
                            partial_r_log=pr_log[0], partial_r_rank=pr_rk[0])
                for meas in MEASURES:
                    if meas == "raw":
                        P, Gq, Fu = rp[0], rg[0], rf[0]
                    elif meas == "adjusted":
                        P, Gq, Fu = (rp[1], rg[1], rf[1]) if ok else (np.nan,) * 3
                    else:
                        P, Gq, Fu = (cv["p"], cv["g"], cv["f"]) if ok else (np.nan,) * 3
                    out.append(dict(base, measure=meas, R2_prestige_only=P, R2_geo_only=Gq, R2_full=Fu,
                                    inc_prestige=Fu - Gq, inc_geo=Fu - P,
                                    shared=P + Gq - Fu))
    D = pd.DataFrame(out)
    return D


def section_b(v):
    D = decompose(v)
    # reproduction check vs scripts/35c's saved per-field table
    ref = pd.read_csv(INTERIM / "er_axis_35c_mechanism.csv").dropna(subset=["incremental_prestige"])
    mine = D[(D["sample"] == "35c sample (PSEO-listed)") & (D.coding == "state FE") & (D.measure == "raw")]
    mm = ref.merge(mine, on="field")
    repro = dict(n_ref=len(ref), n_mine=len(mine), n_match=len(mm),
                 max_abs_diff=float(np.max(np.abs(mm.incremental_prestige - mm.inc_prestige))) if len(mm) else np.nan)
    S = []
    P35, PF_ = "35c sample (PSEO-listed)", "full (Scorecard state)"
    for (samp, cod, meas), g in D.groupby(["sample", "coding", "measure"], sort=False):
        g = g.dropna(subset=["inc_prestige"])
        row = dict(sample=samp, coding=cod, measure=meas, k=len(g))
        f35 = set(D[(D["sample"] == P35) & (D.coding == cod) & (D.measure == meas)].dropna(subset=["inc_prestige"]).field)
        gc = g[g.field.isin(f35)]
        row["inc_prestige_common35c"] = bs(gc.licensure, gc.inc_prestige, fields=gc.field)
        recb("b_increment", f"{cod} | {meas}", samp, "spearman(licensure,inc_prestige)_on_35c_fields",
             row["inc_prestige_common35c"])
        gg = g.dropna(subset=["R2_geo_only"]).sort_values("R2_geo_only", ascending=False).reset_index(drop=True)
        rk = gg.index[gg.field == "nursing"]
        row["nurse_geo_rank"] = (int(rk[0]) + 1 if len(rk) else np.nan, len(gg))
        row["top_geo"] = list(gg.field.head(3))
        rec("b_increment", f"{cod} | {meas}", samp, "nursing_rank_R2_geo_only_desc", row["nurse_geo_rank"][0],
            n=len(gg), note="1 = field whose earnings geography predicts best")
        for stat, col, sub in [("inc_prestige", "inc_prestige", g), ("inc_geo", "inc_geo", g),
                               ("R2_geo_only", "R2_geo_only", g), ("R2_prestige_only", "R2_prestige_only", g),
                               ("inc_prestige_dropNCD", "inc_prestige", g[~g.field.isin(DROP_NCD)]),
                               ("inc_prestige_nothin", "inc_prestige", g[~g.thin.astype(bool)]),
                               ("inc_prestige_dfr10", "inc_prestige", g[g.dfr_full >= 10]),
                               ("inc_geo_dfr10", "inc_geo", g[g.dfr_full >= 10]),
                               ("inc_prestige_broadlic", "inc_prestige", g)]:
            xcol = "lic_broad" if stat.endswith("broadlic") else "licensure"
            r = bs(sub[xcol], sub[col], fields=sub.field)
            row[stat] = r
            recb("b_increment", f"{cod} | {meas}", samp, f"spearman(licensure,{stat})", r,
                 note="licensure = broad share" if xcol == "lic_broad" else "licensure = strict share")
        row["cl_inc"] = cluster_diag(g.licensure, g.inc_prestige, g.field)
        rec_diag("b_increment", f"{cod} | {meas}", samp, "spearman(licensure,inc_prestige)", row["cl_inc"])
        # leave-one-field-out range of the headline correlation
        lo_vals = [spearmanr(g[g.field != f].licensure, g[g.field != f].inc_prestige)[0] for f in g.field]
        row["loo_range"] = (float(np.min(lo_vals)), float(np.max(lo_vals))) if lo_vals else (np.nan, np.nan)
        rec("c_robust", f"{cod} | {meas}", samp, "spearman(licensure,inc_prestige)_leave_one_field_out_min",
            row["loo_range"][0], n=len(g) - 1)
        rec("c_robust", f"{cod} | {meas}", samp, "spearman(licensure,inc_prestige)_leave_one_field_out_max",
            row["loo_range"][1], n=len(g) - 1)
        for f in ["nursing", "accounting", "computer_science", "economics", "special_education",
                  "communication_disorders", "biology", "social_work", "public_health"]:
            h = g[g.field == f]
            if len(h):
                h = h.iloc[0]
                row[f] = h
                for st in ("inc_prestige", "inc_geo", "R2_geo_only", "R2_prestige_only"):
                    rec("b_field", f"{cod} | {meas}", samp, st, h[st], n=h.n, field=f)
        S.append(row)
    # nursing: between-region vs within-region association of prestige and pay (BEA regions), both samples.
    # Descriptive only: a possible reason why prestige alone explains so little, not an established one.
    nurse_reg = {}
    nu_all = v[(v.field == "nursing")].dropna(subset=["earnings", "F", "REGION", "STABBR"])
    for samp, nu in (("full (Scorecard state)", nu_all),
                     ("35c sample (PSEO-listed)", nu_all[nu_all.institution_state.notna()])):
        rm = nu.groupby("REGION").agg(F=("F", "mean"), earn=("earnings", "mean"), n=("F", "size"))
        between = float(np.corrcoef(rm.F, rm.earn)[0, 1])
        within = spearmanr(nu.earnings - nu.groupby("REGION").earnings.transform("mean"),
                           nu.F - nu.groupby("REGION").F.transform("mean"))
        nurse_reg[samp] = dict(between=between, k_reg=len(rm), within=float(within[0]), within_p=float(within[1]),
                               n=len(nu), top_pay_region=str(rm.earn.idxmax()), top_pay=float(rm.earn.max()),
                               top_pay_F=float(rm.F.loc[rm.earn.idxmax()]),
                               F_rank_of_top_pay=int(rm.F.rank().loc[rm.earn.idxmax()]))
        rec("b_field", "nursing: corr across BEA regions of mean F and mean earnings", samp,
            "pearson_between_regions", between, n=len(rm), field="nursing")
        rec("b_field", "nursing: Spearman of region-demeaned F and earnings", samp,
            "spearman_within_regions", within[0], n=len(nu), field="nursing", note=f"p={within[1]:.4f}")
    # nursing's partial r(F, earnings | geography) in all 6 sample x coding versions, and where it ranks
    # among the eligible fields (partial r does not depend on the R^2 measure, so use the raw rows)
    NPR = []
    for (samp, cod), g in D[D.measure == "raw"].groupby(["sample", "coding"], sort=False):
        g = g.dropna(subset=["partial_r"]).set_index("field")
        if "nursing" not in g.index:
            continue
        h = g.loc["nursing"]
        rk = int(g.partial_r.rank(method="min")["nursing"])
        rk_abs = int(g.partial_r.abs().rank(method="min")["nursing"])
        rk_inc = int(g.inc_prestige.rank(method="min")["nursing"])
        below = list(g[g.partial_r < h.partial_r].sort_values("partial_r").index)
        below_pos = [f for f in below if g.loc[f, "partial_r"] >= 0]
        lic_of = {f: float(g.loc[f, "licensure"]) for f in below}
        g10 = g[g.dfr_full >= 10]                   # fields whose partial r is not dominated by few residual df
        rk10 = int(g10.partial_r.rank(method="min")["nursing"])
        rka10 = int(g10.partial_r.abs().rank(method="min")["nursing"])
        ci0 = bool(h.partial_lo <= 0 <= h.partial_hi)
        NPR.append(dict(sample=samp, coding=cod, r=h.partial_r, lo=h.partial_lo, hi=h.partial_hi, n=int(h.n),
                        n_geo=int(h.n_geo), dfr=int(h.dfr_full), k=len(g), rank=rk, rank_abs=rk_abs,
                        rank_inc=rk_inc, below=below, below_pos=below_pos, lic_of=lic_of, ci0=ci0,
                        k10=len(g10), rank10=rk10, rank_abs10=rka10))
        rec("b_nursing_partial", cod, samp, "rank_partial_r_asc_dfr10", rk10, n=len(g10), field="nursing",
            note="as rank_partial_r_asc, among fields whose full model has >= 10 residual df")
        rec("b_nursing_partial", cod, samp, "rank_abs_partial_r_asc_dfr10", rka10, n=len(g10), field="nursing",
            note="as rank_abs_partial_r_asc, among fields whose full model has >= 10 residual df")
        rec("b_nursing_partial", cod, samp, "partial_r(F,earn|geo)", h.partial_r, h.partial_lo, h.partial_hi,
            h.n, field="nursing", note=f"Fisher-z CI; CI includes 0: {ci0}")
        rec("b_nursing_partial", cod, samp, "rank_partial_r_asc", rk, n=len(g), field="nursing",
            note="1 = lowest partial r among eligible fields; fields below: " + ", ".join(below))
        rec("b_nursing_partial", cod, samp, "rank_abs_partial_r_asc", rk_abs, n=len(g), field="nursing",
            note="1 = smallest |partial r|")
        rec("b_nursing_partial", cod, samp, "rank_inc_prestige_raw_asc", rk_inc, n=len(g), field="nursing",
            note="1 = smallest raw incremental prestige R^2 (sign-blind)")
    NPR = pd.DataFrame(NPR)
    # identity (raw R^2): incremental prestige = partial r^2 x (1 - R^2(geo only)). So a small increment is
    # compatible with a sizeable partial r wherever geography explains a lot; checked on every field x version.
    Dr = D[(D.measure == "raw")].dropna(subset=["partial_r"])
    ident_dev = float(np.max(np.abs(Dr.inc_prestige - Dr.partial_r ** 2 * (1 - Dr.R2_geo_only))))
    rec("b_check", "inc_prestige(raw) = partial_r^2 * (1 - R2_geo_only(raw)), all fields x versions", "both",
        "max_abs_deviation", ident_dev, n=len(Dr))
    # the signed, geography-netted cross-field gradient: Spearman(licensure, partial r(F, earn | geo)).
    # partial r is signed (like the project's rho_f); incremental R^2 is sign-blind and scaled by 1 - R^2(geo).
    PRG = []
    f35_raw = {cod: set(D[(D["sample"] == P35) & (D.coding == cod) & (D.measure == "raw")]
                        .dropna(subset=["partial_r"]).field) for cod in CODINGS}
    for (samp, cod), g in D[D.measure == "raw"].groupby(["sample", "coding"], sort=False):
        g = g.dropna(subset=["partial_r"])
        row = dict(sample=samp, coding=cod, k=len(g))
        g35 = g[g.field.isin(f35_raw[cod])]
        g10 = g[g.dfr_full >= 10]
        for stat, xs, ys, fs in [("signed", g.licensure, g.partial_r, g.field),
                                 ("signed_35cfields", g35.licensure, g35.partial_r, g35.field),
                                 ("signed_dfr10", g10.licensure, g10.partial_r, g10.field),
                                 ("abs", g.licensure, g.partial_r.abs(), g.field),
                                 ("inc_prestige_same_fields", g.licensure, g.inc_prestige, g.field),
                                 ("signed_log", g.licensure, g.partial_r_log, g.field),
                                 ("signed_rank", g.licensure, g.partial_r_rank, g.field),
                                 ("signed_broad", g.lic_broad, g.partial_r, g.field)]:
            r = bs(xs, ys, fields=fs)
            row[stat] = r
            recb("b_partial_gradient", cod, samp, f"spearman(licensure,partial_r)_{stat}", r,
                 note="partial r(F, earn | geo), raw rows; 35c field bootstrap B=5000 seed 7; CIP-2 cluster "
                      f"bootstrap B={B_CL} in *_cip2; licensure = "
                      + ("broad share" if stat == "signed_broad" else "strict share"))
        row["cl"] = cluster_diag(g.licensure, g.partial_r, g.field)
        rec_diag("b_partial_gradient", cod, samp, "spearman(licensure,partial_r)_signed", row["cl"])
        row["cl_broad"] = cluster_diag(g.lic_broad, g.partial_r, g.field)
        rec_diag("b_partial_gradient", cod, samp, "spearman(licensure_broad,partial_r)_signed", row["cl_broad"])
        # within the Health family (CIP-2 51): the fields and their partial r, for the text
        row["health"] = g[g.field.map(CIP2) == "51"][["field", "licensure", "lic_broad", "partial_r"]].to_dict("records")
        lo_vals = [spearmanr(g[g.field != f].licensure, g[g.field != f].partial_r)[0] for f in g.field]
        row["loo_range"] = (float(np.min(lo_vals)), float(np.max(lo_vals)))
        rec("b_partial_gradient", cod, samp, "spearman(licensure,partial_r)_leave_one_field_out_min",
            row["loo_range"][0], n=len(g) - 1)
        rec("b_partial_gradient", cod, samp, "spearman(licensure,partial_r)_leave_one_field_out_max",
            row["loo_range"][1], n=len(g) - 1)
        PRG.append(row)
    # nursing vs comparison fields: Fisher-z difference of the partial r (independent-sample approximation;
    # the fields share some institutions, so the p-values are approximate)
    NCMP = []
    for (samp, cod), g in D[D.measure == "raw"].groupby(["sample", "coding"], sort=False):
        g = g.dropna(subset=["partial_r"]).set_index("field")
        if "nursing" not in g.index:
            continue
        hn = g.loc["nursing"]
        for comp in ("biology", "computer_science", "economics", "accounting"):
            if comp not in g.index:
                continue
            hc = g.loc[comp]
            z = (np.arctanh(hn.partial_r) - np.arctanh(hc.partial_r)) / np.sqrt(1 / hn.dfz + 1 / hc.dfz)
            p = float(2 * norm.sf(abs(z)))
            NCMP.append(dict(sample=samp, coding=cod, comp=comp, r_n=hn.partial_r, r_c=hc.partial_r,
                             lo_c=hc.partial_lo, hi_c=hc.partial_hi, n_c=int(hc.n), z=float(z), p=p,
                             r_n_log=hn.partial_r_log, r_n_rank=hn.partial_r_rank,
                             r_c_log=hc.partial_r_log, r_c_rank=hc.partial_r_rank,
                             inc_n=hn.inc_prestige, inc_c=hc.inc_prestige,
                             r2geo_n=hn.R2_geo_only, r2geo_c=hc.R2_geo_only))
            rec("b_nursing_vs", cod, samp, "fisher_z_diff_nursing_minus_comp", float(z), n=int(hn.n + hc.n),
                field=comp, note=f"two-sided p={p:.4f}; nursing r={hn.partial_r:+.4f}, {comp} r={hc.partial_r:+.4f}; "
                                 "independent-sample approximation")
        for f_ in ("nursing", "biology"):
            if f_ in g.index:
                rec("b_nursing_vs", cod, samp, "partial_r_log_earnings", g.loc[f_, "partial_r_log"],
                    n=int(g.loc[f_, "n"]), field=f_)
                rec("b_nursing_vs", cod, samp, "partial_r_rank_based", g.loc[f_, "partial_r_rank"],
                    n=int(g.loc[f_, "n"]), field=f_)
    NCMP = pd.DataFrame(NCMP)
    # fields that only the full sample admits: signed partial r vs sign-blind incremental R^2
    add_f = sorted(set(D[D["sample"] == PF_].field) - set(D[D["sample"] == P35].field))
    ADDED = []
    for f_ in add_f:
        h = D[(D["sample"] == PF_) & (D.field == f_) & (D.measure == "raw")].set_index("coding")
        ADDED.append(dict(field=f_, lic=float(h.licensure.iloc[0]), n=int(h.n.iloc[0]),
                          **{f"pr_{c}": float(h.partial_r.get(c, np.nan)) for c in CODINGS},
                          **{f"inc_{c}": float(h.inc_prestige.get(c, np.nan)) for c in CODINGS},
                          **{f"dfr_{c}": (int(h.dfr_full[c]) if c in h.index else -1) for c in CODINGS}))
    ADDED = pd.DataFrame(ADDED).sort_values("lic", ascending=False)
    # per-field table into the CSV (all fields, all specs)
    for _, r in D.iterrows():
        for st in ("inc_prestige", "inc_geo", "R2_geo_only", "R2_prestige_only", "R2_full"):
            rec("b_all_fields", f"{r.coding} | {r.measure}", r["sample"], st, r[st], n=r.n, field=r.field,
                note=f"licensure_strict={r.licensure:.4f}; n_geo={r.n_geo}; dfr_full={r.dfr_full}")
        if r.measure == "raw":
            rec("b_all_fields", f"{r.coding}", r["sample"], "partial_r(F,earn|geo)", r.partial_r, r.partial_lo,
                r.partial_hi, n=r.n, field=r.field, note="Fisher-z CI; NaN when the full model has 0 residual df")
            rec("b_all_fields", f"{r.coding}", r["sample"], "partial_r(F,log_earn|geo)", r.partial_r_log, n=r.n,
                field=r.field)
            rec("b_all_fields", f"{r.coding}", r["sample"], "partial_r(rank F,rank earn|geo)", r.partial_r_rank,
                n=r.n, field=r.field)
    return D, S, repro, nurse_reg, NPR, dict(ident_dev=ident_dev, ident_n=len(Dr)), PRG, NCMP, ADDED


# =============================================================================================
# (d) selectivity-adjusted coupling as the outcome
# =============================================================================================
D_OUT = [  # outcome column, SE column, description, family
    ("rho_raw", "se_rho_raw", "baseline coupling ρ_f, full sample", "raw"),
    ("rho_raw_sat", "se_rho_raw_sat", "raw ρ_f, SAT sample", "raw"),
    ("rho_sel", "se_rho_sel", "partial | SAT + ADM (SAT sample)", "partial"),
    ("rho_pcs", "se_rho_pcs", "partial | Pell + control + state level (SAT sample)", "partial"),
    ("rho_full_stlev", "se_rho_full_stlev", "partial | broad: SAT + ADM + Pell + control + state level", "partial"),
    ("rho_full_state", "se_rho_full_state", "partial | strict: SAT + ADM + Pell + control + state FE", "partial"),
    ("rho_full_ie", "se_rho_full_ie", "partial | broad + institution-wide earnings", "partial"),
    ("rho_raw_adm", "se_rho_raw_adm", "raw ρ_f, ADM sample", "raw"),
    ("rho_adm_stlev", "se_rho_adm_stlev", "partial | ADM + Pell + control + state level (ADM sample)", "partial"),
    ("rho_adm_state", "se_rho_adm_state", "partial | ADM + Pell + control + state FE (ADM sample)", "partial"),
    ("fe_a_beta", "fe_a_se", "within-institution β_f (institution FE), spec (a)", "fe"),
    ("fe_b_beta", "fe_b_se", "within-institution β_f + field-specific SAT/ADM/Pell slopes (b)", "fe"),
    ("fe_c_beta", "fe_c_se", "within-institution β_f + selectivity + brand slopes (c)", "fe"),
    ("rho_sat_earn", "se_rho_sat_earn", "selectivity pricing ρ(SAT_AVG, earn)", "pricing"),
    ("rho_adm_earn", "se_rho_adm_earn", "selectivity pricing ρ(−ADM_RATE, earn)", "pricing"),
    ("rho_ie_earn", "se_rho_ie_earn", "ρ(institution-wide earnings, field earnings)", "pricing"),
    ("cG_raw_sat", "se_cG_raw_sat", "brand coupling c_G, raw (SAT sample)", "brand"),
    ("cG_full_stlev", "se_cG_full_stlev", "brand coupling c_G, broad partial", "brand"),
]
D_PAIRS = [("rho_raw_sat", "rho_full_stlev"), ("rho_raw_sat", "rho_full_state"), ("rho_raw_sat", "rho_sel"),
           ("rho_raw_sat", "rho_full_ie"), ("rho_raw_adm", "rho_adm_stlev"), ("rho_raw_adm", "rho_adm_state"),
           ("cG_raw_sat", "cG_full_stlev")]
D_SE = {c: s for c, s, _, _ in D_OUT}


def wls_slope(x, y, se):
    m = sm.WLS(y, sm.add_constant(x, has_constant="add"), weights=1 / se ** 2).fit()
    return float(m.params[1])


def section_d():
    S = pd.read_csv(INTERIM / "selectivity_fields.csv")
    anch = pd.read_parquet(INTERIM / "acs_occ_anchors.parquet")[["field", "licensure_strict", "licensure_broad"]]
    S = S.merge(anch, on="field", how="left")
    res = []
    for k, (col, sec, desc, fam) in enumerate(D_OUT):
        for lic in ("licensure_strict", "licensure_broad"):
            for tag, sub in (("all", S), ("drop nursing+comm.dis.", S[~S.field.isin(DROP_NCD)])):
                d = sub.dropna(subset=[col, sec, lic])
                d = d[d[sec] > 0]
                if len(d) < 8:
                    continue
                x, y, se = d[lic].to_numpy(float), d[col].to_numpy(float), d[sec].to_numpy(float)
                b = wls_slope(x, y, se)
                bb = wls_slope_boot(x, y, se, B_SLOPE, SEED * 100 + k)
                blo, bhi = np.percentile(bb, [2.5, 97.5])
                r = bs(x, y, fields=d.field)
                row = dict(outcome=col, desc=desc, family=fam, lic=lic, subset=tag, k=len(d),
                           mean=float(y.mean()), b=b, b_lo=blo, b_hi=bhi, rho=r[0], rho_lo=r[1], rho_hi=r[2],
                           rho_clo=r.clo, rho_chi=r.chi, rho_t=r, cl=None)
                rec("d_selectivity", f"{col}: {desc}", tag, f"wls_slope_on_{lic}", b, blo, bhi, len(d))
                recb("d_selectivity", f"{col}: {desc}", tag, f"spearman_with_{lic}", r)
                rec("d_selectivity", f"{col}: {desc}", tag, "mean_outcome", float(y.mean()), n=len(d))
                if tag == "all":
                    row["cl"] = cluster_diag(x, y, d.field)
                    rec_diag("d_selectivity", f"{col}: {desc}", tag, f"spearman_with_{lic}", row["cl"])
                res.append(row)
    R = pd.DataFrame(res)
    # matched raw -> partial comparisons (same fields; paired field bootstrap)
    P = []
    for j, (c0, c1) in enumerate(D_PAIRS):
        for tag, sub in (("all", S), ("drop nursing+comm.dis.", S[~S.field.isin(DROP_NCD)])):
            d = sub.dropna(subset=[c0, c1, D_SE[c0], D_SE[c1], "licensure_strict"])
            d = d[(d[D_SE[c0]] > 0) & (d[D_SE[c1]] > 0)]
            x = d.licensure_strict.to_numpy(float)
            y0, y1 = d[c0].to_numpy(float), d[c1].to_numpy(float)
            s0, s1 = d[D_SE[c0]].to_numpy(float), d[D_SE[c1]].to_numpy(float)
            b0, b1 = wls_slope(x, y0, s0), wls_slope(x, y1, s1)
            u0 = wls_slope_boot(x, y0, s0, B_SLOPE, SEED * 1000 + j)   # same seed -> same (paired) draws
            u1 = wls_slope_boot(x, y1, s1, B_SLOPE, SEED * 1000 + j)
            db = u1 - u0
            lo, hi = np.percentile(db, [2.5, 97.5])
            r0, r1 = bs(x, y0, fields=d.field), bs(x, y1, fields=d.field)
            m0, m1 = float(y0.mean()), float(y1.mean())
            P.append(dict(raw=c0, partial=c1, subset=tag, k=len(d), b_raw=b0, b_partial=b1, diff=b1 - b0,
                          diff_lo=lo, diff_hi=hi, rho_raw=r0, rho_partial=r1,
                          mean_raw=m0, mean_partial=m1, level_ratio=m1 / m0))
            rec("d_selectivity", f"{c0} -> {c1}", tag, "wls_slope_partial_minus_raw", b1 - b0, lo, hi, len(d))
            rec("d_selectivity", f"{c0} -> {c1}", tag, "level_ratio_partial_over_raw", m1 / m0, n=len(d))
    named = S[S.field.isin(["nursing", "communication_disorders", "accounting", "special_education",
                            "pharmacy", "computer_science", "economics", "biology", "social_work",
                            "public_health"])]
    return R, pd.DataFrame(P), named, S


SEL_CMP = ["biology", "public_health", "computer_science", "economics", "accounting", "social_work",
           "special_education"]
SEL_SPECS = [("rho_full_stlev", "se_rho_full_stlev", "broad partial (SAT + ADM + Pell + control + state level)"),
             ("rho_full_state", "se_rho_full_state", "strict partial (SAT + ADM + Pell + control + state FE)"),
             ("fe_a_beta", "fe_a_se", "within-institution β_f, spec (a)"),
             ("fe_b_beta", "fe_b_se", "within-institution β_f + field-specific selectivity slopes, spec (b)")]


def nursing_after_selectivity(S):
    """Issue: the nursing verdict must also handle selectivity. Nursing vs comparison fields on scripts/55's
    selectivity-adjusted couplings and within-institution slopes (estimates ± 1.96 joint-bootstrap SE;
    z-test of the difference treating the two fields' estimates as independent, which is approximate because
    fields share institutions), nursing's rank among all fields with an estimate, the fields below it, and how
    many fields' 95% CI includes 0."""
    Si = S.set_index("field")
    out, ranks = [], {}
    for col, sec, desc in SEL_SPECS:
        d = S.dropna(subset=[col, sec])
        d = d[d[sec] > 0]
        di = d.set_index("field")
        if "nursing" not in di.index:
            continue
        n_ = di.loc["nursing"]
        rk = int(d[col].rank(method="min")[d.field == "nursing"].iloc[0])
        lo_all, hi_all = d[col] - 1.96 * d[sec], d[col] + 1.96 * d[sec]
        inc0 = (lo_all <= 0) & (hi_all >= 0)
        low = d[d.licensure_strict < 0.20]
        below = d[d[col] < n_[col]].sort_values(col)
        ranks[col] = dict(desc=desc, rank=rk, k=len(d), mean=float(d[col].mean()), n_inc0=int(inc0.sum()),
                          n_inc0_low=int((inc0 & (d.licensure_strict < 0.20)).sum()), k_low=len(low),
                          n_sig_pos=int((lo_all > 0).sum()),
                          below=[(f_, float(v_), float(l_) if np.isfinite(l_) else np.nan)
                                 for f_, v_, l_ in zip(below.field, below[col], below.licensure_strict)],
                          r=float(n_[col]), se=float(n_[sec]))
        rec("d_nursing_selectivity", desc, "scripts/55", "nursing_rank_asc", rk, n=len(d), field="nursing",
            note=f"1 = lowest; field mean {d[col].mean():+.4f}; fields whose ±1.96 SE CI includes 0: {int(inc0.sum())} "
                 f"of {len(d)}; fields below nursing: " + ", ".join(below.field))
        for f_ in ["nursing"] + SEL_CMP:
            if f_ not in di.index:
                continue
            c_ = di.loc[f_]
            z = (n_[col] - c_[col]) / np.sqrt(n_[sec] ** 2 + c_[sec] ** 2) if f_ != "nursing" else np.nan
            p = float(2 * norm.sf(abs(z))) if f_ != "nursing" else np.nan
            out.append(dict(spec=col, desc=desc, field=f_, est=float(c_[col]), se=float(c_[sec]),
                            lo=float(c_[col] - 1.96 * c_[sec]), hi=float(c_[col] + 1.96 * c_[sec]),
                            lic=float(c_.licensure_strict), lic_b=float(c_.licensure_broad), z=z, p=p))
            rec("d_nursing_selectivity", desc, "scripts/55", "estimate_pm_1.96se", c_[col], c_[col] - 1.96 * c_[sec],
                c_[col] + 1.96 * c_[sec], field=f_,
                note=(f"SE {c_[sec]:.4f}; nursing minus this field z={z:+.3f}, p={p:.4f} (independent-estimate "
                      "approximation)") if f_ != "nursing" else f"SE {c_[sec]:.4f}")
    return pd.DataFrame(out), ranks


# =============================================================================================
# (c2) regressor / field-set robustness: young-worker licensure share, pre-med fields
# =============================================================================================
ENG_FIELDS = sorted(f for f, c in CIP2.items() if c == ENG)
SUBSETS = {"all": [], "without nursing + comm. dis.": DROP_NCD, "without pre-med fields": PREMED,
           "without both": DROP_NCD + PREMED, "without engineering (CIP-14)": ENG_FIELDS}
REGS = {"licensure_strict": "all ages (strict)", "licensure_broad": "all ages (broad)",
        "lic_young": "ages 22–27 (strict)"}


def section_c2(a, Dfull, Ssel, young):
    ym = dict(zip(young.field, young.lic_young))
    dcol = {"licensure_strict": "licensure", "licensure_broad": "lic_broad", "lic_young": "lic_young"}
    out = []
    qb = [("35c sample (PSEO-listed)", "state FE", "raw"), ("35c sample (PSEO-listed)", "state FE", "LOO-CV"),
          ("full (Scorecard state)", "BEA region FE", "LOO-CV"),
          ("full (Scorecard state)", "state earnings level (1 df)", "LOO-CV")]
    qd = ["rho_raw_sat", "rho_full_stlev", "rho_full_state", "rho_adm_stlev", "fe_b_beta"]
    for reg in REGS:
        for j, (sub, drop) in enumerate(SUBSETS.items()):
            # (a) scripts/20 WLS on the gap, HC1 CI; CIP-2 cluster bootstrap CI alongside
            d = a.assign(lic_young=a.field.map(ym)).dropna(subset=[reg])
            d = d[~d.field.isin(drop)]
            b, se = fit_wls(d, ["absorption_acs", reg])
            clo, chi, _ = boot_fit_cluster(d, ["absorption_acs", reg], B_CL, SEED_CL)
            out.append(dict(q="b_licensure: WLS gap ~ licensure + absorption (scripts/20 sample)", reg=reg, sub=sub,
                            est=b, lo=b - 1.96 * se, hi=b + 1.96 * se, k=len(d), kind="b (HC1 CI)",
                            clo=clo, chi=chi, t=None))
            # (b) Spearman(licensure, incremental prestige)
            for samp, cod, meas in qb:
                g = Dfull[(Dfull["sample"] == samp) & (Dfull.coding == cod) & (Dfull.measure == meas)].dropna(subset=["inc_prestige"])
                g = g.assign(lic_young=g.field.map(ym))
                g = g[~g.field.isin(drop)]
                r = bs(g[dcol[reg]], g.inc_prestige, fields=g.field)
                out.append(dict(q=f"Spearman(licensure, incr. prestige R²): {samp.split(' (')[0]}, {cod}, {meas}", reg=reg,
                                sub=sub, est=r[0], lo=r[1], hi=r[2], k=r[3], kind="Spearman", clo=r.clo, chi=r.chi, t=r))
            # (b') Spearman(licensure, signed partial r(F, earn | geo)): all 6 sample x coding versions
            for samp in ("35c sample (PSEO-listed)", "full (Scorecard state)"):
                for cod in CODINGS:
                    g = Dfull[(Dfull["sample"] == samp) & (Dfull.coding == cod) & (Dfull.measure == "raw")]
                    g = g.dropna(subset=["partial_r"]).assign(lic_young=lambda z: z.field.map(ym))
                    g = g[~g.field.isin(drop)]
                    r = bs(g[dcol[reg]], g.partial_r, fields=g.field)
                    out.append(dict(q=f"Spearman(licensure, signed partial r): {samp.split(' (')[0]}, {cod}", reg=reg,
                                    sub=sub, est=r[0], lo=r[1], hi=r[2], k=r[3], kind="Spearman (partial r)",
                                    clo=r.clo, chi=r.chi, t=r))
            # (d) Spearman(licensure, selectivity outcome)
            S = Ssel.assign(lic_young=Ssel.field.map(ym))
            S = S[~S.field.isin(drop)]
            for oc in qd:
                dd = S.dropna(subset=[oc, reg])
                r = bs(dd[reg], dd[oc], fields=dd.field)
                out.append(dict(q=f"Spearman(licensure, {oc})", reg=reg, sub=sub, est=r[0], lo=r[1], hi=r[2],
                                k=r[3], kind="Spearman", clo=r.clo, chi=r.chi, t=r))
    R = pd.DataFrame(out)
    for _, r in R.iterrows():
        rec("c_robust2", r.q, r["sub"], f"{r.kind} on {r.reg}", r.est, r.lo, r.hi, r.k, clo=r.clo, chi=r.chi,
            note="CIP-2 cluster bootstrap CI in *_cip2")
    return R


# =============================================================================================
# (c3) setting test: public-sector employment share (ACS COW) vs the geography-netted coupling
# =============================================================================================
def partial_spearman(x1, y, x2):
    """Rank partial correlation of x1 and y given x2 (Pearson of the residuals of rank(x1) and rank(y) on
    rank(x2))."""
    r1, ry, r2 = rankdata(x1), rankdata(y), rankdata(x2)
    Z = np.column_stack([np.ones(len(r2)), r2])
    e1 = r1 - Z @ np.linalg.lstsq(Z, r1, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    den = np.sqrt((e1 @ e1) * (ey @ ey))
    return float(e1 @ ey / den) if den > 0 else np.nan


def partial_spearman_boot(x1, y, x2, fields, B=B_CL, seed=SEED_CL):
    """Point estimate + field-bootstrap and CIP-2 cluster-bootstrap percentile CIs (B draws each)."""
    x1, y, x2 = (np.asarray(v, float) for v in (x1, y, x2))
    ok = np.isfinite(x1) & np.isfinite(y) & np.isfinite(x2)
    x1, y, x2, f = x1[ok], y[ok], x2[ok], np.asarray(fields, dtype=object)[ok]
    est = partial_spearman(x1, y, x2)
    n = len(y)
    I = np.random.default_rng(seed).integers(0, n, (B, n))
    fb = np.array([partial_spearman(x1[i], y[i], x2[i]) for i in I
                   if min(len(np.unique(x1[i])), len(np.unique(y[i])), len(np.unique(x2[i]))) > 2])
    cl = cip2_of(f).astype(str)
    labs = np.unique(cl)
    members = [np.flatnonzero(cl == c) for c in labs]
    D = np.random.default_rng(seed + 1).integers(0, len(labs), (B, len(labs)))
    cb = []
    for row in D:
        i = np.concatenate([members[j] for j in row])
        if min(len(np.unique(x1[i])), len(np.unique(y[i])), len(np.unique(x2[i]))) > 2:
            cb.append(partial_spearman(x1[i], y[i], x2[i]))
    cb = np.array(cb)
    return (est, float(np.nanpercentile(fb, 2.5)), float(np.nanpercentile(fb, 97.5)), n,
            float(np.nanpercentile(cb, 2.5)), float(np.nanpercentile(cb, 97.5)))


SET_VARS = {"gov": "government share (COW 3–5), all ages", "gov_young": "government share, ages 22–27",
            "govnp": "government + nonprofit share (COW 2–5), all ages"}


def section_setting(Dfull, Ssel, sect):
    """Does a measured 'setting' variable (public-sector employment share) predict the geography-netted and the
    selectivity-adjusted coupling, and does the strict licensure share still predict it net of that share?"""
    sm_ = sect.set_index("field")
    out = []
    targets = []
    for samp in ("35c sample (PSEO-listed)", "full (Scorecard state)"):
        for cod in CODINGS:
            g = Dfull[(Dfull["sample"] == samp) & (Dfull.coding == cod) & (Dfull.measure == "raw")].dropna(subset=["partial_r"])
            targets.append((f"signed partial r: {samp.split(' (')[0]}, {cod}", g.field.to_numpy(), g.partial_r.to_numpy(float),
                            g.licensure.to_numpy(float)))
    for oc in ("rho_full_stlev", "rho_full_state", "fe_b_beta"):
        d = Ssel.dropna(subset=[oc, "licensure_strict"])
        targets.append((f"selectivity-adjusted: {oc}", d.field.to_numpy(), d[oc].to_numpy(float),
                        d.licensure_strict.to_numpy(float)))
    for j, (q, f, y, lic) in enumerate(targets):
        row = dict(q=q, k=len(y))
        for v in SET_VARS:
            xv = np.array([sm_[v].get(x, np.nan) for x in f], float)
            row[v] = bs(xv, y, fields=f)
            recb("c_setting", q, "ACS COW", f"spearman({v},outcome)", row[v])
        gv = np.array([sm_["gov"].get(x, np.nan) for x in f], float)
        row["lic_given_gov"] = partial_spearman_boot(lic, y, gv, f, B_CL, SEED_CL + 300 + j)
        row["gov_given_lic"] = partial_spearman_boot(gv, y, lic, f, B_CL, SEED_CL + 400 + j)
        row["lic_gov"] = spearmanr(lic[np.isfinite(gv)], gv[np.isfinite(gv)])[0]
        for nm in ("lic_given_gov", "gov_given_lic"):
            t = row[nm]
            rec("c_setting", q, "ACS COW", f"partial_spearman_{nm}", t[0], t[1], t[2], t[3], clo=t[4], chi=t[5],
                note="rank partial correlation; field bootstrap CI in ci_lo/ci_hi, CIP-2 cluster bootstrap in *_cip2")
        out.append(row)
    return out


# =============================================================================================
# (e) UK nursing (scripts/41) with adjusted and LOO-CV R^2
# =============================================================================================
def section_e():
    s40 = s41.s40
    e = s40.tag_cah2(s40.extract_uk_edges())
    dens, pres = s40.uk_prestige_and_density(e)
    usable = sorted(set(dens[dens.usable].cah2))
    leo = s41.load_leo_geo()
    leo["earn"] = pd.to_numeric(leo.earnings_median, errors="coerce")
    leo["p"] = leo.provider_name.map(s41.norm)
    leo["geo"] = np.where((leo.provider_country_name == "England") & (leo.provider_region_name != "Total"),
                          leo.provider_region_name, leo.provider_country_name)
    leo = leo.dropna(subset=["earn", "geo"])
    ref = pd.read_csv(INTERIM / "uk_licensing_decomp.csv").set_index("cah2")
    rows = []
    for cah in usable:
        j = pres[cah].merge(leo[leo.cah2_subject_name == cah][["p", "earn", "geo"]], on="p", how="inner") \
                     .dropna(subset=["s", "earn", "geo"])
        if len(j) < s41.MIN_PROV or j.geo.nunique() < s41.MIN_GEO:
            continue
        y = j.earn.to_numpy(float); sv = j.s.to_numpy(float); geo = j.geo.astype(str).to_numpy()
        Xg = _design(geo, "cat"); Xp = np.column_stack([np.ones(len(y)), sv]); Xf = np.column_stack([Xg, sv])
        rp, rg, rf = ols_fit(y, Xp), ols_fit(y, Xg), ols_fit(y, Xf)
        cvp, cvg, cvf = loo_r2(y, sv, geo, True, None), loo_r2(y, sv, geo, False, "cat"), loo_r2(y, sv, geo, True, "cat")
        pr = partial_r(y, sv, geo, "cat")
        r = dict(cah2=cah, n=len(y), n_geo=int(j.geo.nunique()), licensed=cah in s41.LICENSED,
                 chance_R2_geo=(rg[2] - 1) / (len(y) - 1),
                 raw_inc_p=rf[0] - rg[0], adj_inc_p=rf[1] - rg[1], cv_inc_p=cvf - cvg,
                 raw_inc_g=rf[0] - rp[0], adj_inc_g=rf[1] - rp[1], cv_inc_g=cvf - cvp,
                 raw_R2_geo=rg[0], adj_R2_geo=rg[1], cv_R2_geo=cvg,
                 raw_R2_p=rp[0], adj_R2_p=rp[1], cv_R2_p=cvp,
                 partial_r=pr[0], partial_lo=pr[1], partial_hi=pr[2],
                 ref_inc_p=float(ref.loc[cah, "incremental_prestige"]) if cah in ref.index else np.nan)
        rows.append(r)
    U = pd.DataFrame(rows)
    for m in ("raw", "adj", "cv"):
        U[f"rank_{m}"] = U[f"{m}_inc_p"].rank(method="min").astype(int)
    for _, r in U.iterrows():
        for st in ("raw_inc_p", "adj_inc_p", "cv_inc_p", "raw_inc_g", "adj_inc_g", "cv_inc_g",
                   "raw_R2_geo", "adj_R2_geo", "cv_R2_geo", "chance_R2_geo"):
            rec("e_uk", "LEO YAG5 median earnings ~ UK ORCID SpringRank + provider region/nation FE", "UK",
                st, r[st], n=r.n, field=r.cah2)
        rec("e_uk", "partial r(prestige, earnings | region FE), Fisher-z CI", "UK", "partial_r",
            r.partial_r, r.partial_lo, r.partial_hi, r.n, field=r.cah2)
    repro = float(np.nanmax(np.abs(U.raw_inc_p.round(3) - U.ref_inc_p)))
    return U, repro


# =============================================================================================
# figure
# =============================================================================================
def make_figure(A, Bsum, Dres, U, gm_obs, NPR, NCMP, C2, PRG, NSEL):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 1.0, "legend.frameon": False,
                         "svg.hashsalt": "64", "pdf.fonttype": 42})
    P35, PF = "35c sample (PSEO-listed)", "full (Scorecard state)"
    SSH = {P35: "35c's", PF: "full"}
    CSH = {"state FE": "state FE", "BEA region FE": "region FE", "state earnings level (1 df)": "state level (1 df)"}
    fig, ax = plt.subplots(2, 3, figsize=(21, 12.5))
    # (a) implied nursing - CS gap difference across b_licensure specs
    a = ax[0, 0]
    keys = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11"]
    Ar = A.set_index("key").loc[keys]
    yy = np.arange(len(keys))[::-1]
    for yv, (k, r) in zip(yy, Ar.iterrows()):
        col = PAL["blue"] if r.regressor in ("licensure_strict", "licensure_broad") else PAL["red"]
        a.plot([r.implied_lo, r.implied_hi], [yv, yv], color=col, lw=1.6)
        a.plot(r.implied, yv, "o", color=col, ms=5)
    a.axvline(0, color="k", lw=0.7)
    a.axvline(gm_obs, color=PAL["grey"], ls="--", lw=1)
    a.set_ylim(-1.0, len(keys) - 0.4)
    a.text(gm_obs - 0.01, -0.65, "observed nursing − CS", fontsize=7, ha="right", va="center", color="#555")
    short = {"A1": "scripts/20 strict share (+absorption)", "A2": "scripts/20 broad share (+absorption)",
             "A3": "strict share only", "A4": "scripts/13 sample, SE floor 0.05", "A5": "+ z(dispersion)",
             "A6": "+ CIP-2 random intercept (REML)", "A7": "REML, broad share",
             "A8": "REML, 4-field binary tag (scripts/13)", "A9": "binary tag, WLS + z(disp.)",
             "A10": "binary tag, scripts/20 WLS", "A11": "REML, binary: strict share ≥ 0.5"}
    a.set_yticks(yy); a.set_yticklabels([short[k] for k in keys], fontsize=7.5)
    a.set_xlabel("implied gap difference, nursing − computer science\n= b_licensure × (x_nursing − x_CS), 95% field-bootstrap CI")
    a.set_title("(a) b_licensure across specifications, on one scale\n(blue: ACS licensure share; red: binary tag)", fontsize=9)
    # (b) Spearman(licensure, incremental prestige R^2) by measure x coding x sample
    b = ax[0, 1]
    order = [(c, m) for c in CODINGS for m in MEASURES]
    yy = np.arange(len(order))[::-1]
    for off, samp, mk, col in ((0.16, P35, "s", PAL["violet"]), (-0.16, PF, "o", PAL["blue"])):
        for yv, (c, m) in zip(yy, order):
            row = [r for r in Bsum if r["sample"] == samp and r["coding"] == c and r["measure"] == m][0]
            r0, lo, hi, k = row["inc_prestige"]
            b.plot([lo, hi], [yv + off, yv + off], color=col, lw=1.4)
            b.plot(r0, yv + off, mk, color=col, ms=5, label=f"{SSH[samp]} sample: incremental prestige R²"
                   if (c, m) == order[0] else None)
            g0 = row["inc_geo"][0]
            b.plot(g0, yv + off, mk, mfc="white", mec=col, ms=5,
                   label=f"{SSH[samp]} sample: incremental geography R²" if (c, m) == order[0] else None)
    b.axvline(0, color="k", lw=0.7)
    b.set_yticks(yy); b.set_yticklabels([f"{CSH[c]} | {m}" for c, m in order], fontsize=7.5)
    b.set_xlabel("Spearman(licensure share, ·) across fields\nfilled + 95% CI: incremental prestige R²; open: incremental geography R²")
    b.set_title("(b) 35c decomposition: raw vs adjusted vs cross-validated R²", fontsize=9)
    b.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2)
    # (b2) nursing vs comparison fields: partial r(F, earn | geo) in all 6 sample x coding versions, then the
    # selectivity-adjusted couplings from scripts/55 (broad and strict partial, ± 1.96 SE)
    n2 = ax[0, 2]
    vers = [(s_, c_) for s_ in (P35, PF) for c_ in CODINGS]
    rows = vers + [("sel", "rho_full_stlev"), ("sel", "rho_full_state")]
    yy2 = np.arange(len(rows))[::-1]
    series = [("nursing", "Nursing", PAL["red"], "o"), ("biology", "Biology", PAL["teal"], "s"),
              ("accounting", "Accounting", PAL["grey"], "^"), ("economics", "Economics", PAL["violet"], "D"),
              ("computer_science", "Computer Science", PAL["blue"], "v")]
    Ni = NPR.set_index(["sample", "coding"])
    Ci = NCMP.set_index(["sample", "coding", "comp"])
    Si = NSEL.set_index(["spec", "field"])
    for j, (fk, lab, col, mk) in enumerate(series):
        off = 0.28 - j * 0.14
        for yv, (s_, c_) in zip(yy2, rows):
            if s_ == "sel":
                if (c_, fk) not in Si.index:
                    continue
                r0, lo, hi = Si.loc[(c_, fk)][["est", "lo", "hi"]]
            elif fk == "nursing":
                if (s_, c_) not in Ni.index:
                    continue
                r0, lo, hi = Ni.loc[(s_, c_)][["r", "lo", "hi"]]
            else:
                if (s_, c_, fk) not in Ci.index:
                    continue
                r0, lo, hi = Ci.loc[(s_, c_, fk)][["r_c", "lo_c", "hi_c"]]
            n2.plot([lo, hi], [yv + off, yv + off], color=col, lw=1.3)
            n2.plot(r0, yv + off, mk, color=col, ms=4.5, mec="k" if fk == "nursing" else col, mew=0.5,
                    label=lab if (s_, c_) == rows[0] else None)
    n2.axvline(0, color="k", lw=0.7)
    for yl in (1.5, 4.5):
        n2.axhline(yl, color=PAL["light"], lw=0.8)
    selab = {"rho_full_stlev": "selectivity-adj.: broad partial", "rho_full_state": "selectivity-adj.: state-FE partial"}
    n2.set_yticks(yy2)
    n2.set_yticklabels([f"{SSH[s_]} | {CSH[c_]}" if s_ != "sel" else selab[c_] for s_, c_ in rows], fontsize=7.5)
    n2.set_xlabel("rows 1–6: partial r(F, earnings | geography), 95% Fisher-z CI\n"
                  "rows 7–8: selectivity-adjusted partial coupling (scripts/55), ± 1.96 SE")
    n2.set_title("(b′) nursing vs comparison fields, net of geography and net of selectivity", fontsize=9)
    n2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5)
    # (c) signed partial-r gradient: field vs CIP-2 cluster bootstrap, within families, without engineering,
    # and with the broad licensure share
    g2 = ax[1, 0]
    yy = np.arange(len(vers))[::-1]
    PGi = {(r_["sample"], r_["coding"]): r_ for r_ in PRG}
    sers = [("strict share, field bootstrap", PAL["blue"], "o", lambda p: (p["signed"][0], p["signed"][1], p["signed"][2])),
            ("strict share, CIP-2 cluster bootstrap", PAL["red"], "s", lambda p: (p["signed"][0], p["signed"].clo, p["signed"].chi)),
            ("strict share without engineering (CIP-14), cluster bootstrap", PAL["teal"], "^",
             lambda p: (p["cl"]["no_eng"][0], p["cl"]["no_eng"].clo, p["cl"]["no_eng"].chi)),
            ("strict share within CIP-2 families (point)", "k", "x", lambda p: (p["cl"]["r_within"], np.nan, np.nan)),
            ("broad share, field bootstrap", PAL["violet"], "D",
             lambda p: (p["signed_broad"][0], p["signed_broad"][1], p["signed_broad"][2]))]
    for j, (lab, col, mk, get) in enumerate(sers):
        off = 0.3 - j * 0.15
        for yv, v_ in zip(yy, vers):
            r0, lo, hi = get(PGi[v_])
            if np.isfinite(lo):
                g2.plot([lo, hi], [yv + off, yv + off], color=col, lw=1.3)
            g2.plot(r0, yv + off, mk, color=col, ms=4.5, label=lab if v_ == vers[0] else None)
    g2.axvline(0, color="k", lw=0.7)
    g2.axhline(2.5, color=PAL["light"], lw=0.8)
    g2.set_yticks(yy); g2.set_yticklabels([f"{SSH[s_]} | {CSH[c_]}" for s_, c_ in vers], fontsize=7.5)
    g2.set_xlabel("Spearman(licensure share, signed partial r(F, earn | geo)) across fields, 95% CI")
    g2.set_title("(c) the signed gradient is a between-discipline contrast, specific to the strict share", fontsize=9)
    g2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    # (d) licensure vs selectivity-adjusted coupling
    c = ax[1, 1]
    S = Dres["S"]
    d = S.dropna(subset=["rho_full_stlev", "licensure_strict", "rho_raw_sat"])
    c.scatter(d.licensure_strict, d.rho_raw_sat, s=22, color=PAL["light"], edgecolor="#777", lw=0.4,
              label="raw ρ_f (SAT sample)")
    c.scatter(d.licensure_strict, d.rho_full_stlev, s=26, color=PAL["blue"], edgecolor="k", lw=0.3,
              label="partial ρ_f, broad selectivity spec")
    xs = np.linspace(0, d.licensure_strict.max(), 20)
    for colname, colr in (("rho_raw_sat", "#777"), ("rho_full_stlev", PAL["blue"])):
        dd = d.dropna(subset=[D_SE[colname]])
        m = sm.WLS(dd[colname], sm.add_constant(dd.licensure_strict), weights=1 / dd[D_SE[colname]] ** 2).fit()
        c.plot(xs, m.params.iloc[0] + m.params.iloc[1] * xs, color=colr, lw=1.3, ls="--" if colname == "rho_raw_sat" else "-")
    for _, r in d.iterrows():
        if r.licensure_strict >= 0.3 or r.field in ("accounting", "computer_science", "economics"):
            c.annotate(LAB.get(r.field, r.field), (r.licensure_strict, r.rho_full_stlev), fontsize=6.5,
                       xytext=(3, 2), textcoords="offset points")
    c.axhline(0, color="k", lw=0.6)
    c.set_xlabel("licensure share (ACS 2023, strict)"); c.set_ylabel("within-field coupling")
    c.set_title("(d) the licensing gradient after selectivity adjustment (WLS lines, 1/SE²)", fontsize=9)
    c.legend(fontsize=7, loc="upper right")
    # (e) UK subjects: incremental prestige raw / adjusted / CV
    e = ax[1, 2]
    Us = U.sort_values("adj_inc_p")
    yy = np.arange(len(Us))
    for off, colm, colr, lab in ((-0.22, "raw_inc_p", PAL["light"], "raw"), (0.0, "adj_inc_p", PAL["blue2"], "adjusted"),
                                 (0.22, "cv_inc_p", PAL["violet"], "LOO-CV")):
        e.barh(yy + off, Us[colm], height=0.22, color=colr, label=lab)
    e.set_yticks(yy)
    e.set_yticklabels([("▶ " if r.licensed else "") + r.cah2 for _, r in Us.iterrows()], fontsize=7)
    e.axvline(0, color="k", lw=0.6)
    e.set_xlabel("incremental prestige R² beyond provider region / nation\n(UK LEO, 5 years after graduation)")
    e.set_title("(e) UK: incremental prestige R² by subject (▶ = licensed)", fontsize=9)
    e.legend(fontsize=7, loc="lower right")
    fig.tight_layout(h_pad=3.0)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight", pad_inches=0.15, metadata={"Software": None})
    plt.close(fig)


# =============================================================================================
# report
# =============================================================================================
def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def get_b(Bsum, samp, cod, meas, stat="inc_prestige"):
    return [r for r in Bsum if r["sample"] == samp and r["coding"] == cod and r["measure"] == meas][0][stat]


def main():
    # the vectorised bootstrap must equal scripts/35c's loop version (checked on 35c's own data)
    ref = pd.read_csv(INTERIM / "er_axis_35c_mechanism.csv").dropna(subset=["incremental_prestige"])
    r_fast, r_loop = bs(ref.licensure, ref.incremental_prestige), s35c.boot_spearman(ref.licensure, ref.incremental_prestige)
    assert np.allclose(r_fast[:3], r_loop[:3], atol=1e-12), (r_fast, r_loop)
    rec("check", "vectorised bootstrap == scripts/35c boot_spearman (35c's saved table)", "35c", "spearman",
        r_fast[0], r_fast[1], r_fast[2], r_fast[3], note=f"loop version {r_loop[0]:+.4f} [{r_loop[1]:+.4f}, {r_loop[2]:+.4f}]")
    # the vectorised CIP-2 cluster bootstrap must equal the loop version draw for draw (same data, B = 1000)
    xr, yr = ref.licensure.to_numpy(float), ref.incremental_prestige.to_numpy(float)
    clr = cip2_of(ref.field)
    v_vec = _bs_cluster_draws(xr, yr, clr, 1000, SEED_CL)[0]
    v_loop = _bs_cluster_ci_loop(xr, yr, clr, 1000, SEED_CL)
    assert np.array_equal(np.isnan(v_vec), np.isnan(v_loop)) and np.allclose(v_vec[~np.isnan(v_vec)],
                                                                             v_loop[~np.isnan(v_loop)], atol=1e-10)
    rec("check", "vectorised CIP-2 cluster bootstrap == loop version, draw by draw (35c's table, B=1000)", "35c",
        "max_abs_draw_difference", float(np.nanmax(np.abs(v_vec - v_loop))), n=len(xr))
    # ---------------- (a) ----------------
    gm, a, b, anch = build_gap_frames()
    A, SIB, corr_tag, gm_obs = section_a(gm, a, b)
    # ---------------- (b), (c) ----------------
    cells, cellinfo = load_cells()
    Dfull, Bsum, repro_b, nurse_reg, NPR, ident, PRG, NCMP, ADDED = section_b(cells)
    # ---------------- (d) ----------------
    Dres, Dpairs, Dnamed, Ssel = section_d()
    # ---------------- (c2) ----------------
    young = young_licensure()
    for _, r in young.iterrows():
        rec("c_robust2", "young-worker strict licensure share (ACS 22-27, full-time BA+)", "ACS", "lic_young",
            r.lic_young, n=r.n_young, field=r.field)
    C2 = section_c2(a, Dfull, Ssel, young)
    corr_young = float(spearmanr(anch.merge(young, on="field").licensure_strict,
                                 anch.merge(young, on="field").lic_young)[0])
    # ---------------- nursing after selectivity; setting (public-sector share) test ----------------
    NSEL, NSELrk = nursing_after_selectivity(Ssel)
    sect, sect_ok = acs_setting_shares(anch)
    for _, r in sect.iterrows():
        for c_ in ("gov", "npo", "govnp", "gov_young"):
            rec("c_setting", "ACS PUMS 2023 employer-sector share (scripts/20 anchor population)", "ACS", c_,
                r[c_], n=r.n_acs_check if not c_.endswith("young") else r.n_young_sector, field=r.field)
    rec("check", "ACS sector-share rows per field == scripts/20 n_acs", "ACS", "all_equal", float(sect_ok))
    SET = section_setting(Dfull, Ssel, sect)
    # ---------------- (e) ----------------
    U, repro_e = section_e()
    # ---------------- Monte Carlo check of the B = 5000 Spearman bootstrap ----------------
    mc = np.array(MC, float)
    mc = mc[np.isfinite(mc).all(1)]
    dev = np.abs(mc[:, 3:5] - mc[:, 1:3])
    ex5 = (mc[:, 1] > 0) | (mc[:, 2] < 0)
    ex1 = (mc[:, 3] > 0) | (mc[:, 4] < 0)
    # label each flipped CI by the output rows that carry exactly the same (estimate, lo, hi)
    lab_of = {}
    for row in ROWS:
        if np.isfinite(row["ci_lo"]) and np.isfinite(row["ci_hi"]):
            lab_of.setdefault((row["value"], row["ci_lo"], row["ci_hi"]), []).append(
                f"{row['section']}: {row['spec']} | {row['sample']} | {row['stat']}")
    flip = {}
    for r_ in mc[ex5 != ex1]:
        key = (float(r_[0]), float(r_[1]), float(r_[2]))
        flip.setdefault(key, (float(r_[3]), float(r_[4]), sorted(set(lab_of.get(key, [])))))
    flip = [(k[0], k[1], k[2], v[0], v[1], v[2]) for k, v in sorted(flip.items())]
    mcinfo = dict(n=len(mc), dmax=float(dev.max()), dmed=float(np.median(dev)), flips=int((ex5 != ex1).sum()),
                  flip_list=flip, keys=set(map(tuple, mc[:, :3].tolist())))
    rec("check", f"Spearman CI bounds: B=5000 (reported) vs B={B_MC} (same seed), all bs() calls", "all",
        "max_abs_bound_difference", mcinfo["dmax"], n=mcinfo["n"],
        note=f"median {mcinfo['dmed']:.4f}; calls whose 'CI excludes 0' status differs: {mcinfo['flips']}")
    for t in flip:
        rec("check", "Spearman CI whose 'excludes 0' status differs between B=5000 and B=1000", "all",
            f"ci_B{B_MC}", t[0], t[3], t[4],
            note=f"B=5000 CI [{t[1]:+.4f}, {t[2]:+.4f}]; reported as: " + (" || ".join(t[5]) if t[5] else "not in CSV"))
    # Monte Carlo check of the CIP-2 cluster bootstrap: the same B = B_CL with a second seed
    mcl = np.array(MC_CL, float)
    mcl = mcl[np.isfinite(mcl).all(1)]
    devc = np.abs(mcl[:, 3:5] - mcl[:, 1:3])
    exa = (mcl[:, 1] > 0) | (mcl[:, 2] < 0)
    exb = (mcl[:, 3] > 0) | (mcl[:, 4] < 0)
    fl = mcl[exa != exb]
    mclinfo = dict(n=len(mcl), dmax=float(devc.max()), dmed=float(np.median(devc)), flips=int(len(fl)),
                   flip_keys={(float(r_[0]), float(r_[1]), float(r_[2])) for r_ in fl},
                   flip_near=float(np.max(np.min(np.abs(fl[:, 1:3]), axis=1))) if len(fl) else np.nan)
    rec("check", f"CIP-2 cluster-bootstrap CI bounds: seed {SEED_CL} (reported) vs seed {SEED_CL + 1}, B={B_CL}", "all",
        "max_abs_bound_difference", mclinfo["dmax"], n=mclinfo["n"],
        note=f"median {mclinfo['dmed']:.4f}; calls whose 'CI excludes 0' status differs: {mclinfo['flips']}; "
             f"largest distance of a reported bound from 0 among them: {mclinfo['flip_near']:.4f}")

    out = pd.DataFrame(ROWS)
    out = out[["section", "spec", "sample", "field", "stat", "value", "ci_lo", "ci_hi", "ci_lo_cip2", "ci_hi_cip2",
               "n", "note"]]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.6g")
    make_figure(A, Bsum, {"S": Ssel}, U, gm_obs, NPR, NCMP, C2, PRG, NSEL)
    write_md(A, SIB, corr_tag, gm_obs, a, b, Dfull, Bsum, repro_b, cellinfo, Dres, Dpairs, Dnamed, U, repro_e,
             nurse_reg, C2, corr_young, young, NPR, ident, PRG, NCMP, ADDED, mcinfo,
             NSEL, NSELrk, sect, sect_ok, SET, mclinfo)
    print(f"done: {len(out)} rows -> {OUT_CSV.relative_to(ROOT)}; figure {OUT_FIG.relative_to(ROOT)}; "
          f"report {OUT_MD.relative_to(ROOT)}")


def fcl(lo, hi, d=2):
    """CIP-2 cluster-bootstrap CI in angle brackets (field-bootstrap CIs use square brackets)."""
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    return f"⟨{lo:+.{d}f}, {hi:+.{d}f}⟩"


def excl0(t):
    return bool(np.isfinite(t[1]) and np.isfinite(t[2]) and (t[1] > 0 or t[2] < 0))


def exclc(t):
    lo, hi = getattr(t, "clo", np.nan), getattr(t, "chi", np.nan)
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def c3(t, k=False):
    """estimate [field-bootstrap CI] ⟨CIP-2 cluster-bootstrap CI⟩ (k=...)"""
    s = f"{fmt(t[0])} {fci(t[1], t[2])}"
    c = fcl(getattr(t, "clo", np.nan), getattr(t, "chi", np.nan))
    return s + (f" {c}" if c else "") + (f" (k={t[3]})" if k else "")


def write_md(A, SIB, corr_tag, gm_obs, a, b, Dfull, Bsum, repro_b, cellinfo, Dres, Dpairs, Dnamed, U, repro_e,
             nurse_reg, C2, corr_young, young, NPR, ident, PRG, NCMP, ADDED, mcinfo,
             NSEL, NSELrk, sect, sect_ok, SET, mclinfo):
    L = []
    Ai = A.set_index("key")
    P35, PF = "35c sample (PSEO-listed)", "full (Scorecard state)"
    ST, RG, SL = "state FE", "BEA region FE", "state earnings level (1 df)"
    Dd = Dres.set_index(["outcome", "lic", "subset"])
    Pp = Dpairs.set_index(["raw", "partial", "subset"])
    nurseU = U[U.cah2 == "Nursing and midwifery"].iloc[0]

    def cell(samp, cod, meas):
        return [r for r in Bsum if r["sample"] == samp and r["coding"] == cod and r["measure"] == meas][0]

    def st(samp, cod, meas, stat="inc_prestige"):
        return cell(samp, cod, meas)[stat]

    def nf(samp, cod, meas, f="nursing"):
        c = cell(samp, cod, meas)
        return c[f] if f in c else None

    def rng(ts):
        return f"{fmt(min(t[0] for t in ts))} to {fmt(max(t[0] for t in ts))}"

    def ci_list(ts):
        return ", ".join(f"{fmt(t[0])} {fci(t[1], t[2])}" for t in ts)

    def pv(p):
        return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"

    def pe(p):
        return "p < 0.001" if p < 0.001 else f"p = {pv(p)}"

    SHORT = {PF: "full sample", P35: "35c's sample"}
    CSH = {ST: "state FE", RG: "BEA-region FE", SL: "the 1-df state earnings level"}
    CSH2 = {ST: "state FE", RG: "region FE", SL: "1-df"}
    BEA = {"1": "New England", "2": "Mid East", "3": "Great Lakes", "4": "Plains", "5": "Southeast",
           "6": "Southwest", "7": "Rocky Mountains", "8": "Far West"}

    # ---------------- quantities carried over from the first version ----------------
    cells35 = [r for r in Bsum if r["sample"] == P35]
    cellsF = [r for r in Bsum if r["sample"] == PF]
    ip35 = [r["inc_prestige"] for r in cells35]; ipF = [r["inc_prestige"] for r in cellsF]
    ig35 = [r["inc_geo"] for r in cells35]; igF = [r["inc_geo"] for r in cellsF]
    ipb = [r["inc_prestige_broadlic"] for r in Bsum]
    rp_all = [r["R2_prestige_only"] for r in Bsum]
    nurse_inc = Dfull[Dfull.field == "nursing"].inc_prestige.dropna()
    n35 = {m: nf(P35, ST, m) for m in MEASURES}
    nST = nf(PF, ST, "raw")
    rkST, rkRG = st(PF, ST, "LOO-CV", "nurse_geo_rank"), st(PF, RG, "LOO-CV", "nurse_geo_rank")
    ncv = {c: nf(PF, c, "LOO-CV") for c in CODINGS}
    chance35 = (n35["raw"].n_geo - 1) / (n35["raw"].n - 1)
    bro = Dd.loc[("rho_full_stlev", "licensure_strict", "all")]
    bro_raw = Dd.loc[("rho_raw_sat", "licensure_strict", "all")]
    fea, feb = Dd.loc[("fe_a_beta", "licensure_strict", "all")], Dd.loc[("fe_b_beta", "licensure_strict", "all")]
    pr_b = Pp.loc[("rho_raw_sat", "rho_full_stlev", "all")]
    sat_, adm_ = Dd.loc[("rho_sat_earn", "licensure_strict", "all")], Dd.loc[("rho_adm_earn", "licensure_strict", "all")]
    ukcv_other = U[U.cah2 != "Nursing and midwifery"].cv_R2_geo.max()
    C2i = C2.set_index(["q", "reg", "sub"])

    def c2(qstart, reg="licensure_strict", sub="all", cl=True):
        m = [k for k in C2i.index if k[0].startswith(qstart) and k[1] == reg and k[2] == sub]
        r = C2i.loc[m[0]]
        return f"{fmt(r.est)} {fci(r.lo, r.hi)}" + (f" {fcl(r.clo, r.chi)}" if cl else "")
    QB, QSEL = "b_licensure", "Spearman(licensure, rho_full_stlev)"
    BOTH, NCD, PM, NENG = "without both", "without nursing + comm. dis.", "without pre-med fields", "without engineering (CIP-14)"
    acc_pr = [nf(s_, c, "raw", "accounting").partial_r for s_ in (PF, P35) for c in CODINGS
              if nf(s_, c, "raw", "accounting") is not None]
    Ni = NPR.set_index(["sample", "coding"])
    n_ci0, n_ver = int(NPR.ci0.sum()), len(NPR)
    npr_ex = NPR[~NPR.ci0]

    def npr_txt(samp, cod):
        r = Ni.loc[(samp, cod)]
        return f"{fmt(r.r)} {fci(r.lo, r.hi)}"
    npr_ex_txt = ("none of the six excludes 0" if len(npr_ex) == 0 else
                  "a positive association with a CI excluding 0 appears only with "
                  + "; ".join(f"{CSH[r.coding]} on the {SHORT[r['sample']]} ({fmt(r.r)} {fci(r.lo, r.hi)})"
                              for _, r in npr_ex.iterrows()))
    rk_rng = (int(NPR['rank'].min()), int(NPR['rank'].max()))
    rka_rng = (int(NPR.rank_abs.min()), int(NPR.rank_abs.max()))
    n_abs1 = int((NPR.rank_abs == 1).sum())
    k_rng = (int(NPR.k.min()), int(NPR.k.max()))
    LOWLIC = 0.20
    lowlic = {}
    for _, r_ in NPR[NPR.coding == RG].iterrows():
        for f_ in r_.below:
            if r_.lic_of[f_] < LOWLIC:
                lowlic[f_] = r_.lic_of[f_]
    lowlic_txt = ", ".join(f"{LAB.get(f_, f_)} {v_:.2f}" for f_, v_ in sorted(lowlic.items(), key=lambda t: t[1]))
    nRG35i, nRGFi = Ni.loc[(P35, RG)], Ni.loc[(PF, RG)]
    nr35, nrF = nurse_reg[P35], nurse_reg[PF]
    ach = pd.read_parquet(INTERIM / "acs_occ_anchors.parquet").set_index("field")
    acc_s, acc_b = float(ach.loc["accounting", "licensure_strict"]), float(ach.loc["accounting", "licensure_broad"])
    tag_small = SIB[SIB.lic_strict < 0.1].sort_values("field")
    tag_txt = " and ".join(f"{LAB.get(r.field, r.field).lower()} {r.lic_strict:.3f}" for _, r in tag_small.iterrows())
    Ssel_all = pd.read_csv(INTERIM / "selectivity_fields.csv")
    zb = Ssel_all.rho_full_stlev / Ssel_all.se_rho_full_stlev
    n_sig_broad, k_sig_broad = int((zb > 1.96).sum()), int(zb.notna().sum())
    acc_sig = bool((zb[Ssel_all.field == "accounting"] > 1.96).any())
    fe_means = Ssel_all[["fe_a_beta", "fe_b_beta"]].mean()
    n_of = {f: int(nf(PF, ST, "raw", f).n) for f in ("special_education", "communication_disorders")}
    loo_a_pos = bool((A.loo_min > 0).all() and (A.loo_max > 0).all())
    loo_b_neg = bool(all(r["loo_range"][1] < 0 for r in Bsum))
    pol = U[U.cah2 == "Politics"].iloc[0] if (U.cah2 == "Politics").any() else None
    both_rows = C2[(C2["sub"] == BOTH) & (C2.reg == "licensure_strict") & (C2.kind != "Spearman (partial r)")]
    rho_pairs = Dpairs[(Dpairs.subset == "all") & Dpairs.partial.str.startswith("rho_")]
    pair_ci0 = int(((rho_pairs.diff_lo <= 0) & (rho_pairs.diff_hi >= 0)).sum())
    part_rows = [(c_, Dd.loc[(c_, "licensure_strict", "all")]) for c_, _, _, fam_ in D_OUT
                 if fam_ == "partial" and (c_, "licensure_strict", "all") in Dd.index]
    part_rows_b = [(c_, Dd.loc[(c_, "licensure_broad", "all")]) for c_, _, _, fam_ in D_OUT
                   if fam_ == "partial" and (c_, "licensure_broad", "all") in Dd.index]
    n_part_ex = sum(excl0(r_.rho_t) for _, r_ in part_rows)
    n_part_exc = sum(exclc(r_.rho_t) for _, r_ in part_rows)
    part_ns_c = [(c_, r_) for c_, r_ in part_rows if not exclc(r_.rho_t)]
    part_edge_c = [(c_, r_) for c_, r_ in part_rows if exclc(r_.rho_t) and min(abs(r_.rho_clo), abs(r_.rho_chi)) < 0.01]
    part_w = [r_.cl["r_within"] for _, r_ in part_rows]
    part_wp = [r_.cl["p_within"] for _, r_ in part_rows]
    n_partb_in = sum(not excl0(r_.rho_t) for _, r_ in part_rows_b)
    n_partb_inc = sum(not exclc(r_.rho_t) for _, r_ in part_rows_b)
    rr = Ssel_all.dropna(subset=["rho_raw"]).set_index("field").rho_raw
    cd_raw, cd_rank, cd_k = float(rr["communication_disorders"]), int(rr.rank(method="min")["communication_disorders"]), len(rr)
    cd_sat = float(Ssel_all.set_index("field").rho_full_stlev.get("communication_disorders", np.nan))
    DbioF = Dfull[(Dfull.field == "biology") & (Dfull["sample"] == PF)]
    bio_incF = DbioF.inc_prestige.dropna()
    bio_prF = DbioF[DbioF.measure == "raw"]
    bio_excl_F = int(((bio_prF.partial_lo > 0) | (bio_prF.partial_hi < 0)).sum())
    Dn = Dfull[(Dfull.field == "nursing") & (Dfull.measure == "raw")].set_index(["sample", "coding"])
    r2g_st = {s_: float(Dn.loc[(s_, ST)].R2_geo_only) for s_ in (P35, PF)}
    imp_r = {s_: float(np.sqrt(nurse_inc.max() / (1 - r2g_st[s_]))) for s_ in (P35, PF)}
    scale_dev = float(np.nanmax(np.abs(np.r_[(Dn.partial_r_log - Dn.partial_r).to_numpy(),
                                             (Dn.partial_r_rank - Dn.partial_r).to_numpy()])))
    nr_min, nr_max = float(NPR.r.min()), float(NPR.r.max())
    Ci = NCMP.set_index(["sample", "coding", "comp"])

    def nsig(comp, samp=None):
        s_ = NCMP[NCMP.comp == comp]
        if samp is not None:
            s_ = s_[s_["sample"] == samp]
        return int((s_.p < 0.05).sum()), len(s_)

    def pz(samp, cod, comp):
        r_ = Ci.loc[(samp, cod, comp)]
        return f"{fmt(r_.r_n)} vs {fmt(r_.r_c)}, {pe(r_.p)}"
    cs_sig, ec_sig, ac_sig = nsig("computer_science"), nsig("economics"), nsig("accounting")
    bioF_sig, bio35_sig = nsig("biology", PF), nsig("biology", P35)
    ac_ns = NCMP[(NCMP.comp == "accounting") & (NCMP.p >= 0.05)]
    ac_ns_txt = "; ".join(f"{SHORT[r_['sample']]}, {CSH[r_.coding]}: p = {pv(r_.p)}" for _, r_ in ac_ns.iterrows())
    csec_pmax = float(NCMP[NCMP.comp.isin(["computer_science", "economics"])].p.max())
    bench_c = NCMP[NCMP.comp.isin(["computer_science", "economics", "accounting"])]
    bench_rng = (float(bench_c.r_c.min()), float(bench_c.r_c.max()))
    PGi = {(r_["sample"], r_["coding"]): r_ for r_ in PRG}
    VERS = [(s_, c_) for s_ in (P35, PF) for c_ in CODINGS]
    sg = [PGi[v]["signed"] for v in VERS]
    sgb = [PGi[v]["signed_broad"] for v in VERS]
    PR2 = C2[C2.kind == "Spearman (partial r)"].set_index(["q", "reg", "sub"])

    def pq(samp, cod):
        return f"Spearman(licensure, signed partial r): {samp.split(' (')[0]}, {cod}"

    def prb(samp, sub, reg="licensure_strict"):
        return [PR2.loc[(pq(samp, c), reg, sub)].t for c in CODINGS]
    pb35, pbF = prb(P35, BOTH), prb(PF, BOTH)
    pbF_excl = sum(excl0(t) for t in pbF)
    pbFy = prb(PF, BOTH, "lic_young")
    sgy = [PR2.loc[(pq(s_, c_), "lic_young", "all")].t for s_, c_ in VERS]
    prc = [f"pr_{c}" for c in CODINGS]
    neg_add = ADDED[ADDED[prc].apply(lambda z: bool((z.dropna() < 0).all()) and z.notna().any(), axis=1)]
    neg_add = neg_add.sort_values("lic", ascending=False)
    neg_add_txt = "; ".join(f"{LAB.get(r_.field, r_.field).lower()} (strict share {r_.lic:.2f}; partial r "
                            + " / ".join(fmt(r_[c]) for c in prc) + ")" for _, r_ in neg_add.iterrows())
    neg_inc = neg_add[[f"inc_{c}" for c in CODINGS]].to_numpy(float)
    lic_medF = float(Dfull[(Dfull["sample"] == PF) & (Dfull.measure == "raw")].drop_duplicates("field").licensure.median())
    absF = {c: PGi[(PF, c)]["abs"] for c in CODINGS}
    incsF = {c: PGi[(PF, c)]["inc_prestige_same_fields"] for c in CODINGS}

    # ---------------- NEW: discipline (CIP-2) structure ----------------
    cl_sg = [PGi[v]["cl"] for v in VERS]
    sg_excl_c = [v for v, t in zip(VERS, sg) if exclc(t)]
    sg_c_txt = "; ".join(f"{SHORT[s_]}, {CSH2[c_]} {fcl(PGi[(s_, c_)]['signed'].clo, PGi[(s_, c_)]['signed'].chi)}"
                         for s_, c_ in VERS)
    w_rng = (min(d["r_within"] for d in cl_sg), max(d["r_within"] for d in cl_sg))
    wp_rng = (min(d["p_within"] for d in cl_sg), max(d["p_within"] for d in cl_sg))
    shw_rng = (min(d["share_within"] for d in cl_sg), max(d["share_within"] for d in cl_sg))
    hb = [d["contrib"].get("51", 0.0) + d["contrib"].get("26", 0.0) for d in cl_sg]
    fam_c = {}
    for d in cl_sg:
        for k_, v_ in d["contrib"].items():
            fam_c[k_] = fam_c.get(k_, 0.0) + v_ / len(cl_sg)
    top_c = sorted(fam_c.items(), key=lambda t: -t[1])[:3]
    gw_rng = (min(d["g_within"] for d in cl_sg), max(d["g_within"] for d in cl_sg))
    loco_rng = (min(d["loco_min"] for d in cl_sg), max(d["loco_max"] for d in cl_sg))
    ne35 = [PGi[(P35, c)]["cl"]["no_eng"] for c in CODINGS]
    neF = [PGi[(PF, c)]["cl"]["no_eng"] for c in CODINGS]
    inc35_c = sum(exclc(t) for t in ip35)
    incF_c = sum(exclc(t) for t in ipF)
    cv35 = st(P35, ST, "LOO-CV")
    cl_cv35 = cell(P35, ST, "LOO-CV")["cl_inc"]
    hl = PGi[(PF, RG)]["health"]
    hl_txt = ", ".join(f"{LAB.get(h['field'], h['field']).lower()} {fmt(h['partial_r'])} (strict share {h['licensure']:.2f})"
                       for h in sorted(hl, key=lambda h: h["partial_r"]))
    sgy_c = sum(exclc(t) for t in sgy)
    # broad share
    sgb_in = sum(not excl0(t) for t in sgb)
    sgb_inc = sum(not exclc(t) for t in sgb)
    ipb_in = sum(not excl0(t) for t in ipb)
    fea_b = Dd.loc[("fe_a_beta", "licensure_broad", "all")]
    feb_b = Dd.loc[("fe_b_beta", "licensure_broad", "all")]
    fec_b = Dd.loc[("fe_c_beta", "licensure_broad", "all")]
    fe_below = NSELrk.get("fe_a_beta", {}).get("below", [])
    n_eng_below = sum(CIP2.get(f_) == ENG for f_, _, _ in fe_below)
    A1c, A2c = Ai.loc["A1"], Ai.loc["A2"]
    b_noeng = C2i.loc[[k for k in C2i.index if k[0].startswith(QB) and k[1] == "licensure_strict" and k[2] == NENG][0]]

    ex_samples = {s_ for s_, _ in sg_excl_c}
    if len(sg_excl_c) == 3 and ex_samples == {PF}:
        cl_where, cl_where_s = " (the full institution sample only)", "only on the full institution sample"
    elif len(sg_excl_c) == 0:
        cl_where, cl_where_s = "", "in none of the sample × geography versions"
    else:
        cl_where = " (" + "; ".join(f"{SHORT[s_]}, {CSH2[c_]}" for s_, c_ in sg_excl_c) + ")"
        cl_where_s = "only in " + "; ".join(f"{SHORT[s_]}, {CSH2[c_]}" for s_, c_ in sg_excl_c)
    FAMN = {"01": "Agriculture", "03": "Natural resources", "04": "Architecture", "11": "Computer science",
            "13": "Education", "14": "Engineering", "23": "English", "26": "Biological sciences",
            "27": "Mathematics and statistics", "38": "Philosophy and religion", "40": "Physical sciences",
            "42": "Psychology", "44": "Public administration and social service", "45": "Social sciences",
            "50": "Arts", "51": "Health", "52": "Business", "54": "History"}
    fam_d = Dfull[(Dfull["sample"] == PF) & (Dfull.coding == RG) & (Dfull.measure == "raw")].dropna(subset=["partial_r"])
    fam_d = fam_d.assign(cip2=fam_d.field.map(CIP2))
    fam_sel = Ssel_all.set_index("field").rho_full_stlev
    fam_d = fam_d.assign(selp=fam_d.field.map(fam_sel), gov=fam_d.field.map(sect.set_index("field").gov))
    FAM = (fam_d.groupby("cip2").agg(k=("field", "size"), lic=("licensure", "mean"), licb=("lic_broad", "mean"),
                                     licmin=("licensure", "min"), licmax=("licensure", "max"),
                                     gov=("gov", "mean"), pr=("partial_r", "mean"), prmin=("partial_r", "min"),
                                     prmax=("partial_r", "max"), selp=("selp", "mean"),
                                     fields=("field", lambda z: ", ".join(LAB.get(x, x) for x in sorted(z))))
           .sort_values(["lic", "k"]))
    pr_mean_all = float(fam_d.partial_r.mean())

    def fam_txt(c):
        r_ = FAM.loc[c]
        return f"{FAMN.get(c, c).lower()} ({r_.lic:.2f}; {fmt(r_.pr)})"
    lowfam = [c for c in ("14", "52", "11", "27") if c in FAM.index]
    highfam = [c for c in ("51", "26") if c in FAM.index]
    base_all = Ssel_all.set_index("field").rho_raw
    base_mean = float(base_all.dropna().mean())
    base_eng = float(base_all[[f_ for f_ in ENG_FIELDS if f_ in base_all.index]].mean())
    # ---------------- NEW: nursing after selectivity ----------------
    NS = NSEL.set_index(["spec", "field"])

    def ns(spec, f):
        r_ = NS.loc[(spec, f)]
        if spec.startswith("fe"):
            return f"{fmt(r_.est, 3)} (SE {r_.se:.3f})"
        return f"{fmt(r_.est)} (SE {r_.se:.2f}) {fci(r_.lo, r_.hi)}"

    def nsp(spec, f):
        r_ = NS.loc[(spec, f)]
        return f"{fmt(r_.est, 3 if spec.startswith('fe') else 2)}, {pe(r_.p)}"
    rkb, rks = NSELrk["rho_full_stlev"], NSELrk["rho_full_state"]
    nb = NS.loc[("rho_full_stlev", "nursing")]
    zbio = NS.loc[("rho_full_stlev", "biology")]
    below_txt = ", ".join(f"{LAB.get(f_, f_)} ({fmt(v_)}; {l_:.2f})" if np.isfinite(l_) else f"{LAB.get(f_, f_)} ({fmt(v_)})"
                          for f_, v_, l_ in rkb["below"])
    geo_hi_max = float(NPR.hi.max())
    fe_cmp = ["computer_science", "economics", "accounting"]
    fe_pmax = max(float(NS.loc[(sp_, f_)].p) for sp_ in ("fe_a_beta", "fe_b_beta") for f_ in fe_cmp)
    st_cmp = ["biology", "computer_science", "economics", "accounting"]
    st_pmin = min(float(NS.loc[("rho_full_state", f_)].p) for f_ in st_cmp)

    # ---------------- NEW: setting test ----------------
    SETi = {r_["q"]: r_ for r_ in SET}
    qpr = [f"signed partial r: {s_.split(' (')[0]}, {c_}" for s_, c_ in VERS]
    gov6 = [SETi[q]["gov"] for q in qpr]
    govnp6 = [SETi[q]["govnp"] for q in qpr]
    lg6 = [SETi[q]["lic_given_gov"] for q in qpr]
    gl6 = [SETi[q]["gov_given_lic"] for q in qpr]
    lic_gov_rng = (min(SETi[q]["lic_gov"] for q in qpr), max(SETi[q]["lic_gov"] for q in qpr))

    def pex(t, cl=False):
        lo, hi = (t[4], t[5]) if cl else (t[1], t[2])
        return bool(lo > 0 or hi < 0)
    sm_ = sect.set_index("field")
    spd = sm_.loc["special_education"]
    swk = sm_.loc["social_work"]
    sw_pr = [nf(s_, c, "raw", "social_work") for s_ in (P35, PF) for c in CODINGS]
    sw_pr = [x for x in sw_pr if x is not None]
    sw_ex = sum(bool(x.partial_lo > 0 or x.partial_hi < 0) for x in sw_pr)
    sp_pr = [nf(PF, c, "raw", "special_education") for c in CODINGS]
    sp_pr = [x for x in sp_pr if x is not None]
    sp_ex = sum(bool(x.partial_lo > 0 or x.partial_hi < 0) for x in sp_pr)
    Sn = Dnamed.set_index("field")
    sp = Sn.loc["special_education"]
    swS = Sn.loc["social_work"]
    gov_rank_sp = int(sect.gov.rank(ascending=False, method="min")[sect.field == "special_education"].iloc[0])

    # ---------------- Monte Carlo checks: which quoted CIs are near the edge ----------------
    ans_keys = set()
    for t in (sg + sgb + sgy + pb35 + pbF + pbFy + ip35 + ipF + ig35 + igF + ipb + rp_all + ne35 + neF
              + list(absF.values()) + list(incsF.values()) + gov6 + govnp6):
        ans_keys.add((float(t[0]), float(t[1]), float(t[2])))
    for _, r_ in both_rows[both_rows.kind.str.startswith("Spearman")].iterrows():
        ans_keys.add((float(r_.est), float(r_.lo), float(r_.hi)))
    for r_ in [r_ for _, r_ in part_rows] + [r_ for _, r_ in part_rows_b] + [bro, bro_raw, fea, feb, fea_b, feb_b, fec_b]:
        ans_keys.add((float(r_.rho), float(r_.rho_lo), float(r_.rho_hi)))
    flip_ans = [t for t in mcinfo["flip_list"] if (t[0], t[1], t[2]) in ans_keys]
    flip_near = max((min(abs(t[1]), abs(t[2])) for t in mcinfo["flip_list"]), default=np.nan)
    assert ans_keys <= mcinfo["keys"], f"Answer CIs not found among bs() calls: {sorted(ans_keys - mcinfo['keys'])}"
    ans_ckeys = set()
    for t in sg + sgb + sgy + ip35 + ipF + ne35 + neF + gov6 + govnp6 + [r_.rho_t for _, r_ in part_rows + part_rows_b]:
        ans_ckeys.add((float(t[0]), float(t.clo), float(t.chi)))
    cflip_ans = sorted(ans_ckeys & mclinfo["flip_keys"])

    # =============================== title + answer ===============================
    L += ["# Licensing battery: is \"prestige carries no pay information beyond geography in licensed fields\" robust?\n",
          "Public data only; descriptive, not causal. Every number below is printed by "
          "`scripts/64_licensing_battery.py` (seeded; outputs byte-identical on re-run). Long-format table: "
          "`data/interim/licensing_battery.csv` (columns `ci_lo`/`ci_hi` = field bootstrap, `ci_lo_cip2`/`ci_hi_cip2` "
          "= CIP-2 cluster bootstrap); figure: `outputs/figures/licensing_battery.png`. Notation: square brackets "
          "[ ] are 95% CIs from resampling fields (or model CIs where stated); angle brackets ⟨ ⟩ are 95% CIs from "
          "resampling whole CIP-2 families (disciplines); k = number of fields.\n",
          "## Answer\n",
          f"**Short answer: once overfitting and selectivity are handled, a much weaker claim survives.** (i) For "
          f"nursing, \"department prestige carries almost no pay information\" holds once selectivity is netted as "
          f"well as geography (broad selectivity-adjusted coupling {fmt(nb.est)} {fci(nb.lo, nb.hi)}), but that is not "
          f"special to nursing or to licensed fields: biology's value is the same ({fmt(zbio.est)}), nursing ranks "
          f"{rkb['rank']} of {rkb['k']} fields, and {rkb['n_inc0']} of {rkb['k']} fields have a CI that includes 0. "
          f"Netting geography alone, nursing's association is weak but not shown to be absent. (ii) How much geography "
          f"explains is inflated by overfitting. (iii) The cross-field licensing gradient is a contrast between "
          f"disciplines, and it appears only with the strict licensure share. Spearman(strict share, prestige–pay "
          f"association net of geography) is {rng(sg)}. Its CI excludes 0 in {sum(excl0(t) for t in sg)} of 6 "
          f"versions when fields are resampled, but in {len(sg_excl_c)} of 6{cl_where} "
          f"when whole CIP-2 families are resampled. Within families it is {fmt(w_rng[0])} to {fmt(w_rng[1])}. With the "
          f"broad share it is {rng(sgb)} (CI includes 0 in {sgb_in} of 6). (iv) \"Pay set by setting\" is the "
          f"project's interpretation, not a result of this battery. The one measured setting proxy, a field's public "
          f"and nonprofit employment share, predicts the geography-netted association at least as well as the strict "
          f"licensure share (the government share alone somewhat less consistently), and net of each other neither is "
          f"consistently distinguishable from zero under the cluster bootstrap. Special education (K-12 teaching on public pay scales) "
          f"keeps a clearly positive association and contradicts both readings.\n",
          # ---- 1. nursing, geography only
          f"1. **Nursing, net of geography only: weak, not shown to be absent.** The partial correlation of prestige "
          f"and pay given geography (table \"Nursing's partial r\" in (b)) is, on 35c's sample "
          f"(n = {int(Ni.loc[(P35, ST)].n)}), {npr_txt(P35, ST)} with state FE, {npr_txt(P35, RG)} with BEA-region FE "
          f"and {npr_txt(P35, SL)} with the 1-df state earnings level. On the full sample (n = {int(Ni.loc[(PF, ST)].n)}) "
          f"it is {npr_txt(PF, ST)}, {npr_txt(PF, RG)} and {npr_txt(PF, SL)}. The CI includes 0 in {n_ci0} of {n_ver}; "
          f"{npr_ex_txt}. Log-earnings and rank-based versions differ by at most {scale_dev:.2f}. Nursing is below "
          f"computer science and economics in all {cs_sig[1]} versions (Fisher-z p ≤ {csec_pmax:.3f}) and below "
          f"accounting in {ac_sig[0]} of {ac_sig[1]}" + (f" (not in: {ac_ns_txt})" if len(ac_ns) else "") + ". "
          f"Against biology the difference has p < 0.05 on 35c's sample in {bio35_sig[0]} of {bio35_sig[1]} codings "
          f"but in {bioF_sig[0]} of {bioF_sig[1]} on the full sample (state FE {pz(PF, ST, 'biology')}; region FE "
          f"{pz(PF, RG, 'biology')}; 1-df {pz(PF, SL, 'biology')}), where biology's own partial r excludes 0 in "
          f"{bio_excl_F} of {len(bio_prF)} codings. So, **netting geography only**, nursing's prestige information is "
          f"weak but on the full sample indistinguishable from biology's, which is positive. That comparison does not "
          f"survive netting selectivity (point 2). Incremental prestige R² (at most {fmt(nurse_inc.max(), 3)} for "
          f"nursing in all {len(nurse_inc)} versions) cannot separate \"none\" from \"moderate\": it equals partial "
          f"r² × (1 − R²(geo only)) (max deviation {ident['ident_dev']:.1e}), so with nursing's state-FE R²(geo) of "
          f"{r2g_st[P35]:.2f}–{r2g_st[PF]:.2f} an increment of {fmt(nurse_inc.max(), 3)} fits a partial r of "
          f"{imp_r[P35]:.2f}–{imp_r[PF]:.2f}. By |partial r| nursing ranks {rka_rng[0]}–{rka_rng[1]} of "
          f"{k_rng[0]}–{k_rng[1]} fields (lowest in {n_abs1} of {n_ver}); by signed partial r {rk_rng[0]}–{rk_rng[1]}. "
          f"With region FE, {len(lowlic)} of the fields below it have strict shares below {LOWLIC:.2f} ({lowlic_txt}).\n",
          # ---- 2. nursing, net of selectivity
          f"2. **Nursing, net of selectivity as well: about zero, with a small upper bound, as in most fields.** "
          f"scripts/55's broad partial coupling nets SAT, admission rate, Pell share, control and the state earnings "
          f"level (joint institution-bootstrap SEs; table \"Nursing after selectivity adjustment\" in (d)). Nursing's "
          f"value is {ns('rho_full_stlev', 'nursing')}, so its upper bound ({fmt(nb.hi)}) is much tighter than the "
          f"geography-only upper bounds (up to {fmt(geo_hi_max)}). That supports \"almost no pay information\" for "
          f"nursing. But it does not single nursing out. Biology is {ns('rho_full_stlev', 'biology')}, the same as "
          f"nursing (z = {fmt(zbio.z)}, {pe(zbio.p)}). Nursing ranks {rkb['rank']} of {rkb['k']} fields (field mean "
          f"{fmt(rkb['mean'])}). The fields below it (value; strict share) are {below_txt}. {rkb['n_inc0']} of "
          f"{rkb['k']} fields have a CI including 0, including {rkb['n_inc0_low']} of the {rkb['k_low']} fields with "
          f"strict share below 0.20. Nursing is still below computer science ({nsp('rho_full_stlev', 'computer_science')}) "
          f"and accounting ({nsp('rho_full_stlev', 'accounting')}), but not economics "
          f"({nsp('rho_full_stlev', 'economics')}). With state FE instead of the state earnings level (strict partial, "
          f"{rks['k']} fields) nursing is {ns('rho_full_state', 'nursing')} and cannot be told apart from biology "
          f"({nsp('rho_full_state', 'biology')}), computer science ({nsp('rho_full_state', 'computer_science')}), "
          f"economics ({nsp('rho_full_state', 'economics')}) or accounting ({nsp('rho_full_state', 'accounting')}). "
          f"Within institutions (institution fixed effects absorb all institution-level selectivity, brand and "
          f"geography), nursing's department-prestige slope is {ns('fe_a_beta', 'nursing')} in spec (a) and "
          f"{ns('fe_b_beta', 'nursing')} in spec (b), and biology's is {ns('fe_a_beta', 'biology')} and "
          f"{ns('fe_b_beta', 'biology')} (nursing − biology: {pe(NS.loc[('fe_a_beta', 'biology')].p)} and "
          f"{pe(NS.loc[('fe_b_beta', 'biology')].p)}). Nursing's slopes are below those of computer science, economics "
          f"and accounting (p ≤ {fe_pmax:.3f}). So once selectivity is handled, nursing's department-prestige coupling is "
          f"about zero with a small upper bound, as in biology and in many unlicensed fields. The evidence about "
          f"licensed fields as a class therefore rests on the cross-field gradient (points 4–5), not on nursing being "
          f"unusual.\n",
          # ---- 3. geography overfitting
          f"3. **The geography half is inflated by overfitting.** On 35c's sample nursing has {int(n35['raw'].n)} "
          f"institutions in {int(n35['raw'].n_geo)} states. Pure-noise state dummies would give an expected in-sample "
          f"R² of {chance35:.2f}, so the published {fmt(n35['raw'].R2_geo_only, 2, False)} is only "
          f"{n35['raw'].R2_geo_only - chance35:.2f} above chance. The adjusted R² is "
          f"{fmt(n35['adjusted'].R2_geo_only, 2, False)} and the cross-validated R² "
          f"{fmt(n35['LOO-CV'].R2_geo_only, 2, False)}. With all {int(nST.n)} institutions the cross-validated value is "
          f"{fmt(ncv[ST].R2_geo_only, 2, False)} (state FE) or {fmt(ncv[RG].R2_geo_only, 2, False)} (region FE), still "
          f"the highest of any field (rank {rkST[0]} of {rkST[1]} and {rkRG[0]} of {rkRG[1]}), so geography does "
          f"predict nursing pay. The UK agrees (nursing is the only UK subject whose geography R² holds up out of "
          f"sample: {fmt(nurseU.cv_R2_geo, 2, False)}; next-highest {fmt(ukcv_other, 2, False)}). 35c's cross-field "
          f"\"geography matters more as licensure rises\" does not survive cross-validation: {fmt(ig35[0][0])} "
          f"{fci(ig35[0][1], ig35[0][2])} raw, {fmt(st(P35, ST, 'LOO-CV', 'inc_geo')[0])} "
          f"{fci(*st(P35, ST, 'LOO-CV', 'inc_geo')[1:3])} under CV; across the 18 versions its CI excludes 0 only in "
          f"{sum(excl0(t) for t in ig35 + igF)}, none cross-validated; on the full sample it lies between "
          f"{fmt(min(t[0] for t in igF))} and {fmt(max(t[0] for t in igF))}.\n",
          # ---- 4. gradient: discipline structure
          f"4. **The strict-share gradient is a contrast between disciplines.** 35c's statistic, Spearman(strict "
          f"share, incremental prestige R²), is {fmt(ip35[0][0])} {fci(ip35[0][1], ip35[0][2])} "
          f"{fcl(ip35[0].clo, ip35[0].chi)} raw and {c3(cv35)} cross-validated on 35c's sample. Over 35c's 9 versions "
          f"the field-bootstrap CI excludes 0 in {sum(excl0(t) for t in ip35)} and the CIP-2 cluster CI in "
          f"{inc35_c}. On the full sample the counts are {sum(excl0(t) for t in ipF)} and {incF_c} of 9. The signed "
          f"partial r(F, earn | geo) is the better statistic: incremental R² ignores sign and shrinks wherever "
          f"geography explains a lot (see (b)). Spearman(strict share, signed partial r) is {rng(sg)}. Its CI excludes "
          f"0 in {sum(excl0(t) for t in sg)} of 6 versions under the field bootstrap but in {len(sg_excl_c)} of 6 under "
          f"the CIP-2 cluster bootstrap ({sg_c_txt}). The contrast runs between families. Family means (strict share; "
          f"partial r, full sample, region FE; field mean {fmt(pr_mean_all)}) are "
          f"{', '.join(fam_txt(c) for c in lowfam)} at the low-share end and {', '.join(fam_txt(c) for c in highfam)} "
          f"at the high-share end (table \"CIP-2 family means\" in (b)). Within CIP-2 families the rank association is {fmt(w_rng[0])} to "
          f"{fmt(w_rng[1])} (within-family permutation p = {wp_rng[0]:.2f}–{wp_rng[1]:.2f}; {gw_rng[0]}–{gw_rng[1]} "
          f"families with ≥ 2 fields). Only {shw_rng[0] * 100:.0f}–{shw_rng[1] * 100:.0f}% of the strict share's rank "
          f"variance lies within families, so this test has little power, but it finds no gradient. Most of that "
          f"within-family rank variance comes from "
          + ", ".join(f"{FAMN.get(k_, k_).lower()} ({v_ * 100:.0f}%; strict shares "
                      f"{FAM.loc[k_].licmin:.2f}–{FAM.loc[k_].licmax:.2f})" if k_ in FAM.index else
                      f"{FAMN.get(k_, k_).lower()} ({v_ * 100:.0f}%)" for k_, v_ in top_c)
          + f" (average over the 6 versions; in ranks, small share differences among low-share fields count fully). Health and Biology "
          f"contribute {min(hb) * 100:.0f}–{max(hb) * 100:.0f}%. Inside Health on the full sample with region FE, the ordering does not follow the "
          f"share: {hl_txt}. Leaving out one CIP-2 family at a time moves the gradient within {fmt(loco_rng[0])} to "
          f"{fmt(loco_rng[1])}. Without engineering it is {ci_list(ne35)} on 35c's sample (field CI includes 0 in "
          f"{sum(not excl0(t) for t in ne35)} of 3) and {ci_list(neF)} on the full sample (cluster CI "
          f"{', '.join(fcl(t.clo, t.chi) for t in neF)}). The young-worker share (ages 22–27) holds up slightly better under the "
          f"cluster bootstrap: cluster CI excludes 0 in {sgy_c} of 6 (table in (c)). So the gradient cannot be "
          f"separated from the discipline (CIP-2) structure that Phase 1 found borderline net of field size. Fields in "
          f"the same family also share institutions, so both bootstraps are approximate.\n",
          # ---- 5. broad share
          f"5. **With the broad licensure share there is no gradient.** The broad share adds health technicians, "
          f"accountants, engineers and counselors/social workers to the strict list. Spearman(broad share, signed "
          f"partial r) is {rng(sgb)} (CI includes 0 in {sgb_in} of 6 with the field bootstrap and {sgb_inc} of 6 with "
          f"the cluster bootstrap). Spearman(broad share, incremental prestige R²) is {rng(ipb)} (CI includes 0 in "
          f"{ipb_in} of {len(ipb)}). Spearman(broad share, selectivity-adjusted partial coupling) is "
          f"{rng([r_.rho_t for _, r_ in part_rows_b])} (CI includes 0 in {n_partb_in} of {len(part_rows_b)} with the "
          f"field bootstrap and {n_partb_inc} of {len(part_rows_b)} with the cluster bootstrap). The added occupations "
          f"are those of engineering, accounting and social-work graduates; the baseline couplings of these fields "
          f"are {fmt(base_eng)} (engineering, mean), {fmt(float(base_all.get('accounting', np.nan)))} (accounting) "
          f"and {fmt(float(base_all.get('social_work', np.nan)))} (social work), against a field mean of "
          f"{fmt(base_mean)}. The gap-level slope on the broad share (scripts/20's +0.45, here {fmt(A2c.b, 3)} "
          f"{fci(A2c.lo, A2c.hi)}) has a field-bootstrap CI excluding 0, but its cluster CI {fcl(A2c.clo, A2c.chi)} "
          + ("includes 0. " if not (A2c.clo > 0 or A2c.chi < 0) else "also excludes 0. ")
          + f"Selectivity pricing and brand coupling, which are not department-prestige couplings, do fall with the "
          f"broad share (table in (d)). One department-prestige statistic also does. The within-institution slope of spec "
          f"(a) (institution FE, no field-specific selectivity slopes) falls with the broad share, "
          f"{c3(fea_b.rho_t)}, also within CIP-2 families ({fmt(fea_b.cl['r_within'])}, p = "
          f"{fea_b.cl['p_within']:.3f}). It weakens once field-specific selectivity slopes are added (spec (b) "
          f"{fmt(feb_b.rho)}, spec (c) {fmt(fec_b.rho)}). {n_eng_below} of the {len(fe_below)} fields whose spec (a) "
          f"slope is below nursing's are engineering fields (family mean broad share {FAM.loc[ENG].licb:.2f}, against a "
          f"field mean of {fam_d.lic_broad.mean():.2f}). The claim should therefore "
          f"read \"falls with the strict (practitioner, teacher, lawyer) licensure share; no gradient with the broad "
          f"share\". That fits \"the credential is not the mechanism\", and it also makes the gradient specific to "
          f"how licensure is defined.\n",
          # ---- 6. selectivity and the gradient
          f"6. **Selectivity adjustment removes most of the level of coupling; the strict-share gradient keeps its sign "
          f"but not its cluster-robust precision everywhere.** On the same {int(bro.k)} fields, Spearman(strict share, "
          f"coupling) is {c3(bro_raw.rho_t)} raw and {c3(bro.rho_t)} for the broad selectivity-adjusted partial, while "
          f"mean coupling falls to {pr_b.level_ratio * 100:.0f}% of raw. The WLS slope moves {fmt(bro_raw.b)} → "
          f"{fmt(bro.b)} (paired difference {fmt(pr_b['diff'])} {fci(pr_b.diff_lo, pr_b.diff_hi)}; across the 6 "
          f"matched comparisons the change is {fmt(rho_pairs['diff'].min())} to {fmt(rho_pairs['diff'].max())}, CI "
          f"includes 0 in {pair_ci0} of {len(rho_pairs)}). Across the {len(part_rows)} selectivity-adjusted partial "
          f"couplings, the strict-share Spearman CI excludes 0 in {n_part_ex} of {len(part_rows)} under the field "
          f"bootstrap and in {n_part_exc} of {len(part_rows)} under the cluster bootstrap"
          + ((" (not for " + "; ".join(f"`{c_}` {c3(r_.rho_t)}" for c_, r_ in part_ns_c)
              + (("; at the edge: " + "; ".join(f"`{c_}` {fcl(r_.rho_clo, r_.rho_chi, 3)}" for c_, r_ in part_edge_c))
                 if part_edge_c else "") + ")") if part_ns_c else "")
          + f". Within CIP-2 families these associations are {fmt(min(part_w))} to {fmt(max(part_w))} "
          f"(p ≥ {min(part_wp):.2f}). The within-institution slopes point the same way with CIs including 0 "
          f"(Spearman {c3(fea.rho_t)} and {c3(feb.rho_t)}, specs a and b). Licensed fields also price selectivity itself "
          f"less (WLS slope of ρ(SAT, earn) on the strict share {fmt(sat_.b)} {fci(sat_.b_lo, sat_.b_hi)}; of "
          f"ρ(−ADM, earn) {fmt(adm_.b)} {fci(adm_.b_lo, adm_.b_hi)}), so the licensing gradient cannot tell prestige "
          f"pricing from selectivity pricing: that is the scripts/55 collinearity again, now across fields.\n",
          # ---- 7. what rests on a few fields
          f"7. **What rests on a few fields.** Without nursing, communication disorders and the five pre-med "
          f"sciences, the strict-share b_licensure on the gap is {c2(QB, sub=BOTH)} (HC1 ⟨cluster⟩) and the "
          f"selectivity-adjusted Spearman {c2(QSEL, sub=BOTH)}. The signed geography-netted gradient on 35c's fields "
          f"stays at {rng(pb35)} (field CI excludes 0 in {sum(excl0(t) for t in pb35)} of 3, cluster CI in "
          f"{sum(exclc(t) for t in pb35)} of 3) and on the full field set weakens to {rng(pbF)} (field CI excludes 0 in "
          f"{pbF_excl} of 3). The gap-level strict-share slope itself is {fmt(A1c.b, 3)} {fci(A1c.lo, A1c.hi)} "
          f"{fcl(A1c.clo, A1c.chi)} (field ⟨cluster⟩ bootstrap) and without engineering {fmt(b_noeng.est, 3)} "
          f"{fcl(b_noeng.clo, b_noeng.chi)}. "
          + ("Leaving out any single field never flips the sign of an (a) or (b) estimate.\n"
             if (loo_a_pos and loo_b_neg) else "Leaving out single fields can flip the sign of some (a) or (b) "
             "estimates (see the leave-one-field-out columns).\n"),
          # ---- 8. three b_licensure numbers
          f"8. **The three b_licensure numbers measure different things; the multilevel one is not a collapse.** "
          f"+0.66 (strict) and +0.45 (broad) are slopes on a continuous ACS licensure share (0–1). The multilevel "
          f"+0.063 is the shift for a hand-coded 4-field tag {{nursing, communication disorders, accounting, civil "
          f"engineering}}, which correlates {fmt(corr_tag)} with the measured strict share. Fitted with the continuous "
          f"share, the same multilevel model gives {fmt(Ai.loc['A6'].b)} {fci(Ai.loc['A6'].lo, Ai.loc['A6'].hi)}. The "
          f"drop to +0.063 comes from the swap of regressor: accounting and civil engineering are tagged but have low "
          f"strict shares ({tag_txt}) and gaps at their CIP-2 siblings' level. On one scale (implied nursing − CS gap; "
          f"observed {fmt(gm_obs)}) the three give {fmt(Ai.loc['A1'].implied)}, {fmt(Ai.loc['A2'].implied)} and "
          f"{fmt(Ai.loc['A8'].implied)}. Without nursing and communication disorders: strict WLS "
          f"{fmt(Ai.loc['A1'].b_drop)}, broad WLS {fmt(Ai.loc['A2'].b_drop)}, multilevel-continuous "
          f"{fmt(Ai.loc['A6'].b_drop)}, multilevel-binary {fmt(Ai.loc['A8'].b_drop, 3)}.\n",
          # ---- 9. mechanism
          f"9. **\"Pay set by setting\" is an interpretation, not a finding of this battery.** The first version of "
          f"this report called setting \"the operative variable\"; nothing in it measured setting, so that was an "
          f"overclaim. This revision adds one measured proxy: the share of a field's employed BA+ holders who work for "
          f"government (ACS PUMS 2023 class of worker 3–5, the population scripts/20 uses; table (c3)). It predicts "
          f"the signed partial r somewhat less consistently than the strict share: Spearman {rng(gov6)}, field CI "
          f"excludes 0 in {sum(excl0(t) for t in gov6)} of 6 and cluster CI in {sum(exclc(t) for t in gov6)} of 6, "
          f"against {sum(excl0(t) for t in sg)} and {len(sg_excl_c)} of 6 for the strict share. With nonprofit "
          f"employers added it does at least as well: {rng(govnp6)} ({sum(excl0(t) for t in govnp6)} and "
          f"{sum(exclc(t) for t in govnp6)} of 6), although that share overlaps the strict list (hospitals and "
          f"schools are often nonprofit or public employers). Across these fields the government share and the strict share correlate {fmt(lic_gov_rng[0])} to "
          f"{fmt(lic_gov_rng[1])}. Holding the other fixed (rank partial correlation), the strict share gives "
          f"{fmt(min(t[0] for t in lg6))} to {fmt(max(t[0] for t in lg6))} and the government share "
          f"{fmt(min(t[0] for t in gl6))} to {fmt(max(t[0] for t in gl6))}. Under the cluster bootstrap the strict "
          f"share's CI excludes 0 in {sum(pex(t, True) for t in lg6)} of 6 versions and the government share's in "
          f"{sum(pex(t, True) for t in gl6)} of 6"
          + (lambda seps: (f"; over all {len(SET)} outcomes in (c3) the two separate in {len(seps)}, "
                           + ("each time in favour of the government share" if all(w_ == "government" for w_ in seps)
                              else "in favour of the " + ", ".join(seps) + " share respectively")
                           + " (see (c3))") if seps else "")(
              ["government" if pex(r_["gov_given_lic"], True) else "strict" for r_ in SET
               if pex(r_["gov_given_lic"], True) != pex(r_["lic_given_gov"], True)])
          + f". So a measured setting proxy does roughly as well as the licence "
          f"share, and this battery cannot tell them apart. Three fields bear on the readings:\n",
          f"   - *Accounting* counters the binary licensing tag and the broad share, not the strict-share gradient. Its "
          f"strict share is {acc_s:.3f}, so its high coupling (partial r | geography {fmt(min(acc_pr))} to "
          f"{fmt(max(acc_pr))}; broad selectivity partial {ns('rho_full_stlev', 'accounting')}) is what the strict "
          f"gradient predicts. Its \"CPA-licensed\" status is also loose: the broad share counts ACS SOC 13-2011, which "
          f"covers all accountants and auditors, not only CPAs.\n",
          f"   - *Special education* contradicts both the licence reading and the setting reading. Its graduates work "
          f"mostly as K-12 teachers (strict share {float(ach.loc['special_education', 'licensure_strict']):.2f}; "
          f"government share {spd.gov:.2f}, rank {gov_rank_sp} of {len(sect)} fields), the textbook case of pay set "
          f"by a district salary schedule. Yet its coupling is positive: baseline {fmt(sp.rho_raw)}, partial r | "
          f"geography {fmt(min(x.partial_r for x in sp_pr))} to {fmt(max(x.partial_r for x in sp_pr))} (CI excludes 0 "
          f"in {sp_ex} of {len(sp_pr)} codings), broad selectivity partial {fmt(sp.rho_full_stlev)} (SE "
          f"{sp.se_rho_full_stlev:.2f}), and within-institution slope (spec b) {fmt(sp.fe_b_beta, 3)} (SE "
          f"{sp.fe_b_se:.3f}). The limit is size: {n_of['special_education']} institutions, full sample only.\n",
          f"   - *Social work* (broad share {float(ach.loc['social_work', 'licensure_broad']):.2f}; government + "
          f"nonprofit share {swk.govnp:.2f}) has partial r | geography {fmt(min(x.partial_r for x in sw_pr))} to "
          f"{fmt(max(x.partial_r for x in sw_pr))} (CI excludes 0 in {sw_ex} of {len(sw_pr)}) and broad selectivity "
          f"partial {fmt(swS.rho_full_stlev)} (SE {swS.se_rho_full_stlev:.2f}): a mostly public or nonprofit field "
          f"with moderate coupling.\n",
          f"10. **UK: nursing is at the bottom, but not uniquely.** With adjusted / CV R², UK nursing's incremental "
          f"prestige is {fmt(nurseU.adj_inc_p, 3)} / {fmt(nurseU.cv_inc_p, 3)} (rank {int(nurseU.rank_adj)} / "
          f"{int(nurseU.rank_cv)} of {len(U)}, 1 = lowest), and its partial r is {fmt(nurseU.partial_r)} "
          f"{fci(nurseU.partial_lo, nurseU.partial_hi)}. The other bottom-three subjects by adjusted R² are "
          f"unlicensed: "
          + "; ".join(f"{r.cah2} {fmt(r.adj_inc_p, 3)} / {fmt(r.cv_inc_p, 3)}"
                      for _, r in U[~U.licensed].sort_values("adj_inc_p").head(2).iterrows()) + ". "
          f"Nursing is the only usable licensed subject, so the UK gives one consistent case, not a gradient.\n",
          "**Suggested wording.** \"Across US fields, how strongly a department's academic prestige is associated "
          "with graduates' pay net of geography falls with the share of graduates in strictly licensed occupations "
          f"(health practitioners, K-12 teachers, lawyers; Spearman {rng(sg)} across samples and geography codings). "
          "The pattern is a contrast between disciplines: engineering, business and computing on one side, health "
          "and the biological sciences on the other. It is not detectable within CIP-2 families, and when whole "
          f"families are resampled it is distinguishable from zero {cl_where_s}. It does not "
          "appear with a broader licensure definition that adds engineers, accountants and social workers. A field's "
          "public and nonprofit employment share predicts it about as well, so these data do not separate licensure "
          "from employer sector. In nursing, once selectivity is netted as well as geography, department prestige carries "
          f"almost no pay information ({fmt(nb.est)} {fci(nb.lo, nb.hi)}). The same holds for biology and for most "
          "other fields.\" Do not write that prestige carries \"no pay information beyond geography in licensed "
          "fields\", that \"geography explains ~71% of nursing pay\", that \"geography adds more as licensure "
          "rises\", or that \"the operative variable is pay set by setting\". Keep 35c's attribution caveat: the "
          "licensure share marks a regulated / public / local bundle, not the credential.\n"]

    # =============================== what changed ===============================
    L += ["## What changed in this revision\n",
          "Independent verifiers raised five major issues. Each is handled as follows; the numbers above and below "
          "are all recomputed.\n",
          f"- **Nursing verdict ignored selectivity** (issue 1). Added point 2 and the table \"Nursing after "
          f"selectivity adjustment\" in (d). \"Weak, not none; indistinguishable from biology\" is now limited to "
          f"geography-only netting. Net of selectivity, nursing's coupling is {fmt(nb.est)} {fci(nb.lo, nb.hi)}, the "
          f"same as biology's, and nursing ranks {rkb['rank']} of {rkb['k']}. The \"licensed fields\" evidence now "
          f"rests on the cross-field gradient.\n",
          "- **\"The operative variable is pay set by setting\" was not tested** (issues 2 and 5). Withdrawn. \"Pay set "
          "by setting\" is now presented as the project's interpretation. A measured proxy (government / nonprofit "
          "employment share from the raw ACS PUMS, section (c3)) was added and tested against the signed partial r "
          "and the selectivity-adjusted couplings. Accounting is restated as a counter-case to the binary tag and "
          "the broad share only; special education and social work are listed as counter-cases to the setting "
          "reading too.\n",
          f"- **Fields treated as independent draws** (issue 3). Every field-level Spearman in (b)–(d) now also has "
          f"a CIP-2 cluster-bootstrap CI (B = {B_CL}; ⟨ ⟩), and the gap-level WLS slopes in (a) and (c) have one too. "
          f"New: the within-CIP-2 rank association with a within-family permutation test, leave-one-family-out "
          f"ranges, and a \"without engineering (CIP-14)\" column. The headline changed from \"CI excludes 0 in 6/6\" "
          f"to \"{sum(excl0(t) for t in sg)}/6 with the field bootstrap, {len(sg_excl_c)}/6{cl_where} with the "
          f"cluster bootstrap\". \"Survives cross-validation\" is qualified the same way: CV {c3(cv35)}.\n",
          "- **Broad-share results were computed but not reported** (issue 4). Added broad-share rows and columns in "
          "(b), (c) and (d), and the claim is restricted to the strict share.\n",
          "- Figure: panel (b′) adds the two selectivity-adjusted couplings for nursing and the comparison fields. "
          "Panel (c) now shows the signed gradient under the field and the cluster bootstrap, within families, "
          "without engineering, and with the broad share.\n"]

    # =============================== (a) ===============================
    L += ["## (a) b_licensure: strict, broad and multilevel side by side\n",
          "Outcome = gap_f = 1 − ρ_f (Scorecard FoS 4-yr median; Wapman field rank), ALL_FIELDS universe. "
          "`implied` = b × (x_nursing − x_CS): the nursing-minus-computer-science gap difference that the "
          f"specification attributes to licensure (observed difference {fmt(gm_obs)}). CI = field bootstrap "
          f"(rows A1–A2: scripts/20's own B = 2000, seed 11; other rows B = {B_BRIDGE}). ⟨cluster⟩ = CIP-2 cluster "
          f"bootstrap (B = {B_CL}; WLS rows with a continuous share only; the REML rows already model CIP-2 with a "
          "random intercept). Model CI = HC1 (WLS) or GLS (REML) ± 1.96 SE. Rows marked → change one thing relative to "
          "the row above.\n",
          "| row | specification | n | regressor | b [95% boot CI] | ⟨CIP-2 cluster CI⟩ | model CI | b per SD of regressor | implied nursing − CS [CI] | b without nursing + comm. dis. | leave-one-field-out range (field giving the min) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    arrow = {"A3", "A4", "A5", "A6", "A8"}
    for k, r in Ai.iterrows():
        L.append(f"| {k} | {'→ ' if k in arrow else ''}{r.label} | {r.n} | {r.regressor}"
                 f"{f' ({int(r.n_lic)} tagged)' if np.isfinite(r.n_lic) else ''} | {fmt(r.b, 3)} {fci(r.lo, r.hi)} | "
                 f"{fcl(r.clo, r.chi) or '—'} | {fci(r.lo_model, r.hi_model)} | {fmt(r['std'], 3)} | "
                 f"{fmt(r.implied)} {fci(r.implied_lo, r.implied_hi)} | "
                 f"{fmt(r.b_drop, 3)} (n={r.n_drop}) | {fmt(r.loo_min, 3)} … {fmt(r.loo_max, 3)} ({LAB.get(r.loo_min_f, r.loo_min_f)}) |")
    L += ["",
          f"Reproduction: A1 = {fmt(Ai.loc['A1'].b, 3)} {fci(Ai.loc['A1'].lo, Ai.loc['A1'].hi)} vs published +0.66 "
          f"[+0.33, +0.97]; A2 = {fmt(Ai.loc['A2'].b, 3)} {fci(Ai.loc['A2'].lo, Ai.loc['A2'].hi)} vs +0.45 [+0.05, +0.72]; "
          f"A8 = {fmt(Ai.loc['A8'].b, 3)} (GLS CI {fci(Ai.loc['A8'].lo_model, Ai.loc['A8'].hi_model)}) vs +0.063 "
          "[−0.05, +0.18].\n",
          "**Why +0.063 is not a collapse of +0.66.** The two use different regressors on different scales and "
          "compare different sets of fields.\n",
          f"- *Regressor.* A1/A2 use the ACS 2023 share of a field's employed BA+ holders in licensed occupations "
          f"(0–1). Strict = health practitioners, lawyers, K-12 teachers, architects, psychologists; broad adds health "
          f"technicians, accountants and auditors (SOC 13-2011), engineers, counselors/social workers. A8 uses the "
          f"ex-ante 4-field binary tag. Pearson(tag, strict share) = {fmt(corr_tag)} on the scripts/13 sample. Special "
          f"education and pharmacy have strict shares ≥ 0.5 but are untagged, while two tagged fields have low strict "
          f"shares ({tag_txt}).\n",
          "- *Comparison set.* With a CIP-2 random intercept, the binary shift is identified mostly from each tagged "
          "field's difference from its untagged CIP-2 siblings:\n",
          "| tagged field | CIP-2 | gap (SE) | strict share | broad share | mean gap of untagged CIP-2 siblings | siblings |",
          "|---|---|---|---|---|---|---|"]
    for _, r in SIB.iterrows():
        L.append(f"| {LAB.get(r.field, r.field)} | {r.cip2} | {r.gap:.2f} ({r.se:.2f}) | {r.lic_strict:.2f} | "
                 f"{r.lic_broad:.2f} | {r.sib_gap:.2f} (n={r.n_sib}) | {r.sib_fields} |")
    L += ["",
          "  Accounting and civil engineering sit at their siblings' level with small SEs. Nursing and communication "
          "disorders sit well above their siblings, but with larger SEs, so the precision weights favour the first "
          "two.\n",
          f"- *Bridge.* Going from A1 to the scripts/13 model one step at a time: dropping absorption "
          f"({fmt(Ai.loc['A3'].b, 3)}), switching to the scripts/13 sample and SE floor ({fmt(Ai.loc['A4'].b, 3)}) and "
          f"adding dispersion ({fmt(Ai.loc['A5'].b, 3)}) change little. Adding the CIP-2 random intercept lowers it to "
          f"{fmt(Ai.loc['A6'].b, 3)} {fci(Ai.loc['A6'].lo, Ai.loc['A6'].hi)} (GLS CI "
          f"{fci(Ai.loc['A6'].lo_model, Ai.loc['A6'].hi_model)}). The high-share fields nursing, communication "
          f"disorders and pharmacy share CIP-51, so part of the contrast becomes a Health cluster effect; the CIP-2 "
          f"cluster bootstrap of A1 ({fcl(A1c.clo, A1c.chi)}) makes the same point. Swapping to the binary tag then "
          f"gives {fmt(Ai.loc['A8'].b, 3)}. The binary tag is also small without the random intercept (A9 "
          f"{fmt(Ai.loc['A9'].b, 3)}, A10 {fmt(Ai.loc['A10'].b, 3)}). Binarising the measured share at 0.5 (A11; "
          f"cutoff fixed in advance, not tuned) gives {fmt(Ai.loc['A11'].b, 3)} {fci(Ai.loc['A11'].lo, Ai.loc['A11'].hi)}.\n",
          f"- *Nursing and communication disorders.* Without them: A1 {fmt(Ai.loc['A1'].b_drop, 3)}, A2 "
          f"{fmt(Ai.loc['A2'].b_drop, 3)}, A6 {fmt(Ai.loc['A6'].b_drop, 3)}, A8 {fmt(Ai.loc['A8'].b_drop, 3)} (point "
          f"estimates, n = {Ai.loc['A1'].n_drop}; the table in (c) gives HC1 and cluster CIs for A1-type fits).\n"]

    # =============================== (b) ===============================
    L += ["## (b) Incremental R² of prestige vs geography: raw, adjusted, cross-validated\n",
          "Per field: earnings (Scorecard FoS 4-yr median, levels) on field prestige F (−Wapman field rank) and a "
          "geography block. incremental prestige = R²(F + geo) − R²(geo); incremental geography = R²(F + geo) − R²(F). "
          "Adjusted R² = 1 − (1 − R²)(n − 1)/(n − p). LOO-CV R² = 1 − Σ(y − ŷ₋ᵢ)²/Σ(y − ȳ)². When a held-out "
          "institution's state or region has no other institution in the training fold, it is predicted from the "
          "count-weighted mean of the training intercepts, so a singleton state dummy carries no out-of-sample "
          "information. CV increments are not bounded by 1: when the geography-only model predicts worse than the "
          "mean, adding prestige can recover more than 100% of the total variance. Adjusted and CV values are "
          "undefined when the full model has zero residual df. Geography codings: state FE (35c); BEA region FE "
          "(8 regions, Scorecard `REGION`); state earnings level = leave-one-out mean log MD_EARN_WNE_P10 of the "
          "state's other predominantly-bachelor's institutions (1 df; scripts/55 `ST_EARN`). Samples: 35c's "
          "(institutions on the PSEO institution list, which is where 35c takes the state from) and full (every "
          "matched cell, with the Scorecard state). Field eligibility as in 35c: n ≥ 15, ≥ 3 states, ACS licensure "
          "available. Correlation CIs: 35c's field bootstrap (B = 5000, seed 7) [ ] and the CIP-2 cluster bootstrap "
          f"(B = {B_CL}) ⟨ ⟩.\n",
          f"Reproduction: on 35c's sample with state FE and raw R², {repro_b['n_match']}/{repro_b['n_ref']} fields "
          f"match `er_axis_35c_mechanism.csv` (max |Δ incremental prestige| = {repro_b['max_abs_diff']:.1e}). "
          f"Cells: {cellinfo['cells']}; Scorecard state missing {cellinfo['stabbr_missing']}; on the PSEO list "
          f"{cellinfo['pseo_state']}; where both exist, the PSEO-list state equals the Scorecard STABBR in "
          f"{cellinfo['state_agree'] * 100:.0f}% of cells; ST_EARN missing {cellinfo['st_earn_missing']}. "
          f"`valuation_residuals.csv` equals a fresh scripts/28 rebuild on its cells: {cellinfo['fresh_rebuild_match']}.\n",
          "**Spearman(licensure_strict, ·) across fields** [field CI] ⟨cluster CI⟩ (k fields):\n",
          "| sample | geography | R² | incremental prestige | same, on 35c's fields | incremental geography | R²(geo only) | R²(prestige only) | incr. prestige, fields with residual df ≥ 10 | leave-one-field-out range (incr. prestige) | nursing's rank in R²(geo only) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in Bsum:
        c35 = r["inc_prestige_common35c"]
        L.append(f"| {r['sample']} | {r['coding']} | {r['measure']} | {c3(r['inc_prestige'], True)} | "
                 f"{c3(c35, True)} | {c3(r['inc_geo'])} | {c3(r['R2_geo_only'])} | {c3(r['R2_prestige_only'])} | "
                 f"{fmt(r['inc_prestige_dfr10'][0])} (k={r['inc_prestige_dfr10'][3]}) | "
                 f"{fmt(r['loo_range'][0])} … {fmt(r['loo_range'][1])} | {r['nurse_geo_rank'][0]} of {r['nurse_geo_rank'][1]} |")
    L += ["",
          "**Incremental prestige R²: discipline structure and the broad share.** Within CIP-2 = rank association "
          "after demeaning the across-field ranks of licensure and of the outcome within CIP-2 families (families "
          f"with ≥ 2 fields), with a within-family permutation p (B = {B_PERM}). Leave-one-family-out = range over "
          "dropping each CIP-2 family:\n",
          "| sample | geography | R² | strict share [field] ⟨cluster⟩ | within CIP-2 (p) | leave-one-family-out range | without engineering (CIP-14) | broad share [field] ⟨cluster⟩ |",
          "|---|---|---|---|---|---|---|---|"]
    for r in Bsum:
        d_ = r["cl_inc"]
        L.append(f"| {SHORT[r['sample']]} | {r['coding']} | {r['measure']} | {c3(r['inc_prestige'], True)} | "
                 f"{fmt(d_['r_within'])} ({d_['p_within']:.2f}) | {fmt(d_['loco_min'])} … {fmt(d_['loco_max'])} | "
                 f"{c3(d_['no_eng'], True)} | {c3(r['inc_prestige_broadlic'], True)} |")
    added = sorted(set(Dfull[Dfull["sample"] == PF].field) - set(Dfull[Dfull["sample"] == P35].field))
    L += ["",
          f"Fields admitted only by the full sample: {', '.join(LAB.get(f, f) for f in added)}.\n",
          "**Named fields** (both samples; cell = incremental prestige / incremental geography / R²(geo only); "
          "partial r does not depend on the R² measure; — = undefined because the full model has no residual df):\n",
          "| field | sample | licensure (strict) | geography | raw | adjusted | LOO-CV | partial r(F, earn \\| geo) [95% CI] |",
          "|---|---|---|---|---|---|---|---|"]
    for f in ["nursing", "accounting", "special_education", "communication_disorders", "social_work", "public_health",
              "biology", "computer_science", "economics"]:
        for samp in (P35, PF):
            for cod in CODINGS:
                vals, pr = [], None
                for meas in MEASURES:
                    h = nf(samp, cod, meas, f)
                    if h is not None:
                        vals.append(f"{fmt(h.inc_prestige, 3)} / {fmt(h.inc_geo, 3)} / {fmt(h.R2_geo_only, 2, False)}")
                        pr = h
                    else:
                        vals.append("—")
                if pr is None:
                    continue
                L.append(f"| {LAB.get(f, f)} (n={int(pr.n)}, {int(pr.n_geo)} geo levels, residual df {int(pr.dfr_full)}) | "
                         f"{SHORT[samp]} | {pr.licensure:.2f} | {cod} | " + " | ".join(vals)
                         + f" | {fmt(pr.partial_r)} {fci(pr.partial_lo, pr.partial_hi)} |")
    L += ["",
          "**Nursing's partial r(F, earnings | geography) in all six sample × coding versions**, with its rank among "
          "the eligible fields (1 = lowest). Incremental R² is sign-blind, so its rank is closer to the |partial r| "
          "rank:\n",
          "| sample | geography | n | geo levels | partial r [95% CI] | CI includes 0 | rank by partial r | rank by \\|partial r\\| | rank by raw incr. prestige R² | same two ranks, fields with residual df ≥ 10 | fields with a lower partial r (strict licensure share) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in NPR.iterrows():
        L.append(f"| {SHORT[r['sample']]} | {r.coding} | {r.n} | {r.n_geo} | {fmt(r.r)} {fci(r.lo, r.hi)} | "
                 f"{'yes' if r.ci0 else 'no'} | {r['rank']} of {r.k} | {r.rank_abs} of {r.k} | {r.rank_inc} of {r.k} | "
                 f"{r.rank10} / {r.rank_abs10} of {r.k10} | "
                 f"{', '.join(f'{LAB.get(x, x)} ({r.lic_of[x]:.2f})' for x in r.below) if r.below else '—'} |")
    L += ["",
          f"35c's sample, nursing, state FE (published: incremental prestige 0.000, incremental geography 0.71): "
          f"raw {fmt(n35['raw'].inc_prestige, 3)} / {fmt(n35['raw'].inc_geo, 3)} / R²(geo) "
          f"{fmt(n35['raw'].R2_geo_only, 2, False)}; adjusted {fmt(n35['adjusted'].inc_prestige, 3)} / "
          f"{fmt(n35['adjusted'].inc_geo, 3)} / {fmt(n35['adjusted'].R2_geo_only, 2, False)}; CV "
          f"{fmt(n35['LOO-CV'].inc_prestige, 3)} / {fmt(n35['LOO-CV'].inc_geo, 3)} / "
          f"{fmt(n35['LOO-CV'].R2_geo_only, 2, False)}. That is {int(n35['raw'].n)} institutions in "
          f"{int(n35['raw'].n_geo)} states, and a pure-noise block of {int(n35['raw'].n_geo) - 1} dummies has "
          f"expected in-sample R² ≈ {chance35:.2f}.\n",
          "Nursing across BEA regions (a possible reason why prestige alone explains so little in nursing; "
          f"descriptive, not established; full-sample R²(prestige only) = {fmt(nST.R2_prestige_only, 3, False)}): "
          + "; ".join(f"{SHORT[sp_]} ({nr['n']} institutions): Pearson(region mean F, region mean earnings) = "
                      f"{fmt(nr['between'])} over {nr['k_reg']} regions; best-paying region "
                      f"{BEA.get(nr['top_pay_region'], '?')} (BEA code {nr['top_pay_region']}; mean ${nr['top_pay']:,.0f}), "
                      f"mean-prestige rank {nr['F_rank_of_top_pay']} of {nr['k_reg']} (1 = lowest); Spearman of "
                      f"region-demeaned F and earnings = {fmt(nr['within'])} (p = {nr['within_p']:.3f}, ignoring that "
                      "the region means are estimated)"
                      for sp_, nr in ((PF, nrF), (P35, nr35))) + ".\n",
          "**Nursing vs comparison fields, net of geography only** (partial r(F, earn | geo); Fisher-z test of nursing "
          "minus the comparison field, z = (atanh r₁ − atanh r₂)/√(1/(n₁ − q₁ − 3) + 1/(n₂ − q₂ − 3)), treating the "
          "two fields as independent samples; they share institutions, so p is approximate). Nursing's partial r is "
          "shown on three earnings scales (levels / log / ranks of earnings and F). The comparisons after "
          "selectivity adjustment are in (d):\n",
          "| sample | geography | nursing: level / log / rank | biology [95% CI] | nursing − biology: z (p) | computer science: r, z (p) | economics: r, z (p) | accounting: r, z (p) |",
          "|---|---|---|---|---|---|---|---|"]
    for (samp, cod), g in NCMP.groupby(["sample", "coding"], sort=False):
        gi = g.set_index("comp")
        bi = gi.loc["biology"]
        L.append(f"| {SHORT[samp]} | {cod} | {fmt(bi.r_n)} / {fmt(bi.r_n_log)} / {fmt(bi.r_n_rank)} | "
                 f"{fmt(bi.r_c)} {fci(bi.lo_c, bi.hi_c)} (log {fmt(bi.r_c_log)}, rank {fmt(bi.r_c_rank)}) | "
                 f"{fmt(bi.z)} ({pv(bi.p)}) | "
                 + " | ".join(f"{fmt(gi.loc[c_].r_c)}, {fmt(gi.loc[c_].z)} ({pv(gi.loc[c_].p)})" if c_ in gi.index else "—"
                              for c_ in ("computer_science", "economics", "accounting")) + " |")
    L += ["",
          f"Incremental prestige R² equals partial r² × (1 − R²(geo only)) for raw R² (max deviation "
          f"{ident['ident_dev']:.1e} over {ident['ident_n']} field × version cells). It is sign-blind and shrinks "
          f"wherever geography explains a lot. Biology's full-sample incremental prestige R² ranges from "
          f"{fmt(bio_incF.min(), 3)} to {fmt(bio_incF.max(), 3)} over its {len(bio_incF)} versions, and nursing's from "
          f"{fmt(nurse_inc.min(), 3)} to {fmt(nurse_inc.max(), 3)} over {len(nurse_inc)}. Among the {len(ADDED)} small "
          f"fields that only the full sample admits are {len(neg_add)} with strict shares of "
          f"{neg_add.lic.min():.2f}–{neg_add.lic.max():.2f} (median field {lic_medF:.2f}) and negative partial r in every "
          f"coding where it is defined: {neg_add_txt}. Their raw incremental prestige R² is "
          f"{fmt(float(np.nanmin(neg_inc)), 3)} to {fmt(float(np.nanmax(neg_inc)), 3)}, so incremental R² counts these "
          f"negative associations as prestige information. That is why the full-sample incremental-R² gradient is "
          f"weaker than the signed one; with |partial r| the full-sample state-FE and 1-df versions fall back to "
          f"{fmt(absF[ST][0])} {fci(absF[ST][1], absF[ST][2])} and {fmt(absF[SL][0])} {fci(absF[SL][1], absF[SL][2])}.\n",
          "**Signed partial r: Spearman(licensure, partial r(F, earn | geo)) across fields** [field CI] ⟨cluster CI⟩. "
          "The partial r does not depend on the R² measure. Fields whose full model has no residual df are excluded. "
          "The last two columns use the same fields as the signed column:\n",
          "| sample | geography | k | strict share | broad share | same, on 35c's fields | fields with residual df ≥ 10 | log earnings | rank-based | leave-one-field-out range (strict) | \\|partial r\\| (strict) | raw incr. prestige R², same fields |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r_ in PRG:
        L.append(f"| {SHORT[r_['sample']]} | {r_['coding']} | {r_['k']} | {c3(r_['signed'])} | {c3(r_['signed_broad'])} | "
                 f"{c3(r_['signed_35cfields'], True)} | {c3(r_['signed_dfr10'], True)} | {c3(r_['signed_log'])} | "
                 f"{c3(r_['signed_rank'])} | {fmt(r_['loo_range'][0])} … {fmt(r_['loo_range'][1])} | {c3(r_['abs'])} | "
                 f"{c3(r_['inc_prestige_same_fields'])} |")
    L += ["",
          "**Discipline structure of the signed gradient.** Share within = share of the licensure measure's "
          "across-field rank variance that lies within CIP-2 families (the power of the within-family test):\n",
          "| sample | geography | licensure | k (CIP-2 families) | Spearman [field] ⟨cluster⟩ | within CIP-2 (permutation p) | families with ≥ 2 fields | share within | leave-one-family-out range (family giving min / max) | without engineering (CIP-14) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r_ in PRG:
        for nm, key, dk in (("strict", "signed", "cl"), ("broad", "signed_broad", "cl_broad")):
            d_ = r_[dk]
            t_ = r_[key]
            L.append(f"| {SHORT[r_['sample']]} | {r_['coding']} | {nm} | {t_[3]} ({t_.g}) | {c3(t_)} | "
                     f"{fmt(d_['r_within'])} ({d_['p_within']:.2f}) | {d_['g_within']} ({d_['k_within']} fields) | "
                     f"{d_['share_within']:.2f} | {fmt(d_['loco_min'])} ({d_['loco_min_c']}) … {fmt(d_['loco_max'])} "
                     f"({d_['loco_max_c']}) | {c3(d_['no_eng'], True)} |")
    L += ["",
          "**CIP-2 family means** (full sample, BEA-region FE; fields with a defined partial r; broad selectivity "
          "partial from scripts/55 where available), sorted by mean strict share:\n",
          "| CIP-2 | family | k | mean strict share | mean broad share | mean government share | mean partial r (min … max) | mean broad selectivity partial | fields |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c_, r_ in FAM.iterrows():
        L.append(f"| {c_} | {FAMN.get(c_, c_)} | {r_.k} | {r_.lic:.2f} | {r_.licb:.2f} | {r_.gov:.2f} | {fmt(r_.pr)} "
                 f"({fmt(r_.prmin)} … {fmt(r_.prmax)}) | {fmt(r_.selp)} | {r_.fields} |")
    L += ["",
          "**Fields admitted only by the full sample** (fewer than 15 institutions or 3 states on 35c's sample), by "
          "strict licensure share. Partial r and raw incremental prestige R² under state FE / region FE / 1-df "
          "(— = undefined, full model has no residual df):\n",
          "| field | strict share | n | residual df: state / region / 1-df | partial r: state / region / 1-df | raw incr. prestige R²: state / region / 1-df |",
          "|---|---|---|---|---|---|"]
    for _, r_ in ADDED.iterrows():
        L.append(f"| {LAB.get(r_.field, r_.field)} | {r_.lic:.2f} | {r_.n} | "
                 + " / ".join(str(r_[f"dfr_{c}"]) if r_[f"dfr_{c}"] >= 0 else "—" for c in CODINGS) + " | "
                 + " / ".join(fmt(r_[f"pr_{c}"]) for c in CODINGS) + " | "
                 + " / ".join(fmt(r_[f"inc_{c}"], 3) for c in CODINGS) + " |")
    L += [""]

    # =============================== (c) ===============================
    L += ["## (c) Robustness: licensure measures, dropped fields and families, counter-cases\n",
          f"Licensure measures: the strict and broad all-ages shares (scripts/20) and the young-worker strict share, "
          f"i.e. the strict share among 22–27-year-old full-time BA+ workers (`data/interim/acs_occ_dist.parquet`, "
          f"scripts/35; scripts/20's SOC prefixes). The young-worker share is closer to the population behind "
          f"Scorecard's 4-year earnings and correlates {fmt(corr_young)} (Spearman) with the all-ages share. Pre-med "
          f"fields ({', '.join(LAB.get(f, f) for f in PREMED)}) get much of their all-ages share from graduate-degree "
          f"physicians (biology {fmt(float(a.set_index('field').licensure_strict.get('biology', np.nan)), 2, False)} "
          f"all ages vs {fmt(float(young.set_index('field').lic_young.get('biology', np.nan)), 2, False)} at 22–27). "
          f"Without engineering drops the {len(ENG_FIELDS)} CIP-14 fields. Cells: estimate [field CI; HC1 for the "
          f"b_licensure rows] ⟨CIP-2 cluster-bootstrap CI⟩ (k).\n",
          "| quantity | licensure measure | all | without nursing + comm. dis. | without pre-med fields | without both | without engineering (CIP-14) |",
          "|---|---|---|---|---|---|---|"]

    def c2cells(g):
        gi = g.set_index("sub")
        return " | ".join(f"{fmt(gi.loc[s_].est)} {fci(gi.loc[s_].lo, gi.loc[s_].hi)} {fcl(gi.loc[s_].clo, gi.loc[s_].chi)} "
                          f"(k={int(gi.loc[s_].k)})" for s_ in SUBSETS)
    for (q, reg), g in C2[C2.kind != "Spearman (partial r)"].groupby(["q", "reg"], sort=False):
        L.append(f"| {q} | {REGS[reg]} | " + c2cells(g) + " |")
    L += ["",
          "**Signed partial r without selected fields or families**: Spearman(licensure, partial r(F, earn | geo)) "
          "across fields, all six sample × coding versions:\n",
          "| quantity | licensure measure | all | without nursing + comm. dis. | without pre-med fields | without both | without engineering (CIP-14) |",
          "|---|---|---|---|---|---|---|"]
    for (q, reg), g in C2[C2.kind == "Spearman (partial r)"].groupby(["q", "reg"], sort=False):
        L.append(f"| {q} | {REGS[reg]} | " + c2cells(g) + " |")

    # ---- (c3) setting test
    L += ["",
          "### (c3) A measured setting proxy: government and nonprofit employment shares\n",
          "From the raw ACS PUMS 2023 person files (`data/raw/acs/psam_pus{a,b}.csv`, read in chunks), on exactly "
          f"scripts/20's anchor population (BA+, full-time, employed, earnings > 0; per-field row counts equal "
          f"scripts/20's `n_acs`: {sect_ok}). Class of worker (COW): government = 3–5 (local, state, federal), "
          "nonprofit = 2. PWGTP-weighted. The government share is the closest public measure of pay set by a public pay "
          "scale. It does not capture state-regulated reimbursement or employer concentration, so it is a partial "
          "proxy for \"setting\".\n",
          "| field | strict share | broad share | government | nonprofit | government + nonprofit | government, ages 22–27 |",
          "|---|---|---|---|---|---|---|"]
    for f_ in ["nursing", "communication_disorders", "pharmacy", "public_health", "special_education", "social_work",
               "accounting", "biology", "economics", "computer_science"]:
        if f_ not in sm_.index:
            continue
        s_ = sm_.loc[f_]
        L.append(f"| {LAB.get(f_, f_)} | {float(ach.loc[f_, 'licensure_strict']):.2f} | {float(ach.loc[f_, 'licensure_broad']):.2f} | "
                 f"{s_.gov:.2f} | {s_.npo:.2f} | {s_.govnp:.2f} | {fmt(s_.gov_young, 2, False)} |")
    L += ["",
          "Across fields, Spearman(outcome, share) [field] ⟨cluster⟩, and rank partial correlations of the outcome with "
          "the strict share given the government share and vice versa:\n",
          "| outcome | k | strict share | government share | government, 22–27 | government + nonprofit | ρ(strict, government) | strict \\| government | government \\| strict |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r_ in SET:
        q = r_["q"]
        if q.startswith("signed partial r"):
            samp_ = P35 if "35c" in q else PF
            cod_ = q.split(", ", 1)[1]
            t_s = PGi[(samp_, cod_)]["signed"]
        else:
            t_s = Dd.loc[(q.split(": ")[1], "licensure_strict", "all")].rho_t
        lgv, glv = r_["lic_given_gov"], r_["gov_given_lic"]
        L.append(f"| {q} | {r_['k']} | {c3(t_s)} | {c3(r_['gov'])} | {c3(r_['gov_young'])} | {c3(r_['govnp'])} | "
                 f"{fmt(r_['lic_gov'])} | {fmt(lgv[0])} {fci(lgv[1], lgv[2])} {fcl(lgv[4], lgv[5])} | "
                 f"{fmt(glv[0])} {fci(glv[1], glv[2])} {fcl(glv[4], glv[5])} |")
    L += ["",
          "Reading. The government share predicts the geography-netted coupling somewhat less consistently than the "
          "strict share; government + nonprofit predicts it at least as strongly, though that share partly overlaps "
          "the strict list (hospital and school employers). Net of each other, both shrink, and under the cluster bootstrap "
          "neither is reliably distinguishable from zero. "
          + ((lambda seps: (f"Under the cluster bootstrap the two separate (one partial CI excludes 0, the other "
                            f"does not) in {len(seps)} of {len(SET)} outcomes: "
                            + "; ".join(f"{q_} in favour of the {w_} share (strict | government "
                                        f"{fmt(r_['lic_given_gov'][0])} {fcl(*r_['lic_given_gov'][4:6])}, government | "
                                        f"strict {fmt(r_['gov_given_lic'][0])} {fcl(*r_['gov_given_lic'][4:6])})"
                                        for q_, w_, r_ in seps) + ". ") if seps else
             "Under the cluster bootstrap the two never separate. ")(
              [(r_["q"], "government" if pex(r_["gov_given_lic"], True) else "strict", r_) for r_ in SET
               if pex(r_["gov_given_lic"], True) != pex(r_["lic_given_gov"], True)]))
          + "The battery therefore cannot say whether the licence or the employer sector is what goes with low "
          "coupling. \"Pay set by setting\" stays an interpretation.\n",
          "**Counter-cases**:\n",
          "| field | strict share | broad share | government share | binary tag | baseline ρ_f | broad partial ρ_f (SE) | strict partial ρ_f (SE) | ρ(SAT, earn) | FE β_f (b) (SE) | partial r(F, earn \\| geo), full sample: state FE / region FE / 1-df | same, 35c's sample | incr. prestige raw / adj / CV (full, state FE) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def pr3(samp, f):
        v3 = [nf(samp, c, "raw", f) for c in CODINGS]
        if all(x is None for x in v3):
            return "—"
        return " / ".join(fmt(x.partial_r) if x is not None else "—" for x in v3)
    for f in ["accounting", "special_education", "social_work", "nursing", "public_health", "communication_disorders",
              "pharmacy", "computer_science"]:
        if f not in Sn.index:
            continue
        s_ = Sn.loc[f]
        inc = [fmt(nf(PF, ST, m, f).inc_prestige, 3) if nf(PF, ST, m, f) is not None else "—" for m in MEASURES]
        gv_ = f"{sm_.loc[f].gov:.2f}" if f in sm_.index else "—"
        L.append(f"| {LAB.get(f, f)} | {fmt(s_.licensure_strict, 2, False)} | {fmt(s_.licensure_broad, 2, False)} | {gv_} | "
                 f"{'yes' if F.is_licensed(f) else 'no'} | {fmt(s_.rho_raw)} | {fmt(s_.rho_full_stlev)} ({fmt(s_.se_rho_full_stlev, 2, False)}) | "
                 f"{fmt(s_.rho_full_state)} ({fmt(s_.se_rho_full_state, 2, False)}) | {fmt(s_.rho_sat_earn)} | "
                 f"{fmt(s_.fe_b_beta, 3)} ({fmt(s_.fe_b_se, 3, False)}) | "
                 f"{pr3(PF, f)} | {pr3(P35, f)} | {' / '.join(inc)} |")
    L += ["",
          f"- **Accounting** is tagged as licensed (CPA) and has a broad share of {acc_b:.2f}, but its strict share is "
          f"{acc_s:.3f}. Its coupling survives selectivity adjustment ("
          + (f"one of the {n_sig_broad} of {k_sig_broad} fields whose broad partial is more than 1.96 SE above zero in "
             "scripts/55" if acc_sig else "although its broad partial is not 1.96 SE above zero")
          + "), and prestige adds information beyond geography under every coding. Because its strict share is low, "
          "this is what the strict-share gradient predicts; accounting counters only the binary tag and the broad "
          "share. The broad share's accountant code, ACS SOC 13-2011, covers all accountants and auditors, not only "
          "CPAs, so accounting's licensure is itself loosely measured. Leaving accounting out moves b from "
          f"{fmt(Ai.loc['A2'].b, 3)} to {fmt(Ai.loc['A2'].b_no_accounting, 3)} (A2, broad share), from "
          f"{fmt(Ai.loc['A8'].b, 3)} to {fmt(Ai.loc['A8'].b_no_accounting, 3)} (A8, binary tag) and from "
          f"{fmt(Ai.loc['A1'].b, 3)} to {fmt(Ai.loc['A1'].b_no_accounting, 3)} (A1, strict share).\n",
          f"- **Special education** (strict share {float(ach.loc['special_education', 'licensure_strict']):.2f}, "
          f"government share {spd.gov:.2f}): K-12 teachers on salary schedules, yet baseline coupling "
          f"{fmt(sp.rho_raw)}, broad partial {fmt(sp.rho_full_stlev)} (SE {fmt(sp.se_rho_full_stlev, 2, False)}) and "
          f"within-institution β_f (b) {fmt(sp.fe_b_beta, 3)} (SE {fmt(sp.fe_b_se, 3, False)}), on "
          f"{n_of['special_education']} institutions. A counter-case to both the licence and the setting readings. "
          f"Its adjusted and CV state-FE increments are unstable (few residual df), so read it through the partial r "
          f"and the 1-df coding. It prices selectivity little (ρ(SAT, earn) {fmt(sp.rho_sat_earn)}).\n",
          f"- **Social work** (strict {float(ach.loc['social_work', 'licensure_strict']):.2f}, broad "
          f"{float(ach.loc['social_work', 'licensure_broad']):.2f}, government + nonprofit {swk.govnp:.2f}): partial r "
          f"| geography {fmt(min(x.partial_r for x in sw_pr))} to {fmt(max(x.partial_r for x in sw_pr))}, broad "
          f"partial {fmt(swS.rho_full_stlev)} (SE {fmt(swS.se_rho_full_stlev, 2, False)}). Moderate coupling in a mostly "
          "public/nonprofit field.\n",
          f"- **Communication disorders** has baseline coupling {fmt(cd_raw)} (rank {cd_rank} of {cd_k}, 1 = lowest) "
          f"but only {n_of['communication_disorders']} institutions; it "
          + ("cannot enter the SAT-sample selectivity specs.\n" if not np.isfinite(cd_sat) else
             f"has an SAT-sample partial of {fmt(cd_sat)}.\n")]

    # =============================== (d) ===============================
    L += ["## (d) Does the licensing gradient survive selectivity adjustment?\n",
          "Outcomes are the per-field estimates in `data/interim/selectivity_fields.csv` (scripts/55; joint "
          "institution-bootstrap SEs). Coupling is ρ_f = 1 − gap, so a negative slope here corresponds to a "
          "positive b_licensure on the gap. WLS weights are 1/SE², with percentile CIs from a field bootstrap "
          f"(B = {B_SLOPE}); these are skewed, because single high-weight fields (nursing above all) move the slope. "
          "Spearman CIs: field bootstrap (35c's, B = 5000) [ ] and CIP-2 cluster bootstrap ⟨ ⟩. Licensure is measured "
          "outside the earnings data (ACS), so, unlike Spearman(baseline, partial) in scripts/55, these statistics do "
          "not share estimation noise between predictor and outcome.\n",
          "| outcome | k | mean | WLS slope on strict share [CI] | Spearman with strict share [field] ⟨cluster⟩ | within CIP-2 (p) | strict, without engineering | Spearman with broad share [field] ⟨cluster⟩ | broad, within CIP-2 (p) | WLS slope on broad share [CI] | WLS slope (strict) without nursing + comm. dis. [CI] |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for col, sec, desc, fam in D_OUT:
        if (col, "licensure_strict", "all") not in Dd.index:
            continue
        r = Dd.loc[(col, "licensure_strict", "all")]
        rb = Dd.loc[(col, "licensure_broad", "all")] if (col, "licensure_broad", "all") in Dd.index else None
        rd = Dd.loc[(col, "licensure_strict", "drop nursing+comm.dis.")]
        L.append(f"| {desc} (`{col}`) | {int(r.k)} | {fmt(r['mean'])} | {fmt(r.b)} {fci(r.b_lo, r.b_hi)} | "
                 f"{c3(r.rho_t)} | {fmt(r.cl['r_within'])} ({r.cl['p_within']:.2f}) | {c3(r.cl['no_eng'], True)} | "
                 + (f"{c3(rb.rho_t)} | {fmt(rb.cl['r_within'])} ({rb.cl['p_within']:.2f}) | {fmt(rb.b)} {fci(rb.b_lo, rb.b_hi)}"
                    if rb is not None else "— | — | —")
                 + f" | {fmt(rd.b)} {fci(rd.b_lo, rd.b_hi)} |")
    L += ["",
          "**Matched raw → partial on the same fields** (paired field bootstrap of the WLS slopes on the strict share; "
          "Spearman [field] ⟨cluster⟩):\n",
          "| raw → partial | subset | k | Spearman raw | Spearman partial | WLS slope raw | WLS slope partial | partial − raw [95% CI] | level ratio (mean partial / mean raw) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in Dpairs.iterrows():
        L.append(f"| `{r.raw}` → `{r.partial}` | {r.subset} | {r.k} | {c3(r.rho_raw)} | {c3(r.rho_partial)} | "
                 f"{fmt(r.b_raw)} | {fmt(r.b_partial)} | {fmt(r['diff'])} {fci(r.diff_lo, r.diff_hi)} | "
                 f"{fmt(r.level_ratio, 2, False)} |")
    L += ["",
          "**Nursing after selectivity adjustment** (scripts/55 estimates ± 1.96 joint-bootstrap SE; z-test of nursing "
          "minus the field, treating the two estimates as independent, approximate because fields share "
          "institutions):\n",
          "| specification | field | estimate (SE) [95% CI] | strict share | nursing − field: z (p) |",
          "|---|---|---|---|---|"]
    for _, r_ in NSEL.iterrows():
        dd_ = 3 if r_.spec.startswith("fe") else 2
        L.append(f"| {r_.desc} (`{r_.spec}`) | {LAB.get(r_.field, r_.field)} | {fmt(r_.est, dd_)} ({r_.se:.{dd_}f}) "
                 f"{fci(r_.lo, r_.hi, dd_)} | {r_.lic:.2f} | "
                 + ("—" if r_.field == "nursing" else f"{fmt(r_.z)} ({pv(r_.p)})") + " |")
    L += ["",
          "Nursing's rank among all fields with an estimate (1 = lowest): "
          + "; ".join(f"{v_['desc']}: {v_['rank']} of {v_['k']} (field mean {fmt(v_['mean'], 3)}; {v_['n_inc0']} of "
                      f"{v_['k']} fields have a CI including 0, {v_['n_sig_pos']} are more than 1.96 SE above 0)"
                      for v_ in NSELrk.values()) + ".\n",
          f"Reading. Selectivity controls remove most of the average coupling (mean falls to "
          f"{rho_pairs.level_ratio.min() * 100:.0f}–{rho_pairs.level_ratio.max() * 100:.0f}% of raw on matched fields); "
          f"the strict-share relation keeps its sign, with paired slope changes in both directions and wide CIs. Under "
          f"the cluster bootstrap the strict-share Spearman excludes 0 for {n_part_exc} of {len(part_rows)} partial "
          f"couplings, and within CIP-2 families none of them shows a gradient. Among the department-prestige couplings "
          f"the broad share shows no gradient either, except the within-institution slope of spec (a) ({c3(fea_b.rho_t)}), which weakens once field-specific "
          f"selectivity slopes are added; selectivity pricing and brand coupling do fall with the broad share. "
          f"Selectivity pricing also falls with the strict share (ρ(SAT, earn) "
          f"{fmt(sat_.b)} {fci(sat_.b_lo, sat_.b_hi)}; ρ(−ADM, earn) {fmt(adm_.b)} {fci(adm_.b_lo, adm_.b_hi)}), so "
          f"the licensing gradient cannot tell prestige pricing from selectivity pricing. Mean β_f is "
          f"{fmt(fe_means['fe_a_beta'], 3)} (a) and {fmt(fe_means['fe_b_beta'], 3)} (b) log points per SD of prestige.\n"]

    # =============================== (e) ===============================
    L += ["## (e) UK nursing with adjusted and cross-validated R²\n",
          f"Same pipeline as scripts/41: UK ORCID SpringRank per CAH2 subject; LEO provider × subject median "
          f"earnings 5 years after graduation, tax year 2022/23; geography = the provider's England region or nation "
          f"(≤ 11 levels). Reproduction: max |Δ raw incremental prestige| vs `uk_licensing_decomp.csv` = {repro_e:.3f}. "
          f"`chance` = expected in-sample R² of the geography block under no association ≈ (levels − 1)/(n − 1).\n",
          "| CAH2 subject | licensed | n | levels | incr. prestige raw / adj / CV | rank (adj, 1 = lowest) | incr. geography raw / adj / CV | R²(geo) raw / adj / CV (chance) | partial r(prestige, earn \\| geo) [CI] |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in U.sort_values("adj_inc_p").iterrows():
        L.append(f"| {r.cah2} | {'yes' if r.licensed else ''} | {r.n} | {r.n_geo} | {fmt(r.raw_inc_p, 3)} / "
                 f"{fmt(r.adj_inc_p, 3)} / {fmt(r.cv_inc_p, 3)} | {r.rank_adj} | {fmt(r.raw_inc_g, 3)} / "
                 f"{fmt(r.adj_inc_g, 3)} / {fmt(r.cv_inc_g, 3)} | {fmt(r.raw_R2_geo, 2, False)} / "
                 f"{fmt(r.adj_R2_geo, 2, False)} / {fmt(r.cv_R2_geo, 2, False)} ({r.chance_R2_geo:.2f}) | "
                 f"{fmt(r.partial_r)} {fci(r.partial_lo, r.partial_hi)} |")
    L += ["",
          f"UK nursing: no prestige information beyond region is detected under any R² (raw "
          f"{fmt(nurseU.raw_inc_p, 3)}, adjusted {fmt(nurseU.adj_inc_p, 3)}, CV {fmt(nurseU.cv_inc_p, 3)}; partial r "
          f"{fmt(nurseU.partial_r)} {fci(nurseU.partial_lo, nurseU.partial_hi)}, so a partial r above "
          f"{fmt(nurseU.partial_hi)} is excluded at the 95% level but a small positive one is not). Its geography R² is "
          f"{fmt(nurseU.raw_R2_geo, 2, False)} raw against a chance level of {nurseU.chance_R2_geo:.2f}, "
          f"{fmt(nurseU.adj_R2_geo, 2, False)} adjusted and {fmt(nurseU.cv_R2_geo, 2, False)} cross-validated. No other "
          f"subject exceeds {fmt(ukcv_other, 2, False)} cross-validated. Unlicensed subjects reach the same near-zero "
          f"prestige increment, so with one licensed subject \"licensed ⇒ no prestige information\" cannot be "
          f"separated from \"some subjects have little prestige information\". Politics' CV increment above 1 is the "
          f"unbounded-CV case described in (b) "
          + (f"({int(pol.n)} providers, {int(pol.n_geo)} levels).\n" if pol is not None else ".\n")]

    # =============================== method + caveats ===============================
    L += ["## Method notes\n",
          "- Gap map: `compute_gap_map` on ALL_FIELDS, Scorecard FoS BA `EARN_MDN_4YR`, Wapman field rank "
          "(internal bootstrap seed 1234, B = 1000), as in scripts/13 and scripts/20. ACS licensure anchors: "
          "`data/interim/acs_occ_anchors.parquet` (scripts/20 `build_anchors`: ACS PUMS 2023 1-year, BA+, "
          "full-time, all ages).\n",
          "- REML meta-regression: `scripts/13.reml_metareg`, V = diag(SE²) + τ²·same-CIP-2.\n",
          "- LOO-CV is deterministic. Holding out an institution that is the only one from its state in the field "
          "forces the model to predict it without a state effect, which separates real geographic information from "
          "singleton dummies.\n",
          "- Partial r: residualise earnings and F on the geography design and correlate the residuals. Fisher-z CI "
          "with n − q − 3 df (q = geography columns).\n",
          f"- CIP-2 families: the first CIP-4 code of each field in `src/crosswalks/fields.ALL_FIELDS` "
          f"({max(t.g for t in sg)} families among the fields of the largest version; engineering = 14, "
          f"{len(ENG_FIELDS)} fields). Cluster bootstrap: draw as many "
          f"families as there are, with replacement, keep all their fields, recompute; B = {B_CL}, seed {SEED_CL} "
          f"(vectorised; checked draw by draw against a loop version); "
          f"draws with ≤ 2 distinct values skipped. For the WLS b_licensure the same resampling refits the WLS. "
          f"Within-CIP-2 association: Pearson correlation of the across-field ranks after demeaning both within "
          f"families (families with ≥ 2 fields), i.e. a rank regression with family fixed effects; p from permuting "
          f"licensure ranks within families (B = {B_PERM}). Rank partial correlations in (c3): residuals of the ranks "
          f"on the rank of the other share, with field and cluster bootstraps (B = {B_CL} each).\n",
          "- Every field-level Spearman CI uses 35c's field bootstrap (B = 5000, seed 7), so that the reproduced 35c "
          "headline and all other correlations share one procedure. This exceeds the project's usual cap of 1000, so "
          f"every one of the {mcinfo['n']} Spearman bootstraps was also run with B = {B_MC} (same seed): CI bounds "
          f"differ by at most {mcinfo['dmax']:.3f} (median {mcinfo['dmed']:.3f}), and whether a CI excludes 0 differs "
          f"in {mcinfo['flips']} of {mcinfo['n']} calls ({len(mcinfo['flip_list'])} distinct estimates, listed in the "
          f"CSV under section `check`; each has a bound within {np.ceil(flip_near * 1000) / 1000:.3f} of 0). "
          + ("None of them is among the field-bootstrap CIs the Answer quotes or counts. " if not flip_ans else
             f"{len(flip_ans)} of them are among the CIs the Answer quotes or counts: "
             + "; ".join(f"{fmt(t[0])} [{t[1]:+.3f}, {t[2]:+.3f}]" for t in flip_ans) + ". ")
          + f"The cluster bootstrap uses B = {B_CL}: in a development run with B = 1000 (not part of the final "
          f"outputs), a second seed moved cluster CI bounds by up to 0.12 and changed \"excludes 0\" for 38 of 646 CIs, "
          f"too noisy for the counts reported here. With B = {B_CL} it was rerun with a second seed ({SEED_CL + 1}): over {mclinfo['n']} CIs, bounds "
          f"differ by at most {mclinfo['dmax']:.3f} (median {mclinfo['dmed']:.3f}) and \"excludes 0\" changes in "
          f"{mclinfo['flips']}"
          + (f" (each with a reported bound within {mclinfo['flip_near']:.3f} of 0)" if mclinfo['flips'] else "")
          + ". "
          + ("None of those is a cluster CI that the Answer counts." if not cflip_ans else
             f"{len(cflip_ans)} of them are cluster CIs the Answer counts ("
             + "; ".join(f"{fmt(t[0])} ⟨{t[1]:+.3f}, {t[2]:+.3f}⟩" for t in cflip_ans)
             + "), so those counts can move by one with the Monte Carlo draw.")
          + " Other bootstraps: scripts/20's B = 2000 (rows A1–A2, reproduction only), B = 1000 elsewhere.\n",
          "- The field bootstrap for Spearman is a vectorised copy of scripts/35c `boot_spearman` (same seed, same "
          "draws). The script asserts that the two agree to 1e-12 on 35c's saved table.\n",
          "## Caveats\n",
          "- Descriptive. The strict licensure share marks a bundle: regulated occupations, public and nonprofit "
          "employers, local labour markets, and the Health / Biology disciplines. This battery cannot separate its "
          "parts (point 9, (c3)).\n",
          "- The cross-field gradient is identified from differences between CIP-2 families. The CIP-2 cluster "
          "bootstrap treats the families as the independent units, which is closer to the design than the field "
          "bootstrap but rests on few units. Fields also share institutions across families, which neither bootstrap "
          "models.\n",
          "- The all-ages ACS licensure share counts graduate-degree holders. In pre-med fields much of it comes from "
          "physicians, not from the early-career BA population that Scorecard measures. The young-worker share and "
          "the no-pre-med rows in (c) address this.\n",
          "- Earnings are 4-year medians for Title-IV completers (UK: LEO 5-year medians). Neither sees the tail.\n",
          "- Geography in (b) is where the institution is, not where graduates work. PSEO flows put 34% of graduates "
          "out of state (README Result 3), so the geography control is imperfect.\n",
          "- A CI that includes 0 is a failure to detect, not evidence of absence. For nursing net of geography only, "
          f"the upper CI bounds (up to {fmt(NPR[NPR['sample'] == P35].hi.max())} on 35c's sample) leave room for "
          f"moderate associations; net of selectivity the upper bound is {fmt(nb.hi)}.\n",
          "- The two samples in (b) differ in the institutions (the PSEO list is public-skewed and misses most elite "
          "privates) and in the fields admitted.\n",
          f"- Only nursing is a large, clean licensed field. Communication disorders ({n_of['communication_disorders']} "
          f"institutions) and special education ({n_of['special_education']}) enter only on the full sample, and "
          "pharmacy never has enough institutions.\n",
          "## Provenance (for SOURCES.md)\n",
          "No new data were downloaded, so nothing needs to be added to `data/raw/SOURCES.md`. All inputs are public "
          "and already documented there; the URL, release and access-date column below is copied from SOURCES.md. "
          "This revision additionally reads the raw ACS PUMS 2023 person files (COW column), which SOURCES.md §9 "
          "already covers as the extracted contents of `csv_pus.zip`. Size and md5 are computed by the script when it "
          "runs:\n",
          "| file | role | source URL, release (access date), per SOURCES.md | size (bytes) | md5 |",
          "|---|---|---|---|---|"]
    FOS_URL = ("`https://ed-public-download.scorecard.network/downloads/Most-Recent-Cohorts-Field-of-Study_06102026.zip`, "
               "\"Most Recent\" release updated 2026-06-10 (accessed 2026-06-20); SOURCES.md §2")
    ACS_URL = ("`https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_pus.zip`, ACS PUMS 2023 "
               "1-year (accessed 2026-06-20); SOURCES.md §9")
    for p, role, src in [
            (INST_FILE, "Scorecard institution file (STABBR, REGION, ST_EARN inputs)",
             "`https://collegescorecard.ed.gov/data/` (Most-Recent-Cohorts-Institution zip), \"Most Recent\" release, "
             "file build 2026-05-27 (accessed 2026-09-23); SOURCES.md §9"),
            (ROOT / "data/raw/scorecard_fos/Most-Recent-Cohorts-Field-of-Study.csv", "Scorecard FoS BA earnings", FOS_URL),
            (ROOT / "data/raw/wapman2022/ranks.csv", "Wapman et al. 2022 field / academia ranks",
             "`https://zenodo.org/api/records/6941651/files/us-faculty-hiring-networks.zip/content`, Zenodo "
             "10.5281/zenodo.6941651 v1, 2022-07-29 (accessed 2026-06-20); SOURCES.md §1"),
            (PSEO_INST, "PSEO institution list (35c's state source)",
             "`https://lehd.ces.census.gov/data/pseo_experimental.html`, PSEO V4.13.0 / 2025Q4; SOURCES.md §7"),
            (INTERIM / "acs_occ_anchors.parquet", "ACS licensure anchors (derived, scripts/20)", "from " + ACS_URL),
            *[(p_, "ACS PUMS 2023 person file (COW class of worker for (c3))", "extracted from " + ACS_URL) for p_ in ACS_RAW],
            (INTERIM / "acs_occ_dist.parquet", "ACS ages 22–27 occupations (derived, scripts/35)", "same ACS PUMS 2023 file"),
            (INTERIM / "acs_workers_expanded.parquet", "ACS workers for the dispersion covariate (derived, src/dispersion)",
             "same ACS PUMS 2023 file"),
            (INTERIM / "valuation_residuals.csv", "inst × field prestige F + earnings (derived, scripts/30)",
             "Wapman ranks + Scorecard FoS above"),
            (INTERIM / "selectivity_fields.csv", "per-field selectivity-adjusted couplings (derived, scripts/55)",
             "Wapman ranks + Scorecard FoS + Scorecard institution file above"),
            (ROOT / "data/raw/leo/leo_dashboard.zip", "UK LEO provider × subject earnings",
             "`https://content.explore-education-statistics.service.gov.uk/api/releases/a13c6267-1527-4761-bf8e-3566d8d26629/files/3896dae9-11af-46ef-b6c6-2eed9acda9e5`, "
             "DfE *Graduate outcomes (LEO) provider level data*, 2022-23 release (accessed 2026-06-22); SOURCES.md §9"),
            (INTERIM / "leo_provider_subject_geo.parquet", "LEO provider × subject with location (derived, scripts/41)",
             "LEO file above"),
            (INTERIM / "orcid_uk_phd_faculty_edges.parquet", "UK ORCID PhD→faculty edges (derived, scripts/40)",
             "ORCID-Derived Academic Mobility Edges, Zenodo 10.5281/zenodo.19651302 version 20260419, "
             "`https://zenodo.org/api/records/19651302/files/20260419.7z/content` (accessed 2026-06-20; SOURCES.md §8) "
             "+ ROR data dump v2.8, 2026-06-02 (accessed 2026-06-22; SOURCES.md §9)")]:
        if p.exists():
            L.append(f"| `{p.relative_to(ROOT)}` | {role} | {src} | {p.stat().st_size} | `{md5(p)}` |")
    L.append("")
    OUT_MD.write_text("\n".join(L))


if __name__ == "__main__":
    main()
