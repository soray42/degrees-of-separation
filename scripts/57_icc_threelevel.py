"""Three-level random-effects refit of the "gap is structured by discipline" (CIP-2 ICC) claim.

Referee objection (methodologist, rated fatal): the headline meta-model gap_f ~ 1 + (1|CIP-2)
(scripts/14, scripts/24; V = diag(SE^2) + tau^2 * same-CIP2) has NO field-level true-heterogeneity
term, so real field-to-field spread is either loaded onto the cluster term or left in a
moment-based "within" variance; the reported ICC = tau2_model / (tau2_model + mean within-cluster
sample variance) mixes a REML variance with a moment variance; the "measurement-corrected" ICC
subtracts mean SE^2 from that moment variance (not a coherent decomposition); 8 of 22 clusters
are singletons; and a size-preserving label-permutation spot check put the observed raw ICC (0.32)
barely above the null 95th percentile (0.31).

This script refits it properly:
  (a) outcome z_f = atanh(rho_f), rho_f = Spearman(prestige, earnings) from
      outputs/expanded66_gap_map.csv. Known sampling variance s_f^2 = 1/(n_f - 3) [main].
      Checks: Bonett-Wright (1 + rho^2/2)/(n_f - 3); the file's bootstrap percentile CI mapped to
      z, ((z_hi - z_lo)/3.92)^2; and two CORRELATED-ERROR specs from a joint institution
      bootstrap (fields in one CIP-2 cluster are measured on largely the same institutions, so
      their sampling errors co-move): 1/(n-3) variances with the joint-bootstrap correlation
      matrix, and the full joint-bootstrap covariance matrix. The per-institution data are rebuilt
      with src.load_ar.load_ar_wapman / src.load_er.load_er_scorecard exactly as src.gap does, and
      every field's n and rho are asserted to reproduce the file.
  (b) three-level random-effects meta-analysis
          z_f = mu + u_c(f) + v_f + e_f,  u_c ~ N(0, tau2_cluster), v_f ~ N(0, tau2_field),
          e ~ N(0, V_e) known (diagonal or full),
      REML written out here (closed-form Woodbury evaluation for diagonal V_e, Cholesky for full
      V_e, both checked against a textbook dense implementation), maximized by grid + L-BFGS-B +
      explicit boundary fits; profile-likelihood 95% CIs for tau2_cluster, tau2_field and
      ICC_true = tau2_cluster / (tau2_cluster + tau2_field).
  (c) size-preserving label-permutation null (seeded) for ICC_true and for the REML
      likelihood-ratio statistic of the cluster component (LR_c = 2[l(full) - l(tau2_c = 0)]);
      plus a parametric-bootstrap null from the fitted field-only model, a power simulation on
      the same design, and a leave-one-cluster-out check.
  (d) Cochran Q / I^2 for total heterogeneity and the multilevel I^2 split (Cheung 2014).
  (e) run on the 57 non-degenerate fields (scripts/14 degeneracy filter) and on the 20-field
      reliable subset; report cluster counts and singletons.
  (f) [fix round] field-size moderator arm. Coupling rises with n (institutions per field) and CIP-2
      clusters differ in n, so the label permutation in (c) credits a size gradient to discipline.
      Refit z = X b + u_CIP2 + v_field + e with X = [1, centred log n] (checks: (log n)^2, coverage
      n / n_Wapman, prestige range, log earnings-cohort size, pairs), REML with |X'V^-1X|; residual
      tau2's, profile CIs, residual ICC_true; LR_c tested by mixture, label permutation with X kept
      attached to fields (conditional on n), parametric bootstrap from the n-adjusted field-only model,
      and label permutation within rank tertiles of n; leave-one-cluster-out and leave-one-field-out
      under the adjustment; within-cluster vs overall log-n slope.
Secondary: mixed-effects moderator model z ~ C(CIP-2) + field RE (Wald Q_M, pseudo-R^2), the
two-level cluster-only model on z (the old structure), and the legacy scripts/14/24 ICC (raw and
"measurement-corrected") with its permutation null (saving the referee's spot check).

Reproducible: SEED fixed, single-threaded BLAS, no live data pulls; permutation / bootstrap fits run
in ICC57_NPROC (default 6) forked workers, with every random draw made in the parent, so outputs
are byte-identical for any worker count.
Run:     PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/57_icc_threelevel.py
Outputs: data/interim/icc_threelevel.csv, outputs/figures/icc_threelevel.png,
         ICC_THREELEVEL_RESULT.md (repo root, local/gitignored).
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
sys.dont_write_bytecode = True
import importlib.util
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar, brentq
from scipy.stats import chi2, norm, rankdata, spearmanr, pearsonr, f as fdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_ar import load_ar_wapman
from src.load_er import load_er_scorecard

SEED = 57
N_PERM = 5000            # label permutations, main spec
N_PERM_CHECK = 2000      # label permutations, every other spec and the leave-one-cluster-out runs
N_PBOOT = 2000           # parametric-bootstrap draws (diagonal specs)
N_PBOOT_DENSE = 1000     # parametric-bootstrap draws (correlated-error specs)
N_POWER = 1000           # draws per power scenario
B_JOINT = 2000           # joint institution-bootstrap draws for the error covariance
POWER_ICC = (0.25, 0.50, 0.75)
UB = 2.0                 # upper bound for any tau2 on the z scale (z range here is about 1.4)
GRID = np.linspace(0.0, np.sqrt(0.5), 41) ** 2        # start grid, diagonal specs
GRID_DENSE = np.linspace(0.0, np.sqrt(0.5), 13) ** 2  # start grid, correlated-error specs
CRIT = float(chi2.ppf(0.95, 1))                        # 3.8415, profile-likelihood cut
Z975 = float(norm.ppf(0.975))
LOG2PI = float(np.log(2 * np.pi))
TIE = 1e-9               # tolerance for ">= observed" in permutation p-values
NPROC = int(os.environ.get("ICC57_NPROC", "6"))   # worker processes; results do not depend on it
N_PERM_ADJ = 2000        # label permutations per field-size-adjusted fit (primary 57-field run: N_PERM)
N_STRAT = 3              # strata (rank tertiles of n) for the n-stratified label permutation

# field-size / measurement moderator sets (covariates centred within each sample)
ADJ_SETS = {
    "log_n": (("log_n",), "log n"),
    "log_n_quad": (("log_n", "log_n_sq"), "log n + (log n)^2"),
    "cov_frac": (("cov_frac",), "coverage"),
    "log_n_cov_frac": (("log_n", "cov_frac"), "log n + coverage"),
    "pct_sd": (("pct_sd",), "prestige range"),
    "log_n_pct_sd": (("log_n", "pct_sd"), "log n + prestige range"),
    "log_cohort": (("log_cohort",), "log cohort size"),
    "log_n_log_cohort": (("log_n", "log_cohort"), "log n + log cohort size"),
}
PRIMARY = "log_n"

GAP_MAP = ROOT / "outputs" / "expanded66_gap_map.csv"
OUT_CSV = ROOT / "data" / "interim" / "icc_threelevel.csv"
OUT_FIG = ROOT / "outputs" / "figures" / "icc_threelevel.png"
OUT_MD = ROOT / "ICC_THREELEVEL_RESULT.md"

# FIELDS66 + label map exactly as scripts/28 (and 52) build them; C2NAME / legacy REML from 24
_spec = importlib.util.spec_from_file_location("s28", ROOT / "scripts" / "28_field_vs_generic_prestige.py")
_s28 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s28)
FIELDS66, LAB, CIP2, C2NAME = _s28.FIELDS66, _s28.LAB, _s28.CIP2, _s28.C2NAME
legacy_reml_tau2 = _s28._cw24.reml_tau2      # the function behind the published 0.30/0.32 ICC

VAR_SPECS = {
    "fisher": "1/(n-3), independent [main]",
    "bonett_wright": "(1+rho^2/2)/(n-3), independent",
    "bootstrap": "file bootstrap CI on z, ((z_hi-z_lo)/3.92)^2, independent",
    "fisher_corr": "1/(n-3) variances x joint-institution-bootstrap correlation",
    "jointboot_cov": "full joint-institution-bootstrap covariance",
}
DENSE_SPECS = {"fisher_corr", "jointboot_cov"}
MAIN, CORR = "fisher", "fisher_corr"


# ============================================================ data ======================
def load_fields():
    gm = pd.read_csv(GAP_MAP, dtype={"cip2": str})
    assert (gm.cip2 == gm.field.map(CIP2)).all(), "cip2 column disagrees with FIELDS66"
    ok = gm.gap.notna()
    assert np.allclose(gm.gap[ok], 1 - gm.spearman[ok]), "gap != 1 - spearman"
    degen = ((gm.gap <= 0.02) | (gm.gap >= 1.6) | (gm.n_institutions < 8)
             | ((gm.ci_hi - gm.ci_lo) < 0.02))          # scripts/14 + 24 filter, verbatim
    d = gm[ok & ~degen].copy().sort_values("field").reset_index(drop=True)
    d["n"] = d.n_institutions.astype(int)
    d["rho"] = d.spearman.astype(float)
    d["z"] = np.arctanh(d.rho)
    d["v_fisher"] = 1.0 / (d.n - 3)
    d["v_bonett_wright"] = (1.0 + d.rho ** 2 / 2) / (d.n - 3)
    rlo = np.clip(1 - d.ci_hi, -0.999, 0.999)           # rho CI = 1 - gap CI (flipped)
    rhi = np.clip(1 - d.ci_lo, -0.999, 0.999)
    d["v_bootstrap"] = ((np.arctanh(rhi) - np.arctanh(rlo)) / (2 * Z975)) ** 2
    d["cname"] = d.cip2.map(lambda c: C2NAME.get(c, c))
    d["label"] = d.field.map(LAB)
    excluded = gm.loc[ok & degen, ["field", "n_institutions", "gap"]]
    meta = dict(n_total=len(gm), n_gap=int(ok.sum()), excluded=excluded)
    return d, meta


def rebuild_fields(d):
    """Rebuild each field's matched (prestige, earnings) table exactly as src.gap.compute_gap does
    and assert n and rho reproduce the file. Also derive per-field MEASUREMENT covariates for the
    field-size moderator arm (all from the same matched tables):
      log_n       log(n_f), n_f = institutions with both prestige and earnings
      cov_frac    n_f / n_wapman_f, n_wapman_f = institutions in the field's Wapman prestige list
      pct_sd      SD of the matched institutions' prestige percentile within the field's full Wapman
                  list (prestige range actually covered; small = range-restricted)
      log_cohort  log of the median Scorecard earnings-cohort size (cohort_n) over matched institutions
    """
    ar = load_ar_wapman(fields=FIELDS66); er = load_er_scorecard(fields=FIELDS66)
    P, E, keys, cov = [], [], [], []
    for f, n, rho in zip(d.field, d.n, d.rho):
        a = ar[ar["field"] == f][["inst_key", "prestige_score"]]
        e = er[(er["field"] == f) & (er["level"] == "undergrad")][["inst_key", "earnings", "cohort_n"]]
        m = a.merge(e, on="inst_key", how="inner").dropna(subset=["prestige_score", "earnings"])
        r = np.corrcoef(rankdata(m.prestige_score), rankdata(m.earnings))[0, 1]
        assert len(m) == n and np.isclose(r, rho), f"rebuild mismatch for {f}"
        P.append(m.prestige_score.to_numpy(float)); E.append(m.earnings.to_numpy(float))
        keys.append(m.inst_key.to_numpy(str))
        pa = a.dropna(subset=["prestige_score"]).copy()
        assert not pa.inst_key.duplicated().any(), f"duplicate Wapman institutions in {f}"
        pa["pct"] = rankdata(pa.prestige_score) / len(pa)
        pct = m[["inst_key"]].merge(pa[["inst_key", "pct"]], on="inst_key").pct.to_numpy(float)
        assert len(pct) == n
        cov.append(dict(field=f, n_wapman=len(pa), cov_frac=n / len(pa), pct_sd=float(np.std(pct, ddof=1)),
                        med_cohort=float(m.cohort_n.median()),
                        n_cohort_missing=int(m.cohort_n.isna().sum())))
    cov = pd.DataFrame(cov)
    assert (cov.med_cohort > 0).all()
    cov["log_n"] = np.log(d.n.to_numpy(float)); cov["log_cohort"] = np.log(cov.med_cohort)
    return dict(P=P, E=E, keys=keys), cov


def joint_bootstrap(d, rng, tables):
    """Joint institution bootstrap on the rebuilt matched tables: per draw resample the union of
    institutions with replacement ONCE and recompute every field's Spearman on its own resampled
    institutions. Returns the correlation and covariance of the z replicates."""
    P, E, keys = tables["P"], tables["E"], tables["keys"]
    U = sorted(set().union(*(set(k) for k in keys))); pos = {u: i for i, u in enumerate(U)}
    idx = [np.array([pos[u] for u in k]) for k in keys]
    k = len(d); Z = np.full((B_JOINT, k), np.nan)
    for b in range(B_JOINT):
        cnt = np.bincount(rng.integers(0, len(U), len(U)), minlength=len(U))
        for j in range(k):
            w = cnt[idx[j]]; x = np.repeat(P[j], w); y = np.repeat(E[j], w)
            if len(np.unique(x)) >= 3 and len(np.unique(y)) >= 3:
                Z[b, j] = np.corrcoef(rankdata(x), rankdata(y))[0, 1]
    ok = ~np.isnan(Z).any(1)
    Zz = np.arctanh(np.clip(Z[ok], -0.999, 0.999))
    return dict(R=np.corrcoef(Zz.T), V=np.cov(Zz.T, ddof=1), n_ok=int(ok.sum()), n_univ=len(U),
                pos={f: i for i, f in enumerate(d.field)})


def spec_inputs(sd, spec, JB):
    if spec not in DENSE_SPECS:
        return dict(s2=sd[f"v_{spec}"].to_numpy())
    ii = [JB["pos"][f] for f in sd.field]
    if spec == "fisher_corr":
        s = np.sqrt(sd.v_fisher.to_numpy())
        return dict(Ve=JB["R"][np.ix_(ii, ii)] * np.outer(s, s))
    return dict(Ve=JB["V"][np.ix_(ii, ii)])


def onehot(codes, C):
    G = np.zeros((len(codes), C)); G[np.arange(len(codes)), codes] = 1.0
    return G


# ---- deterministic parallel map: every random draw is made in the parent BEFORE the map, the
# workers only fit, and results come back in input order, so output is identical for any NPROC ----
_W = {}


def _pworker(i):
    return _W["fn"](_W["items"][i])


def pmap(fn, items):
    items = list(items)
    if NPROC <= 1 or len(items) < 40:
        return [fn(x) for x in items]
    _W["fn"], _W["items"] = fn, items
    try:
        with mp.get_context("fork").Pool(NPROC) as pool:
            out = pool.map(_pworker, range(len(items)), chunksize=max(1, len(items) // (NPROC * 8)))
    finally:
        _W.clear()
    return out


# ============================================================ model =====================
class ThreeLevel:
    """REML for z = mu + u_cluster + v_field + e, e ~ N(0, V_e) with V_e known.

    Diagonal V_e = diag(s2): closed form via Woodbury. With d_i = s2_i + tf, w_i = 1/d_i, cluster
    sums W_c = sum w_i, S_c = sum w_i z_i and a_c = tc / (1 + tc W_c):
      log|V| = sum log d_i + sum_c log(1 + tc W_c),  1'V^-1 1 = sum w - sum_c a_c W_c^2, etc.
    Full V_e: Cholesky of V = V_e + tf I + tc * same-cluster.
    """

    def __init__(self, y, s2=None, Ve=None, X=None):
        self.y = np.asarray(y, float); self.k = len(self.y)
        self.dense = Ve is not None
        if self.dense:
            self.Ve = np.asarray(Ve, float); self.s2 = np.diag(self.Ve).copy()
            self.I = np.eye(self.k); self.one = np.ones(self.k)
        else:
            self.s2 = np.asarray(s2, float)
        # fixed-effect design: None = intercept only (the exact code path of the unadjusted model);
        # otherwise an (k, p) matrix whose first column is the intercept (moderator arm)
        self.X = None if X is None else np.asarray(X, float)
        self.p = 1 if X is None else self.X.shape[1]
        if self.X is not None:
            assert np.allclose(self.X[:, 0], 1.0) and np.linalg.matrix_rank(self.X) == self.p

    # ---- log-likelihood (exact REML, incl. the -(k-p)/2 log 2pi constant) ----
    def loglik(self, tc, tf, g, C):
        if self.dense:
            return self._ll_dense(tc, tf, g)
        if self.X is not None:
            return self._ll_diag_X(tc, tf, g, C)
        y = self.y; d = self.s2 + tf; w = 1.0 / d
        W = np.bincount(g, w, C); S = np.bincount(g, w * y, C)
        a = tc / (1.0 + tc * W)
        ldV = np.log(d).sum() + np.log1p(tc * W).sum()
        s11 = w.sum() - (a * W * W).sum()
        s1y = (w * y).sum() - (a * W * S).sum()
        syy = (w * y * y).sum() - (a * S * S).sum()
        return -0.5 * (ldV + np.log(s11) + syy - s1y * s1y / s11 + (self.k - 1) * LOG2PI)

    def _diag_X_parts(self, tc, tf, g, C):
        """Woodbury pieces with covariates: X'V^-1X, X'V^-1y, y'V^-1y, log|V| (diagonal V_e)."""
        X, y = self.X, self.y; d = self.s2 + tf; w = 1.0 / d
        W = np.bincount(g, w, C); a = tc / (1.0 + tc * W)
        Sx = np.column_stack([np.bincount(g, w * X[:, j], C) for j in range(self.p)])
        Sy = np.bincount(g, w * y, C)
        XVX = (X * w[:, None]).T @ X - (Sx * a[:, None]).T @ Sx
        XVy = (X * w[:, None]).T @ y - (Sx * a[:, None]).T @ Sy
        yVy = (w * y * y).sum() - (a * Sy * Sy).sum()
        ldV = np.log(d).sum() + np.log1p(tc * W).sum()
        return XVX, XVy, yVy, ldV

    def _ll_diag_X(self, tc, tf, g, C):
        XVX, XVy, yVy, ldV = self._diag_X_parts(tc, tf, g, C)
        ldX = np.linalg.slogdet(XVX)[1]
        rss = yVy - XVy @ np.linalg.solve(XVX, XVy)
        return -0.5 * (ldV + ldX + rss + (self.k - self.p) * LOG2PI)

    def _V(self, tc, tf, g):
        return self.Ve + tf * self.I + tc * (g[:, None] == g[None, :])

    def _ll_dense(self, tc, tf, g):
        cf = cho_factor(self._V(tc, tf, g), lower=True, check_finite=False)
        ld = 2.0 * np.log(np.diag(cf[0])).sum()
        if self.X is not None:
            X = self.X; A = cho_solve(cf, np.column_stack([X, self.y]), check_finite=False)
            XVX = X.T @ A[:, :self.p]; XVy = X.T @ A[:, self.p]; yVy = self.y @ A[:, self.p]
            rss = yVy - XVy @ np.linalg.solve(XVX, XVy)
            return -0.5 * (ld + np.linalg.slogdet(XVX)[1] + rss + (self.k - self.p) * LOG2PI)
        a = cho_solve(cf, np.column_stack([self.one, self.y]), check_finite=False)
        s11 = a[:, 0].sum(); s1y = a[:, 1].sum(); syy = self.y @ a[:, 1]
        return -0.5 * (ld + np.log(s11) + syy - s1y * s1y / s11 + (self.k - 1) * LOG2PI)

    def beta(self, tc, tf, g, C):
        """GLS fixed effects and their covariance (X'V^-1X)^-1 at given variance components."""
        X = self.X if self.X is not None else np.ones((self.k, 1))
        if self.dense:
            cf = cho_factor(self._V(tc, tf, g), lower=True, check_finite=False)
            A = cho_solve(cf, np.column_stack([X, self.y]), check_finite=False)
            XVX = X.T @ A[:, :X.shape[1]]; XVy = X.T @ A[:, X.shape[1]]
        elif self.X is not None:
            XVX, XVy, _, _ = self._diag_X_parts(tc, tf, g, C)
        else:
            w = 1.0 / (self.s2 + tf); W = np.bincount(g, w, C); S = np.bincount(g, w * self.y, C)
            a = tc / (1.0 + tc * W)
            XVX = np.array([[w.sum() - (a * W * W).sum()]]); XVy = np.array([(w * self.y).sum() - (a * W * S).sum()])
        cov = np.linalg.inv(XVX)
        return cov @ XVy, cov

    def loglik_grid(self, TC, TF, g, C):
        """loglik over parameter arrays TC, TF (same shape (m,)); vectorised for diagonal V_e."""
        if self.dense:
            return np.array([self._ll_dense(a, b, g) for a, b in zip(TC, TF)])
        if self.X is not None:
            return self._ll_grid_X(TC, TF, g, C)
        G = onehot(g, C); y = self.y
        d = self.s2[None, :] + TF[:, None]; w = 1.0 / d
        W = w @ G; S = (w * y[None, :]) @ G
        a = TC[:, None] / (1.0 + TC[:, None] * W)
        ldV = np.log(d).sum(1) + np.log1p(TC[:, None] * W).sum(1)
        s11 = w.sum(1) - (a * W * W).sum(1)
        s1y = (w * y).sum(1) - (a * W * S).sum(1)
        syy = (w * y * y).sum(1) - (a * S * S).sum(1)
        return -0.5 * (ldV + np.log(s11) + syy - s1y * s1y / s11 + (self.k - 1) * LOG2PI)

    def _ll_grid_X(self, TC, TF, g, C):
        G = onehot(g, C); y = self.y; X = self.X; P = self.p
        d = self.s2[None, :] + TF[:, None]; w = 1.0 / d
        W = w @ G; a = TC[:, None] / (1.0 + TC[:, None] * W)
        Sx = [w @ (G * X[:, [j]]) for j in range(P)]; Sy = w @ (G * y[:, None])
        XVX = np.empty((len(TC), P, P)); XVy = np.empty((len(TC), P))
        for i in range(P):
            XVy[:, i] = w @ (X[:, i] * y) - (a * Sx[i] * Sy).sum(1)
            for j in range(i, P):
                XVX[:, i, j] = XVX[:, j, i] = w @ (X[:, i] * X[:, j]) - (a * Sx[i] * Sx[j]).sum(1)
        yVy = w @ (y * y) - (a * Sy * Sy).sum(1)
        ldV = np.log(d).sum(1) + np.log1p(TC[:, None] * W).sum(1)
        ldX = np.linalg.slogdet(XVX)[1]
        b = np.linalg.solve(XVX, XVy[:, :, None])[:, :, 0]
        rss = yVy - (XVy * b).sum(1)
        return -0.5 * (ldV + ldX + rss + (self.k - self.p) * LOG2PI)

    def loglik_textbook(self, tc, tf, g):
        """textbook dense REML via explicit inverse (verification only)."""
        Ve = self.Ve if self.dense else np.diag(self.s2)
        Z = onehot(g, g.max() + 1); V = Ve + tf * np.eye(self.k) + tc * Z @ Z.T
        Vi = np.linalg.inv(V); X = self.X if self.X is not None else np.ones((self.k, 1))
        XVX = X.T @ Vi @ X; b = np.linalg.solve(XVX, X.T @ Vi @ self.y); r = self.y - X @ b
        return -0.5 * (np.linalg.slogdet(V)[1] + np.linalg.slogdet(XVX)[1] + r @ Vi @ r
                       + (self.k - X.shape[1]) * LOG2PI)

    # ---- 1-D maximiser on [0, UB]: sqrt grid then bounded Brent in the bracketing cell ----
    @staticmethod
    def max1d(f, fvec=None, ub=UB):
        grid = np.linspace(0.0, np.sqrt(ub), 61) ** 2
        vals = fvec(grid) if fvec is not None else np.array([f(t) for t in grid])
        i = int(np.argmax(vals))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        best_x, best_f = float(grid[i]), float(vals[i])
        r = minimize_scalar(lambda t: -f(t), bounds=(lo, hi), method="bounded",
                            options=dict(xatol=1e-12))
        if -r.fun > best_f:
            best_x, best_f = float(r.x), float(-r.fun)
        return best_x, best_f

    def fit_null(self):
        """field-only two-level model (tau2_cluster = 0); label-independent."""
        g0 = np.zeros(self.k, int); z = lambda t: np.zeros_like(t)
        fvec = None if self.dense else (lambda t: self.loglik_grid(z(t), t, g0, 1))
        tf, ll = self.max1d(lambda t: self.loglik(0.0, t, g0, 1), fvec)
        return dict(tc=0.0, tf=tf, ll=ll)

    def fit_conly(self, g, C):
        """cluster-only two-level model (tau2_field = 0) = the old scripts/14 structure."""
        fvec = None if self.dense else (lambda t: self.loglik_grid(t, np.zeros_like(t), g, C))
        tc, ll = self.max1d(lambda t: self.loglik(t, 0.0, g, C), fvec)
        return dict(tc=tc, tf=0.0, ll=ll)

    def fit(self, g, C, null, conly=None):
        """full three-level REML. Candidates: best grid point refined by L-BFGS-B, and the two
        boundary fits (tc=0 -> null, tf=0 -> cluster-only). Returns the highest REML."""
        conly = conly if conly is not None else self.fit_conly(g, C)
        grid = GRID_DENSE if self.dense else GRID
        TC, TF = np.meshgrid(grid, grid, indexing="ij")
        vals = self.loglik_grid(TC.ravel(), TF.ravel(), g, C)
        j = int(np.argmax(vals))
        x0 = np.array([TC.ravel()[j], TF.ravel()[j]])
        r = minimize(lambda t: -self.loglik(t[0], t[1], g, C), x0, method="L-BFGS-B",
                     bounds=[(0.0, UB), (0.0, UB)], options=dict(ftol=1e-14, gtol=1e-10))
        cands = [(float(-r.fun), float(r.x[0]), float(r.x[1])),
                 (null["ll"], 0.0, null["tf"]), (conly["ll"], conly["tc"], 0.0),
                 (float(vals[j]), float(x0[0]), float(x0[1]))]
        ll, tc, tf = max(cands, key=lambda c: c[0])
        return dict(tc=tc, tf=tf, ll=ll,
                    icc=(tc / (tc + tf)) if (tc + tf) > 1e-12 else np.nan,
                    lr=max(0.0, 2 * (ll - null["ll"])), conly=conly)

    def gls(self, tc, tf, g, C):
        """GLS mean, its SE, cluster BLUPs u_c and their conditional variances (mu treated known)."""
        y = self.y
        if self.dense:
            X = onehot(g, C); cf = cho_factor(self._V(tc, tf, g), lower=True)
            Vi1 = cho_solve(cf, self.one); s11 = Vi1.sum(); mu = (Vi1 @ y) / s11
            Vir = cho_solve(cf, y - mu); ViX = cho_solve(cf, X)
            u = tc * (X.T @ Vir); u_var = tc - tc ** 2 * np.diag(X.T @ ViX)
            return mu, float(np.sqrt(1 / s11)), u, u_var
        w = 1.0 / (self.s2 + tf)
        W = np.bincount(g, w, C); S = np.bincount(g, w * y, C); a = tc / (1 + tc * W)
        s11 = w.sum() - (a * W * W).sum(); s1y = (w * y).sum() - (a * W * S).sum()
        mu = s1y / s11
        R = np.bincount(g, w * (y - mu), C)
        u = tc * R / (1 + tc * W)
        u_var = tc - tc ** 2 * W / (1 + tc * W)
        return mu, float(np.sqrt(1 / s11)), u, u_var

    # ---- mixed-effects moderator model: z ~ C(cluster) + field RE ----
    def fit_moderator(self, g, C):
        y = self.y
        if self.dense:
            X = onehot(g, C)

            def parts(t):
                cf = cho_factor(self.Ve + t * self.I, lower=True, check_finite=False)
                ViX = cho_solve(cf, X, check_finite=False); Viy = cho_solve(cf, y, check_finite=False)
                A = X.T @ ViX; cfa = cho_factor(A, lower=True, check_finite=False)
                b = cho_solve(cfa, X.T @ Viy, check_finite=False)
                return cf, cfa, A, b, Viy

            def ll(t):
                cf, cfa, A, b, Viy = parts(t)
                rss = y @ Viy - (X.T @ Viy) @ b
                return -0.5 * (2 * np.log(np.diag(cf[0])).sum() + 2 * np.log(np.diag(cfa[0])).sum()
                               + rss + (self.k - C) * LOG2PI)
            t, l = self.max1d(ll)
            _, _, A, b, _ = parts(t)
            Lc = np.hstack([np.eye(C - 1), -np.ones((C - 1, 1))])     # b_c - b_C contrasts
            Cov = np.linalg.inv(A); Lb = Lc @ b
            QM = float(Lb @ np.linalg.solve(Lc @ Cov @ Lc.T, Lb))
            return dict(tf=t, ll=l, QM=QM, df=C - 1, p=float(chi2.sf(QM, C - 1)))

        def ll_vec(T):
            d = self.s2[None, :] + T[:, None]; w = 1 / d; G = onehot(g, C)
            W = w @ G; S = (w * y) @ G
            rss = (w * y * y).sum(1) - (S * S / W).sum(1)
            return -0.5 * (np.log(d).sum(1) + np.log(W).sum(1) + rss + (self.k - C) * LOG2PI)
        t, l = self.max1d(lambda t: float(ll_vec(np.array([t]))[0]), ll_vec)
        w = 1 / (self.s2 + t); W = np.bincount(g, w, C); S = np.bincount(g, w * y, C)
        b = S / W; bbar = (W * b).sum() / W.sum()
        QM = float((W * (b - bbar) ** 2).sum())
        return dict(tf=t, ll=l, QM=QM, df=C - 1, p=float(chi2.sf(QM, C - 1)))

    def cochran(self):
        """Cochran Q (generalised: (z - zbar)' V_e^-1 (z - zbar) for a full V_e), I^2, H^2.
        With covariates: residual Q_E = r' V_e^-1 r, r = z - X b_FE, df = k - p."""
        y = self.y; k = self.k
        X = self.X if self.X is not None else np.ones((k, 1))
        Wm = np.linalg.inv(self.Ve) if self.dense else np.diag(1 / self.s2)
        b = np.linalg.solve(X.T @ Wm @ X, X.T @ Wm @ y); r = y - X @ b
        Q = float(r @ Wm @ r); ybar = float(b[0])
        df = k - X.shape[1]; w = 1 / self.s2
        v_typ = (k - 1) * w.sum() / (w.sum() ** 2 - (w ** 2).sum())   # Higgins-Thompson typical s^2
        return dict(Q=Q, df=df, p=float(chi2.sf(Q, df)), I2=max(0.0, (Q - df) / Q), H2=Q / df,
                    v_typ=float(v_typ), mu_fe=float(ybar))


def profile_ci(prof, x_hat, l_max, lo_b, hi_b):
    """95% profile-likelihood CI: {x : 2[l_max - prof(x)] <= 3.84}; returns (lo, hi, hit_bound)."""
    h = lambda x: prof(x) - (l_max - CRIT / 2)
    lo = lo_b if h(lo_b) >= 0 else brentq(h, lo_b, x_hat, xtol=1e-10)
    hi_hit = h(hi_b) >= 0
    hi = hi_b if hi_hit else brentq(h, x_hat, hi_b, xtol=1e-10)
    return float(lo), float(hi), bool(hi_hit)


def reml_fixed_point(y, s2, tol=1e-14, it=100000):
    """classic two-level REML estimating equation, iterated (independent check):
    tau2 <- sum w^2 [(y - mu)^2 - s2] / sum w^2 + 1 / sum w, truncated at 0."""
    t = 0.01
    for _ in range(it):
        w = 1 / (s2 + t); mu = (w * y).sum() / w.sum()
        tn = max(0.0, (w ** 2 * ((y - mu) ** 2 - s2)).sum() / (w ** 2).sum() + 1 / w.sum())
        if abs(tn - t) < tol:
            return tn
        t = tn
    return t


def reml_scoring_X(y, s2, X, tol=1e-13, it=10000):
    """two-level meta-regression REML by Fisher scoring on the score equation (independent check of
    the covariate path): tau2 <- tau2 + [r'W^2 r - tr P] / tr(P P), P = W - W X (X'WX)^-1 X'W,
    truncated at 0."""
    t = 0.01
    for _ in range(it):
        w = 1 / (s2 + t); Xw = X * w[:, None]
        P = np.diag(w) - Xw @ np.linalg.solve(X.T @ Xw, Xw.T); Py = P @ y
        tn = max(0.0, t + (Py @ Py - np.trace(P)) / np.trace(P @ P))
        if abs(tn - t) < tol:
            return tn
        t = tn
    return t


def perm_p(null, obs):
    return float((1 + np.sum(null >= obs - TIE)) / (len(null) + 1))


def mixture_p(lr):
    """0.5 chi2_0 + 0.5 chi2_1 reference; LR below 1e-8 is treated as exactly 0 (p = 1) so that
    optimizer noise at the boundary does not flip the reported p between 0.5 and 1."""
    return 0.5 * float(chi2.sf(lr, 1)) if lr > 1e-8 else 1.0


def mc_se(p, n):
    return float(np.sqrt(p * (1 - p) / n))


# ============================================================ legacy ICC ================
def legacy_icc(gap, se, g, C):
    """scripts/14/24 ICC on the gap scale with the file's floored SE, same optimizer settings;
    raw = tau2/(tau2 + mean within-cluster sample var), corr = tau2/(tau2 + within - mean SE^2)."""
    m = ThreeLevel(gap, se ** 2)
    r = minimize_scalar(lambda lt: -m.loglik(np.exp(lt), 0.0, g, C),
                        bounds=(np.log(1e-6), np.log(2.0)), method="bounded")
    tau2 = float(np.exp(r.x))
    n = np.bincount(g, minlength=C); s1 = np.bincount(g, gap, C); s2 = np.bincount(g, gap ** 2, C)
    se2 = np.bincount(g, se ** 2, C); multi = n >= 2
    var_c = (s2[multi] - s1[multi] ** 2 / n[multi]) / (n[multi] - 1)
    within = float(var_c.mean()); meas = float((se2[multi] / n[multi]).mean())
    return dict(tau2=tau2, icc_raw=tau2 / (tau2 + within),
                icc_corr=tau2 / (tau2 + max(within - meas, 1e-6)), within=within, meas=meas)


# ============================================================ analysis ==================
def core_fit(m, g, C):
    null = m.fit_null(); conly = m.fit_conly(g, C); full = m.fit(g, C, null, conly)
    return null, conly, full


def analyse(sd, sample, spec, JB, perms, rng_pboot, rng_pow, n_pboot, do_power):
    y = sd.z.to_numpy(); inp = spec_inputs(sd, spec, JB)
    cl, g = np.unique(sd.cip2.to_numpy(), return_inverse=True); C = len(cl)
    sizes = np.bincount(g, minlength=C)
    m = ThreeLevel(y, **inp)
    null, conly, full = core_fit(m, g, C)
    tc, tf = full["tc"], full["tf"]
    mu, mu_se, u, u_var = m.gls(tc, tf, g, C)
    mu0, _, _, _ = m.gls(0.0, null["tf"], g, C)
    lr_f = max(0.0, 2 * (full["ll"] - conly["ll"]))               # field component test
    lr2 = max(0.0, 2 * (conly["ll"] - m.loglik(0.0, 0.0, g, C)))  # cluster-only vs fixed effect
    mod = m.fit_moderator(g, C)
    r2_mod = max(0.0, (null["tf"] - mod["tf"]) / null["tf"]) if null["tf"] > 0 else np.nan
    het = m.cochran()
    vt = het["v_typ"]; I2c = tc / (tc + tf + vt); I2f = tf / (tc + tf + vt)

    # ---- profile likelihood CIs ----
    prof_c = lambda t: m.max1d(lambda x: m.loglik(t, x, g, C))[1]
    prof_f = lambda t: m.max1d(lambda x: m.loglik(x, t, g, C))[1]
    prof_icc = lambda r: m.max1d(lambda T: m.loglik(r * T, (1 - r) * T, g, C))[1]
    tc_ci = profile_ci(prof_c, tc, full["ll"], 0.0, UB)
    tf_ci = profile_ci(prof_f, tf, full["ll"], 0.0, UB)
    icc_hat = full["icc"] if np.isfinite(full["icc"]) else 0.0
    icc_ci = profile_ci(prof_icc, icc_hat, full["ll"], 0.0, 1.0)
    rgrid = np.linspace(0, 1, 51)
    icc_prof = np.array([2 * (full["ll"] - prof_icc(r)) for r in rgrid])

    # ---- permutation null (size-preserving: permute the label vector) ----
    ll00 = m.loglik(0.0, 0.0, g, C)                                  # label-free

    def one_perm(p):
        gp = g[p]
        fp = m.fit(gp, C, null); mod_p = m.fit_moderator(gp, C)
        return ((fp["icc"] if np.isfinite(fp["icc"]) else 0.0), fp["lr"], fp["tc"],
                max(0.0, 2 * (fp["conly"]["ll"] - ll00)), mod_p["QM"],
                max(0.0, (null["tf"] - mod_p["tf"]) / null["tf"]) if null["tf"] > 0 else 0.0)
    out = np.array(pmap(one_perm, perms))
    P = {k_: out[:, i].copy() for i, k_ in enumerate(("icc", "lr", "tc", "lr2", "QM", "r2"))}

    res = dict(sample=sample, spec=spec, k=len(y), C=C, n_single=int((sizes == 1).sum()),
               n_multi=int((sizes >= 2).sum()), fields_in_multi=int(sizes[sizes >= 2].sum()),
               mu=mu, mu_se=mu_se, null=null, conly=conly, full=full,
               tc=tc, tf=tf, tc_ci=tc_ci, tf_ci=tf_ci, icc=full["icc"], icc_ci=icc_ci,
               lr=full["lr"], p_lr_asym=mixture_p(full["lr"]), lr_f=lr_f, p_lrf_asym=mixture_p(lr_f),
               lr2=lr2, p_lr2_asym=mixture_p(lr2), mod=mod, r2_mod=r2_mod, het=het, I2c=I2c, I2f=I2f,
               icc_prof=(rgrid, icc_prof), perm=P, n_perm=len(perms),
               p_perm_icc=perm_p(P["icc"], icc_hat), p_perm_lr=perm_p(P["lr"], full["lr"]),
               p_perm_lr2=perm_p(P["lr2"], lr2), p_perm_QM=perm_p(P["QM"], mod["QM"]),
               p_perm_r2=perm_p(P["r2"], r2_mod),
               q95={k_: float(np.quantile(P[k_], .95)) for k_ in ("icc", "lr", "lr2", "QM", "r2")},
               perm_icc_zero=float(np.mean(P["icc"] < 1e-6)),
               g=g, cl=cl, sizes=sizes, u=u, u_var=u_var, y=y, s2=m.s2)

    # ---- parametric bootstrap null from the fitted field-only model (+ power, main spec) ----
    Le = np.linalg.cholesky(m.Ve) if m.dense else None

    def draw(tc_s, tf_s, rng):          # all draws in the parent, in a fixed order
        e = (Le @ rng.standard_normal(len(y))) if m.dense else np.sqrt(m.s2) * rng.standard_normal(len(y))
        return mu0 + np.sqrt(tc_s) * rng.standard_normal(C)[g] + np.sqrt(tf_s) * rng.standard_normal(len(y)) + e

    def fit_sim(ys):
        ms = ThreeLevel(ys, **inp); ns = ms.fit_null(); fs = ms.fit(g, C, ns)
        return fs["lr"], (fs["icc"] if np.isfinite(fs["icc"]) else 0.0)
    pb = np.array(pmap(fit_sim, [draw(0.0, null["tf"], rng_pboot) for _ in range(n_pboot)]))
    crit = float(np.quantile(pb[:, 0], 0.95))
    res.update(pboot_lr=pb[:, 0], pboot_crit=crit, n_pboot=n_pboot,
               p_pboot_lr=perm_p(pb[:, 0], full["lr"]), p_pboot_icc=perm_p(pb[:, 1], icc_hat),
               pboot_icc_q95=float(np.quantile(pb[:, 1], .95)))
    if do_power:
        T = tc + tf
        scen = [("point estimate", tc, tf)] + [(f"ICC={r:.2f}", r * T, (1 - r) * T) for r in POWER_ICC]
        pw = []
        for name, a, b_ in scen:
            sims = np.array(pmap(fit_sim, [draw(a, b_, rng_pow) for _ in range(N_POWER)]))
            pw.append(dict(scenario=name, tc=a, tf=b_, icc_true=a / (a + b_) if a + b_ > 0 else np.nan,
                           power=float(np.mean(sims[:, 0] > crit)),
                           icc_med=float(np.median(sims[:, 1])),
                           icc_lo=float(np.quantile(sims[:, 1], .05)),
                           icc_hi=float(np.quantile(sims[:, 1], .95))))
        res["power"] = pw
    return res


def leave_one_cluster_out(sd, JB, rng):
    """drop each CIP-2 cluster in turn (57 fields), unadjusted and log-n-adjusted (covariate
    re-centred in each subsample). Main spec: LR_c, ICC_true, mixture p and permutation p
    (N_PERM_CHECK perms each; the SAME permutations for both arms). Correlated-error spec: LR_c,
    ICC_true, mixture p."""
    out = []
    for c in sorted(sd.cip2.unique()):
        sub = sd[sd.cip2 != c].reset_index(drop=True)
        cl, g = np.unique(sub.cip2.to_numpy(), return_inverse=True); C = len(cl)
        row = dict(dropped=c, name=C2NAME.get(c, c), n_dropped=int((sd.cip2 == c).sum()), k=len(sub), C=C)
        pl = None
        for adj in ("none", PRIMARY):
            X = None if adj == "none" else design(sub, ADJ_SETS[adj][0]); sfx = "" if adj == "none" else "_adj"
            for spec in (MAIN, CORR):
                m = ThreeLevel(sub.z.to_numpy(), X=X, **spec_inputs(sub, spec, JB))
                null, conly, full = core_fit(m, g, C)
                row[f"lr_{spec}{sfx}"] = full["lr"]; row[f"icc_{spec}{sfx}"] = full["icc"]
                row[f"tc_{spec}{sfx}"] = full["tc"]
                row[f"p_asym_{spec}{sfx}"] = mixture_p(full["lr"])
                if spec == MAIN:
                    if pl is None:
                        pl = [rng.permutation(len(g)) for _ in range(N_PERM_CHECK)]
                    lrs = np.array(pmap(lambda q: m.fit(g[q], C, null)["lr"], pl))
                    row[f"p_perm_fisher{sfx}"] = perm_p(lrs, full["lr"])
        out.append(row)
    return pd.DataFrame(out)


def leave_one_field_out(sd, JB):
    """drop each field in turn (57 fields): LR_c and mixture p, unadjusted and log-n-adjusted,
    independent and correlated errors."""
    def one(i):
        sub = sd.drop(index=i).reset_index(drop=True)
        cl, g = np.unique(sub.cip2.to_numpy(), return_inverse=True); C = len(cl)
        row = dict(field=sd.field[i], cip2=sd.cip2[i], k=len(sub), C=C)
        for adj in ("none", PRIMARY):
            X = None if adj == "none" else design(sub, ADJ_SETS[adj][0]); sfx = "" if adj == "none" else "_adj"
            for spec in (MAIN, CORR):
                m = ThreeLevel(sub.z.to_numpy(), X=X, **spec_inputs(sub, spec, JB))
                full = core_fit(m, g, C)[2]
                row[f"lr_{spec}{sfx}"] = full["lr"]; row[f"p_asym_{spec}{sfx}"] = mixture_p(full["lr"])
        return row
    return pd.DataFrame(pmap(one, range(len(sd))))


# ============================================================ field-size moderator arm ==
def design(sd, cols):
    """intercept + covariates centred within the sample; log_n_sq = (centred log n)^2, centred."""
    Z = []
    for c in cols:
        x = ((sd.log_n - sd.log_n.mean()) ** 2 if c == "log_n_sq" else sd[c]).to_numpy(float)
        Z.append(x - x.mean())
    return np.column_stack([np.ones(len(sd))] + Z)


def strat_perms(n, N, rng):
    """size-preserving label permutations WITHIN strata of n (rank tertiles; ties broken by the
    alphabetical field order): a field can only receive the label of a field of similar size."""
    r = rankdata(n, method="ordinal").astype(int) - 1
    st = (r * N_STRAT) // len(n)
    groups = [np.where(st == s_)[0] for s_ in range(N_STRAT)]
    out = []
    for _ in range(N):
        q = np.arange(len(n))
        for ix in groups:
            q[ix] = ix[rng.permutation(len(ix))]
        out.append(q)
    return out


def analyse_adj(sd, sample, spec, JB, adj, perms, sperms, rng_pb, n_pboot):
    """three-level REML with fixed moderators: z = X b + u_CIP2 + v_field + e. tau2's are RESIDUAL
    variance components; ICC_true = tau2_c/(tau2_c + tau2_f) of the residual. Tests of tau2_c:
    mixture, label permutation with X kept attached to the fields (i.e. conditional on the
    covariates), n-stratified label permutation, parametric bootstrap from the fitted covariate
    + field-only model."""
    y = sd.z.to_numpy(); inp = spec_inputs(sd, spec, JB)
    cl, g = np.unique(sd.cip2.to_numpy(), return_inverse=True); C = len(cl)
    cols = () if adj == "none" else ADJ_SETS[adj][0]
    X = design(sd, cols) if cols else None
    m = ThreeLevel(y, X=X, **inp)
    null, conly, full = core_fit(m, g, C)
    tc, tf = full["tc"], full["tf"]
    b, bcov = m.beta(tc, tf, g, C); b0, b0cov = m.beta(0.0, null["tf"], g, C)
    prof_c = lambda t: m.max1d(lambda x: m.loglik(t, x, g, C))[1]
    prof_f = lambda t: m.max1d(lambda x: m.loglik(x, t, g, C))[1]
    prof_icc = lambda r: m.max1d(lambda T: m.loglik(r * T, (1 - r) * T, g, C))[1]
    icc_hat = full["icc"] if np.isfinite(full["icc"]) else 0.0
    rgrid = np.linspace(0, 1, 51)
    res = dict(sample=sample, spec=spec, adj=adj, cols=cols, k=len(y), C=C, null=null, full=full,
               tc=tc, tf=tf, icc=full["icc"], lr=full["lr"], p_lr_asym=mixture_p(full["lr"]),
               lr_f=max(0.0, 2 * (full["ll"] - conly["ll"])), het=m.cochran(),
               b=b, b_se=np.sqrt(np.diag(bcov)), b0=b0, b0_se=np.sqrt(np.diag(b0cov)),
               tc_ci=profile_ci(prof_c, tc, full["ll"], 0.0, UB),
               tf_ci=profile_ci(prof_f, tf, full["ll"], 0.0, UB),
               icc_ci=profile_ci(prof_icc, icc_hat, full["ll"], 0.0, 1.0),
               icc_prof=(rgrid, np.array([2 * (full["ll"] - prof_icc(r)) for r in rgrid])))
    res["p_lrf_asym"] = mixture_p(res["lr_f"])

    def perm_fit(q):
        fp = m.fit(g[q], C, null)
        return fp["lr"], (fp["icc"] if np.isfinite(fp["icc"]) else 0.0)
    for tag, pl in (("perm", perms), ("strat", sperms)):
        if pl:
            o = np.array(pmap(perm_fit, pl))
            res[f"{tag}_lr"] = o[:, 0]; res[f"n_{tag}"] = len(pl)
            res[f"p_{tag}_lr"] = perm_p(o[:, 0], full["lr"]); res[f"q95_{tag}_lr"] = float(np.quantile(o[:, 0], .95))
            res[f"p_{tag}_icc"] = perm_p(o[:, 1], icc_hat)
    if n_pboot:
        Xd = X if X is not None else np.ones((len(y), 1)); mu0 = Xd @ b0
        Le = np.linalg.cholesky(m.Ve) if m.dense else None
        ys_all = []
        for _ in range(n_pboot):
            e = (Le @ rng_pb.standard_normal(len(y))) if m.dense else np.sqrt(m.s2) * rng_pb.standard_normal(len(y))
            ys_all.append(mu0 + np.sqrt(null["tf"]) * rng_pb.standard_normal(len(y)) + e)

        def fit_sim(ys):
            ms = ThreeLevel(ys, X=X, **inp); return ms.fit(g, C, ms.fit_null())["lr"]
        pb = np.array(pmap(fit_sim, ys_all))
        res.update(n_pboot=n_pboot, p_pboot_lr=perm_p(pb, full["lr"]), pboot_crit=float(np.quantile(pb, .95)))
    return res


def size_diagnostics(samples, JB):
    """how coupling and CIP-2 relate to field size n (number of matched institutions)."""
    out = {}
    for sname, sd in samples.items():
        y = sd.z.to_numpy(); ln = sd.log_n.to_numpy()
        cl, g = np.unique(sd.cip2.to_numpy(), return_inverse=True); C = len(cl)
        sp = spearmanr(sd.z, sd.n); pr = pearsonr(sd.z, ln)
        cm = np.bincount(g, ln, C) / np.bincount(g, minlength=C)
        eta2 = float(((cm[g] - ln.mean()) ** 2).sum() / ((ln - ln.mean()) ** 2).sum())
        Fst = (eta2 / (C - 1)) / ((1 - eta2) / (len(y) - C))
        o = dict(spearman_z_n=float(sp[0]), spearman_p=float(sp[1]), pearson_z_logn=float(pr[0]),
                 pearson_p=float(pr[1]), eta2_logn_cip2=eta2, eta2_F_p=float(fdist.sf(Fst, C - 1, len(y) - C)),
                 k=len(y), C=C)
        s2 = sd.v_fisher.to_numpy(); Xo = design(sd, ("log_n",))
        # overall slope (field-only RE), three-level slope (both RE), within-cluster slope (CIP-2 fixed
        # effects + field RE; only multi-field clusters inform it)
        m = ThreeLevel(y, s2, X=Xo); nl = m.fit_null(); fl = m.fit(g, C, nl)
        b, cv = m.beta(0.0, nl["tf"], g, C); o["slope_overall"], o["slope_overall_se"] = b[1], np.sqrt(cv[1, 1])
        b, cv = m.beta(fl["tc"], fl["tf"], g, C); o["slope_3level"], o["slope_3level_se"] = b[1], np.sqrt(cv[1, 1])
        Xw = np.column_stack([np.ones(len(y)), onehot(g, C)[:, 1:], ln - ln.mean()])
        mw = ThreeLevel(y, s2, X=Xw); nw = mw.fit_null()
        b, cv = mw.beta(0.0, nw["tf"], g, C); o["slope_within"], o["slope_within_se"] = b[-1], np.sqrt(cv[-1, -1])
        o["tau2_field_within_model"] = nw["tf"]
        sizes = np.bincount(g, minlength=C)
        o["within_fields"] = int(sizes[sizes >= 2].sum()); o["within_clusters"] = int((sizes >= 2).sum())
        o["cluster_mean_n"] = {cl[c]: float(sd.n.to_numpy()[g == c].mean()) for c in range(C)}
        out[sname] = o
    return out


def verification(d, JB):
    """V1 closed form / Cholesky == textbook REML; V2 two-level optimizer == REML fixed point;
    V3 legacy reproduction; V4 grid check of the observed optimum; V5 recovery on a large
    synthetic three-level design."""
    out = {}
    y = d.z.to_numpy(); s2 = d.v_fisher.to_numpy()
    cl, g = np.unique(d.cip2.to_numpy(), return_inverse=True); C = len(cl)
    m = ThreeLevel(y, s2); mdiag = ThreeLevel(y, Ve=np.diag(s2)); mc = ThreeLevel(y, **spec_inputs(d, CORR, JB))
    rng = np.random.default_rng(SEED + 1000)
    d1, d2, d3 = [], [], []
    for tc, tf in [(0, 0), (0.01, 0.02), (0.1, 0.001), (0.5, 0.3), (0.03, 0.012)]:
        for gg in (g, g[rng.permutation(len(g))]):
            d1.append(abs(m.loglik(tc, tf, gg, C) - m.loglik_textbook(tc, tf, gg)))
            d2.append(abs(mdiag.loglik(tc, tf, gg, C) - m.loglik(tc, tf, gg, C)))
            d3.append(abs(mc.loglik(tc, tf, gg, C) - mc.loglik_textbook(tc, tf, gg)))
    # covariate (moderator) path: X = [1, log n, coverage], diagonal Woodbury, vectorised grid and
    # Cholesky with correlated V_e, each against the textbook dense REML
    Xc = design(d, ("log_n", "cov_frac"))
    mx = ThreeLevel(y, s2, X=Xc); mxd = ThreeLevel(y, X=Xc, **spec_inputs(d, CORR, JB))
    d4, d5, d6 = [], [], []
    TCg = np.array([0, 0.01, 0.1, 0.5, 0.03]); TFg = np.array([0, 0.02, 0.001, 0.3, 0.012])
    for gg in (g, g[rng.permutation(len(g))]):
        for tc, tf in zip(TCg, TFg):
            d4.append(abs(mx.loglik(tc, tf, gg, C) - mx.loglik_textbook(tc, tf, gg)))
            d6.append(abs(mxd.loglik(tc, tf, gg, C) - mxd.loglik_textbook(tc, tf, gg)))
        d5.append(np.abs(mx.loglik_grid(TCg, TFg, gg, C) - np.array([mx.loglik(a, b, gg, C) for a, b in zip(TCg, TFg)])).max())
    out["V1_X_woodbury_vs_textbook_maxabs"] = float(max(d4))
    out["V1_X_grid_vs_pointwise_maxabs"] = float(max(d5))
    out["V1_X_cholesky_corr_vs_textbook_maxabs"] = float(max(d6))
    out["V1_woodbury_vs_textbook_maxabs"] = float(max(d1))
    out["V1_cholesky_diag_vs_woodbury_maxabs"] = float(max(d2))
    out["V1_cholesky_corr_vs_textbook_maxabs"] = float(max(d3))
    v2 = []
    for spec in ("fisher", "bonett_wright", "bootstrap"):
        for sub in (d, d[d.reliable]):
            mm = ThreeLevel(sub.z.to_numpy(), sub[f"v_{spec}"].to_numpy())
            v2.append(abs(mm.fit_null()["tf"] - reml_fixed_point(sub.z.to_numpy(), sub[f"v_{spec}"].to_numpy())))
    out["V2_twolevel_vs_fixedpoint_maxabs"] = float(max(v2))
    v2x = []
    for spec in ("fisher", "bonett_wright", "bootstrap"):
        for sub in (d, d[d.reliable].reset_index(drop=True)):
            for cols in (("log_n",), ("log_n", "cov_frac")):
                Xs = design(sub, cols); ys_ = sub.z.to_numpy(); vs_ = sub[f"v_{spec}"].to_numpy()
                v2x.append(abs(ThreeLevel(ys_, vs_, X=Xs).fit_null()["tf"] - reml_scoring_X(ys_, vs_, Xs)))
    out["V2_X_twolevel_vs_scoring_maxabs"] = float(max(v2x))
    leg_orig, _ = legacy_reml_tau2(d.gap.values, d.se.values, d.cip2.values)
    leg_fast = legacy_icc(d.gap.to_numpy(), d.se.to_numpy(), g, C)
    out["V3_legacy_tau2_original"] = float(leg_orig)
    out["V3_legacy_tau2_fast"] = leg_fast["tau2"]
    out["V3_legacy_icc_raw"] = leg_fast["icc_raw"]; out["V3_legacy_icc_corr"] = leg_fast["icc_corr"]
    null = m.fit_null(); full = m.fit(g, C, null)
    gc = np.linspace(0, 0.12, 241); gf = np.linspace(0, 0.06, 241)
    TC, TF = np.meshgrid(gc, gf, indexing="ij")
    out["V4_grid_max_ll"] = float(m.loglik_grid(TC.ravel(), TF.ravel(), g, C).max())
    out["V4_fit_ll"] = full["ll"]
    rs = np.random.default_rng(SEED + 2000)
    Cs, per = 150, 4; gs = np.repeat(np.arange(Cs), per); tc0, tf0 = 0.03, 0.012
    est = []
    for _ in range(200):
        s2s = rs.choice(s2, Cs * per)
        ys = 0.4 + np.sqrt(tc0) * rs.standard_normal(Cs)[gs] + np.sqrt(tf0 + s2s) * rs.standard_normal(Cs * per)
        ms = ThreeLevel(ys, s2s); fs = ms.fit(gs, Cs, ms.fit_null())
        est.append((fs["tc"], fs["tf"]))
    est = np.array(est)
    out["V5_true_tc"], out["V5_true_tf"] = tc0, tf0
    out["V5_mean_tc"], out["V5_mean_tf"] = float(est[:, 0].mean()), float(est[:, 1].mean())
    out["V5_sd_tc"], out["V5_sd_tf"] = float(est[:, 0].std(ddof=1)), float(est[:, 1].std(ddof=1))
    return out


def error_corr_summary(d, JB):
    R = JB["R"]; g = d.cip2.to_numpy(str); same = g[:, None] == g[None, :]
    off = ~np.eye(len(d), dtype=bool)
    per = {}
    for c in sorted(set(g)):
        k = np.where(g == c)[0]
        if len(k) > 1:
            per[c] = float(R[np.ix_(k, k)][np.triu_indices(len(k), 1)].mean())
    return dict(within=float(R[same & off].mean()), between=float(R[~same].mean()), per=per,
                var_ratio_med=float(np.median(np.diag(JB["V"]) / d.v_fisher.to_numpy())))


# ============================================================ outputs ===================
def build_table(R, legacy, ver, loo, ec, JB, dd, A, SZ, lofo):
    rows = []

    def add(sample, spec, stat, est, lo=np.nan, hi=np.nan, p=np.nan, p_method="", note="", adjust="none"):
        rows.append(dict(sample=sample, var_spec=spec, adjust=adjust, statistic=stat, estimate=est, ci_lo=lo,
                         ci_hi=hi, p_value=p, p_method=p_method, note=note))
    for r in R.values():
        s, v = r["sample"], r["spec"]
        add(s, v, "k_fields", r["k"]); add(s, v, "n_clusters", r["C"])
        add(s, v, "n_singleton_clusters", r["n_single"]); add(s, v, "n_multi_clusters", r["n_multi"])
        add(s, v, "fields_in_multi_clusters", r["fields_in_multi"])
        add(s, v, "Q_cochran", r["het"]["Q"], p=r["het"]["p"], p_method="chi2(k-1)",
            note=f"df={r['het']['df']}")
        add(s, v, "I2_total", r["het"]["I2"]); add(s, v, "H2", r["het"]["H2"])
        add(s, v, "typical_sampling_var", r["het"]["v_typ"])
        add(s, v, "mu_z_threelevel", r["mu"], r["mu"] - Z975 * r["mu_se"], r["mu"] + Z975 * r["mu_se"])
        add(s, v, "mu_rho_threelevel", np.tanh(r["mu"]), np.tanh(r["mu"] - Z975 * r["mu_se"]),
            np.tanh(r["mu"] + Z975 * r["mu_se"]))
        add(s, v, "tau2_cluster", r["tc"], *r["tc_ci"][:2], note="profile-likelihood 95% CI")
        add(s, v, "tau2_field", r["tf"], *r["tf_ci"][:2], note="profile-likelihood 95% CI")
        add(s, v, "ICC_true", r["icc"], *r["icc_ci"][:2], p=r["p_perm_icc"], p_method="label permutation",
            note=f"profile CI; perm null q95={r['q95']['icc']:.4f}; share of perms at 0={r['perm_icc_zero']:.3f}")
        add(s, v, "ICC_true", r["icc"], p=r["p_pboot_icc"], p_method="parametric bootstrap",
            note=f"pboot null q95={r['pboot_icc_q95']:.4f}")
        add(s, v, "LR_cluster", r["lr"], p=r["p_lr_asym"], p_method="0.5*chi2_1 mixture")
        add(s, v, "LR_cluster", r["lr"], p=r["p_perm_lr"], p_method="label permutation",
            note=f"perm null q95={r['q95']['lr']:.4f}")
        add(s, v, "LR_cluster", r["lr"], p=r["p_pboot_lr"], p_method="parametric bootstrap",
            note=f"pboot null q95={r['pboot_crit']:.4f}; N={r['n_pboot']}")
        add(s, v, "LR_field", r["lr_f"], p=r["p_lrf_asym"], p_method="0.5*chi2_1 mixture",
            note="tau2_field > 0 given tau2_cluster")
        add(s, v, "I2_cluster_multilevel", r["I2c"]); add(s, v, "I2_field_multilevel", r["I2f"])
        add(s, v, "tau2_field_only_model", r["null"]["tf"], note="two-level, no cluster term")
        add(s, v, "tau2_cluster_only_model", r["conly"]["tc"], note="two-level, no field term (old structure, z scale)")
        add(s, v, "LR_cluster_only_model", r["lr2"], p=r["p_lr2_asym"], p_method="0.5*chi2_1 mixture")
        add(s, v, "LR_cluster_only_model", r["lr2"], p=r["p_perm_lr2"], p_method="label permutation",
            note=f"perm null q95={r['q95']['lr2']:.4f}")
        add(s, v, "QM_moderator", r["mod"]["QM"], p=r["mod"]["p"], p_method=f"chi2({r['mod']['df']})",
            note=f"z ~ C(CIP2) + field RE; residual tau2={r['mod']['tf']:.5f}")
        add(s, v, "QM_moderator", r["mod"]["QM"], p=r["p_perm_QM"], p_method="label permutation",
            note=f"perm null q95={r['q95']['QM']:.4f}")
        add(s, v, "pseudoR2_moderator", r["r2_mod"], p=r["p_perm_r2"], p_method="label permutation",
            note=f"(tau2_field_only - tau2_resid)/tau2_field_only; perm null q95={r['q95']['r2']:.4f}")
        add(s, v, "n_permutations", r["n_perm"])
        for pw in r.get("power", []):
            add(s, v, "power_LR_cluster", pw["power"], p_method="parametric simulation",
                note=(f"scenario={pw['scenario']}; true ICC={pw['icc_true']:.3f}; tau2_c={pw['tc']:.5f}; "
                      f"tau2_f={pw['tf']:.5f}; ICC_hat median={pw['icc_med']:.3f} "
                      f"[5%={pw['icc_lo']:.3f}, 95%={pw['icc_hi']:.3f}]; N={N_POWER}"))
        if v in (MAIN, CORR):
            for c, n_c, u, uv in zip(r["cl"], r["sizes"], r["u"], r["u_var"]):
                zc = r["mu"] + u; sd = np.sqrt(max(uv, 0) + r["mu_se"] ** 2)
                add(s, v, "cluster_conditional_rho", np.tanh(zc), np.tanh(zc - Z975 * sd),
                    np.tanh(zc + Z975 * sd), note=f"cip2={c} {C2NAME.get(c, c)}; n_fields={n_c}")
    for s, L in legacy.items():
        add(s, "legacy_gap_SEfloor", "legacy_tau2", L["obs"]["tau2"], note="scripts/14/24 gap ~ 1+(1|CIP2)")
        add(s, "legacy_gap_SEfloor", "legacy_ICC_raw", L["obs"]["icc_raw"], p=L["p_raw"],
            p_method="label permutation", note=f"perm null q95={L['q95_raw']:.4f}; N={N_PERM}")
        add(s, "legacy_gap_SEfloor", "legacy_ICC_measurement_corrected", L["obs"]["icc_corr"], p=L["p_corr"],
            p_method="label permutation", note=f"perm null q95={L['q95_corr']:.4f}; N={N_PERM}")
    # ---- field-size moderator arm ----
    for (s, v, adj), a in A.items():
        cols = ",".join(a["cols"]) if a["cols"] else "intercept only"
        if adj != "none":
            for j, c in enumerate(a["cols"]):
                add(s, v, f"beta_{c}", a["b"][j + 1], a["b"][j + 1] - Z975 * a["b_se"][j + 1],
                    a["b"][j + 1] + Z975 * a["b_se"][j + 1], p=2 * norm.sf(abs(a["b"][j + 1] / a["b_se"][j + 1])),
                    p_method="Wald (3-level fit)", note="covariate centred within sample", adjust=adj)
            add(s, v, "Q_E_residual", a["het"]["Q"], p=a["het"]["p"], p_method=f"chi2({a['het']['df']})",
                note=f"I2_residual={a['het']['I2']:.4f}", adjust=adj)
        add(s, v, "tau2_cluster", a["tc"], *a["tc_ci"][:2], note="residual; profile-likelihood 95% CI", adjust=adj)
        add(s, v, "tau2_field", a["tf"], *a["tf_ci"][:2], note="residual; profile-likelihood 95% CI", adjust=adj)
        add(s, v, "ICC_true", a["icc"], *a["icc_ci"][:2], note="residual ICC; profile CI", adjust=adj)
        add(s, v, "LR_cluster", a["lr"], p=a["p_lr_asym"], p_method="0.5*chi2_1 mixture", note=cols, adjust=adj)
        if "p_perm_lr" in a:
            add(s, v, "LR_cluster", a["lr"], p=a["p_perm_lr"], p_method="label permutation (covariates fixed to fields)",
                note=f"perm null q95={a['q95_perm_lr']:.4f}; N={a['n_perm']}", adjust=adj)
        if "p_strat_lr" in a:
            add(s, v, "LR_cluster", a["lr"], p=a["p_strat_lr"], p_method="n-stratified label permutation",
                note=f"within {N_STRAT} rank strata of n; null q95={a['q95_strat_lr']:.4f}; N={a['n_strat']}", adjust=adj)
        if "p_pboot_lr" in a:
            add(s, v, "LR_cluster", a["lr"], p=a["p_pboot_lr"], p_method="parametric bootstrap",
                note=f"from fitted covariate + field-only model; null q95={a['pboot_crit']:.4f}; N={a['n_pboot']}",
                adjust=adj)
        add(s, v, "LR_field", a["lr_f"], p=a["p_lrf_asym"], p_method="0.5*chi2_1 mixture", adjust=adj)
    for s, o in SZ.items():
        add(s, "fisher", "spearman_z_vs_n", o["spearman_z_n"], p=o["spearman_p"], p_method="Spearman")
        add(s, "fisher", "pearson_z_vs_log_n", o["pearson_z_logn"], p=o["pearson_p"], p_method="Pearson")
        add(s, "fisher", "eta2_log_n_on_cip2", o["eta2_logn_cip2"], p=o["eta2_F_p"], p_method="one-way F",
            note=f"k={o['k']}; C={o['C']}")
        for nm, lab in (("overall", "z ~ log n + field RE"), ("3level", "z ~ log n + CIP2 RE + field RE"),
                        ("within", f"z ~ C(CIP2) + log n + field RE; {o['within_fields']} fields in "
                                   f"{o['within_clusters']} multi-field clusters inform it")):
            b_, se_ = o[f"slope_{nm}"], o[f"slope_{nm}_se"]
            add(s, "fisher", f"slope_log_n_{nm}", b_, b_ - Z975 * se_, b_ + Z975 * se_,
                p=2 * norm.sf(abs(b_ / se_)), p_method="Wald", note=lab)
        for c, v_ in o["cluster_mean_n"].items():
            add(s, "", "cluster_mean_n_institutions", v_, note=f"cip2={c} {C2NAME.get(c, c)}")
    for _, r in loo.iterrows():
        tag = f"drop cip2={r.dropped} {r['name']} ({r.n_dropped} fields); k={r.k}; C={r.C}"
        for adj, sfx in (("none", ""), (PRIMARY, "_adj")):
            add("nondegen57_loo", MAIN, "LR_cluster", r[f"lr_fisher{sfx}"], p=r[f"p_perm_fisher{sfx}"],
                p_method="label permutation", adjust=adj,
                note=tag + f"; ICC_true={r[f'icc_fisher{sfx}']:.4f}; tau2_c={r[f'tc_fisher{sfx}']:.5f}; "
                           f"mixture p={r[f'p_asym_fisher{sfx}']:.4g}; N={N_PERM_CHECK}")
            add("nondegen57_loo", CORR, "LR_cluster", r[f"lr_fisher_corr{sfx}"], p=r[f"p_asym_fisher_corr{sfx}"],
                p_method="0.5*chi2_1 mixture", adjust=adj,
                note=tag + f"; ICC_true={r[f'icc_fisher_corr{sfx}']:.4f}; tau2_c={r[f'tc_fisher_corr{sfx}']:.5f}")
    for _, r in lofo.iterrows():
        tag = f"drop field={r.field} (cip2={r.cip2}); k={r.k}; C={r.C}"
        for adj, sfx in (("none", ""), (PRIMARY, "_adj")):
            for v in (MAIN, CORR):
                add("nondegen57_lofo", v, "LR_cluster", r[f"lr_{v}{sfx}"], p=r[f"p_asym_{v}{sfx}"],
                    p_method="0.5*chi2_1 mixture", note=tag, adjust=adj)
    add("nondegen57", "joint_bootstrap", "error_corr_within_cluster_mean", ec["within"],
        note=f"z replicates; B_ok={JB['n_ok']}/{B_JOINT}; institution universe={JB['n_univ']}")
    add("nondegen57", "joint_bootstrap", "error_corr_between_cluster_mean", ec["between"])
    for c, v_ in ec["per"].items():
        add("nondegen57", "joint_bootstrap", "error_corr_within_cluster", v_, note=f"cip2={c} {C2NAME.get(c, c)}")
    add("nondegen57", "joint_bootstrap", "jointboot_var_over_fisher_median", ec["var_ratio_med"])
    add("nondegen57", "bootstrap", "boot_over_fisher_var_median", float(np.median(dd.v_bootstrap / dd.v_fisher)))
    for k_, v_ in ver.items():
        add("verification", "", k_, v_)
    t = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT_CSV, index=False, float_format="%.8g")
    return t


def make_figure(R, legacy_main, loo, A, dd):
    plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "axes.linewidth": 1.0})
    BLUE, BLUE2, RED, NEU, TEAL, VIO = "#0F4D92", "#3775BA", "#B64342", "#9A9A9A", "#42949E", "#9A4D8E"
    ORA = "#C7771F"
    a, ac, b = R[("nondegen57", MAIN)], R[("nondegen57", CORR)], R[("reliable20", MAIN)]
    aa = A[("nondegen57", MAIN, PRIMARY)]
    fig = plt.figure(figsize=(13.5, 16.5))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.2, 1, 1, 1.05], hspace=0.95, wspace=0.36)

    # (a) fields by cluster
    ax = fig.add_subplot(gs[0, :]); r = a
    order = np.argsort(r["mu"] + r["u"], kind="stable")
    for xi, c in enumerate(order):
        idx = np.where(r["g"] == c)[0]
        offs = np.linspace(-0.25, 0.25, len(idx)) if len(idx) > 1 else np.array([0.0])
        col = NEU if len(idx) == 1 else BLUE2
        ys = r["y"][idx]; es = Z975 * np.sqrt(r["s2"][idx]); o = np.argsort(ys, kind="stable")
        ax.errorbar(xi + offs, ys[o], yerr=es[o], fmt="o", ms=3.5, color=col, elinewidth=0.8, capsize=0)
        ax.plot([xi - 0.32, xi + 0.32], [r["mu"] + r["u"][c]] * 2, color=RED, lw=2.0)
    ax.axhline(r["mu"], color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(order)))
    mean_n = {c_: float(dd.n[dd.cip2 == c_].mean()) for c_ in r["cl"]}
    ax.set_xticklabels([f"{C2NAME.get(r['cl'][c], r['cl'][c])} ({r['sizes'][c]}; n={mean_n[r['cl'][c]]:.0f})"
                        for c in order], rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("Fisher z of coupling $\\rho_f$")
    sec = ax.secondary_yaxis("right", functions=(np.tanh, np.arctanh)); sec.set_ylabel("$\\rho_f$")
    ax.set_title(f"(a) {r['k']} non-degenerate fields by CIP-2 cluster (n fields; mean n institutions). Dots: field z "
                 "$\\pm$1.96 SE [1/(n-3)], grey = singleton cluster. Red: cluster conditional mean (3-level BLUP, "
                 "unadjusted, main spec). Dashed: $\\hat\\mu$.", fontsize=9, loc="left")

    # (b) profile likelihood of ICC_true
    ax = fig.add_subplot(gs[1, 0])
    for rr, col, ls, lab in ((a, BLUE, "-", "57 fields, 1/(n-3)"),
                             (ac, VIO, "--", "57 fields, correlated errors"),
                             (b, RED, "-", "reliable 20, 1/(n-3)"),
                             (aa, ORA, "-.", "57 fields, 1/(n-3), + log n")):
        rg, pr = rr["icc_prof"]; ax.plot(rg, pr, color=col, ls=ls, lw=1.6, label=lab)
    ax.axhline(CRIT, color="k", ls=":", lw=0.8)
    ax.set_ylim(0, 12); ax.set_xlabel("ICC$_{true}$ = $\\tau^2_c/(\\tau^2_c+\\tau^2_f)$")
    ax.set_ylabel("2$\\Delta$ REML log-lik (profile)"); ax.legend(fontsize=7)
    ax.set_title("(b) profile likelihood of ICC$_{true}$\n(dotted = 3.84, 95% cut)", fontsize=9, loc="left")

    def null_panel(ax, null, obs, q95, title, xlabel, col, rng_=None, extra=None):
        hi = max(null.max(), obs, extra or 0) * 1.05 if rng_ is None else rng_
        ax.hist(null, bins=np.linspace(0, hi, 41), color=NEU, edgecolor="white", lw=0.3)
        ax.axvline(obs, color=col, lw=2)
        ax.axvline(q95, color="k", ls=":", lw=1)
        if extra is not None:
            ax.axvline(extra, color=NEU, ls="--", lw=1.2)
        ax.set_xlabel(xlabel); ax.set_ylabel("count")
        ax.set_title(title + f"\nobserved {obs:.2f} (solid); null 95th pct {q95:.2f} (dotted)",
                     fontsize=9, loc="left")

    null_panel(fig.add_subplot(gs[1, 1]), a["perm"]["icc"], a["icc"], a["q95"]["icc"],
               f"(c) ICC$_{{true}}$ null, 57 fields, 1/(n-3)\nperm p = {fmt_p(a['p_perm_icc'])} "
               f"({a['n_perm']} perms)", "ICC$_{true}$ under label permutation", BLUE, 1.0)
    null_panel(fig.add_subplot(gs[1, 2]), a["perm"]["lr"], a["lr"], a["q95"]["lr"],
               f"(d) LR$_{{cluster}}$ null, 57 fields, 1/(n-3), unadjusted\nperm p = {fmt_p(a['p_perm_lr'])}; "
               f"param. boot p = {fmt_p(a['p_pboot_lr'])}", "LR$_{cluster}$ under label permutation", BLUE)
    null_panel(fig.add_subplot(gs[2, 0]), ac["perm"]["lr"], ac["lr"], ac["q95"]["lr"],
               f"(e) LR$_{{cluster}}$ null, 57 fields, corr. errors\nperm p = {fmt_p(ac['p_perm_lr'])}; "
               f"param. boot p = {fmt_p(ac['p_pboot_lr'])}", "LR$_{cluster}$ under label permutation", VIO)
    L = legacy_main
    null_panel(fig.add_subplot(gs[2, 1]), L["null_raw"], L["obs"]["icc_raw"], L["q95_raw"],
               f"(f) legacy raw ICC (scripts/14/24), 57 fields\nperm p = {fmt_p(L['p_raw'])}",
               "legacy raw ICC under label permutation", TEAL, 1.0)

    # (g) leave one cluster out, unadjusted and log-n-adjusted
    ax = fig.add_subplot(gs[2, 2])
    xs = np.arange(len(loo))
    ax.scatter(xs, loo.p_perm_fisher, color=BLUE, s=14, label="unadjusted, 1/(n-3): perm p", zorder=3)
    ax.scatter(xs, loo.p_asym_fisher_corr, color=VIO, marker="s", s=12, label="unadjusted, corr.: mixture p", zorder=3)
    ax.scatter(xs, loo.p_perm_fisher_adj, color=ORA, marker="D", s=14, label="+ log n, 1/(n-3): perm p", zorder=3)
    ax.scatter(xs, loo.p_asym_fisher_corr_adj, facecolor="none", edgecolor=ORA, marker="s", s=16,
               label="+ log n, corr.: mixture p", zorder=3)
    ax.axhline(0.05, color="k", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([f"{n} ({k})" for n, k in zip(loo["name"], loo.n_dropped)],
                                          rotation=70, ha="right", fontsize=6.5)
    ax.set_ylim(1e-5, 1)
    ax.set_ylabel("p for LR$_{cluster}$ (log)"); ax.legend(fontsize=6.3, loc="lower center", ncol=2, columnspacing=0.8)
    ax.set_title("(g) leave one CIP-2 cluster out (57 fields)\n(dotted = 0.05)", fontsize=9, loc="left")

    # (h) coupling vs field size
    ax = fig.add_subplot(gs[3, 0])
    big = [c_ for c_, _ in sorted(dd.cip2.value_counts().items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
    pal = dict(zip(big, [BLUE, RED, TEAL, VIO, ORA]))
    oth = ~dd.cip2.isin(big)
    ax.scatter(dd.n[oth], dd.z[oth], s=14, color=NEU, label="other clusters", zorder=2)
    for c_ in big:
        k_ = dd.cip2 == c_
        ax.scatter(dd.n[k_], dd.z[k_], s=18, color=pal[c_], label=f"{C2NAME.get(c_, c_)} ({int(k_.sum())})", zorder=3)
    ln0 = float(dd.log_n.mean()); xx = np.linspace(dd.log_n.min(), dd.log_n.max(), 50)
    b0, b1 = aa["b0"][0], aa["b0"][1]
    ax.plot(np.exp(xx), b0 + b1 * (xx - ln0), color="k", lw=1.4, label=f"overall slope {b1:.3f}/log n")
    ax.set_xscale("log"); ax.set_xlabel("n institutions matched (log scale)"); ax.set_ylabel("Fisher z of $\\rho_f$")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("(h) coupling rises with field size n\n(57 fields; line: field-only REML fit z ~ log n)",
                 fontsize=9, loc="left")

    # (i) permutation null of LR_c conditional on n
    null_panel(fig.add_subplot(gs[3, 1]), aa["perm_lr"], aa["lr"], aa["q95_perm_lr"],
               f"(i) LR$_{{cluster}}$ null, log n in the model\n57 fields, 1/(n-3): perm p = {fmt_p(aa['p_perm_lr'])}\n"
               f"dashed = unadjusted LR {a['lr']:.2f}",
               "LR$_{cluster}$ | log n, under label permutation", ORA, extra=a["lr"])

    # (j) tau2_cluster across adjustment sets
    ax = fig.add_subplot(gs[3, 2])
    items = [("unadjusted", ("nondegen57", MAIN, "none")), ("unadjusted, corr.", ("nondegen57", CORR, "none"))]
    items += [(ADJ_SETS[k_][1], ("nondegen57", MAIN, k_)) for k_ in ADJ_SETS]
    items += [("log n, corr.", ("nondegen57", CORR, PRIMARY)), ("rel. 20 unadjusted", ("reliable20", MAIN, "none")),
              ("rel. 20 log n", ("reliable20", MAIN, PRIMARY))]
    for yi, (lab, key) in enumerate(items):
        q = A[key]
        pq = q.get("p_perm_lr", R[(key[0], key[1])]["p_perm_lr"] if key[2] == "none" else np.nan)
        col = ORA if key[2] != "none" else BLUE
        ax.plot([q["tc_ci"][0], q["tc_ci"][1]], [yi, yi], color=col, lw=1.4)
        ax.plot([q["tc"]], [yi], "o", color=col, ms=4)
        ax.text(0.132, yi, f"p={pq:.3f}", va="center", fontsize=6.5)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([t for t, _ in items], fontsize=6.5)
    ax.invert_yaxis(); ax.set_xlim(0, 0.16); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("$\\tau^2_{cluster}$ (residual) with 95% profile CI")
    ax.set_title("(j) CIP-2 variance by moderator set\n(57 fields, 1/(n-3) unless noted; p = label perm.)",
                 fontsize=9, loc="left")
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


# ============================================================ report ====================
def fmt_p(p):
    return f"{p:.4f}" if p >= 0.001 else f"{p:.1e}"


def ci_str(x, c, f=".4f"):
    return f"{x:{f}} [{c[0]:{f}}, {c[1]:{f}}]" + (" (upper CI at bound)" if c[2] else "")


def write_report(R, legacy, ver, meta, loo, ec, JB, dd, A, SZ, lofo):
    a, b = R[("nondegen57", MAIN)], R[("reliable20", MAIN)]
    ac, bc = R[("nondegen57", CORR)], R[("reliable20", CORR)]
    La, Lb = legacy["nondegen57"], legacy["reliable20"]
    L = []
    L.append("# Three-level ICC refit: is the discipline (CIP-2) structure of coupling real?\n")
    L.append("Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/57_icc_threelevel.py` "
             f"(seed {SEED}; {N_PERM} label permutations for the main spec, {N_PERM_CHECK} for every other "
             f"spec, each moderator fit and each leave-one-cluster-out fit ({N_PERM} for the primary log-n fit on "
             f"57 fields); {N_PBOOT} / {N_PBOOT_DENSE} parametric-bootstrap draws for independent / "
             f"correlated-error specs; {N_POWER} draws per power scenario; {B_JOINT} joint institution-bootstrap "
             "draws). Outputs: `data/interim/icc_threelevel.csv` (every number below; column `adjust` names the "
             "moderator set), `outputs/figures/icc_threelevel.png`.\n")
    L.append("## Answer\n")
    L.append(_answer_text(R, A, SZ, a, b, ac, bc, La, loo, lofo, ec))

    L.append("\n## What changed in this revision\n")
    aa = A[("nondegen57", MAIN, PRIMARY)]
    L.append("- The previous version answered an unqualified yes. It did not test a confound the verifiers "
             "raised: the number of institutions per field (n). Coupling rises with n, and CIP-2 clusters differ "
             "in mean n, so the size-preserving label permutation credited part of a size gradient to discipline. "
             "Its robustness claims (all variance specs, every leave-one-cluster-out drop) hold only without "
             "that adjustment.")
    L.append("- Added a moderator arm: the same three-level REML with log n (and, as checks, (log n)^2, coverage "
             "n / n_Wapman, prestige range, log earnings-cohort size and pairs of these) as fixed covariates. For "
             "each fit it reports tau2_cluster, tau2_field and ICC_true with profile CIs, the mixture p, a label "
             "permutation p with the covariates kept attached to the fields, and, for log n, a parametric "
             "bootstrap from the n-adjusted field-only model and a label permutation stratified by n. Leave-one-"
             "cluster-out and leave-one-field-out are rerun under the log-n adjustment.")
    n_loo_bad = int(((loo.p_perm_fisher_adj > 0.05) | (loo.p_asym_fisher_corr_adj > 0.05)).sum())
    L.append(f"- Result: the unadjusted numbers are unchanged. Net of log n (57 fields, 1/(n-3)) LR_c goes "
             f"{a['lr']:.2f} -> {aa['lr']:.2f} and the permutation p {fmt_p(a['p_perm_lr'])} -> "
             f"{fmt_p(aa['p_perm_lr'])}; the profile CI of tau2_cluster "
             + ("now reaches 0" if aa["tc_ci"][0] < 1e-6 else f"still excludes 0 (lower bound {aa['tc_ci'][0]:.4f})")
             + f"; {n_loo_bad} of {len(loo)} leave-one-cluster-out drops give p > 0.05 under the adjustment. The "
             "Answer, the recommendation and the robustness claims were rewritten to match.")
    L.append("- Permutation, bootstrap and power loops now run in worker processes. Every random draw is made in "
             "the parent before the workers fit, so the output does not depend on the number of workers.")

    L.append("\n## Key numbers: unadjusted three-level model\n")
    L.append("Outcome z_f = atanh(rho_f); three-level REML z_f = mu + u_CIP2 + v_field + e_f. \"1/(n-3)\" = "
             "independent sampling errors with variance 1/(n_f-3) (the main spec requested). \"corr.\" = same "
             "variances with the joint institution-bootstrap error correlation. CIs are 95% profile likelihood; "
             "permutation / bootstrap p = (1 + #{null >= obs}) / (N + 1).\n")
    L.append("| quantity | 57 fields, 1/(n-3) | 57 fields, corr. | reliable 20, 1/(n-3) | reliable 20, corr. | note |\n"
             "|---|---|---|---|---|---|")

    def row(q, f, note=""):
        L.append(f"| {q} | {f(a)} | {f(ac)} | {f(b)} | {f(bc)} | {note} |")
    row("fields / clusters / singleton clusters", lambda r: f"{r['k']} / {r['C']} / {r['n_single']}")
    row("fields in multi-field clusters", lambda r: f"{r['fields_in_multi']} (in {r['n_multi']} clusters)",
        "only these separate tau2_field from tau2_cluster")
    row("Cochran Q (df), p", lambda r: f"{r['het']['Q']:.1f} ({r['het']['df']}), {fmt_p(r['het']['p'])}",
        "corr. = generalized Q with full V_e")
    row("I^2 total", lambda r: f"{r['het']['I2']:.3f}", "share of observed z variance beyond sampling error")
    row("pooled rho (3-level mu)", lambda r: f"{np.tanh(r['mu']):.3f}")
    row("tau2_cluster [CI]", lambda r: ci_str(r["tc"], r["tc_ci"]), "z scale")
    row("tau2_field [CI]", lambda r: ci_str(r["tf"], r["tf_ci"]), "z scale")
    row("ICC_true [CI]", lambda r: ci_str(r["icc"], r["icc_ci"], ".3f"), "profiled over (ICC, total)")
    row("multilevel I^2 cluster / field", lambda r: f"{r['I2c']:.3f} / {r['I2f']:.3f}")
    row("LR_cluster", lambda r: f"{r['lr']:.3f}", "2[l(full) - l(tau2_c = 0)]")
    row("  p: 0.5 chi2_1 mixture", lambda r: fmt_p(r["p_lr_asym"]), "asymptotic")
    row("  p: label permutation (null q95)",
        lambda r: f"{fmt_p(r['p_perm_lr'])} ({r['q95']['lr']:.2f}); MC SE {mc_se(r['p_perm_lr'], r['n_perm']):.4f}",
        f"size-preserving; {N_PERM} perms for 1/(n-3), {N_PERM_CHECK} for corr.")
    row("  p: parametric bootstrap (null q95)", lambda r: f"{fmt_p(r['p_pboot_lr'])} ({r['pboot_crit']:.2f})",
        "from fitted field-only model")
    L.append(f"|   p: label permutation within n tertiles | {fmt_p(A[('nondegen57', MAIN, 'none')]['p_strat_lr'])} | "
             f"{fmt_p(A[('nondegen57', CORR, 'none')]['p_strat_lr'])} | {fmt_p(A[('reliable20', MAIN, 'none')]['p_strat_lr'])} | "
             f"{fmt_p(A[('reliable20', CORR, 'none')]['p_strat_lr'])} | model-free control for n; {N_PERM_CHECK} perms |")
    row("ICC_true: permutation p (null q95)", lambda r: f"{fmt_p(r['p_perm_icc'])} ({r['q95']['icc']:.3f})",
        "weak statistic when tau2_field is often 0")
    row("LR_field (tau2_field > 0 given cluster), p", lambda r: f"{r['lr_f']:.2f}, {fmt_p(r['p_lrf_asym'])}",
        "mixture p")
    row("moderator Q_M (df): p chi2 / p perm",
        lambda r: f"{r['mod']['QM']:.1f} ({r['mod']['df']}): {fmt_p(r['mod']['p'])} / {fmt_p(r['p_perm_QM'])}",
        "z ~ C(CIP2) + field RE, Wald")
    row("moderator pseudo-R^2, p perm", lambda r: f"{r['r2_mod']:.3f}, {fmt_p(r['p_perm_r2'])}",
        "share of field-only tau2 absorbed by CIP-2 dummies")
    row("old structure on z (cluster-only): LR, p mixture / p perm",
        lambda r: f"{r['lr2']:.1f}, {fmt_p(r['p_lr2_asym'])} / {fmt_p(r['p_perm_lr2'])}", "no field term")
    L.append(f"| legacy raw ICC (scripts/14/24), perm p (null q95) | {La['obs']['icc_raw']:.3f}, {fmt_p(La['p_raw'])} "
             f"({La['q95_raw']:.3f}) | | {Lb['obs']['icc_raw']:.3f}, {fmt_p(Lb['p_raw'])} ({Lb['q95_raw']:.3f}) | | "
             "gap scale, SE floored at 0.08 |")
    L.append(f"| legacy 'measurement-corrected' ICC, perm p (null q95) | {La['obs']['icc_corr']:.3f}, {fmt_p(La['p_corr'])} "
             f"({La['q95_corr']:.3f}) | | {Lb['obs']['icc_corr']:.3f}, {fmt_p(Lb['p_corr'])} ({Lb['q95_corr']:.3f}) | | "
             "not a coherent decomposition |")

    # ---------------- field-size arm ----------------
    L.append("\n## Field size: the confound the permutation null did not control\n")
    s57, s20 = SZ["nondegen57"], SZ["reliable20"]
    cm = s57["cluster_mean_n"]
    lo_c, hi_c = min(cm, key=cm.get), max(cm, key=cm.get)
    L.append(f"n = number of institutions with both a Wapman prestige score and Scorecard earnings for the field. "
             f"57 fields: Spearman(z, n) = {s57['spearman_z_n']:.2f} (p = {fmt_p(s57['spearman_p'])}), Pearson(z, "
             f"log n) = {s57['pearson_z_logn']:.2f} (p = {fmt_p(s57['pearson_p'])}). Reliable 20: Spearman "
             f"{s20['spearman_z_n']:.2f} (p = {fmt_p(s20['spearman_p'])}), Pearson(z, log n) "
             f"{s20['pearson_z_logn']:.2f} (p = {fmt_p(s20['pearson_p'])}). CIP-2 cluster mean n runs from "
             f"{cm[lo_c]:.0f} ({C2NAME.get(lo_c, lo_c)}) to {cm[hi_c]:.0f} ({C2NAME.get(hi_c, hi_c)}); eta^2 of log n "
             f"on CIP-2 = {s57['eta2_logn_cip2']:.2f} on 57 fields (one-way F p = {fmt_p(s57['eta2_F_p'])}, "
             f"{s57['C']} clusters) and {s20['eta2_logn_cip2']:.2f} on the reliable 20 (F p = "
             f"{fmt_p(s20['eta2_F_p'])}). Slope of z on log n (1/(n-3), REML, 57 fields): "
             f"{s57['slope_overall']:.3f} (SE {s57['slope_overall_se']:.3f}) with a field random effect only; "
             f"{s57['slope_3level']:.3f} (SE {s57['slope_3level_se']:.3f}) in the three-level model; "
             f"{s57['slope_within']:.3f} (SE {s57['slope_within_se']:.3f}) within clusters (CIP-2 fixed effects; "
             f"{s57['within_fields']} fields in {s57['within_clusters']} multi-field clusters inform it). The "
             f"within-cluster slope is {s57['slope_within'] / s57['slope_overall']:.2f} of the overall slope. "
             + _within_phrase(s57) +
             f" (Reliable 20: overall {s20['slope_overall']:.3f} (SE {s20['slope_overall_se']:.3f}), "
             f"within {s20['slope_within']:.3f} (SE {s20['slope_within_se']:.3f}) from {s20['within_fields']} "
             f"fields in {s20['within_clusters']} clusters.)\n")

    L.append("### Three-level model with log n as a fixed moderator\n")
    L.append("z_f = b0 + b1 (log n_f - mean) + u_CIP2 + v_field + e_f. tau2's are residual variance components, "
             "ICC_true is the residual share. 'Permutation, n kept' permutes the CIP-2 labels while log n stays "
             "attached to its field (a test conditional on n). 'Within n tertiles' permutes labels only among "
             "fields in the same rank tertile of n.\n")
    L.append("| quantity | 57 fields, 1/(n-3) | 57 fields, corr. | reliable 20, 1/(n-3) | reliable 20, corr. |\n"
             "|---|---|---|---|---|")
    keys = [("nondegen57", MAIN), ("nondegen57", CORR), ("reliable20", MAIN), ("reliable20", CORR)]

    def arow(q, f):
        L.append(f"| {q} | " + " | ".join(f(A[(s_, v_, PRIMARY)], R[(s_, v_)], A[(s_, v_, 'none')]) for s_, v_ in keys) + " |")
    arow("slope of log n [95% CI]", lambda q, r, n0: f"{q['b'][1]:.3f} [{q['b'][1] - Z975 * q['b_se'][1]:.3f}, "
                                                     f"{q['b'][1] + Z975 * q['b_se'][1]:.3f}]")
    arow("tau2_cluster: unadjusted -> + log n [CI]", lambda q, r, n0: f"{r['tc']:.4f} -> {ci_str(q['tc'], q['tc_ci'])}")
    arow("share of unadjusted tau2_cluster removed", lambda q, r, n0: f"{1 - q['tc'] / r['tc']:.2f}")
    arow("tau2_field + log n [CI]", lambda q, r, n0: ci_str(q["tf"], q["tf_ci"]))
    arow("ICC_true: unadjusted -> + log n [CI]", lambda q, r, n0: f"{r['icc']:.3f} -> {ci_str(q['icc'], q['icc_ci'], '.3f')}")
    arow("LR_cluster: unadjusted -> + log n", lambda q, r, n0: f"{r['lr']:.2f} -> {q['lr']:.2f}")
    arow("  p: 0.5 chi2_1 mixture", lambda q, r, n0: fmt_p(q["p_lr_asym"]))
    arow("  p: label permutation, n kept (N; null q95)",
         lambda q, r, n0: f"{fmt_p(q['p_perm_lr'])} ({q['n_perm']}; {q['q95_perm_lr']:.2f}); MC SE {mc_se(q['p_perm_lr'], q['n_perm']):.4f}")
    arow("  p: parametric bootstrap from log n + field-only model (null q95)",
         lambda q, r, n0: f"{fmt_p(q['p_pboot_lr'])} ({q['pboot_crit']:.2f})")
    arow("  p: label permutation within n tertiles", lambda q, r, n0: fmt_p(q["p_strat_lr"]))
    arow("unadjusted LR_c, p: label permutation within n tertiles", lambda q, r, n0: fmt_p(n0["p_strat_lr"]))
    arow("LR_field + log n, mixture p", lambda q, r, n0: f"{q['lr_f']:.2f}, {fmt_p(q['p_lrf_asym'])}")
    arow("residual Q_E (df), I^2_res", lambda q, r, n0: f"{q['het']['Q']:.1f} ({q['het']['df']}), {q['het']['I2']:.3f}")

    L.append("\n### log n adjustment under every sampling-variance specification\n")
    for sname, title in (("nondegen57", "57 non-degenerate fields"), ("reliable20", "reliable 20 fields")):
        L.append(f"{title}:\n")
        L.append("| var spec | LR_c unadj. -> + log n | tau2_cluster [CI] | tau2_field [CI] | ICC_true [CI] | p perm (n kept) | p pboot | p mixture |\n"
                 "|---|---|---|---|---|---|---|---|")
        for s_, desc in VAR_SPECS.items():
            q = A[(sname, s_, PRIMARY)]; r = R[(sname, s_)]
            L.append(f"| {s_} | {r['lr']:.2f} -> {q['lr']:.2f} | {ci_str(q['tc'], q['tc_ci'])} | {ci_str(q['tf'], q['tf_ci'])} | "
                     f"{ci_str(q['icc'], q['icc_ci'], '.3f')} | {fmt_p(q['p_perm_lr'])} | {fmt_p(q['p_pboot_lr'])} | "
                     f"{fmt_p(q['p_lr_asym'])} |")
        L.append("")

    L.append("### Other measurement moderators\n")
    L.append("Covariates centred within sample. coverage = n / n_Wapman (share of the field's Wapman institutions "
             "that have Scorecard earnings); prestige range = SD of the matched institutions' prestige percentile "
             "within the field's full Wapman list; cohort size = median Scorecard earnings-cohort count over matched "
             f"institutions. p perm = label permutation with the covariates kept attached to fields ({N_PERM_CHECK} "
             f"perms; the log-n row uses {N_PERM} on 57 fields, 1/(n-3)).\n")
    L.append("| moderator set | slopes (57, 1/(n-3)) | 57, 1/(n-3): tau2_cluster [CI] | ICC_true [CI] | LR_c | p mixture | p perm | 57, corr.: LR_c, p perm | reliable 20, 1/(n-3): LR_c, p perm |\n"
             "|---|---|---|---|---|---|---|---|---|")
    L.append(f"| none | | {ci_str(a['tc'], a['tc_ci'])} | {ci_str(a['icc'], a['icc_ci'], '.3f')} | {a['lr']:.2f} | "
             f"{fmt_p(a['p_lr_asym'])} | {fmt_p(a['p_perm_lr'])} | {ac['lr']:.2f}, {fmt_p(ac['p_perm_lr'])} | "
             f"{b['lr']:.2f}, {fmt_p(b['p_perm_lr'])} |")
    for adj, (cols, desc) in ADJ_SETS.items():
        q, qc, q20 = A[("nondegen57", MAIN, adj)], A[("nondegen57", CORR, adj)], A[("reliable20", MAIN, adj)]
        sl = "; ".join(f"{c} {q['b'][j + 1]:+.3f} (SE {q['b_se'][j + 1]:.3f})" for j, c in enumerate(cols))
        L.append(f"| {desc} | {sl} | {ci_str(q['tc'], q['tc_ci'])} | {ci_str(q['icc'], q['icc_ci'], '.3f')} | {q['lr']:.2f} | "
                 f"{fmt_p(q['p_lr_asym'])} | {fmt_p(q['p_perm_lr'])} | {qc['lr']:.2f}, {fmt_p(qc['p_perm_lr'])} | "
                 f"{q20['lr']:.2f}, {fmt_p(q20['p_perm_lr'])} |")

    L.append("\n### Leave one CIP-2 cluster out (57 fields), unadjusted and + log n\n")
    L.append(f"Permutation p uses the same {N_PERM_CHECK} label permutations for both arms; log n re-centred in each "
             "subsample.\n")
    L.append("| dropped cluster (fields) | LR_c 1/(n-3) | perm p | ICC_true | LR_c corr. | mixture p corr. | LR_c + log n | perm p + log n | ICC_true + log n | LR_c corr. + log n | mixture p corr. + log n |\n"
             "|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in loo.iterrows():
        L.append(f"| {r['name']} ({r.n_dropped}) | {r.lr_fisher:.2f} | {fmt_p(r.p_perm_fisher)} | {r.icc_fisher:.2f} | "
                 f"{r.lr_fisher_corr:.2f} | {fmt_p(r.p_asym_fisher_corr)} | {r.lr_fisher_adj:.2f} | "
                 f"{fmt_p(r.p_perm_fisher_adj)} | {r.icc_fisher_adj:.2f} | {r.lr_fisher_corr_adj:.2f} | "
                 f"{fmt_p(r.p_asym_fisher_corr_adj)} |")
    L.append("")
    L.append(_lofo_text(lofo))

    L.append("\n### All sampling-variance specifications (unadjusted)\n")
    for sname, title in (("nondegen57", "57 non-degenerate fields"), ("reliable20", "reliable 20 fields")):
        L.append(f"{title}:\n")
        L.append("| var spec | I^2 | tau2_cluster [CI] | tau2_field [CI] | ICC_true [CI] | LR_c | p perm | p pboot | p mixture |\n"
                 "|---|---|---|---|---|---|---|---|---|")
        for s, desc in VAR_SPECS.items():
            r = R[(sname, s)]
            L.append(f"| {s}: {desc} | {r['het']['I2']:.3f} | {ci_str(r['tc'], r['tc_ci'])} | {ci_str(r['tf'], r['tf_ci'])} | "
                     f"{ci_str(r['icc'], r['icc_ci'], '.3f')} | {r['lr']:.3f} | {fmt_p(r['p_perm_lr'])} | "
                     f"{fmt_p(r['p_pboot_lr'])} | {fmt_p(r['p_lr_asym'])} |")
        L.append("")
    L.append(f"Joint institution bootstrap ({JB['n_ok']} of {B_JOINT} draws usable; {JB['n_univ']} institutions in the "
             f"union): mean sampling-error correlation between two fields in the SAME CIP-2 cluster "
             f"{ec['within']:.3f}, in DIFFERENT clusters {ec['between']:.3f}. By cluster: "
             + ", ".join(f"{C2NAME.get(c, c)} {v:.2f}" for c, v in sorted(ec["per"].items(), key=lambda kv: -kv[1]))
             + f". Joint-bootstrap z variance / (1/(n-3)): median {ec['var_ratio_med']:.2f}; file-bootstrap "
             f"z variance / (1/(n-3)): median {np.median(dd.v_bootstrap / dd.v_fisher):.2f}.\n")

    L.append("### Power of the design (parametric simulation, 1/(n-3) spec, unadjusted)\n")
    L.append("Data simulated from the three-level model on the exact field/cluster design and sampling "
             "variances; total true heterogeneity held at the fitted tau2_c + tau2_f; test = LR_cluster "
             "against the parametric-bootstrap 95% critical value.\n")
    L.append("| sample | scenario | true ICC | power | ICC_hat median [5%, 95%] |\n|---|---|---|---|---|")
    for r in (a, b):
        for pw in r["power"]:
            L.append(f"| {r['sample']} | {pw['scenario']} | {pw['icc_true']:.3f} | {pw['power']:.3f} | "
                     f"{pw['icc_med']:.3f} [{pw['icc_lo']:.3f}, {pw['icc_hi']:.3f}] |")

    L.append("\n### Descriptive map: cluster conditional means (57 fields, unadjusted)\n")
    L.append("Conditional mean of z per CIP-2 cluster = mu + BLUP(u_c), back-transformed to rho; interval = "
             "+/-1.96 sqrt(conditional var of u_c + SE(mu)^2) (approximate; ignores uncertainty in the tau2 "
             "estimates). Shrunk toward mu by the estimated tau2_cluster; singletons are shrunk hardest. The "
             "mean-n column is there because field size is a competing explanation for the ordering.\n")
    L.append("| CIP-2 | cluster | n fields | mean n institutions | rho, 1/(n-3) [approx 95%] | rho, corr. [approx 95%] | fields |\n"
             "|---|---|---|---|---|---|---|")
    order = np.argsort(a["mu"] + a["u"], kind="stable")
    for c in order:
        cells = []
        for r in (a, ac):
            zc = r["mu"] + r["u"][c]; sd = np.sqrt(max(r["u_var"][c], 0) + r["mu_se"] ** 2)
            cells.append(f"{np.tanh(zc):.3f} [{np.tanh(zc - Z975 * sd):.3f}, {np.tanh(zc + Z975 * sd):.3f}]")
        members = ", ".join(dd.loc[a["g"] == c, "label"])
        L.append(f"| {a['cl'][c]} | {C2NAME.get(a['cl'][c], a['cl'][c])} | {a['sizes'][c]} | {cm[a['cl'][c]]:.0f} | "
                 f"{cells[0]} | {cells[1]} | {members} |")

    L.append("\n## Method\n")
    ex = meta["excluded"]
    L.append(f"- **Sample.** `outputs/expanded66_gap_map.csv`: {meta['n_total']} fields, {meta['n_gap']} with a "
             f"computed gap. The scripts/14/24 degeneracy filter (gap <= 0.02 or >= 1.6, n < 8, or bootstrap "
             f"CI width < 0.02) drops {len(ex)} ({', '.join(f'{f} n={n}' for f, n in zip(ex.field, ex.n_institutions))}), "
             f"leaving {a['k']} fields in {a['C']} CIP-2 clusters. Reliable subset = the file's `reliable` flag "
             f"({b['k']} fields, {b['C']} clusters).")
    L.append("- **Outcome.** z_f = atanh(rho_f), rho_f = 1 - gap_f = Spearman(Wapman SpringRank prestige, "
             "Scorecard 4-year median earnings) across n_f institutions.")
    L.append("- **Sampling (co)variance specs.** (1) 1/(n_f-3), independent [main]; (2) Bonett-Wright "
             "(1+rho^2/2)/(n_f-3); (3) the file's 95% percentile bootstrap CI for rho (1000 institution "
             "resamples, src/gap.py) mapped to z, ((z_hi - z_lo)/3.92)^2; (4) 1/(n_f-3) variances combined "
             "with the correlation matrix of a joint institution bootstrap; (5) the full joint-bootstrap "
             "covariance. For (4)-(5) each field's matched (prestige, earnings) table is rebuilt with "
             "`load_ar_wapman` / `load_er_scorecard` and the src.gap merge (n and rho asserted equal to the "
             f"file for all {a['k']} fields); each draw resamples the union of institutions once, with "
             "replacement, and recomputes every field's Spearman on its own resampled institutions, so fields "
             "sharing institutions get correlated replicates.")
    L.append("- **Model.** z_f = x_f'b + u_c + v_f + e_f, u_c ~ N(0, tau2_cluster), v_f ~ N(0, tau2_field), "
             "e ~ N(0, V_e) known; x_f = 1 (unadjusted) or 1 plus centred moderators. Marginal V = V_e + "
             "tau2_field I + tau2_cluster * same-CIP2. REML log-likelihood -1/2[log|V| + log|X'V^-1 X| + "
             "r'V^-1 r + (k-p) log 2pi], r = z - X b_GLS; closed form (Woodbury over disjoint clusters) for "
             "diagonal V_e, Cholesky for full V_e. Maximized over [0, 2]^2: sqrt-spaced grid, L-BFGS-B from the "
             "best grid point, and the two boundary fits (tau2_cluster = 0, tau2_field = 0) as explicit "
             "candidates; the highest REML wins.")
    L.append("- **CIs.** Profile REML likelihood, 95% cut 2*delta <= 3.84 (chi2_1; conservative at a "
             "boundary). ICC_true is profiled by writing (tau2_c, tau2_f) = (r T, (1-r) T) and maximizing over T "
             "for each r.")
    L.append("- **Tests of the cluster component.** LR_c = 2[l_REML(full) - l_REML(tau2_c = 0)], same X in both. "
             "Null distributions: (i) 0.5 chi2_0 + 0.5 chi2_1 (Self-Liang, asymptotic); (ii) size-preserving label "
             "permutations (the CIP-2 label vector is permuted across fields, so the multiset of cluster sizes, "
             "including the singletons, is unchanged; the full model is refit each time; V_e and the moderators "
             "stay attached to the fields, so with moderators the test is conditional on them); (iii) parametric "
             "bootstrap from the fitted field-only model (with moderators: from the fitted moderator + field-only "
             "model; for correlated specs the simulated errors carry the full V_e); (iv) label permutation within "
             f"{N_STRAT} rank strata of n (ties broken by the alphabetical field order), a model-free control for "
             "field size. The unadjusted and moderator fits use the same permutations.")
    L.append("- **Moderators.** From the rebuilt matched tables: log n; coverage n / n_Wapman (n_Wapman = "
             "institutions in the field's Wapman list with a prestige score); prestige range = SD of the matched "
             "institutions' prestige percentile (rank / n_Wapman) within the field's list; log of the median "
             "Scorecard earnings-cohort count over matched institutions. The within-cluster slope comes from z ~ "
             "C(CIP2) + log n + field RE (REML, tau2_cluster not in the model).")
    L.append("- **Heterogeneity.** Cochran Q = sum w_f (z_f - z_bar_w)^2, w = 1/s^2, df = k-1 (generalized "
             "(z - z_bar)'V_e^-1(z - z_bar) for full V_e; with moderators the residual Q_E on k-p df); "
             "I^2 = (Q-df)/Q. Multilevel split (Cheung 2014): I^2_cluster = tau2_c/(tau2_c + tau2_f + s~^2), "
             "I^2_field likewise, s~^2 = Higgins-Thompson typical sampling variance.")
    L.append("- **Secondary.** Moderator model z ~ C(CIP2) + field RE (REML residual tau2; Wald Q_M on C-1 df; "
             "pseudo-R^2 = (tau2_field-only - tau2_resid)/tau2_field-only). Legacy ICC recomputed with the "
             "scripts/14/24 formulas on the gap scale with SE floored at 0.08. Leave-one-cluster-out: each of "
             f"the {a['C']} clusters dropped in turn; leave-one-field-out: each of the {a['k']} fields.")
    L.append("\n### Estimator verification (all from this run)\n")
    ok = lambda x, t: ("< %g (pass)" % t) if x < t else ("= %.2e (FAIL)" % x)
    L.append(f"- V1 REML implementations agree with a textbook explicit-inverse REML on the 57-field data over 10 "
             f"parameter/label settings: Woodbury {ok(ver['V1_woodbury_vs_textbook_maxabs'], 1e-10)}; Cholesky "
             f"with diagonal V_e vs Woodbury {ok(ver['V1_cholesky_diag_vs_woodbury_maxabs'], 1e-10)}; Cholesky "
             f"with correlated V_e vs textbook {ok(ver['V1_cholesky_corr_vs_textbook_maxabs'], 1e-10)}. With "
             f"moderators X = [1, log n, coverage]: Woodbury vs textbook {ok(ver['V1_X_woodbury_vs_textbook_maxabs'], 1e-10)}; "
             f"vectorized grid vs pointwise {ok(ver['V1_X_grid_vs_pointwise_maxabs'], 1e-10)}; Cholesky with "
             f"correlated V_e vs textbook {ok(ver['V1_X_cholesky_corr_vs_textbook_maxabs'], 1e-10)}.")
    L.append(f"- V2 two-level special case (tau2_cluster = 0): optimizer tau2 vs the classic REML fixed-point "
             f"iteration, 6 sample x spec cases: max |diff| {ok(ver['V2_twolevel_vs_fixedpoint_maxabs'], 1e-8)}. "
             f"With moderators (log n; log n + coverage), optimizer tau2 vs Fisher scoring on the REML score "
             f"equation, 12 cases: max |diff| {ok(ver['V2_X_twolevel_vs_scoring_maxabs'], 1e-8)}.")
    L.append(f"- V3 legacy reproduction: scripts/24 `reml_tau2` gives tau2 = {ver['V3_legacy_tau2_original']:.4f}; "
             f"this script's version {ver['V3_legacy_tau2_fast']:.4f}; legacy raw / corrected ICC = "
             f"{ver['V3_legacy_icc_raw']:.2f} / {ver['V3_legacy_icc_corr']:.2f} (CROSSWALK_AUDIT_RESULT.md reports "
             "0.0207 and 0.32 / 0.57).")
    L.append(f"- V4 dense 241x241 grid over tau2_c in [0, 0.12], tau2_f in [0, 0.06] (57 fields, 1/(n-3)): grid "
             f"max REML {ver['V4_grid_max_ll']:.5f}, fitted {ver['V4_fit_ll']:.5f} (fit >= grid: "
             f"{ver['V4_fit_ll'] >= ver['V4_grid_max_ll'] - 1e-9}).")
    L.append(f"- V5 recovery on a large synthetic design (150 clusters x 4 fields, sampling variances resampled "
             f"from the data, 200 draws): true (tau2_c, tau2_f) = ({ver['V5_true_tc']}, {ver['V5_true_tf']}); "
             f"mean estimates ({ver['V5_mean_tc']:.4f}, {ver['V5_mean_tf']:.4f}), SD ({ver['V5_sd_tc']:.4f}, "
             f"{ver['V5_sd_tf']:.4f}).")
    L.append(f"- Moderator values: the rebuilt tables reproduce every field's n and rho in the gap map (asserted); "
             f"no field is missing an earnings-cohort count ({int(dd.n_cohort_missing.sum())} missing institution "
             "counts in total).")
    L.append("\n## Caveats\n")
    L.append(_caveats_text(a, b, ec, A, SZ))
    OUT_MD.write_text("\n".join(L) + "\n")


def _loo_phrase(loo):
    parts = []
    for col, lab in (("p_perm_fisher_adj", "independent errors, permutation p"),
                     ("p_asym_fisher_corr_adj", "correlated errors, mixture p")):
        bad = loo[loo[col] > 0.05].sort_values(col, ascending=False, kind="stable")
        parts.append(f"{lab} > 0.05 after dropping {len(bad)} of {len(loo)} clusters"
                     + (" (" + ", ".join(f"{nm} {fmt_p(p_)}" for nm, p_ in zip(bad["name"], bad[col])) + ")" if len(bad) else ""))
    return "; ".join(parts)


def _within_phrase(o):
    p_w = 2 * norm.sf(abs(o["slope_within"] / o["slope_within_se"]))
    if p_w < 0.05:
        return ("Field size tracks coupling among fields of the same discipline as well, so it is not only a "
                "proxy for discipline.")
    return (f"Inside disciplines the gradient is weaker and not distinguishable from zero (p = {fmt_p(p_w)}), so "
            "field size is largely collinear with discipline here; it may also act within disciplines, but these "
            "data cannot show it.")


def _lofo_text(lofo):
    t = []
    for sfx, lab in (("", "unadjusted"), ("_adj", "+ log n")):
        for v, vl in ((MAIN, "1/(n-3)"), (CORR, "corr.")):
            col = f"p_asym_{v}{sfx}"; bad = lofo[lofo[col] > 0.05].sort_values(col, ascending=False, kind="stable")
            t.append(f"{lab}, {vl}: {len(bad)} of {len(lofo)} drops give mixture p > 0.05 (max "
                     f"{fmt_p(lofo[col].max())}, after dropping {lofo.loc[lofo[col].idxmax(), 'field']})"
                     + (f"; these: {', '.join(f'{f} ({fmt_p(p)})' for f, p in zip(bad.field, bad[col]))}" if len(bad) else "")
                     + ".")
    return "Leave one field out (57 fields, mixture p for LR_c): " + " ".join(t)


def _answer_text(R, A, SZ, a, b, ac, bc, La, loo, lofo, ec):
    """Plain-language verdict, generated from the numbers so re-runs stay consistent."""
    sig = lambda r: r["p_perm_lr"] < 0.05 and r["p_pboot_lr"] < 0.05
    aa, aac = A[("nondegen57", MAIN, PRIMARY)], A[("nondegen57", CORR, PRIMARY)]
    ba, bac = A[("reliable20", MAIN, PRIMARY)], A[("reliable20", CORR, PRIMARY)]
    s57, s20 = SZ["nondegen57"], SZ["reliable20"]
    adj_p = {("1/(n-3)", "mixture"): aa["p_lr_asym"], ("1/(n-3)", "permutation, n kept"): aa["p_perm_lr"],
             ("1/(n-3)", "parametric bootstrap"): aa["p_pboot_lr"], ("1/(n-3)", "within n tertiles"): aa["p_strat_lr"],
             ("corr.", "mixture"): aac["p_lr_asym"], ("corr.", "permutation, n kept"): aac["p_perm_lr"],
             ("corr.", "parametric bootstrap"): aac["p_pboot_lr"], ("corr.", "within n tertiles"): aac["p_strat_lr"]}
    pmin, pmax = min(adj_p.values()), max(adj_p.values())
    n_adj_over = sum(v >= 0.05 for v in adj_p.values())
    unadj = sig(a) and sig(ac)
    tc_zero = aa["tc_ci"][0] < 1e-6 or aac["tc_ci"][0] < 1e-6
    loo_bad = loo[(loo.p_perm_fisher_adj > 0.05) | (loo.p_asym_fisher_corr_adj > 0.05)]
    lofo_bad = int((lofo.p_asym_fisher_adj > 0.05).sum())
    assert int(((lofo.p_asym_fisher > 0.05) | (lofo.p_asym_fisher_corr > 0.05)).sum()) == 0, "update LOFO wording"
    others = {(k, v): A[("nondegen57", v, k)]["p_perm_lr"] for k in ADJ_SETS for v in (MAIN, CORR)}
    oth_over = [(k, v, p_) for (k, v), p_ in others.items() if p_ >= 0.05]
    removed = 1 - aa["tc"] / a["tc"]
    robust_adj = pmax < 0.01 and not tc_zero and len(loo_bad) == 0 and lofo_bad == 0
    t = []
    if not unadj:
        t.append("**No: on the 57 non-degenerate fields the discipline (CIP-2) component is not distinguishable "
                 "from chance and field-level heterogeneity even before field size is considered. The paper should "
                 "demote it to a descriptive map.**\n")
    elif robust_adj:
        t.append("**Yes, on the 57 non-degenerate fields, and it survives adjustment for field size. Its share is "
                 "not pinned down.**\n")
    elif n_adj_over == 0:
        extra = []
        if tc_zero:
            extra.append("its profile confidence interval reaches 0")
        if len(loo_bad) or lofo_bad:
            extra.append("it fails some leave-one-out checks")
        t.append("**Borderline. The CIP-2 component is detected in the unadjusted three-level model, but adding "
                 f"field size (log n) as a covariate cuts its variance by about {removed:.0%}. Net of field size it clears "
                 f"5% only narrowly (p from {fmt_p(pmin)} to {fmt_p(pmax)} across four nulls, with independent or "
                 "correlated errors)"
                 + ("".join(f", {e}" for e in extra[:-1]) + (f", and {extra[-1]}" if extra else "")) +
                 ". The paper should present discipline structure as a descriptive map with a borderline variance "
                 "component, and name field size as a competing explanation. It should not present it as an "
                 "established majority share.**\n")
    elif n_adj_over < len(adj_p):
        t.append("**Borderline. The CIP-2 component is detected in the unadjusted three-level model, but adding "
                 f"field size (log n) as a covariate cuts its variance by about {removed:.0%}, and net of field size it is "
                 f"marginal (p from {fmt_p(pmin)} to {fmt_p(pmax)}; {n_adj_over} of {len(adj_p)} null/spec "
                 "combinations are at or above 0.05). The paper should present discipline structure as a descriptive "
                 "map with a borderline variance component, not as an established variance share, and name field "
                 "size as a competing explanation.**\n")
    else:
        t.append("**Not once field size is controlled. The CIP-2 component is detected in the unadjusted "
                 f"three-level model, but with log n as a moderator p >= 0.05 under every null "
                 f"(p from {fmt_p(pmin)} to {fmt_p(pmax)}). The paper should demote discipline structure to a "
                 "descriptive map and name field size as a competing explanation.**\n")

    t.append(f"**Unadjusted.** In a three-level model that includes field-level heterogeneity, the CIP-2 variance "
             f"beats random partitions of the same sizes and a field-only model: 57 fields, LR_c = {a['lr']:.2f}, "
             f"permutation p = {fmt_p(a['p_perm_lr'])}, parametric-bootstrap p = {fmt_p(a['p_pboot_lr'])} "
             f"(1/(n-3)); correlated errors p = {fmt_p(ac['p_perm_lr'])} / {fmt_p(ac['p_pboot_lr'])}. ICC_true = "
             f"{a['icc']:.2f} (95% profile CI {a['icc_ci'][0]:.2f} to {a['icc_ci'][1]:.2f}).")
    t.append(f"\n**Field size.** Coupling rises with n, the number of institutions that have both prestige and "
             f"earnings for the field: Spearman(z, n) = {s57['spearman_z_n']:.2f} (p = {fmt_p(s57['spearman_p'])}) "
             f"on 57 fields, {s20['spearman_z_n']:.2f} on the reliable 20. Clusters differ in n (eta^2 of log n on "
             f"CIP-2 = {s57['eta2_logn_cip2']:.2f}; with {s57['C']} clusters for {s57['k']} fields a one-way F test "
             f"gives p = {fmt_p(s57['eta2_F_p'])}). The label permutation breaks the link between labels and n, "
             f"so it counts a size gradient as discipline. With log n as a fixed moderator (57 fields, 1/(n-3)): "
             f"tau2_cluster {a['tc']:.4f} -> {aa['tc']:.4f} [{aa['tc_ci'][0]:.4f}, {aa['tc_ci'][1]:.4f}], "
             f"ICC_true {a['icc']:.2f} -> {aa['icc']:.2f} [{aa['icc_ci'][0]:.2f}, {aa['icc_ci'][1]:.2f}], LR_c "
             f"{a['lr']:.2f} -> {aa['lr']:.2f}: mixture p = {fmt_p(aa['p_lr_asym'])}, permutation p with n kept "
             f"attached = {fmt_p(aa['p_perm_lr'])} ({aa['n_perm']} perms), parametric-bootstrap p = "
             f"{fmt_p(aa['p_pboot_lr'])}, permutation within n tertiles = {fmt_p(aa['p_strat_lr'])}. Correlated "
             f"errors: LR_c {ac['lr']:.2f} -> {aac['lr']:.2f}, mixture p = {fmt_p(aac['p_lr_asym'])}, permutation "
             f"p = {fmt_p(aac['p_perm_lr'])}, parametric-bootstrap p = {fmt_p(aac['p_pboot_lr'])}, ICC_true "
             f"{aac['icc']:.2f} [{aac['icc_ci'][0]:.2f}, {aac['icc_ci'][1]:.2f}]. A model-free control, permuting "
             f"labels only within n tertiles without the covariate, gives p = "
             f"{fmt_p(A[('nondegen57', MAIN, 'none')]['p_strat_lr'])} for the unadjusted LR; the covariate "
             "adjustment is the stricter of the two.")
    ev = lambda v: "1/(n-3)" if v == MAIN else "corr."
    no_n = [k for k, (cols, _) in ADJ_SETS.items() if not any(c in ("log_n", "log_n_sq", "cov_frac") for c in cols)]
    t.append(f"\n**Robustness of the adjusted result.** Across the {len(ADJ_SETS)} moderator sets, each with "
             f"independent and correlated errors (57 fields), the permutation p runs from {fmt_p(min(others.values()))} "
             f"to {fmt_p(max(others.values()))}; {len(oth_over)} of {len(others)} are at or above 0.05"
             + (": " + ", ".join(f"{ADJ_SETS[k][1]} ({ev(v)}) {fmt_p(p_)}" for k, v, p_ in oth_over) if oth_over else "")
             + ". Across the five sampling-variance specs with log n, the permutation p runs from "
             + f"{fmt_p(min(A[('nondegen57', v, PRIMARY)]['p_perm_lr'] for v in VAR_SPECS))} to "
             + f"{fmt_p(max(A[('nondegen57', v, PRIMARY)]['p_perm_lr'] for v in VAR_SPECS))}, and the tau2_cluster "
             + f"profile CI reaches 0 in {sum(A[('nondegen57', v, PRIMARY)]['tc_ci'][0] < 1e-6 for v in VAR_SPECS)} of "
             + f"{len(VAR_SPECS)}"
             + ("".join(f"; under {v} its lower bound is {A[('nondegen57', v, PRIMARY)]['tc_ci'][0]:.4f}"
                        for v in VAR_SPECS if A[('nondegen57', v, PRIMARY)]['tc_ci'][0] >= 1e-6))
             + ". Moderators that do not involve n leave LR_c close to its unadjusted value (1/(n-3): "
             + ", ".join(f"{ADJ_SETS[k][1]} {A[('nondegen57', MAIN, k)]['lr']:.2f}" for k in no_n)
             + f", against {a['lr']:.2f} unadjusted)"
             + (", so the reduction comes from field size and coverage, not from prestige range or earnings-cohort size"
                if all(A[("nondegen57", MAIN, k)]["lr"] >= 0.8 * a["lr"] for k in no_n) else "")
             + ". Leave one cluster out with log n: "
             + _loo_phrase(loo)
             + f". Leave one field out with log n (mixture p): {lofo_bad} of {len(lofo)} drops give p > 0.05 with "
             f"independent errors (max {fmt_p(lofo.p_asym_fisher_adj.max())}) and "
             f"{int((lofo.p_asym_fisher_corr_adj > 0.05).sum())} of {len(lofo)} with correlated errors (max "
             f"{fmt_p(lofo.p_asym_fisher_corr_adj.max())}), against 0 of {len(lofo)} without the adjustment "
             f"(max {fmt_p(max(lofo.p_asym_fisher.max(), lofo.p_asym_fisher_corr.max()))}). The within-cluster slope of log n is "
             f"{s57['slope_within']:.3f} (SE {s57['slope_within_se']:.3f}), about "
             f"{s57['slope_within'] / s57['slope_overall']:.2f} of the overall {s57['slope_overall']:.3f}. "
             + _within_phrase(s57))
    t.append(f"\n**Reliable 20** ({b['C']} clusters, {b['n_single']} singletons, {b['fields_in_multi']} fields in "
             f"{b['n_multi']} multi-field clusters). "
             + ("The component holds up better here" if ba["p_perm_lr"] < aa["p_perm_lr"] else "The component does not hold up better here")
             + f": with log n, LR_c "
             f"{b['lr']:.2f} -> {ba['lr']:.2f}, mixture p = {fmt_p(ba['p_lr_asym'])}, permutation p = "
             f"{fmt_p(ba['p_perm_lr'])}, parametric-bootstrap p = {fmt_p(ba['p_pboot_lr'])}; correlated errors "
             f"permutation p = {fmt_p(bac['p_perm_lr'])}. But tau2_field is estimated at {ba['tf']:.4f}, so "
             f"ICC_true = {ba['icc']:.2f} [{ba['icc_ci'][0]:.2f}, {ba['icc_ci'][1]:.2f}] is weakly identified: "
             f"power of the unadjusted cluster test at a true ICC of 0.50 is "
             f"{[p['power'] for p in b['power'] if p['scenario'] == 'ICC=0.50'][0]:.2f} here, against "
             f"{[p['power'] for p in a['power'] if p['scenario'] == 'ICC=0.50'][0]:.2f} on 57 fields.")
    t.append(f"\n**Confounder or mediator.** The data cannot say which. If small n is a measurement artifact "
             "(range restriction, program-offering selection, noisier small samples), the unadjusted component "
             "is inflated and the adjusted one is the better estimate. If being widely offered is part of what "
             "a discipline is, adjusting removes real discipline signal. The paper should report both and not "
             "pick the favourable one.")
    t.append(f"\n**The reported statistic is not supported.** Recomputed on the same 57 fields, the legacy raw ICC "
             f"({La['obs']['icc_raw']:.3f}) and the 'measurement-corrected' ICC ({La['obs']['icc_corr']:.3f}) do "
             f"not clear a size-preserving permutation null at 5% (p = {fmt_p(La['p_raw'])} and "
             f"{fmt_p(La['p_corr'])}). The old two-level structure (no field term) explains why. On z, its cluster "
             f"LR is {a['lr2']:.1f} (asymptotic p = {fmt_p(a['p_lr2_asym'])}), but random partitions of the same "
             f"sizes reach {a['q95']['lr2']:.1f} at their 95th percentile. Without a field term, any field "
             "heterogeneity gets loaded onto whatever grouping is supplied. The paper's 0.30 / 0.45 come from the "
             "earlier 48-field set, which this script does not rerun; they use the same statistic.")
    order = np.argsort(a["mu"] + a["u"], kind="stable")
    nm = lambda c: C2NAME.get(a["cl"][c], a["cl"][c])
    low, high = ", ".join(nm(c) for c in order[:4]), ", ".join(nm(c) for c in order[::-1][:4])
    t.append(f"\n**What the paper can say.** (1) Coupling varies across fields well beyond sampling error "
             f"(Cochran Q = {a['het']['Q']:.1f} on {a['het']['df']} df, I^2 = {a['het']['I2']:.2f}). This replaces "
             f"the '15% inside / 65% above the null band' statement. Coupling also rises with the number of "
             f"institutions offering the field. (2) A CIP-2 component is detected without adjustment (p about "
             f"{fmt_p(a['p_perm_lr'])}). Net of field size it is borderline (p {fmt_p(pmin)} to {fmt_p(pmax)} on 57 "
             f"fields), and its share runs from {aa['icc_ci'][0]:.2f} to {aa['icc_ci'][1]:.2f}. No between-discipline "
             f"share should be quoted. (3) The cluster ordering can be shown as a shrunken descriptive map, with "
             f"mean n alongside. Lowest coupling: {low}. Highest: {high}. **What it should drop:** ICC 0.30 / 0.45 / "
             "0.57, the eta^2 = 0.50 variance-share reading, 'structured by discipline' as an established finding, "
             "and any statement that discipline carries most of the between-field variance.")
    return "\n".join(t)


def _caveats_text(a, b, ec, A, SZ):
    c = []
    c.append(f"- **Few clusters, many singletons.** 57 fields: {a['n_single']} of {a['C']} clusters are singletons "
             f"and Engineering (CIP 14) alone holds {int(a['sizes'].max())} fields. Singletons inform only "
             "tau2_cluster + tau2_field jointly; the split comes from the multi-field clusters. That is why the "
             "ICC_true intervals are wide and the power simulation shows modest power for ICC_true near 0.25.")
    s57 = SZ["nondegen57"]; p_w = 2 * norm.sf(abs(s57["slope_within"] / s57["slope_within_se"]))
    c.append("- **Field size is a competing explanation, not a settled nuisance.** n is partly collinear with "
             f"discipline (eta^2 of log n on CIP-2 = {s57['eta2_logn_cip2']:.2f}); within clusters its slope is "
             f"{s57['slope_within']:.3f} (SE {s57['slope_within_se']:.3f}, p = {fmt_p(p_w)}), "
             + ("so it also acts inside disciplines. " if p_w < 0.05 else "so whether it also acts inside disciplines "
                "is not resolved. ")
             + "Adjusting for it is the conservative reading; the unadjusted model is the generous one. Neither is "
             "a causal decomposition, and log n is one functional form (the quadratic and coverage variants are in "
             "the moderator table).")
    c.append("- **Adjusted ICC_true is a residual share.** With moderators, tau2_cluster and tau2_field are what "
             "is left after the fixed effects, so the adjusted ICC_true is not comparable one-to-one with the "
             "unadjusted one.")
    c.append("- **Sampling variance is approximate.** 1/(n-3) is the Pearson Fisher-z variance; for Spearman it is "
             "optimistic (Bonett-Wright inflates it by 1 + rho^2/2). All specs treat institutions as the sampling "
             "unit. Noise not captured by institution resampling (small-cohort earnings medians, SpringRank "
             "estimation error) attenuates rho by field-specific amounts, and the model counts it as true "
             "heterogeneity. The earnings-cohort-size moderator tests one version of this: with log median "
             f"cohort size in the model, LR_c = {A[('nondegen57', MAIN, 'log_cohort')]['lr']:.2f} against "
             f"{a['lr']:.2f} unadjusted (57 fields, 1/(n-3)).")
    c.append(f"- **Correlated errors are estimated, not known.** The joint institution bootstrap gives a mean "
             f"within-cluster error correlation of {ec['within']:.2f}. It resamples from the union of institutions, "
             "so each field's resampled n varies across draws. Draws in which any field had fewer than 3 distinct "
             "values were dropped (complete cases). The correlated-error specs treat this estimated matrix as "
             "known.")
    c.append("- **What the permutation null tests.** Whether the actual CIP-2 partition groups similar fields more "
             "than a random partition with the same sizes. It conditions on the observed z values (and, with "
             "moderators, on them too); it does not test whether CIP-2 is the right grouping. With correlated "
             "errors the parametric bootstrap (which simulates those correlations) is the cleaner null. The n-tertile "
             "permutation controls n only coarsely (three strata).")
    c.append("- **ICC_true as a test statistic** is weak: it equals 1 whenever tau2_field is estimated at 0, which "
             "happens often in permutations of small designs. The LR statistic is the one to read for "
             "significance; ICC_true and its profile interval are for size.")
    c.append("- **Scope.** The coupling estimates come from `outputs/expanded66_gap_map.csv` as given. The 57-field "
             "set includes fields the reliability screen rejects; the reliable-20 run is reported alongside.")
    c.append("- **What Cochran Q replaces.** The old '15% inside / 65% above the null band' tested each field against "
             "a perfect-agreement placebo. Q / I^2 test a different, standard question: do fields' couplings "
             "differ by more than sampling error? They do. That is necessary for a discipline structure but not "
             "sufficient; tau2_cluster and ICC_true address the structure itself.")
    return "\n".join(c)


# ============================================================ main ======================
def main():
    d, meta = load_fields()
    tables, cov = rebuild_fields(d)
    assert list(cov.field) == list(d.field)
    d = pd.concat([d, cov.drop(columns="field")], axis=1)
    samples = {"nondegen57": d, "reliable20": d[d.reliable].reset_index(drop=True)}
    assert len(samples["nondegen57"]) == 57 and len(samples["reliable20"]) == 20
    ss = np.random.SeedSequence(SEED)
    kid = dict(zip(["jb", "perm57", "perm20", "pb", "pw", "loo"], ss.spawn(6)))
    kid2 = dict(zip(["strat57", "strat20", "pb_adj"], ss.spawn(3)))      # added streams (fix round)
    JB = joint_bootstrap(d, np.random.default_rng(kid["jb"]), tables)
    ec = error_corr_summary(d, JB)
    print(f"joint bootstrap: {JB['n_ok']}/{B_JOINT} usable, univ={JB['n_univ']}, "
          f"within-cluster corr {ec['within']:.3f}, between {ec['between']:.3f}")
    pb_seeds = iter(kid["pb"].spawn(2 * len(VAR_SPECS))); pw_seeds = iter(kid["pw"].spawn(2))
    R, legacy, PERMS = {}, {}, {}
    for sname, sd in samples.items():
        prng = np.random.default_rng(kid["perm57" if sname == "nondegen57" else "perm20"])
        PERMS[sname] = perms = [prng.permutation(len(sd)) for _ in range(N_PERM)]
        for spec in VAR_SPECS:
            n_perm = N_PERM if spec == MAIN else N_PERM_CHECK
            R[(sname, spec)] = r = analyse(
                sd, sname, spec, JB, perms[:n_perm], np.random.default_rng(next(pb_seeds)),
                np.random.default_rng(next(pw_seeds)) if spec == MAIN else None,
                N_PBOOT_DENSE if spec in DENSE_SPECS else N_PBOOT, do_power=(spec == MAIN))
            print(f"{sname:11s} {spec:14s} k={r['k']} C={r['C']} single={r['n_single']} "
                  f"I2={r['het']['I2']:.3f} tc={r['tc']:.5f} tf={r['tf']:.5f} ICC={r['icc']:.3f} "
                  f"[{r['icc_ci'][0]:.3f},{r['icc_ci'][1]:.3f}] LR={r['lr']:.3f} "
                  f"p_perm={r['p_perm_lr']:.4f} p_pboot={r['p_pboot_lr']:.4f} p_asym={r['p_lr_asym']:.4f}", flush=True)
        cl, g = np.unique(sd.cip2.to_numpy(), return_inverse=True); C = len(cl)
        gap, se = sd.gap.to_numpy(), sd.se.to_numpy()
        obs = legacy_icc(gap, se, g, C)
        nulls = pmap(lambda q: legacy_icc(gap, se, g[q], C), perms)
        nr = np.array([x["icc_raw"] for x in nulls]); nc = np.array([x["icc_corr"] for x in nulls])
        legacy[sname] = dict(obs=obs, null_raw=nr, null_corr=nc,
                             q95_raw=float(np.quantile(nr, .95)), q95_corr=float(np.quantile(nc, .95)),
                             p_raw=perm_p(nr, obs["icc_raw"]), p_corr=perm_p(nc, obs["icc_corr"]))
        print(f"{sname:11s} legacy ICC raw={obs['icc_raw']:.3f} (q95 {legacy[sname]['q95_raw']:.3f}, "
              f"p={legacy[sname]['p_raw']:.4f}) corr={obs['icc_corr']:.3f} (p={legacy[sname]['p_corr']:.4f})")

    # ---- field-size moderator arm (fix round) ----
    A = {}
    pba = iter(kid2["pb_adj"].spawn(2 * len(VAR_SPECS)))
    for sname, sd in samples.items():
        sp = strat_perms(sd.n.to_numpy(), N_PERM_CHECK,
                         np.random.default_rng(kid2["strat57" if sname == "nondegen57" else "strat20"]))
        perms = PERMS[sname]
        for spec in (MAIN, CORR):      # unadjusted model, n-stratified permutations
            A[(sname, spec, "none")] = a_ = analyse_adj(sd, sname, spec, JB, "none", None, sp, None, 0)
            print(f"{sname:11s} {spec:14s} none       LR={a_['lr']:.3f} p_strat={a_['p_strat_lr']:.4f}", flush=True)
        for spec in VAR_SPECS:         # primary adjustment, every variance spec
            n_p = N_PERM if (sname == "nondegen57" and spec == MAIN) else N_PERM_CHECK
            A[(sname, spec, PRIMARY)] = a_ = analyse_adj(
                sd, sname, spec, JB, PRIMARY, perms[:n_p], sp if spec in (MAIN, CORR) else None,
                np.random.default_rng(next(pba)), N_PBOOT_DENSE if spec in DENSE_SPECS else N_PBOOT)
            print(f"{sname:11s} {spec:14s} {PRIMARY:10s} tc={a_['tc']:.5f} tf={a_['tf']:.5f} ICC={a_['icc']:.3f} "
                  f"[{a_['icc_ci'][0]:.3f},{a_['icc_ci'][1]:.3f}] LR={a_['lr']:.3f} p_asym={a_['p_lr_asym']:.4f} "
                  f"p_perm={a_['p_perm_lr']:.4f} p_pboot={a_['p_pboot_lr']:.4f}"
                  + (f" p_strat={a_['p_strat_lr']:.4f}" if "p_strat_lr" in a_ else ""), flush=True)
        for adj in ADJ_SETS:           # other measurement moderators
            if adj == PRIMARY:
                continue
            for spec in (MAIN, CORR):
                A[(sname, spec, adj)] = a_ = analyse_adj(sd, sname, spec, JB, adj, perms[:N_PERM_CHECK], None, None, 0)
                print(f"{sname:11s} {spec:14s} {adj:16s} tc={a_['tc']:.5f} ICC={a_['icc']:.3f} LR={a_['lr']:.3f} "
                      f"p_asym={a_['p_lr_asym']:.4f} p_perm={a_['p_perm_lr']:.4f}", flush=True)
    SZ = size_diagnostics(samples, JB)
    for k_, v_ in SZ.items():
        print(k_, {kk: vv for kk, vv in v_.items() if kk != "cluster_mean_n"})
    loo = leave_one_cluster_out(d, JB, np.random.default_rng(kid["loo"]))
    print(loo.to_string())
    lofo = leave_one_field_out(d, JB)
    print(lofo.sort_values("p_asym_fisher_adj").tail(8).to_string())
    ver = verification(d, JB)
    for k_, v_ in ver.items():
        print(f"  {k_}: {v_}")
    build_table(R, legacy, ver, loo, ec, JB, d, A, SZ, lofo)
    make_figure(R, legacy["nondegen57"], loo, A, d)
    write_report(R, legacy, ver, meta, loo, ec, JB, d, A, SZ, lofo)
    print("wrote", OUT_CSV, OUT_FIG, OUT_MD)


if __name__ == "__main__":
    main()
