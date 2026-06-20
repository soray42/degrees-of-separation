"""Diagnostic — is the AR–ER gap real signal or a low-SNR measurement artifact?

Gap = 1 − Spearman(prestige_rank, earnings_rank). In a field where cross-institution
earnings have little true spread and/or small cohorts, the earnings ranking is
sampling-noise-dominated, so its rank-correlation with prestige is mechanically
attenuated toward zero ⇒ a high gap BY CONSTRUCTION, even under perfect market
agreement. These functions quantify how much of the observed gap that channel explains.

All inputs are already in the repo (Wapman ranks + Scorecard FoS, incl. the earnings
cohort-count column EARN_COUNT_WNE_4YR). No new data, no pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm

from . import tier0 as T
from .crosswalks import fields as F
from .crosswalks.institutions import normalize_institution_name

MED_SE_CONST = 1.2533  # SE(sample median) ≈ 1.2533 · σ / sqrt(n) for ~normal data


# ---------------------------------------------------------------------------
# Matched per-field data WITH cohort sizes (for sampling-noise calibration)
# ---------------------------------------------------------------------------
def load_matched(earn_col: str = "EARN_MDN_4YR", count_col: str = "EARN_COUNT_WNE_4YR",
                 credlev: str = "3") -> pd.DataFrame:
    """Per matched institution × field: prestige_score, earn, cohort_n.
    earn = cohort-weighted mean of the field's CIP-4 medians at the institution;
    cohort_n = summed earnings cohort over those CIP-4 rows."""
    prest = T.load_prestige()
    use = ["INSTNM", "CIPCODE", "CREDLEV", earn_col, count_col]
    df = pd.read_csv(T.SCORECARD_FOS, usecols=use, dtype=str)
    df = df[df["CREDLEV"] == credlev].copy()
    df["cip4"] = df["CIPCODE"].astype(str).str.zfill(4)
    df["field_key"] = df["cip4"].map(F.cip4_to_field_key())
    df = df[df["field_key"].notna()].copy()
    df["earn"] = pd.to_numeric(df[earn_col], errors="coerce")
    df["cohort"] = pd.to_numeric(df[count_col], errors="coerce")
    df = df.dropna(subset=["earn", "cohort"])
    df = df[df["cohort"] > 0]
    df["inst_norm"] = df["INSTNM"].map(normalize_institution_name)
    # cohort-weighted earnings + summed cohort per institution×field
    def wmean(g):
        return np.average(g["earn"], weights=g["cohort"])
    agg = (df.groupby(["field_key", "inst_norm"])
             .apply(lambda g: pd.Series({"earn": wmean(g), "cohort": g["cohort"].sum()}))
             .reset_index())
    m = prest.merge(agg, on=["field_key", "inst_norm"], how="inner")
    return m[["field_key", "inst_norm", "prestige_score", "earn", "cohort"]]


# ---------------------------------------------------------------------------
# Test 1 — null-model / placebo calibration (decisive)
# ---------------------------------------------------------------------------
def null_model_field(prest: np.ndarray, earn: np.ndarray, cohort: np.ndarray,
                     alpha: float = 0.6, B: int = 1000, rng=None) -> dict:
    """Perfect market agreement + realistic sampling noise → distribution of synth gap.

    signal_i  = monotone-in-prestige (normal-quantile of prestige rank), scaled to the
                de-noised between-institution earnings SD (so synth CV ≈ observed CV).
    noise_i   ~ N(0, SE_i), SE_i = 1.2533 · α · earn_i / sqrt(cohort_i)  (median SE;
                α = assumed within-institution individual-earnings CV).
    synth_gap = 1 − Spearman(prestige, signal + noise).  Perfect agreement by design;
    any gap it produces is pure attenuation. If it reproduces the OBSERVED gap, that
    field's gap is indistinguishable from a noise artifact.
    """
    rng = rng or np.random.default_rng(0)
    N = len(earn)
    obs_var = float(np.var(earn, ddof=1))
    se = MED_SE_CONST * alpha * earn / np.sqrt(cohort)
    noise_var = float(np.mean(se ** 2))
    signal_var = max(obs_var - noise_var, 0.02 * obs_var)   # floor: keep a sliver of signal
    signal_sd = np.sqrt(signal_var)
    rk = pd.Series(prest).rank(method="average").values     # higher prestige → higher rank
    signal = earn.mean() + signal_sd * norm.ppf((rk - 0.5) / N)
    gaps = np.empty(B)
    for b in range(B):
        synth = signal + rng.normal(0.0, se)
        rho, _ = spearmanr(prest, synth)
        gaps[b] = 1.0 - rho
    return dict(N=N, obs_earn_var=obs_var, noise_var=noise_var,
                signal_frac=signal_var / obs_var,
                noise_to_signal=np.sqrt(noise_var) / signal_sd,
                synth_gap_mean=float(gaps.mean()),
                synth_gap_lo=float(np.percentile(gaps, 2.5)),
                synth_gap_hi=float(np.percentile(gaps, 97.5)))


def null_model(matched: pd.DataFrame, observed_gap: pd.Series, alpha: float = 0.6,
               B: int = 1000, min_n: int = 8, seed: int = 12345) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k in F.field_keys():
        g = matched[matched["field_key"] == k]
        if len(g) < min_n:
            continue
        r = null_model_field(g["prestige_score"].values, g["earn"].values,
                             g["cohort"].values, alpha=alpha, B=B, rng=rng)
        r["field_key"] = k
        r["observed_gap"] = float(observed_gap.get(k, np.nan))
        r["covered"] = bool(r["synth_gap_lo"] <= r["observed_gap"] <= r["synth_gap_hi"])
        r["excess_gap"] = r["observed_gap"] - r["synth_gap_mean"]  # >0 ⇒ more than noise
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 2 — bootstrap reliability of the within-field Spearman
# ---------------------------------------------------------------------------
def bootstrap_field(prest: np.ndarray, earn: np.ndarray, B: int = 1000, rng=None) -> dict:
    rng = rng or np.random.default_rng(0)
    N = len(earn)
    obs, _ = spearmanr(prest, earn)
    bs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, N, N)
        if len(np.unique(prest[idx])) < 3 or len(np.unique(earn[idx])) < 3:
            bs[b] = np.nan; continue
        bs[b], _ = spearmanr(prest[idx], earn[idx])
    bs = bs[~np.isnan(bs)]
    return dict(spearman=float(obs), boot_lo=float(np.percentile(bs, 2.5)),
                boot_hi=float(np.percentile(bs, 97.5)), boot_se=float(bs.std(ddof=1)))


def bootstrap(matched: pd.DataFrame, B: int = 1000, min_n: int = 8, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k in F.field_keys():
        g = matched[matched["field_key"] == k]
        if len(g) < min_n:
            continue
        r = bootstrap_field(g["prestige_score"].values, g["earn"].values, B=B, rng=rng)
        r["field_key"] = k; r["N"] = len(g); r["gap"] = 1 - r["spearman"]
        rows.append(r)
    return pd.DataFrame(rows)
