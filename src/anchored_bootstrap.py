"""Published-rank-ANCHORED generated-regressor bootstrap (the recalibrated estimator).

Why: the public Wapman edge lists only let any SpringRank (our from-scratch OR the canonical
cdebacco/LarremoreLab package) reproduce the published ranks at Spearman ≈ 0.80 — a DATA
ceiling (the public aggregated edges are lossy vs the proprietary AARC census), not a code
bug. Re-estimating ranks each draw therefore (a) is inconsistent with the published-rank
point estimate and (b) injects excess implementation/data-mismatch error that ATTENUATES P1.

The deposit ships no rank SE/CI/replicates (Task 0a), but it DOES ship the per-institution
continuous SpringRank score (institution-stats `PrestigeRank`) and the integer ranks. So we
ANCHOR at the published ranks and only inject a data-driven sampling SPREAD:
  * per matched institution, estimate the edge-sampling SD of its (percentile) rank by
    multinomially resampling the placement edges and re-running the CANONICAL SpringRank
    (this measures how volatile a rank is under finite placements — the generated-regressor
    uncertainty), computed once;
  * each bootstrap draw perturbs the PUBLISHED percentile rank by N(0, that SD), re-ranks,
    adds earnings cohort-SE and ACS-dispersion bootstrap-SE, recomputes gap and the cross-
    field P1 Spearman.
Center = published (consistent point estimate, no attenuation); spread = edge-sampling
uncertainty. This is the Option-1 path (perturb scores, re-rank) with a data-derived SD
since none is reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import springrank

from .crosswalks.institutions import normalize_institution_name

MED_SE = 1.2533
SR_ALPHA = 1.0   # recovered best-reproduction setting (Task 1 sweep)


def _canonical_scores(src, dst, counts, n):
    A = np.zeros((n, n))
    np.add.at(A, (src, dst), counts)
    m = springrank.SpringRank(alpha=SR_ALPHA, inverse_temp_fit_warning=False)
    m.fit(A)
    return np.asarray(m.ranks, float)


def precompute_field_anchored(edges_field, matched, prestige_score, disp, disp_se,
                              alpha_earn=0.6, B_se=200, seed=0):
    """Build the anchored per-field structure.
    matched: DataFrame[inst_key, earn, cohort]; prestige_score: dict inst_key->(-Rank).
    Returns None if too few matched institutions or edges."""
    e = edges_field[["DegreeInstitutionName", "InstitutionName", "Total"]].dropna()
    e = e[e.DegreeInstitutionName != e.InstitutionName]
    if len(e) < 20 or len(matched) < 4:
        return None
    nodes = sorted(set(e.DegreeInstitutionName) | set(e.InstitutionName))
    idx = {nm: i for i, nm in enumerate(nodes)}
    src = e.DegreeInstitutionName.map(idx).to_numpy()
    dst = e.InstitutionName.map(idx).to_numpy()
    counts = pd.to_numeric(e.Total, errors="coerce").fillna(0).to_numpy(float)
    node_key = np.array([normalize_institution_name(nm) for nm in nodes])
    mk = matched.set_index("inst_key")
    keep = [k for k in mk.index if k in set(node_key) and k in prestige_score]
    if len(keep) < 4:
        return None
    pos = np.array([np.where(node_key == k)[0][0] for k in keep])
    nmatch = len(keep)
    # published percentile rank of prestige among matched (higher = more prestigious)
    pub = np.array([prestige_score[k] for k in keep])
    pub_pct = rankdata(pub) / nmatch
    # edge-sampling SD of each matched institution's percentile rank (canonical SpringRank)
    rng = np.random.default_rng(seed)
    N = int(counts.sum()); p = counts / counts.sum()
    pcts = np.empty((B_se, nmatch))
    for b in range(B_se):
        nc = rng.multinomial(N, p).astype(float)
        s = _canonical_scores(src, dst, nc, len(nodes))
        pcts[b] = rankdata(s[pos]) / nmatch
    se_pct = pcts.std(axis=0, ddof=1)
    earn = mk.loc[keep, "earn"].to_numpy(float)
    cohort = np.clip(mk.loc[keep, "cohort"].fillna(np.nanmedian(mk["cohort"])).to_numpy(float), 5, None)
    earn_se = MED_SE * alpha_earn * earn / np.sqrt(cohort)
    return dict(pub_pct=pub_pct, se_pct=se_pct, earn=earn, earn_se=earn_se,
                disp=disp, disp_se=disp_se, n=nmatch)


def _draw_gap_anchored(fs, rng):
    pct = fs["pub_pct"] + rng.normal(0, fs["se_pct"])          # perturb published rank, re-rank
    prest = rankdata(pct)
    earn = fs["earn"] + rng.normal(0, fs["earn_se"])
    rho, _ = spearmanr(prest, earn)
    return 1.0 - rho


def bootstrap_p1_anchored(field_structs: dict, B: int = 1000, seed: int = 20260620,
                          resample_fields: bool = False):
    rng = np.random.default_rng(seed)
    keys = list(field_structs); k = len(keys)
    rhos = np.empty(B)
    for b in range(B):
        draw = list(rng.choice(keys, k, replace=True)) if resample_fields else keys
        gaps = np.array([_draw_gap_anchored(field_structs[key], rng) for key in draw])
        disp = np.array([field_structs[key]["disp"] + rng.normal(0, field_structs[key]["disp_se"]) for key in draw])
        rhos[b], _ = spearmanr(disp, gaps)
    rhos = rhos[np.isfinite(rhos)]
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    p = 2 * min((rhos >= 0).mean(), (rhos <= 0).mean())
    return dict(n_fields=k, B=len(rhos), point=float(np.median(rhos)), mean=float(rhos.mean()),
                ci_lo=float(lo), ci_hi=float(hi), p_boot=float(min(p, 1.0)))
