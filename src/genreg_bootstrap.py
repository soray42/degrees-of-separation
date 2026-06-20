"""Nested generated-regressor bootstrap for the P1 cross-field test (proposal §8).

Each gap_field carries noise from TWO sources, both propagated here:
  (1) earnings/dispersion sampling noise  — per-institution median SE from cohort counts
      (1.2533·α·earn/√cohort), plus a per-field ACS-dispersion bootstrap SE;
  (2) SpringRank rank-estimation noise (the generated regressor) — edges are
      multinomially resampled and SpringRank is RE-RUN per draw (the proper path; the
      Wapman Zenodo release ships the raw per-field edge lists, so we do not fall back to
      parametric rank perturbation).
Each draw recomputes every gap_field and dispersion_field and the cross-field
Spearman(dispersion, gap); the spread over draws is a CI/p that accounts for the generated
regressor and the small n. Adjacency is rebuilt from fixed (src,dst) index arrays per draw
for speed (no per-edge Python loop).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import spsolve

from .crosswalks.institutions import normalize_institution_name

MED_SE = 1.2533


def _springrank_from_counts(src, dst, counts, n, alpha=1e-3):
    A = csr_matrix((counts, (src, dst)), shape=(n, n), dtype=float)
    d_out = np.asarray(A.sum(axis=1)).flatten()
    d_in = np.asarray(A.sum(axis=0)).flatten()
    M = alpha * identity(n) + diags(d_out + d_in) - (A + A.T)
    s = spsolve(csr_matrix(M), d_out - d_in)
    return s - s.mean()


def precompute_field(edges_field, matched, disp, disp_se, alpha_earn=0.6):
    """Build the per-field structure reused across draws.
    edges_field: rows with DegreeInstitutionName, InstitutionName, Total.
    matched: DataFrame[inst_key, earn, cohort] over Wapman∩Scorecard institutions for the field.
    Returns a dict, or None if the field cannot support the test (too few matched/edges)."""
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
    # matched institutions present among the SpringRank nodes
    mk = matched.set_index("inst_key")
    keep = [k for k in mk.index if k in set(node_key)]
    if len(keep) < 4:
        return None
    pos = {k: np.where(node_key == k)[0][0] for k in keep}
    earn = mk.loc[keep, "earn"].to_numpy(float)
    cohort = np.clip(mk.loc[keep, "cohort"].fillna(np.nanmedian(mk["cohort"])).to_numpy(float), 5, None)
    se = MED_SE * alpha_earn * earn / np.sqrt(cohort)
    return dict(src=src, dst=dst, counts=counts, n=len(nodes),
                node_pos=np.array([pos[k] for k in keep]), earn=earn, se=se,
                disp=disp, disp_se=disp_se)


def _draw_field_gap(fs, rng):
    new_counts = rng.multinomial(int(fs["counts"].sum()), fs["counts"] / fs["counts"].sum()).astype(float)
    s = _springrank_from_counts(fs["src"], fs["dst"], new_counts, fs["n"])
    prest = s[fs["node_pos"]]                       # higher s = more prestigious
    earn = fs["earn"] + rng.normal(0, fs["se"])
    rho, _ = spearmanr(prest, earn)
    return 1.0 - rho                                # gap = 1 − Spearman(prestige, earnings)


def bootstrap_p1(field_structs: dict, B: int = 500, seed: int = 20260620,
                 resample_fields: bool = True):
    """Nested bootstrap of P1 = Spearman(dispersion, gap) across the given fields.

    `resample_fields=True` (default) is a TWO-LEVEL (cluster) bootstrap: the outer level
    resamples the field set with replacement (propagating the **small-n** field-sampling
    uncertainty) and the inner level redraws each field's edges (SpringRank re-run) and
    earnings/dispersion noise (the **generated-regressor + measurement** uncertainty). With
    `resample_fields=False` only the inner (generated-regressor) noise is propagated — useful
    to isolate whether SpringRank estimation noise alone overturns P1.
    """
    rng = np.random.default_rng(seed)
    keys = list(field_structs)
    k = len(keys)
    rhos = np.empty(B)
    for b in range(B):
        draw = list(rng.choice(keys, k, replace=True)) if resample_fields else keys
        gaps = np.array([_draw_field_gap(field_structs[key], rng) for key in draw])
        disp = np.array([field_structs[key]["disp"] + rng.normal(0, field_structs[key]["disp_se"]) for key in draw])
        rhos[b], _ = spearmanr(disp, gaps)
    rhos = rhos[np.isfinite(rhos)]
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    p = 2 * min((rhos >= 0).mean(), (rhos <= 0).mean())
    return dict(n_fields=k, B=len(rhos), point=float(np.median(rhos)),
                mean=float(rhos.mean()), ci_lo=float(lo), ci_hi=float(hi), p_boot=float(min(p, 1.0)))


def acs_dispersion_boot_se(workers_field, earn_col="PERNP", B=200, seed=7):
    """Bootstrap SE of a field's (p75−p25)/p50 dispersion by resampling ACS individuals
    (weighted by PWGTP via weighted resampling)."""
    rng = np.random.default_rng(seed)
    v = workers_field[earn_col].to_numpy(float)
    w = workers_field["PWGTP"].to_numpy(float)
    n = len(v)
    if n < 50:
        return np.nan
    p = w / w.sum()
    out = np.empty(B)
    for b in range(B):
        idx = rng.choice(n, n, replace=True, p=p)
        vv = v[idx]
        q25, q50, q75 = np.percentile(vv, [25, 50, 75])
        out[b] = (q75 - q25) / q50 if q50 else np.nan
    return float(np.nanstd(out))


def permutation_p(x, y, B=10000, seed=11):
    """Two-sided permutation p for Spearman(x, y)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    obs, _ = spearmanr(x, y)
    cnt = 0
    for _ in range(B):
        if abs(spearmanr(rng.permutation(x), y)[0]) >= abs(obs):
            cnt += 1
    return float(obs), (cnt + 1) / (B + 1)
