"""Theory-derived prediction tests. Tier 0 ships P1 only.

P1 (proposal v2 §4.3, H2): the gap is *smaller* where graduate productivity is more
observable to employers, proxied by within-field earnings **dispersion** (a market that
prices individual productivity spreads earnings out). **Predicted sign: negative.**

- Primary proxy  — PSEO within-institution cross-individual spread `(p75-p25)/p50`,
  aggregated to the field (median over institutions). This is the theory's notion.
- Robustness proxy — cross-institution earnings CV within field (Scorecard, full coverage;
  the variant the prior diagnostic found correct-signed at ≈ −0.41).

Tested across **reliability-passing fields only** (noise-dominated fields excluded), with the
included-unreliable variant reported for honesty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def field_dispersion_pseo(er_pseo: pd.DataFrame) -> pd.DataFrame:
    """Primary P1 proxy: median within-institution (p75-p25)/p50 per field (PSEO)."""
    g = (er_pseo.dropna(subset=["dispersion"]).groupby("field")
         .agg(disp_pseo=("dispersion", "median"), n_inst_pseo=("dispersion", "size"))
         .reset_index())
    return g


def field_dispersion_cv(er_scorecard: pd.DataFrame) -> pd.DataFrame:
    """Robustness P1 proxy: cross-institution earnings CV per field (Scorecard)."""
    g = (er_scorecard.groupby("field")
         .agg(earn_mean=("earnings", "mean"), earn_sd=("earnings", "std"),
              n_inst_sc=("earnings", "size")).reset_index())
    g["disp_cv"] = g["earn_sd"] / g["earn_mean"]
    return g[["field", "disp_cv", "n_inst_sc"]]


def load_nyfed_outcomes(path=None) -> pd.DataFrame:
    """NY Fed 'Labor Market for Recent College Graduates' by-major outcomes, mapped to the
    project fields (clean 1:1 majors only; folded majors -> NaN). Columns: field,
    unemployment, underemployment, wage_early, wage_mid, grad_degree_share."""
    from pathlib import Path
    from .crosswalks import fields as F
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "raw" / "nyfed" / "College-labor-data.xlsx"
    raw = pd.read_excel(path, "outcomes by major", header=None, dtype=str)
    hdr = next(i for i in range(len(raw))
               if raw.iloc[i].astype(str).str.contains("Major", case=False, na=False).any())
    t = pd.read_excel(path, "outcomes by major", header=hdr).dropna(how="all").dropna(axis=1, how="all")
    t = t.rename(columns={"Major": "major", "Unemployment Rate": "unemployment",
                          "Underemployment Rate": "underemployment",
                          "Median Wage Early Career": "wage_early",
                          "Median Wage Mid-Career": "wage_mid",
                          "Share with Graduate Degree": "grad_degree_share"})
    maj2key = {v: k for k, v in F.NYFED_MAJOR_BY_KEY.items() if v}
    t["field"] = t["major"].map(maj2key)
    t = t[t["field"].notna()].copy()
    for c in ["unemployment", "underemployment", "wage_early", "wage_mid", "grad_degree_share"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    return t[["field", "major", "unemployment", "underemployment", "wage_early", "wage_mid",
              "grad_degree_share"]]


def assign_regime(field: str, grad_share: float, hi_pipeline: float) -> str:
    """Ex-ante regime from EXTERNAL data only (never the gap):
    license-standardization > PhD-pipeline (grad-degree share >= hi_pipeline) > prestige-transmission."""
    from .crosswalks import fields as F
    if F.is_licensed(field):
        return "license-standardization"
    if grad_share == grad_share and grad_share >= hi_pipeline:
        return "PhD-pipeline"
    return "prestige-transmission"


def test_p1(gap_map: pd.DataFrame, dispersion: pd.DataFrame, proxy: str,
            reliable_only: bool = True) -> dict:
    """Spearman(gap, dispersion proxy). P1 predicts negative."""
    g = gap_map.copy()
    if reliable_only:
        g = g[g["reliable_flag"]]
    m = g.merge(dispersion, on="field", how="inner").dropna(subset=["gap", proxy])
    if len(m) < 4:
        return dict(proxy=proxy, reliable_only=reliable_only, n=len(m),
                    spearman=np.nan, p=np.nan, sign_ok=None, holds=None)
    rho, p = spearmanr(m[proxy], m["gap"])
    return dict(proxy=proxy, reliable_only=reliable_only, n=int(len(m)),
                spearman=float(rho), p=float(p),
                sign_ok=bool(rho < 0), holds=bool(rho < 0 and p < 0.10),
                fields=list(m["field"]))
