"""Tier-0 go/no-go analysis engine (proposal §7).

Builds the field-level Academic–Employer reputation gap from already-public,
already-computed data and evaluates the three GO conditions. All heavy logic lives
here so the notebook (``notebooks/tier0_go_no_go.ipynb``) stays a thin, readable
orchestration layer and the computation is unit-checkable.

Gap definition (within field, across institutions):

    prestige_score = -Rank                      # Wapman ranks.csv, 0 = most prestigious
    Gap_field      = 1 - Spearman(prestige_score, ER_earnings)

where ER_earnings is College Scorecard Field-of-Study Bachelor's median earnings.
A small/zero gap means academic prestige and labor-market reward agree (integrated
markets, e.g. CS); a large gap means they disagree.

Integration proxy = industry employment share by field:
    primary  SDR 2021 Table 12-3 (fine field):  Business-or-industry / All employed
    fallback SED 2021 Table 2-6 (broad field, humanities not in SDR)

Data provenance is in ``data/raw/SOURCES.md``. CIP / field / SDR mappings come from
``src.crosswalks`` — never redefined here.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .crosswalks import fields as F
from .crosswalks.institutions import normalize_institution_name

# Anchor data paths at the repo root (src/ is one level below it) so loaders work
# regardless of the caller's working directory (notebook runs from notebooks/, etc.).
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
WAPMAN_RANKS = RAW / "wapman2022" / "ranks.csv"
SCORECARD_FOS = RAW / "scorecard_fos" / "Most-Recent-Cohorts-Field-of-Study.csv"
SDR_TAB_12_3 = RAW / "sdr2021" / "tab012-003.xlsx"
SED_TAB_2_6 = RAW / "sed2021" / "tab002-006.xlsx"


# ---------------------------------------------------------------------------
# Academic Reputation (AR): Wapman SpringRank prestige
# ---------------------------------------------------------------------------
def load_prestige(path: Path = WAPMAN_RANKS) -> pd.DataFrame:
    """Per-institution prestige for each Tier-0 field from Wapman ranks.csv.

    ``Rank`` is the SpringRank prestige ordinal (0 = most prestigious; confirmed by
    Caltech=0 academia-wide, Stanford=0 in CS — a prestige, not production, result).
    Returns long: field_key, InstitutionName, inst_norm, rank, prestige_score(=-rank).
    """
    r = pd.read_csv(path)
    r = r[r["TaxonomyLevel"] == "Field"].copy()
    wf2key = {f["wapman_field"]: f["key"] for f in F.TIER0_FIELDS}
    r = r[r["TaxonomyValue"].isin(wf2key)].copy()
    r["field_key"] = r["TaxonomyValue"].map(wf2key)
    r["inst_norm"] = r["InstitutionName"].map(normalize_institution_name)
    r["prestige_score"] = -r["Rank"].astype(float)
    return r[["field_key", "InstitutionName", "inst_norm", "Rank", "prestige_score"]]


# ---------------------------------------------------------------------------
# Employer Reputation (ER): College Scorecard Field-of-Study earnings
# ---------------------------------------------------------------------------
def load_earnings(path: Path = SCORECARD_FOS, credlev: str = "3",
                  earn_col: str = "EARN_MDN_4YR") -> pd.DataFrame:
    """Per-institution median earnings for each Tier-0 field from Scorecard FoS.

    credlev "3" = Bachelor's (the undergraduate-level ER; PhD-prestige vs undergrad-
    earnings level mismatch is a documented Tier-0 caveat). Suppressed cells ('PS')
    -> NaN and dropped. Where a field spans several CIP-4 codes at one institution,
    the institution's earnings is the mean of available CIP medians (rank-stable).
    Returns: field_key, inst_norm, INSTNM, earn, n_cip.
    """
    cip2key = F.cip4_to_field_key()
    use = ["UNITID", "INSTNM", "CONTROL", "CIPCODE", "CREDLEV", earn_col]
    df = pd.read_csv(path, usecols=use, dtype=str)
    df = df[df["CREDLEV"] == credlev].copy()
    df["cip4"] = df["CIPCODE"].astype(str).str.zfill(4)
    df["field_key"] = df["cip4"].map(cip2key)
    df = df[df["field_key"].notna()].copy()
    df["earn"] = pd.to_numeric(df[earn_col], errors="coerce")
    df = df[df["earn"].notna()].copy()
    df["inst_norm"] = df["INSTNM"].map(normalize_institution_name)
    agg = (df.groupby(["field_key", "inst_norm"])
             .agg(earn=("earn", "mean"), INSTNM=("INSTNM", "first"), n_cip=("earn", "size"))
             .reset_index())
    return agg


# ---------------------------------------------------------------------------
# Gap: within-field disagreement of prestige and earnings orderings
# ---------------------------------------------------------------------------
def field_gap(field_key: str, prestige: pd.DataFrame, earnings: pd.DataFrame,
              min_n: int = 8) -> dict:
    """Gap for one field: 1 - Spearman(prestige_score, earn) over matched institutions."""
    p = prestige[prestige["field_key"] == field_key]
    e = earnings[earnings["field_key"] == field_key]
    m = p.merge(e, on="inst_norm", how="inner")
    n = len(m)
    out = dict(field_key=field_key, n_prestige=len(p), n_earn=len(e), n_matched=n)
    if n < min_n:
        out.update(rho=np.nan, pval=np.nan, gap=np.nan, ok=False)
        return out
    rho, pval = spearmanr(m["prestige_score"], m["earn"])
    out.update(rho=float(rho), pval=float(pval), gap=float(1.0 - rho), ok=True)
    return out


def all_field_gaps(prestige: pd.DataFrame, earnings: pd.DataFrame,
                   min_n: int = 8) -> pd.DataFrame:
    """Gap table for every Tier-0 field, annotated with label/domain."""
    rows = [field_gap(f["key"], prestige, earnings, min_n=min_n) for f in F.TIER0_FIELDS]
    g = pd.DataFrame(rows)
    meta = pd.DataFrame(F.TIER0_FIELDS)[["key", "label", "domain"]].rename(columns={"key": "field_key"})
    return g.merge(meta, on="field_key").sort_values("gap").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Integration proxy: industry employment share by field
# ---------------------------------------------------------------------------
def _num(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan


def load_sdr_industry_share(path: Path = SDR_TAB_12_3) -> dict:
    """{field_key: industry_share%} from SDR 2021 Table 12-3 (fine field).
    industry_share = Business-or-industry / All employed. SEH fields only."""
    raw = pd.read_excel(path, header=None, dtype=str)
    rows = {}
    for i in range(5, len(raw)):
        lab = str(raw.iloc[i, 0]).strip()
        if lab and lab != "nan":
            rows[lab] = (_num(raw.iloc[i, 1]), _num(raw.iloc[i, 5]))  # All, Business
    out = {}
    for f in F.TIER0_FIELDS:
        lab = F.sdr_fine_for(f["key"])
        if lab and lab in rows:
            alln, bus = rows[lab]
            if alln and bus and alln > 0:
                out[f["key"]] = 100.0 * bus / alln
    return out


def load_sed_humanities_share(path: Path = SED_TAB_2_6, year: str = "2021") -> float:
    """SED 2021 Table 2-6 industry share (%) for 'Humanities and arts' (broad field).
    Used for english/history/philosophy, which SDR (SEH-only) does not cover.

    Table layout: a 'Industry or business (%)' block whose year rows have columns
    [year, All, S&E total, Life sci, Physical+earth, Math+CS, Psych+social, Eng,
     Non-S&E total, Education, Humanities and arts, Other]. Humanities = column 10.
    """
    raw = pd.read_excel(path, header=None, dtype=str)
    in_block = False
    HUMANITIES_COL = 10
    for i in range(len(raw)):
        c0 = str(raw.iloc[i, 0]).strip()
        if c0.startswith("Industry or business"):
            in_block = True
            continue
        if in_block:
            if c0 and c0[0].isalpha():       # next sector block header -> stop
                break
            if c0 == year:
                return _num(raw.iloc[i, HUMANITIES_COL])
    return np.nan


def industry_share_table() -> pd.DataFrame:
    """Combined industry-share proxy per field: SDR fine where available, else SED
    'Humanities and arts' broad fallback. Columns: field_key, industry_share, source."""
    sdr = load_sdr_industry_share()
    hum = load_sed_humanities_share()
    rows = []
    for f in F.TIER0_FIELDS:
        k = f["key"]
        if k in sdr:
            rows.append(dict(field_key=k, industry_share=sdr[k], proxy_source="SDR fine"))
        elif F.sed_broad_for(k) == "Humanities and arts":
            rows.append(dict(field_key=k, industry_share=hum, proxy_source="SED broad (humanities)"))
        else:
            rows.append(dict(field_key=k, industry_share=np.nan, proxy_source="missing"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GO / NO-GO evaluation (three conditions, proposal §7)
# ---------------------------------------------------------------------------
def evaluate_go(gaps: pd.DataFrame, proxy: pd.DataFrame,
                sd_threshold: float = 0.10) -> dict:
    """Evaluate the three GO conditions and return a verdict dict.

    (1) variation:  cross-field SD of Gap_field is non-trivial (>= sd_threshold).
    (2) mechanism:  Gap_field correlates NEGATIVELY with the industry-share proxy.
                    Judged on the CONSISTENTLY-measured SDR fine fields only — the
                    3 humanities fields use a different-universe SED proxy (commitment
                    flow vs employed stock) pinned at one value, so including them can
                    manufacture a spurious sign. The honest test excludes them.
    (3) headline:   computer science is below the median gap (low-gap tail).

    Strict reading: all three required (proposal §7). The full-sample correlation is
    also reported for transparency.
    """
    g = gaps[gaps["ok"]].copy()
    sd = float(g["gap"].std(ddof=1))
    gap_range = (float(g["gap"].min()), float(g["gap"].max()))

    m = g.merge(proxy, on="field_key").dropna(subset=["industry_share", "gap"])
    rho_all, p_all = spearmanr(m["industry_share"], m["gap"])
    pear_all = float(np.corrcoef(m["industry_share"], m["gap"])[0, 1])
    # consistently-measured subset: SDR fine fields only (drop SED-fallback humanities)
    m_sdr = m[m["proxy_source"] == "SDR fine"]
    if len(m_sdr) > 3:
        rho_sdr, p_sdr = spearmanr(m_sdr["industry_share"], m_sdr["gap"])
    else:
        rho_sdr, p_sdr = np.nan, np.nan

    med = float(g["gap"].median())
    cs = g[g["field_key"] == F.HEADLINE_FIELD]
    cs_gap = float(cs["gap"].iloc[0]) if len(cs) else np.nan
    cs_rank = int((g["gap"] < cs_gap).sum()) + 1 if len(cs) else None  # 1 = lowest gap

    cond1 = sd >= sd_threshold
    cond2 = bool(rho_sdr < 0) if rho_sdr == rho_sdr else bool(rho_all < 0)  # honest test
    cond3 = bool(cs_gap < med) if cs_gap == cs_gap else False
    verdict = "GO" if (cond1 and cond2 and cond3) else "NO-GO"
    return dict(
        n_fields=len(g), gap_sd=sd, gap_range=gap_range, gap_median=med,
        cs_gap=cs_gap, cs_rank_of_n=(cs_rank, len(g)),
        spearman_gap_vs_industry_all=float(rho_all), p_all=float(p_all),
        pearson_gap_vs_industry_all=pear_all,
        spearman_gap_vs_industry_sdr_only=float(rho_sdr) if rho_sdr == rho_sdr else np.nan,
        p_sdr_only=float(p_sdr) if p_sdr == p_sdr else np.nan, n_sdr_only=len(m_sdr),
        cond1_variation=bool(cond1), cond2_mechanism_negative=bool(cond2),
        cond3_cs_low_gap=bool(cond3), verdict=verdict,
        merged=m,
    )
