"""Tier 0.5 — diagnose the real driver of the AR–ER gap.

The Tier-0 gap construction is ACCEPTED and unchanged:
    Gap_field = 1 − Spearman(Wapman SpringRank prestige, Scorecard bachelor's EARN_MDN_4YR)
AR is research/PhD-level (academic-market price); ER is the undergraduate graduate mass
(labor-market price). That is the AR–ER distinction, not a confound.

Tier 0.5 tests four *competing* explanators of the cross-field gap, each pre-registered
with a predicted direction; none is dropped on significance:

  1. task_distance   (H2)  Gathmann–Schönberg angular distance between a field's
                           academic-research task vector (O*NET "<field> Teachers,
                           Postsecondary" Work-Activities IM×LV) and its employment-
                           weighted modal-industry task vector (CIP→O*NET-SOC
                           destinations).  PREDICTED: gap RISES with task distance.
  2. earnings_dispersion   within-field SD/CV of institution-level bachelor's earnings.
                           PREDICTED: gap FALLS as dispersion rises (compressed earnings
                           ⇒ noisy Spearman ⇒ mechanically high gap).
  3. gradschool_share      field-level grad-school pull = IPEDS doctoral/bachelor degree
                           ratio.  PREDICTED: gap RISES with grad-school pull (prestige
                           depts send more to grad school ⇒ lower 4-yr earnings).
  4. price_wedge           (industry_salary − academic_salary)/mean from SDR Table 54
                           by sector. PREDICTED: gap RISES with |wedge|. Note: the gap
                           measures RANK agreement between markets; the wedge measures
                           PRICE agreement — whether they coincide is itself a result.

Crosswalks (field↔teacher-SOC, field↔SDR-salary-row) live in src.crosswalks.fields.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import tier0 as T
from .crosswalks import fields as F

RAW = T.RAW
ONET = RAW / "onet"
WORK_ACTIVITIES = ONET / "Work Activities.txt"
CIP_SOC_XWALK = ONET / "Education_CIP_to_ONET_SOC.xlsx"
SDR_TAB_54 = RAW / "sdr2021" / "tab054.xlsx"
OEWS_DIR = RAW / "oews"
IPEDS_DIR = RAW / "ipeds"


# ---------------------------------------------------------------------------
# 1. Task distance (O*NET)
# ---------------------------------------------------------------------------
def _soc_vectors(path: Path = WORK_ACTIVITIES) -> pd.DataFrame:
    """O*NET-SOC × 41 Generalized-Work-Activity matrix of IM×LV products."""
    wa = pd.read_csv(path, sep="\t", dtype=str)
    wa["Data Value"] = pd.to_numeric(wa["Data Value"], errors="coerce")
    piv = wa.pivot_table(index=["O*NET-SOC Code", "Element ID"], columns="Scale ID",
                         values="Data Value")
    piv["IMLV"] = piv["IM"] * piv["LV"]
    elements = sorted(wa["Element ID"].unique())
    return piv["IMLV"].unstack("Element ID").reindex(columns=elements)


def _cip_soc_crosswalk(path: Path = CIP_SOC_XWALK) -> pd.DataFrame:
    cw = pd.read_excel(path, skiprows=3, header=None,
                       names=["cip", "cip_title", "soc", "soc_title"], dtype=str)
    cw = cw.dropna(subset=["cip", "soc"])
    cw = cw[cw["cip"] != "2020 CIP Code"].copy()
    cw["cip4"] = cw["cip"].str.replace(".", "", regex=False).str[:4]
    return cw


def load_oews_employment() -> dict:
    """{6-digit SOC: total US employment} from OEWS national file, if present; else {}.
    Used to employment-weight industry destination occupations. Equal-weight otherwise."""
    if not OEWS_DIR.exists():
        return {}
    for f in list(OEWS_DIR.glob("*.xlsx")) + list(OEWS_DIR.glob("*.csv")):
        try:
            df = pd.read_excel(f, dtype=str) if f.suffix == ".xlsx" else pd.read_csv(f, dtype=str)
            cols = {c.upper(): c for c in df.columns}
            occ = cols.get("OCC_CODE"); emp = cols.get("TOT_EMP")
            if occ and emp:
                df = df[df[occ].astype(str).str.match(r"^\d{2}-\d{4}$", na=False)]
                e = pd.to_numeric(df[emp].astype(str).str.replace(",", "").str.replace("**", "", regex=False),
                                  errors="coerce")
                return dict(zip(df[occ], e))
        except Exception:
            continue
    return {}


def _vec_for(code: str, V: pd.DataFrame):
    if code in V.index:
        return V.loc[code].values
    base = code.split(".")[0]
    sub = V[V.index.str.startswith(base)]
    return sub.mean().values if len(sub) else None


def _angular(a, b):
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    cos = min(1.0, max(-1.0, cos))
    return float(np.degrees(np.arccos(cos))), float(1.0 - cos)


def task_distance(weighting: str = "oews") -> pd.DataFrame:
    """Per-field academia↔industry task distance. weighting 'oews' employment-weights
    the industry destinations (falls back to equal if OEWS absent); 'equal' forces equal."""
    V = _soc_vectors()
    cw = _cip_soc_crosswalk()
    oews = load_oews_employment() if weighting == "oews" else {}
    rows = []
    for f in F.TIER0_FIELDS:
        k = f["key"]
        tsoc = F.teacher_soc_for(k)
        av = _vec_for(tsoc, V) if tsoc else None
        dest = set()
        for c4 in f["cip4"]:
            dest |= set(cw.loc[cw["cip4"] == c4, "soc"])
        dest = {d for d in dest if not d.startswith("25-10")}  # drop teaching SOCs
        vecs, wts = [], []
        for d in sorted(dest):
            v = _vec_for(d, V)
            if v is None:
                continue
            vecs.append(v)
            wts.append(oews.get(d.split(".")[0], np.nan))
        if av is None or not vecs:
            rows.append(dict(field_key=k, teacher_soc=tsoc, n_dest=len(vecs),
                             task_distance=np.nan, task_cos_dist=np.nan, weighting="none"))
            continue
        wts = np.array(wts, dtype=float)
        if oews and np.isfinite(wts).sum() >= max(1, len(wts) // 2):
            w = np.where(np.isfinite(wts), wts, np.nanmedian(wts)); w = w / w.sum()
            iv = np.average(np.vstack(vecs), axis=0, weights=w); used = "oews"
        else:
            iv = np.mean(np.vstack(vecs), axis=0); used = "equal"
        ang, cosd = _angular(av, iv)
        rows.append(dict(field_key=k, teacher_soc=tsoc, n_dest=len(vecs),
                         task_distance=ang, task_cos_dist=cosd, weighting=used))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Earnings dispersion (Scorecard)
# ---------------------------------------------------------------------------
def earnings_dispersion(credlev: str = "3", earn_col: str = "EARN_MDN_4YR") -> pd.DataFrame:
    e = T.load_earnings(credlev=credlev, earn_col=earn_col)
    g = e.groupby("field_key")["earn"]
    out = pd.DataFrame({
        "earn_mean": g.mean(), "earn_sd": g.std(ddof=1),
        "earn_iqr": g.quantile(0.75) - g.quantile(0.25), "n_inst": g.size(),
    }).reset_index()
    out["earn_cv"] = out["earn_sd"] / out["earn_mean"]
    return out


# ---------------------------------------------------------------------------
# 3. Grad-school pull (IPEDS doctoral/bachelor degree ratio)
# ---------------------------------------------------------------------------
def gradschool_share(ipeds_csv: Path | None = None) -> pd.DataFrame:
    """Field-level grad-school pull = national doctoral / bachelor completions ratio,
    from IPEDS Completions (CIPCODE × AWLEVEL; 5 = Bachelor's, 17 = Doctor's research).
    Returns empty (NaN) if the IPEDS file is absent so the rest of Tier 0.5 still runs."""
    if ipeds_csv is None:
        cands = [p for p in IPEDS_DIR.glob("*.csv")
                 if "_a" in p.name.lower() and "dict" not in p.name.lower()] if IPEDS_DIR.exists() else []
        cands.sort(key=lambda p: ("_rv" in p.name.lower(), len(p.name)))  # prefer plain over _RV
        ipeds_csv = cands[0] if cands else None
    if ipeds_csv is None or not Path(ipeds_csv).exists():
        return pd.DataFrame({"field_key": F.field_keys(), "doc_ba_ratio": np.nan})
    df = pd.read_csv(ipeds_csv, dtype=str, encoding="latin-1")
    cols = {c.upper(): c for c in df.columns}
    cip, awl, tot = cols["CIPCODE"], cols["AWLEVEL"], cols["CTOTALT"]
    # restrict to the "total across race/sex, all majors" rows when MAJORNUM present
    if "MAJORNUM" in cols:
        df = df[df[cols["MAJORNUM"]].astype(str).isin(["1", "1.0"])]
    df["cip4"] = df[cip].astype(str).str.replace(".", "", regex=False).str.zfill(4).str[:4]
    df["awl"] = pd.to_numeric(df[awl], errors="coerce")
    df["ct"] = pd.to_numeric(df[tot], errors="coerce")
    c2k = F.cip4_to_field_key()
    df["field_key"] = df["cip4"].map(c2k)
    df = df[df["field_key"].notna()]
    ba = df[df["awl"] == 5].groupby("field_key")["ct"].sum()
    doc = df[df["awl"] == 17].groupby("field_key")["ct"].sum()
    out = pd.DataFrame({"ba": ba, "doc": doc}).reindex(F.field_keys()).fillna(0.0)
    out["doc_ba_ratio"] = out["doc"] / out["ba"].replace(0, np.nan)
    return out.reset_index().rename(columns={"index": "field_key"})


# ---------------------------------------------------------------------------
# 4. Academia–industry price wedge (SDR Table 54)
# ---------------------------------------------------------------------------
def price_wedge(path: Path = SDR_TAB_54) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None, dtype=str)
    # col1 = All median, col3 = 4-yr educational institution, col7 = Private for-profit
    rows = {}
    for i in range(5, len(raw)):
        lab = str(raw.iloc[i, 0]).strip()
        if lab and lab != "nan":
            rows[lab] = (T._num(raw.iloc[i, 1]), T._num(raw.iloc[i, 3]), T._num(raw.iloc[i, 7]))
    out = []
    for f in F.TIER0_FIELDS:
        lab = F.sdr_salary_field_for(f["key"])
        allm, acad, ind = rows.get(lab, (np.nan, np.nan, np.nan)) if lab else (np.nan, np.nan, np.nan)
        wedge = (ind - acad) / allm if (allm and acad and ind) else np.nan
        out.append(dict(field_key=f["key"], academic_sal=acad, industry_sal=ind,
                        mean_sal=allm, price_wedge=wedge,
                        price_wedge_abs=abs(wedge) if wedge == wedge else np.nan))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Assemble + per-explanator relationships
# ---------------------------------------------------------------------------
EXPLANATORS = {
    "task_distance":      (+1, "gap RISES with task distance (H2)"),
    "earn_cv":            (-1, "gap FALLS as earnings dispersion rises (artifact)"),
    "doc_ba_ratio":       (+1, "gap RISES with grad-school pull"),
    "price_wedge_abs":    (+1, "gap RISES with |price wedge|"),
}


def assemble(gaps: pd.DataFrame, task_weighting: str = "oews") -> pd.DataFrame:
    td = task_distance(weighting=task_weighting)
    ed = earnings_dispersion()
    gs = gradschool_share()
    pw = price_wedge()
    meta = pd.DataFrame(F.TIER0_FIELDS)[["key", "label", "domain"]].rename(columns={"key": "field_key"})
    df = (gaps[["field_key", "gap", "rho", "n_matched", "ok"]]
          .merge(td, on="field_key").merge(ed, on="field_key")
          .merge(gs[["field_key", "doc_ba_ratio"]], on="field_key")
          .merge(pw, on="field_key").merge(meta, on="field_key"))
    return df


def per_explanator(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["ok"]]
    out = []
    for var, (pred_sign, desc) in EXPLANATORS.items():
        m = g.dropna(subset=[var, "gap"])
        if len(m) < 4:
            out.append(dict(explanator=var, n=len(m), spearman=np.nan, p=np.nan,
                            predicted_sign=pred_sign, sign_ok=None, desc=desc))
            continue
        rho, p = spearmanr(m[var], m["gap"])
        out.append(dict(explanator=var, n=len(m), spearman=float(rho), p=float(p),
                        predicted_sign=pred_sign,
                        sign_ok=bool(np.sign(rho) == pred_sign), desc=desc))
    return pd.DataFrame(out)


def combined_ols(df: pd.DataFrame, predictors=("task_distance", "earn_cv", "doc_ba_ratio", "price_wedge_abs")):
    """Standardized OLS: gap ~ z(predictors). Returns (fitted model, n, used df).
    Drops only rows missing any included predictor; never drops a predictor."""
    import statsmodels.api as sm
    cols = list(predictors)
    m = df[df["ok"]].dropna(subset=cols + ["gap"]).copy()
    if len(m) < len(cols) + 2:
        return None, len(m), m
    X = m[cols].apply(lambda c: (c - c.mean()) / c.std(ddof=0))
    X = sm.add_constant(X)
    return sm.OLS(m["gap"].values, X).fit(), len(m), m
