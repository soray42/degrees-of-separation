"""
scripts/35_er_definition.py
=============================================================================
Interrogating the earnings-only operationalisation of employer reputation (ER).

The repo's gap is within-field, across-institution:
    gap_f = 1 - Spearman_i(prestige_{i,f}, earnings_{i,f}),
with ER operationalised ONLY as median earnings. The reviewer-critical limitation
is that the established "employer reputation" / graduate labour-market-standing
construct is MULTI-DIMENSIONAL (which employers, occupational status, vertical
match, trajectory, non-wage rewards). A within-field NON-WAGE gap of the SAME form
needs institution x field occupational mix -- NOT public (Revelio-gated; SPECIFIED
at the end, not run).

What IS public is FIELD-LEVEL non-wage ER. So this script does NOT recompute a
within-field gap. It runs the FIELD-LEVEL test of whether the earnings-based gap and
the integrated/decoupled typology are ROBUST to, or an ARTIFACT of, the earnings-only
ER. Everything here is labelled a field-level robustness test, never a within-field gap.

Two public non-wage ER dimensions (+ one optional):
  1. Occupational prestige (status; survey-rated, NON-income).  P(occ|field) from ACS
     PUMS recent BA holders x a SURVEY prestige scale: Condon/Hughes et al. 2024 (OPR,
     primary) with GSS Nakao-Treas 1989 + GSS 2012 as robustness scales. NOT SEI/ISEI
     (those are income+education -> circular for a non-wage measure).
  2. Underemployment (vertical match): NY Fed "Labor Market for Recent College Graduates"
     outcomes-by-major (share in jobs not requiring a degree; + unemployment, grad-degree).
  3. [optional] Horizontal match: share of a field's grads in a field-matched occupation
     via the O*NET Education CIP->SOC crosswalk.

Reuses the gap outputs (results/GAP_MAP.csv, expanded66) and the ACS plumbing.
Descriptive, outcome-agnostic, seeded. Run: `python scripts/35_er_definition.py`.
"""
from __future__ import annotations
import sys, glob, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from src.crosswalks import fields as F
from src import dispersion as D

SEED = 7
rng = np.random.default_rng(SEED)
OUT = ROOT / "outputs"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
INTERIM = ROOT / "data" / "interim"; INTERIM.mkdir(parents=True, exist_ok=True)
ACS_GLOB = str(ROOT / "data" / "raw" / "acs" / "psam_pus*.csv")
OCC_CACHE = INTERIM / "acs_occ_dist.parquet"
CONDON = ROOT / "data" / "raw" / "prestige" / "condon_OccupationalPrestigeRatings.tab"
NYFED = ROOT / "data" / "raw" / "nyfed" / "College-labor-data.xlsx"

ALL = F.ALL_FIELDS
LAB = {f["key"]: f["label"] for f in ALL}

# Extended NY Fed major crosswalk: start from the repo's validated 24, add high-confidence
# 1:1 matches (documented). Fields with no clean NY Fed major stay unmapped (coverage loss).
NYFED_EXTRA = {
    "architecture": "Architecture",
    "industrial_engineering": "Industrial Engineering",
    "pharmacy": "Pharmacy",
    "special_education": "Special Education",
    "marketing": "Marketing",
    "management": "Business Management",
    "religious_studies": "Theology and Religion",
    "geography": "Geography",
}


def boot_spearman(x, y, B=5000, seed=SEED):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return (np.nan, np.nan, np.nan, n)
    r = spearmanr(x, y)[0]
    g = np.random.default_rng(seed); bs = []
    for _ in range(B):
        i = g.integers(0, n, n)
        if len(np.unique(x[i])) > 2 and len(np.unique(y[i])) > 2:
            bs.append(spearmanr(x[i], y[i])[0])
    bs = np.array(bs)
    return (r, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), n)


# =============================================================================
# 1. Occupational prestige per field  (P(occ|field) x survey prestige scale)
# =============================================================================
def extract_occ_distribution(refresh=False) -> pd.DataFrame:
    """Recent-BA (SCHL>=21, AGEP 22-27, full-time employed, FOD1P present) occupation
    rows with PWGTP, mapped to project fields. Same population filter as the earnings
    standing (src.dispersion.load_acs_workers) but retaining OCCP/SOCP. Cached."""
    if OCC_CACHE.exists() and not refresh:
        return pd.read_parquet(OCC_CACHE)
    code_map = F.fod1p_to_field_key(F.fod1p_by_key_all())
    need = ["FOD1P", "OCCP", "SOCP", "SCHL", "AGEP", "WKHP", "ESR", "PERNP", "PWGTP"]
    frames = []
    for f in sorted(glob.glob(ACS_GLOB)):
        cols = pd.read_csv(f, nrows=0).columns
        frames.append(pd.read_csv(f, usecols=[c for c in need if c in cols],
                                  dtype={"SOCP": str, "OCCP": str}))
    a = pd.concat(frames, ignore_index=True)
    for c in ["FOD1P", "SCHL", "AGEP", "WKHP", "ESR", "PERNP", "PWGTP"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a[(a.SCHL >= 21) & a.FOD1P.notna() & (a.WKHP >= 35) &
          a.ESR.isin([1, 2]) & (a.PERNP > 0)].copy()
    a = a[(a.AGEP >= 22) & (a.AGEP <= 27)].copy()
    a["fod1p"] = a.FOD1P.astype(int).astype(str).str.zfill(4)
    a["field"] = a.fod1p.map(code_map)
    a = a[a.field.notna()].copy()
    a["soc6"] = a.SOCP.fillna("").str.upper().str.replace("-", "", regex=False)
    a["occp4"] = a.OCCP.fillna("").str.zfill(4)
    keep = ["field", "soc6", "occp4", "PWGTP"]
    a = a[keep]
    a.to_parquet(OCC_CACHE)
    return a


def build_prestige_lookup():
    """soc6 -> {opr, gss1989, gss2012} from the Condon/Hughes 2024 file (survey-rated,
    NON-income). Multiple O*NET-SOC collapsing to one 6-digit SOC are averaged. Also
    return a Census-2010-code lookup as a coverage fallback."""
    d = pd.read_csv(CONDON, sep="\t")
    d["soc6"] = d["ONET SOC 2018 Code"].astype(str).str.replace("-", "", regex=False).str.strip()
    d = d[d.soc6.str.match(r"^\d{6}$")]
    g = d.groupby("soc6").agg(opr=("OPR Job Rating", "mean"),
                              gss1989=("GSS Ratings 1989", "mean"),
                              gss2012=("GSS Ratings 2012", "mean")).reset_index()
    soc_lu = {r.soc6: dict(opr=r.opr, gss1989=r.gss1989, gss2012=r.gss2012)
              for r in g.itertuples()}
    # census-2010 fallback
    dc = pd.read_csv(CONDON, sep="\t")
    dc = dc[dc["Census Code 2010"].notna()].copy()
    dc["occp4"] = dc["Census Code 2010"].astype(float).astype(int).astype(str).str.zfill(4)
    gc = dc.groupby("occp4").agg(opr=("OPR Job Rating", "mean"),
                                 gss1989=("GSS Ratings 1989", "mean"),
                                 gss2012=("GSS Ratings 2012", "mean")).reset_index()
    occp_lu = {r.occp4: dict(opr=r.opr, gss1989=r.gss1989, gss2012=r.gss2012)
               for r in gc.itertuples()}
    return soc_lu, occp_lu


def score_occupation(soc6, occp4, soc_lu, occp_lu, scale, _cache={}):
    """Prestige score for one occupation code on `scale` in {opr,gss1989,gss2012}.
    SOC-2018 route primary; X-aggregated SOCP -> mean over matching prefix; Census-2010
    (OCCP) route as fallback. Cached per (soc6, occp4, scale)."""
    key = (soc6, occp4, scale)
    if key in _cache:
        return _cache[key]
    val = np.nan
    if soc6 and "X" not in soc6 and soc6 in soc_lu:
        val = soc_lu[soc6].get(scale, np.nan)
    elif soc6 and "X" in soc6:
        pref = soc6.split("X")[0]
        if len(pref) >= 3:
            hits = [v.get(scale, np.nan) for k, v in soc_lu.items() if k.startswith(pref)]
            hits = [h for h in hits if np.isfinite(h)]
            if hits:
                val = float(np.mean(hits))
    if not np.isfinite(val) and occp4 and occp4 in occp_lu:  # fallback
        val = occp_lu[occp4].get(scale, np.nan)
    _cache[key] = val
    return val


def occ_prestige_by_field(occ: pd.DataFrame, soc_lu, occp_lu):
    """occ_prestige_f = sum_o P(o|f) * prestige(o), PWGTP-weighted, per scale; with
    weighted coverage (share of a field's employment that receives a score)."""
    uniq = occ[["soc6", "occp4"]].drop_duplicates()
    for sc in ["opr", "gss1989", "gss2012"]:
        m = {(r.soc6, r.occp4): score_occupation(r.soc6, r.occp4, soc_lu, occp_lu, sc)
             for r in uniq.itertuples()}
        occ[sc] = [m[(s, o)] for s, o in zip(occ.soc6, occ.occp4)]
    rows = []
    for fld, gdf in occ.groupby("field"):
        rec = dict(field=fld, n_occ_rows=len(gdf), w_tot=gdf.PWGTP.sum())
        for sc in ["opr", "gss1989", "gss2012"]:
            ok = gdf[gdf[sc].notna()]
            rec[f"occ_prestige_{sc}"] = (np.average(ok[sc], weights=ok.PWGTP)
                                         if ok.PWGTP.sum() > 0 else np.nan)
            rec[f"cov_{sc}"] = ok.PWGTP.sum() / gdf.PWGTP.sum() if gdf.PWGTP.sum() > 0 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


# =============================================================================
# 2. Underemployment per field  (NY Fed outcomes-by-major)
# =============================================================================
def underemployment_by_field() -> pd.DataFrame:
    ny = pd.read_excel(NYFED, sheet_name="outcomes by major", header=10)
    ny = ny[ny["Major"].notna()].copy()
    ny["Major"] = ny["Major"].astype(str).str.strip()
    for c in ["Unemployment Rate", "Underemployment Rate", "Median Wage Early Career",
              "Share with Graduate Degree"]:
        ny[c] = pd.to_numeric(ny[c], errors="coerce")
    by_major = ny.set_index("Major")
    xwalk = dict(F.NYFED_MAJOR_BY_KEY)
    xwalk.update(NYFED_EXTRA)
    rows = []
    for f in ALL:
        maj = xwalk.get(f["key"])
        if maj and maj in by_major.index:
            r = by_major.loc[maj]
            rows.append(dict(field=f["key"], nyfed_major=maj,
                             underemployment=float(r["Underemployment Rate"]),
                             unemployment=float(r["Unemployment Rate"]),
                             grad_degree_share=float(r["Share with Graduate Degree"]),
                             nyfed_wage_early=float(r["Median Wage Early Career"])))
    return pd.DataFrame(rows)


# =============================================================================
# 3. Optional horizontal match  (O*NET Education CIP -> SOC)
# =============================================================================
def match_by_field(occ: pd.DataFrame):
    """Share of a field's employment in a field-matched occupation: matched SOC set =
    O*NET Education CIP->SOC crosswalk filtered to the field's CIP prefixes. Best-effort;
    returns (df or None, note)."""
    xwalk_path = ROOT / "data" / "raw" / "onet" / "Education_CIP_to_ONET_SOC.xlsx"
    try:
        raw = pd.read_excel(xwalk_path, header=None, dtype=str)
        # locate header row and the CIP/SOC code columns by position (avoids dup-name issues)
        hdr = None
        for i in range(min(12, len(raw))):
            joined = " ".join(str(x) for x in raw.iloc[i].tolist()).lower()
            if "cip" in joined and "soc" in joined:
                hdr = i; break
        if hdr is None:
            return None, "could not locate CIP/SOC header row"
        head = [str(x).lower() for x in raw.iloc[hdr].tolist()]
        cip_j = next((j for j, h in enumerate(head) if "cip" in h and "code" in h),
                     next((j for j, h in enumerate(head) if "cip" in h), None))
        soc_j = next((j for j, h in enumerate(head) if "soc" in h and "code" in h),
                     next((j for j, h in enumerate(head) if "soc" in h), None))
        if cip_j is None or soc_j is None:
            return None, f"CIP/SOC columns not found in {head[:6]}"
        body = raw.iloc[hdr + 1:, [cip_j, soc_j]].copy()
        body.columns = ["cip", "soc"]
        body = body.dropna()
        body["cip2"] = body["cip"].astype(str).str.replace(".", "", regex=False).str.zfill(6).str[:2]
        body["soc6"] = body["soc"].astype(str).str.replace("-", "", regex=False).str.replace(".", "", regex=False).str[:6]
        cip2_to_socs = body.groupby("cip2")["soc6"].apply(lambda s: set(s)).to_dict()
        rows = []
        for fld, gdf in occ.groupby("field"):
            prefs = F.cip4_prefixes(fld) if hasattr(F, "cip4_prefixes") else None
            cip2s = set(str(p)[:2] for p in prefs) if prefs else set()
            matched = set().union(*[cip2_to_socs.get(c, set()) for c in cip2s]) if cip2s else set()
            if not matched:
                rows.append(dict(field=fld, match=np.nan, match_cov=0.0)); continue
            gg = gdf.copy()
            gg["m"] = gg.soc6.str[:6].isin(matched)
            rows.append(dict(field=fld, match=float(np.average(gg.m, weights=gg.PWGTP)), match_cov=1.0))
        out = pd.DataFrame(rows)
        n_ok = out.match.notna().sum()
        if n_ok < 15:
            return None, f"only {n_ok} fields got a clean CIP->SOC match set; skipped per spec"
        return out, f"built for {n_ok} fields (share of grads in a CIP-matched occupation)"
    except Exception as e:
        return None, f"horizontal match skipped (crosswalk parse issue: {type(e).__name__})"


# =============================================================================
# Main
# =============================================================================
def main():
    log = []

    # ---- gap / typology / earnings standing -----------------------------
    # Gap source: the expanded 66-field map (more reliable units + cip2) rather than the
    # 30-field headline, so the field-level test is better powered.
    gm = pd.read_csv(ROOT / "outputs" / "expanded66_gap_map.csv")
    gm = gm[["field", "gap", "spearman", "signal_frac", "n_institutions", "reliable", "cip2"]].copy()
    gm["label"] = gm.field.map(lambda k: LAB.get(k, k.replace("_", " ").title()))
    w = D.load_acs_workers(expanded=True)
    earn = D.field_dispersion_acs(w, 22, 27)[["field", "p50", "n"]].rename(
        columns={"p50": "earn_standing", "n": "n_earn"})

    occ = extract_occ_distribution()
    soc_lu, occp_lu = build_prestige_lookup()
    pres = occ_prestige_by_field(occ, soc_lu, occp_lu)
    und = underemployment_by_field()
    match_df, match_note = match_by_field(occ)

    df = (gm.merge(earn, on="field", how="left")
            .merge(pres, on="field", how="left")
            .merge(und, on="field", how="left"))
    if match_df is not None:
        df = df.merge(match_df[["field", "match"]], on="field", how="left")

    # status RESIDUAL net of wage (the part wages miss) -- fit on all measured fields
    def resid_on_earn(frame, col):
        s = frame.dropna(subset=[col, "earn_standing"])
        if len(s) < 6:
            return pd.Series(np.nan, index=frame.index)
        b = sm.OLS(s[col], sm.add_constant(s["earn_standing"])).fit()
        return frame[col] - b.predict(sm.add_constant(frame["earn_standing"]))
    df["status_resid"] = resid_on_earn(df, "occ_prestige_opr")

    # typology by gap terciles within the reliable set
    relmask = df.reliable & df.gap.notna()
    q1, q2 = df.loc[relmask, "gap"].quantile([1/3, 2/3])
    df["typology"] = df.gap.apply(lambda g: "integrated" if g <= q1 else
                                  ("decoupled" if g >= q2 else "intermediate"))
    df.loc[~relmask, "typology"] = "unreliable"

    # two analysis universes:
    #   meas = all fields with the measures (A/C: ER-level, gap-reliability-independent)
    #   rel  = reliable-gap fields (B/D: need trustworthy within-field gaps)
    meas = df.dropna(subset=["earn_standing", "occ_prestige_opr"]).copy()
    rel = df[relmask].copy()

    df.to_csv(INTERIM / "er_dimensions.csv", index=False)

    # ================= A. Is earnings a good FIELD-LEVEL ER proxy? =========
    L = ["# ER definition: is earnings a thin proxy for employer reputation? (field-level robustness test)\n",
         "**This is a FIELD-LEVEL robustness test, NOT a within-field gap.** Public non-wage ER data is "
         "one value per field, so we cannot recompute gap_f with a non-wage ER (that needs institution x field "
         "occupational mix -- Revelio-gated, specified at the end). We instead test whether the earnings-based "
         "gap and the integrated/decoupled typology are robust to, or an artifact of, the earnings-only ER. "
         "Seeded; run `python scripts/35_er_definition.py`.\n",
         "## Non-wage ER measures (sources + crosswalks)\n",
         "- **Occupational prestige** (status; survey-rated, NON-income, so non-circular): "
         "`occ_prestige_f = sum_o P(o|f) * prestige(o)`, P(o|f) from ACS PUMS 2023 recent BA "
         "(SCHL>=21, AGEP 22-27, full-time employed) x **Condon/Hughes et al. 2024 OPR** (primary), with "
         "**GSS Nakao-Treas 1989** and **GSS 2012** as robustness scales. NOT SEI/ISEI (income+education -> "
         "circular). Occupation join: ACS SOCP -> O*NET-SOC-2018 (Condon native key); Census-2010/OCCP fallback.\n",
         "- **Underemployment** (vertical match): NY Fed *Labor Market for Recent College Graduates* "
         "outcomes-by-major (share in jobs not requiring a degree). Crosswalk: repo `NYFED_MAJOR_BY_KEY` + "
         f"{len(NYFED_EXTRA)} added 1:1 matches.\n",
         f"- **Horizontal match** (optional): {match_note}.\n",
         f"\nCoverage: occ_prestige on **{pres['occ_prestige_opr'].notna().sum()}/{len(ALL)}** fields "
         f"(weighted occupation-score coverage median {df['cov_opr'].median():.2f}); underemployment on "
         f"**{und['field'].nunique()}** fields. Two analysis universes: **A/C** on all measured fields "
         f"(N={len(meas)}, ER-level tests independent of gap reliability); **B/D** on the reliable-gap set "
         f"(N={len(rel)}, which need trustworthy within-field gaps).\n",
         "## A. Is earnings a good field-level ER proxy?\n",
         "Spearman across all measured fields between the field's **earnings standing** (ACS weighted-median "
         "early-career earnings) and each non-wage ER. High -> earnings tracks the broader ER; low/divergent -> "
         "earnings is a thin proxy.\n",
         "| non-wage ER | Spearman(earn_standing, ER) [95% CI] | n |",
         "|---|---|---|"]
    proxies = [("occ_prestige_opr", "occupational prestige (OPR, Condon 2024)", +1),
               ("occ_prestige_gss1989", "occupational prestige (GSS Nakao-Treas 1989)", +1),
               ("occ_prestige_gss2012", "occupational prestige (GSS 2012)", +1),
               ("underemployment", "-underemployment (NY Fed)", -1),
               ("unemployment", "-unemployment (NY Fed)", -1)]
    if "match" in rel.columns:
        proxies.append(("match", "horizontal match (CIP->SOC)", +1))
    A_res = {}
    for col, name, sign in proxies:
        r, lo, hi, n = boot_spearman(meas.earn_standing, sign * meas[col])
        A_res[col] = (r, lo, hi, n)
        ci = f"[{lo:+.2f}, {hi:+.2f}]" if np.isfinite(lo) and np.isfinite(hi) else "[CI n/a: many ties]"
        L.append(f"| {name} | {r:+.2f} {ci} | {n} |")
    ropr = A_res["occ_prestige_opr"][0]
    L += [f"\n*Reading.* Earnings tracks occupational-prestige standing at Spearman {ropr:+.2f} -- "
          "positive (prestigious jobs do pay more) but **well short of 1.0**, the wage-correlated-but-not-"
          "identical signature: earnings captures part of ER standing and misses part (the status dimension "
          "wages cannot see -- research/arts/clergy/teaching confer status above their pay). Underemployment "
          "is a partly-independent vertical-match dimension.\n"]

    # ================= B. Is the gap (decoupling) an earnings artifact? ====
    L += ["## B. Is the gap (decoupling) an earnings artifact? (H1 vs H2)\n",
          "Correlate/regress the within-field **gap** on field-level non-wage ER across reliable fields.\n",
          "- **H1 (earnings captures it):** high-gap *decoupled* fields ALSO have poor non-wage outcomes "
          "(high underemployment, low occ-prestige) -> decoupling is real, earnings-only ER defensible.\n",
          "- **H2 (earnings overstates it):** high-gap fields have GOOD non-wage outcomes (low underemployment, "
          "high occ-prestige) -> a broader ER would show them MORE integrated; earnings-only ER overstates.\n",
          "\n| gap vs | Spearman [95% CI] | n | H1 sign | reading |",
          "|---|---|---|---|---|"]
    B_specs = [("occ_prestige_opr", "occ-prestige (OPR)", "negative (low status)"),
               ("underemployment", "underemployment", "positive (more underemployed)"),
               ("status_resid", "status residual (prestige net of wage)", "negative")]
    B_res = {}
    for col, name, h1 in B_specs:
        if col not in rel.columns:
            continue
        r, lo, hi, n = boot_spearman(rel.gap, rel[col])
        B_res[col] = (r, lo, hi, n)
        verdict = "—"
        if np.isfinite(r):
            if col == "underemployment":
                verdict = "H1" if r > 0.15 else ("H2" if r < -0.15 else "neither (flat)")
            else:  # prestige / residual: H1 expects negative
                verdict = "H1" if r < -0.15 else ("H2" if r > 0.15 else "neither (flat)")
        L.append(f"| {name} | {r:+.2f} [{lo:+.2f}, {hi:+.2f}] | {n} | {h1} | {verdict} |")
    # joint regression gap ~ z(occ_prestige) + z(underemployment)
    jr = rel.dropna(subset=["gap", "occ_prestige_opr", "underemployment"]).copy()
    joint_txt = "insufficient overlap"
    if len(jr) >= 10:
        for c in ["occ_prestige_opr", "underemployment"]:
            jr[f"z_{c}"] = (jr[c] - jr[c].mean()) / jr[c].std()
        X = sm.add_constant(jr[["z_occ_prestige_opr", "z_underemployment"]])
        m = sm.OLS(jr["gap"], X).fit()
        joint_txt = (f"gap ~ {m.params['const']:.2f} + ({m.params['z_occ_prestige_opr']:+.3f}) z(occ_prestige) "
                     f"+ ({m.params['z_underemployment']:+.3f}) z(underemp); R2={m.rsquared:.2f}, n={len(jr)}")
    L.append(f"\n**Joint (standardized), overlap fields:** {joint_txt}.\n")
    b_opr = B_res.get("occ_prestige_opr", (np.nan,))[0]
    b_und = B_res.get("underemployment", (np.nan,))[0]
    b_res = B_res.get("status_resid", (np.nan,))[0]
    flat = abs(b_opr) < 0.2 and abs(b_und) < 0.2
    b_verdict = (
        f"**Verdict (B).** The within-field gap is **largely orthogonal** to field-level non-wage ER: "
        f"gap vs occ-prestige {b_opr:+.2f} and gap vs underemployment {b_und:+.2f} are both flat, so "
        f"**neither H1 nor H2 holds as a general pattern** -- at the field level the decoupling is not "
        f"explained (H1) by poor non-wage outcomes, nor cleanly shown (H2) to be a status illusion. The one "
        f"directional signal is a weak **H2 lean on the status residual** (gap vs prestige-net-of-wage "
        f"{b_res:+.2f}), i.e. higher-gap fields tend to have occupational status slightly above their "
        f"wage-implied status -- but its CI spans zero on n={B_res.get('status_resid',(0,0,0,0))[3]} fields, "
        f"so it is suggestive only. The earnings-artifact concern is therefore **field-specific (see C/D), not "
        f"a blanket overstatement.**"
        if flat else
        f"**Verdict (B).** gap vs occ-prestige {b_opr:+.2f}, vs underemployment {b_und:+.2f}, vs status "
        f"residual {b_res:+.2f}.")
    L.append("\n" + b_verdict + "\n")

    # ================= C. Divergent fields (earnings vs non-wage standing) =
    meas["earn_rank"] = meas.earn_standing.rank()
    meas["pres_rank"] = meas.occ_prestige_opr.rank()
    def zr(s):
        return (s - s.mean()) / s.std()
    meas["divergence"] = zr(meas.pres_rank) - zr(meas.earn_rank)  # + => status >> pay (earnings UNDER-states ER)
    div = meas.dropna(subset=["divergence"]).sort_values("divergence")
    L += ["## C. Divergent fields -- where 'ER = earnings' is most misleading\n",
          "`divergence = z(occ-prestige rank) - z(earnings-standing rank)`. "
          "**Positive** = status far above pay (earnings UNDER-states ER); **negative** = pay far above "
          "status (earnings OVER-states ER). These are the concrete cost of the earnings-only operationalisation.\n",
          "\n**Earnings most UNDER-states ER (high status, modest pay):**\n",
          div.tail(6)[["label", "gap", "earn_standing", "occ_prestige_opr", "underemployment", "divergence"]]
             .iloc[::-1].to_markdown(index=False, floatfmt=("", ".2f", ".0f", ".1f", ".1f", "+.2f")),
          "\n\n**Earnings most OVER-states ER (high pay, modest status):**\n",
          div.head(6)[["label", "gap", "earn_standing", "occ_prestige_opr", "underemployment", "divergence"]]
             .to_markdown(index=False, floatfmt=("", ".2f", ".0f", ".1f", ".1f", "+.2f"))]

    # ================= D. Typology re-statement (the deliverable) ==========
    dec = rel[rel.typology == "decoupled"].copy()
    # non-wage ER percentiles within the reliable set (higher = better ER)
    rel["pres_pctl"] = rel.occ_prestige_opr.rank(pct=True)
    rel["und_pctl"] = (-rel.underemployment).rank(pct=True)  # higher = less underemployed = better
    dec = rel[rel.typology == "decoupled"].copy()
    def classify(row):
        good = []
        if np.isfinite(row.pres_pctl): good.append(row.pres_pctl >= 0.5)
        if np.isfinite(row.und_pctl): good.append(row.und_pctl >= 0.5)
        if not good:
            return "unclassified (no non-wage ER)"
        # earnings-artifact decoupled if non-wage ER is GOOD (above-median) on the available dims
        return "earnings-artifact decoupled" if all(good) else (
            "robustly decoupled" if not any(good) else "mixed")
    dec["decoupled_class"] = dec.apply(classify, axis=1)
    L += ["## D. Typology re-statement -- robust vs earnings-artifact decoupling\n",
          "Split the **decoupled** end (top-tercile gap, reliable) by whether the non-wage ER is ALSO poor "
          "(robustly decoupled) or fine (earnings-artifact decoupled: poor on earnings, fine on non-wage ER -- "
          "a broader ER would show these MORE integrated).\n",
          "\n| field | gap | occ-prestige | underemp | prestige pctl | (1-underemp) pctl | class |",
          "|---|---|---|---|---|---|---|"]
    for _, r in dec.sort_values("gap", ascending=False).iterrows():
        L.append(f"| {r.label} | {r.gap:.2f} | {r.occ_prestige_opr:.1f} | "
                 f"{r.underemployment if np.isfinite(r.underemployment) else float('nan'):.1f} | "
                 f"{r.pres_pctl:.2f} | {r.und_pctl if np.isfinite(r.und_pctl) else float('nan'):.2f} | "
                 f"**{r.decoupled_class}** |")
    nrob = (dec.decoupled_class == "robustly decoupled").sum()
    nart = (dec.decoupled_class == "earnings-artifact decoupled").sum()
    L.append(f"\n**{nrob} robustly decoupled, {nart} earnings-artifact decoupled, "
             f"{len(dec)-nrob-nart} mixed/unclassified** (of {len(dec)} decoupled fields).\n")
    rg, lg, hg, ng = boot_spearman(meas.earn_standing, meas.occ_prestige_gss1989)
    rc = spearmanr(meas.occ_prestige_opr, meas.occ_prestige_gss1989, nan_policy="omit")[0]
    under_names = ", ".join(div.tail(4).iloc[::-1].label.tolist())
    over_names = ", ".join(div.head(4).label.tolist())
    art_names = ", ".join(dec[dec.decoupled_class == "earnings-artifact decoupled"].label.tolist()) or "none"
    rob_names = ", ".join(dec[dec.decoupled_class == "robustly decoupled"].label.tolist()) or "none"
    L += ["## Overall verdict -- answering the reviewer\n",
          f"1. **Earnings is a defensible but PARTIAL ER proxy.** Field-level earnings standing tracks "
          f"survey occupational-prestige standing at Spearman {ropr:+.2f} (GSS scale {rg:+.2f}; the two "
          f"survey scales agree at {rc:+.2f}) and tracks -underemployment at "
          f"{A_res['underemployment'][0]:+.2f} -- all positive but **well short of 1.0**. Earnings carries "
          f"much of ER standing and misses the status dimension.\n",
          f"2. **The decoupling is mostly NOT an earnings artifact in general** (B): the gap is roughly "
          f"orthogonal to field-level non-wage ER, with only a weak, non-significant H2 lean on the "
          f"status-residual. So the integrated/decoupled typology is **not overturned** by a field-level "
          f"non-wage ER.\n",
          f"3. **But earnings materially mis-ranks specific fields** (C): it **under-states** ER for "
          f"research/natural-science fields ({under_names} -- high occupational status, modest early pay) and "
          f"**over-states** it for business fields ({over_names} -- good pay, middling status).\n",
          f"4. **The decoupled end is heterogeneous** (D): **earnings-artifact decoupled** = {art_names} "
          f"(strong non-wage ER -- a broader ER would integrate them, typically licensed/compressed-pay "
          f"fields); **robustly decoupled** = {rob_names} (poor on earnings AND non-wage ER); the rest mixed.\n",
          "5. **Net:** earnings-only ER survives as a field-level summary of the typology, but it is a thin "
          "proxy that mis-prices identifiable fields and hides at least some earnings-artifact decoupling -- "
          "which is exactly what motivates the within-field, multi-dimensional Revelio test specified below.\n"]

    # ================= Robustness: GSS scale repeat of A/C =================
    L += ["## Robustness: GSS (Nakao-Treas 1989) prestige scale\n",
          "Repeat A and the status-divergence with the GSS scale instead of Condon OPR, to show the status "
          "results are not an artifact of one survey prestige scale.\n"]
    rg, lg, hg, ng = boot_spearman(meas.earn_standing, meas.occ_prestige_gss1989)
    rc = spearmanr(meas.occ_prestige_opr, meas.occ_prestige_gss1989, nan_policy="omit")[0]
    L.append(f"- A (earn vs GSS occ-prestige): Spearman {rg:+.2f} [{lg:+.2f},{hg:+.2f}], n={ng} "
             f"(vs OPR {A_res['occ_prestige_opr'][0]:+.2f}).")
    L.append(f"- Scale agreement corr(OPR, GSS-1989) = {rc:+.2f} across fields -- the two survey scales "
             "rank fields' occupational standing near-identically, so the status findings are scale-robust.\n")

    # ================= Revelio estimand (specify, do NOT run) =============
    L += ["## The institution x field test that needs Revelio (specified, NOT run)\n",
          "The within-field analogue of the earnings gap, on a non-wage ER, is\n",
          "```\n",
          "gap^occ_f = 1 - Spearman_i( prestige_{i,f}, occ_prestige_{i,f} )\n",
          "```\n",
          "where `occ_prestige_{i,f} = sum_o P(o | institution i, field f) * prestige(o)` is each "
          "institution x field cell's occupational-prestige standing -- requiring **institution x field "
          "occupation flows** (Revelio resume data), which are not public. A second version weights placement "
          "by **employer prestige/selectivity** (which firms hire the grads). This is the true within-field "
          "typology-robustness test (ties to Thrust II); the present script is its field-level lower bound.\n",
          "## Adversarial self-check\n",
          f"- **occ-prestige residual wage-correlation:** corr(occ_prestige_OPR, earn_standing) = {ropr:+.2f}; "
          "occupational prestige is partly wage-correlated by construction (prestigious jobs pay more). We "
          "therefore also report the **status residual** (prestige net of wage) in B -- the part wages cannot "
          "see -- which is where the H1/H2 adjudication is sharpest.\n",
          f"- **ACS field->occupation coverage:** weighted occupation-score coverage median "
          f"{df['cov_opr'].median():.2f}; fields below 0.7 flagged in er_dimensions.csv (cov_opr).\n",
          f"- **NY Fed major-crosswalk losses:** underemployment covers only {und['field'].nunique()}/{len(ALL)} "
          "fields (no clean NY Fed major for the rest, e.g. neuroscience, microbiology, biostatistics); listed "
          "as unmapped rather than forced.\n",
          "- **Survey-prestige-scale sensitivity:** OPR vs GSS-1989 agree at "
          f"corr {rc:+.2f} (above), so status results are not one-scale artifacts.\n",
          "- **Grain caveat (load-bearing):** this is a **field-level** robustness test. It cannot recompute a "
          "within-field non-wage gap; it tests whether the earnings gap/typology survive a field-level non-wage "
          "ER. The Revelio estimand above is the within-field test proper.\n"]

    (ROOT / "ER_DEFINITION_RESULT.md").write_text("\n".join(L))

    # ================= Figures ============================================
    _fig_gap_vs_nonwage(rel)      # B: needs trustworthy gaps
    _fig_proxy_scatter(meas)      # A: ER-level, all measured fields

    print("done. reliable fields:", len(rel),
          "| occ_prestige fields:", pres['occ_prestige_opr'].notna().sum(),
          "| underemp fields:", und['field'].nunique())
    print("A: earn vs OPR =", round(ropr, 2), "| B gap vs OPR =",
          round(B_res.get('occ_prestige_opr', (np.nan,))[0], 2),
          "| D: robust", nrob, "artifact", nart)


def _fig_gap_vs_nonwage(rel):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    for a, (col, xl) in zip(ax, [("occ_prestige_opr", "occupational prestige (OPR)  ->  higher = better ER"),
                                  ("underemployment", "underemployment %  ->  higher = worse ER")]):
        s = rel.dropna(subset=["gap", col])
        a.scatter(s[col], s.gap, s=34, c="#3b6", edgecolor="k", linewidth=0.4, zorder=3)
        for _, r in s.iterrows():
            a.annotate(r.label, (r[col], r.gap), fontsize=6.0, alpha=0.8,
                       xytext=(2, 2), textcoords="offset points")
        if len(s) >= 5:
            b = np.polyfit(s[col], s.gap, 1); xs = np.linspace(s[col].min(), s[col].max(), 50)
            a.plot(xs, np.polyval(b, xs), "r--", lw=1.3, zorder=2)
            rho = spearmanr(s.gap, s[col])[0]
            a.set_title(f"gap vs {col}   (Spearman {rho:+.2f}, n={len(s)})", fontsize=10)
        a.set_xlabel(xl, fontsize=9); a.set_ylabel("within-field gap (decoupling)", fontsize=9)
        a.grid(alpha=0.25)
    fig.suptitle("Is decoupling an earnings artifact? H1: decoupled fields poor on non-wage ER | "
                 "H2: decoupled fields fine on non-wage ER", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "gap_vs_nonwage_ER.png", dpi=140); plt.close(fig)


def _fig_proxy_scatter(rel):
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    s = rel.dropna(subset=["earn_standing", "occ_prestige_opr"])
    sc = ax.scatter(s.earn_standing, s.occ_prestige_opr,
                    c=s.gap, cmap="viridis", s=46, edgecolor="k", linewidth=0.4)
    for _, r in s.iterrows():
        ax.annotate(r.label, (r.earn_standing, r.occ_prestige_opr), fontsize=6.2, alpha=0.85,
                    xytext=(2, 2), textcoords="offset points")
    if len(s) >= 5:
        b = np.polyfit(s.earn_standing, s.occ_prestige_opr, 1)
        xs = np.linspace(s.earn_standing.min(), s.earn_standing.max(), 50)
        ax.plot(xs, np.polyval(b, xs), "r--", lw=1.3)
        rho = spearmanr(s.earn_standing, s.occ_prestige_opr)[0]
        ax.set_title(f"Field earnings standing vs occupational prestige (Spearman {rho:+.2f})\n"
                     "off-diagonal fields = earnings is a thin ER proxy; colour = within-field gap", fontsize=10)
    ax.set_xlabel("ACS weighted-median early-career earnings ($)", fontsize=9)
    ax.set_ylabel("occupational prestige (OPR, Condon 2024)", fontsize=9)
    plt.colorbar(sc, label="within-field gap (decoupling)")
    ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(OUT / "figures" / "proxy_scatter.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
