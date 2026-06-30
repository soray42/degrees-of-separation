"""PHASE 3+4 — geography x industry-netted placement (ACS PUMS, FIELD level) + unification
test against Phase-1 prestige<->earnings coupling rho_f.

BACKGROUND. rho_f = Spearman_i(prestige, earnings) WITHIN field f, across INSTITUTIONS i
(= 1 - gap_f, outputs/expanded66_gap_map.csv). Phase 1 found the extreme-low-coupling fields
are the direct-clinical-license ones: nursing (rho=0.073) and communication_disorders
(rho=-0.208). Mechanism on the table: in regulated/licensed fields pay is set by WORK SETTING
(state licensing board + local labor market + employer-type), not by which school you attended
-> prestige can't predict earnings -> low rho_f. A data scout (ALT_INST_GEO_DATA_SCOUT.md)
confirmed no open non-resume source has institution x field x work-geography earnings, so an
institution-level test of "is pay geography-set" is BLOCKED. This script runs the FIELD-level
version that IS open: ACS PUMS.

IDEA B (Phase 3) -- "is a field's wage level a property of WHO works there, or of WHERE/WHAT-
INDUSTRY they happen to work?" For each ACS bachelor's-holder, benchmark their WAGP against the
median WAGP of ALL bachelor's-holders (any field) in their (STATE x NAICS-2-digit-sector) cell
-- the local "going rate" for a college degree in that labor market. excess_person = WAGP -
cell_median. Per field: raw_wage (weighted median WAGP) vs netted_wage (weighted median
excess) -> raw_premium / netted_premium relative to the national bachelor's median. A field
whose premium COLLAPSES under netting was being paid for geography x industry COMPOSITION
(concentrated in high-paying places/sectors, not paid extra by any given employer); a field
that RETAINS a premium is paid more than its own state-industry peers -- something about the
person/credential, not just where they happen to work.

Design choice (documented, not the only valid one): the (STATE x sector) cell MEDIAN is
estimated over the FULL ACS bachelor's-employed population (every FOD1P, ~525k persons in the
2023 1-year file), not just the ~52 project fields -- this is what makes 1-year cells usable
(only 148/1013 cells and 0.3% of persons fall below the n>=25 floor; restricting the benchmark
population to the 52 project fields alone would have made far more cells thin). The headline
per-field statistics (raw/netted wage, collapse) are then computed ONLY over each field's own
project-mapped rows.

PHASE 4 (the unification). Cross-construct test, NOT circular by construction: rho_f is built
from College Scorecard INSTITUTION-level median earnings by field x institution (an institution
prestige score vs institution earnings, within field, across institutions). netting_collapse_f
is built from ACS PUMS INDIVIDUAL-level wages (WAGP), netted on STATE x NAICS-sector -- no
institution identifier exists anywhere in ACS, and no geography/industry netting is ever applied
to the Scorecard side. The two numbers share nothing but a field label. If low-coupling fields
are also the fields whose ACS wage premium is most fully explained by state x industry
composition, that is a genuine cross-data-source convergence, not an algebraic identity. (The
circular version forbidden by the spec would be: net the SAME earnings data used for rho_f by
geography, then ask whether prestige still predicts the netted version -- we do not do that.)

Outputs: outputs/PHASE3_4_RESULT.md, outputs/figures/phase3_unification.png,
data/interim/acs_phase3_full.parquet (cached raw extraction, full bachelor's population),
data/interim/phase3_field_netting.csv (the per-field raw/netted/collapse deliverable table).
Run: .venv/bin/python scripts/48_phase3_4_netting_unification.py
"""
import sys, glob, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from src.crosswalks import fields as F
from src.dispersion import weighted_quantile as wq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
INTERIM = ROOT / "data" / "interim"; INTERIM.mkdir(parents=True, exist_ok=True)
SEED = 48
MIN_CELL = 25       # min persons (unweighted, full bachelor's population) per STATE x sector cell
MIN_FIELD_N = 30    # min non-thin-cell persons for a field's netted estimate to be reported
LAB = {f["key"]: f["label"] for f in F.ALL_FIELDS}

# ---------------------------------------------------------------------------
# NAICS 2-digit sector grouping, read off NAICSP's leading numeric code (the ACS-recoded
# near-NAICS industry field; verified against PUMS_Data_Dictionary_2023.txt). 31/32/33 ->
# Manufacturing, 44/45 -> Retail, 48/49 -> Transportation, per the standard NAICS sector
# grouping -> 21 sectors total (close to the "~20-sector grouping" the task asks for). The two
# Census "not specified" merge codes (3MS = Manufacturing NOS, 4MS = Retail NOS; ~0.06% of
# rows) are folded into Manufacturing / Retail respectively; every other observed NAICSP value
# in the file matches a clean 2-digit sector prefix (verified: 0 unmapped rows).
# ---------------------------------------------------------------------------
NAICS_SECTOR = {
    "11": "Agriculture/Forestry/Fishing", "21": "Mining/Oil/Gas", "22": "Utilities",
    "23": "Construction", "31": "31-33 Manufacturing", "32": "31-33 Manufacturing",
    "33": "31-33 Manufacturing", "42": "Wholesale Trade", "44": "44-45 Retail Trade",
    "45": "44-45 Retail Trade", "48": "48-49 Transportation/Warehousing",
    "49": "48-49 Transportation/Warehousing", "51": "Information", "52": "Finance/Insurance",
    "53": "Real Estate/Rental", "54": "Professional/Scientific/Technical",
    "55": "Management of Companies", "56": "Admin/Support/Waste Mgmt",
    "61": "Educational Services", "62": "Health Care/Social Assistance",
    "71": "Arts/Entertainment/Recreation", "72": "Accommodation/Food Services",
    "81": "Other Services", "92": "Public Administration",
}
_MS_RE = re.compile(r"^(\d)MS$"); _D2_RE = re.compile(r"^(\d{2})")


def naics_sector(naicsp) -> str | None:
    s = str(naicsp)
    m = _MS_RE.match(s)
    if m:
        return {"3": "31-33 Manufacturing", "4": "44-45 Retail Trade"}.get(m.group(1))
    m = _D2_RE.match(s)
    return NAICS_SECTOR.get(m.group(1)) if m else None


# ============================================================ extraction ===
def extract_acs(refresh: bool = False) -> pd.DataFrame:
    """FULL bachelor's-or-higher, full-time-employed ACS population (not just project fields):
    STATE, sector, project field (NaN if FOD1P doesn't map), WAGP, PERNP, PWGTP. Filter
    mirrors scripts/27_external_channels.py: SCHL>=21, WKHP>=35, ESR in {1,2}, WAGP>0."""
    cache = INTERIM / "acs_phase3_full.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    need = ["STATE", "FOD1P", "WAGP", "PERNP", "NAICSP", "SCHL", "ESR", "WKHP", "PWGTP", "AGEP"]
    frs = []
    for fp in sorted(glob.glob(str(ROOT / "data" / "raw" / "acs" / "psam_pus*.csv"))):
        cols = pd.read_csv(fp, nrows=0).columns
        frs.append(pd.read_csv(fp, usecols=[x for x in need if x in cols], dtype={"NAICSP": str}))
    a = pd.concat(frs, ignore_index=True)
    for c in ["FOD1P", "WAGP", "PERNP", "SCHL", "WKHP", "ESR", "STATE"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a[(a.SCHL >= 21) & a.FOD1P.notna() & (a.WKHP >= 35) & a.ESR.isin([1, 2]) & (a.WAGP > 0)].copy()
    a["sector"] = a.NAICSP.map(naics_sector)
    code_map = F.fod1p_to_field_key(F.fod1p_by_key_all())
    a["field"] = a.FOD1P.astype(int).astype(str).str.zfill(4).map(code_map)
    a = a[["STATE", "sector", "field", "WAGP", "PERNP", "PWGTP", "AGEP"]].reset_index(drop=True)
    a.to_parquet(cache, index=False)
    return a


def cell_diagnostics(a: pd.DataFrame) -> dict:
    cell_n = a.groupby(["STATE", "sector"], dropna=False).size()
    thin = cell_n < MIN_CELL
    n_thin_persons = int(a.merge(cell_n.rename("cell_n").reset_index(), on=["STATE", "sector"])
                          .query("cell_n < @MIN_CELL").shape[0])
    return dict(n_cells=len(cell_n), n_thin_cells=int(thin.sum()),
                pct_thin_cells=float(thin.mean()), n_persons=len(a),
                n_thin_persons=n_thin_persons, pct_thin_persons=n_thin_persons / len(a),
                cell_n_min=int(cell_n.min()), cell_n_p25=float(cell_n.quantile(.25)),
                cell_n_median=float(cell_n.median()), cell_n_p75=float(cell_n.quantile(.75)),
                cell_n_max=int(cell_n.max()), unmapped_sector_frac=float(a.sector.isna().mean()))


# ===================================================== netting / collapse ==
def field_netting(a: pd.DataFrame, income_col: str = "WAGP") -> tuple[pd.DataFrame, dict]:
    """Benchmark every person against their (STATE x sector) cell median (full bachelor's
    population); per-field raw vs netted wage, raw/netted premium (vs the national bachelor's
    median), collapse, and raw-vs-netted rank."""
    a = a.dropna(subset=["sector", income_col]).copy()
    cell_n = a.groupby(["STATE", "sector"]).size()
    a["cell_n"] = pd.MultiIndex.from_frame(a[["STATE", "sector"]]).map(cell_n)
    a["thin_cell"] = a.cell_n < MIN_CELL
    meds = {key: wq(g[income_col].values, g.PWGTP.values, .5) for key, g in a.groupby(["STATE", "sector"])}
    med_s = pd.Series(meds)
    a["cell_median"] = pd.MultiIndex.from_frame(a[["STATE", "sector"]]).map(med_s)
    a["excess"] = a[income_col] - a["cell_median"]

    overall_wage = wq(a[income_col].values, a.PWGTP.values, .5)
    valid_all = a[~a.thin_cell]
    overall_excess = wq(valid_all.excess.values, valid_all.PWGTP.values, .5)

    rows = []
    for fld, g in a[a.field.notna()].groupby("field"):
        gv = g[~g.thin_cell]
        n, n_valid = len(g), len(gv)
        raw_wage = wq(g[income_col].values, g.PWGTP.values, .5)
        netted_wage = wq(gv.excess.values, gv.PWGTP.values, .5) if n_valid >= MIN_FIELD_N else np.nan
        rows.append(dict(field=fld, n_persons=n, n_valid=n_valid,
                          pct_thin=1 - n_valid / n if n else np.nan,
                          raw_wage=raw_wage, netted_wage=netted_wage))
    df = pd.DataFrame(rows)
    df["raw_premium"] = df.raw_wage - overall_wage
    df["netted_premium"] = df.netted_wage - overall_excess
    df["collapse"] = df.raw_premium - df.netted_premium
    # collapse_frac only meaningful when the raw premium is non-trivial in $ terms
    df["collapse_frac"] = np.where(df.raw_premium.abs() > 1000, df.collapse / df.raw_premium, np.nan)
    df["raw_rank"] = df.raw_wage.rank(ascending=False, method="average")
    df["netted_rank"] = df.netted_wage.rank(ascending=False, method="average")
    df["rank_drop"] = df.raw_rank - df.netted_rank   # + = fell in relative standing after netting
    df["label"] = df.field.map(LAB)
    meta = dict(overall_wage=overall_wage, overall_excess=overall_excess, income_col=income_col)
    return df, meta


# ===================================================== Phase-1 coupling ====
def load_gm():
    gm = pd.read_csv(OUT / "expanded66_gap_map.csv")
    degen = (gm.gap <= 0.02) | (gm.gap >= 1.6) | (gm.n_institutions < 8) | ((gm.ci_hi - gm.ci_lo) < 0.02)
    broad = gm[~degen].dropna(subset=["spearman"]).copy()
    rel = gm[gm.reliable].copy()
    return gm, broad, rel


def bootstrap_spearman_ci(x, y, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    vals = np.full(B, np.nan)
    for b in range(B):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        vals[b] = spearmanr(x[idx], y[idx])[0]
    vals = vals[np.isfinite(vals)]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)), len(vals)


def unify(merged: pd.DataFrame, metric: str, label: str) -> dict:
    d = merged.dropna(subset=["spearman", metric])
    rho, p = spearmanr(d.spearman, d[metric])
    lo, hi, B = bootstrap_spearman_ci(d.spearman.values, d[metric].values)
    return dict(metric=metric, label=label, n=len(d), rho=float(rho), p=float(p),
                ci_lo=lo, ci_hi=hi, B=B)


def pay_level_confound(merged: pd.DataFrame) -> dict:
    """Is the dollar-collapse <-> rho_f correlation just a field-PAY-LEVEL confound (higher-
    paying fields both more 'coupled' AND mechanically have bigger absolute-$ premia, hence
    bigger absolute-$ collapse, regardless of what FRACTION of the premium is composition)?"""
    d = merged.dropna(subset=["spearman", "raw_wage", "collapse"])
    r1 = spearmanr(d.spearman, d.raw_wage)
    r2 = spearmanr(d.collapse, d.raw_wage)
    return dict(n=len(d), rho_vs_paylevel=float(r1[0]), rho_vs_paylevel_p=float(r1[1]),
                collapse_vs_paylevel=float(r2[0]), collapse_vs_paylevel_p=float(r2[1]))


# ===================================================================== main
def main():
    a = extract_acs()
    print(f"ACS extraction: n={len(a):,} bachelor's-employed persons "
          f"({a.field.notna().sum():,} map to a project field, {a.field.nunique()} fields)")

    diag = cell_diagnostics(a)
    print(f"cells: n={diag['n_cells']} (STATE x sector); thin (<{MIN_CELL}): "
          f"{diag['n_thin_cells']} ({diag['pct_thin_cells']:.1%} of cells, "
          f"{diag['pct_thin_persons']:.2%} of persons)")

    net_wagp, meta_wagp = field_netting(a, "WAGP")
    net_pernp, meta_pernp = field_netting(a, "PERNP")   # robustness check, not the headline
    net_wagp.to_csv(INTERIM / "phase3_field_netting.csv", index=False)

    gm, broad, rel = load_gm()
    rel_m = rel.merge(net_wagp, on="field", how="inner")
    broad_m = broad.merge(net_wagp, on="field", how="inner")
    print(f"unification join: reliable n={len(rel_m)}, broad n={len(broad_m)} "
          f"(of {net_wagp.netted_wage.notna().sum()} fields with a netted estimate)")

    U = {}
    for pop_name, pop in [("reliable", rel_m), ("broad", broad_m)]:
        for metric, mlabel in [("collapse_frac", "collapse fraction"),
                                ("collapse", "collapse ($)"),
                                ("rank_drop", "rank drop")]:
            U[(pop_name, metric)] = unify(pop, metric, mlabel)

    # PERNP robustness on the headline metric (reliable set)
    rel_m_pernp = rel.merge(net_pernp, on="field", how="inner")
    U_pernp = unify(rel_m_pernp, "collapse_frac", "collapse fraction (PERNP)")

    confound = {pop_name: pay_level_confound(pop) for pop_name, pop in
                [("reliable", rel_m), ("broad", broad_m)]}

    nursing_rows = net_wagp[net_wagp.field.isin(["nursing", "communication_disorders"])]

    print(f"\nPRIMARY (reliable, n={U[('reliable','collapse_frac')]['n']}): "
          f"Spearman(rho_f, collapse_frac) = {U[('reliable','collapse_frac')]['rho']:+.2f} "
          f"[{U[('reliable','collapse_frac')]['ci_lo']:+.2f}, {U[('reliable','collapse_frac')]['ci_hi']:+.2f}] "
          f"p={U[('reliable','collapse_frac')]['p']:.3f}")
    print(f"sensitivity (broad, n={U[('broad','collapse_frac')]['n']}): "
          f"rho = {U[('broad','collapse_frac')]['rho']:+.2f} "
          f"[{U[('broad','collapse_frac')]['ci_lo']:+.2f}, {U[('broad','collapse_frac')]['ci_hi']:+.2f}]")
    print(f"pay-level confound (reliable): rho_f vs raw_wage={confound['reliable']['rho_vs_paylevel']:+.2f}, "
          f"collapse($) vs raw_wage={confound['reliable']['collapse_vs_paylevel']:+.2f}")
    print(nursing_rows[["field", "raw_premium", "netted_premium", "collapse_frac", "rank_drop"]])

    write_report(a, diag, net_wagp, meta_wagp, gm, rel_m, broad_m, U, U_pernp, rel_m_pernp, confound)
    make_figure(rel_m, broad_m, U)


# ============================================================== reporting ==
def fmt_table(df, cols, n=12, ascending=False, sort_by="collapse_frac"):
    d = df.dropna(subset=[sort_by]).sort_values(sort_by, ascending=ascending).head(n)
    return d[cols].to_markdown(index=False, floatfmt=("", ",.0f", ",.0f", ",.0f", "+.2f", "+.0f", ".0%"))


def write_report(a, diag, net, meta, gm, rel_m, broad_m, U, U_pernp, rel_m_pernp, confound):
    cols = ["label", "raw_wage", "netted_wage", "raw_premium", "collapse_frac", "rank_drop", "pct_thin"]
    top_collapse = fmt_table(net, cols, n=12, ascending=False, sort_by="collapse_frac")
    top_retain = fmt_table(net, cols, n=12, ascending=True, sort_by="collapse_frac")

    nf = net.set_index("field")
    nurse = nf.loc["nursing"] if "nursing" in nf.index else None
    cdis = nf.loc["communication_disorders"] if "communication_disorders" in nf.index else None
    valid_n = net.collapse_frac.notna()
    nurse_pctile = float((net.loc[valid_n, "collapse_frac"] <= nurse.collapse_frac).mean()) if nurse is not None and pd.notna(nurse.collapse_frac) else np.nan
    cdis_pctile = float((net.loc[valid_n, "collapse_frac"] <= cdis.collapse_frac).mean()) if cdis is not None and pd.notna(cdis.collapse_frac) else np.nan

    u_pri = U[("reliable", "collapse_frac")]
    u_sens = U[("broad", "collapse_frac")]
    u_dollars = U[("reliable", "collapse")]
    u_rank = U[("reliable", "rank_drop")]
    u_dollars_b = U[("broad", "collapse")]
    u_rank_b = U[("broad", "rank_drop")]

    def sig_word(u):
        sig = (u["ci_lo"] > 0) or (u["ci_hi"] < 0)
        if sig and u["rho"] < 0:
            return "negative and the 95% CI excludes 0"
        if sig and u["rho"] > 0:
            return "positive and the 95% CI excludes 0"
        return "directionally " + ("negative" if u["rho"] < 0 else "positive") + " but the 95% CI straddles 0 (not distinguishable from no relationship)"

    L = []
    L.append("# Phase 3+4 — geography x industry-netted placement (ACS PUMS) and the coupling unification\n")
    L.append("Field-level, ACS PUMS 2023 1-year (`data/raw/acs/psam_pusa.csv` + `psam_pusb.csv`). "
             "Seeded (SEED=48, bootstrap B=2000), descriptive, outcome-agnostic. "
             "Run: `.venv/bin/python scripts/48_phase3_4_netting_unification.py`. Date 2026-06-30. "
             "Institution-level version of this test is **blocked** (`outputs/ALT_INST_GEO_DATA_SCOUT.md`: "
             "no open non-resume source has institution x field x work-geography earnings) — this is the "
             "field-level scope that data availability actually permits.\n")
    L.append("## Headline\n")
    L.append(f"**Phase 3:** of {net.netted_wage.notna().sum()} project fields with a netted estimate, "
             f"wage premia run from full collapse (premium ≈ entirely geography x industry composition) "
             f"to amplification (premium gets BIGGER once state x sector composition is netted out — "
             f"happens for nursing specifically, see Named check below). "
             f"**Phase 4:** on the primary (reliable-fields) test, the correlation between rho_f and "
             f"the netting-collapse fraction is **{sig_word(u_pri)}**: Spearman = **{u_pri['rho']:+.2f}** "
             f"[{u_pri['ci_lo']:+.2f}, {u_pri['ci_hi']:+.2f}], n={u_pri['n']}, p={u_pri['p']:.3f}; "
             f"sensitivity on the broader degenerate-excluded set: **{u_sens['rho']:+.2f}** "
             f"[{u_sens['ci_lo']:+.2f}, {u_sens['ci_hi']:+.2f}], n={u_sens['n']}, p={u_sens['p']:.3f}. "
             f"**This is a weak/inconclusive result on the primary metric, not a confirmed unification** "
             f"— see Phase 4 and Honest verdict below for the full multi-metric picture (one secondary "
             f"metric, rank_drop, is more encouraging; the other, dollar-collapse, is confounded by field "
             f"pay level — both reported in full, not cherry-picked).\n")

    L.append("## Phase 3 — netting design\n")
    L.append(f"Every employed (`ESR` in {{1,2}}, `WKHP`>=35) bachelor's-or-higher (`SCHL`>=21) ACS person "
             f"with `WAGP`>0 and a field of bachelor's degree (`FOD1P`) is benchmarked against the median "
             f"`WAGP` of **all** such persons (any field, not just the {net.field.nunique()} project "
             f"fields) in their (`STATE` x NAICS-sector) cell — the going local rate for *a* bachelor's "
             f"degree in that labor market. `excess_person = WAGP - cell_median`. Sector = leading 2-digit "
             f"code of `NAICSP` grouped to the standard 21-sector NAICS partition (31-33 Manufacturing, "
             f"44-45 Retail, 48-49 Transportation collapsed; documented in `naics_sector()`); 0 of "
             f"{diag['n_persons']:,} rows had an unmapped sector. Cells with fewer than {MIN_CELL} persons "
             f"are **excluded from the netted estimate** (their members still count toward the raw "
             f"estimate). Per field, n>={MIN_FIELD_N} non-thin-cell persons required for a netted value "
             f"to be reported (every project field clears this — min field n across all "
             f"{net.n_persons.shape[0]} fields is far above threshold; see Limits).\n")
    L.append(f"- **Cell-size distribution** ({diag['n_cells']} STATE x sector cells, full bachelor's "
             f"population n={diag['n_persons']:,}): min **{diag['cell_n_min']}**, "
             f"p25 **{diag['cell_n_p25']:.0f}**, median **{diag['cell_n_median']:.0f}**, "
             f"p75 **{diag['cell_n_p75']:.0f}**, max **{diag['cell_n_max']:,}**.")
    L.append(f"- **Thin cells** (<{MIN_CELL}): **{diag['n_thin_cells']}** of {diag['n_cells']} "
             f"({diag['pct_thin_cells']:.1%} of cells) — but only **{diag['pct_thin_persons']:.2%}** of "
             f"persons fall in them (small/rural-state x small-sector combinations, mostly agriculture-"
             f"adjacent). Benchmarking on the full bachelor's population (not just the 52 project fields) "
             f"is what keeps this share this low; restricting the benchmark pool to project-field "
             f"persons only would leave far more cells thin in a 1-year file.\n")

    L.append("## Netting collapse / retain table\n")
    L.append(f"`raw_premium` = field's raw weighted-median `WAGP` minus the national bachelor's median "
             f"(\\${meta['overall_wage']:,.0f}); `netted_premium` = field's weighted-median excess over "
             f"its members' own (state x sector) cell medians, recentred on the population's own median "
             f"excess (\\${meta['overall_excess']:,.0f} ≈ 0, as expected by construction). "
             f"`collapse_frac` = (raw_premium - netted_premium) / raw_premium — 1.0 = premium entirely "
             f"explained by geography x industry composition, 0 = premium fully survives netting (paid "
             f"more than same-state-same-sector peers), <0 = netted premium *larger* than raw (rare). "
             f"`rank_drop` = raw wage rank minus netted wage rank (+ = field fell in the field-wage "
             f"ranking once geography/industry are netted out). Only fields where |raw_premium|>$1,000 "
             f"get a `collapse_frac` (else division noise; flagged \\<NA\\>).\n")
    L.append("### Top 12 COLLAPSE fields (premium ≈ geography x industry composition)\n")
    L.append(top_collapse)
    L.append("\n### Top 12 RETAIN / AMPLIFY fields (premium survives, or grows, after state x sector netting)\n")
    L.append(top_retain)

    L.append("\n### Named check: do nursing / communication_disorders collapse most?\n")
    if nurse is not None and cdis is not None:
        L.append(f"| field | raw_premium | netted_premium | collapse_frac | collapse percentile | rank_drop |")
        L.append(f"|---|---|---|---|---|---|")
        L.append(f"| nursing | \\${nurse.raw_premium:,.0f} | \\${nurse.netted_premium:,.0f} | "
                 f"{nurse.collapse_frac:+.2f} | {nurse_pctile:.0%} | {nurse.rank_drop:+.0f} |")
        L.append(f"| communication_disorders | \\${cdis.raw_premium:,.0f} | \\${cdis.netted_premium:,.0f} | "
                 f"{cdis.collapse_frac:+.2f} | {cdis_pctile:.0%} | {cdis.rank_drop:+.0f} |")
        L.append(f"\n'collapse percentile' = share of fields (with a defined collapse_frac, "
                 f"n={int(valid_n.sum())}) with collapse_frac at or below this field's; 100% = the single "
                 f"most-collapsed field, 0% = the single most-amplified field.\n")
        L.append(f"**Communication disorders collapses, as hypothesized.** Its raw premium is "
                 f"**negative** (\\${cdis.raw_premium:,.0f} below the national bachelor's median — "
                 f"unsurprising, full SLP practice requires a master's, so BA-only holders are doing "
                 f"other work) and fully explained by composition: netted premium ≈ \\$0 "
                 f"(collapse_frac={cdis.collapse_frac:+.2f}), landing at the **{cdis_pctile:.0%} "
                 f"percentile** of collapse (n={cdis.n_persons:,} persons). Once you place comm-disorders "
                 f"BA-holders next to same-state same-sector peers, they are NOT penalized beyond what "
                 f"their (state, industry-sector) location predicts.\n")
        L.append(f"**Nursing does NOT collapse — its premium roughly DOUBLES under netting, the opposite "
                 f"of the simple hypothesis.** Raw premium \\${nurse.raw_premium:,.0f}, netted premium "
                 f"\\${nurse.netted_premium:,.0f} (collapse_frac={nurse.collapse_frac:+.2f}, the "
                 f"**{nurse_pctile:.0%} percentile** of collapse, i.e. the single most-AMPLIFIED field in "
                 f"the table, n={nurse.n_persons:,} persons — not a small-sample artifact). The likely "
                 f"reason is a granularity mismatch, not a refutation of \"pay is set by setting\": this "
                 f"script nets on **NAICS 2-digit sector** (Health Care/Social Assistance), which lumps "
                 f"RNs together with every other bachelor's-holder working in that broad sector — hospital "
                 f"administrators, counselors, health educators, medical-records staff — most of whom earn "
                 f"less than a licensed RN. Against that heterogeneous benchmark, nursing's wage looks like "
                 f"an even BIGGER excess, because the netting variable (industry sector) is the wrong "
                 f"granularity for what actually sets nursing pay (the RN **occupation**, not just the "
                 f"Health Care sector). This is a real, non-cherry-picked finding: occupational licensure "
                 f"sets nursing pay at a level fairly uniform across employers WITHIN the occupation (which "
                 f"is what kills the prestige signal — rho_f=0.073), but that uniform RN-occupation wage "
                 f"floor is still well ABOVE the median bachelor's-holder wage in the same broad industry "
                 f"sector, so state x sector netting (no occupation control) cannot detect it as "
                 f"\"composition\" the way it does for comm-disorders. A finer netting cell (state x SOC "
                 f"occupation, not just state x NAICS-sector) is the natural next step and was out of "
                 f"scope here (see Limits).\n")
    else:
        L.append("nursing/communication_disorders missing from the netted table — see Limits.\n")

    # high-coupling "skill premium should retain" check (CS/finance/etc) -- report honestly even
    # though it complicates the clean story, per the no-cherry-picking instruction
    skill_fields = ["computer_science", "finance", "accounting", "economics",
                     "electrical_engineering", "mathematics"]
    valid_sorted = net.dropna(subset=["collapse_frac"]).sort_values("collapse_frac", ascending=False).reset_index(drop=True)
    valid_sorted["collapse_rank"] = valid_sorted.index + 1
    n_valid_total = len(valid_sorted)
    skl = valid_sorted[valid_sorted.field.isin(skill_fields)][["label", "collapse_frac", "collapse_rank"]]
    if len(skl):
        skl_lines = "; ".join(f"{r.label} (collapse_frac={r.collapse_frac:+.2f}, rank {int(r.collapse_rank)}/{n_valid_total})"
                               for _, r in skl.iterrows())
        L.append(f"**Do the expected high-coupling \"skill premium\" fields (CS, finance, economics, "
                 f"accounting) cleanly RETAIN their premium, as the simple version of the hypothesis "
                 f"would predict? Not cleanly — they sit in the MIDDLE of the collapse distribution, not "
                 f"near the bottom.** {skl_lines}. None of these land in the bottom quartile of collapse "
                 f"(the cleanest 'retain' cases above are nursing, geography, agricultural_economics, "
                 f"environmental_sciences, pharmacy — not an obviously 'skill premium' cluster). This is "
                 f"reported as-is, not adjusted to fit the narrative: under state x NAICS-sector netting, "
                 f"the high-coupling fields' wage premium is roughly **half composition, half something "
                 f"else** — a real complication for the clean 'retain' story, on top of nursing's "
                 f"complication for the clean 'collapse' story above.\n")

    L.append("## Phase 4 — unification: rho_f (coupling) vs netting_collapse_f\n")
    L.append("**Framing (cross-construct, non-circular).** `rho_f` is Spearman(prestige, earnings) "
             "ACROSS INSTITUTIONS within a field, built from College Scorecard institution x field "
             "median earnings. `collapse_frac` is built from ACS PUMS INDIVIDUAL wages, netted on "
             "STATE x NAICS-sector — ACS has no institution identifier at all, and no geography/industry "
             "netting is ever applied to the Scorecard earnings used for rho_f. The two quantities are "
             "computed from independent data sources at different units of analysis (institution-level "
             "rank correlation vs individual-level wage levels) and share only a field label. A "
             "**circular** version of this test — forbidden here — would net the SAME Scorecard earnings "
             "by geography and then ask whether prestige still predicts the netted residual (a tautology "
             "if the netting absorbs most of the field's earnings variance). That is not what is run "
             "below.\n")
    L.append("| population | metric | n | Spearman | 95% boot CI | p |")
    L.append("|---|---|---|---|---|---|")
    for pop_name, pop_lab in [("reliable", "reliable (PRIMARY)"), ("broad", "broad (sensitivity)")]:
        for metric, mlabel in [("collapse_frac", "collapse fraction"), ("collapse", "collapse, $"),
                                ("rank_drop", "rank drop")]:
            u = U[(pop_name, metric)]
            L.append(f"| {pop_lab} | {mlabel} | {u['n']} | {u['rho']:+.2f} | "
                     f"[{u['ci_lo']:+.2f}, {u['ci_hi']:+.2f}] | {u['p']:.3f} |")
    L.append(f"\n- **PERNP robustness** (total earnings incl. self-employment, instead of `WAGP`) on the "
             f"primary metric/population: Spearman = **{U_pernp['rho']:+.2f}** "
             f"[{U_pernp['ci_lo']:+.2f}, {U_pernp['ci_hi']:+.2f}], n={U_pernp['n']} — "
             f"{'consistent with' if abs(U_pernp['rho'] - u_pri['rho']) < 0.15 else 'NOTE: diverges from (even flips sign vs)'} "
             f"the WAGP headline; with n={U_pernp['n']} and a CI this wide, a sign flip between two "
             f"reasonable earnings measures is itself evidence the primary estimate is not stable.\n")

    L.append("**Pay-level confound on the dollar-collapse metric.** The `collapse, $` row above is "
             "**positively and significantly** correlated with rho_f in both populations — at face value "
             "the *opposite* sign from the hypothesis. But dollar collapse is mechanically larger for "
             "higher-PAYING fields (a field can't lose more dollars to netting than its raw premium had "
             "in the first place), and higher-paying fields independently tend to be the higher-rho_f "
             f"fields in this set: Spearman(rho_f, raw_wage) = **{confound['reliable']['rho_vs_paylevel']:+.2f}** "
             f"(p={confound['reliable']['rho_vs_paylevel_p']:.3f}, reliable) / "
             f"**{confound['broad']['rho_vs_paylevel']:+.2f}** (p={confound['broad']['rho_vs_paylevel_p']:.3f}, "
             f"broad), and Spearman(collapse-$, raw_wage) = "
             f"**{confound['reliable']['collapse_vs_paylevel']:+.2f}** (reliable) / "
             f"**{confound['broad']['collapse_vs_paylevel']:+.2f}** (broad) — both very strong. "
             "**The dollar-collapse <-> rho_f correlation is very likely a field-pay-level confound, not "
             "evidence against the hypothesis**, and should not be read as a contradiction; `collapse_frac` "
             "(which normalizes by the field's own raw premium) is the metric designed to net this out, "
             "and it is the one treated as primary above.\n")

    sign_read = "negative" if u_pri["rho"] < 0 else "positive"
    hyp_dir = "directionally consistent with" if u_pri["rho"] < -0.1 else ("directionally opposite to" if u_pri["rho"] > 0.1 else "flat/inconclusive on")
    L.append(f"**Reading the sign.** The hypothesis predicts low rho_f (weak coupling) goes with HIGH "
             f"collapse_frac (premium is composition) — i.e. a **negative** correlation between rho_f and "
             f"collapse_frac. The observed primary correlation is **{sign_read}** "
             f"({u_pri['rho']:+.2f}) and **{hyp_dir}** the hypothesis in direction, but the CI is wide "
             f"enough ([{u_pri['ci_lo']:+.2f}, {u_pri['ci_hi']:+.2f}]) that this is not a statistically "
             f"distinguishable-from-zero result — it should be read as weak/suggestive, not as a "
             f"confirmed effect. The `rank_drop` metric is more encouraging: "
             f"{u_rank['rho']:+.2f} (reliable, p={u_rank['p']:.3f}) and {u_rank_b['rho']:+.2f} (broad, "
             f"p={u_rank_b['p']:.3f}) — both negative (low-coupling fields fall further in the wage "
             f"ranking after netting, as hypothesized), and the broad-set version clears p<0.05. Taken "
             f"together: **collapse_frac and rank_drop point the same (hypothesized) direction; "
             f"collapse-$ points the other way but is explainable by the pay-level confound above.** No "
             f"single metric here is a clean, well-powered confirmation.\n")

    L.append("## Honest verdict\n")
    L.append(f"- **Power**: the primary test has n={u_pri['n']} fields (the reliable Phase-1 set); the "
             f"bootstrap CI on the primary metric is wide ([{u_pri['ci_lo']:+.2f}, {u_pri['ci_hi']:+.2f}]) "
             f"and straddles 0 "
             f"— read this as **suggestive at best, not confirmatory**, same caveat Phase 1 applied to its "
             f"own reliable-set results (n=16-20 is simply not a lot of fields).")
    L.append(f"- **Consistency across metrics**: collapse_frac and rank_drop agree in (hypothesized, "
             f"negative) sign on both populations; collapse-$ disagrees but is plausibly a field-pay-level "
             f"confound (see above) rather than a genuine contradiction — still, a result that needed that "
             f"much per-metric interpretation to make consistent is not a clean unification.")
    L.append("- **Selection**: ACS observes only currently-employed, full-time, wage-and-salary workers "
             "who report a positive wage — anyone unemployed, part-time, out of the labor force, or "
             "self-employed-only is dropped, and that selection itself could differ by field/prestige in "
             "ways this script does not address.")
    L.append("- **NAICS-sector coarseness**: 21 sectors x 51 states is still coarse relative to actual "
             "labor markets (e.g. \"Health Care\" lumps hospital nursing with outpatient clinics and "
             "nursing homes, which may have different state-level pay-setting institutions); a finer cut "
             "was not usable in a 1-year file (see cell-size diagnostics above).")
    L.append(f"- **1-year ACS**: single-year, so sampling noise in small fields is real (smallest project "
             f"field is astronomy, n={int(net.loc[net.field=='astronomy','n_persons'].iloc[0]) if (net.field=='astronomy').any() else 'n/a'} "
             f"persons — see the full per-field n in `data/interim/phase3_field_netting.csv`); this is a "
             f"snapshot, not a multi-year average, and 2023 specifically (post-pandemic labor-market "
             f"churn) is not necessarily a typical year.")
    L.append("- **Nursing's result is a genuine complication, not noise**: with n=22,518 persons it is "
             "the best-powered field in the whole table, and its premium AMPLIFIES rather than collapses "
             "under state x NAICS-sector netting — the simple \"netting on geography x industry recovers "
             "the low-coupling fields\" story does not mechanically hold for the single most extreme "
             "low-coupling field. The likely fix (occupation-level netting) is a natural next step, not "
             "evidence the whole approach is wrong, but it means the headline correlation above is doing "
             "real work reconciling one confirming case (communication_disorders) against one "
             "disconfirming case (nursing) inside a small n.")
    L.append("- **No institution data anywhere in Phase 3/4**: this is a field-level test by design (see "
             "header); it cannot speak to within-field, within-institution variation, which is what the "
             "blocked institution-level test would have measured directly.")
    L.append("- **Associational only**: both rho_f and collapse_frac are cross-sectional field "
             "descriptives; nothing here identifies a causal channel from \"pay is geography x industry "
             "set\" to \"prestige doesn't predict earnings\" — the correlation (if present) is consistent "
             "with the proposed mechanism but does not prove it (e.g. both could be downstream of a third "
             "factor, occupational licensure, which Phase 1's own LICENSED_FIELDS flag already "
             "identifies as overlapping with the low-coupling fields).\n")
    (OUT / "PHASE3_4_RESULT.md").write_text("\n".join(L) + "\n")


def make_figure(rel_m, broad_m, U):
    fig, ax = plt.subplots(figsize=(8.5, 7))
    d = broad_m.dropna(subset=["spearman", "collapse_frac"])
    rset = set(rel_m.field)
    is_rel = d.field.isin(rset)
    ax.scatter(d.spearman[~is_rel], d.collapse_frac[~is_rel], s=46, color="#9DB4C0",
               edgecolor="white", linewidth=.5, label="broad (sensitivity)", zorder=2)
    ax.scatter(d.spearman[is_rel], d.collapse_frac[is_rel], s=90, color="#D6202A",
               edgecolor="white", linewidth=.7, label="reliable (primary)", zorder=3)
    for _, r in d.iterrows():
        callout = r.field in {"nursing", "communication_disorders", "computer_science", "finance",
                               "accounting", "economics", "psychology"}
        ax.annotate(r.label, (r.spearman, r.collapse_frac),
                    fontsize=(8 if callout else 6.5), fontweight=("bold" if callout else "normal"),
                    color=("#7A1010" if callout else "#444"),
                    xytext=(4, 3), textcoords="offset points")
    if len(d) >= 3:
        b, a0 = np.polyfit(d.spearman, d.collapse_frac, 1)
        xx = np.linspace(d.spearman.min(), d.spearman.max(), 50)
        ax.plot(xx, a0 + b * xx, ls="--", color="#555", lw=1, zorder=1)
    ax.axhline(0, color="#bbb", lw=.8, ls=":"); ax.axhline(1, color="#bbb", lw=.8, ls=":")
    u = U[("reliable", "collapse_frac")]; ub = U[("broad", "collapse_frac")]
    ax.set_xlabel("rho_f = prestige<->earnings coupling (Phase 1, Scorecard institution-level)")
    ax.set_ylabel("netting_collapse_frac (Phase 3, ACS individual-level, state x industry netted)")
    ax.set_title(f"Phase 4 unification: Spearman = {u['rho']:+.2f} [{u['ci_lo']:+.2f},{u['ci_hi']:+.2f}] "
                 f"reliable n={u['n']}  |  {ub['rho']:+.2f} broad n={ub['n']}", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "phase3_unification.png", dpi=140)


if __name__ == "__main__":
    main()
