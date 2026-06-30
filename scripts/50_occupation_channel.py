"""scripts/50_occupation_channel.py
=============================================================================
Per-field OCCUPATION-channel measures for the occupation-channel test.

Two pieces, both ACS PUMS / PWGTP-weighted, joined to the Phase-1 prestige<->earnings
coupling rho_f from the expanded 66-field gap map:

  1. CONCENTRATION (reuses the cached recent-BA occupation-mix file built by
     scripts/35_er_definition.py:extract_occ_distribution -- data/interim/acs_occ_dist.parquet,
     cols field/soc6/occp4/PWGTP). Per field, from the PWGTP-weighted OCCP (occp4)
     distribution:
       occ_hhi        = Herfindahl index of P(occp4 | field) = sum_o p(o|f)^2
       top_occ_share  = share of the field's grads in the single most common occupation
       occ_entropy    = Shannon entropy of P(occp4 | field), in BITS: -sum_o p(o|f) log2 p(o|f)
     High occ_hhi / top_occ_share (low occ_entropy) = the field PINS its grads to one
     occupation (nursing -> RN, communication_disorders -> SLP); low occ_hhi / high
     occ_entropy = grads spread across many occupations (computer_science, economics,
     business -- general-purpose degrees with no single destination occupation).

  2. OCC-NETTED WAGE COLLAPSE (fresh ACS extraction, this script -- the CORRECTED netting
     variable). scripts/48_phase3_4_netting_unification.py netted on STATE x NAICS-2-digit
     INDUSTRY SECTOR and found nursing's premium AMPLIFIED (collapse_frac = -1.0; netted
     premium > raw premium), not explained, by industry-sector composition -- a revealed-
     demand paradox for the "pay is set by work setting" story when "work setting" = industry.
     The occupation-channel hypothesis is that for licensed/pinned fields pay is set by the
     OCCUPATION (RN, SLP) itself, not the industry sector employing it -- so the correct
     netting cell is STATE x OCCP, not STATE x NAICS-sector. For each ACS bachelor's-holder,
     full-time employed, WAGP>0: benchmark WAGP against the median WAGP of ALL bachelor's-
     holders (any field, full population) in their (STATE x OCCP) cell (>=25 persons/cell;
     thin cells excluded from the netted estimate, flagged). occ_netted_collapse = fraction
     of the field's raw wage premium (vs the national BA median) that vanishes once
     state x occupation composition is netted out. Filter mirrors scripts/48/27:
     SCHL>=21, FOD1P present, WKHP>=35, ESR in {1,2}, WAGP>0 -- swapping NAICSP for OCCP.

Cross-construct, non-circular by the same logic as Phase 4 (script 48): rho_f is built from
College Scorecard INSTITUTION-level data; occ_netted_collapse is built from ACS PUMS
INDIVIDUAL-level WAGP netted on STATE x OCCP. No institution identifier exists in ACS and no
occupation netting is ever applied to the Scorecard earnings used for rho_f.

Output: data/interim/occupation_channel_fields.csv -- one row per field (the 52 fields with
ACS occupation-mix coverage), with occ_hhi / top_occ_share / occ_entropy / occ_netted_collapse
(+ raw/netted wage detail) and rho_f (= gap_map 'spearman') / cip2 / reliable joined from
outputs/expanded66_gap_map.csv.

Run: .venv/bin/python scripts/50_occupation_channel.py [--refresh]
"""
from __future__ import annotations
import sys, glob, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from src.crosswalks import fields as F
from src.dispersion import weighted_quantile as wq

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"; INTERIM.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "outputs"
ACS_GLOB = str(ROOT / "data" / "raw" / "acs" / "psam_pus*.csv")
OCC_DIST = INTERIM / "acs_occ_dist.parquet"            # built by scripts/35_er_definition.py
NET_CACHE = INTERIM / "acs_occ_netting_full.parquet"    # fresh extraction (this script)
RESULT_CSV = INTERIM / "occupation_channel_fields.csv"

SEED = 50
MIN_CELL = 25       # min persons/cell (STATE x OCCP), full bachelor's population, non-thin
MIN_FIELD_N = 30    # min non-thin-cell persons for a field's netted estimate to be reported
LAB = {f["key"]: f["label"] for f in F.ALL_FIELDS}


# ============================================================ concentration
def occ_concentration() -> pd.DataFrame:
    """Per-field PWGTP-weighted OCCP (occp4) concentration from the cached recent-BA
    occupation mix (data/interim/acs_occ_dist.parquet)."""
    if not OCC_DIST.exists():
        raise FileNotFoundError(f"{OCC_DIST} missing -- run scripts/35_er_definition.py first "
                                 f"(extract_occ_distribution) to build it.")
    occ = pd.read_parquet(OCC_DIST)
    rows = []
    for fld, g in occ.groupby("field"):
        w = g.groupby("occp4")["PWGTP"].sum()
        p = (w / w.sum()).values
        p = p[p > 0]
        rows.append(dict(
            field=fld,
            occ_hhi=float((p ** 2).sum()),
            top_occ_share=float(p.max()),
            occ_entropy=float(-(p * np.log2(p)).sum()),
            occ_n_occs=int(len(p)),
            occ_dist_n=int(len(g)),
            occ_dist_wtot=float(w.sum()),
        ))
    return pd.DataFrame(rows)


# ============================================================ extraction ===
def extract_occ_netting(refresh: bool = False) -> pd.DataFrame:
    """Fresh ACS PUMS slice: bachelor's-or-higher, full-time employed, WAGP>0, with OCCP
    (occupation, not industry) retained, mapped to project fields. Filter mirrors
    scripts/48_phase3_4_netting_unification.py / scripts/27_external_channels.py
    (SCHL>=21, FOD1P present, WKHP>=35, ESR in {1,2}, WAGP>0), swapping NAICSP for OCCP."""
    if NET_CACHE.exists() and not refresh:
        return pd.read_parquet(NET_CACHE)
    need = ["STATE", "FOD1P", "OCCP", "WAGP", "SCHL", "ESR", "WKHP", "PWGTP"]
    frs = []
    for fp in sorted(glob.glob(ACS_GLOB)):
        cols = pd.read_csv(fp, nrows=0).columns
        frs.append(pd.read_csv(fp, usecols=[c for c in need if c in cols], dtype={"OCCP": str}))
    a = pd.concat(frs, ignore_index=True)
    for c in ["FOD1P", "WAGP", "SCHL", "WKHP", "ESR", "STATE"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a[(a.SCHL >= 21) & a.FOD1P.notna() & (a.WKHP >= 35) & a.ESR.isin([1, 2]) & (a.WAGP > 0)].copy()
    a["occp4"] = a.OCCP.fillna("").str.zfill(4)
    code_map = F.fod1p_to_field_key(F.fod1p_by_key_all())
    a["field"] = a.FOD1P.astype(int).astype(str).str.zfill(4).map(code_map)
    a = a[["STATE", "occp4", "field", "WAGP", "PWGTP"]].reset_index(drop=True)
    a.to_parquet(NET_CACHE, index=False)
    return a


def occ_netted_collapse(a: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Benchmark every person's WAGP against their (STATE x OCCP) cell median (full
    bachelor's-employed population, >=25 persons/cell; thin cells flagged + excluded from
    the netted estimate). Per field: raw (weighted-median WAGP) vs netted (weighted-median
    excess) premium relative to the national BA median; occ_netted_collapse = the fraction
    of the raw premium that vanishes after netting (collapse / raw_premium)."""
    a = a.copy()
    cell_n = a.groupby(["STATE", "occp4"]).size()
    a["cell_n"] = pd.MultiIndex.from_frame(a[["STATE", "occp4"]]).map(cell_n)
    a["thin_cell"] = a.cell_n < MIN_CELL
    meds = {key: wq(g.WAGP.values, g.PWGTP.values, .5) for key, g in a.groupby(["STATE", "occp4"])}
    med_s = pd.Series(meds)
    a["cell_median"] = pd.MultiIndex.from_frame(a[["STATE", "occp4"]]).map(med_s)
    a["excess"] = a.WAGP - a.cell_median

    overall_wage = wq(a.WAGP.values, a.PWGTP.values, .5)
    valid_all = a[~a.thin_cell]
    overall_excess = wq(valid_all.excess.values, valid_all.PWGTP.values, .5)

    rows = []
    for fld, g in a[a.field.notna()].groupby("field"):
        gv = g[~g.thin_cell]
        n, n_valid = len(g), len(gv)
        raw_wage = wq(g.WAGP.values, g.PWGTP.values, .5)
        netted_wage = wq(gv.excess.values, gv.PWGTP.values, .5) if n_valid >= MIN_FIELD_N else np.nan
        rows.append(dict(field=fld, occ_n_persons=n, occ_n_valid=n_valid,
                          occ_pct_thin=1 - n_valid / n if n else np.nan,
                          raw_wage=raw_wage, netted_wage=netted_wage))
    df = pd.DataFrame(rows)
    df["raw_premium"] = df.raw_wage - overall_wage
    df["netted_premium"] = df.netted_wage - overall_excess
    df["collapse"] = df.raw_premium - df.netted_premium
    # occ_netted_collapse only meaningful when the raw $ premium is non-trivial
    df["occ_netted_collapse"] = np.where(df.raw_premium.abs() > 1000,
                                          df.collapse / df.raw_premium, np.nan)

    thin = cell_n < MIN_CELL
    diag = dict(n_cells=int(len(cell_n)), n_thin_cells=int(thin.sum()),
                pct_thin_cells=float(thin.mean()), n_persons=int(len(a)),
                pct_thin_persons=float(a.thin_cell.mean()),
                overall_wage=float(overall_wage), overall_excess=float(overall_excess))
    return df, diag


# ===================================================================== main
def build(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    conc = occ_concentration()
    a = extract_occ_netting(refresh=refresh)
    net, diag = occ_netted_collapse(a)

    gm = pd.read_csv(OUT / "expanded66_gap_map.csv")[["field", "spearman", "cip2", "reliable"]]
    gm = gm.rename(columns={"spearman": "rho_f"})

    df = conc.merge(net, on="field", how="outer").merge(gm, on="field", how="left")
    df["label"] = df.field.map(lambda k: LAB.get(k, k.replace("_", " ").title()))
    df = df.sort_values("occ_hhi", ascending=False).reset_index(drop=True)
    df.to_csv(RESULT_CSV, index=False)
    return df, diag


def report(df: pd.DataFrame, diag: dict) -> str:
    L = []
    L.append("# Occupation-channel measures (scripts/50_occupation_channel.py)\n")
    L.append(f"n fields = {len(df)}. STATE x OCCP netting cells: {diag['n_cells']} "
             f"({diag['n_thin_cells']} thin, {diag['pct_thin_cells']:.1%}); "
             f"{diag['n_persons']} persons ({diag['pct_thin_persons']:.1%} in thin cells). "
             f"National BA median WAGP = ${diag['overall_wage']:,.0f}; "
             f"national BA median excess-over-own-cell = ${diag['overall_excess']:,.0f}.\n")

    focus = ["nursing", "communication_disorders", "computer_science", "economics",
              "finance", "biology"]
    L.append("## occ_hhi / top_occ_share / occ_entropy for the named fields\n")
    cols = ["field", "occ_hhi", "top_occ_share", "occ_entropy", "occ_n_occs",
            "occ_netted_collapse", "raw_premium", "rho_f"]
    sub = df.set_index("field").reindex(focus)[cols[1:]]
    L.append(sub.round(4).to_string())
    L.append("")

    nurse = df[df.field == "nursing"].iloc[0]
    L.append("## Nursing: occupation-netting vs industry-sector netting\n")
    L.append(f"occ_hhi={nurse.occ_hhi:.3f}, top_occ_share={nurse.top_occ_share:.3f} "
              f"(occupation-pinned: {'YES' if nurse.occ_hhi > df.occ_hhi.median() else 'no'}).")
    L.append(f"occ_netted_collapse (STATE x OCCP) = {nurse.occ_netted_collapse:+.3f} "
              f"(raw_premium=${nurse.raw_premium:,.0f}, netted_premium=${nurse.netted_premium:,.0f}).")
    old = OUT.parent / "data" / "interim" / "phase3_field_netting.csv"
    if (ROOT / "data" / "interim" / "phase3_field_netting.csv").exists():
        old_df = pd.read_csv(ROOT / "data" / "interim" / "phase3_field_netting.csv")
        on = old_df[old_df.field == "nursing"]
        if len(on):
            L.append(f"Industry-sector (STATE x NAICS-2) collapse_frac (script 48) = "
                      f"{on.collapse_frac.iloc[0]:+.3f} (negative = AMPLIFIED, did not collapse).")
    L.append("")

    L.append("## Per-field table, head sorted by occ_hhi desc\n")
    show = df[["field", "occ_hhi", "top_occ_share", "occ_entropy", "occ_netted_collapse",
               "raw_premium", "rho_f", "reliable"]].head(15)
    L.append(show.round(4).to_string(index=False))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-extract the ACS netting slice")
    args = ap.parse_args()
    df, diag = build(refresh=args.refresh)
    print(f"saved {RESULT_CSV} ({len(df)} rows, {len(df.columns)} cols)\n")
    print(report(df, diag))


if __name__ == "__main__":
    main()
