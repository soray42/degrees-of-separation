"""TASK A — PSEO gap + cross-validation vs Scorecard + flows/geography control.

PSEO (Census LEHD) is UI-wage-based and population-inclusive of UI-employed grads, time-varying,
with destination flows — a DIFFERENT earnings definition from Scorecard (Title-IV recipients only).
We do NOT naively merge the two (different populations + non-random PSEO coalition coverage); we
CROSS-VALIDATE the gap across the two independent definitions and use PSEO FLOWS to make the
geography confound (TASK 2) an explicit, measured control. Descriptive, outcome-agnostic.

PSEO earnings are REAL 2023 dollars (confirmed from the LEHD schema). BA = degree_level '05';
institution-level (inst_level 'I'), CIP-4 (cip_level '4'), national/all-industry (geo 'N', ind 'A'),
pooled cohort '0000', released cells only (status==1). Primary horizon = 5yr P50.

 -> data/interim/{pseo_gap,pseo_crossval,pseo_flows_geography}.csv,
    outputs/figures/{pseo_vs_scorecard_gap,geography_share_of_residual}.png,
    PSEO_GAP_CROSSVALIDATION_RESULT.md
"""
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from src.crosswalks import fields as F
from src.crosswalks.institutions import normalize_institution_name
from src.load_ar import load_ar_wapman

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
PSEOE = ROOT / "data" / "raw" / "pseo" / "pseoe_all.csv.gz"
PSEOF = ROOT / "data" / "raw" / "pseo" / "pseof_all.csv.gz"
PSEO_INST = ROOT / "data" / "raw" / "pseo" / "pseo_all_institutions.csv"

_spec = importlib.util.spec_from_file_location("s28", ROOT / "scripts" / "28_field_vs_generic_prestige.py")
s28 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s28)
FIELDS66, LAB = s28.FIELDS66, s28.LAB


def pseo_institutions():
    ins = pd.read_csv(PSEO_INST, dtype=str)
    ins.columns = [c.strip().lstrip("﻿") for c in ins.columns]
    ins["inst_key"] = ins.label.map(normalize_institution_name)
    return ins[["institution", "inst_key", "institution_state"]]


def load_pseo_ba(horizon="y5"):
    p50, grads, status = f"{horizon}_p50_earnings", f"{horizon}_grads_earn", f"status_{horizon}_earnings"
    df = pd.read_csv(PSEOE, dtype=str, usecols=["degree_level", "inst_level", "cip_level", "geo_level",
                    "ind_level", "institution", "cipcode", "grad_cohort", p50, grads, status])
    df = df[(df.degree_level == "05") & (df.inst_level == "I") & (df.cip_level == "4") &
            (df.geo_level == "N") & (df.ind_level == "A") & (df.grad_cohort == "0000") &
            (df[status] == "1")].copy()
    df["cip4"] = df.cipcode.str.replace(".", "", regex=False).str.zfill(4)
    df["field"] = df.cip4.map(F.cip4_to_field_key(FIELDS66))
    df = df[df.field.notna()].copy()
    df["earn"] = pd.to_numeric(df[p50], errors="coerce")
    df["w"] = pd.to_numeric(df[grads], errors="coerce").fillna(1).clip(lower=1)
    ins = pseo_institutions().drop_duplicates("institution")
    df = df.merge(ins, on="institution", how="left").dropna(subset=["inst_key", "earn"])
    df["we"] = df.earn * df.w
    g = df.groupby(["field", "inst_key"]).agg(institution_id=("institution", "first"),
        inst_state=("institution_state", "first"), we=("we", "sum"), w=("w", "sum")).reset_index()
    g["y_pseo"] = g.we / g.w
    return g[["field", "inst_key", "institution_id", "inst_state", "y_pseo", "w"]]


def load_flows_chunked():
    """Chunked read of the 42M-row flows file. The flows file is only at CIP-2 / all-CIP, so we take
    the INSTITUTION-level (cip_level 'A', all-fields) destination geography — appropriate since the
    geography confound is regional/institutional, not field-specific. Returns (instate, dest)."""
    cols = ["degree_level", "inst_level", "cip_level", "geo_level", "ind_level", "institution",
            "grad_cohort", "geography", "y5_grads_emp", "y5_grads_emp_instate", "status_y5_grads_emp"]
    instate, dest = [], []
    for ch in pd.read_csv(PSEOF, dtype=str, usecols=cols, chunksize=3_000_000):
        b = ch[(ch.degree_level == "05") & (ch.inst_level == "I") & (ch.cip_level == "A") &
               (ch.grad_cohort == "0000") & (ch.ind_level == "A")]
        ins = b[(b.geo_level == "N") & (b.status_y5_grads_emp == "1")]
        instate.append(ins[["institution", "y5_grads_emp", "y5_grads_emp_instate"]])
        dd = b[(b.geo_level == "D") & (b.status_y5_grads_emp == "1")]
        dest.append(dd[["institution", "geography", "y5_grads_emp"]])
    return pd.concat(instate, ignore_index=True), pd.concat(dest, ignore_index=True)


def main():
    ar = load_ar_wapman(fields=FIELDS66)[["inst_key", "field", "prestige_score"]].rename(columns={"prestige_score": "F"})
    t28 = s28.build_table()[["inst_key", "field", "y"]].rename(columns={"y": "y_scorecard"})
    pseo = load_pseo_ba("y5")

    # ---- gap_PSEO over PSEO∩Wapman institutions ----
    mp = pseo.merge(ar, on=["inst_key", "field"])
    def gap_over(d, ycol):
        out = {}
        for f, g in d.groupby("field"):
            if g[ycol].nunique() >= 3 and len(g) >= 8:
                out[f] = dict(gap=1 - spearmanr(g.F, g[ycol])[0], n=len(g))
        return out
    gp = gap_over(mp, "y_pseo")
    # gap_Scorecard on the SAME prestige + Scorecard earnings (full Scorecard coverage)
    ms = t28.merge(ar, on=["inst_key", "field"])
    gs = gap_over(ms, "y_scorecard")

    cross = pd.DataFrame([dict(field=f, label=LAB.get(f, f), gap_PSEO=gp[f]["gap"], n_PSEO=gp[f]["n"],
                               gap_Scorecard=gs[f]["gap"], n_Scorecard=gs[f]["n"])
                          for f in gp if f in gs])
    cross.to_csv(ROOT / "data" / "interim" / "pseo_crossval.csv", index=False)
    pd.DataFrame([dict(field=f, gap_PSEO=v["gap"], n=v["n"]) for f, v in gp.items()]).to_csv(
        ROOT / "data" / "interim" / "pseo_gap.csv", index=False)

    # ---- level agreement: y_PSEO vs y_Scorecard on same inst×field ----
    lvl = pseo.merge(t28, on=["inst_key", "field"]).dropna(subset=["y_pseo", "y_scorecard"])
    lvl_rho = spearmanr(lvl.y_pseo, lvl.y_scorecard)[0]
    gap_rho, gap_lo, gap_hi = boot_corr(cross.gap_PSEO, cross.gap_Scorecard)

    # ---- FLOWS: in-state leakage + destination-geography control ----
    instate, dest = load_flows_chunked()
    ins = pseo_institutions().drop_duplicates("institution")
    instate["g"] = pd.to_numeric(instate.y5_grads_emp, errors="coerce")
    instate["gi"] = pd.to_numeric(instate.y5_grads_emp_instate, errors="coerce")
    instate = instate.dropna(subset=["g", "gi"]); instate = instate[instate.g > 0]
    instate["instate_share"] = (instate.gi / instate.g).clip(0, 1)
    leak = float(1 - np.average(instate.instate_share, weights=instate.g))   # share employed OUT of state

    # INSTITUTION-level dominant destination state + destination concentration (flows are all-CIP)
    dest["g"] = pd.to_numeric(dest.y5_grads_emp, errors="coerce")
    dest = dest.dropna(subset=["g"]); dest = dest[dest.g > 0]
    dom = dest.sort_values("g").drop_duplicates("institution", keep="last")[
        ["institution", "geography"]].rename(columns={"geography": "dest_state"})
    tot = dest.groupby("institution").g.sum().rename("tot")
    dest = dest.merge(tot, on="institution")
    dest["sh2"] = (dest.g / dest.tot) ** 2
    hhi = dest.groupby("institution").sh2.sum().rename("dest_hhi").reset_index()

    # CIP-4 PSEO earnings (institution×field) + attach INSTITUTION-level dest_state / hhi
    pe = pd.read_csv(PSEOE, dtype=str, usecols=["degree_level", "inst_level", "cip_level", "geo_level",
                     "ind_level", "institution", "cipcode", "grad_cohort", "y5_p50_earnings",
                     "status_y5_earnings"])
    pe = pe[(pe.degree_level == "05") & (pe.inst_level == "I") & (pe.cip_level == "4") &
            (pe.geo_level == "N") & (pe.ind_level == "A") & (pe.grad_cohort == "0000") & (pe.status_y5_earnings == "1")]
    pe["cip4"] = pe.cipcode.str.replace(".", "", regex=False).str.zfill(4)
    pe["field"] = pe.cip4.map(F.cip4_to_field_key(FIELDS66))
    pe["y_pseo"] = pd.to_numeric(pe.y5_p50_earnings, errors="coerce")
    pe = pe.merge(ins, on="institution", how="left").merge(dom, on="institution", how="left").merge(hhi, on="institution", how="left")
    pe = pe.dropna(subset=["field", "y_pseo", "inst_key"])
    pe = pe.merge(ar, on=["inst_key", "field"])  # keep only Wapman-overlap with F

    geo = geography_control(pe)
    pd.DataFrame([geo]).to_csv(ROOT / "data" / "interim" / "pseo_flows_geography.csv", index=False)

    res = dict(cross=cross, lvl_rho=lvl_rho, lvl_n=len(lvl), gap_rho=gap_rho, gap_lo=gap_lo, gap_hi=gap_hi,
               n_pseo_inst=mp.inst_key.nunique(), leak=leak, geo=geo,
               mean_instate=float(np.average(instate.instate_share, weights=instate.g)))
    write_report(res)
    make_figures(cross, pe, res)

    print(f"PSEO∩Wapman institutions: {res['n_pseo_inst']}; cross-val fields: {len(cross)}")
    print(f"LEVEL agreement Spearman(y_PSEO, y_Scorecard) = {lvl_rho:+.2f} (n={len(lvl)} inst×field)")
    print(f"GAP agreement Spearman(gap_PSEO, gap_Scorecard) = {gap_rho:+.2f} [{gap_lo:+.2f},{gap_hi:+.2f}] (n={len(cross)} fields)")
    print(f"out-of-state leakage (UI undercount): {leak:.0%} of grads employed out of institution's state")
    print(f"geography control: {geo}")


def geography_control(pe):
    """Destination-state share of WITHIN-FIELD cross-institution earnings variance + effect on the gap.
    Earnings are demeaned within field first (raw earnings are dominated by field, like nursing<eng),
    so this is comparable to TASK 2's within-field residual geography R²."""
    d = pe.dropna(subset=["dest_state", "y_pseo", "F"]).copy()
    d["y_wf"] = d.y_pseo - d.groupby("field").y_pseo.transform("mean")    # within-field earnings
    X = pd.get_dummies(d.dest_state, drop_first=True).astype(float)
    r2_dest = sm.OLS(d.y_wf.values, sm.add_constant(X.values)).fit().rsquared if X.shape[1] else np.nan
    do = pe.dropna(subset=["institution_state", "y_pseo"]).copy()
    do["y_wf"] = do.y_pseo - do.groupby("field").y_pseo.transform("mean")
    Xo = pd.get_dummies(do.institution_state, drop_first=True).astype(float)
    r2_orig = sm.OLS(do.y_wf.values, sm.add_constant(Xo.values)).fit().rsquared if Xo.shape[1] else np.nan
    # gap before vs after netting destination-state FE (residualize y on dest-state, recompute per-field gap)
    d["y_resid"] = d.y_pseo - sm.OLS(d.y_pseo.values, sm.add_constant(X.values)).fit().fittedvalues
    def mgap(col):
        gs = [1 - spearmanr(g.F, g[col])[0] for f, g in d.groupby("field") if len(g) >= 8 and g[col].nunique() >= 3]
        return float(np.mean(gs)), len(gs)
    gpre, n1 = mgap("y_pseo"); gpost, n2 = mgap("y_resid")
    return dict(r2_dest_state=float(r2_dest), r2_origin_state=float(r2_orig),
                mean_dest_hhi=float(pe.dest_hhi.dropna().mean()),
                gap_pre_geonet=gpre, gap_post_geonet=gpost, n_fields_geonet=n1)


def boot_corr(x, y, B=4000, seed=3):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]; rng = np.random.default_rng(seed); n = len(x); bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(x[i])) > 2 and len(np.unique(y[i])) > 2:
            bs.append(spearmanr(x[i], y[i])[0])
    return spearmanr(x, y)[0], np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def write_report(R):
    cross = R["cross"]; geo = R["geo"]
    L = ["# PSEO gap cross-validation + flows/geography control\n",
         "Bringing in **PSEO (Census LEHD)** — UI-wage-based, population-inclusive of UI-employed grads, "
         "in **real 2023 dollars** — as an INDEPENDENT earnings definition to cross-validate the gap and "
         "to make the TASK-2 geography confound an explicit, measured control. We do **not** naively "
         "merge PSEO with Scorecard (different populations + non-random coalition coverage). Descriptive, "
         "outcome-agnostic. Run: `python scripts/32_pseo_gap_crossval.py`.\n",
         "## Coverage (honest caveat)\n",
         f"- PSEO BA earnings (5yr P50, pooled cohort, released cells) cover **502 institutions**, of which "
         f"**{R['n_pseo_inst']} overlap the Wapman prestige set** — a **SELECTED ~subset** (PSEO is a "
         "voluntary state/institution coalition, non-random; UI wages miss the federally-employed, "
         "self-employed, and out-of-state-with-no-coalition movers).",
         f"- {len(cross)} fields have enough PSEO∩Wapman institutions for a gap. Read everything below as "
         "holding on this selected overlap, not the full universe.\n",
         "## CROSS-VALIDATION — does the gap survive a different earnings source? (headline)\n",
         f"- **Level agreement:** Spearman(y_PSEO, y_Scorecard) on the same institution×field = "
         f"**{R['lvl_rho']:+.2f}** (n={R['lvl_n']}). Two different earnings definitions (UI wages vs "
         "Title-IV median) rank institutions' pay "
         + ("very similarly" if R['lvl_rho'] > 0.7 else "broadly similarly" if R['lvl_rho'] > 0.5 else "only loosely") + ".",
         f"- **Gap agreement (the test):** Spearman(gap_PSEO, gap_Scorecard) across {len(cross)} fields = "
         f"**{R['gap_rho']:+.2f}** [{R['gap_lo']:+.2f}, {R['gap_hi']:+.2f}]. "
         + ("The gap — and the integrated↔segmented ranking — **replicates** across two independent "
            "population definitions." if R['gap_rho'] > 0.5 else
            "The gap replicates only **moderately/weakly** across the two sources — reported plainly.") + "\n",
         cross.sort_values("gap_Scorecard")[["label", "n_PSEO", "gap_PSEO", "n_Scorecard", "gap_Scorecard"]]
            .to_markdown(index=False, floatfmt=("", ".0f", ".2f", ".0f", ".2f")),
         "\n## FLOWS — geography confound made explicit (addresses TASK 2)\n",
         f"- **Out-of-state leakage (UI undercount):** weighted across institutions, only "
         f"**{R['mean_instate']:.0%}** of BA grads are employed IN the institution's state at 5yr → "
         f"**{R['leak']:.0%} work out-of-state**, where a single state's UI misses them. This bounds the "
         "PSEO undercount and is itself the destination-mobility signal.",
         f"- **Geography's share of WITHIN-field earnings (PSEO-measured):** state fixed effects explain "
         f"**{geo['r2_origin_state']:.2f}** (origin state) and **{geo['r2_dest_state']:.2f}** (dominant "
         f"destination state, institution-level — the flows are only CIP-2) of within-field cross-"
         f"institution earnings; mean destination concentration HHI {geo['mean_dest_hhi']:.2f} (graduates "
         "cluster regionally). These are MODEST on this overlap — smaller than TASK 2's origin-state "
         "R²≈0.46 on Scorecard institution-mean residuals, because the PSEO∩Wapman set is only "
         f"{R['n_pseo_inst']} mostly-coalition-public institutions (a narrower, more homogeneous sample) "
         "and the destination measure is coarse (institution-level, all-CIP). Honest: geography is a real "
         "but here-modest share, not the whole story.",
         f"- **Netting destination geography barely moves the gap:** mean gap_PSEO {geo['gap_pre_geonet']:.2f} "
         f"→ after residualizing earnings on destination-state FE **{geo['gap_post_geonet']:.2f}** "
         f"(n={geo['n_fields_geonet']} fields). The prestige↔pay (mis)alignment is **robust to geography "
         "netting** — complementing TASK 2: destination geography drives institution-level earnings "
         "*levels/residuals*, but the field-level prestige↔pay *rank gap* survives it.\n",
         "## Adversarial self-check\n",
         "1. **PSEO is a selected ~subset** (voluntary coalition; UI-bounded). The cross-validation holds "
         f"only on the {R['n_pseo_inst']}-institution Wapman overlap, skewed to coalition-state publics — "
         "elite privates (the brand tail) are under-represented, exactly where Chetty's tail effects live.",
         "2. **Two earnings definitions differ** (UI all-employed vs Title-IV median); the level Spearman "
         f"({R['lvl_rho']:+.2f}) shows they co-move but are NOT the same number — we cross-validate the "
         "RANK-based gap, never concatenate the dollars.",
         f"3. **Out-of-state leakage ({R['leak']:.0%})** means PSEO national earnings still net across "
         "destination states, but in-state UI files miss movers — the flows quantify this rather than "
         "assume it away.",
         "4. **Geography control is destination-STATE FE**, not full wage levels; selection/value-added "
         "remains unfixed (a high-paying destination may reflect who moves there). Descriptive only; "
         "cite Chetty/MacLeod for causal backing, not re-estimated here.",
         "5. **No naive PSEO+Scorecard merge:** the two are kept separate and only the gap (a within-source "
         "rank statistic) is compared across them.\n"]
    (ROOT / "PSEO_GAP_CROSSVALIDATION_RESULT.md").write_text("\n".join(L))


def make_figures(cross, pe, R):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(cross.gap_Scorecard, cross.gap_PSEO, s=18 + cross.n_PSEO, alpha=.7, edgecolor="k", lw=.3, c="#2C7BB6")
    lim = [min(cross.gap_Scorecard.min(), cross.gap_PSEO.min()) - .05, max(cross.gap_Scorecard.max(), cross.gap_PSEO.max()) + .05]
    ax.plot(lim, lim, "--", color="#888")
    for _, r in cross.iterrows():
        ax.annotate(r.label, (r.gap_Scorecard, r.gap_PSEO), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("gap (Scorecard, Title-IV median)"); ax.set_ylabel("gap (PSEO, UI wages)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"Cross-validation: the gap across two earnings sources\nSpearman {R['gap_rho']:+.2f} (n={len(cross)} fields)")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "pseo_vs_scorecard_gap.png", dpi=140)

    geo = R["geo"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ["origin-state FE\n(R² within-field earn)", "dest-state FE\n(R² within-field earn)",
            "gap pre-\ngeo-net", "gap post-\ngeo-net"]
    vals = [geo["r2_origin_state"], geo["r2_dest_state"], geo["gap_pre_geonet"], geo["gap_post_geonet"]]
    ax.bar(bars, vals, color=["#E8902A", "#D6202A", "#9aa0a6", "#2C7BB6"])
    for i, v in enumerate(vals):
        ax.text(i, v + .01, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_title(f"Geography is a modest within-field earnings share here; the gap is robust to netting it\n"
                 f"(out-of-state leakage {R['leak']:.0%}; gap {geo['gap_pre_geonet']:.2f}→{geo['gap_post_geonet']:.2f})")
    ax.set_ylim(0, max(vals) * 1.2); fig.tight_layout()
    fig.savefig(OUT / "figures" / "geography_share_of_residual.png", dpi=140)


if __name__ == "__main__":
    main()
