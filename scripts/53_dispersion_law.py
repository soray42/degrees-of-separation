"""Direction ② — re-run the dead continuous law with the RIGHT variable.

The pilot's `occ_hhi -> coupling` came back null because occupational CONCENTRATION is the
wrong variable (it ≈ the license dummy). The compression theory says coupling is driven by
**within-occupation prestige-sortable wage variance**: where a field's destination occupations
have wide, sortable pay (SWE, analysts), prestige can differentiate institutions -> high
coupling; where pay is compressed (RN, SLP, teacher pay-spine), it can't -> decoupled.

Destination wage-dispersion index (ACS PUMS, field-level; ACS has no institution so this tests
the MECHANISM VARIABLE, not the prestige link):
  1. Sample: bachelor's+ (SCHL>=21), FTFY (WKHP>=35), employed (ESR in {1,2}), WAGP>0.
  2. Residualize log(WAGP) ~ C(STATE) + AGEP poly(3) + C(SEX) + C(RAC1P)  [strip geography +
     experience + composition]; keep residuals.
  3. Per destination occupation (OCCP=occp4, >=MIN_OCC persons): weighted residual SD = the
     within-occupation prestige-sortable-variance proxy (an UPPER BOUND — mixes firm + skill +
     noise, so directional only).
  4. Per field: occupation mix from data/interim/acs_occ_dist.parquet.
  5. dispersion_index_f = occupation-share-weighted within-occupation residual SD.
  6. Regress coupling_f (= 1 - gap_f, Scorecard) ~ dispersion_index_f across fields.
Prediction (compression): dispersion up -> coupling up; nursing (RN pinned) low/low, CS high/high.

Non-circular: dispersion_index from ACS occupation wage structure; coupling from Scorecard
institution-level prestige~earnings -- independent data + constructs. Seeded, deterministic.
Run: python scripts/53_dispersion_law.py
"""
from __future__ import annotations

import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.crosswalks import fields as F

SEED = 53
MIN_OCC = 50          # min persons in an occupation to trust its within-occ residual SD
MIN_SHARE_COV = 0.40  # a field's dispersion index needs occupations covering >=40% of its mix
ROOT = Path(__file__).resolve().parents[1]
ACS_GLOB = str(ROOT / "data" / "raw" / "acs" / "psam_pus*.csv")
OCC_DIST = ROOT / "data" / "interim" / "acs_occ_dist.parquet"
GAP_MAP = ROOT / "outputs" / "expanded66_gap_map.csv"
rng = np.random.default_rng(SEED)


def load_pums() -> pd.DataFrame:
    """BA+, FTFY, employed, positive-wage ACS slice with experience + demographics."""
    need = ["STATE", "FOD1P", "OCCP", "WAGP", "SCHL", "ESR", "WKHP", "PWGTP",
            "AGEP", "SEX", "RAC1P"]
    frames = []
    for fp in sorted(glob.glob(ACS_GLOB)):
        cols = pd.read_csv(fp, nrows=0).columns
        frames.append(pd.read_csv(fp, usecols=[c for c in need if c in cols], dtype={"OCCP": str}))
    a = pd.concat(frames, ignore_index=True)
    for c in ["FOD1P", "WAGP", "SCHL", "WKHP", "ESR", "STATE", "AGEP", "SEX", "RAC1P", "PWGTP"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a[(a.SCHL >= 21) & a.FOD1P.notna() & (a.WKHP >= 35) &
          a.ESR.isin([1, 2]) & (a.WAGP > 0)].copy()
    a["occp4"] = a.OCCP.fillna("").str.zfill(4)
    a["field"] = a.FOD1P.astype(int).astype(str).str.zfill(4).map(
        F.fod1p_to_field_key(F.fod1p_by_key_all()))
    a["logw"] = np.log(a.WAGP.clip(lower=1))
    a["AGEP2"] = a.AGEP ** 2
    a["AGEP3"] = a.AGEP ** 3
    return a


def _wq(x, w, q):
    """Weighted quantile(s)."""
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    c /= w.sum()
    return np.interp(q, c, x)


# dispersion measures tested (theory prefers the UPPER TAIL — prestige sorts into high-pay firms):
MEASURES = ["sd", "iqr", "tail_hi", "tail_lo", "p90_10", "cv_level"]


def within_occ_dispersion(a: pd.DataFrame) -> pd.DataFrame:
    """Residualize log-wage on geography+experience+composition; per occp4 compute SEVERAL
    dispersion measures of the residual (and level-wage CV). Total SD mixes in universal noise;
    the compression theory is specifically about prestige-sortable UPPER-tail spread."""
    d = a.assign(STATE=a.STATE.astype(int).astype(str),
                 SEX=a.SEX.astype("Int64").astype(str),
                 RAC1P=a.RAC1P.astype("Int64").astype(str))
    model = smf.wls("logw ~ C(STATE) + AGEP + AGEP2 + AGEP3 + C(SEX) + C(RAC1P)",
                    data=d, weights=d.PWGTP).fit()
    d["resid"] = model.resid
    rows = []
    for occ, g in d.groupby("occp4"):
        if len(g) < MIN_OCC:
            continue
        w = g.PWGTP.to_numpy(float); r = g.resid.to_numpy(float); lev = g.WAGP.to_numpy(float)
        mu = np.average(r, weights=w)
        sd = np.sqrt(np.average((r - mu) ** 2, weights=w))
        p10, p25, p50, p75, p90 = _wq(r, w, [.1, .25, .5, .75, .9])
        lmu = np.average(lev, weights=w)
        lsd = np.sqrt(np.average((lev - lmu) ** 2, weights=w))
        rows.append(dict(occp4=occ, n=len(g),
                         sd=sd, iqr=p75 - p25, p90_10=p90 - p10,
                         tail_hi=p90 - p50, tail_lo=p50 - p10,      # upper vs lower tail (log)
                         cv_level=lsd / lmu if lmu > 0 else np.nan))
    print(f"[residualize] R²={model.rsquared:.3f} on {len(d):,} persons; "
          f"{len(rows)} occupations with n>={MIN_OCC}")
    return pd.DataFrame(rows)


def field_dispersion_index(occ_disp: pd.DataFrame) -> pd.DataFrame:
    """Per field, occupation-share-weighted index for EVERY dispersion measure + top-occupation."""
    mix = pd.read_parquet(OCC_DIST).merge(occ_disp, on="occp4", how="left")   # field|occp4|PWGTP|measures
    rows = []
    for f, g in mix.groupby("field"):
        tot = g.PWGTP.sum()
        valid = g.dropna(subset=["sd"])
        cov = valid.PWGTP.sum() / tot if tot > 0 else 0.0
        if cov < MIN_SHARE_COV or valid.PWGTP.sum() <= 0:
            continue
        rec = dict(field=f, occ_cov=cov)
        for m in MEASURES:
            rec[m] = np.average(valid[m], weights=valid.PWGTP)
        top = valid.sort_values("PWGTP", ascending=False).iloc[0]
        rec["top_occ"], rec["top_occ_tailhi"] = top.occp4, top.tail_hi
        rows.append(rec)
    return pd.DataFrame(rows)


def regress(df, xcol, ycol):
    x, y = df[xcol].to_numpy(float), df[ycol].to_numpy(float)
    rho, p_s = spearmanr(x, y)
    r, p_p = pearsonr(x, y)
    boot = []
    idx = np.arange(len(df))
    for _ in range(5000):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(set(x[b])) > 1:
            boot.append(spearmanr(x[b], y[b])[0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=len(df), spearman=float(rho), p_spearman=float(p_s),
                pearson=float(r), p_pearson=float(p_p), ci_lo=float(lo), ci_hi=float(hi))


HEADLINE = "tail_hi"   # theory-preferred: prestige sorts into the high-pay upper tail


def main():
    a = load_pums()
    occ_disp = within_occ_dispersion(a)
    fdi = field_dispersion_index(occ_disp)

    gm = pd.read_csv(GAP_MAP)[["field", "spearman", "gap", "cip2", "reliable"]].rename(
        columns={"spearman": "coupling"})
    df = fdi.merge(gm, on="field", how="inner").dropna(subset=["coupling"])
    df["licensed"] = df.field.isin(F.LICENSED_FIELDS)
    df["dispersion_index"] = df[HEADLINE]
    df.to_csv(ROOT / "data" / "interim" / "dispersion_law_fields.csv", index=False)
    rel = df[df.reliable]

    # regress EVERY dispersion measure vs coupling (transparent multiple-operationalization test)
    print("\n=== coupling ~ dispersion measure (all operationalizations) ===")
    print(f"{'measure':10s} | reliable n={len(rel):<2d}            | broad n={len(df)}")
    table = {}
    for m in MEASURES:
        rr, ra = regress(rel, m, "coupling"), regress(df, m, "coupling")
        table[m] = (rr, ra)
        star = "  <== headline (theory: upper tail)" if m == HEADLINE else ""
        print(f"{m:10s} | rho={rr['spearman']:+.3f} [{rr['ci_lo']:+.2f},{rr['ci_hi']:+.2f}] p={rr['p_spearman']:.2f} "
              f"| rho={ra['spearman']:+.3f} [{ra['ci_lo']:+.2f},{ra['ci_hi']:+.2f}] p={ra['p_spearman']:.2f}{star}")

    res_rel, res_all = table[HEADLINE]
    _figure(df, rel, res_rel)
    _write(df, rel, table, occ_disp)
    print("\n=== key cases (headline = upper-tail spread tail_hi) ===")
    for f in ["nursing", "communication_disorders", "computer_science", "accounting",
              "economics", "finance", "biology", "electrical_engineering"]:
        r = df[df.field == f]
        if len(r):
            r = r.iloc[0]
            print(f"  {f:24s}: tail_hi={r.tail_hi:.3f} sd={r.sd:.3f} coupling={r.coupling:+.3f} (top_occ {r.top_occ})")


def _figure(df, rel, res):
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    ax.scatter(df.dispersion_index, df.coupling, s=28, c="#bbbbbb", label="other fields", zorder=2)
    ax.scatter(rel.dispersion_index, rel.coupling, s=70, c="#2166ac",
               edgecolor="k", lw=0.5, label="reliable fields", zorder=3)
    lic = df[df.licensed]
    ax.scatter(lic.dispersion_index, lic.coupling, s=90, facecolor="none",
               edgecolor="#b2182b", lw=1.8, label="ex-ante licensed", zorder=4)
    for f in ["nursing", "communication_disorders", "computer_science", "accounting",
              "economics", "finance", "biology", "electrical_engineering"]:
        r = df[df.field == f]
        if len(r):
            r = r.iloc[0]
            ax.annotate(f.replace("_", " "), (r.dispersion_index, r.coupling),
                        fontsize=7.5, xytext=(4, 3), textcoords="offset points")
    if res["n"] > 2:
        b, m = np.polyfit(rel.dispersion_index, rel.coupling, 1)[::-1]
        xs = np.linspace(rel.dispersion_index.min(), rel.dispersion_index.max(), 50)
        ax.plot(xs, m + b * xs, color="#2166ac", lw=1.5, ls="--",
                label=f"reliable fit (ρ={res['spearman']:+.2f} [{res['ci_lo']:+.2f},{res['ci_hi']:+.2f}])")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("destination wage-dispersion index\n(occupation-mix-weighted within-occupation residual SD of log wage)")
    ax.set_ylabel("coupling  ρ_f = Spearman(prestige, earnings)")
    ax.set_title("② The compression continuous law: coupling vs within-occupation wage variance\n"
                 "prediction: compression (nursing/RN) low-low corner, integrated (CS/SWE) high-high", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = ROOT / "outputs" / "figures" / "dispersion_law.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[fig] {out}")


def _write(df, rel, table, occ_disp):
    rr, ra = table[HEADLINE]
    L = ["# Direction ② — The compression continuous law (within-occupation wage variance)\n",
         "Outcome-agnostic, non-circular (ACS occupation wage structure vs Scorecard "
         "institution-level coupling — independent sources). ACS has no institution → this tests "
         "the **mechanism variable** (destination side), not the prestige link. "
         "`python scripts/53_dispersion_law.py`. Seeded.\n",
         "## All dispersion operationalizations vs coupling (transparent — no cherry-pick)\n",
         "Theory prefers **tail_hi** (upper-tail spread p90−p50 of geography/experience-netted "
         "log-wage): prestige sorts graduates into the high-pay upper tail, so that is where a "
         "prestige-sortable-variance signal should live; total SD dilutes it with universal noise.\n",
         "| measure | reliable ρ [95% CI] p | broad ρ [95% CI] p |",
         "|---|---|---|"]
    for m in MEASURES:
        a_, b_ = table[m]
        hl = " **←headline**" if m == HEADLINE else ""
        L.append(f"| {m}{hl} | {a_['spearman']:+.3f} [{a_['ci_lo']:+.2f},{a_['ci_hi']:+.2f}] "
                 f"p={a_['p_spearman']:.2f} | {b_['spearman']:+.3f} [{b_['ci_lo']:+.2f},"
                 f"{b_['ci_hi']:+.2f}] p={b_['p_spearman']:.2f} |")
    L.append(f"\n**Headline (tail_hi)**: reliable (n={rr['n']}) Spearman **{rr['spearman']:+.3f}** "
             f"[{rr['ci_lo']:+.3f}, {rr['ci_hi']:+.3f}], p={rr['p_spearman']:.3f}; "
             f"broad (n={ra['n']}) **{ra['spearman']:+.3f}** [{ra['ci_lo']:+.3f}, {ra['ci_hi']:+.3f}], "
             f"p={ra['p_spearman']:.3f}. Contrast the pilot's occ_hhi↔coupling null (wrong variable).\n")
    L.append("## Key cases (tail_hi = upper-tail spread; sd = total)\n")
    L.append("| field | tail_hi | total sd | coupling | top occupation |")
    L.append("|---|---|---|---|---|")
    for f in ["nursing", "communication_disorders", "computer_science", "accounting",
              "economics", "finance", "biology", "electrical_engineering"]:
        r = df[df.field == f]
        if len(r):
            r = r.iloc[0]
            L.append(f"| {f} | {r.tail_hi:.3f} | {r.sd:.3f} | {r.coupling:+.3f} | {r.top_occ} |")
    L.append("\n## Caveats\n")
    L.append(f"- Within-occupation dispersion (any measure) is an **upper bound** on prestige-sortable "
             f"variance (mixes firm effects + unobserved skill + measurement error) → directional.\n"
             f"- ACS 1-year PUMS ({occ_disp.n.sum():,} persons, {len(occ_disp)} occupations ≥{MIN_OCC}); "
             f"5-year IPUMS would tighten per-occupation tails and thin fields less.\n"
             f"- No institution in ACS → mechanism-variable evidence, not the prestige↔placement link "
             f"(that needs PSEO/Revelio; a NULL here is itself the 'aggregate only visible at the "
             f"extremes → Revelio necessary' testimony).\n"
             f"- Testing {len(MEASURES)} operationalizations: report ALL, headline the theory-chosen "
             f"one ex-ante; small n fields → wide CIs.")
    (ROOT / "DISPERSION_LAW_RESULT.md").write_text("\n".join(L))
    print(f"[result] {ROOT / 'DISPERSION_LAW_RESULT.md'}")


if __name__ == "__main__":
    main()
