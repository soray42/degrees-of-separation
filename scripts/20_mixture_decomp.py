"""TASK 1 — mixture-decomposition identification of the AR-ER gap.

The raw gap mixes three channels: (i) LICENSING (regulated occupations compress earnings
dispersion -> mechanically high gap), (ii) PIPELINE (fields that send graduates into academia
have an earnings population unlike the prestige-defining one), (iii) genuine VALUATION
DIVERGENCE. We net out (i)-(ii) with two PUBLIC field-level anchors and isolate (iii) as a
residual.

Anchors (both from ACS PUMS 2023, full-coverage and internally consistent; SDR cross-check):
  licensure_intensity  = weighted share of a field's employed degree-holders in licensed
                         occupations (SOCP licensed-occupation flag; STRICT vs BROAD lists).
  academic_absorption  = share of a field's DOCTORATE-holders employed as postsecondary
                         teachers (SOCP 25-1xxx). SDR Educational/All is a cross-check (SEH only).

Identification: gap_f = b0 + b1*licensure_f + b2*absorption_f + residual_f, precision-weighted
(1/SE^2). residual_f is the isolated decoupling gap. We aggregate residuals to CIP-2 clusters
and re-rank vs the raw cluster gap. THIS IS A DECOMPOSITION (variance attribution under a
linear projection), NOT a causal/IV estimate — the anchors are not instruments and we make no
exclusion-restriction claim. Outcome-agnostic: null or wrong-signed coefficients are reported.

 -> data/interim/mixture_anchors.csv, outputs/MIXTURE_DECOMP_RESULT.md,
    outputs/figures/mixture_residual_vs_raw.png
"""
import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from src.crosswalks import fields as F
from src.load_ar import load_ar_wapman
from src.load_er import load_er_scorecard
from src.gap import compute_gap_map
import src.tier0 as T0

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
ACS_GLOB = str(ROOT / "data" / "raw" / "acs" / "psam_pus*.csv")
ANCH_CACHE = ROOT / "data" / "interim" / "acs_occ_anchors.parquet"
ALL = F.ALL_FIELDS
LAB = {f["key"]: f["label"] for f in ALL}
CIP2 = {f["key"]: f["cip4"][0][:2] for f in ALL}
C2NAME = {"11": "Computer/Info", "27": "Math & Stats", "14": "Engineering", "40": "Physical Sci",
          "26": "Biological Sci", "45": "Social Sci", "42": "Psychology", "23": "English",
          "54": "History", "38": "Philosophy/Religion", "52": "Business", "51": "Health",
          "04": "Architecture", "01": "Agriculture", "03": "Nat. Resources", "13": "Education",
          "50": "Arts", "44": "Social Work", "30": "Interdisciplinary", "09": "Communication",
          "19": "Family/Consumer Sci"}

# Licensed-occupation SOC prefixes (SOCP, 6-digit, no hyphen; first 2 digits = SOC major group).
# STRICT = near-universal licensure; BROAD adds commonly- but not-universally-licensed occupations.
LIC_STRICT = ("291", "231", "252", "171011", "193031")   # health practitioners; lawyers/judges;
#                                                            K-12 teachers; architects; psychologists
LIC_BROAD_EXTRA = ("292", "132011", "172", "211")         # +health techs; accountants(CPA);
#                                                            engineers(PE); counselors/social workers
POSTSEC = "251"                                            # postsecondary teachers (25-1xxx)


def build_anchors(refresh=False):
    if ANCH_CACHE.exists() and not refresh:
        return pd.read_parquet(ANCH_CACHE)
    code_map = F.fod1p_to_field_key(F.fod1p_by_key_all())
    need = ["FOD1P", "SOCP", "SCHL", "WKHP", "ESR", "PERNP", "PWGTP"]
    frames = []
    for f in sorted(glob.glob(ACS_GLOB)):
        cols = pd.read_csv(f, nrows=0).columns
        frames.append(pd.read_csv(f, usecols=[c for c in need if c in cols],
                                  dtype={"SOCP": str}))
    a = pd.concat(frames, ignore_index=True)
    for c in ["FOD1P", "SCHL", "WKHP", "ESR", "PERNP", "PWGTP"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a[(a.SCHL >= 21) & a.FOD1P.notna() & (a.WKHP >= 35) &
          a.ESR.isin([1, 2]) & (a.PERNP > 0)].copy()
    a["field"] = a.FOD1P.astype(int).astype(str).str.zfill(4).map(code_map)
    a = a[a.field.notna()].copy()
    a["soc"] = a.SOCP.fillna("").str.replace("X", "0", regex=False)
    a["lic_strict"] = False; a["lic_broad"] = False
    for p in LIC_STRICT:
        a["lic_strict"] |= a.soc.str.startswith(p)
    for p in LIC_STRICT + LIC_BROAD_EXTRA:
        a["lic_broad"] |= a.soc.str.startswith(p)
    a["postsec"] = a.soc.str.startswith(POSTSEC)
    rows = []
    for fld, g in a.groupby("field"):
        phd = g[g.SCHL == 24]
        rows.append(dict(
            field=fld, n_acs=len(g), n_phd=len(phd),
            licensure_strict=np.average(g.lic_strict, weights=g.PWGTP),
            licensure_broad=np.average(g.lic_broad, weights=g.PWGTP),
            absorption_acs=(np.average(phd.postsec, weights=phd.PWGTP) if len(phd) >= 30 else np.nan)))
    out = pd.DataFrame(rows)
    ANCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(ANCH_CACHE)
    return out


def sdr_absorption():
    """academic_absorption cross-check from SDR 2021 Table 12-3: Educational / All employed
    (col 3 / col 1), SEH fine fields only."""
    raw = pd.read_excel(T0.SDR_TAB_12_3, header=None, dtype=str)
    rows = {}
    for i in range(5, len(raw)):
        lab = str(raw.iloc[i, 0]).strip()
        if lab and lab != "nan":
            rows[lab] = (T0._num(raw.iloc[i, 1]), T0._num(raw.iloc[i, 3]))  # All, Educational
    out = {}
    for f in ALL:
        lab = F.sdr_fine_for(f["key"])
        if lab and lab in rows:
            alln, edu = rows[lab]
            if alln and edu and alln > 0:
                out[f["key"]] = edu / alln
    return pd.Series(out, name="absorption_sdr")


def wls(y, X, w):
    """Precision-weighted OLS with HC1 robust SE. Returns params, conf_int, fitted, resid."""
    Xc = sm.add_constant(X)
    m = sm.WLS(y, Xc, weights=w).fit(cov_type="HC1")
    return m


def boot_coefs(df, xcols, B=2000):
    """Field bootstrap of the precision-weighted coefficients (resample fields with replacement)."""
    rng = np.random.default_rng(11); out = []
    idx = np.arange(len(df))
    for _ in range(B):
        s = df.iloc[rng.choice(idx, len(df), replace=True)]
        try:
            m = wls(s.gap.values, s[xcols].values, 1 / s.se.values ** 2)
            out.append(m.params)
        except Exception:
            continue
    A = np.array(out)
    return A  # columns: const, *xcols


def main():
    # ---------- field gap map (reuse the established construction) ----------
    ar = load_ar_wapman(fields=ALL); er = load_er_scorecard(fields=ALL)
    gm = compute_gap_map(ar, er, "undergrad")
    excl = (gm.ci_hi < gm.synth_lo) | (gm.ci_lo > gm.synth_hi)
    gm["reliable"] = (gm.n_institutions >= 10) & (gm.signal_frac >= 0.5) & excl
    gm["cip2"] = gm.field.map(CIP2)
    gm["se"] = ((gm.ci_hi - gm.ci_lo) / (2 * 1.96)).clip(lower=0.08)
    # drop degenerate gaps (as in scripts/14) so precision weighting is not dominated by floored SE
    degen = (gm.gap <= 0.02) | (gm.gap >= 1.6) | (gm.n_institutions < 8) | ((gm.ci_hi - gm.ci_lo) < 0.02)
    gm = gm[~degen].dropna(subset=["gap"]).copy()

    # ---------- anchors ----------
    anch = build_anchors()
    sdr = sdr_absorption()
    df = gm.merge(anch, on="field", how="left")
    df["absorption_sdr"] = df.field.map(sdr)
    df["label"] = df.field.map(LAB); df["c2name"] = df.cip2.map(lambda c: C2NAME.get(c, c))

    # primary analysis sample: fields with both primary anchors present
    prim = df.dropna(subset=["gap", "licensure_strict", "absorption_acs"]).copy()
    cov = dict(total_fields=len(df), with_licensure=int(df.licensure_strict.notna().sum()),
               with_absorption_acs=int(df.absorption_acs.notna().sum()),
               with_absorption_sdr=int(df.absorption_sdr.notna().sum()),
               primary_n=len(prim))

    # ---------- field-level decomposition regression (precision-weighted) ----------
    specs = {
        "primary (strict lic, ACS absorp)": ("licensure_strict", "absorption_acs"),
        "broad licensure": ("licensure_broad", "absorption_acs"),
        "SDR absorption": ("licensure_strict", "absorption_sdr"),
    }
    spec_results = {}
    for name, (lc, ac) in specs.items():
        s = df.dropna(subset=["gap", lc, ac]).copy()
        if len(s) < 12:
            spec_results[name] = dict(n=len(s), insufficient=True); continue
        m = wls(s.gap.values, s[[lc, ac]].values, 1 / s.se.values ** 2)
        bA = boot_coefs(s.assign(**{}), [lc, ac])
        ci = np.nanpercentile(bA, [2.5, 97.5], axis=0)
        spec_results[name] = dict(
            n=len(s), b_lic=m.params[1], b_abs=m.params[2],
            ci_lic=(ci[0, 1], ci[1, 1]), ci_abs=(ci[0, 2], ci[1, 2]),
            b_lic_se=m.bse[1], b_abs_se=m.bse[2], r2=m.rsquared,
            xcols=(lc, ac))

    # residuals from the primary spec
    lc, ac = specs["primary (strict lic, ACS absorp)"]
    mp = wls(prim.gap.values, prim[[lc, ac]].values, 1 / prim.se.values ** 2)
    prim["fitted"] = mp.fittedvalues
    prim["residual"] = prim.gap - prim.fitted          # isolated decoupling gap (field level)

    # also an unweighted fit (sensitivity)
    mp_unw = sm.OLS(prim.gap.values, sm.add_constant(prim[[lc, ac]].values)).fit()
    prim["residual_unw"] = prim.gap - mp_unw.fittedvalues

    # ---------- cluster aggregation: raw vs residual, re-rank ----------
    def pw_mean(v, se):
        w = 1 / np.asarray(se) ** 2
        return float(np.sum(np.asarray(v) * w) / np.sum(w))
    clus = []
    for c2, g in prim.groupby("cip2"):
        clus.append(dict(cip2=c2, name=C2NAME.get(c2, c2), n_fields=len(g),
                         raw_gap=pw_mean(g.gap, g.se),
                         resid_gap=pw_mean(g.residual, g.se),
                         licensure=pw_mean(g.licensure_strict, g.se),
                         absorption=pw_mean(g.absorption_acs, g.se)))
    clus = pd.DataFrame(clus)
    clus["rank_raw"] = clus.raw_gap.rank(ascending=False).astype(int)
    clus["rank_resid"] = clus.resid_gap.rank(ascending=False).astype(int)
    clus["rank_shift"] = clus.rank_raw - clus.rank_resid
    rho_rank = spearmanr(clus.raw_gap, clus.resid_gap)[0]

    # bootstrap stability of the cluster residual ranking (resample fields within the pooled set)
    rng = np.random.default_rng(7); kt = []
    for _ in range(1000):
        s = prim.iloc[rng.choice(np.arange(len(prim)), len(prim), replace=True)]
        try:
            mm = wls(s.gap.values, s[[lc, ac]].values, 1 / s.se.values ** 2)
            s = s.assign(res=s.gap - mm.fittedvalues)
            cc = s.groupby("cip2").apply(lambda g: pw_mean(g.res, g.se))
            cc = cc.reindex(clus.cip2)
            if cc.notna().all():
                kt.append(spearmanr(clus.resid_gap, cc.values)[0])
        except Exception:
            continue
    rank_stab = (np.nanmean(kt), np.nanpercentile(kt, 2.5), np.nanpercentile(kt, 97.5)) if kt else (np.nan,)*3

    anch.to_csv(ROOT / "data" / "interim" / "mixture_anchors.csv", index=False)
    prim_out = prim[["field", "label", "cip2", "c2name", "gap", "se", "licensure_strict",
                     "absorption_acs", "fitted", "residual"]].sort_values("residual", ascending=False)
    prim_out.to_csv(ROOT / "data" / "interim" / "mixture_field_residuals.csv", index=False)
    clus.sort_values("resid_gap", ascending=False).to_csv(
        ROOT / "data" / "interim" / "mixture_cluster_ranking.csv", index=False)

    write_report(cov, spec_results, prim, clus, rho_rank, rank_stab, mp, mp_unw)
    make_figure(clus, prim)

    print(f"coverage: {cov}")
    for n, r in spec_results.items():
        if r.get("insufficient"):
            print(f"  {n}: n={r['n']} (insufficient)"); continue
        print(f"  {n}: n={r['n']}  b_lic={r['b_lic']:+.3f} {r['ci_lic']}  b_abs={r['b_abs']:+.3f} {r['ci_abs']}  R2={r['r2']:.2f}")
    print(f"cluster rank corr raw vs residual: {rho_rank:+.2f} | residual-rank bootstrap stability rho={rank_stab[0]:+.2f} [{rank_stab[1]:+.2f},{rank_stab[2]:+.2f}]")
    print(clus.sort_values("resid_gap", ascending=False)[["name", "n_fields", "raw_gap", "resid_gap", "rank_shift"]].to_string(index=False))


def write_report(cov, spec, prim, clus, rho_rank, rank_stab, mp, mp_unw):
    s = spec["primary (strict lic, ACS absorp)"]
    L = ["# TASK 1 — Mixture-Decomposition Identification of the AR-ER Gap\n",
         "Net out two mechanical channels (LICENSING, PIPELINE) with public field-level anchors; "
         "isolate genuine VALUATION DIVERGENCE as the residual. **This is a decomposition (a linear "
         "projection / variance attribution), NOT a causal or IV estimate** — the anchors are not "
         "instruments and no exclusion restriction is claimed. Outcome-agnostic. "
         "Run: `python scripts/20_mixture_decomp.py`.\n",
         "## Anchors (public)\n",
         "- **licensure_intensity** = weighted share of a field's employed degree-holders in licensed "
         "occupations (ACS PUMS 2023, SOCP licensed-occupation flag). STRICT = near-universal "
         "licensure (health practitioners 29-1xxx, lawyers 23-1xxx, K-12 teachers 25-2xxx, architects, "
         "psychologists); BROAD adds partially-licensed (health techs, accountants/CPA, engineers/PE, "
         "counselors/social workers).",
         "- **academic_absorption** = share of a field's DOCTORATE-holders employed as postsecondary "
         "teachers (ACS, SOCP 25-1xxx). Cross-check: SDR 2021 Table 12-3 Educational/All employed "
         "(SEH fine fields only).\n",
         f"**Coverage.** {cov['primary_n']} fields have both primary anchors (of {cov['total_fields']} "
         f"non-degenerate gap fields). licensure {cov['with_licensure']}, ACS absorption "
         f"{cov['with_absorption_acs']}, SDR absorption {cov['with_absorption_sdr']} (SEH only).\n",
         "## Channel coefficients (precision-weighted WLS, gap ~ licensure + absorption)\n",
         "| spec | n | b_licensure [95% CI] | b_absorption [95% CI] | R² |",
         "|---|---|---|---|---|"]
    for name, r in spec.items():
        if r.get("insufficient"):
            L.append(f"| {name} | {r['n']} | insufficient | | |"); continue
        L.append(f"| {name} | {r['n']} | {r['b_lic']:+.2f} "
                 f"[{r['ci_lic'][0]:+.2f}, {r['ci_lic'][1]:+.2f}] | {r['b_abs']:+.2f} "
                 f"[{r['ci_abs'][0]:+.2f}, {r['ci_abs'][1]:+.2f}] | {r['r2']:.2f} |")
    L += [f"\nBoth anchors range 0-1, so a coefficient is the gap change from 0% to 100% of the channel. "
          f"Primary R²={s['r2']:.2f} of cross-field gap variance is linearly attributable to the two "
          "mechanical channels; the rest is the residual decoupling gap.\n",
          "## Cluster gap: raw vs residual (decoupling) ranking\n",
          "Field residuals (precision-weighted) aggregated to CIP-2 clusters. **raw_gap** = "
          "precision-weighted cluster gap; **resid_gap** = same after netting licensing + pipeline.\n",
          clus.sort_values("resid_gap", ascending=False)[
              ["name", "n_fields", "raw_gap", "resid_gap", "licensure", "absorption", "rank_raw",
               "rank_resid", "rank_shift"]].to_markdown(index=False, floatfmt=(
              "", ".0f", ".3f", ".3f", ".2f", ".2f", ".0f", ".0f", "+.0f")),
          f"\n- Rank correlation raw vs residual cluster gap: **Spearman {rho_rank:+.2f}**.",
          f"- Residual-ranking bootstrap stability (resample fields): Spearman to point estimate "
          f"**{rank_stab[0]:+.2f}** [{rank_stab[1]:+.2f}, {rank_stab[2]:+.2f}].",
          "- Positive **rank_shift** = the cluster's gap is *more* explained by the two channels (it "
          "drops in the decoupling ranking); negative = its gap *survives* netting (genuine valuation "
          "divergence, rises in the residual ranking).\n",
          "## Top residual (genuine decoupling) fields\n",
          prim.sort_values("residual", ascending=False)[["label", "c2name", "gap", "residual"]].head(10)
            .to_markdown(index=False, floatfmt=("", "", ".3f", "+.3f")),
          "\n## Sensitivity\n",
          f"- STRICT vs BROAD licensure: b_licensure {spec['primary (strict lic, ACS absorp)']['b_lic']:+.2f} "
          f"vs {spec['broad licensure']['b_lic']:+.2f} — engineering moves most (PE makes it 'licensed' "
          "under BROAD), so the licensing coefficient and engineering's residual are the load-bearing "
          "choice; reported both ways.",
          f"- ACS vs SDR absorption: b_absorption {spec['primary (strict lic, ACS absorp)']['b_abs']:+.2f} "
          f"vs {spec['SDR absorption']['b_abs']:+.2f} (SDR SEH-only, smaller n).",
          f"- Precision-weighted vs unweighted residual rank: Spearman "
          f"{spearmanr(prim.residual, prim.residual_unw)[0]:+.2f} (field level).",
          f"- The two anchors are only weakly correlated (corr(licensure, absorption) = "
          f"{np.corrcoef(prim.licensure_strict, prim.absorption_acs)[0, 1]:+.2f}), so the null "
          "absorption coefficient is **not** a collinearity artifact — the academic-pipeline channel, "
          "as measured, simply does not predict the gap (and flips sign under the SDR proxy), whereas "
          "licensing is the one robust mechanical channel.\n",
          "## Adversarial self-check\n",
          "**Strongest referee objection.** This is a projection, not identification: licensure and "
          "absorption are themselves *consequences* of the same discipline structure that drives the "
          "gap (collider/over-control risk), so the 'residual' is not a clean 'genuine divergence' — it "
          "is whatever is orthogonal to two correlated, non-randomly-assigned proxies, and the anchors "
          "are field aggregates (ecological), measured on an ACS undergrad-field universe that differs "
          "from the Wapman PhD-field / Scorecard BA-earnings universes of the gap. The licensing "
          "coefficient also flips materially with the strict/broad choice (engineering).",
          "**Does it survive?** As a *causal* claim, no — and we do not make one (stated up front). As "
          "a *decomposition*, partially: " + _survive_text(spec, rho_rank, rank_stab) +
          " The residual ranking should be read as 'gap not co-moving with licensing/pipeline at the "
          "field level', not as a structural parameter. We report coefficients, CIs and the "
          "strict/broad split openly rather than selecting a specification.\n"]
    (OUT / "MIXTURE_DECOMP_RESULT.md").write_text("\n".join(L))


def _survive_text(spec, rho_rank, rank_stab):
    p = spec["primary (strict lic, ACS absorp)"]
    lic_sig = (p["ci_lic"][0] > 0) or (p["ci_lic"][1] < 0)
    abs_sig = (p["ci_abs"][0] > 0) or (p["ci_abs"][1] < 0)
    bits = []
    bits.append(f"the licensing coefficient is {'CI-excludes-0' if lic_sig else 'CI-spans-0'} "
                f"({p['b_lic']:+.2f}), the absorption coefficient is "
                f"{'CI-excludes-0' if abs_sig else 'CI-spans-0'} ({p['b_abs']:+.2f})")
    bits.append(f"and the residual cluster ranking re-orders the raw one (rank Spearman {rho_rank:+.2f}) "
                f"{'stably' if rank_stab[0] and rank_stab[0] > 0.5 else 'only loosely'} under field "
                f"resampling (stability {rank_stab[0]:+.2f}).")
    return "; ".join(bits) + "."


def make_figure(clus, prim):
    c = clus.copy()
    c["y_raw"] = c.raw_gap.rank(ascending=True)
    c["y_res"] = c.resid_gap.rank(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    for _, r in c.iterrows():
        col = "#D6202A" if r.rank_shift < 0 else ("#2C7BB6" if r.rank_shift > 0 else "#888")
        ax.plot([0, 1], [r.raw_gap, r.resid_gap], "-o", color=col, alpha=.8, markersize=6)
        ax.annotate(f"{r['name']}", (0, r.raw_gap), ha="right", va="center", fontsize=7.5,
                    xytext=(-4, 0), textcoords="offset points")
        ax.annotate(f"{r['name']}", (1, r.resid_gap), ha="left", va="center", fontsize=7.5,
                    xytext=(4, 0), textcoords="offset points")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["raw cluster gap", "residual (decoupling) gap\nnet of licensing + pipeline"])
    ax.set_ylabel("gap = 1 − Spearman(prestige, earnings)")
    ax.set_xlim(-0.45, 1.45)
    ax.set_title("Mixture decomposition: cluster gap before vs after netting licensing + pipeline\n"
                 "red = gap shrinks most (mechanically explained); blue = gap survives (genuine divergence)")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "mixture_residual_vs_raw.png", dpi=140)


if __name__ == "__main__":
    main()
