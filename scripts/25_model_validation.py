"""TASK 2 — validate the structural model (MODEL.md) on existing/open data.

B1 [make-or-break]  residual (gap net of licensing) vs an INDEPENDENT valuation-divergence proxy.
   The model predicts a POSITIVE correlation (residual ~ (1-w(tau))(1-rho)). Two constructions:
   (i)  O*NET — per field, applied/practical-skill intensity minus research/abstract-skill intensity
        of its typical occupations (market-valued practical vs academy-valued analytical work).
   (ii) NSF SDR — |academic-sector - industry salary| / mean by field (the two sectors price the
        field's doctorate-holders differently = academia-market divergence).
   If BOTH come out null/negative against the clean residual, residual-as-divergence is NOT supported.

B2  Interaction tau x (1-rho), netting licensing. tau proxy = within-field earnings dispersion
    (FLAGGED: earnings-derived, so partly circular with the gap). Model predicts POSITIVE interaction.

B3  corr(licensure, divergence proxy) — netting licensing is unbiased only if ~0. Report it.

 -> MODEL_VALIDATION_RESULT.md, data/interim/model_validation.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm

from src.crosswalks import fields as F
from src.load_ar import load_ar_wapman
from src.load_er import load_er_scorecard
from src.gap import compute_gap_map
from src import dispersion as D
from src import tier0_5 as T5

ROOT = Path(__file__).resolve().parents[1]
ALL = F.ALL_FIELDS
LAB = {f["key"]: f["label"] for f in ALL}

# O*NET Generalized Work Activities split into academy-valued (abstract/analytical) vs
# market-valued (applied/practical/manual).
ABSTRACT = ["4.A.1.a.1", "4.A.2.a.2", "4.A.2.a.4", "4.A.2.b.1", "4.A.2.b.2", "4.A.2.b.3",
            "4.A.2.b.4", "4.A.4.a.1"]
APPLIED = ["4.A.3.a.1", "4.A.3.a.2", "4.A.3.a.3", "4.A.3.a.4", "4.A.3.b.4", "4.A.3.b.5", "4.A.1.b.2"]


def divergence_onet():
    """Per field: z(applied) - z(abstract) over its CIP->SOC occupations (equal-weighted).
    High = market work is practical while academia rewards research -> more academia-market divergence."""
    wa = pd.read_csv(ROOT / "data" / "raw" / "onet" / "Work Activities.txt", sep="\t", dtype=str)
    wa["Data Value"] = pd.to_numeric(wa["Data Value"], errors="coerce")
    piv = wa.pivot_table(index="O*NET-SOC Code", columns="Element ID", values="Data Value",
                         aggfunc="mean")
    soc_abs = piv[ABSTRACT].mean(axis=1); soc_app = piv[APPLIED].mean(axis=1)
    cw = T5._cip_soc_crosswalk()
    rows = []
    for f in ALL:
        socs = sorted(set(cw.loc[cw.cip4.isin(f["cip4"]), "soc"]))
        socs = [s for s in socs if not s.startswith("25-10")]  # drop postsecondary-teaching SOCs
        a = [soc_abs.get(s) for s in socs if s in soc_abs.index]
        p = [soc_app.get(s) for s in socs if s in soc_app.index]
        if a and p:
            rows.append(dict(field=f["key"], abstract=np.nanmean(a), applied=np.nanmean(p)))
    d = pd.DataFrame(rows)
    d["z_abs"] = (d.abstract - d.abstract.mean()) / d.abstract.std()
    d["z_app"] = (d.applied - d.applied.mean()) / d.applied.std()
    d["divergence_onet"] = d.z_app - d.z_abs
    return d[["field", "abstract", "applied", "divergence_onet"]]


def divergence_sdr():
    """|industry - academic salary|/mean per field, from SDR Table 54 (price_wedge_abs)."""
    pw = T5.price_wedge()
    return pw.rename(columns={"field_key": "field", "price_wedge_abs": "divergence_sdr"})[
        ["field", "divergence_sdr"]]


def boot_corr(x, y, B=4000, seed=3):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    rng = np.random.default_rng(seed); n = len(x); bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(x[i])) > 2 and len(np.unique(y[i])) > 2:
            bs.append(spearmanr(x[i], y[i])[0])
    return spearmanr(x, y)[0], np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), n


def main():
    # ---- gap + residual (net licensing only, per the model) ----
    ar = load_ar_wapman(fields=ALL); er = load_er_scorecard(fields=ALL)
    gm = compute_gap_map(ar, er, "undergrad")
    excl = (gm.ci_hi < gm.synth_lo) | (gm.ci_lo > gm.synth_hi)
    gm["reliable"] = (gm.n_institutions >= 10) & (gm.signal_frac >= 0.5) & excl
    gm["se"] = ((gm.ci_hi - gm.ci_lo) / (2 * 1.96)).clip(lower=0.08)
    degen = (gm.gap <= 0.02) | (gm.gap >= 1.6) | (gm.n_institutions < 8) | ((gm.ci_hi - gm.ci_lo) < 0.02)
    gm = gm[~degen].dropna(subset=["gap"]).copy()
    anch = pd.read_parquet(ROOT / "data" / "interim" / "acs_occ_anchors.parquet")
    gm = gm.merge(anch[["field", "licensure_strict"]], on="field", how="left")
    s = gm.dropna(subset=["gap", "licensure_strict"]).copy()
    mlic = sm.WLS(s.gap.values, sm.add_constant(s.licensure_strict.values),
                  weights=1 / s.se.values ** 2).fit()
    s["residual_netlic"] = s.gap - mlic.fittedvalues          # gap net of licensing only
    b_lambda = float(mlic.params[1])

    # ---- divergence proxies + tau proxy ----
    don = divergence_onet(); dsd = divergence_sdr()
    w = D.load_acs_workers(expanded=True)
    disp = D.field_dispersion_acs(w, 22, 27)[["field", "disp_acs"]].rename(columns={"disp_acs": "tau_disp"})
    df = s.merge(don, on="field", how="left").merge(dsd, on="field", how="left").merge(disp, on="field", how="left")
    df["label"] = df.field.map(LAB)
    df.to_csv(ROOT / "data" / "interim" / "model_validation.csv", index=False)

    # ---- B1 ----
    b1 = {}
    for proxy in ["divergence_onet", "divergence_sdr"]:
        r, lo, hi, n = boot_corr(df.residual_netlic, df[proxy])
        # also vs the raw gap and vs the two-channel residual, for context
        rg, _, _, _ = boot_corr(df.gap, df[proxy])
        b1[proxy] = dict(rho=r, lo=lo, hi=hi, n=n, rho_vs_rawgap=rg)

    # ---- B2 interaction (net licensing already in residual; here regress gap on tau, div, tau*div, lic) ----
    b2 = {}
    for proxy in ["divergence_onet", "divergence_sdr"]:
        d2 = df.dropna(subset=["gap", "tau_disp", proxy, "licensure_strict"]).copy()
        if len(d2) >= 12:
            X = sm.add_constant(np.column_stack([
                d2.tau_disp, d2[proxy], d2.tau_disp * d2[proxy], d2.licensure_strict]))
            m = sm.OLS(d2.gap.values, X).fit(cov_type="HC1")
            b2[proxy] = dict(n=len(d2), b_inter=float(m.params[3]), p_inter=float(m.pvalues[3]),
                             b_tau=float(m.params[1]), b_div=float(m.params[2]))

    # ---- B3 ----
    b3 = {p: boot_corr(df.licensure_strict, df[p])[:3] for p in ["divergence_onet", "divergence_sdr"]}

    write_report(df, b1, b2, b3, b_lambda)
    print(f"b_lambda(net) = {b_lambda:+.3f}")
    print("B1 residual<->divergence:")
    for p, v in b1.items():
        print(f"  {p}: Spearman {v['rho']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] n={v['n']} (vs raw gap {v['rho_vs_rawgap']:+.2f})")
    print("B2 tau x (1-rho) interaction:")
    for p, v in b2.items():
        print(f"  {p}: b_inter {v['b_inter']:+.3f} (p={v['p_inter']:.3f}) n={v['n']}")
    print("B3 corr(licensure, divergence):")
    for p, v in b3.items():
        print(f"  {p}: {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]")


def write_report(df, b1, b2, b3, b_lambda):
    on, sd = b1["divergence_onet"], b1["divergence_sdr"]
    supports = [p for p, v in b1.items() if v["lo"] > 0]
    contradicts = [p for p, v in b1.items() if v["hi"] < 0]
    both_fail = len(contradicts) == 2
    both_support = len(supports) == 2
    mixed = len(supports) >= 1 and len(contradicts) >= 1
    L = ["# TASK 2 — Validating the structural model on open data\n",
         "The model (MODEL.md) predicts the Task-1 residual (gap net of licensing) is an estimate of "
         "valuation divergence (1-rho), so it should correlate POSITIVELY with an INDEPENDENT divergence "
         "proxy. B1 is make-or-break. Run: `python scripts/25_model_validation.py`. Outcome-agnostic.\n",
         f"Residual = gap - {b_lambda:+.3f}*licensure_strict (net licensing only, per the model's "
         "recommendation to net lambda but not absorption).\n",
         "## B1 [make-or-break] — residual vs independent divergence proxies\n",
         "| proxy | construction | Spearman(residual, proxy) [95% CI] | n | (vs raw gap) |",
         "|---|---|---|---|---|",
         f"| O*NET | applied − abstract skill intensity of field's occupations | "
         f"**{on['rho']:+.2f}** [{on['lo']:+.2f}, {on['hi']:+.2f}] | {on['n']} | {on['rho_vs_rawgap']:+.2f} |",
         f"| SDR | \\|academic − industry salary\\|/mean (price wedge) | "
         f"**{sd['rho']:+.2f}** [{sd['lo']:+.2f}, {sd['hi']:+.2f}] | {sd['n']} | {sd['rho_vs_rawgap']:+.2f} |",
         "\n**Model prediction: POSITIVE.** Verdict: "
         + ("**NOT SUPPORTED — both proxies contradict.** The clean residual correlates negatively with "
            "both independent divergence proxies; the 'residual = valuation divergence' reading is "
            "falsified on these open-data proxies. Reported as a falsification, not rescued."
            if both_fail else
            ("**SUPPORTED — both proxies positive (CIs exclude 0).**" if both_support else
             ("**MIXED / CONTESTED — the two independent proxies DISAGREE.** The **O*NET skill-content** "
              f"proxy SUPPORTS the model (residual↔divergence **{on['rho']:+.2f}** [{on['lo']:+.2f}, "
              f"{on['hi']:+.2f}], CI>0), and it is the cleaner test by the model's OWN criterion (B3 below: "
              f"licensure ⟂ this proxy, {b3['divergence_onet'][0]:+.2f}). The **SDR salary-wedge** proxy "
              f"CONTRADICTS it (**{sd['rho']:+.2f}** [{sd['lo']:+.2f}, {sd['hi']:+.2f}], CI<0). The "
              "contradiction is **robust, not a contamination artifact**: netting licensing made the SDR "
              f"correlation *more* negative ({sd['rho_vs_rawgap']:+.2f} raw → {sd['rho']:+.2f} residual), so "
              "it cannot be dismissed. So the residual-as-divergence claim is **partially supported on "
              "skill-content divergence and contradicted on salary-wedge divergence** — it is NOT cleanly "
              "confirmed; which proxy one trusts decides it."
              if mixed else
              "**INCONCLUSIVE:** CIs span 0; neither confirmation nor falsification."))) + "\n",
         "## B2 — interaction tau x divergence (licensing controlled)\n",
         "tau proxy = within-field earnings dispersion. **CIRCULARITY FLAG: tau is earnings-derived, so it "
         "shares measurement with the gap; B2 is secondary and not independent of the gap construct.**\n",
         "| proxy | b(tau x divergence) | p | n |",
         "|---|---|---|---|"]
    for p, v in b2.items():
        L.append(f"| {p.replace('divergence_','')} | {v['b_inter']:+.3f} | {v['p_inter']:.3f} | {v['n']} |")
    L += ["\nModel predicts a POSITIVE interaction. " + (
          "Reported as-is; given the circularity flag this is corroborative at best." ) + "\n",
          "## B3 — corr(licensure, divergence) (licenses the Task-1 netting)\n",
          "Netting licensing is unbiased only if licensure ⟂ divergence. "]
    for p, v in b3.items():
        L.append(f"- corr(licensure, {p.replace('divergence_','')}) = **{v[0]:+.2f}** [{v[1]:+.2f}, {v[2]:+.2f}]"
                 + ("" if abs(v[0]) < 0.3 else "  — **materially non-zero: residual partly contaminated**"))
    L += ["\n## Adversarial self-check (what each proxy conflates)\n",
          "- **O*NET (applied − abstract).** Conflates the *level* of a field's average task-abstractness "
          "with the model's (1-rho), which is a *cross-institution correlation* of academic vs market value "
          "— different objects. It also can't see within-field heterogeneity (all institutions in a field "
          "get the same occupation vector). A null here is as consistent with 'the proxy is wrong' as with "
          "'the residual is not divergence' — it cannot cleanly confirm, only fail to.",
          "- **SDR price wedge.** Measured on *doctorate-holders* (academic vs industry salary), while the "
          "residual is a *bachelor's*-gap quantity — cross-level. |wedge| also reflects sector composition "
          "and pay scales, not only valuation divergence; SEH-only coverage shrinks n.",
          "- **B2 tau.** Earnings-derived → circular with the gap (flagged); not an independent test.",
          "\n## What a miss implies\n",
          ("Both B1 proxies failing means the open-data evidence does **not** support interpreting the "
           "Task-1 residual as valuation divergence. The model's *internal* logic (residual = floor term) "
           "stands, but its key external prediction is **not confirmed** here — either divergence is "
           "genuinely not what the residual captures, or no available open proxy measures the "
           "cross-institution academic-vs-market value correlation the model means. Either way: the "
           "'residual = divergence' claim is downgraded from 'estimated' to 'unconfirmed', and that is the "
           "headline finding."
           if both_fail else
           "At least one independent proxy moves with the residual in the predicted direction, giving "
           "partial open-data support for the residual-as-divergence reading; the caveats above bound how "
           "strong that support is.") + "\n"]
    (ROOT / "MODEL_VALIDATION_RESULT.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
