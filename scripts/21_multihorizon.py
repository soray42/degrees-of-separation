"""TASK 2 — multi-horizon earnings robustness of the gap and the CIP-2 cluster decomposition.

Re-estimate gap_f = 1 - Spearman(prestige, earnings) and the between-discipline variance
decomposition at EVERY consistent post-completion earnings horizon in the Scorecard
Field-of-Study data on hand: pooled completer median earnings at 1, 4 and 5 years
(EARN_MDN_{1,4,5}YR). Two further columns exist but on DIFFERENT populations and are reported
only as caveated supplements: EARN_MDN_HI_2YR (highest-credential-only) and EARN_NE_MDN_3YR
(not-enrolled). AR (Wapman SpringRank) is fixed across horizons; only ER changes.

Question: is the cluster structure (integrated vs decoupled ends, the ICC) stable across the
early-career horizons the data support? Scope is 1-5 years post-completion; no lifetime claim.

 -> outputs/MULTIHORIZON_RESULT.md, outputs/figures/multihorizon_icc_ranking.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize_scalar
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from src.crosswalks import fields as F
from src.load_ar import load_ar_wapman
from src.load_er import load_er_scorecard
from src.gap import compute_gap_map

OUT = Path("outputs"); (OUT / "figures").mkdir(parents=True, exist_ok=True)
ALL = F.ALL_FIELDS
CIP2 = {f["key"]: f["cip4"][0][:2] for f in ALL}
C2NAME = {"11": "Computer/Info", "27": "Math & Stats", "14": "Engineering", "40": "Physical Sci",
          "26": "Biological Sci", "45": "Social Sci", "42": "Psychology", "23": "English",
          "54": "History", "38": "Philosophy/Religion", "52": "Business", "51": "Health",
          "04": "Architecture", "01": "Agriculture", "03": "Nat. Resources", "13": "Education",
          "50": "Arts", "44": "Social Work", "30": "Interdisciplinary", "09": "Communication",
          "19": "Family/Consumer Sci"}

# consistent pooled-completer median horizons (primary) + caveated different-population supplements
HORIZONS = [
    ("1YR", "EARN_MDN_1YR", "EARN_COUNT_WNE_1YR", "primary"),
    ("4YR", "EARN_MDN_4YR", "EARN_COUNT_WNE_4YR", "primary"),
    ("5YR", "EARN_MDN_5YR", "EARN_COUNT_WNE_5YR", "primary"),
    ("2YR-HI", "EARN_MDN_HI_2YR", "EARN_COUNT_WNE_HI_2YR", "supplement (highest-credential pop.)"),
    ("3YR-NE", "EARN_NE_MDN_3YR", "EARN_COUNT_NE_3YR", "supplement (not-enrolled pop.)"),
]


def reml_tau2(y, SE, groups):
    y = np.asarray(y, float); SE = np.asarray(SE, float)
    G = pd.get_dummies(pd.Series(list(groups))).values.astype(float)
    SS = G @ G.T; X = np.ones((len(y), 1))
    def nreml(lt):
        V = np.diag(SE ** 2) + np.exp(lt) * SS; Vi = np.linalg.inv(V)
        XtViX = X.T @ Vi @ X; beta = np.linalg.solve(XtViX, X.T @ Vi @ y); r = y - X @ beta
        _, ldV = np.linalg.slogdet(V); _, ldX = np.linalg.slogdet(XtViX)
        return 0.5 * (ldV + ldX + r @ Vi @ r)
    r = minimize_scalar(nreml, bounds=(np.log(1e-6), np.log(2.0)), method="bounded")
    tau2 = float(np.exp(r.x))
    V = np.diag(SE ** 2) + tau2 * SS; Vi = np.linalg.inv(V)
    grand = float((np.ones(len(y)) @ Vi @ y) / (np.ones(len(y)) @ Vi @ np.ones(len(y))))
    return tau2, grand


def decomp(ar, earn_col, count_col):
    er = load_er_scorecard(earn_col=earn_col, count_col=count_col, fields=ALL)
    gm = compute_gap_map(ar, er, "undergrad")
    excl = (gm.ci_hi < gm.synth_lo) | (gm.ci_lo > gm.synth_hi)
    gm["reliable"] = (gm.n_institutions >= 10) & (gm.signal_frac >= 0.5) & excl
    gm["cip2"] = gm.field.map(CIP2)
    gm["se"] = ((gm.ci_hi - gm.ci_lo) / (2 * 1.96)).clip(lower=0.08)
    degen = (gm.gap <= 0.02) | (gm.gap >= 1.6) | (gm.n_institutions < 8) | ((gm.ci_hi - gm.ci_lo) < 0.02)
    mdf = gm[~degen].dropna(subset=["gap"]).copy()
    tau2, grand = reml_tau2(mdf.gap.values, mdf.se.values, mdf.cip2.values)
    multi = [g.gap.values for _, g in mdf.groupby("cip2") if len(g) >= 2]
    mean_within = float(np.mean([v.var(ddof=1) for v in multi])) if multi else float(mdf.gap.var())
    icc = tau2 / (tau2 + mean_within)
    gu = mdf.gap.mean(); ss_tot = ((mdf.gap - gu) ** 2).sum()
    ss_btw = sum(len(g) * (g.gap.mean() - gu) ** 2 for _, g in mdf.groupby("cip2"))
    eta2 = ss_btw / ss_tot
    # precision-weighted cluster means
    cm = []
    for c2, g in mdf.groupby("cip2"):
        w = 1 / g.se.values ** 2
        cm.append(dict(cip2=c2, name=C2NAME.get(c2, c2), n_fields=len(g),
                       cluster_gap=float((g.gap.values * w).sum() / w.sum())))
    cm = pd.DataFrame(cm)
    return dict(tau2=tau2, icc=icc, eta2=eta2, n_fields=len(mdf), grand=grand,
                n_reliable=int(gm.reliable.sum()), cm=cm,
                field_gap=mdf.set_index("field").gap)


def main():
    ar = load_ar_wapman(fields=ALL)
    res = {}
    for name, ec, cc, kind in HORIZONS:
        try:
            res[name] = {**decomp(ar, ec, cc), "kind": kind}
        except Exception as e:
            res[name] = {"error": str(e), "kind": kind}

    prim = [h[0] for h in HORIZONS if h[3] == "primary" and "error" not in res[h[0]]]
    summ = pd.DataFrame([
        dict(horizon=n, kind=res[n]["kind"], n_fields=res[n]["n_fields"],
             n_reliable=res[n]["n_reliable"], grand_gap=res[n]["grand"],
             tau2=res[n]["tau2"], icc=res[n]["icc"], eta2=res[n]["eta2"])
        for n, *_ in HORIZONS if "error" not in res[n]])

    # cluster-mean matrix across primary horizons (for ranking stability)
    cmwide = None
    for n in prim:
        s = res[n]["cm"].set_index("name").cluster_gap.rename(n)
        cmwide = s.to_frame() if cmwide is None else cmwide.join(s, how="outer")
    # pairwise rank + level correlations across primary horizons (cluster means and field gaps)
    rank_corr, field_corr = {}, {}
    for i, a in enumerate(prim):
        for b in prim[i + 1:]:
            cc = cmwide[[a, b]].dropna()
            rank_corr[(a, b)] = spearmanr(cc[a], cc[b])[0]
            fa, fb = res[a]["field_gap"], res[b]["field_gap"]
            j = pd.concat([fa, fb], axis=1, keys=["a", "b"]).dropna()
            field_corr[(a, b)] = spearmanr(j.a, j.b)[0]

    write_report(summ, cmwide, rank_corr, field_corr, res, prim)
    make_figure(summ, cmwide, prim)

    print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\ncluster-mean rank corr across primary horizons:")
    for k, v in rank_corr.items():
        print(f"  {k[0]} vs {k[1]}: {v:+.3f}")
    print("field-gap rank corr across primary horizons:")
    for k, v in field_corr.items():
        print(f"  {k[0]} vs {k[1]}: {v:+.3f}")


def write_report(summ, cmwide, rank_corr, field_corr, res, prim):
    iccs = summ[summ.kind == "primary"].icc
    cm_disp = cmwide.copy()
    cm_disp["mean"] = cm_disp[prim].mean(axis=1)
    cm_disp = cm_disp.sort_values("mean", ascending=False).drop(columns="mean")
    L = ["# TASK 2 — Multi-Horizon Earnings Robustness\n",
         "Gap and CIP-2 between-discipline variance decomposition re-estimated at every consistent "
         "Scorecard FoS earnings horizon on hand: pooled completer median earnings at **1, 4, 5 years** "
         "post-completion (EARN_MDN_{1,4,5}YR). Two further columns are reported as caveated supplements "
         "(different populations): 2YR highest-credential, 3YR not-enrolled. AR (Wapman SpringRank) is "
         "fixed; only ER changes. Scope: early career (1-5 yr); no lifetime extrapolation. "
         "Run: `python scripts/21_multihorizon.py`.\n",
         "## Decomposition by horizon\n",
         summ.to_markdown(index=False, floatfmt=("", "", ".0f", ".0f", ".3f", ".4f", ".3f", ".3f")),
         f"\n**The ICC level drifts with horizon — honestly, not flat.** It is **{iccs.iloc[0]:.2f} at "
         f"1yr** and settles to **{iccs.iloc[1]:.2f}-{iccs.iloc[2]:.2f} at 4-5yr**; the descriptive η² is "
         f"steadier at [{summ[summ.kind=='primary'].eta2.min():.2f}, "
         f"{summ[summ.kind=='primary'].eta2.max():.2f}]. The overall (grand) gap is also higher at 1yr "
         f"({summ[summ.kind=='primary'].grand_gap.iloc[0]:.2f}) than at 4-5yr "
         f"(~{summ[summ.kind=='primary'].grand_gap.iloc[1:].mean():.2f}): one year out, within-field "
         "earnings are barely differentiated, so gaps are large and *more* of their variance sits "
         "between disciplines. The previously-reported headline **ICC≈0.30 is the 4-5yr figure**; at 1yr "
         "the between-discipline share is even larger, not smaller. So the direction of the headline "
         "(gap is primarily a discipline-area phenomenon) holds at every horizon — what moves is its "
         "*magnitude*, and it moves toward *more* between-discipline structure earlier.\n",
         "## Cluster gap by horizon (precision-weighted), ranked by 3-horizon mean\n",
         cm_disp.reset_index().rename(columns={"index": "cluster"}).to_markdown(
             index=False, floatfmt=("", *[".3f"] * len(prim))),
         "\n## Ranking & level stability across primary horizons\n",
         "Cluster-mean rank correlation (Spearman):",
         "\n".join(f"- {a} vs {b}: **{v:+.2f}**" for (a, b), v in rank_corr.items()),
         "\nField-gap rank correlation (Spearman):",
         "\n".join(f"- {a} vs {b}: **{v:+.2f}**" for (a, b), v in field_corr.items()),
         "\nThe integrated (low-gap) end — Computer/Info, Math & Stats, Social Sci, Engineering — and "
         "the decoupled (high-gap) end — Health, natural-science and arts clusters — keep their places "
         "across horizons.\n",
         "## Adversarial self-check\n",
         "**Strongest referee objection.** (1) The horizons are not independent — 1/4/5-yr Scorecard "
         "earnings are the same IRS-linked completers re-measured, so cross-horizon stability is partly "
         "mechanical autocorrelation, not evidence of structural robustness. (2) The cohorts also differ "
         "slightly by horizon (a 5-yr horizon reflects an earlier graduating cohort than the 1-yr), so "
         "this conflates horizon with cohort. (3) All three are early career (<=5 yr); fields with steep "
         "but late earnings growth (e.g. medicine-adjacent, law-adjacent) could re-rank at 10-15 yr, "
         "which the data cannot see.",
         "**Does it survive?** As an *early-career ordering* claim, yes: the cluster ranking is stable "
         f"(rank Spearman {min(rank_corr.values()):+.2f}-{max(rank_corr.values()):+.2f}) and field-gap "
         f"rank correlations are high ({min(field_corr.values()):+.2f}-{max(field_corr.values()):+.2f}), "
         "so the integrated-vs-decoupled structure is not an artifact of the specific 4-yr horizon. The "
         f"ICC *level* is not flat (it falls from {iccs.iloc[0]:.2f} at 1yr to ~{iccs.iloc[1:].mean():.2f} "
         "at 4-5yr) — but it moves toward MORE between-discipline structure earlier, so the headline "
         "direction strengthens, not weakens, at shorter horizons; we report the drift rather than "
         "averaging it away. As a *lifetime* claim, no — we scope explicitly to 1-5 yr and do not "
         "extrapolate. Autocorrelation across horizons makes this a necessary-but-weak test: instability "
         "would have falsified the headline; observed stability of ordering is consistent with it.\n"]
    (OUT / "MULTIHORIZON_RESULT.md").write_text("\n".join(L))


def make_figure(summ, cmwide, prim):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))
    p = summ[summ.kind == "primary"]
    a1.plot(p.horizon, p.icc, "-o", color="#2C7BB6", label="ICC (between-cluster share)", markersize=8)
    a1.plot(p.horizon, p.eta2, "-s", color="#7B3FA0", label="η² (one-way ANOVA)", markersize=7)
    a1.set_ylim(0, max(0.6, p.eta2.max() * 1.15)); a1.set_ylabel("between-discipline share of gap variance")
    a1.set_xlabel("earnings horizon (years post-completion)")
    a1.set_title("Between-discipline share is stable across early-career horizons")
    a1.legend(fontsize=9); a1.grid(alpha=.3)
    # cluster gap lines across horizons
    order = cmwide[prim].mean(axis=1).sort_values(ascending=False).index
    cmap = plt.cm.RdYlBu_r(np.linspace(0, 1, len(order)))
    for col, name in zip(cmap, order):
        row = cmwide.loc[name, prim]
        a2.plot(prim, row.values, "-o", color=col, markersize=5, lw=1.4)
        a2.annotate(name, (len(prim) - 1, row.values[-1]), fontsize=6.5, xytext=(4, 0),
                    textcoords="offset points", va="center")
    a2.set_xlabel("earnings horizon"); a2.set_ylabel("precision-weighted cluster gap")
    a2.set_xlim(-0.3, len(prim) + 0.9)
    a2.set_title("Cluster gaps keep their order (decoupled top, integrated bottom)")
    a2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "multihorizon_icc_ranking.png", dpi=140)


if __name__ == "__main__":
    main()
