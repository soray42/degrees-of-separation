"""TASK 3 (optional) — can the model reproduce the ICC-vs-horizon DECLINE?

Empirically the between-cluster ICC FALLS with horizon (0.50@1yr -> 0.30@4-5yr). The default
model (scripts/23) gets between-field variance slightly RISING (early common noise compresses all
fields together; they fan out toward heterogeneous floors). Proposed mechanism: early earnings
compression is DISCIPLINE-VARYING and strong (some disciplines have huge, structured early
compression that resolves). We make the early sort-noise sigma0 field-specific and tie part of it
to licensing (licensed fields are compressed hardest early), and test whether between-field
variance now FALLS 1yr->5yr -- WITHOUT breaking P2 (licensing), P3 (absorption null), P5 (lead-lag).

 -> MODEL_ICC_REFINEMENT.md, outputs/figures/model_icc_refinement.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = Path("outputs"); (OUT / "figures").mkdir(parents=True, exist_ok=True)
P = dict(N=150, sigma_A=0.5, sigma_inf=0.30, delta=0.70)


def sim_gap(rho, tau, lam, t, sig0, rng, N=None):
    N = N or P["N"]
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    z = rng.standard_normal((N, 2)) @ L.T
    thA, thM = z[:, 0], z[:, 1]
    A = thA + P["sigma_A"] * rng.standard_normal(N)
    w = np.exp(-tau * t)
    S = w * thA + (1 - w) * thM
    sigE = np.sqrt(P["sigma_inf"] ** 2 + sig0 ** 2 * np.exp(-P["delta"] * t))
    E = (1 - lam) * S + sigE * rng.standard_normal(N)
    return 1 - spearmanr(A, E)[0]


def panel(sig0_mode, M=300, seed=99):
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.2, 0.95, M); tau = rng.uniform(0.05, 0.4, M); lam = rng.uniform(0, 0.7, M)
    absn = rng.uniform(0, 1, M)
    if sig0_mode == "constant":
        sig0 = np.full(M, 1.0)
    else:  # discipline-varying + tied to licensing (early compression strongest in licensed fields)
        sig0 = 0.5 + 2.2 * lam + 1.4 * rng.uniform(0, 1, M)   # strong, field-specific, lambda-linked
    g1 = np.array([sim_gap(rho[i], tau[i], lam[i], 1, sig0[i], rng) for i in range(M)])
    g5 = np.array([sim_gap(rho[i], tau[i], lam[i], 5, sig0[i], rng) for i in range(M)])
    div = 1 - rho
    out = dict(mode=sig0_mode, mean1=g1.mean(), mean5=g5.mean(),
               bvar1=g1.var(), bvar5=g5.var(), bvar_falls=bool(g5.var() < g1.var()),
               grand_falls=bool(g5.mean() < g1.mean()),
               # P2 licensing at t=4 (slope), P3 absorption null, P5 lead-lag
               p2_slope=float(np.polyfit(lam, np.array(
                   [sim_gap(rho[i], tau[i], lam[i], 4, sig0[i], rng) for i in range(M)]), 1)[0]),
               p5_corr_dgap_div=float(spearmanr(g5 - g1, div)[0]))
    g4 = np.array([sim_gap(rho[i], tau[i], lam[i], 4, sig0[i], rng) for i in range(M)])
    X = sm.add_constant(np.column_stack([lam, absn]))
    m = sm.OLS(g4, X).fit(cov_type="HC1")
    out["p3_b_abs"] = float(m.params[2]); out["p3_p_abs"] = float(m.pvalues[2])
    out["p2_b_lam_t4"] = float(m.params[1])
    out["_g1"], out["_g5"], out["_div"] = g1, g5, div
    return out


def main():
    base = panel("constant"); refined = panel("disc_varying")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, d, ttl in [(ax[0], base, "constant sigma0 (default)"), (ax[1], refined, "discipline-varying sigma0")]:
        a.bar(["1yr", "5yr"], [d["bvar1"], d["bvar5"]], color=["#D6202A", "#2C7BB6"])
        a.set_title(f"{ttl}\nbetween-field var {d['bvar1']:.3f}->{d['bvar5']:.3f} "
                    + ("(FALLS)" if d["bvar_falls"] else "(rises)"))
        a.set_ylabel("between-field gap variance")
    fig.suptitle("TASK 3: does discipline-varying early compression reproduce the ICC decline?")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "model_icc_refinement.png", dpi=140)

    breaks = []
    if refined["p2_b_lam_t4"] <= 0: breaks.append("P2 licensing")
    if abs(refined["p3_b_abs"]) > 0.1 and refined["p3_p_abs"] < 0.05: breaks.append("P3 absorption null")
    if refined["p5_corr_dgap_div"] <= 0: breaks.append("P5 lead-lag")

    L = ["# TASK 3 — ICC-vs-horizon: can a discipline-varying early-compression recalibration reproduce the decline?\n",
         "Empirically between-cluster ICC falls 0.50@1yr -> 0.30@4-5yr; the default model gets "
         "between-field variance slightly RISING. Test: make early sort-noise sigma0 field-specific and "
         "strong, tied to licensing (licensed fields compressed hardest early). Run: "
         "`python scripts/26_icc_refinement.py`.\n",
         "## Between-field variance by horizon\n",
         "| sigma0 | between-var 1yr | between-var 5yr | direction | grand gap falls? |",
         "|---|---|---|---|---|",
         f"| constant (default) | {base['bvar1']:.4f} | {base['bvar5']:.4f} | "
         f"{'FALLS' if base['bvar_falls'] else 'rises'} | {base['grand_falls']} |",
         f"| discipline-varying + λ-linked | {refined['bvar1']:.4f} | {refined['bvar5']:.4f} | "
         f"{'**FALLS**' if refined['bvar_falls'] else 'rises'} | {refined['grand_falls']} |",
         f"\n**Result:** the discipline-varying / licensing-linked early-compression recalibration "
         + ("**reproduces the decline** — between-field variance now falls with horizon, matching the "
            "empirical ICC drop, because licensed/heavily-compressed disciplines have very large but "
            "structured 1-yr gaps that resolve by 5 yr (between-field spread is largest at 1 yr)."
            if refined["bvar_falls"] else
            "does **not** reproduce the decline (between-field variance still rises) — the floor "
            "differentiation still dominates; reported as an unresolved miss.") + "\n",
         "## Does it break P2 / P3 / P5?\n",
         f"- **P2 licensing** (t=4 slope ∂gap/∂λ): **{refined['p2_b_lam_t4']:+.3f}** (default "
         f"{base['p2_b_lam_t4']:+.3f}) — {'still positive, holds' if refined['p2_b_lam_t4'] > 0 else 'BROKEN'}.",
         f"- **P3 absorption null**: b_abs = **{refined['p3_b_abs']:+.3f}** (p={refined['p3_p_abs']:.2f}) — "
         f"{'still null, holds' if not (abs(refined['p3_b_abs'])>0.1 and refined['p3_p_abs']<0.05) else 'BROKEN'}.",
         f"- **P5 lead-lag** (corr(Δgap, 1-ρ)): **{refined['p5_corr_dgap_div']:+.2f}** (default "
         f"{base['p5_corr_dgap_div']:+.2f}) — {'still positive, holds' if refined['p5_corr_dgap_div'] > 0 else 'BROKEN'}.",
         "\n**Verdict:** " + (
             f"a defensible recalibration (early compression discipline-varying and λ-linked) reproduces "
             f"the ICC decline **without breaking** P2/P3/P5 — so the default model's ICC miss is a "
             f"CALIBRATION limitation, not a structural one. The cost is one added degree of freedom "
             f"(field-specific σ0 instead of common σ0), disclosed here, not tuned silently."
             if refined["bvar_falls"] and not breaks else
             (f"reproducing the decline REQUIRES breaking {', '.join(breaks)} — a genuine tension, reported."
              if breaks else
              "the recalibration does not reproduce the decline; the miss stands as an honest structural "
              "limitation of the model.")) + "\n",
         "## Adversarial self-check\n",
         "**Objection.** Tying σ0 to λ and adding strong field-specific early noise is exactly the extra "
         "freedom the default model lacked — so 'reproducing' the decline is close to fitting it. The "
         "mechanism (licensed disciplines hugely compressed at 1 yr, resolving by 5 yr) is plausible and "
         "matches the empirical 1-yr picture (Health/credential gaps near 1.0 at 1 yr), but it is asserted, "
         "not independently measured. **Honest status:** the refinement shows the decline is *achievable* "
         "within the model's structure with a disclosed extra parameter, not that the model *predicts* it "
         "from first principles. We do not fold this into the headline calibration.\n"]
    (Path("MODEL_ICC_REFINEMENT.md")).write_text("\n".join(L))
    print(f"baseline: bvar {base['bvar1']:.4f}->{base['bvar5']:.4f} ({'falls' if base['bvar_falls'] else 'rises'})")
    print(f"refined:  bvar {refined['bvar1']:.4f}->{refined['bvar5']:.4f} ({'FALLS' if refined['bvar_falls'] else 'rises'})")
    print(f"P2 slope t4={refined['p2_b_lam_t4']:+.3f} | P3 b_abs={refined['p3_b_abs']:+.3f}(p={refined['p3_p_abs']:.2f}) | P5 corr={refined['p5_corr_dgap_div']:+.2f}")
    print(f"breaks: {breaks or 'none'}")


if __name__ == "__main__":
    main()
