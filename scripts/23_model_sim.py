"""Structural model of the AR-ER gap: Monte-Carlo confirmation of P1-P5 + Task-1 recovery.

Model (one parameter vector, NOT tuned per fact). Field f, institutions i:
  (theta_A, theta_M) ~ N(0, [[1,rho],[rho,1]])        rho[f] = field alignment; 1-rho = divergence
  A   = theta_A + sigma_A * eps                        academic prestige (ranks academic value, noisy)
  w(tau,t) = exp(-tau*t)                               credential weight (Altonji-Pierret learning;
                                                       decreasing in observability tau and horizon t)
  S   = w*theta_A + (1-w)*theta_M                      systematic earnings (drifts toward market value)
  S_lam = (1-lambda) * S                               licensing compresses cross-institution signal
  sigma_E(t)^2 = sigma_inf^2 + sigma0^2 * exp(-delta t) sort/sampling noise, decreasing in t
  E(t) = S_lam + sigma_E(t)*eps
  gap(t) = 1 - Spearman_i(A, E(t))

Key analytic facts (Pearson proxy, validated against Spearman by MC):
  corr(A,E) = (1-lam)(w+(1-w)rho) / sqrt((1+sigma_A^2)*((1-lam)^2*B + sigma_E^2))
  with B = w^2+(1-w)^2+2w(1-w)rho.
  * floor (sigma_E small): gap -> 1 - (1-lam)(w+(1-w)rho)/sqrt((1+sigma_A^2)((1-lam)^2 B + sigma_inf^2));
    increasing in (1-rho); at rho=1 it is independent of w/tau.
  * licensing raises gap ONLY when sigma_E>0 (pure scaling is rank-invariant) -> licensing x noise.
  * tau enters via w; d gap/d tau magnitude scales with (1-rho), ->0 as rho->1.

Outcome-agnostic: each P is checked; a miss is reported, not tuned away.
 -> outputs/figures/model_*.png, data/interim/model_results.json
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = Path("outputs"); (OUT / "figures").mkdir(parents=True, exist_ok=True)

# ----- ONE structural parameter vector (calibration; not re-tuned per fact) -----
P = dict(N=150, sigma_A=0.5, sigma_inf=0.30, sigma0=1.00, delta=0.70)
RES = {"params": P}


def w_cred(tau, t):
    return np.exp(-tau * t)


def sig_E(t):
    return np.sqrt(P["sigma_inf"] ** 2 + P["sigma0"] ** 2 * np.exp(-P["delta"] * t))


def sim_field(rho, tau, lam, t, rng, N=None):
    N = N or P["N"]
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    z = rng.standard_normal((N, 2)) @ L.T
    thA, thM = z[:, 0], z[:, 1]
    A = thA + P["sigma_A"] * rng.standard_normal(N)
    w = w_cred(tau, t)
    S = w * thA + (1 - w) * thM
    E = (1 - lam) * S + sig_E(t) * rng.standard_normal(N)
    return 1 - spearmanr(A, E)[0], A, E


def analytic_gap(rho, tau, lam, t):
    w = w_cred(tau, t); sE2 = P["sigma_inf"] ** 2 + P["sigma0"] ** 2 * np.exp(-P["delta"] * t)
    K = w + (1 - w) * rho
    B = w ** 2 + (1 - w) ** 2 + 2 * w * (1 - w) * rho
    corr = (1 - lam) * K / np.sqrt((1 + P["sigma_A"] ** 2) * ((1 - lam) ** 2 * B + sE2))
    return 1 - corr


def mc_gap(rho, tau, lam, t, R=60, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([sim_field(rho, tau, lam, t, rng)[0] for _ in range(R)]))


# ============================ P1 — horizon ============================
def p1():
    taus, lam = 0.20, 0.10
    ts = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 6, 8])
    rhos = [0.90, 0.60, 0.30]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    out = {}
    for rho, c in zip(rhos, ["#2C7BB6", "#7B3FA0", "#D6202A"]):
        g = [mc_gap(rho, taus, lam, t, seed=int(t * 100) + int(rho * 10)) for t in ts]
        ga = [analytic_gap(rho, taus, lam, t) for t in ts]
        ax.plot(ts, g, "-o", color=c, label=f"1-rho={1-rho:.1f} (MC)", markersize=5)
        ax.plot(ts, ga, "--", color=c, alpha=.5, lw=1)
        floor = analytic_gap(rho, taus, lam, 30)
        out[f"div_{1-rho:.1f}"] = dict(gap_1yr=g[1], gap_4yr=g[5], gap_5yr=g[6],
                                       floor=floor, falls_1_to_5=bool(g[6] < g[1]))
    ax.axvspan(4, 5, color="#ffe9b0", alpha=.4, zorder=0, label="4-5yr (near floor)")
    ax.set_xlabel("horizon t (years post-completion)"); ax.set_ylabel("gap = 1 - Spearman(A, E)")
    ax.set_title("P1: gap falls in t toward a floor that increases in divergence (1-rho)\n"
                 "(dashed = analytic Pearson approx; MC = Spearman)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "figures" / "model_gap_vs_t.png", dpi=140)
    RES["P1"] = out
    return out


# ============================ P2 — licensing ============================
def p2():
    rho, tau = 0.50, 0.20
    lams = np.linspace(0, 0.85, 12)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    out = {}
    for t, c in [(1, "#D6202A"), (4, "#7B3FA0"), (8, "#2C7BB6")]:
        g = [mc_gap(rho, tau, lam, t, seed=int(lam * 1000) + t) for lam in lams]
        ax.plot(lams, g, "-o", color=c, label=f"t={t}yr", markersize=4)
        # slope of gap on licensure (gap at lam=1 extrapolated via fit over [0,0.85])
        b = np.polyfit(lams, g, 1)[0]
        out[f"t{t}"] = dict(slope_dgap_dlam=float(b), gap_lam0=g[0], gap_lam085=g[-1])
    ax.set_xlabel("licensure intensity lambda"); ax.set_ylabel("gap")
    ax.set_title("P2: gap increases in licensure lambda (steeper early = licensing x noise)\n"
                 "licensing compresses the systematic signal; with noise present, SNR & rank-corr fall")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "figures" / "model_gap_vs_lambda.png", dpi=140)
    RES["P2"] = out
    return out


# ============================ P3 — pipeline NULL ============================
def p3():
    rng = np.random.default_rng(42); M = 250
    rho = rng.uniform(0.2, 0.95, M); tau = rng.uniform(0.05, 0.40, M); lam = rng.uniform(0, 0.7, M)
    gap = np.array([sim_field(rho[i], tau[i], lam[i], 4, rng)[0] for i in range(M)])
    absn_indep = rng.uniform(0, 1, M)                       # absorption INDEPENDENT of primitives
    absn_conf = 0.6 * (1 - rho) + 0.4 * rng.uniform(0, 1, M)  # absorption CONFOUNDED with divergence
    def fit(a):
        X = sm.add_constant(np.column_stack([lam, a]))
        m = sm.OLS(gap, X).fit(cov_type="HC1")
        return m.params, m.bse, m.pvalues
    p_i, se_i, pv_i = fit(absn_indep)
    p_c, se_c, pv_c = fit(absn_conf)
    out = dict(
        indep=dict(b_lam=float(p_i[1]), b_abs=float(p_i[2]), p_abs=float(pv_i[2]),
                   ci_abs=[float(p_i[2] - 1.96 * se_i[2]), float(p_i[2] + 1.96 * se_i[2])]),
        confounded=dict(b_lam=float(p_c[1]), b_abs=float(p_c[2]), p_abs=float(pv_c[2]),
                        ci_abs=[float(p_c[2] - 1.96 * se_c[2]), float(p_c[2] + 1.96 * se_c[2])]))
    RES["P3"] = out
    return out


# ============================ P4 — observability ============================
def p4():
    taus = np.linspace(0.03, 0.5, 10); rhos = [0.95, 0.7, 0.4]
    fig, ax = plt.subplots(figsize=(7.5, 5.5)); out = {}
    for rho, c in zip(rhos, ["#2C7BB6", "#7B3FA0", "#D6202A"]):
        g = [mc_gap(rho, tau, 0.1, 4, seed=int(tau * 1000) + int(rho * 10)) for tau in taus]
        ax.plot(taus, g, "-o", color=c, label=f"1-rho={1-rho:.2f}", markersize=4)
        out[f"div_{1-rho:.2f}"] = dict(dgap_dtau=float(np.polyfit(taus, g, 1)[0]))
    ax.set_xlabel("observability tau (employer-learning speed)"); ax.set_ylabel("gap (t=4yr)")
    ax.set_title("P4: d gap/d tau scales with divergence (1-rho); ~0 when rho->1\n"
                 "near the floor the tau MAIN effect is weak -> use the tau x (1-rho) interaction")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "figures" / "model_gap_vs_tau.png", dpi=140)

    # pooled main effect vs interaction (panel of fields) across horizons:
    # the tau MAIN effect is non-monotonic in t (weak under early noise -> strong mid-convergence
    # -> weak again near the floor), so a weak empirical 4-5yr gradient = fields near their floors.
    rng = np.random.default_rng(7); M = 300
    rho = rng.uniform(0.2, 0.95, M); tau = rng.uniform(0.05, 0.4, M); lam = rng.uniform(0, 0.6, M)
    for t in (1, 4, 8, 12):
        gap = np.array([sim_field(rho[i], tau[i], lam[i], t, rng)[0] for i in range(M)])
        div = 1 - rho
        Xm = sm.add_constant(np.column_stack([tau]))
        Xi = sm.add_constant(np.column_stack([tau, div, tau * div]))
        mm = sm.OLS(gap, Xm).fit(cov_type="HC1"); mi = sm.OLS(gap, Xi).fit(cov_type="HC1")
        out[f"pooled_t{t}"] = dict(b_tau_main=float(mm.params[1]), p_tau_main=float(mm.pvalues[1]),
                                   b_tau_x_div=float(mi.params[3]), p_tau_x_div=float(mi.pvalues[3]))
    RES["P4"] = out
    return out


# ============ panel horizon: connect to TASK-2 grand-gap-falls + ICC-falls ============
def panel_horizon():
    rng = np.random.default_rng(99); M = 300
    rho = rng.uniform(0.2, 0.95, M); tau = rng.uniform(0.05, 0.4, M); lam = rng.uniform(0, 0.7, M)
    div = 1 - rho
    g1 = np.array([sim_field(rho[i], tau[i], lam[i], 1, rng)[0] for i in range(M)])
    g5 = np.array([sim_field(rho[i], tau[i], lam[i], 5, rng)[0] for i in range(M)])
    out = dict(mean_gap_1yr=float(g1.mean()), mean_gap_5yr=float(g5.mean()),
               between_var_1yr=float(g1.var()), between_var_5yr=float(g5.var()),
               grand_falls=bool(g5.mean() < g1.mean()),
               between_var_falls=bool(g5.var() < g1.var()),
               corr_dgap_div=float(spearmanr(g5 - g1, div)[0]),
               corr_dgap_lam=float(spearmanr(g5 - g1, lam)[0]))
    RES["panel_horizon"] = out
    return out


# ============================ P5 — lead-lag direction ============================
def p5():
    rng = np.random.default_rng(11); M = 300
    rho = rng.uniform(0.2, 0.95, M); tau = rng.uniform(0.1, 0.4, M); lam = rng.uniform(0, 0.4, M)
    t1, t2 = 1.0, 5.0
    dA, dE, dgap = [], [], []
    for i in range(M):
        N = P["N"]; L = np.linalg.cholesky(np.array([[1, rho[i]], [rho[i], 1.0]]))
        z = rng.standard_normal((N, 2)) @ L.T; thA, thM = z[:, 0], z[:, 1]
        A = thA + P["sigma_A"] * rng.standard_normal(N)          # prestige: SAME (slow prior) at both waves
        def E_at(t):
            w = w_cred(tau[i], t)
            return (1 - lam[i]) * (w * thA + (1 - w) * thM) + sig_E(t) * rng.standard_normal(N)
        E1, E2 = E_at(t1), E_at(t2)
        from scipy.stats import rankdata
        dE.append(np.mean(np.abs(rankdata(E2) - rankdata(E1))) / N)   # earnings-rank drift
        dgap.append((1 - spearmanr(A, E2)[0]) - (1 - spearmanr(A, E1)[0]))
    dE = np.array(dE); dgap = np.array(dgap); div = 1 - rho
    out = dict(corr_dErank_div=float(spearmanr(dE, div)[0]),
               corr_dgap_div=float(spearmanr(dgap, div)[0]),
               mean_dgap_lowdiv=float(dgap[div < 0.35].mean()),
               mean_dgap_highdiv=float(dgap[div > 0.55].mean()))
    RES["P5"] = out
    return out


# ===================== Task-1 residual recovery =====================
def task1_recovery():
    rng = np.random.default_rng(2024); M = 300
    rho = rng.uniform(0.2, 0.95, M); tau = rng.uniform(0.05, 0.4, M)
    lam_indep = rng.uniform(0, 0.7, M)                       # lambda _|_ rho  (legitimate netting)
    lam_conf = np.clip(0.5 * (1 - rho) + 0.25 * rng.uniform(0, 1, M), 0, 0.9)  # lambda correlated w/ rho
    absn = rng.uniform(0, 1, M)                              # absorption (not in mechanism)
    div = 1 - rho

    def recover(lam):
        gap = np.array([sim_field(rho[i], tau[i], lam[i], 4, rng)[0] for i in range(M)])
        X = sm.add_constant(np.column_stack([lam, absn]))
        m = sm.OLS(gap, X).fit()
        resid = gap - m.fittedvalues
        return dict(b_lam=float(m.params[1]), b_abs=float(m.params[2]),
                    corr_resid_div=float(spearmanr(resid, div)[0]),
                    corr_lam_div=float(spearmanr(lam, div)[0]))
    out = dict(lambda_indep_of_rho=recover(lam_indep),
               lambda_confounded_with_rho=recover(lam_conf))
    RES["task1"] = out

    # figure: residual vs true divergence (legitimate case)
    gap = np.array([sim_field(rho[i], tau[i], lam_indep[i], 4, rng)[0] for i in range(M)])
    X = sm.add_constant(np.column_stack([lam_indep, absn]))
    resid = gap - sm.OLS(gap, X).fit().fittedvalues
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(div, resid, c=tau, cmap="viridis", s=22)
    plt.colorbar(sc, label="observability tau")
    ax.set_xlabel("true valuation divergence (1 - rho)")
    ax.set_ylabel("Task-1 residual  (gap net of lambda, absorption)")
    ax.set_title(f"Task-1 fix: residual ESTIMATES divergence (Spearman "
                 f"{out['lambda_indep_of_rho']['corr_resid_div']:+.2f})\n"
                 "spread along tau = the (1-w(tau)) earnings-anchor-departure factor")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "model_task1_recovery.png", dpi=140)
    return out


def main():
    print("P1 horizon:"); print(p1())
    print("\nP2 licensing:"); print(p2())
    print("\nP3 pipeline null:"); print(p3())
    print("\nP4 observability:"); print(p4())
    print("\nP5 lead-lag:"); print(p5())
    print("\nPanel horizon (grand-gap / between-var vs TASK-2):"); print(panel_horizon())
    print("\nTask-1 recovery:"); print(task1_recovery())
    (Path("data/interim") / "model_results.json").write_text(json.dumps(RES, indent=2))
    print("\nsaved -> data/interim/model_results.json + outputs/figures/model_*.png")


if __name__ == "__main__":
    main()
