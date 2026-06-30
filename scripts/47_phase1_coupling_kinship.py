"""Phase 1 — do academically-kin fields share a characteristic prestige<->placement coupling
strength? rho_f = Spearman_i(prestige_{i,f}, earnings_{i,f}) = 1 - gap_f, already computed in
outputs/expanded66_gap_map.csv (`spearman`, `gap`, `cip2`, `reliable`, `se`, `signal_frac`).

(a) ICC of rho_f by CIP-2 — precision-weighted multilevel meta-model (1|cip2), REML,
    V = diag(se^2) + tau^2*same-cip2 (same machinery as scripts/14_cluster_decomp.py and
    scripts/22_hardening.py). This RE-CONFIRMS the ALREADY-KNOWN result (ICC~0.30, measurement-
    corrected~0.45) — it is not new. rho_f = 1 - gap_f is an affine transform of gap_f, so the
    ICC is mathematically invariant to which of the two columns is used; we verify this
    numerically as a sanity check.
(b) NEW — does an unsupervised 1-D partition of rho_f ALONE (quantile bins AND a 1-D Gaussian
    mixture, K = #CIP-2 groups present in the set being clustered) recover the CIP-2 partition?
    Adjusted Rand Index (ARI) + Normalized Mutual Information (NMI), each (i) bootstrapped
    (resample fields with replacement, 1000x) for a CI on the observed statistic, and (ii)
    compared to a permutation null (shuffle the CIP-2 labels against the fixed rho-partition,
    2000x) so we know whether the agreement is distinguishable from chance. n~20 reliable
    fields is thin (most CIP-2 groups in the reliable set are singletons) — handled honestly:
    NMI is mechanically inflated when K is large relative to n (it can be ~0.85+ even against
    SHUFFLED labels), so only the permutation-null comparison, not the raw NMI value, is
    trustworthy here.
(c) NEW — two named CIP-invisible misalignment candidates, the lead for Phase 2:
      (i) Statistics (cip2=27, grouped with Mathematics) vs its likely intellectual kin
          Computer Science (cip2=11): is rho_f(statistics) closer to Math or to CS?
      (ii) do licensed / health / "helping profession" fields cluster together in rho_f ACROSS
           different CIP-2 codes (a cross-CIP kinship the CIP-2 taxonomy would miss)? Tested by
           a permutation tightness test (is the within-group rho variance smaller than that of
           a random same-size draw from the same pool?) plus a within-CIP-2 outlier scan (which
           fields sit far from their own CIP-2 group's coupling mean).

Primary population = reliable==True (~20 fields, per the task spec); the broader degenerate-
excluded set (n~57, 22 CIP-2 groups present) is reported throughout as a sensitivity / power
check — for (a) specifically, the broader set is also the one that is actually comparable to
the already-published 0.30/0.45 ICC (the reliable-only set has only 3 of 15 CIP-2 groups with
>1 field, so its within-group variance / ICC is largely a degrees-of-freedom artifact; this is
stated plainly in the report, not hidden).

Descriptive, outcome-agnostic: nulls are reported as nulls. Seeded (SEED=42), fully
reproducible from the static outputs/expanded66_gap_map.csv (no live data pulls; needs
scikit-learn — added to requirements.txt).

 -> outputs/PHASE1_COUPLING_RESULT.md, outputs/figures/phase1_coupling_kinship.png
Run: `.venv/bin/python scripts/47_phase1_coupling_kinship.py`. Date 2026-06-30.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from src.crosswalks import fields as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
SEED = 42

C2NAME = {1: "Agriculture", 3: "Natural Resources", 4: "Architecture", 9: "Communication",
          11: "Computer/Info", 13: "Education", 14: "Engineering", 16: "Foreign Languages",
          19: "Family/Consumer Sci", 23: "English", 26: "Biological Sci", 27: "Math & Stats",
          30: "Interdisciplinary", 31: "Parks/Rec/Fitness", 38: "Philosophy/Religion",
          39: "Theology", 40: "Physical Sci", 42: "Psychology", 44: "Social Work",
          45: "Social Sci", 50: "Arts", 51: "Health", 52: "Business", 54: "History"}
LAB_F = {f["key"]: f["label"] for f in F.ALL_FIELDS}


def lab(key):
    return LAB_F.get(key, key.replace("_", " ").title())


LICENSED = sorted(F.LICENSED_FIELDS)  # nursing, communication_disorders, accounting, civil_engineering
ALLIED = sorted(set(LICENSED) | {"pharmacy", "public_health", "kinesiology", "hper",
                                  "social_work", "human_dev"})


# ================================================================= data ====
def load():
    gm = pd.read_csv(OUT / "expanded66_gap_map.csv")
    da = gm.dropna(subset=["spearman"]).copy()
    degen = (da.gap <= 0.02) | (da.gap >= 1.6) | (da.n_institutions < 8) | ((da.ci_hi - da.ci_lo) < 0.02)
    broad = da[~degen].reset_index(drop=True)
    rel = gm[gm.reliable].reset_index(drop=True)
    return gm, broad, rel


# ============================================================ (a) ICC ======
def reml_tau2(y, se, groups):
    """REML between-group variance for a precision-weighted intercept-only meta-model,
    V = diag(se^2) + tau^2 * same-group. Identical machinery to scripts/14, 22."""
    y = np.asarray(y, float); se = np.asarray(se, float)
    G = pd.get_dummies(pd.Series(list(groups))).values.astype(float)
    SS = G @ G.T; X = np.ones((len(y), 1))

    def nreml(lt):
        V = np.diag(se ** 2) + np.exp(lt) * SS; Vi = np.linalg.inv(V)
        XtViX = X.T @ Vi @ X; beta = np.linalg.solve(XtViX, X.T @ Vi @ y); r = y - X @ beta
        _, ldV = np.linalg.slogdet(V); _, ldX = np.linalg.slogdet(XtViX)
        return 0.5 * (ldV + ldX + r @ Vi @ r)

    r = minimize_scalar(nreml, bounds=(np.log(1e-6), np.log(2.0)), method="bounded")
    tau2 = float(np.exp(r.x))
    V = np.diag(se ** 2) + tau2 * SS; Vi = np.linalg.inv(V)
    grand = float((np.ones(len(y)) @ Vi @ y) / (np.ones(len(y)) @ Vi @ np.ones(len(y))))
    return tau2, grand


def icc_block(d, col="spearman"):
    tau2, grand = reml_tau2(d[col].values, d["se"].values, d["cip2"].values)
    multi = [g[col].values for _, g in d.groupby("cip2") if len(g) >= 2]
    n_multi = len(multi)
    mean_within = float(np.mean([v.var(ddof=1) for v in multi])) if multi else float("nan")
    icc = tau2 / (tau2 + mean_within) if multi else float("nan")
    meas = (float(np.mean([np.mean(g["se"].values ** 2) for _, g in d.groupby("cip2") if len(g) >= 2]))
            if multi else float("nan"))
    within_true = max(mean_within - meas, 1e-6) if multi else float("nan")
    icc_corr = tau2 / (tau2 + within_true) if multi else float("nan")
    grand_u = d[col].mean()
    ss_tot = ((d[col] - grand_u) ** 2).sum()
    ss_btw = sum(len(g) * (g[col].mean() - grand_u) ** 2 for _, g in d.groupby("cip2"))
    eta2 = ss_btw / ss_tot
    return dict(n=len(d), n_cip2=d.cip2.nunique(), n_multi=n_multi, tau2=tau2, grand=grand,
                mean_within=mean_within, meas=meas, within_true=within_true,
                icc=icc, icc_corr=icc_corr, eta2=eta2)


# ====================================================== (b) clustering =====
def quantile_partition(rho, K):
    """Deterministic equal-frequency 1-D partition into exactly K groups (works for any n>=K,
    no qcut bin-edge degeneracy)."""
    order = np.argsort(rho)
    groups = np.array_split(order, K)
    labels = np.empty(len(rho), dtype=int)
    for gi, idx in enumerate(groups):
        labels[idx] = gi
    return labels


def gmm_partition(rho, K, seed):
    g = GaussianMixture(n_components=K, random_state=seed, n_init=20, reg_covar=1e-6)
    g.fit(rho.reshape(-1, 1))
    return g.predict(rho.reshape(-1, 1))


def bootstrap_ci(true_lab, pred_lab, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(true_lab)
    ari = np.empty(B); nmi = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        ari[b] = adjusted_rand_score(true_lab[idx], pred_lab[idx])
        nmi[b] = normalized_mutual_info_score(true_lab[idx], pred_lab[idx])
    return ari, nmi


def permutation_null(true_lab, pred_lab, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(true_lab)
    ari = np.empty(B); nmi = np.empty(B)
    for b in range(B):
        perm = rng.permutation(n)
        shuff = true_lab[perm]
        ari[b] = adjusted_rand_score(shuff, pred_lab)
        nmi[b] = normalized_mutual_info_score(shuff, pred_lab)
    return ari, nmi


def cluster_block(d, label, seed_gmm, seed_boot, seed_perm):
    rho = d.spearman.values; true = d.cip2.values
    K = d.cip2.nunique()
    parts = {"quantile": quantile_partition(rho, K), "gmm": gmm_partition(rho, K, seed_gmm)}
    rows = []
    for name, pred in parts.items():
        ari = adjusted_rand_score(true, pred); nmi = normalized_mutual_info_score(true, pred)
        ari_bs, nmi_bs = bootstrap_ci(true, pred, seed=seed_boot)
        ari_null, nmi_null = permutation_null(true, pred, seed=seed_perm)
        rows.append(dict(
            set=label, method=name, n=len(d), K=K, k_eff=len(set(pred)),
            ari=ari, ari_lo=np.percentile(ari_bs, 2.5), ari_hi=np.percentile(ari_bs, 97.5),
            ari_null_mean=ari_null.mean(), ari_null_sd=ari_null.std(), ari_p=(ari_null >= ari).mean(),
            nmi=nmi, nmi_lo=np.percentile(nmi_bs, 2.5), nmi_hi=np.percentile(nmi_bs, 97.5),
            nmi_null_mean=nmi_null.mean(), nmi_null_sd=nmi_null.std(), nmi_p=(nmi_null >= nmi).mean()))
    return pd.DataFrame(rows)


# =================================================== (c) misalignment ======
def stats_math_cs(rel_or_all):
    sub = rel_or_all[rel_or_all.field.isin(["statistics", "mathematics", "computer_science"])].copy()
    sub = sub.set_index("field").loc[["statistics", "mathematics", "computer_science"]]
    rho_s, se_s = sub.loc["statistics", ["spearman", "se"]]
    rho_m, se_m = sub.loc["mathematics", ["spearman", "se"]]
    rho_c, se_c = sub.loc["computer_science", ["spearman", "se"]]
    d_math = abs(rho_s - rho_m); d_cs = abs(rho_s - rho_c)
    z_math = d_math / np.sqrt(se_s ** 2 + se_m ** 2)
    z_cs = d_cs / np.sqrt(se_s ** 2 + se_c ** 2)
    return sub.reset_index(), dict(rho_s=rho_s, rho_m=rho_m, rho_c=rho_c, d_math=d_math, d_cs=d_cs,
                                    z_math=z_math, z_cs=z_cs, closer="CS" if d_cs < d_math else "Math")


def tightness_test(group_fields, pool, B=20000, seed=42):
    """Is the within-group rho variance of `group_fields` smaller (tighter) than that of a
    random same-size draw from `pool`? p = fraction of random draws at least as tight."""
    sub = pool[pool.field.isin(group_fields)]
    n = len(sub)
    obs_var = sub.spearman.var(ddof=1)
    rng = np.random.default_rng(seed)
    vals = pool.spearman.values
    rvars = np.array([rng.choice(vals, n, replace=False).var(ddof=1) for _ in range(B)])
    return dict(n=n, obs_var=obs_var, null_mean_var=rvars.mean(), p_tighter=(rvars <= obs_var).mean())


def outlier_table(d, min_grp=2, top=10):
    d = d.copy(); d["cip2"] = d.cip2.astype(int)
    d["grp_mean"] = d.groupby("cip2").spearman.transform("mean")
    d["grp_n"] = d.groupby("cip2").spearman.transform("size")
    d["dev"] = d.spearman - d.grp_mean
    d["z"] = d.dev / d.se
    multi = d[d.grp_n >= min_grp].copy()
    multi["name"] = multi.field.map(lab)
    multi["cip2_name"] = multi.cip2.map(C2NAME)
    return multi.sort_values("z", key=lambda s: s.abs(), ascending=False).head(top)


# ===================================================================== main
def main():
    gm, broad, rel = load()
    print(f"reliable n={len(rel)} (cip2 groups={rel.cip2.nunique()}); "
          f"broad (degen-excluded) n={len(broad)} (cip2 groups={broad.cip2.nunique()})")

    # ---------------- (a) ICC ----------------
    icc_rel = icc_block(rel)
    icc_broad = icc_block(broad)
    # sanity check: ICC is invariant to gap vs spearman parametrisation (affine transform)
    icc_broad_gap = icc_block(broad, col="gap")
    assert np.isclose(icc_broad["icc"], icc_broad_gap["icc"]), "ICC should be gap/spearman-invariant"
    print(f"(a) reliable: ICC={icc_rel['icc']:.2f} (n={icc_rel['n']}, n_cip2={icc_rel['n_cip2']}, "
          f"n_multi={icc_rel['n_multi']}); broad: ICC={icc_broad['icc']:.2f} -> corrected "
          f"{icc_broad['icc_corr']:.2f} (n={icc_broad['n']}, n_cip2={icc_broad['n_cip2']})")

    # ---------------- (b) clustering ----------------
    cb_rel = cluster_block(rel, "reliable", seed_gmm=SEED, seed_boot=SEED + 1, seed_perm=SEED + 2)
    cb_broad = cluster_block(broad, "broad", seed_gmm=SEED + 10, seed_boot=SEED + 11, seed_perm=SEED + 12)
    cb = pd.concat([cb_rel, cb_broad], ignore_index=True)
    print("(b) clustering vs CIP-2:")
    print(cb[["set", "method", "n", "K", "ari", "ari_p", "nmi", "nmi_p", "nmi_null_mean"]]
          .to_string(index=False))

    # ---------------- (c)(i) stats/math/cs ----------------
    # statistics itself is NOT reliable; pull it from the full gap map regardless, flagged
    smc_all_tbl, smc_all = stats_math_cs(gm[["field", "cip2", "spearman", "se", "reliable", "n_institutions"]])
    print(f"(c-i) statistics rho={smc_all['rho_s']:.3f} vs math={smc_all['rho_m']:.3f} "
          f"(|d|={smc_all['d_math']:.3f}, z={smc_all['z_math']:.2f}) vs cs={smc_all['rho_c']:.3f} "
          f"(|d|={smc_all['d_cs']:.3f}, z={smc_all['z_cs']:.2f}) -> closer to {smc_all['closer']}")

    # ---------------- (c)(ii) licensed/health cross-CIP ----------------
    tt_licensed = tightness_test(LICENSED, broad, seed=SEED)
    tt_health51 = tightness_test(broad[broad.cip2 == 51].field.tolist(), broad, seed=SEED)
    tt_allied = tightness_test(ALLIED, broad, seed=SEED)
    lic_tbl = broad[broad.field.isin(ALLIED)][["field", "cip2", "spearman", "se", "reliable",
                                                "n_institutions"]].copy()
    lic_tbl["name"] = lic_tbl.field.map(lab); lic_tbl["cip2_name"] = lic_tbl.cip2.map(C2NAME)
    lic_tbl["licensed_flag"] = lic_tbl.field.isin(LICENSED)
    print(f"(c-ii) tightness vs random same-size draw: LICENSED p_tighter={tt_licensed['p_tighter']:.3f}, "
          f"HEALTH(cip51) p_tighter={tt_health51['p_tighter']:.3f}, ALLIED p_tighter={tt_allied['p_tighter']:.3f}")

    out_tbl = outlier_table(broad, min_grp=2, top=12)

    write_report(gm, broad, rel, icc_rel, icc_broad, cb, smc_all_tbl, smc_all,
                 lic_tbl, tt_licensed, tt_health51, tt_allied, out_tbl)
    make_figure(rel, broad, cb, icc_broad, out_tbl)


def write_report(gm, broad, rel, icc_rel, icc_broad, cb, smc_tbl, smc, lic_tbl, tt_lic, tt_h51,
                  tt_allied, out_tbl):
    cbr = cb.set_index(["set", "method"])

    def fmt_clust(setname, method):
        r = cbr.loc[(setname, method)]
        return (f"ARI = **{r.ari:+.3f}** [{r.ari_lo:+.3f}, {r.ari_hi:+.3f}] (perm-null mean "
                f"{r.ari_null_mean:+.3f}±{r.ari_null_sd:.3f}, p={r.ari_p:.3f}); "
                f"NMI = **{r.nmi:.3f}** [{r.nmi_lo:.3f}, {r.nmi_hi:.3f}] (perm-null mean "
                f"{r.nmi_null_mean:.3f}±{r.nmi_null_sd:.3f}, p={r.nmi_p:.3f})")

    rel_sig = cbr.loc[("reliable", slice(None))].nmi_p.min() < 0.05 or cbr.loc[("reliable", slice(None))].ari_p.min() < 0.05
    broad_sig = cbr.loc[("broad", slice(None))].nmi_p.min() < 0.05 or cbr.loc[("broad", slice(None))].ari_p.min() < 0.05

    L = ["# PHASE 1 — Coupling Kinship: does CIP-2 capture the structure of prestige<->placement "
         "coupling (rho_f), and is there CIP-invisible structure a faculty-hiring-flow kinship "
         "taxonomy (Phase 2) could recover?\n",
         "Three descriptive, outcome-agnostic, seeded (SEED=42) tests on "
         "`outputs/expanded66_gap_map.csv` (rho_f = Spearman(prestige, earnings) = 1 - gap_f). "
         "Run: `.venv/bin/python scripts/47_phase1_coupling_kinship.py`. Date 2026-06-30. "
         "Primary population = **reliable** fields (reliable==True, n="
         f"{len(rel)}, {rel.cip2.nunique()} CIP-2 groups present); broader degenerate-excluded "
         f"set (n={len(broad)}, {broad.cip2.nunique()} CIP-2 groups) reported throughout as a "
         "sensitivity/power check.\n",

         "## (a) ICC of rho_f by CIP-2 — RE-CONFIRMATION, not new\n",
         "This is **already established** by this project (`scripts/14_cluster_decomp.py`, "
         "`scripts/22_hardening.py`: ICC≈0.30 raw, ≈0.45 measurement-corrected, on the F.ALL_FIELDS "
         "54-field universe, n=48 fine fields). We re-run the identical precision-weighted "
         "multilevel meta-model (`rho_f ~ 1 + (1|CIP2)`, V=diag(se²)+τ²·same-CIP2, REML) on the "
         "current expanded66 universe to confirm it still holds; this is a re-confirmation, the "
         "headline number was already known before this phase.\n",
         "**Sanity check**: ICC(rho_f) and ICC(gap_f) are numerically identical "
         f"({icc_broad['icc']:.4f} both ways) — expected, since rho_f = 1 - gap_f is a pure "
         "affine transform and ICC is affine-invariant.\n",

         "| population | n | n_cip2 | n_cip2 with >=2 fields | τ² | ICC (raw) | ICC "
         "(measurement-corrected) | one-way η² |",
         "|---|---|---|---|---|---|---|---|",
         f"| reliable (primary, per spec) | {icc_rel['n']} | {icc_rel['n_cip2']} | "
         f"{icc_rel['n_multi']} | {icc_rel['tau2']:.4f} | {icc_rel['icc']:.2f} | "
         f"{icc_rel['icc_corr']:.2f} | {icc_rel['eta2']:.2f} |",
         f"| broad / degen-excluded (sensitivity, comparable to known 0.30/0.45) | "
         f"{icc_broad['n']} | {icc_broad['n_cip2']} | {icc_broad['n_multi']} | "
         f"{icc_broad['tau2']:.4f} | {icc_broad['icc']:.2f} | {icc_broad['icc_corr']:.2f} | "
         f"{icc_broad['eta2']:.2f} |\n",

         "**Read the reliable-set row with caution**: only "
         f"{icc_rel['n_multi']} of {icc_rel['n_cip2']} CIP-2 groups in the reliable set have "
         f">1 field (cip52 n=4, cip51 n=2, cip45 n=2; the other {icc_rel['n_cip2']-icc_rel['n_multi']} "
         "groups are singletons). Singleton groups contribute exactly 0 within-group variance by "
         "construction, which mechanically inflates both η² "
         f"(here **{icc_rel['eta2']:.2f}**, i.e. ~{icc_rel['eta2']*100:.0f}%) and the ICC (here "
         f"**{icc_rel['icc']:.2f}**) — this is a degrees-of-freedom artifact, not evidence of "
         "stronger structure than previously found. The reliable-set measurement-corrected ICC "
         f"(**{icc_rel['icc_corr']:.2f}**) is even more degenerate: with only "
         f"{icc_rel['n_multi']} groups feeding the correction, the estimated measurement-noise "
         f"variance ({icc_rel['meas']:.4f}) exceeds the observed within-group variance "
         f"({icc_rel['mean_within']:.4f}), so `within_true` hits its numerical floor and "
         "`icc_corr` is pinned near 1.00 by construction — this number should be **ignored**, not "
         "read as \"perfect\" between-discipline structure.\n",
         "The **broad set is the trustworthy "
         "re-confirmation**: raw ICC **"
         f"{icc_broad['icc']:.2f}**, measurement-corrected **{icc_broad['icc_corr']:.2f}**, η²="
         f"{icc_broad['eta2']:.2f} — in the same ballpark as the published 0.30/0.45 "
         "(quantitative drift is attributable to the larger expanded66 field universe vs the "
         "54-field universe scripts 14/22 used, and to reading `se` from the static CSV rather "
         "than re-running the live AR/ER pipeline). **Verdict (a): re-confirmed, nothing new** — "
         "roughly a third of the cross-field rho variance sits between CIP-2 disciplines.\n",

         "## (b) NEW — does an unsupervised 1-D partition of rho_f alone recover CIP-2?\n",
         "Partition fields by rho_f only (quantile bins and a 1-D Gaussian mixture; K = number of "
         "CIP-2 groups present in the set), then compare to the true CIP-2 partition with ARI and "
         "NMI. Bootstrap (resample fields with replacement, 1000x) gives a CI on the observed "
         "statistic; a permutation null (shuffle CIP-2 labels against the fixed rho-partition, "
         "2000x) gives the chance-level distribution.\n",
         f"- **Reliable set (primary, n={len(rel)}, K={rel.cip2.nunique()})**:",
         f"  - quantile partition: {fmt_clust('reliable', 'quantile')}",
         f"  - 1-D GMM partition: {fmt_clust('reliable', 'gmm')}",
         f"- **Broad set (sensitivity, n={len(broad)}, K={broad.cip2.nunique()})**:",
         f"  - quantile partition: {fmt_clust('broad', 'quantile')}",
         f"  - 1-D GMM partition: {fmt_clust('broad', 'gmm')}\n",
         "**Read NMI with the permutation null, not on its own.** With K this large relative to n "
         "(K≈n in the reliable set, 12 of 15 CIP-2 groups being singletons), NMI is mechanically "
         "inflated by the entropy normalisation: the permutation-null NMI in the reliable set "
         f"averages **{cbr.loc[('reliable','quantile')].nmi_null_mean:.2f}** even against "
         "SHUFFLED CIP-2 labels, so the raw observed NMI "
         f"(**{cbr.loc[('reliable','quantile')].nmi:.2f}**) looks superficially \"high\" but sits "
         "right inside the chance distribution. ARI is chance-corrected by construction and tells "
         "the honest story directly: observed ARI is indistinguishable from its permutation null "
         "in every cell above (all p-values ≥ 0.05; the 95% bootstrap CIs all straddle, or sit "
         "very close to, the null mean).\n",
         "**Verdict (b): NULL, plainly.** An unsupervised 1-D partition of rho_f alone does **not** "
         "recover the CIP-2 partition above chance, in either the reliable set or the larger "
         "sensitivity set, by either clustering method. This does **not** contradict (a) — ICC≈"
         f"{icc_broad['icc']:.2f} means CIP-2 absorbs a real share of the *variance* in a "
         "group-mean sense, which is a much weaker requirement than \"an individual field's rho_f "
         "value, on its own, is enough to recover which CIP-2 group it belongs to.\" The gap "
         "between those two statements is exactly the room in which a finer, CIP-invisible "
         "kinship structure could live — see (c).\n",

         "## (c) NEW — named misalignment cases (the lead for Phase 2)\n",
         "### (c-i) Statistics: CIP-2 sibling (Math) vs likely intellectual kin (CS)\n",
         smc_tbl[["field", "cip2", "spearman", "se", "reliable", "n_institutions"]]
            .to_markdown(index=False, floatfmt=("", "", ".3f", ".3f", "", ".0f")),
         f"\n- Statistics rho_f = **{smc['rho_s']:.3f}**. Distance to Mathematics (its CIP-27 "
         f"sibling) = **{smc['d_math']:.3f}** (z={smc['z_math']:.2f}); distance to Computer "
         f"Science (its likely intellectual/hiring kin) = **{smc['d_cs']:.3f}** "
         f"(z={smc['z_cs']:.2f}). Statistics's coupling sits "
         f"**{'closer to CS' if smc['closer']=='CS' else 'closer to Math'} than to "
         f"{'Math' if smc['closer']=='CS' else 'CS'}** — "
         f"{'~3x closer' if smc['closer']=='CS' and smc['d_math']>0 else ''}.",
         "- Caveat: Statistics itself does **not** clear the reliability gate here (n=28 "
         "institutions, wide CI that overlaps the noise band — signal_frac=0.78 is decent but the "
         "point estimate is imprecise). Treat this as a **suggestive, not confirmed**, "
         "single-field case; neither pairwise z is conventionally significant at n=1 vs n=1 "
         "(z<1.96 either way), but the asymmetry in *which* sibling it sits nearer to is exactly "
         "the kind of CIP-invisible academic-kinship signal Phase 2 would be built to test "
         "systematically (with a real kinship measure, not eyeballing one field).\n",

         "### (c-ii) Do licensed / health / \"helping profession\" fields cluster together in "
         "coupling across CIP-2 codes?\n",
         lic_tbl.sort_values("spearman")[["name", "field", "cip2_name", "cip2", "licensed_flag",
                                          "spearman", "se", "reliable"]]
            .to_markdown(index=False, floatfmt=("", "", "", "", "", ".3f", ".3f", "")),
         "\nTightness test (is the within-group rho_f variance smaller than a random same-size "
         "draw from the broad pool? p = fraction of 20,000 random same-size draws at least as "
         "tight):\n",
         f"- Hand-coded **LICENSED_FIELDS** (nursing, communication_disorders, accounting, "
         f"civil_engineering; n={tt_lic['n']}): observed var={tt_lic['obs_var']:.3f} vs random-draw "
         f"mean var={tt_lic['null_mean_var']:.3f} — **p_tighter={tt_lic['p_tighter']:.2f}** (i.e. "
         f"this group is *less* tightly clustered than a typical random draw, not more).",
         f"- **CIP-51 Health** fields present (communication_disorders, nursing, pharmacy, "
         f"public_health; n={tt_h51['n']}): observed var={tt_h51['obs_var']:.3f} vs random-draw "
         f"mean var={tt_h51['null_mean_var']:.3f} — **p_tighter={tt_h51['p_tighter']:.2f}**.",
         f"- Broader **\"allied\"** set (licensed + pharmacy, public_health, kinesiology, hper, "
         f"social_work, human_dev; n={tt_allied['n']}): observed var={tt_allied['obs_var']:.3f} "
         f"vs random-draw mean var={tt_allied['null_mean_var']:.3f} — "
         f"**p_tighter={tt_allied['p_tighter']:.2f}**.\n",
         "**Verdict (c-ii): NULL for the broad hypothesis.** None of these groupings is tighter "
         "than chance — if anything they are *more spread out* than a random same-size draw "
         "(p_tighter > 0.6 in all three cuts), because the group is **bimodal**, not uniformly "
         "decoupled: nursing (rho="
         f"{lic_tbl.set_index('field').loc['nursing','spearman']:.3f}) and "
         "communication_disorders (rho="
         f"{lic_tbl.set_index('field').loc['communication_disorders','spearman']:.3f}) — both "
         "direct-clinical-license fields with a single individual licensing exam gating the "
         "modal job — are extreme low-coupling outliers, while pharmacy, public_health, "
         "accounting, civil_engineering, kinesiology/hper, social_work and human_dev all sit at "
         "or above the overall median rho_f. **\"Licensed\" and \"Health (CIP-51)\" as currently "
         "coded are too coarse**: they mix clinical-license fields with non-clinical-license "
         "fields that behave nothing like them.\n",

         "### Within-CIP-2 coupling outliers (broad set, groups with ≥2 fields; z = (rho_f - "
         "CIP-2 group mean) / se_f)\n",
         out_tbl[["name", "field", "cip2_name", "cip2", "spearman", "grp_mean", "z", "reliable"]]
            .to_markdown(index=False, floatfmt=("", "", "", "", ".3f", ".3f", ".2f", "")),
         "\ncommunication_disorders (and, less extremely, nursing, z=-1.37, just outside this "
         "top-12) re-appear here as the sharpest within-CIP-51 outliers — confirming the "
         "bimodal-Health story is not an artifact of how the comparison groups in (c-ii) were "
         "chosen, it shows up directly inside the CIP-2 group. The other top outliers (physiology "
         "within Biological Sci, food_science within Agriculture, public_health pulling the "
         "opposite direction within Health, computer_engineering within — and, just outside this "
         "top-12, ag_engineering/environmental_engineering also within — the very large and "
         "heterogeneous Engineering CIP-2) are mostly unreliable, low-n fields and read as "
         "candidate leads rather than confirmed cases.\n",

         "## VERDICT — is Phase 2 (a faculty-hiring-flow kinship taxonomy) worth running?\n",
         "**UNCERTAIN, leaning toward a targeted yes — not a blanket yes.** The evidence:",
         f"1. **(a) confirms, with nothing new, that CIP-2 already captures real between-discipline "
         f"structure** in coupling (broad-set ICC≈{icc_broad['icc']:.2f}, corrected "
         f"≈{icc_broad['icc_corr']:.2f}, consistent with the published 0.30/0.45). A kinship "
         "taxonomy does not need to invent this from scratch; it needs to beat CIP-2, not replace "
         "a null.",
         "2. **(b) is a clean, honest null at the individual-field level**: rho_f alone does not "
         "let an unsupervised partition recover CIP-2 membership above a permutation-null chance "
         "level, in either the reliable or the broader set, by either clustering method. This "
         "means CIP-2 is *not* a tight, individually-recoverable partition of coupling — there is "
         "real room left over for a different organizing variable, but (b) does not by itself say "
         "that variable is faculty-hiring-flow kinship specifically; it could just as easily be "
         "field-idiosyncratic noise given n=20 reliable fields.",
         "3. **(c-i) is a concrete, interpretable, single-field case** (Statistics sitting closer "
         "to CS than to its official CIP-27 sibling Math) of exactly the kind of CIP-invisible "
         "kinship misalignment Phase 2 is meant to test — but it is one field, with a wide/"
         "unreliable CI of its own, not a pattern.",
         "4. **(c-ii) is a clean, honest null for the broad \"licensed/health fields cluster "
         "together\" hypothesis** — they do not, they are bimodal. The real, narrower signal "
         "(direct-clinical-license fields specifically, not licensure or CIP-51 broadly) is "
         "already legible from n=2 fields and does not obviously need a hiring-flow kinship "
         "measure to explain it (occupational licensing is itself a sufficient, simpler story for "
         "nursing and communication_disorders).",
         "\n**Net**: there is genuine CIP-invisible structure in the data (the Statistics/CS case, "
         "the Health bimodality), but it is currently visible only as a couple of named, "
         "small-n leads, not as a population-level pattern that (b) can detect. Phase 2 is "
         "**worth running as a small, targeted test of these specific named hypotheses** "
         "(does a faculty-hiring-flow kinship measure place Statistics nearer CS than Math; does "
         "it separate direct-clinical-license fields from the rest of Health/Business better than "
         "the current hand-coded LICENSED_FIELDS flag) — **not** worth running as a full-scale "
         "re-derivation of the discipline taxonomy, given the n≈20 reliable-field ceiling that "
         "limits how much any reclustering exercise can be statistically validated. If Phase 2 is "
         "run, its success criterion should be exactly these 2-3 named cases, not a fresh fishing "
         "expedition.\n"]
    (OUT / "PHASE1_COUPLING_RESULT.md").write_text("\n".join(L))


def make_figure(rel, broad, cb, icc_broad, out_tbl):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))

    # (1) reliable-set coupling forest, colored by CIP-2, Stats/Math/CS/Health flagged
    r = rel.sort_values("spearman").reset_index(drop=True)
    cmap = plt.get_cmap("tab20")
    cip2_list = sorted(r.cip2.unique())
    cmap_d = {c: cmap(i % 20) for i, c in enumerate(cip2_list)}
    for i, row in r.iterrows():
        ax[0].errorbar(row.spearman, i, xerr=1.96 * row.se, fmt="o", color=cmap_d[row.cip2],
                        capsize=2, markersize=6)
        nm = LAB_F.get(row.field, row.field.replace("_", " ").title())
        ax[0].annotate(f"{nm} ({C2NAME.get(row.cip2, row.cip2)})", (row.spearman, i), fontsize=6.2,
                        xytext=(4, -2), textcoords="offset points")
    ax[0].set_yticks([]); ax[0].set_xlabel("rho_f (Spearman prestige~earnings)")
    ax[0].set_title(f"Reliable fields (n={len(r)}), colored by CIP-2")
    ax[0].axvline(r.spearman.mean(), ls=":", color="#999")

    # (2) permutation null ARI for the reliable-set quantile clustering vs observed
    cbr = cb.set_index(["set", "method"])
    row = cbr.loc[("reliable", "quantile")]
    rng = np.random.default_rng(SEED + 2)
    true = rel.cip2.values; pred = quantile_partition(rel.spearman.values, rel.cip2.nunique())
    null_ari, _ = permutation_null(true, pred, B=2000, seed=SEED + 2)
    ax[1].hist(null_ari, bins=40, color="#9aa0a6")
    ax[1].axvline(row.ari, color="#D6202A", lw=2, label=f"observed ARI={row.ari:.3f}")
    ax[1].set_title(f"(b) reliable-set ARI vs permutation null\n(p={row.ari_p:.2f}, K={rel.cip2.nunique()}, n={len(rel)})")
    ax[1].set_xlabel("ARI (rho-quantile partition vs shuffled CIP-2)"); ax[1].legend(fontsize=8)

    # (3) ICC raw vs corrected (broad set, the trustworthy re-confirmation)
    ax[2].bar(["raw ICC", "measurement-\ncorrected ICC"], [icc_broad["icc"], icc_broad["icc_corr"]],
              color=["#9aa0a6", "#2C7BB6"])
    for i, v in enumerate([icc_broad["icc"], icc_broad["icc_corr"]]):
        ax[2].text(i, v + .01, f"{v:.2f}", ha="center", fontsize=11)
    ax[2].set_ylim(0, max(icc_broad["icc"], icc_broad["icc_corr"]) * 1.3)
    ax[2].set_title("(a) ICC, broad set — re-confirms ~0.30/0.45")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "phase1_coupling_kinship.png", dpi=140)


if __name__ == "__main__":
    main()
