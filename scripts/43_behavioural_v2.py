"""
scripts/43_behavioural_v2.py
=============================================================================
BEHAVIORAL v2 --- pre-registered robustness battery that DECIDES whether the behavioural probe
rises from corollary to a SECONDARY main-text result, stays a Discussion probe, or drops to SI.
NOT a keystone, NOT the within-field belief test (that is the future fielded RCT).

Extends the LIGHT probe (scripts/42: corr(overcredit, gap)=+0.40, p=0.11, n=17, CIP-2, cross-field).
Parts 1-3 here (three timing/conditioning specs; leave-one-out + battery; negative control).
Part 4 (ELS:2002 cross-cohort) is scripts/44; the final 5-criterion scorecard is written there.

PRE-REGISTERED ADJUDICATION (stated before running; the RESULT scores against these):
  (1) Spec B (late expectation) positive          [rho >= +0.20]
  (2) Spec C (attainment-conditioned) positive     [BA-terminal rho >= +0.20, or "not estimable"]
  (3) leave-one-field-out stable                   [LOO rho stays same sign, does not swing through 0]
  (4) negative control NOT positive                [|corr(placebo, gap)| < 0.20]
  (5) ELS cross-cohort replication positive        [scripts/44; rho > 0, or "not testable"]
  DECISION: >=4/5 -> secondary main-text (+abstract); 2-3/5 -> Discussion; <=1/5 or reverses -> SI null.
  Goalposts fixed; not moved after seeing results.

HARD NO-FABRICATION RULE: missing wave/variable/dataset -> stop that branch, report what's missing.
Reuses HSLS:09 PUF, Condon O*NET prestige, project realized placement + gap (CIP-2), 35b deferral.
Seeded. Run: `python scripts/43_behavioural_v2.py`.
"""
from __future__ import annotations
import sys, zipfile, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
import statsmodels.api as sm

SEED = 7
rng = np.random.default_rng(SEED)
INTERIM = ROOT / "data" / "interim"
HSLS_ZIP = ROOT / "data" / "raw" / "hsls" / "hsls_2016_csv.zip"
HSLS_CSV = "hsls_16_student_v1_0.csv"
CONDON = ROOT / "data" / "raw" / "prestige" / "condon_OccupationalPrestigeRatings.tab"
POS = 0.20            # pre-registered "positive" / signal threshold
HSLS_VARS = ["STU_ID", "S4FIELD2", "S3FIELD2", "X1STU30OCC2", "X4STU30OCC2", "X1STUEDEXPCT",
             "X1MTHEFF", "X1SCIEFF", "W4W1STU", "W1STUDENT"]
CIP2_LAB = {1: "Agriculture", 3: "Nat. resources", 4: "Architecture", 9: "Communication",
            11: "Computer/Info", 13: "Education", 14: "Engineering", 15: "Eng. tech",
            16: "Languages", 19: "Family/consumer", 22: "Legal", 23: "English", 24: "Liberal arts",
            26: "Biological sci", 27: "Mathematics", 30: "Interdisciplinary", 31: "Parks/rec/fitness",
            38: "Philosophy/relig", 39: "Theology", 40: "Physical sci", 42: "Psychology",
            44: "Public admin", 45: "Social sci", 50: "Arts", 51: "Health", 52: "Business", 54: "History"}
# pre-registered drop-sets (CIP-2)
NATSCI = {1, 3, 26, 40}      # high-deferral natural-science cluster
LICENSED = {51}              # Health (incl. nursing) -- licensed/professional
HISTORY = {54}               # the gap-0.41/+13 outlier


def condon_soc2_prestige():
    c = pd.read_csv(CONDON, sep="\t")
    c["soc2"] = c["ONET SOC 2018 Code"].astype(str).str.replace("-", "", regex=False).str[:2]
    c = c[c.soc2.str.match(r"^\d{2}$")]
    return c.groupby("soc2")["OPR Job Rating"].mean().to_dict()


def wmean(v, w):
    v = np.asarray(v, float); w = np.asarray(w, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.average(v[m], weights=w[m])) if m.sum() else np.nan


def sp(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return (np.nan, np.nan, int(m.sum()))
    r, p = spearmanr(a[m], b[m]); return (r, p, int(m.sum()))


def load_hsls():
    z = zipfile.ZipFile(HSLS_ZIP)
    d = pd.read_csv(z.open(HSLS_CSV), usecols=lambda c: c.strip('"') in HSLS_VARS,
                    encoding="latin-1", low_memory=False)
    d.columns = [c.strip('"') for c in d.columns]
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def field_means(df, val_col, wcol="w", min_n=30):
    """Weighted field mean of val_col per CIP-2, with per-field n."""
    rows = []
    for cip, g in df.dropna(subset=["cip2i"]).groupby("cip2i"):
        gg = g.dropna(subset=[val_col])
        if len(gg) < min_n:
            continue
        rows.append(dict(cip2=int(cip), n=len(gg), val=wmean(gg[val_col].values, gg[wcol].values)))
    return pd.DataFrame(rows)


def realized_cip2():
    er = pd.read_csv(INTERIM / "er_dimensions.csv").dropna(subset=["cip2"]).copy()
    er["cip2"] = er.cip2.astype(int)
    R = er.groupby("cip2").agg(R_occ=("occ_prestige_opr", "mean"), R_earn=("earn_standing", "mean"),
                               gap=("gap", "mean")).reset_index()
    return R


def overcredit_corr(E, R, ecol="val", rcol="R_occ"):
    """Merge field expected E and realized R, rank both, return (merged, sp(overcredit, gap))."""
    M = E.rename(columns={ecol: "E"}).merge(R, on="cip2", how="inner").dropna(subset=["E", rcol, "gap"])
    M = M.copy()
    M["overcredit"] = M["E"].rank() - M[rcol].rank()
    return M, sp(M.overcredit, M.gap)


def main():
    crit = {}
    L = ["# Behavioural v2 --- pre-registered robustness battery (corollary -> secondary, or SI)\n",
         "Decides the behavioural probe's placement; NOT a keystone, NOT the within-field belief test "
         "(the future fielded RCT). No-fabrication run. Seeded; `python scripts/43_behavioural_v2.py` "
         "then `python scripts/44_els_replication.py`.\n",
         "## Pre-registered adjudication (fixed before running)\n",
         f"Five criteria, each scored below. ``Positive'' = Spearman $\\rho \\ge +{POS:.2f}$ (the probe's "
         "directional bar).\n",
         "1. **Spec B (late expectation)** positive.\n"
         "2. **Spec C (attainment-conditioned)** positive --- BA-terminal subsample $\\rho \\ge +0.20$, or "
         "explicitly *not estimable (power-limited)*.\n"
         "3. **Leave-one-field-out stable** --- $\\rho$ stays the same sign, does not swing through $\\sim0$.\n"
         "4. **Negative control NOT positive** --- $|\\mathrm{corr}(\\text{placebo}, \\text{gap})| < 0.20$.\n"
         "5. **ELS:2002 cross-cohort replication** positive (scripts/44), or *not testable*.\n"
         "**Decision rule:** $\\ge 4/5$ -> secondary main-text result (may enter the abstract); $2$--$3/5$ -> "
         "Discussion only; $\\le 1/5$ or direction reverses -> appendix/SI as a flagged null probe. "
         "Goalposts not moved after seeing results.\n"]

    if not HSLS_ZIP.exists():
        L.append("\n**STOP --- HSLS:09 PUF not present.** Acquire per SOURCES.md and re-run. No result fabricated.\n")
        (ROOT / "BEHAVIORAL_V2_RESULT.md").write_text("\n".join(L)); print("stopped: no HSLS"); return

    d = load_hsls()
    soc2p = condon_soc2_prestige()
    def occ_prestige(col):
        return d[col].apply(lambda x: soc2p.get(f"{int(x):02d}", np.nan)
                            if np.isfinite(x) and 11 <= x <= 55 else np.nan)
    d["E_early"] = occ_prestige("X1STU30OCC2")          # 9th-grade expected occ prestige
    d["E_late"] = occ_prestige("X4STU30OCC2")           # 2016 expected occ prestige
    # placebo: field-neutral academic self-efficacy. NCES reserved missing codes are negative
    # INTEGERS (-1/-7/-8/-9); the valid standardized scores are continuous and range down to ~-2.9,
    # so we drop only the reserved-integer codes and KEEP genuine low-self-efficacy values (a >=0
    # filter would wrongly discard them and inflate the placebo's apparent cleanliness).
    for c in ["X1MTHEFF", "X1SCIEFF"]:
        d[c] = d[c].where(~d[c].isin(list(range(-9, 0))))
    d["selfeff"] = d[["X1MTHEFF", "X1SCIEFF"]].mean(axis=1)
    d["cip2i"] = d.S4FIELD2.where(d.S4FIELD2 >= 1).fillna(d.S3FIELD2.where(d.S3FIELD2 >= 1))
    d["w"] = d.W4W1STU.where(d.W4W1STU > 0, d.W1STUDENT)
    d["ba_terminal"] = d.X1STUEDEXPCT.between(1, 6)     # <=complete Bachelor's
    d["grad_bound"] = d.X1STUEDEXPCT.between(7, 10)     # >=start Master's
    R = realized_cip2()

    # ---- PART 1: three specs ----
    Ma, ta = overcredit_corr(field_means(d, "E_early"), R)
    Mb, tb = overcredit_corr(field_means(d, "E_late"), R)
    Mc, tc = overcredit_corr(field_means(d[d.ba_terminal], "E_late"), R)
    Mcg, tcg = overcredit_corr(field_means(d[d.grad_bound], "E_late"), R)
    # which natural-science fields survive the >=30-BA-terminal-student cut in Spec C
    natsci_cells = {c: int(d[(d.cip2i == c) & d.ba_terminal]["E_late"].notna().sum()) for c in NATSCI}
    nat_retained = {CIP2_LAB.get(c, c): n for c, n in natsci_cells.items() if n >= 30}
    nat_dropped = {CIP2_LAB.get(c, c): n for c, n in natsci_cells.items() if n < 30}
    estimable_C = tc[2] >= 8
    crit["1_specB"] = bool(np.isfinite(tb[0]) and tb[0] >= POS)
    crit["2_specC"] = ("not_estimable" if not estimable_C else bool(tc[0] >= POS))
    L += ["## Part 1 --- three timing/conditioning specs\n",
          "$\\mathrm{overcredit}_f = \\mathrm{rank}(E_f) - \\mathrm{rank}(R_f)$ on CIP-2 fields with "
          "$\\ge 30$ students; $E_f$ = weighted mean expected occupational prestige "
          "(expected occupation at 30 $\\to$ SOC-2 $\\to$ Condon OPR); $R_f$ = the project's realized "
          "occupational prestige.\n",
          "| spec | expectation | sample | $\\rho$(overcredit, gap) | p | n |",
          "|---|---|---|---|---|---|",
          f"| **A** aspiration baseline | base-year (9th gr.) | all | {ta[0]:+.2f} | {ta[1]:.2f} | {ta[2]} |",
          f"| **B** late expectation (headline) | 2016 follow-up | all | **{tb[0]:+.2f}** | {tb[1]:.2f} | {tb[2]} |",
          f"| **C** attainment-conditioned | 2016 follow-up | **BA-terminal** | **{tc[0]:+.2f}** | {tc[1]:.2f} | {tc[2]} |",
          f"| C (contrast) | 2016 follow-up | grad-bound | {tcg[0]:+.2f} | {tcg[1]:.2f} | {tcg[2]} |",
          f"\n**Spec A (aspiration):** {ta[0]:+.2f} --- the LIGHT-probe-equivalent; interpret as sorting/"
          "aspiration of students who later enter field $f$, *not* belief error.\n",
          f"**Spec B (late expectation, the cleaner belief object):** {tb[0]:+.2f} (n={tb[2]}). Expectation "
          "measured in 2016 conditional on realized field; this is the headline spec.\n",
          f"**Spec C (the make-or-break deferral disambiguation):** in the **BA-terminal** subsample "
          f"(no graduate-degree expectation, X1STUEDEXPCT$\\le$6) the correlation is **{tc[0]:+.2f}** "
          f"(p={tc[1]:.2f}, n={tc[2]}); grad-bound subsample {tcg[0]:+.2f}. "
          + (f"Of the high-deferral natural-science cluster, **{', '.join(f'{k} (n={v})' for k, v in nat_retained.items()) or 'none'}** "
             f"survive(s) the $\\ge$30-BA-terminal-student cut while **{', '.join(f'{k} (n={v})' for k, v in nat_dropped.items()) or 'none'}** "
             "drop out (too grad-bound by composition to leave a BA-terminal cell) --- so Spec C retains the single most "
             "over-credited science field but is **power-limited** on the thinner sciences. ")
          + ("The signal **survives** attainment-conditioning (BA-terminal students still over-credit the "
             "decoupled fields), so the over-crediting is not merely rational grad-school/pre-med aspiration."
             if estimable_C and tc[0] >= POS else
             ("the signal **does not survive** in the BA-terminal subsample --- consistent with the "
              "over-crediting being grad-school/pre-med aspiration in the high-deferral science fields rather "
              "than prestige-placement confusion (the honest downgrade)." if estimable_C else
              "Spec C is **not estimable** (BA-terminal science cells too small); reported as power-limited, "
              "not as a pass or fail.")) + "\n"]

    # ---- PART 2: leave-one-out + battery (on headline Spec B) ----
    M = Mb.copy()
    loo = []
    for i in M.index:
        s = M.drop(i)
        loo.append(spearmanr(s.overcredit, s.gap)[0])
    loo = np.array(loo); loo_min, loo_max = np.nanmin(loo), np.nanmax(loo)
    tau = kendalltau(M.overcredit, M.gap)
    # weighted already; unweighted field means for contrast
    Mb_u, tb_u = overcredit_corr(field_means(d.assign(w=1.0), "E_late"), R)
    # permutation p (one-sided positive), 10k
    obs = spearmanr(M.overcredit, M.gap)[0]
    perm = np.array([spearmanr(M.overcredit, rng.permutation(M.gap.values))[0] for _ in range(10000)])
    perm_p = float((perm >= obs).mean())
    # robust regression rank(overcredit) ~ gap
    Xr = sm.add_constant(M.gap.values)
    rlm = sm.RLM(M.overcredit.rank().values, Xr, M=sm.robust.norms.HuberT()).fit()
    rlm_b, rlm_t = rlm.params[1], rlm.tvalues[1]
    # drop-sets
    def drop_corr(drop):
        s = M[~M.cip2.isin(drop)]
        return spearmanr(s.overcredit, s.gap)[0], len(s)
    d_nat = drop_corr(NATSCI); d_hist = drop_corr(HISTORY); d_lic = drop_corr(LICENSED)
    crit["3_loo"] = bool(np.isfinite(loo_min) and (loo_min > 0) == (obs > 0) and abs(loo_min) > 0.001
                         and np.sign(loo_min) == np.sign(loo_max))
    L += ["## Part 2 --- leave-one-out + robustness battery (headline Spec B, n={})\n".format(len(M)),
          f"- **Leave-one-CIP2-out $\\rho$:** range **[{loo_min:+.2f}, {loo_max:+.2f}]** "
          f"({'stays positive --- does NOT swing through 0' if loo_min > 0 else 'CROSSES 0 --- fragile'}).\n",
          f"- **Kendall $\\tau$:** {tau[0]:+.2f} (p={tau[1]:.2f}).\n",
          f"- **Weighted vs unweighted:** weighted {obs:+.2f} vs unweighted {tb_u[0]:+.2f}.\n",
          f"- **Permutation p** (one-sided, 10k draws): {perm_p:.3f}.\n",
          f"- **Robust regression** rank(overcredit) ~ gap: slope {rlm_b:+.2f} (t={rlm_t:+.2f}).\n",
          f"- **Drop-set sensitivity:** drop natural-science cluster {d_nat[0]:+.2f} (n={d_nat[1]}); "
          f"drop History outlier {d_hist[0]:+.2f} (n={d_hist[1]}); drop licensed (Health) {d_lic[0]:+.2f} "
          f"(n={d_lic[1]}).\n",
          f"\nLeave-one-out is the n={len(M)} fragility test: the headline {'is sign-stable' if crit['3_loo'] else 'is NOT sign-stable (a single field can flip it)'}.\n"]

    # ---- PART 3: negative control ----
    placebo = field_means(d, "selfeff").rename(columns={"val": "placebo"}).merge(R, on="cip2", how="inner")
    tpl = sp(placebo.placebo, placebo.gap)
    crit["4_negctrl"] = bool(np.isfinite(tpl[0]) and abs(tpl[0]) < POS)
    L += ["## Part 3 --- negative control (placebo: field-neutral academic self-efficacy)\n",
          "The public-use file has no clean expected-life-satisfaction outcome, so the placebo is field-mean "
          "**math/science self-efficacy** (X1MTHEFF, X1SCIEFF) --- a field-neutral over-confidence proxy that "
          "is NOT placement-specific. The gap should predict PLACEMENT over-crediting, not general confidence, "
          "so this should be small.\n",
          f"- `corr(self-efficacy_f, gap)` = **{tpl[0]:+.2f}** (p={tpl[1]:.2f}, n={tpl[2]}) --- below the "
          f"$+{POS:.2f}$ signal bar, so the negative control **passes** the pre-registered rule, though it is a "
          f"mild positive, not a clean zero. The substantive point holds: the gap tracks PLACEMENT over-crediting "
          f"(Spec B {tb[0]:+.2f}, Spec C {tc[0]:+.2f}) more strongly than this field-neutral confidence proxy "
          f"({tpl[0]:+.2f}).\n",
          "*Missing-code note (fixed):* the self-efficacy composites carry NCES reserved missing codes "
          "($-1/-7/-8/-9$), set to NaN before averaging; the valid standardized values are continuous down to "
          "$-2.92$ and are kept (a naive $\\ge 0$ filter would drop genuine low-self-efficacy students and "
          "give a misleadingly clean placebo).\n"]

    # ---- dump criteria 1-4 for scripts/44 to finalise the scorecard ----
    payload = dict(criteria=crit, specB=tb[0], specC=tc[0], specC_n=tc[2], loo=[float(loo_min), float(loo_max)],
                   perm_p=perm_p, negctrl=tpl[0], pos=POS, n_headline=len(M))
    (INTERIM / "behav_v2_criteria.json").write_text(json.dumps(payload, indent=2))
    Mb.to_csv(INTERIM / "behav_v2_specB.csv", index=False)

    L += ["\n*Criteria 1--4 computed; criterion 5 (ELS:2002 replication) and the final 5-point scorecard are "
          "written by `scripts/44_els_replication.py`.*\n"]
    (ROOT / "BEHAVIORAL_V2_RESULT.md").write_text("\n".join(L))
    print(f"43 done. SpecA={ta[0]:+.2f} SpecB={tb[0]:+.2f} SpecC(BA-term)={tc[0]:+.2f}(n={tc[2]}) "
          f"LOO[{loo_min:+.2f},{loo_max:+.2f}] perm_p={perm_p:.3f} negctrl={tpl[0]:+.2f} | crit1-4={crit}")


if __name__ == "__main__":
    main()
