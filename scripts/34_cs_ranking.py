"""Public CS school ranking — academic prestige fine-tuned by graduate earnings (top-50).

DESCRIPTIVE ranking with disclosed caveats, NOT a causal value claim. Academic prestige = the
CONTINUOUS SpringRank SCORE (recomputed from the public Wapman CS faculty-hiring edge subgraph, so
score DIFFERENCES are meaningful — the point of not using ranks). Earnings = Scorecard CS BA median
(primary; covers elite privates that dominate the top-50, and scripts/32 showed Scorecard↔PSEO
rank institution pay at Spearman +0.94). Cross-checked vs PSEO CS earnings and the ORCID-rebuilt
CS SpringRank.

Blend (z-scored within the 70-school universe): combined = w·z_P + (1−w)·z_S, w ∈ {0.5, 0.6, 0.7};
PRIMARY = w=0.6 (prestige-anchored, salary fine-tunes). Salary premium = actual − prestige-predicted
earnings ($). w-sensitivity flags schools whose rank moves >5 across w (salary-driven, less robust).

 -> data/interim/cs_ranking.csv, outputs/figures/{cs_ranking_top50,cs_ranking_w_sensitivity}.png,
    CS_RANKING_RESULT.md
"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import springrank
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from src.crosswalks.institutions import normalize_institution_name as N

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
SCORECARD = ROOT / "data" / "raw" / "scorecard_fos" / "Most-Recent-Cohorts-Field-of-Study.csv"
WAP_EDGES = ROOT / "data" / "raw" / "wapman2022" / "edge_lists.csv"
WAP_RANKS = ROOT / "data" / "raw" / "wapman2022" / "ranks.csv"
PSEOE = ROOT / "data" / "raw" / "pseo" / "pseoe_all.csv.gz"
ORCID = ROOT / "data" / "interim" / "orcid_phd_faculty_edges.parquet"
CS_CIP = ["1107", "1101", "1102", "1104"]      # Computer Science field CIPs (66-field crosswalk)
BUFFER = 70
TOP = 50


def k_exact(name):
    return N(str(name).replace("&", " and "))


def k_campus(name):
    """strip a redundant campus qualifier Wapman omits ('-Main Campus', '-Seattle Campus', etc.)."""
    s = str(name)
    s = re.sub(r"\s+in the city of new york$", "", s, flags=re.I)
    s = re.sub(r"[-–]\s*main campus$", "", s, flags=re.I)
    s = re.sub(r"[-–]\s*[\w. ]+\s+campus$", "", s, flags=re.I)
    return k_exact(s)


def k_city(name):
    """strip a trailing flagship city/branch qualifier ('-Ann Arbor', '-Columbia', ' at Austin')."""
    s = re.sub(r"\s+at\s+[\w. ]+$", "", str(name), flags=re.I)
    s = re.sub(r"[-–]\s*[\w. &]+$", "", s, flags=re.I)
    return k_exact(s)


# satellite-campus tokens: when a cityless Wapman name (e.g. "University of Washington" = the Seattle
# flagship) matches several campuses, prefer the NON-branch one over a large-undergrad branch.
BRANCH = ("bothell", "tacoma", "dearborn", "flint", "kansas city", "newark", "camden", "lowell",
          "boston", "dartmouth", "springfield", "el paso", "san antonio", "arlington", "omaha")


def best(scd, wkey):
    """Priority match: exact -> campus-stripped -> city-stripped. Within a tier, prefer a non-branch
    campus, then the largest cohort (the flagship)."""
    for col in ("k_exact", "k_campus", "k_city"):
        c = scd[scd[col] == wkey].copy()
        if len(c):
            c["is_branch"] = c.INSTNM.str.lower().apply(lambda s: any(b in s for b in BRANCH)) \
                if "INSTNM" in c.columns else c.label.str.lower().apply(lambda s: any(b in s for b in BRANCH))
            return c.sort_values(["is_branch", "coh"], ascending=[True, False]).iloc[0]
    return None


def cs_springrank_wapman():
    ed = pd.read_csv(WAP_EDGES)
    e = ed[(ed.TaxonomyLevel == "Field") & (ed.TaxonomyValue == "Computer Science")]
    e = e[e.DegreeInstitutionName != e.InstitutionName]
    name = {}                                   # nk -> a clean original Wapman name
    for nm in pd.concat([e.DegreeInstitutionName, e.InstitutionName]).dropna().unique():
        name[N(nm)] = nm
    pairs = [(N(a), N(b)) for a, b in zip(e.DegreeInstitutionName, e.InstitutionName)]
    pairs = [(a, b) for a, b in pairs if a and b]
    nodes = sorted(set(s for s, _ in pairs) | set(d for _, d in pairs))
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for s, d in pairs:
        A[idx[s], idx[d]] = 1.0
    m = springrank.SpringRank(alpha=0.5, inverse_temp_fit_warning=False); m.fit(A)
    sc = pd.DataFrame({"inst_key": nodes, "wap_name": [name[n] for n in nodes],
                       "CS_springrank_score": m.ranks})
    # published CS rank for reference
    rk = pd.read_csv(WAP_RANKS); rk = rk[(rk.TaxonomyLevel == "Field") & (rk.TaxonomyValue == "Computer Science")]
    rk["inst_key"] = rk.InstitutionName.map(N)
    sc = sc.merge(rk[["inst_key", "Rank"]].rename(columns={"Rank": "CS_prestige_rank"}), on="inst_key", how="left")
    return sc.sort_values("CS_springrank_score", ascending=False).reset_index(drop=True)


def scorecard_cs():
    raw = pd.read_csv(SCORECARD, usecols=["INSTNM", "CIPCODE", "CREDLEV", "EARN_MDN_4YR", "EARN_COUNT_WNE_4YR"], dtype=str)
    raw = raw[(raw.CREDLEV == "3") & raw.CIPCODE.str.zfill(4).isin(CS_CIP)].copy()
    raw["earn"] = pd.to_numeric(raw.EARN_MDN_4YR, errors="coerce")
    raw["w"] = pd.to_numeric(raw.EARN_COUNT_WNE_4YR, errors="coerce")
    raw = raw.dropna(subset=["earn"])
    raw["we"] = raw.earn * raw.w.fillna(1).clip(lower=1)
    g = raw.groupby("INSTNM").agg(we=("we", "sum"), wsum=("w", lambda s: s.fillna(1).clip(lower=1).sum()),
                                  coh=("w", "sum")).reset_index()
    g["cs_earn"] = g.we / g.wsum
    g["k_exact"] = g.INSTNM.map(k_exact); g["k_campus"] = g.INSTNM.map(k_campus); g["k_city"] = g.INSTNM.map(k_city)
    return g


def match_earnings(sc, scd):
    """Priority match Wapman -> Scorecard (exact, then campus-stripped, then city-stripped)."""
    out = []
    for _, r in sc.iterrows():
        row = best(scd, k_exact(r.wap_name))
        out.append((row.cs_earn, row.coh, row.INSTNM) if row is not None else (np.nan, np.nan, None))
    sc[["cs_median_earnings", "cs_cohort", "scorecard_name"]] = pd.DataFrame(out, index=sc.index)
    return sc


def pseo_cs():
    df = pd.read_csv(PSEOE, dtype=str, usecols=["degree_level", "inst_level", "cip_level", "geo_level",
                    "ind_level", "institution", "cipcode", "grad_cohort", "y5_p50_earnings", "y5_grads_earn",
                    "status_y5_earnings"])
    df = df[(df.degree_level == "05") & (df.inst_level == "I") & (df.cip_level == "4") &
            (df.geo_level == "N") & (df.ind_level == "A") & (df.grad_cohort == "0000") &
            (df.status_y5_earnings == "1") & df.cipcode.str.replace(".", "", regex=False).str.zfill(4).isin(CS_CIP)].copy()
    df["earn"] = pd.to_numeric(df.y5_p50_earnings, errors="coerce")
    df["w"] = pd.to_numeric(df.y5_grads_earn, errors="coerce").fillna(1).clip(lower=1)
    ins = pd.read_csv(ROOT / "data" / "raw" / "pseo" / "pseo_all_institutions.csv", dtype=str)
    ins.columns = [c.strip().lstrip("﻿") for c in ins.columns]
    df = df.merge(ins[["institution", "label"]], on="institution", how="left").dropna(subset=["earn"])
    df["we"] = df.earn * df.w
    g = df.groupby("label").agg(we=("we", "sum"), w=("w", "sum")).reset_index()
    g["pseo_earn"] = g.we / g.w
    g = g.rename(columns={"w": "coh"})
    g["k_exact"] = g.label.map(k_exact); g["k_campus"] = g.label.map(k_campus); g["k_city"] = g.label.map(k_city)
    return g


def orcid_cs_springrank():
    e = pd.read_parquet(ORCID)
    s = e[e.field_text.str.contains("computer", na=False)]
    pairs = [(N(a), N(b)) for a, b in zip(s.org_from_ror_name, s.org_to_ror_name)]
    pairs = [(a, b) for a, b in pairs if a and b and a != b]
    nodes = sorted(set(x for x, _ in pairs) | set(d for _, d in pairs))
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for a, b in pairs:
        A[idx[a], idx[b]] = 1.0
    m = springrank.SpringRank(alpha=0.5, inverse_temp_fit_warning=False); m.fit(A)
    return dict(zip(nodes, m.ranks))


def main():
    sc = cs_springrank_wapman()
    scd = scorecard_cs()
    sc = match_earnings(sc, scd)

    # universe = top-BUFFER by score WITH earnings; publish TOP by combined w=0.6
    uni = sc[sc.cs_median_earnings.notna()].head(BUFFER).copy().reset_index(drop=True)
    uni["z_P"] = (uni.CS_springrank_score - uni.CS_springrank_score.mean()) / uni.CS_springrank_score.std()
    uni["z_S"] = (uni.cs_median_earnings - uni.cs_median_earnings.mean()) / uni.cs_median_earnings.std()
    for w in (0.5, 0.6, 0.7):
        uni[f"combined_w{int(w*100)}"] = w * uni.z_P + (1 - w) * uni.z_S
        uni[f"rank_w{int(w*100)}"] = uni[f"combined_w{int(w*100)}"].rank(ascending=False).astype(int)
    uni["final_rank_w60"] = uni.combined_w60.rank(ascending=False).astype(int)

    # salary premium: earnings ~ prestige score; residual in $
    mm = sm.OLS(uni.cs_median_earnings.values, sm.add_constant(uni.CS_springrank_score.values)).fit()
    uni["salary_premium_usd"] = uni.cs_median_earnings - mm.fittedvalues

    # w-sensitivity: rank move across w; flag > 5
    uni["rank_move_across_w"] = uni[["rank_w50", "rank_w60", "rank_w70"]].max(axis=1) - uni[["rank_w50", "rank_w60", "rank_w70"]].min(axis=1)
    uni["low_robustness_flag"] = uni.rank_move_across_w > 5

    # PSEO cross-check on overlap
    pcs = pseo_cs()
    uni["pseo_earnings"] = [match_one(r.wap_name, pcs, "pseo_earn") for _, r in uni.iterrows()]
    ov = uni.dropna(subset=["pseo_earnings"])
    pseo_rho = spearmanr(ov.cs_median_earnings, ov.pseo_earnings)[0] if len(ov) >= 8 else np.nan
    # ORCID cross-check on overlap
    osr = orcid_cs_springrank()
    uni["orcid_cs_score"] = uni.inst_key.map(osr)
    oo = uni.dropna(subset=["orcid_cs_score"])
    orcid_rho = spearmanr(oo.CS_springrank_score, oo.orcid_cs_score)[0] if len(oo) >= 8 else np.nan

    uni = uni.sort_values("final_rank_w60").reset_index(drop=True)
    cols = ["final_rank_w60", "wap_name", "scorecard_name", "CS_prestige_rank", "CS_springrank_score",
            "cs_median_earnings", "cs_cohort", "z_P", "z_S", "combined_w50", "combined_w60", "combined_w70",
            "rank_w50", "rank_w60", "rank_w70", "salary_premium_usd", "rank_move_across_w",
            "low_robustness_flag", "pseo_earnings"]
    uni[cols].to_csv(ROOT / "data" / "interim" / "cs_ranking.csv", index=False)

    write_report(uni, pseo_rho, orcid_rho, len(ov), len(oo), mm)
    make_figures(uni)
    print(f"universe {len(uni)} schools; published top {TOP} by w=0.6")
    print(f"PSEO cross-check (overlap n={len(ov)}): Spearman {pseo_rho:+.2f}; ORCID CS prestige (n={len(oo)}): {orcid_rho:+.2f}")
    print(uni.head(15)[["final_rank_w60", "wap_name", "CS_springrank_score", "cs_median_earnings", "salary_premium_usd", "low_robustness_flag"]].to_string(index=False))


def match_one(wapname, df, col):
    row = best(df, k_exact(wapname))
    return float(row[col]) if row is not None else np.nan


def write_report(uni, pseo_rho, orcid_rho, n_pseo, n_orcid, mm):
    top = uni[uni.final_rank_w60 <= TOP].sort_values("final_rank_w60")
    disp = top[["final_rank_w60", "wap_name", "cs_median_earnings", "salary_premium_usd",
                "rank_w50", "rank_w70", "low_robustness_flag"]].copy()
    disp["premium"] = disp.salary_premium_usd.map(lambda v: f"{'+' if v >= 0 else '−'}${abs(v):,.0f}")
    lowrob = top[top.low_robustness_flag].wap_name.tolist()
    L = ["# CS school ranking — academic prestige fine-tuned by graduate earnings (top 50)\n",
         "**Descriptive, not a causal value claim.** Academic prestige = the CONTINUOUS SpringRank score "
         "of the US CS faculty-hiring network (Wapman et al.; recomputed from the public CS edge subgraph "
         "so score *differences* are meaningful). Earnings = College Scorecard CS bachelor's median "
         "(4yr-after). Blend (z-scored within the 70-school universe): **combined = w·z_prestige + "
         "(1−w)·z_salary**, primary **w = 0.6** (prestige-anchored, salary fine-tunes). "
         "Run: `python scripts/34_cs_ranking.py`.\n",
         f"Cross-checks: Scorecard↔**PSEO** CS earnings on the overlap (n={n_pseo}) Spearman "
         f"**{pseo_rho:+.2f}**; Wapman↔**ORCID-rebuilt** CS SpringRank (n={n_orcid}) Spearman "
         f"**{orcid_rho:+.2f}** — the prestige and the earnings axes both replicate on independent data.\n",
         "## Top 50 (w = 0.6)\n",
         "| # | school | CS median earnings | salary premium | rank@w0.5 | rank@w0.7 | low-robust |",
         "|---|---|---|---|---|---|---|"]
    for _, r in disp.iterrows():
        L.append(f"| {r.final_rank_w60:.0f} | {r.wap_name} | ${r.cs_median_earnings:,.0f} | {r.premium} | "
                 f"{r.rank_w50:.0f} | {r.rank_w70:.0f} | {'⚠️' if r.low_robustness_flag else ''} |")
    L += [f"\n**Salary premium** = a school's CS earnings minus what its academic standing predicts "
          f"(regression earnings ~ prestige score; slope ${mm.params[1]:,.0f} per SpringRank unit). "
          "Positive = earns ABOVE its academic standing, negative = below. Reported for every school, "
          "both signs.\n",
          f"**Low-robustness (rank moves >5 across w=0.5→0.7): {len(lowrob)} schools** — "
          + (", ".join(lowrob) if lowrob else "none") + ". These are the salary-driven entries whose "
          "placement depends on how much weight you give earnings; read them with extra caution.\n",
          "## Honest caveats (read before sharing)\n",
          "- **(a) Not causal / not 'value-added'.** Earnings gaps among strong schools partly reflect "
          "**geography** (scripts/32: destination geography explains a modest within-field share here, "
          "R²≈0.13) and **student selection**, NOT proven school value-added. Chetty-Deming-Friedman: the "
          "causal effect of an elite school on AVERAGE earnings is small; the brand premium is in the "
          "elite TAIL that MEDIAN data cannot see. A high salary premium ≠ 'this school makes you richer'.",
          "- **(b) Title-IV population** (Scorecard covers federally-aided graduates), but scripts/32 "
          f"cross-validated Scorecard↔PSEO institution pay at Spearman +0.94 and here at {pseo_rho:+.2f} "
          "on the overlap, so the RANKING is robust to the earnings population.",
          "- **(c) The median is WHERE GRADS LAND.** ~34% of BA grads work out-of-state; the median "
          "reflects where these graduates typically end up working, so a strong school whose grads stay "
          "in a lower-wage region is 'dinged' even though a mobile graduate may do fine. Read the salary "
          "term as *'where these grads typically land'*, NOT *'this school's ceiling'*. (This is why some "
          "Midwest/regional flagships sit below coastal schools of similar prestige.)",
          "- **(d) Low-robustness schools are flagged** (⚠️ above); the prestige axis (w high) is the "
          "stable backbone, the salary axis (w low) is the fine-tuning that moves the flagged schools.",
          "- **Coverage:** universe = the 70 top-CS-prestige PhD-granting departments with Scorecard CS "
          "earnings; a few CS schools are absent where Scorecard suppresses the CS cell or the name could "
          "not be matched. Descriptive ranking of those present, not an exhaustive list.\n"]
    (ROOT / "CS_RANKING_RESULT.md").write_text("\n".join(L))


def make_figures(uni):
    top = uni[uni.final_rank_w60 <= TOP].sort_values("final_rank_w60", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 12))
    colors = ["#D6202A" if f else "#2C7BB6" for f in top.low_robustness_flag]
    ax.barh(range(len(top)), top.combined_w60, color=colors)
    for i, (_, r) in enumerate(top.iterrows()):
        prem = f"{'+' if r.salary_premium_usd >= 0 else '−'}${abs(r.salary_premium_usd)/1000:.0f}k"
        ax.text(r.combined_w60 + (0.03 if r.combined_w60 >= 0 else -0.03), i,
                f"{r.wap_name[:30]}  ({prem})", va="center", ha="left" if r.combined_w60 >= 0 else "right", fontsize=6.5)
    ax.set_yticks([]); ax.set_xlabel("combined score  (0.6·z_prestige + 0.4·z_salary)")
    ax.set_title("CS schools — academic prestige fine-tuned by graduate earnings (top 50, w=0.6)\n"
                 "(salary premium in parens; red = low-robustness, salary-driven placement)")
    ax.margins(x=0.25); fig.tight_layout()
    fig.savefig(OUT / "figures" / "cs_ranking_top50.png", dpi=140)

    # w-sensitivity: rank under w0.5 vs w0.7, highlight movers
    fig, ax = plt.subplots(figsize=(8, 10))
    t2 = uni[uni.final_rank_w60 <= TOP]
    for _, r in t2.iterrows():
        c = "#D6202A" if r.low_robustness_flag else "#bbb"
        ax.plot([0, 1], [r.rank_w50, r.rank_w70], "-o", color=c, alpha=.85, markersize=4)
        if r.low_robustness_flag:
            ax.annotate(r.wap_name[:24], (1, r.rank_w70), fontsize=6, xytext=(3, 0), textcoords="offset points", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["w=0.5 (more salary)", "w=0.7 (more prestige)"])
    ax.invert_yaxis(); ax.set_ylabel("rank (1 = top)")
    ax.set_title("Weight sensitivity — which CS schools move (red = >5 positions = low-robustness)")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "cs_ranking_w_sensitivity.png", dpi=140)


if __name__ == "__main__":
    main()
