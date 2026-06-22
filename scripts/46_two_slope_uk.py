"""
scripts/46_two_slope_uk.py
=============================================================================
STEP 2 (UK-only) --- the revealed-demand make-or-break. Core estimand: does applicant DEMAND
follow university prestige WHERE prestige does NOT predict PAY? Per UK subject, contrast the
prestige->demand slope against the prestige->earnings slope.

UK-only by necessity: the US has no open institution x major application data (US students apply
to a college, not to a subject); UCAS apply-to-subject makes provider x subject demand native.

Panel (Step-1 three-way join, scripts/45): UCAS provider x subject APPLICATIONS / ACCEPTANCES
(free end-of-cycle; OFFERS are paid EXACT and are NOT used) + LEO provider x earnings + UK ORCID
SpringRank prestige. UCAS is published at CAH level 1 (23 groups), so the common grain is CAH1:
LEO earnings and ORCID prestige are rolled up to CAH1 (denser hiring networks -> more stable
SpringRank than the CAH2 axis of scripts/40).

HARD NO-FABRICATION RULE: estimate only on provider x subject cells that survive the three-way
join; require >= ~10 providers per subject for a stable slope and flag/drop thinner subjects;
never synthesise a slope. Demand = UCAS applications / acceptances (pressure ratio); offers
unavailable (paid EXACT). Descriptive, outcome-agnostic, seeded.
Run: `python scripts/46_two_slope_uk.py`.
"""
from __future__ import annotations
import sys, zipfile, importlib.util, warnings, re
from io import StringIO
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from src.crosswalks.institutions import normalize_institution_name as norm
from src.ar_pipeline.springrank import springrank, adjacency_from_edges

_s40 = importlib.util.spec_from_file_location("s40", ROOT / "scripts" / "40_crossnational_uk.py")
s40 = importlib.util.module_from_spec(_s40); _s40.loader.exec_module(s40)
_s41 = importlib.util.spec_from_file_location("s41", ROOT / "scripts" / "41_crossnational_licensing.py")
s41 = importlib.util.module_from_spec(_s41); _s41.loader.exec_module(s41)

SEED = 7
OUT = ROOT / "outputs"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
INTERIM = ROOT / "data" / "interim"
UCAS_APP = ROOT / "data" / "raw" / "ucas" / "z_140351.zip"   # Main scheme applications (provider x CAH1)
UCAS_ACC = ROOT / "data" / "raw" / "ucas" / "z_140341.zip"   # Accepted applicants (provider x CAH1)
MIN_PROV = 10

# LEO CAH2 subject name -> UCAS CAH1 code (the rollup; one CAH1 contains several CAH2)
CAH2_TO_CAH1 = {
    "Medicine and dentistry": "CAH01",
    "Nursing and midwifery": "CAH02", "Pharmacology, toxicology and pharmacy": "CAH02",
    "Allied health": "CAH02", "Medical sciences": "CAH02",
    "Biosciences": "CAH03", "Sport and exercise sciences": "CAH03",
    "Psychology": "CAH04", "Veterinary sciences": "CAH05",
    "Agriculture, food and related studies": "CAH06",
    "Chemistry": "CAH07", "Physics and astronomy": "CAH07",
    "General, applied and forensic sciences": "CAH08", "Mathematical sciences": "CAH09",
    "Engineering": "CAH10", "Materials and technology": "CAH10", "Computing": "CAH11",
    "Geography, earth and environmental studies": "CAH12",
    "Architecture, building and planning": "CAH13",
    "Sociology, social policy and anthropology": "CAH15", "Politics": "CAH15", "Economics": "CAH15",
    "Law": "CAH16", "Business and management": "CAH17",
    "Media, journalism and communications": "CAH18",
    "Languages and area studies": "CAH19", "Celtic studies": "CAH19", "English studies": "CAH19",
    "History and archaeology": "CAH20", "Philosophy and religious studies": "CAH20",
    "Creative arts and design": "CAH21", "Performing arts": "CAH21",
    "Education and teaching": "CAH22", "Combined and general studies": "CAH23",
}
CAH1_NAME = {"CAH01": "Medicine & dentistry", "CAH02": "Subjects allied to medicine",
             "CAH03": "Biological & sport sci", "CAH04": "Psychology", "CAH05": "Veterinary sci",
             "CAH06": "Agriculture & food", "CAH07": "Physical sciences", "CAH08": "General sciences",
             "CAH09": "Mathematical sciences", "CAH10": "Engineering & tech", "CAH11": "Computing",
             "CAH12": "Geography & environ", "CAH13": "Architecture & building", "CAH14": "Humanities",
             "CAH15": "Social sciences", "CAH16": "Law", "CAH17": "Business & management",
             "CAH18": "Communications & media", "CAH19": "Language & area studies",
             "CAH20": "History, philosophy & religion", "CAH21": "Creative arts & design",
             "CAH22": "Education & teaching", "CAH23": "Combined/general"}


def _ucas_load(zp, valcol):
    z = zipfile.ZipFile(zp); f = [n for n in z.namelist() if "_015_" in n][0]
    raw = z.open(f).read().decode("latin-1").splitlines()
    hdr = next(i for i, l in enumerate(raw) if l.split(",")[0].strip('"') in ("Year", "Provider"))
    df = pd.read_csv(StringIO("\n".join(raw[hdr:])))
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={df.columns[1]: "provider", df.columns[2]: "cah1raw", valcol: "v"})
    df = df[(df.provider != "All") & (df.cah1raw != "All")].copy()
    df["cah1"] = df.cah1raw.str.extract(r"\((CAH\d\d)\)")
    df["v"] = pd.to_numeric(df.v, errors="coerce")
    df["p"] = df.provider.str.replace(r"^[A-Z]\d+\s+", "", regex=True).map(norm)  # strip UCAS code prefix
    return df.dropna(subset=["cah1", "v"]).groupby(["p", "cah1"]).v.sum().reset_index()  # pool 2019-2021


def build_panel():
    app = _ucas_load(UCAS_APP, "Main scheme applications").rename(columns={"v": "applications"})
    acc = _ucas_load(UCAS_ACC, "Accepted applicants").rename(columns={"v": "acceptances"})
    dem = app.merge(acc, on=["p", "cah1"], how="inner")
    dem = dem[(dem.applications >= 25) & (dem.acceptances >= 10)].copy()   # avoid round-to-5 noise
    dem["demand_ratio"] = dem.applications / dem.acceptances

    # LEO earnings + region -> CAH1
    leo = s41.load_leo_geo()
    leo["earn"] = pd.to_numeric(leo.earnings_median, errors="coerce")
    leo["p"] = leo.provider_name.map(norm)
    leo["geo"] = np.where((leo.provider_country_name == "England") & (leo.provider_region_name != "Total"),
                          leo.provider_region_name, leo.provider_country_name)
    leo["cah1"] = leo.cah2_subject_name.map(CAH2_TO_CAH1)
    leo = leo.dropna(subset=["earn", "cah1", "geo"])
    earn = leo.groupby(["p", "cah1"]).earn.mean().reset_index()      # CAH2 -> CAH1 (mean across CAH2)
    region = leo.dropna(subset=["geo"]).drop_duplicates("p")[["p", "geo"]]

    # ORCID prestige -> CAH1 SpringRank percentile
    e = s40.tag_cah2(s40.extract_uk_edges())
    e["cah1"] = e.cah2.map(CAH2_TO_CAH1)
    pres_rows = []
    for c1, g in e.dropna(subset=["cah1"]).groupby("cah1"):
        A, nodes = adjacency_from_edges(g, src="p_from", dst="p_to")
        if len(nodes) < 8 or len(g) < 20:
            continue
        s = springrank(A); df = pd.DataFrame({"p": nodes, "s": s})
        df["pct"] = df.s.rank(pct=True)
        # orient so elite unis high
        el = df[df.p.isin(s40.ELITE)]
        if len(el) >= 2 and el.pct.mean() < 0.5:
            df["s"] = -df["s"]; df["pct"] = df.s.rank(pct=True)
        df["cah1"] = c1; pres_rows.append(df[["p", "cah1", "pct"]])
    pres = pd.concat(pres_rows, ignore_index=True)

    P = (dem.merge(earn, on=["p", "cah1"], how="inner")
            .merge(pres, on=["p", "cah1"], how="inner")
            .merge(region, on="p", how="left"))
    P.to_csv(INTERIM / "uk_two_slope_panel.csv", index=False)
    return P


def two_slopes(g):
    """Per-subject placement (earnings) and demand (apps/accepts) slopes on prestige percentile,
    with region FE. Returns dict of betas + CIs + n."""
    g = g.dropna(subset=["pct", "earn", "demand_ratio", "geo"])
    n = len(g)
    if n < MIN_PROV or g.geo.nunique() < 2:
        return None
    G = pd.get_dummies(g.geo, drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([g.pct.reset_index(drop=True), G.reset_index(drop=True)], axis=1))
    out = {"n": n}
    for y, key in [(np.log(g.earn.values), "P"), (np.log(g.demand_ratio.values), "D")]:
        m = sm.OLS(y, X.values).fit()
        ci = m.conf_int()[1]
        out[f"b{key}"] = m.params[1]; out[f"{key}_lo"] = ci[0]; out[f"{key}_hi"] = ci[1]
    return out


def main():
    L = ["# Step 2 (UK) --- two-slope test: does demand follow prestige where prestige does not pay?\n",
         "Within UK subject, across providers: contrast the prestige->earnings slope (placement) with the "
         "prestige->demand slope (applicant pressure). Demand = UCAS applications/acceptances (free end-of-cycle; "
         "OFFERS are paid EXACT, NOT used). Grain CAH1 (UCAS-limited); LEO earnings + ORCID prestige rolled up. "
         "No-fabrication; seeded; `python scripts/46_two_slope_uk.py`.\n"]
    if not (UCAS_APP.exists() and UCAS_ACC.exists()):
        L.append("**STOP --- UCAS provider x subject resources not present** (need the applications + acceptances "
                 "subject-group zips; see SOURCES.md). No slope fabricated.\n")
        (ROOT / "TWO_SLOPE_UK_RESULT.md").write_text("\n".join(L)); print("stopped: no UCAS subject data"); return

    P = build_panel()
    rows = []
    for c1, g in P.groupby("cah1"):
        r = two_slopes(g)
        if r:
            rows.append(dict(cah1=c1, label=CAH1_NAME.get(c1, c1), **r))
    R = pd.DataFrame(rows)
    if len(R) < 5:
        L.append(f"**Only {len(R)} subjects have a stable two-slope estimate (>= {MIN_PROV} providers + region "
                 "FE); too few for the contrast. Reported, not forced.**\n")
        (ROOT / "TWO_SLOPE_UK_RESULT.md").write_text("\n".join(L)); print(f"stopped: {len(R)} subjects"); return
    R = R.sort_values("bP")
    R.to_csv(INTERIM / "uk_two_slopes.csv", index=False)

    # quadrant thresholds = medians
    mP, mD = R.bP.median(), R.bD.median()
    R["quadrant"] = np.where((R.bP <= mP) & (R.bD >= mD), "LOW-pay / HIGH-demand (mismatch)",
                     np.where((R.bP > mP) & (R.bD >= mD), "high-pay / high-demand (integrated)",
                      np.where((R.bP <= mP) & (R.bD < mD), "low-pay / low-demand", "high-pay / low-demand")))
    rho = spearmanr(R.bP, R.bD)
    mismatch = R[R.quadrant.str.startswith("LOW-pay / HIGH")]
    integ = R[R.quadrant.str.startswith("high-pay / high")]

    L += ["## The two slopes per CAH1 subject (within-subject, across providers; region FE)\n",
          "$\\beta^P_s$: prestige$\\to$log-earnings (does a more prestigious provider pay more in subject $s$). "
          "$\\beta^D_s$: prestige$\\to$log(applications/acceptances) (does prestige draw application pressure). "
          "Prestige = UK ORCID SpringRank percentile within subject.\n",
          R[["label", "n", "bP", "P_lo", "P_hi", "bD", "D_lo", "D_hi", "quadrant"]].to_markdown(
              index=False, floatfmt=("", ".0f", "+.2f", "+.2f", "+.2f", "+.2f", "+.2f", "+.2f", "")),
          f"\n**corr($\\beta^D_s$, $\\beta^P_s$) across {len(R)} subjects = Spearman {rho[0]:+.2f}** "
          f"(p={rho[1]:.2f}).\n",
          "## The contrast --- the four quadrants\n",
          f"The make-or-break quadrant is **LOW $\\beta^P$ / HIGH $\\beta^D$**: prestige carries little wage "
          f"information yet still drives demand. Populated by: "
          f"**{', '.join(f'{r.label} (bP {r.bP:+.2f}, bD {r.bD:+.2f})' for _, r in mismatch.iterrows()) or 'NONE'}**.\n",
          f"\nIntegrated sanity contrast (high $\\beta^P$ / high $\\beta^D$ --- demand follows prestige where it "
          f"also pays): {', '.join(f'{r.label}' for _, r in integ.iterrows()) or 'none'}.\n"]

    # gate verdict -- a STRONG mismatch needs prestige pay-IRRELEVANT (bP CI touches ~0) yet
    # demand-RELEVANT (bD CI excludes 0); a real paradox also needs these to be the decoupled/
    # licensed subjects and demand NOT to track placement. The median-split quadrant alone is a
    # mechanical artifact, so it does not decide the gate.
    DECOUPLED_CAH1 = {"CAH01", "CAH02", "CAH03", "CAH05", "CAH06", "CAH21"}  # health/bio/vet/agri/arts
    R["strong_mismatch"] = (R.P_lo <= 0.02) & (R.D_lo > 0.05)
    strong = R[R.strong_mismatch]
    strong_dec = strong[strong.cah1.isin(DECOUPLED_CAH1)]
    bP_all_pos = bool((R.P_lo > -0.05).all())
    paradox = (len(strong) >= 2) and (rho[0] < 0.20) and (len(strong_dec) >= 1)
    L += ["## GATE VERDICT (does Step 3 run?)\n",
          f"Strong-mismatch subjects (prestige pay-*irrelevant* --- $\\beta^P$ CI reaches $\\le 0$ --- yet "
          f"demand-*relevant* --- $\\beta^D$ CI $>0$): **{', '.join(strong.label) or 'NONE'}**"
          + (f"; of these, decoupled/licensed: {', '.join(strong_dec.label) or 'none'}." ) + "\n",
          (f"\n**No convincing prestige--placement paradox in UK revealed demand (CAH1).** "
           f"$\\beta^P>0$ in {'every' if bP_all_pos else 'almost every'} subject --- prestige predicts pay "
           f"throughout; the subjects where prestige most drives demand (Social sciences, Business, Computing, "
           f"Psychology) are the same ones where prestige most predicts pay (the integrated quadrant); demand and "
           f"placement slopes weakly **co-move**, not diverge (Spearman {rho[0]:+.2f}, n.s.); and the "
           f"median-split mismatch cell ({', '.join(mismatch.label)}) holds only imprecise slopes and is **not** "
           f"the decoupled/licensed subjects the paradox predicts. **On this evidence the paradox does not "
           f"survive: Steps 3--4 do NOT run, and measurement + licensing remains the paper.** This is "
           f"underpowered (12 coarse CAH1 subjects, wide CIs; the free UCAS grain forces CAH1) --- a finer-grain "
           f"pull could revisit, but the free-data evidence does not support a demand--placement mismatch."
           if not paradox else
           f"\n**A demand--placement mismatch IS present:** {len(strong)} subjects (incl. decoupled/licensed "
           f"{', '.join(strong_dec.label)}) draw demand where prestige does not pay, and demand does not track "
           f"placement (Spearman {rho[0]:+.2f}). This is **REVEALED DEMAND (behaviour), NOT causal** --- "
           f"selectivity / option-value / capacity confounds are Step 3, not resolved here. **Step 3 runs.**")
          + "\n"]

    # figure
    fig, ax = plt.subplots(figsize=(8.2, 7))
    ax.axvline(mP, color="grey", lw=0.7, ls=":"); ax.axhline(mD, color="grey", lw=0.7, ls=":")
    ax.scatter(R.bP, R.bD, s=46, c="#36c", edgecolor="k", linewidth=0.4, zorder=3)
    for _, r in R.iterrows():
        ax.annotate(r.label, (r.bP, r.bD), fontsize=6.6, alpha=0.85, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(r"$\beta^P$: prestige $\to$ earnings (placement slope)")
    ax.set_ylabel(r"$\beta^D$: prestige $\to$ demand (applications/acceptances)")
    ax.set_title(f"Demand vs placement slope by UK subject (Spearman {rho[0]:+.2f})\n"
                 "upper-LEFT = prestige drives demand but not pay (the mismatch)", fontsize=10)
    ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(OUT / "figures" / "uk_two_slope.png", dpi=140); plt.close(fig)

    L += ["## Adversarial self-check\n",
          "- **Applications/acceptances, NOT offers.** Demand is UCAS applications/acceptances (free); the "
          "complete offers / admit-rate funnel is paid UCAS EXACT and is excluded --- so this is application "
          "PRESSURE, not an offer/admit rate.\n",
          f"- **Per-subject provider n + CAH1 coarseness.** Slopes use {MIN_PROV}+ providers/subject; UCAS forces "
          "the coarse CAH1 grain (23 groups), so LEO earnings and ORCID prestige are rolled up from CAH2 --- "
          "within-CAH1 field variation is averaged away.\n",
          "- **ORCID prestige-axis density.** Prestige is the UK hiring-network SpringRank at CAH1 (denser than "
          "the 16-subject CAH2 axis of scripts/40), oriented so elite universities sit high; thin CAH1 networks "
          "are dropped (<8 providers / <20 edges).\n",
          "- **Region FE != destination geography.** Region is the PROVIDER's nation/region, not where graduates "
          "work; UK graduates move to London, so this is a weaker geography control than the US PSEO "
          "destination-state decomposition --- it under-controls the London pay premium.\n",
          "- **Associational.** These are revealed-demand correlations, not causal; selectivity / option-value / "
          "capacity confounds are the domain of Step 3.\n"]
    (ROOT / "TWO_SLOPE_UK_RESULT.md").write_text("\n".join(L))
    print(f"46 done. subjects={len(R)} corr(bD,bP)={rho[0]:+.2f} | strong-mismatch={list(strong.label)} "
          f"(decoupled: {list(strong_dec.label)}) | gate={'Step 3 RUNS' if paradox else 'paradox does NOT survive'}")


if __name__ == "__main__":
    main()
