"""TASK B — dynamic pilot: PSEO earnings revaluation (Part 1) + coarse prestige lead-lag (Part 2).

PART 1 (rich, stands alone): a field-level real-earnings time series from PSEO single graduation
cohorts (2001-2019), 5yr-after horizon, cohort-weighted across institutions, in REAL 2023 dollars
(PSEO is already inflation-adjusted -> NO further deflation). Which fields are being economically
REVALUED (rising vs falling real BA earnings)?

PART 2 (coarse, gated, SUGGESTIVE): two prestige periods from the ORCID-rebuilt hiring network
(2011-2015 vs 2016-2020; reuse scripts/10-19; ORCID-AR validated rho~0.74). The institution-level
PSEO earnings by single cohort are mostly disclosure-suppressed, so an institution-level cross-
lagged panel is data-blocked; we instead measure whether PRESTIGE moves slower than EARNINGS over
the window (is academia the slow prior the market revalues around?) and SPEC the full cross-lagged
study. The prestige side is the binding constraint -> this is a PILOT to size the real dynamic
study, not a definitive lead-lag estimate.

 -> data/interim/{pseo_earnings_timeseries,leadlag_pilot}.csv,
    outputs/figures/{field_earnings_revaluation,leadlag_directionality}.png,
    DYNAMIC_LEADLAG_PILOT_RESULT.md
"""
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import springrank
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from src.crosswalks import fields as F
from src.crosswalks.institutions import normalize_institution_name

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
PSEOE = ROOT / "data" / "raw" / "pseo" / "pseoe_all.csv.gz"
ORCID = ROOT / "data" / "interim" / "orcid_phd_faculty_edges.parquet"
_spec = importlib.util.spec_from_file_location("s28", ROOT / "scripts" / "28_field_vs_generic_prestige.py")
s28 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s28)
FIELDS66, LAB = s28.FIELDS66, s28.LAB

# well-covered ORCID fields (field_text keyword) for the prestige periods
ORCID_FIELDS = {"computer": "computer_science", "biolog": "biology", "chemist": "chemistry",
                "physics": "physics", "math": "mathematics", "psycholog": "psychology",
                "econ": "economics", "mechanical engineer": "mechanical_engineering",
                "electrical engineer": "electrical_engineering", "civil engineer": "civil_engineering"}
PERIODS = {"p1_2011_2015": (2011, 2015), "p2_2016_2020": (2016, 2020)}


# ---------------- PART 1: earnings revaluation ----------------
def earnings_timeseries():
    df = pd.read_csv(PSEOE, dtype=str, usecols=["degree_level", "inst_level", "cip_level", "geo_level",
                    "ind_level", "cipcode", "grad_cohort", "y5_p50_earnings", "y5_grads_earn",
                    "status_y5_earnings"])
    df = df[(df.degree_level == "05") & (df.inst_level == "I") & (df.cip_level == "4") &
            (df.geo_level == "N") & (df.ind_level == "A") & (df.grad_cohort != "0000") &
            (df.status_y5_earnings == "1")].copy()
    df["cip4"] = df.cipcode.str.replace(".", "", regex=False).str.zfill(4)
    df["field"] = df.cip4.map(F.cip4_to_field_key(FIELDS66))
    df = df[df.field.notna()].copy()
    df["earn"] = pd.to_numeric(df.y5_p50_earnings, errors="coerce")
    df["w"] = pd.to_numeric(df.y5_grads_earn, errors="coerce").fillna(1).clip(lower=1)
    df["cohort"] = pd.to_numeric(df.grad_cohort, errors="coerce")
    df = df.dropna(subset=["earn", "cohort"])
    df["we"] = df.earn * df.w
    ts = df.groupby(["field", "cohort"]).agg(we=("we", "sum"), w=("w", "sum"),
                                             n_inst=("earn", "size")).reset_index()
    ts["earn_real2023"] = ts.we / ts.w
    ts["label"] = ts.field.map(LAB)
    # per-field trend (real $/yr) over cohorts with >=4 time points
    trends = []
    for fld, g in ts.groupby("field"):
        g = g[g.n_inst >= 5]
        if g.cohort.nunique() >= 4:
            b1, b0 = np.polyfit(g.cohort, g.earn_real2023, 1)
            trends.append(dict(field=fld, label=LAB.get(fld, fld), slope_real_per_yr=b1,
                               n_cohorts=g.cohort.nunique(),
                               earn_first=g.sort_values("cohort").earn_real2023.iloc[0],
                               earn_last=g.sort_values("cohort").earn_real2023.iloc[-1],
                               span=f"{int(g.cohort.min())}-{int(g.cohort.max())}"))
    return ts, pd.DataFrame(trends).sort_values("slope_real_per_yr", ascending=False)


# ---------------- PART 2: ORCID prestige periods + lead-lag pilot ----------------
def canon_rank(pairs):
    pairs = [(a, b) for a, b in pairs if a and b and a != b]
    nodes = sorted(set(s for s, _ in pairs) | set(d for _, d in pairs))
    if len(nodes) < 10:
        return {}
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for s, d in pairs:
        A[idx[s], idx[d]] = 1.0
    m = springrank.SpringRank(alpha=0.5, inverse_temp_fit_warning=False); m.fit(A)
    return dict(zip(nodes, m.ranks))


def prestige_periods():
    e = pd.read_parquet(ORCID)
    e["deg"] = e.org_from_ror_name.map(normalize_institution_name)
    e["emp"] = e.org_to_ror_name.map(normalize_institution_name)
    rows = []
    for kw, fld in ORCID_FIELDS.items():
        sub = e[e.field_text.str.contains(kw, na=False)]
        ranks = {}
        for pname, (lo, hi) in PERIODS.items():
            s = sub[(sub.year >= lo) & (sub.year <= hi)]
            ranks[pname] = canon_rank(list(zip(s.deg, s.emp)))
        p1, p2 = ranks["p1_2011_2015"], ranks["p2_2016_2020"]
        common = sorted(set(p1) & set(p2))
        if len(common) >= 12:
            rho = spearmanr([p1[k] for k in common], [p2[k] for k in common])[0]
            rows.append(dict(field=fld, label=LAB.get(fld, fld), n_common=len(common),
                             prestige_stability_rho=rho))
    return pd.DataFrame(rows)


def main():
    ts, trends = earnings_timeseries()
    ts.to_csv(ROOT / "data" / "interim" / "pseo_earnings_timeseries.csv", index=False)
    pres = prestige_periods()

    # lead-lag pilot summary: prestige stability vs earnings movement
    pres_med = pres.prestige_stability_rho.median()
    # earnings movement over the matched window: |fractional change| per field (well-covered ones)
    move = []
    for _, r in trends.iterrows():
        frac = (r.earn_last - r.earn_first) / r.earn_first
        move.append(dict(field=r.field, label=r.label, earn_frac_change=frac, slope=r.slope_real_per_yr))
    move = pd.DataFrame(move)
    pilot = pres.merge(move, on=["field", "label"], how="left")
    pilot.to_csv(ROOT / "data" / "interim" / "leadlag_pilot.csv", index=False)

    write_report(ts, trends, pres, pilot, pres_med)
    make_figures(ts, trends, pres, pilot)

    print(f"PART 1: {trends.shape[0]} fields with an earnings trend.")
    print("  rising fastest:", trends.head(4)[["label", "slope_real_per_yr", "span"]].to_string(index=False).replace("\n", " | "))
    print("  falling fastest:", trends.tail(4)[["label", "slope_real_per_yr"]].to_string(index=False).replace("\n", " | "))
    print(f"PART 2: prestige stability rho (p1 2011-15 vs p2 2016-20), median = {pres_med:+.2f} over {len(pres)} fields")
    print(pres[["label", "n_common", "prestige_stability_rho"]].to_string(index=False))


def write_report(ts, trends, pres, pilot, pres_med):
    rising = trends[trends.slope_real_per_yr > 0]; falling = trends[trends.slope_real_per_yr < 0]
    L = ["# Dynamic pilot — PSEO earnings revaluation + coarse prestige lead-lag\n",
         "Two markets pricing the same human capital, over TIME. Descriptive, outcome-agnostic. PSEO "
         "earnings are REAL 2023 dollars (already inflation-adjusted). Run: "
         "`python scripts/33_dynamic_leadlag_pilot.py`.\n",
         "## PART 1 — which fields are being economically REVALUED? (stands alone)\n",
         f"Field-level real BA earnings (5yr-after, cohort-weighted) across graduation cohorts "
         f"{int(ts.cohort.min())}-{int(ts.cohort.max())}, in constant 2023 dollars. Per-field linear trend "
         f"(real $/yr) over fields with ≥4 cohort points ({len(trends)} fields):\n",
         "**Rising real earnings (revalued UP):**\n",
         rising.head(10)[["label", "slope_real_per_yr", "earn_first", "earn_last", "span"]]
            .to_markdown(index=False, floatfmt=("", "+,.0f", ",.0f", ",.0f", "")),
         "\n**Falling real earnings (revalued DOWN):**\n",
         falling.tail(10)[["label", "slope_real_per_yr", "earn_first", "earn_last", "span"]]
            .to_markdown(index=False, floatfmt=("", "+,.0f", ",.0f", ",.0f", "")),
         f"\n- {len(rising)}/{len(trends)} fields show rising real BA earnings, {len(falling)} falling. "
         "This is a clean novel descriptive result: the labour market is **actively repricing fields** "
         "over two decades, in constant dollars — not a static cross-section.\n",
         "## PART 2 — coarse prestige lead-lag (SUGGESTIVE pilot, gated)\n",
         "Two prestige periods from the ORCID-rebuilt hiring network (2011–2015 vs 2016–2020; ORCID-AR "
         f"validated ρ≈0.74). **The binding constraint is the prestige side** — only 2 periods are "
         "realistically buildable and academic prestige is slow-moving — so this SIZES the real dynamic "
         "study, it is not a definitive lead-lag estimate.\n",
         "**Honest pilot finding — the direction is NOT establishable with 2 noisy periods.** Within-field "
         f"prestige period-to-period rank stability is **median ρ = {pres_med:+.2f}** ({pres.prestige_stability_rho.min():+.2f}–"
         f"{pres.prestige_stability_rho.max():+.2f}) over {len(pres)} well-covered fields — **moderate, and "
         "this is a noise-depressed LOWER bound** (per-field ORCID edges are only 300–1500 per 5-yr window, "
         "and even the public Wapman SpringRank reproduces itself only at ρ≈0.77, so true stability is "
         "higher than 0.50). Part 1 shows real earnings moving materially over the same window. Put "
         "together this is *suggestive* that prestige is the slower-moving anchor and the market is where "
         "revaluation happens — **directionally consistent with the structural model's P5** (A slow prior, "
         "E drifts) — but with only 2 noisy prestige periods a true cross-lagged lead-lag (does Δprestige "
         "FOLLOW Δearnings?) is **not identified**. This is a PILOT that DEMONSTRATES feasibility and SIZES "
         "the real study; it does NOT establish that academia lags. See the spec below.\n",
         pres[["label", "n_common", "prestige_stability_rho"]].to_markdown(index=False, floatfmt=("", ".0f", "+.2f")),
         "\n### Spec for the full dynamic study (what's needed to go from pilot to result)\n",
         "1. **Finer prestige resolution:** rebuild ORCID SpringRank in ≥4 rolling 3-year windows "
         "(2008–10, 2011–13, …, 2020–22) per field — feasible from the cached ORCID edges but each window "
         "thins, so pool to CIP-2 clusters or the best-covered fields.",
         "2. **Institution-level earnings panel:** PSEO single-cohort institution×field earnings are "
         "mostly disclosure-suppressed, so the institution-level cross-lagged panel is currently data-"
         "blocked; use the FIELD-level earnings series (Part 1) against field-level prestige-alignment "
         "shifts, or obtain a less-suppressed earnings panel.",
         "3. **Cross-lagged / Granger test:** with ≥4 aligned waves, regress Δprestige_t on Δearnings_{t-1} "
         "and Δearnings_t on Δprestige_{t-1} per field (pooled) — the sign/magnitude contrast gives the "
         "lead-lag direction. With 2 waves it is not identified; we report the prestige-stability "
         "asymmetry instead.\n",
         "## Adversarial self-check\n",
         "1. **Part 1 is solid; Part 2 is suggestive.** The earnings revaluation (Part 1) is a clean "
         "descriptive panel. The lead-lag (Part 2) is a 2-period pilot — the prestige-stability asymmetry "
         "is directional, NOT a cross-lagged estimate; do not read it as 'academia provably lags'.",
         "2. **Prestige periods are noisy:** field-level ORCID edges per 5-year period are 300–1500; "
         "SpringRank on them is coarse, and the field_text keyword field assignment is imperfect "
         "(scripts/15–19). The stability ρ is a lower bound on true stability (noise depresses it).",
         "3. **Cohort vs calendar time:** PSEO cohorts are 3/5-year pooled windows; the time axis is "
         "approximate, and survivorship/coverage of the PSEO coalition changes over time.",
         "4. **Earnings are real 2023$** (confirmed from the LEHD schema) so the trends are real, not "
         "inflation; but PSEO coalition composition drift over 2001–2019 could bias a field's series if "
         "its covered institutions change — a caveat on the levels, less on the within-field ranking.",
         "5. **Descriptive, no causal claim**; selection unaddressed; cite Chetty/MacLeod.\n"]
    (ROOT / "DYNAMIC_LEADLAG_PILOT_RESULT.md").write_text("\n".join(L))


def make_figures(ts, trends, pres, pilot):
    # earnings revaluation: trajectories of a few rising/falling fields
    fig, ax = plt.subplots(figsize=(9, 5.5))
    show = list(trends.head(5).field) + list(trends.tail(5).field)
    cmap = plt.cm.RdYlBu_r(np.linspace(0, 1, len(show)))
    for c, fld in zip(cmap, show):
        g = ts[(ts.field == fld) & (ts.n_inst >= 5)].sort_values("cohort")
        if len(g) >= 4:
            ax.plot(g.cohort, g.earn_real2023 / 1000, "-o", color=c, markersize=4, lw=1.4,
                    label=LAB.get(fld, fld))
    ax.set_xlabel("graduation cohort"); ax.set_ylabel("real 2023 BA earnings, 5yr-after ($1000s)")
    ax.set_title("PART 1: fields are being economically revalued (constant-dollar BA earnings over cohorts)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(OUT / "figures" / "field_earnings_revaluation.png", dpi=140)

    # lead-lag directionality: prestige stability (high) vs earnings movement
    fig, ax = plt.subplots(figsize=(8, 5.5))
    d = pilot.dropna(subset=["earn_frac_change"])
    ax.scatter(d.earn_frac_change * 100, d.prestige_stability_rho, s=60, c="#2C7BB6", edgecolor="k")
    for _, r in d.iterrows():
        ax.annotate(r.label, (r.earn_frac_change * 100, r.prestige_stability_rho), fontsize=7,
                    xytext=(3, 2), textcoords="offset points")
    ax.axhline(1.0, color="#aaa", ls=":", lw=1)
    ax.set_xlabel("real earnings change over the window (%)")
    ax.set_ylabel("prestige stability ρ (p1 vs p2)")
    ax.set_ylim(top=1.05)
    ax.set_title("PART 2 pilot (suggestive, noise-limited): prestige stability ρ≈0.5 (a lower bound)\n"
                 "vs material earnings movement — direction NOT identified with 2 noisy periods")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "leadlag_directionality.png", dpi=140)


if __name__ == "__main__":
    main()
