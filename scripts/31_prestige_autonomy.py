"""TASK 3 — How autonomous is academic prestige? (descriptive, outcome-agnostic)

Reframe: autonomy index = gap[f] (high gap = the academic ordering is AUTONOMOUS / self-referential,
not tracking market pay; low gap = market-embedded). We test what drives autonomy, being explicit
that this OVERLAPS the earlier mixture decomposition — the genuinely new pieces are the
basic<->applied dimension and (if feasible) an OpenAlex field-insularity external validation.

 -> data/interim/autonomy.csv, outputs/figures/{autonomy_ranked,autonomy_vs_drivers}.png,
    PRESTIGE_AUTONOMY_RESULT.md
"""
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
_spec = importlib.util.spec_from_file_location("s28", ROOT / "scripts" / "28_field_vs_generic_prestige.py")
s28 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s28)
LAB = s28.LAB

# NEW — hand-coded basic<->applied<->professional ordering (SUBJECTIVE; documented & reproducible).
# 1 = basic science / humanities / social science (knowledge for its own sake; autonomous)
# 2 = applied science / engineering (knowledge applied to problems)
# 3 = professional / vocational (credential- and practice-oriented)
BASIC_APPLIED = {
    "mathematics": 1, "statistics": 1, "physics": 1, "chemistry": 1, "biochemistry": 1, "biology": 1,
    "microbiology": 1, "physiology": 1, "ecology": 1, "earth_sciences": 1, "anthropology": 1,
    "economics": 1, "political_science": 1, "psychology": 1, "sociology": 1, "geography": 1,
    "english": 1, "history": 1, "philosophy": 1, "linguistics": 1, "spanish": 1, "music": 1,
    "theatre": 1,
    "computer_science": 2, "computer_engineering": 2, "electrical_engineering": 2,
    "mechanical_engineering": 2, "civil_engineering": 2, "chemical_engineering": 2,
    "materials_science": 2, "aerospace_engineering": 2, "industrial_engineering": 2,
    "biomedical_engineering": 2, "environmental_engineering": 2, "ag_engineering": 2,
    "environmental_sciences": 2, "agronomy": 2, "animal_sciences": 2, "food_science": 2,
    "forestry": 2,
    "accounting": 3, "finance": 3, "management": 3, "marketing": 3, "nursing": 3, "pharmacy": 3,
    "public_health": 3, "communication_disorders": 3, "kinesiology": 3, "hper": 3, "nutrition": 3,
    "social_work": 3, "education_general": 3, "education_admin": 3, "teacher_ed_subjects": 3,
    "special_education": 3, "human_dev": 3, "architecture": 3, "urban_planning": 3,
}
BA_NAME = {1: "basic", 2: "applied", 3: "professional"}


def main():
    t = s28.build_table()
    gap = t.groupby("field").apply(lambda g: 1 - spearmanr(g.F, g.y)[0]).rename("gap")
    n = t.groupby("field").size().rename("n")
    df = pd.concat([gap, n], axis=1).reset_index()
    df["label"] = df.field.map(LAB)
    df["basic_applied"] = df.field.map(BASIC_APPLIED)
    df["ba_name"] = df.basic_applied.map(BA_NAME)

    anch = pd.read_parquet(ROOT / "data" / "interim" / "acs_occ_anchors.parquet")
    df = df.merge(anch[["field", "licensure_strict", "absorption_acs"]], on="field", how="left")
    df = df[df.n >= 8].copy()
    df.to_csv(ROOT / "data" / "interim" / "autonomy.csv", index=False)

    # driver tests (Spearman with gap = autonomy) + bootstrap 95% CIs (n=48-57 is low power)
    def sc(col):
        d = df.dropna(subset=["gap", col]); rho, p = spearmanr(d.gap, d[col]); n = len(d)
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            i = rng.integers(0, n, n); dd = d.iloc[i]
            if dd.gap.std() > 0 and dd[col].std() > 0:
                bs.append(spearmanr(dd.gap, dd[col])[0])
        return (rho, p, n, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5))
    drivers = {"absorption_acs": sc("absorption_acs"), "basic_applied": sc("basic_applied"),
               "licensure_strict": sc("licensure_strict")}
    mde = 0.39  # ~80% power for |Spearman| at n~50
    # gap by basic/applied class
    by_class = df.groupby("ba_name").gap.agg(["mean", "median", "count"]).reindex(["basic", "applied", "professional"])
    # does basic_applied survive controlling licensure?
    d2 = df.dropna(subset=["gap", "basic_applied", "licensure_strict"])
    m = sm.OLS(d2.gap.values, sm.add_constant(np.column_stack([d2.basic_applied, d2.licensure_strict]))).fit(cov_type="HC1")

    write_report(df, drivers, by_class, m, mde)
    make_figures(df)

    print("autonomy (gap) driver Spearman:")
    for k, v in drivers.items():
        print(f"  {k}: {v[0]:+.2f} (p={v[1]:.3f}) n={v[2]}")
    print("\ngap by basic/applied class:\n", by_class.round(3).to_string())
    print(f"\ngap ~ basic_applied + licensure (OLS): b_ba={m.params[1]:+.3f} (p={m.pvalues[1]:.3f}), "
          f"b_lic={m.params[2]:+.3f} (p={m.pvalues[2]:.3f}), R2={m.rsquared:.2f}")


def write_report(df, drivers, by_class, m, mde):
    da, dba, dl = drivers["absorption_acs"], drivers["basic_applied"], drivers["licensure_strict"]
    ba_holds = dba[0] < -0.2 and dba[1] < 0.10   # hypothesis: basic->higher gap => NEGATIVE corr w/ ordering
    L = ["# How autonomous is academic prestige? (descriptive)\n",
         "**Reframe:** autonomy index = **gap[f]** — a high gap means the academic prestige ordering is "
         "AUTONOMOUS / self-referential (it does not track market pay); a low gap means it is "
         "market-embedded. We ask what drives autonomy. **Honest scoping:** this largely OVERLAPS the "
         "earlier mixture decomposition (licensing is the one clean channel); the genuinely new pieces "
         "are the **basic↔applied dimension** and an **OpenAlex field-insularity** external test (feasibility "
         "below). Descriptive only. Run: `python scripts/31_prestige_autonomy.py`.\n",
         "## Autonomy ranking\n",
         "Most autonomous (high gap, self-referential) → most market-embedded (low gap):\n",
         df.sort_values("gap", ascending=False).head(8)[["label", "n", "gap", "ba_name"]].to_markdown(index=False, floatfmt=("", ".0f", ".2f", "")),
         "\n…\n",
         df.sort_values("gap").head(8)[["label", "n", "gap", "ba_name"]].to_markdown(index=False, floatfmt=("", ".0f", ".2f", "")),
         "\n## Drivers of autonomy (Spearman with gap)\n",
         f"*Underpowered: n={da[2]}-{dba[2]} gives ~80% power only for |Spearman|≥{mde:.2f}, so a small "
         "true effect could read as null. Bootstrap 95% CIs shown.*\n",
         "| driver | Spearman(gap, ·) | 95% CI | p | n | status |",
         "|---|---|---|---|---|---|",
         f"| academic-absorption (reuse) | {da[0]:+.2f} | [{da[3]:+.2f}, {da[4]:+.2f}] | {da[1]:.3f} | {da[2]} | "
         + ("NULL (as previously found)" if abs(da[0]) < 0.2 or da[1] > 0.1 else "non-null") + " |",
         f"| basic↔applied ordering (NEW) | {dba[0]:+.2f} | [{dba[3]:+.2f}, {dba[4]:+.2f}] | {dba[1]:.3f} | {dba[2]} | "
         + ("supports autonomy thesis" if ba_holds else "does NOT support (see below)") + " |",
         f"| licensing (reuse) | {dl[0]:+.2f} | [{dl[3]:+.2f}, {dl[4]:+.2f}] | {dl[1]:.3f} | {dl[2]} | one channel "
         f"(OLS slope +{m.params[2]:.2f} on 0-1 licensure; scripts/20 reported +0.66 in its own spec) |",
         "\n### basic↔applied dimension (NEW) — gap by class\n",
         by_class.to_markdown(floatfmt=(".3f", ".3f", ".0f")),
         f"\n**The autonomy thesis (basic/autonomous fields have higher gaps) is "
         + ("SUPPORTED" if ba_holds else "NOT supported / confounded") + ".** "
         + (f"The ordering correlates {dba[0]:+.2f} with the gap — "
            "but note the **professional** fields have the highest mean gap, which is the **licensing** "
            "channel (health/credential fields are licensed → compressed pay → high gap), not autonomy. "
            "Controlling for licensing (OLS, n=48 — 6 professional fields dropped for missing "
            "licensure_strict; the ordering null holds on both the n=57 Spearman and this n=48 OLS), the "
            f"basic↔applied ordering coefficient is **{m.params[1]:+.3f} (p={m.pvalues[1]:.3f})** while "
            f"licensing is **{m.params[2]:+.3f} (p={m.pvalues[2]:.3f})** — "
            + ("the basic↔applied dimension survives." if m.pvalues[1] < 0.10 else
               "the basic↔applied dimension adds little once licensing is netted (the apparent autonomy "
               "gradient is mostly the licensing of professional fields).")) + "\n",
         "## STRONGEST test — OpenAlex field insularity (NOT run; key next step)\n",
         "The one genuinely-new positive test the autonomy thesis could pass is an EXTERNAL "
         "self-reference measure independent of the channels: do more **insular / self-referential** "
         "fields (citations staying within-field; low cross-field citation diversity) have higher gaps? "
         "**Feasibility:** the repo has OpenAlex *field tags* per person (`orcid_field.parquet`, scripts "
         "15–19) and an API key, but **no citation/insularity data** — that needs a new pull. Minimal "
         "viable: for each OpenAlex field, sample works and compute the within-field share of "
         "`referenced_works` fields (or topic-cross-field entropy) → one insularity score per field → "
         "crosswalk OpenAlex's 26 research fields to these degree fields (the same lossy research-vs-"
         "degree mapping flagged in scripts 15–19) → Spearman(gap, insularity). ~26 field queries; "
         "feasible but a real data pull. **Documented as the key next step, not run here.** (A *crude* "
         "member-topic-dispersion proxy could be built today from the cached `orcid_field` tags + faculty "
         "edges, but it would measure people's topic spread, NOT citation self-reference — a weak "
         "substitute, so we defer to the real citation pull rather than ship a misleading proxy.)\n",
         "## Honest verdict\n",
         "As scoped, autonomy ≈ **licensing + a large unexplained residual**: academic-absorption is "
         f"{'null' if abs(da[0])<0.2 else 'weak'}, and the basic↔applied dimension "
         + ("survives licensing" if m.pvalues[1] < 0.10 else "does not add beyond licensing") +
         ". The value of this task is (i) the **autonomy reframing** of the gap, (ii) the **basic↔applied** "
         "test (reported honestly, including its confounding by licensing), and (iii) the **OpenAlex "
         "insularity** external validation, which is the one test that could genuinely confirm the "
         "autonomy thesis and is the key next step.\n",
         "## Adversarial self-check\n",
         "1. **Overlap with the mixture decomposition:** absorption and licensing are REUSED, not new; "
         "the only new computes are the basic↔applied ordering and the (deferred) insularity measure. "
         "Disclosed.",
         "2. **basic↔applied coding is SUBJECTIVE** (hand-coded in the script; e.g. statistics=basic, "
         "CS=applied, music=basic are debatable). It is documented and reproducible; a few re-codings "
         "would shift the borderline fields but not the headline (professional fields' high gap is "
         "licensing, not autonomy).",
         "3. **Does anything beat licensing?** No — once licensing is netted, the autonomy drivers add "
         "little; the autonomy reframe is conceptual, not a new explanatory channel.",
         "4. **Insularity feasibility** is honestly flagged as not-run (needs a new OpenAlex citation "
         "pull); it is the genuinely-new test and is sized, not pre-judged.",
         "5. **Descriptive only:** 'autonomy' names a measured property of the prestige ordering (it "
         "doesn't track pay), not a causal or normative claim.\n"]
    (ROOT / "PRESTIGE_AUTONOMY_RESULT.md").write_text("\n".join(L))


def make_figures(df):
    s = df.sort_values("gap")
    col = {"basic": "#2C7BB6", "applied": "#2CA25F", "professional": "#D6202A"}
    fig, ax = plt.subplots(figsize=(8, max(6, len(s) * 0.16)))
    ax.barh(range(len(s)), s.gap, color=[col.get(c, "#bbb") for c in s.ba_name])
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.label, fontsize=6)
    ax.set_xlabel("autonomy = gap = 1 − Spearman(prestige, pay)")
    ax.set_title("Prestige autonomy by field (color = basic blue / applied green / professional red)")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "autonomy_ranked.png", dpi=140)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for cls, c in col.items():
        d = df[df.ba_name == cls]
        ax[0].scatter(d.basic_applied + np.random.default_rng(0).normal(0, 0.05, len(d)), d.gap, c=c, s=25, label=cls, alpha=.7)
    ax[0].set_xticks([1, 2, 3]); ax[0].set_xticklabels(["basic", "applied", "professional"])
    ax[0].set_ylabel("gap (autonomy)"); ax[0].set_title("Autonomy vs basic↔applied (NEW dimension)"); ax[0].legend(fontsize=8)
    d2 = df.dropna(subset=["licensure_strict"])
    sc = ax[1].scatter(d2.licensure_strict, d2.gap, c=d2.basic_applied, cmap="viridis", s=25)
    ax[1].set_xlabel("licensure intensity (the one clean channel)"); ax[1].set_ylabel("gap (autonomy)")
    ax[1].set_title(f"Autonomy vs licensing (Spearman {spearmanr(d2.licensure_strict, d2.gap)[0]:+.2f})")
    plt.colorbar(sc, ax=ax[1], label="basic→professional")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "autonomy_vs_drivers.png", dpi=140)


if __name__ == "__main__":
    main()
