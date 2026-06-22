"""
scripts/42_behavioural_feasibility.py
=============================================================================
LIGHT feasibility for the behavioural half of the combined paper --- NOT the full analysis
(that is joint work with a subjective-expectations co-author and, for the within-field claim,
a fielded RCT). Two questions: (1) do the public expectation panels carry the needed variables,
and (2) a rough first cut --- do students over-credit the placement of exactly the fields the
paper identifies as DECOUPLED?

HARD NO-FABRICATION RULE: if a dataset/variable is missing, the branch stops and reports what is
missing; nothing is synthesised.

Grain (stated honestly): HSLS supports a FIELD-LEVEL (cross-field) belief test --- do students
mis-rank which FIELDS place well --- which is RELATED TO but NOT the WITHIN-field gap (does a
prestigious program within a field place better). The within-field belief test needs
institution x field belief data -> the future fielded RCT (the co-author's domain), specified
not run.

Data: HSLS:09 public-use file (NCES, Base Year through 2016 Second Follow-up), already downloaded.
Reuses the Condon/Hughes O*NET occupational-prestige scale and the project field/CIP-2 taxonomy +
realized placement (er_dimensions.csv) and gap (expanded66_gap_map.csv). Descriptive,
outcome-agnostic, seeded. Run: `python scripts/42_behavioural_feasibility.py`.
"""
from __future__ import annotations
import sys, zipfile, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

SEED = 7
INTERIM = ROOT / "data" / "interim"
HSLS_ZIP = ROOT / "data" / "raw" / "hsls" / "hsls_2016_csv.zip"
HSLS_CSV = "hsls_16_student_v1_0.csv"
CONDON = ROOT / "data" / "raw" / "prestige" / "condon_OccupationalPrestigeRatings.tab"

# HSLS variables (confirmed present in the PUF):
#   S4FIELD2 / S3FIELD2  postsecondary field of study, 2-digit CIP (the field grouping)
#   X1STU30OCC2          base-year expected occupation at age 30, 2-digit SOC major group
#   S4OCC30EARN          expected earnings at age 30 (2016 wave)
#   X1STUEDEXPCT         expected educational attainment (ordinal)
#   W4W1STU              base-year -> 2016 panel analytic weight
HSLS_VARS = ["STU_ID", "S4FIELD2", "S3FIELD2", "X1STU30OCC2", "X4STU30OCC2",
             "S4OCC30EARN", "S2OCC30EARN", "X1STUEDEXPCT", "W4W1STU", "W1STUDENT"]


def load_hsls():
    z = zipfile.ZipFile(HSLS_ZIP)
    d = pd.read_csv(z.open(HSLS_CSV), usecols=lambda c: c.strip('"') in HSLS_VARS,
                    encoding="latin-1", low_memory=False)
    d.columns = [c.strip('"') for c in d.columns]
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def condon_soc2_prestige():
    """Mean Condon OPR occupational prestige by SOC-2010 2-digit major group."""
    c = pd.read_csv(CONDON, sep="\t")
    c["soc2"] = c["ONET SOC 2018 Code"].astype(str).str.replace("-", "", regex=False).str[:2]
    c = c[c.soc2.str.match(r"^\d{2}$")]
    return c.groupby("soc2")["OPR Job Rating"].mean().to_dict()


def wmean(v, w):
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.average(v[m], weights=w[m])) if m.sum() else np.nan


def main():
    L = ["# Behavioural-half feasibility (LIGHT): do students over-credit the decoupled fields?\n",
         "Viability check feeding the co-author conversation --- NOT the full behavioural analysis. "
         "No-fabrication run. Seeded; `python scripts/42_behavioural_feasibility.py`.\n"]

    # ---- variable availability ----
    if not HSLS_ZIP.exists():
        L += ["## STOP --- HSLS:09 not acquired\n",
              "Access path: NCES public-use file, `https://nces.ed.gov/EDAT/Data/Zip/HSLS_2016_v1_0_CSV_Datasets.zip` "
              "(or the EdSurvey R package / ICPSR NADAC 36423). Re-run once present. No result fabricated.\n"]
        (ROOT / "BEHAVIORAL_FEASIBILITY_RESULT.md").write_text("\n".join(L))
        print("stopped: HSLS not present"); return
    d = load_hsls()
    avail = {c: int((d[c] >= 0).sum()) for c in d.columns if c != "STU_ID"}
    L += ["## Variable availability (codebook check)\n",
          "HSLS:09 public-use file (NCES, Base Year 2009 through Second Follow-up 2016), "
          f"{len(d):,} students. Confirmed variables and non-missing counts:\n",
          "| variable | meaning | n valid |",
          "|---|---|---|",
          f"| `S4FIELD2` / `S3FIELD2` | postsecondary field of study (2-digit CIP) | {avail.get('S4FIELD2',0):,} / {avail.get('S3FIELD2',0):,} |",
          f"| `X1STU30OCC2` | base-year expected occupation at 30 (2-digit SOC) | {avail.get('X1STU30OCC2',0):,} |",
          f"| `S4OCC30EARN` / `S2OCC30EARN` | expected earnings at 30 (\\$) | {avail.get('S4OCC30EARN',0):,} / {avail.get('S2OCC30EARN',0):,} |",
          f"| `X1STUEDEXPCT` | expected educational attainment (ordinal) | {avail.get('X1STUEDEXPCT',0):,} |",
          f"| `W4W1STU` | base-year->2016 panel weight | {avail.get('W4W1STU',0):,} |",
          "\n**The 6-digit occupation/field codes (`*OCC6`, `*FIELD6`) are disclosure-suppressed in the public-use "
          "file** --- only 2-digit SOC (occupation) and 2-digit CIP (field) are released, so the cut is at the "
          "**CIP-2 grain** (which is exactly the project's discipline-cluster / integrated--decoupled typology grain). "
          "NLSY97 (secondary) is **access-gated**: variables must be hand-selected and extracted via the BLS NLS "
          "Investigator (not one-click); flagged as a second step, not run here.\n"]

    # ---- expected outcome E_f (field level, CIP-2) ----
    soc2p = condon_soc2_prestige()
    d["exp_occ_prestige"] = d.X1STU30OCC2.apply(
        lambda x: soc2p.get(f"{int(x):02d}", np.nan) if np.isfinite(x) and 11 <= x <= 55 else np.nan)
    d["exp_earn"] = d.S4OCC30EARN.where(d.S4OCC30EARN.between(1, 250000))
    d["cip2"] = d.S4FIELD2.where(d.S4FIELD2 >= 1).fillna(d.S3FIELD2.where(d.S3FIELD2 >= 1))
    d["w"] = d.W4W1STU.where(d.W4W1STU > 0, d.W1STUDENT)
    df = d.dropna(subset=["cip2"]).copy()
    df["cip2i"] = df.cip2.astype(int)
    E = []
    for cip, g in df.groupby("cip2i"):
        if len(g) < 30:                       # need enough students for a field mean
            continue
        E.append(dict(cip2=int(cip), n_students=len(g),
                      E_occ_prestige=wmean(g.exp_occ_prestige.values, g.w.values),
                      E_earn=wmean(g.exp_earn.values, g.w.values)))
    E = pd.DataFrame(E)

    # ---- realized placement R_f + gap (project, CIP-2) ----
    er = pd.read_csv(INTERIM / "er_dimensions.csv").dropna(subset=["cip2"]).copy()
    er["cip2i"] = er.cip2.astype(int)
    R = er.groupby("cip2i").agg(
        R_occ_prestige=("occ_prestige_opr", "mean"), R_earn=("earn_standing", "mean"),
        gap=("gap", "mean"), n_fields=("field", "size")).reset_index().rename(columns={"cip2i": "cip2"})

    M = E.merge(R, on="cip2", how="inner")
    CIP2_LAB = {11: "Computer/Info", 13: "Education", 14: "Engineering", 16: "Languages", 23: "English",
                26: "Biological sci", 27: "Mathematics", 40: "Physical sci", 42: "Psychology",
                44: "Public admin", 45: "Social sci", 50: "Arts", 51: "Health", 52: "Business",
                54: "History", 9: "Communication", 3: "Nat. resources", 1: "Agriculture", 15: "Eng. tech",
                4: "Architecture", 19: "Family/consumer sci", 30: "Interdisciplinary",
                31: "Parks/rec/fitness", 38: "Philosophy/religion", 39: "Theology", 22: "Legal", 24: "Liberal arts"}
    M["label"] = M.cip2.map(CIP2_LAB).fillna(M.cip2.astype(str))

    if len(M) < 6:
        L += [f"\n## Rough cut not run --- only {len(M)} CIP-2 fields overlap HSLS x project (need >=6).\n",
              "Variables are present (above) but the field overlap is too thin for even a signal check; "
              "no correlation fabricated.\n"]
        (ROOT / "BEHAVIORAL_FEASIBILITY_RESULT.md").write_text("\n".join(L))
        print(f"stopped: only {len(M)} overlapping fields"); return

    # ---- the rough cut (RANKS, to net general optimism) ----
    M["rank_E"] = M.E_occ_prestige.rank()
    M["rank_R"] = M.R_occ_prestige.rank()
    M["overcredit"] = M.rank_E - M.rank_R        # + => students rank field HIGHER than realized
    # earnings variant
    M["rank_Ee"] = M.E_earn.rank(); M["rank_Re"] = M.R_earn.rank()
    M["overcredit_earn"] = M.rank_Ee - M.rank_Re

    def sp(a, b):                                         # Spearman on finite pairs; (rho, p, n)
        a = np.asarray(a, float); b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 5:
            return (np.nan, np.nan, int(m.sum()))
        r, p = spearmanr(a[m], b[m]); return (r, p, int(m.sum()))
    t1 = sp(M.overcredit, M.gap)                          # core: over-credit concentrated in decoupled?
    t1e = sp(M.overcredit_earn, M.gap)
    t2 = sp(M.E_occ_prestige, M.R_occ_prestige)           # do expectations track realized placement?
    t2e = sp(M.E_earn, M.R_earn)
    M.to_csv(INTERIM / "behavioural_field_cut.csv", index=False)
    top = M.sort_values("overcredit", ascending=False).head(5)

    L += ["## The rough cut (CIP-2 field level; ranks, to net general optimism)\n",
          f"Merged on CIP-2: **{len(M)} fields** with both a HSLS expected outcome (>=30 students/field) and the "
          "project's realized placement + gap. `overcredit = rank(E_f) - rank(R_f)`: positive = students rank a "
          "field's placement HIGHER than it realizes.\n",
          M.sort_values("gap")[["label", "n_students", "E_occ_prestige", "R_occ_prestige",
                                 "overcredit", "gap"]].to_markdown(index=False, floatfmt=("", ".0f", ".1f", ".1f", "+.1f", ".2f")),
          "",
          "### Test 1 (core) --- is over-crediting concentrated in the DECOUPLED (high-gap) fields?\n",
          f"- `corr(overcredit, gap)` = **Spearman {t1[0]:+.2f}** (p={t1[1]:.2f}, n={t1[2]}); "
          f"expected-earnings variant {t1e[0]:+.2f} (p={t1e[1]:.2f}, n={t1e[2]}).\n",
          "### Test 2 (cross-check) --- do expectations track realized placement only weakly?\n",
          f"- `corr(E_f, R_f)` occupational prestige = Spearman {t2[0]:+.2f} (p={t2[1]:.2f}); "
          f"earnings {t2e[0]:+.2f} (p={t2e[1]:.2f}). Weak-to-moderate tracking is the ``fooled-by-prestige'' "
          f"signature (expectations not strongly aligned with where fields actually place).\n",
          f"\n**Most over-credited fields (rank E >> rank R):** "
          f"{', '.join(f'{r.label} (gap {r.gap:.2f}, +{r.overcredit:.0f})' for _, r in top.iterrows())}.\n"]

    # ---- verdict ----
    signal = (np.isfinite(t1[0]) and t1[0] > 0.2) or (np.isfinite(t1e[0]) and t1e[0] > 0.2)
    verdict = ("VIABLE --- a directional signal is present" if signal else
               "UNCERTAIN --- no clear field-level signal on this coarse cut")
    L += ["## Verdict\n",
          f"**The behavioural half is {verdict}** for development with a subjective-expectations co-author. The "
          "variables exist in public data (HSLS:09), the crosswalks resolve at CIP-2, and the rough cut "
          f"{'shows the hypothesised direction (students over-credit the decoupled fields)' if signal else 'is inconclusive at the coarse CIP-2 / 2-digit-SOC grain'}. "
          "This is a **signal check, not a clean estimate**. The full version needs: (i) a rigorous "
          "belief-measurement design (probabilistic elicitation of expected placement by field x tier, not a "
          "single expected-occupation code); (ii) an explicit **general-optimism benchmark** (students over-expect "
          "on everything --- handled here with ranks, but a calibrated design is better); and (iii) for the "
          "*within-field* claim --- does a prestigious program within a field place better, the true analog of the "
          "paper's gap --- an **institution x field fielded RCT** (the co-author's domain), which public panels "
          "cannot deliver.\n",
          "## Adversarial self-check\n",
          "- **General optimism handled by ranks.** Students over-expect on almost everything; levels would just "
          "show uniform optimism. We use rank(E)-rank(R), so only RELATIVE over-crediting (which fields are ranked "
          "too high vs realized) enters --- the right object for ``fooled-by-prestige.''\n",
          "- **FIELD-LEVEL, not within-field.** This tests whether students mis-rank which FIELDS place well; the "
          "paper's gap is WITHIN-field across institutions. The two are related (both are prestige!=placement) but "
          "distinct; the within-field belief test is the future RCT, specified not run.\n",
          "- **HSLS realized outcomes are early/incomplete**, so R_f here uses the project's mature ACS/Scorecard "
          "realized placement (occ-prestige / earnings) as the benchmark, not HSLS's own (young) follow-up "
          "outcomes --- a deliberate choice, flagged.\n",
          "- **Indirect measurement.** HSLS gives expected OCCUPATION (2-digit SOC) and expected EARNINGS, not a "
          "direct belief about a field's prestige--placement gap. E_f is a constructed proxy (expected "
          "occupational prestige via Condon), and the SOC-2 / CIP-2 grain is coarse --- a signal check, not the "
          "designed instrument.\n",
          f"- **n and crosswalk losses.** Only {len(M)} CIP-2 fields enter (others lack >=30 HSLS students or a "
          "project realized value); the 6-digit codes are suppressed; SOC-2 occupational prestige averages over a "
          "major group. Small n + coarse grain => underpowered; read as direction, not magnitude.\n"]

    (ROOT / "BEHAVIORAL_FEASIBILITY_RESULT.md").write_text("\n".join(L))
    print(f"done. fields={len(M)} | Test1 corr(overcredit,gap)={t1[0]:+.2f} (earn {t1e[0]:+.2f}) | "
          f"Test2 corr(E,R) prestige={t2[0]:+.2f} earn={t2e[0]:+.2f} | verdict: {verdict.split(' ---')[0]}")


if __name__ == "__main__":
    main()
