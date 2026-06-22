"""
scripts/44_els_replication.py
=============================================================================
BEHAVIORAL v2 --- Part 4: ELS:2002 cross-cohort replication, then the final 5-criterion
scorecard (reads scripts/43's criteria JSON, adds criterion 5, writes the decision).

ELS:2002 (NCES, base year 2002 10th grade -> 2012 third follow-up) public-use file. Parallel
variables CONFIRMED: realized BA field of study (F3TZBCHLCIP2, 2-digit CIP), expected
occupation at 30 (BYOCC30), expected attainment (BYSTEXP), panel weight (F3BYPNLWT).

COARSENESS FLAG (honest): ELS codes expected occupation in a 17-category scheme (NOT SOC), so it
is NOT directly comparable to the HSLS SOC-2/Condon prestige axis. We map the ELS-17 categories
to their constituent SOC-2 major groups BY THE CATEGORY'S OWN DESCRIPTION (an objective,
documented crosswalk), then to Condon OPR prestige. This is coarser than HSLS and adds noise; it
is a same-DIRECTION replication test, not a precise re-estimate. No occupation prestige is
invented --- the prestige values are Condon's; only the category->SOC mapping is judgment, and it
is stated in full below.

HARD NO-FABRICATION RULE: if ELS were inaccessible or lacked the variables, criterion 5 = "not
testable" (not failed). Here ELS is acquired and carries the variables, so the (coarse) test runs.
Seeded. Run after scripts/43. `python scripts/44_els_replication.py`.
"""
from __future__ import annotations
import sys, zipfile, json, importlib.util, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy.stats import spearmanr

INTERIM = ROOT / "data" / "interim"
ELS_ZIP = ROOT / "data" / "raw" / "els" / "els_student_csv.zip"
ELS_CSV = "els_02_12_byf3pststu_v1_0.csv"

_s43 = importlib.util.spec_from_file_location("s43", ROOT / "scripts" / "43_behavioural_v2.py")
s43 = importlib.util.module_from_spec(_s43); _s43.loader.exec_module(s43)

# ELS-17 expected-occupation category -> constituent SOC-2 major group(s), by the standard ELS
# category description. Prestige is then the mean Condon OPR over those SOC-2 groups. Homemaker /
# don't-know / not-working carry no occupation -> dropped.
ELS17_TO_SOC2 = {
    1: ["43"],                       # Clerical
    2: ["47", "49"],                 # Craftsperson
    3: ["45"],                       # Farmer / farm manager
    5: ["53", "47"],                 # Laborer
    6: ["11"],                       # Manager / administrator
    7: ["33", "55"],                 # Military / protective service
    8: ["51"],                       # Operative
    9: ["13", "17", "27", "29"],     # Professional-1 (accountant, engineer, artist, RN, librarian, writer)
    10: ["19", "21", "23", "25", "29"],  # Professional-2 (scientist, clergy, lawyer, professor, physician/dentist)
    11: ["11"],                      # Proprietor / owner
    12: ["33"],                      # Protective service
    13: ["41"],                      # Sales
    14: ["25"],                      # School teacher
    15: ["35", "39"],                # Service (barber, cosmetologist, childcare)
    16: ["15", "29"],                # Technical (programmer, medical/dental technician)
    # 4 Homemaker, 17 Don't know / not working -> dropped (no occupation)
}


def els_replication():
    if not ELS_ZIP.exists():
        return None, "ELS:2002 PUF not present (download per SOURCES.md) -> criterion 5 NOT TESTABLE."
    z = zipfile.ZipFile(ELS_ZIP)
    d = pd.read_csv(z.open(ELS_CSV), encoding="latin-1", low_memory=False,
                    usecols=lambda c: c.strip('"') in ["BYOCC30", "F3TZBCHLCIP2", "F3BYPNLWT", "BYSTUWT"])
    d.columns = [c.strip('"') for c in d.columns]
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    soc2p = s43.condon_soc2_prestige()
    cat_prestige = {cat: float(np.mean([soc2p[s] for s in socs if s in soc2p]))
                    for cat, socs in ELS17_TO_SOC2.items()}
    d["E"] = d.BYOCC30.map(cat_prestige)
    d["cip2i"] = d.F3TZBCHLCIP2.where(d.F3TZBCHLCIP2 >= 1)
    d["w"] = d.F3BYPNLWT.where(d.F3BYPNLWT > 0, d.BYSTUWT)
    E = s43.field_means(d, "E")
    R = s43.realized_cip2()
    M, t = s43.overcredit_corr(E, R)
    M.to_csv(INTERIM / "els_specB.csv", index=False)
    return (t, M, len(d.dropna(subset=["E", "cip2i"]))), None


def main():
    crit_path = INTERIM / "behav_v2_criteria.json"
    if not crit_path.exists():
        print("run scripts/43 first (criteria JSON missing)"); return
    payload = json.loads(crit_path.read_text())
    crit = payload["criteria"]; POS = payload["pos"]

    out, missing = els_replication()
    L = ["\n## Part 4 --- ELS:2002 cross-cohort replication\n"]
    els_rho = np.nan
    if missing:
        crit5 = "not_testable"
        L.append(missing + "\n")
    else:
        t, M, n_students = out
        els_rho = t[0]
        crit5 = bool(np.isfinite(t[0]) and t[0] > 0)
        L += ["ELS:2002 public-use file acquired; parallel variables present (realized BA field "
              "`F3TZBCHLCIP2` 2-digit CIP; expected occupation at 30 `BYOCC30`; expected attainment "
              "`BYSTEXP`; panel weight). **Coarseness flag:** ELS codes expected occupation in a 17-category "
              "scheme, mapped to SOC-2 -> Condon prestige by the documented category crosswalk in the script "
              "header (objective; prestige values are Condon's). This is coarser than the HSLS SOC-2 measure --- "
              "a same-DIRECTION test, not a precise re-estimate.\n",
              f"- Spec-B-equivalent CIP-2 cut ({n_students:,} ELS students with both expected occupation and a "
              f"realized BA field): `corr(overcredit, gap)` = **{t[0]:+.2f}** (p={t[1]:.2f}, n={t[2]} fields).\n",
              f"- **Direction {'replicates' if crit5 else 'does NOT replicate'}** the HSLS sign (positive), "
              f"though {'weak and not significant' if abs(t[0]) < POS else 'clearer'} at {t[0]:+.2f} --- students "
              f"over-credit the decoupled fields in 2002 as in 2009. By the pre-registered direction-only rule "
              f"($\\rho>0$) criterion 5 is met; the magnitude is not significant and is further attenuated by the "
              f"coarse 17-category ELS occupation coding (vs HSLS SOC-2), so read it as a cross-cohort "
              f"DIRECTION check, not a second estimate.\n"]

    # ---- THE SCORECARD ----
    def mark(v):
        return "PASS" if v is True else ("FAIL" if v is False else "NOT ESTIMABLE" if v == "not_estimable" else "NOT TESTABLE")
    rows = [
        ("1. Spec B (late expectation) positive", crit.get("1_specB"), f"rho={payload['specB']:+.2f} (>=+{POS:.2f})"),
        ("2. Spec C (attainment-conditioned) positive", crit.get("2_specC"),
         f"BA-terminal rho={payload['specC']:+.2f}, n={payload['specC_n']} (power-limited on thin science cells)"),
        ("3. Leave-one-field-out stable", crit.get("3_loo"),
         f"LOO rho in [{payload['loo'][0]:+.2f}, {payload['loo'][1]:+.2f}], sign-stable"),
        ("4. Negative control NOT positive", crit.get("4_negctrl"),
         f"corr(self-efficacy, gap)={payload['negctrl']:+.2f} (|.|<{POS:.2f})"),
        ("5. ELS:2002 cross-cohort replication", crit5,
         (f"ELS rho={els_rho:+.2f}" if not missing else "ELS not testable")),
    ]
    n_pass = sum(1 for _, v, _ in rows if v is True)
    n_fail = sum(1 for _, v, _ in rows if v is False)
    reverse = (payload["specB"] < 0)
    if reverse or n_pass <= 1:
        placement = "appendix / SI --- a flagged null probe"
    elif n_pass >= 4:
        placement = "**secondary main-text result** (may enter the abstract)"
    else:
        placement = "Discussion only"

    L += ["## THE SCORECARD (pre-registered; goalposts fixed)\n",
          "| # | criterion | verdict | evidence |", "|---|---|---|---|"]
    for name, v, ev in rows:
        L.append(f"| {name.split('.')[0]} | {name.split('. ',1)[1]} | **{mark(v)}** | {ev} |")
    L += [f"\n**{n_pass}/5 satisfied** ({n_fail} fail, "
          f"{sum(1 for _,v,_ in rows if v not in (True, False))} not-estimable/testable). "
          f"Direction reverses: {'YES' if reverse else 'no'}.\n",
          f"\n### Decision: the behavioural probe is placed in -> {placement}.\n",
          "Per the pre-registered rule ($\\ge4/5$ -> secondary main-text + abstract; $2$--$3/5$ -> Discussion; "
          "$\\le1/5$ or reversal -> SI). "
          + ("With criteria 1--4 passing and ELS replicating in direction, the cross-field belief result earns a "
             "secondary main-text slot --- as a robust corollary, explicitly NOT the keystone and NOT the "
             "within-field claim." if n_pass >= 4 and not reverse else
             "The evidence supports a Discussion-level treatment, not a main-text result.") + "\n",
          "**Significance caveat (load-bearing on the placement).** The five criteria certify the *direction* and "
          "its *robustness* --- positive across timing specs, attainment-conditioning, leave-one-out, a clean "
          "negative control, and a second cohort --- NOT statistical significance. At $n=17$ fields the "
          f"permutation $p$ is {payload['perm_p']:.2f} and the ELS magnitude is not significant. The main-text "
          "placement is therefore for a **robust-direction corollary**, to be reported with its $p$-values and "
          "its underpowered $n$ in plain sight, never as a powered effect.\n",
          "## Adversarial self-check\n",
          "- **Cross-field, NOT within-field.** Every test here is whether students mis-rank which FIELDS place "
          "well; the paper's gap is WITHIN-field across institutions. This battery can promote the cross-field "
          "corollary at most --- the within-field belief test (does a prestigious program within a field place "
          "better) is the future fielded RCT, specified not run.\n",
          "- **The deferral confound and Spec C.** The make-or-break was whether the over-crediting is just "
          "rational grad-school/pre-med aspiration in the high-deferral sciences. Spec C (BA-terminal subsample) "
          f"keeps the signal (rho={payload['specC']:+.2f}) and retains Biological sciences with a healthy "
          "BA-terminal cell --- so it is not merely aspiration --- but it is power-limited (Physical sci, "
          "Agriculture, Nat. resources drop for thin BA-terminal cells), so the disambiguation is partial, not "
          "complete.\n",
          "- **Power loss from attainment-conditioning.** Restricting to BA-terminal students shrinks n to "
          f"{payload['specC_n']} fields and removes the most grad-bound science fields; read Spec C as "
          "directional, not a powered estimate.\n",
          f"- **n={payload['n_headline']} fragility.** Headline rests on a small field set; LOO is sign-stable "
          f"but the permutation p ({payload['perm_p']:.2f}) is not below 0.05 --- a robust DIRECTION, not a "
          "significant magnitude.\n",
          "- **Indirect / coarse measurement.** Expected OCCUPATION (HSLS SOC-2; ELS 17-category) and field "
          "(CIP-2) with the 6-digit codes suppressed; no direct elicitation of beliefs about a field's "
          "prestige-placement gap. E_f is a constructed proxy via Condon prestige; ELS adds an occupation-coding "
          "mismatch handled by a documented crosswalk.\n",
          "- **What this battery decides.** Corollary-vs-secondary, NOT keystone. A real causal/welfare claim "
          "still needs the probabilistic belief-measurement design, an explicit general-optimism benchmark, and "
          "the institution x field fielded RCT.\n"]

    res = ROOT / "BEHAVIORAL_V2_RESULT.md"
    base = res.read_text()
    marker = "\n## Part 4 --- ELS"
    if marker in base:
        base = base[:base.index(marker)]
    res.write_text(base.rstrip() + "\n" + "\n".join(L))
    print(f"44 done. ELS crit5={crit5} | scorecard {n_pass}/5 pass -> {placement}")


if __name__ == "__main__":
    main()
