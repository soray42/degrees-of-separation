"""Direction ① — career-time coupling: does coupling_f(h) = Spearman(prestige, earnings@h)
change with years since graduation, and does its SHAPE differ by field type?

Employer-learning signatures (Farber-Gibbons 1996; Altonji-Pierret 2001; Lange 2007):
  - prestige = SIGNAL           -> coupling DECAYS with tenure (employers learn true ability)
  - prestige = human capital/net -> coupling FLAT or RISES (elite-tail compounding)
  - COMPRESSION fields           -> coupling pinned at ~0 at every horizon (nothing to learn)

Two earnings panels, deliberately:
  * Scorecard FoS  -- broad coverage, but 1YR/4YR/5YR of the SAME release are DIFFERENT COHORTS,
                      so its cross-horizon comparison conflates career time with cohort/calendar
                      time -> the COVERAGE version, read as suggestive only.
  * PSEO (pseoe)   -- pooled-cohort y1/y5/y10 track the SAME graduates over time -> the CLEAN
                      career-time version; thin (only ~public partners overlap the prestige set).

Prestige (AR) is held fixed (Wapman pooled SpringRank) across horizons, so any movement in
coupling is a placement-side (earnings) effect. Reuses src.gap.compute_gap_map verbatim.

Reproducible: SEED fixed. Run: python scripts/52_career_time_coupling.py
Outputs: data/interim/career_time_coupling.csv, outputs/figures/career_time_coupling.png,
         CAREER_TIME_COUPLING_RESULT.md (local; root *.md is gitignored).
"""
from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.load_ar import load_ar_wapman
from src.load_er import load_er_scorecard, load_er_pseo
from src.gap import compute_gap_map
from src.crosswalks import fields as F

SEED = 52
NMIN = 15                       # min institutions in a (field,horizon) cell for a stable rho
B_GAP = 250                     # bootstrap draws inside compute_gap (reliability side-info)
ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(SEED)

# FIELDS66 + label map, exactly as scripts/28/32 build them
_spec = importlib.util.spec_from_file_location("s28", ROOT / "scripts" / "28_field_vs_generic_prestige.py")
_s28 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s28)
FIELDS66, LAB = _s28.FIELDS66, _s28.LAB

# horizon -> (earn_col, count_col) for Scorecard; horizon label -> years
SC_HORIZONS = {
    "1YR": ("EARN_MDN_1YR", "EARN_COUNT_WNE_1YR"),
    "4YR": ("EARN_MDN_4YR", "EARN_COUNT_WNE_4YR"),
    "5YR": ("EARN_MDN_5YR", "EARN_COUNT_WNE_5YR"),
}
SC_YEARS = {"1YR": 1, "4YR": 4, "5YR": 5}
PS_HORIZONS = ["y1", "y5", "y10"]
PS_YEARS = {"y1": 1, "y5": 5, "y10": 10}


def coupling_panel(ar, load_er_fn, horizons, level, years) -> pd.DataFrame:
    """coupling_f(h) for each horizon -> long table field|horizon|years|coupling|n|reliable."""
    rows = []
    for h in horizons:
        er = load_er_fn(h)
        gm = compute_gap_map(ar, er, level, B=B_GAP, seed=SEED)
        for _, r in gm.iterrows():
            rows.append(dict(field=r["field"], horizon=h, years=years[h],
                             coupling=r["spearman"], n=int(r["n_institutions"]),
                             reliable=bool(r["reliable_flag"])))
    return pd.DataFrame(rows)


def classify(baseline: pd.DataFrame) -> pd.DataFrame:
    """field_type from the 4YR (baseline) Scorecard coupling; annotate ex-ante license."""
    b = baseline.set_index("field")
    out = []
    for f in b.index:
        rho, n, rel = b.loc[f, "coupling"], b.loc[f, "n"], b.loc[f, "reliable"]
        licensed = f in F.LICENSED_FIELDS
        if not rel or n < NMIN or not np.isfinite(rho):
            grp = "unreliable"
        elif rho >= 0.45:
            grp = "integrated"
        elif rho < 0.25:
            grp = "decoupled"
        else:
            grp = "middle"
        out.append(dict(field=f, baseline_rho=rho, baseline_n=n, group=grp, licensed=licensed))
    return pd.DataFrame(out)


def group_slope(long: pd.DataFrame, cls: pd.DataFrame, grp: str, source: str):
    """Per-field OLS slope of coupling on years (cells with n>=NMIN, >=2 horizons), then the
    field-level mean slope with a bootstrap-over-fields 95% CI. Returns (mean, lo, hi, k, detail)."""
    fields = cls[cls.group == grp].field.tolist()
    sub = long[(long.source == source) & (long.field.isin(fields)) &
               (long.n >= NMIN) & long.coupling.notna()]
    slopes = {}
    for f, g in sub.groupby("field"):
        if g.years.nunique() < 2:
            continue
        slopes[f] = np.polyfit(g.years.to_numpy(float), g.coupling.to_numpy(float), 1)[0]
    if not slopes:
        return np.nan, np.nan, np.nan, 0, slopes
    s = np.array(list(slopes.values()))
    boot = [np.mean(rng.choice(s, size=len(s), replace=True)) for _ in range(2000)]
    return float(s.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(s), slopes


def main():
    ar = load_ar_wapman(fields=FIELDS66)

    sc = coupling_panel(ar, lambda h: load_er_scorecard(
        "undergrad", earn_col=SC_HORIZONS[h][0], count_col=SC_HORIZONS[h][1], fields=FIELDS66),
        SC_HORIZONS.keys(), "undergrad", SC_YEARS)
    sc["source"] = "scorecard"

    ps = coupling_panel(ar, lambda h: load_er_pseo("undergrad", h),
                        PS_HORIZONS, "undergrad_pseo", PS_YEARS)
    ps["source"] = "pseo"

    long = pd.concat([sc, ps], ignore_index=True)
    long["label"] = long.field.map(LAB).fillna(long.field)
    long.to_csv(ROOT / "data" / "interim" / "career_time_coupling.csv", index=False)

    # field types from the 4YR Scorecard baseline
    baseline = sc[sc.horizon == "4YR"][["field", "coupling", "n", "reliable"]]
    cls = classify(baseline)

    # ---- shape analysis: group-mean coupling(h) + group-mean per-field slope ----
    report = {}
    for source, years_map in (("scorecard", SC_YEARS), ("pseo", PS_YEARS)):
        gm_traj, slopes = {}, {}
        for grp in ("integrated", "decoupled", "middle"):
            fields = cls[cls.group == grp].field.tolist()
            sub = long[(long.source == source) & long.field.isin(fields) & (long.n >= NMIN)]
            traj = sub.groupby("years").coupling.agg(["mean", "count"])
            gm_traj[grp] = traj
            slopes[grp] = group_slope(long, cls, grp, source)
        report[source] = (gm_traj, slopes)

    # licensed-field trajectories (nursing / comm_disorders = compression; accounting = counterex.)
    lic = {}
    for f in sorted(F.LICENSED_FIELDS):
        for source in ("scorecard", "pseo"):
            sub = long[(long.source == source) & (long.field == f) & (long.n >= NMIN)].sort_values("years")
            if len(sub):
                lic[(f, source)] = sub[["years", "coupling", "n"]].to_dict("records")

    _figure(long, cls, report, LAB)
    _write_result(long, cls, report, lic, LAB)
    _print_summary(cls, report, lic)


def _figure(long, cls, report, LAB):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    colors = {"integrated": "#1b7837", "decoupled": "#b2182b", "middle": "#999999"}
    for ax, source in zip(axes, ("scorecard", "pseo")):
        gm_traj, slopes = report[source]
        for grp in ("integrated", "decoupled"):
            fields = cls[cls.group == grp].field.tolist()
            sub = long[(long.source == source) & long.field.isin(fields) & (long.n >= NMIN)]
            for f, g in sub.groupby("field"):     # faint per-field lines
                g = g.sort_values("years")
                ax.plot(g.years, g.coupling, color=colors[grp], alpha=0.18, lw=0.9)
            traj = gm_traj[grp]
            m, lo, hi, k, _ = slopes[grp]
            ax.plot(traj.index, traj["mean"], color=colors[grp], lw=2.6, marker="o",
                    label=f"{grp} (n={len(fields)}f, slope={m:+.3f}/yr [{lo:+.3f},{hi:+.3f}])")
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_title(f"{source.upper()}  " +
                     ("(coverage; 1/4/5yr = different cohorts)" if source == "scorecard"
                      else "(clean; same-cohort y1/y5/y10)"), fontsize=10)
        ax.set_xlabel("years since graduation")
        ax.legend(fontsize=7.5, loc="best")
    axes[0].set_ylabel("coupling  ρ_f(h) = Spearman(prestige, earnings@h)")
    fig.suptitle("Career-time coupling by field type — employer-learning signatures "
                 "(decay=signal / flat-rise=human-capital / pinned≈0=compression)", fontsize=11)
    fig.tight_layout()
    out = ROOT / "outputs" / "figures" / "career_time_coupling.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[fig] {out}")


def _fmt_traj(traj):
    return ", ".join(f"{int(y)}yr {row['mean']:+.3f}(n={int(row['count'])}f)"
                     for y, row in traj.iterrows())


def _write_result(long, cls, report, lic, LAB):
    L = ["# Direction ① — Career-time coupling (employer-learning signatures)\n",
         "Outcome-agnostic. Prestige (Wapman SpringRank) held fixed across horizons; movement is "
         "placement-side only. `coupling_f(h)=Spearman(prestige,earnings@h)`, cells need "
         f"n≥{NMIN} institutions. **Scorecard** 1/4/5yr = coverage but different cohorts "
         "(confounds career vs calendar time); **PSEO** y1/y5/y10 = clean same-cohort. "
         "Run: `python scripts/52_career_time_coupling.py`.\n",
         "## Field types (from 4YR Scorecard baseline)\n"]
    for grp in ("integrated", "decoupled", "middle", "unreliable"):
        fs = cls[cls.group == grp]
        names = ", ".join(f"{LAB.get(f,f)}{'*' if l else ''}"
                          for f, l in zip(fs.field, fs.licensed))
        L.append(f"- **{grp}** (n={len(fs)}): {names or '—'}")
    L.append("\n_(*) = ex-ante licensed field. Prediction: compression (nursing, comm-disorders) "
             "pinned ≈0 at all horizons; integrated either decays (signalling) or holds/rises "
             "(human capital); accounting is licensed-but-uncompressed → behaves integrated.\n")
    for source in ("scorecard", "pseo"):
        gm_traj, slopes = report[source]
        tag = "COVERAGE (cohort-confounded)" if source == "scorecard" else "CLEAN (same-cohort)"
        L.append(f"## {source.upper()} — {tag}\n")
        L.append("| group | mean coupling trajectory | mean per-field slope /yr [95% CI] | k fields |")
        L.append("|---|---|---|---|")
        for grp in ("integrated", "decoupled", "middle"):
            m, lo, hi, k, _ = slopes[grp]
            traj = gm_traj[grp]
            sl = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]" if k else "— (no cells with ≥2 horizons)"
            L.append(f"| {grp} | {_fmt_traj(traj) or '—'} | {sl} | {k} |")
        L.append("")
    L.append("## Licensed-field trajectories (the mechanism cases)\n")
    for (f, source), recs in sorted(lic.items()):
        traj = ", ".join(f"{r['years']}yr ρ={r['coupling']:+.3f}(n={r['n']})" for r in recs)
        L.append(f"- **{LAB.get(f,f)}** [{source}]: {traj}")
    L.append("\n## Read\n")
    L.append("See the slope signs above: a **negative integrated slope** is the signalling "
             "signature (prestige depreciates as employers learn); **flat/positive** is human "
             "capital. Compression/decoupled fields sitting near 0 across horizons = the "
             "no-information-to-learn baseline. PSEO is the identifying panel; Scorecard corroborates "
             "on breadth but cannot separate career from cohort time. Caveats: PSEO thin (few fields "
             f"clear n≥{NMIN} at y10); Scorecard suppression (n<30 cells) thins small fields; "
             "field counts are small so slope CIs are wide — directional evidence.\n")
    p = ROOT / "CAREER_TIME_COUPLING_RESULT.md"
    p.write_text("\n".join(L))
    print(f"[result] {p}")


def _print_summary(cls, report, lic):
    print("\n=== field types ===")
    print(cls.groupby("group").size().to_dict())
    for source in ("scorecard", "pseo"):
        _, slopes = report[source]
        print(f"\n=== {source} mean per-field slope (coupling/yr) ===")
        for grp in ("integrated", "decoupled", "middle"):
            m, lo, hi, k, _ = slopes[grp]
            print(f"  {grp:11s}: {m:+.3f} [{lo:+.3f},{hi:+.3f}]  (k={k})")
    print("\n=== licensed cases (compression test) ===")
    for (f, source), recs in sorted(lic.items()):
        print(f"  {f:24s} [{source:9s}]: " +
              " ".join(f"{r['years']}y={r['coupling']:+.2f}(n{r['n']})" for r in recs))


if __name__ == "__main__":
    main()
