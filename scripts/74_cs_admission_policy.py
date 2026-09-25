"""CS admission policy and the computer-science department remainder (referee lane 74).

REFEREE PROBLEM. The one individually clear department remainder of scripts/58 is computer science: within
institutions, net of field-specific returns to SAT, admit rate, institution Pell share and brand G, one within-CS SD of
department prestige goes with +0.050 [+0.012, +0.096] log points of earnings (per-field beta_f of the main spec,
averaged over the 4-, 1- and 5-yr Scorecard medians, 128 programs; SELECTIVITY_DEEP_RESULT.md). A referee reads this
as selection into the major: where CS admits students directly or caps the major, CS graduates are selected beyond
the institution's own admission, and prestigious departments may be the ones that select. The public test is to code,
from official university pages, how students enter each CS program in the scripts/58 CS sample and to ask whether the
remainder differs between the programs that restrict entry and those that do not. Descriptive, not causal.

CODING (data/raw/cs_admission/cs_admission_coding.csv; pages saved in data/raw/cs_admission/pages/ and listed with URL,
fetch time and md5 in pages_manifest.csv). One row per institution (IPEDS UNITID), coded from an official page of the
university (catalog / bulletin, department, college, registrar or admissions office), with the URL, access date, saved
page and a sentence quoted verbatim from it. Rules (as applied after Revision 1, 2026-09-25):
  direct_admit : an official page says that entry to the CS major (or to the college that houses it) for students
                 already enrolled at the university (change of major, internal transfer, advancement from a pre-major)
                 is competitive, decided by review, space-limited, closed or not guaranteed for students who meet the
                 published minimums, and a first-year route into the major or college, decided with the university
                 application, is documented;
  capped       : the same statement about entry after matriculation, with no first-year direct route documented on a
                 saved page;
  open         : enrolled students may declare CS on admission or at any time, or on meeting fixed published minimums
                 (GPA, courses), with no statement of competition, review or capacity for that route. Programs whose
                 saved pages document only first-year entry to the college (by published criteria, or by review with a
                 pre-major or committee route for the rest) and say nothing about later entry are coded open and
                 flagged ambiguous (uniform rule of Revision 1);
  unknown      : no official statement was found (never guessed); these rows carry no page.
  ambiguous = yes where (i) entry for enrolled students requires an application or request to the major, college or
  professional program, with minimum criteria, and the page states neither that meeting them guarantees entry nor
  that entry is competitive or capacity-limited (routine declaration forms without criteria do not count); (ii) the
  page guarantees entry above one threshold and reviews applicants below it; (iii) the page adds a discretionary
  denial clause to fixed criteria; (iv) the statement is generic (e.g. transfers between a university's schools);
  or (v) the page documents only first-year college entry and not later entry. scope: cs (statement about CS),
  school (a college-wide rule that covers CS), institution (a university-wide declaration rule; no CS-specific
  restriction found on the saved pages). The coding describes CURRENT policy (pages accessed September 2026); where the
  page dates the policy (policy_start_stated / policy_start) that is recorded. The earnings cohorts entered about a
  decade earlier (Scorecard data dictionary, asserted below), so current policy may not be the policy those graduates
  faced; a sensitivity drops programs whose page dates the policy after the cohorts' entry.

WHAT WAS FIXED IN ADVANCE AND WHAT WAS NOT. The task (2026-09-24) fixed the question -- does the CS department
remainder of scripts/58 differ between restricted (direct-admit + capped) and open programs -- together with a
bootstrap CI, power, and the rule that the test runs only if >= 60% of the programs are coded. The operationalisation
below (reading "scripts/58 partial" as the within-institution per-field beta_f, with the partial-correlation reading
as T2; the slope-split model with CS x group intercepts; the six sensitivities; the >= 5 institutions rule) is this
lane's own. It was written in this docstring, but the script was not under version control when it was written, so
whether it preceded the results cannot be verified from a version record.
  (T1, primary) the scripts/58 main within-institution per-field model (institution FE, field FE, field x {SAT, admit
    rate, institution Pell share, brand G}, field-specific beta_f z(F)) on the common programs where the 4-, 1- and 5-yr
    medians are all released (scripts/58 FE2_h3y*: 128 CS programs), with the CS slope split by entry policy:
    beta_R z(F) for restricted programs (direct_admit + capped), beta_O z(F) for open programs, beta_U z(F) for
    unknown / uncoded ones, plus CS x group intercepts (open = reference), everything else as scripts/58. z(F) is the
    scripts/58 within-CS z-score (all CS programs), so the slopes are in the same units. Estimand
    Delta = beta_R - beta_O, averaged over the three horizons (as the headline CS +0.050). This is a test of slope
    heterogeneity; a level difference between the groups is carried by the intercepts (see the exploratory level-shift
    analysis). Inference: institution pairs cluster bootstrap (the scripts/58 CS inference), B = 999, one SeedSequence
    child per replicate, 95% percentile CI; a replicate counts only if >= 5 distinct restricted and >= 5 distinct open
    CS institutions are drawn (and >= 8 CS institutions, the scripts/58 per-field rule). Power: minimum detectable
    |Delta| at 80% power (two-sided 5%, normal approximation) and power at |Delta| = the pooled CS beta and half of
    it, with the SE taken as the half-width of the 95% percentile CI / 1.96 (Revision 1; the SD of the replicates is
    sensitive to their thick tails and is reported beside it), and a Monte Carlo check with an independent seed.
  (T2, the partial-correlation reading of "scripts/58 partial") the within-CS rank partial correlation of department
    prestige and earnings given the selectivity set + G (scripts/58 F_broadG: ranks of SAT, admit rate, institution Pell
    share, state earnings level and G, CONTROL dummies), residuals fitted on all CS programs of the same sample,
    correlated within restricted and within open programs; Delta_rho = rho_R - rho_O averaged over the horizons;
    bootstrap over CS programs (one per institution), B = 999, same seed scheme; power as in T1.
  Sensitivities of T1 (paired: same bootstrap draws): no CS x group intercepts; ambiguous codes moved to unknown;
  ambiguous codes flipped (restricted <-> open); institution-wide statements moved to unknown; programs whose page
  dates the policy after the latest entry year of the earnings cohorts moved to unknown; the 4-yr main sample
  (scripts/58 FE2, 142 CS programs).
  EXPLORATORY (labelled as such in the output): direct_admit, capped and open slopes separately (a replicate is used
  if it draws >= 3 distinct institutions of each group -- the least that over-identifies a group's slope and
  intercept; Revision 1, previously >= 5, which kept a conditional subset of replicates); whether a restricted-entry
  level shift changes the pooled CS slope; descriptive contrasts of the groups; the Revision 1 comparison with the
  previous coding; and a diagnostic of which Wapman CS departments are missing from the scripts/58 CS sample and why
  (name merge in scripts/55, missing SAT_AVG), using the scripts/34 campus / city name rules by import.
  Inference note: the task's two-way (field, institution) standard of scripts/59 applies where fields and institutions
  both recur in a statistic; Delta is a contrast inside one field (other fields enter only through the shared
  institution fixed effects), so the institution cluster bootstrap of scripts/58 is the relevant variance here.

Reuses scripts/58 (sample construction fe_sample / fe_build, the per-field bootstrap for the reproduction check),
scripts/55 (cells, institution file) and scripts/34 (campus / city name rules, diagnostic only) by import; none is
modified.
Reproducible: seeded (SEED = 74), bootstrap replicates run in worker processes with one BLAS thread and one
SeedSequence child each, so outputs are byte-identical on re-run and for any worker count.
Run: OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
     .venv/bin/python scripts/74_cs_admission_policy.py [--workers N]
Outputs: data/interim/cs_admission_programs.csv (one row per CS program of the samples), data/interim/
         cs_admission_estimates.csv (every number of the write-up), CS_ADMISSION_RESULT.md (root, local only).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import sys
import re
import io
import html
import hashlib
import argparse
import unicodedata
import importlib.util
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("s58", ROOT / "scripts" / "58_selectivity_deep.py")
s58 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s58)
s55 = s58.s55

SEED = 74
B = 999                          # bootstrap replicates (as the scripts/58 within-institution bootstrap)
COVERAGE_MIN = 0.60              # task rule: tests only if >= 60% of the primary-sample CS programs are coded
MIN_GRP = 5                      # distinct restricted / open CS institutions a T1 replicate needs
MIN_GRP_EXPL = 3                 # exploratory three-group split: distinct institutions per group (Revision 1)
MIN_CS = s58.FE_PF_MIN_DISTINCT  # 8 distinct CS institutions (scripts/58 per-field rule)
SEED_MAIN, SEED_MC = (SEED, 1), (SEED, 3)   # bootstrap draws; independent draws for the Monte Carlo check
CS = "computer_science"
Z975, Z80 = norm.ppf(0.975), norm.ppf(0.80)
POLICIES = ("direct_admit", "capped", "open", "unknown")
RESTRICTED = ("direct_admit", "capped")
TOPN = 50                        # Wapman CS departments listed in the sample diagnostic
ST_H3, ST_Y4 = "in the primary sample", "4-yr sample only (1- or 5-yr median not released)"
ST_CAMPUS, ST_CITY = "lost in the name merge (campus suffix)", "lost in the name merge (city / branch qualifier)"
ST_NOMATCH = "not matched to a Scorecard CS program by these name rules"

# Revision 1 (2026-09-25): previous values of the coding cells changed after the verifier's consistency check
# (uniform rule for first-year-only college entry; the ambiguous definition applied to every row). Used only to
# report the previous-coding numbers beside the revised ones; asserted to differ from the current file.
REV1 = {"126580": {"policy": "direct_admit"}, "240453": {"policy": "direct_admit"},
        "171100": {"ambiguous": "no"}, "178411": {"ambiguous": "no"}, "155317": {"ambiguous": "no"},
        "167358": {"ambiguous": "no"}, "229115": {"ambiguous": "no"}, "131469": {"ambiguous": "no"},
        "110671": {"ambiguous": "no"}}
REV1_NOTE_ONLY = ("155399",)     # note corrected, coding unchanged

ADM_DIR = ROOT / "data" / "raw" / "cs_admission"
CODING = ADM_DIR / "cs_admission_coding.csv"
MANIFEST = ADM_DIR / "pages_manifest.csv"
PAGES = ADM_DIR / "pages"
DICT_XLSX = ROOT / "data" / "raw" / "scorecard_hist" / "CollegeScorecardDataDictionary.xlsx"
OUT_PROG = ROOT / "data" / "interim" / "cs_admission_programs.csv"
OUT_EST = ROOT / "data" / "interim" / "cs_admission_estimates.csv"
OUT_MD = ROOT / "CS_ADMISSION_RESULT.md"

HZ = [("y4", "earnings"), ("y1", "earn1"), ("y5", "earn5")]
QUICK = False                     # --quick (debug) mode
EST = []                          # every reported number: dict rows -> OUT_EST


def rec(section, stat, estimate, sample="", n=np.nan, lo=np.nan, hi=np.nan, se=np.nan, p=np.nan, note="",
        unit="log points"):
    EST.append(dict(section=section, stat=stat, sample=sample, n=n, estimate=estimate, lo=lo, hi=hi, se=se, p=p,
                    unit=unit, note=note))


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# coding: verification against the saved pages
# ---------------------------------------------------------------------------------------------
_REPL = (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-"),
         (" ", " "), ("­", ""), ("​", ""))


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for a, b in _REPL:
        s = s.replace(a, b)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Co")    # private-use glyphs (PDF bullet fonts)
    return re.sub(r"\s+", " ", s).strip()


def page_text(p: Path) -> str:
    b = p.read_bytes()
    if b[:5] == b"%PDF-":
        import logging
        import pypdf
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        r = pypdf.PdfReader(io.BytesIO(b))
        return norm_text(" ".join((pg.extract_text() or "") for pg in r.pages))
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(b, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return norm_text(html.unescape(soup.get_text(" ")))


def load_coding() -> tuple[pd.DataFrame, dict]:
    """Reads the coding and the page manifest and verifies every row against its saved page: file present, md5 of
    the file = manifest md5 = recorded md5, URL and access date as in the manifest, HTTP 200, and the quoted sentence
    present verbatim (after whitespace / typographic normalisation) in the page text. Any failure stops the run."""
    cod = pd.read_csv(CODING, dtype=str, keep_default_na=False)
    man = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    assert cod.unitid.is_unique, "one row per institution"
    assert man.file.is_unique
    bad = sorted(set(cod.policy) - set(POLICIES))
    assert not bad, f"unknown policy labels {bad}"
    assert set(cod.ambiguous) <= {"yes", "no"} and set(cod.policy_start_stated) <= {"yes", "no"}
    mi = man.set_index("file")
    fails = []
    for _, r in cod.iterrows():
        if r.policy == "unknown" and r.page_file == "":
            continue
        m = mi.loc[r.page_file]
        p = PAGES / r.page_file
        h = md5(p)
        checks = {"md5_file": h == m.md5, "md5_row": r.page_md5 == m.md5, "url": r.url == m.url,
                  "date": r.access_date == m.fetched_utc[:10], "http": m.http_code == "200",
                  "unitid": m.unitid == r.unitid}
        if r.policy != "unknown":
            checks["quote"] = norm_text(r.quote) in page_text(p)
        for k, ok in checks.items():
            if not ok:
                fails.append((r.unitid, r.instnm, k))
    assert not fails, f"coding rows that do not verify against their saved pages: {fails}"
    # every saved page is in the manifest with its md5
    on_disk = sorted(x.name for x in PAGES.iterdir() if x.is_file())
    assert on_disk == sorted(man.file), "pages/ and pages_manifest.csv differ"
    for _, m in man.iterrows():
        assert md5(PAGES / m.file) == m.md5, f"md5 mismatch {m.file}"
    # Revision 1 changes are real (the file holds the revised values) and marked in the notes
    ci_ = cod.set_index("unitid")
    for u, d in REV1.items():
        for k, v in d.items():
            assert ci_.loc[u, k] != v, (u, k, "REV1 lists a change that is not in the coding file")
        assert "Revision 1" in ci_.loc[u, "note"], u
    for u in REV1_NOTE_ONLY:
        assert "Revision 1" in ci_.loc[u, "note"], u
    # snapshots of an archive (the live page blocked scripted access): snapshot date from the archive URL
    wb = []
    for _, r in cod.iterrows():
        m = re.search(r"web\.archive\.org/web/(\d{8})", r.url)
        if m:
            wb.append(dict(unitid=r.unitid, instnm=r.instnm, snapshot=f"{m.group(1)[:4]}-{m.group(1)[4:6]}-"
                                                                      f"{m.group(1)[6:]}", access=r.access_date))
    info = dict(n_rows=len(cod), n_pages=len(man), n_pages_used=int(cod.page_file.replace("", np.nan).nunique()),
                coding_md5=md5(CODING), manifest_md5=md5(MANIFEST),
                access_dates=sorted(set(cod.access_date) - {""}),
                http=man.http_code.value_counts().to_dict(), wayback=wb)
    return cod, info


def previous_coding(cod: pd.DataFrame) -> pd.DataFrame:
    """The coding as it was before Revision 1 (only the cells listed in REV1 differ)."""
    p = cod.set_index("unitid").copy()
    for u, d in REV1.items():
        for k, v in d.items():
            p.loc[u, k] = v
    return p.reset_index()


def cohort_entry() -> dict:
    """Latest entry fall of the earnings cohorts (on-time four-year completers), from the Scorecard data dictionary
    (FieldOfStudy_Cohort_Map, FieldOfStudyMostRecent datafile); asserted."""
    x = pd.ExcelFile(DICT_XLSX)
    fos = pd.read_excel(x, "FieldOfStudy_Cohort_Map", header=None, dtype=str)
    col = list(fos.iloc[0]).index("FieldOfStudyMostRecent datafile")
    txt = {v: str(fos[fos[0] == v].iloc[0][col]) for v in ("EARN_MDN_4YR", "EARN_MDN_1YR", "EARN_MDN_5YR")}
    assert "AY2017-2018, AY2018-2019" in txt["EARN_MDN_4YR"], txt
    assert "AY2018-19, AY2019-20" in txt["EARN_MDN_1YR"], txt
    assert "AY2014-2015, AY2015-2016" in txt["EARN_MDN_5YR"], txt
    # completers of award year t-(t+1) who took four years entered in fall t-3
    entry = {"y4": (2014, 2015), "y1": (2015, 2016), "y5": (2011, 2012)}
    return dict(text=txt, entry=entry, last_entry=max(max(v) for v in entry.values()))


def start_year(s: str) -> float:
    m = re.search(r"(19|20)\d{2}", s or "")
    return float(m.group(0)) if m else np.nan


# ---------------------------------------------------------------------------------------------
# samples and designs (scripts/58)
# ---------------------------------------------------------------------------------------------
def build_cells():
    """scripts/58 compute() cell construction (the columns the FE2 specs use): scripts/55 cells + 1-yr / 5-yr medians.
    Also returns the Wapman and Scorecard tables and the institution file for the sample diagnostic."""
    ar = s58.load_ar_wapman(fields=s58.FIELDS66)
    er = s58.load_er_scorecard("undergrad", fields=s58.FIELDS66)
    inst = s55.load_institutions()
    cells = s55.build_cells(ar, er, inst)
    n0 = len(cells)
    for tag, ec, cc in s58.HORIZONS:
        e = s58.load_er_scorecard("undergrad", earn_col=ec, count_col=cc, fields=s58.FIELDS66)
        cells = cells.merge(e[["inst_key", "field", "earnings", "cohort_n"]].rename(
            columns={"earnings": tag, "cohort_n": f"{tag}_n"}), on=["field", "inst_key"], how="left")
    assert len(cells) == n0
    return cells, ar[ar.field == CS].copy(), er[er.field == CS].copy(), inst


def designs(cells, sample: str) -> dict:
    """Per-field FE2 models of scripts/58 (per_field=True). sample 'h3': the three horizons on the common programs
    (FE2_h3y4/y1/y5, identical rows and X, asserted); 'y4': the 4-yr main sample (FE2)."""
    names = [f"FE2_h3{h}" for h, _ in HZ] if sample == "h3" else ["FE2"]
    Ms = []
    for nm in names:
        slopes, need, yvar = s58.FE_SPECS[nm]
        Ms.append(s58.fe_build(cells, slopes, need, yvar, per_field=True))
    M0 = Ms[0]
    for M in Ms[1:]:
        assert M["d"].inst_key.equals(M0["d"].inst_key) and M["d"].field.equals(M0["d"].field)
        assert np.array_equal(M["X"], M0["X"]) and M["names"] == M0["names"]
    return dict(M=M0, Y=np.column_stack([M["y"] for M in Ms]), spec=names)


def demean(v: np.ndarray, groups: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    s = np.bincount(groups, weights=v, minlength=cnt.size)
    return v - (s / cnt)[groups]


def split_X(M: dict, lab: np.ndarray, level: bool = True, ref: str = "O") -> tuple[np.ndarray, list]:
    """Replaces the CS slope column b_computer_science by one slope per CS group label (lab: per-row label, '' off
    CS) and, if level, adds CS x group intercepts (reference group ref); new columns demeaned within institution."""
    j = M["names"].index(f"b_{CS}")
    cs = (M["d"].field.to_numpy() == CS)
    gl = sorted(str(g) for g in set(lab[cs]))
    cols, nm = [], []
    for g in gl:
        cols.append(demean(np.where(cs & (lab == g), M["z"], 0.0), M["groups"], M["cnt"])); nm.append(f"b_cs_{g}")
    if level:
        for g in gl:
            if g == ref:
                continue
            cols.append(demean((cs & (lab == g)).astype(float), M["groups"], M["cnt"])); nm.append(f"d_cs_{g}")
    X = np.column_stack([np.delete(M["X"], j, axis=1)] + cols)
    names = [n for i, n in enumerate(M["names"]) if i != j] + nm
    return X, names


def level_X(M: dict, lab: np.ndarray, ref: str = "O") -> tuple[np.ndarray, list]:
    """Common CS slope (scripts/58) plus CS x group intercepts (exploratory level shift)."""
    cs = (M["d"].field.to_numpy() == CS)
    cols, nm = [], []
    for g in sorted(str(g) for g in set(lab[cs])):
        if g == ref:
            continue
        cols.append(demean((cs & (lab == g)).astype(float), M["groups"], M["cnt"])); nm.append(f"d_cs_{g}")
    return np.column_stack([M["X"]] + cols), list(M["names"]) + nm


def ols(X, Y):
    return np.linalg.lstsq(X, Y, rcond=None)[0]


# ---------------------------------------------------------------------------------------------
# bootstrap engine (worker processes; one BLAS thread; one SeedSequence child per replicate)
# ---------------------------------------------------------------------------------------------
_W: dict = {}


def _init_worker():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:                                   # pragma: no cover
        pass


def _fe_chunk(args):
    key, lo, hi = args
    w = _W[key]
    A, BV, G, tj, gmask, seeds = w["A"], w["BV"], w["G"], w["tj"], w["gmask"], w["seeds"]
    out = np.full((hi - lo, len(tj), BV.shape[2]), np.nan)
    for r in range(lo, hi):
        rng = np.random.default_rng(seeds[r])
        c = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
        ok = all(int(((c > 0) & m).sum()) >= need for m, need in gmask)
        if not ok:
            continue
        XtX = np.tensordot(c, A, axes=1)
        Xty = np.tensordot(c, BV, axes=1)
        keep = np.diag(XtX) > 1e-10
        if not all(keep[j] for j in tj):
            continue
        sol = np.linalg.lstsq(XtX[np.ix_(keep, keep)], Xty[keep], rcond=None)[0]
        full = np.full((keep.size, BV.shape[2]), np.nan)
        full[keep] = sol
        out[r - lo] = full[tj]
    return lo, out


def fe_boot(key, X, Y, groups, targets, gmask, workers, seed=SEED_MAIN) -> np.ndarray:
    """Institution pairs cluster bootstrap of the coefficients `targets` (column indices) of the within-institution
    OLS of Y (n x H, several horizons sharing X) on X (rows demeaned within institution; each resampled copy of an
    institution keeps its own FE, as scripts/58 fe_group_boot / fe_boot_pf). gmask: list of (institution mask, min
    distinct) rules. seed: SeedSequence entropy; the same seed gives the same institution draws for every model on the
    same institutions (paired comparisons). Returns (B, len(targets), H)."""
    G = int(groups.max()) + 1
    idx = [np.where(groups == g)[0] for g in range(G)]
    A = np.stack([X[r].T @ X[r] for r in idx])
    BV = np.stack([X[r].T @ Y[r] for r in idx])
    full = np.linalg.lstsq(A.sum(0), BV.sum(0), rcond=None)[0]
    assert np.allclose(full, ols(X, Y), atol=1e-8), "normal equations drifted"
    seeds = np.random.SeedSequence(list(seed)).spawn(B)
    _W[key] = dict(A=A, BV=BV, G=G, tj=list(targets), gmask=gmask, seeds=seeds)
    chunks = [(key, lo, min(B, lo + 50)) for lo in range(0, B, 50)]
    res = np.full((B, len(targets), Y.shape[1]), np.nan)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"), initializer=_init_worker) as ex:
        for lo, o in ex.map(_fe_chunk, chunks):
            res[lo:lo + o.shape[0]] = o
    del _W[key]
    return res


def _pc_chunk(args):
    key, lo, hi = args
    w = _W[key]
    out = np.full((hi - lo, 3, w["Y"].shape[1]), np.nan)       # rho_R, rho_O, rho_all per horizon
    n = w["n"]
    for r in range(lo, hi):
        rng = np.random.default_rng(w["seeds"][r])
        rows = rng.integers(0, n, n)
        out[r - lo] = partial_groups(w, rows)
    return lo, out


def partial_groups(w: dict, rows: np.ndarray) -> np.ndarray:
    """Within-CS rank partial correlation (scripts/58 F_broadG): ranks of P, Y and the continuous controls among the
    (resampled) programs, CONTROL dummies; residuals on all programs, correlated within R, within O and overall."""
    lab = w["lab"][rows]
    res = np.full((3, w["Y"].shape[1]), np.nan)
    if min(int((lab == "R").sum()), int((lab == "O").sum())) < MIN_GRP or len(np.unique(rows[lab == "R"])) < MIN_GRP \
            or len(np.unique(rows[lab == "O"])) < MIN_GRP:
        return res
    cols = [np.ones((len(rows), 1))] + [rankdata(w["C"][rows, k])[:, None] for k in range(w["C"].shape[1])]
    _, inv = np.unique(w["CTRL"][rows], return_inverse=True)
    L = inv.max() + 1
    if L > 1:
        cols.append(np.eye(L)[inv][:, 1:])
    Z = np.hstack(cols)
    M = np.column_stack([rankdata(w["P"][rows])] + [rankdata(w["Y"][rows, h]) for h in range(w["Y"].shape[1])])
    coef, _, rank, _ = np.linalg.lstsq(Z, M, rcond=None)
    if len(rows) - rank - 1 < s58.DFMIN:
        return res
    E = M - Z @ coef
    for gi, sel in enumerate((lab == "R", lab == "O", np.ones(len(rows), bool))):
        ex = E[sel, 0] - E[sel, 0].mean()
        for h in range(w["Y"].shape[1]):
            ey = E[sel, h + 1] - E[sel, h + 1].mean()
            den = np.sqrt((ex @ ex) * (ey @ ey))
            res[gi, h] = ex @ ey / den if den > 1e-12 else np.nan
    return res


def pc_boot(key, w, workers) -> np.ndarray:
    w = dict(w, seeds=np.random.SeedSequence([SEED, 2]).spawn(B), n=len(w["P"]))
    _W[key] = w
    chunks = [(key, lo, min(B, lo + 100)) for lo in range(0, B, 100)]
    res = np.full((B, 3, w["Y"].shape[1]), np.nan)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"), initializer=_init_worker) as ex:
        for lo, o in ex.map(_pc_chunk, chunks):
            res[lo:lo + o.shape[0]] = o
    del _W[key]
    return res


def summarize(draws: np.ndarray, est: float) -> dict:
    """Percentile CI, two-sided percentile-bootstrap p, and two SEs: se = half-width of the 95% percentile CI / 1.96
    (used for power; robust to thick replicate tails) and se_sd = SD of the replicates."""
    d = draws[np.isfinite(draws)]
    lo, hi = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    se = (hi - lo) / (2 * Z975)
    n_far = min(int((d <= 0).sum()), int((d >= 0).sum()))
    return dict(est=float(est), lo=lo, hi=hi, se=se, se_sd=float(np.std(d, ddof=1)),
                p=min(1.0, 2 * n_far / len(d)), n_ok=int(len(d)), mde80=float((Z975 + Z80) * se))


def power_at(delta: float, se: float) -> float:
    z = abs(delta) / se
    return float(norm.cdf(z - Z975) + norm.cdf(-z - Z975))


def mem_available_mb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


# ---------------------------------------------------------------------------------------------
# sample diagnostic (exploratory): which Wapman CS departments are missing from the scripts/58 CS sample, and why
# ---------------------------------------------------------------------------------------------
def sample_gaps(cells, ar_cs, er_cs, inst, D, cod) -> pd.DataFrame:
    """One row per Wapman CS department with a prestige score: where it ends up. Departments not in the scripts/55
    cells are matched to Scorecard CS programs (4-yr median released) with the scripts/34 name rules by import
    (exact -> campus suffix stripped -> city / branch qualifier stripped; non-branch campus, then largest cohort, preferred
    within a rule). A diagnostic only: nothing here enters the tests."""
    sp = importlib.util.spec_from_file_location("s34", ROOT / "scripts" / "34_cs_ranking.py")
    s34 = importlib.util.module_from_spec(sp); sp.loader.exec_module(s34)
    a = ar_cs[ar_cs.prestige_score.notna()].drop_duplicates("inst_key").copy()
    a["wrank"] = a.prestige_score.rank(ascending=False, method="min").astype(int)
    scd = pd.DataFrame({"INSTNM": er_cs.institution_name.astype(str).to_numpy(),
                        "UNITID": er_cs.institution_id.astype(str).to_numpy(),
                        "coh": er_cs.cohort_n.fillna(0).to_numpy(float)})
    for k in ("k_exact", "k_campus", "k_city"):
        scd[k] = scd.INSTNM.map(getattr(s34, k))
    cc = cells[cells.field == CS].set_index("inst_key")
    samp = {s: set(D[s]["M"]["d"].loc[D[s]["M"]["d"].field == CS, "inst_key"]) for s in ("h3", "y4")}
    ii = inst.set_index("UNITID")
    ci_ = cod.set_index("unitid")
    rows = []
    for _, r in a.sort_values(["wrank", "inst_key"]).iterrows():
        k = r.inst_key
        uid, scn, sat = "", "", ""
        if k in cc.index:
            uid, scn = str(cc.loc[k, "UNITID"]), str(cc.loc[k, "INSTNM"])
            if k in samp["h3"]:
                st = ST_H3
            elif k in samp["y4"]:
                st = ST_Y4
            else:
                miss = [v for v in ("SAT_AVG", "ADM_RATE", "PCTPELL", "G") if not np.isfinite(cc.loc[k, v])]
                st = ("dropped: SAT_AVG missing" if "SAT_AVG" in miss else
                      f"dropped: {', '.join(miss)} missing" if miss else "dropped: scripts/58 sample rules")
        else:
            hit = s34.best(scd, s34.k_exact(r.institution_name))
            wk = s34.k_exact(r.institution_name)
            assert not (scd.k_exact == wk).any(), ("an exact name match should be in the cells", wk)
            if hit is None:
                st = ST_NOMATCH
            else:
                uid, scn = str(hit.UNITID), str(hit.INSTNM)
                st = ST_CAMPUS if (scd.k_campus == wk).any() else ST_CITY
                sat = "yes" if (uid in ii.index and np.isfinite(ii.loc[uid, "SAT_AVG"])) else "no"
        pol = ci_.loc[uid, "policy"] if uid in ci_.index else ""
        rows.append(dict(wrank=int(r.wrank), wapman=str(r.institution_name), status=st, scorecard=scn, unitid=uid,
                         sat_if_matched=sat, policy=pol))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# main computation
# ---------------------------------------------------------------------------------------------
def labels_for(d: pd.DataFrame, cod: pd.DataFrame, variant: str, last_entry: int) -> np.ndarray:
    """Per-row CS group label for a design's rows d (UNITID as str): 'R' restricted, 'O' open, 'U' unknown / uncoded;
    variant 'three' gives 'D' / 'C' / 'O' / 'U'. '' off CS."""
    c = cod.set_index("unitid")
    lab = np.full(len(d), "", dtype=object)
    for i, (f, u) in enumerate(zip(d.field.to_numpy(), d.UNITID.astype(str).to_numpy())):
        if f != CS:
            continue
        if u not in c.index or c.loc[u, "policy"] == "unknown":
            lab[i] = "U"
            continue
        r = c.loc[u]
        pol = r.policy
        g = {"direct_admit": "R", "capped": "R", "open": "O"}[pol]
        if variant == "amb_unknown" and r.ambiguous == "yes":
            g = "U"
        elif variant == "amb_flip" and r.ambiguous == "yes":
            g = {"R": "O", "O": "R"}[g]
        elif variant == "cs_specific" and r.scope == "institution":
            g = "U"
        elif variant == "dated" and r.policy_start_stated == "yes" and start_year(r.policy_start) > last_entry:
            g = "U"
        elif variant == "three":
            g = {"direct_admit": "D", "capped": "C", "open": "O"}[pol]
        lab[i] = g
    return lab.astype(str)


def fe_variant(D: dict, lab: np.ndarray, level: bool, workers: int, key: str, groups_needed=("R", "O"),
               min_grp: int = MIN_GRP, seed=SEED_MAIN):
    M, Y = D["M"], D["Y"]
    X, names = split_X(M, lab, level=level)
    beta = ols(X, Y)
    cs = M["d"].field.to_numpy() == CS
    gl = sorted(str(g) for g in set(lab[cs]))
    tj = [names.index(f"b_cs_{g}") for g in gl]
    G = int(M["cnt"].size)
    gmask = []
    for g in groups_needed:
        m = np.zeros(G, bool); m[M["groups"][cs & (lab == g)]] = True
        gmask.append((m, min_grp))
    mcs = np.zeros(G, bool); mcs[M["groups"][cs]] = True
    gmask.append((mcs, MIN_CS))
    bb = fe_boot(key, X, Y, M["groups"], tj, gmask, workers, seed=seed)
    pt = {g: beta[names.index(f"b_cs_{g}")] for g in gl}
    bd = {g: bb[:, k, :] for k, g in enumerate(gl)}
    lev = {g: beta[names.index(f"d_cs_{g}")] for g in gl if f"d_cs_{g}" in names}
    n = {g: int((cs & (lab == g)).sum()) for g in gl}
    return dict(pt=pt, boot=bd, lev=lev, n=n, groups=gl)


def t1_summary(v: dict, H: int) -> dict:
    bR, bO = np.mean(v["pt"]["R"]), np.mean(v["pt"]["O"])
    dR, dO = v["boot"]["R"].mean(axis=1), v["boot"]["O"].mean(axis=1)
    out = dict(n=v["n"], H=H, R=summarize(dR, bR), O=summarize(dO, bO), D=summarize(dR - dO, bR - bO),
               hz={h: (float(v["pt"]["R"][k] - v["pt"]["O"][k])) for k, (h, _) in enumerate(HZ[:H])},
               lev=v["lev"], U=(summarize(v["boot"]["U"].mean(axis=1), np.mean(v["pt"]["U"]))
                                if "U" in v["pt"] else None))
    if "R" in v["lev"]:
        out["levR"] = float(np.mean(v["lev"]["R"]))
    return out


def three_summary(v3: dict, have) -> dict:
    out = {g: summarize(v3["boot"][g].mean(axis=1), np.mean(v3["pt"][g])) for g in v3["pt"]}
    for g in have:
        out[f"{g}-O"] = summarize(v3["boot"][g].mean(axis=1) - v3["boot"]["O"].mean(axis=1),
                                  np.mean(v3["pt"][g]) - np.mean(v3["pt"]["O"]))
    return out


def compute(workers: int) -> dict:
    R = {}
    cod, info = load_coding()
    R["coding_info"] = info
    ce = cohort_entry()
    R["cohort"] = ce
    cells, ar_cs, er_cs, inst = build_cells()
    D = {"h3": designs(cells, "h3"), "y4": designs(cells, "y4")}

    # --- reproduction of scripts/58 (per-field CS beta_f; point and, for h3, the percentile CI) ---
    T58 = pd.read_csv(s58.OUT_CSV).set_index("field").loc[CS]
    rep = {}
    for smp, dd in D.items():
        M = dd["M"]
        b = ols(M["X"], dd["Y"])[M["names"].index(f"b_{CS}")]
        rep[smp] = b
    for k, (h, _) in enumerate(HZ):
        assert abs(rep["h3"][k] - T58[f"fe_FE2_h3{h}_beta"]) < 1e-5, (h, rep["h3"][k])
    assert abs(rep["y4"][0] - T58["fe_FE2_beta"]) < 1e-5
    M = D["h3"]["M"]
    bb58 = []
    for k, nm in enumerate(D["h3"]["spec"] if not QUICK else []):
        Mk = dict(M, y=D["h3"]["Y"][:, k])
        bb = s58.fe_boot_pf(Mk, [s58.SEED, 11], s58.B_FE, s58.FE_PF_MIN_DISTINCT)
        bb58.append(bb[:, M["fields"].index(CS)])
    if QUICK:                                                    # debug mode: reproduction CI not re-run
        rep["h3_ci"] = (float(T58["fe_FE2_h3avg_ci_lo"]), float(T58["fe_FE2_h3avg_ci_hi"]))
    else:
        a58 = np.mean(bb58, axis=0)
        rep["h3_ci"] = (float(np.nanquantile(a58, 0.025)), float(np.nanquantile(a58, 0.975)))
    assert abs(rep["h3_ci"][0] - T58["fe_FE2_h3avg_ci_lo"]) < 1e-5 and abs(rep["h3_ci"][1] - T58["fe_FE2_h3avg_ci_hi"]) < 1e-5
    R["rep"] = rep
    R["beta_cs_h3avg"] = float(np.mean(rep["h3"]))
    rec("reproduction", "CS beta_f, main spec, 4/1/5-yr average (scripts/58)", R["beta_cs_h3avg"],
        "common programs", int((M["d"].field == CS).sum()), *rep["h3_ci"],
        note="CI: scripts/58 per-field bootstrap re-run (seed [58, 11], B = 999); matches data/interim/selectivity_deep.csv")
    for k, (h, _) in enumerate(HZ):
        rec("reproduction", f"CS beta_f, main spec, {h}", float(rep["h3"][k]), "common programs",
            int((M["d"].field == CS).sum()))
    rec("reproduction", "CS beta_f, main spec, 4-yr", float(rep["y4"][0]), "4-yr main sample",
        int((D["y4"]["M"]["d"].field == CS).sum()))

    # --- program table and coverage ---
    cod_i = cod.set_index("unitid")
    progs = []
    for smp, dd in D.items():
        M = dd["M"]
        cs = (M["d"].field == CS).to_numpy()
        d = M["d"][cs]
        for (_, r), z in zip(d.iterrows(), M["z"][cs]):
            progs.append(dict(sample=smp, unitid=str(r.UNITID), instnm=r.INSTNM, inst_key=r.inst_key, z_F=z,
                              prestige_score=r.prestige_score, SAT_AVG=r.SAT_AVG, ADM_RATE=r.ADM_RATE,
                              PCTPELL=r.PCTPELL, G=r.G, CONTROL=r.CONTROL, STABBR=r.STABBR, earn4=r.earnings,
                              earn1=r.earn1, earn5=r.earn5))
    P = pd.DataFrame(progs)
    P["policy"] = P.unitid.map(cod_i.policy).fillna("uncoded")
    for c_ in ("ambiguous", "scope", "gate", "admit_unit", "policy_start_stated", "policy_start"):
        P[c_] = P.unitid.map(cod_i[c_]).fillna("")
    P["group"] = np.where(P.policy.isin(RESTRICTED), "restricted", np.where(P.policy == "open", "open", "unknown"))
    R["P"] = P
    cov = {}
    for smp in ("h3", "y4"):
        q = P[P["sample"] == smp]
        n = len(q)
        vc = q.policy.value_counts()
        coded = q.policy.isin(RESTRICTED + ("open",))
        cov[smp] = dict(n=n, coded=int(coded.sum()),
                        **{k: int(vc.get(k, 0)) for k in POLICIES + ("uncoded",)},
                        amb=int((coded & (q.ambiguous == "yes")).sum()),
                        inst_scope=int((coded & (q.scope == "institution")).sum()),
                        unamb=int((coded & (q.ambiguous == "no")).sum()),
                        unamb_ni=int((coded & (q.ambiguous == "no") & (q.scope != "institution")).sum()),
                        dated=int((q.policy_start_stated == "yes").sum()))
        for k in ("coded", "unamb", "unamb_ni"):
            cov[smp][f"share_{k}"] = cov[smp][k] / n
        cov[smp]["share"] = cov[smp]["share_coded"]
        smp_lab = {"h3": "common programs (primary)", "y4": "4-yr main sample"}[smp]
        rec("coverage", "CS programs in sample", n, smp_lab, unit="count")
        rec("coverage", "coded (direct_admit + capped + open)", cov[smp]["coded"], smp_lab,
            note=f"share {cov[smp]['share']:.3f}", unit="count")
        for k in POLICIES + ("uncoded",):
            rec("coverage", f"policy = {k}", cov[smp][k], smp_lab, unit="count")
        rec("coverage", "coded ambiguous", cov[smp]["amb"], smp_lab, unit="count")
        rec("coverage", "coded from an institution-wide statement", cov[smp]["inst_scope"], smp_lab, unit="count")
        rec("coverage", "coded, not ambiguous", cov[smp]["unamb"], smp_lab,
            note=f"share {cov[smp]['share_unamb']:.3f}", unit="count")
        rec("coverage", "coded, not ambiguous, not from an institution-wide statement", cov[smp]["unamb_ni"], smp_lab,
            note=f"share {cov[smp]['share_unamb_ni']:.3f}", unit="count")
        rec("coverage", "page dates the policy", cov[smp]["dated"], smp_lab, unit="count")
    R["cov"] = cov
    R["run_tests"] = cov["h3"]["share"] >= COVERAGE_MIN
    rec("coverage", "task threshold met (>= 60% coded, primary sample)", float(R["run_tests"]), "common programs",
        unit="indicator")

    # --- sample diagnostic (exploratory) and coded rows outside both samples ---
    gaps = sample_gaps(cells, ar_cs, er_cs, inst, D, cod)
    R["gaps"] = gaps
    top = gaps[gaps.wrank <= TOPN]
    for lab_, g in (("all", gaps), (f"top {TOPN}", top)):
        for st, n_ in g.status.value_counts().sort_index().items():
            rec("sample diagnostic (exploratory)", f"Wapman CS departments with prestige ({lab_}): {st}", n_,
                "Wapman CS departments", unit="count")
        rec("sample diagnostic (exploratory)", f"Wapman CS departments with prestige ({lab_})", len(g),
            "Wapman CS departments", unit="count")
    in_any = set(P.unitid)
    oos = []
    ccu = cells[cells.field == CS].copy()
    ccu["UNITID"] = ccu.UNITID.astype(str)
    ccu = ccu.set_index("UNITID")
    for _, r in cod.iterrows():
        if r.unitid in in_any:
            continue
        if r.unitid in ccu.index:
            miss = [v for v in ("SAT_AVG", "ADM_RATE", "PCTPELL", "G") if not np.isfinite(ccu.loc[r.unitid, v])]
            why = f"in the scripts/55 CS cells; {', '.join(miss)} missing" if miss else "in the cells; sample rules"
        else:
            why = "not in the scripts/55 CS cells"
        oos.append(dict(unitid=r.unitid, instnm=r.instnm, policy=r.policy, why=why))
    R["oos"] = pd.DataFrame(oos)
    rec("sample diagnostic (exploratory)", "coded institutions outside both analysis samples", len(oos),
        "coding file", unit="count")
    if not R["run_tests"]:
        return R

    # --- descriptive contrasts of the groups (exploratory) ---
    desc = []
    q = P[P["sample"] == "h3"].copy()
    q["log_earn4"] = np.log(q.earn4)
    q["public"] = (q.CONTROL == 1).astype(float)
    for g, s in [(g, q[q.group == g]) for g in ("restricted", "open", "unknown")] + \
                [(g, q[q.policy == g]) for g in ("direct_admit", "capped")]:
        desc.append(dict(group=g, n=len(s), z_F=s.z_F.mean(), SAT_AVG=s.SAT_AVG.median(),
                         ADM_RATE=s.ADM_RATE.median(), G_rank=(-s.G).median(), public=s.public.mean(),
                         log_earn4=s.log_earn4.mean()))
    R["desc"] = pd.DataFrame(desc)
    units = dict(z_F="SD", SAT_AVG="SAT points (median)", ADM_RATE="share (median)", G_rank="rank (median)",
                 public="share", log_earn4="log dollars (mean)")
    for _, r in R["desc"].iterrows():
        for c_ in ("z_F", "SAT_AVG", "ADM_RATE", "G_rank", "public", "log_earn4"):
            rec("descriptive (exploratory)", f"{c_}, {r.group}", float(r[c_]), "common programs", int(r.n),
                unit=units[c_])

    # --- T1 primary and sensitivities ---
    VAR = [("primary", "h3", "primary", True), ("no_level", "h3", "primary", False),
           ("amb_unknown", "h3", "amb_unknown", True), ("amb_flip", "h3", "amb_flip", True),
           ("cs_specific", "h3", "cs_specific", True), ("dated", "h3", "dated", True),
           ("y4_sample", "y4", "primary", True)]
    T1 = {}
    for name, smp, lv, level in VAR:
        dd = D[smp]
        lab = labels_for(dd["M"]["d"], cod, lv, ce["last_entry"])
        T1[name] = t1_summary(fe_variant(dd, lab, level, workers, name), dd["Y"].shape[1])
    R["T1"] = T1
    t = T1["primary"]
    R["power"] = dict(se=t["D"]["se"], se_sd=t["D"]["se_sd"], mde80=t["D"]["mde80"],
                      pow_full=power_at(R["beta_cs_h3avg"], t["D"]["se"]),
                      pow_half=power_at(R["beta_cs_h3avg"] / 2, t["D"]["se"]))
    # Monte Carlo check: the primary T1 with independent bootstrap draws
    dd = D["h3"]
    lab = labels_for(dd["M"]["d"], cod, "primary", ce["last_entry"])
    mc = t1_summary(fe_variant(dd, lab, True, workers, "mc", seed=SEED_MC), dd["Y"].shape[1])
    R["mc"] = dict(D=mc["D"], pow_full=power_at(R["beta_cs_h3avg"], mc["D"]["se"]),
                   pow_half=power_at(R["beta_cs_h3avg"] / 2, mc["D"]["se"]),
                   pow_full_sd=power_at(R["beta_cs_h3avg"], mc["D"]["se_sd"]),
                   pow_full_sd_main=power_at(R["beta_cs_h3avg"], t["D"]["se_sd"]))

    # --- Revision 1: the previous coding on the same draws (primary T1) ---
    cod_prev = previous_coding(cod)
    labp = labels_for(dd["M"]["d"], cod_prev, "primary", ce["last_entry"])
    R["prev"] = t1_summary(fe_variant(dd, labp, True, workers, "prev"), dd["Y"].shape[1])
    R["prev_nolevel"] = t1_summary(fe_variant(dd, labp, False, workers, "prev_nolevel"), dd["Y"].shape[1])

    # --- exploratory: direct_admit vs capped vs open (>= 3 distinct institutions per group) ---
    lab3 = labels_for(dd["M"]["d"], cod, "three", ce["last_entry"])
    cs3 = dd["M"]["d"].field.to_numpy() == CS
    have = [g for g in ("D", "C") if int((cs3 & (lab3 == g)).sum()) >= MIN_GRP_EXPL]
    v3 = fe_variant(dd, lab3, True, workers, "three", groups_needed=tuple(have) + ("O",), min_grp=MIN_GRP_EXPL)
    R["three"], R["three_n"] = three_summary(v3, have), v3["n"]
    # the previous rule (>= 5 per group) on the previous coding, and the new rule on the previous coding
    lab3p = labels_for(dd["M"]["d"], cod_prev, "three", ce["last_entry"])
    R["three_prev"] = {}
    for mg in (MIN_GRP, MIN_GRP_EXPL):
        vp = fe_variant(dd, lab3p, True, workers, f"three_prev{mg}", groups_needed=("D", "C", "O"), min_grp=mg)
        R["three_prev"][mg] = dict(s=three_summary(vp, ("D", "C")), n=vp["n"])

    # --- exploratory: restricted-entry level shift and the pooled CS slope (common slope) ---
    M = dd["M"]
    XL, nL = level_X(M, lab)
    bL = ols(XL, dd["Y"])
    jb = nL.index(f"b_{CS}")
    tj = [jb] + [nL.index(n_) for n_ in nL if n_.startswith("d_cs_")]
    G = int(M["cnt"].size)
    mcs = np.zeros(G, bool); mcs[M["groups"][M["d"].field.to_numpy() == CS]] = True
    bbL = fe_boot("level", XL, dd["Y"], M["groups"], tj, [(mcs, MIN_CS)], workers)
    bb0 = fe_boot("base", M["X"], dd["Y"], M["groups"], [M["names"].index(f"b_{CS}")], [(mcs, MIN_CS)], workers)
    b0 = R["beta_cs_h3avg"]
    R["level"] = dict(beta=summarize(bbL[:, 0, :].mean(axis=1), float(np.mean(bL[jb]))),
                      base=summarize(bb0[:, 0, :].mean(axis=1), b0),
                      diff=summarize(bbL[:, 0, :].mean(axis=1) - bb0[:, 0, :].mean(axis=1), float(np.mean(bL[jb])) - b0),
                      dR=summarize(bbL[:, tj.index(nL.index("d_cs_R")), :].mean(axis=1),
                                   float(np.mean(bL[nL.index("d_cs_R")]))))

    # --- T2: partial-correlation reading ---
    q = M["d"][M["d"].field == CS].reset_index(drop=True)
    labq = lab[M["d"].field.to_numpy() == CS]
    w = dict(P=q.prestige_score.to_numpy(float), Y=np.column_stack([q[c].to_numpy(float) for _, c in HZ]),
             C=np.column_stack([q[c].to_numpy(float) for c in s58.BROAD + ["G"]]),
             CTRL=q.CONTROL.fillna(-1).to_numpy(int), lab=labq)
    pt = partial_groups(w, np.arange(len(q)))
    bb = pc_boot("pc", w, workers)
    R["T2"] = dict(n=dict(R=int((labq == "R").sum()), O=int((labq == "O").sum()), U=int((labq == "U").sum()),
                          all=len(q)),
                   R=summarize(bb[:, 0, :].mean(axis=1), float(pt[0].mean())),
                   O=summarize(bb[:, 1, :].mean(axis=1), float(pt[1].mean())),
                   all=summarize(bb[:, 2, :].mean(axis=1), float(pt[2].mean())),
                   D=summarize(bb[:, 0, :].mean(axis=1) - bb[:, 1, :].mean(axis=1), float(pt[0].mean() - pt[1].mean())),
                   hz={h: float(pt[0, k] - pt[1, k]) for k, (h, _) in enumerate(HZ)},
                   ref58=float(T58["F_broadG_h3avg"]))
    R["T2"]["pow_ref"] = power_at(R["T2"]["all"]["est"], R["T2"]["D"]["se"])
    # same 128 programs as scripts/58's common-program partial for CS: the all-program partial reproduces it
    assert abs(R["T2"]["all"]["est"] - R["T2"]["ref58"]) < 1e-5, (R["T2"]["all"]["est"], R["T2"]["ref58"])
    return R


# ---------------------------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------------------------
def f3(x):
    return f"{x:+.3f}"


def ci(s, fmt=f3):
    return f"{fmt(s['est'])} [{fmt(s['lo'])}, {fmt(s['hi'])}]"


def pct(x):
    return f"{100 * x:.0f}%"


GLAB = {"D": "direct-admit", "C": "capped", "O": "open", "U": "unknown / uncoded"}
SENS = (("no_level", "without CS × group intercepts"), ("amb_unknown", "ambiguous codes set to unknown"),
        ("amb_flip", "ambiguous codes flipped"), ("cs_specific", "institution-wide statements set to unknown"),
        ("dated", "policies dated after the cohorts' entry set to unknown"),
        ("y4_sample", "4-yr main sample (4-yr earnings only)"))


def excl0(s):
    return not (s["lo"] <= 0 <= s["hi"])


def write(R: dict):
    P = R["P"]
    P.sort_values(["sample", "unitid"]).to_csv(OUT_PROG, index=False, float_format="%.6g")
    cv, ce = R["cov"], R["cohort"]
    T = R.get("T1")
    S1 = "T1 within-institution CS slope by entry policy"
    if R["run_tests"]:
        t = T["primary"]
        for name, o in T.items():
            smp = "4-yr main sample" if name == "y4_sample" else "common programs"
            for g in ("R", "O"):
                rec(S1, f"beta_{g} [{name}]", o[g]["est"], smp, o["n"].get(g), o[g]["lo"], o[g]["hi"], o[g]["se"],
                    o[g]["p"], note=f"replicates used {o[g]['n_ok']} of {B}")
            rec(S1, f"Delta = beta_R - beta_O [{name}]", o["D"]["est"], smp,
                o["n"].get("R", 0) + o["n"].get("O", 0), o["D"]["lo"], o["D"]["hi"], o["D"]["se"], o["D"]["p"],
                note=f"replicates used {o['D']['n_ok']} of {B}; MDE80 {o['D']['mde80']:.4f}")
            if o.get("U") is not None:
                rec(S1, f"beta_U [{name}]", o["U"]["est"], smp, o["n"].get("U"),
                    o["U"]["lo"], o["U"]["hi"], o["U"]["se"], o["U"]["p"], note=f"replicates used {o['U']['n_ok']} of {B}")
            if "levR" in o:
                rec(S1, f"CS x restricted intercept [{name}]", o["levR"], smp)
        for h, v in t["hz"].items():
            rec(S1, f"Delta at {h} [primary]", v, "common programs")
        pw, mc = R["power"], R["mc"]
        SP = "T1 power"
        rec(SP, "SE of Delta (half-width of the 95% percentile CI / 1.96; used for power)", pw["se"], "common programs")
        rec(SP, "SE of Delta (SD of the bootstrap replicates; for comparison)", pw["se_sd"], "common programs")
        rec(SP, "minimum detectable |Delta| at 80% power (two-sided 5%)", pw["mde80"], "common programs")
        rec(SP, "power at |Delta| = pooled CS beta", pw["pow_full"], "common programs",
            note=f"pooled CS beta {R['beta_cs_h3avg']:+.4f}", unit="probability")
        rec(SP, "power at |Delta| = half the pooled CS beta", pw["pow_half"], "common programs", unit="probability")
        rec(SP, "power at |Delta| = pooled CS beta, SD-based SE (for comparison)", mc["pow_full_sd_main"],
            "common programs", unit="probability")
        SM = "T1 Monte Carlo check (independent bootstrap seed [74, 3])"
        rec(SM, "Delta = beta_R - beta_O [primary]", mc["D"]["est"], "common programs", np.nan, mc["D"]["lo"],
            mc["D"]["hi"], mc["D"]["se"], mc["D"]["p"], note=f"replicates used {mc['D']['n_ok']} of {B}")
        rec(SM, "SE of Delta (percentile half-width / 1.96)", mc["D"]["se"], "common programs")
        rec(SM, "SE of Delta (SD of replicates)", mc["D"]["se_sd"], "common programs")
        rec(SM, "minimum detectable |Delta| at 80% power", mc["D"]["mde80"], "common programs")
        rec(SM, "power at |Delta| = pooled CS beta", mc["pow_full"], "common programs", unit="probability")
        rec(SM, "power at |Delta| = pooled CS beta, SD-based SE", mc["pow_full_sd"], "common programs",
            unit="probability")
        SX = "exploratory: direct_admit / capped / open"
        for g, s in R["three"].items():
            rec(SX, f"beta_{g}".replace("-", " - beta_"), s["est"], "common programs", R["three_n"].get(g.split("-")[0]), s["lo"], s["hi"],
                s["se"], s["p"], note=f"replicates used {s['n_ok']} of {B} (>= {MIN_GRP_EXPL} distinct institutions "
                                      "per group)")
        SV = "Revision 1: previous coding (same bootstrap draws)"
        pv = R["prev"]
        for g in ("R", "O"):
            rec(SV, f"beta_{g} [primary, previous coding]", pv[g]["est"], "common programs", pv["n"].get(g),
                pv[g]["lo"], pv[g]["hi"], pv[g]["se"], pv[g]["p"])
        rec(SV, "Delta = beta_R - beta_O [primary, previous coding]", pv["D"]["est"], "common programs",
            pv["n"].get("R", 0) + pv["n"].get("O", 0), pv["D"]["lo"], pv["D"]["hi"], pv["D"]["se"], pv["D"]["p"],
            note=f"replicates used {pv['D']['n_ok']} of {B}")
        pn = R["prev_nolevel"]
        rec(SV, "Delta = beta_R - beta_O [no_level, previous coding]", pn["D"]["est"], "common programs",
            pn["n"].get("R", 0) + pn["n"].get("O", 0), pn["D"]["lo"], pn["D"]["hi"], pn["D"]["se"], pn["D"]["p"],
            note=f"replicates used {pn['D']['n_ok']} of {B}")
        for mg, o in R["three_prev"].items():
            for g in ("D", "C", "O", "D-O"):
                s = o["s"][g]
                rec(SV, f"beta_{g} [three-group, previous coding, >= {mg} distinct institutions per group]".replace(
                    "-O", " - beta_O"), s["est"],
                    "common programs", o["n"].get(g.split("-")[0]), s["lo"], s["hi"], s["se"], s["p"],
                    note=f"replicates used {s['n_ok']} of {B}")
        L = R["level"]
        for k_, lab_, nt in (("base", "pooled CS beta (no level shift)",
                              "this lane's institution bootstrap (seed [74, 1], >= 8 CS institutions); the "
                              "reproduction row's CI is scripts/58's per-field bootstrap, a different set of draws"),
                             ("beta", "pooled CS beta with CS x restricted / unknown intercepts", "same draws as above"),
                             ("diff", "change in the pooled CS beta", "paired, same draws"),
                             ("dR", "CS x restricted intercept (log points)", "same draws")):
            rec("exploratory: restricted-entry level shift", lab_, L[k_]["est"], "common programs", np.nan,
                L[k_]["lo"], L[k_]["hi"], L[k_]["se"], L[k_]["p"], note=nt)
        T2 = R["T2"]
        for g in ("R", "O", "all"):
            rec("T2 within-CS partial correlation by entry policy", f"rho_{g}", T2[g]["est"], "common programs",
                T2["n"][g], T2[g]["lo"], T2[g]["hi"], T2[g]["se"], T2[g]["p"], unit="rank correlation",
                note=f"replicates used {T2[g]['n_ok']} of {B}")
        rec("T2 within-CS partial correlation by entry policy", "Delta_rho = rho_R - rho_O", T2["D"]["est"],
            "common programs", T2["n"]["R"] + T2["n"]["O"], T2["D"]["lo"], T2["D"]["hi"], T2["D"]["se"], T2["D"]["p"],
            note=f"replicates used {T2['D']['n_ok']} of {B}; MDE80 {T2['D']['mde80']:.3f}", unit="rank correlation")
        rec("T2 power", "power at |Delta_rho| = the all-program partial", T2["pow_ref"], "common programs",
            unit="probability")
        rec("reproduction", "CS rho(F, Y | selectivity set + G), 4/1/5-yr average, scripts/58 h3 partial sample",
            T2["ref58"], "scripts/58 selectivity_deep.csv", unit="rank correlation")
    E = pd.DataFrame(EST)
    E.to_csv(OUT_EST, index=False, float_format="%.6g")

    L = []
    w = L.append
    gp = R["gaps"]
    top = gp[gp.wrank <= TOPN]
    top_miss = top[top.status != ST_H3]
    n_merge_top = int(top.status.str.startswith("lost in the name merge").sum())
    n_sat_top = int((top.status == "dropped: SAT_AVG missing").sum())
    n_merge_all = int(gp.status.str.startswith("lost in the name merge").sum())
    w("# CS admission policy: does the computer-science department remainder differ between programs that restrict "
      "entry and programs that do not?")
    w("")
    w("Script: `scripts/74_cs_admission_policy.py` (seeded, byte-identical on re-run and for any worker count). Coding: "
      "`data/raw/cs_admission/cs_admission_coding.csv` (one row per institution; URL, access date, saved page, md5 and a "
      "verbatim sentence from an official university page), pages in `data/raw/cs_admission/pages/` (manifest "
      "`pages_manifest.csv`). Per-program table: `data/interim/cs_admission_programs.csv`; every number below: "
      "`data/interim/cs_admission_estimates.csv`. Public data only; descriptive, not causal. Revised on 2026-09-25 "
      "after an independent check (see Revisions).")
    w("")
    w("## Answer")
    w("")
    ch, cy = cv["h3"], cv["y4"]
    w("**Key question: in the scripts/58 CS sample, is the within-institution slope of earnings on department prestige "
      f"({R['beta_cs_h3avg']:+.3f} log points per SD) different in CS programs that admit students directly or cap the "
      "major than in programs that students can enter by declaring or on fixed minimums?**")
    w("")
    if not R["run_tests"]:
        w(f"**Coverage only.** {ch['coded']} of the {ch['n']} CS programs of the primary sample ({100 * ch['share']:.0f}%) "
          f"could be coded from an official page, below the task's 60%, so the test was not run.")
    else:
        t, T2, pw, mc = T["primary"], R["T2"], R["power"], R["mc"]
        inc0 = t["D"]["lo"] <= 0 <= t["D"]["hi"]
        ex_s = [(k, lab_) for k, lab_ in SENS if excl0(T[k]["D"])]
        if not ex_s and not excl0(T2["D"]):
            sens_txt = "The partial-correlation reading and every sensitivity give a CI for the difference that includes zero."
        else:
            sens_txt = (("The partial-correlation reading gives a CI that includes zero" if not excl0(T2["D"]) else
                         "The partial-correlation reading gives a CI that excludes zero")
                        + f"; {len(SENS) - len(ex_s)} of the {len(SENS)} sensitivities give a CI that includes zero"
                        + ("" if not ex_s else "; the exception" + ("s are " if len(ex_s) > 1 else " is ")
                           + ", ".join(f"the variant {lab_} (Δ = {ci(T[k]['D'])})" for k, lab_ in ex_s))
                        + (". Without the intercepts, a level difference between the groups can load onto their "
                           "slopes (answer 5)." if "no_level" in dict(ex_s) else "."))
        w("**Short answer: " + ("no difference in the slope between restricted-entry and open CS programs is detected"
                                if inc0 else "the slope differs between restricted-entry and open CS programs")
          + (", and the test is too weak to rule out a difference as large as the pooled slope.** " if
             (inc0 and pw["mde80"] > R["beta_cs_h3avg"]) else ".** ")
          + f"The within-institution slope is β_R = {ci(t['R'])} in restricted programs (n = {t['n'].get('R', 0)}) and "
          f"β_O = {ci(t['O'])} in open ones (n = {t['n'].get('O', 0)}); the difference is Δ = {ci(t['D'])}. The "
          f"smallest difference this sample detects with 80% power is {pw['mde80']:.3f}, "
          f"{pw['mde80'] / R['beta_cs_h3avg']:.1f} times the pooled CS slope. " + sens_txt
          + " The conclusion is limited to the scripts/58 CS sample, which lacks many well-known CS departments "
          "(answer 8).")
        w("")
        w(f"Coverage: {ch['coded']} of the {ch['n']} CS programs of the primary sample ({100 * ch['share']:.0f}%) are coded "
          f"from official pages ({ch['direct_admit']} direct-admit, {ch['capped']} capped, {ch['open']} open; "
          f"{ch['unknown'] + ch['uncoded']} without a usable official statement), above the task's 60%, so the tests "
          "were run. "
          + ("The threshold is met only because ambiguous codes count as coded: " if ch["share_unamb"] < COVERAGE_MIN
             else "Without the ambiguous codes the threshold would still be met: ")
          + f"{ch['unamb']} programs "
          f"({100 * ch['share_unamb']:.0f}%) are coded without an ambiguity flag, and {ch['unamb_ni']} "
          f"({100 * ch['share_unamb_ni']:.0f}%) without a flag and from a CS- or college-level statement; "
          f"{ch['amb']} codes are flagged ambiguous. The sensitivity that sets ambiguous codes to unknown (answer 4) is "
          "the analysis on unambiguous codes only.")
        w("")
        w(f"1. **Within-institution slope by entry policy (T1, primary).** Restricted programs "
          f"(direct-admit + capped, n = {t['n'].get('R', 0)}): β_R = {ci(t['R'])}; open programs (n = {t['n'].get('O', 0)}): "
          f"β_O = {ci(t['O'])} (log points per within-CS SD of department prestige, average of the 4-, 1- and 5-yr "
          f"medians). Difference Δ = β_R − β_O = {ci(t['D'])} (institution cluster bootstrap, {t['D']['n_ok']} of {B} "
          f"replicates usable; percentile-bootstrap p = {t['D']['p']:.2f}). By horizon (point): "
          + ", ".join(f"{h} {v:+.3f}" for h, v in t["hz"].items()) + ".")
        w(f"2. **Power.** Taking the SE of Δ as the half-width of its 95% percentile CI / 1.96 ({pw['se']:.3f}; the SD of "
          f"the replicates is {pw['se_sd']:.3f}), the minimum detectable |Δ| at 80% power is {pw['mde80']:.3f}, "
          f"{pw['mde80'] / R['beta_cs_h3avg']:.1f} times the pooled CS slope ({R['beta_cs_h3avg']:+.3f}). Power to detect "
          f"a difference as large as the pooled CS slope is {pct(pw['pow_full'])}, and half of it {pct(pw['pow_half'])}. "
          f"With an independent bootstrap seed, Δ = {ci(mc['D'])}, SE {mc['D']['se']:.3f} (SD {mc['D']['se_sd']:.3f}), "
          f"minimum detectable |Δ| {mc['D']['mde80']:.3f} and power {pct(mc['pow_full'])} at the pooled slope; the "
          f"SD-based SE gives {pct(mc['pow_full_sd_main'])} (main seed) and {pct(mc['pow_full_sd'])} (second seed)."
          + (" A null here is weak evidence either way." if inc0 else ""))
        w(f"3. **Partial-correlation reading (T2).** Within CS, the rank partial correlation of department prestige with "
          f"earnings given the selectivity set and G (residuals from all {T2['n']['all']} programs) is {ci(T2['R'])} in "
          f"restricted programs (n = {T2['n']['R']}) and {ci(T2['O'])} in open ones (n = {T2['n']['O']}); "
          f"Δρ = {ci(T2['D'])} (p = {T2['D']['p']:.2f}; minimum detectable |Δρ| at 80% power {T2['D']['mde80']:.2f}). "
          f"All programs: {ci(T2['all'])}.")
        s = T
        w("4. **Sensitivities of T1 (Δ = β_R − β_O).** "
          + "; ".join(f"{lab_.replace('the cohorts' + chr(39) + ' entry', 'fall ' + str(ce['last_entry']))} "
                      f"{ci(s[k]['D'])} (n_R = {s[k]['n'].get('R', 0)}, n_O = {s[k]['n'].get('O', 0)})"
                      for k, lab_ in SENS) + "."
          + (" The variant without intercepts is the only one whose CI excludes zero; it is the one that lets a "
             f"level difference (the CS × restricted intercept is {t['levR']:+.3f} in the primary slope-split model) "
             "load onto the slopes." if [k for k, _ in ex_s] == ["no_level"] else ""))
        L3 = R["level"]
        dsc = R["desc"].set_index("group")
        w(f"5. **What T1 answers and what it does not.** T1 tests whether the prestige slope differs between the groups. "
          "Selection into the major would show first as a level difference (graduates of restricted programs earning "
          "more, net of the controls), and it would bias the pooled slope if restricted programs were also the more "
          "prestigious departments"
          + (", which they are here" if dsc.loc["restricted", "z_F"] > dsc.loc["open", "z_F"] else
             ", which they are not here")
          + f" (mean z(F) {dsc.loc['restricted', 'z_F']:+.2f} restricted, "
          f"{dsc.loc['open', 'z_F']:+.2f} open). It predicts a slope difference only if the strength of selection "
          "rises with department prestige. The direct check of the confounding reading is exploratory: adding "
          "CS × restricted / unknown intercepts to the scripts/58 model (common CS slope) moves the pooled CS slope "
          f"from {L3['base']['est']:+.3f} to {ci(L3['beta'])} (paired change {ci(L3['diff'])}); the CS × restricted "
          f"intercept in this common-slope model is {ci(L3['dR'])} log points relative to open programs"
          + (f" (about {100 * np.expm1(L3['dR']['est']):.0f}%): relative to the same institution's other departments, "
             "CS graduates earn more where entry to CS is restricted, at the same department prestige. That is the "
             "kind of level difference selection into the major would produce, but restricted entry may also mark "
             "the institutions where demand for CS is highest, so it does not show that selection causes it"
             if excl0(L3["dR"]) and L3["dR"]["est"] > 0 else "")
          + ". Being exploratory, this is not a main result.")
        th = R["three"]
        w(f"6. **Exploratory: direct-admit, capped and open separately.** "
          + "; ".join(f"{GLAB.get(g, g)} {ci(v)} (n = {R['three_n'].get(g)})" for g, v in th.items() if "-" not in g)
          + "; differences from open: "
          + "; ".join(f"{GLAB[g.split('-')[0]]} {ci(v)}" for g, v in th.items() if "-" in g)
          + f". A replicate is used if it draws at least {MIN_GRP_EXPL} distinct institutions of each group "
          f"({th['O']['n_ok']} of {B} usable)"
          + (f"; with only {R['three_n'].get('D', 0)} direct-admit institutions this is still a conditional subset "
             "of the draws" if th["O"]["n_ok"] < 0.95 * B else "")
          + f", and the upper ends (up to {max(v['hi'] for g, v in th.items() if '-' not in g):+.2f} log points per SD) "
          f"show that the slopes of the {R['three_n'].get('D', 0)} direct-admit and {R['three_n'].get('C', 0)} capped "
          "programs are barely identified. Neither split is a finding.")
        w("7. **Descriptive (exploratory).** "
          + "; ".join(f"{g} programs (n = {int(dsc.loc[g, 'n'])}): mean z(F) {dsc.loc[g, 'z_F']:+.2f}, median SAT "
                      f"{dsc.loc[g, 'SAT_AVG']:.0f}, median admit rate {dsc.loc[g, 'ADM_RATE']:.2f}, median academia-wide "
                      f"rank {dsc.loc[g, 'G_rank']:.0f}, public share {dsc.loc[g, 'public']:.2f}"
                      for g in ("restricted", "open", "unknown"))
          + ". Restricted and uncoded programs differ from open ones in institution type, so the groups are not "
            "comparable except through the model's controls.")
        tc = top.status.value_counts()
        w(f"8. **The sample (exploratory diagnostic).** Of the {len(top)} most prestigious Wapman CS departments, "
          f"{len(top) - len(top_miss)} are in the primary sample and {tc.get(ST_Y4, 0)} in the 4-yr sample only (a 1- or "
          f"5-yr median is not released); {n_sat_top} drop out because the 2026 institution file has no SAT_AVG for "
          f"them; {n_merge_top} are lost in the scripts/55 name merge because their Scorecard names carry a campus or "
          f"city qualifier that Wapman omits ({tc.get(ST_CAMPUS, 0)} campus suffix, {tc.get(ST_CITY, 0)} city / branch "
          f"qualifier); and {tc.get(ST_NOMATCH, 0)} are not matched by these name rules (name forms differ). Across all "
          f"{len(gp)} Wapman CS departments with a prestige score, {n_merge_all} are lost in the name merge by the "
          f"scripts/34 name rules and {int((gp.status == ST_NOMATCH).sum())} are not matched by them. The entry "
          "policies of the name-merge losses are not coded (outside the sample; list at the end), so the restricted "
          "group here is not a sample of all restricted-entry CS programs, and the result applies to the scripts/58 "
          "sample only.")
    w("")
    w("Coding describes current policy (pages accessed " + ", ".join(R["coding_info"]["access_dates"]) + "). The "
      f"Scorecard earnings cohorts entered around fall {ce['entry']['y5'][0]}–{ce['last_entry']} if they completed in "
      "four years (data dictionary: "
      + "; ".join(f"{k}: {v.split(' pooled')[0]}" for k, v in ce["text"].items()) + "); slower completers entered "
      f"earlier. A policy adopted since then did not apply to them; {ch['dated']} of the {ch['coded']} coded programs "
      "of the primary sample have a page that dates the policy.")
    w("")
    w("## Key numbers")
    w("")
    w("Slopes: log points of median earnings per within-CS SD of department prestige, from the scripts/58 main "
      "within-institution model with the CS slope split by entry policy. CI = 95% percentile institution cluster "
      f"bootstrap (B = {B}); p = two-sided percentile-bootstrap p; n = CS programs. The unit of each row is in "
      "`data/interim/cs_admission_estimates.csv`.")
    w("")
    w("| quantity | sample / spec | n | estimate [95% CI] | p |")
    w("|---|---|---|---|---|")
    for _, r in E.iterrows():
        est = r.estimate
        if r.unit in ("count", "indicator"):
            val = f"{int(est)}"
        elif r.unit == "probability":
            val = pct(est)
        elif r.section.startswith("descriptive"):
            val = f"{est:.1f}" if abs(est) >= 100 else f"{est:.3f}"
        elif np.isfinite(r.lo):
            val = f"{est:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]"
        elif r.section in ("T1 power",) or r.stat.startswith(("SE of", "minimum detectable")):
            val = f"{est:.4f}"
        else:
            val = f"{est:+.4f}" if abs(est) < 10 else f"{est:.1f}"
        pv = ("" if not np.isfinite(r.p) else f"< {2 / B:.3f}" if r.p == 0 else f"{r.p:.3f}")
        n_ = "" if not np.isfinite(r.n) else f"{int(r.n)}"
        w(f"| {r.section}: {r.stat.replace('|', '∣')} | {r['sample']}{'; ' + r['note'] if r['note'] else ''} | {n_} | "
          f"{val} | {pv} |")
    w("")
    w("## Method")
    w("")
    w("- **What was fixed in advance.** The task (2026-09-24) fixed the question (does the CS department remainder of "
      "scripts/58 differ between restricted and open programs), a bootstrap CI and power, and the rule that the test "
      "runs only if at least 60% of the programs are coded. The operationalisation is this lane's own and is written "
      "in the script docstring: reading 'scripts/58 partial' as the within-institution per-field slope β_f (T1), with "
      "the partial-correlation reading as T2; the slope split with CS × group intercepts; the six sensitivities; the "
      "rule that a replicate needs at least 5 distinct institutions per group. The script was not under version "
      "control, so whether these choices preceded the results cannot be verified from a version record. "
      + ("Both readings (T1 and T2) give a CI for the difference that includes zero. " if R["run_tests"] and all(
          R[k]["D"]["lo"] <= 0 <= R[k]["D"]["hi"] for k in ("T2",)) and
         R["T1"]["primary"]["D"]["lo"] <= 0 <= R["T1"]["primary"]["D"]["hi"] else "")
      + "Everything labelled exploratory was chosen after the fact.")
    w("- **Sample.** The CS programs of scripts/58's within-institution design: primary = the common programs where the "
      "4-, 1- and 5-yr Scorecard medians are all released (FE2_h3; the sample of the headline CS slope); sensitivity = "
      "the 4-yr main sample (FE2). Rebuilt by importing scripts/58 / scripts/55; the CS slope (point, each horizon) and "
      "its horizon-averaged percentile CI are reproduced exactly (asserted against `data/interim/selectivity_deep.csv`).")
    w("- **Coding rules** (script docstring; applied uniformly after Revision 1). direct_admit: an official page says "
      "entry to CS (or its college) for students already enrolled is competitive, decided by review, space-limited, "
      "closed or not guaranteed at the published minimums, and a first-year route decided with the university "
      "application is documented; capped: the same, with no documented first-year route; open: enrolled students may "
      "declare, or enter on fixed published minimums, with no competition, review or capacity statement; programs "
      "whose pages document only first-year entry to the college (by criteria, or by review with a pre-major route "
      "for the rest) are coded open and flagged ambiguous; unknown: no official statement found. ambiguous marks an "
      "application or request with minimum criteria but no guarantee or competition statement, a guaranteed threshold "
      "with review below it, a discretionary denial clause, a generic statement, or a page that documents only "
      "first-year entry. Each row is verified at run time against its saved page (md5, URL, access date, HTTP 200, "
      "verbatim quote). Pages were found by browsing official sites (catalog, department, college and admissions "
      "pages, their links and sitemaps); pages not reachable that way were not found.")
    w("- **T1.** In the scripts/58 per-field FE2 design the CS slope column is replaced by one slope per group "
      "(restricted, open, unknown / uncoded) plus CS × group intercepts (open = reference). Δ = β_R − β_O, averaged "
      "over the three horizons (same design matrix, three outcomes). Institution pairs cluster bootstrap with one "
      f"SeedSequence child per replicate (seed [{SEED}, 1]); a replicate is used if it draws ≥ {MIN_GRP} distinct "
      f"restricted and open CS institutions and ≥ {MIN_CS} CS institutions. Power from the normal approximation with "
      "the SE taken as the half-width of the 95% percentile CI / 1.96; the SD of the replicates and an independent "
      f"seed ([{SEED}, 3]) are reported as Monte Carlo checks.")
    w("- **T2.** Rank partial correlation as scripts/58 F_broadG (ranks of prestige, earnings, SAT, admit rate, "
      "institution Pell share, state earnings level and G among the CS programs; CONTROL dummies); residuals from all CS "
      "programs of the sample, correlated within each group; bootstrap over CS programs (one per institution) with ranks "
      "recomputed in each replicate.")
    w(f"- **Exploratory three-group split.** As T1 with separate direct-admit, capped and open slopes; a replicate is "
      f"used if it draws ≥ {MIN_GRP_EXPL} distinct institutions of each group (the least that over-identifies a group's "
      "slope and intercept).")
    w("- **Sample diagnostic.** Wapman CS departments with a prestige score are followed through the scripts/55 merge "
      "and the scripts/58 sample rules; those not in the cells are matched to Scorecard CS programs with a released "
      "4-yr median by the scripts/34 name rules (exact, then campus suffix stripped, then city / branch qualifier "
      "stripped; non-branch campus, then largest cohort, preferred). Matches by the city rule are listed for audit. "
      "Name forms these rules miss (for example 'Rutgers - New Brunswick', 'Stony Brook University, State University "
      "of New York') are counted as not matched.")
    w("- **Two-way clustering.** Δ is a within-field contrast; other fields enter only through the shared institution "
      "fixed effects, so the institution cluster bootstrap of scripts/58 is used rather than the two-way (field, "
      "institution) variance of scripts/59, which is for statistics averaged over fields.")
    w("")
    w("## Caveats")
    w("")
    w("- **Current policy, older cohorts.** Pages were read in September 2026; the earnings cohorts entered in "
      f"fall {ce['entry']['y5'][0]}–{ce['last_entry']} if they completed on time (the entry years assume four-year "
      "completion; slower completers entered earlier). Several restrictions are recent (pages that date them are "
      "listed in the table below); the dated-policy sensitivity drops them, but undated policies may also be recent.")
    wb = R["coding_info"]["wayback"]
    if wb:
        w("- **Archived snapshots.** " + "; ".join(f"{x['instnm']} is coded from an Internet Archive snapshot dated "
                                                     f"{x['snapshot']} (retrieved {x['access']})" for x in wb)
          + ", because the live site blocked scripted access. The access date in the coding is the retrieval date, "
            "not the date of the page.")
    w("- **Coding is coarse.** 'Open' includes programs with fixed GPA / course gateways, which also select students "
      "somewhat; 'restricted' merges first-year direct admission with competitive later entry. Some pages describe a "
      "college-wide or university-wide rule (scope column).")
    w("- **Only CS was coded.** The task's optional extension to engineering and business programs was not done.")
    w("- **Groups differ in more than policy.** Restricted programs sit at different institutions (see the descriptive "
      "rows); the model nets out field-specific returns to SAT, admit rate, Pell share and brand, but other differences "
      "between the institutions remain. Descriptive association only.")
    w("- **Small groups.** The restricted group is small, so Δ is imprecise (power above).")
    sat_names = "; ".join(top.loc[top.status == "dropped: SAT_AVG missing", "wapman"])
    w(f"- **The sample leaves out many prominent CS departments.** The scripts/58 CS sample inherits two exclusions from "
      f"scripts/55: the exact-name merge of Wapman and Scorecard loses institutions whose Scorecard name carries a campus "
      f"or city qualifier ({n_merge_top} of the top {TOPN} Wapman CS departments by the scripts/34 rules, and "
      f"{int((top.status == ST_NOMATCH).sum())} more whose names these rules do not match; list at the end), and the "
      f"SAT control drops institutions with no SAT_AVG in the 2026 institution file ({n_sat_top} of the top {TOPN}: "
      f"{sat_names}). This lane codes the sample as given (fixing the merge is a change to scripts/55 and scripts/58, "
      "outside this lane), so its conclusion applies to the scripts/58 sample and not to CS programs in general.")
    if len(R["oos"]):
        w(f"- **Coded rows outside the samples.** {len(R['oos'])} coded institutions are in neither analysis sample: "
          + "; ".join(f"{', '.join(f'{r.instnm} ({r.policy})' for _, r in g.iterrows())} -- {why}"
                      for why, g in R["oos"].groupby("why", sort=True))
          + ". Their codes are kept in the coding file for reference and enter no statistic.")
    if R["run_tests"]:
        o = R["T1"]["primary"]
        w(f"- **Coverage is not random.** {cv['h3']['unknown'] + cv['h3']['uncoded']} programs of the primary sample have "
          "no official statement that could be found (the official pages reached state no entry rule); they keep "
          f"their own slope in the model (β_U = {ci(o['U'])}) and do not enter Δ. The later part of the coding was done by "
          "browsing official sites (catalog, department, college and admissions pages, their links and sitemaps) after "
          "the web-search budget ran out, so pages not linked from those sites were not found; an uncoded program is "
          "not evidence that its entry is open.")
    w("")
    if R["run_tests"]:
        pv, t = R["prev"], R["T1"]["primary"]
        p5, p3 = R["three_prev"][MIN_GRP]["s"], R["three_prev"][MIN_GRP_EXPL]["s"]
        cpu = " and ".join(f"{u} ({d_})" for u, d_ in (("UCCS", "126580"), ("UW-Milwaukee", "240453")))
        w("## Revisions")
        w("")
        w("**Revision 1 (2026-09-25), after an independent check of the first version.** Every number below is "
          "computed by the current script; 'previous' numbers come from applying the pre-revision values of the changed "
          "coding cells (listed in the script as REV1) to the same bootstrap draws.")
        w("")
        nD = R["three_prev"][MIN_GRP]["n"].get("D", 0)
        ex5 = not (p5["D"]["lo"] <= 0 <= p5["D"]["hi"])
        in3 = p3["D"]["lo"] <= 0 <= p3["D"]["hi"]
        w(f"1. **Exploratory three-group CIs were conditional.** The first version required ≥ {MIN_GRP} distinct "
          f"institutions of each group per replicate. On the previous coding ({nD} direct-admit programs) that kept "
          f"{p5['D']['n_ok']} of {B} replicates, each with at least {MIN_GRP} of the {nD} direct-admit institutions, so "
          f"the interval was conditional on that: direct-admit {ci(p5['D'])}. With ≥ {MIN_GRP_EXPL} institutions on the "
          f"same coding, {p3['D']['n_ok']} of {B} replicates are usable and direct-admit is {ci(p3['D'])} "
          f"(direct-admit minus open {ci(p3['D-O'])})"
          + ("; the exclusion of zero came from the rule" if (ex5 and in3) else "")
          + f". The rule is now ≥ {MIN_GRP_EXPL}, usable counts are printed for every interval, and the split is "
          "labelled not a finding (answer 6).")
        w("2. **Sample representativeness.** Added the sample diagnostic (answer 8, the caveat and the list at the "
          "end) and the explanation of the coded rows outside both samples. The scripts/55 name merge itself is "
          "unchanged (outside this lane).")
        w(f"3. **Coding consistency.** One rule for programs whose pages document only first-year college entry: open, "
          f"flagged ambiguous. {cpu} move from direct_admit to open (both were already flagged ambiguous). The ambiguous "
          "definition (application or request with minimum criteria and no guarantee or competition statement) is now "
          "applied to every row: Michigan State, Missouri S&T, Kansas, Northeastern, Texas Tech, George Washington "
          "and UC Riverside (outside the samples) are now flagged ambiguous; Kansas State's note now reports the "
          "page's seat-limit clause. The primary estimate on the previous coding was β_R = "
          f"{ci(pv['R'])} (n_R = {pv['n'].get('R', 0)}), Δ = {ci(pv['D'])}; on the revised coding β_R = {ci(t['R'])} "
          f"(n_R = {t['n'].get('R', 0)}), Δ = {ci(t['D'])}. The sensitivity rows that use the ambiguous flag change "
          "accordingly"
          + (f"; the variant without intercepts moves from {ci(R['prev_nolevel']['D'])} to {ci(T['no_level']['D'])}"
             if excl0(R["prev_nolevel"]["D"]) != excl0(T["no_level"]["D"]) else "")
          + "." + (" Both intervals for the primary Δ include zero." if (pv["D"]["lo"] <= 0 <= pv["D"]["hi"] and
                                                                        t["D"]["lo"] <= 0 <= t["D"]["hi"]) else ""))
        w(f"4. **Coverage.** The Answer now reports coverage without ambiguous codes ({cv['h3']['unamb']} programs, "
          f"{100 * cv['h3']['share_unamb']:.0f}%) and without ambiguous or institution-wide codes "
          f"({cv['h3']['unamb_ni']}, {100 * cv['h3']['share_unamb_ni']:.0f}%), and says that the 60% threshold is met "
          "only by counting ambiguous codes.")
        w("5. **Pre-specification.** The Method and the docstring now separate what the task fixed (the question, a "
          "bootstrap CI and power, the 60% rule) from this lane's operationalisation, which cannot be dated from a "
          "version record.")
        w(f"6. **Power.** Power and minimum detectable effects now use the half-width of the 95% percentile interval / "
          "1.96 as the SE instead of the SD of the replicates, which is more sensitive to their thick tails; both are "
          f"reported for two seeds (percentile SE {R['power']['se']:.3f} and {R['mc']['D']['se']:.3f}; SD "
          f"{R['power']['se_sd']:.3f} and {R['mc']['D']['se_sd']:.3f}; power at the pooled slope "
          f"{pct(R['power']['pow_full'])} and {pct(R['mc']['pow_full'])}).")
        w("7. **Framing.** The key question now asks about a slope difference, and answer 5 says what T1 can and "
          "cannot show about selection into the major.")
        w("8. **Disclosure.** Added: the coded rows outside the samples, that engineering and business were not coded, "
          "the Internet Archive snapshot dates, the on-time-completion assumption behind the entry years, a note that "
          "the two CIs of the pooled CS slope come from different bootstrap draws, and percentages for power.")
        w("")
    w("## Coding table (primary sample)")
    w("")
    w("| UNITID | institution | policy | gate | scope | ambiguous | page dates policy | URL (accessed) |")
    w("|---|---|---|---|---|---|---|---|")
    cod = pd.read_csv(CODING, dtype=str, keep_default_na=False).set_index("unitid")
    for _, r in P[P["sample"] == "h3"].sort_values("instnm").iterrows():
        if r.unitid in cod.index and cod.loc[r.unitid, "policy"] == "unknown":
            w(f"| {r.unitid} | {r.instnm} | unknown | | | | | no official statement found |")
        elif r.unitid in cod.index:
            c = cod.loc[r.unitid]
            w(f"| {r.unitid} | {r.instnm} | {c.policy} | {c.gate} | {c.scope} | {c.ambiguous} | "
              f"{c.policy_start if c.policy_start_stated == 'yes' else 'no'} | {c.url} ({c.access_date}) |")
        else:
            w(f"| {r.unitid} | {r.instnm} | uncoded | | | | | |")
    w("")
    w(f"Coding file md5 `{R['coding_info']['coding_md5']}`; manifest md5 `{R['coding_info']['manifest_md5']}`; "
      f"{R['coding_info']['n_pages']} saved pages ({R['coding_info']['n_pages_used']} cited).")
    w("")
    w(f"## Top {TOPN} Wapman CS departments not in the primary sample (exploratory diagnostic)")
    w("")
    w("Rank = rank of the Wapman CS prestige score among the departments with one (1 = most prestigious). "
      "'SAT_AVG' = whether the institution file has SAT_AVG for the matched Scorecard institution (name-merge losses "
      "only). 'policy' = code in this lane's coding file, if any.")
    w("")
    w("| rank | Wapman department | status | Scorecard match | UNITID | SAT_AVG | policy |")
    w("|---|---|---|---|---|---|---|")
    for _, r in top_miss.iterrows():
        w(f"| {r.wrank} | {r.wapman} | {r.status} | {r.scorecard} | {r.unitid} | {r.sat_if_matched} | {r.policy} |")
    OUT_MD.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="debug only: B = 49 and s58 reproduction CI skipped; "
                    "outputs go to the scratch paths below, never to the result files")
    a = ap.parse_args()
    if a.quick:
        global B, OUT_PROG, OUT_EST, OUT_MD, QUICK
        B, QUICK = 49, True
        tmp = Path(os.environ.get("TMPDIR", "/tmp"))
        OUT_PROG, OUT_EST, OUT_MD = tmp / "q74_programs.csv", tmp / "q74_estimates.csv", tmp / "q74_result.md"
    avail = mem_available_mb()
    workers = 1 if avail < 1500 else max(1, min(a.workers, 6))
    print(f"MemAvailable {avail:.0f} MB -> {workers} worker(s)")
    R = compute(workers)
    write(R)
    cv = R["cov"]
    for smp in ("h3", "y4"):
        print(smp, cv[smp])
    print("tests run:", R["run_tests"])
    print(R["gaps"].status.value_counts().to_string())
    if R["run_tests"]:
        for k, o in R["T1"].items():
            print(f"T1 {k}: n {o['n']} Delta {o['D']['est']:+.4f} [{o['D']['lo']:+.4f}, {o['D']['hi']:+.4f}] "
                  f"R {o['R']['est']:+.4f} O {o['O']['est']:+.4f} ok {o['D']['n_ok']}")
        print("prev", {k: round(R["prev"][k]["est"], 4) for k in ("R", "O", "D")})
        print("power", R["power"])
        print("mc", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in R["mc"]["D"].items()})
        print("three", {g: (round(s["est"], 4), round(s["lo"], 4), round(s["hi"], 4), s["n_ok"]) for g, s in R["three"].items()})
        for mg, o in R["three_prev"].items():
            print("three_prev", mg, {g: (round(s["est"], 4), round(s["lo"], 4), round(s["hi"], 4), s["n_ok"])
                                    for g, s in o["s"].items()})
        print("T2", {k: (round(v["est"], 4), round(v["lo"], 4), round(v["hi"], 4)) for k, v in R["T2"].items()
                     if isinstance(v, dict) and "est" in v})
    print(f"wrote {OUT_PROG}, {OUT_EST}, {OUT_MD}")


if __name__ == "__main__":
    main()
