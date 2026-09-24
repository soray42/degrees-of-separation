"""PSEO refresh -- the August 2026 PSEO release (V4.14.1 / 2026Q2) against the repo's V4.13.0 (2025Q4).

QUESTION. Census released a new PSEO in August 2026 (1,117 institutions vs 952). Does it change any
Phase 1 conclusion that rests on PSEO, and does prestige-earnings coupling rise with the earnings
quantile (p75 coupling > median coupling, a within-program upper-tail signal)?

WHAT THIS SCRIPT DOES
  (0) Provenance: md5, size, version strings, row / institution / state counts of both releases;
      a cell-level diff of the rows the two releases share (revisions).
  (a) Coverage change: which Wapman institutions are new (canonical name join of
      src/crosswalks/institutions.py; plus an explicit alias sensitivity for four new institutions the
      name join misses: Rutgers New Brunswick / Newark / Camden and UT Knoxville), and how many released
      bachelor's institution x CIP-4 earnings cells they add, by horizon and cohort. Wapman coverage by
      academia-rank quartile (this script's definition, and README section 1's scripts/01 definition).
  (b) The canonical fixed-cohort career-time analysis of scripts/52 (functions imported, not edited;
      scripts/52 module globals are redirected to the chosen release), run on four data variants:
        old         = V4.13.0 (repo file)                       -> must reproduce scripts/52
        new_oldinst = V4.14.1 restricted to V4.13.0 institutions -> revisions only
        new         = V4.14.1, canonical join                    -> the refresh (primary)
        new_alias   = V4.14.1 + the 4 aliases for added institutions
        new_joinfix = V4.14.1 + those 4 + 19 aliases for V4.13.0 institutions the name join misses
                      (JOIN_AUDIT; a sensitivity, not the canonical spec)
      All-field and integrated slopes, integrated-minus-others, y5->y10 segment, calendar-matched bound,
      coverage control, field prestige F vs academia-wide rank G loadings. Besides scripts/52's own
      two-stage CIs (kept so that V4.13.0 reproduces scripts/52), every statistic gets the same
      institution-clustered inference as (c): a crossed field x institution bootstrap and an
      institution-cluster bootstrap with fields fixed (one institution multinomial draw shared by every
      panel, coverage and calendar cell), and the slopes an institution-clustered horizon-label permutation.
  (c) Quantile coupling at fixed cohorts: Spearman(F, p25/p50/p75 earnings) by horizon on a balanced
      panel (same construction as (b)). Inference for p75-p50 and p25-p50 respects that the same
      institutions recur across cohorts and fields: an institution-clustered quantile-label permutation
      (one flip per institution, shared by all its field x cohort cells and horizons), a crossed
      field x institution bootstrap (fields resampled; one institution multinomial draw applied to every
      cell; Owen 2007) and an institution-cluster bootstrap with fields fixed. The earlier paired
      two-stage bootstrap (institutions redrawn separately per cell) and the per-cell independent-flip
      permutation are kept for comparison. Aggregation sensitivity (Fisher z, n-weighted cells, cells
      with n>=30, CIP-4 cells without within-field aggregation). Spearman(F, p75/p50) (upper-tail
      stretch), and the same by academia-wide rank G.
  (d) Cross-source replication of scripts/32 (Scorecard 4YR vs PSEO y5 pooled field coupling, and the
      institution-level earnings agreement) on each variant; scripts/32's field bootstrap plus a crossed
      field x institution bootstrap (one institution draw shared by the PSEO and Scorecard couplings).
  Inference standard (revision 3): for every statistic of (b)-(d), the two-way (field, institution) cluster
      variance of Cameron-Gelbach-Miller (2011), V_field + V_institution - V_field x institution, from the
      between-field variance, the fields-fixed replicates of the shared institution draw, and those of an
      independent institution draw per field. A calibration under a within-cell permutation null on the V4.14.1
      panel compares each interval's variance with the true sampling variance.

Descriptive and not causal. Seeded; outputs byte-identical on re-run.
Run: python scripts/59_pseo_refresh.py
Outputs: data/interim/pseo_refresh.csv, outputs/figures/pseo_refresh.png, PSEO_REFRESH_RESULT.md (root,
         local only).
"""
from __future__ import annotations

import sys
import time
import itertools
import zlib
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_ar import load_ar_wapman
from src.load_er import (load_er_scorecard, load_er_pseo, _pseo_institutions, PSEO_EARN, PSEO_INST,
                         PSEO_DEGREE_LEVEL)
from src.gap import compute_gap_map
from src.crosswalks import fields as F
from src.crosswalks.institutions import normalize_institution_name


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s52 = _load("s52", "52_career_time_coupling.py")      # canonical fixed-cohort engine (not edited)
s32 = _load("s32", "32_pseo_gap_crossval.py")         # cross-source replication loader (not edited)
s28 = s52._s28
FIELDS66, LAB = s52.FIELDS66, s52.LAB
COHORTS, LATE, HZ, YRS, W_SLOPE, NMIN = s52.COHORTS, s52.LATE, s52.HZ, s52.YRS, s52.W_SLOPE, s52.NMIN
rk, corr_rows, ci, perm_p, two_stage = s52.rk, s52.corr_rows, s52.ci, s52.perm_p, s52.two_stage

SEED = 59
# Replicates of THIS script's own resampling (quantile analysis, section c): capped at 1000 (shared-VM
# resource rule). The scripts/52 engine used in section (b) keeps its own constants (NBOOT=2000,
# NPERM=2000, NPERM_FIELD=10000) so that the V4.13.0 run reproduces scripts/52 exactly, CIs included.
NBOOT = 1000
NPERM = 1000
CHUNK = 200_000          # rows per chunk when this script reads a pseoe_all.csv.gz itself
QS = ["p25", "p50", "p75"]
AVG = "mean y1/y5/y10"   # post hoc summary over horizons (not in the spec; reported as such)
NMIN_BIG = 30            # aggregation sensitivity: cells with at least this many institutions
OLD_DIR = ROOT / "data" / "raw" / "pseo"
NEW_DIR = ROOT / "data" / "raw" / "pseo_2026q2"
NEW_EARN = NEW_DIR / "pseoe_all.csv.gz"
NEW_INST = NEW_DIR / "pseo_all_institutions.csv"
OUT_CSV = ROOT / "data" / "interim" / "pseo_refresh.csv"
OUT_FIG = ROOT / "outputs" / "figures" / "pseo_refresh.png"
OUT_MD = ROOT / "PSEO_REFRESH_RESULT.md"

# PSEO normalized name -> Wapman normalized name, for new-release institutions the canonical join misses.
# Identified by hand (same institution, different surface form); used ONLY in the *_alias variant.
ALIAS = {"rutgers state university new jersey new brunswick campus": "rutgers new brunswick",
         "rutgers state university new jersey newark campus": "rutgers newark",
         "rutgers state university new jersey camden campus": "rutgers university camden",
         "university tennessee knoxville": "university tennessee"}

# Pre-existing (V4.13.0) PSEO institutions that are Wapman institutions under another name and that the
# canonical join misses (PSEO id -> Wapman InstitutionName). Used ONLY in the new_joinfix sensitivity; the
# canonical join (src/crosswalks/institutions.py) is left unchanged.
JOIN_AUDIT = {
    "00283800": "Stony Brook University, State University of New York",
    "00283700": "University at Buffalo, State University of New York",
    "00283500": "University at Albany, State University of New York",
    "00186900": "Iowa State University",
    "00251600": "University of Missouri",
    "00201000": "Louisiana State University",
    "00344800": "University of South Carolina",
    "00161000": "University of Hawaii",
    "00253200": "Montana State University",
    "00269000": "Queens College (CUNY)",
    "00268800": "City College (CUNY)",
    "00268700": "Brooklyn College (CUNY)",
    "00727300": "Baruch College (CUNY)",
    "00702200": "Lehman College (CUNY)",
    "00269300": "John Jay College of Criminal Justice (CUNY)",
    "00495200": "University of Texas Medical Branch, The",
    "00165900": "Rosalind Franklin University",
    "00363900": "Texas A&M Kingsville",
    "01116100": "Texas A&M Corpus Christi",
}

URL_BASE = "https://lehd.ces.census.gov/data/pseo/latest_release/all/"
LAST_MODIFIED = "Tue, 25 Aug 2026 21:07:46 GMT"   # HTTP header of pseoe_all.csv.gz at download
DOWNLOAD_DATE = "2026-09-23"
# Completeness check of the national earnings file (curl -sI on URL_BASE + pseoe_all.csv.gz,
# response Date: Wed, 23 Sep 2026 22:33:12 GMT): Content-Length and Last-Modified as below. The
# directory holds one national earnings file (pseoe_all.csv.gz, listed as 13M), so the local file is
# the complete release if its size equals SERVER_CONTENT_LENGTH (asserted in provenance()).
SERVER_CONTENT_LENGTH = 13_895_857
# re-checked 2026-09-24 11:51:55 GMT (same Content-Length and Last-Modified; the directory listing again
# shows one earnings file, pseoe_all.csv.gz 13M, besides pseof_all.csv.gz 187M and three small files)
HEAD_CHECK = "2026-09-23 22:33 GMT, re-checked 2026-09-24 11:51 GMT"

REL = {
    "old": dict(label="V4.13.0 (2025Q4, repo)", earn=PSEO_EARN, inst=PSEO_INST, alias=None),
    "new_oldinst": dict(label="V4.14.1, V4.13.0 institutions only", earn=NEW_EARN, inst=PSEO_INST, alias=None),
    "new": dict(label="V4.14.1 (2026Q2)", earn=NEW_EARN, inst=NEW_INST, alias=None),
    "new_alias": dict(label="V4.14.1 + 4 name aliases", earn=NEW_EARN, inst=NEW_INST, alias=ALIAS),
    "new_joinfix": dict(label="V4.14.1 + 23 name aliases", earn=NEW_EARN, inst=NEW_INST, alias=None),  # set in main
}
VARIANTS = list(REL)


def rng_for(tag: str) -> np.random.Generator:
    return np.random.default_rng([SEED, zlib.crc32(tag.encode())])


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


ROWS: list[dict] = []   # every number that goes to the CSV


def rec(section, stat, estimate, variant="", lo=np.nan, hi=np.nan, p=np.nan, k=np.nan, n=np.nan,
        horizon="", quantile="", field="", sample="", note=""):
    ROWS.append(dict(section=section, variant=variant, stat=stat, field=field, horizon=horizon,
                     quantile=quantile, estimate=estimate, ci_lo=lo, ci_hi=hi, p_value=p, k=k, n=n,
                     sample=sample, note=note))


# ---------------------------------------------------------------------------------------------
# release plumbing
# ---------------------------------------------------------------------------------------------
def alias_keys(d: pd.DataFrame, amap: dict) -> pd.DataFrame:
    return d.assign(inst_key=d["inst_key"].replace(amap)) if amap else d


def alias_full() -> dict:
    """ALIAS plus the JOIN_AUDIT pairs, keyed by the PSEO institution's normalized name."""
    ins = _pseo_institutions(NEW_INST).drop_duplicates("institution").set_index("institution")
    extra = {ins.loc[pid, "inst_key"]: normalize_institution_name(w) for pid, w in JOIN_AUDIT.items()}
    assert not set(extra) & set(ALIAS)
    return {**ALIAS, **extra}


def institutions(v: str) -> pd.DataFrame:
    r = REL[v]
    d = _pseo_institutions(r["inst"])
    return alias_keys(d, r["alias"])


def er_pseo(v: str, level="undergrad", horizon="y5", fields=None, grad_cohort="0000"):
    r = REL[v]
    d = load_er_pseo(level, horizon, fields=fields, grad_cohort=grad_cohort, path=r["earn"],
                     inst_path=r["inst"])
    return alias_keys(d, r["alias"])


def set_release(v: str):
    """Redirect the scripts/52 module's data globals to variant v (its code is unchanged)."""
    s52.PSEO_EARN = REL[v]["earn"]
    s52.load_er_pseo = lambda level="undergrad", horizon="y5", fields=None, grad_cohort="0000": \
        er_pseo(v, level, horizon, fields, grad_cohort)
    s52._pseo_institutions = lambda: institutions(v)


_RAW = {}


def raw_ba(earn: Path) -> pd.DataFrame:
    """Bachelor's, institution-level, CIP-4 rows (all cohorts, all statuses) of an earnings file."""
    if earn not in _RAW:
        cols = ["inst_level", "institution", "degree_level", "cip_level", "cipcode", "grad_cohort"]
        for h in HZ:
            cols += [f"{h}_p25_earnings", f"{h}_p50_earnings", f"{h}_p75_earnings", f"{h}_grads_earn",
                     f"status_{h}_earnings"]
        parts = []
        for ch in pd.read_csv(earn, dtype=str, usecols=cols, chunksize=CHUNK):
            parts.append(ch[(ch["inst_level"] == "I") & (ch["cip_level"] == "4") &
                            (ch["degree_level"] == PSEO_DEGREE_LEVEL["undergrad"])]
                         .drop(columns=["inst_level", "cip_level", "degree_level"]))
        d = pd.concat(parts, ignore_index=True)
        del parts
        d["cip4"] = d["cipcode"].astype(str).str.replace(".", "", regex=False).str.zfill(4)
        d["field"] = d["cip4"].map(F.cip4_to_field_key(FIELDS66))
        _RAW[earn] = d.reset_index(drop=True)
    return _RAW[earn]


# ---------------------------------------------------------------------------------------------
# (0) provenance
# ---------------------------------------------------------------------------------------------
def provenance():
    out = {}
    for tag, d in (("old", OLD_DIR), ("new", NEW_DIR)):
        info = {}
        for fn in ("pseoe_all.csv.gz", "pseo_all_institutions.csv", "version_pseo.txt", "pseo_all_partners.txt"):
            p = d / fn
            if p.exists():
                info[fn] = dict(md5=md5(p), size=p.stat().st_size)
                rec("provenance", f"md5:{fn}", np.nan, tag, note=info[fn]["md5"], n=info[fn]["size"])
        vl = [ln.split() for ln in (d / "version_pseo.txt").read_text().splitlines() if ln.strip()]
        vers = sorted({(x[4], x[5]) for x in vl})
        info["version"] = " / ".join(f"{a} {b}" for a, b in vers)
        info["version_states"] = sorted({x[1] for x in vl if x[0] == "PSEOE"})
        info["years"] = {x[1]: x[3] for x in vl if x[0] == "PSEOE"}
        info["stamps"] = sorted({"_".join(x[6].split("_")[2:]) for x in vl if len(x) > 6})
        info["ncols"] = len(pd.read_csv(d / "pseoe_all.csv.gz", dtype=str, nrows=1).columns)
        nrow, nrow_i, insts = 0, 0, set()
        for ch in pd.read_csv(d / "pseoe_all.csv.gz", dtype=str, usecols=["inst_level", "institution"],
                              chunksize=CHUNK):
            nrow += len(ch)
            isI = ch.inst_level == "I"
            nrow_i += int(isI.sum())
            insts |= set(ch.loc[isI, "institution"])
        ins = pd.read_csv(d / "pseo_all_institutions.csv", dtype=str)
        ins.columns = [c.strip().lstrip("﻿") for c in ins.columns]
        info.update(rows=nrow, rows_I=nrow_i, inst_earn=len(insts),
                    inst_file=len(ins), states=sorted(ins.institution_state.unique()))
        for k in ("rows", "rows_I", "inst_earn", "inst_file"):
            rec("provenance", k, info[k], tag)
        if tag == "new":   # the local national earnings file is the complete one on the server
            assert info["pseoe_all.csv.gz"]["size"] == SERVER_CONTENT_LENGTH, info["pseoe_all.csv.gz"]
        rec("provenance", "version", np.nan, tag, note=info["version"])
        rec("provenance", "n_jurisdictions_excl_US", len([s for s in info["states"] if s != "US"]), tag,
            note=",".join(info["states"]))
        out[tag] = info
    # cell-level diff of shared rows (bachelor's, institution level, CIP-4)
    o, n = raw_ba(PSEO_EARN), raw_ba(NEW_EARN)
    key = ["institution", "cipcode", "grad_cohort"]
    assert not o.duplicated(key).any() and not n.duplicated(key).any()
    m = o.merge(n, on=key, how="outer", suffixes=("_o", "_n"), indicator=True)
    both = m[m["_merge"] == "both"]
    diff = dict(shared=len(both), old_only=int((m["_merge"] == "left_only").sum()),
                new_only=int((m["_merge"] == "right_only").sum()))
    for h in HZ:
        so, sn = both[f"status_{h}_earnings_o"], both[f"status_{h}_earnings_n"]
        rel = (so == "1") & (sn == "1")
        p_o = pd.to_numeric(both.loc[rel, f"{h}_p50_earnings_o"])
        p_n = pd.to_numeric(both.loc[rel, f"{h}_p50_earnings_n"])
        diff[f"{h}_released_both"] = int(rel.sum())
        diff[f"{h}_p50_changed"] = int((p_o != p_n).sum())
        diff[f"{h}_status_changed"] = int((so.fillna("") != sn.fillna("")).sum())
        diff[f"{h}_newly_released"] = int(((so != "1") & (sn == "1")).sum())
        diff[f"{h}_newly_suppressed"] = int(((so == "1") & (sn != "1")).sum())
    for k, v in diff.items():
        rec("revisions", k, v, "old_vs_new", sample="bachelor's, institution x CIP-4 rows")
    out["diff"] = diff
    return out


# ---------------------------------------------------------------------------------------------
# (a) coverage change
# ---------------------------------------------------------------------------------------------
def released_cells(v: str, ar: pd.DataFrame) -> pd.DataFrame:
    """Long table: institution x CIP-4 x cohort x horizon rows with released earnings, flagged usable
    (CIP-4 in FIELDS66 and the institution has a Wapman rank in that field)."""
    r = REL[v]
    d = raw_ba(r["earn"])
    ins = institutions(v).drop_duplicates("institution")
    d = d.merge(ins, on="institution", how="inner")
    d = d[d["inst_key"].notna() & (d["inst_key"] != "")]
    ark = set(zip(ar.inst_key, ar.field))
    out = []
    for h in HZ:
        x = d[(d[f"status_{h}_earnings"] == "1") & d[f"{h}_p50_earnings"].notna()]
        x = x[["institution", "label", "inst_key", "cip4", "field", "grad_cohort"]].assign(horizon=h)
        out.append(x)
    x = pd.concat(out, ignore_index=True)
    x["usable"] = [isinstance(f, str) and (k, f) in ark for k, f in zip(x.inst_key, x.field)]
    return x


def coverage(ar: pd.DataFrame, gen: pd.DataFrame):
    ins_o, ins_n = institutions("old"), institutions("new")
    new_ids = sorted(set(ins_n.institution) - set(ins_o.institution))
    rec("coverage", "institutions_added", len(new_ids), "old_vs_new")
    newi = ins_n[ins_n.institution.isin(new_ids)]
    canon = newi.merge(gen, on="inst_key")
    ali = alias_keys(newi, ALIAS).merge(gen, on="inst_key")
    ali = ali[~ali.institution.isin(canon.institution)]
    assert set(ALIAS.values()) <= set(gen.inst_key), "alias target not a Wapman institution"
    assert len(ali) == len(ALIAS), "every alias should add exactly one institution"
    cells = {v: released_cells(v, ar) for v in ("old", "new", "new_alias", "new_joinfix")}
    ca = cells["new_alias"]
    tab = []
    for join, dfj in (("canonical", canon), ("alias", ali)):
        for _, r in dfj.sort_values("g_rank").iterrows():
            c = ca[ca.institution == r.institution]
            row = dict(institution=r.institution, label=r.label, g_rank=int(r.g_rank), join=join,
                       n_wapman_fields=int((ar.inst_key == r.inst_key).sum()))
            for h in HZ:
                ch = c[c.horizon == h]
                row[f"cells_{h}"] = len(ch)
                row[f"usable_{h}"] = int(ch.usable.sum())
                row[f"usable_fixed_{h}"] = int((ch.usable & ch.grad_cohort.isin(COHORTS)).sum())
                row[f"cohorts_{h}"] = "+".join(sorted(ch[ch.usable].grad_cohort.unique()))
            tab.append(row)
    tab = pd.DataFrame(tab)
    tab["state"] = tab.institution.map(dict(zip(ins_n.institution, _state_of(NEW_INST))))
    for _, r in tab.iterrows():
        for h in HZ:
            rec("new_wapman_institution", f"usable_cells_{h}", r[f"usable_{h}"], "new_alias" if r.join == "alias" else "new",
                horizon=h, field=r.label, n=r.g_rank,
                note=f"{r.institution}; {r.state}; join={r.join}; all_cells={r[f'cells_{h}']}; "
                     f"usable_fixed_2001-2010={r[f'usable_fixed_{h}']}")
    # totals by horizon and cohort group
    groups = {"all cohorts": None, "pooled 0000": ["0000"], "fixed 2001-2010": COHORTS,
              "late 2013-2019 (y1 calendar contrast)": ["2013", "2016", "2019"]}
    tot = []
    for v in ("old", "new", "new_alias", "new_joinfix"):
        c = cells[v]
        for gname, coh in groups.items():
            cc = c if coh is None else c[c.grad_cohort.isin(coh)]
            for h in HZ:
                ch = cc[cc.horizon == h]
                t = dict(variant=v, cohorts=gname, horizon=h, cells=len(ch), usable=int(ch.usable.sum()),
                         usable_insts=int(ch[ch.usable].inst_key.nunique()))
                if v != "old":
                    isnew = ch.institution.isin(new_ids)
                    t.update(cells_from_new=int(isnew.sum()), usable_from_new=int((isnew & ch.usable).sum()))
                tot.append(t)
                rec("coverage_cells", "usable_cells", t["usable"], v, horizon=h, n=t["usable_insts"],
                    sample=gname, note=f"all released BA CIP-4 cells={t['cells']}" +
                    (f"; usable from added institutions={t['usable_from_new']}" if v != "old" else ""))
    tot = pd.DataFrame(tot)
    # Wapman coverage by academia-rank quartile (usable released y5 cell in any cohort)
    g = gen.sort_values("g_rank").reset_index(drop=True)
    g["quartile"] = pd.qcut(np.arange(len(g)), 4, labels=["Q1 (top)", "Q2", "Q3", "Q4 (bottom)"])
    quart = []
    for v in ("old", "new", "new_alias", "new_joinfix"):
        c = cells[v]
        have = set(c[(c.horizon == "y5") & c.usable].inst_key)
        for q, gq in g.groupby("quartile", observed=True):
            k = int(gq.inst_key.isin(have).sum())
            quart.append(dict(variant=v, quartile=str(q), covered=k, total=len(gq)))
            rec("coverage_quartile", "wapman_insts_with_usable_y5_cell", k, v, n=len(gq), field=str(q),
                sample="Wapman academia-ranked institutions; usable released y5 BA CIP-4 cell, any cohort")
    cnt = newi.merge(pd.DataFrame({"institution": ins_n.institution, "st": _state_of(NEW_INST)}),
                     on="institution").groupby("st").size()
    added_states = pd.Series(dict(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))))
    for st, k in added_states.items():
        rec("coverage", "institutions_added_by_state", int(k), "old_vs_new", field=st)
    return tab, tot, pd.DataFrame(quart), new_ids, added_states


def join_audit(ar: pd.DataFrame, gen: pd.DataFrame) -> pd.DataFrame:
    """Verify each JOIN_AUDIT pair (both names exist, the canonical keys differ) and count the usable
    released cells the pair would add (V4.13.0 file; identical ids in V4.14.1)."""
    ins = _pseo_institutions(PSEO_INST).drop_duplicates("institution").set_index("institution")
    gkeys = dict(zip(gen.inst_key, gen.g_rank))
    ark = set(zip(ar.inst_key, ar.field))
    d = raw_ba(PSEO_EARN)
    rows = []
    for pid, wname in JOIN_AUDIT.items():
        wkey = normalize_institution_name(wname)
        assert pid in ins.index and wkey in gkeys, (pid, wname)
        assert ins.loc[pid, "inst_key"] != wkey and ins.loc[pid, "inst_key"] not in gkeys, (pid, wname)
        assert wkey not in set(ins.inst_key), (pid, wname)          # no PSEO institution carries the key
        x = d[d.institution == pid]
        r = dict(institution=pid, pseo_label=ins.loc[pid, "label"], wapman_name=wname, g_rank=int(gkeys[wkey]))
        for h in HZ:
            rel = (x[f"status_{h}_earnings"] == "1") & x.field.notna()
            use = rel & np.array([(wkey, f) in ark for f in x.field], dtype=bool)
            r[f"usable_{h}"] = int(use.sum())
            r[f"usable_fixed_{h}"] = int((use & x.grad_cohort.isin(COHORTS)).sum())
        rows.append(r)
        rec("join_audit", "usable_cells_y5_if_joined", r["usable_y5"], "old", field=wname, n=r["g_rank"],
            note=f"{pid}; PSEO label '{r['pseo_label']}'; usable fixed-cohort y10 cells={r['usable_fixed_y10']}")
    return pd.DataFrame(rows)


def readme_coverage() -> pd.DataFrame:
    """The README section 1 / scripts/01 coverage definition, re-run on each variant: a Wapman
    academia-ranked institution is 'PSEO-covered' if it has any row in load_er_pseo('undergrad')
    (default 30-field TIER0 map, y5, pooled cohort 0000, released); quartiles = pd.qcut of the Wapman
    academia Rank (as scripts/01_probe.py). Not the same definition as coverage() above."""
    r = pd.read_csv(ROOT / "data" / "raw" / "wapman2022" / "ranks.csv",
                    usecols=["Rank", "InstitutionName", "TaxonomyLevel"])
    acad = r[r.TaxonomyLevel == "Academia"].copy()
    acad["inst_key"] = acad.InstitutionName.map(normalize_institution_name)
    acad["pq"] = pd.qcut(acad.Rank, 4, labels=["Q1 (top)", "Q2", "Q3", "Q4 (bottom)"])
    rows = []
    for v in ("old", "new", "new_alias", "new_joinfix"):
        keys = set(er_pseo(v, "undergrad").inst_key)
        inps = acad.inst_key.isin(keys)
        rows.append(dict(variant=v, quartile="all", covered=int(inps.sum()), total=len(acad)))
        for q, idx in acad.groupby("pq", observed=True).groups.items():
            rows.append(dict(variant=v, quartile=str(q), covered=int(inps.loc[idx].sum()), total=len(idx)))
    out = pd.DataFrame(rows)
    for r_ in out.itertuples():
        rec("coverage_readme_def", "wapman_insts_in_load_er_pseo_undergrad", r_.covered, r_.variant,
            n=r_.total, field=r_.quartile,
            sample="scripts/01 definition: any released y5 pooled-0000 BA cell in the 30 TIER0 fields")
    return out


def _state_of(inst_path: Path) -> list:
    ins = pd.read_csv(inst_path, dtype=str)
    ins.columns = [c.strip().lstrip("﻿") for c in ins.columns]
    return list(ins.institution_state)


# ---------------------------------------------------------------------------------------------
# (b) career time via scripts/52
# ---------------------------------------------------------------------------------------------
def career(v: str, ar, gen, grp_of, ref_fields=None):
    set_release(v)
    er = {"y1": s52.load_fixed("y1", COHORTS + [LATE[c] for c in COHORTS]),
          "y5": s52.load_fixed("y5", COHORTS), "y10": s52.load_fixed("y10", COHORTS)}
    panel = s52.fixed_panel(er, ar, gen)
    assert panel["G"].notna().all()
    s52.REDRAWN.clear()
    cell_df, fl = s52.run_fixed(panel)
    fields_all = sorted(fl)
    INT = [f for f in fields_all if grp_of.get(f, "unclassified") == "integrated"]
    OTH = [f for f in fields_all if grp_of.get(f, "unclassified") != "integrated"]
    R = {}

    def put(name, est, bs=None, p=np.nan, k=np.nan, sample=""):
        lo, hi = ci(bs) if bs is not None else (np.nan, np.nan)
        mcse = mc_se_endpoint(bs) if bs is not None else np.nan
        R[name] = dict(est=float(est), lo=lo, hi=hi, p=p, k=k, sample=sample, mcse=mcse)

    for name, fs in (("all", fields_all), ("integrated", INT)):
        o, bs = s52.group_stat(fl, fs, "rhoF_slope", f"slope_{name}")
        null = np.mean([fl[f]["perm"] for f in fs], axis=0)
        put(f"slope_{name}", o, bs, perm_p(null, o), len(fs), f"fixed cohorts 2001-2010, balanced; {name}")
    o, bs = s52.contrast(fl, INT, OTH, "rhoF_slope", "slope_int_minus_oth")
    p = s52.field_label_perm({f: fl[f]["rhoF_slope"] for f in fields_all}, INT, fields_all, "lab_slope")
    put("slope_int_minus_others", o, bs, p, f"{len(INT)} vs {len(OTH)}", "integrated minus others")
    for seg in ("early", "late"):
        o, bs = s52.group_stat(fl, fields_all, f"{seg}_slope", f"{seg}_all")
        put(f"{seg}_slope_all", o, bs, np.nan, len(fields_all), f"segment {seg}; all")
    if ref_fields is not None:
        fs = [f for f in ref_fields if f in fl]
        o, bs = s52.group_stat(fl, fs, "rhoF_slope", "slope_all")
        null = np.mean([fl[f]["perm"] for f in fs], axis=0)
        put("slope_all_reffields", o, bs, perm_p(null, o), len(fs), "same fields as V4.13.0 panel")
    for name, fs in (("all", fields_all), ("integrated", INT)):
        for key in ("bF", "bG"):
            o, bs = s52.group_stat(fl, fs, f"{key}_slope", f"bf_{name}_{key}")
            put(f"d{key}_{name}", o, bs, np.nan, len(fs), f"slope of {key}; {name}")
        o, bs = s52.group_stat(fl, fs, "bG_slope", f"bf_{name}_bG2")
        o2, bs2 = s52.group_stat(fl, fs, "bF_slope", f"bf_{name}_bG2")
        put(f"dG_minus_dF_{name}", o - o2, bs - bs2, np.nan, len(fs), f"dG - dF; {name}")
    bfh = {}
    for key in ("bF", "bG"):
        o, bs = s52.group_stat(fl, fields_all, f"{key}_h", f"bfh_all_{key}")
        bfh[key] = o
        for i, h in enumerate(HZ):
            put(f"{key}_{h}_all", o[i], bs[:, i], np.nan, len(fields_all), f"{key} at {h}; all")
    cal_df, cal = s52.run_calendar(er, ar)
    cal_t = sorted(f for f in cal if "D_cal" in cal[f])
    for name, fs in (("all", cal_t), ("integrated", [f for f in cal_t if f in INT])):
        for key in ("D_within", "D_cal", "D_cross"):
            o = np.mean([cal[f][key] for f in fs])
            bs = two_stage(np.stack([cal[f][key + "_b"] for f in fs]), s52.rng_for(f"cal_{name}_{key}"))
            put(f"{key}_{name}", o, bs, np.nan, len(fs), f"triple-matched; {name}")
    cov_cells, cov = s52.run_coverage(panel)
    cf = sorted(cov)
    rng = s52.rng_for("cov_all")
    K = len(cf)
    fi = rng.integers(K, size=(s52.NBOOT, K)); cix = rng.integers(s52.NBOOT, size=(s52.NBOOT, K))
    raw_b = np.stack([cov[f]["raw_b"] for f in cf])[fi, cix].mean(1)
    par_b = np.stack([cov[f]["par_b"] for f in cf])[fi, cix].mean(1)
    ro, po = np.mean([cov[f]["raw"] for f in cf]), np.mean([cov[f]["par"] for f in cf])
    put("cov_partial_all", po, par_b, np.nan, K, "partial coupling given coverage; all")
    put("cov_share_removed_all", 1 - po / ro, 1 - par_b / raw_b, np.nan, K, "1 - partial/raw; all")
    info = dict(panel_rows=len(panel), panel_insts=panel.inst_key.nunique(),
                cells=int(sum(len(fl[f]["cohorts"]) for f in fields_all)), k=len(fields_all),
                k_int=len(INT), redrawn=dict(s52.REDRAWN),
                traj=np.mean([fl[f]["rhoF_h"] for f in fields_all], axis=0))
    for kk in ("panel_rows", "panel_insts", "cells", "k", "k_int"):
        rec("career_panel", kk, info[kk], v)
    for name, x in R.items():
        rec("career_time", name, x["est"], v, x["lo"], x["hi"], x["p"], x["k"], sample=x["sample"])
    for f in fields_all:
        lo, hi = ci(fl[f]["rhoF_slope_b"])
        rec("career_field_slope", "slope_per_yr", fl[f]["rhoF_slope"], v, lo, hi, k=len(fl[f]["cohorts"]),
            n=int(min(fl[f]["n"])), field=f, note=f"{LAB.get(f, f)}; group={grp_of.get(f, 'unclassified')}; "
            f"n max={max(fl[f]['n'])}")
    info["clustered"] = career_clustered(v, R, panel, er, ar, fl, cal, cov, fields_all, INT, OTH, ref_fields)
    return dict(R=R, fl=fl, fields=fields_all, INT=INT, panel=panel, info=info, bfh=bfh)


# --- institution-clustered inference for the career-time statistics ------------------------------
# scripts/52's two-stage bootstrap redraws institutions separately in every field x cohort cell, and its
# horizon-label permutation shuffles each institution's horizons independently in every cell. The same
# institutions recur across the 4 cohorts and many fields, so both treat repeated information as independent
# (the flaw removed from section (c) in revision 1). The functions below give every section (b) statistic
# the section (c) standard: one institution multinomial draw per replicate, shared by every panel, coverage
# and calendar cell (Spearman computed exactly from the draw counts with wrank / wcorr), with fields
# resampled (crossed; Owen 2007) or held fixed; and a horizon-label permutation with one permutation per
# institution shared by all its cells.
PERMS3 = np.array(list(itertools.permutations(range(3))))
DEGEN = {"fix:bF": "b_F", "fix:bG": "b_G", "fix:rhoF": "coupling", "fix:rhoG": "coupling with G",
         "fix:rFG": "Spearman(F, G)", "cov:raw": "coupling in coverage cells", "cov:par": "the coverage partial",
         "cal:rho": "calendar-cell coupling"}


def calendar_cells(er: dict, ar: pd.DataFrame) -> list:
    """Triple-matched calendar cells ((c, y1), (c, y10), (c+9, y1); n>=NMIN), built exactly as
    scripts/52.run_calendar builds them (checked against its output in career_clustered)."""
    a = ar[["field", "inst_key", "prestige_score"]].rename(columns={"prestige_score": "P"})
    y1, y10 = er["y1"], er["y10"]
    out = []
    for c in COHORTS:
        c9 = LATE[c]
        s0 = y1[y1.grad_cohort == c][["field", "inst_key", "earnings"]].rename(columns={"earnings": "e0"})
        s1 = y10[y10.grad_cohort == c][["field", "inst_key", "earnings"]].rename(columns={"earnings": "e1"})
        s2 = y1[y1.grad_cohort == c9][["field", "inst_key", "earnings"]].rename(columns={"earnings": "e2"})
        pair = s1.merge(s2, on=["field", "inst_key"]).merge(a, on=["field", "inst_key"])
        trip = pair.merge(s0, on=["field", "inst_key"])
        for fld in sorted(set(pair.field)):
            gt = trip[trip.field == fld].sort_values("inst_key")
            if len(gt) >= NMIN:
                out.append(((fld, c), gt.reset_index(drop=True)))
    return out


def w_fixed(P, G, E, W):
    """scripts/52.cell_metrics under institution weights W (B, n): rho_F, rho_G (B,3), r_FG (B,), b_F, b_G."""
    rP, rG = wrank(P, W), wrank(G, W)
    rE = [wrank(E[:, k], W) for k in range(E.shape[1])]
    rhoF = np.stack([wcorr(rP, r, W) for r in rE], -1)
    rhoG = np.stack([wcorr(rG, r, W) for r in rE], -1)
    rFG = wcorr(rP, rG, W)
    d = (1 - rFG ** 2)[:, None]
    d = np.where(d > 1e-12, d, np.nan)
    return dict(rhoF=rhoF, rhoG=rhoG, rFG=rFG, bF=(rhoF - rFG[:, None] * rhoG) / d,
                bG=(rhoG - rFG[:, None] * rhoF) / d)


def w_cov(P, E, C, W):
    """scripts/52.run_coverage's raw and coverage-partial coupling (B,3) under weights W."""
    rP = wrank(P, W)
    raw, par = [], []
    for k in range(3):
        re_, rc = wrank(E[:, k], W), wrank(C[:, k], W)
        a, b, c = wcorr(rP, re_, W), wcorr(rP, rc, W), wcorr(re_, rc, W)
        with np.errstate(invalid="ignore", divide="ignore"):
            par.append((a - b * c) / np.sqrt((1 - b ** 2) * (1 - c ** 2)))
        raw.append(a)
    return dict(raw=np.stack(raw, -1), par=np.stack(par, -1))


def w_cal(P, E, W):
    rP = wrank(P, W)
    return dict(rho=np.stack([wcorr(rP, wrank(E[:, k], W), W) for k in range(E.shape[1])], -1))


def _bad_rows(res: list) -> np.ndarray:
    m = None
    for r in res:
        for x in r.values():
            b = ~np.isfinite(x.reshape(x.shape[0], -1)).all(1)
            m = b if m is None else (m | b)
    return m


def draw_valid(n_inst: int, tag: str, ev, kinds=None, max_tries: int = 100):
    """NBOOT institution multinomial draws; a replicate in which any cell statistic is not finite is redrawn
    as a whole (as draw_inst_weights and scripts/52's boot_valid). ev(W) -> list of per-cell dicts.
    kinds (one label per cell): the first draw's degenerate replicates are counted by label:statistic."""
    rng = rng_for(tag)
    p = np.full(n_inst, 1.0 / n_inst)
    W = rng.multinomial(n_inst, p, size=NBOOT).astype(float)
    res = ev(W)
    first = {}
    if kinds is not None:
        for kd, r in zip(kinds, res):
            for key, x in r.items():
                b = ~np.isfinite(x.reshape(x.shape[0], -1)).all(1)
                first[f"{kd}:{key}"] = first.get(f"{kd}:{key}", np.zeros(len(b), bool)) | b
        first = {k: int(b.sum()) for k, b in first.items()}
        first["any"] = int(_bad_rows(res).sum())
    redrawn = 0
    for _ in range(max_tries):
        bad = np.flatnonzero(_bad_rows(res))
        if not len(bad):
            return (W, res, redrawn, first) if kinds is not None else (W, res, redrawn)
        redrawn += len(bad)
        W[bad] = rng.multinomial(n_inst, p, size=len(bad)).astype(float)
        new = ev(W[bad])
        for r, rn in zip(res, new):
            for k in r:
                r[k][bad] = rn[k]
    raise AssertionError(f"could not draw non-degenerate institution weights for {tag}")


def career_clustered(v, R, panel, er, ar, fl, cal, cov, fields_all, INT, OTH, ref_fields):
    """Crossed and fields-fixed institution-cluster CIs for the section (b) statistics (added to R[name] as
    xlo/xhi/xmcse/xle0 and clo/chi/cmcse/cle0), and institution-clustered horizon-permutation p (p_clu) for
    the slopes. Observed values are recomputed with unit weights and checked against scripts/52's."""
    fx = s52.cells_of(panel)
    cvc = s52.cells_of(panel.dropna(subset=[f"cov_{h}" for h in HZ]).copy())
    tc = calendar_cells(er, ar)
    univ = sorted(set().union(*[set(g.inst_key) for _, g in fx + cvc + tc]))
    iid = {k: i for i, k in enumerate(univ)}
    NI = len(univ)
    A = []
    for kind, cells in (("fix", fx), ("cov", cvc), ("cal", tc)):
        for (fld, _), g in cells:
            E = g[["e0", "e1", "e2"]] if kind == "cal" else g[[f"e_{h}" for h in HZ]]
            A.append((kind, fld, g.P.to_numpy(float), g.G.to_numpy(float) if kind == "fix" else None,
                      E.to_numpy(float), g[[f"cov_{h}" for h in HZ]].to_numpy(float) if kind == "cov" else None,
                      np.array([iid[k] for k in g.inst_key])))

    def ev(Wf, js=None):
        out = []
        for j in (range(len(A)) if js is None else js):
            kind, _, P, G, E, C, ix = A[j]
            Wc = Wf[:, ix]
            out.append(w_fixed(P, G, E, Wc) if kind == "fix" else w_cov(P, E, C, Wc) if kind == "cov"
                       else w_cal(P, E, Wc))
        return out

    def fstats(rs):
        def fmean(kind, key):
            d = {}
            for a_, r in zip(A, rs):
                if a_[0] == kind:
                    d.setdefault(a_[1], []).append(r[key])
            return {f: np.mean(x, 0) for f, x in d.items()}
        rF, bF, bG = fmean("fix", "rhoF"), fmean("fix", "bF"), fmean("fix", "bG")
        raw, par, cr = fmean("cov", "raw"), fmean("cov", "par"), fmean("cal", "rho")
        s = dict(slope={f: x @ W_SLOPE for f, x in rF.items()},
                 early={f: (x[:, 1] - x[:, 0]) / 4 for f, x in rF.items()},
                 late={f: (x[:, 2] - x[:, 1]) / 5 for f, x in rF.items()},
                 dF={f: x @ W_SLOPE for f, x in bF.items()}, dG={f: x @ W_SLOPE for f, x in bG.items()},
                 raw={f: x @ W_SLOPE for f, x in raw.items()}, par={f: x @ W_SLOPE for f, x in par.items()},
                 D_within={f: (x[:, 1] - x[:, 0]) / 9 for f, x in cr.items()},
                 D_cal={f: (x[:, 1] - x[:, 2]) / 9 for f, x in cr.items()},
                 D_cross={f: (x[:, 2] - x[:, 0]) / 9 for f, x in cr.items()})
        s["dGmF"] = {f: s["dG"][f] - s["dF"][f] for f in s["dG"]}
        for i, h in enumerate(HZ):
            s[f"bF_{h}"] = {f: x[:, i] for f, x in bF.items()}
            s[f"bG_{h}"] = {f: x[:, i] for f, x in bG.items()}
        return s

    so = fstats(ev(np.ones((1, NI))))
    # unit weights reproduce scripts/52's observed per-field values
    assert sorted(so["slope"]) == sorted(fields_all)
    for f in fields_all:
        assert np.isclose(so["slope"][f][0], fl[f]["rhoF_slope"], rtol=0, atol=1e-10), (v, f)
        assert np.isclose(so["dF"][f][0], fl[f]["bF_slope"], rtol=1e-9, atol=1e-9), (v, f)
        assert np.isclose(so["dG"][f][0], fl[f]["bG_slope"], rtol=1e-9, atol=1e-9), (v, f)
    cal_t = sorted(f for f in cal if "D_cal" in cal[f])
    assert sorted(so["D_cal"]) == cal_t, v
    for f in cal_t:
        for key in ("D_within", "D_cal", "D_cross"):
            assert np.isclose(so[key][f][0], cal[f][key], rtol=0, atol=1e-10), (v, f, key)
    assert sorted(so["raw"]) == sorted(cov), v
    for f in cov:
        assert np.isclose(so["raw"][f][0], cov[f]["raw"], rtol=0, atol=1e-10)
        assert np.isclose(so["par"][f][0], cov[f]["par"], rtol=1e-9, atol=1e-9)
    W, res, redrawn, first = draw_valid(NI, f"ct_inst_{v}", ev, kinds=[a_[0] for a_ in A])
    sb = fstats(res)
    del res, W
    # independent institution draw per field, shared by that field's panel, coverage and calendar cells (the
    # field x institution intersection cluster of the two-way variance); a replicate is redrawn for that field only
    cells_of_f = {}
    for j, a_ in enumerate(A):
        cells_of_f.setdefault(a_[1], []).append(j)
    res_i, redrawn_i = [None] * len(A), 0
    for f in sorted(cells_of_f):
        js = cells_of_f[f]
        _, rf_, rd_ = draw_valid(NI, f"ct_inst_ind_{v}_{f}", lambda Wf, js=js: ev(Wf, js))
        redrawn_i += rd_
        for j, r_ in zip(js, rf_):
            res_i[j] = r_
    si = fstats(res_i)
    del res_i
    groups = {"all": fields_all, "integrated": INT, "others": OTH, "cal_all": cal_t,
              "cal_integrated": [f for f in cal_t if f in INT], "cov": sorted(cov)}
    if ref_fields is not None:
        groups["ref"] = [f for f in ref_fields if f in fl]
    FI = {g: rng_for(f"ct_fields_{v}_{g}").integers(len(fs), size=(NBOOT, len(fs))) for g, fs in groups.items()}
    bidx = np.arange(NBOOT)[:, None]

    def agg(stat, g):
        """observed mean, crossed replicates, fields-fixed replicates (shared draw), fields-fixed replicates
        (independent draw per field), per-field observed values, field-bootstrap replicates of the observed mean"""
        fs = groups[g]
        X = np.stack([sb[stat][f] for f in fs])                       # (K, NBOOT)
        Xi = np.stack([si[stat][f] for f in fs])
        pf = np.array([so[stat][f][0] for f in fs])
        return float(pf.mean()), X[FI[g], bidx].mean(1), X.mean(0), Xi.mean(0), pf, pf[FI[g]].mean(1)

    def put(name, o, xb, cb, ib, vf, fb):
        r = R[name]
        assert np.isclose(o, r["est"], rtol=1e-9, atol=1e-9), (v, name, o, r["est"])
        (r["xlo"], r["xhi"]), (r["clo"], r["chi"]) = ci(xb), ci(cb)
        r.update(xmcse=mc_se_endpoint(xb), cmcse=mc_se_endpoint(cb), xle0=float(np.mean(xb <= 0)),
                 cle0=float(np.mean(cb <= 0)), xmed=float(np.median(xb)))
        r.update(twoway(r["est"], vf, cb, ib, fb, xb))
        rec("career_time", name + " [crossed bootstrap]", r["est"], v, r["xlo"], r["xhi"], k=r["k"],
            sample=r["sample"], note=f"fields resampled x one institution multinomial draw shared by every panel, "
            f"coverage and calendar cell ({NBOOT}; {NI} institutions); MC SE of CI endpoints={r['xmcse']:.2g}; "
            f"share of replicates <= 0: {r['xle0']:.3f}")
        rec("career_time", name + " [institution-cluster bootstrap, fields fixed]", r["est"], v, r["clo"], r["chi"],
            k=r["k"], sample=r["sample"], note=f"fields held fixed; one institution multinomial draw shared by every "
            f"cell ({NBOOT}); MC SE of CI endpoints={r['cmcse']:.2g}; share of replicates <= 0: {r['cle0']:.3f}")
        rec("career_time", name + " [two-way field x institution cluster variance]", r["est"], v, r["tlo"], r["thi"],
            k=r["k"], sample=r["sample"], note=tw_note(r) + f" ({NBOOT} replicates per draw)")

    spec = [("slope_all", "slope", "all"), ("slope_integrated", "slope", "integrated"),
            ("early_slope_all", "early", "all"), ("late_slope_all", "late", "all"),
            ("dbF_all", "dF", "all"), ("dbG_all", "dG", "all"), ("dG_minus_dF_all", "dGmF", "all"),
            ("dbF_integrated", "dF", "integrated"), ("dbG_integrated", "dG", "integrated"),
            ("dG_minus_dF_integrated", "dGmF", "integrated")]
    spec += [(f"{b}_{h}_all", f"{b}_{h}", "all") for b in ("bF", "bG") for h in HZ]
    spec += [(f"{k}_{g}", k, f"cal_{g}") for g in ("all", "integrated") for k in ("D_within", "D_cal", "D_cross")]
    if ref_fields is not None:
        spec.append(("slope_all_reffields", "slope", "ref"))
    for name, stat, g in spec:
        o, xb, cb, ib, pf, fb = agg(stat, g)
        put(name, o, xb, cb, ib, pf.var(ddof=1) / len(pf), fb)
    oa, xa, ca, ia, pa, fa = agg("slope", "integrated")
    ob, xo, co, io, po, fo = agg("slope", "others")
    put("slope_int_minus_others", oa - ob, xa - xo, ca - co, ia - io,
        pa.var(ddof=1) / len(pa) + po.var(ddof=1) / len(po), fa - fo)
    o_r, x_r, c_r, i_r, p_r, f_r = agg("raw", "cov")
    o_p, x_p, c_p, i_p, p_p, f_p = agg("par", "cov")
    put("cov_partial_all", o_p, x_p, c_p, i_p, p_p.var(ddof=1) / len(p_p), f_p)
    put("cov_share_removed_all", 1 - o_p / o_r, 1 - x_p / x_r, 1 - c_p / c_r, 1 - i_p / i_r,
        jack_var(lambda a, b: 1 - a.mean() / b.mean(), p_p, p_r), 1 - f_p / f_r)
    # institution-clustered horizon-label permutation (one of the 6 horizon orders per institution, shared by
    # every field x cohort cell in which the institution appears)
    S = rng_for(f"ct_hperm_{v}").integers(6, size=(NPERM, NI))
    nulls = {}
    for kind, fld, P, G, E, C, ix in A:
        if kind != "fix":
            continue
        n = len(P)
        Rk = np.stack([rankdata(E[:, k]) for k in range(3)], -1)
        Rp = Rk[np.arange(n)[None, :, None], PERMS3[S[:, ix]]]                  # (NPERM, n, 3)
        rP = rankdata(P)[None, :]
        nulls.setdefault(fld, []).append(np.stack([corr_rows(rP, rk(Rp[:, :, k])) for k in range(3)], -1) @ W_SLOPE)
    nf = {f: np.mean(x, 0) for f, x in nulls.items()}
    for name, g in (("slope_all", "all"), ("slope_integrated", "integrated")):
        null = np.mean([nf[f] for f in groups[g]], 0)
        R[name]["p_clu"] = perm_p(null, R[name]["est"])
        rec("career_time", name + " [institution-clustered horizon permutation]", R[name]["est"], v,
            p=R[name]["p_clu"], k=R[name]["k"], sample=R[name]["sample"],
            note=f"one horizon permutation per institution shared by all its cells ({NPERM}; minimum attainable "
                 f"p {1 / (NPERM + 1):.4f})")
    for kk, vv in (("clustered_universe_insts", NI), ("clustered_replicates_redrawn", redrawn),
                   ("clustered_calendar_cells", len(tc)), ("clustered_coverage_cells", len(cvc))):
        rec("career_panel", kk, vv, v)
    rec("career_panel", "clustered_independent_draw_field_replicates_redrawn", redrawn_i, v,
        n=NBOOT * len(cells_of_f), note="field x replicate draws redrawn (independent institution draw per field) "
                                        f"because some cell's statistic was undefined; {len(cells_of_f)} fields")
    for kk, vv in first.items():
        rec("career_panel", "clustered_first_draw_degenerate_replicates", vv, v, field=kk, n=NBOOT,
            note="replicates of the first institution draw in which some cell's statistic was undefined")
    # widths of the three kinds of interval, for the text
    wid = {name: ((r["xhi"] - r["xlo"]) / (r["hi"] - r["lo"]), (r["chi"] - r["clo"]) / (r["hi"] - r["lo"]))
           for name, r in R.items() if "xlo" in r and np.isfinite(r["lo"])}
    return dict(n_inst=NI, redrawn=redrawn, redrawn_i=redrawn_i, n_fields_i=len(cells_of_f), cal_cells=len(tc),
                fix_cells=len(fx), cov_cells=len(cvc), first=first,
                width_ratio_x=float(np.median([a for a, _ in wid.values()])),
                width_ratio_c=float(np.median([b for _, b in wid.values()])),
                width_ratio_tx=float(np.median([(r["thi"] - r["tlo"]) / (r["xhi"] - r["xlo"])
                                                for r in R.values() if "tlo" in r])))


# ---------------------------------------------------------------------------------------------
# (c) quantile coupling at fixed cohorts
# ---------------------------------------------------------------------------------------------
def load_q(v: str, h: str, cohorts: list[str], by_cip4: bool = False) -> pd.DataFrame:
    """p25/p50/p75 per field x institution x cohort; same rows, weights and aggregation as
    load_er_pseo (cohort-size-weighted mean over the field's CIP-4 codes, applied to each quantile).
    by_cip4=True keeps each CIP-4 code separate (no within-field aggregation; sensitivity only)."""
    r = REL[v]
    d = raw_ba(r["earn"])
    d = d[d.grad_cohort.isin(cohorts) & (d[f"status_{h}_earnings"] == "1") & d.field.notna()].copy()
    cols = {q: f"{h}_{q}_earnings" for q in QS}
    for c in list(cols.values()) + [f"{h}_grads_earn"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[cols["p50"]])
    assert d[[cols["p25"], cols["p75"]]].notna().all().all(), "p25/p75 missing where p50 released"
    d["w"] = d[f"{h}_grads_earn"].fillna(1.0).clip(lower=1.0)
    ins = institutions(v).drop_duplicates("institution")
    d = d.merge(ins, on="institution", how="left")
    d = d[d.inst_key.notna() & (d.inst_key != "")]
    for q, c in cols.items():
        d[f"w_{q}"] = d[c] * d["w"]
    keys = ["field", "cip4", "inst_key", "grad_cohort"] if by_cip4 else ["field", "inst_key", "grad_cohort"]
    g = d.groupby(keys)[[f"w_{q}" for q in QS] + ["w"]].sum().reset_index()
    for q in QS:
        g[q] = g[f"w_{q}"] / g["w"]
    if by_cip4:
        return g[keys + QS]
    # consistency with the canonical loader
    ref = er_pseo(v, "undergrad", h, fields=FIELDS66, grad_cohort=cohorts)
    chk = g.merge(ref[["field", "inst_key", "grad_cohort", "earnings"]], on=["field", "inst_key", "grad_cohort"])
    assert len(chk) == len(g) == len(ref) and np.allclose(chk.p50, chk.earnings, rtol=1e-12, atol=0)
    return g[["field", "inst_key", "grad_cohort"] + QS]


def q_panel(v: str, ar, gen, by_cip4: bool = False) -> pd.DataFrame:
    keys = ["field", "cip4", "inst_key", "grad_cohort"] if by_cip4 else ["field", "inst_key", "grad_cohort"]
    wide = None
    for h in HZ:
        x = load_q(v, h, COHORTS, by_cip4).rename(columns={q: f"{q}_{h}" for q in QS})
        wide = x if wide is None else wide.merge(x, on=keys, how="inner")
    wide = wide.merge(ar[["field", "inst_key", "prestige_score"]].rename(columns={"prestige_score": "P"}),
                      on=["field", "inst_key"], how="inner").merge(gen, on="inst_key", how="left")
    assert wide.G.notna().all()
    return wide.sort_values([k for k in keys if k != "inst_key"] + ["inst_key"]).reset_index(drop=True)


Z975 = 1.959963984540054


def mc_se_endpoint(bs) -> float:
    """Normal-approximation Monte Carlo SE of a 2.5% / 97.5% percentile endpoint estimated from
    len(bs) bootstrap replicates: sqrt(p(1-p)/B) / density at the quantile, density = phi(z)/sd."""
    bs = np.asarray(bs, float)
    phi = np.exp(-Z975 ** 2 / 2) / np.sqrt(2 * np.pi)
    return float(np.sqrt(0.025 * 0.975 / len(bs)) * bs.std(ddof=1) / phi)


IQR_Z = 1.3489795003921634          # IQR of a standard normal


def varvar(x) -> float:
    """Monte Carlo variance of the sample variance of the replicates x (fourth-moment formula, so heavy tails
    widen it)."""
    x = np.asarray(x, float)
    d = x - x.mean()
    B = len(x)
    m2, m4 = float((d ** 2).mean()), float((d ** 4).mean())
    return (m4 - m2 ** 2 * (B - 3) / (B - 1)) / B


def iqr_var(x) -> float:
    """Robust variance: (IQR / 1.349)^2."""
    q1, q3 = np.percentile(np.asarray(x, float), [25, 75])
    return float(((q3 - q1) / IQR_Z) ** 2)


def twoway(est, v_field, cb, ib, fb=None, xb=None) -> dict:
    """Two-way (field, institution) cluster variance in the Cameron-Gelbach-Miller (2011) form
    V = V_field + V_inst - V_field x inst:
      V_field  = between-field variance of the per-field values / K (clusters = fields; for a non-linear
                 statistic the jackknife over fields, which equals s^2/K for a mean),
      V_inst   = variance of the fields-fixed replicates under one institution draw shared by all fields (cb),
      V_fxi    = variance of the fields-fixed replicates under an independent institution draw per field (ib;
                 clusters = field x institution).
    Normal 95% CI est +/- 1.96 SE. If V <= 0 the larger one-way variance is used (flagged). tmcse: Monte Carlo SE
    of a CI endpoint (from the Monte Carlo variance of V_inst and V_fxi). Robust version: the same with
    (IQR/1.349)^2 in place of each variance (field part from the field bootstrap replicates fb). xb (crossed
    replicates) gives V_crossed - 2 V_fxi as a second estimator of the same quantity."""
    vs, vi = float(np.var(cb, ddof=1)), float(np.var(ib, ddof=1))
    v2 = v_field + vs - vi
    fallback = not v2 > 0
    se = float(np.sqrt(v2 if not fallback else max(v_field, vs)))
    mc_v = float(np.sqrt(max(varvar(cb) + varvar(ib), 0.0)))
    out = dict(tlo=est - Z975 * se, thi=est + Z975 * se, tse=se, tz=est / se if se > 0 else np.nan,
               tmcse=Z975 * mc_v / (2 * se) if se > 0 else np.nan, tvf=float(v_field), tvs=vs, tvi=vi,
               tfallback=bool(fallback))
    if fb is not None:
        vr = iqr_var(fb) + iqr_var(cb) - iqr_var(ib)
        out.update(tse_r=float(np.sqrt(vr)) if vr > 0 else np.nan,
                   tz_r=est / np.sqrt(vr) if vr > 0 else np.nan)
    if xb is not None:
        va = float(np.var(xb, ddof=1)) - 2 * vi
        out.update(tse_alt=float(np.sqrt(va)) if va > 0 else np.nan, txse=float(np.std(xb, ddof=1)))
    return out


def tw_note(r: dict) -> str:
    return (f"normal CI est +/- 1.96 SE; SE={r['tse']:.4g}, z={r['tz']:.3g}; V_field={r['tvf']:.3g}, "
            f"V_inst(shared draw)={r['tvs']:.3g}, V_field x inst(independent draw per field)={r['tvi']:.3g}"
            + ("; V<=0, larger one-way variance used" if r["tfallback"] else "")
            + (f"; robust (IQR) SE={r['tse_r']:.4g}, z={r['tz_r']:.3g}" if np.isfinite(r.get("tse_r", np.nan)) else "")
            + (f"; SE from V_crossed - 2 V_field x inst={r['tse_alt']:.4g}" if np.isfinite(r.get("tse_alt", np.nan)) else "")
            + f"; MC SE of CI endpoints={r['tmcse']:.2g}")


def jack_var(fn, *cols) -> float:
    """Jackknife variance over fields of fn(*cols) (each col indexed by field along axis 0)."""
    K = len(cols[0])
    th = np.array([fn(*[np.delete(c, i, axis=0) for c in cols]) for i in range(K)])
    return float((K - 1) / K * ((th - th.mean()) ** 2).sum())


def x_status(lo, hi, mcse) -> str:
    """'above 0' / 'below 0' / 'touches 0' (an endpoint within 2 Monte Carlo SE of 0) / 'includes 0'."""
    tol = 2 * mcse
    if lo > tol:
        return "above 0"
    if hi < -tol:
        return "below 0"
    if abs(lo) <= tol or abs(hi) <= tol:
        return "touches 0"
    return "includes 0"


def cohort_desc(c: str) -> str:
    xs = c.split("+")
    other = [x for x in xs if x != "0000"]
    out = (["the pooled cell"] if "0000" in xs else []) + (
        [("cohort " if len(other) == 1 else "cohorts ") + join_and(other)] if other else [])
    return join_and(out) if out else "none"


def q_metrics(P, G, E, idx):
    """E (n, 3 quantiles, 3 horizons) -> rhoF, rhoG (B,3,3); tail = rho(F, p75/p50), low = rho(F, p50/p25)."""
    rP, rG = rk(P[idx]), rk(G[idx])
    rF = np.empty((idx.shape[0], 3, 3)); rGq = np.empty_like(rF)
    tail = np.empty((idx.shape[0], 3)); low = np.empty_like(tail)
    for j in range(3):
        for k in range(3):
            re_ = rk(E[:, j, k][idx])
            rF[:, j, k] = corr_rows(rP, re_); rGq[:, j, k] = corr_rows(rG, re_)
    for k in range(3):
        tail[:, k] = corr_rows(rP, rk((E[:, 2, k] / E[:, 1, k])[idx]))
        low[:, k] = corr_rows(rP, rk((E[:, 1, k] / E[:, 0, k])[idx]))
    return dict(rhoF=rF, rhoG=rGq, tail=tail, low=low)


# --- crossed (field x institution) bootstrap helpers ---------------------------------------------
# One replicate draws every institution of the panel's universe a multinomial number of times, W[b, i],
# and applies that same draw to every cell (all fields, cohorts, quantiles, horizons). Within a cell the
# resample is "institution i appears W[b, i] times"; its Spearman correlation is computed exactly from
# weights (wrank + wcorr) without materializing the resample (checked against the materialized
# resample in quantile()).
def wrank(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Average ranks (ties averaged, as rankdata) that each observation takes in a resample in which
    observation i appears W[b, i] times. x (n,), W (B, n) -> (B, n)."""
    u, inv = np.unique(x, return_inverse=True)
    oh = np.zeros((len(x), len(u)))
    oh[np.arange(len(x)), inv] = 1.0
    gw = W @ oh                                   # (B, u) copies per distinct value
    C = np.cumsum(gw, axis=1)
    return (C - (gw - 1.0) / 2.0)[:, inv]


def wcorr(x: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Row-wise weighted Pearson correlation; x, y, W (B, n)."""
    s = W.sum(1, keepdims=True)
    xc = x - (W * x).sum(1, keepdims=True) / s
    yc = y - (W * y).sum(1, keepdims=True) / s
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W * xc * yc).sum(1) / np.sqrt((W * xc ** 2).sum(1) * (W * yc ** 2).sum(1))


def w_metrics(P, G, E, W):
    """Spearman(F, quantile) and Spearman(G, quantile) (B, 3 quantiles, 3 horizons) under weights W."""
    rP = wrank(P, W)
    rG = wrank(G, W) if G is not None else None
    rF = np.empty((W.shape[0], 3, 3)); rGq = np.full_like(rF, np.nan)
    for j in range(3):
        for k in range(3):
            re_ = wrank(E[:, j, k], W)
            rF[:, j, k] = wcorr(rP, re_, W)
            if rG is not None:
                rGq[:, j, k] = wcorr(rG, re_, W)
    return rF, rGq


def draw_inst_weights(n_inst: int, tag: str, cell_arrays: list, max_tries: int = 100):
    """NBOOT institution multinomial draws, conditioned on non-degenerate replicates (as scripts/52's
    boot_valid): a replicate in which any cell's coupling is not finite (e.g. only two distinct institutions
    drawn in a 15-institution cell) is redrawn as a whole. cell_arrays: [(P, G or None, E, ix)].
    Returns W (NBOOT, n_inst) and the number of redrawn replicates."""
    rng = rng_for(tag)
    p = np.full(n_inst, 1.0 / n_inst)
    W = rng.multinomial(n_inst, p, size=NBOOT).astype(float)
    rows = np.arange(NBOOT)
    redrawn = 0
    for _ in range(max_tries):
        bad = np.zeros(len(rows), bool)
        for P, G, E, ix in cell_arrays:
            bF, bG = w_metrics(P, G, E, W[rows][:, ix])
            bad |= ~np.isfinite(bF).all((1, 2))
            if G is not None:
                bad |= ~np.isfinite(bG).all((1, 2))
        rows = rows[bad]
        if not len(rows):
            return W, redrawn
        redrawn += len(rows)
        W[rows] = rng.multinomial(n_inst, p, size=len(rows)).astype(float)
    raise AssertionError(f"could not draw non-degenerate institution weights for {tag}")


def cell_arrays(cells, iid, with_G=True):
    out = []
    for _, g in cells:
        E = np.stack([np.stack([g[f"{q}_{h}"].to_numpy(float) for h in HZ], -1) for q in QS], 1)
        out.append((g.P.to_numpy(float), g.G.to_numpy(float) if with_G else None, E,
                    np.array([iid[k] for k in g.inst_key])))
    return out


def fisher(x):
    return np.arctanh(np.clip(x, -0.999999, 0.999999))


def field_agg(CR: list, key: str, tf=None, min_n: int = 0, weight: bool = False):
    """Per-field observed (K,3,3) and crossed-replicate (K,B,3,3) values from cell records.
    Default: tf(cell value) averaged over cohorts within unit, then over units within field (unit = field
    in the primary spec, CIP-4 code in the CIP-4 sensitivity). weight=True: per-field sums of n * value and
    of n over cells (for an n-weighted mean over all cells)."""
    tf = tf or (lambda x: x)
    cr = [c for c in CR if c["n"] >= min_n]
    fields = sorted({c["field"] for c in cr})
    O, Bs, Wn = [], [], []
    for f in fields:
        cf = [c for c in cr if c["field"] == f]
        if weight:
            O.append(sum(c["n"] * tf(c["o" + key]) for c in cf))
            Bs.append(sum(c["n"] * tf(c["b" + key]) for c in cf))
            Wn.append(float(sum(c["n"] for c in cf)))
        else:
            units = sorted({c["unit"] for c in cf})
            O.append(np.mean([np.mean([tf(c["o" + key]) for c in cf if c["unit"] == u], 0) for u in units], 0))
            Bs.append(np.mean([np.mean([tf(c["b" + key]) for c in cf if c["unit"] == u], 0) for u in units], 0))
    return fields, np.stack(O), np.stack(Bs), (np.array(Wn) if weight else None)


def summarize(O, Bs, Wn, fi):
    """Across-field mean: observed (3,3), crossed replicates (fields resampled by fi; B,3,3) and
    institution-cluster replicates with fields fixed (B,3,3)."""
    b = np.arange(Bs.shape[1])[:, None]
    if Wn is None:
        return O.mean(0), Bs[fi, b].mean(1), Bs.mean(0)
    return (O.sum(0) / Wn.sum(), Bs[fi, b].sum(1) / Wn[fi].sum(1)[:, None, None], Bs.sum(0) / Wn.sum())


def diffs(X):
    """(..., 3 quantiles, 3 horizons) -> dict of p75-p50 and p25-p50 by horizon (..., 3)."""
    return {"p75-p50": X[..., 2, :] - X[..., 1, :], "p25-p50": X[..., 0, :] - X[..., 1, :]}


def cip4_sensitivity(v: str, ar, gen) -> dict:
    """Quantile coupling on CIP-4 x cohort cells (each CIP-4 code's own p25/p50/p75, no within-field
    aggregation); balanced on y1/y5/y10, n>=15 per cell. Mean over cohorts within CIP-4, over CIP-4 codes
    within field, then over fields; crossed bootstrap over fields and institutions."""
    qp = q_panel(v, ar, gen, by_cip4=True)
    cells = [(k, g) for k, g in qp.groupby(["field", "cip4", "grad_cohort"], sort=True) if len(g) >= NMIN]
    univ = sorted(set().union(*[set(g.inst_key) for _, g in cells]))
    iid = {k: i for i, k in enumerate(univ)}
    W, redrawn = draw_inst_weights(len(univ), f"q_cip4_inst_{v}", cell_arrays(cells, iid, with_G=False))
    # independent institution draw per field (all of the field's CIP-4 x cohort cells), for the two-way variance
    arrs = cell_arrays(cells, iid, with_G=False)
    fcells = {}
    for j, ((fld, _, _), _) in enumerate(cells):
        fcells.setdefault(fld, []).append(j)
    WI, redrawn_i = {}, 0
    for fld in sorted(fcells):
        WI[fld], rd_ = draw_inst_weights(len(univ), f"q_cip4_ind_inst_{v}_{fld}", [arrs[j] for j in fcells[fld]])
        redrawn_i += rd_
    CR = []
    for (fld, c4, coh), g in cells:
        n = len(g)
        P = g.P.to_numpy(float)
        E = np.stack([np.stack([g[f"{q}_{h}"].to_numpy(float) for h in HZ], -1) for q in QS], 1)
        oF, _ = w_metrics(P, None, E, np.ones((1, n)))
        ix = [iid[k] for k in g.inst_key]
        bF, _ = w_metrics(P, None, E, W[:, ix])
        bFi, _ = w_metrics(P, None, E, WI[fld][:, ix])
        CR.append(dict(field=fld, unit=c4, coh=coh, n=n, oF=oF[0], bF=bF, oFi=oF[0], bFi=bFi))
    del WI
    fields, O, Bs, _ = field_agg(CR, "F")
    _, _, Bsi, _ = field_agg(CR, "Fi")
    fi = rng_for(f"q_cip4_fields_{v}").integers(len(fields), size=(NBOOT, len(fields)))
    o, x, c = summarize(O, Bs, None, fi)
    assert np.isfinite(x).all() and np.isfinite(c).all(), "non-finite crossed replicate (CIP-4)"
    info = dict(cells=len(CR), cip4=len({c_["unit"] for c_ in CR}), k=len(fields), insts=len(univ),
                redrawn=redrawn, redrawn_i=redrawn_i)
    return dict(o=o, x=x, c=c, K=len(fields), info=info, i=Bsi.mean(0), pf=O, fb=O[fi].mean(1))


def quantile(v: str, ar, gen):
    qp = q_panel(v, ar, gen)
    cells = [(k, g) for k, g in qp.groupby(["field", "grad_cohort"], sort=True) if len(g) >= NMIN]
    # institution universe of the analysed cells: the cluster for the permutation and the crossed bootstrap
    univ = sorted(set().union(*[set(g.inst_key) for _, g in cells]))
    iid = {k: i for i, k in enumerate(univ)}
    NI = len(univ)
    W, redrawn = draw_inst_weights(NI, f"q_crossed_inst_{v}", cell_arrays(cells, iid))  # (NBOOT, NI)
    rec("quantile", "crossed_bootstrap_replicates_redrawn", redrawn, v, n=NBOOT,
        note=f"replicates redrawn because some cell's coupling was not finite; {NI} institutions")
    # independent institution draw per field (shared by the field's cohorts): the field x institution
    # intersection cluster of the two-way variance
    arrs = cell_arrays(cells, iid)
    fcells = {}
    for j, ((fld, _), _) in enumerate(cells):
        fcells.setdefault(fld, []).append(j)
    WI, redrawn_i = {}, 0
    for fld in sorted(fcells):
        WI[fld], rd_ = draw_inst_weights(NI, f"q_ind_inst_{v}_{fld}", [arrs[j] for j in fcells[fld]])
        redrawn_i += rd_
    del arrs
    rec("quantile", "independent_draw_field_replicates_redrawn", redrawn_i, v, n=NBOOT * len(fcells),
        note="field x replicate draws redrawn (independent institution draw per field) because some cell's coupling "
             "was not finite")
    FLIP = {pr: rng_for(f"q_perm_clust_{v}_{pr}").integers(2, size=(NPERM, NI)).astype(bool)
            for pr in ("p75-p50", "p25-p50")}                                         # one flip per institution
    rp = rng_for(f"q_perm_{v}")        # superseded per-cell independent flips (kept for comparison)
    per, CR = {}, []
    for (fld, coh), g in cells:
        n = len(g)
        P, G = g.P.to_numpy(float), g.G.to_numpy(float)
        E = np.stack([np.stack([g[f"{q}_{h}"].to_numpy(float) for h in HZ], -1) for q in QS], 1)   # (n,3,3)
        ix = np.array([iid[k] for k in g.inst_key])
        obs = q_metrics(P, G, E, np.arange(n)[None, :])
        bt = q_metrics(P, G, E, rng_for(f"q_boot_{v}_{fld}_{coh}").integers(n, size=(NBOOT, n)))
        # quantile-label permutation within institution (p75 vs p50; p25 vs p50): independent flips per
        # cell (superseded) and institution-clustered flips (one per institution, shared by every cell and
        # horizon in which the institution appears)
        rP = rankdata(P)[None, :]
        pn, pc = {}, {}
        for a, b in (("p75", "p50"), ("p25", "p50")):
            ia, ib = QS.index(a), QS.index(b)
            n_ind, n_cl = [], []
            for k in range(3):
                Ra, Rb = rankdata(E[:, ia, k]), rankdata(E[:, ib, k])
                for S, dst in ((rp.integers(2, size=(NPERM, n)).astype(bool), n_ind),
                               (FLIP[f"{a}-{b}"][:, ix], n_cl)):
                    A = np.where(S, Rb, Ra); B = np.where(S, Ra, Rb)       # A carries label a
                    dst.append(corr_rows(rP, rk(A)) - corr_rows(rP, rk(B)))
            pn[f"{a}-{b}"] = np.stack(n_ind, -1)                                  # (NPERM,3)
            pc[f"{a}-{b}"] = np.stack(n_cl, -1)
        # crossed bootstrap: the shared institution draw restricted to this cell
        bF, bG = w_metrics(P, G, E, W[:, ix])
        bFi, bGi = w_metrics(P, G, E, WI[fld][:, ix])
        CR.append(dict(field=fld, unit=fld, coh=coh, n=n, oF=obs["rhoF"][0], oG=obs["rhoG"][0], bF=bF, bG=bG,
                       oFi=obs["rhoF"][0], oGi=obs["rhoG"][0], bFi=bFi, bGi=bGi))
        d = per.setdefault(fld, dict(n=[], cohorts=[], o={}, b={}, perm={}, pclu={}, r5075=[]))
        d["n"].append(n); d["cohorts"].append(coh)
        d["r5075"].append([spearmanr(E[:, 1, k], E[:, 2, k])[0] for k in range(3)])
        for key in ("rhoF", "rhoG", "tail", "low"):
            d["o"].setdefault(key, []).append(obs[key][0]); d["b"].setdefault(key, []).append(bt[key])
        for key in pn:
            d["perm"].setdefault(key, []).append(pn[key]); d["pclu"].setdefault(key, []).append(pc[key])
    # exactness check of the weighted Spearman against materialized resamples (first cell, 3 replicates)
    (_, g0), c0 = cells[0], CR[0]
    E0 = np.stack([np.stack([g0[f"{q}_{h}"].to_numpy(float) for h in HZ], -1) for q in QS], 1)
    w0 = W[:, [iid[k] for k in g0.inst_key]].astype(int)
    for bb in range(3):
        idx = np.repeat(np.arange(len(g0)), w0[bb])
        ref = spearmanr(g0.P.to_numpy(float)[idx], E0[idx, 2, 1])[0]
        assert np.isclose(ref, c0["bF"][bb, 2, 1], rtol=0, atol=1e-10), (ref, c0["bF"][bb, 2, 1])
    fl = {}
    for f, d in per.items():
        fl[f] = dict(n=d["n"], cohorts=d["cohorts"], r5075=np.mean(d["r5075"], 0),
                     **{k: np.mean(v_, 0) for k, v_ in d["o"].items()},
                     **{k + "_b": np.mean(v_, 0) for k, v_ in d["b"].items()},
                     **{"perm_" + k: np.mean(v_, 0) for k, v_ in d["perm"].items()},
                     **{"pclu_" + k: np.mean(v_, 0) for k, v_ in d["pclu"].items()})
    fields = sorted(fl)
    K = len(fields)
    used = pd.concat([g[["field", "grad_cohort", "inst_key"]] for _, g in cells])
    per_inst = used.groupby("inst_key").agg(nf=("field", "nunique"), nc=("field", "size"))
    del WI
    out = dict(fl=fl, fields=fields, panel=qp, n_inst=NI, n_cells=len(CR), redrawn=redrawn, redrawn_i=redrawn_i,
               max_fields_inst=int(per_inst.nf.max()), median_cells_inst=float(per_inst.nc.median()),
               max_cells_inst=int(per_inst.nc.max()))
    for kk in ("n_inst", "n_cells", "max_fields_inst", "median_cells_inst", "max_cells_inst"):
        rec("quantile_panel", kk, out[kk], v, sample="analysed field x cohort cells (n>=15)")
    # paired two-stage bootstrap over everything at once (same field and replicate draws)
    rng = rng_for(f"q_two_stage_{v}")
    fi = rng.integers(K, size=(NBOOT, K)); cix = rng.integers(NBOOT, size=(NBOOT, K))
    G_ = {k: np.stack([fl[f][k + "_b"] for f in fields])[fi, cix].mean(1) for k in ("rhoF", "rhoG", "tail", "low")}
    O_ = {k: np.mean([fl[f][k] for f in fields], 0) for k in ("rhoF", "rhoG", "tail", "low")}
    Pm = {k: np.mean([fl[f]["perm_" + k] for f in fields], 0) for k in ("p75-p50", "p25-p50")}
    Pc = {k: np.mean([fl[f]["pclu_" + k] for f in fields], 0) for k in ("p75-p50", "p25-p50")}
    # crossed bootstrap and institution-cluster bootstrap (fields fixed)
    fi_x = rng_for(f"q_crossed_fields_{v}").integers(K, size=(NBOOT, K))
    fF, OF, BF, _ = field_agg(CR, "F")
    _, OG, BG, _ = field_agg(CR, "G")
    assert fF == fields
    oF, X_F, C_F = summarize(OF, BF, None, fi_x)
    oG, X_G, C_G = summarize(OG, BG, None, fi_x)
    assert np.allclose(oF, O_["rhoF"], rtol=0, atol=1e-12) and np.allclose(oG, O_["rhoG"], rtol=0, atol=1e-12)
    for arr in (X_F, C_F, X_G, C_G):
        assert np.isfinite(arr).all(), f"non-finite crossed replicate ({v})"
    _, _, BFi, _ = field_agg(CR, "Fi")
    _, _, BGi, _ = field_agg(CR, "Gi")
    I_F, I_G = BFi.mean(0), BGi.mean(0)             # fields fixed, independent institution draw per field
    assert np.isfinite(I_F).all() and np.isfinite(I_G).all()
    FB = {"F": OF[fi_x].mean(1), "G": OG[fi_x].mean(1)}   # field bootstrap of the observed per-field values
    ARR = {"F": (OF, I_F, FB["F"]), "G": (OG, I_G, FB["G"])}

    def twa(src, T):
        """two-way inputs for a statistic T of the (…, 3 quantiles, 3 horizons) coupling arrays"""
        O, I, Fb = ARR[src]
        return dict(ib=T(I), pf=T(O), fb=T(Fb))
    S = {}

    def put(stat, est, bs, p=np.nan, h="", q="", sample="", xb=None, cb=None, p_ind=np.nan, ib=None, pf=None,
            fb=None):
        lo, hi = ci(bs)
        se = mc_se_endpoint(bs)
        r = dict(est=float(est), lo=lo, hi=hi, p=p, mcse=se, p_ind=p_ind)
        smp = sample or "fixed cohorts 2001-2010, balanced on p25/p50/p75 at y1/y5/y10, n>=15"
        rec("quantile", stat, float(est), v, lo, hi, p, K, horizon=h, quantile=q, sample=smp,
            note=f"CI: paired two-stage bootstrap ({NBOOT}; institutions redrawn separately per cell); "
                 + (f"p: institution-clustered quantile-label permutation ({NPERM}); " if np.isfinite(p) else "")
                 + f"MC SE of CI endpoints={se:.2g}")
        if xb is not None:
            r["xlo"], r["xhi"] = ci(xb)
            r["xmcse"] = mc_se_endpoint(xb)
            rec("quantile", stat + " [crossed bootstrap]", float(est), v, r["xlo"], r["xhi"], np.nan, K,
                horizon=h, quantile=q, sample=smp,
                note=f"fields resampled x one institution multinomial draw applied to every cell ({NBOOT}; "
                     f"{NI} institutions); MC SE of CI endpoints={r['xmcse']:.2g}")
        if cb is not None:
            r["clo"], r["chi"] = ci(cb)
            rec("quantile", stat + " [institution-cluster bootstrap, fields fixed]", float(est), v, r["clo"],
                r["chi"], np.nan, K, horizon=h, quantile=q, sample=smp,
                note=f"fields held fixed; one institution multinomial draw applied to every cell ({NBOOT})")
        if ib is not None:
            r.update(twoway(float(est), pf.var(ddof=1) / len(pf), cb, ib, fb, xb))
            rec("quantile", stat + " [two-way field x institution cluster variance]", float(est), v, r["tlo"],
                r["thi"], np.nan, K, horizon=h, quantile=q, sample=smp, note=tw_note(r) + f" ({NBOOT} replicates per draw)")
        if np.isfinite(p_ind):
            rec("quantile", stat + " [independent-flip permutation, superseded]", float(est), v, p=p_ind, k=K,
                horizon=h, quantile=q, sample=smp,
                note=f"labels flipped independently in every field x cohort cell ({NPERM}); ignores that "
                     "institutions recur across cells, so the null is too narrow")
        S[(stat, h, q)] = r

    dF_x, dF_c, dF_o = diffs(X_F), diffs(C_F), diffs(O_["rhoF"])
    dG_x, dG_c, dG_o = diffs(X_G), diffs(C_G), diffs(O_["rhoG"])
    for k, h in enumerate(HZ):
        for j, q in enumerate(QS):
            put("coupling_F", O_["rhoF"][j, k], G_["rhoF"][:, j, k], h=h, q=q, xb=X_F[:, j, k])
            put("coupling_G", O_["rhoG"][j, k], G_["rhoG"][:, j, k], h=h, q=q, xb=X_G[:, j, k])
        for a, b in (("p75", "p50"), ("p25", "p50")):
            ia, ib, pr = QS.index(a), QS.index(b), f"{a}-{b}"
            o = dF_o[pr][k]
            Tk = (lambda X, ia=ia, ib=ib, k=k: X[..., ia, k] - X[..., ib, k])
            put(f"F_{a}_minus_{b}", o, G_["rhoF"][:, ia, k] - G_["rhoF"][:, ib, k],
                p=perm_p(Pc[pr][:, k], o), p_ind=perm_p(Pm[pr][:, k], o), h=h,
                xb=dF_x[pr][:, k], cb=dF_c[pr][:, k], **twa("F", Tk))
            put(f"G_{a}_minus_{b}", dG_o[pr][k], G_["rhoG"][:, ia, k] - G_["rhoG"][:, ib, k], h=h,
                xb=dG_x[pr][:, k], cb=dG_c[pr][:, k], **twa("G", Tk))
        put("F_p75_minus_p25", O_["rhoF"][2, k] - O_["rhoF"][0, k], G_["rhoF"][:, 2, k] - G_["rhoF"][:, 0, k], h=h,
            xb=X_F[:, 2, k] - X_F[:, 0, k])
        put("F_vs_p75_over_p50", O_["tail"][k], G_["tail"][:, k], h=h)
        put("F_vs_p50_over_p25", O_["low"][k], G_["low"][:, k], h=h)
        r5075 = float(np.mean([fl[f]["r5075"][k] for f in fields]))
        S[("rank_p50_p75", h, "")] = dict(est=r5075)
        rec("quantile", "spearman_p50_p75_across_institutions", r5075, v, k=K, horizon=h,
            sample="within field x cohort cell, mean over cohorts then fields")
        nf = sum(fl[f]["rhoF"][2, k] > fl[f]["rhoF"][1, k] for f in fields)
        rec("quantile", "fields_p75_gt_p50", nf, v, k=K, horizon=h)
        S[("fields_p75_gt_p50", h, "")] = dict(est=nf)
    # averaged over horizons (post hoc summary), and career-time slope by quantile
    for a, b in (("p75", "p50"), ("p25", "p50")):
        ia, ib, pr = QS.index(a), QS.index(b), f"{a}-{b}"
        o = dF_o[pr].mean()
        Ta = (lambda X, ia=ia, ib=ib: (X[..., ia, :] - X[..., ib, :]).mean(-1))
        put(f"F_{a}_minus_{b}", o, (G_["rhoF"][:, ia] - G_["rhoF"][:, ib]).mean(-1),
            p=perm_p(Pc[pr].mean(-1), o), p_ind=perm_p(Pm[pr].mean(-1), o), h=AVG,
            xb=dF_x[pr].mean(-1), cb=dF_c[pr].mean(-1),
            sample="post hoc mean over y1/y5/y10; fixed cohorts 2001-2010, balanced, n>=15", **twa("F", Ta))
        for tag, null in (("independent-flip", Pm[pr].mean(-1)), ("institution-clustered", Pc[pr].mean(-1))):
            sd = float(null.std(ddof=1))
            S[(f"null_sd_{pr}", AVG, tag)] = dict(est=sd)
            rec("quantile", f"perm_null_sd_F_{a}_minus_{b}", sd, v, k=K, horizon=AVG, note=f"{tag} permutation ({NPERM})")
    put("G_p75_minus_p50", dG_o["p75-p50"].mean(), (G_["rhoG"][:, 2] - G_["rhoG"][:, 1]).mean(-1), h=AVG,
        xb=dG_x["p75-p50"].mean(-1), cb=dG_c["p75-p50"].mean(-1),
        **twa("G", lambda X: (X[..., 2, :] - X[..., 1, :]).mean(-1)))
    for j, q in enumerate(QS):
        put("slope_F", O_["rhoF"][j] @ W_SLOPE, G_["rhoF"][:, j] @ W_SLOPE, h="per year", q=q, xb=X_F[:, j] @ W_SLOPE)
    put("slope_F_p75_minus_p50", (O_["rhoF"][2] - O_["rhoF"][1]) @ W_SLOPE,
        (G_["rhoF"][:, 2] - G_["rhoF"][:, 1]) @ W_SLOPE, h="per year", xb=dF_x["p75-p50"] @ W_SLOPE,
        cb=dF_c["p75-p50"] @ W_SLOPE, **twa("F", lambda X: (X[..., 2, :] - X[..., 1, :]) @ W_SLOPE))
    # aggregation sensitivity (crossed bootstrap for every spec)
    # each spec: observed (3,3), crossed (B,3,3), fields fixed (B,3,3), k, fields fixed with an independent draw
    # per field (B,3,3), per-field values (linearized for the n-weighted mean; K,3,3), field bootstrap (B,3,3)
    SENS = {"primary (field mean of cohort-mean Spearman)": (oF, X_F, C_F, K, I_F, OF, FB["F"])}
    _, O1, B1, _ = field_agg(CR, "F", tf=fisher)
    _, _, B1i, _ = field_agg(CR, "Fi", tf=fisher)
    SENS["Fisher z (difference in mean z)"] = (*summarize(O1, B1, None, fi_x), K, B1i.mean(0), O1,
                                               O1[fi_x].mean(1))
    _, O2, B2, W2 = field_agg(CR, "F", weight=True)
    _, _, B2i, _ = field_agg(CR, "Fi", weight=True)
    e2 = O2.sum(0) / W2.sum()
    SENS["n-weighted mean over field x cohort cells"] = (*summarize(O2, B2, W2, fi_x), K, B2i.sum(0) / W2.sum(),
                                                         (O2 - e2 * W2[:, None, None]) / W2.mean(),
                                                         O2[fi_x].sum(1) / W2[fi_x].sum(1)[:, None, None])
    f3, O3, B3, _ = field_agg(CR, "F", min_n=NMIN_BIG)
    _, _, B3i, _ = field_agg(CR, "Fi", min_n=NMIN_BIG)
    fi3 = rng_for(f"q_crossed_fields_n30_{v}").integers(len(f3), size=(NBOOT, len(f3)))
    SENS[f"cells with n>={NMIN_BIG} only"] = (*summarize(O3, B3, None, fi3), len(f3), B3i.mean(0), O3,
                                              O3[fi3].mean(1))
    out["n_cells_big"] = sum(c["n"] >= NMIN_BIG for c in CR)
    if v == "new":
        c4 = cip4_sensitivity(v, ar, gen)
        SENS["CIP-4 x cohort cells, no within-field aggregation"] = (c4["o"], c4["x"], c4["c"], c4["K"], c4["i"],
                                                                      c4["pf"], c4["fb"])
        out["cip4_info"] = c4["info"]
        for kk, vv in c4["info"].items():
            rec("quantile_sensitivity", f"cip4_panel_{kk}", vv, v)
    sens = {}
    for spec, (o, x, c, k_, i_, pf_, fb_) in SENS.items():
        do, dx, dc, di, dp, dfb = diffs(o), diffs(x), diffs(c), diffs(i_), diffs(pf_), diffs(fb_)
        for pr in ("p75-p50", "p25-p50"):
            for k, h in enumerate(HZ + [AVG]):
                est = float(do[pr][k] if h != AVG else do[pr].mean())
                xb = dx[pr][:, k] if h != AVG else dx[pr].mean(-1)
                cb = dc[pr][:, k] if h != AVG else dc[pr].mean(-1)
                ib = di[pr][:, k] if h != AVG else di[pr].mean(-1)
                pv = dp[pr][:, k] if h != AVG else dp[pr].mean(-1)
                fbv = dfb[pr][:, k] if h != AVG else dfb[pr].mean(-1)
                (xlo, xhi), (clo, chi) = ci(xb), ci(cb)
                sens[(spec, pr, h)] = dict(est=est, xlo=xlo, xhi=xhi, clo=clo, chi=chi, k=k_)
                sens[(spec, pr, h)].update(twoway(est, pv.var(ddof=1) / len(pv), cb, ib, fbv, xb))
                rec("quantile_sensitivity", f"F_{pr.replace('-', '_minus_')}", est, v, xlo, xhi, k=k_, horizon=h,
                    sample=spec, note=f"CI: crossed bootstrap ({NBOOT}); fields-fixed institution-cluster CI "
                                      f"[{clo:.4f}, {chi:.4f}]")
                t_ = sens[(spec, pr, h)]
                rec("quantile_sensitivity", f"F_{pr.replace('-', '_minus_')} [two-way field x institution cluster "
                    "variance]", est, v, t_["tlo"], t_["thi"], k=k_, horizon=h, sample=spec, note=tw_note(t_))
    out.update(S=S, K=K, sens=sens)
    for f in fields:
        for k, h in enumerate(HZ):
            rec("quantile_field", "coupling_F_p50", fl[f]["rhoF"][1, k], v, horizon=h, field=f,
                n=int(min(fl[f]["n"])), k=len(fl[f]["cohorts"]))
            rec("quantile_field", "coupling_F_p75_minus_p50", fl[f]["rhoF"][2, k] - fl[f]["rhoF"][1, k], v,
                horizon=h, field=f, n=int(min(fl[f]["n"])), k=len(fl[f]["cohorts"]))
    return out


# --- calibration of the intervals under a within-cell null ---------------------------------------
# Null data on the real panel structure (cells, institutions, F, G): in every field x cohort cell the
# institutions' earnings vectors (all quantiles and horizons jointly) are permuted, independently across cells.
# This keeps only within-cell sampling noise (no field effect, no institution effect shared across cells), so the
# variance of a field-mean statistic across permuted datasets is its true sampling variance. Each interval's
# variance estimate, computed on single permuted datasets, is compared with it.
NCAL_TRUTH, NCAL_DATA, NCAL_B = 1000, 8, 400
CAL_STATS = ["slope_all", "dG_minus_dF_all", "bF_y1_all", "F_p75_minus_p50 (mean y1/y5/y10)",
             "F_p25_minus_p50 (mean y1/y5/y10)"]


def cal_from(rF, rG, rFG):
    """(B,3 quantiles,3 horizons) Spearman arrays with F and G, r_FG (B,) -> (B, 5) CAL_STATS per cell."""
    d = (1 - rFG ** 2)[:, None]
    d = np.where(d > 1e-12, d, np.nan)
    bF = (rF[:, 1] - rFG[:, None] * rG[:, 1]) / d
    bG = (rG[:, 1] - rFG[:, None] * rF[:, 1]) / d
    return np.stack([rF[:, 1] @ W_SLOPE, (bG - bF) @ W_SLOPE, bF[:, 0], (rF[:, 2] - rF[:, 1]).mean(-1),
                     (rF[:, 0] - rF[:, 1]).mean(-1)], -1)


def cal_cell(P, G, E, W):
    rF, rG = w_metrics(P, G, E, W)
    return cal_from(rF, rG, wcorr(wrank(P, W), wrank(G, W), W))


def calibration(v: str, qp: pd.DataFrame) -> dict:
    cells = [(k, g) for k, g in qp.groupby(["field", "grad_cohort"], sort=True) if len(g) >= NMIN]
    univ = sorted(set().union(*[set(g.inst_key) for _, g in cells]))
    iid = {k: i for i, k in enumerate(univ)}
    NI = len(univ)
    fields = sorted({f for (f, _), _ in cells})
    K = len(fields)
    cf = np.array([fields.index(f) for (f, _), _ in cells])
    ncf = np.bincount(cf, minlength=K)
    arrs = cell_arrays(cells, iid)
    rng = rng_for(f"calib_{v}")

    def fmean(S):                                     # (ncell, B, 5) -> (K, B, 5)
        out = np.zeros((K,) + S.shape[1:])
        np.add.at(out, cf, S)
        return out / ncf[:, None, None]

    def draw(Xs, B):
        """cell statistics (len(Xs), B, 5) under B institution draws; replicates with an undefined cell statistic
        are redrawn (as in the main analysis)."""
        p = np.full(NI, 1.0 / NI)
        W = rng.multinomial(NI, p, size=B).astype(float)
        S = np.stack([cal_cell(P, G, E, W[:, ix]) for P, G, E, ix in Xs])
        for _ in range(100):
            bad = ~np.isfinite(S).all((0, 2))
            if not bad.any():
                return S
            W[bad] = rng.multinomial(NI, p, size=int(bad.sum())).astype(float)
            S[:, bad] = np.stack([cal_cell(P, G, E, W[bad][:, ix]) for P, G, E, ix in Xs])
        raise AssertionError("calibration: could not draw non-degenerate weights")

    # truth
    T = []
    for P, G, E, ix in arrs:
        n = len(P)
        perm = np.argsort(rng.random((NCAL_TRUTH, n)), axis=1)
        rP, rG_ = rk(P)[None, :], rk(G)[None, :]
        rF, rGq = np.empty((NCAL_TRUTH, 3, 3)), np.empty((NCAL_TRUTH, 3, 3))
        for j in range(3):
            for k in range(3):
                re_ = rk(E[:, j, k][perm])
                rF[:, j, k], rGq[:, j, k] = corr_rows(rP, re_), corr_rows(rG_, re_)
        T.append(cal_from(rF, rGq, np.repeat(corr_rows(rP, rG_), NCAL_TRUTH)))
    TT = fmean(np.stack(T)).mean(0)                   # (NCAL_TRUTH, 5)
    assert np.isfinite(TT).all()
    vtrue = TT.var(0, ddof=1)
    rows = []
    for dset in range(NCAL_DATA):
        Xs = [(P, G, E[rng.permutation(len(P))], ix) for P, G, E, ix in arrs]
        pf = fmean(np.stack([cal_cell(P, G, E, np.ones((1, len(P)))) for P, G, E, _ in Xs]))[:, 0]   # (K, 5)
        Ssh = fmean(draw(Xs, NCAL_B))                                                          # (K, B, 5)
        Sin = np.empty_like(Ssh)
        for i in range(K):
            js = np.flatnonzero(cf == i)
            Sin[i] = draw([Xs[j] for j in js], NCAL_B).mean(0)
        fi = rng.integers(K, size=(NCAL_B, K))
        xb = Ssh[fi, np.arange(NCAL_B)[:, None]].mean(1)
        cb, ib = Ssh.mean(0), Sin.mean(0)
        vf = pf.var(0, ddof=1) / K
        vx, vs, vi = xb.var(0, ddof=1), cb.var(0, ddof=1), ib.var(0, ddof=1)
        for m, st in enumerate(CAL_STATS):
            rows.append(dict(dataset=dset, stat=st, crossed=vx[m] / vtrue[m], twoway=(vf[m] + vs[m] - vi[m]) / vtrue[m],
                             alt=(vx[m] - 2 * vi[m]) / vtrue[m], field=vf[m] / vtrue[m], shared=vs[m] / vtrue[m],
                             indep=vi[m] / vtrue[m]))
    df = pd.DataFrame(rows)
    agg_ = df.groupby("stat", sort=False)[["crossed", "twoway", "alt", "field", "shared", "indep"]].agg(["mean", "min", "max"])
    for st in CAL_STATS:
        for col in ("crossed", "twoway", "alt", "field", "shared", "indep"):
            rec("calibration", f"variance ratio to true sampling variance: {col}", float(agg_.loc[st, (col, "mean")]),
                v, float(agg_.loc[st, (col, "min")]), float(agg_.loc[st, (col, "max")]), k=K, n=NCAL_DATA,
                field=st, note=f"within-cell permutation null; truth from {NCAL_TRUTH} permuted datasets; mean (CI "
                               f"columns: min, max) over {NCAL_DATA} permuted datasets x {NCAL_B} replicates per draw")
        rec("calibration", "true sampling SD", float(np.sqrt(vtrue[CAL_STATS.index(st)])), v, k=K, field=st,
            note=f"{NCAL_TRUTH} permuted datasets")
    return dict(agg=agg_, K=K, n_cells=len(cells), n_inst=NI, vtrue=dict(zip(CAL_STATS, vtrue)))


# ---------------------------------------------------------------------------------------------
# (d) cross-source replication (scripts/32 logic)
# ---------------------------------------------------------------------------------------------
def gap_over(d, ycol):
    out = {}
    for f, g in d.groupby("field"):
        if g[ycol].nunique() >= 3 and len(g) >= 8:
            out[f] = dict(gap=1 - spearmanr(g.F, g[ycol])[0], n=len(g))
    return out


def crosssource(v: str, ar, t28):
    s32.PSEOE, s32.PSEO_INST = REL[v]["earn"], REL[v]["inst"]
    pseo = s32.load_pseo_ba("y5")
    pseo = alias_keys(pseo, REL[v]["alias"])
    arF = ar[["inst_key", "field", "prestige_score"]].rename(columns={"prestige_score": "F"})
    mp, ms = pseo.merge(arF, on=["inst_key", "field"]), t28.merge(arF, on=["inst_key", "field"])
    gp = gap_over(mp, "y_pseo")
    gs = gap_over(ms, "y_scorecard")
    cross = pd.DataFrame([dict(field=f, c_PSEO=1 - gp[f]["gap"], n_PSEO=gp[f]["n"],
                               c_Scorecard=1 - gs[f]["gap"], n_Scorecard=gs[f]["n"]) for f in sorted(gp) if f in gs])
    lvl = pseo.merge(t28, on=["inst_key", "field"]).dropna(subset=["y_pseo", "y_scorecard"])
    lvl_rho = spearmanr(lvl.y_pseo, lvl.y_scorecard)[0]
    rho, lo, hi = s32.boot_corr(cross.c_PSEO, cross.c_Scorecard)
    big = cross[(cross.n_PSEO >= NMIN)]
    rho15, lo15, hi15 = s32.boot_corr(big.c_PSEO, big.c_Scorecard)
    n_inst = pseo.merge(arF, on=["inst_key", "field"]).inst_key.nunique()
    res = dict(lvl_rho=lvl_rho, lvl_n=len(lvl), rho=rho, lo=lo, hi=hi, k=len(cross), rho15=rho15, lo15=lo15,
               hi15=hi15, k15=len(big), n_inst=n_inst, cross=cross,
               mean_c_pseo=cross.c_PSEO.mean(), mean_c_sc=cross.c_Scorecard.mean(),
               mcse=mcse_from_ci(lo, hi, 4000))
    smp = "PSEO y5 p50 pooled 0000 vs Scorecard EARN_MDN_4YR; fields with n>=8"
    rec("crosssource", "level_spearman_inst_x_field", lvl_rho, v, n=len(lvl), sample="same institution x field")
    rec("crosssource", "spearman_field_coupling", rho, v, lo, hi, k=len(cross), sample=smp)
    res.update(crosssource_clustered(v, mp, ms, cross, rho))
    rec("crosssource", "spearman_field_coupling [crossed bootstrap]", rho, v, res["xlo"], res["xhi"], k=len(cross),
        sample=smp, note=f"fields resampled x one institution multinomial draw shared by the PSEO and Scorecard "
                         f"couplings ({NBOOT}; {res['xs_n_inst']} institutions); MC SE of CI endpoints="
                         f"{res['xmcse']:.2g}; replicates redrawn={res['xs_redrawn']}")
    rec("crosssource", "spearman_field_coupling [institution-cluster bootstrap, fields fixed]", rho, v, res["clo"],
        res["chi"], k=len(cross), sample=smp, note=f"fields held fixed ({NBOOT}); MC SE of CI endpoints={res['cmcse']:.2g}")
    rec("crosssource", "spearman_field_coupling [two-way field x institution cluster variance]", rho, v, res["tlo"],
        res["thi"], k=len(cross), sample=smp, note=tw_note(res) + f" ({NBOOT} replicates per draw; V_field = jackknife "
        f"over fields; independent-draw replicates redrawn={res['xs_redrawn_i']})")
    rec("crosssource", "spearman_field_coupling replicate median [crossed bootstrap]", res["xmed"], v, k=len(cross),
        note="median of the crossed replicates (attenuation check)")
    rec("crosssource", "spearman_field_coupling replicate median [institution-cluster bootstrap, fields fixed]",
        res["cmed"], v, k=len(cross), note="median of the fields-fixed replicates (attenuation check)")
    rec("crosssource", "spearman_field_coupling_nPSEO>=15", rho15, v, lo15, hi15, k=len(big),
        sample=smp.replace("n>=8", "n_PSEO>=15"))
    rec("crosssource", "pseo_wapman_institutions", n_inst, v)
    rec("crosssource", "mean_coupling_PSEO", res["mean_c_pseo"], v, k=len(cross))
    rec("crosssource", "mean_coupling_Scorecard", res["mean_c_sc"], v, k=len(cross))
    for _, r in cross.iterrows():
        rec("crosssource_field", "coupling_PSEO", r.c_PSEO, v, field=r.field, n=r.n_PSEO,
            note=f"coupling_Scorecard={r.c_Scorecard:.6g}; n_Scorecard={r.n_Scorecard}")
    return res


def mcse_from_ci(lo: float, hi: float, B: int) -> float:
    """Approximate Monte Carlo SE of a percentile-CI endpoint when only the CI is available (scripts/32's
    boot_corr): bootstrap SD taken as (hi - lo) / (2 * 1.96)."""
    phi = np.exp(-Z975 ** 2 / 2) / np.sqrt(2 * np.pi)
    return float(np.sqrt(0.025 * 0.975 / B) * ((hi - lo) / (2 * Z975)) / phi)


def crosssource_clustered(v: str, mp: pd.DataFrame, ms: pd.DataFrame, cross: pd.DataFrame, rho: float) -> dict:
    """Crossed (fields resampled x one institution draw) and fields-fixed institution-cluster CIs for
    Spearman(field coupling PSEO, field coupling Scorecard). The same institution draw is applied to both
    sources' field couplings, so institution sampling noise that the two sources share is carried into the
    interval (scripts/32's field bootstrap holds each field's couplings fixed)."""
    fields = list(cross.field)
    mp, ms = mp[mp.field.isin(fields)], ms[ms.field.isin(fields)]
    univ = sorted(set(mp.inst_key) | set(ms.inst_key))
    iid = {k: i for i, k in enumerate(univ)}
    A = []
    for f in fields:
        for d, ycol in ((mp, "y_pseo"), (ms, "y_scorecard")):
            g = d[d.field == f]
            A.append((g.F.to_numpy(float), g[ycol].to_numpy(float), np.array([iid[k] for k in g.inst_key])))

    def ev(Wf, js=None):
        return [dict(c=wcorr(wrank(P, Wf[:, ix]), wrank(y, Wf[:, ix]), Wf[:, ix]))
                for P, y, ix in (A if js is None else [A[j] for j in js])]

    obs = ev(np.ones((1, len(univ))))
    oP = np.array([obs[2 * i]["c"][0] for i in range(len(fields))])
    oS = np.array([obs[2 * i + 1]["c"][0] for i in range(len(fields))])
    assert np.allclose(oP, cross.c_PSEO, rtol=0, atol=1e-10) and np.allclose(oS, cross.c_Scorecard, rtol=0, atol=1e-10)
    assert np.isclose(spearmanr(oP, oS)[0], rho, rtol=0, atol=1e-12)
    W, res, redrawn = draw_valid(len(univ), f"xs_inst_{v}", ev)
    cP = np.stack([res[2 * i]["c"] for i in range(len(fields))])           # (K, NBOOT)
    cS = np.stack([res[2 * i + 1]["c"] for i in range(len(fields))])
    fi = rng_for(f"xs_fields_{v}").integers(len(fields), size=(NBOOT, len(fields)))
    b = np.arange(NBOOT)[:, None]
    xb = corr_rows(rk(cP[fi, b]), rk(cS[fi, b]))
    cb = corr_rows(rk(cP.T), rk(cS.T))
    assert np.isfinite(xb).all() and np.isfinite(cb).all()
    (xlo, xhi), (clo, chi) = ci(xb), ci(cb)
    # two-way variance: independent institution draw per field (shared by that field's PSEO and Scorecard
    # couplings); V_field = jackknife over fields of the Spearman correlation
    cPi, cSi, redrawn_i = np.empty_like(cP), np.empty_like(cS), 0
    for i, f in enumerate(fields):
        _, r_, rd_ = draw_valid(len(univ), f"xs_inst_ind_{v}_{f}", lambda Wf, i=i: ev(Wf, [2 * i, 2 * i + 1]))
        cPi[i], cSi[i] = r_[0]["c"], r_[1]["c"]
        redrawn_i += rd_
    ib = corr_rows(rk(cPi.T), rk(cSi.T))
    fb = corr_rows(rk(oP[fi]), rk(oS[fi]))
    vf = jack_var(lambda a, b: spearmanr(a, b)[0], oP, oS)
    assert np.isfinite(ib).all() and np.isfinite(fb).all()
    out = dict(xlo=xlo, xhi=xhi, xmcse=mc_se_endpoint(xb), clo=clo, chi=chi, cmcse=mc_se_endpoint(cb),
               xmed=float(np.median(xb)), cmed=float(np.median(cb)), imed=float(np.median(ib)), xs_n_inst=len(univ),
               xs_redrawn=redrawn, xs_redrawn_i=redrawn_i,
               xs_overlap=len(set(zip(mp.inst_key, mp.field)) & set(zip(ms.inst_key, ms.field))) / len(mp))
    out.update(twoway(rho, vf, cb, ib, fb, xb))
    return out


# ---------------------------------------------------------------------------------------------
_T0 = time.perf_counter()


def stage(msg: str):
    """Progress to stdout only (never written to an output file)."""
    print(f"[{time.perf_counter() - _T0:6.0f}s] {msg}", flush=True)


def main():
    # the explicit-path loader must reproduce the default loader on the pinned release
    a = load_er_pseo("undergrad", "y5", fields=FIELDS66)
    b = load_er_pseo("undergrad", "y5", fields=FIELDS66, path=PSEO_EARN, inst_path=PSEO_INST)
    assert a.equals(b)

    REL["new_joinfix"]["alias"] = alias_full()
    stage("loader check done")
    prov = provenance()
    stage("provenance done")
    ar = load_ar_wapman(fields=FIELDS66)
    gen_raw = s28.load_generic()
    gen = gen_raw.assign(G=-gen_raw["g_rank"].astype(float))[["inst_key", "G"]]

    # (a)
    newtab, tot, quart, new_ids, added_states = coverage(ar, gen_raw)
    ja = join_audit(ar, gen_raw)
    rcov = readme_coverage()
    stage("coverage done")

    # field groups: Scorecard 4YR baseline, exactly as scripts/52
    sc4 = load_er_scorecard("undergrad", earn_col="EARN_MDN_4YR", count_col="EARN_COUNT_WNE_4YR", fields=FIELDS66)
    gm4 = compute_gap_map(ar, sc4, "undergrad", B=s52.B_GAP, seed=s52.SEED)
    cls = s52.classify(gm4)
    grp_of = dict(zip(cls.field, cls.group))

    # (b)
    CT = {}
    CT["old"] = career("old", ar, gen, grp_of)
    stage("career old done")
    for v in VARIANTS[1:]:
        CT[v] = career(v, ar, gen, grp_of, ref_fields=CT["old"]["fields"])
        stage(f"career {v} done")
    new_panel_insts = sorted(set(CT["new"]["panel"].inst_key) - set(CT["old"]["panel"].inst_key))
    new_panel_insts_a = sorted(set(CT["new_alias"]["panel"].inst_key) - set(CT["old"]["panel"].inst_key))
    rec("career_panel", "institutions_new_in_panel", len(new_panel_insts), "new", note="; ".join(new_panel_insts))
    rec("career_panel", "institutions_new_in_panel", len(new_panel_insts_a), "new_alias", note="; ".join(new_panel_insts_a))
    # per-field agreement of slopes, old vs new
    fo, fn = CT["old"]["fl"], CT["new"]["fl"]
    common = sorted(set(fo) & set(fn))
    fs_rho = spearmanr([fo[f]["rhoF_slope"] for f in common], [fn[f]["rhoF_slope"] for f in common])[0]
    fs_mad = float(np.mean([abs(fo[f]["rhoF_slope"] - fn[f]["rhoF_slope"]) for f in common]))
    rec("career_time", "field_slope_spearman_old_vs_new", fs_rho, "old_vs_new", k=len(common))
    rec("career_time", "field_slope_mean_abs_diff_old_vs_new", fs_mad, "old_vs_new", k=len(common))

    # (c)
    QT = {}
    for v in ("old", "new", "new_alias", "new_joinfix"):
        QT[v] = quantile(v, ar, gen)
        # the quantile panel is the scripts/52 p50 panel: same rows, same p50 values, same p50 coupling
        qp, cp = QT[v]["panel"], CT[v]["panel"]
        m = qp.merge(cp, on=["field", "inst_key", "grad_cohort"], how="outer", indicator=True)
        assert (m["_merge"] == "both").all() and len(m) == len(qp) == len(cp), v
        assert all(np.allclose(m[f"p50_{h}"].to_numpy(float), m[f"e_{h}"].to_numpy(float), rtol=1e-12, atol=0)
                   for h in HZ), v
        assert QT[v]["fields"] == CT[v]["fields"], v
        assert all(np.allclose(QT[v]["fl"][f]["rhoF"][1], CT[v]["fl"][f]["rhoF_h"], rtol=0, atol=1e-12)
                   for f in QT[v]["fields"]), v
        del m
        stage(f"quantile {v} done")

    CAL = calibration("new", QT["new"]["panel"])
    stage("calibration done")

    # (d)
    t28 = s28.build_table()[["inst_key", "field", "y"]].rename(columns={"y": "y_scorecard"})
    CS = {v: crosssource(v, ar, t28) for v in VARIANTS}
    stage("cross-source done")

    write_csv()
    make_figure(quart, CT, QT, CS)
    write_md(prov, newtab, tot, quart, new_ids, added_states, ja, CT, QT, CS, new_panel_insts, new_panel_insts_a,
             fs_rho, fs_mad, common, cls, rcov, CAL)
    for v in VARIANTS:
        R = CT[v]["R"]
        print(v, {k: round(R[k]["est"], 4) for k in ("slope_all", "slope_integrated", "slope_int_minus_others",
                                                       "D_cal_all", "D_within_all", "dbF_all", "dbG_all")})


# ---------------------------------------------------------------------------------------------
def write_csv():
    df = pd.DataFrame(ROWS)
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"[csv] {OUT_CSV} ({len(df)} rows)")


COL = {"old": "#2a78d6", "new_oldinst": "#8c8c8c", "new": "#eb6834", "new_alias": "#1baf7a",
       "new_joinfix": "#4a3aa7"}


def make_figure(quart, CT, QT, CS):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), layout="constrained")
    # A. coverage by quartile
    ax = axes[0, 0]
    qs = list(dict.fromkeys(quart.quartile))
    vs = ["old", "new", "new_alias", "new_joinfix"]
    w = 0.2
    for i, v in enumerate(vs):
        d = quart[quart.variant == v].set_index("quartile").loc[qs]
        x = np.arange(len(qs)) + (i - 1.5) * w
        ax.bar(x, d.covered, width=w - 0.02, color=COL[v], label=REL[v]["label"])
        for xx, c, t in zip(x, d.covered, d.total):
            ax.text(xx, c + 0.8, f"{c}", ha="center", va="bottom", fontsize=7.5, color="#333333")
    ax.set_xticks(np.arange(len(qs))); ax.set_xticklabels([f"{q}\n(n={int(quart[quart.quartile == q].total.iloc[0])})" for q in qs])
    ax.set_ylabel("Wapman institutions with a usable released y5 BA cell")
    ax.set_title("A. PSEO coverage of Wapman institutions by academia-wide rank quartile", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # B. career-time forest
    ax = axes[0, 1]
    items = [("slope, all fields", "slope_all"), ("slope, integrated", "slope_integrated"),
             ("integrated minus others", "slope_int_minus_others"), ("y5->y10 segment, all", "late_slope_all"),
             ("within-cohort (y10-y1)/9", "D_within_all"), ("calendar-matched /9", "D_cal_all"),
             ("cross-cohort drift at y1 /9", "D_cross_all"), ("dF (field prestige coef.)", "dbF_all"),
             ("dG (academia-wide rank coef.)", "dbG_all"), ("dG - dF", "dG_minus_dF_all")]
    off = {"old": 0.30, "new_oldinst": 0.15, "new": 0.0, "new_alias": -0.15, "new_joinfix": -0.30}
    for i, (lab, key) in enumerate(items):
        y = len(items) - i
        for v in VARIANTS:
            r = CT[v]["R"][key]
            ax.plot([r["xlo"], r["xhi"]], [y + off[v]] * 2, color=COL[v], lw=1.6)
            ax.plot([r["tlo"], r["thi"]], [y + off[v]] * 2, color=COL[v], lw=4.0, alpha=0.45, solid_capstyle="butt")
            ax.plot(r["est"], y + off[v], "o", color=COL[v], ms=5,
                    label=REL[v]["label"] if i == 0 else None)
    ax.axvline(0, color="#555555", lw=0.7, ls=":")
    ax.set_yticks(range(len(items), 0, -1)); ax.set_yticklabels([l for l, _ in items], fontsize=8.5)
    ax.set_xlabel("change in coupling (or coefficient) per year; thick band: 95% two-way field x institution\n"
                  "cluster CI (standard); thin line: 95% crossed field x institution bootstrap CI (conservative)")
    ax.set_title(f"B. Fixed-cohort career-time analysis (scripts/52 engine), by data release ({NBOOT} replicates)",
                 fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # C. paired quantile differences (the key comparison), both releases
    ax = axes[1, 0]
    xpos = {h: i for i, h in enumerate(HZ + [AVG])}
    series = [("p75", "p75 - p50", "#08306b", -0.2), ("p25", "p25 - p50", "#d9822b", 0.12)]
    S = QT["new"]["S"]
    for a_, lab, col, x0 in series:
        for kind, dx, face, lw, lo_k, hi_k in (("two-stage (earlier; ignores recurring institutions)", 0.0, "white", 1.2, "lo", "hi"),
                                               ("crossed field x institution (conservative)", 0.09, "#bbbbbb", 1.2, "xlo", "xhi"),
                                               ("two-way field x institution (standard)", 0.18, col, 2.4, "tlo", "thi")):
            for h, xi in xpos.items():
                r = S[(f"F_{a_}_minus_p50", h, "")]
                x = xi + x0 + dx
                ax.plot([x, x], [r[lo_k], r[hi_k]], color=col, lw=lw)
                ax.plot(x, r["est"], "o", ms=6, mfc=face, mec=col, mew=1.4,
                        label=(f"{lab}, 95% CI {kind}" if h == "y1" else None))
            if kind.startswith("two-way"):
                for h, xi in xpos.items():
                    r = S[(f"F_{a_}_minus_p50", h, "")]
                    if np.isfinite(r["p"]):
                        ax.text(xi + x0 + dx + 0.05, r["thi"] if a_ == "p75" else r["tlo"], pq(r['p']),
                                fontsize=7, color=col, va="bottom" if a_ == "p75" else "top")
    ax.axhline(0, color="#555555", lw=0.7, ls=":")
    ax.axvline(len(HZ) - 0.5, color="#cccccc", lw=0.6)
    y0, y1_ = ax.get_ylim()
    ax.set_ylim(y0 - 0.32 * (y1_ - y0), y1_ + 0.05 * (y1_ - y0))    # room for the legend below the data
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels(["y1", "y5", "y10", "mean of horizons\n(post hoc)"])
    ax.set_xlabel("years since graduation")
    ax.set_ylabel("difference in coupling vs p50 (Spearman with field prestige)")
    ax.set_title(f"C. Quantile coupling minus median coupling, V4.14.1 fixed cohorts 2001-2010 (k={QT['new']['K']})\n"
                 f"({NBOOT} bootstrap replicates; p = institution-clustered permutation, {NPERM})", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", ncol=1)
    ax.spines[["top", "right"]].set_visible(False)

    # D. cross-source scatter
    ax = axes[1, 1]
    c = CS["new"]["cross"]
    co = CS["old"]["cross"]
    ax.scatter(co.c_Scorecard, co.c_PSEO, s=30, facecolors="none", edgecolors=COL["old"], lw=1.0, zorder=3,
               label=f"V4.13.0: Spearman {CS['old']['rho']:+.2f} (k={CS['old']['k']})")
    ax.scatter(c.c_Scorecard, c.c_PSEO, s=22, color=COL["new"],
               label=f"V4.14.1: Spearman {CS['new']['rho']:+.2f}, two-way CI [{CS['new']['tlo']:+.2f}, {CS['new']['thi']:+.2f}] (k={CS['new']['k']})")
    lim = [min(c.c_Scorecard.min(), c.c_PSEO.min(), co.c_PSEO.min()) - 0.05, 1.0]
    ax.plot(lim, lim, color="#999999", lw=0.7, ls=":")
    ax.axhline(0, color="#bbbbbb", lw=0.5); ax.axvline(0, color="#bbbbbb", lw=0.5)
    ax.set_xlabel("field coupling, Scorecard 4YR median (all Scorecard x Wapman institutions)")
    ax.set_ylabel("field coupling, PSEO y5 median, pooled cohorts (PSEO x Wapman)")
    ax.set_title("D. Cross-source replication of field coupling (scripts/32 spec)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    print(f"[fig] {OUT_FIG}")


# ---------------------------------------------------------------------------------------------
def fm(v, d=3):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:+.{d}f}"


def fp(p):
    return "—" if not np.isfinite(p) else "<0.001" if p < 0.001 else f"{p:.3f}"


def pe(p):
    """'p<0.001' / 'p=0.123' for prose and table cells."""
    return "p " + fp(p) if np.isfinite(p) and p < 0.001 else "p=" + fp(p)


def cis(r, d=3):
    return f"[{fm(r['lo'], d)}, {fm(r['hi'], d)}]"


def xci(r, d=3):
    """crossed field x institution bootstrap CI"""
    return f"[{fm(r['xlo'], d)}, {fm(r['xhi'], d)}]"


def cci(r, d=3):
    """institution-cluster bootstrap CI, fields fixed"""
    return f"[{fm(r['clo'], d)}, {fm(r['chi'], d)}]"


def tci(r, d=3):
    """two-way (field, institution) cluster-variance CI"""
    return f"[{fm(r['tlo'], d)}, {fm(r['thi'], d)}]"


def tciz(r, d=3):
    """two-way CI with z"""
    return f"{tci(r, d)} (z={r['tz']:.2f})"


def fpq(p):
    """p of this script's permutations (NPERM): the minimum attainable 1/(NPERM+1) is shown as '≤0.001'."""
    return "—" if not np.isfinite(p) else "≤0.001" if p <= 1 / (NPERM + 1) + 1e-12 else f"{p:.3f}"


def pq(p):
    return "p" + fpq(p) if fpq(p).startswith("≤") else "p=" + fpq(p)


def qdiff(r):
    """'est; two-way CI (z); crossed CI; fields-fixed CI; two-stage CI; p' for a quantile difference (parts that
    exist)."""
    return (f"{fm(r['est'])}" + (f"; {tciz(r)}" if "tlo" in r else "") + f"; {xci(r)}"
            + (f"; {cci(r)}" if "clo" in r else "") + f"; {cis(r)}"
            + (f"; {pq(r['p'])}" if np.isfinite(r["p"]) else ""))


def join_and(xs):
    xs = list(xs)
    if not xs:
        return ""
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def excl0(r):
    return (r["lo"] > 0) or (r["hi"] < 0)


def other_variants_text(QT) -> str:
    """How the quantile pattern looks in the non-primary variants (old, new_alias, new_joinfix)."""
    vs = ("old", "new_alias", "new_joinfix")
    r75 = {v: {h: QT[v]["S"][("F_p75_minus_p50", h, "")] for h in HZ + [AVG]} for v in vs}
    r25 = {v: {h: QT[v]["S"][("F_p25_minus_p50", h, "")] for h in HZ + [AVG]} for v in vs}
    x75_incl0 = all(r75[v][h]["xlo"] <= 0 <= r75[v][h]["xhi"] for v in vs for h in HZ + [AVG])
    x25_below = all(r25[v][h]["xhi"] < 0 for v in vs for h in HZ + [AVG])
    p_early = [r75[v][h]["p"] for v in vs for h in ("y1", "y5")]
    p_late = [r75[v]["y10"]["p"] for v in vs]
    over = [f"{REL[v]['label']} at {h} (p={fpq(r75[v][h]['p'])})" for v in vs for h in ("y1", "y5")
            if r75[v][h]["p"] >= 0.05]
    tstat = {(v, h): x_status(r75[v][h]["tlo"], r75[v][h]["thi"], r75[v][h]["tmcse"]) for v in vs for h in HZ + [AVG]}
    t25 = all(r25[v][h]["thi"] < 0 for v in vs for h in HZ + [AVG])
    return ("In V4.13.0 and the two alias variants, the two-way CI for the average of p75 - p50 is "
            + "; ".join(f"{REL[v]['label']} {tciz(r75[v][AVG])}" for v in vs)
            + " (by horizon, above 0 at: "
            + "; ".join(f"{REL[v]['label']} {join_and([h for h in HZ if tstat[(v, h)] == 'above 0']) or 'none'}"
                        for v in vs)
            + ("), and every two-way CI for p25 - p50 is below 0. " if t25 else "). ")
            + "Also in these variants, "
            + ("every crossed CI for p75 - p50 includes 0" if x75_incl0 else "not every crossed CI for p75 - p50 includes 0")
            + (", every crossed CI for p25 - p50 is below 0" if x25_below else ", not every crossed CI for p25 - p50 is below 0")
            + f", and the clustered permutation p for p75 - p50 ranges from {fpq(min(p_early))} to {fpq(max(p_early))} "
            f"at y1 and y5 and from {fpq(min(p_late))} to {fpq(max(p_late))} at y10"
            + (f"; it is above 0.05 in {join_and(over)}" if over else "") + " (tables below).")


INF = ("two-way", "crossed", "fields fixed", "two-stage")
ORDER = {"does not hold": 0, "edge": 1, "holds": 2}


def st_pos(lo, hi, mcse) -> str:
    """Status of 'CI > 0': holds / edge (lower endpoint within 2 Monte Carlo SE of 0) / does not hold."""
    s_ = x_status(lo, hi, mcse)
    return "holds" if s_ == "above 0" else "edge" if s_ == "touches 0" else "does not hold"


def st_span(lo, hi, mcse) -> str:
    """Status of 'CI spans 0': holds / edge (an endpoint within 2 Monte Carlo SE of 0) / does not hold."""
    s_ = x_status(lo, hi, mcse)
    return "holds" if s_ == "includes 0" else "edge" if s_ == "touches 0" else "does not hold"


def st_min(*xs) -> str:
    return min(xs, key=ORDER.get)


def ci3(r: dict, kind: str) -> tuple:
    """(lo, hi, Monte Carlo SE of the endpoints) of one of the three kinds of interval."""
    return {"two-way": (r["tlo"], r["thi"], r["tmcse"]), "crossed": (r["xlo"], r["xhi"], r["xmcse"]),
            "fields fixed": (r["clo"], r["chi"], r["cmcse"]), "two-stage": (r["lo"], r["hi"], r["mcse"])}[kind]


def conclusions(CT, CS):
    """Qualitative Phase 1 / Result 3 statements that rest on PSEO: status per variant under each kind of
    interval (two-way field x institution cluster variance; crossed field x institution bootstrap;
    institution-cluster bootstrap with fields fixed; scripts/52's two-stage bootstrap, or scripts/32's field
    bootstrap for the cross-source row)."""
    out = {}
    for v in VARIANTS:
        R, C = CT[v]["R"], CS[v]
        out[v] = {}
        for kind in INF:
            def pos(key):
                return st_pos(*ci3(R[key], kind))

            def spn(key):
                return st_span(*ci3(R[key], kind))
            out[v][kind] = {
                "rise": pos("slope_all"),
                "not_int_specific": spn("slope_int_minus_others"),
                "career_part_pos": st_min(pos("D_cal_all"),
                                          "holds" if R["D_cross_all"]["est"] >= 0 else "does not hold"),
                "brand_G": st_min(pos("dbG_all"), spn("dbF_all")),
                "brand_diff": pos("dG_minus_dF_all"),
                "xsource": st_min(st_pos(*ci3(C, kind)), "holds" if C["rho"] > 0.5 else "does not hold"),
            }
    return out


def changed_between(conc, k, vs) -> bool:
    """A statement counts as changed between variants only if it holds in one and does not hold in another
    under the same kind of interval ('edge' is within Monte Carlo error of both)."""
    return any({"holds", "does not hold"} <= {conc[v][kind][k] for v in vs} for kind in INF)


def field_change_text(CT):
    fo, fn = CT["old"]["fl"], CT["new"]["fl"]
    added = sorted(set(fn) - set(fo)); dropped = sorted(set(fo) - set(fn))
    common = sorted(set(fo) & set(fn))
    big = max(common, key=lambda f: (abs(fn[f]["rhoF_slope"] - fo[f]["rhoF_slope"]), f))
    t = (f"Fields entering the V4.14.1 panel: {', '.join(LAB.get(f, f) for f in added) or 'none'}; leaving: "
         f"{', '.join(LAB.get(f, f) for f in dropped) or 'none'}. Largest per-field change: {LAB.get(big, big)} "
         f"{fm(fo[big]['rhoF_slope'], 4)} -> {fm(fn[big]['rhoF_slope'], 4)}/yr (cohorts "
         f"{'+'.join(fo[big]['cohorts'])} -> {'+'.join(fn[big]['cohorts'])}; n min {min(fo[big]['n'])} -> "
         f"{min(fn[big]['n'])}).")
    return t


CONC_LAB = {"rise": "coupling rises with years since graduation (all-field slope CI > 0)",
            "not_int_specific": "rise not specific to integrated fields (integrated-minus-others CI spans 0)",
            "career_part_pos": "calendar-matched contrast CI > 0 and cross-cohort drift >= 0 (bound applies)",
            "brand_G": "the academia-wide rank's loading grows and field prestige's does not (dG CI > 0, dF CI spans 0)",
            "brand_diff": "G's loading grows faster than F's (dG - dF CI > 0)",
            "xsource": "Scorecard-PSEO field coupling replicates (Spearman > 0.5, CI > 0)"}


def write_md(prov, newtab, tot, quart, new_ids, added_states, ja, CT, QT, CS, npi, npia, fs_rho, fs_mad, common, cls,
             rcov, CAL):
    L = []
    Ro, Rn, Rr, Ra = (CT[v]["R"] for v in ("old", "new_oldinst", "new", "new_alias"))
    Sn, So, Sa = QT["new"]["S"], QT["old"]["S"], QT["new_alias"]["S"]
    po, pn, dff = prov["old"], prov["new"], prov["diff"]
    conc = conclusions(CT, CS)
    cag = CAL["agg"]

    def cal_lo(col):
        return float(cag[(col, "mean")].min())

    def cal_hi(col):
        return float(cag[(col, "mean")].max())
    changed = [k for k in CONC_LAB if changed_between(conc, k, VARIANTS)]
    changed_rel = [k for k in CONC_LAB if changed_between(conc, k, ("old", "new"))]
    L.append("# PSEO refresh — the August 2026 release (V4.14.1 / 2026Q2)\n")
    L.append("Script: `scripts/59_pseo_refresh.py` (seeded; outputs byte-identical on re-run). It reuses the "
             "fixed-cohort engine of `scripts/52` and the cross-source loader of `scripts/32` by import "
             "(neither file edited) and reads the new release through the optional `path=`/`inst_path=` "
             "arguments added to `src/load_er.load_er_pseo` (defaults unchanged). Descriptive, not causal. "
             "Tables: `data/interim/pseo_refresh.csv`; figure: `outputs/figures/pseo_refresh.png`.\n")

    # ------------------------------------------------------------------ answer
    L.append("## Answer\n")
    ttl = "No." if not changed_rel else "Yes, in part."
    kn = int((newtab["join"] == "canonical").sum())
    ka = int((newtab["join"] == "alias").sum())
    yrs = pn["years"]
    late_states = [st for st in added_states.index if int(yrs.get(st, "0000-")[:4]) > int(COHORTS[-1])]
    n_late = int(sum(added_states[st] for st in late_states))
    fixed0 = newtab[newtab.state.isin(late_states)]
    rb = raw_ba(NEW_EARN)
    ins_n = pd.read_csv(NEW_INST, dtype=str)
    ins_n.columns = [c.strip().lstrip("\ufeff") for c in ins_n.columns]
    late_ids = set(ins_n[ins_n.institution_state.isin(late_states)].institution) & set(new_ids)
    n_late_fixed = int(sum(((rb.institution.isin(late_ids)) & rb.grad_cohort.isin(COHORTS) &
                            (rb[f"status_{h}_earnings"] == "1")).sum() for h in HZ))
    same_panel = CT["old"]["panel"].equals(CT["new_oldinst"]["panel"])
    same_status_rel = all(conc["old"][kind][k] == conc["new"][kind][k] for k in CONC_LAB for kind in INF)
    key_of = {"rise": "slope_all", "not_int_specific": "slope_int_minus_others", "career_part_pos": "D_cal_all",
              "brand_diff": "dG_minus_dF_all"}
    rel_diff_txt = []
    for k in CONC_LAB:
        for kind in INF:
            a_, b_ = conc["old"][kind][k], conc["new"][kind][k]
            if a_ == b_:
                continue
            t = f"'{CONC_LAB[k]}' under the {kind} interval, {a_} on V4.13.0 and {b_} on V4.14.1"
            if k in key_of:
                lo_o, _, se_o = ci3(Ro[key_of[k]], kind)
                lo_n, _, se_n = ci3(Rr[key_of[k]], kind)
                gap, tol2 = abs(lo_o - lo_n), 2 * np.hypot(se_o, se_n)
                t += (f" (lower endpoints {fm(lo_o, 4)} and {fm(lo_n, 4)}; they differ by {gap:.4f}, "
                      + ("less" if gap <= tol2 else "more") + f" than 2 Monte Carlo SE of a difference, {tol2:.4f})")
            rel_diff_txt.append(t)
    L.append(
        f"**Does the new release change a Phase 1 conclusion? {ttl}** Only two Phase 1 inputs read PSEO: the "
        "career-time result (scripts/52) and the Scorecard-PSEO replication (scripts/32). Selectivity (55), "
        "prestige reliability (56) and the three-level ICC (57) do not read PSEO, so the refresh cannot "
        "change them. "
        + ("Every PSEO-based statement in Table 1 has the same status on V4.13.0 and V4.14.1 under each of the "
           "three kinds of interval" if same_status_rel else
           ("No PSEO-based statement in Table 1 holds on one release and fails on the other under the same kind "
            "of interval. " if not changed_rel else
            "These statements hold on one release and not on the other: "
            + "; ".join(CONC_LAB[k] for k in changed_rel) + ". ")
           + "Statuses that differ between the releases: " + "; ".join(rel_diff_txt))
        + (". Across the five data variants (including the 4 aliases for added institutions and 19 further "
           "aliases for V4.13.0 institutions that the project's name join misses), no statement holds in one and "
           "does not hold in another (Table 1). " if not changed else
           ". Across all five data variants these statements change status: "
           + "; ".join(CONC_LAB[k] for k in changed) + " (Table 1). ")
        + "The inference standard of this note (revision 3) is the two-way (field, institution) cluster variance "
        "of Cameron, Gelbach and Miller (2011): V_field + V_institution - V_field x institution. It treats both "
        "fields and institutions as sampled (the same institutions recur across cohorts and fields) and counts the "
        "within-cell sampling noise once. Beside it are the crossed field x institution ('pigeonhole') bootstrap, "
        f"which is conservative here: under a within-cell null on the V4.14.1 panel its variance is "
        f"{cal_lo('crossed'):.1f} to {cal_hi('crossed'):.1f} times the true sampling variance, against "
        f"{cal_lo('twoway'):.2f} to {cal_hi('twoway'):.2f} for the two-way variance (section 6); and the "
        "institution-cluster bootstrap with fields held fixed. scripts/52's two-stage intervals, which redraw "
        "institutions separately in each cell, are shown only to reproduce scripts/52. "
        f"The all-field career-time slope is {fm(Ro['slope_all']['est'])}/yr on V4.13.0 and "
        f"{fm(Rr['slope_all']['est'])}/yr on V4.14.1 ({Rr['slope_all']['k']} fields; two-way CI "
        f"{tci(Rr['slope_all'])}; crossed CI {xci(Rr['slope_all'])}; institution-clustered horizon permutation "
        f"{pq(Rr['slope_all']['p_clu'])}). "
        f"The career-time bound moves from [{fm(Ro['D_cal_all']['est'])}, {fm(Ro['D_within_all']['est'])}] to "
        f"[{fm(Rr['D_cal_all']['est'])}, {fm(Rr['D_within_all']['est'])}]/yr; its lower end, the calendar-matched "
        f"contrast, has two-way CI {tci(Rr['D_cal_all'])} (crossed {xci(Rr['D_cal_all'])}) on V4.14.1. The "
        f"cross-source field-coupling correlation is {fm(CS['old']['rho'], 2)} on V4.13.0 and "
        f"{fm(CS['new']['rho'], 2)} on V4.14.1 (two-way CI [{fm(CS['new']['tlo'], 2)}, {fm(CS['new']['thi'], 2)}]; "
        f"field bootstrap [{fm(CS['new']['lo'], 2)}, {fm(CS['new']['hi'], 2)}]; the crossed percentile CI "
        f"[{fm(CS['new']['xlo'], 2)}, {fm(CS['new']['xhi'], 2)}] is biased downward, see section 5, and stays above "
        f"0), with {CS['new']['n_inst']} rather than {CS['old']['n_inst']} PSEO x Wapman institutions.\n")
    # the brand statements (Phase 1 wording check)
    bG_all_hold = all(conc[v][kind]["brand_G"] == "holds" for v in VARIANTS for kind in INF)
    d_tw = {v: conc[v]["two-way"]["brand_diff"] for v in VARIANTS}
    d_x = {v: conc[v]["crossed"]["brand_diff"] for v in VARIANTS}
    dgo, dgn = Ro["dG_minus_dF_all"], Rr["dG_minus_dF_all"]
    diff_tw_all = all(x == "holds" for x in d_tw.values())
    diff_x_not = [v for v in VARIANTS if d_x[v] != "holds"]
    ztw = [CT[v]["R"]["dG_minus_dF_all"]["tz"] for v in VARIANTS]
    zr = [CT[v]["R"]["dG_minus_dF_all"]["tz_r"] for v in VARIANTS]
    if diff_tw_all:
        diff_txt = (
            f"G's loading also grows faster than F's: dG - dF is {fm(dgn['est'])}/yr on V4.14.1, two-way CI "
            f"{tciz(dgn)}, and {fm(dgo['est'])} on V4.13.0, {tciz(dgo)}. It holds under the two-way interval in "
            f"every variant (z {min(ztw):.2f} to {max(ztw):.2f}; with IQR-based variances, which discount the heavy "
            f"tails of the b_F and b_G replicates, z {np.nanmin(zr):.2f} to {np.nanmax(zr):.2f}). "
            + (f"Only the conservative crossed bootstrap reaches 0 (V4.14.1 {xci(dgn)}, {dgn['xle0']:.1%} of "
               f"replicates <= 0; V4.13.0 {xci(dgo)}; status by variant in Table 1). "
               if diff_x_not else "The crossed bootstrap agrees. ")
            + "Revision 2 of this note took the crossed interval as the standard, concluded that the brand-over-field "
            "difference was clear only conditional on the observed fields, and suggested weaker README/ROADMAP "
            f"wording. That conclusion rested on an interval whose variance is {cal_lo('crossed'):.1f} to "
            f"{cal_hi('crossed'):.1f} times the true sampling variance under a within-cell null (section 6); it and "
            "the suggested wording are withdrawn. The Phase 1 statement stands as written.")
    else:
        diff_txt = ("The difference dG - dF has these statuses (two-way / crossed): "
                    + "; ".join(f"{REL[v]['label']} {d_tw[v]} / {d_x[v]}" for v in VARIANTS) + ".")
    L.append(
        ("**Brand vs field loadings: the Phase 1 statement stands.** " if diff_tw_all and bG_all_hold
         else "**Brand vs field loadings.** ")
        + f"The academia-wide rank's loading grows over careers (dG {fm(Rr['dbG_all']['est'])}/yr on V4.14.1, "
        f"two-way CI {tci(Rr['dbG_all'])}) and field prestige's does not (dF {fm(Rr['dbF_all']['est'])}, two-way CI "
        f"{tci(Rr['dbF_all'])})"
        + ("; both hold under all four kinds of interval in every variant. " if bG_all_hold else
           "; statuses by variant are in Table 1. ")
        + diff_txt + "\n")
    L.append(
        "**Weak test.** "
        "This is a weak test of the career-time result, because the new release barely changes the "
        f"fixed-cohort panel. {n_late} of the {len(new_ids)} added institutions are in states whose PSEO "
        "series start after 2010 (version_pseo.txt): "
        + join_and([f"{st} ({int(added_states[st])} institution{'s' if added_states[st] != 1 else ''}; "
                    f"from {yrs[st][:4]})" for st in late_states])
        + ". These institutions "
        f"(including {len(fixed0)} Wapman-matched ones) have {n_late_fixed} released cells in the "
        "2001-2010 cohorts. "
        + (f"Tennessee's series starts in {yrs['TN'][:4]}, so its {int(added_states.get('TN', 0))} added "
           "institutions can enter only the 2010 cohort. " if "TN" in added_states.index and yrs.get("TN", "")[:4] == "2010" else "")
        + f"The balanced panel grows from {CT['old']['info']['panel_insts']} to "
        f"{CT['new']['info']['panel_insts']} institutions ({CT['new_alias']['info']['panel_insts']} with aliases)"
        + (". The panel built from V4.14.1 restricted to V4.13.0 institutions is identical to the V4.13.0 "
           "panel, so no revision touches a fixed-cohort panel cell" if same_panel else
           ". Revisions change some V4.13.0 panel cells (see section 3)")
        + (". Revisions do reach the late-cohort (2013-2019) y1 cells that the calendar contrast uses: on "
           "V4.13.0 institutions only, the calendar-matched contrast moves from "
           f"{fm(Ro['D_cal_all']['est'], 4)} to {fm(Rn['D_cal_all']['est'], 4)}/yr and the triple-matched "
           f"within-cohort change from {fm(Ro['D_within_all']['est'], 4)} to {fm(Rn['D_within_all']['est'], 4)}/yr.\n"
           if (Ro["D_cal_all"]["est"] != Rn["D_cal_all"]["est"] or Ro["D_within_all"]["est"] != Rn["D_within_all"]["est"])
           else ".\n"))
    Qn = QT["new"]
    q = {h: Sn[("F_p75_minus_p50", h, "")] for h in HZ + [AVG]}
    q25 = {h: Sn[("F_p25_minus_p50", h, "")] for h in HZ + [AVG]}
    Kq = Qn["K"]
    nf = {h: Sn[("fields_p75_gt_p50", h, "")]["est"] for h in HZ}
    perm_sig = [h for h in HZ if q[h]["p"] < 0.05]
    perm_ns = [h for h in HZ if q[h]["p"] >= 0.05]
    xst = {h: x_status(q[h]["xlo"], q[h]["xhi"], q[h]["xmcse"]) for h in HZ + [AVG]}
    x_above = [h for h in HZ if xst[h] == "above 0"]
    tst = {h: x_status(q[h]["tlo"], q[h]["thi"], q[h]["tmcse"]) for h in HZ + [AVG]}
    t_above = [h for h in HZ if tst[h] == "above 0"]
    t_touch = [h for h in HZ if tst[h] == "touches 0"]
    t_incl = [h for h in HZ if tst[h] == "includes 0"]
    sens = Qn["sens"]
    specs = list(dict.fromkeys(k[0] for k in sens))
    alt = specs[1:]
    avg_alt = [sens[(s_, "p75-p50", AVG)]["est"] for s_ in alt]
    x_above_specs = [s_ for s_ in specs if sens[(s_, "p75-p50", AVG)]["xlo"] > 0]
    t_above_specs = [s_ for s_ in specs if x_status(sens[(s_, "p75-p50", AVG)]["tlo"], sens[(s_, "p75-p50", AVG)]["thi"],
                                                    sens[(s_, "p75-p50", AVG)]["tmcse"]) == "above 0"]
    if tst[AVG] == "above 0" and len(t_above) == len(HZ):
        verdict = f"Yes, slightly: by about {q[AVG]['est']:+.2f}, at every horizon."
    elif tst[AVG] == "above 0":
        verdict = (f"Slightly: by about {q[AVG]['est']:+.2f} averaged over horizons (two-way CI {tci(q[AVG])}), "
                   + (f"clearly at {join_and(t_above)}" if t_above else "but not clearly at any single horizon")
                   + (f", not at {join_and(t_incl + t_touch)}." if (t_incl or t_touch) and t_above else "."))
    elif tst[AVG] == "touches 0" or t_above:
        verdict = (f"Marginally: by about {q[AVG]['est']:+.2f} averaged over horizons; the two-way CI "
                   f"{tci(q[AVG])} reaches 0 within Monte Carlo error.")
    elif all(q[h]["est"] > 0 for h in HZ):
        verdict = "No clear difference: the point estimates are small and positive, and the two-way CIs include 0."
    else:
        verdict = "No."
    q25_x_below = [h for h in HZ if q25[h]["xhi"] < 0]
    q25_p_min = [h for h in HZ if q25[h]["p"] <= 1 / (NPERM + 1) + 1e-12]

    def perm_list(hs):
        return join_and([f"{h} ({pq(q[h]['p'])})" for h in hs])

    L.append(
        f"**Does p75 coupling exceed median coupling? {verdict}** On V4.14.1 fixed cohorts ({Kq} fields, "
        f"{Qn['n_cells']} field x cohort cells, {Qn['n_inst']} institutions), p75 coupling is above p50 coupling "
        "by " + ", ".join(f"{fm(q[h]['est'])} at {h}" for h in HZ)
        + "; p75 > p50 in " + ", ".join(f"{nf[h]}/{Kq}" for h in HZ) + " fields (y1, y5, y10). "
        f"Averaged over the three horizons (a post hoc summary; the spec asked for horizons) it is "
        f"{fm(q[AVG]['est'])}. The size is stable across aggregations: the averaged difference is "
        f"{fm(min(avg_alt))} to {fm(max(avg_alt))} under Fisher-z averaging, n-weighted cells, cells with "
        f"n>={NMIN_BIG} only and CIP-4 cells without within-field aggregation (Table 3). Whether it differs from 0 "
        f"depends on the inference model, because the same institutions recur across the 4 cohorts and up to "
        f"{Qn['max_fields_inst']} fields (median {Qn['median_cells_inst']:.0f} cells per institution). "
        "Under the two-way field x institution cluster variance, which treats fields and institutions as sampled, "
        "the 95% CI is " + ", ".join(f"{h} {tciz(q[h])}" for h in HZ) + f"; average {tciz(q[AVG])}. "
        + (f"Of the {len(alt)} alternative aggregations, "
           + (f"{len([x for x in t_above_specs if x != specs[0]])} give a two-way CI above 0 for the average: "
              + "; ".join(f"'{s_}' {fm(sens[(s_, 'p75-p50', AVG)]['est'])} {tciz(sens[(s_, 'p75-p50', AVG)])}"
                          for s_ in t_above_specs if s_ != specs[0])
              if [x for x in t_above_specs if x != specs[0]] else "none gives a two-way CI above 0 for the average")
           + " (Table 3; the n>=30 and CIP-4 variants were proposed after the first results were seen). ")
        + f"Conditional on these {Kq} fields, an institution-clustered quantile-label permutation gives "
        + (f"p<0.05 at {perm_list(perm_sig)}" if perm_sig else "no p<0.05 at any horizon")
        + (f", not at {perm_list(perm_ns)}" if perm_ns and perm_sig else f" ({perm_list(perm_ns)})" if perm_ns else "")
        + f", and {pq(q[AVG]['p'])} for the average; the fields-fixed institution-cluster bootstrap gives "
        f"{cci(q[AVG])} for the average. The conservative crossed bootstrap "
        + ("includes 0 at every horizon" if not x_above else f"is above 0 at {join_and(x_above)} only")
        + ": " + ", ".join(f"{h} {xci(q[h])}" for h in HZ) + f"; average {xci(q[AVG])}"
        + (f" (above 0 only for '{join_and(x_above_specs)}' among the aggregations)" if x_above_specs else "")
        + ". Revision 2 read the crossed result as 'clear only conditional on the observed fields'; "
        + ("with the calibrated two-way interval that qualifier no longer applies to the average (section 6). "
           if tst[AVG] == "above 0" else
           f"the calibrated two-way interval puts the average at the edge of 0 (z={q[AVG]['tz']:.2f}), so the "
           "difference stays marginal once fields are treated as sampled (section 6). " if tst[AVG] == "touches 0" else
           "the calibrated two-way interval does not clear 0 either, so that reading stands (section 6). ")
        + "Revision 1 of this note said p<0.001; that came from a permutation that flipped labels "
        "independently in every cell and ignored the recurring institutions (see Revisions). "
        "The robust quantile pattern is at the bottom: p25 coupling is below p50 coupling by "
        + ", ".join(f"{fm(q25[h]['est'])} at {h}" for h in HZ)
        + f" ({fm(q25[AVG]['est'])} averaged; two-way CI {tciz(q25[AVG])}; crossed CI {xci(q25[AVG])}), with the "
        "crossed CI below 0 at "
        + (join_and(q25_x_below) if q25_x_below else "no horizon")
        + " and the clustered permutation "
        + (f"at its minimum attainable p (0.001 with {NPERM} permutations) at {join_and(q25_p_min)}"
           if q25_p_min else "p=" + "/".join(fpq(q25[h]['p']) for h in HZ))
        + ". Coupling therefore rises across quantiles mainly because p25 is less ordered by prestige than the "
        "median, not because p75 is more ordered. PSEO quartiles cannot see the elite tail (top 1-10%, where "
        "Chetty-Deming-Friedman find brand effects), so this tests the within-program upper quartile only.\n")

    # ------------------------------------------------------------------ table 1: conclusions
    L.append("### Table 1 — PSEO-based statements, by data variant and kind of interval\n")
    L.append(
        "Each cell gives the status under four intervals, in this order: the two-way field x institution cluster "
        "variance (the standard of this note; normal CI), the crossed field x institution bootstrap (conservative, "
        "section 6), the institution-cluster bootstrap with fields held fixed, and scripts/52's two-stage bootstrap "
        "(for the cross-source row, scripts/32's bootstrap over fields). The first two treat fields and "
        "institutions as sampled; the last two are conditional on the observed fields, and the two-stage one also "
        "treats each cell's institutions as a fresh sample. 'holds': the CI lies on the stated side of 0 by more "
        "than 2 Monte Carlo SE of its endpoint (for 'spans 0', both endpoints are more than 2 MC SE from 0). "
        f"'edge': the deciding endpoint is within 2 MC SE of 0, so with {NBOOT} replicates it is not determined "
        "whether the 95% CI excludes 0. 'does not hold': otherwise. A statement counts as changed between variants "
        "only if it holds in one and does not hold in another under the same kind of interval. Cells that are not "
        "'holds' under all four are in bold. Revision 2 of this table had no two-way column and took the crossed "
        "interval as the standard; revision 1 used the two-stage interval only.\n")
    L.append("| statement | " + " | ".join(REL[v]["label"] for v in VARIANTS) + " |")
    L.append("|---|" + "---|" * len(VARIANTS))
    for k, lab in CONC_LAB.items():
        cells = []
        for v in VARIANTS:
            t = " / ".join(conc[v][kind][k] for kind in INF)
            cells.append(t if all(conc[v][kind][k] == "holds" for kind in INF) else f"**{t}**")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("")

    # ------------------------------------------------------------------ table 2: key numbers
    L.append("### Table 2 — key numbers, V4.13.0 vs V4.14.1\n")
    L.append("Cells: estimate; two-way 95% CI (z); crossed 95% CI; fields-fixed 95% CI; two-stage 95% CI; p where "
             "given (k = fields). Two-way: estimate +/- 1.96 SE with SE^2 = V_field + V_institution - V_field x "
             "institution (Cameron-Gelbach-Miller), from the between-field variance and two institution draws "
             f"with fields fixed ({NBOOT} replicates each; one draw shared by all fields, one independent per field). "
             f"Crossed: fields resampled x one institution multinomial draw shared by every cell ({NBOOT}). Fields "
             f"fixed: the shared institution draw with fields held fixed ({NBOOT}). Two-stage: fields resampled, then "
             f"institutions redrawn separately in each cell (spec A: scripts/52's own, {s52.NBOOT} replicates; spec B: "
             f"{NBOOT}). p: institution-clustered quantile-label permutation ({NPERM}). For the cross-source row the "
             "two-stage slot holds scripts/32's bootstrap over fields (4000). The other data variants are in sections 3-5. "
             "Spec A: PSEO bachelor's fixed cohorts 2001/2004/2007/2010, institutions with released earnings at y1, "
             "y5 and y10 (balanced), n>=15 per field x cohort; per-field OLS slope of coupling on years since "
             "graduation, mean over fields. Spec B: the same panel; Spearman(field prestige, earnings quantile) per "
             "field x cohort, mean over cohorts then fields. The mean over horizons is a post hoc summary.\n")
    L.append("| quantity | sample / spec | V4.13.0 (2025Q4, repo) | V4.14.1 (2026Q2) |")
    L.append("|---|---|---|---|")
    fx, qx = "A", "B"

    def c_ct(v, key):
        r = CT[v]["R"][key]
        return f"{fm(r['est'])}; {tciz(r)}; {xci(r)}; {cci(r)}; {cis(r)} (k={r['k']})"

    def c_q(v, stat, h, q=""):
        return qdiff(QT[v]["S"][(stat, h, q)]) + f" (k={QT[v]['K']})"

    t2 = [("career-time slope, all fields (per yr)", fx, lambda v: c_ct(v, "slope_all")),
          ("career-time slope, integrated fields (per yr)", fx, lambda v: c_ct(v, "slope_integrated")),
          ("integrated minus other fields (per yr)", fx, lambda v: c_ct(v, "slope_int_minus_others")),
          ("career-time bound [calendar-matched, within-cohort] (per yr)",
           "triple-matched institutions; bound valid if calendar and cohort effects are both >= 0",
           lambda v: f"[{fm(CT[v]['R']['D_cal_all']['est'])}, {fm(CT[v]['R']['D_within_all']['est'])}]"),
          ("calendar-matched contrast, lower end of the bound (per yr)", "A; triple-matched institutions",
           lambda v: c_ct(v, "D_cal_all")),
          ("dG: slope of academia-wide-rank coefficient (per yr)", "A; rank regression on F and G per cell",
           lambda v: c_ct(v, "dbG_all")),
          ("dF: slope of field-prestige coefficient (per yr)", "A; rank regression on F and G per cell",
           lambda v: c_ct(v, "dbF_all")),
          ("dG - dF (per yr)", "A; rank regression on F and G per cell", lambda v: c_ct(v, "dG_minus_dF_all")),
          ("Spearman(field coupling PSEO, field coupling Scorecard)",
           "PSEO y5 p50 pooled cohorts vs Scorecard EARN_MDN_4YR; fields with n>=8; crossed CI biased downward "
           "(section 5)",
           lambda v: f"{CS[v]['rho']:+.2f}; [{CS[v]['tlo']:+.2f}, {CS[v]['thi']:+.2f}] (z={CS[v]['tz']:.1f}); "
                     f"[{CS[v]['xlo']:+.2f}, {CS[v]['xhi']:+.2f}]; [{CS[v]['clo']:+.2f}, {CS[v]['chi']:+.2f}]; "
                     f"[{CS[v]['lo']:+.2f}, {CS[v]['hi']:+.2f}] (k={CS[v]['k']})")]
    for h in HZ:
        t2.append((f"coupling p25 / p50 / p75 at {h}", qx,
                   lambda v, h=h: " / ".join(fm(QT[v]["S"][("coupling_F", h, q)]["est"]) for q in QS)
                   + f" (k={QT[v]['K']})"))
    t2.insert(8, ("b_F at y1: field-prestige coefficient (scripts/52 item 5)", "A; rank regression on F and G per cell",
                  lambda v: c_ct(v, "bF_y1_all")))
    for h in HZ + [AVG]:
        t2.append((f"p75 minus p50 coupling, {h}" + (" (post hoc)" if h == AVG else ""), qx,
                   lambda v, h=h: c_q(v, "F_p75_minus_p50", h)))
    t2.append((f"p25 minus p50 coupling, {AVG} (post hoc)", qx, lambda v: c_q(v, "F_p25_minus_p50", AVG)))
    t2.append(("career-time slope of p75 minus that of p50 (per yr)", qx,
               lambda v: c_q(v, "slope_F_p75_minus_p50", "per year")))
    for lab, spec, fn in t2:
        L.append(f"| {lab} | {spec} | {fn('old')} | {fn('new')} |")
    L.append("")

    # ------------------------------------------------------------------ release facts
    L.append("## 1. The release\n")
    L.append("| | V4.13.0 (repo, `data/raw/pseo/`) | V4.14.1 (`data/raw/pseo_2026q2/`) |")
    L.append("|---|---|---|")
    L.append(f"| version string(s) in version_pseo.txt | {po['version']} (file lists the US pseudo-state only) | {pn['version']} (all {len(pn['version_states'])} entries) |")
    L.append(f"| earnings rows (pseoe_all.csv.gz) | {po['rows']:,} | {pn['rows']:,} |")
    L.append(f"| institution-level rows | {po['rows_I']:,} | {pn['rows_I']:,} |")
    L.append(f"| distinct institutions with rows | {po['inst_earn']:,} | {pn['inst_earn']:,} |")
    L.append(f"| rows in pseo_all_institutions.csv | {po['inst_file']:,} | {pn['inst_file']:,} |")
    no = [s for s in po["states"] if s != "US"]; nn = [s for s in pn["states"] if s != "US"]
    L.append(f"| partner jurisdictions (states + DC) | {len(no)} | {len(nn)} (added: {', '.join(sorted(set(nn) - set(no)))}) |")
    L.append(f"| md5 pseoe_all.csv.gz | `{po['pseoe_all.csv.gz']['md5']}` | `{pn['pseoe_all.csv.gz']['md5']}` |")
    L.append("")
    L.append(
        f"The new release is a superset of the old one. Among bachelor's institution x CIP-4 rows, "
        f"{dff['shared']:,} are shared, {dff['old_only']} exist only in V4.13.0 and {dff['new_only']:,} only in "
        f"V4.14.1. Of rows released in both, the p50 changed in {dff['y1_p50_changed']} of "
        f"{dff['y1_released_both']:,} at y1, {dff['y5_p50_changed']} of {dff['y5_released_both']:,} at y5 and "
        f"{dff['y10_p50_changed']} of {dff['y10_released_both']:,} at y10; release status changed in "
        f"{dff['y1_status_changed']} / {dff['y5_status_changed']} / {dff['y10_status_changed']} rows "
        f"(y1 / y5 / y10). The local V4.14.1 earnings file is the complete national file: its size equals the "
        f"server's Content-Length ({SERVER_CONTENT_LENGTH:,} bytes; HEAD request {HEAD_CHECK}), `gzip -t` passes, "
        "and the release directory holds no other earnings file (see Provenance).\n")

    # ------------------------------------------------------------------ coverage
    L.append("## 2. Coverage change (question a)\n")
    L.append(
        f"{len(new_ids)} institutions are added. {kn} of them match a Wapman institution through the "
        f"project's canonical name join (`src/crosswalks/institutions.py`); {ka} more are the same "
        "institutions under a different name and are matched only in the alias sensitivity (Rutgers "
        "New Brunswick, Newark, Camden; University of Tennessee Knoxville). 'Usable' cells are released "
        "bachelor's institution x CIP-4 earnings cells whose CIP-4 maps to one of the 66 fields and whose "
        "institution has a Wapman rank in that field.\n")
    L.append("| Wapman academia rank | institution | state | join | Wapman fields | usable cells y1 / y5 / y10 (all cohorts) | of which fixed cohorts 2001-2010 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in newtab.sort_values("g_rank").itertuples():
        L.append(f"| {r.g_rank} | {r.label} | {r.state} | {r.join} | {r.n_wapman_fields} | "
                 f"{r.usable_y1} / {r.usable_y5} / {r.usable_y10} | "
                 f"{r.usable_fixed_y1} / {r.usable_fixed_y5} / {r.usable_fixed_y10} |")
    L.append("")
    L.append("Usable released cells (all Wapman-matched institutions), by horizon and cohort group:\n")
    L.append("| cohorts | horizon | V4.13.0 | V4.14.1 | added by new institutions | V4.14.1 + 4 aliases | added (4 aliases) | V4.14.1 + 23 aliases |")
    L.append("|---|---|---|---|---|---|---|---|")
    for gname in tot.cohorts.drop_duplicates():
        for h in HZ:
            t = tot[(tot.cohorts == gname) & (tot.horizon == h)].set_index("variant")
            L.append(f"| {gname} | {h} | {t.loc['old', 'usable']:,} | {t.loc['new', 'usable']:,} | "
                     f"{int(t.loc['new', 'usable_from_new']):,} | {t.loc['new_alias', 'usable']:,} | "
                     f"{int(t.loc['new_alias', 'usable_from_new']):,} | {t.loc['new_joinfix', 'usable']:,} |")
    L.append("")
    L.append("Wapman institutions (academia-wide rank) with at least one usable released y5 cell, any cohort:\n")
    L.append("| quartile of academia rank | institutions | V4.13.0 | V4.14.1 | V4.14.1 + 4 aliases | V4.14.1 + 23 aliases |")
    L.append("|---|---|---|---|---|---|")
    for q in quart.quartile.drop_duplicates():
        t = quart[quart.quartile == q].set_index("variant")
        tt = int(t.loc["old", "total"])
        L.append(f"| {q} | {tt} | " + " | ".join(f"{int(t.loc[v, 'covered'])} ({t.loc[v, 'covered'] / tt:.0%})"
                                             for v in ("old", "new", "new_alias", "new_joinfix")) + " |")
    L.append("")

    def rc(v, q):
        x = rcov[(rcov.variant == v) & (rcov.quartile == q)].iloc[0]
        return f"{x.covered}/{x.total} ({x.covered / x.total:.1%})"
    L.append(
        "The README section 1 coverage figures (PSEO ~29% of Wapman institutions, 21.5% of the top quartile) "
        "use the scripts/01 definition: an institution counts if it has any released y5 pooled-cohort "
        "bachelor's cell in the 30 original (TIER0) fields, whether or not it has a Wapman rank in that "
        "field; quartiles by `pd.qcut` of the Wapman academia rank. Re-run on each release (all / top "
        "quartile): "
        + "; ".join(f"{REL[v]['label']} {rc(v, 'all')} / {rc(v, 'Q1 (top)')}"
                    for v in ("old", "new", "new_alias", "new_joinfix")) + ".\n")

    # ------------------------------------------------------------------ career time
    L.append("## 3. Fixed-cohort career-time analysis (question b)\n")
    L.append(
        f"Reproduction check: on V4.13.0 this script's all-field slope is {fm(Ro['slope_all']['est'], 4)} "
        f"{cis(Ro['slope_all'], 4)}, integrated {fm(Ro['slope_integrated']['est'], 4)}, calendar-matched "
        f"{fm(Ro['D_cal_all']['est'], 4)}, dG {fm(Ro['dbG_all']['est'], 4)}. These equal the values in "
        "`CAREER_TIME_COUPLING_RESULT.md` (scripts/52) to the three decimals reported there, because the "
        "same functions and random streams are used. "
        f"Panel sizes: V4.13.0 {CT['old']['info']['panel_insts']} institutions / {CT['old']['info']['cells']} "
        f"field x cohort cells / {CT['old']['info']['k']} fields; V4.14.1 {CT['new']['info']['panel_insts']} / "
        f"{CT['new']['info']['cells']} / {CT['new']['info']['k']}; with aliases {CT['new_alias']['info']['panel_insts']} / "
        f"{CT['new_alias']['info']['cells']} / {CT['new_alias']['info']['k']}; with all 23 aliases "
        f"{CT['new_joinfix']['info']['panel_insts']} / {CT['new_joinfix']['info']['cells']} / "
        f"{CT['new_joinfix']['info']['k']}. Institutions new to the panel: "
        f"{', '.join(npi) if npi else 'none'} (canonical); alias adds {', '.join(sorted(set(npia) - set(npi))) or 'none'}. "
        f"Per-field slopes, V4.13.0 vs V4.14.1 on the {len(common)} common fields: Spearman {fs_rho:+.3f}, mean "
        f"absolute difference {fs_mad:.4f}/yr. "
        + field_change_text(CT) + "\n")
    L.append("Slope = change in coupling per year since graduation; CI = 95% two-stage bootstrap (fields, then "
             f"institutions within field x cohort), {s52.NBOOT} replicates; p for slopes = horizon-label permutation "
             f"({s52.NPERM}); p for the contrast = field-label permutation ({s52.NPERM_FIELD}). These are scripts/52's "
             "own settings, kept so that the V4.13.0 column reproduces scripts/52 exactly. The variants share most "
             "institutions, so their CIs are not independent and differences between columns are not tested. The "
             "two-stage CI redraws institutions separately in each cell although the same institutions recur across "
             "cohorts and fields; the two-way intervals in the second table below are the standard used for Table 1.\n")
    L.append("| quantity | " + " | ".join(REL[v]["label"] for v in VARIANTS) + " |")
    L.append("|---|" + "---|" * len(VARIANTS))
    items = [("slope_all", "within-cohort slope, all fields"),
             ("slope_all_reffields", "same, restricted to the V4.13.0 fields"),
             ("slope_integrated", "within-cohort slope, integrated"),
             ("slope_int_minus_others", "integrated minus others"),
             ("early_slope_all", "segment y1->y5, all"),
             ("late_slope_all", "segment y5->y10, all"),
             ("D_within_all", "within-cohort (y10-y1)/9, triple-matched, all"),
             ("D_cal_all", "calendar-matched (c@y10 - c+9@y1)/9, all"),
             ("D_cross_all", "cross-cohort drift at y1 (c+9 - c)/9, all"),
             ("D_cal_integrated", "calendar-matched, integrated"),
             ("cov_share_removed_all", "share of slope removed by coverage control"),
             ("dbF_all", "dF: slope of field-prestige coefficient, all"),
             ("dbG_all", "dG: slope of academia-wide-rank coefficient, all"),
             ("dG_minus_dF_all", "dG - dF, all"),
             ("dbF_integrated", "dF, integrated"),
             ("dbG_integrated", "dG, integrated"),
             ("bF_y1_all", "b_F at y1"), ("bF_y10_all", "b_F at y10"),
             ("bG_y1_all", "b_G at y1"), ("bG_y10_all", "b_G at y10")]
    for key, lab in items:
        cells = []
        for v in VARIANTS:
            r = CT[v]["R"].get(key)
            if r is None:
                cells.append("—")
                continue
            p = "" if not np.isfinite(r["p"]) else f"; {pe(r['p'])}"
            cells.append(f"{fm(r['est'])} {cis(r)} (k={r['k']}{p})")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("")
    L.append(
        "Bound on the career-time component (if calendar and cohort effects are both non-negative): "
        + "; ".join(f"{REL[v]['label']} [{fm(CT[v]['R']['D_cal_all']['est'])}, {fm(CT[v]['R']['D_within_all']['est'])}]/yr"
                    for v in VARIANTS) + ".\n")
    # institution-clustered intervals for the same statistics
    cl = {v: CT[v]["info"]["clustered"] for v in VARIANTS}
    L.append("**Institution-clustered intervals for the section 3 statistics.** Each cell: two-way CI / crossed CI / "
             f"fields-fixed CI (95%; {NBOOT} replicates per draw). Two-way: estimate +/- 1.96 SE, SE^2 = V_field + "
             "V_institution - V_field x institution. V_field is the between-field variance of the per-field values "
             "divided by the number of fields (for the contrast, the sum over the two groups; for the share removed "
             "by the coverage control, the jackknife over fields). V_institution is the variance of the fields-fixed "
             "replicates under one institution multinomial draw per replicate, shared by every fixed-cohort panel "
             "cell, coverage cell and triple-matched calendar cell, so the bound's two ends are resampled jointly. "
             "V_field x institution is the same with an independent institution draw for each field (shared by that "
             "field's cohorts and cell types). Crossed: fields resampled with replacement (integrated and other "
             "fields separately for the contrast) x the shared institution draw. Fields fixed: the shared draw, "
             "fields held fixed. Estimates are those of the table above. For the two slopes, p is an "
             "institution-clustered horizon-label permutation (one of the 6 horizon orders per institution, shared "
             f"by all its cells; {NPERM}; minimum attainable p {1 / (NPERM + 1):.3f}). Institutions in the draw: "
             + "; ".join(f"{REL[v]['label']} {cl[v]['n_inst']}" for v in VARIANTS) + ".\n")
    L.append("| quantity | " + " | ".join(REL[v]["label"] for v in VARIANTS) + " |")
    L.append("|---|" + "---|" * len(VARIANTS))
    i_ = [k_ for k_, _ in items].index("dbG_integrated") + 1
    items_x = items[:i_] + [("dG_minus_dF_integrated", "dG - dF, integrated")] + items[i_:]
    for key, lab in items_x:
        cells = []
        for v in VARIANTS:
            r = CT[v]["R"].get(key)
            if r is None:
                cells.append("—")
                continue
            p = f"; {pq(r['p_clu'])}" if "p_clu" in r else ""
            cells.append(f"{tci(r)} / {xci(r)} / {cci(r)}{p}")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("")
    rx = CT["new"]["R"]
    above_t = [lab for key, lab in items_x if key in rx and "tlo" in rx[key]
               and st_pos(*ci3(rx[key], "two-way")) == "holds"]
    above_x = [lab for key, lab in items_x if key in rx and "xlo" in rx[key]
               and st_pos(*ci3(rx[key], "crossed")) == "holds"]
    only_t = [lab for lab in above_t if lab not in above_x]
    bF_st = {(v, h): {kind: st_pos(*ci3(CT[v]["R"][f"bF_{h}_all"], kind)) for kind in INF}
             for v in VARIANTS for h in ("y1", "y10")}
    bF_t_all = all(bF_st[(v, h)]["two-way"] == "holds" for v in VARIANTS for h in ("y1", "y10"))
    bF_x_bad = [v for v in VARIANTS if bF_st[(v, "y1")]["crossed"] != "holds"]
    bz = [CT[v]["R"]["bF_y1_all"]["tz"] for v in VARIANTS]
    bzr = [CT[v]["R"]["bF_y1_all"]["tz_r"] for v in VARIANTS]
    bF_txt = (
        f"b_F, the field-prestige coefficient itself, is above 0 at y1 and y10 under the two-way interval in every "
        f"variant (y1: V4.13.0 {fm(CT['old']['R']['bF_y1_all']['est'])} {tciz(CT['old']['R']['bF_y1_all'])}, V4.14.1 "
        f"{fm(rx['bF_y1_all']['est'])} {tciz(rx['bF_y1_all'])}; z at y1 {min(bz):.2f} to {max(bz):.2f} across "
        f"variants, {np.nanmin(bzr):.2f} to {np.nanmax(bzr):.2f} with IQR-based variances). "
        if bF_t_all else
        "b_F at y1 / y10 under the two-way interval: "
        + "; ".join(f"{REL[v]['label']} {bF_st[(v, 'y1')]['two-way']} / {bF_st[(v, 'y10')]['two-way']}" for v in VARIANTS)
        + ". ")
    if bF_x_bad:
        bF_txt += (
            "Under the crossed bootstrap its lower end at y1 is near 0 ("
            + "; ".join(f"{REL[v]['label']} {bF_st[(v, 'y1')]['crossed']}" for v in VARIANTS)
            + f"; V4.13.0 {xci(CT['old']['R']['bF_y1_all'])}, V4.14.1 {xci(rx['bF_y1_all'])}). The b_F replicates "
            "are heavy-tailed (r_FG is high, so 1 - r_FG^2 in the denominator is small), which makes percentile "
            "endpoints unstable; the crossed replicates' SD is "
            f"{rx['bF_y1_all']['txse']:.3f} against a two-way SE of {rx['bF_y1_all']['tse']:.3f} on V4.14.1. ")
    bF_txt += ("scripts/52's result note (CAREER_TIME_COUPLING_RESULT.md, item 5) argues from b_F > 0 at y1 that a "
               "component tracked better by F is needed. "
               + ("That premise holds under the calibrated two-way interval in every variant and is marginal only "
                  "under the conservative crossed bootstrap. Revision 2 flagged the premise as not clear; that flag is "
                  "withdrawn. " if bF_t_all else "Its status under the two-way interval is as listed above. "))
    xcent = max(abs(r["xmed"] - r["est"]) / (r["xhi"] - r["xlo"]) for r in rx.values() if "xmed" in r)
    fallbacks = [f"{REL[v]['label']}: {k_}" for v in VARIANTS for k_, r in CT[v]["R"].items()
                 if r.get("tfallback")]
    L.append(
        "Reading the table (V4.14.1). The two-way intervals are narrower than the crossed ones (median width ratio "
        f"{cl['new']['width_ratio_tx']:.2f} over the listed statistics); the crossed intervals are "
        f"{cl['new']['width_ratio_x']:.2f} times as wide as the two-stage ones (median), the fields-fixed ones "
        f"{cl['new']['width_ratio_c']:.2f} times. Two-way CIs above 0 by more than 2 Monte Carlo SE: "
        + "; ".join(above_t) + ". "
        + (f"Of these, only under the two-way interval (the crossed CI reaches 0): {'; '.join(only_t)}. " if only_t else
           "The crossed CIs are above 0 for the same statistics. ")
        + f"dF {tci(rx['dbF_all'])} and integrated minus others {tci(rx['slope_int_minus_others'])} span 0. "
        + f"dG - dF: all fields {tciz(rx['dG_minus_dF_all'])}; integrated fields {tciz(rx['dG_minus_dF_integrated'])}. "
        + bF_txt
        + (f"In {len(fallbacks)} cases the two-way variance was not positive and the larger one-way variance was used "
           f"({'; '.join(fallbacks)}). " if fallbacks else "The two-way variance was positive for every statistic and variant. ")
        + f"The crossed replicates are centred on the estimates (largest |median - estimate| over the listed "
          f"statistics: {xcent:.0%} of the CI width), unlike the section 5 correlation. "
        + "Replicates redrawn because some cell's statistic was undefined (shared draw): "
        + ", ".join(f"{REL[v]['label']} {cl[v]['redrawn']} of {NBOOT}" for v in ("old", "new"))
        + "; independent draw per field: "
        + ", ".join(f"{REL[v]['label']} {cl[v]['redrawn_i']} of {NBOOT * cl[v]['n_fields_i']} field x replicate "
                    "draws" for v in ("old", "new"))
        + ". In the first shared draw these were due to "
        + join_and(sorted({DEGEN.get(k_, k_) for v in ("old", "new")
                           for k_, n_ in cl[v]["first"].items() if k_ != "any" and n_ > 0}))
        + " (b_F, b_G undefined when a resampled cell's F and G ranks are perfectly concordant; the coverage "
        "partial when a resampled cell's coverage ranks are perfectly concordant with prestige or earnings)"
        + ("; coupling itself and the calendar contrasts were defined in every replicate of the first draw.\n"
           if all(cl[v]["first"].get(k_, 0) == 0 for v in ("old", "new") for k_ in ("fix:rhoF", "cal:rho", "cov:raw"))
           else ".\n"))

    # ------------------------------------------------------------------ quantiles
    L.append("## 4. Quantile coupling at fixed cohorts (question c)\n")
    L.append(
        "Coupling = Spearman(field prestige F, earnings quantile) across institutions within field x cohort, "
        "averaged over cohorts and then over fields. Balanced panel: institutions with released p25, p50 "
        "and p75 at y1, y5 and y10 (PSEO releases all three quantiles together, so this is the same panel "
        "as section 3's p50 panel). The same institution enters up to 4 cohorts of a field and many fields, so "
        "the cells are not independent. Quantile differences are shown as: estimate; two-way CI (z); crossed CI; "
        "fields-fixed CI; two-stage CI; p. The two-way CI (estimate +/- 1.96 SE, SE^2 = V_field + V_institution - "
        "V_field x institution; the standard of this note) uses the between-field variance, the fields-fixed "
        "replicates of one institution multinomial draw applied to every cell, and those of an independent "
        f"institution draw per field ({NBOOT} replicates each). The crossed CI resamples fields and applies the shared "
        f"institution draw to every cell ({NBOOT} replicates; Owen 2007); it is conservative for these statistics "
        "(section 6). The fields-fixed CI uses the shared institution draw with fields held fixed. The two-stage CI "
        "(fields resampled, then institutions redrawn separately in each cell; paired across quantiles and "
        f"horizons; {NBOOT} replicates) is the earlier interval and ignores the recurring institutions. "
        "p is from an institution-clustered "
        "quantile-label permutation: one flip per institution, shared by all its field x cohort cells and "
        f"horizons ({NPERM} permutations; conditional on the observed fields; minimum attainable p 0.001). "
        "Coupling levels are shown with the two-stage CI.\n")
    for v, S in (("new", Sn), ("old", So), ("new_alias", Sa), ("new_joinfix", QT["new_joinfix"]["S"])):
        Qv = QT[v]
        L.append(f"**{REL[v]['label']}** (k={Qv['K']} fields, {Qv['n_cells']} field x cohort cells, "
                 f"{Qv['n_inst']} institutions)\n")
        L.append("| horizon | coupling p25 | coupling p50 | coupling p75 | p75 - p50: est; two-way (z); crossed; "
                 "fields fixed; two-stage; p | p25 - p50: est; two-way (z); crossed; fields fixed; two-stage; p | "
                 "fields p75 > p50 | Spearman(F, p75/p50) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for h in HZ:
            c = [S[("coupling_F", h, q_)] for q_ in QS]
            L.append(f"| {h} | " + " | ".join(f"{fm(x['est'])} {cis(x)}" for x in c)
                     + f" | {qdiff(S[('F_p75_minus_p50', h, '')])} | {qdiff(S[('F_p25_minus_p50', h, '')])} | "
                     f"{S[('fields_p75_gt_p50', h, '')]['est']}/{Qv['K']} | "
                     f"{fm(S[('F_vs_p75_over_p50', h, '')]['est'])} {cis(S[('F_vs_p75_over_p50', h, '')])} |")
        L.append(f"| mean of horizons (post hoc) | | | | {qdiff(S[('F_p75_minus_p50', AVG, '')])} | "
                 f"{qdiff(S[('F_p25_minus_p50', AVG, '')])} | | |")
        L.append("")
        sl = [S[("slope_F", "per year", q_)] for q_ in QS]
        sd = S[("slope_F_p75_minus_p50", "per year", "")]
        L.append("Career-time slope by quantile (per year; two-stage CI; crossed CI): "
                 + ", ".join(f"{q_} {fm(x['est'])} {cis(x)} {xci(x)}" for q_, x in zip(QS, sl))
                 + f"; p75 minus p50 {fm(sd['est'])}, two-way {tci(sd)}, crossed {xci(sd)}, fields fixed "
                 f"{cci(sd)}, two-stage {cis(sd)}.\n")
        if v == "new":
            nsd = {t: S[(f"null_sd_p75-p50", AVG, t)]["est"] for t in ("independent-flip", "institution-clustered")}
            L.append(
                "Reading the V4.14.1 table. The two-way CI for p75 - p50 is "
                + ", ".join(f"{h} {tciz(S[('F_p75_minus_p50', h, '')])}" for h in HZ)
                + f" and {tciz(S[('F_p75_minus_p50', AVG, '')])} for the average; the two-way SE of the average "
                f"({S[('F_p75_minus_p50', AVG, '')]['tse']:.4f}) lies between the fields-fixed one "
                f"({np.sqrt(S[('F_p75_minus_p50', AVG, '')]['tvs']):.4f}) and the crossed replicates' SD "
                f"({S[('F_p75_minus_p50', AVG, '')]['txse']:.4f}). "
                "With fields held fixed, the institution-cluster bootstrap gives "
                + ", ".join(f"{h} {cci(S[('F_p75_minus_p50', h, '')])}" for h in HZ)
                + f" and {cci(S[('F_p75_minus_p50', AVG, '')])} for the average of p75 - p50. The crossed CI "
                "is wider than the fields-fixed one by about "
                f"{np.mean([(S[('F_p75_minus_p50', h, '')]['xhi'] - S[('F_p75_minus_p50', h, '')]['xlo']) / (S[('F_p75_minus_p50', h, '')]['chi'] - S[('F_p75_minus_p50', h, '')]['clo']) for h in HZ + [AVG]]):.1f} times"
                + ("; all crossed CIs include 0. " if all(S[('F_p75_minus_p50', h, '')]['xlo'] <= 0 for h in HZ + [AVG])
                   else ". ")
                + 
                f"Clustering the permutation by institution multiplies the null SD of the averaged p75 - p50 "
                f"difference by {nsd['institution-clustered'] / nsd['independent-flip']:.1f}, from "
                f"{nsd['independent-flip']:.4f} (independent flips per cell, the earlier test) to "
                f"{nsd['institution-clustered']:.4f}. "
                "Prestige does predict a wider upper half within programs: Spearman(F, p75/p50) is "
                + ", ".join(f"{fm(S[('F_vs_p75_over_p50', h, '')]['est'], 2)} at {h}" for h in HZ)
                + ". But within a field x cohort cell the institutions' p75 and p50 correlate at a mean Spearman "
                "of " + ", ".join(f"{S[('rank_p50_p75', h, '')]['est']:.2f} at {h}" for h in HZ)
                + ", so this stretch moves p75 coupling above p50 coupling only by the small amounts in the "
                f"table. The career-time rise does not differ by quantile (slope of p75 - p50 {fm(sd['est'])}/yr, "
                f"two-way CI {tci(sd)}). " + other_variants_text(QT)
                + (f" {Qv['redrawn']} of {NBOOT} crossed replicates {'was' if Qv['redrawn'] == 1 else 'were'} "
                   "redrawn because a cell's coupling was "
                   "undefined (e.g. only two distinct institutions drawn in a 15-institution cell); scripts/52 "
                   "handles degenerate draws the same way." if Qv["redrawn"] else "") + "\n")
            # table 3: aggregation sensitivity
            sens = Qv["sens"]
            specs = list(dict.fromkeys(k_[0] for k_ in sens))
            ci4 = Qv.get("cip4_info", {})
            L.append("### Table 3 — aggregation sensitivity of the quantile differences (V4.14.1)\n")
            L.append(
                f"Estimate with crossed CI ({NBOOT} replicates); last two columns: the post hoc mean over horizons "
                "with the two-way CI (z) and with the institution-cluster CI (fields fixed). For the n-weighted mean "
                "V_field uses the linearized per-field contributions. Fisher z: cell Spearman transformed by atanh before "
                "averaging; the difference is in z units. n-weighted: mean over all field x cohort cells weighted "
                f"by the number of institutions. n>={NMIN_BIG}: only cells with at least {NMIN_BIG} institutions "
                f"({Qv['n_cells_big']} of {Qv['n_cells']} cells). CIP-4: coupling on institution x CIP-4 x cohort "
                "cells with each CIP-4 code's own p25/p50/p75 (no within-field aggregation), balanced, n>=15 per "
                "cell; mean over cohorts, then CIP-4 codes within field, then fields"
                + (f" ({ci4['cells']} cells, {ci4['cip4']} CIP-4 codes, {ci4['insts']} institutions)" if ci4 else "")
                + ". The n>=30 subset and the CIP-4 grain were proposed after the first results were seen.\n")
            L.append(f"| spec | difference | k fields | y1 | y5 | y10 | mean (post hoc) | mean, two-way | mean, fields fixed |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for s_ in specs:
                for pr in ("p75-p50", "p25-p50"):
                    rr = [sens[(s_, pr, h)] for h in HZ + [AVG]]
                    L.append(f"| {s_} | {pr.replace('-', ' - ')} | {rr[0]['k']} | "
                             + " | ".join(f"{fm(x['est'])} {xci(x)}" for x in rr)
                             + f" | {tciz(rr[-1])} | {cci(rr[-1])} |")
            L.append("")
    L.append("Academia-wide rank G instead of field prestige F (V4.14.1; coupling levels with two-stage CI):\n")
    L.append("| horizon | coupling_G p25 | p50 | p75 | G: p75 - p50: est; two-way (z); crossed; fields fixed; two-stage |")
    L.append("|---|---|---|---|---|")
    for h in HZ:
        c = [Sn[("coupling_G", h, q_)] for q_ in QS]
        dg = Sn[("G_p75_minus_p50", h, "")]
        L.append(f"| {h} | " + " | ".join(f"{fm(x['est'])} {cis(x)}" for x in c) + f" | {qdiff(dg)} |")
    dg = Sn[("G_p75_minus_p50", AVG, "")]
    L.append(f"| mean of horizons (post hoc) | | | | {qdiff(dg)} |")
    L.append("")

    # ------------------------------------------------------------------ cross-source
    L.append("## 5. Cross-source replication (question d)\n")
    L.append("Spec of scripts/32: PSEO y5 p50, pooled cohort 0000, released cells, FIELDS66 map, PSEO x Wapman "
             "institutions; Scorecard EARN_MDN_4YR on all Scorecard x Wapman institutions; field coupling needs "
             "n>=8 and >=3 distinct earnings values; bootstrap CI over fields (4000, seed 3).\n")
    L.append("| | " + " | ".join(REL[v]["label"] for v in VARIANTS) + " |")
    L.append("|---|" + "---|" * len(VARIANTS))
    L.append("| PSEO x Wapman institutions | " + " | ".join(str(CS[v]["n_inst"]) for v in VARIANTS) + " |")
    L.append("| institution x field earnings agreement, Spearman(PSEO, Scorecard) | "
             + " | ".join(f"{CS[v]['lvl_rho']:+.3f} (n={CS[v]['lvl_n']})" for v in VARIANTS) + " |")
    L.append("| Spearman(field coupling PSEO, Scorecard) | "
             + " | ".join(f"{CS[v]['rho']:+.3f} [{CS[v]['lo']:+.3f}, {CS[v]['hi']:+.3f}] (k={CS[v]['k']})" for v in VARIANTS) + " |")
    L.append("| same, two-way field x institution cluster CI (z) | "
             + " | ".join(f"[{CS[v]['tlo']:+.3f}, {CS[v]['thi']:+.3f}] (z={CS[v]['tz']:.1f})" for v in VARIANTS) + " |")
    L.append("| same, crossed field x institution CI | "
             + " | ".join(f"[{CS[v]['xlo']:+.3f}, {CS[v]['xhi']:+.3f}]" for v in VARIANTS) + " |")
    L.append("| same, institution-cluster CI with fields fixed | "
             + " | ".join(f"[{CS[v]['clo']:+.3f}, {CS[v]['chi']:+.3f}]" for v in VARIANTS) + " |")
    L.append("| median of the crossed / fields-fixed / independent-draw replicates | "
             + " | ".join(f"{CS[v]['xmed']:+.3f} / {CS[v]['cmed']:+.3f} / {CS[v]['imed']:+.3f}" for v in VARIANTS) + " |")
    L.append("| same, fields with n_PSEO>=15 | "
             + " | ".join(f"{CS[v]['rho15']:+.3f} [{CS[v]['lo15']:+.3f}, {CS[v]['hi15']:+.3f}] (k={CS[v]['k15']})" for v in VARIANTS) + " |")
    L.append("| mean field coupling PSEO / Scorecard | "
             + " | ".join(f"{CS[v]['mean_c_pseo']:+.3f} / {CS[v]['mean_c_sc']:+.3f}" for v in VARIANTS) + " |")
    L.append("")
    cn = CS["new"]
    L.append(
        "scripts/32's interval resamples fields and keeps each field's two couplings fixed, so it leaves out "
        "institution sampling. The crossed bootstrap adds it: one institution multinomial draw per replicate is "
        f"applied to both sources' field couplings ({cn['xs_n_inst']} institutions on V4.14.1; {NBOOT} replicates; "
        f"{cn['xs_redrawn']} redrawn because a field coupling was undefined). For this statistic the percentile "
        "interval is biased downward. Each replicate adds a second dose of institution sampling noise to field "
        "couplings that already carry one, and noise in both inputs attenuates a correlation across fields. On "
        f"V4.14.1 the fields-fixed replicates have median {cn['cmed']:+.2f} against the observed {cn['rho']:+.2f}, "
        f"and their 95% range [{cn['clo']:+.2f}, {cn['chi']:+.2f}] "
        + ("does not contain the observed value" if not (cn["clo"] <= cn["rho"] <= cn["chi"]) else
           "contains the observed value only near its upper end")
        + ". The crossed interval is therefore not a two-sided CI for this statistic. Its lower end is likely "
        f"conservative, and it stays above 0 in every variant (lowest: {min(CS[v]['xlo'] for v in VARIANTS):+.2f}). "
        "The two-way interval is a normal interval around the estimate built from variances, so this shift of the "
        f"replicates does not move it; on V4.14.1 it is [{cn['tlo']:+.2f}, {cn['thi']:+.2f}] (SE {cn['tse']:.3f}, "
        f"V_field from the jackknife over fields; lowest lower end across variants "
        f"{min(CS[v]['tlo'] for v in VARIANTS):+.2f}). "
        + ("The replication claim (Spearman > 0.5, CI > 0) holds under it in every variant, and the lower end of the "
           "crossed interval is above 0 as well. " if all(conc[v]["two-way"]["xsource"] == "holds" for v in VARIANTS)
           else "The replication claim's status under it is in Table 1. ")
        + "Institution noise that the two sources share could also raise the observed correlation "
        f"({cn['xs_overlap']:.0%} of the PSEO institution x field rows used on V4.14.1 are also in the Scorecard "
        "field couplings); this is not separated here.\n")
    # ------------------------------------------------------------------ calibration
    L.append("## 6. Calibration of the intervals\n")
    L.append(
        "Null data on the V4.14.1 panel structure (the section 4 panel: "
        f"{CAL['n_cells']} field x cohort cells, {CAL['K']} fields, {CAL['n_inst']} institutions, their F and G): in "
        "every cell the institutions' earnings vectors (all quantiles and horizons jointly) are permuted, "
        "independently across cells. This keeps only within-cell sampling noise, with no field effect and no "
        "institution effect shared across cells, so the variance of a statistic across permuted datasets is its "
        f"true sampling variance ({NCAL_TRUTH} permuted datasets). On each of {NCAL_DATA} further permuted datasets "
        f"the variance estimates behind the intervals are computed ({NCAL_B} replicates per institution draw, "
        "degenerate replicates redrawn as in the main analysis). Cells: ratio of each variance estimate to the true "
        f"sampling variance, mean over the {NCAL_DATA} datasets (min to max). 1 = calibrated; above 1 = too wide.\n")
    L.append("| statistic | true SD | crossed bootstrap | two-way (V_field + V_inst - V_field x inst) | "
             "V_crossed - 2 V_field x inst | V_field only | fields fixed, shared draw (V_inst) | "
             "fields fixed, independent draw per field (V_field x inst) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for st in CAL_STATS:
        def cc(col, st=st):
            return (f"{cag.loc[st, (col, 'mean')]:.2f} ({cag.loc[st, (col, 'min')]:.2f} to "
                    f"{cag.loc[st, (col, 'max')]:.2f})")
        L.append(f"| {st} | {np.sqrt(CAL['vtrue'][st]):.4f} | {cc('crossed')} | {cc('twoway')} | {cc('alt')} | "
                 f"{cc('field')} | {cc('shared')} | {cc('indep')} |")
    L.append("")
    L.append(
        f"Under this null the crossed bootstrap's variance is {cal_lo('crossed'):.1f} to {cal_hi('crossed'):.1f} "
        "times the true sampling variance (means over datasets, by statistic). Each of its components already "
        "carries the within-cell noise: the spread of the per-field values across fields (V_field only, "
        f"{cal_lo('field'):.2f} to {cal_hi('field'):.2f}), the shared institution draw ({cal_lo('shared'):.2f} to "
        f"{cal_hi('shared'):.2f}), and the draw again inside the resampled fields; the crossed bootstrap adds "
        "them. The two-way variance subtracts the field x institution term, which with no shared institution "
        f"effect carries only that noise ({cal_lo('indep'):.2f} to {cal_hi('indep'):.2f}), and comes out at "
        f"{cal_lo('twoway'):.2f} to {cal_hi('twoway'):.2f} on average; on single datasets it ranges from "
        f"{float(cag[('twoway', 'min')].min()):.2f} to {float(cag[('twoway', 'max')].max()):.2f}, because with "
        f"{CAL['K']} fields the variance estimate is itself noisy. V_crossed - 2 V_field x inst gives "
        f"{cal_lo('alt'):.2f} to {cal_hi('alt'):.2f}. Revision 2 called the crossed bootstrap 'mildly conservative "
        "(Owen 2007)'; for these statistics it is not mild, and the phrase is removed. The check has limits. The "
        "null has no institution effect shared across cells, so it tests how each interval counts within-cell "
        "noise, not how V_institution - V_field x institution counts a real shared institution effect. It uses "
        f"{NCAL_DATA} datasets, so the mean ratios carry Monte Carlo error. Given the dataset-to-dataset spread, a "
        "two-way z near 2 is borderline and is reported as such.\n")

    # ------------------------------------------------------------------ revision note
    Qn = QT["new"]
    L.append("## Revisions\n")
    r3d = {v: CT[v]["R"]["dG_minus_dF_all"] for v in ("old", "new")}
    r3b = {v: CT[v]["R"]["bF_y1_all"] for v in ("old", "new")}
    r3q = {v: QT[v]["S"][("F_p75_minus_p50", AVG, "")] for v in ("old", "new")}
    t_changes = [f"'{CONC_LAB[k]}' {' or '.join(sorted({conc[v]['two-way'][k] for v in VARIANTS}))} under the two-way "
                 f"interval in every variant, versus {' or '.join(sorted({conc[v]['crossed'][k] for v in VARIANTS}))} "
                 "under the crossed one"
                 for k in CONC_LAB if any(conc[v]["two-way"][k] != conc[v]["crossed"][k] for v in VARIANTS)]
    L.append("### Revision 3 (2026-09-24): calibrated two-way intervals\n")
    L.append(
        "An independent check found that revision 2's standard, the crossed (pigeonhole) field x institution "
        "bootstrap, is not mildly but substantially conservative for these statistics. Resampling fields and "
        "applying a shared institution draw counts the independent within-cell sampling noise about three times. "
        "It proposed the two-way (field, institution) cluster variance of Cameron, Gelbach and Miller (2011), "
        "V_field + V_institution - V_field x institution, and showed that under it dG - dF and b_F at y1 are "
        "clearly above 0 in both releases. This version accepts the finding. It adds an independent institution "
        "draw per field to sections 3, 4 and 5 (new random streams; the existing draws are unchanged) and reports the "
        "two-way CI and z for every section 3 statistic, every quantile difference, every aggregation in Table 3 "
        "and the cross-source correlation. It bases Table 1 on the two-way interval, with the crossed, fields-fixed "
        "and two-stage statuses beside it. Section 6 repeats the calibration in the script: under a within-cell "
        f"null the crossed variance is {cal_lo('crossed'):.1f} to {cal_hi('crossed'):.1f} times the true sampling "
        f"variance and the two-way variance {cal_lo('twoway'):.2f} to {cal_hi('twoway'):.2f} times. "
        "Consequences: "
        + ("; ".join(t_changes) + ". " if t_changes else "no Table 1 statement changes status between the two "
           "intervals. ")
        + "Revision 2's reading that the brand-over-field difference is 'clear only conditional on the observed "
        "fields', its suggested README/ROADMAP rewording, its flag on scripts/52 item 5 (b_F at y1) and the phrase "
        "'mildly conservative' are withdrawn. The p75 - p50 answer now rests on the two-way interval: the average is "
        f"{fm(r3q['new']['est'])} with z={r3q['new']['tz']:.2f} on V4.14.1 (V4.13.0 z={r3q['old']['tz']:.2f}), "
        + ("at the edge of 0, so 'marginal' replaces revision 2's 'clear only conditional on the observed fields'; "
           "in substance the answer is unchanged. " if tst[AVG] == "touches 0" else
           "clearly above 0. " if tst[AVG] == "above 0" else "not above 0. ")
        + "No estimate changes, and every row of the revision 2 CSV is reproduced; the two-way intervals, the "
        "calibration and the independent-draw redraw counts are new rows. The release answer stays "
        f"'{ttl[:-1]}'.\n")
    L.append("| | V4.13.0 (2025Q4, repo) | V4.14.1 (2026Q2) |")
    L.append("|---|---|---|")
    for lab, rr in (("dG - dF (per yr)", r3d), ("b_F at y1", r3b), ("p75 - p50 coupling, mean y1/y5/y10", r3q)):
        L.append(f"| {lab}: estimate | {fm(rr['old']['est'])} | {fm(rr['new']['est'])} |")
        L.append(f"| {lab}: crossed CI (revision 2 standard) | {xci(rr['old'])} | {xci(rr['new'])} |")
        L.append(f"| {lab}: two-way CI (z) (revision 3 standard) | {tciz(rr['old'])} | {tciz(rr['new'])} |")
        L.append(f"| {lab}: two-way z with IQR-based variances | {rr['old']['tz_r']:.2f} | {rr['new']['tz_r']:.2f} |")
    L.append("")
    L.append("### Revision 2 (2026-09-24): one inference standard for sections 3 and 5 (standard superseded by revision 3)\n")
    t1_before = {k: CONC_LAB[k] for k in ("brand_G", "brand_diff")}
    INF2 = INF[1:]                       # the three kinds of interval of revision 2
    bG_all_hold2 = all(conc[v][kind]["brand_G"] == "holds" for v in VARIANTS for kind in INF2)
    other_all_hold = [k for k in CONC_LAB if k not in t1_before
                      and all(conc[v][kind][k] == "holds" for v in VARIANTS for kind in INF2)]
    other_not = [k for k in CONC_LAB if k not in t1_before and k not in other_all_hold]
    L.append(
        "An independent check found that this note applied two standards of inference. Revision 1 stopped relying "
        "on the two-stage bootstrap for the quantile comparison, because it redraws institutions separately in "
        "each cell while the same institutions recur across cohorts and fields. But Table 1, the release answer "
        "and the section 3 conclusions still rested on scripts/52's two-stage intervals, which have the same flaw. "
        "This version adds crossed and fields-fixed institution-cluster CIs to every section 3 statistic. They use "
        "the weighted-Spearman machinery of section 4 on the p50 panel, the coverage cells and the calendar "
        "cells, with one institution draw per replicate shared by all of them. It also adds an "
        "institution-clustered horizon-label permutation for the slopes and crossed and fields-fixed CIs for the "
        "section 5 correlation. Table 1 now gives each statement's status under all three kinds of interval and "
        "splits the brand statement in two. The previous version marked 'rise loads on academia-wide rank G (dG CI "
        "> 0, dF CI spans 0, dG-dF CI > 0)' as 'holds' in every variant. Now 'dG CI > 0, dF CI spans 0' is "
        + ("'holds' under all three intervals in every variant" if bG_all_hold2 else "as in Table 1")
        + ", and 'dG - dF CI > 0' is (crossed / fields fixed / two-stage) "
        + "; ".join(f"{REL[v]['label']} {' / '.join(conc[v][kind]['brand_diff'] for kind in INF2)}" for v in VARIANTS)
        + ". "
        + ("The other statements hold under all three intervals in every variant. " if not other_not else
           "Statements not 'holds' everywhere: " + "; ".join(CONC_LAB[k] for k in other_not) + ". ")
        + ("The new intervals also show that b_F at y1, which scripts/52's result note uses to argue for a "
           "component tracked better by F, is not clearly above 0 under the crossed interval (section 3; this "
           "reading is withdrawn in revision 3). "
           if bF_x_bad else "")
        + f"The answer to the release question stays '{ttl[:-1]}'. This version changes no estimate. It changes no "
        "number of the previous version either: every row of the previous CSV is reproduced, and the new "
        "intervals are added as new rows. dG - dF, all fields:\n")
    L.append("| | " + " | ".join(REL[v]["label"] for v in ("old", "new")) + " |")
    L.append("|---|---|---|")
    dd = {v: CT[v]["R"]["dG_minus_dF_all"] for v in ("old", "new")}
    L.append("| estimate (per yr) | " + " | ".join(fm(dd[v]["est"]) for v in ("old", "new")) + " |")
    L.append(f"| two-stage CI (scripts/52, {s52.NBOOT}; the previous basis of Table 1) | "
             + " | ".join(cis(dd[v]) for v in ("old", "new")) + " |")
    L.append(f"| institution-cluster CI, fields fixed ({NBOOT}) | " + " | ".join(cci(dd[v]) for v in ("old", "new")) + " |")
    L.append(f"| crossed field x institution CI ({NBOOT}) | " + " | ".join(xci(dd[v]) for v in ("old", "new")) + " |")
    L.append("| share of crossed replicates <= 0 | " + " | ".join(f"{dd[v]['xle0']:.3f}" for v in ("old", "new")) + " |")
    L.append("| Monte Carlo SE of a crossed CI endpoint | " + " | ".join(f"{dd[v]['xmcse']:.4f}" for v in ("old", "new")) + " |")
    L.append("")
    chk = {"new": (-0.002, 0.066), "old": (0.001, 0.075)}       # reported by the independent check
    dmax = max(max(abs(dd[v]["xlo"] - a_), abs(dd[v]["xhi"] - b_)) for v, (a_, b_) in chk.items())
    L.append(
        "The independent check reported crossed CIs for dG - dF from its own runs (1000 replicates, 4 seeds): "
        "[-0.002, +0.066] on V4.14.1, with the lower bound between -0.001 and -0.004 across seeds, and [+0.001, "
        f"+0.075] on V4.13.0. The endpoints of the single draw here differ from those by at most {dmax:.3f}; "
        f"the Monte Carlo SE of one endpoint is about {max(dd[v]['xmcse'] for v in dd):.3f}.\n")
    L.append("### Revision 1 (2026-09-24): inference for the quantile comparison\n")
    L.append(
        "An independent check found that the evidence for 'p75 coupling exceeds median coupling' was overstated. "
        "The earlier permutation flipped the p50/p75 labels independently in every field x cohort cell, and the "
        "earlier two-stage bootstrap redrew institutions separately in every cell. The same "
        f"{Qn['n_inst']} institutions recur across 4 cohorts and up to {Qn['max_fields_inst']} fields, so both "
        "treated repeated information as independent, and the null distribution and the intervals were too "
        "narrow. This version "
        "replaces the permutation with an institution-clustered one, adds the crossed and the fields-fixed "
        "institution-cluster bootstraps, adds the aggregation checks of Table 3, and labels the average over "
        "horizons as post hoc. The earlier answer read 'Slightly, averaged over horizons, but not clearly at any "
        "single horizon', with permutation p<0.001 for the average. Revision 1 left sections 1-3 and 5 unchanged. The "
        "estimates, the two-stage CIs and the independent-flip p-values below are recomputed on the same random "
        "streams as before and equal the earlier ones. V4.14.1, p75 - p50:\n")
    L.append("| | " + " | ".join(HZ) + " | mean (post hoc) |")
    L.append("|---|---|---|---|---|")
    rv = [Sn[("F_p75_minus_p50", h, "")] for h in HZ + [AVG]]
    L.append("| estimate | " + " | ".join(fm(r["est"]) for r in rv) + " |")
    L.append("| permutation p, independent flips per cell (earlier) | " + " | ".join(fpq(r["p_ind"]) for r in rv) + " |")
    L.append("| permutation p, institution-clustered (now) | " + " | ".join(fpq(r["p"]) for r in rv) + " |")
    L.append("| two-stage bootstrap CI (earlier) | " + " | ".join(cis(r) for r in rv) + " |")
    L.append("| institution-cluster bootstrap CI, fields fixed (now) | " + " | ".join(cci(r) for r in rv) + " |")
    L.append("| crossed field x institution bootstrap CI (now) | " + " | ".join(xci(r) for r in rv) + " |")
    L.append("")

    any_fb = (any(r.get("tfallback", False) for v in VARIANTS for r in CT[v]["R"].values())
              or any(r.get("tfallback", False) for v in QT for r in QT[v]["S"].values())
              or any(r.get("tfallback", False) for v in QT for r in QT[v]["sens"].values())
              or any(CS[v].get("tfallback", False) for v in CS))
    L.append("## Method\n")
    L.append(
        "- **Data variants.** `old` = the repo file (V4.13.0). `new_oldinst` = V4.14.1 earnings read with "
        "the V4.13.0 institutions file, which drops the added institutions and isolates revisions. `new` = "
        "V4.14.1 with its own institutions file (primary). `new_alias` = `new` plus four hand-checked name "
        "aliases for added institutions (PSEO -> Wapman: Rutgers New Brunswick, Newark and Camden campuses; "
        "University of Tennessee - Knoxville -> 'University of Tennessee, The'). `new_joinfix` = `new_alias` "
        "plus 19 hand-checked aliases for V4.13.0 institutions (`JOIN_AUDIT` in the script; the script "
        "asserts that both names exist, that the canonical keys differ and that no PSEO institution already "
        "carries the Wapman key). `new` stays the primary because it uses the same join as Phase 1.\n"
        "- **Loader.** `load_er_pseo(..., path=, inst_path=)`: identical filtering and aggregation to the "
        "default. The script asserts that the explicit-path call on the V4.13.0 files equals the default call.\n"
        "- **Career time (b).** `scripts/52` is imported and its module globals `PSEO_EARN`, "
        "`load_er_pseo` and `_pseo_institutions` are pointed at the chosen variant; `load_fixed`, "
        "`fixed_panel`, `run_fixed`, `run_calendar`, `run_coverage`, `group_stat`, `contrast` and "
        "`field_label_perm` run unchanged with scripts/52's seed and random-stream tags. Field groups come from "
        "the Scorecard 4YR baseline (`compute_gap_map`, B=250, seed 52, as scripts/52), which does not depend "
        "on PSEO. Only the statistics needed here are computed. The high-coverage subsample, pooled-0000, legacy "
        "and Scorecard-descriptive parts of scripts/52 are skipped.\n"
        "- **Institution-clustered inference for (b) and (d).** The same machinery as (c). Each replicate draws "
        "every institution of the variant's universe a Multinomial(N, 1/N) number of times. The universe is all "
        "institutions in the fixed-cohort panel cells, the coverage cells and the triple-matched calendar cells. "
        "The draw is shared by every cell. Each cell's Spearman correlations, its b_F/b_G rank regression (from "
        "the weighted Spearman correlations, as scripts/52's cell_metrics) and its coverage partial are computed "
        "exactly from the draw counts. With unit weights they reproduce scripts/52's observed per-field values "
        "(asserted). A replicate with an undefined cell statistic is redrawn whole. Crossed: fields are resampled "
        "with replacement within each field set (all; integrated; others; calendar fields; coverage fields) and "
        "paired with the replicate's institution draw. Fields fixed: the mean over the observed fields. "
        "Institution-clustered horizon-label permutation for the slopes: one of the 6 orders of (y1, y5, y10) per "
        f"institution, shared by all its cells ({NPERM}). Cross-source: one institution draw is applied to both "
        "sources' field couplings, and fields are resampled for the crossed CI. "
        f"{NBOOT} replicates each. The Monte Carlo SE of a percentile endpoint uses the normal approximation. "
        "Table 1's statuses use 2 Monte Carlo SE as the tolerance around 0.\n"
        "- **Two-way (field, institution) cluster variance (all sections; the standard of this note).** "
        "V = V_field + V_institution - V_field x institution (Cameron, Gelbach and Miller 2011, J. Business & "
        "Economic Statistics 29(2)). V_field: between-field variance of the per-field values divided by the number "
        "of fields (sum over the two groups for a contrast; jackknife over fields for the share removed by the "
        "coverage control and the cross-source Spearman correlation; linearized per-field contributions for the "
        "n-weighted mean). V_institution: variance of the fields-fixed replicates under the shared institution draw "
        "above. V_field x institution: variance of the fields-fixed replicates under an independent institution "
        "multinomial draw for each field, shared by that field's cohorts and cell types (for the cross-source row, "
        f"by the field's PSEO and Scorecard couplings); {NBOOT} replicates, degenerate draws redrawn for that field "
        "only. CI: estimate +/- 1.96 sqrt(V); if V <= 0 the larger one-way variance is used (flagged in the CSV; "
        + ("it did not occur). " if not any_fb else "it occurred, see the CSV notes). ")
        + "The Monte Carlo SE of the CI endpoints comes from the Monte Carlo variance of the "
        "two bootstrap variances (fourth-moment formula). Robust version: every variance replaced by "
        "(IQR / 1.349)^2 of the corresponding replicates (V_field from the field bootstrap of the observed "
        "per-field values). Calibration: section 6.\n"
        "- **Quantiles (c).** For each quantile q in p25/p50/p75: the cohort-size-weighted mean of the "
        "institution's CIP-4 q values within field, with the same rows and weights as `load_er_pseo` (asserted: "
        "the p50 aggregate equals `load_er_pseo` earnings). Balanced field x cohort panel (all three quantiles "
        "at y1, y5 and y10, a field rank, n>=15). Coupling = Spearman(F, q) per cell, averaged over cohorts "
        "and then over fields. Inference for the quantile differences, all paired across quantiles and "
        "horizons: (1) institution-clustered quantile-label permutation: for each permutation one Bernoulli flip "
        "per institution swaps its two quantile values (ranked within cell) in every field x cohort cell and "
        f"horizon where it appears; {NPERM} permutations; conditional on the observed fields; two-sided. "
        "(2) Crossed bootstrap (Owen 2007, 'pigeonhole'): each replicate resamples fields with replacement and "
        "draws every institution of the panel a Multinomial(N, 1/N) number of times, the same draw in every cell; "
        "within a cell the Spearman correlation of that resample is computed exactly from the draw counts "
        "(average ranks of tied copies; the script checks this against materialized resamples); replicates with an "
        f"undefined cell coupling are redrawn as a whole; {NBOOT} replicates. (3) The same institution draw with "
        "fields held fixed (institution-cluster bootstrap). (4) The earlier two-stage bootstrap (fields; "
        f"institutions redrawn separately in each cell; {NBOOT}) and the earlier per-cell independent-flip "
        "permutation are kept in the CSV for comparison; both treat each cell's institutions as a fresh sample. "
        "Upper-tail stretch = Spearman(F, p75/p50). The same statistics are computed with G = academia-wide rank. "
        "Aggregation sensitivity: see Table 3.\n"
        "- **Cross-source (d).** `scripts/32.load_pseo_ba('y5')` with its file globals redirected; field "
        "coupling = 1 - gap on PSEO x Wapman (n>=8, >=3 distinct values) and on Scorecard x Wapman "
        "(`scripts/28.build_table`, EARN_MDN_4YR); Spearman across common fields with `scripts/32.boot_corr` "
        "(4000 draws, seed 3).\n"
        "- **Coverage (a).** A usable cell is a released (`status_y{h}_earnings == 1`) bachelor's "
        "institution x CIP-4 cell whose CIP-4 maps to FIELDS66 and whose institution has a Wapman rank in that "
        f"field. Quartiles are of the {int(quart[quart.variant == 'old'].total.sum())} Wapman academia-ranked "
        "institutions (`scripts/28.load_generic`). "
        "This is not the README section 1 definition (scripts/01); section 2 also reports that one, "
        "re-computed on each release.\n")

    # ------------------------------------------------------------------ caveats
    ja_y5, ja_fx = int(ja.usable_y5.sum()), int(ja.usable_fixed_y10.sum())
    old_y5 = int(tot[(tot.variant == "old") & (tot.cohorts == "all cohorts") & (tot.horizon == "y5")].usable.iloc[0])
    old_fx = int(tot[(tot.variant == "old") & (tot.cohorts == "fixed 2001-2010") & (tot.horizon == "y10")].usable.iloc[0])
    new_fx = int(tot[(tot.variant == "new") & (tot.cohorts == "fixed 2001-2010") & (tot.horizon == "y10")].usable_from_new.iloc[0])
    q1_max = int(quart[quart.variant == "old"].total.iloc[0])   # quartiles are in rank order; Q1 = top
    trunc = {}
    for r in newtab[newtab.g_rank <= q1_max].sort_values("g_rank").itertuples():
        m = tuple(h for h in HZ if getattr(r, f"usable_{h}") == 0)
        if m:
            trunc.setdefault(m, []).append(r)
    trunc_txt = []
    for m, rs in trunc.items():
        who = join_and([r.label for r in rs])
        t = f"{who} {'has' if len(rs) == 1 else 'have'} no usable {' or '.join(m)} cell"
        if "y5" in m and "y1" not in m:
            t += " (its usable cells are y1 cells from " + "; ".join(
                (f"{r.label}: " if len(rs) > 1 else "") + cohort_desc(r.cohorts_y1) for r in rs) + ")"
        trunc_txt.append(t)
    tw_only = ({f"'{CONC_LAB[k]}'" for k in CONC_LAB for v in VARIANTS
                if conc[v]["two-way"][k] == "holds" and conc[v]["crossed"][k] != "holds"}
               | ({"b_F > 0 at y1 (section 3)"} if bF_x_bad and bF_t_all else set())
               | ({"the p75 - p50 average (V4.14.1)"} if (tst[AVG] == "above 0" and xst[AVG] != "above 0") else set()))
    L.append("## Caveats\n")
    L.append(
        "- **Weak test for career time.** Most panel institutions and all of their fixed-cohort cells are "
        "the same in both releases. Agreement between the releases therefore shows that the additions do "
        "not move the result; it is not an independent replication.\n"
        f"- **The name join misses institutions in both releases.** The canonical join misses the four "
        f"aliased institutions, and it also misses at least {len(ja)} Wapman institutions already in V4.13.0. "
        "The script checks each of these pairs (both names exist; the normalized keys differ). Among them are "
        + ", ".join(f"{r.wapman_name} ({r.g_rank})" for r in ja.sort_values("g_rank").head(8).itertuples())
        + f" (Wapman academia rank in parentheses). Joined, these {len(ja)} would add {ja_y5:,} usable y5 cells "
        f"(any cohort) to the {old_y5:,} now in use, and {ja_fx:,} usable fixed-cohort y10 cells to {old_fx:,}. "
        f"For the fixed-cohort panel that gain is larger than what V4.14.1's added institutions bring "
        f"({new_fx} such cells). The `new_joinfix` variant uses them. It gives an all-field slope of "
        f"{fm(CT['new_joinfix']['R']['slope_all']['est'])} (crossed CI {xci(CT['new_joinfix']['R']['slope_all'])}) on "
        f"{CT['new_joinfix']['info']['panel_insts']} panel institutions, and the Table 1 statements "
        + ("have the same statuses there as on V4.14.1. "
           if all(conc["new_joinfix"][kind][k] == conc["new"][kind][k] for k in CONC_LAB for kind in INF)
           else "do not change between holding and failing there relative to V4.14.1 (Table 1). "
           if not any(changed_between(conc, k, ("new", "new_joinfix")) for k in CONC_LAB)
           else "change status there (Table 1). ")
        + "Making the fix canonical is a separate task in `src/crosswalks/institutions.py`. It affects both "
        "releases equally and should be followed by re-running scripts/52 and scripts/32. The Scorecard side "
        "of the level-agreement statistic has the same kind of misses: the 4 aliases add "
        f"{CS['new_alias']['n_inst'] - CS['new']['n_inst']} PSEO x Wapman institutions but raise the number of "
        f"matched PSEO-Scorecard institution x field pairs only from {CS['new']['lvl_n']:,} to "
        f"{CS['new_alias']['lvl_n']:,}.\n"
        "- **Quantiles are aggregated, not pooled.** A field's quantile is the weighted mean of its CIP-4 "
        "quantiles, not the quantile of the pooled distribution. Rank statistics are unaffected by monotone "
        "transforms but not by this aggregation. PSEO earnings cover graduates with earnings in UI-covered "
        "employment. Lower quantiles are plausibly the most sensitive to part-year work and graduate "
        "enrolment, and scripts/52 shows that the share of graduates with PSEO earnings falls with prestige. "
        "The p25 gap may therefore partly reflect labour-market attachment rather than pay rates; this is "
        "not tested here.\n"
        "- **Inference depends on what is treated as sampled (sections 3-5).** The clustered permutations and the "
        "fields-fixed bootstraps are conditional on these fields. The two-way variance and the crossed bootstrap "
        "treat fields and institutions as sampled; the crossed bootstrap is conservative for these statistics "
        f"(variance {cal_lo('crossed'):.1f} to {cal_hi('crossed'):.1f} times the truth under the within-cell null "
        "of section 6). The two-way variance is close to the truth on average "
        f"({cal_lo('twoway'):.2f} to {cal_hi('twoway'):.2f} times) but noisy from one dataset to the next "
        f"({float(cag[('twoway', 'min')].min()):.2f} to {float(cag[('twoway', 'max')].max()):.2f}), so a two-way z "
        "near 2 is borderline. Statements that clear 0 under the two-way "
        "interval but not under the crossed one: "
        + (join_and(sorted(tw_only)) if tw_only else "none")
        + ". The average over horizons is a post hoc summary, and the n>=30 and CIP-4 variants in Table "
        f"3 were proposed after the first results; p-values carry Monte Carlo error (about +/-0.005 near p=0.03 "
        f"with {NPERM} permutations), and so do CI endpoints (Table 1's 'edge').\n"
        "- **Conditioning on defined replicates.** The institution-clustered career-time bootstrap redrew "
        + " and ".join(f"{CT[v]['info']['clustered']['redrawn']} of {NBOOT} replicates on {REL[v]['label']}"
                       for v in ("old", "new"))
        + ", all because a resampled cell's b_F/b_G or coverage partial was undefined. The independent draw per "
        "field is conditioned the same way, field by field. scripts/52 conditions the same way within cells. "
        "Near-collinear resamples that are still defined stay in and give the b_F and b_G intervals heavy tails; "
        "the IQR-based z values in the text discount them.\n"
        "- **Top truncation remains.** Even after the refresh, only "
        + f"{int(quart[(quart.variant == 'new') & (quart.quartile == 'Q1 (top)')].covered.iloc[0])} of "
        + f"{int(quart[(quart.variant == 'new') & (quart.quartile == 'Q1 (top)')].total.iloc[0])} top-quartile "
        "Wapman institutions have a usable y5 cell. Among the added top-quartile institutions, "
        + "; ".join(trunc_txt)         + " (section 2 table).\n"
        "- **Non-independent columns.** The variants share most institutions and use common random streams, "
        "so differences between columns are descriptive and not tested.\n"
        "- Descriptive and not causal: institution-level earnings mix value-added with selection (see "
        "SELECTIVITY_RESULT.md).\n")

    # ------------------------------------------------------------------ Phase 1 wording
    L.append("## Phase 1 career-time wording\n")
    ro = Ro
    if diff_tw_all and bG_all_hold:
        L.append(
            "README.md (the career-time bullet) says the rise 'loads on academia-wide brand (+0.038/yr) rather than "
            "field prestige (−0.001/yr)', and ROADMAP.md (Phase 1, item 4) says 'loading on academia-wide brand "
            "rather than field prestige'. Revision 2 of this note suggested weakening both; that suggestion is "
            "withdrawn. Under the two-way interval, on V4.13.0 (the release Phase 1 used): dG "
            f"{fm(ro['dbG_all']['est'])}/yr {tci(ro['dbG_all'])}, dF {fm(ro['dbF_all']['est'])}/yr {tci(ro['dbF_all'])}, "
            f"dG - dF {fm(ro['dG_minus_dF_all']['est'])}/yr {tciz(ro['dG_minus_dF_all'])}; on V4.14.1 dG - dF "
            f"{fm(Rr['dG_minus_dF_all']['est'])}/yr {tciz(Rr['dG_minus_dF_all'])}. The statement holds as written. If "
            "an interval is wanted next to it, a one-line addition would be:\n")
        L.append(
            f"> (dG - dF {fm(ro['dG_minus_dF_all']['est'])}/yr, two-way field x institution cluster CI "
            f"{tci(ro['dG_minus_dF_all'])}; re-checked on the August 2026 PSEO release by scripts/59: "
            f"{fm(Rr['dG_minus_dF_all']['est'])}/yr {tci(Rr['dG_minus_dF_all'])}. Marginal only under the "
            f"conservative pigeonhole bootstrap, {xci(ro['dG_minus_dF_all'])}.)\n")
        L.append("README.md and ROADMAP.md are outside this task's files and were not edited.\n")
    else:
        L.append("The brand statements have the statuses in Table 1 (two-way interval first). README.md and ROADMAP.md "
                 "are outside this task's files and were not edited.\n")

    # ------------------------------------------------------------------ provenance
    L.append("## Provenance (for SOURCES.md)\n")
    L.append(
        f"- **Source:** Census Bureau LEHD Post-Secondary Employment Outcomes (PSEO), release **{pn['version']}** "
        f"(version_pseo.txt: {len(pn['version_states'])} PSEOE entries, {len([x for x in pn['version_states'] if x != 'US'])} "
        f"partner jurisdictions plus the US pseudo-state; build stamp(s) `pseopu_<st>_{'/'.join(pn['stamps'])}`). "
        "Public use.\n"
        f"- **URL:** `{URL_BASE}` (directory listing: all files dated 2026-08-25 17:07; HTTP Last-Modified of "
        f"pseoe_all.csv.gz: {LAST_MODIFIED}).\n"
        f"- **Accessed:** {DOWNLOAD_DATE}, `curl -R` into `data/raw/pseo_2026q2/` (gitignored).\n"
        "- **Files:**\n"
        + "".join(f"  - `{fn}` — {pn[fn]['size']:,} bytes, md5 `{pn[fn]['md5']}`\n"
                  for fn in ("pseoe_all.csv.gz", "pseo_all_institutions.csv", "pseo_all_partners.txt", "version_pseo.txt"))
        + f"- **Contents:** {pn['rows']:,} earnings rows ({pn['rows_I']:,} institution-level), {pn['inst_earn']:,} "
        f"institutions ({pn['inst_file']:,} rows in the institutions file), {len(nn)} partner jurisdictions "
        f"(added vs V4.13.0: {', '.join(sorted(set(nn) - set(no)))}). "
        + (f"Same {pn['ncols']}-column schema as V4.13.0.\n" if pn["ncols"] == po["ncols"] else
           f"{pn['ncols']} columns vs {po['ncols']} in V4.13.0.\n")
        + "- **Not downloaded:** `pseof_all.csv.gz` (flows, 187M in the listing); not needed here.\n"
        f"- **Completeness:** HEAD request {HEAD_CHECK}: Content-Length {SERVER_CONTENT_LENGTH:,} bytes (equal to "
        f"the local file; asserted by the script), Last-Modified {LAST_MODIFIED}; `gzip -t` passes. The "
        "directory holds a single national earnings file, so this is the complete release, not a subset.\n"
        f"- **Note on the V4.13.0 entry in SOURCES.md section 7:** it says 985 institutions. The repo file "
        f"has {po['inst_earn']} institutions with rows and {po['inst_file']} rows in "
        f"pseo_all_institutions.csv, and {len(no)} partner jurisdictions ({len(no) - int('DC' in no)} states + DC). The repo's "
        "version_pseo.txt holds only the US pseudo-state line (V4.13.0 2025Q4). md5 of the repo "
        f"pseoe_all.csv.gz: `{po['pseoe_all.csv.gz']['md5']}`.\n")
    OUT_MD.write_text("\n".join(L))
    print(f"[result] {OUT_MD}")


if __name__ == "__main__":
    main()
