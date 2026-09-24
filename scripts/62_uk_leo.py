"""
scripts/62_uk_leo.py
=============================================================================
UK LEO: does the prestige-pay coupling survive a direct prior-attainment control, and does it
rise over careers?

Public data only: DfE LEO provider-level graduate outcomes (provider x CAH2 subject x cohort x
years after graduation, with breakdowns by prior attainment, sex, POLAR4) and the ORCID-derived
UK PhD->faculty hiring network (GB->GB edges, cached by scripts/40). Descriptive, not causal.

  (a) G = academia-wide UK SpringRank: one SpringRank over ALL GB->GB PhD->faculty edges, nodes
      keyed by ROR id. Face validity: Russell Group vs others, split-half / bootstrap reliability,
      correlation with LEO-measured student prior attainment.
  (b) coupling rho = Spearman across providers of G and LEO median earnings, per CAH2 subject
      (providers >= NMIN), vs the old field-tagged per-CAH2 SpringRank of scripts/40; the US-UK
      cross-subject correlation decomposed into axis / provider sample / earnings window.
  (c) prior attainment: coupling WITHIN LEO prior-attainment bands (same band, across providers)
      vs all-graduate coupling on the same providers; composition-standardised earnings with
      within-provider band gradients (and with national band medians, which over-correct);
      partial coupling given the provider-subject share of high-attainment graduates.
  (d) fixed-cohort career time: cohorts 2013/14-2016/17 observed at YAG 1/3/5 (balanced
      providers), slope per year; calendar-matched contrast (cohort c at YAG5 and cohort c+4 at
      YAG1 are observed in the same tax year).
  (e) setting-priced subjects (national pay scales: nursing, medicine/dentistry, teaching) vs
      the rest.

Seeded (SEED=62); every random draw comes from an order-free stream per analysis. Run:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/62_uk_leo.py
Outputs: data/interim/uk_leo_*.csv, outputs/figures/uk_leo.png, UK_LEO_RESULT.md.
"""
from __future__ import annotations

import sys
import re
import zlib
import zipfile
import importlib.util
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.crosswalks.institutions import normalize_institution_name as norm
from src.ar_pipeline.springrank import springrank

_s40 = importlib.util.spec_from_file_location("s40", ROOT / "scripts" / "40_crossnational_uk.py")
s40 = importlib.util.module_from_spec(_s40); _s40.loader.exec_module(s40)

SEED = 62
NMIN = 15            # providers per cell
DEGMIN = 5           # min hiring-network degree (in+out edges) for a provider's G to be used
NBOOT = 1000         # two-stage bootstrap replicates
NPERM = 1000         # YAG-label permutations
# subject-label contrasts (task e) use exact enumeration of all C(K, k) label assignments
B_EDGE = 200         # edge bootstrap for G stability
N_SPLIT = 100        # split-half replicates for G reliability

LEO_ZIP = ROOT / "data" / "raw" / "leo" / "leo_dashboard.zip"
ROR_ZIP = ROOT / "data" / "raw" / "ror" / "ror-data.zip"
US_GAP = ROOT / "outputs" / "expanded66_gap_map.csv"
US_FVG = ROOT / "outputs" / "field_vs_generic.csv"
INTERIM = ROOT / "data" / "interim"
OUT_PRES = INTERIM / "uk_leo_prestige.csv"
OUT_SUBJ = INTERIM / "uk_leo_subject_coupling.csv"
OUT_BAND = INTERIM / "uk_leo_band_cells.csv"
OUT_CAREER = INTERIM / "uk_leo_career_cells.csv"
OUT_SUM = INTERIM / "uk_leo_summary.csv"
OUT_FIG = ROOT / "outputs" / "figures" / "uk_leo.png"
OUT_MD = ROOT / "UK_LEO_RESULT.md"

COHORTS = [2013, 2014, 2015, 2016]          # academic-year start; observed at YAG 1, 3, 5
YAGS = [1, 3, 5]
YRS = np.array([1.0, 3.0, 5.0])
W_SLOPE = (YRS - YRS.mean()) / ((YRS - YRS.mean()) ** 2).sum()
PA_BANDS = ["PA1", "PA2", "PA3", "PA4", "PA5"]            # UCAS-tariff A-level bands, high -> low
PA_ALL = ["PA1", "PA2", "PA3", "PA4", "PA5", "PA6", "PA7", "PA8", "PA9", "PA_NK"]
PA_LABEL = {"PA1": "4 As or more", "PA2": "360 points", "PA3": "300-359", "PA4": "240-299",
            "PA5": "180-239", "PA6": "below 180", "PA7": "1-2 A-level passes", "PA8": "BTEC",
            "PA9": "Other", "PA_NK": "Not known"}

# setting-priced subjects: pay set by national scales (NHS Agenda for Change / doctors' and dentists'
# national pay scales / teachers' statutory pay scales). "broad" adds Allied health (Agenda for Change).
SETTING_CORE = ["Nursing and midwifery", "Medicine and dentistry", "Education and teaching"]
SETTING_BROAD = SETTING_CORE + ["Allied health"]

# Russell Group members present in LEO (Queen's University Belfast: Northern Ireland providers are
# not in this LEO file). UKPRNs from the LEO file itself.
RUSSELL = {"10006840": "Birmingham", "10007786": "Bristol", "10007788": "Cambridge",
           "10007814": "Cardiff", "10007143": "Durham", "10007790": "Edinburgh",
           "10007792": "Exeter", "10007794": "Glasgow", "10003270": "Imperial",
           "10003645": "King's College London", "10007795": "Leeds", "10006842": "Liverpool",
           "10004063": "LSE", "10007798": "Manchester", "10007799": "Newcastle",
           "10007154": "Nottingham", "10007774": "Oxford", "10007775": "Queen Mary",
           "10007157": "Sheffield", "10007158": "Southampton", "10007784": "UCL",
           "10007163": "Warwick", "10007167": "York"}

# LEO UKPRN -> ROR id, for providers whose LEO (HESA) name does not match any ROR name variant of a
# hiring-network node, or matches the wrong record. Checked by hand against the ROR dump.
ROR_ALIAS = {
    "10007164": "https://ror.org/02nwg5t34",   # University of The West of England, Bristol
    "10007141": "https://ror.org/010jbqd54",   # University of Central Lancashire (ROR: University of Lancashire)
    "10006022": "https://ror.org/05xydav19",   # Solent University (ROR: Southampton Solent University)
    "10002718": "https://ror.org/01khx4a30",   # Goldsmiths' College
    "10006841": "https://ror.org/01t884y44",   # The University of Bolton
    "10007843": "https://ror.org/0067fqk38",   # St Mary's University, Twickenham
    "10007833": "https://ror.org/048kc0s52",   # Glyndwr University (ROR: Wrexham University)
    "10007782": "https://ror.org/040f08y74",   # St. George's Hospital Medical School
    "10005337": "https://ror.org/002g3cb31",   # Queen Margaret University, Edinburgh
    "10007832": "https://ror.org/009tnsj43",   # Birmingham Newman University
    "10007780": "https://ror.org/04vrxay34",   # The School of Oriental and African Studies
    "10031982": "https://ror.org/04tj7zv24",   # BPP University Limited
    "10039956": "https://ror.org/00bge3r76",   # The University of Law Limited
    "10005700": "https://ror.org/044e2ja82",   # SRUC
    "10005523": "https://ror.org/00q3dfs08",   # Rose Bruford College
    "10001478": "https://ror.org/04489at23",   # City (pre-2024-merger cohorts) -> "City, University of London"
}
# excluded from the primary sample (kept in a sensitivity): umbrella node / predominantly
# part-time mature-student providers whose graduates are not comparable first-degree cohorts
EXCLUDE = {"10007797": "University of London (federal umbrella node)",
           "10007773": "The Open University (distance, part-time, mature)",
           "10007760": "Birkbeck College (evening, part-time, mature)"}


def rng_for(tag: str) -> np.random.Generator:
    return np.random.default_rng([SEED, zlib.crc32(tag.encode())])


# =============================================================================================
# data: LEO
# =============================================================================================
def load_leo() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Provider x CAH2 rows (all characteristics but ethnicity) + UK-HEI national subject rows +
    provider-total rows, in compact form (ukprn, subject, grp, tax, cohort, yag + three numeric
    fields; suppressed 'c', 'x', 'low' -> NaN), plus a small provider-attribute table (name, type,
    country, region by tax year and cohort). Read in chunks; each chunk is reduced before it is kept,
    so peak memory stays well below the size of the 787 MB CSV."""
    keep = ["tax_year", "academic_year", "YAG", "ukprn", "provider_name", "provider_type",
            "provider_country_name", "provider_region_name", "cah2_subject_name", "grads",
            "grads_earnings_include", "earnings_median", "characteristic_type", "characteristic_value"]
    z = zipfile.ZipFile(LEO_ZIP)
    name = [n for n in z.namelist() if n.startswith("provider_data")][0]
    parts, attrs, ptot = [], [], []
    for ch in pd.read_csv(z.open(name), encoding="latin-1", dtype=str, usecols=keep, chunksize=200000):
        ch = ch[ch.characteristic_type != "ethnicity"]
        nat = (ch.ukprn == "Total") & (ch.provider_type == "HEI") & \
              (ch.provider_country_name == "Total") & (ch.provider_region_name == "Total")
        ch = ch[(ch.ukprn != "Total") | nat]
        tax = ch.tax_year.str[:4].astype(int)
        coh = ch.academic_year.str[:4].astype(int)
        yag = ch.YAG.astype(int)
        a = ch.loc[ch.ukprn != "Total", ["ukprn", "provider_name", "provider_type",
                                         "provider_country_name", "provider_region_name"]]
        attrs.append(a.assign(tax=tax[a.index], cohort=coh[a.index], yag=yag[a.index]).drop_duplicates())
        t = (ch.ukprn != "Total") & (ch.cah2_subject_name == "Total") & (ch.characteristic_type == "All graduates")
        ptot.append(pd.DataFrame({"ukprn": ch.ukprn[t].values, "cohort": coh[t].values, "yag": yag[t].values,
                                  "reg": ch.provider_region_name[t].values,
                                  "grads": pd.to_numeric(ch.grads[t], errors="coerce").values}))
        # analysis rows: cohorts from 2013/14 on (earlier cohorts enter no analysis)
        k = (coh >= COHORTS[0]).values
        ch, tax, coh, yag = ch[k], tax[k], coh[k], yag[k]
        ct, cv = ch.characteristic_type, ch.characteristic_value
        grp = cv.copy()
        grp[ct == "All graduates"] = "ALL"
        pa = ct == "prior_attainment_code"
        grp[pa] = cv[pa].str.replace("prior_attainment_", "PA", regex=False).replace({"Not known": "PA_NK"})
        pol = ct == "POLAR4"
        grp[pol] = cv[pol].replace({"Not known": "POLAR4_NK"})
        parts.append(pd.DataFrame({
            "ukprn": ch.ukprn.values, "subject": ch.cah2_subject_name.values, "grp": grp.values,
            "tax": tax.values, "cohort": coh.values, "yag": yag.values,
            "reg": ch.provider_region_name.values,
            "grads": pd.to_numeric(ch.grads, errors="coerce").values,
            "grads_earnings_include": pd.to_numeric(ch.grads_earnings_include, errors="coerce").values,
            "earnings_median": pd.to_numeric(ch.earnings_median, errors="coerce").values}))
        del ch, a, grp
    d = pd.concat(parts, ignore_index=True)
    del parts
    # A few multi-site providers are published once per provider region (e.g. The University of Law,
    # London and South East campuses, cohorts 2018/19 on). Keep the region with the most graduates in the
    # provider's all-subject, all-graduate row of that cohort x YAG (ties: region name), for every row.
    pt = pd.concat(ptot, ignore_index=True)
    pt["g"] = pt.grads.fillna(-1.0)
    main = pt.sort_values(["ukprn", "cohort", "yag", "g", "reg"], ascending=[True, True, True, False, True],
                          kind="mergesort").drop_duplicates(["ukprn", "cohort", "yag"]) \
             .set_index(["ukprn", "cohort", "yag"]).reg
    multi = pt.groupby(["ukprn", "cohort", "yag"]).reg.nunique()
    multi = multi[multi > 1].index

    def _drop_minor(x: pd.DataFrame, regcol: str) -> pd.DataFrame:
        cand = x.ukprn.isin(multi.get_level_values(0).unique())
        if not cand.any():
            return x
        sub = x[cand]
        key = pd.MultiIndex.from_frame(sub[["ukprn", "cohort", "yag"]])
        bad = key.isin(multi) & (sub[regcol].values != main.reindex(key).values)
        return x.drop(index=sub.index[bad])
    n0 = len(d)
    d = _drop_minor(d, "reg").drop(columns="reg").reset_index(drop=True)
    pattr = pd.concat(attrs, ignore_index=True).drop_duplicates(ignore_index=True)
    pattr = _drop_minor(pattr, "provider_region_name").drop(columns="yag") \
        .drop_duplicates(ignore_index=True)
    load_leo.minor_rows = n0 - len(d)
    load_leo.multi = sorted(set(multi.get_level_values(0)))
    # one released value per provider x subject x cohort x YAG x group (else pivots would hide duplicates)
    assert not d.duplicated(["ukprn", "subject", "cohort", "yag", "grp"]).any()
    assert not pattr.duplicated(["ukprn", "tax", "cohort"]).any()
    return d, pattr


# =============================================================================================
# data: UK hiring network and the academia-wide prestige G
# =============================================================================================
def load_edges() -> pd.DataFrame:
    e = s40.extract_uk_edges()            # cached GB->GB PhD->faculty edges (one per person)
    return e[["person_orcid", "org_from_ror_id", "org_from_ror_name", "org_to_ror_id",
              "org_to_ror_name", "year"]].copy()


def springrank_ids(e: pd.DataFrame, nodes: list[str] | None = None) -> pd.Series:
    """SpringRank over ROR-id nodes; one unit of weight per person-edge; self-loops dropped."""
    e = e[e.org_from_ror_id != e.org_to_ror_id]
    if nodes is None:
        nodes = sorted(set(e.org_from_ror_id) | set(e.org_to_ror_id))
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    np.add.at(A, (e.org_from_ror_id.map(idx).values, e.org_to_ror_id.map(idx).values), 1.0)
    return pd.Series(springrank(A), index=nodes)


def ror_variants(ids: set[str]) -> dict[str, set[str]]:
    z = zipfile.ZipFile(ROR_ZIP)
    csv = [n for n in z.namelist() if n.endswith(".csv")][0]
    r = pd.read_csv(z.open(csv), dtype=str,
                    usecols=["id", "names.types.label", "names.types.alias", "names.types.ror_display"])
    r = r[r.id.isin(ids)]
    vm: dict[str, set[str]] = {}
    for _, row in r.iterrows():
        for c in ["names.types.label", "names.types.alias", "names.types.ror_display"]:
            v = row[c]
            if isinstance(v, str):
                for part in v.split(";"):
                    part = re.sub(r"^\s*[a-z_]+:\s*", "", part.strip())
                    if part:
                        vm.setdefault(norm(part), set()).add(row.id)
    return vm


def _swap(k: str) -> str | None:
    m = re.match(r"^university (.+)$", k)
    if m:
        return m.group(1) + " university"
    m = re.match(r"^(.+) university$", k)
    if m:
        return "university " + m.group(1)
    return None


def match_providers(prov: pd.DataFrame, vm: dict[str, set[str]]) -> pd.Series:
    def one(ukprn, name):
        if ukprn in ROR_ALIAS:
            return ROR_ALIAS[ukprn]
        k = norm(name)
        for kk in [k, _swap(k), k.replace(" upon tyne", ""), _swap(k.replace(" upon tyne", ""))]:
            if kk and kk in vm and len(vm[kk]) == 1:
                return next(iter(vm[kk]))
        return None
    return pd.Series([one(u, n) for u, n in zip(prov.ukprn, prov.provider_name)], index=prov.index)


def build_prestige(e: pd.DataFrame, pattr: pd.DataFrame):
    """G for LEO providers (attributes from the latest tax year / cohort in which the provider
    appears). Returns provider table and diagnostics."""
    nodes = sorted(set(e.org_from_ror_id) | set(e.org_to_ror_id))
    G = springrank_ids(e, nodes)
    deg = pd.concat([e.org_from_ror_id, e.org_to_ror_id]).value_counts()
    rname = pd.concat([e[["org_from_ror_id", "org_from_ror_name"]].set_axis(["id", "nm"], axis=1),
                       e[["org_to_ror_id", "org_to_ror_name"]].set_axis(["id", "nm"], axis=1)]) \
              .drop_duplicates("id").set_index("id").nm
    # post-2000 variant (first faculty job in 2000 or later)
    e2 = e[e.year >= 2000]
    G2 = springrank_ids(e2)

    p = pattr.sort_values(["tax", "cohort"], kind="mergesort")
    prov = p.groupby("ukprn").agg(provider_name=("provider_name", "last"),
                                  provider_type=("provider_type", "last"),
                                  country=("provider_country_name", "last"),
                                  region=("provider_region_name", "last")).reset_index()
    prov["ror"] = match_providers(prov, ror_variants(set(nodes)))
    prov["ror_name"] = prov.ror.map(rname)
    prov["deg"] = prov.ror.map(deg).fillna(0).astype(int)
    prov["G"] = prov.ror.map(G)
    prov["G_post2000"] = prov.ror.map(G2)
    prov["russell"] = prov.ukprn.isin(RUSSELL)
    prov["geo"] = np.where(prov.country == "England", prov.region, prov.country)
    prov["excluded"] = prov.ukprn.map(EXCLUDE).fillna("")
    prov["primary"] = (prov.provider_type == "HEI") & prov.G.notna() & (prov.deg >= DEGMIN) & \
                      (prov.excluded == "")
    # a ROR id used by two UKPRNs would double-count a node: keep the one with the most rows
    dup = prov[prov.primary].ror.duplicated(keep=False)
    assert not dup.any(), prov[prov.primary][dup]
    return prov, G, deg, e


def prestige_reliability(e: pd.DataFrame, prov: pd.DataFrame) -> dict:
    """Edge-bootstrap rank stability and split-half reliability of G over the primary providers."""
    nodes = sorted(set(e.org_from_ror_id) | set(e.org_to_ror_id))
    base = springrank_ids(e, nodes)
    P = prov[prov.primary].ror.values
    rng = rng_for("edge_boot")
    persons = e.reset_index(drop=True)
    n = len(persons)
    bs = []
    for _ in range(B_EDGE):
        s = persons.iloc[rng.integers(0, n, n)]
        g = springrank_ids(s, nodes)
        bs.append(spearmanr(base[P], g[P])[0])
    rng = rng_for("split_half")
    sh = []
    for _ in range(N_SPLIT):
        m = rng.random(n) < 0.5
        g1 = springrank_ids(persons[m], nodes)
        g2 = springrank_ids(persons[~m], nodes)
        sh.append(spearmanr(g1[P], g2[P])[0])
    sh = np.array(sh)
    r_half = float(np.median(sh))
    return dict(boot_median=float(np.median(bs)), boot_p05=float(np.percentile(bs, 5)),
                split_half_median=r_half, split_half_sb=2 * r_half / (1 + r_half),
                split_half_p05=float(np.percentile(sh, 5)), split_half_p95=float(np.percentile(sh, 95)))


# =============================================================================================
# frequency-weighted rank statistics (exact equivalents of resampling providers with replacement)
# =============================================================================================
def wrank(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Midranks of x (n,) under integer frequency weights W (B, n): the average rank each unit
    would get in the expanded resample (ties -> average rank, as scipy.stats.spearmanr)."""
    o = np.argsort(x, kind="mergesort")
    xs = x[o]
    Ws = W[:, o]
    starts = np.r_[0, np.flatnonzero(np.diff(xs) != 0) + 1]
    gw = np.add.reduceat(Ws, starts, axis=1).astype(float)
    gr = np.cumsum(gw, axis=1) - gw + (gw + 1.0) / 2.0
    gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(xs)]))
    inv = np.empty_like(o)
    inv[o] = np.arange(len(o))
    return gr[:, gid[inv]]


def wcorr(a: np.ndarray, b: np.ndarray, W: np.ndarray) -> np.ndarray:
    sw = W.sum(1, keepdims=True)
    ca = a - (W * a).sum(1, keepdims=True) / sw
    cb = b - (W * b).sum(1, keepdims=True) / sw
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W * ca * cb).sum(1) / np.sqrt((W * ca * ca).sum(1) * (W * cb * cb).sum(1))


def wspear(x, y, W):
    return wcorr(wrank(x, W), wrank(y, W), W)


def wpartial(x, y, Zc, W):
    """Partial Spearman of x, y given covariates Zc (n, k): all variables rank-transformed under
    the weights, x and y WLS-residualised on [1, ranks of Zc], residuals correlated."""
    rx, ry = wrank(x, W), wrank(y, W)
    cols = [np.ones_like(rx)] + [wrank(Zc[:, j], W) for j in range(Zc.shape[1])]
    Z = np.stack(cols, axis=2)
    ZW = Z * W[:, :, None]
    Minv = np.linalg.pinv(np.einsum("bnk,bnl->bkl", ZW, Z))

    def res(r):
        beta = np.einsum("bkl,bl->bk", Minv, np.einsum("bnk,bn->bk", ZW, r))
        return r - np.einsum("bnk,bk->bn", Z, beta)
    return wcorr(res(rx), res(ry), W)


def ones(n):
    return np.ones((1, n))


# =============================================================================================
# bootstrap scaffolding: providers resampled within subject (one draw shared by every cell and
# every analysis of that subject), subjects resampled in the outer stage
# =============================================================================================
class Draws:
    def __init__(self, universe: dict[str, list[str]]):
        self.pos, self.C = {}, {}
        for s in sorted(universe):
            u = sorted(universe[s])
            self.pos[s] = {p: i for i, p in enumerate(u)}
            rng = rng_for(f"providers|{s}")
            idx = rng.integers(len(u), size=(NBOOT, len(u)))
            C = np.zeros((NBOOT, len(u)), dtype=np.int32)
            np.add.at(C, (np.repeat(np.arange(NBOOT), len(u)), idx.ravel()), 1)
            self.C[s] = C

    def W(self, s, provs):
        return self.C[s][:, [self.pos[s][p] for p in provs]]


def two_stage(stack: np.ndarray, tag: str) -> np.ndarray:
    """stack (K subjects, NBOOT): outer stage resamples subjects, each drawn subject contributes an
    independently chosen inner replicate (as scripts/52)."""
    K, B = stack.shape
    rng = rng_for(f"outer|{tag}")
    fi = rng.integers(K, size=(B, K))
    ci = rng.integers(B, size=(B, K))
    return np.nanmean(stack[fi, ci], axis=1)


def ci95(bs) -> tuple[float, float]:
    bs = np.asarray(bs, float)
    bs = bs[np.isfinite(bs)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def perm_p(null, obs) -> float:
    null = np.asarray(null, float)
    return float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + len(null)))


# =============================================================================================
# within-provider prior-attainment gradients
# =============================================================================================
def band_gradients(leo: pd.DataFrame, prov: pd.DataFrame, mode: str = "lambda",
                   weighted: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Within-provider band gradients. National subject x band medians mix the effect of prior
    attainment with institution effects (high-band graduates are concentrated at high-paying
    institutions), so standardising with them removes part of the institution effect. Per subject x
    cohort x YAG cell, over every HEI in the LEO file that releases >= 2 band medians in the cell
    (all ten PA categories), fitted by OLS (or WLS, weight = graduates in the band's earnings figure):
      mode "lambda": log band median = provider FE + lambda * log national band median. The national
                     gradient shrunk (lambda < 1) or stretched to its within-provider size; defined for
                     every band with a national median, so coverage equals the national version's.
      mode "fe":     log band median = provider FE + band FE (free shape). Bands are kept only in the
                     largest connected band component (bands linked through providers that release
                     both); bands never released at such providers (often 4 As or more, below 180) stay
                     unidentified (NaN), so fewer providers pass the coverage rule.
    Returns (gradients in log points up to a cell constant, index (subject, cohort, yag), columns
    PA_ALL; lambda per cell or NaN for mode "fe"). Uses no prestige information."""
    hei = set(prov.ukprn[prov.provider_type == "HEI"])
    d = leo[leo.ukprn.isin(hei) & (leo.subject != "Total") & leo.grp.isin(PA_ALL) &
            leo.cohort.isin(COHORTS) & (leo.earnings_median > 0)]
    d = d[["subject", "cohort", "yag", "ukprn", "grp", "earnings_median", "grads_earnings_include"]]
    nat = np.log(leo[(leo.ukprn == "Total") & (leo.subject != "Total") & leo.cohort.isin(COHORTS)]
                 .pivot_table(index=["subject", "cohort", "yag"], columns="grp", values="earnings_median",
                              aggfunc="first").reindex(columns=PA_ALL))
    rows, lam = {}, {}
    for key, g in d.groupby(["subject", "cohort", "yag"], sort=True):
        if mode == "lambda":
            if key not in nat.index:
                continue
            nx = nat.loc[key]
            g = g[g.grp.map(nx).notna()]
        g = g[g.groupby("ukprn").grp.transform("size") >= 2]
        if g.empty:
            continue
        if mode == "lambda":
            provs = sorted(g.ukprn.unique())
            pi = g.ukprn.map({p: i for i, p in enumerate(provs)}).values
            y = np.log(g.earnings_median.values.astype(float))
            w = np.sqrt(np.nan_to_num(g.grads_earnings_include.values.astype(float), nan=1.0).clip(1.0))
            X = np.zeros((len(g), len(provs) + 1))
            X[np.arange(len(g)), pi] = 1.0
            X[:, -1] = g.grp.map(nx).values
            if weighted:
                X, y = X * w[:, None], y * w
            lam_ = float(np.linalg.lstsq(X, y, rcond=None)[0][-1])
            lam[key] = lam_
            rows[key] = (lam_ * nx).to_dict()
            continue
        parent = {b: b for b in g.grp.unique()}

        def find(b):
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            return b
        for _, gg in g.groupby("ukprn"):
            bb = list(gg.grp)
            for b in bb[1:]:
                parent[find(b)] = find(bb[0])
        root = g.grp.map(find)
        keep = root.value_counts().sort_index(kind="mergesort").idxmax()
        g = g[root == keep]
        bands = [b for b in PA_ALL if b in set(g.grp)]
        if len(bands) < 2:
            continue
        provs = sorted(g.ukprn.unique())
        pi = g.ukprn.map({p: i for i, p in enumerate(provs)}).values
        bi = g.grp.map({b: i for i, b in enumerate(bands)}).values
        X = np.zeros((len(g), len(provs) + len(bands) - 1))
        X[np.arange(len(g)), pi] = 1.0
        m = bi > 0
        X[np.flatnonzero(m), len(provs) + bi[m] - 1] = 1.0
        y = np.log(g.earnings_median.values.astype(float))
        if weighted:
            w = np.sqrt(np.nan_to_num(g.grads_earnings_include.values.astype(float), nan=1.0).clip(1.0))
            X, y = X * w[:, None], y * w
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        rows[key] = dict(zip(bands, np.r_[0.0, beta[len(provs):]]))
    B = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=PA_ALL)
    B.index = pd.MultiIndex.from_tuples(B.index, names=["subject", "cohort", "yag"])
    L = pd.Series(lam, dtype=float)
    if len(L):
        L.index = pd.MultiIndex.from_tuples(L.index, names=["subject", "cohort", "yag"])
    return B, L


# =============================================================================================
# analysis tables
# =============================================================================================
def build_tables(leo: pd.DataFrame, prov: pd.DataFrame, grads: dict[str, pd.DataFrame] | None = None):
    """Wide provider x subject x cohort x YAG table (primary providers), national subject x band
    medians, provider-total composition and regional earnings level."""
    P = prov[prov.primary]
    d = leo[leo.ukprn.isin(P.ukprn) & (leo.subject != "Total")]
    key = ["subject", "cohort", "yag", "ukprn"]
    med = d.pivot_table(index=key, columns="grp", values="earnings_median", aggfunc="first")
    inc = d.pivot_table(index=key, columns="grp", values="grads_earnings_include", aggfunc="first")
    grd = d.pivot_table(index=key, columns="grp", values="grads", aggfunc="first")
    nat = leo[(leo.ukprn == "Total") & (leo.subject != "Total")] \
        .pivot_table(index=["subject", "cohort", "yag"], columns="grp", values="earnings_median",
                     aggfunc="first")
    # composition-standardised expected median: national subject x band medians weighted by the
    # provider-subject's band mix of graduates included in earnings (all ten PA categories)
    shares = inc.reindex(columns=PA_ALL).fillna(0.0)
    tot = shares.sum(1)
    natb = nat.reindex(columns=PA_ALL).reindex(med.index.droplevel("ukprn"))
    natb.index = med.index
    have = natb.notna() & (shares > 0)
    wsum = shares.where(have, 0).sum(1)
    expected = (shares.where(have, 0) * natb.fillna(0)).sum(1) / wsum
    cov_comp = tot / inc["ALL"]
    T = pd.DataFrame({"earn": med["ALL"], "n_earn": inc["ALL"], "grads": grd["ALL"],
                      "expected": expected, "comp_cover": cov_comp,
                      "comp_wcover": wsum / inc["ALL"]})
    T["earn_std"] = np.log(T.earn) - np.log(T.expected)
    T.loc[~(T.comp_wcover >= 0.8), "earn_std"] = np.nan     # band mix must cover >= 80% of the earnings sample
    # the same standardisation with within-provider band gradients (band_gradients) in place of the
    # national band medians; same >= 80% coverage rule over the identified bands
    for tag, Bg in (grads or {}).items():
        bg = Bg.reindex(columns=PA_ALL).reindex(med.index.droplevel("ukprn"))
        bg.index = med.index
        hv = bg.notna() & (shares > 0)
        ws = shares.where(hv, 0).sum(1)
        ex = (shares.where(hv, 0) * np.exp(bg.fillna(0))).sum(1) / ws
        T[f"earn_std_{tag}"] = np.log(T.earn) - np.log(ex)
        T[f"wcover_{tag}"] = ws / inc["ALL"]
        T.loc[~(T[f"wcover_{tag}"] >= 0.8), f"earn_std_{tag}"] = np.nan
    known = grd.reindex(columns=PA_ALL[:-1]).fillna(0).sum(1)
    T["top_share"] = grd.reindex(columns=["PA1", "PA2"]).fillna(0).sum(1) / known
    T["coverage"] = T.n_earn / T.grads
    T = T.reset_index()
    T["G"] = T.ukprn.map(P.set_index("ukprn").G)

    # provider-total rows: institution-wide top share and regional earnings level
    pt = leo[(leo.ukprn != "Total") & (leo.subject == "Total")]
    ptg = pt.pivot_table(index=["cohort", "yag", "ukprn"], columns="grp", values="grads", aggfunc="first")
    ptm = pt.pivot_table(index=["cohort", "yag", "ukprn"], columns="grp", values="earnings_median",
                         aggfunc="first")
    known = ptg.reindex(columns=PA_ALL[:-1]).fillna(0).sum(1)
    inst = pd.DataFrame({"inst_top": ptg.reindex(columns=["PA1", "PA2"]).fillna(0).sum(1) / known,
                         "inst_earn": ptm["ALL"]}).reset_index()
    geo = prov.set_index("ukprn").geo
    ptype = prov.set_index("ukprn").provider_type
    inst["geo"] = inst.ukprn.map(geo)
    inst["hei"] = inst.ukprn.map(ptype).eq("HEI")
    inst["lg"] = np.log(inst.inst_earn)
    h = inst[inst.hei & inst.lg.notna()]
    s = h.groupby(["cohort", "yag", "geo"]).lg.agg(["sum", "count"]).reset_index()
    inst = inst.merge(s, on=["cohort", "yag", "geo"], how="left")
    own = np.where(inst.hei & inst.lg.notna(), inst.lg, 0.0)
    cnt = inst["count"] - (inst.hei & inst.lg.notna()).astype(int)
    inst["geo_level"] = (inst["sum"] - own) / cnt.where(cnt > 0)
    T = T.merge(inst[["cohort", "yag", "ukprn", "inst_top", "geo_level"]],
                on=["cohort", "yag", "ukprn"], how="left")
    return T, med, inc, nat


# =============================================================================================
# cells
# =============================================================================================
def cells_cross(T: pd.DataFrame, yag: int = 5) -> list[dict]:
    """Subject x cohort cells at one YAG: all-graduate coupling sample, and the composition
    sample (valid composition-standardised earnings and covariates) nested in it."""
    out = []
    x = T[(T.yag == yag) & T.cohort.isin(COHORTS) & T.earn.notna() & T.G.notna()]
    for (s, c), g in x.groupby(["subject", "cohort"]):
        if len(g) < NMIN:
            continue
        g = g.sort_values("ukprn")
        comp = g[g.earn_std.notna() & g.earn_std_wp.notna() & g.top_share.notna() &
                 g.inst_top.notna() & g.geo_level.notna()]
        assert comp.earn_std_wpw.notna().all()
        out.append(dict(subject=s, cohort=c, yag=yag, provs=list(g.ukprn), G=g.G.values,
                        earn=g.earn.values.astype(float), comp=comp.reset_index(drop=True)))
    return out


def stats_cross(cell, W, Wc):
    r = {"raw": wspear(cell["G"], cell["earn"], W)}
    c = cell["comp"]
    if Wc is not None:
        G, y = c.G.values, c.earn.values.astype(float)
        r["raw_c"] = wspear(G, y, Wc)
        r["std"] = wspear(G, c.earn_std.values, Wc)
        r["std_wp"] = wspear(G, c.earn_std_wp.values, Wc)
        r["std_wpw"] = wspear(G, c.earn_std_wpw.values, Wc)
        r["std_wp_geo"] = wpartial(G, c.earn_std_wp.values, c[["geo_level"]].values, Wc)
        # free band-FE gradients: only providers whose identified bands cover >= 80% of the earnings
        # sample; raw and national-median versions recomputed on the same providers
        m = c.earn_std_wpfe.notna().values
        if m.sum() >= NMIN:
            r["fe_raw"] = wspear(G[m], y[m], Wc[:, m])
            r["fe_nat"] = wspear(G[m], c.earn_std.values[m], Wc[:, m])
            r["fe_std"] = wspear(G[m], c.earn_std_wpfe.values[m], Wc[:, m])
        r["p_prog"] = wpartial(G, y, c[["top_share"]].values, Wc)
        r["p_inst"] = wpartial(G, y, c[["inst_top"]].values, Wc)
        r["sel"] = wspear(c.inst_top.values, y, Wc)      # coupling with institution selectivity instead of G
        r["p_geo"] = wpartial(G, y, c[["geo_level"]].values, Wc)
        r["p_prog_geo"] = wpartial(G, y, c[["top_share", "geo_level"]].values, Wc)
        r["std_geo"] = wpartial(G, c.earn_std.values, c[["geo_level"]].values, Wc)
    return r


def cells_group(T: pd.DataFrame, med: pd.DataFrame, groups: list[str], yag: int = 5,
                with_sel: bool = False) -> list[dict]:
    """Within-group cells (group = prior-attainment band, sex or POLAR4 quintile): providers with a
    released group median AND a released all-graduate median, same subject x cohort x YAG."""
    m = med.reset_index()
    m = m[(m.yag == yag) & m.cohort.isin(COHORTS)]
    Gm = T.drop_duplicates("ukprn").set_index("ukprn").G
    sel = T.drop_duplicates(["cohort", "yag", "ukprn"]).set_index(["cohort", "yag", "ukprn"]).inst_top
    out = []
    for grp in groups:
        if grp not in m.columns:
            continue
        x = m[m[grp].notna() & m["ALL"].notna()].copy()
        x["G"] = x.ukprn.map(Gm)
        x = x[x.G.notna()]
        for (s, c), g in x.groupby(["subject", "cohort"]):
            if len(g) < NMIN:
                continue
            g = g.sort_values("ukprn")
            sv = np.array([sel.get((c, yag, p), np.nan) for p in g.ukprn], float)
            out.append(dict(subject=s, cohort=c, yag=yag, grp=grp, provs=list(g.ukprn),
                            G=g.G.values, earn=g["ALL"].values.astype(float),
                            gearn=g[grp].values.astype(float),
                            sel=sv if (with_sel and np.isfinite(sv).all()) else None))
    return out


def stats_group(cell, W):
    r = {"within": wspear(cell["G"], cell["gearn"], W), "matched": wspear(cell["G"], cell["earn"], W)}
    if cell.get("sel") is not None:
        r["within_sel"] = wspear(cell["sel"], cell["gearn"], W)
        r["within_p_sel"] = wpartial(cell["G"], cell["gearn"], cell["sel"][:, None], W)
        r["within_sel_p_G"] = wpartial(cell["sel"], cell["gearn"], cell["G"][:, None], W)
        r["matched_sel"] = wspear(cell["sel"], cell["earn"], W)
        r["matched_p_sel"] = wpartial(cell["G"], cell["earn"], cell["sel"][:, None], W)
    return r


def cells_retest(T: pd.DataFrame, med: pd.DataFrame, groups: list[str], yag: int = 5) -> list[dict]:
    """Test-retest cells for the attenuation check: the same providers' group median and
    all-graduate median in adjacent cohorts (c, c+1), same subject, same YAG."""
    m = med.reset_index()
    m = m[(m.yag == yag) & m.cohort.isin(COHORTS)]
    Gm = T.drop_duplicates("ukprn").set_index("ukprn").G
    m = m[m.ukprn.map(Gm).notna()]
    out = []
    for grp in groups:
        w = m.pivot_table(index=["subject", "ukprn"], columns="cohort", values=[grp, "ALL"], aggfunc="first")
        for s in sorted(w.index.get_level_values(0).unique()):
            ws = w.loc[s]
            for c in COHORTS[:-1]:
                cols = [(grp, c), (grp, c + 1), ("ALL", c), ("ALL", c + 1)]
                if not all(k in ws.columns for k in cols):
                    continue
                g = ws[cols].dropna()
                if len(g) < NMIN:
                    continue
                out.append(dict(subject=s, cohort=c, grp=grp, provs=list(g.index), Y=g.values.astype(float)))
    return out


def stats_retest(cell, W):
    Y = cell["Y"]
    return {"r_band": wspear(Y[:, 0], Y[:, 1], W), "r_all": wspear(Y[:, 2], Y[:, 3], W)}


def corrected_share(sb: dict, sr: dict, tag: str):
    """(mean within / mean matched) * sqrt(mean r_all / mean r_band) over subjects present in both
    analyses; one outer draw shared by the four stacks."""
    ss = [s for s in sb["within"] if s in sr["r_band"]]
    stacks = [np.stack([d[s][1] for s in ss]) for d in (sb["within"], sb["matched"], sr["r_all"], sr["r_band"])]
    pts = [np.mean([d[s][0] for s in ss]) for d in (sb["within"], sb["matched"], sr["r_all"], sr["r_band"])]
    K, B = stacks[0].shape
    rng = rng_for(f"outer|{tag}")
    fi = rng.integers(K, size=(B, K)); ci_ = rng.integers(B, size=(B, K))
    w, mm, ra, rb = [np.nanmean(x[fi, ci_], 1) for x in stacks]
    bs = (w / mm) * np.sqrt(ra / rb)
    est = (pts[0] / pts[1]) * np.sqrt(pts[2] / pts[3])
    lo, hi = ci95(bs)
    return dict(est=float(est), lo=lo, hi=hi, k=len(ss), r_all=float(pts[2]), r_band=float(pts[3]),
                raw_ratio=float(pts[0] / pts[1]))


def cells_career(T: pd.DataFrame, med: pd.DataFrame | None = None, grp: str | None = None,
                 col: str = "earn") -> list[dict]:
    """Balanced fixed-cohort panels: subject x cohort, providers with earnings at YAG 1, 3 and 5."""
    if grp is None:
        x = T[T.cohort.isin(COHORTS) & T[col].notna() & T.G.notna()]
        w = x.pivot_table(index=["subject", "cohort", "ukprn"], columns="yag", values=col, aggfunc="first")
        extra = None
        if col == "earn":
            extra = x.pivot_table(index=["subject", "cohort", "ukprn"], columns="yag", values="coverage",
                                  aggfunc="first")
    else:
        m = med.reset_index()
        m = m[m.cohort.isin(COHORTS) & m[grp].notna()]
        w = m.pivot_table(index=["subject", "cohort", "ukprn"], columns="yag", values=grp, aggfunc="first")
        extra = None
    w = w.reindex(columns=YAGS).dropna()
    Gm = T.drop_duplicates("ukprn").set_index("ukprn").G
    out = []
    for (s, c), g in w.groupby(level=[0, 1]):
        provs = list(g.index.get_level_values(2))
        if len(provs) < NMIN:
            continue
        cell = dict(subject=s, cohort=c, grp=grp or "ALL", col=col, provs=provs,
                    G=Gm.reindex(provs).values, Y=g.values.astype(float))
        if extra is not None:
            e = extra.reindex(g.index).reindex(columns=YAGS)
            cell["cov"] = e.values.astype(float) if e.notna().all().all() else None
        out.append(cell)
    return out


def stats_career(cell, W):
    rh = np.stack([wspear(cell["G"], cell["Y"][:, k], W) for k in range(3)], axis=1)
    r = {"y1": rh[:, 0], "y3": rh[:, 1], "y5": rh[:, 2], "slope": rh @ W_SLOPE}
    if cell.get("cov") is not None:
        pc = np.stack([wpartial(cell["G"], cell["Y"][:, k], cell["cov"][:, [k]], W) for k in range(3)], axis=1)
        r["slope_pcov"] = pc @ W_SLOPE
        r["slope_rawcov"] = r["slope"]
    return r


def cells_calendar(T: pd.DataFrame) -> list[dict]:
    """Triple-matched providers: cohort c at YAG1 and YAG5, cohort c+4 at YAG1 (same tax year as
    c at YAG5)."""
    x = T[T.earn.notna() & T.G.notna() & (T.yag.isin([1, 5]))]
    w = x.pivot_table(index=["subject", "ukprn"], columns=["cohort", "yag"], values="earn", aggfunc="first")
    Gm = T.drop_duplicates("ukprn").set_index("ukprn").G
    out = []
    for s in sorted(w.index.get_level_values(0).unique()):
        ws = w.loc[s]
        for c in COHORTS:
            cols = [(c, 1), (c, 5), (c + 4, 1)]
            if not all(k in ws.columns for k in cols):
                continue
            g = ws[cols].dropna()
            if len(g) < NMIN:
                continue
            out.append(dict(subject=s, cohort=c, provs=list(g.index), G=Gm.reindex(g.index).values,
                            Y=g.values.astype(float)))
    return out


def stats_calendar(cell, W):
    a = wspear(cell["G"], cell["Y"][:, 0], W)      # c @ YAG1
    b = wspear(cell["G"], cell["Y"][:, 1], W)      # c @ YAG5
    l = wspear(cell["G"], cell["Y"][:, 2], W)      # c+4 @ YAG1 (same tax year as b)
    return {"within": (b - a) / 4, "calendar": (b - l) / 4, "drift": (l - a) / 4}


# =============================================================================================
# run a cell analysis: point estimates per cell, per-subject bootstrap stacks
# =============================================================================================
def run_cells(cells, statfn, draws: Draws, comp: bool = False):
    rows, boot = [], {}
    for cell in cells:
        s = cell["subject"]
        if comp:
            c = cell["comp"]
            okc = len(c) >= NMIN
            pt = statfn(cell, ones(len(cell["provs"])), ones(len(c)) if okc else None)
            bs = statfn(cell, draws.W(s, cell["provs"]), draws.W(s, list(c.ukprn)) if okc else None)
            n_comp = len(c)
        else:
            pt = statfn(cell, ones(len(cell["provs"])))
            bs = statfn(cell, draws.W(s, cell["provs"]))
            n_comp = np.nan
        row = {k: v for k, v in cell.items() if k in ("subject", "cohort", "yag", "grp", "col")}
        row["n"] = len(cell["provs"])
        if comp:
            row["n_comp"] = n_comp
        for k, v in pt.items():
            row[k] = float(v[0])
        rows.append(row)
        for k, v in bs.items():
            boot.setdefault(k, {}).setdefault(s, []).append(v)
    df = pd.DataFrame(rows)
    # per-subject: mean over the subject's cells (point) and nanmean over cells per replicate (boot)
    subj = {}
    for k in boot:
        pts = df.dropna(subset=[k]).groupby("subject")[k].mean()
        subj[k] = {s: (float(pts[s]), np.nanmean(np.stack(boot[k][s]), axis=0)) for s in pts.index}
    return df, subj


def group_mean(subj_k: dict, subjects: list[str], tag: str):
    ss = [s for s in subjects if s in subj_k]
    if not ss:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, k=0, bs=None)
    est = float(np.mean([subj_k[s][0] for s in ss]))
    bs = two_stage(np.stack([subj_k[s][1] for s in ss]), tag)
    lo, hi = ci95(bs)
    return dict(est=est, lo=lo, hi=hi, k=len(ss), bs=bs)


def paired_diff(subj_a: dict, subj_b: dict, subjects: list[str], tag: str):
    """mean over subjects of (a - b), same draws for a and b within subject."""
    ss = [s for s in subjects if s in subj_a and s in subj_b]
    d = {s: (subj_a[s][0] - subj_b[s][0], subj_a[s][1] - subj_b[s][1]) for s in ss}
    return group_mean(d, ss, tag)


def ratio_of_means(subj_a: dict, subj_b: dict, subjects: list[str], tag: str):
    """mean(a)/mean(b) over subjects with both, bootstrap with shared outer draws."""
    ss = [s for s in subjects if s in subj_a and s in subj_b]
    A = np.stack([subj_a[s][1] for s in ss]); Bm = np.stack([subj_b[s][1] for s in ss])
    K, B = A.shape
    rng = rng_for(f"outer|{tag}")
    fi = rng.integers(K, size=(B, K)); ci_ = rng.integers(B, size=(B, K))
    ra = np.nanmean(A[fi, ci_], 1); rb = np.nanmean(Bm[fi, ci_], 1)
    est = float(np.mean([subj_a[s][0] for s in ss]) / np.mean([subj_b[s][0] for s in ss]))
    lo, hi = ci95(ra / rb)
    return dict(est=est, lo=lo, hi=hi, k=len(ss))


def heterogeneity(subj_k: dict, subjects: list[str]) -> dict:
    """DerSimonian-Laird heterogeneity of subject couplings. Each subject's sampling variance = variance of
    its provider-bootstrap replicates (providers resampled within subject). tau = SD of the subjects' true
    couplings beyond sampling noise; Q-test p from chi2(K - 1)."""
    from scipy.stats import chi2
    ss = [s for s in subjects if s in subj_k]
    y = np.array([subj_k[s][0] for s in ss])
    v = np.array([np.nanvar(subj_k[s][1], ddof=1) for s in ss])
    w = 1.0 / v
    mu = (w * y).sum() / w.sum()
    Q = float((w * (y - mu) ** 2).sum())
    df = len(ss) - 1
    tau2 = max(0.0, (Q - df) / (w.sum() - (w ** 2).sum() / w.sum()))
    return dict(tau=float(np.sqrt(tau2)), I2=max(0.0, (Q - df) / Q), Q=Q, df=df, p=float(chi2.sf(Q, df)),
                sd=float(np.std(y, ddof=1)), se_med=float(np.sqrt(np.median(v))), k=len(ss))


def xsubj_spear(subj_a: dict, subj_b: dict, subjects: list[str], tag: str) -> dict:
    """Spearman across subjects of two subject-level couplings; CI from the two-stage bootstrap with one
    outer draw (subjects) and one inner replicate per drawn subject shared by a and b."""
    ss = [s for s in subjects if s in subj_a and s in subj_b]
    est = float(spearmanr([subj_a[s][0] for s in ss], [subj_b[s][0] for s in ss])[0])
    A = np.stack([subj_a[s][1] for s in ss]); Bm = np.stack([subj_b[s][1] for s in ss])
    K, B = A.shape
    rng = rng_for(f"outer|{tag}")
    fi = rng.integers(K, size=(B, K)); ci_ = rng.integers(B, size=(B, K))
    bs = np.array([spearmanr(A[fi[i], ci_[i]], Bm[fi[i], ci_[i]])[0] for i in range(B)])
    lo, hi = ci95(bs)
    return dict(est=est, lo=lo, hi=hi, k=len(ss))


def label_perm(vals: dict, A: list[str], pool: list[str]) -> tuple[float, int]:
    """Exact two-sided subject-label permutation p for mean(A) - mean(pool \\ A): every one of the
    C(K, k) ways of labelling k of the K subjects as group A (the observed labelling included).
    Returns (p, number of labellings)."""
    from itertools import combinations
    ss = [s for s in pool if s in vals]
    v = np.array([vals[s] for s in ss])
    a = np.array([s in A for s in ss])
    K, k = len(v), int(a.sum())
    if k == 0 or k == K:
        return np.nan, 0
    obs = v[a].mean() - v[~a].mean()
    idx = np.array(list(combinations(range(K), k)))
    sa = v[idx].sum(1)
    null = sa / k - (v.sum() - sa) / (K - k)
    p = float(np.sum(np.abs(null) >= abs(obs) - 1e-12) / len(null))
    return p, len(null)


def career_perm(cells, subjects: list[str], tag: str) -> float:
    """YAG-label permutation: within each cell, each provider's three within-YAG earnings ranks are
    shuffled across YAG 1/3/5; statistic = mean over subjects of mean over cohorts of the slope."""
    rng = rng_for(f"yagperm|{tag}")
    by_s = {}
    obs_by_s = {}
    for cell in cells:
        if cell["subject"] not in subjects:
            continue
        rg = rankdata(cell["G"])
        R = np.column_stack([rankdata(cell["Y"][:, k]) for k in range(3)])
        n = len(rg)
        perm = np.argsort(rng.random((NPERM, n, 3)), axis=2)
        Rp = np.take_along_axis(np.broadcast_to(R, (NPERM, n, 3)), perm, axis=2)
        rh = np.stack([corr_rows_(rg, rankdata(Rp[:, :, k], axis=1)) for k in range(3)], axis=1)
        by_s.setdefault(cell["subject"], []).append(rh @ W_SLOPE)
        obs_by_s.setdefault(cell["subject"], []).append(
            np.array([spearmanr(cell["G"], cell["Y"][:, k])[0] for k in range(3)]) @ W_SLOPE)
    null = np.mean([np.mean(np.stack(v), 0) for v in by_s.values()], 0)
    obs = np.mean([np.mean(v) for v in obs_by_s.values()])
    return perm_p(null, obs)


def corr_rows_(x, Y):
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=1, keepdims=True)
    return (Yc @ xc) / np.sqrt((Yc ** 2).sum(1) * (xc ** 2).sum())


# =============================================================================================
# old field-tagged prestige (scripts/40) and the US comparison
# =============================================================================================
def old_field_prestige(prov: pd.DataFrame) -> pd.DataFrame:
    """scripts/40's per-CAH2 SpringRank F (field-tagged edges, oriented, usable subjects only),
    joined to LEO providers through the ROR name."""
    e = s40.tag_cah2(s40.extract_uk_edges())
    dens, pres = s40.uk_prestige_and_density(e)
    usable = set(dens[dens.usable].cah2)
    key = prov[prov.primary].assign(p=lambda d: d.ror_name.map(norm))[["ukprn", "p"]]
    rows = []
    for cah in sorted(usable):
        f = pres[cah].merge(key, on="p", how="inner")
        rows.append(f.assign(subject=cah)[["subject", "ukprn", "s"]].rename(columns={"s": "F"}))
    return pd.concat(rows, ignore_index=True), dens


def us_coupling_by_cah2() -> tuple[pd.Series, pd.Series]:
    """US field coupling rolled up to CAH2 (mean over project fields): field prestige
    (1 - gap, outputs/expanded66_gap_map.csv) and academia-wide brand (c_G,
    outputs/field_vs_generic.csv from scripts/28)."""
    us = pd.read_csv(US_GAP)[["field", "gap"]]
    us["cah2"] = us.field.map(s40.F2CAH)
    us["rho_us"] = 1 - us.gap
    usF = us.dropna(subset=["cah2"]).groupby("cah2").rho_us.mean()
    g = pd.read_csv(US_FVG)[["field", "c_G"]]
    g["cah2"] = g.field.map(s40.F2CAH)
    usG = g.dropna(subset=["cah2", "c_G"]).groupby("cah2").c_G.mean()
    return usF, usG


# =============================================================================================
# main
# =============================================================================================
def analyse(T, med, tag="", sens=False):
    """All cell analyses on one provider sample. Returns results dict."""
    cx = cells_cross(T, 5)
    bands = cells_group(T, med, PA_BANDS, 5, with_sel=not sens)
    out = dict(cx=cx, bands=bands)
    if not sens:
        out["bands_ext"] = cells_group(T, med, ["PA7", "PA8"], 5)
        out["bands_y1"] = cells_group(T, med, PA_BANDS, 1)
        out["bands_y3"] = cells_group(T, med, PA_BANDS, 3)
        out["sex"] = cells_group(T, med, ["F", "M"], 5)
        out["polar"] = cells_group(T, med, [f"POLAR4_{q}" for q in range(1, 6)], 5)
    out["cc"] = cells_career(T)
    if not sens:
        out["ccs"] = cells_career(T, col="earn_std")
        out["ccs_wp"] = cells_career(T, col="earn_std_wp")
        out["cal"] = cells_calendar(T)
        out["cband"] = [c for b in PA_BANDS for c in cells_career(T, med, grp=b)]
        out["retest"] = cells_retest(T, med, PA_BANDS, 5)
    univ = {}
    for k, cl in out.items():
        for c in cl:
            univ.setdefault(c["subject"], set()).update(c["provs"])
            if "comp" in c:
                univ[c["subject"]].update(c["comp"].ukprn)
    dr = Draws(univ)
    res = {}
    res["cx"] = run_cells(cx, stats_cross, dr, comp=True)
    res["bands"] = run_cells(bands, stats_group, dr)
    res["cc"] = run_cells(out["cc"], stats_career, dr)
    if not sens:
        for k in ["bands_ext", "bands_y1", "bands_y3", "sex", "polar"]:
            res[k] = run_cells(out[k], stats_group, dr)
        res["ccs"] = run_cells(out["ccs"], stats_career, dr)
        res["ccs_wp"] = run_cells(out["ccs_wp"], stats_career, dr)
        res["cal"] = run_cells(out["cal"], stats_calendar, dr)
        res["cband"] = run_cells(out["cband"], stats_career, dr)
        res["retest"] = run_cells(out["retest"], stats_retest, dr)
        res["cells"] = out
    return res


def main():
    print("loading LEO ...")
    leo, pattr = load_leo()
    e = load_edges()
    prov, G, deg, e = build_prestige(e, pattr)
    rel = prestige_reliability(e, prov)
    GB = {"wp": band_gradients(leo, prov, "lambda"), "wpw": band_gradients(leo, prov, "lambda", weighted=True),
          "wpfe": band_gradients(leo, prov, "fe")}
    GRADS = {k: v[0] for k, v in GB.items()}
    T, med, inc, nat = build_tables(leo, prov, GRADS)
    R = []           # key-number rows

    def put(name, spec, est, lo=np.nan, hi=np.nan, p=np.nan, k=np.nan, section=""):
        R.append(dict(section=section, quantity=name, spec=spec, est=est, lo=lo, hi=hi, p=p, k=k))

    def putg(name, spec, g, p=np.nan, section=""):
        put(name, spec, g["est"], g["lo"], g["hi"], p, g["k"], section)

    # ---------------- (a) prestige ----------------
    P = prov[prov.primary].copy()
    P["G_pct"] = P.G.rank(pct=True)
    from scipy.stats import mannwhitneyu
    rg = P[P.russell]
    auc = mannwhitneyu(rg.G, P[~P.russell].G).statistic / (len(rg) * (len(P) - len(rg)))
    it = T[(T.yag == 5) & T.cohort.isin(COHORTS)].groupby("ukprn").inst_top.mean()
    P["inst_top"] = P.ukprn.map(it)
    r_sel = spearmanr(P.G, P.inst_top, nan_policy="omit")[0]
    n_sel = int(P.inst_top.notna().sum())
    r_rg_sel = spearmanr(P.russell, P.inst_top, nan_policy="omit")[0]
    r_post = spearmanr(P.G, P.G_post2000)[0]
    sec = "a"
    put("HEIs in the LEO file", "provider_type HEI, any cohort", float((prov.provider_type == "HEI").sum()), section=sec)
    put("HEIs matched to a hiring-network node", "ROR name variants + hand aliases", float(((prov.provider_type == "HEI") & prov.G.notna()).sum()), section=sec)
    put("primary providers", f"matched HEIs, network degree >= {DEGMIN}, 3 exclusions", float(len(P)), section=sec)
    put("hiring edges", "GB->GB PhD->faculty ORCID edges (scripts/40 cache), one per person", float(len(e)), section=sec)
    put("hiring-network nodes", "ROR ids", float(len(set(e.org_from_ror_id) | set(e.org_to_ror_id))), section=sec)
    put("median first-faculty-job year of the edges", "edges with a start year", float(e.year.median()), k=int(e.year.notna().sum()), section=sec)
    put("Russell Group mean G percentile", f"primary providers; {len(rg)} RG members", float(rg.G_pct.mean()), section=sec)
    put("non-RG mean G percentile", "primary providers", float(P[~P.russell].G_pct.mean()), section=sec)
    put("AUC(RG vs others)", "primary providers", float(auc), section=sec)
    put("RG members in top quartile of G", "primary providers", float((rg.G_pct > 0.75).sum()), k=len(rg), section=sec)
    put("edge-bootstrap rank stability of G", f"median Spearman(base, resample), B={B_EDGE}", rel["boot_median"],
        rel["boot_p05"], section=sec)
    put("split-half reliability of G", f"median Spearman of halves (N={N_SPLIT}); Spearman-Brown",
        rel["split_half_median"], rel["split_half_p05"], rel["split_half_p95"], section=sec)
    put("split-half reliability of G, Spearman-Brown", "full-network reliability", rel["split_half_sb"], section=sec)
    put("Spearman(G, institution top-band share)", "primary providers; share of graduates with >= 360 tariff points, cohorts 2013-16", float(r_sel), k=n_sel, section=sec)
    put("Spearman(G, G from post-2000 hires only)", "primary providers", float(r_post), section=sec)

    # ---------------- main analyses ----------------
    res = analyse(T, med)
    dfx, sx = res["cx"]
    subs = sorted(sx["raw"])
    compsubs = sorted(sx["std"])
    sec = "b"
    putg("coupling rho(G, earnings), YAG5", f"{len(subs)} CAH2 subjects; mean over cohorts 2013-16; providers>=15", group_mean(sx["raw"], subs, "raw"), section=sec)
    covx = T[(T.yag == 5) & T.cohort.isin(COHORTS) & T.earn.notna() & T.G.notna()].coverage
    put("median earnings coverage (included / graduates)", "YAG5 cross-section rows", float(covx.median()), section=sec)

    # old field-tagged F on the same cells (16 subjects)
    F, dens = old_field_prestige(prov)
    Fm = F.set_index(["subject", "ukprn"]).F
    frows = []
    for c in res["cells"]["cx"]:
        f = np.array([Fm.get((c["subject"], p), np.nan) for p in c["provs"]])
        ok = np.isfinite(f)
        if ok.sum() >= NMIN:
            frows.append(dict(subject=c["subject"], cohort=c["cohort"], n=int(ok.sum()),
                              rho_F=spearmanr(f[ok], c["earn"][ok])[0],
                              rho_G=spearmanr(c["G"][ok], c["earn"][ok])[0],
                              rho_FG=spearmanr(f[ok], c["G"][ok])[0]))
    FR = pd.DataFrame(frows)
    fs = FR.groupby("subject")[["rho_F", "rho_G", "rho_FG", "n"]].mean()
    old = pd.read_csv(INTERIM / "us_uk_gap_comparison.csv")
    old["rho_old"] = 1 - old.uk_gap
    fs["rho_old_s40"] = old.set_index("cah2").rho_old.reindex(fs.index)
    put("mean rho with field-tagged F (scripts/40 axis)", f"{len(fs)} subjects; same providers; YAG5 cohorts 2013-16",
        float(fs.rho_F.mean()), k=len(fs), section=sec)
    put("mean rho with academia-wide G", f"same {len(fs)} subjects and providers", float(fs.rho_G.mean()), k=len(fs), section=sec)
    put("mean Spearman(F, G) within subject", f"{len(fs)} subjects", float(fs.rho_FG.mean()), k=len(fs), section=sec)
    put("subjects with rho_G > rho_F", f"{len(fs)} subjects", float((fs.rho_G > fs.rho_F).sum()), k=len(fs), section=sec)
    put("mean UK coupling in scripts/40 (1 - uk_gap)", "scripts/40: tax year 2022/23 YAG5, name-matched providers", float(old.rho_old.mean()), k=len(old), section=sec)
    us, usG = us_coupling_by_cah2()
    ukG = pd.Series({s: sx["raw"][s][0] for s in subs})
    both = [s for s in old.cah2 if s in ukG.index]
    assert sorted(both) == sorted(fs.index), (both, list(fs.index))
    oldr = old.set_index("cah2").rho_old

    def xcorr(a: pd.Series, b: pd.Series, kk: list[str]):
        """Spearman across subjects kk of a and b, dropping subjects missing in either."""
        z = pd.DataFrame({"a": a.reindex(kk), "b": b.reindex(kk)}).dropna()
        r_, p_ = spearmanr(z.a, z.b)
        return float(r_), float(p_), len(z)

    # Like-for-like decomposition of the US-UK cross-subject correlation over scripts/40's 16 subjects.
    # The scripts/40 value differs from the G value in three ways at once: axis (F vs G), provider sample
    # (providers with a field-tagged F vs all primary providers) and earnings window (tax year 2022/23
    # vs the mean of cohorts 2013-16 at YAG5). Cohort 2016/17 at YAG5 is tax year 2022/23.
    f16 = FR[FR.cohort == 2016].set_index("subject")
    ukG16 = dfx[dfx.cohort == 2016].set_index("subject").raw
    win = "cohort 2016/17 at YAG5 (= tax year 2022/23, scripts/40's window)"
    avg = "YAG5, mean of cohorts 2013-16"
    XN = [("s40", "F (field-tagged), scripts/40 as published", "tax year 2022/23 YAG5; scripts/40's name-matched providers", oldr),
          ("F_fp_w", "F on F-providers", f"{win}; primary providers with F", f16.rho_F),
          ("G_fp_w", "G on F-providers", f"{win}; primary providers with F", f16.rho_G),
          ("G_all_w", "G on all primary providers", f"{win}", ukG16),
          ("F_fp", "F on F-providers", f"{avg}; primary providers with F (the cells of the within-subject F vs G comparison)", fs.rho_F),
          ("G_fp", "G on F-providers", f"{avg}; primary providers with F (same cells)", fs.rho_G),
          ("G_all", "G on all primary providers", f"{avg}; all primary providers", ukG)]
    kk16 = sorted(both)
    U16 = us.reindex(kk16).values
    brng = rng_for("xnat_boot")
    bidx = brng.integers(len(kk16), size=(NBOOT, len(kk16)))
    xboot = {}
    for key in ["F_fp", "G_fp", "G_all"]:
        v = dict((k, s) for k, _, _, s in XN)[key].reindex(kk16).values
        xboot[key] = np.array([spearmanr(v[i], U16[i])[0] for i in bidx])
    for key, lab, spec, ser in XN:
        r_, p_, n_ = xcorr(ser, us, kk16)
        lo_, hi_ = ci95(xboot[key]) if key in xboot else (np.nan, np.nan)
        put(f"US-UK cross-subject Spearman (16 subjects): UK {lab} vs US field prestige"
            + (" [scripts/40 window]" if key.endswith("_w") else ""),
            spec + ("; CI = subject bootstrap" if key in xboot else ""), r_, lo_, hi_, p=p_, k=n_, section=sec)
    put("subject-bootstrap SE of the 16-subject US-UK Spearman, G on all primary providers",
        f"{NBOOT} resamples of the 16 subjects", float(np.nanstd(xboot["G_all"])), k=len(kk16), section=sec)
    ptx = {k: xcorr(s, us, kk16)[0] for k, _, _, s in XN}
    for lab, a_, b_ in [("axis effect: G minus F, same F-providers and cells", "G_fp", "F_fp"),
                        ("provider-sample effect: G on all primary providers minus G on F-providers", "G_all", "G_fp"),
                        ("total: G on all primary providers minus F on F-providers", "G_all", "F_fp")]:
        put(f"US-UK 16-subject Spearman, {lab}", f"{avg}; paired subject bootstrap", ptx[a_] - ptx[b_],
            *ci95(xboot[a_] - xboot[b_]), k=len(kk16), section=sec)
    loo = [xcorr(ukG.drop(s), us, [x for x in kk16 if x != s])[0] for s in kk16]
    put("US-UK 16-subject Spearman, G on all primary providers: leave-one-subject-out",
        "estimate = full 16; range in CI column = min, max over the 16 drops", ptx["G_all"], min(loo), max(loo),
        k=len(kk16), section=sec)
    r_, p_, n_ = xcorr(ukG, us, [x for x in kk16 if x != "Nursing and midwifery"])
    put("US-UK Spearman, G on all primary providers, without Nursing and midwifery", "15 subjects", r_, p=p_, k=n_, section=sec)
    # every single cohort x YAG cross-section (G on all primary providers)
    xs = []
    for y in YAGS:
        for cl in cells_cross(T, y):
            xs.append(dict(subject=cl["subject"], cohort=cl["cohort"], yag=y, rho=spearmanr(cl["G"], cl["earn"])[0]))
    XS = pd.DataFrame(xs)
    allm = [s for s in ukG.index if s in us.index and s in usG.index]
    for lab, kk in [("16 subjects", kk16), (f"{len(allm)} mapped subjects", allm)]:
        vals = [xcorr(g.set_index("subject").rho, us, kk) for _, g in XS.groupby(["cohort", "yag"])]
        rr = np.array([v[0] for v in vals])
        put(f"US-UK Spearman ({lab}), G on all primary providers: single cohort x YAG cross-sections",
            f"{len(rr)} cross-sections (cohorts 2013-16 x YAG 1/3/5); estimate = median; range in CI column = min, max; "
            f"k = min subjects per cross-section", float(np.median(rr)), float(rr.min()), float(rr.max()),
            k=min(v[2] for v in vals), section=sec)
    for lab, a_, b_, kk, spec in [
            ("UK F on F-providers vs US academia-wide brand c_G", fs.rho_F, usG, kk16, avg),
            ("UK G on F-providers vs US academia-wide brand c_G", fs.rho_G, usG, kk16, avg),
            ("UK G on all primary providers vs US academia-wide brand c_G", ukG, usG, kk16, avg),
            ("UK standardised coupling, national band medians (over-corrects) vs US field prestige",
             pd.Series({s: sx["std"][s][0] for s in sx["std"]}), us, kk16, avg + "; composition sample"),
            ("UK standardised coupling, within-provider band gradients vs US field prestige",
             pd.Series({s: sx["std_wp"][s][0] for s in sx["std_wp"]}), us, kk16, avg + "; composition sample")]:
        r_, p_, n_ = xcorr(a_, b_, kk)
        put(f"US-UK cross-subject Spearman (16 subjects): {lab}", spec, r_, p=p_, k=n_, section=sec)
    for lab, b in [("US field prestige", us), ("US academia-wide brand c_G", usG)]:
        r_, p_, n_ = xcorr(ukG, b, allm)
        put(f"US-UK cross-subject Spearman: UK G vs {lab}, all mapped subjects", f"CAH2 subjects with a US counterpart; {avg}",
            r_, p=p_, k=n_, section=sec)

    # ---------------- (c) prior attainment ----------------
    sec = "c"
    for k, lab in [("raw_c", "raw coupling, composition sample"),
                   ("std", "composition-standardised earnings (national subject x band medians)"),
                   ("std_wp", "composition-standardised earnings (within-provider band gradients)"),
                   ("std_wpw", "composition-standardised earnings (within-provider band gradients, count-weighted)"),
                   ("p_prog", "partial | program share of graduates with >= 360 points"),
                   ("p_inst", "partial | institution-wide share with >= 360 points"),
                   ("p_geo", "partial | regional earnings level"),
                   ("std_geo", "composition-standardised + partial | regional earnings level"),
                   ("std_wp_geo", "within-provider-standardised + partial | regional earnings level"),
                   ("p_prog_geo", "partial | program share + regional earnings level")]:
        putg(lab, "YAG5; cohorts 2013-16; subjects with composition sample>=15", group_mean(sx[k], compsubs, k), section=sec)
    for k in ["std", "std_wp", "std_wpw", "p_prog", "p_inst"]:
        put(f"share surviving: {k} / raw_c", "ratio of subject means", **{kk: v for kk, v in
            ratio_of_means(sx[k], sx["raw_c"], compsubs, "ratio" + k).items() if kk in ("est", "lo", "hi", "k")}, section=sec)
    fsubs = sorted(sx["fe_std"])
    for k, lab in [("fe_raw", "free band-FE subsample: raw coupling"),
                   ("fe_nat", "free band-FE subsample: standardised, national band medians"),
                   ("fe_std", "free band-FE subsample: standardised, within-provider band FE")]:
        putg(lab, "YAG5; cohorts 2013-16; providers whose FE-identified bands cover >= 80% of the earnings sample",
             group_mean(sx[k], fsubs, k), section=sec)
    for k in ["fe_nat", "fe_std"]:
        put(f"share surviving: {k} / fe_raw", "free band-FE subsample; ratio of subject means", **{kk: v for kk, v in
            ratio_of_means(sx[k], sx["fe_raw"], fsubs, "ratio" + k).items() if kk in ("est", "lo", "hi", "k")}, section=sec)
    ncx = [(len(cl["comp"]), int(cl["comp"].earn_std_wpfe.notna().sum())) for cl in res["cells"]["cx"]]
    put("providers per cell: composition sample vs free band-FE subsample", "mean over cross-section cells; CI column = FE subsample",
        float(np.mean([a for a, _ in ncx])), float(np.mean([b for _, b in ncx])), section=sec)
    # why the national-median standardisation over-corrects: band gradients, national vs within provider
    Bw = GRADS["wpfe"]
    Bw5 = Bw[(Bw.index.get_level_values("yag") == 5)]
    natl = np.log(nat.reindex(columns=PA_ALL)).reindex(Bw5.index)
    for hi_b, lo_b in [("PA2", "PA4"), ("PA3", "PA5")]:
        ok = Bw5[hi_b].notna() & Bw5[lo_b].notna() & natl[hi_b].notna() & natl[lo_b].notna()
        wg = (Bw5[hi_b] - Bw5[lo_b])[ok]
        ng = (natl[hi_b] - natl[lo_b])[ok]
        put(f"log earnings gap {hi_b} - {lo_b}: national band medians", "median over subject x cohort cells, YAG5, cohorts 2013-16",
            float(ng.median()), k=int(ok.sum()), section=sec)
        put(f"log earnings gap {hi_b} - {lo_b}: within provider (provider FE + band FE)", "same cells; all HEIs releasing >= 2 band medians",
            float(wg.median()), k=int(ok.sum()), section=sec)
    for key, lab in [("wp", "OLS"), ("wpw", "WLS, count-weighted")]:
        L5 = GB[key][1][GB[key][1].index.get_level_values("yag") == 5]
        put(f"lambda: within-provider / national band gradient ({lab})", "median over subject x cohort cells, YAG5, cohorts 2013-16; CI column = IQR",
            float(L5.median()), float(L5.quantile(0.25)), float(L5.quantile(0.75)), k=len(L5), section=sec)
    dfb, sb = res["bands"]
    bsubs = sorted(sb["within"])
    gw = group_mean(sb["within"], bsubs, "within"); gm = group_mean(sb["matched"], bsubs, "matched")
    putg("within-band coupling (bands 1-5)", "YAG5; subject x cohort x band cells, providers>=15 with a released band median", gw, section=sec)
    putg("all-graduate coupling, same providers", "matched to the within-band cells", gm, section=sec)
    putg("within-band minus matched", "paired", paired_diff(sb["within"], sb["matched"], bsubs, "wdiff"), section=sec)
    rom = ratio_of_means(sb["within"], sb["matched"], bsubs, "wratio")
    put("share surviving: within-band / matched", "ratio of subject means", rom["est"], rom["lo"], rom["hi"], k=rom["k"], section=sec)
    dfr, sr = res["retest"]
    cs = corrected_share(sb, sr, "corr")
    put("test-retest reliability, band medians", "adjacent cohorts, same providers, YAG5 (subject mean)", cs["r_band"], k=cs["k"], section=sec)
    put("test-retest reliability, all-graduate medians", "same providers and cohort pairs", cs["r_all"], k=cs["k"], section=sec)
    put("share surviving, unadjusted, same subjects as the attenuation check", "within/matched; subjects in both analyses",
        cs["raw_ratio"], k=cs["k"], section=sec)
    put("share surviving, attenuation-adjusted", "(within/matched) * sqrt(r_all/r_band); subjects in both analyses",
        cs["est"], cs["lo"], cs["hi"], k=cs["k"], section=sec)
    ssel = sorted(sb["within_sel"])
    for k, lab in [("within_sel", "within-band: Spearman(institution top-band share, band earnings)"),
                   ("within_p_sel", "within-band: partial rho(G, band earnings | institution top-band share)"),
                   ("within_sel_p_G", "within-band: partial rho(institution top-band share, band earnings | G)"),
                   ("matched_sel", "matched all-graduate: Spearman(institution top-band share, earnings)"),
                   ("matched_p_sel", "matched all-graduate: partial rho(G, earnings | institution top-band share)")]:
        putg(lab, "band cells with institution share for every provider", group_mean(sb[k], ssel, k), section=sec)
    gp = P.set_index("ukprn").G_pct
    sd_b = np.mean([np.std(gp.reindex(c["provs"]).values) for c in res["cells"]["bands"]])
    sd_x = np.mean([np.std(gp.reindex(c["provs"]).values) for c in res["cells"]["cx"]])
    put("SD of G percentile within cells: band cells", "range-restriction diagnostic", float(sd_b), section=sec)
    put("SD of G percentile within cells: all-graduate cells", "range-restriction diagnostic", float(sd_x), section=sec)
    for b in PA_BANDS:
        x = dfb[dfb.grp == b]
        if len(x):
            put(f"band {b} ({PA_LABEL[b]}): within / matched", f"{len(x)} cells, {x.subject.nunique()} subjects (cell means)",
                float(x.within.mean()), float(x.matched.mean()), k=x.subject.nunique(), section=sec)
    for key, lab in [("bands_ext", "bands 7-8 (1-2 A-level passes, BTEC)"), ("bands_y1", "bands 1-5 at YAG1"),
                     ("bands_y3", "bands 1-5 at YAG3"), ("sex", "within sex"), ("polar", "within POLAR4 quintile")]:
        d_, s_ = res[key]
        ss = sorted(s_["within"])
        if ss:
            a = group_mean(s_["within"], ss, key + "w"); m = group_mean(s_["matched"], ss, key + "m")
            dd = paired_diff(s_["within"], s_["matched"], ss, key + "d")
            put(f"{lab}: within", f"{len(d_)} cells", a["est"], a["lo"], a["hi"], k=a["k"], section=sec)
            put(f"{lab}: matched all-graduate", "same providers", m["est"], m["lo"], m["hi"], k=m["k"], section=sec)
            put(f"{lab}: within minus matched", "paired", dd["est"], dd["lo"], dd["hi"], k=dd["k"], section=sec)

    # does the cross-subject variation survive the control? (heterogeneity beyond provider-sampling noise)
    nonset = [s for s in compsubs if s not in SETTING_CORE]
    HET = []
    for key, dct, pool, lab in [
            ("raw", sx["raw"], subs, "all graduates, full cross-section"),
            ("raw_c", sx["raw_c"], compsubs, "all graduates, composition sample"),
            ("std_wp", sx["std_wp"], compsubs, "standardised, within-provider band gradients"),
            ("std", sx["std"], compsubs, "standardised, national band medians (over-corrects)"),
            ("p_inst", sx["p_inst"], compsubs, "partial | institution selectivity"),
            ("within", sb["within"], bsubs, "within prior-attainment band (bands 1-5)"),
            ("matched", sb["matched"], bsubs, "all graduates, band-cell providers"),
            ("sel", sx["sel"], compsubs, "coupling with institution selectivity instead of G"),
            ("raw_c_ns", sx["raw_c"], nonset, "all graduates, composition sample, without the 3 setting-priced subjects"),
            ("std_wp_ns", sx["std_wp"], nonset, "standardised within provider, without the 3 setting-priced subjects"),
            ("p_inst_ns", sx["p_inst"], nonset, "partial | institution selectivity, without the 3 setting-priced subjects")]:
        h = heterogeneity(dct, pool)
        HET.append(dict(key=key, **h))
        put(f"cross-subject SD of true coupling (DL tau): {lab}",
            f"YAG5, cohorts 2013-16; observed SD {h['sd']:.3f}, median sampling SE {h['se_med']:.3f}; "
            f"I2 = {h['I2']:.2f}; Q = {h['Q']:.0f} on {h['df']} df (p = Q-test)", h["tau"], p=h["p"], k=h["k"], section=sec)
    for a_, b_, pool, lab in [("raw_c", "std_wp", compsubs, "all graduates vs standardised within provider"),
                              ("raw_c", "p_inst", compsubs, "all graduates vs partial | institution selectivity"),
                              ("matched", "within", bsubs, "all graduates (band-cell providers) vs within band"),
                              ("raw_c", "sel", compsubs, "coupling with G vs coupling with institution selectivity")]:
        src = sb if a_ in ("matched", "within") else sx
        xs_ = xsubj_spear(src[a_], src[b_], pool, f"xs{a_}{b_}")
        put(f"cross-subject Spearman of subject couplings: {lab}", "YAG5, cohorts 2013-16; CI = two-stage bootstrap",
            xs_["est"], xs_["lo"], xs_["hi"], k=xs_["k"], section=sec)
    putg("coupling with institution selectivity instead of G (all graduates)",
         "composition sample; Spearman(institution share >= 360 points, earnings)", group_mean(sx["sel"], compsubs, "sel"), section=sec)

    # ---------------- (d) career time ----------------
    sec = "d"
    dfc, sc = res["cc"]
    csubs = sorted(sc["slope"])
    pperm = career_perm(res["cells"]["cc"], csubs, "all")
    putg("career slope of rho(G, earnings) per year", "fixed cohorts 2013-16 at YAG 1/3/5, balanced providers>=15",
         group_mean(sc["slope"], csubs, "slope"), p=pperm, section=sec)
    for y in ["y1", "y3", "y5"]:
        putg(f"coupling at {y.upper().replace('Y', 'YAG')}", "balanced fixed-cohort panel", group_mean(sc[y], csubs, "lvl" + y), section=sec)
    by_coh = dfc.groupby("cohort").slope.mean()
    put("cohorts with positive mean slope", "cohorts 2013-16", float((by_coh > 0).sum()), k=len(by_coh), section=sec)
    seg1 = {s: ((sc["y3"][s][0] - sc["y1"][s][0]) / 2, (sc["y3"][s][1] - sc["y1"][s][1]) / 2) for s in csubs}
    seg2 = {s: ((sc["y5"][s][0] - sc["y3"][s][0]) / 2, (sc["y5"][s][1] - sc["y3"][s][1]) / 2) for s in csubs}
    putg("segment YAG1->3 per year", "balanced", group_mean(seg1, csubs, "seg1"), section=sec)
    putg("segment YAG3->5 per year", "balanced", group_mean(seg2, csubs, "seg2"), section=sec)
    cs_ = sorted(sc["slope_pcov"])
    putg("slope of partial coupling | coverage", "cells with coverage at all YAGs", group_mean(sc["slope_pcov"], cs_, "pcov"), section=sec)
    putg("raw slope, same cells", "cells with coverage at all YAGs", group_mean(sc["slope_rawcov"], cs_, "rawcov"), section=sec)
    dfs, ss_ = res["ccs"]
    putg("career slope, composition-standardised", "national band medians (over-corrects); balanced panels of standardised earnings", group_mean(ss_["slope"], sorted(ss_["slope"]), "sslope"), section=sec)
    for y in ["y1", "y3", "y5"]:
        putg(f"composition-standardised coupling at {y.upper().replace('Y', 'YAG')}", "national band medians (over-corrects); balanced panels",
             group_mean(ss_[y], sorted(ss_[y]), "slvl" + y), section=sec)
    dfw, sw_ = res["ccs_wp"]
    putg("career slope, within-provider-standardised", "within-provider band gradients; balanced panels of standardised earnings",
         group_mean(sw_["slope"], sorted(sw_["slope"]), "wslope"), section=sec)
    for y in ["y1", "y3", "y5"]:
        putg(f"within-provider-standardised coupling at {y.upper().replace('Y', 'YAG')}", "within-provider band gradients; balanced panels",
             group_mean(sw_[y], sorted(sw_[y]), "wlvl" + y), section=sec)
    dfcb, scb = res["cband"]
    putg("career slope, within prior-attainment band", "band 1-5 balanced panels, providers>=15", group_mean(scb["slope"], sorted(scb["slope"]), "bslope"), section=sec)
    for y in ["y1", "y3", "y5"]:
        putg(f"within-band coupling at {y.upper().replace('Y', 'YAG')}", "band 1-5 balanced panels, providers>=15",
             group_mean(scb[y], sorted(scb[y]), "blvl" + y), section=sec)
    for b in PA_BANDS:
        x = dfcb[dfcb.grp == b]
        if len(x):
            put(f"career slope within band {b}", f"{len(x)} cells, {x.subject.nunique()} subjects (cell mean)",
                float(x.slope.mean()), k=x.subject.nunique(), section=sec)
    dfl, sl = res["cal"]
    lsubs = sorted(sl["within"])
    for k, lab in [("within", "within-cohort (c@5 - c@1)/4"), ("calendar", "calendar-matched (c@5 - (c+4)@1)/4, same tax year"),
                   ("drift", "cross-cohort drift at YAG1 ((c+4)@1 - c@1)/4")]:
        putg(lab, "triple-matched providers>=15", group_mean(sl[k], lsubs, "cal" + k), section=sec)

    # ---------------- (e) setting-priced subjects ----------------
    sec = "e"
    others = [s for s in subs if s not in SETTING_CORE]
    for key, dct, pool, lab in [("raw", sx["raw"], subs, "raw coupling YAG5"),
                                ("std", sx["std"], compsubs, "composition-standardised coupling YAG5 (national band medians, over-corrects)"),
                                ("stdwp", sx["std_wp"], compsubs, "within-provider-standardised coupling YAG5"),
                                ("geo", sx["p_geo"], compsubs, "coupling | regional earnings level YAG5"),
                                ("slope", sc["slope"], csubs, "career slope")]:
        for grpname, grp in [("core", SETTING_CORE), ("broad", SETTING_BROAD)]:
            A = [s for s in grp if s in dct]
            Bo = [s for s in pool if s not in grp and s in dct]
            ga = group_mean(dct, A, f"set{key}{grpname}a"); gb = group_mean(dct, Bo, f"set{key}{grpname}b")
            dif = ga["bs"] - gb["bs"]
            p, nlab = label_perm({s: dct[s][0] for s in pool if s in dct}, A, pool)
            put(f"{lab}: setting-priced ({grpname}) mean", ", ".join(A), ga["est"], ga["lo"], ga["hi"], k=ga["k"], section=sec)
            put(f"{lab}: others mean (vs {grpname})", f"{len(Bo)} subjects", gb["est"], gb["lo"], gb["hi"], k=gb["k"], section=sec)
            put(f"{lab}: setting-priced ({grpname}) minus others",
                f"independent outer draws; exact subject-label permutation p ({nlab:,} labellings)",
                ga["est"] - gb["est"], *ci95(dif), p=p, k=ga["k"], section=sec)

    # ---------------- sensitivities ----------------
    sens = []
    variants = {
        "all matched HEIs (any degree, no exclusions)": dict(primary=(prov.provider_type == "HEI") & prov.G.notna()),
        "degree >= 20": dict(primary=prov.primary & (prov.deg >= 20)),
        "G from post-2000 hires": dict(primary=prov.primary, G="G_post2000"),
    }
    for vname, v in variants.items():
        pv = prov.copy()
        pv["primary"] = v["primary"]
        if "G" in v:
            pv["G"] = pv[v["G"]]
        assert not pv[pv.primary].ror.duplicated().any(), vname
        Tv, medv, _, _ = build_tables(leo, pv, GRADS)
        rv = analyse(Tv, medv, sens=True)
        _, sxv = rv["cx"]; _, sbv = rv["bands"]; _, scv = rv["cc"]
        sv = sorted(sxv["raw"]); cv = sorted(sxv["std"]); bv = sorted(sbv["within"]); kv = sorted(scv["slope"])
        rr = dict(variant=vname, n_providers=int(pv.primary.sum()),
                  raw=group_mean(sxv["raw"], sv, "v")["est"], k_raw=len(sv),
                  raw_c=group_mean(sxv["raw_c"], cv, "v")["est"], std=group_mean(sxv["std"], cv, "v")["est"],
                  std_wp=group_mean(sxv["std_wp"], cv, "v")["est"],
                  within=group_mean(sbv["within"], bv, "v")["est"], matched=group_mean(sbv["matched"], bv, "v")["est"],
                  k_band=len(bv), slope=group_mean(scv["slope"], kv, "vslope")["est"],
                  slope_lo=group_mean(scv["slope"], kv, "vslope")["lo"], slope_hi=group_mean(scv["slope"], kv, "vslope")["hi"],
                  k_slope=len(kv))
        sens.append(rr)
    SENS = pd.DataFrame(sens)

    # ---------------- write tables ----------------
    Rdf = pd.DataFrame(R)
    Rdf.to_csv(OUT_SUM, index=False, float_format="%.6g")
    per = pd.DataFrame({k: {s: v[0] for s, v in sx[k].items()} for k in ["raw", "raw_c", "std", "std_wp", "p_prog", "p_inst", "p_geo", "sel"]})
    per["n_mean"] = dfx.groupby("subject").n.mean()
    per["within_band"] = pd.Series({s: v[0] for s, v in sb["within"].items()})
    per["matched_band"] = pd.Series({s: v[0] for s, v in sb["matched"].items()})
    per["n_band_cells"] = dfb.groupby("subject").size()
    for y in ["y1", "y3", "y5", "slope"]:
        per[f"career_{y}"] = pd.Series({s: v[0] for s, v in sc[y].items()})
    per["career_slope_lo"] = pd.Series({s: ci95(v[1])[0] for s, v in sc["slope"].items()})
    per["career_slope_hi"] = pd.Series({s: ci95(v[1])[1] for s, v in sc["slope"].items()})
    per["career_slope_std"] = pd.Series({s: v[0] for s, v in ss_["slope"].items()})
    per["career_slope_std_wp"] = pd.Series({s: v[0] for s, v in sw_["slope"].items()})
    per = per.join(fs[["rho_F", "rho_G", "rho_FG", "rho_old_s40"]].add_prefix("s40_"), how="left")
    per["us_rho_field"] = us.reindex(per.index)
    per["us_rho_brand"] = usG.reindex(per.index)
    per["setting"] = np.where(per.index.isin(SETTING_CORE), "core", np.where(per.index.isin(SETTING_BROAD), "broad", ""))
    per.index.name = "subject"
    per.sort_values("raw").to_csv(OUT_SUBJ, float_format="%.4f")
    pd.concat([dfb.assign(analysis="band_yag5"), dfr.assign(analysis="retest_yag5"), res["bands_ext"][0].assign(analysis="band_ext_yag5"),
               res["bands_y1"][0].assign(analysis="band_yag1"), res["bands_y3"][0].assign(analysis="band_yag3"),
               res["sex"][0].assign(analysis="sex_yag5"), res["polar"][0].assign(analysis="polar_yag5")],
              ignore_index=True).to_csv(OUT_BAND, index=False, float_format="%.4f")
    pd.concat([dfc.assign(analysis="all_graduates"), dfs.assign(analysis="composition_standardised_national"),
               dfw.assign(analysis="composition_standardised_within_provider"),
               dfcb.assign(analysis="within_band"), dfl.assign(analysis="calendar")],
              ignore_index=True).to_csv(OUT_CAREER, index=False, float_format="%.4f")
    pcols = ["ukprn", "provider_name", "ror", "ror_name", "deg", "G", "G_post2000", "G_pct", "russell",
             "inst_top", "geo"]
    P[pcols].sort_values("G", ascending=False).to_csv(OUT_PRES, index=False, float_format="%.5f")
    SENS.to_csv(INTERIM / "uk_leo_sensitivity.csv", index=False, float_format="%.4f")
    pd.concat([GB["wp"][0].assign(fit="lambda_ols", lam=GB["wp"][1]),
               GB["wpw"][0].assign(fit="lambda_wls_count", lam=GB["wpw"][1]),
               GB["wpfe"][0].assign(fit="band_fe_ols", lam=np.nan)]).reset_index() \
        .to_csv(INTERIM / "uk_leo_band_gradients.csv", index=False, float_format="%.5f")
    XS.assign(us_rho_field=XS.subject.map(us), in_s40_16=XS.subject.isin(kk16)) \
        .sort_values(["cohort", "yag", "subject"]).to_csv(INTERIM / "uk_leo_xnat_sections.csv", index=False, float_format="%.4f")

    make_figure(P, per, res, Rdf)
    write_md(Rdf, per, P, prov, res, SENS, dens, rel, fs, kk16, {h["key"]: h for h in HET})
    print(Rdf.to_string())


# =============================================================================================
# figure
# =============================================================================================
C_BLUE, C_ORANGE, C_AQUA, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9892"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def _row(Rdf, q):
    r = Rdf[Rdf.quantity == q]
    assert len(r) == 1, q
    return r.iloc[0]


def _style(ax):
    ax.set_facecolor(SURF)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(GRID); ax.spines[sp].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, lw=1); ax.set_axisbelow(True)


def make_figure(P, per, res, Rdf):
    fig = plt.figure(figsize=(14, 10.5), facecolor=SURF)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1], hspace=0.55, wspace=0.55)
    # A: per-subject raw vs composition-standardised coupling
    ax = fig.add_subplot(gs[:, 0]); _style(ax)
    d = per.sort_values("raw")
    y = np.arange(len(d))
    for yi, (s, r) in zip(y, d.iterrows()):
        if np.isfinite(r["std"]):
            ax.plot([min(r["raw"], r["std"], r["std_wp"]), max(r["raw"], r["std"], r["std_wp"])], [yi, yi],
                    color=GRID, lw=2, zorder=1)
    ax.scatter(d["raw"], y, s=34, facecolor=SURF, edgecolor=C_GRAY, lw=1.6, zorder=3, label="all graduates (raw)")
    ax.scatter(d["std_wp"], y, s=38, color=C_BLUE, edgecolor=SURF, lw=1.5, zorder=5,
               label="standardised, within-provider band gradients")
    ax.scatter(d["std"], y, s=22, color=C_ORANGE, edgecolor=SURF, lw=1.0, zorder=4,
               label="standardised, national band medians (over-corrects)")
    lab = [f"{s} \u2020" if s in SETTING_CORE else s for s in d.index]
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=7.5, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("coupling: Spearman(academia-wide prestige G, LEO median earnings), YAG 5", fontsize=8.5, color=INK2)
    ax.set_title("A  Coupling by subject, before and after prior-attainment standardisation\n"
                 "mean of cohorts 2013/14-2016/17; \u2020 = national pay scale (nursing, medicine, teaching)",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.045), fontsize=8, frameon=False, ncol=2)
    ax.set_ylim(-0.8, len(d) - 0.2)

    # B: specification ladder
    ax = fig.add_subplot(gs[0, 1]); _style(ax)
    rows = [("raw coupling, composition sample", "all graduates", C_GRAY),
            ("composition-standardised earnings (within-provider band gradients)", "standardised, within-provider gradients", C_BLUE),
            ("composition-standardised earnings (national subject x band medians)", "standardised, national medians (over-corrects)", C_ORANGE),
            ("partial | program share of graduates with >= 360 points", "partial | programme share >= 360 pts", C_BLUE),
            ("partial | institution-wide share with >= 360 points", "partial | institution share >= 360 pts", C_BLUE),
            ("all-graduate coupling, same providers", "all graduates, band-cell providers", C_GRAY),
            ("within-band coupling (bands 1-5)", "within prior-attainment band", C_BLUE)]
    yy = [6.5, 5.5, 4.5, 3.5, 2.5, 1, 0]
    for (q, lab_, col), yv in zip(rows, yy):
        r = _row(Rdf, q)
        ax.plot([r.lo, r.hi], [yv, yv], color=col, lw=2, solid_capstyle="round")
        ax.scatter([r.est], [yv], s=40, color=col, edgecolor=SURF, lw=1.5, zorder=3)
        ax.annotate(f"{r.est:+.2f}", (r.hi, yv), xytext=(4, -3), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_yticks(yy); ax.set_yticklabels([r[1] for r in rows], fontsize=7.5, color=INK)
    ax.axhline(1.75, color=GRID, lw=1)
    ax.set_xlim(0, 0.7)
    ax.set_title("B  Mean coupling (95% CI): composition sample (top), band cells (bottom)",
                 fontsize=9.5, color=INK, loc="left")

    # C: career trajectories
    ax = fig.add_subplot(gs[1, 1]); _style(ax); ax.grid(axis="y", color=GRID, lw=1)
    series = [("coupling at {}", "career slope of rho(G, earnings) per year", "all graduates", C_BLUE),
              ("within-provider-standardised coupling at {}", "career slope, within-provider-standardised",
               "standardised (within provider)", C_AQUA),
              ("within-band coupling at {}", "career slope, within prior-attainment band", "within band", C_ORANGE)]
    for qf, qs, lab_, col in series:
        est = [_row(Rdf, qf.format(f"YAG{y}")).est for y in YAGS]
        lo = [_row(Rdf, qf.format(f"YAG{y}")).lo for y in YAGS]
        hi = [_row(Rdf, qf.format(f"YAG{y}")).hi for y in YAGS]
        ax.fill_between(YAGS, lo, hi, color=col, alpha=0.10, lw=0)
        ax.plot(YAGS, est, color=col, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=lab_)
        sl = _row(Rdf, qs)
        ax.annotate(f"{lab_}: {sl.est:+.3f}/yr", (5, est[-1]), xytext=(6, -3), textcoords="offset points",
                    fontsize=7.5, color=INK2)
    ax.set_xticks(YAGS); ax.set_xticklabels(["YAG 1", "YAG 3", "YAG 5"], fontsize=8)
    ax.set_xlim(0.7, 7.2); ax.set_ylim(0, 0.6)
    ax.set_ylabel("mean coupling", fontsize=8.5, color=INK2)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False, ncol=1)
    ax.set_title("C  Fixed cohorts 2013/14-2016/17 at YAG 1/3/5 (balanced providers; 95% CI)",
                 fontsize=9.5, color=INK, loc="left")

    # D: band by band
    ax = fig.add_subplot(gs[2, 1]); _style(ax)
    db = res["bands"][0]
    bb = [b for b in PA_BANDS if (db.grp == b).sum() >= 10][::-1]      # highest band at the top
    for i, b in enumerate(bb):
        x = db[db.grp == b]
        ax.plot([x.matched.mean(), x.within.mean()], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter([x.matched.mean()], [i], s=34, facecolor=SURF, edgecolor=C_GRAY, lw=1.6, zorder=3,
                   label="all graduates, same providers" if i == len(bb) - 1 else None)
        ax.scatter([x.within.mean()], [i], s=38, color=C_BLUE, edgecolor=SURF, lw=1.5, zorder=4,
                   label="within band" if i == len(bb) - 1 else None)
        ax.annotate(f"{len(x)} cells", (max(x.matched.mean(), x.within.mean()), i), xytext=(8, -3),
                    textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(bb))); ax.set_yticklabels([f"{PA_LABEL[b]} points" if b != "PA2" else "360 points (e.g. 3 As)"
                                                         for b in bb], fontsize=7.5, color=INK)
    ax.set_ylim(-0.6, len(bb) - 0.4)
    ax.set_xlim(0, 0.55)
    ax.set_xlabel("mean coupling over subject x cohort cells (YAG 5)", fontsize=8.5, color=INK2)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax.set_title("D  Same-band graduates compared across providers, by UCAS-tariff band",
                 fontsize=9.5, color=INK, loc="left")
    fig.savefig(OUT_FIG, dpi=150, facecolor=SURF, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


# =============================================================================================
# write-up
# =============================================================================================
def _md5(path: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_md(Rdf, per, P, prov, res, SENS, dens, rel, fs, kk16, het):
    def r(q):
        return _row(Rdf, q)

    def f(v, d=3, sign=True):
        if v is None or not np.isfinite(v):
            return "—"
        return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"

    def c(q, d=3):
        x = r(q)
        return f"{f(x.est, d)} [{f(x.lo, d)}, {f(x.hi, d)}]"

    def pct(q):
        x = r(q)
        return f"{100 * x.est:.0f}% [{100 * x.lo:.0f}%, {100 * x.hi:.0f}%]"

    def excl0(q):
        x = r(q)
        return (x.lo > 0) or (x.hi < 0)

    L = []
    A = L.append
    A("# UK LEO — does prestige–pay coupling survive a prior-attainment control, and does it rise over careers?\n")
    A("Script: `scripts/62_uk_leo.py` (seed 62; every number below is written by the script from its own "
      "computations; re-run byte-identical, see Method). Public data only: DfE LEO provider-level graduate "
      "outcomes, the ORCID-derived UK PhD→faculty hiring network (edges cached by `scripts/40`), the ROR dump. "
      "Coupling = Spearman, across providers within one CAH2 subject, of academia-wide UK prestige G and LEO "
      "median earnings. Descriptive, not causal. Tables: `data/interim/uk_leo_summary.csv` (all key numbers), "
      "`uk_leo_subject_coupling.csv`, `uk_leo_band_cells.csv`, `uk_leo_career_cells.csv`, `uk_leo_prestige.csv`, "
      "`uk_leo_sensitivity.csv`; figure `outputs/figures/uk_leo.png`.\n")

    # ------------------------------------------------------------------ answer
    A("## Answer\n")
    A("**Key question: does UK prestige–pay coupling survive a direct prior-attainment control, and does it rise "
      "over careers?** Mostly yes on both counts, with one qualification that matters for the working headline: "
      "what survives the control tracks the institution's intake selectivity, and academia-wide research prestige "
      "adds little beyond it.\n")
    sh_wb = r('share surviving: within-band / matched').est
    sh_att = r('share surviving, attenuation-adjusted').est
    sh_wp = r('share surviving: std_wp / raw_c').est
    sh_fe = r('share surviving: fe_std / fe_raw').est
    sh_nat = r('share surviving: std / raw_c').est
    lo_sh, hi_sh = min(sh_wb, sh_att, sh_wp, sh_fe), max(sh_wb, sh_att, sh_wp, sh_fe)
    cl_lo, cl_hi = min(sh_att, sh_wp, sh_fe), max(sh_att, sh_wp, sh_fe)
    A(f"1. **Most of the coupling survives the control: {100 * lo_sh:.0f}–{100 * hi_sh:.0f}% depending on the method, "
      f"with the noise-corrected and within-provider estimates at {100 * cl_lo:.0f}–{100 * cl_hi:.0f}%.** Mean coupling "
      f"at 5 years after graduation (YAG5) is {c('coupling rho(G, earnings), YAG5')} across "
      f"{int(r('coupling rho(G, earnings), YAG5').k)} CAH2 subjects. The control is applied in three ways:\n")
    A(f"   - *Same-band graduates across providers.* Within a prior-attainment band the coupling is "
      f"{c('within-band coupling (bands 1-5)')}, against {c('all-graduate coupling, same providers')} for all "
      f"graduates of the same providers: {pct('share surviving: within-band / matched')} survives. Band medians "
      f"rest on fewer graduates and are noisier (adjacent-cohort test–retest {f(r('test-retest reliability, band medians').est, 2, False)} "
      f"vs {f(r('test-retest reliability, all-graduate medians').est, 2, False)} for all-graduate medians on the same "
      f"providers), which pulls the within-band coupling toward zero; correcting for that gives "
      f"{pct('share surviving, attenuation-adjusted')} (unadjusted on the same {int(r('share surviving, attenuation-adjusted').k)} "
      f"subjects: {100 * r('share surviving, unadjusted, same subjects as the attenuation check').est:.0f}%).")
    A(f"   - *Standardising each provider's earnings for its band mix, with within-provider band gradients.* "
      f"{c('composition-standardised earnings (within-provider band gradients)')} vs "
      f"{c('raw coupling, composition sample')} on the same cells: {pct('share surviving: std_wp / raw_c')} survives "
      f"(gradient = the national band gradient rescaled to its within-provider size, λ = "
      f"{f(r('lambda: within-provider / national band gradient (OLS)').est, 2, False)} at the median cell). With a "
      f"free band gradient (provider FE + band FE), which can be estimated only where every band of a provider's mix "
      f"is released somewhere within provider, {pct('share surviving: fe_std / fe_raw')} survives on that subsample "
      f"(raw {f(r('free band-FE subsample: raw coupling').est)}; national-median version on the same providers "
      f"{pct('share surviving: fe_nat / fe_raw')}).")
    A(f"   - *Standardising with national subject × band medians* gives "
      f"{c('composition-standardised earnings (national subject x band medians)')}, i.e. {pct('share surviving: std / raw_c')}. "
      f"This over-corrects. National band medians mix the effect of prior attainment with institution effects, "
      f"because high-band graduates are concentrated at high-paying institutions, so subtracting them removes part "
      f"of the institution effect the control is meant to leave in. Within the same provider the log earnings gap "
      f"between band 2 (360 points) and band 4 (240–299) is {f(r('log earnings gap PA2 - PA4: within provider (provider FE + band FE)').est)} "
      f"(median over {int(r('log earnings gap PA2 - PA4: within provider (provider FE + band FE)').k)} subject × cohort "
      f"cells at YAG5), against {f(r('log earnings gap PA2 - PA4: national band medians').est)} in the national "
      f"medians; band 3 vs band 5: {f(r('log earnings gap PA3 - PA5: within provider (provider FE + band FE)').est)} vs "
      f"{f(r('log earnings gap PA3 - PA5: national band medians').est)}. So {100 * sh_nat:.0f}% is a conservative "
      f"lower bound, not an estimate.\n")
    A(f"   So {100 * lo_sh:.0f}% (within band, pulled down by band-median noise) to {100 * hi_sh:.0f}% of the coupling "
      f"survives a direct control for students' own prior attainment, and {100 * sh_nat:.0f}% is a conservative "
      f"floor. None of these is a clean bound. The bands are coarse, so finer sorting within a band is not "
      f"controlled, which biases every figure up. Within-provider gradients would understate the attainment effect "
      f"if providers admit lower-band students for strengths the bands do not record (Caveats).\n")
    A(f"2. **What survives is mostly institutional selectivity; academia-wide research prestige adds little beyond "
      f"it.** The institution's intake selectivity (its share of graduates with ≥360 UCAS points, all subjects) "
      f"couples with all-graduate pay more strongly than G does on the same cells "
      f"({c('coupling with institution selectivity instead of G (all graduates)')} vs "
      f"{c('raw coupling, composition sample')}). Among same-band graduates, selectivity predicts pay "
      f"({c('within-band: Spearman(institution top-band share, band earnings)')}); G adds "
      f"{c('within-band: partial rho(G, band earnings | institution top-band share)')} beyond it, while selectivity "
      f"keeps {c('within-band: partial rho(institution top-band share, band earnings | G)')} beyond G. On the "
      f"all-graduate medians of the same providers G keeps a little more "
      f"({c('matched all-graduate: partial rho(G, earnings | institution top-band share)')}). On the full composition "
      f"sample, partialling institution selectivity out of the coupling leaves "
      f"{c('partial | institution-wide share with >= 360 points')} ({pct('share surviving: p_inst / raw_c')} of raw): "
      f"small, but its CI excludes zero. The reduction is comparable to the US result that SAT/admit-rate/Pell/control/state controls cut mean coupling "
      f"from +0.43 to +0.14 (scripts/55; different controls). "
      f"G and institution selectivity correlate {f(r('Spearman(G, institution top-band share)').est, 2)} across "
      f"the {int(r('Spearman(G, institution top-band share)').k)} providers. Within band, the selectivity measure "
      f"can also pick up residual within-band selection (a 300–359-point student at a highly selective institution "
      f"is not the same as one at an unselective institution), so this does not separate the institution's status "
      f"or value added from finer student sorting.\n")
    def hp(key):
        x = het[key]
        return "< 0.0001" if x["p"] < 1e-4 else f"{x['p']:.4f}"

    def ht(key):
        x = het[key]
        return f"τ = {x['tau']:.2f} (I² {x['I2']:.2f}, Q-test p {hp(key)})"
    xq = "cross-subject Spearman of subject couplings: "
    A(f"3. **How much status pays still varies across subjects after the prior-attainment control, but the variation "
      f"is nearly the same thing as how much each subject pays for institutional selectivity.** Across the "
      f"{het['raw_c']['k']} subjects of the composition sample, the SD of the subjects' true couplings beyond "
      f"provider-sampling noise (DerSimonian–Laird τ) is {het['raw_c']['tau']:.2f} for all graduates (I² "
      f"{het['raw_c']['I2']:.2f}) and {het['std_wp']['tau']:.2f} after within-provider standardisation (I² "
      f"{het['std_wp']['I2']:.2f}, Q-test p {hp('std_wp')}), and the subject ordering barely moves (Spearman "
      f"{c(xq + 'all graduates vs standardised within provider', 2)}). It is not only the pay-scale subjects: "
      f"without them, {ht('std_wp_ns')}. Within prior-attainment bands, {ht('within')}, against "
      f"τ = {het['matched']['tau']:.2f} for all graduates of the same providers (band cells cover fewer, more "
      f"similar providers). After partialling out "
      f"institution selectivity, less variation remains and it is ordered differently: {ht('p_inst')}; Spearman "
      f"with the all-graduate ordering {c(xq + 'all graduates vs partial | institution selectivity', 2)}. Subjects "
      f"whose pay couples strongly with G are the subjects whose pay couples strongly with institution selectivity "
      f"(Spearman across subjects {c(xq + 'coupling with G vs coupling with institution selectivity', 2)}); the US "
      f"analogue is the true-score r of +0.94 between field coupling and field-specific pricing of selectivity "
      f"(scripts/55). So, as in the US, the subject ordering is real and survives the attainment control, and "
      f"public data cannot tell it apart from subject-specific pricing of institutional selectivity.\n")
    sl = r("career slope of rho(G, earnings) per year")
    A(f"4. **Coupling rises over careers, at about the US rate.** On fixed cohorts 2013/14–2016/17 observed at "
      f"YAG 1, 3 and 5 (balanced providers), coupling rises {c('career slope of rho(G, earnings) per year')}/yr "
      f"(YAG-label permutation p = {sl.p:.4f}; {int(sl.k)} subjects), from {f(r('coupling at YAG1').est)} at YAG1 "
      f"to {f(r('coupling at YAG5').est)} at YAG5, positive in {int(r('cohorts with positive mean slope').est)}/"
      f"{int(r('cohorts with positive mean slope').k)} cohorts. The US fixed-cohort figure is +0.031/yr over years "
      f"1–10 and +0.034/yr over years 1–5 (scripts/52; field prestige × PSEO), so the UK rise over years 1–5 is "
      f"similar to or somewhat below the US one. The rise survives the prior-attainment controls: standardised with within-provider band "
      f"gradients {c('career slope, within-provider-standardised')}/yr, within band "
      f"{c('career slope, within prior-attainment band')}/yr (attenuated by the same band-median noise); the "
      f"national-median standardisation, which over-corrects, gives {c('career slope, composition-standardised')}/yr. "
      f"It is not only a YAG1 effect (further study depresses some YAG1 "
      f"medians): YAG1→3 {f(r('segment YAG1->3 per year').est)}/yr, YAG3→5 {c('segment YAG3->5 per year')}/yr. A "
      f"coverage control lowers it to {c('slope of partial coupling | coverage')}/yr. The calendar-matched contrast "
      f"(cohort c at YAG5 vs cohort c+4 at YAG1, the same tax year) is "
      f"{c('calendar-matched (c@5 - (c+4)@1)/4, same tax year')}/yr, the within-cohort change "
      f"{c('within-cohort (c@5 - c@1)/4')}/yr, and the cross-cohort drift at YAG1 "
      f"{c('cross-cohort drift at YAG1 ((c+4)@1 - c@1)/4')}/yr. The calendar-matched contrast removes calendar-time "
      f"change and the within-cohort contrast removes cohort differences; both are positive and of similar size, so "
      f"neither a pure calendar effect nor a pure cohort effect accounts for the rise. In the US the drift was "
      f"+0.017/yr and the career-time part could only be bounded to [+0.014, +0.032]/yr (scripts/52); under the same "
      f"sign assumption (calendar and cohort effects both non-negative) the near-zero UK drift implies both are "
      f"about zero, putting the UK career-time component at roughly the within-cohort and calendar-matched values. "
      f"Without that assumption, offsetting calendar and cohort effects cannot be ruled out (Caveats).\n")
    A(f"5. **Where pay is set by national scales, coupling is low; for medicine and nursing it is also flat over "
      f"careers.** Nursing and midwifery, Medicine and dentistry, and Education and teaching average "
      f"{c('raw coupling YAG5: setting-priced (core) mean')} against {c('raw coupling YAG5: others mean (vs core)')} "
      f"for the other subjects (difference {c('raw coupling YAG5: setting-priced (core) minus others')}, exact "
      f"subject-label permutation p = {r('raw coupling YAG5: setting-priced (core) minus others').p:.4f}); after "
      f"prior-attainment standardisation with within-provider gradients "
      f"{f(r('within-provider-standardised coupling YAG5: setting-priced (core) mean').est)} vs "
      f"{f(r('within-provider-standardised coupling YAG5: others mean (vs core)').est)} (p = "
      f"{r('within-provider-standardised coupling YAG5: setting-priced (core) minus others').p:.4f}; with national "
      f"band medians, which over-correct, "
      f"{f(r('composition-standardised coupling YAG5 (national band medians, over-corrects): setting-priced (core) mean').est)} "
      f"vs {f(r('composition-standardised coupling YAG5 (national band medians, over-corrects): others mean (vs core)').est)}, "
      f"p = {r('composition-standardised coupling YAG5 (national band medians, over-corrects): setting-priced (core) minus others').p:.4f}); with a regional "
      f"earnings-level control {f(r('coupling | regional earnings level YAG5: setting-priced (core) mean').est)} vs "
      f"{f(r('coupling | regional earnings level YAG5: others mean (vs core)').est)} (p = "
      f"{r('coupling | regional earnings level YAG5: setting-priced (core) minus others').p:.4f}). Their career slope "
      f"averages {c('career slope: setting-priced (core) mean')}/yr vs {c('career slope: others mean (vs core)')}/yr "
      f"(difference {c('career slope: setting-priced (core) minus others')}, p = "
      f"{r('career slope: setting-priced (core) minus others').p:.3f}; only three subjects, so this difference is not "
      f"established). Per subject (Appendix A): medicine {f(per.loc['Medicine and dentistry', 'raw'])} "
      f"(slope {f(per.loc['Medicine and dentistry', 'career_slope'])}/yr), nursing "
      f"{f(per.loc['Nursing and midwifery', 'raw'])} (slope {f(per.loc['Nursing and midwifery', 'career_slope'])}/yr), "
      f"teaching {f(per.loc['Education and teaching', 'raw'])} (standardised with within-provider gradients "
      f"{f(per.loc['Education and teaching', 'std_wp'])}; with national medians, over-corrected, "
      f"{f(per.loc['Education and teaching', 'std'])}). "
      f"Teaching differs on the trajectory: its coupling is {f(per.loc['Education and teaching', 'career_y1'])} at YAG1 "
      f"and {f(per.loc['Education and teaching', 'career_y5'])} at YAG5 (slope "
      f"{f(per.loc['Education and teaching', 'career_slope'])}/yr "
      f"[{f(per.loc['Education and teaching', 'career_slope_lo'])}, {f(per.loc['Education and teaching', 'career_slope_hi'])}]). "
      f"The subject-level classification is imperfect: Allied health, many of whose graduates are paid on NHS "
      f"Agenda for Change bands, couples at {f(per.loc['Allied health', 'raw'])}, close to the other subjects; adding it "
      f"to the group shrinks the difference to {c('raw coupling YAG5: setting-priced (broad) minus others')} (p = "
      f"{r('raw coupling YAG5: setting-priced (broad) minus others').p:.4f}).\n")
    q16 = "US-UK cross-subject Spearman (16 subjects): UK "
    xF, xGf, xGa = (r(q16 + "F on F-providers vs US field prestige"), r(q16 + "G on F-providers vs US field prestige"),
                    r(q16 + "G on all primary providers vs US field prestige"))
    xs40 = r(q16 + "F (field-tagged), scripts/40 as published vs US field prestige")
    wF, wGf, wGa = [r(q16 + s + " vs US field prestige [scripts/40 window]") for s in
                    ["F on F-providers", "G on F-providers", "G on all primary providers"]]
    rng16 = r("US-UK Spearman (16 subjects), G on all primary providers: single cohort x YAG cross-sections")
    rng21 = r(f"US-UK Spearman ({int(r('US-UK cross-subject Spearman: UK G vs US field prestige, all mapped subjects').k)} "
              f"mapped subjects), G on all primary providers: single cohort x YAG cross-sections")
    loo = r("US-UK 16-subject Spearman, G on all primary providers: leave-one-subject-out")
    nonur = r("US-UK Spearman, G on all primary providers, without Nursing and midwifery")
    ax_ = r("US-UK 16-subject Spearman, axis effect: G minus F, same F-providers and cells")
    ps_ = r("US-UK 16-subject Spearman, provider-sample effect: G on all primary providers minus G on F-providers")
    se_ = r("subject-bootstrap SE of the 16-subject US-UK Spearman, G on all primary providers")
    all21 = r("US-UK cross-subject Spearman: UK G vs US field prestige, all mapped subjects")
    xlo = min(xs40.est, xF.est, xGf.est, xGa.est, wF.est, wGf.est, wGa.est, rng16.lo)
    xhi = max(xGa.est, rng16.hi)
    A(f"6. **Academia-wide prestige G couples with pay more strongly than scripts/40's field-tagged axis within "
      f"subjects. Across subjects, the US–UK agreement in how strongly subjects couple does not come from the "
      f"prestige axis, and it is only suggestive.** "
      f"G (one SpringRank over all {int(r('hiring edges').est):,} GB→GB hiring edges) places the "
      f"Russell Group at a mean percentile of {f(r('Russell Group mean G percentile').est, 2, False)} vs "
      f"{f(r('non-RG mean G percentile').est, 2, False)} (AUC {f(r('AUC(RG vs others)').est, 2, False)}; "
      f"{int(r('RG members in top quartile of G').est)}/{int(r('RG members in top quartile of G').k)} members in the "
      f"top quartile), with split-half reliability {f(r('split-half reliability of G, Spearman-Brown').est, 2, False)} "
      f"(Spearman–Brown). On the {len(kk16)} subjects where scripts/40's field-tagged per-subject axis F is usable, "
      f"coupling with G is higher than with F in {int(r('subjects with rho_G > rho_F').est)}/{len(kk16)} on the same "
      f"providers and cells (mean {f(r('mean rho with academia-wide G').est)} vs "
      f"{f(r('mean rho with field-tagged F (scripts/40 axis)').est)}). The cross-subject comparison with the US is a "
      f"different matter. On the same {len(kk16)} subjects, the Spearman correlation between the UK and US subject "
      f"orderings is {f(xs40.est, 2)} (p = {xs40.p:.2f}) in scripts/40 (F, tax year 2022/23, its name-matched "
      f"providers). Varying one thing at a time:\n")
    A(f"   - *Axis, same providers and cells (cohorts 2013–16 at YAG5, providers with F):* F {f(xF.est, 2)} "
      f"[{f(xF.lo, 2)}, {f(xF.hi, 2)}] (p = {xF.p:.2f}), G {f(xGf.est, 2)} [{f(xGf.lo, 2)}, {f(xGf.hi, 2)}] "
      f"(p = {xGf.p:.2f}). Swapping F for G lowers the agreement ({f(ax_.est, 2)} [{f(ax_.lo, 2)}, {f(ax_.hi, 2)}]).")
    A(f"   - *Provider sample:* G on all {len(P)} primary providers gives {f(xGa.est, 2)} [{f(xGa.lo, 2)}, "
      f"{f(xGa.hi, 2)}] (p = {xGa.p:.3f}), {f(ps_.est, 2)} [{f(ps_.lo, 2)}, {f(ps_.hi, 2)}] above G on the "
      f"F-providers. This, not the axis, is where the gain comes from.")
    A(f"   - *Earnings window:* in scripts/40's own window (cohort 2016/17 at YAG5 = tax year 2022/23) the values are "
      f"F {f(wF.est, 2)}, G on F-providers {f(wGf.est, 2)}, and G on all providers {f(wGa.est, 2)} (p = {wGa.p:.2f}). "
      f"Across the {rng16.spec.split(' ')[0]} single cohort × YAG cross-sections, G on all providers ranges from "
      f"{f(rng16.lo, 2)} to {f(rng16.hi, 2)} (median {f(rng16.est, 2)}).")
    A(f"   - *Fragility of the {f(xGa.est, 2)}:* leaving out one subject gives {f(loo.lo, 2)} to {f(loo.hi, 2)}; "
      f"dropping Nursing gives {f(nonur.est, 2)} (p = {nonur.p:.3f}); the subject-bootstrap SE is "
      f"{f(se_.est, 2, False)}. On all {int(all21.k)} subjects with a US counterpart it is {f(all21.est, 2)} "
      f"(p = {all21.p:.3f}; single cross-sections {f(rng21.lo, 2)} to {f(rng21.hi, 2)}).\n")
    A(f"   So the US–UK subject-ordering correlation is somewhere between about {f(xlo, 1)} and {f(xhi, 2)}, depending "
      f"on the provider sample and earnings window (n = {len(kk16)} subjects). It is not driven by the prestige "
      f"axis, and it should be read as suggestive. Setting scripts/40's {f(xs40.est, 2)} beside the "
      f"{f(xGa.est, 2)} here would wrongly credit the prestige axis, because the two numbers differ in provider "
      f"sample and earnings window as well as axis. The within-subject result (G > F in 16/16, same "
      f"providers) stands. The field-tagged axis is much noisier (usable field networks of "
      f"{int(dens[dens.usable].edges.min())}–{int(dens[dens.usable].edges.max())} edges, vs "
      f"{int(r('hiring edges').est):,} for G), so that result is not evidence that field prestige is worse than brand "
      f"in truth. It shows only that field prestige is no better, as in the US (scripts/56).\n")
    A("**Reading against the working headline.** The UK data are consistent with the three parts of the headline, "
      "with one sharpening and one exception. Pay tracks institutional status. In the UK the status measure that "
      "carries it is intake selectivity; academia-wide research prestige adds little once selectivity is held "
      "fixed. How much status pays varies strongly by subject, also after the attainment control and also without "
      "the pay-scale subjects, and grows with years since graduation. As in the US, the subject ordering cannot be "
      "separated from subject-specific pricing of selectivity. It is small "
      "where pay follows national scales for medicine and nursing. Teaching is low after standardisation but rises "
      "over the first five years. Allied health, also largely on national pay bands, is the exception and couples "
      "like the other subjects. What the UK adds is a direct student-level check the US data lack: most of the "
      f"coupling ({100 * lo_sh:.0f}–{100 * hi_sh:.0f}%) is not explained by graduates' own prior attainment. It "
      "remains among graduates in the same tariff band and after standardising for band mix with within-provider "
      "gradients. Public aggregates cannot say whether that remainder is the institution's value added, peer "
      "effects, employer screening on the institution's name, or sorting within a tariff band. The subject "
      "orderings of the US and UK coupling agree only loosely, and how well they agree depends on sample choices "
      "rather than on the prestige axis.\n")

    # ------------------------------------------------------------------ corrections
    A("## Corrections to an earlier draft\n")
    A("An earlier draft of this file (an interrupted previous run of this task) made two claims that do not hold:\n")
    A(f"- That with G the UK subject ordering lines up with the US one, rising from +0.18 (scripts/40) to +0.58 on the "
      f"same 16 subjects. Only the subjects were the same. Like for like, F gives {f(xF.est, 2)} and G {f(xGf.est, 2)} "
      f"on the same providers and cells; G reaches {f(xGa.est, 2)} only on all providers ({f(wGa.est, 2)} in "
      f"scripts/40's window). Answer 6 gives the decomposition.")
    A(f"- That 62–87% of the coupling survives the prior-attainment control. The {100 * sh_nat:.0f}% came from the "
      f"national-median standardisation, which over-corrects (answer 1). The range is "
      f"{100 * lo_sh:.0f}–{100 * hi_sh:.0f}%, with {100 * sh_nat:.0f}% as a conservative floor.")
    A("")

    # ------------------------------------------------------------------ key numbers
    A("## Key numbers\n")
    A("CI = 95% two-stage bootstrap (subjects resampled; providers resampled within subject, one draw shared by all "
      f"of a subject's cells), {NBOOT} replicates, unless the row says otherwise. p for the career slope = YAG-label "
      f"permutation ({NPERM}); p for setting-priced contrasts = exact subject-label permutation (all labellings); p for "
      "US–UK correlations = scipy Spearman p (n = k). k = subjects entering the statistic. Section letters follow "
      "the task: (a) prestige, (b) coupling, (c) prior attainment, (d) career time, (e) setting-priced subjects.\n")
    A("| sec | quantity | sample / spec | estimate | 95% CI | p | k |")
    A("|---|---|---|---|---|---|---|")
    for _, x in Rdf.iterrows():
        est = x.est
        if x.quantity.startswith("median first-faculty-job year"):
            es = f"{est:.0f}"
        elif "providers per cell" in x.quantity:
            es = f"{est:.1f}"
        elif abs(est) >= 100:
            es = f"{est:,.0f}"
        elif float(est).is_integer() and ("RG members" in x.quantity or "subjects with" in x.quantity or "cohorts with" in x.quantity):
            es = f"{est:.0f}"
        else:
            es = f(est)
        cis = "—" if not np.isfinite(x.lo) else (f"[{f(x.lo)}, {f(x.hi)}]" if np.isfinite(x.hi) else f"5th pct {f(x.lo)}")
        if x.quantity.startswith("band PA") and np.isfinite(x.lo) and not np.isfinite(x.hi):
            cis = f"matched {f(x.lo)}"
        if "range in CI column" in x.spec:
            cis = f"range [{f(x.lo)}, {f(x.hi)}]"
        if "CI column = IQR" in x.spec:
            cis = f"IQR [{f(x.lo)}, {f(x.hi)}]"
        if "CI column = FE subsample" in x.spec:
            cis = f"FE subsample {x.lo:.1f}"
        ps = "—" if not np.isfinite(x.p) else ("<0.0001" if x.p < 1e-4 else f"{x.p:.4f}")
        ks = "—" if not np.isfinite(x.k) else f"{x.k:.0f}"
        qq, sp = x.quantity.replace("|", "\\|"), x.spec.replace("|", "\\|")      # literal pipes break the table
        A(f"| {x.section} | {qq} | {sp} | {es} | {cis} | {ps} | {ks} |")
    A("\nBand rows (`band PAx: within / matched`) give the mean within-band coupling in *estimate* and the matched "
      "all-graduate coupling in the CI column (cell means, no CI). Rows `edge-bootstrap rank stability` give the "
      "median and 5th percentile; `split-half reliability` gives the median and the 5th–95th percentile range. "
      "`range` = min and max (not a CI); `IQR` = interquartile range over cells. Rows marked *national band "
      "medians (over-corrects)* use the national-median standardisation, which removes part of the institution "
      "effect (answer 1); the *within-provider* rows are the corrected version. US–UK rows marked *[scripts/40 "
      "window]* use cohort 2016/17 at YAG5 only.\n")
    A("**Sensitivity (point estimates; career slope with CI):**\n")
    A("| provider sample / prestige | providers | raw coupling (k) | raw, composition sample | standardised, within-provider | standardised, national (over-corrects) | within-band | matched | career slope [95% CI] (k) |")
    A("|---|---|---|---|---|---|---|---|---|")
    base = dict(variant=f"primary (matched HEIs, degree >= {DEGMIN}, 3 exclusions)", n_providers=len(P),
                raw=r("coupling rho(G, earnings), YAG5").est, k_raw=r("coupling rho(G, earnings), YAG5").k,
                raw_c=r("raw coupling, composition sample").est,
                std=r("composition-standardised earnings (national subject x band medians)").est,
                std_wp=r("composition-standardised earnings (within-provider band gradients)").est,
                within=r("within-band coupling (bands 1-5)").est, matched=r("all-graduate coupling, same providers").est,
                slope=sl.est, slope_lo=sl.lo, slope_hi=sl.hi, k_slope=sl.k)
    for _, v in pd.concat([pd.DataFrame([base]), SENS], ignore_index=True).iterrows():
        A(f"| {v.variant} | {v.n_providers:.0f} | {f(v.raw)} ({v.k_raw:.0f}) | {f(v.raw_c)} | {f(v.std_wp)} | {f(v['std'])} | "
          f"{f(v.within)} | {f(v.matched)} | {f(v.slope)} [{f(v.slope_lo)}, {f(v.slope_hi)}] ({v.k_slope:.0f}) |")
    A("")

    # ------------------------------------------------------------------ method
    top = P.sort_values("G", ascending=False).head(12)
    A("## Method\n")
    A(f"- **Prestige G (task a).** SpringRank (`src/ar_pipeline/springrank.py`, alpha = 1e-3) on all GB→GB "
      f"PhD→faculty edges from the ORCID shards (scripts/40 filter: doctoral education role → faculty employment "
      f"role, both organisations in GB with a ROR id; one edge per person, first faculty job), nodes keyed by ROR "
      f"id, one unit of weight per person, self-hires dropped. {int(r('hiring edges').est):,} edges, "
      f"{int(r('hiring-network nodes').est)} nodes. No field tagging: this is the academia-wide variant. Higher G = "
      f"the institution's PhDs are hired as faculty by lower-ranked institutions more than the reverse "
      f"(SpringRank's direction; no sign flip was needed). Top of the ranking among primary providers: "
      + ", ".join(f"{n} ({g:+.2f})" for n, g in zip(top.provider_name, top.G)) + ".")
    A(f"- **Provider matching.** LEO providers (UKPRN) are matched to hiring-network ROR ids through every ROR name "
      f"variant (label, alias, display name; normalised as in `src/crosswalks/institutions.py`, with a 'University "
      f"of X' ↔ 'X University' swap), plus {len(ROR_ALIAS)} hand-checked aliases (renamed or HESA-named providers, "
      f"e.g. Imperial's legal name, UCLan → University of Lancashire; City mapped to the pre-merger 'City, "
      f"University of London' record). {int(r('HEIs matched to a hiring-network node').est)} of "
      f"{int(r('HEIs in the LEO file').est)} LEO HEIs match. **Primary sample**: matched HEIs with network degree ≥ "
      f"{DEGMIN} (in+out edges), excluding the University of London umbrella node, the Open University and "
      f"Birkbeck (predominantly part-time, mature students) → {len(P)} providers. Further-education colleges and "
      f"alternative providers are excluded (few are in the hiring network). Sensitivity rows vary the degree "
      f"threshold, drop the exclusions, and use G from post-2000 hires only (Spearman with G "
      f"{f(r('Spearman(G, G from post-2000 hires only)').est, 3)}).")
    A(f"- **Reliability of G.** Edge (person) bootstrap, B = {B_EDGE}: median Spearman with the full-sample G over "
      f"the primary providers {f(rel['boot_median'], 3)} (5th pct {f(rel['boot_p05'], 3)}). Split-half: persons "
      f"split at random into halves {N_SPLIT} times, SpringRank on each half, median Spearman "
      f"{f(rel['split_half_median'], 3)}, Spearman–Brown {f(rel['split_half_sb'], 3)}.")
    A("- **Earnings (LEO).** `provider_data_20250716.csv`: provider × CAH2 subject × academic-year cohort × YAG "
      "(1/3/5; tax years 2015/16–2022/23; no YAG 10 in this file) median earnings as published, with rows for all "
      "graduates and for each prior-attainment band, sex and POLAR4 quintile (marginal breakdowns only; no "
      "cross-tabulations). Suppressed cells ('c', 'x', 'low') are missing. Nominal pounds; all statistics are within-cell ranks, so deflation is irrelevant. Region-"
      "adjusted medians are suppressed at this grain (as scripts/40 found). Rows of cohorts before 2013/14 "
      "are not used (they enter no analysis).")
    A(f"- **Multi-site providers.** {len(load_leo.multi)} providers are published once per provider region in some "
      f"cohort × YAG (UKPRNs {', '.join(load_leo.multi)}; The University of Law, whose London and South East sites "
      f"are separate rows from cohort 2018/19, and one alternative provider). For those cohort × YAG pairs only the "
      f"region with the most graduates in the provider's all-subject, all-graduate row is kept "
      f"({load_leo.minor_rows} rows dropped); without this a provider would carry two medians in one cell.")
    A(f"- **Cross-section (task b).** Cell = subject × cohort at YAG5, cohorts 2013/14–2016/17 (tax years "
      f"2019/20–2022/23), providers with G and a released median, n ≥ {NMIN}. Subject coupling = mean over its "
      f"cells; headline = unweighted mean over subjects. The old field-tagged axis F is rebuilt exactly by "
      f"scripts/40's own functions (per-CAH2 SpringRank on field-tagged edges, orientation check, bootstrap "
      f"stability gate) and compared with G on the same providers and cells; the scripts/40 values themselves "
      f"(tax year 2022/23, all name-matched providers) are read from `data/interim/us_uk_gap_comparison.csv`. US "
      f"couplings are rolled up to CAH2 with scripts/40's `F2CAH` map (mean over project fields): field prestige "
      f"= 1 − gap from `outputs/expanded66_gap_map.csv`; academia-wide brand c_G from "
      f"`outputs/field_vs_generic.csv` (scripts/28). The US–UK cross-subject Spearman is decomposed one factor at "
      f"a time over scripts/40's {len(kk16)} subjects: axis (F vs G on the same F-providers and cells), provider "
      f"sample (G on the F-providers vs all primary providers), earnings window (mean of cohorts 2013–16 at YAG5 vs "
      f"cohort 2016/17 at YAG5 = tax year 2022/23), plus every single cohort × YAG cross-section "
      f"(`data/interim/uk_leo_xnat_sections.csv`), leave-one-subject-out and a subject bootstrap ({NBOOT} "
      f"resamples; paired for differences, the US vector fixed per resample). The subject bootstrap covers "
      f"between-subject sampling only, not the sampling error of each subject's UK or US coupling.")
    A("- **Prior attainment (task c).** LEO bands are UCAS-tariff groups of the graduate's entry qualifications: "
      "1 = 4 As or more, 2 = 360 points (e.g. 3 As), 3 = 300–359, 4 = 240–299, 5 = 180–239, 6 = below 180, "
      "7 = 1–2 A-level passes, 8 = BTEC, 9 = other, plus not known. (i) *Within band*: cell = subject × cohort × "
      f"band at YAG5, providers with a released band median and a released all-graduate median, n ≥ {NMIN}; "
      "within = Spearman(G, band median), matched = Spearman(G, all-graduate median) on the same providers; bands "
      "1–5 primary (band 6 never reaches n ≥ 15; bands 7–8 and YAG1/YAG3 as extra rows). (ii) *Composition-"
      "standardised*: expected median = Σ_b share_b × national median_b (UK HEIs, same subject, cohort, YAG), with "
      "share_b the provider-subject's band mix of graduates included in earnings (all ten categories); earnings "
      "measure = log(median) − log(expected); cells need band counts covering ≥ 80% of the earnings sample. This "
      "treats the median of a mixture as the mixture of medians — an approximation. National band medians "
      "over-correct: they carry institution effects, because high-band graduates are concentrated at "
      "high-paying institutions. (ii′) *Within-provider standardisation* (the corrected version): the same formula "
      "with the national band gradient replaced by its within-provider size. Per subject × cohort × YAG cell, over "
      "every HEI in the LEO file that releases ≥ 2 band medians in the cell (prestige plays no part), OLS of log band "
      "median on provider FE + λ × log national band median; expected = Σ_b share_b × exp(λ × log national median_b). "
      "This covers every band the national version covers, so the composition sample is identical. A count-weighted "
      "WLS fit of λ is a robustness row. As a check, a free band gradient (provider FE + band FE, bands in the largest "
      "connected component) is used on the subsample of providers whose identified bands cover ≥ 80% of the "
      "earnings sample; raw and national-median coupling are recomputed on that subsample. Bands that are rarely "
      "released within provider (4 As or more; below 180 points) are often unidentified, so the free-FE subsample "
      "drops some providers with many such graduates. Gradients: `data/interim/uk_leo_band_gradients.csv`. "
      "(iii) *Partials*: rank-"
      "residualise G and earnings on the programme share of graduates with ≥ 360 points (bands 1–2 over known "
      "bands), on the institution-wide share (all subjects, same cohort and YAG), and/or on the regional earnings "
      "level (leave-one-out mean log all-subject median of other HEIs in the same English region / nation). "
      "(iv) *Attenuation check*: test–retest Spearman of the same providers' medians in adjacent cohorts "
      "(2013→14, 14→15, 15→16), for band medians and all-graduate medians on the same providers; corrected share = "
      "(within/matched) × √(r_all / r_band). This assumes true provider effects are equally stable across adjacent "
      "cohorts for band and all-graduate medians.")
    A(f"- **Career time (task d).** Cell = subject × cohort (2013/14–2016/17) with providers released at YAG 1, 3 and "
      f"5 (balanced), n ≥ {NMIN}; slope = OLS of coupling on years {{1, 3, 5}}; subject slope = mean over cohorts; "
      f"headline = mean over subjects. The same design on standardised earnings (within-provider and national "
      f"gradients) and on band medians (balanced within band). Coverage = graduates included in earnings / graduates; partial coupling given "
      f"same-YAG coverage. Calendar contrast on triple-matched providers (cohort c at YAG1 and YAG5, cohort c+4 at "
      f"YAG1); cohort c at YAG5 and cohort c+4 at YAG1 fall in exactly the same tax year. Under a linear "
      f"age–period–cohort reading (coupling = a·age + b·year + g·cohort) within-cohort = a+b, calendar-matched = "
      f"a−g, drift = b+g.")
    A(f"- **Setting-priced subjects (task e).** Core = {', '.join(SETTING_CORE)} (NHS Agenda for Change, doctors' "
      f"and dentists' national scales, teachers' statutory pay scales); broad adds Allied health (Agenda for "
      f"Change) as a sensitivity. The core three are the subjects named in the task specification, not picked "
      f"from the results. Group means with independent outer draws; difference p from an exact subject-label "
      f"permutation: all C(K, k) ways of labelling k of the K subjects as setting-priced (the count is in the "
      f"spec column of each row).")
    A("- **Heterogeneity across subjects.** For each coupling measure, subject coupling = mean over its YAG5 cells "
      "(cohorts 2013–16) and its sampling variance = variance of its provider-bootstrap replicates (the inner "
      "stage of the bootstrap below). DerSimonian–Laird τ² = max(0, (Q − (K − 1)) / (Σw − Σw²/Σw)) with "
      "w = 1/variance; I² = (Q − (K − 1))/Q; p from χ²(K − 1). Cross-subject Spearman correlations between two "
      "measures take their CI from the two-stage bootstrap, one outer draw of subjects and one inner replicate per "
      "drawn subject shared by both measures. *Coupling with institution selectivity* = Spearman across providers "
      "of the institution-wide share of graduates with ≥ 360 points and all-graduate earnings, on the composition "
      "sample.")
    A("- **Inference.** Frequency-weighted rank statistics reproduce resampling with replacement exactly (checked "
      "against scipy on resampled data). Providers are resampled within subject with one draw shared by every cell "
      "and analysis of that subject (cohorts, YAGs and bands of the same provider move together); subjects are "
      "resampled in the outer stage, each contributing an independently chosen inner replicate (as scripts/52). "
      "Streams are seeded per analysis (`rng_for`), so results do not depend on execution order. Two consecutive "
      "runs give byte-identical tables, figure and write-up (md5-checked). Peak resident memory about 0.9 GB, single "
      "process, about 1.5 minutes.")
    A("")

    # ------------------------------------------------------------------ caveats
    A("## Caveats\n")
    A("- **Prior-attainment bands are coarse.** Band 2 spans 3 As to about A*A*A*. A band-3 graduate of a highly "
      "selective provider may differ from one of an unselective provider (subject choice at A level, contextual "
      "offers, finer grades). Within-band coupling therefore overstates what would survive a *perfect* attainment "
      "control, and noise in band medians pulls the raw within-band figure the other way. The attenuation "
      "correction rests on the stability assumption stated in Method. The top band (4 As or more) reaches n ≥ 15 "
      "in only a handful of cells, so the very top of the attainment distribution is barely covered.")
    gratio = (r('log earnings gap PA2 - PA4: within provider (provider FE + band FE)').est /
              r('log earnings gap PA2 - PA4: national band medians').est)
    A(f"- **Standardisation, direction of bias.** National band medians over-correct (answer 1), so the "
      f"national-median figures ({100 * sh_nat:.0f}% surviving; the (e) and career rows so labelled) understate what "
      f"survives. Within-provider gradients remove only band differences at their within-provider size. They would "
      f"understate the attainment effect, and so overstate survival, if providers admit lower-band students on "
      f"strengths the band does not record (contextual offers, interviews, portfolios); coarse bands add to that. "
      f"The two within-provider versions differ in shape: the one-parameter λ fit (λ = "
      f"{f(r('lambda: within-provider / national band gradient (OLS)').est, 2, False)} at the median cell, IQR "
      f"{f(r('lambda: within-provider / national band gradient (OLS)').lo, 2, False)}–"
      f"{f(r('lambda: within-provider / national band gradient (OLS)').hi, 2, False)}) corrects more than the free "
      f"band-FE fit, whose band 2 vs band 4 gap is {gratio:.2f} of the national one at the median cell. The truth "
      f"for a perfect control is probably below the within-provider figures and above the national-median one.")
    A(f"- **Range restriction.** Band cells cover a narrower range of G than all-graduate cells (mean within-cell "
      f"SD of the G percentile {f(r('SD of G percentile within cells: band cells').est, 3, False)} vs "
      f"{f(r('SD of G percentile within cells: all-graduate cells').est, 3, False)}): a released band median needs "
      "enough graduates of that band at the provider. The matched comparison holds the providers fixed, so the "
      "survival ratio is not driven by this, but the within-band coupling describes a middle slice of providers.")
    A("- **Selectivity vs status.** Institution selectivity is measured from the same LEO cohorts (entry "
      "qualifications, not earnings). It is a status measure in its own right (it reflects demand for places), so "
      "'G adds nothing beyond selectivity' is not the same as 'status does not pay'. Public aggregates cannot "
      "separate value added, peer effects, employer screening and residual sorting.")
    A("- **Career time is still not separately identified.** The calendar-matched and within-cohort contrasts "
      "agree because the cross-cohort drift is near zero; under the linear APC reading this means calendar and "
      "cohort effects are small or offsetting. Tax years 2020/21–2021/22 (COVID, furlough) fall in the YAG5 window "
      "of the fixed cohorts and in the YAG1 cells of the calendar contrast. Only three horizons (1/3/5) and four "
      "cohorts are available; YAG10 is not in this file.")
    A("- **YAG1 earnings and further study.** At YAG1 some graduates of higher-G providers are in further study "
      "(e.g. Medical sciences: coupling negative at YAG1); the coverage control removes part of the slope. The "
      "YAG3→5 segment avoids the YAG1 point.")
    A(f"- **LEO population.** Graduates of English, Welsh and Scottish providers as published (the file has no "
      f"Northern Ireland providers, so Queen's Belfast is absent from the Russell Group check). Earnings medians "
      f"cover the graduates the release counts in its earnings figures (`grads_earnings_include`; median share of "
      f"graduates {f(r('median earnings coverage (included / graduates)').est, 2, False)} in the cross-section "
      f"cells). Medians cannot show tail effects.")
    A(f"- **Prestige network.** ORCID self-reports are a sample of UK academics, skewed to recent careers (median "
      f"first faculty year in the edges {r('median first-faculty-job year of the edges').est:.0f}) and to people who maintain ORCID records; small specialist "
      "institutions (e.g. the Courtauld, Royal Veterinary College) get extreme G from few edges. Results are "
      "unchanged with degree ≥ 20 (sensitivity table).")
    A("- **Heterogeneity test.** Subjects share providers, and the Q-test treats their couplings as independent "
      "given the provider-bootstrap variances. That variance covers provider sampling only, not cohort-to-cohort "
      "noise beyond it or error in G, so τ can overstate true between-subject variation somewhat. The cross-subject "
      "correlations in answer 3 compare two couplings computed from the same earnings medians, so their sampling "
      "errors are positively correlated. Shared noise pushes their correlation up; noise that is not shared "
      f"pulls it down. With median sampling SEs of {het['raw_c']['se_med']:.2f} (G) and "
      f"{het['sel']['se_med']:.2f} (selectivity) against a between-subject τ of {het['raw_c']['tau']:.2f} and "
      f"{het['sel']['tau']:.2f}, the effect is modest, but the "
      f"{f(r('cross-subject Spearman of subject couplings: coupling with G vs coupling with institution selectivity').est, 2)} "
      "is not a clean true-score correlation.")
    A("- **Setting-priced group has three subjects.** Group contrasts are descriptive; teaching (which pools teacher training with other "
      "education degrees) behaves differently from medicine and nursing on the trajectory.")
    A("- **Cross-national comparison.** US couplings are field-prestige (Wapman) or brand (Wapman academia) × "
      "Scorecard at 4 years; UK couplings are UK brand G × LEO at 5 years. The correlation is across subject "
      f"orderings only, with {len(kk16)} (or 21) subjects, and CAH2 rolls several US fields into one subject. It "
      "moves by ±0.2 or more with the provider sample, the earnings window or one subject (answer 6), so any single "
      "value of it is fragile.")
    A("")

    # ------------------------------------------------------------------ appendix
    A("## Appendix A — per subject\n")
    A("raw / p_prog / p_inst / p_geo: YAG5 coupling, mean over cohorts 2013/14–2016/17 (p_* = partial "
      "given programme selectivity / institution selectivity / regional earnings level). std wp / std nat: "
      "standardised with within-provider band gradients / with national band medians (over-corrects). "
      "within/matched: bands 1–5, cell means. Career: balanced fixed cohorts; slope CI = provider bootstrap within "
      "subject; wp slope = slope of the within-provider-standardised coupling. F (s40) / G (s40 cells): coupling with "
      "the field-tagged axis / with G on the same F-providers and cells. † = setting-priced (core).\n")
    A("| subject | n | raw | std wp | std nat | p_prog | p_inst | p_geo | within band | matched | YAG1 | YAG3 | YAG5 | slope/yr [95% CI] | wp slope | F (s40) | G (s40 cells) | US field |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sname, x in per.sort_values("raw", ascending=False).iterrows():
        dag = " †" if sname in SETTING_CORE else ""
        A(f"| {sname}{dag} | {x.n_mean:.0f} | {f(x.raw)} | {f(x.std_wp)} | {f(x['std'])} | {f(x.p_prog)} | {f(x.p_inst)} | "
          f"{f(x.p_geo)} | {f(x.within_band)} | {f(x.matched_band)} | {f(x.career_y1)} | {f(x.career_y3)} | "
          f"{f(x.career_y5)} | {f(x.career_slope)} [{f(x.career_slope_lo)}, {f(x.career_slope_hi)}] | "
          f"{f(x.career_slope_std_wp)} | {f(x.s40_rho_F)} | {f(x.s40_rho_G)} | {f(x.us_rho_field)} |")
    A("")

    # ------------------------------------------------------------------ provenance
    A("## Provenance (for SOURCES.md)\n")
    A(f"- **DfE LEO provider-level graduate outcomes** — Explore Education Statistics, *Graduate outcomes (LEO): "
      f"provider level data* (the release already recorded in `data/raw/SOURCES.md`, accessed 2026-06-22; not "
      f"re-downloaded for this analysis). URL: `https://content.explore-education-statistics.service.gov.uk/api/"
      f"releases/a13c6267-1527-4761-bf8e-3566d8d26629/files/3896dae9-11af-46ef-b6c6-2eed9acda9e5`. File "
      f"`data/raw/leo/leo_dashboard.zip`, {LEO_ZIP.stat().st_size:,} bytes, md5 `{_md5(LEO_ZIP)}`; contains "
      f"`provider_data_20250716.csv` (787,425,870 bytes uncompressed, latin-1; tax years 2015/16–2022/23, YAG 1/3/5, "
      f"academic-year cohorts 2009/10–2020/21) and `variable_names_lookup.csv`. Columns used: tax_year, "
      f"academic_year, YAG, ukprn, provider_name, provider_type, provider_country_name, provider_region_name, "
      f"cah2_subject_name, grads, grads_earnings_include, earnings_median, characteristic_type, "
      f"characteristic_value (prior_attainment_code, sex, POLAR4, All graduates).")
    A(f"- **ROR data dump** v2.8 (2026-06-02), Zenodo record 20512981, `data/raw/ror/ror-data.zip`, "
      f"{ROR_ZIP.stat().st_size:,} bytes, md5 `{_md5(ROR_ZIP)}` (already in SOURCES.md; accessed 2026-06-22). Used "
      f"for name variants of hiring-network nodes.")
    A(f"- **ORCID-derived mobility edges** — Zenodo 10.5281/zenodo.19651302, version 20260419 (already in "
      f"SOURCES.md). UK edges read from the scripts/40 cache `data/interim/orcid_uk_phd_faculty_edges.parquet` "
      f"(md5 `{_md5(INTERIM / 'orcid_uk_phd_faculty_edges.parquet')}`).")
    A("- **US comparison inputs** (repo outputs, not raw data): `outputs/expanded66_gap_map.csv`, "
      "`outputs/field_vs_generic.csv`, `data/interim/us_uk_gap_comparison.csv` (scripts/40).")
    A("- No new data were downloaded for this analysis.")
    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
