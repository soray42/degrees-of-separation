"""Prestige reliability: is "brand >= field" (scripts/28) a differential-measurement-error artifact?

scripts/28 found that the academia-wide Wapman rank G predicts a field's Scorecard 4-yr BA median
earnings at least as well as the field-specific Wapman rank F (mean c_G = +0.45 >= mean c_F = +0.40,
57 fields). Critique: F is estimated from far fewer hiring edges than G, so F is noisier, and
classical measurement error attenuates c_F more than c_G. This script quantifies that with public
data only and disattenuates c_F, c_G and ADV = c_F - c_G.

Steps (all seeded; outputs byte-identical on re-run):
  1. Public edge lists (data/raw/wapman2022/edge_lists.csv), self-loops dropped (SpringRank ignores
     them). Person-level split-half: each of the Total persons on an edge goes to half A or B with
     p = 1/2 (binomial thinning), R_SPLIT seeded replicates. SpringRank (src/ar_pipeline/springrank.py,
     alpha = 1e-3; alpha = 1 as sensitivity) on each half, per field and for the Academia network.
     Split-half r = Spearman(half A, half B) over (a) all ranked institutions in the public network
     and (b) the scripts/28 analysis sample (field x institution rows with F, G, earnings).
     Spearman-Brown: rel = 2r / (1 + r).
  2. Reconstruction check: Spearman(full-public-edge SpringRank, published ordinal Rank from
     ranks.csv - the quantity src/load_ar.py uses), vs the sqrt(rel) expected if the published rank
     were error-free and the only difference were public-edge sampling noise.
  3. The public edge list has no edge with Total < 2 (single-person institution pairs are absent), so
     the published ranks were estimated from MORE hires than the public edges. Published-rank
     reliability is therefore bracketed:
       P1 (lower bound): published rank treated as if estimated from the public edges only.
       P2 (extrapolated): Spearman-Brown step-up to the full hire count, m_full = m_pub / retention,
           retention = public non-self hires / (stats.csv Doctorate(US) - SelfHires) (field median
           imputed where stats.csv has no row). rel(m) = m / (m + m0), m0 = m_pub (1 - rel) / rel.
  4. Earnings-side reliability: rel_y = signal_frac from src.gap.compute_gap (null model, alpha = 0.6
     within-institution earnings CV, floor 0.02), on exactly the scripts/28 sample. It multiplies
     c_F and c_G of a field by the same factor, so it cannot change a field's sign of ADV.
  5. Disattenuate: c* = c / sqrt(rel_prestige * rel_y). Institution bootstrap within field (B_BOOT)
     for CIs; reliability uncertainty propagated by drawing one split replicate per bootstrap draw.
  6. Checks: (i) classical-error prediction c_half / c_full = sqrt(r_half / rel_full) on the public
     reconstruction; (ii) c_F, c_G on the subset of institutions present in the public field network,
     and the same using the public reconstructions (apples-to-apples public-data version).

Descriptive, no causal claims. Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/56_prestige_reliability.py
Outputs: data/interim/prestige_reliability.csv, PRESTIGE_RELIABILITY_RESULT.md (repo root, local).
"""
from __future__ import annotations

import sys
import zlib
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.ar_pipeline.springrank import springrank           # noqa: E402
from src.load_ar import load_ar_wapman                       # noqa: E402
from src.load_er import load_er_scorecard                    # noqa: E402
from src.gap import compute_gap                              # noqa: E402
from src.crosswalks.institutions import normalize_institution_name  # noqa: E402

SEED = 56
R_SPLIT = 200          # split-half replicates (primary alpha)
R_SENS = 100           # split-half replicates for the alpha = 1 sensitivity
B_BOOT = 1000          # institution bootstrap draws per field
ALPHA = 1e-3           # repo SpringRank default (scripts/08, src/genreg_bootstrap.py)
ALPHA_SENS = 1.0
MIN_N = 15             # scripts/28 MIN_N (fields with >= 15 institutions)
MIN_SUB = 6            # min institutions to compute a split-half correlation on a subset
B_GAP = 200            # compute_gap bootstrap draws (signal_frac itself is analytic)

WAP = ROOT / "data" / "raw" / "wapman2022"
OUT_CSV = ROOT / "data" / "interim" / "prestige_reliability.csv"
OUT_MD = ROOT / "PRESTIGE_RELIABILITY_RESULT.md"

# FIELDS66 + label map, exactly as scripts/52 loads them from scripts/28
_spec = importlib.util.spec_from_file_location("s28", ROOT / "scripts" / "28_field_vs_generic_prestige.py")
_s28 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s28)
FIELDS66, LAB = _s28.FIELDS66, _s28.LAB
WF = {f["key"]: f["wapman_field"] for f in FIELDS66}

# stats.csv uses a shorter name for three fields whose edge/rank name carries ", General"
STATS_NAME = {"Physics, General": "Physics", "Psychology, General": "Psychology",
              "Biological Sciences, General": "Biological Sciences"}


def _rng(*parts) -> np.random.Generator:
    """Deterministic per-unit generator (crc32 of the name; Python's hash() is salted per process)."""
    return np.random.default_rng(np.random.SeedSequence(
        [SEED] + [zlib.crc32(str(p).encode()) for p in parts]))


def sb(r):
    """Spearman-Brown step-up of a split-half correlation to the full-length reliability."""
    r = np.asarray(r, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 2 * r / (1 + r)
    return np.where(r > 0, out, np.nan)


def rowwise_spearman(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Spearman per row (Pearson on average ranks) - identical to scipy.stats.spearmanr per row."""
    rx = rankdata(X, axis=1); ry = rankdata(Y, axis=1)
    rx = rx - rx.mean(1, keepdims=True); ry = ry - ry.mean(1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (rx * ry).sum(1) / den, np.nan)


# ---------------------------------------------------------------------------
# 1. networks + split-half SpringRank
# ---------------------------------------------------------------------------
def build_net(edges: pd.DataFrame, level: str, value: str) -> dict:
    d = edges[(edges.TaxonomyLevel == level) & (edges.TaxonomyValue == value)]
    d = d[d.DegreeInstitutionId != d.InstitutionId]          # self-hires carry no SpringRank info
    ids = np.array(sorted(set(d.DegreeInstitutionId) | set(d.InstitutionId)), dtype=float)
    pos = {v: i for i, v in enumerate(ids)}
    return dict(level=level, value=value, ids=ids, pos=pos,
                src=d.DegreeInstitutionId.map(pos).to_numpy(), dst=d.InstitutionId.map(pos).to_numpy(),
                w=d.Total.to_numpy(np.int64), m=int(d.Total.sum()), n_edges=len(d))


def sr_scores(net: dict, w: np.ndarray, alpha: float) -> np.ndarray:
    n = len(net["ids"])
    A = csr_matrix((w.astype(float), (net["src"], net["dst"])), shape=(n, n))
    return springrank(A, alpha=alpha)


def split_halves(net: dict, R: int, alpha: float, tag: str):
    """R person-level split-halves -> (HA, HB) arrays [R, n_nodes] of SpringRank scores."""
    rng = _rng("split", net["level"], net["value"], tag)
    n = len(net["ids"])
    HA, HB = np.empty((R, n)), np.empty((R, n))
    for k in range(R):
        h = rng.binomial(net["w"], 0.5)
        HA[k] = sr_scores(net, h, alpha)
        HB[k] = sr_scores(net, net["w"] - h, alpha)
    return HA, HB


def split_r(HA, HB, idx) -> np.ndarray:
    """Per-replicate split-half Spearman over node positions idx."""
    idx = np.asarray(idx, int)
    if len(idx) < MIN_SUB:
        return np.full(HA.shape[0], np.nan)
    return rowwise_spearman(HA[:, idx], HB[:, idx])


# ---------------------------------------------------------------------------
# 2. analysis sample (scripts/28) + earnings reliability
# ---------------------------------------------------------------------------
def analysis_sample():
    t = _s28.build_table()                     # scripts/28 per-(institution x field) table
    pf, _ = _s28.per_field(t)                  # scripts/28 c_F, c_G, ADV per field (n >= 8)
    t = t[t.field.isin(pf.field)].copy()
    t["F_id"] = pd.to_numeric(t.institution_id, errors="coerce")
    rk = pd.read_csv(WAP / "ranks.csv")
    ac = rk[rk.TaxonomyLevel == "Academia"].copy()
    ac["inst_key"] = ac.InstitutionName.map(normalize_institution_name)
    ac = ac[ac.inst_key != ""].drop_duplicates("inst_key")     # same rule as scripts/28 load_generic
    t = t.merge(ac[["inst_key", "InstitutionId"]].rename(columns={"InstitutionId": "G_id"}),
                on="inst_key", how="left")
    return t, pf


def earnings_reliability(t: pd.DataFrame) -> pd.DataFrame:
    """signal_frac from src.gap.compute_gap on exactly the scripts/28 sample of each field."""
    ar = load_ar_wapman(fields=FIELDS66)
    er = load_er_scorecard("undergrad", earn_col="EARN_MDN_4YR", count_col="EARN_COUNT_WNE_4YR",
                           fields=FIELDS66)
    rows = []
    for f, g in t.groupby("field"):
        a = ar[(ar.field == f) & ar.inst_key.isin(g.inst_key)]
        rec = compute_gap(a, er, f, "undergrad", B=B_GAP, seed=SEED)
        rows.append(dict(field=f, signal_frac=rec["signal_frac"], n_gap=rec["n_institutions"],
                         c_F_gap=rec["spearman"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. retention (public share of hires) from stats.csv
# ---------------------------------------------------------------------------
def retention_table(nets: dict) -> dict:
    st = pd.read_csv(WAP / "stats.csv")
    out = {}
    for key, net in nets.items():
        name = net["value"]
        sname = STATS_NAME.get(name, name)
        row = st[(st.TaxonomyLevel == net["level"]) & (st.TaxonomyValue == sname)]
        if len(row):
            full = float(row["Doctorate (US)"].iloc[0] - row["SelfHires"].iloc[0])
            src = "stats.csv" if sname == name else "stats.csv (name-matched)"
            out[key] = (net["m"] / full, full, src)
        else:
            out[key] = (np.nan, np.nan, "imputed (field median)")
    med = float(np.nanmedian([v[0] for k, v in out.items() if k != "ACADEMIA"]))
    for k, v in out.items():
        if not np.isfinite(v[0]):
            out[k] = (med, nets[k]["m"] / med, v[2])
    return out, med


def extrapolate(rel, m_pub, m_full):
    """Spearman-Brown family rel(m) = m / (m + m0); m0 from the public reliability."""
    rel = np.asarray(rel, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = m_pub * (1 - rel) / rel
        return np.where(rel > 0, m_full / (m_full + m0), np.nan)


# ---------------------------------------------------------------------------
# scenarios: which c's, which sample, which prestige reliabilities
# ---------------------------------------------------------------------------
#   name      sample  c's      reliability of (F, G)
SCEN = [
    ("U",      "full", "pub", None),     # scripts/28 as published: no correction
    ("LB",     "full", "pub", "LB"),     # public-edge split-half reliability (lower bound for published rank)
    ("LBa1",   "full", "pub", "LBa1"),   # same, SpringRank alpha = 1
    ("EXT",    "full", "pub", "EXT"),    # Spearman-Brown extrapolated to the full hire count
    ("Ucov",   "cov",  "pub", None),     # institutions present in the public field network only
    ("LBcov",  "cov",  "pub", "LB"),     # ... reliabilities measured on exactly these institutions
    ("EXTcov", "cov",  "pub", "EXT"),
    ("RECcov", "cov",  "rec", None),     # public-edge reconstructions of F and G (not the published ranks)
    ("RECcovLB", "cov", "rec", "LB"),    # ... disattenuated with their own measured reliabilities
]
SCEN_LABEL = {
    "U": "uncorrected (scripts/28 as published)",
    "LB": "corrected, public-edge reliability (lower bound for F)",
    "LBa1": "corrected, public-edge reliability, SpringRank alpha=1",
    "EXT": "corrected, reliability extrapolated to full hire count",
    "Ucov": "covered subset, uncorrected",
    "LBcov": "covered subset, public-edge reliability",
    "EXTcov": "covered subset, extrapolated reliability",
    "RECcov": "covered subset, public-edge reconstructions, uncorrected",
    "RECcovLB": "covered subset, public-edge reconstructions, corrected",
}


def rel_pair(st: dict, sample: str, kind: str, draw=None):
    """(rel_F, rel_G) point estimate (draw=None) or per-draw arrays (draw = (kF, kG) replicate idx)."""
    if kind is None:
        return 1.0, 1.0
    rF, rG = (st["rFc"], st["rGc"]) if sample == "cov" else \
             ((st["rA1"], st["rG1"]) if kind == "LBa1" else (st["rA"], st["rG"]))
    if draw is None:
        relF, relG = sb(np.nanmean(rF)), sb(np.nanmean(rG))
    else:
        relF, relG = sb(rF[draw[0] % len(rF)]), sb(rG[draw[1] % len(rG)])
    if kind == "EXT":
        relF = extrapolate(relF, st["m"], st["mfull"])
        relG = extrapolate(relG, st["mA"], st["mfullA"])
    return relF, relG


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    edges = pd.read_csv(WAP / "edge_lists.csv")
    ranks = pd.read_csv(WAP / "ranks.csv")
    min_total = {lv: int(edges[edges.TaxonomyLevel == lv].Total.min()) for lv in ("Academia", "Domain", "Field")}

    t, pf = analysis_sample()
    fields = sorted(pf.field)
    ery = earnings_reliability(t).set_index("field")

    nets = {"ACADEMIA": build_net(edges, "Academia", "Academia")}
    for f in fields:
        nets[f] = build_net(edges, "Field", WF[f])
    ret, ret_med = retention_table(nets)

    full, halves, halves1 = {}, {}, {}
    for key, net in nets.items():
        full[key] = sr_scores(net, net["w"], ALPHA)
        halves[key] = split_halves(net, R_SPLIT, ALPHA, "a1e-3")
        halves1[key] = split_halves(net, R_SENS, ALPHA_SENS, "a1")
    print(f"split-halves done for {len(nets)} networks", flush=True)

    # ---- Academia network (G), over all ranked institutions present in the public network ----
    nA = nets["ACADEMIA"]
    rk_ac = ranks[ranks.TaxonomyLevel == "Academia"].set_index("InstitutionId").Rank
    idx_ac = [nA["pos"][i] for i in rk_ac.index if i in nA["pos"]]
    rA_net = split_r(*halves["ACADEMIA"], idx_ac)
    rA_net1 = split_r(*halves1["ACADEMIA"], idx_ac)
    retA, mfullA, srcA = ret["ACADEMIA"]
    relA_net = float(sb(np.nanmean(rA_net)))
    acad = dict(unit="academia_network", field="ACADEMIA", label="Academia-wide (G)", wapman_field="Academia",
                n_ranked=len(rk_ac), n_nodes_public=len(nA["ids"]), n_ranked_in_public=len(idx_ac),
                m_pub=nA["m"], retention=retA, m_full_est=mfullA, retention_source=srcA,
                rhalf_net=float(np.nanmean(rA_net)), rel_net=relA_net,
                rel_net_alpha1=float(sb(np.nanmean(rA_net1))),
                rel_net_ext=float(extrapolate(relA_net, nA["m"], mfullA)),
                m0_net=nA["m"] * (1 - relA_net) / relA_net,
                rho_recon_pub_net=spearmanr(full["ACADEMIA"][idx_ac], -rk_ac.loc[nA["ids"][idx_ac]].to_numpy())[0],
                sqrt_rel_net=float(np.sqrt(relA_net)))

    # ---- per field ----
    rows, store = [], {}
    GA, GB = halves["ACADEMIA"]; GA1, GB1 = halves1["ACADEMIA"]
    for f in fields:
        net = nets[f]; g = t[t.field == f].reset_index(drop=True)
        rk_f = ranks[(ranks.TaxonomyLevel == "Field") & (ranks.TaxonomyValue == WF[f])].set_index("InstitutionId").Rank
        HA, HB = halves[f]; HA1, HB1 = halves1[f]
        # (a) network-wide reliability: ranked institutions present in the public field network
        idx_net = [net["pos"][i] for i in rk_f.index if i in net["pos"]]
        r_net = split_r(HA, HB, idx_net); r_net1 = split_r(HA1, HB1, idx_net)
        rel_net = float(sb(np.nanmean(r_net)))
        rho_rp_net = spearmanr(full[f][idx_net], -rk_f.loc[net["ids"][idx_net]].to_numpy())[0]
        # (b) analysis sample (scripts/28 rows); F reliability over rows covered by the public field network
        covF = g.F_id.isin(net["pos"]).to_numpy(); covG = g.G_id.isin(nA["pos"]).to_numpy()
        pF_A = np.array([net["pos"][i] for i in g.F_id[covF]], int)
        pG_A = np.array([nA["pos"][i] for i in g.G_id[covG]], int)
        cov = covF & covG; gc = g[cov]
        pF_c = np.array([net["pos"][i] for i in gc.F_id], int)
        pG_c = np.array([nA["pos"][i] for i in gc.G_id], int)
        retf, mfull, rsrc = ret[f]
        st = dict(F=g.F.to_numpy(float), G=g.G.to_numpy(float), y=g.y.to_numpy(float),
                  Fc=gc.F.to_numpy(float), Gc=gc.G.to_numpy(float), yc=gc.y.to_numpy(float),
                  sFc=full[f][pF_c], sGc=full["ACADEMIA"][pG_c],
                  rA=split_r(HA, HB, pF_A), rG=split_r(GA, GB, pG_A),
                  rA1=split_r(HA1, HB1, pF_A), rG1=split_r(GA1, GB1, pG_A),
                  rFc=split_r(HA, HB, pF_c), rGc=split_r(GA, GB, pG_c),
                  m=net["m"], mfull=mfull, mA=nA["m"], mfullA=mfullA,
                  sf=float(ery.loc[f, "signal_frac"]))
        store[f] = st
        # classical-error check on the public reconstruction: c at half the edges vs at all edges
        if len(gc) >= MIN_SUB:
            Y = np.broadcast_to(st["yc"], (R_SPLIT, len(gc)))
            cFr_half = float(np.nanmean(np.r_[rowwise_spearman(HA[:, pF_c], Y), rowwise_spearman(HB[:, pF_c], Y)]))
            cGr_half = float(np.nanmean(np.r_[rowwise_spearman(GA[:, pG_c], Y), rowwise_spearman(GB[:, pG_c], Y)]))
            rho_rp_A = spearmanr(st["sFc"], st["Fc"])[0]      # F = -published Rank: + = agreement
        else:
            cFr_half = cGr_half = rho_rp_A = np.nan
        p = pf.set_index("field").loc[f]
        row = dict(unit="field", field=f, label=LAB.get(f, f), wapman_field=WF[f], n_inst=len(g),
                   n_ranked=len(rk_f), n_nodes_public=len(net["ids"]), n_ranked_in_public=len(idx_net),
                   m_pub=net["m"], retention=retf, m_full_est=mfull, retention_source=rsrc,
                   rhalf_net=float(np.nanmean(r_net)), rel_net=rel_net,
                   rel_net_alpha1=float(sb(np.nanmean(r_net1))),
                   rel_net_ext=float(extrapolate(rel_net, net["m"], mfull)),
                   m0_net=net["m"] * (1 - rel_net) / rel_net,
                   rho_recon_pub_net=rho_rp_net, sqrt_rel_net=float(np.sqrt(rel_net)),
                   n_cov=int(cov.sum()), cov_share=float(cov.mean()), rho_recon_pub_cov=rho_rp_A,
                   signal_frac=st["sf"], n_gap=int(ery.loc[f, "n_gap"]), c_F_gap=float(ery.loc[f, "c_F_gap"]),
                   c_F=float(p.c_F), c_G=float(p.c_G), rho_PP=float(p.rho_PP))
        for kind, tag in (("LB", "LB"), ("LBa1", "LBa1"), ("EXT", "EXT")):
            rF, rG = rel_pair(st, "full", kind)
            row[f"relF_{tag}"], row[f"relG_{tag}"] = float(rF), float(rG)
        row["relF_LB_sd"] = float(np.nanstd(sb(st["rA"]))); row["relG_LB_sd"] = float(np.nanstd(sb(st["rG"])))
        for kind, tag in (("LB", "LBcov"), ("EXT", "EXTcov")):
            rF, rG = rel_pair(st, "cov", kind)
            row[f"relF_{tag}"], row[f"relG_{tag}"] = float(rF), float(rG)
        row["cFrec_half_obs"], row["cGrec_half_obs"] = cFr_half, cGr_half
        rows.append(row)
    df = pd.DataFrame(rows)

    # ---- point estimates per scenario ----
    def c_pair(st, sample, cs):
        if sample == "full":
            return spearmanr(st["F"], st["y"])[0], spearmanr(st["G"], st["y"])[0]
        if len(st["yc"]) < MIN_SUB:
            return np.nan, np.nan
        if cs == "pub":
            return spearmanr(st["Fc"], st["yc"])[0], spearmanr(st["Gc"], st["yc"])[0]
        return spearmanr(st["sFc"], st["yc"])[0], spearmanr(st["sGc"], st["yc"])[0]

    for name, sample, cs, kind in SCEN:
        cF_, cG_ = [], []
        for f in df.field:
            st = store[f]; a, b = c_pair(st, sample, cs); rF, rG = rel_pair(st, sample, kind)
            cF_.append(a / np.sqrt(rF)); cG_.append(b / np.sqrt(rG))
        df[f"cF_{name}"], df[f"cG_{name}"] = cF_, cG_
        df[f"ADV_{name}"] = df[f"cF_{name}"] - df[f"cG_{name}"]
    assert np.allclose(df.cF_U, df.c_F) and np.allclose(df.cG_U, df.c_G)
    # classical-error prediction for the half-edge reconstruction correlations
    rFc_m = np.array([np.nanmean(store[f]["rFc"]) for f in df.field])
    rGc_m = np.array([np.nanmean(store[f]["rGc"]) for f in df.field])
    df["cFrec_half_pred"] = df.cF_RECcov * np.sqrt(rFc_m / sb(rFc_m))
    df["cGrec_half_pred"] = df.cG_RECcov * np.sqrt(rGc_m / sb(rGc_m))
    df["breakeven_ratio"] = np.where((df.c_F > 0) & (df.c_G > 0), (df.c_F / df.c_G) ** 2, np.nan)
    df["relratio_LB"] = df.relF_LB / df.relG_LB
    df["relratio_EXT"] = df.relF_EXT / df.relG_EXT

    # ---- institution bootstrap within field (+ one split replicate per draw for the reliabilities) ----
    boot = {name: {} for name, *_ in SCEN}
    for f in df.field:
        st = store[f]; rng = _rng("boot", f)
        n, nc = len(st["y"]), len(st["yc"])
        idx = rng.integers(0, n, size=(B_BOOT, n))
        idc = rng.integers(0, max(nc, 1), size=(B_BOOT, max(nc, 1)))
        kF = rng.integers(0, R_SPLIT, B_BOOT); kG = rng.integers(0, R_SPLIT, B_BOOT)
        cb = {("full", "pub"): (rowwise_spearman(st["F"][idx], st["y"][idx]),
                                rowwise_spearman(st["G"][idx], st["y"][idx]))}
        if nc >= MIN_SUB:
            cb[("cov", "pub")] = (rowwise_spearman(st["Fc"][idc], st["yc"][idc]),
                                  rowwise_spearman(st["Gc"][idc], st["yc"][idc]))
            cb[("cov", "rec")] = (rowwise_spearman(st["sFc"][idc], st["yc"][idc]),
                                  rowwise_spearman(st["sGc"][idc], st["yc"][idc]))
        else:
            cb[("cov", "pub")] = cb[("cov", "rec")] = (np.full(B_BOOT, np.nan),) * 2
        for name, sample, cs, kind in SCEN:
            a, b = cb[(sample, cs)]
            rF, rG = rel_pair(st, sample, kind, draw=(kF, kG)) if kind else (1.0, 1.0)
            boot[name][f] = a / np.sqrt(rF) - b / np.sqrt(rG)
    for name in ("U", "LB", "EXT", "Ucov", "LBcov", "EXTcov"):
        M = np.vstack([boot[name][f] for f in df.field])
        df[f"ADV_{name}_lo"], df[f"ADV_{name}_hi"] = np.nanpercentile(M, [2.5, 97.5], axis=1)

    # ---- across-field summaries ----
    masks = {"all57": np.ones(len(df), bool), "n>=15": (df.n_inst >= MIN_N).to_numpy(),
             "signal_frac>=0.5": (df.signal_frac >= 0.5).to_numpy()}
    sf = df.signal_frac.to_numpy()
    summ = []
    for name, sample, cs, kind in SCEN:
        for mname in ("all57", "n>=15"):
            mk = masks[mname] & df[f"ADV_{name}"].notna().to_numpy()
            M = np.vstack([boot[name][f] for f in df.field[mk]]); mb = np.nanmean(M, axis=0)
            summ.append(dict(scenario=name, earnings="rel_y=1", sample=mname, k_fields=int(mk.sum()),
                             mean_cF=df.loc[mk, f"cF_{name}"].mean(), mean_cG=df.loc[mk, f"cG_{name}"].mean(),
                             mean_ADV=df.loc[mk, f"ADV_{name}"].mean(),
                             ci_lo=np.nanpercentile(mb, 2.5), ci_hi=np.nanpercentile(mb, 97.5),
                             n_pos=int((df.loc[mk, f"ADV_{name}"] > 0).sum()),
                             n_ci_pos=int((df.loc[mk, f"ADV_{name}_lo"] > 0).sum()) if f"ADV_{name}_lo" in df else -1,
                             n_ci_neg=int((df.loc[mk, f"ADV_{name}_hi"] < 0).sum()) if f"ADV_{name}_hi" in df else -1))
    for name in ("U", "LB", "EXT"):                # earnings-side correction: rel_y = signal_frac
        for mname in ("all57", "signal_frac>=0.5"):
            mk = masks[mname]
            M = np.vstack([boot[name][f] / np.sqrt(store[f]["sf"]) for f in df.field[mk]]); mb = np.nanmean(M, axis=0)
            summ.append(dict(scenario=name, earnings="rel_y=signal_frac", sample=mname, k_fields=int(mk.sum()),
                             mean_cF=(df.loc[mk, f"cF_{name}"] / np.sqrt(sf[mk])).mean(),
                             mean_cG=(df.loc[mk, f"cG_{name}"] / np.sqrt(sf[mk])).mean(),
                             mean_ADV=(df.loc[mk, f"ADV_{name}"] / np.sqrt(sf[mk])).mean(),
                             ci_lo=np.nanpercentile(mb, 2.5), ci_hi=np.nanpercentile(mb, 97.5),
                             n_pos=int((df.loc[mk, f"ADV_{name}"] > 0).sum()), n_ci_pos=-1, n_ci_neg=-1))
    S = pd.DataFrame(summ)

    out = pd.concat([pd.DataFrame([acad]), df], ignore_index=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.6f")
    write_report(df, acad, S, ret_med, min_total)
    with pd.option_context("display.width", 200):
        print(S.round(4).to_string(index=False))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _verdict(lo, hi):
    if hi < 0:
        return "brand > field survives (CI below 0)"
    if lo > 0:
        return "reverses: field > brand (CI above 0)"
    return "indistinguishable from 0 (CI spans 0)"


def write_report(df, acad, S, ret_med, min_total):
    def s(name, sample="all57", earn="rel_y=1"):
        return S[(S.scenario == name) & (S["sample"] == sample) & (S.earnings == earn)].iloc[0]

    U, LB, EXT, LBa1 = s("U"), s("LB"), s("EXT"), s("LBa1")
    Uc, LBc, EXTc, Rc, RcLB = s("Ucov"), s("LBcov"), s("EXTcov"), s("RECcov"), s("RECcovLB")
    U15, LB15, EXT15 = s("U", "n>=15"), s("LB", "n>=15"), s("EXT", "n>=15")
    Uy, LBy, EXTy = s("U", earn="rel_y=signal_frac"), s("LB", earn="rel_y=signal_frac"), s("EXT", earn="rel_y=signal_frac")
    Uy5, LBy5, EXTy5 = (s(k, "signal_frac>=0.5", "rel_y=signal_frac") for k in ("U", "LB", "EXT"))
    lam = (df.c_F.mean() / df.c_G.mean()) ** 2
    lam15 = (df[df.n_inst >= MIN_N].c_F.mean() / df[df.n_inst >= MIN_N].c_G.mean()) ** 2
    fb = lambda v: f"{v:+.4f}" if abs(v) < 0.0005 else f"{v:+.3f}"      # keep a bound like +0.0004 visible
    ci = lambda r: f"{r.mean_ADV:+.3f} [{fb(r.ci_lo)}, {fb(r.ci_hi)}]"
    rel_m = spearmanr(df.rel_net, df.m_pub)[0]
    rel_mn = spearmanr(df.rel_net, df.m_pub / df.n_ranked_in_public)[0]
    below = int((df.rho_recon_pub_net < df.sqrt_rel_net).sum())
    sf_floor = int((df.signal_frac <= 0.0201).sum())
    sf_viol = int((np.maximum(df.c_F.abs(), df.c_G.abs()) > np.sqrt(df.signal_frac)).sum())
    flipLB = int(((df.ADV_U < 0) & (df.ADV_LB > 0)).sum()); flipEXT = int(((df.ADV_U < 0) & (df.ADV_EXT > 0)).sum())
    chk = float(np.nanmax(np.abs(df.c_F - df.c_F_gap))); nchk = bool((df.n_gap == df.n_inst).all())
    vLB, vEXT = _verdict(LB.ci_lo, LB.ci_hi), _verdict(EXT.ci_lo, EXT.ci_hi)
    vEXT15 = _verdict(EXT15.ci_lo, EXT15.ci_hi)
    n_rows, n_cov_rows = int(df.n_inst.sum()), int(df.n_cov.sum())
    n_stats = int((df.retention_source != "imputed (field median)").sum()); n_imp = len(df) - n_stats
    absent_med = float((1 - df.n_ranked_in_public / df.n_ranked).median())
    cs = df.set_index("field").loc["computer_science"] if "computer_science" in set(df.field) else None
    ratio_med = acad["m_pub"] / df.m_pub.median()
    floor_big = df[df.signal_frac <= 0.0201].sort_values("n_inst", ascending=False).head(2)
    floor_ex = " and ".join(f"{r.label} (n={r.n_inst}, c_F={r.c_F:+.2f})" for r in floor_big.itertuples())
    big15 = df.n_inst >= MIN_N
    rr15 = float(df.loc[big15, "relratio_LB"].mean())
    full_scen = [U.mean_ADV, LB.mean_ADV, LBa1.mean_ADV, EXT.mean_ADV]
    reverses = (LB.ci_lo > 0) or (EXT.ci_lo > 0)

    if reverses:
        head = "**It reverses under at least one reliability scenario** (see table)."
    elif LB.ci_hi >= 0 and EXT.ci_hi < 0:
        head = ("**It does not reverse. Whether it survives or becomes indistinguishable depends on the reliability "
                "of the published field rank, and public data can only bracket that.** At the pessimistic end "
                "(field-prestige reliability as measured from the public edges alone), brand's advantage shrinks "
                "to about zero and is indistinguishable from zero. At the optimistic end (reliability extrapolated "
                "to the hire counts behind the published ranks), it stays close to its uncorrected size and its "
                f"CI only just excludes zero ({vEXT15.split(' (')[0]} in the {int(EXT15.k_fields)} fields with "
                "n >= 15). No scenario gives a CI above zero, so no scenario shows field prestige predicting "
                "earnings better than brand. The weak form of the claim (field is not better than brand) holds "
                "throughout. The strict form (brand is better) does not survive the pessimistic correction.")
    elif LB.ci_hi >= 0 and EXT.ci_hi >= 0:
        head = ("**It becomes indistinguishable.** The sign does not reverse, but after correction brand's "
                "advantage is not distinguishable from zero under either reliability scenario.")
    else:
        head = "**It survives** under both reliability scenarios (CIs below 0)."

    L = []
    L.append("# Prestige reliability: is \"brand >= field\" a measurement artifact?\n")
    L.append("Script: `scripts/56_prestige_reliability.py` (seeded; re-run gives byte-identical outputs). "
             "Table: `data/interim/prestige_reliability.csv` (one row per field plus one Academia-network row). "
             "Public data only (Wapman et al. 2022 public release + College Scorecard FoS). Descriptive; no causal claims.\n")
    L.append("## Answer\n")
    L.append(head + "\n")
    L.append(f"- Uncorrected (scripts/28, 57 fields): mean ADV = c_F - c_G = {ci(U)}; ADV > 0 in {U.n_pos}/57 fields.")
    L.append(f"- Field prestige corrected with its **public-edge reliability** (a lower bound for the published "
             f"rank; mean rel_F = {df.relF_LB.mean():.2f} vs rel_G = {df.relG_LB.mean():.2f}): mean ADV* = {ci(LB)} "
             f"-> {vLB}. ADV* > 0 in {LB.n_pos}/57 fields.")
    L.append(f"- Reliability **extrapolated to the full hire count** (mean rel_F = {df.relF_EXT.mean():.2f} vs "
             f"rel_G = {df.relG_EXT.mean():.2f}): mean ADV* = {ci(EXT)} -> {vEXT}. ADV* > 0 in {EXT.n_pos}/57 fields.")
    L.append(f"- Break-even: the across-field mean flips sign only if field prestige is less than "
             f"(mean c_F / mean c_G)^2 = **{lam:.3f}** as reliable as brand (uniform ratio; {lam15:.3f} for the "
             f"{int((df.n_inst >= MIN_N).sum())} fields with n >= 15). The measured public-edge ratio rel_F/rel_G "
             f"averages {df.relratio_LB.mean():.3f} ({rr15:.3f} for n >= 15). That sits at break-even, so the "
             f"pessimistic correction removes most of the gap without reversing it. The extrapolated ratio "
             f"averages {df.relratio_EXT.mean():.3f}. G is nearly error-free on these samples, so the whole "
             f"correction comes from F.")
    L.append(f"- Why this is a bracket and not a point: the public edge list has **no edge with Total < 2** (minimum "
             f"Total = {min_total['Field']} at Field, {min_total['Academia']} at Academia level). Single-person "
             f"institution pairs are therefore absent. In the {n_stats} of 57 fields with a stats.csv row, the public "
             f"edges carry a median {ret_med:.2f} of US-doctorate non-self-hire faculty; the Academia network carries "
             f"{acad['retention']:.2f}. ranks.csv, which scripts/28 uses, also ranks institutions with no public "
             f"field edge at all: a median {absent_med:.0%} of ranked institutions per field are absent from the "
             f"public field network"
             + (f" (Computer Science: {int(cs.n_ranked)} ranked, {int(cs.n_ranked_in_public)} present)" if cs is not None else "")
             + ". The published ranks were therefore estimated from more hires than the public edges contain, and "
             "their sampling reliability should lie above the public-edge value. How far above cannot be measured "
             "from public data.")
    L.append(f"- Plain reading: the claim \"field-specific prestige predicts earnings **better** than brand\" is not "
             f"supported under any correction. The claim \"brand predicts **better** than field\" is not robust to "
             f"differential reliability: the pessimistic bound absorbs it. On the full 57-field sample every "
             f"correction shrinks it, from {U.mean_ADV:+.3f} to between {max(full_scen[1:]):+.3f} and "
             f"{min(full_scen[1:]):+.3f}. The defensible statement is \"field prestige is no better than the "
             f"academia-wide brand, and brand's small edge may be partly or wholly a reliability artifact.\"\n")

    L.append("## Key numbers\n")
    L.append("| quantity | value | sample / spec |")
    L.append("|---|---|---|")
    kv = [
        ("mean c_F, mean c_G (uncorrected)", f"{U.mean_cF:+.3f}, {U.mean_cG:+.3f}", "57 fields, scripts/28 table; Spearman; published Wapman ranks vs Scorecard 4-yr BA median"),
        ("mean ADV uncorrected [95% CI]", ci(U), "57 fields; CI = institution bootstrap within field, B=1000"),
        ("mean ADV uncorrected, n>=15 fields", ci(U15), f"{int(U15.k_fields)} fields"),
        ("mean ADV*, public-edge reliability (LB)", ci(LB), "57 fields; rel_F, rel_G = Spearman-Brown split-half on the analysis sample; alpha=1e-3; rel_y=1"),
        ("mean ADV*, LB, n>=15 fields", ci(LB15), f"{int(LB15.k_fields)} fields"),
        ("mean ADV*, LB with SpringRank alpha=1", ci(LBa1), "57 fields; sensitivity"),
        ("mean ADV*, extrapolated reliability (EXT)", ci(EXT), "57 fields; rel(m)=m/(m+m0) at m_full = m_pub/retention"),
        ("mean ADV*, EXT, n>=15 fields", ci(EXT15), f"{int(EXT15.k_fields)} fields"),
        ("mean ADV, covered subset uncorrected", ci(Uc), f"{int(Uc.k_fields)} fields; only institutions present in the public field network ({n_cov_rows} of {n_rows} rows)"),
        ("mean ADV*, covered subset, LB", ci(LBc), "reliabilities measured on exactly these institutions"),
        ("mean ADV*, covered subset, EXT", ci(EXTc), "same, extrapolated"),
        ("mean ADV from public-edge reconstructions (F_rec, G_rec)", ci(Rc), "covered subset; SpringRank on public edges instead of published ranks"),
        ("... disattenuated with their own reliabilities", ci(RcLB), "covered subset"),
        ("mean c_F*, c_G*: LB; EXT", f"{LB.mean_cF:+.3f}, {LB.mean_cG:+.3f}; {EXT.mean_cF:+.3f}, {EXT.mean_cG:+.3f}", "57 fields; prestige-side correction only"),
        ("mean rel_F public-edge (analysis sample), range", f"{df.relF_LB.mean():.3f} ({df.relF_LB.min():.2f}-{df.relF_LB.max():.2f})", "57 fields; alpha=1e-3; R=200 splits"),
        ("mean rel_G public-edge (same samples), range", f"{df.relG_LB.mean():.3f} ({df.relG_LB.min():.2f}-{df.relG_LB.max():.2f})", "Academia network restricted to each field's sample"),
        ("mean rel_F, rel_G extrapolated", f"{df.relF_EXT.mean():.3f}, {df.relG_EXT.mean():.3f}", "EXT"),
        ("break-even rel_F/rel_G for the mean", f"{lam:.3f}", "(mean c_F / mean c_G)^2, 57 fields"),
        ("mean rel_F/rel_G: LB, EXT", f"{df.relratio_LB.mean():.3f}, {df.relratio_EXT.mean():.3f}", f"57 fields (LB, n>=15: {rr15:.3f})"),
        ("fields whose ADV changes sign - to +: LB, EXT", f"{flipLB}, {flipEXT}", f"of {int((df.ADV_U < 0).sum())} fields with ADV<0 uncorrected"),
        ("per-field CIs: ADV* > 0 / < 0 (LB)", f"{LB.n_ci_pos} / {LB.n_ci_neg}", "field-level 95% bootstrap CIs; uncorrected: " + f"{U.n_ci_pos} / {U.n_ci_neg}"),
        ("Academia network: public hires, rel (all ranked), rel extrapolated", f"{acad['m_pub']:,}, {acad['rel_net']:.3f}, {acad['rel_net_ext']:.3f}", f"{acad['n_ranked_in_public']} ranked institutions in the public network"),
        ("field networks: median public hires, median rel (all ranked)", f"{int(df.m_pub.median())}, {df.rel_net.median():.3f}", "57 fields, network-wide split-half"),
        ("Spearman(rel_F network-wide, public hires) across fields", f"{rel_m:+.2f}", f"hires per ranked node: {rel_mn:+.2f}"),
        ("Spearman(public SpringRank, published Rank): Academia; field median (range)",
         f"{acad['rho_recon_pub_net']:.3f}; {df.rho_recon_pub_net.median():.3f} ({df.rho_recon_pub_net.min():.2f}-{df.rho_recon_pub_net.max():.2f})",
         f"all ranked institutions in public network; below sqrt(rel) in {below}/57 fields"),
        ("earnings side: fields with signal_frac at its 0.02 floor", f"{sf_floor}/57", "compute_gap null model, alpha=0.6"),
        ("earnings side: fields where max(|c_F|,|c_G|) > sqrt(signal_frac)", f"{sf_viol}/57", "impossible under classical error, so signal_frac is too low there"),
        ("mean ADV* with rel_y = signal_frac (prestige side U, LB, EXT)", f"{ci(Uy)}; {ci(LBy)}; {ci(EXTy)}", "57 fields (dominated by floor fields; see caveats)"),
        ("same, fields with signal_frac >= 0.5", f"{ci(Uy5)}; {ci(LBy5)}; {ci(EXTy5)}", f"{int(Uy5.k_fields)} fields"),
    ]
    for a, b, c in kv:
        L.append(f"| {a} | {b} | {c} |")

    L.append("\n## Per-field reliability vs number of hiring edges\n")
    L.append("Sorted by public non-self hires (m_pub). rel_net = Spearman-Brown split-half reliability of public-edge "
             "SpringRank over all published-ranked institutions in the public field network. rel_F = the same, on "
             "the scripts/28 analysis sample (covered rows). rel_F EXT = extrapolated to m_full. rho_pub = "
             "Spearman(public SpringRank, published Rank) over ranked institutions, and sqrt(rel) is its expected "
             "value if public-edge sampling noise were the only difference. ADV CIs: institution bootstrap, B=1000.\n")
    tb = df.sort_values("m_pub")[["label", "n_inst", "m_pub", "retention", "n_ranked", "n_ranked_in_public",
                                  "rel_net", "rho_recon_pub_net", "sqrt_rel_net", "relF_LB", "relF_EXT", "relG_LB",
                                  "c_F", "c_G", "ADV_U", "ADV_LB", "ADV_LB_lo", "ADV_LB_hi", "ADV_EXT"]].copy()
    tb["ADV* LB [CI]"] = [f"{a:+.2f} [{b:+.2f},{c:+.2f}]" for a, b, c in zip(tb.ADV_LB, tb.ADV_LB_lo, tb.ADV_LB_hi)]
    tb = tb.drop(columns=["ADV_LB", "ADV_LB_lo", "ADV_LB_hi"]).rename(columns={
        "label": "field", "n_inst": "n", "m_pub": "m_pub", "retention": "ret", "n_ranked": "ranked",
        "n_ranked_in_public": "in pub", "rel_net": "rel_net", "rho_recon_pub_net": "rho_pub",
        "sqrt_rel_net": "sqrt(rel)", "relF_LB": "rel_F", "relF_EXT": "rel_F EXT", "relG_LB": "rel_G",
        "ADV_U": "ADV", "ADV_EXT": "ADV* EXT"})
    L.append(tb.to_markdown(index=False, floatfmt=("", ".0f", ".0f", ".2f", ".0f", ".0f", ".2f", ".2f", ".2f",
                                                    ".2f", ".2f", ".3f", "+.2f", "+.2f", "+.2f", "+.2f", "")))
    L.append(f"\nAcademia network (G): m_pub = {acad['m_pub']:,} hires, {acad['n_ranked_in_public']} of "
             f"{acad['n_ranked']} ranked institutions in the public network, rel_net = {acad['rel_net']:.3f} "
             f"(alpha=1: {acad['rel_net_alpha1']:.3f}), rho_pub = {acad['rho_recon_pub_net']:.3f} vs sqrt(rel) = "
             f"{acad['sqrt_rel_net']:.3f}. Restricted to each field's analysis sample, rel_G ranges "
             f"{df.relG_LB.min():.3f}-{df.relG_LB.max():.3f}. Those institutions are better connected than the "
             f"academia-wide tail.")
    L.append(f"\nReliability vs edges: across the 57 field networks rel_net rises with public hires "
             f"(Spearman {rel_m:+.2f}) and with hires per ranked institution (Spearman {rel_mn:+.2f}). The "
             f"Spearman-Brown constant m0 = m(1-rel)/rel (hires needed for rel = 0.5) has median "
             f"{df.m0_net.median():.0f} across fields, vs {acad['m0_net']:,.0f} for the Academia network, which "
             f"has {acad['n_ranked_in_public']} institutions to order. Even so, the Academia network has "
             f"{ratio_med:.0f}x the median field's public hires, and G is the more reliable of the two on "
             f"{int((df.relG_LB > df.relF_LB).sum())}/57 fields' samples.\n")

    L.append("## Method\n")
    L.append(f"1. **Sample.** Exactly the scripts/28 table (`build_table()` + `per_field()` imported from "
             f"scripts/28): 57 fields with >= 8 institutions, {n_rows:,} institution x field rows. F = -published field "
             f"Rank, G = -published Academia Rank (ranks.csv), y = Scorecard 4-yr BA median earnings. c_F, c_G "
             f"are Spearman correlations. Check: c_F equals the compute_gap Spearman to max abs difference "
             f"{chk:.1e}, and the n match in every field: {nchk}.")
    L.append(f"2. **Split-half reliability.** Public edge list, self-hires dropped (they do not enter SpringRank). "
             f"Each person on an edge (Total = number of faculty) is assigned to half A or B with p = 1/2, which is "
             f"binomial thinning of the edge counts. SpringRank from `src/ar_pipeline/springrank.py` (alpha = "
             f"{ALPHA:g}; alpha = {ALPHA_SENS:g} as sensitivity with R = {R_SENS}) is estimated on each half. "
             f"Spearman(half A, half B) is taken over the chosen institutions, averaged over R = {R_SPLIT} seeded "
             f"splits, and stepped up with Spearman-Brown rel = 2r/(1+r). Institutions with no edges in a half "
             f"get a common tied score, and that tie counts as noise. rel_F uses the field network over the analysis "
             f"rows present in it. rel_G uses the Academia network over the same field's analysis rows, so range "
             f"restriction is matched.")
    L.append("3. **Published rank vs reconstruction.** Spearman(full-public-edge SpringRank, -Rank) over ranked "
             "institutions in the public network, compared with sqrt(rel_net). That is the correlation expected if "
             "the published rank were error-free truth and public-edge sampling noise were the only difference. "
             "Ordinal vs continuous SpringRank is irrelevant here because every statistic is rank-based.")
    L.append(f"4. **Published-rank reliability bracket.** LB: the published rank is treated as if estimated from "
             f"the public edges only. Wapman's rank used a superset of these hires, so for covered institutions this "
             f"is expected to be a lower bound on its sampling reliability. EXT: rel(m) = m/(m+m0), with m0 = m_pub(1-rel)/rel from "
             f"the public split-half, evaluated at m_full = m_pub/retention. retention = public non-self hires / "
             f"(stats.csv 'Doctorate (US)' - 'SelfHires'). stats.csv has no row for {n_imp} of the 57 fields, and those "
             f"use the median retention of the {n_stats} fields that have one ({ret_med:.3f}). Three fields are matched by "
             f"name without the ', General' suffix: Physics, Psychology, Biological Sciences. The Academia "
             f"retention is {acad['retention']:.3f}.")
    L.append(f"5. **Disattenuation.** c* = c / sqrt(rel_prestige * rel_y). **Earnings-side assumption (primary): "
             f"rel_y = 1**, i.e. no earnings correction. Within a field, rel_y multiplies c_F and c_G by the same "
             f"factor, so it cannot change which predictor is larger or the sign of ADV. It only rescales "
             f"magnitudes and reweights the across-field mean. As a secondary spec, rel_y = compute_gap signal_frac "
             f"(null model: 1 - mean sampling variance of the median / observed cross-institution variance; "
             f"within-institution CV alpha = 0.6; floor 0.02) on exactly the same rows. This spec fails a "
             f"consistency check (see Checks) and is reported only for completeness.")
    L.append(f"6. **Uncertainty.** Institutions are resampled within field (B = {B_BOOT}). In each draw the "
             f"reliability comes from one randomly chosen split replicate, which propagates split-to-split spread. "
             f"Mean-ADV CIs are percentiles of the across-field mean over draws. The fields themselves are held fixed.")
    L.append("7. **Classical-error check.** On the covered subset, c computed from half-edge reconstructions "
             "should equal c_full * sqrt(r_half / rel_full) if SpringRank noise attenuates earnings correlations "
             "the way classical error does.\n")

    L.append("## Checks\n")
    L.append(f"- **Classical attenuation holds on the public reconstruction.** Across the 57 fields, the mean "
             f"half-edge c_F_rec is {df.cFrec_half_obs.mean():+.3f} observed vs {df.cFrec_half_pred.mean():+.3f} "
             f"predicted, and c_G_rec is {df.cGrec_half_obs.mean():+.3f} vs {df.cGrec_half_pred.mean():+.3f}. So "
             f"Spearman-Brown-scaled reliabilities describe how rank noise attenuates the earnings correlation "
             f"(on average; per-field noise is larger).")
    L.append(f"- **The published rank differs from the public-edge reconstruction by more than public-edge noise "
             f"explains.** Spearman(public SpringRank, published Rank) is below sqrt(rel_net) in {below}/57 fields "
             f"(median {df.rho_recon_pub_net.median():.2f} vs {df.sqrt_rel_net.median():.2f}; Academia "
             f"{acad['rho_recon_pub_net']:.3f} vs {acad['sqrt_rel_net']:.3f}). A median {absent_med:.0%} of "
             f"published-ranked institutions per field do not appear in the public field network at all. "
             f"This fits the suppression of single-hire pairs: the published rank carries information the "
             f"public edges lack. The public-edge reliability is therefore a statement about a public "
             f"reconstruction, not a direct measurement of the published rank's error.")
    L.append(f"- **The published F out-predicts its public reconstruction** on the same institutions. On the "
             f"covered subset, mean c_F is {Uc.mean_cF:+.3f} with the published rank vs {Rc.mean_cF:+.3f} with the "
             f"public SpringRank, and still {RcLB.mean_cF:+.3f} after disattenuating the reconstruction for its own "
             f"noise. Mean c_G barely moves ({Uc.mean_cG:+.3f} vs {Rc.mean_cG:+.3f}). This is consistent with LB "
             f"understating the published rank's reliability. It also means analyses that rebuild field prestige "
             f"from public edges lose field signal: public-only ADV is {ci(Rc)}.")
    L.append(f"- **Covered subset.** Restricted to institutions present in the public field network, with "
             f"reliabilities measured on exactly those institutions, the uncorrected ADV is {ci(Uc)}. It becomes "
             f"{ci(LBc)} under LB and {ci(EXTc)} under EXT. So the conclusion does not hinge on applying "
             f"covered-subset reliabilities to uncovered rows.")
    L.append(f"- **Earnings-side reliability (signal_frac) is not usable as a disattenuation factor here.** "
             f"{sf_floor}/57 fields sit at the 0.02 floor. In {sf_viol}/57 fields the observed max(|c_F|, |c_G|) "
             f"exceeds sqrt(signal_frac), which is impossible if signal_frac were the true reliability. The "
             f"alpha = 0.6 null model overstates sampling noise in those fields, which include large ones, e.g. "
             f"{floor_ex}, both at the floor. Using it "
             f"anyway gives mean c* > 1 ({Uy.mean_cF:.2f}, {Uy.mean_cG:.2f} uncorrected on the prestige side), "
             f"and the across-field mean is dominated by the floor fields (weight 1/sqrt(0.02) = 7.1). "
             f"Restricted to the {int(Uy5.k_fields)} fields with signal_frac >= 0.5 (compute_gap's own "
             f"threshold), the earnings correction gives U {ci(Uy5)}, LB {ci(LBy5)} and EXT {ci(EXTy5)}: the same "
             f"qualitative pattern as the primary spec.")
    L.append(f"- **SpringRank alpha.** With alpha = 1, public-edge rel_F is higher (mean "
             f"{df.relF_LBa1.mean():.3f} vs {df.relF_LB.mean():.3f}), so the LB correction is smaller: mean ADV* = "
             f"{ci(LBa1)}.\n")

    L.append("## Caveats\n")
    L.append("- The central unknown is the reliability of the **published** field rank. Public data identify only "
             "the reliability of a SpringRank rebuilt from singleton-suppressed edges. LB and EXT bracket the "
             "published value under stated assumptions. LB is conservative for covered institutions. It may not "
             f"be conservative for the {1 - n_cov_rows / n_rows:.0%} of analysis rows with no public field edge at "
             f"all: their published rank rests only on unreleased single hires. The covered-subset analysis ({ci(LBc)} under "
             "LB) addresses this. EXT assumes unreleased hires are as informative per person as the released "
             "ones, and uses stats.csv 'Doctorate (US)', which may include hires from outside the ranked network. "
             "Both push EXT up. Narrowing the bracket needs person-level hiring data without the single-pair "
             "suppression (the AARC-based source behind Wapman et al., or another person-level academic-hiring "
             "source). The unknown sits on the prestige side, so additional earnings data would not narrow it.")
    L.append("- Spearman's correction for attenuation is derived for Pearson correlations of linearly related "
             "variables with classical error. Applying it to Spearman correlations and to Spearman split-half "
             "reliabilities is an approximation. The half-edge check above supports it on average, but only for the "
             "step from half to full public edges. It does not test the extrapolation to the published rank.")
    L.append("- Only sampling noise in the prestige estimates is corrected. Construct differences are not measurement "
             "error and are not corrected. Examples: field prestige measures hiring in PhD programs while earnings "
             "are for BA graduates; brand also proxies selectivity and student composition.")
    L.append("- CIs hold the 57 fields fixed and treat institutions as the sampling unit. Field-level CIs are wide "
             "(most per-field ADV* CIs span 0). The across-field mean is the quantity with usable precision.")
    L.append("- Earnings are medians only. Selection vs value-added is not addressed (see FIELD_VS_GENERIC_RESULT.md).")
    L.append("\n## CSV columns (data/interim/prestige_reliability.csv)\n")
    L.append("Row `field == ACADEMIA` is the Academia network (G) over all ranked institutions; other rows are fields. "
             "`m_pub` public non-self hires; `retention`, `m_full_est`, `retention_source` for EXT; `rhalf_net`, "
             "`rel_net`, `rel_net_alpha1`, `rel_net_ext`, `m0_net` network-wide split-half r, Spearman-Brown "
             "reliability, alpha=1 version, extrapolated, SB constant; `rho_recon_pub_net` / `sqrt_rel_net` "
             "reconstruction check; `rho_recon_pub_cov` same on the covered analysis rows; `relF_*`, `relG_*` "
             "reliabilities by scenario (LB, LBa1, EXT, LBcov, EXTcov; `_sd` = split-to-split SD); `cF_<S>`, "
             "`cG_<S>`, `ADV_<S>` (disattenuated) correlations per scenario S in {U, LB, LBa1, EXT, Ucov, LBcov, "
             "EXTcov, RECcov, RECcovLB}; `ADV_<S>_lo/_hi` 95% institution-bootstrap CI; `signal_frac` compute_gap "
             "earnings reliability (alpha=0.6); `breakeven_ratio` = (c_F/c_G)^2; `relratio_LB/EXT` = rel_F/rel_G; "
             "`cFrec_half_obs/_pred`, `cGrec_half_obs/_pred` classical-error check.")
    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
