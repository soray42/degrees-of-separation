"""Faculty-hiring FLOW network — a data-driven disciplinary KINSHIP graph (a graph, not the
CIP tree). Field x field counts of "PhD trained in X, hired into a department of Y" recover
intellectual kinship empirically: who actually exchanges faculty with whom, as opposed to who
the CIP-2020 taxonomy says is "near" whom.

SOURCE. data/interim/orcid_phd_faculty_edges.parquet — one row per ORCID person, the first US
PhD->faculty placement (scripts/10_orcid_build.py), with field_text = "degree_role | dept_from
| dept_to" (dept_from = doctoral department, dept_to = hiring department; lower-cased, "|"-
joined; split on the FIRST and LAST "|" — a handful of rows have an extra "|" inside a dept
string, so dept_from = parts[1] and dept_to = parts[-1], matching exactly how scripts/10 builds
the string). data/interim/orcid_field.parquet (per-person OpenAlex-concept field, route/
confidence) is the other listed source but is NOT used here: it is a different, person-level
taxonomy (OpenAlex concepts), while the cross-field signal this script needs lives entirely in
dept_from vs dept_to within field_text, exactly as specified.

MAPPING. dept_from and dept_to free text -> project field key via src.crosswalks.fields.
match_wapman_field, used AS-IS (no custom text cleanup added on top). That function only checks
TIER0_FIELDS (the curated ~30-field core, not the +24-field EXPANSION set) -- so the resulting
network's node universe is exactly those 30 fields, not the full 54-field project taxonomy.
This matters for the report: 6 of the ~20 "reliable" fields in outputs/expanded66_gap_map.csv
(human_dev, kinesiology, management, marketing, spanish, teacher_ed_subjects) are simply absent
from the flow network -- not sparse, structurally unmatchable by this crosswalk -- and are kept
in faculty_flow_fields.csv (flagged in_flow_network=False) rather than silently dropped.

n_flow[X][Y] = number of persons with PhD-field X hired into a dept-field Y (both dept_from AND
dept_to must resolve to a TIER0 field; rows where either side fails to match are excluded from
the network entirely, not coerced into the diagonal). Saved as a FULL 30x30 grid (incl. zero
cells, so "no observed flow" is explicit) with BOTH the diagonal (within-field re-hiring) and
the off-diagonal (the actual kinship signal) -- the off-diagonal is what scripts/47's "data
scout" lead (Phase 1(c)) and the report below test directly.

Seeded (SEED=42): a person-level bootstrap (B=1000, resample-with-replacement over the matched
edges) gives a CI on (i) total cross-field flow, (ii) the count of reliable-field pairs with
>=1 / >=5 cross-flow, and (iii) how often each top-10 kin pair survives in the resampled top 10
-- the honesty check the task asks for: a thin network makes any single n_flow cell noisy, this
quantifies how noisy.

 -> data/interim/faculty_flow_network.parquet (from_field, to_field, n_flow; full 30x30 grid)
 -> data/interim/faculty_flow_fields.csv (field roster + rho_f/cip2/reliable + flow diagnostics)
Run: `.venv/bin/python scripts/49_faculty_flow_net.py`. Date 2026-06-30.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
from itertools import combinations

import numpy as np
import pandas as pd

from src.crosswalks import fields as F

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
SEED = 42

EDGES_PATH = INTERIM / "orcid_phd_faculty_edges.parquet"
GAPMAP_PATH = ROOT / "outputs" / "expanded66_gap_map.csv"
NET_OUT = INTERIM / "faculty_flow_network.parquet"
FIELDS_OUT = INTERIM / "faculty_flow_fields.csv"

TIER0_KEYS = [f["key"] for f in F.TIER0_FIELDS]
LAB = {f["key"]: f["label"] for f in F.TIER0_FIELDS}


# ============================================================== load/map ===
def parse_depts(field_text: pd.Series) -> tuple[pd.Series, pd.Series]:
    """dept_from = 2nd '|'-segment, dept_to = LAST '|'-segment — matches scripts/10_orcid_build
    construction (role_from | dept_from | dept_to) and is robust to the ~0.04% of rows whose
    dept string itself contains an extra '|' (dept_to is always the final segment)."""
    parts = field_text.str.split("|")
    dept_from = parts.str[1].str.strip()
    dept_to = parts.str[-1].str.strip()
    return dept_from, dept_to


def match_series(s: pd.Series) -> pd.Series:
    """F.match_wapman_field on each distinct dept string (cached — ~thousands of distinct
    strings, tens of thousands of rows)."""
    cache: dict[str, str | None] = {}

    def _match(name: str):
        if name not in cache:
            r = F.match_wapman_field(name)
            cache[name] = r["key"] if r else None
        return cache[name]

    return s.map(_match)


def build_flow():
    e = pd.read_parquet(EDGES_PATH)
    n_persons = len(e)
    assert e.person_orcid.is_unique, "expect one row per person (first faculty job)"

    dept_from, dept_to = parse_depts(e["field_text"])
    from_field = match_series(dept_from)
    to_field = match_series(dept_to)

    from_ok = from_field.notna()
    to_ok = to_field.notna()
    both_ok = from_ok & to_ok

    diag = pd.DataFrame({"from_field": from_field[both_ok], "to_field": to_field[both_ok]})
    counts = diag.groupby(["from_field", "to_field"]).size().reset_index(name="n_flow")

    grid = pd.MultiIndex.from_product([TIER0_KEYS, TIER0_KEYS],
                                       names=["from_field", "to_field"]).to_frame(index=False)
    net = grid.merge(counts, on=["from_field", "to_field"], how="left")
    net["n_flow"] = net["n_flow"].fillna(0).astype(int)

    match_stats = dict(
        n_persons=n_persons,
        n_dept_from_matched=int(from_ok.sum()), n_dept_to_matched=int(to_ok.sum()),
        n_both_matched=int(both_ok.sum()),
        match_rate_from=float(from_ok.mean()), match_rate_to=float(to_ok.mean()),
        match_rate_both=float(both_ok.mean()),
    )
    return net, diag, match_stats


# ============================================================ fields csv ===
def build_fields_table(net: pd.DataFrame, diag_rows: pd.DataFrame) -> pd.DataFrame:
    gm = pd.read_csv(GAPMAP_PATH)[["field", "cip2", "spearman", "gap", "reliable"]]
    gm = gm.rename(columns={"spearman": "rho_f"})

    rows = []
    for f in F.TIER0_FIELDS:
        k = f["key"]
        within = int(net.loc[(net.from_field == k) & (net.to_field == k), "n_flow"].iloc[0])
        out_off = net[(net.from_field == k) & (net.to_field != k)]
        in_off = net[(net.to_field == k) & (net.from_field != k)]
        cross_out, cross_in = int(out_off.n_flow.sum()), int(in_off.n_flow.sum())
        n_partners_out = int((out_off.n_flow > 0).sum())
        n_partners_in = int((in_off.n_flow > 0).sum())
        n_partners = len(set(out_off[out_off.n_flow > 0].to_field) |
                          set(in_off[in_off.n_flow > 0].from_field))
        rows.append(dict(
            field=k, label=f["label"], domain=f["domain"], wapman_field=f["wapman_field"],
            in_flow_network=True,
            n_within=within, n_cross_out=cross_out, n_cross_in=cross_in,
            n_cross_total=cross_out + cross_in,
            n_partners_out=n_partners_out, n_partners_in=n_partners_in, n_partners=n_partners,
        ))
    ft = pd.DataFrame(rows)
    ft = ft.merge(gm, on="field", how="left")

    # 6 "reliable" gap-map fields that match_wapman_field can never resolve (not in TIER0_FIELDS'
    # wapman-label vocabulary, even though some — management, marketing — exist in the broader
    # EXPANSION_FIELDS set that match_wapman_field does not search). Kept, flagged, not hidden.
    rel = gm[gm.reliable].field.tolist()
    missing = [k for k in rel if k not in ft.field.values]
    if missing:
        miss_rows = gm[gm.field.isin(missing)].copy()
        miss_rows["in_flow_network"] = False
        for c in ["n_within", "n_cross_out", "n_cross_in", "n_cross_total",
                  "n_partners_out", "n_partners_in", "n_partners"]:
            miss_rows[c] = 0
        miss_rows["label"] = miss_rows["field"].map(lambda k: k.replace("_", " ").title())
        miss_rows["domain"] = np.nan
        miss_rows["wapman_field"] = np.nan
        ft = pd.concat([ft, miss_rows[ft.columns]], ignore_index=True)

    cols = ["field", "label", "domain", "wapman_field", "cip2", "rho_f", "gap", "reliable",
            "in_flow_network", "n_within", "n_cross_out", "n_cross_in", "n_cross_total",
            "n_partners_out", "n_partners_in", "n_partners"]
    return ft[cols].sort_values(["in_flow_network", "field"], ascending=[False, True]).reset_index(drop=True)


# ============================================================== bootstrap ==
def bootstrap_density(diag_rows: pd.DataFrame, top_pairs: list[tuple[str, str]],
                       B=1000, seed=SEED):
    """Resample matched OFF-DIAGONAL edges with replacement (fixed n -> total cross-flow count
    itself is mechanically invariant, NOT reported); recompute, per draw, (i) the count of
    unordered field pairs with >=1 / >=5 cross flow -- how much the density headline would move
    under resampling noise -- (ii) each top-10 kin pair's resampled n_flow (a per-cell CI) and
    (iii) whether it survives in the resampled top 10. Quantifies how much a thin network
    jitters."""
    rng = np.random.default_rng(seed)
    off = diag_rows[diag_rows.from_field != diag_rows.to_field].reset_index(drop=True)
    a, b = off.from_field.values, off.to_field.values
    n_off = len(off)
    pair_str = np.where(a < b, np.char.add(np.char.add(a, "||"), b),
                         np.char.add(np.char.add(b, "||"), a))

    n_pairs1 = np.empty(B); n_pairs5 = np.empty(B)
    top_keys = ["||".join(sorted(p)) for p in top_pairs]
    top_counts = {p: np.empty(B, dtype=int) for p in top_pairs}
    top_hits = {p: 0 for p in top_pairs}
    for i in range(B):
        idx = rng.integers(0, n_off, n_off)
        vc = pd.Series(pair_str[idx]).value_counts()
        n_pairs1[i] = (vc >= 1).sum()
        n_pairs5[i] = (vc >= 5).sum()
        resampled_top = set(vc.sort_values(ascending=False).head(10).index)
        for p, key in zip(top_pairs, top_keys):
            c = int(vc.get(key, 0))
            top_counts[p][i] = c
            if key in resampled_top:
                top_hits[p] += 1
    return dict(
        pairs1_lo=np.percentile(n_pairs1, 2.5), pairs1_hi=np.percentile(n_pairs1, 97.5),
        pairs5_lo=np.percentile(n_pairs5, 2.5), pairs5_hi=np.percentile(n_pairs5, 97.5),
        top_survival={p: top_hits[p] / B for p in top_pairs},
        top_ci={p: (np.percentile(top_counts[p], 2.5), np.percentile(top_counts[p], 97.5))
                for p in top_pairs},
    )


# =================================================================== misc ==
def unordered_pairs(net: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    idx = net.set_index(["from_field", "to_field"]).n_flow
    for x, y in combinations(keys, 2):
        a_to_b = int(idx.get((x, y), 0)); b_to_a = int(idx.get((y, x), 0))
        rows.append(dict(field_a=x, field_b=y, a_to_b=a_to_b, b_to_a=b_to_a, total=a_to_b + b_to_a))
    return pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)


def cell(net: pd.DataFrame, x: str, y: str) -> int:
    r = net[(net.from_field == x) & (net.to_field == y)]
    return int(r.n_flow.iloc[0]) if len(r) else 0


# ===================================================================== main
def main():
    INTERIM.mkdir(parents=True, exist_ok=True)
    net, diag_rows, ms = build_flow()
    net.to_parquet(NET_OUT, index=False)

    fields_tbl = build_fields_table(net, diag_rows)
    fields_tbl.to_csv(FIELDS_OUT, index=False)

    print(f"persons (edges) total: {ms['n_persons']}")
    print(f"dept_from matched: {ms['n_dept_from_matched']} ({ms['match_rate_from']:.1%}); "
          f"dept_to matched: {ms['n_dept_to_matched']} ({ms['match_rate_to']:.1%}); "
          f"BOTH matched (-> in network): {ms['n_both_matched']} ({ms['match_rate_both']:.1%})")
    print(f"saved {NET_OUT} ({len(net)} rows = {len(TIER0_KEYS)}x{len(TIER0_KEYS)} full grid)")
    print(f"saved {FIELDS_OUT} ({len(fields_tbl)} rows)")

    off = net[net.from_field != net.to_field]
    print(f"\noff-diagonal (cross-field) ordered cells: {len(off)} possible, "
          f"{(off.n_flow > 0).sum()} nonzero, total cross-field flow = {off.n_flow.sum()}")
    print(f"diagonal (within-field) total flow = {net[net.from_field == net.to_field].n_flow.sum()}")

    all_pairs = unordered_pairs(net, TIER0_KEYS)
    print(f"\nunordered field pairs (30 fields, C(30,2)={len(all_pairs)}): "
          f">=1 cross flow: {(all_pairs.total >= 1).sum()}, "
          f">=5 cross flow: {(all_pairs.total >= 5).sum()}, "
          f">=10 cross flow: {(all_pairs.total >= 10).sum()}")
    print("\ntop-10 cross-field kin pairs by total n_flow (both directions summed):")
    top10 = all_pairs.head(10)
    print(top10.assign(label_a=top10.field_a.map(LAB), label_b=top10.field_b.map(LAB))
          .to_string(index=False))

    gm = pd.read_csv(GAPMAP_PATH)
    rel_present = [k for k in TIER0_KEYS if k in set(gm[gm.reliable].field)]
    rel_pairs = unordered_pairs(net, rel_present)
    print(f"\nreliable fields present in network: {len(rel_present)}/20 "
          f"({', '.join(rel_present)})")
    print(f"reliable x reliable unordered pairs: {len(rel_pairs)} "
          f"(C({len(rel_present)},2)); >=1 cross flow: {(rel_pairs.total >= 1).sum()}, "
          f">=5 cross flow: {(rel_pairs.total >= 5).sum()}")

    print("\nstatistics / mathematics / computer_science cross flows:")
    for x, y in [("statistics", "computer_science"), ("statistics", "mathematics"),
                 ("mathematics", "computer_science")]:
        print(f"  {x} -> {y}: {cell(net, x, y)}, {y} -> {x}: {cell(net, y, x)}, "
              f"total: {cell(net, x, y) + cell(net, y, x)}")

    print("\nnursing cross-field rows (off-diagonal only):")
    nr = net[((net.from_field == "nursing") | (net.to_field == "nursing")) &
             (net.from_field != net.to_field) & (net.n_flow > 0)]
    print(nr.to_string(index=False))
    print("\ncommunication_disorders cross-field rows (off-diagonal only):")
    cd = net[((net.from_field == "communication_disorders") | (net.to_field == "communication_disorders")) &
              (net.from_field != net.to_field) & (net.n_flow > 0)]
    print(cd.to_string(index=False))

    # ---------------- bootstrap (seeded) ----------------
    top_pairs = list(zip(top10.field_a, top10.field_b))
    bs = bootstrap_density(diag_rows, top_pairs, B=1000, seed=SEED)
    print(f"\nbootstrap (B=1000, seed={SEED}) 95% CI on density headline (n is fixed by "
          f"resampling, only WHICH pairs clear the bar moves): "
          f"#pairs>=1 [{bs['pairs1_lo']:.0f}, {bs['pairs1_hi']:.0f}] of {len(all_pairs)}; "
          f"#pairs>=5 [{bs['pairs5_lo']:.0f}, {bs['pairs5_hi']:.0f}]")
    print("top-10 pair: resampled n_flow 95% CI, and survival rate (fraction of 1000 draws "
          "the pair is still in the resampled top 10):")
    for p, rate in bs["top_survival"].items():
        lo, hi = bs["top_ci"][p]
        print(f"  {LAB[p[0]]} <-> {LAB[p[1]]}: n_flow 95% CI [{lo:.0f}, {hi:.0f}], "
              f"survival={rate:.2f}")


if __name__ == "__main__":
    main()
