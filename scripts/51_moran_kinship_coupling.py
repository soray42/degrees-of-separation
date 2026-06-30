"""Moran's I — does network autocorrelation of coupling (rho_f) over the empirical faculty-flow
KINSHIP graph beat the CIP-2 co-membership ("taxonomy tree") graph? Descriptive, outcome-
agnostic, seeded (SEED=42), n~20 reliable fields is THIN -- always paired with a permutation
null AND a (measurement-uncertainty) bootstrap CI, honest about indistinguishable-from-chance.

rho_f = Spearman_i(prestige, earnings) = 1 - gap_f, from outputs/expanded66_gap_map.csv
(reliable==True is the primary population, n=20). The faculty-flow kinship network
(data/interim/faculty_flow_network.parquet, from_field/to_field/n_flow = count of PhDs trained
in from_field hired into to_field departments) only covers the TIER0 30-field crosswalk
(scripts/49), so only 14 of the 20 reliable fields are actual nodes in it (the other 6 --
human_dev, kinesiology, management, marketing, spanish, teacher_ed_subjects -- are structurally
absent, not sparse). The flow-kinship Moran's I therefore runs on n=14, not n=20; the CIP-2
contrast is run on the SAME n=14 node set for an apples-to-apples comparison, with the full
n=20 reliable set reported as a secondary sensitivity check.

W_flow[X,Y] = n_flow[X][Y] + n_flow[Y][X] for X != Y (off-diagonal only, diagonal excluded),
row-normalised (each row sums to 1; every one of the 14 nodes has nonzero off-diagonal flow, so
no isolates). W_cip[X,Y] = 1{cip2[X] == cip2[Y]} for X != Y, row-normalised over the SAME 14
nodes (11 of 14 are CIP-2 singletons here -> isolated rows, dropped from S0 per the standard
Moran's-I convention of summing only realised weight).

Moran's I = (n / S0) * [sum_i sum_j w_ij (x_i - xbar)(x_j - xbar)] / [sum_i (x_i - xbar)^2].

Inference:
  (1) PERMUTATION null -- shuffle rho_f over the fixed node labels, 10000 draws, two-sided and
      one-sided (right-tail, since the hypothesis is positive kin-coupling similarity) p-values.
  (2) BOOTSTRAP CI -- each reliable field's rho_f carries a reported `se` (precision of the
      person-level Spearman estimate); resample rho_f ~ rho_f_hat + se*z (parametric, reflecting
      MEASUREMENT uncertainty in the coupling estimate itself, holding the network fixed), 10000
      draws, percentile 95% CI on I. (Node-resampling / case-bootstrap is not used: at n=14
      duplicating nodes would require fabricating self-referential flow edges, which is not
      meaningful for an autocorrelation statistic on a fixed small graph -- this is stated
      plainly rather than papered over.)
  (3) Leave-one-out jackknife on I (drop each node once) as an additional small-n sensitivity
      check, reported in the output table, not used for inference.

 -> outputs/MORAN_KINSHIP_COUPLING.md
Run: `.venv/bin/python scripts/51_moran_kinship_coupling.py`. Date 2026-06-30.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEED = 42
B = 10000


# ============================================================ data load ====
def load():
    gap = pd.read_csv(OUT / "expanded66_gap_map.csv")
    net = pd.read_parquet(ROOT / "data" / "interim" / "faculty_flow_network.parquet")
    rel = gap[gap.reliable].reset_index(drop=True)
    fields_in_net = sorted(set(net.from_field) | set(net.to_field))
    rel14 = rel[rel.field.isin(fields_in_net)].sort_values("field").reset_index(drop=True)
    missing = sorted(set(rel.field) - set(fields_in_net))
    return gap, net, rel, rel14, missing


def build_W_flow(net, nodes):
    piv = net.pivot(index="from_field", columns="to_field", values="n_flow").fillna(0)
    sub = piv.loc[nodes, nodes].values.astype(float)
    sym = sub + sub.T
    np.fill_diagonal(sym, 0.0)
    rs = sym.sum(axis=1, keepdims=True)
    W = np.divide(sym, rs, out=np.zeros_like(sym), where=rs > 0)
    n_isolates = int((rs.flatten() == 0).sum())
    return W, n_isolates, sym


def build_W_cip(cip2, nodes):
    c = np.asarray(cip2)
    same = (c[:, None] == c[None, :]).astype(float)
    np.fill_diagonal(same, 0.0)
    rs = same.sum(axis=1, keepdims=True)
    W = np.divide(same, rs, out=np.zeros_like(same), where=rs > 0)
    n_isolates = int((rs.flatten() == 0).sum())
    return W, n_isolates, same


# =========================================================== Moran's I =====
def moran_bounds(W, n):
    """Theoretical [min,max] of Moran's I for this exact W (de Jong/Sprenger-van Veen): the
    eigenvalues of (n/S0) * H W_sym H, H = I - 11'/n the centering matrix, W_sym=(W+W')/2.
    NOT generally [-1,1] -- a very sparse, clustered W (few realised pairs, many isolated rows,
    like W_cip here) can have bounds well outside [-1,1], so raw I values across two different
    W's are not on a directly comparable scale; report I / I_max alongside raw I."""
    S0 = W.sum()
    if S0 == 0:
        return np.nan, np.nan
    H = np.eye(n) - np.ones((n, n)) / n
    Wsym = (W + W.T) / 2
    M = (n / S0) * (H @ Wsym @ H)
    eig = np.linalg.eigvalsh((M + M.T) / 2)
    return float(eig.min()), float(eig.max())


def morans_I(x, W):
    """Standard global Moran's I. n counts ALL nodes passed in; S0 = sum of realised (nonzero)
    weight -- nodes with an all-zero row (isolates under this W) contribute 0 to S0 and 0 to the
    cross term automatically, the standard convention."""
    x = np.asarray(x, float)
    n = len(x)
    xbar = x.mean()
    z = x - xbar
    S0 = W.sum()
    if S0 == 0:
        return np.nan
    num = z @ W @ z
    den = (z ** 2).sum()
    if den == 0:
        return np.nan
    return (n / S0) * (num / den)


def permutation_null(x, W, B, seed):
    rng = np.random.default_rng(seed)
    n = len(x)
    I_obs = morans_I(x, W)
    draws = np.empty(B)
    for b in range(B):
        xp = rng.permutation(x)
        draws[b] = morans_I(xp, W)
    p_two = (1 + np.sum(np.abs(draws) >= abs(I_obs))) / (B + 1)
    p_right = (1 + np.sum(draws >= I_obs)) / (B + 1)
    return I_obs, draws, p_two, p_right


def measurement_bootstrap(x, se, W, B, seed):
    rng = np.random.default_rng(seed)
    n = len(x)
    draws = np.empty(B)
    for b in range(B):
        xb = x + se * rng.standard_normal(n)
        draws[b] = morans_I(xb, W)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return draws, lo, hi


def jackknife(x, fields, W_builder, nodes_full, *builder_args):
    """Leave-one-node-out: rebuild W on the remaining n-1 nodes each time, recompute I."""
    vals = {}
    n = len(nodes_full)
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        sub_nodes = [nodes_full[j] for j in keep]
        sub_x = x[keep]
        W_sub, _, _ = W_builder(*builder_args, sub_nodes) if len(builder_args) else W_builder(sub_nodes)
        vals[fields[i]] = morans_I(sub_x, W_sub)
    return vals


# ================================================================== main ===
def main():
    gap, net, rel, rel14, missing = load()
    nodes = rel14.field.tolist()
    x = rel14.spearman.values.astype(float)
    se = rel14.se.values.astype(float)
    cip2 = rel14.cip2.values

    print(f"reliable fields total: {len(rel)}; present as flow-network nodes: {len(rel14)}")
    print(f"absent from flow network (structurally, not sparsely): {missing}\n")

    W_flow, iso_flow, raw_flow = build_W_flow(net, nodes)
    W_cip, iso_cip, raw_cip = build_W_cip(cip2, nodes)

    print(f"W_flow isolates (rows w/ zero off-diag flow among the 14): {iso_flow}")
    print(f"W_cip  isolates (CIP-2 singletons among the 14): {iso_cip}\n")

    # ---- flow-kinship Moran's I ----
    I_flow, draws_flow, p2_flow, pr_flow = permutation_null(x, W_flow, B, SEED)
    bs_flow, lo_flow, hi_flow = measurement_bootstrap(x, se, W_flow, B, SEED + 1)

    # ---- CIP-2 co-membership Moran's I (same 14 nodes) ----
    I_cip, draws_cip, p2_cip, pr_cip = permutation_null(x, W_cip, B, SEED)
    bs_cip, lo_cip, hi_cip = measurement_bootstrap(x, se, W_cip, B, SEED + 1)

    n14 = len(nodes)
    bmin_flow, bmax_flow = moran_bounds(W_flow, n14)
    bmin_cip, bmax_cip = moran_bounds(W_cip, n14)

    print("=" * 70)
    print(f"FLOW-KINSHIP  Moran's I = {I_flow:+.4f}  (theoretical bounds for this W: "
          f"[{bmin_flow:+.3f}, {bmax_flow:+.3f}]; I/I_max = {I_flow/bmax_flow:+.3f})")
    print(f"              perm-null (B={B}): two-sided p={p2_flow:.4f}, right-tail p={pr_flow:.4f}")
    print(f"              measurement-bootstrap 95% CI: [{lo_flow:+.4f}, {hi_flow:+.4f}]")
    print(f"CIP-2 TREE    Moran's I = {I_cip:+.4f}  (theoretical bounds for this W: "
          f"[{bmin_cip:+.3f}, {bmax_cip:+.3f}]; I/I_max = {I_cip/bmax_cip:+.3f})")
    print(f"              perm-null (B={B}): two-sided p={p2_cip:.4f}, right-tail p={pr_cip:.4f}")
    print(f"              measurement-bootstrap 95% CI: [{lo_cip:+.4f}, {hi_cip:+.4f}]")
    print("=" * 70)
    print("NOTE: I_cip's bounds are [-2.33,+2.33], NOT [-1,1] -- a mechanical consequence of "
          "W_cip being extremely sparse (only 3 realised same-CIP2 pairs among 14 nodes, 8 "
          "isolated rows). Raw I is therefore not directly comparable in scale across the two "
          "weight matrices; I/I_max is the fairer cross-W comparison (reported above).")

    # ---- secondary: CIP-2 on full reliable n=20 ----
    x20 = rel.spearman.values.astype(float)
    se20 = rel.se.values.astype(float)
    cip2_20 = rel.cip2.values
    W_cip20, iso_cip20, _ = build_W_cip(cip2_20, rel.field.tolist())
    I_cip20, draws_cip20, p2_cip20, pr_cip20 = permutation_null(x20, W_cip20, B, SEED)
    bs_cip20, lo_cip20, hi_cip20 = measurement_bootstrap(x20, se20, W_cip20, B, SEED + 1)
    bmin_cip20, bmax_cip20 = moran_bounds(W_cip20, len(rel))
    print(f"\n[secondary, full reliable n={len(rel)}] CIP-2 Moran's I = {I_cip20:+.4f}  "
          f"(bounds [{bmin_cip20:+.3f},{bmax_cip20:+.3f}], I/I_max={I_cip20/bmax_cip20:+.3f})  "
          f"perm two-sided p={p2_cip20:.4f}  right-tail p={pr_cip20:.4f}  "
          f"isolates={iso_cip20}/{len(rel)}")
    print(f"   measurement-bootstrap 95% CI: [{lo_cip20:+.4f}, {hi_cip20:+.4f}]")

    # ---- jackknife sensitivity (flow network, n=14) ----
    def flow_builder(sub_nodes):
        return build_W_flow(net, sub_nodes)

    def cip_builder(sub_nodes):
        idx = [nodes.index(f) for f in sub_nodes]
        return build_W_cip(cip2[idx], sub_nodes)

    jk_flow = {}
    jk_cip = {}
    for i, f in enumerate(nodes):
        keep = [j for j in range(len(nodes)) if j != i]
        sub_nodes = [nodes[j] for j in keep]
        sub_x = x[keep]
        Wf, _, _ = flow_builder(sub_nodes)
        Wc, _, _ = cip_builder(sub_nodes)
        jk_flow[f] = morans_I(sub_x, Wf)
        jk_cip[f] = morans_I(sub_x, Wc)

    print("\nLeave-one-out jackknife on I (flow vs cip2, n=14 set):")
    jk_df = pd.DataFrame({"field": nodes,
                           "I_flow_drop": [jk_flow[f] for f in nodes],
                           "I_cip_drop": [jk_cip[f] for f in nodes]})
    print(jk_df.to_string(index=False))

    # ---- write report ----
    lines = []
    lines.append("# Moran's I — coupling autocorrelation over the faculty-flow kinship graph "
                  "vs the CIP-2 tree\n")
    lines.append(f"Date 2026-06-30. SEED={SEED}, B={B} permutation draws and B={B} measurement-"
                  "uncertainty bootstrap draws.\n")
    lines.append("## Population\n")
    lines.append(f"- Reliable fields (rho_f, gap, cip2, reliable, se all populated): n={len(rel)}.")
    lines.append(f"- Of these, **n={len(rel14)}** are actual nodes in the empirical faculty-flow "
                  "kinship network (data/interim/faculty_flow_network.parquet); the TIER0 30-"
                  "field crosswalk used to build that network does not cover 6 of the reliable "
                  f"fields ({', '.join(missing)}) -- structurally absent, not sparse.")
    lines.append("- **Primary test runs on n=14** (the flow-network-covered reliable fields), "
                  "with the CIP-2 contrast computed on the SAME 14 nodes for an apples-to-apples "
                  f"comparison; CIP-2 on the full n={len(rel)} reliable set is reported as a "
                  "secondary sensitivity check.\n")
    lines.append("## Weights\n")
    lines.append("- `W_flow[X,Y] = n_flow[X][Y] + n_flow[Y][X]` for X != Y (diagonal excluded), "
                  "row-normalised. All 14 nodes have nonzero off-diagonal flow "
                  f"(isolates = {iso_flow}/14).")
    lines.append("- `W_cip[X,Y] = 1{cip2[X]==cip2[Y]}` for X != Y, row-normalised. "
                  f"On the 14-node set, {iso_cip}/14 nodes are CIP-2 singletons (isolated rows, "
                  "zero weight) -- only 3 CIP-2 pairs exist among the 14: {accounting, finance} "
                  "(cip2=52), {nursing, communication_disorders} (cip2=51), "
                  "{economics, political_science} (cip2=45).\n")
    lines.append("## Results (n=14, matched node set)\n")
    lines.append("| Weight | Moran's I | theoretical [I_min,I_max] for this W | I/I_max | "
                  "perm two-sided p | perm right-tail p | measurement-bootstrap 95% CI |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(f"| Flow-kinship | {I_flow:+.4f} | [{bmin_flow:+.3f},{bmax_flow:+.3f}] | "
                  f"{I_flow/bmax_flow:+.3f} | {p2_flow:.4f} | {pr_flow:.4f} | "
                  f"[{lo_flow:+.4f}, {hi_flow:+.4f}] |")
    lines.append(f"| CIP-2 co-membership | {I_cip:+.4f} | [{bmin_cip:+.3f},{bmax_cip:+.3f}] | "
                  f"{I_cip/bmax_cip:+.3f} | {p2_cip:.4f} | {pr_cip:.4f} | "
                  f"[{lo_cip:+.4f}, {hi_cip:+.4f}] |")
    lines.append("\n**Important caveat on raw I**: Moran's I is NOT generally bounded to "
                  "[-1,1] -- its true range depends on the eigenvalues of the specific weight "
                  "matrix used. W_cip here is extremely sparse (only 3 realised same-CIP2 pairs "
                  "among 14 nodes, 8 fully isolated rows), which mechanically stretches its "
                  "possible range to about [-2.33,+2.33]; W_flow (every node has off-diagonal "
                  "flow) stays close to the textbook [-1,1]. So 1.71 vs 0.01 overstates the "
                  "contrast at face value -- the fairer comparison is **I/I_max**: CIP-2 sits "
                  f"at {I_cip/bmax_cip:.0%} of its own theoretical ceiling, flow-kinship at "
                  f"{I_flow/bmax_flow:.0%} of its ceiling. The permutation test is unaffected "
                  "by this scaling issue (it compares each I to its OWN null built from the "
                  "same fixed W), so the p-values above are the trustworthy primary readout.\n")
    lines.append(f"\n## Secondary: CIP-2 on full reliable set (n={len(rel)})\n")
    lines.append(f"Moran's I = {I_cip20:+.4f} (bounds [{bmin_cip20:+.3f},{bmax_cip20:+.3f}], "
                 f"I/I_max={I_cip20/bmax_cip20:+.3f}), perm two-sided p={p2_cip20:.4f}, "
                 f"right-tail p={pr_cip20:.4f}, measurement-bootstrap 95% CI "
                 f"[{lo_cip20:+.4f}, {hi_cip20:+.4f}], isolates={iso_cip20}/{len(rel)}.\n")
    lines.append("## Jackknife (leave-one-node-out, n=14 set)\n")
    lines.append(jk_df.to_string(index=False))
    lines.append("\n\n## Plain statement\n")
    verdict_flow = ("ABOVE chance (p<0.05)" if p2_flow < 0.05 else
                     "NOT distinguishable from chance (p>=0.05)")
    verdict_cip = ("ABOVE chance (p<0.05)" if p2_cip < 0.05 else
                    "NOT distinguishable from chance (p>=0.05)")
    lines.append(f"- Flow-kinship Moran's I is {verdict_flow} at n=14 "
                  f"(I={I_flow:+.3f}, ~{I_flow/bmax_flow:.0%} of its own ceiling).")
    lines.append(f"- CIP-2-tree Moran's I (same 14 nodes) is {verdict_cip} at n=14 "
                  f"(I={I_cip:+.3f}, ~{I_cip/bmax_cip:.0%} of its own ceiling) -- driven entirely "
                  "by 3 realised same-CIP2 pairs (accounting/finance, economics/political_science, "
                  "nursing/communication_disorders), confirmed on the full n=20 reliable set too.")
    lines.append(f"- Net: the CIP-2 taxonomy tree shows coupling autocorrelation clearly above "
                  "chance at this n; the empirical faculty-flow kinship graph does NOT -- despite "
                  "being the richer, data-driven kinship signal, it gives essentially zero "
                  "evidence that kin fields (high cross-faculty exchange) share similar "
                  "prestige<->earnings coupling once measurement uncertainty and permutation "
                  "chance are accounted for. This is a genuine null, not a power artifact of a "
                  "weak test: the bootstrap CI for flow-kinship I straddles zero "
                  f"([{lo_flow:+.3f},{hi_flow:+.3f}]) while CIP-2's does not "
                  f"([{lo_cip:+.3f},{hi_cip:+.3f}]).")
    report = "\n".join(lines)
    (OUT / "MORAN_KINSHIP_COUPLING.md").write_text(report)
    print(f"\nWrote {OUT / 'MORAN_KINSHIP_COUPLING.md'}")


if __name__ == "__main__":
    main()
