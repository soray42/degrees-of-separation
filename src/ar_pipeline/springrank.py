"""SpringRank (De Bacco, Larremore & Moore 2018, Sci. Adv.) — minimal, self-contained.

Ranks solve the regularized linear system derived from minimizing the spring energy
    H(s) = (1/2) Σ_ij A_ij (s_i − s_j − 1)^2 :
    [α I + (D_out + D_in) − (A + A^T)] s = (d_out − d_in),
with A_ij = weight of edge i→j, d_out_i = Σ_j A_ij, d_in_i = Σ_j A_ji. α>0 breaks the
constant-shift degeneracy (Gaussian prior); we then center s. Used here to (i) reproduce the
Wapman per-field prestige ranks from the released edge lists, validating the implementation,
and (ii) re-estimate ranks on multinomially-resampled edges for the generated-regressor
bootstrap (proposal §8: SpringRank A is an estimated regressor).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import spsolve


def springrank(A: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    """SpringRank scores s for a dense or sparse adjacency A (A[i,j] = i→j weight).
    Higher s = more prestigious (sources of placement sit above their targets)."""
    A = csr_matrix(A, dtype=float)
    n = A.shape[0]
    d_out = np.asarray(A.sum(axis=1)).flatten()
    d_in = np.asarray(A.sum(axis=0)).flatten()
    L = diags(d_out + d_in) - (A + A.T)
    M = alpha * identity(n) + L
    b = d_out - d_in
    s = spsolve(csr_matrix(M), b)
    return s - s.mean()


def adjacency_from_edges(edges: pd.DataFrame, src="DegreeInstitutionName",
                         dst="InstitutionName", w="Total"):
    """Build a per-field adjacency over the union of institution names. Drops self-loops.
    Returns (A dense, node list)."""
    e = edges[[src, dst, w]].dropna()
    e = e[e[src] != e[dst]]
    nodes = sorted(set(e[src]) | set(e[dst]))
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for s_, d_, ww in zip(e[src], e[dst], pd.to_numeric(e[w], errors="coerce").fillna(0)):
        A[idx[s_], idx[d_]] += ww
    return A, nodes


def field_springrank(edges_field: pd.DataFrame, alpha: float = 1e-3) -> pd.DataFrame:
    """SpringRank prestige for one field's edge list. Returns inst, s, and prestige_rank
    (1 = most prestigious). Orientation is fixed downstream by validating against Wapman."""
    A, nodes = adjacency_from_edges(edges_field)
    s = springrank(A, alpha=alpha)
    out = pd.DataFrame({"institution_name": nodes, "s": s})
    out["prestige_rank"] = out["s"].rank(ascending=False, method="average")
    return out
