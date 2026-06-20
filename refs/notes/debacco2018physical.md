# debacco2018physical
**Full citation:** De Bacco, C., Larremore, D. B., & Moore, C. (2018). A physical model for efficient ranking in networks. *Science Advances*, 4(7), eaar8260.
**DOI / URL:** doi:10.1126/sciadv.aar8260 — open access at https://www.science.org/doi/10.1126/sciadv.aar8260 ; preprint arXiv:1709.09002 ; reference code https://github.com/LarremoreLab/SpringRank and https://github.com/cdebacco/SpringRank
**Access level:** full text (abstract + algorithm verified from arXiv listing and both reference-implementation source files; the publisher landing page returned HTTP 403, but the algorithm details below are read directly from the canonical source code authored by the paper authors)

## What it contributes to THIS project
This is the engine that produces the Academic Reputation (AR) axis. The whole project defines AR as a field-specific *prestige rank* of institutions inferred from the faculty-hiring network (who-trains-whom edges), and SpringRank is exactly the method we run on that directed network to assign each institution a real-valued rank `s_i`. It is the same algorithm Wapman et al. 2022 use on US faculty hiring (our validation target) and the same family of prestige estimation we replicate for the 2026 panel via the ORCID+OpenAlex hiring pipeline. Crucially, SpringRank ships with an *inverse temperature* `beta` that converts ranks into edge-direction probabilities, which lets us turn raw ranks into a calibrated prestige/probability scale and lets us bootstrap significance — useful when we compute `Gap_field = 1 - Spearman(prestige_rank, ER_rank)` and need confidence intervals on the AR ranking. SpringRank is the concrete Step-5 input that the Gathmann-Schoenberg task-distance hypothesis is ultimately regressed against (prestige_rank vs. ER_rank gap per field).

## Specific equation / result / dataset we reuse
We reuse the full SpringRank estimator, implemented via `pip install springrank` (LarremoreLab/SpringRank, sparse scipy version — this is the implementation we adopt for Step 5; cdebacco/SpringRank is the original Python/MATLAB/SAS reference).

**Energy / Hamiltonian.** Each directed edge `A_ij` (i -> j) is a spring with rest length 1; the optimal ranks minimize
H(s) = (1/2) * sum_{i,j} A_ij * (s_i - s_j - 1)^2.
Edges pull a node one unit above the node it points to; ranks are continuous (real-valued), not ordinal.

**Linear system (the load-bearing step we lift).** Minimizing H gives a sparse linear system. With
- `k_out` = out-degree = row sums of A  (`A.sum(axis=1)`)
- `k_in`  = in-degree  = column sums of A (`A.sum(axis=0)`)
the system matrix and RHS built in `_solve_springrank` are:
  operator = diag(k_out + k_in) - (A + A^T)
  b        = (k_out - k_in)
i.e. solve  [ diag(k_out+k_in) - (A + A^T) ] s = (k_out - k_in).
This matrix is a (weighted) graph Laplacian and is singular by one global translation (ranks fixed up to an additive constant), so the implementation appends a Lagrange-multiplier row/column: operator is expanded to (N+1, N+1) with `operator[N,0]=1`, `operator[0,N]=1`, and b is appended with a 0, pinning the gauge. The system is solved with a sparse solver (scipy), so it is O(network sparsity) and scales to the full national hiring network.

**Regularization.** The L2-regularized variant (`_solve_springrank_regularized`) adds `alpha * I`:
  operator = alpha * I + diag(k_out + k_in) - (A + A^T),   b = (k_out - k_in).
`alpha` (default 0) shrinks ranks toward 0 and makes the system strictly positive-definite (no gauge row needed) — we set a small alpha>0 for numerical stability on sparse or near-disconnected field subnetworks.

**Inverse temperature beta (ranks -> probability scale).** The generative model says the probability that an edge between i and j points i -> j is logistic in the rank gap:
  P_ij(beta) = 1 / (1 + exp(-2*beta*(s_i - s_j))).
beta is fit by maximum likelihood. The "global" fit (`get_inverse_temperature`, paper Eq. S39) root-solves
  sum_{i,j} (s_i - s_j) * [ A_ij - (A_ij + A_ji) / (1 + exp(-2*beta*(s_i - s_j))) ] = 0
(equivalently the tanh form: A_ij - (A_ij+A_ji)*sigmoid(...) ), found via a 1-D root solver, with a `max_beta` cap (default 20) to avoid divergence on perfectly hierarchical fields. A "local" alternative minimizes mean absolute edge-prediction error,
  beta_hat = argmin_beta sum_{i,j} | A_ij - (A_ij+A_ji)/(1 + exp(-2*beta*(s_i - s_j))) |.
Large beta = sharp, strongly hierarchical field (steep prestige ladder); small beta = flat field. We use beta both to (a) map AR ranks onto a 0-1 prestige/probability scale comparable across fields and (b) report how "steep" each field's academic hierarchy is, a covariate alongside the task-distance predictor.

**API we call (Step 5):** `m = SpringRank(alpha=...); m.fit(A); ranks = m.ranks; beta = m.get_beta(); scaled = m.get_scaled_ranks(win_rate); m.predict([i,j])`.
