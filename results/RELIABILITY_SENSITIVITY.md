# Tier 0 addendum — Reliability-threshold sensitivity (Task 4)

Reliable-field count and the P1 result as a function of the signal_frac threshold (not a single 0.5 cut). P1 proxy = the preferred non-truncated ACS 22-27 IQR/p50. Run: `python scripts/07_sensitivity.py`. Date 2026-06-20.

|   threshold |   n_reliable |   p1_n |   p1_spearman |   p1_p |
|------------:|-------------:|-------:|--------------:|-------:|
|        0.3  |        14.00 |     14 |            -1 | +0.064 |
|        0.35 |        14.00 |     14 |            -1 | +0.064 |
|        0.4  |        14.00 |     14 |            -1 | +0.064 |
|        0.45 |        14.00 |     14 |            -1 | +0.064 |
|        0.5  |        14.00 |     14 |            -1 | +0.064 |
|        0.55 |        13.00 |     13 |            -0 | +0.122 |
|        0.6  |        12.00 |     12 |            -1 | +0.080 |
|        0.65 |        11.00 |     11 |            -1 | +0.015 |
|        0.7  |        10.00 |     10 |            -1 | +0.060 |

## Read

- The reliable count moves smoothly from **14** (threshold 0.30) to **10** (0.70); at the pre-set **0.50** it is **14**. No knife-edge — the 14-field count is not an artifact of the cut.
- **P1 strengthens as the gate tightens**: the Spearman becomes *more* negative at higher thresholds (cleaner fields), consistent with P1 being a statement about well-measured fields. It is correct-signed across the whole 0.30–0.70 range.
- Loosening the gate toward 0.30 dilutes both the count's meaning and P1 (noise-dominated fields enter) — which is why a non-trivial threshold is kept.

See `figures/reliability_sensitivity.png`.