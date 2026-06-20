# Tier 0 — P1 result

**P1 (proposal v2 §4.3 / H2):** the gap is *smaller* where graduate productivity is more observable, proxied by within-field earnings dispersion. **Predicted sign: negative.** Run: `python scripts/03_p1.py`. Date 2026-06-20.

## Spearman(gap, dispersion)

| proxy | fields | n | Spearman | p | sign | holds (neg & p<.10) |
|---|---|---|---|---|---|---|
| PSEO within-inst (primary) | reliable only | 14 | -0.279 | 0.334 | negative ✓ | False |
| PSEO within-inst (primary) | all (incl. unreliable) | 28 | +0.134 | 0.496 | positive ✗ | False |
| cross-inst CV (robustness) | reliable only | 14 | -0.270 | 0.350 | negative ✓ | False |
| cross-inst CV (robustness) | all (incl. unreliable) | 28 | -0.233 | 0.234 | negative ✓ | False |

## Verdict on P1

- On **reliable fields**, the sign is **negative on both proxies** (PSEO primary -0.28, p=0.33; CV robustness -0.27, p=0.35) — the **predicted direction**, consistent with the prior diagnostic (≈ −0.41), **but not statistically significant** at n=14.
- It is **fragile**: including unreliable fields **flips the primary PSEO proxy to +0.13** (wrong sign) while the CV proxy stays weakly negative. Noise-dominated fields (chemistry, earth sciences, several engineering branches) must be excluded for the sign to hold.
- **Conclusion: P1 holds in *direction* but is *not confirmed*** — correct-signed, underpowered, and dependent on the reliability gate. This is 'initial support', not a clean prediction success. See `figures/p1_scatter.png`.
