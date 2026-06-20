# License-Dependent Field Handling + P1 Identification Check

Outcome-agnostic. License tags are **ex-ante** (required-to-practice credential), never from the gap. Reliable set = 0.50 gate. Run: `python scripts/12_license.py`. Date 2026-06-20.

## Ex-ante license tags (reliable set)

| label                   |   gap |   within_field_spearman |   signal_frac | license   | regime                  |
|:------------------------|------:|------------------------:|--------------:|:----------|:------------------------|
| Computer Science        |  0.29 |                    0.71 |          0.92 | False     | prestige-transmission   |
| Economics               |  0.31 |                    0.69 |          0.90 | False     | prestige-transmission   |
| Political Science       |  0.34 |                    0.66 |          0.72 | False     | PhD-pipeline            |
| Mathematics             |  0.38 |                    0.62 |          0.87 | False     | PhD-pipeline            |
| Finance                 |  0.41 |                    0.59 |          0.94 | False     | prestige-transmission   |
| History                 |  0.41 |                    0.59 |          0.73 | False     | prestige-transmission   |
| Psychology              |  0.43 |                    0.57 |          0.77 | False     | PhD-pipeline            |
| Marketing               |  0.48 |                    0.52 |          0.86 | False     | prestige-transmission   |
| Electrical Engineering  |  0.49 |                    0.51 |          0.71 | False     | prestige-transmission   |
| Accounting              |  0.49 |                    0.51 |          0.83 | True      | license-standardization |
| English                 |  0.51 |                    0.49 |          0.58 | False     | prestige-transmission   |
| Management              |  0.52 |                    0.48 |          0.85 | False     | prestige-transmission   |
| Biology                 |  0.65 |                    0.35 |          0.70 | False     | PhD-pipeline            |
| Philosophy              |  0.66 |                    0.34 |          0.60 | False     | PhD-pipeline            |
| Nursing                 |  0.93 |                    0.07 |          0.67 | True      | license-standardization |
| Communication Disorders |  1.21 |                   -0.21 |          0.51 | True      | license-standardization |

Civil Engineering is tagged LICENSE (PE) but is **unreliable** (dropped by the gate) — carried as a dimension only. No reliable field falls outside the hand-coded mapping.

## Task 1 — does the license compress cross-institution earnings?

Cross-institution earnings dispersion (Scorecard inst×field — the gap's own earnings):

| group | n | CV | IQR/median | ACS early-career IQR/p50 (the P1 x) |
|---|---|---|---|---|
| LICENSE | 3 | mean=0.236 median=0.214 | mean=0.265 median=0.256 | mean=0.602 median=0.602 |
| NON-LICENSE | 13 | mean=0.216 median=0.204 | mean=0.228 median=0.226 | mean=0.724 median=0.713 |

Per-field (license fields + the FLAT vs DECOUPLED call):

| label                   |   xinst_cv |   xinst_iqr_med |   acs_iqr |   spearman |   signal_frac |
|:------------------------|-----------:|----------------:|----------:|-----------:|--------------:|
| Accounting              |      0.214 |           0.256 |     0.602 |       0.51 |          0.83 |
| Nursing                 |      0.186 |           0.158 |     0.523 |       0.07 |          0.67 |
| Communication Disorders |      0.308 |           0.380 |     0.680 |      -0.21 |          0.51 |

**FLAT vs DECOUPLED** (FLAT = low cross-inst dispersion; DECOUPLED = dispersion present but within-field Spearman(prestige,earnings) ≤ 0). Genuinely-flat fields are auto-excluded by the reliability gate (flat → low signal_frac), so reliable license fields should be DECOUPLED, not flat:
- **Accounting**: CV=0.214, within-Spearman=+0.51, signal_frac=0.83 → **COUPLED (prestige still tracks earnings)**
- **Nursing**: CV=0.186, within-Spearman=+0.07, signal_frac=0.67 → **DECOUPLED**
- **Communication Disorders**: CV=0.308, within-Spearman=-0.21, signal_frac=0.51 → **DECOUPLED**

*(Footnote: Mann–Whitney CV license-vs-non-license p=0.80; n≈3 license fields — descriptive only, no significance claimed.)*

## Task 2 — P1 identification check (gap ~ early-career dispersion; expect negative)

| spec | fields | n | Spearman / slope | 95% bootstrap CI |
|---|---|---|---|---|
| (A) POOLED | all reliable | 16 | -0.491 | [-0.87, +0.09] |
| (B) CLEAN | non-license reliable | 13 | -0.357 | [-0.88, +0.39] |
| (C) CONTROLLED — z(disp) coef | all + license dummy | 16 | -0.002 | (OLS p=0.97) |
| (C) CONTROLLED — license dummy (gap shift) | | 16 | +0.417 | (OLS p=0.01) |

**Confound = (A) − (B) = -0.491 − (-0.357) = -0.134.**

Interpretation (what the numbers warrant, not what we want):
- **(B) CLEAN stays negative but CI spans 0** (-0.36, [-0.88,+0.39]) → the sign survives removing license fields but is n-limited (13 fields).
- Removing license fields **weakened** P1 by 0.13 — the license cluster was propping up the pooled slope; report the cleaner (B) as the identified estimate.
- **(C)**: within-reliable the dispersion slope is -0.002 (p=0.97) while the license dummy separately **shifts the gap by +0.42** (p=0.01) — i.e. license fields sit above the line (a credential-standardization level shift), distinct from the observability slope. Interaction z×license = +0.18 (p=0.21; n=16, 4 params — underpowered).

## Task 3 — two-dimensional regime table (license fields kept, not deleted)

| regime                  |   n |   mean_gap |   mean_acs_iqr |   mean_xinst_cv |
|:------------------------|----:|-----------:|---------------:|----------------:|
| PhD-pipeline            |   5 |      0.493 |          0.743 |           0.215 |
| license-standardization |   3 |      0.874 |          0.602 |           0.236 |
| prestige-transmission   |   8 |      0.429 |          0.712 |           0.217 |

The gap is **≥2 dimensions**: an **observability / integration axis** (prestige-transmission, where P1 operates) and a **credential-standardization axis** (license-standardization fields, *off* the P1 mechanism — earnings set by the license, not the school). PhD-pipeline is a third (talent-exit) axis. License fields are reported as their own dimension, not folded into or removed from P1.

## Verdict

**P1 does NOT cleanly survive the license-confound check.** The pooled gap–dispersion correlation (-0.49) weakens to -0.36 once the 3 license fields are removed (confound -0.13, both CIs span 0 at n=16/13), and — decisively — the dispersion slope **collapses to -0.002 (p=0.97)** when a license dummy is included, while that dummy carries a **significant +0.42 gap shift (p=0.01)**. So the pooled P1 was substantially the license fields sitting as a high-gap, low-(early-career-)dispersion cluster, not a continuous observability gradient. Among non-license fields the gradient is directionally negative (-0.36) but not significant (n=13). Cross-institution earnings are **not** compressed by the license (CV 0.24 ≈ 0.22); license fields are DECOUPLED (dispersion present, prestige ⊥ earnings), not flat — a credential-standardization axis distinct from observability. The identified result is **two-dimensional**: a separate license level-shift on the gap, and a weak/uncertain observability slope that does not stand on its own once license is controlled. See `figures/p1_by_regime.png`.

## Figure
`outputs/figures/p1_by_regime.png`