# Tier 0 addendum — Multi-dimensional gap explanation (Task 3)

The gap stays salary-anchored, but its **explanation** now draws on employer demand and continuation, not salary alone. **Descriptive correlates only — not causal.** NY Fed 'Labor Market for Recent College Graduates' by-major, reliable fields. Run: `python scripts/06_explanation.py`. Date 2026-06-20.

## Spearman(gap, correlate) on reliable fields

| correlate                                             |   n |   spearman |     p |
|:------------------------------------------------------|----:|-----------:|------:|
| employer demand (underemployment %)                   |  13 |     +0.049 | 0.873 |
| employer demand (unemployment %)                      |  13 |     -0.313 | 0.297 |
| early-career median wage                              |  13 |     -0.429 | 0.143 |
| continuation (share w/ graduate degree)               |  13 |     +0.148 | 0.629 |
| continuation — ACS grad-degree share (all 30 covered) |  14 |     +0.284 | 0.326 |

## Read (report-the-sign-whatever-it-is)

- **The gap does NOT track employer demand.** Underemployment ≈ 0 (+0.05); unemployment weak (−0.31). The gap is not a demand story.
- **Continuation (graduate-degree share) is correct-signed but modest** — +0.28 on the all-30-covered ACS measure, +0.15 on NY Fed (reliable fields): fields that send more BA talent onward tend to have larger gaps (the selection fingerprint), but the continuous correlation is weak. The signal is much **clearer as the discrete regime split** (Task 2: PhD-pipeline 0.65 vs prestige-transmission 0.40) than as a continuous Spearman — the typology, not the scalar, is where continuation bites.
- **The strongest single correlate is early-career wage (−0.43)** — higher-paying fields have lower gaps — but that is itself salary, so it partly re-expresses the salary-anchored gap rather than adding a new dimension.
- **Honest bottom line:** no *single non-salary* continuous correlate is strong; the multi-dimensional value comes from the **regime typology (licensure + pipeline, Task 2)** and the **observability test (P1)**, not from the NY Fed scalars. The gap's structure is legible, but through discrete channels, not one demand/continuation index.

## Per-field correlate table (reliable fields)

| label                  |   gap |   underemployment |   unemployment |   wage_early |   grad_degree_share |   grad_share_acs |
|:-----------------------|------:|------------------:|---------------:|-------------:|--------------------:|-----------------:|
| Nursing                | 0.927 |              12.8 |            2.1 |        70000 |                30.3 |              31% |
| Philosophy             | 0.657 |              47.1 |            5.1 |        52000 |                56.9 |              59% |
| Biology                | 0.650 |              51.1 |            4.3 |        45000 |                64.0 |              62% |
| English                | 0.506 |              48.5 |            6.1 |        48000 |                46.2 |              50% |
| Accounting             | 0.488 |              21.2 |            2.6 |        68000 |                33.7 |              34% |
| Electrical Engineering | 0.487 |              21.1 |            3.2 |        82000 |                47.7 |              49% |
| Psychology             | 0.429 |              48.3 |            5.0 |        45000 |                51.9 |              53% |
| History                | 0.413 |              50.1 |            4.3 |        47500 |                51.4 |              50% |
| Finance                | 0.413 |              27.8 |            2.8 |        70000 |                31.5 |              31% |
| Mathematics            | 0.384 |              26.2 |            5.8 |        70000 |                51.3 |              53% |
| Political Science      | 0.343 |              48.7 |            4.5 |        52000 |                52.7 |              53% |
| Economics              | 0.313 |              33.1 |            3.5 |        72000 |                41.6 |              43% |
| Computer Science       | 0.292 |              19.1 |            7.0 |        87000 |                32.7 |              34% |