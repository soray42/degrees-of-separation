# Behavioural-half feasibility (LIGHT): do students over-credit the decoupled fields?

Viability check feeding the co-author conversation --- NOT the full behavioural analysis. No-fabrication run. Seeded; `python scripts/42_behavioural_feasibility.py`.

## Variable availability (codebook check)

HSLS:09 public-use file (NCES, Base Year 2009 through Second Follow-up 2016), 23,503 students. Confirmed variables and non-missing counts:

| variable | meaning | n valid |
|---|---|---|
| `S4FIELD2` / `S3FIELD2` | postsecondary field of study (2-digit CIP) | 12,134 / 11,639 |
| `X1STU30OCC2` | base-year expected occupation at 30 (2-digit SOC) | 21,018 |
| `S4OCC30EARN` / `S2OCC30EARN` | expected earnings at 30 (\$) | 14,178 / 12,477 |
| `X1STUEDEXPCT` | expected educational attainment (ordinal) | 21,444 |
| `W4W1STU` | base-year->2016 panel weight | 23,503 |

**The 6-digit occupation/field codes (`*OCC6`, `*FIELD6`) are disclosure-suppressed in the public-use file** --- only 2-digit SOC (occupation) and 2-digit CIP (field) are released, so the cut is at the **CIP-2 grain** (which is exactly the project's discipline-cluster / integrated--decoupled typology grain). NLSY97 (secondary) is **access-gated**: variables must be hand-selected and extracted via the BLS NLS Investigator (not one-click); flagged as a second step, not run here.

## The rough cut (CIP-2 field level; ranks, to net general optimism)

Merged on CIP-2: **21 fields** with both a HSLS expected outcome (>=30 students/field) and the project's realized placement + gap. `overcredit = rank(E_f) - rank(R_f)`: positive = students rank a field's placement HIGHER than it realizes.

| label               |   n_students |   E_occ_prestige |   R_occ_prestige |   overcredit |   gap |
|:--------------------|-------------:|-----------------:|-----------------:|-------------:|------:|
| Computer/Info       |          569 |             62.8 |             65.7 |         -5.0 |  0.29 |
| Mathematics         |           98 |             59.8 |             60.4 |        -13.0 |  0.32 |
| Parks/rec/fitness   |          282 |             64.3 |            nan   |       +nan   |  0.40 |
| History             |          118 |             63.9 |             52.3 |        +13.0 |  0.41 |
| Family/consumer sci |           75 |             62.3 |            nan   |       +nan   |  0.42 |
| Psychology          |          622 |             62.4 |             54.2 |         +4.0 |  0.43 |
| Social sci          |          439 |             62.1 |             54.1 |         +3.0 |  0.47 |
| Education           |          938 |             61.1 |             56.3 |         -5.0 |  0.47 |
| Business            |         2103 |             60.7 |             54.6 |         -5.0 |  0.48 |
| English             |          149 |             61.1 |             52.1 |         +3.0 |  0.51 |
| Engineering         |         1249 |             63.8 |             65.4 |         -1.0 |  0.51 |
| Physical sci        |          260 |             64.6 |             60.0 |         +5.0 |  0.51 |
| Public admin        |          101 |             62.0 |             53.1 |         +4.0 |  0.55 |
| Languages           |           63 |             61.3 |            nan   |       +nan   |  0.56 |
| Interdisciplinary   |          110 |             63.6 |            nan   |       +nan   |  0.73 |
| Health              |         2805 |             65.1 |             60.8 |         +4.0 |  0.79 |
| Arts                |          727 |             60.6 |             51.0 |         +1.0 |  0.79 |
| Agriculture         |          153 |             63.0 |             53.7 |         +8.0 |  0.79 |
| Architecture        |           81 |             65.4 |             59.0 |         +8.0 |  0.83 |
| Biological sci      |         1198 |             66.8 |             58.2 |        +10.0 |  0.83 |
| Nat. resources      |           94 |             61.4 |             57.0 |         -3.0 |  0.96 |

### Test 1 (core) --- is over-crediting concentrated in the DECOUPLED (high-gap) fields?

- `corr(overcredit, gap)` = **Spearman +0.40** (p=0.11, n=17); expected-earnings variant +0.32 (p=0.21, n=17).

### Test 2 (cross-check) --- do expectations track realized placement only weakly?

- `corr(E_f, R_f)` occupational prestige = Spearman +0.37 (p=0.15); earnings +0.14 (p=0.60). Weak-to-moderate tracking is the ``fooled-by-prestige'' signature (expectations not strongly aligned with where fields actually place).


**Most over-credited fields (rank E >> rank R):** History (gap 0.41, +13), Biological sci (gap 0.83, +10), Agriculture (gap 0.79, +8), Architecture (gap 0.83, +8), Physical sci (gap 0.51, +5).

## Verdict

**The behavioural half is VIABLE --- a directional signal is present** for development with a subjective-expectations co-author. The variables exist in public data (HSLS:09), the crosswalks resolve at CIP-2, and the rough cut shows the hypothesised direction (students over-credit the decoupled fields). This is a **signal check, not a clean estimate**. The full version needs: (i) a rigorous belief-measurement design (probabilistic elicitation of expected placement by field x tier, not a single expected-occupation code); (ii) an explicit **general-optimism benchmark** (students over-expect on everything --- handled here with ranks, but a calibrated design is better); and (iii) for the *within-field* claim --- does a prestigious program within a field place better, the true analog of the paper's gap --- an **institution x field fielded RCT** (the co-author's domain), which public panels cannot deliver.

## Adversarial self-check

- **General optimism handled by ranks.** Students over-expect on almost everything; levels would just show uniform optimism. We use rank(E)-rank(R), so only RELATIVE over-crediting (which fields are ranked too high vs realized) enters --- the right object for ``fooled-by-prestige.''

- **FIELD-LEVEL, not within-field.** This tests whether students mis-rank which FIELDS place well; the paper's gap is WITHIN-field across institutions. The two are related (both are prestige!=placement) but distinct; the within-field belief test is the future RCT, specified not run.

- **HSLS realized outcomes are early/incomplete**, so R_f here uses the project's mature ACS/Scorecard realized placement (occ-prestige / earnings) as the benchmark, not HSLS's own (young) follow-up outcomes --- a deliberate choice, flagged.

- **Indirect measurement.** HSLS gives expected OCCUPATION (2-digit SOC) and expected EARNINGS, not a direct belief about a field's prestige--placement gap. E_f is a constructed proxy (expected occupational prestige via Condon), and the SOC-2 / CIP-2 grain is coarse --- a signal check, not the designed instrument.

- **n and crosswalk losses.** Only 21 CIP-2 fields enter (others lack >=30 HSLS students or a project realized value); the 6-digit codes are suppressed; SOC-2 occupational prestige averages over a major group. Small n + coarse grain => underpowered; read as direction, not magnitude.
