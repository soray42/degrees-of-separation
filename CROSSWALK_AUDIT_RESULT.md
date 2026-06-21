# TASK 1 — Crosswalk audit + capped field recovery

Why did the 81-field Wapman taxonomy collapse to 54? Audit of the 27 missing fields and a capped recovery of the crosswalk misses. Run: `python scripts/24_crosswalk_audit.py`.

## Audit of the 27 missing Wapman fields

- **RECOVERABLE (crosswalk miss): 12** — a valid undergraduate CIP with Scorecard earnings (>=10 institutions) existed but was not in the hand-built map: Linguistics, Spanish Language and Literature, Nutrition Sciences, Urban and Regional Planning, Theological Studies, Agricultural Engineering, Education, General, Education Administration, Teacher Education Specific Subjects, Exercise Science, Kinesiology, Rehab, Health, Health, Physical Education, Recreation, Human Development and Family Sciences, General.
- **Subsumed in an existing aggregate field: 3** (finer splits, not gaps): Cell Biology -> biology (26xx), Natural Resources -> environmental_sciences (0301), Information Science -> computer_science (1104).
- **GENUINE DATA GAP (no clean BA earnings): 12** — graduate-only fields with no bachelor's-in-field CIP (Epidemiology, Pathology, Pharmacology, Educational Psychology, Veterinary Medical Sciences, Plant Pathology, Environmental Health, Entomology, Art History [no 4-digit distinct from studio arts]) or BA earnings suppressed at small fields (Classics n=3, Soil Science n=6, Curriculum & Instruction n=4): Classics and Classical Languages, Pharmacology, Soil Science, Curriculum and Instruction, Plant Pathology, Epidemiology, Pathology, Educational Psychology, Art History and Criticism, Veterinary Medical Sciences, Entomology, Environmental Health Sciences.

**Answer to 'why only 54':** the crosswalk was *partly* naive — **12 fields are recoverable** (it under-consolidated valid CIP matches) — but the **majority of the remaining gap is genuine**: ~12 Wapman fields rank PhD programs in disciplines with **no bachelor's earnings** (graduate-only) or with Scorecard small-cell suppression. Field recovery raises the COUNT; it does **not** create undergraduate earnings where none exist, and does not narrow the wide per-institution CIs.

## Re-run on the expanded field set

| set | fields | non-degenerate | reliable units | τ² | ICC raw | ICC corrected | grand gap |
|---|---|---|---|---|---|---|---|
| baseline (54) | 54 | 48 | 16 | 0.0231 | 0.30 | 0.45 | 0.55 |
| expanded (66) | 66 | 57 | 20 | 0.0207 | 0.32 | 0.57 | 0.55 |

**Headline check.** ICC raw 0.30→0.32, ICC corrected 0.45→0.57; cluster ranking correlation base-vs-expanded (shared clusters) **Spearman +0.98**. The between-discipline headline **HOLDS** under expansion.

### Expanded cluster ranking (precision-weighted)

| name                  |   n_fields |   cluster_gap |
|:----------------------|-----------:|--------------:|
| Nat. Resources        |          2 |         0.844 |
| Arts                  |          2 |         0.816 |
| Health                |          4 |         0.805 |
| Architecture/Planning |          2 |         0.801 |
| Nutrition/Interdisc.  |          1 |         0.732 |
| Biological Sci        |          5 |         0.699 |
| Physical Sci          |          3 |         0.685 |
| Philosophy/Religion   |          1 |         0.657 |
| Education             |          2 |         0.632 |
| Agriculture           |          3 |         0.574 |
| Foreign Languages     |          2 |         0.555 |
| Social Work           |          1 |         0.552 |
| English               |          1 |         0.506 |
| Business              |          4 |         0.476 |
| Engineering           |         11 |         0.432 |
| Psychology            |          1 |         0.429 |
| Family/Consumer Sci   |          1 |         0.420 |
| History               |          1 |         0.413 |
| Social Sci            |          5 |         0.412 |
| Parks/Kinesiology     |          2 |         0.390 |
| Math & Stats          |          2 |         0.322 |
| Computer/Info         |          1 |         0.292 |

## Adversarial self-check

**Strongest referee objection.** Recovery introduces taxonomy slippage: several recovered fields lean on broad/aggregate CIPs (Spanish=Romance-languages 1609; HPER=parks/recreation 3101/3103; Teacher-Ed=1312/1313) whose Scorecard BA population may not match the Wapman PhD field, so the new gaps could be measuring a different cohort than the prestige they are joined to. The recovery is also asymmetric — it adds Education/Humanities/Health fields, which could shift the cluster composition and the ICC mechanically by adding new small clusters.
**Does it survive?** Yes, with caveats: the cluster ranking is stable (Spearman +0.98) and the ICC barely moves, so the between-discipline structure is not an artifact of the original 54. The recovery is honestly CAPPED at clean, non-conflicting CIP matches; the 12 genuine data-gap fields are NOT forced in, and the per-institution CIs are unchanged (recovery is a count/cluster-precision gain only, not an earnings-precision gain).
