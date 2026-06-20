# Tier 0 addendum — Ex-ante field typology (Task 2)

Regimes assigned from **external** data only (never the gap): licensure is a hand-coded ex-ante binary; pipeline intensity is the ACS graduate-degree share. Run: `python scripts/05_regime.py`. Date 2026-06-20.

## The classification rule (documented, ex-ante)

- **license-standardization** — entry to the field's modal occupation requires a standardized external license/credential that prices graduates largely independent of department prestige. Hand-coded TRUE for **nursing** (RN/NCLEX), **communication disorders** (CCC-SLP / state license), **accounting** (CPA), **civil engineering** (PE, effectively required to practice). Other engineering: PE is optional for most industry roles → FALSE. (Validates against the CPS certification/licensing question; see SOURCES.md.)
- **PhD-pipeline** — non-licensed fields whose **ACS graduate-degree share** is in the top tercile (cut = 54%); the signal-carrying BA graduates continue to grad school, so BA earnings undercount the department.
- **prestige-transmission** — the remaining fields (BA labor market prices the graduate directly).
- **Scope boundary (not a limitation):** professional-entry fields (law, medicine) are excluded by entry-degree — there is no BA-to-practice pipeline and Wapman does not cover professional-school prestige.

## Gap within regime (reliable fields)

| regime                  |   size |   mean |   median |
|:------------------------|-------:|-------:|---------:|
| PhD-pipeline            |      2 |  0.653 |    0.653 |
| license-standardization |      3 |  0.874 |    0.927 |
| prestige-transmission   |      9 |  0.398 |    0.413 |

## The reliable HIGH-gap fields, classified by channel

| label                   |   gap | regime                  |   grad_share_acs |
|:------------------------|------:|:------------------------|-----------------:|
| Communication Disorders | 1.208 | license-standardization |              79% |
| Nursing                 | 0.927 | license-standardization |              31% |
| Philosophy              | 0.657 | PhD-pipeline            |              59% |
| Biology                 | 0.650 | PhD-pipeline            |              62% |
| English                 | 0.506 | prestige-transmission   |              50% |
| Accounting              | 0.488 | license-standardization |              34% |
| Electrical Engineering  | 0.487 | prestige-transmission   |              49% |

**Reading the high-gap tail by channel — the high gap is not one phenomenon:**
- **license-standardization** (nursing 0.93, communication disorders 1.21): a standardized license, not department prestige, sets earnings → prestige and earnings decouple by construction. These are the highest gaps and are *expected*, not anomalous.
- **PhD-pipeline / talent-exit** (biology, and the unreliable chemistry/physics/earth sciences): the ablest BA graduates leave for PhDs, so BA earnings cannot track research prestige. Biology is the reliable exemplar.
- **genuine quality-disagreement** (English, philosophy): neither licensed nor strongly pipelined, yet a real gap — the residual the integration frame is actually about.

This converts the structurally-special fields from a limitations footnote into an ex-ante typology: the gap means different things in different regimes, and the integration construct is cleanest within **prestige-transmission**.

## Full typology

| field                   | label                   |   n_institutions |     gap | reliable_flag   |   signal_frac |   grad_share_acs |   grad_degree_share | licensed   | regime                  |
|:------------------------|:------------------------|-----------------:|--------:|:----------------|--------------:|-----------------:|--------------------:|:-----------|:------------------------|
| biomedical_engineering  | Biomedical Engineering  |               48 |   0.351 | False           |          0.02 |              63% |               nan   | False      | PhD-pipeline            |
| materials_science       | Materials Science       |               23 |   0.488 | False           |          0.02 |              54% |               nan   | False      | PhD-pipeline            |
| biochemistry            | Biochemistry            |               47 |   0.578 | False           |          0.02 |              69% |                71.9 | False      | PhD-pipeline            |
| biology                 | Biology                 |              137 |   0.650 | True            |          0.70 |              62% |                64.0 | False      | PhD-pipeline            |
| physics                 | Physics                 |               43 |   0.655 | False           |          0.34 |              67% |                67.3 | False      | PhD-pipeline            |
| philosophy              | Philosophy              |               41 |   0.657 | True            |          0.60 |              59% |                56.9 | False      | PhD-pipeline            |
| chemistry               | Chemistry               |               76 |   0.687 | False           |          0.02 |              66% |                67.5 | False      | PhD-pipeline            |
| astronomy               | Astronomy               |                3 | nan     | False           |        nan    |              58% |               nan   | False      | PhD-pipeline            |
| neuroscience            | Neuroscience            |                1 | nan     | False           |        nan    |              65% |               nan   | False      | PhD-pipeline            |
| accounting              | Accounting              |               94 |   0.488 | True            |          0.83 |              34% |                33.7 | True       | license-standardization |
| civil_engineering       | Civil Engineering       |              105 |   0.494 | False           |          0.02 |              40% |                37.1 | True       | license-standardization |
| nursing                 | Nursing                 |               99 |   0.927 | True            |          0.67 |              31% |                30.3 | True       | license-standardization |
| communication_disorders | Communication Disorders |               17 |   1.208 | True            |          0.51 |              79% |               nan   | True       | license-standardization |
| statistics              | Statistics              |               28 |   0.246 | False           |          0.78 |              54% |               nan   | False      | prestige-transmission   |
| computer_science        | Computer Science        |              161 |   0.292 | True            |          0.92 |              34% |                32.7 | False      | prestige-transmission   |
| computer_engineering    | Computer Engineering    |               74 |   0.303 | False           |          0.54 |              41% |                39.4 | False      | prestige-transmission   |
| economics               | Economics               |              116 |   0.313 | True            |          0.90 |              43% |                41.6 | False      | prestige-transmission   |
| political_science       | Political Science       |              139 |   0.343 | True            |          0.72 |              53% |                52.7 | False      | prestige-transmission   |
| mechanical_engineering  | Mechanical Engineering  |              133 |   0.382 | False           |          0.18 |              40% |                39.1 | False      | prestige-transmission   |
| mathematics             | Mathematics             |              120 |   0.384 | True            |          0.87 |              53% |                51.3 | False      | prestige-transmission   |
| finance                 | Finance                 |               89 |   0.413 | True            |          0.94 |              31% |                31.5 | False      | prestige-transmission   |
| history                 | History                 |              131 |   0.413 | True            |          0.73 |              50% |                51.4 | False      | prestige-transmission   |
| psychology              | Psychology              |              173 |   0.429 | True            |          0.77 |              53% |                51.9 | False      | prestige-transmission   |
| sociology               | Sociology               |              114 |   0.430 | False           |          0.51 |              41% |                39.1 | False      | prestige-transmission   |
| aerospace_engineering   | Aerospace Engineering   |               35 |   0.444 | False           |          0.02 |              52% |                45.5 | False      | prestige-transmission   |
| chemical_engineering    | Chemical Engineering    |               91 |   0.452 | False           |          0.02 |              48% |                48.1 | False      | prestige-transmission   |
| electrical_engineering  | Electrical Engineering  |              118 |   0.487 | True            |          0.71 |              49% |                47.7 | False      | prestige-transmission   |
| english                 | English                 |              135 |   0.506 | True            |          0.58 |              50% |                46.2 | False      | prestige-transmission   |
| anthropology            | Anthropology            |               75 |   0.508 | False           |          0.54 |              49% |                48.2 | False      | prestige-transmission   |
| earth_sciences          | Earth Sciences          |               50 |   0.715 | False           |          0.02 |              50% |                45.6 | False      | prestige-transmission   |