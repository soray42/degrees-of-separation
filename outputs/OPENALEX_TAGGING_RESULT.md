# OpenAlex field-tagging → per-field ORCID-AR re-validation vs Wapman

**Question.** Replace the noisy free-text `field_text` (degree/department string) field assignment
with principled OpenAlex publication-topic **field** tags, then ask: does better field tagging
push per-field ORCID-AR↔Wapman agreement up from the 0.45–0.68 `field_text` baseline toward the
~0.79 public-Wapman ceiling? (The "顶上去" test.)

**Answer (outcome-agnostic).** No — and the *reason* is the finding. Better field tagging does
**not** improve per-field reproduction of Wapman, but **not because the tags are wrong**: at matched
edge count OpenAlex is as faithful to Wapman as the degree text. OpenAlex underperforms only because
**research-field and degree-field are different taxonomies** — tagging by what people *publish*
scatters each degree cohort across several OpenAlex fields, thinning every per-field subnetwork. The
residual gap to the ceiling is a **coverage/sparsity ceiling, not a tagging-noise ceiling**, and
cannot be closed by relabelling.

---

## 1. Pipeline & coverage

- **Module** `src/openalex_tag.py`; OpenAlex API key auth (`api_key=` query param, 2026 convention),
  `mailto=` polite pool, on-disk JSONL cache (`data/interim/openalex_cache/`) for resume/budget.
- **Route A** (ORCID → `/authors?filter=orcid:…|…` → modal field over `author.topics`, batch 50).
- **Route B** (A-miss → `/works?filter=author.orcid:` → modal `primary_topic.field`).
- **Population**: 63,711 distinct US PhD→faculty people (one first-faculty-job edge each) from
  Yifeng's ORCID mobility dump (`data/interim/orcid_phd_faculty_edges.parquet`).

**Pilot (2,000 random ORCIDs, seed 42):**

| route | rate |
|---|---|
| Route A (ORCID→author) | **79.8%** |
| Route B rescue (works-by-ORCID) | **0.6%** |
| Union (ID-based) | **80.3%** |
| residual (→ name match) | 19.7% |

Route B rescues almost nothing because it is keyed on the *same* ORCID→OpenAlex linkage as A — if
the author isn't indexed, neither are the works. The genuine residual lever is the **DOI path** (ORCID
public API → DOIs → OpenAlex works), which the dump can't feed (no DOI/works table; only `edge_aff`).
Route B was therefore dropped for the full run (≈0.6% rescue for ~13k extra calls is poor ROI and
would eat the 10k-call/day budget). **Full run: Route A only → `data/interim/orcid_field.parquet`.**

**Full population:** 80.3% tagged (83.2% of 2011–2020 window edges); **all 26** OpenAlex fields
populated. The population skews Social Sciences (17%), Medicine (16%), Engineering, Psychology,
Biochem — OpenAlex's coarse 26-field grain lumps several Wapman fields into mega-buckets.

---

## 2. Per-field re-validation (the head-to-head)

For each cleanly 1:1-mappable field, three SpringRank-vs-Wapman Spearmans on the **same** Wapman
published field ranks (2011–2020, canonical SpringRank α=0.5 binary): OpenAlex-tagged subnetwork,
`field_text`-keyword subnetwork (the scripts/11 baseline), and Wapman's own public field edges (the
ceiling). `scripts/17_openalex_field_validate.py`.

| field | OA edges | ρ_OpenAlex | ft edges | ρ_field_text | ρ_benchmark | Δ(OA−ft) | ceiling gap |
|---|--:|--:|--:|--:|--:|--:|--:|
| Computer Science | 418 | 0.598 | 498 | 0.613 | 0.755 | −0.015 | 0.158 |
| Mathematics | 240 | 0.488 | 406 | 0.521 | 0.837 | −0.033 | 0.350 |
| Physics | 181 | 0.482 | 278 | 0.446 | 0.806 | **+0.036** | 0.324 |
| Psychology | 571 | 0.546 | 569 | 0.551 | 0.841 | −0.005 | 0.295 |
| Economics | 229 | 0.620 | 405 | 0.678 | 0.797 | −0.058 | 0.177 |
| Chemistry | 46 | 0.381 | 357 | 0.552 | 0.736 | −0.171 | 0.355 |
| Biology | 474 | 0.491 | 420 | 0.498 | 0.761 | −0.007 | 0.270 |

- OpenAlex beats `field_text` in **1/7** fields (Physics only). Median Δ = **−0.015**.
- mean ρ_OpenAlex **0.515** vs ρ_field_text **0.551** vs ρ_benchmark **0.791**.
- OpenAlex retains **fewer** edges per Wapman field (coverage ×0.74), starkest for Chemistry
  (46 vs 357).

*This raw −0.036 average is misleading — §3 shows it is a wash once the leaky degree-text baseline
is cleaned and Chemistry (one extreme-drift field) is set aside.*

*Mechanical / Civil Engineering from the scripts/11 baseline are **not separable** at OpenAlex's
26-field grain (both collapse into "Engineering"); dropped from the head-to-head and noted.*

---

## 3. Adversarial check — coverage effect, not tag-quality (`scripts/18`)

The §2 head-to-head was hardened after independent adversarial review, which flagged two ways the
raw −0.036 "OpenAlex worse" average is misleading. Both **strengthen** the no-difference reading.

**(a) Symmetric equal-n.** Subsample **both** subnetworks to their shared min edge count (200 paired
seeds), report P(ρ_OpenAlex ≥ ρ_field_text):

| field | shared n | ρ_OA @ n | ρ_ft @ n | P(OA ≥ ft) |
|---|--:|--:|--:|--:|
| Computer Science | 418 | 0.598 | 0.582 | **0.71** |
| Mathematics | 240 | 0.488 | 0.441 | **0.85** |
| Physics | 181 | 0.482 | 0.425 | **0.87** |
| Economics | 229 | 0.620 | 0.596 | **0.68** |
| Psychology | 569 | 0.546 | 0.551 | 0.05 *(Δ=−0.005, a wash)* |
| Biology | 420 | 0.486 | 0.498 | 0.34 |
| Chemistry | 46 | 0.381 | 0.444 | 0.27 |

**At matched edge count OpenAlex is favored in 4/7 fields, tied in Psychology (|Δ|=0.005), and
genuinely behind only in Biology and Chemistry** — the two fields where research-publication topic
drifts furthest from the degree program. OpenAlex's §2 shortfall is **a coverage/n effect, not
mis-assignment.**

**(b) De-leaked baseline.** The raw `str.contains(kw)` field_text baseline leaks cross-field degrees
("computer" catches electrical-&-computer engineering, "psycholog" catches educational psychology),
inflating field_text's ρ. Dropping rows that also name a competing field:

| | mean ρ |
|---|--:|
| OpenAlex | **0.515** |
| field_text (raw) | 0.551 |
| field_text (de-leaked) | **0.517** |

**Once the degree-text baseline is de-leaked, OpenAlex and degree-text are a dead heat (0.515 vs
0.517).** OpenAlex ≥ de-leaked field_text in 4/7 fields. The lone field where degree text still
clearly wins is **Chemistry** (OA 0.381 vs de-leaked ft 0.569) — exactly where drift is most extreme.

**(c) Chemistry leverage.** The raw −0.036 mean gap is one pathological field: mean Δ excluding
Chemistry is **−0.014 (a wash)**, and Chemistry alone is **40%** of the total OpenAlex edge deficit.
Chemistry's ρ=0.381 is real signal (bootstrap CI [0.272, 0.488]), not small-n degeneracy.

**Why does OpenAlex keep fewer edges?** Research-vs-degree drift — where OpenAlex sends each
degree-field cohort:

| degree field (field_text) | in OpenAlex's matching field | top OpenAlex destinations |
|---|--:|---|
| Computer Science | 54% | CS 54%; Engineering 22% |
| Mathematics | 41% | Math 41%; CS 13%; Engineering 10% |
| Physics | 35% | Physics 35%; Biochem 14%; Medicine 10% |
| Psychology | 44% | Psychology 44%; Social Sciences 20%; Medicine 13% |
| Economics | 36% | Economics 36%; Social Sciences 25%; Business 10% |
| Chemistry | **11%** | Biochem 33%; Materials 16%; Medicine 12% |
| Biology | 45% | Biochem 31%; Medicine 21%; Environmental 14% |

Only 35–54% (Chemistry 11%) of each degree cohort lands in the matching OpenAlex field; the rest
scatter into adjacent research fields — sensibly (chemistry→biochemistry/materials,
psychology→social sciences/medicine, economics→social sciences/business). This scattering thins each
OpenAlex per-field subnetwork and is the whole mechanism behind §2.

---

## 4. Verdict & implications

1. **OpenAlex field tagging does not "顶上去" the per-field ceiling — but it is not worse, either.**
   On a like-for-like (de-leaked) basis OpenAlex and degree-text reproduce Wapman equally (mean ρ
   0.515 vs 0.517); the raw −0.036 gap is Chemistry (→ −0.014 without it) plus keyword leakage that
   had inflated the degree-text baseline. The ~0.28 gap to the ~0.79 public-Wapman benchmark is
   **coverage-bound** (sparse ORCID edges vs Wapman's proprietary AARC census), not field-noise-bound
   — neither tagging scheme reaches the benchmark even at matched n. Better labels cannot close it;
   only more edges can.
2. **OpenAlex tags are not noise.** At matched edge count they reproduce Wapman as well or better in
   4/7 fields (CS/Math/Physics/Econ, P 0.68–0.87), tie in Psychology, and lag only in Biology and
   Chemistry. They simply answer a different question — what a person *researches*, not which *degree
   program* placed them — and Wapman is degree-program-based, so the two diverge most exactly where
   research topic drifts furthest from the doctoral field (Chemistry: 11% in-target).
3. **This reinforces the cluster-grain headline.** Because OpenAlex's 26 research fields cross-cut
   Wapman's degree fields (in-target 35–54%), the unit where the two taxonomies converge is the
   coarse **discipline cluster**, not the fine field — exactly the between-discipline-cluster level at
   which the AR–ER gap structure is robust (ICC≈0.30). For degree-program network validation, keep
   the degree-text field; reserve OpenAlex tags for *research-field* cuts, where they are the right
   instrument.

**Deliverables:** `src/openalex_tag.py`, `scripts/15`–`19`, `data/interim/orcid_field.parquet`
(63,711 tagged), `outputs/figures/oa_field_validation.png`, `oa_research_vs_degree_drift.png`.
**No raw OpenAlex cache, dump, or parquet committed** (gitignored); API key in gitignored `secrets.json`.
