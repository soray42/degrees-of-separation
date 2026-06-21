# Structure changes: hardening + revealed-placement reframe

This records the restructuring of `paper/main.tex` from the prior 8-section draft into a
hardened, four-pillar journal article. **No new analysis** — every number and figure is reused
from the `*_RESULT.md` record (indexed in `CLAIMS_LEDGER.md`). Two diagnoses drove the change:
(1) the prior draft read like a defensive memo (the contribution was swallowed by reflexive
hedging); (2) the second axis is now reframed as **revealed labour-market placement**.

## Global reframe
- The second axis is renamed throughout from "employer reputation (ER)" to **revealed
  labour-market placement** (earnings as the proxy). The object is academic **reputation** (AR,
  hiring-network SpringRank) vs revealed **placement** — an honest asymmetry (perception vs
  realised outcome), not "reputation vs reputation." Title, abstract, and prose updated.
- Title: "...labour-market **placement** of fields" (was "...pricing of fields").

## Main-text spine: four pillars, stated as findings
| § | Pillar | Content |
|---|---|---|
| 1 Introduction & stakes | — | field-agnostic-ranking problem; one stakes sentence; no hedging |
| 2 Data & the wedge | **1 Measurement** | AR (continuous SpringRank), revealed placement (earnings proxy), CIP join, gap definition, reliability — stated as a measurable, reliable object |
| 3 Discipline structure | **2 Structure** | multilevel decomposition, ICC, integrated/decoupled typology, gap map — led as the finding |
| 4 Salary as placement proxy + licensing | **3 Proxy validation + 4 Mechanism** | (a) cross-source validation (PSEO +0.94/+0.68; non-wage placement +0.65–0.84); (b) **featured** licensing mechanism (b_licensure=+0.66; setting-not-school, incremental-prestige corr −0.59; Nursing ≈0); (c) one tight paragraph on pipeline-deferral |
| 5 Robustness | — | one compressed paragraph + SI pointer |
| 6 Discussion | — | ranking-critique implication; revealed-placement framing; one forward-frontier paragraph (dynamic + Revelio) |
| 7 Limitations | — | one paragraph, each load-bearing |

## Moved to SI / appendix (relocated, NOT deleted; tagged `SI:` in `\appendix`)
- **Two-signal model** → **SI A** (`\label{si:model}`), entirely. Tagged "an organising model, not
  validated." Removed from the main-text spine; one pointer sentence remains in §6. *This single
  move removed most of the "unvalidated" prose from the main text.*
- **Three ruled-out alternatives** (field-vs-generic, integration-shape, program-mispricing) →
  compressed to **one main-text sentence** (end of §4c) + full treatment in **SI B**
  (`\label{si:ruleout}`).
- **Dynamic / temporal pilot** → **one-paragraph teaser in §6** + **SI D** (`\label{si:dynamic}`)
  pointing to the full version in `DYNAMIC_LEADLAG_PILOT_RESULT.md` / the companion. The
  revaluation figure moved to SI D.
- **Robustness walk-through** → compressed to one §5 paragraph; per-check detail pointer in
  **SI C** (`\label{si:robust}`).

## Hedging discipline
- Reflexive hedging in the main text cut from ~30 instances to 2 (both in the navigational
  sentence pointing to §7 and the confident "honest framing" reframe in §6). The honesty content
  was **relocated**, not deleted: it now lives in the single §7 Limitations paragraph and in the
  SI. Words removed from the main line: "scope condition," "caveat," "not a failure," "unvalidated"
  (now only in the SI model title), reflexive "we do not / we cannot."

## New material integrated
- §4a now includes the placement-proxy-by-horizon validation (`placement_proxy_by_horizon.png`,
  from `ER_AXIS_IDENTIFICATION_RESULT.md` 35a): salary tracks non-wage placement and improves with
  horizon (+0.75→+0.83), replicated on PSEO.
- §4b features the licensing **mechanism** (`licensing_mechanism.png`, 35c): the commonality /
  incremental-R² result that pay in regulated/locally-employed fields is set by setting, not school.
- §4c integrates pipeline deferral (35b) as the proxy's one timing failure, explicitly distinct
  from the null absorption channel and not touching the within-field gap.

## Figures in main text
`cluster_means_forest` (§3), `gap_by_cluster` (§3), `pseo_vs_scorecard_gap` + `placement_proxy_by_horizon`
(§4a), `licensing_mechanism` (§4b); `field_earnings_revaluation` moved to SI D.

## Open markers
- One `\todo{}` in §5: formal FDR/BH q-values. **No FDR analysis exists in the record**; the
  per-field reliability filter (signal-fraction + bootstrap-CI screen) is the de-facto false-positive
  control, and the `\todo` flags a formal multiple-testing statistic as the only genuinely missing
  value (per the "[TODO: source]" instruction). All other numbers map to a `*_RESULT.md`.
- `\coauthor{}` notes retained only where a senior decision is genuinely needed (authorship; whether
  to feature the CS ranking).

## Follow-ups (out of scope here)
- `CLAIMS_LEDGER.md` remains the source of truth; no numbers changed, so it is unedited (its
  topic organisation still maps every value to a `*_RESULT.md`).
- `paper/methods_companion.tex` cross-references the main paper's *old* section numbers (`\S2`–`\S8`);
  after this restructure those mappings are stale and would need one reconciliation pass if the
  companion is kept in lock-step. Not done here (the task scope is `main.tex`).

## Compilation
`pdflatex`/`bibtex` are not installed in the authoring environment, so the PDF was not built here.
Compile with `pdflatex main && bibtex main && pdflatex main && pdflatex main`. LaTeX integrity was
checked mechanically (balanced braces/math/environments; all `\ref` resolve; all six referenced
figures present).
