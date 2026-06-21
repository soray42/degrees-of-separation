# Co-author decision points

These are the calls that need senior judgment before the draft goes anywhere. The draft (`main.tex`)
takes a default position on each, flagged inline with `% COAUTHOR:` — these notes collect them so they
can be decided deliberately, not by inertia.

## 1. Venue / positioning (the biggest one)
The draft is written to a **computational-social-science / data-science** tier (Nature Human Behaviour,
PNAS Nexus, EPJ Data Science). It is an honest **descriptive** contribution, not a causal or behavioural
"finding." Options:
- **(a) Descriptive CSS venue** (EPJ Data Science / PNAS Nexus) — best fit for what we actually have; lower
  risk; the "ruling-out" framing reads as rigor.
- **(b) Push for NHB** — needs a sharper single "so-what" (see #2) and probably the UK replication (#4); higher
  risk of a "so what / descriptive" desk reject.
- **(c) Economics-of-education field journal** — would demand more causal identification we explicitly do not
  have (Chetty-style admin data), so likely a poor fit without a major data addition.
**Default in draft:** (a), with the abstract written so it could be re-aimed at (b). **Needs a decision.**

## 2. How hard to push the "so-what"
The descriptive spine is solid; the question is the headline framing. Candidates, in increasing boldness:
- "We measure a prestige--pay wedge and show it is mostly structural (discipline) + licensing + irreducible
  residual" (safe, what we have).
- "Field-agnostic prestige rankings (QS/US News overall) are systematically misleading because the prestige↔pay
  structure is field-specific" (the policy hook — defensible, moderately bold).
- "The labour market actively reprices fields while academic prestige lags" (the temporal result — **boldest**,
  but the lead-lag is NOT identified yet, so this would over-claim unless softened to the revaluation fact only).
**Default in draft:** lead with the wedge + the ranking critique; present revaluation as a clean fact and the
lead-lag as future work. **Decide how hard to lean on the ranking critique vs. play it as an implication.**

## 3. Model framing (§6) — how much weight to give it
The two-signal employer-learning model is **built but its key external prediction (residual = valuation
divergence) is UNCONFIRMED** (mixed O*NET vs SDR proxies). Options:
- **(a)** Keep it as a short "organizing framework / interpretive lens" section, explicitly not validated
  (current default). Honest, lower-risk.
- **(b)** Cut it to an appendix and lead purely with the empirics.
- **(c)** Invest in validating it (needs a better divergence proxy / the lead-lag) before submission.
**Default:** (a). A reviewer who dislikes unvalidated models may push for (b) — easy to do.

## 4. UK / cross-national replication before submission?
The design is cross-national-ready (any country with a faculty-hiring network + an earnings-by-field source).
A UK replication (e.g. a UK prestige hierarchy + LEO graduate-earnings data) would materially strengthen a
push to (1b). Cost: a real new data pull + crosswalk. **Decision: add it now (stronger, slower) or cite it as
the obvious next step (faster)?** Default in draft: cite as next step.

## 5. Authorship / outreach targets
The extended abstract is also for faculty outreach (Giustinelli on subjective expectations / beliefs about
returns; Sinatra on science-of-science / bibliometric networks). **Decide:** who to approach, in what order,
and whether either becomes a co-author shaping the framing (Giustinelli → beliefs/decision angle; Sinatra →
networks/science-of-science angle). This changes §1 and §8 emphasis.

## 6. Scope of the public ranking artifact
The CS-school ranking (scripts/34) is a compelling **public-facing demonstrator** of the method but is NOT the
scientific contribution. **Decide:** include it as a short "application" subsection / figure (current default —
it makes the ranking-critique concrete), relegate to SI, or spin it off as a separate public artifact (a website
/ a 小红书-style post) and keep the paper purely methodological. Over-featuring it risks the paper reading as
"another ranking" rather than a critique of rankings.

## Minor / mechanical (decide in passing)
- Earnings horizon: draft uses Scorecard 4yr as primary (matches the analysis); confirm vs 1yr/5yr framing.
- Title: working title "Degrees of Separation: a measurable wedge between academic prestige and the labour-market
  pricing of fields." Alternatives welcome.
- Whether to report the `[TODO: source]` items (a few numbers not in a RESULT file) by re-running the relevant
  script, or to drop those sentences. See CLAIMS_LEDGER.md for the list.
