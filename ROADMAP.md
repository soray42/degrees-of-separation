# Degrees of Separation — Roadmap (pending / Act Two)

> Companion to [`README.md`](README.md) (the finished platform). This is the **pending** side:
> extension directions, engineering plans, ordering, and the Revelio boundary. **Working notes to
> myself** — engineering detail is **not abbreviated** (data columns, pipelines, caveats all kept).
> Source: current post-pilot strategy + `paper/next_stage_agenda.tex`.

**Strategy in one line.** The pilot proved there is **no second mechanism beyond licensing** — stop
hunting for one. What the pilot actually bought is the upgrade of **licensing into a special case of a
more general *wage-compression* mechanism**. Every direction below serves one coherent
**"Act Two: the compression mechanism — when / why / where prestige stops paying,"** and **directions
①–④ are entirely Revelio-free**.

---

## 0. Pilot trophies (the "why" foundation — don't lose these)

- ICC(ρ | CIP-2) = 0.30 / 0.45: discipline structure is real but modest = something we **already had**,
  not a new finding.
- Kinship-shared coupling (Idea A) is **dead**: Moran's I over the faculty-flow graph = +0.011, p = 0.96,
  robust across 7 W-constructions.
- The CIP-2 co-membership signal (Moran's I = +1.71, p = 0.002) **looks significant** but collapses on
  leave-pair-out: 84% comes from the single nursing/comm-disorders license pair → **licensing in disguise,
  a false positive caught red-handed by adversarial verification**.
- Stats↔CS dual-kinship **falsified**: Statistics is a Mathematics satellite (9.2:1).
- STATE×OCCP netting: nursing raw \$5k → \$0 (n = 22,518) = a nursing grad's wage simply **is** the RN
  occupation wage in their state — a non-circular correction.
- occ_hhi ↔ coupling is a **clean null**; CS is the counterexample (occupation-pinned, high occ_hhi, yet
  the **highest** coupling 0.708); occ_hhi ≈ the license dummy (ρ = 0.643, p = 0.007).
- **⇒ Conceptual upgrade.** The sufficient condition for decoupling is **not "occupation pinning" but
  "the occupation lacks prestige-sortable within-occupation wage variance."** Compression is the
  mechanism; licensing is its sharpest special case.

---

## 1. Three time axes (don't fixate on the dead one)

| Axis | What it is | Data | Free & immediate? | Revelio boundary |
|---|---|---|---|---|
| ① calendar / cohort time | Scorecard FoS pooled 2014-15→2018-19, ~4–6 cohort points; IPEDS completions = a decades-long annual supply series | Scorecard FoS / IPEDS | ✅ thin — enough for a shock, not for a trend | — |
| ② **career / experience time** (the richest) | coupling(t) as a function of years since graduation | Scorecard 1yr+4yr(+5yr); PSEO **1 / 5 / 10 yr** | ✅ **never touches the prestige side** | individual trajectories: no |
| ③ prestige time | time-varying SpringRank | ORCID affiliations (dated) | ❌ per-window independent estimation hits edge density | — |

**Axis ② comes with its own theoretical baggage — employer learning** (Farber–Gibbons 1996 /
Altonji–Pierret 2001 / Lange 2007 / Arcidiacono–Bayer–Hizmo 2010). Three mechanisms, three signatures —
**there is a story to tell whichever way it comes out:**

- prestige is a **signal** → coupling(t) **decays** with tenure (employers learn true ability, the alma
  mater depreciates);
- prestige is **human capital / networks** → flat or **rising** (elite-tail compounding);
- your **compression** fields → pinned at ≈0 throughout (there is nothing to learn).

**Two live paths for axis ③** (brute-force re-estimation was the original cause of death — don't repeat it):
- The clever route: **state-space smoothed rankings** — Whole-History Rating (Coulom 2008), TrueSkill
  Through Time (Dangauthier et al., NIPS 2007). A temporal-smoothness prior lets sparse windows borrow
  edges from neighbours, demoting edge density from a **fatal flaw** to a **regularization-strength choice.**
  ⚠️ This is a self-contained methods sub-project with a distinct minimax-bounds flavour — a natural piece to spin off as a separate track.
- The lazy route (legitimate version): Wapman 2022's title says "dynamics" and its conclusion is that the
  2011–2020 hierarchy is **extremely stable** → cite it, fix prestige as **quasi-static** → gap dynamics =
  placement dynamics, and identification is actually *cleaner*. Backed by the literature, not a concession.
- Nuance: even with windowing, feasibility is **per-field** — CS/bio (thousands of hires/yr) might support
  2–3 windows; small fields, forget it.

---

## 2. Ranked directions + concrete engineering plans

### 🅢① career-time coupling (= axis ②) — run this first
**Verdict: free, immediate, theory-backed, a direct extension of the compression mechanism.**

- **Data.** Scorecard FoS (in hand — confirm the 4yr earnings column exists, name ≈ `EARN_MDN_4YR`,
  against the data dictionary); the PSEO Earnings table (`pseoe`, via the Census PSEO Explorer / API),
  institution × CIP × degree × {y1, y5, y10}.
- **Pipeline.**
  1. For each horizon h: placement vector = median earnings@h over (institution × field).
  2. Merge the per-institution SpringRank the core paper already has (restricted to that field's departments).
  3. For each field f: `coupling_f(h) = Spearman ρ(prestige_i, earnings_i@h)` across institutions teaching f.
  4. Obtain the short curve coupling_f(h) per field.
  5. Test: `coupling ~ horizon × field_type`, comparing the **curve shapes** of the compression vs
     integrated groups (decay / pinned-at-zero / rising).
- **Output.** coupling(t) curves grouped by field_type; the headline is the shape difference across the
  three signatures.
- **⚠️ Load-bearing caveat.** In Scorecard, **the 1yr and 4yr columns of the same release are different
  cohorts** → comparing them directly conflates career time with calendar time. **Only PSEO, tracking one
  cohort at 1/5/10, is clean.** Design: PSEO = the clean version, Scorecard = the coverage version, each a
  robustness check on the other. Suppression (cells with n<30 dropped) thins small fields; each field needs
  ≥15–20 institutions per horizon for a stable ρ.

### 🅢② re-run the dead continuous law with the *right* variable — most worth doing
**Verdict: win-win. The pilot's null was the wrong variable fed in.** The mechanism variable is **not
occ_hhi** (we proved it ≈ the license dummy) — it is **within-occupation wage variance.**

- **Data.** ACS PUMS **5-year** (IPUMS). Variables: `FOD1P` (bachelor's field of degree), `OCCP`, `INDP`,
  `WAGP`/`PERNP` (wage), `ST`/`PUMA`, `AGEP` (experience proxy), `SCHL` (restrict to bachelor's+),
  `SEX`/`RAC1P` (residualization), `WKHP`/`WKWN`/`ESR` (restrict to full-time full-year), `PWGTP` (weights).
- **Pipeline.**
  1. Sample: bachelor's+ (`SCHL≥21`), FTFY (usual hours ≥35 and ≥50 weeks), employed, positive wage.
  2. Residualize: `log(wage) ~ state FE + experience polynomial + demographics` → take residuals (strips out
     geography + composition, leaving only within-cell dispersion).
  3. For each destination occupation (`OCCP`, optionally ×`ST`) compute the **residual SD** = "wage spread
     that survives within one occupation/place" = the proxy for prestige-sortable variance.
  4. For each field: build the field→occupation empirical distribution from `FOD1P × OCCP`.
  5. Field-level **"destination wage-dispersion index"** = the within-occupation residual SD weighted by the
     field's occupation distribution.
  6. Regress `coupling_f ~ dispersion_index_f` across fields. Prediction (compression theory): dispersion↑
     → coupling↑; dispersion≈0 (nursing → RN wage pinned) → decoupled.
- **Output.** coupling vs dispersion-index scatter, with nursing in the low-dispersion/low-coupling corner
  and CS in the high-dispersion/high-coupling corner; the regression coefficient + fit = **the continuous
  law the pilot failed to find.**
- **⚠️ Caveat.** ACS within-occupation *total* variance is an **upper bound** on prestige-sortable variance
  (it mixes firm effects + unobserved skill + measurement error) → directional evidence only. ACS has **no
  institution** → this validates the *mechanism variable* (destination side) only; it does not touch the
  prestige-vs-placement link itself. If it lights up → the law revives and the wrong-variable diagnosis is
  confirmed; if still null → it becomes **written testimony that aggregate data is only visible at the
  extremes and Revelio is genuinely necessary.** Both outcomes win.

### 🅐③ pry compression apart from licensing — UK vs US teaching
**Verdict: the blade points straight at the identification weak spot, but rank it third (vet LEO first).**
The US soft spot: licensed / unionized / public-pay-scale are highly collinear (teaching, social work hit
all three at once), so you can't tell which one is doing the work.

- **Data.** UK LEO (gov.uk "Graduate outcomes (LEO)") provider × subject earnings; UK prestige from the
  ORCID / faculty-hiring network already built during the nursing replication. US: Scorecard education CIP +
  SpringRank.
- **Pipeline.**
  1. `coupling_edu^UK = ρ(prestige, LEO earnings)` across UK providers.
  2. `coupling_edu^US = ρ(prestige, Scorecard earnings)` across US institutions.
  3. Both sides are licensed (license pins pay); UK = a national pay spine (compression maxed), US =
     district autonomy (high variance). Prediction: if compression drives decoupling, then
     `|coupling^UK| < |coupling^US|`.
  4. **US within-country strengthening:** group by state-level teacher pay-schedule rigidity (uniform state
     schedule vs district autonomy, NCTQ / state-policy data) and see whether education coupling moves with
     rigidity.
- **Output.** UK-vs-US contrast + within-US rigidity gradient → separating compression from licensing.
- **⚠️ Caveat.** Cross-national comparability (different prestige networks, different earnings
  definitions/currency, CAH vs CIP taxonomies) → suggestive, not decisive. **First vet whether LEO delivers
  provider×subject at the needed grain and whether a UK prestige network can be built at that grain** —
  hence the later ranking. (Union compression, Freeman 1980, is theoretically also a compression source, but
  collinear as above — which is exactly why the contrast design is needed.)

### 🅐④ acquit bio — a second placement axis (NSF SED)
**Verdict: a positive patch to the placement-proxy pillar; reviewers will like it.** Bio is currently
labelled "decoupled," but its placement market is the **PhD / med-school pipeline**, invisible to year-1/4
wages (grad students earn a stipend). The data to fix it is **open and free.**

- **Data.** NSF NCSES **SED baccalaureate origins** (counts of doctorate recipients by undergraduate
  institution × doctoral field, via NCSES data tables / WebCASPAR); denominator from IPEDS completions
  (bachelor's degrees conferred by institution × CIP).
- **Pipeline.**
  1. For each undergraduate institution × field, the **PhD-production rate** = SED count (numerator) / IPEDS
     bachelor's count (denominator), loosely cohort-aligned (PhD lags the bachelor's by ~6–8 years).
  2. `coupling_f^PhD = ρ(prestige_i, PhD-production-rate_i)` across institutions.
  3. Contrast bio's coupling on the earnings axis (≈decoupled) vs the PhD axis.
- **Output.** A dual-axis coupling table; if bio is **strongly coupled on the PhD axis**, "decoupled" is
  reclassified as **"coupled to a different (delayed) market."**
- **⚠️ Caveat.** The SED baccalaureate-origins table may only go to broad field — check the grain; the
  production rate **must** be normalized (never raw counts, which large institutions would dominate).

### 🅑⑤ quantile coupling (a probe done in passing)
**Verdict: a zero-cost directional probe, not a headline.**
- **Data.** Scorecard/PSEO p25/p75 columns (in hand).
- **Pipeline.** Compute coupling separately at p25/p50/p75 per field; compare `coupling(p75)` vs `coupling(p50)`.
- **Output.** If coupling rises with the quantile (especially for integrated fields) → directional evidence
  for an elite-tail channel.
- **⚠️ Caveat.** p75 ≠ a true p99 elite tail; institution-level quantiles are within-program spread, a
  coarse proxy.

### 🅒⑥ PERM green-card data (flagged dead — don't write a word before verifying)
- **Step 0.** Download the latest raw DOL/OFLC PERM disclosure file and **confirm whether the education
  fields actually contain an institution name.** No work until that is confirmed.
- If yes: free individual-level "institution × wage × employer." Selection is huge (a sponsored-immigrant
  job slice) → at most a selected-sample robustness check.

### 🗑️ Discarded (don't exhume the corpses)
- **Do not reopen behavioural.** The six-family data scan still stands — free data cannot power a
  within-field behavioural result; the UCAS revealed-demand line is cleanly dead (see the 45/46 no-go in
  `results`).
- **Cohort-axis COVID / travel-nursing shock.** Cute, but first check whether the data vintage reaches
  2021; only then. And travel pay sorts on certification, not alma mater — most likely just reinforces the
  compression story anyway.

---

## 3. What it assembles into + what's left for Revelio

①②③④ are not four parts — they are **one coherent Act Two:**
**"the compression mechanism — when / why / where prestige stops paying"** = the temporal dimension
(learning, ①) + the variance dimension (dispersion, ②) + the institutional dimension
(license / pay-scale / union, ③) + a placement redefinition (④). **All entirely Revelio-free.**

**Revelio is squeezed down to three exclusives** (for the Revelio pitch):
1. individual trajectories (who is climbing; firm-to-firm jumps);
2. the true firm elite tail above p75;
3. the **prestige-sortable decomposition** of within-occupation variance (ACS gives only the upper bound;
   Revelio has institution + firm to decompose it cleanly).

**⇒ The Revelio ask sharpened:** "I finished all of Act Two on aggregate data — I only need these three
things" is ten times sharper than "I need multi-dimensional placement." It reads as someone coming to
*close out* the work, not to *ask for* data.

---

## 4. Suggested order + magnitude

1. **①** career-time coupling — off-the-shelf Scorecard/PSEO, a few days.
2. **②** re-run the continuous law with the dispersion index — ACS PUMS, one to two weeks, **highest value.**
3. **④** acquit bio — NSF SED + IPEDS, one week.
4. **⑤** folded into ① in passing.
5. **③** UK-vs-US — **vet LEO grain first**, schedule if it passes.
6. **⑥** only if you personally confirm the PERM fields carry an institution name.

---

## 5. Platform frontier (`paper/next_stage_agenda.tex` — three thrusts, complementary to the above)

The static paper is a **platform**; the placement axis is currently four things to relax — **a single
scalar, at the institution grain, US-only, and median-level** (plus disconnected from student beliefs).
Three thrusts, ordered by access cost:

- **Thrust A — cross-national external validity.** Runs on public data now (= ③ above: UK LEO + a rebuilt
  prestige network).
- **Thrust B — the multi-dimensional, within-field placement gap (the flagship).** **Revelio-gated;** the
  residual-mechanism, tail, and dynamic questions all converge here (= the three Revelio exclusives).
- **Thrust C — the behavioural layer.** Descriptive / quasi-experimental tiers run on public data now (but
  42–46 already showed they're under-powered); the flagship RCT is gated on a subjective-expectations
  partnership.

**Two leverage points.** A Revelio subscription unlocks B + the placement half of the dynamic question (the
prestige-temporal half needs a separate dynamic-SpringRank upgrade, edge density its binding constraint =
axis ③); a subjective-expectations partnership unlocks C.

---

## 6. Immediate next steps (pick up and go)

- [ ] **① start.** Confirm the Scorecard `EARN_MDN_4YR` (and 1yr/5yr) columns against the data dictionary,
      stand up the `coupling_f(h)` skeleton (reuse `src/gap.py` + the existing SpringRank); fold ⑤'s p25/p75
      in on the way.
- [ ] **② main thrust.** Pull ACS PUMS 5-year (IPUMS), build the "destination wage-dispersion index" per the
      6 steps above, run `coupling_f ~ dispersion_index_f`. This is the **highest-value** shot and the most
      likely to revive the continuous law.
- [ ] **④.** Locate the NSF SED baccalaureate-origins table + IPEDS completions, check the grain, build
      bio's PhD-axis coupling.
- [ ] **③ prerequisite.** Vet UK LEO provider×subject grain + whether a UK prestige network is buildable at
      that grain (schedule only if it passes).
- [ ] **⑥ prerequisite.** Get the latest PERM disclosure and confirm whether the education fields carry an
      institution name (otherwise, not a word).
- [ ] **Data handoff.** Ship raw+interim or fetch raw via SOURCES.md (see README §4 footprint); axis ③'s
      state-space smoothed ranking (WHR/TTT) is a self-contained methods sub-project — a natural piece to
      spin off.
- [ ] **Co-author decisions.** The six items in `paper/COAUTHOR_NOTES.md` (venue / so-what / keep-or-cut the
      model / do UK first or cite as next step / outreach targets / scope of the CS-ranking artifact).
