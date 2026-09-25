r"""scripts/75_theory_tests.py -- registered tests T0-T8 of the theory section (implementation).

Status (2026-09-25): the binding texts below are unchanged. Two blocks are copied verbatim from THEORY_NOTES.md
section 9: the original specification B0 (between the PRESPEC marker lines) and amendment A1 (between the
AMENDMENT A1 marker lines), written the same day after an adversarial review and before any statistic named in
either was computed. The SHA-256 of each block (UTF-8, the text strictly between its marker lines, including the
final newline) is PRESPEC_SHA256 and A1_SHA256 below; main() refuses to run if either digest does not match.
The implementation follows the blocks; every clarification and deviation is listed in THEORY_TESTS_RESULT.md
(section "Deviations and clarifications"). The A1 preamble asks the author to deposit this file on a public
time-stamped record before any test is run; this workflow may not commit, so the tests were run without that
deposit. The result file states this and records the SHA-256 of the pre-implementation stub (which already held
both blocks) and of this file at run time.

Run (all tests, then the report):
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
      .venv/bin/python scripts/75_theory_tests.py [--workers 4]
  Subsets: --only T0,T1 --cache DIR (writes DIR/<test>.json), then --report --cache DIR (THEORY_TESTS_RESULT.md
  from the cached results; no statistic is recomputed; --compare DIR2 records a byte-for-byte comparison of the
  per-test JSON with a second run's cache). Every number is printed as it is computed.
Revision 1 (2026-09-25, after an independent verification): converged degree-corrected BT fits, the literal-order
  T1(a') benchmark computed, per-band T4 and per-field T6 estimates, validity diagnostic in T0 (see the Revisions
  section of THEORY_TESTS_RESULT.md). The binding texts below are unchanged.
Outputs: THEORY_TESTS_RESULT.md (repo root, local only). Nothing else is written unless --cache is given.
Public data only; existing scripts are imported, never edited; descriptive, not causal.

-----BEGIN PRESPEC-----
PRE-SPECIFICATION OF THEORY TESTS T1-T3 (binding text)
Degrees of Separation, theory section (paper/theory_section.tex, paper/theory_si.tex).
Written 2026-09-25, before any statistic named below was computed.

Outcome-blind checks made before this text was written (none computes a statistic named below):
(1) every row of the public Wapman edge list has Total >= 2; self-hire rows exist; 9.4% of persons have no
    recorded gender (Total - Women - Men > 0);
(2) no script in scripts/ or src/ reads the Women/Men columns, GiniCoefficient, FractionUpHierarchyHires or
    ProductionRank;
(3) ORCID feasibility count over all 681 shards, computing no destination rank and no earnings: persons with a
    US bachelor's episode and a later US doctorate episode at a different ROR institution, both matching the
    same field keyword, number 968 (Economics) to 10,112 (Chemistry); bachelor's-institution cells with at least
    3 persons number 100 to 701 per field, before matching to Wapman institutions;
(4) UGDS is present in the Scorecard institution file and C2023_a.csv in data/raw/ipeds.

GENERAL RULES
G1. The tests are implemented in scripts/75_theory_tests.py, whose docstring reproduces this text verbatim.
    Results go to THEORY_TESTS_RESULT.md, and every verdict is reported whatever it is.
G2. Randomness: numpy SeedSequence(75), spawned once per test and once per replicate, so that results do not
    depend on the number of workers.
G3. Intervals are 95% percentile intervals from B = 1000 draws unless stated otherwise. MDE80 = 2.80 x the
    bootstrap SE of the primary statistic; it is written to the result file before the point estimate.
G4. Every deviation from this text is listed, with its reason, in a Deviations section of the result file. An
    estimate produced by a deviation never replaces the pre-specified estimate in a verdict.
G5. Percentiles: for a Wapman ranking with N ranked institutions, pct = 1 - Rank/(N - 1), so 1 is the top.
    F = field-level pct and G = Academia-level pct, both from data/raw/wapman2022/ranks.csv.

T1. SINGLE-INDEX AUDIT OF THE HIRING MEASURE (Proposition 1a). No earnings data are read.
Data. data/raw/wapman2022/edge_lists.csv, TaxonomyLevel == "Field", every field. Rows with
DegreeInstitutionId == InstitutionId (self-hires) are dropped. Each row is expanded into Total persons.
T1(a) Role consistency.
  For institution u in field f:
  - P_u = mean field-f pct of the employers of persons whose degree is from u (placement);
  - H_u = mean field-f pct of the degree institutions of persons employed at u (attraction).
  Persons whose counterpart institution has no field-f rank are dropped. Eligible institutions have at least
  4 persons placed and 4 hired in the full public network.
  Persons are split 200 times into halves A and B by independent Bernoulli(1/2) draws. In each split and field,
  over eligible institutions with P and H defined in both halves:
  - r_PP = Spearman(P_A, P_B) and r_HH = Spearman(H_A, H_B);
  - r_PH = [Spearman(P_A, H_B) + Spearman(P_B, H_A)] / 2.
  N_f = mean of r_PH over splits, and D_f = sqrt(max(mean r_PP, 0) x max(mean r_HH, 0)). A field is excluded
  if it has fewer than 15 eligible institutions or D_f < 0.20; the number excluded is reported.
  Statistic: R = sum_f N_f / sum_f D_f. Interval: resample fields with replacement (B = 1000) and recompute R
  from the stored N_f and D_f.
  Prediction: one index governs both roles, so R is close to 1.
  Rule: supported if the lower bound of R is >= 0.80; contradicted if the upper bound is < 0.80; inconclusive
  otherwise.
T1(b) Gender invariance.
  Only persons of recorded gender are used. For each field, SpringRank (src/ar_pipeline/springrank.springrank,
  alpha = 1e-3, as in scripts/56) is fitted separately to the men's network and to the women's network.
  c_f = Spearman of the two score vectors over institutions with at least 3 persons (in plus out) in each
  network. A field enters if at least 30 institutions qualify; K is the number of fields entering, and it is
  reported.
  Null: 200 permutations of the gender labels across persons within the field, holding the numbers of men and
  women fixed and applying the same qualification rule to each permuted split.
  z_f = (c_f - mean_null) / sd_null, and p_f = (1 + #{null <= c_f}) / 201.
  Prediction: the ordering does not depend on whose hires are used, so c_f lies within permutation noise.
  Rule: supported if (i) the number of fields with p_f < 0.05 is at most the 95th percentile of
  Binomial(K, 0.05) and (ii) mean z_f >= -1.645 / sqrt(K); contradicted if both (i) and (ii) fail;
  inconclusive otherwise.
T1 verdict. The single-valuation reading of F passes if (a) and (b) are both supported. It fails if either is
contradicted, and it is undecided otherwise. The Academia network is analysed the same way and reported, but
it does not enter the verdict.

T2. DOUBLE DISSOCIATION (Propositions 1-3).
Data. ORCID mobility edges (data/orcid/all/edge_aff, 681 shards; li2026orcid). Use education episodes with
org_country == "us" and a ROR id, deduplicated by (person_orcid, affiliation node id).
  - A bachelor's episode has a role that matches BAC and does not match DOC.
  - A research-doctorate episode has a role that matches DOC and does not match PROF.
  The regexes are case-insensitive:
    DOC  = ph\.?\s?d|d\.phil|dphil|sc\.?d|\bdoctor|doctoral   (as in scripts/10)
    BAC  = bachelor|\bb\.?\s?s\.?c?\b|\bb\.?\s?a\.?\b|\bbsc\b|\ba\.?b\.?\b|\bs\.?b\.?\b|undergrad|\bbs\b|\bba\b
    PROF = \bm\.?\s?d\b|medicine|medical|pharm|juris|\bj\.?\s?d\b|\blaw\b|dental|\bdds\b|\bdmd\b|osteopath|
           nursing|\bdnp\b|physical therap|\bdpt\b|audiolog|psy\.?\s?d|\bed\.?\s?d\b|veterinar|\bdvm\b|optometr|
           chiropract|occupational therap|ministry|\bd\.?\s?min\b|\bdba\b|business admin|public health|\bdrph\b
Fields. The frozen keyword map of scripts/17 is used:
  Computer Science "computer"; Mathematics "math"; Physics, General "physics"; Psychology, General "psycholog";
  Economics, General "econ"; Chemistry "chemist"; Biological Sciences, General "biolog".
  A keyword matches if it is a substring of the lower-cased (role + " | " + department) of an episode. A person
  enters field f if the bachelor's episode and the doctorate episode both match f; a person may enter several
  fields. The doctorate must be at a different ROR institution, and it must not start before the bachelor's
  when both start years are recorded. There is one record per person x bachelor's institution x field, using
  the earliest qualifying doctorate.
Matching. ROR names are matched to Wapman institutions with src.crosswalks.institutions.
normalize_institution_name (as in scripts/11), and the match rates are reported.
Variables.
  - D_if = mean field-f pct of the doctorate institutions of the persons in cell (i, f). Persons whose doctorate
    institution has no field-f rank are dropped.
  - F_if and G_i are the pct values of the bachelor's institution.
  - A cell needs F_if, G_i and at least 3 persons. A field needs at least 15 cells, and at least 4 fields must
    enter; otherwise the verdict is "infeasible".
  - Pay: Y_if = log EARN_MDN_4YR from build_table of scripts/28, on the same cells.
Statistics (Spearman over the cells within a field).
  - Delta^D_f = rho(F, D) - rho(G, D).
  - rho(F, D | G), the partial Spearman correlation.
  - Delta^Y_f = rho(F, Y) - rho(G, Y), computed on the cells with Y, in fields with at least 15 such cells.
  Means are unweighted over fields. The mean of (Delta^D - Delta^Y) is taken over the fields that have both.
Inference. Crossed field x institution bootstrap: fields and institutions are resampled independently, B = 1000,
as in scripts/59. Per-field estimates with institution-bootstrap intervals are also reported.
Predictions. The academy sees departments, so mean Delta^D > 0 and mean rho(F, D | G) > 0. For pay the market
prices names, so mean (Delta^D - Delta^Y) > 0.
Rule. Supported if the intervals of mean Delta^D and of mean (Delta^D - Delta^Y) both lie above 0. Contradicted
if the interval of mean Delta^D lies below 0, or if the upper bound of mean rho(F, D | G) is below +0.05.
Inconclusive otherwise, with MDE80 reported.
Fixed reading. Support shows that the academy's choice among the same bachelor's graduates tracks department
standing more closely than the market's does. It does not separate information (the academy sees departments)
from content (department standing matters more for research preparation than for market productivity). F is
less reliable than G, so a positive Delta^D is conservative.

T3. WHERE THE CAREER RISE OCCURS (Proposition 4).
Data. The PSEO V4.14.1 fixed-cohort panel of scripts/66, sample A (cohorts 2001, 2004, 2007, 2010; horizons y1,
y5, y10; its minimum-cell rule NMIN = 15), built by import.
  - Primary visibility: V_i = log UGDS, from the Scorecard institution file.
  - Secondary visibility: V_i = log of total first-major bachelor's completions (IPEDS C2023_a, AWLEVEL 5,
    MAJORNUM 1).
  - Institutions without V are dropped from a cell before the minimum-cell rule is applied.
Design.
  1. In each field x cohort x horizon cell, rank-transform log p50 (Y), G and V within the cell.
  2. V_perp = the residual of rank(V) on rank(G). Standardise G and V_perp.
  3. Fit OLS: Y ~ G + V_perp + G*V_perp. b_GV = the coefficient on the product.
  4. For each field x cohort, take the slope of b_GV on years since graduation (1, 5, 10). Average the slopes
     over cohorts within a field.
  5. theta_GV = the unweighted mean over fields.
Inference. The two-way (field, institution) variance of scripts/59, B = 1000. MDE80 is reported first.
Secondary analyses (reported; they do not enter the verdict): (i) V from IPEDS completions; (ii) the scripts/66
S2 controls added (SAT, -ADM, Pell, control, state level); (iii) the y5 -> y10 segment alone.
Predictions.
  - Learning about names that employers rarely see (omega_i < 1 at low V) lets the G gradient catch up at
    low-visibility institutions: theta_GV < 0.
  - Status-correlated growth concentrated at nationally recruited institutions: theta_GV > 0.
  - Precisely known names, with growth unrelated to visibility: theta_GV near 0.
Rule.
  - "Learning signature" if the interval lies below 0. Proposition 4's baseline (omega_i near 1) then fails for
    low-visibility institutions.
  - "Access signature" if the interval lies above 0.
  - "Null" if the interval includes 0 and MDE80 <= 0.019/yr (half the unconditional dG of 0.038/yr).
  - "Underpowered" otherwise.
-----END PRESPEC-----
-----BEGIN AMENDMENT A1-----
AMENDMENT A1 TO THE PRE-SPECIFICATION OF THEORY TESTS (binding text)
Degrees of Separation, theory section (paper/theory_section.tex, paper/theory_si.tex).
Written 2026-09-25, after an adversarial review of the theory and of the binding text B0 (SHA-256
bfaa08a283f6ebb5aa96face4d3e8768e2911572066750588d97210083243441), and before any statistic named in B0 or in
A1 was computed. No test of B0 or A1 has been run. The date is self-attested: the digest of this text lets a
reader detect later edits; it does not establish when the text was written. Before any test is run, the author
deposits B0 and A1 on a public time-stamped record (a commit of scripts/75_theory_tests.py to the public
repository, or an OSF or Zenodo deposit) and records the commit hash or DOI in THEORY_TESTS_RESULT.md.

A0. STANDING OF B0 AND A1
A0.1 B0 is not edited. Every B0 statistic and verdict is computed as B0 states, with the clarifications of A2,
     and is reported under the label "B0 verdict".
A0.2 A1 adds registered statistics and tests, each with its own verdict ("A1 verdict"). Each item is tagged:
     [A] resolves an ambiguity in B0 and changes nothing B0 states explicitly; [F] feasibility; [V] validity
     addition made in response to review, before any data were examined. [V] items never change a B0 verdict.
A0.3 The paper's claims follow the claim map of A9, which states which verdict governs each claim. B0 and A1
     verdicts are always reported side by side.
A0.4 B0's G1-G5 apply to A1 unless A1 states otherwise. Randomness for A1: numpy SeedSequence(751), spawned once
     per A1 test and once per replicate. Where A1 uses scripts/59's two-way (field, institution) variance, the
     interval is normal (estimate +/- 1.96 SE), MDE80 = 2.80 SE, a non-positive two-way variance is replaced by
     the larger one-way variance (flagged, scripts/59's fallback), and an endpoint within 2 Monte Carlo SE of 0
     is flagged "edge".
A0.5 Erratum to B0's outcome-blind check (1): 9.4% is the share of persons without recorded gender pooled over
     the Academia, Domain and Field rows, which count the same persons several times. At TaxonomyLevel ==
     "Field" the share is 6.2%, and 5.3% after dropping self-hires.
A0.6 Data already analysed for other results. T0 and T1 use the Wapman edges (split-half SpringRank in
     scripts/56). T2 uses F, G and Scorecard pay analysed in scripts/28 and 56; Delta^Y is already known for all
     seven T2 fields (computer science -0.045, mathematics +0.004, physics -0.052, psychology -0.035, economics
     -0.005, chemistry -0.073, biology -0.091), so only the D side of T2 is new. T3 uses the PSEO sample-A panel
     of scripts/66 and was chosen knowing the rise of the G loading. T4 uses the UK band panels of scripts/62,
     where the within-band rise of raw G coupling (+0.017/yr) is known. T5 uses Scorecard pay; its selectivity
     analogue (true-score r +0.94, scripts/55) is known. T6-T8 use PSEO Flows, whose national sector shares were
     analysed in scripts/61.

A2. CLARIFICATIONS OF B0 [A]
A2.1 T1: "every field" means every TaxonomyValue with TaxonomyLevel == "Field".
A2.2 T2 episodes: edge_aff rows are edges between two episodes of a person. Episodes are rebuilt by stacking the
     "from" and "to" endpoints (aff_node_id, org_country, org_ror_id, org_ror_name, role, role_type, org_dept,
     epi_start_year, each taken from its _from or _to column) and deduplicating by (person_orcid, aff_node_id).
     B0's "org_country" means org_country_from/_to, its "affiliation node id" aff_node_id_from/_to, its "ROR id"
     org_from_ror_id/org_to_ror_id, and an education episode has role_type == "education".
A2.3 T2 doctorate episodes: B0 defines them by intent (research doctorates) and by a regex whose term "doctoral"
     also matches postdoctoral and predoctoral roles. The intent governs: an episode whose role matches
     POSTPRE = post\W?doc|pre\W?doc (case-insensitive) is not a doctorate episode. The literal B0 regex is run as a
     sensitivity and reported.
A2.4 T2 field keys for Y: the Wapman field maps to the scripts/28 key through the wapman_field attribute of
     src/crosswalks/fields.py: Computer Science -> computer_science; Mathematics -> mathematics; Physics, General
     -> physics; Psychology, General -> psychology; Economics, General -> economics; Chemistry -> chemistry;
     Biological Sciences, General -> biology.
A2.5 T2 partial Spearman: the Pearson correlation of the within-field rank residuals of the two variables on
     rank(G), as in scripts/55. The Gaussian-copula partial (from r = 2 sin(pi rho / 6)) is a reported sensitivity.
A2.6 T2 inference: B0's crossed bootstrap is run as B0 describes it (fields and institutions resampled
     independently, B = 1000, percentile intervals). scripts/59's two-way normal interval is reported alongside.
A2.7 T2: "MDE80 of the primary statistic" refers to mean Delta^D.
A2.8 T2: if B0's "supported" and "contradicted" conditions both hold, the B0 verdict is "inconclusive".
A2.9 T3: Y in step 1 is the within-cell rank of log p50 standardised within the cell (mean 0, SD 1), as in
     scripts/66.
A2.10 T3: the interval is scripts/59's normal two-way interval with the conventions of A0.4 (B0 names that
     variance, so G3's percentile default does not apply). The 0.019/yr threshold is kept as registered.

A3. T0. PRODUCTION AUDIT OF F [V]
Question. Is F an ordering by the direction of exchange net of volume, or by production-to-hiring volume? With
no valuation and random matching, E A_uv is proportional to out_u x in_v; every pair then satisfies B0's
vertical-sorting assumption with departments ordered by out/in, and SpringRank, which has no degree correction,
ranks net exporters higher.
Data. Wapman public edges, TaxonomyLevel == "Field", self-hires dropped; ranks.csv; IPEDS C2023_a (AWLEVEL 17,
CTOTALT, all MAJORNUM) summed over each field's cip4 codes in src/crosswalks/fields.py (fields without a cip4
entry are skipped); institution-stats.csv ProductionRank joined by (TaxonomyValue, OrdinalPrestigeRank) as in
scripts/09. IPEDS UNITIDs are matched to Wapman institutions by normalize_institution_name, as scripts/28
matches Scorecard institutions.
Variables (field f, institution u): p_u = log(1 + research doctorates in f's CIP codes); x_u = log((d_out_u + 1)
/ (d_in_u + 1)) in the public field network (export ratio); q_u = d_out_u / max(doctorates, 1) (placement rate).
(a) No earnings. Per field with >= 15 institutions: Spearman of F with p, x, q and -ProductionRank. Means over
    fields; intervals by resampling fields (B = 1000).
(b) Degree-corrected ranks. Per field, a Bradley-Terry model of the direction of exchange with an offset:
    A_uv | T_uv ~ Bin(T_uv, logistic(theta_u - theta_v + o_uv)), o_uv = log(d_out_u d_in_v / (d_out_v d_in_u)),
    ridge penalty 1e-3 on theta; F_DC = theta. F_perp = residual of rank(F) on rank(p) and rank(x) within the
    field. On the scripts/28 Scorecard 4-yr cells: c_F = Spearman(F, Y), c_DC = Spearman(F_DC, Y), c_perp =
    Spearman(F_perp, Y); G_DC from the same fit on the Academia network, and c_GDC. Statistics: mean
    Spearman(F, F_DC); retained shares R_DC = mean c_DC / mean c_F and R_perp = mean c_perp / mean c_F; mean
    (c_GDC - c_G). Interval: scripts/59 two-way variance.
(c) No earnings. Minimum-violation ordering per field by local search from the SpringRank order (adjacent swaps
    and random reinsertions until no move lowers the count of upward hires; 50 random restarts; the best kept).
    Mean Spearman(F, MVR) and the upward-hire counts of both orderings.
Rule (b). "Production-robust" if the lower bounds of R_DC and R_perp are both >= 0.75. "Production-dominated" if
the upper bound of either is < 0.50. "Partly production" otherwise. MDE80 of R_DC is reported first.

A4. T1 ADDITIONS [V]
A4.1 T1(a') cross-fitted and benchmarked. In each of 100 person splits (Bernoulli(1/2)), SpringRank (alpha =
     1e-3) is fitted on half A and on half B; P and H of half A use percentiles of the ranks fitted on half B,
     and vice versa; R' is formed from these as B0 forms R. Benchmark: per field, fit SpringRank to the full
     public network (s-hat) and beta by maximising sum_{u,v} A_uv log logistic(2 beta (s-hat_u - s-hat_v));
     simulate 100 networks A*_uv ~ Poisson(c exp(-beta/2 (s-hat_u - s-hat_v - 1)^2)), c matching the field's
     total count; drop A* < 2 (the public censoring); apply B0's eligibility rules; compute R' on each network
     with the identical pipeline. R0 = mean over simulations. Statistic Delta_R = R' - R0; interval by
     resampling fields jointly for the real and simulated N_f and D_f (B = 1000). Rule: "one index" if the lower
     bound of Delta_R >= -0.10; "not one index" if the upper bound < -0.10; "inconclusive" otherwise.
A4.2 T1(b) is an audit of H3 (steering by subfield, geography or dual careers that differs by gender), not of a
     single valuation: a uniform gender penalty in placement leaves both orderings unchanged. T1(b') adds an
     equivalence rule. d_f = mean_null(c) - c_f; dbar = mean of d_f over the K fields; SE = sqrt(sum_f
     sd_null_f^2) / K; MDE80 = 2.80 SE, reported first. "Invariant" if dbar + 1.645 SE <= 0.10; "not invariant"
     if dbar - 1.96 SE > 0; "uninformative" otherwise.

A5. T2 ADDITIONS [V]
A5.1 T2 is a consistency check with one new prediction, not a double dissociation (A0.6). Auxiliary assumption:
     the ordering that governs faculty hiring in field f also governs doctoral admission in f and applicants'
     choice among doctoral programs in f. Rivals that predict the same sign: advisor-network and subfield
     routing; geography (doctorates near the bachelor's institution, with F correlated with region); content
     (strong departments prepare researchers better); applicants' self-selection; same-scale bias (D is on F's
     scale, and SpringRank errors are correlated between connected nodes).
A5.2 T2' statistics, on B0's T2 cells: Pi^D = mean over fields of rho(F, D | G); Pi^DG = the same with D^G, the
     mean Academia-level pct of the persons' doctorate institutions; Pi^Y = mean rho(F, Y | G) on the cells with
     Y; K = mean over fields with both of [rho(F, D | G) - rho(F, Y | G)]. Partials as in A2.5.
A5.3 Inference: scripts/59 two-way variance; a leave-one-field-out jackknife interval is reported alongside.
A5.4 Rule. "Supported" if the lower bounds of Pi^D, Pi^DG and K are all > 0. "Contradicted" if the upper bounds of
     Pi^D and Pi^DG are both < 0. "Inconclusive" otherwise, and also when a supported and a contradicted
     condition hold at once. MDE80 of Pi^D is reported first.
A5.5 Reported with T2' (no verdict): split-half reliability of D (persons split at random within cells, 200
     times; Spearman across cells; Spearman-Brown); Delta^D corrected for reliability, rho(F,D)/sqrt(rel_F) -
     rho(G,D)/sqrt(0.99), at both ends of scripts/56's rel_F bracket; the share of persons whose doctorate is at
     their bachelor's institution (dropped by B0) by tercile of F; Pi^D with the cell's share of doctorates in
     the bachelor's institution's state as an added control; Pi^D with word-anchored keywords (\bcomputer,
     \bmath, \bphysics, \bpsycholog, \becon, \bchemist, \bbiolog).
A5.6 A B0 "contradicted" verdict that arises from its Delta^D clause while the reliability-corrected Delta^D of
     A5.5 is not below 0 at both ends of the bracket is reported as "contradicted as registered; attributable to
     reliability".

A6. T3' [F][V]
A6.1 Reasons. B0's "null" requires MDE80 <= 0.019/yr, which the two-way SEs of comparable slopes in scripts/66
     (0.0086 to 0.0139) make nearly unreachable. B0's sign mapping is not exclusive: if employers' entry prior
     leans on status (a halo), learning makes theta_GV > 0. Low UGDS also marks small private institutions that
     recruit nationally.
A6.2 Design: B0's T3 steps 1-5 on the same sample-A cells, with Y, G and V as in A2.9, and with public/private
     control C (Scorecard CONTROL 1 against 2 or 3) and G x C added to the regression.
A6.3 Statistics: theta' = per-year slope of b_GV, averaged as in B0; g1 = mean b_GV at y1 (the entry gap in the
     G gradient between high- and low-visibility institutions); simple slopes dG_low = slope of (b_G - b_GV) and
     dG_high = slope of (b_G + b_GV), the G gradients at V_perp = -1 and +1; theta' within public and within
     private institutions separately (cells with >= 10 institutions of the type).
A6.4 Power, computed and written before any estimate: MDE80 of theta' = 2.80 x SD of theta' over 200
     permutations of the Y ranks within cells (outcome-blind: the standardised ranks are fixed numbers).
     Benchmark = per-year slope of the G-only standardised coefficient on the same cells.
A6.5 Decision table (intervals as in A0.4):
     - g1 > 0 and theta' < 0: "learning signature" (the G gradient catches up where names are rarely seen),
       provided theta' has the same sign in public and in private institutions (point estimates); otherwise
       "catch-up not robust to control type".
     - g1 > 0 and theta' > 0: "access or growth at high visibility" (the gap widens).
     - g1 < 0 and theta' > 0: "halo correction or access" (not separated).
     - g1 < 0 and theta' < 0: "growth concentrated at low visibility".
     - g1 interval includes 0, theta' interval excludes 0: theta' < 0 reads "catch-up at low visibility (entry
       gap not detected)"; theta' > 0 reads "consistent with access or halo correction".
     - theta' interval includes 0: "null" if MDE80 <= half the benchmark, "underpowered" otherwise.
A6.6 Secondary (reported, not in the verdict): (i) partial-correlation form: per cell, the partial Spearman of Y
     with G x V_perp given G, V_perp, C and G x C; (ii) V = share of the institution's employed graduates working
     outside its state at y1 (PSEO Flows, bachelor's, all CIP, pooled cohorts, national row, all industries; one
     minus the employment-weighted in-state share, the scripts/65 formula); learning predicts faster catch-up
     where this share is high, and national recruiting predicts the same sign, so this measure does not
     discriminate on its own; (iii) B0's IPEDS completions V.
A6.7 B0's T3 labels are reported as registered; in the paper, B0's "access signature" is read as "not consistent
     with learning from a prior that under-weights status; consistent with access or with halo correction".

A7. NEW TESTS [V]
T4. UK WITHIN-BAND DECAY OF THE SCHOOL-MEAN PREMIUM.
Data: scripts/62's UK LEO provider x subject x prior-attainment-band panels at YAG 1, 3 and 5 (balanced providers,
>= 15 providers per cell); S = institution selectivity (share of graduates with >= 360 UCAS points, scripts/62);
G = scripts/62's UK hiring-network rank.
Statistics: per subject x cohort x band cell, (i) the partial Spearman of the band median log earnings with S
given G (primary); (ii) the within-cell OLS slope of log band median on z(rank S) and z(rank G), in log points per
SD of S. Per-year slope over YAG 1/3/5, averaged over bands, then cohorts, then subjects. Interval: scripts/59
two-way (subject, provider) variance.
Prediction (employer learning with graduates' own attainment unseen by employers): the premium on S given own band
falls with YAG. If employers see attainment, or S carries value added or growth, it does not fall.
Rule (i): "decay" if the interval lies below 0; "rise" if above 0; "null" if it includes 0 and MDE80 <= 0.02/yr;
"underpowered" otherwise.

T5. INVARIANCE OF THE FIELD MAP TO NON-ACADEMIC STATUS INDICES.
Data: scripts/28 Scorecard 4-yr cells; Scorecard institution file: Z1 = AVGFACSAL, Z2 = INEXPFTE, Z3 = ENDOWBEGIN
/ UGDS.
Per field (>= 60 institutions with F, Y and all three Z): c_F = Spearman(F, Y) and c_Zj = Spearman(Zj, Y) on the
same institutions. 300 random halvings of institutions within every field; r_obs(j) = mean over halvings and both
directions of the Spearman across fields between c_F on one half and c_Zj on the other; rel_F and rel_Zj = mean
Spearman across fields between the two halves' maps; r*(j) = r_obs(j) / sqrt(rel_F rel_Zj). Interval: field
bootstrap of the whole procedure (B = 1000). Fewer than 10 fields: "infeasible".
Prediction (one institution factor): r*(j) near 1 for every j.
Rule: "supported" if the lower bound of r*(j) >= 0.70 for all three j; "contradicted" if the upper bound < 0.70 for
at least two j; "inconclusive" otherwise.

T6. QUANTITIES WHERE PAY IS SET BY SCHEDULE.
Data: PSEO Flows V4.14.1, bachelor's, institution x field (education: CIP-2 13; registered nursing: CIP 51.38,
CIP-2 51 where 51.38 is not released), pooled cohorts, y5, geo_level == "D"; ACS 2023 PUMS, full-time full-year
workers aged 23-35 with a bachelor's degree: log median wage of teachers (SOCP beginning 2520) and of registered
nurses (SOCP 291141) by census division of work (POWSP mapped to divisions).
Index: T_i = sum over divisions of (the field's share of employed graduates working there) x (that division's log
median wage of the field's scheduled occupation).
Statistic: mean over the two fields of Spearman(G, T) across institutions (>= 15 institutions per field).
Interval: institution bootstrap, B = 1000, percentile. Secondary: field-family prestige P of scripts/61 in place of
G; the share of employed graduates outside the scheduled industry (NAICS 61 for education, 62 for nursing).
Prediction (status acts on quantities where prices are scheduled): Spearman(G, T) > 0.
Rule: "supported" if the lower bound > 0; "contradicted" if the upper bound < +0.05; "inconclusive" otherwise.
MDE80 first.

T7. DESTINATION LOCATION.
Data: scripts/52's PSEO fixed-cohort panel (field x cohort x horizon cells); PSEO Flows division shares of the same
institution, cohort and horizon at the CIP code of the scripts/52 field (CIP-2 where the finer code is not
released); ACS 2023 PUMS: log median wage of full-time full-year bachelor's holders aged 23-35 by census division
of work (L_d).
Variables: W_h = sum_d share_d,h x L_d; O_h = 1 - in-state share at horizon h.
Statistics: (i) at y5, R_L = mean partial rho(F, Y | W) / mean rho(F, Y) on the same cells (primary); (ii) the
per-year slope over y1/y5/y10 of the partial given W_h, against the raw slope on the same cells; (iii) (i) and (ii)
with O_h and W_h together. Interval: scripts/59 two-way variance.
Rule (i): "location-robust" if the lower bound of R_L >= 0.75; "location-dominated" if the upper bound < 0.50;
"partly location" otherwise. (ii) and (iii): the share removed is reported, with no verdict.

T8. BETWEEN- AND WITHIN-SECTOR COUPLING (registered decomposition, no verdict).
Data: scripts/61's W cells (institution x CIP-2 family, y5, pooled cohorts); PSEO Flows sector shares (national
row); ACS 2023 PUMS national log median wage of full-time full-year bachelor's holders aged 23-35 by the same
sectors; C = log sum_k share_k x wage_k (scripts/61's sector-wage-mix index).
Statistics: raw coupling rho(P, log p50) and within-sector coupling rho(P, log p50 | C), means over families;
between-sector share B = 1 - within/raw. Interval: scripts/59 two-way (family, institution) variance. Reported
whatever it is. C was used in scripts/61 as a moderator, not as a control on coupling.

A9. CLAIM MAP (binding for the paper's text)
C1 "F orders departments by the direction of exchange, not by production volume": kept if T0 is "production-
   robust"; written as "partly production" if "partly production"; F renamed "placement position" if
   "production-dominated".
C2 "F behaves as one index of standing, as producer and as employer": governed by T1(a'). "One index": kept.
   "Not one index": withdrawn; F is then described as the ordering revealed by net placement (its position in
   the hiring network), and only the reading of F as a belief about the quality of a department's PhDs is
   dropped. "Inconclusive": undecided. B0's T1 verdict is reported alongside. T1(b) and T1(b') bear only on H3:
   "not invariant" is reported as evidence of gender-specific steering.
C3 "The academy's own choice among the same bachelor's graduates tracks the department-specific part of F": kept
   (as a consistency check, with the rivals of A5.1 named) if T2' is "supported"; withdrawn if "contradicted";
   undecided otherwise. B0's T2 verdict is reported alongside, read with A5.6.
C4 The source of the career rise: governed by T3' (A6.5). A "learning signature" rewrites Proposition 4's
   baseline for low-visibility institutions; "null" or "underpowered" leaves both readings open.
C5 "Employers price a school-mean premium beyond graduates' own attainment and learn it away" (employer
   reputation in the learning sense): supported by a T4 "decay"; a "rise" or "null" means the within-band
   premium does not decay, and the paper then does not describe it as reputation that learning corrects.
C6 "The field map is a map of how each field prices institution-wide status, whatever the index": T5.
C7 "Where pay is set by schedule, status acts on quantities": T6; "contradicted" withdraws this clause of
   Proposition 5.
C8 Construct claims about pay are written as "not netted of destination location" until T7 is run. After it,
   "location-dominated" restates them as agreement between the academy's ordering and where graduates work.
C9 T8's between-sector share is reported next to every construct statement about pay.
No verdict of B0 or A1 licenses a claim about the mechanism below the name (employers' information versus the
content of department standing): no registered test separates the two.
-----END AMENDMENT A1-----
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, binom
from scipy.optimize import minimize_scalar
from scipy.special import expit
from threadpoolctl import threadpool_limits

PRESPEC_SHA256 = "bfaa08a283f6ebb5aa96face4d3e8768e2911572066750588d97210083243441"
A1_SHA256 = "5e1b9d7ec132a29a9a722c1bebf86b49f94cd3b46cb95dff06f80122d2a498c1"
STUB_SHA256 = "948d7699efffdb7064aca850eec53ad8b956bb31754186463fd61906ab77c0ab"   # pre-implementation file
STUB_KEEP = "notes/prereg/75_theory_tests_stub_2026-09-25.py"                      # kept copy (git-ignored)


def _block(begin: str, end: str) -> str:
    return __doc__.split(begin + "\n", 1)[1].split(end, 1)[0]


def prespec_ok() -> bool:
    """True if B0 (the pre-specification block) is byte-identical to the one recorded on 2026-09-25."""
    body = _block("-----BEGIN PRESPEC-----", "-----END PRESPEC-----")
    return hashlib.sha256(body.encode("utf-8")).hexdigest() == PRESPEC_SHA256


def amendment_ok() -> bool:
    """True if amendment A1 is byte-identical to the one recorded on 2026-09-25."""
    body = _block("-----BEGIN AMENDMENT A1-----", "-----END AMENDMENT A1-----")
    return hashlib.sha256(body.encode("utf-8")).hexdigest() == A1_SHA256


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ar_pipeline.springrank import springrank as springrank_src          # noqa: E402
from src.crosswalks.institutions import normalize_institution_name as norm    # noqa: E402

EDGES = ROOT / "data" / "raw" / "wapman2022" / "edge_lists.csv"
RANKS = ROOT / "data" / "raw" / "wapman2022" / "ranks.csv"
ISTATS = ROOT / "data" / "raw" / "wapman2022" / "institution-stats.csv"
IPEDS = ROOT / "data" / "raw" / "ipeds" / "C2023_a.csv"
SC_INST = ROOT / "data" / "raw" / "scorecard_inst" / "Most-Recent-Cohorts-Institution.csv"
ORCID_DIR = ROOT / "data" / "orcid" / "all" / "edge_aff"
ROR_ZIP = ROOT / "data" / "raw" / "ror" / "ror-data.zip"
ACS = [ROOT / "data" / "raw" / "acs" / "psam_pusa.csv", ROOT / "data" / "raw" / "acs" / "psam_pusb.csv"]
PSEOF_OLD = ROOT / "data" / "raw" / "pseo" / "pseof_all.csv.gz"
PSEOF_NEW = ROOT / "data" / "raw" / "pseo_flows_2026q2" / "pseof_all.csv.gz"
REL56 = ROOT / "data" / "interim" / "prestige_reliability.csv"
UK_CAREER = ROOT / "data" / "interim" / "uk_leo_career_cells.csv"
OUT_MD = ROOT / "THEORY_TESTS_RESULT.md"

B = 1000                   # G3: bootstrap draws
Z975 = 1.959963984540054
Z80 = 0.8416212335729143
MDE_K = 2.80               # G3 / A0.4: MDE80 = 2.80 x SE
ALPHA_SR = 1e-3            # SpringRank alpha (B0 T1(b), scripts/56)
TESTS = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]

# G2 / A0.4 randomness: SeedSequence(75) for B0 (T1, T2, T3), SeedSequence(751) for A1 (T0, T1', T2', T3', T4-T8);
# each test's child spawns one child per procedure (fixed order) and each procedure one child per replicate.
SS_B0 = dict(zip(["T1", "T2", "T3"], np.random.SeedSequence(75).spawn(3)))
SS_A1 = dict(zip(["T0", "T1'", "T2'", "T3'", "T4", "T5", "T6", "T7", "T8"], np.random.SeedSequence(751).spawn(9)))
_PROCS: dict = {}


def proc_seeds(test_ss: np.random.SeedSequence, names: list[str]) -> dict:
    """One SeedSequence per named procedure of a test (spawned once, in the order given)."""
    key = id(test_ss)
    if key not in _PROCS:
        _PROCS[key] = dict(zip(names, test_ss.spawn(len(names))))
    assert list(_PROCS[key]) == names, (list(_PROCS[key]), names)
    return _PROCS[key]


def gens(ss: np.random.SeedSequence, n: int) -> list:
    return [np.random.default_rng(s) for s in ss.spawn(n)]


T_START = time.time()
RES: dict = {}              # test -> {key: value} (JSON-serialisable), written to the cache / report
CUR = {"test": None}


def stage(msg: str):
    print(f"[{time.time() - T_START:8.1f}s] {msg}", flush=True)


def put(key: str, value, note: str = ""):
    """Record a number (or small table) of the current test and print it."""
    v = _jsonable(value)
    RES.setdefault(CUR["test"], {})[key] = v
    if isinstance(v, float):
        s = f"{v:+.4f}"
    elif isinstance(v, (list, dict)):
        s = json.dumps(v)[:400]
    else:
        s = str(v)
    print(f"  {CUR['test']} | {key} = {s}" + (f"   ({note})" if note else ""), flush=True)


def _jsonable(v):
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float):
        return None if not np.isfinite(v) else float(np.round(v, 10))
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


_MODS: dict = {}


def mod(name: str):
    """Import an existing script by path (never edited): s28, s52, s55, s59, s61, s62, s66."""
    files = {"s28": "28_field_vs_generic_prestige.py", "s55": "55_selectivity.py", "s59": "59_pseo_refresh.py",
             "s61": "61_flows_placement.py", "s62": "62_uk_leo.py", "s66": "66_employer_learning.py"}
    if name not in _MODS:
        if name == "s52":
            _MODS[name] = mod("s59").s52
        elif name == "s66":
            spec = importlib.util.spec_from_file_location("s66", ROOT / "scripts" / files[name])
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            _MODS[name] = m
            _MODS.setdefault("s59", m.s59)
            _MODS.setdefault("s61", m.s61)
            _MODS.setdefault("s62", m.s62)
            _MODS.setdefault("s55", m.s55)
        else:
            if name in ("s59", "s61", "s62", "s55"):
                return mod("s66") and _MODS[name]
            spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / files[name])
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            _MODS[name] = m
    return _MODS[name]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


# =============================================================================================================
# rank statistics (unit weights) and weighted versions under institution multinomial counts W (B, n)
# =============================================================================================================
def rk(x):
    """Average ranks (1-based, ties averaged; identical to scipy.stats.rankdata, faster for short vectors)."""
    a = np.asarray(x, float)
    o = np.argsort(a, kind="mergesort")
    sa = a[o]
    new = np.r_[True, sa[1:] != sa[:-1]]
    idx = np.cumsum(new) - 1
    cnt = np.bincount(idx)
    avg = np.cumsum(cnt) - cnt + (cnt - 1) / 2.0 + 1.0
    r = np.empty(len(a))
    r[o] = avg[idx]
    return r


def pear(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    den = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    return float((xc * yc).sum() / den) if den > 0 else np.nan


def spear(x, y) -> float:
    if len(x) < 3:
        return np.nan
    return pear(rk(x), rk(y))


def partial_spear(x, y, Z: list) -> float:
    """scripts/55 partial_rank without the df rule: ranks of x and y residualised on [1, ranks of Z]."""
    n = len(x)
    X = np.column_stack([np.ones(n)] + [rk(z) for z in Z])
    Yv = np.column_stack([rk(x), rk(y)])
    coef = np.linalg.lstsq(X, Yv, rcond=None)[0]
    E = Yv - X @ coef
    return pear(E[:, 0], E[:, 1])


def rowwise_spear(X, Y):
    """Spearman per row of two (R, k) arrays (ties averaged)."""
    rx = rankdata(X, axis=1)
    ry = rankdata(Y, axis=1)
    xc = rx - rx.mean(1, keepdims=True)
    yc = ry - ry.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (xc * yc).sum(1) / np.sqrt((xc ** 2).sum(1) * (yc ** 2).sum(1))


def wrank(x, W):
    return mod("s59").wrank(np.asarray(x, float), W)


def wcorr(x, y, W):
    return mod("s59").wcorr(x, y, W)


def wstd(r, W, dummy=False):
    return mod("s66").wstd(r, W, dummy)


def wls(D, Y, W):
    return mod("s66").wls(D, Y, W)


def wspear(x, y, W):
    """Spearman of x and y under weights W (B, n) -> (B,)."""
    return wcorr(wrank(x, W), wrank(y, W), W)


def wresid(T, Zcols, W):
    """Residuals of the columns of T (B, n, m) on [1, Zcols...] (each (B, n)) by WLS with weights W."""
    one = np.ones(W.shape)
    D = np.stack([one] + list(Zcols), -1)
    return T - np.einsum("bnk,bkh->bnh", D, wls(D, T, W))


def wpartial(x, y, Z: list, W):
    """Partial Spearman of x, y given Z (continuous, ranked) under weights W -> (B,)."""
    rx, ry = wrank(x, W), wrank(y, W)
    zc = [wrank(z, W) for z in Z]
    res = wresid(np.stack([rx, ry], -1), zc, W)
    return wcorr(res[:, :, 0], res[:, :, 1], W)


# =============================================================================================================
# bootstrap engine: observed pass (unit weights), shared institution draw (one multinomial over the institution
# universe per replicate, applied to every field), independent draw per field (scripts/59 twoway conventions);
# degenerate replicates (a non-finite statistic) are redrawn from the same replicate stream.
# =============================================================================================================
def _mn(g, NI):
    return g.multinomial(NI, np.full(NI, 1.0 / NI)).astype(float)


def boot_engine(fields: list, NI: int, eval_field, ss_shared, ss_indep, nboot: int = B, max_tries: int = 100,
                tag: str = ""):
    """eval_field(f, W (b, NI)) -> {stat: (b,) array} for field f. Returns obs {stat: {f: float}},
    cb {stat: {f: (B,)}}, ib {stat: {f: (B,)}} and redraw counts."""
    obs = {}
    for f in fields:
        o = eval_field(f, np.ones((1, NI)))
        for k, v in o.items():
            obs.setdefault(k, {})[f] = float(v[0])
    # shared draw
    G = gens(ss_shared, nboot)
    W = np.stack([_mn(g, NI) for g in G])
    cb = {}
    rows = np.arange(nboot)
    red_s = 0
    for it in range(max_tries + 1):
        bad = np.zeros(len(rows), bool)
        part = {}
        for f in fields:
            o = eval_field(f, W[rows])
            for k, v in o.items():
                part.setdefault(k, {})[f] = v
                bad |= ~np.isfinite(v)
        for k, d in part.items():
            for f, v in d.items():
                arr = cb.setdefault(k, {}).setdefault(f, np.full(nboot, np.nan))
                arr[rows] = v
        if not bad.any():
            break
        assert it < max_tries, f"{tag}: could not draw non-degenerate shared institution weights"
        rows = rows[bad]
        red_s += len(rows)
        for r in rows:
            W[r] = _mn(G[r], NI)
    del W
    # independent draw per field
    ib, red_i = {}, 0
    ssf = ss_indep.spawn(len(fields))
    for f, s in zip(fields, ssf):
        G = gens(s, nboot)
        W = np.stack([_mn(g, NI) for g in G])
        rows = np.arange(nboot)
        for it in range(max_tries + 1):
            o = eval_field(f, W[rows])
            bad = np.zeros(len(rows), bool)
            for k, v in o.items():
                arr = ib.setdefault(k, {}).setdefault(f, np.full(nboot, np.nan))
                arr[rows] = v
                bad |= ~np.isfinite(v)
            if not bad.any():
                break
            assert it < max_tries, f"{tag}: could not draw non-degenerate weights for field {f}"
            rows = rows[bad]
            red_i += len(rows)
            for r in rows:
                W[r] = _mn(G[r], NI)
        del W
    return obs, cb, ib, dict(redrawn_shared=red_s, redrawn_indep=red_i)


def infer(stat_fn, obs: dict, cb: dict, ib: dict, fields: list, ss_fields, names: list, label: str = "") -> dict:
    """Two-way (field, institution) variance of scripts/59 (twoway()) for a statistic of per-field values.
    stat_fn(d) takes {name: array (K,) or (K, B)} and returns a scalar or (B,). v_field = jackknife over fields
    (s^2/K for a mean). Also: the crossed (fields x shared institution draw) percentile interval, the field
    bootstrap, the leave-one-field-out jackknife normal interval, and MDE80 = 2.80 x SE."""
    s59 = mod("s59")
    K = len(fields)
    po = {n: np.array([obs[n][f] for f in fields]) for n in names}
    pcb = {n: np.stack([cb[n][f] for f in fields]) for n in names}
    pib = {n: np.stack([ib[n][f] for f in fields]) for n in names}
    est = float(stat_fn(po))
    th = np.array([float(stat_fn({n: np.delete(po[n], i) for n in names})) for i in range(K)])
    vj = float((K - 1) / K * ((th - th.mean()) ** 2).sum())
    cbs = np.asarray(stat_fn(pcb), float)
    ibs = np.asarray(stat_fn(pib), float)
    g = np.random.default_rng(ss_fields)
    FI = g.integers(K, size=(B, K))
    fb = np.asarray(stat_fn({n: po[n][FI.T] for n in names}), float)
    xb = np.asarray(stat_fn({n: pcb[n][FI.T, np.arange(B)[None, :]] for n in names}), float)
    tw = s59.twoway(est, vj, cbs, ibs, fb, xb)
    se = tw["tse"]
    sej = float(np.sqrt(vj))
    out = dict(est=est, k=K, se=se, lo=tw["tlo"], hi=tw["thi"], mde=MDE_K * se, mcse=tw["tmcse"],
               fallback=bool(tw["tfallback"]), v_field=tw["tvf"], v_inst=tw["tvs"], v_fxi=tw["tvi"],
               xlo=float(np.percentile(xb, 2.5)), xhi=float(np.percentile(xb, 97.5)), xse=float(np.std(xb, ddof=1)),
               flo=float(np.percentile(fb, 2.5)), fhi=float(np.percentile(fb, 97.5)),
               jlo=est - Z975 * sej, jhi=est + Z975 * sej, jse=sej,
               edge=bool(min(abs(tw["tlo"]), abs(tw["thi"])) <= 2 * tw["tmcse"]),
               fields=list(fields))
    if label:
        print(f"  {CUR['test']} | {label}: MDE80 {out['mde']:.4f}; est {est:+.4f} two-way 95% "
              f"[{out['lo']:+.4f}, {out['hi']:+.4f}] (SE {se:.4f}{', fallback' if out['fallback'] else ''}"
              f"{', edge' if out['edge'] else ''}); crossed pct [{out['xlo']:+.4f}, {out['xhi']:+.4f}]; "
              f"LOFO jackknife [{out['jlo']:+.4f}, {out['jhi']:+.4f}]; k={K}", flush=True)
    return out


def mean_of(name):
    return lambda d: np.mean(d[name], axis=0)


def ratio_of(a, b):
    return lambda d: np.mean(d[a], axis=0) / np.mean(d[b], axis=0)


def diff_of(a, b):
    return lambda d: np.mean(d[a] - d[b], axis=0)


def slope_w(yrs):
    yrs = np.asarray(yrs, float)
    return (yrs - yrs.mean()) / ((yrs - yrs.mean()) ** 2).sum()


W_US = slope_w([1, 5, 10])
W_UK = slope_w([1, 3, 5])


def fmt(v, d=3, sign=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def ci_s(r: dict, d=3, lo="lo", hi="hi"):
    return f"[{fmt(r[lo], d)}, {fmt(r[hi], d)}]"


# =============================================================================================================
# Wapman helpers
# =============================================================================================================
_WAP: dict = {}


def wapman():
    if not _WAP:
        e = pd.read_csv(EDGES)
        r = pd.read_csv(RANKS)
        _WAP["e"], _WAP["r"] = e, r
    return _WAP["e"], _WAP["r"]


def pct_map(level: str, value: str) -> pd.Series:
    """G5: InstitutionId -> pct = 1 - Rank/(N - 1) for one published ranking."""
    _, r = wapman()
    x = r[(r.TaxonomyLevel == level) & (r.TaxonomyValue == value)]
    N = len(x)
    return pd.Series(1.0 - x.Rank.to_numpy(float) / (N - 1), index=x.InstitutionId.to_numpy(float))


def field_net(level: str, value: str) -> dict:
    """Public edges of one network, self-hires dropped. Nodes = union of degree and employer institutions
    (sorted InstitutionId). Rows: src (degree), dst (employer), tot, men, women."""
    e, _ = wapman()
    x = e[(e.TaxonomyLevel == level) & (e.TaxonomyValue == value)]
    x = x[x.DegreeInstitutionId != x.InstitutionId]
    nodes = np.array(sorted(set(x.DegreeInstitutionId) | set(x.InstitutionId)), float)
    ix = {v: i for i, v in enumerate(nodes)}
    src = x.DegreeInstitutionId.map(ix).to_numpy(int)
    dst = x.InstitutionId.map(ix).to_numpy(int)
    n = len(nodes)
    tot = x.Total.to_numpy(float)
    A = np.zeros((n, n))
    np.add.at(A, (src, dst), tot)
    names = {}
    for a, b in zip(x.DegreeInstitutionId, x.DegreeInstitutionName):
        names[a] = b
    for a, b in zip(x.InstitutionId, x.InstitutionName):
        names[a] = b
    return dict(level=level, value=value, nodes=nodes, n=n, src=src, dst=dst, tot=tot,
                men=x.Men.to_numpy(float), women=x.Women.to_numpy(float), A=A,
                names=[names[v] for v in nodes])


def sr_batch(A, alpha: float = ALPHA_SR):
    """SpringRank scores for a stack of dense adjacencies A (m, n, n): the linear system of
    src/ar_pipeline/springrank.springrank, [alpha I + D_out + D_in - (A + A^T)] s = d_out - d_in, centred.
    Solved densely in a batch (identical equations; checked against the sparse implementation)."""
    A = np.asarray(A, float)
    if A.ndim == 2:
        A = A[None]
    dout, din = A.sum(2), A.sum(1)
    n = A.shape[1]
    M = -(A + np.transpose(A, (0, 2, 1)))
    idx = np.arange(n)
    M[:, idx, idx] += alpha + dout + din
    s = np.linalg.solve(M, (dout - din)[..., None])[..., 0]
    return s - s.mean(1, keepdims=True)


def pct_from_scores(s, present):
    """Percentiles 1 - Rank/(N-1) (Rank 0 = highest score, ties averaged) among nodes with present == True;
    NaN elsewhere. s, present (m, n)."""
    out = np.full(s.shape, np.nan)
    for i in range(s.shape[0]):
        p = present[i]
        N = int(p.sum())
        if N < 2:
            continue
        rank0 = rankdata(-s[i, p]) - 1.0
        out[i, p] = 1.0 - rank0 / (N - 1)
    return out


# =============================================================================================================
# T1 (B0) and T1(a'), T1(b') (A1): single-index audit of the hiring measure. No earnings data are read.
# =============================================================================================================
N_SPLIT_B0 = 200
N_SPLIT_A1 = 100
N_SIM = 100
N_PERM_T1 = 200
MIN_ELIG = 4          # persons placed and hired in the full public network
MIN_INST_T1 = 15
MIN_D = 0.20
MIN_DEG_GENDER = 3
MIN_Q_GENDER = 30


def half_counts(src, dst, nA, n):
    """Per split s: placed / hired person counts per node from row counts nA (S, r)."""
    S = nA.shape[0]
    off = (np.arange(S)[:, None] * n)
    pl = np.bincount((off + src[None, :]).ravel(), nA.ravel(), S * n).reshape(S, n)
    hi = np.bincount((off + dst[None, :]).ravel(), nA.ravel(), S * n).reshape(S, n)
    return pl, hi


def role_means(src, dst, nX, pct_for_X, n):
    """P (mean pct of employers of persons placed by u) and H (mean pct of degree institutions of persons hired
    by u) for the persons of one half: nX (S, r) row counts, pct_for_X (S, n) the percentiles used for them.
    Persons whose counterpart has no percentile are dropped."""
    S = nX.shape[0]
    off = (np.arange(S)[:, None] * n)
    pd_ = np.take_along_axis(pct_for_X, np.broadcast_to(dst, (S, len(dst))), 1)      # employer pct
    ps_ = np.take_along_axis(pct_for_X, np.broadcast_to(src, (S, len(src))), 1)      # degree-inst pct
    okd, oks = np.isfinite(pd_), np.isfinite(ps_)
    wP = np.where(okd, nX, 0.0)
    wH = np.where(oks, nX, 0.0)
    Pn = np.bincount((off + src[None, :]).ravel(), (wP * np.where(okd, pd_, 0.0)).ravel(), S * n).reshape(S, n)
    Pd = np.bincount((off + src[None, :]).ravel(), wP.ravel(), S * n).reshape(S, n)
    Hn = np.bincount((off + dst[None, :]).ravel(), (wH * np.where(oks, ps_, 0.0)).ravel(), S * n).reshape(S, n)
    Hd = np.bincount((off + dst[None, :]).ravel(), wH.ravel(), S * n).reshape(S, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(Pd > 0, Pn / Pd, np.nan), np.where(Hd > 0, Hn / Hd, np.nan)


def split_r(PA, PB, HA, HB, elig):
    """r_PP, r_HH, r_PH per split over eligible institutions with P and H defined in both halves."""
    S = PA.shape[0]
    out = np.full((S, 3), np.nan)
    for s in range(S):
        m = elig & np.isfinite(PA[s]) & np.isfinite(PB[s]) & np.isfinite(HA[s]) & np.isfinite(HB[s])
        if m.sum() < 3:
            continue
        pa, pb, ha, hb = rk(PA[s, m]), rk(PB[s, m]), rk(HA[s, m]), rk(HB[s, m])
        out[s] = [pear(pa, pb), pear(ha, hb), (pear(pa, hb) + pear(pb, ha)) / 2]
    return out


def nd_of(r3):
    mPP, mHH, mPH = np.nanmean(r3[:, 0]), np.nanmean(r3[:, 1]), np.nanmean(r3[:, 2])
    return mPH, float(np.sqrt(max(mPP, 0.0) * max(mHH, 0.0))), mPP, mHH


def eligibility(net) -> np.ndarray:
    """B0: >= 4 persons placed and >= 4 hired in the full public network (self-hires dropped)."""
    dout = np.bincount(net["src"], net["tot"], net["n"])
    din = np.bincount(net["dst"], net["tot"], net["n"])
    return (dout >= MIN_ELIG) & (din >= MIN_ELIG)


def t1a_b0(nets: dict, pcts: dict, ss) -> dict:
    """B0 T1(a): 200 Bernoulli(1/2) person splits; one stream per split, fields in sorted order."""
    fl = sorted(nets)
    G = gens(ss, N_SPLIT_B0)
    R3 = {f: np.full((N_SPLIT_B0, 3), np.nan) for f in fl}
    for s, g in enumerate(G):
        for f in fl:
            net = nets[f]
            nA = g.binomial(net["tot"].astype(np.int64), 0.5).astype(float)[None, :]
            nB = net["tot"][None, :] - nA
            pc = pcts[f][None, :]
            PA, HA = role_means(net["src"], net["dst"], nA, pc, net["n"])
            PB, HB = role_means(net["src"], net["dst"], nB, pc, net["n"])
            R3[f][s] = split_r(PA, PB, HA, HB, net["elig"])[0]
    return R3


def springrank_check(nets: dict) -> float:
    """max |sr_batch - src springrank| over the observed networks (validation of the batched solver)."""
    mx = 0.0
    for f, net in nets.items():
        a = sr_batch(net["A"])[0]
        b = springrank_src(net["A"], alpha=ALPHA_SR)
        mx = max(mx, float(np.abs(a - b).max()))
    return mx


def crossfit_R(src, dst, tot_rows, n, elig, gens_split, chunk: int = 50):
    """A1 T1(a'): per split, SpringRank on each half; P, H of half A use percentiles fitted on half B and vice
    versa (percentiles among nodes with a person in that half). Returns r3 (S, 3)."""
    S = len(gens_split)
    nA = np.stack([g.binomial(tot_rows.astype(np.int64), 0.5) for g in gens_split]).astype(float)
    nB = tot_rows[None, :] - nA
    out = np.full((S, 3), np.nan)
    for c0 in range(0, S, chunk):
        sl = slice(c0, min(S, c0 + chunk))
        m = sl.stop - sl.start
        pcs = []
        for nX in (nA[sl], nB[sl]):
            flat = (np.arange(m)[:, None] * (n * n) + (src * n + dst)[None, :]).ravel()
            Ah = np.bincount(flat, nX.ravel(), m * n * n).reshape(m, n, n)
            s = sr_batch(Ah)
            present = (Ah.sum(2) + Ah.sum(1)) > 0
            pcs.append(pct_from_scores(s, present))
            del Ah
        pctA, pctB = pcs
        PA, HA = role_means(src, dst, nA[sl], pctB, n)
        PB, HB = role_means(src, dst, nB[sl], pctA, n)
        out[sl] = split_r(PA, PB, HA, HB, elig)
    return out


def fit_beta(A, s) -> float:
    """beta maximising sum_{u,v} A_uv log logistic(2 beta (s_u - s_v)) (SpringRank's direction likelihood)."""
    u, v = np.nonzero(A)
    w, d = A[u, v], s[u] - s[v]

    def nll(b):
        return float((w * np.logaddexp(0.0, -2.0 * b * d)).sum())
    r = minimize_scalar(nll, bounds=(1e-6, 100.0), method="bounded", options=dict(xatol=1e-8))
    return float(r.x)


def sim_field(args):
    """Worker: R' pipeline on N_SIM networks simulated from the fitted SpringRank model of one field."""
    f, A, ss_field, calib = args
    n = A.shape[0]
    s_hat = sr_batch(A)[0]
    beta = fit_beta(A, s_hat)
    D = s_hat[:, None] - s_hat[None, :]
    E = np.exp(-beta / 2.0 * (D - 1.0) ** 2)
    np.fill_diagonal(E, 0.0)
    c = A.sum() / E.sum()                       # as registered: E[sum A*] = the field's public total
    if calib == "post":
        # deviation (feasibility): E[sum A* 1{A* >= 2}] = the public total, i.e. matched after the censoring;
        # for Poisson(l), E[X 1{X >= 2}] = l (1 - exp(-l))
        from scipy.optimize import brentq
        tot = A.sum()
        c = brentq(lambda cc: float((cc * E * (1.0 - np.exp(-cc * E))).sum()) - tot, c, c * 1e6, xtol=1e-12)
    rows = []
    kept = []
    for s_ss in ss_field.spawn(N_SIM):
        k_net, k_split = s_ss.spawn(2)
        g = np.random.default_rng(k_net)
        As = g.poisson(c * E).astype(float)
        kept.append(float(As[As >= 2].sum() / max(As.sum(), 1.0)))
        As[As < 2] = 0.0
        src, dst = np.nonzero(As)
        tot = As[src, dst]
        dout = np.bincount(src, tot, n)
        din = np.bincount(dst, tot, n)
        elig = (dout >= MIN_ELIG) & (din >= MIN_ELIG)
        if elig.sum() < MIN_INST_T1:
            rows.append((np.nan, np.nan, np.nan, np.nan, int(elig.sum())))
            continue
        r3 = crossfit_R(src, dst, tot, n, elig, gens(k_split, N_SPLIT_A1))
        Nf, Df, mPP, mHH = nd_of(r3)
        rows.append((Nf, Df, mPP, mHH, int(elig.sum())))
    return f, beta, c, np.array(rows, float), float(np.mean(kept))


def gender_c(net, men_rows, women_rows, fixed_q=None):
    """c = Spearman(SpringRank on men's network, SpringRank on women's network) over institutions with >= 3
    persons (in + out) in each network; also the number qualifying. men_rows, women_rows (m, r). fixed_q
    (exploratory): also return c over a fixed institution set."""
    m = men_rows.shape[0]
    n, src, dst = net["n"], net["src"], net["dst"]
    out = np.full(m, np.nan)
    outf = np.full(m, np.nan)
    nq = np.zeros(m, int)
    for c0 in range(0, m, 25):
        sl = slice(c0, min(m, c0 + 25))
        k = sl.stop - sl.start
        rows = np.concatenate([men_rows[sl], women_rows[sl]])
        flat = (np.arange(2 * k)[:, None] * (n * n) + (src * n + dst)[None, :]).ravel()
        Ah = np.bincount(flat, rows.ravel(), 2 * k * n * n).reshape(2 * k, n, n)
        deg = Ah.sum(2) + Ah.sum(1)
        s = sr_batch(Ah)
        del Ah
        for j in range(k):
            q = (deg[j] >= MIN_DEG_GENDER) & (deg[k + j] >= MIN_DEG_GENDER)
            nq[c0 + j] = int(q.sum())
            if q.sum() >= 3:
                out[c0 + j] = pear(rk(s[j, q]), rk(s[k + j, q]))
            if fixed_q is not None:
                outf[c0 + j] = pear(rk(s[j, fixed_q]), rk(s[k + j, fixed_q]))
    if fixed_q is not None:
        return out, nq, outf
    return out, nq


def t1b(nets: dict, ss) -> dict:
    """B0 T1(b): men's vs women's SpringRank; 200 permutations of gender labels within the field."""
    fl = sorted(nets)
    ssf = dict(zip(fl, ss.spawn(len(fl))))
    res = {}
    for f in fl:
        net = nets[f]
        men, women = net["men"], net["women"]
        c, nq = gender_c(net, men[None, :], women[None, :])
        rec = dict(c=float(c[0]), nq=int(nq[0]), enters=bool(nq[0] >= MIN_Q_GENDER))
        n_ = net["n"]
        Am = np.bincount(net["src"] * n_ + net["dst"], men, n_ * n_).reshape(n_, n_)
        Aw = np.bincount(net["src"] * n_ + net["dst"], women, n_ * n_).reshape(n_, n_)
        q_obs = ((Am.sum(0) + Am.sum(1)) >= MIN_DEG_GENDER) & ((Aw.sum(0) + Aw.sum(1)) >= MIN_DEG_GENDER)
        if rec["enters"]:
            size = (men + women).astype(np.int64)
            M = int(men.sum())
            G = gens(ssf[f], N_PERM_T1)
            pm = np.stack([g.multivariate_hypergeometric(size, M, method="marginals") for g in G]).astype(float)
            pw = size[None, :] - pm
            nul, nqn, nulf = gender_c(net, pm, pw, fixed_q=q_obs)
            nqn = nqn[np.isfinite(nul)]
            nul = nul[np.isfinite(nul)]
            nulf = nulf[np.isfinite(nulf)]
            rec.update(null_mean=float(nul.mean()), null_sd=float(nul.std(ddof=1)), n_null=int(len(nul)),
                       z=float((rec["c"] - nul.mean()) / nul.std(ddof=1)),
                       p=float((1 + np.sum(nul <= rec["c"])) / (1 + len(nul))),
                       null_nq_mean=float(nqn.mean()),
                       z_fixedset=float((rec["c"] - nulf.mean()) / nulf.std(ddof=1)),
                       null_mean_fixedset=float(nulf.mean()))
        res[f] = rec
    return res


def t1b_verdicts(res: dict) -> dict:
    ent = {f: r for f, r in res.items() if r["enters"]}
    K = len(ent)
    if K == 0:
        return dict(K=0, verdict_b="infeasible", verdict_bp="infeasible")
    nsig = int(sum(r["p"] < 0.05 for r in ent.values()))
    q95 = int(binom.ppf(0.95, K, 0.05))
    zbar = float(np.mean([r["z"] for r in ent.values()]))
    thr = -1.645 / np.sqrt(K)
    ci_, cii = nsig <= q95, zbar >= thr
    vb = "supported" if (ci_ and cii) else ("contradicted" if (not ci_ and not cii) else "inconclusive")
    d = np.array([r["null_mean"] - r["c"] for r in ent.values()])
    se = float(np.sqrt(np.sum([r["null_sd"] ** 2 for r in ent.values()])) / K)
    dbar = float(d.mean())
    if dbar + 1.645 * se <= 0.10:
        vbp = "invariant"
    elif dbar - 1.96 * se > 0:
        vbp = "not invariant"
    else:
        vbp = "uninformative"
    return dict(K=K, nsig=nsig, q95=q95, zbar=zbar, zthr=float(thr), cond_i=bool(ci_), cond_ii=bool(cii),
                verdict_b=vb, dbar=dbar, se_bp=se, mde_bp=MDE_K * se, upper_bp=dbar + 1.645 * se,
                lower_bp=dbar - 1.96 * se, verdict_bp=vbp)


def run_T1(workers: int = 1):
    CUR["test"] = "T1"
    stage("T1: Wapman field networks")
    e, r = wapman()
    fl = sorted(e[e.TaxonomyLevel == "Field"].TaxonomyValue.unique())
    put("n_fields_total", len(fl), "TaxonomyLevel == Field (A2.1)")
    nets, pcts = {}, {}
    for f in fl:
        net = field_net("Field", f)
        net["elig"] = eligibility(net)
        pm = pct_map("Field", f)
        nets[f] = net
        pcts[f] = net["nodes"].astype(float) * np.nan
        idx = pd.Index(net["nodes"])
        hit = idx.isin(pm.index)
        pcts[f][hit] = pm.reindex(net["nodes"][hit]).to_numpy(float)
    acad = field_net("Academia", "Academia")
    acad["elig"] = eligibility(acad)
    pa = pct_map("Academia", "Academia")
    acad_p = np.full(acad["n"], np.nan)
    hit = pd.Index(acad["nodes"]).isin(pa.index)
    acad_p[hit] = pa.reindex(acad["nodes"][hit]).to_numpy(float)
    put("springrank_batch_vs_src_maxabs", springrank_check({**nets, "Academia": acad}),
        "dense batched solver vs src/ar_pipeline/springrank (same equations)")
    persons = sum(n["tot"].sum() for n in nets.values())
    nog = sum((n["tot"] - n["men"] - n["women"]).sum() for n in nets.values())
    put("persons_field_level_no_selfhire", persons)
    put("share_no_recorded_gender", nog / persons, "A0.5 erratum check: 5.3% expected")
    both = np.concatenate([(n["men"] >= 1) & (n["women"] >= 1) for n in nets.values()])
    put("share_rows_with_a_man_and_a_woman", float(both.mean()),
        "data property: every public edge row has >= 1 man and >= 1 woman, so the men's and women's networks share "
        "their edge set")

    pb0 = proc_seeds(SS_B0["T1"], ["a_splits", "a_fieldboot", "b_perm", "acad_a_splits", "acad_b_perm"])
    pa1 = proc_seeds(SS_A1["T1'"], ["ap_splits", "ap_sims", "ap_fieldboot", "ap_sims_post", "ap_fieldboot_post"])

    # ---------------- B0 T1(a) ----------------
    stage("T1(a): 200 splits")
    R3 = t1a_b0(nets, pcts, pb0["a_splits"])
    rows, keep = [], []
    for f in fl:
        Nf, Df, mPP, mHH = nd_of(R3[f])
        ne = int(nets[f]["elig"].sum())
        ok = ne >= MIN_INST_T1 and Df >= MIN_D
        rows.append(dict(field=f, n_elig=ne, N=Nf, D=Df, rPP=mPP, rHH=mHH, enters=bool(ok),
                         nan_splits=int(np.isnan(R3[f][:, 2]).sum())))
        if ok:
            keep.append(f)
    tab = pd.DataFrame(rows)
    put("a_fields_excluded", int(len(fl) - len(keep)), f"of {len(fl)}; <{MIN_INST_T1} eligible or D < {MIN_D}")
    put("a_fields_excluded_by_n", int((tab.n_elig < MIN_INST_T1).sum()))
    kt = tab[tab.enters]
    Nv, Dv = kt.N.to_numpy(), kt.D.to_numpy()
    R = float(Nv.sum() / Dv.sum())
    g = np.random.default_rng(pb0["a_fieldboot"])
    FI = g.integers(len(kt), size=(B, len(kt)))
    Rb = Nv[FI].sum(1) / Dv[FI].sum(1)
    se = float(Rb.std(ddof=1))
    put("a_MDE80", MDE_K * se, "2.80 x field-bootstrap SE (reported before the estimate)")
    put("a_R", R)
    lo, hi = float(np.percentile(Rb, 2.5)), float(np.percentile(Rb, 97.5))
    put("a_R_ci", [lo, hi], "field bootstrap, B=1000, percentile")
    put("a_k", int(len(kt)))
    put("a_mean_rPP", float(kt.rPP.mean()))
    put("a_mean_rHH", float(kt.rHH.mean()))
    put("a_mean_N", float(Nv.mean()))
    va = "supported" if lo >= 0.80 else ("contradicted" if hi < 0.80 else "inconclusive")
    put("a_verdict", va)
    put("a_table", tab.round(4).to_dict("records"))

    # ---------------- B0 T1(b) ----------------
    stage("T1(b): gender SpringRank + 200 permutations per field")
    gb = t1b(nets, pb0["b_perm"])
    vb = t1b_verdicts(gb)
    put("b_MDE80_equivalence", vb.get("mde_bp", np.nan), "T1(b') 2.80 x SE (reported before the estimate)")
    for k in ["K", "nsig", "q95", "zbar", "zthr", "cond_i", "cond_ii", "verdict_b", "dbar", "se_bp",
              "upper_bp", "lower_bp", "verdict_bp"]:
        put(f"b_{k}", vb.get(k))
    put("b_table", [dict(field=f, **v) for f, v in gb.items()])
    put("b_mean_c", float(np.mean([v["c"] for v in gb.values() if v["enters"]])) if vb["K"] else np.nan)
    ent_ = [v for v in gb.values() if v["enters"]]
    if ent_:
        put("b_exploratory_mean_nq_observed_vs_null", [float(np.mean([v["nq"] for v in ent_])),
                                                       float(np.mean([v["null_nq_mean"] for v in ent_]))],
            "institutions qualifying (>= 3 persons in each network): observed vs relabelled splits")
        put("b_exploratory_mean_z_fixed_set", float(np.mean([v["z_fixedset"] for v in ent_])),
            "exploratory: null computed on the observed qualifying institutions")
        put("b_exploratory_mean_null_c_fixed_set", float(np.mean([v["null_mean_fixedset"] for v in ent_])))
    va_b = vb["verdict_b"]
    overall = "passes" if (va == "supported" and va_b == "supported") else (
        "fails" if ("contradicted" in (va, va_b)) else "undecided")
    put("B0_T1_verdict", overall, "single-valuation reading of F (B0 T1 verdict)")

    # ---------------- Academia network (reported, not in the verdict) ----------------
    stage("T1: Academia network, same analyses (reported only)")
    R3a = t1a_b0({"Academia": acad}, {"Academia": acad_p}, pb0["acad_a_splits"])["Academia"]
    Na, Da, pPa, pHa = nd_of(R3a)
    put("acad_a_N", Na)
    put("acad_a_D", Da)
    put("acad_a_R", Na / Da if Da > 0 else np.nan, "Academia network; single network, no interval")
    put("acad_a_n_elig", int(acad["elig"].sum()))
    gba = t1b({"Academia": acad}, pb0["acad_b_perm"])["Academia"]
    put("acad_b", gba)

    # ---------------- A1 T1(a'): cross-fitted R' and the SpringRank-model benchmark R0 ----------------
    stage("T1(a'): cross-fitted R' on the real networks (100 splits)")
    ssp = dict(zip(fl, pa1["ap_splits"].spawn(len(fl))))
    rowsp = []
    for f in fl:
        net = nets[f]
        r3 = crossfit_R(net["src"], net["dst"], net["tot"], net["n"], net["elig"], gens(ssp[f], N_SPLIT_A1))
        Nf, Df, mPP, mHH = nd_of(r3)
        ne = int(net["elig"].sum())
        rowsp.append(dict(field=f, n_elig=ne, N=Nf, D=Df, rPP=mPP, rHH=mHH,
                          enters=bool(ne >= MIN_INST_T1 and Df >= MIN_D)))
    tp = pd.DataFrame(rowsp)
    keep_p = list(tp[tp.enters].field)
    put("ap_k", len(keep_p))
    put("ap_real_mean_eligible", float(tp[tp.enters].n_elig.mean()),
        "mean eligible institutions per real field network entering R'")
    Rp = float(tp[tp.enters].N.sum() / tp[tp.enters].D.sum())
    stage(f"T1(a'): R' = {Rp:+.4f}; simulating {N_SIM} networks per field from the fitted SpringRank model")
    Nr = tp.set_index("field").loc[keep_p].N.to_numpy()
    Dr = tp.set_index("field").loc[keep_p].D.to_numpy()
    SIMS = {}
    for calib, pre, sk, fk in [("pre", "ap", "ap_sims", "ap_fieldboot"),
                               ("post", "ap_post", "ap_sims_post", "ap_fieldboot_post")]:
        ssim = dict(zip(fl, pa1[sk].spawn(len(fl))))
        jobs = [(f, nets[f]["A"], ssim[f], calib) for f in keep_p]
        sims = {}
        if workers > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(workers) as pool:
                for f, beta, c, arr, kept in pool.imap(sim_field, jobs):
                    sims[f] = (beta, c, arr, kept)
        else:
            for j in jobs:
                f, beta, c, arr, kept = sim_field(j)
                sims[f] = (beta, c, arr, kept)
        SIMS[calib] = sims
        SN = np.stack([sims[f][2][:, 0] for f in keep_p])     # (K, N_SIM)
        SD = np.stack([sims[f][2][:, 1] for f in keep_p])
        OK = np.isfinite(SN) & np.isfinite(SD) & (SD >= MIN_D)
        put(f"{pre}_sim_field_excluded_share", float(1 - OK.mean()),
            "field x simulated network cells failing B0's eligibility or D rules")
        put(f"{pre}_sim_kept_person_share", float(np.mean([sims[f][3] for f in keep_p])),
            "share of simulated persons in pairs with A* >= 2")
        put(f"{pre}_sim_mean_eligible", float(np.nanmean(np.stack([sims[f][2][:, 4] for f in keep_p]))),
            "mean eligible institutions per simulated field network")
        SNz, SDz = np.where(OK, SN, 0.0), np.where(OK, SD, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            R0s = SNz.sum(0) / SDz.sum(0)
        n_undef = int((SDz.sum(0) <= 0).sum())
        put(f"{pre}_sim_networks_without_any_field", n_undef, f"of {N_SIM} simulated replicate sets")
        # revision 3: R0 is computed as A4.1 states it whatever the share of failing cells (the first run declared
        # the literal-order benchmark infeasible by a rule set after the run; A4.1 has no such category)
        put(f"{pre}_sim_cells_used", int(OK.sum()), f"of {OK.size} field x simulated-network cells enter R0")
        put(f"{pre}_sim_fields_with_any_cell", int(OK.any(1).sum()), f"of {len(keep_p)} fields")
        if OK.any(1).sum() < len(keep_p):
            put(f"{pre}_sim_fields_with_any_cell_names", [f for f, k in zip(keep_p, OK.any(1)) if k])
        g = np.random.default_rng(pa1[fk])
        FI = g.integers(len(keep_p), size=(B, len(keep_p)))
        R0 = float(np.nanmean(R0s))
        Rpb = Nr[FI].sum(1) / Dr[FI].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            R0b = np.array([np.nanmean(SNz[FI[b]].sum(0) / SDz[FI[b]].sum(0)) for b in range(B)])
        dR = Rpb - R0b
        dR = dR[np.isfinite(dR)]
        put(f"{pre}_MDE80", MDE_K * float(dR.std(ddof=1)), "2.80 x field-bootstrap SE of Delta_R (before the estimate)")
        put(f"{pre}_Rprime_ci", [float(np.percentile(Rpb, 2.5)), float(np.percentile(Rpb, 97.5))])
        put(f"{pre}_R0", R0)
        put(f"{pre}_R0_ci", [float(np.nanpercentile(R0b, 2.5)), float(np.nanpercentile(R0b, 97.5))])
        put(f"{pre}_DeltaR", Rp - R0)
        lo, hi = float(np.percentile(dR, 2.5)), float(np.percentile(dR, 97.5))
        put(f"{pre}_DeltaR_ci", [lo, hi], f"fields resampled jointly for real and simulated N, D; {len(dR)} of {B}")
        put(f"{pre}_DeltaR_finite_draws", int(len(dR)), f"of {B} field-bootstrap draws with R0 defined")
        vap = "one index" if lo >= -0.10 else ("not one index" if hi < -0.10 else "inconclusive")
        put(f"{pre}_rule", vap, "A1 T1(a') rule applied to the " + ("literal-order calibration (c matched to the "
            "public total before the A* >= 2 censoring; degenerate)" if calib == "pre" else
            "post-censoring calibration (c matched to the public total after the censoring)"))
        put(f"{pre}_beta", {f: sims[f][0] for f in keep_p})
    put("ap_Rprime", Rp)
    # A4.1 ("c matching the field's total count") does not say whether the simulated total is matched before or
    # after the A* >= 2 censoring. The public total is a censored total; only the post-censoring reading gives a
    # benchmark comparable to the public network (the literal-order one keeps about a tenth of its persons). This
    # reading was adopted after the literal-order benchmark had been seen to degenerate and before any post-censoring
    # statistic was computed. Claim C2 follows it; both readings are reported.
    put("ap_verdict", RES["T1"]["ap_post_rule"], "A1 T1(a'), post-censoring reading of A4.1 (governs C2; literal-order "
        f"reading: {RES['T1']['ap_rule']})")
    sims = SIMS["post"]
    tp["R0_f"] = [float(np.nanmean(sims[f][2][:, 0]) / np.nanmean(sims[f][2][:, 1])) if f in sims else np.nan
                  for f in tp.field]
    tp["Rp_f"] = tp.N / tp.D
    put("ap_table", tp.round(4).to_dict("records"))
    return RES["T1"]


# =============================================================================================================
# T0 (A1 [V]): production audit of F
# =============================================================================================================
N_RESTART = 50
RIDGE_BT = 1e-3


def scorecard_names() -> pd.DataFrame:
    d = pd.read_csv(SC_INST, usecols=["UNITID", "OPEID", "INSTNM", "MAIN", "UGDS", "CONTROL", "AVGFACSAL",
                                      "INEXPFTE", "ENDOWBEGIN", "STABBR"], dtype=str)
    for c in ["MAIN", "UGDS", "CONTROL", "AVGFACSAL", "INEXPFTE", "ENDOWBEGIN"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["inst_key"] = d.INSTNM.map(norm)
    return d


def ipeds_doctorates(fields66) -> pd.DataFrame:
    """IPEDS C2023_a, AWLEVEL 17 (research/scholarship doctorates), CTOTALT, all MAJORNUM, summed over each
    field's cip4 codes; UNITID -> Scorecard INSTNM -> normalised name (scripts/28's join rule); UNITIDs sharing a
    normalised name are summed. Returns inst_key x field key -> doctorates."""
    d = pd.read_csv(IPEDS, usecols=["UNITID", "CIPCODE", "MAJORNUM", "AWLEVEL", "CTOTALT"], dtype=str,
                    encoding="utf-8-sig")
    d = d[d.AWLEVEL == "17"].copy()
    d["cip4"] = d.CIPCODE.str.replace(".", "", regex=False).str[:4]
    d["n"] = pd.to_numeric(d.CTOTALT, errors="coerce").fillna(0.0)
    sc = scorecard_names()[["UNITID", "inst_key"]].drop_duplicates("UNITID")
    d = d.merge(sc, on="UNITID", how="left")
    rows = []
    for f in fields66:
        x = d[d.cip4.isin(f["cip4"])]
        s = x.groupby("inst_key").n.sum()
        rows.append(pd.DataFrame({"inst_key": s.index, "field": f["key"], "doct": s.to_numpy(float)}))
    out = pd.concat(rows, ignore_index=True)
    ipeds_doctorates.matched_unitids = int(d.inst_key.notna().sum())
    ipeds_doctorates.all_unitids = int(len(d))
    return out, set(sc.inst_key.dropna())


BT_GTOL = 1e-9             # stopping rule: max |gradient| of the penalised log-likelihood ...
BT_DEC = 1e-20             # ... or a Newton decrement at machine precision
BT_GASSERT = 1e-6          # convergence check asserted on every fit (revision 1)


def bt_offset(A, ridge: float = RIDGE_BT, max_iter: int = 500, smooth: float = 1.0):
    """Degree-corrected Bradley-Terry fit of the direction of exchange:
    A_uv | T_uv ~ Bin(T_uv, logistic(theta_u - theta_v + o_uv)), o_uv = log(dout_u din_v / (dout_v din_u)),
    ridge penalty (ridge/2)||theta||^2. The penalised negative log-likelihood is strictly convex; it is minimised by
    damped Newton (backtracking line search, Armijo constant 1e-4; single-threaded LAPACK, so the fit does not depend
    on the BLAS thread count) until max |gradient| < BT_GTOL or the Newton decrement is below BT_DEC, and every fit
    asserts max |gradient| < BT_GASSERT (revision 1: the undamped Newton of the first run did not converge in the
    exact limit, nor for one field and the Academia network with the +1 convention). The registered offset is undefined when a degree is zero (common in the public network: nodes that
    only hire or only place). smooth = 1 (primary) adds 1 to every degree, the convention A3 itself uses for the
    export ratio x; smooth = 0 is the exact limit, in which pairs with an infinite offset are fixed at probability 0
    or 1 and are left out of the likelihood (a node with no pair left gets theta = 0 from the ridge, the middle).
    Returns theta, the number of pairs in the likelihood, the nodes with no pair in it and the final max |gradient|."""
    n = A.shape[0]
    dout, din = A.sum(1) + smooth, A.sum(0) + smooth
    iu, iv = np.triu_indices(n, 1)
    T = A[iu, iv] + A[iv, iu]
    m = T > 0
    iu, iv, T, y = iu[m], iv[m], T[m], A[iu[m], iv[m]]
    with np.errstate(divide="ignore"):
        o = np.log(dout[iu]) + np.log(din[iv]) - np.log(dout[iv]) - np.log(din[iu])
    fin = np.isfinite(o)
    iu, iv, T, y, o = iu[fin], iv[fin], T[fin], y[fin], o[fin]
    dg = np.arange(n) * (n + 1)

    def nll(t):
        eta = t[iu] - t[iv] + o
        return float((T * np.logaddexp(0.0, eta) - y * eta).sum() + 0.5 * ridge * float(t @ t))

    def grad(t):
        p = expit(t[iu] - t[iv] + o)
        r = y - T * p
        return np.bincount(iu, r, n) - np.bincount(iv, r, n) - ridge * t, p

    th = np.zeros(n)
    fv = nll(th)
    g, p = grad(th)
    with threadpool_limits(limits=1, user_api="blas"):
        for _ in range(max_iter):
            if np.abs(g).max() < BT_GTOL:
                break
            w = T * p * (1.0 - p)
            off = np.bincount(iu * n + iv, w, n * n).reshape(n, n)
            H = -(off + off.T)
            H.flat[dg] += np.bincount(iu, w, n) + np.bincount(iv, w, n) + ridge
            step = np.linalg.solve(H, g)
            dec = float(g @ step)
            if dec < BT_DEC:
                break
            t = 1.0
            while True:
                thn = th + t * step
                fn = nll(thn)
                if fn <= fv - 1e-4 * t * dec or t < 1e-12:
                    break
                t *= 0.5
            if fn >= fv:
                break
            th, fv = thn, fn
            g, p = grad(th)
    gmax = float(np.abs(g).max())
    assert gmax < BT_GASSERT, f"degree-corrected BT did not converge: max |gradient| {gmax:.3g}"
    informed = np.zeros(n, bool)
    informed[iu] = True
    informed[iv] = True
    return th, int(len(iu)), ~informed, gmax


def bt_offset_first_run(A, ridge: float = RIDGE_BT, max_iter: int = 200, smooth: float = 1.0):
    """Revision audit only: the undamped Newton of the first run (full steps, stopped when max |step| < 1e-10 or
    after 200 iterations), kept to report how far from the optimum it ended. Returns (final max |gradient|,
    max |theta|). Nothing else uses it."""
    n = A.shape[0]
    dout, din = A.sum(1) + smooth, A.sum(0) + smooth
    iu, iv = np.triu_indices(n, 1)
    T = A[iu, iv] + A[iv, iu]
    m = T > 0
    iu, iv, T, y = iu[m], iv[m], T[m], A[iu[m], iv[m]]
    with np.errstate(divide="ignore"):
        o = np.log(dout[iu]) + np.log(din[iv]) - np.log(dout[iv]) - np.log(din[iu])
    fin = np.isfinite(o)
    iu, iv, T, y, o = iu[fin], iv[fin], T[fin], y[fin], o[fin]
    th = np.zeros(n)
    with threadpool_limits(limits=1, user_api="blas"), np.errstate(over="ignore"):
        for _ in range(max_iter):
            p = expit(th[iu] - th[iv] + o)
            r = y - T * p
            g = np.bincount(iu, r, n) - np.bincount(iv, r, n) - ridge * th
            w = T * p * (1 - p)
            H = np.zeros((n, n))
            np.add.at(H, (iu, iu), w)
            np.add.at(H, (iv, iv), w)
            np.add.at(H, (iu, iv), -w)
            np.add.at(H, (iv, iu), -w)
            H[np.arange(n), np.arange(n)] += ridge
            step = np.linalg.solve(H, g)
            th = th + step
            if np.abs(step).max() < 1e-10:
                break
        p = expit(th[iu] - th[iv] + o)
        r = y - T * p
        g = np.bincount(iu, r, n) - np.bincount(iv, r, n) - ridge * th
    return float(np.abs(g).max()), float(np.abs(th).max())


def upward(A, order) -> float:
    """Persons hired upward (employer placed above the degree institution) under an ordering (top first)."""
    M = A[np.ix_(order, order)]
    return float(np.tril(M, -1).sum())


def mvr_search(A, start, g) -> tuple:
    """Local search for a minimum-violation ordering: adjacent swaps, then reinsertion of every node (random
    order) at the position that lowers the count of upward hires most; repeated until no move lowers it."""
    Nt = A - A.T
    order = list(start)
    n = len(order)
    while True:
        moved = False
        for k in range(n - 1):
            x, y = order[k], order[k + 1]
            if Nt[x, y] < 0:
                order[k], order[k + 1] = y, x
                moved = True
        for x in g.permutation(n):
            i = order.index(x)
            r = Nt[x, order]
            best, bj = 0.0, i
            if i + 1 < n:
                dn = np.cumsum(r[i + 1:])
                j = int(np.argmin(dn))
                if dn[j] < best - 1e-12:
                    best, bj = float(dn[j]), i + 1 + j
            if i > 0:
                up = np.cumsum(-r[:i][::-1])
                j = int(np.argmin(up))
                if up[j] < best - 1e-12:
                    best, bj = float(up[j]), i - 1 - j
            if bj != i:
                order.pop(i)
                order.insert(bj, x)
                moved = True
        if not moved:
            break
    return np.array(order), upward(A, np.array(order))


def mvr_field(args):
    f, A, ss = args
    s = sr_batch(A)[0]
    start = np.argsort(-s, kind="mergesort")
    gs = gens(ss, N_RESTART + 1)
    best_o, best_c = mvr_search(A, start, gs[0])
    for k in range(1, N_RESTART + 1):
        o, c = mvr_search(A, gs[k].permutation(A.shape[0]), gs[k])
        if c < best_c:
            best_o, best_c = o, c
    return f, best_o, best_c, upward(A, start)


def run_T0(workers: int = 1):
    CUR["test"] = "T0"
    s28 = mod("s28")
    stage("T0: data (Wapman, IPEDS AWLEVEL 17, institution-stats, scripts/28 cells)")
    e, r = wapman()
    F66 = s28.FIELDS66
    wf2key = {f["wapman_field"]: f["key"] for f in F66}
    fl_all = sorted(e[e.TaxonomyLevel == "Field"].TaxonomyValue.unique())
    doct, sc_keys = ipeds_doctorates(F66)
    put("ipeds_awlevel17_rows_matched_to_scorecard_name", ipeds_doctorates.matched_unitids,
        f"of {ipeds_doctorates.all_unitids} rows")
    ist = pd.read_csv(ISTATS)
    ist = ist[ist.TaxonomyLevel == "Field"][["TaxonomyValue", "OrdinalPrestigeRank", "ProductionRank"]]
    # the last six procedures were appended in revision 1 (spawned after the first five, whose streams are unchanged)
    pa0 = proc_seeds(SS_A1["T0"], ["a_fieldboot", "b_shared", "b_indep", "b_fields", "c_restarts",
                                   "s0_shared", "s0_indep", "s0_fields", "s1_shared", "s1_indep", "s1_fields"])

    # per-field institution table (ranked institutions)
    INST = {}
    for f in fl_all:
        net = field_net("Field", f)
        x = r[(r.TaxonomyLevel == "Field") & (r.TaxonomyValue == f)].copy()
        N = len(x)
        x["F"] = 1.0 - x.Rank / (N - 1)
        x["inst_key"] = x.InstitutionName.map(norm)
        pos = pd.Series(np.arange(net["n"]), index=net["nodes"])
        dout = np.bincount(net["src"], net["tot"], net["n"])
        din = np.bincount(net["dst"], net["tot"], net["n"])
        ni = x.InstitutionId.map(pos)
        x["in_net"] = ni.notna()
        x["dout"] = np.where(ni.notna(), dout[ni.fillna(0).astype(int)], 0.0)
        x["din"] = np.where(ni.notna(), din[ni.fillna(0).astype(int)], 0.0)
        x["x"] = np.log((x.dout + 1) / (x.din + 1))
        key = wf2key.get(f)
        if key is not None:
            dd = doct[doct.field == key].set_index("inst_key").doct
            x["doct"] = x.inst_key.map(dd)
            x.loc[x.doct.isna() & x.inst_key.isin(sc_keys), "doct"] = 0.0     # matched, no doctorates in f
            x["p"] = np.log1p(x.doct)
            x["q"] = x.dout / np.maximum(x.doct, 1.0)
        else:
            x["doct"] = np.nan
            x["p"] = np.nan
            x["q"] = np.nan
        ps = ist[ist.TaxonomyValue == f].set_index("OrdinalPrestigeRank").ProductionRank
        x["negPR"] = -x.Rank.map(ps)
        INST[f] = (x, net)

    # ---------------- (a) no earnings ----------------
    stage("T0(a): Spearman of F with log production, export ratio, placement rate, -ProductionRank")
    arows = []
    for f in fl_all:
        x, net = INST[f]
        rec = dict(field=f, key=wf2key.get(f), n_ranked=len(x), share_in_net=float(x.in_net.mean()))
        for v in ["p", "x", "q", "negPR"]:
            y = x.dropna(subset=[v])
            rec[f"n_{v}"] = len(y)
            rec[f"r_{v}"] = spear(y.F, y[v]) if len(y) >= MIN_INST_T1 else np.nan
        arows.append(rec)
    at = pd.DataFrame(arows)
    g = np.random.default_rng(pa0["a_fieldboot"])
    for v in ["p", "x", "q", "negPR"]:
        vals = at[f"r_{v}"].dropna().to_numpy()
        FI = g.integers(len(vals), size=(B, len(vals)))
        mb = vals[FI].mean(1)
        put(f"a_mean_rho_F_{v}", float(vals.mean()), f"{len(vals)} fields with >= 15 institutions")
        put(f"a_mean_rho_F_{v}_ci", [float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))])
    put("a_table", at.round(4).to_dict("records"))

    # ---------------- (b) degree-corrected ranks and F_perp; coupling on scripts/28 cells ----------------
    stage("T0(b): degree-corrected Bradley-Terry per field and on the Academia network")
    BT, BT0 = {}, {}
    gm1, gm0 = 0.0, 0.0
    for f in fl_all:
        x, net = INST[f]
        th, npairs, uninf, gm = bt_offset(net["A"])
        gm1 = max(gm1, gm)
        BT[f] = (pd.Series(th, index=net["nodes"]), pd.Series(uninf, index=net["nodes"]), npairs)
        th0, np0, un0, gm = bt_offset(net["A"], smooth=0.0)
        gm0 = max(gm0, gm)
        BT0[f] = (pd.Series(th0, index=net["nodes"]), pd.Series(un0, index=net["nodes"]), np0)
    put("b_bt_max_abs_gradient_primary", gm1, f"largest final max |gradient| over {len(fl_all)} field fits (+1 "
        f"convention); asserted < {BT_GASSERT:g}")
    put("b_bt_max_abs_gradient_exact_limit", gm0, f"same, exact-limit fits; asserted < {BT_GASSERT:g}")
    put("b_bt_max_abs_theta_exact_limit", float(max(np.abs(BT0[f][0].to_numpy()).max() for f in fl_all)))
    put("b_exact_limit_unidentified_node_share", float(np.mean(np.concatenate([BT0[f][1].to_numpy() for f in fl_all]))),
        "outcome-blind identification check: share of network nodes with no finite-offset pair (exact limit)")
    an = field_net("Academia", "Academia")
    gth, gnp, guninf, ggm = bt_offset(an["A"])
    put("b_academia_bt_max_abs_gradient", ggm)
    # revision audit (revision 1): where the first run's undamped Newton ended
    put("audit_n_field_fits", len(fl_all))
    aud = {sm: [bt_offset_first_run(INST[f][1]["A"], smooth=sm) + (f,) for f in fl_all] for sm in (1.0, 0.0)}
    for sm, lab in [(1.0, "primary"), (0.0, "exact_limit")]:
        bad = [a for a in aud[sm] if a[0] >= BT_GASSERT]
        put(f"audit_first_run_{lab}_fields_not_converged", len(bad), f"of {len(fl_all)} field fits: final max "
            f"|gradient| >= {BT_GASSERT:g} under the first run's undamped Newton")
        put(f"audit_first_run_{lab}_max_abs_gradient", max(a[0] for a in aud[sm]))
        put(f"audit_first_run_{lab}_max_abs_theta", max(a[1] for a in aud[sm]))
        put(f"audit_first_run_{lab}_fields", sorted(a[2] for a in bad))
    ga, ta = bt_offset_first_run(an["A"])
    put("audit_first_run_academia", [ga, ta], "final max |gradient| and max |theta| of the first run's Academia fit")
    akey = pd.Series([norm(nm) for nm in an["names"]], index=an["nodes"])
    GDC = pd.Series(gth, index=akey.values)
    GDC = GDC[~GDC.index.duplicated()]
    put("b_academia_bt_informative_pairs", gnp)
    put("b_academia_bt_uninformed_nodes", int(guninf.sum()), f"of {an['n']}")
    t = s28.build_table()
    key2wf = {v: k for k, v in wf2key.items()}
    t = t[t.field.isin(set(key2wf))]
    counts = t.groupby("field").size()
    fields_b = sorted(counts[counts >= 8].index)                     # scripts/28's per_field rule (n >= 8)
    cells, info = {}, []
    for k in fields_b:
        wf = key2wf[k]
        x, net = INST[wf]
        th, un, npairs = BT[wf]
        xx = x.set_index("inst_key")
        xx = xx[~xx.index.duplicated()]
        c = t[t.field == k][["inst_key", "F", "G", "y"]].copy()
        idn = c.inst_key.map(xx.InstitutionId)
        c["FDC"] = idn.map(th)
        c["FDC0"] = idn.map(BT0[wf][0])
        c["uninf"] = idn.map(BT0[wf][1])
        c["x"] = c.inst_key.map(xx.x)                                  # export ratio (validity diagnostic)
        # F_perp: residual of rank(F) on rank(p), rank(x) over the field's ranked institutions with p
        xr = x.dropna(subset=["p"])
        if len(xr) >= MIN_INST_T1:
            Xd = np.column_stack([np.ones(len(xr)), rk(xr.p), rk(xr.x)])
            yF = rk(xr.F)
            res = yF - Xd @ np.linalg.lstsq(Xd, yF, rcond=None)[0]
            fp = pd.Series(res, index=xr.inst_key.values)
            fp = fp[~fp.index.duplicated()]
            c["Fperp"] = c.inst_key.map(fp)
        else:
            c["Fperp"] = np.nan
        c["GDC"] = c.inst_key.map(GDC)
        n_all = len(c)
        c = c.dropna(subset=["F", "FDC", "Fperp", "GDC", "G", "y"]).sort_values("inst_key").reset_index(drop=True)
        info.append(dict(field=k, n_scripts28=n_all, n_common=len(c), bt_pairs=npairs,
                         bt_uninformed_in_cells=int(c.uninf.sum()),
                         rho_F_FDC_network=spear(x[x.in_net].F, x[x.in_net].InstitutionId.map(th)),
                         rho_F_Fperp=spear(c.F, c.Fperp) if len(c) >= 3 else np.nan))
        if len(c) >= 8:
            cells[k] = c
    fb = sorted(cells)
    put("b_fields", len(fb), "scripts/28 fields (n >= 8) with >= 8 common cells")
    put("b_cells_total", int(sum(len(c) for c in cells.values())),
        f"of {int(sum(i['n_scripts28'] for i in info))} scripts/28 cells in these fields")
    univ = sorted(set().union(*[set(c.inst_key) for c in cells.values()]))
    iid = {k: i for i, k in enumerate(univ)}
    arr = {k: dict(ix=np.array([iid[v] for v in c.inst_key]), F=c.F.to_numpy(float), G=c.G.to_numpy(float),
                   Y=c.y.to_numpy(float), FDC=c.FDC.to_numpy(float), Fp=c.Fperp.to_numpy(float),
                   GDC=c.GDC.to_numpy(float), unin=c.uninf.to_numpy(bool)) for k, c in cells.items()}

    def ev(f, W):
        a = arr[f]
        Wc = W[:, a["ix"]]
        rY = wrank(a["Y"], Wc)
        out = {}
        for nm in ["F", "FDC", "Fp", "G", "GDC"]:
            out["c_" + nm] = wcorr(wrank(a[nm], Wc), rY, Wc)
        return out
    stage("T0(b): bootstrap (shared and per-field institution draws)")
    obs, cb, ib, red = boot_engine(fb, len(univ), ev, pa0["b_shared"], pa0["b_indep"], tag="T0b")
    put("b_redrawn", red)
    ssf = pa0["b_fields"].spawn(6)
    names = ["c_F", "c_FDC", "c_Fp", "c_G", "c_GDC"]
    rDC = infer(ratio_of("c_FDC", "c_F"), obs, cb, ib, fb, ssf[0], names, "R_DC = mean c_DC / mean c_F")
    put("b_MDE80_R_DC", rDC["mde"], "reported first")
    put("b_R_DC", rDC)
    rP = infer(ratio_of("c_Fp", "c_F"), obs, cb, ib, fb, ssf[1], names, "R_perp = mean c_perp / mean c_F")
    put("b_R_perp", rP)
    gd = infer(diff_of("c_GDC", "c_G"), obs, cb, ib, fb, ssf[2], names, "mean (c_GDC - c_G)")
    put("b_GDC_minus_G", gd)
    for nm, j in [("c_F", 3), ("c_FDC", 4), ("c_Fp", 5)]:
        put(f"b_mean_{nm}", infer(mean_of(nm), obs, cb, ib, fb, ssf[j], names, f"mean {nm}"))
    put("b_mean_c_G", float(np.mean([obs["c_G"][f] for f in fb])))
    put("b_mean_c_GDC", float(np.mean([obs["c_GDC"][f] for f in fb])))
    it = pd.DataFrame(info)
    put("b_mean_rho_F_FDC_network", float(it.rho_F_FDC_network.mean()),
        "mean Spearman(F, F_DC) over ranked network nodes, all fields entering (b)")
    put("b_mean_rho_F_FDC_cells", float(np.mean([spear(cells[f].F, cells[f].FDC) for f in fb])))
    def t0_rule(r_dc, r_p):
        if r_dc["lo"] >= 0.75 and r_p["lo"] >= 0.75:
            return "production-robust"
        if r_dc["hi"] < 0.50 or r_p["hi"] < 0.50:
            return "production-dominated"
        return "partly production"
    v = t0_rule(rDC, rP)
    put("b_verdict", v, "A1 T0 rule (b); governs claim C1")

    # sensitivities (not in the verdict; revision 1): the registered offset completed in the exact limit, with the
    # converged fit. (s0) every common cell, theta = 0 (the middle) for nodes with no finite-offset pair; (s1) only
    # cells whose node has a finite-offset pair, fields with >= 8 such cells (R_perp recomputed on the same cells).
    # Two-way intervals from their own seeds; the registered rule is applied to each for information.
    stage("T0(b): sensitivities of the zero-degree completion (exact limit), bootstrap")
    put("b_sens_s0_unidentified_cell_share", float(np.mean(np.concatenate([cells[f].uninf.to_numpy(bool) for f in fb]))),
        "share of common cells whose node has theta = 0 from the ridge alone in the exact limit")
    arr0 = {k: dict(ix=arr[k]["ix"], F=arr[k]["F"], Y=arr[k]["Y"], FDC0=cells[k].FDC0.to_numpy(float))
            for k in fb}

    def ev0(f, W):
        a = arr0[f]
        Wc = W[:, a["ix"]]
        rY = wrank(a["Y"], Wc)
        return {"c_F": wcorr(wrank(a["F"], Wc), rY, Wc), "c_FDC0": wcorr(wrank(a["FDC0"], Wc), rY, Wc)}
    obs0, cb0, ib0, red0 = boot_engine(fb, len(univ), ev0, pa0["s0_shared"], pa0["s0_indep"], tag="T0s0")
    put("b_sens_s0_redrawn", red0)
    rS0 = infer(ratio_of("c_FDC0", "c_F"), obs0, cb0, ib0, fb, pa0["s0_fields"], ["c_F", "c_FDC0"],
                "(s0) R_DC, exact limit, theta = 0 for unidentified nodes")
    put("b_sens_R_DC_exact_limit", rS0, f"{len(fb)} fields, every common cell")
    put("b_sens_s0_rule_reading", t0_rule(rS0, rP), "registered rule with (s0) R_DC and the primary R_perp "
        "(same cells); information only")
    fb1 = [f for f in fb if int((~cells[f].uninf.astype(bool)).sum()) >= 8]
    arr1 = {}
    for k in fb1:
        mk = ~cells[k].uninf.to_numpy(bool)
        arr1[k] = dict(ix=arr[k]["ix"][mk], F=arr[k]["F"][mk], Y=arr[k]["Y"][mk], Fp=arr[k]["Fp"][mk],
                       FDC0=cells[k].FDC0.to_numpy(float)[mk])

    def ev1(f, W):
        a = arr1[f]
        Wc = W[:, a["ix"]]
        rY = wrank(a["Y"], Wc)
        return {"c_F": wcorr(wrank(a["F"], Wc), rY, Wc), "c_FDC0": wcorr(wrank(a["FDC0"], Wc), rY, Wc),
                "c_Fp": wcorr(wrank(a["Fp"], Wc), rY, Wc)}
    obs1, cb1, ib1, red1 = boot_engine(fb1, len(univ), ev1, pa0["s1_shared"], pa0["s1_indep"], tag="T0s1")
    put("b_sens_s1_redrawn", red1)
    put("b_sens_s1_fields", len(fb1), "fields with >= 8 common cells whose node is identified in the exact limit")
    put("b_sens_s1_cells", int(sum(len(arr1[k]["ix"]) for k in fb1)))
    ss1 = pa0["s1_fields"].spawn(2)
    rS1 = infer(ratio_of("c_FDC0", "c_F"), obs1, cb1, ib1, fb1, ss1[0], ["c_F", "c_FDC0", "c_Fp"],
                "(s1) R_DC, exact limit, identified cells only")
    rS1p = infer(ratio_of("c_Fp", "c_F"), obs1, cb1, ib1, fb1, ss1[1], ["c_F", "c_FDC0", "c_Fp"],
                 "(s1) R_perp on the same identified cells")
    put("b_sens_R_DC_exact_limit_identified_only", rS1)
    put("b_sens_R_perp_identified_only", rS1p)
    put("b_sens_s1_rule_reading", t0_rule(rS1, rS1p), "registered rule on the identified cells; information only")
    # validity diagnostic (exploratory, revision 1): does F_DC still carry the export ratio x?
    dx = np.array([(spear(cells[f].F, cells[f].x), spear(cells[f].FDC, cells[f].x), spear(cells[f].FDC0, cells[f].x))
                   for f in fb])
    put("b_diag_mean_rho_F_x_cells", float(np.nanmean(dx[:, 0])), f"exploratory; mean over {len(fb)} fields, common cells")
    put("b_diag_mean_rho_FDC_x_cells", float(np.nanmean(dx[:, 1])), "exploratory; F_DC with the +1 convention")
    put("b_diag_mean_rho_FDC0_x_cells", float(np.nanmean(dx[:, 2])), "exploratory; F_DC in the exact limit (s0)")
    put("b_diag_fields_rho_FDC_x_above_rho_F_x", int(np.sum(dx[:, 1] > dx[:, 0])), f"of {len(fb)} fields")
    tb = pd.DataFrame([dict(field=f, n=len(cells[f]), c_F=obs["c_F"][f], c_DC=obs["c_FDC"][f],
                            c_perp=obs["c_Fp"][f], c_G=obs["c_G"][f], c_GDC=obs["c_GDC"][f]) for f in fb])
    tb = tb.merge(it, on="field", how="left")
    put("b_table", tb.round(4).to_dict("records"))

    # ---------------- (c) minimum-violation ordering ----------------
    stage("T0(c): minimum-violation orderings (local search, 50 random restarts)")
    ssr = dict(zip(fl_all, pa0["c_restarts"].spawn(len(fl_all))))
    jobs = [(f, INST[f][1]["A"], ssr[f]) for f in fl_all]
    MV = {}
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            for f, o, cst, c0 in pool.imap(mvr_field, jobs):
                MV[f] = (o, cst, c0)
    else:
        for j in jobs:
            f, o, cst, c0 = mvr_field(j)
            MV[f] = (o, cst, c0)
    crows = []
    for f in fl_all:
        x, net = INST[f]
        o, cst, c0 = MV[f]
        posm = np.empty(net["n"])
        posm[o] = np.arange(net["n"])
        mvr_pct = pd.Series(1.0 - posm / (net["n"] - 1), index=net["nodes"])
        xin = x[x.in_net]
        rho = spear(xin.F, xin.InstitutionId.map(mvr_pct))
        # upward hires among ranked network nodes: F order vs MVR order restricted to the same nodes
        ids = xin.sort_values("Rank").InstitutionId.to_numpy(float)
        pos = pd.Series(np.arange(net["n"]), index=net["nodes"])
        sub = pos.reindex(ids).to_numpy(int)
        Asub = net["A"][np.ix_(sub, sub)]
        up_F = upward(Asub, np.arange(len(sub)))
        mv_sub = np.argsort(posm[sub], kind="mergesort")
        up_M = upward(Asub, mv_sub)
        crows.append(dict(field=f, n_net=net["n"], persons=float(net["tot"].sum()), up_MVR=cst, up_SR=c0,
                          up_F_ranked=up_F, up_MVR_ranked=up_M, rho_F_MVR=rho))
    ct = pd.DataFrame(crows)
    put("c_mean_rho_F_MVR", float(ct.rho_F_MVR.mean()), f"{len(ct)} fields")
    put("c_share_upward_MVR", float(ct.up_MVR.sum() / ct.persons.sum()))
    put("c_share_upward_SpringRank_start", float(ct.up_SR.sum() / ct.persons.sum()))
    put("c_upward_F_vs_MVR_ranked", [float(ct.up_F_ranked.sum()), float(ct.up_MVR_ranked.sum())],
        "summed over fields, edges among ranked network nodes")
    put("c_table", ct.round(4).to_dict("records"))
    return RES["T0"]


# =============================================================================================================
# T5 (A1 [V]): invariance of the field map to non-academic status indices
# =============================================================================================================
N_HALV = 300
MIN_T5 = 60
ZNAMES = {"Z1": "AVGFACSAL", "Z2": "INEXPFTE", "Z3": "ENDOWBEGIN / UGDS"}


def run_T5():
    CUR["test"] = "T5"
    from src.load_er import load_er_scorecard
    s28 = mod("s28")
    stage("T5: scripts/28 4-yr cells + Scorecard institution indices")
    t = s28.build_table()
    er = load_er_scorecard(earn_col="EARN_MDN_4YR", count_col="EARN_COUNT_WNE_4YR", fields=s28.FIELDS66)
    t = t.merge(er[["inst_key", "field", "institution_id"]].rename(columns={"institution_id": "UNITID"}),
                on=["inst_key", "field"], how="left")
    sc = scorecard_names().drop_duplicates("UNITID")
    sc["Z1"] = sc.AVGFACSAL
    sc["Z2"] = sc.INEXPFTE
    sc["Z3"] = sc.ENDOWBEGIN / sc.UGDS.where(sc.UGDS > 0)
    t = t.merge(sc[["UNITID", "Z1", "Z2", "Z3"]], on="UNITID", how="left")
    t = t.dropna(subset=["F", "y", "Z1", "Z2", "Z3"]).sort_values(["field", "inst_key"])
    n = t.groupby("field").size()
    fl = sorted(n[n >= MIN_T5].index)
    put("n_fields", len(fl), f">= {MIN_T5} institutions with F, Y and all three Z")
    if len(fl) < 10:
        put("verdict", "infeasible")
        return RES["T5"]
    pa = proc_seeds(SS_A1["T5"], ["fields", "halvings"])
    g = np.random.default_rng(pa["fields"])
    K = len(fl)
    FI = g.integers(K, size=(B, K))
    mult = max(int(np.bincount(FI[b], minlength=K).max()) for b in range(B))
    put("max_multiplicity", mult, "halving blocks per field (independent halvings for repeated fields)")
    NH = N_HALV * mult
    ssf = dict(zip(fl, pa["halvings"].spawn(K)))
    vars_ = ["F", "Z1", "Z2", "Z3"]
    C = {v: np.full((2, NH, K), np.nan) for v in vars_}      # [half, halving, field]
    full = {}
    for j, f in enumerate(fl):
        x = t[t.field == f]
        Y = x.y.to_numpy(float)
        X = {v: x[v].to_numpy(float) for v in vars_}
        full[f] = {v: spear(X[v], Y) for v in vars_}
        m = len(x)
        perm = np.stack([gg.permutation(m) for gg in gens(ssf[f], NH)])          # (NH, m), one stream per halving
        for hf, idx in enumerate((perm[:, : m // 2], perm[:, m // 2:])):
            ry = Y[idx]
            for v in vars_:
                C[v][hf, :, j] = rowwise_spear(X[v][idx], ry)
    put("full_sample_map", {f: full[f] for f in fl})
    put("mean_c_F", float(np.mean([full[f]["F"] for f in fl])))

    def stats(cols, copy_of):
        """cols: field positions (K,); copy_of: copy number of each position -> r_obs, rel_F, rel_Z, r*."""
        blk = copy_of[:, None] * N_HALV + np.arange(N_HALV)[None, :]          # (K, 300)
        A_ = {v: C[v][0][blk.T, cols[None, :]] for v in vars_}                 # (300, K)
        B_ = {v: C[v][1][blk.T, cols[None, :]] for v in vars_}
        relF = float(np.nanmean(rowwise_spear(A_["F"], B_["F"])))
        out = {}
        for z in ["Z1", "Z2", "Z3"]:
            robs = float(np.nanmean(np.concatenate([rowwise_spear(A_["F"], B_[z]), rowwise_spear(B_["F"], A_[z])])))
            relZ = float(np.nanmean(rowwise_spear(A_[z], B_[z])))
            rs = robs / np.sqrt(relF * relZ) if relF > 0 and relZ > 0 else np.nan
            out[z] = (robs, relZ, rs)
        return relF, out

    relF, o = stats(np.arange(K), np.zeros(K, int))
    put("rel_F_map", relF, "mean Spearman across fields between the two halves' c_F maps")
    boots = {z: np.full(B, np.nan) for z in ZNAMES}
    for b in range(B):
        cols = FI[b]
        cnt = {}
        cp = np.empty(K, int)
        for i, c in enumerate(cols):
            cp[i] = cnt.get(c, 0)
            cnt[c] = cp[i] + 1
        _, ob = stats(cols, cp)
        for z in ZNAMES:
            boots[z][b] = ob[z][2]
    verd = []
    for z, lab in ZNAMES.items():
        bz = boots[z][np.isfinite(boots[z])]
        lo, hi = float(np.percentile(bz, 2.5)), float(np.percentile(bz, 97.5))
        put(f"{z}_MDE80", MDE_K * float(bz.std(ddof=1)), "2.80 x field-bootstrap SE of r* (before the estimate)")
        put(f"{z}_r_obs", o[z][0], lab)
        put(f"{z}_rel_Z", o[z][1])
        put(f"{z}_r_star", o[z][2])
        put(f"{z}_r_star_ci", [lo, hi], f"field bootstrap of the whole procedure; {len(bz)} finite of {B}")
        verd.append((lo, hi))
    if all(lo >= 0.70 for lo, _ in verd):
        v = "supported"
    elif sum(hi < 0.70 for _, hi in verd) >= 2:
        v = "contradicted"
    else:
        v = "inconclusive"
    put("verdict", v, "A1 T5; governs claim C6")
    return RES["T5"]


# =============================================================================================================
# T2 (B0) and T2' (A1): the academy's choice among the same bachelor's graduates (ORCID)
# =============================================================================================================
T2_FIELDS = {"Computer Science": ("computer", "computer_science"), "Mathematics": ("math", "mathematics"),
             "Physics, General": ("physics", "physics"), "Psychology, General": ("psycholog", "psychology"),
             "Economics, General": ("econ", "economics"), "Chemistry": ("chemist", "chemistry"),
             "Biological Sciences, General": ("biolog", "biology")}
RE_DOC = re.compile(r"ph\.?\s?d|d\.phil|dphil|sc\.?d|\bdoctor|doctoral", re.I)
RE_BAC = re.compile(r"bachelor|\bb\.?\s?s\.?c?\b|\bb\.?\s?a\.?\b|\bbsc\b|\ba\.?b\.?\b|\bs\.?b\.?\b|undergrad|\bbs\b|\bba\b",
                    re.I)
RE_PROF = re.compile(r"\bm\.?\s?d\b|medicine|medical|pharm|juris|\bj\.?\s?d\b|\blaw\b|dental|\bdds\b|\bdmd\b|"
                     r"osteopath|nursing|\bdnp\b|physical therap|\bdpt\b|audiolog|psy\.?\s?d|\bed\.?\s?d\b|"
                     r"veterinar|\bdvm\b|optometr|chiropract|occupational therap|ministry|\bd\.?\s?min\b|\bdba\b|"
                     r"business admin|public health|\bdrph\b", re.I)
RE_POSTPRE = re.compile(r"post\W?doc|pre\W?doc", re.I)
RE_ANY_KW = re.compile("computer|math|physics|psycholog|econ|chemist|biolog")
MIN_CELL_PERSONS = 3
MIN_CELLS_T2 = 15
N_REL_SPLIT = 200


def orcid_episodes() -> pd.DataFrame:
    """A2.2: episodes rebuilt from the edge endpoints, deduplicated by (person_orcid, aff_node_id); kept: US
    education episodes with a ROR id whose role matches BAC or DOC and whose role + department contains one of
    the seven field keywords (a superset of every variant used below)."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pcm
    shards = sorted(ORCID_DIR.glob("*.parquet"))
    parts = []
    for i, p in enumerate(shards):
        cols = ["person_orcid"]
        for s in ("from", "to"):
            cols += [f"aff_node_id_{s}", f"org_country_{s}", f"org_{s}_ror_id", f"org_{s}_ror_name", f"role_{s}",
                     f"role_type_{s}", f"org_dept_{s}", f"epi_start_year_{s}"]
        t = pq.read_table(p, columns=cols)
        for s in ("from", "to"):
            m = pcm.and_(pcm.and_(pcm.equal(t[f"role_type_{s}"], "education"), pcm.equal(t[f"org_country_{s}"], "us")),
                         pcm.is_valid(t[f"org_{s}_ror_id"]))
            x = t.filter(pcm.fill_null(m, False))
            if x.num_rows == 0:
                continue
            d = pd.DataFrame({"person": x["person_orcid"].to_pandas(), "node": x[f"aff_node_id_{s}"].to_pandas(),
                              "ror": x[f"org_{s}_ror_id"].to_pandas(), "rorname": x[f"org_{s}_ror_name"].to_pandas(),
                              "role": x[f"role_{s}"].to_pandas().fillna(""),
                              "dept": x[f"org_dept_{s}"].to_pandas().fillna(""),
                              "start": pd.to_numeric(x[f"epi_start_year_{s}"].to_pandas(), errors="coerce")})
            txt = (d.role + " | " + d.dept).str.lower()
            keep = (d.role.str.contains(RE_BAC) | d.role.str.contains(RE_DOC)) & txt.str.contains(RE_ANY_KW)
            parts.append(d[keep])
        if i % 100 == 0:
            stage(f"T2: ORCID shard {i}/{len(shards)}")
    ep = pd.concat(parts, ignore_index=True)
    del parts
    ep = ep.sort_values(["person", "node", "start"], na_position="last").drop_duplicates(["person", "node"])
    ep["text"] = (ep.role + " | " + ep.dept).str.lower()
    ep["bac"] = ep.role.str.contains(RE_BAC) & ~ep.role.str.contains(RE_DOC)
    doc = ep.role.str.contains(RE_DOC) & ~ep.role.str.contains(RE_PROF)
    ep["doc_lit"] = doc
    ep["doc"] = doc & ~ep.role.str.contains(RE_POSTPRE)
    return ep.reset_index(drop=True)


def t2_records(ep: pd.DataFrame, doc_col: str, anchored: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(different-institution records, one per person x bachelor's ROR x field with the earliest qualifying
    doctorate; same-institution flags per person x bachelor's ROR x field)."""
    recs, same = [], []
    for wf, (kw, key) in T2_FIELDS.items():
        pat = re.compile(r"\b" + kw) if anchored else re.compile(re.escape(kw))
        m = ep.text.str.contains(pat)
        b = ep[m & ep.bac][["person", "node", "ror", "rorname", "start"]]
        d = ep[m & ep[doc_col]][["person", "node", "ror", "rorname", "start"]]
        j = b.merge(d, on="person", suffixes=("_b", "_d"))
        j = j[~(j.start_d < j.start_b)]                      # not before the bachelor's when both recorded
        s = j.assign(same=j.ror_b == j.ror_d).groupby(["person", "ror_b", "rorname_b"]).same.agg(
            ["any", "all"]).reset_index()
        s["field"] = wf
        same.append(s)
        j = j[j.ror_b != j.ror_d].copy()
        j["_s"] = j.start_d.fillna(np.inf)
        j = j.sort_values(["person", "ror_b", "_s", "node_d"]).drop_duplicates(["person", "ror_b"])
        j["field"] = wf
        recs.append(j[["person", "field", "ror_b", "rorname_b", "ror_d", "rorname_d", "start_b", "start_d"]])
    return pd.concat(recs, ignore_index=True), pd.concat(same, ignore_index=True)


def ror_states() -> pd.Series:
    with zipfile.ZipFile(ROR_ZIP) as z:
        n = [x for x in z.namelist() if x.endswith(".csv")][0]
        d = pd.read_csv(z.open(n), usecols=["id", "locations.geonames_details.country_code",
                                             "locations.geonames_details.country_subdivision_code"], dtype=str)
    d = d[d["locations.geonames_details.country_code"].fillna("").str.startswith("US")]
    st = d["locations.geonames_details.country_subdivision_code"].str.split(";").str[0]
    return pd.Series(st.values, index=d["id"].values)


def t2_cells(rec: pd.DataFrame, states: pd.Series | None, ylog: pd.DataFrame) -> tuple[dict, dict]:
    """Cells (bachelor's institution x field): D, D^G, F, G, persons, state share, Y."""
    rk_ = wapman()[1]
    info = {}
    acad = rk_[rk_.TaxonomyLevel == "Academia"].copy()
    acad["key"] = acad.InstitutionName.map(norm)
    acad["pct"] = 1.0 - acad.Rank / (len(acad) - 1)
    gmap = acad.drop_duplicates("key").set_index("key").pct
    cells = {}
    for wf, (kw, key) in T2_FIELDS.items():
        x = rk_[(rk_.TaxonomyLevel == "Field") & (rk_.TaxonomyValue == wf)].copy()
        x["key"] = x.InstitutionName.map(norm)
        x["pct"] = 1.0 - x.Rank / (len(x) - 1)
        fmap = x.drop_duplicates("key").set_index("key").pct
        r = rec[rec.field == wf].copy()
        r["kb"] = r.rorname_b.map(norm)
        r["kd"] = r.rorname_d.map(norm)
        r["F"] = r.kb.map(fmap)
        r["G"] = r.kb.map(gmap)
        r["D"] = r.kd.map(fmap)
        r["DG"] = r.kd.map(gmap)
        info[wf] = dict(records=len(r), bac_matched_F=float(r.F.notna().mean()) if len(r) else np.nan,
                        doc_matched_D=float(r.D.notna().mean()) if len(r) else np.nan,
                        bac_matched_G=float(r.G.notna().mean()) if len(r) else np.nan)
        if states is not None:
            sb, sd = r.ror_b.map(states), r.ror_d.map(states)
            r["same_state"] = np.where(sb.notna() & sd.notna(), (sb == sd).astype(float), np.nan)
        else:
            r["same_state"] = np.nan
        r = r[r.D.notna()]
        g = r.groupby("kb").agg(n=("person", "size"), D=("D", "mean"), DG=("DG", "mean"), F=("F", "first"),
                                G=("G", "first"), SS=("same_state", "mean")).reset_index()
        g = g[(g.n >= MIN_CELL_PERSONS) & g.F.notna() & g.G.notna()].rename(columns={"kb": "inst_key"})
        yy = ylog[ylog.field == key].set_index("inst_key").logy
        g["Y"] = g.inst_key.map(yy[~yy.index.duplicated()])
        info[wf].update(cells=len(g), cells_Y=int(g.Y.notna().sum()), persons=int(g.n.sum()))
        cells[wf] = (g.sort_values("inst_key").reset_index(drop=True), r[r.kb.isin(g.inst_key)])
    return cells, info


def t2_eval_factory(cells: dict, iid: dict):
    arr = {}
    for f, (g, _) in cells.items():
        a = dict(ix=np.array([iid[k] for k in g.inst_key]), F=g.F.to_numpy(float), G=g.G.to_numpy(float),
                 D=g.D.to_numpy(float), DG=g.DG.to_numpy(float), Y=g.Y.to_numpy(float), SS=g.SS.to_numpy(float))
        a["mY"] = np.isfinite(a["Y"])
        a["mDG"] = np.isfinite(a["DG"])
        a["mSS"] = np.isfinite(a["SS"])
        arr[f] = a

    def ev(f, W):
        a = arr[f]
        Wc = W[:, a["ix"]]
        out = {}
        rF, rG, rD = wrank(a["F"], Wc), wrank(a["G"], Wc), wrank(a["D"], Wc)
        out["rFD"] = wcorr(rF, rD, Wc)
        out["rGD"] = wcorr(rG, rD, Wc)
        out["dD"] = out["rFD"] - out["rGD"]
        res = wresid(np.stack([rF, rD], -1), [rG], Wc)
        out["pFD"] = wcorr(res[:, :, 0], res[:, :, 1], Wc)
        if a["mDG"].sum() >= MIN_CELLS_T2:
            m = a["mDG"]
            out["pFDG"] = wpartial(a["F"][m], a["DG"][m], [a["G"][m]], Wc[:, m])
        if a["mY"].sum() >= MIN_CELLS_T2:
            m = a["mY"]
            Wm = Wc[:, m]
            rFm, rGm, rYm = wrank(a["F"][m], Wm), wrank(a["G"][m], Wm), wrank(a["Y"][m], Wm)
            out["rFY"] = wcorr(rFm, rYm, Wm)
            out["dY"] = out["rFY"] - wcorr(rGm, rYm, Wm)
            out["dD_minus_dY"] = out["dD"] - out["dY"]
            r2 = wresid(np.stack([rFm, rYm, wrank(a["D"][m], Wm)], -1), [rGm], Wm)
            out["pFY"] = wcorr(r2[:, :, 0], r2[:, :, 1], Wm)
            out["pFD_Ycells"] = wcorr(r2[:, :, 0], r2[:, :, 2], Wm)
            out["Kf"] = out["pFD"] - out["pFY"]
            out["Kf_same"] = out["pFD_Ycells"] - out["pFY"]
        if a["mSS"].sum() >= MIN_CELLS_T2:
            m = a["mSS"]
            out["pFD_state"] = wpartial(a["F"][m], a["D"][m], [a["G"][m], a["SS"][m]], Wc[:, m])
        return out
    return ev


def t2_run_boot(cells, ss_proc: dict, label: str):
    univ = sorted(set().union(*[set(g.inst_key) for g, _ in cells.values()]))
    iid = {k: i for i, k in enumerate(univ)}
    fl = sorted(cells)
    ev = t2_eval_factory(cells, iid)
    obs, cb, ib, red = boot_engine(fl, len(univ), ev, ss_proc["shared"], ss_proc["indep"], tag=label)
    return fl, obs, cb, ib, red, len(univ)


def fields_with(obs, name):
    return sorted(obs.get(name, {}))


def run_T2():
    CUR["test"] = "T2"
    s28 = mod("s28")
    stage("T2: ORCID episodes (681 shards)")
    ep = orcid_episodes()
    put("episodes_kept", len(ep), "US education episodes with ROR, BAC or DOC role, a field keyword")
    put("persons", int(ep.person.nunique()))
    t = s28.build_table()
    ylog = t[["inst_key", "field", "y"]].assign(logy=lambda d: np.log(d.y))
    states = ror_states()
    pb0 = proc_seeds(SS_B0["T2"], ["shared", "indep", "fields", "lit_shared", "lit_indep", "lit_fields"])
    pa1 = proc_seeds(SS_A1["T2'"], ["shared", "indep", "fields", "rel_splits", "anch_shared", "anch_indep",
                                    "anch_fields", "lit_shared", "lit_indep", "lit_fields"])
    variants = {}
    for vname, doc_col, anch in [("primary", "doc", False), ("literal_B0_regex", "doc_lit", False),
                                 ("anchored_keywords", "doc", True)]:
        rec, same = t2_records(ep, doc_col, anch)
        cells, info = t2_cells(rec, states, ylog)
        variants[vname] = (rec, same, cells, info)
        put(f"{vname}_counts", {f: dict(persons_diff_inst=int((rec.field == f).sum()), **info[f]) for f in T2_FIELDS})
    del ep
    gc.collect()
    rec, same, cells, info = variants["primary"]
    ok = {f: c for f, c in cells.items() if len(c[0]) >= MIN_CELLS_T2}
    put("fields_entering", sorted(ok), f">= {MIN_CELLS_T2} cells of >= {MIN_CELL_PERSONS} persons")
    if len(ok) < 4:
        put("B0_verdict", "infeasible")
        put("A1_verdict", "infeasible")
        return RES["T2"]
    cells = ok

    # ---------------- B0 ----------------
    stage("T2: B0 bootstrap (crossed field x institution)")
    fl, obs, cb, ib, red, NI = t2_run_boot(cells, pb0, "T2-B0")
    put("B0_redrawn", red)
    put("B0_n_institutions", NI)
    fY = fields_with(obs, "dY")
    ssf = pb0["fields"].spawn(4)
    rdD = infer(mean_of("dD"), obs, cb, ib, fl, ssf[0], ["dD"], "B0 mean Delta^D")
    put("B0_MDE80_meanDeltaD", MDE_K * rdD["xse"], "2.80 x crossed-bootstrap SE (G3; reported first)")
    put("B0_mean_DeltaD", rdD)
    rpar = infer(mean_of("pFD"), obs, cb, ib, fl, ssf[1], ["pFD"], "B0 mean rho(F,D|G)")
    put("B0_mean_pFD", rpar)
    rdd = infer(mean_of("dD_minus_dY"), obs, cb, ib, fY, ssf[2], ["dD_minus_dY"], "B0 mean (Delta^D - Delta^Y)")
    put("B0_mean_dD_minus_dY", rdd)
    rdY = infer(mean_of("dY"), obs, cb, ib, fY, ssf[3], ["dY"], "B0 mean Delta^Y (T2 cells with Y)")
    put("B0_mean_DeltaY", rdY)
    sup = rdD["xlo"] > 0 and rdd["xlo"] > 0
    con = rdD["xhi"] < 0 or rpar["xhi"] < 0.05
    v0 = "inconclusive" if (sup and con) else ("supported" if sup else ("contradicted" if con else "inconclusive"))
    put("B0_verdict_raw", v0, "B0 rule on crossed percentile intervals (A2.6, A2.8)")
    per = []
    for f in fl:
        rw = dict(field=f, n_cells=len(cells[f][0]), persons=int(cells[f][0].n.sum()))
        for k in ["rFD", "rGD", "dD", "pFD", "pFDG", "rFY", "dY", "pFY", "pFD_state"]:
            if f in obs.get(k, {}):
                v = ib[k][f]
                rw[k] = obs[k][f]
                rw[k + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        per.append(rw)
    put("B0_per_field", per)

    # ---------------- A1 T2' ----------------
    stage("T2': A1 bootstrap (two-way)")
    fl1, obs1, cb1, ib1, red1, _ = t2_run_boot(cells, pa1, "T2-A1")
    put("A1_redrawn", red1)
    ssa = pa1["fields"].spawn(8)
    fDG = fields_with(obs1, "pFDG")
    fY1 = fields_with(obs1, "Kf")
    PiD = infer(mean_of("pFD"), obs1, cb1, ib1, fl1, ssa[0], ["pFD"], "Pi^D")
    put("A1_MDE80_PiD", PiD["mde"], "reported first")
    put("A1_PiD", PiD)
    PiDG = infer(mean_of("pFDG"), obs1, cb1, ib1, fDG, ssa[1], ["pFDG"], "Pi^DG")
    put("A1_PiDG", PiDG)
    PiY = infer(mean_of("pFY"), obs1, cb1, ib1, fY1, ssa[2], ["pFY"], "Pi^Y")
    put("A1_PiY", PiY)
    Kr = infer(mean_of("Kf"), obs1, cb1, ib1, fY1, ssa[3], ["Kf"], "K")
    put("A1_K", Kr)
    Ks = infer(mean_of("Kf_same"), obs1, cb1, ib1, fY1, ssa[4], ["Kf_same"], "K on the Y cells only (sensitivity)")
    put("A1_K_same_cells", Ks)
    sup = PiD["lo"] > 0 and PiDG["lo"] > 0 and Kr["lo"] > 0
    con = PiD["hi"] < 0 and PiDG["hi"] < 0
    v1 = "inconclusive" if (sup and con) else ("supported" if sup else ("contradicted" if con else "inconclusive"))
    put("A1_verdict", v1, "A1 T2' rule (A5.4); governs claim C3")
    fS = fields_with(obs1, "pFD_state")
    if fS:
        put("A5.5_PiD_state_control", infer(mean_of("pFD_state"), obs1, cb1, ib1, fS, ssa[5], ["pFD_state"],
                                            "Pi^D + same-state share control"))
    # Gaussian-copula partial (A2.5 sensitivity)
    gc_ = []
    for f in fl1:
        g = cells[f][0]
        rr = {k: 2 * np.sin(np.pi * spear(g[a], g[b]) / 6) for k, (a, b) in
              {"FD": ("F", "D"), "FG": ("F", "G"), "DG": ("D", "G")}.items()}
        gc_.append((rr["FD"] - rr["FG"] * rr["DG"]) / np.sqrt((1 - rr["FG"] ** 2) * (1 - rr["DG"] ** 2)))
    put("A2.5_mean_gaussian_copula_partial", float(np.mean(gc_)))

    # A5.5: reliability of D (persons split at random within cells), reliability-corrected Delta^D
    stage("T2': reliability of D and the other A5.5 items")
    gg = gens(pa1["rel_splits"], N_REL_SPLIT)
    relD = {}
    for f in fl1:
        g, r = cells[f]
        rr = r.sort_values(["kb", "person"])
        groups = [x.D.to_numpy(float) for _, x in rr.groupby("kb", sort=True)]
        vals = []
        for gen_ in gg:
            DA, DB = [], []
            for dv in groups:
                p = gen_.permutation(len(dv))
                h = len(dv) // 2
                DA.append(dv[p[:h]].mean())
                DB.append(dv[p[h:]].mean())
            vals.append(spear(np.array(DA), np.array(DB)))
        rh = float(np.nanmean(vals))
        relD[f] = dict(r_half=rh, rel_SB=2 * rh / (1 + rh))
    put("A5.5_reliability_D", relD)
    rel = pd.read_csv(REL56)
    rel = rel[rel.unit == "field"]
    corr_rows = []
    for f in fl1:
        key = T2_FIELDS[f][1]
        rr = rel[rel.field == key]
        if not len(rr):
            continue
        lb, ext = float(rr.relF_LB.iloc[0]), float(rr.relF_EXT.iloc[0])
        a, b_ = obs1["rFD"][f], obs1["rGD"][f]
        corr_rows.append(dict(field=f, relF_LB=lb, relF_EXT=ext, dD=a - b_,
                              dD_corr_LB=a / np.sqrt(lb) - b_ / np.sqrt(0.99),
                              dD_corr_EXT=a / np.sqrt(ext) - b_ / np.sqrt(0.99)))
    cr = pd.DataFrame(corr_rows)
    put("A5.5_DeltaD_corrected", cr.round(4).to_dict("records"))
    put("A5.5_mean_DeltaD_corrected_LB", float(cr.dD_corr_LB.mean()))
    put("A5.5_mean_DeltaD_corrected_EXT", float(cr.dD_corr_EXT.mean()))
    if v0 == "contradicted" and rdD["xhi"] < 0 and not (cr.dD_corr_LB.mean() < 0 and cr.dD_corr_EXT.mean() < 0):
        v0 = "contradicted as registered; attributable to reliability"
    put("B0_verdict", v0, "B0 verdict read with A5.6")
    # same-institution retention by tercile of F (records dropped by B0)
    rk_ = wapman()[1]
    ret = []
    for f in fl1:
        x = rk_[(rk_.TaxonomyLevel == "Field") & (rk_.TaxonomyValue == f)].copy()
        x["key"] = x.InstitutionName.map(norm)
        fmap = x.assign(p=1.0 - x.Rank / (len(x) - 1)).drop_duplicates("key").set_index("key").p
        s = same[same.field == f].copy()
        s["kb"] = s.rorname_b.map(norm)
        s["F"] = s.kb.map(fmap)
        s = s[s.F.notna()]
        cnt = s.groupby("kb").size()
        s = s[s.kb.isin(cnt[cnt >= MIN_CELL_PERSONS].index)]
        if s.kb.nunique() < 9:
            continue
        fk = s.drop_duplicates("kb").set_index("kb").F
        terc = pd.qcut(fk.rank(method="first"), 3, labels=["low", "mid", "high"])
        s["terc"] = s.kb.map(terc)
        sh = s.groupby("terc", observed=True)["all"].mean()
        ret.append(dict(field=f, **{f"same_share_{k}": float(v) for k, v in sh.items()}, n_persons=len(s)))
    put("A5.5_same_institution_share_by_F_tercile", ret,
        "persons with a qualifying doctorate at the bachelor's institution (ROR name-matched institutions only)")

    # sensitivities: anchored keywords (A1 seeds) and the literal B0 regex (B0 and A1 seeds)
    stage("T2: sensitivities (anchored keywords; literal B0 doctorate regex)")
    for vname, seeds_b0, seeds_a1 in [("anchored_keywords", None, {"shared": pa1["anch_shared"],
                                                                   "indep": pa1["anch_indep"],
                                                                   "fields": pa1["anch_fields"]}),
                                      ("literal_B0_regex", {"shared": pb0["lit_shared"], "indep": pb0["lit_indep"],
                                                            "fields": pb0["lit_fields"]},
                                       {"shared": pa1["lit_shared"], "indep": pa1["lit_indep"],
                                        "fields": pa1["lit_fields"]})]:
        cv = {f: c for f, c in variants[vname][2].items() if len(c[0]) >= MIN_CELLS_T2}
        if len(cv) < 4:
            put(f"{vname}_status", f"infeasible ({len(cv)} fields)")
            continue
        flv, o, c_, i_, _, _ = t2_run_boot(cv, seeds_a1, vname)
        s_ = seeds_a1["fields"].spawn(2)
        put(f"{vname}_PiD", infer(mean_of("pFD"), o, c_, i_, flv, s_[0], ["pFD"], f"{vname}: Pi^D"))
        if seeds_b0 is not None:
            flb, ob, cbb, ibb, _, _ = t2_run_boot(cv, seeds_b0, vname + "-B0")
            s0 = seeds_b0["fields"].spawn(2)
            put(f"{vname}_B0_mean_DeltaD", infer(mean_of("dD"), ob, cbb, ibb, flb, s0[0], ["dD"],
                                                 f"{vname}: mean Delta^D"))
            fYv = fields_with(ob, "dD_minus_dY")
            if fYv:
                put(f"{vname}_B0_mean_dD_minus_dY", infer(mean_of("dD_minus_dY"), ob, cbb, ibb, fYv, s0[1],
                                                          ["dD_minus_dY"], f"{vname}: mean (Delta^D - Delta^Y)"))
    return RES["T2"]


# =============================================================================================================
# T3 (B0) and T3' (A1): where the career rise occurs (PSEO V4.14.1 sample A of scripts/66)
# =============================================================================================================
N_PERM_T3 = 200
NMIN_T3 = 15           # scripts/66 NMIN (B0)
NMIN_T3_CTRL = 18      # scripts/66 NMIN_C for cells with the S2 controls (residual df >= 10)
NMIN_TYPE = 10         # A6.3: cells with >= 10 institutions of the type
T3_B0_NULL = 0.019


def flows_rows(path: Path, aggs: list[str], cohorts: list[str] | None, cips: list[str] | None, hz: list[str],
               instate: bool = False) -> pd.DataFrame:
    """Bachelor's rows of a PSEO Flows file (streamed): agg levels, cohorts, CIP-2 codes; employed counts (and the
    in-state counts) at the given horizons, NaN unless status == 1."""
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.compute as pcm
    scols = ["agg_level_pseo", "institution", "degree_level", "cipcode", "grad_cohort", "geography", "industry"]
    vcols = [f"{h}_grads_emp" for h in hz] + ([f"{h}_grads_emp_instate" for h in hz] if instate else [])
    fcols = [f"status_{c}" for c in vcols]
    types = {**{c: pa.string() for c in scols}, **{c: pa.float64() for c in vcols}, **{c: pa.int8() for c in fcols}}
    co = pacsv.ConvertOptions(include_columns=scols + vcols + fcols, column_types=types)
    rd = pacsv.open_csv(path, convert_options=co, read_options=pacsv.ReadOptions(block_size=1 << 23, use_threads=False))
    keep = []
    for b in rd:
        t = pa.Table.from_batches([b])
        m = pcm.and_(pcm.is_in(t.column("agg_level_pseo"), value_set=pa.array(aggs)),
                     pcm.equal(t.column("degree_level"), "05"))
        if cohorts is not None:
            m = pcm.and_(m, pcm.is_in(t.column("grad_cohort"), value_set=pa.array(cohorts)))
        if cips is not None:
            m = pcm.and_(m, pcm.is_in(t.column("cipcode"), value_set=pa.array(cips)))
        t = t.filter(m)
        if t.num_rows == 0:
            continue
        arrs = {c: t.column(c) for c in ["agg_level_pseo", "institution", "cipcode", "grad_cohort", "geography",
                                         "industry"]}
        for c in vcols:
            ok = pcm.fill_null(pcm.equal(t.column(f"status_{c}"), 1), False)
            arrs[c] = pcm.if_else(ok, t.column(c), pa.scalar(None, pa.float64()))
        keep.append(pa.table(arrs))
    d = pa.concat_tables(keep).to_pandas()
    d["cipcode"] = d.cipcode.str.replace(".", "", regex=False).str.zfill(2)
    return d


def outstate_y1(path: Path = PSEOF_NEW) -> pd.Series:
    """A6.6(ii): 1 - (y1 graduates employed in the institution's state / y1 employed), institution level, all
    CIP, pooled cohorts, national row, all industries (the scripts/65 formula)."""
    d = flows_rows(path, ["38"], ["0000"], None, ["y1"], instate=True)
    d = d[(d.geography == "00") & (d.industry == "00")]
    d = d.dropna(subset=["y1_grads_emp", "y1_grads_emp_instate"])
    d = d[d.y1_grads_emp > 0]
    s = 1.0 - np.clip(d.y1_grads_emp_instate / d.y1_grads_emp, 0, 1)
    return pd.Series(s.to_numpy(float), index=d.institution.to_numpy())


def ipeds_ba_total() -> pd.Series:
    d = pd.read_csv(IPEDS, usecols=["UNITID", "CIPCODE", "MAJORNUM", "AWLEVEL", "CTOTALT"], dtype=str,
                    encoding="utf-8-sig")
    d = d[(d.CIPCODE == "99") & (d.AWLEVEL == "5") & (d.MAJORNUM == "1")]
    return pd.Series(pd.to_numeric(d.CTOTALT, errors="coerce").to_numpy(float), index=d.UNITID.to_numpy())


def t3_cells(panel: pd.DataFrame, vcol: str, need: list[str], nmin: int, sub=None) -> list:
    cells = []
    for (fld, coh), g in panel.groupby(["field", "grad_cohort"], sort=True):
        x = g.dropna(subset=["G", vcol] + need)
        if sub is not None:
            x = x[sub(x)]
        if len(x) < nmin:
            continue
        x = x.sort_values("inst_key")
        c = dict(field=fld, cohort=coh, n=len(x), keys=list(x.inst_key), G=x.G.to_numpy(float),
                 V=x[vcol].to_numpy(float), Y=x[["e_y1", "e_y5", "e_y10"]].to_numpy(float))
        for v in need:
            c[v] = x[v].to_numpy(float)
        cells.append(c)
    return cells


def t3_cell_stats(c: dict, W, Yperm=None, ctrl: str = "none") -> dict:
    """Per cell under weights W (b, n): the B0 / T3' regression Y ~ G + V_perp + G x V_perp [+ C + G x C]
    [+ S2 controls] per horizon; returns slope, g1 and simple-slope statistics."""
    Y = c["Y"] if Yperm is None else Yperm
    rG = wrank(c["G"], W)
    zG = wstd(rG, W)
    rV = wrank(c["V"], W)
    Vp = wresid(rV[:, :, None], [rG], W)[:, :, 0]
    zVp = wstd(Vp, W)
    GV = zG * zVp
    Yz = np.stack([wstd(wrank(Y[:, h], W), W) for h in range(3)], -1)
    cols = [np.ones(W.shape), zG, zVp, GV]
    if ctrl in ("C", "partial"):
        zC = wstd(np.broadcast_to(c["PRIV"], W.shape).astype(float), W, dummy=True)
        cols += [zC, zG * zC]
    if ctrl == "S2":
        for v in ["SAT", "NADM", "PELL", "ST"]:
            cols.append(wstd(wrank(c[v], W), W))
        cols.append(wstd(np.broadcast_to(c["PRIV"], W.shape).astype(float), W, dummy=True))
    D = np.stack(cols, -1)
    b = wls(D, Yz, W)                                 # (b, k, 3)
    bG, bGV = b[:, 1, :], b[:, 3, :]
    out = dict(theta=bGV @ W_US, g1=bGV[:, 0], dGlow=(bG - bGV) @ W_US, dGhigh=(bG + bGV) @ W_US,
               seg510=(bGV[:, 2] - bGV[:, 1]) / 5.0, bGV_y5=bGV[:, 1], bGV_y10=bGV[:, 2])
    bg = wls(np.stack([np.ones(W.shape), zG], -1), Yz, W)
    out["bench"] = bg[:, 1, :] @ W_US
    if ctrl == "partial":
        others = [zG, zVp] + cols[4:6]
        res = wresid(np.concatenate([GV[:, :, None], Yz], -1), others, W)
        pc = np.stack([wcorr(res[:, :, 0], res[:, :, 1 + h], W) for h in range(3)], -1)
        out["ptheta"] = pc @ W_US
        out["pg1"] = pc[:, 0]
    return out


def t3_field_eval(cells: list, iid: dict, ctrl: str, stats: list):
    by = {}
    for c in cells:
        c["ix"] = np.array([iid[k] for k in c["keys"]])
        by.setdefault(c["field"], []).append(c)

    def ev(f, W):
        acc = {}
        for c in by[f]:
            o = t3_cell_stats(c, W[:, c["ix"]], ctrl=ctrl)
            for k in stats:
                acc.setdefault(k, []).append(o[k])
        return {k: np.mean(v, axis=0) for k, v in acc.items()}
    return ev, sorted(by)


def t3_type_eval(cells_pub: list, cells_priv: list, iid: dict):
    by = {}
    for tag, cl in (("pub", cells_pub), ("priv", cells_priv)):
        for c in cl:
            c["ix"] = np.array([iid[k] for k in c["keys"]])
            by.setdefault(c["field"], {}).setdefault(tag, []).append(c)

    def ev(f, W):
        out = {}
        for tag, cl in by[f].items():
            out[f"theta_{tag}"] = np.mean([t3_cell_stats(c, W[:, c["ix"]])["theta"] for c in cl], axis=0)
        return out
    return ev, sorted(by)


def t3_perm_mde(cells: list, ss, ctrl: str) -> tuple[float, np.ndarray]:
    """A6.4: SD of theta' over 200 permutations of the Y rows (institution trajectories kept) within cells."""
    G = gens(ss, N_PERM_T3)
    vals = []
    for g in G:
        per = {}
        for c in cells:
            p = g.permutation(c["n"])
            o = t3_cell_stats(c, np.ones((1, c["n"])), Yperm=c["Y"][p], ctrl=ctrl)
            per.setdefault(c["field"], []).append(float(o["theta"][0]))
        vals.append(np.mean([np.mean(v) for v in per.values()]))
    vals = np.array(vals)
    return MDE_K * float(vals.std(ddof=1)), vals


def run_T3():
    CUR["test"] = "T3"
    s66, s61 = mod("s66"), mod("s61")
    stage("T3: PSEO V4.14.1 sample A (scripts/66 us_data)")
    ar, gen, er, panel_a, panel_b, cov, start = s66.us_data()
    del panel_b, er
    gc.collect()
    sc = s61.scorecard_institutions()[["institution", "UNITID"]]
    scr = scorecard_names().drop_duplicates("UNITID").set_index("UNITID")
    sc["UGDS"] = sc.UNITID.map(scr.UGDS)
    comp = ipeds_ba_total()
    sc["COMP"] = sc.UNITID.map(comp)
    osh = outstate_y1()
    V = sc.set_index("institution")
    p = panel_a.copy()
    p["V"] = np.log(p.institution_id.map(V.UGDS).where(lambda s: s > 0))
    p["V_comp"] = np.log(p.institution_id.map(V.COMP).where(lambda s: s > 0))
    p["V_out"] = p.institution_id.map(osh)
    put("panel_rows", len(p))
    put("panel_share_with_V", float(p.V.notna().mean()))
    put("panel_share_with_V_comp", float(p.V_comp.notna().mean()))
    put("panel_share_with_V_out", float(p.V_out.notna().mean()))
    pb0 = proc_seeds(SS_B0["T3"], ["main_shared", "main_indep", "main_fields", "comp_shared", "comp_indep",
                                   "comp_fields", "s2_shared", "s2_indep", "s2_fields"])
    pa1 = proc_seeds(SS_A1["T3'"], ["perm", "main_shared", "main_indep", "main_fields", "type_shared",
                                    "type_indep", "type_fields", "out_shared", "out_indep", "out_fields",
                                    "comp_shared", "comp_indep", "comp_fields"])

    def boot(cells, ctrl, stats, pr, tag):
        univ = sorted(set().union(*[set(c["keys"]) for c in cells]))
        iid = {k: i for i, k in enumerate(univ)}
        ev, fl = t3_field_eval(cells, iid, ctrl, stats)
        obs, cb, ib, red = boot_engine(fl, len(univ), ev, pr[0], pr[1], tag=tag)
        return fl, obs, cb, ib, red

    # ---------------- B0 T3 ----------------
    cB0 = t3_cells(p, "V", [], NMIN_T3)
    put("B0_cells", len(cB0))
    put("B0_fields", len({c["field"] for c in cB0}))
    stage("T3: B0 bootstrap")
    fl, obs, cb, ib, red = boot(cB0, "none", ["theta", "bench", "seg510"], (pb0["main_shared"],
                                                                             pb0["main_indep"]), "T3-B0")
    put("B0_redrawn", red)
    s_ = pb0["main_fields"].spawn(3)
    th = infer(mean_of("theta"), obs, cb, ib, fl, s_[0], ["theta"], "B0 theta_GV")
    put("B0_MDE80", th["mde"], "2.80 x two-way SE (reported first)")
    put("B0_theta_GV", th)
    bench_b0 = infer(mean_of("bench"), obs, cb, ib, fl, s_[1], ["bench"], "per-year slope of the G-only loading")
    put("B0_benchmark_dG", bench_b0)
    if th["hi"] < 0:
        v0 = "learning signature"
    elif th["lo"] > 0:
        v0 = "access signature"
    else:
        v0 = "null" if th["mde"] <= T3_B0_NULL else "underpowered"
    put("B0_verdict", v0, "B0 T3 rule (A2.10 interval)")
    put("B0_secondary_iii_y5_y10_segment", infer(mean_of("seg510"), obs, cb, ib, fl, s_[2], ["seg510"],
                                                 "B0 (iii) b_GV (y10 - y5)/5"))
    put("B0_per_field_theta", {f: obs["theta"][f] for f in fl})
    stage("T3: B0 secondary (i) IPEDS completions, (ii) S2 controls")
    cC = t3_cells(p, "V_comp", [], NMIN_T3)
    flc, oc, cbc, ibc, _ = boot(cC, "none", ["theta"], (pb0["comp_shared"], pb0["comp_indep"]), "T3-B0-comp")
    put("B0_secondary_i_ipeds_completions", infer(mean_of("theta"), oc, cbc, ibc, flc, pb0["comp_fields"],
                                                  ["theta"], "B0 (i) V = log IPEDS BA completions"))
    cS = t3_cells(p, "V", ["SAT", "NADM", "PELL", "ST", "PRIV"], NMIN_T3_CTRL)
    put("B0_secondary_ii_cells", len(cS))
    fls, os_, cbs, ibs, _ = boot(cS, "S2", ["theta"], (pb0["s2_shared"], pb0["s2_indep"]), "T3-B0-S2")
    put("B0_secondary_ii_S2_controls", infer(mean_of("theta"), os_, cbs, ibs, fls, pb0["s2_fields"], ["theta"],
                                             "B0 (ii) + SAT, -ADM, Pell, state level, private"))

    # ---------------- A1 T3' ----------------
    c1 = t3_cells(p, "V", ["PRIV"], NMIN_T3)
    put("A1_cells", len(c1))
    stage("T3': permutation MDE (computed before any estimate)")
    mde_p, pv = t3_perm_mde(c1, pa1["perm"], "C")
    put("A1_MDE80_perm", mde_p, "2.80 x SD of theta' over 200 within-cell permutations of the Y rows")
    put("A1_perm_mean", float(pv.mean()))
    stage("T3': bootstrap")
    fl1, o1, c1b, i1b, red1 = boot(c1, "partial", ["theta", "g1", "dGlow", "dGhigh", "bench", "ptheta", "pg1"],
                                   (pa1["main_shared"], pa1["main_indep"]), "T3-A1")
    put("A1_redrawn", red1)
    sf = pa1["main_fields"].spawn(7)
    R = {}
    for j, (k, lab) in enumerate([("theta", "theta'"), ("g1", "g1 (b_GV at y1)"), ("dGlow", "dG at V_perp=-1"),
                                  ("dGhigh", "dG at V_perp=+1"), ("bench", "benchmark: slope of G-only loading"),
                                  ("ptheta", "partial-correlation form: slope"), ("pg1", "partial form at y1")]):
        R[k] = infer(mean_of(k), o1, c1b, i1b, fl1, sf[j], [k], lab)
        put(f"A1_{k}", R[k])
    # within public and within private institutions
    cpub = t3_cells(p, "V", ["PRIV"], NMIN_TYPE, sub=lambda x: x.PRIV == 0)
    cpri = t3_cells(p, "V", ["PRIV"], NMIN_TYPE, sub=lambda x: x.PRIV == 1)
    univ = sorted(set().union(*[set(c["keys"]) for c in cpub + cpri]))
    iid = {k: i for i, k in enumerate(univ)}
    evt, flt = t3_type_eval(cpub, cpri, iid)
    ot, cbt, ibt, _ = boot_engine(flt, len(univ), evt, pa1["type_shared"], pa1["type_indep"], tag="T3-type")
    st_ = pa1["type_fields"].spawn(2)
    for j, tag in enumerate(["pub", "priv"]):
        ff = fields_with(ot, f"theta_{tag}")
        if ff:
            R[tag] = infer(mean_of(f"theta_{tag}"), ot, cbt, ibt, ff, st_[j], [f"theta_{tag}"], f"theta' within {tag}")
        else:
            R[tag] = None
        put(f"A1_theta_{tag}", R[tag] if R[tag] is not None else "undefined: no cell has >= 10 institutions of the type")
    put("A1_type_cells", [len(cpub), len(cpri)])
    put("A1_max_private_per_cell", int(max([int(np.nansum(c["PRIV"])) for c in c1] or [0])),
        "largest number of private institutions in a T3' cell")
    # decision table A6.5
    th1, g1 = R["theta"], R["g1"]
    half_bench = abs(R["bench"]["est"]) / 2.0
    g1s = "pos" if g1["lo"] > 0 else ("neg" if g1["hi"] < 0 else "zero")
    if th1["lo"] <= 0 <= th1["hi"]:
        v1 = "null" if mde_p <= half_bench else "underpowered"
    else:
        ts = "pos" if th1["lo"] > 0 else "neg"
        if R["pub"] is not None and R["priv"] is not None:
            same_sign = np.sign(R["pub"]["est"]) == np.sign(R["priv"]["est"])
        else:
            same_sign = None
        if g1s == "pos" and ts == "neg":
            v1 = ("learning signature" if same_sign else "catch-up not robust to control type") if same_sign is not \
                None else "catch-up at low visibility; the control-type proviso cannot be checked"
        elif g1s == "pos" and ts == "pos":
            v1 = "access or growth at high visibility"
        elif g1s == "neg" and ts == "pos":
            v1 = "halo correction or access"
        elif g1s == "neg" and ts == "neg":
            v1 = "growth concentrated at low visibility"
        elif ts == "neg":
            v1 = "catch-up at low visibility (entry gap not detected)"
        else:
            v1 = "consistent with access or halo correction"
    put("A1_half_benchmark", half_bench)
    put("A1_verdict", v1, "A1 T3' decision table (A6.5); governs claim C4")
    # A6.6 secondary (ii) out-of-state share and (iii) IPEDS completions as V
    stage("T3': secondary visibility measures")
    for vcol, pr, lab in [("V_out", ("out_shared", "out_indep", "out_fields"), "V = out-of-state share at y1"),
                          ("V_comp", ("comp_shared", "comp_indep", "comp_fields"), "V = log IPEDS completions")]:
        cv = t3_cells(p, vcol, ["PRIV"], NMIN_T3)
        flv, ov, cbv, ibv, _ = boot(cv, "C", ["theta", "g1"], (pa1[pr[0]], pa1[pr[1]]), "T3-" + vcol)
        s2 = pa1[pr[2]].spawn(2)
        put(f"A1_secondary_{vcol}_theta", infer(mean_of("theta"), ov, cbv, ibv, flv, s2[0], ["theta"], lab + ": theta'"))
        put(f"A1_secondary_{vcol}_g1", infer(mean_of("g1"), ov, cbv, ibv, flv, s2[1], ["g1"], lab + ": g1"))
    put("A1_per_field", {f: dict(theta=o1["theta"][f], g1=o1["g1"][f]) for f in fl1})
    return RES["T3"]


# =============================================================================================================
# T4 (A1 [V]): UK within-band decay of the school-mean premium
# =============================================================================================================
def leo_band_medians(P: pd.DataFrame) -> pd.DataFrame:
    """LEO provider x CAH2 x prior-attainment band medians (scripts/62 load_leo rules: cohorts 2013-2016, YAG 1/3/5,
    multi-region providers reduced to the region with most graduates in the provider's all-subject row)."""
    s62 = mod("s62")
    keep = ["tax_year", "academic_year", "YAG", "ukprn", "provider_region_name", "cah2_subject_name", "grads",
            "earnings_median", "characteristic_type", "characteristic_value"]
    z = zipfile.ZipFile(s62.LEO_ZIP)
    name = [n for n in z.namelist() if n.startswith("provider_data")][0]
    parts, ptot = [], []
    prov = set(P.ukprn)
    for ch in pd.read_csv(z.open(name), encoding="latin-1", dtype=str, usecols=keep, chunksize=200000):
        ch = ch[ch.ukprn != "Total"]
        coh = ch.academic_year.str[:4].astype(int)
        yag = ch.YAG.astype(int)
        t = (ch.cah2_subject_name == "Total") & (ch.characteristic_type == "All graduates")
        ptot.append(pd.DataFrame({"ukprn": ch.ukprn[t].values, "cohort": coh[t].values, "yag": yag[t].values,
                                  "reg": ch.provider_region_name[t].values,
                                  "grads": pd.to_numeric(ch.grads[t], errors="coerce").values}))
        k = ((ch.characteristic_type == "prior_attainment_code") & (ch.cah2_subject_name != "Total") &
             coh.isin(s62.COHORTS) & yag.isin(s62.YAGS) & ch.ukprn.isin(prov)).values
        x = ch[k]
        grp = x.characteristic_value.str.replace("prior_attainment_", "PA", regex=False).replace({"Not known": "PA_NK"})
        parts.append(pd.DataFrame({"ukprn": x.ukprn.values, "subject": x.cah2_subject_name.values,
                                   "cohort": coh.values[k], "yag": yag.values[k], "grp": grp.values,
                                   "reg": x.provider_region_name.values,
                                   "earnings_median": pd.to_numeric(x.earnings_median, errors="coerce").values}))
    d = pd.concat(parts, ignore_index=True)
    pt = pd.concat(ptot, ignore_index=True)
    pt["g"] = pt.grads.fillna(-1.0)
    main = pt.sort_values(["ukprn", "cohort", "yag", "g", "reg"], ascending=[True, True, True, False, True],
                          kind="mergesort").drop_duplicates(["ukprn", "cohort", "yag"]).set_index(
        ["ukprn", "cohort", "yag"]).reg
    multi = pt.groupby(["ukprn", "cohort", "yag"]).reg.nunique()
    multi = multi[multi > 1].index
    key = pd.MultiIndex.from_frame(d[["ukprn", "cohort", "yag"]])
    drop = key.isin(multi) & (d.reg.values != main.reindex(key).values)
    d = d[~drop].drop(columns="reg")
    assert not d.duplicated(["ukprn", "subject", "cohort", "yag", "grp"]).any()
    return d.pivot_table(index=["subject", "cohort", "yag", "ukprn"], columns="grp", values="earnings_median",
                         aggfunc="first")


def run_T4():
    CUR["test"] = "T4"
    s62 = mod("s62")
    stage("T4: UK LEO band panels (scripts/62 cells_career)")
    P = pd.read_csv(s62.OUT_PRES, dtype={"ukprn": str})
    P = P[P.G.notna()]
    med = leo_band_medians(P)
    Tmin = P[["ukprn", "G"]].copy()
    cells = [c for b in s62.PA_BANDS for c in s62.cells_career(Tmin, med, grp=b)]
    # reproduction of scripts/62's within-band career cells (raw G coupling at YAG 1/3/5)
    ref = pd.read_csv(UK_CAREER)
    ref = ref[ref.analysis == "within_band"].set_index(["subject", "cohort", "grp"])
    dmax, nbad = 0.0, 0
    for c in cells:
        r0 = ref.loc[(c["subject"], c["cohort"], c["grp"])]
        rh = [spear(c["G"], c["Y"][:, k]) for k in range(3)]
        dmax = max(dmax, float(np.max(np.abs(np.array(rh) - r0[["y1", "y3", "y5"]].to_numpy(float)))))
        nbad += int(r0.n != len(c["provs"]))
    put("reproduction_cells", [len(cells), len(ref)], "band cells rebuilt vs scripts/62 within_band rows")
    put("reproduction_max_abs_diff", dmax)
    put("reproduction_n_mismatch", nbad)
    S = P.set_index("ukprn").inst_top
    cl = []
    for c in cells:
        s = S.reindex(c["provs"]).to_numpy(float)
        ok = np.isfinite(s)
        if ok.sum() >= s62.NMIN:
            cl.append(dict(subject=c["subject"], cohort=c["cohort"], band=c["grp"], keys=list(np.array(c["provs"])[ok]),
                           G=np.asarray(c["G"], float)[ok], S=s[ok], Y=np.log(c["Y"][ok])))
    put("cells_with_S", len(cl), f"cells with >= {s62.NMIN} providers having selectivity")
    put("subjects", len({c["subject"] for c in cl}))
    univ = sorted(set().union(*[set(c["keys"]) for c in cl]))
    iid = {k: i for i, k in enumerate(univ)}
    by = {}
    for c in cl:
        c["ix"] = np.array([iid[k] for k in c["keys"]])
        by.setdefault(c["subject"], {}).setdefault(c["cohort"], []).append(c)

    def ev(f, W):
        per_coh = {}
        for coh, cs in by[f].items():
            acc = {}
            for c in cs:
                Wc = W[:, c["ix"]]
                rG, rS = wrank(c["G"], Wc), wrank(c["S"], Wc)
                zG, zS = wstd(rG, Wc), wstd(rS, Wc)
                one = np.ones(Wc.shape)
                pr, ols, rawG, rawS = [], [], [], []
                bo = wls(np.stack([one, zS, zG], -1), np.broadcast_to(c["Y"][None], Wc.shape + (3,)), Wc)
                for h in range(3):
                    rY = wrank(c["Y"][:, h], Wc)
                    res = wresid(np.stack([rS, rY], -1), [rG], Wc)
                    pr.append(wcorr(res[:, :, 0], res[:, :, 1], Wc))
                    rawG.append(wcorr(rG, rY, Wc))
                    rawS.append(wcorr(rS, rY, Wc))
                acc.setdefault("pSel", []).append(np.stack(pr, -1) @ W_UK)
                acc.setdefault("olsSel", []).append(bo[:, 1, :] @ W_UK)
                acc.setdefault("rawG", []).append(np.stack(rawG, -1) @ W_UK)
                acc.setdefault("rawS", []).append(np.stack(rawS, -1) @ W_UK)
                acc.setdefault("pSel_y1", []).append(pr[0])
                acc.setdefault("pSel_y5", []).append(pr[2])
            for k, v in acc.items():
                per_coh.setdefault(k, []).append(np.mean(v, axis=0))
        return {k: np.mean(v, axis=0) for k, v in per_coh.items()}
    # "band_*" appended in revision 8 (exploratory per-band slopes); the first three streams are unchanged
    pa = proc_seeds(SS_A1["T4"], ["shared", "indep", "fields", "band_shared", "band_indep", "band_fields"])
    fl = sorted(by)
    stage("T4: bootstrap (subject, provider)")
    obs, cb, ib, red = boot_engine(fl, len(univ), ev, pa["shared"], pa["indep"], tag="T4")
    put("redrawn", red)
    ss = pa["fields"].spawn(6)
    r = infer(mean_of("pSel"), obs, cb, ib, fl, ss[0], ["pSel"], "slope of partial rho(S, Y | G)")
    put("MDE80", r["mde"], "reported first")
    put("slope_partial_S_given_G", r)
    if r["hi"] < 0:
        v = "decay"
    elif r["lo"] > 0:
        v = "rise"
    else:
        v = "null" if r["mde"] <= 0.02 else "underpowered"
    put("verdict", v, "A1 T4 rule (i); governs claim C5")
    put("slope_ols_logpoints_per_SD", infer(mean_of("olsSel"), obs, cb, ib, fl, ss[1], ["olsSel"],
                                             "(ii) OLS slope, log points per SD of S per year"))
    put("check_slope_raw_G", infer(mean_of("rawG"), obs, cb, ib, fl, ss[2], ["rawG"],
                                   "raw G coupling slope within band (known +0.017, scripts/62)"))
    put("slope_raw_S", infer(mean_of("rawS"), obs, cb, ib, fl, ss[3], ["rawS"], "raw S coupling slope within band"))
    put("level_partial_y1", infer(mean_of("pSel_y1"), obs, cb, ib, fl, ss[4], ["pSel_y1"], "partial rho(S,Y|G) YAG1"))
    put("level_partial_y5", infer(mean_of("pSel_y5"), obs, cb, ib, fl, ss[5], ["pSel_y5"], "partial rho(S,Y|G) YAG5"))
    put("per_subject", {f: dict(pSel=obs["pSel"][f], olsSel=obs["olsSel"][f]) for f in fl})

    # exploratory (not registered; revision 8): the same slopes band by band (PA1 = top band, 4 As or more),
    # averaged over cohorts within subject, then over subjects having the band; two-way intervals, own seeds
    stage("T4: exploratory per-band slopes (not registered)")
    bands = sorted({c["band"] for c in cl})
    ssb = dict(zip(bands, zip(pa["band_shared"].spawn(len(bands)), pa["band_indep"].spawn(len(bands)),
                              pa["band_fields"].spawn(len(bands)))))
    per_band = {}
    for b in bands:
        byb = {}
        for c in cl:
            if c["band"] == b:
                byb.setdefault(c["subject"], {}).setdefault(c["cohort"], []).append(c)

        def evb(f, W, byb=byb):
            per_coh = {}
            for coh, cs in byb[f].items():
                acc = {}
                for c in cs:
                    Wc = W[:, c["ix"]]
                    rG, rS = wrank(c["G"], Wc), wrank(c["S"], Wc)
                    zG, zS = wstd(rG, Wc), wstd(rS, Wc)
                    one = np.ones(Wc.shape)
                    bo = wls(np.stack([one, zS, zG], -1), np.broadcast_to(c["Y"][None], Wc.shape + (3,)), Wc)
                    pr = []
                    for h in range(3):
                        rY = wrank(c["Y"][:, h], Wc)
                        res = wresid(np.stack([rS, rY], -1), [rG], Wc)
                        pr.append(wcorr(res[:, :, 0], res[:, :, 1], Wc))
                    acc.setdefault("pSel", []).append(np.stack(pr, -1) @ W_UK)
                    acc.setdefault("olsSel", []).append(bo[:, 1, :] @ W_UK)
                for k, v in acc.items():
                    per_coh.setdefault(k, []).append(np.mean(v, axis=0))
            return {k: np.mean(v, axis=0) for k, v in per_coh.items()}
        flb = sorted(byb)
        s_sh, s_in, s_fl = ssb[b]
        obs_b, cb_b, ib_b, red_b = boot_engine(flb, len(univ), evb, s_sh, s_in, tag=f"T4{b}")
        sfl = s_fl.spawn(2)
        rb = infer(mean_of("pSel"), obs_b, cb_b, ib_b, flb, sfl[0], ["pSel"], f"{b}: slope of partial rho(S, Y | G)")
        ro = infer(mean_of("olsSel"), obs_b, cb_b, ib_b, flb, sfl[1], ["olsSel"], f"{b}: OLS slope, log points per SD")
        per_band[b] = dict(subjects=len(flb), cells=int(sum(len(v) for d in byb.values() for v in d.values())),
                           slope_partial=rb, slope_ols=ro, redrawn=red_b,
                           subject_names=flb if len(flb) <= 3 else None)
        put(f"band_{b}", per_band[b], "exploratory, not registered")
    return RES["T4"]


# =============================================================================================================
# ACS 2023 PUMS extract (T6-T8): full-time full-year wage and salary employees (COW 1-5, WKHP >= 35, WKWN >= 50,
# WAGP > 0, as scripts/61 acs_sectors) with a bachelor's degree or higher (SCHL >= 21), aged 22-40 (subset later)
# =============================================================================================================
STATE_DIV = {1: [9, 23, 25, 33, 44, 50], 2: [34, 36, 42], 3: [17, 18, 26, 39, 55], 4: [19, 20, 27, 29, 31, 38, 46],
             5: [10, 11, 12, 13, 24, 37, 45, 51, 54], 6: [1, 21, 28, 47], 7: [5, 22, 40, 48],
             8: [4, 8, 16, 30, 32, 35, 49, 56], 9: [2, 6, 15, 41, 53]}
FIPS_DIV = {s: d for d, ss in STATE_DIV.items() for s in ss}
DIVS = [str(d) for d in range(1, 10)]
_ACS: dict = {}


def acs_extract() -> pd.DataFrame:
    if "d" in _ACS:
        return _ACS["d"]
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.compute as pcm
    num = ["PWGTP", "AGEP", "COW", "SCHL", "WAGP", "WKHP", "WKWN", "POWSP"]
    types = {**{c: pa.float64() for c in num}, "NAICSP": pa.string(), "SOCP": pa.string()}
    parts = []
    for p in ACS:
        co = pacsv.ConvertOptions(include_columns=num + ["NAICSP", "SOCP"], column_types=types)
        rd = pacsv.open_csv(p, convert_options=co, read_options=pacsv.ReadOptions(block_size=1 << 23, use_threads=False))
        for b in rd:
            t = pa.Table.from_batches([b])
            c = {k: t.column(k) for k in num}
            m = pcm.and_(pcm.and_(pcm.and_(pcm.greater_equal(c["AGEP"], 22), pcm.less_equal(c["AGEP"], 40)),
                                  pcm.and_(pcm.greater_equal(c["SCHL"], 21),
                                           pcm.and_(pcm.greater_equal(c["COW"], 1), pcm.less_equal(c["COW"], 5)))),
                         pcm.and_(pcm.greater_equal(c["WKHP"], 35),
                                  pcm.and_(pcm.greater_equal(c["WKWN"], 50), pcm.greater(c["WAGP"], 0))))
            parts.append(t.filter(pcm.fill_null(m, False)).to_pandas())
            del t, b
    d = pd.concat(parts, ignore_index=True)
    d["div"] = d.POWSP.map(lambda v: FIPS_DIV.get(int(v)) if np.isfinite(v) else None)
    _ACS["d"] = d
    return d


def wmedian(x, w) -> float:
    """Weighted median as scripts/61 acs_sectors computes it."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x, kind="mergesort")
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, cw[-1] / 2)])


def acs_div_logmed(mask_fn=None, age=(23, 35), schl_min=21) -> pd.Series:
    d = acs_extract()
    x = d[d.AGEP.between(*age) & (d.SCHL >= schl_min) & d["div"].notna()]
    if mask_fn is not None:
        x = x[mask_fn(x)]
    out = {}
    for dv, g in x.groupby("div"):
        out[str(int(dv))] = np.log(wmedian(g.WAGP, g.PWGTP))
    return pd.Series(out).reindex(DIVS), x.groupby("div").size()


def div_shares(fl: pd.DataFrame, h: str, keys: list[str]) -> pd.DataFrame:
    """Division shares of employed graduates per key (rows with all nine divisions released)."""
    x = fl[fl.geography.isin(DIVS)].pivot_table(index=keys, columns="geography", values=f"{h}_grads_emp",
                                                  aggfunc="first").reindex(columns=DIVS)
    x = x.dropna()
    tot = x.sum(1)
    x = x[tot > 0]
    return x.div(x.sum(1), axis=0)


# =============================================================================================================
# T6 (A1 [V]): quantities where pay is set by schedule
# =============================================================================================================
T6_FIELDS = {"education": ("13", lambda x: x.SOCP.fillna("").str.startswith("2520"), "61"),
             "nursing": ("51", lambda x: x.SOCP.fillna("") == "291141", "62")}


def run_T6():
    CUR["test"] = "T6"
    s28, s61 = mod("s28"), mod("s61")
    stage("T6: PSEO Flows V4.14.1 division rows (CIP-2 13 and 51, pooled, y5) and ACS")
    fl = flows_rows(PSEOF_NEW, ["136", "40", "88"], ["0000"], ["13", "51"], ["y5"])
    put("note_cip", "Flows are released at CIP-2 only (cip_level 2 or A); CIP 51.38 is never released, so CIP-2 51 "
        "is used for registered nursing, as registered")
    ins = s61.pseo_institutions(s61.PSEO_INST_NEW)
    gen = s28.load_generic()
    gen["G"] = -gen.g_rank.astype(float)
    gmap = gen.drop_duplicates("inst_key").set_index("inst_key").G
    _i = pd.read_csv(s61.PSEO_INST_NEW, dtype=str)
    _i.columns = [c.strip().lstrip("\ufeff") for c in _i.columns]
    fips_of = _i.drop_duplicates("institution").set_index("institution").statefips
    earn_n, c4n, c2nn = s61.read_earnings(s61.PSEOE_NEW)
    prest, _ = s61.family_prestige(c4n, c2nn, ins)
    del earn_n
    rows = {}
    for fname, (cip, socm, naics) in T6_FIELDS.items():
        lm, nobs = acs_div_logmed(socm)
        lmBA, _ = acs_div_logmed(lambda x, s=socm: s(x) & (x.SCHL == 21))
        put(f"{fname}_acs_log_median_by_division", lm.to_dict(), f"n obs by division {nobs.to_dict()}")
        put(f"{fname}_distinct_division_values", int(lm.round(10).nunique()),
            "of 9 divisions (ACS medians are often round numbers, so divisions tie); nominal wages, no regional "
            "price adjustment")
        sh = div_shares(fl[(fl.agg_level_pseo == "136") & (fl.cipcode == cip)], "y5", ["institution"])
        T = pd.Series(sh.to_numpy() @ lm.to_numpy(), index=sh.index)
        Tba = pd.Series(sh.to_numpy() @ lmBA.to_numpy(), index=sh.index)
        sec = fl[(fl.agg_level_pseo == "88") & (fl.cipcode == cip)].pivot_table(
            index="institution", columns="industry", values="y5_grads_emp", aggfunc="first")
        out_share = 1.0 - sec[naics] / sec.sum(1)
        d = pd.DataFrame({"T": T, "T_BAonly": Tba}).join(ins.set_index("institution")[["inst_key"]], how="left")
        # exploratory: the index value if every graduate worked in the institution's own census division
        stfips = pd.to_numeric(pd.Series(np.asarray(d.index.map(fips_of), dtype=object), index=d.index),
                               errors="coerce")
        d["T_home"] = stfips.map(lambda v: lm.get(str(FIPS_DIV.get(int(v)))) if np.isfinite(v) else np.nan)
        d["G"] = d.inst_key.map(gmap)
        pp = prest[prest.cip2 == cip].set_index("institution").P
        d["P"] = d.index.map(pp)
        d["out"] = d.index.map(out_share)
        d = d.dropna(subset=["G"]).sort_index()
        rows[fname] = d
        put(f"{fname}_n_institutions", len(d), "institutions with released division rows and G")
    ok = [f for f in T6_FIELDS if len(rows[f]) >= 15]
    if len(ok) < 2:
        put("verdict", "infeasible")
        return RES["T6"]
    univ = sorted(set().union(*[set(rows[f].index) for f in ok]))
    iid = {k: i for i, k in enumerate(univ)}
    pa = proc_seeds(SS_A1["T6"], ["boot"])
    G = gens(pa["boot"], B)
    NI = len(univ)
    W = np.stack([_mn(g, NI) for g in G])

    def stat(Wx, col_x, col_y):
        vals = []
        for f in ok:
            d = rows[f].dropna(subset=[col_x, col_y])
            ix = np.array([iid[k] for k in d.index])
            vals.append(wspear(d[col_x].to_numpy(float), d[col_y].to_numpy(float), Wx[:, ix]))
        return np.mean(vals, axis=0), [float(v[0]) if v.shape[0] == 1 else np.nan for v in vals], vals
    # exploratory (not registered): how much of T is the institution's own division, and G's association with T
    # net of it (partial Spearman given T_home)
    ex = {}
    for f in ok:
        d = rows[f].dropna(subset=["G", "T", "T_home"])
        ex[f] = dict(n=len(d), rho_T_Thome=spear(d["T"], d["T_home"]), rho_G_Thome=spear(d["G"], d["T_home"]),
                     partial_G_T_given_Thome=partial_spear(d.G.to_numpy(float), d["T"].to_numpy(float),
                                                           [d.T_home.to_numpy(float)]))
    put("exploratory_home_division", ex, "not registered: T_home = index if all graduates worked in the "
        "institution's own division")
    put("exploratory_mean_partial_G_T_given_Thome", float(np.mean([v["partial_G_T_given_Thome"] for v in ex.values()])))
    for lab, cx, cy in [("primary", "G", "T"), ("P_in_place_of_G", "P", "T"), ("G_outside_scheduled_industry", "G", "out"),
                        ("G_T_bachelors_only_wages", "G", "T_BAonly")]:
        est, per, _ = stat(np.ones((1, NI)), cx, cy)
        bs, _, bsf = stat(W, cx, cy)
        # per-field estimates with percentile intervals from the same institution draws (revision 7)
        pf = {}
        for f, e_, b_ in zip(ok, per, bsf):
            b_ = b_[np.isfinite(b_)]
            pf[f] = dict(est=e_, ci=[float(np.percentile(b_, 2.5)), float(np.percentile(b_, 97.5))],
                         n=int(len(rows[f].dropna(subset=[cx, cy]))))
        bad = ~np.isfinite(bs)
        if bad.any():
            put(f"{lab}_nonfinite_replicates", int(bad.sum()))
            bs = bs[~bad]
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        if lab == "primary":
            put("MDE80", MDE_K * float(bs.std(ddof=1)), "2.80 x institution-bootstrap SE (reported first)")
        put(f"{lab}_mean_spearman", float(est[0]), f"per field {dict(zip(ok, per))}")
        put(f"{lab}_ci", [lo, hi], "institution bootstrap, B=1000, percentile")
        if lab == "primary":
            v = "supported" if lo > 0 else ("contradicted" if hi < 0.05 else "inconclusive")
            put("verdict", v, "A1 T6; governs claim C7")
        put(f"{lab}_per_field", pf, "institution bootstrap, same draws as the mean (revision 7)")
    return RES["T6"]


# =============================================================================================================
# T7 (A1 [V]): destination location (scripts/52 fixed-cohort panel, V4.13.0, and the same release's Flows)
# =============================================================================================================
def run_T7():
    CUR["test"] = "T7"
    s59, s28 = mod("s59"), mod("s28")
    s52 = mod("s52")
    from src.load_ar import load_ar_wapman
    s59.set_release("old")                       # scripts/52's own release (V4.13.0); its code is unchanged
    stage("T7: scripts/52 fixed-cohort panel (V4.13.0)")
    ar = load_ar_wapman(fields=s52.FIELDS66)
    gen = s28.load_generic()
    gen = gen.assign(G=-gen["g_rank"].astype(float))[["inst_key", "G"]]
    er = {h: s52.load_fixed(h, s52.COHORTS) for h in s52.HZ}
    panel = s52.fixed_panel(er, ar, gen)
    ids = pd.concat([er[h][["inst_key", "institution_id"]] for h in s52.HZ]).drop_duplicates()
    assert ids.inst_key.is_unique, "inst_key -> institution id not unique"
    panel["institution"] = panel.inst_key.map(ids.set_index("inst_key").institution_id)
    panel["cip2"] = panel.field.map(s28.CIP2)
    put("panel_rows", len(panel))
    stage("T7: Flows V4.13.0 division rows by cohort (agg 142) and in-state counts (agg 46)")
    fl = flows_rows(PSEOF_OLD, ["142", "46"], s52.COHORTS, None, s52.HZ, instate=True)
    lm, nobs = acs_div_logmed()
    put("acs_L_d", lm.to_dict(), f"log median wage, FTFY BA+ aged 23-35, by division of work; n {nobs.to_dict()}")
    for h in s52.HZ:
        sh = div_shares(fl[fl.agg_level_pseo == "142"], h, ["institution", "cipcode", "grad_cohort"])
        Wh = pd.Series(sh.to_numpy() @ lm.to_numpy(), index=sh.index)
        nat = fl[(fl.agg_level_pseo == "46") & (fl.geography == "00")].set_index(
            ["institution", "cipcode", "grad_cohort"])
        e_, i_ = nat[f"{h}_grads_emp"], nat[f"{h}_grads_emp_instate"]
        Oh = (1.0 - np.clip(i_ / e_.where(e_ > 0), 0, 1)).dropna()
        k = pd.MultiIndex.from_frame(panel[["institution", "cip2", "grad_cohort"]])
        panel[f"W_{h}"] = Wh.reindex(k).to_numpy()
        panel[f"O_{h}"] = Oh.reindex(k).to_numpy()
        put(f"share_with_W_{h}", float(panel[f"W_{h}"].notna().mean()))
        put(f"share_with_O_{h}", float(panel[f"O_{h}"].notna().mean()))
    panel = panel[panel.P.notna() & panel.G.notna()] if "G" in panel else panel
    pa = proc_seeds(SS_A1["T7"], ["i_shared", "i_indep", "i_fields", "ii_shared", "ii_indep", "ii_fields"])

    def mk_cells(need, nmin=15):
        cells = []
        for (f, c), g in panel.groupby(["field", "grad_cohort"], sort=True):
            x = g.dropna(subset=need).sort_values("inst_key")
            if len(x) >= nmin:
                cells.append(dict(field=f, cohort=c, keys=list(x.inst_key), F=x.P.to_numpy(float),
                                  Y=x[["e_y1", "e_y5", "e_y10"]].to_numpy(float),
                                  W=x[["W_y1", "W_y5", "W_y10"]].to_numpy(float),
                                  O=x[["O_y1", "O_y5", "O_y10"]].to_numpy(float)))
        return cells

    def run(cells, kind, pr, labs):
        univ = sorted(set().union(*[set(c["keys"]) for c in cells]))
        iid = {k: i for i, k in enumerate(univ)}
        by = {}
        for c in cells:
            c["ix"] = np.array([iid[k] for k in c["keys"]])
            by.setdefault(c["field"], []).append(c)

        def ev(f, W):
            acc = {}
            for c in by[f]:
                Wc = W[:, c["ix"]]
                if kind == "y5":
                    o = dict(raw=wspear(c["F"], c["Y"][:, 1], Wc),
                             pW=wpartial(c["F"], c["Y"][:, 1], [c["W"][:, 1]], Wc),
                             pWO=wpartial(c["F"], c["Y"][:, 1], [c["W"][:, 1], c["O"][:, 1]], Wc))
                else:
                    raw = np.stack([wspear(c["F"], c["Y"][:, h], Wc) for h in range(3)], -1)
                    pW = np.stack([wpartial(c["F"], c["Y"][:, h], [c["W"][:, h]], Wc) for h in range(3)], -1)
                    pWO = np.stack([wpartial(c["F"], c["Y"][:, h], [c["W"][:, h], c["O"][:, h]], Wc)
                                    for h in range(3)], -1)
                    o = dict(raw=raw @ W_US, pW=pW @ W_US, pWO=pWO @ W_US)
                for k, v in o.items():
                    acc.setdefault(k, []).append(v)
            return {k: np.mean(v, axis=0) for k, v in acc.items()}
        fl_ = sorted(by)
        obs, cb, ib, red = boot_engine(fl_, len(univ), ev, pa[pr[0]], pa[pr[1]], tag="T7" + kind)
        put(f"{kind}_redrawn", red)
        ss = pa[pr[2]].spawn(5)
        out = {}
        out["raw"] = infer(mean_of("raw"), obs, cb, ib, fl_, ss[0], ["raw"], labs[0])
        out["pW"] = infer(mean_of("pW"), obs, cb, ib, fl_, ss[1], ["pW"], labs[1])
        out["R_W"] = infer(ratio_of("pW", "raw"), obs, cb, ib, fl_, ss[2], ["pW", "raw"], labs[2])
        out["pWO"] = infer(mean_of("pWO"), obs, cb, ib, fl_, ss[3], ["pWO"], labs[3])
        out["R_WO"] = infer(ratio_of("pWO", "raw"), obs, cb, ib, fl_, ss[4], ["pWO", "raw"], labs[4])
        return out
    c5 = mk_cells(["e_y5", "W_y5"])
    put("i_cells", len(c5))
    put("i_fields", len({c["field"] for c in c5}))
    stage("T7 (i): y5 cells")
    o5 = run(c5, "y5", ("i_shared", "i_indep", "i_fields"),
             ["raw rho(F,Y) at y5", "partial rho(F,Y|W) at y5", "R_L = mean partial / mean raw",
              "partial rho(F,Y|W,O) at y5", "R_L with W and O"])
    put("i_MDE80_R_L", o5["R_W"]["mde"], "reported first")
    for k, v in o5.items():
        put(f"i_{k}", v)
    lo, hi = o5["R_W"]["lo"], o5["R_W"]["hi"]
    v = "location-robust" if lo >= 0.75 else ("location-dominated" if hi < 0.50 else "partly location")
    put("verdict", v, "A1 T7 rule (i); governs claim C8")
    # (ii), (iii): slopes over y1/y5/y10 on cells with W and O at every horizon (O needed for (iii) only)
    cs = mk_cells(["W_y1", "W_y5", "W_y10", "O_y1", "O_y5", "O_y10"])
    put("ii_cells", len(cs))
    stage("T7 (ii)/(iii): career slopes")
    osl = run(cs, "slope", ("ii_shared", "ii_indep", "ii_fields"),
              ["raw slope", "slope of partial | W_h", "slope ratio partial|W / raw", "slope of partial | W_h, O_h",
               "slope ratio partial|W,O / raw"])
    for k, v in osl.items():
        put(f"ii_{k}", v)
    put("ii_share_removed_W", 1.0 - osl["R_W"]["est"])
    put("iii_share_removed_WO", 1.0 - osl["R_WO"]["est"])
    return RES["T7"]


# =============================================================================================================
# T8 (A1 [V]): between- and within-sector coupling (registered decomposition, no verdict)
# =============================================================================================================
def run_T8():
    CUR["test"] = "T8"
    s61 = mod("s61")
    stage("T8: scripts/61 W cells (V4.13.0, pooled, y5)")
    old_coh = s61.COHORTS
    s61.COHORTS = ["0000"]                        # memory: only the pooled cohort's rows are needed here
    try:
        sec, tot = s61.read_flows()
        earn, c4, c2n = s61.read_earnings()
    finally:
        s61.COHORTS = old_coh
    ins = s61.pseo_institutions()
    sc = s61.scorecard_institutions()
    brand = s61.generic_brand()
    prest, nat_cover = s61.family_prestige(c4, c2n, ins)
    fams = sorted(c for c, v in nat_cover.items() if v >= s61.FAM_COVER)
    cells = s61.build_cells(sec, tot, earn, prest, ins, sc, brand, "0000", "y5", fams)
    del sec, tot, earn
    gc.collect()
    W = s61.samp_W(cells)
    put("W_cells", len(W))
    put("W_families", int(W.cip2.nunique()))
    put("W_institutions", int(W.institution.nunique()))
    d = acs_extract()
    x = d[d.AGEP.between(23, 35) & d.NAICSP.notna()].copy()
    n_ = x.NAICSP.str.strip()
    x = x[~n_.str.startswith("92811") & (n_ != "999920")]
    two = x.NAICSP.str.strip().str[:2]
    x["sector"] = two.replace({"31": "31-33", "32": "31-33", "33": "31-33", "3M": "31-33", "44": "44-45",
                               "45": "44-45", "4M": "44-45", "48": "48-49", "49": "48-49"})
    x = x[x.sector.isin(s61.SECTORS)]
    med = pd.Series({k: wmedian(g.WAGP, g.PWGTP) for k, g in x.groupby("sector")}).reindex(s61.SECTORS)
    put("acs_sector_median_wage_23_35", med.to_dict())
    Wc = W.copy()
    s61.add_wage_mix(Wc, med)
    Wc = Wc.rename(columns={"C": "C2335"})
    acs61 = s61.acs_sectors()
    med61 = acs61[acs61.sector.isin(s61.SECTORS)].set_index("sector").med_wage.reindex(s61.SECTORS)
    s61.add_wage_mix(Wc, med61)
    Wc = Wc.rename(columns={"C": "C2240"})
    Wc = Wc.dropna(subset=["P", "log_earn", "C2335", "C2240"]).sort_values(["cip2", "institution"])
    univ = sorted(Wc.institution.unique())
    iid = {k: i for i, k in enumerate(univ)}
    by = {}
    for f, g in Wc.groupby("cip2"):
        if len(g) >= s61.NMIN:
            by[f] = dict(ix=np.array([iid[k] for k in g.institution]), P=g.P.to_numpy(float),
                         Y=g.log_earn.to_numpy(float), C=g.C2335.to_numpy(float), C61=g.C2240.to_numpy(float))

    def ev(f, Wt):
        a = by[f]
        Wx = Wt[:, a["ix"]]
        return dict(raw=wspear(a["P"], a["Y"], Wx), within=wpartial(a["P"], a["Y"], [a["C"]], Wx),
                    within61=wpartial(a["P"], a["Y"], [a["C61"]], Wx))
    pa = proc_seeds(SS_A1["T8"], ["shared", "indep", "fields"])
    fl = sorted(by)
    stage("T8: bootstrap (family, institution)")
    obs, cb, ib, red = boot_engine(fl, len(univ), ev, pa["shared"], pa["indep"], tag="T8")
    put("redrawn", red)
    ss = pa["fields"].spawn(5)
    put("raw", infer(mean_of("raw"), obs, cb, ib, fl, ss[0], ["raw"], "raw rho(P, log p50)"))
    put("within", infer(mean_of("within"), obs, cb, ib, fl, ss[1], ["within"], "partial rho(P, log p50 | C)"))
    Bs = infer(lambda dd: 1.0 - np.mean(dd["within"], 0) / np.mean(dd["raw"], 0), obs, cb, ib, fl, ss[2],
               ["within", "raw"], "between-sector share B = 1 - within/raw")
    put("MDE80_B", Bs["mde"])
    put("B_share", Bs, "registered decomposition, no verdict")
    put("within_C_scripts61_ages22_40", infer(mean_of("within61"), obs, cb, ib, fl, ss[3], ["within61"],
                                              "partial | C with scripts/61's own C (ages 22-40)"))
    put("B_share_C_scripts61", infer(lambda dd: 1.0 - np.mean(dd["within61"], 0) / np.mean(dd["raw"], 0), obs, cb,
                                     ib, fl, ss[4], ["within61", "raw"], "B with scripts/61's C"))
    put("per_family", {f: dict(raw=obs["raw"][f], within=obs["within"][f]) for f in fl})
    return RES["T8"]


# =============================================================================================================
# report
# =============================================================================================================
CLAIM_MAP = {
    "T0": ("C1", "F orders departments by the direction of exchange, not by production volume",
           {"production-robust": "kept", "partly production": "written as 'partly production' (the same reading under "
            "all three zero-degree completions); an upper-bound statement, since R_perp also removes net export "
            "driven by valuation",
            "production-dominated": "F renamed 'placement position'"}),
    "T1": ("C2", "F behaves as one index of standing, as producer and as employer",
           {"one index": "kept", "not one index": "withdrawn: F described as the ordering revealed by net placement; "
            "only the reading of F as a belief about the quality of a department's PhDs is dropped",
            "inconclusive": "undecided"}),
    "T2": ("C3", "the academy's own choice among the same bachelor's graduates tracks the department-specific part "
           "of F", {"supported": "kept, as a consistency check with the rivals of A5.1 named",
                    "contradicted": "withdrawn", "inconclusive": "undecided"}),
    "T3": ("C4", "the source of the career rise",
           {"learning signature": "Proposition 4's baseline rewritten for low-visibility institutions",
            "null": "both readings left open", "underpowered": "both readings left open"}),
    "T4": ("C5", "employers price a school-mean premium beyond graduates' own attainment and learn it away",
           {"decay": "supported", "rise": "the within-band premium does not decay on average over bands (per-band "
            "slopes reported as exploratory); not described as reputation that learning corrects", "null": "the within-band premium does not decay; not described as "
            "reputation that learning corrects", "underpowered": "undecided (no registered consequence)"}),
    "T5": ("C6", "the field map is a map of how each field prices institution-wide status, whatever the index",
           {"supported": "kept", "contradicted": "withdrawn: F's construct identity matters at the level of the map",
            "inconclusive": "undecided"}),
    "T6": ("C7", "where pay is set by schedule, status acts on quantities",
           {"supported": "kept, with two qualifiers: T largely records where the institution is, and it uses nominal "
            "division wages (no price adjustment); per-field estimates reported",
            "contradicted": "the quantity clause of Proposition 5 is withdrawn",
            "inconclusive": "undecided"}),
    "T7": ("C8", "pay construct claims net of destination location",
           {"location-robust": "y5 coupling unchanged net of destination-division wages; one fifth to one third of "
            "the career rise moves with destination wages and out-of-state share; 'not netted of location within "
            "destination divisions' kept",
            "partly location": "construct claims about pay are qualified as partly location",
            "location-dominated": "construct claims restated as agreement between the academy's ordering and "
            "where graduates work"}),
    "T8": ("C9", "between-sector share reported next to every construct statement about pay", {}),
}


def g(res, t, k, default=None):
    return res.get(t, {}).get(k, default)


def iv(r, d=3):
    """'est [lo, hi]' for a two-way inference dict."""
    if isinstance(r, str):
        return r
    if not isinstance(r, dict):
        return "n/a"
    return f"{fmt(r['est'], d)} [{fmt(r['lo'], d)}, {fmt(r['hi'], d)}]"


def ivx(r, d=3):
    if not isinstance(r, dict):
        return "n/a"
    return f"{fmt(r['est'], d)} [{fmt(r['xlo'], d)}, {fmt(r['xhi'], d)}]"


def write_report(res: dict, meta: dict):
    L = []
    A = L.append
    A("# Registered theory tests T0–T8 (binding texts B0 and amendment A1)\n")
    A("Script: `scripts/75_theory_tests.py` (seeded: numpy SeedSequence(75) for B0, SeedSequence(751) for A1, one "
      "child per test, procedure and replicate; byte-identical on re-run). Every statistic below is the one named in "
      "the binding texts B0 and A1, stored verbatim in the script's docstring and in THEORY_NOTES.md §9. Public data "
      "only. Descriptive, not causal: every coupling here is a rank association across institutions.\n")
    A("## Pre-registration status (read first)\n")
    A(f"- **Digests.** B0 SHA-256 `{PRESPEC_SHA256}` and A1 SHA-256 `{A1_SHA256}` were checked at run time: "
      f"B0 {'matches' if meta['b0_ok'] else 'DOES NOT MATCH'}, A1 {'matches' if meta['a1_ok'] else 'DOES NOT MATCH'}.")
    A("- **No public time-stamped deposit before the run.** The A1 preamble asks the author to commit "
      "`scripts/75_theory_tests.py` (or deposit it on OSF/Zenodo) before any test is run and to record the commit "
      "hash or DOI here. This workflow is not allowed to commit or push, so the tests were run without that deposit. "
      f"What can be checked: the pre-implementation stub (binding texts only, no test code) had SHA-256 "
      f"`{STUB_SHA256}`; the script at run time had SHA-256 `{meta['script_sha']}`. Neither digest establishes when "
      "the text was written. Commit hash or DOI: **none (to be filled by the author)**. Readers should treat the "
      "registration as self-attested. The stub itself is kept, unchanged, at "
      f"`{STUB_KEEP}` (local and git-ignored; its digest is checked when this report is written: "
      f"{meta['stub_check']}), so the author can deposit it with the digests. Every summary of these results, and "
      "the paper, should say: \"self-attested registration, not publicly deposited before the run\".")
    A("- **The theory was written after most results were seen**, and the tests share data with results already "
      "seen (A0.6): the pay side of T2 (Delta^Y) was known for all seven fields, T3 was chosen knowing the G loading "
      "rises, T4 knowing the within-band raw G rise (+0.017/yr), T5 knowing the selectivity analogue (+0.94), and "
      "T6–T8 use PSEO Flows data analysed in scripts/61.")
    A(f"- Run: {meta['date']}; tests run: {', '.join(meta['tests'])}.\n")

    # ---------------- answer ----------------
    A("## Answer\n")
    A("MDE80 (2.80 x SE of the primary statistic) is given before each estimate, as G3 and A0.4 require. Intervals are "
      "95%. \"Two-way\" = scripts/59 twoway() (field and institution clusters, normal interval); \"crossed\" = fields "
      "resampled together with a shared institution draw (percentile).\n")
    items = []
    t0 = res.get("T0", {})
    if t0:
        items.append(
            f"**T0 (production audit of F): {t0.get('b_verdict')}.** MDE80 of R_DC {fmt(t0.get('b_MDE80_R_DC'), 3, False)}. "
            f"Replacing F by a degree-corrected Bradley–Terry rank keeps R_DC = {iv(t0.get('b_R_DC'))} of mean "
            f"coupling; residualising F on log PhD production and the export ratio keeps R_perp = "
            f"{iv(t0.get('b_R_perp'))} ({g(res, 'T0', 'b_fields')} fields, {g(res, 'T0', 'b_cells_total')} cells). "
            f"Across fields F correlates {fmt(t0.get('a_mean_rho_F_p'))} with log production, "
            f"{fmt(t0.get('a_mean_rho_F_x'))} with the export ratio, {fmt(t0.get('a_mean_rho_F_q'))} with the "
            f"placement rate and {fmt(t0.get('a_mean_rho_F_negPR'))} with -ProductionRank; Spearman(F, F_DC) averages "
            f"{fmt(t0.get('b_mean_rho_F_FDC_network'))} and Spearman(F, MVR) {fmt(t0.get('c_mean_rho_F_MVR'))}. "
            f"Mean (c_GDC - c_G) = {iv(t0.get('b_GDC_minus_G'))}. The registered offset is undefined at zero degrees "
            f"({fmt(t0.get('b_exact_limit_unidentified_node_share'), 2, False)} of network nodes have one); all three "
            f"ways of completing it give the same reading under the rule: +1 on every degree (primary, A3's "
            f"convention for x) \"{t0.get('b_verdict')}\"; the exact limit with theta = 0 (the middle) for the "
            f"unidentified nodes R_DC = {iv(t0.get('b_sens_R_DC_exact_limit'))}, \"{t0.get('b_sens_s0_rule_reading')}\" "
            f"(lower, because it puts those nodes in the middle); the exact limit on identified cells only R_DC = "
            f"{iv(t0.get('b_sens_R_DC_exact_limit_identified_only'))} and R_perp = "
            f"{iv(t0.get('b_sens_R_perp_identified_only'))} ({t0.get('b_sens_s1_fields')} fields), "
            f"\"{t0.get('b_sens_s1_rule_reading')}\" (Deviations, item 4; the first run's exact-limit values came "
            f"from a Newton solve that had not converged). In the primary fit the lower bound of R_DC "
            f"({fmt(g_(t0, 'b_R_DC', 'lo'))}) sits just below 0.75, so the reading rests mainly on R_perp, and R_perp "
            f"also removes the net export that valuation itself produces (H1): \"partly production\" bounds the "
            f"production share from above. F_DC (+1) still carries the export ratio (mean Spearman with x on the common "
            f"cells {fmt(t0.get('b_diag_mean_rho_FDC_x_cells'))} against {fmt(t0.get('b_diag_mean_rho_F_x_cells'))} "
            f"for F; exploratory), so neither statistic isolates production cleanly (Caveats).")
    t1 = res.get("T1", {})
    if t1:
        items.append(
            f"**T1 (one index?): A1 T1(a') {t1.get('ap_verdict')} (post-censoring reading of A4.1; the "
            f"literal-order reading is degenerate); B0 T1 {t1.get('B0_T1_verdict')}.** "
            f"Cross-fitted R' = {fmt(t1.get('ap_Rprime'))} {fmt_ci(t1.get('ap_post_Rprime_ci'))} ({t1.get('ap_k')} "
            f"fields). A4.1's \"c matching the field's total count\" does not say whether the simulated total is "
            f"matched before or after the A* >= 2 censoring. Matched after it (the public total is a censored total; "
            f"the only non-degenerate benchmark): MDE80 {fmt(t1.get('ap_post_MDE80'), 3, False)}, R0 = "
            f"{fmt(t1.get('ap_post_R0'))}, Delta_R = {fmt(t1.get('ap_post_DeltaR'))} "
            f"{fmt_ci(t1.get('ap_post_DeltaR_ci'))}, \"{t1.get('ap_post_rule')}\" (the upper bound is close to "
            f"-0.10). Matched before it (literal sentence order): the simulated networks keep "
            f"{fmt(t1.get('ap_sim_kept_person_share'), 3, False)} of their persons and "
            f"{fmt(t1.get('ap_sim_field_excluded_share'), 3, False)} of field x network cells fail B0's rules; R0 = "
            f"{fmt(t1.get('ap_R0'))} comes from {t1.get('ap_sim_cells_used')} of "
            f"{int(t1.get('ap_k') or 0) * N_SIM} cells ({t1.get('ap_sim_fields_with_any_cell')} field(s): "
            f"{', '.join(t1.get('ap_sim_fields_with_any_cell_names') or [])}), MDE80 {fmt(t1.get('ap_MDE80'), 3, False)}, "
            f"Delta_R = "
            f"{fmt(t1.get('ap_DeltaR'))} {fmt_ci(t1.get('ap_DeltaR_ci'))}, \"{t1.get('ap_rule')}\" (degenerate). "
            f"Claim C2 follows the post-censoring reading, which was adopted after the literal-order benchmark had "
            f"been seen to degenerate: F is described as the net-export position in the hiring network, not as one "
            f"belief about the quality of a department's PhDs. B0 T1(a): "
            f"R = {fmt(t1.get('a_R'))} {fmt_ci(t1.get('a_R_ci'))} ({t1.get('a_verdict')}; MDE80 "
            f"{fmt(t1.get('a_MDE80'), 3, False)}). B0 T1(b) gender audit: {t1.get('b_verdict_b')} "
            f"({t1.get('b_nsig')} of {t1.get('b_K')} fields with p < 0.05, allowed {t1.get('b_q95')}; mean z "
            f"{fmt(t1.get('b_zbar'), 2)} vs threshold {fmt(t1.get('b_zthr'), 2)}); T1(b') equivalence: "
            f"{t1.get('b_verdict_bp')} (mean shortfall {fmt(t1.get('b_dbar'))}, SE {fmt(t1.get('b_se_bp'), 3, False)}, "
            f"MDE80 {fmt(t1.get('b_MDE80_equivalence'), 3, False)}); every public edge row has at least one man and "
            f"one woman, so the two gender networks share their edges by construction and the gender audit carries "
            f"little information.")
    t2 = res.get("T2", {})
    if t2:
        items.append(
            f"**T2 (the academy's choice among the same graduates): A1 T2' {t2.get('A1_verdict')}; B0 "
            f"{t2.get('B0_verdict')}.** MDE80 of Pi^D {fmt(t2.get('A1_MDE80_PiD'), 3, False)}. Pi^D = "
            f"{iv(t2.get('A1_PiD'))}, Pi^DG = {iv(t2.get('A1_PiDG'))}, Pi^Y = {iv(t2.get('A1_PiY'))}, K = "
            f"{iv(t2.get('A1_K'))} (two-way). B0 (MDE80 {fmt(t2.get('B0_MDE80_meanDeltaD'), 3, False)}): mean "
            f"Delta^D = {ivx(t2.get('B0_mean_DeltaD'))}, mean (Delta^D - Delta^Y) = "
            f"{ivx(t2.get('B0_mean_dD_minus_dY'))}, mean rho(F,D|G) = {ivx(t2.get('B0_mean_pFD'))} (crossed). "
            f"Fields: {'; '.join(t2.get('fields_entering', []))}.")
    t3 = res.get("T3", {})
    if t3:
        items.append(
            f"**T3 (where the career rise occurs): A1 T3' {t3.get('A1_verdict')}; B0 {t3.get('B0_verdict')}.** "
            f"Permutation MDE80 of theta' {fmt(t3.get('A1_MDE80_perm'), 4, False)}/yr (half the benchmark "
            f"{fmt(t3.get('A1_half_benchmark'), 4, False)}). theta' = {iv(t3.get('A1_theta'), 4)}/yr, g1 = "
            f"{iv(t3.get('A1_g1'))}; within public {iv(t3.get('A1_theta_pub'), 4)}, within private "
            f"{iv(t3.get('A1_theta_priv'), 4)}. B0 (MDE80 {fmt(t3.get('B0_MDE80'), 4, False)} vs registered 0.019): "
            f"theta_GV = {iv(t3.get('B0_theta_GV'), 4)}/yr.")
    t4 = res.get("T4", {})
    if t4:
        items.append(
            f"**T4 (UK within-band decay of the selectivity premium): {t4.get('verdict')}.** MDE80 "
            f"{fmt(t4.get('MDE80'), 4, False)}/yr. Per-year slope of the partial Spearman of band median pay with "
            f"institution selectivity given G: {iv(t4.get('slope_partial_S_given_G'), 4)}; OLS form "
            f"{iv(t4.get('slope_ols_logpoints_per_SD'), 4)} log points per SD per year; reproduction of the known "
            f"raw within-band G slope: {iv(t4.get('check_slope_raw_G'), 4)} (scripts/62: +0.017). Exploratory, not "
            f"registered: band by band the slope is "
            + "; ".join(f"{b} {iv_band(t4.get('band_' + b, {}).get('slope_partial'), 3)}"
                        for b in sorted(k[5:] for k in t4 if k.startswith("band_PA")))
            + " (PA1 = top band, 4 As or more; its cells come from "
            + ", ".join(t4.get("band_PA1", {}).get("subject_names") or ["n/a"])
            + " only" + sched_note(t4.get("band_PA1", {}).get("subject_names"))
            + "). The registered \"rise\" (and C5's \"does not decay\") is a statement about the average "
              "over bands: the rise sits in the middle bands, and the top and lowest bands show no rise.")
    t5 = res.get("T5", {})
    if t5:
        items.append(
            f"**T5 (field map invariant to non-academic status indices): {t5.get('verdict')}.** "
            + "; ".join(f"{ZNAMES[z]}: MDE80 {fmt(t5.get(z + '_MDE80'), 3, False)}, r* = {fmt(t5.get(z + '_r_star'))} "
                        f"{fmt_ci(t5.get(z + '_r_star_ci'))}" for z in ZNAMES)
            + f" ({t5.get('n_fields')} fields with >= 60 institutions; map reliability of c_F "
              f"{fmt(t5.get('rel_F_map'))}).")
    t6 = res.get("T6", {})
    if t6:
        items.append(
            f"**T6 (status and quantities where pay is scheduled): {t6.get('verdict')}.** MDE80 "
            f"{fmt(t6.get('MDE80'), 3, False)}. Mean Spearman(G, T) over education and nursing = "
            f"{fmt(t6.get('primary_mean_spearman'))} {fmt_ci(t6.get('primary_ci'))}; by field "
            + ", ".join(f"{k} {fmt(v.get('est'))} {fmt_ci(v.get('ci'))}"
                        for k, v in (t6.get('primary_per_field') or {}).items())
            + (lambda z: f" ({', '.join(z)}: interval includes zero, so the registered \"supported\" rests mainly on "
                          f"the other field)" if z else "")(
                [k for k, v in (t6.get('primary_per_field') or {}).items() if (v.get('ci') or [0])[0] <= 0])
            + f". T uses nominal division wages with "
              f"no regional price adjustment, and its division medians tie (distinct values: education "
              f"{t6.get('education_distinct_division_values')} of 9, nursing {t6.get('nursing_distinct_division_values')} "
              f"of 9). With scripts/61's P in place of G "
            f"{fmt(t6.get('P_in_place_of_G_mean_spearman'))} {fmt_ci(t6.get('P_in_place_of_G_ci'))}; G with the share "
            f"working outside the scheduled industry {fmt(t6.get('G_outside_scheduled_industry_mean_spearman'))} "
            f"{fmt_ci(t6.get('G_outside_scheduled_industry_ci'))}. Exploratory (not registered, no interval): T "
            f"tracks the institution's own census division (Spearman(T, T_home) "
            + ", ".join(f"{k} {fmt(v.get('rho_T_Thome'), 2)}" for k, v in (t6.get('exploratory_home_division') or {}).items())
            + "), G is only weakly related to that home-division index ("
            + ", ".join(f"{k} {fmt(v.get('rho_G_Thome'), 2)}" for k, v in (t6.get('exploratory_home_division') or {}).items())
            + f"), and net of it the mean partial Spearman(G, T) is "
              f"{fmt(t6.get('exploratory_mean_partial_G_T_given_Thome'))}.")
    t7 = res.get("T7", {})
    if t7:
        items.append(
            f"**T7 (destination location): {t7.get('verdict')}.** MDE80 of R_L "
            f"{fmt(t7.get('i_MDE80_R_L'), 3, False)}. R_L = {iv(t7.get('i_R_W'))} (raw y5 coupling "
            f"{iv(t7.get('i_raw'))}, partial given W {iv(t7.get('i_pW'))}); with W and O {iv(t7.get('i_R_WO'))}. "
            f"Career slope: raw {iv(t7.get('ii_raw'), 4)}, given W_h {iv(t7.get('ii_pW'), 4)}, given W_h and O_h "
            f"{iv(t7.get('ii_pWO'), 4)}; share removed {fmt(t7.get('ii_share_removed_W'))} (W) and "
            f"{fmt(t7.get('iii_share_removed_WO'))} (W and O), no verdict. Reading: net of the wage level of the "
            f"census divisions where graduates work, y5 coupling is unchanged; about one fifth to one third of the "
            f"career rise moves with destination-division wages and the out-of-state share. W nets only differences "
            f"between the nine divisions, not location within them (metropolitan or not, state), so pay claims keep "
            f"the qualifier \"not netted of location within destination divisions\".")
    t8 = res.get("T8", {})
    if t8:
        items.append(
            f"**T8 (between-sector share, no verdict):** B = {iv(t8.get('B_share'))} (MDE80 "
            f"{fmt(t8.get('MDE80_B'), 3, False)}); raw coupling {iv(t8.get('raw'))}, within-sector "
            f"{iv(t8.get('within'))}; with scripts/61's own C (ages 22–40) B = {iv(t8.get('B_share_C_scripts61'))}.")
    for i, s in enumerate(items, 1):
        A(f"{i}. {s}")
    A("")
    A("No verdict of B0 or A1 licenses a claim about the mechanism below the name (employers' information versus the "
      "content of department standing); A9 says so, and nothing here changes it.\n")

    # ---------------- verdict table ----------------
    A("## Verdicts and the claim map (A9)\n")
    A("| test | B0 verdict | A1 verdict | claim | consequence under A9 |")
    A("|---|---|---|---|---|")
    vt = {"T0": (None, g(res, "T0", "b_verdict")), "T1": (g(res, "T1", "B0_T1_verdict"), g(res, "T1", "ap_verdict")),
          "T2": (g(res, "T2", "B0_verdict"), g(res, "T2", "A1_verdict")),
          "T3": (g(res, "T3", "B0_verdict"), g(res, "T3", "A1_verdict")), "T4": (None, g(res, "T4", "verdict")),
          "T5": (None, g(res, "T5", "verdict")), "T6": (None, g(res, "T6", "verdict")),
          "T7": (None, g(res, "T7", "verdict")), "T8": (None, "reported (no verdict)")}
    for t, (v0, v1) in vt.items():
        if t not in res:
            continue
        c, lab, cons = CLAIM_MAP[t]
        cons_s = cons.get(v1, "undecided" if cons else "B reported next to every construct statement about pay")
        if t == "T3" and v1 not in cons:
            cons_s = "no registered rewrite; the decision-table label is reported"
        A(f"| {t} | {v0 or '—'} | {v1} | {c}: {lab} | {cons_s} |")
    extra = []
    if "T1" in res:
        extra.append(f"T1(a) B0: {g(res, 'T1', 'a_verdict')}; T1(b) B0: {g(res, 'T1', 'b_verdict_b')}; T1(b') A1: "
                     f"{g(res, 'T1', 'b_verdict_bp')} (T1(b) and (b') bear only on H3, gender-specific steering).")
    if "T2" in res:
        extra.append(f"T2 B0 raw verdict before A5.6: {g(res, 'T2', 'B0_verdict_raw')}.")
    for e in extra:
        A(f"\n{e}")
    A("")

    # ---------------- per-test sections ----------------
    if "T0" in res:
        A(section_T0(res["T0"]))
    if "T1" in res:
        A(section_T1(res["T1"]))
    if "T2" in res:
        A(section_T2(res["T2"]))
    if "T3" in res:
        A(section_T3(res["T3"]))
    if "T4" in res:
        A(section_T4(res["T4"]))
    if "T5" in res:
        A(section_T5(res["T5"]))
    if "T6" in res:
        A(section_T6(res["T6"]))
    if "T7" in res:
        A(section_T7(res["T7"]))
    if "T8" in res:
        A(section_T8(res["T8"]))
    t0_, t1_ = res.get("T0", {}), res.get("T1", {})
    A(DEVIATIONS_T.format(
        t0_unid=fmt(t0_.get("b_exact_limit_unidentified_node_share"), 2, False),
        t0_rdc=iv(t0_.get("b_R_DC")), t0_v=t0_.get("b_verdict"),
        t0_exact=iv(t0_.get("b_sens_R_DC_exact_limit")), t0_ident=iv(t0_.get("b_sens_R_DC_exact_limit_identified_only")),
        t0_identp=iv(t0_.get("b_sens_R_perp_identified_only")), t0_s0=t0_.get("b_sens_s0_rule_reading"),
        t0_s1=t0_.get("b_sens_s1_rule_reading"),
        t0_dep=("the verdict does not depend on the completion" if t0_.get("b_verdict") ==
                t0_.get("b_sens_s0_rule_reading") == t0_.get("b_sens_s1_rule_reading") else
                "the readings differ, so the verdict depends on the completion"),
        t1_kept=fmt(t1_.get("ap_sim_kept_person_share"), 3, False), t1_elig=fmt(t1_.get("ap_sim_mean_eligible"), 1, False),
        t1_fail=fmt(t1_.get("ap_sim_field_excluded_share"), 3, False), t1_cells=t1_.get("ap_sim_cells_used"),
        t1_nf=t1_.get("ap_sim_fields_with_any_cell"),
        t1_r0=fmt(t1_.get("ap_R0")), t1_dr=f"{fmt(t1_.get('ap_DeltaR'))} {fmt_ci(t1_.get('ap_DeltaR_ci'))}",
        t1_rule=t1_.get("ap_rule"), t1_pdr=f"{fmt(t1_.get('ap_post_DeltaR'))} {fmt_ci(t1_.get('ap_post_DeltaR_ci'))}",
        t1_prule=t1_.get("ap_post_rule")))
    A(METHOD)
    A(CAVEATS)
    A(revisions(res))
    A("## Provenance\n")
    A(f"- Script SHA-256 when this report was written: `{meta['script_sha']}`; stub SHA-256 (pre-implementation): "
      f"`{STUB_SHA256}`.")
    for t in TESTS:
        if t in res and isinstance(res[t].get("_run"), dict):
            rr = res[t]["_run"]
            A(f"- {t}: run with script SHA-256 `{rr['script_sha256'][:16]}…`, workers {rr['workers']}, threads "
              f"{rr['threads']}.")
    for k, v in meta.get("inputs", {}).items():
        A(f"- `{k}`: {v}")
    if meta.get("compare"):
        same = [t for t, v in meta["compare"].items() if v]
        diff = [t for t, v in meta["compare"].items() if not v]
        A(f"- Re-run check: every test was run twice with this script in separate processes (same thread and worker "
          f"settings); the per-test JSON outputs are byte-identical for {', '.join(same) or 'none'}"
          + (f" and differ for {', '.join(diff)}" if diff else "") + ".")
    A("")
    txt = "\n".join(L)
    OUT_MD.write_text(txt)
    print(txt)


def sched_note(names) -> str:
    """Flags a UK subject whose pay follows national pay scales (Proposition 5), where a school-mean premium is not
    expected to move with employer learning."""
    return (", a subject whose pay follows national (NHS) pay scales" if names and "Medicine and dentistry" in names
            else "")


def iv_band(r, d=3):
    """Band slope with its two-way interval, or, when a band has too few subjects for the two-way variance, the
    crossed (institution) percentile interval, marked as such."""
    if not isinstance(r, dict):
        return "n/a"
    if r.get("lo") is None or r.get("hi") is None:
        return f"{fmt(r['est'], d)} [{fmt(r['xlo'], d)}, {fmt(r['xhi'], d)}] (k = {r['k']}; institution percentile)"
    return iv(r, d)


def revisions(res: dict) -> str:
    """Revision log after an independent verification of the first run (2026-09-25). Old values are those of the
    first report; new values are this run's."""
    t0, t1, t4, t6 = res.get("T0", {}), res.get("T1", {}), res.get("T4", {}), res.get("T6", {})
    bands = sorted(k[5:] for k in t4 if k.startswith("band_PA"))
    band_s = "; ".join(f"{b} {iv_band(t4.get('band_' + b, {}).get('slope_partial'), 3)}" for b in bands)
    pf = t6.get("primary_per_field") or {}
    pf_s = ", ".join(f"{k} {fmt(v.get('est'))} {fmt_ci(v.get('ci'))}" for k, v in pf.items())
    aud_a = t0.get("audit_first_run_academia") or [float("nan"), float("nan")]
    t5m = [res.get("T5", {}).get(z + "_MDE80") for z in ZNAMES] if "T5" in res else [float("nan")]
    t5m = [x for x in t5m if x is not None] or [float("nan")]
    L = ["## Revisions (after an independent verification of the first run)\n",
         "The first run's report was checked by an independent verifier, who re-derived the main numbers of T0, T4, "
         "T6 and T7 independently and raised one major and seven minor points. Each was checked before it was acted on; binding texts B0 and A1 "
         "are unchanged (digests above). All nine tests were re-run with the revised script; no registered rule was "
         "changed.\n",
         f"1. **T0 exact-limit sensitivity came from a Newton solve that had not converged (major; confirmed and "
         f"wider than reported).** Re-running the first run's undamped Newton: in the exact limit "
         f"{t0.get('audit_first_run_exact_limit_fields_not_converged')} of {t0.get('audit_n_field_fits')} field "
         f"fits ended with max |gradient| >= 1e-6 (largest {t0.get('audit_first_run_exact_limit_max_abs_gradient', float('nan')):.3g}, "
         f"|theta| up to {t0.get('audit_first_run_exact_limit_max_abs_theta', float('nan')):.3g}); with the +1 "
         f"convention {t0.get('audit_first_run_primary_fields_not_converged')} field fit(s) "
         f"({', '.join(t0.get('audit_first_run_primary_fields') or []) or 'none'}) and the Academia fit (max "
         f"|gradient| {aud_a[0]:.3g}, |theta| up to {aud_a[1]:.3g}) had not converged either. Every fit now uses "
         f"damped Newton with a line search and asserts max |gradient| < 1e-6 (largest now: "
         f"{t0.get('b_bt_max_abs_gradient_primary', float('nan')):.1e} +1, "
         f"{t0.get('b_bt_max_abs_gradient_exact_limit', float('nan')):.1e} exact limit, "
         f"{t0.get('b_academia_bt_max_abs_gradient', float('nan')):.1e} Academia). Primary: R_DC +0.839 -> "
         f"{iv(t0.get('b_R_DC'))}, R_perp +0.598 -> {iv(t0.get('b_R_perp'))}, mean (c_GDC - c_G) -0.013 -> "
         f"{iv(t0.get('b_GDC_minus_G'))}; verdict \"{t0.get('b_verdict')}\". Exact limit: +0.358 -> "
         f"{iv(t0.get('b_sens_R_DC_exact_limit'))} (theta = 0 for unidentified nodes, rule "
         f"\"{t0.get('b_sens_s0_rule_reading')}\"); +0.712 -> {iv(t0.get('b_sens_R_DC_exact_limit_identified_only'))} "
         f"(identified cells, {t0.get('b_sens_s1_fields')} fields, rule \"{t0.get('b_sens_s1_rule_reading')}\"). The "
         f"sentence \"the verdict depends on how the registered formula is completed at zero degrees\" is withdrawn.",
         "2. **Summary labels (minor; confirmed).** The workflow summary had mapped T0, T2 and T5 to \"underpowered\" and "
         "T8 to \"supported\". The registered labels are: T0 \"partly production\", T2 (B0 and A1) \"inconclusive\", T5 "
         "\"inconclusive\", T8 no verdict (registered decomposition). Where a summary format has no such label, the "
         "registered label is stated next to it; T2 and T5 are substantively underpowered (MDE80 of Pi^D "
         f"{fmt(g_(res.get('T2', {}), 'A1_PiD', 'mde'), 3, False)}; MDE80 of r* "
         f"{fmt(min(t5m), 3, False)} to {fmt(max(t5m), 3, False)}), "
         "T0 is not (MDE80 of R_DC "
         f"{fmt(t0.get('b_MDE80_R_DC'), 3, False)}).",
         f"3. **T1(a') \"infeasible as registered\" used a rule set after the run (minor; confirmed).** R0 is now "
         f"computed as A4.1 states it in both readings: literal order R0 {fmt(t1.get('ap_R0'))} from "
         f"{t1.get('ap_sim_cells_used')} cells, Delta_R {fmt(t1.get('ap_DeltaR'))} {fmt_ci(t1.get('ap_DeltaR_ci'))} "
         f"(\"{t1.get('ap_rule')}\", degenerate); post-censoring Delta_R {fmt(t1.get('ap_post_DeltaR'))} "
         f"{fmt_ci(t1.get('ap_post_DeltaR_ci'))} (\"{t1.get('ap_post_rule')}\"). A4.1 is ambiguous about the "
         f"calibration; the post-censoring reading is the only non-degenerate one and now governs C2 (verdict "
         f"\"{t1.get('ap_verdict')}\"): F is described as the net-export position in the hiring network, not as one "
         f"belief about the quality of a department's PhDs.",
         "4. **C1 wording (minor; confirmed).** In the primary fit the reading rests mainly on R_perp (the lower bound "
         f"of R_DC is {fmt(g_(t0, 'b_R_DC', 'lo'))}), and R_perp also removes net export driven by valuation, so "
         "\"partly production\" is an upper-bound statement. F_DC keeps the export-ratio information (mean Spearman "
         f"with x on the common cells {fmt(t0.get('b_diag_mean_rho_FDC_x_cells'))} against "
         f"{fmt(t0.get('b_diag_mean_rho_F_x_cells'))} for F): added to Caveats as a validity limit of T0.",
         f"5. **Deposit (minor; confirmed).** The stub is kept at `{STUB_KEEP}` (git-ignored, digest checked); the "
         "registration is described everywhere as self-attested and not deposited before the run.",
         "6. **C8 overstated (minor; confirmed).** W nets differences between the nine census divisions only. The pay "
         "claims now read: y5 coupling unchanged net of destination-division wages (R_L "
         f"{fmt(g_(res.get('T7', {}), 'i_R_W', 'est'), 2)}); one fifth to one third of the career rise moves with "
         "destination-division wages and the out-of-state share; \"not netted of location within destination "
         "divisions\" is kept.",
         f"7. **T6 uneven across fields (minor; confirmed).** Per-field estimates are reported: {pf_s}. T uses nominal "
         f"division wages (no price adjustment), and its division medians tie (education "
         f"{t6.get('education_distinct_division_values')} and nursing {t6.get('nursing_distinct_division_values')} "
         "distinct values of 9). C7 is kept with the location and price qualifiers.",
         f"8. **T4 averages over bands (minor; confirmed).** Exploratory per-band slopes: {band_s}. C5's \"does not "
         "decay\" is a statement about the average over bands. The top band's negative point estimate comes from "
         f"{', '.join(t4.get('band_PA1', {}).get('subject_names') or ['n/a'])} alone"
         f"{sched_note(t4.get('band_PA1', {}).get('subject_names'))}, so it cannot show whether the premium decays "
         "where own attainment is most uniform.",
         ""]
    return "\n".join(L)


def fmt_ci(c, d=3):
    if not c or c[0] is None:
        return "[n/a]"
    return f"[{fmt(c[0], d)}, {fmt(c[1], d)}]"


def row(label, r, d=3, note=""):
    if not isinstance(r, dict):
        return f"| {label} | n/a | | | | {note} |"
    return (f"| {label} | {fmt(r['mde'], d, False)} | {fmt(r['est'], d)} | [{fmt(r['lo'], d)}, {fmt(r['hi'], d)}]"
            f"{' (edge)' if r.get('edge') else ''}{' (fallback)' if r.get('fallback') else ''} | "
            f"[{fmt(r['xlo'], d)}, {fmt(r['xhi'], d)}] | k={r['k']}{'; ' + note if note else ''} |")


HDR = ("| statistic | MDE80 | estimate | two-way 95% CI | crossed 95% CI | notes |\n"
       "|---|---|---|---|---|---|")


def md_table(recs, cols, d=3):
    if not recs:
        return "(none)\n"
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in recs:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                cells.append(fmt(v, d, sign=not c.startswith("n")))
            elif isinstance(v, list):
                cells.append(fmt_ci(v, d))
            else:
                cells.append("" if v is None else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def section_T0(r):
    L = ["## T0 — production audit of F (A1 A3)\n",
         "Question: is F an ordering by the direction of exchange net of volume, or by production-to-hiring volume? "
         "Data: Wapman public field edges without self-hires; IPEDS C2023_a AWLEVEL 17 doctorates in each field's "
         "CIP codes; institution-stats ProductionRank; scripts/28 Scorecard 4-yr cells.\n",
         "**(a) No earnings.** Mean over fields (field bootstrap 95% CI):\n",
         "| Spearman(F, ·) | mean | 95% CI |", "|---|---|---|"]
    for v, lab in [("p", "log(1 + research doctorates), p"), ("x", "export ratio log((d_out+1)/(d_in+1)), x"),
                   ("q", "placement rate d_out / doctorates, q"), ("negPR", "-ProductionRank")]:
        L.append(f"| {lab} | {fmt(r.get('a_mean_rho_F_' + v))} | {fmt_ci(r.get('a_mean_rho_F_' + v + '_ci'))} |")
    L += ["", f"**(b) Degree-corrected ranks** ({r.get('b_fields')} fields, {r.get('b_cells_total')} common cells; "
          f"MDE80 of R_DC {fmt(r.get('b_MDE80_R_DC'), 3, False)} first). Verdict: **{r.get('b_verdict')}**.\n", HDR,
          row("R_DC = mean c_DC / mean c_F", r.get("b_R_DC")), row("R_perp = mean c_perp / mean c_F", r.get("b_R_perp")),
          row("mean (c_GDC - c_G)", r.get("b_GDC_minus_G")), row("mean c_F (common cells)", r.get("b_mean_c_F")),
          row("mean c_DC", r.get("b_mean_c_FDC")), row("mean c_perp", r.get("b_mean_c_Fp")), "",
          f"Mean Spearman(F, F_DC) over ranked network nodes: {fmt(r.get('b_mean_rho_F_FDC_network'))}; on the "
          f"common cells {fmt(r.get('b_mean_rho_F_FDC_cells'))}. Mean c_G {fmt(r.get('b_mean_c_G'))}, mean c_GDC "
          f"{fmt(r.get('b_mean_c_GDC'))}.\n",
          f"*Zero-degree completion (sensitivities, not in the verdict; revision 1).* The registered offset is "
          f"undefined at zero degrees; {fmt(r.get('b_exact_limit_unidentified_node_share'), 3, False)} of network "
          f"nodes (and {fmt(r.get('b_sens_s0_unidentified_cell_share'), 3, False)} of the common cells) have no "
          f"finite-offset pair in the exact limit. Every fit is solved by damped Newton and checked for convergence: "
          f"largest final max |gradient| {r.get('b_bt_max_abs_gradient_primary', float('nan')):.1e} (+1 convention), "
          f"{r.get('b_bt_max_abs_gradient_exact_limit', float('nan')):.1e} (exact limit), "
          f"{r.get('b_academia_bt_max_abs_gradient', float('nan')):.1e} (Academia); largest |theta| in the exact "
          f"limit {fmt(r.get('b_bt_max_abs_theta_exact_limit'), 2, False)}. The registered rule is applied to each "
          f"completion for information.\n", HDR,
          row("(s0) R_DC, exact limit, theta = 0 for unidentified nodes", r.get("b_sens_R_DC_exact_limit"),
              note=f"rule with the primary R_perp: {r.get('b_sens_s0_rule_reading')}"),
          row(f"(s1) R_DC, exact limit, identified cells only ({r.get('b_sens_s1_cells')} cells)",
              r.get("b_sens_R_DC_exact_limit_identified_only")),
          row("(s1) R_perp on the same identified cells", r.get("b_sens_R_perp_identified_only"),
              note=f"rule: {r.get('b_sens_s1_rule_reading')}"), "",
          f"*Validity diagnostic (exploratory, revision 4).* Mean Spearman over the {r.get('b_fields')} fields, on the "
          f"common cells, of the export ratio x with F {fmt(r.get('b_diag_mean_rho_F_x_cells'))}, with F_DC (+1) "
          f"{fmt(r.get('b_diag_mean_rho_FDC_x_cells'))} and with F_DC (exact limit) "
          f"{fmt(r.get('b_diag_mean_rho_FDC0_x_cells'))}; F_DC (+1) is closer to x than F in "
          f"{r.get('b_diag_fields_rho_FDC_x_above_rho_F_x')} of {r.get('b_fields')} fields. The degree correction with "
          f"the +1 convention does not remove the export-ratio information from the ranking, so R_DC does not isolate "
          f"the part of F that is free of volume (the exact-limit rank is nearly unrelated to x, but it puts "
          f"{fmt(r.get('b_sens_s0_unidentified_cell_share'), 2, False)} of the cells in the middle). R_perp removes x, "
          f"but by H1 x is itself partly produced by valuation (valued departments export on net), so R_perp "
          f"over-removes. Neither branch separates production cleanly; \"partly production\" is therefore an "
          f"upper-bound statement about the production share.\n",
          f"**(c) Minimum-violation orderings.** Mean Spearman(F, MVR) = {fmt(r.get('c_mean_rho_F_MVR'))}; share of "
          f"persons hired upward under the MVR {fmt(r.get('c_share_upward_MVR'), 3, False)}, under the SpringRank "
          f"start {fmt(r.get('c_share_upward_SpringRank_start'), 3, False)}; among ranked network nodes, upward "
          f"hires under the published F order vs the MVR: {r.get('c_upward_F_vs_MVR_ranked')}.\n",
          "Per-field coupling (b):\n",
          md_table(r.get("b_table", []), ["field", "n", "c_F", "c_DC", "c_perp", "c_G", "c_GDC", "rho_F_FDC_network"])]
    return "\n".join(L)


def section_T1(r):
    L = ["## T1 — single-index audit of the hiring measure (B0 T1; A1 A4)\n",
         f"No earnings data. {r.get('n_fields_total')} field networks; persons (no self-hires) "
         f"{int(r.get('persons_field_level_no_selfhire') or 0):,}; share without recorded gender "
         f"{fmt(r.get('share_no_recorded_gender'), 4, False)} (A0.5 erratum: 5.3%). The batched SpringRank solver "
         f"matches src/ar_pipeline/springrank to {r.get('springrank_batch_vs_src_maxabs'):.1e}.\n",
         f"**B0 T1(a) role consistency.** MDE80 {fmt(r.get('a_MDE80'), 3, False)}. R = {fmt(r.get('a_R'))} "
         f"{fmt_ci(r.get('a_R_ci'))} over {r.get('a_k')} fields ({r.get('a_fields_excluded')} excluded, "
         f"{r.get('a_fields_excluded_by_n')} of them for fewer than 15 eligible institutions); mean r_PP "
         f"{fmt(r.get('a_mean_rPP'))}, r_HH {fmt(r.get('a_mean_rHH'))}, r_PH {fmt(r.get('a_mean_N'))}. "
         f"Verdict: **{r.get('a_verdict')}** (rule: lower bound >= 0.80).\n",
         f"**A property of the public edge list that limits T1(b) and the T1(a') benchmark.** Every public edge row "
         f"(share {fmt(r.get('share_rows_with_a_man_and_a_woman'), 3, False)}) has at least one man and at least one "
         f"woman. The men's and the women's networks therefore contain exactly the same institution pairs, while the "
         f"registered relabelling null does not keep that property; T1(b) and T1(b') cannot detect gender-specific "
         f"steering in these data, and their verdicts below are reported as registered but carry little information. "
         f"The same property means the public censoring is not only 'A >= 2', so A4.1's simulated censoring (A* >= 2) "
         f"is an approximation.\n",
         f"**B0 T1(b) gender invariance.** K = {r.get('b_K')} fields; fields with permutation p < 0.05: "
         f"{r.get('b_nsig')} (95th percentile of Binomial(K, 0.05) = {r.get('b_q95')}); mean z = "
         f"{fmt(r.get('b_zbar'), 3)} (threshold {fmt(r.get('b_zthr'), 3)}); mean c = {fmt(r.get('b_mean_c'))}. "
         f"Verdict: **{r.get('b_verdict_b')}**. A positive mean z means the men's and women's orderings agree more "
         f"than orderings built from random relabellings of the same persons (the >= 3-person qualification rule is "
         f"reapplied to every relabelled split, so the institutions compared differ between the observed and the "
         f"relabelled splits). Exploratory: institutions qualifying, observed vs relabelled (mean over fields) "
         f"{r.get('b_exploratory_mean_nq_observed_vs_null')}; with the null computed on the observed qualifying "
         f"institutions, the mean relabelled c is {fmt(r.get('b_exploratory_mean_null_c_fixed_set'))} and the mean z "
         f"{fmt(r.get('b_exploratory_mean_z_fixed_set'), 2)}. B0 T1 overall: **{r.get('B0_T1_verdict')}**.\n",
         f"**A1 T1(b') equivalence.** MDE80 {fmt(r.get('b_MDE80_equivalence'), 3, False)}; mean shortfall "
         f"dbar = {fmt(r.get('b_dbar'))}, SE {fmt(r.get('b_se_bp'), 4, False)}; dbar + 1.645 SE = "
         f"{fmt(r.get('b_upper_bp'))}, dbar - 1.96 SE = {fmt(r.get('b_lower_bp'))}. Verdict: "
         f"**{r.get('b_verdict_bp')}**.\n",
         f"**A1 T1(a') cross-fitted and benchmarked.** R' = {fmt(r.get('ap_Rprime'))} "
         f"{fmt_ci(r.get('ap_post_Rprime_ci'))} over {r.get('ap_k')} fields (real networks: "
         f"{fmt(r.get('ap_real_mean_eligible'), 1, False)} eligible institutions on average). Benchmark: 100 networks "
         f"per field from the fitted SpringRank model, A* < 2 dropped. A4.1 says \"c matching the field's total "
         f"count\" without saying whether the simulated total is matched before or after that censoring; both "
         f"readings are computed with the identical pipeline and the registered rule is applied to each.\n",
         "| reading of A4.1 | persons kept | eligible inst. (mean) | cells failing B0 rules | cells in R0 | MDE80 | R0 "
         "| Delta_R [95% CI] | rule |",
         "|---|---|---|---|---|---|---|---|---|",
         f"| post-censoring: E[total after A* >= 2] = public total (governs C2) | "
         f"{fmt(r.get('ap_post_sim_kept_person_share'), 3, False)} | {fmt(r.get('ap_post_sim_mean_eligible'), 1, False)} | "
         f"{fmt(r.get('ap_post_sim_field_excluded_share'), 3, False)} | {r.get('ap_post_sim_cells_used')} | "
         f"{fmt(r.get('ap_post_MDE80'), 3, False)} | {fmt(r.get('ap_post_R0'))} {fmt_ci(r.get('ap_post_R0_ci'))} | "
         f"{fmt(r.get('ap_post_DeltaR'))} {fmt_ci(r.get('ap_post_DeltaR_ci'))} | {r.get('ap_post_rule')} |",
         f"| literal order: E[total before censoring] = public total (degenerate) | "
         f"{fmt(r.get('ap_sim_kept_person_share'), 3, False)} | {fmt(r.get('ap_sim_mean_eligible'), 1, False)} | "
         f"{fmt(r.get('ap_sim_field_excluded_share'), 3, False)} | {r.get('ap_sim_cells_used')} | "
         f"{fmt(r.get('ap_MDE80'), 3, False)} | {fmt(r.get('ap_R0'))} {fmt_ci(r.get('ap_R0_ci'))} | "
         f"{fmt(r.get('ap_DeltaR'))} {fmt_ci(r.get('ap_DeltaR_ci'))} | {r.get('ap_rule')} |", "",
         f"Total cells: {int(r.get('ap_k') or 0) * N_SIM} (fields x simulated networks). In the literal-order reading "
         f"the simulated networks keep about a tenth of their persons; fields contributing any cell: "
         f"{r.get('ap_sim_fields_with_any_cell')} of {r.get('ap_k')} "
         f"({', '.join(r.get('ap_sim_fields_with_any_cell_names') or [])}); replicate sets with no field: "
         f"{r.get('ap_sim_networks_without_any_field')} of 100. Its R0 is therefore the "
         f"benchmark of that field alone, and its interval uses only the {r.get('ap_DeltaR_finite_draws')} of 1000 "
         f"field-bootstrap draws that contain it: defined, but degenerate. "
         f"The post-censoring reading is the only calibration whose simulated networks resemble the public one "
         f"({r.get('ap_post_sim_fields_with_any_cell')} of {r.get('ap_k')} fields contribute; the model spreads the "
         f"same persons over more institutions than the public network, so it is not an exact analogue). It was "
         f"adopted after the literal-order benchmark had been seen to degenerate and before any post-censoring "
         f"statistic was computed. Verdict (A1 T1(a')): **{r.get('ap_verdict')}**, with the upper bound of Delta_R "
         f"close to -0.10; claim C2 is written accordingly: F is the net-export position in the hiring network, and "
         f"the reading of F as one belief about the quality of a department's PhDs is not claimed. The first run "
         f"reported the literal-order benchmark as \"infeasible as registered\" by a rule set after the run (revision "
         f"3).\n",
         f"**Academia network (reported, not in the verdict).** R = {fmt(r.get('acad_a_R'))} (N "
         f"{fmt(r.get('acad_a_N'))}, D {fmt(r.get('acad_a_D'))}, {r.get('acad_a_n_elig')} eligible institutions); "
         f"gender: {json.dumps(r.get('acad_b'))}.\n",
         "Per-field T1(a) (B0) and T1(a') (A1):\n",
         md_table(merge_recs(r.get("a_table", []), r.get("ap_table", []), "field",
                             ["N", "D", "enters", "Rp_f", "R0_f"]),
                  ["field", "n_elig", "N", "D", "enters", "N_ap", "D_ap", "enters_ap", "Rp_f_ap", "R0_f_ap"]),
         "Per-field T1(b):\n",
         md_table(r.get("b_table", []), ["field", "c", "nq", "enters", "null_mean", "null_sd", "z", "p"])]
    return "\n".join(L)


def merge_recs(a, b, key, cols):
    bm = {x[key]: x for x in b}
    out = []
    for x in a:
        y = dict(x)
        for c in cols:
            y[c + "_ap"] = bm.get(x[key], {}).get(c)
        out.append(y)
    return out


def section_T2(r):
    L = ["## T2 — the academy's choice among the same bachelor's graduates (B0 T2; A1 A5)\n",
         "ORCID education episodes (US, ROR-resolved), bachelor's -> later research doctorate at a different "
         "institution in the same field; destinations scored on the field rank (D) and on the academia-wide rank "
         "(D^G); pay Y = log Scorecard 4-yr median on the same cells.\n",
         f"Fields entering (>= 15 cells of >= 3 persons): {'; '.join(r.get('fields_entering', []))}.\n",
         "Counts and match rates (primary variant; records = persons x bachelor's institution with a later doctorate "
         "elsewhere; matched = share of records whose institution has a Wapman rank; persons = persons in cells):\n",
         md_table([dict(field=f, **v) for f, v in (r.get("primary_counts") or {}).items()],
                  ["field", "records", "bac_matched_F", "doc_matched_D", "cells", "cells_Y", "persons"]),
         f"**B0** (crossed bootstrap, percentile; MDE80 of mean Delta^D = "
         f"{fmt(r.get('B0_MDE80_meanDeltaD'), 3, False)} first). Verdict: **{r.get('B0_verdict')}**.\n", HDR,
         row("mean Delta^D = rho(F,D) - rho(G,D)", r.get("B0_mean_DeltaD")),
         row("mean rho(F, D | G)", r.get("B0_mean_pFD")),
         row("mean (Delta^D - Delta^Y)", r.get("B0_mean_dD_minus_dY")),
         row("mean Delta^Y (T2 cells with Y)", r.get("B0_mean_DeltaY")), "",
         f"**A1 T2'** (two-way; MDE80 of Pi^D {fmt(r.get('A1_MDE80_PiD'), 3, False)} first). Verdict: "
         f"**{r.get('A1_verdict')}**.\n", HDR,
         row("Pi^D = mean rho(F, D | G)", r.get("A1_PiD")), row("Pi^DG (destinations on G)", r.get("A1_PiDG")),
         row("Pi^Y = mean rho(F, Y | G)", r.get("A1_PiY")), row("K = mean [rho(F,D|G) - rho(F,Y|G)]", r.get("A1_K")),
         row("K, both partials on the Y cells (sensitivity)", r.get("A1_K_same_cells")),
         row("Pi^D + same-state share control (A5.5)", r.get("A5.5_PiD_state_control")),
         row("Pi^D, word-anchored keywords (A5.5)", r.get("anchored_keywords_PiD")),
         row("Pi^D, literal B0 doctorate regex (A2.3)", r.get("literal_B0_regex_PiD")),
         row("B0 mean Delta^D, literal B0 regex", r.get("literal_B0_regex_B0_mean_DeltaD")),
         row("B0 mean (Delta^D - Delta^Y), literal B0 regex", r.get("literal_B0_regex_B0_mean_dD_minus_dY")), "",
         f"LOFO jackknife intervals: Pi^D {fmt_ci([g_(r, 'A1_PiD', 'jlo'), g_(r, 'A1_PiD', 'jhi')])}, Pi^DG "
         f"{fmt_ci([g_(r, 'A1_PiDG', 'jlo'), g_(r, 'A1_PiDG', 'jhi')])}, K "
         f"{fmt_ci([g_(r, 'A1_K', 'jlo'), g_(r, 'A1_K', 'jhi')])}. Gaussian-copula partial (A2.5), mean over "
         f"fields: {fmt(r.get('A2.5_mean_gaussian_copula_partial'))}.\n",
         f"**A5.5 (reported, no verdict).** Reliability of D (persons split within cells, Spearman-Brown): "
         f"{json.dumps({k: round(v['rel_SB'], 3) for k, v in (r.get('A5.5_reliability_D') or {}).items()})}. "
         f"Reliability-corrected Delta^D, mean over fields: {fmt(r.get('A5.5_mean_DeltaD_corrected_LB'))} at the "
         f"public-edge lower bound of rel_F and {fmt(r.get('A5.5_mean_DeltaD_corrected_EXT'))} at the extrapolated "
         f"end.\n",
         "Same-institution doctorates (dropped by B0), share of persons by tercile of F:\n",
         md_table(r.get("A5.5_same_institution_share_by_F_tercile", []),
                  ["field", "same_share_low", "same_share_mid", "same_share_high", "n_persons"]),
         "Per-field B0 estimates (institution-bootstrap 95% CI):\n",
         md_table(r.get("B0_per_field", []), ["field", "n_cells", "persons", "dD", "dD_ci", "pFD", "pFD_ci", "pFDG",
                                              "dY", "pFY"])]
    return "\n".join(L)


def g_(r, k, s):
    x = r.get(k)
    return x.get(s) if isinstance(x, dict) else None


def section_T3(r):
    L = ["## T3 — where the career rise occurs (B0 T3; A1 A6)\n",
         f"PSEO V4.14.1 fixed cohorts 2001/2004/2007/2010 at y1/y5/y10 (scripts/66 sample A; NMIN 15). Visibility V = "
         f"log UGDS (Scorecard, OPEID main campus as scripts/61). Cells: B0 {r.get('B0_cells')} "
         f"({r.get('B0_fields')} fields); A1 {r.get('A1_cells')}; within type (>= 10 institutions) "
         f"{r.get('A1_type_cells')} (public, private).\n",
         f"**B0** (MDE80 {fmt(r.get('B0_MDE80'), 4, False)} first; registered null threshold 0.019/yr). Verdict: "
         f"**{r.get('B0_verdict')}**.\n", HDR,
         row("theta_GV (per-year slope of b_GV)", r.get("B0_theta_GV"), 4),
         row("benchmark: per-year slope of the G-only loading", r.get("B0_benchmark_dG"), 4),
         row("(i) V = log IPEDS BA completions", r.get("B0_secondary_i_ipeds_completions"), 4),
         row(f"(ii) + SAT, -ADM, Pell, state level, private ({r.get('B0_secondary_ii_cells')} cells, n >= 18)",
             r.get("B0_secondary_ii_S2_controls"), 4),
         row("(iii) y5 -> y10 segment of b_GV", r.get("B0_secondary_iii_y5_y10_segment"), 4), "",
         f"**A1 T3'** (permutation MDE80 {fmt(r.get('A1_MDE80_perm'), 4, False)}, computed before any estimate; "
         f"half the benchmark {fmt(r.get('A1_half_benchmark'), 4, False)}). Verdict: **{r.get('A1_verdict')}**.\n",
         HDR, row("theta' (with control type and G x control)", r.get("A1_theta"), 4),
         row("g1 = b_GV at y1", r.get("A1_g1")), row("dG at V_perp = -1 (low visibility)", r.get("A1_dGlow"), 4),
         row("dG at V_perp = +1 (high visibility)", r.get("A1_dGhigh"), 4),
         row("benchmark (G-only loading slope)", r.get("A1_bench"), 4),
         row("theta' within public institutions", r.get("A1_theta_pub"), 4),
         row("theta' within private institutions", r.get("A1_theta_priv"), 4),
         row("partial-correlation form, slope", r.get("A1_ptheta"), 4),
         row("partial-correlation form, y1", r.get("A1_pg1")),
         row("V = out-of-state share at y1: theta'", r.get("A1_secondary_V_out_theta"), 4),
         row("V = out-of-state share at y1: g1", r.get("A1_secondary_V_out_g1")),
         row("V = log IPEDS completions: theta'", r.get("A1_secondary_V_comp_theta"), 4),
         row("V = log IPEDS completions: g1", r.get("A1_secondary_V_comp_g1")), "",
         f"The within-private slope needs cells with >= 10 private institutions; the largest number of private "
         f"institutions in any T3' cell is {r.get('A1_max_private_per_cell')}, so theta' within private institutions is "
         f"{'undefined and the control-type proviso of A6.5 cannot be checked' if not isinstance(r.get('A1_theta_priv'), dict) else 'reported above'}. "
         f"Mean of theta' over the 200 permutations: {fmt(r.get('A1_perm_mean'), 4)}.\n"]
    return "\n".join(L)


def section_T4(r):
    L = ["## T4 — UK within-band decay of the school-mean premium (A1 A7)\n",
         f"DfE LEO provider x CAH2 x prior-attainment band panels (bands 1–5), cohorts 2013/14–2016/17 at YAG 1/3/5, "
         f"balanced providers, >= 15 providers (scripts/62 cells_career). Reproduction of scripts/62's within-band "
         f"cells: {r.get('reproduction_cells')} (rebuilt, reference), max |difference| in raw G coupling "
         f"{r.get('reproduction_max_abs_diff'):.1e}, n mismatches {r.get('reproduction_n_mismatch')}. Cells with "
         f"selectivity S for >= 15 providers: {r.get('cells_with_S')} ({r.get('subjects')} subjects).\n",
         f"MDE80 {fmt(r.get('MDE80'), 4, False)}/yr first. Verdict: **{r.get('verdict')}** (null requires MDE80 <= "
         f"0.02/yr).\n", HDR,
         row("per-year slope, partial rho(S, Y | G) (primary)", r.get("slope_partial_S_given_G"), 4),
         row("(ii) per-year slope, OLS log points per SD of S", r.get("slope_ols_logpoints_per_SD"), 4),
         row("check: raw G coupling slope within band", r.get("check_slope_raw_G"), 4),
         row("raw S coupling slope within band", r.get("slope_raw_S"), 4),
         row("level: partial rho(S, Y | G) at YAG1", r.get("level_partial_y1")),
         row("level: partial rho(S, Y | G) at YAG5", r.get("level_partial_y5")), ""]
    bands = sorted(k[5:] for k in r if k.startswith("band_PA"))
    if bands:
        L += ["*Exploratory, not registered (revision 8): the per-year slopes band by band* (PA1 = top band, 4 As or "
              "more; PA5 = lowest band analysed). Each band's slope is averaged over cohorts within subject and then "
              "over the subjects that have the band; two-way intervals from their own seeds.\n", HDR]
        for b in bands:
            x = r.get("band_" + b, {})
            L.append(row(f"{b}: slope of partial rho(S, Y | G)", x.get("slope_partial"), 4,
                         note=f"subjects {x.get('subjects')}, cells {x.get('cells')}"))
            L.append(row(f"{b}: OLS slope, log points per SD of S", x.get("slope_ols"), 4))
        top = r.get("band_" + bands[0], {})
        L += ["", "The registered statistic averages the bands, so the registered \"rise\", and claim C5's \"does not "
              "decay\", describe the average over bands. They do not say that the premium rises in every band. The "
              f"top band ({bands[0]}), where graduates' own attainment is most uniform and where employer learning about "
              f"the school mean should matter most, has cells for {top.get('subjects')} subject(s) only "
              f"({', '.join(top.get('subject_names') or ['n/a'])}{sched_note(top.get('subject_names'))}); its two-way "
              "interval is undefined with one subject, so the institution-percentile interval is shown in the crossed "
              "column. The top band therefore says little about employer learning where own attainment is most "
              "uniform.\n"]
    return "\n".join(L)


def section_T5(r):
    L = ["## T5 — invariance of the field map to non-academic status indices (A1 A7)\n",
         f"scripts/28 Scorecard 4-yr cells; fields with >= 60 institutions having F, Y and all three indices: "
         f"{r.get('n_fields')}. 300 random halvings of each field's institutions; r* = r_obs / sqrt(rel_F rel_Z); "
         f"field bootstrap of the whole procedure (repeated fields get independent halvings; max multiplicity "
         f"{r.get('max_multiplicity')}). Map reliability of c_F across halves: {fmt(r.get('rel_F_map'))}.\n",
         "| index | MDE80 | r_obs | rel_Z | r* | 95% CI |", "|---|---|---|---|---|---|"]
    for z, lab in ZNAMES.items():
        L.append(f"| {lab} | {fmt(r.get(z + '_MDE80'), 3, False)} | {fmt(r.get(z + '_r_obs'))} | "
                 f"{fmt(r.get(z + '_rel_Z'))} | {fmt(r.get(z + '_r_star'))} | {fmt_ci(r.get(z + '_r_star_ci'))} |")
    L += ["", f"Verdict: **{r.get('verdict')}** (supported if every lower bound >= 0.70).\n"]
    return "\n".join(L)


def section_T6(r):
    L = ["## T6 — quantities where pay is set by schedule (A1 A7)\n",
         f"PSEO Flows V4.14.1, bachelor's, pooled cohorts, y5, census-division rows; T_i = sum_d share_d x log median "
         f"ACS 2023 wage of the field's scheduled occupation in division d (FTFY wage and salary workers aged 23–35 "
         f"with a bachelor's degree or higher). {r.get('note_cip')}. Institutions: education "
         f"{r.get('education_n_institutions')}, nursing {r.get('nursing_n_institutions')}.\n",
         f"MDE80 {fmt(r.get('MDE80'), 3, False)} first. Verdict: **{r.get('verdict')}**.\n",
         "| statistic | mean Spearman over the two fields | 95% CI (institution bootstrap) |", "|---|---|---|"]
    for lab, k in [("Spearman(G, T) (primary)", "primary"), ("Spearman(P, T), scripts/61 family prestige", "P_in_place_of_G"),
                   ("Spearman(G, share outside the scheduled industry)", "G_outside_scheduled_industry"),
                   ("Spearman(G, T), wages of workers whose highest degree is a bachelor's", "G_T_bachelors_only_wages")]:
        L.append(f"| {lab} | {fmt(r.get(k + '_mean_spearman'))} | {fmt_ci(r.get(k + '_ci'))} |")
    pf = {k: r.get(k + "_per_field") or {} for k in ["primary", "P_in_place_of_G", "G_outside_scheduled_industry",
                                                      "G_T_bachelors_only_wages"]}
    flds = list(pf["primary"])
    if flds:
        L += ["", "Per field (revision 7; same institution draws):\n",
              "| statistic | " + " | ".join(f"{f} (n)" for f in flds) + " |", "|---|" + "---|" * len(flds)]
        for lab, k in [("Spearman(G, T) (primary)", "primary"), ("Spearman(P, T)", "P_in_place_of_G"),
                       ("Spearman(G, share outside)", "G_outside_scheduled_industry"),
                       ("Spearman(G, T), bachelor's-only wages", "G_T_bachelors_only_wages")]:
            L.append(f"| {lab} | " + " | ".join(
                f"{fmt(pf[k].get(f, {}).get('est'))} {fmt_ci(pf[k].get(f, {}).get('ci'))} ({pf[k].get(f, {}).get('n')})"
                for f in flds) + " |")
    L += ["", f"ACS log median wage by division: education {json.dumps(r.get('education_acs_log_median_by_division'))}; "
          f"nursing {json.dumps(r.get('nursing_acs_log_median_by_division'))}. Distinct values among the nine "
          f"divisions: education {r.get('education_distinct_division_values')}, nursing "
          f"{r.get('nursing_distinct_division_values')} (ACS medians are often round numbers and tie), so T is a coarse "
          f"index. The wages are nominal: T is not adjusted for price differences between divisions, so a higher T "
          f"can mean higher-cost destinations as well as better-paid ones.\n",
          f"*Exploratory (not registered, no interval).* Most graduates work in or near their institution's state, so "
          f"T partly records where the institution is. T_home = the index if every graduate worked in the "
          f"institution's own division: {json.dumps(r.get('exploratory_home_division'))}. Mean partial Spearman(G, T | "
          f"T_home) = {fmt(r.get('exploratory_mean_partial_G_T_given_Thome'))}. The registered statistic does not by "
          f"itself separate status from institution location; this partial is the only check, and it is exploratory.\n"]
    return "\n".join(L)


def section_T7(r):
    L = ["## T7 — destination location (A1 A7)\n",
         f"scripts/52 fixed-cohort panel (PSEO V4.13.0, cohorts 2001–2010, y1/y5/y10) with PSEO Flows V4.13.0 division "
         f"shares of the same institution, CIP-2 family and cohort; W_h = sum_d share_d,h x L_d (ACS log median wage of "
         f"FTFY bachelor's-or-higher workers aged 23–35 by division of work); O_h = 1 - in-state share. Cells (i): "
         f"{r.get('i_cells')} ({r.get('i_fields')} fields); cells (ii)/(iii): {r.get('ii_cells')}.\n",
         f"MDE80 of R_L {fmt(r.get('i_MDE80_R_L'), 3, False)} first. Verdict: **{r.get('verdict')}**.\n", HDR,
         row("R_L = mean rho(F,Y|W) / mean rho(F,Y), y5", r.get("i_R_W")),
         row("mean rho(F, Y) at y5", r.get("i_raw")), row("mean rho(F, Y | W) at y5", r.get("i_pW")),
         row("mean rho(F, Y | W, O) at y5", r.get("i_pWO")), row("R_L with W and O", r.get("i_R_WO")),
         row("(ii) raw per-year slope", r.get("ii_raw"), 4), row("(ii) slope of rho(F, Y | W_h)", r.get("ii_pW"), 4),
         row("(ii) slope ratio (partial / raw)", r.get("ii_R_W")),
         row("(iii) slope of rho(F, Y | W_h, O_h)", r.get("ii_pWO"), 4),
         row("(iii) slope ratio (partial / raw)", r.get("ii_R_WO")), "",
         f"Share of the slope removed (no verdict): W {fmt(r.get('ii_share_removed_W'))}; W and O "
         f"{fmt(r.get('iii_share_removed_WO'))}. ACS L_d: {json.dumps(r.get('acs_L_d'))}.\n",
         "What the check covers (revision 6). W is a weighted average of nine census-division wage levels (all "
         "occupations, bachelor's or higher), so it nets only differences between divisions in where graduates work; "
         "differences within a division (metropolitan against non-metropolitan areas, one state against another) are "
         "not netted. The verdict therefore supports \"y5 coupling is unchanged net of the wage level of graduates' "
         "destination divisions\", not \"pay construct claims are free of destination location\"; and the career "
         "slope is not location-free either, since about one fifth (W) to one third (W and O) of it moves with "
         "destination-division wages and the out-of-state share.\n"]
    return "\n".join(L)


def section_T8(r):
    L = ["## T8 — between- and within-sector coupling (A1 A7; registered decomposition, no verdict)\n",
         f"scripts/61 W cells (institution x CIP-2 family, pooled cohorts, y5, PSEO V4.13.0): {r.get('W_cells')} cells, "
         f"{r.get('W_families')} families, {r.get('W_institutions')} institutions. C = log sum_k share_k x ACS national "
         f"median wage of FTFY bachelor's-or-higher workers aged 23–35 in sector k.\n", HDR,
         row("between-sector share B = 1 - within/raw", r.get("B_share")),
         row("raw coupling rho(P, log p50)", r.get("raw")), row("within-sector rho(P, log p50 | C)", r.get("within")),
         row("within, scripts/61's own C (ages 22–40)", r.get("within_C_scripts61_ages22_40")),
         row("B with scripts/61's own C", r.get("B_share_C_scripts61")), "",
         "B counts as employers' valuation only under demand rationing, which these data cannot show (A7 T8).\n"]
    return "\n".join(L)


DEVIATIONS_T = """## Deviations and clarifications (G4)

Every item below either resolves an ambiguity in the binding texts or departs from them. No estimate produced by a
deviation replaces a pre-specified estimate in a verdict.

1. **No public deposit before the run (A1 preamble).** The workflow that ran the tests may not commit; the tests were
   run without the time-stamped deposit A1 asks for. The stub and script digests are recorded under
   "Pre-registration status".
2. **Randomness (G2, A0.4).** SeedSequence(75) spawns B0's T1, T2, T3 in that order; SeedSequence(751) spawns T0,
   T1', T2', T3', T4, T5, T6, T7, T8. Each test spawns one child per named procedure (fixed order in the script) and
   each procedure one child per replicate (per field and replicate for per-field draws). Where B0 and A1 statistics
   share a design (T2, T3), each uses draws from its own seed.
3. **Two-way variance.** scripts/59 `twoway()` is called with V_field = the leave-one-field-out jackknife of the
   statistic (s^2/K for a mean), V_inst from one institution multinomial draw per replicate shared by all fields, and
   V_field x inst from an independent draw per field (scripts/59/66 conventions); replicates with an undefined
   statistic are redrawn from the same stream (counts reported). The "crossed" interval resamples fields together
   with the shared institution draw (scripts/66 `infer`). Ratios (R_DC, R_perp, R_L, B) use the same machinery.
4. **T0.** (a) x and q are defined for every ranked institution (0 and 0 when it has no public edge); p requires an
   IPEDS match by normalised Scorecard name (a matched institution without doctorates in the field has p = 0).
   Fields for (a): every field with >= 15 institutions having F and the variable; p and q only for the FIELDS66 fields
   (with the scripts/24 recovered fields, as scripts/28 uses them). (b) Fields: scripts/28's per-field rule (n >= 8);
   cells: scripts/28 cells whose institution is a node of the public field network and of the Academia network and
   has p. **Degree correction at zero degrees (clarification).** The registered offset
   o_uv = log(d_out_u d_in_v / (d_out_v d_in_u)) is undefined when a degree is zero, and in the public network
   {t0_unid} of nodes have a zero in- or out-degree (institutions that only hire or only place in the Total >= 2 edges).
   Taken literally (the exact limit), every pair touching such a node has an infinite offset and carries no
   information, and a node left with no finite-offset pair gets theta = 0 from the ridge (the middle of the
   ranking). Before any coupling statistic of T0 was computed, and after this outcome-blind identification check (on
   Computer Science and Mathematics only), the primary fit was set to add 1 to every degree, the convention A3 itself
   uses for the export ratio x. The exact limit is reported as two sensitivities with two-way intervals: (s0) every
   common cell, theta = 0 for unidentified nodes, R_DC {t0_exact}; (s1) identified cells only, R_DC {t0_ident} and
   R_perp {t0_identp}. The registered rule reads "{t0_v}" for the primary fit (R_DC {t0_rdc}), "{t0_s0}" for (s0)
   and "{t0_s1}" for (s1): {t0_dep}. (s0) is lower because it places the
   unidentified nodes in the middle. Every fit is solved by damped Newton with a backtracking line search and is
   checked for convergence (max |gradient| < 1e-6, asserted; revision 1: the first run's exact-limit fits used an
   undamped Newton that had not converged, and its exact-limit values, +0.36 and +0.71, and the statement that the
   verdict depends on the completion, are withdrawn). The ridge is (1e-3 / 2)||theta||^2. F_perp is the residual of
   rank(F) on rank(p) and rank(x) over the field's ranked institutions with p. (c) The local search starts once from the SpringRank order of the public network and 50
   times from random permutations; upward hires under F are counted on edges among ranked network nodes.
5. **T1.** Eligibility (>= 4 placed, >= 4 hired) counts all persons of the full public network (self-hires dropped),
   whether or not the counterpart is ranked; persons whose counterpart has no rank are dropped from P and H only.
   Split halves are drawn by row (Binomial(Total, 1/2)), which is the same as independent Bernoulli(1/2) draws per
   person. SpringRank is solved densely in batches with the equations of src/ar_pipeline/springrank (agreement
   reported). In T1(a') each half's percentiles are computed among the nodes with at least one person in that half.
   The benchmark networks cover the same field set as the real R'; in a simulated network a field that fails the
   eligibility or D rules is left out of that network's sums. The Academia network is analysed for B0 T1(a) and
   T1(b) only. **A4.1's calibration is ambiguous (clarification that decides C2; read this).** "c matching the
   field's total count" does not say whether the simulated total is matched before or after the A* >= 2 censoring.
   Read in sentence order (before), the simulated networks keep {t1_kept} of their persons, have {t1_elig} eligible
   institutions per field on average (the rule needs 15), and {t1_fail} of field x network cells fail B0's rules;
   R0 is still defined (from {t1_cells} cells) but rests on {t1_nf} field(s): R0 {t1_r0}, Delta_R {t1_dr}, rule
   "{t1_rule}". Read after the censoring (the public total is itself a censored total, so this is the reading under
   which simulated and public networks are comparable), c solves E[total after censoring] = public total (for
   Poisson(l), E[X 1{{X >= 2}}] = l(1 - exp(-l))); separate seeds (procedures "ap_sims_post", "ap_fieldboot_post"):
   Delta_R {t1_pdr}, rule "{t1_prule}". The post-censoring reading governs C2. It was adopted after the literal-order
   benchmark had been seen to degenerate and before any post-censoring statistic was computed, so it is not a clean
   pre-specified verdict; it is also the reading less favourable to the theory. The first run instead declared the
   literal-order benchmark "infeasible as registered" by a rule set after the run (more than half of the cells
   failing), a category A4.1 does not have, and left C2 undecided; that is withdrawn (revision 3).
6. **T2.** One record per person x bachelor's institution x field; the earliest qualifying doctorate is chosen by
   start year (missing last), then by episode id. The same-state share uses ROR's country subdivision of both
   institutions. The same-institution share (A5.5) counts persons whose only qualifying doctorates are at the
   bachelor's institution. K is computed as A5.2 defines it (rho(F,D|G) on the B0 cells minus rho(F,Y|G) on the cells
   with Y); the version with both partials on the Y cells is a reported sensitivity.
7. **T3.** Institutions without G are dropped with those without V. In T3', C enters as the standardised private
   dummy (so b_G is the G gradient at the sample's control mix), with G x C. The permutation MDE permutes whole
   institution trajectories (the three horizons together) within cells. No T3' cell has 10 or more private
   institutions (at most 5), so theta' within private institutions is undefined and the control-type proviso of A6.5
   cannot be checked; theta' within public institutions uses every field x cohort cell with >= 10 public
   institutions (including cells below the NMIN = 15 of the main sample). B0 secondary (ii) uses cells with >= 18
   institutions (scripts/66's rule for controlled cells). IPEDS completions use the Scorecard UNITID that scripts/61
   assigns to the OPEID.
8. **T4.** Selectivity S is scripts/62's provider-level share of graduates with >= 360 tariff points (as scripts/66
   uses it); cells keep providers with S and need >= 15 of them.
9. **T5.** Halvings are drawn within each field; in the field bootstrap a field drawn more than once uses independent
   halvings for each copy.
10. **T6.** Flows are released at CIP-2 only, so nursing is CIP-2 51 throughout. ACS "full-time full-year workers
    with a bachelor's degree" is implemented as scripts/61 does (wage and salary employees, WKHP >= 35, WKWN >= 50,
    WAGP > 0) with SCHL >= 21; the bachelor's-only version (SCHL = 21) is reported. The institution bootstrap uses one
    multinomial draw per replicate shared by both fields. Exploratory, not registered: the index each institution
    would have if all its graduates worked in its own census division (T_home), and G's association with T net of
    it, because T largely records where the institution is.
11. **T7.** The panel is scripts/52's own release (V4.13.0), so the Flows division and in-state counts come from the
    same release (V4.13.0 Flows); each field uses the CIP-2 family of its first CIP-4 code (scripts/28 CIP2).
12. **T8.** scripts/61's pooled y5 cells are rebuilt with its own functions; to save memory its flows reader is run
    with the cohort list reduced to the pooled cohort, which leaves the pooled cells unchanged.
"""

METHOD = """## Method

- Rank statistics are Spearman correlations (average ranks for ties); partial Spearman = Pearson correlation of rank
  residuals on [1, ranks of the controls] (scripts/55), computed under bootstrap weights with weighted ranks and
  weighted least squares (scripts/59 `wrank`/`wcorr`, scripts/66 `wstd`/`wls`, imported).
- Slopes over years since graduation use OLS weights on (1, 5, 10) (US) or (1, 3, 5) (UK).
- Verdict rules are applied exactly as registered; "edge" marks an interval endpoint within 2 Monte Carlo SE of 0.
- Existing scripts are imported, never edited: scripts/28 (build_table, load_generic), scripts/52 (fixed panel), scripts/55
  (partial-correlation convention), scripts/59 (twoway, weighted ranks, release plumbing), scripts/61 (Flows and
  Scorecard readers, family prestige, W cells, sector wage mix), scripts/62 (UK constants, cells_career),
  scripts/66 (sample A, weighted regression helpers).
"""

CAVEATS = """## Caveats

- Registered does not mean confirmatory in the usual sense: the theory was written after most results were seen, the
  tests share data with those results, and the registration is self-attested (no deposit before the run).
- Every statistic is an association across institutions; none identifies an effect of prestige on pay or placement.
- T2 is a consistency check with named rivals (advisor networks, subfield routing, geography, content, self-selection,
  same-scale bias); it does not separate information from content.
- T3's secondary out-of-state measure predicts the same sign under learning and national recruiting.
- The public Wapman edge list contains only institution pairs with at least one man and one woman (found while running
  T1; B0's outcome-blind check (1) recorded Total >= 2 but not this). The gender audit T1(b)/(b') is therefore
  nearly uninformative, and the A1 benchmark's A* >= 2 censoring only approximates the public release rule.
- T6–T8 use division- or sector-level wages from the ACS as fixed weights; they describe where graduates work, not what
  any graduate is paid. The wages are nominal (no regional price adjustment), and the ACS division medians tie, so
  T and W are coarse.
- T0's two branches do not isolate production cleanly (revision 4). The degree-corrected rank F_DC (+1 convention)
  is closer to the export ratio x than F itself (validity diagnostic in T0), so R_DC does not measure the part of F
  that is free of volume; F_perp removes x, but under H1 departments that are valued export on net, so R_perp also
  removes valuation and understates the retained share. "Partly production" is read as an upper bound on the production
  share, not as an estimate of it.
- T1(a')'s governing reading of A4.1 (c matched after the censoring) was adopted after the literal-order benchmark
  had been seen to degenerate; both readings are reported (revision 3).
- T4's registered statistic averages the prior-attainment bands; in the exploratory per-band slopes the rise sits in
  the middle bands, not in the top band (one subject, medicine and dentistry, with national pay scales) or the
  lowest (revision 8), so "does not decay" is a statement about the average.
- T6's registered "supported" is uneven across the two fields; the per-field estimates are reported (revision 7).
"""


# =============================================================================================================
# driver
# =============================================================================================================
RUNNERS = {"T0": run_T0, "T1": run_T1, "T2": run_T2, "T3": run_T3, "T4": run_T4, "T5": run_T5, "T6": run_T6,
           "T7": run_T7, "T8": run_T8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--compare", default="", help="with --report: a second cache directory whose per-test JSON is "
                    "compared byte for byte with --cache (re-run check, recorded in Provenance)")
    a = ap.parse_args()
    b0, a1 = prespec_ok(), amendment_ok()
    print(f"B0 digest matches: {b0}\nA1 digest matches: {a1}", flush=True)
    if not (b0 and a1):
        raise SystemExit("binding text changed: refusing to run")
    script_sha = sha256_file(Path(__file__))
    print(f"script SHA-256: {script_sha}", flush=True)
    cache = Path(a.cache) if a.cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    todo = [t.strip() for t in a.only.split(",") if t.strip()] if a.only else ([] if a.report else TESTS)
    for t in todo:
        fn = RUNNERS[t]
        stage(f"==== {t} ====")
        if t in ("T0", "T1"):
            fn(workers=a.workers)
        else:
            fn()
        RES[t]["_run"] = dict(script_sha256=script_sha, workers=a.workers if t in ("T0", "T1") else 1,
                              threads={k: os.environ.get(k, "") for k in
                                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
        if cache:
            (cache / f"{t}.json").write_text(json.dumps(RES[t], indent=1, sort_keys=True))
        gc.collect()
    if a.report or not a.only:
        res = {}
        for t in TESTS:
            if t in RES:
                res[t] = RES[t]
            elif cache and (cache / f"{t}.json").exists():
                res[t] = json.loads((cache / f"{t}.json").read_text())
        sp = ROOT / STUB_KEEP
        stub_check = ("present, SHA-256 matches" if sp.exists() and sha256_file(sp) == STUB_SHA256 else
                      ("present, SHA-256 DOES NOT MATCH" if sp.exists() else "MISSING"))
        cmp = {}
        if a.compare and cache:
            for t in TESTS:
                pa_, pb_ = cache / f"{t}.json", Path(a.compare) / f"{t}.json"
                cmp[t] = bool(pa_.exists() and pb_.exists() and pa_.read_bytes() == pb_.read_bytes())
        meta = dict(b0_ok=b0, a1_ok=a1, script_sha=script_sha, date=time.strftime("%Y-%m-%d"), stub_check=stub_check,
                    compare=cmp,
                    tests=[t for t in TESTS if t in res],
                    inputs={"wapman edge_lists.csv sha256": sha256_file(EDGES)[:16] + "…",
                            "IPEDS C2023_a.csv sha256": sha256_file(IPEDS)[:16] + "…",
                            "ACS PUMS": "psam_pusa.csv, psam_pusb.csv (2023 1-year)",
                            "PSEO": "V4.14.1 (data/raw/pseo_2026q2, pseo_flows_2026q2) and V4.13.0 (data/raw/pseo)",
                            "ORCID": "data/orcid/all/edge_aff (681 shards)", "LEO": "data/raw/leo/leo_dashboard.zip"})
        write_report(res, meta)


if __name__ == "__main__":
    main()
