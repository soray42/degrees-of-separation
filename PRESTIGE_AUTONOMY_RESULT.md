# How autonomous is academic prestige? (descriptive)

**Reframe:** autonomy index = **gap[f]** — a high gap means the academic prestige ordering is AUTONOMOUS / self-referential (it does not track market pay); a low gap means it is market-embedded. We ask what drives autonomy. **Honest scoping:** this largely OVERLAPS the earlier mixture decomposition (licensing is the one clean channel); the genuinely new pieces are the **basic↔applied dimension** and an **OpenAlex field-insularity** external test (feasibility below). Descriptive only. Run: `python scripts/31_prestige_autonomy.py`.

## Autonomy ranking

Most autonomous (high gap, self-referential) → most market-embedded (low gap):

| label                    |   n |   gap | ba_name      |
|:-------------------------|----:|------:|:-------------|
| Physiology               |  16 |  1.34 | basic        |
| Communication Disorders  |  17 |  1.21 | professional |
| Forestry                 |  17 |  1.15 | applied      |
| Music                    |  76 |  0.96 | basic        |
| Microbiology             |  17 |  0.93 | basic        |
| Agronomy                 |  11 |  0.93 | applied      |
| Nursing                  |  99 |  0.93 | professional |
| Agricultural Engineering |  14 |  0.91 | applied      |

…

| label                    |   n |   gap | ba_name      |
|:-------------------------|----:|------:|:-------------|
| Food Science             |  16 |  0.24 | applied      |
| Statistics               |  28 |  0.25 | basic        |
| Computer Science         | 161 |  0.29 | applied      |
| Computer Engineering     |  74 |  0.30 | applied      |
| Economics                | 116 |  0.31 | basic        |
| Political Science        | 139 |  0.34 | basic        |
| Biomedical Engineering   |  48 |  0.35 | applied      |
| Kinesiology/Exercise Sci |  70 |  0.38 | professional |

## Drivers of autonomy (Spearman with gap)

| driver | Spearman(gap, ·) | p | n | status |
|---|---|---|---|---|
| academic-absorption (reuse) | -0.00 | 0.983 | 47 | NULL (as previously found) |
| basic↔applied↔professional ordering (NEW) | -0.01 | 0.926 | 57 | does NOT support (see below) |
| licensing (reuse, cite +0.66 on gap) | +0.37 | 0.009 | 48 | the one clean channel |

### basic↔applied dimension (NEW) — gap by class

| ba_name      |   mean |   median |   count |
|:-------------|-------:|---------:|--------:|
| basic        |  0.605 |        1 |      23 |
| applied      |  0.577 |        0 |      17 |
| professional |  0.614 |        1 |      17 |

**The autonomy thesis (basic/autonomous fields have higher gaps) is NOT supported / confounded.** The ordering correlates -0.01 with the gap — but note the **professional** fields have the highest mean gap, which is the **licensing** channel (health/credential fields are licensed → compressed pay → high gap), not autonomy. Controlling for licensing, the basic↔applied ordering coefficient is **-0.023 (p=0.545)** while licensing is **+0.616 (p=0.009)** — the basic↔applied dimension adds little once licensing is netted (the apparent autonomy gradient is mostly the licensing of professional fields).

## STRONGEST test — OpenAlex field insularity (NOT run; key next step)

The one genuinely-new positive test the autonomy thesis could pass is an EXTERNAL self-reference measure independent of the channels: do more **insular / self-referential** fields (citations staying within-field; low cross-field citation diversity) have higher gaps? **Feasibility:** the repo has OpenAlex *field tags* per person (`orcid_field.parquet`, scripts 15–19) and an API key, but **no citation/insularity data** — that needs a new pull. Minimal viable: for each OpenAlex field, sample works and compute the within-field share of `referenced_works` fields (or topic-cross-field entropy) → one insularity score per field → crosswalk OpenAlex's 26 research fields to these degree fields (the same lossy research-vs-degree mapping flagged in scripts 15–19) → Spearman(gap, insularity). ~26 field queries; feasible but a real data pull. **Documented as the key next step, not run here.**

## Honest verdict

As scoped, autonomy ≈ **licensing + a large unexplained residual**: academic-absorption is null, and the basic↔applied dimension does not add beyond licensing. The value of this task is (i) the **autonomy reframing** of the gap, (ii) the **basic↔applied** test (reported honestly, including its confounding by licensing), and (iii) the **OpenAlex insularity** external validation, which is the one test that could genuinely confirm the autonomy thesis and is the key next step.

## Adversarial self-check

1. **Overlap with the mixture decomposition:** absorption and licensing are REUSED, not new; the only new computes are the basic↔applied ordering and the (deferred) insularity measure. Disclosed.
2. **basic↔applied coding is SUBJECTIVE** (hand-coded in the script; e.g. statistics=basic, CS=applied, music=basic are debatable). It is documented and reproducible; a few re-codings would shift the borderline fields but not the headline (professional fields' high gap is licensing, not autonomy).
3. **Does anything beat licensing?** No — once licensing is netted, the autonomy drivers add little; the autonomy reframe is conceptual, not a new explanatory channel.
4. **Insularity feasibility** is honestly flagged as not-run (needs a new OpenAlex citation pull); it is the genuinely-new test and is sized, not pre-judged.
5. **Descriptive only:** 'autonomy' names a measured property of the prestige ordering (it doesn't track pay), not a causal or normative claim.
