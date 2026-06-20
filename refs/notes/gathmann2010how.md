# gathmann2010how
**Full citation:** Gathmann, C., & Schönberg, U. (2010). How General Is Human Capital? A Task-Based Approach. *Journal of Labor Economics*, 28(1), 1-49. (Working paper version: IZA Discussion Paper No. 3067, 2007, EconStor handle 10419/34436.)
**DOI / URL:** doi:10.1086/649786 ; working-paper PDF read in full: https://www.econstor.eu/bitstream/10419/34436/1/556772660.pdf
**Access level:** full text (IZA/EconStor working-paper version, identical methodology to the published JOLE article; published JOLE PDF itself paywalled but the formula, data, and task list were read verbatim from the open working paper)

## What it contributes to THIS project
This paper supplies the **core mechanism metric** of the entire project: the operational definition of *task distance* between two occupations. Our central hypothesis is that the academia-vs-industry **task distance** of a field predicts the AR-ER gap (CS = near-zero distance, near-zero gap; law/finance = high distance, high gap). Gathmann-Schoenberg give us the exact, citable, peer-reviewed object to compute that distance: the **angular separation (uncentered correlation) of two task-frequency vectors**, converted to a [0,1] distance. They establish the economic interpretation we lean on — that distance in task space measures the *non-transferability of human capital*, which is precisely the friction that should drive a wedge between faculty-hiring prestige (AR) and labor-market earnings (ER). Their finding that skills are far more portable than occupation labels suggest is the conceptual backbone for why an academic research "occupation" and the modal industry destination of a field's graduates can be near or far in task space. We import their formula directly and re-derive the task vectors from O\*NET instead of the German BIBB/IAB survey, keeping the math identical so the regressor is methodologically grounded.

## Specific equation / result / dataset we reuse

**The exact distance formula (the thing we lift).**
Each occupation `o` is described by a task vector `q_o = (q_{o1}, ..., q_{oJ})` where `q_{jo}` is the *fraction of workers in occupation o who perform task j*. The angular separation (uncentered correlation) between occupations `o` and `o'` is:

```
AngSep_{oo'} = ( Σ_{j=1..J} q_{jo} · q_{jo'} )  /  [ ( Σ_{j=1..J} q_{jo}^2 ) · ( Σ_{k=1..J} q_{ko'}^2 ) ]^{1/2}
```

i.e. the cosine of the angle between the two task vectors (denominator = product of the two Euclidean norms). Their **distance measure** is the modified version:

```
Dis_{oo'} = 1 − AngSep_{oo'}
```

It varies in [0, 1]: **0** for occupations using identical skill sets, **1** for completely different skill sets. Key normalization property they stress: unlike Euclidean distance, angular separation is **insensitive to vector length** — two occupations that use *all* tasks moderately count as similar, not distant; only the *direction* (relative task mix) matters. They do NOT renormalize the vectors to sum to 1; the normalization is entirely the cosine denominator (division by the two L2 norms). The borrowed-from-innovation-literature lineage is Jaffe (1986), used for technological proximity of firms.

**Task vector construction (the data recipe we mirror).**
- Source: repeated cross-section **German Qualification and Career Survey** (BIBB/IAB; waves 1979, 1985, 1991/92, 1998/99; ~30,000 employees aged 16-65 per wave; men only in their analysis).
- Respondents report whether they perform each of **J = 19 different tasks** (e.g. repairing, cleaning, buying/selling, teaching, planning), and whether each is their *main* activity.
- `q_{jo}` = share of workers in occupation `o` performing task `j`, aggregated to occupation level. (They also note the 19 tasks can be collapsed into 3 Autor-Levy-Murnane / Spitz-Öner aggregate groups — analytical, manual, interactive — but the distance metric is computed over the full 19-dim vector.)
- Reported descriptive moment we can sanity-check against: in their data the **mean inter-occupation distance is 0.24 (SD 0.22)**; most-similar move (paper/pulp processing vs printer/typesetter) ≈ **0.002**; most-distant move ≈ banker vs unskilled construction worker.

**How we ADAPT it for the AR-ER gap (our `task_distance` regressor).**
We keep the formula `Dis = 1 − AngSep` byte-for-byte and only swap the data layer:
1. **Task dictionary.** Replace the 19 German survey tasks with O\*NET task descriptors — specifically the O\*NET **Work Activities** (and/or **Skills**) element ratings, giving a vector of length J = (number of O\*NET WA/Skill elements, e.g. the 41 Generalized Work Activities or the 35 Skills). Each SOC occupation `o` gets a vector `q_o` whose entries are O\*NET importance/level ratings per element (the continuous analogue of "fraction of workers performing task j"; we standardize/scale ratings consistently across occupations before the cosine, matching their direction-only logic).
2. **Two occupations per field.** For each field (indexed by CIP — the project's join key) we build TWO task vectors:
   - an **academic-research task vector** `q_acad,field` — the task profile of the professor/researcher SOC for that field (e.g. postsecondary-teacher / research-scientist SOC codes mapped from the CIP), and
   - a **modal-industry-destination task vector** `q_ind,field` — the task profile of the *modal industry occupation* that the field's graduates flow into, identified via the **CIP→SOC crosswalk** (NCES/BLS) weighted by where graduates actually land (e.g. SDR sector/occupation distribution, Scorecard-linked occupations).
3. **Compute the regressor.** `task_distance_field = 1 − AngSep( q_acad,field , q_ind,field )` using the identical cosine formula above. This is the field-level academia-vs-industry task distance that we regress the `Gap_field = 1 − Spearman(prestige_rank, ER_rank)` outcome on. Prediction: CS has small `task_distance` (academic CS research tasks ≈ industry SWE tasks) and a small gap; credential fields (law, finance) have large `task_distance` and large gaps. The CIP→SOC crosswalk is exactly the bridge that lets us assemble both vectors from O\*NET on a common axis, the way Gathmann-Schoenberg assembled occupation vectors from a common 19-task axis.
