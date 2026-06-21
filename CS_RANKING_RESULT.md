# CS school ranking — academic prestige fine-tuned by graduate earnings (top 50)

**Descriptive, not a causal value claim.** Academic prestige = the CONTINUOUS SpringRank score of the US CS faculty-hiring network (Wapman et al.; recomputed from the public CS edge subgraph so score *differences* are meaningful). Earnings = College Scorecard CS bachelor's median (4yr-after). Blend (z-scored within the 70-school universe): **combined = w·z_prestige + (1−w)·z_salary**, primary **w = 0.6** (prestige-anchored, salary fine-tunes). Run: `python scripts/34_cs_ranking.py`.

Cross-checks: Scorecard↔**PSEO** CS earnings on the overlap (n=29) Spearman **+0.84**; Wapman↔**ORCID-rebuilt** CS SpringRank (n=61) Spearman **+0.69** — the prestige and the earnings axes both replicate on independent data.

## Top 50 (w = 0.6)

| # | school | CS median earnings | salary premium | rank@w0.5 | rank@w0.7 | low-robust |
|---|---|---|---|---|---|---|
| 1 | Carnegie Mellon University | $246,966 | +$48,910 | 1 | 3 |  |
| 2 | Stanford University | $214,907 | +$7,572 | 2 | 2 |  |
| 3 | University of California, Berkeley | $204,379 | −$6,379 | 4 | 1 |  |
| 4 | Massachusetts Institute of Technology | $225,141 | +$23,796 | 3 | 4 |  |
| 5 | University of California, Los Angeles | $216,722 | +$33,300 | 6 | 5 |  |
| 6 | University of Pennsylvania | $241,380 | +$71,295 | 5 | 7 |  |
| 7 | Cornell University | $201,227 | +$20,174 | 7 | 6 |  |
| 8 | Princeton University | $217,973 | +$48,071 | 8 | 8 |  |
| 9 | Harvard University | $203,169 | +$41,095 | 9 | 10 |  |
| 10 | Columbia University | $188,265 | +$24,016 | 10 | 12 |  |
| 11 | Johns Hopkins University | $196,467 | +$37,325 | 11 | 15 |  |
| 12 | University of Rochester | $155,464 | −$19,079 | 14 | 9 |  |
| 13 | University of California, Santa Barbara | $149,190 | −$26,147 | 15 | 11 |  |
| 14 | University of Illinois at Urbana-Champaign | $178,250 | +$15,888 | 13 | 16 |  |
| 15 | Brown University | $214,479 | +$66,137 | 12 | 17 |  |
| 16 | Georgia Institute of Technology | $150,628 | −$20,604 | 16 | 14 |  |
| 17 | University of Wisconsin - Madison | $119,655 | −$60,573 | 17 | 13 |  |
| 18 | Rice University | $182,443 | +$40,177 | 19 | 21 |  |
| 19 | University of Southern California | $192,897 | +$55,340 | 18 | 24 | ⚠️ |
| 20 | University of Texas at Austin, The | $144,185 | −$12,020 | 21 | 19 |  |
| 21 | Ohio State University, The | $118,833 | −$45,832 | 27 | 18 | ⚠️ |
| 22 | Washington University in St. Louis | $177,066 | +$37,783 | 20 | 27 | ⚠️ |
| 23 | University of California, San Diego | $159,487 | +$14,286 | 23 | 25 |  |
| 24 | George Washington University | $121,099 | −$38,887 | 28 | 20 | ⚠️ |
| 25 | Northwestern University | $158,473 | +$14,378 | 24 | 26 |  |
| 26 | University of Massachusetts Amherst | $123,519 | −$32,069 | 29 | 22 | ⚠️ |
| 27 | University of Chicago, The | $178,068 | +$45,739 | 25 | 31 | ⚠️ |
| 28 | University of Washington | $135,414 | −$13,175 | 30 | 28 |  |
| 29 | Dartmouth College | $201,702 | +$79,328 | 22 | 39 | ⚠️ |
| 30 | University of North Carolina at Chapel Hill | $129,461 | −$20,156 | 34 | 29 |  |
| 31 | University of Cincinnati, The | $99,245 | −$61,814 | 38 | 23 | ⚠️ |
| 32 | New York University | $142,495 | −$1,700 | 32 | 30 |  |
| 33 | Yale University | $188,157 | +$62,652 | 26 | 40 | ⚠️ |
| 34 | Virginia Polytechnic Institute and State University | $144,103 | +$4,841 | 35 | 33 |  |
| 35 | University of Michigan | $166,063 | +$35,390 | 31 | 41 | ⚠️ |
| 36 | Vanderbilt University | $160,021 | +$27,094 | 33 | 38 |  |
| 37 | University of Virginia | $142,041 | +$2,781 | 36 | 34 |  |
| 38 | University of Maryland, College Park | $127,995 | −$16,332 | 39 | 32 | ⚠️ |
| 39 | Tufts University | $156,343 | +$27,914 | 37 | 44 | ⚠️ |
| 40 | Missouri University of Science and Technology | $101,412 | −$46,840 | 40 | 36 |  |
| 41 | University of Missouri | $100,234 | −$48,018 | 41 | 37 |  |
| 42 | Pennsylvania State University, The | $96,042 | −$53,752 | 43 | 35 | ⚠️ |
| 43 | University of Pittsburgh | $96,501 | −$51,369 | 45 | 42 |  |
| 44 | Kent State University | $85,696 | −$62,556 | 49 | 43 | ⚠️ |
| 45 | University of California, Irvine | $122,507 | −$11,240 | 44 | 46 |  |
| 46 | University of California, Davis | $138,673 | +$12,864 | 42 | 48 | ⚠️ |
| 47 | University of California, Riverside | $117,705 | −$15,176 | 47 | 47 |  |
| 48 | Iowa State University | $92,709 | −$49,323 | 50 | 45 |  |
| 49 | Texas A&M University | $123,438 | −$3,921 | 48 | 50 |  |
| 50 | Colorado State University | $111,948 | −$18,622 | 51 | 49 |  |

**Salary premium** = a school's CS earnings minus what its academic standing predicts (regression earnings ~ prestige score; slope $68,449 per SpringRank unit). Positive = earns ABOVE its academic standing, negative = below. Reported for every school, both signs.

**Low-robustness (rank moves >5 across w=0.5→0.7): 15 schools** — University of Southern California, Ohio State University, The, Washington University in St. Louis, George Washington University, University of Massachusetts Amherst, University of Chicago, The, Dartmouth College, University of Cincinnati, The, Yale University, University of Michigan, University of Maryland, College Park, Tufts University, Pennsylvania State University, The, Kent State University, University of California, Davis. These are the salary-driven entries whose placement depends on how much weight you give earnings; read them with extra caution.

## Honest caveats (read before sharing)

- **(a) Not causal / not 'value-added'.** Earnings gaps among strong schools partly reflect **geography** (scripts/32: destination geography explains a modest within-field share here, R²≈0.13) and **student selection**, NOT proven school value-added. Chetty-Deming-Friedman: the causal effect of an elite school on AVERAGE earnings is small; the brand premium is in the elite TAIL that MEDIAN data cannot see. A high salary premium ≠ 'this school makes you richer'.
- **(b) Title-IV population** (Scorecard covers federally-aided graduates), but scripts/32 cross-validated Scorecard↔PSEO institution pay at Spearman +0.94 and here at +0.84 on the overlap, so the RANKING is robust to the earnings population.
- **(c) The median is WHERE GRADS LAND.** ~34% of BA grads work out-of-state; the median reflects where these graduates typically end up working, so a strong school whose grads stay in a lower-wage region is 'dinged' even though a mobile graduate may do fine. Read the salary term as *'where these grads typically land'*, NOT *'this school's ceiling'*. (This is why some Midwest/regional flagships sit below coastal schools of similar prestige.)
- **(d) Low-robustness schools are flagged** (⚠️ above); the prestige axis (w high) is the stable backbone, the salary axis (w low) is the fine-tuning that moves the flagged schools.
- **Coverage:** universe = the 70 top-CS-prestige PhD-granting departments with Scorecard CS earnings; a few CS schools are absent where Scorecard suppresses the CS cell or the name could not be matched. Descriptive ranking of those present, not an exhaustive list.
