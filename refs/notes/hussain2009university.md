# hussain2009university
**Full citation:** Hussain, Iftikhar, Sandra McNally, and Shqiponja Telhaj (2009). University Quality and Graduate Wages in the UK. IZA Discussion Paper No. 4043 (also CEE/CEP Discussion Paper 0099, LSE).
**DOI / URL:** https://docs.iza.org/dp4043.pdf
**Access level:** full text (read the IZA dp4043 PDF abstract and front matter)

## What it contributes to THIS project
This paper is a direct methodological precedent for our "ER as quality" measurement problem: it explicitly studies how to combine several noisy institution-quality measures into one aggregate index and then links that index to graduate earnings. That is the same move we make when we summarize Employer Reputation from multiple behavioral signals (Scorecard earnings, SDR salary/sector) and when we worry that any single prestige proxy is error-laden. Its central empirical result — a roughly 6% earnings differential per one-standard-deviation rise in university quality, but a strongly non-linear relationship with much higher returns at the top — warns us that an ER ranking will compress in the middle and spread at the top, which matters for how we rank-transform ER before computing Spearman against SpringRank. It is also a useful non-US (UK) external-validity check on the quality-to-wage gradient that underlies the ER side of the gap.

## Specific equation / result / dataset we reuse
We reuse two things: (1) the practice of constructing an aggregate university-quality index from multiple component quality measures and regressing log graduate wages on it (their finding: ~6% per SD, increasing returns at the top of the quality distribution, possibly rising over time); and (2) the explicit treatment of non-linearity, which justifies our choice to rank-transform ER (Spearman) rather than assume a linear prestige-earnings map. The aggregation-of-multiple-quality-proxies step is the concrete algorithmic idea we carry into building a composite ER from Scorecard + SDR columns.
