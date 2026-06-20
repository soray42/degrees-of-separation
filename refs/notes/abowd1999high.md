# abowd1999high
**Full citation:** Abowd, John M., Kramarz, Francis, & Margolis, David N. (1999). High Wage Workers and High Wage Firms. Econometrica, 67(2), 251-333.
**DOI / URL:** 10.1111/1468-0262.00020 (https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00020); NBER w4917 (https://www.nber.org/papers/w4917)
**Access level:** abstract only (read the NBER abstract and the AKM equation form as restated in accessible secondary sources, e.g. Borovickova-Shimer; the Econometrica full text is paywalled)

## What it contributes to THIS project
AKM is the canonical decomposition of wages into a *worker* component and a *firm/employer* component, and it is the workhorse that our ER construct conceptually mirrors: ER is a behavioral, employer-side wage signal, analogous to the AKM firm effect ψ, distinct from the worker's own ability/credential (the person effect). It is the empirical reference point for "does the employer side add value beyond the worker," which is exactly the academia-vs-industry question — does where you trained (prestige) line up with what employers pay (firm-side premium). We cite it both as the origin of two-sided fixed-effects wage models and as the method our later citations (Eeckhout-Kircher 2011; Borovickova-Shimer) critique for limited-mobility / incidental-parameter bias, which is why we build ER from observed earnings rankings rather than estimated AKM firm effects.

## Specific equation / result / dataset we reuse
The equation we reuse is the AKM log-earnings decomposition:
log w_it = θ_i + ψ_{J(i,t)} + x_it'β + ε_it,
where θ_i is the person/worker fixed effect, ψ_{J(i,t)} is the fixed effect of the firm employing worker i at time t, x_it'β is observed time-varying covariates, and ε_it is the residual. Their headline finding — person effects dominate wage variation (≈92% of inter-industry wage differentials) while firm effects are distinct — is the template for separating a worker/credential channel from an employer-premium channel, which underpins our conceptual split between Academic Reputation (prestige of training) and Employer Reputation (firm-side earnings signal).
