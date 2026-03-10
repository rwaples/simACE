# EPIMIGHT Heritability: Liability-Scale vs Observed-Scale

## Summary

EPIMIGHT estimates of h² are substantially lower than the "true" h² reported in `true_parameters.json`. This is **expected behavior**, not a bug. The two quantities measure different things:

| Quantity | Definition | Example (long_term, d1) |
|----------|-----------|------------------------|
| Liability-scale h² | Var(A) / Var(A+C+E) | 0.4998 |
| Observed-scale h² (EPIMIGHT) | Falconer's formula on binary affected status | 0.10 - 0.13 |

## Why the discrepancy

The liability-scale h² assumes a deterministic threshold: you're affected if and only if your liability exceeds a cutoff. Falconer's method recovers this h² when the threshold model holds.

ACE uses a **frailty/time-to-event model** instead:

```
P(event by age t | L) = 1 - exp(-(t/scale)^rho * exp(beta * L))
```

Even two individuals with identical liability can have different outcomes because the hazard process is stochastic. This acts like additional environmental noise that is not captured by the A/C/E decomposition, attenuating the heritability of the binary outcome.

## Verification

Using the `long_term` scenario (h²_liability = 0.50, K ≈ 0.10, all cohorts followed to age 80):

**Genetic correlations are correct** — PO correlation of A1 is 0.4996 (expected 0.5).

**But binary outcome correlations are attenuated:**

| Measure | PO pairs | Expected if threshold model |
|---------|----------|---------------------------|
| corr(L1_child, L1_parent) | 0.252 | 0.250 (= h²/2) |
| corr(affected1_child, affected1_parent) | 0.023 | would be higher under threshold model |
| K_r (children of affected parents) | 12.1% | 20.0% (Falconer prediction) |
| Falconer h² from direct PO pairs | 0.126 | 0.500 |

The Falconer h² computed directly from the simulation's own PO pairs (0.126) matches EPIMIGHT's PO estimate (0.103), confirming that EPIMIGHT is correctly estimating the observable quantity. The remaining gap (0.126 vs 0.103) likely reflects EPIMIGHT using CIF-based estimation rather than simple prevalence ratios.

## Implications

1. **The `true_parameters.json` values are not directly comparable to EPIMIGHT estimates.** The dashed "true" lines on the atlas plots overstate what EPIMIGHT can recover.

2. **The attenuation depends on the hazard model.** Weibull with low rho (< 1) produces a more gradual sigmoid relating liability to event probability, causing more attenuation. Higher rho produces a steeper relationship closer to a threshold.

3. **Higher prevalence reduces attenuation** — trait d2 (K ≈ 0.27) shows less attenuation than d1 (K ≈ 0.10), consistent with theory. Observed EPIMIGHT h² for d2 (0.16 - 0.23) is closer to the true 0.50 than d1 (0.10 - 0.13).

4. **To make the atlas "true" lines meaningful**, we would need to compute the expected Falconer h² under the frailty model — either analytically or by computing the theoretical CIF for the general population vs relatives-of-affected and applying Falconer's formula.

## References

- Falconer DS (1965). The inheritance of liability to certain diseases, estimated from the incidence among relatives. *Ann Hum Genet* 29:51-76.
- Dempster ER, Lerner IM (1950). Heritability of threshold characters. *Genetics* 35:212-236.
