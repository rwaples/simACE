# ADR 0024: Reject infeasible realised two-trait assortment

## Status

Accepted 2026-10-08; implemented 2026-10-08 in
`simace/simulation/simulate.py` (`_assortative_pair_partners`).

## Context

Two-trait assortment validates its configured mate-correlation matrix against
the configured within-person liability correlation. Pairing uses the realised
within-female and within-male correlations. These can differ after assortment
or through sampling.

The current implementation clips a negative conditional covariance eigenvalue
and continues. It also substitutes the configured correlation when realised
traits are constant or nearly collinear. Both behaviors can draw targets with
a different distribution from the real males. The current behavior is
described in [Methods](../concepts/methods.md#assortative-mating).

## Decision

1. Stop the replicate when the realised within-sex correlations make the
   requested mate-correlation matrix infeasible for the Gaussian conditional
   draw. Report the requested matrix, realised correlations, offending
   eigenvalue, and mating iteration.
2. Preserve the configured target matrix. Do not project it to a feasible
   matrix or substitute configured correlations for realised ones.
3. Reject constant or nearly collinear traits during two-trait pairing.
   Direct users to single-trait assortment. A matcher for feasible singular
   distributions is outside this fix.
4. Permit clipping only for negative eigenvalues within the numerical
   roundoff tolerance. Document that tolerance separately from mathematical
   feasibility.

The design interview chose stopping over adjusting the target and chose
rejecting degenerate two-trait mating over adding a lower-dimensional matcher.

## Consequences

Some configurations that currently finish with warnings will fail. The
replicate remains incomplete under the existing pipeline contract.

Configured feasibility remains an early check; realised feasibility is checked
for each mating population. This is a necessary covariance check, not a proof
that a finite one-to-one permutation can meet every target exactly. An
optimizer's failure to converge alone does not prove covariance infeasibility.

The covariance formula is the Gaussian conditional Schur complement, as
derived in [Marginal and conditional distributions of a multivariate normal
vector](https://www.statlect.com/probability-distributions/multivariate-normal-distribution-partitioning).
